"""下游任务收益：用世界模型做潜空间 MPC（随机打靶 + CEM）。

这是 JD「将世界模型用于机器人行为想象、model predictive control、策略规划、
数据增强」的落地检验 —— 如果世界模型学得对，用它规划应该**显著优于随机动作**；
如果它只是"会重建画面但不懂动作因果"，规划成功率会和随机差不多。

流程（receding horizon）：
    1. 把当前真实观测编码成 z0；
    2. 采样 N 条候选动作序列（长度 H），在本体状态里按 MAX_STEP 累积执行器位置；
    3. 用世界模型在潜空间 rollout，取任务头预测的"目标距离 / 成功概率"打分；
    4. 把最优序列的**第一个动作**施加到真实环境；
    5. 回到 1（真正的闭环，不是开环播一遍）。

对照基线：
    - random      ：随机动作（下界）
    - scripted    ：带特权 GT 的脚本专家（上界参考，不是学习方法）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..env.physics import MAX_STEP
from ..env.world import PushWorld
from ..models.world_model import WorldModel
from ..data.collector import PushPlanner


@dataclass
class PlanConfig:
    n_samples: int = 96          # 每轮候选动作序列数
    horizon: int = 6             # 规划视界
    n_rounds: int = 2            # CEM 迭代轮数（1 = 纯随机打靶）
    elite_frac: float = 0.1
    action_std: float = 0.8
    seed: int = 0
    max_steps: int = 32          # 真实环境里最多走多少步


def _proprio_rollout(eff0: np.ndarray, actions: np.ndarray,
                     step0: int, horizon: int, env_horizon: int) -> np.ndarray:
    """按动作累积执行器位置，构造本体状态序列 [N,H,6]。

    注：这里**忽略桌边裁剪**，是刻意的近似 —— 规划器不应该读特权物理。
    偏差只发生在贴着桌边时，本项目里很少见。
    """
    N, H, _ = actions.shape
    p = np.zeros((N, H, 6), dtype=np.float32)
    eff = np.tile(eff0[None, :], (N, 1))
    prev = np.zeros((N, 2), dtype=np.float32)
    for t in range(H):
        eff = eff + actions[:, t] * MAX_STEP
        frac = min(1.0, (step0 + t + 1) / max(1, env_horizon))
        p[:, t, 0:2] = eff
        p[:, t, 2:4] = actions[:, t]
        p[:, t, 4] = frac
        p[:, t, 5] = 1.0 - frac
    return p


@torch.no_grad()
def plan_action(model: WorldModel, obs_x: torch.Tensor, eff0: np.ndarray,
                step0: int, lang: torch.Tensor, device: torch.device,
                cfg: PlanConfig, env_horizon: int, rng: np.random.Generator,
                mean: Optional[np.ndarray] = None,
                std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """返回 (要执行的动作[2], 调试信息)。"""
    model.eval()
    # encode 返回的 z 已经是 [B,Z]（B=1）；不要再切维度
    z0, _, _, _ = model.encode(obs_x.to(device), sample=False)
    z0 = z0[:, 0] if z0.dim() == 3 else z0
    assert z0.dim() == 2 and z0.shape[-1] == model.cfg.z_dim, \
        "z0 形状不对: %s" % (tuple(z0.shape),)
    N, H = cfg.n_samples, cfg.horizon

    if mean is None:
        cand = rng.uniform(-1.0, 1.0, size=(N, H, 2)).astype(np.float32)
    else:
        s = std if std is not None else cfg.action_std
        cand = np.clip(mean[None] + rng.normal(0, s, size=(N, H, 2)), -1, 1).astype(np.float32)
        cand[0] = np.clip(mean, -1, 1)

    a_t = torch.from_numpy(cand).to(device)
    p_t = torch.from_numpy(_proprio_rollout(eff0, cand, step0, H, env_horizon)).to(device)
    z_rep = z0.expand(N, -1).contiguous()
    lg = lang.expand(N, -1).contiguous()

    zs = model.rollout(z_rep, a_t, p_t, lg)            # [N,H+1,Z]
    _, _, task = model.heads(zs)                       # [N,H+1,4]
    dist = task[:, :, 0]
    succ = task[:, :, 1]

    # 打分：越接近目标越好；成功概率作为加分；越靠后的步权重越高
    w = torch.linspace(0.4, 1.0, H + 1, device=device)[None, :]
    score = (-(dist.clamp(min=0.0)) * w + 0.5 * succ * w).sum(dim=1)
    best = int(torch.argmax(score).item())
    info = {"plan_pred_dist": float(dist[best, -1]), "plan_pred_succ": float(succ[best, -1]),
            "plan_score": float(score[best])}
    return cand[best, 0].astype(np.float32), (info, cand, score.cpu().numpy())


def _cem(model, obs_x, eff0, step0, lang, device, cfg, env_horizon, rng):
    """CEM：先随机打靶，再围绕精英样本迭代收紧分布。"""
    mean = np.zeros(2 * cfg.horizon, dtype=np.float32)
    std = np.full(2 * cfg.horizon, cfg.action_std, dtype=np.float32)
    cand_top, info = None, {}
    for r in range(max(1, cfg.n_rounds)):
        m2 = mean.reshape(cfg.horizon, 2)
        s2 = std.reshape(cfg.horizon, 2)
        a0, (info, cand, scores) = plan_action(model, obs_x, eff0, step0, lang, device,
                                               cfg, env_horizon, rng, m2, s2)
        k = max(2, int(cfg.n_samples * cfg.elite_frac))
        idx = np.argsort(-scores)[:k]
        elite = cand[idx]
        mean = elite.reshape(k, -1).mean(axis=0).astype(np.float32)
        std = (elite.reshape(k, -1).std(axis=0) + 1e-3).astype(np.float32)
        cand_top = a0
    return cand_top, info


# ---------------------------------------------------------------- 闭环评估


def run_mpc_episode(model: WorldModel, world: PushWorld, seed: int,
                    device: torch.device, cfg: PlanConfig,
                    policy: str = "mpc") -> Dict[str, float]:
    """在真实环境里闭环跑一条，返回结果。"""
    from ..data.dataset import DEPTH_SCALE, sem_to_onehot

    rng = np.random.default_rng(cfg.seed + seed)
    obs = world.reset(seed=seed)
    planner = PushPlanner()
    steps, info = 0, {}
    while steps < cfg.max_steps:
        if policy == "random":
            a = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        elif policy == "scripted":
            a = planner.act(world)
        else:
            rgb = torch.from_numpy(
                np.clip(obs.rgb, 0, 1).transpose(2, 0, 1)[None]).float()
            dep = torch.from_numpy(
                np.clip(obs.depth_masked / DEPTH_SCALE, 0, 1)[None, None]).float()
            sem = torch.from_numpy(sem_to_onehot(obs.semantic[None])).float()
            x = torch.cat([rgb, dep, sem], dim=1).to(device)
            lang = model.lang_vec(torch.from_numpy(obs.instruction[None]).to(device))
            a, info = _cem(model, x, world.physics.effector, steps, lang, device,
                           cfg, world.horizon, rng)
        obs, reward, done, gt = world.step(a)
        steps += 1
        if done:
            break
    t = world.physics.task_info()
    return {"success": float(t["success"]), "steps": float(steps),
            "final_dist": float(t["target_dist"]),
            "pred_dist": float(info.get("plan_pred_dist", np.nan)),
            "dropped": float(t["n_dropped"])}


def run_mpc(model: WorldModel, seeds: Sequence[int], device: torch.device,
            cfg: PlanConfig, policy: str = "mpc", world: Optional[PushWorld] = None,
            verbose: bool = True) -> Dict[str, float]:
    world = world or PushWorld(size=48, horizon=cfg.max_steps)
    res = [run_mpc_episode(model, world, s, device, cfg, policy) for s in seeds]
    succ = float(np.mean([r["success"] for r in res]))
    out = {"policy": policy, "n": len(res), "success_rate": round(succ, 4),
           "mean_steps": round(float(np.mean([r["steps"] for r in res])), 2),
           "mean_final_dist": round(float(np.mean([r["final_dist"] for r in res])), 4)}
    if policy == "mpc":
        out["mean_pred_dist"] = round(float(np.nanmean([r["pred_dist"] for r in res])), 4)
    if verbose:
        print("[mpc] %-9s 成功 %2d/%2d = %.1f%%  末距 %.3f  平均步数 %.1f"
              % (policy, int(round(succ * len(res))), len(res), succ * 100,
                 out["mean_final_dist"], out["mean_steps"]), flush=True)
    return out
