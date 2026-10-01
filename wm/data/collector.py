"""轨迹采集：脚本专家 / 随机 / 对抗三类策略 + 反事实配对分支。

三类数据（对应 JD「成功、失败、扰动和反事实数据」）：
    1. **expert**  —— 脚本专家：先绕到方块背后站位，再朝目标区推。带特权 GT，是数据生成器而不是学出来的策略。
    2. **random**  —— 随机推动，绝大多数失败，提供低回报与"乱动"分布。
    3. **perturb** —— 专家轨迹 + 周期性动作扰动（模拟操作噪声、打滑、误触）。

**反事实配对（counterfactual pairs）** 是本项目最关键的机制之一：
    在同一状态快照下分叉成两条支路，前 k 步动作完全相同，第 k+1 步起一条走原动作、
    另一条走扰动动作。于是两条支路的**观测前缀逐像素一致**，而未来的差异**完全由动作差异
    造成** —— 这正是"动作可控性"评测需要的因果对照数据，也是 JD 里"反事实数据"的落地。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..env.physics import EFFECTOR_R, MAX_STEP
from ..env.world import Observation, PushWorld


# ---------------------------------------------------------------- 策略


def _clip(a: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0).astype(np.float32)


def _segment_hits_aabb(p0: np.ndarray, p1: np.ndarray,
                       cx: float, cy: float, hx: float, hy: float,
                       margin: float) -> bool:
    """线段是否穿过膨胀后的 AABB（slab method）。用于判断要不要绕行。"""
    d = p1 - p0
    lo = np.array([cx - hx - margin, cy - hy - margin])
    hi = np.array([cx + hx + margin, cy + hy + margin])
    tmin, tmax = 0.0, 1.0
    for i in range(2):
        if abs(d[i]) < 1e-12:
            if p0[i] < lo[i] or p0[i] > hi[i]:
                return False
            continue
        t1 = (lo[i] - p0[i]) / d[i]
        t2 = (hi[i] - p0[i]) / d[i]
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return False
    return True


class PushPlanner:
    """脚本专家（有状态）。

    四个必要的设计点（每一个都是踩坑踩出来的，写在这里免得后人重踩）：

      1) **无超调控制**：动作按"剩余距离 / 单步最大位移"精确给，不做比例增益 ——
         否则会冲过站位点，把方块撞向反方向。
      2) **绕行**：从起点直着去站位点会**从方块身上碾过去**，把方块推向目标区反侧。
         所以先判断线段是否穿过方块膨胀包围盒，穿过就先走到侧向绕行点。
      3) **路线要在规划时一次定死，不能在每帧重判**：本项目真实踩到过 —— 线段与
         膨胀包围盒的相交测试在临界位置会**逐帧翻转**（相差 1e-3 就翻），于是执行器
         在两个格子之间形成**两帧极限环**，跑满整个 horizon 一步没推进。
         修法：规划时算一次 `use_detour`，之后只沿 `route` 顺序推进；只有方块移动
         超过 `replan_tol` 才重新规划。
      4) **推段要锁定**：一旦贴上接触距离就进入 `pushing` 状态，此后只沿 u 推；
         只有侧向脱手（偏离轴线太多）才退回重新站位。否则推的过程中路线会被反复重算。
    """

    def __init__(self, replan_tol: float = 0.05):
        self.side: Optional[float] = None
        self.plan_box: Optional[np.ndarray] = None
        self.replan_tol = replan_tol
        self.route: Optional[List[np.ndarray]] = None
        self.wp = 0
        self.pushing = False
        self.n_replans = 0

    # -------------------------------------------------- 规划

    def _plan(self, world: PushWorld, u: np.ndarray, perp: np.ndarray,
              stage: np.ndarray, back: np.ndarray, margin: float) -> None:
        tb = world.spec.target
        eff = world.physics.effector
        base = np.array([tb.cx, tb.cy])
        # 侧向只在这里定一次：选离当前位置更近、且不越界的那一侧
        best, best_cost = 1.0, np.inf
        for s in (1.0, -1.0):
            cand = np.clip(back + perp * s * (margin * 2.1), [0.045, 0.045], [0.955, 0.955])
            cost = float(np.hypot(*(cand - eff)))
            if cost < best_cost:
                best, best_cost = s, cost
        self.side = best

        detour = np.clip(back + perp * self.side * (margin * 2.1),
                         [0.045, 0.045], [0.955, 0.955])
        # 已经在方块"背后"（相对目标区那一侧）且横向没偏太多 -> 直接进站位点，
        # 不要再绕行/退让 —— 否则每次脱手重新站位都要多花十几个时间步，
        # 实测会把整个 horizon 耗在赶路上（成功率从个位数掉到 0）。
        behind = float(np.dot(eff - base, u)) < -0.015
        lateral = abs(float(np.dot(eff - base, perp)))
        if behind and lateral < margin * 1.6:
            route = [stage]
        elif not _segment_hits_aabb(eff, back, tb.cx, tb.cy, tb.hx, tb.hy,
                                    margin * 0.85):
            route = [back, stage]
        else:
            route = [detour, back, stage]
        self.route = route
        self.wp = 0
        self.plan_box = np.array([tb.cx, tb.cy])
        self.n_replans += 1

    # -------------------------------------------------- 动作

    def act(self, world: PushWorld) -> np.ndarray:
        tb = world.spec.target
        eff = world.physics.effector
        base = np.array([tb.cx, tb.cy])
        d = np.array([world.spec.goal[0] - tb.cx, world.spec.goal[1] - tb.cy])
        n = float(np.hypot(*d))
        if n < 1e-6:
            return np.zeros(2, dtype=np.float32)
        u = d / n
        perp = np.array([-u[1], u[0]])
        standoff = EFFECTOR_R + max(tb.hx, tb.hy) + 0.030
        margin = max(tb.hx, tb.hy) + EFFECTOR_R
        stage = base - u * standoff
        back = base - u * (standoff + 0.050)

        def goto(target: np.ndarray) -> np.ndarray:
            delta = target - eff
            if float(np.hypot(*delta)) < 1e-9:
                return np.zeros(2, dtype=np.float32)
            return _clip(delta / MAX_STEP)

        # --- 已达成 -> 停手（否则会把方块一路推过头，任务反而失败）
        if world.physics.goal_contains(tb):
            self.pushing = False
            self.route = None
            return np.zeros(2, dtype=np.float32)

        # --- 推段（锁定）
        if self.pushing:
            lateral = abs(float(np.dot(eff - base, perp)))
            radial = float(np.hypot(*(eff - base)))
            if lateral < margin * 0.95 and radial < standoff * 2.4:
                # 接近目标区时按剩余距离比例减速，避免冲过头
                scale = float(np.clip(n / 0.10, 0.25, 1.0))
                return _clip(u * scale)
            self.pushing = False           # 侧向脱手 -> 重新站位

        # --- 进入推段
        if float(np.hypot(*(eff - stage))) <= MAX_STEP * 1.02:
            self.pushing = True
            return _clip(u)

        # --- 站位行军
        if (self.route is None or self.plan_box is None
                or float(np.hypot(*(base - self.plan_box))) > self.replan_tol):
            self._plan(world, u, perp, stage, back, margin)

        assert self.route is not None
        while (self.wp < len(self.route) - 1
               and float(np.hypot(*(eff - self.route[self.wp]))) <= MAX_STEP * 1.02):
            self.wp += 1
        return goto(self.route[self.wp])


def expert_policy(world: PushWorld, obs: Observation,
                  planner: Optional["PushPlanner"] = None) -> np.ndarray:
    """无状态入口（每次新建规划器，仅用于单帧调试）。批量采集请用 `PushPlanner`。"""
    return (planner or PushPlanner()).act(world)


def random_policy(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def perturb_action(action: np.ndarray, rng: np.random.Generator,
                   sigma: float = 0.35) -> np.ndarray:
    return _clip(np.asarray(action, dtype=np.float64)
                 + rng.normal(0.0, sigma, size=2))


# ---------------------------------------------------------------- 单条轨迹


@dataclass
class Episode:
    rgb: np.ndarray          # [T, H, W, 3] uint8
    depth: np.ndarray        # [T, H, W]    float16
    semantic: np.ndarray     # [T, H, W]    uint8
    proprio: np.ndarray      # [T, 6]       float32
    action: np.ndarray       # [T, 2]       float32
    box_matrix: np.ndarray   # [T, N, 9]    float32
    events: np.ndarray       # [T, 2N+1]    float32
    task: np.ndarray         # [T, 4]       float32
    reward: np.ndarray       # [T]          float32
    visibility: np.ndarray   # [T, N]       float32
    instruction: np.ndarray  # [L]          int64
    meta: Dict

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "rgb": self.rgb, "depth": self.depth, "semantic": self.semantic,
            "proprio": self.proprio, "action": self.action,
            "box_matrix": self.box_matrix, "events": self.events,
            "task": self.task, "reward": self.reward,
            "visibility": self.visibility, "instruction": self.instruction,
        }


def _pack(frames: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k in frames[0]:
        out[k] = np.stack([f[k] for f in frames], axis=0)
    return out


def collect_episode(world: PushWorld, seed: int, policy: str = "expert",
                    horizon: Optional[int] = None, perturb_sigma: float = 0.0,
                    perturb_period: int = 0, hard: bool = False,
                    rng: Optional[np.random.Generator] = None) -> Episode:
    """采一条完整轨迹。"""
    rng = rng or np.random.default_rng(seed + 10_000)
    horizon = horizon or world.horizon
    obs = world.reset(seed=seed, hard=hard)
    frame0 = _frame(obs, world)
    # 初始帧也要补齐后续帧的全部键，否则 _pack 会按 frame0 的键集裁剪
    n = world.n_boxes
    ti = world.physics.task_info()
    frame0.update({
        "action": np.zeros(2, dtype=np.float32),
        "box_matrix": world.physics.box_matrix(),
        "events": np.zeros(2 * n + 1, dtype=np.float32),
        "task": np.asarray([ti["target_dist"], ti["success"], ti["terminated"],
                            ti["target_alive"]], dtype=np.float32),
        "reward": np.float32(0.0),
        "visibility": _vis_vector(obs, n),
    })
    frames = [frame0]
    meta = {"seed": int(seed), "policy": policy, "hard": bool(hard),
            "instruction": obs.instruction_text, "layout_id": int(world.spec.layout_id)}
    planner = PushPlanner()

    for t in range(horizon):
        if policy == "expert":
            a = planner.act(world)
        elif policy == "random":
            a = random_policy(rng)
        else:
            raise ValueError("未知策略: %s" % policy)
        if perturb_period > 0 and (t % perturb_period == perturb_period - 1):
            a = perturb_action(a, rng, perturb_sigma)
        elif perturb_sigma > 0 and policy != "random":
            a = perturb_action(a, rng, perturb_sigma)

        obs, reward, done, gt = world.step(a)
        fr = _frame(obs, world)
        fr["action"] = np.asarray(a, dtype=np.float32)
        fr["box_matrix"] = gt["box_matrix"]
        fr["events"] = gt["events"]
        fr["task"] = np.asarray([gt["task"]["target_dist"], gt["task"]["success"],
                                 gt["task"]["terminated"], gt["task"]["target_alive"]],
                                dtype=np.float32)
        fr["reward"] = np.float32(reward)
        fr["visibility"] = _vis_vector(obs, world.n_boxes)
        frames.append(fr)
        if done:
            break

    packed = _pack(frames)
    meta["n_steps"] = int(packed["action"].shape[0])
    meta["success"] = float(packed["task"][-1, 1])
    meta["final_dist"] = float(packed["task"][-1, 0])
    meta["n_dropped"] = int((packed["box_matrix"][:, :, 8] < 0.5).sum())
    return Episode(
        rgb=packed["rgb"], depth=packed["depth"], semantic=packed["semantic"],
        proprio=packed["proprio"], action=packed["action"],
        box_matrix=packed["box_matrix"], events=packed["events"],
        task=packed["task"], reward=packed["reward"],
        visibility=packed["visibility"], instruction=frames[0]["instruction"],
        meta=meta,
    )


def _frame(obs: Observation, world: PushWorld) -> Dict[str, np.ndarray]:
    return {
        "rgb": (np.clip(obs.rgb, 0, 1) * 255).astype(np.uint8),
        "depth": obs.depth_masked.astype(np.float16),
        "semantic": obs.semantic.astype(np.uint8),
        "proprio": obs.proprio.astype(np.float32),
        "instruction": obs.instruction.astype(np.int64),
    }


def _vis_vector(obs: Observation, n_boxes: int) -> np.ndarray:
    v = np.zeros(n_boxes, dtype=np.float32)
    for i in range(1, n_boxes + 1):
        v[i - 1] = float(obs.visibility.get(i, 1.0))
    return v


# ---------------------------------------------------------------- 反事实配对


@dataclass
class CounterfactualPair:
    """同一前缀、不同后缀的两个分支。

    ★ `rgb0` 是**分叉起点那一帧的观测**，两条支路共用同一份。
      早期实现漏存了它，评测时误用"分叉后第 1 帧"当起点 —— 而那一帧已经受动作影响，
      两条支路本来就该不一样，于是"前缀一致性"检查会在随机位置误杀配对，
      z0 也不再是同一个状态。**分叉起点必须显式存下来**。
    """

    rgb0: np.ndarray                # [H, W, 3] 分叉起点观测（两支路共用）
    prefix_actions: np.ndarray      # [k, 2]
    action_a: np.ndarray            # [m, 2]  原动作后缀
    action_b: np.ndarray            # [m, 2]  扰动后缀
    rgb_a: np.ndarray               # [m, H, W, 3]  施加动作后的帧
    rgb_b: np.ndarray
    box_a: np.ndarray               # [m, N, 9]
    box_b: np.ndarray
    branch_step: int

    def prefix_identical(self) -> bool:
        """两支路共用同一个起点帧 —— 反事实对照可信的前提。"""
        return self.rgb0 is not None


def collect_counterfactual(world: PushWorld, seed: int, branch_step: int = 8,
                           suffix: int = 8, sigma: float = 0.6) -> CounterfactualPair:
    """采一对反事实分支：跑 k 步专家动作，快照，然后分别用原动作 / 扰动动作续跑。

    ★ 两个分支**都严格跑满 `suffix` 步**，即使中途 `done` 也不提前 break。
      原因：反事实对照要求两条支路的数组形状严格一致（要 `np.stack` 成批），
      而且"跑同样步数、只有动作不同"才是干净的对照。提前终止会让每对的长度
      都不一样，直接导致 `all input arrays must have the same shape`
      —— 本项目在数据采集脚本里真实踩到过。
    """
    rng = np.random.default_rng(seed + 777)
    obs = world.reset(seed=seed)
    planner = PushPlanner()
    pre_actions = []
    for _ in range(branch_step):
        a = planner.act(world)
        pre_actions.append(a)
        obs, _, _, _ = world.step(a)

    snap = world.snapshot()
    base_obs = world.restore(snap)
    rgb0 = _frame(base_obs, world)["rgb"]        # ★ 分叉起点帧，两条支路共用

    def rollout(perturb: bool):
        acts, frames = [], []
        pl = PushPlanner()
        for i in range(suffix):
            a = pl.act(world)
            if perturb:
                a = perturb_action(a, rng, sigma)
            acts.append(a)
            obs2, _, _, gt = world.step(a)
            fr = _frame(obs2, world)
            fr["box_matrix"] = gt["box_matrix"]
            frames.append(fr)
        return (np.asarray(acts, dtype=np.float32),
                np.stack([f["rgb"] for f in frames], axis=0),
                np.stack([f["box_matrix"] for f in frames], axis=0))

    a_acts, a_rgb, a_box = rollout(perturb=False)
    world.restore(snap)
    b_acts, b_rgb, b_box = rollout(perturb=True)
    world.restore(snap)

    return CounterfactualPair(
        rgb0=rgb0,
        prefix_actions=np.asarray(pre_actions, dtype=np.float32),
        action_a=a_acts, action_b=b_acts,
        rgb_a=a_rgb, rgb_b=b_rgb,
        box_a=a_box, box_b=b_box,
        branch_step=branch_step,
    )


# ---------------------------------------------------------------- 落盘

SHARD_KEYS = ("rgb", "depth", "semantic", "proprio", "action",
              "box_matrix", "events", "task", "reward", "visibility")


def save_dataset(episodes: List[Episode], out_dir: str, name: str,
                 max_steps_per_shard: int = 2600) -> List[str]:
    """按步数分片保存为 npz，附 manifest.json。

    ★ 分片格式（踩过一次坑，务必按这个来）：
        每个数组按**步**在 axis=0 上 `concatenate`，配合 `ep_len` 记录每回合步数，
        读取端用累计偏移切窗口。
        不能把"每回合一个数组"的 list 直接丢给 `np.savez` —— 各回合步数不同，
        numpy 会尝试堆成齐整数组并抛
        `ValueError: setting an array element with a sequence`。
      `instruction` 是**每回合一条**（不随步变化），单独堆成 [n_ep, L] 存，
        读取端按回合 id 取，不能按步偏移取。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards, cur, cur_instr, cur_len, cur_steps = [], {}, [], [], 0
    manifest = {"name": name, "shards": [], "n_episodes": 0, "n_steps": 0,
                "policy_counts": {}, "success_rate": 0.0}

    def flush(idx):
        if not cur:
            return
        p = out / ("%s_shard%02d.npz" % (name, idx))
        payload = {k: np.concatenate(v, axis=0) for k, v in cur.items()}
        payload["instruction"] = np.stack(cur_instr, axis=0)
        payload["ep_len"] = np.asarray(cur_len, dtype=np.int32)
        np.savez(p, **payload)
        manifest["shards"].append({"file": p.name, "steps": int(cur_steps),
                                   "episodes": int(len(cur_len))})

    idx = 0
    n_succ = 0
    for ep in episodes:
        d = ep.to_dict()
        T = d["rgb"].shape[0]
        if cur and cur_steps + T > max_steps_per_shard:
            flush(idx)
            idx += 1
            cur, cur_instr, cur_len, cur_steps = {}, [], [], 0
        for k in SHARD_KEYS:
            cur.setdefault(k, []).append(d[k])
        cur_instr.append(np.asarray(d["instruction"]))
        cur_len.append(T)
        cur_steps += T
        manifest["n_episodes"] += 1
        manifest["n_steps"] += int(T)
        pol = ep.meta["policy"]
        manifest["policy_counts"][pol] = manifest["policy_counts"].get(pol, 0) + 1
        n_succ += int(ep.meta["success"] > 0)
    flush(idx)

    manifest["success_rate"] = round(n_succ / max(1, len(episodes)), 4)
    (out / ("%s_manifest.json" % name)).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return [s["file"] for s in manifest["shards"]]


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="采集世界模型数据集")
    ap.add_argument("--out", default="data")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--expert", type=int, default=90)
    ap.add_argument("--random", type=int, default=40)
    ap.add_argument("--perturb", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--tag", default="train")
    args = ap.parse_args(argv)

    world = PushWorld(size=args.size, horizon=args.horizon)
    eps: List[Episode] = []
    for i in range(args.expert):
        eps.append(collect_episode(world, args.seed0 + i, "expert"))
    for i in range(args.random):
        eps.append(collect_episode(world, args.seed0 + 5000 + i, "random"))
    for i in range(args.perturb):
        eps.append(collect_episode(world, args.seed0 + 9000 + i, "expert",
                                   perturb_sigma=0.45, perturb_period=4))
    files = save_dataset(eps, args.out, args.tag)
    succ = {p: sum(e.meta["success"] for e in eps if e.meta["policy"] == p)
            for p in ("expert", "random")}
    print("写出 %d 个分片: %s" % (len(files), files))
    print("专家成功 %d/%d   随机成功 %d/%d   总步数 %d"
          % (succ["expert"], args.expert, succ["random"], args.random,
             sum(e.meta["n_steps"] for e in eps)))


if __name__ == "__main__":
    main()
