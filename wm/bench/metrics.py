"""世界模型评测指标 —— 每一项都对应 JD 里点名的一个评测维度。

    JD 原话                                            本模块实现
    ─────────────────────────────────────────────────  ────────────────────────────────
    「评估视觉质量之外的动作可控性」                      action_controllability（反事实配对）
    「动力学准确性」                                      dynamics_accuracy（方块位置 ADE/FDE）
    「空间一致性」                                        spatial_consistency（深度 MAE + 占据 IoU）
    「长期预测」                                          long_horizon（误差-步长曲线）
    「物体持久性」                                        object_permanence（按遮挡分段统计）
    「未见场景和未见动作组合上的泛化能力」                 generalization（val / hard 留出布局）
    「分析模型在遮挡、碰撞、物体状态变化上的失败模式」      failure_modes（按接触/遮挡分桶）
    「下游任务收益」                                      bench.planning（潜空间 MPC 成功率）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data.dataset import DEPTH_SCALE, TrajectoryDataset, sem_to_onehot
from ..models.world_model import WorldModel


# ---------------------------------------------------------------- rollout


@torch.no_grad()
def rollout_episode(model: WorldModel, ds: TrajectoryDataset, eid: int,
                    horizon: int, device: torch.device) -> Dict[str, np.ndarray]:
    """对一条轨迹做潜空间自回归 rollout，返回预测与真值对照。"""
    model.eval()
    b = ds.episode_batch(eid, device=str(device))
    T = min(horizon, b["action"].shape[0] - 1)
    if T < 2:
        return {}
    x0 = b["x"][0:1]                        # [T,C,H,W] -> 取第 0 帧 = [1,C,H,W]
    assert x0.shape[0] == 1 and x0.shape[1] == model.cfg.in_ch, \
        "起点帧形状不对: %s" % (tuple(x0.shape),)
    z0, _, _, _ = model.encode(x0, sample=False)   # [1,Z]
    lang = model.lang_vec(b["instr"][None])
    zs = model.rollout(z0, b["action"][None, :T], b["proprio"][None, :T], lang)[0]
    st, ev, tk = model.heads(zs)
    rgb, dep, sem = model.decode(zs)
    return {
        "z_pred": zs.cpu().numpy(),
        "box_pred": st.reshape(T + 1, -1, 9).cpu().numpy(),
        "box_gt": b["box"][:T + 1].cpu().numpy(),
        "event_pred": torch.sigmoid(ev).cpu().numpy(),
        "event_gt": b["events"][:T + 1].cpu().numpy(),
        "task_pred": tk.cpu().numpy(),
        "task_gt": b["task"][:T + 1].cpu().numpy(),
        "rgb_pred": rgb.cpu().numpy(),
        "depth_pred": dep.cpu().numpy(),
        "time": np.arange(T + 1),
        "visibility": b["visibility"][:T + 1].cpu().numpy(),
        "sem_gt": b["sem"][:T + 1].cpu().numpy(),
        "depth_gt": b["depth"][:T + 1].cpu().numpy(),
    }


def collect_rollouts(model: WorldModel, ds: TrajectoryDataset, horizon: int = 16,
                     max_episodes: int = 30, device: torch.device = None,
                     verbose: bool = True) -> List[Dict[str, np.ndarray]]:
    """收集若干条 rollout。

    ★ 必须把所有 rollout **截断到公共的最短长度**：轨迹可能提前终止（成功或掉桌），
      于是每条的长度不一样；而 `dynamics_accuracy` 等指标内部要 `np.stack` 成批，
      长度不齐会直接抛 `all input arrays must have the same shape`
      —— 在 hard 分片上真实踩到过（该分片的 episode 更容易提前结束）。
    """
    device = device or torch.device("cpu")
    out = []
    for eid in range(min(max_episodes, len(ds.ep_index))):
        r = rollout_episode(model, ds, eid, horizon, device)
        if r:
            out.append(r)
    if not out:
        return out
    Tmin = min(r["time"].shape[0] for r in out)
    for r in out:
        for k, v in list(r.items()):
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] != Tmin \
                    and k not in ("rgb_pred", "sem_gt", "depth_pred", "depth_gt",
                                  "z_pred", "box_pred", "box_gt", "event_pred",
                                  "event_gt", "task_pred", "task_gt", "visibility"):
                r[k] = v[:Tmin]
            elif isinstance(v, np.ndarray) and v.shape[0] > Tmin:
                r[k] = v[:Tmin]
    if verbose:
        print("[rollout] %d 条轨迹 x %d 步（已截断到公共长度）" % (len(out), Tmin))
    return out


# ---------------------------------------------------------------- 指标


def dynamics_accuracy(rolls: Sequence[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """动力学准确性：方块中心位置误差随步长的 ADE / FDE（米）。

    ★ `err_t0` 是**最重要的诊断量**：它是第 0 帧的位置误差，而第 0 帧的潜在
      来自**真实观测的编码**，没有任何预测成分。所以 `err_t0` 衡量的是
      "状态头能不能从真实观测里读出物体位置"，即**表征天花板**。
      如果 err_t0 本身就很大，那 rollout 误差再小也说明不了动力学好 ——
      问题出在表征/泛化，而不是动力学。本项目就是这种情况（见 docs/benchmark.md）。
    """
    if not rolls:
        return {}
    errs = np.stack([np.linalg.norm(r["box_pred"][:, :, :2] - r["box_gt"][:, :, :2], axis=-1)
                     for r in rolls])                      # [E,T+1,N]
    per_t = errs.mean(axis=(0, 2))
    return {
        "ade": float(errs.mean()),
        "fde": float(errs[:, -1].mean()),
        "err_t0": float(per_t[0]),
        "err_t1": float(per_t[1]) if len(per_t) > 1 else float("nan"),
        "err_t4": float(per_t[min(4, len(per_t) - 1)]),
        "err_t8": float(per_t[min(8, len(per_t) - 1)]),
        "err_curve": [round(float(v), 5) for v in per_t],
    }


def spatial_consistency(rolls: Sequence[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """空间/3D 一致性：深度 MAE 与占据 IoU（用解码深度对比真值深度）。"""
    if not rolls:
        return {}
    maes, ious = [], []
    for r in rolls:
        dp = r["depth_pred"][:, 0]                  # [T+1,H,W]
        dg = r["depth_gt"][:, 0]
        maes.append(float(np.abs(dp - dg).mean()))
        op = dp > 0.01
        og = dg > 0.01
        inter = np.logical_and(op, og).sum(axis=(1, 2))
        union = np.logical_or(op, og).sum(axis=(1, 2))
        ious.append(float((inter / np.maximum(union, 1)).mean()))
    return {"depth_mae": float(np.mean(maes)), "occupancy_iou": float(np.mean(ious))}


def long_horizon(rolls: Sequence[Dict[str, np.ndarray]],
                 marks: Sequence[int] = (1, 2, 4, 8, 12, 16)) -> Dict[str, float]:
    """长期预测：分步长的位置误差与任务量（目标距离）误差。"""
    if not rolls:
        return {}
    errs = np.stack([np.linalg.norm(r["box_pred"][:, :, :2] - r["box_gt"][:, :, :2], axis=-1)
                     for r in rolls]).mean(axis=(0, 2))
    terr = np.stack([np.abs(r["task_pred"][:, 0] - r["task_gt"][:, 0]) for r in rolls]).mean(axis=0)
    out = {}
    for m in marks:
        if m < len(errs):
            out["pos_err_t%d" % m] = round(float(errs[m]), 5)
            out["task_err_t%d" % m] = round(float(terr[m]), 5)
    out["degradation_ratio_t16_t1"] = round(
        float(errs[min(16, len(errs) - 1)] / max(1e-9, errs[1])), 4)
    return out


def object_permanence(rolls: Sequence[Dict[str, np.ndarray]],
                      occl_thresh: float = 0.05) -> Dict[str, float]:
    """物体持久性：把误差按"目标物体当帧是否被遮挡"分段。

    这是本项目最有意思的一项 —— 遮挡墙后面的方块在观测里**完全不可见**
    （可见比例 0），模型只能靠记忆维持它的状态。如果模型有持久性表征，
    被遮挡段的误差应与可见段接近；如果没有，被遮挡段误差会显著抬高。
    """
    vis_err, occ_err = [], []
    for r in rolls:
        e = np.linalg.norm(r["box_pred"][:, :, :2] - r["box_gt"][:, :, :2], axis=-1)  # [T+1,N]
        v = r["visibility"]                                                          # [T+1,N]
        m = v < occl_thresh
        if m.any():
            occ_err.append(float(e[m].mean()))
        if (~m).any():
            vis_err.append(float(e[~m].mean()))
    out = {"visible_err": float(np.mean(vis_err)) if vis_err else float("nan")}
    if occ_err:
        out["occluded_err"] = float(np.mean(occ_err))
        out["occlusion_penalty"] = round(out["occluded_err"] / max(1e-9, out["visible_err"]), 3)
    else:
        out["occluded_err"] = float("nan")
        out["occlusion_penalty"] = float("nan")
    out["n_occluded_points"] = int(sum(1 for _ in occ_err))
    return out


def failure_modes(rolls: Sequence[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """失败模式分桶：有接触 / 无接触 / 目标被遮挡。"""
    buckets = {"contact": [], "no_contact": [], "occluded": []}
    for r in rolls:
        e = np.linalg.norm(r["box_pred"][:, :, :2] - r["box_gt"][:, :, :2], axis=-1).mean(axis=1)
        contact = r["event_gt"][:, :r["box_gt"].shape[1]].max(axis=1) > 0.5
        occl = (r["visibility"] < 0.05).any(axis=1)
        for k, m in (("contact", contact), ("no_contact", ~contact), ("occluded", occl)):
            if m.any():
                buckets[k].append(float(e[m].mean()))
    return {("%s_err" % k): (float(np.mean(v)) if v else float("nan"))
            for k, v in buckets.items()}


def action_controllability(model: WorldModel, cf_path: str, device: torch.device,
                           max_pairs: int = 60, suffix: int = 8) -> Dict[str, float]:
    """动作可控性（本项目的核心指标）。

    做法：取反事实配对 —— 两条支路观测前缀完全一致，只有第 k+1 步起的动作不同。
    把**同一帧**编码成 z0，然后分别用两条动作后缀在潜空间 rollout，得到两条预测轨迹。
    同时把真值未来的观测也编码出来，得到真值潜在轨迹。然后看：

        预测两条支路的"分歧"是否跟真值的"分歧"同向、同序

    - `divergence_corr`：逐对的预测分歧 vs 真值分歧的皮尔逊相关（越接近 1 越说明
      模型把动作差异正确地传播成了状态差异）。
    - `sensitivity_ratio`：**预测分歧 / 真值分歧**的均值之比。这是"敏感度标定"——
      接近 1 说明模型对动作的响应幅度和真实世界一致；远小于 1 说明动作被忽略，
      远大于 1 说明模型对动作过度敏感。
    - `action_vs_zero_ratio`：预测分歧 / "把动作全置零"的基线分歧。用来抓那种
      "不管给什么动作都预测同一件事"的退化模型（该值应显著大于 1）。
    """
    z = np.load(cf_path)
    n = min(max_pairs, z["rgb_a"].shape[0])
    model.eval()
    pred_div, gt_div, base_div = [], [], []
    with torch.no_grad():
        for i in range(n):
            m = min(suffix, z["rgb_a"].shape[1], z["rgb_b"].shape[1])
            if m < 3:
                continue
            # ★ 分叉起点帧两条支路共用同一份（rgb0），这是反事实对照成立的前提
            if "rgb0" in z.files:
                rgb0 = z["rgb0"][i].astype(np.float32) / 255.0
            else:                                    # 兼容旧数据
                rgb0 = z["rgb_a"][i, 0].astype(np.float32) / 255.0

            def to_x(rgb_seq):
                """接受 [H,W,3] / [T,H,W,3] / [T,3,H,W]，统一成 [T,C,H,W]。"""
                t = torch.from_numpy(np.asarray(rgb_seq, dtype=np.float32) / 255.0)
                if t.dim() == 3:                       # [H,W,3] 单帧
                    t = t.permute(2, 0, 1)[None]
                elif t.dim() == 4 and t.shape[-1] == 3:   # [T,H,W,3]
                    t = t.permute(0, 3, 1, 2)
                T, C, H, W = t.shape
                assert C == 3, "RGB 通道数不对: %s" % (tuple(t.shape),)
                dep = torch.zeros(T, 1, H, W)        # 反事实集只存了 RGB
                sem = torch.zeros(T, 9, H, W)
                return torch.cat([t, dep, sem], dim=1).to(device)

            xa, xb = to_x(z["rgb_a"][i, :m]), to_x(z["rgb_b"][i, :m])
            za, _, _, _ = model.encode(xa, sample=False)
            zb, _, _, _ = model.encode(xb, sample=False)
            gt_d = float((za - zb).pow(2).mean().sqrt())

            z0 = model.encode(to_x(rgb0), sample=False)[0]      # [1,Z]
            aa = torch.from_numpy(z["action_a"][i, :m - 1]).float().to(device)[None]
            ab = torch.from_numpy(z["action_b"][i, :m - 1]).float().to(device)[None]
            # 本体状态是 6 维（执行器 xy、上一步动作 xy、步数占比、剩余占比），
            # 不能用 zeros_like(aa)（那是 2 维）—— 这是本项目真实踩到的形状错。
            prop = torch.zeros_like(aa)[..., :1].repeat(1, 1, 6)
            lang = torch.zeros(1, model.cfg.lang_dim, device=device)

            ra = model.rollout(z0, aa, prop, lang)[0, -1]
            rb = model.rollout(z0, ab, prop, lang)[0, -1]
            r0 = model.rollout(z0, torch.zeros_like(aa), prop, lang)[0, -1]
            pred_div.append(float((ra - rb).pow(2).mean().sqrt()))
            base_div.append(float((ra - r0).pow(2).mean().sqrt()))
            gt_div.append(gt_d)

    pred_div = np.asarray(pred_div)
    gt_div = np.asarray(gt_div)
    base_div = np.asarray(base_div)
    if len(pred_div) < 3:
        return {"n_pairs": int(len(pred_div)), "divergence_corr": float("nan")}
    corr = float(np.corrcoef(pred_div, gt_div)[0, 1]) if pred_div.std() > 0 else float("nan")
    return {
        "n_pairs": int(len(pred_div)),
        "divergence_corr": round(corr, 4),
        "pred_divergence_mean": round(float(pred_div.mean()), 5),
        "gt_divergence_mean": round(float(gt_div.mean()), 5),
        "sensitivity_ratio": round(float(pred_div.mean() / max(1e-9, gt_div.mean())), 4),
        "action_vs_zero_ratio": round(float(pred_div.mean() / max(1e-9, base_div.mean())), 4),
    }


# ---------------------------------------------------------------- 汇总


def summarize(rolls: Sequence[Dict[str, np.ndarray]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out.update(dynamics_accuracy(rolls))
    out.update(spatial_consistency(rolls))
    out.update(long_horizon(rolls))
    out.update(object_permanence(rolls))
    out.update(failure_modes(rolls))
    if rolls:
        tp = np.concatenate([r["task_pred"][:, 1] for r in rolls])
        tg = np.concatenate([r["task_gt"][:, 1] for r in rolls])
        out["success_auc"] = round(_auc(tg, tp), 4)
        out["success_pred_acc"] = round(float(((tp > 0.5) == (tg > 0.5)).mean()), 4)
        # 成功事件在短 rollout 里非常稀疏，AUC 会是 NaN；补一个连续量的相关性
        dp = np.concatenate([r["task_pred"][:, 0] for r in rolls])
        dg = np.concatenate([r["task_gt"][:, 0] for r in rolls])
        out["dist_corr"] = (round(float(np.corrcoef(dp, dg)[0, 1]), 4)
                            if dp.std() > 1e-9 and dg.std() > 1e-9 else float("nan"))
        out["dist_mae"] = round(float(np.abs(dp - dg).mean()), 5)
        out["n_success_frames"] = int((tg > 0.5).sum())
    return out


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC（Mann-Whitney U）。**必须处理并列分数** —— 用平均秩，
    否则全部分数相同时会算出 1.0（本项目的测试抓到过这个错）。"""
    y = (y > 0.5).astype(np.int64)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
