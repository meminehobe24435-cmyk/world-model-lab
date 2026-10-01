# -*- coding: utf-8 -*-
"""评测层测试：指标正确性（用构造的真值验证）+ 潜空间规划可运行性。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wm.bench import metrics as M
from wm.bench.planning import PlanConfig, _proprio_rollout, run_mpc
from wm.env.physics import MAX_STEP
from wm.env.world import PushWorld
from wm.models.world_model import WMConfig, WorldModel


def synth_rollout(T=6, N=2, H=8, W=8, offset=0.0):
    """构造一条"预测 = 真值 + 常数偏移"的假 rollout。"""
    box_gt = np.zeros((T + 1, N, 9), dtype=np.float32)
    box_gt[:, :, 0] = np.linspace(0.1, 0.2, T + 1)[:, None]
    box_gt[:, :, 1] = 0.5
    box_gt[:, :, 8] = 1.0
    box_pred = box_gt.copy()
    box_pred[:, :, 0] += offset
    depth_gt = np.zeros((T + 1, 1, H, W), dtype=np.float32)
    depth_gt[:, :, 2:5, 2:5] = 0.5
    return {
        "box_pred": box_pred, "box_gt": box_gt,
        "depth_pred": depth_gt.copy(), "depth_gt": depth_gt,
        "task_pred": np.zeros((T + 1, 4), dtype=np.float32),
        "task_gt": np.zeros((T + 1, 4), dtype=np.float32),
        "event_pred": np.zeros((T + 1, 2 * N + 1), dtype=np.float32),
        "event_gt": np.zeros((T + 1, 2 * N + 1), dtype=np.float32),
        "visibility": np.ones((T + 1, N), dtype=np.float32),
        "sem_gt": np.zeros((T + 1, H, W), dtype=np.uint8),
        "rgb_pred": np.zeros((T + 1, 3, H, W), dtype=np.float32),
        "z_pred": np.zeros((T + 1, 4), dtype=np.float32),
        "time": np.arange(T + 1),
    }


# ---------------------------------------------------------------- 指标


def test_dynamics_accuracy_zero_for_perfect_prediction():
    r = M.dynamics_accuracy([synth_rollout(offset=0.0)])
    assert r["ade"] == pytest.approx(0.0, abs=1e-9)
    assert r["fde"] == pytest.approx(0.0, abs=1e-9)
    assert len(r["err_curve"]) == 7


def test_dynamics_accuracy_matches_known_offset():
    r = M.dynamics_accuracy([synth_rollout(offset=0.03)])
    assert r["ade"] == pytest.approx(0.03, abs=1e-6)
    assert r["err_t8"] == pytest.approx(0.03, abs=1e-6)


def test_spatial_consistency_perfect_and_broken():
    good = M.spatial_consistency([synth_rollout()])
    assert good["depth_mae"] == pytest.approx(0.0, abs=1e-9)
    assert good["occupancy_iou"] == pytest.approx(1.0, abs=1e-6)

    bad = synth_rollout()
    bad["depth_pred"] = np.zeros_like(bad["depth_pred"])
    r = M.spatial_consistency([bad])
    assert r["occupancy_iou"] == pytest.approx(0.0, abs=1e-6)
    assert r["depth_mae"] > 0.0


def test_object_permanence_splits_by_visibility():
    r = synth_rollout(offset=0.05)
    r["visibility"][3:, 0] = 0.0          # 第 3 步起第 0 块被遮挡
    out = M.object_permanence([r])
    assert out["n_occluded_points"] == 1
    assert out["occluded_err"] > 0.0
    assert out["visible_err"] >= 0.0
    assert np.isfinite(out["occlusion_penalty"])


def test_failure_mode_buckets():
    r = synth_rollout(offset=0.02)
    r["event_gt"][2, 0] = 1.0              # 有接触
    out = M.failure_modes([r])
    for k in ("contact_err", "no_contact_err", "occluded_err"):
        assert k in out


def test_long_horizon_degradation_ratio_finite():
    out = M.long_horizon([synth_rollout(offset=0.02)], marks=(1, 4))
    assert "pos_err_t1" in out and "pos_err_t4" in out
    assert np.isfinite(out["degradation_ratio_t16_t1"])


def test_auc_known_values():
    y = np.array([0, 0, 1, 1], dtype=np.float32)
    assert M._auc(y, np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(1.0)
    assert M._auc(y, np.array([0.4, 0.3, 0.2, 0.1])) == pytest.approx(0.0)
    assert M._auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_summarize_returns_all_dimensions():
    out = M.summarize([synth_rollout()])
    for k in ("ade", "fde", "depth_mae", "occupancy_iou", "occluded_err",
              "contact_err", "success_auc"):
        assert k in out, "汇总缺维度 %s" % k


# ---------------------------------------------------------------- 规划


def test_proprio_rollout_shapes_and_accumulation():
    eff0 = np.array([0.5, 0.1])
    actions = np.ones((3, 4, 2), dtype=np.float32)
    p = _proprio_rollout(eff0, actions, step0=0, horizon=4, env_horizon=32)
    assert p.shape == (3, 4, 6)
    # 每步都应沿 +MAX_STEP 累积
    assert p[0, 0, 0] == pytest.approx(0.5 + MAX_STEP, abs=1e-6)
    assert p[0, 3, 0] == pytest.approx(0.5 + 4 * MAX_STEP, abs=1e-6)
    assert p[0, 3, 1] == pytest.approx(0.1 + 4 * MAX_STEP, abs=1e-6)
    # 步数占比单调递增且在 [0,1]
    assert np.all(np.diff(p[0, :, 4]) > 0)
    assert p[0, -1, 4] <= 1.0


@pytest.mark.parametrize("policy", ["random", "scripted"])
def test_baseline_policies_run(policy):
    w = PushWorld(size=16, horizon=6)
    m = WorldModel(WMConfig(img=16, z_dim=8, base=8, d_model=32, n_layer=1,
                            n_head=2, n_boxes=4, n_events=9))
    cfg = PlanConfig(n_samples=4, horizon=2, n_rounds=1, max_steps=4, seed=0)
    r = run_mpc(m, [1, 2], torch.device("cpu"), cfg, policy=policy, world=w,
                verbose=False)
    assert r["n"] == 2
    assert 0.0 <= r["success_rate"] <= 1.0
    assert r["mean_steps"] >= 1.0


def test_mpc_policy_runs_and_respects_step_budget():
    w = PushWorld(size=16, horizon=6)
    m = WorldModel(WMConfig(img=16, z_dim=8, base=8, d_model=32, n_layer=1,
                            n_head=2, n_boxes=4, n_events=9))
    cfg = PlanConfig(n_samples=6, horizon=2, n_rounds=1, max_steps=5, seed=0)
    r = run_mpc(m, [3, 4], torch.device("cpu"), cfg, policy="mpc", world=w,
                verbose=False)
    assert r["n"] == 2
    assert r["mean_steps"] <= 5.0
    assert "mean_pred_dist" in r


def test_mpc_does_not_crash_with_no_contacts():
    """即使模型完全没学到东西（随机初始化），规划也必须安全跑完。"""
    w = PushWorld(size=16, horizon=4)
    m = WorldModel(WMConfig(img=16, z_dim=8, base=8, d_model=32, n_layer=1,
                            n_head=2, n_boxes=4, n_events=9))
    cfg = PlanConfig(n_samples=4, horizon=2, n_rounds=2, max_steps=3, seed=1)
    r = run_mpc(m, [7], torch.device("cpu"), cfg, policy="mpc", world=w, verbose=False)
    assert np.isfinite(r["success_rate"])
    assert np.isfinite(r["mean_final_dist"])
