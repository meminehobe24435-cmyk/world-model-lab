# -*- coding: utf-8 -*-
"""数据层测试：轨迹打包、分片读写、**反事实配对的前缀一致性**。"""

from __future__ import annotations

import numpy as np
import pytest

from wm.data.collector import (Episode, PushPlanner, collect_counterfactual,
                               collect_episode, perturb_action, random_policy,
                               save_dataset)
from wm.data.dataset import (SEM_IDS, TrajectoryDataset, sem_to_onehot)
from wm.env.world import PushWorld


@pytest.fixture(scope="module")
def world():
    return PushWorld(size=32, horizon=10)


def test_episode_keys_and_shapes(world):
    e = collect_episode(world, 100, "expert", horizon=10)
    d = e.to_dict()
    for k in ("rgb", "depth", "semantic", "proprio", "action", "box_matrix",
              "events", "task", "reward", "visibility"):
        assert k in d, "缺字段 %s" % k
        assert d[k].shape[0] == d["rgb"].shape[0]
    T = d["rgb"].shape[0]
    assert T == 11, "首帧 + 10 步 = 11 帧"
    assert d["rgb"].dtype == np.uint8
    assert d["depth"].dtype == np.float16
    assert d["action"].shape == (T, 2)
    assert d["box_matrix"].shape[1] == world.n_boxes
    assert d["events"].shape[1] == 2 * world.n_boxes + 1
    assert d["task"].shape[1] == 4
    assert d["visibility"].shape[1] == world.n_boxes
    assert e.meta["n_steps"] == T


def test_episode_initial_frame_has_all_keys(world):
    """初始帧必须补齐后续帧的键，否则 _pack 会按首帧键集裁剪（真实踩过的坑）。"""
    e = collect_episode(world, 101, "expert", horizon=6)
    assert e.action.shape[0] == 7
    assert np.allclose(e.action[0], 0.0), "首帧动作应为 0"
    assert np.allclose(e.events[0], 0.0), "首帧事件应为 0"
    assert float(e.reward[0]) == 0.0


def test_random_policy_range(world):
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = random_policy(rng)
        assert a.shape == (2,) and np.all(np.abs(a) <= 1.0)


def test_perturb_action_clipped():
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = perturb_action(np.array([1.0, -1.0]), rng, sigma=2.0)
        assert np.all(np.abs(a) <= 1.0)


def test_expert_planner_reaches_contact(world):
    """专家至少要在有限步内接触到目标方块。"""
    world.reset(seed=100)
    pl = PushPlanner()
    touched = False
    for _ in range(40):
        a = pl.act(world)
        _, _, done, gt = world.step(a)
        if gt["events"][:world.n_boxes].sum() > 0:
            touched = True
            break
        if done:
            break
    assert touched, "脚本专家没有接触到目标方块"


def test_expert_planner_moves_target_toward_goal():
    """专家应当把目标方块推得离目标区更近（这是数据可用性的前提）。"""
    world = PushWorld(size=32, horizon=48, n_boxes=4)
    world.reset(seed=1000)
    d0 = world.physics.task_info()["target_dist"]
    pl = PushPlanner()
    for _ in range(48):
        a = pl.act(world)
        _, _, done, _ = world.step(a)
        if done:
            break
    d1 = world.physics.task_info()["target_dist"]
    assert d1 < d0, "专家没有把方块推向目标区: %.3f -> %.3f" % (d0, d1)


def test_expert_success_rate_nonzero():
    """在 20 个布局上专家必须至少成功若干次 —— 否则数据里几乎没有正样本。"""
    world = PushWorld(size=32, horizon=48, n_boxes=4)
    ok = sum(int(collect_episode(world, 1000 + i, "expert").meta["success"] > 0)
             for i in range(20))
    assert ok >= 2, "专家成功率过低（%d/20），数据里缺正样本" % ok


def test_planner_stateful_no_limit_cycle():
    """★ 回归测试：执行器不能在两个位置之间无限来回（两帧极限环）。

    真实踩过的坑 —— 绕行判定逐帧翻转会让执行器卡死在两格之间，
    跑满整个 horizon 一步都不推进。这里用"位置集合的势"来抓它。
    """
    world = PushWorld(size=32, horizon=40, n_boxes=4)
    world.reset(seed=1000)
    pl = PushPlanner()
    seen = set()
    n_repeat = 0
    for _ in range(40):
        a = pl.act(world)
        _, _, done, _ = world.step(a)
        key = (round(float(world.physics.effector[0]), 4),
               round(float(world.physics.effector[1]), 4))
        if key in seen:
            n_repeat += 1
        seen.add(key)
        if done:
            break
    assert n_repeat < 5, "执行器出现极限环（重复位置 %d 次）" % n_repeat


def test_counterfactual_prefix_identical(world):
    """★ 反事实数据的可信前提：两条支路必须共用**同一个分叉起点帧**。

    真实踩过的坑：早期实现没有显式保存分叉起点，评测时误用"分叉后第 1 帧"
    当起点 —— 那一帧已经被动作影响了，两条支路本来就该不同，于是起点一致性
    根本不成立，z0 也不再是同一个状态。
    """
    p = collect_counterfactual(world, 100, branch_step=4, suffix=4)
    assert p.rgb0 is not None, "必须显式保存分叉起点帧"
    assert p.rgb0.shape[-1] == 3
    assert p.prefix_identical()
    assert p.branch_step == 4
    assert p.prefix_actions.shape[0] == 4, "前缀必须跑满，不能提前终止"
    # 两条支路都严格跑满 suffix 步（否则没法 stack 成批）
    assert p.action_a.shape[0] == 4 and p.action_b.shape[0] == 4
    assert p.rgb_a.shape == p.rgb_b.shape
    assert p.box_a.shape == p.box_b.shape


def test_counterfactual_branch_start_is_deterministic(world):
    """同一个 seed 重采，分叉起点帧必须逐像素一致（确定性）。"""
    p1 = collect_counterfactual(world, 123, branch_step=3, suffix=3)
    p2 = collect_counterfactual(world, 123, branch_step=3, suffix=3)
    assert np.array_equal(p1.rgb0, p2.rgb0), "分叉起点不确定"
    assert np.allclose(p1.action_a, p2.action_a)


def test_counterfactual_actions_differ(world):
    """扰动后缀必须与原子后缀不同，否则对照没有意义。"""
    p = collect_counterfactual(world, 100, branch_step=4, suffix=5, sigma=0.9)
    assert not np.allclose(p.action_a, p.action_b), "两条支路动作完全相同"
    d = np.abs(p.box_a[-1, 0, :2] - p.box_b[-1, 0, :2]).sum()
    assert d > 1e-4, "扰动没有造成状态差异，对照无效"


def test_save_and_load_shards(world, tmp_path):
    eps = [collect_episode(world, 200 + i, "expert", horizon=6) for i in range(3)]
    files = save_dataset(eps, str(tmp_path), "tiny", max_steps_per_shard=100)
    assert files, "没有写出分片"
    ds = TrajectoryDataset(str(tmp_path), tags=("tiny",), seq_len=4, stride=2,
                           verbose=False)
    assert ds.n_steps == sum(e.meta["n_steps"] for e in eps)
    assert ds.n_boxes == world.n_boxes
    b = ds[0]
    assert b["x"].shape[1] == 4 + 9
    assert b["x"].shape[0] == 4
    assert b["sem"].shape[1:] == (32, 32)
    assert float(b["x"].min()) >= 0.0 and float(b["x"].max()) <= 1.0
    assert len(ds) > 0


def test_sem_to_onehot_exact():
    sem = np.zeros((2, 4, 4), dtype=np.uint8)
    sem[0, 0, 0] = 1
    sem[1, 3, 3] = 101
    oh = sem_to_onehot(sem)
    assert oh.shape == (2, len(SEM_IDS), 4, 4)
    assert oh.sum(axis=1).max() == 1.0
    assert oh[0, SEM_IDS.index(1), 0, 0] == 1.0
    assert oh[1, SEM_IDS.index(101), 3, 3] == 1.0
    assert oh[0, SEM_IDS.index(0)].sum() == 15.0
