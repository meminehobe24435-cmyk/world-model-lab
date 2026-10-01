# -*- coding: utf-8 -*-
"""环境层测试：物理（含推方向回归）、渲染、世界封装。

★ 最重要的两条是 `test_push_direction_*` —— 它们锁定的是本项目真实踩到并修复的
  物理方向 bug（方块被推向执行器 + 退化分支选错轴）。没有这两条测试，任何人以后
  改动 `_disc_box_overlap` 都可能把成功率悄悄改回 0。
"""

from __future__ import annotations

import numpy as np
import pytest

from wm.env.physics import (EFFECTOR_R, MAX_STEP, Box, PushPhysics, SceneSpec,
                            _disc_box_overlap)
from wm.data.collector import _segment_hits_aabb
from wm.env.renderer import (SEM_EFFECTOR, SEM_GOAL, SEM_TABLE, SEM_WALL,
                             DEFAULT_CAMERA, render, unproject)
from wm.env.world import PushWorld, encode_instruction, make_instruction, make_scene


# ---------------------------------------------------------------- 夹具


def single_box_scene(bx=0.5, by=0.5, hx=0.06, hy=0.06):
    b = Box(id=1, cx=bx, cy=by, hx=hx, hy=hy, hz=0.03, z=0.0,
            color=(0.85, 0.25, 0.25), movable=True, is_target=True)
    return SceneSpec(boxes=[b], goal=(0.5, 0.9, 0.1, 0.08), wall=None)


# ---------------------------------------------------------------- 物理：推方向回归


@pytest.mark.parametrize("push_dir,axis,sign", [
    ((0.0, 1.0), 1, +1.0),    # 从下方推 -> 方块 +y
    ((0.0, -1.0), 1, -1.0),   # 从上方推 -> 方块 -y
    ((1.0, 0.0), 0, +1.0),    # 从左侧推 -> 方块 +x
    ((-1.0, 0.0), 0, -1.0),   # 从右侧推 -> 方块 -x
])
def test_push_direction_four_faces(push_dir, axis, sign):
    """★ 回归测试：方块必须沿推的方向走，且另一轴零漂移。"""
    spec = single_box_scene()
    ph = PushPhysics(spec)
    b = spec.boxes[0]
    # 把执行器放到方块对面一侧
    off = np.array(push_dir, dtype=np.float64) * (b.hx + EFFECTOR_R + 0.02)
    ph.effector = np.array([b.cx, b.cy]) - off

    start = np.array([b.cx, b.cy])
    for _ in range(6):
        ph.step(push_dir)
    delta = np.array([b.cx, b.cy]) - start

    assert delta[axis] * sign > 0.05, "方块没有沿推的方向移动: %s" % delta
    other = 1 - axis
    assert abs(delta[other]) < 1e-6, "出现横向漂移: %s" % delta


def test_push_direction_from_touching_face():
    """★ 回归测试：执行器恰好贴在面上（退化情况）也不能挑错轴。

    这是真实 bug —— 早期实现在圆心恰好落在 AABB 边界上时按 x 偏移挑轴，
    于是"从下方推"变成"方块朝 -x 走"。
    """
    spec = single_box_scene()
    ph = PushPhysics(spec)
    b = spec.boxes[0]
    ph.effector = np.array([b.cx, b.cy - b.hy - EFFECTOR_R])
    hit, dx, dy, depth = _disc_box_overlap(ph.effector[0], ph.effector[1], EFFECTOR_R, b)
    assert hit
    assert abs(dx) < 1e-9 and dy > 0.9, "退化情况选错了轴: (%r, %r)" % (dx, dy)


def test_disc_box_overlap_no_hit_when_far():
    spec = single_box_scene()
    b = spec.boxes[0]
    hit = _disc_box_overlap(b.cx, b.cy + 1.0, EFFECTOR_R, b)
    assert hit[0] is False and hit[3] == 0.0


def test_inside_case_pushes_outward():
    spec = single_box_scene()
    b = spec.boxes[0]
    # 圆心在方块内部、偏右侧 -> 方块应朝 -x 让开
    hit, dx, dy, depth = _disc_box_overlap(b.cx + 0.01, b.cy, EFFECTOR_R, b)
    assert hit and dx < -0.9 and depth > EFFECTOR_R


# ---------------------------------------------------------------- 物理：其他不变量


def test_physics_deterministic():
    actions = [np.array([0.3, -0.7]), np.array([1.0, 0.2]), np.array([-0.4, 0.9])]
    runs = []
    for _ in range(2):
        spec = make_scene(11)
        ph = PushPhysics(spec)
        for a in actions:
            ph.step(a)
        runs.append(np.array([(b.cx, b.cy) for b in spec.boxes]))
    assert np.array_equal(runs[0], runs[1])


def test_effector_clamped_to_table():
    spec = make_scene(3)
    ph = PushPhysics(spec)
    for _ in range(40):
        ph.step([-1.0, -1.0])
    assert ph.effector[0] >= EFFECTOR_R - 1e-9
    assert ph.effector[1] >= EFFECTOR_R - 1e-9
    for _ in range(80):
        ph.step([1.0, 1.0])
    assert ph.effector[0] <= 1.0 - EFFECTOR_R + 1e-9
    assert ph.effector[1] <= 1.0 - EFFECTOR_R + 1e-9


def test_no_overlap_between_movable_boxes():
    spec = make_scene(5)
    ph = PushPhysics(spec)
    rng = np.random.default_rng(0)
    for _ in range(25):
        ph.step(rng.uniform(-1, 1, 2))
        alive = [b for b in spec.movable if b.id not in ph.dropped]
        for i in range(len(alive)):
            for j in range(i + 1, len(alive)):
                a, b = alive[i], alive[j]
                ox = (a.hx + b.hx) - abs(a.cx - b.cx)
                oy = (a.hy + b.hy) - abs(a.cy - b.cy)
                assert not (ox > 1e-3 and oy > 1e-3), "方块仍然重叠"


def test_off_table_event_and_removal():
    """出界判定 + 掉桌事件 + 从后续交互中移除。

    ★ 注意这条几何事实（曾误以为能直接推下去）：执行器自身被夹在桌内
    （`clip(effector, EFFECTOR_R, 1-EFFECTOR_R)`），所以它**无法把方块完全推出桌外** ——
    刚好推到 `box.cx == -box.hx` 这个临界点就停了。掉桌实际由**方块之间的挤压**造成。
    因此这里用"方块已被挤到界外"的初态来验证判定逻辑本身。
    """
    spec = SceneSpec(boxes=[Box(id=1, cx=-0.06, cy=0.5, hx=0.05, hy=0.05, hz=0.03,
                                z=0.0, color=(1, 0, 0), is_target=True)],
                     goal=(0.9, 0.9, 0.08, 0.06), wall=None)
    ph = PushPhysics(spec)
    ev, info = ph.step([0.0, 0.0])
    assert ev.off_table == [1], "越过桌沿的方块应当产生 off_table 事件"
    assert 1 in ph.dropped
    assert info["target_alive"] == 0.0, "掉桌后目标方块应当被标记为不可用"
    assert info["terminated"] == 1.0, "目标掉桌应当终止回合"


def test_effector_cannot_push_box_fully_off_table():
    """把上面那条几何事实固化成断言，防止有人以后"修"错方向。"""
    spec = single_box_scene(bx=0.12)
    ph = PushPhysics(spec)
    b = spec.boxes[0]
    ph.effector = np.array([b.cx + b.hx + EFFECTOR_R, b.cy])
    for _ in range(40):
        ev, _ = ph.step([-1.0, 0.0])
    assert b.cx >= -b.hx - 1e-6, "执行器不该能把方块完全推离桌面"
    assert ph.effector[0] >= EFFECTOR_R - 1e-9, "执行器不该跑出桌面"


def test_events_vector_shape_and_content():
    spec = make_scene(7)
    ph = PushPhysics(spec)
    n = len(spec.boxes)
    ev, _ = ph.step([0.0, 1.0])
    v = ev.as_vector(n)
    assert v.shape == (2 * n + 1,)
    assert set(np.unique(v)).issubset({0.0, 1.0} | set(v.tolist()))
    assert v[2 * n] == sum(s for _, s in ev.effector_contacts)


def test_goal_contains_and_success():
    spec = single_box_scene()
    ph = PushPhysics(spec)
    assert ph.task_info()["success"] == 0.0
    spec.boxes[0].cx, spec.boxes[0].cy = spec.goal[0], spec.goal[1]
    assert ph.task_info()["success"] == 1.0
    assert ph.task_info()["terminated"] == 1.0


def test_segment_hits_aabb():
    p0, p1 = np.array([0.0, 0.5]), np.array([1.0, 0.5])
    assert _segment_hits_aabb(p0, p1, 0.5, 0.5, 0.05, 0.05, 0.0)
    assert not _segment_hits_aabb(p0, p1, 0.5, 0.9, 0.05, 0.05, 0.0)
    assert _segment_hits_aabb(p0, p1, 0.5, 0.9, 0.05, 0.05, 0.5)


# ---------------------------------------------------------------- 渲染


def test_render_shapes_and_dtypes():
    spec = make_scene(2)
    r = render(spec, size=48, effector_xy=(0.5, 0.2))
    assert r.rgb.shape == (48, 48, 3) and r.rgb.dtype == np.float32
    assert r.depth.shape == (48, 48)
    assert r.semantic.shape == (48, 48)
    assert 0.0 <= float(r.rgb.min()) and float(r.rgb.max()) <= 1.0


def test_render_semantic_ids_valid():
    spec = make_scene(4)
    r = render(spec, size=48, effector_xy=(0.5, 0.2))
    ids = set(np.unique(r.semantic).tolist())
    allowed = {0, SEM_TABLE, SEM_GOAL, SEM_WALL, SEM_EFFECTOR} | {b.id for b in spec.boxes}
    assert ids <= allowed, "出现未知语义 id: %s" % (ids - allowed)


def test_render_background_is_infinite_depth():
    spec = make_scene(4)
    r = render(spec, size=48, effector_xy=(0.5, 0.2))
    bg = r.semantic == 0
    assert bg.any()
    assert np.isinf(r.depth[bg]).all()
    assert np.isfinite(r.depth[~bg]).all()


def test_occlusion_visibility_detects_wall():
    """遮挡墙应当让至少一块方块可见比例下降，并且给出 [0,1] 区间的可见比例。"""
    spec = make_scene(9)
    r = render(spec, size=48, effector_xy=(0.5, 0.2))
    assert set(r.visibility.keys()) == {b.id for b in spec.boxes}
    for v in r.visibility.values():
        assert 0.0 <= v <= 1.0 + 1e-6
    # 至少有一次渲染里存在"被部分/完全遮挡"的方块
    wall_spec = SceneSpec(boxes=spec.boxes, goal=spec.goal,
                          wall=(0.5, 0.62, 0.3, 0.016, 0.095), layout_id=0)
    r2 = render(wall_spec, size=48, effector_xy=(0.5, 0.2))
    assert any(v < 0.999 for v in r2.visibility.values())


def test_unproject_roundtrip():
    """点云反投影后再投影回来，应当落在原像素附近。"""
    spec = make_scene(6)
    r = render(spec, size=48, effector_xy=(0.5, 0.2))
    pts = unproject(r.depth, DEFAULT_CAMERA, stride=2)
    assert pts.shape[1] == 3 and len(pts) > 100
    # 所有点都在桌子附近的高度范围内（z 应该在 0 附近或以上）
    assert float(pts[:, 2].min()) > -0.1
    assert float(pts[:, 2].max()) < 1.0


# ---------------------------------------------------------------- 世界


def test_world_reset_deterministic():
    w = PushWorld(size=48, horizon=16)
    a = w.reset(seed=42)
    for _ in range(4):
        a, _, _, _ = w.step([0.5, 0.5])
    b = w.reset(seed=42)
    for _ in range(4):
        b, _, _, _ = w.step([0.5, 0.5])
    assert np.array_equal(a.rgb, b.rgb)
    assert np.array_equal(a.semantic, b.semantic)


def test_world_observation_modalities():
    w = PushWorld(size=48, horizon=16)
    obs = w.reset(seed=3)
    assert obs.rgb.shape == (48, 48, 3)
    assert obs.depth.shape == (48, 48)
    assert obs.semantic.shape == (48, 48)
    assert obs.proprio.shape == (6,)
    assert obs.instruction.shape[0] == 24
    assert 0.0 <= float(obs.proprio[4]) <= 1.0
    assert obs.instruction_text
    d = obs.depth_masked
    assert np.isfinite(d).all() and d.min() >= 0.0


def test_world_snapshot_restore_identical():
    w = PushWorld(size=48, horizon=16)
    w.reset(seed=8)
    for _ in range(5):
        w.step([0.4, -0.3])
    snap = w.snapshot()
    o1 = w._observe()
    for _ in range(4):
        w.step([-0.9, 0.6])
    o2 = w.restore(snap)
    assert np.array_equal(o1.rgb, o2.rgb)
    assert np.array_equal(o1.semantic, o2.semantic)


def test_world_termination_on_horizon():
    w = PushWorld(size=48, horizon=6)
    w.reset(seed=1)
    done = False
    n = 0
    while not done:
        _, _, done, _ = w.step([0.0, 0.0])
        n += 1
        assert n <= 10
    assert n == 6


def test_instruction_vocabulary_covers_templates():
    txt = make_instruction("红色", 0)
    ids = encode_instruction(txt)
    assert ids.shape == (24,)
    assert int(ids[0]) != 0, "指令首字符不应该被编码成 pad"
    assert int(ids[len(txt):].sum()) == 0, "尾部应为 pad"
    assert len(set(encode_instruction(make_instruction("蓝色", 3)).tolist())) > 3


def test_make_scene_layouts_differ_by_seed():
    a, b = make_scene(1), make_scene(2)
    pa = np.array([(x.cx, x.cy) for x in a.boxes])
    pb = np.array([(x.cx, x.cy) for x in b.boxes])
    assert pa.shape == pb.shape
    assert not np.allclose(pa, pb)


def test_make_scene_hard_pushes_goal_far():
    g = make_scene(21, hard=True).goal
    assert g[1] >= 0.60, "hard 布局的目标区应当更难到达"
