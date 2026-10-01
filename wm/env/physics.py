"""确定性桌面推动物理（quasi-static position-based）。

设计取舍（务必如实理解，写进 README）：
    这是一个**简化版可交互 3D 场景**：物体是桌面上的长方体，运动限制在水平面内
    （推动任务），高度只用于渲染排序与遮挡。它**不是完整刚体物理引擎** —— 没有
    转动惯量、没有摩擦锥、没有连续碰撞检测。选它的理由是：
      1) **完全确定性**：同一 seed + 同一动作序列 -> 逐比特一致，反事实数据才可信；
      2) **无条件可复现**：纯 numpy，无编译依赖，CI 上能跑；
      3) **GT 完整**：每一步的真实位置、速度、接触事件、遮挡标志都直接可得。

物理模型：position-based（准静态）
    每步先移动末端执行器；凡与执行器圆盘重叠的方块，沿最短分离方向推出；再做若干轮
    方块间分离迭代；被推出桌面边界的方块记 `off_table` 事件并移除。
    接触强度 = 该步方块位移量，用作"接触事件"的强度标签。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- 常量

TABLE_W = 1.0          # 桌面 x 方向尺寸（米）
TABLE_H = 1.0          # 桌面 y 方向尺寸（米）
MAX_STEP = 0.038       # 单步执行器最大位移（米）
EFFECTOR_R = 0.026     # 执行器圆盘半径（米）
SEPARATION_ITERS = 8   # 方块间分离迭代轮数
CONTACT_EPS = 1e-6


# ---------------------------------------------------------------- 数据结构


@dataclass
class Box:
    """轴对齐长方体。 (cx, cy) 为桌面平面上的中心，z 为**底面**高度。"""

    id: int
    cx: float
    cy: float
    hx: float          # x 半长
    hy: float          # y 半长
    hz: float          # z 半高
    z: float           # 底面高度
    color: Tuple[float, float, float]
    movable: bool = True
    is_target: bool = False

    @property
    def cz(self) -> float:
        return self.z + self.hz


@dataclass
class SceneSpec:
    """一个场景实例的完整定义（布局 + 目标区 + 遮挡墙）。"""

    boxes: List[Box]
    goal: Tuple[float, float, float, float]      # cx, cy, hx, hy
    wall: Optional[Tuple[float, float, float, float, float]]  # cx, cy, hx, hy, hz
    layout_id: int = 0

    @property
    def target(self) -> Box:
        for b in self.boxes:
            if b.is_target:
                return b
        raise ValueError("场景里没有 is_target 的方块")

    @property
    def movable(self) -> List[Box]:
        return [b for b in self.boxes if b.movable]


@dataclass
class StepEvents:
    """单步产生的全部事件（世界模型的监督信号之一）。"""

    effector_contacts: List[Tuple[int, float]] = field(default_factory=list)   # (box_id, 强度)
    box_contacts: List[Tuple[int, int, float]] = field(default_factory=list)   # (id_a, id_b, 强度)
    off_table: List[int] = field(default_factory=list)                        # 掉下桌的 box_id

    def as_vector(self, n_boxes: int) -> np.ndarray:
        """压成定长向量：[每块是否被执行器接触, 每块是否掉下桌, 接触强度总和]。"""
        v = np.zeros(2 * n_boxes + 1, dtype=np.float32)
        for bid, s in self.effector_contacts:
            if 0 <= bid < n_boxes:
                v[bid] = 1.0
                v[2 * n_boxes] += s
        for bid in self.off_table:
            if 0 <= bid < n_boxes:
                v[n_boxes + bid] = 1.0
        return v


# ---------------------------------------------------------------- 几何工具


def _aabb_overlap(a: Box, b: Box) -> Tuple[float, float]:
    """返回两个方块在 xy 平面上的重叠量 (ox, oy)，无重叠则为 0。"""
    ox = (a.hx + b.hx) - abs(a.cx - b.cx)
    oy = (a.hy + b.hy) - abs(a.cy - b.cy)
    if ox > CONTACT_EPS and oy > CONTACT_EPS:
        return max(ox, 0.0), max(oy, 0.0)
    return 0.0, 0.0


def _disc_box_overlap(cx: float, cy: float, r: float, b: Box) -> Tuple[bool, float, float, float]:
    """圆盘与 AABB 的重叠。返回 (是否重叠, 方块移动方向 dx, dy, 推出深度)。

    ★ 方向约定：返回的是**方块应该移动的方向**（即"远离执行器"），不是"从方块指向
      执行器"的方向。本项目真实踩过这个坑 —— 早期版本把方向取反，方块被**推向**
      执行器，表现为接触时方块左右乱跳而不前进，成功率恒为 0；因为 x/y 都有微小
      位移，看起来"像是在动"，极易误判成策略调参问题（我在策略上白调了很多轮）。

    ★ 分支必须按**面法向**判定，不能有"退化兜底随便挑一个轴"。第二个坑就在这里：
      执行器恰好贴在方块某个面上时（`d == 0` 的退化情况），早期实现按 x 偏移挑轴，
      于是执行器从下方推、方块却朝 -x 走。正确做法是把"在哪个方向外侧"算清楚：
          gx = |dx| - hx，gy = |dy| - hy
          gx<=0 且 gy<=0 -> 圆心在方块内 -> 沿最小穿出轴推（排除执行器所在侧）
          gy<=0 且 gx>0  -> 在 x 外侧、y 在跨度内 -> 沿 x 推
          gx<=0 且 gy>0  -> 在 y 外侧、x 在跨度内 -> 沿 y 推
          gx>0  且 gy>0  -> 角点最近 -> 沿角点法向推
      这样任何一步都不会出现"轴选错"，四面推的行为都与直觉一致。
    """
    dx = cx - b.cx
    dy = cy - b.cy
    sx = 1.0 if dx >= 0.0 else -1.0
    sy = 1.0 if dy >= 0.0 else -1.0
    gx = abs(dx) - b.hx
    gy = abs(dy) - b.hy

    # --- 圆心在方块内部：沿穿出代价最小的轴推出（方向远离执行器）
    if gx <= 0.0 and gy <= 0.0:
        if -gx <= -gy:
            return True, -sx, 0.0, (-gx) + r
        return True, 0.0, -sy, (-gy) + r

    # --- 角点区：沿角点法向推
    if gx > 0.0 and gy > 0.0:
        vx = dx - sx * b.hx          # 最近角点 -> 圆心
        vy = dy - sy * b.hy
        d = float(np.hypot(vx, vy))
        if d > r + CONTACT_EPS:
            return False, 0.0, 0.0, 0.0
        if d < 1e-12:                # 圆心恰在角点上：沿对角推
            k = 1.0 / np.sqrt(2.0)
            return True, -sx * k, -sy * k, r
        return True, -vx / d, -vy / d, (r - d)

    # --- 单面区：x 外侧
    if gx > 0.0:
        if gx > r + CONTACT_EPS:
            return False, 0.0, 0.0, 0.0
        return True, -sx, 0.0, (r - gx)

    # --- 单面区：y 外侧
    if gy > r + CONTACT_EPS:
        return False, 0.0, 0.0, 0.0
    return True, 0.0, -sy, (r - gy)


def _separate_pair(a: Box, b: Box) -> float:
    """把两个重叠方块沿最小重叠轴分开（等质量各让一半）。返回分离距离。"""
    ox, oy = _aabb_overlap(a, b)
    if ox <= 0.0 and oy <= 0.0:
        return 0.0
    if ox <= oy:
        s = ox / 2.0
        sign = 1.0 if a.cx >= b.cx else -1.0
        a.cx += sign * s
        b.cx -= sign * s
        return s
    s = oy / 2.0
    sign = 1.0 if a.cy >= b.cy else -1.0
    a.cy += sign * s
    b.cy -= sign * s
    return s


# ---------------------------------------------------------------- 物理世界


class PushPhysics:
    """可交互桌面场景的确定性物理。"""

    def __init__(self, spec: SceneSpec):
        self.table_w = TABLE_W
        self.table_h = TABLE_H
        self._orig = [(b.cx, b.cy, b.z) for b in spec.boxes]
        self.spec = spec
        self.reset()

    # -------------------------------------------------- 生命周期

    def reset(self, scene: Optional[SceneSpec] = None) -> None:
        if scene is not None:
            self.spec = scene
            self._orig = [(b.cx, b.cy, b.z) for b in scene.boxes]
        for b, (cx, cy, z) in zip(self.spec.boxes, self._orig):
            b.cx, b.cy, b.z = cx, cy, z
        self.effector = np.array([self.table_w * 0.5, 0.06], dtype=np.float64)
        self.dropped: List[int] = []
        self.step_index = 0
        self._last_disp: Dict[int, Tuple[float, float]] = {}

    # -------------------------------------------------- 状态读取

    def box_matrix(self) -> np.ndarray:
        """[N, 9] = cx, cy, z, hx, hy, hz, vx, vy, alive。GT 状态向量。"""
        m = np.zeros((len(self.spec.boxes), 9), dtype=np.float32)
        for i, b in enumerate(self.spec.boxes):
            alive = 0.0 if b.id in self.dropped else 1.0
            m[i] = (b.cx, b.cy, b.z, b.hx, b.hy, b.hz,
                    self._last_disp.get(b.id, (0.0, 0.0))[0],
                    self._last_disp.get(b.id, (0.0, 0.0))[1], alive)
        return m

    # -------------------------------------------------- 推进

    def step(self, action: Sequence[float]) -> Tuple[StepEvents, Dict[str, float]]:
        """推进一步。action = (dx, dy) ∈ [-1, 1]，被裁剪到桌面上。"""
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] < 2:
            raise ValueError("action 至少需要 2 个分量 (dx, dy)")
        a = np.clip(a[:2], -1.0, 1.0)
        self.effector = np.clip(
            self.effector + a * MAX_STEP,
            [EFFECTOR_R, EFFECTOR_R],
            [self.table_w - EFFECTOR_R, self.table_h - EFFECTOR_R],
        )
        ev = StepEvents()
        self._last_disp = {}

        # 1) 执行器推方块
        for b in self.spec.movable:
            if b.id in self.dropped:
                continue
            hit, dx, dy, depth = _disc_box_overlap(self.effector[0], self.effector[1],
                                                   EFFECTOR_R, b)
            if not hit or depth <= 0.0:
                continue
            b.cx += dx * depth
            b.cy += dy * depth
            self._last_disp[b.id] = (dx * depth, dy * depth)
            ev.effector_contacts.append((b.id, float(depth)))

        # 2) 方块间分离迭代（位置式，收敛快且确定）
        movers = [b for b in self.spec.movable if b.id not in self.dropped]
        for _ in range(SEPARATION_ITERS):
            moved = 0.0
            for i in range(len(movers)):
                for j in range(i + 1, len(movers)):
                    s = _separate_pair(movers[i], movers[j])
                    if s > CONTACT_EPS:
                        moved += s
                        ev.box_contacts.append((movers[i].id, movers[j].id, float(s)))
            if moved < CONTACT_EPS:
                break

        # 3) 出界 -> 掉下桌
        for b in self.spec.movable:
            if b.id in self.dropped:
                continue
            if (b.cx < -b.hx or b.cx > self.table_w + b.hx
                    or b.cy < -b.hy or b.cy > self.table_h + b.hy):
                self.dropped.append(b.id)
                ev.off_table.append(b.id)

        self.step_index += 1
        return ev, self.task_info()

    # -------------------------------------------------- 任务判定

    def goal_contains(self, b: Box) -> bool:
        gx, gy, ghx, ghy = self.spec.goal
        return (abs(b.cx - gx) <= ghx - b.hx * 0.5) and (abs(b.cy - gy) <= ghy - b.hy * 0.5)

    def task_info(self) -> Dict[str, float]:
        t = self.spec.target
        gx, gy, _, _ = self.spec.goal
        dist = float(np.hypot(t.cx - gx, t.cy - gy))
        success = 1.0 if (t.id not in self.dropped and self.goal_contains(t)) else 0.0
        terminated = 1.0 if (success > 0 or t.id in self.dropped) else 0.0
        return {
            "target_dist": dist,
            "success": success,
            "terminated": terminated,
            "target_alive": 0.0 if t.id in self.dropped else 1.0,
            "n_dropped": float(len(self.dropped)),
        }

    def reward(self, prev_dist: float) -> float:
        info = self.task_info()
        r = (prev_dist - info["target_dist"]) * 10.0
        if info["success"] > 0:
            r += 1.0
        if info["target_alive"] < 1.0:
            r -= 1.0
        return float(r)
