"""可交互桌面场景环境：物理 + 渲染 + 语言指令 + 完整 GT。

任务：把**目标方块（红色）**推到**绿色目标区**。
动作：连续二维推动 (dx, dy) ∈ [-1, 1]^2（末端执行器在桌面平面内移动）。

每步对外给出：
    - 多模态观测：RGB / 深度 / 语义分割 / 本体状态 / 语言指令
    - 完整 GT（训练监督 + 评测真值）：方块状态矩阵、接触事件、成功 / 终止 / 奖励、可见比例

★ 明确边界（不许对外含糊）：语言指令是**模板生成的短句**，用字符级词表 + 嵌入做条件，
  不是 LLM 指令理解；场景是**简化版**可交互 3D 场景（平面推动 + 高度排序），不是完整刚体仿真。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .physics import (Box, EFFECTOR_R, PushPhysics, SceneSpec, StepEvents,
                      TABLE_H, TABLE_W)
from .renderer import (DEFAULT_CAMERA, Camera, RenderResult, SEM_EFFECTOR,
                       render, unproject)

# ---------------------------------------------------------------- 语言指令

COLOR_NAMES = {
    (0.85, 0.25, 0.25): "红色",
    (0.25, 0.55, 0.90): "蓝色",
    (0.30, 0.80, 0.40): "绿色",
    (0.92, 0.78, 0.22): "黄色",
    (0.70, 0.40, 0.85): "紫色",
    (0.25, 0.80, 0.80): "青色",
}

TEMPLATES = [
    "把{color}方块推到目标区",
    "请将{color}的方块移动到绿色区域",
    "帮我推动{color}方块到指定位置",
    "把{color}方块推入绿色目标区里",
    "任务：将{color}方块推到目标区",
]


def make_instruction(color_name: str, variant: int) -> str:
    return TEMPLATES[variant % len(TEMPLATES)].format(color=color_name)


def build_vocab() -> List[str]:
    chars = set()
    for t in TEMPLATES:
        chars.update(t.replace("{color}", ""))
    for c in COLOR_NAMES.values():
        chars.update(c)
    return ["<pad>"] + sorted(chars)


VOCAB = build_vocab()
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}


def encode_instruction(text: str, max_len: int = 24) -> np.ndarray:
    ids = [CHAR2ID.get(c, 0) for c in text][:max_len]
    ids += [0] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int64)


# ---------------------------------------------------------------- 场景随机化

_PALETTE = [(0.85, 0.25, 0.25), (0.25, 0.55, 0.90), (0.30, 0.80, 0.40),
            (0.92, 0.78, 0.22), (0.70, 0.40, 0.85), (0.25, 0.80, 0.80)]


def make_scene(seed: int, n_boxes: int = 4, hard: bool = False,
               wall: bool = True) -> SceneSpec:
    """生成一个随机布局。`hard=True` 时目标区和遮挡墙更刁钻（用于泛化评测）。"""
    rng = np.random.default_rng(seed)
    boxes: List[Box] = []
    tries = 0
    while len(boxes) < n_boxes and tries < 500:
        tries += 1
        hx = float(rng.uniform(0.046, 0.078))
        hy = float(rng.uniform(0.046, 0.078))
        hz = float(rng.uniform(0.028, 0.046))
        cx = float(rng.uniform(0.09 + hx, 0.91 - hx))
        cy = float(rng.uniform(0.24 + hy, 0.92 - hy))
        if any(abs(cx - b.cx) < hx + b.hx + 0.02 and abs(cy - b.cy) < hy + b.hy + 0.02
               for b in boxes):
            continue
        boxes.append(Box(id=len(boxes) + 1, cx=cx, cy=cy, hx=hx, hy=hy, hz=hz,
                         z=0.0, color=_PALETTE[len(boxes) % len(_PALETTE)],
                         movable=True, is_target=(len(boxes) == 0)))

    target = boxes[0]
    for _ in range(200):
        gx = float(rng.uniform(0.16, 0.84))
        gy = float(rng.uniform(0.10 if not hard else 0.64, 0.90))
        if np.hypot(gx - target.cx, gy - target.cy) > 0.28:
            break
    goal = (gx, gy, 0.105, 0.082)

    wall_spec = None
    if wall:
        wall_spec = (float(rng.uniform(0.42, 0.58)), 0.62, 0.30, 0.016, 0.095)
    return SceneSpec(boxes=boxes, goal=goal, wall=wall_spec, layout_id=seed)


# ---------------------------------------------------------------- 观测


@dataclass
class Observation:
    rgb: np.ndarray            # [H, W, 3] float32
    depth: np.ndarray          # [H, W] float32（相机空间 z，背景 inf）
    semantic: np.ndarray       # [H, W] int16
    proprio: np.ndarray        # [6] float32：执行器 xy、上一步动作 xy、步数占比、剩余占比
    instruction: np.ndarray    # [L] int64 字符 id
    instruction_text: str
    visibility: Dict[int, float]

    @property
    def depth_masked(self) -> np.ndarray:
        """把背景的 inf 换成 0，便于直接网络输入。"""
        d = self.depth.copy()
        d[~np.isfinite(d)] = 0.0
        return d


@dataclass
class Snapshot:
    """世界完整状态快照 —— 反事实分支就靠"从同一快照分叉"。"""

    boxes: List[Tuple[float, float, float]]
    effector: np.ndarray
    dropped: List[int]
    step_index: int


class PushWorld:
    """可交互 3D 桌面场景（简化刚体）+ 多模态观测 + 完整 GT。"""

    def __init__(self, size: int = 64, horizon: int = 32,
                 camera: Camera = DEFAULT_CAMERA, wall: bool = True,
                 n_boxes: int = 4):
        self.size = size
        self.horizon = horizon
        self.camera = camera
        self.use_wall = wall
        self.n_boxes_cfg = n_boxes
        self._seed = 0
        self._spec: Optional[SceneSpec] = None
        self.physics: Optional[PushPhysics] = None
        self._last_action = np.zeros(2, dtype=np.float32)
        self._instruction_text = ""

    # -------------------------------------------------- 生命周期

    def reset(self, seed: int = 0, hard: bool = False) -> Observation:
        self._seed = int(seed)
        self._spec = make_scene(self._seed, n_boxes=self.n_boxes_cfg,
                                hard=hard, wall=self.use_wall)
        self.physics = PushPhysics(self._spec)
        self._last_action = np.zeros(2, dtype=np.float32)

        # 执行器起始位置：站在目标方块**背对目标区**的一侧后方。
        # 理由：推动类操纵回合本来就从"末端接近物体"开始；从固定角落出发会让脚本
        # 专家把整个 horizon 耗在赶路上（实测成功率被压到 0），而世界模型要学的是
        # 动力学而不是导航。起始点随场景变化，初始状态多样性仍然保留。
        tb = self._spec.target
        gx, gy, _, _ = self._spec.goal
        dvec = np.array([gx - tb.cx, gy - tb.cy], dtype=np.float64)
        nvec = float(np.hypot(*dvec))
        u = dvec / nvec if nvec > 1e-9 else np.array([1.0, 0.0])
        standoff = EFFECTOR_R + max(tb.hx, tb.hy) + 0.20
        start = np.array([tb.cx, tb.cy]) - u * standoff
        self.physics.effector = np.clip(
            start, [EFFECTOR_R + 0.02, EFFECTOR_R + 0.02],
            [TABLE_W - EFFECTOR_R - 0.02, TABLE_H - EFFECTOR_R - 0.02])

        color = COLOR_NAMES.get(tuple(np.round(self._spec.target.color, 2)), "目标")
        self._instruction_text = make_instruction(color, self._seed % len(TEMPLATES))
        return self._observe()

    def step(self, action: Sequence[float]):
        assert self.physics is not None, "请先 reset"
        prev_dist = self.physics.task_info()["target_dist"]
        self._last_action = np.clip(np.asarray(action, dtype=np.float32)[:2], -1.0, 1.0)
        events, info = self.physics.step(action)
        reward = self.physics.reward(prev_dist)
        done = bool(info["terminated"] > 0 or self.physics.step_index >= self.horizon)
        obs = self._observe()
        gt = {
            "box_matrix": self.physics.box_matrix(),
            "events": events.as_vector(len(self._spec.boxes)),
            "events_raw": events,
            "task": info,
            "reward": reward,
        }
        return obs, reward, done, gt

    # -------------------------------------------------- 观测组装

    def _render_spec(self) -> SceneSpec:
        assert self._spec is not None and self.physics is not None
        alive = [b for b in self._spec.boxes if b.id not in self.physics.dropped]
        return SceneSpec(boxes=alive, goal=self._spec.goal, wall=self._spec.wall,
                         layout_id=self._spec.layout_id)

    def _observe(self) -> Observation:
        assert self.physics is not None
        res: RenderResult = render(self._render_spec(), self.camera, self.size,
                                   effector_xy=self.physics.effector)
        frac = self.physics.step_index / max(1, self.horizon)
        proprio = np.asarray([
            self.physics.effector[0], self.physics.effector[1],
            self._last_action[0], self._last_action[1],
            frac, 1.0 - frac,
        ], dtype=np.float32)
        return Observation(
            rgb=res.rgb, depth=res.depth, semantic=res.semantic,
            proprio=proprio,
            instruction=encode_instruction(self._instruction_text),
            instruction_text=self._instruction_text,
            visibility=res.visibility,
        )

    def point_cloud(self, obs: Observation, stride: int = 2) -> np.ndarray:
        return unproject(obs.depth, self.camera, stride=stride)

    # -------------------------------------------------- 反事实快照

    def snapshot(self) -> Snapshot:
        assert self._spec is not None and self.physics is not None
        return Snapshot(
            boxes=[(b.cx, b.cy, b.z) for b in self._spec.boxes],
            effector=self.physics.effector.copy(),
            dropped=list(self.physics.dropped),
            step_index=self.physics.step_index,
        )

    def restore(self, snap: Snapshot) -> Observation:
        assert self._spec is not None and self.physics is not None
        for b, (cx, cy, z) in zip(self._spec.boxes, snap.boxes):
            b.cx, b.cy, b.z = cx, cy, z
        self.physics.effector = snap.effector.copy()
        self.physics.dropped = list(snap.dropped)
        self.physics.step_index = snap.step_index
        return self._observe()

    # -------------------------------------------------- 便捷属性

    @property
    def n_boxes(self) -> int:
        assert self._spec is not None
        return len(self._spec.boxes)

    @property
    def spec(self) -> SceneSpec:
        assert self._spec is not None
        return self._spec
