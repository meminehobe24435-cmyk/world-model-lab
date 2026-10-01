"""纯 numpy 软件光栅化渲染器：RGB + 深度 + 语义分割 + 点云。

为什么自己写渲染器而不是用 MuJoCo / Isaac Sim：
    - **无条件可复现**：纯 numpy，无编译依赖、无 GPU、无外部资产，CI 上能跑；
    - **GT 完整**：每个像素的语义 id 与相机空间深度都是精确值，不需要标注；
    - **遮挡可量化**：见 `RenderResult.visibility` —— 用"去掉遮挡物再渲一遍"的
      反事实渲染算出每块物体的**可见比例**，这正是"物体持久性"评测需要的量。

渲染管线（与 raster-pipeline-lab 同思路，这里是 numpy 版）：
    世界三角形 -> 视图变换 -> 透视投影 -> 屏幕空间三角形 -> z-buffer 光栅化。
    深度按 1/z 在屏幕空间线性插值（透视正确），逐像素做深度测试。
    着色为逐面 Lambert 漫反射 + 环境光（低多边形风格，确定性）。

语义 id 约定（写进 README）：
    0        背景
    1..N     可移动方块（id 从 1 开始）
    100      遮挡墙
    101      桌面
    102      目标区
    103      末端执行器
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .physics import Box, EFFECTOR_R, SceneSpec, TABLE_H, TABLE_W

SEM_BACKGROUND = 0
SEM_WALL = 100
SEM_TABLE = 101
SEM_GOAL = 102
SEM_EFFECTOR = 103

_LIGHT_DIR = np.array([-0.35, -0.55, 0.75], dtype=np.float64)
_LIGHT_DIR /= np.linalg.norm(_LIGHT_DIR)
_AMBIENT = 0.32


@dataclass
class Camera:
    """针孔相机。eye / target 世界坐标，fov_y 为垂直视场角（度）。"""

    eye: Sequence[float]
    target: Sequence[float]
    fov_y: float = 50.0
    up: Sequence[float] = (0.0, 0.0, 1.0)

    def basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eye = np.asarray(self.eye, dtype=np.float64)
        fwd = np.asarray(self.target, dtype=np.float64) - eye
        fwd = fwd / np.linalg.norm(fwd)
        up = np.asarray(self.up, dtype=np.float64)
        right = np.cross(fwd, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        return eye, right, true_up, fwd


DEFAULT_CAMERA = Camera(eye=(0.50, -0.66, 0.54), target=(0.50, 0.36, 0.02), fov_y=50.0)


@dataclass
class RenderResult:
    rgb: np.ndarray          # [H, W, 3] float32 in [0, 1]
    depth: np.ndarray        # [H, W] float32，相机空间 z；背景为 inf
    semantic: np.ndarray     # [H, W] int16
    visibility: dict         # box_id -> 可见比例 in [0, 1]（相对"无遮挡物"渲染）

    @property
    def occluded_ids(self) -> List[int]:
        return sorted(k for k, v in self.visibility.items() if v < 0.05)


# ---------------------------------------------------------------- 几何构造


def _box_triangles(cx: float, cy: float, z0: float,
                   hx: float, hy: float, hz: float) -> np.ndarray:
    """返回长方体 12 个三角形 [12, 3, 3]（世界坐标）。底面高度 z0。"""
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = z0, z0 + 2.0 * hz
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    tris = []
    for a, b, c, d in quads:
        tris.append([v[a], v[b], v[c]])
        tris.append([v[a], v[c], v[d]])
    return np.array(tris, dtype=np.float64)


def _face_normal(tri: np.ndarray) -> np.ndarray:
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])


def scene_triangles(spec: SceneSpec, effector_xy: Optional[Sequence[float]],
                    include_wall: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """把场景展开成三角形列表与对应语义 id。"""
    tris: List[np.ndarray] = []
    sems: List[int] = []

    # 桌面（很薄的板）
    tris.append(_box_triangles(TABLE_W / 2, TABLE_H / 2, -0.012,
                               TABLE_W / 2, TABLE_H / 2, 0.006))
    sems.extend([SEM_TABLE] * 12)

    # 目标区（贴在桌面上方的薄片）
    gx, gy, ghx, ghy = spec.goal
    tris.append(_box_triangles(gx, gy, 0.0005, ghx, ghy, 0.0012))
    sems.extend([SEM_GOAL] * 12)

    # 遮挡墙
    if include_wall and spec.wall is not None:
        wx, wy, whx, why, whz = spec.wall
        tris.append(_box_triangles(wx, wy, 0.0, whx, why, whz))
        sems.extend([SEM_WALL] * 12)

    # 可移动方块
    for b in spec.boxes:
        if b.id in getattr(spec, "_dropped", ()):  # pragma: no cover - 防御性
            continue
        tris.append(_box_triangles(b.cx, b.cy, b.z, b.hx, b.hy, b.hz))
        sems.extend([b.id] * 12)

    # 末端执行器（矮圆柱用扁方盒近似）
    if effector_xy is not None:
        tris.append(_box_triangles(effector_xy[0], effector_xy[1], 0.0,
                                   EFFECTOR_R, EFFECTOR_R, 0.010))
        sems.extend([SEM_EFFECTOR] * 12)

    return np.concatenate(tris, axis=0), np.asarray(sems, dtype=np.int16)


# ---------------------------------------------------------------- 光栅化


def _project(tris: np.ndarray, cam: Camera, H: int, W: int):
    """世界 -> 相机 -> 屏幕。返回 (screen_uv [T,3,2], invz [T,3], cam_z [T,3], valid [T])。"""
    eye, right, up, fwd = cam.basis()
    rel = tris - eye
    xc = rel @ right
    yc = rel @ up
    zc = rel @ fwd
    f = 0.5 * H / np.tan(np.deg2rad(cam.fov_y) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        invz = np.where(np.abs(zc) > 1e-9, 1.0 / zc, 0.0)
    u = W / 2.0 + f * xc * invz
    v = H / 2.0 - f * yc * invz
    screen = np.stack([u, v], axis=-1)
    # 三角形至少有一个顶点在相机前方才有效
    valid = np.all(zc > 1e-6, axis=1)
    return screen, invz, zc, valid


def render(spec: SceneSpec, camera: Camera = DEFAULT_CAMERA,
           size: int = 64, effector_xy: Optional[Sequence[float]] = None,
           compute_visibility: bool = True) -> RenderResult:
    """渲染一帧。`compute_visibility=True` 时额外渲一遍"无遮挡墙"的场景，
    用两帧语义图的像素计数比得到每块物体的**可见比例**（物体持久性 GT）。"""
    H = W = size
    rgb, depth, sem = _rasterize(spec, camera, H, W, effector_xy, include_wall=True)

    if not compute_visibility:
        return RenderResult(rgb, depth, sem, {})

    _, _, sem_nowall = _rasterize(spec, camera, H, W, effector_xy, include_wall=False)
    visibility = {}
    for b in spec.boxes:
        full = int(np.count_nonzero(sem == b.id))
        free = int(np.count_nonzero(sem_nowall == b.id))
        visibility[b.id] = 1.0 if free == 0 else float(min(1.0, full / free))
    return RenderResult(rgb, depth, sem, visibility)


def _rasterize(spec: SceneSpec, camera: Camera, H: int, W: int,
               effector_xy: Optional[Sequence[float]],
               include_wall: bool):
    tris, sems = scene_triangles(spec, effector_xy, include_wall=include_wall)
    screen, invz, _, valid = _project(tris, camera, H, W)

    depth_buf = np.full((H, W), np.inf, dtype=np.float64)
    invz_buf = np.zeros((H, W), dtype=np.float64)
    sem_buf = np.zeros((H, W), dtype=np.int16)
    rgb_buf = np.zeros((H, W, 3), dtype=np.float64)

    order = np.argsort(-_tri_depth(tris, camera))  # 远 -> 近；z-buffer 保证正确性
    for ti in order:
        if not valid[ti]:
            continue
        tri = screen[ti]
        umin = max(int(np.floor(tri[:, 0].min())), 0)
        umax = min(int(np.ceil(tri[:, 0].max())), W - 1)
        vmin = max(int(np.floor(tri[:, 1].min())), 0)
        vmax = min(int(np.ceil(tri[:, 1].max())), H - 1)
        if umax < umin or vmax < vmin:
            continue

        uu, vv = np.meshgrid(np.arange(umin, umax + 1) + 0.5,
                             np.arange(vmin, vmax + 1) + 0.5)
        p = np.stack([uu, vv], axis=-1)
        b0, b1, b2 = tri[0], tri[1], tri[2]
        area = (b1[0] - b0[0]) * (b2[1] - b0[1]) - (b2[0] - b0[0]) * (b1[1] - b0[1])
        if abs(area) < 1e-12:
            continue
        w0 = ((b1[0] - p[..., 0]) * (b2[1] - p[..., 1])
              - (b2[0] - p[..., 0]) * (b1[1] - p[..., 1])) / area
        w1 = ((b2[0] - p[..., 0]) * (b0[1] - p[..., 1])
              - (b0[0] - p[..., 0]) * (b2[1] - p[..., 1])) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue

        iz = w0 * invz[ti, 0] + w1 * invz[ti, 1] + w2 * invz[ti, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(iz > 1e-12, 1.0 / iz, np.inf)

        sub_d = depth_buf[vmin:vmax + 1, umin:umax + 1]
        closer = inside & (z < sub_d)
        if not closer.any():
            continue

        # 逐面 Lambert 着色
        n = _face_normal(tris[ti])
        lam = max(float(np.dot(n, _LIGHT_DIR)), 0.0)
        shade = _AMBIENT + (1.0 - _AMBIENT) * lam
        col = _face_color(sems[ti], spec) * shade

        sub_d[closer] = z[closer]
        invz_buf[vmin:vmax + 1, umin:umax + 1][closer] = iz[closer]
        sem_buf[vmin:vmax + 1, umin:umax + 1][closer] = sems[ti]
        rgb_buf[vmin:vmax + 1, umin:umax + 1][closer] = col

    rgb_out = np.clip(rgb_buf, 0.0, 1.0).astype(np.float32)
    depth_out = depth_buf.astype(np.float32)
    return rgb_out, depth_out, sem_buf


def _tri_depth(tris: np.ndarray, camera: Camera) -> np.ndarray:
    eye, _, _, fwd = camera.basis()
    return ((tris.mean(axis=1) - eye) @ fwd)


_PALETTE = [
    (0.85, 0.25, 0.25), (0.25, 0.55, 0.90), (0.30, 0.80, 0.40),
    (0.92, 0.78, 0.22), (0.70, 0.40, 0.85), (0.25, 0.80, 0.80),
]


def _face_color(sem: int, spec: SceneSpec) -> np.ndarray:
    if sem == SEM_TABLE:
        return np.array([0.78, 0.76, 0.72])
    if sem == SEM_GOAL:
        return np.array([0.20, 0.62, 0.30])
    if sem == SEM_WALL:
        return np.array([0.55, 0.53, 0.58])
    if sem == SEM_EFFECTOR:
        return np.array([0.16, 0.16, 0.20])
    for b in spec.boxes:
        if b.id == sem:
            return np.array(b.color, dtype=np.float64)
    return np.array([0.5, 0.5, 0.5])


# ---------------------------------------------------------------- 点云


def unproject(depth: np.ndarray, camera: Camera = DEFAULT_CAMERA,
              stride: int = 1) -> np.ndarray:
    """深度图 -> 世界坐标点云 [M, 3]。背景（inf）像素被丢弃。"""
    H, W = depth.shape
    eye, right, up, fwd = camera.basis()
    f = 0.5 * H / np.tan(np.deg2rad(camera.fov_y) / 2.0)
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[0:H:stride, 0:W:stride]
    m = np.isfinite(z)
    if not m.any():
        return np.zeros((0, 3), dtype=np.float32)
    u = us[m] + 0.5
    v = vs[m] + 0.5
    zz = z[m]
    xc = (u - W / 2.0) / f * zz
    yc = (H / 2.0 - v) / f * zz
    pts = (eye[None, :] + xc[:, None] * right[None, :]
           + yc[:, None] * up[None, :] + zz[:, None] * fwd[None, :])
    return pts.astype(np.float32)
