# -*- coding: utf-8 -*-
"""渲染自检：跑一遍场景，存出 RGB / 深度 / 语义 三联图，并打印关键统计。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.env import PushWorld  # noqa: E402
from wm.env.renderer import SEM_EFFECTOR, SEM_GOAL, SEM_TABLE, SEM_WALL  # noqa: E402


def to_png(arr, path):
    """把 [H,W] 或 [H,W,3] 存成 PNG（不依赖 imageio，手写最小 PNG 编码）。"""
    import struct, zlib
    a = np.asarray(arr)
    if a.ndim == 2:
        a = (a - np.nanmin(a)) / max(1e-9, (np.nanmax(a) - np.nanmin(a)))
        a = np.stack([a] * 3, axis=-1)
    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def main():
    w = PushWorld(size=64, horizon=32)
    obs = w.reset(seed=7)
    print("指令:", obs.instruction_text)
    print("RGB", obs.rgb.shape, obs.rgb.dtype, "范围", float(obs.rgb.min()), float(obs.rgb.max()))
    fin = np.isfinite(obs.depth)
    print("深度有效像素 %d / %d" % (int(fin.sum()), obs.depth.size))
    ids, cnt = np.unique(obs.semantic, return_counts=True)
    print("语义分布:", dict(zip(ids.tolist(), cnt.tolist())))
    print("可见比例:", {k: round(v, 3) for k, v in obs.visibility.items()})
    print("点云点数:", w.point_cloud(obs, stride=2).shape)

    # 推 12 步看看动力学是否合理
    for t in range(12):
        obs, r, done, gt = w.step([0.6, 0.9])
        if t % 4 == 3:
            print("  t=%2d 目标距离 %.3f 成功 %.0f 掉桌 %d 事件 %s"
                  % (t, gt["task"]["target_dist"], gt["task"]["success"],
                     gt["task"]["n_dropped"], np.flatnonzero(gt["events"]).tolist()))
    print("可见比例(推动后):", {k: round(v, 3) for k, v in obs.visibility.items()})
    print("被遮挡:", obs.visibility and [k for k, v in obs.visibility.items() if v < 0.05])

    # 检查语义 id 是否都在预期集合里
    unknown = set(ids.tolist()) - ({0, SEM_TABLE, SEM_GOAL, SEM_WALL, SEM_EFFECTOR}
                                   | set(range(1, w.n_boxes + 1)))
    print("未知语义 id:", unknown)

    out = Path(__file__).resolve().parents[1] / "docs"
    out.mkdir(exist_ok=True)
    to_png(obs.rgb, out / "selfcheck_rgb.png")
    d = obs.depth.copy()
    d[~np.isfinite(d)] = np.nan
    dm = np.where(np.isfinite(d), d, np.nanmax(d[np.isfinite(d)]) if np.isfinite(d).any() else 1.0)
    to_png(1.0 - (dm - dm.min()) / max(1e-9, dm.max() - dm.min()), out / "selfcheck_depth.png")
    to_png((obs.semantic % 7) / 6.0, out / "selfcheck_semantic.png")
    print("已写出 docs/selfcheck_*.png")


if __name__ == "__main__":
    main()
