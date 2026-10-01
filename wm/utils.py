"""通用工具：随机种子、实验目录、指标落盘、rollout 可视化（手写 PNG，无额外依赖）。"""

from __future__ import annotations

import json
import os
import random
import struct
import time
import zlib
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------- 可复现


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- 实验目录


class Experiment:
    """实验目录：<root>/<name>/{config.json,metrics.json,ckpt.pt,figures/}"""

    def __init__(self, root: str, name: str, config: Optional[dict] = None):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "figures").mkdir(exist_ok=True)
        self.metrics_path = self.dir / "metrics.json"
        self.metrics: Dict[str, object] = {"name": name, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                           "history": []}
        if config:
            self.metrics["config"] = config
            (self.dir / "config.json").write_text(
                json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    def log(self, **kw) -> None:
        self.metrics.update(kw)
        self.flush()

    def log_epoch(self, rec: dict) -> None:
        self.metrics["history"].append(rec)
        self.flush()

    def flush(self) -> None:
        self.metrics_path.write_text(
            json.dumps(self.metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    def ckpt(self, name: str = "ckpt.pt") -> Path:
        return self.dir / name

    def fig(self, name: str) -> Path:
        return self.dir / "figures" / name


# ---------------------------------------------------------------- 最小 PNG 写出


def write_png(arr: np.ndarray, path) -> None:
    """[H,W] 或 [H,W,3]，取值 [0,1]。手写实现，避免引入 imageio/Pillow 依赖。"""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        a = (a - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(a)
        a = np.stack([a] * 3, axis=-1)
    a = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8)
    h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def tile(images: Sequence[np.ndarray], cols: int = 4, pad: int = 2) -> np.ndarray:
    """把若干 [H,W,3] 图拼成网格。"""
    imgs = [np.asarray(i, dtype=np.float64) for i in images]
    n = len(imgs)
    rows = (n + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    out = np.ones((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), dtype=np.float64)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        out[y:y + h, x:x + w] = im[:, :, :3] if im.ndim == 3 else np.stack([im] * 3, -1)
    return out


# ---------------------------------------------------------------- 环境信息


def env_info() -> Dict[str, object]:
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
    }
