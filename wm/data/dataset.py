"""数据集：加载 npz 分片、切定长序列、把语义图转成多通道掩码。

多模态输入拼装（对应 JD「融合 RGB、深度、点云、语义、机器人本体状态、动作、语言指令」）：
    - RGB       3 通道，归一化到 [0,1]
    - 深度       1 通道，按固定尺度裁剪归一化
    - 语义       9 通道 one-hot（背景 / 4 个方块 / 遮挡墙 / 桌面 / 目标区 / 执行器）
    - 点云       由深度反投影得到，评测"空间一致性"时用；训练时以深度 + 语义等价表达
    - 本体状态   6 维（执行器 xy、上一步动作 xy、步数占比、剩余占比）
    - 语言指令   字符 id 序列，模型侧做嵌入 + 平均池化
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# 语义 id -> one-hot 通道
SEM_IDS = (0, 1, 2, 3, 4, 100, 101, 102, 103)
SEM_TO_CH = {v: i for i, v in enumerate(SEM_IDS)}
N_SEM_CH = len(SEM_IDS)
DEPTH_SCALE = 1.2      # 相机空间 z 的归一化上限（米）


def sem_to_onehot(sem: np.ndarray) -> np.ndarray:
    """[T,H,W] uint8 -> [T,C,H,W] float32"""
    T, H, W = sem.shape
    out = np.zeros((T, N_SEM_CH, H, W), dtype=np.float32)
    for sid, ch in SEM_TO_CH.items():
        out[:, ch] = (sem == sid).astype(np.float32)
    return out


class TrajectoryDataset(Dataset):
    """按定长窗口切轨迹。每个样本是一条 seq_len 步的窗口。"""

    def __init__(self, data_dir: str, tags: Sequence[str] = ("train",),
                 seq_len: int = 8, stride: int = 1, limit_shards: Optional[int] = None,
                 verbose: bool = True):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.arrays: Dict[str, np.ndarray] = {}
        self.ep_index: List[Tuple[int, int, int]] = []   # (episode_id, start, length)

        files: List[Path] = []
        for tag in tags:
            files.extend(sorted(self.data_dir.glob("%s_shard*.npz" % tag)))
        if limit_shards:
            files = files[:limit_shards]
        if not files:
            raise FileNotFoundError("在 %s 没找到 %s 的分片" % (self.data_dir, list(tags)))

        chunks: Dict[str, List[np.ndarray]] = {}
        ep_id = 0
        for f in files:
            with np.load(f) as z:
                lens = z["ep_len"]
                for k in z.files:
                    if k in ("ep_len", "instruction"):
                        continue
                    chunks.setdefault(k, []).append(z[k])
                chunks.setdefault("__instr__", []).append(z["instruction"])
                off = 0
                for L in lens:
                    L = int(L)
                    if L >= seq_len:
                        self.ep_index.append((ep_id, off, L))
                    off += L
                    ep_id += 1
        for k, v in chunks.items():
            self.arrays[k] = np.concatenate(v, axis=0)
        # 指令是"每回合一条"，按回合 id 取，不能按步偏移取
        self.instr = self.arrays.pop("__instr__")
        self.n_steps = int(self.arrays["action"].shape[0])
        self.n_boxes = int(self.arrays["box_matrix"].shape[1])
        self.instr_len = int(self.instr.shape[1])
        self.starts: List[Tuple[int, int]] = []
        for eid, off, L in self.ep_index:
            for s in range(0, L - seq_len + 1, stride):
                self.starts.append((eid, off + s))
        if verbose:
            print("[dataset] %s 分片 %d 个 / 轨迹 %d 条 / %d 步 / 窗口 %d 个 (seq_len=%d)"
                  % (list(tags), len(files), len(self.ep_index), self.n_steps,
                     len(self.starts), seq_len))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        eid, off = self.starts[i]
        T = self.seq_len
        sl = slice(off, off + T)
        rgb = self.arrays["rgb"][sl].astype(np.float32) / 255.0
        depth = self.arrays["depth"][sl].astype(np.float32)
        depth = np.clip(depth / DEPTH_SCALE, 0.0, 1.0)
        sem = self.arrays["semantic"][sl]
        vis = self.arrays["visibility"][sl]
        mask = sem_to_onehot(sem)
        x = np.concatenate([rgb.transpose(0, 3, 1, 2), depth[:, None], mask], axis=1)

        return {
            "x": torch.from_numpy(x),                                    # [T,C,H,W]
            "depth": torch.from_numpy(depth[:, None]),                   # [T,1,H,W]
            "sem": torch.from_numpy(sem.astype(np.int64)),               # [T,H,W]
            "proprio": torch.from_numpy(self.arrays["proprio"][sl]),
            "instr": torch.from_numpy(self.instr[eid].astype(np.int64)),  # [L]
            "action": torch.from_numpy(self.arrays["action"][sl]),
            "box": torch.from_numpy(self.arrays["box_matrix"][sl]),      # [T,N,9]
            "events": torch.from_numpy(self.arrays["events"][sl]),
            "task": torch.from_numpy(self.arrays["task"][sl]),           # [T,4]
            "visibility": torch.from_numpy(vis),                         # [T,N]
        }

    # -------------------------------------------------- 便捷读取

    def episode_batch(self, eid: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        e_id, off, L = self.ep_index[eid]
        sl = slice(off, off + L)
        rgb = self.arrays["rgb"][sl].astype(np.float32) / 255.0
        depth = np.clip(self.arrays["depth"][sl].astype(np.float32) / DEPTH_SCALE, 0, 1)
        x = np.concatenate([rgb.transpose(0, 3, 1, 2), depth[:, None],
                            sem_to_onehot(self.arrays["semantic"][sl])], axis=1)
        return {
            "x": torch.from_numpy(x).to(device),
            "depth": torch.from_numpy(depth[:, None]).to(device),
            "sem": torch.from_numpy(self.arrays["semantic"][sl].astype(np.int64)).to(device),
            "proprio": torch.from_numpy(self.arrays["proprio"][sl]).to(device),
            "instr": torch.from_numpy(self.instr[e_id].astype(np.int64)).to(device),
            "action": torch.from_numpy(self.arrays["action"][sl]).to(device),
            "box": torch.from_numpy(self.arrays["box_matrix"][sl]).to(device),
            "events": torch.from_numpy(self.arrays["events"][sl]).to(device),
            "task": torch.from_numpy(self.arrays["task"][sl]).to(device),
            "visibility": torch.from_numpy(self.arrays["visibility"][sl]).to(device),
        }


class CounterfactualDataset:
    """反事实配对：同一前缀 + 两条不同动作后缀，用于动作可控性评测。"""

    def __init__(self, path: str):
        z = np.load(path)
        self.rgb_a = z["rgb_a"]
        self.rgb_b = z["rgb_b"]
        self.box_a = z["box_a"]
        self.box_b = z["box_b"]
        self.action_a = z["action_a"]
        self.action_b = z["action_b"]
        self.prefix_actions = z["prefix_actions"]

    def __len__(self) -> int:
        return int(self.rgb_a.shape[0])


def load_manifest(data_dir: str, tag: str) -> Optional[dict]:
    p = Path(data_dir) / ("%s_manifest.json" % tag)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
