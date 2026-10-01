# -*- coding: utf-8 -*-
"""世界模型评测主程序：一次跑完所有维度并落盘 benchmark.json + 图。

用法：
    python -m wm.bench.run_benchmark --ckpt runs/base/ckpt.pt --data data \
        --out runs/base --tag val
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import CounterfactualDataset, TrajectoryDataset
from ..models.world_model import WMConfig, WorldModel
from ..utils import pick_device, set_seed, tile, write_png
from . import metrics as M
from .planning import PlanConfig, run_mpc


def load_model(ckpt: str, device: torch.device) -> WorldModel:
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = WMConfig(**blob["cfg"])
    model = WorldModel(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    return model


def rollout_figure(rolls, path, n_show: int = 6, cols: int = 6) -> None:
    """GT 与预测的 RGB / 深度对照图。"""
    if not rolls:
        return
    r = rolls[0]
    T = min(n_show, r["rgb_pred"].shape[0])
    rows = []
    rows.append(tile([np.clip(r["rgb_pred"][t].transpose(1, 2, 0), 0, 1) for t in range(T)], cols))
    sem = r["sem_gt"]
    gt_rgb = np.zeros_like(r["rgb_pred"])
    for t in range(T):
        gt_rgb[t] = np.stack([(sem[t] == i).astype(np.float32) for i in (101, 1, 102)], axis=0)
    rows.append(tile([np.clip(gt_rgb[t].transpose(1, 2, 0), 0, 1) for t in range(T)], cols))
    rows.append(tile([1.0 - np.clip(r["depth_pred"][t, 0], 0, 1) for t in range(T)], cols))
    rows.append(tile([1.0 - np.clip(r["depth_gt"][t, 0], 0, 1) for t in range(T)], cols))
    h = sum(x.shape[0] for x in rows)
    w = max(x.shape[1] for x in rows)
    canvas = np.ones((h, w, 3))
    y = 0
    for x in rows:
        canvas[y:y + x.shape[0], :x.shape[1]] = x
        y += x.shape[0]
    write_png(canvas, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None, help="结果输出目录（默认与 ckpt 同目录）")
    ap.add_argument("--tag", default="val", help="评测分片：val | hard")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--cf-pairs", type=int, default=48)
    ap.add_argument("--mpc-episodes", type=int, default=16)
    ap.add_argument("--mpc-samples", type=int, default=96)
    ap.add_argument("--mpc-horizon", type=int, default=6)
    ap.add_argument("--mpc-rounds", type=int, default=2)
    ap.add_argument("--mpc-max-steps", type=int, default=32)
    ap.add_argument("--skip-mpc", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    device = pick_device(args.device)
    model = load_model(args.ckpt, device)
    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    ds = TrajectoryDataset(args.data, tags=(args.tag,), seq_len=4, stride=4, verbose=True)
    rolls = M.collect_rollouts(model, ds, horizon=args.horizon,
                              max_episodes=args.episodes, device=device)
    summary = M.summarize(rolls)
    summary["n_rollouts"] = len(rolls)

    cf_path = Path(args.data) / "counterfactual.npz"
    if cf_path.exists():
        summary["action_controllability"] = M.action_controllability(
            model, str(cf_path), device, max_pairs=args.cf_pairs)
    rollout_figure(rolls, out_dir / ("rollout_%s.png" % args.tag))

    if not args.skip_mpc:
        pc = PlanConfig(n_samples=args.mpc_samples, horizon=args.mpc_horizon,
                        n_rounds=args.mpc_rounds, seed=args.seed,
                        max_steps=args.mpc_max_steps)
        seeds = list(range(500, 500 + args.mpc_episodes))
        summary["downstream_mpc"] = {
            "mpc": run_mpc(model, seeds, device, pc, "mpc"),
            "random": run_mpc(model, seeds, device, pc, "random"),
            "scripted": run_mpc(model, seeds, device, pc, "scripted"),
        }

    summary["tag"] = args.tag
    summary["seconds"] = round(time.time() - t0, 1)
    (out_dir / ("benchmark_%s.json" % args.tag)).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "err_curve"},
                     ensure_ascii=False, indent=2))
    print("\n-> %s" % (out_dir / ("benchmark_%s.json" % args.tag)))


if __name__ == "__main__":
    main()
