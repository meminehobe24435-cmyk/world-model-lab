# -*- coding: utf-8 -*-
"""训练世界模型（含消融开关）。

用法示例：
    python -m wm.train --data data --out runs --name base
    python -m wm.train --data data --out runs --name no_action --no-action
    python -m wm.train --data data --out runs --name gru --dyn gru
    python -m wm.train --data data --out runs --name vq --vq
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import TrajectoryDataset
from .models.world_model import WMConfig, WorldModel
from .utils import Experiment, env_info, pick_device, set_seed


def build_loaders(args):
    ds = TrajectoryDataset(args.data, tags=("train",), seq_len=args.seq_len,
                           stride=args.stride)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    drop_last=True)
    return ds, dl


@torch.no_grad()
def evaluate(model: WorldModel, dl: DataLoader) -> dict:
    model.eval()
    agg = {}
    n = 0
    for batch in dl:
        _, parts = model.losses(batch)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    return {k: v / max(1, n) for k, v in agg.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default="base")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--img", type=int, default=48)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--dyn", default="transformer", choices=("transformer", "gru"))
    ap.add_argument("--vq", action="store_true", help="用 VQ 码本（tokenizer 口径）")
    ap.add_argument("--no-action", action="store_true", help="消融：把动作输入置零")
    ap.add_argument("--roll-k", type=int, default=4,
                    help="多步潜 rollout 损失步数（0 = 关掉）")
    ap.add_argument("--roll-w", type=float, default=0.5)
    ap.add_argument("--state-w", type=float, default=2.0)
    ap.add_argument("--kl-beta", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    dev = pick_device(args.device)
    exp = Experiment(args.out, args.name, config=vars(args))
    exp.log(env=env_info(), device=str(dev))

    ds, dl = build_loaders(args)
    cfg = WMConfig(img=args.img, z_dim=args.z_dim, base=args.base, dyn=args.dyn,
                   vq=args.vq, n_boxes=ds.n_boxes,
                   n_events=2 * ds.n_boxes + 1, vocab=64, max_lang=ds.instr_len,
                   in_ch=4 + 9, roll_k=args.roll_k, roll_w=args.roll_w,
                   state_w=args.state_w, kl_beta=args.kl_beta)
    model = WorldModel(cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    exp.log(params=n_par, trainable=sum(p.numel() for p in model.parameters()
                                        if p.requires_grad))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print("[train] %s | 参数 %.2fM | 设备 %s | 窗口 %d | 序列 %d"
          % (args.name, n_par / 1e6, dev, len(ds), args.seq_len), flush=True)

    t0 = time.time()
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        agg, n = {}, 0
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            if args.no_action:
                batch["action"] = torch.zeros_like(batch["action"])
            loss, parts = model.losses(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            agg["total"] = agg.get("total", 0.0) + float(loss)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        sched.step()
        rec = {k: round(v / max(1, n), 5) for k, v in agg.items()}
        rec.update(epoch=ep, lr=round(sched.get_last_lr()[0], 6),
                   sec=round(time.time() - t0, 1))
        exp.log_epoch(rec)
        print("[epoch %2d] loss %.4f | rgb %.4f dep %.4f sem %.4f dyn %.4f "
              "roll %.4f state %.4f ev %.4f task %.4f kl %.4f | %.0fs"
              % (ep, rec["total"], rec["rgb"], rec["depth"], rec["sem"], rec["dyn"],
                 rec["roll"], rec["state"], rec["event"], rec["task"], rec["kl"],
                 rec["sec"]),
              flush=True)
        if rec["state"] < best:
            # 按**状态头损失**挑最优权重（而不是总损失）：
            # 总损失被多步 rollout 项的早期尖峰主导，挑出来的不一定状态最准，
            # 而"动力学准确性"是我们对外报的头号指标。
            best = rec["state"]
            torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                        "epoch": ep, "loss": rec["total"], "state": best}, exp.ckpt())
    exp.log(best_loss=best, total_seconds=round(time.time() - t0, 1),
            status="done")
    print("[done] 最优 state loss %.4f -> %s" % (best, exp.ckpt()))


if __name__ == "__main__":
    main()
