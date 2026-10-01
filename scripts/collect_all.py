# -*- coding: utf-8 -*-
"""一次性生成全部数据集：训练 / 留出布局 / 困难布局 / 反事实配对。"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.data.collector import collect_counterfactual, collect_episode  # noqa: E402
from wm.data.collector import save_dataset  # noqa: E402
from wm.env import PushWorld  # noqa: E402

CF_KEYS = ("rgb0", "prefix_actions", "action_a", "action_b",
           "rgb_a", "rgb_b", "box_a", "box_b")


def gen_split(world, out, tag, seed0, n_expert, n_random, n_perturb, hard=False, H=64):
    eps = []
    for i in range(n_expert):
        eps.append(collect_episode(world, seed0 + i, "expert", horizon=H, hard=hard))
    for i in range(n_random):
        eps.append(collect_episode(world, seed0 + 5000 + i, "random", horizon=H, hard=hard))
    for i in range(n_perturb):
        eps.append(collect_episode(world, seed0 + 9000 + i, "expert", horizon=H,
                                   perturb_sigma=0.45, perturb_period=4))
    files = save_dataset(eps, out, tag)
    succ = {p: int(sum(e.meta["success"] for e in eps if e.meta["policy"] == p))
            for p in ("expert", "random")}
    stats = {"tag": tag, "episodes": len(eps),
             "steps": int(sum(e.meta["n_steps"] for e in eps)),
             "expert_success": succ.get("expert", 0), "n_expert": n_expert,
             "random_success": succ.get("random", 0), "n_random": n_random,
             "shards": files}
    print("[%s] %d 回合 / %d 步  专家 %d/%d  随机 %d/%d -> %s"
          % (tag, stats["episodes"], stats["steps"], stats["expert_success"], n_expert,
             stats["random_success"], n_random, files), flush=True)
    return stats


def gen_counterfactual(world, out, n=160, branch_step=8, suffix=10, seed0=40000):
    acc = {k: [] for k in CF_KEYS}
    skipped = 0
    for i in range(n):
        p = collect_counterfactual(world, seed0 + i, branch_step=branch_step, suffix=suffix)
        if p.prefix_actions.shape[0] != branch_step:
            skipped += 1
            continue
        if p.action_a.shape[0] != suffix or p.action_b.shape[0] != suffix:
            skipped += 1
            continue
        # 扰动必须真的造成了状态差异，否则这一对没有对照价值
        if np.abs(p.box_a[-1, 0, :2] - p.box_b[-1, 0, :2]).sum() < 1e-4:
            skipped += 1
            continue
        for k in CF_KEYS:
            acc[k].append(getattr(p, k))
    path = Path(out) / "counterfactual.npz"
    np.savez_compressed(path, **{k: np.stack(v) for k, v in acc.items()})
    print("[counterfactual] %d 对（跳过 %d）-> %s" % (len(acc["action_a"]), skipped, path),
          flush=True)
    return {"pairs": len(acc["action_a"]), "skipped": skipped, "file": path.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--cf", type=int, default=160)
    args = ap.parse_args()

    sc = args.scale
    w = PushWorld(size=args.size, horizon=args.horizon, n_boxes=4)
    t0 = time.time()
    report = {"size": args.size, "horizon": args.horizon}
    report["train"] = gen_split(w, args.out, "train", 1000,
                                int(110 * sc), int(50 * sc), int(40 * sc), H=args.horizon)
    report["val"] = gen_split(w, args.out, "val", 20000, int(15 * sc), int(8 * sc), 0,
                              H=args.horizon)
    report["hard"] = gen_split(w, args.out, "hard", 30000, int(15 * sc), int(8 * sc), 0,
                               hard=True, H=args.horizon)
    report["counterfactual"] = gen_counterfactual(w, args.out, n=int(args.cf * sc))
    report["seconds"] = round(time.time() - t0, 1)
    Path(args.out, "collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("总耗时 %.1fs" % report["seconds"])


if __name__ == "__main__":
    main()
