# -*- coding: utf-8 -*-
"""跑三个消融并各自评测，结果汇总到 runs/ablations.json。

    no_action : 把动作输入置零   -> 证明"动作条件"确实在起作用
    gru       : GRU 代替 Transformer -> 证明序列建模结构的贡献
    vq        : VQ 码本代替连续 VAE  -> 离散 tokenizer 口径的对照
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15

CASES = [
    ("no_action", ["--no-action"]),
    ("gru", ["--dyn", "gru"]),
    ("vq", ["--vq"]),
]

out = {}
for name, extra in CASES:
    print("=" * 70, flush=True)
    print("[ablation] %s %s" % (name, extra), flush=True)
    cmd = [PY, "-m", "wm.train", "--data", "data", "--out", "runs",
           "--name", name, "--epochs", str(EPOCHS), "--batch", "16",
           "--seq-len", "8", "--stride", "8", "--z-dim", "96",
           "--kl-beta", "0.01", "--roll-w", "0.5", "--state-w", "10.0"] + extra
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        out[name] = {"error": "train failed rc=%d" % r.returncode}
        continue

    cmd = [PY, "-m", "wm.bench.run_benchmark", "--ckpt", "runs/%s/ckpt.pt" % name,
           "--data", "data", "--out", "runs/%s" % name, "--tag", "val",
           "--horizon", "16", "--episodes", "20", "--cf-pairs", "40", "--skip-mpc"]
    subprocess.run(cmd, cwd=str(ROOT))
    p = ROOT / "runs" / name / "benchmark_val.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        hist = json.loads((ROOT / "runs" / name / "metrics.json").read_text(encoding="utf-8"))
        last = hist["history"][-1] if hist.get("history") else {}
        out[name] = {
            "ade": d.get("ade"), "fde": d.get("fde"),
            "depth_mae": d.get("depth_mae"),
            "occupancy_iou": d.get("occupancy_iou"),
            "pos_err_t16": d.get("pos_err_t16"),
            "occlusion_penalty": d.get("occlusion_penalty"),
            "action_controllability": d.get("action_controllability"),
            "train_last": {k: last.get(k) for k in ("total", "dyn", "roll", "state", "rgb")},
        }
    print("[ablation] %s -> %s" % (name, json.dumps(out.get(name, {}), ensure_ascii=False)),
          flush=True)

(ROOT / "runs" / "ablations.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\n汇总 -> runs/ablations.json", flush=True)
