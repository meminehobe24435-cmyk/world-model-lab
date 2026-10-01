# -*- coding: utf-8 -*-
"""物理自检：用"朝目标方块走"的脚本策略，确认接触、推动、成功判定都真的работа。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.env import PushWorld  # noqa: E402


def goto_policy(world, obs, tgt_xy, gain=1.6):
    """朝指定 xy 走的比例控制（动作是 [-1,1] 的归一化步长）。"""
    from wm.env.physics import MAX_STEP
    ex, ey = world.physics.effector
    d = np.asarray(tgt_xy, dtype=np.float64) - np.asarray([ex, ey])
    n = float(np.hypot(*d))
    if n < 1e-9:
        return np.zeros(2, dtype=np.float32)
    return np.clip(d / MAX_STEP * gain, -1.0, 1.0).astype(np.float32)


def main():
    ok = 0
    for seed in range(12):
        w = PushWorld(size=64, horizon=48)
        obs = w.reset(seed=seed)
        tb = w.spec.target
        first_contact = None
        total_push = 0.0
        for t in range(48):
            tb = w.spec.target
            aim = (tb.cx, tb.cy)  # 直接撞向目标方块
            a = goto_policy(w, obs, aim)
            prev = (tb.cx, tb.cy)
            obs, r, done, gt = w.step(a)
            moved = float(np.hypot(tb.cx - prev[0], tb.cy - prev[1]))
            total_push += moved
            if first_contact is None and gt["events"][:w.n_boxes].sum() > 0:
                first_contact = t
            if done:
                break
        info = w.physics.task_info()
        ok += int(info["success"] > 0)
        print("seed %2d: 首次接触 t=%s  累计推动 %.3f m  目标距目标区 %.3f  成功 %d  掉桌 %d"
              % (seed, first_contact, total_push, info["target_dist"],
                 info["success"], info["n_dropped"]))
    print("\n成功 %d / 12" % ok)


if __name__ == "__main__":
    main()
