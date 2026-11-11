"""Smoothness of Stage 7A planner demos: posture deviation and turn per 1 s window.  Usage: python3 demo_stats.py a.npz [b.npz ...]"""
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from rl_env_stage7 import CURL, turn_of  # noqa: E402
from stage7_teacher import World  # noqa: E402

w = World(posture=CURL)
names = [w.m.actuator(i).name for i in range(w.m.nu)]
fing = np.array(["WRJ" not in n for n in names])
span = w.m.actuator_ctrlrange[:, 1] - w.m.actuator_ctrlrange[:, 0]
for f in sys.argv[1:]:
    z = dict(np.load(f))
    p = np.degrees(turn_of(z["balls"], z["center"]))
    dev = np.abs(z["ctrl"] - w.u0)[:, fing] / span[fing]
    pose = dev.sum(1)
    win = p[50::50] - p[:-50:50]
    du = np.abs(np.diff(z["ctrl"], axis=0))[:, fing].sum(1)
    jump = np.abs(np.diff(z["balls"], axis=0)).max() * 100
    print(f"{f}\n  {len(p)} steps = {len(p) * 0.02:.0f} s, turned {p[-1]:.1f} deg, max ball step {jump:.2f} cm")
    print(f"  posture deviation mean {pose.mean():.1f}  p90 {np.percentile(pose, 90):.1f}  max {pose.max():.1f}"
          f";  finger ctrl change per step {du.mean():.3f} rad")
    print(f"  turn per 1 s: p50 {np.median(win):.1f}  p90 {np.percentile(win, 90):.1f}  max {win.max():.1f} deg"
          f";  windows < 1 deg: {np.mean(win < 1) * 100:.0f}%")
    for a, b in ((0, 90), (90, 180), (180, 270), (270, 361)):
        s = (p >= a) & (p < b)
        if s.any():
            print(f"    {a:3d}-{b:3d} deg: {s.sum() * 0.02:5.0f} s  pose {pose[s].mean():.1f}")
