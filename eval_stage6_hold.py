"""Stage 6: can the Stage 3 bowl hold THREE balls, held still for 20 s? Per ball size, three touching
balls placed in a triangle at the bowl's ball centre (raised for the ball size) at several triangle
orientations; a ball counts as lost once it is past the knuckle row (palm-frame z > 10.5cm), more
than 7cm sideways from the trio's starting centre, or 8cm off the palm.

Usage:
    python3 eval_stage6_hold.py --radii 0.0175 0.019 0.02 0.021 0.0225
"""
import argparse

import mujoco as mj
import numpy as np

from hand_env import BALL_RADIUS, BOWL_BALLS_CENTER, bowl_target
from three_balls import THREE_BALL_SCENE, grasp, in_palm, places, set_radius


def hold(radius, angle, seconds=20.0, target=None):
    m = mj.MjModel.from_xml_path(THREE_BALL_SCENE)
    d = mj.MjData(m)
    set_radius(m, radius)
    target = bowl_target() if target is None else target
    center = BOWL_BALLS_CENTER + np.array([0.0, 0.0, radius - BALL_RADIUS])
    grasp(m, d, target, places(center, angle, radius))
    p0 = np.mean(in_palm(m, d), axis=0)
    for k in range(int(seconds / m.opt.timestep)):
        mj.mj_step(m, d)
        if k % 50 == 0:
            for q in in_palm(m, d):
                if q[2] > 0.105 or np.hypot(*(q - p0)[[0, 2]]) > 0.07 or abs(q[1] - p0[1]) > 0.08:
                    return k * m.opt.timestep, in_palm(m, d)
    return None, in_palm(m, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[0.0175, 0.019, 0.02, 0.021, 0.0225])
    ap.add_argument("--angles", type=int, default=6, help="triangle orientations over one 120 deg period")
    args = ap.parse_args()
    for r in args.radii:
        res = [hold(r, a) for a in np.arange(args.angles) * (2 * np.pi / 3) / args.angles]
        lost = [t for t, _ in res if t is not None]
        z = np.array([[q[2] for q in qs] for t, qs in res if t is None])
        print(f"{2000 * r:.0f}mm: held {len(res) - len(lost)}/{len(res)}"
              + (f", lost at {np.round(lost, 1)} s" if lost else "")
              + (f" | held balls along the palm z {np.round(100 * z.min(), 1)}-{np.round(100 * z.max(), 1)} cm"
                 if len(z) else ""), flush=True)


if __name__ == "__main__":
    main()
