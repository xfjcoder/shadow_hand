"""Stage 6 (rebuild, step 2): sweep the circle's phase through the held postures of step 1.

The finger targets follow phase(t) = phase0 + sign * rate * t, interpolated (periodic, 120 deg) between
the CEM postures of `eval_stage6_circle_hold.py --save`; nothing holds the balls. Prints how far the
trio actually turned (mean ball angle round its centroid, unwrapped) against the command, the worst
ball's distance from its moving target and when a ball was lost.

Usage:
    python3 eval_stage6_posture_sweep.py data/stage6_circle_hold.npz --center 0.1,6.5 --rate 20 --seconds 20
"""
import argparse

import mujoco as mj
import numpy as np

from eval_stage6_circle_hold import WRIST, Probe

PERIOD = 2 * np.pi / 3


def table(path, center):
    z = np.load(path)
    sel = np.all(np.isclose(z["centers"], center), axis=1)
    order = np.argsort(z["phases"][sel])
    return z["phases"][sel][order], z["fingers"][sel][order]


def posture(phases, fingers, phase):
    """Periodic linear interpolation of the finger table at `phase`."""
    u = (phase % PERIOD) / PERIOD * len(phases)
    i = int(np.floor(u)) % len(phases)
    a = u - np.floor(u)
    return (1 - a) * fingers[i] + a * fingers[(i + 1) % len(phases)]


def sweep(pr, phases, fingers, center, sign, rate, seconds, frames=None):
    """-> commanded and actual turn (rad), worst tracking error (cm), lost time (s or None)."""
    c = np.array(center) / 100
    pr.rollout(fingers[0], pr.places(c, 0.0), hold=1.0)
    def angles():
        b = np.array(pr.palm())
        g = b.mean(0)
        return np.arctan2(b[:, 0] - g[0], b[:, 2] - g[2])
    a0, prev, turn = angles(), angles(), 0.0
    dt = pr.m.opt.timestep
    worst, lost = 0.0, None
    for k in range(int(seconds / dt)):
        ph = sign * rate * k * dt
        pr.d.ctrl[WRIST:] = np.clip(posture(phases, fingers, ph), pr.lo[WRIST:], pr.hi[WRIST:])
        mj.mj_step(pr.m, pr.d)
        if k % 25 == 24:
            a = angles()
            turn += np.mean(np.angle(np.exp(1j * (a - prev))))
            prev = a
            tg = pr.places(c, ph)
            b = pr.palm()
            # nearest-target assignment (the balls are identical)
            e = max(min(np.linalg.norm(x - t) for t in tg) for x in b) * 100
            worst = max(worst, e)
            if frames is not None and k % 50 == 49:
                frames(pr)
            if any(np.linalg.norm(x - np.mean(b, 0)) > 0.06 for x in b) or e > 4.0:
                lost = k * dt
                break
    return sign * rate * k * dt, turn, worst, lost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--center", default="0.1,6.5")
    ap.add_argument("--rate", type=float, default=20.0, help="deg/s of commanded phase")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--radius", type=float, default=0.0225)
    args = ap.parse_args()
    center = tuple(float(v) for v in args.center.split(","))
    phases, fingers = table(args.table, center)
    pr = Probe(args.radius)
    for sign in (+1, -1):
        cmd, turn, worst, lost = sweep(pr, phases, fingers, center, sign, np.radians(args.rate), args.seconds)
        print(f"centre {center} sign {sign:+d} rate {args.rate:.0f} deg/s: commanded {cmd:+.2f} rad, trio turned {turn:+.2f} rad, "
              f"worst ball off its target {worst:.1f} cm" + (f", LOST at {lost:.1f}s" if lost is not None else ""), flush=True)


if __name__ == "__main__":
    main()
