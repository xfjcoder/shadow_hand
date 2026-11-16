"""Stage 7A, the long cup, step 4a: can the FINGERS carry a ball along the finger end of the cup?

The eight fingers of the two hands side by side make one row across the cup (y: right FF, MF, RF, LF | seam |
left LF, RF, MF, FF, ~2.2cm apart). A 38mm ball resting on the fingers spans two of them. A WAVE of finger
lifts runs along the row: finger i's knuckle (J3) lifts by amp max(0, sin(2 pi (t/T - i lag)))^2, so the
finger behind the ball rises while the one ahead stays down and the ball rolls onto it -- a peristaltic
conveyor. Fingers that are down lie flat (`base`), so they do not block the ball rolling over them.

One ball (the other three parked off the hands) is set on the first fingers of the row; the test reports how
far it travels along the row (y) and whether it stays on the hands, over a grid of wave settings.

Usage:
    python3 stage7_wave.py                       # grid search, prints the best settings
    python3 stage7_wave.py --one T,amp,lag,base,tip,flex,roll --record data/stage7_wave.npz
"""
import argparse
from itertools import product
from multiprocessing import Pool

import mujoco as mj
import numpy as np

from stage7_cup import cup_posture
from stage7_layout import PALM_Z, set_ball
from stage7_teacher import BALLS, CTRL_DT, World

S = mj.mjtState.mjSTATE_INTEGRATION
ROW = [("rh", f) for f in ("FF", "MF", "RF", "LF")] + [("L_lh", f) for f in ("LF", "RF", "MF", "FF")]   # y increasing
PARK = [(0.0, -0.3 - 0.1 * k, 0.05) for k in range(3)]                     # the unused balls, off the hands


TH4 = 1.1                                                        # thumb abduction (raised: an end wall -- and a pocket)


def make(base, tip, flex, ball=0.019):
    post = cup_posture(curl=base, tip=tip, lf=base, lf_tip=tip, th4=TH4, th5=0.3, th1=0.3)
    w = World(ball=ball, tilt=np.radians(25), flex=flex, posture=post, roll=np.radians(20), roll_at="seam")
    w.settle(along=0.32, ring=0.045, seconds=0.5)
    m, d = w.m, w.d
    ids = [m.body(f"{h}_{f.lower()}proximal").id for h, f in ROW]
    row = d.xpos[ids].copy()                                     # knuckle positions along the row
    w.st0 = np.empty(mj.mj_stateSize(m, S))                      # the settled hands: every run starts from here
    mj.mj_getState(m, d, w.st0, S)
    return w, row


def start(w, row, x_off, i0=0.5):
    """Ball 0 over the proximal phalanges between fingers i0-0.5 and i0+0.5, the rest parked; settle 1 s."""
    m, d = w.m, w.d
    mj.mj_setState(m, d, w.st0, S)
    y = np.interp(i0, np.arange(8), row[:, 1])
    set_ball(m, d, BALLS[0], [row[:, 0].mean() + x_off, y, PALM_Z + w.r + 0.01])
    for b, p in zip(BALLS[1:], PARK):
        set_ball(m, d, b, p)
    d.ctrl[:] = w.u0
    for _ in range(int(1.0 / m.opt.timestep)):
        mj.mj_step(m, d)


def wave(w, T, amp, lag, roll, seconds, record=False, end_y=None):
    """Run the wave (and a roll bias of the whole cup, rad, + tilts it toward +y). -> y travel (m), loss time, log,
    the time the ball first reached `end_y` (the row's last finger; it is then off the end, so the run stops)."""
    m, d = w.m, w.d
    kn = [m.actuator(f"{h}_A_{f}J3").id for h, f in ROW]
    ro = [m.actuator(f"{h}_A_FRJ").id for h in ("rh", "L_lh")]
    per = int(round(CTRL_DT / m.opt.timestep))
    b0 = w.balls()[0].copy()
    log = {k: [] for k in ("qpos", "balls", "phase")}
    when = arrived = None
    for i in range(int(round(seconds / CTRL_DT))):
        t = i * CTRL_DT
        g = min(t / 0.5, 1.0)
        u = w.u0.copy()
        u[ro] += -g * roll                                       # FRJ - rolls balls toward +y
        for k, a in enumerate(kn):
            u[a] += g * amp * max(0.0, np.sin(2 * np.pi * (t / T - k * lag))) ** 2
        d.ctrl[:] = np.clip(u, w.lo, w.hi)
        for _ in range(per):
            mj.mj_step(m, d)
        b = w.balls()[0]
        if record:
            log["qpos"].append(d.qpos.copy()); log["balls"].append(w.balls().copy()); log["phase"].append(0.0)
        if end_y is not None and b[1] >= end_y:
            arrived = t + CTRL_DT
            break
        if b[2] < PALM_Z - 0.05 or abs(b[0] - b0[0]) > 0.08:
            when = t + CTRL_DT
            break
    return w.balls()[0] - b0, when, log, arrived


_w = {}


def job(p):
    T, amp, lag, base, tip, flex, roll, x_off, seconds = p
    key = (base, tip, flex)
    if key not in _w:
        _w[key] = make(base, tip, flex)
    w, row = _w[key]
    start(w, row, x_off)
    dy, when, _, arrived = wave(w, T, amp, lag, roll, seconds, end_y=row[-1, 1])
    return p, dy, when, arrived


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", default="0.4,0.8,1.6")
    ap.add_argument("--amps", default="0.4,0.8,1.2")
    ap.add_argument("--lags", default="0.125,0.25", help="fraction of a period between neighbouring fingers")
    ap.add_argument("--bases", default="0.0,0.2")
    ap.add_argument("--tips", default="0.3,0.8", help="J0 (outer joints): a lip at the fingertips")
    ap.add_argument("--flexes", default="-0.1,0.05,0.2")
    ap.add_argument("--rolls", default="0.0")
    ap.add_argument("--x-off", type=float, default=0.02, help="m beyond the knuckles the ball starts")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--one", default=None)
    ap.add_argument("--record", default=None)
    args = ap.parse_args()
    fl = lambda s: [float(v) for v in s.split(",")]
    if args.one:
        T, amp, lag, base, tip, flex, roll = fl(args.one)
        w, row = make(base, tip, flex)
        print("row y (cm):", np.round(row[:, 1] * 100, 1))
        start(w, row, args.x_off)
        dy, when, log, arrived = wave(w, T, amp, lag, roll, args.seconds, record=True, end_y=row[-1, 1])
        print(f"ball moved dx {dy[0] * 100:+.1f} dy {dy[1] * 100:+.1f} cm  "
              f"{f'reached the row end at {arrived:.2f}s' if arrived else 'kept' if when is None else f'lost at {when:.1f}s'}")
        if args.record:
            c = row[:, :2].mean(0)
            np.savez(args.record, qpos=np.array(log["qpos"]), balls=np.array(log["balls"]), phase=np.array(log["phase"]),
                     center=c, rc=0.03, ball=0.019, roll_at="seam")
            print("recorded", args.record)
        return
    grid = list(product(fl(args.periods), fl(args.amps), fl(args.lags), fl(args.bases), fl(args.tips), fl(args.flexes),
                        fl(args.rolls), [args.x_off], [args.seconds]))
    grid.sort(key=lambda p: p[3:6])                              # one World per posture in each worker, mostly
    with Pool(15) as pool:
        res = pool.map(job, grid, chunksize=4)
    res.sort(key=lambda r: (r[3] is not None, -(r[3] or 0), r[2] is None, r[1][1]), reverse=True)   # reached, soonest
    for p, dy, when, arrived in res[:20]:
        print(f"T {p[0]:.1f} amp {p[1]:.1f} lag {p[2]:.3f} base {p[3]:.1f} tip {p[4]:.1f} flex {p[5]:+.2f} roll {p[6]:+.2f}: "
              f"dy {dy[1] * 100:+5.1f} cm dx {dy[0] * 100:+5.1f}  "
              f"{f'reached the end at {arrived:.2f}s' if arrived else 'kept' if when is None else f'lost at {when:.1f}s'}")
    print(f"{sum(r[3] is not None for r in res)}/{len(res)} carried the ball the length of the row (~15 cm); "
          f"{sum(r[2] is not None for r in res)} dropped it")


if __name__ == "__main__":
    main()
