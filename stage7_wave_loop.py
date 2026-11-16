"""Stage 7A, the long cup, step 4b: one ball round a LOOP -- the finger wave (stage7_wave.py) the drive.

The finger wave carries the ball along the finger row (+y) uphill against a roll of the whole cup toward -y;
at the row's end the last fingers lift harder (`end`) and flick it back up onto the palm (-x); there gravity
(the cup rolled toward -y, the wrists flexed so the fingers are downhill) rolls it diagonally across the palm
back down onto the finger row near its start, and the wave picks it up again. Only the finger row is driven,
so in this loop no finger stands in the ball's way on the return.

Reports the ball's turn round the loop centre (between the finger row and the palm) over the run.

Usage:
    python3 stage7_wave_loop.py                  # grid
    python3 stage7_wave_loop.py --one T,amp,lag,flex,roll,end,nend,wall --record data/stage7_wave_loop.npz
"""
import argparse
from itertools import product
from multiprocessing import Pool

import mujoco as mj
import numpy as np

from stage7_layout import PALM_Z
from stage7_teacher import CTRL_DT
from stage7_wave import ROW, make, start

BASE, TIP = 0.0, 0.3
FAR_WALL = True


def loop(w, row, T, amp, lag, roll, end, nend, seconds, record=False, wall=0.0):
    m, d = w.m, w.d
    kn = [m.actuator(f"{h}_A_{f}J3").id for h, f in ROW]
    ro = [m.actuator(f"{h}_A_FRJ").id for h in ("rh", "L_lh")]
    per = int(round(CTRL_DT / m.opt.timestep))
    c = np.array([row[:, 0].mean() - 0.04, row[:, 1].mean()])
    th_prev = turn = 0.0
    b = w.balls()[0]
    th_prev = np.arctan2(b[1] - c[1], b[0] - c[0])
    log = {k: [] for k in ("qpos", "balls", "phase")}
    when = None
    for i in range(int(round(seconds / CTRL_DT))):
        t = i * CTRL_DT
        g = min(t / 0.5, 1.0)
        u = w.u0.copy()
        u[ro] += g * roll                                        # FRJ + rolls the balls toward -y
        if wall:                                                 # the index fingers at the row's two ends: curled walls
            u[[kn[0], kn[-1]]] = w.u0[[kn[0], kn[-1]]] + g * wall
        for k, a in enumerate(kn):
            if wall and k in (0, 7):
                continue
            A = amp + (end if k >= (7 if wall else 8) - nend and k < (7 if wall else 8) else 0.0)
            u[a] += g * A * max(0.0, np.sin(2 * np.pi * (t / T - k * lag))) ** 2
        d.ctrl[:] = np.clip(u, w.lo, w.hi)
        for _ in range(per):
            mj.mj_step(m, d)
        b = w.balls()[0]
        th = np.arctan2(b[1] - c[1], b[0] - c[0])
        turn += np.angle(np.exp(1j * (th - th_prev)))
        th_prev = th
        if record:
            log["qpos"].append(d.qpos.copy()); log["balls"].append(w.balls().copy()); log["phase"].append(turn)
        if b[2] < PALM_Z - 0.05 or np.linalg.norm(b[:2] - c) > 0.12:
            when = t + CTRL_DT
            break
    return np.degrees(turn), when, log, c


_w = {}


def job(p):
    T, amp, lag, flex, roll, end, nend, wall, seconds = p
    if flex not in _w:
        _w[flex] = make(BASE, TIP, flex)
    w, row = _w[flex]
    start(w, row, 0.02, i0=1.5 if wall else 0.5)
    deg, when, _, _ = loop(w, row, T, amp, lag, roll, end, int(nend), seconds, wall=wall)
    return p, deg, when


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--periods", default="0.4,0.8")
    ap.add_argument("--amps", default="0.8,1.2")
    ap.add_argument("--lags", default="0.125,0.25")
    ap.add_argument("--flexes", default="-0.1,-0.15")
    ap.add_argument("--rolls", default="0.03,0.06,0.1,0.15")
    ap.add_argument("--ends", default="0.0,0.4,0.8")
    ap.add_argument("--nends", default="1,2")
    ap.add_argument("--walls", default="0.0,0.8,1.3", help="rad: the index fingers' extra curl (0: no walls, in the wave)")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--one", default=None)
    ap.add_argument("--record", default=None)
    args = ap.parse_args()
    fl = lambda s: [float(v) for v in s.split(",")]
    if args.one:
        T, amp, lag, flex, roll, end, nend, wall = fl(args.one)
        w, row = make(BASE, TIP, flex)
        start(w, row, 0.02, i0=1.5 if wall else 0.5)
        deg, when, log, c = loop(w, row, T, amp, lag, roll, end, int(nend), args.seconds, record=True, wall=wall)
        print(f"turn {deg:+.0f} deg ({deg / 360:+.2f} laps)  {'kept' if when is None else f'lost at {when:.1f}s'}")
        if args.record:
            np.savez(args.record, qpos=np.array(log["qpos"]), balls=np.array(log["balls"]), phase=np.array(log["phase"]),
                     center=c, rc=0.04, ball=0.019, roll_at="seam")
            print("recorded", args.record)
        return
    grid = list(product(fl(args.periods), fl(args.amps), fl(args.lags), fl(args.flexes), fl(args.rolls), fl(args.ends),
                        fl(args.nends), fl(args.walls), [args.seconds]))
    grid.sort(key=lambda p: p[3])
    with Pool(15) as pool:
        res = pool.map(job, grid, chunksize=4)
    res.sort(key=lambda r: (r[2] is None, r[1] if r[2] is None else r[2]), reverse=True)
    for p, deg, when in res[:20]:
        print(f"T {p[0]:.1f} amp {p[1]:.1f} lag {p[2]:.3f} flex {p[3]:+.2f} roll {p[4]:.2f} end +{p[5]:.1f} x{p[6]:.0f} wall {p[7]:.1f}: "
              f"turn {deg:+5.0f} deg ({deg / 360:+.2f} laps)  {'kept' if when is None else f'lost at {when:.1f}s'}")
    print(f"{sum(r[2] is None for r in res)}/{len(res)} kept the ball {args.seconds:g} s")


if __name__ == "__main__" and "--fsm" not in __import__("sys").argv:
    main()


def fsm(w, row, T, amp, lag, wall, flex_row, flex_up, roll_back, y_up, y_down, seconds, record=False, thumb=None):
    """FEEDBACK on the ball's place, three legs:  ROW: the wave carries it +y (fingers downhill, flex_row);
    UP: past y_up (the row's far end) the cup tips toward the wrists (flex_up) and rolls toward -y (roll_back),
    so it rolls up off the fingers and back along the palm; DOWN: past y_down (back at the start) the cup tips
    toward the fingers again and it drops onto the row. The wave runs all the time.
    `thumb` = (Ts, k, th5, y_th): while the ball is on the way back and beyond y_th (the far end's hollow at the
    left thumb's base) the left thumb SWEEPS it back across the palm: once every Ts s it swings in (THJ4 up,
    THJ2 and THJ1 curled, THJ5 at th5) by the fraction k of the full sweep and out again. -> turn (deg), loss, log."""
    m, d = w.m, w.d
    kn = [m.actuator(f"{h}_A_{f}J3").id for h, f in ROW]
    ro = [m.actuator(f"{h}_A_FRJ").id for h in ("rh", "L_lh")]
    fx = [m.actuator(f"{h}_A_WRJ1").id for h in ("rh", "L_lh")]
    per = int(round(CTRL_DT / m.opt.timestep))
    c = np.array([row[:, 0].mean() - 0.04, row[:, 1].mean()])
    b = w.balls()[0]
    th_prev, turn = np.arctan2(b[1] - c[1], b[0] - c[0]), 0.0
    log = {k: [] for k in ("qpos", "balls", "phase")}
    when, leg = None, "row"
    th = {j: m.actuator(f"L_lh_A_{j}").id for j in ("THJ5", "THJ4", "THJ2", "THJ1")}
    sweep = {"THJ4": 1.22, "THJ2": 0.7, "THJ1": 1.2}
    t_sw = None
    flex, roll = flex_row, 0.0
    for i in range(int(round(seconds / CTRL_DT))):
        t = i * CTRL_DT
        b = w.balls()[0]
        if leg == "row" and b[1] > y_up:
            leg = "back"
        elif leg == "back" and b[1] < y_down:
            leg = "row"
        tf, tr = (flex_row, 0.0) if leg == "row" else (flex_up, roll_back)
        flex += np.clip(tf - flex, -0.02, 0.02)                  # tilts slewed at 1 rad/s
        roll += np.clip(tr - roll, -0.02, 0.02)
        u = w.u0.copy()
        if thumb:
            Ts, k, th5, y_th = thumb
            if t_sw is None and leg == "back" and b[1] > y_th:
                t_sw = t                                         # start a sweep (it always finishes)
            if t_sw is not None:
                ph = (t - t_sw) / Ts
                s = np.sin(np.pi * ph) ** 2 if ph < 1.0 else 0.0
                u[th["THJ5"]] += s * (th5 - w.u0[th["THJ5"]])
                for j, v in sweep.items():
                    u[th[j]] += s * k * (v - w.u0[th[j]])
                if ph >= 1.0:
                    t_sw = None
        u[fx] = flex
        u[ro] += roll
        if wall:                                                 # the near end's index finger a wall; the far end's
            u[kn[0]] += wall                                     # too, unless FAR_WALL is off (it makes a pocket)
            if FAR_WALL:
                u[kn[-1]] += wall
        for k, a in enumerate(kn):
            if wall and (k == 0 or (k == 7 and FAR_WALL)):
                continue
            u[a] += amp * max(0.0, np.sin(2 * np.pi * (t / T - k * lag))) ** 2
        d.ctrl[:] = np.clip(u, w.lo, w.hi)
        for _ in range(per):
            mj.mj_step(m, d)
        b = w.balls()[0]
        ang = np.arctan2(b[1] - c[1], b[0] - c[0])
        turn += np.angle(np.exp(1j * (ang - th_prev)))
        th_prev = ang
        if record:
            log["qpos"].append(d.qpos.copy()); log["balls"].append(w.balls().copy()); log["phase"].append(turn)
        if b[2] < PALM_Z - 0.05 or np.linalg.norm(b[:2] - c) > 0.12:
            when = t + CTRL_DT
            break
    return np.degrees(turn), when, log, c


def fsm_job(p):
    T, amp, lag, wall, flex_row, flex_up, roll_back, y_up, y_down, seconds = p[:10]
    if flex_row not in _w:
        _w[flex_row] = make(BASE, TIP, flex_row)
    w, row = _w[flex_row]
    start(w, row, 0.02, i0=1.5)
    deg, when, _, _ = fsm(w, row, T, amp, lag, wall, flex_row, flex_up, roll_back, y_up, y_down, seconds,
                          thumb=p[10:] or None)
    return p, deg, when


def fsm_main():
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsm", action="store_true")
    ap.add_argument("--one", default=None, help="T,amp,lag,wall,flex_row,flex_up,roll_back,y_up,y_down[,Ts,k,th5,y_th]")
    ap.add_argument("--thumb", action="store_true", help="grid over the thumb sweep too")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--record", default=None)
    args = ap.parse_args()
    if args.one:
        v = [float(x) for x in args.one.split(",")]
        w, row = make(BASE, TIP, v[4])
        start(w, row, 0.02, i0=1.5)
        deg, when, log, c = fsm(w, row, *v[:9], args.seconds, record=True, thumb=tuple(v[9:]) or None)
        print(f"turn {deg:+.0f} deg ({deg / 360:+.2f} laps)  {'kept' if when is None else f'lost at {when:.1f}s'}")
        if args.record:
            np.savez(args.record, qpos=np.array(log["qpos"]), balls=np.array(log["balls"]), phase=np.array(log["phase"]),
                     center=c, rc=0.04, ball=0.019, roll_at="seam")
            print("recorded", args.record)
        return
    grid = list(product([0.4], [0.8, 1.0], [0.125, 0.18], [0.8], [-0.1], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3],
                        [0.04, 0.055, 0.07], [0.0, 0.02], [args.seconds]))
    if args.thumb:
        grid = list(product([0.4], [0.8, 1.0], [0.125], [0.8], [-0.1], [0.1, 0.3], [0.1, 0.2], [0.07], [0.0],
                            [args.seconds], [0.4, 0.8], [0.6, 1.0], [-0.5, 0.3, 1.0], [0.08, 0.095]))
    with Pool(15) as pool:
        res = pool.map(fsm_job, grid, chunksize=2)
    res.sort(key=lambda r: (r[2] is None, r[1] if r[2] is None else r[2]), reverse=True)
    for p, deg, when in res[:15]:
        print(f"T {p[0]} amp {p[1]} lag {p[2]} wall {p[3]} flex row {p[4]:+.2f} up {p[5]:+.2f} roll back {p[6]:.2f} "
              f"y_up {p[7] * 100:.0f} y_down {p[8] * 100:.0f} cm{' thumb Ts %.1f k %.1f th5 %+.1f y_th %.1f' % (p[10], p[11], p[12], p[13] * 100) if len(p) > 10 else ''}: turn {deg:+5.0f} ({deg / 360:+.2f} laps)  "
              f"{'kept' if when is None else f'lost at {when:.1f}s'}")
    print(f"{sum(r[2] is None for r in res)}/{len(res)} kept")


if __name__ == "__main__" and "--fsm" in __import__("sys").argv:
    fsm_main()
