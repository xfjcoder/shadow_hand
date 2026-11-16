"""Stage 7A, the long cup: both hands side by side, palms up, cupped into ONE long bowl -- the curled fingers
the wall at the finger end, the thumbs the walls at the two far ends, the wrists flexed so the palms' heels
make a sloped wall at the wrist end, the little fingers flat at the seam so balls can cross. Four balls in
the cup; a circular tilt of the whole cup (both wrists' flex + both forearms' roll, in step) should swirl
them round its rim like wine round a glass.

Step 1 (default): settle the balls in the cup, then hold tilts each way for a few seconds; report what
stays, save two views.

Usage:
    python3 stage7_cup.py --curl 0.7 --tip 0.9 --out media_full/stage7_cup.png
"""
import argparse

import mujoco as mj
import numpy as np
from PIL import ImageDraw

from stage7_layout import PALM_Z, views
from stage7_teacher import CTRL_DT, World

S = mj.mjtState.mjSTATE_INTEGRATION
FINGERS = ("FF", "MF", "RF")
RIM_R = 0.025                  # m: a ball nearer the cup centre than this is not going round the rim


def cup_posture(curl, tip, lf, lf_tip, th4, th5, th1):
    p = {f"rh_A_{f}J3": curl for f in FINGERS}
    p.update({f"rh_A_{f}J0": tip for f in FINGERS})
    p.update({"rh_A_LFJ3": lf, "rh_A_LFJ0": lf_tip, "rh_A_THJ4": th4, "rh_A_THJ5": th5, "rh_A_THJ1": th1,
              "rh_A_FFJ4": 0.0, "rh_A_MFJ4": 0.0, "rh_A_RFJ4": 0.0, "rh_A_LFJ4": 0.0})
    return p


def make(args):
    post = cup_posture(args.curl, args.tip, args.lf, args.lf_tip, args.th4, args.th5, args.th1)
    w = World(ball=args.ball, tilt=np.radians(25), flex=args.flex, posture=post, roll=np.radians(20), roll_at=args.roll_at)
    w.settle(along=args.along, ring=args.ring, seconds=args.settle, phase=np.radians(45))
    return w


def lost(w, c, far):
    b = w.balls()
    return bool(np.any(b[:, 2] < PALM_Z - 0.02) or np.any(np.linalg.norm(b[:, :2] - c, axis=1) > far))


def hold_test(w, deg, seconds, far):
    """From the settled cup, ramp to a tilt each way (flex and roll of both hands together) and hold."""
    st = np.empty(mj.mj_stateSize(w.m, S))
    mj.mj_getState(w.m, w.d, st, S)
    c = w.balls()[:, :2].mean(0)
    ids = {k: [w.m.actuator(f"{h}_A_{k}").id for h in ("rh", "L_lh")] for k in ("WRJ1", "FRJ")}
    out = []
    for name, k, s in (("toward wrist", "WRJ1", 1), ("toward fingers", "WRJ1", -1), ("toward +y (left)", "FRJ", -1),
                       ("toward -y (right)", "FRJ", 1)):
        mj.mj_setState(w.m, w.d, st, S)
        u = w.u0.copy()
        u[ids[k]] += s * np.radians(deg)
        when = None
        for i in range(int(seconds / w.m.opt.timestep)):
            w.d.ctrl[:] = np.clip(w.u0 + (u - w.u0) * min(i * w.m.opt.timestep / 0.5, 1.0), w.lo, w.hi)
            mj.mj_step(w.m, w.d)
            if when is None and lost(w, c, far):
                when = i * w.m.opt.timestep
        out.append((name, when, np.round((w.balls()[:, :2] - c) * 100, 1)))
    mj.mj_setState(w.m, w.d, st, S)
    mj.mj_forward(w.m, w.d)
    return out


def swirl(w, st, c, T, a_flex, a_roll, sense, ramp, seconds, far, record=False, flick=(0.0, 0.0, 0.0)):
    """Tilt the whole cup round a circle: both wrists' flex a_flex sin(wt), both rolls a_roll sin(wt + sense 90deg)
    (rad), the amplitudes ramped in over `ramp` periods. `flick` = (amp, phase, offset): every finger's knuckle
    (J3) of the right hand adds amp max(0, sin(wt + phase))^2 -- a pulse once a circle that lifts the ball
    out of the crease where the fingers meet the palm -- and the left hand the same `offset` rad later.
    -> each ball's turn round the cup centre (deg), loss time."""
    m, d = w.m, w.d
    mj.mj_setState(m, d, st, S)
    fl = [m.actuator(f"{h}_A_WRJ1").id for h in ("rh", "L_lh")]
    ro = [m.actuator(f"{h}_A_FRJ").id for h in ("rh", "L_lh")]
    knuckles = [[m.actuator(f"{h}_A_{f}J3").id for f in FINGERS + ("LF",)] for h in ("rh", "L_lh")]
    per = int(round(CTRL_DT / m.opt.timestep))
    th_prev = np.arctan2(*(w.balls()[:, :2] - c)[:, ::-1].T)
    turn, when = np.zeros(4), None
    log = {k: [] for k in ("qpos", "balls", "phase")}
    for i in range(int(round(seconds / CTRL_DT))):
        t = i * CTRL_DT
        g = min(t / (ramp * T), 1.0)
        u = w.u0.copy()
        u[fl] += g * a_flex * np.sin(2 * np.pi * t / T)
        u[ro] += g * a_roll * np.sin(2 * np.pi * t / T + sense * np.pi / 2)
        for k, ids in enumerate(knuckles):
            u[ids] += g * flick[0] * max(0.0, np.sin(2 * np.pi * t / T + flick[1] + k * flick[2])) ** 2
        d.ctrl[:] = np.clip(u, w.lo, w.hi)
        for _ in range(per):
            mj.mj_step(m, d)
        b = w.balls()
        th = np.arctan2(b[:, 1] - c[1], b[:, 0] - c[0])
        rim = np.linalg.norm(b[:, :2] - c, axis=1) > RIM_R      # only round the rim counts (not swings past the centre)
        turn += rim * np.angle(np.exp(1j * (th - th_prev)))
        th_prev = th
        if record:
            log["qpos"].append(d.qpos.copy()); log["balls"].append(b.copy()); log["phase"].append(float(turn.mean()))
        if np.any(b[:, 2] < PALM_Z - 0.05) or np.any(np.linalg.norm(b[:, :2] - c, axis=1) > far):
            when = t + CTRL_DT
            break
    return np.degrees(turn), when, log


_job = None


def _init(args):
    global _job
    w = make(args)
    st = np.empty(mj.mj_stateSize(w.m, S))
    mj.mj_getState(w.m, w.d, st, S)
    _job = (w, st, w.balls()[:, :2].mean(0), args)


def _swirl(p):
    w, st, c, args = _job
    deg, when, _ = swirl(w, st, c, *p[:4], ramp=args.ramp, seconds=args.seconds, far=args.far, flick=p[4:] or (0.0, 0.0, 0.0))
    return p, deg, when


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curl", type=float, default=0.7, help="FF/MF/RF J3 (knuckle) -- the finger-end wall")
    ap.add_argument("--tip", type=float, default=0.9, help="FF/MF/RF J0 (the two outer joints)")
    ap.add_argument("--lf", type=float, default=0.5, help="little finger J3 (curled: else the finger-end wall has a gap at the seam)")
    ap.add_argument("--lf-tip", type=float, default=0.6)
    ap.add_argument("--th4", type=float, default=1.1, help="thumb abduction: raised as the far-end wall")
    ap.add_argument("--th5", type=float, default=0.3)
    ap.add_argument("--th1", type=float, default=0.3)
    ap.add_argument("--flex", type=float, default=-0.1, help="wrist flex (- : heels higher, balls lean on the fingers)")
    ap.add_argument("--along", type=float, default=0.32)
    ap.add_argument("--ring", type=float, default=0.045)
    ap.add_argument("--ball", type=float, default=0.019)
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--deg", type=float, default=8.0, help="hold-test tilt")
    ap.add_argument("--far", type=float, default=0.10)
    ap.add_argument("--roll-at", default="seam", choices=("seam", "palm"), help="the forearm rolls' axis (seam: one rigid cup)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--swirl", action="store_true", help="step 3: a grid of circular cup tilts (period x flex x roll x sense)")
    ap.add_argument("--periods", default="0.6,0.8,1.0,1.4")
    ap.add_argument("--flexes", default="0.05,0.1,0.15", help="rad, flex amplitudes")
    ap.add_argument("--rolls", default="0.05,0.1,0.15,0.2", help="rad, roll amplitudes")
    ap.add_argument("--ramp", type=float, default=3.0, help="periods to ramp the swirl in over")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--one", default=None, help="T,a_flex,a_roll,sense[,flick amp,phase,offset]: run one swirl and --record it")
    ap.add_argument("--flicks", default=None, help="amps (rad): also search a finger flick (8 phases x 4 hand offsets)")
    ap.add_argument("--record", default=None)
    args = ap.parse_args()
    if args.swirl:
        from itertools import product
        from multiprocessing import Pool
        fl = lambda s: [float(v) for v in s.split(",")]
        grid = list(product(fl(args.periods), fl(args.flexes), fl(args.rolls), (1, -1)))
        if args.flicks:
            grid = [g + f for g in grid for f in product(fl(args.flicks), np.arange(8) * np.pi / 4, (0.0, np.pi / 2, -np.pi / 2, np.pi))]
        with Pool(15, initializer=_init, initargs=(args,)) as pool:
            res = pool.map(_swirl, grid)
        def rank(r):                                   # the slowest ball's turn, if all four go the same way
            same = np.all(np.sign(r[1]) == np.sign(r[1][0]))
            return (r[2] is None, np.abs(r[1]).min() if same else 0.0, np.abs(r[1]).mean())
        res.sort(key=rank, reverse=True)
        for p, deg, when in res[:20]:
            fk = f" flick {p[4]:.1f} @{np.degrees(p[5]):.0f} off {np.degrees(p[6]):+.0f}" if len(p) > 4 else ""
            print(f"T {p[0]:.1f}s flex {p[1]:.2f} roll {p[2]:.2f} sense {p[3]:+d}{fk}: turns {np.round(deg).astype(int)} "
                  f"{'kept' if when is None else f'lost at {when:.1f}s'}")
        n_kept = sum(r[2] is None for r in res)
        print(f"{n_kept}/{len(res)} kept for {args.seconds:g} s")
        return
    if args.one:
        _init(args)
        w, st, c, _ = _job
        v = [float(x) for x in args.one.split(",")]
        deg, when, log = swirl(w, st, c, v[0], v[1], v[2], int(v[3]), args.ramp, args.seconds, args.far, record=True,
                               flick=tuple(v[4:7]) if len(v) > 4 else (0.0, 0.0, 0.0))
        print(f"turns {np.round(deg).astype(int)} {'kept' if when is None else f'lost at {when:.1f}s'}")
        if args.record:
            b0 = log["balls"][0]
            np.savez(args.record, qpos=np.array(log["qpos"]), balls=np.array(log["balls"]), phase=np.array(log["phase"]),
                     center=c, rc=float(np.mean(np.linalg.norm(b0[:, :2] - c, axis=1))), ball=args.ball, roll_at=args.roll_at)
            print("recorded", args.record)
        return
    w = make(args)
    b = w.balls()
    c = b[:, :2].mean(0)
    print(f"settled: centre ({c[0] * 100:.1f},{c[1] * 100:.1f}) cm; balls rel (cm) + z:",
          " ".join(f"({x:+.1f},{y:+.1f},z{z * 100:.1f})" for (x, y), z in zip((b[:, :2] - c) * 100, b[:, 2])),
          "| on the hands" if not lost(w, c, args.far) else "| LOST already")
    for name, when, rel in hold_test(w, args.deg, 3.0, args.far):
        print(f"  tilt {args.deg:g} deg {name:18s}: {'kept' if when is None else f'lost at {when:.1f}s'}   balls {rel.tolist()}")
    if args.out:
        r = mj.Renderer(w.m, height=360, width=480)
        im = views(w.m, w.d, r, [c[0], c[1], PALM_Z + 0.02])
        r.close()
        ImageDraw.Draw(im).text((8, 6), f"long cup: curl {args.curl} tip {args.tip} th4 {args.th4} flex {args.flex}",
                                fill=(255, 255, 255))
        im.save(args.out)
        print("saved", args.out)


if __name__ == "__main__":
    main()
