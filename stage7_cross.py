"""Stage 7A, open-hand loop: a SCRIPTED crossing cycle -- each palm a tilting tray worked by two joints,
    wrist flex WRJ1 (+ rolls its balls toward the wrist, - toward the fingers) and
    forearm roll FRJ (right - / left + send its balls across the seam to the other hand),
in the user's pattern: the right palm holds its pair at the wrist end while the left holds its pair at the
finger end; the rolls send the left's top pair across to the right along the top and the right's bottom
pair to the left along the bottom; the flexes flip (right: top -> bottom, left: bottom -> top); repeat.
One period T is half a loop. Every joint follows one wave of the cycle,
    u_j(t) = u0_j + dc_j + A_j sin(2 pi t / T + phi_j)      (the flex waves of the two hands in antiphase)
with optional finger pushes (every finger's J3 of a hand together). A small CEM (13-17 numbers, unlike the
~160 of `stage7_rhythm.py`) tunes them from the hand-designed cycle for the slowest ball's turn.

Usage:
    python3 stage7_cross.py --try                      # the hand-designed cycle only
    python3 stage7_cross.py --iters 40 --out data/stage7_cross.npz
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

import stage7_rhythm
from stage7_layout import PALM_Z
from stage7_rhythm import make_world, rollout

JOINTS = ("rh_A_WRJ1", "L_lh_A_WRJ1", "rh_A_FRJ", "L_lh_A_FRJ")
FINGERS = [f"{h}_A_{f}J3" for h in ("rh", "L_lh") for f in ("FF", "MF", "RF", "LF")]
# x = [log T, A x4, phi x4, dc x4, (push amp, push phase) x2 hands]
NAMES = (["logT"] + [f"A_{j}" for j in JOINTS] + [f"phi_{j}" for j in JOINTS] + [f"dc_{j}" for j in JOINTS]
         + ["push_r", "pphi_r", "push_l", "pphi_l"])
# the hand-designed cycle (rad; one period = half a loop). Flex: the right palm stays tipped toward the wrist
# (+: a pair arriving along the top rolls down to the bottom) and the left toward the fingers (-: bottom to
# top), with a little wave. Rolls: half a period "send" (right -, left +: the right's bottom pair and the
# left's top pair cross together), half "receive" (right +, left -: the arrivals pushed out to the ring).
DESIGN = {"logT": np.log(2.0), "dc_rh_A_WRJ1": 0.15, "dc_L_lh_A_WRJ1": -0.15,
          "A_rh_A_WRJ1": 0.1, "A_L_lh_A_WRJ1": 0.1, "phi_rh_A_WRJ1": 0.0, "phi_L_lh_A_WRJ1": np.pi,
          "A_rh_A_FRJ": 0.2, "A_L_lh_A_FRJ": 0.2, "phi_rh_A_FRJ": np.pi, "phi_L_lh_A_FRJ": 0.0}


def design():
    x = np.zeros(len(NAMES))
    for k, v in DESIGN.items():
        x[NAMES.index(k)] = v
    return x


def controls(x, t, w, K=None):
    T = float(np.clip(np.exp(x[0]), 0.6, 4.0))
    A, ph, dc = x[1:5], x[5:9], x[9:13]
    ramp = min(t / T, 1.0)
    u = w.u0.copy()
    ids = [w.m.actuator(j).id for j in JOINTS]
    s = np.sin(2 * np.pi * t / T + ph)
    u[ids] += ramp * (dc + A * s)
    for hand, (amp, pph) in zip(("rh", "L_lh"), (x[13:15], x[15:17])):
        if amp:
            fi = [w.m.actuator(f).id for f in FINGERS if f.startswith(hand + "_")]
            u[fi] += ramp * amp * max(np.sin(2 * np.pi * t / T + pph), 0.0)   # a push, no pull back past open
    return np.clip(u, w.lo, w.hi)


stage7_rhythm.controls = controls                    # rollout() looks the controls up in its own module
_w, _cfg = None, None


def _init(cfg):
    global _w, _cfg
    _cfg, _w = cfg, make_world(cfg)


def evaluate(x):
    w, state, center = _w
    s, deg, lost_at, _ = rollout(x, w, state, center, _cfg)
    return s, deg, lost_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--try", dest="try_", action="store_true", help="run the hand-designed cycle (or --x) only")
    ap.add_argument("--x", default=None, help="json list: a parameter vector to start from (or try)")
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--pop", type=int, default=90)
    ap.add_argument("--elite", type=int, default=10)
    ap.add_argument("--procs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spin", type=float, default=0.0, help="deg/s the balls start rolling round at")
    ap.add_argument("--push", action="store_true", help="also search the finger pushes")
    ap.add_argument("--along", type=float, default=0.33, help="world x of the loop centre (0.30: the palms' hollow, walls both ends)")
    ap.add_argument("--j3", type=float, default=0.1, help="every finger's J3 in the posture (0.3: a soft wall at the finger end)")
    ap.add_argument("--sign", type=float, default=-1.0, help="the loop's direction scored: -1 = the user's pattern "
                    "(top row left -> right hand, bottom row right -> left)")
    ap.add_argument("--out", default="data/stage7_cross.npz")
    args = ap.parse_args()
    cfg = dict(ball=0.019, flex=0.10, band=0.45, thumb_band=0.15, wrist_tilt=25.0, roll=20.0, ring=0.045, along=args.along,
               settle=1.5, spin=args.spin, far=0.07, min_r=0.03, crowd_w=20.0, seconds=args.seconds, K=None,
               palm_z=PALM_Z - 0.015,          # a rolled palm carries its balls lower than the flat palm's top
               sign=args.sign, posture={f"rh_A_{f}J3": args.j3 for f in ("FF", "MF", "RF", "LF")})
    w, state, center = make_world(cfg)
    x0 = np.array(json.loads(args.x)) if args.x else design()
    if args.try_:
        s, deg, lost_at, log = rollout(x0, w, state, center, cfg, record=True)
        print(f"score {s:.1f}  turns {np.round(deg).astype(int)}  {'kept' if lost_at is None else f'lost at {lost_at:.1f}s'}")
        save(args.out, log, center, x0, cfg)
        return
    sd = np.concatenate([[0.2], 0.08 * np.ones(4), 0.6 * np.ones(4), 0.04 * np.ones(4),
                         (0.15 if args.push else 0.0) * np.ones(1), [0.8 if args.push else 0.0],
                         (0.15 if args.push else 0.0) * np.ones(1), [0.8 if args.push else 0.0]])
    rng = np.random.default_rng(args.seed)
    mu, best, t0 = x0.copy(), (-np.inf, None, None, None), time.time()
    with Pool(args.procs, initializer=_init, initargs=(cfg,)) as pool:
        for it in range(args.iters):
            xs = [mu] + [mu + sd * rng.standard_normal(len(mu)) for _ in range(args.pop - 1)]
            if args.push:
                for x in xs[1:]:
                    x[[13, 15]] = np.maximum(x[[13, 15]], 0.0)
            res = pool.map(evaluate, xs)
            sc = np.array([r[0] for r in res])
            order = np.argsort(-sc)
            if sc[order[0]] > best[0]:
                best = (sc[order[0]], xs[order[0]].copy(), res[order[0]][1], res[order[0]][2])
            el = np.array([xs[i] for i in order[:args.elite]])
            mu, sd = el.mean(0), 0.7 * sd + 0.3 * el.std(0)
            lost = sum(r[2] is not None for r in res)
            print(f"iter {it + 1:3d}  best {best[0]:7.1f} (turns {np.round(best[2]).astype(int)}, "
                  f"{'kept' if best[3] is None else f'lost at {best[3]:.1f}s'}, T {np.exp(best[1][0]):.2f}s)  "
                  f"this iter top {sc[order[0]]:7.1f} median {np.median(sc):7.1f} lost {lost}/{len(res)} ({time.time() - t0:.0f}s)",
                  flush=True)
            np.save(args.out.replace(".npz", "_params.npy"), best[1])
    for secs, out in ((args.seconds, args.out), (3 * args.seconds, args.out.replace(".npz", "_long.npz"))):
        c = dict(cfg, seconds=secs)
        s, deg, lost_at, log = rollout(best[1], w, state, center, c, record=True)
        save(out, log, center, best[1], c)
        print(f"{secs:.0f} s: turns {np.round(deg).astype(int)} deg, {'kept all' if lost_at is None else f'lost at {lost_at:.1f} s'} -> {out}")
    print("params", json.dumps([round(float(v), 4) for v in best[1]]))


def save(out, log, center, x, cfg):
    np.savez(out, **{k: np.array(v) for k, v in log.items()}, center=center,
             rc=float(np.mean(np.linalg.norm(log["balls"][0][:, :2] - center, axis=1))), sign=1.0, ball=cfg["ball"],
             params=x, cfg=json.dumps({k: v for k, v in cfg.items() if k != "K"}))


if __name__ == "__main__":
    main()
