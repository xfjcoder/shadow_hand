"""Stage 7A, open-hand loop: search for a RHYTHM -- every actuator follows a smooth periodic wave,
    u(t) = u0 + sum_k a_k sin(2 pi k t / T) + b_k cos(2 pi k t / T)     (k = 1..K, amplitudes ramped in over a cycle)
so the two hands move in one shared rhythm, and a CEM tunes the amplitudes and phases (and the period) for
the balls rolling round the loop together: the score is the SLOWEST ball's turn (all four must keep going),
less the spread between the balls' turns, crowding (two balls closer than 1cm) and a ball lost (dropped,
too far out, or through the middle). The segment planner (`stage7_teacher.py`) could creep the loop round
but not keep it rolling -- it holds still at every segment's end.

The scene is the open-hand one: 38mm balls, two per palm on a 4.5cm ring, fingers within `--band` of open,
thumbs within `--thumb-band`, wrists +-`--wrist-tilt` deg, and the forearm roll joints (+-`--roll` deg)
that tilt the palms toward each other. `--spin` starts the balls already rolling round (deg/s).
Saves the best rhythm (params) and its rollout (an npz render_stage7.py can draw).

Usage:
    python3 stage7_rhythm.py --period 1.5 --seconds 20 --iters 60 --out data/stage7_rhythm_p15.npz
"""
import argparse
import json
import time
from multiprocessing import Pool

import mujoco as mj
import numpy as np

from stage7_teacher import CTRL_DT, World

OPEN = {f"rh_A_{f}J{j}": v for f in ("FF", "MF", "RF", "LF") for j, v in ((3, 0.1), (0, 0.1))}
_w, _cfg = None, None
PERIOD = (0.8, 2.5)          # s: the rhythm must REPEAT within the scored rollout (an 8 s search drifted to one 6.9 s sweep
                             # that dropped a ball right after the scored 8 s)


def make_world(cfg):
    w = World(ball=cfg["ball"], tilt=np.radians(cfg["wrist_tilt"]), flex=cfg["flex"], posture=dict(OPEN, **cfg.get("posture", {})),
              band=cfg["band"],
              roll=np.radians(cfg["roll"]), thumb_band=cfg["thumb_band"])
    w.settle(along=cfg["along"], ring=cfg["ring"], seconds=cfg["settle"], phase=np.radians(45))
    b = w.balls()
    center = b[:, :2].mean(0)
    if cfg["spin"]:                                     # the loop's momentum: each ball rolling round the centre
        om = np.radians(cfg["spin"])
        for q, v in zip(w.bq, w.bv):
            dx, dy = w.d.qpos[q] - center[0], w.d.qpos[q + 1] - center[1]
            w.d.qvel[v:v + 2] = [-om * dy, om * dx]
    state = np.empty(mj.mj_stateSize(w.m, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(w.m, w.d, state, mj.mjtState.mjSTATE_INTEGRATION)
    return w, state, center


def _init(cfg):
    global _w, _cfg
    _cfg = cfg
    _w = make_world(cfg)


def controls(x, t, w, K):
    """The rhythm's controls at time t: x = [log period, a (K x nu), b (K x nu)]."""
    nu = w.m.nu
    T = float(np.clip(np.exp(x[0]), *PERIOD))
    a, b = x[1:1 + K * nu].reshape(K, nu), x[1 + K * nu:].reshape(K, nu)
    ramp = min(t / T, 1.0)
    u = w.u0.copy()
    for k in range(K):
        u += ramp * (a[k] * np.sin(2 * np.pi * (k + 1) * t / T) + b[k] * np.cos(2 * np.pi * (k + 1) * t / T))
    return np.clip(u, w.lo, w.hi)


def rollout(x, w, state, center, cfg, record=False):
    m, d = w.m, w.d
    mj.mj_setState(m, d, state, mj.mjtState.mjSTATE_INTEGRATION)
    per = int(round(CTRL_DT / m.opt.timestep))
    n = int(round(cfg["seconds"] / CTRL_DT))
    th_prev = np.arctan2(*(w.balls()[:, :2] - center)[:, ::-1].T)
    turn, crowd, lost_at = np.zeros(4), 0.0, None
    log = {k: [] for k in ("ctrl", "balls", "phase", "qpos", "qvel")}
    for i in range(n):
        d.ctrl[:] = controls(x, i * CTRL_DT, w, cfg["K"])
        for _ in range(per):
            mj.mj_step(m, d)
        b = w.balls()
        th = np.arctan2(b[:, 1] - center[1], b[:, 0] - center[0])
        turn += np.angle(np.exp(1j * (th - th_prev)))
        th_prev = th
        r = np.linalg.norm(b[:, :2] - center, axis=1)
        g = [np.linalg.norm(b[p, :2] - b[q, :2]) * 100 - 200 * w.r for p in range(4) for q in range(p + 1, 4)]
        crowd += float(np.sum(np.maximum(1.0 - np.array(g), 0.0))) * CTRL_DT
        if record:
            for k, v in (("ctrl", d.ctrl.copy()), ("balls", b.copy()), ("phase", float(turn.mean())),
                         ("qpos", d.qpos.copy()), ("qvel", d.qvel.copy())):
                log[k].append(v)
        if np.any(b[:, 2] < cfg["palm_z"]) or np.any(r > cfg["far"]) or np.any(r < cfg["min_r"]):
            lost_at = (i + 1) * CTRL_DT
            break
    deg = np.degrees(turn)
    score = (cfg.get("sign", 1.0) * deg).min() - 0.5 * deg.std() - cfg["crowd_w"] * crowd
    if lost_at is not None:                              # a loss: worse than any full run, better the later it comes
        score -= 200.0 * (1.0 - lost_at / cfg["seconds"]) + 50.0
    return score, deg, lost_at, log


def evaluate(x):
    w, state, center = _w
    s, deg, lost_at, _ = rollout(x, w, state, center, _cfg)
    return s, deg, lost_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=float, default=1.5, help="s, the rhythm's starting period (also searched)")
    ap.add_argument("--fix-period", action="store_true", help="do not search the period")
    ap.add_argument("--K", type=int, default=1, help="harmonics")
    ap.add_argument("--seconds", type=float, default=20.0, help="rollout length scored (several periods)")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--elite", type=int, default=12)
    ap.add_argument("--amp", type=float, default=0.15, help="CEM start std of each amplitude, fraction of the range")
    ap.add_argument("--procs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ball", type=float, default=0.019)
    ap.add_argument("--flex", type=float, default=0.10)
    ap.add_argument("--band", type=float, default=0.45)
    ap.add_argument("--thumb-band", type=float, default=0.15)
    ap.add_argument("--wrist-tilt", type=float, default=25.0)
    ap.add_argument("--roll", type=float, default=20.0)
    ap.add_argument("--ring", type=float, default=0.045)
    ap.add_argument("--along", type=float, default=0.33)
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--spin", type=float, default=30.0, help="deg/s the balls start rolling round at (0: from rest)")
    ap.add_argument("--far", type=float, default=0.07)
    ap.add_argument("--min-r", type=float, default=0.03)
    ap.add_argument("--crowd-w", type=float, default=20.0)
    ap.add_argument("--out", default="data/stage7_rhythm.npz")
    args = ap.parse_args()
    from stage7_layout import PALM_Z
    cfg = dict(ball=args.ball, flex=args.flex, band=args.band, thumb_band=args.thumb_band, wrist_tilt=args.wrist_tilt,
               roll=args.roll, ring=args.ring, along=args.along, settle=args.settle, spin=args.spin, far=args.far,
               min_r=args.min_r, crowd_w=args.crowd_w, seconds=args.seconds, K=args.K, palm_z=PALM_Z)
    w, state, center = make_world(cfg)
    nu = w.m.nu
    span = w.hi - w.lo
    rng = np.random.default_rng(args.seed)
    dim = 1 + 2 * args.K * nu
    mu = np.zeros(dim)
    mu[0] = np.log(args.period)
    sd = np.concatenate([[0.0 if args.fix_period else 0.15], np.tile(args.amp * span, 2 * args.K)])
    best = (-np.inf, None, None, None)
    t0 = time.time()
    with Pool(args.procs, initializer=_init, initargs=(cfg,)) as pool:
        for it in range(args.iters):
            xs = [mu] + [mu + sd * rng.standard_normal(dim) for _ in range(args.pop - 1)]
            res = pool.map(evaluate, xs)
            sc = np.array([r[0] for r in res])
            order = np.argsort(-sc)
            if sc[order[0]] > best[0]:
                best = (sc[order[0]], xs[order[0]].copy(), res[order[0]][1], res[order[0]][2])
            el = np.array([xs[i] for i in order[:args.elite]])
            mu, sd = el.mean(0), np.maximum(0.7 * sd + 0.3 * el.std(0), 0.005 * np.concatenate([[1.0], np.tile(span, 2 * args.K)]))
            if args.fix_period:
                sd[0] = 0.0
            lost = sum(r[2] is not None for r in res)
            print(f"iter {it + 1:3d}  best {best[0]:7.1f}  (turns {np.round(best[2]).astype(int)}, "
                  f"{'kept' if best[3] is None else f'lost at {best[3]:.1f}s'}, period {np.clip(np.exp(best[1][0]), *PERIOD):.2f}s)  "
                  f"this iter: top {sc[order[0]]:7.1f} median {np.median(sc):7.1f}  lost {lost}/{len(res)}  ({time.time() - t0:.0f}s)",
                  flush=True)
            np.save(args.out.replace(".npz", "_params.npy"), best[1])
    # the best rhythm's rollout, scored length and a longer one, for rendering and a stability check
    for secs, out in ((args.seconds, args.out), (3 * args.seconds, args.out.replace(".npz", "_long.npz"))):
        c = dict(cfg, seconds=secs)
        s, deg, lost_at, log = rollout(best[1], w, state, center, c, record=True)
        np.savez(out, **{k: np.array(v) for k, v in log.items()}, center=center,
                 rc=float(np.mean(np.linalg.norm(log["balls"][0][:, :2] - center, axis=1))), sign=1.0, ball=args.ball,
                 params=best[1], cfg=json.dumps(cfg))
        print(f"{secs:.0f} s rollout: turns {np.round(deg).astype(int)} deg, "
              f"{'kept all' if lost_at is None else f'lost at {lost_at:.1f} s'} -> {out}")


if __name__ == "__main__":
    main()
