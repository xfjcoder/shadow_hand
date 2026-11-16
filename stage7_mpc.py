"""Stage 7A, open-hand loop: a RECEDING-HORIZON planner (sampling MPC). The segment planner
(`stage7_teacher.py`) plays a whole 1.5-2 s plan blind and then holds still -- the loop's momentum dies there
and the hands never fall into a rhythm. Here every `--exec` s (0.1) a CEM re-plans the next `--horizon` s (2.0)
from where the balls really are, warm-started from the last plan shifted on, and only the plan's first
`--exec` s is played: feedback and continuous motion.

A plan: control targets for all actuators at knots every `--knot` s, linearly interpolated from the current
controls. Its cost, per control step over the horizon:
    mean over balls of (lag / 10 deg)^2     each ball's lag behind a reference turning at `--rate` deg/s from
                                            the balls' mean phase now (so: advance, and all four TOGETHER)
    + W_RING * cm off the ring + W_CROWD * cm two balls come closer than 1cm + W_EFF * control change
and a lost ball (dropped, beyond FAR, within MIN_R of the centre) costs LOSS and ends the rollout.
The scene is the open-hand one (38mm balls, two per palm, fingers within 45% of open, thumbs 15%, wrists
+-25 deg, forearm roll +-20 deg) -- the same as `rl_env_stage7_loop.LoopEnv`, so the run is a demo for it.
A lost ball in the real run backtracks `--back` s and re-plans with a fresh seed and twice the samples.

Usage:
    python3 stage7_mpc.py --rate 10 --degrees 360 --out data/stage7_mpc.npz
"""
import argparse
import time
from multiprocessing import Pool

import mujoco as mj
import numpy as np

from stage7_layout import PALM_Z
from stage7_teacher import CTRL_DT, World

OPEN = {f"rh_A_{f}J{j}": v for f in ("FF", "MF", "RF", "LF") for j, v in ((3, 0.1), (0, 0.1))}
SPEC = mj.mjtState.mjSTATE_INTEGRATION
FAR, MIN_R = 0.07, 0.03
W_RING, W_CROWD, W_EFF, LOSS = 0.3, 2.0, 0.5, 1000.0
_w, _cfg = None, None


def make_world(ring=0.045, along=0.33):
    w = World(ball=0.019, tilt=np.radians(25.0), flex=0.10, posture=OPEN, band=0.45, roll=np.radians(20.0), thumb_band=0.15)
    w.settle(along=along, ring=ring, seconds=1.5, phase=np.radians(45))
    return w


def _init(cfg):
    global _w, _cfg, FAR, MIN_R
    _w, _cfg = make_world(cfg["ring"], cfg["along"]), cfg
    FAR, MIN_R = cfg["far"], cfg["min_r"]


def plan_controls(u_now, knots, t, knot_dt):
    """Controls at time t into the plan: linear from u_now through the knots (the first knot at knot_dt)."""
    pts = np.vstack([u_now, knots])
    f = min(t / knot_dt, len(knots))
    j = min(int(f), len(knots) - 1)
    a = f - j
    return (1 - a) * pts[j] + a * pts[j + 1]


def rollout_cost(job):
    """Cost of one plan from `state` (the worker's world)."""
    state, u_now, knots, center, rc, e0, cfg = job
    w = _w
    m, d = w.m, w.d
    mj.mj_setState(m, d, state, SPEC)
    per = int(round(CTRL_DT / m.opt.timestep))
    n = int(round(cfg["horizon"] / CTRL_DT))
    om = np.radians(cfg["rate"]) * CTRL_DT
    th_prev = np.arctan2(*(w.balls()[:, :2] - center)[:, ::-1].T)
    dth = np.zeros(4)
    cost, u_prev = 0.0, u_now
    for i in range(n):
        u = np.clip(plan_controls(u_now, knots, (i + 1) * CTRL_DT, cfg["knot"]), w.lo, w.hi)
        d.ctrl[:] = u
        for _ in range(per):
            mj.mj_step(m, d)
        b = w.balls()
        th = np.arctan2(b[:, 1] - center[1], b[:, 0] - center[0])
        dth += np.angle(np.exp(1j * (th - th_prev)))
        th_prev = th
        lag = np.degrees(e0 + dth - om * (i + 1))      # each ball vs the common reference (negative: behind)
        r = np.linalg.norm(b[:, :2] - center, axis=1)
        g = np.array([np.linalg.norm(b[p, :2] - b[q, :2]) * 100 - 200 * w.r for p in range(4) for q in range(p + 1, 4)])
        cost += (float(np.mean((lag / 10.0) ** 2)) + W_RING * float(np.sum(np.abs(r - rc))) * 100
                 + W_CROWD * float(np.sum(np.maximum(1.0 - g, 0.0))) + W_EFF * float(np.sum(np.abs(u - u_prev))))
        u_prev = u
        if np.any(b[:, 2] < PALM_Z) or np.any(r > FAR) or np.any(r < MIN_R):
            return cost + LOSS * (1.0 + (n - i) / n)
    return cost


def main():
    global FAR, MIN_R
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=10.0, help="deg/s the loop should turn at")
    ap.add_argument("--degrees", type=float, default=360.0)
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--knot", type=float, default=0.25)
    ap.add_argument("--exec", type=float, default=0.1, help="s of each plan played before re-planning")
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--spread", type=float, default=0.08, help="CEM std, fraction of each actuator's range")
    ap.add_argument("--lead", type=float, default=0.0,
                    help="deg: the reference accumulates at --rate and may lead the balls' mean phase by up to this "
                         "(0: it restarts at the mean phase every re-plan -- a weak pull: mpc1 crept at ~2.5 deg/s)")
    ap.add_argument("--spin", type=float, default=0.0, help="deg/s the balls start rolling round at")
    ap.add_argument("--back", type=float, default=1.5, help="s to back up after a lost ball")
    ap.add_argument("--backtracks", type=int, default=40)
    ap.add_argument("--max-seconds", type=float, default=200.0, help="cap on simulated time")
    ap.add_argument("--ring", type=float, default=0.045, help="m: start ring radius (0.035 at --along 0.30: the crossings "
                                                              "fall mid-palm; at 0.045/0.33 they fell on the little fingers and heels)")
    ap.add_argument("--along", type=float, default=0.33)
    ap.add_argument("--far", type=float, default=FAR)
    ap.add_argument("--min-r", type=float, default=MIN_R)
    ap.add_argument("--procs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/stage7_mpc.npz")
    args = ap.parse_args()
    FAR, MIN_R = args.far, args.min_r
    cfg = dict(horizon=args.horizon, knot=args.knot, rate=args.rate, ring=args.ring, along=args.along, far=args.far, min_r=args.min_r)
    rng = np.random.default_rng(args.seed)
    w = make_world(args.ring, args.along)
    m, d = w.m, w.d
    b0 = w.balls()
    center = b0[:, :2].mean(0)
    rc = float(np.linalg.norm(b0[:, :2] - center, axis=1).mean())
    if args.spin:
        om = np.radians(args.spin)
        for q, v in zip(w.bq, w.bv):
            d.qvel[v:v + 2] = [-om * (d.qpos[q + 1] - center[1]), om * (d.qpos[q] - center[0])]
        mj.mj_forward(m, d)
    span = w.hi - w.lo
    nk = int(round(args.horizon / args.knot))
    n_exec = int(round(args.exec / CTRL_DT))
    per = int(round(CTRL_DT / m.opt.timestep))
    mean = np.tile(d.ctrl.copy(), (nk, 1))
    phase = np.zeros(4)                                  # each ball's unwrapped turn since the start
    ref = 0.0                                            # the accumulating reference (with --lead)
    th_prev = np.arctan2(*(w.balls()[:, :2] - center)[:, ::-1].T)
    log = {k: [] for k in ("ctrl", "balls", "phase", "qpos", "qvel")}
    snaps = []                                           # (state, mean, phase, th_prev, log length) every exec
    backtracks, boost, t0 = 0, 1, time.time()
    state = np.empty(mj.mj_stateSize(m, SPEC))
    with Pool(args.procs, initializer=_init, initargs=(cfg,)) as pool:
        while len(log["ctrl"]) * CTRL_DT < args.max_seconds:
            mj.mj_getState(m, d, state, SPEC)
            snaps.append((state.copy(), mean.copy(), phase.copy(), th_prev.copy(), len(log["ctrl"])))
            snaps = snaps[-int(5.0 / args.exec):]
            if args.lead:
                ref = min(ref + np.radians(args.rate) * args.exec, phase.mean() + np.radians(args.lead))
                e0 = phase - ref                         # each ball's lag behind the accumulated reference
            else:
                e0 = phase - phase.mean()                # deviations from the mean phase (the reference starts at it)
            u_now = d.ctrl.copy()
            mu, sd = mean.copy(), np.tile(args.spread * span, (nk, 1))
            best = (np.inf, mean)
            for _ in range(args.iters):
                cands = [mu] + [mu + sd * rng.standard_normal(mu.shape) for _ in range(args.pop * boost - 1)]
                cands = [np.clip(c, w.lo, w.hi) for c in cands]
                costs = np.array(pool.map(rollout_cost, [(state, u_now, c, center, rc, e0, cfg) for c in cands]))
                order = np.argsort(costs)
                if costs[order[0]] < best[0]:
                    best = (costs[order[0]], cands[order[0]])
                el = np.array([cands[i] for i in order[:args.elite]])
                mu, sd = el.mean(0), np.maximum(0.5 * sd + 0.5 * el.std(0), 0.01 * span)
            plan = best[1]
            lost = False
            for i in range(n_exec):                      # play the plan's first `exec` s
                d.ctrl[:] = np.clip(plan_controls(u_now, plan, (i + 1) * CTRL_DT, args.knot), w.lo, w.hi)
                for _ in range(per):
                    mj.mj_step(m, d)
                b = w.balls()
                th = np.arctan2(b[:, 1] - center[1], b[:, 0] - center[0])
                phase += np.angle(np.exp(1j * (th - th_prev)))
                th_prev = th
                for k, v in (("ctrl", d.ctrl.copy()), ("balls", b.copy()), ("phase", float(phase.mean())),
                             ("qpos", d.qpos.copy()), ("qvel", d.qvel.copy())):
                    log[k].append(v)
                r = np.linalg.norm(b[:, :2] - center, axis=1)
                if np.any(b[:, 2] < PALM_Z) or np.any(r > FAR) or np.any(r < MIN_R):
                    lost = True
                    break
            # the warm start: the plan's knots shifted on by the time played
            ts = args.exec + args.knot * np.arange(1, nk + 1)
            mean = np.array([plan_controls(u_now, plan, t, args.knot) for t in ts])
            if lost:
                if backtracks >= args.backtracks or len(snaps) < 2:
                    print("lost a ball, no backtracks left", flush=True)
                    break
                k = max(len(snaps) - 1 - int(round(args.back / args.exec)), 0)
                st, mean, phase, th_prev, nlog = snaps[k]
                phase, th_prev, mean = phase.copy(), th_prev.copy(), mean + 0.3 * args.spread * span * rng.standard_normal(mean.shape)
                mj.mj_setState(m, d, st, SPEC)
                mj.mj_forward(m, d)
                for key in log:
                    del log[key][nlog:]
                del snaps[k:]
                backtracks += 1
                boost = 2
                print(f"  lost a ball at {len(log['ctrl']) * CTRL_DT:.1f} s: back {args.back:g} s (backtrack {backtracks}/{args.backtracks})",
                      flush=True)
                continue
            boost = 1
            tsim = len(log["ctrl"]) * CTRL_DT
            if int(tsim / args.exec) % 10 == 0:
                deg = np.degrees(phase)
                print(f"t {tsim:6.1f} s  turned {deg.mean():+7.1f} deg (balls {np.round(deg).astype(int)})  "
                      f"rate {deg.mean() / max(tsim, 1e-9):5.1f} deg/s  cost {best[0]:8.1f}  ({time.time() - t0:.0f}s)", flush=True)
                np.savez(args.out, **{k: np.array(v) for k, v in log.items()}, center=center, rc=rc, sign=1.0, ball=0.019)
            if np.degrees(phase).min() >= args.degrees:
                break
    np.savez(args.out, **{k: np.array(v) for k, v in log.items()}, center=center, rc=rc, sign=1.0, ball=0.019)
    deg = np.degrees(phase)
    print(f"done: {len(log['ctrl']) * CTRL_DT:.1f} s, balls turned {np.round(deg).astype(int)} deg, {backtracks} backtracks -> {args.out}")


if __name__ == "__main__":
    main()
