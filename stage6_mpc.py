"""Stage 6 (rebuild, step 3b): closed-loop sampling MPC (predictive sampling, Howell et al. 2022) that
makes each ball TRACK its own target moving round a fixed circle (the MyoChallenge tracking objective).

Every `--replan` s, from the true state, `--samples` finger plans (piecewise-linear knots every KNOT s
over `--horizon` s; the shifted previous best plus Gaussian noise) are rolled out on the MuJoCo model;
the cheapest is kept and its first `--replan` s executed. Cost per control step: each ball's distance
(cm) to its own target, targets at phase(t) round `--center`, plus a big penalty once a ball escapes.
The commanded phase advances at `--rate` deg/s but never more than LEAD ahead of the trio (it waits at a
hump instead of running away). The wrist is held unless `--wrist-tilt` lets it tilt the palm a little. Saves a log (npz, for distillation) and a GIF.

Usage:
    python3 stage6_mpc.py --rate 30 --degrees 720 --out data/stage6_mpc.npz --gif media_full/stage6_mpc.gif
"""
import argparse
import time
from multiprocessing import Pool

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

from eval_stage6_circle_hold import WRIST, Probe

CTRL_DT = 0.02
KNOT = 0.2
LEAD = np.radians(35.0)
SPEC = mj.mjtState.mjSTATE_INTEGRATION
SLICK = "assets/scene_three_balls_slick.xml"

_pr = None


def _init(radius, scene, tilt):
    global _pr, _lo, _hi
    _pr = Probe(radius, scene)
    _lo, _hi = limits(_pr, tilt)


def limits(pr, tilt):
    """Actuator limits: fingers their full range, the wrist within `tilt` rad of the bowl posture."""
    lo, hi = pr.lo.copy(), pr.hi.copy()
    lo[:WRIST] = np.maximum(lo[:WRIST], pr.target0[:WRIST] - tilt)
    hi[:WRIST] = np.minimum(hi[:WRIST], pr.target0[:WRIST] + tilt)
    return lo, hi


def ctrl_at(knots, t):
    """Piecewise-linear finger targets at time t (s from plan start), knots at 0, KNOT, 2 KNOT, ..."""
    u = min(t / KNOT, len(knots) - 1.0)
    i = min(int(u), len(knots) - 2)
    a = u - i
    return (1 - a) * knots[i] + a * knots[i + 1]


def escaped(b):
    return np.any(np.abs(b[:, 1] + 0.034) > 0.035) or np.any(np.linalg.norm(b - b.mean(0), axis=1) > 0.042)


def rollout(job):
    """-> cost of `knots` from `state`: per-step labeled tracking error + escape penalty."""
    state, knots, horizon, center, slots, phase0, rate, sign, wrist_w = job
    pr = _pr
    mj.mj_setState(pr.m, pr.d, state, SPEC)
    per = int(round(CTRL_DT / pr.m.opt.timestep))
    n = int(round(horizon / CTRL_DT))
    total = 0.0
    for k in range(n):
        pr.d.ctrl[:] = np.clip(ctrl_at(knots, k * CTRL_DT), _lo, _hi)
        for _ in range(per):
            mj.mj_step(pr.m, pr.d)
        b = np.array(pr.palm())
        if escaped(b):
            return total + 200.0 * (n - k) / n + 100.0
        ph = phase0 + sign * rate * (k + 1) * CTRL_DT
        tg = pr.places(center, ph)
        e = np.array([np.linalg.norm(b[i] - tg[slots[i]]) for i in range(3)]) * 100
        total += (e.mean() + 0.5 * e.max() + wrist_w * np.abs(pr.d.ctrl[:WRIST] - pr.target0[:WRIST]).sum()) / n
    v = sum(np.linalg.norm(pr.d.joint(pr.m.body(x).jntadr[0]).qvel[:3]) for x in ("ball1", "ball2", "ball3"))
    return total + 0.5 * v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", default="0.1,6.5")
    ap.add_argument("--radius", type=float, default=0.0225)
    ap.add_argument("--scene", default=SLICK)
    ap.add_argument("--table", default="data/stage6_circle_hold_slick.npz")
    ap.add_argument("--rate", type=float, default=30.0, help="deg/s of the moving targets")
    ap.add_argument("--sign", type=float, default=1.0)
    ap.add_argument("--degrees", type=float, default=720.0)
    ap.add_argument("--seconds", type=float, default=90.0, help="give up after this much sim time")
    ap.add_argument("--horizon", type=float, default=0.8)
    ap.add_argument("--replan", type=float, default=0.1)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--noise", type=float, default=0.08, help="knot noise std, fraction of each finger's range")
    ap.add_argument("--procs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wrist-tilt", type=float, default=0.0, help="deg each wrist joint may move from the bowl posture (0: fixed)")
    ap.add_argument("--wrist-weight", type=float, default=2.0, help="cost (cm-equivalent) per rad of wrist deviation")
    ap.add_argument("--out", default="data/stage6_mpc.npz")
    ap.add_argument("--gif", default=None)
    args = ap.parse_args()
    center = np.array([float(v) for v in args.center.split(",")]) / 100
    rate = np.radians(args.rate)
    rng = np.random.default_rng(args.seed)

    pr = Probe(args.radius, args.scene)
    z = np.load(args.table)
    sel = np.all(np.isclose(z["centers"], center * 100), axis=1)
    pr.rollout(z["fingers"][sel][np.argmin(z["phases"][sel])], pr.places(center, 0.0), hold=1.0)
    b = np.array(pr.palm())
    th = np.arctan2(b[:, 0] - center[0], b[:, 2] - center[1])
    phase = np.angle(np.mean(np.exp(3j * th))) / 3              # the trio's starting phase (mod 120)
    slot_pos = pr.places(center, phase)
    slots = [int(np.argmin([np.linalg.norm(x - s) for s in slot_pos])) for x in b]
    phase_start = phase

    lo, hi = limits(pr, np.radians(args.wrist_tilt))
    span = hi - lo
    span[:WRIST] = np.maximum(span[:WRIST], 1e-9)
    nk = int(round(args.horizon / KNOT)) + 1
    plan = np.tile(pr.d.ctrl.copy(), (nk, 1))
    state = np.empty(mj.mj_stateSize(pr.m, SPEC))
    per = int(round(CTRL_DT / pr.m.opt.timestep))
    steps_per_plan = int(round(args.replan / CTRL_DT))

    log = dict(ctrl=[], balls=[], phase=[], qpos=[], qvel=[])
    renderer = mj.Renderer(pr.m, height=240, width=320) if args.gif else None
    frames = []
    turned, t, lost = 0.0, 0.0, False
    th_prev = th.copy()
    t0 = time.time()
    with Pool(args.procs, initializer=_init, initargs=(args.radius, args.scene, np.radians(args.wrist_tilt))) as pool:
        while t < args.seconds and abs(turned) < np.radians(args.degrees):
            mj.mj_getState(pr.m, pr.d, state, SPEC)
            # the command waits for the trio: never more than LEAD ahead of the balls' own turn
            lead_ok = args.sign * (phase - (phase_start + turned)) < LEAD
            cands = [plan] + [plan + args.noise * span * rng.standard_normal(plan.shape) for _ in range(args.samples - 1)]
            r = rate if lead_ok else 0.0
            costs = pool.map(rollout, [(state, c, args.horizon, center, slots, phase, r, args.sign, args.wrist_weight) for c in cands])
            plan = cands[int(np.argmin(costs))]
            for k in range(steps_per_plan):
                pr.d.ctrl[:] = np.clip(ctrl_at(plan, k * CTRL_DT), lo, hi)
                for _ in range(per):
                    mj.mj_step(pr.m, pr.d)
                phase += args.sign * r * CTRL_DT
                t += CTRL_DT
                b = np.array(pr.palm())
                th = np.arctan2(b[:, 0] - center[0], b[:, 2] - center[1])
                turned += np.mean(np.angle(np.exp(1j * (th - th_prev))))
                th_prev = th
                log["ctrl"].append(pr.d.ctrl.copy())
                log["balls"].append(b)
                log["phase"].append(phase)
                log["qpos"].append(pr.d.qpos.copy())
                log["qvel"].append(pr.d.qvel.copy())
                if renderer is not None and len(log["ctrl"]) % 3 == 0:
                    R = pr.d.body("rh_palm").xmat.reshape(3, 3)
                    cam = mj.MjvCamera()
                    cam.type = mj.mjtCamera.mjCAMERA_FREE
                    cam.lookat[:] = pr.d.body("rh_palm").xpos + R @ np.array([0.0, -0.04, 0.07])
                    imgs = []
                    for az, el in ((180, -60), (220, -35)):
                        cam.distance, cam.azimuth, cam.elevation = 0.25, az, el
                        renderer.update_scene(pr.d, camera=cam)
                        imgs.append(renderer.render().copy())
                    img = Image.fromarray(np.concatenate(imgs, 1))
                    ImageDraw.Draw(img).text((8, 6), f"MPC, {'fingers only' if args.wrist_tilt == 0 else f'wrist tilt <= {args.wrist_tilt:.0f} deg'}  t={t:.1f}s  target {np.degrees(phase):+.0f} deg  "
                                                     f"balls turned {np.degrees(turned):+.0f} deg = {abs(turned) / 2 / np.pi:.2f} circles",
                                             fill=(255, 255, 255))
                    frames.append(img)
                if escaped(b):
                    lost = True
                    break
            if lost:
                break
            # shift the plan by `replan` s (hold the last knot)
            ts = np.arange(nk) * KNOT + args.replan
            plan = np.array([ctrl_at(plan, s) for s in ts])
            if len(log["ctrl"]) % 50 == 0:
                tg = pr.places(center, phase)
                e = np.array([np.linalg.norm(b[i] - tg[slots[i]]) for i in range(3)]) * 100
                print(f"t {t:5.1f}s  target {np.degrees(phase):+6.0f}  turned {np.degrees(turned):+6.0f} deg  "
                      f"off target mean {e.mean():.2f} max {e.max():.2f} cm  ({time.time() - t0:.0f}s)", flush=True)
                np.savez(args.out, **{k: np.array(v) for k, v in log.items()}, center=center, sign=args.sign)
    np.savez(args.out, **{k: np.array(v) for k, v in log.items()}, center=center, sign=args.sign)
    print(f"END t {t:.1f}s turned {np.degrees(turned):+.0f} deg = {abs(turned) / 2 / np.pi:.2f} circles"
          + ("  LOST a ball" if lost else ""), flush=True)
    if renderer is not None:
        renderer.close()
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=60, loop=0)
        print(f"saved {len(frames)} frames to {args.gif}")


if __name__ == "__main__":
    main()
