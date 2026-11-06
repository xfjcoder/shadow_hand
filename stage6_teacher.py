"""Stage 6 (rebuild, step 3): a sampling-MPC "teacher" that turns the three balls round a fixed circle.

Step 1 showed every placement on the circle is holdable; step 2 that blending the holding postures does
not carry the balls between them. Here each SEGMENT is planned on the MuJoCo model itself (as in MuJoCo
MPC / predictive sampling, Howell et al. 2022; Hess et al. 2024): from the trio's actual state, a CEM over
`--waypoints` finger-target waypoints (each ramped over SEG s, then HOLD s at the last) minimises the
balls' distance to the circle's places at the next commanded phase (+`--step` deg), their speed at the
end, and a loss penalty. The best segment is executed, and the next is planned from where the balls
really are. The wrist never moves. Saves the per-control-step (20 ms) finger targets, ball states and
phases (an npz) for distillation, and a GIF.

Usage:
    python3 stage6_teacher.py --center 0.1,6.5 --degrees 720 --out data/stage6_teacher.npz --gif media_full/stage6_teacher.gif
"""
import argparse
import time
from multiprocessing import Pool

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

from eval_stage6_circle_hold import WRIST, Probe
from hand_env import classify_contacts
from stage6_mpc import limits

SEG, HOLD = 0.35, 0.35         # s per waypoint ramp; s held at the last waypoint
CTRL_DT = 0.02
SPEC = mj.mjtState.mjSTATE_INTEGRATION

_pr = None


LO = HI = None
TIP_W = 0.0                    # cm-equivalent per finger (not the thumb) pressing a ball with its tip, mean over the plan

FINGERS = ("ff", "mf", "rf", "lf")


def tips_on(pr):
    """Fingers (not the thumb) pressing a ball with the tip (`hand_env.classify_contacts`, >= 0.2 N)."""
    return len({dg for b in ("ball1", "ball2", "ball3") for dg, sg, face, fn in classify_contacts(pr.m, pr.d, b)
                if face == "tip" and dg in FINGERS and fn >= 0.2})


def _init(radius, scene, seg=None, hold=None, tilt=0.0, tip_w=0.0):
    global _pr, SEG, HOLD, LO, HI, TIP_W
    TIP_W = tip_w
    _pr = Probe(radius, scene)
    LO, HI = limits(_pr, tilt)
    SEG, HOLD = seg or SEG, hold or HOLD


def errors(pr, targets, labeled=False):
    """Per ball distance (m) to the nearest of `targets` (the balls are identical), or with `labeled`
    to its own target (targets in ball order: a whole "click" of 120 deg ends where it began, unlabeled)."""
    if labeled:
        return np.array([np.linalg.norm(b - t) for b, t in zip(pr.palm(), targets)])
    return np.array([min(np.linalg.norm(b - t) for t in targets) for b in pr.palm()])


def play(pr, state, start_ctrl, wps, record=None):
    """From `state`, ramp the fingers through `wps` (each over SEG), hold HOLD. -> lost flag."""
    mj.mj_setState(pr.m, pr.d, state, SPEC)
    n, h = int(SEG / pr.m.opt.timestep), int(HOLD / pr.m.opt.timestep)
    per = int(CTRL_DT / pr.m.opt.timestep)
    prev, k = start_ctrl, 0
    pr.tips, pr.tip_n = 0, 0
    for w in list(wps) + [None]:
        steps = n if w is not None else h
        for i in range(steps):
            u = prev if w is None else prev + (w - prev) * (i + 1) / steps
            pr.d.ctrl[:] = np.clip(u, LO, HI)
            mj.mj_step(pr.m, pr.d)
            k += 1
            if k % per == 0:
                if record is not None:
                    record(pr)
                if TIP_W:
                    pr.tips, pr.tip_n = pr.tips + tips_on(pr), pr.tip_n + 1
        if w is not None:
            prev = w
        b = np.array(pr.palm())
        if np.any(np.abs(b[:, 1] + 0.034) > 0.04) or np.any(np.linalg.norm(b - b.mean(0), axis=1) > 0.045):
            return True
    return False


def cost(job):
    state, start_ctrl, wps, targets, labeled = job
    lost = play(_pr, state, start_ctrl, wps)
    e = errors(_pr, targets, labeled) * 100
    v = sum(np.linalg.norm(_pr.d.joint(_pr.m.body(b).jntadr[0]).qvel[:3]) for b in ("ball1", "ball2", "ball3"))
    effort = sum(np.abs(w - p).sum() for w, p in zip(wps, [start_ctrl] + list(wps[:-1])))
    tip = TIP_W * _pr.tips / max(_pr.tip_n, 1)
    return e.mean() + 0.5 * e.max() + 10 * v + 0.02 * effort + tip + (100.0 if lost else 0.0), e.mean(), lost


def main():
    global SEG, HOLD, LO, HI, TIP_W
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", default="0.1,6.5")
    ap.add_argument("--radius", type=float, default=0.0225)
    ap.add_argument("--step", type=float, default=15.0, help="deg of commanded phase per segment")
    ap.add_argument("--degrees", type=float, default=720.0)
    ap.add_argument("--sign", type=float, default=1.0)
    ap.add_argument("--waypoints", type=int, default=3)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--table", default="data/stage6_circle_hold.npz", help="step 1 postures, for the start")
    ap.add_argument("--out", default="data/stage6_teacher.npz")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=15)
    ap.add_argument("--resume", default=None, help="an earlier run's npz to continue from")
    ap.add_argument("--resume-at", type=int, default=-1, help="the logged control step to continue from")
    ap.add_argument("--max-boost", type=int, default=4)
    ap.add_argument("--backtracks", type=int, default=0, help="dead ends (every plan loses a ball) to back out of")
    ap.add_argument("--retries", type=int, default=2, help="re-searches before running a plan that loses a ball")
    ap.add_argument("--wrist-tilt", type=float, default=0.0, help="deg each wrist joint may move from the bowl posture (0: fixed)")
    ap.add_argument("--labeled", action="store_true", help="each ball to its own target (needed for 120 deg clicks)")
    ap.add_argument("--seg", type=float, default=SEG, help="s per waypoint ramp")
    ap.add_argument("--hold", type=float, default=HOLD, help="s held at the last waypoint")
    ap.add_argument("--scene", default="assets/scene_three_balls.xml", help="assets/scene_three_balls_slick.xml: polished balls")
    ap.add_argument("--tip-weight", type=float, default=0.0,
                    help="cost per finger pressing a ball with its tip (mean over the plan); the user: ring/middle tips 'not very elegant'")
    ap.add_argument("--spread", type=float, default=0.25, help="CEM start std, fraction of each finger's range")
    args = ap.parse_args()
    SEG, HOLD, TIP_W = args.seg, args.hold, args.tip_weight
    center = np.array([float(v) for v in args.center.split(",")]) / 100
    rng = np.random.default_rng(args.seed)

    pr = Probe(args.radius, args.scene)
    z = np.load(args.table)
    sel = np.all(np.isclose(z["centers"], center * 100), axis=1)
    start = z["fingers"][sel][np.argmin(z["phases"][sel])]
    pr.rollout(start, pr.places(center, 0.0), hold=1.0)
    if args.resume:                                   # continue from a logged step of an earlier run
        r = np.load(args.resume)
        pr.d.qpos[:], pr.d.qvel[:] = r["qpos"][args.resume_at], r["qvel"][args.resume_at]
        c = r["ctrl"][args.resume_at]
        pr.d.ctrl[-len(c):] = c
        mj.mj_forward(pr.m, pr.d)
    state = np.empty(mj.mj_stateSize(pr.m, SPEC))
    mj.mj_getState(pr.m, pr.d, state, SPEC)
    LO, HI = limits(pr, np.radians(args.wrist_tilt))
    ctrl = pr.d.ctrl.copy()

    span = np.maximum(HI - LO, 1e-9)
    log = dict(ctrl=[], balls=[], phase=[], qpos=[], qvel=[])
    renderer = mj.Renderer(pr.m, height=240, width=320) if args.gif else None
    frames = []
    phase, n_seg = 0.0, 4 * int(round(args.degrees / args.step))
    t0 = time.time()

    def record(p):
        log["ctrl"].append(p.d.ctrl.copy())
        log["balls"].append(np.array(p.palm()))
        log["phase"].append(phase)
        log["qpos"].append(p.d.qpos.copy())
        log["qvel"].append(p.d.qvel.copy())
        if renderer is not None and len(log["ctrl"]) % 3 == 0:
            R = p.d.body("rh_palm").xmat.reshape(3, 3)
            cam = mj.MjvCamera()
            cam.type = mj.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = p.d.body("rh_palm").xpos + R @ np.array([0.0, -0.04, 0.07])
            cam.distance, cam.azimuth, cam.elevation = 0.25, 180, -60
            renderer.update_scene(p.d, camera=cam)
            img = Image.fromarray(renderer.render().copy())
            ImageDraw.Draw(img).text((8, 6), f"MPC teacher  t={len(log['ctrl']) * CTRL_DT:.1f}s  target {np.degrees(phase):+.0f} deg  "
                                             f"turned {np.degrees(turned):+.0f} deg", fill=(255, 255, 255))
            frames.append(img)

    def angles():
        b = np.array(pr.palm())
        g = b.mean(0)
        return np.arctan2(b[:, 0] - g[0], b[:, 2] - g[2])

    def ball_angles():
        b = np.array(pr.palm())
        return np.arctan2(b[:, 0] - center[0], b[:, 2] - center[1])

    def trio_phase():
        """The trio's phase round the circle's centre, mod 120 deg (circular mean of 3x each ball's angle)."""
        b = np.array(pr.palm())
        th = np.arctan2(b[:, 0] - center[0], b[:, 2] - center[1])
        return np.angle(np.mean(np.exp(3j * th))) / 3

    turned, prev_a = 0.0, angles()
    actual = prev_p = trio_phase()
    prev0 = actual
    boost = 1                                         # search budget x boost after a stalled segment
    with Pool(args.procs, initializer=_init, initargs=(args.radius, args.scene, args.seg, args.hold, np.radians(args.wrist_tilt), args.tip_weight)) as pool:
        checkpoints, fails, backtracks, s = [], {}, 0, -1
        def snapshot():
            return dict(state=state.copy(), ctrl=ctrl.copy(), actual=actual, prev_p=prev_p, turned=turned,
                        prev_a=prev_a.copy(), n=len(log["ctrl"]))
        checkpoints.append(snapshot())
        while s + 1 < n_seg:
            s += 1
            phase = actual + args.sign * np.radians(args.step)
            targets = pr.places(center, phase)
            if args.labeled:      # ball k's target: its own slot, turned on by the step
                slots = pr.places(center, actual)
                targets = [targets[int(np.argmin([np.linalg.norm(b - s) for s in slots]))] for b in pr.palm()]
            for attempt in range(args.retries + 1):   # a plan whose best rollout loses a ball is not run: search harder
                nw = args.waypoints + (boost > 1) * 2
                mu = np.tile(ctrl, (nw, 1))
                sd = np.tile(args.spread * span, (nw, 1))
                best = (np.inf, None, None)
                for _ in range(args.iters * min(boost, 2)):
                    cands = [mu] + [mu + sd * rng.standard_normal(mu.shape) for _ in range(args.pop * boost - 1)]
                    res = pool.map(cost, [(state, ctrl, c, targets, args.labeled) for c in cands])
                    order = np.argsort([r[0] for r in res])
                    if res[order[0]][0] < best[0]:
                        best = (res[order[0]][0], cands[order[0]], res[order[0]])
                    elite = np.array([cands[i] for i in order[:8]])
                    mu, sd = elite.mean(0), np.maximum(0.6 * sd + 0.4 * elite.std(0), 0.01 * span)
                if not best[2][2]:
                    break
                boost = min(boost * 2, args.max_boost)
                print(f"  every plan loses a ball: re-searching at boost x{boost}", flush=True)
            depth = len(checkpoints) - 1
            if best[2][2] and backtracks < args.backtracks and depth > 0:
                # dead end: step back (further after repeated failures here) and take another way in
                fails[depth] = fails.get(depth, 0) + 1
                back = min(1 + (fails[depth] - 1) // 2, depth)
                del checkpoints[len(checkpoints) - back:]
                c = checkpoints[-1]
                mj.mj_setState(pr.m, pr.d, c["state"], SPEC)
                state, ctrl, actual, prev_p, turned, prev_a = c["state"].copy(), c["ctrl"].copy(), c["actual"], c["prev_p"], c["turned"], c["prev_a"].copy()
                for k in log:
                    del log[k][c["n"]:]
                backtracks += 1
                boost = 1
                print(f"  dead end at seg {s + 1}: back {back} to {np.degrees(actual):+.0f} deg (backtrack {backtracks}/{args.backtracks})", flush=True)
                s = depth - back
                continue
            th0 = ball_angles()
            lost = play(pr, state, ctrl, best[1], record=record)
            a = angles()
            turned += np.mean(np.angle(np.exp(1j * (a - prev_a))))
            prev_a = a
            p = trio_phase()
            gain = np.angle(np.exp(3j * (p - prev_p))) / 3
            if args.labeled:      # a whole click is invisible mod 120 deg: each ball's own turn round the centre
                gain = np.mean(np.angle(np.exp(1j * (ball_angles() - th0))))
            actual += gain
            prev_p = p
            boost = 1 if args.sign * np.degrees(gain) > 2.0 else min(boost * 2, args.max_boost)
            mj.mj_getState(pr.m, pr.d, state, SPEC)
            ctrl = best[1][-1].copy()
            e = errors(pr, targets, args.labeled) * 100
            print(f"seg {s + 1:3d}/{n_seg} target {np.degrees(phase):+5.0f} deg  trio phase {np.degrees(actual):+6.1f}, turned {np.degrees(turned):+6.1f} deg  "
                  f"off target mean {e.mean():.2f} max {e.max():.2f} cm  next boost x{boost}  tips {pr.tips / max(pr.tip_n, 1):.2f}{'  LOST' if lost else ''}  ({time.time() - t0:.0f}s)", flush=True)
            np.savez(args.out, **{k: np.array(v) for k, v in log.items()}, center=center, sign=args.sign)
            if lost:
                break
            checkpoints.append(snapshot())
            if abs(actual - prev0) >= np.radians(args.degrees):
                break
    np.savez(args.out, **{k: np.array(v) for k, v in log.items()}, center=center, sign=args.sign)
    if renderer is not None:
        renderer.close()
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=60, loop=0)
        print(f"saved {len(frames)} frames to {args.gif}")


if __name__ == "__main__":
    main()
