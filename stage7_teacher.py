"""Stage 7A, step 2: a sampling-MPC "teacher" that turns four balls, packed as a 2x2 square across the
seam of the two touching palms, round the square's centre -- each ball crossing from one hand to the other
every half turn.

The Stage 6 planner (`stage6_teacher.py`) on the two-hand scene: from the balls' actual state, a CEM over
`--waypoints` targets for all 36 finger actuators and the 4 wrist ones (within `--wrist-tilt` deg of
straight) minimises each ball's horizontal distance to its own corner of the square turned on by `--step`
deg, the balls' speed at the end, and a loss penalty; the best segment is run and the next is planned from
where the balls really are. Stalls and losses backtrack to a checkpoint a few segments back
(`--backtracks`). `--band B` holds every finger/thumb actuator within B x its range of the `--posture`
(a hard limit: the smoother demo; a soft `--pose-weight` cost only made the planner hold still).
`--resume` continues a saved run (its log is kept, so stitched runs stay one continuous demo).
The start is the settled layout of `stage7_layout.py` (the balls roll into the square on their own).
Saves the per-control-step (20 ms) controls, ball states, qpos/qvel and phases (an npz) for RL tracking.

Usage (the smooth demo data/stage7_demo_smooth360.npz: 0-183 deg as below, then resumed with --band 0.35
--wrist-tilt 15 to 274 deg, then --band 0.25 --wrist-tilt 15 to 364 deg):
    python3 stage7_teacher.py --posture '{"rh_A_FFJ3":0.7,"rh_A_FFJ0":1.0,...}' --wrist-tilt 10 --band 0.25 \\
        --degrees 360 --procs 15 --backtracks 40 --out data/stage7_teacher_band25.npz
"""
import argparse
import json
import time
from multiprocessing import Pool

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

from hand_env import ACTUATOR_NAMES
import stage7_layout
from stage7_layout import GAP, PALM_Z, ring_places, set_ball
from two_hands import LEFT_BALLS, RIGHT_BALLS, build, left, wrist_qpos

SEG, HOLD = 0.35, 0.35         # s per waypoint ramp; s held at the last waypoint
CTRL_DT = 0.02
SPEC = mj.mjtState.mjSTATE_INTEGRATION
BALLS = RIGHT_BALLS + LEFT_BALLS
BALL_R = 0.0225
FAR = 0.06                     # m: a ball this far (horizontally) from the square's centre is lost

_w = None                      # the worker's World
POSE_W = 0.0                   # cost per unit of finger/thumb distance from the posture (each actuator / its range, summed)


class World:
    def __init__(self, gap=GAP, ball=BALL_R, tilt=0.0, flex=None, posture=None, band=None):
        self.posture = POSTURE = dict(stage7_layout.POSTURE, **(posture or {}))
        _, self.m = build(gap, forearms=False)
        m = self.m
        for b in BALLS:                                # resize as the Stage 5 env does (no mj_setConst)
            g, bid = m.geom(b).id, m.body(b).id
            k = ball / m.geom_size[g, 0]
            m.geom_size[g, 0] = m.geom_rbound[g] = ball
            m.body_mass[bid] *= k ** 3
            m.body_inertia[bid] *= k ** 5
        self.d = mj.MjData(m)
        self.r = ball
        self.lo, self.hi = m.actuator_ctrlrange.T.copy()
        self.u0 = np.zeros(m.nu)
        for a in ACTUATOR_NAMES:
            self.u0[m.actuator(a).id] = self.u0[m.actuator(left(a)).id] = POSTURE[a]
        self.flex = POSTURE["rh_A_WRJ1"] if flex is None else flex   # wrist flex: + levels the palms
        self.u0[m.actuator("rh_A_WRJ1").id] = self.u0[m.actuator(left("rh_A_WRJ1")).id] = self.flex
        wrist = [m.actuator(a).id for a in ("rh_A_WRJ1", "rh_A_WRJ2")]
        wrist += [m.actuator(left(a)).id for a in ("rh_A_WRJ1", "rh_A_WRJ2")]
        self.lo[wrist] = np.maximum(self.lo[wrist], self.u0[wrist] - tilt)   # the wrists within `tilt` rad
        self.hi[wrist] = np.minimum(self.hi[wrist], self.u0[wrist] + tilt)
        if band:                                       # fingers and thumbs within `band` x their range of the posture
            span = self.hi - self.lo
            f = np.array([i for i in range(m.nu) if i not in wrist])
            self.lo[f] = np.maximum(self.lo[f], self.u0[f] - band * span[f])
            self.hi[f] = np.minimum(self.hi[f], self.u0[f] + band * span[f])
        self.bq = [m.jnt_qposadr[m.body_jntadr[m.body(b).id]] for b in BALLS]
        self.bv = [m.jnt_dofadr[m.body_jntadr[m.body(b).id]] for b in BALLS]

    def balls(self):
        return np.array([self.d.qpos[q:q + 3] for q in self.bq])

    def speed(self):
        return sum(np.linalg.norm(self.d.qvel[v:v + 3]) for v in self.bv)

    def settle(self, along=0.30, ring=0.04, seconds=3.0):
        """The layout: the hands take the posture, the balls are set on a ring over the seam and roll."""
        m, d = self.m, self.d
        mj.mj_resetData(m, d)
        d.ctrl[:] = self.u0
        wrist_qpos(m, d, self.flex, self.posture["rh_A_WRJ2"])
        for _ in range(int(1.0 / m.opt.timestep)):
            mj.mj_step(m, d)
        seam_y = 0.5 * (d.body("rh_palm").xpos[1] + d.body(left("rh_palm")).xpos[1])
        places = ring_places([along, seam_y, PALM_Z + self.r + 0.002], ring, np.pi / 4)
        for b, i in zip(BALLS, sorted(range(4), key=lambda i: places[i][1])):
            set_ball(m, d, b, places[i])
        for _ in range(int(seconds / m.opt.timestep)):
            mj.mj_step(m, d)


def _init(gap, ball, seg, hold, tilt, flex, posture, pose_w=0.0, band=None):
    global _w, SEG, HOLD, POSE_W
    _w = World(gap, ball, tilt, flex, posture, band)
    POSE_W = pose_w
    SEG, HOLD = seg, hold


def lost_now(w, center):
    b = w.balls()
    return bool(np.any(b[:, 2] < PALM_Z) or np.any(np.linalg.norm(b[:, :2] - center, axis=1) > FAR))


def play(w, state, start_ctrl, wps, center, record=None):
    """From `state`, ramp the fingers through `wps` (each over SEG), hold HOLD. -> lost flag."""
    mj.mj_setState(w.m, w.d, state, SPEC)
    n, h = int(SEG / w.m.opt.timestep), int(HOLD / w.m.opt.timestep)
    per = int(CTRL_DT / w.m.opt.timestep)
    prev, k = start_ctrl, 0
    for wp in list(wps) + [None]:
        steps = n if wp is not None else h
        for i in range(steps):
            u = prev if wp is None else prev + (wp - prev) * (i + 1) / steps
            w.d.ctrl[:] = np.clip(u, w.lo, w.hi)
            mj.mj_step(w.m, w.d)
            k += 1
            if record is not None and k % per == 0:
                record(w)
        if wp is not None:
            prev = wp
        if lost_now(w, center):
            return True
    return False


def corners(center, rc, angles):
    return center + rc * np.stack([np.cos(angles), np.sin(angles)], 1)


def cost(job):
    state, start_ctrl, wps, center, targets = job
    lost = play(_w, state, start_ctrl, wps, center)
    e = np.linalg.norm(_w.balls()[:, :2] - targets, axis=1) * 100
    effort = sum(np.abs(a - b).sum() for a, b in zip(wps, [start_ctrl] + list(wps[:-1])))
    pose = np.mean([pose_dev(_w, wp) for wp in wps]) if POSE_W else 0.0
    return (e.mean() + 0.5 * e.max() + 10 * _w.speed() + 0.02 * effort + POSE_W * pose + (100.0 if lost else 0.0),
            e.mean(), lost)


def pose_dev(w, u):
    """How far the fingers and thumbs (not the wrists) are from the posture: each |u - u0| / its range, summed."""
    if not hasattr(w, "fingers"):
        span = np.maximum(w.m.actuator_ctrlrange[:, 1] - w.m.actuator_ctrlrange[:, 0], 1e-9)
        wr = [w.m.actuator(a).id for a in ("rh_A_WRJ1", "rh_A_WRJ2", "L_lh_A_WRJ1", "L_lh_A_WRJ2")]
        w.fingers = np.array([i for i in range(w.m.nu) if i not in wr])
        w.fspan = span[w.fingers]
    u = np.clip(u, w.lo, w.hi)                          # as executed: samples past a joint's range are clipped
    return float(np.sum(np.abs(u[w.fingers] - w.u0[w.fingers]) / w.fspan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=GAP)
    ap.add_argument("--ball", type=float, default=BALL_R)
    ap.add_argument("--step", type=float, default=15.0, help="deg of commanded turn per segment")
    ap.add_argument("--degrees", type=float, default=90.0)
    ap.add_argument("--sign", type=float, default=1.0, help="+1: counter-clockwise seen from above")
    ap.add_argument("--waypoints", type=int, default=3)
    ap.add_argument("--pop", type=int, default=90)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--spread", type=float, default=0.25, help="CEM start std, fraction of each actuator's range")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--max-boost", type=int, default=4)
    ap.add_argument("--seg", type=float, default=SEG)
    ap.add_argument("--hold", type=float, default=HOLD)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=15)
    ap.add_argument("--out", default="data/stage7_teacher.npz")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--wrist-tilt", type=float, default=0.0, help="deg each wrist joint may move (0: fixed)")
    ap.add_argument("--flex", type=float, default=0.0, help="rad, both wrists' WRJ1: the palms' fore-aft tilt")
    ap.add_argument("--posture", default=None, help="json of actuator targets over stage7_layout.POSTURE (right-hand names)")
    ap.add_argument("--along", type=float, default=0.30, help="m, world x of the start ring (fingers +x)")
    ap.add_argument("--settle", type=float, default=3.0, help="s the balls settle into the start square")
    ap.add_argument("--resume", default=None, help="an earlier run's npz to continue from (its centre and corners)")
    ap.add_argument("--resume-at", type=int, default=-1, help="the logged control step to continue from")
    ap.add_argument("--pose-weight", type=float, default=0.0,
                    help="cost per unit of finger/thumb distance from the posture (the user: smoother demo first)")
    ap.add_argument("--band", type=float, default=None,
                    help="fingers and thumbs held within this fraction of their range of the posture (hard limits)")
    ap.add_argument("--stall", type=int, default=3, help="segments without progress that count as a dead end")
    ap.add_argument("--backtracks", type=int, default=0, help="dead ends (stalls or a lost ball) to back out of")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    tilt = np.radians(args.wrist_tilt)
    posture = json.loads(args.posture) if args.posture else None
    w = World(args.gap, args.ball, tilt, args.flex, posture, args.band)
    w.settle(along=args.along, seconds=args.settle)
    b0 = w.balls()
    center = b0[:, :2].mean(0)
    rc = float(np.linalg.norm(b0[:, :2] - center, axis=1).mean())
    own = np.arctan2(*(b0[:, :2] - center)[:, ::-1].T)   # each ball's own corner angle; the square's phase = 0
    turned = 0.0
    if args.resume:                                   # continue from a logged step of an earlier run
        r = np.load(args.resume)
        center, rc = r["center"], float(r["rc"])
        own = np.arctan2(*(r["balls"][0][:, :2] - center)[:, ::-1].T)
        w.d.qpos[:], w.d.qvel[:], w.d.ctrl[:] = r["qpos"][args.resume_at], r["qvel"][args.resume_at], r["ctrl"][args.resume_at]
        mj.mj_forward(w.m, w.d)
        # the turn so far, summed step by step over the old log (a single wrapped angle fails past 180 deg)
        upto = args.resume_at % len(r["ctrl"]) + 1
        th = np.arctan2(r["balls"][:upto, :, 1] - center[1], r["balls"][:upto, :, 0] - center[0])
        turned = float(np.sum(np.mean(np.angle(np.exp(1j * np.diff(th, axis=0))), axis=1)))
    print(f"settled square: centre x {100 * center[0]:.1f} y {100 * center[1]:.1f} cm, corner radius {100 * rc:.2f} cm "
          f"(packed: {100 * np.sqrt(2) * args.ball:.2f})", flush=True)
    state = np.empty(mj.mj_stateSize(w.m, SPEC))
    mj.mj_getState(w.m, w.d, state, SPEC)
    ctrl = w.d.ctrl.copy()
    span = np.maximum(w.hi - w.lo, 1e-9)
    ang = lambda: np.arctan2(*(w.balls()[:, :2] - center)[:, ::-1].T)

    log = dict(ctrl=[], balls=[], phase=[], qpos=[], qvel=[])
    if args.resume:                                   # the new log continues the old one (one demo)
        upto = args.resume_at % len(r["ctrl"]) + 1
        for k in log:
            log[k] = list(r[k][:upto])
    renderer = mj.Renderer(w.m, height=300, width=400) if args.gif else None
    frames = []
    phase = 0.0
    n_seg = 4 * int(round(args.degrees / args.step))
    t0 = time.time()

    def record(p):
        log["ctrl"].append(p.d.ctrl.copy())
        log["balls"].append(p.balls())
        log["phase"].append(phase)
        log["qpos"].append(p.d.qpos.copy())
        log["qvel"].append(p.d.qvel.copy())
        if renderer is not None and len(log["ctrl"]) % 3 == 0:
            cam = mj.MjvCamera()
            cam.type = mj.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [center[0], center[1], PALM_Z + 0.02]
            cam.distance, cam.azimuth, cam.elevation = 0.36, 180, -60
            renderer.update_scene(p.d, camera=cam)
            img = Image.fromarray(renderer.render().copy())
            ImageDraw.Draw(img).text((8, 6), f"two-hand MPC teacher  t={len(log['ctrl']) * CTRL_DT:.1f}s  "
                                             f"target {np.degrees(phase):+.0f}  turned {np.degrees(turned):+.0f} deg",
                                     fill=(255, 255, 255))
            frames.append(img)

    boost, stalled, backtracks, fails = 1, 0, 0, {}
    checkpoints = [dict(state=state.copy(), ctrl=ctrl.copy(), turned=turned, n=len(log["ctrl"]))]
    with Pool(args.procs, initializer=_init, initargs=(args.gap, args.ball, args.seg, args.hold, tilt, args.flex, posture, args.pose_weight, args.band)) as pool:
        for s in range(4 * n_seg):                      # a budget: backtracking re-plans segments
            phase = turned + args.sign * np.radians(args.step)
            targets = corners(center, rc, own + phase)
            for attempt in range(args.retries + 1):
                nw = args.waypoints + (boost > 1) * 2
                mu = np.tile(ctrl, (nw, 1))
                sd = np.tile(args.spread * span, (nw, 1))
                best = (np.inf, None, None)
                for _ in range(args.iters * min(boost, 2)):
                    cands = [mu] + [mu + sd * rng.standard_normal(mu.shape) for _ in range(args.pop * boost - 1)]
                    res = pool.map(cost, [(state, ctrl, c, center, targets) for c in cands])
                    order = np.argsort([r[0] for r in res])
                    if res[order[0]][0] < best[0]:
                        best = (res[order[0]][0], cands[order[0]], res[order[0]])
                    elite = np.array([cands[i] for i in order[:8]])
                    mu, sd = elite.mean(0), np.maximum(0.6 * sd + 0.4 * elite.std(0), 0.01 * span)
                if not best[2][2]:
                    break
                boost = min(boost * 2, args.max_boost)
                print(f"  every plan loses a ball: re-searching at boost x{boost}", flush=True)
            a0 = ang()
            lost = play(w, state, ctrl, best[1], center, record=record)
            gain = float(np.mean(np.angle(np.exp(1j * (ang() - a0)))))
            turned += gain
            moved = args.sign * np.degrees(gain) > 2.0
            boost = 1 if moved else min(boost * 2, args.max_boost)
            stalled = 0 if moved else stalled + 1
            mj.mj_getState(w.m, w.d, state, SPEC)
            ctrl = best[1][-1].copy()
            e = np.linalg.norm(w.balls()[:, :2] - targets, axis=1) * 100
            print(f"seg {s + 1:3d}/{n_seg} target {np.degrees(phase):+6.1f}  turned {np.degrees(turned):+6.1f} deg  "
                  f"off target mean {e.mean():.2f} max {e.max():.2f} cm  pose {pose_dev(w, ctrl):.1f}  next boost x{boost}"
                  f"{'  LOST' if lost else ''}  ({time.time() - t0:.0f}s)", flush=True)
            np.savez(args.out, **{k: np.array(v) for k, v in log.items()}, center=center, rc=rc, sign=args.sign)
            if abs(turned) >= np.radians(args.degrees):
                break
            if (lost or stalled >= args.stall) and backtracks < args.backtracks and len(checkpoints) > 1:
                # dead end: step back (further after repeated failures from the same place) and re-plan
                depth = len(checkpoints) - 1
                fails[depth] = fails.get(depth, 0) + 1
                back = min(2 + (fails[depth] - 1), depth)
                del checkpoints[len(checkpoints) - back:]
                c = checkpoints[-1]
                state, ctrl, turned = c["state"].copy(), c["ctrl"].copy(), c["turned"]
                mj.mj_setState(w.m, w.d, state, SPEC)
                for k in log:
                    del log[k][c["n"]:]
                backtracks += 1
                boost, stalled = 1, 0
                print(f"  dead end ({'lost' if lost else 'stalled'}): back {back} segments to {np.degrees(turned):+.1f} deg "
                      f"(backtrack {backtracks}/{args.backtracks})", flush=True)
                continue
            if lost:
                break
            if moved:
                checkpoints.append(dict(state=state.copy(), ctrl=ctrl.copy(), turned=turned, n=len(log["ctrl"])))
    if renderer is not None:
        renderer.close()
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=60, loop=0)
        print(f"saved {len(frames)} frames to {args.gif}")


if __name__ == "__main__":
    main()
