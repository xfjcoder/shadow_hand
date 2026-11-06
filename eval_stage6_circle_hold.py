"""Stage 6 (rebuild, step 1): is a three-ball circle HOLDABLE at every phase?

Baoding along a fixed circle needs every intermediate trio placement to be a state the fingers can hold
(the static stage of the SDS curriculum, Chiappa et al. 2024). For a circle centre (palm frame, cm) and
phases over one 120 deg period, the three balls are held kinematically at their places while the fingers
close to a candidate posture (wrist fixed at the bowl posture), then released; a CEM over the 18 finger
targets minimises the balls' drift from their places over the next `--hold` seconds. Prints per phase
the best drift (mean / max over balls, cm) and whether a ball was lost.

Usage:
    python3 eval_stage6_circle_hold.py --centers 0.1,6.5 0,7.5 --phases 12
"""
import argparse
from multiprocessing import Pool

import mujoco as mj
import numpy as np

from hand_env import bowl_target
from three_balls import BALLS3, THREE_BALL_SCENE, set_radius

WRIST = 2                      # the first two actuators: WRJ2, WRJ1, held at the bowl posture
REST_Y = -0.034                # palm-frame y of a 22.5mm ball's centre resting on the palm


class Probe:
    def __init__(self, radius, scene=THREE_BALL_SCENE):
        self.m = mj.MjModel.from_xml_path(scene)
        set_radius(self.m, radius)
        self.d = mj.MjData(self.m)
        self.r = radius
        self.lo, self.hi = self.m.actuator_ctrlrange.T
        self.target0 = np.clip(bowl_target(), self.lo, self.hi)
        self.adr = [(self.m.jnt_qposadr[self.m.body(b).jntadr[0]], self.m.jnt_dofadr[self.m.body(b).jntadr[0]]) for b in BALLS3]
        wj = [self.m.actuator_trnid[i, 0] for i in range(WRIST)]
        self.wrist_q = [self.m.jnt_qposadr[j] for j in wj]
        self._reset()
        self.R, self.p = self.d.body("rh_palm").xmat.reshape(3, 3).copy(), self.d.body("rh_palm").xpos.copy()

    def _reset(self):
        mj.mj_resetData(self.m, self.d)
        for q, u in zip(self.wrist_q, self.target0[:WRIST]):
            self.d.qpos[q] = u
        mj.mj_forward(self.m, self.d)

    def places(self, center, phase):
        """Palm-frame centres: circumradius 2r/sqrt(3) round `center` (x, z), angle from the fingers (+z)."""
        rc = 2 * self.r / np.sqrt(3)
        y = REST_Y + (self.r - 0.0225)
        return [np.array([center[0] + rc * np.sin(phase + k * 2 * np.pi / 3), y, center[1] + rc * np.cos(phase + k * 2 * np.pi / 3)])
                for k in range(3)]

    def palm(self):
        """Ball centres in the palm's CURRENT frame (the same as the start frame while the wrist is held)."""
        R, p = self.d.body("rh_palm").xmat.reshape(3, 3), self.d.body("rh_palm").xpos
        return [R.T @ (self.d.body(b).xpos - p) for b in BALLS3]

    def rollout(self, fingers, targets, close=0.4, hold=2.0):
        """Drift (cm) of the balls from `targets` after release: (score, mean final, max final, lost)."""
        self._reset()
        ctrl = np.concatenate([self.target0[:WRIST], np.clip(fingers, self.lo[WRIST:], self.hi[WRIST:])])
        rest = np.clip(0.0, self.lo, self.hi)
        rest[:WRIST] = ctrl[:WRIST]
        world = [self.p + self.R @ t for t in targets]
        n = int(close / self.m.opt.timestep)
        for k in range(n):
            self.d.ctrl[:] = rest + (ctrl - rest) * min(1.0, 1.5 * (k + 1) / n)
            for (q, v), w in zip(self.adr, world):
                self.d.qpos[q:q + 3], self.d.qpos[q + 3:q + 7], self.d.qvel[v:v + 6] = w, (1, 0, 0, 0), 0.0
            mj.mj_step(self.m, self.d)
        acc, lost = 0.0, False
        steps = int(hold / self.m.opt.timestep)
        for k in range(steps):
            mj.mj_step(self.m, self.d)
            if k % 50 == 49:
                e = np.array([np.linalg.norm(b - t) for b, t in zip(self.palm(), targets)])
                acc += e.mean()
                if e.max() > 0.04:
                    lost = True
                    break
        e = np.array([np.linalg.norm(b - t) for b, t in zip(self.palm(), targets)]) * 100
        score = e.mean() + 0.5 * e.max() + 100 * acc / (steps / 50) + (50.0 if lost else 0.0)
        return score, e.mean(), e.max(), lost


def solve(job):
    center, phase, radius, iters, pop, seed, scene = job
    pr = Probe(radius, scene)
    rng = np.random.default_rng(seed)
    targets = pr.places(np.array(center) / 100, phase)
    span = (pr.hi - pr.lo)[WRIST:]
    mu, sd = pr.target0[WRIST:].copy(), 0.2 * span
    best = (np.inf, None, None)
    for _ in range(iters):
        xs = [mu] + [mu + sd * rng.standard_normal(len(mu)) for _ in range(pop - 1)]
        res = [pr.rollout(x, targets) for x in xs]
        order = np.argsort([r[0] for r in res])
        if res[order[0]][0] < best[0]:
            best = (res[order[0]][0], res[order[0]], xs[order[0]])
        elite = np.array([xs[i] for i in order[:max(4, pop // 6)]])
        mu, sd = elite.mean(0), np.maximum(0.6 * sd + 0.4 * elite.std(0), 0.02 * span)
    return center, phase, best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centers", nargs="+", default=["0.1,6.5", "0,7.0", "0,7.5", "0,8.0"], help="palm-frame x,z in cm")
    ap.add_argument("--phases", type=int, default=12)
    ap.add_argument("--radius", type=float, default=0.0225)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--procs", type=int, default=15)
    ap.add_argument("--save", default=None, help="npz of the best postures")
    ap.add_argument("--scene", default=THREE_BALL_SCENE, help="assets/scene_three_balls_slick.xml: polished balls")
    args = ap.parse_args()
    centers = [tuple(float(v) for v in c.split(",")) for c in args.centers]
    jobs = [(c, k * (2 * np.pi / 3) / args.phases, args.radius, args.iters, args.pop, i * 100 + k, args.scene)
            for i, c in enumerate(centers) for k in range(args.phases)]
    with Pool(args.procs) as pool:
        out = pool.map(solve, jobs)
    for c in centers:
        rows = [o for o in out if o[0] == c]
        cells = " ".join(f"{np.degrees(ph):3.0f}:{r[1]:.1f}{'L' if r[3] else ''}" for _, ph, r, _ in rows)
        print(f"centre x={c[0]:+.1f} z={c[1]:.1f} cm | phase deg: mean drift cm ('L' lost) | {cells} "
              f"| worst {max(r[1] for _, _, r, _ in rows):.1f}, lost {sum(r[3] for _, _, r, _ in rows)}/{len(rows)}", flush=True)
    if args.save:
        np.savez(args.save, centers=np.array([o[0] for o in out]), phases=np.array([o[1] for o in out]),
                 fingers=np.array([o[3] for o in out]), drift=np.array([o[2][1] for o in out]), lost=np.array([o[2][3] for o in out]))


if __name__ == "__main__":
    main()
