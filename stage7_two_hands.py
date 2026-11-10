"""Stage 7B: two-hand, four-ball baoding -- each hand turns its own pair (the Curtis Zimmerman pattern),
both driven by the Stage 5 policy (baoding_15 @60M), in ONE simulation (`two_hands.py`).

The policy was trained on the right hand alone. Each hand gets a PROXY: a single-hand Stage 5 env
(`BaodingEnv9`) that never steps physics; every control step the hand's joints and its two balls are
copied into it -- the left hand's through the mirror (its palm frame with x negated is the right
palm's) -- and the proxy's own bookkeeping (relay clock, ball tracker, swaps, drop limits) runs on
that copy, so the policy sees exactly the observation it was trained on. Its action goes back to the
hand's actuators in the two-hand scene (the menagerie left hand takes the same joint angles for the
mirrored pose). The left pair turns the mirrored way round, as a person's hands do.

Usage:
    python3 stage7_two_hands.py runs/baoding_15/checkpoints/ppo_60000000_steps.zip --n 8
    python3 stage7_two_hands.py runs/baoding_15/checkpoints/ppo_60000000_steps.zip --n 1 --seed 1 \
        --gif media_full/stage7_two_hands.gif
"""
import argparse

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from demo_stage5_baoding_rl import vecnormalize_path
from hand_env import ACTUATOR_NAMES, BALLS, JOINT_NAMES
from rl_env_baoding import CONTROL_DT, MAX_H
from rl_env_baoding2 import BALL_Z_DROP, MOVING, WRIST, BaodingEnv9
from two_hands import LEFT_BALLS, RIGHT_BALLS, build, left

S = np.diag([-1.0, 1.0, 1.0])          # the mirror, in palm coordinates


def quat_mat(q):
    R = np.zeros(9)
    mj.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


class Side:
    """One hand of the two-hand scene and its single-hand proxy env."""

    def __init__(self, model, proxy, mirror):
        self.m, self.p, self.mirror = model, proxy, mirror
        name = left if mirror else (lambda n: n)
        pm = proxy.model
        self.palm = name("rh_palm")
        self.balls = LEFT_BALLS if mirror else RIGHT_BALLS
        self.S = S if mirror else np.eye(3)
        # (scene qpos adr, proxy qpos adr), same for dofs, for the 24 hand joints
        self.q = [(model.jnt_qposadr[model.joint(name(j)).id], pm.jnt_qposadr[pm.joint(j).id]) for j in JOINT_NAMES]
        self.v = [(model.jnt_dofadr[model.joint(name(j)).id], pm.jnt_dofadr[pm.joint(j).id]) for j in JOINT_NAMES]
        self.u = [(model.actuator(name(a)).id, pm.actuator(a).id) for a in ACTUATOR_NAMES]
        self.fb = [(model.body(b).id, pm.body(pb).id) for b, pb in zip(self.balls, BALLS)]

    def _frames(self, d):
        pd = self.p.data
        Rs, ps = d.body(self.palm).xmat.reshape(3, 3), d.body(self.palm).xpos
        Rp, pp = pd.body("rh_palm").xmat.reshape(3, 3), pd.body("rh_palm").xpos
        return Rs, ps, Rp, pp

    def to_proxy(self, d):
        """Copy this hand and its balls from the scene into the proxy (mirrored for the left hand)."""
        pm, pd = self.p.model, self.p.data
        for a, b in self.q:
            pd.qpos[b] = d.qpos[a]
        for a, b in self.v:
            pd.qvel[b] = d.qvel[a]
        for a, b in self.u:
            pd.ctrl[b] = d.ctrl[a]
        mj.mj_kinematics(pm, pd)                       # the proxy palm's pose, from the copied joints
        Rs, ps, Rp, pp = self._frames(d)
        M = Rp @ self.S @ Rs.T                         # scene vectors -> proxy vectors
        for sb, pb in self.fb:
            qa, va = self.m.jnt_qposadr[self.m.body_jntadr[sb]], self.m.jnt_dofadr[self.m.body_jntadr[sb]]
            qb, vb = pm.jnt_qposadr[pm.body_jntadr[pb]], pm.jnt_dofadr[pm.body_jntadr[pb]]
            w = quat_mat(d.qpos[qa + 3:qa + 7]) @ d.qvel[va + 3:va + 6]    # body-frame -> world
            pd.qpos[qb:qb + 3] = M @ (d.qpos[qa:qa + 3] - ps) + pp
            pd.qpos[qb + 3:qb + 7] = [1, 0, 0, 0]
            pd.qvel[vb:vb + 3] = M @ d.qvel[va:va + 3]
            pd.qvel[vb + 3:vb + 6] = np.linalg.det(self.S) * (M @ w)
        mj.mj_forward(pm, pd)

    def ball_props(self):
        """The proxy's randomized ball mass / size / friction onto this hand's balls -- no mj_setConst,
        as in the Stage 5 env (the solver constants stay the compiled 45mm / 80g balls')."""
        pm = self.p.model
        for sb, pb in self.fb:
            self.m.body_mass[sb], self.m.body_inertia[sb] = pm.body_mass[pb], pm.body_inertia[pb]
            sg, pg = self.m.body_geomadr[sb], pm.body_geomadr[pb]
            self.m.geom_size[sg], self.m.geom_friction[sg] = pm.geom_size[pg], pm.geom_friction[pg]
            self.m.geom_rbound[sg] = pm.geom_rbound[pg]

    def from_proxy(self, d):
        """The inverse: put the proxy's (settled start) state into the scene."""
        pm, pd = self.p.model, self.p.data
        for a, b in self.q:
            d.qpos[a] = pd.qpos[b]
        for a, b in self.v:
            d.qvel[a] = pd.qvel[b]
        for a, b in self.u:
            d.ctrl[a] = pd.ctrl[b]
        mj.mj_kinematics(self.m, d)
        Rs, ps, Rp, pp = self._frames(d)
        M = Rs @ self.S @ Rp.T                         # proxy vectors -> scene vectors
        for sb, pb in self.fb:
            qa, va = self.m.jnt_qposadr[self.m.body_jntadr[sb]], self.m.jnt_dofadr[self.m.body_jntadr[sb]]
            qb, vb = pm.jnt_qposadr[pm.body_jntadr[pb]], pm.jnt_dofadr[pm.body_jntadr[pb]]
            w = quat_mat(pd.qpos[qb + 3:qb + 7]) @ pd.qvel[vb + 3:vb + 6]
            d.qpos[qa:qa + 3] = M @ (pd.qpos[qb:qb + 3] - pp) + ps
            d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
            d.qvel[va:va + 3] = M @ pd.qvel[vb:vb + 3]
            d.qvel[va + 3:va + 6] = np.linalg.det(self.S) * (M @ w)

    def ctrl(self, d, action):
        """The proxy env's own action rule (rate-limited targets, wrist held), written to this hand."""
        p = self.p
        c = p.data.ctrl.copy()
        c[MOVING] += np.clip(action, -1, 1) * p.rate[MOVING]
        c[WRIST] = p.base[WRIST]
        c = np.clip(c, p.lo, p.hi)
        for a, b in self.u:
            d.ctrl[a] = c[b]

    def bookkeep(self):
        """The proxy env's step after physics: clock, progress, swaps, drop -- on the copied state."""
        p = self.p
        p.t += 1
        p._clock()
        p.progress, offsets = p._progress()
        p.best = max(p.best, p.progress)
        while p.progress >= p.next_swap:
            p.swaps += 1
            p.next_swap += p.cycle
        ball_z = [float(q[2]) for q in p._balls_in_palm()]
        return any(r > p.max_r or h > MAX_H for r, h in offsets) or max(ball_z) > BALL_Z_DROP


def render(renderer, d, hands, t):
    mid = 0.5 * (d.body("rh_palm").xpos + d.body(left("rh_palm")).xpos)
    R = d.body("rh_palm").xmat.reshape(3, 3)
    look = mid + R @ np.array([0.0, -0.04, 0.07])
    imgs = []
    for az, el, dist in ((180, -75, 0.42), (200, -40, 0.42)):
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        renderer.update_scene(d, camera=cam)
        imgs.append(renderer.render().copy())
    im = Image.fromarray(np.concatenate(imgs, 1))
    txt = "  ".join(f"{s}: swaps {h.p.swaps} ({h.p.progress:+.1f} rad){' DROPPED' if h.dropped else ''}"
                    for s, h in zip(("left", "right"), hands))
    ImageDraw.Draw(im).text((8, 6), f"two hands, Stage 5 policy on each  t={t:.1f}s  {txt}", fill=(255, 255, 255))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--direction", type=float, default=-1.0, help="baoding_15 trained with -1")
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--gif", default=None, help="GIF of the first episode")
    args = ap.parse_args()

    spec, m = build()
    d = mj.MjData(m)
    n_sub = int(round(CONTROL_DT / m.opt.timestep))
    proxies = [BaodingEnv9(direction=args.direction, level=args.level, episode_seconds=args.seconds) for _ in range(2)]
    vn = VecNormalize.load(str(vecnormalize_path(args.ckpt)), DummyVecEnv([lambda: BaodingEnv9(direction=args.direction, level=args.level)]))
    vn.training, vn.norm_reward = False, False
    model = PPO.load(args.ckpt, device="cpu")
    hands = [Side(m, proxies[0], mirror=True), Side(m, proxies[1], mirror=False)]   # left, right
    rows = []
    for ep in range(args.n):
        mj.mj_resetData(m, d)
        for k, h in enumerate(hands):                 # each proxy settles its own random start
            h.p.reset(seed=args.seed + 1000 * ep + k)
            h.p.swaps, h.dropped, h.t_drop = 0, False, None
            h.ball_props()
        for h in hands:
            h.from_proxy(d)
        mj.mj_forward(m, d)
        for h in hands:                               # the proxies now hold the scene's (identical) state
            h.to_proxy(d)
        renderer = mj.Renderer(m, height=240, width=320) if args.gif and ep == 0 else None
        frames = []
        steps = int(round(args.seconds / CONTROL_DT))
        for t in range(steps):
            for h in hands:
                h.to_proxy(d)
                if not h.dropped:
                    a, _ = model.predict(vn.normalize_obs(h.p._obs()[None]), deterministic=True)
                    h.ctrl(d, a[0])
            for _ in range(n_sub):
                mj.mj_step(m, d)
            for h in hands:
                h.to_proxy(d)
                if not h.dropped and h.bookkeep():
                    h.dropped, h.t_drop = True, (t + 1) * CONTROL_DT
            if renderer is not None and t % 5 == 0:
                frames.append(render(renderer, d, hands, (t + 1) * CONTROL_DT))
            if all(h.dropped for h in hands):
                break
        if renderer is not None:
            renderer.close()
            frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
        rows.append([(h.p.swaps, h.dropped) for h in hands])
        print(f"ep {ep}  " + "  ".join(
            f"{s}: {h.p.swaps:2d} swaps ({h.p.progress:+5.1f} rad) "
            + (f"DROPPED at {h.t_drop:4.1f}s" if h.dropped else "kept") for s, h in zip(("left", "right"), hands)), flush=True)
    both = sum(not l[1] and not r[1] for l, r in rows)
    print(f"SUMMARY {args.n} x {args.seconds:.0f}s: both hands kept {both}/{args.n}; "
          f"left kept {sum(not l[1] for l, _ in rows)}, right kept {sum(not r[1] for _, r in rows)}; "
          f"swaps mean left {np.mean([l[0] for l, _ in rows]):.1f} right {np.mean([r[0] for _, r in rows]):.1f}")


if __name__ == "__main__":
    main()
