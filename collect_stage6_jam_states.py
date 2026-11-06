"""Stage 6: start states for training past the jam (`rl_env_baoding3.BaodingEnv3BallJam`).

Run three_05 @64M (45mm, env 13) from randomized starts; every policy episode that turns the trio >= 2 rad
stalls at one orientation, the balls wedged between the thumb's base segment, the little finger's metacarpal
and the palm (an ideal 0.3-1 N forward push on each ball moves them < 0.1 rad, harder pops one out). JAM
states: snapshots of those at 12, 16, 20 and 24 s. PASSAGE states: along a 5-waypoint finger sequence (no
push) found by a cross-entropy search from start seed 14's stall (`data/stage6_jam_passage_seed14.npy`),
which takes the trio from 2.80 to 4.53 rad and leaves the balls at rest -- every 0.25 s from its start to
the end of its 1.5 s settle. Each state keeps the ball mass/friction of its episode and the fingertip
positions of its episode's settled start (the bowl the stroke "home" and shape band refer to).

Usage:
    python3 collect_stage6_jam_states.py --n 64
"""
import argparse

import mujoco as mj
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from demo_stage5_baoding_rl import vecnormalize_path
from rl_env_baoding2 import MOVING
from rl_env_baoding3 import BaodingEnv3BallGuide

CK = "runs/three_05/checkpoints/ppo_64000000_steps.zip"
PASSAGE = "data/stage6_jam_passage_seed14.npy"
OUT = "data/stage6_jam_states.npz"
NW, SEG, HOLD = 5, 0.75, 1.5


def snap(env, tips0):
    m, d = env.model, env.data
    return dict(mass=m.body_mass[env.ball_ids].copy(), friction=m.geom_friction[env.geom_ids, 0].copy(),
                qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(), act=d.act.copy(), tips0=tips0.copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    model = PPO.load(CK, device="cpu")
    states = {"jam": [], "passage": []}
    for seed in range(args.n):
        env = BaodingEnv3BallGuide(direction=-1, randomize=True, episode_seconds=30, level=4)
        v = VecNormalize.load(str(vecnormalize_path(CK)), DummyVecEnv([lambda: env]))
        v.training = v.norm_reward = False
        v.seed(seed)
        obs = v.reset()
        tips0, got = env.tips0.copy(), []
        for t in range(1, 1201):
            obs, _, done, _ = v.step(model.predict(obs, deterministic=True)[0])
            if done[0]:
                got = []        # the episode dropped: its snapshots led there
                break
            if t in (600, 800, 1000, 1200) and env.progress >= 2.0:
                got.append(snap(env, tips0))
            if seed == 14 and t == 1000:
                passage = _passage(env, tips0)
        states["jam"] += got
        print(f"seed {seed}: {len(got)} jam states", flush=True)
    states["passage"] = passage
    np.savez(OUT, **{f"{k}_{f}": np.array([s[f] for s in v_]) for k, v_ in states.items()
                     for f in ("mass", "friction", "qpos", "qvel", "ctrl", "act", "tips0")})
    print(f"saved {len(states['jam'])} jam + {len(states['passage'])} passage states to {OUT}")


def _passage(env, tips0):
    """Play the searched sequence from seed 14's stall (the env is left at its end); snapshots along it."""
    m, d = env.model, env.data
    base, wps = d.ctrl.copy(), [d.ctrl.copy()]
    for w in np.load(PASSAGE).reshape(NW, len(MOVING)):
        t = base.copy()
        t[MOVING] += w
        wps.append(np.clip(t, env.lo, env.hi))
    n, out, k = int(round(SEG / 0.02)), [], 0
    for i in range(NW):
        for s in range(n):
            d.ctrl[:] = wps[i] + (wps[i + 1] - wps[i]) * (s + 1) / n
            for _ in range(env.n_sub):
                mj.mj_step(m, d)
            k += 1
            if k % 12 == 1:
                out.append(snap(env, tips0))
    for s in range(int(HOLD / 0.02)):
        for _ in range(env.n_sub):
            mj.mj_step(m, d)
        k += 1
        if k % 12 == 1:
            out.append(snap(env, tips0))
    out.append(snap(env, tips0))
    return out


if __name__ == "__main__":
    main()
