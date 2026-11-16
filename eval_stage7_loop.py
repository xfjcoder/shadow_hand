"""Evaluate a Stage 7A open-hand loop checkpoint (`rl_env_stage7_loop.LoopEnv`): episodes from random spins,
optionally record the one that turned furthest (render with render_stage7.py).

Usage:
    python3 eval_stage7_loop.py runs/loop_01/checkpoints/ppo_4000000_steps.zip --seconds 30 --n 8 --record data/loop.npz
"""
import argparse
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
from stable_baselines3 import PPO

from rl_env_stage7_loop import BALL_R, LoopEnv, LoopTrackEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--spin", type=float, default=None, help="deg/s start spin for every episode (default: random)")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--record", default=None)
    ap.add_argument("--demo", default=None, help="a --track checkpoint: its demo (episodes start at the layout, demo moment 0)")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--jitter", action="store_true", help="--demo: jittered layout and random start spin per episode (else the one exact layout)")
    args = ap.parse_args()
    p = Path(args.ckpt)
    vn = pickle.load(open(p.parent / f"ppo_vecnormalize_{p.stem.split('_')[-2]}_steps.pkl", "rb"))
    model = PPO.load(args.ckpt, device="cpu")
    if args.demo:
        env = LoopTrackEnv(demo=args.demo, level=args.level, episode_seconds=args.seconds, randomize=False)
    else:
        env = LoopEnv(episode_seconds=args.seconds, spin_max=args.spin if args.spin is not None else 45.0,
                      randomize=args.spin is None)
    kept, best = 0, None
    for ep in range(args.n):
        o, _ = env.reset(seed=args.seed + ep)
        if args.jitter and args.demo:                    # the track env starts at the one exact layout: jitter the balls
            for q in env.w.bq:
                env.d.qpos[q:q + 2] += env.np_random.normal(0, 0.002, 2)
            mj.mj_forward(env.m, env.d)
            env.th_prev = env._angles(env.w.balls())
            o = env._obs()
        log = {k: [] for k in ("qpos", "balls", "phase")}
        while True:
            a, _ = model.predict(vn.normalize_obs(o), deterministic=not args.stochastic)
            o, r, term, trunc, info = env.step(a)
            log["qpos"].append(env.d.qpos.copy())
            log["balls"].append(env.w.balls().copy())
            log["phase"].append(float(np.radians(env.turn.mean())))
            if term or trunc:
                break
        kept += not info["lost"]
        print(f"episode {ep}: {'kept' if not info['lost'] else 'LOST'} {info['seconds']:5.1f} s  turned mean "
              f"{info['turned']:+6.0f}  slowest {info['turned_min']:+6.0f} deg", flush=True)
        if best is None or info["turned_min"] > best[0]:
            best = (info["turned_min"], log)
    print(f"kept {kept}/{args.n} over {args.seconds:g} s")
    if args.record:
        np.savez(args.record, qpos=np.array(best[1]["qpos"]), balls=np.array(best[1]["balls"]),
                 phase=np.array(best[1]["phase"]), center=env.center, rc=env.rc, ball=BALL_R)
        print(f"recorded the episode whose slowest ball turned {best[0]:+.0f} deg to {args.record}")


if __name__ == "__main__":
    main()
