"""Stage 6: evaluate a three-ball policy over randomized starts (`--start-seed` 0..n-1 of the demo:
triangle orientation, ball mass, friction, target noise), 20 s each, deterministic, a FRESH env per
seed (see eval_stage5_baoding.py). Prints per seed the cycles (2pi/3 each), progress, drop and when.

Usage:
    python3 eval_stage6_baoding.py runs/three_01/checkpoints/ppo_32000000_steps.zip --level 3 --n 32
"""
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from demo_stage5_baoding_rl import vecnormalize_path
from rl_env_baoding3 import BaodingEnv3Ball, BaodingEnv3BallLF, BaodingEnv3BallRelay, BaodingEnv3BallGuide, BaodingEnv3BallJam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--direction", type=float, default=-1.0)
    ap.add_argument("--env", type=int, default=10, choices=(10, 11, 12, 13, 14))
    args = ap.parse_args()
    model = PPO.load(args.checkpoint, device="cpu")
    rows = []
    for seed in range(args.n):
        env = {10: BaodingEnv3Ball, 11: BaodingEnv3BallLF, 12: BaodingEnv3BallRelay, 13: BaodingEnv3BallGuide, 14: BaodingEnv3BallJam}[args.env](direction=args.direction, randomize=True, episode_seconds=20.0, level=args.level)
        venv = VecNormalize.load(str(vecnormalize_path(args.checkpoint)), DummyVecEnv([lambda: env]))
        venv.training, venv.norm_reward = False, False
        venv.seed(seed)
        obs = venv.reset()
        t = 0
        while True:
            obs, _, done, infos = venv.step(model.predict(obs, deterministic=True)[0])
            t += 1
            if done[0]:
                break
        i = infos[0]
        rows.append((i["swaps"], i["dropped"], i["progress"], t * 0.02))
        print(f"seed {seed:2d} cycles {i['swaps']:2d} progress {i['progress']:+.2f} "
              + (f"DROPPED at {t * 0.02:.1f}s" if i["dropped"] else ""), flush=True)
    r = np.array(rows, dtype=float)
    print(f"{2000 * env.radius:.0f}mm, n={len(r)}: mean cycles {r[:, 0].mean():.2f}, >=1 cycle {int((r[:, 0] >= 1).sum())}, "
          f"dropped {int(r[:, 1].sum())} (median at {np.median(r[r[:, 1] > 0, 3]) if r[:, 1].any() else 0:.1f}s), "
          f"mean progress {r[:, 2].mean():+.2f} rad")


if __name__ == "__main__":
    main()
