"""Stage 5: evaluate a policy over randomized starts (`--start-seed` 0..n-1 of the demo: start
arrangement, ball mass, friction, target noise), 20 s each, deterministic. Each seed gets a FRESH env:
`reset(seed)` on a used env picks a different start than the first reset with that seed. Prints per
seed the start arrangement (by the settled pair-line angle: bowl / corner / third), swaps, drop, and a
summary per arrangement.

Usage:
    python3 eval_stage5_baoding.py runs/baoding_13/checkpoints/ppo_44000000_steps.zip --env 8 --level 6 --n 32
"""
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from demo_stage5_baoding_rl import vecnormalize_path
from hand_env import balls_in_palm
from rl_env_baoding2 import BaodingEnv2, BaodingEnv3, BaodingEnv4, BaodingEnv5, BaodingEnv6, BaodingEnv7, BaodingEnv8, BaodingEnv9

ARRANGEMENTS = {"bowl": (0.38, -2.77), "corner": (-0.75, 2.39), "third": (1.7, -1.45)}   # pair angle, dir -1


def arrangement(a):
    gap = lambda b: abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return min(ARRANGEMENTS, key=lambda k: min(gap(b) for b in ARRANGEMENTS[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--env", type=int, default=8, choices=(2, 3, 4, 5, 6, 7, 8, 9))
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--direction", type=float, default=-1.0)
    args = ap.parse_args()
    cls = {2: BaodingEnv2, 3: BaodingEnv3, 4: BaodingEnv4, 5: BaodingEnv5, 6: BaodingEnv6, 7: BaodingEnv7,
           8: BaodingEnv8, 9: BaodingEnv9}[args.env]
    model = PPO.load(args.checkpoint, device="cpu")
    rows = []
    for seed in range(args.n):
        env = cls(direction=args.direction, randomize=True, episode_seconds=20.0, level=args.level)
        venv = VecNormalize.load(str(vecnormalize_path(args.checkpoint)), DummyVecEnv([lambda: env]))
        venv.training, venv.norm_reward = False, False
        venv.seed(seed)
        obs = venv.reset()
        p1, p2 = balls_in_palm(env.model, env.data)
        kind = arrangement(np.arctan2(p2[2] - p1[2], p2[0] - p1[0]))
        while True:
            obs, _, done, infos = venv.step(model.predict(obs, deterministic=True)[0])
            if done[0]:
                break
        i = infos[0]
        rows.append((kind, i["swaps"], i["dropped"]))
        print(f"seed {seed:2d} {kind:6s} swaps {i['swaps']:2d} dropped {i['dropped']} progress {i['progress']:+.2f}",
              flush=True)
    for kind in ARRANGEMENTS:
        r = [x for x in rows if x[0] == kind]
        if r:
            print(f"{kind}: n={len(r)}, mean swaps {np.mean([x[1] for x in r]):.2f}, "
                  f">=1 swap {sum(x[1] >= 1 for x in r)}, dropped {sum(x[2] for x in r)}")
    print(f"all: n={len(rows)}, mean swaps {np.mean([x[1] for x in rows]):.2f}, "
          f">=1 swap {sum(x[1] >= 1 for x in rows)}, dropped {sum(x[2] for x in rows)}")


if __name__ == "__main__":
    main()
