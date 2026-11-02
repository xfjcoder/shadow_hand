"""Stage 5: PPO on `rl_env_baoding.BaodingEnv` (runs 01-06) or, with `--env 2`, on
`rl_env_baoding2.BaodingEnv2` (07+: work-based reward, fixed wrist, ball-size curriculum -- see the
docstrings for the rewards).

Usage:
    python3 train_stage5_baoding_rl.py --run-name baoding_07 --env 2 --timesteps 30000000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from rl_env_baoding import BaodingEnv
from rl_env_baoding2 import RADII, RADII_TO_50, BaodingEnv2, BaodingEnv3, BaodingEnv4, BaodingEnv5, BaodingEnv6, BaodingEnv7, BaodingEnv8, BaodingEnv9

HERE = Path(__file__).parent


class EpisodeStats(BaseCallback):
    """Logs the recent episodes' progress (rad, pi = a swap) and the fraction that kept both balls."""

    def _on_rollout_end(self):
        eps = list(self.model.ep_info_buffer)
        if eps:
            self.logger.record("baoding/progress", float(np.mean([e["progress"] for e in eps])))
            self.logger.record("baoding/progress_max", float(np.max([e["progress"] for e in eps])))
            self.logger.record("baoding/kept_balls", float(np.mean([e["success"] for e in eps])))
            if "swapped" in eps[0]:
                self.logger.record("baoding/swapped", float(np.mean([e["swapped"] for e in eps])))

    def _on_step(self):
        return True


class Curriculum(BaseCallback):
    """Ball-size curriculum (`rl_env_baoding2.RADII`): once `threshold` of the last `window` episodes
    at the current size kept both balls AND swapped at least once, every env moves to the next size."""

    def __init__(self, level=0, threshold=0.5, window=200, radii=RADII):
        super().__init__()
        self.level, self.threshold, self.window, self.recent = level, threshold, window, []
        self.radii = radii

    def _on_step(self):
        for info in self.locals["infos"]:
            if "swapped" in info and abs(info.get("radius", -1) - self.radii[self.level]) < 1e-9:
                self.recent.append(info["swapped"])
        self.recent = self.recent[-self.window:]
        if (len(self.recent) == self.window and np.mean(self.recent) >= self.threshold
                and self.level < len(self.radii) - 1):
            self.level += 1
            self.training_env.env_method("set_level", self.level)
            self.recent = []
            print(f"curriculum: level {self.level}, ball radius {self.radii[self.level] * 1000:.1f}mm "
                  f"at {self.num_timesteps} steps", flush=True)
        self.logger.record("baoding/ball_radius_mm", self.radii[self.level] * 1000)
        return True


def make_env(seed, direction, env_version=1, level=0):
    def _init():
        env = ({2: BaodingEnv2, 3: BaodingEnv3, 4: BaodingEnv4, 5: BaodingEnv5, 6: BaodingEnv6, 7: BaodingEnv7, 8: BaodingEnv8, 9: BaodingEnv9}[env_version](direction=direction, level=level) if env_version >= 2
               else BaodingEnv(direction=direction))
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--timesteps", type=int, default=20_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--direction", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="checkpoint .zip; its vecnormalize .pkl is found next to it")
    ap.add_argument("--checkpoint-freq", type=int, default=2_000_000)
    ap.add_argument("--env", type=int, default=1, choices=(1, 2, 3, 4, 5, 6, 7, 8, 9))
    ap.add_argument("--level", type=int, default=0, help="--env 2/3: starting curriculum level")
    args = ap.parse_args()
    torch.set_num_threads(1)   # more threads spin on the 16-obs rollout batches and starve the envs

    run_dir = HERE / "runs" / args.run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    env = VecMonitor(SubprocVecEnv([make_env(args.seed + i, args.direction, args.env, args.level)
                                    for i in range(args.n_envs)]),
                     info_keywords=("success", "progress") + (("swapped", "radius") if args.env >= 2 else ()))
    if args.resume:
        steps = Path(args.resume).stem.split("_")[1]
        env = VecNormalize.load(str(Path(args.resume).parent / f"ppo_vecnormalize_{steps}_steps.pkl"), env)
        model = PPO.load(args.resume, env=env, device="cpu")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma)
        model = PPO("MlpPolicy", env, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
                    n_epochs=args.n_epochs, gamma=args.gamma, policy_kwargs=dict(net_arch=[256, 256]), device="cpu",
                    tensorboard_log=str(run_dir / "logs"), seed=args.seed, verbose=1)
    callbacks = [EpisodeStats()] + ([Curriculum(args.level)] if 2 <= args.env <= 5 else
                                   [Curriculum(args.level, radii=RADII_TO_50)] if args.env >= 7 else [])   # env 6: fixed 50mm balls
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume,
                callback=CallbackList(callbacks + [CheckpointCallback(
                    save_freq=max(args.checkpoint_freq // args.n_envs, 1), save_path=str(run_dir / "checkpoints"),
                    name_prefix="ppo", save_vecnormalize=True)]))
    model.save(run_dir / "checkpoints" / "final")
    env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))
    print(f"done, saved to {run_dir}/checkpoints/")


if __name__ == "__main__":
    main()
