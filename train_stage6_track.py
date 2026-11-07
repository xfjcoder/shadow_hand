"""Stage 6 (rebuild, step 4): PPO on `rl_env_track3.TrackEnv` (balls track targets moving round a circle),
optionally warm-started by behaviour cloning on the planner demonstration (`--bc`).

Curriculum: the targets' speed (`rl_env_track3.RATES_DEG`); every env moves up a level once `--threshold`
of the last `--window` episodes at the current level kept all three balls to the end with a mean
tracking error under 1.5cm.

`--mode demo` (`rl_env_track3.DemoTrackEnv`): each ball tracks its own position in the demo, played back
at the curriculum's speed-up (`SPEEDS`); a level passes on no ball lost and mean error < 1.5cm over 10 s.

Usage:
    python3 train_stage6_track.py --run-name track_01 --bc --timesteps 60000000
    python3 train_stage6_track.py --run-name track_01 --mode demo --resume runs/track_01/checkpoints/ppo_26000000_steps.zip --level 0
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from rl_env_track3 import DEMO, RATES_DEG, SPEEDS, DemoTrackEnv, TrackEnv

HERE = Path(__file__).parent
torch.set_num_threads(1)
INFO_KEYS = ("lost", "mean_err", "turned", "level", "tip_rate", "level_ok")


class Curriculum(BaseCallback):
    def __init__(self, level, threshold, window, levels=RATES_DEG, unit="deg/s"):
        super().__init__()
        self.levels, self.unit = levels, unit
        self.level, self.threshold, self.window, self.recent, self.eps = level, threshold, window, [], []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "level_ok" in info:
                self.eps.append(info)
                if info["level"] == self.level:
                    self.recent.append(info["level_ok"])
        self.recent = self.recent[-self.window:]
        if len(self.recent) == self.window and np.mean(self.recent) >= self.threshold and self.level < len(self.levels) - 1:
            self.level += 1
            self.training_env.env_method("set_level", self.level)
            self.recent = []
            print(f"curriculum: level {self.level}, targets at {self.levels[self.level]:g} {self.unit} at {self.num_timesteps} steps", flush=True)
        return True

    def _on_rollout_end(self):
        eps, self.eps = self.eps[-400:], []
        if eps:
            for k in INFO_KEYS[:3] + INFO_KEYS[4:]:
                self.logger.record(f"track/{k}", float(np.mean([e[k] for e in eps])))
            self.logger.record("track/turned_deg_max", float(np.degrees(max(e["turned"] for e in eps))))
        self.logger.record("track/level_value", self.levels[self.level])
        self.logger.record("track/level_ok_recent", float(np.mean(self.recent)) if self.recent else 0.0)


def make_env(seed, level, cls=TrackEnv, demo=DEMO, seconds=None):
    def _init():
        env = cls(level=level, demo=demo, episode_seconds=seconds)
        env.reset(seed=seed)
        return env
    return _init


def behaviour_clone(model, venv, epochs, lr, cls=TrackEnv, demo=DEMO):
    """Fit the policy's mean action to the demo's (obs, action) pairs; obs normalisation from the demo."""
    obs, act = cls(demo=demo).demo_pairs()
    venv.obs_rms.mean, venv.obs_rms.var, venv.obs_rms.count = obs.mean(0), obs.var(0) + 1e-6, 1e4
    x = torch.as_tensor(venv.normalize_obs(obs), dtype=torch.float32)
    y = torch.as_tensor(act, dtype=torch.float32)
    pol = model.policy
    opt = torch.optim.Adam(list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters()), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(x))
        tot = 0.0
        for i in range(0, len(x), 256):
            j = perm[i:i + 256]
            lat = pol.mlp_extractor.forward_actor(pol.extract_features(x[j], pol.pi_features_extractor))
            loss = ((pol.action_net(lat) - y[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"bc epoch {ep:4d}  mse {tot / len(x):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--timesteps", type=int, default=60_000_000)
    ap.add_argument("--n-envs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--mode", default="circle", choices=("circle", "demo"),
                    help="demo: each ball tracks its own demo position, played back SPEEDS x faster")
    ap.add_argument("--demo", default=str(DEMO), help="planner demo npz (data/stage6_demo_pads.npz: the pad demo)")
    ap.add_argument("--bc", action="store_true", help="behaviour-clone the planner demo first")
    ap.add_argument("--bc-epochs", type=int, default=200)
    ap.add_argument("--log-std", type=float, default=-1.0)
    ap.add_argument("--resume", default=None, help="checkpoint .zip; its vecnormalize .pkl is found next to it")
    ap.add_argument("--episode-seconds", type=float, default=None,
                    help="episode length (default: the env's own, 10 s in demo mode)")
    ap.add_argument("--checkpoint-freq", type=int, default=2_000_000)
    args = ap.parse_args()

    run = HERE / "runs" / args.run_name
    cls = DemoTrackEnv if args.mode == "demo" else TrackEnv
    env = VecMonitor(SubprocVecEnv([make_env(args.seed + i, args.level, cls, args.demo, args.episode_seconds) for i in range(args.n_envs)]),
                     info_keywords=())
    if args.resume:
        steps = Path(args.resume).stem.split("_")[-2]
        env = VecNormalize.load(str(Path(args.resume).parent / f"ppo_vecnormalize_{steps}_steps.pkl"), env)
        model = PPO.load(args.resume, env=env, device="cpu", tensorboard_log=str(run / "tb"))
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma)
        model = PPO("MlpPolicy", env, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
                    n_epochs=args.n_epochs, gamma=args.gamma, seed=args.seed, device="cpu", verbose=1,
                    tensorboard_log=str(run / "tb"),
                    policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512]), log_std_init=args.log_std))
        if args.bc:
            behaviour_clone(model, env, args.bc_epochs, 1e-3, cls, args.demo)
            model.save(run / "bc_policy")
            env.save(str(run / "bc_vecnormalize.pkl"))
    cbs = [Curriculum(args.level, args.threshold, args.window,
                      *((SPEEDS, "x demo speed") if args.mode == "demo" else (RATES_DEG, "deg/s"))),
           CheckpointCallback(max(args.checkpoint_freq // args.n_envs, 1), str(run / "checkpoints"), name_prefix="ppo",
                              save_vecnormalize=True)]
    model.learn(args.timesteps, callback=cbs, reset_num_timesteps=not args.resume)
    model.save(run / "final")
    env.save(str(run / "final_vecnormalize.pkl"))


if __name__ == "__main__":
    main()
