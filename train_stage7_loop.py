"""Stage 7A, open-hand loop, step 2: PPO on `rl_env_stage7_loop.LoopEnv` (the momentum reward, no demo).

Usage:
    python3 train_stage7_loop.py --run-name loop_01 --timesteps 60000000
    python3 train_stage7_loop.py --run-name loop_01b --resume runs/loop_01/checkpoints/ppo_20000000_steps.zip --episode-seconds 20
    python3 train_stage7_loop.py --run-name track_l1 --track --demo data/stage7_spaced_r4c.npz --demo-frac 0.7 --bc --ref-deg 5 \
        --alive 0.1 --drop 100 --gamma 0.995
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from rl_env_stage7_loop import SPEEDS, LoopEnv, LoopTrackEnv

HERE = Path(__file__).parent
torch.set_num_threads(1)
KEYS = ("lost", "turned", "turned_min", "seconds")
TKEYS = ("mean_err", "level_ok", "spread", "ref_err", "late", "touch", "touch_end")


class Curriculum(BaseCallback):
    """--track: the demo's playback speed (SPEEDS) steps up once `threshold` of the last `window` episodes at the
    current level passed (no loss, error < 1.5cm, at least half the demo's turn)."""

    def __init__(self, level, threshold, window):
        super().__init__()
        self.level, self.threshold, self.window, self.recent = level, threshold, window, []

    def _on_step(self):
        for i in self.locals["infos"]:
            if "level_ok" in i and i["level"] == self.level:
                self.recent.append(i["level_ok"])
        self.recent = self.recent[-self.window:]
        if len(self.recent) == self.window and np.mean(self.recent) >= self.threshold and self.level + 1 < len(SPEEDS):
            self.level += 1
            self.training_env.env_method("set_level", self.level)
            self.recent = []
            print(f"curriculum: level {self.level}, demo at {SPEEDS[self.level]:g}x at {self.num_timesteps} steps", flush=True)
        return True

    def _on_rollout_end(self):
        self.logger.record("loop/level", self.level)
        self.logger.record("loop/level_ok_recent", float(np.mean(self.recent)) if self.recent else 0.0)


class Log(BaseCallback):
    def __init__(self):
        super().__init__()
        self.eps = []

    def _on_step(self):
        self.eps += [i for i in self.locals["infos"] if "lost" in i]
        return True

    def _on_rollout_end(self):
        eps, self.eps = self.eps[-400:], []
        for k in KEYS + TKEYS:
            if eps and k in eps[0]:
                self.logger.record(f"loop/{k}", float(np.mean([e[k] for e in eps])))


def behaviour_clone(model, venv, env, epochs, lr=1e-3):
    """Fit the policy's mean action to the demo's (obs, action) pairs; obs normalisation from the demo."""
    obs, act = env.demo_pairs()
    venv.obs_rms.mean, venv.obs_rms.var, venv.obs_rms.count = obs.mean(0), obs.var(0) + 1e-6, 1e4
    x = torch.as_tensor(venv.normalize_obs(obs), dtype=torch.float32)
    y = torch.as_tensor(act)
    pol = model.policy
    opt = torch.optim.Adam(list(pol.mlp_extractor.policy_net.parameters()) + list(pol.action_net.parameters()), lr=lr)
    for ep in range(epochs):
        perm, tot = torch.randperm(len(x)), 0.0
        for i in range(0, len(x), 256):
            j = perm[i:i + 256]
            loss = ((pol.action_net(pol.mlp_extractor.forward_actor(x[j])) - y[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"bc epoch {ep:4d}  mse {tot / len(x):.4f}", flush=True)


def make_env(seed, seconds, spin_max, alive=0.5, w_turn=1.0, drop=20.0, prog_max=1.5, demo=None, demo_frac=0.0, ref_deg=45.0,
             track=False, level=0):
    def _init():
        kw = dict(episode_seconds=seconds, spin_max=spin_max, alive=alive, w_turn=w_turn, drop=drop, prog_max=prog_max,
                  demo=demo, demo_frac=demo_frac, ref_deg=ref_deg)
        env = LoopTrackEnv(level=level, **kw) if track else LoopEnv(**kw)
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--timesteps", type=int, default=60_000_000)
    ap.add_argument("--n-envs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--log-std", type=float, default=-2.0, help="small: random actions of std 0.3 drop a ball in ~1 s")
    ap.add_argument("--episode-seconds", type=float, default=None)
    ap.add_argument("--spin-max", type=float, default=45.0)
    ap.add_argument("--alive", type=float, default=0.5)
    ap.add_argument("--w-turn", type=float, default=1.0)
    ap.add_argument("--drop", type=float, default=20.0)
    ap.add_argument("--ref-deg", type=float, default=45.0, help="deg/s the turn reward calls full rate (make it reachable)")
    ap.add_argument("--demo", default=None, help="planner run in this scene: demo starts and/or --bc")
    ap.add_argument("--demo-frac", type=float, default=0.0, help="share of resets from a random demo moment")
    ap.add_argument("--bc", action="store_true", help="behaviour-clone the demo first")
    ap.add_argument("--track", action="store_true", help="LoopTrackEnv: also track the demo (DeepMimic), speed curriculum")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--bc-epochs", type=int, default=200)
    ap.add_argument("--prog-max", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--checkpoint-freq", type=int, default=2_000_000)
    args = ap.parse_args()
    run = HERE / "runs" / args.run_name
    env = VecMonitor(SubprocVecEnv([make_env(args.seed + i, args.episode_seconds, args.spin_max, args.alive, args.w_turn, args.drop, args.prog_max,
                                            args.demo, args.demo_frac, args.ref_deg, args.track, args.level) for i in range(args.n_envs)]))
    if args.resume:
        steps = Path(args.resume).stem.split("_")[-2]
        env = VecNormalize.load(str(Path(args.resume).parent / f"ppo_vecnormalize_{steps}_steps.pkl"), env)
        model = PPO.load(args.resume, env=env, device="cpu", tensorboard_log=str(run / "tb"))
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma)
        model = PPO("MlpPolicy", env, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=5,
                    gamma=args.gamma, seed=args.seed, device="cpu", verbose=1, tensorboard_log=str(run / "tb"),
                    policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512]), log_std_init=args.log_std))
        if args.bc:
            bc_env = (LoopTrackEnv(demo=args.demo, episode_seconds=args.episode_seconds) if args.track
                      else LoopEnv(demo=args.demo, episode_seconds=args.episode_seconds))
            behaviour_clone(model, env, bc_env, args.bc_epochs)
            model.save(run / "bc_policy")
            env.save(str(run / "bc_vecnormalize.pkl"))
    cbs = [Log(), *([Curriculum(args.level, args.threshold, args.window)] if args.track else []), CheckpointCallback(max(args.checkpoint_freq // args.n_envs, 1), str(run / "checkpoints"), name_prefix="ppo",
                                     save_vecnormalize=True)]
    model.learn(args.timesteps, callback=cbs, reset_num_timesteps=not args.resume)
    model.save(run / "final")
    env.save(str(run / "final_vecnormalize.pkl"))


if __name__ == "__main__":
    main()
