"""Stage 7A, the long cup: PPO residual policy on the swirl-and-flick rhythm (`rl_env_stage7_cup.CupEnv`), or
(--env wave) on the finger-wave loop, one ball (`rl_env_stage7_wave.WaveEnv`).

Usage:
    python3 train_stage7_cup.py --run-name cup_01 --timesteps 30000000
    python3 train_stage7_cup.py --env wave --run-name wave_01 --ref-deg 180 --timesteps 20000000
"""
import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from rl_env_stage7_cup import CupEnv
from rl_env_stage7_wave import WaveEnv
from train_stage7_loop import Log

HERE = Path(__file__).parent
torch.set_num_threads(1)


def make_env(seed, seconds, ref_deg, kind="cup"):
    def _init():
        env = (WaveEnv if kind == "wave" else CupEnv)(episode_seconds=seconds, ref_deg=ref_deg)
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--env", default="cup", choices=("cup", "wave"))
    ap.add_argument("--timesteps", type=int, default=30_000_000)
    ap.add_argument("--n-envs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--log-std", type=float, default=-1.5, help="residual noise (x RES): small, the rhythm is fragile")
    ap.add_argument("--episode-seconds", type=float, default=None)
    ap.add_argument("--ref-deg", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    run = HERE / "runs" / args.run_name
    env = VecMonitor(SubprocVecEnv([make_env(args.seed + i, args.episode_seconds, args.ref_deg, args.env) for i in range(args.n_envs)]))
    if args.resume:
        steps = Path(args.resume).stem.split("_")[-2]
        env = VecNormalize.load(str(Path(args.resume).parent / f"ppo_vecnormalize_{steps}_steps.pkl"), env)
        model = PPO.load(args.resume, env=env, device="cpu", tensorboard_log=str(run / "tb"))
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=args.gamma)
        model = PPO("MlpPolicy", env, learning_rate=args.lr, n_steps=2048, batch_size=8192, n_epochs=5, gamma=args.gamma,
                    seed=args.seed, device="cpu", verbose=1, tensorboard_log=str(run / "tb"),
                    policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512]), log_std_init=args.log_std))
        with torch.no_grad():                            # start AT the rhythm: zero mean residual
            model.policy.action_net.weight.mul_(0.01)
            model.policy.action_net.bias.zero_()
    cbs = [Log(), CheckpointCallback(max(2_000_000 // args.n_envs, 1), str(run / "checkpoints"), name_prefix="ppo",
                                     save_vecnormalize=True)]
    model.learn(args.timesteps, callback=cbs, reset_num_timesteps=not args.resume)
    model.save(run / "final")
    env.save(str(run / "final_vecnormalize.pkl"))


if __name__ == "__main__":
    main()
