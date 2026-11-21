"""Stage 7A, the long cup: PPO residual policy on the swirl-and-flick rhythm (`rl_env_stage7_cup.CupEnv`), or
(--env wave) on the finger-wave loop, one ball (`rl_env_stage7_wave.WaveEnv`).

Usage:
    python3 train_stage7_cup.py --run-name cup_01 --timesteps 30000000
    python3 train_stage7_cup.py --env wave --run-name wave_01 --ref-deg 180 --timesteps 20000000
    python3 train_stage7_cup.py --env wave --n-balls 2 --run-name wave2_01 --ref-deg 180 --timesteps 30000000
    python3 train_stage7_cup.py --env fig8 --run-name fig8_01 --ref-deg 180 --episode-seconds 30 --drop 200
    python3 train_stage7_cup.py --env fig8 --n-balls 2 --run-name fig8_09 --resume CKPT --track-rate 0.1 \
        --slot-tol 0.4 --slot-tol-min 0.1 --ref-deg 180 --episode-seconds 30 --drop 200
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from rl_env_stage7_cup import CupEnv
from rl_env_stage7_fig8 import FLEX_8, Fig8Env
from rl_env_stage7_wave import MultiWaveEnv, WaveEnv
from train_stage7_loop import Log

HERE = Path(__file__).parent
torch.set_num_threads(1)


class SlotCurriculum(BaseCallback):
    """--slot-tol: tighten the slot tolerance by `step` (down to `tol_min`) once the median of the last `window`
    episodes lasted `seconds` or more."""

    def __init__(self, tol, tol_min, step, seconds, window=300):
        super().__init__()
        self.tol, self.tol_min, self.step, self.seconds, self.window, self.recent = tol, tol_min, step, seconds, window, []

    def _on_step(self):
        self.recent += [i["seconds"] for i in self.locals["infos"] if "seconds" in i]
        self.recent = self.recent[-self.window:]
        if len(self.recent) == self.window and np.median(self.recent) >= self.seconds and self.tol > self.tol_min + 1e-9:
            self.tol = round(max(self.tol - self.step, self.tol_min), 4)
            self.training_env.set_attr("slot_tol", self.tol)
            self.recent = []
            print(f"curriculum: slot tolerance {self.tol:g} laps at {self.num_timesteps} steps", flush=True)
        return True

    def _on_rollout_end(self):
        self.logger.record("loop/slot_tol", self.tol)
        self.logger.record("loop/seconds_median", float(np.median(self.recent)) if self.recent else 0.0)


def make_env(seed, seconds, ref_deg, kind="cup", n_balls=1, w_edge=0.0, w_fast=0.0, flex_n=None, drop=50.0,
             w_spread=0.0, track_rate=None, slot_tol=None, ball_act=False, grace=0.0, tilt_k=0.1, late_cost=0.0):
    def _init():
        if kind == "fig8":
            env = Fig8Env(n=n_balls, episode_seconds=seconds, ref_deg=ref_deg, drop=drop,
                          flex_n=FLEX_8 if flex_n is None else flex_n, w_spread=w_spread,
                          track_rate=track_rate, slot_tol=slot_tol, ball_act=ball_act, grace=grace, tilt_k=tilt_k, late_cost=late_cost)
        elif kind == "wave" and n_balls > 1:
            env = MultiWaveEnv(n=n_balls, episode_seconds=seconds, ref_deg=ref_deg, w_edge=w_edge, w_fast=w_fast,
                               flex_n=0.0 if flex_n is None else flex_n, drop=drop)
        else:
            env = (WaveEnv if kind == "wave" else CupEnv)(episode_seconds=seconds, ref_deg=ref_deg)
        env.reset(seed=seed)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--env", default="cup", choices=("cup", "wave", "fig8"))
    ap.add_argument("--n-balls", type=int, default=1, help="--env wave: balls round the loop (2-4: MultiWaveEnv)")
    ap.add_argument("--timesteps", type=int, default=30_000_000)
    ap.add_argument("--n-envs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--log-std", type=float, default=-1.5, help="residual noise (x RES): small, the rhythm is fragile")
    ap.add_argument("--episode-seconds", type=float, default=None)
    ap.add_argument("--ref-deg", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-edge", type=float, default=0.0, help="--n-balls > 1: penalty per ball past the palms' heel")
    ap.add_argument("--w-fast", type=float, default=0.0, help="--n-balls > 1: penalty per ball rushing wristward off the palm leg")
    ap.add_argument("--flex-n", type=float, default=None, help="the base wrist flex (default 0 for wave, -0.05 for fig8)")
    ap.add_argument("--drop", type=float, default=50.0, help="--n-balls > 1: the cost of a lost ball")
    ap.add_argument("--w-spread", type=float, default=0.0, help="--env fig8: cost of balls out of their slots (a clump)")
    ap.add_argument("--track-rate", type=float, default=None, help="--env fig8: a timetable round the 8, laps/s")
    ap.add_argument("--slot-tol", type=float, default=None, help="--track-rate: a ball this many laps out of its slot ends the episode")
    ap.add_argument("--slot-tol-min", type=float, default=None, help="--slot-tol: tighten it down to this (a curriculum)")
    ap.add_argument("--slot-step", type=float, default=0.05)
    ap.add_argument("--slot-seconds", type=float, default=15.0, help="--slot-tol-min: tighten once the median episode lasts this long")
    ap.add_argument("--ball-act", action="store_true", help="--env fig8: per-ball tilt actions (Fig8Env ball_act)")
    ap.add_argument("--grace", type=float, default=0.0, help="--track-rate: spin-up seconds before the timetable runs")
    ap.add_argument("--tilt-k", type=float, default=0.1, help="--ball-act: rad of hand tilt per unit of a per-ball action")
    ap.add_argument("--late-cost", type=float, default=0.0, help="--slot-tol: the cost of a ball out of its slot ending it")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    run = HERE / "runs" / args.run_name
    env = VecMonitor(SubprocVecEnv([make_env(args.seed + i, args.episode_seconds, args.ref_deg, args.env, args.n_balls, args.w_edge, args.w_fast, args.flex_n, args.drop, args.w_spread, args.track_rate, args.slot_tol, args.ball_act, args.grace, args.tilt_k, args.late_cost) for i in range(args.n_envs)]))
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
    if args.slot_tol and args.slot_tol_min:
        cbs.append(SlotCurriculum(args.slot_tol, args.slot_tol_min, args.slot_step, args.slot_seconds))
    model.learn(args.timesteps, callback=cbs, reset_num_timesteps=not args.resume)
    model.save(run / "final")
    env.save(str(run / "final_vecnormalize.pkl"))


if __name__ == "__main__":
    main()
