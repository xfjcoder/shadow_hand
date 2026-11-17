"""Evaluate a Stage 7A long-cup checkpoint (`rl_env_stage7_cup.CupEnv`): episodes from the settled cup
(`--settled`, no randomization) or from random moments of the rhythm's run; prints every ball's rim turn,
optionally records the episode whose slowest ball turned furthest (render with render_stage7.py).
`--zero` runs the rhythm alone (zero residual) for the baseline. `--env wave`: the finger-wave loop, one ball
(`--n-balls 2..4`: several).

Usage:
    python3 eval_stage7_cup.py runs/cup_01/checkpoints/ppo_16000000_steps.zip --settled --seconds 30 --record data/cup.npz
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl_env_stage7_cup import BALL_R, RIM_R, CupEnv
from rl_env_stage7_wave import MultiWaveEnv, WaveEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--settled", action="store_true", help="one episode from the settled cup, no randomization")
    ap.add_argument("--zero", action="store_true", help="zero residual: the rhythm alone")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--record", default=None)
    ap.add_argument("--env", default="cup", choices=("cup", "wave"))
    ap.add_argument("--flex-n", type=float, default=0.0, help="--n-balls > 1: the base wrist flex")
    ap.add_argument("--n-balls", type=int, default=1, help="--env wave: balls round the loop")
    args = ap.parse_args()
    p = Path(args.ckpt)
    vn = pickle.load(open(p.parent / f"ppo_vecnormalize_{p.stem.split('_')[-2]}_steps.pkl", "rb"))
    model = PPO.load(args.ckpt, device="cpu")
    wave = args.env == "wave"
    kw = dict(episode_seconds=args.seconds, randomize=not args.settled)
    env = (MultiWaveEnv(n=args.n_balls, flex_n=args.flex_n, **kw) if args.n_balls > 1 else WaveEnv(**kw)) if wave else CupEnv(**kw)
    center = env.c if wave else env.center
    n = 1 if args.settled else args.n
    kept, best = 0, None
    for ep in range(n):
        o, _ = env.reset(seed=args.seed + ep)
        log = {k: [] for k in ("qpos", "balls", "phase")}
        rim = np.zeros(4)
        while True:
            a = np.zeros(env.action_space.shape) if args.zero else \
                model.predict(vn.normalize_obs(o), deterministic=not args.stochastic)[0]
            o, r, term, trunc, info = env.step(a)
            b = env.w.balls()
            rim += np.linalg.norm(b[:, :2] - center, axis=1) > RIM_R
            log["qpos"].append(env.d.qpos.copy())
            log["balls"].append(b.copy())
            log["phase"].append(float(np.mean(env.turn)))
            if term or trunc:
                break
        kept += not info["lost"]
        deg = np.atleast_1d(np.degrees(env.turn))
        if wave and len(deg) > 1:
            print(f"episode {ep}: {'kept' if not info['lost'] else 'LOST'} {info['seconds']:5.1f} s  turns "
                  f"{np.round(deg).astype(int)} deg  (mean {deg.mean() / 360:+.2f} laps, slowest {deg.min() / 360:+.2f} laps, "
                  f"{deg.min() / info['seconds']:.0f} deg/s)", flush=True)
        elif wave:
            print(f"episode {ep}: {'kept' if not info['lost'] else 'LOST'} {info['seconds']:5.1f} s  turn {deg[0]:+.0f} deg "
                  f"({deg[0] / 360:+.2f} laps, {deg[0] / info['seconds']:.0f} deg/s)", flush=True)
        else:
            print(f"episode {ep}: {'kept' if not info['lost'] else 'LOST'} {info['seconds']:5.1f} s  turns "
                  f"{np.round(deg).astype(int)} deg  (mean {deg.mean():+.0f}, slowest {deg.min():+.0f})  "
                  f"on the rim {np.round(100 * rim / env.t).astype(int)}% of the time", flush=True)
        if best is None or info["turned_min"] > best[0]:
            best = (info["turned_min"], log)
    print(f"kept {kept}/{n} over {args.seconds:g} s")
    if args.record:
        np.savez(args.record, qpos=np.array(best[1]["qpos"]), balls=np.array(best[1]["balls"]),
                 phase=np.array(best[1]["phase"]), center=center,
                 rc=0.04 if wave else float(np.mean(np.linalg.norm(best[1]["balls"][0][:, :2] - center, axis=1))), ball=BALL_R,
                 roll_at="seam")
        print(f"recorded the episode whose slowest ball turned {best[0]:+.0f} deg to {args.record}")


if __name__ == "__main__":
    main()
