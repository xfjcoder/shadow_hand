"""Stage 8 step 2: how good must the ball state be? The three-ball teacher (fig8_27 @104M) run on MEASURED ball
positions: the true ones `delay` control steps (20 ms) old plus Gaussian noise of `sigma` cm per axis, through
stage8_sense.BallObserver (alpha-beta filter, latency-compensated). Kept / slowest-ball laps / slot error / touching
per (sigma, delay), on fresh seeds.

    python3 eval_stage8_noise.py runs/fig8_27/checkpoints/ppo_103999740_steps.zip --sigma 0 0.25 0.5 1 2 --delay 0 2 --n 40
"""
import argparse
import pickle
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl_env_stage7_fig8 import Fig8Env
from stage8_sense import BallObserver, run_episode

TEACHER = dict(n=3, track_rate=0.13, ball_act=True, episode_seconds=30, grace=3, tilt_k=0.3, pace="data/pace_fig8_3.npy")


def load(ck):
    p = Path(ck)
    vn = pickle.load(open(p.parent / f"ppo_vecnormalize_{p.stem.split('_')[-2]}_steps.pkl", "rb"))
    return PPO.load(ck, device="cpu"), vn


def job(a):
    ck, sigma, delay, seed, alpha, beta, truev = a
    model, vn = load(ck)
    env = Fig8Env(**TEACHER)
    rng = np.random.default_rng(seed + 99991)
    buf = deque(maxlen=delay + 1)

    def measure(e):
        buf.append(e.balls_n().copy())
        return buf[0] + rng.normal(0, sigma / 100, buf[0].shape)

    ob = BallObserver(env, latency=delay, alpha=alpha, beta=beta)
    if truev:                                   # diagnostic: the simulator's ball velocities, not estimated
        upd = ob.update
        ob.update = lambda m: (upd(m), setattr(ob, 'v', np.array([env.d.qvel[q:q + 3] for q in env.w.bv[:env.n]])))[0]
    info, log = run_episode(env, lambda o: model.predict(vn.normalize_obs(o), deterministic=True)[0], measure,
                            ob, seed)
    return not info["lost"], env.turn.min() / (2 * np.pi), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--sigma", type=float, nargs="+", default=[0.0])
    ap.add_argument("--delay", type=int, nargs="+", default=[0])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--true-v", action="store_true")
    ap.add_argument("--procs", type=int, default=15)
    args = ap.parse_args()
    with Pool(args.procs) as P:
        for d in args.delay:
            for s in args.sigma:
                R = P.map(job, [(args.ckpt, s, d, args.seed0 + i, args.alpha, args.beta, args.true_v) for i in range(args.n)])
                L = np.concatenate([r[2] for r in R])
                print(f"noise {s:4.2f} cm  delay {d * 20:3d} ms: kept {sum(r[0] for r in R)}/{args.n}  "
                      f"laps/30 s {np.mean([r[1] for r in R]):.2f}  worst slot err median {np.median(L[:, 1]):.3f}  "
                      f"touching {np.mean(L[:, 0] < 0.1):.0%}  gap median {np.median(L[:, 0]):.1f} cm  "
                      f"estimate err median {np.median(L[:, 2]):.2f} cm", flush=True)


if __name__ == "__main__":
    main()
