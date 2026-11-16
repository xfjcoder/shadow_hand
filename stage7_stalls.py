"""Stage 7A, planner-corrected stalls, step 1: run a tracking policy from spread demo starts and save the
states where it stalls (the balls turning < STALL_DEG in STALL_S s), each as a one-step "log" npz that
`stage7_teacher.py --resume <file> --resume-at 0` can plan on from.

Usage:
    python3 stage7_stalls.py runs/two_03/checkpoints/ppo_21999945_steps.zip --n 40 --out data/stalls
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rl_env_stage7 import CONTROL_DT, TwoHandTrackEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=40, help="stall states to save")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--out", default="data/stalls")
    args = ap.parse_args()
    p = Path(args.ckpt)
    vn = pickle.load(open(p.parent / f"ppo_vecnormalize_{p.stem.split('_')[-2]}_steps.pkl", "rb"))
    model = PPO.load(args.ckpt, device="cpu")
    env = TwoHandTrackEnv(randomize=False, episode_seconds=args.seconds)
    rc = float(np.mean(np.linalg.norm(env.demo["balls"][0][:, :2] - env.center, axis=1)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    k = int(round(env.STALL_S / CONTROL_DT))
    saved, tries = 0, 0
    while saved < args.n and tries < 20 * args.n:
        tries += 1
        o, _ = env.reset(options={"start": int(rng.choice(len(env.start_p), p=env.start_p))})
        hist = []
        while True:
            a, _ = model.predict(vn.normalize_obs(o), deterministic=not args.stochastic)
            o, r, term, trunc, info = env.step(a)
            hist.append(env.turned)
            if len(hist) > k and env.turned - hist[-1 - k] < np.radians(env.STALL_DEG):
                np.savez(out / f"stall_{saved:03d}.npz", qpos=env.d.qpos[None].copy(), qvel=env.d.qvel[None].copy(),
                         ctrl=env.d.ctrl[None].copy(), balls=env.w.balls()[None].copy(), phase=np.zeros(1),
                         center=env.center, rc=rc, sign=1.0, moment=env._aligned(env.turned),
                         demo_turn=np.degrees(env.phase[env._aligned(env.turned)]))
                print(f"stall {saved:3d}: start {env.i0:5d}  after {env.t * CONTROL_DT:4.1f} s  turned "
                      f"{np.degrees(env.turned):+5.0f}  at demo turn {np.degrees(env.phase[env._aligned(env.turned)]):5.0f} deg",
                      flush=True)
                saved += 1
                break
            if term or trunc:
                break
    print(f"saved {saved} stall states from {tries} episodes to {out}")


if __name__ == "__main__":
    main()
