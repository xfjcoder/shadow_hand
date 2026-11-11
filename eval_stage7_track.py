"""Evaluate a Stage 7A two-hand tracking checkpoint: episodes from spread demo starts, optionally record one to npz
(render with render_stage7.py).

Usage:
    python3 eval_stage7_track.py runs/two_01/checkpoints/ppo_6000000_steps.zip --level 0 --seconds 30 --n 8
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
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--record", default=None, help="npz for the episode that turned furthest")
    args = ap.parse_args()
    p = Path(args.ckpt)
    steps = p.stem.split("_")[-2]
    vn = p.parent / f"ppo_vecnormalize_{steps}_steps.pkl"
    if not vn.exists():
        vn = p.parent / f"{p.stem}_vecnormalize.pkl"
    vn = pickle.load(open(vn, "rb"))
    model = PPO.load(args.ckpt, device="cpu")
    env = TwoHandTrackEnv(level=args.level, randomize=False, episode_seconds=args.seconds)
    rng = np.random.default_rng(args.seed)
    starts = rng.choice(len(env.start_p), size=args.n, p=env.start_p)
    kept, best = 0, None
    for s in starts:
        o, _ = env.reset(options={"start": int(s)})
        log = {k: [] for k in ("qpos", "balls", "phase")}
        while True:
            a, _ = model.predict(vn.normalize_obs(o), deterministic=not args.stochastic)
            o, r, term, trunc, info = env.step(a)
            log["qpos"].append(env.d.qpos.copy())
            log["balls"].append(env.w.balls().copy())
            log["phase"].append(info.get("turned", env.turned))
            if term or trunc:
                break
        kept += not info["lost"]
        deg = np.degrees(info["turned"])
        print(f"start {int(s):5d}: {'kept' if not info['lost'] else 'LOST'} {env.t * CONTROL_DT:5.1f} s  "
              f"turned {deg:+6.0f} deg  err {info['mean_err']:.2f} cm", flush=True)
        if best is None or deg > best[0]:
            best = (deg, log)
    print(f"kept {kept}/{args.n} over {args.seconds:g} s at {env.speed:g}x")
    if args.record:
        np.savez(args.record, qpos=np.array(best[1]["qpos"]), balls=np.array(best[1]["balls"]),
                 phase=np.array(best[1]["phase"]), center=env.center)
        print(f"recorded the {best[0]:+.0f} deg episode to {args.record}")


if __name__ == "__main__":
    main()
