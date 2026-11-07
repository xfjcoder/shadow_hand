"""Stage 6 (rebuild, step 4): evaluate a `train_stage6_track.py` policy -- episodes from random demo
moments at a fixed target speed: kept all balls?, turn (deg), tracking error, fingertip use; optional GIF.

Usage:
    python3 eval_stage6_track.py runs/track_01/checkpoints/ppo_20000000_steps.zip --rate-level 4 --seconds 90 --n 8
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env_track3 import DEMO, RATES_DEG, SPEEDS, DemoTrackEnv, TrackEnv


def vecnorm_path(ckpt):
    p = Path(ckpt)
    if p.stem.startswith("ppo_"):
        return p.parent / f"ppo_vecnormalize_{p.stem.split('_')[-2]}_steps.pkl"
    return p.parent / f"{p.stem.replace('_policy', '')}_vecnormalize.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--rate-level", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--start", type=int, default=None, help="demo step to start from (default random)")
    ap.add_argument("--demo", default=str(DEMO))
    ap.add_argument("--mode", default="circle", choices=("circle", "demo"))
    ap.add_argument("--gif", default=None, help="GIF of the first episode")
    args = ap.parse_args()

    cls = DemoTrackEnv if args.mode == "demo" else TrackEnv
    env = cls(level=args.rate_level, episode_seconds=args.seconds, demo=args.demo)
    vn = VecNormalize.load(str(vecnorm_path(args.ckpt)), DummyVecEnv([lambda: cls(level=args.rate_level, demo=args.demo)]))
    label = f"{SPEEDS[args.rate_level]:g}x demo speed" if args.mode == "demo" else f"{RATES_DEG[args.rate_level]:.0f} deg/s"
    vn.training, vn.norm_reward = False, False
    model = PPO.load(args.ckpt, device="cpu")
    rng = np.random.default_rng(args.seed)
    rows = []
    for ep in range(args.n):
        start = args.start if args.start is not None else int(rng.integers(len(env.demo["qpos"]) - 1))
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)), options={"start": start})
        frames, renderer = [], (mj.Renderer(env.m, height=240, width=320) if args.gif and ep == 0 else None)
        while True:
            a, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=True)
            obs, r, term, trunc, info = env.step(a[0])
            if renderer is not None and env.t % 3 == 0:
                R = env.d.body("rh_palm").xmat.reshape(3, 3)
                cam = mj.MjvCamera()
                cam.type = mj.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = env.d.body("rh_palm").xpos + R @ np.array([0.0, -0.04, 0.07])
                imgs = []
                for az, el in ((180, -60), (220, -35)):
                    cam.distance, cam.azimuth, cam.elevation = 0.25, az, el
                    renderer.update_scene(env.d, camera=cam)
                    imgs.append(renderer.render().copy())
                im = Image.fromarray(np.concatenate(imgs, 1))
                ImageDraw.Draw(im).text((8, 6), f"RL, targets {label}  t={env.t * 0.02:.1f}s  "
                                                f"turned {np.degrees(env.turned):+.0f} deg = {abs(env.turned) / 2 / np.pi:.2f} circles  "
                                                f"err {info['err']:.1f}cm  tips {','.join(info['tips']) or '-'}", fill=(255, 255, 255))
                frames.append(im)
            if term or trunc:
                break
        if renderer is not None:
            renderer.close()
            frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=60, loop=0)
        rows.append(info)
        print(f"ep {ep} start {start:4d}  {'LOST' if info['lost'] else 'kept'} at {env.t * 0.02:5.1f}s  "
              f"turned {np.degrees(info['turned']):+6.0f} deg  err {info['mean_err']:.2f}cm  tip rate {info['tip_rate']:.2f}", flush=True)
    print(f"SUMMARY level {args.rate_level} ({label}, {args.seconds:.0f}s): kept {sum(not r['lost'] for r in rows)}/{len(rows)}  "
          f"turned mean {np.degrees(np.mean([r['turned'] for r in rows])):+.0f} max {np.degrees(max(r['turned'] for r in rows)):+.0f} deg  "
          f"err {np.mean([r['mean_err'] for r in rows]):.2f}cm  tip rate {np.mean([r['tip_rate'] for r in rows]):.2f}")


if __name__ == "__main__":
    main()
