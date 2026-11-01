"""Stage 5, runs 07+ (`rl_env_baoding2`): roll out a trained policy (deterministic, nominal balls of
the chosen curriculum level) and record a real-time GIF with the Stage 4 demo's two views. Prints
what the user judges the GIFs by: swaps, which digits do work on the balls (and in what order),
how far the fingertips move RELATIVE TO THE PALM (the wrist is fixed now), how far they stray from
the bowl, where the balls sit along the palm, pad facing and back-of-finger contacts.

Usage:
    python3 demo_stage5_baoding_rl2.py runs/baoding_07/checkpoints/ppo_4000000_steps.zip --level 0 \
        --record media/stage5_baoding_07_4M.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from demo_stage5_baoding_rl import vecnormalize_path
from hand_env import DIGITS
from rl_env_baoding2 import RADII, BaodingEnv2, BaodingEnv3, BaodingEnv4, BaodingEnv5, BaodingEnv6, BaodingEnv7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--level", type=int, default=0, help=f"ball radius index into {RADII}")
    ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--direction", type=float, default=1.0)
    ap.add_argument("--env", type=int, default=2, choices=(2, 3, 4, 5, 6, 7), help="the env version the run trained on")
    ap.add_argument("--start-seed", type=int, default=None,
                    help="start from a randomized training start (ball arrangement, mass, friction) chosen by this seed")
    ap.add_argument("--frame-every", type=int, default=5, help="control steps per GIF frame")
    args = ap.parse_args()

    env = {2: BaodingEnv2, 3: BaodingEnv3, 4: BaodingEnv4, 5: BaodingEnv5, 6: BaodingEnv6, 7: BaodingEnv7}[args.env](direction=args.direction, randomize=args.start_seed is not None, episode_seconds=args.seconds, level=args.level)
    venv = VecNormalize.load(str(vecnormalize_path(args.checkpoint)), DummyVecEnv([lambda: env]))
    venv.training, venv.norm_reward = False, False
    model = PPO.load(args.checkpoint, device="cpu")

    renderer = mj.Renderer(env.model, height=240, width=290) if args.record else None
    frames, steps, n_dorsal = [], 0, 0
    work_pos, work_neg, tips, zs, strays, face = {}, {}, [], [], [], {}
    doers = []   # digit doing the most forward work, when any does > 1 mW
    if args.start_seed is not None:
        venv.seed(args.start_seed)
    obs = venv.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = venv.step(action)
        info = infos[0]
        steps += 1
        for d, p in info["work"].items():
            if p > 0:
                work_pos[d] = work_pos.get(d, 0.0) + p * 0.02
            else:
                work_neg[d] = work_neg.get(d, 0.0) + p * 0.02
        top = max(info["work"].items(), key=lambda kv: kv[1], default=(None, 0.0))
        if top[1] > 1e-3 and (not doers or doers[-1] != top[0]):
            doers.append(top[0])
        n_dorsal += info["n_dorsal"]
        zs.append(info["ball_z"])
        strays.append(info["stray_cm"])
        for d, c in info["facing"].items():
            face[d] = face.get(d, 0.0) + c
        if not done[0]:
            tips.append(env._tips())
        if done[0]:
            break
        if renderer is not None and steps % args.frame_every == 0:
            data = env.data
            R = data.body("rh_palm").xmat.reshape(3, 3)
            lookat = data.body("rh_palm").xpos + R @ np.array([0.0, -0.04, 0.07])
            imgs = []
            for az, el in ((180, -60), (220, -35)):
                cam = mj.MjvCamera()
                cam.type = mj.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = lookat
                cam.distance, cam.azimuth, cam.elevation = 0.25, az, el
                renderer.update_scene(data, camera=cam)
                imgs.append(renderer.render().copy())
            img = Image.fromarray(np.concatenate(imgs, 1))
            draw = ImageDraw.Draw(img)
            draw.text((8, 6), f"RL policy  t={steps * 0.02:.1f}s  balls {2000 * env.radius:.0f}mm  "
                              f"turn {info['turn']}  swaps {info['swaps']}", fill=(255, 255, 255))
            draw.text((8, 20), f"arc gold {info['arcs'][0]:+.2f} red {info['arcs'][1]:+.2f}  pair "
                               f"{info['pair']:+.2f}  progress {info['progress']:+.2f} (swap = 3.14)",
                      fill=(255, 255, 255))
            frames.append(img)
    if renderer is not None:
        renderer.close()

    tips = np.array(tips) * 1000
    print(f"balls {2000 * env.radius:.0f}mm, {steps * 0.02:.1f}s, dropped={info['dropped']}, swaps {info['swaps']}: "
          f"progress {info['progress']:+.2f} (arcs {info['arcs'][0]:+.2f} {info['arcs'][1]:+.2f}, pair {info['pair']:+.2f})")
    print("work on the balls along the orbit, forward / braking (mJ): "
          + ", ".join(f"{d} {1000 * work_pos.get(d, 0):.1f}/{1000 * work_neg.get(d, 0):.1f}" for d in DIGITS)
          + f" | credited {1000 * info['work_credit']:.1f} mJ")
    print(f"main working digit, in order: {' '.join(doers[:40])}")
    print("fingertip motion relative to the palm (path length mm / std mm): "
          + ", ".join(f"{d} {np.linalg.norm(np.diff(tips[:, i], axis=0), axis=1).sum():.0f}/"
                      f"{np.linalg.norm(tips[:, i].std(0)):.1f}" for i, d in enumerate(DIGITS)))
    print(f"bowl: tips strayed past the {100 * env.shape_band:.1f}cm band, mean {np.mean(strays):.2f} cm (sum over tips) | "
          f"ball z along the palm, median {np.round(100 * np.median(zs, 0), 1)} cm (knuckles ~9.5) | "
          f"dorsal contacts {n_dorsal}")
    print("pads facing the balls (mean cos): " + ", ".join(f"{d} {face.get(d, 0) / steps:+.2f}" for d in DIGITS))
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        ms = round(args.frame_every * 20)
        frames[0].save(args.record, save_all=True, append_images=frames[1:], duration=ms, loop=0)
        print(f"saved {len(frames)} frames to {args.record} at {ms}ms/frame")


if __name__ == "__main__":
    main()
