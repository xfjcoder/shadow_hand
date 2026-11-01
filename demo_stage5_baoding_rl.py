"""Stage 5: roll out a trained baoding policy (deterministic, nominal balls) from the bowl and record
a real-time GIF with the same two palm-aligned views as the Stage 4 demo, each ball's own angle and
the progress (`rl_env_baoding` reward) printed on the frames. Also prints, per digit, the fraction
of steps it touched a ball with its pad, and the count of back-of-segment contacts.

Usage:
    python3 demo_stage5_baoding_rl.py runs/baoding_01/checkpoints/ppo_4000000_steps.zip \
        --record media/stage5_baoding_rl.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env_baoding import BaodingEnv


def vecnormalize_path(ckpt):
    ckpt = Path(ckpt)
    if ckpt.stem == "final":
        return ckpt.parent / "vecnormalize_final.pkl"
    return ckpt.parent / f"ppo_vecnormalize_{ckpt.stem.split('_')[1]}_steps.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--record", default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--direction", type=float, default=1.0)
    ap.add_argument("--frame-every", type=int, default=5, help="control steps per GIF frame")
    args = ap.parse_args()

    env = BaodingEnv(direction=args.direction, randomize=False, episode_seconds=args.seconds)
    venv = VecNormalize.load(str(vecnormalize_path(args.checkpoint)), DummyVecEnv([lambda: env]))
    venv.training, venv.norm_reward = False, False
    model = PPO.load(args.checkpoint, device="cpu")

    renderer = mj.Renderer(env.model, height=240, width=290) if args.record else None
    frames, pad_steps, n_dorsal, steps = [], {}, 0, 0
    fwd, back, n_touch, gaps, face, n_all, devs, pushers = {}, {}, 0, [], {}, 0, [], []
    turn_pushes, palm_n, zs = [], {}, []
    obs = venv.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = venv.step(action)
        info = infos[0]
        steps += 1
        for d in info["good"]:
            pad_steps[d] = pad_steps.get(d, 0) + 1
        n_dorsal += info["n_dorsal"]
        n_touch += info["touching"]
        gaps.append(info["gap"])
        n_all += info["n_pad"] == 5
        devs.append(info["bowl_dev"])
        turn_pushes.append(info["turn_push"])
        zs.append(info["ball_z"])
        for b in info["on_palm"]:
            palm_n[b] = palm_n.get(b, 0) + 1
        if info["pusher"] and (not pushers or pushers[-1] != info["pusher"]):
            pushers.append(info["pusher"])
        for d, c in info["facing"].items():
            face[d] = face.get(d, 0.0) + c
        for d, w in info["push"].items():
            fwd[d], back[d] = fwd.get(d, 0) + (w > 0.1), back.get(d, 0) + (w < -0.1)
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
            a = info["angles"]
            draw.text((8, 6), f"RL policy  t={steps * 0.02:.1f}s  pusher {info['pusher'] or '-'}  "
                              f"relays {info['relays']}  turn {info['turn']}", fill=(255, 255, 255))
            draw.text((8, 20), f"arc gold {info['arcs'][0]:+.2f} red {info['arcs'][1]:+.2f}  pair "
                               f"{info['pair']:+.2f}  progress {info['progress']:+.2f} (swap = 3.14)",
                      fill=(255, 255, 255))
            frames.append(img)
    if renderer is not None:
        renderer.close()

    a = info["angles"]
    print(f"{steps * 0.02:.1f}s, dropped={info['dropped']}: angles gold {a[0]:+.2f} red {a[1]:+.2f}, "
          f"progress {info['progress']:+.2f} (arcs {info['arcs'][0]:+.2f} {info['arcs'][1]:+.2f}, "
          f"pair {info['pair']:+.2f})")
    print("pad contact (fraction of steps): "
          + ", ".join(f"{d} {pad_steps.get(d, 0) / steps:.2f}" for d in ("ff", "mf", "rf", "lf", "th"))
          + f" | dorsal contacts {n_dorsal}")
    print("pad pushing the ball round forward / BACKWARD (fraction of steps, > 0.1 rad/s): "
          + ", ".join(f"{d} {fwd.get(d, 0) / steps:.2f}/{back.get(d, 0) / steps:.2f}"
                      for d in ("ff", "mf", "rf", "lf", "th"))
          + f" | balls touching {n_touch / steps:.2f}, median gap {1000 * np.median(gaps):.1f}mm")
    print("pads facing the balls (mean cos, +1 inside faces them, -1 back does): "
          + ", ".join(f"{d} {face.get(d, 0) / steps:+.2f}" for d in ("ff", "mf", "rf", "lf", "th"))
          + f" | all five pads on balls {n_all / steps:.2f}")
    print(f"relays (bonus handoffs) {info['relays']}, pusher sequence: {' '.join(pushers[:40])}"
          f" | RMS target offset from the bowl {np.sqrt(np.mean(devs)):.2f} rad")
    print(f"relay clock: turn digit pushing forward (mean tanh) {np.mean(turn_pushes):+.2f}, "
          f"fraction > 0.2: {np.mean(np.array(turn_pushes) > 0.2):.2f}")
    print("balls on the palm (fraction): " + ", ".join(f"{b} {palm_n.get(b, 0) / steps:.2f}" for b in ("ball1", "ball2"))
          + f" | ball z along the palm, median (cm; knuckles ~9.5): {np.round(100 * np.median(zs, 0), 1)}")
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        ms = round(args.frame_every * 20)
        frames[0].save(args.record, save_all=True, append_images=frames[1:], duration=ms, loop=0)
        print(f"saved {len(frames)} frames to {args.record} at {ms}ms/frame")


if __name__ == "__main__":
    main()
