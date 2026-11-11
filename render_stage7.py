"""Render a Stage 7A planner log (an npz with qpos per 20 ms control step) to a GIF: top and oblique views.

Usage:
    python3 render_stage7.py data/stage7_teacher_curlWm.npz media_full/stage7_teacher_curlWm.gif [--every 3] [--speed 1]
"""
import argparse

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

from stage7_layout import PALM_Z
from stage7_teacher import CTRL_DT, World


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("gif")
    ap.add_argument("--every", type=int, default=3, help="control steps per frame")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed")
    ap.add_argument("--title", default="two-hand MPC teacher")
    args = ap.parse_args()
    z = dict(np.load(args.npz))
    c = z["center"]
    w = World()
    r = mj.Renderer(w.m, height=270, width=360)
    frames = []
    for k in range(0, len(z["qpos"]), args.every):
        w.d.qpos[:] = z["qpos"][k]
        mj.mj_forward(w.m, w.d)
        imgs = []
        for az, el, dist in ((180, -88, 0.30), (180, -40, 0.34)):
            cam = mj.MjvCamera()
            cam.type = mj.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [c[0], c[1], PALM_Z + 0.02]
            cam.distance, cam.azimuth, cam.elevation = dist, az, el
            r.update_scene(w.d, camera=cam)
            imgs.append(r.render().copy())
        img = Image.fromarray(np.concatenate(imgs, 1))
        ImageDraw.Draw(img).text((8, 6), f"{args.title}  t={k * CTRL_DT:.1f}s  phase {np.degrees(z['phase'][k]):+.0f} deg",
                                 fill=(255, 255, 255))
        frames.append(img)
    r.close()
    frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 * args.every * CTRL_DT / args.speed), loop=0)
    print(f"saved {len(frames)} frames to {args.gif}")


if __name__ == "__main__":
    main()
