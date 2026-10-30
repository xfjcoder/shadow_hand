"""Stage 4 demo GIF, real-time playback, two palm-aligned views (looking down onto the palm, and a
3/4 view), each ball's own cumulative angle printed on the frame:

  1. a `search_stage4_baoding.py` result from the Stage 3 bowl -- the press wave or, with 28
     params, the circular fingertip motion (`hand_env.BaodingCircle`);
  2. the forced-circulation check: the hand holds the bowl still while an ideal external push (a
     tangential force on each ball around the pair's midpoint, in the palm plane) tries to drive
     the balls round, first one way, then the other -- does the bowl let them circulate at all?

Usage:
    python3 demo_stage4_baoding.py data/stage4_baoding_s1.json --record media/stage4_wave_and_push.gif
    python3 demo_stage4_baoding.py data/stage4_circle_s0.json --no-push --record media/stage4_circle.gif
"""
import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

from hand_env import (BALLS, BOWL_BALLS_ANGLE, BOWL_BALLS_CENTER, TWO_BALL_SCENE, BallAngles,
                      BaodingCircle, BaodingWave, bowl_target, grasp_balls, push_forces, two_ball_places)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wave")
    ap.add_argument("--record", type=str, required=True)
    ap.add_argument("--wave-seconds", type=float, default=8.0)
    ap.add_argument("--push-seconds", type=float, default=4.0)
    ap.add_argument("--push-force", type=float, default=1.0)
    ap.add_argument("--no-push", action="store_true", help="only the controller segment")
    ap.add_argument("--frame-stride", type=int, default=40)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(TWO_BALL_SCENE)
    data = mj.MjData(model)
    dt = model.opt.timestep
    renderer = mj.Renderer(model, height=240, width=290)
    ball_ids = [model.body(b).id for b in BALLS]
    frames = []

    def frame(label, tracker):
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
        a = tracker.total
        draw.text((8, 6), label, fill=(255, 255, 255))
        draw.text((8, 20), f"gold {a[0]:+.2f} rad   red {a[1]:+.2f} rad   (swap needs both >= 3.14)",
                  fill=(255, 255, 255))
        frames.append(img)

    def segment(label, seconds, ctrl_fn, force_fn=None):
        grasp_balls(model, data, bowl_target(), two_ball_places(BOWL_BALLS_CENTER, BOWL_BALLS_ANGLE))
        tracker = BallAngles(model, data)
        for k in range(int(seconds / dt)):
            t = k * dt
            data.ctrl[:] = ctrl_fn(t)
            data.xfrc_applied[:] = 0.0
            if force_fn is not None:
                for bid, f in zip(ball_ids, force_fn(t, tracker)):
                    data.xfrc_applied[bid, :3] = f
            mj.mj_step(model, data)
            if k % 10 == 0:
                tracker.update(model, data)
            if k % args.frame_stride == 0:
                frame(label, tracker)
        print(f"{label}: angles {np.round(tracker.total, 2)}")

    x = json.loads(Path(args.wave).read_text())["x"]
    circle = len(x) == BaodingCircle.N_PARAMS
    wave = (BaodingCircle if circle else BaodingWave)(x)
    segment(f"1. scripted {'circular fingertip motion' if circle else 'press wave'} ({wave.freq:.2f} Hz)",
            args.wave_seconds, wave.ctrl)
    hold = bowl_target()
    for sign, name in () if args.no_push else ((1.0, "one way"), (-1.0, "other way")):
        segment(f"2. bowl held still, ideal {args.push_force:.1f} N push on each ball, {name}",
                args.push_seconds, lambda t: hold,
                lambda t, tr, s=sign: push_forces(model, data, tr.center, s * args.push_force * min(1.0, t / 0.5)))
    renderer.close()

    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    duration_ms = round(args.frame_stride * dt * 1000)
    frames[0].save(args.record, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"saved {len(frames)} frames to {args.record} at {duration_ms}ms/frame")


if __name__ == "__main__":
    main()
