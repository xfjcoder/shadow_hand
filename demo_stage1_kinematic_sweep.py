"""Stage 1 demo: each of the Shadow Hand's 20 actuators swept, one at a time, to the top and bottom of
its range and back to rest, under real closed-loop physics (mj_step with gravity), then the scene's
free object dropped onto the open palm. Real-time playback: one frame every `frame_stride` physics
steps, GIF duration = frame_stride * timestep.

Usage:
    python3 demo_stage1_kinematic_sweep.py --record media/stage1_kinematic_sweep.gif
"""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import ACTUATOR_NAMES, SCENE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--seconds-per-target", type=float, default=0.2)
    ap.add_argument("--frame-stride", type=int, default=40)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(SCENE)
    data = mj.MjData(model)
    dt = model.opt.timestep
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    rest = np.clip(0.0, lo, hi)
    # Hold the object out of the way during the sweep, drop it at the end.
    obj_q = model.jnt_qposadr[model.body("object").jntadr[0]]
    obj_v = model.jnt_dofadr[model.body("object").jntadr[0]]
    parked = np.array([0.3, 0.0, 0.25])

    renderer = mj.Renderer(model, height=240, width=300)
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.35, 0.0, 0.03]
    cam.distance = 0.26
    cam.elevation = -40
    frames, step = [], [0]

    def run(seconds, park=True):
        for _ in range(int(seconds / dt)):
            if park:
                data.qpos[obj_q:obj_q + 3] = parked
                data.qvel[obj_v:obj_v + 6] = 0.0
            mj.mj_step(model, data)
            step[0] += 1
            if step[0] % args.frame_stride == 0:
                cam.azimuth = 200 + 3.0 * step[0] * dt
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())

    data.ctrl[:] = rest
    for i in range(model.nu):
        for target in (hi[i], lo[i], rest[i]):
            data.ctrl[i] = target
            run(args.seconds_per_target)
    run(2.0, park=False)
    renderer.close()
    print(f"swept {len(ACTUATOR_NAMES)} actuators; object came to rest at "
          f"{np.round(data.body('object').xpos, 3)}; {len(frames)} frames")

    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in frames]
        duration_ms = round(args.frame_stride * dt * 1000)
        imgs[0].save(args.record, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)
        print(f"Saved {len(frames)} frames to {args.record} at {duration_ms}ms/frame")


if __name__ == "__main__":
    main()
