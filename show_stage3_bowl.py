"""Render a Stage 3 two-ball bowl (a `search_stage3_bowl.py` result) with cameras aimed relative to
the palm, since the wrist tilts the whole hand: straight down onto the palm, from the thumb side, from
the little-finger side, and a 3/4 view. Prints each fingertip's height above the palm and the
contacts on each ball.

Usage:
    python3 show_stage3_bowl.py data/stage3_bowl_rim.json media/stage3_bowl_b.png
"""
import argparse
import json

import mujoco as mj
import numpy as np
from PIL import Image

from hand_env import (ACTUATOR_NAMES, BALLS, N_ACTUATORS, TWO_BALL_SCENE, classify_contacts,
                      grasp_balls, two_ball_places)
from search_stage3_bowl import HOLD_TIME, tip_positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("out")
    args = ap.parse_args()
    b = json.loads(open(args.result).read())
    model = mj.MjModel.from_xml_path(TWO_BALL_SCENE)
    data = mj.MjData(model)
    target = np.zeros(N_ACTUATORS)
    for k, v in b["targets"].items():
        target[ACTUATOR_NAMES.index(k)] = v
    grasp_balls(model, data, target, two_ball_places(b["center"], b["angle"]))
    for _ in range(int(HOLD_TIME / model.opt.timestep)):
        mj.mj_step(model, data)

    for dg, t in zip(("ff", "mf", "rf", "lf", "th"), tip_positions(model, data)):
        print(f"{dg} tip: {100 * t[1]:.1f}cm above the palm")
    for ball in BALLS:
        for c in sorted(classify_contacts(model, data, ball), key=lambda c: -c[3]):
            if c[3] >= 0.05:
                print(f"{ball}: {c[0]} {c[1]} {c[2]} {c[3]:.2f}N")

    R = data.body("rh_palm").xmat.reshape(3, 3)
    lookat = data.body("rh_palm").xpos + R @ np.array([0.0, -0.04, 0.07])
    renderer = mj.Renderer(model, height=420, width=560)
    imgs = []
    for az, el in ((180, -60), (90, -10), (270, -10), (220, -35)):
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = 0.25, az, el
        renderer.update_scene(data, camera=cam)
        imgs.append(renderer.render().copy())
    renderer.close()
    Image.fromarray(np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:], 1)], 0)).save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
