"""Stage 7A, step 1: the touching-palms layout for one four-ball loop across both hands (the spheres108
"4 Baoding Balls" pattern). Both palms up, little-finger edges `gap` apart, forearm geoms left out
(`two_hands.build(forearms=False)`), four balls on a ring centred on the seam between the palms (two on
each palm), the fingers held in a posture; settle, report each ball's drift, save two views.

Usage:
    python3 stage7_layout.py --out media_full/stage7_layout.png
"""
import argparse
import json

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

from hand_env import ACTUATOR_NAMES
from two_hands import LEFT_BALLS, RIGHT_BALLS, build, left, wrist_qpos

GAP = 0.083                     # m: the palms touch at ~0.082
PALM_Z = 0.022                  # world z of the palm surface under the balls (Stage 2's BALL_PLACE)
# a mild cup: fingers a little curled, thumbs raised as outer walls, little fingers flat at the seam
POSTURE = {"rh_A_WRJ2": 0.0, "rh_A_WRJ1": 0.0,
           "rh_A_THJ5": 0.3, "rh_A_THJ4": 0.7, "rh_A_THJ3": 0.0, "rh_A_THJ2": 0.0, "rh_A_THJ1": 0.3,
           "rh_A_FFJ4": 0.0, "rh_A_FFJ3": 0.25, "rh_A_FFJ0": 0.5,
           "rh_A_MFJ4": 0.0, "rh_A_MFJ3": 0.25, "rh_A_MFJ0": 0.5,
           "rh_A_RFJ4": 0.0, "rh_A_RFJ3": 0.25, "rh_A_RFJ0": 0.5,
           "rh_A_LFJ5": 0.0, "rh_A_LFJ4": 0.0, "rh_A_LFJ3": 0.15, "rh_A_LFJ0": 0.3}


def ring_places(center, radius, phase, n=4):
    """n ball centres evenly round a horizontal ring (world xy) about `center`."""
    a = phase + 2 * np.pi * np.arange(n) / n
    return [np.array(center) + radius * np.array([np.cos(t), np.sin(t), 0.0]) for t in a]


def set_ball(m, d, name, pos):
    j = m.body_jntadr[m.body(name).id]
    qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
    d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = pos, [1, 0, 0, 0]
    d.qvel[va:va + 6] = 0.0


def views(m, d, renderer, look):
    imgs = []
    for az, el, dist in ((180, -88, 0.36), (180, -35, 0.40)):
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        renderer.update_scene(d, camera=cam)
        imgs.append(renderer.render().copy())
    return Image.fromarray(np.concatenate(imgs, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=GAP)
    ap.add_argument("--ring", type=float, default=0.04, help="m, ring radius")
    ap.add_argument("--ball", type=float, default=0.0225, help="m, ball radius")
    ap.add_argument("--along", type=float, default=0.30, help="world x of the ring centre (fingers +x)")
    ap.add_argument("--phase", type=float, default=np.pi / 4)
    ap.add_argument("--posture", default=None, help="json file of actuator targets (right-hand names)")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", default="media_full/stage7_layout.png")
    args = ap.parse_args()

    posture = dict(POSTURE)
    if args.posture:
        posture.update(json.load(open(args.posture)))
    spec, m = build(args.gap, forearms=False)
    d = mj.MjData(m)
    for b in RIGHT_BALLS + LEFT_BALLS:                 # resize as the Stage 5 env does (no mj_setConst)
        g, bid = m.geom(b).id, m.body(b).id
        k = args.ball / m.geom_size[g, 0]
        m.geom_size[g, 0] = m.geom_rbound[g] = args.ball
        m.body_mass[bid] *= k ** 3
        m.body_inertia[bid] *= k ** 5
    for a in ACTUATOR_NAMES:                           # the same targets on both hands = mirrored postures
        v = posture[a]
        d.ctrl[m.actuator(a).id] = d.ctrl[m.actuator(left(a)).id] = v
    wrist_qpos(m, d, posture["rh_A_WRJ1"], posture["rh_A_WRJ2"])
    for _ in range(int(1.0 / m.opt.timestep)):          # the hands reach the posture with no balls yet
        mj.mj_step(m, d)
    seam_y = 0.5 * (d.body("rh_palm").xpos[1] + d.body(left("rh_palm")).xpos[1])
    center = np.array([args.along, seam_y, PALM_Z + args.ball + 0.002])
    places = ring_places(center, args.ring, args.phase)
    # which palm each place is over: y below the seam = the right palm (right palm at y=0)
    names = [b for b in RIGHT_BALLS + LEFT_BALLS]
    order = sorted(range(4), key=lambda i: places[i][1])
    for b, i in zip(names, order):
        set_ball(m, d, b, places[i])
    mj.mj_forward(m, d)
    start = {b: d.body(b).xpos.copy() for b in names}
    for _ in range(int(args.seconds / m.opt.timestep)):
        mj.mj_step(m, d)
    renderer = mj.Renderer(m, height=360, width=480)
    im = views(m, d, renderer, center)
    renderer.close()
    lines = []
    for b in names:
        p = d.body(b).xpos
        drift = np.linalg.norm((p - start[b])[:2]) * 100
        lines.append(f"{b}: drift {drift:4.1f}cm  z {100 * p[2]:5.1f}cm  y {100 * (p[1] - seam_y):+5.1f}cm from seam")
    fell = [b for b in names if d.body(b).xpos[2] < PALM_Z]
    draw = ImageDraw.Draw(im)
    draw.text((8, 6), f"gap {100 * args.gap:.1f}cm  ring r {100 * args.ring:.1f}cm  balls {2000 * args.ball:.0f}mm  "
                      f"after {args.seconds:.0f}s  fell: {fell or 'none'}", fill=(255, 255, 255))
    im.save(args.out)
    print("\n".join(lines))
    print(f"fell off: {fell or 'none'}; saved {args.out}")


if __name__ == "__main__":
    main()
