"""Stage 7: two Shadow Hands, palms up side by side (little fingers facing, as in the two-hand
four-ball videos), with two balls on each palm.

The scene is built with MjSpec: the Stage 5 two-ball scene (right hand + ball1, ball2) plus the
vendored left hand attached with the prefix `L_` (both hand files name their meshes after the same
.obj files, so the two cannot simply be included together), and ball3, ball4 for the left palm.

The menagerie left hand is the right hand mirrored in its palm frame's x (the sideways axis: ff at
+3.3cm on the right palm, -3.3cm on the left), with joint axes flipped where needed so the SAME joint
angles give the mirrored pose. The left hand is placed `gap` metres along the right palm's -x (its
little-finger side), so the scene is mirror-symmetric about the plane halfway between the palms.

Usage:
    python3 two_hands.py            # build, check the mirror symmetry, save a still
"""
from pathlib import Path

import mujoco as mj
import numpy as np

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
GAP = 0.13                      # m between the two palm frames (B: hands apart; A will close it)
BALL_R = 0.025                  # m, Stage 5's final 50mm balls
RIGHT_BALLS, LEFT_BALLS = ("ball1", "ball2"), ("ball3", "ball4")
L = "L_"                        # prefix of everything attached from left_hand.xml


def left(name):
    """The left hand's name for a right-hand joint/actuator/body name (rh_X -> L_lh_X)."""
    assert name.startswith("rh_"), name
    return L + "lh_" + name[3:]


def palm_x_axis():
    """The right palm's x axis (world) and position at its default pose -- fixes the mirror plane."""
    m = mj.MjModel.from_xml_path(str(ASSETS / "scene_two_balls.xml"))
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    return d.body("rh_palm").xmat.reshape(3, 3)[:, 0].copy(), d.body("rh_palm").xpos.copy()


def build(gap=GAP):
    spec = mj.MjSpec.from_file(str(ASSETS / "scene_two_balls.xml"))
    spec.modelname = "two shadow hands, four balls"
    # the balls are compiled as in the Stage 5 scene (45mm, 80g) and resized at run time without
    # mj_setConst, exactly as the Stage 5 env does: the solver's compile-time constants (invweight0)
    # then match the ones the policy was trained with
    ax, _ = palm_x_axis()
    lspec = mj.MjSpec.from_file(str(ASSETS / "left_hand.xml"))
    frame = spec.worldbody.add_frame(pos=-gap * ax)        # translation only: lh_forearm has the same quat
    frame.attach_body(lspec.body("lh_forearm"), L, "")
    mat = {"ball3": "ball1", "ball4": "ball2"}
    for b in LEFT_BALLS:
        body = spec.worldbody.add_body(name=b, pos=[0.3, -gap, 0.05])
        body.add_freejoint()
        body.add_geom(name=b, type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.0225, 0, 0], mass=0.08,
                      material=mat[b], condim=6, priority=1, friction=[0.8, 0.01, 0.001])
    spec.stat.extent = 0.45
    return spec, spec.compile()


def mirror_check(model):
    """Same joint angles on both hands -> are the fingertips mirror images across the mid-plane?"""
    d = mj.MjData(model)
    rng = np.random.default_rng(0)
    worst = 0.0
    ax, _ = palm_x_axis()                    # the mirror plane's normal: fixed, not the (wrist-turned) palm's
    for _ in range(5):
        for j in range(model.njnt):
            name = model.joint(j).name
            if name.startswith("rh_"):
                lo, hi = model.jnt_range[j]
                q = rng.uniform(lo, hi)
                d.qpos[model.jnt_qposadr[j]] = q
                d.qpos[model.jnt_qposadr[model.joint(left(name)).id]] = q
        mj.mj_forward(model, d)
        mid = 0.5 * (d.body("rh_forearm").xpos + d.body(left("rh_forearm")).xpos)
        for b in ("rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal", "rh_thdistal", "rh_palm"):
            p = d.body(b).xpos
            mirrored = p - 2 * ((p - mid) @ ax) * ax
            worst = max(worst, float(np.linalg.norm(mirrored - d.body(left(b)).xpos)))
    return worst


if __name__ == "__main__":
    spec, m = build()
    print(f"nq {m.nq}  nu {m.nu}  nbody {m.nbody}  ngeom {m.ngeom}")
    print(f"mirror check, worst fingertip/palm mismatch over 5 random poses: {1000 * mirror_check(m):.3f} mm")
