"""Stage 6: three-ball baoding -- scene helpers (the Stage 3-5 helpers in `hand_env` assume two balls).

Three touching balls sit in a triangle; `places` puts their centres on a circle round `center` (world
xyz) in the palm-up hand's horizontal plane, the first at `angle` from the fingers' direction (+x).
"""
import mujoco as mj
import numpy as np

THREE_BALL_SCENE = "assets/scene_three_balls.xml"
BALLS3 = ("ball1", "ball2", "ball3")


def places(center, angle, radius):
    """Centres of three touching balls of `radius` round `center`: circumradius 2r/sqrt(3)."""
    rc = 2 * radius / np.sqrt(3)
    return [np.array(center) + rc * np.array([np.cos(angle + k * 2 * np.pi / 3), np.sin(angle + k * 2 * np.pi / 3), 0.0])
            for k in range(3)]


def set_radius(model, radius):
    """All three balls to `radius` (m), mass and inertia scaled with volume (same material)."""
    for b in BALLS3:
        g, bid = model.geom(b).id, model.body(b).id
        k = radius / model.geom_size[g, 0]
        model.geom_size[g, 0] = model.geom_rbound[g] = radius
        model.body_mass[bid] *= k ** 3
        model.body_inertia[bid] *= k ** 5


def grasp(model, data, target, ball_places, close_time=0.6, settle_time=0.3):
    """Balls held at `ball_places` while the hand ramps from rest to `target`, then released."""
    mj.mj_resetData(model, data)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    rest, target = np.clip(0.0, lo, hi), np.clip(target, lo, hi)
    adr = [(model.jnt_qposadr[model.body(b).jntadr[0]], model.jnt_dofadr[model.body(b).jntadr[0]]) for b in BALLS3]
    n = int(close_time / model.opt.timestep)
    for k in range(n):
        data.ctrl[:] = rest + (target - rest) * (k + 1) / n
        for (q, v), p in zip(adr, ball_places):
            data.qpos[q:q + 3], data.qpos[q + 3:q + 7], data.qvel[v:v + 6] = p, (1, 0, 0, 0), 0.0
        mj.mj_step(model, data)
    for _ in range(int(settle_time / model.opt.timestep)):
        mj.mj_step(model, data)


def in_palm(model, data):
    """Each ball's centre in the palm body's frame."""
    R = data.body("rh_palm").xmat.reshape(3, 3)
    return [R.T @ (data.body(b).xpos - data.body("rh_palm").xpos) for b in BALLS3]
