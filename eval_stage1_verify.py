"""Stage 1 verification of the vendored Shadow Hand model, under real closed-loop physics (mj_step
with gravity, not forward kinematics on a set qpos):

1. every actuator tracks commanded targets (each one swept to 25% / 75% of its range, others held);
2. `hand_env.palmar_axis` matches how each flexion segment actually moves when flexed;
3. the coupled middle/distal tendon actuators split their command between J2 and J1;
4. the scene's free object, dropped on the open hand, comes to rest in the palm.

Usage:
    python3 eval_stage1_verify.py
"""
import mujoco as mj
import numpy as np

from hand_env import ACTUATOR_NAMES, FLEX_SEGMENTS, JOINT_NAMES, SCENE, palmar_axis


def actuated_value(model, data, act):
    """Current value of what actuator `act` drives: its joint, or its tendon (J1+J2)."""
    trn = model.actuator_trnid[act, 0]
    if model.actuator_trntype[act] == mj.mjtTrn.mjTRN_TENDON:
        return data.ten_length[trn]
    return data.qpos[model.jnt_qposadr[trn]]


def settle(model, data, seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        mj.mj_step(model, data)


def main():
    model = mj.MjModel.from_xml_path(SCENE)
    data = mj.MjData(model)
    assert [model.actuator(i).name for i in range(model.nu)] == ACTUATOR_NAMES
    assert all(model.joint(n).id >= 0 for n in JOINT_NAMES)
    print(f"{len(JOINT_NAMES)} hand joints, {model.nu} actuators ({model.ntendon} coupled-tendon), "
          f"timestep {model.opt.timestep}s")

    # 1. tracking
    worst = (0.0, "")
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    rest = np.clip(0.0, lo, hi)
    for i, name in enumerate(ACTUATOR_NAMES):
        for frac in (0.25, 0.75):
            mj.mj_resetData(model, data)
            data.ctrl[:] = rest
            target = lo[i] + frac * (hi[i] - lo[i])
            data.ctrl[i] = target
            settle(model, data, 1.0)
            err = abs(actuated_value(model, data, i) - target)
            if err > worst[0]:
                worst = (err, f"{name} at {frac:.0%} of range")
    print(f"[1] tracking: worst steady-state error {worst[0]:.3f} rad ({worst[1]}) over 40 sweeps")

    # 2. palmar axis vs actual flexion motion
    bad = []
    for digit, bodies in FLEX_SEGMENTS.items():
        for body in bodies:
            mj.mj_resetData(model, data)
            mj.mj_forward(model, data)
            b = model.body(body)
            g = b.geomadr[0] + b.geomnum[0] - 1
            p0, R = data.geom_xpos[g].copy(), data.xmat[b.id].reshape(3, 3).copy()
            data.qpos[model.jnt_qposadr[b.jntadr[0]]] += 0.1
            mj.mj_forward(model, data)
            moved = R.T @ (data.geom_xpos[g] - p0)
            cos = moved @ palmar_axis(model, body) / np.linalg.norm(moved)
            if cos < 0.9:
                bad.append((body, round(float(cos), 2)))
    n = sum(len(v) for v in FLEX_SEGMENTS.values())
    print(f"[2] palmar axis: {n - len(bad)}/{n} flexion segments move along their derived palmar "
          f"axis when flexed (cos > 0.9)" + (f" -- MISMATCH: {bad}" if bad else ""))

    # 3. coupling
    mj.mj_resetData(model, data)
    data.ctrl[:] = rest
    data.ctrl[model.actuator("rh_A_FFJ0").id] = 1.5
    settle(model, data, 1.0)
    j2, j1 = (data.qpos[model.joint(j).qposadr[0]] for j in ("rh_FFJ2", "rh_FFJ1"))
    print(f"[3] coupling: rh_A_FFJ0 = 1.5 -> FFJ2 {j2:.3f}, FFJ1 {j1:.3f} (sum {j1 + j2:.3f})")

    # 4. object drops into the palm
    mj.mj_resetData(model, data)
    data.ctrl[:] = rest
    settle(model, data, 2.0)
    p = data.body("object").xpos
    palm = data.body("rh_palm").xpos
    print(f"[4] object after 2s on the open hand: {np.round(p, 3)} (palm origin {np.round(palm, 3)}), "
          f"speed {np.linalg.norm(data.qvel[-6:-3]):.4f} m/s")


if __name__ == "__main__":
    main()
