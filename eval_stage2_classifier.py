"""Stage 2 check of `hand_env.classify_contacts`: for every flexion segment of every digit, place the
ball just touching the segment from a KNOWN direction -- its palmar face, its back, its side, and
(distal segments) straight off its end -- and confirm the classifier labels that contact with the
right face. Uses mj_forward on a placed ball (a geometric test of the labelling, not a physics run).

Usage:
    python3 eval_stage2_classifier.py
"""
import mujoco as mj
import numpy as np

from hand_env import (BALL_RADIUS, FLEX_SEGMENTS, ONE_BALL_SCENE, TIP_SEGMENTS, _segment_top,
                      classify_contacts, palmar_axis)


def probe(model, data, body, direction_local, z_local, depth=0.001):
    """Put the ball against `body`, approaching along `direction_local` (body frame) at height
    `z_local` on the segment's axis; return the labels of the ball's contacts with that body."""
    bid = model.body(body).id
    mj.mj_forward(model, data)
    R, x = data.xmat[bid].reshape(3, 3), data.xpos[bid]
    d = direction_local / np.linalg.norm(direction_local)
    q = model.jnt_qposadr[model.body("ball").jntadr[0]]
    # Walk the ball in from outside along d until it first touches the body, then press `depth`.
    for s in np.arange(0.06, 0.0, -0.0002):
        data.qpos[q:q + 3] = x + R @ (np.array([0.0, 0.0, z_local]) + (s + BALL_RADIUS) * d)
        data.qpos[q + 3:q + 7] = [1, 0, 0, 0]
        mj.mj_forward(model, data)
        hits = [c for c in classify_contacts(model, data, "ball") if f"rh_{c[0]}{c[1]}" == body]
        if hits:
            data.qpos[q:q + 3] -= R @ (depth * d)
            mj.mj_forward(model, data)
            return {c[2] for c in classify_contacts(model, data, "ball") if f"rh_{c[0]}{c[1]}" == body}
    return set()


def main():
    model = mj.MjModel.from_xml_path(ONE_BALL_SCENE)
    data = mj.MjData(model)
    # Open hand with the fingers spread apart and the thumb swung out, so a probed segment is
    # reachable without the ball also hitting a neighbour from the chosen direction.
    data.qpos[model.joint("rh_FFJ4").qposadr[0]] = -0.3
    data.qpos[model.joint("rh_LFJ4").qposadr[0]] = -0.3
    data.qpos[model.joint("rh_THJ5").qposadr[0]] = -1.0
    n_ok, n, bad = 0, 0, []
    for digit, bodies in FLEX_SEGMENTS.items():
        for body in bodies:
            a = palmar_axis(model, body)
            side = np.cross([0.0, 0.0, 1.0], a)
            top, cap_r = _segment_top(model, model.body(body).id)
            mid = 0.5 * top if body[3:] not in TIP_SEGMENTS else top - 2.5 * cap_r
            cases = [("palmar", a, mid), ("DORSAL", -a, mid), ("side", side, mid), ("side", -side, mid)]
            if body[3:] in TIP_SEGMENTS:
                cases.append(("tip", np.array([0.0, 0.0, 1.0]), top - cap_r))
            for want, d, z in cases:
                got = probe(model, data, body, d, z)
                n += 1
                if got == {want}:
                    n_ok += 1
                else:
                    bad.append((body, want, sorted(got)))
    print(f"classifier: {n_ok}/{n} probe contacts labelled correctly")
    for b in bad:
        print("  WRONG:", b)


if __name__ == "__main__":
    main()
