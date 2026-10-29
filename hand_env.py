"""Shared constants and helpers for the Shadow Hand (E3M5, right) project.

The vendored model (assets/right_hand.xml, mujoco_menagerie `shadow_hand`) lies palm-UP out of the
box: forearm along world +x, fingers pointing +x, palm facing +z, so an object dropped on it lands
in the palm -- the same working orientation the Allegro project used, no re-mounting needed.
"""
import mujoco as mj
import numpy as np

SCENE = "assets/scene_right.xml"

DIGITS = ("ff", "mf", "rf", "lf", "th")
# 24 hand joints, model order (wrist, ff, mf, rf, lf, th). J1/J2 of ff/mf/rf/lf are coupled by a
# fixed tendon and driven by ONE actuator (`rh_A_xxJ0`, ctrlrange 0..pi = J1+J2), like a human
# finger's distal two joints -- measured: commanding 1.5 splits ~0.73 / 0.77.
JOINT_NAMES = (["rh_WRJ2", "rh_WRJ1"]
               + [f"rh_{f}J{j}" for f in ("FF", "MF", "RF") for j in (4, 3, 2, 1)]
               + [f"rh_LFJ{j}" for j in (5, 4, 3, 2, 1)]
               + [f"rh_THJ{j}" for j in (5, 4, 3, 2, 1)])
# 20 actuators, model order.
ACTUATOR_NAMES = (["rh_A_WRJ2", "rh_A_WRJ1"] + [f"rh_A_THJ{j}" for j in (5, 4, 3, 2, 1)]
                  + [f"rh_A_{f}J{j}" for f in ("FF", "MF", "RF") for j in (4, 3, 0)]
                  + [f"rh_A_LFJ{j}" for j in (5, 4, 3, 0)])
N_ACTUATORS = len(ACTUATOR_NAMES)

# Each digit's segment bodies, root to tip.
SEGMENTS = {
    "ff": ["rh_ffknuckle", "rh_ffproximal", "rh_ffmiddle", "rh_ffdistal"],
    "mf": ["rh_mfknuckle", "rh_mfproximal", "rh_mfmiddle", "rh_mfdistal"],
    "rf": ["rh_rfknuckle", "rh_rfproximal", "rh_rfmiddle", "rh_rfdistal"],
    "lf": ["rh_lfmetacarpal", "rh_lfknuckle", "rh_lfproximal", "rh_lfmiddle", "rh_lfdistal"],
    "th": ["rh_thbase", "rh_thproximal", "rh_thhub", "rh_thmiddle", "rh_thdistal"],
}
# Segments whose own joint is a FLEXION joint (curls the digit toward the palm). The others
# (knuckle/metacarpal/thbase/thhub) carry spread or twist joints and are the digit's root.
FLEX_SEGMENTS = {
    "ff": ["rh_ffproximal", "rh_ffmiddle", "rh_ffdistal"],
    "mf": ["rh_mfproximal", "rh_mfmiddle", "rh_mfdistal"],
    "rf": ["rh_rfproximal", "rh_rfmiddle", "rh_rfdistal"],
    "lf": ["rh_lfproximal", "rh_lfmiddle", "rh_lfdistal"],
    "th": ["rh_thproximal", "rh_thmiddle", "rh_thdistal"],
}


def palmar_axis(model, body):
    """Unit vector, in `body`'s own frame, pointing out of the segment's PALMAR (inside) face.

    A segment's flexion joint rotates it about `axis` (given in the body frame), and the segment
    extends along its own local +z, so positive flexion swings it toward `axis x z`. That is the
    palmar direction -- derived from the model rather than hardcoded, because it differs between
    the fingers (local -y) and the thumb's middle segment (local -x). Lesson carried over from the
    Allegro project, where a contact count that ignored WHICH face touched was satisfied by the
    thumb pressing with its back."""
    b = model.body(body)
    assert b.jntnum[0] == 1, f"{body} has {b.jntnum[0]} joints, expected one flexion joint"
    axis = model.jnt_axis[b.jntadr[0]]
    v = np.cross(axis, [0.0, 0.0, 1.0])
    return v / np.linalg.norm(v)


def set_ctrl_by_name(model, data, values):
    for name, v in values.items():
        data.ctrl[model.actuator(name).id] = v


# Stage 2: contact-face classification and the one-ball hold.
ONE_BALL_SCENE = "assets/scene_one_ball.xml"
BALL_RADIUS = 0.0225
# body name -> (digit, segment), segment = the body name minus "rh_" and the digit prefix.
BODY_SEGMENT = {b: (d, b[3 + len(d):]) for d, bodies in SEGMENTS.items() for b in bodies}
# Bodies that are part of the palm surface rather than a digit's own segment.
PALM_BODIES = ("rh_palm", "rh_lfmetacarpal")
TIP_SEGMENTS = ("ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal")


def classify_contacts(model, data, obj):
    """List of (digit, segment, face, normal_force) for every hand contact on body `obj`.

    `digit` is ff/mf/rf/lf/th, or "palm" for the palm and the little finger's metacarpal. Face of a
    flexion segment (proximal/middle/distal), from the contact point's offset from the segment's own
    long axis (local z), in the segment's frame:
      - "tip"    -- distal segments only: past the start of the rounded end cap and within 45 deg
                    of the long axis (the very end of the finger, not its pad);
      - "palmar" -- the offset is within 60 deg of `palmar_axis` (the inside face; the pad);
      - "DORSAL" -- within 60 deg of the opposite direction (the back -- the Allegro thumb's fault);
      - "side"   -- in between.
    Root segments (knuckle, thbase, thhub) carry spread/twist joints, not flexion, and get face
    "root". The force is the contact's normal force (N), so callers can ignore grazing touches."""
    obj_id = model.body(obj).id
    out = []
    force = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if obj_id not in (b1, b2):
            continue
        hand_body = model.body(b2 if b1 == obj_id else b1).name
        mj.mj_contactForce(model, data, i, force)
        fn = float(force[0])
        if hand_body in PALM_BODIES:
            out.append(("palm", hand_body[3:], "palm", fn))
            continue
        if hand_body not in BODY_SEGMENT:
            continue
        digit, seg = BODY_SEGMENT[hand_body]
        if seg not in [s[3 + len(digit):] for s in FLEX_SEGMENTS[digit]]:
            out.append((digit, seg, "root", fn))
            continue
        bid = model.body(hand_body).id
        p = data.xmat[bid].reshape(3, 3).T @ (c.pos - data.xpos[bid])
        radial = np.array([p[0], p[1], 0.0])
        if hand_body[3:] in TIP_SEGMENTS:
            top, cap_r = _segment_top(model, bid)
            cap = p - np.array([0.0, 0.0, top - cap_r])
            if cap[2] > 0 and cap[2] > np.linalg.norm(radial):
                out.append((digit, seg, "tip", fn))
                continue
        cos = radial @ palmar_axis(model, hand_body) / max(np.linalg.norm(radial), 1e-9)
        face = "palmar" if cos > 0.5 else "DORSAL" if cos < -0.5 else "side"
        out.append((digit, seg, face, fn))
    return out


def _segment_top(model, bid):
    """(z of the segment's far end, end-cap radius) in the body frame, from its collision geoms'
    bounding boxes."""
    top, r = 0.0, 0.0
    for g in range(model.body_geomadr[bid], model.body_geomadr[bid] + model.body_geomnum[bid]):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        center, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        z = model.geom_pos[g][2] + center[2] + half[2]
        if z > top:
            top, r = z, float(max(half[0], half[1]))
    return top, r


# Where the ball goes for `grasp_ball(..., pin=True)`: mid-palm (palm-frame z 0.046 of the palm's
# ~0.095 length), resting on its surface (top of the palm collision boxes at world z 0.022). On the
# open hand a ball here does not stay -- it rolls off over the fingers -- so a pinned grasp holds it
# in place while the hand closes, like a person setting it in the palm.
BALL_PLACE = np.array([0.293, 0.0, 0.022 + 0.0225])
# Gravity directions for the hold test, world frame. With a fixed base, pointing gravity elsewhere
# is the same physics as turning the hand over: "+z" = palm DOWN, "+x" = fingers pointing down,
# "-y"/"+y" = thumb side down / little-finger side down.
GRAVITY_TESTS = {"palm up": (0, 0, -1), "palm down": (0, 0, 1), "fingers down": (1, 0, 0),
                 "fingers up": (-1, 0, 0), "thumb down": (0, -1, 0), "little down": (0, 1, 0)}


def grasp_ball(model, data, target, place=BALL_PLACE, pin=True, close_time=0.6, hold_time=0.6):
    """Reset with the ball at `place`, then ramp every actuator linearly from rest to `target`
    (len N_ACTUATORS, clipped to ctrlrange) over `close_time` s and hold `hold_time` s, palm up under
    normal gravity. pin=True: the ball is held at `place` (position fixed, velocity zeroed every
    step) while the hand closes, then released. pin=False: it is dropped from `place` onto the open
    hand and left 0.5 s to roll before the hand closes on wherever it got to."""
    mj.mj_resetData(model, data)
    model.opt.gravity[:] = (0, 0, -9.81)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    rest, target = np.clip(0.0, lo, hi), np.clip(target, lo, hi)
    j = model.body("ball").jntadr[0]
    q, v = model.jnt_qposadr[j], model.jnt_dofadr[j]
    data.qpos[q:q + 3] = place
    data.ctrl[:] = rest
    dt = model.opt.timestep
    if not pin:
        for _ in range(int(0.5 / dt)):
            mj.mj_step(model, data)
    n = int(close_time / dt)
    for k in range(n):
        data.ctrl[:] = rest + (target - rest) * (k + 1) / n
        if pin:
            data.qpos[q:q + 3], data.qpos[q + 3:q + 7], data.qvel[v:v + 6] = place, (1, 0, 0, 0), 0.0
        mj.mj_step(model, data)
    for _ in range(int(hold_time / dt)):
        mj.mj_step(model, data)


def ball_in_palm(model, data):
    """Ball center in the palm body's frame."""
    R = data.body("rh_palm").xmat.reshape(3, 3)
    return R.T @ (data.body("ball").xpos - data.body("rh_palm").xpos)


def gravity_hold_test(model, data, seconds=2.0, max_drift=0.01):
    """From the current (grasped) state, for each GRAVITY_TESTS direction: restore the state, point
    gravity that way and run `seconds`. Returns {name: drift_m} where drift is how far the ball
    moved in the palm frame (np.inf if it left the hand), and whether every drift <= max_drift.
    Restores the state and normal gravity afterward."""
    saved = (data.qpos.copy(), data.qvel.copy(), data.act.copy(), data.ctrl.copy(), data.time)
    p0 = ball_in_palm(model, data)
    out = {}
    for name, g in GRAVITY_TESTS.items():
        data.qpos[:], data.qvel[:], data.act[:], data.ctrl[:], data.time = saved
        mj.mj_forward(model, data)
        model.opt.gravity[:] = 9.81 * np.asarray(g, float)
        for _ in range(int(seconds / model.opt.timestep)):
            mj.mj_step(model, data)
        drift = float(np.linalg.norm(ball_in_palm(model, data) - p0))
        out[name] = drift if drift < 0.05 else np.inf
    data.qpos[:], data.qvel[:], data.act[:], data.ctrl[:], data.time = saved
    model.opt.gravity[:] = (0, 0, -9.81)
    mj.mj_forward(model, data)
    return out, all(d <= max_drift for d in out.values())


# The Stage 2 one-ball hold, from `search_stage2_grasp.py --drop --search-place --seed 0
# --std-floor 0.06 --perturbed 7 --iters 12` (pad terms averaged over 7 fixed perturbations): the
# ball dropped near mid-palm rolls toward the knuckles, and mf, rf and lf wrap it with the palmar
# faces of their proximal and distal segments; ff curled alongside, thumb across the palm, not on
# the ball. Holds under gravity from all six directions. Wrist at 0. Targets beyond the contact angle
# are what squeeze (position actuators, kp 0.5-1.5). Image: media/stage2_grasp_d_robust_no_thumb.png.
# Searches with a stronger thumb bonus put the thumb on only as a graze (0.3 N); placing the ball
# mid-palm (pin=True) instead held as well but with one pad. Left there: one 45mm ball is a poor
# stand-in for baoding, where two touching balls fill the palm's whole length.
ONE_BALL_GRASP = {
    "rh_A_THJ5": -0.1268, "rh_A_THJ4": 0.5561, "rh_A_THJ3": -0.0803, "rh_A_THJ2": 0.2447,
    "rh_A_THJ1": 1.0233,
    "rh_A_FFJ4": 0.0482, "rh_A_FFJ3": 0.3119, "rh_A_FFJ0": 2.3079,
    "rh_A_MFJ4": -0.2516, "rh_A_MFJ3": 0.9553, "rh_A_MFJ0": 2.556,
    "rh_A_RFJ4": -0.2172, "rh_A_RFJ3": 0.3167, "rh_A_RFJ0": 2.2604,
    "rh_A_LFJ5": 0.5286, "rh_A_LFJ4": -0.3351, "rh_A_LFJ3": 0.3949, "rh_A_LFJ0": 2.8199,
}
ONE_BALL_PLACE = np.array([0.3076, 0.0042, 0.06])   # dropped from here (pin=False)
ONE_BALL_PIN = False


def contact_summary(contacts, min_force=0.05, pad_faces=("palmar", "tip")):
    """(digits touching with a `pad_faces` face of their middle/distal segment, count of DORSAL
    contacts, whether the palm touches), ignoring touches under `min_force` N."""
    pads, n_dorsal, palm = set(), 0, False
    for digit, seg, face, fn in contacts:
        if fn < min_force:
            continue
        if digit == "palm":
            palm = True
        elif face == "DORSAL":
            n_dorsal += 1
        elif face in pad_faces and seg in ("middle", "distal"):
            pads.add(digit)
    return pads, n_dorsal, palm
