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
