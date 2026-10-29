"""Stage 2 evaluation of the one-ball hold (`hand_env.ONE_BALL_GRASP`):

1. nominal: which face of which segment touches the ball, the hold under gravity from all six
   directions (`gravity_hold_test`), and the peak actuator effort as a fraction of its force limit;
2. perturbed: `--trials` repeats with every joint target, the placement, the ball's mass and
   friction randomized (ranges below) -- a grasp that only works at its exact tuned numbers is a
   knife edge, the Allegro toss-catch lesson.

Usage:
    python3 eval_stage2_hold.py [--trials 100]
"""
import argparse

import mujoco as mj
import numpy as np

from hand_env import (ACTUATOR_NAMES, GRAVITY_TESTS, ONE_BALL_PIN, ONE_BALL_PLACE, ONE_BALL_GRASP,
                      ONE_BALL_SCENE, classify_contacts, contact_summary, grasp_ball,
                      gravity_hold_test)

TARGET_NOISE = 0.05            # rad, uniform +-, on every actuator target
PLACE_NOISE = 0.003             # m, uniform +-, x and y
MASS_RANGE = (0.04, 0.15)      # kg (nominal 0.08)
FRICTION_RANGE = (0.5, 1.0)    # sliding friction (nominal 0.8)


def target_vector():
    t = np.zeros(len(ACTUATOR_NAMES))
    for name, v in ONE_BALL_GRASP.items():
        t[ACTUATOR_NAMES.index(name)] = v
    return t


def peak_effort(model, data, seconds=1.0):
    """Max over actuators and `seconds` of the hold of |actuator force| / force limit."""
    lim = np.abs(model.actuator_forcerange[:, 1])
    peak = 0.0
    for _ in range(int(seconds / model.opt.timestep)):
        mj.mj_step(model, data)
        peak = max(peak, float(np.max(np.abs(data.actuator_force) / lim)))
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(ONE_BALL_SCENE)
    data = mj.MjData(model)
    ball_geom = model.geom("ball").id
    nominal_mass, nominal_friction = model.body_mass[model.body("ball").id], model.geom_friction[ball_geom, 0]
    target = target_vector()

    grasp_ball(model, data, target, place=ONE_BALL_PLACE, pin=ONE_BALL_PIN)
    contacts = classify_contacts(model, data, "ball")
    print("[1] nominal contacts on the ball (digit, segment, face, normal force N):")
    for c in sorted(contacts, key=lambda c: -c[3]):
        print(f"      {c[0]:4s} {c[1]:8s} {c[2]:6s} {c[3]:.2f}")
    pads, n_dorsal, palm = contact_summary(contacts, pad_faces=("palmar",))
    drifts, ok = gravity_hold_test(model, data)
    print(f"    pads: {sorted(pads)}, dorsal contacts: {n_dorsal}, palm touching: {palm}")
    print("    hold, ball drift after 2s: " + ", ".join(f"{k} {v * 1000:.1f}mm" for k, v in drifts.items())
          + f" -> {'HELD in all 6' if ok else 'NOT held in all 6'}")
    worst = 0.0
    for g in GRAVITY_TESTS.values():
        grasp_ball(model, data, target, place=ONE_BALL_PLACE, pin=ONE_BALL_PIN)
        model.opt.gravity[:] = 9.81 * np.asarray(g, float)
        worst = max(worst, peak_effort(model, data))
    model.opt.gravity[:] = (0, 0, -9.81)
    print(f"    peak actuator effort over the six holds: {worst:.0%} of its force limit")

    rng = np.random.default_rng(args.seed)
    n_held, pad_counts, n_any_dorsal = 0, [], 0
    for _ in range(args.trials):
        model.body_mass[model.body("ball").id] = rng.uniform(*MASS_RANGE)
        model.geom_friction[ball_geom, 0] = rng.uniform(*FRICTION_RANGE)
        place = ONE_BALL_PLACE + np.append(rng.uniform(-PLACE_NOISE, PLACE_NOISE, 2), 0.0)
        grasp_ball(model, data, target + rng.uniform(-TARGET_NOISE, TARGET_NOISE, len(target)), place=place, pin=ONE_BALL_PIN)
        pads, n_dorsal, _ = contact_summary(classify_contacts(model, data, "ball"), pad_faces=("palmar",))
        _, ok = gravity_hold_test(model, data)
        n_held += ok
        pad_counts.append(len(pads))
        n_any_dorsal += n_dorsal > 0
    model.body_mass[model.body("ball").id], model.geom_friction[ball_geom, 0] = nominal_mass, nominal_friction
    counts = np.bincount(pad_counts, minlength=6)
    print(f"[2] perturbed ({args.trials} trials: targets +-{TARGET_NOISE} rad, placement +-{PLACE_NOISE * 1000:.0f}mm, "
          f"mass {MASS_RANGE[0] * 1000:.0f}-{MASS_RANGE[1] * 1000:.0f}g, friction {FRICTION_RANGE[0]}-{FRICTION_RANGE[1]}):")
    print(f"    held in all 6 directions: {n_held}/{args.trials}; any dorsal contact: {n_any_dorsal}/{args.trials}")
    print("    digits on the ball with their pad: " + ", ".join(f"{k}: {counts[k]}" for k in range(6) if counts[k]))


if __name__ == "__main__":
    main()
