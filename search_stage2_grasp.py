"""Stage 2 grasp search: cross-entropy method over the 18 non-wrist actuator targets (wrist held at
0) for a one-ball hold. Each candidate: place the ball at mid-palm, close to the candidate targets
while it is held there, release it (`hand_env.grasp_ball`; or with --drop, drop it and close on
wherever it rolled), then `gravity_hold_test` -- the hold must survive gravity from all
six directions, i.e. the hand turned palm-down, fingers-down, on either side.

Score = 10 per gravity direction held (drift <= 1cm)
      + 6 per digit pressing the ball (>= 0.2 N) with its PAD (palmar face of middle or distal)
      + `--thumb-bonus` (default 3) more if the thumb is one of them (an opposed thumb, not four
        fingers pinning the ball to the palm -- with 3, the perturbation-averaged search settled on
        exactly that, mf/rf/lf pads and the thumb off the ball; the frozen grasp used 12)
      + 8 if the palm carries the ball (>= 0.2 N) -- the ball sits IN the hand, not up at the
        knuckles pinched by one or two fingers, as every grasp found before placement was added
      - 5 per DORSAL contact, - 2 per digit poking with its fingertip END, - 1 per SIDE contact
      - 100 * mean drift over the held directions (prefer a firm hold to a barely-held one).
Holding alone is easy -- most random samples in the first round already held in all six
directions, by caging the ball with fingertip ends and the sides of proximal segments -- so the
contact-face terms are what select a real wrap grasp. "tip" is scored apart from "palmar" because
the first search's best grasp counted fingertip ends as pads and poked the ball with them.

Usage:
    python3 search_stage2_grasp.py --out data/stage2_grasp_search.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import mujoco as mj
import numpy as np

from hand_env import (ACTUATOR_NAMES, BALL_PLACE, ONE_BALL_SCENE, classify_contacts, contact_summary,
                      grasp_ball, gravity_hold_test)

FREE = np.arange(2, len(ACTUATOR_NAMES))   # everything but the two wrist actuators
DROP_POINT = np.array([0.30, 0.0, 0.06])   # --drop: the ball falls from here onto the open hand
PAD_FORCE = 0.2   # N; a lighter touch is a graze (the thumb-bonus search's thumb pressed 0.3 N)
_model = _data = _perturbations = _thumb_bonus = _pin = _base = None


def _init(n_perturbed, thumb_bonus, pin=True):
    """Per worker: the model, and ONE fixed set of perturbations (same seed in every worker) so all
    candidates are compared on the same draws -- the ranges of eval_stage2_hold.py."""
    global _model, _data, _perturbations, _thumb_bonus, _pin, _base
    _thumb_bonus, _pin, _base = thumb_bonus, pin, (BALL_PLACE if pin else DROP_POINT)
    from eval_stage2_hold import PLACE_NOISE, FRICTION_RANGE, MASS_RANGE, TARGET_NOISE
    _model = mj.MjModel.from_xml_path(ONE_BALL_SCENE)
    _data = mj.MjData(_model)
    rng = np.random.default_rng(12345)
    _perturbations = [dict(dt=rng.uniform(-TARGET_NOISE, TARGET_NOISE, len(ACTUATOR_NAMES)),
                           dd=np.append(rng.uniform(-PLACE_NOISE, PLACE_NOISE, 2), 0.0),
                           mass=rng.uniform(*MASS_RANGE), friction=rng.uniform(*FRICTION_RANGE))
                      for _ in range(n_perturbed)]


def _face_terms(target, place):
    grasp_ball(_model, _data, target, place=place, pin=_pin)
    contacts = classify_contacts(_model, _data, "ball")
    pads, n_dorsal, palm = contact_summary(contacts, min_force=PAD_FORCE, pad_faces=("palmar",))
    firm = [c for c in contacts if c[3] >= PAD_FORCE]
    tips = {c[0] for c in firm if c[2] == "tip"} - pads
    n_side = sum(c[2] == "side" for c in firm)
    return pads, tips, n_side, n_dorsal, palm


def evaluate(x):
    x = np.round(x, 4)   # score exactly what gets frozen into hand_env
    target = np.zeros(len(ACTUATOR_NAMES))
    target[FREE] = x[:len(FREE)]
    place = _base + np.array([x[len(FREE)], x[len(FREE) + 1], 0.0]) if len(x) > len(FREE) else _base
    pads, tips, n_side, n_dorsal, palm = terms = _face_terms(target, place)
    drifts, _ = gravity_hold_test(_model, _data)   # on the nominal grasp
    held = [d for d in drifts.values() if d <= 0.01]
    # Face terms: averaged over the nominal grasp and every fixed perturbation.
    ball, geom = _model.body("ball").id, _model.geom("ball").id
    m0, f0 = _model.body_mass[ball], _model.geom_friction[geom, 0]
    terms = [terms]
    for p in _perturbations:
        _model.body_mass[ball], _model.geom_friction[geom, 0] = p["mass"], p["friction"]
        terms.append(_face_terms(target + p["dt"], place + p["dd"]))
    _model.body_mass[ball], _model.geom_friction[geom, 0] = m0, f0
    face = np.mean([6 * len(pads) + _thumb_bonus * ("th" in pads) + 8 * palm - 5 * n_dorsal
                    - 2 * len(tips) - n_side for pads, tips, n_side, n_dorsal, palm in terms])
    score = 10 * len(held) + face - 100 * (np.mean(held) if held else 0.0)
    return dict(score=float(score), x=[float(v) for v in x], n_held=len(held), pads=sorted(pads),
                mean_pads=float(np.mean([len(t[0]) for t in terms])),
                tips=sorted(tips), n_side=n_side, n_dorsal=n_dorsal, palm=palm, drifts=drifts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--pop", type=int, default=320)
    ap.add_argument("--elite", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--std-floor", type=float, default=0.02)
    ap.add_argument("--thumb-bonus", type=float, default=3.0)
    ap.add_argument("--drop", action="store_true",
                    help="drop the ball from 6cm and let it roll before closing (pin=False), instead of "
                         "placing it at mid-palm and holding it there while the hand closes")
    ap.add_argument("--perturbed", type=int, default=0,
                    help="average the contact-face terms over the nominal grasp + this many fixed perturbed ones")
    ap.add_argument("--search-place", action="store_true",
                    help="also search where the ball is placed/dropped, +-15mm in x and y")
    ap.add_argument("--out", type=str, default="data/stage2_grasp_search.json")
    args = ap.parse_args()

    model = mj.MjModel.from_xml_path(ONE_BALL_SCENE)
    lo, hi = model.actuator_ctrlrange[FREE, 0], model.actuator_ctrlrange[FREE, 1]
    if args.search_place:
        lo, hi = np.append(lo, [-0.015, -0.015]), np.append(hi, [0.015, 0.015])
    rng = np.random.default_rng(args.seed)
    mean, std = (lo + hi) / 2, (hi - lo) / 3
    best = None
    _init(args.perturbed, args.thumb_bonus, not args.drop)   # fail here, not in a Pool that respawns crashing workers
    with Pool(16, initializer=_init, initargs=(args.perturbed, args.thumb_bonus, not args.drop)) as pool:
        for it in range(args.iters):
            xs = np.clip(rng.normal(mean, std, size=(args.pop, len(lo))), lo, hi)
            if best is not None:
                xs[0] = best["x"]
            results = pool.map(evaluate, list(xs))
            results.sort(key=lambda r: -r["score"])
            elite = np.array([r["x"] for r in results[:args.elite]])
            mean, std = elite.mean(0), np.maximum(elite.std(0), args.std_floor * (hi - lo))
            best = results[0] if best is None or results[0]["score"] > best["score"] else best
            print(f"iter {it}: best {best['score']:.1f} held {best['n_held']}/6 pads {best['pads']} "
                  f"(mean over perturbations {best['mean_pads']:.2f}) palm {best['palm']} "
                  f"tips {best['tips']} side {best['n_side']} dorsal {best['n_dorsal']} | elite mean score "
                  f"{np.mean([r['score'] for r in results[:args.elite]]):.1f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best["targets"] = {ACTUATOR_NAMES[i]: v for i, v in zip(FREE, best["x"])}
    if args.search_place:
        best["place"] = [float(v) for v in _base + np.append(best["x"][len(FREE):], 0.0)]
    Path(args.out).write_text(json.dumps(best, indent=1, default=str))
    print(json.dumps(best["targets"]))
    print({k: round(v, 4) for k, v in best["drifts"].items()})


if __name__ == "__main__":
    main()
