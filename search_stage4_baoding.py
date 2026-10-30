"""Stage 4 search: tune the `hand_env.BaodingWave` press wave (frequency; each digit's phase and two
signed press amplitudes; a small wrist circle) by the cross-entropy method, starting every run from
the Stage 3 bowl.

Score per run (radians-scaled), averaged over the nominal run and `--perturbed` fixed perturbations
(ball mass 40-150g, friction 0.5-1.0 -- the Stage 2 ranges):
      + min over the two balls of s * (that ball's own cumulative angle round the bowl center), with
        s = +-1 the better direction: BOTH balls must travel round the same way -- one ball circling
        while the other sits still is not baoding. Each ball's angle is its WORST over the last
        `WINDOW` s, at least one full wave cycle (freq >= 0.4 Hz): the first search scored the angle
        at the final instant and found a wave that just rocks the jammed pair +-0.4 rad each cycle
        and ends the run at the top of a rock -- no circulation at all
      - 3 if a ball is dropped (the run stops there, keeping the progress made so far)
      + 0.15 per digit that touched a ball with >= 0.2 N during the run -- the user: "not only little
        finger and thumb, all fingers should involved as needed"

Usage:
    python3 search_stage4_baoding.py --out data/stage4_baoding_search.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import mujoco as mj
import numpy as np

from eval_stage2_hold import FRICTION_RANGE, MASS_RANGE
from hand_env import (ACTUATOR_NAMES, BALLS, TWO_BALL_SCENE, BaodingCircle, BaodingWave, bowl_target,
                      run_baoding)

SECONDS = 10.0
WINDOW = 2.5
# Search ranges per BaodingWave parameter.
LO = np.concatenate([[0.4], np.zeros(5), [-0.9, -0.9, -0.9, -0.9, -1.0], [-1.2, -1.2, -1.2, -1.2, -0.7],
                     [0.0, 0.0, 0.0]])
HI = np.concatenate([[1.5], np.full(5, 2 * np.pi), [0.9, 0.9, 0.9, 0.9, 1.0], [1.2, 1.2, 1.2, 1.2, 0.7],
                     [0.4, 0.4, 2 * np.pi]])
# BaodingCircle (`--controller circle`): freq, phase[5], amp1[5], amp2[5], lag[5], amp3[4] (J4), wrist.
CIRCLE_LO = np.concatenate([[0.4], np.zeros(5), [-0.6] * 4 + [-0.6], [-0.8] * 4 + [-0.5], np.full(5, -np.pi),
                            [-0.25] * 4, [0.0, 0.0, 0.0]])
CIRCLE_HI = np.concatenate([[1.5], np.full(5, 2 * np.pi), [0.6] * 4 + [0.6], [0.8] * 4 + [0.5], np.full(5, np.pi),
                            [0.25] * 4, [0.4, 0.4, 2 * np.pi]])
CONTROLLERS = {"wave": (BaodingWave, LO, HI), "circle": (BaodingCircle, CIRCLE_LO, CIRCLE_HI)}
_model = _data = _perturbations = _controller = _bowl = None


def load_bowl(path):
    """(target, center, angle) of a `search_stage3_bowl.py` result; None = the frozen bowl e."""
    if path is None:
        return None
    b = json.loads(Path(path).read_text())
    target = bowl_target() * 0.0
    for k, v in b["targets"].items():
        target[ACTUATOR_NAMES.index(k)] = v
    return target, np.array(b["center"]), b["angle"]


def _init(n_perturbed, controller="wave", bowl_path=None):
    global _model, _data, _perturbations, _controller, _bowl
    _controller, _bowl = CONTROLLERS[controller][0], load_bowl(bowl_path)
    _model = mj.MjModel.from_xml_path(TWO_BALL_SCENE)
    _data = mj.MjData(_model)
    rng = np.random.default_rng(12345)
    _perturbations = [dict(mass=rng.uniform(*MASS_RANGE), friction=rng.uniform(*FRICTION_RANGE))
                      for _ in range(n_perturbed)]


def run_scored(params):
    window = []   # each ball's angle over the last WINDOW s

    def on_step(k, t):
        if t >= SECONDS - WINDOW and k % 10 == 0:
            window.append(tracker_ref[0].total.copy())

    tracker_ref = []
    ctrl = _controller(params, base=None if _bowl is None else _bowl[0])
    angles, dropped, touched, t_end = run_baoding(_model, _data, ctrl, SECONDS, on_step=on_step,
                                                  tracker_out=tracker_ref, bowl=_bowl)
    s = 1.0 if angles.sum() >= 0 else -1.0
    if dropped or not window:
        progress = float(min(s * angles[0], s * angles[1]))
    else:
        w = s * np.array(window)
        progress = float(min(w[:, 0].min(), w[:, 1].min()))
    score = progress - 3.0 * dropped + 0.15 * len(touched)
    return dict(score=score, progress=progress, angles=[float(a) for a in angles], dropped=bool(dropped),
                touched=sorted(touched), t_end=float(t_end))


def evaluate(x):
    x = np.round(x, 4)
    ball_ids = [_model.body(b).id for b in BALLS]
    geom_ids = [_model.geom(b).id for b in BALLS]
    m0, f0 = _model.body_mass[ball_ids[0]], _model.geom_friction[geom_ids[0], 0]
    runs = [run_scored(x)]
    for p in _perturbations:
        _model.body_mass[ball_ids] = p["mass"]
        _model.geom_friction[geom_ids, 0] = p["friction"]
        runs.append(run_scored(x))
    _model.body_mass[ball_ids], _model.geom_friction[geom_ids, 0] = m0, f0
    return dict(score=float(np.mean([r["score"] for r in runs])), x=[float(v) for v in x], nominal=runs[0],
                mean_progress=float(np.mean([r["progress"] for r in runs])),
                frac_dropped=float(np.mean([r["dropped"] for r in runs])),
                mean_touched=float(np.mean([len(r["touched"]) for r in runs])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--pop", type=int, default=320)
    ap.add_argument("--elite", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--std-floor", type=float, default=0.04)
    ap.add_argument("--perturbed", type=int, default=2)
    ap.add_argument("--init", type=str, default=None)
    ap.add_argument("--controller", choices=sorted(CONTROLLERS), default="wave")
    ap.add_argument("--bowl", type=str, default=None, help="a search_stage3_bowl.py result (default bowl e)")
    ap.add_argument("--out", type=str, default="data/stage4_baoding_search.json")
    args = ap.parse_args()

    _init(args.perturbed, args.controller, args.bowl)   # fail here, not in a crashing Pool
    _, LO, HI = CONTROLLERS[args.controller]
    rng = np.random.default_rng(args.seed)
    mean, std = (LO + HI) / 2, (HI - LO) / 3
    if args.init:
        mean, std = np.array(json.loads(Path(args.init).read_text())["x"]), 0.1 * (HI - LO)
    best = None
    with Pool(16, initializer=_init, initargs=(args.perturbed, args.controller, args.bowl)) as pool:
        for it in range(args.iters):
            xs = np.clip(rng.normal(mean, std, size=(args.pop, len(LO))), LO, HI)
            if best is not None:
                xs[0] = best["x"]
            results = pool.map(evaluate, list(xs))
            results.sort(key=lambda r: -r["score"])
            elite = np.array([r["x"] for r in results[:args.elite]])
            mean, std = elite.mean(0), np.maximum(elite.std(0), args.std_floor * (HI - LO))
            best = results[0] if best is None or results[0]["score"] > best["score"] else best
            n = best["nominal"]
            print(f"iter {it}: best {best['score']:.2f} | progress {best['mean_progress']:.2f} rad "
                  f"dropped {best['frac_dropped']:.2f} digits {best['mean_touched']:.1f} | nominal angles "
                  f"{[round(a, 2) for a in n['angles']]} touched {n['touched']} | freq {best['x'][0]:.2f}Hz",
                  flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(best, indent=1))
    print(json.dumps(best["x"]))


if __name__ == "__main__":
    main()
