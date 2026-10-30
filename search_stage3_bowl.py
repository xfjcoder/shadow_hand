"""Stage 3 grasp search: the two-ball BOWL baoding starts from. Cross-entropy method over all 20
actuator targets (wrist included) plus where the pair of touching balls is placed (midpoint +-2cm in
x and y around mid-palm, and the pair's angle in the palm plane). Each candidate: both balls placed
and held while the hand closes, released (`hand_env.grasp_balls`), then 1.5s.

The shape is the user's: "all 5 finger point up and curl a little inside and make bowl shape with
two balls sit in the palm". So the search space is limited to that shape -- each finger's base
joint (J3) 0.9-1.57 rad, i.e. standing up off the palm, and its curl (J0, the coupled middle+distal)
0.1-1.2 rad, i.e. only a little inward -- and the score rewards what makes it a bowl. The wrist is
free because a level palm is open on the wrist side: a hand-set bowl dropped one ball straight out
(media/stage3_bowl_manual.png); tilting toward the fingers lets the balls settle into the cup.
Score, per run:
      + 15 per ball still in place (moved < 1cm in the palm frame over the 1.5s)
      + 8 if the two balls touch each other
      + 6 per ball resting on the palm ("sit in the palm")
      + 4 per digit touching a ball with anything but its back (the bowl's walls)
      + up to 3 per digit for its tip standing up off the palm (full at 7cm, along the palm normal),
        thumb included -- it is a wall too, not a bar across the palm
      - 400 * the std of the five tip heights (m): the user, on the first bowl, whose thumb stuck
        out sideways: "tips of 5 fingers almost like in same height while making the bowl shape" --
        the tips are the bowl's rim (the thumb CAN reach the fingers' ~8cm: up to 8.8cm)
      - 300 * how far (m) each tip is outside 2.5-6cm from the ball pair's midpoint, measured in the
        palm plane: the rim rings the balls rather than bunching on one side
      - 800 * the std of the five gaps between neighbouring tips around the rim (thumb, ff, mf, rf,
        lf, back to thumb; m). The user, on the second bowl (tips bunched 5.2cm across, gaps
        thumb-ff 3.6cm and lf-thumb 6.4cm vs 2.6-3.0cm between fingers): "make 4 fingers separated
        more to make gap between fingers and thumb smaller on both side". Spreading (J4) barely
        moves an upright finger's tip -- it twists the finger about its own length, 0.2 rad moving
        the tip <= 0.3cm -- so the fingers may lean out at the base (J3 down to 0.5) and curl back
        up to the rim
      - 300 * how far (m) the ff tip is inside 4.5cm across the palm (thumb side) and the lf tip
        inside 4.0cm (other side), and the thumb tip inside 4.0cm: the gap term alone was gamed --
        the search closed the gaps by moving the THUMB in across the palm while the finger span
        shrank (5.7 -> 5.3cm). Kinematically, with tips 7cm up, ff can reach 5.1cm out and lf 4.6cm
        (spread joints at their outward limit, J3 ~0.7-0.8)
      - 5 per DORSAL contact, - 1 per SIDE contact
      - 1 per N of total finger force on the balls above 8 N (a bowl, not a clamp: the balls must
        stay free to roll round each other)
      - 200 * mean drift of the held balls
      + (`--circulate`) `--circ-weight` (10) * the ideal-push circulation, capped at 2 pi: with the
        hand holding the bowl still, a tangential `PUSH_FORCE` on each ball round the pair's
        midpoint (`hand_env.push_forces`) for `PUSH_TIME` s, each way from the same settled state;
        circulation = the better way's min over the two balls of that ball's own angle
        (`hand_env.BallAngles`), 0 if a ball leaves the bowl during that push, and - 20 more if both
        ways drop one. Added for Stage 4: in bowl e the scripted wave only rocked the balls, and even this ideal push stalled them after ~1/4 turn -- the gold
        ball jams against the ff/mf bases or the thumb's proximal segment lying across the palm, the
        red against the lf base (media/stage4_wave_and_push.gif). Only on the nominal run and the
        first `--push-runs` - 1 perturbed ones (it costs 4x the hold), and the WORST of those runs
        counts, added once to the run-averaged score: with the mean, a bowl averaged 2.2 rad from push
        runs of 0.46 (nominal), 0.42 and one lucky 5.6 -- it didn't circulate
averaged over the nominal run and `--perturbed` fixed perturbations (targets +-0.05 rad, placement
+-3mm, mass 40-150g, friction 0.5-1.0 -- the Stage 2 ranges).

Usage:
    python3 search_stage3_bowl.py --out data/stage3_bowl_search.json [--init <previous result json>]
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import mujoco as mj
import numpy as np

from eval_stage2_hold import FRICTION_RANGE, MASS_RANGE, PLACE_NOISE, TARGET_NOISE
from hand_env import (ACTUATOR_NAMES, BALL_PLACE, BALLS, BALL_RADIUS, TWO_BALL_SCENE, BallAngles,
                      balls_in_palm, _segment_top, classify_contacts, grasp_balls, push_forces,
                      set_ball_radius, two_ball_places)

FREE = np.arange(len(ACTUATOR_NAMES))
# The bowl shape: fingers standing up (J3), tips only a little curled in (J0). Other actuators keep
# their full ctrlrange.
SHAPE_RANGES = {f"rh_A_{f}J3": (0.5, 1.5708) for f in ("FF", "MF", "RF", "LF")}
SHAPE_RANGES.update({f"rh_A_{f}J0": (0.1, 1.2) for f in ("FF", "MF", "RF", "LF")})
TIP_HEIGHT = 0.07
RING = (0.025, 0.06)
SPAN = {"ff": 0.045, "lf": 0.04, "th": 0.04}   # min outward tip offset across the palm (m)   # tip distance from the ball pair's midpoint, in the palm plane (m)
PAD_FORCE = 0.2
HOLD_TIME = 1.5
PUSH_FORCE = 1.0
PUSH_TIME = 3.0
# Extra search dimensions after the actuator targets: pair midpoint dx, dy (m), pair angle (rad).
EXTRA_LO, EXTRA_HI = np.array([-0.02, -0.02, 0.0]), np.array([0.02, 0.02, np.pi])
_model = _data = _snap = _perturbations = None
_push_runs = 0
_circ_weight = 10.0
_radius = BALL_RADIUS


def _init(n_perturbed, push_runs=0, circ_weight=10.0, radius=BALL_RADIUS):
    global _model, _data, _snap, _perturbations, _push_runs, _circ_weight, _radius
    _push_runs, _circ_weight, _radius = push_runs, circ_weight, radius
    _model = mj.MjModel.from_xml_path(TWO_BALL_SCENE)
    set_ball_radius(_model, radius)
    _data = mj.MjData(_model)
    _snap = mj.MjData(_model)
    rng = np.random.default_rng(12345)
    _perturbations = [dict(dt=rng.uniform(-TARGET_NOISE, TARGET_NOISE, len(ACTUATOR_NAMES)),
                           dd=np.append(rng.uniform(-PLACE_NOISE, PLACE_NOISE, 2), 0.0),
                           mass=rng.uniform(*MASS_RANGE), friction=rng.uniform(*FRICTION_RANGE))
                      for _ in range(n_perturbed)]


def decode(x):
    target = np.zeros(len(ACTUATOR_NAMES))
    target[FREE] = x[:len(FREE)]
    dx, dy, angle = x[len(FREE):]
    return target, BALL_PLACE + np.array([dx, dy, 0.0]), angle


def tip_positions(model, data):
    """Each digit's fingertip (far end of its distal segment), ff/mf/rf/lf/th, as (across the palm,
    height above the palm surface, along the palm toward the fingers) in m. The palm body's frame
    has the palm normal along local -y; the palm surface is ~1.2cm out from its origin."""
    R = data.body("rh_palm").xmat.reshape(3, 3)
    out = []
    for dg in ("ff", "mf", "rf", "lf", "th"):
        bid = model.body(f"rh_{dg}distal").id
        top, _ = _segment_top(model, bid)
        tip = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ np.array([0.0, 0.0, top])
        p = R.T @ (tip - data.body("rh_palm").xpos)
        out.append(np.array([p[0], -p[1] - 0.012, p[2]]))
    return out


def push_circulation():
    """The ideal-push test from the current (settled) state, both ways: (circulation, dropped) of
    the better way. The state is restored afterwards."""
    mj.mj_copyData(_snap, _model, _data)
    ball_ids = [_model.body(b).id for b in BALLS]
    best = (-np.inf, True)
    for sign in (1.0, -1.0):
        mj.mj_copyData(_data, _model, _snap)
        tracker = BallAngles(_model, _data)
        dropped = False
        dt = _model.opt.timestep
        for k in range(int(PUSH_TIME / dt)):
            f = sign * PUSH_FORCE * min(1.0, k * dt / 0.3)
            for bid, fv in zip(ball_ids, push_forces(_model, _data, tracker.center, f)):
                _data.xfrc_applied[bid, :3] = fv
            mj.mj_step(_model, _data)
            if k % 10 == 0:
                tracker.update(_model, _data)
                if any(r > 0.06 or h > 0.05 + _radius for r, h in tracker.offsets(_model, _data)):
                    dropped = True
                    break
        # A dropped push earns no circulation: the first version kept the angle gained up to the
        # drop, and the search found bowls where the push flings the balls round and out (4.5 rad,
        # 2 of 3 pushes dropping a ball).
        circ = 0.0 if dropped else float(min(sign * tracker.total[0], sign * tracker.total[1]))
        if (circ - 2.0 * dropped) > (best[0] - 2.0 * best[1]):
            best = (circ, dropped)
    _data.xfrc_applied[:] = 0.0
    mj.mj_copyData(_data, _model, _snap)
    return best


def run_once(target, center, angle, push=False):
    """One cradle run; returns its score terms."""
    grasp_balls(_model, _data, target, two_ball_places(center, angle, _radius))
    p0 = balls_in_palm(_model, _data)
    contacts = []
    for b in BALLS:
        contacts += classify_contacts(_model, _data, b)
    for _ in range(int(HOLD_TIME / _model.opt.timestep)):
        mj.mj_step(_model, _data)
    p1 = balls_in_palm(_model, _data)
    circ, push_dropped = push_circulation() if push else (0.0, False)
    tips = tip_positions(_model, _data)
    heights = [t[1] for t in tips]
    upright = sum(min(max(h, 0.0), TIP_HEIGHT) / TIP_HEIGHT for h in heights)
    rim_std = float(np.std(heights))
    mid = 0.5 * (p1[0] + p1[1])
    ring = [float(np.hypot(t[0] - mid[0], t[2] - mid[2])) for t in tips]
    ring_out = sum(max(0.0, RING[0] - r) + max(0.0, r - RING[1]) for r in ring)
    order = [4, 0, 1, 2, 3]   # th, ff, mf, rf, lf around the rim
    gaps = [float(np.hypot(*(tips[order[i]] - tips[order[(i + 1) % 5]])[[0, 2]])) for i in range(5)]
    gap_std = float(np.std(gaps))
    # across: + toward the thumb side. ff and th must be out on the + side, lf out on the - side.
    span_in = (max(0.0, SPAN["ff"] - tips[0][0]) + max(0.0, SPAN["lf"] + tips[3][0])
               + max(0.0, SPAN["th"] - tips[4][0]))
    drifts = [float(np.linalg.norm(a - b)) for a, b in zip(p0, p1)]
    held = [d for d in drifts if d < 0.01]
    gap = float(np.linalg.norm(_data.body("ball1").xpos - _data.body("ball2").xpos))
    firm = [c for c in contacts if c[3] >= PAD_FORCE]
    walls = {c[0] for c in firm if c[0] != "palm" and c[2] != "DORSAL"}
    on_palm = sum(any(c[0] == "palm" for c in classify_contacts(_model, _data, b)) for b in BALLS)
    n_dorsal = sum(c[2] == "DORSAL" for c in firm)
    n_side = sum(c[2] == "side" for c in firm)
    grip = sum(c[3] for c in contacts if c[0] != "palm")
    score = (15 * len(held) + 8 * (gap < 2 * _radius + 0.003) + 6 * on_palm + 4 * len(walls)
             + 3 * upright - 400 * rim_std - 300 * ring_out - 800 * gap_std - 300 * span_in - 5 * n_dorsal - n_side - max(0.0, grip - 8.0)
             - 200 * (np.mean(held) if held else 0.0) - 20 * push_dropped)
    return dict(score=score, n_held=len(held), touching=gap < 2 * _radius + 0.003, on_palm=on_palm,
                walls=sorted(walls), upright=upright, rim_std=rim_std, ring_out=ring_out,
                gaps=gaps, gap_std=gap_std, span_in=span_in,
                tips_across=[float(t[0]) for t in tips],
                n_dorsal=n_dorsal, grip=grip, push=push, circ=circ, push_dropped=bool(push_dropped))


def evaluate(x):
    x = np.round(x, 4)   # score exactly what gets frozen
    target, center, angle = decode(x)
    ball_ids = [_model.body(b).id for b in BALLS]
    geom_ids = [_model.geom(b).id for b in BALLS]
    m0, f0 = _model.body_mass[ball_ids[0]], _model.geom_friction[geom_ids[0], 0]
    runs = [run_once(target, center, angle, push=_push_runs > 0)]
    for i, p in enumerate(_perturbations):
        _model.body_mass[ball_ids] = p["mass"]
        _model.geom_friction[geom_ids, 0] = p["friction"]
        runs.append(run_once(target + p["dt"], center + p["dd"], angle, push=i + 1 < _push_runs))
    _model.body_mass[ball_ids], _model.geom_friction[geom_ids, 0] = m0, f0
    nominal = runs[0]
    circs = [r["circ"] for r in runs if r["push"]] or [0.0]
    return dict(score=float(np.mean([r["score"] for r in runs]) + _circ_weight * min(min(circs), 2 * np.pi)),
                x=[float(v) for v in x], circs=circs,
                nominal={k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in nominal.items()},
                frac_both_held=float(np.mean([r["n_held"] == 2 for r in runs])),
                frac_touching=float(np.mean([r["touching"] for r in runs])),
                mean_on_palm=float(np.mean([r["on_palm"] for r in runs])),
                mean_walls=float(np.mean([len(r["walls"]) for r in runs])),
                mean_upright=float(np.mean([r["upright"] for r in runs])),
                mean_rim_std=float(np.mean([r["rim_std"] for r in runs])),
                mean_ring_out=float(np.mean([r["ring_out"] for r in runs])),
                mean_gap_std=float(np.mean([r["gap_std"] for r in runs])),
                mean_span_in=float(np.mean([r["span_in"] for r in runs])),
                mean_circ=float(min(circs)),
                frac_push_dropped=float(np.mean([r["push_dropped"] for r in runs if r["push"]] or [0.0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--pop", type=int, default=320)
    ap.add_argument("--elite", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--std-floor", type=float, default=0.06)
    ap.add_argument("--perturbed", type=int, default=5)
    ap.add_argument("--out", type=str, default="data/stage3_bowl_search.json")
    ap.add_argument("--circulate", action="store_true", help="add the ideal-push circulation term")
    ap.add_argument("--push-runs", type=int, default=4)
    ap.add_argument("--ball-radius", type=float, default=BALL_RADIUS)
    ap.add_argument("--circ-weight", type=float, default=10.0,
                    help="per rad of worst-case push circulation; 10 left the bowl shape dominating "
                         "(search from bowl e ended at 0.27-0.57 rad)")
    ap.add_argument("--init", type=str, default=None,
                    help="start the search around a previous result's x (std 10%% of each range)")
    args = ap.parse_args()

    push_runs = args.push_runs if args.circulate else 0
    _init(args.perturbed, push_runs, args.circ_weight, args.ball_radius)   # fail here, not in a Pool that respawns crashing workers
    lo, hi = _model.actuator_ctrlrange[FREE, 0].copy(), _model.actuator_ctrlrange[FREE, 1].copy()
    for name, (a, b) in SHAPE_RANGES.items():
        lo[ACTUATOR_NAMES.index(name)], hi[ACTUATOR_NAMES.index(name)] = a, b
    lo, hi = np.append(lo, EXTRA_LO), np.append(hi, EXTRA_HI)
    rng = np.random.default_rng(args.seed)
    mean, std = (lo + hi) / 2, (hi - lo) / 3
    if args.init:
        mean, std = np.array(json.loads(Path(args.init).read_text())["x"]), 0.1 * (hi - lo)
    best = None
    with Pool(16, initializer=_init, initargs=(args.perturbed, push_runs, args.circ_weight, args.ball_radius)) as pool:
        for it in range(args.iters):
            xs = np.clip(rng.normal(mean, std, size=(args.pop, len(lo))), lo, hi)
            if best is not None:
                xs[0] = best["x"]
            results = pool.map(evaluate, list(xs))
            results.sort(key=lambda r: -r["score"])
            elite = np.array([r["x"] for r in results[:args.elite]])
            mean, std = elite.mean(0), np.maximum(elite.std(0), args.std_floor * (hi - lo))
            best = results[0] if best is None or results[0]["score"] > best["score"] else best
            print(f"iter {it}: best {best['score']:.1f} | both held {best['frac_both_held']:.2f} "
                  f"touching {best['frac_touching']:.2f} on palm {best['mean_on_palm']:.2f}/2 "
                  f"walls {best['mean_walls']:.2f}/5 upright {best['mean_upright']:.2f}/5 "
                  f"rim std {100 * best['mean_rim_std']:.1f}cm ring out {100 * best['mean_ring_out']:.1f}cm "
                  f"gap std {100 * best['mean_gap_std']:.1f}cm span in {100 * best['mean_span_in']:.1f}cm | "
                  f"tips across {[round(100 * a, 1) for a in best['nominal']['tips_across']]} gaps "
                  f"{[round(100 * g, 1) for g in best['nominal']['gaps']]} "
                  f"walls {best['nominal']['walls']} grip {best['nominal']['grip']:.1f}N | circ "
                  f"{[round(c, 2) for c in best['circs']]} rad push-dropped {best['frac_push_dropped']:.2f}", flush=True)
    target, center, angle = decode(np.array(best["x"]))
    best["targets"] = {ACTUATOR_NAMES[i]: float(target[i]) for i in FREE}
    best["center"], best["angle"] = [float(v) for v in center], float(angle)
    best["ball_radius"] = args.ball_radius
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(best, indent=1))
    print(json.dumps(best["targets"]), best["center"], best["angle"])


if __name__ == "__main__":
    main()
