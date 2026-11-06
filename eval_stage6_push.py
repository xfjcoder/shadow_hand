"""Stage 6: the ideal-push test for three balls (Stage 4's for two). The hand holds a bowl still while
a tangential force on each ball drives it round the trio's starting centre, in the palm plane. Each
ball's own angle round that centre is tracked; progress = the LEAST of the three (all must go round;
2*pi/3 = each ball has moved into the next one's place). If even this push can't carry them round,
no finger motion inside that (still) bowl will. Bowls: Stage 3's bowl e, and bowl e opened (knuckles
J3 and curl J0 of the four fingers reduced, thumb THJ5 swung) -- the opening that let two balls swap.

Usage:
    python3 eval_stage6_push.py
"""
import mujoco as mj
import numpy as np

from hand_env import ACTUATOR_NAMES, BALL_RADIUS, BOWL_BALLS_CENTER, bowl_target
from three_balls import BALLS3, THREE_BALL_SCENE, grasp, in_palm, places, set_radius

IDX = {a: i for i, a in enumerate(ACTUATOR_NAMES)}


def opened(dk, dc, dt):
    t = bowl_target().copy()
    for f in ("FF", "MF", "RF", "LF"):
        t[IDX[f"rh_A_{f}J3"]] -= dk
        t[IDX[f"rh_A_{f}J0"]] -= dc
    t[IDX["rh_A_THJ5"]] += dt
    return t


def push(radius, angle, direction, force, target, seconds=8.0):
    m = mj.MjModel.from_xml_path(THREE_BALL_SCENE)
    d = mj.MjData(m)
    set_radius(m, radius)
    # settle in bowl e, then open gradually with the balls in it
    grasp(m, d, bowl_target(), places(BOWL_BALLS_CENTER + np.array([0.0, 0.0, radius - BALL_RADIUS]), angle, radius))
    start = d.ctrl.copy()
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    target = np.clip(target, lo, hi)
    for k in range(250):
        d.ctrl[:] = start + (target - start) * (k + 1) / 250
        mj.mj_step(m, d)
    c = np.mean(in_palm(m, d), axis=0)
    ang = lambda q: np.arctan2(q[2] - c[2], q[0] - c[0])
    prev, total = [ang(q) for q in in_palm(m, d)], np.zeros(3)
    ids = [m.body(b).id for b in BALLS3]
    R = d.body("rh_palm").xmat.reshape(3, 3)
    for k in range(int(seconds / m.opt.timestep)):
        if k % 10 == 0:
            qs = in_palm(m, d)
            for j, q in enumerate(qs):
                a = ang(q)
                total[j] += direction * ((a - prev[j] + np.pi) % (2 * np.pi) - np.pi)
                prev[j] = a
                if q[2] > 0.105 or np.hypot(q[0] - c[0], q[2] - c[2]) > 0.07 or abs(q[1] - c[1]) > 0.08:
                    return total.min(), total, k * m.opt.timestep
                t = direction * np.array([-(q[2] - c[2]), 0.0, q[0] - c[0]])
                d.xfrc_applied[ids[j], :3] = force * R @ (t / max(np.linalg.norm(t), 1e-6))
        mj.mj_step(m, d)
    return total.min(), total, None


def main():
    bowls = {"bowl e": (0, 0, 0), "open .3/.3/-.3": (0.3, 0.3, -0.3), "open .5/.5/-.3": (0.5, 0.5, -0.3),
             "open .3/.3/+.3": (0.3, 0.3, 0.3)}
    for radius in (0.019, 0.021, 0.0225):
        for name, o in bowls.items():
            res = []
            for direction in (1, -1):
                for force in (0.5, 1.0):
                    best = max((push(radius, a, direction, force, opened(*o)) for a in (0.0, 0.7, 1.4)),
                               key=lambda r: r[0])
                    p, tot, drop = best
                    res.append(f"dir {direction:+d} {force}N {p:+.2f}" + (f" (drop {drop:.1f}s)" if drop is not None else ""))
            print(f"{2000 * radius:.0f}mm {name:15s}: " + " | ".join(res), flush=True)


if __name__ == "__main__":
    main()
