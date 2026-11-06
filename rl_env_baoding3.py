"""Stage 6: RL baoding with THREE balls. The Stage 5 env that worked (`rl_env_baoding2.BaodingEnv9`)
with the balls and the progress measure generalized; every reward term, limit and constant is its.

Checked first (`eval_stage6_hold.py`, `eval_stage6_push.py`): the Stage 3 bowl e holds three touching
balls 30-45mm for 20 s from ~2/3 of placements (the rest drop a ball onto a finger within 0.5 s), but
an ideal tangential push on each ball round the trio's centre, in bowl e or opened versions of it,
turns them at best 1.4 rad and a ball escapes between the fingers within 0.5 s: as with two balls, the
fingers have to move with the balls.

Progress: the least of
  - each ball's own signed arc round the trio's starting centroid (palm frame), divided by the
    circumradius of three touching balls (2r/sqrt(3)), so each ball moving into the next one's place
    is 2pi/3 (a CYCLE, the three-ball "swap"); a ball farther out than that counts as at it (settled
    trios are not exactly equilateral: a rigid full turn of one read 5.0/8.2/8.1 rad uncapped), and
  - the turn of the triangle itself: the mean of the three balls' angle steps round their CURRENT
    centroid (the Stage 5 pair-line turn; the trio drifting across the palm does not count).
Bonus per cycle: the Stage 5 swap bonus per rad (25 per pi), so 2pi/3 earns 16.7.

Starts: three touching balls dropped in a triangle at the bowl's ball centre (raised for the ball
size), the triangle at a random orientation; redrawn (up to 20 tries) until every ball has settled
clear of the knuckle drop line, is still, and is within 4mm of touching distance of the trio's centre
(env8's filter, for three balls). Curriculum: RADII3, 38mm -> 45mm.
"""
from pathlib import Path

import numpy as np

from hand_env import ACTUATOR_NAMES, BALL_RADIUS, BOWL_BALLS_CENTER, DIGITS
from rl_env_baoding2 import (BALL_Z_DROP, CONTROL_DT, FRICTION_RANGE, MASS_RANGE, MAX_R_50MM, SWAP_BONUS, TARGET_NOISE,
                             WRIST, BaodingEnv9)
from three_balls import BALLS3, THREE_BALL_SCENE, grasp, in_palm, places, set_radius

RADII3 = (0.019, 0.020, 0.021, 0.022, BALL_RADIUS)   # m; 38mm -> 45mm
CYCLE = 2 * np.pi / 3
START_SLACK = 0.004            # m beyond the touching circumradius a settled ball may sit


class TrioAngles:
    """Each ball's own cumulative angle round the trio's starting centroid in the palm plane
    (`hand_env.BallAngles` for three balls)."""

    def __init__(self, model, data):
        self.center = np.mean(in_palm(model, data), axis=0)
        self.prev = [self._angle(q) for q in in_palm(model, data)]
        self.total = np.zeros(3)

    def _angle(self, q):
        d = q - self.center
        return float(np.arctan2(d[2], d[0]))

    def update(self, model, data):
        for k, q in enumerate(in_palm(model, data)):
            a = self._angle(q)
            self.total[k] += (a - self.prev[k] + np.pi) % (2 * np.pi) - np.pi
            self.prev[k] = a
        return self.total

    def offsets(self, model, data):
        out = []
        for q in in_palm(model, data):
            d = q - self.center
            out.append((float(np.hypot(d[0], d[2])), float(-d[1])))
        return out


class BaodingEnv3Ball(BaodingEnv9):
    scene, balls = THREE_BALL_SCENE, BALLS3
    cycle, cycle_bonus = CYCLE, SWAP_BONUS * CYCLE / np.pi
    max_r = MAX_R_50MM

    @property
    def radius(self):
        return RADII3[self.level]

    def set_level(self, level):
        level = int(min(max(level, 0), len(RADII3) - 1))
        if level != self.level:
            self.level, self.pool = level, None
        return self.level

    def _set_radius(self, radius):
        set_radius(self.model, radius)

    def _balls_in_palm(self):
        return in_palm(self.model, self.data)

    def _make_tracker(self):
        return TrioAngles(self.model, self.data)

    def _pair_angle(self):
        """Each ball's angle round the trio's CURRENT centroid (the triangle's orientation)."""
        p = in_palm(self.model, self.data)
        c = np.mean(p, axis=0)
        return np.array([np.arctan2(q[2] - c[2], q[0] - c[0]) for q in p])

    def _progress(self):
        before = self.tracker.total.copy()
        steps = self.direction * (self.tracker.update(self.model, self.data) - before)
        offsets = self.tracker.offsets(self.model, self.data)
        rc = 2 * self.radius / np.sqrt(3)
        self.arc += steps * np.minimum(1.0, np.array([r for r, _ in offsets]) / rc)
        a = self._pair_angle()
        self.pair += self.direction * float(np.mean((a - self.pair_prev + np.pi) % (2 * np.pi) - np.pi))
        self.pair_prev = a
        return float(min(self.arc.min(), self.pair)), offsets

    def _ball_gap(self):
        p = [self.data.xpos[b] for b in self.ball_ids]
        return min(float(np.linalg.norm(p[i] - p[j])) for i, j in ((0, 1), (1, 2), (0, 2))) - 2 * self.radius

    def _settle(self, rng):
        m, d = self.model, self.data
        set_radius(m, self.radius)
        k3 = (self.radius / BALL_RADIUS) ** 3
        center = BOWL_BALLS_CENTER + np.array([0.0, 0.0, self.radius - BALL_RADIUS])
        rc = 2 * self.radius / np.sqrt(3)
        for k in range(20):
            angle = rng.uniform(0, CYCLE) if self.randomize else k * CYCLE / 20   # nominal: first that passes
            target, mass, friction = self.base.copy(), np.full(3, 0.08 * k3), self.friction0
            if self.randomize:
                mass = np.full(3, rng.uniform(*MASS_RANGE) * k3)
                friction = np.full(3, rng.uniform(*FRICTION_RANGE))
                noise = rng.uniform(-TARGET_NOISE, TARGET_NOISE, target.shape)
                noise[WRIST] = 0.0
                target = np.clip(target + noise, self.lo, self.hi)
            m.body_mass[self.ball_ids], m.geom_friction[self.geom_ids, 0] = mass, friction
            grasp(m, d, target, places(center, angle, self.radius), settle_time=0.3)
            p = in_palm(m, d)
            c = np.mean(p, axis=0)
            speed = [np.linalg.norm(d.qvel[m.jnt_dofadr[m.body_jntadr[b]]:][:3]) for b in self.ball_ids]
            if (max(q[2] for q in p) < BALL_Z_DROP - 0.015 and max(speed) < 0.05
                    and max(np.hypot(*(q - c)[[0, 2]]) for q in p) < rc + START_SLACK):
                break
        return mass, friction, d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.act.copy()


# ---- the little finger as a second thumb (runs three_03+)
# The user, on three_01 @40M's GIFs: "seems the little finger should be like thumb to help balls move
# around". Over 8 starts at 45mm the policy held LFJ5 (the little finger's metacarpal cupping it across
# the palm, toward the thumb) at the Stage 5 cap (bowl + 0.5 rad) 55% of steps, and in the good episode
# the little finger pushed 10.7 mJ forward and braked 11.8 (the thumb 44 forward). So: LFJ5 and LFJ4 get
# their full joint range (the pad-facing penalty still guards against run 03's back-rolled finger), and
# the little finger's forward work is credited double -- inside the same progress-bounded cap and its
# DIGIT_SHARE, so it re-divides the credit toward the little finger, it adds none.
LF_FULL_RANGE = ("rh_A_LFJ5", "rh_A_LFJ4")
LF_WEIGHT = 2.0


class BaodingEnv3BallLF(BaodingEnv3Ball):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for a in LF_FULL_RANGE:
            i = ACTUATOR_NAMES.index(a)
            self.lo[i], self.hi[i] = self.model.actuator_ctrlrange[i]

    def _credit(self, work, turn_digit):
        return super()._credit({dg: p * LF_WEIGHT if dg == "lf" and p > 0 else p for dg, p in work.items()},
                               turn_digit)


# ---- push, come back, hand on: a relay round the ring (runs three_04+)
# three_03 @50M, forward work over 16 starts: ff 69%, th 16%, lf 10%, mf 4%, rf 0% -- the ring finger is
# on a ball 44% of steps (its pads the closest of any digit, median 2.9mm) but only holds it. The user:
# "when little finger pass balls to the finger next to middle finger, that finger should be involved to
# pass balls to middle finger, then index finger, and so on" and "the little finger push ball to middle of
# hand, the little finger should come back for the ball come to it's position, and so on". So each digit's
# credit comes in STROKES:
#   - a digit is ARMED while its tip is within RETURN_BAND of where it stood in the settled bowl; an armed
#     digit starts a stroke when its forward work rate passes STROKE_RATE;
#   - the stroke ends when it stops pushing for STROKE_END s, or after STROKE_MAX s; the digit is then
#     spent -- its forward work earns nothing -- until its tip has come back within RETURN_BAND;
#   - a stroke by the digit NEXT round the ring (`next_digit`: lf -> rf -> mf -> ff -> th -> lf in
#     direction -1) after the last digit to start one counts RELAY_WEIGHT; skipping a digit, 1.
# Braking still counts in full. The relay clock's double weight and LF_WEIGHT are dropped (the relay
# order now comes from the strokes themselves); the per-digit share and the progress-bounded cap stay, so
# this re-divides the credit, it adds none. The ring finger's spread (RFJ4) gets its full range: three_03
# held it at the bowl - 0.3 cap 35% of steps.
STROKE_RATE = 0.002        # W of forward work: "pushing"
STROKE_END = 0.2           # s without pushing ends a stroke
STROKE_MAX = 1.0           # s
RETURN_BAND = 0.015        # m from the tip's settled bowl position: back, armed again
RELAY_WEIGHT = 2.0


class BaodingEnv3BallRelay(BaodingEnv3BallLF):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        i = ACTUATOR_NAMES.index("rh_A_RFJ4")
        self.lo[i], self.hi[i] = self.model.actuator_ctrlrange[i]

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.armed = {dg: True for dg in DIGITS}
        self.stroke_t = {dg: None for dg in DIGITS}     # control steps into its stroke, None = not stroking
        self.idle = {dg: 0 for dg in DIGITS}
        self.relay_stroke = {dg: False for dg in DIGITS}
        self.last_stroker, self.strokes, self.relay_strokes = None, [], 0
        self.relay_next = self._relay_ring()
        return obs, info

    def _relay_ring(self):
        """{digit: the digit a stroke should hand on to}: the ring round the balls in `direction`."""
        return dict(self.next_digit)

    def _credit(self, work, turn_digit):
        home = np.linalg.norm(self._tips() - self.tips0, axis=1)
        weighted = {}
        for i, dg in enumerate(DIGITS):
            p = work.get(dg, 0.0)
            if self.stroke_t[dg] is not None:
                self.stroke_t[dg] += 1
                self.idle[dg] = 0 if p > STROKE_RATE else self.idle[dg] + 1
                if self.idle[dg] * CONTROL_DT >= STROKE_END or self.stroke_t[dg] * CONTROL_DT >= STROKE_MAX:
                    self.stroke_t[dg], self.armed[dg] = None, False
            elif not self.armed[dg] and home[i] < RETURN_BAND:
                self.armed[dg] = True
            if self.stroke_t[dg] is None and self.armed[dg] and p > STROKE_RATE:
                self.stroke_t[dg], self.idle[dg] = 0, 0
                relay = dg in self.relay_next
                self.relay_stroke[dg] = relay and self.last_stroker is not None and dg == self.relay_next[self.last_stroker]
                self.relay_strokes += self.relay_stroke[dg]
                if relay:
                    self.last_stroker = dg
                self.strokes.append(dg)
            if p > 0:
                p = 0.0 if self.stroke_t[dg] is None else p * (RELAY_WEIGHT if self.relay_stroke[dg] else 1.0)
            if dg in work:
                weighted[dg] = p
        return BaodingEnv9._credit(self, weighted, None)   # no relay-clock weight, no LF_WEIGHT

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["strokes"], info["relay_strokes"] = len(self.strokes), self.relay_strokes
        info["armed"] = [dg for dg in DIGITS if self.armed[dg]]
        info["stroke_seq"] = self.strokes[-40:]
        return obs, reward, terminated, truncated, info


# ---- the ring finger as a guide (runs three_05+)
# three_04 @56M (6M steps on BaodingEnv3BallRelay): rf still 0% of the forward work, 5-6 of ~40 strokes
# in ring order. Probing from 40 states of a good cycle (policy frozen, every 1-3 joint combination of a
# digit driven either way for 0.3 s): the ring finger could do forward work on a ball in 1 state (10.6
# mJ; else <= 0.4), the little finger in most (up to 121 mJ) -- where the trio passes it, the ring
# finger's curl presses a ball toward the palm, not along its path. The user chose: "ring finger as
# guide". The relay ring skips it: lf -> mf -> ff -> th -> lf (direction -1). Its strokes still earn the
# plain credit but neither count as a hand-off nor break the chain.
GUIDES = ("rf",)


class BaodingEnv3BallGuide(BaodingEnv3BallRelay):

    def _relay_ring(self):
        ring = {}
        for dg, nxt in self.next_digit.items():
            if dg in GUIDES:
                continue
            while nxt in GUIDES:
                nxt = self.next_digit[nxt]
            ring[dg] = nxt
        return ring


# ---- training past the jam (runs three_06+)
# The user asked for 2+ circles. Run 90 s, every policy (three_01 16-40M, three_05 64-76M, 38-45mm) still
# stops at 2.2-2.8 rad: the trio wedges between the thumb's base segment, the little finger's metacarpal and
# the palm at one orientation (an ideal 0.3-1 N forward push on each ball moves them < 0.15 rad at 40, 42
# and 45mm; harder pops a ball out). A cross-entropy search over 5-waypoint finger sequences (no push) from
# a stall found one that passes it under control: 2.80 -> 4.53 rad, the balls at rest after. Handed back,
# three_05 dropped a ball 4.5 s later -- it never saw the jam's far side. So a reverse curriculum: a share
# of starts are JAM states (the policy's own stalls) and PASSAGE states (along the found sequence),
# from `collect_stage6_jam_states.py`; the reward is BaodingEnv3BallGuide's.
JAM_STATES = Path(__file__).parent / "data" / "stage6_jam_states.npz"
P_JAM, P_PASSAGE = 0.4, 0.2


class BaodingEnv3BallJam(BaodingEnv3BallGuide):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        z = np.load(JAM_STATES)
        self.jam_states = {k: [tuple(z[f"{k}_{f}"][i] for f in ("mass", "friction", "qpos", "qvel", "ctrl", "act", "tips0"))
                               + (k,) for i in range(len(z[f"{k}_mass"]))] for k in ("jam", "passage")}

    def _settle(self, rng):
        u = rng.uniform() if self.randomize else 1.0
        if self.level == len(RADII3) - 1 and u < P_JAM + P_PASSAGE:   # the states are 45mm
            k = "jam" if u < P_JAM else "passage"
            return self.jam_states[k][rng.integers(len(self.jam_states[k]))]
        return super()._settle(rng)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.start_kind = self.start_entry[7] if len(self.start_entry) > 6 else "start"
        if len(self.start_entry) > 6:
            self.tips0 = self.start_entry[6].copy()     # the bowl of that state's episode
        return self._obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["start_kind"] = self.start_kind
        return obs, reward, terminated, truncated, info
