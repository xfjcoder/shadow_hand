"""Stage 5, second reward design (runs baoding_07+). Runs 01-06 (`rl_env_baoding`) patched one
loophole per run and the policy found the next: by baoding_06 the per-step "style" terms (pads on
balls, facing, reach, relay pushes, palm) paid ~180 per episode against ~51 for a swap, and the
policy earned them by wobbling the WRIST (WRJ2 std 0.14 rad) while the fingertips only jiggled
1-5mm against the balls, which barely turned (0.38 rad in 20 s). The user: "the hand move with
wrist move, fingers relative to the wrist not move, and fingers not make a round bowl shape, and
finger's move should side push balls cross each other, the goal is to swap balls, so we need think
through rl reward structure."

Principles:
  1. Only BALL MOTION earns positive reward. Everything about style is a limit or a penalty, so
     holding a nice pose never beats swapping.
  2. What the user specified is a hard limit where possible: the wrist is not an action (held at
     the bowl tilt); joint targets stay near the bowl (`SPAN`, `SPAN_OF`).
  3. A digit's credit is the WORK it does on a ball along the orbit: (its contact force's component
     along the ball's forward tangent round the pair center) x (the ball's speed along that tangent,
     relative to the palm). A pad rubbing a still ball does none, a wrist wobble carrying hand and
     balls together does none (palm-relative speed), squeezing from both sides cancels.

Reward:
  + W_ORBIT x the step in PROGRESS (as `rl_env_baoding`: the least of each ball's signed arc and the
    pair-line rotation, pi = a swap), + SWAP_BONUS each time progress passes a new multiple of pi.
  + W_WORK x the pad faces' forward work on the balls (J), the digit whose turn it is on the relay
    clock (observed, `TURN_SECONDS` per digit round the ring in `direction`) counted double. Only
    while the ball moves forward, and the positive part only while progress is at its best so far
    (FRONTIER) and capped at WORK_CAP per episode; the negative part (braking) never capped. Checked
    before training: holding still, a +-0.3 rad 1 Hz wrist wobble and random finger jiggling all
    earn ~0 J; balls driven forward by an external force against still fingers earn negative.
  - W_SHAPE per cm any fingertip strays more than SHAPE_BAND from where it stood in the settled bowl
    (palm frame): the round bowl -- tips on a ring round the balls at one height (Stage 3).
  - W_OUT per cm a ball's center is past BALL_Z_MAX toward the fingers (on the finger bases, not the
    palm); past BALL_Z_DROP it counts as dropped.
  - W_FACE per digit, for its middle/distal pads facing more than 60 deg away from the nearest ball.
  - W_DORSAL per firm back-of-segment contact on a ball. - W_RATE x squared target change.
  - DROP + DROP_PER_SECOND x seconds left; the episode ends.
The balls need not touch each other (user: "balls don't have touch each other all times as long as
they can swap").

Curriculum (user: "build it with the curriculum"): Stage 4's ideal-push test showed two 45mm balls
cannot pass each other in the bowl (jam between the knuckle row and the thumb's base segment).
Episodes start at `RADII[0]`; `set_level` (called by the training callback when enough episodes
swap) moves to the next radius, up to the real 45mm ball.

baoding_07 (SHAPE_BAND 1.5cm, direction +1) stalled at progress 0.36, exactly the ideal-push ceiling
of the closed bowl: with the hand still, NO bowl lets the balls swap -- one ball is always pinned by
the knuckle row, the little finger side or the thumb base (even 40mm balls; Stage 4's "4.6 rad" was
one ball's angle, not a swap). Opening the bowl (knuckles -0.3..0.5 rad, curl, thumb) moves the tips
2-3.5cm and lets the balls get much further, more so in direction -1, but a fixed open pose drops
them: the fingers must open out of a ball's way and close behind it. So baoding_08+ allow 3.5cm.
"""
import mujoco as mj
import numpy as np
from gymnasium import spaces

from hand_env import (ACTUATOR_NAMES, BALL_RADIUS, BALLS, BOWL_BALLS_ANGLE, BOWL_BALLS_CENTER, DIGITS,
                      balls_in_palm, classify_contacts, grasp_balls, set_ball_radius, two_ball_places)
from rl_env_baoding import (BaodingEnv, CONTROL_DT, DROP, DROP_PER_SECOND, FRICTION_RANGE, MASS_RANGE,
                            MAX_H, MAX_R, N_HAND_Q, PAD_FORCE, POOL, TARGET_NOISE, W_RATE)

RADII = (0.020, 0.021, 0.022, BALL_RADIUS)   # m; 40mm -> 45mm

W_ORBIT = 10.0
SWAP_BONUS = 25.0
W_WORK = 300.0             # per J of forward work on the balls by pads (a steady 1 N push in the bowl
                           # does ~0.01 J in 4 s: the balls move slowly)
WORK_CAP = 0.05            # J of positive (credited) work per episode -> at most 15 reward
FRONTIER = 0.02            # rad: work is credited only while progress is within this of its best
TURN_WEIGHT = 2.0          # the digit on turn's work counts double
W_SHAPE = 0.02             # per cm beyond SHAPE_BAND, per fingertip
SHAPE_BAND = 0.035         # m (baoding_07: 0.015, which locked the bowl closed -- see the docstring)
W_OUT = 0.05               # per cm, per ball
BALL_Z_MAX = 0.09          # palm frame z; the knuckle row is at ~9.5cm
BALL_Z_DROP = 0.105
W_FACE = 0.02              # x (0.5 - cos) when a digit's pads face > 60 deg from the nearest ball
W_DORSAL = 0.1

WRIST = [i for i, a in enumerate(ACTUATOR_NAMES) if a.startswith("rh_A_WR")]
MOVING = [i for i in range(len(ACTUATOR_NAMES)) if i not in WRIST]


class BaodingEnv2(BaodingEnv):
    work_cap = WORK_CAP
    shape_band = SHAPE_BAND

    def __init__(self, direction=1.0, randomize=True, episode_seconds=20.0, level=0):
        super().__init__(direction=direction, randomize=randomize, episode_seconds=episode_seconds)
        self.r0 = float(self.model.geom_size[self.geom_ids[0], 0])
        self.level = level
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(MOVING),), dtype=np.float32)
        obs_dim = 2 * N_HAND_Q + len(MOVING) + 6 + 6 + 4 + len(DIGITS) + 2 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    # ---- curriculum
    @property
    def radius(self):
        return RADII[self.level]

    def set_level(self, level):
        level = int(min(max(level, 0), len(RADII) - 1))
        if level != self.level:
            self.level, self.pool = level, None   # the next reset settles a new pool at this size
        return self.level

    def _settle(self, rng):
        set_ball_radius(self.model, self.radius)
        k3 = (self.radius / BALL_RADIUS) ** 3     # the Stage 2 mass range is for 45mm balls
        target, mass, friction = self.base.copy(), np.full(2, 0.08 * k3), self.friction0
        if self.randomize:
            mass = np.full(2, rng.uniform(*MASS_RANGE) * k3)
            friction = np.full(2, rng.uniform(*FRICTION_RANGE))
            noise = rng.uniform(-TARGET_NOISE, TARGET_NOISE, target.shape)
            noise[WRIST] = 0.0
            target = np.clip(target + noise, self.lo, self.hi)
        self.model.body_mass[self.ball_ids], self.model.geom_friction[self.geom_ids, 0] = mass, friction
        grasp_balls(self.model, self.data, target,
                    two_ball_places(BOWL_BALLS_CENTER, BOWL_BALLS_ANGLE, self.radius), settle_time=0.3)
        d = self.data
        return mass, friction, d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.act.copy()

    def reset(self, seed=None, options=None):
        if self.pool is not None:
            set_ball_radius(self.model, self.radius)
        obs, info = super().reset(seed=seed, options=options)
        self.tips0 = self._tips()
        self.work_credit = 0.0
        self.best = 0.0
        self.swaps = 0
        self.next_swap = np.pi
        return self._obs(), info

    # ---- measurements
    def _tips(self):
        R, palm = self._palm_R(), self.data.body("rh_palm").xpos
        return np.array([R.T @ (self.data.body(f"rh_{d}distal").xpos - palm) for d in DIGITS])

    def _pad_work(self):
        """{digit: forward work rate (W) of its pad contacts on the balls}: per contact, the force on
        the ball along the ball's forward orbit tangent times the ball's palm-relative speed along it.
        Also the count of firm back-of-segment contacts."""
        m, d = self.model, self.data
        R, palm = self._palm_R(), d.body("rh_palm").xpos
        f6, v6, vp = np.zeros(6), np.zeros(6), np.zeros(6)
        mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_XBODY, m.body("rh_palm").id, vp, 0)
        work, n_dorsal = {}, 0
        skip = self._no_credit()
        for j, (b, bid) in enumerate(zip(BALLS, self.ball_ids)):
            q = R.T @ (d.xpos[bid] - palm) - self.tracker.center
            rq = np.hypot(q[0], q[2])
            if rq < 1e-4:
                continue
            tangent = self.direction * np.array([-q[2], 0.0, q[0]]) / rq     # palm frame, forward
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_XBODY, bid, v6, 0)
            v_palm_at_ball = vp[3:] + np.cross(vp[:3], d.xpos[bid] - palm)
            # forward motion only: a pad resisting a ball that rolls BACK (gravity) would otherwise
            # earn credit, and rocking back and forth would pile it up (work check, scratchpad)
            v_t = max(0.0, float((R.T @ (v6[3:] - v_palm_at_ball)) @ tangent))
            if j in skip:
                v_t = 0.0          # still count its dorsal contacts below
            for digit, seg, face, fn, i in classify_contacts(m, d, b, with_index=True):
                if digit == "palm" or fn < PAD_FORCE:
                    continue
                if face == "DORSAL":
                    n_dorsal += 1
                    continue
                if face not in ("palmar", "tip"):
                    continue
                c = d.contact[i]
                mj.mj_contactForce(m, d, i, f6)
                f_world = c.frame.reshape(3, 3).T @ f6[:3]          # on geom2, from geom1
                if m.geom_bodyid[c.geom1] == bid:
                    f_world = -f_world
                work[digit] = work.get(digit, 0.0) + float((R.T @ f_world) @ tangent) * v_t
        return work, n_dorsal

    def _no_credit(self):
        """Indices of balls whose forward work earns nothing."""
        return ()

    def _obs(self):
        o = super()._obs()
        # drop the wrist targets from the ctrl block; add the ball radius (curriculum level)
        ctrl0 = 2 * N_HAND_Q
        o = np.concatenate([o[:ctrl0], o[ctrl0:ctrl0 + self.model.nu][MOVING], o[ctrl0 + self.model.nu:],
                            [100.0 * (self.radius - RADII[0])]])
        return o.astype(np.float32)

    # ---- step
    def step(self, action):
        prev_ctrl = self.data.ctrl.copy()
        ctrl = prev_ctrl.copy()
        ctrl[MOVING] += np.clip(action, -1, 1) * self.rate[MOVING]
        ctrl[WRIST] = self.base[WRIST]
        ctrl = np.clip(ctrl, self.lo, self.hi)
        self.data.ctrl[:] = ctrl
        for _ in range(self.n_sub):
            mj.mj_step(self.model, self.data)
        self.t += 1
        turn_digit = DIGITS[self.turn]
        self._clock()

        before = self.tracker.total.copy()
        steps = self.direction * (self.tracker.update(self.model, self.data) - before)
        offsets = self.tracker.offsets(self.model, self.data)
        self.arc += steps * np.array([r for r, _ in offsets]) / self.radius
        a = self._pair_angle()
        self.pair += self.direction * ((a - self.pair_prev + np.pi) % (2 * np.pi) - np.pi)
        self.pair_prev = a
        progress = float(min(self.arc.min(), self.pair))
        d_progress, self.progress = progress - self.progress, progress

        work, n_dorsal = self._pad_work()
        w = sum(p * (TURN_WEIGHT if dg == turn_digit else 1.0) for dg, p in work.items()) * CONTROL_DT
        self.best = max(self.best, self.progress)
        if w > 0:   # positive work only on new ground (rocking back and forth earns it once)
            w = 0.0 if self.progress < self.best - FRONTIER else min(w, max(0.0, self.work_cap - self.work_credit))
            self.work_credit += w
        tips = self._tips()
        stray_cm = float(np.sum(np.maximum(0.0, np.linalg.norm(tips - self.tips0, axis=1) - self.shape_band))) * 100
        ball_z = [float(q[2]) for q in balls_in_palm(self.model, self.data)]
        out_cm = sum(max(0.0, z - BALL_Z_MAX) for z in ball_z) * 100
        facing = self._pad_facing()
        away = sum(max(0.0, 0.5 - c) for c in facing.values())

        reward = (W_WORK * w - W_SHAPE * stray_cm - W_OUT * out_cm - W_FACE * away - W_DORSAL * n_dorsal
                  - W_RATE * float(np.sum((ctrl - prev_ctrl) ** 2)))
        dropped = any(r > MAX_R or h > MAX_H for r, h in offsets) or max(ball_z) > BALL_Z_DROP
        if dropped:
            reward -= DROP + DROP_PER_SECOND * (self.episode_steps - self.t) * CONTROL_DT
        else:
            reward += W_ORBIT * d_progress
            while self.progress >= self.next_swap:
                reward += SWAP_BONUS
                self.swaps += 1
                self.next_swap += np.pi
        truncated = self.t >= self.episode_steps
        info = {"progress": self.progress, "arcs": self.arc.copy(), "pair": self.pair,
                "angles": self.tracker.total.copy(), "dropped": dropped, "swaps": self.swaps,
                "work": work, "work_credit": self.work_credit, "n_dorsal": n_dorsal, "turn": turn_digit,
                "stray_cm": stray_cm, "ball_z": ball_z, "facing": facing, "radius": self.radius,
                "gap": self._ball_gap()}
        if dropped or truncated:
            info["success"] = not dropped
            info["swapped"] = float(self.swaps >= 1 and not dropped)
        return self._obs(), reward, dropped, truncated, info


# ---- reward v3 (runs baoding_09+)
# baoding_08 @10M (bowl band 3.5cm, direction -1) turned the balls 1.2 rad in the first second, then
# CLAMPED them (thumb pressing the gold ball into the palm, ~5 N; the red one wedged between the thumb
# tip and the rf/lf bases; an ideal 0.8 N push moved them 0.08 rad from there) for the rest of the
# episode. User: "balls make one move then thumb and little finger hold two balls ... balls should move
# continuously around and around". Why, from the reward: 25.5 of its 27.2 came in that first second
# (WORK_CAP spent at 0.58 s), holding cost nothing, and opening the clamp risked a drop worth up to
# ~65 for at most 10 per rad of new progress. It never got past the thumb base, so never saw a swap.
#   1. Holding still costs: W_STALL per second, scaled by how far the BEST progress (the frontier, so
#      rocking back and forth cannot fake it) fell short of STALL_RATE over the last STALL_WINDOW s.
#      Less than the drop's DROP_PER_SECOND, so dropping is still worse than stalling.
#   2. No per-episode work cap: work is still credited only on new ground (FRONTIER), so it is bounded
#      by real progress, and keeps paying all the way round.
#   3. Episodes start at different points of the cycle: the balls placed at START_ANGLES settle into
#      three distinct arrangements (pair line ~0, ~1.35, ~2.0 rad round from the bowl's, each with
#      either ball on either side), so the policy practises the passage past the thumb base it never
#      reached from the bowl start.
#   4. Without the cap, one ball circling while the other stays pinned (what a static bowl allows)
#      would earn work forever at the frontier, so only pushes on a ball not already LEAD ahead of
#      the progress (the lagging one, or both while they move together) earn credit.
LEAD = 0.2                 # rad: no work credit on a ball whose arc is this far ahead of progress
W_STALL = 1.0              # per second
STALL_RATE = 0.25          # rad/s of frontier progress expected
STALL_WINDOW = 2.0         # s
START_ANGLES = BOWL_BALLS_ANGLE - np.arange(8) * np.pi / 4


class BaodingEnv3(BaodingEnv2):
    work_cap = np.inf          # see 2. above

    def _settle(self, rng):
        angle = START_ANGLES[rng.integers(len(START_ANGLES))] if self.randomize else BOWL_BALLS_ANGLE
        set_ball_radius(self.model, self.radius)
        k3 = (self.radius / BALL_RADIUS) ** 3
        target, mass, friction = self.base.copy(), np.full(2, 0.08 * k3), self.friction0
        if self.randomize:
            mass = np.full(2, rng.uniform(*MASS_RANGE) * k3)
            friction = np.full(2, rng.uniform(*FRICTION_RANGE))
            noise = rng.uniform(-TARGET_NOISE, TARGET_NOISE, target.shape)
            noise[WRIST] = 0.0
            target = np.clip(target + noise, self.lo, self.hi)
        self.model.body_mass[self.ball_ids], self.model.geom_friction[self.geom_ids, 0] = mass, friction
        grasp_balls(self.model, self.data, target,
                    two_ball_places(BOWL_BALLS_CENTER, angle, self.radius), settle_time=0.3)
        d = self.data
        return mass, friction, d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.act.copy()

    def _no_credit(self):
        return [j for j in range(2) if self.arc[j] > self.progress + LEAD]

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.best_hist = []
        return obs, info

    def step(self, action):
        obs, reward, dropped, truncated, info = super().step(action)
        self.best_hist.append(self.best)
        n = int(round(STALL_WINDOW / CONTROL_DT))
        stall = 0.0
        if len(self.best_hist) > n and not dropped:
            advance = self.best_hist[-1] - self.best_hist[-1 - n]
            stall = max(0.0, 1.0 - advance / (STALL_RATE * STALL_WINDOW))
            reward -= W_STALL * stall * CONTROL_DT
        info["stall"] = stall
        return obs, reward, dropped, truncated, info


# ---- relaxed bowl (runs baoding_10+)
# baoding_09 ended like 08: ~1.2 rad, then stuck where one ball sinks into the thumb-base/wrist corner.
# The penalties were all ~0 (the policy kept the style), the stall cost a flat ~-16 per episode, and an
# open-loop search over finger motions within the same limits moved the balls 0.04 rad from there.
# User: "we can relax shape limit, and make it larger, as long as balls not drop out of hand". The
# flexion joints get their full range (no SPAN), the spread (J4) and little-finger cupping (LFJ5) caps
# are doubled (the pad-facing penalty still guards against the back-rolled little finger of baoding_03),
# and the fingertip band is 6cm. Drops are penalized as before.
SHAPE_BAND_RELAXED = 0.06
SPAN_RELAXED = {"rh_A_LFJ5": 0.5, "rh_A_FFJ4": 0.3, "rh_A_MFJ4": 0.3, "rh_A_RFJ4": 0.3, "rh_A_LFJ4": 0.3}


class BaodingEnv4(BaodingEnv3):
    shape_band = SHAPE_BAND_RELAXED

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lo, hi = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        span = np.array([SPAN_RELAXED.get(a, np.inf) for a in ACTUATOR_NAMES])
        self.lo, self.hi = np.maximum(lo, self.base - span), np.minimum(hi, self.base + span)
        self.lo[WRIST] = self.hi[WRIST] = self.base[WRIST]


# ---- work credit bounded by progress (runs baoding_11+)
# baoding_10 @10M: the uncapped work credit paid 43 per episode against 11 for the progress itself, all
# on steps that DID gain new ground -- the policy pressed harder (index finger 170 mJ, the energy going
# into friction) instead of moving the balls further; progress flattened from ~6M. Work is force x
# speed, so pushing harder pays as much as pushing further. Credited work is now capped at WORK_PER_RAD
# per rad of best progress: the "which finger pushed" signal stays, extra force earns nothing.
WORK_PER_RAD = 0.05        # J -> at most 15 reward per rad, the ratio of baoding_08's honest first move


class BaodingEnv5(BaodingEnv4):

    @property
    def work_cap(self):
        return WORK_PER_RAD * max(self.best, 0.0)


# ---- 50mm balls in a bowl that wraps them (runs baoding_12+)
# baoding_10/11 with 40mm balls: the balls ROCKED in a bowl with room to spare (each centre travelling
# ~12cm per 20 s for ~1.6cm of net turn round each other). User: "we need much big balls so that balls
# should completely occupy the as big as possible bowl shaped by five fingers, then fingers should make
# efforts to make two balls rotate around each other, and our reward design should be based on this";
# chose 50mm. Checked first: 50mm balls stay in opened bowls that drop 40/45mm ones (they wedge between
# fingers and palm); an open-loop finger search turned a 50mm pair 1.0 rad in 3 s from the bowl.
# The bowl: `BOWL_50MM` below. No ball-size curriculum. The reward is BaodingEnv5's: with
# two balls packed in the bowl they turn together, so progress (the least of each ball's arc and the
# turn of the line between them) is the pair rotating round each other; the bonus is per half turn
# (a swap). "Fill the bowl" is the setup, not a reward term (style rewards were gamed in runs 04-06).
import json
from pathlib import Path

# Searched bowls (data/stage5_bowl_50mm*.json: the thumb would not stay on the balls, and the pair rolled
# out through the thumb-index gap within 2-8 s held still) held worse than the Stage 3 bowl e, which keeps
# two 50mm balls 20 s with ff, mf, lf and the palm on them -- so bowl e it is, the balls 2.5mm higher.
BOWL_50MM = Path(__file__).parent / "data" / "stage5_bowl_e_50mm.json"
RADIUS_50MM = 0.025


class BaodingEnv6(BaodingEnv5):

    def __init__(self, *args, bowl=BOWL_50MM, **kwargs):
        super().__init__(*args, **kwargs)
        b = json.loads(Path(bowl).read_text())
        self.base = np.array([b["targets"][a] for a in ACTUATOR_NAMES])
        self.center, self.angle = np.array(b["center"]), float(b["angle"])
        lo, hi = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        span = np.array([SPAN_RELAXED.get(a, np.inf) for a in ACTUATOR_NAMES])
        self.lo, self.hi = np.maximum(lo, self.base - span), np.minimum(hi, self.base + span)
        self.lo[WRIST] = self.hi[WRIST] = self.base[WRIST]

    @property
    def radius(self):
        return RADIUS_50MM

    def set_level(self, level):
        return 0

    def _settle(self, rng):
        # only the bowl arrangement, either ball on either side: 50mm balls placed at the other start
        # angles (`START_ANGLES`, for the 40mm corner) land on the fingers and fall out
        angle = self.angle + rng.integers(2) * np.pi if self.randomize else self.angle
        set_ball_radius(self.model, self.radius)
        k3 = (self.radius / BALL_RADIUS) ** 3
        target, mass, friction = self.base.copy(), np.full(2, 0.08 * k3), self.friction0
        if self.randomize:
            mass = np.full(2, rng.uniform(*MASS_RANGE) * k3)
            friction = np.full(2, rng.uniform(*FRICTION_RANGE))
            noise = rng.uniform(-TARGET_NOISE, TARGET_NOISE, target.shape)
            noise[WRIST] = 0.0
            target = np.clip(target + noise, self.lo, self.hi)
        self.model.body_mass[self.ball_ids], self.model.geom_friction[self.geom_ids, 0] = mass, friction
        grasp_balls(self.model, self.data, target, two_ball_places(self.center, angle, self.radius), settle_time=0.3)
        d = self.data
        return mass, friction, d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.act.copy()


# ---- the ball-size curriculum continued to 50mm (runs baoding_11 resumed at 30M, "baoding_13")
# baoding_12 (BaodingEnv6, 50mm from scratch) held the balls but turned them only ~1.2 rad per 20 s at
# 30M, never a swap; baoding_11 (BaodingEnv5) grew from 40mm to 45mm by the curriculum and swaps ~3x per
# 20 s at 44mm. So the 50mm balls are reached the way that worked: the same env, the curriculum going on.
RADII_TO_50 = RADII + (0.0235, 0.024, 0.025)   # m; 45mm -> 47, 48, 50mm


class BaodingEnv7(BaodingEnv5):

    @property
    def radius(self):
        return RADII_TO_50[self.level]

    def set_level(self, level):
        level = int(min(max(level, 0), len(RADII_TO_50) - 1))
        if level != self.level:
            self.level, self.pool = level, None
        return self.level

    # 48-50mm balls placed at some `START_ANGLES` settle with one ball up on the knuckles (z ~10cm,
    # dropped at 10.5) and the episode ends in a step: draw another start until both balls settle clear
    # of the drop limits (the 40-45mm starts nearly always pass on the first draw)
    def _settle(self, rng):
        for _ in range(20):
            out = super()._settle(rng)
            ball_z = [float(q[2]) for q in balls_in_palm(self.model, self.data)]
            if max(ball_z) < BALL_Z_DROP - 0.01:
                return out
        return out
