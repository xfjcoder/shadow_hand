"""Stage 5: RL baoding from the Stage 3 bowl. Stage 4 showed why a scripted controller can't do it:
in every bowl found, a ball pushed round jams after ~1/4 turn against the knuckle row or the
thumb's proximal segment, so the digits have to move OUT of a ball's way as it arrives and push it
from behind -- motion keyed to where the balls are, which is what a policy observing the balls can
learn.

Episode: the bowl grasp (`hand_env.BOWL_GRASP`, both balls held while the hand closes, released,
settled), then `EPISODE_SECONDS` of 50 Hz control. Each action moves every actuator target by up to
`RATE` rad per step (fingers/thumb; the wrist slower), kept within `SPAN` rad of the bowl grasp.

Reward, lessons carried from the Allegro project's five baoding RL rounds:
  + W_ORBIT * the step in PROGRESS = the least of: each ball's own signed ARC round the pair's
    starting midpoint in the palm frame (radius * angle step, `hand_env.BallAngles`), accumulated
    per ball, divided by the ball radius -- so a swap (each ball half way round at one radius) is
    pi -- and the rotation of the line between the two balls (else both balls carried round the
    midpoint side by side, never passing each other, would count). Signed in the chosen `direction`: going backwards costs the same. The min of
    the two balls' totals means one ball circling while the other sits pinned earns nothing (the
    Allegro's pair-vector bug).
    The first version (run baoding_01) credited per-step "matched" ANGLE steps -- the smaller of the
    two balls' steps when both moved the same way -- and was gamed twice at once: the red ball sat
    0.3-1.5cm from the midpoint and jiggled (tiny motions, huge angle steps), and the gold ball
    rocked back and forth, so its forward half-swings were credited (red moving the same way) and
    its backward ones dropped (red moving the other way): 10 rad of "matched" in 7.5s with gold
    net +6 rad and lifted 4cm off the palm. Arc length kills the first; accumulating each ball's
    signed total BEFORE taking the min kills the second (rocking cancels). The Allegro's matched
    reward had the same rectification flaw.
  + MILESTONE each time progress reaches a new pi/4.
  + W_PUSH per digit whose pad (palmar or tip face) is pressing a ball while that contact point
    sweeps round the pair center in `direction` -- tanh of its rate, so pushing backwards costs as
    much. The user, on baoding_02's GIF: "all fingers should move one direction, either clockwise
    or anti-clockwise to try to make balls cross each and touch each other sliding". baoding_02 at
    14M: ff/mf/rf on the balls 96% of steps but sweeping forward only 5-7%, backward 10-11%; lf and
    th hardly touching. (baoding_02 had W_GOOD per digit merely in pad contact while fingers moved.)
  + W_TOUCH while the pair turns in `direction`, the closer the balls the more (the user's "touch
    each other sliding"; baoding_02 kept them a median 9.3mm apart, never touching).
  + W_FACE per digit, cos of the angle between its middle/distal pads and the nearest ball -- the
    user, on baoding_03's little finger turned back-out: "all fingers should use inside to face
    balls". Plus W_ALL while all five digits press a ball with a pad.
  + W_REACH per digit, closer to a ball the more (baoding_04 at 20M: lf and th never touched).
  + RELAY each time the pushing passes to the next digit round the ring in `direction` (user: "fingers
    should sequentially move clockwise or counter-clockwise, push balls cross each other in a relay
    way"), only after RELAY_MIN of pair rotation since the last handoff.
  - W_BOWL x mean squared offset of the targets from the bowl grasp (user: "keep fingers and hand in
    bowl shape").
  + W_TURN for the digit whose turn it is on a RELAY CLOCK (observed; round the ring in `direction`,
    TURN_SECONDS each) sweeping a ball forward with its pad -- baoding_05's sparse RELAY bonus gave
    4 handoffs in 20 s and an index finger pushing backward. W_PAD per digit on a ball with a pad.
  + W_PALM per ball on the palm, - W_OUT per cm a ball is past BALL_Z_MAX toward the fingers (user:
    "two balls should be mostly in palm instead of on bottom of three middle fingers").
  - W_DORSAL per firm contact by the BACK of a segment (the Allegro thumb pushed with its back).
  - W_RATE * squared target change (smoothness).
  - DROP + DROP_PER_SECOND x the seconds left, and the episode ends when a ball leaves the bowl: > 6cm from the start midpoint in the
    palm plane, or > `MAX_H` above its start height (baoding balls roll ON the palm; baoding_01's
    gold ball was carried up between the fingers).
Episodes are long (20 s) so stopping early has consequences (the Allegro's 10 s episodes trained
policies that stalled right after).

`randomize=True` draws ball mass and friction (the Stage 2 ranges, narrowed) and jitters the bowl
targets. The grasp takes 450 physics steps -- 45 control steps' worth -- and in a SubprocVecEnv all
16 envs wait for any one that is resetting (training ran at ~1.5k steps/s with workers ~40% busy),
so each env settles a pool of `POOL` randomized grasps once and a reset restores one of them.
"""
import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from hand_env import (ACTUATOR_NAMES, BALL_RADIUS, BALLS, BOWL_BALLS_ANGLE, BOWL_BALLS_CENTER, DIGITS,
                      TWO_BALL_SCENE, BallAngles, balls_in_palm, bowl_target, classify_contacts,
                      grasp_balls, palmar_axis, two_ball_places)

CONTROL_DT = 0.02
EPISODE_SECONDS = 20.0
SETTLE_SECONDS = 0.3
RATE = 0.05                # rad per control step, fingers and thumb
WRIST_RATE = 0.02
SPAN = 0.9                 # how far (rad) any target may move from the bowl grasp
# ... except these. baoding_03 cupped LFJ5 (an oblique axis: it rolls the whole little finger) to
# its 0.785 limit and LFJ4 to its limit within 1 s, turning the little finger's back to the balls,
# then straightened it out of the way. An upright finger's J4 mostly twists it (Stage 3).
SPAN_OF = {"rh_A_LFJ5": 0.25, "rh_A_FFJ4": 0.15, "rh_A_MFJ4": 0.15, "rh_A_RFJ4": 0.15, "rh_A_LFJ4": 0.15}
N_HAND_Q = 24              # hand joints come first in qpos/qvel; then the two balls' free joints

W_ORBIT = 10.0
MILESTONE_RAD = np.pi / 4
MILESTONE = 5.0
W_GOOD = 0.0               # baoding_02: 0.01; replaced by W_PUSH in baoding_03
W_PUSH = 0.01              # per digit pushing a ball in `direction` with its pad (tanh of its rate)
PUSH_RATE = 0.5            # rad/s round the pair center: a push this fast earns ~0.76 W_PUSH
W_TOUCH = 0.03             # balls close / touching while the pair turns in `direction`: x (1 - gap/TOUCH_FAR)
TOUCH_GAP = 0.001          # m between the balls' surfaces that counts as touching (reported)
TOUCH_FAR = 0.01           # baoding_02 at 14M kept them a median 9.3mm apart
W_FACE = 0.01              # per digit: cos(its middle+distal pads' angle to the nearest ball), in [-1, 1]
W_ALL = 0.02               # all five digits pressing a ball with their pads
W_REACH = 0.005            # per digit: exp(-gap / REACH_SCALE), gap from its middle/distal segments to a ball
REACH_SCALE = 0.01         # bowl start: lf 18mm, ff 23, rf 28, th 31, mf 32 (tips stand above the balls)
RELAY = 1.0                # the push passing to the NEXT digit round the ring in `direction`
RELAY_RATE = 0.2           # rad/s: a pad sweeping a ball forward at least this fast is "pushing"
RELAY_MIN = 0.1            # rad of pair rotation since the last handoff, else no bonus (no flicking)
W_BOWL = 0.05              # x mean squared target offset from the bowl grasp (rad^2): keep the bowl
W_DORSAL = 0.1             # baoding_02/03: 0.02, 04/05: 0.05
W_PAD = 0.005              # per digit pressing a ball with a pad (baoding_05: lf on a ball 1% of steps)
W_TURN = 0.05              # the digit whose turn it is on the relay clock sweeping a ball forward (tanh)
TURN_SECONDS = 0.6         # relay clock: each digit's turn, round the ring in `direction`
W_PALM = 0.01              # per ball resting on the palm
W_OUT = 0.05               # per ball, per cm its center is past BALL_Z_MAX toward the fingers
BALL_Z_MAX = 0.09          # palm frame z (knuckle row at ~9.5cm); baoding_05 kept the gold ball at a
                           # median 12.1cm, on the ring finger's proximal/middle segments, palm 2%
W_RATE = 0.1
DROP = 10.0
DROP_PER_SECOND = 3.0      # plus this per second of the episode left (baoding_02 swapped, then dropped)
PAD_FORCE = 0.2
MIN_FINGER_SPEED = 0.1     # rad/s, norm over hand joints: "moving" for the W_GOOD gate
MAX_R, MAX_H = 0.06, 0.035

MASS_RANGE = (0.06, 0.11)
FRICTION_RANGE = (0.6, 1.0)
TARGET_NOISE = 0.03
POOL = 16


class BaodingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, direction=1.0, randomize=True, episode_seconds=EPISODE_SECONDS):
        super().__init__()
        self.model = mj.MjModel.from_xml_path(TWO_BALL_SCENE)
        self.data = mj.MjData(self.model)
        self.direction, self.randomize = float(direction), randomize
        self.episode_steps = int(episode_seconds / CONTROL_DT)
        self.n_sub = int(round(CONTROL_DT / self.model.opt.timestep))
        self.base = bowl_target()
        lo, hi = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        span = np.array([SPAN_OF.get(a, SPAN) for a in ACTUATOR_NAMES])
        self.lo, self.hi = np.maximum(lo, self.base - span), np.minimum(hi, self.base + span)
        self.pad_axes = {d: [(self.model.body(f"rh_{d}{s}").id, palmar_axis(self.model, f"rh_{d}{s}"))
                             for s in ("middle", "distal")] for d in DIGITS}
        self.rate = np.array([WRIST_RATE if a.startswith("rh_A_WR") else RATE for a in ACTUATOR_NAMES])
        self.ball_ids = [self.model.body(b).id for b in BALLS]
        self.geom_ids = [self.model.geom(b).id for b in BALLS]
        self.mass0 = self.model.body_mass[self.ball_ids].copy()
        self.friction0 = self.model.geom_friction[self.geom_ids, 0].copy()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = 2 * N_HAND_Q + self.model.nu + 6 + 6 + 4 + len(DIGITS) + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.pool = None
        self.reach_geoms = {d: [g for g in range(self.model.ngeom)
                                if self.model.geom_bodyid[g] in (self.model.body(f"rh_{d}middle").id,
                                                                 self.model.body(f"rh_{d}distal").id)
                                and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])]
                            for d in DIGITS}
        self.next_digit = None

    def _pair_angle(self):
        p1, p2 = balls_in_palm(self.model, self.data)
        return float(np.arctan2(p2[2] - p1[2], p2[0] - p1[0]))

    def _palm_R(self):
        return self.data.body("rh_palm").xmat.reshape(3, 3)

    def _pad_push_rates(self):
        """{digit: rate (rad/s, signed in `direction`) at which its pad's contact point on a ball
        travels round the pair's start midpoint, in the palm frame} for every digit pressing a ball
        with a palmar or tip face (the strongest contact if several). Positive = sweeping the ball
        the way it should circulate; negative = shoving it backwards."""
        R, palm = self._palm_R(), self.data.body("rh_palm").xpos
        vel, best = np.zeros(6), {}
        for b in BALLS:
            ball = self.model.body(b).id
            for digit, seg, face, fn, i in classify_contacts(self.model, self.data, b, with_index=True):
                if digit == "palm" or face not in ("palmar", "tip") or fn < PAD_FORCE:
                    continue
                c = self.data.contact[i]
                b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
                bid, pos = (b2 if b1 == ball else b1), c.pos
                mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_XBODY, bid, vel, 0)
                v = R.T @ (vel[3:] + np.cross(vel[:3], pos - self.data.xpos[bid]))
                r = R.T @ (pos - palm) - self.tracker.center
                w = self.direction * (r[0] * v[2] - r[2] * v[0]) / max(r[0] ** 2 + r[2] ** 2, 1e-6)
                if digit not in best or fn > best[digit][0]:
                    best[digit] = (fn, w)
        return {d: w for d, (fn, w) in best.items()}

    def _pad_facing(self):
        """{digit: mean over its middle and distal segments of cos(angle between the pad's outward
        normal and the direction, across the segment, to the nearer ball)}: +1 = the inside faces the
        balls, -1 = the back does (the user: "all fingers should use inside to face balls")."""
        balls = [self.data.xpos[b] for b in self.ball_ids]
        out = {}
        for d, segs in self.pad_axes.items():
            cs = []
            for bid, ax in segs:
                R, p = self.data.xmat[bid].reshape(3, 3), self.data.xpos[bid]
                u = min(balls, key=lambda q: np.linalg.norm(q - p)) - p
                u = u - (u @ R[:, 2]) * R[:, 2]
                cs.append(float((R @ ax) @ u / max(np.linalg.norm(u), 1e-9)))
            out[d] = float(np.mean(cs))
        return out

    def _reach_gaps(self):
        """{digit: distance (m) from its middle/distal segments to the nearer ball's surface}."""
        return {d: min(mj.mj_geomDistance(self.model, self.data, g, b, 0.1, None)
                       for g in gs for b in self.geom_ids) for d, gs in self.reach_geoms.items()}

    def _ring_order(self):
        """{digit: the next digit round the balls in `direction`}, from where each fingertip stands
        round the pair's start midpoint (bowl: th -28 deg, ff 47, mf 68, rf 99, lf 146; the wrist
        side open) -- the user: "fingers should sequentially move clockwise or counter-clockwise,
        push balls cross each other in a relay way"."""
        R, palm = self._palm_R(), self.data.body("rh_palm").xpos
        ang = {}
        for d in DIGITS:
            q = R.T @ (self.data.body(f"rh_{d}distal").xpos - palm) - self.tracker.center
            ang[d] = np.arctan2(q[2], q[0])
        ring = sorted(DIGITS, key=ang.get)[::1 if self.direction > 0 else -1]
        return {d: ring[(i + 1) % len(ring)] for i, d in enumerate(ring)}

    def _ball_gap(self):
        """Surface gap between the two balls (m); the bowl grasp settles them ~3.5mm apart."""
        p1, p2 = (self.data.body(b).xpos for b in BALLS)
        return float(np.linalg.norm(p1 - p2)) - 2 * self.model.geom_size[self.geom_ids[0], 0]

    def _obs(self):
        R = self._palm_R()
        rel = [q - self.tracker.center for q in balls_in_palm(self.model, self.data)]
        vel = [R.T @ self.data.qvel[self.model.jnt_dofadr[self.model.body(b).jntadr[0]]:][:3] for b in BALLS]
        ang = [np.arctan2(r[2], r[0]) for r in rel]
        return np.concatenate([
            self.data.qpos[:N_HAND_Q], 0.1 * self.data.qvel[:N_HAND_Q], self.data.ctrl,
            10.0 * np.concatenate(rel), np.concatenate(vel),
            np.cos(ang), np.sin(ang),
            np.eye(len(DIGITS))[self.turn], [np.cos(2 * np.pi * self.turn_phase), np.sin(2 * np.pi * self.turn_phase)],
        ]).astype(np.float32)

    def _settle(self, rng):
        """One settled grasp: (mass, friction, qpos, qvel, ctrl, act)."""
        target, mass, friction = self.base.copy(), self.mass0, self.friction0
        if self.randomize:
            mass, friction = np.full(2, rng.uniform(*MASS_RANGE)), np.full(2, rng.uniform(*FRICTION_RANGE))
            target = np.clip(target + rng.uniform(-TARGET_NOISE, TARGET_NOISE, target.shape), self.lo, self.hi)
        self.model.body_mass[self.ball_ids], self.model.geom_friction[self.geom_ids, 0] = mass, friction
        grasp_balls(self.model, self.data, target, two_ball_places(BOWL_BALLS_CENTER, BOWL_BALLS_ANGLE),
                    settle_time=SETTLE_SECONDS)
        d = self.data
        return mass, friction, d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), d.act.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.pool is None:
            self.pool = [self._settle(self.np_random) for _ in range(POOL if self.randomize else 1)]
        mass, friction, qpos, qvel, ctrl, act = self.pool[self.np_random.integers(len(self.pool))]
        self.model.body_mass[self.ball_ids], self.model.geom_friction[self.geom_ids, 0] = mass, friction
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:], self.data.qvel[:], self.data.ctrl[:], self.data.act[:] = qpos, qvel, ctrl, act
        mj.mj_forward(self.model, self.data)
        self.tracker = BallAngles(self.model, self.data)
        self.t = 0
        self.arc = np.zeros(2)
        self.pair = 0.0
        self.pair_prev = self._pair_angle()
        self.progress = 0.0
        self.next_milestone = MILESTONE_RAD
        if self.next_digit is None:
            self.next_digit = self._ring_order()
        self.pusher, self.pair_at_handoff, self.relays = None, 0.0, 0
        self.ring = [DIGITS[0]]
        while len(self.ring) < len(DIGITS):
            self.ring.append(self.next_digit[self.ring[-1]])
        self.turn0 = int(self.np_random.integers(len(DIGITS)))   # index into DIGITS
        self._clock()
        return self._obs(), {}

    def _clock(self):
        """The relay clock: whose turn it is (`self.turn`, an index into DIGITS) and how far into it."""
        n = self.t * CONTROL_DT / TURN_SECONDS
        k = (self.ring.index(DIGITS[self.turn0]) + int(n)) % len(self.ring)
        self.turn, self.turn_phase = DIGITS.index(self.ring[k]), n % 1.0

    def step(self, action):
        prev_ctrl = self.data.ctrl.copy()
        ctrl = np.clip(prev_ctrl + np.clip(action, -1, 1) * self.rate, self.lo, self.hi)
        self.data.ctrl[:] = ctrl
        for _ in range(self.n_sub):
            mj.mj_step(self.model, self.data)
        self.t += 1
        turn_digit = DIGITS[self.turn]   # whose turn it was during this step
        self._clock()

        before = self.tracker.total.copy()
        steps = self.direction * (self.tracker.update(self.model, self.data) - before)
        offsets = self.tracker.offsets(self.model, self.data)
        self.arc += steps * np.array([r for r, _ in offsets]) / BALL_RADIUS
        a = self._pair_angle()
        d_pair = (a - self.pair_prev + np.pi) % (2 * np.pi) - np.pi
        self.pair += self.direction * d_pair
        self.pair_prev = a
        progress = float(min(self.arc.min(), self.pair))
        d_progress, self.progress = progress - self.progress, progress

        good, n_dorsal = set(), 0
        for b in BALLS:
            for digit, seg, face, fn in classify_contacts(self.model, self.data, b):
                if fn < PAD_FORCE or digit == "palm":
                    continue
                if face == "palmar":
                    good.add(digit)
                elif face == "DORSAL":
                    n_dorsal += 1
        moving = np.linalg.norm(self.data.qvel[:N_HAND_Q]) >= MIN_FINGER_SPEED
        push = self._pad_push_rates()
        facing = self._pad_facing()
        n_pad = len(push)
        gap = self._ball_gap()
        touching = gap < TOUCH_GAP
        turning = self.direction * d_pair > 0

        reach = self._reach_gaps()
        relay = 0.0
        pushing = {d for d, w in push.items() if w > RELAY_RATE}
        if pushing:
            if self.pusher is None:
                self.pusher = max(pushing, key=push.get)
            else:
                # the role only moves FORWARD round the ring (one or two places); the bonus only for
                # the next digit, and only if the balls turned since the last handoff
                nxt = self.next_digit[self.pusher]
                if nxt in pushing:
                    if self.pair - self.pair_at_handoff >= RELAY_MIN:
                        relay, self.relays = RELAY, self.relays + 1
                    self.pusher, self.pair_at_handoff = nxt, self.pair
                elif self.pusher not in pushing and self.next_digit[nxt] in pushing:
                    self.pusher, self.pair_at_handoff = self.next_digit[nxt], self.pair
        bowl_dev = float(np.mean((ctrl - self.base) ** 2))
        on_palm = set()
        for b in BALLS:
            if any(dg == "palm" and fn > 0.05 for dg, sg, f, fn in classify_contacts(self.model, self.data, b)):
                on_palm.add(b)
        ball_z = [float(q[2]) for q in balls_in_palm(self.model, self.data)]
        out_cm = sum(max(0.0, z - BALL_Z_MAX) * 100 for z in ball_z)
        turn_push = float(np.tanh(push[turn_digit] / PUSH_RATE)) if turn_digit in push else 0.0

        dropped = any(r > MAX_R or h > MAX_H for r, h in offsets)
        reward = (relay - W_BOWL * bowl_dev + W_TURN * turn_push + W_PAD * n_pad
                  + W_PALM * len(on_palm) - W_OUT * out_cm
                  + W_REACH * float(sum(np.exp(-max(g, 0.0) / REACH_SCALE) for g in reach.values()))
                  + W_GOOD * len(good) * moving - W_DORSAL * n_dorsal
                  + W_PUSH * float(sum(np.tanh(w / PUSH_RATE) for w in push.values()))
                  + W_FACE * sum(facing.values()) + W_ALL * (n_pad == len(DIGITS))
                  + W_TOUCH * turning * max(0.0, 1.0 - max(gap, 0.0) / TOUCH_FAR)
                  - W_RATE * float(np.sum((ctrl - prev_ctrl) ** 2)))
        if dropped:   # no credit for the push that threw a ball out
            reward -= DROP + DROP_PER_SECOND * (self.episode_steps - self.t) * CONTROL_DT
        else:
            reward += W_ORBIT * d_progress
            while self.progress >= self.next_milestone:
                reward += MILESTONE
                self.next_milestone += MILESTONE_RAD
        truncated = self.t >= self.episode_steps
        info = {"progress": self.progress, "arcs": self.arc.copy(), "pair": self.pair,
                "angles": self.tracker.total.copy(), "dropped": dropped,
                "good": sorted(good), "n_dorsal": n_dorsal, "push": push, "touching": touching, "gap": gap,
                "facing": facing, "n_pad": n_pad, "reach": reach, "relays": self.relays,
                "pusher": self.pusher, "bowl_dev": bowl_dev, "turn": turn_digit, "turn_push": turn_push,
                "on_palm": sorted(on_palm), "ball_z": ball_z}
        if dropped or truncated:
            info["success"] = not dropped
        return self._obs(), reward, dropped, truncated, info
