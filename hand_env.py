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


# Stage 3: two balls, touching, cradled palm-up -- the starting grasp for baoding.
TWO_BALL_SCENE = "assets/scene_two_balls.xml"
BALLS = ("ball1", "ball2")


def two_ball_places(center, angle, radius=BALL_RADIUS):
    """Centers of two touching balls whose midpoint is `center` (world xyz) and whose line makes
    `angle` (rad) with the fingers' direction (+x) in the palm plane: 0 = one behind the other along
    the palm, pi/2 = side by side across it."""
    d = radius * np.array([np.cos(angle), np.sin(angle), 0.0])
    return np.array(center) - d, np.array(center) + d


def set_ball_radius(model, radius):
    """Resize both balls of the two-ball scene to `radius` (m), mass and inertia scaled with
    volume (a smaller ball of the same material). Stage 4: two 45mm balls have no room to pass
    each other between the knuckle row and the thumb's base segment, even under an ideal push."""
    for b in BALLS:
        g, bid = model.geom(b).id, model.body(b).id
        k = radius / model.geom_size[g, 0]
        model.geom_size[g, 0] = model.geom_rbound[g] = radius
        model.body_mass[bid] *= k ** 3
        model.body_inertia[bid] *= k ** 5


def grasp_balls(model, data, target, places, close_time=0.6, settle_time=0.3):
    """Two-ball version of `grasp_ball(pin=True)`: both balls held at `places` while the hand ramps
    from rest to `target` over `close_time` s, then released and left `settle_time` s."""
    mj.mj_resetData(model, data)
    model.opt.gravity[:] = (0, 0, -9.81)
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    rest, target = np.clip(0.0, lo, hi), np.clip(target, lo, hi)
    adr = [(model.jnt_qposadr[model.body(b).jntadr[0]], model.jnt_dofadr[model.body(b).jntadr[0]])
           for b in BALLS]
    data.ctrl[:] = rest
    n = int(close_time / model.opt.timestep)
    for k in range(n):
        data.ctrl[:] = rest + (target - rest) * (k + 1) / n
        for (q, v), p in zip(adr, places):
            data.qpos[q:q + 3], data.qpos[q + 3:q + 7], data.qvel[v:v + 6] = p, (1, 0, 0, 0), 0.0
        mj.mj_step(model, data)
    for _ in range(int(settle_time / model.opt.timestep)):
        mj.mj_step(model, data)


def balls_in_palm(model, data):
    """Each ball's center in the palm body's frame."""
    R = data.body("rh_palm").xmat.reshape(3, 3)
    return [R.T @ (data.body(b).xpos - data.body("rh_palm").xpos) for b in BALLS]


# The Stage 3 bowl, reviewed by the user as an image (media/stage3_bowl_e.png): all five digits
# standing up off the palm with their tips at one height (8.0-8.9cm) and curled a little in, the two
# touching balls resting on the palm inside (gold on the thumb side, red on the little-finger side),
# the wrist tilting the palm ~25 deg toward the fingers so the balls settle into the cup instead of
# rolling out of its open wrist side. Light: ~1.4 N total finger force, the balls free to roll.
# From `search_stage3_bowl.py --seed 3 --init <previous bowl>`; held both balls, touching, on the
# palm in all 6 runs (nominal + 5 perturbed).
BOWL_GRASP = {
    "rh_A_WRJ2": 0.0019, "rh_A_WRJ1": -0.4441,
    "rh_A_THJ5": -0.1528, "rh_A_THJ4": 1.2217, "rh_A_THJ3": 0.142, "rh_A_THJ2": 0.061,
    "rh_A_THJ1": 1.0921,
    "rh_A_FFJ4": -0.1527, "rh_A_FFJ3": 1.147, "rh_A_FFJ0": 0.8176,
    "rh_A_MFJ4": -0.2669, "rh_A_MFJ3": 1.128, "rh_A_MFJ0": 0.2817,
    "rh_A_RFJ4": 0.1308, "rh_A_RFJ3": 1.4069, "rh_A_RFJ0": 0.2109,
    "rh_A_LFJ5": 0.0423, "rh_A_LFJ4": -0.0376, "rh_A_LFJ3": 1.5419, "rh_A_LFJ0": 0.8765,
}
BOWL_BALLS_CENTER = np.array([0.3116, -0.0011, 0.0445])
BOWL_BALLS_ANGLE = 2.4911


def bowl_target():
    t = np.zeros(N_ACTUATORS)
    for name, v in BOWL_GRASP.items():
        t[ACTUATOR_NAMES.index(name)] = v
    return t


# Stage 4: scripted baoding from the bowl. Every digit takes part -- the user: "not only little
# finger and thumb, all fingers should involved as needed to make baoding trick".
# Each digit presses with two joints: a finger leans in at its base (J3) and curls (J0, the coupled
# middle+distal); the thumb swings across at its base (THJ5) and flexes its middle joint (THJ2).
PRESS_JOINTS = {
    "ff": ("rh_A_FFJ3", "rh_A_FFJ0"), "mf": ("rh_A_MFJ3", "rh_A_MFJ0"),
    "rf": ("rh_A_RFJ3", "rh_A_RFJ0"), "lf": ("rh_A_LFJ3", "rh_A_LFJ0"),
    "th": ("rh_A_THJ5", "rh_A_THJ2"),
}
PRESS_DIGITS = ("ff", "mf", "rf", "lf", "th")


class BaodingWave:
    """A travelling press wave around the bowl's rim. Every digit's two press joints are offset
    from the bowl grasp by amp * s_i(t), s_i(t) = 0.5 * (1 + sin(2 pi f t - phase_i)) in [0, 1]: each
    digit presses in and relaxes once per cycle, and the phases order the presses around the rim, so
    the push travels round the bowl and drags the balls with it. The wrist can add a small circular
    tilt (WRJ1 sin, WRJ2 cos, same frequency) -- the cup itself circling, as a human hand does.
    Ramps in over the first `ramp` s so the balls start from the settled bowl.

    params: freq (Hz), phase[5], amp1[5], amp2[5] (rad, signed; per PRESS_DIGITS / PRESS_JOINTS),
    wrist_amp (WRJ1, WRJ2) (rad), wrist_phase."""

    N_PARAMS = 1 + 5 * 3 + 3

    def __init__(self, params, ramp=0.5, base=None):
        p = np.asarray(params, float)
        self.freq = p[0]
        self.phase, self.amp1, self.amp2 = p[1:6], p[6:11], p[11:16]
        self.wrist_amp, self.wrist_phase = p[16:18], p[18]
        self.ramp = ramp
        self.base = bowl_target() if base is None else np.array(base, float)
        self.idx = [(ACTUATOR_NAMES.index(a), ACTUATOR_NAMES.index(b))
                    for a, b in (PRESS_JOINTS[d] for d in PRESS_DIGITS)]
        self.wrist = (ACTUATOR_NAMES.index("rh_A_WRJ1"), ACTUATOR_NAMES.index("rh_A_WRJ2"))

    def ctrl(self, t):
        k = min(1.0, t / self.ramp)
        w = 2 * np.pi * self.freq * t
        u = self.base.copy()
        s = 0.5 * (1 + np.sin(w - self.phase))
        for (i, j), a1, a2, si in zip(self.idx, self.amp1, self.amp2, s):
            u[i] += k * a1 * si
            u[j] += k * a2 * si
        u[self.wrist[0]] += k * self.wrist_amp[0] * np.sin(w - self.wrist_phase)
        u[self.wrist[1]] += k * self.wrist_amp[1] * np.cos(w - self.wrist_phase)
        return u


class BaodingCircle(BaodingWave):
    """Each fingertip traces a small LOOP instead of pressing along a line -- the user, on the press
    wave's GIF (balls only rocked): "fingers maybe should make circular move to push balls cross
    each other". A digit's two press joints get offsets a1 sin(w t - phase_i) and
    a2 sin(w t - phase_i - lag_i): with lag ~ +-pi/2 the tip goes round an ellipse (one way or the
    other), pushing the ball along while touching it and coming back clear of it; lag 0 or pi
    collapses it to the old in-out press. Plus each finger's spread (J4) a3 sin(w t - phase_i - lag_i/2)
    -- small, it twists an upright finger, letting its pad roll the ball sideways.

    params: freq (Hz), phase[5], amp1[5], amp2[5], lag[5], amp3[4] (ff/mf/rf/lf J4), wrist_amp
    (WRJ1, WRJ2), wrist_phase."""

    N_PARAMS = 1 + 5 * 4 + 4 + 3
    SPREAD = ("rh_A_FFJ4", "rh_A_MFJ4", "rh_A_RFJ4", "rh_A_LFJ4")

    def __init__(self, params, ramp=0.5, base=None):
        p = np.asarray(params, float)
        super().__init__(np.concatenate([p[:16], p[25:28]]), ramp, base)
        self.lag, self.amp3 = p[16:21], p[21:25]
        self.spread = [ACTUATOR_NAMES.index(a) for a in self.SPREAD]

    def ctrl(self, t):
        k = min(1.0, t / self.ramp)
        w = 2 * np.pi * self.freq * t
        u = self.base.copy()
        for n, ((i, j), a1, a2, ph, lag) in enumerate(zip(self.idx, self.amp1, self.amp2, self.phase, self.lag)):
            u[i] += k * a1 * np.sin(w - ph)
            u[j] += k * a2 * np.sin(w - ph - lag)
            if n < 4:
                u[self.spread[n]] += k * self.amp3[n] * np.sin(w - ph - 0.5 * lag)
        u[self.wrist[0]] += k * self.wrist_amp[0] * np.sin(w - self.wrist_phase)
        u[self.wrist[1]] += k * self.wrist_amp[1] * np.cos(w - self.wrist_phase)
        return u


class BallAngles:
    """Each ball's OWN cumulative angle (unwrapped, rad) around a fixed point in the palm plane --
    the midpoint of the pair at the start -- measured in the palm frame so the wrist moving the hand
    doesn't count. Lesson from the Allegro: the angle of the vector BETWEEN the balls also turns when
    one ball just wobbles; each ball's own angle only grows if that ball actually travels round."""

    def __init__(self, model, data):
        p = balls_in_palm(model, data)
        self.center = 0.5 * (p[0] + p[1])
        self.prev = [self._angle(q) for q in p]
        self.total = np.zeros(2)

    def _angle(self, q):
        d = q - self.center
        return float(np.arctan2(d[2], d[0]))   # palm frame: x across, z along toward the fingers

    def update(self, model, data):
        for k, q in enumerate(balls_in_palm(model, data)):
            a = self._angle(q)
            self.total[k] += (a - self.prev[k] + np.pi) % (2 * np.pi) - np.pi
            self.prev[k] = a
        return self.total

    def offsets(self, model, data):
        """Each ball's distance from the center (m), in the palm plane and along the palm normal."""
        out = []
        for q in balls_in_palm(model, data):
            d = q - self.center
            out.append((float(np.hypot(d[0], d[2])), float(-d[1])))
        return out


def run_baoding(model, data, controller, seconds, on_step=None, tracker_out=None, bowl=None):
    """Bowl grasp, then `controller.ctrl(t)` for `seconds`. Returns (angles[2] rad, dropped,
    digits that touched a ball with >= 0.2 N, t_end). A ball counts as dropped once it is more than
    6cm from the start center in the palm plane or 5cm off the palm; the run stops there.
    `on_step(k, t)` is called after every physics step; if `tracker_out` is a list, the BallAngles
    tracker is appended to it (so on_step can read the running angles). `bowl` = (target, pair center,
    pair angle) of another bowl; default the frozen Stage 3 one."""
    target, center, angle = bowl if bowl is not None else (bowl_target(), BOWL_BALLS_CENTER, BOWL_BALLS_ANGLE)
    grasp_balls(model, data, target, two_ball_places(center, angle))
    tracker = BallAngles(model, data)
    if tracker_out is not None:
        tracker_out.append(tracker)
    touched = set()
    dt = model.opt.timestep
    n = int(seconds / dt)
    for k in range(n):
        t = k * dt
        data.ctrl[:] = controller.ctrl(t)
        mj.mj_step(model, data)
        if k % 10 == 0:
            tracker.update(model, data)
            for b in BALLS:
                touched |= {c[0] for c in classify_contacts(model, data, b)
                            if c[3] >= 0.2 and c[0] != "palm"}
            if any(r > 0.06 or h > 0.05 + BALL_RADIUS for r, h in tracker.offsets(model, data)):
                return tracker.total.copy(), True, touched, t
        if on_step is not None:
            on_step(k, t)
    return tracker.total.copy(), False, touched, seconds


def push_forces(model, data, center, force):
    """Stage 4 diagnostic, the ideal push: a tangential world force (N) on each ball around `center`
    (palm frame), in the palm plane, to apply with `data.xfrc_applied`. If the bowl holds still and
    even this cannot carry the balls round each other, no finger motion inside that bowl will."""
    R = data.body("rh_palm").xmat.reshape(3, 3)
    out = []
    for q in balls_in_palm(model, data):
        d = q - center
        t = np.array([-d[2], 0.0, d[0]])
        out.append(force * R @ (t / max(np.linalg.norm(t), 1e-6)))
    return out


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
