"""Stage 6 (rebuild, step 4): RL three-ball baoding as TRACKING (the MyoChallenge objective, Chiappa &
Mathis 2024): each ball follows its own target moving round a fixed circle in the palm, and the target
speed is the curriculum (SDS: static holds first, then slowly faster).

The planner demonstration (`data/stage6_demo_two_circles.npz`, 717 deg in 98 s, fingers only, from
`stage6_teacher.py --labeled --backtracks`) supplies the starts: a random moment of it, so every phase of
the circle -- the jam included -- is a start from the first episode (a reverse curriculum), and the
targets start where that moment's trio is. It also supplies behaviour cloning targets
(`train_stage6_track.py --bc`).

The user on the demo's GIF: "ring finger and middle finger sometime use tip of the finger to keep baoding
trick going, not very elegant". Measured over the demo (frames with a contact of >= 0.2 N): ring tip
22%, middle tip 6%, little tip 54%, index and thumb ~0. So every finger (not the thumb) pressing a ball
with its tip -- `hand_env.classify_contacts` face "tip", the rounded end cap -- costs W_TIP, and the
back of a segment W_DORSAL.

Reward per control step:  mean over balls of exp(-error / TRACK_SCALE)   (error: cm to its target)
                          - W_TIP * fingers on a ball by the tip - W_DORSAL * dorsal contacts
                          - W_RATE * |action|^2;   a lost ball ends the episode with -DROP.
Curriculum (from 12M): see OK_ERR/OK_TURN and EP_DEGREES; the targets circle the demo trio's mean centroid.
The wrist is held at the bowl posture (the demo did not need it); the fingers have their full range.
"""
from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from eval_stage6_circle_hold import WRIST, Probe
from hand_env import DIGITS, classify_contacts
from three_balls import BALLS3

HERE = Path(__file__).parent
SLICK = str(HERE / "assets" / "scene_three_balls_slick.xml")
DEMO = HERE / "data" / "stage6_demo_two_circles.npz"
CONTROL_DT = 0.02
N_HAND_Q = 24
RATE = 0.12                        # rad per control step at |action| = 1 (the demo's p99 is 0.11)
RATES_DEG = (0.0, 3.0, 6.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0)   # curriculum: target speed, deg/s
MIN_SECONDS, MAX_SECONDS, EP_DEGREES = 8.0, 20.0, 60.0    # target travel per episode (was 150 deg, up to 30 s)
OK_ERR, OK_TURN = 2.0, 0.5         # level passed: err < 2cm and the balls turned >= half the targets' travel
AHEAD = 0.5                        # s: the observation also shows the targets this far ahead
TRACK_SCALE = 1.5                  # cm (MyoChallenge: within 1.5cm)
W_TIP = 0.15
W_DORSAL = 0.1
W_RATE = 0.005
DROP = 20.0
TIP_FORCE = 0.2                    # N
LOST_Y, LOST_SPREAD = 0.04, 0.045  # m: off the palm / from the trio's centroid (the planner's limits)
REST_Y = -0.034


def demo_phase(balls, center):
    """(continuous trio phase per logged step, slot of each ball): ball k sits at phase + slot_k * 120."""
    th = np.arctan2(balls[:, :, 0] - center[0], balls[:, :, 2] - center[1])
    ph0 = np.angle(np.mean(np.exp(3j * th[0]))) / 3
    slots = [int(np.argmin([abs(np.angle(np.exp(1j * (th[0, k] - ph0 - s * 2 * np.pi / 3)))) for s in range(3)]))
             for k in range(3)]
    raw = np.angle(np.mean(np.exp(1j * (th - np.array(slots) * 2 * np.pi / 3)), axis=1))
    return np.unwrap(raw), slots


class TrackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sign=1.0, level=0, randomize=True, demo=DEMO, scene=SLICK, episode_seconds=None):
        super().__init__()
        self.pr = Probe(0.0225, scene)
        self.m, self.d = self.pr.m, self.pr.d
        self.n_sub = int(round(CONTROL_DT / self.m.opt.timestep))
        self.lo, self.hi = self.pr.lo.copy(), self.pr.hi.copy()
        self.lo[:WRIST] = self.hi[:WRIST] = self.pr.target0[:WRIST]
        self.sign, self.level, self.randomize, self.fixed_seconds = float(sign), int(level), randomize, episode_seconds
        z = np.load(demo)
        self.demo = {k: z[k] for k in ("qpos", "qvel", "ctrl", "balls")}
        # the circle the demo's trio actually went round (its centroid, palm x/z): the planner's centre
        # (0.1, 6.5) sat 0.5cm behind it, which left the demo itself 1.2cm from its own-phase targets
        self.center = z["balls"].mean(1).mean(0)[[0, 2]]
        self.demo_phase, self.slots = demo_phase(z["balls"], self.center)
        self.ball_dof = [self.m.jnt_dofadr[self.m.body(b).jntadr[0]] for b in BALLS3]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.m.nu,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2 * N_HAND_Q + self.m.nu + 9 * 4 + 3,), dtype=np.float32)

    # ---- curriculum
    @property
    def rate(self):
        return np.radians(RATES_DEG[self.level])

    def set_level(self, level):
        self.level = int(min(max(level, 0), len(RATES_DEG) - 1))
        return self.level

    def seconds(self):
        if self.fixed_seconds:
            return self.fixed_seconds
        r = RATES_DEG[self.level]
        return MIN_SECONDS if r == 0 else float(np.clip(EP_DEGREES / r, MIN_SECONDS, MAX_SECONDS))

    # ---- geometry
    def balls(self):
        return np.array(self.pr.palm())

    def targets(self, phase):
        tg = self.pr.places(self.center, phase)
        return np.array([tg[s] for s in self.slots])

    def phase_at(self, t):
        return self.phase0 + self.sign * self.rate * t

    # LOST_HOLD > 1: the spread limit must be exceeded that many control steps in a row (the pad demo itself
    # bulges 0.5mm past it for 2 steps at demo step 1755, mid-click); off the palm, or spread > LOST_SPREAD_HARD,
    # is still lost at once. Default 1 = the original rule.
    LOST_HOLD = 1
    LOST_SPREAD_HARD = 0.06

    def lost(self, b):
        spread = float(np.max(np.linalg.norm(b - b.mean(0), axis=1)))
        self.n_spread = getattr(self, "n_spread", 0) + 1 if spread > LOST_SPREAD else 0
        return bool(np.any(np.abs(b[:, 1] - REST_Y) > LOST_Y) or spread > self.LOST_SPREAD_HARD
                    or self.n_spread >= self.LOST_HOLD)

    def contacts(self):
        """(fingers pressing a ball with the tip, dorsal contacts)."""
        tips, dorsal = set(), 0
        for b in BALLS3:
            for dg, sg, face, fn in classify_contacts(self.m, self.d, b):
                if fn < TIP_FORCE or (dg in ("palm", "th") and face == "tip"):
                    continue
                if face == "tip":
                    tips.add(dg)
                elif face == "DORSAL":
                    dorsal += 1
        return tips, dorsal

    def obs_for(self, tg, ta, phase, rate):
        b = self.balls()
        R = self.d.body("rh_palm").xmat.reshape(3, 3)
        c3 = np.array([self.center[0], REST_Y, self.center[1]])
        v = np.concatenate([R.T @ self.d.qvel[a:a + 3] for a in self.ball_dof])
        return np.concatenate([
            self.d.qpos[:N_HAND_Q], 0.1 * self.d.qvel[:N_HAND_Q], self.d.ctrl,
            10 * (b - c3).ravel(), v, 10 * (tg - c3).ravel(), 10 * (ta - b).ravel(),
            [np.cos(3 * phase), np.sin(3 * phase), rate],
        ]).astype(np.float32)

    def target_pos(self, t):
        return self.targets(self.phase_at(t))

    def _obs(self):
        t = self.t * CONTROL_DT
        return self.obs_for(self.target_pos(t), self.target_pos(t + AHEAD), self.phase_at(t), self.sign * self.rate)

    def level_ok(self, lost, truncated, mean_err):
        return bool(not lost and truncated and mean_err < OK_ERR and self.turned >= OK_TURN * self.rate * self.t * CONTROL_DT)

    # ---- episode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        n = len(self.demo["qpos"])
        i = int(self.np_random.integers(n - 1)) if self.randomize else 0
        if options and "start" in options:
            i = int(options["start"])
        mj.mj_resetData(self.m, self.d)
        self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = self.demo["qpos"][i], self.demo["qvel"][i], self.demo["ctrl"][i]
        mj.mj_forward(self.m, self.d)
        self.phase0, self.start = float(self.demo_phase[i]), i
        self.i0 = i
        self.t, self.steps = 0, int(round(self.seconds() / CONTROL_DT))
        self.err_sum, self.n_tip, self.th_prev, self.turned = 0.0, 0, self._angles(), 0.0
        self.n_spread = 0
        return self._obs(), {}

    def _angles(self):
        b = self.balls()
        return np.arctan2(b[:, 0] - self.center[0], b[:, 2] - self.center[1])

    def step(self, action):
        a = np.clip(action, -1, 1)
        self.d.ctrl[:] = np.clip(self.d.ctrl + a * RATE, self.lo, self.hi)
        for _ in range(self.n_sub):
            mj.mj_step(self.m, self.d)
        self.t += 1
        b = self.balls()
        th = self._angles()
        self.turned += self.sign * float(np.mean(np.angle(np.exp(1j * (th - self.th_prev)))))
        self.th_prev = th
        e = np.linalg.norm(b - self.target_pos(self.t * CONTROL_DT), axis=1) * 100
        tips, dorsal = self.contacts()
        lost = self.lost(b)
        reward = (float(np.mean(np.exp(-e / TRACK_SCALE))) - W_TIP * len(tips) - W_DORSAL * dorsal
                  - W_RATE * float(np.sum(a[WRIST:] ** 2)))
        self.err_sum += float(e.mean())
        self.n_tip += len(tips)
        if lost:
            reward -= DROP
        truncated = self.t >= self.steps
        info = {"err": float(e.mean()), "tips": sorted(tips), "dorsal": dorsal}
        if lost or truncated:
            mean_err = self.err_sum / self.t
            info.update(lost=lost, mean_err=mean_err, turned=self.turned, level=self.level, tip_rate=self.n_tip / self.t,
                        level_ok=self.level_ok(lost, truncated, mean_err))
        return self._obs(), reward, lost, truncated and not lost, info

    # ---- behaviour cloning pairs from the demo
    def demo_pairs(self):
        """(obs, action) for every demo step: the targets are where the trio itself is AHEAD s later (the
        demo moves in clicks; this is its own pace), the action the demo's control change / RATE."""
        z = self.demo
        n, k = len(z["qpos"]), int(round(AHEAD / CONTROL_DT))
        obs, act = [], []
        for i in range(n - 1):
            self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = z["qpos"][i], z["qvel"][i], z["ctrl"][i]
            mj.mj_forward(self.m, self.d)
            ph, pa = self.demo_phase[min(i + k, n - 1)], self.demo_phase[min(i + 2 * k, n - 1)]
            obs.append(self.obs_for(self.targets(ph), self.targets(pa), ph, (pa - ph) / AHEAD))
            act.append(np.clip((z["ctrl"][i + 1] - z["ctrl"][i]) / RATE, -1, 1))
        return np.array(obs), np.array(act, dtype=np.float32)


# ---- following the demo's own timing (runs track_01 from 25M on)
# Circle targets at a constant speed stalled the curriculum at 3 deg/s for 13M steps: the trio passes the
# jam in a quick hop ("going uphill and then they fall in place"), which a smoothly moving target asks it to
# creep through. The user chose: "switch to following the demo timing". Each ball now tracks where IT was
# in the demo at the matching moment (DeepMimic-style motion imitation, starts at random demo moments), and
# the curriculum plays the demo back faster: SPEEDS x the demo's own pace (7.3 deg/s average). Past the
# demo's end it loops, the trio rotated by the demo's net turn minus two circles (-2 deg; same labels).
SPEEDS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
DEMO_EP_SECONDS = 10.0
DEMO_OK_ERR = 1.5                  # cm: MyoChallenge's "within 1.5cm"
DEMO_OK_TURN = 0.5                 # and the balls turned >= half what the demo turned in the window
START_FLOOR_DEG = 10.0


class DemoTrackEnv(TrackEnv):

    def __init__(self, *args, hard_frac=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.hard_frac = float(hard_frac)
        # starts weighted by how far the demo turns over the next episode (at 1x): 42% of its 10 s windows
        # turn < 20 deg (the planner's stalls before backtracking), so uniform starts were mostly holding
        n, k = len(self.demo_phase), int(round(DEMO_EP_SECONDS / CONTROL_DT))
        ahead = self.demo_phase[np.minimum(np.arange(n - 1) + k, n - 1)] - self.demo_phase[:-1]
        w = np.maximum(ahead, 0.0) + np.radians(START_FLOOR_DEG)
        self.start_p = w / w.sum()
        self.loop_turn = float(self.demo_phase[-1] - self.demo_phase[0] - 4 * np.pi)
        self.demo_rate = float((self.demo_phase[-1] - self.demo_phase[0]) / ((len(self.demo_phase) - 1) * CONTROL_DT))

    @property
    def speed(self):
        return SPEEDS[self.level]

    @property
    def rate(self):
        return self.speed * self.demo_rate

    def set_level(self, level):
        self.level = int(min(max(level, 0), len(SPEEDS) - 1))
        return self.level

    def seconds(self):
        return self.fixed_seconds or DEMO_EP_SECONDS

    def _demo_at(self, t):
        """(ball positions, trio phase) of the demo `t` s of env time after this episode's start moment."""
        n = len(self.demo_phase) - 1
        k, f = divmod(self.i0 + self.speed * t / CONTROL_DT, n)
        j = int(f)
        a = f - j
        p = (1 - a) * self.demo["balls"][j] + a * self.demo["balls"][j + 1]
        ph = (1 - a) * self.demo_phase[j] + a * self.demo_phase[j + 1] + k * (self.loop_turn + 4 * np.pi)
        if k:
            d = p[:, [0, 2]] - self.center
            th, r = np.arctan2(d[:, 0], d[:, 1]) + k * self.loop_turn, np.linalg.norm(d, axis=1)
            p = p.copy()
            p[:, 0], p[:, 2] = self.center[0] + r * np.sin(th), self.center[1] + r * np.cos(th)
        return p, ph

    def target_pos(self, t):
        return self._demo_at(t)[0]

    def _obs(self):
        t = self.t * CONTROL_DT
        return self.obs_for(self.target_pos(t), self.target_pos(t + AHEAD), self._demo_at(t)[1], self.rate)

    def demo_pairs(self):
        """(obs, action) per demo step at 1x: the targets are the demo's own ball positions now and AHEAD s on."""
        z, k = self.demo, int(round(AHEAD / CONTROL_DT))
        n = len(z["qpos"])
        obs, act = [], []
        for i in range(n - 1):
            self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = z["qpos"][i], z["qvel"][i], z["ctrl"][i]
            mj.mj_forward(self.m, self.d)
            obs.append(self.obs_for(z["balls"][i], z["balls"][min(i + k, n - 1)], self.demo_phase[i], self.demo_rate))
            act.append(np.clip((z["ctrl"][i + 1] - z["ctrl"][i]) / RATE, -1, 1))
        return np.array(obs), np.array(act, dtype=np.float32)

    def level_ok(self, lost, truncated, mean_err):
        want = self._demo_at(self.t * CONTROL_DT)[1] - self._demo_at(0.0)[1]
        return bool(not lost and truncated and mean_err < DEMO_OK_ERR and self.turned >= DEMO_OK_TURN * want)

    # Failure-focused starts (`hard_frac` of resets, training only): each env remembers where its balls were
    # lost -- the demo moment (restart from the demo's own state HARD_BACK s of env time earlier) and the
    # policy's own state HARD_BACK s before the drop (restart there: practise recovering from that slip).
    HARD_BACK = (1.0, 3.0)
    HARD_KEEP = 100
    SNAP_EVERY = 25                # control steps between snapshots (0.5 s)

    def reset(self, seed=None, options=None):
        if not hasattr(self, "hard_moments"):
            self.hard_moments, self.hard_states, self.snaps = deque(maxlen=self.HARD_KEEP), deque(maxlen=self.HARD_KEEP), deque(maxlen=8)
            self.n_hard = 0
        self.snaps.clear()
        self.hard_start = False
        if self.randomize and not (options and "start" in options):
            gym.Env.reset(self, seed=seed)
            n = len(self.demo_phase) - 1
            if self.hard_moments and self.np_random.random() < self.hard_frac:
                self.hard_start = True
                if self.hard_states and self.np_random.random() < 0.5:
                    obs, info = super().reset(seed=seed, options={"start": 0})
                    qpos, qvel, ctrl, i0 = self.hard_states[int(self.np_random.integers(len(self.hard_states)))]
                    self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = qpos, qvel, ctrl
                    mj.mj_forward(self.m, self.d)
                    self.i0 = i0
                    self.th_prev = self._angles()
                    return self._obs(), info
                m = self.hard_moments[int(self.np_random.integers(len(self.hard_moments)))]
                back = self.np_random.uniform(*self.HARD_BACK) * self.speed / CONTROL_DT
                options = dict(options or {}, start=int(round(m - back)) % n)
            else:
                options = dict(options or {}, start=int(self.np_random.choice(len(self.start_p), p=self.start_p)))
        return super().reset(seed=seed, options=options)

    def step(self, action):
        if self.hard_frac and self.t % self.SNAP_EVERY == 0:
            self.snaps.append((self.d.qpos.copy(), self.d.qvel.copy(), self.d.ctrl.copy(),
                               self.i0 + self.speed * self.t))
        obs, reward, term, trunc, info = super().step(action)
        if term and self.hard_frac:
            n = len(self.demo_phase) - 1
            self.hard_moments.append((self.i0 + self.speed * self.t) % n)
            k = int(round(np.mean(self.HARD_BACK) / (self.SNAP_EVERY * CONTROL_DT)))
            if len(self.snaps) > k:
                self.hard_states.append(self.snaps[-1 - k])
        if term or trunc:
            self.n_hard += self.hard_start
            info["hard_start"] = float(self.hard_start)
        return obs, reward, term, trunc, info
