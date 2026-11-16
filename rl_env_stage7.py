"""Stage 7A, step 3: RL two-hand four-ball spinning as demo TRACKING (the Stage 6 recipe, `rl_env_track3.
DemoTrackEnv`, on the two-hand scene).

The planner demo (`data/stage7_demo_smooth360.npz`, one full counter-clockwise turn of the packed 2x2 square
in 106 s, from `stage7_teacher.py` with curled fingers held within 25-35% of their range of it, wrists free
+-15 deg and backtracking; the older, jerkier `data/stage7_demo_ccw360.npz` used +-10 deg) supplies
each ball's target -- where that ball was in the demo at the matching moment (DeepMimic-style), played
back SPEEDS x faster by the curriculum -- the starts (random demo moments, weighted by how far the demo
turns next), and the behaviour cloning pairs. Past the demo's end it loops: one full turn brings every
ball back to its own corner (the few degrees' overshoot is rotated in).

Reward per control step:  mean over balls of exp(-error / TRACK_SCALE)   (error: cm to its target)
                          + w_prog * clip(the balls' turn this step / the targets' nominal turn per step, -1, 1)
                          - W_RATE * |action|^2;   a lost ball ends the episode with -DROP.

A waiting target (`lead_deg`, off by default): the targets' demo clock runs only while they lead the balls'
own turn by less than `lead_deg`, so a policy stuck at a hard spot keeps a target just ahead of it (with
the target running away, the error grows to 4-5cm and exp(-error) leaves nothing pulling it on -- runs
two_01-03 stalled so). `w_prog` (off by default) pays for turning on round the square directly. The level
test still asks for half of the demo's NOMINAL turn, so waiting does not pass levels (neither variant helped:
two_05/06 idled or gambled).

Stall-focused starts (`stall_starts`, with `hard_frac`): a stall -- the balls turning < STALL_DEG in STALL_S s --
is remembered as a hard spot like a loss, at the demo moment where the demo's turn matches the BALLS' (not
the run-away target's), so the restart puts the target back beside the balls just before the hand-off.
(two_07: no better -- it stalled more.)

Planner fixes (`fixes`, `fix_frac`): `stage7_stalls.py` saves states where a policy stalls and
`stage7_teacher.py --resume` plans on from each (a "fix"). A `fix_frac` share of training resets starts from
a fix's first state and tracks the fix's balls, then the demo on from the matching turn; `fix_pairs()` gives
behaviour cloning pairs from the fixes as `demo_pairs()` does from the demo (DAgger-style, the planner as the
expert where the policy fails).
Lost (the planner's rule): a ball below the palm surface or more than FAR from the square's centre.
Actions: 40 actuator target changes (both hands; the wrists within +-WRIST_TILT of straight, as the demo).
"""
from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from stage7_layout import PALM_Z
from stage7_teacher import BALLS, FAR, World

HERE = Path(__file__).parent
DEMO = HERE / "data" / "stage7_demo_smooth360.npz"
CURL = {f"rh_A_{f}J{j}": v for f in ("FF", "MF", "RF", "LF") for j, v in ((3, 0.7), (0, 1.0))}
WRIST_TILT = np.radians(15.0)
CONTROL_DT = 0.02
RATE = 0.12                        # rad per control step at |action| = 1 (the demo's p99 is 0.04, max 0.13)
SPEEDS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
EP_SECONDS = 10.0
AHEAD = 0.5                        # s: the observation also shows the targets this far ahead
TRACK_SCALE = 1.5                  # cm
OK_ERR, OK_TURN = 1.5, 0.5         # level passed: no loss, mean error < 1.5cm, turned >= half the demo's turn
START_FLOOR_DEG = 10.0
W_RATE = 0.005
DROP = 20.0


def turn_of(balls, center):
    """Cumulative turn (rad) of the balls round `center` per logged step: the mean of each ball's step."""
    th = np.arctan2(balls[:, :, 1] - center[1], balls[:, :, 0] - center[0])
    return np.concatenate([[0.0], np.cumsum(np.mean(np.angle(np.exp(1j * np.diff(th, axis=0))), axis=1))])


class TwoHandTrackEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level=0, randomize=True, demo=DEMO, episode_seconds=None, hard_frac=0.0,
                 lead_deg=None, w_prog=0.0, stall_starts=False, fixes=None, fix_frac=0.0):
        super().__init__()
        self.stall_starts = bool(stall_starts)
        self.fix_frac = float(fix_frac)
        self.lead = None if lead_deg is None else np.radians(lead_deg)
        self.w_prog = float(w_prog)
        self.w = World(posture=CURL, tilt=WRIST_TILT)
        self.m, self.d = self.w.m, self.w.d
        self.n_sub = int(round(CONTROL_DT / self.m.opt.timestep))
        self.lo, self.hi = self.w.lo, self.w.hi
        self.level, self.randomize, self.fixed_seconds = int(level), randomize, episode_seconds
        self.hard_frac = float(hard_frac)
        z = np.load(demo)
        self.demo = {k: z[k] for k in ("qpos", "qvel", "ctrl", "balls")}
        self.center = np.array(z["center"])
        self.c3 = np.array([*self.center, z["balls"][:, :, 2].mean()])
        self.phase = turn_of(z["balls"], self.center)
        self.loop_turn = float(self.phase[-1] - 2 * np.pi)
        self.demo_rate = float(self.phase[-1] / ((len(self.phase) - 1) * CONTROL_DT))
        n, k = len(self.phase), int(round(EP_SECONDS / CONTROL_DT))
        ahead = self.phase[np.minimum(np.arange(n - 1) + k, n - 1)] - self.phase[:-1]
        wgt = np.maximum(ahead, 0.0) + np.radians(START_FLOOR_DEG)
        self.start_p = wgt / wgt.sum()
        self.mono = np.maximum.accumulate(self.phase)
        self.clip = None
        self.fixes = load_fixes(fixes, self) if fixes else []
        hinge = [j for j in range(self.m.njnt) if self.m.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE]
        self.hq = np.array([self.m.jnt_qposadr[j] for j in hinge])
        self.hv = np.array([self.m.jnt_dofadr[j] for j in hinge])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.m.nu,), dtype=np.float32)
        nobs = 2 * len(self.hq) + self.m.nu + 4 * len(BALLS) * 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(nobs,), dtype=np.float32)

    # ---- curriculum
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
        return self.fixed_seconds or EP_SECONDS

    # ---- the demo
    def _demo_at(self, t):
        """(ball positions, phase) of the demo `t` s of env time after this episode's start moment."""
        if self.clip is not None:                       # a fix episode: the fix, then the demo on from it
            f = min(self.i0 + self.speed * t / CONTROL_DT, len(self.clip["phase"]) - 1.001)
            j, a = int(f), f - int(f)
            c = self.clip
            return (1 - a) * c["balls"][j] + a * c["balls"][j + 1], (1 - a) * c["phase"][j] + a * c["phase"][j + 1]
        n = len(self.phase) - 1
        k, f = divmod(self.i0 + self.speed * t / CONTROL_DT, n)
        j = int(f)
        a = f - j
        p = (1 - a) * self.demo["balls"][j] + a * self.demo["balls"][j + 1]
        ph = (1 - a) * self.phase[j] + a * self.phase[j + 1] + k * self.phase[-1]
        if k:
            d = p[:, :2] - self.center
            th, r = np.arctan2(d[:, 1], d[:, 0]) + k * self.loop_turn, np.linalg.norm(d, axis=1)
            p = p.copy()
            p[:, 0], p[:, 1] = self.center[0] + r * np.cos(th), self.center[1] + r * np.sin(th)
        return p, ph

    def target_pos(self, t):
        return self._demo_at(t)[0]

    # ---- observation
    def obs_for(self, tg, ta, phase, rate):
        b = self.w.balls()
        v = np.concatenate([self.d.qvel[a:a + 3] for a in self.w.bv])
        return np.concatenate([
            self.d.qpos[self.hq], 0.1 * self.d.qvel[self.hv], self.d.ctrl,
            10 * (b - self.c3).ravel(), v, 10 * (tg - self.c3).ravel(), 10 * (ta - b).ravel(),
            [np.cos(phase), np.sin(phase), rate],
        ]).astype(np.float32)

    def _obs(self):
        t = self.tau
        return self.obs_for(self.target_pos(t), self.target_pos(t + AHEAD), self._demo_at(t)[1], self.rate)

    def lost(self, b):
        return bool(np.any(b[:, 2] < PALM_Z) or np.any(np.linalg.norm(b[:, :2] - self.center, axis=1) > FAR))

    def _angles(self):
        b = self.w.balls()
        return np.arctan2(b[:, 1] - self.center[1], b[:, 0] - self.center[0])

    def level_ok(self, lost, truncated, mean_err):
        want = self._demo_at(self.t * CONTROL_DT)[1] - self._demo_at(0.0)[1]
        return bool(not lost and truncated and mean_err < OK_ERR and self.turned >= OK_TURN * want)

    # ---- episode
    # Failure-focused starts (`hard_frac` of resets, training only), as rl_env_track3.DemoTrackEnv: restart
    # from the demo HARD_BACK s of env time before a remembered loss, or from the policy's own state then.
    HARD_BACK = (1.0, 3.0)
    HARD_KEEP = 100
    SNAP_EVERY = 25
    STALL_S, STALL_DEG = 5.0, 3.0

    def _moment(self):
        """The demo step the targets are at (the waiting target's clock `tau`, s of env time)."""
        return self.i0 + self.speed * self.tau / CONTROL_DT

    def _aligned(self, turned):
        """The demo step whose turn matches the balls' after turning `turned` from this episode's start."""
        n = len(self.phase) - 1
        r = (self.phase[int(self.i0) % n] + turned) % self.phase[-1]
        return int(min(np.searchsorted(self.mono, r), n - 1))

    def _start(self, i):
        mj.mj_resetData(self.m, self.d)
        self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = self.demo["qpos"][i], self.demo["qvel"][i], self.demo["ctrl"][i]
        mj.mj_forward(self.m, self.d)
        self.i0 = i

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not hasattr(self, "hard_moments"):
            self.hard_moments, self.hard_states, self.snaps = deque(maxlen=self.HARD_KEEP), deque(maxlen=self.HARD_KEEP), deque(maxlen=16)
        self.snaps.clear()
        self.hard_start, self.stalled, self.fix_start, self.clip = False, False, False, None
        self.turn_hist = deque(maxlen=int(round(self.STALL_S / CONTROL_DT)) + 1)
        n = len(self.phase) - 1
        if options and "start" in options:
            self._start(int(options["start"]))
        elif not self.randomize:
            self._start(0)
        elif self.fixes and self.np_random.random() < self.fix_frac:
            self.hard_start = self.fix_start = True
            fx = self.fixes[int(self.np_random.integers(len(self.fixes)))]
            self._start(0)
            self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = fx["qpos"], fx["qvel"], fx["ctrl"]
            mj.mj_forward(self.m, self.d)
            self.clip, self.i0 = fx, 0
        elif self.hard_moments and self.np_random.random() < self.hard_frac:
            self.hard_start = True
            if self.hard_states and self.np_random.random() < 0.5:
                qpos, qvel, ctrl, i0 = self.hard_states[int(self.np_random.integers(len(self.hard_states)))]
                self._start(0)
                self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = qpos, qvel, ctrl
                mj.mj_forward(self.m, self.d)
                self.i0 = i0
            else:
                mo = self.hard_moments[int(self.np_random.integers(len(self.hard_moments)))]
                back = self.np_random.uniform(*self.HARD_BACK) * self.speed / CONTROL_DT
                self._start(int(round(mo - back)) % n)
        else:
            self._start(int(self.np_random.choice(len(self.start_p), p=self.start_p)))
        self.t, self.tau, self.steps = 0, 0.0, int(round(self.seconds() / CONTROL_DT))
        self.err_sum, self.th_prev, self.turned = 0.0, self._angles(), 0.0
        return self._obs(), {}

    def step(self, action):
        if self.hard_frac and self.t % self.SNAP_EVERY == 0:
            self.snaps.append((self.d.qpos.copy(), self.d.qvel.copy(), self.d.ctrl.copy(), self._moment(), self.turned))
        a = np.clip(action, -1, 1)
        self.d.ctrl[:] = np.clip(self.d.ctrl + a * RATE, self.lo, self.hi)
        for _ in range(self.n_sub):
            mj.mj_step(self.m, self.d)
        self.t += 1
        b = self.w.balls()
        th = self._angles()
        dth = float(np.mean(np.angle(np.exp(1j * (th - self.th_prev)))))
        self.turned += dth
        self.th_prev = th
        self.turn_hist.append(self.turned)
        if (self.stall_starts and self.hard_frac and not self.stalled and not self.fix_start and len(self.turn_hist) == self.turn_hist.maxlen
                and self.turned - self.turn_hist[0] < np.radians(self.STALL_DEG)):
            self.stalled = True                  # once per episode, from where the balls are (not the target)
            self.hard_moments.append(self._aligned(self.turned))
            k = int(round((self.STALL_S + np.mean(self.HARD_BACK)) / (self.SNAP_EVERY * CONTROL_DT)))
            if len(self.snaps) > k:
                q, v, c, _, tn = self.snaps[-1 - k]
                self.hard_states.append((q, v, c, self._aligned(tn)))
        if self.lead is None or self._demo_at(self.tau)[1] - self._demo_at(0.0)[1] - self.turned < self.lead:
            self.tau += CONTROL_DT
        e = np.linalg.norm(b - self.target_pos(self.tau), axis=1) * 100
        lost = self.lost(b)
        reward = float(np.mean(np.exp(-e / TRACK_SCALE))) - W_RATE * float(np.sum(a ** 2))
        if self.w_prog:
            reward += self.w_prog * float(np.clip(dth / (self.rate * CONTROL_DT), -1.0, 1.0))
        self.err_sum += float(e.mean())
        if lost:
            reward -= DROP
        truncated = self.t >= self.steps
        info = {"err": float(e.mean())}
        if lost or truncated:
            mean_err = self.err_sum / self.t
            info.update(lost=lost, mean_err=mean_err, turned=self.turned, level=self.level, tip_rate=0.0,
                        level_ok=self.level_ok(lost, truncated, mean_err), hard_start=float(self.hard_start),
                        wait_frac=1.0 - self.tau / (self.t * CONTROL_DT), stalled=float(self.stalled),
                        fix_start=float(self.fix_start))
            if lost and self.hard_frac and not self.fix_start:
                self.hard_moments.append(self._moment() % (len(self.phase) - 1))
                k = int(round(np.mean(self.HARD_BACK) / (self.SNAP_EVERY * CONTROL_DT)))
                if len(self.snaps) > k:
                    self.hard_states.append(self.snaps[-1 - k][:4])
        return self._obs(), reward, lost, truncated and not lost, info

    # ---- behaviour cloning pairs from the demo
    def demo_pairs(self):
        """(obs, action) per demo step at 1x: the targets are the demo's own ball positions now and AHEAD s on."""
        z, k = self.demo, int(round(AHEAD / CONTROL_DT))
        n = len(z["qpos"])
        obs, act = [], []
        for i in range(n - 1):
            self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = z["qpos"][i], z["qvel"][i], z["ctrl"][i]
            mj.mj_forward(self.m, self.d)
            obs.append(self.obs_for(z["balls"][i], z["balls"][min(i + k, n - 1)], self.phase[i], self.demo_rate))
            act.append(np.clip((z["ctrl"][i + 1] - z["ctrl"][i]) / RATE, -1, 1))
        return np.array(obs), np.array(act, dtype=np.float32)

    def fix_pairs(self):
        """(obs, action) per step of each fix's planned part, its targets the fix's own balls (as demo_pairs)."""
        k, obs, act = int(round(AHEAD / CONTROL_DT)), [], []
        for fx in self.fixes:
            z, c = fx["log"], fx
            n = len(z["qpos"])
            for i in range(n - 1):
                self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = z["qpos"][i], z["qvel"][i], z["ctrl"][i]
                mj.mj_forward(self.m, self.d)
                obs.append(self.obs_for(c["balls"][i], c["balls"][i + k], c["phase"][i], self.demo_rate))
                act.append(np.clip((z["ctrl"][i + 1] - z["ctrl"][i]) / RATE, -1, 1))
        return np.array(obs), np.array(act, dtype=np.float32)


FIX_MIN_DEG = 20.0
CLIP_TAIL_S = 40.0


def load_fixes(path, env):
    """The planner fixes in directory `path` that turned >= FIX_MIN_DEG without a loss, each as a clip: the
    fix's balls, then the demo's on from the demo step matching the fix's end turn (CLIP_TAIL_S s of it)."""
    out, n = [], len(env.phase) - 1
    for f in sorted(Path(path).glob("fix_*.npz")):
        z = np.load(f)
        s = np.load(Path(path).parent / "stalls" / f.name.replace("fix_", "stall_"))
        if len(z["balls"]) < 10:
            continue
        tn = turn_of(z["balls"], env.center)
        if np.degrees(tn[-1]) < FIX_MIN_DEG or env.lost(z["balls"][-1]):
            continue
        ph = env.phase[int(s["moment"])] + tn
        j = int(min(np.searchsorted(env.mono, ph[-1] % env.phase[-1]), n - 1))
        idx = j + np.arange(1, int(CLIP_TAIL_S / CONTROL_DT))
        wrap = idx // n
        idx = idx % n
        tail_ph = env.phase[idx] + wrap * env.phase[-1] - env.phase[j] + ph[-1]
        out.append(dict(qpos=z["qpos"][0], qvel=z["qvel"][0], ctrl=z["ctrl"][0],
                        balls=np.concatenate([z["balls"], env.demo["balls"][idx]]),
                        phase=np.concatenate([ph, tail_ph]),
                        log={k: z[k] for k in ("qpos", "qvel", "ctrl")}))
    return out
