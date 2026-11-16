"""Stage 7A, open-hand loop, step 2: RL with a MOMENTUM reward -- no demo to track. The open-hand scene
(38mm balls, two per palm on a 4.5cm ring, fingers within BAND of open, thumbs within THUMB_BAND, wrists
+-WRIST_TILT, forearm roll joints +-ROLL that tilt the palms toward each other); episodes start with the
balls already rolling round the loop at a random rate up to SPIN_MAX.

Reward per control step:
    ALIVE + W_TURN * (0.5 mean + 0.5 min) over balls of clip(ball's turn this step / the reference turn, -1, 1.5)
    - W_SPREAD * spread of the balls' total turns (deg, /100)      (the four go round together)
    - W_CROWD  * cm two balls come closer than 1cm                  (they stay apart)
    - W_RING   * cm each ball is off the ring (|r - rc|, summed)     (they stay a circle)
    - W_RATE * |action|^2;    a lost ball (dropped, beyond FAR or within MIN_R of the centre) ends it with -DROP.
The reference turn is REF_DEG per second. A planner could creep this loop round but not keep it rolling,
and a fixed rhythm (stage7_rhythm.py) cannot correct the slow outward drift -- it needs feedback.
"""
import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from stage7_layout import PALM_Z
from stage7_teacher import World

OPEN = {f"rh_A_{f}J{j}": v for f in ("FF", "MF", "RF", "LF") for j, v in ((3, 0.1), (0, 0.1))}
BALL_R, FLEX, BAND, THUMB_BAND = 0.019, 0.10, 0.45, 0.15
WRIST_TILT, ROLL = np.radians(25.0), np.radians(20.0)
RING, ALONG = 0.045, 0.33
FAR, MIN_R = 0.07, 0.03
LOST_Z = PALM_Z - 0.015          # a ball this low is off the hands (a palm rolled 20 deg carries a ball below PALM_Z)
CONTROL_DT, RATE = 0.02, 0.12
EP_SECONDS = 10.0
SPIN_MAX = 45.0                  # deg/s
REF_DEG = 45.0                   # deg/s the reward calls a full turn rate -- too high: the planner managed ~2.5 deg/s, so
                                 # reachable turning paid ~0.05/step vs alive .3 and every run learned to hold (loop_05: 5)
W_TURN, W_SPREAD, W_CROWD, W_RING, W_RATE, DROP = 1.0, 0.2, 0.5, 0.02, 0.005, 20.0
ALIVE = 0.5                      # per step: holding on must beat dropping a ball early -- but loop_01 then parked the
                                 # balls ~30 deg on and just held them (alive pays, turning risks a drop); loop_02 (0.1/2.0)
                                 # then FLUNG them round for 1.4 s and dropped one (the burst beat -20 at gamma .99):
                                 # loop_03 alive .3, turn 1, drop 100, prog capped at 1, gamma .995


class LoopEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_seconds=None, spin_max=SPIN_MAX, ref_deg=REF_DEG, randomize=True, alive=ALIVE, w_turn=W_TURN,
                 drop=DROP, prog_max=1.5, demo=None, demo_frac=0.0):
        super().__init__()
        self.demo_frac = float(demo_frac)
        self.demo = None
        if demo:                                         # a planner run in this same scene (stage7_teacher.py --roll ...)
            z = np.load(demo)
            self.demo = {k: z[k] for k in ("qpos", "qvel", "ctrl")}
        self.alive, self.w_turn, self.drop, self.prog_max = alive, w_turn, drop, prog_max
        self.w = World(ball=BALL_R, tilt=WRIST_TILT, flex=FLEX, posture=OPEN, band=BAND, roll=ROLL, thumb_band=THUMB_BAND)
        self.m, self.d = self.w.m, self.w.d
        self.n_sub = int(round(CONTROL_DT / self.m.opt.timestep))
        self.seconds = episode_seconds or EP_SECONDS
        self.spin_max, self.ref = spin_max, np.radians(ref_deg) * CONTROL_DT
        self.randomize = randomize
        self.w.settle(along=ALONG, ring=RING, seconds=1.5, phase=np.radians(45))
        self.center = self.w.balls()[:, :2].mean(0)
        self.rc = float(np.linalg.norm(self.w.balls()[:, :2] - self.center, axis=1).mean())
        self.spec = mj.mjtState.mjSTATE_INTEGRATION
        self.state0 = np.empty(mj.mj_stateSize(self.m, self.spec))
        mj.mj_getState(self.m, self.d, self.state0, self.spec)
        hinge = [j for j in range(self.m.njnt) if self.m.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE]
        self.hq = np.array([self.m.jnt_qposadr[j] for j in hinge])
        self.hv = np.array([self.m.jnt_dofadr[j] for j in hinge])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.m.nu,), dtype=np.float32)
        nobs = 2 * len(self.hq) + self.m.nu + 4 * 3 + 4 * 3 + 4 * 2 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(nobs,), dtype=np.float32)

    def _angles(self, b):
        return np.arctan2(b[:, 1] - self.center[1], b[:, 0] - self.center[0])

    def _obs(self):
        b = self.w.balls()
        v = np.concatenate([self.d.qvel[a:a + 3] for a in self.w.bv])
        th = self._angles(b)
        return np.concatenate([
            self.d.qpos[self.hq], 0.1 * self.d.qvel[self.hv], self.d.ctrl,
            10 * (b - [*self.center, b[:, 2].mean()]).ravel(), v, np.cos(th), np.sin(th),
            [self.t / self.steps]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_setState(self.m, self.d, self.state0, self.spec)
        rng = self.np_random
        if self.demo is not None and self.randomize and rng.random() < self.demo_frac:
            i = int(rng.integers(len(self.demo["qpos"]) - 1))    # a moment of the demo, balls mid-loop and moving
            self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = self.demo["qpos"][i], self.demo["qvel"][i], self.demo["ctrl"][i]
            mj.mj_forward(self.m, self.d)
            self.t, self.steps = 0, int(round(self.seconds / CONTROL_DT))
            self.th_prev, self.turn = self._angles(self.w.balls()), np.zeros(4)
            return self._obs(), {}
        if self.randomize:                               # a little jitter in the balls' places
            for q in self.w.bq:
                self.d.qpos[q:q + 2] += rng.normal(0, 0.002, 2)
        om = np.radians(rng.uniform(0, self.spin_max)) if self.randomize else np.radians(self.spin_max)
        for q, v in zip(self.w.bq, self.w.bv):           # already rolling round the loop
            dx, dy = self.d.qpos[q] - self.center[0], self.d.qpos[q + 1] - self.center[1]
            self.d.qvel[v:v + 2] = [-om * dy, om * dx]
        mj.mj_forward(self.m, self.d)
        self.t, self.steps = 0, int(round(self.seconds / CONTROL_DT))
        self.th_prev, self.turn = self._angles(self.w.balls()), np.zeros(4)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        self.d.ctrl[:] = np.clip(self.d.ctrl + a * RATE, self.w.lo, self.w.hi)
        for _ in range(self.n_sub):
            mj.mj_step(self.m, self.d)
        self.t += 1
        b = self.w.balls()
        th = self._angles(b)
        dth = np.angle(np.exp(1j * (th - self.th_prev)))
        self.th_prev = th
        self.turn += dth
        prog = np.clip(dth / self.ref, -1.0, self.prog_max)
        r = np.linalg.norm(b[:, :2] - self.center, axis=1)
        g = np.array([np.linalg.norm(b[p, :2] - b[q, :2]) * 100 - 200 * BALL_R for p in range(4) for q in range(p + 1, 4)])
        reward = (self.alive + self.w_turn * (0.5 * prog.mean() + 0.5 * prog.min())
                  - W_SPREAD * float(np.std(np.degrees(self.turn))) / 100
                  - W_CROWD * float(np.sum(np.maximum(1.0 - g, 0.0)))
                  - W_RING * float(np.sum(np.abs(r - self.rc))) * 100
                  - W_RATE * float(np.sum(a ** 2)))
        lost = bool(np.any(b[:, 2] < LOST_Z) or np.any(r > FAR) or np.any(r < MIN_R))
        if lost:
            reward -= self.drop
        trunc = self.t >= self.steps
        info = {}
        if lost or trunc:
            deg = np.degrees(self.turn)
            info = dict(lost=lost, turned=float(deg.mean()), turned_min=float(deg.min()), seconds=self.t * CONTROL_DT)
        return self._obs(), reward, lost, trunc and not lost, info

    def demo_pairs(self):
        """(obs, action) per demo step, for behaviour cloning: the action that reproduces the demo's next controls."""
        z, obs, act = self.demo, [], []
        self.t, self.steps = 0, int(round(self.seconds / CONTROL_DT))
        for i in range(len(z["qpos"]) - 1):
            self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = z["qpos"][i], z["qvel"][i], z["ctrl"][i]
            mj.mj_forward(self.m, self.d)
            obs.append(self._obs())
            act.append(np.clip((z["ctrl"][i + 1] - z["ctrl"][i]) / RATE, -1, 1))
        return np.array(obs), np.array(act, dtype=np.float32)


# ---------------------------------------------------------------------------------------------------------
# Demo tracking on the loop (DeepMimic-style, the packed square's best recipe): each ball also tracks its own
# position in a planner demo of this scene at the matching moment, played back SPEEDS x faster by a curriculum.
SPEEDS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
AHEAD, TRACK_SCALE, W_TRACK = 0.5, 1.5, 1.0
OK_ERR, OK_TURN = 1.5, 0.5          # a level passed: no loss, mean error < 1.5cm, turned >= half the demo's turn


class LoopTrackEnv(LoopEnv):
    def __init__(self, demo, level=0, w_track=W_TRACK, **kw):
        kw.setdefault("ref_deg", 5.0)
        kw.setdefault("alive", 0.1)
        kw.setdefault("drop", 100.0)
        super().__init__(demo=demo, **kw)
        from rl_env_stage7 import turn_of
        z = np.load(demo)
        self.demo["balls"] = z["balls"]
        self.phase = turn_of(z["balls"], self.center)
        self.loop_turn = float(self.phase[-1] - 2 * np.pi)
        self.full_loop = self.phase[-1] >= np.radians(350)   # a partial demo is held at its end, not wrapped round
        self.demo_rate = float(self.phase[-1] / ((len(self.phase) - 1) * CONTROL_DT))
        self.level, self.w_track = int(level), w_track
        self.c3 = np.array([*self.center, z["balls"][:, :, 2].mean()])
        nobs = self.observation_space.shape[0] + 4 * 3 * 2 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(nobs,), dtype=np.float32)

    @property
    def speed(self):
        return SPEEDS[self.level]

    def set_level(self, level):
        self.level = int(min(max(level, 0), len(SPEEDS) - 1))
        return self.level

    def _demo_at(self, t):
        """(ball positions, phase) of the demo `t` s after this episode's start moment (it loops: one full turn
        brings each ball back to its own place, the few degrees' overshoot rotated in)."""
        n = len(self.phase) - 1
        if not self.full_loop:
            f = min(self.i0 + self.speed * t / CONTROL_DT, n - 1e-6)
            j, a = int(f), f - int(f)
            return ((1 - a) * self.demo["balls"][j] + a * self.demo["balls"][j + 1],
                    (1 - a) * self.phase[j] + a * self.phase[j + 1])
        k, f = divmod(self.i0 + self.speed * t / CONTROL_DT, n)
        j = int(f)
        a = f - j
        p = (1 - a) * self.demo["balls"][j] + a * self.demo["balls"][j + 1]
        ph = (1 - a) * self.phase[j] + a * self.phase[j + 1] + k * self.phase[-1]
        if k:
            dd = p[:, :2] - self.center
            th, r = np.arctan2(dd[:, 1], dd[:, 0]) + k * self.loop_turn, np.linalg.norm(dd, axis=1)
            p = p.copy()
            p[:, 0], p[:, 1] = self.center[0] + r * np.cos(th), self.center[1] + r * np.sin(th)
        return p, ph

    def _track_obs(self, tg, ta, ph, b):
        return np.concatenate([10 * (tg - self.c3).ravel(), 10 * (ta - b).ravel(),
                               [np.cos(ph), np.sin(ph), self.speed * self.demo_rate]]).astype(np.float32)

    def _obs(self):
        base = super()._obs()
        if not hasattr(self, "i0"):
            return np.concatenate([base, np.zeros(27, np.float32)])
        t = self.t * CONTROL_DT
        tg, ph = self._demo_at(t)
        return np.concatenate([base, self._track_obs(tg, self._demo_at(t + AHEAD)[0], ph, self.w.balls())])

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        rng = self.np_random
        mj.mj_setState(self.m, self.d, self.state0, self.spec)
        if self.randomize and rng.random() < self.demo_frac:
            hi = len(self.demo["qpos"]) - 1
            if not self.full_loop:                       # the whole episode inside the demo
                hi = max(hi - int(self.speed * self.seconds / CONTROL_DT), 1)
            self.i0 = int(rng.integers(hi))
            self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = (self.demo[k][self.i0] for k in ("qpos", "qvel", "ctrl"))
        else:
            self.i0 = 0
        if options and "start" in options:
            self.i0 = int(options["start"])
            self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = (self.demo[k][self.i0] for k in ("qpos", "qvel", "ctrl"))
        mj.mj_forward(self.m, self.d)
        self.t, self.steps = 0, int(round(self.seconds / CONTROL_DT))
        self.th_prev, self.turn, self.err_sum = self._angles(self.w.balls()), np.zeros(4), 0.0
        return self._obs(), {}

    def step(self, action):
        obs, reward, lost, trunc, info = super().step(action)
        e = np.linalg.norm(self.w.balls() - self._demo_at(self.t * CONTROL_DT)[0], axis=1) * 100
        reward += self.w_track * float(np.mean(np.exp(-e / TRACK_SCALE)))
        self.err_sum += float(e.mean())
        if info:
            mean_err = self.err_sum / self.t
            want = np.degrees(self._demo_at(self.t * CONTROL_DT)[1] - self._demo_at(0.0)[1])
            info.update(mean_err=mean_err, level=self.level,
                        level_ok=bool(not lost and trunc and mean_err < OK_ERR and info["turned"] >= OK_TURN * want))
        return obs, reward, lost, trunc, info

    def demo_pairs(self):
        """(obs, action) per demo step at 1x: targets the demo's own balls now and AHEAD s on."""
        z, k = self.demo, int(round(AHEAD / CONTROL_DT))
        n = len(z["qpos"])
        self.t, self.steps, self.i0 = 0, int(round(self.seconds / CONTROL_DT)), 0
        obs, act = [], []
        for i in range(n - 1):
            self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:] = z["qpos"][i], z["qvel"][i], z["ctrl"][i]
            mj.mj_forward(self.m, self.d)
            base = LoopEnv._obs(self)
            obs.append(np.concatenate([base, self._track_obs(z["balls"][i], z["balls"][min(i + k, n - 1)], self.phase[i],
                                                              self.w.balls())]))
            act.append(np.clip((z["ctrl"][i + 1] - z["ctrl"][i]) / RATE, -1, 1))
        return np.array(obs), np.array(act, dtype=np.float32)
