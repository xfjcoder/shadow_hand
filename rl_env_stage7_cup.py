"""Stage 7A, the long cup (stage7_cup.py): RL on top of the swirl-and-flick rhythm.

Both hands cupped into one long bowl (the forearm rolls on one axis along the seam, so the hands tilt as one
rigid cup), four 38mm balls. The open-loop rhythm -- the whole cup tilted round a circle, the knuckles
flicking once a circle -- stirs the balls across both hands but cannot keep them going round the rim: they
churn through the middle and one parks. The policy acts as a RESIDUAL on the rhythm,
    u = rhythm(t) + RES * action          (zero action = the rhythm)
seeing the balls, so it can keep them out on the rim, spaced and all going one way.

Reward per control step:
    ALIVE + W_TURN * (0.5 mean + 0.5 min) over balls of clip(rim turn this step / REF, -1, 1)
    - W_SPREAD * spread of the balls' total turns (deg / 100)
    - W_CROWD  * cm two balls come closer than 1cm
    - W_CENTER * cm a ball is inside RIM_R (through the middle is not round the rim)
    - W_RATE   * |action|^2;    a ball off the hands ends it with -DROP.
A ball's turn counts only while it is beyond RIM_R from the cup centre. Episodes start from the rhythm's own
open-loop run at a random moment (balls already moving), so the policy meets the states the rhythm makes.
"""
import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

from stage7_cup import FINGERS, cup_posture
from stage7_layout import PALM_Z
from stage7_teacher import World

BALL_R = 0.019
CUP = dict(curl=0.5, tip=1.0, lf=0.5, lf_tip=0.6, th4=1.1, th5=0.3, th1=0.3)
FLEX, ALONG, RING = 0.05, 0.31, 0.045
# the rhythm (stage7_cup.py --swirl --flicks: the most even of the kept runs, turns 49/40/38/51 deg in 15 s)
RHYTHM = dict(T=0.4, a_flex=0.10, a_roll=0.20, sense=1, flick=(0.6, 1.5 * np.pi, 0.5 * np.pi), ramp=3.0)
RES_WRIST, RES_FINGER = 0.15, 0.25        # rad: the residual's reach on the wrists/rolls and on every finger/thumb joint
CONTROL_DT = 0.02
EP_SECONDS = 10.0
RIM_R, FAR, LOST_Z = 0.025, 0.10, PALM_Z - 0.05
REF_DEG = 30.0                            # deg/s of rim turn per ball paid in full (the rhythm makes ~3)
ALIVE, W_TURN, W_SPREAD, W_CROWD, W_CENTER, W_RATE, DROP = 0.1, 1.0, 0.2, 0.05, 0.2, 0.01, 100.0
# W_CROWD small: in a cup the balls touch all the time (at 0.5 it was -0.65/step under the rhythm, 6x the turn pay)
START_POOL_S = 12.0                       # s of the rhythm's open-loop run to start episodes from


class CupEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_seconds=None, randomize=True, ref_deg=REF_DEG, res_scale=1.0):
        super().__init__()
        self.w = World(ball=BALL_R, tilt=np.radians(25), flex=FLEX, posture=cup_posture(**CUP), roll=np.radians(20),
                       roll_at="seam")
        self.m, self.d = self.w.m, self.w.d
        self.n_sub = int(round(CONTROL_DT / self.m.opt.timestep))
        self.seconds = episode_seconds or EP_SECONDS
        self.ref = np.radians(ref_deg) * CONTROL_DT
        self.randomize = randomize
        self.w.settle(along=ALONG, ring=RING, seconds=3.0, phase=np.radians(45))
        self.center = self.w.balls()[:, :2].mean(0)
        self.spec = mj.mjtState.mjSTATE_INTEGRATION
        m = self.m
        self.fl = [m.actuator(f"{h}_A_WRJ1").id for h in ("rh", "L_lh")]
        self.ro = [m.actuator(f"{h}_A_FRJ").id for h in ("rh", "L_lh")]
        self.kn = [[m.actuator(f"{h}_A_{f}J3").id for f in FINGERS + ("LF",)] for h in ("rh", "L_lh")]
        wrist = set(self.fl + self.ro + [m.actuator(f"{h}_A_WRJ2").id for h in ("rh", "L_lh")])
        self.res = res_scale * np.array([RES_WRIST if i in wrist else RES_FINGER for i in range(m.nu)])
        hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE]
        self.hq = np.array([m.jnt_qposadr[j] for j in hinge])
        self.hv = np.array([m.jnt_dofadr[j] for j in hinge])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(m.nu,), dtype=np.float32)
        nobs = 2 * len(self.hq) + 4 * 3 + 4 * 3 + 4 * 2 + 4 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(nobs,), dtype=np.float32)
        self._start_pool()

    def rhythm(self, t):
        R = RHYTHM
        w = 2 * np.pi * t / R["T"]
        g = min(t / (R["ramp"] * R["T"]), 1.0)
        u = self.w.u0.copy()
        u[self.fl] += g * R["a_flex"] * np.sin(w)
        u[self.ro] += g * R["a_roll"] * np.sin(w + R["sense"] * np.pi / 2)
        amp, ph, off = R["flick"]
        for k, ids in enumerate(self.kn):
            u[ids] += g * amp * max(0.0, np.sin(w + ph + k * off)) ** 2
        return u

    def _start_pool(self):
        """The rhythm run open-loop from the settled cup: (time, state) every 0.1 s while all four balls are on."""
        st0 = np.empty(mj.mj_stateSize(self.m, self.spec))
        mj.mj_getState(self.m, self.d, st0, self.spec)
        self.pool = [(0.0, st0.copy())]
        for i in range(int(round(START_POOL_S / CONTROL_DT))):
            t = i * CONTROL_DT
            self.d.ctrl[:] = np.clip(self.rhythm(t), self.w.lo, self.w.hi)
            for _ in range(self.n_sub):
                mj.mj_step(self.m, self.d)
            if self._lost(self.w.balls()):
                break
            if (i + 1) % 5 == 0:
                st = np.empty_like(st0)
                mj.mj_getState(self.m, self.d, st, self.spec)
                self.pool.append((t + CONTROL_DT, st))
        mj.mj_setState(self.m, self.d, st0, self.spec)

    def _lost(self, b):
        return bool(np.any(b[:, 2] < LOST_Z) or np.any(np.linalg.norm(b[:, :2] - self.center, axis=1) > FAR))

    def _angles(self, b):
        return np.arctan2(b[:, 1] - self.center[1], b[:, 0] - self.center[0])

    def _obs(self):
        b = self.w.balls()
        rel = b[:, :2] - self.center
        v = np.concatenate([self.d.qvel[a:a + 3] for a in self.w.bv])
        th = self._angles(b)
        ph = 2 * np.pi * self.t_abs / RHYTHM["T"]
        return np.concatenate([
            self.d.qpos[self.hq], 0.1 * self.d.qvel[self.hv],
            10 * np.column_stack([rel, b[:, 2] - PALM_Z]).ravel(), v, np.cos(th), np.sin(th),
            10 * np.linalg.norm(rel, axis=1), [np.cos(ph), np.sin(ph), min(self.t_abs / (RHYTHM["ramp"] * RHYTHM["T"]), 1.0)]
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        k = int(rng.integers(len(self.pool))) if self.randomize else 0
        self.t_abs, st = self.pool[k]
        mj.mj_setState(self.m, self.d, st, self.spec)
        if self.randomize:
            for q in self.w.bq:
                self.d.qpos[q:q + 2] += rng.normal(0, 0.001, 2)
        mj.mj_forward(self.m, self.d)
        self.t, self.steps = 0, int(round(self.seconds / CONTROL_DT))
        self.th_prev, self.turn = self._angles(self.w.balls()), np.zeros(4)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        self.d.ctrl[:] = np.clip(self.rhythm(self.t_abs) + self.res * a, self.w.lo, self.w.hi)
        for _ in range(self.n_sub):
            mj.mj_step(self.m, self.d)
        self.t += 1
        self.t_abs += CONTROL_DT
        b = self.w.balls()
        r = np.linalg.norm(b[:, :2] - self.center, axis=1)
        th = self._angles(b)
        dth = (r > RIM_R) * np.angle(np.exp(1j * (th - self.th_prev)))
        self.th_prev = th
        self.turn += dth
        prog = np.clip(RHYTHM["sense"] * dth / self.ref, -1.0, 1.0)
        g = np.array([np.linalg.norm(b[p, :2] - b[q, :2]) * 100 - 200 * BALL_R for p in range(4) for q in range(p + 1, 4)])
        reward = (ALIVE + W_TURN * (0.5 * prog.mean() + 0.5 * prog.min())
                  - W_SPREAD * float(np.std(np.degrees(self.turn))) / 100
                  - W_CROWD * float(np.sum(np.maximum(1.0 - g, 0.0)))
                  - W_CENTER * float(np.sum(np.maximum(RIM_R - r, 0.0))) * 100
                  - W_RATE * float(np.sum(a ** 2)))
        lost = self._lost(b)
        if lost:
            reward -= DROP
        trunc = self.t >= self.steps
        info = {}
        if lost or trunc:
            deg = RHYTHM["sense"] * np.degrees(self.turn)
            info = dict(lost=lost, turned=float(deg.mean()), turned_min=float(deg.min()), seconds=self.t * CONTROL_DT)
        return self._obs(), reward, lost, trunc and not lost, info
