"""Stage 7A, the long cup with the FINGER WAVE (stage7_wave.py, stage7_wave_loop.py): RL, one ball first.

The scripted loop: a wave of finger lifts carries the ball along the finger row (+y) across both hands, the
index fingers curled as walls at the row's ends, the thumbs low (0.3); when the ball passes y_up the cup tips
toward the wrists and rolls toward -y so it comes back along the palm, and past y_down it tips toward the
fingers again (`base()`, feedback on the ball's place). The out leg works; the way back stalls in the hollow
at the left thumb's base (best 1.3 laps in 15 s, usually 0.3). The policy is a RESIDUAL on that script,
    u = base(t, ball) + RES * action          (zero action = the script)
on every joint (fingers and thumbs with twice the cup env's reach), to get the ball out of the pockets.

Reward per control step: ALIVE + clip(turn round the loop centre this step / REF, -1, 1) (counted only beyond
RIM from the centre) - W_RATE |action|^2; a ball off the hands ends it with -DROP. Episodes start with the
ball at a random place on the finger row or the palm and the wave at a random phase.
"""
import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

import stage7_wave
from stage7_layout import PALM_Z, set_ball
from stage7_teacher import BALLS
from stage7_wave import PARK, ROW

stage7_wave.TH4 = 0.3
BALL_R = 0.019
# the scripted loop (stage7_wave_loop.py --fsm, thumbs 0.3: 1.3 laps in 15 s, the best)
BASE = dict(T=0.4, amp=1.0, lag=0.18, wall=0.8, flex_row=-0.1, flex_up=0.1, roll_back=0.1, y_up=0.07, y_down=0.02)
RES_WRIST, RES_FINGER = 0.15, 0.5
CONTROL_DT = 0.02
EP_SECONDS = 10.0
RIM, FAR, LOST_Z = 0.02, 0.14, PALM_Z - 0.05       # FAR: the fingertips are ~12 cm from the centre
REF_DEG = 180.0                           # deg/s round the loop paid in full (the script makes ~30)
ALIVE, W_RATE, DROP = 0.05, 0.01, 50.0
PALM_START = 0.3                          # fraction of episodes starting with the ball on the palm


class WaveEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_seconds=None, randomize=True, ref_deg=REF_DEG):
        super().__init__()
        self.w, self.row = stage7_wave.make(0.0, 0.3, BASE["flex_row"], ball=BALL_R)
        self.m, self.d = self.w.m, self.w.d
        m = self.m
        self.n_sub = int(round(CONTROL_DT / m.opt.timestep))
        self.seconds = episode_seconds or EP_SECONDS
        self.ref = np.radians(ref_deg) * CONTROL_DT
        self.randomize = randomize
        self.c = np.array([self.row[:, 0].mean() - 0.04, self.row[:, 1].mean()])
        self.kn = [m.actuator(f"{h}_A_{f}J3").id for h, f in ROW]
        self.ro = [m.actuator(f"{h}_A_FRJ").id for h in ("rh", "L_lh")]
        self.fx = [m.actuator(f"{h}_A_WRJ1").id for h in ("rh", "L_lh")]
        wrist = set(self.fx + self.ro + [m.actuator(f"{h}_A_WRJ2").id for h in ("rh", "L_lh")])
        self.res = np.array([RES_WRIST if i in wrist else RES_FINGER for i in range(m.nu)])
        hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE]
        self.hq = np.array([m.jnt_qposadr[j] for j in hinge])
        self.hv = np.array([m.jnt_dofadr[j] for j in hinge])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(m.nu,), dtype=np.float32)
        nobs = 2 * len(self.hq) + 14
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(nobs,), dtype=np.float32)

    def ball(self):
        return self.w.balls()[0]

    def base(self):
        """The scripted loop's controls now (advances its leg and slewed tilts)."""
        B, w = BASE, self.w
        b = self.ball()
        if self.leg == 0 and b[1] > B["y_up"]:
            self.leg = 1
        elif self.leg == 1 and b[1] < B["y_down"]:
            self.leg = 0
        tf, tr = (B["flex_row"], 0.0) if self.leg == 0 else (B["flex_up"], B["roll_back"])
        self.flex += np.clip(tf - self.flex, -0.02, 0.02)
        self.roll += np.clip(tr - self.roll, -0.02, 0.02)
        u = w.u0.copy()
        u[self.fx] = self.flex
        u[self.ro] += self.roll
        u[[self.kn[0], self.kn[-1]]] += B["wall"]
        for k in range(1, 7):
            u[self.kn[k]] += B["amp"] * max(0.0, np.sin(2 * np.pi * (self.t_wave / B["T"] - k * B["lag"]))) ** 2
        return u

    def _angle(self, b):
        return np.arctan2(b[1] - self.c[1], b[0] - self.c[0])

    def _obs(self):
        b = self.ball()
        rel = b[:2] - self.c
        v = self.d.qvel[self.w.bv[0]:self.w.bv[0] + 3]
        th = self._angle(b)
        ph = 2 * np.pi * self.t_wave / BASE["T"]
        return np.concatenate([
            self.d.qpos[self.hq], 0.1 * self.d.qvel[self.hv],
            10 * np.array([rel[0], rel[1], b[2] - PALM_Z]), v, [np.cos(th), np.sin(th), 10 * np.linalg.norm(rel)],
            [np.cos(ph), np.sin(ph), self.leg, self.flex, self.roll]
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng, m, d, w = self.np_random, self.m, self.d, self.w
        mj.mj_setState(m, d, w.st0, stage7_wave.S)
        if self.randomize and rng.random() < PALM_START:
            p = [rng.uniform(0.28, 0.31), rng.uniform(-0.01, 0.09), PALM_Z + BALL_R + 0.01]
        else:
            i0 = rng.uniform(1.0, 6.0) if self.randomize else 1.5
            x_off = rng.uniform(0.01, 0.03) if self.randomize else 0.02
            p = [self.row[:, 0].mean() + x_off, np.interp(i0, np.arange(8), self.row[:, 1]), PALM_Z + BALL_R + 0.01]
        set_ball(m, d, BALLS[0], p)
        for b, q in zip(BALLS[1:], PARK):
            set_ball(m, d, b, q)
        on_palm = p[0] < self.row[:, 0].mean() - 0.02            # on the palm: the way back
        self.leg = int(on_palm)
        self.flex, self.roll = (BASE["flex_up"], BASE["roll_back"]) if on_palm else (BASE["flex_row"], 0.0)
        self.t_wave = rng.uniform(0, BASE["T"]) if self.randomize else 0.0
        for _ in range(10):                                      # 0.2 s under the script (walls up, the wave going):
            d.ctrl[:] = np.clip(self.base(), w.lo, w.hi)         # flat fingers alone let it roll off the tips
            for _ in range(self.n_sub):
                mj.mj_step(m, d)
            self.t_wave += CONTROL_DT
        self.t, self.steps = 0, int(round(self.seconds / CONTROL_DT))
        self.th_prev, self.turn = self._angle(self.ball()), 0.0
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        self.d.ctrl[:] = np.clip(self.base() + self.res * a, self.w.lo, self.w.hi)
        for _ in range(self.n_sub):
            mj.mj_step(self.m, self.d)
        self.t += 1
        self.t_wave += CONTROL_DT
        b = self.ball()
        th = self._angle(b)
        dth = float(np.linalg.norm(b[:2] - self.c) > RIM) * np.angle(np.exp(1j * (th - self.th_prev)))
        self.th_prev = th
        self.turn += dth
        reward = ALIVE + float(np.clip(dth / self.ref, -1.0, 1.0)) - W_RATE * float(np.sum(a ** 2))
        lost = bool(b[2] < LOST_Z or np.linalg.norm(b[:2] - self.c) > FAR)
        if lost:
            reward -= DROP
        trunc = self.t >= self.steps
        info = {}
        if lost or trunc:
            deg = float(np.degrees(self.turn))
            info = dict(lost=lost, turned=deg, turned_min=deg, seconds=self.t * CONTROL_DT)
        return self._obs(), reward, lost, trunc and not lost, info


# ---------------------------------------------------------------------------------------------------------------
# Several balls round the same loop. The one-ball script switches the cup tilt on where THE ball is; with balls
# on both legs at once (one on the finger row, one on the palm) they need opposite tilts, so the base tilt here
# is a fixed compromise (BASE_FLEX_N, BASE_ROLL_N: midway between the two legs' tilts) and the policy steers it
# (its +-RES_WRIST reaches both legs' tilts). The wave and the index-finger walls run as before.
BASE_FLEX_N, BASE_ROLL_N = 0.0, 0.05
W_CROWD_N = 0.05                          # per cm two balls come closer than 1cm
EDGE_X, EDGE_W = 0.25, 0.015              # m: the heel of the palms; balls past it roll off toward the wrists
FAST_X, FAST_V, FAST_W = 0.30, 0.10, 0.10 # balls below x 30cm moving wristward faster than 10cm/s overshoot the palm leg


class MultiWaveEnv(WaveEnv):
    """`n` balls (2-4) round the finger-wave loop. Reward per step: ALIVE + 0.5 mean + 0.5 min over balls of
    clip(ball's turn round the loop centre / REF, -1, 1) (each counted only beyond RIM) - W_CROWD_N * crowding
    - W_RATE |a|^2; any ball lost ends it with -DROP. Starts: the balls spaced evenly round the loop's path (a
    random offset, a little jitter), so with two balls one is on the finger row while the other comes back."""

    def __init__(self, n=2, episode_seconds=None, randomize=True, ref_deg=REF_DEG, w_edge=0.0, w_fast=0.0, flex_n=BASE_FLEX_N, drop=DROP):
        self.n, self.w_edge = n, w_edge          # w_edge: penalty per ball, ramping 0 -> w_edge from EDGE_X to EDGE_X - EDGE_W
        self.drop = drop                         # the cost of a lost ball
        self.w_fast = w_fast                     # w_fast: per ball below FAST_X, ramping 0 -> w_fast as -vx goes FAST_V -> +FAST_W
        self.flex_n, self.roll_n = flex_n, BASE_ROLL_N            # the base, per instance (script variants)
        self.wall_r = self.wall_l = BASE["wall"]                  # right / left index-finger end walls
        self.amp_k = np.ones(8)                                   # per-finger scale of the wave's lift
        super().__init__(episode_seconds=episode_seconds, randomize=randomize, ref_deg=ref_deg)
        nobs = 2 * len(self.hq) + 9 * n + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(nobs,), dtype=np.float32)
        x_row, x_palm = self.row[:, 0].mean() + 0.02, 0.285       # the loop's path: out along the row, back on the palm
        y0, y1 = self.row[1, 1], self.row[6, 1]
        self.path = np.array([[x_row, y0], [x_row, y1], [x_palm, y1], [x_palm, y0]])
        seg = np.linalg.norm(np.roll(self.path, -1, 0) - self.path, axis=1)
        self.seg_cum = np.concatenate([[0.0], np.cumsum(seg)])

    def _on_path(self, s):
        s = (s % 1.0) * self.seg_cum[-1]
        i = int(np.searchsorted(self.seg_cum, s, side="right") - 1)
        a = (s - self.seg_cum[i]) / (self.seg_cum[i + 1] - self.seg_cum[i])
        return (1 - a) * self.path[i] + a * self.path[(i + 1) % 4]

    def balls_n(self):
        return self.w.balls()[:self.n]

    def base(self):
        B, w = BASE, self.w
        u = w.u0.copy()
        u[self.fx] = self.flex_n
        u[self.ro] += self.roll_n
        u[self.kn[0]] += self.wall_r
        u[self.kn[-1]] += self.wall_l
        for k in range(1, 7):
            u[self.kn[k]] += self.amp_k[k] * B["amp"] * max(0.0, np.sin(2 * np.pi * (self.t_wave / B["T"] - k * B["lag"]))) ** 2
        return u

    def _angles(self, b):
        return np.arctan2(b[:, 1] - self.c[1], b[:, 0] - self.c[0])

    def _obs(self):
        b = self.balls_n()
        rel = b[:, :2] - self.c
        v = np.concatenate([self.d.qvel[q:q + 3] for q in self.w.bv[:self.n]])
        th = self._angles(b)
        ph = 2 * np.pi * self.t_wave / BASE["T"]
        return np.concatenate([
            self.d.qpos[self.hq], 0.1 * self.d.qvel[self.hv],
            10 * np.column_stack([rel, b[:, 2] - PALM_Z]).ravel(), v, np.cos(th), np.sin(th),
            10 * np.linalg.norm(rel, axis=1), [np.cos(ph), np.sin(ph)]
        ]).astype(np.float32)

    def _lost(self, b):
        return bool(np.any(b[:, 2] < LOST_Z) or np.any(np.linalg.norm(b[:, :2] - self.c, axis=1) > FAR))

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        rng, m, d, w = self.np_random, self.m, self.d, self.w
        mj.mj_setState(m, d, w.st0, stage7_wave.S)
        s0 = rng.random() if self.randomize else 0.1
        for k, name in enumerate(BALLS):
            if k < self.n:
                xy = self._on_path(s0 + k / self.n) + (rng.normal(0, 0.003, 2) if self.randomize else 0.0)
                set_ball(m, d, name, [xy[0], xy[1], PALM_Z + BALL_R + 0.01])
            else:
                set_ball(m, d, name, PARK[k - self.n])
        self.t_wave = rng.uniform(0, BASE["T"]) if self.randomize else 0.0
        for _ in range(10):
            d.ctrl[:] = np.clip(self.base(), w.lo, w.hi)
            for _ in range(self.n_sub):
                mj.mj_step(m, d)
            self.t_wave += CONTROL_DT
        self.t, self.steps = 0, int(round(self.seconds / CONTROL_DT))
        self.th_prev, self.turn = self._angles(self.balls_n()), np.zeros(self.n)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        self.d.ctrl[:] = np.clip(self.base() + self.res * a, self.w.lo, self.w.hi)
        for _ in range(self.n_sub):
            mj.mj_step(self.m, self.d)
        self.t += 1
        self.t_wave += CONTROL_DT
        b = self.balls_n()
        th = self._angles(b)
        r = np.linalg.norm(b[:, :2] - self.c, axis=1)
        dth = (r > RIM) * np.angle(np.exp(1j * (th - self.th_prev)))
        self.th_prev = th
        self.turn += dth
        prog = np.clip(dth / self.ref, -1.0, 1.0)
        g = np.array([np.linalg.norm(b[p, :2] - b[q, :2]) * 100 - 200 * BALL_R
                      for p in range(self.n) for q in range(p + 1, self.n)])
        reward = (ALIVE + 0.5 * prog.mean() + 0.5 * prog.min() - W_CROWD_N * float(np.sum(np.maximum(1.0 - g, 0.0)))
                  - W_RATE * float(np.sum(a ** 2))
                  - self.w_edge * float(np.sum(np.clip((EDGE_X - b[:, 0]) / EDGE_W, 0.0, 1.0)))
                  - self.w_fast * float(np.sum((b[:, 0] < FAST_X) * np.clip(
                      (-self.d.qvel[self.w.bv[:self.n]] - FAST_V) / FAST_W, 0.0, 1.0))))
        lost = self._lost(b)
        if lost:
            reward -= self.drop
        trunc = self.t >= self.steps
        info = {}
        if lost or trunc:
            deg = np.degrees(self.turn)
            info = dict(lost=lost, turned=float(deg.mean()), turned_min=float(deg.min()), seconds=self.t * CONTROL_DT)
        return self._obs(), reward, lost, trunc and not lost, info
