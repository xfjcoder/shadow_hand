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
