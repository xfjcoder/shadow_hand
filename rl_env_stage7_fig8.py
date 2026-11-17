"""Stage 7A, pattern A with crossing: the balls run a FIGURE-8 across both cupped hands (rl_env_stage7_wave's long
cup and finger wave). One lobe round each hand, the two lobes turned opposite ways; the paths cross at the
centre, where the hands meet, so balls going from one hand to the other cross there.

The path is a 1:2 Lissajous figure-8 over the hands (`fig8(s)`, s in [0, 1)): from the centre out across the
left hand's fingers to its outer edge, back along its palm to the centre, then out across the right hand's
fingers and back along its palm. On the finger side the balls move OUTWARD (away from the seam), so the base's
finger wave runs outward from the seam on both hands; walls (index fingers) at the outer ends as before, the
wrists flexed back -0.05 (rl_env_stage7_wave: that keeps balls off the palms' heel). The policy is the same
residual on every joint and steers the tilts (each hand's forearm roll on its own: a valley at the seam brings
balls back inward on the palms).

Each ball's place on the 8 is tracked (`s`, searched near its last value, so the crossing is not ambiguous).
Reward per step: ALIVE + 0.5 mean + 0.5 min over balls of clip(ds / REF, -1, 1) - W_OFF per ball off the path
(ramping from OFF0 to OFF0 + OFF_W away; going round one hand's lobe and skipping the crossing is off the path)
- W_CROWD_N * crowding - W_RATE |a|^2; a ball lost ends it with -drop.
"""
import gymnasium as gym
import mujoco as mj
import numpy as np
from gymnasium import spaces

import stage7_wave
from rl_env_stage7_wave import (ALIVE, BALL_R, BASE, CONTROL_DT, DROP, REF_DEG, W_CROWD_N, W_RATE, MultiWaveEnv)
from stage7_layout import PALM_Z, set_ball
from stage7_teacher import BALLS
from stage7_wave import PARK

FLEX_8 = -0.05
OFF0, OFF_W, W_OFF = 0.015, 0.015, 0.3     # m off the 8 before it costs, ramp, per-ball cost at full
WIN = np.linspace(-0.04, 0.08, 61)        # where to look for a ball's s next step (fractions of the 8)


class Fig8Env(MultiWaveEnv):
    def __init__(self, n=1, episode_seconds=None, randomize=True, ref_deg=REF_DEG, drop=DROP, flex_n=FLEX_8):
        super().__init__(n=n, episode_seconds=episode_seconds, randomize=randomize, ref_deg=ref_deg, drop=drop,
                         flex_n=flex_n)
        self.roll_n = 0.0
        x_row, x_palm = self.row[:, 0].mean() + 0.02, 0.285
        y0, y1 = self.row[1, 1], self.row[6, 1]
        self.c8 = np.array([(x_row + x_palm) / 2, (y0 + y1) / 2])       # the crossing
        self.a8 = np.array([(x_row - x_palm) / 2, (y1 - y0) / 2])
        self.ref_s = ref_deg / 360.0 * CONTROL_DT                        # REF_DEG/360 laps of the 8 per second
        g = np.linspace(0, 1, 2001)[:-1]
        p = self.fig8(g)
        self.length = float(np.sum(np.linalg.norm(np.roll(p, -1, 0) - p, axis=1)))
        # outward from the seam: finger k (1..6 on the row) lifts lag * (its place from the seam) later
        self.out = np.array([0, 2, 1, 0, 0, 1, 2, 0])

    def spacing(self):
        """Start offsets: the 8 passes the crossing at s = 0 and 1/2, so balls half a lap apart would meet there.
        Offsets distinct mod 1/2 (k / 2n apart), every other one on the other lobe: they cross by turns."""
        k = np.arange(self.n)
        return k * 0.5 / self.n + (k % 2) * 0.5

    def fig8(self, s):
        s = np.asarray(s)
        return np.stack([self.c8[0] + self.a8[0] * np.sin(4 * np.pi * s),
                         self.c8[1] + self.a8[1] * np.sin(2 * np.pi * s)], -1)

    def base(self):
        B, w = BASE, self.w
        u = w.u0.copy()
        u[self.fx] = self.flex_n
        u[self.ro] += self.roll_n
        u[self.kn[0]] += self.wall_r
        u[self.kn[-1]] += self.wall_l
        for k in range(1, 7):
            u[self.kn[k]] += B["amp"] * max(0.0, np.sin(2 * np.pi * (self.t_wave / B["T"] - self.out[k] * B["lag"]))) ** 2
        return u

    def _track(self, b):
        """Each ball's s near its last one, and its distance from the 8 there."""
        s, dist = np.empty(self.n), np.empty(self.n)
        for i in range(self.n):
            cand = self.s[i] + WIN
            dd = np.linalg.norm(self.fig8(cand) - b[i, :2], axis=1)
            j = int(np.argmin(dd))
            s[i], dist[i] = cand[j], dd[j]
        return s, dist

    def _obs(self):
        b = self.balls_n()
        rel = b[:, :2] - self.c8
        v = np.concatenate([self.d.qvel[q:q + 3] for q in self.w.bv[:self.n]])
        ph = 2 * np.pi * self.t_wave / BASE["T"]
        return np.concatenate([
            self.d.qpos[self.hq], 0.1 * self.d.qvel[self.hv],
            10 * np.column_stack([rel, b[:, 2] - PALM_Z]).ravel(), v, np.cos(2 * np.pi * self.s),
            np.sin(2 * np.pi * self.s), 10 * self.dist, [np.cos(ph), np.sin(ph)]
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        rng, m, d, w = self.np_random, self.m, self.d, self.w
        mj.mj_setState(m, d, w.st0, stage7_wave.S)
        s0 = rng.random() if self.randomize else 0.1
        self.s = (s0 + self.spacing()) % 1.0
        for k, name in enumerate(BALLS):
            if k < self.n:
                xy = self.fig8(self.s[k]) + (rng.normal(0, 0.003, 2) if self.randomize else 0.0)
                set_ball(m, d, name, [xy[0], xy[1], PALM_Z + BALL_R + 0.01])
            else:
                set_ball(m, d, name, PARK[k - self.n])
        self.t_wave = rng.uniform(0, BASE["T"]) if self.randomize else 0.0
        for _ in range(10):
            d.ctrl[:] = np.clip(self.base(), w.lo, w.hi)
            for _ in range(self.n_sub):
                mj.mj_step(m, d)
            self.t_wave += CONTROL_DT
        self.s, self.dist = self._track(self.balls_n())
        self.t, self.steps = 0, int(round(self.seconds / CONTROL_DT))
        self.turn = np.zeros(self.n)                 # radians: 2 pi per lap of the 8 (as the loop envs report turns)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        self.d.ctrl[:] = np.clip(self.base() + self.res * a, self.w.lo, self.w.hi)
        for _ in range(self.n_sub):
            mj.mj_step(self.m, self.d)
        self.t += 1
        self.t_wave += CONTROL_DT
        b = self.balls_n()
        s, self.dist = self._track(b)
        ds = s - self.s
        self.s = s
        self.turn += 2 * np.pi * ds
        prog = np.clip(ds / self.ref_s, -1.0, 1.0)
        off = np.clip((self.dist - OFF0) / OFF_W, 0.0, 1.0)
        g = np.array([np.linalg.norm(b[p, :2] - b[q, :2]) * 100 - 200 * BALL_R
                      for p in range(self.n) for q in range(p + 1, self.n)])
        reward = (ALIVE + 0.5 * prog.mean() + 0.5 * prog.min() - W_OFF * float(off.sum())
                  - W_CROWD_N * float(np.sum(np.maximum(1.0 - g, 0.0))) - W_RATE * float(np.sum(a ** 2)))
        lost = self._lost(b)
        if lost:
            reward -= self.drop
        trunc = self.t >= self.steps
        info = {}
        if lost or trunc:
            deg = np.degrees(self.turn)
            info = dict(lost=lost, turned=float(deg.mean()), turned_min=float(deg.min()), seconds=self.t * CONTROL_DT,
                        off=float(np.mean(self.dist)))
        return self._obs(), reward, lost, trunc and not lost, info
