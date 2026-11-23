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
- W_CROWD_N * crowding - W_RATE |a|^2 - w_spread * mean over balls of min(|slot error| / SPREAD_TOL, 1)
(each ball's s against its start offset from the others, `_slot_err`: keeps them spread round the 8 at their
offsets instead of riding as a clump); a ball lost ends it with -drop.

Timetable mode (track_rate, laps/s): the progress terms are replaced by (1 + cos 2 pi err) / 2 per ball, err = the
ball's s against its slot on a timetable that runs round the 8 at track_rate (ref + offset_k, ref starting at the
balls' own mean place): spread AND moving, a stalled ball falls out of its slot. With slot_tol, a ball more than
slot_tol laps out of its slot ends the episode (no cost beyond the pay forgone: keeping time is staying alive). (A narrow exp(-(err / 0.05)^2)
taught nothing in 4.5M steps: from the clump nearly every ball is a quarter lap out, where it pays ~0.) The obs gets each ball's
timetable error and the timetable's phase. With grace (s), the first grace seconds are a spin-up: the
timetable sits on the balls' own mean place (only their spacing counts) and nobody is late; then it runs at
track_rate from there (balls start at rest, and at full pace from t = 0 half the episodes ended late in 3 s).
With late_cost, a late end costs that (free, past what a policy can hold, slow and spaced was the safe play: fig8_16, fig8_19 slowed after tightening). With pace (the balls' natural pace over the 8), the timetable's clock runs at that pace: slots even in time, not in s. With bank, a share p_bank of episodes starts mid-run from a good policy's states instead (past the spin-up). In the spin-up the slot tolerance still applies (to spacing: the timetable is on the balls) and a ball is paid only as far as it moves at the timetable's pace (paid for spacing alone, fig8_16
learned at slot_tol 0.1, where episodes end soon after it, to keep the balls still and spaced through it).
With lead (laps), the timetable waits for the balls: it never runs more than lead ahead of their mean place. For
running a policy trained with slot_tol: it never saw errors past slot_tol, and the error is wrapped, so a ball
over half a lap behind reads as ahead; without lead the balls then wait a whole lap for the timetable.
With clear (cm), each ball's timetable pay scales with its surface gap to the nearest ball, 0 touching, full at clear
(the crowd cost alone, 0.05 a step against pay up to 1, left fig8_22's balls touching 23% of the time; exact slots
clear 1.6-1.8 cm). With touch_end, two balls touching ends the episode as a late ball does.
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
SPREAD_TOL = 0.1                          # laps out of its slot at which a ball costs w_spread in full
TOUCH = 0.1                               # cm: two balls this close count as touching
TILT_K = 0.1                              # rad of hand tilt per unit of a per-ball action (ball_act; default)
# (at the outer walls 0.1 moves a ball ~0.008 laps in 0.5 s, too weak to unstall it; 0.3 moves it 0.04-0.08)
SPEC_BANK = mj.mjtState.mjSTATE_INTEGRATION       # what a bank state (bank_stage7_fig8.py) holds
SEAM_W = 0.005                            # m: a ball within ~this of the seam's y tilts both hands, blended


class Fig8Env(MultiWaveEnv):
    def __init__(self, n=1, episode_seconds=None, randomize=True, ref_deg=REF_DEG, drop=DROP, flex_n=FLEX_8,
                 w_spread=0.0, track_rate=None, slot_tol=None, ball_act=False, grace=0.0, tilt_k=TILT_K, lead=None, late_cost=0.0, bank=None, p_bank=0.0, pace=None, clear=None, touch_end=False, ball_r=BALL_R):
        super().__init__(n=n, episode_seconds=episode_seconds, randomize=randomize, ref_deg=ref_deg, drop=drop,
                         flex_n=flex_n, ball_r=ball_r)
        self.roll_n = 0.0
        self.w_spread = w_spread
        self.track_rate = track_rate                 # laps/s: a timetable for the balls instead of paying progress
        self.slot_tol = slot_tol                     # laps: timetable mode, any ball further out of its slot ends it
        self.tilt_k = tilt_k                         # rad of hand tilt per unit of a per-ball action
        self.bank = dict(np.load(bank)) if bank else None   # mid-run states (bank_stage7_fig8.py): a share
        if pace is not None:                         # the balls' natural pace over s in [0, 1/2) (the 8 is
            v = np.tile(np.load(pace) if isinstance(pace, str) else np.asarray(pace), 2)   # symmetric), equal bins
            c = np.concatenate([[0.0], np.cumsum(1.0 / v)])
            self.pace_s, self.pace_ph = np.linspace(0, 1, len(c)), c / c[-1]
        self.pace = pace is not None                 # timetable in natural-pace phase (slots even in time)
        self.p_bank = p_bank                         # p_bank of episodes starts from one, past the spin-up
        self.late_cost = late_cost                   # a ball out of its slot (slot_tol) ends it with -late_cost
        self.clear = clear                           # cm: a ball's pay scales with its gap to the nearest ball, full at clear
        self.touch_end = touch_end                   # two balls touching ends it (like a late ball)
        self.lead = lead                             # laps: the timetable waits, never leading the balls by more
        self.grace = grace                           # s: timetable spin-up, it follows the balls
        self.ball_act = ball_act                     # n more actions: speed each ball along the 8 by tilting its hand
        if ball_act:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(self.m.nu + n,), dtype=np.float32)
            self.tilt_x = [self.m.actuator(f"{h}_A_WRJ1").id for h in ("rh", "L_lh")]   # palm normal -x per rad
            self.tilt_y = [self.m.actuator(f"{h}_A_FRJ").id for h in ("rh", "L_lh")]    # palm normal -y per rad
        if track_rate:
            n_obs = self.observation_space.shape[0] + n + 2
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
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
        Offsets distinct mod 1/2 (k / 2n apart), every other one on the other lobe: they cross by turns.
        Four balls that way pass 2.8 cm apart (they are 3.8 cm); the offsets below (a grid search for the
        widest closest pass, moving at a constant rate in s) keep them 4.0 cm apart. Two balls go as near half a
        lap apart as clears the crossing: 0.4 (closest pass 4.4 cm; at 0.45 it is 2.3 cm), not 1/4."""
        if self.n == 2:
            return np.array([0.0, 0.4])
        if self.n == 4:
            return np.array([0.0, 0.396, 0.604, 0.792])
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

    def tangent(self, s):
        """The 8's direction of travel at s (unit, in the plane)."""
        t = np.stack([self.a8[0] * 4 * np.pi * np.cos(4 * np.pi * s), self.a8[1] * 2 * np.pi * np.cos(2 * np.pi * s)], -1)
        return t / np.linalg.norm(t, axis=-1, keepdims=True)

    def _tilt(self, u):
        """Per-ball actions -> each hand's wrist flex and forearm roll: tilt the palm so gravity pushes each ball on
        it along the 8 (u > 0 forward). A tilt moves every ball on the hand, but balls on the finger leg and the
        palm leg run opposite ways, so one tilt can speed one and hold back the other (probe_stage7_fig8.py).
        Calibrated: +1 rad of WRJ1 tilts the palm normal ~-1 in x, +1 rad of FRJ ~-0.97 in y, either hand."""
        b = self.balls_n()
        left = 1.0 / (1.0 + np.exp(-(b[:, 1] - self.c8[1]) / SEAM_W))      # the left hand is y > the seam
        v = u[:, None] * self.tangent(self.s % 1.0)
        du = np.zeros(self.m.nu)
        for h, w in ((0, 1.0 - left), (1, left)):                          # rh = 0, L_lh = 1 as in tilt_x / tilt_y
            n = self.tilt_k * (w[:, None] * v).sum(0)
            du[self.tilt_x[h]] -= n[0]
            du[self.tilt_y[h]] -= n[1] / 0.97
        return du

    def _track(self, b):
        """Each ball's s near its last one, and its distance from the 8 there."""
        s, dist = np.empty(self.n), np.empty(self.n)
        for i in range(self.n):
            cand = self.s[i] + WIN
            dd = np.linalg.norm(self.fig8(cand) - b[i, :2], axis=1)
            j = int(np.argmin(dd))
            s[i], dist[i] = cand[j], dd[j]
        return s, dist

    def _slot_err(self):
        """Each ball's place on the 8 against its slot: s_k - offset_k about their circular mean, in laps (0 when
        the balls keep their start spacing, however fast they go)."""
        if self.n < 2:
            return np.zeros(self.n)
        ph = 2 * np.pi * (self.s - self.spacing())
        m = np.angle(np.mean(np.exp(1j * ph)))
        return np.angle(np.exp(1j * (ph - m))) / (2 * np.pi)

    def _phase(self):
        """Each ball's place on the timetable's clock: s itself, or with pace its natural-pace phase (balls go
        round the outer walls ~5x slower than through the crossing; on a clock even in s that alone swings each
        ball +-0.09 lap in and out of its slot every lap)."""
        return np.interp(self.s % 1.0, self.pace_s, self.pace_ph) if self.pace else self.s

    def _mean_place(self):
        return np.angle(np.mean(np.exp(2j * np.pi * (self._phase() - self.spacing())))) / (2 * np.pi)

    def _ref_err(self):
        """Timetable mode: each ball's s against its slot on the timetable (ref + offset_k), in laps, wrapped."""
        return np.angle(np.exp(2j * np.pi * (self._phase() - self.ref - self.spacing()))) / (2 * np.pi)

    def _gaps(self, b):
        """cm: each ball's surface gap to its nearest other ball (centres in 3D)."""
        if self.n < 2:
            return np.full(1, np.inf)
        D = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=2) * 100 - 200 * self.ball_r
        np.fill_diagonal(D, np.inf)
        return D.min(1)

    def _obs(self):
        b = self.balls_n()
        rel = b[:, :2] - self.c8
        v = np.concatenate([self.d.qvel[q:q + 3] for q in self.w.bv[:self.n]])
        ph = 2 * np.pi * self.t_wave / BASE["T"]
        return np.concatenate([
            self.d.qpos[self.hq], 0.1 * self.d.qvel[self.hv],
            10 * np.column_stack([rel, b[:, 2] - PALM_Z]).ravel(), v, np.cos(2 * np.pi * self.s),
            np.sin(2 * np.pi * self.s), 10 * self.dist, [np.cos(ph), np.sin(ph)]]
            + ([10 * self._ref_err(), [np.cos(2 * np.pi * self.ref), np.sin(2 * np.pi * self.ref)]] if self.track_rate else [])
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        rng, m, d, w = self.np_random, self.m, self.d, self.w
        if self.bank is not None and rng.random() < self.p_bank:
            i = rng.integers(len(self.bank["ref"]))
            mj.mj_setState(m, d, self.bank["state"][i], SPEC_BANK)
            self.t_wave, self.s = float(self.bank["t_wave"][i]), self.bank["s"][i].copy()
            self.dist, self.ref = self.bank["dist"][i].copy(), float(self.bank["ref"][i])
            if self.pace:                            # the bank's clock was even in s
                self.ref = self._mean_place()
            self.steps = int(round(self.seconds / CONTROL_DT))
            self.t = int(np.ceil(self.grace / CONTROL_DT)) + 1         # past the spin-up: the timetable runs
            self.turn, self.n_touch = np.zeros(self.n), 0
            return self._obs(), {}
        mj.mj_setState(m, d, w.st0, stage7_wave.S)
        s0 = rng.random() if self.randomize else 0.1
        self.s = (s0 + self.spacing()) % 1.0
        for k, name in enumerate(BALLS):
            if k < self.n:
                xy = self.fig8(self.s[k]) + (rng.normal(0, 0.003, 2) if self.randomize else 0.0)
                set_ball(m, d, name, [xy[0], xy[1], PALM_Z + self.ball_r + 0.01])
            else:
                set_ball(m, d, name, PARK[k - self.n])
        self.t_wave = rng.uniform(0, BASE["T"]) if self.randomize else 0.0
        for _ in range(10):
            d.ctrl[:] = np.clip(self.base(), w.lo, w.hi)
            for _ in range(self.n_sub):
                mj.mj_step(m, d)
            self.t_wave += CONTROL_DT
        self.s, self.dist = self._track(self.balls_n())
        self.ref = self._mean_place()
        self.t, self.steps = 0, int(round(self.seconds / CONTROL_DT))
        self.n_touch = 0
        self.turn = np.zeros(self.n)                 # radians: 2 pi per lap of the 8 (as the loop envs report turns)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        u = self.base() + (self._tilt(a[self.m.nu:]) if self.ball_act else 0.0)
        a = a[:self.m.nu]
        self.d.ctrl[:] = np.clip(u + self.res * a, self.w.lo, self.w.hi)
        for _ in range(self.n_sub):
            mj.mj_step(self.m, self.d)
        self.t += 1
        self.t_wave += CONTROL_DT
        b = self.balls_n()
        s, self.dist = self._track(b)
        ds = s - self.s
        self.s = s
        self.turn += 2 * np.pi * ds
        if self.track_rate:
            spin = self.t * CONTROL_DT <= self.grace
            self.ref = self._mean_place() if spin else self.ref + self.track_rate * CONTROL_DT
            if self.lead and not spin:               # waiting timetable (for running, not training): keeps the
                ahead = -np.angle(np.mean(np.exp(2j * np.pi * self._ref_err()))) / (2 * np.pi)   # policy's
                self.ref -= max(ahead - self.lead, 0.0)                                          # error in range
            prog = 0.5 * (1 + np.cos(2 * np.pi * self._ref_err()))   # on the timetable 1, a quarter lap out 1/2
            if spin:                                 # spin-up: spaced AND getting up to pace (spaced and still paid
                prog = prog * np.clip(ds / (self.track_rate * CONTROL_DT), 0.0, 1.0)   # in full: fig8_16 stopped)
        else:
            prog = np.clip(ds / self.ref_s, -1.0, 1.0)
        gk = self._gaps(b)                            # cm, each ball's surface gap to its nearest ball
        touch = bool(self.n > 1 and gk.min() < TOUCH)
        self.n_touch += touch
        if self.track_rate and self.clear:
            prog = prog * np.clip(gk / self.clear, 0.0, 1.0)
        off = np.clip((self.dist - OFF0) / OFF_W, 0.0, 1.0)
        g = np.array([np.linalg.norm(b[p, :2] - b[q, :2]) * 100 - 200 * self.ball_r
                      for p in range(self.n) for q in range(p + 1, self.n)])
        err = self._slot_err()
        reward = (ALIVE + 0.5 * prog.mean() + 0.5 * prog.min() - W_OFF * float(off.sum())
                  - self.w_spread * float(np.mean(np.minimum(np.abs(err) / SPREAD_TOL, 1.0)))
                  - W_CROWD_N * float(np.sum(np.maximum(1.0 - g, 0.0))) - W_RATE * float(np.sum(a ** 2)))
        lost = self._lost(b)
        if lost:
            reward -= self.drop
        late = bool(self.track_rate and self.slot_tol and np.any(np.abs(self._ref_err()) > self.slot_tol))   # in the
        # spin-up the timetable sits on the balls, so this is spacing alone (unchecked, fig8_18 clumped through it)
        if late and not lost:
            reward -= self.late_cost
        late = late or (self.touch_end and touch and not lost)
        trunc = self.t >= self.steps
        info = {}
        if lost or late or trunc:
            deg = np.degrees(self.turn)
            info = dict(lost=lost, turned=float(deg.mean()), turned_min=float(deg.min()), seconds=self.t * CONTROL_DT,
                        off=float(np.mean(self.dist)), spread=float(np.mean(np.abs(err))),
                        **(dict(ref_err=float(np.mean(np.abs(self._ref_err()))), late=late) if self.track_rate else {}),
                        touch=self.n_touch / max(self.t, 1), touch_end=bool(touch and not lost))
        return self._obs(), reward, lost or late, trunc and not (lost or late), info
