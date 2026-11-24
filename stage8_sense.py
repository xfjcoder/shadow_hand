"""Stage 8: baoding from senses a real hand has. The Stage 7A policies (the teachers) read each ball's exact place,
velocity and place on the 8 from the simulator. Here the balls are only SEEN (an overhead camera) and FELT (touch
pads: the normal force from the balls on each palm / finger segment), with noise and latency.

`Senses(env)`   the camera (a small RGB image, shadows / reflections off: 0.7 ms a frame on the CPU with EGL; 42 ms
                with shadows) and the touch pads (48 hand bodies, the summed normal force of ball contacts on each),
                each through its own delay line, with noise. The balls are recoloured red / green / blue / yellow so
                the camera can tell them apart (the Stage 7 scene has two pairs of the same colour).
`BallObserver`  the teacher's observation rebuilt from MEASURED ball positions instead of the simulator's: an
                alpha-beta filter (position and velocity), forward-predicted over the measurement's latency, each
                ball's place on the 8 tracked from the estimate, the timetable error from that. Its estimate also
                drives the per-ball tilt actions (Fig8Env.est), so nothing of the true ball state reaches the policy.
`run_episode`   a teacher episode on measured ball state, with a waiting timetable driven by the estimate too.
"""
from collections import deque

import mujoco as mj
import numpy as np

from rl_env_stage7_fig8 import WIN
from rl_env_stage7_wave import BASE, CONTROL_DT
from stage7_layout import PALM_Z
from stage7_teacher import BALLS

COLORS = np.array([[0.85, 0.15, 0.15, 1], [0.15, 0.7, 0.2, 1], [0.2, 0.35, 0.9, 1], [0.9, 0.8, 0.15, 1]])
CAM_DIST = 0.30                       # m above the crossing, looking straight down (the hands fill the frame)


def recolor(m):
    for k, name in enumerate(BALLS):
        g = m.geom(name).id
        m.geom_matid[g] = -1
        m.geom_rgba[g] = COLORS[k]


class Senses:
    def __init__(self, env, size=64, cam_delay=1, touch_delay=0, pix_noise=0.0, touch_noise=0.1, touch_floor=0.02,
                 seed=0):
        """cam_delay / touch_delay in control steps (20 ms); pix_noise: std of added pixel noise (0-255 scale);
        touch_noise: multiplicative std; touch_floor: N, readings below it read 0 (and additive std of floor / 2)."""
        self.env, m = env, env.m
        recolor(m)
        self.r = mj.Renderer(m, height=size, width=size)
        for f in ("SHADOW", "REFLECTION", "SKYBOX"):
            self.r.scene.flags[getattr(mj.mjtRndFlag, f"mjRND_{f}")] = 0
        self.cam = mj.MjvCamera()
        self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = [env.c8[0], env.c8[1], PALM_Z]
        self.cam.distance, self.cam.elevation, self.cam.azimuth = CAM_DIST, -90, 90
        self.pads = np.array(sorted({m.geom_bodyid[g] for g in range(m.ngeom)
                                     if m.body(m.geom_bodyid[g]).name.startswith(("rh_", "L_lh_"))}))
        self.pad_of = {b: i for i, b in enumerate(self.pads)}
        self.ball_b = {m.body(b).id for b in BALLS}
        self.cam_delay, self.touch_delay = cam_delay, touch_delay
        self.pix_noise, self.touch_noise, self.touch_floor = pix_noise, touch_noise, touch_floor
        self.rng = np.random.default_rng(seed)
        self._f6 = np.zeros(6)

    def reset(self):
        self.cams = deque(maxlen=self.cam_delay + 1)
        self.touches = deque(maxlen=self.touch_delay + 1)

    def image(self):
        self.r.update_scene(self.env.d, self.cam)
        return self.r.render()

    def touch(self):
        """N: each pad's summed normal force from the balls."""
        m, d = self.env.m, self.env.d
        out = np.zeros(len(self.pads))
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            pad = self.pad_of.get(b1) if b2 in self.ball_b else self.pad_of.get(b2) if b1 in self.ball_b else None
            if pad is not None:
                mj.mj_contactForce(m, d, i, self._f6)
                out[pad] += abs(self._f6[0])
        return out

    def read(self):
        """(image, touch) as the policy gets them now: delayed by cam_delay / touch_delay steps, with noise."""
        self.cams.append(self.image())
        self.touches.append(self.touch())
        img = self.cams[0]
        if self.pix_noise:
            img = np.clip(img + self.rng.normal(0, self.pix_noise, img.shape), 0, 255).astype(np.uint8)
        t = self.touches[0] * (1 + self.rng.normal(0, self.touch_noise, len(self.pads)))
        t = t + self.rng.normal(0, self.touch_floor / 2, len(self.pads))
        return img, np.where(t > self.touch_floor, t, 0.0)

    def close(self):
        self.r.close()


class BallObserver:
    """The teacher's observation from measured ball positions (n x 3, m), `latency` control steps old."""

    def __init__(self, env, latency=0, alpha=0.6, beta=0.2):
        self.env, self.latency, self.alpha, self.beta = env, latency, alpha, beta

    def reset(self, b):
        e = self.env
        self.b, self.v, self.s = b.copy(), np.zeros_like(b), e.s.copy()   # s at reset: the balls are placed on it

    def update(self, meas):
        """meas: the measured positions (or None: no measurement, coast). -> the estimate now."""
        pred = self.b + self.v * CONTROL_DT
        if meas is None:
            self.b = pred
        else:
            r = meas - pred
            self.b = pred + self.alpha * r
            self.v = self.v + self.beta * r / CONTROL_DT
        now = self.b + self.v * CONTROL_DT * self.latency               # predicted over the latency
        e = self.env
        s, dist = np.empty(e.n), np.empty(e.n)
        for i in range(e.n):
            cand = self.s[i] + WIN
            dd = np.linalg.norm(e.fig8(cand) - now[i, :2], axis=1)
            j = int(np.argmin(dd))
            s[i], dist[i] = cand[j], dd[j]
        self.s, self.dist, self.now = s, dist, now
        e.est = (now, s)
        return now

    def ref_err(self):
        e = self.env
        ph = np.interp(self.s % 1.0, e.pace_s, e.pace_ph) if e.pace else self.s
        return np.angle(np.exp(2j * np.pi * (ph - e.ref - e.spacing()))) / (2 * np.pi)

    def obs(self):
        """As Fig8Env._obs, every ball term from the estimate."""
        e, b = self.env, self.now
        rel = b[:, :2] - e.c8
        ph = 2 * np.pi * e.t_wave / BASE["T"]
        return np.concatenate([
            e.d.qpos[e.hq], 0.1 * e.d.qvel[e.hv],
            10 * np.column_stack([rel, b[:, 2] - PALM_Z]).ravel(), self.v.ravel(), np.cos(2 * np.pi * self.s),
            np.sin(2 * np.pi * self.s), 10 * self.dist, [np.cos(ph), np.sin(ph)]]
            + ([10 * self.ref_err(), [np.cos(2 * np.pi * e.ref), np.sin(2 * np.pi * e.ref)]] if e.track_rate else [])
        ).astype(np.float32)


def run_episode(env, policy, measure, observer, seed, lead=0.1, deterministic=True, on_step=None):
    """A teacher episode on measured ball state. measure(env) -> the measured ball positions (n x 3) or None;
    policy(obs) -> action. The timetable waits for the ESTIMATED balls (lead laps), as Fig8Env(lead) does for the
    true ones. -> info at the end, and per-step (true gap to the nearest ball, true |slot error|, estimate error cm)."""
    env.lead = None
    env.reset(seed=seed)
    env.est = None
    observer.reset(env.balls_n())
    observer.update(measure(env))
    log = []
    while True:
        o = observer.obs()
        _, r, te, tr, info = env.step(policy(o))
        observer.update(measure(env))
        if env.t * CONTROL_DT > env.grace:
            ahead = -np.angle(np.mean(np.exp(2j * np.pi * observer.ref_err()))) / (2 * np.pi)
            env.ref -= max(ahead - lead, 0.0)
            b = env.balls_n()
            log.append((env._gaps(b).min(), np.abs(env._ref_err()).max(),
                        np.linalg.norm(observer.now - b, axis=1).max() * 100))
        if on_step:
            on_step(env)
        if te or tr:
            env.est = None
            return info, np.array(log).reshape(-1, 3)
