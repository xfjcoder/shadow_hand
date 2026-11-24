"""Stage 8 step 5: a student that does three-ball baoding from senses alone, distilled from the teacher (fig8_27
@104M) with DAgger. The student sees, each 20 ms step:
  - the tracker's ball positions (train_stage8_tracker.py) from the last K camera frames, the camera cam_delay steps
    late (a history: it can learn the balls' velocity and the latency itself; the teacher broke on filtered
    finite-difference velocity and on 20-40 ms of delay, eval_stage8_noise.py), and whether each frame is fresh
  - the touch pads (touch_delay late, noisy), log-compressed
  - the hand's joint angles and velocities, the rhythm's phase and the timetable's phase
and outputs the hand's residual (42) and the four tilt offsets (both wrists' flex and both forearms' roll, which
the teacher's per-ball actions map to through the TRUE ball state: Fig8Env._tilt; the student gives them directly,
Fig8Env.du). Labels are the teacher's deterministic actions from the true state, on every state visited.
DAgger: iteration 0 the teacher drives; iteration k each step is the teacher's with probability beta0 * 0.5^k,
else the student's; all data is kept and the student retrained on it. The timetable waits for the balls as
estimated from the tracker (BallObserver), never the true state.

Step 6 ablations: --cam-drop P (each camera frame lost with probability P: the last estimate is held, the fresh
flag 0), --no-cam (touch and the hand only), --no-touch.

    MUJOCO_GL=egl python3 train_stage8_student.py --tracker runs/stage8_tracker/tracker.pt --run-name student_01
"""
import argparse
import json
import os
import time
from collections import deque
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import torch.nn as nn

from eval_stage8_noise import TEACHER, load
from eval_stage8_tracker import CKPT, Measure, load_tracker
from rl_env_stage7_fig8 import Fig8Env
from rl_env_stage7_wave import BASE, CONTROL_DT
from stage8_sense import BallObserver, Senses


class StudentSense:
    """The student's input vector, built from the camera (through the tracker), the touch pads and the hand."""

    def __init__(self, env, net, origin, cfg, seed):
        self.env, self.cfg = env, cfg
        self.sen = Senses(env, size=net.coords.shape[-1], cam_delay=cfg["cam_delay"], touch_delay=cfg["touch_delay"],
                          pix_noise=cfg["pix_noise"], touch_noise=cfg["touch_noise"], seed=seed)
        self.meas = Measure(self.sen, net, origin)
        self.obs = BallObserver(env, latency=cfg["cam_delay"])          # for the waiting timetable only
        self.rng = np.random.default_rng(seed + 7)
        self.origin = origin

    def reset(self):
        self.sen.reset()
        self.hist = deque(maxlen=self.cfg["k"])
        self.obs.reset(self.env.balls_n())
        self.last = None

    def step(self):
        """Read the senses now -> the student's input vector."""
        e, cfg = self.env, self.cfg
        img, touch = self.sen.read()
        self.img, self.touch = img, touch
        fresh = self.last is None or self.rng.random() >= cfg["cam_drop"]
        if fresh:
            with torch.no_grad():
                x = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
                y = self.meas.net(x, torch.from_numpy(touch.astype(np.float32))[None])[0][0].numpy()
            self.last = y
        self.obs.update(self.last / 100.0 + self.origin)
        e.est = None                                  # the teacher's labels and its tilts use the true state
        self.hist.append(np.concatenate([self.last.ravel() / 10.0, [float(fresh)]]))
        while len(self.hist) < cfg["k"]:
            self.hist.append(self.hist[-1])
        ph = 2 * np.pi * e.t_wave / BASE["T"]
        parts = [e.d.qpos[e.hq], 0.1 * e.d.qvel[e.hv], [np.cos(ph), np.sin(ph), np.cos(2 * np.pi * e.ref),
                                                       np.sin(2 * np.pi * e.ref)]]
        if not cfg["no_cam"]:
            parts.append(np.concatenate(self.hist))
        if not cfg["no_touch"]:
            parts.append(np.log1p(touch))
        return np.concatenate(parts).astype(np.float32)

    def wait_clock(self, lead=0.1):
        e = self.env
        if e.t * CONTROL_DT > e.grace:
            ahead = -np.angle(np.mean(np.exp(2j * np.pi * self.obs.ref_err()))) / (2 * np.pi)
            e.ref -= max(ahead - lead, 0.0)


class Student(nn.Module):
    def __init__(self, n_in, n_out, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean))
        self.register_buffer("std", torch.as_tensor(std))
        self.net = nn.Sequential(nn.Linear(n_in, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, n_out))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def tilt_ids(env):
    return np.array(env.tilt_x + env.tilt_y)


def rollout(a):
    """One episode: (inputs, teacher labels, summary)."""
    cfg, seed, beta, student_path, record = a
    torch.set_num_threads(1)
    net, origin = load_tracker(cfg["tracker"])
    model, vn = load(cfg["teacher"])
    env = Fig8Env(**TEACHER)
    env.lead = None
    sense = StudentSense(env, net, origin, cfg, seed)
    student = None
    if student_path and beta < 1.0:
        ck = torch.load(student_path, weights_only=False)
        student = Student(ck["n_in"], ck["n_out"], ck["mean"], ck["std"])
        student.load_state_dict(ck["state"])
        student.eval()
    rng = np.random.default_rng(seed + 13)
    env.reset(seed=seed)
    env.est, env.du = None, None
    sense.reset()
    nu, ti = env.m.nu, tilt_ids(env)
    X, Y, G, E, frames = [], [], [], [], []
    while True:
        x = sense.step()
        at = model.predict(vn.normalize_obs(env._obs()), deterministic=True)[0]
        du = env._tilt(np.clip(at[nu:], -1, 1))
        y = np.concatenate([np.clip(at[:nu], -1, 1), du[ti] / env.tilt_k]).astype(np.float32)
        X.append(x), Y.append(y)
        if student is None or rng.random() < beta:
            env.du = None
            act = at
        else:
            with torch.no_grad():
                p = student(torch.from_numpy(x)[None])[0].numpy()
            env.du = np.zeros(nu)
            env.du[ti] = p[nu:] * env.tilt_k
            act = np.concatenate([p[:nu], np.zeros(env.n)])
        _, r, te, tr, info = env.step(act)
        sense.wait_clock()
        if record:
            frames.append((env.d.qpos.copy(), env.balls_n().copy(), sense.img.copy(), sense.touch.copy(),
                           sense.last / 100.0 + sense.origin))
        if env.t * CONTROL_DT > env.grace:
            b = env.balls_n()
            G.append(env._gaps(b).min()), E.append(np.abs(env._ref_err()).max())
        if te or tr:
            break
    sense.sen.close()
    summ = dict(kept=not info["lost"], laps=float(env.turn.min() / (2 * np.pi)), seconds=info["seconds"],
                touch=float(np.mean(np.array(G) < 0.1)) if G else 1.0, err=float(np.median(E)) if E else 0.5)
    return np.array(X), np.array(Y), summ, frames


def fit(X, Y, init, epochs, out, n_in, n_out, threads=16):
    torch.set_num_threads(threads)
    mean, std = X.mean(0), X.std(0) + 1e-3
    net = Student(n_in, n_out, mean, std)
    if init is not None:                              # warm start (the normalisation is refitted)
        net.net.load_state_dict(init.net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        L = []
        for i in range(0, len(idx), 1024):
            j = torch.from_numpy(idx[i:i + 1024])
            loss = ((net(Xt[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            L.append(loss.item())
    torch.save({"state": net.state_dict(), "n_in": n_in, "n_out": n_out, "mean": mean, "std": std}, out)
    return net, float(np.mean(L))


def summarise(S):
    return (f"kept {sum(s['kept'] for s in S)}/{len(S)}  laps/30 s {np.mean([s['laps'] for s in S]):.2f}  "
            f"seconds median {np.median([s['seconds'] for s in S]):.1f}  slot err median {np.median([s['err'] for s in S]):.3f}  "
            f"touching {np.mean([s['touch'] for s in S]):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--teacher", default=CKPT)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--eval-episodes", type=int, default=40)
    ap.add_argument("--beta0", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--k", type=int, default=4, help="camera frames of history")
    ap.add_argument("--cam-delay", type=int, default=2, help="control steps (20 ms)")
    ap.add_argument("--touch-delay", type=int, default=1)
    ap.add_argument("--pix-noise", type=float, default=3.0)
    ap.add_argument("--touch-noise", type=float, default=0.1)
    ap.add_argument("--cam-drop", type=float, default=0.0)
    ap.add_argument("--no-cam", action="store_true")
    ap.add_argument("--no-touch", action="store_true")
    ap.add_argument("--procs", type=int, default=15)
    args = ap.parse_args()
    cfg = dict(tracker=args.tracker, teacher=args.teacher, k=args.k, cam_delay=args.cam_delay,
               touch_delay=args.touch_delay, pix_noise=args.pix_noise, touch_noise=args.touch_noise,
               cam_drop=args.cam_drop, no_cam=args.no_cam, no_touch=args.no_touch)
    run = Path("runs") / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(run / "args.json", "w"), indent=1)
    XS, YS, net, path = [], [], None, None
    with Pool(args.procs) as P:
        for it in range(args.iters + 1):
            beta = 1.0 if it == 0 else args.beta0 * 0.5 ** (it - 1)
            t0 = time.time()
            R = P.map(rollout, [(cfg, 30000 + 1000 * it + i, beta, path, False) for i in range(args.episodes)])
            XS += [r[0] for r in R]
            YS += [r[1] for r in R]
            X, Y = np.concatenate(XS), np.concatenate(YS)
            path = str(run / f"student_{it}.pt")
            net, loss = fit(X, Y, net, args.epochs, path, X.shape[1], Y.shape[1])
            E = P.map(rollout, [(cfg, 7000 + i, 0.0, path, False) for i in range(args.eval_episodes)])
            print(f"iter {it} (beta {beta:.3f}, data {len(X)}): rollouts {summarise([r[2] for r in R])} | fit loss "
                  f"{loss:.4f} | STUDENT ALONE: {summarise([r[2] for r in E])}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
