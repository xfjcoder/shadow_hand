"""Stage 8 step 4: the teacher on the tracker. Each step the camera image (cam_delay steps old) and the touch pads
(touch_delay old, noisy) go through the trained tracker (train_stage8_tracker.py) to ball positions, then through
stage8_sense.BallObserver (filter, latency-compensated) into the teacher's observation. No true ball state reaches the
policy (Fig8Env.est drives the per-ball tilts, the waiting timetable runs on the estimate).

    MUJOCO_GL=egl python3 eval_stage8_tracker.py runs/stage8_tracker/tracker.pt --cam-delay 0 1 2 --n 40
"""
import argparse
import os
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

from eval_stage8_noise import TEACHER, load
from rl_env_stage7_fig8 import Fig8Env
from stage8_sense import BallObserver, Senses, run_episode
from train_stage8_tracker import Tracker

CKPT = "runs/fig8_27/checkpoints/ppo_103999740_steps.zip"


def load_tracker(path):
    ck = torch.load(path, weights_only=False)
    net = Tracker(n=ck["n"], touch=ck["touch"], size=ck["size"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck["origin"]


class Measure:
    """env -> the tracker's ball positions (m) from what Senses reads now."""

    def __init__(self, sen, net, origin):
        self.sen, self.net, self.origin = sen, net, origin

    def __call__(self, env):
        img, t = self.sen.read()
        with torch.no_grad():
            x = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
            y = self.net(x, torch.from_numpy(t.astype(np.float32))[None])[0][0].numpy()
        return y / 100.0 + self.origin


def job(a):
    tracker, ck, seed, cam_delay, touch_delay, alpha, beta, pix_noise = a
    torch.set_num_threads(1)
    net, origin = load_tracker(tracker)
    model, vn = load(ck)
    env = Fig8Env(**TEACHER)
    sen = Senses(env, size=net.coords.shape[-1], cam_delay=cam_delay, touch_delay=touch_delay, pix_noise=pix_noise,
                 seed=seed)
    sen.reset()
    info, log = run_episode(env, lambda o: model.predict(vn.normalize_obs(o), deterministic=True)[0],
                            Measure(sen, net, origin), BallObserver(env, latency=cam_delay, alpha=alpha, beta=beta), seed)
    sen.close()
    return not info["lost"], env.turn.min() / (2 * np.pi), log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracker")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--cam-delay", type=int, nargs="+", default=[0])
    ap.add_argument("--touch-delay", type=int, default=0)
    ap.add_argument("--pix-noise", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.7])
    ap.add_argument("--beta", type=float, nargs="+", default=[0.3])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--procs", type=int, default=15)
    args = ap.parse_args()
    with Pool(args.procs) as P:
        for d in args.cam_delay:
            for al, be in zip(args.alpha, args.beta):
                R = P.map(job, [(args.tracker, args.ckpt, args.seed0 + i, d, args.touch_delay, al, be, args.pix_noise)
                                for i in range(args.n)])
                L = np.concatenate([r[2] for r in R])
                print(f"camera delay {d * 20:3d} ms  filter {al}/{be}: kept {sum(r[0] for r in R)}/{args.n}  "
                      f"laps/30 s {np.mean([r[1] for r in R]):.2f}  worst slot err median {np.median(L[:, 1]):.3f}  "
                      f"touching {np.mean(L[:, 0] < 0.1):.0%}  gap median {np.median(L[:, 0]):.1f} cm  "
                      f"estimate err median {np.median(L[:, 2]):.2f} cm (p90 {np.percentile(L[:, 2], 90):.2f})", flush=True)


if __name__ == "__main__":
    main()
