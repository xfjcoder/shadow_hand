"""Stage 8 step 3 data: teacher episodes recorded as the student would sense them. Every control step: the overhead
camera image (uint8), the touch pads (N), and the true ball positions (the tracker's labels), the hand's joint angles.
The teacher acts stochastically, and a share of episodes runs it on noisy ball state (NOISE cm), so the data also
holds the off-nominal states a tracker-driven policy will get into.

    MUJOCO_GL=egl python3 collect_stage8.py runs/fig8_27/checkpoints/ppo_103999740_steps.zip --episodes 120 --out data/stage8_frames.npz
"""
import argparse
import os
from collections import deque
from multiprocessing import Pool

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

from eval_stage8_noise import TEACHER, load
from rl_env_stage7_fig8 import Fig8Env
from stage8_sense import BallObserver, Senses, run_episode

NOISE = (0.0, 0.0, 0.3, 0.6)          # cm of ball-state noise for the teacher, by episode (cycled)


def job(a):
    ck, seed, size = a
    model, vn = load(ck)
    env = Fig8Env(**TEACHER)
    sen = Senses(env, size=size, cam_delay=0, touch_delay=0, touch_noise=0.0, touch_floor=0.0)
    sen.reset()
    rng = np.random.default_rng(seed)
    sigma = NOISE[seed % len(NOISE)]
    rec = {k: [] for k in ("img", "touch", "balls", "hand")}

    def measure(e):
        return e.balls_n() + rng.normal(0, sigma / 100, (e.n, 3))

    def on_step(e):
        img, t = sen.read()
        rec["img"].append(img)
        rec["touch"].append(t.astype(np.float32))
        rec["balls"].append(e.balls_n().astype(np.float32))
        rec["hand"].append(e.d.qpos[e.hq].astype(np.float32))

    run_episode(env, lambda o: model.predict(vn.normalize_obs(o), deterministic=False)[0], measure,
                BallObserver(env, alpha=1.0, beta=1.0), seed, on_step=on_step)
    sen.close()
    return {k: np.array(v) for k, v in rec.items()}, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=20000)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--procs", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with Pool(args.procs) as P:
        R = P.map(job, [(args.ckpt, args.seed0 + i, args.size) for i in range(args.episodes)])
    out = {k: np.concatenate([r[0][k] for r in R]) for k in R[0][0]}
    out["episode"] = np.concatenate([np.full(len(r[0]["balls"]), r[1]) for r in R])
    np.savez(args.out, **out)
    print(f"{len(out['balls'])} frames from {args.episodes} episodes -> {args.out}")


if __name__ == "__main__":
    main()
