"""Stage 7A figure-8: a bank of mid-run states to start training episodes from (reference-state starts).

With the slot tolerance tight, episodes from a standing start end in 3-5 s, so nearly all training is the
spin-up and the steady circulation a policy is judged on drifts (fig8_16, fig8_19 slowed, fig8_21 dropped).
This runs a good policy (with the waiting timetable, so its errors stay in range) and keeps, every EVERY s past
the spin-up, the full simulator state and the env's own (wave clock, places on the 8, timetable) whenever every
ball is within KEEP laps of its slot. Fig8Env(bank=...) starts a share of its episodes from these.

    python3 bank_stage7_fig8.py runs/fig8_20/checkpoints/ppo_51999870_steps.zip 3 --track-rate 0.13 --tilt-k 0.3 --out data/bank_fig8_3.npz
"""
import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import mujoco as mj
import numpy as np
from stable_baselines3 import PPO

from rl_env_stage7_fig8 import Fig8Env, SPEC_BANK
from rl_env_stage7_wave import CONTROL_DT

EVERY = 0.5                  # s between kept states
KEEP = 0.1                   # laps: every ball this close to its slot


def job(a):
    ck, n, track, tilt_k, seed, seconds, pace, offsets, keep = a
    p = Path(ck)
    vn = pickle.load(open(p.parent / f"ppo_vecnormalize_{p.stem.split('_')[-2]}_steps.pkl", "rb"))
    model = PPO.load(ck, device="cpu")
    env = Fig8Env(n=n, track_rate=track, ball_act=True, grace=3, tilt_k=tilt_k, lead=0.1, episode_seconds=seconds,
                  pace=pace, **({"offsets": offsets} if offsets else {}))
    o, _ = env.reset(seed=seed)
    out, every = [], int(round(EVERY / CONTROL_DT))
    while True:
        o, r, te, tr, info = env.step(model.predict(vn.normalize_obs(o), deterministic=False)[0])
        if te or tr:
            break
        if env.t * CONTROL_DT > env.grace + 1 and env.t % every == 0 and np.all(np.abs(env._ref_err()) < keep):
            st = np.empty(mj.mj_stateSize(env.m, SPEC_BANK))
            mj.mj_getState(env.m, env.d, st, SPEC_BANK)
            out.append((st, env.t_wave, env.s.copy(), env.dist.copy(), env.ref))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("n", type=int)
    ap.add_argument("--track-rate", type=float, required=True)
    ap.add_argument("--tilt-k", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--keep", type=float, default=KEEP)
    ap.add_argument("--pace", default=None)
    ap.add_argument("--offsets", type=float, nargs="+", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = [(args.ckpt, args.n, args.track_rate, args.tilt_k, args.seed0 + i, args.seconds, args.pace, args.offsets, args.keep) for i in range(args.seeds)]
    with Pool(args.procs) as P:
        rows = [r for rs in P.map(job, jobs) for r in rs]
    np.savez(args.out, state=np.array([r[0] for r in rows]), t_wave=np.array([r[1] for r in rows]),
             s=np.array([r[2] for r in rows]), dist=np.array([r[3] for r in rows]), ref=np.array([r[4] for r in rows]))
    print(f"{len(rows)} states from {args.seeds} episodes -> {args.out}")


if __name__ == "__main__":
    main()
