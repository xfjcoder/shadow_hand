"""Stage 7A figure-8: warm-start an n-ball policy from an (n_old < n)-ball one. The observation grows with the balls
(Fig8Env: 2H hand + 3n positions + 3n velocities + n cos s + n sin s + n distances + 2 wave phase), so the
first layer of the policy and value nets is widened: the old balls keep their weights, the new balls' inputs
start at zero weight (the policy at first runs the old balls as before and ignores the new ones), and the
running obs normalisation of each new ball is copied from ball (k mod n_old). Everything else is copied.

    python3 widen_stage7_fig8.py runs/fig8_01/checkpoints/ppo_19999950_steps.zip 2 runs/fig8_02w/ppo_0_steps.zip
then train with --resume runs/fig8_02w/ppo_0_steps.zip --n-balls 2.
--track-rate R adds the timetable inputs (Fig8Env track_rate: each ball's timetable error, the timetable's phase)
at zero weight, normalised from mean 0 / variance 1; if the old model has them they carry over like the rest.
--ball-act adds the per-ball tilt actions (Fig8Env ball_act) at zero weight (the policy starts as before), their
noise at the old actions' mean; N may equal the old ball count then (actions only).
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_env_stage7_fig8 import Fig8Env


def layout(H, k, track):
    idx = [("h", j, 0) for j in range(2 * H)]
    idx += [("p", b, c) for b in range(k) for c in range(3)]
    idx += [("v", b, c) for b in range(k) for c in range(3)]
    idx += [(f, b, 0) for f in ("cos", "sin", "dist") for b in range(k)]
    idx += [("ph", j, 0) for j in range(2)]
    if track:
        idx += [("err", b, 0) for b in range(k)] + [("tph", j, 0) for j in range(2)]
    return idx


def old_balls(H, n_obs):
    """The old model's ball count and whether it has the timetable inputs (2H + 9n + 2 [+ n + 2]; n <= 4 is unique)."""
    if (n_obs - 2 * H - 2) % 9 == 0:
        return (n_obs - 2 * H - 2) // 9, False
    assert (n_obs - 2 * H - 4) % 10 == 0, n_obs
    return (n_obs - 2 * H - 4) // 10, True


def obs_map(H, n_old, track_old, n, track):
    """For each obs index of the new layout: (index in the old layout it copies, whether it starts at zero weight:
    a new ball's, or a timetable input the old model did not have)."""
    old = {key: i for i, key in enumerate(layout(H, n_old, track_old))}
    src, new = [], []
    for f, b, c in layout(H, n, track):
        if f in ("err", "tph") and not track_old:
            src.append(-1), new.append(True)
        elif f in ("h", "ph", "tph"):
            src.append(old[(f, b, c)]), new.append(False)
        else:
            src.append(old[(f, b % n_old, c)]), new.append(b >= n_old)
    return np.array(src), np.array(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("n", type=int)
    ap.add_argument("out")
    ap.add_argument("--track-rate", type=float, default=None)
    ap.add_argument("--ball-act", action="store_true")
    args = ap.parse_args()
    p = Path(args.ckpt)
    steps = p.stem.split("_")[-2]
    old = PPO.load(args.ckpt, device="cpu")
    vn = pickle.load(open(p.parent / f"ppo_vecnormalize_{steps}_steps.pkl", "rb"))
    env = Fig8Env(n=args.n, track_rate=args.track_rate, ball_act=args.ball_act)
    H = len(env.hq)
    n_old, track_old = old_balls(H, old.observation_space.shape[0])
    src, new = obs_map(H, n_old, track_old, args.n, bool(args.track_rate))
    fresh = src < 0                                               # timetable inputs the old model lacks
    src = np.where(fresh, 0, src)
    assert len(src) == env.observation_space.shape[0]
    nu_old = old.action_space.shape[0]

    model = PPO("MlpPolicy", env, learning_rate=old.learning_rate, n_steps=old.n_steps, batch_size=old.batch_size,
                n_epochs=old.n_epochs, gamma=old.gamma, device="cpu", verbose=1, policy_kwargs=old.policy_kwargs)
    sd = old.policy.state_dict()
    with torch.no_grad():
        for name, t in model.policy.state_dict().items():
            o = sd[name]
            if t.shape == o.shape:
                t.copy_(o)
            elif name.startswith("action_net") or name == "log_std":   # more actions: zero rows, noise at the mean
                pad = torch.full((t.shape[0] - nu_old, *o.shape[1:]), float(o.mean()) if name == "log_std" else 0.0)
                t.copy_(torch.cat([o, pad]))
            else:                                        # a first layer: [out, obs]
                w = o[:, src].clone()
                w[:, torch.as_tensor(new)] = 0.0
                t.copy_(w)
    model.num_timesteps = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)

    nv = VecNormalize(DummyVecEnv([lambda: env]), gamma=vn.gamma, clip_obs=vn.clip_obs, clip_reward=vn.clip_reward)
    nv.obs_rms.mean, nv.obs_rms.var = vn.obs_rms.mean[src].copy(), vn.obs_rms.var[src].copy()
    nv.obs_rms.mean[fresh], nv.obs_rms.var[fresh] = 0.0, 1.0
    nv.obs_rms.count, nv.ret_rms = vn.obs_rms.count, vn.ret_rms
    nv.save(str(out.parent / f"ppo_vecnormalize_{out.stem.split('_')[-2]}_steps.pkl"))
    print(f"widened {n_old} -> {args.n} balls ({len(src)} obs, {int(new.sum())} new inputs at zero; "
          f"{nu_old} -> {env.action_space.shape[0]} actions) -> {out}")


if __name__ == "__main__":
    main()
