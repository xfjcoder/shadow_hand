"""Stage 7A figure-8: can the hands time balls one by one? A controllability probe.

Snapshots are taken from a trained policy's own rollouts (one every SNAP_EVERY s). From each, the policy's next
H steps of actions are recorded and replayed open-loop, once as they are and then with one actuator's residual
nudged by +-DELTA (in action units, x the residual's reach) for the whole H steps. The change in each ball's place
on the 8 after H steps, (s(+) - s(-)) / 2 DELTA, gives one column of J: laps per unit of action, ball x actuator.

Per snapshot, the balls on each hand (left lobe y > the crossing's y, right lobe below; balls within NEAR of the
crossing are left out) give a block of J. With two or more balls on a hand:
  common   = |mean of the rows|_1 / H: how fast all of them can be sped up together (laps/s at full residual),
  diff     = |row_a - row_b|_1 / H, the worst pair: how fast one can be sped up against another,
  sv ratio = sigma_min / sigma_max of the block: 1 means each ball has its own handle, 0 that they move as one.
(L1 norms: every actuator pushed to its full +-1 residual in the helpful sign; linear in DELTA, so an indication.)
linearity: |J(+DELTA) + J(-DELTA)| / |J(+DELTA) - J(-DELTA)| over the snapshot (0 = linear).

    python3 probe_stage7_fig8.py runs/fig8_04/checkpoints/ppo_7999980_steps.zip 4 --out data/probe_fig8_4.npz
"""
import argparse
import pickle
from multiprocessing import Pool
from pathlib import Path

import mujoco as mj
import numpy as np
from stable_baselines3 import PPO

from rl_env_stage7_fig8 import Fig8Env

H = (10, 25)                 # control steps (0.2 s, 0.5 s): J at each
DELTA = 0.3
ALPHA = 0.5                  # the check's push, per actuator (action units)
SNAP_EVERY = 1.0             # s
NEAR = 0.01                  # m in y from the crossing: neither hand's
SPEC = mj.mjtState.mjSTATE_INTEGRATION
ATTRS = ("t_wave", "s", "dist", "t", "turn", "ref")


def save(env):
    st = np.empty(mj.mj_stateSize(env.m, SPEC))
    mj.mj_getState(env.m, env.d, st, SPEC)
    return st, {k: np.copy(getattr(env, k)) for k in ATTRS if hasattr(env, k)}


def load(env, snap):
    mj.mj_setState(env.m, env.d, snap[0], SPEC)
    for k, v in snap[1].items():
        setattr(env, k, np.copy(v))


def replay(env, snap, acts):
    """Replay the actions open-loop from the snapshot; each ball's s after each horizon in H."""
    load(env, snap)
    out = {}
    for i, a in enumerate(acts):
        env.step(np.clip(a, -1, 1))
        if i + 1 in H:
            out[i + 1] = env.s.copy()
    return out


def probe(job):
    ck, n, seed, seconds, track = job
    p = Path(ck)
    vn = pickle.load(open(p.parent / f"ppo_vecnormalize_{p.stem.split('_')[-2]}_steps.pkl", "rb"))
    model = PPO.load(ck, device="cpu")
    env = Fig8Env(n=n, episode_seconds=seconds + 1, track_rate=track)
    act = lambda o: model.predict(vn.normalize_obs(o), deterministic=True)[0]
    o, _ = env.reset(seed=seed)
    rows, every = [], int(round(SNAP_EVERY / 0.02))
    for t in range(int(seconds / 0.02)):
        if t >= every and t % every == 0:
            snap = save(env)
            b = env.balls_n().copy()
            acts, oo = [], o
            for _ in range(max(H)):                     # the policy's own next actions, then the snapshot back
                acts.append(act(oo))
                oo, _, te, _, _ = env.step(acts[-1])
                if te:
                    break
            if len(acts) == max(H):
                acts = np.array(acts)
                base = replay(env, snap, acts)
                J = {h: np.zeros((n, env.m.nu)) for h in H}
                Jp = {h: np.zeros((n, env.m.nu)) for h in H}
                for j in range(env.m.nu):
                    e = np.zeros(env.m.nu)
                    e[j] = DELTA
                    up, dn = replay(env, snap, acts + e), replay(env, snap, acts - e)
                    for h in H:
                        J[h][:, j] = (up[h] - dn[h]) / (2 * DELTA)
                        Jp[h][:, j] = ((up[h] - base[h]) + (dn[h] - base[h])) / (2 * DELTA)   # the curvature
                # the check: for each pair (a, b) on one hand, push every actuator ALPHA in sign(J_a - J_b) (the
                # direction that should speed a past b), and a random-sign push as the control; realised change
                # in s_a - s_b after max(H), (+push - -push) / 2, against the predicted ALPHA |J_a - J_b|_1
                h, y, rng = max(H), b[:, 1], np.random.default_rng(seed * 1000 + t)
                pairs = []
                for side in (y > env.c8[1] + NEAR, y < env.c8[1] - NEAR):
                    k = np.flatnonzero(side)
                    for i in range(len(k)):
                        for i2 in range(i + 1, len(k)):
                            a, c = k[i], k[i2]
                            res = []
                            for d in (np.sign(J[h][a] - J[h][c]), rng.choice([-1.0, 1.0], env.m.nu)):
                                up, dn = replay(env, snap, acts + ALPHA * d), replay(env, snap, acts - ALPHA * d)
                                ds = lambda r: np.angle(np.exp(2j * np.pi * (r[h][a] - r[h][c]))) / (2 * np.pi)
                                res.append((ds(up) - ds(dn)) / 2)
                            pairs.append((a, c, ALPHA * np.abs(J[h][a] - J[h][c]).sum(), *res))
                rows.append(dict(seed=seed, t=t * 0.02, balls=b, **{f"J{h}": J[h] for h in H},
                                 **{f"Jc{h}": Jp[h] for h in H}, pairs=pairs))
            load(env, snap)
        o, _, te, tr, _ = env.step(act(o))
        if te:
            break
    return rows


def analyse(rows, c_y, n):
    names = None
    for h in H:
        com, dif, svr, lin = [], [], [], []
        for r in rows:
            J, Jc, y = r[f"J{h}"], r[f"Jc{h}"], r["balls"][:, 1]
            lin.append(np.abs(Jc).sum() / max(np.abs(J).sum(), 1e-12))
            for side in (y > c_y + NEAR, y < c_y - NEAR):
                k = np.flatnonzero(side)
                if len(k) < 2:
                    continue
                B = J[k]
                com.append(np.abs(B.mean(0)).sum() / (h * 0.02))
                dif.append(min(np.abs(B[a] - B[c]).sum() for a in range(len(k)) for c in range(a + 1, len(k))) / (h * 0.02))
                sv = np.linalg.svd(B, compute_uv=False)
                svr.append(sv[-1] / sv[0])
        q = lambda x: "  ".join(f"{v:.3f}" for v in np.percentile(x, [10, 50, 90])) if len(x) else "-"
        print(f"H = {h * 0.02:.1f} s: {len(com)} hand-snapshots with >= 2 balls (of {len(rows)} snapshots)")
        print(f"  common speed-up, laps/s   p10/50/90  {q(com)}")
        print(f"  worst-pair differential   p10/50/90  {q(dif)}")
        print(f"  sigma_min / sigma_max     p10/50/90  {q(svr)}")
        print(f"  nonlinearity |curv|/|J|   p10/50/90  {q(lin)}")
    P = np.array([p[2:] for r in rows for p in r["pairs"]])
    if len(P):
        h = max(H) * 0.02
        print(f"check over {h:.1f} s, {len(P)} same-hand pairs, push {ALPHA} on every actuator:")
        print(f"  predicted s_a - s_b, laps      median {np.median(P[:, 0]):.3f}")
        print(f"  realised, aimed push           median {np.median(P[:, 1]):+.3f}  p10/90 {np.percentile(P[:, 1], 10):+.3f} "
              f"{np.percentile(P[:, 1], 90):+.3f}  right sign {np.mean(P[:, 1] > 0):.0%}")
        print(f"  realised, random push (|.|)    median {np.median(np.abs(P[:, 2])):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("n", type=int)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=700)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--track-rate", type=float, default=None)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    jobs = [(args.ckpt, args.n, args.seed0 + i, args.seconds, args.track_rate) for i in range(args.seeds)]
    with Pool(args.procs) as P:
        rows = [r for rs in P.map(probe, jobs) for r in rs]
    env = Fig8Env(n=args.n)
    analyse(rows, env.c8[1], args.n)
    if args.out:
        pairs = np.array([(i, *pp) for i, r in enumerate(rows) for pp in r["pairs"]]).reshape(-1, 6)
        np.savez(args.out, pairs=pairs, **{k: np.array([r[k] for r in rows]) for k in rows[0] if k != "pairs"},
                 names=np.array([mj.mj_id2name(env.m, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(env.m.nu)]))


if __name__ == "__main__":
    main()
