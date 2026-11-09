"""Stage 6: how replayable is a planner demo? From every `--every`-th logged step, restore the logged state and
replay the demo's own controls open-loop for `--seconds`; report the steps whose replay loses a ball (moves too
knife-edge to resume from a saved frame -- RL starts there are near-impossible).

Usage:
    python3 eval_stage6_demo_replay.py data/stage6_demo_pads.npz --seconds 2 --every 10
"""
import argparse
from multiprocessing import Pool

import mujoco as mj
import numpy as np

from rl_env_track3 import DemoTrackEnv

ENV = None


def _init(demo):
    global ENV
    ENV = DemoTrackEnv(level=0, demo=demo, episode_seconds=1e6)


def replay(args):
    start, steps = args
    e, z = ENV, ENV.demo
    e.reset(options={"start": start})
    for k in range(steps):
        i = start + k + 1
        if i >= len(z["ctrl"]):
            return None
        e.d.ctrl[:] = np.clip(z["ctrl"][i], e.lo, e.hi)
        for _ in range(e.n_sub):
            mj.mj_step(e.m, e.d)
        if e.lost(e.balls()):
            return k
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    n = len(np.load(args.demo)["ctrl"])
    steps = int(round(args.seconds / 0.02))
    starts = list(range(0, n - steps - 1, args.every))
    with Pool(args.procs, initializer=_init, initargs=(args.demo,)) as pool:
        res = pool.map(replay, [(s, steps) for s in starts])
    bad = [(s, r) for s, r in zip(starts, res) if r is not None and r >= 0]
    print(f"{args.demo}: {len(bad)}/{len(starts)} starts lose a ball within {args.seconds:g} s of open-loop replay")
    if bad:
        groups, cur = [], [bad[0]]
        for b in bad[1:]:
            if b[0] - cur[-1][0] <= args.every:
                cur.append(b)
            else:
                groups.append(cur)
                cur = [b]
        groups.append(cur)
        for g in groups:
            ends = sorted({s + r for s, r in g})
            print(f"  starts {g[0][0]}-{g[-1][0]}: ball lost at demo steps {ends[0]}-{ends[-1]}")


if __name__ == "__main__":
    main()
