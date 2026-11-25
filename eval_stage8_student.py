"""Stage 8: evaluate DAgger students (train_stage8_student.py) alone on fresh seeds, with the senses they trained on
(args.json of the run), or overridden.

    MUJOCO_GL=egl python3 eval_stage8_student.py runs/student_01/student_4.pt runs/student_01/student_8.pt --n 120
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
from train_stage8_student import rollout, summarise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("students", nargs="+")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--cam-delay", type=int)
    ap.add_argument("--cam-drop", type=float)
    ap.add_argument("--dr", type=float, help="camera domain randomisation scale")
    ap.add_argument("--tracker", help="override the run's tracker")
    ap.add_argument("--procs", type=int, default=15)
    args = ap.parse_args()
    with Pool(args.procs) as P:
        for s in args.students:
            a = json.load(open(Path(s).parent / "args.json"))
            cfg = dict(tracker=a["tracker"], teacher=a["teacher"], k=a["k"], cam_delay=a["cam_delay"],
                       touch_delay=a["touch_delay"], pix_noise=a["pix_noise"], touch_noise=a["touch_noise"],
                       cam_drop=a["cam_drop"], no_cam=a["no_cam"], no_touch=a["no_touch"], dr=a.get("dr", 0.0))
            for k in ("cam_delay", "cam_drop", "dr", "tracker"):
                if getattr(args, k) is not None:
                    cfg[k] = getattr(args, k)
            R = P.map(rollout, [(cfg, args.seed0 + i, 0.0, s, False) for i in range(args.n)])
            print(f"{s} (camera {cfg['cam_delay'] * 20} ms, drop {cfg['cam_drop']}, dr {cfg['dr']}): "
                  f"{summarise([r[2] for r in R])}", flush=True)


if __name__ == "__main__":
    main()
