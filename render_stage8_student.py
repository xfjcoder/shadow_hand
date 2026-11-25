"""Stage 8: GIF of a DAgger student (train_stage8_student.py) alone. Left: the scene (oblique). Middle: what the
student's camera gives it this step (64 px, cam_delay late, with pixel noise), enlarged. Right: top view of the balls,
true (filled) and the tracker's estimate the student got (ring, from the late frame), and the touch pads' total force.
Runs n episodes and keeps the kept one that turned furthest.

    MUJOCO_GL=egl python3 render_stage8_student.py runs/student_01/student_4.pt media_full/stage8_student_01.gif
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

from eval_stage8_noise import TEACHER
from rl_env_stage7_fig8 import Fig8Env
from rl_env_stage7_wave import CONTROL_DT
from stage7_layout import PALM_Z
from stage8_sense import COLORS, recolor
from train_stage8_student import rollout

H = 270


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("student")
    ap.add_argument("gif")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=500)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--title", default="DAgger student, camera+touch")
    args = ap.parse_args()
    a = json.load(open(Path(args.student).parent / "args.json"))
    cfg = dict(tracker=a["tracker"], teacher=a["teacher"], k=a["k"], cam_delay=a["cam_delay"],
               touch_delay=a["touch_delay"], pix_noise=a["pix_noise"], touch_noise=a["touch_noise"],
               cam_drop=a["cam_drop"], no_cam=a["no_cam"], no_touch=a["no_touch"], dr=a.get("dr", 0.0))
    with Pool(min(args.n, 15)) as P:
        R = P.map(rollout, [(cfg, args.seed0 + i, 0.0, args.student, True) for i in range(args.n)])
    S = [r[2] for r in R]
    print("episodes:", [(s["kept"], round(s["laps"], 2)) for s in S])
    best = max(range(len(R)), key=lambda i: (S[i]["kept"], S[i]["laps"]))
    frames = R[best][3]
    env = Fig8Env(**TEACHER)
    recolor(env.m)
    r = mj.Renderer(env.m, height=H, width=360)
    c = np.array([env.c8[0], env.c8[1]])
    allb = np.concatenate([f[1][:, :2] for f in frames])
    span = 1.15 * np.abs(allb - c).max()
    out = []
    for k in range(0, len(frames), args.every):
        q, b, img, touch, est = frames[k]
        env.d.qpos[:] = q
        mj.mj_forward(env.m, env.d)
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [c[0], c[1], PALM_Z + 0.02]
        cam.distance, cam.azimuth, cam.elevation = 0.34, 180, -40
        r.update_scene(env.d, camera=cam)
        scene = Image.fromarray(r.render().copy())
        seen = Image.fromarray(img).resize((H, H), Image.NEAREST)
        top = Image.new("RGB", (H, H), (25, 25, 30))
        d = ImageDraw.Draw(top)
        px = lambda p: ((p[0] - c[0]) / span * H / 2 + H / 2, H / 2 - (p[1] - c[1]) / span * H / 2)
        for i in range(len(b)):
            col = tuple(int(255 * v) for v in COLORS[i][:3])
            x, y = px(b[i])
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=col)
            x, y = px(est[i])
            d.ellipse([x - 13, y - 13, x + 13, y + 13], outline=(255, 255, 255), width=2)
        err = np.linalg.norm(est - b, axis=1).max() * 100
        d.text((6, 6), f"filled: true  ring: tracker estimate (worst {err:.1f} cm)", fill=(220, 220, 220))
        d.text((6, H - 16), f"touch pads total {touch.sum():.1f} N", fill=(220, 220, 220))
        frame = Image.new("RGB", (360 + 2 * H, H))
        frame.paste(scene, (0, 0)), frame.paste(seen, (360, 0)), frame.paste(top, (360 + H, 0))
        g = ImageDraw.Draw(frame)
        g.text((8, 6), f"{args.title}  t={k * CONTROL_DT:.1f}s", fill=(255, 255, 255))
        g.text((366, 6), f"student's camera ({cfg['cam_delay'] * 20} ms late)", fill=(255, 255, 255))
        out.append(frame)
    r.close()
    out[0].save(args.gif, save_all=True, append_images=out[1:], duration=int(1000 * args.every * CONTROL_DT), loop=0)
    print(f"episode seed {args.seed0 + best}: {S[best]}; saved {len(out)} frames to {args.gif}")


if __name__ == "__main__":
    main()
