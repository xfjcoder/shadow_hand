"""Stage 8 step 3: the ball tracker. Overhead camera image (+ touch pads) -> each ball's 3D position, trained on
collect_stage8.py frames (labels: the true positions). A small conv net with one heat map per ball (the balls are
coloured apart) and a spatial soft-argmax (sub-pixel image coordinates), then an MLP from those coordinates, the map
peaks (how sure), pooled image features and the touch pads to the position (cm from the crossing / palm height).
Training adds pixel noise and brightness jitter. Validation on held-out episodes.

    python3 train_stage8_tracker.py data/stage8_frames.npz --out runs/stage8_tracker/tracker.pt
    python3 train_stage8_tracker.py data/stage8_frames.npz --out runs/stage8_tracker_cam/tracker.pt --no-touch
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_stage8_noise import TEACHER
from rl_env_stage7_fig8 import Fig8Env
from stage7_layout import PALM_Z


def origin():
    e = Fig8Env(**TEACHER)
    return np.array([e.c8[0], e.c8[1], PALM_Z], np.float32)


class Tracker(nn.Module):
    def __init__(self, n=3, n_touch=48, touch=True, size=64):
        super().__init__()
        self.n, self.touch = n, touch
        self.trunk = nn.Sequential(
            nn.Conv2d(5, 32, 5, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.heat = nn.Conv2d(64, n, 1)
        self.temp = nn.Parameter(torch.tensor(1.0))
        h = size // 2
        g = torch.linspace(-1, 1, h)
        self.register_buffer("gy", g.view(1, 1, h, 1).expand(1, 1, h, h).reshape(1, 1, -1))
        self.register_buffer("gx", g.view(1, 1, 1, h).expand(1, 1, h, h).reshape(1, 1, -1))
        c = torch.linspace(-1, 1, size)
        self.register_buffer("coords", torch.stack([c.view(size, 1).expand(size, size), c.view(1, size).expand(size, size)])[None])
        self.tmlp = nn.Sequential(nn.Linear(n_touch, 64), nn.ReLU()) if touch else None
        k = 3 * n + 64 + (64 if touch else 0)
        self.head = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3 * n))

    def forward(self, img, touch=None):
        """img: B x 3 x H x W in [0, 1]; touch: B x 48 (N). -> B x n x 3 cm from the origin, and the image coords."""
        x = torch.cat([img, self.coords.expand(img.shape[0], -1, -1, -1)], 1)
        f = self.trunk(x)
        hm = self.heat(f).flatten(2) * self.temp
        p = F.softmax(hm, -1)
        u, v = (p * self.gx).sum(-1), (p * self.gy).sum(-1)
        z = [torch.stack([u, v, p.max(-1).values], -1).flatten(1), f.mean((2, 3))]
        if self.touch:
            z.append(self.tmlp(torch.log1p(touch)))
        return self.head(torch.cat(z, 1)).view(-1, self.n, 3), torch.stack([u, v], -1)


def batches(idx, bs, rng):
    idx = rng.permutation(idx)
    for i in range(0, len(idx), bs):
        yield np.sort(idx[i:i + bs])


def to_t(D, j, aug, rng):
    img = torch.from_numpy(D["img"][j]).float().permute(0, 3, 1, 2) / 255.0
    if aug:
        img = img * torch.from_numpy(rng.uniform(0.85, 1.15, (len(j), 1, 1, 1))).float()
        img = img + torch.randn_like(img) * torch.from_numpy(rng.uniform(0, 8 / 255, (len(j), 1, 1, 1))).float()
        img = img.clamp(0, 1)
    return img, torch.from_numpy(D["touch"][j]), torch.from_numpy(D["y"][j])


def evaluate(net, D, idx, bs=1024):
    net.eval()
    E = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            img, t, y = to_t(D, j, False, None)
            E.append((net(img, t)[0] - y).norm(dim=-1).numpy())
    net.train()
    return np.concatenate(E)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-touch", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    raw = np.load(args.data)
    D = {"img": raw["img"], "touch": raw["touch"].astype(np.float32),
         "y": ((raw["balls"] - origin()) * 100).astype(np.float32)}
    ep = raw["episode"]
    eps = np.unique(ep)
    val_eps = eps[:: 10]
    val = np.where(np.isin(ep, val_eps))[0]
    tr = np.where(~np.isin(ep, val_eps))[0]
    print(f"{len(tr)} train / {len(val)} val frames ({len(val_eps)} val episodes)")
    n = D["y"].shape[1]
    net = Tracker(n=n, touch=not args.no_touch, size=D["img"].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.default_rng(0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = np.inf
    for epoch in range(args.epochs):
        t0, L = time.time(), []
        for j in batches(tr, args.bs, rng):
            img, t, y = to_t(D, j, True, rng)
            loss = F.smooth_l1_loss(net(img, t)[0], y, beta=0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()
            L.append(loss.item())
        sched.step()
        e = evaluate(net, D, val)
        print(f"epoch {epoch + 1}: train loss {np.mean(L):.3f}  val error median {np.median(e):.2f} cm  p90 "
              f"{np.percentile(e, 90):.2f}  p99 {np.percentile(e, 99):.2f}  ({time.time() - t0:.0f} s)", flush=True)
        if np.median(e) < best:
            best = np.median(e)
            torch.save({"state": net.state_dict(), "n": n, "touch": not args.no_touch, "size": D["img"].shape[1],
                        "origin": origin()}, out)
    print(f"best val median {best:.2f} cm -> {out}")


if __name__ == "__main__":
    main()
