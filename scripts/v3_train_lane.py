"""Train the V3 lane-marking U-Net on the CARLA dataset (``scripts/v3_collect_lane_data.py``).

Split by map: Town10HD_Opt and Town04 -> 90 % train / 10 % validation (random, seeded); Town03 is held
out completely as the test set. Loss BCE + Dice (as in the author's earlier lane U-Net), AdamW,
cosine schedule, AMP on the GPU, horizontal flip + brightness/contrast augmentation.
Writes ``models/lane_unet_s3c32.pt`` (git-ignored) and ``reports/evidence/v3_lane_training.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch import nn

from npu_ad.config import resolve
from npu_ad.perception.lane_net import LaneUNet, export_for_npu


def load(root, ids):
    xs = np.stack([cv2.imread(str(root / "rgb" / f"{i}.jpg"))[:, :, ::-1] for i in ids])
    ys = np.stack([cv2.imread(str(root / "mask" / f"{i}.png"), 0) > 127 for i in ids])
    return xs, ys


def to_t(xs, ys, dev):
    x = torch.from_numpy(np.ascontiguousarray(xs.transpose(0, 3, 1, 2))).to(dev).float().div_(255)
    y = torch.from_numpy(ys[:, None].astype(np.float32)).to(dev)
    return x, y


def dice_loss(logits, y, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + eps)).mean()


@torch.no_grad()
def evaluate(model, xs, ys, dev, bs=16) -> dict:
    tp = fp = fn = 0
    for i in range(0, len(xs), bs):
        x, y = to_t(xs[i : i + bs], ys[i : i + bs], dev)
        p = torch.sigmoid(model(x)) > 0.5
        tp += (p & (y > 0.5)).sum().item()
        fp += (p & (y < 0.5)).sum().item()
        fn += (~p & (y > 0.5)).sum().item()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"iou": tp / max(tp + fp + fn, 1), "f1": 2 * prec * rec / max(prec + rec, 1e-9), "precision": prec,
            "recall": rec, "n": len(xs)}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/v3/lane_dataset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/lane_unet_s3c32.pt")
    a = ap.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    root = resolve(a.data)
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    meta = {s["id"]: s for s in man["samples"]}
    test_ids = [s["id"] for s in man["samples"] if s["map"] == "Town03"]
    rest = [s["id"] for s in man["samples"] if s["map"] != "Town03"]
    random.Random(a.seed).shuffle(rest)
    n_val = len(rest) // 10
    val_ids, train_ids = rest[:n_val], rest[n_val:]
    dev = torch.device("cuda:0")
    xtr, ytr = load(root, train_ids)
    xva, yva = load(root, val_ids)
    xte, yte = load(root, test_ids)
    pos = ytr.mean()
    model = LaneUNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(xtr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([min(10.0, (1 - pos) / pos)], device=dev))
    scaler = torch.amp.GradScaler("cuda")
    hist, best, t0 = [], -1.0, time.perf_counter()
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(len(xtr))
        tot = 0.0
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            idx = perm[i : i + a.bs]
            xb, yb = xtr[idx].copy(), ytr[idx].copy()
            flip = np.random.rand(len(idx)) < 0.5
            xb[flip], yb[flip] = xb[flip][:, :, ::-1], yb[flip][:, :, ::-1]
            x, y = to_t(xb, yb, dev)
            x = (x * torch.empty(len(idx), 1, 1, 1, device=dev).uniform_(0.7, 1.3)
                 + torch.empty(len(idx), 1, 1, 1, device=dev).uniform_(-0.1, 0.1)).clamp(0, 1)  # fmt: skip
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = bce(logits.float(), y) + dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(idx)
        model.eval()
        v = evaluate(model, xva, yva, dev)
        hist.append({"epoch": ep + 1, "train_loss": tot / len(xtr), "val": v})
        print(f"ep {ep + 1:2d} loss {tot / len(xtr):.4f} val IoU {v['iou']:.3f} F1 {v['f1']:.3f}", flush=True)
        if v["iou"] > best:
            best = v["iou"]
            torch.save({"arch": {"base": 32, "levels": 4}, "state_dict": model.state_dict(), "epoch": ep + 1},
                       resolve(a.out))  # fmt: skip
    ck = torch.load(resolve(a.out), map_location=dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    res = {
        "data": a.data, "split": {"train": len(train_ids), "val": len(val_ids), "test_Town03_heldout": len(test_ids)},
        "positive_pixel_frac_train": float(pos), "epochs": a.epochs, "best_epoch": ck["epoch"], "seed": a.seed,
        "train_s": time.perf_counter() - t0, "history": hist,
        "val": evaluate(model, xva, yva, dev), "test_Town03": evaluate(model, xte, yte, dev),
        "params": sum(p.numel() for p in model.parameters()),
    }  # fmt: skip
    by = defaultdict(list)
    for i in test_ids:
        by[meta[i]["weather"]].append(i)
    res["test_by_weather"] = {w: evaluate(model, *load(root, ids), dev) for w, ids in sorted(by.items())}
    exp = export_for_npu(model.cpu())
    x = torch.rand(1, 3, 256, 512)
    with torch.no_grad():
        res["export_max_abs_vs_trained_cpu"] = float((exp(x) - model.cpu()(x)).abs().max())
    resolve("reports/evidence/v3_lane_training.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("val", res["val"], "\ntest Town03", res["test_Town03"], "\nexport diff", res["export_max_abs_vs_trained_cpu"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
