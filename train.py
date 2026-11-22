"""Train any model:  python train.py --model unet --pairs PATH/pairs.csv --folder PATH --out runs/unet"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CHANNELS, FloodDataset, get_ids
from models import build_model


def dice_bce(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + y.sum((0, 2, 3))
    return bce + (1 - (2 * inter + 1) / (den + 1)).mean()


@torch.no_grad()
def evaluate(model, loader, dev, thr=0.5):
    model.eval()
    n = len(CHANNELS)
    tp, fp, fn = torch.zeros(n), torch.zeros(n), torch.zeros(n)
    for pre, post, y in loader:
        pre, post, y = pre.to(dev), post.to(dev), y.to(dev)
        with torch.autocast("cuda", enabled=dev == "cuda"):
            out = model(pre, post)
        p = (torch.sigmoid(out.float()) > thr).float()[:, :, :1300, :1300]  # cut the padding off
        y = y[:, :, :1300, :1300]
        tp += (p * y).sum((0, 2, 3)).cpu()
        fp += (p * (1 - y)).sum((0, 2, 3)).cpu()
        fn += ((1 - p) * y).sum((0, 2, 3)).cpu()
    e = 1e-9
    iou, f1 = tp / (tp + fp + fn + e), 2 * tp / (2 * tp + fp + fn + e)
    prec, rec = tp / (tp + fp + e), tp / (tp + fn + e)
    return {"iou": iou.tolist(), "f1": f1.tolist(), "precision": prec.tolist(), "recall": rec.tolist(),
            "mean_iou": iou.mean().item(), "mean_f1": f1.mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--test", action="store_true", help="score best.pt on the TEST split (use once, at the end)")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    tr = FloodDataset(a.folder, get_ids(a.pairs, "train"), train=True)
    va = FloodDataset(a.folder, get_ids(a.pairs, "val"))
    tl = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    vl = DataLoader(va, batch_size=2, num_workers=a.workers)

    model = build_model(a.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    best, bad, hist = -1.0, 0, []
    print(f"{a.model} | {dev} | train crops {len(tr)} | val pairs {len(va)}")
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for pre, post, y in tl:
            pre, post, y = pre.to(dev), post.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=dev == "cuda"):
                out = model(pre, post)
            loss = dice_bce(out.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * pre.size(0)
            n += pre.size(0)
        sched.step()
        m = evaluate(model, vl, dev)
        hist.append({"epoch": ep, "train_loss": tot / n, "val": m})
        print(f"ep {ep:2d} loss {tot / n:.4f} | val mIoU {m['mean_iou']:.4f} F1 {m['mean_f1']:.4f} "
              f"| IoU {np.round(m['iou'], 3)} | {time.time() - t0:.0f}s", flush=True)
        json.dump(hist, open(f"{a.out}/history.json", "w"), indent=1)
        if m["mean_iou"] > best:
            best, bad = m["mean_iou"], 0
            torch.save(model.state_dict(), f"{a.out}/best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop")
                break

    if a.test:
        model.load_state_dict(torch.load(f"{a.out}/best.pt", map_location=dev))
        te = DataLoader(FloodDataset(a.folder, get_ids(a.pairs, "test")), batch_size=2, num_workers=a.workers)
        res = evaluate(model, te, dev)
        json.dump(res, open(f"{a.out}/test_metrics.json", "w"), indent=1)
        print("TEST:", {k: (round(v, 4) if isinstance(v, float) else np.round(v, 3)) for k, v in res.items()})


if __name__ == "__main__":
    main()
