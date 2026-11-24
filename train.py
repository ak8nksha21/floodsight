"""Train any model:  python train.py --model unet --pairs PATH/pairs.csv --folder PATH --out runs/unet"""
import argparse, json, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CHANNELS, FloodDataset, get_ids
from models import build_model

THS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def dice_bce(logits, y, w=None):
    """BCE + soft Dice per channel; w = per-channel weights (e.g. more weight on the flooded channels)."""
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none").mean((0, 2, 3))
    p = torch.sigmoid(logits)
    inter = (p * y).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + y.sum((0, 2, 3))
    per = bce + (1 - (2 * inter + 1) / (den + 1))
    if w is None:
        return per.mean()
    return (per * w).sum() / w.sum()


@torch.no_grad()
def evaluate(model, loader, dev, thr=None, tta=False):
    """thr=None : try every threshold in THS and keep the best one per channel (use on VALIDATION).
    thr=[t0..t3]: use these fixed thresholds (use on TEST, with the ones chosen on validation)."""
    model.eval()
    n = len(CHANNELS)
    if thr is None:
        T = torch.tensor(THS, dtype=torch.float32).view(-1, 1).expand(-1, n).to(dev)
    else:
        T = torch.tensor(thr, dtype=torch.float32).view(1, n).to(dev)
    K = T.shape[0]
    tp, fp, fn = (torch.zeros(K, n, device=dev, dtype=torch.float64) for _ in range(3))
    for pre, post, y in loader:
        pre, post, y = pre.to(dev), post.to(dev), y.to(dev)
        with torch.autocast("cuda", enabled=dev == "cuda"):
            out = model(pre, post)
            p = torch.sigmoid(out.float())
            if tta:  # average with the left-right and up-down flipped predictions
                for dims in ([3], [2]):
                    o2 = model(pre.flip(dims), post.flip(dims))
                    p = p + torch.sigmoid(o2.float()).flip(dims)
                p = p / 3
        p = p[:, :, :1300, :1300]  # cut the padding off
        y = y[:, :, :1300, :1300]
        for k in range(K):
            b = (p > T[k].view(1, n, 1, 1)).float()
            tp[k] += (b * y).sum((0, 2, 3)).double()
            fp[k] += (b * (1 - y)).sum((0, 2, 3)).double()
            fn[k] += ((1 - b) * y).sum((0, 2, 3)).double()
    e = 1e-9
    iou_all = tp / (tp + fp + fn + e)
    ar = torch.arange(n, device=dev)
    best = iou_all.argmax(0) if thr is None else torch.zeros(n, dtype=torch.long, device=dev)
    tp, fp, fn, T = tp[best, ar], fp[best, ar], fn[best, ar], T[best, ar]
    iou, f1 = tp / (tp + fp + fn + e), 2 * tp / (2 * tp + fp + fn + e)
    prec, rec = tp / (tp + fp + e), tp / (tp + fn + e)
    return {"thresholds": T.tolist(), "iou": iou.tolist(), "f1": f1.tolist(),
            "precision": prec.tolist(), "recall": rec.tolist(),
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
    ap.add_argument("--use", default="post", help="image the single-image models see: post or pre")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weights", default="imagenet")
    ap.add_argument("--encoder", default="resnet34", help="encoder of siamese / single_mt (e.g. resnet34, resnet50)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off, 0.999 is typical)")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=2, help="random crops per pair per epoch")
    ap.add_argument("--max_minutes", type=float, default=0, help="stop training after this many minutes (0 = no limit)")
    ap.add_argument("--flood_w", type=float, default=1.0, help="loss weight of the two flooded channels")
    ap.add_argument("--tta", action="store_true", help="flip test-time augmentation for the final thresholds + test")
    ap.add_argument("--trainval", action="store_true", help="also train on the validation pairs (needs --thr_file)")
    ap.add_argument("--thr_file", default=None, help="thresholds.json of an earlier validated run (for --trainval)")
    ap.add_argument("--test", action="store_true", help="score best.pt on the TEST split (use once, at the end)")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    ids_tr = get_ids(a.pairs, "train") + (get_ids(a.pairs, "val") if a.trainval else [])
    tr = FloodDataset(a.folder, ids_tr, train=True, crop=a.crop, repeat=a.repeat)
    va = FloodDataset(a.folder, get_ids(a.pairs, "val"))
    tl = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    vl = DataLoader(va, batch_size=2, num_workers=a.workers)

    model = build_model(a.model, weights=a.weights, use=a.use, dropout=a.dropout, encoder=a.encoder).to(dev)
    ema = None
    if a.ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)
    T0 = time.time()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    lw = torch.tensor([1.0, 1.0, a.flood_w, a.flood_w], device=dev)
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
            loss = dice_bce(out.float(), y, lw)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None: ema.update_parameters(model)
            tot += loss.item() * pre.size(0)
            n += pre.size(0)
        sched.step()
        em = ema.module if ema is not None else model  # the model that is scored and saved
        m = evaluate(em, vl, dev)
        hist.append({"epoch": ep, "train_loss": tot / n, "val": m})
        print(f"ep {ep:2d} loss {tot / n:.4f} | val mIoU {m['mean_iou']:.4f} F1 {m['mean_f1']:.4f} "
              f"| IoU {np.round(m['iou'], 3)} | thr {np.round(m['thresholds'], 2)} | {time.time() - t0:.0f}s", flush=True)
        json.dump(hist, open(f"{a.out}/history.json", "w"), indent=1)
        if a.trainval:  # validation is part of the training data, so no model selection: keep the last epoch
            torch.save(em.state_dict(), f"{a.out}/best.pt")
            continue
        if m["mean_iou"] > best:
            best, bad = m["mean_iou"], 0
            torch.save(em.state_dict(), f"{a.out}/best.pt")
            json.dump(m["thresholds"], open(f"{a.out}/thresholds.json", "w"))
            json.dump(m, open(f"{a.out}/best_val.json", "w"), indent=1)
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop")
                break
        if a.max_minutes and (time.time() - T0) / 60 > a.max_minutes:
            print("time budget reached, stopping")
            break

    if a.test:
        model = em
        model.load_state_dict(torch.load(f"{a.out}/best.pt", map_location=dev))
        if a.trainval:
            thr = json.load(open(a.thr_file))  # thresholds of the validated run
        else:
            mv = evaluate(model, vl, dev, None, a.tta)  # pick thresholds on validation (with TTA if asked)
            thr = mv["thresholds"]
            json.dump(mv, open(f"{a.out}/best_val.json", "w"), indent=1)
            print("VAL (final):", {k: (round(v, 4) if isinstance(v, float) else np.round(v, 3)) for k, v in mv.items()})
        json.dump(thr, open(f"{a.out}/thresholds.json", "w"))
        te = DataLoader(FloodDataset(a.folder, get_ids(a.pairs, "test")), batch_size=2, num_workers=a.workers)
        res = evaluate(model, te, dev, thr, a.tta)
        json.dump(res, open(f"{a.out}/test_metrics.json", "w"), indent=1)
        print("TEST:", {k: (round(v, 4) if isinstance(v, float) else np.round(v, 3)) for k, v in res.items()})

if __name__ == "__main__":
    main()
