"""Inference-only improvements: ensemble of saved models + 'flooded must be building/road' gating + thresholds tuned on validation.
python ensemble_eval.py --pairs P --folder F --results results --out results/ensemble"""
import argparse, json, os
import numpy as np, torch
from data import FloodDataset, get_ids
from models import build_model

MODELS = [("siamese", "siamese"), ("siamese_v2", "siamese"), ("resnet3d", "resnet3d")]  # (results folder, architecture)
THS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
VARIANTS = ["single", "ens", "ens_gate"]
GATE = 0.3  # a flooded building / road is only kept where the building / road probability is above this


def load(name, arch, results, dev):
    m = build_model(arch, weights="none").to(dev)
    sd = torch.load(f"{results}/{name}/best_fp16.pth", map_location=dev)
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
    return m.eval()


@torch.no_grad()
def predict(model, pre, post, tta, dev):
    with torch.autocast("cuda", enabled=dev == "cuda"):
        p = torch.sigmoid(model(pre, post).float())
        if tta:
            for dims in ([3], [2]):
                p = p + torch.sigmoid(model(pre.flip(dims), post.flip(dims)).float()).flip(dims)
            p = p / 3
    return p[:, :, :1300, :1300]


def variant_probs(ps):
    single = ps[0]
    ens = torch.stack(ps).mean(0)
    gate = ens.clone()
    gate[:, 2] = gate[:, 2] * (ens[:, 0] > GATE)
    gate[:, 3] = gate[:, 3] * (ens[:, 1] > GATE)
    return {"single": single, "ens": ens, "ens_gate": gate}


def run(split, args, models, dev, thr=None):
    ids = get_ids(args.pairs, split)[: args.limit or None]
    dl = torch.utils.data.DataLoader(FloodDataset(args.folder, ids), batch_size=1, num_workers=2)
    K = len(THS) if thr is None else 1
    acc = {v: torch.zeros(3, K, 4, dtype=torch.float64, device=dev) for v in VARIANTS}  # tp, fp, fn
    for i, (pre, post, y) in enumerate(dl):
        pre, post, y = pre.to(dev), post.to(dev), y[:, :, :1300, :1300].to(dev)
        vp = variant_probs([predict(m, pre, post, args.tta, dev) for m in models])
        for v in VARIANTS:
            for k in range(K):
                t = (torch.tensor(THS[k], device=dev).expand(4) if thr is None else torch.tensor(thr[v], device=dev)).view(1, 4, 1, 1)
                b = (vp[v] > t).float()
                acc[v][0, k] += (b * y).sum((0, 2, 3)).double()
                acc[v][1, k] += (b * (1 - y)).sum((0, 2, 3)).double()
                acc[v][2, k] += ((1 - b) * y).sum((0, 2, 3)).double()
        if (i + 1) % 20 == 0: print(split, i + 1, "/", len(ids), flush=True)
    out = {}
    for v in VARIANTS:
        tp, fp, fn = acc[v]
        iou = tp / (tp + fp + fn + 1e-9)
        if thr is None:
            best = iou.argmax(0)
            ar = torch.arange(4, device=dev)
            tp, fp, fn, iou = tp[best, ar], fp[best, ar], fn[best, ar], iou[best, ar]
            th = [THS[j] for j in best.tolist()]
        else:
            tp, fp, fn, iou = tp[0], fp[0], fn[0], iou[0]
            th = thr[v]
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
        out[v] = {"thresholds": th, "iou": iou.tolist(), "f1": f1.tolist(), "precision": (tp / (tp + fp + 1e-9)).tolist(),
                  "recall": (tp / (tp + fn + 1e-9)).tolist(), "mean_iou": iou.mean().item(), "mean_f1": f1.mean().item()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True); ap.add_argument("--folder", required=True)
    ap.add_argument("--results", default="results"); ap.add_argument("--out", default="results/ensemble")
    ap.add_argument("--tta", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = [load(n, arch, a.results, dev) for n, arch in MODELS]
    os.makedirs(a.out, exist_ok=True)
    val = run("val", a, models, dev)                                   # thresholds chosen here
    thr = {v: val[v]["thresholds"] for v in VARIANTS}
    test = run("test", a, models, dev, thr)                            # fixed thresholds, test looked at once
    json.dump({"val": val, "test": test, "gate": GATE, "models": [m[0] for m in MODELS], "tta": a.tta},
              open(f"{a.out}/ensemble_metrics.json", "w"), indent=1)
    print(f"{'variant':10s} {'VAL mIoU':>9s} {'TEST mIoU':>10s} | TEST IoU [bld road fl_bld fl_road]  thr")
    for v in VARIANTS:
        print(f"{v:10s} {val[v]['mean_iou']:9.4f} {test[v]['mean_iou']:10.4f} | {np.round(test[v]['iou'], 3)}  {thr[v]}")
    json.dump({"models": [m[0] for m in MODELS]}, open(f"{a.out}/models.json", "w"))


if __name__ == "__main__":
    main()
