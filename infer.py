"""MC Dropout inference + uncertainty + flood impact ranges + grid priority.
python infer.py --model siamese --run runs/siamese_s1 --pairs P --folder F --split test --out outputs/siamese"""
import argparse, json, os
import numpy as np, pandas as pd, torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from data import FloodDataset, get_ids
from models import build_model

NB = 10  # bins for reliability / uncertainty curves


def enable_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            m.train()


def impact(bmask, rmask, fbmask, frmask, mpp):
    """bmask/rmask: building, road masks; fbmask/frmask: flooded building / road masks (bool arrays)."""
    lab, n = ndi.label(bmask)
    if n:
        tot = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        fl = np.bincount(lab.ravel(), weights=fbmask.ravel().astype(np.float32), minlength=n + 1)[1:]
        n_flood_b = int(((fl / np.maximum(tot, 1)) >= 0.5).sum())
    else:
        n_flood_b = 0
    skel = skeletonize(frmask)
    road_m = float(skel.sum()) * mpp
    area_m2 = float((fbmask | frmask).sum()) * mpp * mpp
    return n_flood_b, road_m, area_m2


def grid_stats(fb_mean, fr_mean, unc, thr, cell, mpp):
    """Per-cell flooded buildings / road length / flooded area / mean uncertainty from the averaged masks."""
    fb, fr = fb_mean > thr[2], fr_mean > thr[3]
    lab, n = ndi.label(fb)
    cents = ndi.center_of_mass(fb, lab, range(1, n + 1)) if n else []
    skel = skeletonize(fr)
    rows = []
    H = fb.shape[0]
    for y in range(0, H, cell):
        for x in range(0, H, cell):
            sl = (slice(y, y + cell), slice(x, x + cell))
            nb = sum(1 for cy, cx in cents if y <= cy < y + cell and x <= cx < x + cell)
            rows.append({"y": y, "x": x, "flooded_buildings": nb,
                         "flooded_road_m": float(skel[sl].sum()) * mpp,
                         "flooded_area_m2": float((fb[sl] | fr[sl]).sum()) * mpp * mpp,
                         "uncertainty": float(unc[sl].mean())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True, help="folder with best.pt and thresholds.json")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mpp", type=float, default=0.4, help="metres per pixel (approximate)")
    ap.add_argument("--cell", type=int, default=260, help="grid cell size in pixels (260 px ~ 100 m)")
    ap.add_argument("--save_maps", type=int, default=6, help="save probability/uncertainty maps for the first N images")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    model = build_model(a.model, weights="none", dropout=a.dropout).to(dev)
    model.load_state_dict(torch.load(f"{a.run}/best.pt", map_location=dev))
    thr = np.array(json.load(open(f"{a.run}/thresholds.json")), dtype=np.float32)
    enable_dropout(model)

    ids = get_ids(a.pairs, a.split)
    if a.limit:
        ids = ids[:a.limit]
    ds = FloodDataset(a.folder, ids)
    T = torch.tensor(thr, device=dev).view(1, 4, 1, 1)

    # calibration (reliability) and uncertainty-vs-error accumulators, per channel
    rel_n, rel_p, rel_y = (np.zeros((4, NB)) for _ in range(3))
    unc_n, unc_err = np.zeros((4, NB)), np.zeros((4, NB))
    UMAX = 0.25
    img_rows, grid_rows = [], []

    for i, id_ in enumerate(ids):
        pre, post, y = ds[i]
        pre, post = pre[None].to(dev), post[None].to(dev)
        y = y[:, :1300, :1300].to(dev)
        s1 = torch.zeros(1, 4, 1300, 1300, device=dev)
        s2 = torch.zeros_like(s1)
        counts = []
        for _ in range(a.passes):
            with torch.no_grad(), torch.autocast("cuda", enabled=dev == "cuda"):
                out = model(pre, post)
            p = torch.sigmoid(out.float())[:, :, :1300, :1300]
            s1 += p
            s2 += p * p
            b = (p > T)[0].cpu().numpy()
            counts.append(impact(b[0], b[1], b[2], b[3], a.mpp))
        mean = s1 / a.passes
        std = (s2 / a.passes - mean ** 2).clamp_min(0).sqrt()
        y_np = y.cpu().numpy()
        gt = impact(y_np[0] > 0.5, y_np[1] > 0.5, y_np[2] > 0.5, y_np[3] > 0.5, a.mpp)  # true values, same definition
        counts = np.array(counts)  # passes x 3
        lo, hi = np.percentile(counts, 5, axis=0), np.percentile(counts, 95, axis=0)
        mu = counts.mean(0)
        img_rows.append({"id": id_,
                         "flooded_buildings": mu[0], "fb_low": lo[0], "fb_high": hi[0],
                         "flooded_road_m": mu[1], "road_low": lo[1], "road_high": hi[1],
                         "flooded_area_m2": mu[2], "area_low": lo[2], "area_high": hi[2],
                         "mean_uncertainty": std[0, 2:].mean().item(),
                         "gt_flooded_buildings": gt[0], "gt_flooded_road_m": gt[1], "gt_flooded_area_m2": gt[2]})

        # reliability + uncertainty/error bins (pixels of the full image, per channel)
        m_np, s_np = mean[0].cpu().numpy(), std[0].cpu().numpy()
        for c in range(4):
            bi = np.minimum((m_np[c] * NB).astype(int), NB - 1).ravel()
            rel_n[c] += np.bincount(bi, minlength=NB)
            rel_p[c] += np.bincount(bi, weights=m_np[c].ravel(), minlength=NB)
            rel_y[c] += np.bincount(bi, weights=y_np[c].ravel(), minlength=NB)
            ui = np.minimum((s_np[c] / UMAX * NB).astype(int), NB - 1).ravel()
            err = ((m_np[c] > thr[c]) != (y_np[c] > 0.5)).ravel()
            unc_n[c] += np.bincount(ui, minlength=NB)
            unc_err[c] += np.bincount(ui, weights=err.astype(np.float64), minlength=NB)

        unc_fl = 0.5 * (s_np[2] + s_np[3])
        for r in grid_stats(m_np[2], m_np[3], unc_fl, thr, a.cell, a.mpp):
            grid_rows.append({"id": id_, **r})
        if i < a.save_maps:  # small files for the dashboard: uint8 maps + 650px jpegs of the two images
            from PIL import Image
            d = np.load(f"{a.folder}/{id_}.npz")
            np.savez_compressed(f"{a.out}/{id_}_maps.npz", mean=(m_np * 255).astype(np.uint8),
                                std=np.clip(s_np / UMAX * 255, 0, 255).astype(np.uint8), gt=d["mask"].astype(np.uint8))
            for k in ("pre", "post"):
                im = Image.fromarray(np.transpose(d[k], (1, 2, 0)).astype(np.uint8)).resize((650, 650))
                im.save(f"{a.out}/{id_}_{k}.jpg", quality=85)
        print(f"{i + 1}/{len(ids)} {id_}  buildings {mu[0]:.0f} [{lo[0]:.0f}-{hi[0]:.0f}]  "
              f"road {mu[1]:.0f}m  area {mu[2]:.0f}m2", flush=True)

    df = pd.DataFrame(img_rows)
    df.to_csv(f"{a.out}/impact_per_image.csv", index=False)
    summ = {}
    for nm, lo_, hi_, mu_, gt_ in [("flooded_buildings", "fb_low", "fb_high", "flooded_buildings", "gt_flooded_buildings"),
                                   ("flooded_road_m", "road_low", "road_high", "flooded_road_m", "gt_flooded_road_m"),
                                   ("flooded_area_m2", "area_low", "area_high", "flooded_area_m2", "gt_flooded_area_m2")]:
        inside = ((df[gt_] >= df[lo_]) & (df[gt_] <= df[hi_])).mean()
        summ[nm] = {"range_covers_truth": float(inside), "mae": float((df[mu_] - df[gt_]).abs().mean()),
                    "mean_truth": float(df[gt_].mean()), "mean_pred": float(df[mu_].mean()),
                    "corr": float(np.corrcoef(df[mu_], df[gt_])[0, 1]) if df[gt_].std() > 0 and df[mu_].std() > 0 else None,
                    "mean_range_width": float((df[hi_] - df[lo_]).mean())}
    json.dump(summ, open(f"{a.out}/impact_summary.json", "w"), indent=1)
    print("IMPACT vs truth:", json.dumps(summ))
    g = pd.DataFrame(grid_rows)
    # priority score: equal-weight mean of the three min-max-normalised impacts; low confidence if uncertainty is high
    for col in ["flooded_buildings", "flooded_road_m", "flooded_area_m2"]:
        rng = g[col].max() - g[col].min()
        g[col + "_n"] = (g[col] - g[col].min()) / rng if rng > 0 else 0.0
    g["priority"] = g[["flooded_buildings_n", "flooded_road_m_n", "flooded_area_m2_n"]].mean(axis=1)
    g["low_confidence"] = g["uncertainty"] > g["uncertainty"].quantile(0.75)
    g.to_csv(f"{a.out}/grid_priority.csv", index=False)

    e = 1e-9
    cal = {"reliability": {}, "uncertainty_vs_error": {}, "ece": {}}
    names = ["building", "road", "flooded_building", "flooded_road"]
    for c, nme in enumerate(names):
        cal["reliability"][nme] = {"mean_pred": (rel_p[c] / (rel_n[c] + e)).tolist(),
                                   "frac_positive": (rel_y[c] / (rel_n[c] + e)).tolist(),
                                   "count": rel_n[c].tolist()}
        cal["ece"][nme] = float((np.abs(rel_p[c] - rel_y[c])).sum() / rel_n[c].sum())
        cal["uncertainty_vs_error"][nme] = {"error_rate": (unc_err[c] / (unc_n[c] + e)).tolist(),
                                            "count": unc_n[c].tolist(), "std_max": UMAX}
    json.dump(cal, open(f"{a.out}/calibration.json", "w"), indent=1)
    print("ECE:", {k: round(v, 4) for k, v in cal["ece"].items()})


if __name__ == "__main__":
    main()
