"""Calibrate the impact ranges on the VALIDATION area, then apply them to the TEST area.
python calibrate.py --val outputs/val/impact_per_image.csv --test outputs/test/impact_per_image.csv --out outputs/calibrated"""
import argparse, json, os
import numpy as np, pandas as pd

Q = [("flooded_buildings", "gt_flooded_buildings", 1.0, "buildings"),
     ("flooded_road_m", "gt_flooded_road_m", 50.0, "road_m"),
     ("flooded_area_m2", "gt_flooded_area_m2", 500.0, "area_m2")]

ap = argparse.ArgumentParser()
ap.add_argument("--val", required=True)
ap.add_argument("--test", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--level", type=float, default=0.9)
a = ap.parse_args()
v, t = pd.read_csv(a.val), pd.read_csv(a.test)
lo_q, hi_q = (1 - a.level) / 2, 1 - (1 - a.level) / 2
summ, params = {}, {}
for mu, gt, eps, nm in Q:
    r = (v[gt] + eps) / (v[mu] + eps)            # how far off the model typically is on validation (ratio)
    lo, hi, med = r.quantile(lo_q), r.quantile(hi_q), r.median()
    params[nm] = {"ratio_low": float(lo), "ratio_median": float(med), "ratio_high": float(hi), "eps": eps}
    for d, name in ((v, "val"), (t, "test")):
        d[f"cal_{nm}"] = np.maximum((d[mu] + eps) * med - eps, 0)
        d[f"cal_{nm}_low"] = np.maximum((d[mu] + eps) * lo - eps, 0)
        d[f"cal_{nm}_high"] = (d[mu] + eps) * hi - eps
    cov = lambda d: float(((d[gt] >= d[f"cal_{nm}_low"]) & (d[gt] <= d[f"cal_{nm}_high"])).mean())
    summ[nm] = {"coverage_val": cov(v), "coverage_test": cov(t),
                "mae_raw_test": float((t[mu] - t[gt]).abs().mean()),
                "mae_cal_test": float((t[f"cal_{nm}"] - t[gt]).abs().mean()),
                "mean_truth_test": float(t[gt].mean()), "mean_cal_pred_test": float(t[f"cal_{nm}"].mean()),
                "mean_width_test": float((t[f"cal_{nm}_high"] - t[f"cal_{nm}_low"]).mean())}
os.makedirs(a.out, exist_ok=True)
t.to_csv(f"{a.out}/impact_test_calibrated.csv", index=False)
json.dump({"params": params, "summary": summ, "level": a.level}, open(f"{a.out}/calibration_summary.json", "w"), indent=1)
print("CALIBRATED:", json.dumps(summ))
