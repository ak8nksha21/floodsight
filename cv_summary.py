"""Mean +- std of the held-out-area results.  python cv_summary.py results"""
import glob, json, sys, numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for d in sorted(glob.glob(f"{R}/cv_*")):
    try: rows.append((d.split("/")[-1], json.load(open(f"{d}/best_val.json")), json.load(open(f"{d}/test_metrics.json"))))
    except Exception: pass
print(f"{'fold':12s} {'VAL mIoU':>8s} {'TEST mIoU':>9s} | TEST IoU [bld road fl_bld fl_road]")
for n, v, t in rows: print(f"{n:12s} {v['mean_iou']:8.3f} {t['mean_iou']:9.3f} | {np.round(t['iou'], 3)}")
if rows:
    T = np.array([t["iou"] for _, _, t in rows]); m = T.mean(1)
    print(f"{'MEAN+-STD':12s} {'':8s} {m.mean():5.3f}+-{m.std():.3f} | {np.round(T.mean(0), 3)} +- {np.round(T.std(0), 3)}")
