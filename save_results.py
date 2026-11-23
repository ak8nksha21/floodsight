"""Pack one trained run into results/<name>/ (fp16 weights + small json/csv files) so it can be committed to GitHub.
python save_results.py --name siamese --run runs/siamese_s1 [--outputs outputs/siamese]"""
import argparse, os, shutil
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--outputs", default=None)
ap.add_argument("--dest", default="results")
a = ap.parse_args()
d = f"{a.dest}/{a.name}"
os.makedirs(d, exist_ok=True)
sd = torch.load(f"{a.run}/best.pt", map_location="cpu")
torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}, f"{d}/best_fp16.pth")
for f in ["thresholds.json", "best_val.json", "history.json", "test_metrics.json"]:
    if os.path.exists(f"{a.run}/{f}"):
        shutil.copy(f"{a.run}/{f}", d)
if a.outputs and os.path.isdir(a.outputs):
    shutil.copytree(a.outputs, f"{d}/outputs", dirs_exist_ok=True)
print("saved", d, round(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) / 1e6, 1), "MB")
