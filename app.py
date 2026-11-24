"""FloodSight dashboard.  Run:  streamlit run app.py   (reads the saved results in ./results/siamese)"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import streamlit as st

R = os.environ.get("FLOODSIGHT_RESULTS", "results/siamese")
OUT = f"{R}/outputs"
CAL = f"{R}/calibrated/impact_test_calibrated.csv"
CELL, SHOW = 260, 650  # grid cell in full-res pixels, size of the displayed images

st.set_page_config(page_title="FloodSight", layout="wide")
st.title("FloodSight: uncertainty-aware flood impact assessment")
st.caption("Siamese multi-task U-Net (pre + post image) with MC Dropout. Test area: lou_0 (never seen in training). "
           "Pixel coordinates only: the aligned images carry no map coordinates, so there is no base map.")


@st.cache_data
def load_tables():
    imp = pd.read_csv(f"{OUT}/impact_per_image.csv")
    grid = pd.read_csv(f"{OUT}/grid_priority.csv")
    cal = pd.read_csv(CAL) if os.path.exists(CAL) else None
    return imp, grid, cal


@st.cache_data
def load_scene(id_):
    d = np.load(f"{OUT}/{id_}_maps.npz")
    ds = lambda a: a[:, ::2, ::2]
    return ds(d["mean"]).astype(np.float32) / 255, ds(d["std"]).astype(np.float32) / 255 * 0.25, ds(d["gt"])


imp, grid, cal = load_tables()
thr = json.load(open(f"{R}/thresholds.json"))
with_maps = sorted(os.path.basename(p).replace("_maps.npz", "") for p in glob.glob(f"{OUT}/*_maps.npz"))

tab1, tab2, tab3 = st.tabs(["Scene and impact", "Priority grid", "How good is it?"])

# ---------------------------------------------------------------- scene
with tab1:
    id_ = st.selectbox("Scene", with_maps + [i for i in imp.id if i not in with_maps],
                       format_func=lambda i: i + ("" if i in with_maps else "  (numbers only)"))
    row = imp[imp.id == id_].iloc[0]
    c = cal[cal.id == id_].iloc[0] if cal is not None and id_ in set(cal.id) else None

    st.subheader("Estimated flood impact")
    cols = st.columns(3)
    items = [("Flooded buildings", "flooded_buildings", "fb_low", "fb_high", "buildings", "{:.0f}"),
             ("Flooded road length (m)", "flooded_road_m", "road_low", "road_high", "road_m", "{:.0f}"),
             ("Flooded area (m²)", "flooded_area_m2", "area_low", "area_high", "area_m2", "{:.0f}")]
    for col, (name, mu, lo, hi, cn, f) in zip(cols, items):
        with col:
            if c is not None:
                st.metric(name, f.format(c[f"cal_{cn}"]), f"range {f.format(c[f'cal_{cn}_low'])} to {f.format(c[f'cal_{cn}_high'])}",
                          delta_color="off")
                st.caption("Calibrated 90% range (widened using the validation area)")
            else:
                st.metric(name, f.format(row[mu]), f"raw MC range {f.format(row[lo])} to {f.format(row[hi])}", delta_color="off")
                st.caption("Raw MC Dropout range: too narrow, see the third tab")
            if "gt_" + mu in row:
                st.write("Ground truth: " + f.format(row["gt_" + mu]))

    if id_ in with_maps:
        mean, std, gt = load_scene(id_)
        post = np.array(Image.open(f"{OUT}/{id_}_post.jpg"))
        pre = np.array(Image.open(f"{OUT}/{id_}_pre.jpg"))
        fb, fr = mean[2] > thr[2], mean[3] > thr[3]
        unc = 0.5 * (std[2] + std[3])
        show_gt = st.checkbox("Show ground truth instead of the prediction", value=False)
        fig, ax = plt.subplots(1, 4, figsize=(20, 5.2))
        ax[0].imshow(pre); ax[0].set_title("Before flood")
        ax[1].imshow(post); ax[1].set_title("After flood")
        ax[2].imshow(post)
        b, r = (gt[2] > 0, gt[3] > 0) if show_gt else (fb, fr)
        ov = np.zeros((*b.shape, 4)); ov[b] = [1, 0, 0, 0.6]; ov[r] = [1, 0.8, 0, 0.8]
        ax[2].imshow(ov); ax[2].set_title("Ground truth" if show_gt else "Predicted flooding (red buildings, yellow roads)")
        im = ax[3].imshow(unc, cmap="magma", vmin=0, vmax=max(float(unc.max()), 1e-3)); ax[3].set_title("Model uncertainty (flooded channels)")
        plt.colorbar(im, ax=ax[3], fraction=0.046)
        for a in ax: a.axis("off")
        plt.tight_layout(); st.pyplot(fig)
    else:
        st.info("Images are saved only for the first 6 test scenes; this one shows numbers only.")

# ---------------------------------------------------------------- grid
with tab2:
    st.write("Priority score per ~100 m cell = mean of three min-max-normalised impacts (flooded buildings, flooded road length, "
             "flooded area), equal weights. Hatched cells have above-average uncertainty (top quartile): treat them as low confidence.")
    id2 = st.selectbox("Scene", imp.id.tolist(), key="g", index=imp.id.tolist().index(id_) if id_ in imp.id.values else 0)
    g = grid[grid.id == id2].copy()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    if id2 in with_maps:
        ax.imshow(np.array(Image.open(f"{OUT}/{id2}_post.jpg")))
    else:
        ax.set_xlim(0, SHOW); ax.set_ylim(SHOW, 0)
    s = CELL // 2
    cmap = plt.get_cmap("Reds")
    for _, q in g.iterrows():
        x, y = q.x // 2, q.y // 2
        ax.add_patch(Rectangle((x, y), s, s, facecolor=cmap(min(q.priority, 1.0)), alpha=0.55, edgecolor="white",
                               hatch="//" if q.low_confidence else None, lw=0.8))
        ax.text(x + s / 2, y + s / 2, f"{q.priority:.2f}", ha="center", va="center", fontsize=9, color="black")
    ax.set_xlim(0, SHOW); ax.set_ylim(SHOW, 0); ax.axis("off")
    st.pyplot(fig)
    top = g.sort_values("priority", ascending=False).head(5)[["x", "y", "priority", "flooded_buildings", "flooded_road_m",
                                                              "flooded_area_m2", "uncertainty", "low_confidence"]]
    st.write("Five highest-priority cells"); st.dataframe(top.round(3), hide_index=True)

# ---------------------------------------------------------------- evaluation
with tab3:
    st.subheader("Model comparison (IoU, mean over the 4 output channels)")
    abl = pd.DataFrame([
        ["U-Net (post image only)", 0.126, 0.098], ["U-Net++ (post only)", 0.152, 0.109],
        ["DeepLabV3+ (post only)", 0.140, 0.099], ["Single-image multi-task (post only)", 0.138, 0.099],
        ["3D ResNet (pre + post)", 0.364, 0.249], ["U-Net (pre image only)", 0.375, 0.271],
        ["Siamese v2 (35 ep, flood loss x2, TTA)", 0.376, 0.231], ["Siamese multi-task (ours)", 0.398, 0.260]],
        columns=["Model", "Validation mIoU", "Test mIoU"])
    tvp = "results/siamese_trainval/test_metrics.json"
    if os.path.exists(tvp):
        abl.loc[len(abl)] = ["Siamese + validation regions in training", np.nan, json.load(open(tvp))["mean_iou"]]
    st.dataframe(abl, hide_index=True)
    st.caption("Thresholds are chosen on validation only. The test area is never used for any choice.")

    st.subheader("Are the impact ranges trustworthy?")
    sm = json.load(open(f"{OUT}/impact_summary.json"))
    rows = [[k, v["range_covers_truth"], v["mae"], v["mean_truth"], v["mean_pred"], v["corr"]] for k, v in sm.items()]
    st.write("Raw MC Dropout ranges (test area): share of images where the true value is inside the range")
    st.dataframe(pd.DataFrame(rows, columns=["quantity", "range covers truth", "mean abs error", "mean true", "mean predicted",
                                             "correlation"]).round(3), hide_index=True)
    cs = f"{R}/calibrated/calibration_summary.json"
    if os.path.exists(cs):
        s2 = json.load(open(cs))["summary"]
        st.write("After calibration on the validation area (target 90%)")
        st.dataframe(pd.DataFrame([[k, v["coverage_val"], v["coverage_test"], v["mae_raw_test"], v["mae_cal_test"]]
                                   for k, v in s2.items()],
                                  columns=["quantity", "coverage val", "coverage test", "MAE raw", "MAE calibrated"]).round(3),
                     hide_index=True)

    st.subheader("Calibration and uncertainty")
    cal_j = json.load(open(f"{OUT}/calibration.json"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for ch, col in (("flooded_building", "tab:red"), ("flooded_road", "tab:orange")):
        rel = cal_j["reliability"][ch]
        keep = [i for i, n in enumerate(rel["count"]) if n > 100]
        ax[0].plot([rel["mean_pred"][i] for i in keep], [rel["frac_positive"][i] for i in keep], "o-", color=col, label=ch)
        ue = cal_j["uncertainty_vs_error"][ch]
        keep = [i for i, n in enumerate(ue["count"]) if n > 100]
        ax[1].plot([(i + 0.5) / 10 * ue["std_max"] for i in keep], [ue["error_rate"][i] for i in keep], "o-", color=col, label=ch)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set_xlabel("predicted probability"); ax[0].set_ylabel("observed flooded fraction")
    ax[0].set_title("Reliability (flooded channels)"); ax[0].legend()
    ax[1].set_xlabel("model uncertainty (std over MC passes)"); ax[1].set_ylabel("pixel error rate")
    ax[1].set_title("Higher uncertainty should mean more errors"); ax[1].legend()
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("Limitations")
    st.markdown(
        "- Trained on 2 regions (Germany, Louisiana), validated on 1, tested on 1 more: flooded detection drops a lot on the unseen test area.\n"
        "- A model that sees only the pre-flood image scores about as well on test, so part of the flooded signal is location context, not seen water.\n"
        "- No flood-water mask exists in SpaceNet 8; flooding is only labelled on buildings and roads.\n"
        "- The post-flood images are coarser than the pre-flood images.\n"
        "- Raw MC Dropout ranges are far too narrow; the calibrated ranges are the ones to use.\n"
        "- Road length has almost no per-image skill. Metres use an assumed 0.4 m per pixel.")
