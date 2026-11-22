"""Data loading for FloodSight: reads the aligned pairs made in the data-prep notebook."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Whole areas go to one split, so neighbouring tiles never end up in different splits.
SPLIT_AREAS = {"train": ["ger_0", "lou_1"], "val": ["lou_2"], "test": ["lou_0"]}
CHANNELS = ["building", "road", "flooded_building", "flooded_road"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def get_ids(pairs_csv, split):
    df = pd.read_csv(pairs_csv)
    df = df[df.status == "kept"]
    area = df.id.str.split("_").str[0] + "_" + df.id.str.split("_").str[1]
    return df.id[area.isin(SPLIT_AREAS[split])].tolist()


def _norm(x):
    return (x.astype(np.float32) / 255.0 - MEAN) / STD


class FloodDataset(Dataset):
    """train=True : random crop + flips/rotations + brightness, `repeat` crops per pair per epoch.
    train=False: the full 1300x1300 pair, reflect-padded to 1312 (a multiple of 32), mask padded
    with zeros. Cut predictions back to 1300x1300 before scoring."""

    def __init__(self, folder, ids, train=False, crop=512, repeat=2):
        self.folder, self.ids, self.train, self.crop = folder, list(ids), train, crop
        self.repeat = repeat if train else 1

    def __len__(self):
        return len(self.ids) * self.repeat

    def __getitem__(self, i):
        d = np.load(f"{self.folder}/{self.ids[i % len(self.ids)]}.npz")
        pre, post, mask = d["pre"], d["post"], d["mask"]
        if self.train:
            rng = np.random.default_rng()
            c = self.crop
            y, x = rng.integers(0, pre.shape[1] - c + 1, 2)
            pre, post, mask = (a[:, y:y + c, x:x + c] for a in (pre, post, mask))
            k = int(rng.integers(0, 4))
            pre, post, mask = (np.rot90(a, k, axes=(1, 2)) for a in (pre, post, mask))
            if rng.random() < 0.5:
                pre, post, mask = (a[:, :, ::-1] for a in (pre, post, mask))
            pre = np.clip(pre * rng.uniform(0.85, 1.15), 0, 255)
            post = np.clip(post * rng.uniform(0.85, 1.15), 0, 255)
        else:
            p = 1312 - pre.shape[1]
            pad = ((0, 0), (0, p), (0, p))
            pre, post = (np.pad(a, pad, mode="reflect") for a in (pre, post))
            mask = np.pad(mask, pad)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return t(_norm(pre)), t(_norm(post)), t(mask).float()
