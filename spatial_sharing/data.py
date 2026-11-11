"""Explicit scene manifests; no filename guessing or test-driven splits."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

TASKS = ("rain", "haze", "snow")


def read_manifest(path, root):
    root = Path(root).expanduser().resolve()
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    ids = set()
    for row in rows:
        if not all(row.get(k) for k in ("id", "scene", "task", "input", "target")):
            raise ValueError("Required CSV columns: id,scene,task,input,target")
        if row["task"] not in TASKS or row["id"] in ids:
            raise ValueError("Task must be rain/haze/snow and image IDs must be unique.")
        ids.add(row["id"])
        for key in ("input", "target"):
            row[key] = str((root / row[key]).resolve())
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
    return rows


def assert_disjoint(*manifests):
    """Scene IDs are globally unique across weather tasks; shared targets also count."""
    for i, first in enumerate(manifests):
        for second in manifests[i + 1:]:
            for key in ("id", "scene"):
                overlap = {r[key] for r in first} & {r[key] for r in second}
                if overlap:
                    raise ValueError(f"Split leakage ({key}): {sorted(overlap)[:3]}")
            paths_a = {r[k] for r in first for k in ("input", "target")}
            paths_b = {r[k] for r in second for k in ("input", "target")}
            if paths_a & paths_b:
                raise ValueError("An image path occurs in more than one split.")


def load_rgb(path):
    with Image.open(path) as image:
        return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy()).permute(2, 0, 1) / 255


class PairedImages(Dataset):
    def __init__(self, rows, crop=None, seed=42):
        self.rows, self.crop, self.seed = rows, crop, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # Sampler provides a draw ID, so augmentations survive workers/resume deterministically.
        draw = 0
        if isinstance(index, tuple):
            index, draw = index
        row = self.rows[index]
        x, y = load_rgb(row["input"]), load_rgb(row["target"])
        if x.shape != y.shape:
            raise ValueError(f"Unaligned pair: {row['id']} {x.shape} != {y.shape}")
        generator = torch.Generator().manual_seed(self.seed + draw)
        if self.crop:
            h, w = x.shape[-2:]
            c = self.crop
            if min(h, w) < c:
                raise ValueError(f"Image {row['id']} is smaller than crop {c}; no implicit resize.")
            def crop_pair(a, b):
                top = int(torch.randint(h - c + 1, (), generator=generator))
                left = int(torch.randint(w - c + 1, (), generator=generator))
                a, b = a[:, top:top+c, left:left+c], b[:, top:top+c, left:left+c]
                for axis in (-1, -2):
                    if bool(torch.randint(2, (), generator=generator)):
                        a, b = a.flip(axis), b.flip(axis)
                k = int(torch.randint(4, (), generator=generator))
                return a.rot90(k, (-2, -1)), b.rot90(k, (-2, -1))
            query, target = crop_pair(x, y)
            key, _ = crop_pair(x, y)
            x, y = query, target
        else:
            h, w = (v // 16 * 16 for v in x.shape[-2:])
            if min(h, w) < 16:
                raise ValueError("Evaluation requires images at least 16 x 16.")
            x, y = x[:, :h, :w], y[:, :h, :w]
            key = x
        return {"input": x, "target": y, "key": key, "id": row["id"], "scene": row["scene"], "task": row["task"]}


class BalancedBatches(Sampler):
    """Round-robin individual draws, giving exact total budgets when divisible by 3."""
    def __init__(self, rows, batch_size, updates, seed=42, start=0):
        self.groups = [[i for i, row in enumerate(rows) if row["task"] == task] for task in TASKS]
        if any(not g for g in self.groups):
            raise ValueError("Training requires all three tasks.")
        self.batch_size, self.updates, self.seed, self.start = batch_size, updates, seed, start

    def __iter__(self):
        for step in range(self.start, self.updates):
            rng = np.random.default_rng(self.seed + step)
            yield [(int(rng.choice(self.groups[(step*self.batch_size+j) % 3])), step*self.batch_size+j)
                   for j in range(self.batch_size)]

    def __len__(self):
        return self.updates - self.start


def validation_subset(rows, limit=128):
    counts, result = defaultdict(int), []
    for row in rows:
        if counts[row["task"]] < limit:
            result.append(row)
            counts[row["task"]] += 1
    if set(counts) != set(TASKS):
        raise ValueError("Validation requires rain, haze, and snow.")
    return result
