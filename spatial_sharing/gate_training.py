"""Frozen-base feature caches and reconstruction-MSE strength learning."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from .data import read_manifest, PairedImages, assert_disjoint, TASKS, validation_subset
from .engine import restore_checkpoint
from .metrics import image_metrics, aggregate
from .modules import StrengthPredictor, gate_features
from .projection import project_affine
from .runtime import sha256, write_json, seed_everything, atomic_save, append_log


@torch.no_grad()
def cache_outputs(checkpoint, manifest, root, directory, split, device="cpu", per_task=2000, seed=42):
    if per_task < 1:
        raise ValueError("per_task must be positive.")
    model, config = restore_checkpoint(checkpoint, device)
    if config["model"]["variant"] != "B2":
        raise ValueError("Paper gate caches require a frozen B2 checkpoint.")
    correction = model.correction
    if correction.log_gain_bound != .4 or correction.bias_bound != .1:
        raise ValueError("The Box-LS gate currently uses the paper's .4/.1 bounds.")
    rows = read_manifest(manifest, root)
    # Check requested split membership against the base model's own manifests.
    base_data = config["data"]
    base_key = {"train": "train_manifest", "val": "val_manifest", "test": "test_manifest"}[split]
    base_rows = read_manifest(base_data[base_key], base_data["root"])
    signature = lambda r: tuple(r[k] for k in ("id", "scene", "task", "input", "target"))
    allowed = {signature(r) for r in base_rows}
    if any(signature(r) not in allowed for r in rows):
        raise ValueError("Cache records do not belong to the declared base-model split.")
    if split == "val":
        rows = validation_subset(rows, config["training"]["validation_limit"])
    if split == "train":
        rng = np.random.default_rng(seed)
        chosen = []
        for task in TASKS:
            task_rows = [r for r in rows if r["task"] == task]
            # Select at most one record per scene, so repeated crops aren't new scenes.
            seen = set()
            for i in rng.permutation(len(task_rows)):
                row = task_rows[i]
                if row["scene"] not in seen and len(seen) < per_task:
                    chosen.append(row)
                    seen.add(row["scene"])
        rows = chosen
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Cache directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index, sample in enumerate(PairedImages(rows)):
        out = model(sample["input"][None].to(device))
        arrays = {k: out[k][0].cpu().numpy() for k in ("y0", "corrected", "gain", "bias")}
        shared, _, _ = project_affine(arrays["y0"], arrays["corrected"], bounded=True)
        delta = arrays["corrected"].astype(np.float64)-shared
        target = sample["target"].numpy().astype(np.float64)
        r = target-shared
        features = gate_features(out["y0"], out["gain"], out["bias"])[0].cpu().numpy()
        moments = np.array([np.mean(r*r), np.mean(r*delta), np.mean(delta*delta)])
        file = f"{index:06d}.npz"
        # Train cache needs only features/K/T/Q; validation keeps image arrays for Y-PSNR selection.
        fields = dict(features=features, moments=moments)
        if split != "train":
            fields.update(shared=shared, delta=delta, target=target)
        np.savez_compressed(directory/file, **fields)
        records.append({k: sample[k] for k in ("id", "scene", "task")})
        records[-1].update(file=file, input=rows[index]["input"], target=rows[index]["target"])
    metadata = {"split": split, "base_sha256": sha256(checkpoint), "backend": config["model"]["backend"],
                "manifest_sha256": sha256(manifest), "seed": seed,
                "feature_order": "Y0_mean,Y0_std,delta_mean,delta_rms,g_mean,g_std,b_mean,b_std; RGB per block",
                "records": records}
    write_json(directory/"index.json", metadata)
    return metadata


def read_cache(directory, expected_split):
    directory = Path(directory)
    meta = json.loads((directory/"index.json").read_text())
    if meta["split"] != expected_split or not meta["records"]:
        raise ValueError(f"Expected a nonempty {expected_split} cache.")
    features, moments = [], []
    for row in meta["records"]:
        with np.load(directory/row["file"], allow_pickle=False) as a:
            features.append(a["features"])
            moments.append(a["moments"])
    return meta, torch.tensor(np.stack(features), dtype=torch.float32), torch.tensor(np.stack(moments), dtype=torch.float32)


@torch.no_grad()
def evaluate_gate(gate, directory, metadata, features, constant=None):
    gate.eval()
    strengths = gate(features).cpu().numpy() if constant is None else np.full(len(features), constant)
    records = []
    for row, strength in zip(metadata["records"], strengths):
        with np.load(Path(directory)/row["file"], allow_pickle=False) as a:
            score = image_metrics(a["shared"]+float(strength)*a["delta"], a["target"], full=False)
        records.append({"task": row["task"], "metrics": score})
    return aggregate(records)["macro"]["y_psnr"]


def train_gate(train_directory, val_directory, run, config, device="cpu"):
    train_meta, features, moments = read_cache(train_directory, "train")
    val_meta, val_features, _ = read_cache(val_directory, "val")
    assert_disjoint(train_meta["records"], val_meta["records"])
    if train_meta["base_sha256"] != val_meta["base_sha256"]:
        raise ValueError("Train/validation caches must use the same frozen B2.")
    if {r["task"] for r in val_meta["records"]} != set(TASKS):
        raise ValueError("Gate validation needs all three tasks.")
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        raise FileExistsError("Use a fresh gate run directory.")
    run.mkdir(parents=True, exist_ok=True)
    seed_everything(config["seed"])
    gate = StrengthPredictor(config["dropout"]).to(device)
    features, moments, val_features = features.to(device), moments.to(device), val_features.to(device)
    gate.fit_normalizer(features)  # Training-only standardization, buffers saved with weights.
    optimizer = torch.optim.AdamW(gate.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best, bad_checks = -float("inf"), 0
    generator = torch.Generator().manual_seed(config["seed"])
    for step in range(config["updates"]):
        gate.train()
        lr = config["min_lr"]+.5*(config["lr"]-config["min_lr"])*(1+math.cos(math.pi*step/max(1, config["updates"]-1)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        indices = torch.randint(len(features), (config["batch_size"],), generator=generator).to(device)
        strength = gate(features[indices])
        k, t, q = moments[indices].unbind(dim=1)
        loss = (k-2*strength*t+strength.square()*q).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite gate loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), config["clip_grad"])
        optimizer.step()
        record = {"step": step+1, "loss": float(loss.detach()), "lr": lr}
        if (step+1) % config["validate_every"] == 0 or step+1 == config["updates"]:
            score = evaluate_gate(gate, val_directory, val_meta, val_features)
            record["val_macro_y_psnr"] = score
            if score > best:
                best, bad_checks = score, 0
                atomic_save(run/"best.pt", {"model": gate.state_dict(), "dropout": config["dropout"],
                    "base_sha256": train_meta["base_sha256"], "config": config, "next_update": step+1, "best_psnr": best})
            else:
                bad_checks += 1
            print(record, flush=True)
        append_log(run/"training.jsonl", record)
        if bad_checks >= config["patience"]:
            break
    constants = [{"strength": float(lam), "val_macro_y_psnr": evaluate_gate(gate, val_directory, val_meta, val_features, lam)}
                 for lam in config["constant_grid"]]
    write_json(run/"selection.json", {"best_gate_validation_psnr": best, "constant_grid": constants,
        "selected_constant": max(constants, key=lambda r: r["val_macro_y_psnr"]), "config": config,
        "train_index_sha256": sha256(Path(train_directory)/"index.json"), "val_index_sha256": sha256(Path(val_directory)/"index.json")})
    return run/"best.pt"
