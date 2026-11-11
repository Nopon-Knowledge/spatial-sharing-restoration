"""Exercise the full pipeline with clearly marked synthetic test fixtures."""
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .analysis import analyze_arrays
from .data import read_manifest, TASKS
from .engine import train, restore_checkpoint, evaluate_model, infer, restore_gate
from .gate_training import cache_outputs, train_gate
from .runtime import read_config, write_json


def run_smoke(directory):
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Choose an empty smoke output directory.")
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(123)
    for split in ("train", "val", "test"):
        rows = []
        for task in TASKS:
            for index in range(2):
                identity = f"SYNTHETIC_{split}_{task}_{index}"
                target = rng.uniform(.15, .8, (32, 32, 3))
                x = np.clip(target*.85+.08+rng.normal(0, .02, target.shape), 0, 1)
                for suffix, array in (("input", x), ("target", target)):
                    Image.fromarray(np.rint(array*255).astype("uint8")).save(directory/f"{identity}_{suffix}.png")
                rows.append(dict(id=identity, scene=identity, task=task, input=f"{identity}_input.png", target=f"{identity}_target.png"))
        with (directory/f"{split}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    config = {
        "device": "cpu", "model": {"backend": "smoke", "variant": "B2"},
        "data": {"root": str(directory), **{f"{s}_manifest": str(directory/f"{s}.csv") for s in ("train", "val", "test")}},
        "training": {"seed": 42, "batch_size": 2, "crop": 32, "workers": 0, "betas": [.9, .999],
            "eps": 1e-8, "weight_decay": 0., "state_weight": 1., "validation_limit": 2, "validate_every": 3,
            "encoder": {"updates": 3, "lr": .001}, "joint": {"updates": 3, "lr": .0001, "min_lr": .000001}}
    }
    checkpoint = train(config, directory/"restoration")
    # Resume from the completed encoder snapshot and compare the joint endpoint.
    resumed_checkpoint = train(config, directory/"resume_check", directory/"restoration"/"encoder_last.pt")
    model, _ = restore_checkpoint(checkpoint, "cpu")
    resumed_model, _ = restore_checkpoint(resumed_checkpoint, "cpu")
    if any(not torch.equal(a, b) for a, b in zip(model.state_dict().values(), resumed_model.state_dict().values())):
        raise AssertionError("Resumed model differs from uninterrupted model.")
    rows = read_manifest(directory/"test.csv", directory)
    report = evaluate_model(model, rows, "cpu", full=True, output_dir=directory/"evaluation")
    with np.load(directory/"evaluation"/"000000.npz") as a:
        analysis, _ = analyze_arrays(a["y0"], a["corrected"], a["gain"], a["bias"], a["target"])
    for split in ("train", "val"):
        cache_outputs(checkpoint, directory/f"{split}.csv", directory, directory/f"cache_{split}", split, per_task=2)
    gate_config = {"seed": 42, "dropout": .1, "batch_size": 4, "updates": 4, "lr": .0003,
                   "min_lr": .000001, "weight_decay": .0001, "clip_grad": 1., "validate_every": 2,
                   "patience": 5, "constant_grid": [0., .5, 1.]}
    gate_checkpoint = train_gate(directory/"cache_train", directory/"cache_val", directory/"gate", gate_config)
    gate = restore_gate(gate_checkpoint, checkpoint, "cpu")
    gated = evaluate_model(model, rows, "cpu", full=True, gate=gate)
    infer(checkpoint, [rows[0]["input"]], directory/"inference", "cpu", gate_checkpoint)
    result = {"status": "passed", "scope": "Synthetic CPU integration test only; no WeatherBench reproduction",
              "encoder_updates": 3, "joint_updates": 3, "gate_updates": 4, "evaluation_images": len(rows),
              "resume_exact": True, "decomposition_relative_error": analysis["identity_relative_error"],
              "gate_parameters": sum(p.numel() for p in gate.parameters()), "finite_gate_metrics": bool(np.isfinite(gated["macro"]["y_psnr"]))}
    write_json(directory/"smoke_report.json", result)
    print(result)
