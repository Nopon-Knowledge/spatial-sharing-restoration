"""Single-GPU, two-stage training and image-level evaluation."""
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import read_manifest, assert_disjoint, validation_subset, PairedImages, BalancedBatches, load_rgb
from .metrics import image_metrics, aggregate, LPIPSMetric
from .model import build_model
from .modules import StrengthPredictor, gate_features
from .projection import project_affine
from .runtime import seed_everything, write_json, append_log, atomic_save, load_checkpoint, provenance, sha256


@torch.no_grad()
def evaluate_model(model, rows, device, full=False, output_dir=None, perceptual=False, gate=None):
    was_training = model.training
    model.eval()
    records = []
    lpips_metric = LPIPSMetric(device) if perceptual else None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(PairedImages(rows)):
        out = model(sample["input"][None].to(device))
        arrays = {key: out[key][0].cpu().numpy() for key in ("y0", "gain", "bias", "corrected")}
        arrays["target"] = sample["target"].numpy()
        output = arrays["corrected"]
        if gate is not None:
            shared, _, _ = project_affine(arrays["y0"], output, bounded=True)
            features = gate_features(out["y0"], out["gain"], out["bias"])
            strength = float(gate(features).item())
            output = shared+strength*(output-shared)
            arrays.update(gated=output, strength=np.array(strength))
        scores = image_metrics(output, arrays["target"], full)
        if lpips_metric:
            scores["lpips_alex"] = lpips_metric(output, arrays["target"])
        records.append({k: sample[k] for k in ("id", "scene", "task")})
        records[-1]["metrics"] = scores
        if output_dir:
            name = f"{index:06d}"
            np.savez_compressed(output_dir/(name+".npz"), **arrays)
            Image.fromarray(np.rint(np.clip(output, 0, 1).transpose(1, 2, 0)*255).astype("uint8")).save(output_dir/(name+".png"))
            records[-1]["file"] = name+".npz"
    model.train(was_training)
    return {"records": records, **aggregate(records)}


def train(config, run, resume=None):
    settings = config["training"]
    seed_everything(settings["seed"])
    device = config["device"]
    if settings["batch_size"] < 1 or settings["crop"] < 16:
        raise ValueError("Batch must be positive and crop at least 16.")
    if config["model"]["backend"] == "airnet" and device not in ("cuda", "cuda:0"):
        raise ValueError("Official AirNet training requires device=cuda:0.")
    data = config["data"]
    training = read_manifest(data["train_manifest"], data["root"])
    validation = read_manifest(data["val_manifest"], data["root"])
    all_splits = [training, validation]
    if data.get("test_manifest"):
        all_splits.append(read_manifest(data["test_manifest"], data["root"]))
    assert_disjoint(*all_splits)
    validation = validation_subset(validation, settings["validation_limit"])
    run = Path(run)
    if run.exists() and any(run.iterdir()) and not resume:
        raise FileExistsError("Run directory is not empty; choose a new run or --resume.")
    run.mkdir(parents=True, exist_ok=True)
    info = provenance(config)
    model = build_model(config).to(device)
    resumed = load_checkpoint(resume) if resume else None
    if resumed and (resumed["config"] != config or resumed["manifest_sha256"] != info["manifest_sha256"]):
        raise ValueError("Resume config/manifests differ from the checkpoint.")
    write_json(run/"provenance.json", info)
    if resumed:
        model.load_state_dict(resumed["model"])
    best = resumed["best_psnr"] if resumed else -float("inf")
    for stage_index, stage in enumerate(("encoder", "joint")):
        stage_settings = settings[stage]
        updates = stage_settings["updates"]
        if updates < 1:
            raise ValueError("Each stage needs at least one update.")
        if resumed and stage_index < resumed["stage_index"]:
            continue
        optimizer = torch.optim.Adam(model.parameters(), lr=stage_settings["lr"],
                                     betas=tuple(settings["betas"]), eps=settings["eps"], weight_decay=settings["weight_decay"])
        start = 0
        if resumed and stage_index == resumed["stage_index"]:
            optimizer.load_state_dict(resumed["optimizer"])
            start = resumed["next_update"]
            torch.set_rng_state(resumed["torch_rng"])
            if torch.cuda.is_available() and resumed.get("cuda_rng"):
                torch.cuda.set_rng_state_all(resumed["cuda_rng"])
        batch_sampler = BalancedBatches(training, settings["batch_size"], updates,
                                         settings["seed"]+stage_index*1000000, start)
        loader = DataLoader(PairedImages(training, settings["crop"], settings["seed"]+stage_index*1000000),
                            batch_sampler=batch_sampler, num_workers=settings["workers"],
                            generator=torch.Generator().manual_seed(settings["seed"]))
        model.train()
        for step, batch in enumerate(loader, start=start):
            if stage == "encoder":
                lr = stage_settings["lr"] * (.1 if step >= int(.6*updates) else 1)
            else:
                progress = step / max(1, updates-1)
                lr = stage_settings["min_lr"] + .5*(stage_settings["lr"]-stage_settings["min_lr"])*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, target, key = (batch[k].to(device) for k in ("input", "target", "key"))
            optimizer.zero_grad(set_to_none=True)
            out = model(x, key, encoder_only=stage == "encoder")
            terms = model.loss(out, x, target, encoder_only=stage == "encoder", state_weight=settings["state_weight"])
            if not torch.isfinite(terms["total"]):
                raise FloatingPointError(f"Nonfinite loss at {stage} update {step+1}")
            terms["total"].backward()
            optimizer.step()
            log = {"stage": stage, "update": step+1, "lr": lr, **{k: float(v.detach()) for k, v in terms.items()}}
            append_log(run/"training.jsonl", log)
            check = (step+1) % settings["validate_every"] == 0 or step+1 == updates
            if not check:
                continue
            improved = False
            if stage == "joint":
                report = evaluate_model(model, validation, device)
                score = report["macro"]["y_psnr"]
                log["val_macro_y_psnr"] = score
                write_json(run/"validation_latest.json", report)
                improved = score > best
                best = max(best, score)
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                     "stage_index": stage_index, "next_update": step+1, "best_psnr": best,
                     "manifest_sha256": info["manifest_sha256"], "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
            atomic_save(run/"last.pt", state)
            if stage == "encoder" and step+1 == updates:
                atomic_save(run/"encoder_last.pt", state)
            if improved:
                atomic_save(run/"best.pt", state)
            print(log, flush=True)
        resumed = None
    return run/"best.pt"


def restore_checkpoint(path, device):
    checkpoint = load_checkpoint(path)
    config = checkpoint["config"]
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def restore_gate(path, base_checkpoint, device):
    checkpoint = load_checkpoint(path)
    if checkpoint["base_sha256"] != sha256(base_checkpoint):
        raise ValueError("Gate checkpoint belongs to a different frozen restoration checkpoint.")
    gate = StrengthPredictor(checkpoint["dropout"]).to(device)
    gate.load_state_dict(checkpoint["model"])
    return gate.eval()


@torch.no_grad()
def infer(checkpoint, inputs, output, device="cpu", gate_checkpoint=None):
    model, config = restore_checkpoint(checkpoint, device)
    gate = restore_gate(gate_checkpoint, checkpoint, device) if gate_checkpoint else None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(inputs):
        x = load_rgb(path)[None].to(device)
        # Whole-image inference. No tiling: tiling changes global states/statistics.
        out = model(x)
        arrays = {k: out[k][0].cpu().numpy() for k in ("y0", "corrected", "gain", "bias")}
        y = arrays["corrected"]
        if gate is not None:
            shared, _, _ = project_affine(arrays["y0"], y, bounded=True)
            lam = float(gate(gate_features(out["y0"], out["gain"], out["bias"])).item())
            y = shared + lam*(y-shared)
            arrays.update(gated=y, strength=np.array(lam))
        name = f"{index:06d}_{Path(path).stem}"
        np.savez_compressed(output/(name+".npz"), **arrays)
        Image.fromarray(np.rint(np.clip(y, 0, 1).transpose(1, 2, 0)*255).astype("uint8")).save(output/(name+".png"))
        records.append({"input": str(path), "output": name+".png", "arrays": name+".npz"})
    write_json(output/"inference.json", {"backend": config["model"]["backend"], "checkpoint_sha256": sha256(checkpoint), "records": records})
