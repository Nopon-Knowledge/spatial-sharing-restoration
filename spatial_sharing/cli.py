import argparse
import json
from pathlib import Path

import numpy as np

from .runtime import read_config, write_json, sha256


def main():
    parser = argparse.ArgumentParser(description="Paper reference code: spatial sharing and adverse weather restoration")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="Train encoder, then joint restoration")
    p.add_argument("--config", default="configs/b2.yaml")
    p.add_argument("--run", required=True)
    p.add_argument("--resume")
    p.add_argument("--variant", choices=["B0", "B1", "B2", "B3", "B4", "B5", "shared", "airnet", "dspc_only"])
    p.add_argument("--seed", type=int)
    p = sub.add_parser("evaluate", help="Evaluate paired data and export raw arrays")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--gate")
    p = sub.add_parser("infer", help="Restore images without clean references")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gate")
    p = sub.add_parser("analyze", help="Fit saved UNCLIPPED outputs and test alignment")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--no-curve", action="store_true")
    p.add_argument("--knots", help="JSON list of training-defined curve knots; default 17 uniform nodes in [0,1]")
    p = sub.add_parser("cache-gate", help="Generate frozen B2 gate records")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--per-task", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("train-gate", help="Fit strength from train cache; select with validation Y-PSNR")
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--config", default="configs/gate.yaml")
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("bootstrap", help="Paired scene-level task-stratified uncertainty")
    p.add_argument("--records", required=True)
    p.add_argument("--first", required=True)
    p.add_argument("--second", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--draws", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("smoke", help="Synthetic CPU integration check; NOT a restoration experiment")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        from .engine import train
        config = read_config(args.config)
        if args.variant:
            config["model"]["variant"] = args.variant
        if args.seed is not None:
            config["training"]["seed"] = args.seed
        print(train(config, args.run, args.resume))
    elif args.command == "evaluate":
        from .data import read_manifest
        from .engine import restore_checkpoint, restore_gate, evaluate_model
        model, config = restore_checkpoint(args.checkpoint, args.device)
        gate = restore_gate(args.gate, args.checkpoint, args.device) if args.gate else None
        report = evaluate_model(model, read_manifest(args.manifest, args.root), args.device, full=True,
                                output_dir=args.output, perceptual=args.lpips, gate=gate)
        report.update(backend=config["model"]["backend"], checkpoint_sha256=sha256(args.checkpoint))
        write_json(Path(args.output)/"metrics.json", report)
        print(report["macro"])
    elif args.command == "infer":
        from .engine import infer
        infer(args.checkpoint, args.inputs, args.output, args.device, args.gate)
    elif args.command == "analyze":
        from .analysis import analyze_arrays
        records = []
        knots = json.loads(Path(args.knots).read_text()) if args.knots else None
        for path in args.inputs:
            with np.load(path, allow_pickle=False) as a:
                record, _ = analyze_arrays(a["y0"], a["corrected"], a.get("gain"), a.get("bias"),
                                           a.get("target"), include_curve=not args.no_curve, knots=knots)
            records.append({"file": path, **record})
        write_json(args.output, {"fit_target": "Unclipped model output, never ground truth", "images": records})
    elif args.command == "cache-gate":
        from .gate_training import cache_outputs
        meta = cache_outputs(args.checkpoint, args.manifest, args.root, args.output, args.split,
                             args.device, args.per_task, args.seed)
        print(f"Cached {len(meta['records'])} {args.split} records.")
    elif args.command == "train-gate":
        from .gate_training import train_gate
        print(train_gate(args.train_cache, args.val_cache, args.run, read_config(args.config), args.device))
    elif args.command == "bootstrap":
        from .analysis import paired_bootstrap
        records = json.loads(Path(args.records).read_text())
        write_json(args.output, paired_bootstrap(records, args.first, args.second, args.draws, args.seed))
    elif args.command == "smoke":
        from .smoke import run_smoke
        run_smoke(args.output)


if __name__ == "__main__":
    main()
