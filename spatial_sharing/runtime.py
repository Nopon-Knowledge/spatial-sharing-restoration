import hashlib
import json
import math
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_config(path):
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # perfect PSNR is explicitly "inf", not invalid JSON
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def append_log(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(value), ensure_ascii=False, allow_nan=False)+"\n")


def atomic_save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def provenance(config):
    import PIL
    import scipy
    import skimage
    result = {"python": platform.python_version(), "platform": platform.platform(), "torch": str(torch.__version__),
              "numpy": np.__version__, "pillow": PIL.__version__, "scipy": scipy.__version__,
              "scikit_image": skimage.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "backend": config["model"]["backend"], "config": config}
    result["source_sha256"] = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    result["manifest_sha256"] = {k: sha256(v) for k, v in config["data"].items() if k.endswith("manifest") and v and Path(v).is_file()}
    if config["model"]["backend"] == "airnet":
        root = config["model"]["airnet_source"]
        source_record = Path(root)/"SOURCE.json"
        if source_record.is_file():
            result["airnet_source_record"] = json.loads(source_record.read_text())
            result["airnet_commit"] = result["airnet_source_record"]["commit"]
        else:
            try:
                result["airnet_commit"] = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
            except (OSError, subprocess.CalledProcessError):
                result["airnet_commit"] = None
        result["airnet_source_sha256"] = {str(p.relative_to(root)): sha256(p) for p in sorted(Path(root).glob("net/*.py"))}
    return result
