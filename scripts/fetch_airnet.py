"""Fetch upstream separately, preserve its notices, and record the exact revision."""
import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--destination", default="third_party/AirNet")
parser.add_argument("--revision", default="bedea82f7f7a71f135bcc026fd02ea9f2feaa710", help="Use a recorded commit SHA for subsequent reproductions")
args = parser.parse_args()
destination = Path(args.destination).resolve()
if destination.exists():
    raise SystemExit(f"Destination exists; not modifying it: {destination}")
destination.parent.mkdir(parents=True, exist_ok=True)
url = "https://github.com/XLearning-SCU/2022-CVPR-AirNet.git"
subprocess.run(["git", "clone", url, str(destination)], check=True)
subprocess.run(["git", "-C", str(destination), "checkout", "--detach", args.revision], check=True)
commit = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
lock = destination.parent/"airnet_revision.json"
lock.write_text(json.dumps({"url": url, "commit": commit}, indent=2)+"\n")
print(f"Fetched {commit}; recorded in {lock}. Install matching CUDA-enabled MMCV ops before training.")
