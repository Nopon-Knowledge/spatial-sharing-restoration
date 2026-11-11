"""Run on the target CUDA machine to validate the real upstream integration."""
import argparse

import torch

from spatial_sharing.model import build_model
from spatial_sharing.runtime import read_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/b2.yaml")
args = parser.parse_args()
config = read_config(args.config)
if config["model"]["backend"] != "airnet":
    raise SystemExit("This check requires the official AirNet backend.")
# Keep batch size unchanged because upstream queue K must divide it.
model = build_model(config).to("cuda:0").train()
size = config["training"]["crop"]
x = torch.rand(config["training"]["batch_size"], 3, size, size, device="cuda:0")
for encoder_only in (True, False):
    model.zero_grad(set_to_none=True)
    out = model(x, x.flip(-1), encoder_only=encoder_only)
    loss = model.loss(out, x, x*.9, encoder_only=encoder_only)["total"]
    loss.backward()
    assert torch.isfinite(loss)
    print(f"Official AirNet {'encoder' if encoder_only else 'joint'} forward/backward passed.")
model.eval()
with torch.no_grad():
    assert model(x[:1])["corrected"].shape == x[:1].shape
print("Official AirNet inference passed. This is an interface test, not benchmark training.")
