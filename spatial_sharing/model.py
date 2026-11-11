"""Official AirNet integration and an explicitly non-paper smoke backbone."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .modules import DSPC, RGLDSE, state_loss


class SmokeBackbone(nn.Module):
    """Small CPU test fixture, NEVER an AirNet reproduction or benchmark model."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 64, 3, padding=1)
        self.embedding = nn.Linear(64, 256)
        self.reconstruction = nn.Sequential(nn.Conv2d(67, 32, 3, padding=1), nn.LeakyReLU(.1),
                                            nn.Conv2d(32, 3, 3, padding=1))

    def encode(self, query, key=None):
        features = self.encoder(query)
        z = self.embedding(features.mean((2, 3)))
        logits, labels = None, None
        if self.training:
            if key is None:
                raise ValueError("Training requires an independent key crop.")
            key_z = self.embedding(self.encoder(key).mean((2, 3)))
            logits = F.normalize(z, dim=1) @ F.normalize(key_z, dim=1).T / .07
            labels = torch.arange(len(z), device=z.device)
        return z, features, logits, labels

    def restore(self, x, guidance):
        return x + self.reconstruction(torch.cat([x, guidance], dim=1))


class OfficialAirNet(nn.Module):
    """Use upstream CBDE/DGRN without substituting deformable convolution.

    Requires the upstream CUDA extension and a single visible CUDA device.
    Upstream MoCo creates labels with .cuda(); do not silently run on CPU/MPS.
    """
    def __init__(self, source, batch_size=16):
        super().__init__()
        root = Path(source).expanduser().resolve()
        if not (root / "net" / "model.py").is_file():
            raise FileNotFoundError(f"AirNet source not found: {root}. See README setup.")
        if not torch.cuda.is_available():
            raise RuntimeError("Official AirNet needs CUDA and its compiled DCN extension; use smoke only for tests.")
        if "net" in sys.modules:
            origins = [Path(p).resolve() for p in getattr(sys.modules["net"], "__path__", [])]
            if root / "net" not in origins:
                raise RuntimeError("A different top-level 'net' package is already imported.")
        sys.path.insert(0, str(root))
        try:
            airnet = importlib.import_module("net.model").AirNet(SimpleNamespace(batch_size=batch_size))
        except (ImportError, OSError) as exc:
            raise RuntimeError("Build the official AirNet DCN extension before using this backend.") from exc
        self.encoder, self.restorer = airnet.E, airnet.R

    def encode(self, query, key=None):
        if query.device.type != "cuda" or query.device.index not in (None, 0):
            raise ValueError("Expose one GPU via CUDA_VISIBLE_DEVICES and use device=cuda:0.")
        if self.training:
            if key is None:
                raise ValueError("CBDE training requires a key view.")
            z, logits, labels, features = self.encoder(query, key)
            return z, features, logits, labels
        z, features = self.encoder(query, query)
        return z, features, None, None

    def restore(self, x, guidance):
        return self.restorer(x, guidance)


class RestorationModel(nn.Module):
    def __init__(self, backbone, variant="B2", state_supervision="both", mismatch=False,
                 global_enabled=True, local_enabled=True, scales=(1, 2, 4), **dspc_options):
        super().__init__()
        if variant not in {"B0", "B1", "B2", "B3", "B4", "B5", "shared", "airnet", "dspc_only"}:
            raise ValueError(f"Unknown variant: {variant}")
        self.backbone, self.variant = backbone, variant
        self.state_supervision, self.mismatch = state_supervision, mismatch
        self.state = None if variant in {"airnet", "dspc_only"} else RGLDSE(global_enabled, local_enabled, scales)
        self.correction = None if variant in {"B0", "B5", "airnet"} else DSPC(
            offset_only=variant == "B1", direct_guidance=variant in {"B3", "B4"},
            shared=variant == "shared", **dspc_options)

    def forward(self, x, key=None, encoder_only=False):
        z, features, logits, labels = self.backbone.encode(x, key)
        states = self.state(z, features) if self.state is not None else None
        guidance = states["guidance"] if states is not None else features
        result = {"states": states, "logits": logits, "labels": labels}
        if encoder_only:
            return result
        y0 = self.backbone.restore(x, guidance)
        if self.correction is None:
            result.update(y0=y0, corrected=y0, gain=torch.ones_like(y0), bias=torch.zeros_like(y0))
        else:
            result.update(self.correction(x, y0, guidance))
        return result

    def loss(self, out, x, target, encoder_only=False, state_weight=1):
        con = F.cross_entropy(out["logits"], out["labels"])
        state = state_loss(out["states"], x, target, self.state_supervision, self.mismatch) if self.state else con * 0
        terms = {"contrastive": con, "state": state}
        if encoder_only:
            terms["total"] = con + state_weight * state
            return terms
        terms["reconstruction"] = F.l1_loss(out["corrected"], target)
        total = terms["reconstruction"] + .1 * con + state_weight * state
        if self.variant in {"B4", "B5"}:
            y = out["corrected"] if self.variant == "B4" else out["y0"]
            coarse = torch.stack([F.l1_loss(F.avg_pool2d(y, s), F.avg_pool2d(target, s)) for s in (8, 16)]).mean()
            terms["coarse"] = coarse
            total = total + .2 * coarse
        terms["total"] = total
        return terms


def build_model(config):
    options = dict(config["model"])
    backend = options.pop("backend")
    source = options.pop("airnet_source", "third_party/AirNet")
    if backend == "airnet":
        backbone = OfficialAirNet(source, config["training"]["batch_size"])
    elif backend == "smoke":
        backbone = SmokeBackbone()
    else:
        raise ValueError(f"Unknown backbone backend: {backend}")
    return RestorationModel(backbone, **options)
