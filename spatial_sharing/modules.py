"""Network components specified in the main article and Supplement S1/S7."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class RGLDSE(nn.Module):
    def __init__(self, global_enabled=True, local_enabled=True, scales=(1, 2, 4)):
        super().__init__()
        self.global_enabled, self.local_enabled = global_enabled, local_enabled
        self.scales = tuple(scales)
        if not self.scales or any(s < 1 for s in self.scales):
            raise ValueError("Local scales must be positive.")
        self.global_state = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(.1), nn.LayerNorm(256))
        self.channel_gate = nn.Linear(256, 64)
        self.severity = nn.Linear(256, 1)
        self.local_states = nn.ModuleList([
            nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.LeakyReLU(.1),
                          nn.Conv2d(32, 1, 1), nn.Sigmoid()) for _ in self.scales
        ])
        self.local_fusion = nn.Sequential(nn.Conv2d(len(scales), 64, 3, padding=1, bias=False),
                                         nn.LeakyReLU(.1), nn.Conv2d(64, 64, 3, padding=1, bias=False))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, z, features):
        h, w = features.shape[-2:]
        if z.shape != (features.shape[0], 256) or features.shape[1] != 64:
            raise ValueError("RGL-DSE requires z=[B,256], F=[B,64,H,W].")
        s = self.global_state(z)
        gates = self.channel_gate(s).sigmoid()[:, :, None, None]
        maps = [head(F.adaptive_avg_pool2d(features, (max(1, h // scale), max(1, w // scale))))
                for scale, head in zip(self.scales, self.local_states)]
        local = self.local_fusion(torch.cat([
            F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False) for m in maps
        ], dim=1))
        guidance = features
        if self.global_enabled:
            guidance = guidance + self.alpha.tanh() * features * gates
        if self.local_enabled:
            guidance = guidance + self.beta.tanh() * local
        return {"guidance": guidance, "severity": self.severity(s).sigmoid(),
                "states": maps, "channel_gate": gates}


def state_loss(states, x, target, mode="both", mismatch=False):
    """GT enters only the loss, never the inference network."""
    residual = (x - target).abs().mean(dim=1, keepdim=True).detach()
    if mismatch:
        if len(residual) < 2:
            raise ValueError("Mismatched supervision needs batch_size >= 2.")
        residual = residual.roll(1, dims=0)  # deterministic nonidentity pairing
    global_loss = F.l1_loss(states["severity"], residual.mean((2, 3)))
    local_loss = torch.stack([F.l1_loss(m, F.adaptive_avg_pool2d(residual, m.shape[-2:]))
                              for m in states["states"]]).mean()
    if mode == "none":
        return global_loss * 0
    if mode == "global":
        return global_loss
    if mode == "local":
        return local_loss
    if mode != "both":
        raise ValueError(f"Unknown state supervision: {mode}")
    return .5 * (global_loss + local_loss)


class DSPC(nn.Module):
    def __init__(self, downsample=16, log_gain_bound=.4, bias_bound=.1,
                 offset_only=False, direct_guidance=False, shared=False):
        super().__init__()
        if downsample < 1 or log_gain_bound < 0 or bias_bound < 0:
            raise ValueError("Invalid DSPC bounds or downsampling factor.")
        self.downsample, self.log_gain_bound, self.bias_bound = downsample, log_gain_bound, bias_bound
        self.offset_only, self.direct_guidance, self.shared = offset_only, direct_guidance, shared
        self.trunk = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1), nn.LeakyReLU(.1),
                                   nn.Conv2d(16, 16, 3, padding=1), nn.LeakyReLU(.1))
        self.gain_head, self.bias_head = nn.Conv2d(16, 3, 1), nn.Conv2d(16, 3, 1)
        # The supplement explicitly registers this inactive path in B2 (1,024 params).
        self.state_projection = nn.Conv2d(64, 16, 1, bias=False)
        for head in (self.gain_head, self.bias_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x, y0, guidance=None):
        h, w = x.shape[-2:]
        size = (math.ceil(h / self.downsample), math.ceil(w / self.downsample))
        v = self.trunk(F.adaptive_avg_pool2d(torch.cat([x, y0], dim=1), size))
        if self.direct_guidance:
            if guidance is None:
                raise ValueError("Direct correction needs guidance.")
            v = v + self.state_projection(F.adaptive_avg_pool2d(guidance, size))
        log_gain = self.log_gain_bound * self.gain_head(v).tanh()
        if self.offset_only:
            log_gain = log_gain * 0
        bias = self.bias_bound * self.bias_head(v).tanh()
        gain = F.interpolate(log_gain, (h, w), mode="bilinear", align_corners=False).exp()
        bias = F.interpolate(bias, (h, w), mode="bilinear", align_corners=False)
        if self.shared:
            # Average gains AFTER exponentiation, not coarse log-gains.
            gain = gain.mean((2, 3), keepdim=True).expand_as(y0)
            bias = bias.mean((2, 3), keepdim=True).expand_as(y0)
        return {"y0": y0, "gain": gain, "bias": bias, "corrected": gain * y0 + bias}


def gate_features(y0, gain, bias):
    """Order: Y0 mean/std, C-Y0 mean/RMS, g mean/std, b mean/std (24)."""
    delta = gain * y0 + bias - y0
    def stats(v):
        return [v.mean((2, 3)), v.std((2, 3), correction=0)]
    return torch.cat(stats(y0) + [delta.mean((2, 3)), delta.square().mean((2, 3)).sqrt()]
                     + stats(gain) + stats(bias), dim=1)


class StrengthPredictor(nn.Module):
    def __init__(self, dropout=.1):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(24))
        self.register_buffer("feature_std", torch.ones(24))
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(24, 32), nn.SiLU(),
                                 nn.Linear(32, 8), nn.SiLU(), nn.Linear(8, 1), nn.Sigmoid())
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    @torch.no_grad()
    def fit_normalizer(self, train_features):
        if train_features.ndim != 2 or train_features.shape[1] != 24 or len(train_features) == 0:
            raise ValueError("Expected nonempty training-only [N,24] features.")
        self.feature_mean.copy_(train_features.mean(0))
        self.feature_std.copy_(train_features.std(0, correction=0).clamp_min(1e-6))

    def forward(self, features):
        return self.net((features - self.feature_mean) / self.feature_std).squeeze(-1)
