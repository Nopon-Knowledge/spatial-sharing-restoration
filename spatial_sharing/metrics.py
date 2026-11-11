"""Paper luminance protocol, unclipped diagnostics, and optional LPIPS."""
from collections import defaultdict

import numpy as np
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity


def luminance(rgb):
    return (16 + 65.481*rgb[0] + 128.553*rgb[1] + 24.966*rgb[2]) / 255


def psnr(a, b):
    mse = float(np.mean((a-b)**2))
    return float("inf") if mse == 0 else float(-10*np.log10(mse))


def image_metrics(output, target, full=True):
    output, target = np.asarray(output, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if output.shape != target.shape or output.ndim != 3 or output.shape[0] != 3:
        raise ValueError("Metrics require equally shaped RGB CHW arrays.")
    if not np.isfinite(output).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite images cannot be scored.")
    if np.any((target < 0) | (target > 1)):
        raise ValueError("Reference pixels must be in [0,1].")
    clipped = np.clip(output, 0, 1)
    yo, yt = luminance(clipped), luminance(target)
    scores = {"y_psnr": psnr(yo, yt), "rgb_psnr": psnr(clipped, target),
              "rgb_mse_raw": float(np.mean((output-target)**2)),
              "rgb_mse_clip": float(np.mean((clipped-target)**2)),
              "out_of_range_percent": float(np.mean((output < 0) | (output > 1))*100)}
    if full:
        options = dict(data_range=1., gaussian_weights=True, sigma=1.5, win_size=11, use_sample_covariance=False)
        scores["y_ssim"] = float(structural_similarity(yo, yt, **options))
        scores["rgb_ssim"] = float(np.mean([structural_similarity(a, b, **options) for a, b in zip(clipped, target)]))
        lab_a, lab_b = rgb2lab(clipped.transpose(1, 2, 0)), rgb2lab(target.transpose(1, 2, 0))
        scores["delta_e76"] = float(np.linalg.norm(lab_a-lab_b, axis=-1).mean())
    return scores


def aggregate(records):
    if not records:
        raise ValueError("No metric records.")
    groups = defaultdict(list)
    for record in records:
        groups[record["task"]].append(record["metrics"])
    tasks = {task: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} for task, rows in groups.items()}
    macro = {key: float(np.mean([row[key] for row in tasks.values()])) for key in next(iter(tasks.values()))}
    return {"tasks": tasks, "macro": macro, "images_per_task": {t: len(r) for t, r in groups.items()}}


class LPIPSMetric:
    def __init__(self, device="cpu"):
        import lpips
        self.device = device
        self.net = lpips.LPIPS(net="alex").to(device).eval()

    def __call__(self, output, target):
        import torch
        def tensor(v):
            return torch.as_tensor(np.clip(v, 0, 1), dtype=torch.float32, device=self.device)[None]*2-1
        with torch.no_grad():
            return float(self.net(tensor(output), tensor(target)).item())
