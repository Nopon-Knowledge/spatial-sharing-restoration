"""Evaluate shared-output references from saved, unclipped RGB model outputs.

Required NPZ arrays: y0 and corrected, each float CHW with three channels.
Optional: target; gain and bias (both CHW). Never fit a reference to target.
No restoration data are distributed with this analysis utility.
"""
import argparse
import json
from pathlib import Path

import numpy as np

GAIN_BOUNDS = (float(np.exp(-0.4)), float(np.exp(0.4)))
BIAS_BOUNDS = (-0.1, 0.1)


def rgb(array):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] != 3 or 0 in array.shape:
        raise ValueError('Expected a nonempty CHW RGB array.')
    if not np.isfinite(array).all():
        raise ValueError('Input contains nonfinite values.')
    return array


def pair(first, second):
    first, second = rgb(first), rgb(second)
    if first.shape != second.shape:
        raise ValueError('Image shapes must match.')
    return first, second


def project_affine(y0, corrected, bounded=False):
    """Fit corrected in output space; return fitted image and 3x1x1 a,d.

    For the bounded fit, enumerate the unconstrained minimizer and minima
    on the four box edges. This solves the two-variable convex problem.
    Constant channels may admit multiple coefficients, but a unique output.
    """
    y0, corrected = pair(y0, corrected)
    gains, offsets = [], []
    for x, c in zip(y0.reshape(3, -1), corrected.reshape(3, -1)):
        xm, cm = float(x.mean()), float(c.mean())
        centered = x - xm
        variance = 0.0 if np.all(x == x[0]) else float(centered @ centered)
        a = float(centered @ (c - cm)) / variance if variance > 0 else 0.0
        d = cm - a * xm
        if bounded:
            alo, ahi = GAIN_BOUNDS
            dlo, dhi = BIAS_BOUNDS
            candidates = []
            if alo <= a <= ahi and dlo <= d <= dhi:
                candidates.append((a, d))
            for edge_a in GAIN_BOUNDS:
                candidates.append((edge_a, float(np.clip(cm-edge_a*xm, dlo, dhi))))
            norm_x = float(x @ x)
            for edge_d in BIAS_BOUNDS:
                edge_a = float(x @ (c-edge_d)) / norm_x if norm_x > 0 else 1.0
                candidates.append((float(np.clip(edge_a, alo, ahi)), edge_d))
            a, d = min(candidates, key=lambda z: float(np.sum((c-z[0]*x-z[1])**2)))
        gains.append(a)
        offsets.append(d)
    gains = np.asarray(gains).reshape(3, 1, 1)
    offsets = np.asarray(offsets).reshape(3, 1, 1)
    return gains*y0+offsets, gains, offsets


def oracle_strength(shared, corrected, target):
    """Paired-reference diagnostic for unclipped RGB MSE, not a predictor."""
    shared, corrected = pair(shared, corrected)
    _, target = pair(shared, target)
    residual = corrected-shared
    energy = float(np.sum(residual**2))
    if energy == 0.0:
        return 0.0
    return float(np.clip(np.sum((target-shared)*residual)/energy, 0, 1))


def metrics(output, target):
    # RGB MSE is the metric in the derivation. This is not the manuscript's
    # clipped luminance PSNR/SSIM protocol; it must not replace those metrics.
    return {'rgb_mse_unclipped': float(np.mean((output-target)**2)),
            'rgb_mse_clipped': float(np.mean((np.clip(output, 0, 1)-target)**2))}


def analyze(path, array_directory=None):
    with np.load(path, allow_pickle=False) as data:
        y0, corrected = pair(data['y0'], data['corrected'])
        shared, a, d = project_affine(y0, corrected)
        bounded, ab, db = project_affine(y0, corrected, bounded=True)
        residual = corrected-shared
        references = {'preliminary': y0, 'original': corrected,
                      'affine_projection': shared, 'bounded_projection': bounded}
        if ('gain' in data) != ('bias' in data):
            raise ValueError('Provide both gain and bias, or neither.')
        if 'gain' in data:
            gain, bias = pair(data['gain'], data['bias'])
            pair(y0, gain)
            if not np.allclose(corrected, gain*y0+bias, atol=2e-6, rtol=2e-6):
                raise ValueError('corrected must equal the UNCLIPPED gain*y0+bias.')
            references['coefficient_mean'] = gain.mean((1, 2), keepdims=True)*y0 + bias.mean((1, 2), keepdims=True)
        record = {'file': str(path), 'pixels_per_channel': int(y0.shape[1]*y0.shape[2]),
                  'fit_target': 'corrected model output; never the clean target',
                  'residual_energy': float(np.sum(residual**2)),
                  'affine_coefficients': {'gain': a.ravel().tolist(), 'bias': d.ravel().tolist()},
                  'bounded_coefficients': {'gain': ab.ravel().tolist(), 'bias': db.ravel().tolist()},
                  'fit_squared_error': {name: float(np.sum((corrected-value)**2))
                                        for name, value in references.items()}}
        if 'target' in data:
            _, target = pair(y0, data['target'])
            projected_target, _, _ = project_affine(y0, target)
            shared_error = float(np.sum((shared-projected_target)**2))
            complementary_error = float(np.sum((residual-(target-projected_target))**2))
            total_error = float(np.sum((corrected-target)**2))
            record['error_decomposition'] = {
                'shared_squared_error': shared_error,
                'complementary_squared_error': complementary_error,
                'total_squared_error': total_error,
                'identity_absolute_error': abs(shared_error+complementary_error-total_error)}
            record['metrics'] = {name: metrics(value, target) for name, value in references.items()}
            record['strength_diagnostics'] = {}
            for name in ('affine_projection', 'bounded_projection', 'coefficient_mean'):
                if name not in references:
                    continue
                base = references[name]
                delta = corrected-base
                remaining = target-base
                dot = float(np.sum(remaining*delta))
                denominator = float(np.linalg.norm(remaining)*np.linalg.norm(delta))
                strength = oracle_strength(base, corrected, target)
                record['strength_diagnostics'][name] = {
                    'residual_energy': float(np.sum(delta**2)),
                    'alignment_inner_product': dot,
                    'alignment_cosine': dot/denominator if denominator > 0 else None,
                    'oracle_lambda': strength,
                    'oracle_metrics': metrics(base+strength*delta, target),
                    'fixed_grid': [{'lambda': float(lam), **metrics(base+lam*delta, target)}
                                   for lam in np.linspace(0, 1, 5)]}
            # Output shifts preserve residual energy. They do not preserve
            # orthogonality to the image-dependent shared affine subspace.
            shifts = {(max(1, y0.shape[1]//4), 0), (0, max(1, y0.shape[2]//4)),
                      (max(1, y0.shape[1]//2), max(1, y0.shape[2]//2))}
            record['shift_controls'] = [
                {'shift_yx': list(shift),
                 'residual_energy': float(np.sum(np.roll(residual, shift, axis=(1, 2))**2)),
                 **metrics(shared+np.roll(residual, shift, axis=(1, 2)), target)}
                for shift in sorted(shifts)]
        if array_directory is not None:
            array_directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(array_directory/(path.stem+'_decomposition.npz'),
                                shared=shared, bounded=bounded, residual=residual,
                                affine_gain=a, affine_bias=d, bounded_gain=ab, bounded_bias=db)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+', help='NPZ files, one image per file')
    parser.add_argument('--output', required=True, type=Path, help='Per-image JSON diagnostics')
    parser.add_argument('--arrays', type=Path, help='Optional output folder for fitted arrays')
    args = parser.parse_args()
    if args.arrays and len({p.stem for p in args.inputs}) != len(args.inputs):
        parser.error('Input stems must be unique when saving arrays.')
    records = [analyze(path, args.arrays) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'metric_scope': 'Unclipped RGB MSE analysis, not luminance PSNR/SSIM.',
                                      'images': records}, indent=2, allow_nan=False)+'\n')
    print(f'Analyzed {len(records)} images; wrote {args.output}')


if __name__ == '__main__':
    main()
