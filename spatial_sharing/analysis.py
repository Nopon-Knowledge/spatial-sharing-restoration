"""Output fitting and alignment controls: never fit against clean targets."""
import numpy as np
from scipy.optimize import lsq_linear

from .projection import project_affine, oracle_strength, pair
from .metrics import image_metrics


def color_matrix_fit(y0, corrected):
    y0, corrected = pair(y0, corrected)
    design = np.column_stack([y0.reshape(3, -1).T, np.ones(y0.shape[1]*y0.shape[2])])
    matrix = np.linalg.lstsq(design, corrected.reshape(3, -1).T, rcond=None)[0]
    return (design @ matrix).T.reshape(y0.shape), matrix


def curve_fit(y0, corrected, knots=None):
    """Monotone shared 17-node piecewise-linear fit, linear endpoint extrapolation.

    Fixed [0,1] nodes are an explicit implementation convention. Pass train-defined
    nodes to compare a different convention; never choose knots using clean targets.
    """
    y0, corrected = pair(y0, corrected)
    knots = np.linspace(0, 1, 17) if knots is None else np.asarray(knots, dtype=float)
    if knots.ndim != 1 or len(knots) < 2 or np.any(np.diff(knots) <= 0):
        raise ValueError("Curve knots must strictly increase.")
    result, coefficients = [], []
    # Parameterize node values as free first value + nonnegative increments.
    cumulative = np.tril(np.ones((len(knots), len(knots))))
    for x, c in zip(y0.reshape(3, -1), corrected.reshape(3, -1)):
        index = np.clip(np.searchsorted(knots, x)-1, 0, len(knots)-2)
        t = (x-knots[index]) / (knots[index+1]-knots[index])
        basis = np.zeros((len(x), len(knots)))
        basis[np.arange(len(x)), index] = 1-t
        basis[np.arange(len(x)), index+1] = t
        fit = lsq_linear(basis @ cumulative, c, bounds=(np.r_[-np.inf, np.zeros(len(knots)-1)], np.inf),
                         method="bvls", tol=1e-10, max_iter=2000)
        if not fit.success:
            raise RuntimeError(f"Monotone curve fit failed: {fit.message}")
        nodes = cumulative @ fit.x
        result.append(basis @ nodes)
        coefficients.append(nodes)
    return np.asarray(result).reshape(y0.shape), np.asarray(coefficients)


def projected_shift(y0, residual, shift, eps=1e-12):
    shifted = np.roll(residual, shift, axis=(1, 2))
    projection, _, _ = project_affine(y0, shifted)
    orthogonal = shifted-projection
    norm = np.linalg.norm(orthogonal)
    if norm < eps:
        return None  # explicitly mark a degenerate control, never divide by zero
    return orthogonal * (np.linalg.norm(residual)/norm)


def analyze_arrays(y0, corrected, gain=None, bias=None, target=None, include_curve=True, knots=None):
    y0, corrected = pair(y0, corrected)
    shared, a, d = project_affine(y0, corrected)
    bounded, ab, db = project_affine(y0, corrected, bounded=True)
    color, matrix = color_matrix_fit(y0, corrected)
    residual = corrected-shared
    references = {"y0": y0, "spatial": corrected, "affine_ls": shared, "box_ls": bounded, "color_matrix": color}
    if (gain is None) != (bias is None):
        raise ValueError("Provide both gain and bias.")
    if gain is not None:
        gain, bias = pair(gain, bias)
        pair(y0, gain)
        if not np.allclose(corrected, gain*y0+bias, atol=2e-6, rtol=2e-6):
            raise ValueError("Output must be the UNCLIPPED gain*y0+bias.")
        references["mean_fields"] = gain.mean((1, 2), keepdims=True)*y0+bias.mean((1, 2), keepdims=True)
    if include_curve:
        references["shared_curve"], _ = curve_fit(y0, corrected, knots)
    h, w = y0.shape[-2:]
    shifts = {(max(1, h//4) % h, 0), (0, max(1, w//4) % w), (max(1, h//2) % h, max(1, w//2) % w)} - {(0, 0)}
    for dy, dx in sorted(shifts):
        references[f"shift_{dy}_{dx}"] = shared + np.roll(residual, (dy, dx), axis=(1, 2))
        shifted = projected_shift(y0, residual, (dy, dx))
        if shifted is not None:
            references[f"projected_shift_{dy}_{dx}"] = shared + shifted
    record = {"affine_gain": a.ravel().tolist(), "affine_bias": d.ravel().tolist(),
              "bounded_gain": ab.ravel().tolist(), "bounded_bias": db.ravel().tolist(),
              "fit_mse": {name: float(np.mean((value-corrected)**2)) for name, value in references.items()},
              "energy": float(np.mean(residual**2)),
              "orthogonality_constant": residual.mean((1, 2)).tolist(),
              "orthogonality_y0": (residual*(y0-y0.mean((1, 2), keepdims=True))).mean((1, 2)).tolist()}
    if target is not None:
        _, target = pair(y0, target)
        py, _, _ = project_affine(y0, target)  # ONLY the diagnostic identity uses target
        lhs = float(np.sum((corrected-target)**2))
        rhs = float(np.sum((shared-py)**2)+np.sum((residual-(target-py))**2))
        record.update(error_identity_lhs=lhs, error_identity_rhs=rhs, identity_relative_error=abs(lhs-rhs)/max(lhs, 1e-30))
        record["utility"] = {}
        for name, base in (("affine", shared), ("bounded", bounded)):
            delta = corrected-base
            lam = oracle_strength(base, corrected, target)
            record["utility"][name] = {"oracle_lambda": lam,
                                      "alignment": float(np.mean((target-base)*delta)),
                                      "energy": float(np.mean(delta**2))}
            references[name+"_oracle"] = base+lam*delta
        record["metrics"] = {name: image_metrics(value, target) for name, value in references.items()}
    return record, {"shared": shared, "bounded": bounded, "residual": residual}


def paired_bootstrap(records, first, second, draws=10000, seed=42):
    """Mean within scene first, then paired scene resampling within task."""
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[row["task"]][row["scene"]].append(row["scores"][first]-row["scores"][second])
    if not grouped:
        raise ValueError("No paired observations.")
    differences = [np.array([np.mean(v) for v in scenes.values()]) for scenes in grouped.values()]
    if any(not np.isfinite(d).all() for d in differences):
        raise ValueError("Paired bootstrap requires finite scores.")
    rng = np.random.default_rng(seed)
    estimates = np.zeros(draws)
    for d in differences:
        # Chunking avoids a draws x large-dataset allocation.
        for start in range(0, draws, 128):
            count = min(128, draws-start)
            estimates[start:start+count] += rng.choice(d, (count, len(d)), replace=True).mean(1)/len(differences)
    return {"mean_difference": float(np.mean([d.mean() for d in differences])),
            "ci95": np.quantile(estimates, [.025, .975]).tolist(), "draws": draws, "seed": seed,
            "scenes_per_task": {t: len(v) for t, v in grouped.items()}}
