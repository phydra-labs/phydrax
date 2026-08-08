#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import PointSampling, SampleLayout

from .._sampling import design_name


def normalize_case_sampling(
    sampling: PointSampling,
    /,
    *,
    labels: tuple[str, ...],
    owner: str,
) -> PointSampling:
    """Validate and canonicalize a uniform empirical-case sampling plan."""
    if not isinstance(sampling, PointSampling):
        raise TypeError(f"{owner} requires a PointSampling plan.")
    if not isinstance(sampling.count, int):
        raise TypeError(f"{owner} requires one integer case count.")
    if sampling.count <= 0:
        raise ValueError(f"{owner} sampling count must be positive.")
    if design_name(sampling.design) != "uniform":
        raise ValueError(f"{owner} supports only uniform sampling.")
    layout = sampling.layout or SampleLayout((labels,))
    return PointSampling(sampling.count, layout=layout, design=sampling.design)

def case_sample_count(sampling: PointSampling, /) -> int:
    """Return the scalar count guaranteed by normalized empirical sampling."""
    count = sampling.count
    if not isinstance(count, int):
        raise RuntimeError("Normalized empirical sampling must have one integer count.")
    return count


def supervised_data_metrics(
    prediction: Array,
    target: Array,
    /,
    *,
    eps: ArrayLike,
) -> dict[str, Array]:
    pred_arr, target_arr = _align_data_metric_shapes(prediction, target)
    residual = pred_arr - target_arr

    mse = jnp.mean(residual * residual)
    rmse = jnp.sqrt(mse)
    residual_norm = jnp.linalg.norm(jnp.ravel(residual))
    target_norm = jnp.linalg.norm(jnp.ravel(target_arr))
    relative_l2 = residual_norm / (target_norm + eps)
    accuracy = 1.0 - relative_l2

    return {
        "data_accuracy": jnp.asarray(accuracy, dtype=float).reshape(()),
        "data_relative_l2_error": jnp.asarray(relative_l2, dtype=float).reshape(()),
        "data_rmse": jnp.asarray(rmse, dtype=float).reshape(()),
    }


def _align_data_metric_shapes(pred: Array, target: Array, /) -> tuple[Array, Array]:
    pred_arr = jnp.asarray(pred, dtype=float)
    target_arr = jnp.asarray(target, dtype=float)

    if pred_arr.shape == target_arr.shape:
        return pred_arr, target_arr
    if pred_arr.ndim == 2 and target_arr.ndim == 1 and int(pred_arr.shape[1]) == 1:
        pred_arr = pred_arr[:, 0]
        if pred_arr.shape == target_arr.shape:
            return pred_arr, target_arr
    if target_arr.ndim == 2 and pred_arr.ndim == 1 and int(target_arr.shape[1]) == 1:
        target_arr = target_arr[:, 0]
        if pred_arr.shape == target_arr.shape:
            return pred_arr, target_arr

    raise ValueError(
        "Data metric prediction and target shapes are incompatible: "
        f"prediction={pred_arr.shape}, target={target_arr.shape}."
    )


def supervised_per_sample_squared_error(
    prediction: Array,
    target: Array,
    /,
) -> Array:
    """Return one squared-error scalar per leading sample."""
    pred_arr, target_arr = _align_data_metric_shapes(prediction, target)
    residual = pred_arr - target_arr
    squared = residual * residual
    if squared.ndim <= 1:
        return squared.reshape((-1,))
    return jnp.sum(squared.reshape((int(squared.shape[0]), -1)), axis=1)


def reduce_supervised_loss(
    per_sample: Array,
    /,
    *,
    reduction: str,
) -> Array:
    """Reduce per-sample supervised losses with a common mean/sum policy."""
    per_sample_arr = jnp.asarray(per_sample, dtype=float)
    if reduction == "mean":
        reduced = jnp.mean(per_sample_arr)
    elif reduction == "sum":
        reduced = jnp.sum(per_sample_arr)
    else:
        raise ValueError("reduction must be either 'mean' or 'sum'.")
    return jnp.asarray(reduced, dtype=float).reshape(())


def validate_supervised_targets(
    values: ArrayLike,
    /,
    *,
    leading_size: int,
    name: str,
) -> Array:
    """Validate a target array with a leading empirical-case axis."""
    arr = jnp.asarray(values, dtype=float)
    if arr.ndim == 0:
        raise ValueError(f"{name} values must have shape (N, ...).")
    if int(arr.shape[0]) != int(leading_size):
        raise ValueError(
            f"{name} leading axis must be N={int(leading_size)}, got {arr.shape[0]}."
        )
    return arr


def validate_case_indices(
    indices: ArrayLike | None,
    /,
    *,
    size: int,
    name: str = "indices",
) -> Array | None:
    """Validate an optional non-empty 1D integer index subset."""
    if indices is None:
        return None
    raw = jnp.asarray(indices)
    if raw.ndim != 1:
        raise ValueError(f"{name} must have shape (K,), got {raw.shape}.")
    if int(raw.shape[0]) <= 0:
        raise ValueError(f"{name} must be non-empty.")
    idx = raw.astype(jnp.int32)
    if bool(jnp.any(idx != raw)):
        raise ValueError(f"{name} must contain integer indices.")
    if bool(jnp.any(idx < 0)) or bool(jnp.any(idx >= int(size))):
        raise ValueError(f"{name} must be within [0, {int(size)}).")
    return idx


def sample_case_indices(
    *,
    size: int,
    num_samples: int,
    key: Key[Array, ""],
    indices: Array | None = None,
) -> Array:
    """Sample empirical-case indices uniformly from all cases or a subset."""
    n = int(num_samples)
    if n < 0:
        raise ValueError("num_samples must be non-negative.")
    if n == 0:
        return jnp.zeros((0,), dtype=jnp.int32)
    if indices is None:
        return _random_int(key, n=n, maxval=int(size))
    idx = jnp.asarray(indices, dtype=jnp.int32).reshape((-1,))
    positions = _random_int(key, n=n, maxval=int(idx.shape[0]))
    return idx[positions]


def _random_int(key: Key[Array, ""], /, *, n: int, maxval: int) -> Array:
    return jr.randint(
        key,
        shape=(int(n),),
        minval=0,
        maxval=int(maxval),
        dtype=jnp.int32,
    )
