#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
from jaxtyping import Array


def supervised_data_metrics(
    prediction: Array,
    target: Array,
    /,
    *,
    eps: Array,
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
