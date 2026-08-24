#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract


ClassificationKind: TypeAlias = Literal["binary", "multiclass", "multilabel", "ordinal"]
ClassificationObjectiveKind: TypeAlias = Literal["nll", "soft_cross_entropy", "focal"]


def _real_array(name: str, values: ArrayLike, /) -> Array:
    result = jnp.asarray(values)
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    return result.astype(jnp.result_type(result, 0.0))


def binary_log_prob_from_logits(
    logits: ArrayLike,
    target: ArrayLike,
    /,
) -> Array:
    """Return stable hard Bernoulli log probabilities from scalar logits."""
    values = _real_array("Binary logits", logits)
    raw_target = jnp.asarray(target)
    if jnp.issubdtype(raw_target.dtype, jnp.complexfloating):
        raise TypeError("Binary targets must be real-valued.")
    if raw_target.shape != values.shape:
        raise ValueError(
            "Binary targets must match logits; "
            f"got logits={values.shape} and target={raw_target.shape}."
        )
    observation = raw_target.astype(jnp.result_type(raw_target, 0.0))
    logits_valid = jnp.isfinite(values)
    target_valid = jnp.isfinite(observation) & (
        (observation == 0.0) | (observation == 1.0)
    )
    safe_logits = jnp.where(logits_valid, values, 0.0)
    safe_target = jnp.where(target_valid, observation, 0.0)
    result = jnp.where(
        safe_target == 1.0,
        jax.nn.log_sigmoid(safe_logits),
        jax.nn.log_sigmoid(-safe_logits),
    )
    return jnp.where(logits_valid & target_valid, result, -jnp.inf)


def binary_probabilities_from_logits(logits: ArrayLike, /) -> Array:
    """Return positive-class Bernoulli probabilities."""
    values = _real_array("Binary logits", logits)
    return jnp.where(jnp.isfinite(values), jax.nn.sigmoid(values), jnp.nan)


def categorical_log_prob_from_logits(
    logits: ArrayLike,
    target: ArrayLike,
    /,
    *,
    class_count: int | None = None,
) -> Array:
    """Return hard categorical log probabilities without one-hot targets."""
    values = _real_array("Categorical logits", logits)
    if values.ndim == 0:
        raise ValueError("Categorical logits must have a terminal class axis.")
    classes = int(values.shape[-1]) if class_count is None else int(class_count)
    if classes < 2 or int(values.shape[-1]) != classes:
        raise ValueError(
            f"Categorical logits must end in class_count={classes}; got {values.shape}."
        )
    raw_target = jnp.asarray(target)
    if jnp.issubdtype(raw_target.dtype, jnp.complexfloating):
        raise TypeError("Categorical targets must be real-valued labels.")
    if raw_target.shape != values.shape[:-1]:
        raise ValueError(
            "Categorical targets must match the logits batch shape; "
            f"got logits={values.shape} and target={raw_target.shape}."
        )
    observation = raw_target.astype(jnp.result_type(raw_target, 0.0))
    target_valid = (
        jnp.isfinite(observation)
        & (observation >= 0.0)
        & (observation < classes)
        & (observation == jnp.floor(observation))
    )
    logits_valid = jnp.all(jnp.isfinite(values), axis=-1)
    safe_target = jnp.where(target_valid, observation, 0.0).astype(jnp.int32)
    safe_logits = jnp.where(logits_valid[..., None], values, 0.0)
    selected = jnp.take_along_axis(safe_logits, safe_target[..., None], axis=-1)[..., 0]
    result = selected - jax.nn.logsumexp(safe_logits, axis=-1)
    return jnp.where(logits_valid & target_valid, result, -jnp.inf)


def categorical_probabilities_from_logits(
    logits: ArrayLike,
    /,
    *,
    class_count: int | None = None,
) -> Array:
    """Return probabilities on a terminal categorical class axis."""
    values = _real_array("Categorical logits", logits)
    if values.ndim == 0:
        raise ValueError("Categorical logits must have a terminal class axis.")
    classes = int(values.shape[-1]) if class_count is None else int(class_count)
    if classes < 2 or int(values.shape[-1]) != classes:
        raise ValueError(
            f"Categorical logits must end in class_count={classes}; got {values.shape}."
        )
    valid = jnp.all(jnp.isfinite(values), axis=-1)
    safe = jnp.where(valid[..., None], values, 0.0)
    probabilities = jax.nn.softmax(safe, axis=-1)
    return jnp.where(valid[..., None], probabilities, jnp.nan)


def independent_bernoulli_log_prob_from_logits(
    logits: ArrayLike,
    target: ArrayLike,
    /,
    *,
    target_mask: ArrayLike | None = None,
) -> Array:
    """Return per-label independent Bernoulli log probabilities."""
    values = _real_array("Multilabel logits", logits)
    raw_target = jnp.asarray(target)
    if jnp.issubdtype(raw_target.dtype, jnp.complexfloating):
        raise TypeError("Multilabel targets must be real-valued.")
    if raw_target.shape != values.shape or values.ndim == 0:
        raise ValueError(
            "Multilabel targets must match logits with a terminal label axis; "
            f"got logits={values.shape} and target={raw_target.shape}."
        )
    if target_mask is None:
        mask = jnp.ones(values.shape, dtype=bool)
    else:
        mask = jnp.asarray(target_mask)
        if mask.dtype != jnp.bool_ or mask.shape != values.shape:
            raise ValueError("target_mask must be Boolean and match multilabel targets.")
    safe_logits = jnp.where(mask, values, 0.0)
    safe_target = jnp.where(mask, raw_target, 0)
    log_prob = binary_log_prob_from_logits(safe_logits, safe_target)
    return jnp.where(mask, log_prob, 0.0)


def independent_bernoulli_probabilities_from_logits(
    logits: ArrayLike,
    /,
) -> Array:
    """Return independent positive-label probabilities without simplex normalization."""
    return binary_probabilities_from_logits(logits)


def soft_binary_cross_entropy_from_logits(
    logits: ArrayLike,
    target: ArrayLike,
    /,
) -> Array:
    """Return expected Bernoulli negative log scores for soft targets."""
    values = _real_array("Binary logits", logits)
    observation = _real_array("Soft binary targets", target)
    if observation.shape != values.shape:
        raise ValueError("Soft binary targets must match logits.")
    valid = (
        jnp.isfinite(values)
        & jnp.isfinite(observation)
        & (observation >= 0.0)
        & (observation <= 1.0)
    )
    safe_logits = jnp.where(valid, values, 0.0)
    safe_target = jnp.where(valid, observation, 0.0)
    result = (1.0 - safe_target) * jax.nn.softplus(
        safe_logits
    ) + safe_target * jax.nn.softplus(-safe_logits)
    return jnp.where(valid, result, jnp.inf)


def soft_categorical_cross_entropy_from_logits(
    logits: ArrayLike,
    target: ArrayLike,
    /,
) -> Array:
    """Return categorical cross entropy against full simplex targets."""
    values = _real_array("Categorical logits", logits)
    observation = _real_array("Soft categorical targets", target)
    if values.ndim == 0 or observation.shape != values.shape:
        raise ValueError("Soft categorical targets must match terminal-class logits.")
    classes = int(values.shape[-1])
    if classes < 2:
        raise ValueError("Soft categorical targets require at least two classes.")
    dtype = jnp.result_type(values, observation)
    tolerance = jnp.asarray(32 * classes, dtype=dtype) * jnp.finfo(dtype).eps
    target_sum = jnp.sum(observation, axis=-1)
    valid = (
        jnp.all(jnp.isfinite(values), axis=-1)
        & jnp.all(jnp.isfinite(observation) & (observation >= 0.0), axis=-1)
        & (jnp.abs(target_sum - 1.0) <= tolerance)
    )
    safe_logits = jnp.where(valid[..., None], values, 0.0)
    safe_target = jnp.where(valid[..., None], observation, 0.0)
    log_probability = jax.nn.log_softmax(safe_logits, axis=-1)
    result = -contract("...k,...k->...", safe_target, log_probability)
    return jnp.where(valid, result, jnp.inf)


def binary_focal_risk_from_logits(
    logits: ArrayLike,
    target: ArrayLike,
    /,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> Array:
    """Return hard binary focal risk without probability clipping."""
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value < 0.0:
        raise ValueError("gamma must be finite and nonnegative.")
    if alpha is not None:
        alpha_value = float(alpha)
        if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
            raise ValueError("Binary focal alpha must lie strictly inside (0, 1).")
    else:
        alpha_value = 1.0
    log_probability = binary_log_prob_from_logits(logits, target)
    cross_entropy = -log_probability
    if gamma_value == 0.0:
        factor = jnp.ones_like(cross_entropy)
    else:
        factor = (-jnp.expm1(log_probability)) ** gamma_value
    if alpha is None:
        class_weight = jnp.ones_like(cross_entropy)
    else:
        observation = jnp.asarray(target, dtype=cross_entropy.dtype)
        class_weight = alpha_value * observation + (1.0 - alpha_value) * (
            1.0 - observation
        )
    return class_weight * factor * cross_entropy


def categorical_focal_risk_from_logits(
    logits: ArrayLike,
    target: ArrayLike,
    /,
    *,
    gamma: float = 2.0,
    alpha: ArrayLike | None = None,
) -> Array:
    """Return hard categorical focal risk through gathered class logits."""
    values = _real_array("Categorical logits", logits)
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value < 0.0:
        raise ValueError("gamma must be finite and nonnegative.")
    log_probability = categorical_log_prob_from_logits(values, target)
    cross_entropy = -log_probability
    if gamma_value == 0.0:
        factor = jnp.ones_like(cross_entropy)
    else:
        factor = (-jnp.expm1(log_probability)) ** gamma_value
    if alpha is None:
        class_weight = jnp.ones_like(cross_entropy)
    else:
        alpha_array = np.asarray(alpha, dtype=float)
        if alpha_array.shape != (int(values.shape[-1]),):
            raise ValueError("Categorical focal alpha must have shape (class_count,).")
        if np.any(~np.isfinite(alpha_array)) or np.any(alpha_array <= 0.0):
            raise ValueError("Categorical focal alpha must be finite and positive.")
        weights = jnp.asarray(alpha_array)
        raw_target = jnp.asarray(target)
        target_valid = (
            jnp.isfinite(raw_target)
            & (raw_target >= 0)
            & (raw_target < values.shape[-1])
            & (raw_target == jnp.floor(raw_target))
        )
        safe_target = jnp.where(target_valid, raw_target, 0).astype(jnp.int32)
        class_weight = weights[safe_target]
    return class_weight * factor * cross_entropy


def ordinal_class_probabilities_from_location(
    location: ArrayLike,
    thresholds: ArrayLike,
    /,
) -> Array:
    """Return ordered-logistic class probabilities from scalar latent locations."""
    eta = _real_array("Ordinal location", location)
    cutpoints = _real_array("Ordinal thresholds", thresholds)
    if cutpoints.ndim != 1 or int(cutpoints.shape[0]) < 2:
        raise ValueError("Ordinal thresholds must be a vector with at least two entries.")
    cumulative = jax.nn.sigmoid(cutpoints - eta[..., None])
    middle = cumulative[..., 1:] - cumulative[..., :-1]
    return jnp.concatenate(
        (
            cumulative[..., :1],
            jnp.maximum(middle, 0.0),
            1.0 - cumulative[..., -1:],
        ),
        axis=-1,
    )


def ordinal_log_prob_from_location(
    location: ArrayLike,
    target: ArrayLike,
    thresholds: ArrayLike,
    /,
) -> Array:
    """Return stable ordered-logistic hard-label log probabilities."""
    eta = _real_array("Ordinal location", location)
    cutpoints = _real_array("Ordinal thresholds", thresholds)
    if cutpoints.ndim != 1 or int(cutpoints.shape[0]) < 2:
        raise ValueError("Ordinal thresholds must be a vector with at least two entries.")
    raw_target = jnp.asarray(target)
    if raw_target.shape != eta.shape:
        raise ValueError("Ordinal targets must match scalar location shape.")
    levels = int(cutpoints.shape[0]) + 1
    observation = raw_target.astype(jnp.result_type(raw_target, 0.0))
    target_valid = (
        jnp.isfinite(observation)
        & (observation >= 0.0)
        & (observation < levels)
        & (observation == jnp.floor(observation))
    )
    location_valid = jnp.isfinite(eta)
    safe_eta = jnp.where(location_valid, eta, 0.0)
    arguments = cutpoints - safe_eta[..., None]
    first = jax.nn.log_sigmoid(arguments[..., :1])
    lower = arguments[..., :-1]
    upper = arguments[..., 1:]
    gap = upper - lower
    middle = (
        upper
        + jnp.log(-jnp.expm1(-gap))
        - jax.nn.softplus(upper)
        - jax.nn.softplus(lower)
    )
    last = jax.nn.log_sigmoid(-arguments[..., -1:])
    log_masses = jnp.concatenate((first, middle, last), axis=-1)
    safe_target = jnp.where(target_valid, observation, 0.0).astype(jnp.int32)
    selected = jnp.take_along_axis(log_masses, safe_target[..., None], axis=-1)[..., 0]
    return jnp.where(location_valid & target_valid, selected, -jnp.inf)


def classification_probabilities(
    logits: ArrayLike,
    /,
    *,
    kind: ClassificationKind,
    class_count: int | None = None,
    thresholds: ArrayLike | None = None,
) -> Array:
    """Convert declared classification coordinates to explicit probabilities."""
    if kind == "binary":
        return binary_probabilities_from_logits(logits)
    if kind == "multiclass":
        return categorical_probabilities_from_logits(logits, class_count=class_count)
    if kind == "multilabel":
        return independent_bernoulli_probabilities_from_logits(logits)
    if thresholds is None:
        raise ValueError("Ordinal probabilities require thresholds.")
    return ordinal_class_probabilities_from_location(logits, thresholds)


def pointwise_classification_loss(
    logits: ArrayLike,
    target: ArrayLike,
    /,
    *,
    kind: ClassificationKind,
    objective: ClassificationObjectiveKind = "nll",
    class_count: int | None = None,
    target_mask: ArrayLike | None = None,
    gamma: float = 2.0,
    alpha: ArrayLike | float | None = None,
    thresholds: ArrayLike | None = None,
) -> Array:
    """Return one unreduced classification score per observation prefix."""
    if target_mask is not None and kind != "multilabel":
        mask = jnp.asarray(target_mask)
        values = jnp.asarray(logits)
        observations = jnp.asarray(target)
        prefix_shape = values.shape[:-1] if kind == "multiclass" else values.shape
        if mask.dtype != jnp.bool_ or mask.shape != prefix_shape:
            raise ValueError(
                "target_mask must be Boolean and match the observation prefix."
            )
        safe_logits = jnp.where(
            mask[..., None] if kind == "multiclass" else mask,
            values,
            0.0,
        )
        target_has_class_axis = kind == "multiclass" and objective == "soft_cross_entropy"
        safe_target = jnp.where(
            mask[..., None] if target_has_class_axis else mask,
            observations,
            0,
        )
        active_loss = pointwise_classification_loss(
            safe_logits,
            safe_target,
            kind=kind,
            objective=objective,
            class_count=class_count,
            gamma=gamma,
            alpha=alpha,
            thresholds=thresholds,
        )
        return jnp.where(mask, active_loss, 0.0)
    if kind == "ordinal":
        if objective != "nll" or thresholds is None:
            raise ValueError(
                "Ordinal classification currently requires NLL and thresholds."
            )
        return -ordinal_log_prob_from_location(logits, target, thresholds)
    if kind == "binary":
        if objective == "nll":
            return -binary_log_prob_from_logits(logits, target)
        if objective == "soft_cross_entropy":
            return soft_binary_cross_entropy_from_logits(logits, target)
        return binary_focal_risk_from_logits(
            logits,
            target,
            gamma=gamma,
            alpha=None if alpha is None else float(alpha),
        )
    if kind == "multiclass":
        if objective == "nll":
            return -categorical_log_prob_from_logits(
                logits, target, class_count=class_count
            )
        if objective == "soft_cross_entropy":
            return soft_categorical_cross_entropy_from_logits(logits, target)
        return categorical_focal_risk_from_logits(
            logits,
            target,
            gamma=gamma,
            alpha=alpha,
        )
    values = _real_array("Multilabel logits", logits)
    if objective == "nll":
        per_label = -independent_bernoulli_log_prob_from_logits(
            values,
            target,
            target_mask=target_mask,
        )
    else:
        if target_mask is None:
            mask = jnp.ones(values.shape, dtype=bool)
        else:
            mask = jnp.asarray(target_mask)
            if mask.dtype != jnp.bool_ or mask.shape != values.shape:
                raise ValueError(
                    "target_mask must be Boolean and match multilabel logits."
                )
        safe_logits = jnp.where(mask, values, 0.0)
        safe_target = jnp.where(mask, jnp.asarray(target), 0)
        if objective == "soft_cross_entropy":
            per_label = soft_binary_cross_entropy_from_logits(safe_logits, safe_target)
        else:
            per_label = binary_focal_risk_from_logits(
                safe_logits,
                safe_target,
                gamma=gamma,
                alpha=None if alpha is None else float(alpha),
            )
        per_label = jnp.where(mask, per_label, 0.0)
    return jnp.sum(per_label, axis=-1)


__all__ = [
    "ClassificationKind",
    "ClassificationObjectiveKind",
    "binary_focal_risk_from_logits",
    "binary_log_prob_from_logits",
    "binary_probabilities_from_logits",
    "categorical_focal_risk_from_logits",
    "categorical_log_prob_from_logits",
    "categorical_probabilities_from_logits",
    "classification_probabilities",
    "independent_bernoulli_log_prob_from_logits",
    "independent_bernoulli_probabilities_from_logits",
    "ordinal_class_probabilities_from_location",
    "ordinal_log_prob_from_location",
    "pointwise_classification_loss",
    "soft_binary_cross_entropy_from_logits",
    "soft_categorical_cross_entropy_from_logits",
]
