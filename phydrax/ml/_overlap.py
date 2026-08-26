#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState


OverlapKind: TypeAlias = Literal["dice", "jaccard", "tversky"]
OverlapClassReduction: TypeAlias = Literal["micro", "macro", "support_weighted"]
OverlapEmptyPolicy: TypeAlias = Literal["zero", "one", "nan", "ignore"]


class OverlapScoreConfig(StrictModule, NonTrainableState):
    """Immutable, JSON-safe overlap score and class-reduction policy."""

    kind: OverlapKind = eqx.field(static=True)
    class_reduction: OverlapClassReduction = eqx.field(static=True)
    empty: OverlapEmptyPolicy = eqx.field(static=True)
    smooth: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    def __init__(
        self,
        kind: OverlapKind = "dice",
        /,
        *,
        class_reduction: OverlapClassReduction = "micro",
        empty: OverlapEmptyPolicy = "zero",
        smooth: float = 0.0,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        if kind not in ("dice", "jaccard", "tversky"):
            raise ValueError("kind must be 'dice', 'jaccard', or 'tversky'.")
        if class_reduction not in ("micro", "macro", "support_weighted"):
            raise ValueError(
                "class_reduction must be 'micro', 'macro', or 'support_weighted'."
            )
        if empty not in ("zero", "one", "nan", "ignore"):
            raise ValueError("empty must be 'zero', 'one', 'nan', or 'ignore'.")
        smooth_value = float(smooth)
        alpha_value = float(alpha)
        beta_value = float(beta)
        if not math.isfinite(smooth_value) or smooth_value < 0.0:
            raise ValueError("smooth must be finite and nonnegative.")
        if not math.isfinite(alpha_value) or alpha_value < 0.0:
            raise ValueError("alpha must be finite and nonnegative.")
        if not math.isfinite(beta_value) or beta_value < 0.0:
            raise ValueError("beta must be finite and nonnegative.")
        if kind == "tversky" and alpha_value == 0.0 and beta_value == 0.0:
            raise ValueError("Tversky alpha and beta cannot both be zero.")
        self.kind = kind
        self.class_reduction = class_reduction
        self.empty = empty
        self.smooth = smooth_value
        self.alpha = alpha_value
        self.beta = beta_value

    def to_dict(self) -> dict[str, str | float]:
        """Return a JSON-serializable representation with no array leaves."""
        return {
            "kind": self.kind,
            "class_reduction": self.class_reduction,
            "empty": self.empty,
            "smooth": self.smooth,
            "alpha": self.alpha,
            "beta": self.beta,
        }


def _statistics(
    intersection: ArrayLike,
    prediction_support: ArrayLike,
    target_support: ArrayLike,
    /,
) -> tuple[Array, Array, Array]:
    intersection_ = jnp.asarray(intersection)
    prediction_ = jnp.asarray(prediction_support)
    target_ = jnp.asarray(target_support)
    dtype = jnp.result_type(intersection_, prediction_, target_, 0.0)
    intersection_ = intersection_.astype(dtype)
    prediction_ = prediction_.astype(dtype)
    target_ = target_.astype(dtype)
    if prediction_.shape != intersection_.shape or target_.shape != intersection_.shape:
        raise ValueError("Overlap sufficient statistics must have identical shapes.")
    scale = jnp.maximum(
        1.0,
        jnp.maximum(jnp.abs(prediction_), jnp.abs(target_)),
    )
    tolerance = 32.0 * jnp.finfo(dtype).eps * scale
    valid = (
        jnp.isfinite(intersection_)
        & jnp.isfinite(prediction_)
        & jnp.isfinite(target_)
        & (intersection_ >= -tolerance)
        & (prediction_ >= -tolerance)
        & (target_ >= -tolerance)
        & (intersection_ <= prediction_ + tolerance)
        & (intersection_ <= target_ + tolerance)
    )
    return tuple(
        jnp.where(valid, jnp.maximum(value, 0.0), jnp.nan)
        for value in (intersection_, prediction_, target_)
    )


def _empty_value(reference: Array, policy: OverlapEmptyPolicy, /) -> Array:
    if policy == "one":
        return jnp.ones_like(reference)
    if policy == "zero":
        return jnp.zeros_like(reference)
    return jnp.full_like(reference, jnp.nan)


def _ratio(
    numerator: Array,
    denominator: Array,
    /,
    *,
    smooth: float,
    empty: OverlapEmptyPolicy,
) -> Array:
    smooth_value = float(smooth)
    if not math.isfinite(smooth_value) or smooth_value < 0.0:
        raise ValueError("smooth must be finite and nonnegative.")
    if empty not in ("zero", "one", "nan", "ignore"):
        raise ValueError("empty must be 'zero', 'one', 'nan', or 'ignore'.")
    empty_support = denominator == 0.0
    safe_denominator = jnp.where(empty_support, 1.0, denominator)
    value = (numerator + smooth_value) / (safe_denominator + smooth_value)
    return jnp.where(empty_support, _empty_value(value, empty), value)


def dice_score(
    intersection: ArrayLike,
    prediction_support: ArrayLike,
    target_support: ArrayLike,
    /,
    *,
    smooth: float = 0.0,
    empty: OverlapEmptyPolicy = "nan",
) -> Array:
    """Compute Dice scores from pre-aggregated sufficient statistics."""
    intersection_, prediction_, target_ = _statistics(
        intersection, prediction_support, target_support
    )
    return _ratio(
        2.0 * intersection_,
        prediction_ + target_,
        smooth=float(smooth),
        empty=empty,
    )


def jaccard_score(
    intersection: ArrayLike,
    prediction_support: ArrayLike,
    target_support: ArrayLike,
    /,
    *,
    smooth: float = 0.0,
    empty: OverlapEmptyPolicy = "nan",
) -> Array:
    """Compute Jaccard scores from pre-aggregated sufficient statistics."""
    intersection_, prediction_, target_ = _statistics(
        intersection, prediction_support, target_support
    )
    return _ratio(
        intersection_,
        prediction_ + target_ - intersection_,
        smooth=float(smooth),
        empty=empty,
    )


def tversky_score(
    intersection: ArrayLike,
    prediction_support: ArrayLike,
    target_support: ArrayLike,
    /,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 0.0,
    empty: OverlapEmptyPolicy = "nan",
) -> Array:
    """Compute Tversky scores from pre-aggregated sufficient statistics."""
    alpha_value = float(alpha)
    beta_value = float(beta)
    if (
        not math.isfinite(alpha_value)
        or not math.isfinite(beta_value)
        or alpha_value < 0.0
        or beta_value < 0.0
        or (alpha_value == 0.0 and beta_value == 0.0)
    ):
        raise ValueError("Tversky alpha/beta must be nonnegative and not both zero.")
    intersection_, prediction_, target_ = _statistics(
        intersection, prediction_support, target_support
    )
    false_positive = prediction_ - intersection_
    false_negative = target_ - intersection_
    return _ratio(
        intersection_,
        intersection_ + alpha_value * false_positive + beta_value * false_negative,
        smooth=float(smooth),
        empty=empty,
    )


def overlap_score(
    intersection: ArrayLike,
    prediction_support: ArrayLike,
    target_support: ArrayLike,
    config: OverlapScoreConfig,
    /,
) -> Array:
    """Compute one configured score without applying a class reduction."""
    if not isinstance(config, OverlapScoreConfig):
        raise TypeError("config must be an OverlapScoreConfig.")
    kwargs = {"smooth": config.smooth, "empty": config.empty}
    if config.kind == "dice":
        return dice_score(intersection, prediction_support, target_support, **kwargs)
    if config.kind == "jaccard":
        return jaccard_score(intersection, prediction_support, target_support, **kwargs)
    return tversky_score(
        intersection,
        prediction_support,
        target_support,
        alpha=config.alpha,
        beta=config.beta,
        **kwargs,
    )


def reduce_overlap_score(
    intersection: ArrayLike,
    prediction_support: ArrayLike,
    target_support: ArrayLike,
    config: OverlapScoreConfig,
    /,
) -> Array:
    """Reduce a terminal class axis by micro, macro, or support weighting."""
    intersection_, prediction_, target_ = _statistics(
        intersection, prediction_support, target_support
    )
    if intersection_.ndim == 0:
        raise ValueError(
            "Class-reduced overlap statistics require a terminal class axis."
        )
    if config.class_reduction == "micro":
        return overlap_score(
            jnp.sum(intersection_, axis=-1),
            jnp.sum(prediction_, axis=-1),
            jnp.sum(target_, axis=-1),
            config,
        )

    per_class = overlap_score(intersection_, prediction_, target_, config)
    both_empty = (prediction_ == 0.0) & (target_ == 0.0)
    finite_class = (
        ~both_empty if config.empty == "ignore" else jnp.ones_like(both_empty, dtype=bool)
    )
    if config.class_reduction == "macro":
        class_weight = finite_class.astype(per_class.dtype)
    else:
        class_weight = jnp.where(finite_class, target_, 0.0)
    safe_score = jnp.where(class_weight > 0.0, per_class, 0.0)
    mass = jnp.sum(class_weight, axis=-1)
    reduced = jnp.sum(class_weight * safe_score, axis=-1) / jnp.where(
        mass > 0.0, mass, 1.0
    )
    fallback = _empty_value(reduced, config.empty)
    return jnp.where(mass > 0.0, reduced, fallback)


__all__ = [
    "dice_score",
    "jaccard_score",
    "OverlapClassReduction",
    "OverlapEmptyPolicy",
    "OverlapKind",
    "overlap_score",
    "OverlapScoreConfig",
    "reduce_overlap_score",
    "tversky_score",
]
