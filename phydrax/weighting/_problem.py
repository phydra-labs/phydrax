#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    RankPolicy,
)
from ..linalg.eigen import SelfAdjointSpectrumPolicy


class ExactMoments(StrictModule):
    """A vector of moments requested exactly on a regular interior branch."""

    values: Array

    def __init__(self, values: ArrayLike, /):
        values_ = _moment_values(values)
        self.values = values_


class QuadraticMoments(StrictModule):
    """Moment targets reconciled through a diagonal quadratic discrepancy."""

    values: Array
    scale: Array

    def __init__(self, values: ArrayLike, /, *, scale: ArrayLike = 1.0):
        values_ = _moment_values(values)
        scale_ = jnp.asarray(scale, dtype=values_.dtype)
        if jnp.broadcast_shapes(scale_.shape, values_.shape) != values_.shape:
            raise ValueError(
                "scale must be scalar or broadcast to the target moment shape."
            )
        self.values = values_
        self.scale = jnp.broadcast_to(scale_, values_.shape)
        invalid = jnp.any(~jnp.isfinite(self.scale) | (self.scale <= 0.0))
        if isinstance(invalid, jax_core.Tracer):
            self.scale = eqx.error_if(
                self.scale,
                invalid,
                "Quadratic moment scales must be finite and strictly positive.",
            )
        elif bool(invalid):
            raise ValueError(
                "Quadratic moment scales must be finite and strictly positive."
            )


MomentTarget: TypeAlias = ExactMoments | QuadraticMoments


class MomentCalibrationPolicy(StrictModule):
    """Affine-rank, regularity, and bounded geometry policy for calibration."""

    rank: RankPolicy
    spectrum: SelfAdjointSpectrumPolicy
    affine_absolute_tolerance: float = eqx.field(static=True)
    affine_relative_tolerance: float = eqx.field(static=True)
    regularity_relative_tolerance: float = eqx.field(static=True)
    maximum_moments: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        rank: RankPolicy | None = None,
        spectrum: SelfAdjointSpectrumPolicy | None = None,
        affine_absolute_tolerance: float = 1e-10,
        affine_relative_tolerance: float = 1e-8,
        regularity_relative_tolerance: float = 1e-10,
        maximum_moments: int = 512,
    ):
        rank_ = RankPolicy() if rank is None else rank
        spectrum_ = SelfAdjointSpectrumPolicy() if spectrum is None else spectrum
        if not isinstance(rank_, RankPolicy):
            raise TypeError("rank must be a RankPolicy or None.")
        if not isinstance(spectrum_, SelfAdjointSpectrumPolicy):
            raise TypeError("spectrum must be a SelfAdjointSpectrumPolicy or None.")
        tolerances = (
            float(affine_absolute_tolerance),
            float(affine_relative_tolerance),
            float(regularity_relative_tolerance),
        )
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Calibration tolerances must be finite and non-negative.")
        maximum = int(maximum_moments)
        if maximum < 1:
            raise ValueError("maximum_moments must be positive.")
        self.rank = rank_
        self.spectrum = spectrum_
        (
            self.affine_absolute_tolerance,
            self.affine_relative_tolerance,
            self.regularity_relative_tolerance,
        ) = tolerances
        self.maximum_moments = maximum


class MomentCalibrationProblem(StrictModule):
    """A finite prior, linear moment map, and exact or soft target moments."""

    moment_map: AbstractLinearOperator
    target: MomentTarget
    prior_log_weights: Array
    mask: Array
    source_points: int = eqx.field(static=True)
    moment_count: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        features: ArrayLike | AbstractLinearOperator,
        target: MomentTarget,
        /,
        *,
        prior_log_weights: ArrayLike | None = None,
        mask: ArrayLike | None = None,
        problem_id: str | None = None,
    ):
        moment_map = _moment_operator(features)
        if not isinstance(target, (ExactMoments, QuadraticMoments)):
            raise TypeError("target must be ExactMoments or QuadraticMoments.")
        source_points = int(moment_map.source.shape[0])
        moment_count = int(moment_map.target.shape[0])
        if target.values.shape != (moment_count,):
            raise ValueError(
                f"Target moments must have shape ({moment_count},); "
                f"got {target.values.shape}."
            )
        dtype = moment_map.source.dtype
        target_ = (
            ExactMoments(target.values.astype(dtype))
            if isinstance(target, ExactMoments)
            else QuadraticMoments(
                target.values.astype(dtype),
                scale=target.scale.astype(dtype),
            )
        )
        if prior_log_weights is None:
            prior = jnp.zeros((source_points,), dtype=dtype)
        else:
            prior = jnp.asarray(prior_log_weights, dtype=dtype)
            if prior.shape != (source_points,):
                raise ValueError(f"prior_log_weights must have shape ({source_points},).")
        if mask is None:
            mask_ = jnp.ones((source_points,), dtype=bool)
        else:
            mask_ = jnp.asarray(mask, dtype=bool)
            if mask_.shape != (source_points,):
                raise ValueError(f"mask must have shape ({source_points},).")
        self.moment_map = moment_map
        self.target = target_
        self.prior_log_weights = prior
        self.mask = mask_
        self.source_points = source_points
        self.moment_count = moment_count
        self.problem_id = (
            canonical_fingerprint(
                {
                    "kind": "moment-calibration",
                    "operator": moment_map.operator_id,
                    "source_points": source_points,
                    "moment_count": moment_count,
                    "target": type(target_).__name__,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not self.problem_id:
            raise ValueError("problem_id must be non-empty.")


def _moment_values(values: ArrayLike, /) -> Array:
    values_ = jnp.asarray(values)
    if values_.ndim != 1 or values_.shape[0] == 0:
        raise ValueError("Moment values must be a non-empty one-dimensional array.")
    if jnp.issubdtype(values_.dtype, jnp.complexfloating):
        raise TypeError("Moment values must be real.")
    if not jnp.issubdtype(values_.dtype, jnp.inexact):
        values_ = values_.astype(float)
    invalid = ~jnp.all(jnp.isfinite(values_))
    if isinstance(invalid, jax_core.Tracer):
        values_ = eqx.error_if(
            values_,
            invalid,
            "Moment target values must be finite.",
        )
    elif bool(invalid):
        raise ValueError("Moment target values must be finite.")
    return values_


def _moment_operator(
    features: ArrayLike | AbstractLinearOperator,
    /,
) -> AbstractLinearOperator:
    if isinstance(features, AbstractLinearOperator):
        operator = features
    else:
        values = jnp.asarray(features)
        if values.ndim != 2 or any(size == 0 for size in values.shape):
            raise ValueError(
                "Dense features must have shape (source_points, moment_count)."
            )
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Moment features must be real.")
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        invalid = ~jnp.all(jnp.isfinite(values))
        if isinstance(invalid, jax_core.Tracer):
            values = eqx.error_if(
                values,
                invalid,
                "Dense moment features must be finite.",
            )
        elif bool(invalid):
            raise ValueError("Dense moment features must be finite.")
        operator = DenseLinearOperator(jnp.swapaxes(values, 0, 1))
    if not isinstance(operator.source, ArraySpace) or not isinstance(
        operator.target, ArraySpace
    ):
        raise TypeError("Moment maps must act between ArraySpace values.")
    if len(operator.source.shape) != 1 or len(operator.target.shape) != 1:
        raise ValueError("Moment maps must act between one-dimensional arrays.")
    if operator.batch_shape:
        raise ValueError("Batched moment maps are not supported.")
    if not operator.capabilities.transpose:
        raise ValueError("Moment maps require a transpose action.")
    if operator.source.dtype != operator.target.dtype:
        raise TypeError("Moment-map source and target dtypes must match.")
    if not np.issubdtype(operator.source.dtype, np.floating):
        raise TypeError("Moment maps must have a real floating-point dtype.")
    return operator


__all__ = [
    "ExactMoments",
    "MomentCalibrationPolicy",
    "MomentCalibrationProblem",
    "MomentTarget",
    "QuadraticMoments",
]
