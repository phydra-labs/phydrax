#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Masked BOLD observations lowered to native nonlinear least squares."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...optim import Bounds, NonlinearLeastSquaresProblem
from ...series import SampledSeries, SeriesSupport
from ._workflow import RegionalSolution


class BOLDObservation(StrictModule, NonTrainableState):
    """Fixed observed fractional BOLD with per-time/per-region missingness.

    Active observations are never dropped because a model solve failed. Missing
    data are sanitized before arithmetic; invalid active predictions produce
    nonfinite residuals and native optimization failure evidence, not a spurious
    zero loss. ``standard_deviation`` uses the same fractional BOLD units.
    """

    series: SampledSeries
    active: Array
    target: Array
    standard_deviation: Array
    region_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        region_ids: Sequence[str],
        series: SampledSeries,
        /,
        *,
        standard_deviation: ArrayLike = 1.0,
        mask: ArrayLike | None = None,
    ):
        labels = tuple(region_ids)
        if (
            not labels
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("Observation region_ids must be nonempty and unique.")
        if not isinstance(series, SampledSeries) or series.alignment != "node":
            raise TypeError("BOLD observations require a node-aligned SampledSeries.")
        if (
            series.support.series_shape
            or not eqx.is_array(series.values)
            or series.values.shape != (series.support.capacity, len(labels))
        ):
            raise ValueError("BOLD observations require time-major values [time,region].")
        if (
            series.support.coordinate_name != "time_s"
            or series.support.coordinate_id != "neuroscience:seconds"
        ):
            raise ValueError(
                "BOLD time coordinates must explicitly declare neuroscience:seconds and time_s."
            )
        if jnp.iscomplexobj(series.values):
            raise ValueError("BOLD observations must be real fractional signals.")
        active = jnp.broadcast_to(series.support.node_valid[:, None], series.values.shape)
        if series.value_valid is not None:
            active = active & series.value_valid
        if mask is not None:
            explicit = jnp.asarray(mask, dtype=bool)
            if explicit.shape != series.values.shape:
                raise ValueError("BOLD mask must have shape [time,region].")
            active = active & explicit
        scale = jnp.asarray(standard_deviation)
        if jnp.iscomplexobj(scale):
            raise ValueError("BOLD standard_deviation must be real.")
        scale = jnp.broadcast_to(scale, series.values.shape)
        scale = eqx.error_if(
            scale,
            jnp.any(active & (~jnp.isfinite(scale) | (scale <= 0.0))),
            "Active BOLD standard deviations must be finite and positive.",
        )
        target = eqx.error_if(
            series.values,
            ~jnp.any(active),
            "BOLD observations require at least one active datum.",
        )
        self.series = jax.tree.map(jax.lax.stop_gradient, series)
        self.active = jax.lax.stop_gradient(active)
        self.target = jax.lax.stop_gradient(jnp.where(active, target, 0.0))
        self.standard_deviation = jax.lax.stop_gradient(jnp.where(active, scale, 1.0))
        self.region_ids = labels

    @classmethod
    def from_samples(
        cls,
        region_ids: Sequence[str],
        times: ArrayLike,
        values: ArrayLike,
        /,
        *,
        time_unit: Literal["s", "ms"] = "s",
        signal_unit: Literal["fraction", "percent"] = "fraction",
        valid: ArrayLike | None = None,
        standard_deviation: ArrayLike = 1.0,
    ) -> BOLDObservation:
        """Convert a declared time/signal unit once, including observation noise."""
        if time_unit not in ("s", "ms") or signal_unit not in ("fraction", "percent"):
            raise ValueError("Expected time_unit s/ms and signal_unit fraction/percent.")
        time_scale = 0.001 if time_unit == "ms" else 1.0
        signal_scale = 0.01 if signal_unit == "percent" else 1.0
        support = SeriesSupport(
            jnp.asarray(times, dtype=float) * time_scale,
            coordinate_name="time_s",
            coordinate_id="neuroscience:seconds",
        )
        series = SampledSeries(
            support,
            jnp.asarray(values) * signal_scale,
            value_valid=None if valid is None else jnp.asarray(valid, dtype=bool),
            series_id="observed-bold-fraction",
        )
        return cls(
            region_ids,
            series,
            standard_deviation=jnp.asarray(standard_deviation) * signal_scale,
        )

    def residual(self, prediction: RegionalSolution, /) -> Array:
        """Whitened fixed-shape residual for a prediction at the exact sample times."""
        if not isinstance(prediction, RegionalSolution) or prediction.bold is None:
            raise TypeError("BOLD residuals require a regional/BOLD prediction.")
        if prediction.region_ids != self.region_ids:
            raise ValueError("BOLD prediction region order differs from observations.")
        predicted = prediction.bold
        if predicted.values.shape != self.target.shape:
            raise ValueError("BOLD prediction sample shape differs from observations.")
        if predicted.support.coordinate_id != self.series.support.coordinate_id:
            raise ValueError("BOLD prediction coordinate units differ from observations.")
        active_rows = jnp.any(self.active, axis=-1)
        values = eqx.error_if(
            predicted.values,
            jnp.any(
                active_rows
                & predicted.support.node_valid
                & (predicted.support.coordinates != self.series.support.coordinates)
            ),
            "BOLD predictions must use the observed sample times; interpolation is explicit.",
        )
        valid = predicted.support.node_valid[:, None] & jnp.isfinite(values)
        if predicted.value_valid is not None:
            valid = valid & predicted.value_valid
        safe_prediction = jnp.where(self.active & valid, values, 0.0)
        residual = (safe_prediction - self.target) / self.standard_deviation
        return jnp.where(self.active, jnp.where(valid, residual, jnp.nan), 0.0).reshape(
            -1
        )

    def least_squares_problem(
        self,
        predict: Callable[[Any, Any], RegionalSolution],
        /,
        *,
        bounds: Bounds | None = None,
    ) -> NonlinearLeastSquaresProblem:
        """Adapt ``predict(parameters,args)`` without supplying another optimizer."""
        if not callable(predict):
            raise TypeError("predict must be callable.")
        return NonlinearLeastSquaresProblem(
            _BOLDResidual(self, predict),
            bounds=bounds,
            problem_id="regional-bold-fit",
        )


class _BOLDResidual(StrictModule):
    observation: BOLDObservation
    predict: Callable[[Any, Any], RegionalSolution]

    def __call__(self, parameters: Any, args: Any, /) -> Array:
        return self.observation.residual(self.predict(parameters, args))


__all__ = ["BOLDObservation"]
