#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..series import SampledSeries, SampledSeriesReconstruction, SeriesSupport
from ._derivative import CallbackDerivativeRule
from ._evaluation import BatchEvaluator
from ._function import DomainFunction
from ._irregular_trajectory_dataset import IrregularTrajectoryDatasetDomain
from ._structure import GridBatch, PointBatch
from ._trajectory_dataset import TRAJECTORY_CASE_INDEX_KEY, TrajectoryDatasetDomain


TrajectorySignalInterpolation = Literal["nearest", "linear", "cubic_hermite"]


def _validate_values(
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
    values: ArrayLike,
    /,
) -> Array:
    array = jnp.asarray(values, dtype=float)
    if array.ndim < 2:
        raise ValueError("values must have shape (N, T, ...) with a time axis.")
    if int(array.shape[0]) != domain.size:
        raise ValueError(
            f"values leading axis must be N={domain.size}, got {array.shape[0]}."
        )
    if int(array.shape[1]) < domain.max_length:
        raise ValueError(
            "values time axis must have at least "
            f"{domain.max_length} entries, got {array.shape[1]}."
        )
    return array


def _trajectory_series(
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
    values: ArrayLike,
    /,
) -> SampledSeries:
    values_ = _validate_values(domain, values)
    if isinstance(domain, IrregularTrajectoryDatasetDomain):
        capacity = int(domain.times.shape[1])
        coordinates = domain.times
        values_ = values_[:, :capacity]
    else:
        capacity = int(values_.shape[1])
        coordinates = domain.start + domain.dt * jnp.arange(capacity, dtype=float)
    valid = jnp.arange(capacity)[None, :] < domain.lengths[:, None]
    support = SeriesSupport(
        coordinates,
        node_valid=valid,
        series_shape=(domain.size,),
        series_axes=("trajectory_case",),
        coordinate_name=domain.time_label,
        coordinate_id=f"trajectory-domain:{domain.time_label}",
    )
    return SampledSeries(
        support,
        values_,
        series_id=f"trajectory-signal:{domain.time_label}",
    )


class _SeriesTrajectorySignal(StrictModule, BatchEvaluator, NonTrainableState):
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain
    reconstruction: SampledSeriesReconstruction
    derivative_order: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
        reconstruction: SampledSeriesReconstruction,
        derivative_order: int = 0,
    ):
        self.domain = domain
        self.reconstruction = reconstruction
        self.derivative_order = int(derivative_order)

    def __call__(self, *args: Any, key=None, **kwargs: Any) -> Array:
        del args, key, kwargs
        raise TypeError("TrajectorySignal requires PointBatch evaluation.")

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        del key, kwargs
        if not isinstance(batch, PointBatch):
            raise TypeError("TrajectorySignal requires PointBatch evaluation.")
        if TRAJECTORY_CASE_INDEX_KEY not in batch:
            raise ValueError(
                "TrajectorySignal requires trajectory batches with internal case indices."
            )
        case_field = batch[TRAJECTORY_CASE_INDEX_KEY]
        time_field = batch[self.domain.time_label]
        if not isinstance(case_field, cx.Field):
            raise TypeError("Trajectory case indices must be stored as a Field.")
        if not isinstance(time_field, cx.Field):
            raise TypeError("Trajectory time values must be stored as a Field.")
        values = jax.tree_util.tree_map(
            jax.lax.stop_gradient, self.reconstruction.series.values
        )
        reconstruction = eqx.tree_at(
            lambda candidate: candidate.series.values,
            self.reconstruction,
            values,
        )
        evaluation = reconstruction.evaluate(
            jnp.asarray(time_field.data, dtype=float),
            jnp.asarray(case_field.data, dtype=jnp.int32),
            derivative_order=self.derivative_order,
        )
        out = jnp.asarray(evaluation.values)
        dims = time_field.dims + (None,) * max(out.ndim - len(time_field.dims), 0)
        return cx.Field(out, dims=dims)


def _signal_function(
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
    reconstruction: SampledSeriesReconstruction,
    /,
    *,
    derivative_order: int,
) -> DomainFunction:
    return DomainFunction(
        domain=domain,
        deps=domain.labels,
        func=_SeriesTrajectorySignal(
            domain=domain,
            reconstruction=reconstruction,
            derivative_order=derivative_order,
        ),
        metadata={},
    )


def TrajectorySignal(
    domain: TrajectoryDatasetDomain | IrregularTrajectoryDatasetDomain,
    values: ArrayLike,
    /,
    *,
    interpolation: TrajectorySignalInterpolation = "linear",
    time_var: str | None = None,
    snap_tol: float = 1e-10,
) -> DomainFunction:
    """Expose fixed trajectory data as a `DomainFunction` over `(data, t)`.

    Use this when an observed ragged time series is an input or forcing term for
    another residual, rather than the supervised output being fitted directly.
    `values` must have one leading row per trajectory case and a padded time axis
    matching the domain lengths. Interpolated signals are non-trainable solver
    state and support time derivatives according to the interpolation order.
    """
    if not isinstance(
        domain, (TrajectoryDatasetDomain, IrregularTrajectoryDatasetDomain)
    ):
        raise TypeError("TrajectorySignal requires a trajectory dataset domain.")
    var = domain.time_label if time_var is None else str(time_var)
    if var != domain.time_label:
        raise ValueError(
            f"time_var must match the trajectory time label {domain.time_label!r}."
        )
    interpolation_ = str(interpolation)
    if interpolation_ not in ("nearest", "linear", "cubic_hermite"):
        raise ValueError("interpolation must be 'nearest', 'linear', or 'cubic_hermite'.")
    if (
        isinstance(domain, IrregularTrajectoryDatasetDomain)
        and interpolation_ == "cubic_hermite"
    ):
        raise ValueError(
            "IrregularTrajectoryDatasetDomain TrajectorySignal supports only "
            "interpolation='nearest' or interpolation='linear'."
        )

    series = _trajectory_series(domain, values)
    tie_policy = (
        "lower" if isinstance(domain, IrregularTrajectoryDatasetDomain) else "round_even"
    )
    reconstruction = SampledSeriesReconstruction(
        series,
        interpolation=interpolation_,
        bounds="clip",
        nearest_tie_policy=tie_policy,
        snap_tolerance=float(snap_tol),
    )
    base = _signal_function(domain, reconstruction, derivative_order=0)

    if interpolation_ == "nearest":

        def _nearest_hook(
            *,
            var: str,
            axis: int | None,
            order: int,
            mode,
            backend,
            basis,
            periodic: bool,
        ) -> DomainFunction | None:
            del mode, backend, basis, periodic
            if var != domain.time_label or axis is not None:
                return None
            if int(order) == 0:
                return base.with_derivative_rule(CallbackDerivativeRule(_nearest_hook))
            raise ValueError(
                "TrajectorySignal with interpolation='nearest' is not differentiable; "
                "use interpolation='linear' or 'cubic_hermite' for time derivatives."
            )

        return base.with_derivative_rule(CallbackDerivativeRule(_nearest_hook))

    def _make_hook(offset: int, /):
        def _hook(
            *,
            var: str,
            axis: int | None,
            order: int,
            mode,
            backend,
            basis,
            periodic: bool,
        ) -> DomainFunction | None:
            del mode, basis, periodic
            if backend not in ("ad", "jet"):
                return None
            if var != domain.time_label or axis is not None:
                return None
            derivative_order = int(offset) + int(order)
            limit = reconstruction.capabilities.maximum_explicit_derivative_order
            if derivative_order > limit:
                if isinstance(domain, IrregularTrajectoryDatasetDomain):
                    raise ValueError(
                        "Irregular TrajectorySignal with interpolation='linear' supports "
                        "time derivatives only up to order 1."
                    )
                raise ValueError(
                    f"interpolation={reconstruction.interpolation!r} supports trajectory "
                    f"signal time derivatives only up to order {limit}."
                )
            result = (
                base
                if derivative_order == 0
                else _signal_function(
                    domain,
                    reconstruction,
                    derivative_order=derivative_order,
                )
            )
            return result.with_derivative_rule(
                CallbackDerivativeRule(_make_hook(derivative_order))
            )

        return _hook

    return base.with_derivative_rule(CallbackDerivativeRule(_make_hook(0)))


__all__ = ["TrajectorySignalInterpolation"]
