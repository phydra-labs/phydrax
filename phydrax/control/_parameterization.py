#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._interpolation import (
    bspline_evaluate,
    BSplineGrid,
    BSplineGridTransfer,
    ProjectionMethod,
)
from .._strict import StrictModule
from ..dynamics import TimeGrid
from ._problem import _identifier, _shape


def _coefficient_array(
    coefficients: ArrayLike,
    case_shape: tuple[int, ...],
    parameter_shape: tuple[int, ...],
    /,
) -> Array:
    values = jnp.asarray(coefficients)
    expected = case_shape + parameter_shape
    if tuple(values.shape) != expected:
        raise ValueError(
            f"Control coefficients must have shape {expected}; got {values.shape}."
        )
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(float)
    return values


def _case_shape(value: Sequence[int], /) -> tuple[int, ...]:
    cases = tuple(int(size) for size in value)
    if any(size <= 0 for size in cases):
        raise ValueError("Control coefficient case dimensions must be positive.")
    return cases


def _query(value: ArrayLike, /) -> Array:
    query = jnp.asarray(value)
    if jnp.issubdtype(query.dtype, jnp.complexfloating):
        raise TypeError("Control evaluation times must be real-valued.")
    return query.astype(jnp.result_type(query, float))


class AbstractControlParameterization(StrictModule):
    """Fixed-shape map from coefficients to physical controls."""

    control_shape: tuple[int, ...] = eqx.field(static=True)
    parameter_shape: tuple[int, ...] = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    @abstractmethod
    def evaluate(
        self,
        coefficients: ArrayLike,
        time: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        state: ArrayLike | None = None,
    ) -> Array:
        """Evaluate with shape ``case_shape + time.shape + control_shape``."""
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        coefficients: ArrayLike,
        times: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        """Sample an open-loop parameterization at shared physical times."""
        raise NotImplementedError


class PiecewiseConstantControlParameterization(AbstractControlParameterization):
    """Left-endpoint-held interval controls on a fixed physical time grid."""

    time_grid: TimeGrid

    def __init__(
        self,
        time_grid: TimeGrid,
        control_shape: Sequence[int],
        /,
        *,
        parameterization_id: str,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        shape = _shape(control_shape, "control_shape")
        self.time_grid = time_grid
        self.control_shape = shape
        self.parameter_shape = (time_grid.num_steps,) + shape
        self.parameterization_id = _identifier(parameterization_id, "parameterization_id")
        self.approximation_id = "control:piecewise-constant:left-endpoint-hold"

    def evaluate(
        self,
        coefficients: ArrayLike,
        time: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        state: ArrayLike | None = None,
    ) -> Array:
        del state
        cases = _case_shape(case_shape)
        values = _coefficient_array(coefficients, cases, self.parameter_shape)
        query = _query(time)
        query = eqx.error_if(
            query,
            jnp.any(~jnp.isfinite(query))
            | jnp.any(query < self.time_grid.t0)
            | jnp.any(query > self.time_grid.t1),
            "Piecewise-constant control time lies outside its physical grid.",
        )
        indices = jnp.searchsorted(self.time_grid.times, query, side="right") - 1
        indices = jnp.minimum(indices, self.time_grid.num_steps - 1)
        return jnp.take(values, indices, axis=len(cases))

    def sample(
        self,
        coefficients: ArrayLike,
        times: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        return self.evaluate(coefficients, times, case_shape=case_shape)


class PiecewiseLinearControlParameterization(AbstractControlParameterization):
    """Continuous nodal controls linearly interpolated in physical time."""

    time_grid: TimeGrid

    def __init__(
        self,
        time_grid: TimeGrid,
        control_shape: Sequence[int],
        /,
        *,
        parameterization_id: str,
    ):
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid.")
        shape = _shape(control_shape, "control_shape")
        self.time_grid = time_grid
        self.control_shape = shape
        self.parameter_shape = (time_grid.num_times,) + shape
        self.parameterization_id = _identifier(parameterization_id, "parameterization_id")
        self.approximation_id = "control:piecewise-linear:nodal"

    def evaluate(
        self,
        coefficients: ArrayLike,
        time: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        state: ArrayLike | None = None,
    ) -> Array:
        del state
        cases = _case_shape(case_shape)
        values = _coefficient_array(coefficients, cases, self.parameter_shape)
        query = _query(time)
        query = eqx.error_if(
            query,
            jnp.any(~jnp.isfinite(query))
            | jnp.any(query < self.time_grid.t0)
            | jnp.any(query > self.time_grid.t1),
            "Piecewise-linear control time lies outside its physical grid.",
        )
        lower_indices = jnp.searchsorted(self.time_grid.times, query, side="right") - 1
        lower_indices = jnp.minimum(lower_indices, self.time_grid.num_steps - 1)
        upper_indices = lower_indices + 1
        lower_time = self.time_grid.times[lower_indices]
        upper_time = self.time_grid.times[upper_indices]
        fraction = (query - lower_time) / (upper_time - lower_time)
        lower = jnp.take(values, lower_indices, axis=len(cases))
        upper = jnp.take(values, upper_indices, axis=len(cases))
        payload_ndim = len(self.control_shape)
        weight = fraction.reshape(
            (1,) * len(cases) + fraction.shape + (1,) * payload_ndim
        )
        return lower + weight * (upper - lower)

    def sample(
        self,
        coefficients: ArrayLike,
        times: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        return self.evaluate(coefficients, times, case_shape=case_shape)


class BSplineControlBoundCertificate(StrictModule):
    """Convex-hull certificate for fixed-grid B-spline control bounds."""

    lower_bound: Array
    upper_bound: Array
    coefficient_minimum: Array
    coefficient_maximum: Array
    certified: Array
    parameterization_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)
    continuous_domain: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        lower_bound: ArrayLike,
        upper_bound: ArrayLike,
        coefficient_minimum: ArrayLike,
        coefficient_maximum: ArrayLike,
        certified: ArrayLike,
        parameterization_id: str,
    ):
        self.lower_bound = jnp.asarray(lower_bound)
        self.upper_bound = jnp.asarray(upper_bound)
        self.coefficient_minimum = jnp.asarray(coefficient_minimum)
        self.coefficient_maximum = jnp.asarray(coefficient_maximum)
        self.certified = jnp.asarray(certified, dtype=bool)
        self.parameterization_id = _identifier(
            parameterization_id, "certificate parameterization_id"
        )
        self.certificate_id = "control-bound:bspline-convex-hull"
        self.continuous_domain = True


class BSplineControlRefinement(StrictModule):
    """A diagnosed B-spline grid transfer and its refined coefficients."""

    parameterization: BSplineControlParameterization
    coefficients: Array
    transfer: BSplineGridTransfer
    source_parameterization_id: str = eqx.field(static=True)
    target_parameterization_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        parameterization: BSplineControlParameterization,
        coefficients: ArrayLike,
        transfer: BSplineGridTransfer,
        source_parameterization_id: str,
    ):
        self.parameterization = parameterization
        self.coefficients = jnp.asarray(coefficients)
        self.transfer = transfer
        self.source_parameterization_id = _identifier(
            source_parameterization_id, "source_parameterization_id"
        )
        self.target_parameterization_id = parameterization.parameterization_id


class BSplineControlParameterization(AbstractControlParameterization):
    """Differentiable fixed-grid B-spline control in physical time."""

    grid: BSplineGrid

    def __init__(
        self,
        grid: BSplineGrid,
        control_shape: Sequence[int],
        /,
        *,
        parameterization_id: str,
    ):
        if not isinstance(grid, BSplineGrid):
            raise TypeError("grid must be a BSplineGrid.")
        shape = _shape(control_shape, "control_shape")
        self.grid = grid
        self.control_shape = shape
        self.parameter_shape = (grid.coefficient_count,) + shape
        self.parameterization_id = _identifier(parameterization_id, "parameterization_id")
        self.approximation_id = f"control:bspline:fixed-grid:degree-{grid.degree}"

    def evaluate(
        self,
        coefficients: ArrayLike,
        time: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        state: ArrayLike | None = None,
    ) -> Array:
        del state
        cases = _case_shape(case_shape)
        values = _coefficient_array(coefficients, cases, self.parameter_shape)
        query = _query(time)
        case_query = jnp.broadcast_to(query, cases + query.shape)
        return bspline_evaluate(
            self.grid.knots,
            values,
            case_query,
            degree=self.grid.degree,
            bounds="error",
            case_shape=cases,
        ).values

    def sample(
        self,
        coefficients: ArrayLike,
        times: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> Array:
        return self.evaluate(coefficients, times, case_shape=case_shape)

    def bound_certificate(
        self,
        coefficients: ArrayLike,
        lower_bound: ArrayLike,
        upper_bound: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
    ) -> BSplineControlBoundCertificate:
        """Certify continuous bounds from the nonnegative partition of unity."""
        cases = _case_shape(case_shape)
        values = _coefficient_array(coefficients, cases, self.parameter_shape)
        lower = jnp.asarray(lower_bound)
        upper = jnp.asarray(upper_bound)
        if lower.shape not in ((), self.control_shape):
            raise ValueError("lower_bound must be scalar or have control_shape.")
        if upper.shape not in ((), self.control_shape):
            raise ValueError("upper_bound must be scalar or have control_shape.")
        lower = jnp.broadcast_to(lower, self.control_shape)
        upper = jnp.broadcast_to(upper, self.control_shape)
        lower = eqx.error_if(
            lower,
            jnp.any(~jnp.isfinite(lower))
            | jnp.any(~jnp.isfinite(upper))
            | jnp.any(lower > upper),
            "B-spline control bounds must be finite and ordered.",
        )
        coefficient_axis = len(cases)
        minimum = jnp.min(values, axis=coefficient_axis)
        maximum = jnp.max(values, axis=coefficient_axis)
        control_axes = tuple(range(len(cases), minimum.ndim))
        within = (minimum >= lower) & (maximum <= upper)
        certified = jnp.all(within, axis=control_axes) if control_axes else within
        return BSplineControlBoundCertificate(
            lower_bound=lower,
            upper_bound=upper,
            coefficient_minimum=minimum,
            coefficient_maximum=maximum,
            certified=certified,
            parameterization_id=self.parameterization_id,
        )

    def refine(
        self,
        new_grid: BSplineGrid,
        coefficients: ArrayLike,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        parameterization_id: str,
        method: ProjectionMethod = "auto",
        maximum_condition: float = 1.0e12,
    ) -> BSplineControlRefinement:
        """Transfer coefficients through the canonical diagnosed grid transfer."""
        if not isinstance(new_grid, BSplineGrid):
            raise TypeError("new_grid must be a BSplineGrid.")
        cases = _case_shape(case_shape)
        values = _coefficient_array(coefficients, cases, self.parameter_shape)
        transfer = BSplineGridTransfer(
            self.grid,
            new_grid,
            method=method,
            maximum_condition=maximum_condition,
        )
        refined = transfer(values, coefficient_axis=len(cases))
        parameterization = BSplineControlParameterization(
            new_grid,
            self.control_shape,
            parameterization_id=parameterization_id,
        )
        return BSplineControlRefinement(
            parameterization=parameterization,
            coefficients=refined,
            transfer=transfer,
            source_parameterization_id=self.parameterization_id,
        )


__all__ = [
    "AbstractControlParameterization",
    "BSplineControlBoundCertificate",
    "BSplineControlParameterization",
    "BSplineControlRefinement",
    "PiecewiseConstantControlParameterization",
    "PiecewiseLinearControlParameterization",
]
