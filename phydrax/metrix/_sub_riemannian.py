#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ._chart import CoordinateChart
from ._density import VolumeDensity
from ._metric import _metric_inverse
from ._utils import _pointwise_array


class HorizontalCometric(StrictModule):
    """Positive cometric on a declared horizontal frame distribution."""

    frame_function: Callable[[Array], Array]
    control_metric_function: Callable[[Array], Array] | None
    chart: CoordinateChart
    rank: int = eqx.field(static=True)

    def __init__(
        self,
        frame_function: Callable[[Array], Array],
        chart: CoordinateChart,
        rank: int,
        /,
        *,
        control_metric: Callable[[Array], Array] | None = None,
    ):
        if not callable(frame_function):
            raise TypeError("frame_function must be callable.")
        if control_metric is not None and not callable(control_metric):
            raise TypeError("control_metric must be callable when supplied.")
        if not isinstance(chart, CoordinateChart):
            raise TypeError("chart must be a CoordinateChart.")
        if int(rank) <= 0 or int(rank) > chart.dimension:
            raise ValueError("Horizontal rank must lie between one and chart dimension.")
        self.frame_function = frame_function
        self.control_metric_function = control_metric
        self.chart = chart
        self.rank = int(rank)

    def frame(self, coordinates: ArrayLike, /) -> Array:
        expected = (self.chart.dimension, self.rank)

        def evaluate(point: Array) -> Array:
            value = jnp.asarray(self.frame_function(point))
            if value.shape != expected:
                raise ValueError(
                    f"Horizontal frame must have shape {expected}; got {value.shape}."
                )
            return value

        return _pointwise_array(evaluate, coordinates, self.chart.dimension)

    def control_metric(self, coordinates: ArrayLike, /) -> Array:
        expected = (self.rank, self.rank)

        def evaluate(point: Array) -> Array:
            if self.control_metric_function is None:
                return jnp.eye(self.rank, dtype=point.dtype)
            value = jnp.asarray(self.control_metric_function(point))
            if value.shape != expected:
                raise ValueError(
                    f"Horizontal control metric must have shape {expected}; "
                    f"got {value.shape}."
                )
            return value

        return _pointwise_array(evaluate, coordinates, self.chart.dimension)

    def __call__(self, coordinates: ArrayLike, /) -> Array:
        frame = self.frame(coordinates)
        control = self.control_metric(coordinates)
        inverse = _metric_inverse(control, positive_definite=True)
        return ein.contract("...ia,...ab,...jb->...ij", frame, inverse, frame)


def horizontal_gradient(
    field: Callable[[Array], Array],
    cometric: HorizontalCometric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Raise ``df`` by a horizontal cometric."""
    if not callable(field):
        raise TypeError("field must be callable.")
    if not isinstance(cometric, HorizontalCometric):
        raise TypeError("cometric must be a HorizontalCometric.")

    def evaluate(point: Array) -> Array:
        value = jnp.asarray(field(point))
        if value.shape != ():
            raise ValueError("horizontal_gradient requires a scalar field.")
        differential = jax.grad(field)(point)
        return ein.contract("ij,j->i", cometric(point), differential)

    return _pointwise_array(evaluate, coordinates, cometric.chart.dimension)


def horizontal_hamiltonian(
    covector: ArrayLike,
    cometric: HorizontalCometric,
    coordinates: ArrayLike,
    /,
) -> Array:
    """Return the normal sub-Riemannian Hamiltonian ``pᵀap / 2``."""
    if not isinstance(cometric, HorizontalCometric):
        raise TypeError("cometric must be a HorizontalCometric.")
    covector_array = jnp.asarray(covector)
    if covector_array.shape[-1] != cometric.chart.dimension:
        raise ValueError(
            "Horizontal Hamiltonian covectors must match the chart dimension."
        )
    return 0.5 * ein.contract(
        "...i,...ij,...j->...",
        covector_array,
        cometric(coordinates),
        covector_array,
    )


def sub_riemannian_hamiltonian_rhs(
    cometric: HorizontalCometric,
    state: ArrayLike,
    /,
) -> Array:
    """Return canonical Hamilton equations for ``(q, p)``."""
    if not isinstance(cometric, HorizontalCometric):
        raise TypeError("cometric must be a HorizontalCometric.")
    values = jnp.asarray(state)
    dimension = cometric.chart.dimension
    if values.shape[-1:] != (2 * dimension,):
        raise ValueError(
            f"Sub-Riemannian Hamiltonian state must end in {2 * dimension} values."
        )
    leading = values.shape[:-1]
    flat = values.reshape((-1, 2 * dimension))

    def evaluate(local_state: Array) -> Array:
        def hamiltonian(candidate: Array) -> Array:
            coordinates = candidate[:dimension]
            covector = candidate[dimension:]
            return horizontal_hamiltonian(covector, cometric, coordinates)

        differential = jax.grad(hamiltonian)(local_state)
        return jnp.concatenate(
            (differential[dimension:], -differential[:dimension]), axis=-1
        )

    result = jax.vmap(evaluate)(flat)
    return result.reshape(leading + (2 * dimension,))


def sub_laplacian(
    field: Callable[[Array], Array],
    cometric: HorizontalCometric,
    coordinates: ArrayLike,
    /,
    *,
    density: VolumeDensity | None = None,
) -> Array:
    """Horizontal divergence of the horizontal gradient."""
    if not callable(field):
        raise TypeError("field must be callable.")
    if not isinstance(cometric, HorizontalCometric):
        raise TypeError("cometric must be a HorizontalCometric.")
    if density is not None and not isinstance(density, VolumeDensity):
        raise TypeError("density must be a VolumeDensity or None.")
    if density is not None and not density.chart.compatible_with(cometric.chart):
        raise ValueError("Density and horizontal cometric charts must match.")

    def evaluate(point: Array) -> Array:
        def weighted_gradient(local_point: Array) -> Array:
            gradient = horizontal_gradient(field, cometric, local_point)
            if density is None:
                return gradient
            return density(local_point) * gradient

        divergence = jnp.trace(jax.jacfwd(weighted_gradient)(point))
        return divergence if density is None else divergence / density(point)

    return _pointwise_array(evaluate, coordinates, cometric.chart.dimension)


def step_two_horizontal_rank(
    cometric: HorizontalCometric,
    coordinates: ArrayLike,
    /,
    *,
    tolerance: float | None = None,
) -> Array:
    """Rank of the horizontal frame augmented by all first Lie brackets."""
    if not isinstance(cometric, HorizontalCometric):
        raise TypeError("cometric must be a HorizontalCometric.")
    if tolerance is not None and tolerance < 0.0:
        raise ValueError("tolerance must be non-negative or None.")

    def evaluate(point: Array) -> Array:
        frame = cometric.frame(point)
        fields = tuple(
            lambda local_point, index=index: cometric.frame(local_point)[:, index]
            for index in range(cometric.rank)
        )
        brackets = []
        for left in range(cometric.rank):
            for right in range(left + 1, cometric.rank):
                left_value = fields[left](point)
                right_value = fields[right](point)
                bracket = (
                    jax.jacfwd(fields[right])(point) @ left_value
                    - jax.jacfwd(fields[left])(point) @ right_value
                )
                brackets.append(bracket)
        augmented = (
            frame
            if not brackets
            else jnp.concatenate((frame, jnp.stack(brackets, axis=-1)), axis=-1)
        )
        return jnp.linalg.matrix_rank(augmented, tol=tolerance)

    return _pointwise_array(evaluate, coordinates, cometric.chart.dimension)


class HorizontalValidationReport(StrictModule):
    valid: Array
    finite: Array
    minimum_frame_singular_value: Array
    step_two_rank: Array

    def __init__(
        self,
        *,
        valid: ArrayLike,
        finite: ArrayLike,
        minimum_frame_singular_value: ArrayLike,
        step_two_rank: ArrayLike,
    ):
        self.valid = jnp.asarray(valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.minimum_frame_singular_value = jnp.asarray(minimum_frame_singular_value)
        self.step_two_rank = jnp.asarray(step_two_rank)


def validate_horizontal_cometric(
    cometric: HorizontalCometric,
    points: ArrayLike,
    /,
    *,
    singular_value_tolerance: float = 1e-10,
    require_step_two_bracket_generating: bool = False,
    raise_on_error: bool = False,
) -> HorizontalValidationReport:
    """Validate frame rank and optional step-two bracket generation."""
    if not isinstance(cometric, HorizontalCometric):
        raise TypeError("cometric must be a HorizontalCometric.")
    if singular_value_tolerance < 0.0:
        raise ValueError("singular_value_tolerance must be non-negative.")
    frame = cometric.frame(points)
    singular_values = jnp.linalg.svd(frame, compute_uv=False)
    minimum = jnp.min(singular_values, axis=-1)
    finite = jnp.all(jnp.isfinite(frame), axis=(-2, -1))
    rank_valid = minimum > singular_value_tolerance
    step_two_rank = step_two_horizontal_rank(
        cometric,
        points,
        tolerance=singular_value_tolerance,
    )
    bracket_valid = step_two_rank == cometric.chart.dimension
    valid = finite & rank_valid
    if require_step_two_bracket_generating:
        valid = valid & bracket_valid
    report = HorizontalValidationReport(
        valid=valid,
        finite=finite,
        minimum_frame_singular_value=minimum,
        step_two_rank=step_two_rank,
    )
    if raise_on_error and not bool(jax.device_get(jnp.all(valid))):
        raise ValueError("Horizontal cometric validation failed.")
    return report


__all__ = [
    "HorizontalCometric",
    "HorizontalValidationReport",
    "horizontal_gradient",
    "horizontal_hamiltonian",
    "sub_riemannian_hamiltonian_rhs",
    "step_two_horizontal_rank",
    "sub_laplacian",
    "validate_horizontal_cometric",
]
