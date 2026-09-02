#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainFunction, PointBatch, SampleLayout

from ..._doc import DOC_KEY0
from ..._frozendict import frozendict
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.spectral import PreparedSpectralAxis


MixedBoundsPolicy = Literal["error", "extrapolate"]


class MixedTensorReconstructionPlan(StrictModule, NonTrainableState):
    """Typed prepared spectral axes for mixed tensor analysis and synthesis."""

    axes: tuple[PreparedSpectralAxis, ...]
    axis_labels: tuple[str, ...] = eqx.field(static=True)
    payload_ndim: int = eqx.field(static=True)
    bounds: MixedBoundsPolicy = eqx.field(static=True)

    def __init__(
        self,
        axes: Sequence[PreparedSpectralAxis],
        axis_labels: Sequence[str],
        /,
        *,
        payload_ndim: int = 0,
        bounds: MixedBoundsPolicy = "error",
    ):
        axes_ = tuple(axes)
        labels = tuple(str(label) for label in axis_labels)
        payload = int(payload_ndim)
        if not axes_ or any(not isinstance(axis, PreparedSpectralAxis) for axis in axes_):
            raise TypeError("axes must contain prepared spectral axes.")
        if (
            len(labels) != len(axes_)
            or len(set(labels)) != len(labels)
            or any(not label for label in labels)
        ):
            raise ValueError("axis_labels must uniquely name every prepared axis.")
        if any(axis.family not in ("fourier", "chebyshev", "legendre") for axis in axes_):
            raise ValueError(
                "Mixed reconstruction supports Fourier, Chebyshev, and Legendre."
            )
        if payload < 0 or bounds not in ("error", "extrapolate"):
            raise ValueError("Invalid payload_ndim or bounds policy.")
        self.axes = axes_
        self.axis_labels = labels
        self.payload_ndim = payload
        self.bounds = bounds

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return tuple(int(axis.nodes.size) for axis in self.axes)


class MixedTensorInterpolant(StrictModule):
    coefficients: Array
    plan: MixedTensorReconstructionPlan

    def _basis(self, axis: PreparedSpectralAxis, coordinate: Array, order: int) -> Array:
        lower = jnp.asarray(axis.domain.lower, dtype=coordinate.dtype)
        upper = jnp.asarray(axis.domain.upper, dtype=coordinate.dtype)

        def base(value):
            if axis.family == "fourier":
                phase = 2.0 * jnp.pi * (value - lower) / (upper - lower)
                normalization = jnp.sqrt(jnp.asarray(axis.length, dtype=coordinate.dtype))
                return jnp.exp(1j * axis.modes.mode_numbers * phase) / normalization
            reference = (2.0 * value - lower - upper) / (upper - lower)
            count = axis.modes.count
            values = [jnp.ones((), dtype=reference.dtype)]
            if count > 1:
                values.append(reference)
            for degree in range(2, count):
                if axis.family == "chebyshev":
                    next_value = 2.0 * reference * values[-1] - values[-2]
                else:
                    next_value = (
                        (2 * degree - 1) * reference * values[-1]
                        - (degree - 1) * values[-2]
                    ) / degree
                values.append(next_value)
            return jnp.stack(values)

        result = base
        for _ in range(int(order)):
            result = jax.jacfwd(result)
        return result(coordinate)

    def __call__(
        self,
        coordinates: ArrayLike,
        /,
        *,
        derivative_orders: Sequence[int] | None = None,
    ) -> Array:
        points = jnp.asarray(coordinates)
        dimension = len(self.plan.axes)
        if points.ndim < 1 or points.shape[-1] != dimension:
            raise ValueError("Mixed tensor coordinates must end in the axis dimension.")
        if not jnp.issubdtype(points.dtype, jnp.inexact) or jnp.issubdtype(
            points.dtype, jnp.complexfloating
        ):
            raise TypeError("Mixed tensor coordinates must be real inexact arrays.")
        orders = (
            (0,) * dimension
            if derivative_orders is None
            else tuple(int(order) for order in derivative_orders)
        )
        if len(orders) != dimension or any(order < 0 for order in orders):
            raise ValueError("derivative_orders must be nonnegative per axis.")
        if self.plan.bounds == "error":
            valid = jnp.asarray(True)
            for axis_index, axis in enumerate(self.plan.axes):
                if axis.periodic:
                    continue
                valid &= jnp.all(
                    (points[..., axis_index] >= axis.domain.lower)
                    & (points[..., axis_index] <= axis.domain.upper)
                )
            points = eqx.error_if(
                points, ~valid, "Polynomial query lies outside support."
            )
        flat = points.reshape((-1, dimension))

        def evaluate(point):
            result = self.coefficients
            for axis, coordinate, order in zip(
                self.plan.axes, point, orders, strict=True
            ):
                basis = self._basis(axis, coordinate, order).astype(result.dtype)
                result = oe.contract("i,i...->...", basis, result)
            return result

        values = jax.vmap(evaluate)(flat)
        return values.reshape((*points.shape[:-1], *self.coefficients.shape[dimension:]))


def _analyze_axis(values: Array, axis: PreparedSpectralAxis, position: int) -> Array:
    moved = jnp.moveaxis(values, position, 0)
    shape = moved.shape
    flattened = moved.reshape((shape[0], -1))
    transformed = jax.vmap(axis.analyze, in_axes=1, out_axes=1)(flattened)
    restored = transformed.reshape((axis.modes.count, *shape[1:]))
    return jnp.moveaxis(restored, 0, position)


def fit_mixed_tensor(
    values: ArrayLike,
    plan: MixedTensorReconstructionPlan,
    /,
) -> MixedTensorInterpolant:
    """Analyze a canonical tensor grid one axis at a time."""
    if not isinstance(plan, MixedTensorReconstructionPlan):
        raise TypeError("plan must be a MixedTensorReconstructionPlan.")
    array = jnp.asarray(values)
    if array.shape[: len(plan.axes)] != plan.sample_shape:
        raise ValueError("Mixed tensor values do not match prepared axis node counts.")
    if array.ndim != len(plan.axes) + plan.payload_ndim:
        raise ValueError("Mixed tensor payload_ndim does not match values rank.")
    coefficients = array
    for position, axis in enumerate(plan.axes):
        coefficients = _analyze_axis(coefficients, axis, position)
    return MixedTensorInterpolant(coefficients=coefficients, plan=plan)


def interpolate_mixed_tensor(
    function: DomainFunction,
    plan: MixedTensorReconstructionPlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> DomainFunction:
    """Fit a DomainFunction on the prepared mixed canonical tensor grid."""
    if not isinstance(function, DomainFunction):
        raise TypeError("interpolate_mixed_tensor requires a DomainFunction.")
    if tuple(function.deps) != plan.axis_labels:
        raise ValueError("DomainFunction dependency order must match axis_labels.")
    mesh = jnp.meshgrid(*(axis.nodes for axis in plan.axes), indexing="ij")
    flattened = tuple(value.reshape((-1,)) for value in mesh)
    dependency_domain = function.domain.factor(plan.axis_labels[0])
    for label in plan.axis_labels[1:]:
        dependency_domain = dependency_domain.join(function.domain.factor(label))
    structure = SampleLayout((plan.axis_labels,)).canonicalize(dependency_domain.labels)
    sample_axis = structure.axis_for(plan.axis_labels[0])
    if sample_axis is None:
        raise RuntimeError("Mixed reconstruction grid has no sample axis.")
    points = PointBatch(
        frozendict(
            {
                label: cx.Field(values, dims=(sample_axis,))
                for label, values in zip(plan.axis_labels, flattened, strict=True)
            }
        ),
        structure,
    )
    fitting = DomainFunction(
        domain=dependency_domain,
        deps=plan.axis_labels,
        func=function.func,
        metadata={},
    )
    evaluated = jnp.asarray(fitting(points, key=key).data)
    values = evaluated.reshape((*plan.sample_shape, *evaluated.shape[1:]))
    interpolant = fit_mixed_tensor(values, plan)
    metadata: Mapping[str, Any] = function.metadata
    return DomainFunction(
        domain=function.domain,
        deps=plan.axis_labels,
        func=lambda *coordinates, key=None, **kwargs: interpolant(
            jnp.stack(
                tuple(jnp.asarray(value) for value in coordinates),
                axis=-1,
            )
        ),
        metadata=metadata,
    )


__all__ = [
    "MixedBoundsPolicy",
    "MixedTensorInterpolant",
    "MixedTensorReconstructionPlan",
    "fit_mixed_tensor",
    "interpolate_mixed_tensor",
]
