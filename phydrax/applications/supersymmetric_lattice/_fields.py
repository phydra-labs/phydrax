#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein
from phydrax.linalg import inverse

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


class PFormLatticePlan(StrictModule):
    """Static placement of oriented coordinate p-cells on a periodic lattice."""

    lattice_shape: tuple[int, ...] = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    orientations: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    endpoint_offsets: Array
    site_count: int = eqx.field(static=True)
    component_count: int = eqx.field(static=True)
    maximum_field_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lattice_shape: Sequence[int],
        degree: int,
        /,
        *,
        maximum_field_elements: int = 10_000_000,
    ):
        shape = tuple(lattice_shape)
        degree_ = int(degree)
        maximum = int(maximum_field_elements)
        if not shape or any(size < 1 for size in shape):
            raise ValueError("lattice_shape must contain positive extents.")
        if not 0 <= degree_ <= len(shape):
            raise ValueError("degree must lie between zero and the lattice dimension.")
        if maximum < 1:
            raise ValueError("maximum_field_elements must be positive.")
        orientations = tuple(combinations(range(len(shape)), degree_))
        offsets = np.zeros((len(orientations), len(shape)), dtype=np.int32)
        for component, axes in enumerate(orientations):
            if axes:
                offsets[component, np.asarray(axes, dtype=np.int32)] = 1
        self.lattice_shape = shape
        self.degree = degree_
        self.orientations = orientations
        self.endpoint_offsets = jnp.asarray(offsets)
        self.site_count = prod(shape)
        self.component_count = len(orientations)
        self.maximum_field_elements = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-coordinate-p-form-lattice",
                "lattice_shape": shape,
                "degree": degree_,
                "orientations": orientations,
                "maximum_field_elements": maximum,
            }
        )

    def configuration_shape(self, matrix_rank: int, /) -> tuple[int, ...]:
        rank = int(matrix_rank)
        if rank < 1:
            raise ValueError("matrix_rank must be positive.")
        count = self.site_count * self.component_count * rank * rank
        if count > self.maximum_field_elements:
            raise ValueError(
                f"p-form field requires {count} scalar elements; capacity is {self.maximum_field_elements}."
            )
        return self.lattice_shape + (self.component_count, rank, rank)


class ComplexifiedPFormField(StrictModule):
    """A fixed-shape GL(N,C)-covariant coordinate p-form field."""

    plan: PFormLatticePlan
    values: Array
    matrix_rank: int = eqx.field(static=True)
    orientation: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PFormLatticePlan,
        values: ArrayLike,
        /,
        *,
        orientation: str = "forward",
    ):
        if not isinstance(plan, PFormLatticePlan):
            raise TypeError("plan must be PFormLatticePlan.")
        if orientation not in ("forward", "reverse"):
            raise ValueError("orientation must be 'forward' or 'reverse'.")
        array = jnp.asarray(values)
        if array.ndim != len(plan.lattice_shape) + 3:
            raise ValueError("p-form values have the wrong rank.")
        matrix_rank = array.shape[-1]
        if array.shape[-2] != matrix_rank:
            raise ValueError("p-form matrix fibers must be square.")
        expected = plan.configuration_shape(matrix_rank)
        if array.shape != expected:
            raise ValueError(f"p-form values must have shape {expected}.")
        if not jnp.issubdtype(array.dtype, jnp.complexfloating):
            array = array.astype(jnp.complex128)
        self.plan = plan
        self.values = array
        self.matrix_rank = matrix_rank
        self.orientation = orientation
        self.field_space_id = canonical_fingerprint(
            {
                "kind": "complexified-gl-p-form-field-space",
                "plan": plan.plan_id,
                "matrix_rank": matrix_rank,
                "orientation": orientation,
                "dtype": str(array.dtype),
            }
        )


def _matrix_product(left: Array, right: Array, /) -> Array:
    return ein.contract("...ij,...jk->...ik", left, right)


def _shift_sites(values: Array, offset: Sequence[int], /) -> Array:
    shifted = values
    for axis, amount in enumerate(offset):
        if amount:
            shifted = jnp.roll(shifted, -int(amount), axis=axis)
    return shifted


def invert_gauge_transform(gauge: ArrayLike, /) -> Array:
    """Invert a fixed batch of GL matrices through the native dense linalg owner."""
    matrices = jnp.asarray(gauge)
    if matrices.ndim < 2 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("gauge must end in square matrix axes.")
    result = inverse(matrices)
    return eqx.error_if(
        result.value,
        jnp.any(~result.successful),
        "Gauge matrices must be nonsingular.",
    )


def transform_p_form(
    field: ComplexifiedPFormField,
    gauge: ArrayLike,
    /,
    *,
    inverse_gauge: ArrayLike | None = None,
) -> ComplexifiedPFormField:
    """Apply the endpoint GL(N,C) action dictated by p-cell orientation."""
    if not isinstance(field, ComplexifiedPFormField):
        raise TypeError("field must be ComplexifiedPFormField.")
    gauge_ = jnp.asarray(gauge, dtype=field.values.dtype)
    expected = field.plan.lattice_shape + (field.matrix_rank, field.matrix_rank)
    if gauge_.shape != expected:
        raise ValueError(f"gauge must have shape {expected}.")
    inverse_ = (
        invert_gauge_transform(gauge_)
        if inverse_gauge is None
        else jnp.asarray(inverse_gauge, dtype=field.values.dtype)
    )
    if inverse_.shape != expected:
        raise ValueError(f"inverse_gauge must have shape {expected}.")
    transformed = []
    for component, axes in enumerate(field.plan.orientations):
        offset = tuple(int(axis in axes) for axis in range(len(field.plan.lattice_shape)))
        gauge_end = _shift_sites(gauge_, offset)
        inverse_end = _shift_sites(inverse_, offset)
        value = field.values[..., component, :, :]
        if field.orientation == "forward":
            transformed.append(
                _matrix_product(_matrix_product(gauge_, value), inverse_end)
            )
        else:
            transformed.append(
                _matrix_product(_matrix_product(gauge_end, value), inverse_)
            )
    return ComplexifiedPFormField(
        field.plan,
        jnp.stack(transformed, axis=-3),
        orientation=field.orientation,
    )


def p_form_placement(field: ComplexifiedPFormField, /) -> tuple[tuple[int, ...], ...]:
    """Return the ordered axes spanning each stored p-cell component."""
    if not isinstance(field, ComplexifiedPFormField):
        raise TypeError("field must be ComplexifiedPFormField.")
    return field.plan.orientations


__all__ = [
    "ComplexifiedPFormField",
    "PFormLatticePlan",
    "invert_gauge_transform",
    "p_form_placement",
    "transform_p_form",
]
