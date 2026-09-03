#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import inverse_small_linear, SmallLinearSolvePlan


class FiniteElementMetricData(StrictModule):
    """Shared pointwise geometry used by dense and factorized cell actions."""

    physical_points: Array
    jacobian: Array
    inverse_jacobian: Array
    cofactor: Array
    measure: Array
    weighted_measure: Array
    weighted_metric: Array

    def __init__(
        self,
        coordinate_basis: ArrayLike,
        coordinate_gradients: ArrayLike,
        cell_coordinates: ArrayLike,
        reference_weights: ArrayLike,
        /,
    ):
        basis = jnp.asarray(coordinate_basis)
        gradients = jnp.asarray(coordinate_gradients)
        coordinates = jnp.asarray(cell_coordinates)
        weights = jnp.asarray(reference_weights)
        if (
            basis.ndim != 2
            or gradients.ndim != 3
            or coordinates.ndim != 3
            or gradients.shape[:2] != basis.shape
            or coordinates.shape[1] != basis.shape[1]
            or weights.shape != (basis.shape[0],)
        ):
            raise ValueError("Coordinate basis, routes, and reference weights disagree.")
        dimension = gradients.shape[-1]
        if coordinates.shape[-1] != dimension or dimension not in (2, 3):
            raise ValueError("Tensor cell metrics require square 2-D or 3-D geometry.")
        physical_points = ein.contract("qi,cid->cqd", basis, coordinates)
        jacobian = ein.contract("qir,cid->cqdr", gradients, coordinates)
        inverse_result = inverse_small_linear(
            SmallLinearSolvePlan(dimension),
            jacobian,
        )
        determinant = inverse_result.determinant
        measure = jnp.abs(determinant)
        measure = eqx.error_if(
            measure,
            jnp.any(
                ~inverse_result.successful | ~jnp.isfinite(measure) | (measure <= 0.0)
            ),
            "Finite-element metric determinant must be positive and finite.",
        )
        inverse_jacobian = inverse_result.value
        cofactor = measure[..., None, None] * inverse_jacobian
        inverse_metric = ein.contract(
            "cqrd,cqsd->cqrs", inverse_jacobian, inverse_jacobian
        )
        self.physical_points = physical_points
        self.jacobian = jacobian
        self.inverse_jacobian = inverse_jacobian
        self.cofactor = cofactor
        self.measure = measure
        self.weighted_measure = measure * weights[None, :]
        self.weighted_metric = inverse_metric * self.weighted_measure[..., None, None]

    def physical_gradients(self, reference_gradients: ArrayLike, /) -> Array:
        gradients = jnp.asarray(reference_gradients)
        if gradients.ndim != 3 or gradients.shape[0] != self.jacobian.shape[1]:
            raise ValueError("Reference gradients do not match the cell metric points.")
        return ein.contract("qir,cqrd->cqid", gradients, self.inverse_jacobian)


class FiniteElementFacetMetricData(StrictModule):
    """Physical weights and one outward normal from a cell-side facet metric."""

    physical_points: Array
    physical_weights: Array
    normal: Array
    measure: Array

    def __init__(
        self,
        cell_metric: FiniteElementMetricData,
        reference_normals: ArrayLike,
        reference_weights: ArrayLike,
        /,
    ):
        if not isinstance(cell_metric, FiniteElementMetricData):
            raise TypeError("cell_metric must be FiniteElementMetricData.")
        normals = jnp.asarray(reference_normals)
        weights = jnp.asarray(reference_weights)
        if normals.shape != (
            cell_metric.jacobian.shape[1],
            cell_metric.jacobian.shape[-1],
        ) or weights.shape != (normals.shape[0],):
            raise ValueError("Facet normal/weight axes do not match the cell metric.")
        surface_vector = ein.contract("qr,cqrd->cqd", normals, cell_metric.cofactor)
        measure = jnp.linalg.norm(surface_vector, axis=-1)
        measure = eqx.error_if(
            measure,
            jnp.any(~jnp.isfinite(measure) | (measure <= 0.0)),
            "Finite-element facet measure must be positive and finite.",
        )
        self.physical_points = cell_metric.physical_points
        self.physical_weights = measure * weights[None, :]
        self.normal = surface_vector / measure[..., None]
        self.measure = measure


class PreparedFacetTrace(StrictModule):
    """One trace and its exact Euclidean transpose in canonical facet order."""

    basis_values: Array
    local_to_canonical: Array
    trace_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis_values: ArrayLike,
        local_to_canonical: ArrayLike,
        /,
    ):
        basis = jnp.asarray(basis_values)
        permutation = np.asarray(local_to_canonical, dtype=np.int32)
        if basis.ndim != 2 or permutation.shape != (basis.shape[0],):
            raise ValueError("Facet trace basis and point permutation are incompatible.")
        if tuple(sorted(int(value) for value in permutation)) != tuple(
            range(permutation.size)
        ):
            raise ValueError("Facet trace permutation must be bijective.")
        self.basis_values = basis
        self.local_to_canonical = jnp.asarray(permutation)
        self.trace_id = canonical_fingerprint(
            {
                "kind": "prepared-facet-trace",
                "basis": array_tree_fingerprint(np.asarray(basis)),
                "permutation": array_tree_fingerprint(permutation),
            }
        )

    def trace(self, coefficients: ArrayLike, /) -> Array:
        local = ein.contract(
            "qi,...i->...q", self.basis_values, jnp.asarray(coefficients)
        )
        canonical = jnp.zeros_like(local)
        return canonical.at[..., self.local_to_canonical].set(local)

    def lift(self, canonical_values: ArrayLike, /) -> Array:
        canonical = jnp.asarray(canonical_values)
        if canonical.shape[-1] != self.local_to_canonical.shape[0]:
            raise ValueError("Canonical facet values have the wrong point width.")
        local = canonical[..., self.local_to_canonical]
        return ein.contract("qi,...q->...i", self.basis_values, local)


class FieldJet(StrictModule):
    value: Array
    gradient: Array | None
    divergence: Array | None
    curl: Array | None

    def __init__(
        self,
        value: ArrayLike,
        /,
        *,
        gradient: ArrayLike | None = None,
        divergence: ArrayLike | None = None,
        curl: ArrayLike | None = None,
    ):
        self.value = jnp.asarray(value)
        self.gradient = None if gradient is None else jnp.asarray(gradient)
        self.divergence = None if divergence is None else jnp.asarray(divergence)
        self.curl = None if curl is None else jnp.asarray(curl)


class FacetJet(StrictModule):
    """Two scalar traces expressed with one plus-oriented facet normal."""

    plus_value: Array
    minus_value: Array
    plus_gradient: Array
    minus_gradient: Array
    plus_normal_derivative: Array
    minus_normal_derivative: Array
    jump: Array
    average: Array
    normal: Array
    measure: Array

    def __init__(
        self,
        plus_value: ArrayLike,
        minus_value: ArrayLike,
        plus_gradient: ArrayLike,
        minus_gradient: ArrayLike,
        normal: ArrayLike,
        measure: ArrayLike,
        /,
    ):
        plus = jnp.asarray(plus_value)
        minus = jnp.asarray(minus_value)
        plus_gradient_ = jnp.asarray(plus_gradient)
        minus_gradient_ = jnp.asarray(minus_gradient)
        normal_ = jnp.asarray(normal)
        measure_ = jnp.asarray(measure)
        if plus.shape != minus.shape:
            raise ValueError("Facet plus/minus values must have identical shapes.")
        if plus_gradient_.shape != minus_gradient_.shape:
            raise ValueError("Facet plus/minus gradients must have identical shapes.")
        if plus_gradient_.shape[:-1] != plus.shape:
            raise ValueError("Facet scalar gradients must extend the value shape.")
        if normal_.shape != plus_gradient_.shape:
            raise ValueError("Facet normals must match scalar-gradient shapes.")
        if measure_.shape != plus.shape:
            raise ValueError("Facet measure must match scalar trace values.")
        self.plus_value = plus
        self.minus_value = minus
        self.plus_gradient = plus_gradient_
        self.minus_gradient = minus_gradient_
        self.plus_normal_derivative = jnp.sum(plus_gradient_ * normal_, axis=-1)
        self.minus_normal_derivative = jnp.sum(minus_gradient_ * normal_, axis=-1)
        self.jump = plus - minus
        self.average = 0.5 * (plus + minus)
        self.normal = normal_
        self.measure = measure_


class CellDerivativeBatch(StrictModule):
    """Cell-local coefficients and evaluated values/gradients for one action."""

    local_coefficients: Array
    value: Array
    gradient: Array

    def __init__(
        self,
        local_coefficients: ArrayLike,
        basis_values: ArrayLike,
        physical_gradients: ArrayLike,
        /,
    ):
        coefficients = jnp.asarray(local_coefficients)
        basis = jnp.asarray(basis_values)
        gradients = jnp.asarray(physical_gradients)
        if coefficients.ndim != 2 or gradients.ndim != 4:
            raise ValueError("Scalar cell derivative staging requires rank-2/4 data.")
        if (
            gradients.shape[0] != coefficients.shape[0]
            or gradients.shape[2] != coefficients.shape[1]
        ):
            raise ValueError("Cell derivative staging local layouts are incompatible.")
        if basis.ndim == 2:
            value = ein.contract("qi,ei->eq", basis, coefficients)
        elif basis.ndim == 3 and basis.shape[:1] == coefficients.shape[:1]:
            value = ein.contract("eqi,ei->eq", basis, coefficients)
        else:
            raise ValueError("Cell derivative basis values have an invalid layout.")
        self.local_coefficients = coefficients
        self.value = value
        self.gradient = ein.contract("eqid,ei->eqd", gradients, coefficients)


class DGTraceBatch(StrictModule):
    """Packed plus/minus DG traces computed once for a facet action group."""

    jet: FacetJet
    plus_local_coefficients: Array
    minus_local_coefficients: Array

    def __init__(
        self,
        plus_local_coefficients: ArrayLike,
        minus_local_coefficients: ArrayLike,
        plus_basis_values: ArrayLike,
        minus_basis_values: ArrayLike,
        plus_physical_gradients: ArrayLike,
        minus_physical_gradients: ArrayLike,
        normal: ArrayLike,
        measure: ArrayLike,
        /,
    ):
        plus = CellDerivativeBatch(
            plus_local_coefficients,
            plus_basis_values,
            plus_physical_gradients,
        )
        minus = CellDerivativeBatch(
            minus_local_coefficients,
            minus_basis_values,
            minus_physical_gradients,
        )
        self.jet = FacetJet(
            plus.value,
            minus.value,
            plus.gradient,
            minus.gradient,
            normal,
            measure,
        )
        self.plus_local_coefficients = plus.local_coefficients
        self.minus_local_coefficients = minus.local_coefficients


def symmetric_gradient(gradient: ArrayLike, /) -> Array:
    gradient_ = jnp.asarray(gradient)
    if gradient_.shape[-1] != gradient_.shape[-2]:
        raise ValueError("Symmetric gradient requires square value/coordinate axes.")
    return 0.5 * (gradient_ + jnp.swapaxes(gradient_, -1, -2))


def divergence(gradient: ArrayLike, /) -> Array:
    gradient_ = jnp.asarray(gradient)
    if gradient_.shape[-1] != gradient_.shape[-2]:
        raise ValueError("Divergence requires matching value/coordinate dimensions.")
    return jnp.trace(gradient_, axis1=-2, axis2=-1)


def curl(gradient: ArrayLike, /) -> Array:
    gradient_ = jnp.asarray(gradient)
    if gradient_.shape[-2:] == (2, 2):
        return gradient_[..., 1, 0] - gradient_[..., 0, 1]
    if gradient_.shape[-2:] == (3, 3):
        return jnp.stack(
            (
                gradient_[..., 2, 1] - gradient_[..., 1, 2],
                gradient_[..., 0, 2] - gradient_[..., 2, 0],
                gradient_[..., 1, 0] - gradient_[..., 0, 1],
            ),
            axis=-1,
        )
    raise ValueError("Curl requires a two- or three-dimensional vector gradient.")


def normal_trace(value: ArrayLike, normal: ArrayLike, /) -> Array:
    value_ = jnp.asarray(value)
    normal_ = jnp.asarray(normal)
    if value_.shape[-1] != normal_.shape[-1]:
        raise ValueError("Normal trace value and normal dimensions must match.")
    return jnp.sum(value_ * normal_, axis=-1)


def tangential_trace(value: ArrayLike, normal: ArrayLike, /) -> Array:
    value_ = jnp.asarray(value)
    normal_ = jnp.asarray(normal)
    if value_.shape[-1] == 2:
        tangent = jnp.stack((-normal_[..., 1], normal_[..., 0]), axis=-1)
        return jnp.sum(value_ * tangent, axis=-1)
    if value_.shape[-1] == 3:
        return value_ - jnp.sum(value_ * normal_, axis=-1, keepdims=True) * normal_
    raise ValueError("Tangential trace requires a two- or three-dimensional value.")


def jump(plus: ArrayLike, minus: ArrayLike, /) -> Array:
    plus_ = jnp.asarray(plus)
    minus_ = jnp.asarray(minus)
    if plus_.shape != minus_.shape:
        raise ValueError("Jump operands must have identical shape.")
    return plus_ - minus_


def average(plus: ArrayLike, minus: ArrayLike, /) -> Array:
    plus_ = jnp.asarray(plus)
    minus_ = jnp.asarray(minus)
    if plus_.shape != minus_.shape:
        raise ValueError("Average operands must have identical shape.")
    return 0.5 * (plus_ + minus_)


__all__ = [
    "CellDerivativeBatch",
    "DGTraceBatch",
    "FiniteElementFacetMetricData",
    "FiniteElementMetricData",
    "FacetJet",
    "FieldJet",
    "PreparedFacetTrace",
    "average",
    "curl",
    "divergence",
    "jump",
    "normal_trace",
    "symmetric_gradient",
    "tangential_trace",
]
