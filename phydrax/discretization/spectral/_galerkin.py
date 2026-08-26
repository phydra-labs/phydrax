#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import barycentric_differentiation_matrix
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    LinearSolveResult,
)
from ._space import TensorSpectralDiscretization


def _apply_axis_matrix(value: Array, matrix: Array, axis: int, /) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    flattened = moved.reshape((moved.shape[0], -1))
    transformed = matrix @ flattened
    restored = transformed.reshape((matrix.shape[0],) + moved.shape[1:])
    return jnp.moveaxis(restored, 0, axis)


def _axis_basis_values(discretization: TensorSpectralDiscretization, axis: int, /):
    prepared = discretization.axes[axis]
    identity = jnp.eye(
        prepared.mode_count,
        dtype=jnp.dtype(discretization.plan.precision.coefficient_dtype),
    )
    values = jax.vmap(prepared.synthesize)(identity).T
    if prepared.family in ("chebyshev", "legendre"):
        derivatives = (
            barycentric_differentiation_matrix(prepared.nodes) @ values
            if prepared.derivative_matrix is None
            else values @ prepared.derivative_matrix
        )
    elif prepared.family == "fourier":
        derivatives = jax.vmap(
            lambda coefficient: prepared.synthesize(
                prepared.derivative_multiplier(1) * coefficient
            )
        )(identity).T
    else:
        from ...operators.differential._array_ops import _basis_nth_derivative

        derivatives = jax.vmap(
            lambda column: _basis_nth_derivative(
                column,
                prepared.nodes,
                axis=0,
                order=1,
                basis=prepared.family,
            ),
            in_axes=1,
            out_axes=1,
        )(values)
    return values, derivatives


class SpectralGalerkinMethodPlan(StrictModule, NonTrainableState):
    """Tensor-product Galerkin mass/stiffness assembly and dense reference solve."""

    maximum_dense_dimension: int = eqx.field(static=True)
    compatibility: Literal["error", "minimum_norm"] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_dense_dimension: int = 512,
        compatibility: Literal["error", "minimum_norm"] = "error",
    ):
        maximum = int(maximum_dense_dimension)
        if maximum <= 0:
            raise ValueError("maximum_dense_dimension must be positive.")
        if compatibility not in ("error", "minimum_norm"):
            raise ValueError("Unknown Galerkin compatibility policy.")
        self.maximum_dense_dimension = maximum
        self.compatibility = compatibility
        self.method_id = canonical_fingerprint(
            {
                "kind": "spectral-galerkin-method",
                "maximum_dense_dimension": maximum,
                "compatibility": compatibility,
            }
        )

    def prepare(
        self,
        discretization: TensorSpectralDiscretization,
        /,
    ) -> "PreparedSpectralGalerkin":
        return PreparedSpectralGalerkin(self, discretization)


class PreparedSpectralGalerkin(StrictModule, NonTrainableState):
    plan: SpectralGalerkinMethodPlan
    discretization: TensorSpectralDiscretization
    mass_matrices: tuple[Array, ...]
    stiffness_matrices: tuple[Array, ...]
    load_matrices: tuple[Array, ...]
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SpectralGalerkinMethodPlan,
        discretization: TensorSpectralDiscretization,
        /,
    ):
        if not isinstance(plan, SpectralGalerkinMethodPlan):
            raise TypeError("plan must be a SpectralGalerkinMethodPlan.")
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        masses = []
        stiffnesses = []
        loads = []
        for axis_index, axis in enumerate(discretization.axes):
            values, derivatives = _axis_basis_values(discretization, axis_index)
            weights = axis.quadrature_weights
            weighted_values = weights[:, None] * values
            weighted_derivatives = weights[:, None] * derivatives
            masses.append(jnp.conj(values.T) @ weighted_values)
            stiffnesses.append(jnp.conj(derivatives.T) @ weighted_derivatives)
            loads.append(jnp.conj(values.T) * weights[None, :])
        self.plan = plan
        self.discretization = discretization
        self.mass_matrices = tuple(masses)
        self.stiffness_matrices = tuple(stiffnesses)
        self.load_matrices = tuple(loads)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-galerkin",
                "plan": plan.method_id,
                "discretization": discretization.prepared_id,
            }
        )

    def mass_action(self, coefficients: ArrayLike, /) -> Array:
        result = self.discretization._validate_leading(
            coefficients,
            self.discretization.modal_shape,
            "Galerkin coefficients",
        )
        for axis, matrix in enumerate(self.mass_matrices):
            result = _apply_axis_matrix(result, matrix, axis)
        return result

    def stiffness_action(self, coefficients: ArrayLike, /) -> Array:
        value = self.discretization._validate_leading(
            coefficients,
            self.discretization.modal_shape,
            "Galerkin coefficients",
        )
        result = jnp.zeros_like(value)
        for derivative_axis in range(len(self.discretization.axes)):
            term = value
            for axis in range(len(self.discretization.axes)):
                matrix = (
                    self.stiffness_matrices[axis]
                    if axis == derivative_axis
                    else self.mass_matrices[axis]
                )
                term = _apply_axis_matrix(term, matrix, axis)
            result = result + term
        return result

    def load(self, values: ArrayLike, /) -> Array:
        result = self.discretization._validate_leading(
            values,
            self.discretization.physical_shape,
            "Galerkin load values",
        )
        for axis, matrix in enumerate(self.load_matrices):
            result = _apply_axis_matrix(result, matrix, axis)
        return result

    def stiffness_matrix(self) -> Array:
        count = self.discretization.num_modes
        if count > self.plan.maximum_dense_dimension:
            raise ValueError("Galerkin dense stiffness exceeds maximum_dense_dimension.")
        identity = jnp.eye(
            count,
            dtype=jnp.dtype(self.discretization.plan.precision.coefficient_dtype),
        )
        columns = jax.vmap(
            lambda vector: self.stiffness_action(
                vector.reshape(self.discretization.modal_shape)
            ).reshape((-1,))
        )(identity)
        return columns.T

    def solve_poisson(
        self,
        right_hand_side: ArrayLike,
        /,
    ) -> tuple[Array, LinearSolveResult]:
        load = self.load(right_hand_side).reshape((-1,))
        operator = DenseLinearOperator(self.stiffness_matrix())
        factorization = factorize(operator, FactorizationPolicy("svd"))
        result = factorization.solve(load)
        if self.plan.compatibility == "error" and not bool(result.successful):
            raise ValueError("Galerkin Poisson right-hand side is incompatible.")
        return result.value.reshape(self.discretization.modal_shape), result


__all__ = [
    "PreparedSpectralGalerkin",
    "SpectralGalerkinMethodPlan",
]
