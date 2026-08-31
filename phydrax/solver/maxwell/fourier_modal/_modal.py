#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import DenseLinearOperator, FailurePolicy
from ....linalg.eigen import (
    DenseSchurQZ,
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenSelection,
    GeneralEigenSolvePolicy,
)
from ._boundary_cascade import _transfer_to_boundary, BoundaryRelation
from ._factorization import _dense_solve
from ._layer import PreparedLayerOperator


class ModalPropagationPolicy(StrictModule, NonTrainableState):
    """Host dense full-spectrum reference propagation policy."""

    maximum_growth_exponent: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, *, maximum_growth_exponent: float = 60.0):
        maximum = float(maximum_growth_exponent)
        if maximum <= 0.0:
            raise ValueError("maximum_growth_exponent must be positive.")
        self.maximum_growth_exponent = maximum
        self.policy_id = canonical_fingerprint(
            {"kind": "modal-propagation-policy", "maximum_growth_exponent": maximum}
        )


class ModalLayerResult(StrictModule):
    eigenvalues: Array
    right_eigenvectors: Array
    left_eigenvectors: Array
    relative_residuals: Array
    condition_estimates: Array
    boundary: BoundaryRelation
    status: Array
    backend: str = eqx.field(static=True)


def prepare_modal_boundary(
    layer: PreparedLayerOperator,
    thickness: ArrayLike,
    policy: ModalPropagationPolicy | None = None,
    /,
) -> ModalLayerResult:
    """Build a transfer/boundary relation from the full host eigensystem.

    This is a small-problem qualification and modal-observable path. Gradients are
    deliberately stopped at the eigensystem.
    """
    policy_ = ModalPropagationPolicy() if policy is None else policy
    matrix = jax.lax.stop_gradient(layer.matrix)
    value = jax.lax.stop_gradient(jnp.asarray(thickness, dtype=matrix.dtype))
    problem = GeneralEigenproblem(DenseLinearOperator(matrix))
    eigen_policy = GeneralEigenSolvePolicy(
        DenseSchurQZ(),
        selection=GeneralEigenSelection.all(),
        failure=FailurePolicy("status"),
    )
    result = general_eigensolve(problem, policy=eigen_policy)
    eigenvalues = jax.lax.stop_gradient(result.eigenvalues)
    right = jax.lax.stop_gradient(result.right_eigenvector_coordinates)
    left = jax.lax.stop_gradient(result.left_eigenvector_coordinates)
    exponents = eigenvalues * value
    exponents = eqx.error_if(
        exponents,
        jnp.max(jnp.abs(jnp.real(exponents))) > policy_.maximum_growth_exponent,
        "Modal transfer propagation would overflow; use boundary cascade.",
    )
    phases = jnp.exp(exponents)
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    inverse_right = _dense_solve(right, identity)
    transfer = right @ (phases[:, None] * inverse_right)
    boundary = _transfer_to_boundary(transfer)
    return ModalLayerResult(
        eigenvalues,
        right,
        left,
        result.diagnostics.right_relative_residuals,
        result.diagnostics.eigenvalue_condition_estimates,
        boundary,
        result.status,
        backend="phydrax-general-dense-schur-qz",
    )


__all__ = [
    "ModalLayerResult",
    "ModalPropagationPolicy",
    "prepare_modal_boundary",
]
