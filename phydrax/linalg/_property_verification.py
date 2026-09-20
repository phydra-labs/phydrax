#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._properties import OperatorProperties


class DensePropertyVerificationPolicy(StrictModule):
    require_hermitian: bool = eqx.field(static=True)
    require_positive_semidefinite: bool = eqx.field(static=True)
    require_positive_definite: bool = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        require_hermitian: bool = True,
        require_positive_semidefinite: bool = False,
        require_positive_definite: bool = False,
        relative_tolerance: float = 64.0,
        absolute_tolerance: float = 0.0,
    ):
        if require_positive_definite:
            require_positive_semidefinite = True
        if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
            raise ValueError("Dense property tolerances must be non-negative.")
        self.require_hermitian = bool(require_hermitian)
        self.require_positive_semidefinite = bool(require_positive_semidefinite)
        self.require_positive_definite = bool(require_positive_definite)
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)


class DensePropertyEvidence(StrictModule):
    matrix: Array
    eigenvalues: Array
    hermitian_defect: Array
    tolerance: Array
    numerical_rank: Array
    condition_estimate: Array
    finite: Array
    hermitian: Array
    positive_semidefinite: Array
    positive_definite: Array
    successful: Array
    properties: OperatorProperties = eqx.field(static=True)


def verify_dense_properties(
    matrix: ArrayLike,
    /,
    *,
    policy: DensePropertyVerificationPolicy | None = None,
) -> DensePropertyEvidence:
    """Verify finite Hermitian rank and definiteness with one eigendecomposition."""
    value = jnp.asarray(matrix)
    if value.ndim < 2 or value.shape[-1] != value.shape[-2]:
        raise ValueError("Dense property verification requires square matrices.")
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype("float64")
    policy_ = DensePropertyVerificationPolicy() if policy is None else policy
    if not isinstance(policy_, DensePropertyVerificationPolicy):
        raise TypeError("policy must be DensePropertyVerificationPolicy or None.")
    adjoint = jnp.swapaxes(jnp.conj(value), -1, -2)
    defect = jnp.max(jnp.abs(value - adjoint), axis=(-2, -1), initial=0.0)
    symmetric = 0.5 * (value + adjoint)
    eigenvalues = jnp.linalg.eigvalsh(symmetric)
    scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues), axis=-1), 1.0)
    tolerance = policy_.absolute_tolerance + (
        policy_.relative_tolerance * jnp.finfo(jnp.real(value).dtype).eps * scale
    )
    rank = jnp.sum(jnp.abs(eigenvalues) > tolerance[..., None], axis=-1)
    maximum = jnp.max(jnp.abs(eigenvalues), axis=-1)
    minimum_nonzero = jnp.min(
        jnp.where(
            jnp.abs(eigenvalues) > tolerance[..., None], jnp.abs(eigenvalues), jnp.inf
        ),
        axis=-1,
    )
    condition = maximum / minimum_nonzero
    finite = jnp.all(jnp.isfinite(value), axis=(-2, -1))
    hermitian = defect <= tolerance
    minimum = jnp.min(eigenvalues, axis=-1)
    psd = hermitian & (minimum >= -tolerance)
    pd = hermitian & (minimum > tolerance)
    successful = finite
    if policy_.require_hermitian:
        successful = successful & hermitian
    if policy_.require_positive_semidefinite:
        successful = successful & psd
    if policy_.require_positive_definite:
        successful = successful & pd
    properties = OperatorProperties(
        self_adjoint=policy_.require_hermitian,
        positive_semidefinite=policy_.require_positive_semidefinite,
        positive_definite=policy_.require_positive_definite,
        evidence={
            name: "verified"
            for name, enabled in (
                ("self_adjoint", policy_.require_hermitian),
                ("positive_semidefinite", policy_.require_positive_semidefinite),
                ("positive_definite", policy_.require_positive_definite),
            )
            if enabled
        },
    )
    return DensePropertyEvidence(
        symmetric,
        eigenvalues,
        defect,
        tolerance,
        rank,
        condition,
        finite,
        hermitian,
        psd,
        pd,
        successful,
        properties,
    )


__all__ = [
    "DensePropertyEvidence",
    "DensePropertyVerificationPolicy",
    "verify_dense_properties",
]
