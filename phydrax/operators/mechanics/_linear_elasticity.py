#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein
import phydrax.linalg as la

from ..._strict import StrictModule


def _mandel_basis(dimension: int, dtype) -> Array:
    values = []
    for row in range(dimension):
        basis = np.zeros((dimension, dimension), dtype=np.float64)
        basis[row, row] = 1.0
        values.append(basis)
    scale = 1.0 / np.sqrt(2.0)
    for row in range(dimension):
        for column in range(row + 1, dimension):
            basis = np.zeros((dimension, dimension), dtype=np.float64)
            basis[row, column] = scale
            basis[column, row] = scale
            values.append(basis)
    return jnp.asarray(np.stack(values), dtype=dtype)


class LinearElasticityTensor(StrictModule):
    """Dimension-declared stiffness tensor with symmetry and coercivity evidence."""

    stiffness: Array
    mandel_evidence: la.DensePropertyEvidence
    minor_symmetry_defect: Array
    major_symmetry_defect: Array
    finite: Array
    successful: Array
    dimension: int = eqx.field(static=True)

    def __init__(self, stiffness: ArrayLike, /):
        value = jnp.asarray(stiffness)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(jnp.float64)
        if value.ndim != 4 or len(set(value.shape)) != 1 or value.shape[0] <= 0:
            raise ValueError("stiffness must have shape (dimension,)*4.")
        dimension = value.shape[0]
        basis = _mandel_basis(dimension, value.dtype)
        mandel = ein.contract("aij,ijkl,bkl->ab", basis, value, basis)
        evidence = la.verify_dense_properties(
            mandel,
            policy=la.DensePropertyVerificationPolicy(
                require_hermitian=True,
                require_positive_definite=True,
            ),
        )
        scale = jnp.maximum(jnp.max(jnp.abs(value)), 1.0)
        tolerance = 128.0 * jnp.finfo(jnp.real(value).dtype).eps * scale
        minor_first = jnp.max(jnp.abs(value - jnp.swapaxes(value, 0, 1)))
        minor_second = jnp.max(jnp.abs(value - jnp.swapaxes(value, 2, 3)))
        major = jnp.max(jnp.abs(value - jnp.transpose(value, (2, 3, 0, 1))))
        minor = jnp.maximum(minor_first, minor_second)
        finite = jnp.all(jnp.isfinite(value))
        successful = (
            finite & (minor <= tolerance) & (major <= tolerance) & evidence.successful
        )
        self.stiffness = value
        self.mandel_evidence = evidence
        self.minor_symmetry_defect = minor
        self.major_symmetry_defect = major
        self.finite = finite
        self.successful = successful
        self.dimension = dimension

    @classmethod
    def isotropic(
        cls,
        dimension: int,
        /,
        *,
        lame_lambda: ArrayLike,
        shear_modulus: ArrayLike,
    ) -> LinearElasticityTensor:
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        lam = jnp.asarray(lame_lambda, dtype=jnp.float64).reshape(())
        mu = jnp.asarray(shear_modulus, dtype=jnp.float64).reshape(())
        identity = jnp.eye(dimension, dtype=jnp.float64)
        stiffness = lam * ein.contract("ij,kl->ijkl", identity, identity) + mu * (
            ein.contract("ik,jl->ijkl", identity, identity)
            + ein.contract("il,jk->ijkl", identity, identity)
        )
        return cls(stiffness)

    def stress(self, strain: ArrayLike, /) -> Array:
        value = jnp.asarray(strain, dtype=self.stiffness.dtype)
        if value.shape[-2:] != (self.dimension, self.dimension):
            raise ValueError(
                f"strain must end in shape ({self.dimension}, {self.dimension})."
            )
        checked = eqx.error_if(
            value,
            ~self.successful,
            "Linear elasticity tensor failed symmetry or coercivity verification.",
        )
        return ein.contract("ijkl,...kl->...ij", self.stiffness, checked)


__all__ = ["LinearElasticityTensor"]
