#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import assert_never

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ...exterior._complex import ComplexBoundary
from ...linalg._complexes import (
    HarmonicSubspace,
    HodgeDecomposition,
    HodgeDecompositionPolicy,
)
from ...linalg._operators import FunctionLinearOperator
from ...linalg._spaces import ArraySpace, DualSpace
from ...typing import parse


def closed_boundary(boundary: ComplexBoundary, /) -> None:
    """Closed realizations have no relative boundary subcomplex."""
    policy = parse(boundary, ComplexBoundary, "boundary")
    match policy:
        case "absolute":
            return
        case "relative":
            raise ValueError("A closed spectral realization has no relative boundary.")
        case _:
            assert_never(policy)


def riesz_operator(space: ArraySpace, identifier: str, /) -> FunctionLinearOperator:
    return FunctionLinearOperator(
        space.riesz,
        source=space,
        target=DualSpace(space),
        transpose_action=space.riesz,
        operator_id=identifier,
    )


def harmonic_projection(
    space: ArraySpace,
    value: Array,
    canonical_basis: Array,
    harmonic: HarmonicSubspace | None,
    complex_id: str,
    degree: int,
    /,
) -> tuple[Array, Array]:
    """Admit a complete metric-orthonormal harmonic frame and use its projector."""
    basis = canonical_basis
    if harmonic is not None:
        if not isinstance(harmonic, HarmonicSubspace):
            raise TypeError("harmonic must be a HarmonicSubspace.")
        if harmonic.complex_id != complex_id or harmonic.degree != degree:
            raise ValueError(
                "Harmonic artifact belongs to a different complex or degree."
            )
        if harmonic.basis.shape != canonical_basis.shape:
            raise ValueError(
                "Harmonic artifact must contain the complete canonical kernel."
            )
        basis = harmonic.basis.astype(value.dtype)
        weights = space.riesz(jnp.ones_like(value))[:, None]
        gram = basis.conj().T @ (weights * basis)
        coefficients = canonical_basis.conj().T @ (weights * basis)
        remainder = basis - canonical_basis @ coefficients
        tolerance = 1024.0 * jnp.finfo(value.real.dtype).eps
        orthogonality = jnp.max(
            jnp.abs(gram - jnp.eye(basis.shape[1], dtype=value.dtype)), initial=0.0
        )
        residual = jnp.max(jnp.abs(remainder), initial=0.0)
        value = eqx.error_if(
            value,
            ~harmonic.valid
            | ~harmonic.dimension_match
            | ~jnp.all(jnp.isfinite(basis))
            | (orthogonality > tolerance)
            | (residual > tolerance),
            "Harmonic artifact must certify the complete orthonormal spectral kernel.",
        )
    weighted = space.riesz(value)
    return value, basis @ (basis.conj().T @ weighted)


def analytic_decomposition_policy(policy: HodgeDecompositionPolicy | None, /) -> None:
    if policy is not None:
        if not isinstance(policy, HodgeDecompositionPolicy):
            raise TypeError("policy must be a HodgeDecompositionPolicy or None.")
        raise ValueError(
            "Closed-form spectral decomposition has no iterative solve policy; use policy=None."
        )


def decomposition_result(
    space: ArraySpace,
    value: Array,
    potential: Array | None,
    exact: Array,
    coexact: Array,
    harmonic: Array,
    /,
) -> HodgeDecomposition:
    orthogonality = jnp.max(
        jnp.abs(
            jnp.stack(
                (
                    space.inner(exact, coexact),
                    space.inner(exact, harmonic),
                    space.inner(coexact, harmonic),
                )
            )
        )
    )
    remainder = value - exact - coexact - harmonic
    reconstruction = jnp.sqrt(
        jnp.maximum(jnp.real(space.inner(remainder, remainder)), 0.0)
    )
    scale = jnp.maximum(1.0, jnp.real(space.inner(value, value)))
    tolerance = 1024.0 * jnp.finfo(value.real.dtype).eps * scale
    valid = (
        jnp.all(jnp.isfinite(value))
        & (orthogonality <= tolerance)
        & (reconstruction <= tolerance)
    )
    return HodgeDecomposition(
        potential,
        exact,
        coexact,
        harmonic,
        orthogonality,
        reconstruction,
        jnp.asarray(0, dtype=jnp.int32),
        valid,
    )
