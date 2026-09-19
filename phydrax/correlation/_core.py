#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frequency-response, modal correlation, and load reconstruction."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)
from ..qualification import CapabilityProfile, SupportTuple


def modal_assurance_criterion(
    reference_modes: ArrayLike, candidate_modes: ArrayLike, /
) -> Array:
    reference = jnp.asarray(reference_modes)
    candidate = jnp.asarray(candidate_modes)
    numerator = jnp.abs(jnp.conj(reference).T @ candidate) ** 2
    reference_norm = jnp.sum(jnp.abs(reference) ** 2, axis=0)
    candidate_norm = jnp.sum(jnp.abs(candidate) ** 2, axis=0)
    return numerator / jnp.maximum(
        reference_norm[:, None] * candidate_norm[None, :], 1.0e-30
    )


def h1_frequency_response(
    cross_output_input: ArrayLike, auto_input: ArrayLike, /
) -> Array:
    return jnp.asarray(cross_output_input) / jnp.maximum(jnp.asarray(auto_input), 1.0e-30)


def reconstruct_force(
    response_matrix: ArrayLike, response: ArrayLike, regularization: float = 0.0, /
) -> Array:
    matrix = jnp.asarray(response_matrix)
    value = jnp.asarray(response)
    normal = jnp.conj(matrix).T @ matrix + float(regularization) * jnp.eye(
        matrix.shape[1], dtype=matrix.dtype
    )
    right = jnp.conj(matrix).T @ value
    space = ArraySpace((matrix.shape[1],), dtype=normal.dtype)
    return solve(
        LinearSystem(DenseLinearOperator(normal, source=space, target=space)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    ).value


def correlation_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("correlation.modal-assurance", "complex-mac"),
        ("correlation.h1-frf", "cross-over-auto"),
        ("correlation.force-reconstruction", "tikhonov-normal-system"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("analytic-control", "conditioning", "public-workflow"),
        )
        for name, formulation in specs
    )


__all__ = [
    "correlation_candidate_profiles",
    "h1_frequency_response",
    "modal_assurance_criterion",
    "reconstruct_force",
]
