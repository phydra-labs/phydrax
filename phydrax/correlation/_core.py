#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frequency-response, modal correlation, and load reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import equinox as eqx
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


@dataclass(frozen=True, slots=True)
class ForceReconstructionResult:
    force: Array
    residual_norm: Array
    successful: Array


def modal_assurance_criterion(
    reference_modes: ArrayLike, candidate_modes: ArrayLike, /
) -> Array:
    reference = jnp.asarray(reference_modes)
    candidate = jnp.asarray(candidate_modes)
    if (
        reference.ndim != 2
        or candidate.ndim != 2
        or reference.shape[0] != candidate.shape[0]
    ):
        raise ValueError("MAC mode matrices must share a nonempty coordinate axis.")
    reference_norm = jnp.sum(jnp.abs(reference) ** 2, axis=0)
    candidate_norm = jnp.sum(jnp.abs(candidate) ** 2, axis=0)
    reference = eqx.error_if(
        reference,
        jnp.any(
            ~jnp.isfinite(reference)
            | ~jnp.isfinite(candidate)
            | (reference_norm <= 0)
            | (candidate_norm <= 0)
        ),
        "MAC modes must be finite and have positive norms.",
    )
    numerator = jnp.abs(jnp.conj(reference).T @ candidate) ** 2
    return numerator / (reference_norm[:, None] * candidate_norm[None, :])


def h1_frequency_response(
    cross_output_input: ArrayLike, auto_input: ArrayLike, /
) -> Array:
    cross = jnp.asarray(cross_output_input)
    auto = jnp.asarray(auto_input)
    cross, auto = jnp.broadcast_arrays(cross, auto)
    auto = eqx.error_if(
        auto,
        jnp.any(
            ~jnp.isfinite(cross)
            | ~jnp.isfinite(auto)
            | (jnp.real(auto) <= 0)
            | (jnp.abs(jnp.imag(auto)) > 1e-12)
        ),
        "H1 auto-spectrum must be finite, real, and strictly positive.",
    )
    return cross / auto


def reconstruct_force(
    response_matrix: ArrayLike, response: ArrayLike, regularization: float = 0.0, /
) -> ForceReconstructionResult:
    if not isfinite(regularization) or regularization < 0:
        raise ValueError(
            "Force-reconstruction regularization must be finite and nonnegative."
        )
    matrix = jnp.asarray(response_matrix)
    value = jnp.asarray(response)
    if matrix.ndim != 2 or value.shape != (matrix.shape[0],):
        raise ValueError("Force-reconstruction response data are incompatible.")
    matrix = eqx.error_if(
        matrix,
        jnp.any(~jnp.isfinite(matrix) | ~jnp.isfinite(value[:, None])),
        "Force-reconstruction response data must be finite.",
    )
    normal = jnp.conj(matrix).T @ matrix + float(regularization) * jnp.eye(
        matrix.shape[1], dtype=matrix.dtype
    )
    right = jnp.conj(matrix).T @ value
    space = ArraySpace((matrix.shape[1],), dtype=normal.dtype)
    solved = solve(
        LinearSystem(DenseLinearOperator(normal, source=space, target=space)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    )
    residual_norm = jnp.linalg.norm(matrix @ solved.value - value)
    successful = (
        solved.successful
        & jnp.all(jnp.isfinite(solved.value))
        & jnp.isfinite(residual_norm)
    )
    return ForceReconstructionResult(solved.value, residual_norm, successful)


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
    "ForceReconstructionResult",
    "h1_frequency_response",
    "modal_assurance_criterion",
    "reconstruct_force",
]
