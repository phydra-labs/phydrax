#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Harmonic response, Craig-Bampton constraint modes, and SEA balances."""

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


def harmonic_response(
    stiffness: ArrayLike,
    mass: ArrayLike,
    damping: ArrayLike,
    force: ArrayLike,
    angular_frequency_rad_s: ArrayLike,
    /,
) -> Array:
    stiffness_ = jnp.asarray(stiffness)
    mass_ = jnp.asarray(mass)
    damping_ = jnp.asarray(damping)
    force_ = jnp.asarray(force)
    omega = jnp.asarray(angular_frequency_rad_s)
    dynamic = stiffness_.astype(jnp.complex128) - omega**2 * mass_ + 1j * omega * damping_
    space = ArraySpace((force_.size,), dtype=jnp.complex128)
    return solve(
        LinearSystem(DenseLinearOperator(dynamic, source=space, target=space)),
        force_.astype(jnp.complex128),
        policy=LinearSolvePolicy(DenseLU()),
    ).value


def craig_bampton_constraint_modes(
    interior_stiffness: ArrayLike, coupling_stiffness: ArrayLike, /
) -> Array:
    interior = jnp.asarray(interior_stiffness)
    coupling = jnp.asarray(coupling_stiffness)
    if (
        interior.ndim != 2
        or interior.shape[0] != interior.shape[1]
        or coupling.shape[0] != interior.shape[0]
    ):
        raise ValueError("Craig-Bampton interior/coupling matrices are incompatible.")
    space = ArraySpace((interior.shape[0],), dtype=interior.dtype)
    operator = DenseLinearOperator(interior, source=space, target=space)
    columns = tuple(
        solve(
            LinearSystem(operator),
            -coupling[:, index],
            policy=LinearSolvePolicy(DenseLU()),
        ).value
        for index in range(coupling.shape[1])
    )
    return jnp.stack(columns, axis=1)


def two_subsystem_sea_energy(
    input_power_w: ArrayLike,
    modal_frequencies_rad_s: ArrayLike,
    internal_loss_factors: ArrayLike,
    coupling_loss_factor: float,
    /,
) -> Array:
    power = jnp.asarray(input_power_w)
    omega = jnp.asarray(modal_frequencies_rad_s)
    internal = jnp.asarray(internal_loss_factors)
    coupling = float(coupling_loss_factor)
    matrix = jnp.asarray(
        (
            (omega[0] * (internal[0] + coupling), -omega[0] * coupling),
            (-omega[1] * coupling, omega[1] * (internal[1] + coupling)),
        )
    )
    space = ArraySpace((2,), dtype=matrix.dtype)
    return solve(
        LinearSystem(DenseLinearOperator(matrix, source=space, target=space)),
        power,
        policy=LinearSolvePolicy(DenseLU()),
    ).value


def structural_dynamics_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("structural-dynamics.harmonic", "direct-complex"),
        ("structural-dynamics.craig-bampton", "constraint-modes"),
        ("structural-dynamics.sea", "two-subsystem"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("analytic-control", "energy-balance", "public-workflow"),
        )
        for name, formulation in specs
    )


__all__ = [
    "craig_bampton_constraint_modes",
    "harmonic_response",
    "structural_dynamics_candidate_profiles",
    "two_subsystem_sea_energy",
]
