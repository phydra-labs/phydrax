#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from .._strict import StrictModule
from ..tensor_network import (
    LocallyPurifiedDensity,
    lpdo_one_site_reduced,
    NearestNeighborHamiltonian,
)
from ._purified_lindblad import LocalKrausChannel
from ._purified_tebd import (
    diagnose_purified_stationarity,
    PurifiedStationarityDiagnostic,
    PurifiedStrangProblem,
    solve_purified_strang,
)


def _generalized_amplitude_kraus(probability_ground: float, damping: float):
    p = float(probability_ground)
    gamma = float(damping)
    if not 0.0 <= p <= 1.0 or not 0.0 <= gamma <= 1.0:
        raise ValueError("Generalized amplitude-damping parameters are invalid.")
    root = jnp.sqrt(1.0 - gamma)
    jump = jnp.sqrt(gamma)
    return jnp.asarray(
        [
            jnp.sqrt(p) * jnp.asarray([[1.0, 0.0], [0.0, root]]),
            jnp.sqrt(p) * jnp.asarray([[0.0, jump], [0.0, 0.0]]),
            jnp.sqrt(1.0 - p) * jnp.asarray([[root, 0.0], [0.0, 1.0]]),
            jnp.sqrt(1.0 - p) * jnp.asarray([[0.0, 0.0], [jump, 0.0]]),
        ],
        dtype=complex,
    )


def boundary_driven_xxz_problem(
    site_count: int,
    /,
    *,
    coupling: float = 1.0,
    anisotropy: float = 1.0,
    left_polarization: float = 0.5,
    right_polarization: float = -0.5,
    boundary_rate: float = 1.0,
    half_step: float = 0.01,
) -> PurifiedStrangProblem:
    count = int(site_count)
    if count < 2 or abs(left_polarization) > 1.0 or abs(right_polarization) > 1.0:
        raise ValueError("XXZ site count or boundary polarization is invalid.")
    sigma_x = jnp.asarray([[0, 1], [1, 0]], dtype=complex)
    sigma_y = jnp.asarray([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = jnp.asarray([[1, 0], [0, -1]], dtype=complex)
    term = float(coupling) * (
        jnp.kron(sigma_x, sigma_x)
        + jnp.kron(sigma_y, sigma_y)
        + float(anisotropy) * jnp.kron(sigma_z, sigma_z)
    )
    hamiltonian = NearestNeighborHamiltonian(
        tuple(term for _ in range(count - 1)),
        tuple(2 for _ in range(count)),
        hamiltonian_id=f"boundary-xxz:{count}",
    )
    local_factor = jnp.zeros((1, 2, 2, 1), dtype=complex)
    local_factor = local_factor.at[0, 0, 0, 0].set(1.0 / jnp.sqrt(2.0))
    local_factor = local_factor.at[0, 1, 1, 0].set(1.0 / jnp.sqrt(2.0))
    initial = LocallyPurifiedDensity(tuple(local_factor for _ in range(count)))
    damping = 1.0 - jnp.exp(-float(boundary_rate) * float(half_step))
    left_ground = 0.5 * (1.0 - float(left_polarization))
    right_ground = 0.5 * (1.0 - float(right_polarization))
    channels = (
        LocalKrausChannel(
            0,
            _generalized_amplitude_kraus(left_ground, damping),
            channel_id="xxz-left-reservoir-half-step",
        ),
        LocalKrausChannel(
            count - 1,
            _generalized_amplitude_kraus(right_ground, damping),
            channel_id="xxz-right-reservoir-half-step",
        ),
    )
    return PurifiedStrangProblem(
        initial,
        hamiltonian,
        channels,
        problem_id=f"boundary-driven-xxz:{count}",
    )


class XXZQualificationResult(StrictModule):
    final_result: object
    magnetization_history: jnp.ndarray
    diagnostic: PurifiedStationarityDiagnostic
    valid: jnp.ndarray

    def __init__(
        self,
        final_result,
        magnetization_history,
        diagnostic: PurifiedStationarityDiagnostic,
        /,
    ):
        self.final_result = final_result
        self.magnetization_history = jnp.asarray(magnetization_history)
        self.diagnostic = diagnostic
        self.valid = diagnostic.valid


def qualify_boundary_driven_xxz(
    problem: PurifiedStrangProblem,
    /,
    *,
    step_size: float,
    steps: int,
    maximum_bond_dimension: int,
    maximum_purification_dimension: int,
    steady_window: int = 4,
) -> XXZQualificationResult:
    sigma_z = jnp.asarray([[1, 0], [0, -1]], dtype=complex)
    current_problem = problem
    magnetization = []
    final_result = None
    for _ in range(int(steps)):
        final_result = solve_purified_strang(
            current_problem,
            step_size=step_size,
            steps=1,
            maximum_bond_dimension=maximum_bond_dimension,
            maximum_purification_dimension=maximum_purification_dimension,
        )
        local = jnp.stack(
            [
                jnp.real(
                    jnp.trace(
                        lpdo_one_site_reduced(final_result.final_state, site) @ sigma_z
                    )
                )
                for site in range(final_result.final_state.site_count)
            ]
        )
        magnetization.append(local)
        current_problem = PurifiedStrangProblem(
            final_result.final_state,
            problem.hamiltonian,
            problem.half_step_channels,
            problem_id=problem.problem_id,
        )
    if final_result is None:
        raise ValueError("XXZ qualification requires at least one step.")
    history = jnp.stack(magnetization)
    diagnostic = diagnose_purified_stationarity(
        final_result,
        history,
        window=min(int(steady_window), max(1, history.shape[0] - 1)),
        tolerance=1e-4,
        truncation_tolerance=1e-6,
    )
    return XXZQualificationResult(final_result, history, diagnostic)


__all__ = [
    "XXZQualificationResult",
    "boundary_driven_xxz_problem",
    "qualify_boundary_driven_xxz",
]
