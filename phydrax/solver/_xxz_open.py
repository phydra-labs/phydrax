#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from ..tensor_network import LocallyPurifiedDensity, NearestNeighborHamiltonian
from ._purified_lindblad import LocalKrausChannel
from ._purified_tebd import PurifiedStrangProblem


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


__all__ = ["boundary_driven_xxz_problem"]
