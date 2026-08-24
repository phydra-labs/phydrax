#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import FermionicGaussianState


class FermionicGaussianProblem(StrictModule):
    drift: Array
    diffusion: Array
    initial_state: FermionicGaussianState
    problem_id: str

    def __init__(
        self,
        drift: ArrayLike,
        diffusion: ArrayLike,
        initial_state: FermionicGaussianState,
        /,
        *,
        problem_id: str = "fermionic-gaussian",
    ):
        drift_ = jnp.asarray(drift, dtype=float)
        diffusion_ = jnp.asarray(diffusion, dtype=float)
        if (
            drift_.shape != initial_state.covariance.shape
            or diffusion_.shape != drift_.shape
        ):
            raise ValueError("Fermionic drift/diffusion shapes are invalid.")
        self.drift = drift_
        self.diffusion = 0.5 * (diffusion_ - diffusion_.T)
        self.initial_state = initial_state
        self.problem_id = str(problem_id)

    def rhs(self, covariance: ArrayLike, /) -> Array:
        value = jnp.asarray(covariance)
        return self.drift @ value + value @ self.drift.T + self.diffusion

    def stationary_state(self) -> FermionicGaussianState:
        dimension = self.drift.shape[0]
        identity = jnp.eye(dimension)
        operator = jnp.kron(identity, self.drift) + jnp.kron(self.drift, identity)
        covariance = jnp.linalg.solve(operator, -self.diffusion.reshape(-1)).reshape(
            self.diffusion.shape
        )
        return FermionicGaussianState(covariance)


class FermionicGaussianSolution(StrictModule):
    covariances: Array
    times: Array
    physicality_margins: Array
    valid: Array
    problem_id: str

    def __init__(
        self,
        covariances: ArrayLike,
        times: ArrayLike,
        /,
        *,
        problem_id: str,
    ):
        values = jnp.asarray(covariances)
        self.covariances = values
        self.times = jnp.asarray(times)
        self.physicality_margins = jax.vmap(
            lambda covariance: FermionicGaussianState(covariance).physicality_margin
        )(values)
        self.valid = jnp.all(jnp.isfinite(values)) & jnp.all(
            self.physicality_margins >= -1e-8
        )
        self.problem_id = str(problem_id)


def solve_fermionic_gaussian(
    problem: FermionicGaussianProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
) -> FermionicGaussianSolution:
    step = jnp.asarray(step_size, dtype=float).reshape(())

    def advance(covariance, _):
        first = problem.rhs(covariance)
        second = problem.rhs(covariance + 0.5 * step * first)
        third = problem.rhs(covariance + 0.5 * step * second)
        fourth = problem.rhs(covariance + step * third)
        result = covariance + step * (first + 2 * second + 2 * third + fourth) / 6
        return 0.5 * (result - result.T), 0.5 * (result - result.T)

    _, trajectory = jax.lax.scan(
        advance, problem.initial_state.covariance, xs=None, length=int(steps)
    )
    values = jnp.concatenate((problem.initial_state.covariance[None, ...], trajectory))
    return FermionicGaussianSolution(
        values, step * jnp.arange(int(steps) + 1), problem_id=problem.problem_id
    )


def damped_fermionic_mode(
    damping: float, occupation: float, /
) -> FermionicGaussianProblem:
    gamma = float(damping)
    target = float(occupation)
    if gamma <= 0.0 or not 0.0 <= target <= 1.0:
        raise ValueError("Fermionic damping/occupation parameters are invalid.")
    drift = -0.5 * gamma * jnp.eye(2)
    target_covariance = jnp.asarray(
        [[0.0, 2.0 * target - 1.0], [1.0 - 2.0 * target, 0.0]]
    )
    diffusion = -drift @ target_covariance - target_covariance @ drift.T
    initial = FermionicGaussianState(jnp.asarray([[0.0, -1.0], [1.0, 0.0]]))
    return FermionicGaussianProblem(
        drift, diffusion, initial, problem_id="damped-fermionic-mode"
    )


def open_kitaev_chain(
    site_count: int,
    /,
    *,
    hopping: float = 1.0,
    pairing: float = 1.0,
    chemical_potential: float = 0.0,
    damping: float = 0.1,
    target_occupation: float = 0.5,
) -> FermionicGaussianProblem:
    count = int(site_count)
    if count < 1 or damping <= 0.0 or not 0.0 <= target_occupation <= 1.0:
        raise ValueError("Open Kitaev-chain parameters are invalid.")
    dimension = 2 * count
    generator = jnp.zeros((dimension, dimension))
    for site in range(count):
        left = 2 * site
        generator = generator.at[left, left + 1].set(chemical_potential)
        generator = generator.at[left + 1, left].set(-chemical_potential)
    for site in range(count - 1):
        current = 2 * site
        following = 2 * (site + 1)
        generator = generator.at[current + 1, following].set(hopping + pairing)
        generator = generator.at[following, current + 1].set(-(hopping + pairing))
        generator = generator.at[current, following + 1].set(hopping - pairing)
        generator = generator.at[following + 1, current].set(-(hopping - pairing))
    drift = generator - 0.5 * float(damping) * jnp.eye(dimension)
    block = jnp.asarray(
        [
            [0.0, 2.0 * target_occupation - 1.0],
            [1.0 - 2.0 * target_occupation, 0.0],
        ]
    )
    target = jnp.kron(jnp.eye(count), block)
    diffusion = -drift @ target - target @ drift.T
    vacuum = jnp.asarray([[0.0, -1.0], [1.0, 0.0]])
    initial = FermionicGaussianState(jnp.kron(jnp.eye(count), vacuum))
    return FermionicGaussianProblem(
        drift,
        diffusion,
        initial,
        problem_id=f"open-kitaev-chain:{count}",
    )


__all__ = [
    "FermionicGaussianProblem",
    "FermionicGaussianSolution",
    "damped_fermionic_mode",
    "open_kitaev_chain",
    "solve_fermionic_gaussian",
]
