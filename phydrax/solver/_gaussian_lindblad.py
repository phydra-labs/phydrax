#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import BosonicGaussianState, canonical_commutation_matrix


class GaussianLindbladProblem(StrictModule):
    drift: Array
    diffusion: Array
    forcing: Array
    initial_state: BosonicGaussianState
    generator_cp_margin: Array
    stability_margin: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        drift: ArrayLike,
        diffusion: ArrayLike,
        forcing: ArrayLike,
        initial_state: BosonicGaussianState,
        /,
        *,
        problem_id: str = "gaussian-lindblad",
    ):
        if not isinstance(initial_state, BosonicGaussianState):
            raise TypeError("initial_state must be BosonicGaussianState.")
        drift_ = jnp.asarray(drift, dtype=initial_state.mean.dtype)
        diffusion_ = jnp.asarray(diffusion, dtype=initial_state.mean.dtype)
        forcing_ = jnp.asarray(forcing, dtype=initial_state.mean.dtype)
        dimension = initial_state.mean.shape[0]
        if drift_.shape != (dimension, dimension) or diffusion_.shape != drift_.shape:
            raise ValueError("Gaussian drift/diffusion shapes are invalid.")
        if forcing_.shape != (dimension,):
            raise ValueError("Gaussian forcing shape is invalid.")
        self.drift = drift_
        self.diffusion = 0.5 * (diffusion_ + diffusion_.T)
        self.forcing = forcing_
        omega = canonical_commutation_matrix(initial_state.mode_count, dtype=drift_.dtype)
        generator_matrix = self.diffusion.astype(complex) - 0.5j * initial_state.hbar * (
            drift_ @ omega + omega @ drift_.T
        )
        generator_margin = jnp.min(jnp.linalg.eigvalsh(generator_matrix))
        if not bool(jax.device_get(generator_margin >= -1e-9)):
            raise ValueError(
                "Gaussian Lindblad drift/diffusion violate the CP generator condition."
            )
        self.generator_cp_margin = generator_margin
        self.stability_margin = -jnp.max(jnp.real(jnp.linalg.eigvals(drift_)))
        self.initial_state = initial_state
        self.problem_id = str(problem_id)

    def rhs(self, mean: Array, covariance: Array, /) -> tuple[Array, Array]:
        return (
            self.drift @ mean + self.forcing,
            self.drift @ covariance + covariance @ self.drift.T + self.diffusion,
        )

    def stationary_state(self) -> BosonicGaussianState:
        if not bool(jax.device_get(self.stability_margin > 0.0)):
            raise ValueError("Gaussian stationary state requires Hurwitz drift.")
        dimension = self.drift.shape[0]
        identity = jnp.eye(dimension, dtype=self.drift.dtype)
        operator = jnp.kron(identity, self.drift) + jnp.kron(self.drift, identity)
        covariance = jnp.linalg.solve(operator, -self.diffusion.reshape(-1)).reshape(
            self.diffusion.shape
        )
        mean = jnp.linalg.solve(self.drift, -self.forcing)
        state = BosonicGaussianState(mean, covariance, hbar=self.initial_state.hbar)
        residual = jnp.linalg.norm(
            self.drift @ covariance + covariance @ self.drift.T + self.diffusion
        )
        if not bool(jax.device_get(state.valid & (residual <= 1e-7))):
            raise ValueError("Gaussian stationary-state certification failed.")
        return state


class GaussianLindbladSolution(StrictModule):
    means: Array
    covariances: Array
    times: Array
    uncertainty_margins: Array
    valid: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        means: ArrayLike,
        covariances: ArrayLike,
        times: ArrayLike,
        /,
        *,
        problem_id: str,
        hbar: float,
    ):
        self.means = jnp.asarray(means)
        self.covariances = jnp.asarray(covariances)
        self.times = jnp.asarray(times)
        margins = jax.vmap(
            lambda mean, covariance: (
                BosonicGaussianState(mean, covariance, hbar=hbar).uncertainty_margin
            )
        )(self.means, self.covariances)
        self.uncertainty_margins = margins
        self.valid = (
            jnp.all(jnp.isfinite(self.means))
            & jnp.all(jnp.isfinite(self.covariances))
            & jnp.all(margins >= -1e-8)
        )
        self.problem_id = str(problem_id)


def solve_gaussian_lindblad(
    problem: GaussianLindbladProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
) -> GaussianLindbladSolution:
    count = int(steps)
    step = jnp.asarray(step_size, dtype=problem.drift.dtype).reshape(())
    if count < 0 or float(step) <= 0.0:
        raise ValueError("steps and step_size must be positive.")

    def advance(state, _):
        mean, covariance = state
        k1_mean, k1_covariance = problem.rhs(mean, covariance)
        k2_mean, k2_covariance = problem.rhs(
            mean + 0.5 * step * k1_mean,
            covariance + 0.5 * step * k1_covariance,
        )
        k3_mean, k3_covariance = problem.rhs(
            mean + 0.5 * step * k2_mean,
            covariance + 0.5 * step * k2_covariance,
        )
        k4_mean, k4_covariance = problem.rhs(
            mean + step * k3_mean,
            covariance + step * k3_covariance,
        )
        next_mean = mean + step * (k1_mean + 2 * k2_mean + 2 * k3_mean + k4_mean) / 6
        next_covariance = (
            covariance
            + step
            * (k1_covariance + 2 * k2_covariance + 2 * k3_covariance + k4_covariance)
            / 6
        )
        next_covariance = 0.5 * (next_covariance + next_covariance.T)
        return (next_mean, next_covariance), (next_mean, next_covariance)

    initial = (problem.initial_state.mean, problem.initial_state.covariance)
    _, trajectory = jax.lax.scan(advance, initial, xs=None, length=count)
    means = jnp.concatenate((initial[0][None, :], trajectory[0]), axis=0)
    covariances = jnp.concatenate((initial[1][None, :, :], trajectory[1]), axis=0)
    return GaussianLindbladSolution(
        means,
        covariances,
        step * jnp.arange(count + 1),
        problem_id=problem.problem_id,
        hbar=problem.initial_state.hbar,
    )


def damped_thermal_oscillator(
    damping: float,
    thermal_occupation: float,
    /,
) -> GaussianLindbladProblem:
    gamma = float(damping)
    occupation = float(thermal_occupation)
    if gamma <= 0.0 or occupation < 0.0:
        raise ValueError("Damping must be positive and occupation non-negative.")
    drift = -0.5 * gamma * jnp.eye(2)
    diffusion = gamma * (occupation + 0.5) * jnp.eye(2)
    initial = BosonicGaussianState(jnp.zeros(2), 0.5 * jnp.eye(2))
    return GaussianLindbladProblem(
        drift,
        diffusion,
        jnp.zeros(2),
        initial,
        problem_id="damped-thermal-oscillator",
    )


__all__ = [
    "GaussianLindbladProblem",
    "GaussianLindbladSolution",
    "damped_thermal_oscillator",
    "solve_gaussian_lindblad",
]
