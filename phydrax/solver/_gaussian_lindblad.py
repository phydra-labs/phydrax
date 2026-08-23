#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._temporal_precision import TemporalPrecisionPolicy
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve as solve_linear,
)
from ..metrix import BosonicGaussianState, canonical_commutation_matrix


class GaussianLindbladProblem(StrictModule):
    drift: Array
    diffusion: Array
    forcing: Array
    initial_state: BosonicGaussianState
    generator_cp_margin: Array
    stability_margin: Array
    linear: LinearSolvePolicy
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        drift: ArrayLike,
        diffusion: ArrayLike,
        forcing: ArrayLike,
        initial_state: BosonicGaussianState,
        /,
        *,
        linear: LinearSolvePolicy | None = None,
        problem_id: str = "gaussian-lindblad",
    ):
        if not isinstance(initial_state, BosonicGaussianState):
            raise TypeError("initial_state must be BosonicGaussianState.")
        linear_ = LinearSolvePolicy(DenseLU()) if linear is None else linear
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
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
        self.linear = linear_
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
        covariance = solve_linear(
            LinearSystem(DenseLinearOperator(operator)),
            -self.diffusion.reshape(-1),
            policy=self.linear,
        ).value.reshape(self.diffusion.shape)
        mean = solve_linear(
            LinearSystem(DenseLinearOperator(self.drift)),
            -self.forcing,
            policy=self.linear,
        ).value
        state = BosonicGaussianState(
            mean,
            covariance,
            hbar=self.initial_state.hbar,
            geometry_precision=self.initial_state.geometry_precision,
            hermitian_precision=self.initial_state.hermitian_precision,
        )
        residual = self.initial_state.geometry_precision.norm(
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
    precision: TemporalPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
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
        precision: TemporalPrecisionPolicy,
        geometry_precision,
        hermitian_precision,
    ):
        if not isinstance(precision, TemporalPrecisionPolicy):
            raise TypeError("precision must be TemporalPrecisionPolicy.")
        means_ = jnp.asarray(means)
        covariances_ = jnp.asarray(covariances)
        times_ = jnp.asarray(times)
        precision.validate_state(means_[0])
        margins = jax.vmap(
            lambda mean, covariance: (
                BosonicGaussianState(
                    mean,
                    covariance,
                    hbar=hbar,
                    geometry_precision=geometry_precision,
                    hermitian_precision=hermitian_precision,
                ).uncertainty_margin
            )
        )(means_, covariances_)
        final_state = BosonicGaussianState(
            means_[-1],
            covariances_[-1],
            hbar=hbar,
            geometry_precision=geometry_precision,
            hermitian_precision=hermitian_precision,
        )
        self.means = precision.output(means_)
        self.covariances = precision.output(covariances_)
        self.times = times_
        self.uncertainty_margins = precision.decision(margins)
        self.valid = (
            jnp.all(jnp.isfinite(self.means))
            & jnp.all(jnp.isfinite(self.covariances))
            & jnp.all(self.uncertainty_margins >= -1e-8)
            & final_state.valid
        )
        self.precision = precision
        self.precision_evidence = precision.evidence_for(
            means_[0],
            times_[0],
            children={
                "final-gaussian-state": final_state.precision_evidence,
            },
        )
        self.problem_id = str(problem_id)


def solve_gaussian_lindblad(
    problem: GaussianLindbladProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    precision: TemporalPrecisionPolicy | None = None,
) -> GaussianLindbladSolution:
    if not isinstance(problem, GaussianLindbladProblem):
        raise TypeError("problem must be GaussianLindbladProblem.")
    precision_ = TemporalPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, TemporalPrecisionPolicy):
        raise TypeError("precision must be TemporalPrecisionPolicy or None.")
    precision_.validate_state(problem.initial_state.mean)
    count = int(steps)
    step = precision_.coefficient(
        jnp.asarray(step_size, dtype=problem.drift.real.dtype)
    ).reshape(())
    if count < 0 or float(step) <= 0.0:
        raise ValueError("steps and step_size must be positive.")

    def advance(state, _):
        mean, covariance = state
        k1_mean, k1_covariance = jax.tree.map(
            precision_.stage,
            problem.rhs(mean, covariance),
        )
        k2_mean, k2_covariance = jax.tree.map(
            precision_.stage,
            problem.rhs(
                mean + 0.5 * step * k1_mean,
                covariance + 0.5 * step * k1_covariance,
            ),
        )
        k3_mean, k3_covariance = jax.tree.map(
            precision_.stage,
            problem.rhs(
                mean + 0.5 * step * k2_mean,
                covariance + 0.5 * step * k2_covariance,
            ),
        )
        k4_mean, k4_covariance = jax.tree.map(
            precision_.stage,
            problem.rhs(
                mean + step * k3_mean,
                covariance + step * k3_covariance,
            ),
        )
        mean_increment = precision_.accumulation(
            k1_mean + 2 * k2_mean + 2 * k3_mean + k4_mean
        )
        covariance_increment = precision_.accumulation(
            k1_covariance + 2 * k2_covariance + 2 * k3_covariance + k4_covariance
        )
        next_mean = jnp.asarray(
            mean + step * mean_increment / 6,
            dtype=mean.dtype,
        )
        next_covariance = jnp.asarray(
            covariance + step * covariance_increment / 6,
            dtype=covariance.dtype,
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
        precision=precision_,
        geometry_precision=problem.initial_state.geometry_precision,
        hermitian_precision=problem.initial_state.hermitian_precision,
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
