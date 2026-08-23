#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import HermitianSpectrum


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


class LindbladProblem(StrictModule):
    hamiltonian: Array
    jump_operators: Array
    initial_density: Array
    dimension: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: ArrayLike,
        jump_operators: ArrayLike,
        initial_density: ArrayLike,
        /,
        *,
        problem_id: str = "lindblad",
    ):
        hamiltonian_ = jnp.asarray(hamiltonian)
        jumps = jnp.asarray(jump_operators)
        density = jnp.asarray(initial_density)
        if hamiltonian_.ndim != 2 or hamiltonian_.shape[0] != hamiltonian_.shape[1]:
            raise ValueError("Hamiltonian must be square.")
        dimension = hamiltonian_.shape[0]
        if jumps.ndim != 3 or jumps.shape[1:] != (dimension, dimension):
            raise ValueError("Jump operators must have shape (count, n, n).")
        if density.shape != (dimension, dimension):
            raise ValueError("Initial density shape does not match the Hamiltonian.")
        if jnp.max(jnp.abs(hamiltonian_ - _adjoint(hamiltonian_))) > 1e-9:
            raise ValueError("Hamiltonian must be Hermitian.")
        spectrum = HermitianSpectrum(density, tolerance=1e-10)
        trace_residual = jnp.abs(jnp.trace(density) - 1.0)
        if not bool(
            jax.device_get(
                spectrum.valid
                & (trace_residual <= 1e-10)
                & (spectrum.minimum_eigenvalue >= -1e-10)
            )
        ):
            raise ValueError(
                "Initial density must be Hermitian, positive semidefinite, and trace one."
            )
        self.hamiltonian = hamiltonian_
        self.jump_operators = jumps
        self.initial_density = density
        self.dimension = dimension
        self.problem_id = str(problem_id)

    def generator(self, density: ArrayLike, /) -> Array:
        rho = jnp.asarray(density)
        commutator = -1j * (self.hamiltonian @ rho - rho @ self.hamiltonian)
        dissipator = jnp.zeros_like(rho)
        for jump in self.jump_operators:
            product = _adjoint(jump) @ jump
            dissipator = (
                dissipator
                + jump @ rho @ _adjoint(jump)
                - 0.5 * (product @ rho + rho @ product)
            )
        return commutator + dissipator

    def generator_matrix(self) -> Array:
        size = self.dimension**2
        basis = jnp.eye(size, dtype=self.initial_density.dtype).reshape(
            (size, self.dimension, self.dimension)
        )
        columns = jax.vmap(lambda matrix: self.generator(matrix).reshape(-1))(basis)
        return jnp.swapaxes(columns, -1, -2)


class LindbladSolution(StrictModule):
    states: Array
    times: Array
    trace_residuals: Array
    hermiticity_residuals: Array
    minimum_eigenvalues: Array
    valid: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        states: ArrayLike,
        times: ArrayLike,
        /,
        *,
        problem_id: str,
    ):
        values = jnp.asarray(states)
        self.states = values
        self.times = jnp.asarray(times)
        self.trace_residuals = jnp.abs(jnp.trace(values, axis1=-2, axis2=-1) - 1.0)
        self.hermiticity_residuals = jnp.max(
            jnp.abs(values - _adjoint(values)), axis=(-2, -1)
        )
        self.minimum_eigenvalues = jnp.min(
            jnp.linalg.eigvalsh(0.5 * (values + _adjoint(values))), axis=-1
        )
        self.valid = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(self.trace_residuals <= 1e-8)
            & jnp.all(self.hermiticity_residuals <= 1e-8)
            & jnp.all(self.minimum_eigenvalues >= -1e-8)
        )
        self.problem_id = str(problem_id)


def solve_lindblad(
    problem: LindbladProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
) -> LindbladSolution:
    if not isinstance(problem, LindbladProblem):
        raise TypeError("problem must be a LindbladProblem.")
    count = int(steps)
    step = jnp.asarray(step_size, dtype=float).reshape(())
    if count < 0 or float(step) <= 0.0:
        raise ValueError("steps and step_size must be positive.")
    channel = jsp.linalg.expm(step * problem.generator_matrix())

    def advance(state, _):
        next_state = (channel @ state.reshape(-1)).reshape(state.shape)
        next_state = 0.5 * (next_state + _adjoint(next_state))
        return next_state, next_state

    _, trajectory = jax.lax.scan(advance, problem.initial_density, xs=None, length=count)
    states = jnp.concatenate((problem.initial_density[None, ...], trajectory), axis=0)
    times = step * jnp.arange(count + 1)
    return LindbladSolution(states, times, problem_id=problem.problem_id)


def amplitude_damping_problem(
    damping_rate: float,
    initial_density: ArrayLike,
    /,
) -> LindbladProblem:
    rate = float(damping_rate)
    if rate < 0.0:
        raise ValueError("damping_rate must be non-negative.")
    lowering = jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    return LindbladProblem(
        jnp.zeros((2, 2), dtype=complex),
        jnp.sqrt(rate) * lowering[None, ...],
        initial_density,
        problem_id="qubit-amplitude-damping",
    )


def dephasing_problem(
    dephasing_rate: float,
    initial_density: ArrayLike,
    /,
) -> LindbladProblem:
    rate = float(dephasing_rate)
    if rate < 0.0:
        raise ValueError("dephasing_rate must be non-negative.")
    sigma_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return LindbladProblem(
        jnp.zeros((2, 2), dtype=complex),
        jnp.sqrt(0.5 * rate) * sigma_z[None, ...],
        initial_density,
        problem_id="qubit-dephasing",
    )


__all__ = [
    "LindbladProblem",
    "LindbladSolution",
    "amplitude_damping_problem",
    "dephasing_problem",
    "solve_lindblad",
]
