#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..operators.quantum import OpenSystemPhysicalityEvidence


class QuantumMemoryKernel(StrictModule):
    action_function: Callable[[Array, Array], Array]
    dimension: int = eqx.field(static=True)
    memory_horizon: float = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: Callable[[Array, Array], Array],
        dimension: int,
        /,
        *,
        memory_horizon: float,
        kernel_id: str,
    ):
        if not callable(action):
            raise TypeError("Memory-kernel action must be callable.")
        if memory_horizon <= 0.0:
            raise ValueError("memory_horizon must be positive.")
        self.action_function = action
        self.dimension = int(dimension)
        self.memory_horizon = float(memory_horizon)
        self.kernel_id = str(kernel_id)

    def __call__(self, lag: ArrayLike, density: ArrayLike, /) -> Array:
        rho = jnp.asarray(density)
        if rho.shape != (self.dimension, self.dimension):
            raise ValueError("Memory-kernel density shape is invalid.")
        result = jnp.asarray(self.action_function(jnp.asarray(lag), rho))
        if result.shape != rho.shape:
            raise ValueError("Memory-kernel action must preserve density shape.")
        return jnp.where(jnp.asarray(lag) <= self.memory_horizon, result, 0.0)


class MemoryKernelMasterEquation(StrictModule):
    local_generator: Callable[[Array, Array], Array]
    kernel: QuantumMemoryKernel
    initial_density: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_generator: Callable[[Array, Array], Array],
        kernel: QuantumMemoryKernel,
        initial_density: ArrayLike,
        /,
        *,
        problem_id: str = "memory-kernel-master",
    ):
        if not callable(local_generator):
            raise TypeError("local_generator must be callable.")
        density = jnp.asarray(initial_density)
        if density.shape != (kernel.dimension, kernel.dimension):
            raise ValueError("Initial density shape does not match the memory kernel.")
        self.local_generator = local_generator
        self.kernel = kernel
        self.initial_density = density
        self.problem_id = str(problem_id)


class TimeLocalOpenSystemProblem(StrictModule):
    generator: Callable[[Array, Array], Array]
    initial_density: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        generator: Callable[[Array, Array], Array],
        initial_density: ArrayLike,
        /,
        *,
        problem_id: str = "time-local-open-system",
    ):
        if not callable(generator):
            raise TypeError("generator must be callable.")
        density = jnp.asarray(initial_density)
        if density.ndim != 2 or density.shape[0] != density.shape[1]:
            raise ValueError("Initial density must be square.")
        self.generator = generator
        self.initial_density = density
        self.problem_id = str(problem_id)


class OpenSystemHistorySolution(StrictModule):
    states: Array
    times: Array
    trace_residuals: Array
    hermiticity_residuals: Array
    minimum_eigenvalues: Array
    execution_valid: Array
    pointwise_density_valid: Array
    physicality: OpenSystemPhysicalityEvidence
    valid: Array
    production_valid: Array
    problem_id: str = eqx.field(static=True)

    def __init__(self, states: ArrayLike, times: ArrayLike, /, *, problem_id: str):
        values = jnp.asarray(states)
        self.states = values
        self.times = jnp.asarray(times)
        self.trace_residuals = jnp.abs(jnp.trace(values, axis1=-2, axis2=-1) - 1.0)
        self.hermiticity_residuals = jnp.max(
            jnp.abs(values - jnp.swapaxes(jnp.conj(values), -1, -2)), axis=(-2, -1)
        )
        self.minimum_eigenvalues = jnp.min(
            jnp.linalg.eigvalsh(0.5 * (values + jnp.swapaxes(jnp.conj(values), -1, -2))),
            axis=-1,
        )
        self.execution_valid = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(self.trace_residuals <= 1e-6)
            & jnp.all(self.hermiticity_residuals <= 1e-6)
        )
        self.pointwise_density_valid = self.execution_valid & jnp.all(
            self.minimum_eigenvalues >= -1e-8
        )
        self.physicality = OpenSystemPhysicalityEvidence(
            trace_residual=jnp.max(self.trace_residuals),
            hermiticity_residual=jnp.max(self.hermiticity_residuals),
            positivity_margin=jnp.min(self.minimum_eigenvalues),
            status="unknown",
        )
        self.valid = self.execution_valid & self.pointwise_density_valid
        self.production_valid = self.valid & self.physicality.valid
        self.problem_id = str(problem_id)


class DynamicalMapPhysicality(StrictModule):
    choi_matrix: Array
    cp_margin: Array
    trace_preservation_residual: Array
    valid: Array

    def __init__(self, superoperator: ArrayLike, dimension: int, /):
        matrix = jnp.asarray(superoperator)
        size = int(dimension)
        if matrix.shape != (size * size, size * size):
            raise ValueError("Superoperator shape is invalid.")
        choi = jnp.zeros((size, size, size, size), dtype=matrix.dtype)
        for row in range(size):
            for column in range(size):
                basis = (
                    jnp.zeros((size, size), dtype=matrix.dtype).at[row, column].set(1.0)
                )
                output = (matrix @ basis.reshape(-1)).reshape((size, size))
                choi = choi.at[row, :, column, :].set(output)
        flat_choi = choi.reshape((size * size, size * size))
        flat_choi = 0.5 * (flat_choi + jnp.conj(flat_choi.T))
        partial_trace = jnp.trace(choi, axis1=1, axis2=3)
        self.choi_matrix = flat_choi
        self.cp_margin = jnp.min(jnp.linalg.eigvalsh(flat_choi))
        self.trace_preservation_residual = jnp.max(
            jnp.abs(partial_trace - jnp.eye(size, dtype=matrix.dtype))
        )
        self.valid = (
            jnp.all(jnp.isfinite(matrix))
            & (self.cp_margin >= -1e-8)
            & (self.trace_preservation_residual <= 1e-8)
        )


def solve_memory_kernel(
    problem: MemoryKernelMasterEquation,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
) -> OpenSystemHistorySolution:
    step = jnp.asarray(step_size, dtype=float).reshape(())
    count = int(steps)
    if count < 0 or float(step) <= 0.0:
        raise ValueError("steps and step_size must be positive.")
    states = [problem.initial_density]
    for current in range(count):
        time = step * current
        density = states[-1]
        memory = jnp.zeros_like(density)
        lower = max(0, current - int(problem.kernel.memory_horizon / float(step)))
        for past in range(lower, current + 1):
            lag = time - step * past
            weight = 0.5 if past in (lower, current) and current > lower else 1.0
            memory = memory + weight * problem.kernel(lag, states[past])
        derivative = problem.local_generator(time, density) + step * memory
        candidate = density + step * derivative
        states.append(candidate)
    values = jnp.stack(states)
    return OpenSystemHistorySolution(
        values, step * jnp.arange(count + 1), problem_id=problem.problem_id
    )


def solve_time_local_open_system(
    problem: TimeLocalOpenSystemProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
) -> OpenSystemHistorySolution:
    step = jnp.asarray(step_size, dtype=float).reshape(())
    count = int(steps)
    if count < 0 or float(step) <= 0.0 or not bool(jnp.isfinite(step)):
        raise ValueError("TCL steps and step_size must be finite and positive.")
    states = [problem.initial_density]
    for index in range(count):
        time = step * index
        state = states[-1]
        derivative = jnp.asarray(problem.generator(time, state))
        if derivative.shape != state.shape:
            raise ValueError("TCL generator must preserve density shape.")
        derivative = eqx.error_if(
            derivative,
            jnp.any(~jnp.isfinite(derivative)),
            "TCL generator returned nonfinite values.",
        )
        candidate = state + step * derivative
        states.append(candidate)
    return OpenSystemHistorySolution(
        jnp.stack(states), step * jnp.arange(count + 1), problem_id=problem.problem_id
    )


def exponential_memory_qubit_problem(
    strength: float,
    decay: float,
    initial_density: ArrayLike,
    /,
) -> MemoryKernelMasterEquation:
    sigma_minus = jnp.asarray([[0, 0], [1, 0]], dtype=complex)
    sigma_plus = jnp.conj(sigma_minus.T)

    def kernel(lag, density):
        commutator = sigma_minus @ density @ sigma_plus - 0.5 * (
            sigma_plus @ sigma_minus @ density + density @ sigma_plus @ sigma_minus
        )
        return float(strength) * jnp.exp(-float(decay) * lag) * commutator

    return MemoryKernelMasterEquation(
        lambda time, density: jnp.zeros_like(density),
        QuantumMemoryKernel(
            kernel,
            2,
            memory_horizon=8.0 / max(float(decay), 1e-12),
            kernel_id="exponential-qubit-memory",
        ),
        initial_density,
        problem_id="exponential-memory-qubit",
    )


__all__ = [
    "DynamicalMapPhysicality",
    "MemoryKernelMasterEquation",
    "OpenSystemHistorySolution",
    "QuantumMemoryKernel",
    "TimeLocalOpenSystemProblem",
    "exponential_memory_qubit_problem",
    "solve_memory_kernel",
    "solve_time_local_open_system",
]
