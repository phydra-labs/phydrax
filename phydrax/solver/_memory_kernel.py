#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._temporal_precision import TemporalPrecisionPolicy
from ..integration import IntegrationPrecisionPolicy
from ..linalg import HermitianPrecisionPolicy, HermitianSpectrum
from ..operators.quantum import (
    ApproximationAxis,
    OpenSystemApproximationEvidence,
    OpenSystemPhysicalityEvidence,
)


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
    geometry_precision: GeometryPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_generator: Callable[[Array, Array], Array],
        kernel: QuantumMemoryKernel,
        initial_density: ArrayLike,
        /,
        *,
        geometry_precision: GeometryPrecisionPolicy | None = None,
        hermitian_precision: HermitianPrecisionPolicy | None = None,
        problem_id: str = "memory-kernel-master",
    ):
        if not callable(local_generator):
            raise TypeError("local_generator must be callable.")
        density = jnp.asarray(initial_density)
        geometry_ = (
            GeometryPrecisionPolicy()
            if geometry_precision is None
            else geometry_precision
        )
        hermitian_ = (
            HermitianPrecisionPolicy()
            if hermitian_precision is None
            else hermitian_precision
        )
        if not isinstance(geometry_, GeometryPrecisionPolicy):
            raise TypeError("geometry_precision must be GeometryPrecisionPolicy or None.")
        if not isinstance(hermitian_, HermitianPrecisionPolicy):
            raise TypeError(
                "hermitian_precision must be HermitianPrecisionPolicy or None."
            )
        geometry_.validate_coordinates(density)
        if density.shape != (kernel.dimension, kernel.dimension):
            raise ValueError("Initial density shape does not match the memory kernel.")
        spectrum = HermitianSpectrum(density, precision=hermitian_)
        self.local_generator = local_generator
        self.kernel = kernel
        self.initial_density = density
        self.geometry_precision = geometry_
        self.hermitian_precision = hermitian_
        self.precision_evidence = geometry_.evidence_for(
            density,
            children={"initial-spectrum": spectrum.precision_evidence},
        )
        self.problem_id = str(problem_id)


class TimeLocalOpenSystemProblem(StrictModule):
    generator: Callable[[Array, Array], Array]
    initial_density: Array
    geometry_precision: GeometryPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        generator: Callable[[Array, Array], Array],
        initial_density: ArrayLike,
        /,
        *,
        geometry_precision: GeometryPrecisionPolicy | None = None,
        hermitian_precision: HermitianPrecisionPolicy | None = None,
        problem_id: str = "time-local-open-system",
    ):
        if not callable(generator):
            raise TypeError("generator must be callable.")
        density = jnp.asarray(initial_density)
        geometry_ = (
            GeometryPrecisionPolicy()
            if geometry_precision is None
            else geometry_precision
        )
        hermitian_ = (
            HermitianPrecisionPolicy()
            if hermitian_precision is None
            else hermitian_precision
        )
        if not isinstance(geometry_, GeometryPrecisionPolicy):
            raise TypeError("geometry_precision must be GeometryPrecisionPolicy or None.")
        if not isinstance(hermitian_, HermitianPrecisionPolicy):
            raise TypeError(
                "hermitian_precision must be HermitianPrecisionPolicy or None."
            )
        geometry_.validate_coordinates(density)
        if density.ndim != 2 or density.shape[0] != density.shape[1]:
            raise ValueError("Initial density must be square.")
        spectrum = HermitianSpectrum(density, precision=hermitian_)
        self.generator = generator
        self.initial_density = density
        self.geometry_precision = geometry_
        self.hermitian_precision = hermitian_
        self.precision_evidence = geometry_.evidence_for(
            density,
            children={"initial-spectrum": spectrum.precision_evidence},
        )
        self.problem_id = str(problem_id)


class OpenSystemHistorySolution(StrictModule):
    states: Array
    times: Array
    trace_residuals: Array
    hermiticity_residuals: Array
    minimum_eigenvalues: Array
    approximation: OpenSystemApproximationEvidence
    physicality: OpenSystemPhysicalityEvidence
    valid: Array
    temporal_precision: TemporalPrecisionPolicy
    geometry_precision: GeometryPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy
    integration_precision: IntegrationPrecisionPolicy | None
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        states: ArrayLike,
        times: ArrayLike,
        /,
        *,
        problem_id: str,
        representation_id: str,
        approximation_axes,
        temporal_precision: TemporalPrecisionPolicy,
        geometry_precision: GeometryPrecisionPolicy,
        hermitian_precision: HermitianPrecisionPolicy,
        integration_precision: IntegrationPrecisionPolicy | None = None,
    ):
        values = jnp.asarray(states)
        times_ = jnp.asarray(times)
        temporal_precision.validate_state(values[0])
        geometry_precision.validate_coordinates(values[0])
        traces = geometry_precision.accumulation(jnp.trace(values, axis1=-2, axis2=-1))
        trace_residuals = geometry_precision.decision(jnp.abs(traces - 1.0))
        hermiticity_residuals = geometry_precision.decision(
            jnp.max(
                jnp.abs(
                    geometry_precision.accumulation(
                        values - jnp.swapaxes(jnp.conj(values), -1, -2)
                    )
                ),
                axis=(-2, -1),
            )
        )
        spectrum = HermitianSpectrum(values, precision=hermitian_precision)
        minimum_eigenvalues = geometry_precision.decision(spectrum.minimum_eigenvalue)
        valid = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(trace_residuals <= 1e-6)
            & jnp.all(hermiticity_residuals <= 1e-6)
            & jnp.all(minimum_eigenvalues >= -1e-6)
            & jnp.all(spectrum.valid)
        )
        children = {
            "state-reduction": geometry_precision.evidence_for(values[0]),
            "state-spectrum": spectrum.precision_evidence,
        }
        policy_ids = [
            temporal_precision.policy_id,
            geometry_precision.policy_id,
            hermitian_precision.policy_id,
        ]
        if integration_precision is not None:
            if not isinstance(integration_precision, IntegrationPrecisionPolicy):
                raise TypeError(
                    "integration_precision must be IntegrationPrecisionPolicy or None."
                )
            children["memory-quadrature"] = integration_precision.evidence_for(values[0])
            policy_ids.append(integration_precision.policy_id)
        self.states = temporal_precision.output(values)
        self.times = times_
        self.trace_residuals = trace_residuals
        self.hermiticity_residuals = hermiticity_residuals
        self.minimum_eigenvalues = minimum_eigenvalues
        self.valid = valid
        self.temporal_precision = temporal_precision
        self.geometry_precision = geometry_precision
        self.hermitian_precision = hermitian_precision
        self.integration_precision = integration_precision
        self.precision_evidence = temporal_precision.evidence_for(
            values[0],
            times_[0],
            children=children,
        )
        self.approximation = OpenSystemApproximationEvidence(
            representation_id,
            tuple(approximation_axes),
            local_error=temporal_precision.decision(approximation_axes[-1].value),
            valid=valid,
            precision_evidence=self.precision_evidence,
            precision_policy_ids=tuple(policy_ids),
        )
        status = "valid" if bool(valid) else "invalid"
        self.physicality = OpenSystemPhysicalityEvidence(
            trace_residual=jnp.max(trace_residuals),
            hermiticity_residual=jnp.max(hermiticity_residuals),
            positivity_margin=jnp.min(minimum_eigenvalues),
            status=status,
            precision_evidence=self.precision_evidence,
        )
        self.problem_id = str(problem_id)


class DynamicalMapPhysicality(StrictModule):
    choi_matrix: Array
    cp_margin: Array
    trace_preservation_residual: Array
    valid: Array
    geometry_precision: GeometryPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)

    def __init__(
        self,
        superoperator: ArrayLike,
        dimension: int,
        /,
        *,
        geometry_precision: GeometryPrecisionPolicy | None = None,
        hermitian_precision: HermitianPrecisionPolicy | None = None,
    ):
        matrix = jnp.asarray(superoperator)
        geometry_ = (
            GeometryPrecisionPolicy()
            if geometry_precision is None
            else geometry_precision
        )
        hermitian_ = (
            HermitianPrecisionPolicy()
            if hermitian_precision is None
            else hermitian_precision
        )
        if not isinstance(geometry_, GeometryPrecisionPolicy):
            raise TypeError("geometry_precision must be GeometryPrecisionPolicy or None.")
        if not isinstance(hermitian_, HermitianPrecisionPolicy):
            raise TypeError(
                "hermitian_precision must be HermitianPrecisionPolicy or None."
            )
        geometry_.validate_coordinates(matrix)
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
        spectrum = HermitianSpectrum(flat_choi, precision=hermitian_)
        partial_trace = geometry_.accumulation(jnp.trace(choi, axis1=1, axis2=3))
        cp_margin = geometry_.decision(spectrum.minimum_eigenvalue)
        trace_residual = geometry_.decision(
            jnp.max(jnp.abs(partial_trace - jnp.eye(size, dtype=matrix.dtype)))
        )
        self.choi_matrix = geometry_.output(flat_choi)
        self.cp_margin = cp_margin
        self.trace_preservation_residual = trace_residual
        self.valid = (
            jnp.all(jnp.isfinite(matrix))
            & spectrum.valid
            & (cp_margin >= -1e-8)
            & (trace_residual <= 1e-8)
        )
        self.geometry_precision = geometry_
        self.hermitian_precision = hermitian_
        self.precision_evidence = geometry_.evidence_for(
            matrix,
            children={"choi-spectrum": spectrum.precision_evidence},
        )


def solve_memory_kernel(
    problem: MemoryKernelMasterEquation,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    temporal_precision: TemporalPrecisionPolicy | None = None,
    integration_precision: IntegrationPrecisionPolicy | None = None,
    geometry_precision: GeometryPrecisionPolicy | None = None,
    hermitian_precision: HermitianPrecisionPolicy | None = None,
) -> OpenSystemHistorySolution:
    if not isinstance(problem, MemoryKernelMasterEquation):
        raise TypeError("problem must be MemoryKernelMasterEquation.")
    temporal_ = (
        TemporalPrecisionPolicy() if temporal_precision is None else temporal_precision
    )
    integration_ = (
        IntegrationPrecisionPolicy()
        if integration_precision is None
        else integration_precision
    )
    geometry_ = (
        problem.geometry_precision if geometry_precision is None else geometry_precision
    )
    hermitian_ = (
        problem.hermitian_precision
        if hermitian_precision is None
        else hermitian_precision
    )
    if not isinstance(temporal_, TemporalPrecisionPolicy):
        raise TypeError("temporal_precision must be TemporalPrecisionPolicy or None.")
    if not isinstance(integration_, IntegrationPrecisionPolicy):
        raise TypeError(
            "integration_precision must be IntegrationPrecisionPolicy or None."
        )
    temporal_.validate_state(problem.initial_density)
    step = temporal_.coefficient(
        jnp.asarray(step_size, dtype=problem.initial_density.real.dtype)
    ).reshape(())
    count = int(steps)
    if count < 0 or float(step) <= 0.0:
        raise ValueError("steps and step_size must be positive.")
    states = [problem.initial_density]
    for current in range(count):
        time = step * current
        density = states[-1]
        lower = max(
            0,
            current - int(problem.kernel.memory_horizon / float(step)),
        )
        memory = integration_.accumulation(jnp.zeros_like(density))
        for past in range(lower, current + 1):
            lag = time - step * past
            weight = 0.5 if past in (lower, current) and current > lower else 1.0
            contribution = integration_.evaluation(problem.kernel(lag, states[past]))
            memory = memory + weight * integration_.accumulation(contribution)
        local = temporal_.stage(problem.local_generator(time, density))
        convolution = temporal_.stage(integration_.output(step * memory))
        derivative = temporal_.residual(local + convolution)
        candidate = jnp.asarray(
            density + step * temporal_.accumulation(derivative),
            dtype=density.dtype,
        )
        states.append(candidate)
    values = jnp.stack(states)
    return OpenSystemHistorySolution(
        values,
        step * jnp.arange(count + 1),
        problem_id=problem.problem_id,
        representation_id="memory-kernel-master",
        approximation_axes=(
            ApproximationAxis(
                "memory-horizon",
                problem.kernel.memory_horizon,
                units="time",
            ),
            ApproximationAxis("time-step", step, units="time"),
        ),
        temporal_precision=temporal_,
        geometry_precision=geometry_,
        hermitian_precision=hermitian_,
        integration_precision=integration_,
    )


def solve_time_local_open_system(
    problem: TimeLocalOpenSystemProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
    temporal_precision: TemporalPrecisionPolicy | None = None,
    geometry_precision: GeometryPrecisionPolicy | None = None,
    hermitian_precision: HermitianPrecisionPolicy | None = None,
) -> OpenSystemHistorySolution:
    if not isinstance(problem, TimeLocalOpenSystemProblem):
        raise TypeError("problem must be TimeLocalOpenSystemProblem.")
    temporal_ = (
        TemporalPrecisionPolicy() if temporal_precision is None else temporal_precision
    )
    geometry_ = (
        problem.geometry_precision if geometry_precision is None else geometry_precision
    )
    hermitian_ = (
        problem.hermitian_precision
        if hermitian_precision is None
        else hermitian_precision
    )
    temporal_.validate_state(problem.initial_density)
    step = temporal_.coefficient(
        jnp.asarray(step_size, dtype=problem.initial_density.real.dtype)
    ).reshape(())
    count = int(steps)
    states = [problem.initial_density]
    for index in range(count):
        time = step * index
        state = states[-1]
        derivative = temporal_.residual(temporal_.stage(problem.generator(time, state)))
        candidate = jnp.asarray(
            state + step * temporal_.accumulation(derivative),
            dtype=state.dtype,
        )
        states.append(candidate)
    return OpenSystemHistorySolution(
        jnp.stack(states),
        step * jnp.arange(count + 1),
        problem_id=problem.problem_id,
        representation_id="time-local-open-system",
        approximation_axes=(ApproximationAxis("time-step", step, units="time"),),
        temporal_precision=temporal_,
        geometry_precision=geometry_,
        hermitian_precision=hermitian_,
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
