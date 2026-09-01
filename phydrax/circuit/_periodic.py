#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import ArraySpace, DenseLinearOperator
from ..linalg.eigen import general_eigensolve, GeneralEigenproblem
from ..nonlinear import (
    AbstractNonlinearMethod,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
    solve_prepared_nonlinear,
)
from ._dae import circuit_dae_problem, PreparedCircuitDAE


class TemporalHarmonicPlan(StrictModule):
    angular_frequency: Array
    sample_count: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, angular_frequency: ArrayLike, sample_count: int, state_size: int, /
    ):
        frequency = jnp.asarray(angular_frequency, dtype=float)
        samples, size = int(sample_count), int(state_size)
        if frequency.shape != () or samples < 3 or size <= 0:
            raise ValueError("Temporal harmonic plan values are invalid.")
        frequency = eqx.error_if(
            frequency,
            ~jnp.isfinite(frequency) | (frequency <= 0.0),
            "Angular frequency must be finite and positive.",
        )
        self.angular_frequency = frequency
        self.sample_count = samples
        self.state_size = size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "temporal-harmonic-plan",
                "samples": samples,
                "state_size": size,
            }
        )

    @property
    def period(self) -> Array:
        return 2.0 * jnp.pi / self.angular_frequency

    @property
    def times(self) -> Array:
        return self.period * jnp.arange(self.sample_count) / self.sample_count

    def derivative(self, waveform: ArrayLike, /) -> Array:
        values = jnp.asarray(waveform)
        if values.shape != (self.sample_count, self.state_size):
            raise ValueError("Harmonic waveform has the wrong shape.")
        coefficients = jnp.fft.fft(values, axis=0)
        wave_numbers = jnp.fft.fftfreq(self.sample_count) * self.sample_count
        derivative = jnp.fft.ifft(
            1j * self.angular_frequency * wave_numbers[:, None] * coefficients,
            axis=0,
        )
        return jnp.real(derivative) if not jnp.iscomplexobj(values) else derivative


class HarmonicBalancePolicy(StrictModule):
    """Resource envelope and aliasing qualification for harmonic balance."""

    maximum_samples: int = eqx.field(static=True)
    maximum_unknowns: int = eqx.field(static=True)
    maximum_waveform_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    aliasing_tail_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_samples: int = 4096,
        maximum_unknowns: int = 2**20,
        maximum_waveform_bytes: int = 2**30,
        maximum_workspace_bytes: int = 2**30,
        aliasing_tail_tolerance: float = 1e-6,
    ):
        limits = (
            int(maximum_samples),
            int(maximum_unknowns),
            int(maximum_waveform_bytes),
            int(maximum_workspace_bytes),
        )
        if any(limit <= 0 for limit in limits):
            raise ValueError("Harmonic-balance resource limits must be positive.")
        tolerance = float(aliasing_tail_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("aliasing_tail_tolerance must be finite and non-negative.")
        self.maximum_samples = limits[0]
        self.maximum_unknowns = limits[1]
        self.maximum_waveform_bytes = limits[2]
        self.maximum_workspace_bytes = limits[3]
        self.aliasing_tail_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "harmonic-balance-policy",
                "maximum_samples": limits[0],
                "maximum_unknowns": limits[1],
                "maximum_waveform_bytes": limits[2],
                "maximum_workspace_bytes": limits[3],
                "aliasing_tail_tolerance": tolerance,
            }
        )


class HarmonicBalanceCostEstimate(StrictModule):
    sample_count: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    unknown_count: int = eqx.field(static=True)
    waveform_bytes: int = eqx.field(static=True)
    coefficient_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


class HarmonicBalancePlan(StrictModule):
    temporal: TemporalHarmonicPlan
    policy: HarmonicBalancePolicy
    cost: HarmonicBalanceCostEstimate
    circuit_dae_plan_id: str = eqx.field(static=True)
    waveform_dtype: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class HarmonicBalanceDiagnostics(StrictModule):
    residual_norm: Array
    relative_residual: Array
    aliasing_tail: Array
    aliasing_tail_valid: Array
    finite: Array


class PreparedHarmonicBalance(StrictModule):
    plan: HarmonicBalancePlan
    prepared_dae: PreparedCircuitDAE
    residual: _HarmonicResidual
    nonlinear: PreparedNonlinearSolve
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class HarmonicBalanceResult(StrictModule):
    waveform: Array
    coefficients: Array
    nonlinear: NonlinearResult
    diagnostics: HarmonicBalanceDiagnostics
    plan: HarmonicBalancePlan
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class CircuitShootingResult(StrictModule):
    initial_state: Array
    final_state: Array
    mismatch: Array
    mismatch_norm: Array
    solution: Any
    successful: Array


class FloquetResult(StrictModule):
    multipliers: Array
    spectral_radius: Array
    stable: Array
    eigensolve: Any


class _HarmonicResidual(StrictModule):
    prepared_dae: PreparedCircuitDAE
    plan: TemporalHarmonicPlan
    args: Any

    def __call__(self, flat_waveform: Array, runtime_args: Any, /) -> Array:
        del runtime_args
        waveform = flat_waveform.reshape((self.plan.sample_count, self.plan.state_size))
        rates = self.plan.derivative(waveform)
        residuals = tuple(
            self.prepared_dae.system.evaluate(time, state, rate, self.args)
            for time, state, rate in zip(self.plan.times, waveform, rates, strict=True)
        )
        return jnp.stack(residuals).reshape((-1,))


def plan_harmonic_balance(
    prepared_dae: PreparedCircuitDAE,
    angular_frequency: ArrayLike,
    sample_count: int,
    policy: HarmonicBalancePolicy | None = None,
    /,
) -> HarmonicBalancePlan:
    """Preflight one exact Fourier-collocation circuit problem."""
    if not isinstance(prepared_dae, PreparedCircuitDAE):
        raise TypeError("prepared_dae must be PreparedCircuitDAE.")
    selected = HarmonicBalancePolicy() if policy is None else policy
    if not isinstance(selected, HarmonicBalancePolicy):
        raise TypeError("policy must be HarmonicBalancePolicy or None.")
    temporal = TemporalHarmonicPlan(
        angular_frequency,
        sample_count,
        prepared_dae.plan.layout.size,
    )
    dtype = jnp.dtype(prepared_dae.plan.state_scale.dtype)
    unknowns = temporal.sample_count * temporal.state_size
    waveform_bytes = unknowns * dtype.itemsize
    coefficient_bytes = unknowns * jnp.dtype(jnp.result_type(dtype, complex)).itemsize
    workspace_bytes = 3 * waveform_bytes + 2 * coefficient_bytes
    if temporal.sample_count > selected.maximum_samples:
        raise MemoryError("Harmonic balance exceeds maximum_samples.")
    if unknowns > selected.maximum_unknowns:
        raise MemoryError("Harmonic balance exceeds maximum_unknowns.")
    if waveform_bytes > selected.maximum_waveform_bytes:
        raise MemoryError("Harmonic balance exceeds maximum_waveform_bytes.")
    if workspace_bytes > selected.maximum_workspace_bytes:
        raise MemoryError("Harmonic balance exceeds maximum_workspace_bytes.")
    cost = HarmonicBalanceCostEstimate(
        temporal.sample_count,
        temporal.state_size,
        unknowns,
        waveform_bytes,
        coefficient_bytes,
        workspace_bytes,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "harmonic-balance-plan",
            "circuit_dae": prepared_dae.plan.plan_id,
            "temporal": temporal.plan_id,
            "waveform_dtype": dtype.name,
            "policy": selected.policy_id,
        }
    )
    return HarmonicBalancePlan(
        temporal,
        selected,
        cost,
        prepared_dae.plan.plan_id,
        dtype.name,
        plan_id,
    )


def _harmonic_initial(
    initial_waveform: ArrayLike,
    plan: HarmonicBalancePlan,
    /,
) -> Array:
    initial = jnp.asarray(initial_waveform, dtype=jnp.dtype(plan.waveform_dtype))
    expected = (plan.cost.sample_count, plan.cost.state_size)
    if initial.shape != expected:
        raise ValueError(f"initial_waveform must have shape {expected}.")
    return initial


def _harmonic_problem(
    prepared_dae: PreparedCircuitDAE,
    temporal: TemporalHarmonicPlan,
    args: Any,
    dtype: str,
    /,
) -> tuple[_HarmonicResidual, NonlinearSystemProblem]:
    residual = _HarmonicResidual(prepared_dae, temporal, args)
    space = ArraySpace(
        (temporal.sample_count * temporal.state_size,),
        dtype=jnp.dtype(dtype),
    )
    problem = NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        problem_id=f"{prepared_dae.plan.circuit.circuit_id}/harmonic-balance",
    )
    return residual, problem


def _harmonic_sample_count(initial_waveform: ArrayLike, /) -> int:
    shape = jnp.asarray(initial_waveform).shape
    if len(shape) != 2:
        raise ValueError("initial_waveform must be one rank-two array.")
    return int(shape[0])


def prepare_harmonic_balance(
    prepared_dae: PreparedCircuitDAE,
    initial_waveform: ArrayLike,
    angular_frequency: ArrayLike,
    plan_or_policy: HarmonicBalancePlan | HarmonicBalancePolicy | None = None,
    /,
    *,
    args: Any = None,
    method: AbstractNonlinearMethod | None = None,
    termination: NonlinearTermination | None = None,
) -> PreparedHarmonicBalance:
    """Bind waveform coefficients and one native nonlinear preparation."""
    if not isinstance(prepared_dae, PreparedCircuitDAE):
        raise TypeError("prepared_dae must be PreparedCircuitDAE.")
    sample_count = _harmonic_sample_count(initial_waveform)
    plan = (
        plan_or_policy
        if isinstance(plan_or_policy, HarmonicBalancePlan)
        else plan_harmonic_balance(
            prepared_dae,
            angular_frequency,
            sample_count,
            plan_or_policy,
        )
    )
    refreshed_plan = plan_harmonic_balance(
        prepared_dae,
        angular_frequency,
        sample_count,
        plan.policy,
    )
    if refreshed_plan.plan_id != plan.plan_id:
        raise ValueError("Harmonic-balance structure changed; replan is required.")
    initial = _harmonic_initial(initial_waveform, refreshed_plan)
    residual, problem = _harmonic_problem(
        prepared_dae,
        refreshed_plan.temporal,
        args,
        refreshed_plan.waveform_dtype,
    )
    nonlinear = prepare_nonlinear(
        problem,
        initial.reshape((-1,)),
        method=method,
        termination=termination,
    )
    prepared_id = canonical_fingerprint(
        {"kind": "prepared-harmonic-balance", "plan": refreshed_plan.plan_id}
    )
    return PreparedHarmonicBalance(
        refreshed_plan,
        prepared_dae,
        residual,
        nonlinear,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_harmonic_balance(
    prepared: PreparedHarmonicBalance,
    prepared_dae: PreparedCircuitDAE,
    initial_waveform: ArrayLike,
    angular_frequency: ArrayLike,
    /,
    *,
    args: Any = None,
) -> PreparedHarmonicBalance:
    """Refresh frequency, circuit coefficients, waveform, and runtime arguments."""
    if not isinstance(prepared, PreparedHarmonicBalance):
        raise TypeError("prepared must be PreparedHarmonicBalance.")
    if not isinstance(prepared_dae, PreparedCircuitDAE):
        raise TypeError("prepared_dae must be PreparedCircuitDAE.")
    sample_count = _harmonic_sample_count(initial_waveform)
    plan = plan_harmonic_balance(
        prepared_dae,
        angular_frequency,
        sample_count,
        prepared.plan.policy,
    )
    if plan.plan_id != prepared.plan.plan_id:
        raise ValueError("Harmonic-balance structure changed; replan is required.")
    initial = _harmonic_initial(initial_waveform, plan)
    residual, problem = _harmonic_problem(
        prepared_dae,
        plan.temporal,
        args,
        plan.waveform_dtype,
    )
    nonlinear = refresh_nonlinear(
        prepared.nonlinear,
        problem,
        initial.reshape((-1,)),
    )
    return PreparedHarmonicBalance(
        plan,
        prepared_dae,
        residual,
        nonlinear,
        prepared.numeric_version + 1,
        prepared.prepared_id,
    )


def solve_prepared_harmonic_balance(
    prepared: PreparedHarmonicBalance,
    /,
) -> HarmonicBalanceResult:
    """Solve one prepared native harmonic-balance problem."""
    if not isinstance(prepared, PreparedHarmonicBalance):
        raise TypeError("prepared must be PreparedHarmonicBalance.")
    nonlinear = solve_prepared_nonlinear(prepared.nonlinear)
    shape = (prepared.plan.cost.sample_count, prepared.plan.cost.state_size)
    waveform = jnp.asarray(nonlinear.state).reshape(shape)
    coefficients = jnp.fft.fft(waveform, axis=0) / prepared.plan.cost.sample_count
    residual_value = prepared.residual(waveform.reshape((-1,)), None)
    residual_norm = jnp.linalg.norm(residual_value)
    scale = jnp.maximum(jnp.linalg.norm(waveform), 1.0)
    cutoff = max(1, prepared.plan.cost.sample_count // 5)
    shifted = jnp.fft.fftshift(coefficients, axes=0)
    tail = jnp.concatenate((shifted[:cutoff], shifted[-cutoff:]), axis=0)
    tail_norm = jnp.linalg.norm(tail)
    diagnostics = HarmonicBalanceDiagnostics(
        residual_norm,
        residual_norm / scale,
        tail_norm,
        tail_norm <= prepared.plan.policy.aliasing_tail_tolerance,
        jnp.all(jnp.isfinite(waveform)) & jnp.all(jnp.isfinite(coefficients)),
    )
    return HarmonicBalanceResult(
        waveform,
        coefficients,
        nonlinear,
        diagnostics,
        prepared.plan,
        prepared.numeric_version,
        prepared.prepared_id,
    )


def solve_harmonic_balance(
    prepared_dae: PreparedCircuitDAE,
    initial_waveform: ArrayLike,
    angular_frequency: ArrayLike,
    /,
    *,
    args: Any = None,
    method: AbstractNonlinearMethod | None = None,
    termination: NonlinearTermination | None = None,
    policy: HarmonicBalancePolicy | None = None,
) -> HarmonicBalanceResult:
    """Plan, prepare, and solve one harmonic-balance problem."""
    prepared = prepare_harmonic_balance(
        prepared_dae,
        initial_waveform,
        angular_frequency,
        policy,
        args=args,
        method=method,
        termination=termination,
    )
    return solve_prepared_harmonic_balance(prepared)


def shoot_periodic_circuit(
    prepared_dae: PreparedCircuitDAE,
    initial_state: ArrayLike,
    time_grid: Any,
    /,
    *,
    initial_state_rate: ArrayLike | None = None,
    args: Any = None,
    policy: Any = None,
) -> CircuitShootingResult:
    if not isinstance(prepared_dae, PreparedCircuitDAE):
        raise TypeError("prepared_dae must be PreparedCircuitDAE.")
    from ..solver import solve_dae

    initial = jnp.asarray(initial_state, dtype=float)
    problem = circuit_dae_problem(
        prepared_dae,
        initial,
        initial_state_rate=initial_state_rate,
        args=args,
    )
    solution = solve_dae(problem, time_grid, policy=policy)
    final_state = solution.states[-1]
    mismatch = final_state - initial
    successful = jnp.all(solution.valid) & jnp.all(jnp.isfinite(final_state))
    return CircuitShootingResult(
        initial,
        final_state,
        mismatch,
        jnp.linalg.norm(mismatch),
        solution,
        successful,
    )


def floquet_multipliers(
    period_map: Callable[[Array], ArrayLike],
    state: ArrayLike,
    /,
    *,
    policy: Any = None,
) -> FloquetResult:
    if not callable(period_map):
        raise TypeError("period_map must be callable.")
    value = jnp.asarray(state, dtype=float)
    if value.ndim != 1 or value.size == 0:
        raise ValueError("Floquet state must be one nonempty vector.")
    monodromy = jax.jacfwd(lambda current: jnp.asarray(period_map(current)))(value)
    result = general_eigensolve(
        GeneralEigenproblem(
            DenseLinearOperator(monodromy, operator_id="circuit-monodromy"),
            problem_id="circuit-floquet",
        ),
        policy=policy,
    )
    multipliers = result.eigenvalues
    radius = jnp.max(jnp.abs(multipliers))
    return FloquetResult(multipliers, radius, radius <= 1.0, result)


__all__ = [
    "CircuitShootingResult",
    "FloquetResult",
    "HarmonicBalanceCostEstimate",
    "HarmonicBalanceDiagnostics",
    "HarmonicBalancePlan",
    "HarmonicBalancePolicy",
    "HarmonicBalanceResult",
    "PreparedHarmonicBalance",
    "TemporalHarmonicPlan",
    "floquet_multipliers",
    "plan_harmonic_balance",
    "prepare_harmonic_balance",
    "refresh_harmonic_balance",
    "shoot_periodic_circuit",
    "solve_harmonic_balance",
    "solve_prepared_harmonic_balance",
]
