#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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
        if (
            frequency.shape != ()
            or bool(~jnp.isfinite(frequency))
            or bool(frequency <= 0.0)
            or samples < 3
            or size <= 0
        ):
            raise ValueError("Temporal harmonic plan values are invalid.")
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


class HarmonicBalanceDiagnostics(StrictModule):
    residual_norm: Array
    relative_residual: Array
    aliasing_tail: Array
    finite: Array


class HarmonicBalanceResult(StrictModule):
    waveform: Array
    coefficients: Array
    nonlinear: NonlinearResult
    diagnostics: HarmonicBalanceDiagnostics
    plan: TemporalHarmonicPlan


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


def solve_harmonic_balance(
    prepared_dae: PreparedCircuitDAE,
    initial_waveform: ArrayLike,
    angular_frequency: ArrayLike,
    /,
    *,
    args: Any = None,
    method: AbstractNonlinearMethod | None = None,
    termination: NonlinearTermination | None = None,
) -> HarmonicBalanceResult:
    if not isinstance(prepared_dae, PreparedCircuitDAE):
        raise TypeError("prepared_dae must be PreparedCircuitDAE.")
    initial = jnp.asarray(initial_waveform, dtype=float)
    if initial.ndim != 2 or initial.shape[1] != prepared_dae.plan.layout.size:
        raise ValueError("initial_waveform must have shape (samples, circuit_state).")
    plan = TemporalHarmonicPlan(angular_frequency, initial.shape[0], initial.shape[1])
    size = initial.size
    space = ArraySpace((size,), dtype=initial.dtype)
    residual = _HarmonicResidual(prepared_dae, plan, args)
    problem = NonlinearSystemProblem(
        residual,
        state_space=space,
        residual_space=space,
        problem_id=f"{prepared_dae.plan.circuit.circuit_id}/harmonic-balance",
    )
    prepared = prepare_nonlinear(
        problem,
        initial.reshape((-1,)),
        method=method,
        termination=termination,
    )
    nonlinear = solve_prepared_nonlinear(prepared)
    waveform = jnp.asarray(nonlinear.state).reshape(initial.shape)
    coefficients = jnp.fft.fft(waveform, axis=0) / plan.sample_count
    residual_value = residual(waveform.reshape((-1,)), None)
    residual_norm = jnp.linalg.norm(residual_value)
    scale = jnp.maximum(jnp.linalg.norm(waveform), 1.0)
    cutoff = max(1, plan.sample_count // 5)
    shifted = jnp.fft.fftshift(coefficients, axes=0)
    tail = jnp.concatenate((shifted[:cutoff], shifted[-cutoff:]), axis=0)
    diagnostics = HarmonicBalanceDiagnostics(
        residual_norm,
        residual_norm / scale,
        jnp.linalg.norm(tail),
        jnp.all(jnp.isfinite(waveform)) & jnp.all(jnp.isfinite(coefficients)),
    )
    return HarmonicBalanceResult(waveform, coefficients, nonlinear, diagnostics, plan)


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
    "HarmonicBalanceDiagnostics",
    "HarmonicBalanceResult",
    "TemporalHarmonicPlan",
    "floquet_multipliers",
    "shoot_periodic_circuit",
    "solve_harmonic_balance",
]
