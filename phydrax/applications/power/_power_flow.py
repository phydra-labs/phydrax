#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Rectangular AC power flow and explicit, bounded PV/reactive-limit mode selection."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._network import compile_network, CompiledNetwork, PowerNetwork, PowerStudy


class FixedModePowerFlowResult(StrictModule):
    """Differentiable fixed-mode solution; derivatives require native success.

    Discrete active-set selection is deliberately outside this map. At a reactive
    switching boundary this is a one-mode derivative, not a derivative of selection.
    """

    voltage: Array
    bus_power: Array
    branch_from: Array
    branch_to: Array
    root: NonlinearResult
    modes: tuple[str, ...] = eqx.field(static=True)

    @property
    def converged(self) -> Array:
        return self.root.successful

    @property
    def residual_norm(self) -> Array:
        return jnp.max(jnp.abs(self.root.residual), initial=0.0)


class PowerFlowResult(StrictModule):
    voltage: Array
    generator_power: Array
    bus_power: Array
    branch_from: Array
    branch_to: Array
    branch_loss: Array
    shunt_power: Array
    external_reference_power: Array
    bus_balance: Array
    total_balance: Array
    residual_norm: Array
    reference_limit_violation: Array
    generator_limit_violation: Array
    voltage_violation: Array
    branch_limit_violation: Array
    converged: Array
    fixed_mode: FixedModePowerFlowResult
    status: str = eqx.field(static=True)
    modes: tuple[str, ...] = eqx.field(static=True)
    mode_history: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    switching_buses: tuple[str, ...] = eqx.field(static=True)

    @property
    def operationally_feasible(self) -> Array:
        return (
            self.converged
            & (self.voltage_violation <= 1e-6)
            & (self.branch_limit_violation <= 1e-6)
        )


def _compiled(
    value: PowerNetwork | CompiledNetwork, study: PowerStudy | None = None
) -> CompiledNetwork:
    if isinstance(value, CompiledNetwork):
        if study is not None:
            raise ValueError("study is already bound in CompiledNetwork.")
        return value
    if study is None:
        raise ValueError("A physical PowerNetwork requires an explicit PowerStudy.")
    return compile_network(value, study)


def fixed_mode_power_flow(
    compiled: CompiledNetwork,
    injections: ArrayLike | None = None,
    *,
    initial_voltage: ArrayLike | None = None,
    modes: tuple[str, ...] | None = None,
    method: AbstractNonlinearMethod | None = None,
    termination: NonlinearTermination | None = None,
) -> FixedModePowerFlowResult:
    """Native matrix-free implicit rectangular root for one declared mode.

    injections are generation-minus-load complex pu at every bus. Reference P/Q
    and PV Q entries are not specified equations. q_min/q_max modes use the given
    reactive injection, so callers differentiating a saturated mode pass its bound.
    Compilation and discrete mode decisions occur outside jit; this solve is JAX
    composable and retains native nonlinear status, work and derivative evidence.
    """
    if not isinstance(compiled, CompiledNetwork):
        raise TypeError("fixed_mode_power_flow requires a CompiledNetwork.")
    n = len(compiled.network.buses)
    modes_ = compiled.control_modes if modes is None else tuple(modes)
    if len(modes_) != n or any(
        m not in ("pq", "pv", "reference", "q_min", "q_max") for m in modes_
    ):
        raise ValueError("One supported fixed mode is required for every bus.")
    if tuple(i for i, m in enumerate(modes_) if m == "reference") != compiled.references:
        # references are island-ordered rather than necessarily bus-ordered.
        if set(i for i, m in enumerate(modes_) if m == "reference") != set(
            compiled.references
        ):
            raise ValueError(
                "Fixed-mode references must match compiled island references."
            )
    power = compiled.specified_power if injections is None else jnp.asarray(injections)
    voltage = (
        compiled.initial_voltage
        if initial_voltage is None
        else jnp.asarray(initial_voltage)
    )
    if power.shape != (n,) or voltage.shape != (n,):
        raise ValueError("Power and initial voltage must contain one scalar per bus.")
    reference = jnp.asarray([mode == "reference" for mode in modes_])
    pv = jnp.asarray([mode == "pv" for mode in modes_])
    target = compiled.initial_voltage

    def residual(coordinates, specified):
        v = coordinates[:n] + 1j * coordinates[n:]
        mismatch = v * jnp.conj(compiled.bus_currents(v)) - specified
        real = jnp.where(reference, v.real - target.real, mismatch.real)
        imag = jnp.where(
            reference,
            v.imag - target.imag,
            jnp.where(pv, jnp.abs(v) ** 2 - compiled.voltage_setpoints**2, mismatch.imag),
        )
        return jnp.concatenate((real, imag))

    problem = NonlinearSystemProblem(
        residual, problem_id="balanced-rectangular-power-flow"
    )
    root = implicit_root_result(
        problem,
        jnp.concatenate((voltage.real, voltage.imag)),
        method=method,
        termination=termination,
        args=power,
    )
    value = root.state[:n] + 1j * root.state[n:]
    branch_from, branch_to = compiled.branch_powers(value)
    return FixedModePowerFlowResult(
        value,
        value * jnp.conj(compiled.bus_currents(value)),
        branch_from,
        branch_to,
        root,
        modes_,
    )


def _allocate(total, initial, lower, upper):
    """Bounded equal incremental participation; at most one saturation per pass."""
    value = np.clip(np.asarray(initial, dtype=float), lower, upper)
    for _ in range(len(value) + 1):
        remainder = float(total) - float(value.sum())
        free = value < upper if remainder > 0 else value > lower
        if abs(remainder) < 1e-13 or not np.any(free):
            break
        value[free] = np.clip(
            value[free] + remainder / np.count_nonzero(free), lower[free], upper[free]
        )
    return value


def _limit_violation(value, lower, upper):
    return jnp.maximum(
        jnp.max(lower - value, initial=0.0), jnp.max(value - upper, initial=0.0)
    )


def solve_power_flow(
    network: PowerNetwork | CompiledNetwork,
    *,
    study: PowerStudy | None = None,
    initial_voltage: ArrayLike | None = None,
    q_tolerance: float = 1e-7,
    maximum_mode_steps: int | None = None,
    method: AbstractNonlinearMethod | None = None,
    termination: NonlinearTermination | None = None,
) -> PowerFlowResult:
    """Host active-set controller around native roots, never a new Newton solver.

    PV Q demand is shared by bounded equal incremental participation around each
    generator's q. A saturated PV bus becomes PQ at its aggregate bound; modes can
    be released if the voltage complementarity sign reverses. Cycles and the work
    cap fail explicitly. Reference P/Q limits fail without changing reference type.
    Multiple in-service generators at a reference share both P and Q by the same
    bounded participation rule. A generator-free reference is an explicit ideal
    external voltage source and is recorded separately, not fabricated generation.
    """
    if not np.isfinite(q_tolerance) or q_tolerance < 0:
        raise ValueError("q_tolerance must be finite and nonnegative.")
    compiled = _compiled(network, study)
    buses, generators = compiled.network.buses, compiled.network.generators
    n = len(buses)
    cap = 2 * n + 2 if maximum_mode_steps is None else int(maximum_mode_steps)
    if cap < 1:
        raise ValueError("maximum_mode_steps must be positive.")
    modes = list(compiled.control_modes)
    injections = compiled.specified_power
    initial = (
        compiled.initial_voltage
        if initial_voltage is None
        else jnp.asarray(initial_voltage)
    )
    history, status = [], "mode_budget_exhausted"
    switching = set()
    fixed = None
    for _ in range(cap):
        mode_tuple = tuple(modes)
        if mode_tuple in history:
            status = "mode_cycle"
            break
        history.append(mode_tuple)
        fixed = fixed_mode_power_flow(
            compiled,
            injections,
            initial_voltage=initial,
            modes=mode_tuple,
            method=method,
            termination=termination,
        )
        if not bool(fixed.converged):
            status = "nonlinear_failure"
            break
        initial = fixed.voltage
        q_required = np.asarray(fixed.bus_power.imag + compiled.load_power.imag)
        magnitudes = np.asarray(jnp.abs(fixed.voltage))
        change = None
        for i, mode in enumerate(modes):
            if mode not in ("pv", "q_min", "q_max"):
                continue
            ids = compiled.generators_at_bus[i]
            qmin = sum(generators[g].q_min for g in ids)
            qmax = sum(generators[g].q_max for g in ids)
            if min(abs(q_required[i] - qmin), abs(q_required[i] - qmax)) <= q_tolerance:
                switching.add(buses[i].id)
            if mode == "pv":
                if q_required[i] < qmin - q_tolerance:
                    change = (i, "q_min", qmin)
                elif q_required[i] > qmax + q_tolerance:
                    change = (i, "q_max", qmax)
            elif (
                mode == "q_min"
                and magnitudes[i] < float(compiled.voltage_setpoints[i]) - q_tolerance
            ):
                change = (i, "pv", sum(generators[g].q for g in ids))
            elif (
                mode == "q_max"
                and magnitudes[i] > float(compiled.voltage_setpoints[i]) + q_tolerance
            ):
                change = (i, "pv", sum(generators[g].q for g in ids))
            if change is not None:
                break
        if change is None:
            status = "success"
            break
        i, modes[i], qvalue = change
        injections = injections.at[i].set(
            injections[i].real + 1j * (qvalue - compiled.load_power[i].imag)
        )
        switching.add(buses[i].id)
    assert fixed is not None
    # Evidence refers to the last actually solved mode, not an unexecuted proposal.
    modes_final = fixed.modes
    generated = np.asarray(
        [complex(g.p, g.q) if g.in_service else 0j for g in generators]
    )
    required = np.asarray(fixed.bus_power + compiled.load_power)
    external = np.zeros(n, dtype=complex)
    reference_violation = 0.0
    for i, ids in enumerate(compiled.generators_at_bus):
        if not ids:
            if modes_final[i] == "reference":
                external[i] = required[i]
            continue
        indices = np.asarray(ids, dtype=int)
        group = [generators[g] for g in ids]
        if modes_final[i] in ("reference", "pv", "q_min", "q_max"):
            lower, upper = (
                np.asarray([g.q_min for g in group]),
                np.asarray([g.q_max for g in group]),
            )
            q = _allocate(required[i].imag, [g.q for g in group], lower, upper)
            generated[indices] = generated[indices].real + 1j * q
            if modes_final[i] == "reference":
                reference_violation = max(
                    reference_violation, abs(required[i].imag - q.sum())
                )
        if modes_final[i] == "reference":
            lower, upper = (
                np.asarray([g.p_min for g in group]),
                np.asarray([g.p_max for g in group]),
            )
            p = _allocate(required[i].real, [g.p for g in group], lower, upper)
            generated[indices] = p + 1j * generated[indices].imag
            reference_violation = max(
                reference_violation, abs(required[i].real - p.sum())
            )
    gen_power = jnp.asarray(generated)
    p_lower = jnp.asarray([g.p_min if g.in_service else 0 for g in generators])
    p_upper = jnp.asarray([g.p_max if g.in_service else 0 for g in generators])
    q_lower = jnp.asarray([g.q_min if g.in_service else 0 for g in generators])
    q_upper = jnp.asarray([g.q_max if g.in_service else 0 for g in generators])
    gen_violation = jnp.maximum(
        _limit_violation(gen_power.real, p_lower, p_upper),
        _limit_violation(gen_power.imag, q_lower, q_upper),
    )
    dispatched = (
        jnp.zeros(n, dtype=gen_power.dtype).at[compiled.generator_indices].add(gen_power)
    )
    balance = dispatched + jnp.asarray(external) - compiled.load_power - fixed.bus_power
    branch_loss = fixed.branch_from + fixed.branch_to
    shunt_power = jnp.abs(fixed.voltage) ** 2 * jnp.conj(compiled.shunt_admittance)
    total_balance = (
        jnp.sum(dispatched + jnp.asarray(external) - compiled.load_power)
        - jnp.sum(branch_loss)
        - jnp.sum(shunt_power)
    )
    voltage_violation = _limit_violation(
        jnp.abs(fixed.voltage),
        jnp.asarray([b.v_min for b in buses]),
        jnp.asarray([b.v_max for b in buses]),
    )
    rates = jnp.asarray([b.rate for b in compiled.network.branches])
    branch_violation = jnp.max(
        jnp.maximum(jnp.abs(fixed.branch_from), jnp.abs(fixed.branch_to)) - rates,
        initial=0.0,
    )
    if status == "success" and reference_violation > q_tolerance:
        status = "reference_limit_failure"
    if status == "success" and float(gen_violation) > q_tolerance:
        status = "generator_dispatch_limit_failure"
    return PowerFlowResult(
        fixed.voltage,
        gen_power,
        fixed.bus_power,
        fixed.branch_from,
        fixed.branch_to,
        branch_loss,
        shunt_power,
        jnp.asarray(external),
        balance,
        total_balance,
        fixed.residual_norm,
        jnp.asarray(reference_violation),
        gen_violation,
        voltage_violation,
        branch_violation,
        jnp.asarray(status == "success"),
        fixed,
        status,
        modes_final,
        tuple(history),
        tuple(sorted(switching)),
    )


__all__ = [
    "FixedModePowerFlowResult",
    "PowerFlowResult",
    "fixed_mode_power_flow",
    "solve_power_flow",
]
