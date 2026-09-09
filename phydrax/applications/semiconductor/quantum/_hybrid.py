# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Stationary current/energy matching between disjoint DD and quantum regions."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._strict import StrictModule
from ....units import AMPERE
from .._continuum import PreparedSemiconductorDevice, SemiconductorOperatingPoint
from .._materials import SemiconductorMaterial
from .._quantities import (
    _positive_scalar,
    _text,
    ELEMENTARY_CHARGE_SI as Q,
    WATT_UNIT,
)
from ._coherent import CoherentDevice, CoherentResult, integrate_coherent
from ._leads import BoundStateOccupation


class QuantumClassicalInterface(StrictModule):
    """One reservoir-matched boundary between explicitly disjoint populations.

    The classical and quantum region identities must differ. The joined carrier
    chemical energy is ``-q*voltage`` on both sides. Quantum lead onsite and
    chemical potential shift together, so changing interface voltage is a
    physical electrostatic gauge/bias operation rather than occupation-only
    injection. No DD mobility, SRH, tunneling or quantum self-energy is copied
    across the boundary.
    """

    voltage_bounds: Array
    current_tolerance: Array
    heat_tolerance: Array
    classical_terminal: str = eqx.field(static=True)
    quantum_terminal: str = eqx.field(static=True)
    classical_region_id: str = eqx.field(static=True)
    quantum_region_id: str = eqx.field(static=True)
    energy_reference: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)

    def __init__(
        self,
        classical_terminal,
        quantum_terminal,
        *,
        classical_region_id,
        quantum_region_id,
        voltage_bounds,
        current_tolerance,
        heat_tolerance,
        energy_reference,
        provenance,
        maximum_steps=48,
    ):
        self.classical_terminal = _text(classical_terminal, "classical terminal")
        if quantum_terminal not in ("left", "right"):
            raise ValueError("quantum_terminal must be 'left' or 'right'.")
        self.quantum_terminal = quantum_terminal
        self.classical_region_id = _text(classical_region_id, "classical region identity")
        self.quantum_region_id = _text(quantum_region_id, "quantum region identity")
        if self.classical_region_id == self.quantum_region_id:
            raise ValueError(
                "Classical and quantum carrier populations require disjoint owners."
            )
        bounds = jnp.asarray(voltage_bounds, dtype=float)
        host = np.asarray(bounds)
        if host.shape != (2,) or not np.all(np.isfinite(host)) or not host[0] < host[1]:
            raise ValueError("voltage_bounds must be two increasing finite SI voltages.")
        self.voltage_bounds = bounds
        self.current_tolerance = _positive_scalar(
            current_tolerance,
            AMPERE,
            AMPERE,
            "hybrid current tolerance",
        )
        self.heat_tolerance = _positive_scalar(
            heat_tolerance,
            WATT_UNIT,
            WATT_UNIT,
            "hybrid heat tolerance",
        )
        if (
            isinstance(maximum_steps, bool)
            or not isinstance(maximum_steps, (int, np.integer))
            or maximum_steps < 1
        ):
            raise ValueError("maximum_steps must be a positive integer.")
        self.maximum_steps = int(maximum_steps)
        self.energy_reference = _text(energy_reference, "hybrid energy reference")
        self.provenance = _text(provenance, "hybrid interface provenance")


class HybridInterfaceEvaluation(StrictModule):
    interface_voltage: Array
    classical: SemiconductorOperatingPoint
    quantum: CoherentResult
    current_defect: Array
    heat_defect: Array
    electrochemical_defect: Array
    classical_electron_current: Array
    classical_hole_current: Array
    quantum_electron_current: Array
    hole_leakage: Array
    successful: Array


class HybridCouplingEvidence(StrictModule):
    bracketed: Array
    current_error: Array
    heat_error: Array
    electrochemical_error: Array
    hole_leakage: Array
    classical_successful: Array
    quantum_successful: Array
    iterations: int = eqx.field(static=True)
    classical_region_id: str = eqx.field(static=True)
    quantum_region_id: str = eqx.field(static=True)
    energy_reference: str = eqx.field(static=True)
    closure: str = eqx.field(
        static=True,
        default=(
            "stationary electron-selective reservoir matching; disjoint "
            "classical/quantum populations"
        ),
    )


class HybridOperatingPoint(StrictModule):
    interface_voltage: Array
    classical: SemiconductorOperatingPoint
    quantum: CoherentResult
    attempts: tuple[HybridInterfaceEvaluation, ...]
    evidence: HybridCouplingEvidence
    successful: Array


def _classical_energy_reference(prepared):
    references = {
        model.thermodynamics.energy_reference
        for model in prepared.plan.material_models
        if isinstance(model, SemiconductorMaterial) and model.thermodynamics is not None
    }
    if len(references) != 1 or any(
        isinstance(model, SemiconductorMaterial) and model.thermodynamics is None
        for model in prepared.plan.material_models
    ):
        raise ValueError(
            "Hybrid coupling requires one explicit classical band-energy reference."
        )
    return references.pop()


def _shift_joined_lead(device, side, voltage):
    lead = device.left if side == "left" else device.right
    target_mu = -Q * voltage
    shifted = lead.shifted(target_mu - lead.chemical_potential)
    return eqx.tree_at(
        (lambda value: value.left) if side == "left" else (lambda value: value.right),
        device,
        shifted,
    )


def solve_quantum_classical_interface(
    prepared,
    coherent,
    interface,
    classical_voltages,
    /,
    *,
    classical_initial=None,
    bound_occupation: BoundStateOccupation | None = None,
    quantum_tolerance=1e-6,
    spectral_tolerance=2e-4,
    initial_panels=16,
    max_refinements=3,
):
    """Bisect the physical interface voltage until particle/energy exchange closes.

    Component failures and an unbracketed current are retained as unsuccessful
    evidence; they are never reinterpreted as an insulating physical solution.
    The classical side must be an ohmic carrier terminal. Heat matching is an
    acceptance gate when the classical model evolves energy and is otherwise
    reported as NaN rather than fabricated from an isothermal thermostat.
    """
    if not isinstance(prepared, PreparedSemiconductorDevice):
        raise TypeError("prepared must be PreparedSemiconductorDevice.")
    if not isinstance(coherent, CoherentDevice):
        raise TypeError("coherent must be CoherentDevice.")
    if not isinstance(interface, QuantumClassicalInterface):
        raise TypeError("interface must be QuantumClassicalInterface.")
    if interface.classical_terminal not in prepared.plan.terminal_names:
        raise ValueError(
            "The declared classical terminal is absent from the device plan."
        )
    classical_index = prepared.plan.terminal_names.index(interface.classical_terminal)
    terminal_nodes = prepared.plan.terminal_index == classical_index
    if not bool(jnp.any(terminal_nodes & prepared.plan.ohmic_mask)):
        raise ValueError(
            "Quantum/classical particle matching requires an ohmic terminal."
        )
    if _classical_energy_reference(prepared) != interface.energy_reference:
        raise ValueError("Classical and hybrid interface energy references differ.")
    if coherent.hamiltonian.energy_reference != interface.energy_reference:
        raise ValueError("Quantum and hybrid interface energy references differ.")
    voltages = jnp.asarray(classical_voltages, dtype=float)
    if voltages.shape != (prepared.num_terminals,) or bool(
        jnp.any(~jnp.isfinite(voltages))
    ):
        raise ValueError(
            "classical_voltages must provide one finite SI value per terminal."
        )
    quantum_index = 0 if interface.quantum_terminal == "left" else 1
    attempts = []

    def evaluate(voltage):
        classical_bias = voltages.at[classical_index].set(voltage)
        classical = prepared.solve(classical_bias, initial=classical_initial)
        shifted = _shift_joined_lead(coherent, interface.quantum_terminal, voltage)
        quantum = integrate_coherent(
            shifted,
            bound_occupation=bound_occupation,
            tolerance=quantum_tolerance,
            spectral_tolerance=spectral_tolerance,
            initial_panels=initial_panels,
            max_refinements=max_refinements,
        )
        sources = prepared._sources(classical.coordinates)
        electron_source, hole_source, electron_kinetic = (
            sources[0],
            sources[1],
            sources[2],
        )
        electron_rate = prepared._terminal_sum(electron_source, prepared.plan.ohmic_mask)[
            classical_index
        ]
        hole_rate = prepared._terminal_sum(hole_source, prepared.plan.ohmic_mask)[
            classical_index
        ]
        classical_electron_current = Q * electron_rate
        classical_hole_current = -Q * hole_rate
        quantum_electron_current = quantum.terminal_currents[quantum_index]
        current = classical_electron_current + quantum_electron_current
        hole_leakage = jnp.abs(classical_hole_current)
        joined_mu = -Q * voltage
        if prepared.plan.electrothermal:
            electron_material_energy = sources[-1][-2] * electron_source
            electron_energy_source = prepared._terminal_sum(
                electron_kinetic + electron_material_energy,
                prepared.plan.ohmic_mask,
            )[classical_index]
            classical_electron_heat = -electron_energy_source + joined_mu * electron_rate
            heat = classical_electron_heat + quantum.heat_currents[quantum_index]
            heat_ok = jnp.abs(heat) <= interface.heat_tolerance
        else:
            heat = jnp.asarray(jnp.nan, dtype=current.dtype)
            heat_ok = jnp.asarray(True)
        quantum_mu = (
            shifted.left.chemical_potential
            if interface.quantum_terminal == "left"
            else shifted.right.chemical_potential
        )
        electrochemical = jnp.abs(joined_mu - quantum_mu)
        valid = (
            classical.successful
            & quantum.successful
            & (jnp.abs(current) <= interface.current_tolerance)
            & (hole_leakage <= interface.current_tolerance)
            & heat_ok
            & (electrochemical <= 64 * jnp.finfo(current.dtype).eps * Q)
        )
        result = HybridInterfaceEvaluation(
            jnp.asarray(voltage),
            classical,
            quantum,
            current,
            heat,
            electrochemical,
            classical_electron_current,
            classical_hole_current,
            quantum_electron_current,
            hole_leakage,
            valid,
        )
        attempts.append(result)
        return result

    lower, upper = map(float, np.asarray(interface.voltage_bounds))
    left, right = evaluate(lower), evaluate(upper)
    selected = (
        left
        if abs(float(left.current_defect)) <= abs(float(right.current_defect))
        else right
    )
    bracketed = jnp.asarray(
        np.isfinite(float(left.current_defect))
        and np.isfinite(float(right.current_defect))
        and float(left.current_defect) * float(right.current_defect) <= 0
    )
    iterations = 0
    if not bool(selected.successful) and bool(bracketed):
        while iterations < interface.maximum_steps:
            middle = 0.5 * (lower + upper)
            candidate = evaluate(middle)
            iterations += 1
            if abs(float(candidate.current_defect)) < abs(float(selected.current_defect)):
                selected = candidate
            if bool(candidate.successful):
                selected = candidate
                break
            if float(left.current_defect) * float(candidate.current_defect) <= 0:
                upper, right = middle, candidate
            else:
                lower, left = middle, candidate
    evidence = HybridCouplingEvidence(
        bracketed=bracketed,
        current_error=jnp.abs(selected.current_defect),
        heat_error=jnp.abs(selected.heat_defect),
        hole_leakage=selected.hole_leakage,
        electrochemical_error=selected.electrochemical_defect,
        classical_successful=selected.classical.successful,
        quantum_successful=selected.quantum.successful,
        iterations=iterations,
        classical_region_id=interface.classical_region_id,
        quantum_region_id=interface.quantum_region_id,
        energy_reference=interface.energy_reference,
    )
    successful = selected.successful & bracketed
    return HybridOperatingPoint(
        selected.interface_voltage,
        selected.classical,
        selected.quantum,
        tuple(attempts),
        evidence,
        successful,
    )


__all__ = [
    "HybridCouplingEvidence",
    "HybridInterfaceEvaluation",
    "HybridOperatingPoint",
    "QuantumClassicalInterface",
    "solve_quantum_classical_interface",
]
