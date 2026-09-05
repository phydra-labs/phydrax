#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Balanced positive-sequence RMS machine/network DAEs.

The closed model is a quasi-steady stator/network, unsaturated classical or
fourth-order two-axis transient machine, constant-P/Q loads by default, and optional
first-order excitation and droop governor. There are no electromagnetic network
transients, negative/zero sequence, subtransient states, saturation, deadbands,
protection relays, or hard state limiters. Controller command limits are explicit
and smooth first-order actuators track the clipped command. Mechanical power,
impedances and inertia use each machine's MVA base and the connected bus voltage
base; network powers use the network MVA base. Time is seconds.
Constant impedance is a separate, explicitly selected load fidelity, initialized
at the PF voltage. A nonzero PQ load has domain V != 0: no voltage floor or
impedance fallback is applied. Singular candidates give nonfinite native DAE
residual evidence; faults with no admissible PQ operating branch fail explicitly.

With exp(+i wt), generated current is outward from a machine (the negative of
its inward terminal current). In rotor coordinates V exp(-i delta)=Vq-i Vd and
I exp(-i delta)=Iq-i Id. Speed state is deviation from synchronous speed in pu.
Infinite buses are explicitly requested ideal voltage sources, never inferred
from a power-flow slack label. All other online generators require a machine.

The stator equations are Vq=Eq'-R Iq-Xd' Id and Vd=Ed'-R Id+Xq' Iq.
The standard small-speed-deviation swing approximation is delta_dot=omega_b nu,
2 H nu_dot=Pm-Pe-D nu, with Pe=Re(V conj(I))+R |I|^2 on machine base.

Scheduled events change only the algebraic network: finite balanced shunt
faults and branch breaker trip/reclose. Differential states remain continuous.
Native consistency solves reconstruct voltages and differential rates; every
post-event integration starts with fresh BDF history. The native event tape
covers differential coordinates only, whose fixed-time reset is the identity;
it does not claim an algebraic projection Jacobian or event-time sensitivity.
Initialization and segmented orchestration are host operations; the residual
and each native fixed-topology solve retain Equinox/JAX composition.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ... import ein
from ..._strict import StrictModule
from ...dynamics import DAEStructure, DifferentialAlgebraicSystem, TimeGrid
from ...solver import (
    dae_consistency_candidate,
    DAEAdaptivePolicy,
    DAEConsistencyCandidate,
    DAEConsistencyPolicy,
    DAEInitializationResult,
    DAEInitializationSpec,
    DAEResetMap,
    DAESolvePolicy,
    DAETerminationStatus,
    DifferentialAlgebraicProblem,
    DifferentialAlgebraicSolution,
    HybridEventPlan,
    HybridSchedulePlan,
    HybridScheduleResult,
    initialize_dae,
    ScheduledHybridEvent,
    solve_dae,
)
from ._network import CompiledNetwork, PowerNetwork, PowerStudy
from ._power_flow import _compiled, PowerFlowResult


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")


def _limits(lower: float, upper: float, name: str) -> None:
    if math.isnan(lower) or math.isnan(upper) or lower >= upper:
        raise ValueError(f"{name} must be ordered non-NaN lower/upper bounds.")


class FixedExciter(StrictModule):
    """Hold the field voltage derived from the power-flow operating point."""


class FirstOrderAVR(StrictModule):
    """T dEfd/dt = clip(K (Vref-|V|), lower, upper) - Efd.

    Vref is derived at initialization, not silently set to the PF bus target.
    Limits are field-voltage pu on machine base; there is no saturation model.
    """

    gain: float = 20.0
    time_constant: float = 0.05
    lower: float = -math.inf
    upper: float = math.inf

    def __check_init__(self):
        _positive(self.gain, "AVR gain")
        _positive(self.time_constant, "AVR time_constant")
        _limits(self.lower, self.upper, "AVR limits")


class FixedGovernor(StrictModule):
    """Hold initialized mechanical power, including stator copper losses."""


class DroopGovernor(StrictModule):
    """T dPm/dt = clip(Pref - speed_deviation/R, lower, upper) - Pm.

    Power and limits are on machine MVA base. No turbine/reheat/deadband model
    is implied by this explicit first-order governor/actuator family.
    """

    droop: float = 0.05
    time_constant: float = 0.5
    lower: float = -math.inf
    upper: float = math.inf

    def __check_init__(self):
        _positive(self.droop, "Governor droop")
        _positive(self.time_constant, "Governor time_constant")
        _limits(self.lower, self.upper, "Governor limits")


class ClassicalMachine(StrictModule):
    """Constant internal EMF behind R+jXd', with two swing states.

    inertia is H in seconds; damping multiplies pu speed deviation to give pu
    power on machine base. base_mva=None explicitly selects the network base.
    """

    generator: str = eqx.field(static=True)
    inertia: float
    damping: float = 0.0
    xd_prime: float = 0.3
    base_mva: float | None = None
    stator_resistance: float = 0.0
    governor: FixedGovernor | DroopGovernor = eqx.field(default_factory=FixedGovernor)

    def __check_init__(self):
        _machine_parameters(self)
        _positive(self.xd_prime, "xd_prime")


class Order4Machine(StrictModule):
    """Unsaturated two-axis transient machine: delta, speed, Eq', Ed'.

    Eq'_dot=(Efd-Eq'-(Xd-Xd')Id)/Td0',
    Ed'_dot=(-Ed'+(Xq-Xq')Iq)/Tq0'. This is not a subtransient GENROU model.
    """

    generator: str = eqx.field(static=True)
    inertia: float
    xd: float
    xq: float
    xd_prime: float
    xq_prime: float
    td0_prime: float
    tq0_prime: float
    damping: float = 0.0
    base_mva: float | None = None
    stator_resistance: float = 0.0
    avr: FixedExciter | FirstOrderAVR = eqx.field(default_factory=FixedExciter)
    governor: FixedGovernor | DroopGovernor = eqx.field(default_factory=FixedGovernor)

    def __check_init__(self):
        _machine_parameters(self)
        for name, value in (
            ("xd", self.xd),
            ("xq", self.xq),
            ("xd_prime", self.xd_prime),
            ("xq_prime", self.xq_prime),
            ("td0_prime", self.td0_prime),
            ("tq0_prime", self.tq0_prime),
        ):
            _positive(value, name)
        if self.xd < self.xd_prime or self.xq < self.xq_prime:
            raise ValueError("Synchronous reactance cannot be below transient reactance.")
        if not isinstance(self.avr, (FixedExciter, FirstOrderAVR)):
            raise TypeError("Unsupported active excitation model.")


Machine = ClassicalMachine | Order4Machine


def _machine_parameters(machine: Machine) -> None:
    if not isinstance(machine.generator, str) or not machine.generator:
        raise ValueError("A machine requires a nonempty generator ID.")
    _positive(machine.inertia, "inertia")
    _nonnegative(machine.damping, "damping")
    _nonnegative(machine.stator_resistance, "stator_resistance")
    if machine.base_mva is not None:
        _positive(machine.base_mva, "base_mva")
    if not isinstance(machine.governor, (FixedGovernor, DroopGovernor)):
        raise TypeError("Unsupported active governor model.")


class PowerTopology(StrictModule):
    """One fixed branch/shunt epoch; the explicitly selected load law is separate."""

    admittance: Array
    branch_closed: tuple[bool, ...] = eqx.field(static=True)
    faults: tuple[tuple[str, complex], ...] = eqx.field(static=True)
    epoch: int = eqx.field(static=True)


class PowerDynamicsModel(StrictModule):
    compiled: CompiledNetwork
    machines: tuple[Machine, ...]
    load_model: Literal["constant_power", "constant_impedance"] = eqx.field(static=True)
    load_admittance: Array | None
    machine_buses: tuple[int, ...] = eqx.field(static=True)
    machine_generators: tuple[int, ...] = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)
    differential_names: tuple[str, ...] = eqx.field(static=True)
    infinite_indices: tuple[int, ...] = eqx.field(static=True)
    infinite_voltage: Array
    base_ratios: Array
    internal_emf: Array
    mechanical_reference: Array
    excitation_reference: Array
    voltage_reference: Array
    initial_topology: PowerTopology

    @property
    def differential_size(self) -> int:
        return len(self.differential_names)

    def load_currents(self, voltage: ArrayLike) -> Array:
        """Consuming load currents on network base, with the selected fidelity.

        Loaded zero-voltage buses are outside the constant-power model domain;
        division remains singular so native DAE failure evidence is preserved.
        An unloaded bus carries exactly zero load current even at zero voltage.
        """
        value = jnp.asarray(voltage)
        if self.load_model == "constant_impedance":
            assert self.load_admittance is not None
            return self.load_admittance * value
        power = self.compiled.load_power
        denominator = jnp.where(power != 0, value, jnp.ones_like(value))
        return jnp.conj(power / denominator)

    def voltage(self, state: ArrayLike) -> Array:
        """Bus positive-sequence RMS voltage from one state or a state history."""
        values = jnp.asarray(state)
        start = self.differential_size
        count = len(self.compiled.network.buses)
        return values[..., start : start + count] + 1j * values[..., start + count :]

    def machine_currents(self, state: Array) -> Array:
        """Generated (outward) currents, on each machine's own MVA base."""
        voltage = self.voltage(state)
        currents = []
        for index, (machine, bus, offset) in enumerate(
            zip(self.machines, self.machine_buses, self.offsets, strict=True)
        ):
            delta = state[offset]
            rotor_voltage = voltage[bus] * jnp.exp(-1j * delta)
            if isinstance(machine, ClassicalMachine):
                eq, ed = self.internal_emf[index], 0.0
                xq_prime = machine.xd_prime
            else:
                eq, ed = state[offset + 2], state[offset + 3]
                xq_prime = machine.xq_prime
            a = eq - rotor_voltage.real
            b = ed + rotor_voltage.imag
            resistance = machine.stator_resistance
            denominator = resistance**2 + machine.xd_prime * xq_prime
            iq = (resistance * a - machine.xd_prime * b) / denominator
            id_ = (xq_prime * a + resistance * b) / denominator
            currents.append((iq - 1j * id_) * jnp.exp(1j * delta))
        return jnp.stack(currents)

    def machine_power(self, state: Array) -> Array:
        """Generated terminal complex powers on the network MVA base."""
        buses = jnp.asarray(self.machine_buses, dtype=jnp.int32)
        return (
            self.voltage(state)[buses]
            * jnp.conj(self.machine_currents(state))
            * self.base_ratios
        )

    def problem(
        self,
        state: ArrayLike,
        *,
        state_rate: ArrayLike | None = None,
        topology: PowerTopology | None = None,
    ) -> DifferentialAlgebraicProblem:
        """Construct a native index-one DAE; differential states are fixed at restart."""
        epoch = self.initial_topology if topology is None else topology
        size = self.differential_size + 2 * len(self.compiled.network.buses)
        roles = ("differential",) * self.differential_size + ("algebraic",) * (
            size - self.differential_size
        )
        system = DifferentialAlgebraicSystem(
            _PowerResidual(self, epoch),
            state_shape=(size,),
            structure=DAEStructure(roles),
            system_id=f"power-rms:{self.load_model}:{','.join(m.generator for m in self.machines)}:epoch:{epoch.epoch}",
        )
        return DifferentialAlgebraicProblem(
            system,
            state,
            initial_state_rate=state_rate,
            initialization=DAEInitializationSpec.index_one(),
        )


class _PowerResidual(StrictModule):
    model: PowerDynamicsModel
    topology: PowerTopology

    def rhs(
        self, state: Array, *, voltage: Array | None = None, currents: Array | None = None
    ) -> Array:
        model = self.model
        voltage = model.voltage(state) if voltage is None else voltage
        currents = model.machine_currents(state) if currents is None else currents
        result = []
        for index, (machine, bus, offset) in enumerate(
            zip(model.machines, model.machine_buses, model.offsets, strict=True)
        ):
            speed = state[offset + 1]
            cursor = offset + (2 if isinstance(machine, ClassicalMachine) else 4)
            if isinstance(machine, Order4Machine) and isinstance(
                machine.avr, FirstOrderAVR
            ):
                field = state[cursor]
                cursor += 1
            else:
                field = model.excitation_reference[index]
            mechanical = (
                state[cursor]
                if isinstance(machine.governor, DroopGovernor)
                else model.mechanical_reference[index]
            )
            current = currents[index]
            electrical = (
                voltage[bus] * jnp.conj(current)
            ).real + machine.stator_resistance * jnp.abs(current) ** 2
            result.extend(
                (
                    2 * jnp.pi * model.compiled.network.frequency * speed,
                    (mechanical - electrical - machine.damping * speed)
                    / (2 * machine.inertia),
                )
            )
            if isinstance(machine, Order4Machine):
                rotor_current = current * jnp.exp(-1j * state[offset])
                iq, id_ = rotor_current.real, -rotor_current.imag
                result.extend(
                    (
                        (
                            field
                            - state[offset + 2]
                            - (machine.xd - machine.xd_prime) * id_
                        )
                        / machine.td0_prime,
                        (-state[offset + 3] + (machine.xq - machine.xq_prime) * iq)
                        / machine.tq0_prime,
                    )
                )
                if isinstance(machine.avr, FirstOrderAVR):
                    target = jnp.clip(
                        machine.avr.gain
                        * (model.voltage_reference[index] - jnp.abs(voltage[bus])),
                        machine.avr.lower,
                        machine.avr.upper,
                    )
                    result.append((target - field) / machine.avr.time_constant)
            if isinstance(machine.governor, DroopGovernor):
                target = jnp.clip(
                    model.mechanical_reference[index] - speed / machine.governor.droop,
                    machine.governor.lower,
                    machine.governor.upper,
                )
                result.append((target - mechanical) / machine.governor.time_constant)
        return jnp.stack(result)

    def __call__(self, time: Array, state: Array, state_rate: Array, args, /) -> Array:
        del time, args
        model = self.model
        voltage = model.voltage(state)
        currents = model.machine_currents(state)
        mismatch = ein.contract("ij,j->i", self.topology.admittance, voltage)
        mismatch = mismatch + model.load_currents(voltage)
        buses = jnp.asarray(model.machine_buses, dtype=jnp.int32)
        mismatch = mismatch.at[buses].add(-currents * model.base_ratios)
        if model.infinite_indices:
            indices = jnp.asarray(model.infinite_indices, dtype=jnp.int32)
            mismatch = mismatch.at[indices].set(voltage[indices] - model.infinite_voltage)
        return jnp.concatenate(
            (
                state_rate[: model.differential_size]
                - self.rhs(state, voltage=voltage, currents=currents),
                mismatch.real,
                mismatch.imag,
            )
        )


class PowerDynamicsInitialization(StrictModule):
    model: PowerDynamicsModel
    problem: DifferentialAlgebraicProblem
    consistency: DAEInitializationResult
    operating_residual: Array
    equilibrium_norm: Array
    valid: Array
    status: str = eqx.field(static=True)


def initialize_power_dynamics(
    network: CompiledNetwork | PowerNetwork,
    power_flow: PowerFlowResult,
    machines: tuple[Machine, ...],
    *,
    study: PowerStudy | None = None,
    infinite_buses: tuple[str, ...] = (),
    load_model: Literal["constant_power", "constant_impedance"] = "constant_power",
    time: float = 0.0,
    policy: DAESolvePolicy | None = None,
    equilibrium_tolerance: float = 1e-7,
) -> PowerDynamicsInitialization:
    """Derive and certify a zero-rate operating point from a converged PF result.

    Constant-P/Q loads retain their PF powers by default, with current conj(S/V).
    Only explicit load_model="constant_impedance" converts them at the PF voltage.
    Raw networks require an explicit PowerStudy; a CompiledNetwork already binds
    its study and does not accept another one.
    Every online generator must be represented, including PF reference units.
    The returned validity also requires equilibrium, not merely DAE consistency.
    Invalid PF inputs/model coverage raise; numerical inconsistency is a status.
    """
    _positive(equilibrium_tolerance, "equilibrium_tolerance")
    if load_model not in ("constant_power", "constant_impedance"):
        raise ValueError(
            "Unsupported load_model; select constant_power or constant_impedance."
        )
    if not math.isfinite(time):
        raise ValueError("Initialization time must be finite.")
    compiled = _compiled(network, study)
    if not isinstance(compiled, CompiledNetwork) or not isinstance(
        power_flow, PowerFlowResult
    ):
        raise TypeError("Expected a compiled/network model and PowerFlowResult.")
    if not bool(np.asarray(power_flow.converged)):
        raise ValueError("Power dynamics requires a converged power flow.")
    selected = tuple(machines)
    if not selected or any(
        not isinstance(machine, (ClassicalMachine, Order4Machine)) for machine in selected
    ):
        raise TypeError("At least one supported classical/order-4 machine is required.")
    net = compiled.network
    bus_ids = tuple(bus.id for bus in net.buses)
    generator_ids = tuple(generator.id for generator in net.generators)
    if len(set(infinite_buses)) != len(infinite_buses) or any(
        bus not in bus_ids for bus in infinite_buses
    ):
        raise ValueError("Infinite buses must be distinct existing bus IDs.")
    if len({machine.generator for machine in selected}) != len(selected):
        raise ValueError("Each generator may have only one machine model.")
    if any(machine.generator not in generator_ids for machine in selected):
        raise ValueError("Machine references an unknown generator.")
    machine_generators = tuple(
        generator_ids.index(machine.generator) for machine in selected
    )
    machine_buses = tuple(
        bus_ids.index(net.generators[index].bus) for index in machine_generators
    )
    infinite_indices = tuple(bus_ids.index(bus) for bus in infinite_buses)
    for index in machine_generators:
        generator = net.generators[index]
        if not generator.in_service or generator.bus in infinite_buses:
            raise ValueError(
                "A machine must be online and not attached to an infinite bus."
            )
    for index, generator in enumerate(net.generators):
        if (
            generator.in_service
            and index not in machine_generators
            and generator.bus not in infinite_buses
        ):
            raise ValueError(
                f"Online generator {generator.id!r} has no supported dynamic model."
            )
    voltage = jnp.asarray(power_flow.voltage)
    generator_power = jnp.asarray(power_flow.generator_power)
    if voltage.shape != (len(net.buses),) or generator_power.shape != (
        len(net.generators),
    ):
        raise ValueError("Power-flow result shape does not match the network.")
    if not bool(np.all(np.isfinite(np.asarray(voltage)))) or bool(
        np.any(np.abs(np.asarray(voltage)) == 0)
    ):
        raise ValueError("Power-flow voltages must be finite and nonzero.")
    if not bool(np.all(np.isfinite(np.asarray(generator_power)))):
        raise ValueError("Power-flow generator powers must be finite.")
    external = jnp.asarray(power_flow.external_reference_power)
    if external.shape != voltage.shape or not bool(
        np.all(np.isfinite(np.asarray(external)))
    ):
        raise ValueError(
            "Power-flow external reference powers must match buses and be finite."
        )
    for reference in compiled.references:
        if (
            abs(complex(external[reference])) > equilibrium_tolerance
            and reference not in infinite_indices
        ):
            raise ValueError(
                "A nonzero external PF reference requires an explicit infinite bus."
            )
    supplied = (
        jnp.zeros_like(voltage).at[compiled.generator_indices].add(generator_power)
        + external
    )
    original_balance = (
        supplied
        - compiled.load_power
        - voltage * jnp.conj(compiled.bus_currents(voltage))
    )
    if float(jnp.max(jnp.abs(original_balance))) > equilibrium_tolerance:
        raise ValueError(
            "Power-flow operating point is inconsistent with the supplied network."
        )
    for load in net.loads:
        if load.in_service:
            if load.p < 0:
                raise ValueError(
                    "Negative active loads require an explicit generator model."
                )
    load_admittance = (
        jnp.conj(compiled.load_power) / jnp.abs(voltage) ** 2
        if load_model == "constant_impedance"
        else None
    )
    topology = PowerTopology(
        compiled.ybus,
        tuple(branch.in_service for branch in net.branches),
        (),
        0,
    )
    if not _energized(
        net, topology.branch_closed, set(machine_buses) | set(infinite_indices)
    ):
        raise ValueError(
            "Every dynamic island requires a machine or infinite-bus source."
        )
    states, offsets, names, ratios, emfs, mechanical_refs, field_refs, voltage_refs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for machine, generator_index, bus in zip(
        selected, machine_generators, machine_buses, strict=True
    ):
        ratio = (
            net.base_mva if machine.base_mva is None else machine.base_mva
        ) / net.base_mva
        current = jnp.conj(generator_power[generator_index] / voltage[bus]) / ratio
        resistance = machine.stator_resistance
        reactance = (
            machine.xd_prime if isinstance(machine, ClassicalMachine) else machine.xq
        )
        internal = voltage[bus] + (resistance + 1j * reactance) * current
        delta = jnp.angle(internal)
        rotor_current = current * jnp.exp(-1j * delta)
        rotor_voltage = voltage[bus] * jnp.exp(-1j * delta)
        iq, id_ = rotor_current.real, -rotor_current.imag
        eq = rotor_voltage.real + resistance * iq + machine.xd_prime * id_
        mechanical = (
            generator_power[generator_index].real / ratio
            + resistance * jnp.abs(current) ** 2
        )
        offsets.append(len(states))
        states.extend((delta, jnp.zeros_like(delta)))
        local_names = ["delta", "speed_deviation"]
        if isinstance(machine, ClassicalMachine):
            field = jnp.abs(internal)
            voltage_ref = jnp.abs(voltage[bus])
        else:
            ed = -rotor_voltage.imag + resistance * id_ - machine.xq_prime * iq
            field = eq + (machine.xd - machine.xd_prime) * id_
            states.extend((eq, ed))
            local_names.extend(("eq_prime", "ed_prime"))
            voltage_ref = jnp.abs(voltage[bus])
            if isinstance(machine.avr, FirstOrderAVR):
                if not machine.avr.lower <= float(field) <= machine.avr.upper:
                    raise ValueError(
                        f"{machine.generator}: PF field voltage violates AVR command limits."
                    )
                voltage_ref = voltage_ref + field / machine.avr.gain
                states.append(field)
                local_names.append("field_voltage")
        if isinstance(machine.governor, DroopGovernor):
            if not machine.governor.lower <= float(mechanical) <= machine.governor.upper:
                raise ValueError(
                    f"{machine.generator}: PF mechanical power violates governor command limits."
                )
            states.append(mechanical)
            local_names.append("mechanical_power")
        names.extend(f"{machine.generator}:{name}" for name in local_names)
        ratios.append(ratio)
        emfs.append(jnp.abs(internal))
        mechanical_refs.append(mechanical)
        field_refs.append(field)
        voltage_refs.append(voltage_ref)
    model = PowerDynamicsModel(
        compiled,
        selected,
        load_model,
        load_admittance,
        machine_buses,
        machine_generators,
        tuple(offsets),
        tuple(names),
        infinite_indices,
        voltage[jnp.asarray(infinite_indices, dtype=jnp.int32)],
        jnp.asarray(ratios),
        jnp.stack(emfs),
        jnp.stack(mechanical_refs),
        jnp.stack(field_refs),
        jnp.stack(voltage_refs),
        topology,
    )
    state = jnp.concatenate((jnp.stack(states), voltage.real, voltage.imag))
    problem = model.problem(state)
    operating_residual = problem.system.evaluate(
        jnp.asarray(time), state, jnp.zeros_like(state)
    )
    equilibrium_norm = jnp.max(jnp.abs(operating_residual))
    consistency = initialize_dae(problem, time, policy=policy)
    valid = (
        consistency.valid
        & jnp.isfinite(equilibrium_norm)
        & (equilibrium_norm <= equilibrium_tolerance)
    )
    if bool(np.asarray(valid)):
        problem = model.problem(consistency.state, state_rate=consistency.state_rate)
        status = "success"
    else:
        status = (
            "initialization_failed"
            if not bool(np.asarray(consistency.valid))
            else "not_an_equilibrium"
        )
    return PowerDynamicsInitialization(
        model, problem, consistency, operating_residual, equilibrium_norm, valid, status
    )


def initialize_smib(
    network: CompiledNetwork | PowerNetwork,
    power_flow: PowerFlowResult,
    machine: Machine,
    *,
    infinite_bus: str,
    study: PowerStudy | None = None,
    load_model: Literal["constant_power", "constant_impedance"] = "constant_power",
    time: float = 0.0,
    policy: DAESolvePolicy | None = None,
    equilibrium_tolerance: float = 1e-7,
) -> PowerDynamicsInitialization:
    """Single machine connected through the supplied network to one infinite bus."""
    return initialize_power_dynamics(
        network,
        power_flow,
        (machine,),
        infinite_buses=(infinite_bus,),
        study=study,
        load_model=load_model,
        time=time,
        policy=policy,
        equilibrium_tolerance=equilibrium_tolerance,
    )


def _energized(
    network: PowerNetwork, closed: tuple[bool, ...], sources: set[int]
) -> bool:
    ids = tuple(bus.id for bus in network.buses)
    neighbors = [set() for _ in ids]
    for branch, enabled in zip(network.branches, closed, strict=True):
        if enabled:
            left, right = ids.index(branch.from_bus), ids.index(branch.to_bus)
            neighbors[left].add(right)
            neighbors[right].add(left)
    unseen = set(range(len(ids)))
    while unseen:
        pending = [unseen.pop()]
        island = set(pending)
        while pending:
            for neighbor in neighbors[pending.pop()] & unseen:
                unseen.remove(neighbor)
                island.add(neighbor)
                pending.append(neighbor)
        if not island & sources:
            return False
    return True


class PowerEvent(StrictModule):
    """Scheduled bus fault/clear or branch trip/reclose, at an exact time.

    fault admittance is finite nonzero pu on network base (conductance >= 0).
    clear removes that bus's active fault. Reclose restores a previously tripped
    branch that was online in the initial network. Simultaneous events execute
    in the caller's tuple order, with consistency checked after each event.
    """

    time: float = eqx.field(static=True)
    kind: Literal["fault", "clear", "trip", "reclose"] = eqx.field(static=True)
    target: str = eqx.field(static=True)
    admittance: complex = eqx.field(static=True, default=0j)

    def __check_init__(self):
        if not math.isfinite(self.time):
            raise ValueError("Event time must be finite.")
        if self.kind not in ("fault", "clear", "trip", "reclose"):
            raise ValueError(
                "Unsupported power event; only bus faults and branch breakers are supported."
            )
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("An event requires a target ID.")
        admittance = complex(self.admittance)
        if self.kind == "fault":
            if (
                not math.isfinite(admittance.real)
                or not math.isfinite(admittance.imag)
                or admittance.real < 0
                or admittance == 0
            ):
                raise ValueError(
                    "A fault requires finite nonzero passive shunt admittance; bolted faults are unsupported."
                )
        elif admittance != 0:
            raise ValueError("Only fault events accept a shunt admittance.")


class PowerEventEvidence(StrictModule):
    event: PowerEvent
    before: Array
    after: Array
    rate_before: Array
    rate_after: Array
    residual_before: Array
    residual_after: Array
    differential_jump: Array
    topology_before: PowerTopology
    topology_after: PowerTopology
    consistency: DAEConsistencyCandidate | None
    scheduled: HybridScheduleResult | None
    applied: Array
    restart_order: int = eqx.field(static=True)
    status: str = eqx.field(static=True)


class PowerSegmentResult(StrictModule):
    start: float = eqx.field(static=True)
    stop: float = eqx.field(static=True)
    topology: PowerTopology
    solution: DifferentialAlgebraicSolution | None
    valid: Array
    status: str = eqx.field(static=True)


class PowerDynamicsResult(StrictModule):
    initialization: PowerDynamicsInitialization
    segments: tuple[PowerSegmentResult, ...]
    events: tuple[PowerEventEvidence, ...]
    final_state: Array
    final_state_rate: Array
    final_time: Array
    valid: Array
    status: str = eqx.field(static=True)

    @property
    def load_model(self) -> str:
        """Load fidelity actually used by every segment and restart."""
        return self.initialization.model.load_model


class _ContinuousReset(StrictModule):
    def __call__(self, time, state, rate, args, /):
        del time, args
        return state, rate


class _TimeGuard(StrictModule):
    time: float

    def __call__(self, time, state, args, /):
        del state, args
        return time - self.time


def _identity_reset(time, state, args, /):
    del time, args
    return state


class _BoundaryState(StrictModule):
    state: Array

    def __call__(self, time, args, /):
        del time, args
        return self.state


class _DifferentialFlow(StrictModule):
    residual: _PowerResidual
    algebraic: Array

    def __call__(self, time, differential, args, /):
        del time, args
        return self.residual.rhs(jnp.concatenate((differential, self.algebraic)))


def _changed_topology(
    model: PowerDynamicsModel, topology: PowerTopology, event: PowerEvent
) -> tuple[PowerTopology, str]:
    network = model.compiled.network
    buses = tuple(bus.id for bus in network.buses)
    branches = tuple(branch.id for branch in network.branches)
    faults = dict(topology.faults)
    closed = list(topology.branch_closed)
    admittance = topology.admittance
    if event.kind in ("fault", "clear"):
        if event.target not in buses:
            return topology, "unknown_bus"
        bus = buses.index(event.target)
        if event.kind == "fault":
            if event.target in faults:
                return topology, "fault_already_active"
            faults[event.target] = complex(event.admittance)
            admittance = admittance.at[bus, bus].add(event.admittance)
        else:
            if event.target not in faults:
                return topology, "no_active_fault"
            admittance = admittance.at[bus, bus].add(-faults.pop(event.target))
    else:
        if event.target not in branches:
            return topology, "unknown_branch"
        index = branches.index(event.target)
        want_closed = event.kind == "reclose"
        if closed[index] == want_closed:
            return (
                topology,
                "branch_already_closed" if want_closed else "branch_already_open",
            )
        if want_closed and not network.branches[index].in_service:
            return topology, "reclose_requires_initially_online_branch"
        sign = 1 if want_closed else -1
        yff, yft, ytf, ytt = model.compiled.branch_admittance[index]
        branch = network.branches[index]
        left, right = buses.index(branch.from_bus), buses.index(branch.to_bus)
        admittance = admittance.at[left, left].add(sign * yff)
        admittance = admittance.at[left, right].add(sign * yft)
        admittance = admittance.at[right, left].add(sign * ytf)
        admittance = admittance.at[right, right].add(sign * ytt)
        closed[index] = want_closed
    candidate = PowerTopology(
        admittance, tuple(closed), tuple(faults.items()), topology.epoch + 1
    )
    if not _energized(
        network,
        candidate.branch_closed,
        set(model.machine_buses) | set(model.infinite_indices),
    ):
        return candidate, "source_free_island"
    return candidate, "success"


def _apply_power_event(
    model: PowerDynamicsModel,
    topology: PowerTopology,
    event: PowerEvent,
    state: Array,
    rate: Array,
    policy: DAESolvePolicy | None,
    consistency_policy: DAEConsistencyPolicy,
) -> tuple[DifferentialAlgebraicProblem | None, PowerEventEvidence]:
    candidate_topology, status = _changed_topology(model, topology, event)
    before_residual = _PowerResidual(model, topology)
    pre = before_residual(jnp.asarray(event.time), state, rate, None)
    size = model.differential_size
    if status != "success":
        evidence = PowerEventEvidence(
            event,
            state,
            state,
            rate,
            rate,
            pre,
            jnp.full_like(pre, jnp.nan),
            jnp.zeros_like(state[:size]),
            topology,
            candidate_topology,
            None,
            None,
            jnp.asarray(False),
            1,
            status,
        )
        return None, evidence
    reset = DAEResetMap(
        _ContinuousReset(),
        DAEInitializationSpec.index_one(),
        reset_id=f"power:{event.kind}:{event.target}:{event.time}",
    )
    state_guess, rate_guess = reset.reset(jnp.asarray(event.time), state, rate, None)
    problem = model.problem(
        state_guess, state_rate=rate_guess, topology=candidate_topology
    )
    problem = eqx.tree_at(lambda item: item.initialization, problem, reset.initialization)
    consistency = dae_consistency_candidate(
        problem, event.time, consistency_policy, solve_policy=policy
    )
    after = consistency.initialization.state
    after_rate = consistency.initialization.state_rate
    post_residual = _PowerResidual(model, candidate_topology)
    post = post_residual(jnp.asarray(event.time), after, after_rate, None)
    jump = after[:size] - state[:size]
    applied = consistency.admissible & jnp.all(jump == 0)
    schedule_result = None
    if bool(np.asarray(applied)):
        # Only differential coordinates enter this tape: algebraic projection
        # is certified by the native consistency candidate, not a fake reset.
        time_tolerance = (
            8 * np.finfo(np.dtype(state.dtype)).eps * max(1.0, abs(event.time))
        )
        plan = HybridEventPlan(
            _TimeGuard(event.time),
            _identity_reset,
            _DifferentialFlow(before_residual, state[size:]),
            _DifferentialFlow(post_residual, after[size:]),
            event_tolerance=time_tolerance,
            event_kind=event.kind,
            plan_id=f"power:{event.target}:{event.time}:{candidate_topology.epoch}",
        )
        schedule = HybridSchedulePlan((ScheduledHybridEvent(plan),), maximum_events=1)
        event_time = jnp.asarray(event.time, dtype=state.dtype)
        bracket = jnp.stack((event_time - time_tolerance, event_time + time_tolerance))[
            None, :
        ]
        schedule_result = schedule.localize(_BoundaryState(state[:size]), bracket)
        applied = (
            applied
            & (schedule_result.event_count == 1)
            & ~schedule_result.capacity_exceeded
        )
    status = "success" if bool(np.asarray(applied)) else "restart_consistency_failed"
    evidence = PowerEventEvidence(
        event,
        state,
        after,
        rate,
        after_rate,
        pre,
        post,
        jump,
        topology,
        candidate_topology,
        consistency,
        schedule_result,
        applied,
        1,
        status,
    )
    return (consistency.apply(problem) if bool(np.asarray(applied)) else None), evidence


def simulate_power_dynamics(
    initialization: PowerDynamicsInitialization,
    times: ArrayLike | TimeGrid,
    *,
    events: tuple[PowerEvent, ...] = (),
    policy: DAESolvePolicy | None = None,
    consistency_policy: DAEConsistencyPolicy | None = None,
) -> PowerDynamicsResult:
    """Execute fixed-topology segments with native solve_dae and exact restarts.

    Every requested sample and event boundary is included. Event endpoints have
    both the pre-event sample in the preceding native solution and post-event
    state in event evidence/the following segment. No interpolation crosses a
    topology change. Failure stops execution; remaining segments are NOT_RUN.
    A failed event's `after` is only a candidate and is never adopted. The final
    state is always the last accepted state, not an invalid solver sample.

    The default is native adaptive BDF, with requested samples as exact save times.
    An explicit policy is used unchanged, including fixed-grid ratio restrictions.
    Distinct times are never merged; extremely close times can exceed native
    numerical resolution and produce an honest solver failure. For events intended
    on requested nodes, derive event times from those nodes rather than a second clock.
    """
    if not isinstance(initialization, PowerDynamicsInitialization):
        raise TypeError("Expected PowerDynamicsInitialization.")
    if policy is not None and policy.failure != "status":
        raise ValueError("Segmented power dynamics requires native failure='status'.")
    selected_policy = policy
    if selected_policy is None:
        adaptive = DAEAdaptivePolicy()
        selected_policy = DAESolvePolicy(adaptive=adaptive)
        algebraic_size = 2 * len(initialization.model.compiled.network.buses)
        total_size = algebraic_size + initialization.model.differential_size
        # RMS_alg <= sqrt(N / N_alg) * RMS_all. A root must satisfy both
        # native acceptance norms, not merely the less restrictive global RMS.
        root_bound = min(
            adaptive.residual_tolerance,
            adaptive.constraint_tolerance * math.sqrt(algebraic_size / total_size),
        )
        selected_policy = replace(
            selected_policy,
            nonlinear_termination=replace(
                selected_policy.nonlinear_termination,
                absolute_residual=math.nextafter(root_bound, 0.0),
                relative_residual=0.0,
            ),
        )
    values = np.asarray(
        times.times if isinstance(times, TimeGrid) else times, dtype=float
    )
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.all(np.isfinite(values))
        or not np.all(np.diff(values) > 0)
    ):
        raise ValueError(
            "times must be a finite strictly increasing vector with at least two nodes."
        )
    schedule = tuple(events)
    if any(not isinstance(event, PowerEvent) for event in schedule):
        raise TypeError("Only PowerEvent schedules are supported.")
    if any(event.time < values[0] or event.time > values[-1] for event in schedule):
        raise ValueError("Events must lie inside the requested time interval.")
    if any(right.time < left.time for left, right in zip(schedule, schedule[1:])):
        raise ValueError(
            "Events must be ordered by time; simultaneous events retain tuple order."
        )
    limits = (
        DAEConsistencyPolicy(1e6, 1e6, 1e6)
        if consistency_policy is None
        else consistency_policy
    )
    if not isinstance(limits, DAEConsistencyPolicy):
        raise TypeError("consistency_policy must be native DAEConsistencyPolicy.")
    boundaries = sorted(
        {float(values[0]), float(values[-1]), *(event.time for event in schedule)}
    )
    model = initialization.model
    state, rate = (
        initialization.problem.initial_state,
        initialization.problem.initial_state_rate,
    )
    topology = model.initial_topology
    problem = initialization.problem
    final_time = jnp.asarray(values[0], dtype=state.dtype)
    valid = bool(np.asarray(initialization.valid))
    status = "success" if valid else "initialization_failed"
    segments, evidence = [], []
    event_index = 0
    for boundary_index, start in enumerate(boundaries):
        while event_index < len(schedule) and schedule[event_index].time == start:
            if valid:
                adopted, transition = _apply_power_event(
                    model,
                    topology,
                    schedule[event_index],
                    state,
                    rate,
                    selected_policy,
                    limits,
                )
                evidence.append(transition)
                valid = bool(np.asarray(transition.applied))
                if valid:
                    assert adopted is not None
                    problem = adopted
                    state, rate = adopted.initial_state, adopted.initial_state_rate
                    topology = transition.topology_after
                else:
                    status = "event_failed"
            else:
                evidence.append(
                    PowerEventEvidence(
                        schedule[event_index],
                        state,
                        state,
                        rate,
                        rate,
                        jnp.full_like(state, jnp.nan),
                        jnp.full_like(state, jnp.nan),
                        jnp.zeros_like(state[: model.differential_size]),
                        topology,
                        topology,
                        None,
                        None,
                        jnp.asarray(False),
                        1,
                        "not_run",
                    )
                )
            event_index += 1
        if boundary_index + 1 == len(boundaries):
            break
        stop = boundaries[boundary_index + 1]
        if not valid:
            segments.append(
                PowerSegmentResult(
                    start, stop, topology, None, jnp.asarray(False), "not_run"
                )
            )
            continue
        segment_times = np.concatenate(
            ([start], values[(values > start) & (values < stop)], [stop])
        )
        grid = TimeGrid(
            jnp.asarray(segment_times, dtype=state.dtype),
            time_id=f"power:{topology.epoch}:{start}:{stop}",
        )
        solution = solve_dae(problem, grid, policy=selected_policy)
        valid = bool(
            np.asarray(
                jnp.all(solution.valid)
                & (solution.termination_status == int(DAETerminationStatus.SUCCESS))
            )
        )
        segments.append(
            PowerSegmentResult(
                start,
                stop,
                topology,
                solution,
                jnp.asarray(valid),
                "success" if valid else "solve_failed",
            )
        )
        if valid:
            state, rate = solution.states[-1], solution.state_rates[-1]
            final_time = solution.times[-1]
        else:
            if bool(np.asarray(solution.initialization.valid)):
                continuation = solution.continuation
                state, rate, final_time = (
                    continuation.state,
                    continuation.state_rate,
                    continuation.time,
                )
            status = "segment_failed"
        # A topology discontinuity must never reuse old BDF history/Jacobians.
        problem = model.problem(state, state_rate=rate, topology=topology)
    return PowerDynamicsResult(
        initialization,
        tuple(segments),
        tuple(evidence),
        state,
        rate,
        final_time,
        jnp.asarray(valid),
        status,
    )


__all__ = [
    "FixedExciter",
    "FirstOrderAVR",
    "FixedGovernor",
    "DroopGovernor",
    "ClassicalMachine",
    "Order4Machine",
    "PowerTopology",
    "PowerDynamicsModel",
    "PowerDynamicsInitialization",
    "PowerEvent",
    "PowerEventEvidence",
    "PowerSegmentResult",
    "PowerDynamicsResult",
    "initialize_power_dynamics",
    "initialize_smib",
    "simulate_power_dynamics",
]
