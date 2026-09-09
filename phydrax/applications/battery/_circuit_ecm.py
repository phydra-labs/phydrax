#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...circuit._dae import CircuitDAEDiagnostics, prepare_circuit_dae, PreparedCircuitDAE
from ...circuit._elements import (
    AbstractImplicitCircuitLaw,
    CircuitElement,
    CircuitElementEvaluation,
    CircuitElementStateLayout,
    implicit_law_for,
    IndependentCurrentSourceLaw,
)
from ...circuit._mna import AbstractMNAComponent, CircuitInstance, NodalCircuit, NodalPort
from ...circuit._ports import ElectricalWaveReference
from ...dynamics import DifferentialAlgebraicSystem
from ...nonlinear._domain_cond import domain_cond
from ...solver import DAEInitializationSpec, DifferentialAlgebraicProblem, HybridGuardPlan
from ._ecm import (
    _model_terms,
    _OBSERVABLE_NAMES,
    _OBSERVABLE_UNITS,
    ThermalEquivalentCircuitInitialCondition,
    ThermalEquivalentCircuitParameters,
    ThermalEquivalentCircuitPlan,
    ThermalEquivalentCircuitState,
)
from ._properties import TabulatedPropertyLaw
from ._results import BatteryDAESolution, BatteryModelOutput
from ._thermal_graph import (
    augment_circuit_thermal_graph,
    ThermalGraphParameters,
    ThermalGraphPlan,
)


def _parameters(args: Any, /) -> ThermalEquivalentCircuitParameters:
    value = (
        args if isinstance(args, ThermalEquivalentCircuitParameters) else args.parameters
    )
    if not isinstance(value, ThermalEquivalentCircuitParameters):
        raise TypeError("Circuit ECM requires ThermalEquivalentCircuitParameters.")
    return value


def _physical(state: Array, /) -> ThermalEquivalentCircuitState:
    return ThermalEquivalentCircuitState(state[..., 0], state[..., 1:-2], state[..., -2])


def _local_support_bounds(parameters, soc, /):
    lower, upper = jnp.asarray(0.0), jnp.asarray(1.0)
    for law in (parameters.open_circuit_voltage, parameters.entropic_coefficient):
        bounds = law.support_bounds
        lower, upper = jnp.maximum(lower, bounds[0]), jnp.minimum(upper, bounds[1])
        if isinstance(law, TabulatedPropertyLaw):
            unsupported = ~(law.source_mask[:-1] & law.source_mask[1:])
            # Stop inside the current connected support interval, including
            # holes in a masked table, before a trial can enter the hole.
            lower = jnp.maximum(
                lower,
                jnp.max(
                    jnp.where(
                        unsupported & (law.nodes[1:] <= soc), law.nodes[1:], -jnp.inf
                    )
                ),
            )
            upper = jnp.minimum(
                upper,
                jnp.min(
                    jnp.where(
                        unsupported & (law.nodes[:-1] >= soc), law.nodes[:-1], jnp.inf
                    )
                ),
            )
    return lower, upper


class CircuitEcmImplicitLaw(AbstractImplicitCircuitLaw):
    """Active storage law with passive terminal current: positive means charging.

    Auxiliary order is q, RC polarization voltages, T, I. The cell owns thermal
    storage and generated heat only; graph cooling is applied once outside it.
    No generic passive energy law is attached to this electrochemical cell.
    """

    branch_count: int = eqx.field(static=True)

    def __init__(self, branch_count: int, /):
        topology = ThermalEquivalentCircuitPlan(branch_count)
        self.branch_count = topology.branch_count
        self.terminal_count = 2
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout(
            ("differential",) * (branch_count + 2) + ("algebraic",)
        )
        self.input_names = ()
        self.law_id = canonical_fingerprint(
            {
                "kind": "battery-circuit-ecm-law",
                "branches": branch_count,
                "thermal": "storage-minus-generated-only",
                "terminal_current": "positive-charging",
            }
        )

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del time, terminal_voltage_rates, inputs
        p = _parameters(args)
        if p.branch_resistances_ohm.shape != (self.branch_count,):
            raise ValueError("Circuit ECM branch parameters do not match topology.")
        current = state[-1]
        physical = _physical(state)
        terms = _model_terms(p, physical, current)
        rc_residual = (
            state_rate[1:-2]
            + physical.polarization_voltages_v
            / (p.branch_resistances_ohm * p.branch_capacitances_f)
            - current / p.branch_capacitances_f
        )
        residual = jnp.concatenate(
            (
                jnp.atleast_1d(state_rate[0] - current),
                rc_residual,
                jnp.atleast_1d(
                    p.heat_capacity_j_per_k * state_rate[-2] - terms[3] - terms[4]
                ),
                jnp.atleast_1d(terminal_voltages[0] - terminal_voltages[1] - terms[1]),
            )
        )
        return CircuitElementEvaluation(jnp.stack((current, -current)), residual)


class CircuitConnectedEcmPlan(StrictModule, NonTrainableState):
    """One native cell and a two-terminal current source, voltage source or load.

    The default driven current source is oriented negative-to-positive. A custom
    autonomous boundary is oriented positive-to-negative (e.g. a resistor or a
    positive voltage source). The 1-ohm RMS Kurokawa reference is observational
    only: no port termination or port power is added to this transient DAE.
    """

    branch_count: int = eqx.field(static=True)
    prescribed_current: bool = eqx.field(static=True)
    boundary: AbstractMNAComponent
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, branch_count: int, /, *, boundary: AbstractMNAComponent | None = None
    ):
        topology = ThermalEquivalentCircuitPlan(branch_count)
        prescribed = boundary is None
        if prescribed:
            boundary = CircuitElement(
                IndependentCurrentSourceLaw(1.0, input_key="current_a"),
                element_id="battery-current-source",
            )
        if not isinstance(boundary, AbstractMNAComponent) or boundary.terminal_count != 2:
            raise TypeError(
                "Circuit ECM boundary must be a native two-terminal component."
            )
        law = implicit_law_for(boundary)
        if not prescribed and (
            law.input_names
            or law.voltage_rate_dependent
            or "differential" in law.state_layout.roles
        ):
            raise ValueError("Custom cell boundary must be autonomous and algebraic.")
        self.branch_count, self.prescribed_current, self.boundary = (
            topology.branch_count,
            prescribed,
            boundary,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "battery-circuit-connected-ecm",
                "branches": branch_count,
                "boundary_law": law.law_id,
                "boundary_layout": law.state_layout.layout_id,
                "prescribed_current": prescribed,
                "nodes": ("negative", "positive"),
                "port_reference_ohm": 1.0,
            }
        )

    def prepare(self, /) -> PreparedCircuitConnectedEcm:
        return PreparedCircuitConnectedEcm(self)


class CircuitConnectedEcmInitialCondition(StrictModule):
    """Only physical q, w, T data; no caller-selected algebraic voltages/currents."""

    charge_c: Array
    polarization_voltages_v: Array
    temperature_k: Array
    relaxed: bool = eqx.field(static=True)

    def __init__(
        self,
        charge_c: ArrayLike,
        temperature_k: ArrayLike,
        polarization_voltages_v: ArrayLike | None = None,
        /,
        *,
        relaxed: bool = False,
    ):
        physical = ThermalEquivalentCircuitInitialCondition(
            charge_c, temperature_k, polarization_voltages_v, relaxed=relaxed
        )
        self.charge_c, self.temperature_k = physical.charge_c, physical.temperature_k
        self.polarization_voltages_v, self.relaxed = (
            physical.polarization_voltages_v,
            physical.relaxed,
        )


class CircuitConnectedEcmStateView(StrictModule):
    """No-copy view into the native flat state, including optional leading axes."""

    states: Array
    cell_start: int = eqx.field(static=True)
    cell_stop: int = eqx.field(static=True)

    @property
    def charge_c(self) -> Array:
        return self.states[..., self.cell_start]

    @property
    def polarization_voltages_v(self) -> Array:
        return self.states[..., self.cell_start + 1 : self.cell_stop - 2]

    @property
    def temperature_k(self) -> Array:
        return self.states[..., self.cell_stop - 2]

    @property
    def current_a(self) -> Array:
        return self.states[..., self.cell_stop - 1]

    @property
    def voltage_v(self) -> Array:
        return self.states[..., 0]

    @property
    def physical_state(self) -> ThermalEquivalentCircuitState:
        return _physical(self.states[..., self.cell_start : self.cell_stop])


def _graph_parameters(args: Any, /) -> ThermalGraphParameters:
    return ThermalGraphParameters((_parameters(args),), jnp.zeros((0,)))


def _source_inputs(time: Array, state: Array, args: Any, /) -> Array:
    return jnp.atleast_1d(args.current(time, state))


class _CircuitTrialValidity(StrictModule, NonTrainableState):
    start: int = eqx.field(static=True)
    stop: int = eqx.field(static=True)

    def __call__(self, time, state, state_rate, args, inputs, /):
        del inputs
        p = _parameters(args)
        physical = state[self.start : self.stop]
        soc = physical[0] / p.reference_capacity_c
        lower = jnp.maximum(0.0, p.open_circuit_voltage.support_bounds[0])
        upper = jnp.minimum(1.0, p.open_circuit_voltage.support_bounds[1])
        coordinate_valid = (
            jnp.isfinite(time)
            & jnp.all(jnp.isfinite(state))
            & jnp.all(jnp.isfinite(state_rate))
            & (soc >= lower)
            & (soc <= upper)
            & (physical[-2] > 0.0)
            & (p.series_resistance_ohm > 0.0)
        )
        # Property support includes interior gaps, not just the outer hull.
        return jax.lax.cond(
            coordinate_valid,
            lambda: (
                p.open_circuit_voltage.evaluate(soc).support
                & p.entropic_coefficient.evaluate(soc).support
            ),
            lambda: jnp.asarray(False),
        )


class _FixedCurrentRuntime(StrictModule):
    parameters: ThermalEquivalentCircuitParameters
    current_a: Array

    def current(self, time, state, /):
        del time, state
        return self.current_a

    def observed_current(self, time, /):
        del time
        return self.current_a


class PreparedCircuitConnectedEcm(StrictModule, NonTrainableState):
    """The augmented system is the only solve/init/event/replay/ledger owner."""

    plan: CircuitConnectedEcmPlan
    base: PreparedCircuitDAE
    thermal_graph: ThermalGraphPlan
    system: DifferentialAlgebraicSystem
    initialization: DAEInitializationSpec
    cell_start: int = eqx.field(static=True)
    cell_stop: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: CircuitConnectedEcmPlan, /):
        if not isinstance(plan, CircuitConnectedEcmPlan):
            raise TypeError("plan must be CircuitConnectedEcmPlan.")
        cell = CircuitElement(
            CircuitEcmImplicitLaw(plan.branch_count), element_id="battery-cell"
        )
        boundary_nodes = (
            ("negative", "positive")
            if plan.prescribed_current
            else ("positive", "negative")
        )
        circuit = NodalCircuit(
            (
                CircuitInstance("cell", cell, ("positive", "negative")),
                CircuitInstance("boundary", plan.boundary, boundary_nodes),
            ),
            (
                NodalPort(
                    "terminal", "positive", "negative", ElectricalWaveReference(1.0)
                ),
            ),
            ground="negative",
            nodes=("negative", "positive"),
            circuit_id=plan.plan_id,
        )
        base = prepare_circuit_dae(circuit)
        start, stop = base.plan.layout.instance_range("cell")
        graph = ThermalGraphPlan(("cell",))
        system = augment_circuit_thermal_graph(
            base,
            graph,
            (stop - 2,),
            _graph_parameters,
            inputs=_source_inputs if plan.prescribed_current else None,
            trial_validity=_CircuitTrialValidity(start, stop),
            trial_validity_id=f"{plan.plan_id}/finite-property-support",
        )
        fixed = tuple(role == "differential" for role in base.plan.layout.roles)
        self.plan, self.base, self.thermal_graph, self.system = plan, base, graph, system
        self.initialization = DAEInitializationSpec.from_masks(
            fixed, tuple(not v for v in fixed)
        )
        self.cell_start, self.cell_stop = start, stop
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-circuit-ecm",
                "plan": plan.plan_id,
                "system": system.system_id,
                "initialization": self.initialization.initialization_id,
            }
        )

    def state_view(self, states: ArrayLike, /) -> CircuitConnectedEcmStateView:
        values = jnp.asarray(states)
        if values.shape[-1:] != self.system.state_shape:
            raise ValueError("Circuit ECM state must end in the native flat state axis.")
        return CircuitConnectedEcmStateView(values, self.cell_start, self.cell_stop)

    def boundary_power(self, time, state, state_rate, args, /) -> Array:
        plan = self.base.plan
        law = plan.laws[1]
        start, stop = plan.layout.auxiliary_ranges[1]
        voltage = state[0]
        voltages = (
            jnp.stack((jnp.zeros_like(voltage), voltage))
            if self.plan.prescribed_current
            else jnp.stack((voltage, jnp.zeros_like(voltage)))
        )
        voltage_rates = (
            jnp.stack((jnp.zeros_like(voltage), state_rate[0]))
            if self.plan.prescribed_current
            else jnp.stack((state_rate[0], jnp.zeros_like(voltage)))
        )
        inputs = (
            _source_inputs(time, state, args)
            if self.plan.prescribed_current
            else jnp.zeros((0,))
        )
        evaluation = law.evaluate(
            time,
            voltages,
            voltage_rates,
            state[start:stop],
            state_rate[start:stop],
            inputs,
            args,
        )
        return jnp.dot(voltages, evaluation.terminal_currents)

    def diagnostics(self, time, state, state_rate, args, /) -> CircuitDAEDiagnostics:
        residual = self.system.evaluate(time, state, state_rate, args)
        view = self.state_view(state)
        power = view.voltage_v * view.current_a + self.boundary_power(
            time, state, state_rate, args
        )
        norms = jnp.stack(
            tuple(
                jnp.linalg.norm(residual[start:stop])
                for start, stop in self.base.plan.layout.auxiliary_ranges
            )
        )
        return CircuitDAEDiagnostics(
            residual,
            jnp.linalg.norm(residual),
            jnp.abs(residual[0]),
            norms,
            power,
            jnp.all(jnp.isfinite(residual)) & jnp.isfinite(power),
        )


def _quadratic_three_node_integral(
    x0: Array,
    x1: Array,
    x2: Array,
    y0: Array,
    y1: Array,
    y2: Array,
    /,
    *,
    final_interval: bool = False,
) -> Array:
    """Integrate one nonuniform three-node quadratic without unsafe padding."""
    h0, h1 = x1 - x0, x2 - x1
    safe_h0 = jnp.where(h0 > 0.0, h0, 1.0)
    safe_h1 = jnp.where(h1 > 0.0, h1, 1.0)
    total = safe_h0 + safe_h1
    if final_interval:
        weights = (
            -(safe_h1**3) / (6.0 * safe_h0 * total),
            safe_h1 * (3.0 * safe_h0 + safe_h1) / (6.0 * safe_h0),
            safe_h1 * (3.0 * safe_h0 + 2.0 * safe_h1) / (6.0 * total),
        )
    else:
        weights = (
            total * (2.0 * safe_h0 - safe_h1) / (6.0 * safe_h0),
            total**3 / (6.0 * safe_h0 * safe_h1),
            total * (2.0 * safe_h1 - safe_h0) / (6.0 * safe_h1),
        )
    return weights[0] * y0 + weights[1] * y1 + weights[2] * y2


def _sampled_integral(times: Array, values: Array, valid: Array, /) -> Array:
    """Composite quadratic integral over a valid prefix of sampled nodes."""
    count = jnp.sum(valid.astype(jnp.int32))
    if times.size == 1:
        return jnp.zeros_like(values[0])
    safe_times = jnp.where(valid, times, times[0])
    pairs = valid[:-1] & valid[1:]
    trapezoid = (safe_times[1] - safe_times[0]) * 0.5 * (values[0] + values[1])
    if times.size == 2:
        return jnp.where(jnp.all(pairs), trapezoid, 0.0)
    starts = jnp.arange(times.size - 2)
    triples = valid[:-2] & valid[1:-1] & valid[2:] & ((starts % 2) == 0)
    quadratic = _quadratic_three_node_integral(
        safe_times[:-2],
        safe_times[1:-1],
        safe_times[2:],
        values[:-2],
        values[1:-1],
        values[2:],
    )
    integral = jnp.sum(jnp.where(triples, quadratic, 0.0), axis=0)
    indices = jnp.maximum(count + jnp.asarray((-3, -2, -1)), 0)
    final = _quadratic_three_node_integral(
        safe_times[indices[0]],
        safe_times[indices[1]],
        safe_times[indices[2]],
        values[indices[0]],
        values[indices[1]],
        values[indices[2]],
        final_interval=True,
    )
    integral = integral + jnp.where((count >= 4) & ((count % 2) == 0), final, 0.0)
    return jnp.where((count == 2) & pairs[0], trapezoid, integral)


def _held_sampled_integral(
    times: Array,
    left_values: Array,
    right_values: Array,
    interval_values: Array,
    pairs: Array,
    /,
) -> Array:
    """Integrate held-input samples without sharing a stencil across a jump."""
    if pairs.size == 0:
        return jnp.zeros((), dtype=left_values.dtype)
    safe_times = jnp.where(
        jnp.concatenate((pairs[:1], pairs)),
        times,
        times[0],
    )
    indices = jnp.arange(pairs.size)
    previous_pair = jnp.concatenate((jnp.asarray((False,)), pairs[:-1]))
    previous_value = jnp.concatenate((interval_values[:1], interval_values[:-1]))
    starts = pairs & (~previous_pair | (interval_values != previous_value))
    run_start = jax.lax.associative_scan(jnp.maximum, jnp.where(starts, indices, 0))
    offset = indices - run_start
    next_pair = jnp.concatenate((pairs[1:], jnp.asarray((False,))))
    next_value = jnp.concatenate((interval_values[1:], interval_values[-1:]))
    same_next = next_pair & (interval_values == next_value)
    full = pairs[:-1] & same_next[:-1] & ((offset[:-1] % 2) == 0)
    full_quadratic = _quadratic_three_node_integral(
        safe_times[:-2],
        safe_times[1:-1],
        safe_times[2:],
        left_values[:-1],
        right_values[:-1],
        right_values[1:],
    )
    integral = jnp.sum(jnp.where(full, full_quadratic, 0.0), axis=0)
    final = pairs & ~same_next & ((offset % 2) == 0)
    first = final & (offset == 0)
    trapezoids = (safe_times[1:] - safe_times[:-1]) * 0.5 * (left_values + right_values)
    previous_left = jnp.concatenate((left_values[:1], left_values[:-1]))
    final_quadratic = _quadratic_three_node_integral(
        jnp.concatenate((safe_times[:1], safe_times[:-2])),
        safe_times[:-1],
        safe_times[1:],
        previous_left,
        left_values,
        right_values,
        final_interval=True,
    )
    integral = integral + jnp.sum(jnp.where(first, trapezoids, 0.0), axis=0)
    return integral + jnp.sum(jnp.where(final & ~first, final_quadratic, 0.0), axis=0)


class CircuitConnectedEcmLedger(StrictModule):
    """Cell-specific storage/heat and native circuit balance evidence.

    Energy integrals use piecewise-quadratic saved-state quadrature and their
    defects are reported, not confused with exact conservation proofs. Success
    requires the local charge/RC/thermal laws and circuit balances at native
    consistent rates, plus zero physical initialization correction. Thus a
    finite bad model fails.
    """

    charge_change_c: Array
    terminal_charge_c: Array
    charge_defect_c: Array
    terminal_energy_j: Array
    rc_energy_change_j: Array
    irreversible_heat_j: Array
    reversible_heat_j: Array
    ambient_heat_loss_j: Array
    thermal_energy_change_j: Array
    thermal_defect_j: Array
    maximum_kcl_defect_a: Array
    maximum_power_defect_w: Array
    maximum_thermal_defect_w: Array
    maximum_scaled_residual: Array
    physical_initialization_correction: Array
    successful: Array


class _CircuitGuard(StrictModule, NonTrainableState):
    prepared: PreparedCircuitConnectedEcm
    coordinate: int = eqx.field(static=True)

    def __call__(self, time, state, args, /):
        del time
        view, p = self.prepared.state_view(state), _parameters(args)
        soc = view.charge_c / p.reference_capacity_c
        lower, upper = _local_support_bounds(p, soc)
        if self.coordinate == 0:
            return soc - lower - 1e-8
        if self.coordinate == 1:
            return upper - soc - 1e-8
        return view.temperature_k - 1.0


class CircuitConnectedEcmAdapter(StrictModule, NonTrainableState):
    plan: CircuitConnectedEcmPlan
    model_id: str = eqx.field(static=True)
    equation_form: Literal["dae"] = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    observable_units: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, plan: CircuitConnectedEcmPlan, /):
        if not isinstance(plan, CircuitConnectedEcmPlan):
            raise TypeError("plan must be CircuitConnectedEcmPlan.")
        self.plan = plan
        self.model_id = "battery:ecm:circuit-connected-electrothermal"
        self.equation_form = "dae"
        self.observable_names, self.observable_units = (
            _OBSERVABLE_NAMES,
            _OBSERVABLE_UNITS,
        )

    def _check(self, prepared, parameters=None, /):
        if (
            not isinstance(prepared, PreparedCircuitConnectedEcm)
            or prepared.plan.plan_id != self.plan.plan_id
        ):
            raise ValueError("Prepared circuit ECM does not belong to this adapter.")
        if parameters is not None:
            if not isinstance(parameters, ThermalEquivalentCircuitParameters):
                raise TypeError(
                    "Circuit ECM parameters must be ThermalEquivalentCircuitParameters."
                )
            if parameters.branch_resistances_ohm.shape != (self.plan.branch_count,):
                raise ValueError("Circuit ECM branch parameters do not match topology.")

    def prepare(self, /) -> PreparedCircuitConnectedEcm:
        return self.plan.prepare()

    def initial_state(self, prepared_model, parameters, initial_condition, /) -> Array:
        self._check(prepared_model, parameters)
        if not isinstance(initial_condition, CircuitConnectedEcmInitialCondition):
            raise TypeError(
                "initial_condition must be CircuitConnectedEcmInitialCondition."
            )
        ic = initial_condition
        polarization = (
            jnp.zeros((self.plan.branch_count,))
            if ic.relaxed
            else ic.polarization_voltages_v
        )
        if polarization.shape != (self.plan.branch_count,):
            raise ValueError(
                "Initial RC polarization must match the prepared branch count."
            )
        physical = ThermalEquivalentCircuitState(
            ic.charge_c, polarization, ic.temperature_k
        )
        state = jnp.zeros(
            prepared_model.system.state_shape,
            dtype=jnp.result_type(
                physical.charge_c,
                physical.temperature_k,
                parameters.series_resistance_ohm,
            ),
        )
        start, stop = prepared_model.cell_start, prepared_model.cell_stop
        state = (
            state.at[start].set(ic.charge_c).at[start + 1 : stop - 2].set(polarization)
        )
        state = state.at[stop - 2].set(ic.temperature_k)
        valid = _CircuitTrialValidity(start, stop)(
            jnp.asarray(0.0), state, jnp.zeros_like(state), parameters, None
        )
        state = eqx.error_if(
            state,
            ~valid,
            "Circuit ECM physical initial condition is outside property support.",
        )
        return state.at[0].set(_model_terms(parameters, physical, jnp.asarray(0.0))[1])

    def problem(
        self, prepared_model, initial_state, runtime_inputs, /
    ) -> DifferentialAlgebraicProblem:
        self._check(prepared_model, _parameters(runtime_inputs))
        prepared_model.state_view(initial_state)
        return DifferentialAlgebraicProblem(
            prepared_model.system,
            initial_state,
            args=runtime_inputs,
            initialization=prepared_model.initialization,
            problem_id=canonical_fingerprint(
                {
                    "kind": "circuit-ecm-problem",
                    "prepared": prepared_model.prepared_id,
                    "parameter": _parameters(runtime_inputs).parameter_id,
                }
            ),
        )

    def native_guards(self, prepared_model, /) -> tuple[HybridGuardPlan, ...]:
        self._check(prepared_model)
        return tuple(
            HybridGuardPlan(
                _CircuitGuard(prepared_model, index),
                direction=-1,
                terminal=True,
                priority=index,
                guard_id=f"{prepared_model.prepared_id}/{label}",
            )
            for index, label in enumerate(("soc-lower", "soc-upper", "temperature-lower"))
        )

    def current(self, prepared_model, times_s, states, runtime_inputs, /) -> Array:
        del runtime_inputs
        self._check(prepared_model)
        result = prepared_model.state_view(states).current_a
        if result.shape != jnp.asarray(times_s).shape:
            raise ValueError("Circuit ECM times must match state leading axes.")
        return result

    def observe(
        self, prepared_model, times_s, states, runtime_inputs, /
    ) -> BatteryModelOutput:
        self._check(prepared_model, _parameters(runtime_inputs))
        view = prepared_model.state_view(states)
        times = jnp.asarray(times_s)
        if times.shape != view.charge_c.shape:
            raise ValueError("Circuit ECM times must match state leading axes.")
        p = _parameters(runtime_inputs)
        terms = _model_terms(p, view.physical_state, view.current_a)
        values = jnp.stack(
            (
                view.voltage_v,
                view.temperature_k,
                terms[0],
                view.charge_c,
                terms[2],
                view.voltage_v * view.current_a,
                terms[3],
                terms[4],
                terms[5],
                terms[6],
            ),
            axis=-1,
        )
        return BatteryModelOutput(
            values,
            terms[7] & jnp.isfinite(times) & jnp.all(jnp.isfinite(states), axis=-1),
        )

    def _ledger_single(
        self, prepared_model, native_solution, runtime_inputs, /
    ) -> CircuitConnectedEcmLedger:
        self._check(prepared_model, _parameters(runtime_inputs))
        p = _parameters(runtime_inputs)
        times, valid = native_solution.times, native_solution.valid
        if times.ndim != 1 or valid.shape != times.shape or times.size == 0:
            raise ValueError(
                "Circuit ECM ledger requires a nonempty native saved-time axis."
            )
        states = jnp.where(
            valid[:, None], native_solution.states, native_solution.initialization.state
        )
        rates = jnp.where(
            valid[:, None],
            native_solution.state_rates,
            native_solution.initialization.state_rate,
        )
        safe_times = jnp.where(valid, times, times[0])
        view = prepared_model.state_view(states)
        terms = _model_terms(p, view.physical_state, view.current_a)
        forcing = (
            native_solution.forcing_currents_a
            if isinstance(native_solution, BatteryDAESolution)
            else jax.vmap(runtime_inputs.observed_current)(safe_times)
        )
        forcing = jnp.where(valid, forcing, forcing[0])
        diagnostics = jax.vmap(
            lambda t, y, yd, current: prepared_model.diagnostics(
                t, y, yd, _FixedCurrentRuntime(p, current)
            )
        )(safe_times, states, rates, forcing)
        count = jnp.sum(valid.astype(jnp.int32))
        last = jnp.maximum(count - 1, 0)
        pairs = valid[:-1] & valid[1:]
        dt = jnp.where(pairs, safe_times[1:] - safe_times[:-1], 0.0)

        def integral(values):
            return _sampled_integral(safe_times, values, valid)

        dq = view.charge_c[last] - view.charge_c[0]
        if self.plan.prescribed_current:
            # Evaluate each held interval at both endpoints under that interval's
            # current. Quadratic stencils stop at jumps rather than mixing sides.
            midpoint = 0.5 * (safe_times[:-1] + safe_times[1:])
            currents = jax.vmap(lambda t, y: runtime_inputs.current(t, y))(
                midpoint, states[:-1]
            )
            left_terms = _model_terms(
                p,
                _physical(
                    states[:-1, prepared_model.cell_start : prepared_model.cell_stop]
                ),
                currents,
            )
            right_terms = _model_terms(
                p,
                _physical(
                    states[1:, prepared_model.cell_start : prepared_model.cell_stop]
                ),
                currents,
            )
            terminal_charge = jnp.sum(dt * currents)
            terminal_energy = _held_sampled_integral(
                safe_times,
                left_terms[1] * currents,
                right_terms[1] * currents,
                currents,
                pairs,
            )
            irreversible, reversible, ambient = tuple(
                _held_sampled_integral(
                    safe_times,
                    left_terms[index],
                    right_terms[index],
                    currents,
                    pairs,
                )
                for index in (3, 4, 5)
            )
            integral_support = jnp.all((~pairs) | (left_terms[7] & right_terms[7]))
        else:
            # Autonomous source/load current is circuit-owned, not a protocol value.
            terminal_charge = integral(view.current_a)
            terminal_energy = integral(view.voltage_v * view.current_a)
            irreversible, reversible, ambient = (
                integral(terms[3]),
                integral(terms[4]),
                integral(terms[5]),
            )
            integral_support = jnp.asarray(True)
        rc_energy = 0.5 * jnp.sum(
            p.branch_capacitances_f
            * (
                view.polarization_voltages_v[last] ** 2
                - view.polarization_voltages_v[0] ** 2
            )
        )
        thermal_energy = p.heat_capacity_j_per_k * (
            view.temperature_k[last] - view.temperature_k[0]
        )
        scaled = jnp.max(
            jnp.abs(diagnostics.residual / prepared_model.system.residual_scale), axis=-1
        )

        def maximum(values):
            return jnp.max(jnp.where(valid, jnp.abs(values), 0.0))

        kcl, power = maximum(diagnostics.kcl_norm), maximum(diagnostics.terminal_power)
        thermal = maximum(diagnostics.residual[:, prepared_model.cell_stop - 2])
        residual = maximum(scaled)
        fixed = jnp.asarray(prepared_model.initialization.fixed_state)
        correction = jnp.max(
            jnp.abs(
                jnp.where(fixed, native_solution.initialization.state_correction, 0.0)
            )
        )
        finite = jnp.all((~valid) | (diagnostics.finite & terms[7])) & integral_support
        prefix = jnp.all(valid == (jnp.arange(times.size) < count))
        rate_valid = jnp.all(
            (~valid[:, None]) | (~fixed[None, :]) | native_solution.rate_valid
        )
        successful = (
            jnp.all(native_solution.successful)
            & (count >= 1)
            & prefix
            & finite
            & rate_valid
            & jnp.all((~pairs) | (dt > 0.0))
            & native_solution.initialization.valid
            & (correction == 0.0)
            & (residual <= 1e-6)
            & (kcl <= 1e-6)
            & (power <= 1e-6 * (1.0 + maximum(view.voltage_v * view.current_a)))
            & (thermal <= 1e-6)
        )
        return CircuitConnectedEcmLedger(
            dq,
            terminal_charge,
            dq - terminal_charge,
            terminal_energy,
            rc_energy,
            irreversible,
            reversible,
            ambient,
            thermal_energy,
            thermal_energy - irreversible - reversible + ambient,
            kcl,
            power,
            thermal,
            residual,
            correction,
            jnp.asarray(successful, dtype=bool),
        )

    def _ledger_checked(self, prepared_model, native_solution, runtime_inputs, /):
        """Evaluate only real valid prefixes; unavailable quantities remain NaN."""
        unknown = jnp.asarray(jnp.nan, dtype=native_solution.states.dtype)
        unavailable = CircuitConnectedEcmLedger(
            charge_change_c=unknown,
            terminal_charge_c=unknown,
            charge_defect_c=unknown,
            terminal_energy_j=unknown,
            rc_energy_change_j=unknown,
            irreversible_heat_j=unknown,
            reversible_heat_j=unknown,
            ambient_heat_loss_j=unknown,
            thermal_energy_change_j=unknown,
            thermal_defect_j=unknown,
            maximum_kcl_defect_a=unknown,
            maximum_power_defect_w=unknown,
            maximum_thermal_defect_w=unknown,
            maximum_scaled_residual=unknown,
            physical_initialization_correction=unknown,
            successful=jnp.asarray(False),
        )
        return domain_cond(
            native_solution.initialization.valid & jnp.any(native_solution.valid),
            lambda _: self._ledger_single(
                prepared_model, native_solution, runtime_inputs
            ),
            lambda _: unavailable,
        )

    def ledger(
        self, prepared_model, native_solution, runtime_inputs, /
    ) -> CircuitConnectedEcmLedger:
        stitched = self._ledger_checked(prepared_model, native_solution, runtime_inputs)
        if not isinstance(native_solution, BatteryDAESolution):
            return stitched
        # Original segments retain both sides and their native rates at a jump.
        # Integrate those, not a sampled stencil spanning a stitched algebraic jump.
        active = native_solution.replay.segment_active
        ledgers = []
        for segment in native_solution.segments:
            forcing = runtime_inputs.current(
                segment.times[0], segment.initialization.state
            )
            ledgers.append(
                self._ledger_checked(
                    prepared_model,
                    segment,
                    _FixedCurrentRuntime(_parameters(runtime_inputs), forcing),
                )
            )

        def summed(select):
            return jnp.sum(
                jnp.where(active, jnp.stack(tuple(select(v) for v in ledgers)), 0.0)
            )

        def maximum(select):
            return jnp.maximum(
                select(stitched),
                jnp.max(
                    jnp.where(active, jnp.stack(tuple(select(v) for v in ledgers)), 0.0)
                ),
            )

        integrated = tuple(
            summed(select)
            for select in (
                lambda v: v.charge_change_c,
                lambda v: v.terminal_charge_c,
                lambda v: v.charge_defect_c,
                lambda v: v.terminal_energy_j,
                lambda v: v.rc_energy_change_j,
                lambda v: v.irreversible_heat_j,
                lambda v: v.reversible_heat_j,
                lambda v: v.ambient_heat_loss_j,
                lambda v: v.thermal_energy_change_j,
                lambda v: v.thermal_defect_j,
            )
        )
        defects = tuple(
            maximum(select)
            for select in (
                lambda v: v.maximum_kcl_defect_a,
                lambda v: v.maximum_power_defect_w,
                lambda v: v.maximum_thermal_defect_w,
                lambda v: v.maximum_scaled_residual,
                lambda v: v.physical_initialization_correction,
            )
        )
        restart_ok = jnp.asarray(True)
        for restart in native_solution.replay.restarts:
            restart_ok = restart_ok & (
                (~restart.active)
                | (restart.differential_state_unchanged & restart.initialization.valid)
            )
        successful = (
            stitched.successful
            & restart_ok
            & jnp.any(active)
            & jnp.all((~active) | jnp.stack(tuple(v.successful for v in ledgers)))
        )
        return CircuitConnectedEcmLedger(*integrated, *defects, successful)


__all__ = [
    "CircuitEcmImplicitLaw",
    "CircuitConnectedEcmPlan",
    "PreparedCircuitConnectedEcm",
    "CircuitConnectedEcmInitialCondition",
    "CircuitConnectedEcmStateView",
    "CircuitConnectedEcmLedger",
    "CircuitConnectedEcmAdapter",
]
