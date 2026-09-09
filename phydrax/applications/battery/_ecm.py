#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import DifferentialProblem
from ._experiment import BatteryRuntimeInputs
from ._properties import ConstantPropertyLaw, TabulatedPropertyLaw
from ._results import BatteryModelOutput


_PropertyLaw: TypeAlias = ConstantPropertyLaw | TabulatedPropertyLaw
_PROPERTY_TYPES = (ConstantPropertyLaw, TabulatedPropertyLaw)

_OBSERVABLE_NAMES = (
    "voltage_v",
    "temperature_k",
    "state_of_charge",
    "charge_c",
    "polarization_voltage_sum_v",
    "terminal_power_w",
    "irreversible_heat_w",
    "reversible_heat_w",
    "cooling_power_w",
    "net_heat_w",
)
_OBSERVABLE_UNITS = ("V", "K", "1", "C", "V", "W", "W", "W", "W", "W")


def _real_array(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if not jnp.issubdtype(result.dtype, jnp.inexact):
        result = result.astype(float)
    return result


def _positive_scalar(value: ArrayLike, name: str, /) -> Array:
    result = _real_array(value, name)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(
        result,
        ~jnp.isfinite(result) | (result <= 0.0),
        f"{name} must be finite and positive.",
    )


def _positive_vector(value: ArrayLike, name: str, /) -> Array:
    result = _real_array(value, name)
    if result.ndim != 1 or int(result.size) < 1:
        raise ValueError(f"{name} must be a nonempty rank-one array.")
    return eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result) | (result <= 0.0)),
        f"{name} must contain only finite positive values.",
    )


def _property_law(
    value: _PropertyLaw,
    name: str,
    /,
    *,
    value_unit: str,
) -> _PropertyLaw:
    if not isinstance(value, _PROPERTY_TYPES):
        raise TypeError(f"{name} must be ConstantPropertyLaw or TabulatedPropertyLaw.")
    if value.coordinate != "state_of_charge" or value.coordinate_unit != "1":
        raise ValueError(f"{name} must use the dimensionless state_of_charge coordinate.")
    if value.value_unit != value_unit:
        raise ValueError(f"{name} must use value unit {value_unit!r}.")
    return value


class ThermalEquivalentCircuitPlan(StrictModule, NonTrainableState):
    """Static branch topology for the prescribed-current thermal ECM."""

    branch_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, branch_count: int, /):
        if isinstance(branch_count, bool) or not isinstance(branch_count, int):
            raise TypeError("Thermal ECM branch_count must be an integer.")
        if branch_count < 1:
            raise ValueError("Thermal ECM requires at least one RC branch.")
        self.branch_count = branch_count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "battery-thermal-equivalent-circuit-plan",
                "branch_count": branch_count,
                "state": ["charge_c", "polarization_voltages_v", "temperature_k"],
                "control": "prescribed-current-and-rest",
                "terminal_current_sign": "positive-enters-positive-terminal",
                "charge_evolution": "dq/dt=I",
                "temperature_model": "lumped-outward-cooling",
                "coulombic_efficiency": "ideal",
                "hysteresis": False,
                "capacity_fade": False,
            }
        )

    def prepare(self, /) -> PreparedThermalEquivalentCircuit:
        return PreparedThermalEquivalentCircuit(self)


class ThermalEquivalentCircuitParameters(StrictModule):
    """Dynamic SI parameters and bounded SOC property laws for one ECM profile."""

    series_resistance_ohm: Array
    branch_resistances_ohm: Array
    branch_capacitances_f: Array
    reference_capacity_c: Array
    heat_capacity_j_per_k: Array
    thermal_conductance_w_per_k: Array
    ambient_temperature_k: Array
    reference_temperature_k: Array
    open_circuit_voltage: _PropertyLaw
    entropic_coefficient: _PropertyLaw
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        series_resistance_ohm: ArrayLike,
        branch_resistances_ohm: ArrayLike,
        branch_capacitances_f: ArrayLike,
        reference_capacity_c: ArrayLike,
        heat_capacity_j_per_k: ArrayLike,
        thermal_conductance_w_per_k: ArrayLike,
        ambient_temperature_k: ArrayLike,
        reference_temperature_k: ArrayLike,
        open_circuit_voltage: _PropertyLaw,
        entropic_coefficient: _PropertyLaw,
        /,
    ):
        series = _positive_scalar(series_resistance_ohm, "series_resistance_ohm")
        resistances = _positive_vector(branch_resistances_ohm, "branch_resistances_ohm")
        capacitances = _positive_vector(branch_capacitances_f, "branch_capacitances_f")
        if resistances.shape != capacitances.shape:
            raise ValueError(
                "Thermal ECM branch resistance and capacitance shapes must match."
            )
        capacity = _positive_scalar(reference_capacity_c, "reference_capacity_c")
        heat_capacity = _positive_scalar(heat_capacity_j_per_k, "heat_capacity_j_per_k")
        conductance = _positive_scalar(
            thermal_conductance_w_per_k, "thermal_conductance_w_per_k"
        )
        ambient = _positive_scalar(ambient_temperature_k, "ambient_temperature_k")
        reference_temperature = _positive_scalar(
            reference_temperature_k, "reference_temperature_k"
        )
        ocv = _property_law(open_circuit_voltage, "open_circuit_voltage", value_unit="V")
        entropic = _property_law(
            entropic_coefficient, "entropic_coefficient", value_unit="V/K"
        )
        support_matches = jnp.all(ocv.support_bounds == entropic.support_bounds)
        series = eqx.error_if(
            series,
            ~support_matches,
            "Thermal ECM OCV and entropic laws must have identical SOC support.",
        )
        self.series_resistance_ohm = series
        self.branch_resistances_ohm = resistances
        self.branch_capacitances_f = capacitances
        self.reference_capacity_c = capacity
        self.heat_capacity_j_per_k = heat_capacity
        self.thermal_conductance_w_per_k = conductance
        self.ambient_temperature_k = ambient
        self.reference_temperature_k = reference_temperature
        self.open_circuit_voltage = ocv
        self.entropic_coefficient = entropic
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "battery-thermal-equivalent-circuit-parameters",
                "branch_count": int(resistances.size),
                "ocv_law_id": ocv.law_id,
                "entropic_law_id": entropic.law_id,
            }
        )


class ThermalEquivalentCircuitInitialCondition(StrictModule):
    """Stored charge, temperature, and explicit or explicitly relaxed RC state."""

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
        if not isinstance(relaxed, bool):
            raise TypeError("relaxed must be boolean.")
        if relaxed == (polarization_voltages_v is not None):
            raise ValueError(
                "Specify polarization_voltages_v or explicitly request relaxed=True, "
                "but not both."
            )
        charge = _real_array(charge_c, "charge_c")
        temperature = _real_array(temperature_k, "temperature_k")
        if charge.shape != () or temperature.shape != ():
            raise ValueError("Thermal ECM initial charge and temperature must be scalar.")
        charge = eqx.error_if(
            charge, ~jnp.isfinite(charge), "Initial stored charge must be finite."
        )
        temperature = eqx.error_if(
            temperature,
            ~jnp.isfinite(temperature),
            "Initial temperature must be finite.",
        )
        if relaxed:
            polarization = jnp.zeros((0,), dtype=jnp.result_type(charge, temperature))
        else:
            polarization = _real_array(polarization_voltages_v, "polarization_voltages_v")
            if polarization.ndim != 1:
                raise ValueError("Initial polarization voltages must be rank one.")
            polarization = eqx.error_if(
                polarization,
                jnp.any(~jnp.isfinite(polarization)),
                "Initial polarization voltages must be finite.",
            )
        self.charge_c = charge
        self.polarization_voltages_v = polarization
        self.temperature_k = temperature
        self.relaxed = relaxed


class ThermalEquivalentCircuitState(StrictModule):
    """Stored charge, branch polarization voltages, and lumped temperature."""

    charge_c: Array
    polarization_voltages_v: Array
    temperature_k: Array

    def __init__(
        self,
        charge_c: ArrayLike,
        polarization_voltages_v: ArrayLike,
        temperature_k: ArrayLike,
        /,
    ):
        charge = _real_array(charge_c, "charge_c")
        polarization = _real_array(polarization_voltages_v, "polarization_voltages_v")
        temperature = _real_array(temperature_k, "temperature_k")
        if temperature.shape != charge.shape:
            raise ValueError("Thermal ECM charge and temperature shapes must match.")
        if (
            polarization.ndim != charge.ndim + 1
            or polarization.shape[:-1] != charge.shape
        ):
            raise ValueError(
                "Thermal ECM polarization voltages require one final branch axis."
            )
        dtype = jnp.result_type(charge, polarization, temperature)
        self.charge_c = charge.astype(dtype)
        self.polarization_voltages_v = polarization.astype(dtype)
        self.temperature_k = temperature.astype(dtype)


class PreparedThermalEquivalentCircuit(StrictModule, NonTrainableState):
    """Prepared, nontrainable fixed-shape ECM topology."""

    plan: ThermalEquivalentCircuitPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ThermalEquivalentCircuitPlan, /):
        if not isinstance(plan, ThermalEquivalentCircuitPlan):
            raise TypeError("plan must be ThermalEquivalentCircuitPlan.")
        self.plan = plan
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-battery-thermal-equivalent-circuit",
                "plan_id": plan.plan_id,
            }
        )


class ThermalEquivalentCircuitLedger(StrictModule):
    """Integrated charge, RC storage, dissipation, and thermal balance evidence."""

    charge_change_c: Array
    terminal_charge_c: Array
    charge_defect_c: Array
    rc_energy_change_j: Array
    resistive_dissipation_j: Array
    thermal_energy_change_j: Array
    irreversible_heat_j: Array
    reversible_heat_j: Array
    ambient_heat_loss_j: Array
    thermal_defect_j: Array
    successful: Array

    def __init__(
        self,
        charge_change_c: ArrayLike,
        terminal_charge_c: ArrayLike,
        charge_defect_c: ArrayLike,
        rc_energy_change_j: ArrayLike,
        resistive_dissipation_j: ArrayLike,
        thermal_energy_change_j: ArrayLike,
        irreversible_heat_j: ArrayLike,
        reversible_heat_j: ArrayLike,
        ambient_heat_loss_j: ArrayLike,
        thermal_defect_j: ArrayLike,
        successful: ArrayLike,
        /,
    ):
        names = (
            "charge_change_c",
            "terminal_charge_c",
            "charge_defect_c",
            "rc_energy_change_j",
            "resistive_dissipation_j",
            "thermal_energy_change_j",
            "irreversible_heat_j",
            "reversible_heat_j",
            "ambient_heat_loss_j",
            "thermal_defect_j",
        )
        values = tuple(
            jnp.asarray(value)
            for value in (
                charge_change_c,
                terminal_charge_c,
                charge_defect_c,
                rc_energy_change_j,
                resistive_dissipation_j,
                thermal_energy_change_j,
                irreversible_heat_j,
                reversible_heat_j,
                ambient_heat_loss_j,
                thermal_defect_j,
            )
        )
        if any(value.shape != () for value in values):
            raise ValueError(
                f"Thermal ECM ledger field {names[0]!r} and peers must be scalar."
            )
        success = jnp.asarray(successful, dtype=bool)
        if success.shape != ():
            raise ValueError("Thermal ECM ledger successful flag must be scalar.")
        (
            self.charge_change_c,
            self.terminal_charge_c,
            self.charge_defect_c,
            self.rc_energy_change_j,
            self.resistive_dissipation_j,
            self.thermal_energy_change_j,
            self.irreversible_heat_j,
            self.reversible_heat_j,
            self.ambient_heat_loss_j,
            self.thermal_defect_j,
        ) = values
        self.successful = success


def _check_prepared(
    adapter_plan: ThermalEquivalentCircuitPlan,
    prepared: PreparedThermalEquivalentCircuit,
    /,
) -> None:
    if not isinstance(prepared, PreparedThermalEquivalentCircuit):
        raise TypeError("prepared_model must be PreparedThermalEquivalentCircuit.")
    if prepared.plan.plan_id != adapter_plan.plan_id:
        raise ValueError("Prepared thermal ECM topology does not belong to this adapter.")


def _check_parameters(
    prepared: PreparedThermalEquivalentCircuit,
    parameters: ThermalEquivalentCircuitParameters,
    /,
) -> None:
    if not isinstance(parameters, ThermalEquivalentCircuitParameters):
        raise TypeError("parameters must be ThermalEquivalentCircuitParameters.")
    if parameters.branch_resistances_ohm.shape != (prepared.plan.branch_count,):
        raise ValueError("Thermal ECM parameter branch count does not match preparation.")


def _check_state_shape(
    prepared: PreparedThermalEquivalentCircuit,
    state: ThermalEquivalentCircuitState,
    /,
) -> None:
    if not isinstance(state, ThermalEquivalentCircuitState):
        raise TypeError("state must be ThermalEquivalentCircuitState.")
    if state.polarization_voltages_v.shape[-1] != prepared.plan.branch_count:
        raise ValueError("Thermal ECM state branch count does not match preparation.")


def _model_terms(
    parameters: ThermalEquivalentCircuitParameters,
    state: ThermalEquivalentCircuitState,
    current_a: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    charge = state.charge_c
    polarization = state.polarization_voltages_v
    temperature = state.temperature_k
    state_of_charge = charge / parameters.reference_capacity_c
    ocv_reference = parameters.open_circuit_voltage.evaluate(state_of_charge)
    entropic = parameters.entropic_coefficient.evaluate(state_of_charge)
    ocv = (
        ocv_reference.values
        + (temperature - parameters.reference_temperature_k) * entropic.values
    )
    polarization_sum = jnp.sum(polarization, axis=-1)
    voltage = ocv + current_a * parameters.series_resistance_ohm + polarization_sum
    series_heat = current_a * current_a * parameters.series_resistance_ohm
    branch_heat = jnp.sum(
        polarization * polarization / parameters.branch_resistances_ohm,
        axis=-1,
    )
    irreversible_heat = series_heat + branch_heat
    reversible_heat = current_a * temperature * entropic.values
    cooling_power = parameters.thermal_conductance_w_per_k * (
        temperature - parameters.ambient_temperature_k
    )
    net_heat = irreversible_heat + reversible_heat - cooling_power
    finite = (
        jnp.isfinite(charge)
        & jnp.isfinite(temperature)
        & jnp.isfinite(current_a)
        & jnp.all(jnp.isfinite(polarization), axis=-1)
        & jnp.isfinite(ocv)
        & jnp.isfinite(voltage)
        & jnp.isfinite(net_heat)
    )
    support = (
        finite
        & (state_of_charge >= 0.0)
        & (state_of_charge <= 1.0)
        & (temperature > 0.0)
        & ocv_reference.support
        & entropic.support
    )
    return (
        state_of_charge,
        voltage,
        polarization_sum,
        irreversible_heat,
        reversible_heat,
        cooling_power,
        net_heat,
        support,
    )


def _thermal_ecm_drift(
    time_s: Array,
    state: ThermalEquivalentCircuitState,
    runtime_inputs: BatteryRuntimeInputs,
    /,
) -> ThermalEquivalentCircuitState:
    parameters = runtime_inputs.parameters
    if not isinstance(parameters, ThermalEquivalentCircuitParameters):
        raise TypeError("Thermal ECM runtime parameters have the wrong type.")
    current = runtime_inputs.current(time_s, state)
    terms = _model_terms(parameters, state, current)
    net_heat = terms[6]
    polarization_rate = (
        -state.polarization_voltages_v
        / (parameters.branch_resistances_ohm * parameters.branch_capacitances_f)
        + current / parameters.branch_capacitances_f
    )
    return ThermalEquivalentCircuitState(
        current,
        polarization_rate,
        net_heat / parameters.heat_capacity_j_per_k,
    )


def _trajectory_execution_currents(
    runtime_inputs: BatteryRuntimeInputs, times_s: Array, /
) -> Array:
    flat_times = times_s.reshape((-1,))
    currents = jax.vmap(
        lambda time: runtime_inputs.current(time, jnp.asarray(0.0, dtype=time.dtype))
    )(flat_times)
    return currents.reshape(times_s.shape)


def _trajectory_observed_currents(
    runtime_inputs: BatteryRuntimeInputs, times_s: Array, /
) -> Array:
    flat_times = times_s.reshape((-1,))
    currents = jax.vmap(runtime_inputs.observed_current)(flat_times)
    return currents.reshape(times_s.shape)


def _interval_heat_integrals(
    parameters: ThermalEquivalentCircuitParameters,
    states: ThermalEquivalentCircuitState,
    times_s: Array,
    pair_valid: Array,
    runtime_inputs: BatteryRuntimeInputs,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    interval_midpoints = 0.5 * (times_s[:-1] + times_s[1:])
    currents = _trajectory_execution_currents(
        runtime_inputs,
        interval_midpoints,
    )
    left_state = ThermalEquivalentCircuitState(
        states.charge_c[:-1],
        states.polarization_voltages_v[:-1],
        states.temperature_k[:-1],
    )
    right_state = ThermalEquivalentCircuitState(
        states.charge_c[1:],
        states.polarization_voltages_v[1:],
        states.temperature_k[1:],
    )
    left_terms = _model_terms(parameters, left_state, currents)
    right_terms = _model_terms(parameters, right_state, currents)
    duration = times_s[1:] - times_s[:-1]

    def integrate(left: Array, right: Array, /) -> Array:
        contribution = 0.5 * (left + right) * duration
        return jnp.sum(jnp.where(pair_valid, contribution, 0.0))

    irreversible = integrate(left_terms[3], right_terms[3])
    reversible = integrate(left_terms[4], right_terms[4])
    ambient_loss = integrate(left_terms[5], right_terms[5])
    net_heat = integrate(left_terms[6], right_terms[6])
    valid = jnp.all(
        (~pair_valid) | (left_terms[7] & right_terms[7] & jnp.isfinite(duration))
    )
    return irreversible, reversible, ambient_loss, net_heat, valid


class ThermalEquivalentCircuitAdapter(StrictModule, NonTrainableState):
    """Battery adapter for a thermal arbitrary-order prescribed-current ECM."""

    plan: ThermalEquivalentCircuitPlan
    model_id: str = eqx.field(static=True)
    equation_form: Literal["ode"] = eqx.field(static=True)
    observable_names: tuple[str, ...] = eqx.field(static=True)
    observable_units: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, plan: ThermalEquivalentCircuitPlan, /):
        if not isinstance(plan, ThermalEquivalentCircuitPlan):
            raise TypeError("plan must be ThermalEquivalentCircuitPlan.")
        self.plan = plan
        self.model_id = "battery:ecm:thermal-prescribed-current"
        self.equation_form = "ode"
        self.observable_names = _OBSERVABLE_NAMES
        self.observable_units = _OBSERVABLE_UNITS

    def prepare(self, /) -> PreparedThermalEquivalentCircuit:
        return self.plan.prepare()

    def initial_state(
        self,
        prepared_model: PreparedThermalEquivalentCircuit,
        parameters: ThermalEquivalentCircuitParameters,
        initial_condition: ThermalEquivalentCircuitInitialCondition,
        /,
    ) -> ThermalEquivalentCircuitState:
        _check_prepared(self.plan, prepared_model)
        _check_parameters(prepared_model, parameters)
        if not isinstance(initial_condition, ThermalEquivalentCircuitInitialCondition):
            raise TypeError(
                "initial_condition must be ThermalEquivalentCircuitInitialCondition."
            )
        if initial_condition.relaxed:
            polarization = jnp.zeros(
                (prepared_model.plan.branch_count,),
                dtype=jnp.result_type(
                    initial_condition.charge_c,
                    initial_condition.temperature_k,
                    parameters.branch_resistances_ohm,
                ),
            )
        else:
            polarization = initial_condition.polarization_voltages_v
            if polarization.shape != (prepared_model.plan.branch_count,):
                raise ValueError(
                    "Initial polarization branch count does not match preparation."
                )
        return ThermalEquivalentCircuitState(
            initial_condition.charge_c,
            polarization,
            initial_condition.temperature_k,
        )

    def problem(
        self,
        prepared_model: PreparedThermalEquivalentCircuit,
        initial_state: ThermalEquivalentCircuitState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> DifferentialProblem:
        _check_prepared(self.plan, prepared_model)
        _check_state_shape(prepared_model, initial_state)
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        _check_parameters(prepared_model, runtime_inputs.parameters)
        return DifferentialProblem(
            _thermal_ecm_drift,
            initial_state,
            t0=runtime_inputs.input_policy.times[0],
            t1=runtime_inputs.input_policy.times[-1],
            args=runtime_inputs,
            problem_id=canonical_fingerprint(
                {
                    "kind": "battery-thermal-equivalent-circuit-problem",
                    "model_id": self.model_id,
                    "prepared_id": prepared_model.prepared_id,
                    "parameter_id": runtime_inputs.parameters.parameter_id,
                    "protocol_id": runtime_inputs.protocol_id,
                }
            ),
        )

    def observe(
        self,
        prepared_model: PreparedThermalEquivalentCircuit,
        times_s: Array,
        states: ThermalEquivalentCircuitState,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> BatteryModelOutput:
        _check_prepared(self.plan, prepared_model)
        _check_state_shape(prepared_model, states)
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        parameters = runtime_inputs.parameters
        _check_parameters(prepared_model, parameters)
        times = _real_array(times_s, "times_s")
        if times.shape != states.charge_c.shape:
            raise ValueError("Thermal ECM times must match the state leading shape.")
        current = _trajectory_observed_currents(runtime_inputs, times)
        (
            state_of_charge,
            voltage,
            polarization_sum,
            irreversible_heat,
            reversible_heat,
            cooling_power,
            net_heat,
            support,
        ) = _model_terms(parameters, states, current)
        terminal_power = voltage * current
        values = jnp.stack(
            (
                voltage,
                states.temperature_k,
                state_of_charge,
                states.charge_c,
                polarization_sum,
                terminal_power,
                irreversible_heat,
                reversible_heat,
                cooling_power,
                net_heat,
            ),
            axis=-1,
        )
        domain_valid = support & jnp.isfinite(times) & jnp.isfinite(terminal_power)
        return BatteryModelOutput(values, domain_valid)

    def ledger(
        self,
        prepared_model: PreparedThermalEquivalentCircuit,
        native_solution: Any,
        runtime_inputs: BatteryRuntimeInputs,
        /,
    ) -> ThermalEquivalentCircuitLedger:
        _check_prepared(self.plan, prepared_model)
        if not isinstance(runtime_inputs, BatteryRuntimeInputs):
            raise TypeError("runtime_inputs must be BatteryRuntimeInputs.")
        parameters = runtime_inputs.parameters
        _check_parameters(prepared_model, parameters)
        states = native_solution.states
        _check_state_shape(prepared_model, states)
        times = jnp.asarray(native_solution.times)
        valid = jnp.asarray(native_solution.valid, dtype=bool)
        if times.ndim != 1 or valid.shape != times.shape:
            raise ValueError("Thermal ECM ledger requires one native saved-time axis.")
        if (
            states.charge_c.shape != times.shape
            or states.temperature_k.shape != times.shape
        ):
            raise ValueError(
                "Thermal ECM ledger state history does not match saved times."
            )
        if states.polarization_voltages_v.shape != (
            times.size,
            prepared_model.plan.branch_count,
        ):
            raise ValueError(
                "Thermal ECM ledger polarization history has the wrong shape."
            )

        count = jnp.sum(valid.astype(jnp.int32))
        expected_prefix = jnp.arange(times.size, dtype=jnp.int32) < count
        prefix_valid = jnp.all(valid == expected_prefix)
        safe_times = jnp.where(valid, times, runtime_inputs.input_policy.times[0])
        safe_charge = jnp.where(valid, states.charge_c, 0.0)
        safe_temperature = jnp.where(
            valid, states.temperature_k, parameters.ambient_temperature_k
        )
        safe_polarization = jnp.where(valid[:, None], states.polarization_voltages_v, 0.0)
        safe_states = ThermalEquivalentCircuitState(
            safe_charge, safe_polarization, safe_temperature
        )
        finite_state = (
            jnp.isfinite(states.charge_c)
            & jnp.isfinite(states.temperature_k)
            & jnp.all(jnp.isfinite(states.polarization_voltages_v), axis=-1)
        )
        sample_finite = jnp.isfinite(times) & finite_state
        pair_valid = valid[:-1] & valid[1:]
        increasing = jnp.all((~pair_valid) | (safe_times[1:] > safe_times[:-1]))
        (
            irreversible_integral,
            reversible_integral,
            ambient_loss,
            net_heat_integral,
            heat_valid,
        ) = _interval_heat_integrals(
            parameters,
            safe_states,
            safe_times,
            pair_valid,
            runtime_inputs,
        )
        sample_ok = jnp.all((~valid) | sample_finite) & heat_valid
        last_index = jnp.clip(count - 1, 0, times.size - 1)

        charge_change = safe_charge[last_index] - safe_charge[0]
        policy_times = runtime_inputs.input_policy.times
        interval_left = jnp.maximum(policy_times[:-1], safe_times[0])
        interval_right = jnp.minimum(policy_times[1:], safe_times[last_index])
        interval_duration = jnp.maximum(interval_right - interval_left, 0.0)
        terminal_charge = jnp.sum(
            interval_duration * runtime_inputs.input_policy.values[:, 0]
        )
        charge_defect = charge_change - terminal_charge
        initial_rc_energy = 0.5 * jnp.sum(
            parameters.branch_capacitances_f * safe_polarization[0] ** 2
        )
        final_rc_energy = 0.5 * jnp.sum(
            parameters.branch_capacitances_f * safe_polarization[last_index] ** 2
        )
        rc_energy_change = final_rc_energy - initial_rc_energy
        resistive_dissipation = irreversible_integral
        thermal_energy_change = parameters.heat_capacity_j_per_k * (
            safe_temperature[last_index] - safe_temperature[0]
        )
        thermal_defect = thermal_energy_change - net_heat_integral
        raw_values = jnp.stack(
            (
                charge_change,
                terminal_charge,
                charge_defect,
                rc_energy_change,
                resistive_dissipation,
                thermal_energy_change,
                irreversible_integral,
                reversible_integral,
                ambient_loss,
                thermal_defect,
            )
        )
        backend_successful = jnp.all(
            jnp.asarray(native_solution.backend_successful, dtype=bool)
        )
        successful = (
            backend_successful
            & (count >= 2)
            & prefix_valid
            & increasing
            & sample_ok
            & jnp.all(jnp.isfinite(raw_values))
        )
        values = jnp.where(successful, raw_values, jnp.zeros_like(raw_values))
        return ThermalEquivalentCircuitLedger(*values, successful)


__all__ = [
    "PreparedThermalEquivalentCircuit",
    "ThermalEquivalentCircuitAdapter",
    "ThermalEquivalentCircuitInitialCondition",
    "ThermalEquivalentCircuitLedger",
    "ThermalEquivalentCircuitParameters",
    "ThermalEquivalentCircuitPlan",
    "ThermalEquivalentCircuitState",
]
