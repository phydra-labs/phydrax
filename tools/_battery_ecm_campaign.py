#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native thermal ECM current/rest experiment and closed-form observation evaluator."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications.battery._ecm import (
    ThermalEquivalentCircuitAdapter,
    ThermalEquivalentCircuitInitialCondition,
    ThermalEquivalentCircuitParameters,
    ThermalEquivalentCircuitPlan,
)
from phydrax.applications.battery._experiment import (
    BatteryDiffraxSolvePlan,
    BatteryExperimentPlan,
    BatteryOutputPlan,
)
from phydrax.applications.battery._properties import ConstantPropertyLaw
from phydrax.applications.battery._protocol import (
    BatteryProtocolPlan,
    BatteryProtocolValues,
    CurrentStepPlan,
    RestStepPlan,
)
from phydrax.applications.battery._qualification import (
    THERMAL_ECM_CANDIDATE,
    THERMAL_ECM_SUPPORT,
)
from phydrax.applications.battery._release_contracts import (
    CampaignCase,
    THERMAL_ECM_SCIENTIFIC_METRICS,
)
from phydrax.applications.battery._results import BatteryExperimentResult
from tools.battery_campaign_registry import (
    CampaignEntry,
    metric_key,
    PreparedCampaign,
)


def _constant_law(value: float, /, *, quantity: str, unit: str) -> ConstantPropertyLaw:
    return ConstantPropertyLaw(
        value,
        (0.0, 1.0),
        quantity=quantity,
        coordinate="state_of_charge",
        value_unit=unit,
        coordinate_unit="1",
        source_id=f"battery-qualification:{quantity}",
    )


def prepare_campaign(sample_times_s: Sequence[float], /) -> PreparedCampaign:
    """Prepare the exact finite native-solver ECM current/rest campaign."""
    kind = "ecm-analytic"
    times = jnp.asarray(tuple(sample_times_s))
    duration = float(times[-1])
    transition_time = 0.5 * duration
    plan = ThermalEquivalentCircuitPlan(2)
    adapter = ThermalEquivalentCircuitAdapter(plan)
    parameters = ThermalEquivalentCircuitParameters(
        0.05,
        jnp.asarray((0.1, 0.2)),
        jnp.asarray((10.0, 20.0)),
        1000.0,
        100.0,
        0.5,
        300.0,
        300.0,
        _constant_law(3.7, quantity="reference-open-circuit-voltage", unit="V"),
        _constant_law(0.0, quantity="entropic-coefficient", unit="V/K"),
    )
    current = jnp.asarray(2.0, dtype=times.dtype)
    protocol = BatteryProtocolPlan(
        (
            CurrentStepPlan(transition_time, label="analytic-current"),
            RestStepPlan(duration - transition_time, label="analytic-rest"),
        ),
        node_side="right",
    )
    output_names = (
        "voltage_v",
        "current_a",
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
    experiment = BatteryExperimentPlan(
        adapter,
        protocol,
        BatteryOutputPlan(output_names),
        BatteryDiffraxSolvePlan(
            dt0=min(0.001, duration / 1000.0),
            relative_tolerance=1.0e-11,
            absolute_tolerance=1.0e-13,
            maximum_steps=16384,
        ),
        times,
        THERMAL_ECM_CANDIDATE,
        THERMAL_ECM_SUPPORT,
    ).prepare()
    protocol_values = BatteryProtocolValues(protocol, jnp.asarray((current,)))
    initial_charge = jnp.asarray(400.0, dtype=times.dtype)
    steady_polarization = parameters.branch_resistances_ohm * current
    current_heat = current**2 * (
        parameters.series_resistance_ohm + jnp.sum(parameters.branch_resistances_ohm)
    )
    steady_temperature = (
        parameters.ambient_temperature_k
        + current_heat / parameters.thermal_conductance_w_per_k
    )
    initial_condition = ThermalEquivalentCircuitInitialCondition(
        initial_charge,
        steady_temperature,
        steady_polarization,
    )

    def analytic_state(query_times: jax.Array) -> tuple[jax.Array, ...]:
        rest_time = jnp.maximum(query_times - transition_time, 0.0)
        in_rest = query_times >= transition_time
        time_constants = (
            parameters.branch_resistances_ohm * parameters.branch_capacitances_f
        )
        rest_polarization = steady_polarization[None, :] * jnp.exp(
            -rest_time[:, None] / time_constants[None, :]
        )
        polarization = jnp.where(
            in_rest[:, None],
            rest_polarization,
            steady_polarization[None, :],
        )
        charge = initial_charge + current * jnp.minimum(query_times, transition_time)
        cooling_rate = (
            parameters.thermal_conductance_w_per_k / parameters.heat_capacity_j_per_k
        )
        decay_rates = 2.0 / time_constants
        forcing = (
            current**2
            * parameters.branch_resistances_ohm
            / parameters.heat_capacity_j_per_k
        )
        thermal_terms = (
            forcing[None, :]
            * (
                jnp.exp(-decay_rates[None, :] * rest_time[:, None])
                - jnp.exp(-cooling_rate * rest_time[:, None])
            )
            / (cooling_rate - decay_rates[None, :])
        )
        initial_temperature_rise = steady_temperature - parameters.ambient_temperature_k
        rest_temperature = parameters.ambient_temperature_k + (
            initial_temperature_rise * jnp.exp(-cooling_rate * rest_time)
            + jnp.sum(thermal_terms, axis=-1)
        )
        temperature = jnp.where(in_rest, rest_temperature, steady_temperature)
        return charge, polarization, temperature

    def expected_outputs(query_times: jax.Array) -> jax.Array:
        charge, polarization, temperature = analytic_state(query_times)
        observed_current = jnp.where(query_times < transition_time, current, 0.0)
        polarization_sum = jnp.sum(polarization, axis=-1)
        voltage = (
            3.7 + observed_current * parameters.series_resistance_ohm + polarization_sum
        )
        irreversible_heat = (
            observed_current** 2 * parameters.series_resistance_ohm
            + jnp.sum(
                polarization**2 / parameters.branch_resistances_ohm,
                axis=-1,
            )
        )
        reversible_heat = jnp.zeros_like(query_times)
        cooling_power = parameters.thermal_conductance_w_per_k * (
            temperature - parameters.ambient_temperature_k
        )
        net_heat = irreversible_heat - cooling_power
        return jnp.stack(
            (
                voltage,
                observed_current,
                temperature,
                charge / parameters.reference_capacity_c,
                charge,
                polarization_sum,
                voltage * observed_current,
                irreversible_heat,
                reversible_heat,
                cooling_power,
                net_heat,
            ),
            axis=-1,
        )

    output_scales = jnp.asarray(
        (4.0, 2.0, 300.0, 1.0, 1000.0, 1.0, 10.0, 10.0, 1.0, 10.0, 10.0),
        dtype=times.dtype,
    )

    def execute() -> tuple[BatteryExperimentResult, jax.Array]:
        result = experiment.run(parameters, initial_condition, protocol_values)
        expected = expected_outputs(result.outputs.times_s)
        output_residuals = jnp.max(
            jnp.abs(result.outputs.values - expected) / output_scales,
            axis=-1,
        )
        native_times = result.native_solution.times
        expected_charge, expected_polarization, expected_temperature = analytic_state(
            native_times
        )
        state_residual = jnp.max(
            jnp.stack(
                (
                    jnp.max(
                        jnp.abs(result.native_solution.states.charge_c - expected_charge)
                        / parameters.reference_capacity_c
                    ),
                    jnp.max(
                        jnp.abs(
                            result.native_solution.states.polarization_voltages_v
                            - expected_polarization
                        )
                    ),
                    jnp.max(
                        jnp.abs(
                            result.native_solution.states.temperature_k
                            - expected_temperature
                        )
                        / parameters.ambient_temperature_k
                    ),
                )
            )
        )
        pair_widths = native_times[1:] - native_times[:-1]
        pair_midpoints = 0.5 * (native_times[:-1] + native_times[1:])
        pair_current = jnp.where(pair_midpoints < transition_time, current, 0.0)

        def integrate_pairs(left: jax.Array, right: jax.Array) -> jax.Array:
            return jnp.sum(0.5 * (left + right) * pair_widths)

        expected_charge_change = expected_charge[-1] - expected_charge[0]
        expected_terminal_charge = current * transition_time
        expected_rc_energy_change = 0.5 * jnp.sum(
            parameters.branch_capacitances_f
            * (expected_polarization[-1] ** 2 - expected_polarization[0] ** 2)
        )
        endpoint_branch_heat = jnp.sum(
            expected_polarization**2 / parameters.branch_resistances_ohm,
            axis=-1,
        )
        pair_series_heat = pair_current**2 * parameters.series_resistance_ohm
        expected_resistive = integrate_pairs(
            endpoint_branch_heat[:-1] + pair_series_heat,
            endpoint_branch_heat[1:] + pair_series_heat,
        )
        endpoint_cooling = parameters.thermal_conductance_w_per_k * (
            expected_temperature - parameters.ambient_temperature_k
        )
        expected_ambient_loss = integrate_pairs(
            endpoint_cooling[:-1], endpoint_cooling[1:]
        )
        expected_thermal_change = parameters.heat_capacity_j_per_k * (
            expected_temperature[-1] - expected_temperature[0]
        )
        expected_ledger = jnp.stack(
            (
                expected_charge_change,
                expected_terminal_charge,
                expected_charge_change - expected_terminal_charge,
                expected_rc_energy_change,
                expected_resistive,
                expected_thermal_change,
                expected_resistive,
                jnp.asarray(0.0, dtype=times.dtype),
                expected_ambient_loss,
                expected_thermal_change - (expected_resistive - expected_ambient_loss),
            )
        )
        observed_ledger = jnp.stack(
            (
                result.ledger.charge_change_c,
                result.ledger.terminal_charge_c,
                result.ledger.charge_defect_c,
                result.ledger.rc_energy_change_j,
                result.ledger.resistive_dissipation_j,
                result.ledger.thermal_energy_change_j,
                result.ledger.irreversible_heat_j,
                result.ledger.reversible_heat_j,
                result.ledger.ambient_heat_loss_j,
                result.ledger.thermal_defect_j,
            )
        )
        ledger_residual = jnp.max(
            jnp.abs(observed_ledger - expected_ledger)
            / jnp.maximum(jnp.abs(expected_ledger), 1.0)
        )
        global_residual = jnp.maximum(state_residual, ledger_residual)
        return result, jnp.maximum(output_residuals, global_residual)

    parameter_id = canonical_fingerprint(
        {
            "kind": "battery-qualification-ecm-parameters",
            "series_resistance_ohm": 0.05,
            "branch_resistances_ohm": [0.1, 0.2],
            "branch_capacitances_f": [10.0, 20.0],
            "reference_capacity_c": 1000.0,
            "heat_capacity_j_per_k": 100.0,
            "thermal_conductance_w_per_k": 0.5,
            "ambient_temperature_k": 300.0,
            "reference_temperature_k": 300.0,
            "open_circuit_voltage_v": 3.7,
            "entropic_coefficient_v_per_k": 0.0,
            "model_parameter_id": parameters.parameter_id,
        }
    )
    return PreparedCampaign(
        kind,
        experiment.preparation_id,
        parameter_id,
        adapter.model_id,
        execute,
        execute,
        resource_observations,
    )


def resource_observations(executed, /) -> dict[str, int | None]:
    result, _ = executed
    stats = result.native_solution.stats
    attempted = stats.get("num_steps")
    return {
        "unsuccessful-executions": int(not bool(np.asarray(result.successful))),
        "solver-iterations": None if attempted is None else int(np.asarray(attempted)),
    }


def raw_output(spec, executed, campaign_directory: Path, /) -> dict[str, object]:
    result, residuals = executed
    host_residuals = np.asarray(residuals, dtype=float)
    if host_residuals.shape != (len(spec.planned_schedule.sample_times_s),):
        raise ValueError("Built-in campaign returned an unexpected raw output shape.")
    if not result.evidence_ready:
        raise ValueError("Concrete battery execution did not seal a run identity.")
    samples = [0.0 if value == 0.0 else float(value) for value in host_residuals]
    ledger = {
        "charge_change_c": float(np.asarray(result.ledger.charge_change_c)),
        "terminal_charge_c": float(np.asarray(result.ledger.terminal_charge_c)),
        "charge_defect_c": float(np.asarray(result.ledger.charge_defect_c)),
        "rc_energy_change_j": float(np.asarray(result.ledger.rc_energy_change_j)),
        "resistive_dissipation_j": float(
            np.asarray(result.ledger.resistive_dissipation_j)
        ),
        "thermal_energy_change_j": float(
            np.asarray(result.ledger.thermal_energy_change_j)
        ),
        "irreversible_heat_j": float(np.asarray(result.ledger.irreversible_heat_j)),
        "reversible_heat_j": float(np.asarray(result.ledger.reversible_heat_j)),
        "ambient_heat_loss_j": float(np.asarray(result.ledger.ambient_heat_loss_j)),
        "thermal_defect_j": float(np.asarray(result.ledger.thermal_defect_j)),
        "successful": bool(np.asarray(result.ledger.successful)),
    }
    terminated = bool(np.asarray(result.termination.terminated))
    termination_time = float(np.asarray(result.termination.time_s))
    if terminated and not math.isfinite(termination_time):
        raise ValueError("Terminated battery execution has no finite event time.")
    usable = bool(np.asarray(result.successful)) and bool(
        np.all(np.isfinite(host_residuals))
    )
    maximum = max(samples) if usable else None
    content: dict[str, object] = {
        "kind": "battery-qualification-raw-output",
        "campaign_kind": spec.campaign_kind,
        "campaign_spec_id": spec.campaign_spec_id,
        "schedule_id": spec.planned_schedule.schedule_id,
        "sample_times_s": list(spec.planned_schedule.sample_times_s),
        "sample_residuals": samples,
        "execution": {
            "run_id": result.require_evidence_identity(),
            "experiment_plan_id": result.experiment_plan_id,
            "parameters_digest": result.parameters_digest,
            "protocol_values_digest": result.protocol_values_digest,
            "initial_state_digest": result.initial_state_digest,
            "preparation_id": result.preparation_id,
            "protocol_id": result.protocol_id,
            "model_id": result.model_id,
            "profile_id": result.profile_id,
            "support_tuple_id": result.support_tuple_id,
            "problem_id": result.problem_id,
            "application_status": int(np.asarray(result.application_status)),
            "termination": {
                "terminated": terminated,
                "time_s": termination_time if terminated else None,
                "reason": result.termination.reason,
            },
        },
        "outputs": {
            "names": list(result.outputs.names),
            "units": list(result.outputs.units),
            "times_s": np.asarray(result.outputs.times_s, dtype=float).tolist(),
            "values": np.asarray(result.outputs.values, dtype=float).tolist(),
            "valid": np.asarray(result.outputs.valid, dtype=bool).tolist(),
        },
        "native": {
            "solver_name": result.native_solution.solver_name,
            "resolved_method": result.native_solution.resolved_method,
            "valid": np.asarray(result.native_solution.valid, dtype=bool).tolist(),
            "charge_c": np.asarray(
                result.native_solution.states.charge_c, dtype=float
            ).tolist(),
            "polarization_voltages_v": np.asarray(
                result.native_solution.states.polarization_voltages_v,
                dtype=float,
            ).tolist(),
            "temperature_k": np.asarray(
                result.native_solution.states.temperature_k, dtype=float
            ).tolist(),
        },
        "ledger": ledger,
        "metrics": {
            metric_key("ecm-analytic", "maximum-normalized-analytic-residual"): {
                "value": maximum,
                "unavailable_reason": None if usable else "native-output-unusable",
            },
            metric_key("ecm-analytic", "application-status-success"): {
                "value": int(bool(np.asarray(result.successful))),
                "unavailable_reason": None,
            },
            metric_key("ecm-analytic", "model-ledger-success"): {
                "value": int(bool(np.asarray(result.ledger.successful))),
                "unavailable_reason": None,
            },
        },
    }
    return _finite_observation(content)


def _finite_observation(value):
    # Missing scientific values remain null, never substituted with a measured zero.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_observation(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_observation(item) for item in value]
    return value


def campaign_entry() -> CampaignEntry:
    return CampaignEntry(
        "ecm-analytic",
        THERMAL_ECM_CANDIDATE,
        THERMAL_ECM_SUPPORT,
        (
            CampaignCase(
                "ecm-analytic",
                (
                    ("branches", 2),
                    ("current_a", 2.0),
                    ("current_fraction", 0.5),
                    ("initial_charge_c", 400.0),
                    ("initial_thermal_state", "current-steady-state"),
                    ("initial_rc_state", "current-steady-state"),
                    ("protocol", "current-then-rest;right-continuous"),
                    ("sample_times", "exact-campaign-schedule;2..4096"),
                    ("maximum_steps", 16384),
                ),
                THERMAL_ECM_SCIENTIFIC_METRICS,
            ),
        ),
        0,
        (),
        prepare_campaign,
        raw_output,
        (
            "kind",
            "campaign_kind",
            "campaign_spec_id",
            "schedule_id",
            "sample_times_s",
            "sample_residuals",
            "execution",
            "outputs",
            "native",
            "ledger",
            "metrics",
        ),
    )
