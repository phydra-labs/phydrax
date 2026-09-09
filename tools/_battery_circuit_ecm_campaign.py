#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Circuit-ECM native DAE campaigns against closed-form RC/thermal solutions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications.battery._circuit_ecm import (
    CircuitConnectedEcmAdapter,
    CircuitConnectedEcmInitialCondition,
    CircuitConnectedEcmPlan,
)
from phydrax.applications.battery._ecm import ThermalEquivalentCircuitParameters
from phydrax.applications.battery._experiment import (
    BatteryDAESolvePlan,
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
from phydrax.applications.battery._release_contracts import (
    CampaignCase,
    CIRCUIT_ECM_CASE_IDS,
    CIRCUIT_ECM_SCIENTIFIC_METRICS,
)
from phydrax.circuit import CircuitElement, IndependentVoltageSourceLaw, Resistor
from phydrax.nonlinear import NonlinearTermination
from phydrax.solver import BDFMethod, DAEAdaptivePolicy, DAESolvePolicy
from tools.battery_campaign_registry import (
    CampaignEntry,
    metric_key,
    PreparedCampaign,
    trajectory_observation,
)


_KIND = "circuit-ecm-analytic"
_OUTPUTS = (
    "charge_c",
    "polarization_voltage_sum_v",
    "temperature_k",
    "current_a",
    "voltage_v",
)
_SCALES = np.asarray((1000.0, 1.0, 300.0, 2.0, 4.0))
_PARAMETERS = {
    "series_resistance_ohm": 0.05,
    "branch_resistance_ohm": 0.1,
    "branch_capacitance_f": 10.0,
    "reference_capacity_c": 1000.0,
    "heat_capacity_j_per_k": 100.0,
    "thermal_conductance_w_per_k": 0.5,
    "ambient_temperature_k": 300.0,
    "open_circuit_voltage_v": 3.7,
    "entropic_coefficient_v_per_k": 0.0,
    "initial_charge_c": 400.0,
    "initial_polarization_v": 0.0,
    "current_magnitude_a": 2.0,
    "load_resistance_ohm": 2.0,
    "clamp_voltage_v": 3.8,
    "duration_s": 2.0,
    "transition_s": 1.0,
}


def _law(value: float, quantity: str, unit: str, /) -> ConstantPropertyLaw:
    return ConstantPropertyLaw(
        value,
        jnp.asarray((0.0, 1.0)),
        value_bounds=jnp.asarray((-1.0, 5.0)),
        quantity=quantity,
        coordinate="state_of_charge",
        value_unit=unit,
        coordinate_unit="1",
        source_id="battery-circuit-analytic-self-authored-constant-law",
    )


def _parameters() -> ThermalEquivalentCircuitParameters:
    return ThermalEquivalentCircuitParameters(
        0.05,
        jnp.asarray((0.1,)),
        jnp.asarray((10.0,)),
        1000.0,
        100.0,
        0.5,
        300.0,
        300.0,
        _law(3.7, "reference-open-circuit-voltage", "V"),
        _law(0.0, "entropic-coefficient", "V/K"),
    )


def _analytic_segment(times, initial, a, b, /):
    """Exact one-RC solution for the physical boundary I = a + b w.

    The fixed cases avoid coincident thermal/electrical rates. This reference
    integrates the equations directly and never calls the production model law.
    """
    t = np.asarray(times, dtype=float)
    q0, w0, temperature0 = initial
    resistance, capacitance, series, heat_capacity, ambient_rate = (
        0.1,
        10.0,
        0.05,
        100.0,
        0.005,
    )
    rate = (1.0 / resistance - b) / capacitance
    equilibrium = a / (1.0 / resistance - b)
    amplitude = w0 - equilibrium
    exponential = np.exp(-rate * t)
    polarization = equilibrium + amplitude * exponential
    steady_current = a + b * equilibrium
    current_amplitude = b * amplitude
    current = steady_current + current_amplitude * exponential
    charge = q0 + steady_current * t - current_amplitude * np.expm1(-rate * t) / rate
    constant_heat = series * steady_current**2 + equilibrium**2 / resistance
    first_heat = 2.0 * (
        series * steady_current * current_amplitude + equilibrium * amplitude / resistance
    )
    second_heat = series * current_amplitude**2 + amplitude**2 / resistance

    def convolution(decay):
        return (np.exp(-decay * t) - np.exp(-ambient_rate * t)) / (ambient_rate - decay)

    temperature = 300.0 + (temperature0 - 300.0) * np.exp(-ambient_rate * t)
    temperature += (
        constant_heat * convolution(0.0)
        + first_heat * convolution(rate)
        + second_heat * convolution(2.0 * rate)
    ) / heat_capacity
    voltage = 3.7 + series * current + polarization
    return np.stack((charge, polarization, temperature, current, voltage), axis=-1)


def analytic_outputs(case: str, times, /) -> np.ndarray:
    initial = (400.0, 0.0, 300.0)
    times = np.asarray(times, dtype=float)
    if case in ("charge-rest", "discharge-rest"):
        current = 2.0 if case == "charge-rest" else -2.0
        before = _analytic_segment(np.minimum(times, 1.0), initial, current, 0.0)
        endpoint = _analytic_segment(np.asarray(1.0), initial, current, 0.0)
        after = _analytic_segment(
            np.maximum(times - 1.0, 0.0), tuple(endpoint[:3]), 0.0, 0.0
        )
        return np.where((times < 1.0)[..., None], before, after)
    if case == "resistive-load":
        return _analytic_segment(times, initial, -3.7 / 2.05, -1.0 / 2.05)
    if case == "voltage-clamp":
        return _analytic_segment(times, initial, (3.8 - 3.7) / 0.05, -1.0 / 0.05)
    raise ValueError(f"Unknown circuit analytic case: {case}")


def prepare_campaign(sample_times_s: Sequence[float], /) -> PreparedCampaign:
    times = np.asarray(tuple(sample_times_s), dtype=float)
    if (
        times.ndim != 1
        or times.size < 3
        or times[0] != 0.0
        or times[-1] != 2.0
        or 1.0 not in times
    ):
        raise ValueError(
            "Circuit campaign requires a [0,2] second schedule containing the transition at 1 second."
        )
    if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0.0):
        raise ValueError("Circuit campaign times must be finite and strictly increasing.")
    from phydrax.applications.battery._qualification import (
        CIRCUIT_ECM_CANDIDATE,
        CIRCUIT_ECM_SUPPORT,
    )

    parameters = _parameters()
    initial = CircuitConnectedEcmInitialCondition(400.0, 300.0, relaxed=True)
    experiments, values = [], []
    for case in CIRCUIT_ECM_CASE_IDS:
        boundary = None
        if case == "resistive-load":
            boundary = Resistor(2.0)
        elif case == "voltage-clamp":
            boundary = CircuitElement(
                IndependentVoltageSourceLaw(3.8), element_id="clamp"
            )
        adapter = CircuitConnectedEcmAdapter(
            CircuitConnectedEcmPlan(1, boundary=boundary)
        )
        native = adapter.prepare()
        guards = tuple(guard.guard_id for guard in adapter.native_guards(native))
        protocol = BatteryProtocolPlan(
            (CurrentStepPlan(1.0), RestStepPlan(1.0))
            if case in ("charge-rest", "discharge-rest")
            else (RestStepPlan(2.0),),
            node_side="right",
        )
        numerical = DAESolvePolicy(
            method=BDFMethod(2),
            nonlinear_termination=NonlinearTermination(
                absolute_step=0.0, relative_step=0.0
            ),
            adaptive=DAEAdaptivePolicy(
                relative_tolerance=1e-9,
                absolute_tolerance=1e-11,
                initial_step=0.001,
                maximum_step=0.02,
                maximum_accepted_steps=4096,
                maximum_attempts=8192,
            ),
        )
        experiment = BatteryExperimentPlan(
            adapter,
            protocol,
            BatteryOutputPlan(_OUTPUTS),
            BatteryDAESolvePlan(numerical, guard_ids=guards),
            jnp.asarray(times),
            CIRCUIT_ECM_CANDIDATE,
            CIRCUIT_ECM_SUPPORT,
        ).prepare()
        amplitude = (
            (2.0,)
            if case == "charge-rest"
            else (-2.0,)
            if case == "discharge-rest"
            else ()
        )
        experiments.append(experiment)
        values.append(BatteryProtocolValues(protocol, jnp.asarray(amplitude)))

    def execute():
        return tuple(
            experiment.run(parameters, initial, protocol_values)
            for experiment, protocol_values in zip(experiments, values, strict=True)
        )

    return PreparedCampaign(
        _KIND,
        canonical_fingerprint(
            {"preparations": [item.preparation_id for item in experiments]}
        ),
        canonical_fingerprint(_PARAMETERS),
        canonical_fingerprint(
            {"cases": CIRCUIT_ECM_CASE_IDS, "model": experiments[0].plan.model.model_id}
        ),
        execute,
        execute,
    )


def raw_output(spec, executed, campaign_directory: Path, /) -> dict[str, object]:
    del spec, campaign_directory
    if not isinstance(executed, tuple) or len(executed) != len(CIRCUIT_ECM_CASE_IDS):
        raise ValueError("Circuit campaign output must contain every ordered case.")
    metrics, observations = {}, []
    for case, result in zip(CIRCUIT_ECM_CASE_IDS, executed, strict=True):
        jax.block_until_ready(result)
        observed = np.asarray(result.outputs.values)
        times = np.asarray(result.outputs.times_s)
        valid_mask = np.asarray(result.outputs.valid)
        valid = bool(np.all(valid_mask))
        finite = bool(np.all(np.isfinite(observed)) and np.all(np.isfinite(times)))
        error = (
            float(np.max(np.abs(observed - analytic_outputs(case, times)) / _SCALES))
            if finite and valid
            else None
        )
        values = {
            "application-status": int(np.asarray(result.application_status)),
            "maximum-normalized-analytic-error": error,
            "maximum-kcl-defect-a": float(np.asarray(result.ledger.maximum_kcl_defect_a)),
            "maximum-power-defect-w": float(
                np.asarray(result.ledger.maximum_power_defect_w)
            ),
            "maximum-thermal-defect-w": float(
                np.asarray(result.ledger.maximum_thermal_defect_w)
            ),
            "charge-balance-defect-c": abs(
                float(np.asarray(result.ledger.charge_defect_c))
            ),
            "thermal-balance-defect-j": abs(
                float(np.asarray(result.ledger.thermal_defect_j))
            ),
            "physical-initialization-correction": float(
                np.asarray(result.ledger.physical_initialization_correction)
            ),
            "ledger-success": int(np.asarray(result.ledger.successful)),
        }
        for name, value in values.items():
            measured = value is not None and np.isfinite(value)
            metrics[metric_key(case, name)] = {
                "value": value if measured else None,
                "unavailable_reason": None
                if measured
                else "native-trajectory-or-metric-unavailable",
            }
        observations.append(
            trajectory_observation(
                case, result.require_evidence_identity(), result.outputs
            )
        )
    return {"metrics": metrics, "observations": observations}


def campaign_entry() -> CampaignEntry:
    from phydrax.applications.battery._qualification import (
        CIRCUIT_ECM_CANDIDATE,
        CIRCUIT_ECM_SUPPORT,
    )

    metrics = CIRCUIT_ECM_SCIENTIFIC_METRICS
    cases = tuple(
        CampaignCase(case, tuple(sorted(_PARAMETERS.items())), metrics)
        for case in CIRCUIT_ECM_CASE_IDS
    )
    return CampaignEntry(
        _KIND,
        CIRCUIT_ECM_CANDIDATE,
        CIRCUIT_ECM_SUPPORT,
        cases,
        0,
        (),
        prepare_campaign,
        raw_output,
        ("metrics", "observations"),
    )
