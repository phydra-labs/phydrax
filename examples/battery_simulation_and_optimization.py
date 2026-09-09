#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Simulate a thermal ECM and optimize current; require explicit execution admission."""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications import battery


_CURRENT_DURATION_S = 20.0
_TARGET_CHARGE_C = 540.0
_CURRENT_BOUNDS_A = (0.25, 3.5)
_TARGET_TOLERANCE_C = 5.0e-3


def _constant_law(
    value: float,
    *,
    quantity: str,
    unit: str,
    value_bounds: tuple[float, float],
) -> battery.ConstantPropertyLaw:
    return battery.ConstantPropertyLaw(
        jnp.asarray(value),
        jnp.asarray((0.0, 1.0)),
        value_bounds=jnp.asarray(value_bounds),
        quantity=quantity,
        coordinate="state_of_charge",
        value_unit=unit,
        coordinate_unit="1",
        source_id="example:bounded-thermal-ecm-law",
    )


def _prepared_experiment(
    profile: phx.qualification.CapabilityProfile,
) -> tuple[
    battery.PreparedBatteryExperiment,
    battery.BatteryProtocolPlan,
    battery.ThermalEquivalentCircuitParameters,
    battery.ThermalEquivalentCircuitInitialCondition,
]:
    protocol = battery.BatteryProtocolPlan(
        (
            battery.CurrentStepPlan(_CURRENT_DURATION_S, label="constant-current"),
            battery.RestStepPlan(10.0, label="rest"),
        ),
        node_side="right",
    )
    times = jnp.linspace(0.0, 30.0, 31)
    parameters = battery.ThermalEquivalentCircuitParameters(
        0.05,
        jnp.asarray((0.10, 0.20)),
        jnp.asarray((10.0, 20.0)),
        1000.0,
        200.0,
        0.5,
        298.15,
        298.15,
        _constant_law(
            3.7,
            quantity="reference-open-circuit-voltage",
            unit="V",
            value_bounds=(2.0, 5.0),
        ),
        _constant_law(
            0.0,
            quantity="entropic-coefficient",
            unit="V/K",
            value_bounds=(-0.01, 0.01),
        ),
    )
    initial = battery.ThermalEquivalentCircuitInitialCondition(
        500.0, 298.15, relaxed=True
    )
    prepared = battery.BatteryExperimentPlan(
        battery.ThermalEquivalentCircuitAdapter(battery.ThermalEquivalentCircuitPlan(2)),
        protocol,
        battery.BatteryOutputPlan(
            (
                "current_a",
                "voltage_v",
                "charge_c",
                "state_of_charge",
                "temperature_k",
                "terminal_power_w",
            )
        ),
        battery.BatteryDiffraxSolvePlan(
            dt0=0.05,
            relative_tolerance=1.0e-7,
            absolute_tolerance=1.0e-9,
            maximum_steps=4096,
        ),
        times,
        profile,
        battery.THERMAL_ECM_SUPPORT,
    ).prepare()
    return prepared, protocol, parameters, initial


def _run(
    prepared: battery.PreparedBatteryExperiment,
    protocol: battery.BatteryProtocolPlan,
    parameters: battery.ThermalEquivalentCircuitParameters,
    initial: battery.ThermalEquivalentCircuitInitialCondition,
    current_a: jnp.ndarray,
) -> battery.BatteryExperimentResult:
    values = battery.BatteryProtocolValues(
        protocol, jnp.reshape(jnp.asarray(current_a), (1,))
    )
    return prepared.run(parameters, initial, values)


def _require_run_margins(
    result: battery.BatteryExperimentResult, label: str, /
) -> dict[str, float]:
    status = int(np.asarray(result.application_status))
    if status != int(battery.BatteryRunStatus.SUCCESS):
        raise RuntimeError(f"{label} failed with status {status}.")
    if bool(np.asarray(result.termination.terminated)):
        raise RuntimeError(f"{label} terminated early: {result.termination.reason}.")
    if not bool(np.all(np.asarray(result.outputs.valid))):
        raise RuntimeError(f"{label} produced an invalid output sample.")
    if not bool(np.all(np.asarray(result.native_solution.valid))):
        raise RuntimeError(f"{label} produced an invalid native solution sample.")
    if not bool(np.asarray(result.ledger.successful)):
        raise RuntimeError(f"{label} produced an unsuccessful conservation ledger.")

    names = result.outputs.names
    values = np.asarray(result.outputs.values)
    state_of_charge = values[:, names.index("state_of_charge")]
    temperature_k = values[:, names.index("temperature_k")]
    thermal_scale = max(
        abs(float(np.asarray(result.ledger.thermal_energy_change_j))),
        abs(float(np.asarray(result.ledger.irreversible_heat_j))),
        abs(float(np.asarray(result.ledger.reversible_heat_j))),
        abs(float(np.asarray(result.ledger.ambient_heat_loss_j))),
        1.0,
    )
    margins = {
        "soc-lower": float(np.min(state_of_charge)),
        "soc-upper": float(1.0 - np.max(state_of_charge)),
        "temperature-positive-k": float(np.min(temperature_k)),
        "charge-balance-c": 1.0e-3
        - abs(float(np.asarray(result.ledger.charge_defect_c))),
        "normalized-thermal-balance": 5.0e-2
        - abs(float(np.asarray(result.ledger.thermal_defect_j))) / thermal_scale,
    }
    if any(not np.isfinite(margin) or margin <= 0.0 for margin in margins.values()):
        raise RuntimeError(f"{label} failed a physical/domain margin: {margins!r}.")
    return margins


def run_example(
    profile: phx.qualification.CapabilityProfile,
) -> None:
    """Compose development simulation and optimization without a workflow-release claim."""
    if profile.released:
        raise ValueError(
            "Numerical-model admission does not authorize this transformed optimization "
            "demonstration. Use --development, or the host-dispatched circuit/SPMe examples "
            "for numerical production execution."
        )
    prepared, protocol, parameters, initial = _prepared_experiment(profile)

    simulation = _run(prepared, protocol, parameters, initial, jnp.asarray(1.25))
    simulation_margins = _require_run_margins(simulation, "ECM simulation")

    charge_index = prepared.plan.outputs.names.index("charge_c")

    def objective(current_a, _args):
        trial = _run(prepared, protocol, parameters, initial, current_a)
        charge_error_a_s = (
            trial.outputs.values[-1, charge_index] - _TARGET_CHARGE_C
        ) / _CURRENT_DURATION_S
        return 0.5 * charge_error_a_s**2

    problem = phx.optim.MinimizationProblem(
        objective,
        bounds=phx.optim.Bounds(*_CURRENT_BOUNDS_A),
        problem_id="example:thermal-ecm-constant-current-amplitude",
    )
    optimization = phx.optim.ProjectedLBFGS(history_size=4).solve(
        problem,
        jnp.asarray(1.0),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1.0e-5,
            relative_optimality=0.0,
            absolute_step=1.0e-8,
            relative_step=0.0,
            maximum_steps=32,
            maximum_evaluations=128,
        ),
        args=None,
    )
    if int(np.asarray(optimization.status)) != int(phx.optim.OptimizationStatus.SUCCESS):
        raise RuntimeError(
            f"Generic minimization failed with status {int(optimization.status)}."
        )

    optimum_a = float(np.asarray(optimization.parameters))
    replay = _run(prepared, protocol, parameters, initial, optimization.parameters)
    replay_margins = _require_run_margins(replay, "optimized ECM replay")
    final_charge_c = float(np.asarray(replay.outputs.values[-1, charge_index]))
    optimization_margins = {
        "target-charge-c": _TARGET_TOLERANCE_C - abs(final_charge_c - _TARGET_CHARGE_C),
        "lower-current-bound-a": optimum_a - _CURRENT_BOUNDS_A[0],
        "upper-current-bound-a": _CURRENT_BOUNDS_A[1] - optimum_a,
        "optimality": 1.0e-4
        - float(np.asarray(optimization.diagnostics.final_optimality_norm)),
    }
    if any(
        not np.isfinite(margin) or margin <= 0.0
        for margin in optimization_margins.values()
    ):
        raise RuntimeError(
            "Optimized replay failed a target, bound, or optimality margin: "
            f"{optimization_margins!r}."
        )

    print(f"profile_id={profile.profile_id}")
    print(f"numerical_admission_id={simulation.numerical_admission_id}")
    print(f"predictive_admission_id={simulation.predictive_admission_id}")
    print(f"simulation_run_id={simulation.require_evidence_identity()}")
    print(f"simulation_margins={simulation_margins}")
    print(f"optimum_current_a={optimum_a:.8f}")
    print(f"optimized_final_charge_c={final_charge_c:.8f}")
    print(f"optimized_run_id={replay.require_evidence_identity()}")
    print(f"optimized_run_margins={replay_margins}")
    print(f"optimization_margins={optimization_margins}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development",
        action="store_true",
        required=True,
        help="Run the explicit candidate optimization demonstration without release claims.",
    )
    parser.parse_args()
    print("admission_scope=unreleased-development")
    run_example(battery.THERMAL_ECM_CANDIDATE)


if __name__ == "__main__":
    main()
