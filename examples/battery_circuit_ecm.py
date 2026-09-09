#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Execute a native circuit ECM current/rest experiment with explicit admission."""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np

from examples._battery_admission import add_admission_arguments, resolve_example_admission
from examples.battery_simulation_and_optimization import _constant_law
from phydrax.applications import battery
from phydrax.nonlinear import NonlinearTermination
from phydrax.solver import BDFMethod, DAEAdaptivePolicy, DAESolvePolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_admission_arguments(parser)
    arguments = parser.parse_args()
    profile, admission, distribution_id = resolve_example_admission(
        parser, arguments, battery.CIRCUIT_ECM_CANDIDATE
    )
    adapter = battery.CircuitConnectedEcmAdapter(battery.CircuitConnectedEcmPlan(1))
    native = adapter.prepare()
    parameters = battery.ThermalEquivalentCircuitParameters(
        0.05,
        jnp.asarray((0.1,)),
        jnp.asarray((10.0,)),
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
            0.0, quantity="entropic-coefficient", unit="V/K", value_bounds=(-0.01, 0.01)
        ),
    )
    protocol = battery.BatteryProtocolPlan(
        (battery.CurrentStepPlan(1.0), battery.RestStepPlan(1.0)), node_side="right"
    )
    numerical = battery.BatteryDAESolvePlan(
        DAESolvePolicy(
            method=BDFMethod(2),
            nonlinear_termination=NonlinearTermination(
                absolute_step=0.0,
                relative_step=0.0,
            ),
            adaptive=DAEAdaptivePolicy(
                relative_tolerance=1e-7,
                absolute_tolerance=1e-9,
                initial_step=0.001,
                maximum_step=0.02,
            ),
        ),
        guard_ids=tuple(guard.guard_id for guard in adapter.native_guards(native)),
    )
    experiment = battery.BatteryExperimentPlan(
        adapter,
        protocol,
        battery.BatteryOutputPlan(
            ("voltage_v", "current_a", "charge_c", "temperature_k")
        ),
        numerical,
        jnp.linspace(0.0, 2.0, 201),
        profile,
        battery.CIRCUIT_ECM_SUPPORT,
    ).prepare(admission=admission, distribution_id=distribution_id)
    result = experiment.run(
        parameters,
        battery.CircuitConnectedEcmInitialCondition(500.0, 298.15, relaxed=True),
        battery.BatteryProtocolValues(protocol, jnp.asarray((2.0,))),
    )
    if not bool(np.asarray(result.successful)) or not bool(
        np.all(np.asarray(result.outputs.valid))
    ):
        raise RuntimeError(
            f"Circuit ECM failed with application status {int(result.application_status)}."
        )
    if abs(float(result.ledger.charge_defect_c)) > 1e-5:
        raise RuntimeError("Circuit ECM charge ledger exceeded the example tolerance.")
    print(f"run_id={result.require_evidence_identity()}")
    print(f"numerical_admission_id={result.numerical_admission_id}")
    print(f"final_voltage_v={float(result.outputs.values[-1, 0]):.9f}")
    print(f"final_charge_c={float(result.outputs.values[-1, 2]):.9f}")
    print(f"maximum_kcl_defect_a={float(result.ledger.maximum_kcl_defect_a):.3e}")
    print(f"maximum_power_defect_w={float(result.ledger.maximum_power_defect_w):.3e}")
    print(
        f"physical_initialization_correction={float(result.ledger.physical_initialization_correction):.3e}"
    )


if __name__ == "__main__":
    main()
