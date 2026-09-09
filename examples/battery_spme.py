#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run canonical isothermal SPMe with self-authored data, not a predictive cell fit."""

from __future__ import annotations

import argparse

import diffrax as dfx
import jax.numpy as jnp
import numpy as np

from examples._battery_admission import add_admission_arguments, resolve_example_admission
from phydrax.applications import battery
from tools._battery_spme_references import native_parameters, SyntheticSpmeData


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_admission_arguments(parser)
    arguments = parser.parse_args()
    profile, admission, distribution_id = resolve_example_admission(
        parser, arguments, battery.MARQUIS_2019_SPME_CANDIDATE
    )
    data = SyntheticSpmeData()
    parameters = native_parameters(data)
    protocol = battery.BatteryProtocolPlan(
        (
            battery.CurrentStepPlan(0.5),
            battery.RestStepPlan(0.25),
            battery.CurrentStepPlan(0.25),
        )
    )
    adapter = battery.Marquis2019SpmeAdapter(
        battery.Marquis2019SpmePlan(
            6,
            negative_electrolyte_cell_count=4,
            separator_electrolyte_cell_count=3,
            positive_electrolyte_cell_count=4,
        )
    )
    experiment = battery.BatteryExperimentPlan(
        adapter,
        protocol,
        battery.BatteryOutputPlan(
            ("voltage_v", "current_a", "total_lithium_mol", "eq49:error_estimate")
        ),
        battery.BatteryDiffraxSolvePlan(
            stepsize_controller=dfx.StepTo(ts=jnp.linspace(0.0, 1.0, 129)),
            adjoint=dfx.DirectAdjoint(),
            maximum_steps=136,
            relative_tolerance=1e-10,
            absolute_tolerance=1e-12,
        ),
        jnp.asarray((0.0, 0.25, 0.5, 0.75, 1.0)),
        profile,
        battery.MARQUIS_2019_SPME_SUPPORT,
    ).prepare(admission=admission, distribution_id=distribution_id)
    result = experiment.run(
        parameters,
        battery.Marquis2019SpmeInitialCondition(*data.initial_stoichiometries),
        battery.BatteryProtocolValues(protocol, jnp.asarray((0.2, -0.05))),
    )
    if not bool(np.asarray(result.successful)) or not bool(
        np.all(np.asarray(result.outputs.valid))
    ):
        raise RuntimeError(
            f"SPMe failed with application status {int(result.application_status)}."
        )
    values = np.asarray(result.outputs.values)
    print(f"run_id={result.require_evidence_identity()}")
    print(f"numerical_admission_id={result.numerical_admission_id}")
    print("parameter_scope=self-authored-equation-example-not-predictive-cell-data")
    print(f"final_voltage_v={values[-1, 0]:.9f}")
    print(f"total_lithium_change_mol={values[-1, 2] - values[0, 2]:.3e}")
    print(f"maximum_eq49_error={np.max(values[:, 3]):.3e}")


if __name__ == "__main__":
    main()
