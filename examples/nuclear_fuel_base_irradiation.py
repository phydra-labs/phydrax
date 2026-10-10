#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Assembled synthetic startup/base/ramp/shutdown/restart with durable checkpoints.

This numerical example is not an IFPE/FUMEX experiment and is not scientific or
regulatory validation. Spectrum, helium table, geometry and irradiation are
explicit declared synthetic inputs. Measured-source extrapolation evidence is
retained by the equation-reproduction profile; source envelopes are not widened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import get_args

import jax
import numpy as np

from phydrax.applications.nuclear_fuel import (
    admit_fuel_rod,
    FUEL_GAS_SPECIES,
    FuelGeometryReduction,
    FuelRodSimulation,
    FuelRodState,
    prepare_fuel_rod,
)
from phydrax.typing import parse
from tools.nuclear_fuel_qualification import history_observables, numerical_rod_inputs


jax.config.update("jax_enable_x64", True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Resume the durable accepted checkpoint, not burnup alone.",
    )
    parser.add_argument("--time-step", type=float, default=120.0)
    parser.add_argument("--fuel-cells", type=int, default=4)
    parser.add_argument("--axial-layers", type=int, default=1)
    parser.add_argument(
        "--reduction", choices=get_args(FuelGeometryReduction), default="layered-radial"
    )
    parser.add_argument("--scattering-barn", type=float, default=5.0)
    parser.add_argument(
        "--scattering-source",
        default="synthetic-numerical-verification:not-spectrum-data",
    )
    parser.add_argument("--base-duration", type=float, default=3.0e7)
    parser.add_argument("--base-power", type=float, default=20000.0)
    parser.add_argument("--transient-duration", type=float, default=3600.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the declared short numerical history instead of an annual base.",
    )
    arguments = parser.parse_args()
    duration, power, transient = (
        (3480.0, 3000.0, 60.0)
        if arguments.smoke
        else (arguments.base_duration, arguments.base_power, arguments.transient_duration)
    )
    specification, history, coolant, bundle = numerical_rod_inputs(
        mode_count=8,
        axial_layers=arguments.axial_layers,
        scattering_cross_section_barn=arguments.scattering_barn,
        scattering_source_id=arguments.scattering_source,
        base_duration=duration,
        base_linear_power=power,
        transient_duration=transient,
    )
    admitted = admit_fuel_rod(
        specification,
        history,
        coolant,
        bundle=bundle,
        reduction=parse(arguments.reduction, FuelGeometryReduction, "reduction"),
        axial_interaction="frictionless",
    )
    prepared = prepare_fuel_rod(
        admitted, fuel_cells=arguments.fuel_cells, cladding_cells=4, quadrature_order=3
    )
    simulation = FuelRodSimulation(
        prepared,
        time_step=arguments.time_step,
        checkpoint_directory=arguments.checkpoint_directory,
        checkpoint_interval=1,
        segment_steps=1,
        minimum_time_step=0.01,
        maximum_time_step=max(4 * arguments.time_step, duration / 100),
    )
    prefix: list[FuelRodState] = []
    if not arguments.restart:
        runtime_state = simulation.runtime.initial_state(prepared.initial_state())
        if not isinstance(runtime_state.accepted_state, FuelRodState):
            raise TypeError(
                "The production runtime omitted the complete initial fuel state."
            )
        prefix.append(runtime_state.accepted_state)
        for _ in range(2):
            runtime_state, attempt = simulation.runtime.step(runtime_state)
            if not bool(attempt.successful):
                raise RuntimeError(
                    f"Startup attempt failed before checkpoint: {runtime_state.status}"
                )
            if not isinstance(runtime_state.accepted_state, FuelRodState):
                raise TypeError(
                    "The production runtime omitted the complete accepted fuel state."
                )
            prefix.append(runtime_state.accepted_state)
        checkpointed = simulation.runtime.checkpoint(runtime_state)
        print(
            json.dumps(
                {
                    "event": "accepted-checkpoint",
                    "time_s": float(checkpointed.time),
                    "checkpoint_id": checkpointed.last_checkpoint_id,
                },
                sort_keys=True,
            )
        )
    # Reload all fields, constitutive histories, gas/energy ledgers and native
    # controller state through the same-runtime checkpoint relation.
    result = simulation.run(restart=True)
    simulation.close()
    states = (*prefix, *result.states)
    last_time = -1.0
    for state in states:
        time = float(state.time)
        if time <= last_time:
            continue
        last_time = time
        energy, gas = state.energy, state.gas
        print(
            json.dumps(
                {
                    "event": "accepted-state",
                    "time_s": time,
                    "observables": history_observables(prepared, state),
                    "energy_J": {
                        "fuel_deposited": float(energy.deposited_fuel),
                        "cladding_deposited": float(energy.deposited_cladding),
                        "direct_coolant_deposited": float(
                            energy.deposited_direct_coolant
                        ),
                        "fuel_stored": float(energy.stored_fuel),
                        "cladding_stored": float(energy.stored_cladding),
                        "delivered_coolant": float(energy.delivered_coolant),
                        "balance_error": float(energy.balance_error),
                    },
                    "gas_mol": {
                        "fill": np.asarray(gas.fill_moles).tolist(),
                        "produced": np.asarray(gas.produced_moles).tolist(),
                        "retained": np.asarray(gas.retained_moles).tolist(),
                        "released": np.asarray(gas.released_moles).tolist(),
                        "balance_error": np.asarray(gas.balance_error).tolist(),
                    },
                    "gas_species_order": FUEL_GAS_SPECIES,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
    print(
        json.dumps(
            {
                "event": "terminal",
                "status": result.runtime_result.state.status,
                "run_id": result.runtime_result.run_id,
                "prepared_id": prepared.prepared_id,
                "bundle_id": bundle.bundle_id,
                "attempted_steps": len(result.attempts),
                "reduction": admitted.reduction,
                "axial_interaction": admitted.axial_interaction,
                "attempt_status_codes": [
                    int(attempt.status) for attempt in result.attempts
                ],
                "source_extrapolated_attempts": [
                    index
                    for index, attempt in enumerate(result.attempts)
                    if not bool(attempt.source_support.all())
                ],
                "scope": "synthetic-equation-reproduction-not-experimental-validation",
                "scattering_source": bundle.radial_power.scattering_source_id,
                "release_status": "candidate-unreleased",
                "external_prerequisites": (
                    "IFPE-originator-agreement",
                    "independent-locked-evaluation",
                ),
            },
            sort_keys=True,
        )
    )
    if not bool(result.runtime_result.successful):
        raise RuntimeError(
            f"Fuel rod ended with native failure: {result.runtime_result.failure}"
        )


if __name__ == "__main__":
    main()
