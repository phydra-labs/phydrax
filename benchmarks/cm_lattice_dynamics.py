#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Decision benchmark for canonical IFC preparation, phonons, QHA, and RTA axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _runtime import capture_environment, logical_array_bytes, measure_repeated

from phydrax.atomistic import AtomisticUnitSystem
from phydrax.chemistry.periodic._lattice_dynamics import HarmonicPhononPlan
from phydrax.chemistry.periodic._lattice_force_constants import (
    IFCConstraintPolicy,
    normalize_second_order_force_constants,
    second_order_force_constant_unit,
    third_order_force_constant_unit,
    ThirdOrderForceConstants,
)
from phydrax.chemistry.periodic._lattice_thermodynamics import HarmonicThermodynamicsPlan
from phydrax.chemistry.periodic._lattice_transport import (
    IFC3ModeVertexPlan,
    ThreePhononRTAPlan,
)
from phydrax.discretization import PeriodicCell
from phydrax.sparse import EdgeRelation


def benchmark_case(qpoints: int, repeats: int) -> dict[str, object]:
    units = AtomisticUnitSystem.reduced()
    cell = PeriodicCell(np.eye(3))
    relation = EdgeRelation([0, 0, 0], [0, 0, 0], source_size=1, target_size=1)
    values = np.asarray([-np.eye(3), 2.0 * np.eye(3), -np.eye(3)])
    ifc, ifc_timing = measure_repeated(
        lambda: normalize_second_order_force_constants(
            relation,
            [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
            values,
            [[0.0, 0.0, 0.0]],
            cell,
            second_order_force_constant_unit(
                units.scale.energy_unit, units.scale.length_unit
            ),
            system_id="benchmark-chain",
            source_kind="analytic-provider",
            source_id="benchmark-ifc2",
            constraint_policy=IFCConstraintPolicy(maximum_relative_correction=1.0e-12),
        ),
        warmup=0,
        repeats=repeats,
    )
    prepared, prepare_timing = measure_repeated(
        lambda: HarmonicPhononPlan(ifc, [1.0], units, maximum_qpoints=qpoints).prepare(),
        warmup=0,
        repeats=repeats,
    )
    q = np.linspace(0.5 / qpoints, 0.5 - 0.5 / qpoints, qpoints)[:, None]
    q = np.pad(q, ((0, 0), (0, 2)))
    dispersion, phonon_timing = measure_repeated(
        lambda: prepared.evaluate(q), warmup=1, repeats=repeats
    )
    thermal_plan = HarmonicThermodynamicsPlan(
        np.full(qpoints, 1.0 / qpoints), [0.5, 1.0, 2.0], units
    )
    thermodynamics, thermal_timing = measure_repeated(
        lambda: thermal_plan.evaluate(dispersion.angular_frequencies),
        warmup=1,
        repeats=repeats,
    )
    analytic = 2.0 * np.abs(np.sin(np.pi * q[:, 0]))
    residual = float(
        np.max(np.abs(np.asarray(dispersion.angular_frequencies[:, 0]) - analytic))
    )
    return {
        "atoms": 1,
        "ifc2_routes": 3,
        "qpoints": qpoints,
        "branches": 3,
        "ifc_preparation": ifc_timing.to_milliseconds_dict(),
        "harmonic_preparation": prepare_timing.to_milliseconds_dict(),
        "phonon_eigensolve": phonon_timing.to_milliseconds_dict(),
        "thermodynamics": thermal_timing.to_milliseconds_dict(),
        "analytic_dispersion_residual": residual,
        "constraint_residual": float(ifc.constraints.corrected_acoustic_residual),
        "thermodynamic_identity_residual": float(
            thermodynamics.thermodynamic_identity_residual
        ),
        "artifact_bytes": logical_array_bytes(
            (ifc.values, dispersion.angular_frequencies, thermodynamics.free_energy)
        ),
        "measured_peak_host_bytes": None,
        "measured_peak_device_bytes": None,
        "successful": bool(dispersion.successful) and bool(thermodynamics.successful),
    }


def benchmark_rta_case(repeats: int) -> dict[str, object]:
    units = AtomisticUnitSystem.reduced()
    triplets = np.asarray(
        [
            (first, second, third)
            for first in range(2)
            for second in range(2)
            for third in range(2)
        ],
        dtype=int,
    )
    signs = np.asarray([1.0, -1.0])
    coefficients = (
        0.01 * signs[triplets[:, 0]] * signs[triplets[:, 1]] * signs[triplets[:, 2]]
    )
    values = coefficients[:, None, None, None] * np.ones((1, 3, 3, 3))
    translations = np.zeros((8, 2, 1), dtype=int)
    ifc3 = ThirdOrderForceConstants(
        triplets,
        translations,
        values,
        values,
        third_order_force_constant_unit(units.scale.energy_unit, units.scale.length_unit),
        atom_count=2,
        system_id="benchmark-anharmonic-crystal",
        source_kind="provider-normalized-ifc3",
        source_id="benchmark-ifc3",
    )
    vertex_plan = IFC3ModeVertexPlan(
        ifc3,
        [[0]],
        (1,),
        [[0.0]],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        [np.eye(6)],
        [1.0, 1.0],
        units,
        phonon_result_id="benchmark-phonons",
    )
    vertices, vertex_timing = measure_repeated(
        vertex_plan.evaluate, warmup=0, repeats=repeats
    )
    plan = ThreePhononRTAPlan(ifc3, [[0]], (1,), [1.0], 0.02, 1.0, 1.0, units)
    velocity = np.zeros((1, 6, 3))
    velocity[0, 0, 0] = 1.0
    result, rta_timing = measure_repeated(
        lambda: plan.evaluate(
            vertices, velocity, np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        ),
        warmup=0,
        repeats=repeats,
    )
    return {
        "ifc3_routes": 8,
        "qpoints": 1,
        "branches": 6,
        "channels": 432,
        "vertex_transform": vertex_timing.to_milliseconds_dict(),
        "rta": rta_timing.to_milliseconds_dict(),
        "detailed_balance_residual": float(result.scattering.detailed_balance_residual),
        "ballistic_modes": int(np.sum(np.asarray(result.ballistic_mask))),
        "artifact_bytes": logical_array_bytes(
            (
                ifc3.values,
                vertices.decay_vertices,
                vertices.coalescence_vertices,
                result.thermal_conductivity,
            )
        ),
        "successful": bool(result.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qpoints", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1 or any(
        value < 2 or value > 32768 for value in arguments.qpoints
    ):
        raise ValueError(
            "Benchmark qpoints must lie in [2,32768] and repeats must be positive."
        )
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": [
            benchmark_case(value, arguments.repeats) for value in arguments.qpoints
        ],
        "rta": benchmark_rta_case(arguments.repeats),
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
