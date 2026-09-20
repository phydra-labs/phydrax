#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent analytic smoke qualification for the lattice-materials candidate slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phydrax.atomistic import AtomisticUnitSystem
from phydrax.chemistry.periodic._lattice_dynamics import HarmonicPhononPlan
from phydrax.chemistry.periodic._lattice_force_constants import (
    IFCConstraintPolicy,
    normalize_second_order_force_constants,
    second_order_force_constant_unit,
    third_order_force_constant_unit,
    ThirdOrderForceConstants,
)
from phydrax.chemistry.periodic._lattice_thermodynamics import (
    HarmonicThermodynamicsPlan,
    QuasiHarmonicPlan,
)
from phydrax.chemistry.periodic._lattice_transport import (
    IFC3ModeVertexPlan,
    ThreePhononRTAPlan,
)
from phydrax.discretization import PeriodicCell
from phydrax.sparse import EdgeRelation


def qualify() -> dict[str, object]:
    units = AtomisticUnitSystem.reduced()
    cell = PeriodicCell(np.eye(3))
    relation = EdgeRelation([0, 0, 0], [0, 0, 0], source_size=1, target_size=1)
    blocks = np.asarray([-np.eye(3), 2.0 * np.eye(3), -np.eye(3)])
    ifc = normalize_second_order_force_constants(
        relation,
        [[-1, 0, 0], [0, 0, 0], [1, 0, 0]],
        blocks,
        [[0.0, 0.0, 0.0]],
        cell,
        second_order_force_constant_unit(
            units.scale.energy_unit, units.scale.length_unit
        ),
        system_id="qualification-monatomic-chain",
        source_kind="independent-analytic-provider",
        source_id="qualification-ifc2",
        constraint_policy=IFCConstraintPolicy(maximum_relative_correction=1.0e-12),
    )
    q = np.asarray([[0.125, 0.0, 0.0], [0.25, 0.0, 0.0], [0.375, 0.0, 0.0]])
    dispersion = HarmonicPhononPlan(ifc, [1.0], units).prepare().evaluate(q)
    expected = 2.0 * np.abs(np.sin(np.pi * q[:, 0]))
    dispersion_residual = float(
        np.max(np.abs(np.asarray(dispersion.angular_frequencies[:, 0]) - expected))
    )
    thermal = HarmonicThermodynamicsPlan([1 / 3, 1 / 3, 1 / 3], [1.0, 2.0, 3.0], units)
    thermodynamics = thermal.evaluate(dispersion.angular_frequencies)
    volumes = np.arange(8.0, 13.0)
    qha = QuasiHarmonicPlan(
        volumes,
        (volumes - 10.0) ** 2,
        np.broadcast_to(np.asarray(dispersion.angular_frequencies), (5, 3, 3)),
        thermal,
    ).evaluate()
    qha_residual = float(np.max(np.abs(np.asarray(qha.equilibrium_volumes) - 10.0)))
    triplets = np.asarray(
        [
            (first, second, third)
            for first in range(2)
            for second in range(2)
            for third in range(2)
        ],
        dtype="int64",
    )
    signs = np.asarray([1.0, -1.0])
    coefficients = (
        0.01 * signs[triplets[:, 0]] * signs[triplets[:, 1]] * signs[triplets[:, 2]]
    )
    cubic = coefficients[:, None, None, None] * np.ones((1, 3, 3, 3))
    translations = np.zeros((8, 2, 1), dtype="int64")
    ifc3 = ThirdOrderForceConstants(
        triplets,
        translations,
        cubic,
        cubic,
        third_order_force_constant_unit(units.scale.energy_unit, units.scale.length_unit),
        atom_count=2,
        system_id="qualification-anharmonic-crystal",
        source_kind="provider-normalized-ifc3",
        source_id="qualification-ifc3",
    )
    vertices = IFC3ModeVertexPlan(
        ifc3,
        [[0]],
        (1,),
        [[0.0]],
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        [np.eye(6)],
        [1.0, 1.0],
        units,
        phonon_result_id="qualification-phonons",
    ).evaluate()
    velocity = np.zeros((1, 6, 3))
    velocity[0, 0, 0] = 1.0
    rta = ThreePhononRTAPlan(
        ifc3,
        [[0]],
        (1,),
        [1.0],
        0.02,
        1.0,
        1.0,
        units,
        detailed_balance_tolerance=1.0e-10,
    ).evaluate(vertices, velocity, [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    criteria = {
        "ifc_pair": float(ifc.constraints.corrected_pair_residual) <= 1.0e-12,
        "ifc_asr": float(ifc.constraints.corrected_acoustic_residual) <= 1.0e-12,
        "ifc_rotation": float(ifc.constraints.corrected_rotational_residual) <= 1.0e-12,
        "dispersion": dispersion_residual <= 1.0e-10,
        "thermodynamic_identity": float(thermodynamics.thermodynamic_identity_residual)
        <= 1.0e-10,
        "qha_interior": qha_residual <= 1.0e-10,
        "ifc3_vertex_binding": vertices.ifc3_id == ifc3.ifc_id,
        "rta_momentum": float(rta.scattering.momentum_residual) == 0.0,
        "rta_detailed_balance": (
            float(rta.scattering.detailed_balance_residual) <= 1.0e-10
        ),
        "rta_finite": bool(rta.finite_conductivity),
    }
    return {
        "case": "independent-monatomic-nearest-neighbor",
        "raw_metrics": {
            "pair_residual": float(ifc.constraints.corrected_pair_residual),
            "asr_residual": float(ifc.constraints.corrected_acoustic_residual),
            "rotation_residual": float(ifc.constraints.corrected_rotational_residual),
            "dispersion_residual": dispersion_residual,
            "thermodynamic_identity_residual": float(
                thermodynamics.thermodynamic_identity_residual
            ),
            "qha_volume_residual": qha_residual,
            "rta_momentum_residual": float(rta.scattering.momentum_residual),
            "rta_detailed_balance_residual": float(
                rta.scattering.detailed_balance_residual
            ),
        },
        "criteria": criteria,
        "successful": all(criteria.values()),
        "nonclaims": [
            "candidate evidence is not release evidence",
            "analytic chain is not material accuracy",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = qualify()
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")
    if not payload["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
