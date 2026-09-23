#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from phydrax.applications import quantum_hall as qh
from phydrax.units import ANGSTROM, ELECTRONVOLT


_ELECTRONVOLT_JOULE = 1.602_176_634e-19


def qualification_record() -> dict[str, object]:
    haldane = qh.HaldaneModelPlan(
        1.0,
        0.0,
        0.2,
        np.pi / 2.0,
        1.0,
        ELECTRONVOLT,
        ANGSTROM,
    )
    bulk = qh.evaluate_haldane_topology(haldane, mesh_shape=(15, 15))
    sphere = qh.HaldaneSpherePlan(
        2,
        qh.MonopoleLandauLevel(3, 0, qh.SPIN_POLARIZED_ELECTRON),
        "fermion",
        qh.QuantumHallEnergyScale(
            ELECTRONVOLT,
            _ELECTRONVOLT_JOULE,
            "electronvolt",
        ),
        filling=Fraction(1, 3),
        shift=3,
    )
    pseudopotentials = qh.HaldanePseudopotentialPlan(
        sphere,
        {1: 1.0, 3: 0.0},
        "v1-parent",
    )
    prepared = qh.prepare_haldane_sphere_hamiltonian(
        pseudopotentials,
        twice_projection=0,
    )
    higher_manifold = qh.MonopoleLandauLevel(
        5,
        1,
        qh.SPIN_POLARIZED_ELECTRON,
    )
    higher_sphere = qh.HaldaneSpherePlan(
        2,
        higher_manifold,
        "fermion",
        qh.QuantumHallEnergyScale(
            ELECTRONVOLT,
            _ELECTRONVOLT_JOULE,
            "electronvolt",
        ),
    )
    higher = qh.coulomb_haldane_pseudopotentials(higher_sphere)
    return {
        "kind": "quantum-hall-candidate-qualification",
        "haldane": {
            "chern": int(np.asarray(bulk.chern.nearest_integer)),
            "quantization_residual": float(np.asarray(bulk.chern.quantization_residual)),
            "minimum_gap": float(np.asarray(bulk.chern.minimum_direct_gap)),
            "successful": bool(np.asarray(bulk.successful)),
        },
        "sphere": {
            "dimension": prepared.many_body.dimension,
            "nonzero_routes": prepared.many_body.evidence.nonzero_routes,
            "hermiticity_residual": float(
                np.asarray(prepared.many_body.evidence.hermiticity_residual)
            ),
            "accepted": bool(np.asarray(prepared.many_body.evidence.accepted)),
        },
        "higher_landau": {
            "level": higher_manifold.landau_level,
            "orbital_count": higher_manifold.orbital_count,
            "channels": higher.relative_channels,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    text = json.dumps(qualification_record(), indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(text, end="")
    else:
        arguments.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
