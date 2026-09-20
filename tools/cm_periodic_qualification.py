#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phydrax.chemistry.periodic._finite import (
    PeriodicFiniteBoundaryPlan,
    PeriodicFiniteOrbitalPlan,
)
from phydrax.chemistry.periodic._orbital_model import (
    PeriodicBlochGauge,
    PeriodicOrbitalBasisPlan,
    PeriodicOrbitalPencilPlan,
)
from phydrax.chemistry.periodic._spectrum import PeriodicSpectrumPlan
from phydrax.chemistry.periodic._topology import (
    identity_cross_k_connection,
    PeriodicBandManifold,
    PeriodicOverlapBundle,
    PeriodicWilsonPlan,
)
from phydrax.discretization import (
    PeriodicCell,
    ReciprocalConnectivityPlan,
    ReciprocalMeshPlan,
)
from phydrax.operators.periodic import periodic_translation_family_from_dense_blocks
from phydrax.units import ANGSTROM, ELECTRONVOLT


def _ssh(intercell: float, intracell: float, mesh_size: int):
    cell = PeriodicCell([[1.0]])
    basis = PeriodicOrbitalBasisPlan(
        cell,
        ("A", "B"),
        [[0.0], [0.5]],
        ANGSTROM,
        PeriodicBlochGauge("lattice"),
    )
    blocks = np.zeros((3, 2, 1, 2, 1), dtype="complex128")
    blocks[1, 0, 0, 1, 0] = intracell
    blocks[1, 1, 0, 0, 0] = intracell
    blocks[0, 0, 0, 1, 0] = intercell
    blocks[2, 1, 0, 0, 0] = intercell
    family = periodic_translation_family_from_dense_blocks([[-1], [0], [1]], blocks)
    pencil = PeriodicOrbitalPencilPlan.orthonormal(
        basis, family.plan, family.state, ELECTRONVOLT
    ).prepare()
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (mesh_size,))
    spectrum = PeriodicSpectrumPlan(pencil, mesh).evaluate()
    connectivity = ReciprocalConnectivityPlan.regular(mesh).prepare()
    manifold = PeriodicBandManifold(spectrum, [0])
    bundle = PeriodicOverlapBundle(
        manifold,
        identity_cross_k_connection(pencil, connectivity),
    )
    edges = np.arange(0, 2 * mesh_size, 2)
    wilson = PeriodicWilsonPlan(bundle, edges).evaluate()
    return pencil, spectrum, wilson, manifold


def qualify():
    calibration_pencil, calibration_spectrum, calibration_wilson, _ = _ssh(1.0, 0.4, 32)
    _, _, locked_wilson, locked_manifold = _ssh(0.9, 0.3, 48)
    finite = PeriodicFiniteOrbitalPlan(
        calibration_pencil, PeriodicFiniteBoundaryPlan.open((12,))
    ).realize()
    finite_matrix = np.asarray(finite.hamiltonian.to_dense())
    calibration = {
        "spectrum_residual": float(
            np.max(np.asarray(calibration_spectrum.eigen_residuals))
        ),
        "metric_residual": float(
            np.max(np.asarray(calibration_spectrum.metric_residuals))
        ),
        "zak_distance_to_pi": abs(abs(float(calibration_wilson.zak_phase)) - np.pi),
        "finite_hermiticity_residual": float(
            np.max(np.abs(finite_matrix - finite_matrix.conj().T))
        ),
    }
    locked = {
        "minimum_direct_gap": float(locked_manifold.minimum_direct_gap),
        "zak_distance_to_pi": abs(abs(float(locked_wilson.zak_phase)) - np.pi),
        "minimum_link_singular_value": float(
            np.min(np.asarray(locked_wilson.link_singular_values))
        ),
    }
    criteria = {
        "calibration_spectrum_residual": 1.0e-9,
        "calibration_metric_residual": 1.0e-9,
        "calibration_zak_distance_to_pi": 1.0e-7,
        "finite_hermiticity_residual": 1.0e-12,
        "locked_minimum_direct_gap": 0.5,
        "locked_zak_distance_to_pi": 1.0e-7,
        "locked_minimum_link_singular_value": 1.0e-4,
    }
    predicates = {
        "calibration_spectrum": calibration["spectrum_residual"]
        <= criteria["calibration_spectrum_residual"],
        "calibration_metric": calibration["metric_residual"]
        <= criteria["calibration_metric_residual"],
        "calibration_zak": calibration["zak_distance_to_pi"]
        <= criteria["calibration_zak_distance_to_pi"],
        "finite_hermiticity": calibration["finite_hermiticity_residual"]
        <= criteria["finite_hermiticity_residual"],
        "locked_gap": locked["minimum_direct_gap"]
        >= criteria["locked_minimum_direct_gap"],
        "locked_zak": locked["zak_distance_to_pi"]
        <= criteria["locked_zak_distance_to_pi"],
        "locked_links": locked["minimum_link_singular_value"]
        >= criteria["locked_minimum_link_singular_value"],
    }
    return {
        "campaign_id": "cm-periodic-core-independent-ssh",
        "calibration": calibration,
        "locked": locked,
        "criteria": criteria,
        "predicates": predicates,
        "successful": all(predicates.values()),
        "nonclaims": [
            "This qualification does not release a capability.",
            "Wilson phases are not localized Wannier functions.",
            "No material-accuracy claim follows from analytic tight-binding controls.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = qualify()
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n", encoding="utf-8")
    if not result["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
