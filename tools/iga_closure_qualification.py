#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from phydrax.discretization import iga
from phydrax.discretization.iga._certificate import (
    CertificateDisposition,
    certify_tensor_nurbs,
)
from phydrax.discretization.iga._compatible import SplineDeRhamComplex
from phydrax.discretization.iga._volume import TensorNURBSVolume


def _geometry_certificate() -> dict[str, object]:
    grid = iga.BSplineGrid.open_uniform(2, 1)
    coordinates = grid.greville_abscissae
    xx, yy = jnp.meshgrid(coordinates, coordinates, indexing="ij")
    geometry = iga.NURBSGeometryState(
        jnp.stack((xx, yy), axis=-1),
        jnp.ones((grid.coefficient_count, grid.coefficient_count)),
    )
    plan = iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        geometry,
        quadrature_policy=iga.IsogeometricQuadraturePolicy(3),
    )
    certificate = certify_tensor_nurbs(
        TensorNURBSVolume("unit-square", plan.basis, geometry)
    )
    return {
        "passed": certificate.disposition is CertificateDisposition.PASS,
        "certificate_id": certificate.certificate_id,
        "cell_count": len(certificate.cells),
        "diagnostics": [value.code for value in certificate.diagnostics],
    }


def _compatible_complex() -> dict[str, object]:
    grid = iga.BSplineGrid.open_uniform(2, 2)
    complex_ = SplineDeRhamComplex((grid, grid))
    defect = float(jnp.max(complex_.d_squared_defects))
    return {
        "passed": defect <= 1.0e-13,
        "complex_id": complex_.complex_id,
        "d_squared_defect": defect,
    }


def _thb() -> dict[str, object]:
    from phydrax.discretization.iga._thb import THBHierarchy, THBLevel

    hierarchy = THBHierarchy(
        (
            THBLevel(0, "coarse", (True,), (True, False)),
            THBLevel(1, "fine", (True, True), (False, True, True)),
        ),
        (np.asarray(((1.0, 0.0), (0.5, 0.5), (0.0, 1.0))),),
    )
    certificate = hierarchy.certify()
    return {
        "passed": certificate.passed,
        "hierarchy_id": hierarchy.hierarchy_id,
        "certificate_id": certificate.certificate_id,
        "partition_defect": certificate.partition_defect,
        "rank": certificate.rank,
        "basis_count": certificate.basis_count,
    }


def run() -> dict[str, object]:
    cases = {
        "geometry_certificate": _geometry_certificate(),
        "compatible_complex": _compatible_complex(),
        "thb_basis": _thb(),
    }
    return {
        "kind": "iga-closure-substrate-qualification",
        "passed": all(bool(value["passed"]) for value in cases.values()),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/iga_closure_qualification.json"),
    )
    arguments = parser.parse_args()
    report = run()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
