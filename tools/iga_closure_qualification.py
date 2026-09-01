#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx
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


def _rom() -> dict[str, object]:
    cases = tuple(
        phx.rom.ROMCaseSpec(f"case-{index}", (("mu", float(index)),))
        for index in range(3)
    )

    def truth(case: phx.rom.ROMCaseSpec) -> phx.rom.TruthSample:
        mu = dict(case.parameters)["mu"]
        state = np.asarray((1.0, mu))
        return phx.rom.TruthSample(
            state,
            f"truth-{case.case_id}",
            operator=np.eye(2),
            rhs=state,
            dual_norm_inverse=np.eye(2),
            stability_lower_bound=1.0,
        )

    corpus = phx.rom.create_corpus(
        cases,
        truth,
        truth_model_id="closure-truth",
        truth_model_revision="canonical",
        split=phx.rom.CorpusSplit(tuple(case.case_id for case in cases)),
    )
    artifact = phx.rom.train_profile(corpus, phx.rom.LinearPODProfile(2))
    evaluation = phx.rom.evaluate(artifact, cases[1], truth_model=truth)
    audit = phx.rom.audit_against_truth(evaluation, truth(cases[1]))
    return {
        "passed": evaluation.source == "rom" and audit.relative_state_error <= 1.0e-12,
        "artifact_id": artifact.artifact_id,
        "relative_state_error": audit.relative_state_error,
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
        "linear_pod_rom": _rom(),
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
