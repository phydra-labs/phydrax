#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iga_closure_qualification import run as run_closure
from iga_h1_qualification import run as run_h1


def run() -> dict[str, object]:
    h1 = run_h1()
    closure = run_closure()
    cases = {
        "core.rational.quarter_annulus": {
            "support": {
                "geometry": "full_dim_2d_nurbs",
                "space": "scalar_H1",
                "basis": "direct_tensor",
                "formulation": "diffusion_reaction",
                "interface": "none",
                "backend": "cpu",
                "precision": "float64",
                "distributed": "single",
                "derivative": "Q2",
                "restart": "none",
            },
            "passed": bool(h1["passed"]),
            "evidence_kind": h1["kind"],
        },
        "core.compatible.2d": {
            "support": {
                "geometry": "full_dim_2d_nurbs",
                "space": "spline_de_rham",
                "basis": "direct_tensor",
                "formulation": "complex_algebra",
                "interface": "none",
                "backend": "cpu",
                "precision": "float64",
                "distributed": "single",
                "derivative": "Q0",
                "restart": "none",
            },
            "passed": bool(closure["cases"]["compatible_complex"]["passed"]),
            "evidence_kind": closure["kind"],
        },
        "core.thb.algebra": {
            "support": {
                "geometry": "fixed_tensor_geometry",
                "space": "thb_H1",
                "basis": "hierarchical",
                "formulation": "basis_algebra",
                "interface": "none",
                "backend": "cpu",
                "precision": "float64",
                "distributed": "single",
                "derivative": "Q6",
                "restart": "none",
            },
            "passed": bool(closure["cases"]["thb_basis"]["passed"]),
            "evidence_kind": closure["kind"],
        },
    }
    return {
        "kind": "iga-r1-candidate-qualification",
        "passed": all(bool(case["passed"]) for case in cases.values()),
        "release_ready": False,
        "reason": "Candidate evidence is unsigned and covers only the listed support tuples.",
        "cases": cases,
        "source_evidence": {
            "h1": h1,
            "closure": closure,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/iga_r1_qualification.json"),
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
