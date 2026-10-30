#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_environment,
    measure_repeated,
    measure_synchronized,
)
from phydrax.discretization import iga
from phydrax.geometry.brep import TSplineSurfacePatch


def _case(span_count: int, *, warmup: int, repeats: int) -> dict[str, Any]:
    breaks = tuple(Fraction(index, span_count) for index in range(span_count + 1))
    mesh = iga.TMesh2D.tensor_product(breaks, breaks)
    greville_points = tuple(
        anchor.local_knots.greville_parameter for anchor in mesh.anchors
    )
    exact_points = tuple((u, v, u * v) for u, v in greville_points)
    patch = TSplineSurfacePatch(
        mesh,
        np.asarray([[float(value) for value in row] for row in exact_points]),
        np.ones((mesh.coefficient_count,), dtype=np.float64),
        exact_control_points=exact_points,
        exact_weights=tuple(Fraction(1) for _ in mesh.anchors),
    )
    trim = iga.UVTrimmedSurface(
        iga.UVTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    )
    plan = iga.TSplineIsogeometricPlan(
        iga.TSplineGeometryState(patch),
        iga.TSplineH1FieldSpec(
            "u", weights_from_geometry=True, anchor_count=mesh.coefficient_count
        ),
        trim=trim,
        quadrature_policy=iga.IsogeometricQuadraturePolicy(5),
    )
    marked = next(
        cell
        for cell in mesh.cells
        if cell.parameter_bounds
        == (
            Fraction(span_count // 2 - 1, span_count),
            Fraction(span_count // 2, span_count),
            Fraction(span_count // 2 - 1, span_count),
            Fraction(span_count // 2, span_count),
        )
    )
    refinement, first_refinement_seconds = measure_synchronized(
        lambda: plan.refine((marked.cell_id,))
    )
    _, refinement_distribution = measure_repeated(
        lambda: plan.refine((marked.cell_id,)), warmup=warmup, repeats=repeats
    )
    prepared, preparation_seconds = measure_synchronized(
        lambda: refinement.target_plan.prepare(
            numeric_version=f"tspline-bench-{span_count}"
        )
    )
    target_mesh = refinement.target_plan.geometry.mesh
    full_tensor_count = (len(target_mesh.axis_breaks[0]) - 1) * (
        len(target_mesh.axis_breaks[1]) - 1
    )
    return {
        "source_span_count_per_axis": span_count,
        "source_cell_count": mesh.cell_count,
        "source_anchor_count": mesh.coefficient_count,
        "marked_cell_ids": list(refinement.basis_refinement.marked_cell_ids),
        "closure_cell_ids": list(refinement.basis_refinement.closure_cell_ids),
        "target_cell_count": target_mesh.cell_count,
        "target_anchor_count": target_mesh.coefficient_count,
        "target_t_junction_count": target_mesh.t_junction_count,
        "cartesian_cells_at_target_breaks": full_tensor_count,
        "transfer_route_count": refinement.basis_refinement.transfer_plan.route_count,
        "basis_certificate_id": refinement.target_plan.geometry.surface_patch.basis_certificate.certificate_id,
        "exact_geometry_certificate_id": refinement.target_plan.exact_geometry_certificate.certificate_id,
        "numeric_geometry_certificate_id": refinement.target_plan.geometry_certificate.certificate_id,
        "prepared_id": prepared.prepared_id,
        "first_refinement_seconds": first_refinement_seconds,
        "refinement_seconds": refinement_distribution.to_seconds_dict(),
        "preparation_seconds": preparation_seconds,
        "local_non_tensor_structure": target_mesh.cell_count < full_tensor_count,
    }


def run(*, smoke: bool, warmup: int, repeats: int) -> dict[str, Any]:
    spans = (4, 8) if smoke else (4, 8, 16)
    return {
        "environment": capture_environment().to_dict(),
        "kind": "iga-tspline-local-refinement-record-only-timing",
        "record_only": True,
        "rows": [
            _case(span_count, warmup=warmup, repeats=repeats) for span_count in spans
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record shared ASTS2D CAD/IGA refinement and preparation timings."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/iga_tspline_refinement.json"),
    )
    arguments = parser.parse_args()
    payload = run(
        smoke=arguments.smoke,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    print(rendered)
    write_json_atomic(arguments.output, payload)


if __name__ == "__main__":
    main()
