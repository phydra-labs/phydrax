# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import qualification_failure
from tools.meshing_qualification.scenarios.layout import _hybrid_gmsh_result


def qualify_hybrid_layer() -> dict[str, object]:

    with TemporaryDirectory(prefix="phydrax-hybrid-layer-") as temporary:
        result, control = _hybrid_gmsh_result(Path(temporary) / "hybrid-slab.brep")
    if not result.audit.passed or not result.compliance.passed:
        raise qualification_failure(
            "qualify_hybrid_layer",
            ("Generated hybrid layer mesh failed audit or compliance."),
            {"result": result},
        )
    block_kinds = tuple(block.cell_kind for block in result.mesh.blocks)
    if block_kinds != ("prism", "tetrahedron"):
        raise qualification_failure(
            "qualify_hybrid_layer",
            ("Generated hybrid layer mesh lost its required cell families."),
            {"block_kinds": block_kinds},
        )
    achieved = dict(result.compliance.achieved)
    return {
        "mesh_id": result.mesh.mesh_id,
        "result_id": result.result_id,
        "control_id": control.control_id,
        "schedule_id": control.schedule.schedule_id,
        "thicknesses": list(control.schedule.thicknesses),
        "block_kinds": block_kinds,
        "cells": result.mesh.entity_set(3).count,
        "maximum_layer_thickness_error": achieved[
            f"layer:{control.control_id}:maximum_thickness_residual"
        ],
        "audit_passed": result.audit.passed,
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_hybrid_layer()
