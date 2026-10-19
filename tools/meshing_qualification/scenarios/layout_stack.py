# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import phydrax as phx
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import qualification_failure
from tools.meshing_qualification.scenarios.layout import _decode_layout


def qualify_layout_stack() -> dict[str, object]:

    decoded = _decode_layout()
    with TemporaryDirectory(prefix="phydrax-layout-stack-") as temporary:
        result = phx.geometry.lower_process_stack(
            phx.geometry.ProcessStack(
                decoded.model.coordinate_contract,
                (
                    phx.geometry.StackRegion(
                        "metal",
                        decoded.model.regions[0].geometry,
                        phx.geometry.ZInterval(
                            0.0, 0.8, unit=decoded.model.coordinate_contract.length_unit
                        ),
                        20,
                    ),
                ),
            ),
            Path(temporary) / "stack.brep",
            operand_directory=Path(temporary) / "operands",
        )
    if not result.named_solid_entity_ids or not result.association_graph.graph_id:
        raise qualification_failure(
            "qualify_layout_stack",
            ("Process stack omitted persisted CAD identity evidence."),
            {"result": result},
        )
    return {
        "revision": result.revision.revision_id,
        "association_graph": result.association_graph.graph_id,
        "solids": {name: len(ids) for name, ids in result.named_solid_entity_ids},
        "faces": {name: len(ids) for name, ids in result.named_face_entity_ids},
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_layout_stack()
