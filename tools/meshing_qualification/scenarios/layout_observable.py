# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import phydrax as phx
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import _contract, qualification_failure
from tools.meshing_qualification.scenarios.layout import _decode_layout


def qualify_layout_observable() -> dict[str, object]:

    result = _decode_layout()
    region = result.model.regions[0]
    expected_layer = phx.interchange.LayoutLayerKey(1, 0)
    if (
        not result.report.valid
        or region.layer != expected_layer
        or result.source_digest != result.resource_manifest.content_sha256
        or result.model.coordinate_contract != _contract()
    ):
        raise qualification_failure(
            "qualify_layout_observable",
            ("Layout decoder omitted observable identity or layer evidence."),
            {"expected_layer": expected_layer, "region": region, "result": result},
        )
    return {
        "layout_model_id": result.model.model_id,
        "source_digest": result.source_digest,
        "layer": [region.layer.layer, region.layer.datatype],
        "occurrence_path": list(region.occurrence_path),
        "provenance_id": region.provenance_id,
        "coordinate_contract": result.model.coordinate_contract.spatial_id,
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_layout_observable()
