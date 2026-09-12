#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from OCP.BRepGProp import BRepGProp  # ty: ignore[unresolved-import]
from OCP.GProp import GProp_GProps  # ty: ignore[unresolved-import]

from phydrax import SpatialCoordinateContract
from phydrax.geometry.brep import read_occt_shape
from phydrax.geometry.process import (
    lower_process_stack,
    ProcessStack,
    StackRegion,
    StackVoid,
    ZInterval,
)
from phydrax.geometry.simplicial import PlanarMeshRegion
from phydrax.units import MILLIMETER


def _rectangle(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    feature_id: str,
) -> PlanarMeshRegion:
    return PlanarMeshRegion(
        np.asarray(
            (
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            )
        ),
        ((0, 1, 2, 3),),
        feature_id=feature_id,
    )


def _volume(path: Path) -> float:
    shape, _, _ = read_occt_shape(path)
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return float(properties.Mass())


def test_process_stack_requires_total_precedence_and_explicit_void_targets():
    contract = SpatialCoordinateContract(MILLIMETER)
    footprint = _rectangle(0, 0, 1, 1, "footprint")

    with pytest.raises(ValueError, match="distinct precedence"):
        ProcessStack(
            contract,
            (
                StackRegion("first", footprint, ZInterval(0, 1), 0),
                StackRegion("second", footprint, ZInterval(0, 1), 0),
            ),
        )

    with pytest.raises(ValueError, match="unknown regions"):
        ProcessStack(
            contract,
            (StackRegion("region", footprint, ZInterval(0, 1), 0),),
            (
                StackVoid(
                    "void",
                    footprint,
                    ZInterval(0, 1),
                    ("not-a-region",),
                ),
            ),
        )


def test_vertical_stack_lowers_to_persisted_exact_partition_history(tmp_path: Path):
    contract = SpatialCoordinateContract(MILLIMETER)
    stack = ProcessStack(
        contract,
        (
            StackRegion(
                "low",
                _rectangle(0, 0, 2, 2, "low-footprint"),
                ZInterval(0, 1),
                0,
            ),
            StackRegion(
                "high",
                _rectangle(1, 0, 3, 2, "high-footprint"),
                ZInterval(0, 1),
                10,
            ),
        ),
        (
            StackVoid(
                "opening",
                _rectangle(0.5, 0.5, 1.5, 1.5, "opening-footprint"),
                ZInterval(0, 1),
                ("high",),
            ),
        ),
    )
    destination = tmp_path / "final-stack.brep"

    result = lower_process_stack(
        stack,
        destination,
        linear_deflection=0.1,
        angular_deflection=0.2,
        trim_samples_per_edge=9,
    )

    assert stack.region_precedence == ("high", "low")
    assert destination.is_file()
    assert _volume(destination) == pytest.approx(5.5)
    assert result.model.report.num_solids == 2
    assert result.model.coordinate_contract.spatial_id == contract.spatial_id
    assert result.revision.revision_id == result.model.source_revision
    assert result.association_graph.target_revision == result.revision
    assert result.association_graph.source_revision != result.revision
    assert result.report.history_certificate_id == (
        result.association_graph.transaction.coverage.certificate_id
    )

    assert [region.name for region in result.regions] == ["high", "low"]
    assert set(result.model.solid_ids) == {
        entity_id
        for _, entity_ids in result.named_solid_entity_ids
        for entity_id in entity_ids
    }
    assert set(result.model.face_ids) == {
        entity_id
        for _, entity_ids in result.named_face_entity_ids
        for entity_id in entity_ids
    }
    assert any(
        patch.name == "interface:high:low"
        and patch.adjacent_region_ids == ("high", "low")
        for patch in result.patches
    )
    assert tuple(name for name, _ in result.operand_models) == (
        "low",
        "high",
        "opening",
    )
    assert all(
        Path(model.source_id).is_file()
        and model.coordinate_contract.spatial_id == contract.spatial_id
        for _, model in result.operand_models
    )
