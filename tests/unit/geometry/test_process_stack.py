#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.geometry.brep import brep_mass_properties
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


def test_process_stack_requires_total_precedence_and_explicit_void_targets() -> None:
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


def test_vertical_stack_lowers_to_in_memory_exact_partition_history() -> None:
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

    result = lower_process_stack(
        stack,
        linear_deflection=0.1,
        angular_deflection=0.2,
        trim_samples_per_edge=9,
    )
    mass = brep_mass_properties(result.model, relative_tolerance=1e-6)
    volume_enclosures = np.asarray(mass.volume_enclosures)

    assert stack.region_precedence == ("high", "low")
    assert mass.status == ("certified", "certified")
    ordered_volumes = volume_enclosures[np.argsort(volume_enclosures[:, 0])]
    assert ordered_volumes[0, 0] <= 2.0 <= ordered_volumes[0, 1]
    assert ordered_volumes[1, 0] <= 3.5 <= ordered_volumes[1, 1]
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
        model.source_id == f"process-stack:{stack.stack_id}:{name}"
        and model.coordinate_contract.spatial_id == contract.spatial_id
        for name, model in result.operand_models
    )
