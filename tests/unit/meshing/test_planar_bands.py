#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from importlib.util import find_spec
from pathlib import Path

import pytest

import phydrax as phx
from tests._support.planar_band_sources import (
    BAND_EMBEDDING,
    band_meshing_controls,
    interface_patch,
    rectangular_partition,
)


# Preserved unit-scale millimeter qualification threshold from the Gmsh front
# contract in test_planar_bands at merge base d2dfcb941.
_FRONT_DISTANCE_TOLERANCE = 1e-13


def test_planar_band_partition_retains_regions_and_named_fronts(
    tmp_path: Path,
) -> None:
    source = rectangular_partition(tmp_path)
    interface = interface_patch(source, "region-0", "region-1")
    # ty: ignore[invalid-argument-type]
    schedule = phx.meshing.LayerSchedule((0.05, 0.1))
    plan = phx.meshing.PlanarBandPlan(
        source,
        BAND_EMBEDDING,
        (
            phx.meshing.PlanarBandControl(
                interface.name,
                {"region-0": schedule, "region-1": schedule},
                0.2,
            ),
        ),
    )

    result = phx.meshing.prepare_planar_bands(plan, destination=tmp_path / "bands.brep")
    with pytest.raises(FileExistsError):
        phx.meshing.prepare_planar_bands(
            plan,
            destination=tmp_path / "bands.brep",
        )

    assert {region.name for region in result.partition.regions} == {
        "region-0",
        "region-1",
    }
    assert len(result.layer_partitions) == 4
    assert result.partition.model.topology.num_faces == (
        source.model.topology.num_faces + 4
    )
    assert all(
        result.partition.patch(layer.front_patch_name).entity_ids
        for layer in result.layer_partitions
    )
    coverage = result.partition.association_graph.transaction.coverage
    assert coverage.source_exhaustive and coverage.target_exhaustive


@pytest.mark.skipif(
    find_spec("gmsh") is None, reason="optional gmsh package is not installed"
)
@pytest.mark.meshing_gmsh
def test_real_gmsh_planar_band_fronts_equal_cumulative_schedule(tmp_path: Path) -> None:
    partition = rectangular_partition(tmp_path)
    interface = interface_patch(partition, "region-0", "region-1")
    # ty: ignore[invalid-argument-type]
    schedule = phx.meshing.LayerSchedule((0.05, 0.1))
    control = phx.meshing.PlanarBandControl(
        interface.name,
        {"region-0": schedule, "region-1": schedule},
        0.2,
    )
    source = phx.meshing.prepare_planar_bands(
        phx.meshing.PlanarBandPlan(partition, BAND_EMBEDDING, (control,)),
        destination=tmp_path / "mesh-bands.brep",
    )
    provider = phx.meshing.GmshProvider()
    scope = provider.whole_scope(source, 2)
    regions, patches = band_meshing_controls(provider, source)
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2,
            2,
            phx.meshing.CellFamilyPolicy(required=("quadrilateral",)),
            geometry_order=1,
        ),
        scope,
        planar_embedding=BAND_EMBEDDING,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.2),),
        region_controls=regions,
        patch_controls=patches,
    )

    result = provider.plan(source, specification).execute()

    achieved = dict(result.compliance.achieved)
    for region_name in ("region-0", "region-1"):
        for index, distance in enumerate((0.05, 0.15), start=1):
            key = f"planar_band:{control.control_id}:{region_name}:front:{index}"
            assert achieved[f"{key}:distance"] == pytest.approx(
                distance, abs=_FRONT_DISTANCE_TOLERANCE
            )
            assert achieved[f"{key}:maximum_residual"] <= _FRONT_DISTANCE_TOLERANCE
    assert "quadrilateral" in {block.cell_kind for block in result.mesh.blocks}
    assert result.boundary is None
    assert result.audit.passed and result.compliance.passed


def test_planar_band_rejects_insufficient_clearance(tmp_path: Path) -> None:
    source = rectangular_partition(tmp_path)
    interface = interface_patch(source, "region-0", "region-1")
    plan = phx.meshing.PlanarBandPlan(
        source,
        BAND_EMBEDDING,
        (
            phx.meshing.PlanarBandControl(
                interface.name,
                {
                    # ty: ignore[invalid-argument-type]
                    "region-0": phx.meshing.LayerSchedule((1.1,)),
                    # ty: ignore[invalid-argument-type]
                    "region-1": phx.meshing.LayerSchedule((0.1,)),
                },
                0.2,
            ),
        ),
    )

    with pytest.raises(ValueError, match="prepare_planar_bands.side.clearance"):
        phx.meshing.prepare_planar_bands(plan, destination=tmp_path / "no-clearance.brep")


def test_planar_band_rejects_colliding_straight_strips(tmp_path: Path) -> None:
    source = rectangular_partition(tmp_path, widths=(1.0, 0.5, 1.0))
    left = interface_patch(source, "region-0", "region-1")
    right = interface_patch(source, "region-1", "region-2")
    controls = (
        phx.meshing.PlanarBandControl(
            left.name,
            {
                # ty: ignore[invalid-argument-type]
                "region-0": phx.meshing.LayerSchedule((0.1,)),
                # ty: ignore[invalid-argument-type]
                "region-1": phx.meshing.LayerSchedule((0.3,)),
            },
            0.2,
        ),
        phx.meshing.PlanarBandControl(
            right.name,
            {
                # ty: ignore[invalid-argument-type]
                "region-1": phx.meshing.LayerSchedule((0.3,)),
                # ty: ignore[invalid-argument-type]
                "region-2": phx.meshing.LayerSchedule((0.1,)),
            },
            0.2,
        ),
    )

    with pytest.raises(ValueError, match="prepare_planar_bands.layers"):
        phx.meshing.prepare_planar_bands(
            phx.meshing.PlanarBandPlan(source, BAND_EMBEDDING, controls),
            destination=tmp_path / "collision.brep",
        )
