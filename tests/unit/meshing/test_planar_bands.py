#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from importlib.util import find_spec

import numpy as np
import pytest

import phydrax as phx


_EMBEDDING = phx.geometry.PlanarEmbedding(
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_COORDINATES = phx.SpatialCoordinateContract(phx.units.MILLIMETER)


def _rectangle(x0, x1, *, feature_id):
    return phx.geometry.PlanarMeshRegion(
        np.asarray(((x0, 0.0), (x1, 0.0), (x1, 1.0), (x0, 1.0))),
        ((0, 1, 2, 3),),
        feature_id=feature_id,
    )


def _partition(tmp_path, widths=(1.0, 1.0)):
    cursor = 0.0
    operands = []
    names = []
    for index, width in enumerate(widths):
        name = f"region-{index}"
        names.append(name)
        operands.append(
            phx.geometry.PlanarPartitionOperand(
                name,
                _rectangle(cursor, cursor + width, feature_id=name),
                phx.geometry.BRepPartitionRole.REGION,
            )
        )
        cursor += width
    return phx.geometry.partition_planar(
        phx.geometry.PlanarPartitionPlan(
            _COORDINATES,
            _EMBEDDING,
            tuple(operands),
            phx.geometry.BRepPartitionPolicy(tuple(names)),
        ),
        destination=tmp_path / "partition.brep",
        linear_deflection=0.05,
        angular_deflection=0.2,
    )


def _interface(partition, first, second):
    adjacent = tuple(sorted((first, second)))
    return next(
        patch
        for patch in partition.patches
        if tuple(sorted(patch.adjacent_region_ids)) == adjacent
    )


def _controls(provider, source):
    regions = tuple(
        phx.meshing.RegionControl(
            provider.entity_scope(source, region.entity_ids),
            region.name,
            f"material:{region.name}",
            phx.meshing.RegionRole.SOLID,
        )
        for region in source.partition.regions
    )
    patches = tuple(
        phx.meshing.PatchControl(
            patch.name,
            provider.entity_scope(source, patch.entity_ids),
            patch.adjacent_region_ids,
        )
        for patch in source.partition.patches
        if len(set(patch.adjacent_region_ids)) == len(patch.adjacent_region_ids)
    )
    return regions, patches


def test_planar_band_partition_retains_regions_and_exact_named_fronts(tmp_path):
    source = _partition(tmp_path)
    interface = _interface(source, "region-0", "region-1")
    schedule = phx.meshing.LayerSchedule((0.05, 0.1))
    plan = phx.meshing.PlanarBandPlan(
        source,
        _EMBEDDING,
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
def test_real_gmsh_planar_band_fronts_equal_cumulative_schedule(tmp_path):
    partition = _partition(tmp_path)
    interface = _interface(partition, "region-0", "region-1")
    schedule = phx.meshing.LayerSchedule((0.05, 0.1))
    control = phx.meshing.PlanarBandControl(
        interface.name,
        {"region-0": schedule, "region-1": schedule},
        0.2,
    )
    source = phx.meshing.prepare_planar_bands(
        phx.meshing.PlanarBandPlan(partition, _EMBEDDING, (control,)),
        destination=tmp_path / "mesh-bands.brep",
    )
    provider = phx.meshing.GmshProvider()
    scope = provider.whole_scope(source, 2)
    regions, patches = _controls(provider, source)
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2,
            2,
            phx.meshing.CellFamilyPolicy(required=("quadrilateral",)),
            geometry_order=1,
        ),
        scope,
        planar_embedding=_EMBEDDING,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.2),),
        region_controls=regions,
        patch_controls=patches,
    )

    result = provider.plan(source, specification).execute()

    achieved = dict(result.compliance.achieved)
    for region_name in ("region-0", "region-1"):
        for index, distance in enumerate((0.05, 0.15), start=1):
            key = f"planar_band:{control.control_id}:{region_name}:front:{index}"
            assert achieved[f"{key}:distance"] == pytest.approx(distance, abs=1e-13)
            assert achieved[f"{key}:maximum_residual"] <= 1e-13
    assert "quadrilateral" in {block.cell_kind for block in result.mesh.blocks}
    assert result.boundary is None
    assert result.audit.passed and result.compliance.passed


def test_planar_band_rejects_insufficient_clearance(tmp_path):
    source = _partition(tmp_path)
    interface = _interface(source, "region-0", "region-1")
    plan = phx.meshing.PlanarBandPlan(
        source,
        _EMBEDDING,
        (
            phx.meshing.PlanarBandControl(
                interface.name,
                {
                    "region-0": phx.meshing.LayerSchedule((1.1,)),
                    "region-1": phx.meshing.LayerSchedule((0.1,)),
                },
                0.2,
            ),
        ),
    )

    with pytest.raises(ValueError, match="clearance"):
        phx.meshing.prepare_planar_bands(plan, destination=tmp_path / "no-clearance.brep")


def test_planar_band_rejects_colliding_straight_strips(tmp_path):
    source = _partition(tmp_path, widths=(1.0, 0.5, 1.0))
    left = _interface(source, "region-0", "region-1")
    right = _interface(source, "region-1", "region-2")
    controls = (
        phx.meshing.PlanarBandControl(
            left.name,
            {
                "region-0": phx.meshing.LayerSchedule((0.1,)),
                "region-1": phx.meshing.LayerSchedule((0.3,)),
            },
            0.2,
        ),
        phx.meshing.PlanarBandControl(
            right.name,
            {
                "region-1": phx.meshing.LayerSchedule((0.3,)),
                "region-2": phx.meshing.LayerSchedule((0.1,)),
            },
            0.2,
        ),
    )

    with pytest.raises(ValueError, match="collide"):
        phx.meshing.prepare_planar_bands(
            phx.meshing.PlanarBandPlan(source, _EMBEDDING, controls),
            destination=tmp_path / "collision.brep",
        )
