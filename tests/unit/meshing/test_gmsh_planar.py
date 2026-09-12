#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from importlib.util import find_spec

import numpy as np
import pytest

import phydrax as phx


pytestmark = [
    pytest.mark.meshing_gmsh,
    pytest.mark.skipif(
        find_spec("gmsh") is None, reason="optional gmsh package is not installed"
    ),
]


_EMBEDDING = phx.geometry.PlanarEmbedding(
    (2.0, -1.0, 3.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
)
_COORDINATES = phx.SpatialCoordinateContract(phx.units.MILLIMETER)


def _rectangle(x0, x1, *, feature_id):
    return phx.geometry.PlanarMeshRegion(
        np.asarray(((x0, 0.0), (x1, 0.0), (x1, 1.0), (x0, 1.0))),
        ((0, 1, 2, 3),),
        feature_id=feature_id,
    )


def _partition(tmp_path, *, hole=False):
    if hole:
        region = phx.geometry.PlanarMeshRegion(
            np.asarray(
                (
                    (0.0, 0.0),
                    (3.0, 0.0),
                    (3.0, 3.0),
                    (0.0, 3.0),
                    (1.0, 1.0),
                    (1.0, 2.0),
                    (2.0, 2.0),
                    (2.0, 1.0),
                )
            ),
            ((0, 1, 2, 3), (4, 5, 6, 7)),
            feature_id="region",
        )
        operands = (
            phx.geometry.PlanarPartitionOperand(
                "region", region, phx.geometry.BRepPartitionRole.REGION
            ),
        )
        precedence = ("region",)
    else:
        operands = (
            phx.geometry.PlanarPartitionOperand(
                "left",
                _rectangle(0.0, 1.0, feature_id="left"),
                phx.geometry.BRepPartitionRole.REGION,
            ),
            phx.geometry.PlanarPartitionOperand(
                "right",
                _rectangle(1.0, 2.0, feature_id="right"),
                phx.geometry.BRepPartitionRole.REGION,
            ),
        )
        precedence = ("left", "right")
    return phx.geometry.partition_planar(
        phx.geometry.PlanarPartitionPlan(
            _COORDINATES,
            _EMBEDDING,
            operands,
            phx.geometry.BRepPartitionPolicy(precedence),
        ),
        destination=tmp_path / ("hole.brep" if hole else "regions.brep"),
        linear_deflection=0.05,
        angular_deflection=0.2,
    )


def _semantic_spec(provider, source, kind, *, embedding=_EMBEDDING):
    scope = provider.whole_scope(source, 2)
    regions = tuple(
        phx.meshing.RegionControl(
            provider.entity_scope(source, region.entity_ids),
            region.name,
            f"material:{region.name}",
            phx.meshing.RegionRole.SOLID,
        )
        for region in source.regions
    )
    patches = tuple(
        phx.meshing.PatchControl(
            patch.name,
            provider.entity_scope(source, patch.entity_ids),
            patch.adjacent_region_ids,
        )
        for patch in source.patches
        if len(set(patch.adjacent_region_ids)) == len(patch.adjacent_region_ids)
    )
    target = phx.meshing.CellMeshingTarget(
        2,
        2,
        phx.meshing.CellFamilyPolicy(required=(kind,)),
        geometry_order=1,
    )
    return phx.meshing.SurfaceMeshingSpec(
        target,
        scope,
        planar_embedding=embedding,
        size_controls=(phx.meshing.UniformSizeControl(scope, 0.2),),
        region_controls=regions,
        patch_controls=patches,
    )


@pytest.mark.parametrize("kind", ("triangle", "quadrilateral"))
def test_real_gmsh_planar_regions_interfaces_and_exterior_patches(tmp_path, kind):
    source = _partition(tmp_path)
    provider = phx.meshing.GmshProvider()

    result = provider.plan(source, _semantic_spec(provider, source, kind)).execute()

    assert result.mesh.ambient_dimension == 2
    assert result.boundary is None
    assert {block.cell_kind for block in result.mesh.blocks} == {kind}
    assert {zone.name for zone in result.zones} == {"left", "right"}
    assert {patch.name for patch in result.patches} == {
        patch.name
        for patch in source.patches
        if len(set(patch.adjacent_region_ids)) == len(patch.adjacent_region_ids)
    }
    assert {association.target_entity_set_id for association in result.associations} == {
        result.mesh.entity_set(1).entity_set_id,
        result.mesh.entity_set(2).entity_set_id,
    }
    assert all(association.complete for association in result.associations)
    assert result.audit.passed and result.compliance.passed
    coordinates = np.asarray(result.mesh.coordinates)
    assert np.min(coordinates[:, 0]) == pytest.approx(0.0)
    assert np.max(coordinates[:, 0]) == pytest.approx(2.0)
    assert np.min(coordinates[:, 1]) == pytest.approx(0.0)
    assert np.max(coordinates[:, 1]) == pytest.approx(1.0)


def test_real_gmsh_planar_hole_has_complete_main_mesh_boundary_patch(tmp_path):
    source = _partition(tmp_path, hole=True)
    provider = phx.meshing.GmshProvider()

    result = provider.plan(source, _semantic_spec(provider, source, "triangle")).execute()

    boundary_edges = np.asarray(result.mesh.connectivity.boundary_edges, dtype=bool)
    assert np.count_nonzero(boundary_edges) > len(source.model.edge_ids)
    assert len(result.patches) == 1
    assert (
        result.patches[0].scope.entity_set_id == result.mesh.entity_set(1).entity_set_id
    )
    assert result.associations[1].complete
    assert result.audit.passed


def test_real_gmsh_rejects_nodes_outside_declared_embedding_plane(tmp_path):
    source = _partition(tmp_path)
    provider = phx.meshing.GmshProvider()
    wrong = phx.geometry.PlanarEmbedding(
        (2.5, -1.0, 3.0),
        _EMBEDDING.x_axis,
        _EMBEDDING.y_axis,
        _EMBEDDING.normal,
    )

    with pytest.raises(phx.meshing.MeshingFailure, match="embedding"):
        provider.plan(
            source,
            _semantic_spec(provider, source, "triangle", embedding=wrong),
        ).execute()
