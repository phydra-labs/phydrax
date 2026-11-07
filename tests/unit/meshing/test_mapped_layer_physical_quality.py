from dataclasses import replace

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import CellGeometrySpec, CellMesh
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import RegionSeed
from phydrax.meshing._layer_core_mapped import mapped_core_quality_findings
from phydrax.meshing._layer_core_topology import LayerCoreConstruction
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._volume_generation import generate_plc_volume, NativeVolumeSchedule
from phydrax.meshing.providers._native_sources import NativePlcSource


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore is unavailable")
def test_mapped_core_quality_uses_physical_not_reference_tetrahedra() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    complex_ = PiecewiseLinearComplex(
        points,
        np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64),
        np.arange(4, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (4, 1)),
        ("solid",),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=("origin", "x", "y", "z"),
        vertex_occurrence_paths=((),) * 4,
        facet_ids=("opposite-origin", "opposite-x", "opposite-y", "opposite-z"),
        facet_occurrence_paths=((),) * 4,
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((),),
    )
    source = NativePlcSource(complex_, "quality-reference", "revision")
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        region_seeds=(
            RegionSeed(
                np.asarray((0.25, 0.25, 0.25), dtype=np.float64),
                "solid",
                "material",
                RegionRole.SOLID,
            ),
        ),
    )
    schedule = NativeVolumeSchedule()
    core = generate_plc_volume(
        complex_,
        specification,
        schedule,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_map_id="",
        frame_id=SpatialCoordinateContract.si().spatial_id,
    )
    reference = core.mesh
    assert isinstance(core.domain, PiecewiseLinearDomain)
    ids = np.sort(np.asarray(reference.entity_set(3).entity_ids, dtype=np.int64))
    construction = LayerCoreConstruction(
        mesh=reference,
        domain=core.domain,
        cell_regions=core.cell_regions,
        zones=(),
        patches=(),
        labels=(),
        attributes=(),
        associations=(),
        geometry=CellGeometrySpec.affine(reference),
        core=core,
        core_vertex_map=np.arange(reference.coordinates.shape[0], dtype=np.int64),
        core_cell_ids=ids,
        region_ids=core.region_ids,
        source_id=source.source_id,
        source_revision=source.source_revision,
        periodic_topology=None,
        periodic_requested=(),
        periodic_achieved=(),
        reference_maps=(),
        source_mesh=None,
        source_geometry=None,
    )
    reference_findings, reference_candidate = mapped_core_quality_findings(
        construction, schedule
    )
    assert not reference_findings

    physical_points = np.array(reference.coordinates, dtype=np.float64)
    physical_points[:, 0] *= 20.0
    physical = CellMesh(
        physical_points,
        reference.blocks,
        vertex_global_ids=reference.vertex_global_ids,
    )
    stretched = replace(
        construction,
        mesh=physical,
        geometry=CellGeometrySpec.affine(physical),
    )
    findings, physical_candidate = mapped_core_quality_findings(stretched, schedule)
    assert physical_candidate != reference_candidate
    assert all(row[2] in set(ids.tolist()) for row in findings)
    assert schedule.radius_edge_bound is not None
    assert any(row[0] == "0" and row[3] > schedule.radius_edge_bound for row in findings)

    reflected_points = np.array(reference.coordinates, dtype=np.float64)
    reflected_points[:, 0] *= -1.0
    reflected = CellMesh(
        reflected_points,
        reference.blocks,
        vertex_global_ids=reference.vertex_global_ids,
    )
    inverted, _ = mapped_core_quality_findings(
        replace(
            construction,
            mesh=reflected,
            geometry=CellGeometrySpec.affine(reflected),
        ),
        schedule,
    )
    assert any(row[0] == "4" and row[2] in ids for row in inverted)
