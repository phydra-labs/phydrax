from __future__ import annotations

from fractions import Fraction
import subprocess
import sys

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._mapped_reference_boundary import (
    _mapped_root_entity_banks,
    MappedReferenceBoundarySource,
)
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing.providers._native_sources import NativePlcSource


def _prism_source() -> NativePlcSource:
    topology = reference_cell_topology("prism")
    vertices = np.asarray(topology.vertices, dtype=np.float64)
    ids = np.asarray((31, 37, 41, 43, 47, 53), dtype=np.int64)
    mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "root",
                "prism",
                np.arange(6, dtype=np.int32)[None, :],
                global_ids=np.asarray((901,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=ids,
    )
    triangles = tuple(
        np.asarray(triangle, dtype=np.int64)
        for face in topology.entities[2]
        for triangle in ((face[0], face[1], face[2]),)
        + (((face[0], face[2], face[3]),) if len(face) == 4 else ())
    )
    segments = np.asarray(topology.entities[1], dtype=np.int64)
    reference = PiecewiseLinearDomain(
        vertices,
        np.asarray(triangles),
        np.tile(np.asarray((0, -1), dtype=np.int64), (len(triangles), 1)),
        ("solid",),
        source_id="mapped-prism",
    )
    geometry = CellGeometrySpec.affine(mesh)
    domain = MappedReferenceDomain(
        reference,
        mesh,
        geometry,
        np.asarray((0,), dtype=np.int64),
        source_id="mapped-prism",
        source_revision="r1",
        reference_domain_vertex_root_ids=ids,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
    )
    complex_ = PiecewiseLinearComplex(
        vertices,
        triangles,
        np.arange(len(triangles), dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (len(triangles), 1)),
        ("solid",),
        segments=segments,
        vertex_ids=tuple(f"vertex-{i}" for i in range(6)),
        vertex_occurrence_paths=((),) * 6,
        facet_ids=tuple(f"facet-{i}" for i in range(len(triangles))),
        facet_occurrence_paths=((),) * len(triangles),
        segment_ids=tuple(f"edge-{i}" for i in range(len(segments))),
        segment_occurrence_paths=((),) * len(segments),
        region_occurrence_paths=((),),
        boundary="fixed",
        geometry=geometry,
    )
    return NativePlcSource(
        complex_,
        "mapped-prism",
        "r1",
        mapped_domain=domain,
        plc_vertex_root_global_ids=ids,
    )


def test_prism_query_covers_mixed_root_faces_and_checks_exact_ancestry() -> None:
    source = _prism_source()
    query = source.boundary_query()
    assert isinstance(query, MappedReferenceBoundarySource)
    assert source.mapped_domain is not None
    query.require_current()
    assert query.is_full_surface_scope
    assert query.boundary_chart_cover() is not None
    faces, cell_faces, _ = _mapped_root_entity_banks(source.mapped_domain.reference_mesh)
    assert sorted(len(face) for face in faces) == [3, 3, 4, 4, 4]
    assert set(query.polygon_source_facet_rows.tolist()) == set(
        range(len(source.complex.facet_ids))
    )
    for chart in query.face_chart_ids:
        row = query.chart_row(2, chart)
        face = int(query.polygon_root_face_global_ids[row])
        cell = int(query.polygon_root_cell_global_ids[row])
        local = int(query.polygon_root_local_face_indices[row])
        assert cell == 901
        assert len(query.chart_face_uv(chart)) == 3
        points = tuple(
            tuple(Fraction(*value) for value in point)
            for point in query.polygon_root_reference_rational_tokens[row]
        )
        with pytest.raises(ValueError):
            query.root_face_uv(face, cell + 1, local, points)
        with pytest.raises(ValueError):
            query.root_face_uv(face, cell, (local + 1) % cell_faces.shape[1], points)
        with pytest.raises(ValueError):
            query.root_face_uv(
                face, cell, local, ((Fraction(2), Fraction(2), Fraction(2)),)
            )


def test_prism_query_reconstructs_exact_source_and_rejects_wrong_vertex_relation() -> (
    None
):
    first = _prism_source()
    second = _prism_source()
    query = first.boundary_query()
    reconstructed = second.boundary_query()
    assert isinstance(query, MappedReferenceBoundarySource)
    assert isinstance(reconstructed, MappedReferenceBoundarySource)
    assert query.query_id == reconstructed.query_id
    assert query.face_chart_ids == reconstructed.face_chart_ids
    assert tuple(query.chart_face_uv(chart) for chart in query.face_chart_ids) == tuple(
        reconstructed.chart_face_uv(chart) for chart in reconstructed.face_chart_ids
    )
    assert first.root_relation is not None
    ids = np.asarray(first.root_relation.vertex_root_global_ids).copy()
    ids[0], ids[1] = ids[1], ids[0]
    with pytest.raises(ValueError):
        NativePlcSource(
            first.complex,
            first.source_id,
            first.source_revision,
            mapped_domain=first.mapped_domain,
            plc_vertex_root_global_ids=ids,
        )


def test_prism_query_reconstructs_in_fresh_process() -> None:
    query = _prism_source().boundary_query()
    assert isinstance(query, MappedReferenceBoundarySource)
    script = (
        "from tests.unit.meshing.test_native_mapped_prism_source import _prism_source; "
        "q = _prism_source().boundary_query(); q.require_current(); "
        "q.boundary_chart_cover(); print(q.query_id)"
    )
    result = subprocess.run(
        (sys.executable, "-c", script), check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == query.query_id
