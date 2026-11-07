from __future__ import annotations

from fractions import Fraction

import equinox as eqx
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._planar_coverage import PlcSourcePointWitness
from phydrax.meshing._envelope_ancestry import EnvelopeAncestry
from phydrax.meshing._surface_envelope import SurfaceEnvelopeSource
from tests.unit.meshing.test_native_surface_envelope_route import (
    _reconstructed_tetrahedron,
)


def _source_and_witness() -> tuple[SurfaceEnvelopeSource, PlcSourcePointWitness, int]:
    original = _reconstructed_tetrahedron()
    source = SurfaceEnvelopeSource(
        original,
        source_id=f"{original.source_id}:unsigned-distance-envelope",
        coordinate_contract=SpatialCoordinateContract.si(),
        offset=1.5,
        spacing=0.4,
        certificate_spacing=0.4,
        maximum_deviation=3.0,
        envelope_mode="remeshed",
    )
    facets = np.asarray(source.domain.facets, dtype=np.int64)
    first, second, third = (int(value) for value in facets[0])
    edges = np.asarray(source.edges, dtype=np.int64)
    edge_row = next(
        row
        for row, edge in enumerate(edges.tolist())
        if set(map(int, edge)) == {first, second}
    )
    faces = facets.copy()
    faces[0] = (third, first, second)
    count = len(faces)
    witness = PlcSourcePointWitness(
        np.arange(5, dtype=np.int64),
        np.array([0, 0, 1, 2, -1], dtype=np.int64),
        np.array([first, second, 0, 0, -1], dtype=np.int64),
        np.array([[0.0, 0.0], [0.0, 0.0], [0.25, 0.0], [0.25, 0.5], [0.0, 0.0]]),
        segments=np.array([[second, first]], dtype=np.int64),
        faces=faces,
        face_groups=np.arange(count, dtype=np.int64),
        face_planes=np.full((count, 3, 3), np.nan),
        face_plane_tolerances=np.zeros(count),
        domain_face_rows=np.arange(count, dtype=np.int64),
        domain_id=source.domain.domain_id,
    )
    return source, witness, edge_row


def test_exact_parameters_follow_source_vertex_order_not_carrier_order() -> None:
    source, witness, edge_row = _source_and_witness()
    ancestry = EnvelopeAncestry(source, witness)
    assert ancestry.facet_parameters((0, 1, 2, 3), 0) == (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(3, 4), Fraction(0)),
        (Fraction(1, 2), Fraction(1, 4)),
    )
    edge = source.edges[edge_row]
    expected = (Fraction(0), Fraction(3, 4), Fraction(1))
    if int(edge[0]) != int(source.domain.facets[0, 0]):
        expected = tuple(1 - value for value in expected)
    assert ancestry.edge_parameters((0, 2, 1), edge_row) == tuple(
        (value,) for value in expected
    )


def test_wrong_edge_or_missing_source_point_is_refused() -> None:
    source, witness, edge_row = _source_and_witness()
    ancestry = EnvelopeAncestry(source, witness)
    with pytest.raises(ValueError, match="not on its declared envelope edge"):
        ancestry.edge_parameters((3,), edge_row)
    with pytest.raises(ValueError, match="no exact envelope source point"):
        ancestry.facet_parameters((4,), 0)
    facets = np.asarray(source.domain.facets).copy()
    facets[0, 2] = facets[0, 1]
    source = eqx.tree_at(lambda value: value.domain.facets, source, facets)
    with pytest.raises(ValueError, match="does not realize the envelope facets"):
        EnvelopeAncestry(source, witness)
