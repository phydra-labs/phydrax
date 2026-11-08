from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._cell_mesh import CellMesh
from phydrax.geometry._affine_plc_domain import AffinePlcDomain
from phydrax.geometry._affine_plc_geometry import (
    AffinePlcGeometrySource,
    map_affine_plc_mesh,
)
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._planar_coverage import (
    OrientedFacet,
    partition_overlaps,
    source_facets,
)


def _tetra() -> tuple[PiecewiseLinearDomain, CellMesh]:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    facets = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64)
    domain = PiecewiseLinearDomain(
        points, facets, np.asarray(((0, -1),) * 4), ("solid",), source_id="source"
    )
    mesh = CellMesh.from_simplices(
        points,
        np.asarray(((0, 1, 2, 3),)),
        dimension=3,
        vertex_global_ids=np.asarray((11, 12, 13, 14)),
        cell_global_ids=np.asarray((71,)),
    )
    return domain, mesh


def test_affine_domain_reflection_and_exact_mesh_source() -> None:
    reference, mesh = _tetra()
    matrix = np.asarray(((-1.0, 0.25, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 0.5)))
    offset = np.asarray((0.1, 0.2, 0.3))
    domain = AffinePlcDomain(reference, matrix, offset, source_revision="revision")
    physical, geometry = map_affine_plc_mesh(domain, mesh)
    assert domain.determinant == -1
    assert domain.exact_region_measures() == (Fraction(1, 6),)
    assert np.array_equal(domain.facets[:, :2], reference.facets[:, 1::-1])
    assert np.array_equal(physical.vertex_global_ids, mesh.vertex_global_ids)
    assert np.array_equal(physical.blocks[0].global_ids, mesh.blocks[0].global_ids)
    assert tuple(np.asarray(physical.blocks[0].vertices)[0]) == (1, 0, 2, 3)
    assert (
        tuple(geometry.source_coordinates())
        == AffinePlcGeometrySource(domain, mesh).source_coordinates()
    )
    assert (
        domain.domain_id
        == AffinePlcDomain(
            reference, matrix, offset, source_revision="revision"
        ).domain_id
    )
    assert (
        domain.domain_id
        != AffinePlcDomain(reference, matrix, offset, source_revision="other").domain_id
    )


def test_affine_domain_rejects_invalid_transform_and_forged_region_pair() -> None:
    reference, _ = _tetra()
    with pytest.raises(ValueError):
        AffinePlcDomain(
            reference, np.zeros((3, 3)), np.zeros(3), source_revision="revision"
        )
    with pytest.raises(ValueError):
        AffinePlcDomain(
            reference,
            np.eye(3),
            np.asarray((0.0, float("nan"), 0.0)),
            source_revision="revision",
        )
    with pytest.raises(ValueError):
        AffinePlcDomain(reference, np.eye(2), np.zeros(3), source_revision="revision")
    domain = AffinePlcDomain(
        reference, np.eye(3), np.zeros(3), source_revision="revision"
    )
    facets = list(source_facets(domain))
    facet = facets[0]
    facets[0] = OrientedFacet(facet.facet_id, (-1, 0), facet.vertices)
    with pytest.raises(ValueError):
        partition_overlaps(domain, (), sources=tuple(facets))


def test_affine_archive_reconstructs_and_refuses_tampered_identity() -> None:
    from phydrax._model._structure import model_archive_projection
    from phydrax.meshing._model_codec import restore_model_archive_value

    reference, mesh = _tetra()
    domain = AffinePlcDomain(
        reference, np.eye(3), np.asarray((0.1, 0.2, 0.3)), source_revision="revision"
    )
    source = AffinePlcGeometrySource(domain, mesh)
    recipe = model_archive_projection(domain)
    restored = restore_model_archive_value(recipe)
    assert restored.domain_id == domain.domain_id
    with pytest.raises(ValueError):
        restore_model_archive_value(recipe._replace(offset=np.zeros(3)))
    source_recipe = model_archive_projection(source)
    restored_source = restore_model_archive_value(source_recipe)
    assert restored_source.source_coordinates() == source.source_coordinates()
    with pytest.raises(ValueError):
        restore_model_archive_value(
            source_recipe._replace(expected_source_map_id="forged")
        )


def test_affine_geometry_uses_exact_reference_source_points() -> None:
    from phydrax.geometry._planar_coverage import PlcSourcePointWitness

    reference, _ = _tetra()
    domain = AffinePlcDomain(
        reference, np.eye(3), np.asarray((0.1, 0.2, 0.3)), source_revision="revision"
    )
    t = Fraction.from_float(0.1)
    points = np.concatenate(
        (reference.vertices, np.asarray(((float(1 - t), float(t), 0.0),)))
    )
    mesh = CellMesh.from_simplices(
        points, np.asarray(((0, 1, 4, 3), (0, 4, 2, 3))), dimension=3
    )
    witness = PlcSourcePointWitness(
        np.asarray(mesh.vertex_global_ids),
        np.asarray((0, 0, 0, 0, 1)),
        np.asarray((0, 1, 2, 3, 0)),
        np.asarray(((0.0, 0.0),) * 4 + ((0.1, 0.0),)),
        segments=np.asarray(((1, 2),)),
        faces=reference.facets,
        face_groups=np.arange(4),
        face_planes=np.full((4, 3, 3), np.nan),
        face_plane_tolerances=np.zeros(4),
        domain_face_rows=np.arange(4),
        domain_id=domain.domain_id,
        vertex_deviations=None,
    )
    physical, geometry = map_affine_plc_mesh(domain, mesh, witness=witness)
    expected = domain.transform_point((1 - t, t, Fraction(0)))
    assert geometry.source_coordinates()[4] == expected
    assert tuple(np.asarray(physical.coordinates)[4]) == tuple(
        float(value) for value in expected
    )
    assert expected != domain.transform_point(
        tuple(Fraction(float(v)) for v in points[4])
    )
