from fractions import Fraction

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain


def _tetrahedral_reference() -> tuple[PiecewiseLinearDomain, CellMesh, CellGeometrySpec]:
    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    domain = PiecewiseLinearDomain(
        vertices,
        np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64),
        np.asarray(((0, -1), (0, -1), (0, -1), (0, -1)), dtype=np.int64),
        ("solid",),
        source_id="tetra-domain",
    )
    mesh = CellMesh(
        vertices,
        (CellBlock("cells", "tetrahedron", np.asarray(((0, 1, 2, 3),))),),
        vertex_global_ids=np.asarray((11, 13, 17, 19), dtype=np.int64),
    )
    return domain, mesh, CellGeometrySpec.affine(mesh)


def test_piecewise_domain_requires_closed_positive_source_boundary() -> None:
    domain, _, _ = _tetrahedral_reference()

    assert domain.exact_region_measures() == (Fraction(1, 6),)

    with pytest.raises(ValueError, match="closed, positively oriented"):
        PiecewiseLinearDomain(
            domain.vertices,
            np.asarray(((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2))),
            domain.facet_regions,
            domain.region_ids,
            source_id="inverted-tetra-domain",
        )


def test_mapped_reference_domain_resolves_source_vertex_occurrences_and_exact_measure() -> (
    None
):
    reference_domain, mesh, geometry = _tetrahedral_reference()
    domain = MappedReferenceDomain(
        reference_domain,
        mesh,
        geometry,
        np.asarray((0,), dtype=np.int64),
        source_id="mapped-tetra",
        source_revision="revision-1",
        coordinate_contract=SpatialCoordinateContract.si(),
        root_vertex_occurrence_paths=((), (), ("instance-A",), ()),
    )

    row = domain.resolve_root_vertex(
        "mapped-tetra", "revision-1", 17, occurrence_path=("instance-A",)
    )
    assert row == 2
    assert domain.root_vertex_id(row) == 17
    assert domain.exact_region_measures() == (Fraction(1, 6),)
    assert (
        domain.coordinate_contract.spatial_id == SpatialCoordinateContract.si().spatial_id
    )
    local_contract = SpatialCoordinateContract(
        domain.coordinate_contract.length_unit,
        reference_frame="part-local",
    )
    local_domain = MappedReferenceDomain(
        reference_domain,
        mesh,
        geometry,
        np.asarray((0,), dtype=np.int64),
        source_id="mapped-tetra",
        source_revision="revision-1",
        coordinate_contract=local_contract,
        root_vertex_occurrence_paths=((), (), ("instance-A",), ()),
    )
    assert local_domain.domain_id != domain.domain_id

    with pytest.raises(ValueError, match="different source revision"):
        domain.resolve_root_vertex("mapped-tetra", "revision-2", 17)
    with pytest.raises(KeyError, match="absent"):
        domain.resolve_root_vertex("mapped-tetra", "revision-1", 17)
