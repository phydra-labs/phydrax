#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer-visible mesh source associations contracts."""

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import tree_content_equal
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
    PolynomialComposedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._embedding_state import coordinate_certificate_request
from phydrax.geometry._mapped_reference_domain import MappedReferenceDomain
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing.providers._native_publication import derived_certificate_limits
from phydrax.meshing.qualification._quad_hex import curved_occupied_source
from tests._support.mesh_certificate_fixtures import (
    coordinate_budget as coordinate_budget,
    mapped_reference_domain,
    reference_budget as reference_budget,
)


@pytest.mark.parametrize("kind", ("tetrahedron", "prism", "hexahedron"))
@pytest.mark.parametrize("bowed_face", (False, True))
def test_mapped_mixed_root_associations_classify_full_polynomial_strata(
    kind: str,
    bowed_face: bool,
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh, source_geometry, reference_domain = mapped_reference_domain(kind)
    domain = MappedReferenceDomain(
        reference_domain,
        mesh,
        source_geometry,
        np.asarray((0,), dtype=np.int64),
        source_id=f"mapped-{kind}",
        source_revision="independent-mixed-root",
    )
    root = source_geometry.elements[0]
    chart = coordinate_lagrange_element(kind, 2)
    nodes = np.asarray(chart.reference_nodes, dtype=np.float64)
    controls = nodes.copy()
    if bowed_face:
        remaining = 1.0 - (
            np.sum(nodes, axis=1)
            if kind == "tetrahedron"
            else np.sum(nodes[:, :2], axis=1)
        )
        controls[:, 1] += (
            nodes[:, 0] * (1.0 - nodes[:, 0]) * (1.0 - nodes[:, 1])
            if kind == "hexahedron"
            else nodes[:, 0] * remaining
        ) / 8
    elif kind == "hexahedron":
        controls[:, 0] += nodes[:, 0] * (1.0 - nodes[:, 0]) / 8
    else:
        displacement = nodes[:, 0] * nodes[:, 1] / 8
        controls[:, 0] += displacement
        controls[:, 1] -= displacement
    exact = tuple(tuple(Fraction(float(value)) for value in row) for row in controls)
    element = PolynomialComposedCellGeometryElement(
        root,
        chart,
        np.asarray([[value.numerator for value in row] for row in exact], dtype=np.int64),
        np.asarray(
            [[value.denominator for value in row] for row in exact], dtype=np.uint64
        ),
        budget=reference_budget,
    )
    block = mesh.blocks[0]
    record = CellGeometryRestrictionSource(
        cell_geometry_id(source_geometry),
        mesh.topology_id,
        {block.name: np.asarray(block.global_ids, dtype=np.int64)},
        {block.name: np.asarray(mesh.vertex_global_ids, dtype=np.int64)[block.vertices]},
    )
    geometry = phx.discretization.CellGeometrySpec(
        {block.name: element},
        {block.name: source_geometry.geometry_dofs[0]},
        source_geometry.coordinates,
        restriction_source=record,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    transfer = phx.meshing.MappedReferenceAssociationTransfer(domain)
    associations = transfer.associations(mesh, geometry)

    assert validity.all_certified
    assert len(associations) == 4
    for degree, association in enumerate(associations):
        assert association.exact and association.complete
        assert association.source_id == domain.source_id
        assert association.source_revision == domain.source_revision
        assert association.target_entity_set_id == mesh.entity_set(degree).entity_set_id
        if not bowed_face or degree in (0, 3):
            assert np.all(np.asarray(association.parent_dimensions) == degree)
            np.testing.assert_array_equal(
                association.parent_ids, mesh.entity_set(degree).entity_ids
            )
    if bowed_face:
        # Every carrier corner stays on its source stratum, but a whole
        # polynomial face bows into the root interior and must name the cell.
        assert np.any(np.asarray(associations[2].parent_dimensions) == 3)
    topology = phx.discretization.reference_cell_topology(kind)
    capped = phx.meshing.MappedReferenceAssociationTransfer(
        domain,
        maximum_support_queries=sum(len(rows) for rows in topology.entities) - 1,
    )
    with pytest.raises(
        phx.meshing.MeshingFailure, match=r"MappedReferenceAssociation\.support_queries"
    ):
        capped.associations(mesh, geometry)


def test_family_mapped_domain_policy_roundtrip(tmp_path: Path) -> None:
    limits = phx.meshing.MeshingLimits(
        maximum_work_units=100_000_000,
        maximum_geometry_queries=10_000_000,
        maximum_scratch_bytes=1_536_000_000,
    )
    source = curved_occupied_source({(0, 0, 0)}, limits=limits)
    repeated = curved_occupied_source({(0, 0, 0)}, limits=limits)
    default = curved_occupied_source({(0, 0, 0)})
    policy = derived_certificate_limits(limits)
    assert tree_content_equal(source.domain.certificate_limits, policy)
    assert source.domain.certificate_limits.maximum_scratch_bytes == 1_536_000_000
    assert default.domain.certificate_limits.maximum_scratch_bytes == 512 * 1024 * 1024
    assert source.domain.domain_id == repeated.domain.domain_id
    assert source.binding_id == repeated.binding_id
    assert tree_content_equal(source, repeated)
    assert source.domain.domain_id != default.domain.domain_id
    assert source.binding_id != default.binding_id
    query = phx.geometry.MappedDomainBoundarySource(source.domain, source.domain.cell_regions)
    assert query.limits is source.domain.certificate_limits
    with pytest.raises(ValueError, match="must match domain.certificate_limits"):
        phx.geometry.MappedDomainBoundarySource(
            source.domain, source.domain.cell_regions,
            limits=phx.geometry.MeshCertificateLimits(),
        )
    receipt = write_meshing_source_closure(tmp_path / "mapped-policy", (source, query))
    restored, restored_query = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert tree_content_equal(restored, source)
    assert tree_content_equal(restored_query.limits, policy)
    assert restored.domain.exact_region_measures() == source.domain.exact_region_measures()


@pytest.mark.parametrize("shortfall", (0, 1))
def test_mapped_domain_authored_stage_exact_scratch_boundary(shortfall: int) -> None:
    required = 1_536_000_000
    limits = phx.meshing.MeshingLimits(
        maximum_work_units=100_000_000,
        maximum_geometry_queries=10_000_000,
        maximum_scratch_bytes=required - shortfall,
    )
    domain = curved_occupied_source({(0, 0, 0)}, limits=limits).domain
    outer = phx.geometry.MeshCertificateLimits(
        maximum_work_units=limits.maximum_work_units,
        maximum_scratch_bytes=required,
    )
    with coordinate_certificate_request(outer) as budget:
        with coordinate_certificate_request(domain.certificate_limits, budget=budget):
            if shortfall:
                with pytest.raises(CoordinateEnclosureResourceError):
                    budget.reserve(0, required)
            else:
                budget.reserve(0, required)
