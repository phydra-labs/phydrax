#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Consumer-visible global embedding certificates contracts."""

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry import (
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import (
    ExactScalar,
    Expression,
    Polynomial,
)
from phydrax.discretization._coordinate_identity import coordinate_scope_key
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._embedding_state import (
    EmbeddingState,
    MeshCertificateLimits,
    PeriodicAffineQuotientEvidence,
    PeriodicAffineTorusDegreeEvidence,
)
from phydrax.geometry._global_embedding import (
    cell_global_ids,
    coordinate_scope,
    EmbeddingCounts,
    GlobalEmbeddingCertificate,
)
from phydrax.geometry._mapped_embedding import certify_mapped_embedding
from phydrax.geometry.brep._patches import SpherePatch
from tests._support.mesh_certificate_fixtures import (
    affine_tetrahedral_root_restrictions,
    annulus_with_island,
    antisymmetric,
    bumped_square,
    coordinate_budget as coordinate_budget,
    curved_shared_hex_interface,
    curved_tetrahedral_components,
    embedding,
    halved_tapered_restriction,
    mixed_degree_hexes,
    reference_budget as reference_budget,
    rotated_neighbor_hexes,
    straight,
    symmetric,
    tapered_hexes,
    TETRAHEDRON,
    two_tetrahedra,
)
from tests._support.meshing_fixtures import finding_checks


@pytest.mark.parametrize("dimension", (2, 3))
def test_exact_affine_rank_proof_avoids_spd_nets_without_relaxing_piece_cap(
    dimension: int,
    monkeypatch: pytest.MonkeyPatch,
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.geometry import _mapped_embedding as owner
    from phydrax.geometry._embedding_cells import Cell, vertices

    kind = "triangle" if dimension == 2 else "tetrahedron"
    coordinates: tuple[Expression, ...] = tuple(
        {
            tuple(int(axis == column) for column in range(dimension)):
            Fraction(1, 2**100) if axis == dimension - 1 else Fraction(1)
        }
        for axis in range(dimension)
    )
    cell = Cell(coordinates, "simplex", kind, dimension,
                tuple(range(dimension + 1)), vertices(kind))
    original_spd = owner.strict_spd

    def unexpected_spd(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an exact full-rank affine map needs no SPD net")

    monkeypatch.setattr(owner, "strict_spd", unexpected_spd)
    state = EmbeddingState([], [])
    assert owner.injective(
        cell, MeshCertificateLimits(), state, budget=reference_budget
    ) is None
    assert state.subdivision_pieces == 1
    capped = EmbeddingState([], [])
    assert owner.injective(
        cell, MeshCertificateLimits(), capped, maximum_work=0, budget=reference_budget
    ) == "local_injectivity_piece_budget"
    assert capped.subdivision_pieces == 0
    monkeypatch.setattr(owner, "strict_spd", original_spd)
    zero: Expression = {}
    singular = Cell((*coordinates[:-1], zero), "simplex", kind, dimension,
                    tuple(range(dimension + 1)), vertices(kind))
    assert owner.injective(
        singular, MeshCertificateLimits(), EmbeddingState([], []), budget=reference_budget
    ) is not None


@pytest.fixture(scope="module")
def closed_affine_quotient_certificate() -> GlobalEmbeddingCertificate:
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    block = phx.discretization.CellBlock(
        "torus", "triangle", np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    )
    lifted = phx.discretization.CellMesh(points, (block,))
    topology = phx.discretization.PeriodicMeshTopology(
        lifted,
        phx.discretization.PeriodicCell(np.eye(2, dtype=np.float64)),
        np.zeros(4, dtype=np.int64),
        points.astype(np.int64),
    )
    mesh = phx.discretization.CellMesh(points, (block,), periodic_topology=topology)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert certificate.status == "certified"
    assert certificate.periodic_affine_degree is not None
    return certificate


def test_global_certificate_restores_bound_periodic_affine_premises(
    closed_affine_quotient_certificate: GlobalEmbeddingCertificate,
) -> None:
    certificate = closed_affine_quotient_certificate
    restored = GlobalEmbeddingCertificate.from_stored_fields(certificate)
    assert restored.certificate_id == certificate.certificate_id
    assert restored.periodic_affine_degree is not None
    assert certificate.periodic_affine_degree is not None
    assert restored.periodic_affine_degree.evidence_id == certificate.periodic_affine_degree.evidence_id
    assert restored.periodic_affine_degree.mesh_id == certificate.binding.mesh_id
    assert restored.periodic_affine_degree.geometry_id == certificate.binding.geometry_id
    assert restored.periodic_affine_degree.covering_degree == 1
    assert restored.periodic_affine_degree.pullback_area == restored.periodic_affine_degree.lattice_area
    assert restored.validity_certificate_id == certificate.validity_certificate_id


@pytest.mark.parametrize(
    "foreign_field", ("mesh_id", "geometry_id", "validity_certificate_id")
)
def test_global_certificate_refuses_foreign_periodic_affine_binding(
    closed_affine_quotient_certificate: GlobalEmbeddingCertificate,
    foreign_field: str,
) -> None:
    certificate = closed_affine_quotient_certificate
    evidence = certificate.periodic_affine_degree
    assert evidence is not None
    identities = {
        "mesh_id": evidence.mesh_id,
        "geometry_id": evidence.geometry_id,
        "periodic_topology_id": evidence.periodic_topology_id,
        "validity_certificate_id": certificate.validity_certificate_id,
        "affinity_certificate_id": evidence.affinity_certificate_id,
        "facet_trace_certificate_id": evidence.trace_certificate_id,
    }
    identities[foreign_field] = phx.canonical_fingerprint(
        ("foreign-periodic-premise", foreign_field)
    )
    foreign = PeriodicAffineQuotientEvidence(**identities)
    with pytest.raises(ValueError, match="GlobalEmbeddingCertificate.periodic_affine"):
        GlobalEmbeddingCertificate(
            certificate.binding,
            certificate.validity_certificate_id,
            certificate.findings,
            certificate.evaluated_checks,
            counts=EmbeddingCounts(
                certificate.cell_count,
                certificate.boundary_facet_count,
                certificate.shell_count,
                certificate.candidate_pair_count,
                certificate.ray_test_count,
                certificate.subdivision_piece_count,
                certificate.periodic_image_count,
            ),
            periodic_affine=foreign,
            periodic_affine_degree=evidence,
        )



@pytest.mark.parametrize("foreign_field", ("mesh_id", "geometry_id"))
def test_global_certificate_refuses_foreign_torus_degree_binding(
    closed_affine_quotient_certificate: GlobalEmbeddingCertificate,
    foreign_field: str,
) -> None:
    certificate = closed_affine_quotient_certificate
    evidence = certificate.periodic_affine_degree
    assert evidence is not None
    identities = {
        name: getattr(evidence, name)
        for name in (
            "mesh_id", "geometry_id", "periodic_topology_id", "input_binding_id",
            "affinity_certificate_id", "manifold_certificate_id",
            "trace_certificate_id", "orientation_certificate_id",
        )
    }
    identities[foreign_field] = phx.canonical_fingerprint(("foreign-degree", foreign_field))
    foreign = PeriodicAffineTorusDegreeEvidence(
        **identities, pullback_area=evidence.pullback_area, lattice_area=evidence.lattice_area
    )
    with pytest.raises(ValueError, match="GlobalEmbeddingCertificate.periodic_affine_degree"):
        GlobalEmbeddingCertificate(
            certificate.binding, certificate.validity_certificate_id,
            certificate.findings, certificate.evaluated_checks,
            counts=EmbeddingCounts(
                certificate.cell_count, certificate.boundary_facet_count,
                certificate.shell_count, certificate.candidate_pair_count,
                certificate.ray_test_count, certificate.subdivision_piece_count,
                certificate.periodic_image_count,
            ),
            periodic_affine_degree=foreign,
        )

@pytest.mark.parametrize(
    ("points", "cells", "status"),
    (
        pytest.param(
            ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 0.5), (1.0, -1.0, 0.5)),
            ((0, 1), (2, 3)),
            "certified",
            id="skew-overlapping-boxes",
        ),
        pytest.param(
            ((-1.0, -1.0, -2.0), (1.0, 1.0, 2.0), (-1.0, 1.0, 0.0), (1.0, -1.0, 0.0)),
            ((0, 1), (2, 3)),
            "violated",
            id="tilted-plane-crossing",
        ),
        pytest.param(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
            ((0, 1), (1, 2)),
            "certified",
            id="shared-axis-endpoint",
        ),
        pytest.param(
            ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0)),
            ((0, 1), (1, 2)),
            "certified",
            id="shared-bent-endpoint",
        ),
        pytest.param(
            ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0, 1), (1, 2)),
            "violated",
            id="shared-axis-overlap",
        ),
        pytest.param(
            ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
            ((0, 1), (2, 3)),
            "violated",
            id="disconnected-axis-overlap",
        ),
        pytest.param(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ((0, 1), (2, 3)),
            "violated",
            id="unshared-endpoint",
        ),
    ),
)
def test_spatial_interval_embedding_uses_exact_contacts(
    points: tuple[tuple[float, ...], ...],
    cells: tuple[tuple[int, int], ...],
    status: str,
) -> None:
    mesh = phx.discretization.CellMesh(
        np.asarray(points, dtype=np.float64),
        (
            phx.discretization.CellBlock(
                "segments", "interval", np.asarray(cells, dtype=np.int32)
            ),
        ),
    )
    _, validity, certificate = embedding(mesh)

    assert validity.all_certified
    assert certificate.status == status
    assert "cell_contact" in certificate.evaluated_checks
    if status == "violated":
        assert "cell_contact" in finding_checks(certificate, "violated")
    else:
        assert certificate.findings == ()


def test_spatial_interval_embedding_preserves_candidate_capacity_refusal() -> None:
    mesh = phx.discretization.CellMesh(
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (2.0, 0.0, 0.0), (3.0, 1.0, 0.0)),
            dtype=np.float64,
        ),
        (
            phx.discretization.CellBlock(
                "segments",
                "interval",
                np.asarray(((0, 1), (1, 2), (2, 3)), dtype=np.int32),
            ),
        ),
    )
    geometry, validity, _ = embedding(mesh)
    certificate = phx.geometry.certify_global_embedding(
        mesh,
        geometry,
        validity,
        limits=phx.geometry.MeshCertificateLimits(maximum_candidate_pairs=1),
    )

    assert certificate.status == "unresolved"
    assert "cell_contact_capacity" in finding_checks(certificate, "unresolved")


@pytest.mark.parametrize(
    "mesh",
    [
        pytest.param(
            phx.discretization.CellMesh.from_tetrahedra(
                np.concatenate((TETRAHEDRON, ((1.0, 1.0, 1.0),))),
                np.asarray(((0, 1, 2, 3), (1, 2, 3, 4))),
            ),
            id="shared-face",
        ),
        pytest.param(two_tetrahedra(TETRAHEDRON + 5.0), id="disjoint-components"),
    ],
)
def test_embedded_tetrahedral_meshes_are_certified(mesh: Any) -> None:
    _, validity, certificate = embedding(mesh)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert certificate.findings == ()
    assert "exterior_degree" in certificate.evaluated_checks
    assert certificate.binding.topology_id == mesh.topology_id


def test_positive_jacobian_overlapping_components_are_not_embedded() -> None:
    mesh = two_tetrahedra(TETRAHEDRON + 0.25)
    _, validity, certificate = embedding(mesh)

    # Every cell is positively oriented; only the global check sees the overlap.
    assert validity.all_certified
    assert certificate.status == "violated"
    assert "boundary_contact" in finding_checks(certificate, "violated")
    contact = next(v for v in certificate.findings if v.check == "boundary_contact")
    assert contact.entity_kind == "facet"
    assert contact.entity_ids


def test_contained_component_without_boundary_contact_is_detected() -> None:
    mesh = two_tetrahedra(0.1 * TETRAHEDRON + 0.2)
    _, validity, certificate = embedding(mesh)
    audit = phx.meshing.audit_cell_mesh(
        mesh, phx.discretization.CellGeometrySpec.affine(mesh)
    )

    # The corner audit (boundary self-intersection) cannot see a contained shell.
    assert audit.passed
    assert validity.all_certified
    assert certificate.status == "violated"
    assert finding_checks(certificate, "violated") == {"nested_component"}


@pytest.mark.parametrize(
    ("origin", "status"),
    [
        pytest.param((1.25, 1.25), "certified", id="island-in-cavity"),
        pytest.param((0.25, 0.25), "violated", id="island-inside-material"),
    ],
)
def test_nested_planar_components_follow_the_exterior_degree(
    origin: tuple[float, float], status: str
) -> None:
    _, _, certificate = embedding(annulus_with_island(origin))

    assert certificate.status == status
    assert certificate.shell_count == 3
    if status == "violated":
        assert finding_checks(certificate, "violated") == {"nested_component"}


def test_continuous_curved_triangle_embedding_is_certified() -> None:
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))), np.asarray(((0, 1, 2),))
    )
    element = phx.discretization.fem.lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    nodes[:, 0] += 0.0625 * nodes[:, 0] * nodes[:, 1]
    geometry = phx.discretization.CellGeometrySpec(
        {"triangles": element}, {"triangles": np.arange(nodes.shape[0])[None]}, nodes
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert "mapped_local_injectivity" in certificate.evaluated_checks
    assert certificate.findings == ()
    assert certificate.source_expression_work_units > 1
    assert certificate.source_expression_peak_bytes > 1
    for limits in (
        phx.geometry.MeshCertificateLimits(maximum_work_units=1),
        phx.geometry.MeshCertificateLimits(maximum_scratch_bytes=1),
    ):
        capped = phx.geometry.certify_global_embedding(
            mesh, geometry, validity, limits=limits
        )
        assert capped.status == "unresolved"
        assert "mapped_source_expression_resource_budget" in finding_checks(
            capped, "unresolved"
        )
        assert capped.source_expression_work_units <= limits.maximum_work_units
        assert capped.source_expression_peak_bytes <= limits.maximum_scratch_bytes


def test_embedding_refuses_a_validity_certificate_of_other_coordinates() -> None:
    mesh = two_tetrahedra(TETRAHEDRON + 5.0)
    moved = mesh.with_coordinates(mesh.coordinates + 1.0, numeric_version="moved")
    validity = phx.discretization.certify_cell_geometry_validity(mesh)

    with pytest.raises(ValueError, match=r"CellValidityCertificate\.geometry_id"):
        phx.geometry.certify_global_embedding(
            moved, phx.discretization.CellGeometrySpec.affine(moved), validity
        )


@pytest.mark.parametrize("degree", (2, 3, 4))
def test_curved_disconnected_volume_components_have_a_continuous_embedding(
    degree: int,
) -> None:
    mesh, geometry, validity = curved_tetrahedral_components(1.0, 3.0, degree)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert certificate.findings == ()


@pytest.mark.parametrize(
    ("scale", "offset"),
    ((1.0, 0.25), (0.1, 0.2)),
    ids=("intersecting-components", "contained-component"),
)
def test_positive_jacobian_curved_volume_overlap_has_certified_counterexample(
    scale: float, offset: float
) -> None:
    mesh, geometry, validity = curved_tetrahedral_components(scale, offset, 2)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert "mapped_cell_overlap" in finding_checks(certificate, "violated")
    assert (
        certificate.subdivision_piece_count
        <= phx.geometry.MeshCertificateLimits().maximum_subdivision_pieces
    )


@pytest.mark.parametrize("kind", ("prism", "pyramid", "hexahedron"))
@pytest.mark.parametrize("degree", (2, 3))
def test_standard_hybrid_curved_maps_require_and_satisfy_local_injectivity(
    kind: str, degree: int
) -> None:

    element = coordinate_lagrange_element(kind, degree)
    points = np.asarray(element.reference_nodes, dtype=np.float64).copy()
    corners = np.asarray(
        phx.discretization.reference_cell_topology(kind).vertices, dtype=np.float64
    ).copy()
    points[:, 0] += 0.03125 * points[:, 2] ** 2
    corners[:, 0] += 0.03125 * corners[:, 2] ** 2
    mesh = phx.discretization.CellMesh(
        corners,
        (
            phx.discretization.CellBlock(
                "cells", kind, np.arange(corners.shape[0], dtype=np.int32)[None]
            ),
        ),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"cells": element},
        {"cells": np.arange(points.shape[0], dtype=np.int32)[None]},
        points,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert "mapped_local_injectivity" in certificate.evaluated_checks


def test_positive_gram_determinant_does_not_certify_a_self_crossing_curve() -> None:

    element = coordinate_lagrange_element("interval", 3)
    t = np.asarray(element.reference_nodes, dtype=np.float64)[:, 0]
    points = np.stack((t * (1.0 - t), (t - 0.5) * (t * (1.0 - t) - 0.125)), axis=-1)
    mesh = phx.discretization.CellMesh(
        points[[0, -1]],
        (
            phx.discretization.CellBlock(
                "cells", "interval", np.asarray(((0, 1),), dtype=np.int32)
            ),
        ),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"cells": element}, {"cells": np.arange(4, dtype=np.int32)[None]}, points
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "violated"
    assert "mapped_self_contact" in finding_checks(certificate, "violated")


def test_nonplanar_trilinear_hex_faces_are_not_silently_triangulated() -> None:
    logical = np.asarray(tuple(np.ndindex((3, 3, 3))), dtype=np.float64) / 2.0
    points = logical.copy()
    points[:, 2] += 0.125 * logical[:, 0] * logical[:, 1]
    index = np.arange(27, dtype=np.int32).reshape((3, 3, 3))
    cells = np.asarray(
        [
            (
                index[i, j, k],
                index[i + 1, j, k],
                index[i + 1, j + 1, k],
                index[i, j + 1, k],
                index[i, j, k + 1],
                index[i + 1, j, k + 1],
                index[i + 1, j + 1, k + 1],
                index[i, j + 1, k + 1],
            )
            for i, j, k in np.ndindex((2, 2, 2))
        ],
        dtype=np.int32,
    )
    mesh = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("cells", "hexahedron", cells),)
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.binding.coordinate_scope == "mapped"
    assert certificate.status == "certified"
    # Bilinear boundary faces are decided as curved patches by the
    # boundary-degree theorem, never triangulated.
    assert "mapped_boundary_degree" in certificate.evaluated_checks
    assert certificate.boundary_degree is not None
    assert certificate.boundary_degree.status == "embedded"


def test_thin_gap_is_unresolved_with_the_exact_requested_subdivision_budget() -> None:

    element = coordinate_lagrange_element("triangle", 2)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    graph = np.column_stack((reference, reference[:, 0] ** 2))
    points = np.concatenate(
        (graph, graph + np.asarray((0.0, 0.0, 0.001), dtype=np.float64))
    )
    corners = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 0.0)), dtype=np.float64
    )
    mesh = phx.discretization.CellMesh.from_triangles(
        np.concatenate(
            (corners, corners + np.asarray((0.0, 0.0, 0.001), dtype=np.float64))
        ),
        np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"triangles": element},
        {
            "triangles": np.arange(points.shape[0], dtype=np.int32).reshape(
                (2, reference.shape[0])
            )
        },
        points,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    limits = phx.geometry.MeshCertificateLimits(maximum_subdivision_depth=0)
    certificate = phx.geometry.certify_global_embedding(
        mesh, geometry, validity, limits=limits
    )

    assert validity.all_certified
    assert certificate.status == "unresolved"
    assert "mapped_intersection_subdivision_depth" in finding_checks(
        certificate, "unresolved"
    )
    assert "mapped_cell_overlap" not in finding_checks(certificate, "violated")
    assert certificate.binding.limits_id == limits.limits_id


@pytest.mark.parametrize(
    "extent", (2, 3), ids=("nonconvex-seven-hexes", "cavity-twenty-six-hexes")
)
@pytest.mark.parametrize(
    "restricted", (False, True), ids=("source-root-maps", "retained-root-restrictions")
)
def test_common_polynomial_extension_certifies_curved_nonconvex_and_cavity_atlases(
    extent: int, restricted: bool
) -> None:

    element = coordinate_lagrange_element("hexahedron", 2)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    points = np.asarray(tuple(np.ndindex((extent + 1,) * 3)), dtype=np.float64)
    index = np.arange(points.shape[0], dtype=np.int32).reshape((extent + 1,) * 3)
    rows = []
    controls = []
    for i, j, k in np.ndindex((extent,) * 3):
        if (i, j, k) == (1, 1, 1):
            continue
        rows.append(
            (
                index[i, j, k],
                index[i + 1, j, k],
                index[i + 1, j + 1, k],
                index[i, j + 1, k],
                index[i, j, k + 1],
                index[i + 1, j, k + 1],
                index[i + 1, j + 1, k + 1],
                index[i, j + 1, k + 1],
            )
        )
        local = reference + np.asarray((i, j, k), dtype=np.float64)
        local[:, 2] += local[:, 0] ** 2 * local[:, 1] / 64.0
        controls.append(local)
    points[:, 2] += points[:, 0] ** 2 * points[:, 1] / 64.0
    cells = np.asarray(rows, dtype=np.int32)
    mesh = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("hex", "hexahedron", cells),)
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"hex": element},
        {
            "hex": np.arange(cells.shape[0] * reference.shape[0], dtype=np.int32).reshape(
                (cells.shape[0], reference.shape[0])
            )
        },
        np.concatenate(controls),
    )
    if restricted:
        record = CellGeometryRestrictionSource(
            cell_geometry_id(geometry),
            mesh.topology_id,
            {"hex": mesh.blocks[0].global_ids},
            {"hex": np.asarray(mesh.vertex_global_ids)[cells]},
        )
        geometry = phx.discretization.CellGeometrySpec(
            {
                "hex": RestrictedCellGeometryElement(
                    element,
                    "hexahedron",
                    np.eye(3, dtype=np.float64),
                    np.zeros((3,), dtype=np.float64),
                )
            },
            {"hex": geometry.geometry_dofs[0]},
            geometry.coordinates,
            restriction_source=record,
        )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert certificate.findings == ()


def test_positive_root_jacobians_do_not_admit_a_cracked_restricted_atlas() -> None:

    points = np.asarray(tuple(np.ndindex((3, 2, 2))), dtype=np.float64)
    lookup = {tuple(point): row for row, point in enumerate(points)}
    corners = np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64)
    cells = np.asarray(
        [
            [
                lookup[tuple(corner + np.asarray((origin, 0, 0), dtype=np.float64))]
                for corner in corners
            ]
            for origin in range(2)
        ],
        dtype=np.int64,
    )
    mesh = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("hex", "hexahedron", cells),)
    )
    element = coordinate_lagrange_element("hexahedron", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    controls = np.stack(
        (nodes.copy(), nodes + np.asarray((1.0, 0.0, 0.0), dtype=np.float64))
    )
    controls[0, :, 2] += 0.125 * nodes[:, 1] * (1.0 - nodes[:, 1])
    routes = np.arange(controls.size // 3, dtype=np.int64).reshape((2, nodes.shape[0]))
    source = phx.discretization.CellGeometrySpec(
        {"hex": element}, {"hex": routes}, controls.reshape((-1, 3))
    )
    record = CellGeometryRestrictionSource(
        cell_geometry_id(source),
        mesh.topology_id,
        {"hex": mesh.blocks[0].global_ids},
        {"hex": np.asarray(mesh.vertex_global_ids)[cells]},
    )
    geometry = phx.discretization.CellGeometrySpec(
        {
            "hex": RestrictedCellGeometryElement(
                element,
                "hexahedron",
                np.eye(3, dtype=np.float64),
                np.zeros((3,), dtype=np.float64),
            )
        },
        {"hex": routes},
        source.coordinates,
        restriction_source=record,
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert validity.all_certified
    assert certificate.status == "violated"


@pytest.mark.parametrize(
    "triangles",
    (((0, 2, 1), (0, 3, 2)), ((0, 1, 3), (1, 2, 3))),
    ids=("original-diagonal", "changed-diagonal"),
)
def test_curved_adjacent_sphere_triangles_have_checked_physical_embedding(
    triangles: tuple[tuple[int, ...], ...],
) -> None:

    surface = SpherePatch(
        np.zeros((3,), dtype=np.float64),
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        1.0,
    )
    charts = np.asarray(
        ((0.125, 0.125), (0.25, 0.125), (0.25, 0.25), (0.125, 0.25)), dtype=np.float64
    )
    element = coordinate_lagrange_element("triangle", 2)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    coordinates = []
    for row in triangles:
        corners = charts[np.asarray(row, dtype=np.int32)]
        reference = corners[0] + nodes @ (corners[1:] - corners[0])
        coordinates.append(
            np.asarray(
                surface.evaluate(jnp.asarray(reference, dtype=jnp.float64)),
                dtype=np.float64,
            )
        )
    mesh = phx.discretization.CellMesh(
        np.asarray(
            surface.evaluate(jnp.asarray(charts, dtype=jnp.float64)), dtype=np.float64
        ),
        (
            phx.discretization.CellBlock(
                "tri", "triangle", np.asarray(triangles, dtype=np.int32)
            ),
        ),
    )
    geometry = phx.discretization.CellGeometrySpec(
        {"tri": element},
        {"tri": np.arange(12, dtype=np.int32).reshape((2, 6))},
        np.concatenate(coordinates),
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert certificate.findings == ()


def test_affine_restricted_root_overlap_is_a_source_premise_not_child_overlap() -> None:
    mesh, geometry = affine_tetrahedral_root_restrictions(TETRAHEDRON + 0.25)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    # These small children are disjoint, but their declared roots cross.
    assert validity.all_certified
    assert certificate.status == "unresolved"
    assert all(finding.status != "violated" for finding in certificate.findings)
    affected = {
        identifier
        for finding in certificate.findings
        if finding.entity_kind == "source_cell"
        for identifier in finding.entity_ids
    }
    assert affected == {101, 205}


def test_affine_restricted_roots_preserve_candidate_pair_exhaustion() -> None:
    mesh, geometry = affine_tetrahedral_root_restrictions(TETRAHEDRON + 5.0)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(
        mesh,
        geometry,
        validity,
        limits=phx.geometry.MeshCertificateLimits(maximum_candidate_pairs=1),
    )
    assert certificate.status == "unresolved"
    assert certificate.candidate_pair_count <= 1
    assert any(finding.entity_kind == "source_cell" for finding in certificate.findings)


def test_rational_restriction_chart_keys_bind_the_complete_source_action(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._coordinate_enclosure import RationalPolynomial
    from phydrax.geometry._embedding_cells import Cell, vertices
    from phydrax.geometry._restricted_embedding import (
        _restriction_chart_key,
        _RestrictionChartCache,
    )

    def chart(denominator: Polynomial) -> Cell:
        return Cell(
            (RationalPolynomial({(1, 0, 0): Fraction(1)}, denominator),),
            "box", "pyramid", 3, tuple(range(5)), vertices("pyramid"),
        )

    with reference_budget.live_storage() as storage:
        cache = _RestrictionChartCache({}, {}, {}, {}, storage)
        first = chart({(0, 0, 0): Fraction(1)})
        equal = chart({(0, 0, 0): Fraction(1)})
        changed = chart({(0, 0, 0): Fraction(1), (0, 0, 1): Fraction(1)})
        assert first.coordinates is not equal.coordinates
        assert _restriction_chart_key(first, cache) == _restriction_chart_key(equal, cache)
        assert _restriction_chart_key(first, cache) != _restriction_chart_key(changed, cache)
        assert storage.bound > 0


@pytest.mark.parametrize("changed", (False, True))
def test_facet_restrictions_reuse_only_the_same_live_source_and_local_facet(
    changed: bool,
    monkeypatch: pytest.MonkeyPatch,
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.geometry import _mapped_embedding as owner
    from phydrax.geometry._embedding_cells import Cell, vertices

    points = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=np.float64,
    )
    rows = np.asarray(((0, 1, 2, 3), (0, 2, 1, 4)), dtype=np.int64)
    all_rows = np.concatenate((rows, np.asarray(((5, 8, 6, 7), (5, 9, 7, 6)))))
    point_bank = np.concatenate((points, points + 3))
    # Oriented boundary of [0,1,2,3], independently of mesh construction.
    boundary = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    for pair, shared in zip(all_rows.reshape(2, 2, 4), ((0, 1, 2), (5, 6, 7)), strict=True):
        orientations = []
        for row in pair:
            (loop,) = (
                tuple(int(row[index]) for index in face)
                for face in boundary if {int(row[index]) for index in face} == set(shared)
            )
            inversions = sum(loop[i] > loop[j] for i in range(3) for j in range(i + 1, 3))
            orientations.append((-1) ** inversions)
        assert orientations[0] == -orientations[1]
    mesh = phx.discretization.CellMesh(
        point_bank, (phx.discretization.CellBlock("tet", "tetrahedron", all_rows),),
    )

    def affine_coordinates(row: np.ndarray, *, perturb: bool) -> tuple[Expression, ...]:
        origin = point_bank[row[0]]
        edges = point_bank[row[1:]] - origin
        result: list[Expression] = []
        for component in range(3):
            polynomial: Polynomial = {(0, 0, 0): Fraction(float(origin[component]))}
            for axis in range(3):
                if edges[axis, component]:
                    polynomial[tuple(int(i == axis) for i in range(3))] = Fraction(
                        float(edges[axis, component])
                    )
            if perturb and component == 1:
                assert polynomial[(0, 1, 0)] == 1
                polynomial[(0, 1, 0)] = Fraction(2)
            result.append(polynomial)
        return tuple(result)

    cells = [
        Cell(
            affine_coordinates(row, perturb=changed and index == 3),
            "simplex", "tetrahedron", 3, tuple(map(int, row)), vertices("tetrahedron"),
        )
        for index, row in enumerate(all_rows)
    ]
    calls = []
    original = owner._facet_trace

    def capture(
        cell: Cell, facet: np.ndarray, *, budget: CoordinateEnclosureBudget
    ) -> tuple[ExactScalar, ...] | tuple[Expression, ...]:
        calls.append((cell.coordinates, tuple(facet)))
        return original(cell, facet, budget=budget)

    monkeypatch.setattr(owner, "_facet_trace", capture)
    state = EmbeddingState([], [])
    owner.trace_continuity(state, mesh, cells, shared_charts=True, budget=reference_budget)
    # Both sides of each oriented interface have different affine actions.
    # The first pair restricts local face {0,1,2}, the second {0,2,3}; all
    # four (live source, ordered local facet) keys are therefore distinct.
    assert len(calls) == 4
    assert ("mapped_trace_mismatch" in {finding.check for finding in state.findings}) == changed


def test_shared_curved_edge_with_opposite_local_orientation_is_one_trace() -> None:
    certificate = bumped_square(antisymmetric, antisymmetric)

    assert "mapped_trace_continuity" in certificate.evaluated_checks
    assert "mapped_trace_mismatch" not in finding_checks(certificate, "violated")


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(straight, symmetric, id="interior-only-quadratic-gap"),
        pytest.param(straight, antisymmetric, id="cubic-gap-vanishing-at-three-points"),
        pytest.param(symmetric, antisymmetric, id="unequal-nonlinear-degrees"),
    ],
)
def test_shared_edge_interior_trace_gap_is_violated(first: Any, second: Any) -> None:
    certificate = bumped_square(first, second)

    assert certificate.status == "violated"
    # The shared diagonal is the single interior facet; the pair decision may
    # separately report the same cells, which is not a trace proof.
    facets = [
        value
        for value in certificate.findings
        if value.check == "mapped_trace_mismatch" and value.entity_kind == "facet"
    ]
    assert len(facets) == 1
    assert facets[0].status == "violated"


@pytest.mark.parametrize("change", ("carrier", "source-bank", "source-route"))
def test_prepared_coordinate_scope_rechecks_every_current_carrier_corner(
    change: str,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
    reference_budget: CoordinateEnclosureBudget,
) -> None:

    mesh = tapered_hexes(0.5)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    ledger = coordinate_budget(1_000_000, 64 * 1024**2)
    preparation = coordinate_scope(mesh, geometry, budget=ledger)
    scope = preparation.coordinate_scope
    mapped = preparation.mapped_cells
    source = preparation.source_values
    assert scope == "mapped"
    np.testing.assert_array_equal(mapped, np.asarray((0, 1), dtype=np.int64))
    changed = np.asarray(mesh.coordinates, dtype=np.float64).copy()
    changed[-1, 0] += 0.125
    if change == "carrier":
        foreign = phx.discretization.CellMesh(changed, mesh.blocks)
        with pytest.raises(ValueError, match="coordinate_scope.carrier"):
            coordinate_scope(foreign, geometry, budget=ledger)
    elif change == "source-bank":
        with pytest.raises(ValueError, match="coordinate_scope.carrier"):
            coordinate_scope(mesh, geometry.with_coordinates(changed), budget=ledger)
    else:
        routes = {
            name: np.asarray(route, dtype=np.int64).copy()
            for name, route in zip(
                geometry.block_names, geometry.geometry_dofs, strict=True
            )
        }
        routes[geometry.block_names[0]][0, :2] = routes[geometry.block_names[0]][0, 1::-1]
        changed_route = phx.discretization.CellGeometrySpec(
            dict(zip(geometry.block_names, geometry.elements, strict=True)),
            routes,
            geometry.coordinates,
        )
        with pytest.raises(ValueError, match="coordinate_scope.carrier"):
            coordinate_scope(mesh, changed_route, budget=ledger)
    repeated_preparation = coordinate_scope(mesh, geometry, budget=ledger)
    repeated_scope = repeated_preparation.coordinate_scope
    repeated_mapped = repeated_preparation.mapped_cells
    repeated_source = repeated_preparation.source_values
    assert repeated_scope == scope and repeated_source == source
    np.testing.assert_array_equal(repeated_mapped, mapped)


@pytest.mark.parametrize(
    "top",
    (0.5, 0.05),
    ids=("corner-controls-at-root", "p-matrix-after-corner-refusal"),
)
def test_planar_trilinear_hexes_decide_through_exact_corner_proofs(top: float) -> None:
    # Strong taper defeats the fixed symmetric-part criterion, but every
    # projected principal minor remains positive on the full reference cube.
    mesh = tapered_hexes(top)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.binding.coordinate_scope == "mapped"
    assert certificate.status == "certified"
    assert {"mapped_trace_continuity", "mapped_straight_boundary_degree"} <= set(
        certificate.evaluated_checks
    )
    assert certificate.subdivision_piece_count == 2
    if top == 0.05:
        assert "mapped_box_p_matrix_univalence" in certificate.evaluated_checks
    else:
        assert "mapped_box_p_matrix_univalence" not in certificate.evaluated_checks


def test_trilinear_corner_proofs_refuse_one_unit_below_their_charged_work() -> None:
    mesh = tapered_hexes(0.5)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    required = certificate.source_expression_work_units
    capped = phx.geometry.certify_global_embedding(
        mesh,
        geometry,
        validity,
        limits=phx.geometry.MeshCertificateLimits(maximum_work_units=required - 1),
    )

    assert certificate.status == "certified" and required > 0
    assert capped.status != "certified"
    assert "mapped_source_expression_resource_budget" in finding_checks(
        capped, "unresolved"
    )


def test_trilinear_face_trace_identity_holds_through_a_rotated_local_chart() -> None:
    mesh = rotated_neighbor_hexes()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert "mapped_trace_mismatch" not in finding_checks(certificate, "violated")
    assert {"mapped_trace_continuity", "mapped_straight_boundary_degree"} <= set(
        certificate.evaluated_checks
    )


@pytest.mark.parametrize(
    ("bulge", "status"),
    ((0.0, "certified"), (0.0625, "violated")),
    ids=("continuous-mixed-degree-face", "bulged-quadratic-face"),
)
def test_mixed_degree_hex_faces_use_the_exact_restriction_owner(
    bulge: float, status: str
) -> None:
    mesh, geometry = mixed_degree_hexes(bulge)
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert "mapped_trace_continuity" in certificate.evaluated_checks
    assert certificate.status == status
    assert ("mapped_trace_mismatch" in finding_checks(certificate, "violated")) is bool(
        bulge
    )


def test_restricted_children_of_planar_trilinear_roots_reuse_the_straight_root_theorem() -> (
    None
):
    mesh, geometry = halved_tapered_restriction()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)

    assert validity.all_certified
    assert certificate.status == "certified"
    assert (
        "restriction_multi_affine_root:mapped_straight_boundary_degree"
        in certificate.evaluated_checks
    )


def test_mapped_embedding_releases_local_root_workspace_but_keeps_complete_maps(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    mesh, geometry = curved_shared_hex_interface()
    ledger = coordinate_budget(2_000_000, 81_920_000)
    state = EmbeddingState([], [])
    with ledger.activate():
        ledger.reserve(1, 128)
        before = ledger.temporary_bytes_upper
        certify_mapped_embedding(
            state,
            mesh,
            geometry,
            cell_global_ids(mesh),
            MeshCertificateLimits(),
            budget=ledger,
        )
        assert ledger.temporary_bytes_upper == before
        prepared = ledger.prepared_cell_cache[coordinate_scope_key(mesh, geometry)]
        assert len(prepared.cell_ids) == 2 and len(prepared.coordinates) == 2
        assert ledger.retained_basis_bytes > 0
        assert ledger.work_units > 1
    assert not state.findings


def _restricted_affine_p2_prisms() -> tuple[
    phx.discretization.CellMesh,
    phx.discretization.CellGeometrySpec,
    phx.discretization.CellMesh,
    phx.discretization.CellGeometrySpec,
]:
    from phydrax.meshing._curving import straight_geometry

    base = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64)
    points = np.concatenate(
        [np.column_stack((base, np.full(4, z, dtype=np.float64))) for z in (0.0, 1.0)]
    )
    root_cells = np.asarray(((0, 1, 2, 4, 5, 6), (0, 2, 3, 4, 6, 7)), dtype=np.int32)
    roots = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("roots", "prism", root_cells),)
    )
    source = straight_geometry(roots, 2)
    element = source.elements[0]
    assert isinstance(element, phx.discretization.fem.FiniteElementSpec)
    child_points = np.concatenate(
        [
            np.column_stack((base, np.full(4, z, dtype=np.float64)))
            for z in (0.0, 0.5, 1.0)
        ]
    )
    blocks = tuple(
        phx.discretization.CellBlock(
            name,
            "prism",
            root_cells + 4 * half,
            global_ids=np.asarray((2 * half + 10, 2 * half + 11), dtype=np.int64),
        )
        for half, name in enumerate(("lower", "upper"))
    )
    mesh = phx.discretization.CellMesh(child_points, blocks)
    parent_ids = {
        block.name: np.asarray(roots.blocks[0].global_ids, dtype=np.int64)
        for block in blocks
    }
    parent_vertices = {block.name: root_cells.astype(np.int64) for block in blocks}
    record = CellGeometryRestrictionSource(
        cell_geometry_id(source), roots.topology_id, parent_ids, parent_vertices
    )
    geometry = phx.discretization.CellGeometrySpec(
        {
            block.name: RestrictedCellGeometryElement(
                element,
                "prism",
                np.diag(np.asarray((1.0, 1.0, 0.5), dtype=np.float64)),
                np.asarray((0.0, 0.0, 0.5 * half), dtype=np.float64),
            )
            for half, block in enumerate(blocks)
        },
        {block.name: source.geometry_dofs[0] for block in blocks},
        source.coordinates,
        restriction_source=record,
    )
    return roots, source, mesh, geometry


def test_restricted_affine_p2_prisms_bind_contact_affinity_on_restore() -> None:
    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
    from tests._support.recipes import recipe_field_position

    _, _, mesh, geometry = _restricted_affine_p2_prisms()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    assert validity.all_certified and certificate.status == "certified"
    assert "mapped_affine_polytope_contact" in certificate.evaluated_checks
    restored = GlobalEmbeddingCertificate.from_stored_fields(certificate)
    assert restored.certificate_id == certificate.certificate_id
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(certificate)
    arrays = model_recipe_array_values(certificate, recipe, prefix="affine-contact")
    recipe["items"][recipe_field_position(recipe, "validity_certificate_id")] = (
        model_structure_recipe(phx.canonical_fingerprint("foreign-validity"))
    )
    tampered = model_from_array_recipe(recipe, arrays, prefix="affine-contact")
    assert isinstance(tampered, GlobalEmbeddingCertificate)
    with pytest.raises(ValueError, match="regenerated proof identity"):
        GlobalEmbeddingCertificate.from_stored_fields(tampered)


def test_restricted_prism_root_import_matches_full_proof_and_archives(
    tmp_path: Path,
) -> None:
    from phydrax.discretization._cell_geometry_predecessor import (
        prepare_cell_geometry_predecessor,
    )
    from phydrax.geometry._embedding_state import MappedRestrictionRootEmbeddingEvidence
    from phydrax.lifecycle import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )
    from phydrax.meshing._certification import require_prepared_embedding
    from tests._support.coordinate_references import coordinate_reference_budget

    roots, source, mesh, geometry = _restricted_affine_p2_prisms()
    source_validity = phx.discretization.certify_cell_geometry_validity(
        source, mesh=roots
    )
    source_embedding = phx.geometry.certify_global_embedding(
        roots, source, source_validity
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    complete = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    with coordinate_reference_budget() as budget:
        predecessor = prepare_cell_geometry_predecessor(
            roots,
            source,
            source_validity,
            source_embedding,
            mesh,
            geometry,
            budget=budget,
        )
        imported = phx.geometry.certify_global_embedding(
            mesh, geometry, validity, predecessor=predecessor, budget=budget
        )
    assert complete.status == imported.status == "certified"
    assert complete.findings == imported.findings == ()
    evidence = imported.restriction_root_embedding
    assert evidence is not None
    assert "mapped_restriction_root_embedding_reuse" in imported.evaluated_checks
    assert "mapped_restriction_reference_domain" in imported.evaluated_checks
    assert evidence.imported_certificate_id == source_embedding.certificate_id
    assert evidence.target_binding_id == imported.binding.binding_id
    assert evidence.validity_certificate_id == validity.certificate_id
    require_prepared_embedding(
        mesh, geometry, validity, imported, MeshCertificateLimits()
    )
    receipt = write_meshing_source_closure(
        tmp_path / "restricted-root-proof", (imported, evidence)
    )
    restored_certificate, restored_evidence = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored_certificate, GlobalEmbeddingCertificate)
    assert isinstance(restored_evidence, MappedRestrictionRootEmbeddingEvidence)
    assert restored_certificate.certificate_id == imported.certificate_id
    assert restored_evidence.evidence_id == evidence.evidence_id
    assert restored_certificate.restriction_root_embedding is not None
    assert (
        restored_certificate.restriction_root_embedding.imported_certificate_id
        == source_embedding.certificate_id
    )


@pytest.mark.parametrize("change", ("coefficient", "corner-id", "foreign-certificate"))
def test_restricted_prism_root_import_refuses_changed_predecessor(change: str) -> None:
    from dataclasses import replace

    from phydrax.discretization._cell_geometry_predecessor import (
        prepare_cell_geometry_predecessor,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    roots, source, mesh, geometry = _restricted_affine_p2_prisms()
    source_validity = phx.discretization.certify_cell_geometry_validity(
        source, mesh=roots
    )
    source_embedding = phx.geometry.certify_global_embedding(
        roots, source, source_validity
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    complete = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    with coordinate_reference_budget() as budget:
        predecessor = prepare_cell_geometry_predecessor(
            roots,
            source,
            source_validity,
            source_embedding,
            mesh,
            geometry,
            budget=budget,
        )
        if change == "coefficient":
            coordinates = np.asarray(source.coordinates, dtype=np.float64).copy()
            coordinates[0, 0] += 1.0 / 64.0
            changed = phx.discretization.CellGeometrySpec(
                dict(zip(source.block_names, source.elements, strict=True)),
                dict(zip(source.block_names, source.geometry_dofs, strict=True)),
                coordinates,
            )
            predecessor = replace(predecessor, source_geometry=changed)
        elif change == "corner-id":
            vertex_ids = np.asarray(roots.vertex_global_ids, dtype=np.int64).copy()
            vertex_ids[0] = 100
            changed_mesh = phx.discretization.CellMesh(
                roots.coordinates, roots.blocks, vertex_global_ids=vertex_ids
            )
            predecessor = replace(predecessor, source_mesh=changed_mesh)
        else:
            foreign = phx.geometry.certify_global_embedding(
                roots,
                source,
                source_validity,
                limits=MeshCertificateLimits(maximum_candidate_pairs=127),
            )
            predecessor = replace(predecessor, source_embedding=foreign)
        result = phx.geometry.certify_global_embedding(
            mesh, geometry, validity, predecessor=predecessor, budget=budget
        )
    assert result.status == complete.status == "certified"
    assert result.findings == complete.findings == ()
    assert result.restriction_root_embedding is None
    assert "mapped_restriction_root_embedding_reuse" not in result.evaluated_checks
    assert result.evaluated_checks == complete.evaluated_checks


_ORIGINAL_ROOT_LIMITS = MeshCertificateLimits(
    maximum_work_units=150_000_000, maximum_scratch_bytes=256 * 1024**2
)


def _restricted_root_request(
    source_limits: MeshCertificateLimits | None,
    current: MeshCertificateLimits,
    *,
    embedding_limits: MeshCertificateLimits = _ORIGINAL_ROOT_LIMITS,
) -> Any:
    from phydrax.discretization._cell_geometry_predecessor import (
        prepare_cell_geometry_predecessor,
    )
    from phydrax.geometry._restricted_embedding import (
        _authenticated_restriction_predecessor,
        _RestrictedRequest,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    roots, source, mesh, geometry = _restricted_affine_p2_prisms()
    source_validity = phx.discretization.certify_cell_geometry_validity(
        source, mesh=roots
    )
    source_embedding = phx.geometry.certify_global_embedding(
        roots, source, source_validity, limits=embedding_limits
    )
    with coordinate_reference_budget() as budget:
        predecessor = prepare_cell_geometry_predecessor(
            roots,
            source,
            source_validity,
            source_embedding,
            mesh,
            geometry,
            source_limits=source_limits,
            budget=budget,
        )
        request = _RestrictedRequest(
            EmbeddingState([], []),
            mesh,
            geometry,
            np.arange(4, dtype=np.int64),
            current,
            budget,
            predecessor,
        )
        return _authenticated_restriction_predecessor(request), source_embedding


def test_restricted_root_import_carries_original_resource_limits() -> None:
    from phydrax.discretization._cell_geometry_predecessor import (
        prepare_cell_geometry_predecessor,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    current = MeshCertificateLimits()
    assert current.limits_id != _ORIGINAL_ROOT_LIMITS.limits_id
    authenticated, source_embedding = _restricted_root_request(
        _ORIGINAL_ROOT_LIMITS, current
    )
    assert not isinstance(authenticated, str)
    _, certificate = authenticated
    assert certificate.certificate_id == source_embedding.certificate_id
    assert certificate.binding.limits_id == _ORIGINAL_ROOT_LIMITS.limits_id

    roots, source, mesh, geometry = _restricted_affine_p2_prisms()
    source_validity = phx.discretization.certify_cell_geometry_validity(
        source, mesh=roots
    )
    original = phx.geometry.certify_global_embedding(
        roots, source, source_validity, limits=_ORIGINAL_ROOT_LIMITS
    )
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    with coordinate_reference_budget() as budget:
        predecessor = prepare_cell_geometry_predecessor(
            roots,
            source,
            source_validity,
            original,
            mesh,
            geometry,
            source_limits=_ORIGINAL_ROOT_LIMITS,
            budget=budget,
        )
        imported = phx.geometry.certify_global_embedding(
            mesh,
            geometry,
            validity,
            limits=current,
            predecessor=predecessor,
            budget=budget,
        )
    assert imported.status == "certified"
    assert imported.binding.limits_id == current.limits_id
    evidence = imported.restriction_root_embedding
    assert evidence is not None
    assert evidence.imported_certificate_id == original.certificate_id
    assert "mapped_restriction_root_embedding_reuse" in imported.evaluated_checks


@pytest.mark.parametrize(
    ("source_limits", "current", "refusal"),
    (
        (None, MeshCertificateLimits(), "policy"),
        (
            _ORIGINAL_ROOT_LIMITS,
            MeshCertificateLimits(maximum_candidate_pairs=4_999_999),
            "policy",
        ),
        (
            _ORIGINAL_ROOT_LIMITS,
            MeshCertificateLimits(maximum_subdivision_depth=11),
            "policy",
        ),
    ),
)
def test_restricted_root_import_refuses_scientific_limit_change(
    source_limits: MeshCertificateLimits | None,
    current: MeshCertificateLimits,
    refusal: str,
) -> None:
    authenticated, _ = _restricted_root_request(source_limits, current)
    assert authenticated == refusal


def test_restricted_root_import_refuses_partial_proof_and_foreign_limits() -> None:
    from phydrax.discretization._cell_geometry_predecessor import (
        prepare_cell_geometry_predecessor,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    starved = MeshCertificateLimits(maximum_candidate_pairs=1)
    authenticated, partial = _restricted_root_request(
        starved, MeshCertificateLimits(), embedding_limits=starved
    )
    assert partial.status != "certified"
    assert authenticated == "partial_proof"

    roots, source, mesh, geometry = _restricted_affine_p2_prisms()
    source_validity = phx.discretization.certify_cell_geometry_validity(
        source, mesh=roots
    )
    embedding = phx.geometry.certify_global_embedding(roots, source, source_validity)
    with (
        coordinate_reference_budget() as budget,
        pytest.raises(ValueError, match="source_limits must be the limits bound"),
    ):
        prepare_cell_geometry_predecessor(
            roots,
            source,
            source_validity,
            embedding,
            mesh,
            geometry,
            source_limits=_ORIGINAL_ROOT_LIMITS,
            budget=budget,
        )


def _restricted_validity_import() -> tuple[Any, ...]:
    from phydrax.discretization._cell_geometry_predecessor import (
        prepare_cell_geometry_predecessor,
    )
    from phydrax.discretization._cell_geometry_validity import (
        _prepare_validity_import,
        CellValidityPolicy,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    roots, source, mesh, geometry = _restricted_affine_p2_prisms()
    source_validity = phx.discretization.certify_cell_geometry_validity(
        source, mesh=roots
    )
    with coordinate_reference_budget() as budget:
        predecessor = prepare_cell_geometry_predecessor(
            roots, source, source_validity, None, mesh, geometry, budget=budget
        )
        policy = CellValidityPolicy()
        imported = _prepare_validity_import(
            predecessor, mesh, geometry, policy, budget=budget
        )
    return roots, source, mesh, geometry, source_validity, predecessor, imported, policy


def test_restricted_children_import_exact_parent_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.discretization._cell_geometry_validity as validity_owner
    from tests._support.coordinate_references import coordinate_reference_budget

    _, _, mesh, geometry, source_validity, predecessor, _, _ = (
        _restricted_validity_import()
    )
    assert all(row < 0 for row in predecessor.source_rows)
    fresh = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)

    def refuse_fresh(*args: object, **kwargs: object) -> None:
        raise AssertionError("restricted children must not be certified freshly")

    monkeypatch.setattr(validity_owner, "_certify_polynomial_block", refuse_fresh)
    with coordinate_reference_budget() as budget:
        imported = phx.discretization.certify_cell_geometry_validity(
            geometry, mesh=mesh, predecessor=predecessor, budget=budget
        )
    assert imported.all_certified and fresh.all_certified
    assert imported.geometry_id == fresh.geometry_id
    parent_lower = np.asarray(source_validity.determinant_lower)
    parent_upper = np.asarray(source_validity.determinant_upper)
    lower = np.asarray(imported.determinant_lower)
    upper = np.asarray(imported.determinant_upper)
    # Each half-height child has det M = 1/2 over its parent's whole cell.
    assert np.all(lower <= np.tile(parent_lower, 2) / 2)
    assert np.all(upper >= np.tile(parent_upper, 2) / 2)
    assert np.all(lower > 0.0)
    assert np.all(lower <= np.asarray(fresh.determinant_upper))


def test_restricted_child_jacobian_is_child_own_differential() -> None:
    from phydrax.discretization._cell_geometry_validity import (
        _physical_jacobian,
        _restricted_child_jacobian,
        _restriction_parent_chart,
        _restriction_parents,
    )
    from phydrax.discretization._coordinate_enclosure import (
        expression_equal,
        prepared_coordinate_source_bank,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    _, _, _, geometry, _, _, imported, _ = _restricted_validity_import()
    record = geometry.restriction_source
    assert record is not None
    with coordinate_reference_budget() as budget:
        bank = prepared_coordinate_source_bank(geometry, budget=budget)
        for name, element, route in zip(
            geometry.block_names, geometry.elements, geometry.geometry_dofs, strict=True
        ):
            assert isinstance(element, RestrictedCellGeometryElement)
            origins = np.asarray(record.block_parent_cell_ids[name], dtype=np.int64)
            for row, origin in zip(np.asarray(route), origins, strict=True):
                local = tuple(bank[int(index)] for index in row)
                (parent,) = _restriction_parents(imported, budget)[int(origin)]
                chart = _restriction_parent_chart(element, parent.element, budget=budget)
                assert chart is not None
                chained = _restricted_child_jacobian(
                    parent, imported, chart, "prism", 3, budget
                )
                own = _physical_jacobian(element, local, "prism", 3, budget)
                assert chained is not None and own is not None
                assert all(
                    expression_equal(first, second, budget=budget)
                    for first_row, second_row in zip(chained, own, strict=True)
                    for first, second in zip(first_row, second_row, strict=True)
                )


@pytest.mark.parametrize(
    "change",
    (
        "wrong-parent",
        "coefficients",
        "not-contained",
        "orientation",
        "partial-proof",
        "scale",
    ),
)
def test_restricted_child_validity_typed_refusals(change: str) -> None:
    from dataclasses import replace

    from phydrax.discretization._cell_geometry_validity import (
        _restricted_child_validity,
        CellValidityStatus,
    )
    from phydrax.discretization._coordinate_enclosure import (
        prepared_coordinate_source_bank,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    _, source, mesh, geometry, _, _, imported, policy = _restricted_validity_import()
    element = geometry.elements[0]
    assert isinstance(element, RestrictedCellGeometryElement)
    parent = int(np.asarray(geometry.restriction_source.block_parent_cell_ids["lower"])[0])
    with coordinate_reference_budget() as budget:
        bank = prepared_coordinate_source_bank(geometry, budget=budget)
        route = np.asarray(geometry.geometry_dofs[0], dtype=np.int64)[0]
        local = tuple(bank[int(index)] for index in route)
        origin = parent
        expected = {
            "wrong-parent": "wrong_parent",
            "coefficients": "coefficients",
            "not-contained": "not_contained",
            "orientation": "orientation",
            "partial-proof": "partial_proof",
            "scale": "scale",
        }[change]
        if change == "wrong-parent":
            origin = 10_000
        elif change == "coefficients":
            local = (
                tuple(value + Fraction(1, 64) for value in local[0]),
                *local[1:],
            )
        elif change == "not-contained":
            element = RestrictedCellGeometryElement(
                element.source_element,
                "prism",
                np.diag(np.asarray((1.0, 1.0, 0.5))),
                np.asarray((0.0, 0.0, 0.75)),
            )
        elif change == "orientation":
            element = RestrictedCellGeometryElement(
                element.source_element,
                "prism",
                np.diag(np.asarray((1.0, 1.0, -0.5))),
                np.asarray((0.0, 0.0, 0.5)),
            )
        elif change == "partial-proof":
            status = imported.status.copy()
            status[:] = CellValidityStatus.UNRESOLVED
            imported = replace(imported, status=status, parents={})
        else:
            imported = replace(
                imported,
                lower=np.full_like(imported.lower, 1.0e-300),
                parents={},
            )
        decision = _restricted_child_validity(
            element, "prism", local, origin, imported, policy, budget
        )
    assert decision == expected


def test_restriction_root_evidence_refuses_foreign_certificate_replay() -> None:
    from phydrax.geometry._embedding_state import MappedRestrictionRootEmbeddingEvidence

    _, _, mesh, geometry = _restricted_affine_p2_prisms()
    validity = phx.discretization.certify_cell_geometry_validity(geometry, mesh=mesh)
    certificate = phx.geometry.certify_global_embedding(mesh, geometry, validity)
    evidence = MappedRestrictionRootEmbeddingEvidence(
        phx.canonical_fingerprint("imported-proof"),
        phx.canonical_fingerprint("foreign-target-binding"),
        validity.certificate_id,
    )
    with pytest.raises(
        ValueError, match="restriction_root_embedding must bind its target"
    ):
        GlobalEmbeddingCertificate(
            certificate.binding,
            validity.certificate_id,
            certificate.findings,
            (*certificate.evaluated_checks, "mapped_restriction_root_embedding_reuse"),
            counts=EmbeddingCounts(
                certificate.cell_count,
                certificate.boundary_facet_count,
                certificate.shell_count,
                certificate.candidate_pair_count,
                certificate.ray_test_count,
                certificate.subdivision_piece_count,
                certificate.periodic_image_count,
            ),
            restriction_root_embedding=evidence,
        )


def test_exact_contact_normalizes_only_requested_vertices(
    reference_budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.geometry._exact_polyhedral_geometry as exact_owner
    from phydrax.geometry._global_embedding import exact_pair_contacts

    points = np.asarray(
        [
            (Fraction(2**200 + row, 2**150 + row), Fraction(row))
            for row in range(100)
        ],
        dtype=object,
    )
    points[[1, 31, 60, 95]] = (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(1, 3)),
        (Fraction(0), Fraction(1, 3)),
        (Fraction(1, 3), Fraction(0)),
    )
    original = exact_owner.coordinate_integers
    normalized_rows = []

    def normalize(
        values: np.ndarray, /, *, budget: CoordinateEnclosureBudget
    ) -> tuple[np.ndarray, Fraction]:
        normalized_rows.append(values.shape[0])
        return original(values, budget=budget)

    monkeypatch.setattr(exact_owner, "coordinate_integers", normalize)
    hit, certain = exact_pair_contacts(
        points,
        np.asarray(((1, 31), (1, 31)), dtype=np.int64),
        np.asarray(((60, 95), (1, 60)), dtype=np.int64),
        budget=reference_budget,
    )
    assert normalized_rows == [4]
    np.testing.assert_array_equal(hit, (True, False))
    np.testing.assert_array_equal(certain, (True, True))


def test_identity_welded_audit_reuses_exact_canonical_facet_table(
    reference_budget: CoordinateEnclosureBudget,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    from phydrax.geometry._embedding_state import mesh_facets

    mesh = phx.discretization.CellMesh(
        np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64),
        (
            phx.discretization.CellBlock(
                "first", "triangle", np.asarray(((0, 1, 2),)), global_ids=np.asarray((0,))
            ),
            phx.discretization.CellBlock(
                "second", "triangle", np.asarray(((0, 2, 3),)), global_ids=np.asarray((1,))
            ),
        ),
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    audit = phx.meshing.audit_cell_mesh(mesh, geometry, budget=reference_budget)
    assert audit.passed
    key = coordinate_scope_key(mesh, geometry)
    assert key in reference_budget.facet_cache
    borrowed = reference_budget.facet_cache[key]
    before = reference_budget.work_units
    assert mesh_facets(mesh, geometry=geometry, budget=reference_budget) is borrowed
    assert reference_budget.work_units - before == 1
    fresh_budget = coordinate_budget(
        reference_budget.maximum_work_units, reference_budget.maximum_memory_bytes
    )
    fresh = mesh_facets(mesh, geometry=geometry, budget=fresh_budget)
    for field in ("rows", "signs", "cells", "group", "counts", "order", "group_offsets"):
        np.testing.assert_array_equal(getattr(borrowed, field), getattr(fresh, field))
    for observed, expected in zip(borrowed.groups, fresh.groups, strict=True):
        np.testing.assert_array_equal(observed, expected)
        assert not observed.flags.writeable


def test_exact_planar_degree_profiles_only_boundary_source_coordinates(
    reference_budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.geometry._exact_polyhedral_geometry as exact_owner
    from phydrax.geometry._embedding_state import mesh_facets
    from phydrax.geometry._global_embedding import _exact_planar_boundary_embedding

    points = np.asarray(
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(1), Fraction(1)),
            (Fraction(0), Fraction(1)),
            (Fraction(1, 3), Fraction(1, 3)),
        ),
        dtype=object,
    )
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(points, dtype=np.float64),
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int64),
    )
    facets = mesh_facets(mesh, budget=reference_budget)
    original = exact_owner._coordinate_integer_profile
    profiled_rows = []

    def profile(
        values: np.ndarray,
        maximum_bits: int | None,
        /,
        *,
        budget: CoordinateEnclosureBudget,
        collect_scalars: bool = False,
    ) -> tuple[tuple[int, int] | None, int, object]:
        profiled_rows.append(values.shape[0])
        return original(
            values, maximum_bits, budget=budget, collect_scalars=collect_scalars,
        )

    # Every public and prepared profile route runs this one canonical scan.
    monkeypatch.setattr(exact_owner, "_coordinate_integer_profile", profile)
    state = EmbeddingState([], [])
    _exact_planar_boundary_embedding(
        state, mesh, points, facets, MeshCertificateLimits(), 256,
        budget=reference_budget, predecessor=None,
    )
    assert state.clean
    assert state.boundary_facets == 4
    assert profiled_rows and max(profiled_rows) == 4
    assert "boundary_contact" in state.checks
    assert "exterior_degree" in state.checks


@pytest.mark.parametrize(("ambient", "arity"), ((2, 2), (3, 2), (3, 3)))
def test_unique_rational_vertex_boxes_preserve_outward_bounds_bitwise(
    ambient: int,
    arity: int,
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.geometry._embedding_state import boxes
    from phydrax.geometry._global_embedding import _exact_contact_boxes

    values = (
        Fraction(1) + Fraction(1, 2**54),
        Fraction(1) - Fraction(1, 2**54),
        Fraction(1, 2**1075),
        Fraction(-1, 2**1075),
        Fraction(0),
        Fraction(1, 3),
        Fraction(-1, 3),
        Fraction(-1) - Fraction(1, 2**54),
    )
    points = np.asarray(
        [tuple(values[(row + axis) % len(values)] for axis in range(ambient))
         for row in range(len(values))],
        dtype=object,
    )
    rows = np.asarray(
        [tuple((row + local) % len(values) for local in range(arity))
         for row in range(len(values))],
        dtype=np.int64,
    )
    expected = boxes(points, rows)
    actual = _exact_contact_boxes(points, rows, budget=reference_budget)
    for observed, original in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(observed.view(np.uint64), original.view(np.uint64))
    from phydrax._geometry_predicates import exact_bits
    from phydrax.geometry._exact_polyhedral_geometry import CoordinateIntegerPreparation

    with reference_budget.temporary_scope():
        preparation = CoordinateIntegerPreparation(points, 4096, budget=reference_budget)
        assert preparation.rational_bits(points, budget=reference_budget) == exact_bits(
            points, budget=reference_budget
        )
        before = reference_budget.work_units
        _exact_contact_boxes(points, rows, budget=reference_budget)
        scanned = reference_budget.work_units - before
        before = reference_budget.work_units
        prepared = _exact_contact_boxes(
            points, rows, budget=reference_budget, integer_preparation=preparation
        )
        # Every bank row is used, so the bound preparation's scan is reused.
        assert reference_budget.work_units - before == scanned - points.size
        for observed, original in zip(prepared, expected, strict=True):
            np.testing.assert_array_equal(
                observed.view(np.uint64), original.view(np.uint64)
            )
        changed = points.copy()
        changed[0, 0] = Fraction(1, 7)
        with pytest.raises(ValueError, match="current source values and ledger"):
            _exact_contact_boxes(
                changed, rows, budget=reference_budget, integer_preparation=preparation
            )


@pytest.mark.parametrize("ambient", (2, 3))
def test_shared_endpoint_separation_preserves_segment_contact_semantics(
    ambient: int,
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.geometry._global_embedding import (
        _shared_segment_separation,
        exact_pair_contacts,
    )

    coordinates = ((0, 0), (1, 0), (0, 1), (1, 1), (-2, -2), (2, 2), (2, 1))
    points = np.asarray(
        [tuple(Fraction(value) for value in row) + (Fraction(0),) * (ambient - 2)
         for row in coordinates],
        dtype=object,
    )
    first = np.asarray(((0, 1), (0, 3), (0, 3), (0, 3), (0, 3)), dtype=np.int64)
    second = np.asarray(((0, 2), (0, 4), (0, 5), (0, 6), (3, 0)), dtype=np.int64)
    separated = _shared_segment_separation(
        points, first, second, budget=reference_budget
    )
    np.testing.assert_array_equal(separated, (True, True, False, False, False))
    hit, certain = exact_pair_contacts(points, first, second, budget=reference_budget)
    assert np.all(certain)
    assert np.all(~hit[separated])
    np.testing.assert_array_equal(hit, (False, False, True, False, True))


def test_prepared_embedding_refusal_names_its_actual_missing_governed_check(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from copy import copy

    from phydrax.meshing._certification import require_prepared_embedding

    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    limits = MeshCertificateLimits()
    validity = phx.discretization.certify_cell_geometry_validity(
        geometry, mesh=mesh, budget=reference_budget,
    )
    certificate = phx.geometry.certify_global_embedding(
        mesh, geometry, validity, limits=limits, budget=reference_budget,
    )
    assert certificate.status == "certified"
    broken = copy(certificate)
    object.__setattr__(
        broken, "evaluated_checks",
        tuple(check for check in certificate.evaluated_checks if check != "exterior_degree"),
    )
    with pytest.raises(ValueError, match="sufficient governed evidence") as refusal:
        require_prepared_embedding(
            mesh, geometry, validity, broken, limits, budget=reference_budget,
        )
    assert "missing_checks=('exterior_degree',)" in str(refusal.value)
    assert "counts_and_limits=" in str(refusal.value)
    assert "validity_ids=" in str(refusal.value)


def test_prepared_integer_bank_binds_values_ledger_and_bit_admission(
    reference_budget: CoordinateEnclosureBudget,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    from phydrax.geometry._exact_polyhedral_geometry import (
        coordinate_integers,
        CoordinateIntegerPreparation,
    )

    points = np.asarray(
        ((Fraction(0), Fraction(1, 3)), (Fraction(2, 3), Fraction(1))), dtype=object
    )
    original = points.copy()
    with reference_budget.temporary_scope():
        preparation = CoordinateIntegerPreparation(points, 256, budget=reference_budget)
        assert preparation.profile is not None
        before = reference_budget.work_units
        integers, scale = coordinate_integers(
            points, budget=reference_budget, preparation=preparation
        )
        assert reference_budget.work_units - before == points.size
        np.testing.assert_array_equal(integers * scale, points)
        assert not integers.flags.writeable
        before = reference_budget.work_units
        reused, reused_scale = coordinate_integers(
            points.copy(), budget=reference_budget, preparation=preparation
        )
        assert reused is integers and reused_scale == scale
        assert reference_budget.work_units == before
        points[0, 0] = Fraction(1, 7)
        with pytest.raises(ValueError, match="current source values and ledger"):
            coordinate_integers(points, budget=reference_budget, preparation=preparation)
        foreign = coordinate_budget(1000, 1 << 20)
        with pytest.raises(ValueError, match="current source values and ledger"):
            coordinate_integers(original, budget=foreign, preparation=preparation)
        refused = CoordinateIntegerPreparation(original, 1, budget=reference_budget)
        assert refused.profile is None
        with pytest.raises(ValueError, match="completed bit profile"):
            coordinate_integers(original, budget=reference_budget, preparation=refused)


def test_integer_normalization_factors_actual_repeated_denominators_and_scalars(
    reference_budget: CoordinateEnclosureBudget, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _exact_polyhedral_geometry as exact_owner

    points = np.asarray(
        ((Fraction(1, 6), Fraction(1, 10), Fraction(1, 6)),
         (Fraction(1, 10), Fraction(1, 6), Fraction(1, 10))),
        dtype=object,
    )
    original_gcd = exact_owner.math.gcd
    visits = []

    def gcd(first: int, second: int) -> int:
        visits.append((first, second))
        return original_gcd(first, second)

    monkeypatch.setattr(exact_owner.math, "gcd", gcd)
    with reference_budget.temporary_scope():
        before = reference_budget.work_units
        profile, _, scalars = exact_owner._coordinate_integer_profile(
            points, 256, budget=reference_budget, collect_scalars=True,
        )
        assert visits == [(1, 6), (6, 10)]
        assert reference_budget.work_units - before == points.size + 2
        assert profile is not None and scalars is not None
        before = reference_budget.work_units
        integers, scale = exact_owner._scale_coordinate_integers(
            points, profile, scalars, budget=reference_budget,
        )
        assert reference_budget.work_units - before == 2
        np.testing.assert_array_equal(integers, ((5, 3, 5), (3, 5, 3)))
        assert scale == Fraction(1, 30)
        np.testing.assert_array_equal(integers * scale, points)


def test_prepared_integer_scalars_reject_changed_fraction_contents(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.geometry._exact_polyhedral_geometry import CoordinateIntegerPreparation

    value = Fraction(1, 3)
    points = np.asarray(((value, Fraction(0)),), dtype=object)
    with reference_budget.temporary_scope():
        preparation = CoordinateIntegerPreparation(points, 256, budget=reference_budget)
        object.__setattr__(value, "_numerator", 2)
        try:
            with pytest.raises(ValueError, match="current source values and ledger"):
                preparation.coordinates(points, budget=reference_budget)
        finally:
            object.__setattr__(value, "_numerator", 1)


def test_restricted_predecessor_reuses_issuer_paid_content_on_same_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _cell_geometry_predecessor as predecessor_owner
    from phydrax.geometry._restricted_embedding import (
        _authenticated_restriction_predecessor,
        _RestrictedRequest,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    roots, source, mesh, geometry = _restricted_affine_p2_prisms()
    validity = phx.discretization.certify_cell_geometry_validity(source, mesh=roots)
    embedding = phx.geometry.certify_global_embedding(
        roots, source, validity, limits=_ORIGINAL_ROOT_LIMITS
    )
    hashes = 0

    def observed_hash[**P](function: Callable[P, str]) -> Callable[P, str]:
        def counted(*args: P.args, **kwargs: P.kwargs) -> str:
            nonlocal hashes
            hashes += 1
            return function(*args, **kwargs)

        return counted

    monkeypatch.setattr(
        predecessor_owner, "_predecessor_content_id",
        observed_hash(predecessor_owner._predecessor_content_id),
    )
    with coordinate_reference_budget() as budget:
        predecessor = predecessor_owner.prepare_cell_geometry_predecessor(
            roots, source, validity, embedding, mesh, geometry,
            source_limits=_ORIGINAL_ROOT_LIMITS, budget=budget,
        )
        assert hashes == 1
        request = _RestrictedRequest(
            EmbeddingState([], []), mesh, geometry,
            np.arange(4, dtype=np.int64), MeshCertificateLimits(), budget, predecessor,
        )
        start = budget.work_units
        first = _authenticated_restriction_predecessor(request)
        first_work = budget.work_units - start
        start = budget.work_units
        second = _authenticated_restriction_predecessor(request)
        second_work = budget.work_units - start
        assert not isinstance(first, str)
        assert not isinstance(second, str)
        assert (
            first[1].certificate_id == second[1].certificate_id == embedding.certificate_id
        )
        assert hashes == 1
        print(
            f"restriction-sourceauth same-ledger-first={first_work} "
            f"same-ledger-repeat={second_work} cumulative={budget.work_units}"
        )
    with coordinate_reference_budget() as independent:
        request = _RestrictedRequest(
            EmbeddingState([], []), mesh, geometry,
            np.arange(4, dtype=np.int64), MeshCertificateLimits(),
            independent, predecessor,
        )
        result = _authenticated_restriction_predecessor(request)
        assert not isinstance(result, str)
        assert hashes == 2
        assert independent.work_units > second_work


def test_binary_row_filter_never_rounds_algebraic_source(
    reference_budget: CoordinateEnclosureBudget,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement
    from phydrax.geometry._global_embedding import source_bank_binary64_rows

    radical = RadicalElement(terms=((2, Fraction(1)),))
    bank = (
        (Fraction(0), Fraction(1)),
        (radical, Fraction(0)),
        (Fraction(1, 3), Fraction(0)),
    )

    def unexpected_rounding(self: RadicalElement) -> float:
        raise AssertionError("Rational-only predicate filtering must not round algebraic rows.")

    monkeypatch.setattr(RadicalElement, "__float__", unexpected_rounding)
    view, represented = source_bank_binary64_rows(bank, budget=reference_budget)
    np.testing.assert_array_equal(represented, (True, False, False))
    np.testing.assert_array_equal(view[0], (0.0, 1.0))
    for array in (view, represented):
        with pytest.raises(ValueError):
            array.flags.writeable = True


@pytest.mark.parametrize("array_index", (0, 1), ids=("coordinates", "mask"))
def test_binary_row_cache_rejects_changed_numeric_view_layout(
    reference_budget: CoordinateEnclosureBudget, array_index: int,
) -> None:
    from phydrax.geometry._global_embedding import source_bank_binary64_rows

    bank = ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)))
    arrays = source_bank_binary64_rows(bank, budget=reference_budget)
    target = arrays[array_index]
    cached = reference_budget.source_binary64_rows_cache[id(bank)]
    changed = target.reshape((1, target.size))
    # Preserve the exact byte owner, pointer and values; only the cached
    # view's shape/strides differ from its previously earned layout proof.
    assert np.shares_memory(changed, target)
    reference_budget.source_binary64_rows_cache[id(bank)] = (
        cached[0],
        changed if array_index == 0 else cached[1],
        changed if array_index == 1 else cached[2],
        cached[3], cached[4],
    )
    try:
        with pytest.raises(ValueError, match="immutable numeric layouts"):
            source_bank_binary64_rows(bank, budget=reference_budget)
    finally:
        reference_budget.source_binary64_rows_cache[id(bank)] = cached


def test_nonbinary_predicate_object_view_is_not_a_mutable_cached_authority(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.geometry._global_embedding import source_bank_predicate_view

    bank = ((Fraction(0), Fraction(1, 3)), (Fraction(1), Fraction(0)))
    first, binary = source_bank_predicate_view(bank, budget=reference_budget)
    assert not binary
    first.flags.writeable = True
    first[0, 0] = Fraction(99)
    second, binary = source_bank_predicate_view(bank, budget=reference_budget)
    assert not binary and second is not first
    assert second[0, 0] is bank[0][0]
    assert second[0, 1] is bank[0][1]
