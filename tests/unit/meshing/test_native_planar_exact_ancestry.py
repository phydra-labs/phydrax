#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Planar rational ancestry and bounded binary64 carriers."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import replace
from fractions import Fraction
from time import monotonic
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import MeshcoreError, NativeExecutionScope
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import CellGeometrySpec, CellMesh, ExactPlcCellGeometrySource
from phydrax.discretization._cell_complex import PolygonalConnectivity
from phydrax.discretization._cell_geometry_transfer import is_affine_cell_geometry
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import ExactScalar
from phydrax.geometry import ConstrainedDelaunayTriangulation, SegmentMesh
from phydrax.meshing import MeshingScope, SizeControlStrength, UniformSizeControl
from phydrax.meshing._contracts import MeshingLimits, SurfaceMeshingSpec
from phydrax.meshing._plc_planar_support import cell_maps
from phydrax.meshing._sizing import SizeStatistic
from phydrax.meshing.providers._native_planar_construction import LATTICE_FRACTION
from phydrax.meshing.providers._native_planar_domain import (
    constraint_coordinate_source,
    PreparedPlanarDomain,
    source_carrier_evidence,
    split_constraints,
    triangulate,
)
from phydrax.meshing.providers._native_sources import NativePlanarSource
from tests._support.native_provider_cases import plate, stage_report


@pytest.fixture
def coordinate_budget() -> Iterator[Callable[[int, int], CoordinateEnclosureBudget]]:
    with ExitStack() as resources:

        def create(work: int, memory: int) -> CoordinateEnclosureBudget:
            ticket = resources.enter_context(
                HostBatchTicket(work, queries_upper=work, bytes_upper=memory)
            )
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=work, maximum_memory_bytes=memory
            )
            resources.enter_context(budget.activate())
            return budget

        yield create


@pytest.fixture
def reference_budget(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> CoordinateEnclosureBudget:
    limits = MeshingLimits()
    return coordinate_budget(limits.maximum_work_units, limits.maximum_scratch_bytes)


def _source(
    bound: float, *, budget: CoordinateEnclosureBudget
) -> ExactPlcCellGeometrySource:
    return ExactPlcCellGeometrySource(
        np.asarray(((0.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        np.empty((0, 3), dtype=np.int64),
        np.asarray(((0, 1),), dtype=np.int64),
        np.asarray((0, 0, 0, 1), dtype=np.int64),
        np.asarray((-1, -1, -1, 0), dtype=np.int64),
        np.asarray(
            ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0 / 3.0, 0.0)), dtype=np.float64
        ),
        domain_source_id="planar",
        domain_source_revision="decimal",
        source_triangle_ids=np.empty((0,), dtype=np.int64),
        source_triangle_bounds=np.empty((0,), dtype=np.float64),
        source_segment_ids=np.asarray((7,), dtype=np.int64),
        source_segment_bounds=np.asarray((bound,), dtype=np.float64),
        vertex_parameter_ratios=(((0, 1), (0, 1)),) * 3 + (((1, 3), (0, 1)),),
        budget=budget,
    )


def _published_affine_carrier_case(
    budget: CoordinateEnclosureBudget,
) -> tuple[CellMesh, CellGeometrySpec, CellGeometrySpec]:
    from phydrax.discretization import CellBlock
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        CellGeometryRestrictionSource,
        coordinate_lagrange_element,
    )
    from phydrax.discretization._cell_geometry_validity import cell_geometry_id

    roots = CellMesh.from_triangles(
        np.asarray(((-0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    root_geometry = CellGeometrySpec.affine(roots)
    mesh = CellMesh(
        np.asarray(((-0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (7.0, 7.0))),
        (
            CellBlock("a", "triangle", np.asarray(((0, 3, 2),)), global_ids=np.asarray((0,))),
            CellBlock("b", "triangle", np.asarray(((3, 1, 2),)), global_ids=np.asarray((1,))),
        ),
    )
    elements = {
        name: BarycentricCellGeometryElement(coordinate_lagrange_element("triangle", 1), rows)
        for name, rows in (
            ("a", np.asarray(((1, 0, 0), (0.5, 0.5, 0), (0, 0, 1)))),
            ("b", np.asarray(((0.5, 0.5, 0), (0, 1, 0), (0, 0, 1)))),
        )
    }
    routes = {name: np.asarray(((0, 1, 2),)) for name in elements}
    cells = {name: np.asarray((0,)) for name in elements}
    vertices = {name: np.asarray(((0, 1, 2),)) for name in elements}
    origin = CellGeometryRestrictionSource(
        cell_geometry_id(root_geometry), roots.topology_id, cells, vertices,
    )
    publication = CellGeometrySpec(
        elements, routes, roots.coordinates, restriction_source=origin, budget=budget,
    )
    geometry = CellGeometrySpec(
        elements, routes, roots.coordinates,
        restriction_source=CellGeometryRestrictionSource(
            cell_geometry_id(root_geometry), roots.topology_id, cells, vertices,
            carrier_publication="published_affine_roots", carrier_geometry=publication,
        ),
        budget=budget,
    )
    return mesh, geometry, publication


@pytest.mark.parametrize("cache_geometry", ("publication", "coefficient"))
def test_published_affine_carrier_reuses_only_its_actual_earned_image_scope(
    reference_budget: CoordinateEnclosureBudget, monkeypatch: pytest.MonkeyPatch,
    cache_geometry: str,
) -> None:
    from phydrax.discretization import (
        _cell_geometry as owner,
        _coordinate_enclosure as algebra,
    )
    from phydrax.discretization._coordinate_identity import coordinate_scope_key

    mesh, geometry, publication = _published_affine_carrier_case(reference_budget)
    elements, routes, _ = publication.resolve(mesh, budget=reference_budget)
    bank = owner.prepared_published_affine_root_bank(publication, budget=reference_budget)
    images = algebra.coordinate_vertex_images(
        tuple((element, np.asarray(route, dtype=np.int64), np.asarray(block.vertices)) for block, element, route
              in zip(mesh.blocks, elements, routes, strict=True)),
        bank, mesh.coordinates.shape[0], mismatch="unequal corners", budget=reference_budget,
    )
    assert images is not None and images[-1] is None  # The unused vertex stays explicit.
    assert coordinate_scope_key(mesh, publication) != coordinate_scope_key(mesh, geometry)
    selected = publication if cache_geometry == "publication" else geometry
    reference_budget.coordinate_vertex_image_cache[coordinate_scope_key(mesh, selected)] = images
    original = algebra.rounded_point
    rounded = []

    def rounding(point: tuple[ExactScalar, ...]) -> np.ndarray:
        rounded.append(point)
        return original(point)

    monkeypatch.setattr(algebra, "rounded_point", rounding)
    owner.require_published_affine_carrier(geometry, mesh, budget=reference_budget)
    assert len(rounded) == (1 if cache_geometry == "publication" else 2)
    assert np.signbit(np.asarray(mesh.coordinates)[0, 0])


@pytest.mark.parametrize("changed", ("root", "descendant", "missing_image"))
def test_earned_published_images_keep_root_bytes_and_descendant_rounding_checks(
    reference_budget: CoordinateEnclosureBudget, changed: str,
) -> None:
    from phydrax.discretization import (
        _cell_geometry as owner,
        _coordinate_enclosure as algebra,
    )
    from phydrax.discretization._coordinate_identity import coordinate_scope_key

    mesh, geometry, publication = _published_affine_carrier_case(reference_budget)
    coordinates = np.array(mesh.coordinates, copy=True)
    if changed == "root":
        coordinates[0, 0] = 0.0  # Numerically equal, but not the retained root's bytes.
    elif changed == "descendant":
        coordinates[3, 0] = np.nextafter(coordinates[3, 0], np.inf)
    mesh = mesh.with_coordinates(coordinates, numeric_version="changed-carrier")
    elements, routes, _ = publication.resolve(mesh, budget=reference_budget)
    bank = owner.prepared_published_affine_root_bank(publication, budget=reference_budget)
    images = algebra.coordinate_vertex_images(
        tuple((element, np.asarray(route, dtype=np.int64), np.asarray(block.vertices)) for block, element, route
              in zip(mesh.blocks, elements, routes, strict=True)),
        bank, mesh.coordinates.shape[0], mismatch="unequal corners", budget=reference_budget,
    )
    assert images is not None
    if changed == "missing_image":
        images = tuple(None if row == 3 else image for row, image in enumerate(images))
    reference_budget.coordinate_vertex_image_cache[coordinate_scope_key(mesh, publication)] = images
    with pytest.raises(ValueError, match="routed corner|retained root bytes"):
        owner.require_published_affine_carrier(geometry, mesh, budget=reference_budget)
    assert coordinate_scope_key(mesh, geometry) not in reference_budget.published_affine_carrier_cache


def test_published_carrier_rows_reuse_only_authenticated_prepared_weights(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization import (
        _cell_geometry as owner,
        _coordinate_enclosure as algebra,
    )
    from phydrax.discretization._coordinate_identity import coordinate_scope_key

    mesh, geometry, publication = _published_affine_carrier_case(reference_budget)
    elements, routes, _ = publication.resolve(mesh, budget=reference_budget)
    bank = owner.prepared_published_affine_root_bank(publication, budget=reference_budget)
    # The target factory's pass prepares this request's corner weights.
    algebra.coordinate_vertex_images(
        tuple((element, np.asarray(route, dtype=np.int64), np.asarray(block.vertices)) for block, element, route
              in zip(mesh.blocks, elements, routes, strict=True)),
        bank, mesh.coordinates.shape[0], mismatch="unequal corners", budget=reference_budget,
    )
    identity = tuple(
        tuple(Fraction(int(row == column)) for column in range(3)) for row in range(3)
    )
    for element in elements:
        rows = owner._published_corner_weight_rows(element, {}, {}, {}, budget=reference_budget)
        images = algebra.coordinate_corner_images(element, identity, budget=reference_budget)
        assert rows is not None and images is not None
        assert rows == images
        assert [type(value) for row in rows for value in row] == [
            type(value) for row in images for value in row
        ]
    # A stale element is reauthenticated: its current rows are prepared again
    # and the carrier check refuses the vertex they no longer reproduce.
    object.__setattr__(
        elements[0], "barycentric_weights",
        jnp.asarray(((1.0, 0.0, 0.0), (0.25, 0.75, 0.0), (0.0, 0.0, 1.0))),
    )
    rows = owner._published_corner_weight_rows(elements[0], {}, {}, {}, budget=reference_budget)
    assert rows is not None
    assert rows[1] == (Fraction(1, 4), Fraction(3, 4), Fraction(0))
    with pytest.raises(ValueError, match="retained root bytes"):
        owner.require_published_affine_carrier(geometry, mesh, budget=reference_budget)
    assert coordinate_scope_key(mesh, geometry) not in reference_budget.published_affine_carrier_cache



def test_non_rne_planar_carrier_retains_exact_rational_coordinate(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    source = _source(1e-15, budget=reference_budget)
    carrier = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (np.nextafter(1.0 / 3.0, 1.0), 1.0 / 3.0)),
        dtype=np.float64,
    )
    prepared = source.prepare(carrier, budget=reference_budget)
    assert prepared.vertices[3] == (Fraction(1, 3), Fraction(1, 3))
    assert prepared.vertices[3][0] is prepared.vertices[3][1]
    assert prepared.vertices[0][0] is prepared.vertices[0][1]
    assert prepared.vertices[0][0] is prepared.vertices[2][0]
    assert source.vertex_parameter_ratios is not None
    assert source.vertex_parameter_ratios[0] is source.vertex_parameter_ratios[1]
    assert prepared.maximum_rounding_error > 0
    mesh = CellMesh.from_triangles(
        carrier, np.asarray(((0, 3, 2), (3, 1, 2)), dtype=np.int32)
    )
    geometry = CellGeometrySpec.plc(mesh, source, budget=reference_budget)
    assert geometry.source_coordinates(budget=reference_budget)[3] == prepared.vertices[3]
    assert is_affine_cell_geometry(mesh, geometry, budget=reference_budget)
    assert source.exact_vertex_parameters()[3] == (Fraction(1, 3), Fraction(0))
    with pytest.raises(
        ValueError, match=r"ExactPlcCellGeometrySource\.source_row_bounds"
    ):
        _source(0.0, budget=reference_budget).prepare(carrier, budget=reference_budget)
    with pytest.raises(
        ValueError, match=r"ExactPlcCellGeometrySource\.source_row_bounds"
    ):
        _source(1e-18, budget=reference_budget).prepare(carrier, budget=reference_budget)


def test_plc_identity_scope_retains_source_bank_without_copy(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._coordinate_enclosure import (
        prepared_coordinate_source_bank,
        prepared_coordinate_vertex_images,
    )
    from phydrax.geometry._global_embedding import coordinate_scope

    source = _source(1e-15, budget=reference_budget)
    # Vertex 3 is not the correctly rounded image of its certified source point;
    # its own-row PLC binding is still that exact point.
    carrier = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (np.nextafter(1.0 / 3.0, 1.0), 1.0 / 3.0)),
        dtype=np.float64,
    )
    mesh = CellMesh.from_triangles(
        carrier, np.asarray(((0, 3, 2), (3, 1, 2)), dtype=np.int32)
    )
    geometry = CellGeometrySpec.plc(mesh, source, budget=reference_budget)
    scope = coordinate_scope(mesh, geometry, budget=reference_budget)
    bank = prepared_coordinate_source_bank(geometry, budget=reference_budget)
    assert scope.coordinate_scope == "affine"
    assert scope.mapped_cells.size == 0
    assert scope.source_values is bank
    assert scope.source_values[3] == (Fraction(1, 3), Fraction(1, 3))
    assert prepared_coordinate_vertex_images(
        mesh, geometry, mismatch="changed source corner", budget=reference_budget
    ) is bank


def test_identity_corner_bank_reuse_requires_every_vertex_to_be_bound(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._coordinate_enclosure import (
        prepared_coordinate_source_bank,
        prepared_coordinate_vertex_images,
    )
    from phydrax.discretization._coordinate_identity import coordinate_scope_key

    source = _source(1e-15, budget=reference_budget)
    carrier = np.asarray(
        ((0, 0), (1, 1), (0, 1), (1.0 / 3.0, 1.0 / 3.0)),
        dtype=np.float64,
    )
    mesh = CellMesh.from_triangles(
        carrier, np.asarray(((0, 1, 2),), dtype=np.int32)
    )
    geometry = CellGeometrySpec.plc(mesh, source, budget=reference_budget)
    assert coordinate_scope_key(mesh, geometry) not in (
        reference_budget.plc_identity_corner_cache
    )
    bank = prepared_coordinate_source_bank(geometry, budget=reference_budget)
    images = prepared_coordinate_vertex_images(
        mesh, geometry, mismatch="changed source corner", budget=reference_budget
    )
    assert images is not None and images is not bank
    assert images[:3] == bank[:3]
    assert images[3] is None


def test_affine_coefficient_actions_earn_planar_audit_embedding(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        coordinate_lagrange_element,
    )
    from phydrax.meshing._audit_topology import prepare_planar_audit_embedding

    roots = np.asarray(((0, 0), (2, 0), (0, 2)), dtype=np.float64)
    mesh = CellMesh.from_triangles(
        np.asarray(((0, 0), (1, 0), (0, 2)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    name = mesh.blocks[0].name
    element = BarycentricCellGeometryElement(
        coordinate_lagrange_element("triangle", 1),
        np.asarray(((1, 0, 0), (0.5, 0.5, 0), (0, 0, 1)), dtype=np.float64),
    )
    geometry = CellGeometrySpec(
        {name: element},
        {name: np.asarray(((0, 1, 2),), dtype=np.int32)},
        roots,
        budget=reference_budget,
    )
    validity, embedding = prepare_planar_audit_embedding(
        mesh,
        geometry,
        policy=phx.meshing.CellMeshAuditPolicy(),
        limits=phx.geometry.MeshCertificateLimits(),
        junction_vertices=None,
        budget=reference_budget,
    )
    assert validity is not None and validity.all_certified
    assert embedding is not None and embedding.status == "certified"


def test_midpoint_release_uses_the_actual_full_p1_producer_owner() -> None:
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        coordinate_lagrange_element,
        RestrictedCellGeometryElement,
    )
    from phydrax.meshing._adaptation import _full_p1_action_sources

    root = coordinate_lagrange_element("triangle", 1)
    action = BarycentricCellGeometryElement(
        root, np.asarray(((1, 0, 0), (0.5, 0.5, 0), (0, 0, 1)))
    )
    restricted = RestrictedCellGeometryElement(
        root, "triangle", np.eye(2), np.zeros((2,))
    )
    assert _full_p1_action_sources((root, action))
    # An affine map alone does not name the full-P1 producer: restrictions
    # retain their other responsible transition and rounded-point admission.
    assert not _full_p1_action_sources((restricted,))


def test_source_embedding_cannot_certify_a_different_rounded_carrier(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.meshing._audit_topology import prepare_planar_audit_embedding

    source = _source(1e-15, budget=reference_budget)
    carrier = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0 / 3.0, 1.0 / 3.0)),
        dtype=np.float64,
    )
    mesh = CellMesh.from_triangles(
        carrier, np.asarray(((0, 3, 2), (3, 1, 2)), dtype=np.int32)
    )
    geometry = CellGeometrySpec.plc(mesh, source, budget=reference_budget)
    validity, embedding = prepare_planar_audit_embedding(
        mesh,
        geometry,
        policy=phx.meshing.CellMeshAuditPolicy(),
        limits=phx.geometry.MeshCertificateLimits(),
        junction_vertices=None,
        budget=reference_budget,
    )
    assert validity is None and embedding is None


def test_exact_witness_renewal_requires_explicit_subset_ratios(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    source = _source(1e-15, budget=reference_budget)
    unchanged = source.with_witnesses(
        source.vertex_strata,
        source.vertex_rows,
        source.vertex_parameters,
        budget=reference_budget,
    )
    assert unchanged.source_id == source.source_id
    with pytest.raises(
        ValueError, match=r"ExactPlcCellGeometrySource\.vertex_parameter_ratios"
    ):
        source.with_witnesses(
            source.vertex_strata[:3],
            source.vertex_rows[:3],
            source.vertex_parameters[:3],
            budget=reference_budget,
        )
    ratios = source.vertex_parameter_ratios
    assert ratios is not None
    subset = source.with_witnesses(
        source.vertex_strata[:3],
        source.vertex_rows[:3],
        source.vertex_parameters[:3],
        vertex_parameter_ratios=ratios[:3],
        budget=reference_budget,
    )
    assert subset.vertex_parameter_ratios == ratios[:3]


def test_exact_witness_cache_is_content_bound_and_reused_only_while_owned(
    reference_budget: CoordinateEnclosureBudget,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    source = _source(1e-15, budget=reference_budget)
    carrier = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (np.nextafter(1.0 / 3.0, 1.0), 1.0 / 3.0)),
        dtype=np.float64,
    )
    budget = coordinate_budget(10000, 1 << 20)
    with budget.activate():
        first = source.prepare(carrier, budget=budget)
        work = budget.work_units
        assert work > 0
        assert source.prepare(carrier.copy(), budget=budget) is first
        assert budget.work_units == work
        moved = carrier.copy()
        moved[3, 0] = np.nextafter(moved[3, 0], 1.0)
        changed = source.prepare(moved, budget=budget)
        assert changed is not first
        assert changed.maximum_rounding_error > first.maximum_rounding_error
        assert budget.work_units > work
        with pytest.raises(
            ValueError, match=r"ExactPlcCellGeometrySource\.source_row_bounds"
        ):
            _source(0.0, budget=reference_budget).prepare(carrier, budget=budget)
    following = coordinate_budget(10000, 1 << 20)
    with following.activate():
        # The first request still owns its issued bank: the same source object
        # reuses it, admitting and retaining its bytes without proving again.
        assert source.prepare(carrier, budget=following) is first
        assert 0 < following.work_units < work
    independent = coordinate_budget(10000, 1 << 20)
    with independent.activate():
        # Issued banks are keyed by their source object, never by equal content.
        other = _source(1e-15, budget=reference_budget)
        assert other.prepare(carrier, budget=independent) is not first
        assert independent.work_units == work


def test_constraint_split_preserves_source_identity_and_exact_dyadic_parameter() -> None:
    vertices = np.asarray(((0.2, 0.2), (0.5, 1.7)), dtype=np.float64)
    with NativeExecutionScope(MeshingLimits().native_limits()) as scope:
        points, pieces, owners, point_sources, parameters = split_constraints(
            vertices,
            np.asarray(((0, 1),), dtype=np.int64),
            0.2,
            np.empty((0, 2, 2), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            float("inf"),
            MeshingLimits(),
            scope=scope,
        )
    assert np.array_equal(point_sources[:2], (-1, -1))
    assert np.all(point_sources[2:] == 0)
    assert np.all(owners == 0)
    assert pieces.shape[0] == points.shape[0] - 1
    exact = tuple(tuple(Fraction(float(value)) for value in row) for row in vertices)
    witnessed = [
        tuple(
            a + Fraction(float(t)) * (b - a)
            for a, b in zip(exact[0], exact[1], strict=True)
        )
        for t in parameters[2:]
    ]
    assert any(
        tuple(Fraction(float(value)) for value in point) != witness
        for point, witness in zip(points[2:], witnessed, strict=True)
    )
    assert all(0 < Fraction(float(t)) < 1 for t in parameters[2:])


def test_native_constrained_steiner_points_publish_composable_witnesses() -> None:
    triangulation = ConstrainedDelaunayTriangulation(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int32),
        max_area=0.03,
        max_steiner=100,
    )
    constrained = np.flatnonzero(triangulation.constraint_sources >= 0)
    assert constrained.size > 0
    for vertex in constrained.tolist():
        parents = triangulation.constraint_parents[vertex]
        assert np.all(parents >= 0)
        assert np.all(parents < vertex)
        assert 0 < triangulation.constraint_ratios[vertex] < 1
    assert np.all(triangulation.constraint_sources[:4] == -1)


def test_planar_hole_carrier_evidence_names_each_source_bound_and_deviation(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    source, template = plate(
        limits=MeshingLimits(
            maximum_vertices=20_000,
            maximum_edges=160_000,
            maximum_faces=160_000,
            maximum_cells=160_000,
            maximum_connectivity_entries=1_280_000,
            maximum_data_bytes=20_480_000,
            maximum_work_units=640_000,
            maximum_cavity_cells=20_000,
            maximum_geometry_queries=1_280_000,
            maximum_scratch_bytes=81_920_000,
            maximum_wall_seconds=600.0,
        ),
    )
    # Preserve the registered tools/_meshing_cases.py planar-hole-feature
    # source revision and embedded-segment identity while sharing its geometry.
    # Source: tests/data/meshing/corpus.json, name=planar-hole-feature.
    revision = "fb13d4042f6bb9faa6581aa4399c848a8c4c899f88d621a6abcbecf3f4f54c2e"
    embedded = source.embedded
    if embedded is None:
        raise AssertionError(
            "fixture.plate.embedded must contain the protected segment; got None"
        )
    source = NativePlanarSource(
        source.region,
        revision,
        embedded=SegmentMesh(
            embedded.vertices,
            jnp.asarray(embedded.edges, dtype=jnp.int32),
            source_id="embedded-feature",
        ),
    )
    scope = MeshingScope(
        template.scope.source_id,
        revision,
        template.scope.entity_kind,
        template.scope.entity_dimension,
        template.scope.entity_set_id,
        template.scope.entity_ids,
    )
    request = SurfaceMeshingSpec(
        template.target,
        scope,
        planar_embedding=template.planar_embedding,
        size_controls=(
            UniformSizeControl(
                scope,
                0.2,
                maximum_size=1.5 * 0.2,
                strength=SizeControlStrength.HARD,
            ),
        ),
        size_compliance=template.size_compliance,
        quality_target=template.quality_target,
        limits=template.limits,
    )
    with NativeExecutionScope(request.limits.native_limits()) as scope:
        prepared = PreparedPlanarDomain(source, request, scope=scope)
        triangulation, _ = triangulate(prepared, monotonic(), request, scope=scope)
    used = np.unique(triangulation.triangles)
    authority, source_deviations, carrier_deviations = constraint_coordinate_source(
        source, prepared, triangulation, used, budget=reference_budget
    )
    vertex_ids = np.arange(used.size, dtype=np.int64)
    requested, achieved = source_carrier_evidence(
        prepared, authority, vertex_ids, carrier_deviations, source_deviations
    )
    bounds, deviations = dict(requested), dict(achieved)
    rows = np.asarray(authority.vertex_rows, dtype=np.int64)
    assert np.any(carrier_deviations > 0.0)
    for vertex in np.flatnonzero(rows >= 0).tolist():
        edge = int(rows[vertex])
        key = f"source:edge:{edge}:vertex:{vertex_ids[vertex]}"
        bound = bounds[f"{key}:maximum_deviation"]
        deviation = deviations[f"{key}:certified_deviation"]
        assert bound == prepared.source_bounds[edge]
        assert deviation == carrier_deviations[vertex]
        assert 0.0 <= deviation <= bound


def test_planar_cell_maps_reuse_proof_and_invalidate_changed_carriers(
    reference_budget: CoordinateEnclosureBudget,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    source = _source(1e-15, budget=reference_budget)
    carrier = np.asarray(
        ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (np.nextafter(1.0 / 3.0, 1.0), 1.0 / 3.0)),
        dtype=np.float64,
    )
    mesh = CellMesh.from_triangles(
        carrier, np.asarray(((0, 3, 2), (3, 1, 2)), dtype=np.int32)
    )
    geometry = CellGeometrySpec.plc(mesh, source, budget=reference_budget)
    budget = coordinate_budget(10000, 1 << 20)
    with budget.activate():
        first = cell_maps(mesh, geometry, budget=budget)
        from phydrax.discretization._coordinate_enclosure import (
            prepared_coordinate_vertex_images,
        )

        assert prepared_coordinate_vertex_images(
            mesh, geometry, mismatch="changed shared vertex", budget=budget
        ) is first.images
        work = budget.work_units
        assert work > 0
        assert cell_maps(mesh, geometry, budget=budget) is first
        assert budget.work_units == work
        moved = carrier.copy()
        moved[3, 0] = np.nextafter(moved[3, 0], 1.0)
        successor = mesh.with_coordinates(moved, numeric_version="moved-carrier")
        new_geometry = geometry.with_coordinates(moved)
        assert cell_maps(successor, new_geometry, budget=budget) is not first
        assert budget.work_units > work


@pytest.mark.parametrize(
    ("statistics", "relative", "minimum"),
    (
        (("p50", "p95"), 1e-12, None),
        (("p50",), 1e-12, None),
        (("p50", "p95"), 0.5, None),
        (("p50",), 0.5, 0.9),
    ),
    ids=("tight-multiple", "tight-median", "multiple-statistics", "minimum-size"),
)
def test_uniform_planar_strategy_preserves_other_hard_size_contracts(
    statistics: tuple[SizeStatistic, ...], relative: float, minimum: float | None
) -> None:
    m = phx.meshing
    source = m.NativePlanarSource(
        phx.geometry.PlanarMeshRegion(
            np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3.0) / 2.0))),
            ((0, 1, 2),),
            feature_id="equilateral",
        ),
        "r1",
    )
    scope = m.MeshingScope(
        source.source_id,
        "r1",
        m.MeshingEntityKind.GEOMETRY,
        2,
        "equilateral-region",
        np.asarray((0,), dtype=np.int64),
    )
    request = SurfaceMeshingSpec(
        m.CellMeshingTarget(2, 2, m.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            UniformSizeControl(
                scope,
                1.0,
                minimum_size=minimum,
                maximum_size=1.0,
                strength=SizeControlStrength.HARD,
            ),
        ),
        size_compliance=m.SizeCompliancePolicy(
            relative_tolerance=relative, target_statistics=statistics
        ),
    )
    result = (
        m.NativeMeshingProvider(m.NativeMeshingOptions("planar_constrained_delaunay"))
        .plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
        .execute()
    )
    assert result.mesh.coordinates.shape[0] == 3
    assert result.mesh.blocks[0].vertices.shape[0] == 1
    assert result.compliance.passed
    assert result.certification is not None and result.certification.passed


@pytest.fixture
def thin_channel_request() -> tuple[NativePlanarSource, SurfaceMeshingSpec]:
    from tools._meshing_cases import meshing_case, prepare_case

    source, request, _ = prepare_case(
        meshing_case("planar-thin-channel"), 4, 20000, 120.0
    )
    return source, request


def test_lattice_preparation_is_admitted_and_cuts_equal_source_pieces(
    thin_channel_request: tuple[NativePlanarSource, SurfaceMeshingSpec],
) -> None:
    source, request = thin_channel_request
    with NativeExecutionScope(request.limits.native_limits()) as scope:
        generic = PreparedPlanarDomain(source, request, scope=scope)
        admitted = PreparedPlanarDomain(source, request, scope=scope, lattice=True)
    assert generic.lattice_spacing == 0.0
    assert admitted.construction_size == generic.construction_size
    assert admitted.lattice_spacing == LATTICE_FRACTION * admitted.construction_size
    assert admitted.prepared_id != generic.prepared_id
    lengths = np.linalg.norm(
        admitted.points[admitted.segments[:, 1]] - admitted.points[admitted.segments[:, 0]],
        axis=1,
    )
    assert np.all(lengths <= admitted.construction_size)
    for edge, direction in enumerate(admitted.source_directions):
        pieces = int(np.ceil(np.hypot(*direction) / admitted.lattice_spacing))
        assert np.count_nonzero(admitted.segment_sources == edge) == pieces
    authored = admitted.point_sources >= 0
    counts = np.bincount(admitted.segment_sources)
    np.testing.assert_array_equal(
        admitted.point_parameters[authored] * counts[admitted.point_sources[authored]],
        np.round(admitted.point_parameters[authored] * counts[admitted.point_sources[authored]]),
    )
    graded = SurfaceMeshingSpec(
        request.target,
        request.scope,
        planar_embedding=request.planar_embedding,
        size_controls=request.size_controls,
        size_compliance=phx.meshing.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50", "p95")
        ),
        quality_target=request.quality_target,
        limits=request.limits,
    )
    with NativeExecutionScope(graded.limits.native_limits()) as scope:
        graded_prepared = PreparedPlanarDomain(source, graded, scope=scope, lattice=True)
    assert graded_prepared.lattice_spacing == 0.0


def test_lattice_construction_is_one_native_call_within_the_edge_bound(
    thin_channel_request: tuple[NativePlanarSource, SurfaceMeshingSpec],
) -> None:
    source, request = thin_channel_request
    with NativeExecutionScope(request.limits.native_limits()) as scope:
        prepared = PreparedPlanarDomain(source, request, scope=scope, lattice=True)
        triangulation, inserted = triangulate(
            prepared, monotonic(), request, scope=scope
        )
    assert inserted == 0
    assert triangulation.evidence.status == "ok"
    assert triangulation.evidence.minimum_angle_degrees >= prepared.minimum_angle_degrees
    free = triangulation.segment_ids < 0
    rows, opposite = np.nonzero(free)
    first = triangulation.triangles[rows, (opposite + 1) % 3]
    second = triangulation.triangles[rows, (opposite + 2) % 3]
    lengths = np.linalg.norm(
        triangulation.points[first] - triangulation.points[second], axis=1
    )
    assert np.all(lengths <= prepared.construction_size)
    limited = SurfaceMeshingSpec(
        request.target,
        request.scope,
        planar_embedding=request.planar_embedding,
        size_controls=request.size_controls,
        size_compliance=request.size_compliance,
        quality_target=request.quality_target,
        limits=MeshingLimits(maximum_work_units=100, maximum_scratch_bytes=81_920_000),
    )
    with (
        pytest.raises(phx.meshing.MeshingFailure) as caught,
        NativeExecutionScope(limited.limits.native_limits()) as scope,
    ):
        triangulate(prepared, monotonic(), limited, scope=scope)
    failure = caught.value
    assert failure.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert failure.evidence.stage == "surface_meshing"
    achieved = dict(failure.evidence.achieved)
    assert achieved["native_planar:work_units"] <= 100
    assert achieved["native_planar:work_refusals"] == 1


def test_compliant_planar_generation_binds_the_admitted_preparation(
    monkeypatch: pytest.MonkeyPatch,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:
    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._coordinate_identity import coordinate_scope_key
    from phydrax.discretization._coordinate_operands import CoordinateOperandCollector
    from phydrax.meshing import _plc_planar_support as support
    from phydrax.meshing.providers import _native_planar as route
    from phydrax.meshing.providers._native_planar_domain import source_arrays
    from phydrax.meshing.qualification._native_lifecycle import execute_native_consumer

    collector = CoordinateOperandCollector()
    measured: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    consumed: list[np.ndarray] = []
    original_measure = route.measure_edge_lengths
    original_decision = route.median_repair_needed
    original_compliance = route.planar_compliance

    def measure(
        vertices: np.ndarray, edges: np.ndarray, /, *, scope: NativeExecutionScope
    ) -> np.ndarray:
        lengths = original_measure(vertices, edges, scope=scope)
        measured.append((vertices, edges, lengths))
        return lengths

    def decision(*args: Any, **kwargs: Any) -> bool:
        consumed.append(args[3])
        return original_decision(*args, **kwargs)

    def compliance(*args: Any, **kwargs: Any) -> Any:
        assert args[2] is measured[-1][0]
        assert args[4] is measured[-1][1]
        consumed.append(args[5])
        return original_compliance(*args, **kwargs)

    monkeypatch.setattr(route, "measure_edge_lengths", measure)
    monkeypatch.setattr(route, "median_repair_needed", decision)
    monkeypatch.setattr(route, "planar_compliance", compliance)
    m = phx.meshing
    source = NativePlanarSource(
        phx.geometry.PlanarMeshRegion(
            np.asarray(
                (
                    (0, 0),
                    (2, 0),
                    (2, 2),
                    (0, 2),
                    (0.5, 0.5),
                    (0.5, 1.5),
                    (1.5, 1.5),
                    (1.5, 0.5),
                ),
                dtype=np.float64,
            ),
            ((0, 1, 2, 3), (4, 5, 6, 7)),
            feature_id="plate",
        ),
        "r1",
    )
    scope = MeshingScope(
        "plate",
        "r1",
        m.MeshingEntityKind.GEOMETRY,
        2,
        "plate-region",
        np.asarray((0,), dtype=np.int64),
    )
    request = SurfaceMeshingSpec(
        m.CellMeshingTarget(2, 2, m.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            UniformSizeControl(
                scope, 1.0, maximum_size=1.5, strength=SizeControlStrength.HARD
            ),
        ),
        size_compliance=m.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
    )
    with NativeExecutionScope(request.limits.native_limits()) as scope:
        prepared = PreparedPlanarDomain(source, request, scope=scope, lattice=True)
        reference, inserted = triangulate(prepared, monotonic(), request, scope=scope)
    assert inserted == 0 and reference.points.shape[0] == 12
    assert reference.triangles.shape[0] == 12
    result = (
        m.NativeMeshingProvider(m.NativeMeshingOptions("planar_constrained_delaunay"))
        .plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
        .execute(coordinate_operands=collector)
    )
    assert result.mesh.coordinates.shape[0] == 12
    assert result.mesh.blocks[0].vertices.shape[0] == 12
    # Preserved-source reference: 12 unit boundary edges and 12 diagonal
    # interior edges of length sqrt(1/2), not a newly relabeled mesh.
    assert dict(result.compliance.achieved)[
        f"size:{request.size_controls[0].control_id}:p50_edge"
    ] == 0.5 * (1.0 + np.sqrt(0.5))
    inspection = stage_report(result, m.MeshingStageKind.SOURCE_INSPECTION)
    generation = stage_report(result, m.MeshingStageKind.SURFACE_MESHING)
    assert inspection.output_ids == (prepared.prepared_id,)
    assert generation.input_ids == inspection.output_ids
    assert generation.output_ids[0] == reference.evidence.evidence_id
    assert result.compliance.passed
    assert result.certification is not None and result.certification.passed
    assert len(measured) == 1
    vertices, edges, lengths = measured[0]
    assert len(consumed) == 2 and all(value is lengths for value in consumed)
    np.testing.assert_array_equal(vertices, np.asarray(result.mesh.coordinates))
    connectivity = result.mesh.connectivity
    assert isinstance(connectivity, PolygonalConnectivity)
    np.testing.assert_array_equal(
        edges, np.sort(np.asarray(connectivity.edges), axis=1)
    )
    np.testing.assert_array_equal(
        lengths, np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    )
    operands = collector.for_result(result.mesh, result.geometry)
    assert operands is not None

    def unexpected_reconstruction(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a paid immutable operand bank must not be reconstructed")

    def reuse_numeric_data(budget: CoordinateEnclosureBudget) -> Any:
        from phydrax.geometry._embedding_state import mesh_facets
        from phydrax.geometry._global_embedding import source_bank_binary64_rows

        before = budget.work_units
        facet_data = mesh_facets(result.mesh, geometry=result.geometry, budget=budget)
        bank = algebra.prepared_coordinate_source_bank(result.geometry, budget=budget)
        view, mask = source_bank_binary64_rows(bank, budget=budget)
        assert facet_data.rows.shape[0] == 3 * result.mesh.blocks[0].cell_count
        assert not facet_data.rows.flags.writeable and not view.flags.writeable
        assert mask.shape == (result.mesh.coordinates.shape[0],)
        assert budget.work_units - before < 16
        return algebra.prepared_coordinate_vertex_images(
            result.mesh, result.geometry, mismatch="unequal corners", budget=budget
        )

    with monkeypatch.context() as reuse:
        reuse.setattr(CellGeometrySpec, "source_coordinates", unexpected_reconstruction)
        consumed_result = execute_native_consumer(
            result,
            request.limits,
            reuse_numeric_data,
            prepared_operands=operands,
            failure_factory=lambda *args: ValueError(str(args)),
        )
    assert consumed_result.value is not None
    assert consumed_result.result.mesh is result.mesh
    with pytest.raises(ValueError, match="exact paid earning receipt"):
        execute_native_consumer(
            result,
            request.limits,
            lambda budget: None,
            prepared_operands=replace(operands, _physical_bytes=0),
            failure_factory=lambda *args: ValueError(str(args)),
        )
    changed_mesh = result.mesh.with_coordinates(
        np.asarray(result.mesh.coordinates) + 0.125, numeric_version="foreign-carrier"
    )
    execution_evidence = result.execution_evidence
    assert execution_evidence is not None
    with pytest.raises(ValueError, match="current source"):
        execute_native_consumer(
            result,
            request.limits,
            lambda budget: operands.require(
                changed_mesh, result.geometry, execution_evidence, budget
            ),
            failure_factory=lambda *args: ValueError(str(args)),
        )
    facet_data = next(iter(operands._facets.values()))
    original_shape = facet_data.rows.shape
    try:
        # In-place view-metadata mutation is the contract under test; no non-deprecated equivalent exists.
        facet_data.rows.shape = (facet_data.rows.size,)  # ty: ignore[deprecated]
        with pytest.raises(ValueError, match="numeric view metadata changed"):
            execute_native_consumer(
                result,
                request.limits,
                lambda budget: None,
                prepared_operands=operands,
                failure_factory=lambda *args: ValueError(str(args)),
            )
    finally:
        facet_data.rows.shape = original_shape  # ty: ignore[deprecated]

    domain = prepared.domain(source)
    _, source_edges, _ = source_arrays(source)
    transfer = m.PlcAssociationTransfer(
        domain,
        result.coordinate_contract,
        source.source_revision,
        edge_vertices=source_edges,
        triangle_vertices=np.empty((0, 3), dtype=np.int64),
        triangle_facets=np.empty((0,), dtype=np.int64),
        facet_regions=domain.facet_regions,
    )
    preparation = coordinate_budget(640000, 81_920_000)
    transfer.classes(result, budget=preparation)
    proof = support.issue_planar_source_proof(transfer, result, budget=preparation)
    assert proof is not None
    execution = coordinate_budget(640000, 81_920_000)
    support.bind_planar_source_proof(transfer, result, proof, budget=execution)
    work, retained = execution.work_units, execution.retained_basis_bytes
    assert support.cell_maps(result.mesh, result.geometry, budget=execution) is proof.maps
    assert algebra.prepared_coordinate_source_bank(
        result.geometry, budget=execution
    ) is proof.source_bank
    result.geometry.resolve(result.mesh, budget=execution)
    assert coordinate_scope_key(result.mesh, result.geometry) in (
        execution.plc_carrier_binding_cache
    )
    support.bind_planar_source_proof(transfer, result, proof, budget=execution)
    assert execution.work_units - work == len(support._planar_proof_fields(proof))
    assert execution.retained_basis_bytes == retained
    with pytest.raises(ValueError, match="foreign or altered proof"):
        support.bind_planar_source_proof(
            transfer, result, replace(proof, source_bank=()), budget=execution
        )
    with pytest.raises(ValueError):
        proof.maps.vertices.setflags(write=True)
    with pytest.raises(ValueError):
        proof.classes[0].dimensions.setflags(write=True)
    dimensions = proof.classes[0].dimensions
    shape = dimensions.shape
    try:
        # In-place view-metadata mutation is the contract under test; no non-deprecated equivalent exists.
        dimensions.shape = (dimensions.size, 1)  # ty: ignore[deprecated]
        with pytest.raises(ValueError, match="foreign or altered proof"):
            support.bind_planar_source_proof(transfer, result, proof, budget=execution)
    finally:
        dimensions.shape = shape  # ty: ignore[deprecated]
    images = proof.maps.images
    try:
        object.__setattr__(proof.maps, "images", tuple(reversed(images)))
        with pytest.raises(ValueError, match="foreign or altered proof"):
            support.bind_planar_source_proof(transfer, result, proof, budget=execution)
    finally:
        object.__setattr__(proof.maps, "images", images)
    try:
        proof.maps.polynomials[0] = (
            {(0, 0): Fraction(17)}, {(0, 0): Fraction(0)}
        )
        with pytest.raises(ValueError, match="foreign or altered proof"):
            support.bind_planar_source_proof(transfer, result, proof, budget=execution)
    finally:
        proof.maps.polynomials.clear()


def test_planar_initial_native_work_refusal_retains_public_prefix() -> None:
    from tools._meshing_cases import meshing_case, prepare_case

    source, template, options = prepare_case(
        meshing_case("planar-thin-channel"), 4, 20000, 120.0
    )
    request = SurfaceMeshingSpec(
        template.target,
        template.scope,
        planar_embedding=template.planar_embedding,
        size_controls=template.size_controls,
        size_compliance=template.size_compliance,
        quality_target=template.quality_target,
        limits=MeshingLimits(maximum_work_units=1, maximum_scratch_bytes=81_920_000),
    )
    with pytest.raises(phx.meshing.MeshingFailure) as caught:
        phx.meshing.NativeMeshingProvider(options).plan(
            source, request, coordinate_contract=phx.SpatialCoordinateContract.si()
        ).execute()
    assert caught.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(caught.value.evidence.requested)["maximum_work_units"] == 1
    prefix = dict(caught.value.evidence.achieved)
    assert prefix["native_planar:work_units"] == 1
    assert prefix["native_planar:work_refusals"] == 1
    cause = caught.value.__cause__
    while isinstance(cause, phx.meshing.MeshingFailure):
        cause = cause.__cause__
    assert isinstance(cause, MeshcoreError)


def test_median_repair_uses_the_selected_candidates_measured_edges(
    thin_channel_request: tuple[NativePlanarSource, SurfaceMeshingSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing.providers import _native_planar as route

    source, request = thin_channel_request
    measured: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    original_measure = route.measure_edge_lengths
    original_compliance = route.planar_compliance

    def measure(
        vertices: np.ndarray, edges: np.ndarray, /, *, scope: NativeExecutionScope
    ) -> np.ndarray:
        lengths = original_measure(vertices, edges, scope=scope)
        measured.append((vertices, edges, lengths))
        return lengths

    def compliance(*args: Any, **kwargs: Any) -> Any:
        assert len(measured) == 2
        assert args[2] is measured[1][0]
        assert args[4] is measured[1][1]
        assert args[5] is measured[1][2]
        assert args[5] is not measured[0][2]
        return original_compliance(*args, **kwargs)

    monkeypatch.setattr(route, "measure_edge_lengths", measure)
    monkeypatch.setattr(route, "planar_compliance", compliance)
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("planar_constrained_delaunay")
        )
        .plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
        .execute()
    )
    assert len(measured) == 2
    connectivity = result.mesh.connectivity
    assert isinstance(connectivity, PolygonalConnectivity)
    np.testing.assert_array_equal(
        measured[1][1], np.sort(np.asarray(connectivity.edges), axis=1)
    )
    assert result.compliance.passed
    assert result.certification is not None and result.certification.passed
