# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Declared radical coordinates govern periodic statistics and durable source identity."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from fractions import Fraction
from pathlib import Path

import equinox as eqx
import numpy as np
import pytest

from phydrax._fingerprint import tree_content_equal
from phydrax._model._structure import model_from_array_recipe, model_structure_recipe
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    PeriodicCell,
    PeriodicMeshTopology,
)
from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._exact_algebraic_geometry import (
    ExactAlgebraicCellGeometrySource,
    RadicalElement,
)
from phydrax.discretization._exact_rational_geometry import rational_source_layouts
from phydrax.geometry._affine_mapped_contact import interiors_overlap
from phydrax.geometry._embedding_cells import Cell, convex_triangle_union, vertices
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry._prism_embedding import adjacent_prism_injectivity
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    reconstruct_meshing_source,
    register_meshing_source_artifacts,
    validate_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    PeriodicQuotientEvidence,
    SizeCompliancePolicy,
    UniformSizeControl,
)
from phydrax.meshing._periodic import (
    certify_periodic_embedding,
    ExactPeriodicEdgeSizeEvidence,
    periodic_edge_size_evidence,
    periodic_edge_size_quantile,
    periodic_edge_size_quantile_compare,
    periodic_edge_squared_lengths,
    periodic_exact_edge_size_evidence,
    simplex_entity_corners,
    source_entity_contains,
)
from phydrax.meshing.providers._native_periodic import _periodic_size_compliance
from tests._support.meshing_fixtures import mesh_scope


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    ticket = HostBatchTicket(10_000_000, bytes_upper=128_000_000)
    with (
        ticket,
        CoordinateEnclosureBudget(
            ticket, maximum_work_units=10_000_000, maximum_memory_bytes=128_000_000
        ).activate() as ledger,
    ):
        yield ledger


def _algebraic_strip(
    height: RadicalElement,
    budget: CoordinateEnclosureBudget,
    *,
    scale: Fraction = Fraction(1),
    vertical_offset: Fraction = Fraction(0),
) -> tuple[CellMesh, CellGeometrySpec, ExactAlgebraicCellGeometrySource]:
    """Four triangles tile an x-periodic strip; no special lattice source exists."""
    bank = tuple(
        (RadicalElement(x * scale), y * scale + vertical_offset)
        for x, y in (
            (Fraction(0), RadicalElement()),
            (Fraction(1), RadicalElement()),
            (Fraction(2), RadicalElement()),
            (Fraction(1, 2), height),
            (Fraction(3, 2), height),
            (Fraction(5, 2), height),
        )
    )
    cells = np.asarray(((0, 1, 3), (1, 4, 3), (1, 2, 4), (2, 5, 4)), dtype=np.int32)
    elements = {"strip": coordinate_lagrange_element("triangle", 1)}
    routes = {"strip": cells}
    source = ExactAlgebraicCellGeometrySource(
        bank,
        rational_source_layouts(elements, routes),
        source_identity="authored-algebraic-periodic-strip",
        maximum_work=100_000,
        maximum_bits=1024,
        budget=budget,
    )
    points = source.representatives(budget=budget)
    blocks = (CellBlock("strip", "triangle", cells),)
    plain = CellMesh(points, blocks)
    topology = PeriodicMeshTopology(
        plain,
        PeriodicCell(
            np.asarray(((float(2 * scale), 0.0), (0.0, 1.0)), dtype=np.float64),
            periodic_axes=(True, False),
        ),
        np.asarray((0, 1, 0, 3, 4, 3), dtype=np.int64),
        np.asarray(((0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (1, 0)), dtype=np.int64),
    )
    mesh = CellMesh(points, blocks, periodic_topology=topology)
    geometry = CellGeometrySpec(
        elements, routes, points, exact_source=source, budget=budget
    )
    return mesh, geometry, source


@pytest.mark.parametrize("scale", (Fraction(1, 2), Fraction(1), Fraction(2)))
def test_declared_algebraic_periodic_quantiles_meet_exact_h(
    scale: Fraction, budget: CoordinateEnclosureBudget
) -> None:
    mesh, geometry, source = _algebraic_strip(
        RadicalElement.sqrt(3) / 2, budget, scale=scale
    )
    squared = periodic_edge_squared_lengths(mesh, geometry=geometry, budget=budget)
    assert squared.shape == (8,)
    assert all(value == scale * scale for value in squared)
    evidence = periodic_exact_edge_size_evidence(mesh, geometry, budget=budget)
    assert isinstance(evidence, ExactPeriodicEdgeSizeEvidence)
    lengths = evidence.representative_lengths
    growth = evidence.maximum_local_edge_ratio
    assert evidence.squared_lengths == tuple(squared)
    assert all(error == 0 for error in evidence.length_rounding_error_bounds)
    quotient = PeriodicQuotientEvidence(mesh, geometry=geometry, budget=budget)
    assert quotient.all_cells_valid
    assert quotient.total_measure == float(RadicalElement.sqrt(3) * scale * scale)
    np.testing.assert_array_equal(lengths, np.full(8, float(scale), dtype=np.float64))
    assert growth == 1.0
    quantiles = np.asarray((0.0, 0.25, 0.5, 0.95, 1.0), dtype=np.float64)
    np.testing.assert_array_equal(
        periodic_edge_size_quantile(mesh, quantiles, geometry=geometry, budget=budget),
        np.full(quantiles.shape, float(scale), dtype=np.float64),
    )
    for quantile in quantiles:
        assert (
            periodic_edge_size_quantile_compare(
                mesh, float(quantile), scale, geometry=geometry, budget=budget
            )
            == 0
        )
    assert geometry.source_coordinates(budget=budget) == source.coefficients
    assert source.prepare(geometry.coordinates, budget=budget).maximum_rounding_error > 0
    control = UniformSizeControl(mesh_scope(mesh, 2), float(scale))
    policy = SizeCompliancePolicy(absolute_tolerance=0.0, relative_tolerance=0.0)
    _, achieved, issues = _periodic_size_compliance(
        mesh, geometry, control, policy, lengths, growth, budget=budget
    )
    assert not issues
    for name in ("p50", "p95"):
        assert dict(achieved)[f"size:{control.control_id}:{name}_edge"] == float(scale)


def test_periodic_quantiles_interpolate_exact_algebraic_edge_ranks(
    budget: CoordinateEnclosureBudget,
) -> None:
    # Four horizontal edges have length one, four slanted edges length two.
    # The median must interpolate the two ranks, not select either endpoint.
    mesh, geometry, _ = _algebraic_strip(RadicalElement.sqrt(15) / 2, budget)
    squared = periodic_edge_squared_lengths(mesh, geometry=geometry, budget=budget)
    assert sorted(squared) == [1] * 4 + [4] * 4
    quantiles = np.asarray(((0.0, 0.25), (0.5, 0.95), (1.0, 0.375)), dtype=np.float64)
    expected = np.asarray(((1.0, 1.0), (1.5, 2.0), (2.0, 1.0)), dtype=np.float64)
    np.testing.assert_array_equal(
        periodic_edge_size_quantile(mesh, quantiles, geometry=geometry, budget=budget),
        expected,
    )
    for target, comparison in (
        (Fraction(3, 2), 0),
        (Fraction(3, 2) - Fraction(1, 2**100), 1),
        (Fraction(3, 2) + Fraction(1, 2**100), -1),
    ):
        assert (
            periodic_edge_size_quantile_compare(
                mesh, 0.5, target, geometry=geometry, budget=budget
            )
            == comparison
        )


def test_subulp_source_perturbation_fails_exact_h_despite_equal_reported_floats(
    budget: CoordinateEnclosureBudget,
) -> None:
    height = RadicalElement.sqrt(3) / 2
    mesh, geometry, source = _algebraic_strip(height, budget)
    perturbed_mesh, perturbed, changed_source = _algebraic_strip(
        height + Fraction(1, 2**100), budget
    )
    assert source.source_id != changed_source.source_id
    assert source.coefficients != changed_source.coefficients
    assert (
        np.asarray(mesh.coordinates).tobytes()
        == np.asarray(perturbed_mesh.coordinates).tobytes()
    )
    assert mesh.mesh_id == perturbed_mesh.mesh_id
    squared = periodic_edge_squared_lengths(mesh, geometry=perturbed, budget=budget)
    assert sum(value == 1 for value in squared) == 4
    assert sum(value > 1 for value in squared) == 4
    quantiles = np.asarray((0.5, 0.95), dtype=np.float64)
    np.testing.assert_array_equal(
        periodic_edge_size_quantile(mesh, quantiles, geometry=perturbed, budget=budget),
        np.ones(2, dtype=np.float64),
    )
    assert (
        periodic_edge_size_quantile_compare(
            mesh, 0.25, Fraction(1), geometry=perturbed, budget=budget
        )
        == 0
    )
    for quantile in quantiles:
        assert (
            periodic_edge_size_quantile_compare(
                mesh, float(quantile), Fraction(1), geometry=perturbed, budget=budget
            )
            == 1
        )
    evidence = periodic_exact_edge_size_evidence(mesh, perturbed, budget=budget)
    assert isinstance(evidence, ExactPeriodicEdgeSizeEvidence)
    lengths = evidence.representative_lengths
    growth = evidence.maximum_local_edge_ratio
    assert any(error > 0 for error in evidence.length_rounding_error_bounds)
    np.testing.assert_array_equal(lengths, np.ones(8, dtype=np.float64))
    control = UniformSizeControl(mesh_scope(mesh, 2), 1.0)
    policy = SizeCompliancePolicy(absolute_tolerance=0.0, relative_tolerance=0.0)
    _, achieved, issues = _periodic_size_compliance(
        mesh, perturbed, control, policy, lengths, growth, budget=budget
    )
    assert set(issues) == {
        f"target_size_p50:{control.control_id}",
        f"target_size_p95:{control.control_id}",
    }
    for name in ("p50", "p95"):
        assert dict(achieved)[f"size:{control.control_id}:{name}_edge"] == 1.0
    assert geometry.source_coordinates(budget=budget) != perturbed.source_coordinates(
        budget=budget
    )


def test_algebraic_source_and_geometry_archive_preserve_canonical_authority(
    tmp_path: Path, budget: CoordinateEnclosureBudget
) -> None:
    mesh, geometry, source = _algebraic_strip(RadicalElement.sqrt(3) / 2, budget)
    _, equivalent, equivalent_source = _algebraic_strip(
        RadicalElement(terms=((12, Fraction(1, 4)),)), budget
    )
    assert source.source_id == equivalent_source.source_id
    assert geometry.geometry_layout_id == equivalent.geometry_layout_id
    register_meshing_source_artifacts()
    receipt = write_meshing_source_closure(
        tmp_path / "algebraic-periodic-source", (source, geometry, mesh)
    )
    restored_source, restored_geometry, restored_mesh = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored_source, ExactAlgebraicCellGeometrySource)
    assert isinstance(restored_geometry, CellGeometrySpec)
    assert isinstance(restored_mesh, CellMesh)
    assert tree_content_equal(restored_source, source)
    assert tree_content_equal(restored_geometry, geometry)
    assert restored_source.source_id == source.source_id
    assert restored_geometry.geometry_layout_id == geometry.geometry_layout_id
    assert restored_geometry.source_coordinates(budget=budget) == source.coefficients
    assert (
        np.asarray(restored_geometry.coordinates).tobytes()
        == np.asarray(geometry.coordinates).tobytes()
    )
    renewed = reconstruct_meshing_source(restored_geometry, meshes=(restored_mesh,))
    assert isinstance(renewed, CellGeometrySpec)
    assert tree_content_equal(renewed, geometry)
    np.testing.assert_array_equal(
        periodic_edge_size_quantile(
            restored_mesh,
            np.asarray((0.5, 0.95), dtype=np.float64),
            geometry=renewed,
            budget=budget,
        ),
        np.ones(2, dtype=np.float64),
    )


def test_algebraic_source_lifecycle_refuses_tampered_authority_and_carrier(
    budget: CoordinateEnclosureBudget,
) -> None:
    _, geometry, source = _algebraic_strip(RadicalElement.sqrt(3) / 2, budget)
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(source)
    source_identity = next(
        index
        for index, field in enumerate(fields(source))
        if field.name == "source_identity"
    )
    recipe["items"][source_identity] = model_structure_recipe("forged-algebraic-source")
    with pytest.raises(ValueError, match="authority_binding"):
        forged = model_from_array_recipe(recipe, {}, prefix="algebraic_source")
        validate_meshing_source_closure(forged)
    changed = np.asarray(geometry.coordinates).copy()
    changed[3, 1] = np.nextafter(changed[3, 1], np.inf)
    with pytest.raises(ValueError, match="representatives"):
        source.prepare(changed, budget=budget)
    corrupted = eqx.tree_at(lambda value: value.coordinates, geometry, changed)
    with pytest.raises(ValueError, match="representatives"):
        validate_meshing_source_closure(corrupted)


def test_float_only_periodic_evidence_remains_bitwise_unchanged(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, geometry, _ = _algebraic_strip(RadicalElement.sqrt(3) / 2, budget)
    topology = mesh.periodic_topology
    assert topology is not None
    leaders = np.asarray(topology.orbit_representatives(1))
    edges = simplex_entity_corners(mesh, 1)[leaders]
    roots = np.asarray(topology.vertex_representatives)
    shifts = np.asarray(topology.vertex_shifts)[edges]
    points = np.asarray(mesh.coordinates)[roots[edges]]
    assert isinstance(topology.cell, PeriodicCell)
    delta = (
        points[:, 1]
        - points[:, 0]
        + (shifts[:, 1] - shifts[:, 0]) @ np.asarray(topology.cell.vectors)
    )
    expected = np.linalg.norm(delta, axis=1)
    before = PeriodicQuotientEvidence(mesh)
    lengths, growth = periodic_edge_size_evidence(mesh)
    assert lengths.tobytes() == expected.tobytes()
    assert growth == float(np.max(expected) / np.min(expected))
    periodic_exact_edge_size_evidence(mesh, geometry, budget=budget)
    after = PeriodicQuotientEvidence(mesh, geometry=None)
    assert tree_content_equal(before, after)
    restored = PeriodicQuotientEvidence.from_stored_fields(before, mesh=mesh)
    assert tree_content_equal(before, restored)
    repeated, repeated_growth = periodic_edge_size_evidence(mesh)
    assert repeated.tobytes() == lengths.tobytes()
    assert repeated_growth == growth


def test_exact_periodic_edge_evidence_reconstruction_refuses_changed_lengths(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, geometry, _ = _algebraic_strip(
        RadicalElement.sqrt(3) / 2 + Fraction(1, 2**100), budget
    )
    evidence = periodic_exact_edge_size_evidence(mesh, geometry, budget=budget)
    restored = ExactPeriodicEdgeSizeEvidence.from_stored_fields(
        evidence, mesh=mesh, geometry=geometry, budget=budget
    )
    assert tree_content_equal(restored, evidence)
    changed = np.asarray(evidence.representative_lengths).copy()
    changed[0] = np.nextafter(changed[0], np.inf)
    corrupted = eqx.tree_at(lambda value: value.representative_lengths, evidence, changed)
    with pytest.raises(ValueError, match=r"ExactPeriodicEdgeSizeEvidence\.fields"):
        ExactPeriodicEdgeSizeEvidence.from_stored_fields(
            corrupted, mesh=mesh, geometry=geometry, budget=budget
        )


@pytest.mark.parametrize("radicands", ((6, 10, 15), (2, 3, 5)))
def test_multiquadratic_coordinate_expression_keeps_periodic_source_identity(
    radicands: tuple[int, ...], budget: CoordinateEnclosureBudget
) -> None:
    assert RadicalElement.sqrt(6) * RadicalElement.sqrt(10) == 2 * RadicalElement.sqrt(15)
    value = sum(
        (RadicalElement.sqrt(radicand) for radicand in radicands),
        RadicalElement(1),
    )
    reciprocal = value.reciprocal()
    assert value > 0
    assert -value < 0
    assert reciprocal > 0
    assert value * reciprocal == 1
    height = RadicalElement.sqrt(3) / 2
    mesh, geometry, source = _algebraic_strip(height, budget)
    _, equivalent_geometry, equivalent_source = _algebraic_strip(
        value * height * reciprocal, budget
    )
    assert equivalent_source.source_id == source.source_id
    assert equivalent_geometry.geometry_layout_id == geometry.geometry_layout_id
    np.testing.assert_array_equal(
        periodic_edge_size_quantile(
            mesh,
            np.asarray((0.5, 0.95), dtype=np.float64),
            geometry=equivalent_geometry,
            budget=budget,
        ),
        np.ones(2, dtype=np.float64),
    )


def test_declared_algebraic_periodic_embedding_uses_source_maps(
    budget: CoordinateEnclosureBudget,
) -> None:
    mesh, geometry, _ = _algebraic_strip(RadicalElement.sqrt(3) / 2, budget)
    evidence = certify_periodic_embedding(mesh, geometry=geometry, budget=budget)
    certificate = evidence.global_embedding
    assert certificate is not None
    assert certificate.status == "certified"
    assert evidence.image_count >= 3


def test_source_work_cap_does_not_reserve_unperformed_preparation(
    budget: CoordinateEnclosureBudget,
) -> None:
    bank = (
        (RadicalElement(), RadicalElement()),
        (RadicalElement(1), RadicalElement()),
        (RadicalElement(), RadicalElement.sqrt(2)),
    )
    elements = {"triangle": coordinate_lagrange_element("triangle", 1)}
    routes = {"triangle": np.asarray(((0, 1, 2),), dtype=np.int32)}
    source = ExactAlgebraicCellGeometrySource(
        bank,
        rational_source_layouts(elements, routes),
        source_identity="algebraic-preparation-demand",
        maximum_work=budget.maximum_work_units,
        maximum_bits=1024,
        budget=budget,
    )
    prepared = source.prepare(source.representatives(budget=budget), budget=budget)
    assert prepared.vertices == bank
    assert 0 < prepared.operation_count < budget.maximum_work_units


@pytest.mark.parametrize(
    ("outer", "expected"),
    (
        ((Fraction(1), Fraction(-1)), True),
        ((Fraction(1), Fraction(1)), False),
        ((Fraction(4), Fraction(-1)), False),
        ((Fraction(1), Fraction(0)), False),
    ),
    ids=("convex", "same-side", "concave", "degenerate"),
)
def test_triangle_union_exact_physical_scalars_preserve_rational_decisions(
    outer: tuple[Fraction, Fraction],
    expected: bool,
    budget: CoordinateEnclosureBudget,
) -> None:
    points = (
        (Fraction(0), Fraction(0)),
        (Fraction(2), Fraction(0)),
        (Fraction(1), Fraction(1)),
        outer,
    )
    start = budget.work_units
    temporary = budget.temporary_bytes_upper
    assert convex_triangle_union(*points, budget=budget) is expected
    assert budget.work_units == start
    root = RadicalElement.sqrt(2)
    transformed = tuple((x + root, y * root) for x, y in points)
    assert convex_triangle_union(*transformed, budget=budget) is expected
    assert budget.work_units > start
    assert budget.temporary_bytes_upper == temporary


def test_cell_physical_corner_bank_preserves_rational_reference_charts() -> None:
    reference = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    root = RadicalElement.sqrt(2)
    corners = tuple((x, y * root) for x, y in reference)
    cell = Cell(
        ({(1, 0): Fraction(1)}, {(0, 1): root}),
        "simplex",
        "triangle",
        2,
        (0, 1, 2),
        reference,
        corners,
    )
    assert cell.reference_vertices is reference
    assert cell.corner_images is corners
    with pytest.raises(ValueError, match=r"Cell\.corner_images"):
        Cell(
            cell.coordinates,
            cell.domain,
            cell.kind,
            cell.dimension,
            cell.vertices,
            reference,
            corners[:2],
        )
    with pytest.raises(TypeError, match=r"Cell\.reference_vertices"):
        Cell(
            cell.coordinates,
            cell.domain,
            cell.kind,
            cell.dimension,
            cell.vertices,
            corners,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("algebraic", (False, True), ids=("rational", "radical"))
def test_prism_atlas_preserves_exact_physical_scalars(
    algebraic: bool, budget: CoordinateEnclosureBudget
) -> None:
    height = RadicalElement.sqrt(2) if algebraic else Fraction(3, 2)
    reference = vertices("prism")
    first = Cell(
        (
            {(1, 0, 0): Fraction(1)},
            {(0, 1, 0): height},
            {(0, 0, 1): Fraction(1)},
        ),
        "prism",
        "prism",
        3,
        (0, 1, 2, 3, 4, 5),
        reference,
    )
    second = Cell(
        (
            {(0, 0, 0): Fraction(1), (0, 1, 0): Fraction(-1)},
            {(1, 0, 0): height, (0, 1, 0): height},
            {(0, 0, 1): Fraction(1)},
        ),
        "prism",
        "prism",
        3,
        (1, 6, 2, 4, 7, 5),
        reference,
    )
    limits = MeshCertificateLimits()
    proven, work = adjacent_prism_injectivity(
        first, second, limits, limits.maximum_subdivision_pieces, budget=budget
    )
    assert proven
    assert work == 2


def test_algebraic_overlap_releases_predicate_temporaries(
    budget: CoordinateEnclosureBudget,
) -> None:
    height = RadicalElement.sqrt(2)
    first = np.asarray(
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), height),
        ),
        dtype=object,
    )
    second = first + np.asarray((Fraction(2), Fraction(0)), dtype=object)
    start_work = budget.work_units
    start_temporary = budget.temporary_bytes_upper
    start_retained = budget.retained_basis_bytes
    for _ in range(16):
        assert not interiors_overlap(first, second, budget=budget)
        assert budget.temporary_bytes_upper == start_temporary
        assert budget.retained_basis_bytes == start_retained
    assert budget.work_units > start_work
    assert budget.peak_bytes_upper > start_retained + start_temporary


@pytest.mark.parametrize("algebraic", (False, True), ids=("rational", "radical"))
@pytest.mark.parametrize("dimension", (2, 3), ids=("edge", "face"))
def test_source_containment_keeps_exact_affine_support_before_parameter_signs(
    algebraic: bool, dimension: int
) -> None:
    height = RadicalElement.sqrt(2) if algebraic else Fraction(3, 2)
    quarter = Fraction(1, 4)
    epsilon = Fraction(1, 2**100)
    if dimension == 2:
        source = np.asarray(((0, 0), (1, height)), dtype=object)
        rows = (
            ((quarter, height / 4), (3 * quarter, 3 * height / 4)),
            ((quarter + epsilon, height / 4), (3 * quarter, 3 * height / 4)),
            ((Fraction(2), 2 * height), (Fraction(3), 3 * height)),
            ((Fraction(0), Fraction(0)), (Fraction(1), height)),
        )
    else:
        source = np.asarray(((0, 0, 0), (1, 0, 0), (0, height, 0)), dtype=object)
        rows = (
            ((quarter, height / 4, Fraction(0)), (quarter, height / 2, Fraction(0))),
            ((quarter, height / 4, epsilon), (quarter, height / 2, Fraction(0))),
            (
                (Fraction(2), Fraction(0), Fraction(0)),
                (Fraction(0), 2 * height, Fraction(0)),
            ),
            (
                (Fraction(0), Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0), Fraction(0)),
            ),
        )
    corners = np.asarray(rows, dtype=object)
    np.testing.assert_array_equal(
        source_entity_contains(corners, source), (True, False, False, True)
    )
    np.testing.assert_array_equal(
        source_entity_contains(corners, np.zeros_like(source)),
        np.zeros(4, dtype=np.bool_),
    )


@pytest.mark.parametrize("short", (0, 1), ids=("derived-bound", "one-byte-short"))
def test_algebraic_source_owned_demand_matches_coordinate_retention_accounting(
    short: int,
    monkeypatch: pytest.MonkeyPatch,
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization import _cell_geometry as geometry_owner

    _, _, source = _algebraic_strip(RadicalElement.sqrt(3) / 2, budget)
    original = geometry_owner.owned_coordinate_preparation
    observed: list[tuple[int, int, int, int]] = []

    @contextmanager
    def preparation(
        maximum_work_units: int,
        maximum_memory_bytes: int,
        /,
        *,
        maximum_geometry_queries: int = 0,
    ) -> Iterator[CoordinateEnclosureBudget]:
        with original(
            maximum_work_units,
            maximum_memory_bytes - short,
            maximum_geometry_queries=maximum_geometry_queries,
        ) as ledger:
            try:
                yield ledger
            finally:
                observed.append(
                    (
                        maximum_work_units,
                        maximum_memory_bytes,
                        ledger.work_units,
                        ledger.peak_bytes_upper,
                    )
                )

    monkeypatch.setattr(geometry_owner, "owned_coordinate_preparation", preparation)

    def reconstruct() -> ExactAlgebraicCellGeometrySource:
        return ExactAlgebraicCellGeometrySource(
            source.coefficients,
            source.layouts,
            source_identity=source.source_identity,
            maximum_work=source.maximum_work,
            maximum_bits=source.maximum_bits,
        )

    if short:
        with pytest.raises(CoordinateEnclosureResourceError) as refusal:
            reconstruct()
        assert refusal.value.resource == "retained_basis"
        assert refusal.value.requested == observed[-1][1]
        assert refusal.value.limit == observed[-1][1] - 1
    else:
        restored = reconstruct()
        assert restored.source_id == source.source_id
        work_bound, memory_bound, work, peak = observed[-1]
        assert work == work_bound
        assert peak == memory_bound


def test_periodic_membership_preparation_reuses_only_the_same_live_inputs() -> None:
    import phydrax as phx
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.meshing.providers._native_periodic import (
        _PeriodicSourceAssociationPreparation,
    )
    from tests._support.periodic_meshing import periodic_seed_request

    source, specification = periodic_seed_request()
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("periodic_delaunay")
        )
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    transfer = phx.meshing.PeriodicAssociationTransfer(source, specification)
    limits = specification.limits
    with NativeExecutionScope(limits.native_limits()) as scope:
        initial = scope.batch(0)
        budget = CoordinateEnclosureBudget(
            initial,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        initial.settle()
        with budget.retained_storage(lambda amount: scope.batch(0, bytes_upper=amount)):
            preparation: list[_PeriodicSourceAssociationPreparation] = []
            first = transfer.source_associations(
                result, scope=scope, budget=budget, _prepared_source=preparation
            )
            work = budget.work_units
            repeated = transfer.source_associations(
                result, scope=scope, budget=budget, _prepared_source=preparation
            )
            assert repeated[0] is first[0]
            assert repeated[1] == first[1]
            assert budget.work_units == work
            with NativeExecutionScope(limits.native_limits()) as execution:
                ticket = execution.batch(0)
                execution_budget = CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=limits.maximum_work_units,
                    maximum_memory_bytes=limits.maximum_scratch_bytes,
                )
                ticket.settle()
                with execution_budget.retained_storage(
                    lambda amount: execution.batch(0, bytes_upper=amount)
                ):
                    reused = transfer.source_associations(
                        result,
                        scope=execution,
                        budget=execution_budget,
                        _prepared_source=preparation,
                    )
                    assert reused[0] is first[0]
                    assert execution_budget.work_units == 0
                    proof = preparation[0]
                    foreign = _PeriodicSourceAssociationPreparation(
                        proof.transfer_source,
                        proof.specification,
                        proof.result,
                        proof.association,
                        proof.associated_degrees,
                    )
                    transfer.source_associations(
                        result,
                        scope=execution,
                        budget=execution_budget,
                        _prepared_source=[foreign],
                    )
                    assert execution_budget.work_units > 0
            changed = eqx.tree_at(
                lambda item: item.associations[0].source_indices,
                result,
                result.associations[0].source_indices.at[0].add(1),
            )
            with pytest.raises(ValueError, match=r"periodic\.transfer\.associations"):
                transfer.source_associations(
                    changed, scope=scope, budget=budget, _prepared_source=preparation
                )
            assert budget.work_units > work


def test_periodic_membership_receipt_crosses_ended_preparation_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import phydrax as phx
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.meshing import _adaptation as adaptation_owner
    from phydrax.meshing._adaptation import (
        _admit_periodic_source_proofs,
        _periodic_preparation_execution_limits,
    )
    from tests._support.periodic_meshing import periodic_seed_request

    source, specification = periodic_seed_request()
    result = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("periodic_delaunay")
        )
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    transfer = phx.meshing.PeriodicAssociationTransfer(source, specification)
    prepared = phx.meshing.prepare_mesh_adaptation(
        result,
        phx.meshing.MarkedMeshAdaptation(result.mesh.entity_set(2).entity_ids),
        policy=phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
            limits=specification.limits,
            association_transfer=transfer,
        ),
    )
    native_limits, evidence = _periodic_preparation_execution_limits(prepared)
    assert evidence is not None
    assert native_limits.maximum_work_units == (
        specification.limits.maximum_work_units
        - int(np.asarray(evidence.total_work_units))
    )
    assert native_limits.maximum_seconds <= (
        specification.limits.maximum_wall_seconds
        - float(np.asarray(evidence.total_elapsed_seconds))
    )
    receipt = prepared.periodic_source_receipt
    assert receipt is not None
    with monkeypatch.context() as patch:
        patch.setattr(
            adaptation_owner.time, "monotonic", lambda: receipt.expires_at - 0.25
        )
        delayed, _ = _periodic_preparation_execution_limits(prepared)
        assert delayed.maximum_seconds == 0.25
        patch.setattr(
            adaptation_owner.time, "monotonic", lambda: receipt.expires_at + 1.0
        )
        expired, _ = _periodic_preparation_execution_limits(prepared)
        assert expired.maximum_seconds == 0.0
    with NativeExecutionScope(native_limits) as execution:
        _admit_periodic_source_proofs(prepared, execution)
        ticket = execution.batch(0)
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=native_limits.maximum_work_units,
            maximum_memory_bytes=native_limits.maximum_managed_bytes,
        )
        ticket.settle()
        with budget.retained_storage(
            lambda amount: execution.batch(0, bytes_upper=amount)
        ):
            association, _ = transfer.source_associations(
                result,
                scope=execution,
                budget=budget,
                _prepared_source=list(prepared.periodic_source_proofs),
            )
            assert association is prepared.periodic_source_proofs[0].association
            assert budget.work_units == 0
    register_meshing_source_artifacts()
    archive = write_meshing_source_closure(tmp_path / "prepared-periodic", prepared)
    restored = read_meshing_source_closure(
        archive.path, expected_content_id=archive.content_id
    )
    assert isinstance(restored, adaptation_owner.PreparedMeshAdaptation)
    assert tree_content_equal(prepared, restored)
    with NativeExecutionScope(native_limits) as execution:
        _admit_periodic_source_proofs(restored, execution)
        ticket = execution.batch(0)
        restored_budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=native_limits.maximum_work_units,
            maximum_memory_bytes=native_limits.maximum_managed_bytes,
        )
        ticket.settle()
        with restored_budget.retained_storage(
            lambda amount: execution.batch(0, bytes_upper=amount)
        ):
            transfer.source_associations(
                result,
                scope=execution,
                budget=restored_budget,
                _prepared_source=list(restored.periodic_source_proofs),
            )
            assert restored_budget.work_units > 0


@pytest.mark.parametrize(
    ("scale", "vertical_offset"),
    (
        (Fraction(1), Fraction(0)),
        (Fraction(1, 3), Fraction(0)),
        (Fraction(1), Fraction(1, 3)),
    ),
    ids=("unit", "third", "translated"),
)
def test_periodic_algebraic_refinement_binds_child_layouts(
    budget: CoordinateEnclosureBudget,
    scale: Fraction,
    vertical_offset: Fraction,
) -> None:
    import phydrax as phx
    from phydrax.discretization._cell_geometry_transfer import is_affine_cell_geometry

    mesh, geometry, source = _algebraic_strip(
        RadicalElement.sqrt(3) / 2,
        budget,
        scale=scale,
        vertical_offset=vertical_offset,
    )
    result = phx.meshing.certify_cell_mesh(
        mesh, phx.SpatialCoordinateContract.si(), geometry=geometry
    )
    refined = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            result,
            phx.meshing.MarkedMeshAdaptation(mesh.entity_set(2).entity_ids),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )
    assert refined.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    target = refined.target.geometry
    target_source = target.exact_source
    assert isinstance(target_source, ExactAlgebraicCellGeometrySource)
    assert target.restriction_source is not None
    assert target.restriction_source.carrier_publication == "source_image"
    assert target.restriction_source.carrier_geometry is None
    assert target_source.source_identity == source.source_identity
    assert target_source.layouts == rational_source_layouts(
        dict(zip(target.block_names, target.elements, strict=True)),
        dict(zip(target.block_names, target.geometry_dofs, strict=True)),
    )
    assert target_source.coefficients == source.coefficients
    target.resolve(refined.target.mesh, budget=budget)
    assert is_affine_cell_geometry(refined.target.mesh, target, budget=budget)
    target_ids = np.asarray(refined.target.mesh.vertex_global_ids)
    order = np.argsort(target_ids)
    retained = order[
        np.searchsorted(target_ids[order], np.asarray(result.mesh.vertex_global_ids))
    ]
    np.testing.assert_array_equal(
        np.asarray(refined.target.mesh.coordinates)[retained].view(np.uint64),
        np.asarray(result.mesh.coordinates).view(np.uint64),
    )
    generated = np.flatnonzero(
        ~np.isin(target_ids, np.asarray(result.mesh.vertex_global_ids))
    )
    vertex = int(generated[0])
    value = np.asarray(refined.target.mesh.coordinates)[vertex, 0]
    changed = eqx.tree_at(
        lambda item: item.coordinates,
        refined.target.mesh,
        refined.target.mesh.coordinates.at[vertex, 0].set(np.nextafter(value, np.inf)),
    )
    assert not is_affine_cell_geometry(changed, target, budget=budget)


def test_algebraic_source_refuses_mismatched_declared_layout(
    budget: CoordinateEnclosureBudget,
) -> None:
    _, geometry, source = _algebraic_strip(RadicalElement.sqrt(3) / 2, budget)
    with pytest.raises(ValueError, match="declared interpolation layout"):
        CellGeometrySpec(
            {"strip": coordinate_lagrange_element("triangle", 2)},
            {"strip": np.arange(6, dtype=np.int32)[None, :]},
            geometry.coordinates,
            exact_source=source,
            budget=budget,
        )


def test_periodic_row_enclosures_release_scalar_interval_storage(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.meshing.providers._native_periodic import _periodic_row_enclosures

    root = RadicalElement.sqrt(2)
    corners = np.full((64, 3, 2), root, dtype=object)
    before = budget.temporary_bytes_upper
    work = budget.work_units
    lower, upper = _periodic_row_enclosures(corners, budget=budget)
    assert budget.temporary_bytes_upper == before + lower.nbytes + upper.nbytes
    assert budget.work_units > work + corners.size
    assert all(Fraction(float(value)) <= root for value in lower.flat)
    assert all(Fraction(float(value)) >= root for value in upper.flat)


def test_prepared_triangle_sat_matches_per_pair(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.geometry._affine_mapped_contact import (
        prepare_triangle_sat,
        prepared_triangle_interiors_overlap,
    )

    root = RadicalElement.sqrt(3)
    base = np.asarray(
        ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), root)),
        dtype=object,
    )
    triangles = tuple(
        base + np.asarray((x, y), dtype=object)
        for x, y in (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(1, 2), Fraction(0)),
            (Fraction(0), root),
            (Fraction(0), -root / 2),
        )
    ) + (base[::-1], np.zeros((3, 2), dtype=object))
    actions = tuple(prepare_triangle_sat(row, budget=budget) for row in triangles)
    for i, first in enumerate(triangles):
        for j, second in enumerate(triangles):
            assert prepared_triangle_interiors_overlap(
                actions[i], actions[j], budget=budget
            ) == interiors_overlap(first, second, budget=budget)
    before = budget.temporary_bytes_upper
    work = budget.work_units
    prepaid = prepare_triangle_sat(base, budget=budget, storage_precharged=True)
    assert prepared_triangle_interiors_overlap(
        prepaid, prepaid, budget=budget, storage_precharged=True
    )
    assert budget.temporary_bytes_upper == before
    assert budget.work_units > work


def test_mapped_dg_integration_goal_requires_rational_reference_basis(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax.discretization.fem._topology_transfer import (
        _mapped_dg_integration_goal,
        MappedDgCell,
    )

    element = coordinate_lagrange_element("triangle", 1)
    route = np.asarray((0,), dtype=np.int64)
    rational_cells: tuple[MappedDgCell, ...] = (
        (element, route, ({(0, 0): Fraction(1)},)),
    )
    assert (
        _mapped_dg_integration_goal(
            rational_cells, rational_cells, (), True, budget=budget
        )
        > 0
    )
    algebraic_cells: tuple[MappedDgCell, ...] = (
        (element, route, ({(0, 0): RadicalElement.sqrt(2)},)),
    )
    with pytest.raises(ValueError, match="integration_goal.reference_basis"):
        _mapped_dg_integration_goal(
            algebraic_cells, algebraic_cells, (), True, budget=budget
        )


def test_rational_source_accessor_preserves_subfield_and_rejects_radicals(
    budget: CoordinateEnclosureBudget,
) -> None:
    _, geometry, source = _algebraic_strip(RadicalElement(Fraction(1, 2)), budget)
    identity = source.source_id
    bank = geometry.source_coordinates(budget=budget)
    rational = geometry.rational_source_coordinates(budget=budget)
    assert all(isinstance(value, Fraction) for row in rational for value in row)
    assert rational == bank
    selected = geometry.rational_source_coordinates(
        np.asarray((2, 0, 2), dtype=np.int64), budget=budget
    )
    assert selected == (rational[2], rational[0], rational[2])
    assert geometry.exact_source is source
    assert source.source_id == identity
    _, algebraic, _ = _algebraic_strip(RadicalElement.sqrt(3) / 2, budget)
    with pytest.raises(ValueError, match="rational_source_coordinates.coefficients"):
        algebraic.rational_source_coordinates(budget=budget)


@pytest.mark.parametrize("dimension", (2, 3))
def test_periodic_stratum_vertex_factorization_preserves_entity_decisions(
    dimension: int,
) -> None:
    from phydrax.meshing._periodic import source_entity_contains

    root = RadicalElement.sqrt(2)
    points = np.asarray(
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (root, Fraction(0), Fraction(0)),
            (root / 2, Fraction(0), Fraction(0)),
            (-root, Fraction(0), Fraction(0)),
            (root / 2, root / 2, Fraction(0)),
            (root / 2, Fraction(0), root / 2),
        ),
        dtype=object,
    )[:, :dimension]
    rows = np.asarray(((0, 1, 2), (0, 2, 3), (0, 2, 4), (2, 4, 5)), dtype=np.int64)
    sources = (points[:1], points[:2], points[[0, 0]])
    if dimension == 3:
        sources += (points[[0, 1, 4]], points[[0, 1, 2]])
    for source in sources:
        vertices = source_entity_contains(points[:, None, :], source)
        np.testing.assert_array_equal(
            np.all(vertices[rows], axis=1),
            source_entity_contains(points[rows], source),
        )
