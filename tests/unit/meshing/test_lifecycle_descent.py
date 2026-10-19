# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent newest-vertex descent qualification and deliberate faults."""

from __future__ import annotations

import os
import warnings
from fractions import Fraction
from pathlib import Path
from sys import float_info

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import canonical_json
from phydrax._meshcore import meshcore_available
from tools._meshing_cases import (
    corpus_identity,
    meshing_case,
    prepare_case,
    source_payload,
)
from tools.meshing_qualification._descent import (
    _rounded_point,
    certify_descent,
    check_angle_bound,
    similarity_bound,
)
from tools.meshing_qualification._runtime import (
    _runtime_identity,
    QualificationFailure,
    StageLedger,
)
from tools.meshing_qualification.scenarios.native_lifecycle import (
    _corpus_acceptance,
    _corpus_cold_worker,
    _corpus_native_consumer,
    qualify_native_lifecycle,
)


@pytest.fixture
def two_generations() -> tuple[
    phx.discretization.CellMesh,
    phx.discretization.CellMesh,
    phx.meshing.BisectionHierarchy,
]:
    # The refinement edge is the vertical leg, not the hypotenuse. The four
    # similarity classes have minimum angles 45, atan(1/2), atan(1/3), atan(1/2)
    # degrees; the worst squared cosine is 9/10.
    source = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 4, 3), (1, 4, 3), (1, 5, 3), (2, 5, 3)), dtype=np.int32)
    oriented_cells = cells.copy()
    oriented_cells[[1, 3]] = oriented_cells[[1, 3]][:, (0, 2, 1)]
    target = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "triangles",
                "triangle",
                oriented_cells,
                global_ids=np.arange(3, 7, dtype=np.int64),
            ),
        ),
    )
    hierarchy = phx.meshing.BisectionHierarchy(
        2,
        np.arange(3, 7, dtype=np.int64),
        cells,
        np.full(4, 2, dtype=np.int32),
        np.full(4, 2, dtype=np.int32),
        record_parent_ids=np.asarray((0, 1, 2), dtype=np.int64),
        record_parent_blocks=np.zeros(3, dtype=np.int32),
        record_parent_rows=np.asarray(((0, 1, 2), (0, 1, 3), (1, 2, 3)), dtype=np.int64),
        record_parent_vertices=np.asarray(
            ((0, 1, 2), (0, 1, 3), (1, 2, 3)), dtype=np.int64
        ),
        record_parent_tags=np.asarray((2, 1, 1), dtype=np.int32),
        record_child_ids=np.asarray(((1, 2), (3, 4), (5, 6)), dtype=np.int64),
        record_vertex_ids=np.asarray((3, 4, 5), dtype=np.int64),
        retired_entity_keys=(np.empty((0, 2), dtype=np.int64),),
        retired_entity_ids=(np.empty((0,), dtype=np.int64),),
        next_vertex_id=6,
        next_cell_id=7,
        uniform_refinement=None,
        scientific_cell_ids=np.arange(7, dtype=np.int64),
        scientific_block_ids=np.zeros(7, dtype=np.int32),
    )
    return source, target, hierarchy


def test_exact_two_generation_descent_has_hand_derived_similarity_bound(
    two_generations: tuple[
        phx.discretization.CellMesh,
        phx.discretization.CellMesh,
        phx.meshing.BisectionHierarchy,
    ],
) -> None:
    source, target, hierarchy = two_generations
    evidence = certify_descent(source, target, hierarchy)
    assert evidence["status"] == "certified"
    assert evidence["bound"] == {
        "cosine_sign": 1,
        "squared_cosine": "9/10",
        "radians": pytest.approx(np.arctan(1 / 3)),
    }
    triangle = tuple(
        tuple(Fraction(float(value)) for value in point)
        for point in np.asarray(source.coordinates)
    )
    bound, count, _ = similarity_bound(triangle, 2)
    assert bound == (1, Fraction(9, 10))
    assert count == 4
    assert evidence["generations"] == [
        {"cell": identity, "ancestor": 0, "generation": 2} for identity in range(3, 7)
    ]


def test_perturbed_target_vertex_refuses_exact_nesting(
    two_generations: tuple[
        phx.discretization.CellMesh,
        phx.discretization.CellMesh,
        phx.meshing.BisectionHierarchy,
    ],
) -> None:
    source, target, hierarchy = two_generations
    points = np.asarray(target.coordinates).copy()
    points[4, 0] = np.nextafter(points[4, 0], 1.0)
    changed = phx.discretization.CellMesh(points, target.blocks)
    with pytest.raises(QualificationFailure, match="nested_vertex"):
        certify_descent(source, changed, hierarchy)


@pytest.mark.parametrize("signed_zero", (False, True), ids=("one-ulp", "signed-zero"))
def test_retained_epoch_zero_vertex_must_be_bitwise_preserved(
    two_generations: tuple[
        phx.discretization.CellMesh,
        phx.discretization.CellMesh,
        phx.meshing.BisectionHierarchy,
    ],
    signed_zero: bool,
) -> None:
    source, target, hierarchy = two_generations
    points = np.asarray(target.coordinates).copy()
    if signed_zero:
        points[0, 0] = -0.0
    else:
        points[1, 0] = np.nextafter(points[1, 0], 2.0)
    target = phx.discretization.CellMesh(points, target.blocks)
    with pytest.raises(QualificationFailure, match="epoch-zero-vertex"):
        certify_descent(source, target, hierarchy)


def test_different_edge_split_refuses_newest_vertex_descent(
    two_generations: tuple[
        phx.discretization.CellMesh,
        phx.discretization.CellMesh,
        phx.meshing.BisectionHierarchy,
    ],
) -> None:
    source, target, hierarchy = two_generations
    # This record attempts to switch the refinement edge of an intermediate
    # triangle while retaining its ancestor label and the published target.
    changed = eqx.tree_at(
        lambda h: h.record_parent_tags, hierarchy, jnp.asarray((2, 2, 1), dtype=jnp.int32)
    )
    with pytest.raises(QualificationFailure, match="newest_vertex_edge"):
        certify_descent(source, target, changed)


def test_missing_ancestor_refinement_edge_refuses_without_guessing(
    two_generations: tuple[
        phx.discretization.CellMesh,
        phx.discretization.CellMesh,
        phx.meshing.BisectionHierarchy,
    ],
) -> None:
    source, target, hierarchy = two_generations
    changed = eqx.tree_at(
        lambda h: h.record_parent_ids,
        hierarchy,
        jnp.asarray((99, 1, 2), dtype=jnp.int64),
    )
    with pytest.raises(QualificationFailure, match="refinement_edge"):
        certify_descent(source, target, changed)


def test_synthetic_angle_below_similarity_bound_refuses_exactly() -> None:
    triangle = tuple(
        tuple(Fraction(value) for value in point) for point in ((0, 0), (1, 0), (0, 1))
    )
    bound, _, _ = similarity_bound(triangle, 2)
    below = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1, 4)),
    )
    with pytest.raises(QualificationFailure, match="minimum_angle"):
        check_angle_bound(below, bound)


@pytest.mark.parametrize(
    ("value", "field"),
    (
        (Fraction(float_info.min) / 2, "realization_subnormal"),
        (Fraction(float_info.max) * 2, "realization_overflow"),
    ),
    ids=("subnormal", "overflow"),
)
def test_unsupported_ieee_realization_regimes_refuse(value: Fraction, field: str) -> None:
    with pytest.raises(QualificationFailure, match=field):
        _rounded_point((value, Fraction(1)))


@pytest.mark.parametrize(
    "one_ulp_off", (False, True), ids=("nearest-even", "one-ulp-off")
)
def test_half_ulp_midpoint_realization_is_certified_without_tolerance(
    two_generations: tuple[
        phx.discretization.CellMesh,
        phx.discretization.CellMesh,
        phx.meshing.BisectionHierarchy,
    ],
    one_ulp_off: bool,
) -> None:
    source, target, hierarchy = two_generations
    high = np.nextafter(1.0, 0.0)
    source_points = np.asarray(((0.0, 0.5), (1.0, 0.5), (0.0, high)), dtype=np.float64)
    ideal_midpoint = (Fraction(0.5) + Fraction(float(high))) / 2
    rounded_midpoint = float(ideal_midpoint)
    assert Fraction(rounded_midpoint) != ideal_midpoint
    assert (
        abs(Fraction(rounded_midpoint) - ideal_midpoint)
        == Fraction(float(np.spacing(rounded_midpoint))) / 2
    )
    points = np.asarray(target.coordinates).copy()
    points[:3] = source_points
    points[3] = (0.0, rounded_midpoint)
    points[4] = (0.5, 0.5)
    points[5] = (0.5, rounded_midpoint)
    if one_ulp_off:
        points[3, 1] = np.nextafter(points[3, 1], 1.0)
    source = phx.discretization.CellMesh(source_points, source.blocks)
    target = phx.discretization.CellMesh(points, target.blocks)
    if one_ulp_off:
        with pytest.raises(QualificationFailure, match="nested_vertex"):
            certify_descent(source, target, hierarchy)
    else:
        evidence = certify_descent(source, target, hierarchy)
        assert evidence["status"] == "certified"
        assert evidence["realization"]
        assert evidence["published_angle_perturbations"]


@pytest.mark.parametrize(
    "round_endpoints", (False, True), ids=("ideal-once", "rounded-endpoints")
)
def test_three_generation_realization_rounds_ideal_vertices_once(
    two_generations: tuple[
        phx.discretization.CellMesh,
        phx.discretization.CellMesh,
        phx.meshing.BisectionHierarchy,
    ],
    round_endpoints: bool,
) -> None:
    source, _, previous = two_generations
    low = 0.5
    high = float(np.nextafter(1.0, 0.0))
    other = low + 3 * float(np.spacing(low))
    root = np.asarray(((0.0, low), (1.0, other), (0.0, high)), dtype=np.float64)
    ideal_first = (Fraction(low) + Fraction(high)) / 2
    ideal_third = (Fraction(other) + ideal_first) / 2
    endpoint_third = (Fraction(other) + Fraction(float(ideal_first))) / 2
    assert float(ideal_third) != float(endpoint_third)
    points = np.concatenate(
        (
            root,
            np.asarray(
                (
                    (0.0, float(ideal_first)),
                    (0.5, float((Fraction(low) + Fraction(other)) / 2)),
                    (0.5, float((Fraction(other) + Fraction(high)) / 2)),
                    (0.5, float(endpoint_third if round_endpoints else ideal_third)),
                ),
                dtype=np.float64,
            ),
        )
    )
    labels = np.asarray(
        ((0, 4, 3), (2, 5, 3), (1, 4, 6), (3, 4, 6), (1, 5, 6), (3, 5, 6)),
        dtype=np.int64,
    )
    cells = labels.copy()
    cells[[1, 2, 5]] = cells[[1, 2, 5]][:, (0, 2, 1)]
    identities = np.asarray((3, 6, 7, 8, 9, 10), dtype=np.int64)
    source = phx.discretization.CellMesh(root, source.blocks)
    target = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "triangles", "triangle", cells, global_ids=identities
            ),
        ),
    )
    hierarchy = phx.meshing.BisectionHierarchy(
        2,
        identities,
        labels,
        np.asarray((2, 2, 1, 1, 1, 1), dtype=np.int32),
        np.asarray((2, 2, 3, 3, 3, 3), dtype=np.int32),
        record_parent_ids=np.asarray((0, 1, 2, 4, 5), dtype=np.int64),
        record_parent_blocks=np.zeros(5, dtype=np.int32),
        record_parent_rows=np.asarray(
            ((0, 1, 2), (0, 1, 3), (1, 2, 3), (1, 4, 3), (1, 5, 3)), dtype=np.int64
        ),
        record_parent_vertices=np.asarray(
            ((0, 1, 2), (0, 1, 3), (1, 2, 3), (1, 4, 3), (1, 5, 3)), dtype=np.int64
        ),
        record_parent_tags=np.asarray((2, 1, 1, 2, 2), dtype=np.int32),
        record_child_ids=np.asarray(
            ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)), dtype=np.int64
        ),
        record_vertex_ids=np.asarray((3, 4, 5, 6, 6), dtype=np.int64),
        retired_entity_keys=previous.retired_entity_keys,
        retired_entity_ids=previous.retired_entity_ids,
        next_vertex_id=7,
        next_cell_id=11,
        uniform_refinement=None,
        scientific_cell_ids=np.arange(11, dtype=np.int64),
        scientific_block_ids=np.zeros(11, dtype=np.int32),
    )
    if round_endpoints:
        with pytest.raises(QualificationFailure, match="rounded-midpoint"):
            certify_descent(source, target, hierarchy)
    else:
        evidence = certify_descent(source, target, hierarchy)
        assert evidence["status"] == "certified"


@pytest.mark.parametrize("height", (1.0, 0.125), ids=("median-band", "angle-floor"))
def test_epoch_zero_keeps_generator_size_band_and_angle_floor(height: float) -> None:
    case = meshing_case("planar-hole-feature")
    _, specification, _ = prepare_case(case, resolution=2, capacity=20000, timeout=120)
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, height)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    result = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    acceptance = _corpus_acceptance(case, result, specification)
    assert not acceptance["passed"]
    assert not acceptance["size_passed"]
    if height == 0.125:
        assert not acceptance["quality_passed"]
    else:
        assert acceptance["quality_passed"]
    assert acceptance["descent"] is None


def test_tetrahedral_descent_closes_maubach_dihedral_classes() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 0.5),
        ),
        dtype=np.float64,
    )
    source = phx.discretization.CellMesh.from_tetrahedra(
        points[:4],
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
    )
    target = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "tetrahedra",
                "tetrahedron",
                np.asarray(((0, 1, 2, 4), (1, 3, 2, 4)), dtype=np.int32),
                global_ids=np.asarray((1, 2), dtype=np.int64),
            ),
        ),
    )
    hierarchy = phx.meshing.BisectionHierarchy(
        3,
        np.asarray((1, 2), dtype=np.int64),
        np.asarray(((0, 1, 2, 4), (1, 2, 3, 4)), dtype=np.int64),
        np.asarray((2, 2), dtype=np.int32),
        np.asarray((1, 1), dtype=np.int32),
        record_parent_ids=np.asarray((0,), dtype=np.int64),
        record_parent_blocks=np.asarray((0,), dtype=np.int32),
        record_parent_rows=np.asarray(((0, 1, 2, 3),), dtype=np.int64),
        record_parent_vertices=np.asarray(((0, 1, 2, 3),), dtype=np.int64),
        record_parent_tags=np.asarray((3,), dtype=np.int32),
        record_child_ids=np.asarray(((1, 2),), dtype=np.int64),
        record_vertex_ids=np.asarray((4,), dtype=np.int64),
        retired_entity_keys=(
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
        ),
        retired_entity_ids=(
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        ),
        next_vertex_id=5,
        next_cell_id=3,
        uniform_refinement=None,
        scientific_cell_ids=np.asarray((0, 1, 2), dtype=np.int64),
        scientific_block_ids=np.zeros(3, dtype=np.int32),
    )
    evidence = certify_descent(source, target, hierarchy)
    assert evidence["status"] == "certified"
    assert evidence["shape_class_cap"] == 192
    assert evidence["generations"] == [
        {"cell": 1, "ancestor": 0, "generation": 1},
        {"cell": 2, "ancestor": 0, "generation": 1},
    ]


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
def test_native_planar_adaptation_preserves_accepted_coordinate_descent() -> None:
    case = meshing_case("planar-hole-feature")
    source, specification, options = prepare_case(
        case, resolution=2, capacity=20000, timeout=120
    )
    assert isinstance(source, phx.meshing.NativePlanarSource)
    initial = (
        phx.meshing.NativeMeshingProvider(options)
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    initial, initial_acceptance = _corpus_native_consumer(
        initial,
        specification,
        lambda: _corpus_acceptance(case, initial, specification),
        native=True,
    )
    assert initial_acceptance["passed"], initial_acceptance
    assert initial.certification is not None
    domain = initial.certification.request.domain
    assert isinstance(domain, phx.geometry.PiecewiseLinearDomain)
    payload = source_payload(case)
    source_points = np.asarray(payload["vertices"], dtype=np.float64)
    embedded = np.asarray(payload["embedded"], dtype=np.float64)
    edges = [
        (loop[index], loop[(index + 1) % len(loop)])
        for loop in payload["loops"]
        for index in range(len(loop))
    ]
    edges.append((source_points.shape[0], source_points.shape[0] + 1))
    association = phx.meshing.PlcAssociationTransfer(
        domain,
        initial.coordinate_contract,
        source.source_revision,
        source_vertices=np.concatenate((source_points, embedded)),
        edge_vertices=np.asarray(edges, dtype=np.int64),
        triangle_vertices=np.empty((0, 3), dtype=np.int64),
        triangle_facets=np.empty((0,), dtype=np.int64),
        facet_regions=domain.facet_regions,
        maximum_support_queries=specification.limits.maximum_geometry_queries,
    )
    identities = np.concatenate(
        [np.asarray(block.global_ids) for block in initial.mesh.blocks]
    )
    request = phx.meshing.MarkedMeshAdaptation(identities, np.empty((0,), dtype=np.int64))
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=phx.meshing.BisectionCompatibility.CONFORMING_CLOSURE,
        limits=specification.limits,
        association_transfer=association,
        audit_policy=phx.meshing.CellMeshAuditPolicy(
            require_complete_association=True,
            watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
        ),
    )
    target, adaptation = _corpus_native_consumer(
        initial,
        specification,
        lambda: phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(initial, request, policy=policy),
        ),
        native=True,
        target=lambda value: value.target,
    )
    hierarchy = adaptation.hierarchy
    assert isinstance(hierarchy, phx.meshing.BisectionHierarchy)
    evidence = certify_descent(initial.mesh, target.mesh, hierarchy)
    assert evidence["status"] == "certified"
    target, acceptance = _corpus_native_consumer(
        target,
        specification,
        lambda: _corpus_acceptance(
            case,
            target,
            specification,
            epoch_zero=initial.mesh,
            hierarchy=hierarchy,
        ),
        native=True,
    )
    assert acceptance["passed"], acceptance


def test_cold_packet_refuses_missing_ordered_roles(tmp_path: Path) -> None:
    receipt = phx.lifecycle.write_meshing_source_closure(
        tmp_path / "incomplete-packet.zip",
        ("incomplete",),
    )
    with pytest.raises(QualificationFailure, match="seven ordered"):
        _corpus_cold_worker(str(receipt.path), receipt.content_id, (), 1, 0.05)


def test_compact_cold_fingerprint_retains_complete_array_signature() -> None:
    scalar = phx.array_tree_fingerprint(np.asarray(1.0, dtype=np.float64))
    vector = phx.array_tree_fingerprint(np.asarray((1.0,), dtype=np.float64))
    # The array digest preserves the legacy scalar byte framing. The fingerprint
    # also records rank, which the compact wire digest must retain.
    assert scalar["sha256"] == vector["sha256"]
    assert scalar["signature"] != vector["signature"]
    assert phx.canonical_fingerprint(scalar) != phx.canonical_fingerprint(vector)


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
def test_planar_lifecycle_restores_and_recertifies_every_accepted_epoch() -> None:
    record = qualify_native_lifecycle(
        "planar-hole-feature",
        resolution=2,
        capacity=20000,
        timeout=120,
        repeats=1,
        target_error=0.05,
        adaptation_rounds=1,
    )
    assert record["status"] == "passed", record
    assert record["case_contract_completion"] == "complete"
    checkpoint = record["checkpoint"]
    assert isinstance(checkpoint, dict)
    assert checkpoint["passed"] is True
    assert checkpoint["process_id"] != os.getpid()
    assert checkpoint["accepted_fields_bitwise_preserved"] is True
    assert checkpoint["prepared_fe_caches_serialized"] is False
    epochs = checkpoint["accepted_epochs"]
    assert isinstance(epochs, list)
    assert len(epochs) == 2
    assert epochs[0]["acceptance"]["passed"] is True
    assert epochs[0]["acceptance"]["descent"] is None
    assert epochs[1]["acceptance"]["passed"] is True
    assert epochs[1]["acceptance"]["descent"]["status"] == "certified"
    assert (
        epochs[1]["acceptance"]["descent"]["retained_vertices_bitwise_preserved"] is True
    )
    assert checkpoint["continued_physical_error"] <= 0.05


def test_corpus_host_metadata_constructs_static_ledger_without_array_warning() -> None:
    case = meshing_case("planar-hole-feature")
    case_record = case.to_record()
    before = {**case_record, "minimum_angle_radians": np.radians(25.0)}
    assert canonical_json(case_record) == canonical_json(before)
    assert type(case_record["minimum_angle_radians"]) is float
    record = {"case": corpus_identity(case), "runtime": _runtime_identity()}
    assert not any(
        isinstance(leaf, np.generic) for leaf in jax.tree_util.tree_leaves(record)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ledger = StageLedger(record)
    assert ledger.record is record
