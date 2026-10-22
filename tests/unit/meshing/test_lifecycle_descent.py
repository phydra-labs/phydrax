# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent newest-vertex descent qualification and deliberate faults."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from sys import float_info

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
import tools.meshing_qualification._descent as descent_oracle
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
    _corpus_native_request,
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
        refinement_rule="maubach",
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
        refinement_rule="maubach",
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


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore unavailable")
def test_fixed_boundary_descent_keeps_soft_size_soft_and_hard_size_enforced() -> None:
    case = meshing_case("plc-fixed-boundary")
    source, specification, options = prepare_case(case, 4, 20000, 120.0)
    result = phx.meshing.NativeMeshingProvider(options).plan(
        source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
    ).execute()
    mesh = result.mesh
    ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    blocks = np.concatenate(
        [np.full(block.cell_count, index) for index, block in enumerate(mesh.blocks)]
    )
    order = np.argsort(ids)
    vertex_ids = np.asarray(mesh.vertex_global_ids)
    hierarchy = phx.meshing.BisectionHierarchy(
        3,
        ids[order],
        np.sort(vertex_ids[cells[order]], axis=1),
        np.full(len(ids), 3, dtype=np.int32),
        np.zeros(len(ids), dtype=np.int32),
        record_parent_ids=np.empty(0, dtype=np.int64),
        record_parent_blocks=np.empty(0, dtype=np.int32),
        record_parent_rows=np.empty((0, 4), dtype=np.int64),
        record_parent_vertices=np.empty((0, 4), dtype=np.int64),
        record_parent_tags=np.empty(0, dtype=np.int32),
        record_child_ids=np.empty((0, 2), dtype=np.int64),
        record_vertex_ids=np.empty(0, dtype=np.int64),
        retired_entity_keys=(
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
        ),
        retired_entity_ids=(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        ),
        next_vertex_id=int(vertex_ids.max()) + 1,
        next_cell_id=int(ids.max()) + 1,
        uniform_refinement=None,
        scientific_cell_ids=ids[order],
        scientific_block_ids=blocks[order],
        refinement_rule="maubach",
    )
    # Identity descent isolates the admission rule without changing even one
    # original boundary triangle. The independent descent/geometry checks run.
    soft = _corpus_acceptance(
        case, result, specification, epoch_zero=mesh, hierarchy=hierarchy
    )
    assert soft["passed"]
    assert soft["fixed_boundary_passed"]
    soft_descent = soft["descent"]
    assert isinstance(soft_descent, dict) and soft_descent["status"] == "certified"
    assert soft["size_control_strength"] == "soft"
    assert soft["size_passed"]
    assert not soft["maximum_size_satisfied"]
    assert soft["maximum_edge"] >= np.sqrt(2.0)
    hard_case = replace(case, controls=("fixed-boundary", "hard-size"))
    _, hard_specification, _ = prepare_case(hard_case, 4, 20000, 120.0)
    hard = _corpus_acceptance(
        hard_case, result, hard_specification, epoch_zero=mesh, hierarchy=hierarchy
    )
    assert hard["fixed_boundary_passed"]
    hard_descent = hard["descent"]
    assert isinstance(hard_descent, dict) and hard_descent["status"] == "certified"
    assert hard["size_control_strength"] == "hard"
    assert not hard["maximum_size_satisfied"]
    assert not hard["size_passed"]
    assert not hard["passed"]


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
        refinement_rule="maubach",
    )
    evidence = certify_descent(source, target, hierarchy)
    assert evidence["status"] == "certified"
    assert evidence["shape_class_cap"] == 192
    assert evidence["generations"] == [
        {"cell": 1, "ancestor": 0, "generation": 1},
        {"cell": 2, "ancestor": 0, "generation": 1},
    ]


def _whole_star_forest(
    edges: tuple[tuple[int, int], ...], *, initial_tag: int = 3
) -> tuple[
    phx.discretization.CellMesh,
    phx.discretization.CellMesh,
    phx.meshing.BisectionHierarchy,
]:
    # Original fixed-P2 prepared source cell 3: global corners (0, 2, 3, 8).
    # Generator-only extraction with the original fixed-boundary preparation
    # option gave vertex 8=(3/4,1/4,1/4). No adapted target feeds this fixture.
    points = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
              (1.0, 1.0, 0.0), (0.75, 0.25, 0.25)]

    def oriented(vertices: tuple[int, ...]) -> tuple[int, ...]:
        corners = np.asarray([points[vertex] for vertex in vertices])
        if np.linalg.det(corners[1:] - corners[0]) < 0:
            return vertices[0], vertices[2], vertices[1], vertices[3]
        return vertices

    source = phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(points), np.asarray((oriented((0, 1, 2, 3)),), dtype=np.int32)
    )
    nodes = {0: ((0, 1, 2, 3), initial_tag, 0)}
    # Independent literal binary subdivisions, not the producer's templates
    # or the qualification oracle's child-construction helper.
    records: list[tuple[int, tuple[int, ...], tuple[int, ...], int, int, int]] = []
    next_cell = 1
    for first, last in edges:
        midpoint = len(points)
        points.append(tuple((a + b) / 2 for a, b in zip(points[first], points[last])))
        selected = [
            identity for identity, (vertices, _, _) in nodes.items()
            if first in vertices and last in vertices
        ]
        assert selected
        for identity in selected:
            vertices, _, generation = nodes.pop(identity)
            parent = (first, *(v for v in vertices if v not in (first, last)), last)
            parent = min(parent, parent[::-1])
            one, two = next_cell, next_cell + 1
            next_cell += 2
            records.append((identity, oriented(vertices), parent, one, two, midpoint))
            for child, corners in (
                (one, (*parent[:3], midpoint)),
                (two, (*parent[1:], midpoint)),
            ):
                canonical = min(corners, (*corners[:3][::-1], corners[3]))
                nodes[child] = canonical, 2, generation + 1
    records.sort()
    active = sorted(nodes)
    target = phx.discretization.CellMesh(
        np.asarray(points),
        (phx.discretization.CellBlock(
            "tetrahedra", "tetrahedron",
            np.asarray([oriented(nodes[identity][0]) for identity in active], dtype=np.int32),
            global_ids=np.asarray(active, dtype=np.int64),
        ),),
    )
    hierarchy = phx.meshing.BisectionHierarchy(
        3,
        np.asarray(active, dtype=np.int64),
        np.asarray([nodes[identity][0] for identity in active], dtype=np.int64),
        np.asarray([nodes[identity][1] for identity in active], dtype=np.int32),
        np.asarray([nodes[identity][2] for identity in active], dtype=np.int32),
        record_parent_ids=np.asarray([row[0] for row in records], dtype=np.int64),
        record_parent_blocks=np.zeros(len(records), dtype=np.int32),
        record_parent_rows=np.asarray([row[1] for row in records], dtype=np.int64).reshape(-1, 4),
        record_parent_vertices=np.asarray([row[2] for row in records], dtype=np.int64).reshape(-1, 4),
        record_parent_tags=np.full(len(records), 3, dtype=np.int32),
        record_child_ids=np.asarray([row[3:5] for row in records], dtype=np.int64).reshape(-1, 2),
        record_vertex_ids=np.asarray([row[5] for row in records], dtype=np.int64),
        retired_entity_keys=(np.empty((0, 2), dtype=np.int64), np.empty((0, 3), dtype=np.int64)),
        retired_entity_ids=(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)),
        next_vertex_id=len(points),
        next_cell_id=next_cell,
        uniform_refinement=None,
        scientific_cell_ids=np.arange(next_cell, dtype=np.int64),
        scientific_block_ids=np.zeros(next_cell, dtype=np.int32),
        refinement_rule="whole_edge_star",
    )
    return source, target, hierarchy


def test_fixed_boundary_announced_edge_closure_proves_exact_regression_shape() -> None:
    source, target, hierarchy = _whole_star_forest(((0, 3), (2, 3)))
    root = tuple(tuple(Fraction(float(x)) for x in p) for p in np.asarray(source.coordinates))
    root_bound, _, _ = similarity_bound(root, 3)
    assert root_bound == (1, Fraction(49, 50))
    # Second split's first child: global (3,2,m(0,8),m(3,8)).
    child = (
        (Fraction(1), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(3, 8), Fraction(1, 8), Fraction(1, 8)),
        (Fraction(7, 8), Fraction(5, 8), Fraction(1, 8)),
    )
    score = (1, Fraction(5041, 5100))
    assert check_angle_bound(child, score) == score
    with pytest.raises(QualificationFailure, match="minimum_angle"):
        check_angle_bound(child, root_bound)
    evidence = certify_descent(source, target, hierarchy)
    assert evidence["quality_floor_rule"] == (
        "finite-forest-announced-edge-tagged-reference-floor"
    )
    assert evidence["shape_class_cap"] is None
    splits = evidence["split_bounds"]
    assert isinstance(splits, list) and len(splits) == 2
    assert {split["edge"] for split in splits} == {(0, 3), (2, 3)}
    assert all(split["child_tag"] == 2 for split in splits)
    realization = evidence["realization"]
    assert isinstance(realization, dict) and realization["status"] == "certified"


@pytest.mark.parametrize("edge", tuple(combinations(range(4), 2)))
@pytest.mark.parametrize("initial_tag", (1, 2, 3))
def test_whole_star_arbitrary_edges_and_inherited_labels(
    edge: tuple[int, int], initial_tag: int
) -> None:
    source, target, hierarchy = _whole_star_forest((edge,), initial_tag=initial_tag)
    assert certify_descent(source, target, hierarchy)["status"] == "certified"
    source, target, hierarchy = _whole_star_forest((), initial_tag=initial_tag)
    assert certify_descent(source, target, hierarchy)["status"] == "certified"


def test_whole_star_multiple_generations_renew_only_announced_parent_bounds() -> None:
    source, target, hierarchy = _whole_star_forest(((0, 3), (2, 3), (1, 4), (4, 5)))
    evidence = certify_descent(source, target, hierarchy)
    assert evidence["status"] == "certified"
    assert max(np.asarray(hierarchy.generations)) >= 3
    splits = evidence["split_bounds"]
    assert isinstance(splits, list)
    assert {split["parent"] for split in splits} == set(np.asarray(hierarchy.record_parent_ids))
    assert all(split["tag"] == 3 and split["child_tag"] == 2 for split in splits)


@pytest.mark.parametrize(
    ("field", "row", "column", "value", "failure"),
    (
        ("record_parent_tags", 1, None, 2, "whole_edge_star_parent"),
        ("record_parent_vertices", 1, 0, 0, "whole_edge_star_parent"),
        ("record_parent_rows", 1, 0, 0, "whole_edge_star_parent"),
        ("record_vertex_ids", 1, None, 4, "midpoint_edge"),
        ("record_child_ids", 1, 0, 1, "forest"),
        ("record_child_ids", 1, 0, 99, "forest_leaves"),
        ("tags", 0, None, 3, "target_label"),
        ("ordered_vertices", 0, 0, 3, "target_label"),
        ("generations", 0, None, 2, "target_label"),
    ),
)
def test_whole_star_forged_provenance_is_refused(
    field: str, row: int, column: int | None, value: int, failure: str
) -> None:
    source, target, hierarchy = _whole_star_forest(((0, 3), (2, 3)))
    array = np.asarray(getattr(hierarchy, field)).copy()
    array[row if column is None else (row, column)] = value
    forged = eqx.tree_at(lambda h: getattr(h, field), hierarchy, jnp.asarray(array))
    with pytest.raises(QualificationFailure, match=failure):
        certify_descent(source, target, forged)


def test_whole_star_changed_announced_edge_cannot_reuse_published_children() -> None:
    source, target, hierarchy = _whole_star_forest(((0, 3), (2, 3)))
    parents = np.asarray(hierarchy.record_parent_vertices).copy()
    parents[1, [0, 1]] = parents[1, [1, 0]]
    forged = eqx.tree_at(lambda h: h.record_parent_vertices, hierarchy, jnp.asarray(parents))
    with pytest.raises(QualificationFailure, match="nested_vertex|target_label"):
        certify_descent(source, target, forged)


def test_whole_star_exact_measure_fault_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    source, target, hierarchy = _whole_star_forest(((0, 3), (2, 3)))
    original = descent_oracle._measure
    bad_corners = frozenset((
        (Fraction(1), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(3, 8), Fraction(1, 8), Fraction(1, 8)),
        (Fraction(7, 8), Fraction(5, 8), Fraction(1, 8)),
    ))

    def forged(simplex: descent_oracle.Simplex) -> Fraction:
        measure = original(simplex)
        return 2 * measure if frozenset(simplex) == bad_corners else measure

    monkeypatch.setattr(descent_oracle, "_measure", forged)
    with pytest.raises(QualificationFailure, match="halved_measure"):
        certify_descent(source, target, hierarchy)


def test_whole_star_missing_partition_cannot_be_hidden_by_matching_leaf_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, hierarchy = _whole_star_forest(((0, 3), (2, 3)))
    # Simulate a faulty replay dropping a valid leaf while the publication and
    # active rows collude. The independent ancestor measure sum must still fail.
    original = descent_oracle._replay_forest
    removed = int(np.asarray(hierarchy.cell_global_ids)[0])
    block = target.blocks[0]
    target = phx.discretization.CellMesh(
        target.coordinates,
        (phx.discretization.CellBlock(
            block.name, "tetrahedron", np.asarray(block.vertices)[1:],
            global_ids=np.asarray(block.global_ids)[1:],
        ),),
    )
    hierarchy = eqx.tree_at(
        lambda h: (h.cell_global_ids, h.ordered_vertices, h.tags, h.generations),
        hierarchy,
        tuple(value[1:] for value in (
            hierarchy.cell_global_ids, hierarchy.ordered_vertices,
            hierarchy.tags, hierarchy.generations,
        )),
    )

    def forged(
        pending: list[tuple[int, descent_oracle._Node]],
        bank: descent_oracle._Realization,
        forest: phx.meshing.BisectionHierarchy,
        records: dict[int, int],
        references: descent_oracle._References,
    ) -> tuple[dict[int, descent_oracle._Node], set[int]]:
        leaves, visited = original(pending, bank, forest, records, references)
        del leaves[removed]
        return leaves, visited

    monkeypatch.setattr(descent_oracle, "_replay_forest", forged)
    with pytest.raises(QualificationFailure, match="ancestor_partition"):
        certify_descent(source, target, hierarchy)


def test_whole_star_deliberately_reused_root_bound_fails_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target, hierarchy = _whole_star_forest(((0, 3), (2, 3)))

    def stale_floor(
        simplex: descent_oracle.Simplex, tag: int
    ) -> tuple[descent_oracle.AngleScore, int, int]:
        return (1, Fraction(49, 50)), 192, 1536

    monkeypatch.setattr(descent_oracle, "similarity_bound", stale_floor)
    with pytest.raises(QualificationFailure, match="minimum_angle"):
        certify_descent(source, target, hierarchy)


@pytest.fixture(scope="module")
def uniform_descent() -> tuple[
    phx.discretization.CellMesh,
    phx.discretization.CellMesh,
    phx.meshing.BisectionHierarchy,
]:
    # Incompatible source labels force the barycentric-subdivision prefix
    # (six flag children per triangle, 1/3 centroids) before tagged bisection.
    points = np.asarray([(0.0, 0.0), (2.0, 0.0), (1.0, 0.5), (1.0, -3.0)])
    mesh = phx.discretization.CellMesh.from_triangles(
        points, np.asarray([(0, 1, 2), (0, 3, 1)])
    )
    source = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    result = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(
                np.asarray((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
                compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
            ),
        )
    )
    assert isinstance(result.hierarchy, phx.meshing.BisectionHierarchy)
    assert result.hierarchy.uniform_refinement is not None
    return source.mesh, result.target.mesh, result.hierarchy


def _with_uniform(
    hierarchy: phx.meshing.BisectionHierarchy, name: str, value: np.ndarray
) -> phx.meshing.BisectionHierarchy:
    return eqx.tree_at(
        lambda item: getattr(item.uniform_refinement, name),
        hierarchy,
        jnp.asarray(value),
    )


def test_uniform_prefix_descent_replays_flags_then_tagged_forest(
    uniform_descent: tuple[
        phx.discretization.CellMesh,
        phx.discretization.CellMesh,
        phx.meshing.BisectionHierarchy,
    ],
) -> None:
    source, target, hierarchy = uniform_descent
    evidence = certify_descent(source, target, hierarchy)
    assert evidence["status"] == "certified"
    assert evidence["uniform_prefix"] == {
        "status": "certified",
        "rule": "barycentric-subdivision-flags",
        "parents": 2,
        "children_per_parent": 6,
        "centroid_vertices": 7,
    }


def _uniform_fault(
    arrays: dict[str, np.ndarray], fault: str
) -> tuple[str, np.ndarray]:
    match fault:
        case "duplicated-child":
            children = arrays["child_ids"].copy()
            children[0, 1] = children[0, 0]
            return "child_ids", children
        case "missing-flag":
            vertices = arrays["child_vertices"].copy()
            vertices[0, 1] = vertices[0, 0]
            return "child_vertices", vertices
        case "wrong-parent":
            rows = arrays["parent_rows"].copy()
            rows[[0, 1]] = rows[[1, 0]]
            return "parent_rows", rows
        case "permuted-child":
            vertices = arrays["child_vertices"].copy()
            vertices[0, 0, [1, 2]] = vertices[0, 0, [2, 1]]
            return "child_vertices", vertices
        case "wrong-centroid":
            weights = arrays["barycentric_weights"].copy()
            row = np.flatnonzero(np.any(weights[0, 0] == 1.0 / 3.0, axis=1))[0]
            weights[0, 0, row] = np.where(weights[0, 0, row] > 0.0, 0.3, 0.0)
            return "barycentric_weights", weights
    raise AssertionError(fault)


@pytest.mark.parametrize(
    ("fault", "field"),
    (
        ("duplicated-child", "uniform_child"),
        ("missing-flag", "uniform_"),
        ("wrong-parent", "uniform_parent"),
        ("permuted-child", "uniform_"),
        ("wrong-centroid", "uniform_centroid_weight"),
    ),
)
def test_uniform_prefix_faults_refuse_before_tagged_forest(
    uniform_descent: tuple[
        phx.discretization.CellMesh,
        phx.discretization.CellMesh,
        phx.meshing.BisectionHierarchy,
    ],
    fault: str,
    field: str,
) -> None:
    source, target, hierarchy = uniform_descent
    uniform = hierarchy.uniform_refinement
    assert uniform is not None
    name, value = _uniform_fault(uniform.host_arrays(), fault)
    with pytest.raises(QualificationFailure, match=f"descent.{field}"):
        certify_descent(source, target, _with_uniform(hierarchy, name, value))


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
        lambda budget: _corpus_acceptance(
            case, initial, specification, budget=budget
        ),
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
    target, adaptation = _corpus_native_request(
        initial,
        specification,
        lambda _budget: phx.meshing.execute_mesh_adaptation(
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
        lambda budget: _corpus_acceptance(
            case,
            target,
            specification,
            epoch_zero=initial.mesh,
            hierarchy=hierarchy,
            budget=budget,
        ),
        native=True,
    )
    assert acceptance["passed"], acceptance


def test_cold_packet_refuses_missing_canonical_accepted_epoch_roles(tmp_path: Path) -> None:
    receipt = phx.lifecycle.write_meshing_source_closure(
        tmp_path / "incomplete-packet.zip",
        ("incomplete",),
    )
    with pytest.raises(QualificationFailure, match="canonical accepted-epoch"):
        _corpus_cold_worker(
            receipt.path, receipt.content_id, (), 1, 0.05,
            "planar-hole-feature", "python", (), budget=None,
        )


def test_native_cold_request_owns_restore_and_physical_continuation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from tools.meshing_qualification._runtime import (
        ColdControlPacket,
        ColdWorkerReport,
        Deadline,
    )
    from tools.meshing_qualification.scenarios import native_lifecycle as owner

    def remaining(deadline: Deadline) -> float:
        return 37.0

    monkeypatch.setattr(Deadline, "remaining", remaining)
    events = []
    actual_budget, restored, continued = object(), object(), object()
    raw = SimpleNamespace(execution_evidence=object(), limits=object())
    packet = ColdControlPacket(
        archive_path=tmp_path / "cold.zip",
        content_id="cold-content",
        controls={
            "expected": [], "repeats": 1, "target_error": 0.05, "engine": "native",
            "case_name": "planar-hole-feature", "epoch_errors": [],
        },
        deadline=Deadline(expires_at=37.0),
    )

    def read_controls(*args: object, **kwargs: object) -> object:
        events.append("raw-controls")
        assert args == (packet.archive_path,)
        assert kwargs["expected_content_id"] == packet.content_id
        return raw

    def restore(*args: object, budget: object) -> tuple[object, ColdWorkerReport]:
        events.append("restore-and-physical")
        assert budget is actual_budget
        return restored, ColdWorkerReport(record={"passed": True}, read_seconds=0.125)

    def request(
        source: object, limits: object, call: Callable[[object], object], **kwargs: object,
    ) -> object:
        events.append("actual-request")
        assert source is raw.execution_evidence and limits is raw.limits
        assert kwargs["workflow_remaining_seconds"] == 37.0
        value = call(actual_budget)
        target = kwargs["target"]
        assert callable(target) and target(value) is restored
        return SimpleNamespace(value=value, result=continued)

    monkeypatch.setattr(owner, "read_meshing_source_execution_controls", read_controls)
    monkeypatch.setattr(owner, "_corpus_cold_worker", restore)
    monkeypatch.setattr(owner, "execute_native_request", request)
    monkeypatch.setattr(owner, "_result_identity", lambda value: {"actual": value})
    report = owner.cold_worker(packet)
    assert events == ["raw-controls", "actual-request", "restore-and-physical"]
    assert report.record == {"passed": True, "continued_result": {"actual": continued}}
    assert report.read_seconds >= 0.125


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
