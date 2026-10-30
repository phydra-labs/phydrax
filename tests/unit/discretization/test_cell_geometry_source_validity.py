from fractions import Fraction
from typing import Any

import numpy as np

from phydrax.discretization import (
    CellBlock,
    CellMesh,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from phydrax.discretization._cell_geometry_source_validity import (
    certify_exact_variable_topology_block,
)


def _exact_bank(
    points: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    return points


def _polygon_mesh(points: tuple[tuple[Fraction, ...], ...]) -> CellMesh:
    rounded = np.asarray([[float(value) for value in point] for point in points])
    vertices = np.arange(len(points), dtype=np.int64)[None, :]
    return CellMesh(rounded, (CellBlock("cells", "polygon", vertices),))


def _polygon_proof(
    mesh: CellMesh,
    points: tuple[tuple[Fraction, ...], ...],
    policy: CellValidityPolicy,
) -> Any:
    routes = {"cells": np.arange(len(points), dtype=np.int32)[None, :]}
    return certify_exact_variable_topology_block(
        mesh,
        routes,
        _exact_bank(points),
        policy,
        block_name="cells",
        cell_rows=np.asarray((0,), dtype=np.int64),
    )


def test_exact_polygon_source_validity_is_not_shadowed_by_rounded_vertices() -> None:
    epsilon = Fraction(1, 1 << 60)
    exact = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(1) + epsilon, Fraction(0)),
        (Fraction(1) + epsilon, Fraction(1)),
        (Fraction(0), Fraction(1)),
    )
    mesh = _polygon_mesh(exact)
    policy = CellValidityPolicy(
        relative_determinant_floor=0.0,
        relative_planarity_tolerance=0.0,
    )

    shadow = certify_cell_geometry_validity(mesh, policy=policy)
    exact_proof = _polygon_proof(mesh, exact, policy)

    assert int(shadow.status[0]) == CellValidityStatus.INVALID
    assert int(exact_proof.status[0]) == CellValidityStatus.CERTIFIED_VALID
    assert float(exact_proof.lower[0]) > 0.0
    np.testing.assert_array_equal(exact_proof.depth, (0,))


def test_exact_polygon_planarity_rejects_a_planar_binary64_shadow() -> None:
    epsilon = Fraction(1, 1 << 60)
    exact = (
        (Fraction(0), Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(1), Fraction(1)),
        (Fraction(0), Fraction(1), Fraction(1) + epsilon),
        (Fraction(0), Fraction(1, 2), Fraction(1)),
    )
    mesh = _polygon_mesh(exact)
    policy = CellValidityPolicy(
        relative_determinant_floor=0.0,
        relative_planarity_tolerance=0.0,
    )

    rounded_bank = tuple(
        tuple(Fraction(float(value)) for value in point) for point in exact
    )
    shadow = _polygon_proof(mesh, rounded_bank, policy)
    exact_proof = _polygon_proof(mesh, exact, policy)

    assert int(shadow.status[0]) == CellValidityStatus.CERTIFIED_VALID
    assert int(exact_proof.status[0]) == CellValidityStatus.INVALID
    np.testing.assert_array_equal(exact_proof.depth, (0,))


def _tetra_polyhedron(
    points: tuple[tuple[Fraction, Fraction, Fraction], ...],
) -> CellMesh:
    rounded = np.asarray([[float(value) for value in point] for point in points])
    faces = (
        (1, 2, 3),
        (0, 3, 2),
        (0, 1, 3),
        (0, 2, 1),
    )
    return CellMesh.from_polyhedra(rounded, (faces,))


def _polyhedron_proof(
    mesh: CellMesh,
    points: tuple[tuple[Fraction, Fraction, Fraction], ...],
    policy: CellValidityPolicy,
) -> Any:
    routes = {mesh.blocks[0].name: np.arange(len(points), dtype=np.int32)[None, :]}
    return certify_exact_variable_topology_block(
        mesh,
        routes,
        _exact_bank(points),
        policy,
        block_name=mesh.blocks[0].name,
        cell_rows=np.asarray((0,), dtype=np.int64),
    )


def test_exact_polyhedron_star_proof_preserves_shadow_and_degeneracy_semantics() -> None:
    valid_points = (
        (Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    mesh = _tetra_polyhedron(valid_points)
    policy = CellValidityPolicy(relative_determinant_floor=0.0)

    shadow = certify_cell_geometry_validity(mesh, policy=policy)
    valid_proof = _polyhedron_proof(mesh, valid_points, policy)
    degenerate_points = (*valid_points[:3], valid_points[0])
    degenerate_proof = _polyhedron_proof(mesh, degenerate_points, policy)

    assert int(shadow.status[0]) == CellValidityStatus.CERTIFIED_VALID
    assert int(valid_proof.status[0]) == CellValidityStatus.CERTIFIED_VALID
    assert float(valid_proof.lower[0]) > 0.0
    assert int(degenerate_proof.status[0]) == CellValidityStatus.UNRESOLVED
    np.testing.assert_array_equal(valid_proof.depth, (0,))
    np.testing.assert_array_equal(degenerate_proof.depth, (0,))
