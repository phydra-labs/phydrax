# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Request-owned validity imports retain full certification evidence."""

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import equinox as eqx
import numpy as np
import pytest
from jax import Array

from phydrax._geometry_predicates import PolygonSimplicityResult, PredicateMode
from phydrax.discretization import (
    _cell_geometry_validity as validity_owner,
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
    fem,
    reference_cell_topology,
)
from phydrax.discretization._cell_geometry_predecessor import (
    prepare_cell_geometry_predecessor,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from tests._support.coordinate_references import coordinate_reference_budget


def _triangle_geometry(
    change: Literal["none", "invalid", "unresolved"],
    *,
    reorder: bool = False,
) -> tuple[CellMesh, CellGeometrySpec]:
    element = fem.lagrange_element("triangle", 2)
    reference = np.asarray(element.reference_nodes, dtype=np.float64)
    local = np.stack([reference + (3.0 * cell, 0.0) for cell in range(3)])
    if change == "invalid":
        local[1, :, 0] = 6.0 - local[1, :, 0]
    elif change == "unresolved":
        # The established curved triangle needs subdivision to prove positivity.
        local[1, 3:] = np.asarray(
            ((0.86, -0.17), (0.54, 0.29), (-0.06, 0.51)), dtype=np.float64
        ) + (3.0, 0.0)
    vertices = np.asarray(reference_cell_topology("triangle").vertices, dtype=np.float64)
    points = np.concatenate([vertices + (3.0 * cell, 0.0) for cell in range(3)])
    if change == "invalid":
        points[3:6, 0] = 6.0 - points[3:6, 0]
    cells = np.arange(9, dtype=np.int32).reshape((3, 3))
    routes = np.arange(local.size // 2, dtype=np.int32).reshape((3, -1))
    first = np.asarray((1, 0) if reorder else (0, 1), dtype=np.int32)
    blocks = (
        CellBlock("first", "triangle", cells[first], global_ids=101 + first),
        CellBlock("second", "triangle", cells[2:], global_ids=np.asarray((103,))),
    )
    mesh = CellMesh(points, blocks[::-1] if reorder else blocks)
    geometry = CellGeometrySpec(
        {"first": element, "second": element},
        {"first": routes[first], "second": routes[2:]},
        local.reshape((-1, 2)),
    )
    return mesh, geometry


def _polygon_geometry(*, changed: bool) -> tuple[CellMesh, CellGeometrySpec]:
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.5, 0.5), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    points = np.concatenate([vertices + (3.0 * cell, 0.0) for cell in range(3)])
    if changed:
        points[5:10, 0] = 7.5 - points[5:10, 0]
    mesh = CellMesh(
        points,
        (
            CellBlock(
                "cells",
                "polygon",
                np.arange(15, dtype=np.int32).reshape((3, 5)),
            ),
        ),
    )
    return mesh, CellGeometrySpec.affine(mesh)


def _assert_same_certificate(
    full: CellValidityCertificate, inherited: CellValidityCertificate
) -> None:
    np.testing.assert_array_equal(inherited.status, full.status)
    np.testing.assert_array_equal(inherited.determinant_lower, full.determinant_lower)
    np.testing.assert_array_equal(inherited.determinant_upper, full.determinant_upper)
    np.testing.assert_array_equal(inherited.depth, full.depth)
    assert inherited.block_names == full.block_names
    assert inherited.block_offsets == full.block_offsets
    assert inherited.unsupported_block_names == full.unsupported_block_names
    assert inherited.unresolved_reasons == full.unresolved_reasons
    assert inherited.certified_valid_count == full.certified_valid_count
    assert inherited.invalid_count == full.invalid_count
    assert inherited.unresolved_count == full.unresolved_count
    assert inherited.certificate_id == full.certificate_id


@pytest.mark.parametrize(
    ("change", "expected"),
    (
        ("none", CellValidityStatus.CERTIFIED_VALID),
        ("invalid", CellValidityStatus.INVALID),
        ("unresolved", CellValidityStatus.UNRESOLVED),
    ),
)
def test_predecessor_preserves_polynomial_proof_and_target_block_order(
    change: Literal["none", "invalid", "unresolved"],
    expected: CellValidityStatus,
) -> None:
    source_mesh, source_geometry = _triangle_geometry("none")
    target_mesh, target_geometry = _triangle_geometry(change, reorder=True)
    policy = CellValidityPolicy(maximum_subdivision_depth=0)
    source = certify_cell_geometry_validity(
        source_geometry, mesh=source_mesh, policy=policy
    )
    with coordinate_reference_budget() as preparation:
        predecessor = prepare_cell_geometry_predecessor(
            source_mesh,
            source_geometry,
            source,
            None,
            target_mesh,
            target_geometry,
            budget=preparation,
        )
    assert predecessor.source_rows == ((2, 1, 0) if change == "none" else (2, -1, 0))
    with coordinate_reference_budget() as full_budget:
        full = certify_cell_geometry_validity(
            target_geometry, mesh=target_mesh, policy=policy, budget=full_budget
        )
    with coordinate_reference_budget() as inherited_budget:
        inherited = certify_cell_geometry_validity(
            target_geometry,
            mesh=target_mesh,
            policy=policy,
            budget=inherited_budget,
            predecessor=predecessor,
        )
    _assert_same_certificate(full, inherited)
    assert inherited.block_names == ("second", "first")
    assert inherited.block_offsets == (0, 1, 3)
    assert tuple(np.asarray(inherited.status)) == (
        CellValidityStatus.CERTIFIED_VALID,
        expected,
        CellValidityStatus.CERTIFIED_VALID,
    )
    assert inherited_budget.work_units < full_budget.work_units


def test_predecessor_certifies_changed_polygon_without_rechecking_unchanged_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_mesh, source_geometry = _polygon_geometry(changed=False)
    target_mesh, target_geometry = _polygon_geometry(changed=True)
    source = certify_cell_geometry_validity(source_geometry, mesh=source_mesh)
    with coordinate_reference_budget() as preparation:
        predecessor = prepare_cell_geometry_predecessor(
            source_mesh,
            source_geometry,
            source,
            None,
            target_mesh,
            target_geometry,
            budget=preparation,
        )
    assert predecessor.source_rows == (0, -1, 2)
    counted_rows = 0
    predicate = validity_owner.polygon_simplicity_2d

    def counted(
        vertices: np.ndarray, /, *, mode: PredicateMode, budget: CoordinateEnclosureBudget
    ) -> PolygonSimplicityResult:
        nonlocal counted_rows
        counted_rows += vertices.shape[0]
        return predicate(vertices, mode=mode, budget=budget)

    monkeypatch.setattr(validity_owner, "polygon_simplicity_2d", counted)
    with coordinate_reference_budget() as full_budget:
        full = certify_cell_geometry_validity(
            target_geometry, mesh=target_mesh, budget=full_budget
        )
    full_rows = counted_rows
    counted_rows = 0
    with coordinate_reference_budget() as inherited_budget:
        inherited = certify_cell_geometry_validity(
            target_geometry,
            mesh=target_mesh,
            budget=inherited_budget,
            predecessor=predecessor,
        )
    _assert_same_certificate(full, inherited)
    assert inherited.invalid_count == 1
    assert int(inherited.status[1]) == CellValidityStatus.INVALID
    assert counted_rows < full_rows


@pytest.mark.parametrize(
    "select",
    (
        pytest.param(lambda value: value.status, id="status"),
        pytest.param(lambda value: value.determinant_lower, id="determinant-lower"),
        pytest.param(lambda value: value.determinant_upper, id="determinant-upper"),
        pytest.param(lambda value: value.depth, id="depth"),
    ),
)
def test_predecessor_refuses_tampered_source_payload_even_with_warm_proof_cache(
    select: Callable[[CellValidityCertificate], Array],
) -> None:
    mesh, geometry = _triangle_geometry("none")
    source = certify_cell_geometry_validity(geometry, mesh=mesh)
    with coordinate_reference_budget() as preparation:
        predecessor = prepare_cell_geometry_predecessor(
            mesh, geometry, source, None, mesh, geometry, budget=preparation
        )
    payload = select(source)
    altered = eqx.tree_at(select, source, payload.at[0].add(1))
    assert altered.certificate_id == source.certificate_id
    tampered = replace(predecessor, source_validity=altered)
    with coordinate_reference_budget() as budget:
        certify_cell_geometry_validity(geometry, mesh=mesh, budget=budget)
        with pytest.raises(ValueError, match="CellGeometryPredecessor.content_id"):
            certify_cell_geometry_validity(
                geometry, mesh=mesh, budget=budget, predecessor=tampered
            )


def test_predecessor_refuses_another_validity_policy() -> None:
    mesh, geometry = _triangle_geometry("none")
    source = certify_cell_geometry_validity(geometry, mesh=mesh)
    with coordinate_reference_budget() as preparation:
        predecessor = prepare_cell_geometry_predecessor(
            mesh, geometry, source, None, mesh, geometry, budget=preparation
        )
    with pytest.raises(ValueError, match="predecessor.policy_id"):
        certify_cell_geometry_validity(
            geometry,
            mesh=mesh,
            policy=CellValidityPolicy(maximum_subdivision_depth=0),
            predecessor=predecessor,
        )
