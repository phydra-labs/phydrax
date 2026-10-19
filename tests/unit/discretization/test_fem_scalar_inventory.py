# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Conserved scalar inventories on periodic affine coefficient-action carriers."""

from dataclasses import dataclass

import numpy as np
import pytest

from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    NestedReferenceWitnesses,
    PeriodicCell,
    PeriodicMeshTopology,
)
from phydrax.discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    coordinate_lagrange_element,
)
from phydrax.discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
    prepare_nested_field_transfer,
)
from phydrax.meshing import refine_periodic_mesh


@dataclass(frozen=True)
class ScalarContinuation:
    coarse: FiniteElementDiscretization
    fine: FiniteElementDiscretization
    coarse_geometry: CellGeometrySpec
    fine_geometry: CellGeometrySpec
    parent_cells: np.ndarray
    reference_vertices: np.ndarray
    coarsening: NestedReferenceWitnesses


def _coefficient_action_geometry(mesh: CellMesh) -> CellGeometrySpec:
    # A cyclic coefficient permutation represents the same affine map while
    # retaining a nonidentity coefficient action instead of a nodal map alias.
    order = (1, 2, 0)
    element = BarycentricCellGeometryElement(
        coordinate_lagrange_element("triangle", 1),
        np.eye(3, dtype=np.float64)[:, order],
    )
    cells = np.concatenate(tuple(np.asarray(block.vertices) for block in mesh.blocks))
    coordinates = np.asarray(mesh.coordinates)[cells][:, order].reshape((-1, 2))
    routes = np.arange(coordinates.shape[0], dtype=np.int32).reshape((-1, 3))
    block_routes: dict[str, np.ndarray] = {}
    start = 0
    for block in mesh.blocks:
        stop = start + block.cell_count
        block_routes[block.name] = routes[start:stop]
        start = stop
    return CellGeometrySpec(
        {block.name: element for block in mesh.blocks}, block_routes, coordinates
    )


@pytest.fixture(scope="module")
def scalar_continuation() -> ScalarContinuation:
    shifts = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.int64)
    lattice = np.asarray(((1.0, 0.25), (0.0, 1.0)), dtype=np.float64)
    # PeriodicCell stores direct lattice vectors as rows.
    coordinates = shifts @ lattice
    block = CellBlock(
        "torus", "triangle", np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32)
    )
    lifted = CellMesh(coordinates, (block,))
    topology = PeriodicMeshTopology(
        lifted, PeriodicCell(lattice), np.zeros(4, dtype=np.int64), shifts
    )
    mesh = CellMesh(coordinates, (block,), periodic_topology=topology)
    refinement = refine_periodic_mesh(mesh)
    coarse_geometry = _coefficient_action_geometry(mesh)
    fine_geometry = _coefficient_action_geometry(refinement.mesh)
    field = FiniteElementFieldSpec("u", lagrange_element("triangle", 2))
    coarse = FiniteElementPlan(mesh, field, coordinate_spec=coarse_geometry).prepare()
    fine = FiniteElementPlan(
        refinement.mesh, field, coordinate_spec=fine_geometry
    ).prepare()
    fine_ids = np.concatenate(
        tuple(np.asarray(part.global_ids) for part in refinement.mesh.blocks)
    )
    coarse_ids = np.concatenate(
        tuple(np.asarray(part.global_ids) for part in mesh.blocks)
    )
    parents = np.asarray(refinement.parent_cells)
    reference = np.asarray(refinement.parent_reference_vertices)
    return ScalarContinuation(
        coarse,
        fine,
        coarse_geometry,
        fine_geometry,
        parents,
        reference,
        NestedReferenceWitnesses(fine_ids, coarse_ids[parents], reference),
    )


def test_periodic_affine_coefficient_action_refinement_preserves_inventory(
    scalar_continuation: ScalarContinuation,
) -> None:
    case = scalar_continuation
    transfer = prepare_nested_field_transfer(
        case.coarse,
        case.fine,
        case.parent_cells,
        field_name="u",
        parent_reference_vertices=case.reference_vertices,
        source_geometry=case.coarse_geometry,
        target_geometry=case.fine_geometry,
    )
    assert transfer.evidence.passed
    assert transfer.transfer.conservative
    assert transfer.source_measures is not None
    assert transfer.target_measures is not None
    values = np.random.default_rng(47).normal(
        size=case.coarse.dof_maps[0].global_dof_count
    )
    source = np.asarray(transfer.source_measures)
    target = np.asarray(transfer.target_measures)
    before = source @ values
    after = target @ np.asarray(transfer.transfer.apply(values))
    # The constructor's 64-ulp, action-conditioned per-column measure bound
    # propagates through the l1 norm of the arbitrary accepted coefficient state.
    bound = (
        64.0
        * np.finfo(np.float64).eps
        * transfer.transfer.action_condition
        * (np.sum(np.abs(source)) + np.sum(np.abs(target)))
        * np.sum(np.abs(values))
    )
    assert abs(after - before) <= bound


def test_periodic_affine_coefficient_action_coarsening_preserves_inventory(
    scalar_continuation: ScalarContinuation,
) -> None:
    case = scalar_continuation
    transfer = prepare_nested_field_transfer(
        case.fine,
        case.coarse,
        None,
        field_name="u",
        source_geometry=case.fine_geometry,
        target_geometry=case.coarse_geometry,
        coarsening_witnesses=case.coarsening,
    )
    assert transfer.evidence.passed
    assert transfer.transfer.conservative
    assert transfer.source_measures is not None
    assert transfer.target_measures is not None
    values = np.random.default_rng(53).normal(size=case.fine.dof_maps[0].global_dof_count)
    source = np.asarray(transfer.source_measures)
    target = np.asarray(transfer.target_measures)
    before = source @ values
    after = target @ np.asarray(transfer.transfer.apply(values))
    bound = (
        64.0
        * np.finfo(np.float64).eps
        * transfer.transfer.action_condition
        * (np.sum(np.abs(source)) + np.sum(np.abs(target)))
        * np.sum(np.abs(values))
    )
    assert abs(after - before) <= bound
