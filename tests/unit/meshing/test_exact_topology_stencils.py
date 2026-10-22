#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact source-owned homogeneous stencils through topology edits and lineage.

The fixture splits the triangle ``A=(0, 0), B=(1, 0), C=(0, 1)`` along the zero set
of the P1 field ``(-3, 7, 5)``. Its exact roots are ``P = (7A + 3B)/10``, which is
nondyadic, and ``Q = (5A + 3C)/8``, with homogeneous weights ``(7, 3)`` and
``(5, 3)``; the published carriers are the correctly rounded exact points. Rounded
barycentrics of ``P`` miss the zero by ``2**-54`` and sum to ``1 - 2**-54``, so only
the retained exact bank reproduces affine data at the root.
"""

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import CellMesh
from phydrax.discretization._cell_geometry import owned_coordinate_preparation
from phydrax.discretization._cell_geometry_transition_types import (
    homogeneous_reference_point,
    NestedReferenceWitnesses,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.lifecycle._meshing_source_records import reconstruct_stored_fields
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    register_meshing_source_artifacts,
    write_meshing_source_closure,
)
from phydrax.meshing import CellMeshTransition, EntityLineageKind, MeshTransitionKind
from phydrax.meshing._lineage_records import (
    rounded_homogeneous_barycentrics,
    VertexInterpolationStencil,
)
from phydrax.meshing._topology_edit import (
    assemble_topology_edit,
    CellTopologyEdit,
    EntityRelations,
    exact_nested_reference_vertices,
    topology_edit_exact_preparation_bounds,
    TopologyEditBlock,
)
from tests._support.coordinate_references import coordinate_reference_budget


_POINTS = {
    1: (Fraction(0), Fraction(0)),
    2: (Fraction(1), Fraction(0)),
    3: (Fraction(0), Fraction(1)),
}
_VALUES = {1: Fraction(-3), 2: Fraction(7), 3: Fraction(5)}
_IDS = np.asarray((1, 2, 3, 4, 5), dtype=np.int64)
_SOURCES = np.asarray(((1, 1), (2, 2), (3, 3), (1, 2), (1, 3)), dtype=np.int64)
_HOMOGENEOUS = np.asarray(
    ((1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (7.0, 3.0), (5.0, 3.0)), dtype=np.float64
)
# Target triangles (A, P, Q), (P, B, C), (P, C, Q) refine source cell 10.
_CELLS = np.asarray(((0, 3, 4), (3, 1, 2), (3, 2, 4)), dtype=np.int32)
_CELL_IDS = np.asarray((11, 12, 13), dtype=np.int64)
_EXACT_ROOTS = {
    4: (Fraction(3, 10), Fraction(0)),
    5: (Fraction(0), Fraction(3, 8)),
}


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    with coordinate_reference_budget() as ledger:
        yield ledger


def _source() -> CellMesh:
    return CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2),), dtype=np.int32),
        vertex_global_ids=np.asarray((1, 2, 3), dtype=np.int64),
        cell_global_ids=np.asarray((10,), dtype=np.int64),
    )


def _exact_point(sources: list[int], homogeneous: list[float]) -> tuple[Fraction, ...]:
    weights = [Fraction(value) for value in homogeneous]
    total = sum(weights, Fraction(0))
    return tuple(
        sum(
            (
                weight / total * _POINTS[source][axis]
                for weight, source in zip(weights, sources, strict=True)
            ),
            Fraction(0),
        )
        for axis in range(2)
    )


def _edit(source: CellMesh, budget: CoordinateEnclosureBudget) -> CellTopologyEdit:
    valid = np.ones(_SOURCES.shape, dtype=np.bool_)
    coordinates = np.asarray(
        [
            [float(value) for value in _exact_point(row, weights)]
            for row, weights in zip(_SOURCES.tolist(), _HOMOGENEOUS.tolist(), strict=True)
        ],
        dtype=np.float64,
    )
    reference, bank = exact_nested_reference_vertices(
        _SOURCES[_CELLS],
        _HOMOGENEOUS[_CELLS],
        np.tile(np.asarray((1, 2, 3)), (3, 1)),
        budget=budget,
    )
    block = source.blocks[0]
    empty = np.zeros(0, dtype=np.int32)
    return CellTopologyEdit(
        "nested_refinement",
        coordinates,
        _IDS,
        (
            TopologyEditBlock(
                block.name, block.cell_kind, block.cell_kind, _CELLS, _CELL_IDS
            ),
        ),
        _SOURCES,
        rounded_homogeneous_barycentrics(_HOMOGENEOUS, valid, budget=budget),
        valid,
        (
            EntityRelations(
                0, np.zeros((0, 1), np.int64), np.zeros((0, 1), np.int64), empty
            ),
            EntityRelations(
                1, np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64), empty
            ),
            EntityRelations(
                2,
                np.full((3, 1), 10, dtype=np.int64),
                _CELL_IDS[:, None],
                np.full(3, int(EntityLineageKind.REFINED_FROM), dtype=np.int32),
            ),
        ),
        refinement=NestedReferenceWitnesses(
            _CELL_IDS, np.full(3, 10, dtype=np.int64), reference, bank
        ),
        stencil_homogeneous=_HOMOGENEOUS,
    )


def _transition(
    source: CellMesh, budget: CoordinateEnclosureBudget
) -> CellMeshTransition:
    edit = _edit(source, budget)
    target, lineage, stencil = assemble_topology_edit(
        source, edit, numeric_version="exact-roots", budget=budget
    )
    return CellMeshTransition(
        source.mesh_id,
        source.topology_id,
        phx.meshing.certify_cell_mesh(target, phx.SpatialCoordinateContract.si()),
        lineage,
        MeshTransitionKind.REFINE,
        vertex_stencil=stencil,
        parents=edit.refinement,
        budget=budget,
    )


def _replay(value: Any) -> Any:
    """Archive replay on a ledger sized only by the record's declared bound."""
    work, storage, queries = value.reconstruction_preparation_bounds()
    with owned_coordinate_preparation(
        work, storage, maximum_geometry_queries=queries
    ) as ledger:
        return reconstruct_stored_fields(value, bindings={"budget": ledger})


def test_exact_roots_survive_edit_lineage_and_archive(
    tmp_path: Path, budget: CoordinateEnclosureBudget
) -> None:
    transition = _transition(_source(), budget)
    receipt = write_meshing_source_closure(tmp_path / "exact-roots", transition)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored, CellMeshTransition)
    assert restored.transition_id == transition.transition_id
    stencil = restored.vertex_stencil
    assert stencil is not None and stencil.homogeneous_weights is not None
    assert transition.vertex_stencil is not None
    assert stencil.stencil_id == transition.vertex_stencil.stencil_id
    bank = np.asarray(stencil.homogeneous_weights)
    weights = np.asarray(stencil.weights)
    np.testing.assert_array_equal(bank.view(np.uint64), _HOMOGENEOUS.view(np.uint64))
    for row, (identifier, sources) in enumerate(
        zip(
            np.asarray(stencil.target_global_ids).tolist(),
            np.asarray(stencil.source_global_ids).tolist(),
            strict=True,
        )
    ):
        exact = [Fraction(value) for value in bank[row].tolist()]
        barycentrics = [value / sum(exact, Fraction(0)) for value in exact]
        assert weights[row].tolist() == [float(value) for value in barycentrics]
        if identifier in _EXACT_ROOTS:
            # Rational affine reproduction: the bank places the root on the P1 zero.
            assert _exact_point(sources, bank[row].tolist()) == _EXACT_ROOTS[identifier]
            value = sum(
                (
                    weight * _VALUES[source]
                    for weight, source in zip(barycentrics, sources, strict=True)
                ),
                Fraction(0),
            )
            assert value == 0
    rounded = [Fraction(value) for value in weights[3].tolist()]
    assert sum(rounded, Fraction(0)) != 1
    assert rounded[0] * _VALUES[1] + rounded[1] * _VALUES[2] != 0


def test_exact_reference_bank_composes_nested_root_charts(
    tmp_path: Path, budget: CoordinateEnclosureBudget
) -> None:
    transition = _transition(_source(), budget)
    receipt = write_meshing_source_closure(tmp_path / "exact-parents", transition)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored, CellMeshTransition)
    assert restored.parent_reference_homogeneous is not None
    parents = np.asarray(restored.parent_cell_ids)
    vertices = np.asarray(restored.parent_reference_vertices)
    bank = np.asarray(restored.parent_reference_homogeneous)
    np.testing.assert_array_equal(parents, (10, 10, 10))
    target_ids = np.asarray(restored.target.mesh.vertex_global_ids)
    target_cells = np.asarray(restored.target.mesh.blocks[0].vertices)
    corners = {1: _POINTS[1], 2: _POINTS[2], 3: _POINTS[3]}
    for cell in range(_CELL_IDS.size):
        for local, row in enumerate(target_cells[cell].tolist()):
            identifier = int(target_ids[row])
            reference = homogeneous_reference_point(bank[cell, local])
            assert reference is not None
            assert vertices[cell, local].tolist() == [float(value) for value in reference]
            # Compose the coarse chart A + xi_1 (B - A) + xi_2 (C - A) exactly.
            physical = tuple(
                corners[1][axis]
                + reference[0] * (corners[2][axis] - corners[1][axis])
                + reference[1] * (corners[3][axis] - corners[1][axis])
                for axis in range(2)
            )
            expected = _EXACT_ROOTS.get(identifier, _POINTS.get(identifier))
            assert physical == expected


def test_assembly_refuses_malformed_or_unbound_exact_banks(
    budget: CoordinateEnclosureBudget,
) -> None:
    source = _source()
    edit = _edit(source, budget)
    witnesses = edit.refinement
    assert witnesses is not None
    shifted = np.asarray(edit.stencil_weights).copy()
    shifted[3, 0] = np.nextafter(shifted[3, 0], 1.0)
    signed = _HOMOGENEOUS.copy()
    signed[0, 1] = -0.0
    scaled = _HOMOGENEOUS.copy()
    scaled[3] *= 2.0
    cases = (
        (
            edit._replace(stencil_weights=shifted),
            r"VertexInterpolationStencil\.weights must be the correctly rounded",
        ),
        (
            edit._replace(stencil_homogeneous=signed),
            r"homogeneous_weights must be finite sign-bit-free",
        ),
        # Same execution weights, different rational content than the witnesses.
        (
            edit._replace(stencil_homogeneous=scaled),
            r"CellTopologyEdit\.refinement must be ",
        ),
        (
            edit._replace(stencil_homogeneous=None),
            r"CellTopologyEdit\.refinement must derive exact reference banks",
        ),
        (
            edit._replace(refinement=witnesses._replace(fine_reference_homogeneous=None)),
            r"CellTopologyEdit\.refinement must retain the exact reference bank",
        ),
    )
    for changed, message in cases:
        with pytest.raises(ValueError, match=message):
            assemble_topology_edit(
                source, changed, numeric_version="refused", budget=budget
            )


def test_archive_replay_is_bounded_and_refuses_changed_exact_content(
    budget: CoordinateEnclosureBudget,
) -> None:
    register_meshing_source_artifacts()
    transition = _transition(_source(), budget)
    stencil = transition.vertex_stencil
    assert stencil is not None and stencil.homogeneous_weights is not None
    assert _replay(stencil).stencil_id == stencil.stencil_id
    assert _replay(transition).transition_id == transition.transition_id
    scaled = np.asarray(stencil.homogeneous_weights).copy()
    scaled[3] *= 2.0
    # The rounded weights still match, but the stencil identity binds the bank.
    forged = eqx.tree_at(
        lambda value: value.homogeneous_weights, stencil, jax.device_put(scaled)
    )
    with pytest.raises(ValueError, match=r"stored_fields\.VertexInterpolationStencil"):
        _replay(forged)
    mismatched = np.asarray(stencil.homogeneous_weights).copy()
    mismatched[3] = (7.0, 4.0)
    forged = eqx.tree_at(
        lambda value: value.homogeneous_weights, stencil, jax.device_put(mismatched)
    )
    with pytest.raises(ValueError, match=r"VertexInterpolationStencil\.weights"):
        _replay(forged)
    assert transition.parent_reference_homogeneous is not None
    bank = np.asarray(transition.parent_reference_homogeneous).copy()
    bank[0, 1] = (0.0, 7.0, 4.0)
    forged_transition = eqx.tree_at(
        lambda value: value.parent_reference_homogeneous,
        transition,
        jax.device_put(bank),
    )
    with pytest.raises(ValueError, match=r"CellMeshTransition\.parents"):
        _replay(forged_transition)
    bank = np.asarray(transition.parent_reference_homogeneous).copy()
    bank[0, 1] *= 2.0
    forged_transition = eqx.tree_at(
        lambda value: value.parent_reference_homogeneous,
        transition,
        jax.device_put(bank),
    )
    with pytest.raises(
        ValueError, match=r"stored_fields\.CellMeshTransition\.transition_id"
    ):
        _replay(forged_transition)


def test_exact_assembly_is_metered_within_its_declared_bound() -> None:
    source = _source()
    with coordinate_reference_budget() as setup:
        edit = _edit(source, setup)
    work, storage = topology_edit_exact_preparation_bounds(edit)
    assert work > 0 and storage > 0
    # Nonperiodic assembly charges only the rational-bank phases to its ledger.
    with owned_coordinate_preparation(work, storage) as ledger:
        _, _, stencil = assemble_topology_edit(
            source, edit, numeric_version="metered", budget=ledger
        )
        charged = ledger.work_units
    assert stencil is not None and stencil.homogeneous_weights is not None
    assert 0 < charged <= work
    with pytest.raises(CoordinateEnclosureResourceError):
        with owned_coordinate_preparation(charged - 1, storage) as ledger:
            assemble_topology_edit(source, edit, numeric_version="refused", budget=ledger)
    witnesses = edit.refinement
    assert witnesses is not None
    rounded = edit._replace(
        stencil_homogeneous=None,
        refinement=witnesses._replace(fine_reference_homogeneous=None),
    )
    assert topology_edit_exact_preparation_bounds(rounded) == (0, 0)


def test_exact_nested_reference_vertices_aggregates_exact_corners(
    budget: CoordinateEnclosureBudget,
) -> None:
    sources = np.asarray((((1, 1), (2, 3), (3, -1)),), dtype=np.int64)
    homogeneous = np.asarray((((1.0, 2.0), (1.0, 2.0), (1.0, 0.0)),))
    corners = np.asarray(((1, 2, 3),))
    start = budget.work_units
    reference, bank = exact_nested_reference_vertices(
        sources, homogeneous, corners, budget=budget
    )
    # One distinct mixed stencil row (two corners share it) and one aggregated
    # fine corner are charged before their rational work.
    assert budget.work_units - start == 4 * 2 + 3 * (2 * 2 + 8)
    np.testing.assert_array_equal(
        bank, (((3.0, 0.0, 0.0), (0.0, 1.0, 2.0), (0.0, 0.0, 1.0)),)
    )
    assert reference.tolist() == [
        [[0.0, 0.0], [float(Fraction(1, 3)), float(Fraction(2, 3))], [0.0, 1.0]]
    ]
    inexact = homogeneous.copy()
    inexact[0, 0] = (2.0**53, 1.0)
    with pytest.raises(ValueError, match="aggregate each coarse corner exactly"):
        exact_nested_reference_vertices(sources, inexact, corners, budget=budget)
    unsupported = homogeneous.copy()
    unsupported[0, 2, 1] = 1.0
    with pytest.raises(ValueError, match="finite sign-bit-free"):
        exact_nested_reference_vertices(sources, unsupported, corners, budget=budget)


def test_ordinary_stencils_stay_general_and_exact_banks_bind_identity() -> None:
    sources = np.asarray(((1, 2),), dtype=np.int64)
    valid = np.ones((1, 2), dtype=np.bool_)
    extrapolating = VertexInterpolationStencil(
        "source", "target", (7,), sources, ((1.5, -0.5),), valid
    )
    assert extrapolating.homogeneous_weights is None
    with pytest.raises(ValueError, match=r"VertexInterpolationStencil\.weights"):
        VertexInterpolationStencil(
            "source",
            "target",
            (7,),
            sources,
            ((1.5, -0.5),),
            valid,
            homogeneous_weights=((3.0, 1.0),),
        )
    rounded = ((0.75, 0.25),)
    ordinary = VertexInterpolationStencil(
        "source", "target", (7,), sources, rounded, valid
    )
    exact = VertexInterpolationStencil(
        "source",
        "target",
        (7,),
        sources,
        rounded,
        valid,
        homogeneous_weights=((3.0, 1.0),),
    )
    rescaled = VertexInterpolationStencil(
        "source",
        "target",
        (7,),
        sources,
        rounded,
        valid,
        homogeneous_weights=((6.0, 2.0),),
    )
    assert len({ordinary.stencil_id, exact.stencil_id, rescaled.stencil_id}) == 3
    with pytest.raises(ValueError, match="constant-preserving"):
        VertexInterpolationStencil(
            "source",
            "target",
            (7,),
            sources,
            rounded,
            valid,
            preserves_constants=False,
            homogeneous_weights=((3.0, 1.0),),
        )
    with pytest.raises(ValueError, match="positive exact sum in every row"):
        VertexInterpolationStencil(
            "source",
            "target",
            (7,),
            sources,
            ((0.0, 0.0),),
            np.zeros((1, 2), dtype=np.bool_),
            homogeneous_weights=((0.0, 0.0),),
        )
