#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
import weakref
from contextlib import nullcontext
from typing import Never

import jax
import numpy as np
import pytest

from phydrax._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax.discretization import (
    _cell_complex as owner,
    CellBlock,
    CellMesh,
    reference_cell_topology,
)
from phydrax.discretization._cell_mesh import cell_mesh_construction_bounds


def _old_shells(
    loop_cells: np.ndarray,
    offsets: np.ndarray,
    values: np.ndarray,
    entry_loops: np.ndarray,
    positions: np.ndarray,
    cell_count: int,
) -> np.ndarray:
    """Frozen pre-streaming algorithm, using independent NumPy row sorting."""
    sizes = np.diff(offsets)
    stops = values[offsets[entry_loops] + (positions + 1) % sizes[entry_loops]]
    sides = np.stack(
        (loop_cells[entry_loops], np.minimum(values, stops), np.maximum(values, stops)),
        axis=1,
    )
    order = np.lexsort(sides.T[::-1])
    ordered = sides[order]
    first = (
        np.flatnonzero(
            np.concatenate(([True], np.any(ordered[1:] != ordered[:-1], axis=1)))
        )
        if order.size
        else np.empty(0, dtype=np.int64)
    )
    run_lengths = np.diff(np.append(first, order.size))
    second = np.minimum(first + 1, order.size - 1)
    opposed = (run_lengths == 2) & (values[order[first]] != values[order[second]])
    return np.bincount(sides[order[first[~opposed]], 0], minlength=cell_count) > 0


def _cases(
    case: str,
) -> tuple[tuple[tuple[np.ndarray, ...] | tuple[str, np.ndarray], ...], int]:
    tetra = tuple(
        np.asarray(face, dtype=np.int32)
        for face in reference_cell_topology("tetrahedron").entities[2]
    )
    if case == "standard":
        return (("hexahedron", np.arange(16, dtype=np.int32).reshape(2, 8)),), 16
    if case == "mixed":
        return (
            ("hexahedron", np.arange(8, dtype=np.int32)[None]),
            tuple(face + 8 for face in tetra),
        ), 12
    if case == "reversed":
        return (tuple(face[::-1] for face in tetra), tetra), 4
    if case == "duplicate":
        return ((tetra[0], tetra[0], *tetra[2:]),), 4
    if case == "open":
        return ((tetra[0][::-1], *tetra[1:]),), 4
    if case == "nonmanifold":
        return (tetra, tetra, tetra), 4
    if case == "precedence":
        return ((tetra[0][::-1], *tetra[1:]), (np.asarray((-1, 1, 2)), *tetra[1:])), 4
    if case == "short":
        return ((tetra[0],), tetra), 4
    return (tetra, tuple(face + 4 for face in tetra)), 8


@pytest.mark.parametrize("chunk_cells", (1, 2, 3, 256))
@pytest.mark.parametrize(
    "case",
    (
        "explicit",
        "standard",
        "mixed",
        "reversed",
        "duplicate",
        "open",
        "nonmanifold",
        "precedence",
        "short",
    ),
)
def test_shell_chunks_preserve_old_connectivity_and_first_defect(
    monkeypatch: pytest.MonkeyPatch, chunk_cells: int, case: str
) -> None:
    cells, vertices = _cases(case)
    streamed = owner._open_shell_cells
    monkeypatch.setattr(owner, "_open_shell_cells", _old_shells)
    expected: owner.PolyhedralConnectivity | tuple[type[ValueError], str]
    try:
        expected = owner.polyhedral_connectivity(cells, vertices)
    except ValueError as error:
        expected = (type(error), str(error))
    monkeypatch.setattr(
        owner,
        "_open_shell_cells",
        lambda *args: streamed(*args, chunk_cells=chunk_cells),
    )
    if isinstance(expected, tuple):
        with pytest.raises(expected[0]) as caught:
            owner.polyhedral_connectivity(cells, vertices)
        assert str(caught.value) == expected[1]
    else:
        actual = owner.polyhedral_connectivity(cells, vertices)
        assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
            expected
        )
        for left, right in zip(
            jax.tree_util.tree_leaves(actual),
            jax.tree_util.tree_leaves(expected),
            strict=True,
        ):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("cell_count", (1, 255, 256, 257, 1025))
def test_shell_chunk_work_and_scratch_are_bounded(
    monkeypatch: pytest.MonkeyPatch, cell_count: int
) -> None:
    loops = owner._polyhedral_face_loops(
        (("hexahedron", np.arange(cell_count * 8, dtype=np.int32).reshape(-1, 8)),),
        cell_count * 8,
    )
    offsets = np.concatenate(([0], np.cumsum(loops.loop_lengths)))
    loop_cells = np.repeat(np.arange(cell_count), loops.cell_face_counts)
    entry_loops = np.repeat(np.arange(loops.loop_lengths.size), loops.loop_lengths)
    positions = np.arange(loops.loop_values.size) - offsets[entry_loops]
    original = owner._open_shell_chunk
    spans = []
    retained = []

    def capture(
        local_cells: np.ndarray,
        local_offsets: np.ndarray,
        values: np.ndarray,
        local_entries: np.ndarray,
        local_positions: np.ndarray,
        count: int,
    ) -> np.ndarray:
        assert all(reference() is None for reference in retained)
        retained[:] = [
            weakref.ref(array) for array in (local_cells, local_offsets, local_entries)
        ]
        spans.append((values.size, count))
        return original(
            local_cells, local_offsets, values, local_entries, local_positions, count
        )

    monkeypatch.setattr(owner, "_open_shell_chunk", capture)
    result = owner._open_shell_cells(
        loop_cells, offsets, loops.loop_values, entry_loops, positions, cell_count
    )
    np.testing.assert_array_equal(
        result,
        _old_shells(
            loop_cells, offsets, loops.loop_values, entry_loops, positions, cell_count
        ),
    )
    assert result.flags.owndata
    assert sum(entries for entries, _ in spans) == 24 * cell_count
    assert max(entries for entries, _ in spans) <= 24 * owner._OPEN_SHELL_CHUNK_CELLS
    assert (
        len(spans)
        == (cell_count + owner._OPEN_SHELL_CHUNK_CELLS - 1)
        // owner._OPEN_SHELL_CHUNK_CELLS
    )
    assert all(reference() is None for reference in retained)


def test_shell_chunk_failure_releases_private_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained = []

    def fail(
        local_cells: np.ndarray,
        local_offsets: np.ndarray,
        values: np.ndarray,
        local_entries: np.ndarray,
        local_positions: np.ndarray,
        count: int,
    ) -> Never:
        retained.extend(
            weakref.ref(array) for array in (local_cells, local_offsets, local_entries)
        )
        raise RuntimeError("injected shell failure")

    monkeypatch.setattr(owner, "_open_shell_chunk", fail)
    try:
        owner._open_shell_cells(
            np.asarray([0]),
            np.asarray([0, 3]),
            np.asarray([0, 1, 2]),
            np.asarray([0, 0, 0]),
            np.asarray([0, 1, 2]),
            1,
        )
    except RuntimeError as error:
        assert str(error) == "injected shell failure"
    else:
        pytest.fail("injected failure was not propagated")
    assert all(reference() is None for reference in retained)


@pytest.mark.meshcore
@pytest.mark.parametrize("cell_count", (2, 258))
@pytest.mark.parametrize("short", (False, True))
def test_polyhedral_construction_exact_storage_admission(
    cell_count: int,
    short: bool,
) -> None:
    """Admit all chunk scratch/output once, before entering the real builder."""
    reference = np.asarray(reference_cell_topology("hexahedron").vertices)
    coordinates = np.concatenate(
        [reference + (2 * index, 0, 0) for index in range(cell_count)]
    )
    rows = np.arange(8 * cell_count, dtype=np.int32).reshape(-1, 8)
    blocks = (
        CellBlock("first", "hexahedron", rows[:1]),
        CellBlock(
            "rest",
            "hexahedron",
            rows[1:],
            global_ids=np.arange(1, cell_count, dtype=np.int64),
        ),
    )
    required = cell_mesh_construction_bounds(
        8 * cell_count,
        3,
        (("hexahedron", 1, 8), ("hexahedron", cell_count - 1, 8)),
    )
    scope = NativeExecutionScope(
        NativeExecutionLimits(1 << 30, 1 << 26, 1 << 24, 1 << 30, math.inf)
    )
    reached = False
    expected = pytest.raises(MeshcoreError) if short else nullcontext()
    with expected as refusal, scope:
        with scope.host_workspace() as workspace, scope.host_workspace() as ballast:
            workspace.reserve_empty()
            ballast.reserve_empty()
            available = scope.remaining().remaining_managed_bytes
            ballast.set_bound(available - required + int(short))
            assert scope.remaining().remaining_managed_bytes == required - int(short)
            workspace.set_bound(required)
            reached = True
            result = CellMesh(coordinates, blocks)
        # Workspace closure must not invalidate any published connectivity.
        assert isinstance(result.connectivity, owner.PolyhedralConnectivity)
        np.testing.assert_array_equal(
            np.asarray(result.connectivity.cell_vertex_values),
            rows.reshape(-1),
        )
    assert reached is not short
    if short:
        assert refusal is not None
        assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
