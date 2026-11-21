#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(),
        reason="native phydrax-meshcore library is not available",
    ),
]


def _cube_grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    """An n^3 grid of unit cubes, six positively oriented tetrahedra per cube."""
    axis = np.arange(n + 1, dtype=np.float64)
    vertices = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        (-1, 3)
    )

    def index(i: int, j: int, k: int) -> int:
        return (i * (n + 1) + j) * (n + 1) + k

    cells = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for path in (
                    (0, 1, 2),
                    (0, 2, 1),
                    (1, 0, 2),
                    (1, 2, 0),
                    (2, 0, 1),
                    (2, 1, 0),
                ):
                    at = [i, j, k]
                    cell = [index(*at)]
                    for direction in path:
                        at[direction] += 1
                        cell.append(index(*at))
                    corners = vertices[cell]
                    if np.linalg.det(corners[1:] - corners[0]) < 0.0:
                        cell[2], cell[3] = cell[3], cell[2]
                    cells.append(cell)
    return vertices, np.asarray(cells, dtype=np.int32)


def _locator(n: int = 3) -> Any:
    vertices, cells = _cube_grid(n)
    mesh = phx.discretization.CellMesh(
        jnp.asarray(vertices),
        (phx.discretization.CellBlock("tet", "tetrahedron", jnp.asarray(cells)),),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("tetrahedron", 1)
        ),
    ).prepare()
    return phx.discretization.PreparedSimplicialCellLocator(
        phx.discretization.fem.prepare_finite_element_cell_map(discretization, 0),
        discretization.default_runtime.coordinates,
        phx.discretization.SimplicialLocationPolicy(64, 8, 4),
    )


def _points() -> np.ndarray:
    random = np.random.default_rng(4).uniform(-0.3, 3.3, (300, 3))
    # Shared vertices, edges, and faces tie between cells; one point is nonfinite.
    ties = np.asarray(
        ((1.0, 1.0, 1.0), (1.5, 1.0, 1.0), (0.5, 0.5, 0.5), (2.0, 2.5, 0.5))
    )
    return np.concatenate((random, ties, np.asarray(((np.nan, 0.0, 0.0),))))


def _accepted(cells: Any) -> list[list[int]]:
    return [sorted(int(cell) for cell in row if cell >= 0) for row in np.asarray(cells)]


def _assert_identical(native: Any, traced: Any) -> None:
    """Equal shapes, dtypes, and bits (signed zeros included)."""
    first, second = np.asarray(native), np.asarray(traced)
    assert first.shape == second.shape and first.dtype == second.dtype
    np.testing.assert_array_equal(
        first.view(np.uint8).reshape(first.shape + (-1,)),
        second.view(np.uint8).reshape(second.shape + (-1,)),
    )


@pytest.mark.parametrize("masked", (False, True), ids=("all-cells", "cell-mask"))
def test_native_affine_location_equals_traced_route(masked: bool) -> None:
    locator = _locator()
    points = _points()
    mask = None
    if masked:
        mask = jnp.asarray(np.arange(locator.cell_count) % 3 != 0)
    native = locator.locate(points, cell_mask=mask)
    traced = jax.jit(lambda values: locator.locate(values, cell_mask=mask))(points)

    for field in (
        "status",
        "cell_ids",
        "inside",
        "successful",
        "candidate_count",
        "barycentric",
        "reference_coordinates",
        "geometry_residual",
        "iterations",
        "jacobian_condition",
        "used_fallback",
        "candidates_complete",
    ):
        _assert_identical(getattr(native, field), getattr(traced, field))
    # Both list every accepting cell ascending; only the padding width differs.
    assert _accepted(native.candidate_cells) == _accepted(traced.candidate_cells)
    inside = np.asarray(native.inside)
    assert np.all(np.asarray(native.geometry_residual)[inside] < 1e-12)
    assert not np.any(np.asarray(native.used_fallback))


def test_affine_location_without_native_library_equals_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax import _bvh_native

    locator = _locator()
    points = _points()
    native = locator.locate(points)
    monkeypatch.setattr(_bvh_native, "meshcore_available", lambda: False)
    eager = locator.locate(points)
    for field in ("status", "cell_ids", "barycentric", "geometry_residual"):
        _assert_identical(getattr(native, field), getattr(eager, field))
    assert _accepted(native.candidate_cells) == _accepted(eager.candidate_cells)


def test_native_affine_location_statuses() -> None:
    locator = _locator(2)
    located = locator.locate(
        np.asarray(((1.0, 1.0, 1.0), (5.0, 0.0, 0.0), (np.inf, 0.0, 0.0)))
    )
    statuses = phx.discretization.CellLocationStatus
    assert np.asarray(located.status).tolist() == [
        statuses.LOCATED,
        statuses.OUTSIDE,
        statuses.NONFINITE,
    ]
    # The interior grid vertex lies in 24 of the cells around it; the lowest wins.
    assert int(located.candidate_count[0]) == 24
    assert int(located.cell_ids[0]) == min(_accepted(located.candidate_cells)[0])
    assert np.asarray(located.cell_ids[1:]).tolist() == [-1, -1]
