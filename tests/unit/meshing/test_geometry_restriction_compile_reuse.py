from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import ArrayLike, DTypeLike

from phydrax.discretization._cell_geometry import coordinate_lagrange_element
from phydrax.meshing._collective_geometry import (
    _GeometryActions,
    _GeometryBlockRestriction,
    _GeometrySourceBlock,
    _prepare_geometry_block_restriction,
)
from tools.compile_counter import CompileCounter


def _array(value: ArrayLike, dtype: DTypeLike) -> Array:
    return jax.device_put(np.asarray(value, dtype=dtype))


@pytest.mark.parametrize("changed_count", [False, True])
def test_geometry_restriction_reuses_bucket_exactly(changed_count: bool) -> None:
    block = _GeometrySourceBlock(
        "surface",
        "triangle",
        _array([20, 10], np.int64),
        _array([[3, 4, 5], [0, 1, 2]], np.int64),
        _array([[103, 104, 105], [100, 101, 102]], np.int64),
        coordinate_lagrange_element("triangle", 1),
    )
    coordinate_ids = _array(np.arange(100, 106), np.int64)
    owners = _array([2] * 6, np.int32)
    sources = _array([10, 20, -1, -1], np.int64)
    cells = _array([10, 20, -1, -1], np.int64)
    cell_owners = _array([0, 1, 2, 2], np.int32)
    corners = _array([[0, 1, 2], [3, 4, 5], [-1] * 3, [-1] * 3], np.int64)
    weights_host = np.arange(36, dtype=np.float64).reshape(4, 1, 3, 3) / 8
    weights_host[0, 0, 0, 0] = -0.0
    weights = _array(weights_host, np.float64)
    first_actions = _GeometryActions(corners, _array([0] * 4, np.int32), weights)
    restored = jnp.asarray(True)

    def prepare(
        count: int, actions: _GeometryActions, targets: Array
    ) -> _GeometryBlockRestriction:
        return _prepare_geometry_block_restriction(
            block,
            sources,
            coordinate_ids,
            owners,
            targets,
            cell_owners,
            count,
            2,
            actions,
            restored,
        )

    first = prepare(2, first_actions, cells)
    jax.block_until_ready(first)
    next_weights_host = weights_host * -2.0
    next_actions = _GeometryActions(
        corners,
        _array([0, 0 if changed_count else 1, 0, 0], np.int32),
        _array(next_weights_host, np.float64),
    )
    next_cells = _array([10, 21, -1, -1], np.int64)
    with CompileCounter() as compiler:
        repeated = prepare(2, first_actions, cells)
        changed = prepare(1 if changed_count else 2, next_actions, next_cells)
        jax.block_until_ready((repeated, changed))
    assert compiler.count == 0
    assert bool(first.restored)
    assert bool(repeated.restored)
    assert not bool(changed.restored)
    first_arrays = dict(first.arrays)
    changed_arrays = dict(changed.arrays)
    expected_weights = next_weights_host[:2].copy()
    if changed_count:
        expected_weights[1] = 0.0
    for actual, expected in (
        (first_arrays["geometry/action_weights/surface"], weights_host[:2]),
        (changed_arrays["geometry/action_weights/surface"], expected_weights),
    ):
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint64), expected.view(np.uint64)
        )
    np.testing.assert_array_equal(
        changed_arrays["geometry/cell_ids/surface"], [10, -1 if changed_count else 21]
    )
    np.testing.assert_array_equal(
        changed_arrays["geometry/routes/surface"],
        [[100, 101, 102], [-1] * 3 if changed_count else [103, 104, 105]],
    )
    np.testing.assert_array_equal(
        changed.coordinate_owners, [0, 0, 0, *([2] * 3 if changed_count else [1] * 3)]
    )
    assert dict(changed.shapes)["geometry/action_weights/surface"] == (
        1 if changed_count else 2,
        1,
        3,
        3,
    )

    invalid_block = _GeometrySourceBlock(
        block.name,
        block.cell_kind,
        block.cell_ids,
        block.vertex_ids,
        block.routes,
        coordinate_lagrange_element("triangle", 2),
    )
    with pytest.raises(ValueError, match="P1 cardinal vertex coefficient bank"):
        _prepare_geometry_block_restriction(
            invalid_block,
            sources,
            coordinate_ids,
            owners,
            cells,
            cell_owners,
            2,
            2,
            first_actions,
            restored,
        )
