from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spatial import (
    MortonAddressPlan,
    SparseVoxelField,
    SparseVoxelGridPlan,
)


def _dense_grid(*, periodic: bool = False):
    address = MortonAddressPlan(
        (0.0, 0.0),
        (1.0, 1.0),
        3,
        periodic_axes=(periodic, periodic),
    )
    coordinates = np.stack(
        np.meshgrid(np.arange(8), np.arange(8), indexing="ij"), axis=-1
    ).reshape((-1, 2))
    return SparseVoxelGridPlan(
        address,
        brick_size=2,
        brick_capacity=16,
    ).prepare(coordinates)


def _coordinate_field(grid):
    centers = grid.voxel_centers()
    values = 2.0 * centers[..., 0] - 3.0 * centers[..., 1] + 0.5
    return SparseVoxelField(grid, values)


def test_sparse_voxel_preparation_deduplicates_and_checks_capacity() -> None:
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 3)
    plan = SparseVoxelGridPlan(address, brick_size=2, brick_capacity=2)
    grid = plan.prepare(jnp.asarray([[0, 0], [0, 0], [1, 1], [4, 4]]))
    assert int(grid.evidence.active_voxels) == 3
    assert int(grid.evidence.active_bricks) == 2
    assert int(grid.evidence.duplicate_voxels) == 1
    with pytest.raises(ValueError, match="requires 3 bricks"):
        plan.prepare(jnp.asarray([[0, 0], [2, 2], [4, 4]]))


def test_sparse_voxel_lookup_and_periodic_wrap() -> None:
    grid = _dense_grid(periodic=True)
    lookup = grid.lookup_integer(jnp.asarray([[0, 0], [7, 7], [8, -1]]))
    assert bool(jnp.all(lookup.supported))
    wrapped = grid.lookup_integer(jnp.asarray([[0, 7]]))
    np.testing.assert_array_equal(lookup.brick_slots[-1], wrapped.brick_slots[0])
    np.testing.assert_array_equal(lookup.local_slots[-1], wrapped.local_slots[0])


def test_sparse_voxel_multilinear_interpolation_is_affine_exact() -> None:
    grid = _dense_grid()
    field = _coordinate_field(grid)
    points = jnp.asarray([[0.25, 0.25], [0.35, 0.6], [0.75, 0.5]])
    result = field.sample_multilinear(points)
    expected = 2.0 * points[:, 0] - 3.0 * points[:, 1] + 0.5
    assert bool(jnp.all(result.supported))
    np.testing.assert_allclose(result.values, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(jnp.sum(result.weights, axis=1), 1.0)


def test_sparse_voxel_missing_support_is_explicit() -> None:
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 3)
    grid = SparseVoxelGridPlan(address, brick_size=2, brick_capacity=1).prepare(
        jnp.asarray([[0, 0]])
    )
    values = jnp.ones((1, 4))
    unsupported = SparseVoxelField(grid, values).sample_nearest(jnp.asarray([[0.8, 0.8]]))
    assert not bool(unsupported.supported[0])
    background = SparseVoxelField(
        grid,
        values,
        background_mode="constant",
        background_value=3.0,
    ).sample_nearest(jnp.asarray([[0.8, 0.8]]))
    assert bool(background.supported[0])
    np.testing.assert_allclose(background.values, [3.0])


def test_sparse_voxel_gather_and_deposit_jit_and_gradient() -> None:
    grid = _dense_grid()
    field = _coordinate_field(grid)
    point = jnp.asarray([[0.35, 0.45]])
    sample = eqx.filter_jit(field.sample_multilinear)(point)
    assert bool(sample.supported[0])
    gradient = jax.grad(lambda value: field.sample_multilinear(value[None, :]).values[0])(
        point[0]
    )
    np.testing.assert_allclose(gradient, [2.0, -3.0], rtol=1.0e-10, atol=1.0e-10)

    amounts = jnp.asarray([2.5])
    deposited = eqx.filter_jit(grid.deposit_multilinear)(point, amounts)
    assert bool(deposited.supported[0])
    np.testing.assert_allclose(jnp.sum(deposited.values), 2.5)
    np.testing.assert_allclose(deposited.weight_sum, [1.0])
