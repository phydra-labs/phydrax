from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spatial import (
    hilbert_encode_integer,
    morton_decode_integer,
    morton_encode_integer,
    MortonAddressPlan,
)


def test_morton_scenario_1() -> None:
    for dimension, depth in [(1, 8), (2, 7), (3, 6)]:
        resolution = 1 << depth
        coordinates = jnp.asarray(
            [
                [0] * dimension,
                [resolution - 1] * dimension,
                [axis + 1 for axis in range(dimension)],
            ],
            dtype=jnp.int64,
        )
        codes = morton_encode_integer(coordinates, depth)
        decoded = morton_decode_integer(codes, dimension, depth)
        np.testing.assert_array_equal(decoded, coordinates)
    plan = MortonAddressPlan(
        (0.0, -1.0),
        (1.0, 1.0),
        4,
        periodic_axes=(True, False),
    )
    encoded = plan.encode(
        jnp.asarray(
            [
                [0.0, -1.0],
                [1.0, 0.0],
                [0.5, 1.0],
                [jnp.nan, 0.0],
            ]
        )
    )
    np.testing.assert_array_equal(encoded.in_domain, [True, True, False, False])
    np.testing.assert_allclose(encoded.coordinates[1], [0.0, 0.0])
    np.testing.assert_array_equal(encoded.integer_coordinates[2:], 0)
    plan = MortonAddressPlan((0.0, 0.0, 0.0), (2.0, 4.0, 8.0), 3)
    coordinates = jnp.asarray([[6, 2, 5]], dtype=jnp.int64)
    code = morton_encode_integer(coordinates, 3)
    prefix = plan.prefix(code, 2)
    start, end = plan.descendant_interval(prefix, jnp.asarray([2]))
    assert bool((start <= code)[0] & (code < end)[0])
    geometry = plan.cell_geometry(prefix, jnp.asarray([2]))
    np.testing.assert_allclose(geometry.upper - geometry.lower, [[0.5, 1.0, 2.0]])
    plan = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 8)
    encode = eqx.filter_jit(plan.encode)
    encoded = encode(jnp.asarray([[0.25, 0.75], [0.75, 0.25]]))
    assert bool(encoded.successful)
    np.testing.assert_array_equal(plan.decode(encoded.codes), encoded.integer_coordinates)
    for dimension, depth in [(1, 5), (2, 4), (3, 3)]:
        resolution = 1 << depth
        grid = np.stack(
            np.meshgrid(*([np.arange(resolution)] * dimension), indexing="ij"), axis=-1
        ).reshape(-1, dimension)
        codes = np.asarray(hilbert_encode_integer(jnp.asarray(grid), depth))
        np.testing.assert_array_equal(np.sort(codes), np.arange(resolution**dimension))
        walk = grid[np.argsort(codes)]
        np.testing.assert_array_equal(walk[0], np.zeros((dimension,)))
        np.testing.assert_array_equal(np.sum(np.abs(np.diff(walk, axis=0)), axis=1), 1)
        with pytest.raises(ValueError, match="code budget"):
            hilbert_encode_integer(jnp.asarray(grid), 63 // dimension + 1)
