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
from phydrax.domain import HyperRectangle, PeriodicIdentification, TimeInterval


_BOX = HyperRectangle(np.asarray([0.0, -1.0]), np.asarray([2.0, 3.0]))
_SEAM_X = PeriodicIdentification(_BOX, "x", component=0)


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


def test_identification_derived_address_binds_bounds_mask_and_seam_identity() -> None:
    address = MortonAddressPlan.from_periodic_identifications((_SEAM_X,), maximum_depth=8)
    assert (address.lower, address.upper) == ((0.0, -1.0), (2.0, 3.0))
    assert address.periodic_axes == (True, False)
    assert address.coordinates == (("x", 0), ("x", 1))
    assert address.identifications == (_SEAM_X.revision, None)
    # Half-open: the upper face of the identified coordinate is the lower face.
    encoded = address.encode(jnp.asarray([[2.0, 0.0], [0.0, 0.0]]))
    assert bool(encoded.successful)
    np.testing.assert_array_equal(encoded.codes[0], encoded.codes[1])
    renamed = PeriodicIdentification(_BOX, "x", component=0, identification_id="other")
    other = MortonAddressPlan.from_periodic_identifications((renamed,), maximum_depth=8)
    raw = MortonAddressPlan((0.0, -1.0), (2.0, 3.0), 8, periodic_axes=(True, False))
    assert (other.lower, other.upper, other.periodic_axes) == (
        address.lower,
        address.upper,
        address.periodic_axes,
    )
    assert len({address.plan_id, other.plan_id, raw.plan_id}) == 3
    transposed = MortonAddressPlan.from_periodic_identifications(
        (_SEAM_X,), maximum_depth=8, coordinates=(("x", 1), ("x", 0))
    )
    assert transposed.periodic_axes == (False, True)
    assert (transposed.lower, transposed.upper) == ((-1.0, 0.0), (3.0, 2.0))


@pytest.mark.parametrize(
    ("identifications", "coordinates", "message"),
    [
        pytest.param((), None, "at least one", id="empty"),
        pytest.param(
            (
                _SEAM_X,
                PeriodicIdentification(_BOX, "x", component=0, identification_id="x"),
            ),
            None,
            "identified more than once",
            id="duplicate-binding",
        ),
        pytest.param(
            (
                _SEAM_X,
                PeriodicIdentification(
                    HyperRectangle(np.zeros(2), np.ones(2)), "x", component=1
                ),
            ),
            None,
            "one fundamental domain",
            id="cross-domain",
        ),
        pytest.param(
            (PeriodicIdentification(_BOX @ TimeInterval(0.0, 1.0), "x", component=0),),
            None,
            "explicit point coordinates",
            id="multi-label-domain",
        ),
        pytest.param((_SEAM_X,), (("x", 1),), "not a point axis", id="unbound-seam"),
    ],
)
def test_identification_derived_address_refuses_ambiguous_bindings(
    identifications: tuple[PeriodicIdentification, ...],
    coordinates: tuple[tuple[str, int | None], ...] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MortonAddressPlan.from_periodic_identifications(
            identifications, maximum_depth=8, coordinates=coordinates
        )
