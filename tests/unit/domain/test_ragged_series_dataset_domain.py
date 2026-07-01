#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.domain import (
    ProductStructure,
    RAGGED_SERIES_INDEX_KEY,
    RaggedSeriesDatasetDomain,
)


def _domain() -> RaggedSeriesDatasetDomain:
    static = jnp.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 4.0]])
    series = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0], [0.0, 0.0]],
            [[11.0, 12.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    lengths = jnp.asarray([2, 3, 1], dtype=jnp.int32)
    return RaggedSeriesDatasetDomain(
        series,
        lengths,
        static=static,
        start=0.25,
        dt=0.5,
    )


def test_ragged_series_dataset_points_from_indices_carries_payload_and_mask():
    domain = _domain()
    indices = jnp.asarray([1, 0], dtype=jnp.int32)
    batch = domain.points_from_indices(
        indices,
        structure=ProductStructure((("data",),)),
    )
    axis = batch.structure.axis_for("data")

    assert axis is not None
    assert batch["data"]["static"].dims == (axis, None)
    assert batch["data"]["series"].dims == (axis, None, None)
    assert batch["data"]["time"].dims == (axis, None)
    assert batch["data"]["mask"].dims == (axis, None)
    assert batch["data"]["length"].dims == (axis,)
    assert jnp.allclose(batch["data"]["length"].data, jnp.asarray([3, 2]))
    assert jnp.allclose(
        batch["data"]["time"].data[0],
        jnp.asarray([0.25, 0.75, 1.25, 1.75]),
    )
    assert jnp.allclose(
        batch["data"]["mask"].data,
        jnp.asarray([[True, True, True, False], [True, True, False, False]]),
    )
    assert jnp.allclose(batch[RAGGED_SERIES_INDEX_KEY].data, indices)


def test_ragged_series_dataset_samples_from_component():
    domain = _domain()
    batch = domain.component().sample(
        5,
        structure=ProductStructure((("data",),)),
        key=jr.key(0),
    )
    axis = batch.structure.axis_for("data")

    assert axis is not None
    assert batch["data"]["series"].data.shape == (5, domain.max_length, 2)
    assert batch["data"]["mask"].data.shape == (5, domain.max_length)


def test_ragged_series_dataset_equivalence_includes_lengths_and_grid():
    domain = _domain()
    same = _domain()
    different_lengths = RaggedSeriesDatasetDomain(
        domain.series,
        jnp.asarray([1, 3, 1], dtype=jnp.int32),
        static=domain.static,
        start=domain.start,
        dt=domain.dt,
    )
    different_dt = RaggedSeriesDatasetDomain(
        domain.series,
        domain.lengths,
        static=domain.static,
        start=domain.start,
        dt=0.25,
    )

    assert domain.equivalent(same)
    assert not domain.equivalent(different_lengths)
    assert not domain.equivalent(different_dt)


def test_ragged_series_dataset_validates_shapes_and_lengths():
    series = jnp.zeros((3, 4, 2))
    static = jnp.zeros((2, 1))

    with pytest.raises(ValueError, match="static data leading case axis"):
        RaggedSeriesDatasetDomain(series, jnp.asarray([2, 3, 1]), static=static)

    with pytest.raises(ValueError, match="positive"):
        RaggedSeriesDatasetDomain(series, jnp.asarray([2, 0, 1]))

    with pytest.raises(ValueError, match="padded time axis"):
        RaggedSeriesDatasetDomain(series, jnp.asarray([2, 5, 1]))

    with pytest.raises(ValueError, match="integer-valued"):
        RaggedSeriesDatasetDomain(series, jnp.asarray([2.0, 2.5, 1.0]))
