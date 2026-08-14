#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.domain import (
    RAGGED_SERIES_INDEX_KEY,
    RaggedSeriesDatasetDomain,
    SampleLayout,
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
        structure=SampleLayout((("data",),)),
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
        batch["data"]["sample_index"].data[0],
        jnp.asarray([0, 1, 2, 2]),
    )
    assert jnp.allclose(batch["data"]["sample_scale"].data, jnp.asarray([1.0, 1.0]))
    assert jnp.allclose(
        batch["data"]["time"].data[0],
        jnp.asarray([0.25, 0.75, 1.25, 1.75]),
    )
    assert jnp.allclose(
        batch["data"]["mask"].data,
        jnp.asarray([[True, True, True, False], [True, True, False, False]]),
    )
    assert jnp.allclose(batch[RAGGED_SERIES_INDEX_KEY].data, indices)


def test_ragged_series_dataset_packed_storage_matches_valid_rows():
    domain = _domain()

    assert jnp.allclose(domain.offsets, jnp.asarray([0, 2, 5, 6], dtype=jnp.int32))
    assert domain.total_observations == 6
    assert domain.series_packed.shape == (6, 2)
    assert jnp.allclose(
        domain.series_packed,
        jnp.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ]
        ),
    )


def test_ragged_series_dataset_samples_from_component():
    domain = _domain()
    batch = domain.component().sample(
        phx.domain.PointSampling(5, layout=SampleLayout((("data",),))), key=jr.key(0)
    )
    axis = batch.structure.axis_for("data")

    assert axis is not None
    assert batch["data"]["series"].data.shape == (5, domain.max_length, 2)
    assert batch["data"]["mask"].data.shape == (5, domain.max_length)


def test_ragged_series_dataset_points_uniform_samples_fixed_width_valid_views():
    domain = _domain()
    batch = domain.sampled_points_from_indices(
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        num_series_points=3,
        sampling="points_uniform",
        structure=SampleLayout((("data",),)),
        key=jr.key(11),
    )

    assert batch["data"]["series"].data.shape == (3, 3, 2)
    assert batch["data"]["time"].data.shape == (3, 3)
    assert batch["data"]["sample_index"].data.shape == (3, 3)
    assert jnp.all(batch["data"]["sample_index"].data < domain.lengths[:, None])
    assert jnp.allclose(
        batch["data"]["mask"].data,
        jnp.asarray(
            [
                [True, True, False],
                [True, True, True],
                [True, False, False],
            ]
        ),
    )
    assert jnp.allclose(
        batch["data"]["sample_scale"].data,
        jnp.asarray([1.0, 1.0, 1.0]),
    )


def test_ragged_series_dataset_window_prefix_and_suffix_sampling():
    domain = _domain()
    indices = jnp.asarray([1, 2], dtype=jnp.int32)
    window = domain.sampled_points_from_indices(
        indices,
        num_series_points=2,
        sampling="window_uniform",
        structure=SampleLayout((("data",),)),
        key=jr.key(12),
    )
    prefix = domain.sampled_points_from_indices(
        indices,
        num_series_points=2,
        sampling="prefix",
        structure=SampleLayout((("data",),)),
        key=jr.key(12),
    )
    suffix = domain.sampled_points_from_indices(
        indices,
        num_series_points=2,
        sampling="suffix",
        structure=SampleLayout((("data",),)),
        key=jr.key(12),
    )

    window_idx = window["data"]["sample_index"].data
    assert jnp.all(jnp.diff(window_idx[0]) == 1)
    assert jnp.allclose(
        prefix["data"]["sample_index"].data, jnp.asarray([[0, 1], [0, 0]])
    )
    assert jnp.allclose(
        suffix["data"]["sample_index"].data, jnp.asarray([[1, 2], [0, 0]])
    )
    assert jnp.allclose(prefix["data"]["mask"].data[1], jnp.asarray([True, False]))


def test_ragged_series_sampled_views_do_not_allocate_global_max_length():
    static = jnp.zeros((2, 1))
    series = jnp.zeros((2, 10_000, 3))
    series = series.at[0, :2, :].set(1.0)
    series = series.at[1, :, :].set(2.0)
    domain = RaggedSeriesDatasetDomain(
        series,
        jnp.asarray([2, 10_000], dtype=jnp.int32),
        static=static,
    )

    sampled = domain.sampled_points_from_indices(
        jnp.asarray([0, 1], dtype=jnp.int32),
        num_series_points=7,
        sampling="window_uniform",
        structure=SampleLayout((("data",),)),
        key=jr.key(13),
    )
    full = domain.points_from_indices(
        jnp.asarray([0, 1], dtype=jnp.int32),
        structure=SampleLayout((("data",),)),
    )

    assert sampled["data"]["series"].data.shape == (2, 7, 3)
    assert sampled["data"]["mask"].data.shape == (2, 7)
    assert full["data"]["series"].data.shape == (2, 10_000, 3)


def test_ragged_series_dataset_from_sequences_builds_equivalent_domain():
    static = jnp.asarray([[1.0], [2.0]])
    seq = (
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        jnp.asarray([[5.0, 6.0]]),
    )
    domain = RaggedSeriesDatasetDomain.from_sequences(seq, static=static, dt=0.25)
    batch = domain.points_from_indices(
        jnp.asarray([0, 1], dtype=jnp.int32),
        structure=SampleLayout((("data",),)),
    )

    assert jnp.allclose(domain.lengths, jnp.asarray([2, 1], dtype=jnp.int32))
    assert domain.max_length == 2
    assert jnp.allclose(batch["data"]["series"].data[1, 1], jnp.zeros((2,)))
    assert not bool(batch["data"]["mask"].data[1, 1])


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

    assert domain.same_support(same)
    assert not domain.same_support(different_lengths)
    assert not domain.same_support(different_dt)


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
