#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.domain import (
    Fixed,
    FixedEnd,
    IrregularTrajectoryDatasetDomain,
    ProductStructure,
)
from phydrax.domain._trajectory_dataset import (
    TRAJECTORY_CASE_INDEX_KEY,
    TRAJECTORY_TIME_INDEX_KEY,
)


def _make_domain(*, sampling="observation_uniform"):
    inputs = jnp.asarray([[0.0], [1.0], [2.0]])
    times = jnp.asarray(
        [
            [0.0, 0.1, 0.4, 0.0],
            [0.2, 0.5, 1.4, 2.0],
            [-0.1, 0.3, 0.0, 0.0],
        ]
    )
    lengths = jnp.asarray([3, 4, 2])
    return IrregularTrajectoryDatasetDomain(
        inputs,
        times,
        lengths,
        sampling=sampling,
    )


def test_irregular_trajectory_observation_uniform_samples_stored_times():
    domain = _make_domain(sampling="observation_uniform")
    batch = domain.component().sample(
        16,
        structure=ProductStructure((("data", "t"),)),
        key=jr.key(0),
    )

    case_indices = jnp.asarray(batch[TRAJECTORY_CASE_INDEX_KEY].data, dtype=jnp.int32)
    time_indices = jnp.asarray(batch[TRAJECTORY_TIME_INDEX_KEY].data, dtype=jnp.int32)
    expected = domain.times[case_indices, time_indices]

    assert jnp.allclose(batch["t"].data, expected)
    assert jnp.allclose(batch["data"].data[:, 0], domain.inputs[case_indices, 0])


def test_irregular_trajectory_fixed_end_is_row_specific():
    domain = _make_domain()
    batch = domain.component({"t": FixedEnd()}).sample(
        12,
        structure=ProductStructure((("data", "t"),)),
        key=jr.key(1),
    )

    case_indices = jnp.asarray(batch[TRAJECTORY_CASE_INDEX_KEY].data, dtype=jnp.int32)
    assert jnp.allclose(batch["t"].data, domain.end_times[case_indices])


def test_irregular_trajectory_fixed_time_samples_only_valid_cases():
    domain = _make_domain()
    batch = domain.component({"t": Fixed(0.6)}).sample(
        10,
        structure=ProductStructure((("data", "t"),)),
        key=jr.key(2),
    )

    case_indices = jnp.asarray(batch[TRAJECTORY_CASE_INDEX_KEY].data, dtype=jnp.int32)
    assert jnp.all(case_indices == 1)
    assert jnp.allclose(batch["t"].data, 0.6)


def test_irregular_trajectory_rejects_non_increasing_valid_times():
    inputs = jnp.asarray([[0.0]])
    times = jnp.asarray([[0.0, 0.2, 0.1]])
    lengths = jnp.asarray([3])

    with pytest.raises(ValueError, match="strictly increasing"):
        IrregularTrajectoryDatasetDomain(inputs, times, lengths)
