#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.data_utils import kfold_indices, train_test_split_indices


def test_train_test_split_indices_partition_cases():
    train, test = train_test_split_indices(10, test_fraction=0.2, key=jr.key(0))

    assert train.shape == (8,)
    assert test.shape == (2,)
    merged = jnp.sort(jnp.concatenate((train, test)))
    assert jnp.all(merged == jnp.arange(10, dtype=jnp.int32))


def test_train_test_split_indices_can_skip_shuffle():
    train, test = train_test_split_indices(5, test_fraction=0.4, shuffle=False)

    assert jnp.all(test == jnp.asarray([0, 1], dtype=jnp.int32))
    assert jnp.all(train == jnp.asarray([2, 3, 4], dtype=jnp.int32))


def test_kfold_indices_cover_each_case_once_as_validation():
    folds = kfold_indices(7, 3, key=jr.key(1))

    validation = jnp.concatenate([fold[1] for fold in folds])
    assert jnp.all(jnp.sort(validation) == jnp.arange(7, dtype=jnp.int32))
    for train, val in folds:
        assert train.shape[0] + val.shape[0] == 7
        assert not bool(jnp.any(jnp.isin(train, val)))


def test_split_helpers_validate_arguments():
    with pytest.raises(ValueError, match="at least 2"):
        train_test_split_indices(1)

    with pytest.raises(ValueError, match="strictly between"):
        train_test_split_indices(4, test_fraction=1.0)

    with pytest.raises(ValueError, match="cannot exceed"):
        kfold_indices(3, 4)
