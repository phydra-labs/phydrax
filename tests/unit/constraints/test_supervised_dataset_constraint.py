#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.constraints import SupervisedDatasetConstraint
from phydrax.domain import DatasetDomain, SampleLayout


def test_supervised_dataset_constraint_supervises_scalar_targets_exactly():
    data = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
    domain = DatasetDomain(data)
    targets = data[:, 0] + 2.0 * data[:, 1]

    @domain.Function("data")
    def u(row):
        return row[0] + 2.0 * row[1]

    constraint = SupervisedDatasetConstraint(
        "u",
        domain.component(),
        targets,
        sampling=phx.domain.PointSampling(12, design="uniform"),
        label="dataset_data",
    )

    loss = constraint.loss({"u": u}, key=jr.key(0))
    metrics = constraint.data_metrics({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0, atol=1e-12)
    assert jnp.allclose(metrics["data_accuracy"], 1.0)
    assert jnp.allclose(metrics["data_relative_l2_error"], 0.0)


def test_supervised_dataset_constraint_supervises_vector_targets_exactly():
    data = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
    domain = DatasetDomain(data)
    targets = jnp.stack((data[:, 0] + data[:, 1], data[:, 0] - data[:, 1]), axis=-1)

    @domain.Function("data")
    def theta(row):
        return jnp.asarray([row[0] + row[1], row[0] - row[1]])

    constraint = SupervisedDatasetConstraint(
        "theta",
        domain.component(),
        targets,
        sampling=phx.domain.PointSampling(12, design="uniform"),
        reduction="sum",
    )

    loss = constraint.loss({"theta": theta}, key=jr.key(1))
    metrics = constraint.data_metrics({"theta": theta}, key=jr.key(1))
    assert jnp.allclose(loss, 0.0, atol=1e-12)
    assert jnp.allclose(metrics["data_rmse"], 0.0)


def test_supervised_dataset_constraint_aligns_sampled_indices_with_targets():
    data = jnp.arange(10.0, dtype=float).reshape((5, 2))
    domain = DatasetDomain(data)
    targets = jnp.asarray([10.0, 20.0, 30.0, 40.0, 50.0])
    constraint = SupervisedDatasetConstraint(
        "u",
        domain.component(),
        targets,
        sampling=phx.domain.PointSampling(8, design="uniform"),
    )

    batch = constraint.sample(key=jr.key(2))
    assert batch.target.shape == (8,)
    assert jnp.allclose(batch.target, targets[batch.indices])
    assert jnp.allclose(batch.points["data"].data, data[batch.indices])


def test_supervised_dataset_constraint_samples_only_index_subset():
    data = jnp.arange(12.0, dtype=float).reshape((6, 2))
    domain = DatasetDomain(data)
    targets = data[:, 0]
    allowed = jnp.asarray([1, 3, 5], dtype=jnp.int32)
    constraint = SupervisedDatasetConstraint(
        "u",
        domain.component(),
        targets,
        sampling=phx.domain.PointSampling(16, design="uniform"),
        indices=allowed,
    )

    batch = constraint.sample(key=jr.key(8))
    assert jnp.all(jnp.isin(batch.indices, allowed))
    assert jnp.allclose(batch.target, targets[batch.indices])


def test_supervised_dataset_constraint_supports_pytree_rows():
    rows = {
        "a": jnp.asarray([[0.0], [1.0], [2.0]]),
        "b": jnp.asarray([1.0, 2.0, 4.0]),
    }
    domain = DatasetDomain(rows)
    targets = rows["a"][:, 0] + rows["b"]

    @domain.Function("data")
    def u(row):
        return row["a"][0] + row["b"]

    constraint = SupervisedDatasetConstraint(
        "u",
        domain.component(),
        targets,
        sampling=phx.domain.PointSampling(12, layout=SampleLayout((("data",),)), design="uniform"),
    )

    loss = constraint.loss({"u": u}, key=jr.key(3))
    assert jnp.allclose(loss, 0.0, atol=1e-12)


def test_supervised_dataset_constraint_validates_targets_and_sampling():
    domain = DatasetDomain(jnp.zeros((3, 2), dtype=float))

    with pytest.raises(ValueError, match="leading axis"):
        SupervisedDatasetConstraint(
            "u",
            domain.component(),
            jnp.zeros((4,), dtype=float),
            sampling=phx.domain.PointSampling(2, design="uniform"),
        )

    with pytest.raises(ValueError, match="sampling count"):
        SupervisedDatasetConstraint(
            "u",
            domain.component(),
            jnp.zeros((3,), dtype=float),
            sampling=phx.domain.PointSampling(0, design="uniform"),
        )

    with pytest.raises(ValueError, match="indices"):
        SupervisedDatasetConstraint(
            "u",
            domain.component(),
            jnp.zeros((3,), dtype=float),
            sampling=phx.domain.PointSampling(2, design="uniform"),
            indices=jnp.asarray([3], dtype=jnp.int32),
        )
