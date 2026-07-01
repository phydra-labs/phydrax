#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.constraints import RaggedSeriesSupervisedConstraint
from phydrax.domain import ProductStructure, RaggedSeriesDatasetDomain


def _domain_and_targets():
    static = jnp.asarray([[1.0, 0.5], [2.0, -1.0], [3.0, 2.0]])
    series = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
            [[11.0, 12.0], [99.0, 99.0], [99.0, 99.0]],
        ]
    )
    lengths = jnp.asarray([2, 3, 1], dtype=jnp.int32)
    domain = RaggedSeriesDatasetDomain(series, lengths, static=static, dt=0.25)
    valid_sum = jnp.asarray([4.0, 21.0, 11.0])
    targets = jnp.stack(
        (
            static[:, 0] + valid_sum,
            static[:, 1] - valid_sum,
        ),
        axis=-1,
    )
    return domain, targets


def test_ragged_series_supervised_constraint_matches_exact_vector_targets():
    domain, targets = _domain_and_targets()

    def exact(payload, *, key=None):
        del key
        series0 = payload.series[..., 0]
        valid_sum = jnp.sum(series0 * payload.mask.astype(series0.dtype), axis=1)
        return jnp.stack(
            (
                payload.static[:, 0] + valid_sum,
                payload.static[:, 1] - valid_sum,
            ),
            axis=-1,
        )

    u = domain.Function("data")(phx.nn.RaggedSeriesModel(exact))
    constraint = RaggedSeriesSupervisedConstraint(
        "u",
        domain.component(),
        targets,
        num_cases=16,
        structure=ProductStructure((("data",),)),
    )

    loss = constraint.loss({"u": u}, key=jr.key(0))
    metrics = constraint.data_metrics({"u": u}, key=jr.key(0))
    assert jnp.allclose(loss, 0.0, atol=1e-12)
    assert jnp.allclose(metrics["data_accuracy"], 1.0)
    assert jnp.allclose(metrics["data_relative_l2_error"], 0.0)


def test_ragged_series_supervised_constraint_samples_index_subset():
    domain, targets = _domain_and_targets()
    allowed = jnp.asarray([0, 2], dtype=jnp.int32)
    constraint = RaggedSeriesSupervisedConstraint(
        "u",
        domain.component(),
        targets,
        num_cases=20,
        indices=allowed,
    )

    batch = constraint.sample(key=jr.key(1))
    assert jnp.all(jnp.isin(batch.indices, allowed))
    assert jnp.allclose(batch.target, targets[batch.indices])


def test_ragged_series_supervised_constraint_validates_domain_and_targets():
    data_domain = phx.domain.DatasetDomain(jnp.zeros((3, 2)))
    domain, targets = _domain_and_targets()

    with pytest.raises(TypeError, match="RaggedSeriesDatasetDomain"):
        RaggedSeriesSupervisedConstraint(
            "u",
            data_domain.component(),
            targets,
            num_cases=4,
        )

    with pytest.raises(ValueError, match="leading axis"):
        RaggedSeriesSupervisedConstraint(
            "u",
            domain.component(),
            jnp.zeros((4, 2)),
            num_cases=4,
        )
