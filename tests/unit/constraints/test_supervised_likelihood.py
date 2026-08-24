#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _zero_field(domain):
    @domain.Function("data")
    def field(row):
        return 0.0 * row[0]

    return field


def test_supervised_likelihood_preserves_targets_filters_cases_and_weights_risk():
    rows = jnp.arange(4.0)[:, None]
    domain = phx.domain.DatasetDomain(rows)
    targets = jnp.asarray([0, 1, -99, 3], dtype=jnp.int32)
    sample_mask = jnp.asarray([True, True, False, True])
    sample_weight = jnp.asarray([1.0, 2.0, jnp.nan, 4.0])
    likelihood = phx.uq.GaussianLikelihood(1.0)
    field = _zero_field(domain)
    term = phx.terms.SupervisedLikelihoodTerm(
        "u",
        domain.component(),
        targets,
        likelihood,
        sampling=phx.domain.PointSampling(2, design="uniform"),
        indices=jnp.asarray([3, 2, 1]),
        sample_mask=sample_mask,
        sample_weight=sample_weight,
    )
    batch = term.observed_batch()
    per_case = -likelihood.log_prob(jnp.zeros((2,)), jnp.asarray([3, 1]))

    assert batch.target.dtype == jnp.int32
    assert jnp.array_equal(batch.indices, jnp.asarray([3, 1]))
    assert jnp.array_equal(batch.target, jnp.asarray([3, 1]))
    assert jnp.array_equal(batch.sample_weight, jnp.asarray([4.0, 2.0]))
    np.testing.assert_allclose(
        term.log_prob({"u": field}, batch=batch),
        -per_case,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        term.loss({"u": field}, batch=batch),
        jnp.sum(batch.sample_weight * per_case) / jnp.sum(batch.sample_weight),
        atol=2e-15,
    )

    summed = phx.terms.SupervisedLikelihoodTerm(
        "u",
        domain.component(),
        targets,
        likelihood,
        sampling=phx.domain.PointSampling(2, design="uniform"),
        indices=jnp.asarray([3, 2, 1]),
        sample_mask=sample_mask,
        sample_weight=sample_weight,
        reduction="sum",
    )
    np.testing.assert_allclose(
        summed.loss({"u": field}, batch=summed.observed_batch()),
        jnp.sum(jnp.asarray([4.0, 2.0]) * per_case),
        atol=2e-15,
    )


def test_supervised_likelihood_rejects_invalid_case_masks_and_weights():
    domain = phx.domain.DatasetDomain(jnp.arange(3.0)[:, None])
    common = {
        "sampling": phx.domain.PointSampling(2, design="uniform"),
    }

    with pytest.raises(TypeError, match="Boolean"):
        phx.terms.SupervisedLikelihoodTerm(
            "u",
            domain.component(),
            jnp.zeros((3,)),
            phx.uq.GaussianLikelihood(1.0),
            sample_mask=jnp.ones((3,)),
            **common,
        )
    with pytest.raises(ValueError, match="non-empty"):
        phx.terms.SupervisedLikelihoodTerm(
            "u",
            domain.component(),
            jnp.zeros((3,)),
            phx.uq.GaussianLikelihood(1.0),
            sample_mask=jnp.zeros((3,), dtype=bool),
            **common,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        phx.terms.SupervisedLikelihoodTerm(
            "u",
            domain.component(),
            jnp.zeros((3,)),
            phx.uq.GaussianLikelihood(1.0),
            sample_weight=jnp.asarray([1.0, 0.0, 1.0]),
            **common,
        )
    with pytest.raises(ValueError, match="shape"):
        phx.terms.SupervisedLikelihoodTerm(
            "u",
            domain.component(),
            jnp.zeros((3,)),
            phx.uq.GaussianLikelihood(1.0),
            sample_weight=jnp.ones((2,)),
            **common,
        )
