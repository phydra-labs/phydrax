#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

from phydrax import DerivativeSurface, DifferentiationRequest
from phydrax.ml import MLBatch
from phydrax.ml.decomposition import IncrementalPCA, IncrementalPCAModel


def _merge_action(first: Array, weight: Array, /) -> Array:
    prior = (
        IncrementalPCA(2).fit_batch(MLBatch(first, sample_weight=weight)).as_trainable()
    )
    assert isinstance(prior, IncrementalPCAModel)
    second = jnp.array(
        [[-0.7, 0.2, 0.6], [0.9, -0.1, -0.3], [0.2, 0.6, 0.5], [-0.4, -0.8, 0.1]]
    )
    final = prior.update_recipe().fit_batch(MLBatch(second)).as_trainable()
    assert isinstance(final, IncrementalPCAModel)
    third = jnp.array([[0.3, -0.5, 0.8], [-0.2, 0.4, 0.2], [0.9, 0.1, -0.7]])
    final = final.update_recipe().fit_batch(MLBatch(third)).as_trainable()
    assert isinstance(final, IncrementalPCAModel)
    return jnp.vdot(
        jnp.array([0.7, -0.3, 0.2]), final.project(jnp.array([0.2, 0.8, -0.4]))
    ).real


def test_prior_repeated_covariance_feature_and_mass_tangents() -> None:
    # The first retained covariance has two equal positive eigenvalues.
    first = jnp.array(
        [[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, -3.0, 0.0]]
    )
    weight = jnp.ones(4)
    direction = jnp.array(
        [[0.2, -0.1, 0.4], [0.3, 0.2, -0.1], [-0.2, 0.1, 0.5], [0.1, 0.3, -0.2]]
    )
    h = 1e-5
    tangent = jax.jvp(
        lambda values: _merge_action(values, weight), (first,), (direction,)
    )[1]
    expected = (
        _merge_action(first + h * direction, weight)
        - _merge_action(first - h * direction, weight)
    ) / (2 * h)
    assert jnp.allclose(tangent, expected, rtol=2e-5, atol=2e-7)
    mass_direction = jnp.array([0.2, -0.1, 0.3, 0.4])
    tangent = jax.jvp(
        lambda weights: _merge_action(first, weights), (weight,), (mass_direction,)
    )[1]
    expected = (
        _merge_action(first, weight + h * mass_direction)
        - _merge_action(first, weight - h * mass_direction)
    ) / (2 * h)
    assert jnp.allclose(tangent, expected, rtol=2e-5, atol=2e-7)


def test_prior_offset_and_total_mass_survive_merge() -> None:
    first = jnp.array(
        [[4.0, -2.0, 0.0], [-2.0, -2.0, 0.0], [1.0, 1.0, 0.0], [1.0, -5.0, 0.0]]
    )
    prior = (
        IncrementalPCA(2)
        .fit_batch(MLBatch(first, sample_weight=jnp.array([1.0, 2.0, 3.0, 4.0])))
        .as_trainable()
    )
    assert isinstance(prior, IncrementalPCAModel)
    second = jnp.array([[2.0, 3.0, 0.4], [-1.0, 2.0, -0.2], [0.5, -0.8, 0.7]])
    current_weight = jnp.array([2.0, 1.0, 3.0])
    final = (
        prior.update_recipe()
        .fit_batch(MLBatch(second, sample_weight=current_weight))
        .as_trainable()
    )
    assert isinstance(final, IncrementalPCAModel)
    expected_mean = (
        prior.offset * prior.total_weight
        + jnp.sum(second * current_weight[:, None], axis=0)
    ) / 16.0
    assert jnp.allclose(final.offset, expected_mean, atol=1e-12)
    assert jnp.allclose(final.total_weight, 16.0, atol=1e-12)


def test_failed_prior_cutoff_is_not_recertified_by_later_merge() -> None:
    first = jnp.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    prior_result = IncrementalPCA(1).fit_batch(MLBatch(first))
    assert bool(prior_result.valid)
    request = DifferentiationRequest((DerivativeSurface.FIT_FEATURES,))
    prior_admission = prior_result.derivative_admission(request, operation="project")
    assert prior_admission.runtime_valid is not None
    assert prior_admission.runtime_status is not None
    assert not bool(jnp.all(prior_admission.runtime_valid))
    prior = prior_result.as_trainable()
    assert isinstance(prior, IncrementalPCAModel)
    second = jnp.array([[-4.0, 0.1], [-1.0, -0.1], [2.0, 0.2], [5.0, -0.2]])
    final = prior.update_recipe().fit_batch(MLBatch(second))
    assert bool(final.valid)
    admission = final.derivative_admission(request, operation="project")
    assert admission.runtime_valid is not None
    assert admission.runtime_status is not None
    assert not bool(jnp.all(admission.runtime_valid))
    assert jnp.array_equal(admission.runtime_status, prior_admission.runtime_status)
    with pytest.raises(eqx.EquinoxRuntimeError):
        final.require_derivative(request, operation="project")
