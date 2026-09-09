import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.special._normal import normal_cdf
from phydrax.stochastic._gaussian_path import (
    brownian_increments_from_normals,
    gaussian_path_from_unit_design,
    GaussianPathConstructionPlan,
    prepare_gaussian_path_construction,
)


def _expected_covariance(times):
    elapsed = np.asarray(times)[1:] - float(np.asarray(times)[0])
    return np.minimum(elapsed[:, None], elapsed[None, :])


@pytest.mark.parametrize(
    "times",
    [
        jnp.asarray(1.0),
        jnp.asarray([0.0]),
        jnp.asarray([[0.0, 1.0]]),
        jnp.asarray([0.0, jnp.nan, 1.0]),
        jnp.asarray([0.0, jnp.inf]),
        jnp.asarray([0.0, 0.5, 0.5]),
        jnp.asarray([0.0, 1.0, 0.5]),
    ],
)
def test_gaussian_path_plan_rejects_invalid_grids(times):
    with pytest.raises(ValueError):
        GaussianPathConstructionPlan(times)


def test_gaussian_path_plan_rejects_invalid_methods_and_ranks():
    times = jnp.asarray([0.0, 0.2, 0.6, 1.0])
    with pytest.raises(ValueError, match="method"):
        GaussianPathConstructionPlan(times, "unknown")
    with pytest.raises(TypeError, match="integer"):
        GaussianPathConstructionPlan(times, "pca", 1.5)
    with pytest.raises(ValueError, match="\[1, 3\]"):
        GaussianPathConstructionPlan(times, "pca", 0)
    with pytest.raises(ValueError, match="\[1, 3\]"):
        GaussianPathConstructionPlan(times, "pca", 4)
    with pytest.raises(ValueError, match="Only the PCA"):
        GaussianPathConstructionPlan(times, "bridge", 2)


def test_chronological_construction_preserves_leading_axes_and_increment_identity():
    times = jnp.asarray([2.0, 2.1, 2.5, 3.4])
    prepared = prepare_gaussian_path_construction(
        GaussianPathConstructionPlan(times, "chronological")
    )
    normals = jnp.linspace(-1.2, 1.4, 2 * 3 * 3).reshape((2, 3, 3))
    expected = normals * jnp.sqrt(jnp.diff(times))
    increments = brownian_increments_from_normals(prepared, normals)
    result = gaussian_path_from_unit_design(prepared, normal_cdf(normals))

    assert increments.shape == result.increments.shape == (2, 3, 3)
    assert result.path.shape == (2, 3, 4)
    assert result.standard_normals.shape == normals.shape
    np.testing.assert_allclose(
        np.asarray(increments), np.asarray(expected), rtol=3e-15, atol=3e-15
    )
    np.testing.assert_allclose(
        np.asarray(result.increments), np.asarray(expected), rtol=3e-14, atol=3e-14
    )
    np.testing.assert_array_equal(np.asarray(result.path[..., 0]), np.zeros((2, 3)))
    np.testing.assert_allclose(
        np.asarray(jnp.diff(result.path, axis=-1)),
        np.asarray(result.increments),
        rtol=3e-14,
        atol=3e-14,
    )
    assert np.asarray(result.valid).all()
    assert np.asarray(result.evidence.finite).all()


def test_bridge_orders_conditional_nodes_and_preserves_endpoint_factor():
    times = jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    prepared = prepare_gaussian_path_construction(
        GaussianPathConstructionPlan(times, "bridge")
    )
    normals = jnp.asarray([[1.25, -0.4, 0.7, -1.1], [-0.75, 0.2, 1.0, 0.3]])
    result = gaussian_path_from_unit_design(prepared, normal_cdf(normals))

    np.testing.assert_array_equal(np.asarray(prepared.ordering), np.asarray([4, 2, 1, 3]))
    np.testing.assert_allclose(
        np.asarray(result.path[:, -1]),
        np.asarray(normals[:, 0]),
        rtol=3e-14,
        atol=3e-14,
    )
    np.testing.assert_allclose(
        np.asarray(result.evidence.endpoint_residual), np.zeros(2), rtol=0.0, atol=3e-15
    )
    np.testing.assert_allclose(
        np.asarray(prepared.factor @ prepared.factor.T),
        _expected_covariance(times),
        rtol=3e-15,
        atol=3e-15,
    )
    assert np.asarray(result.valid).all()


@pytest.mark.parametrize("method", ["chronological", "bridge", "pca"])
def test_full_rank_constructions_preserve_finite_grid_brownian_covariance(method):
    times = jnp.asarray([1.3, 1.35, 1.7, 2.4, 3.0])
    prepared = prepare_gaussian_path_construction(
        GaussianPathConstructionPlan(times, method)
    )
    expected = _expected_covariance(times)

    np.testing.assert_allclose(
        np.asarray(prepared.factor @ prepared.factor.T),
        expected,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        np.asarray(prepared.covariance_residual),
        np.zeros_like(expected),
        rtol=0.0,
        atol=2e-14,
    )
    assert float(prepared.relative_covariance_residual) < 2e-14


def test_rank_truncated_pca_exposes_exact_covariance_residual():
    times = jnp.asarray([0.0, 0.05, 0.2, 0.6, 1.1, 2.0])
    plan = GaussianPathConstructionPlan(times, "pca", 2)
    prepared = prepare_gaussian_path_construction(plan)
    expected = jnp.asarray(_expected_covariance(times))
    reconstructed = prepared.factor @ prepared.factor.T

    assert prepared.factor.shape == (5, 2)
    assert prepared.increment_factor.shape == (5, 2)
    np.testing.assert_allclose(
        np.asarray(prepared.covariance_residual),
        np.asarray(reconstructed - expected),
        rtol=0.0,
        atol=0.0,
    )
    assert float(prepared.relative_covariance_residual) > 0.0
    assert float(prepared.relative_covariance_residual) < 1.0

    design = jnp.asarray([[0.2, 0.8], [0.4, 0.6]])
    result = gaussian_path_from_unit_design(prepared, design)
    np.testing.assert_array_equal(
        np.asarray(result.covariance_residual), np.asarray(prepared.covariance_residual)
    )
    assert np.asarray(result.valid).all()


def test_gaussian_path_validates_factor_shapes_and_active_values():
    prepared = prepare_gaussian_path_construction(
        GaussianPathConstructionPlan(jnp.asarray([0.0, 0.5, 1.0]), "bridge")
    )
    with pytest.raises(ValueError, match="final factor dimension 2"):
        brownian_increments_from_normals(prepared, jnp.ones((4, 3)))
    with pytest.raises(ValueError, match="active values must be finite"):
        brownian_increments_from_normals(prepared, jnp.asarray([0.0, jnp.nan]))
    with pytest.raises(ValueError, match="final factor dimension 2"):
        gaussian_path_from_unit_design(prepared, jnp.ones((4, 3)))
    for invalid in (
        jnp.asarray([0.0, 0.5]),
        jnp.asarray([1.0, 0.5]),
        jnp.asarray([jnp.nan, 0.5]),
        jnp.asarray([jnp.inf, 0.5]),
    ):
        with pytest.raises(ValueError, match="finite and lie in"):
            gaussian_path_from_unit_design(prepared, invalid)


def test_construction_and_realization_identities_bind_order_and_unit_design():
    times = jnp.asarray([0.0, 0.25, 0.8, 1.0])
    first = prepare_gaussian_path_construction(
        GaussianPathConstructionPlan(times, "bridge")
    )
    repeated = prepare_gaussian_path_construction(
        GaussianPathConstructionPlan(times, "bridge")
    )
    chronological = prepare_gaussian_path_construction(
        GaussianPathConstructionPlan(times, "chronological")
    )
    design = jnp.asarray([[0.2, 0.4, 0.8], [0.7, 0.6, 0.3]])
    first_result = gaussian_path_from_unit_design(first, design)
    repeated_result = gaussian_path_from_unit_design(repeated, design)
    changed_result = gaussian_path_from_unit_design(first, design.at[0, 0].set(0.25))

    assert first.plan.plan_id == repeated.plan.plan_id
    assert first.construction_id == repeated.construction_id
    assert first.construction_id != chronological.construction_id
    assert first_result.realization_id == repeated_result.realization_id
    assert first_result.realization_id != changed_result.realization_id
    assert first_result.plan_id == first.plan.plan_id
    assert first_result.construction_id == first.construction_id
