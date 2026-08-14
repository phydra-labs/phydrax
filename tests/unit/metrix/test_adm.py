#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _chart():
    return phx.metrix.CoordinateChart("adm", ("t", "x", "y", "z"))


def _reference_fields():
    def lapse(q):
        return 1.4 + 0.1 * q[0]

    def shift(q):
        return jnp.array([0.1 * q[0], -0.2, 0.05], dtype=q.dtype)

    def spatial_metric(q):
        return jnp.array(
            [
                [1.2 + 0.05 * q[0], 0.1, 0.0],
                [0.1, 1.8, -0.05],
                [0.0, -0.05, 2.4],
            ],
            dtype=q.dtype,
        )

    return lapse, shift, spatial_metric


@pytest.mark.parametrize("convention", ("mostly_plus", "mostly_minus"))
def test_adm_decomposition_round_trips_reference_fields_in_batches(convention):
    chart = _chart()
    lapse, shift, spatial_metric = _reference_fields()
    metric = phx.metrix.adm_metric(
        lapse,
        shift,
        spatial_metric,
        chart=chart,
        convention=convention,
    )
    points = jnp.array(
        [[0.0, 0.2, -0.1, 0.3], [0.5, -0.2, 0.4, -0.3]],
    )
    decomposition = phx.metrix.decompose_adm_metric(metric, points)
    expected_lapse = jax.vmap(lapse)(points)
    expected_shift = jax.vmap(shift)(points)
    expected_spatial = jax.vmap(spatial_metric)(points)
    matrices = metric(points)

    assert jnp.allclose(decomposition.lapse, expected_lapse)
    assert jnp.allclose(decomposition.shift, expected_shift)
    assert jnp.allclose(decomposition.spatial_metric, expected_spatial)
    assert jnp.allclose(decomposition.spacetime_metric(), matrices)
    assert jnp.allclose(decomposition.spacetime_inverse, metric.inverse(points))

    report = phx.metrix.validate_adm_decomposition(
        decomposition,
        reference_metric=matrices,
    )
    assert bool(report.valid)
    assert bool(report.signature_matches)
    assert report.minimum_lapse > 0.0
    assert report.minimum_spatial_eigenvalue > 0.0
    assert report.maximum_inverse_residual < 1e-10
    assert report.maximum_reconstruction_residual < 1e-12


def test_adm_parameterization_enforces_lapse_and_spatial_positivity_under_jit():
    chart = _chart()
    parameterization = phx.metrix.ADMParameterization(
        lambda q: jnp.asarray(-100.0, dtype=q.dtype),
        lambda q: jnp.array([q[0], -0.2, 0.3], dtype=q.dtype),
        lambda q: jnp.array(
            [[-100.0, 9.0, -4.0], [0.2, -80.0, 5.0], [-0.1, 0.3, -60.0]],
            dtype=q.dtype,
        ),
        chart=chart,
        minimum_lapse=0.1,
        minimum_spatial_diagonal=0.2,
    )
    points = jnp.array(
        [[0.0, 0.0, 0.0, 0.0], [0.4, -0.2, 0.1, 0.3]],
    )
    decomposition = parameterization(points)
    matrices = jax.jit(parameterization.metric())(points)
    recovered = phx.metrix.decompose_adm_metric(parameterization.metric(), points)

    assert jnp.all(decomposition.lapse >= 0.1)
    assert jnp.all(jnp.linalg.eigvalsh(decomposition.spatial_metric) > 0.0)
    assert jnp.allclose(recovered.lapse, decomposition.lapse)
    assert jnp.allclose(recovered.shift, decomposition.shift)
    assert jnp.allclose(recovered.spatial_metric, decomposition.spatial_metric)
    assert bool(
        phx.metrix.validate_lorentzian_metric(
            parameterization.metric(),
            points,
        ).valid
    )
    assert jnp.all(jnp.isfinite(matrices))


def test_parameterized_adm_metric_is_differentiable_in_every_raw_field():
    chart = _chart()
    point = jnp.array([0.2, -0.1, 0.3, 0.4])

    def objective(raw):
        parameterization = phx.metrix.ADMParameterization(
            lambda q: raw[0] + 0.1 * q[0],
            lambda q: raw[1:4] + 0.0 * q[1:],
            lambda q: raw[4:].reshape((3, 3)) + 0.0 * q[0],
            chart=chart,
        )
        decomposition = parameterization(point)
        return (
            decomposition.lapse
            + jnp.sum(decomposition.shift)
            + jnp.sum(decomposition.spatial_metric)
        )

    raw = jnp.linspace(-0.4, 0.5, 13)
    gradient = jax.jit(jax.grad(objective))(raw)

    assert gradient.shape == raw.shape
    assert jnp.all(jnp.isfinite(gradient))
    active = jnp.array([0, 1, 2, 3, 4, 7, 8, 10, 11, 12])
    ignored_upper_triangle = jnp.array([5, 6, 9])
    assert jnp.all(jnp.abs(gradient[active]) > 0.0)
    assert jnp.array_equal(
        gradient[ignored_upper_triangle],
        jnp.zeros((3,), dtype=gradient.dtype),
    )


def test_adm_validation_reports_invalid_fields_without_repairing_them():
    decomposition = phx.metrix.ADMDecomposition(
        jnp.asarray(-1.0),
        jnp.zeros((3,)),
        jnp.diag(jnp.array([1.0, -0.5, 2.0])),
        chart=_chart(),
    )
    report = phx.metrix.validate_adm_decomposition(decomposition)

    assert not bool(report.valid)
    assert not bool(report.lapse_positive)
    assert report.minimum_spatial_eigenvalue < 0.0
    with pytest.raises(ValueError, match="ADM validation failed"):
        phx.metrix.validate_adm_decomposition(
            decomposition,
            raise_on_error=True,
        )


def test_adm_parameterization_rejects_invalid_static_and_field_contracts():
    chart = _chart()

    def valid_shift(q):
        return jnp.zeros((3,), dtype=q.dtype)

    def valid_factor(q):
        return jnp.zeros((3, 3), dtype=q.dtype)

    with pytest.raises(ValueError, match="minimum_lapse"):
        phx.metrix.ADMParameterization(
            lambda q: jnp.asarray(0.0),
            valid_shift,
            valid_factor,
            chart=chart,
            minimum_lapse=0.0,
        )
    malformed = phx.metrix.ADMParameterization(
        lambda q: jnp.zeros((1,), dtype=q.dtype),
        valid_shift,
        valid_factor,
        chart=chart,
    )
    with pytest.raises(ValueError, match="raw_lapse"):
        malformed(jnp.zeros((4,)))
