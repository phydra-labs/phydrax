import coordax as cx
import jax.numpy as jnp

import phydrax as phx
import phydrax.discretization as spectral


def _tensor_grid():
    x_axis = phx.discretization.FourierAxisSpec(8).materialize(0.0, 1.0)
    y_axis = phx.discretization.CosineAxisSpec(7).materialize(-1.0, 1.0)
    return phx.discretization.SeparableSpectralDiscretization((x_axis, y_axis))


def test_spatial_measure_preserves_separable_tensor_weights_and_output_axes():
    discretization = _tensor_grid()
    target = phx.integration.spatial_measure(
        discretization,
        spatial_dims=("x", "y"),
    )
    x = discretization.axes[0].nodes[:, None]
    y = discretization.axes[1].nodes[None, :]
    scalar = x**2 + y**2
    values = cx.Field(
        jnp.stack((scalar, 2.0 * scalar), axis=-1),
        dims=("x", "y", "channel"),
    )

    estimate = phx.integration.integrate(values, target)

    expected = jnp.sum(
        discretization.quadrature_weights[..., None] * jnp.asarray(values.data),
        axis=(0, 1),
    )
    weights = target.weights
    assert not isinstance(weights, cx.Field)
    assert tuple(weights) == ("x", "y")
    assert estimate.value.dims == ("channel",)
    assert jnp.allclose(jnp.asarray(estimate.value.data), expected)
    assert jnp.all(estimate.successful)
    assert estimate.error_estimate is None


def test_spatial_measure_exposes_physical_coordinates_to_callables():
    discretization = _tensor_grid()
    target = phx.integration.spatial_measure(
        discretization,
        spatial_dims=("x", "y"),
    )

    estimate = phx.integration.integrate(
        lambda points: cx.Field(
            jnp.sum(points.data**2, axis=-1),
            dims=("x", "y"),
        ),
        target,
    )

    coordinates = target.points.data
    expected = jnp.sum(
        discretization.quadrature_weights * jnp.sum(coordinates**2, axis=-1)
    )
    assert jnp.allclose(jnp.asarray(estimate.value.data), expected)


def test_normalized_spatial_measure_and_mask_use_physical_quadrature_mass():
    discretization = _tensor_grid()
    mask = jnp.ones(discretization.state_shape, dtype=bool).at[0].set(False)
    target = phx.integration.spatial_measure(
        discretization,
        spatial_dims=("x", "y"),
        mask=mask,
        normalized=True,
    )
    values = cx.Field(
        jnp.broadcast_to(jnp.asarray(3.0), discretization.state_shape),
        dims=("x", "y"),
    )

    estimate = phx.integration.integrate(values, target)

    assert jnp.allclose(jnp.asarray(estimate.value.data), 3.0)
    assert jnp.allclose(
        estimate.diagnostics.target_mass,
        jnp.sum(jnp.where(mask, discretization.quadrature_weights, 0.0)),
    )


def test_spectral_spatial_measure_reduces_precomputed_fields_without_coordinates():
    plan = spectral.SpectralDecomposition.from_eigenpairs(
        jnp.asarray([0.0, 1.0, 4.0]),
        jnp.eye(3),
        jnp.asarray([0.2, 0.3, 0.5]),
        decomposition_id="integration-plan",
    )
    discretization = phx.discretization.SpectralDiscretization(plan)
    target = phx.integration.spatial_measure(discretization)
    values = cx.Field(jnp.asarray([1.0, 2.0, 4.0]), dims=("space",))

    estimate = phx.integration.integrate(values, target)

    assert target.points is None
    assert jnp.allclose(jnp.asarray(estimate.value.data), 2.8)
