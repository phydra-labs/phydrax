import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.layers import MeasureNormalizedConvND


_DIMENSION_NUMBERS = {
    1: ("NWC", "WIO", "NWC"),
    2: ("NHWC", "HWIO", "NHWC"),
    3: ("NDHWC", "DHWIO", "NDHWC"),
}


@pytest.mark.parametrize(
    ("spatial_ndim", "sample_shape"),
    ((1, (7,)), (2, (5, 6)), (3, (4, 5, 6))),
)
def test_uniform_full_measure_matches_ordinary_convolution(spatial_ndim, sample_shape):
    layer = MeasureNormalizedConvND(
        spatial_ndim=spatial_ndim,
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        key=jr.key(spatial_ndim),
    )
    values = jr.normal(jr.key(20 + spatial_ndim), (2,) + sample_shape + (2,))
    quadrature = jnp.full(sample_shape, 0.25)

    actual = eqx.filter_jit(layer)(values, quadrature=quadrature)
    expected = jax.lax.conv_general_dilated(
        values,
        layer.weight.astype(values.dtype),
        window_strides=(1,) * spatial_ndim,
        padding="SAME",
        dimension_numbers=_DIMENSION_NUMBERS[spatial_ndim],
    )
    expected = expected + layer.bias.astype(values.dtype)
    assert jnp.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_missing_observations_are_renormalized_without_nan_contamination():
    layer = MeasureNormalizedConvND(
        spatial_ndim=1,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        use_bias=False,
        key=jr.key(4),
    )
    layer = eqx.tree_at(lambda node: node.weight, layer, jnp.ones((3, 1, 1)))
    mask = jnp.asarray([True, False, True, True, False, True])
    values = jnp.where(mask, 2.0, jnp.nan)[..., None]

    output = layer(values, source_mask=mask, target_mask=jnp.ones_like(mask))
    full_reference = layer(jnp.full_like(values, 2.0))

    assert jnp.all(jnp.isfinite(output))
    assert jnp.allclose(output, full_reference)


def test_all_masked_stencils_and_invalid_targets_return_exact_zero():
    layer = MeasureNormalizedConvND(
        spatial_ndim=2,
        in_channels=2,
        out_channels=2,
        kernel_size=3,
        key=jr.key(7),
    )
    values = jnp.full((4, 5, 2), jnp.nan)
    source_mask = jnp.zeros((4, 5), dtype=bool)
    target_mask = jnp.asarray(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )

    output = layer(
        values,
        source_mask=source_mask,
        target_mask=target_mask,
        quadrature=jnp.ones((4, 5)),
    )
    assert jnp.array_equal(output, jnp.zeros_like(output))


def test_measure_convolution_has_finite_value_weight_and_quadrature_gradients():
    layer = MeasureNormalizedConvND(
        spatial_ndim=1,
        in_channels=1,
        out_channels=2,
        kernel_size=3,
        key=jr.key(10),
    )
    values = jnp.linspace(-1.0, 1.0, 8)[:, None]
    quadrature = jnp.linspace(0.5, 1.5, 8)
    mask = jnp.asarray([True, True, False, True, True, True, False, True])

    value_gradient = jax.grad(
        lambda current: jnp.sum(
            layer(current, source_mask=mask, quadrature=quadrature) ** 2
        )
    )(values)
    weight_gradient = jax.grad(
        lambda weight: jnp.sum(
            eqx.tree_at(lambda node: node.weight, layer, weight)(
                values, source_mask=mask, quadrature=quadrature
            )
            ** 2
        )
    )(layer.weight)
    quadrature_gradient = jax.grad(
        lambda weights: jnp.sum(layer(values, source_mask=mask, quadrature=weights) ** 2)
    )(quadrature)

    assert jnp.all(jnp.isfinite(value_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))
    assert jnp.all(jnp.isfinite(quadrature_gradient))


def test_measure_convolution_preserves_leading_case_axes():
    layer = MeasureNormalizedConvND(
        spatial_ndim=2,
        in_channels=1,
        out_channels=2,
        kernel_size=(3, 1),
        strides=(2, 1),
        padding="VALID",
        key=jr.key(13),
    )
    values = jnp.ones((2, 3, 7, 5, 1))
    mask = jnp.ones((7, 5), dtype=bool)

    output = eqx.filter_jit(layer)(values, source_mask=mask)
    assert output.shape == (2, 3, math.floor((7 - 3) / 2) + 1, 5, 2)
