#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
import pytest

import phydrax as phx


def _periodic_nodes(size):
    return -1.0 + 2.0 * jnp.arange(size, dtype=float) / size


def _lattice(nodes):
    return jnp.stack(jnp.meshgrid(*nodes, indexing="ij"), axis=-1)


@pytest.mark.parametrize("spatial_shape", ((7,), (5, 6), (4, 5, 6)))
def test_rectilinear_sampling_identity_and_periodic_cell_translation(spatial_shape):
    dimensions = len(spatial_shape)
    nodes = tuple(_periodic_nodes(size) for size in spatial_shape)
    coordinates = _lattice(nodes)
    values = jnp.arange(2 * int(jnp.prod(jnp.array(spatial_shape))), dtype=float)
    values = values.reshape(spatial_shape + (2,))
    boundary = ("periodic",) * dimensions

    identity = phx.nn.layers.sample_rectilinear_grid(
        values,
        coordinates,
        spatial_ndim=dimensions,
        boundary=boundary,
        axis_nodes=nodes,
    )
    translated = phx.nn.layers.sample_rectilinear_grid(
        values,
        coordinates + jnp.array([2.0 / size for size in spatial_shape]),
        spatial_ndim=dimensions,
        boundary=boundary,
        axis_nodes=nodes,
    )
    expected = values
    for axis in range(dimensions):
        expected = jnp.roll(expected, -1, axis=axis)

    assert jnp.allclose(identity, values)
    assert jnp.allclose(translated, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("boundary", "coordinates", "fill_value", "expected", "support"),
    (
        (
            "periodic",
            [-1.5, -1.0, -0.5, 1.0, 1.5],
            -9.0,
            [3.0, 0.0, 1.0, 0.0, 1.0],
            [True, True, True, True, True],
        ),
        (
            "clamp",
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            -9.0,
            [0.0, 0.0, 1.5, 3.0, 3.0],
            [True, True, True, True, True],
        ),
        (
            "reflect",
            [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            -9.0,
            [1.5, 0.0, 1.5, 3.0, 1.5, 0.0],
            [True, True, True, True, True, True],
        ),
        (
            "constant",
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            -9.0,
            [-9.0, 0.0, 1.5, 3.0, -9.0],
            [False, True, True, True, False],
        ),
    ),
)
def test_rectilinear_boundary_modes(boundary, coordinates, fill_value, expected, support):
    sampled, actual_support = phx.nn.layers.sample_rectilinear_grid(
        jnp.arange(4.0)[:, None],
        jnp.asarray(coordinates)[:, None],
        spatial_ndim=1,
        boundary=(boundary,),
        fill_value=fill_value,
        return_support=True,
    )

    assert jnp.allclose(sampled[:, 0], jnp.asarray(expected))
    assert jnp.array_equal(actual_support, jnp.asarray(support))


def test_nonuniform_rectilinear_nodes_are_affine_exact_eager_and_jit():
    physical_x = jnp.array([2.0, 3.0, 6.0, 10.0])
    physical_y = jnp.array([-4.0, -1.0, 0.5, 5.0, 8.0])
    x = phx.nn.layers.normalized_axis_nodes(physical_x, periodic=False)
    y = phx.nn.layers.normalized_axis_nodes(physical_y, periodic=False)
    periodic = phx.nn.layers.normalized_axis_nodes(
        jnp.array([0.0, 1.0, 3.0, 6.0]),
        periodic=True,
        period=8.0,
    )
    values = (1.25 + 2.0 * x[:, None] - 0.75 * y[None, :])[..., None]
    query = jnp.array([[-0.8, -0.7], [-0.1, 0.2], [0.75, 0.9]])

    def evaluate(field, coordinates, x_nodes, y_nodes):
        return phx.nn.layers.sample_rectilinear_grid(
            field,
            coordinates,
            spatial_ndim=2,
            boundary=("clamp", "clamp"),
            axis_nodes=(x_nodes, y_nodes),
        )

    eager = evaluate(values, query, x, y)
    compiled = eqx.filter_jit(evaluate)(values, query, x, y)
    expected = 1.25 + 2.0 * query[:, 0] - 0.75 * query[:, 1]

    assert jnp.allclose(x, jnp.array([-1.0, -0.75, 0.0, 1.0]))
    assert jnp.allclose(periodic, jnp.array([-1.0, -0.75, -0.25, 0.5]))
    assert jnp.allclose(eager[:, 0], expected, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(compiled, eager, rtol=1e-6, atol=1e-6)


def test_rectilinear_sampling_promotes_integral_inputs_before_interpolation():
    result = phx.nn.layers.sample_rectilinear_grid(
        jnp.array([0, 2])[:, None],
        jnp.array([[0.0]]),
        spatial_ndim=1,
        boundary=("clamp",),
    )

    assert jnp.issubdtype(result.dtype, jnp.inexact)
    assert jnp.allclose(result, jnp.array([[1.0]]))


@pytest.mark.parametrize(
    ("mask_mode", "fill_value", "expected", "expected_support"),
    (
        ("renormalize", -7.0, 1.0, True),
        ("strict", -7.0, -7.0, False),
    ),
)
def test_masked_nan_corners_follow_nonreject_mask_semantics(
    mask_mode,
    fill_value,
    expected,
    expected_support,
):
    sampled, support = phx.nn.layers.sample_rectilinear_grid(
        jnp.array([1.0, jnp.nan, 5.0])[:, None],
        jnp.array([[-0.5]]),
        spatial_ndim=1,
        boundary=("clamp",),
        source_mask=jnp.array([True, False, True]),
        mask_mode=mask_mode,
        fill_value=fill_value,
        return_support=True,
    )

    assert jnp.isfinite(sampled[0, 0])
    assert sampled[0, 0] == expected
    assert support[0] == expected_support


def test_reject_mask_mode_rejects_holes_even_when_payload_is_nan():
    with pytest.raises(
        eqx.EquinoxRuntimeError,
        match="reject mode does not permit source holes",
    ):
        result = phx.nn.layers.sample_rectilinear_grid(
            jnp.array([1.0, jnp.nan, 5.0])[:, None],
            jnp.array([[-0.5]]),
            spatial_ndim=1,
            boundary=("clamp",),
            source_mask=jnp.array([True, False, True]),
            mask_mode="reject",
        )
        jax.block_until_ready(result)


def test_affine_warp_jacobian_and_determinant_are_exact_on_nonuniform_nodes():
    x = jnp.array([-1.0, -0.45, 0.2, 1.0])
    y = jnp.array([-1.0, -0.7, 0.15, 0.6, 1.0])
    coordinates = _lattice((x, y))
    gradient = jnp.array([[0.2, -0.1], [0.3, 0.4]])
    displacement = oe.contract("...j,ij->...i", coordinates, gradient)

    jacobian = phx.nn.layers.warp_jacobian(
        displacement,
        boundary=("clamp", "clamp"),
        axis_nodes=(x, y),
    )
    expected = jnp.eye(2) + gradient

    assert jnp.allclose(jacobian, expected, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(jnp.linalg.det(jacobian), jnp.linalg.det(expected))


@pytest.mark.parametrize(
    ("variance", "components"),
    (
        (("contravariant",), jnp.array([2.0, -3.0])),
        (("covariant",), jnp.array([2.0, -3.0])),
        (
            ("contravariant", "covariant"),
            jnp.array([[2.0, -3.0], [4.0, 5.0]]),
        ),
    ),
)
def test_vector_covector_and_rank_two_tensor_transformation_laws(variance, components):
    x = jnp.linspace(-1.0, 1.0, 4)
    y = jnp.linspace(-1.0, 1.0, 5)
    coordinates = _lattice((x, y))
    scales = jnp.array([1.25, 0.8])
    displacement = coordinates * (scales - 1.0)
    values = jnp.broadcast_to(components, (4, 5) + components.shape)

    transformed = phx.nn.layers.warp_field(
        values,
        displacement,
        boundary=("clamp", "clamp"),
        axis_nodes=(x, y),
        field_spec=phx.metrix.TensorType(variance),
    )
    if variance == ("contravariant",):
        expected = components / scales
    elif variance == ("covariant",):
        expected = components * scales
    else:
        expected = components / scales[:, None] * scales[None, :]

    assert jnp.allclose(transformed, expected, rtol=1e-5, atol=1e-6)


def test_periodic_density_remap_preserves_discrete_total_mass():
    size = 24
    x = _periodic_nodes(size)
    density = jnp.ones((size,))
    displacement = (0.05 * jnp.sin(jnp.pi * x))[..., None]

    remapped, diagnostics = phx.nn.layers.conservative_remap(
        density,
        displacement,
        boundary=("periodic",),
        axis_nodes=(x,),
        return_diagnostics=True,
    )

    assert jnp.all(diagnostics.determinant > 0.0)
    assert not jnp.allclose(remapped, density)
    assert jnp.allclose(jnp.sum(remapped), jnp.sum(density), rtol=1e-6, atol=1e-6)


def test_manifold_projection_retraction_and_masked_nan_payloads():
    points = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    ambient = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [-2.0, 1.0, 0.5],
            [0.5, -1.0, 2.0],
            [3.0, 0.5, -2.0],
        ]
    )
    tangent = phx.nn.layers.sphere_tangent_projection(points, ambient)
    retracted = phx.nn.layers.sphere_retraction(points, 0.2 * tangent)

    layer = phx.nn.layers.ManifoldMultiheadWarp(
        ambient_dim=3,
        in_channels=2,
        out_channels=2,
        num_heads=1,
        tangent_projection=phx.nn.layers.sphere_tangent_projection,
        retraction=phx.nn.layers.sphere_retraction,
        kernel_scale=0.5,
        key=jr.key(10),
    )
    mask = jnp.array([True, True, False, True])
    masked_points = points.at[2].set(jnp.nan)
    values = jnp.array([[1.0, 2.0], [3.0, 4.0], [jnp.nan, jnp.nan], [7.0, 8.0]])
    output = layer(values, masked_points, source_mask=mask)
    diagnostics = layer.diagnostics(values, masked_points, source_mask=mask)
    valid_tangent = diagnostics.tangent_displacement[mask, 0]
    valid_points = points[mask]

    assert jnp.allclose(jnp.sum(tangent * points, axis=-1), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.linalg.norm(retracted, axis=-1), 1.0, atol=1e-6)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.array_equal(output[~mask], jnp.zeros((1, 2)))
    assert jnp.all(jnp.isfinite(diagnostics.interpolation_weights))
    assert jnp.allclose(
        jnp.sum(valid_tangent * valid_points, axis=-1),
        0.0,
        atol=1e-6,
    )
    assert jnp.allclose(
        jnp.linalg.norm(diagnostics.transported_points[mask], axis=-1),
        1.0,
        atol=1e-6,
    )
    assert jnp.allclose(jnp.sum(diagnostics.interpolation_weights, axis=1), 1.0)


def test_probabilistic_warp_mean_and_sample_routes_are_coherent_and_differentiable():
    layer = phx.nn.layers.ProbabilisticMultiheadWarp(
        spatial_ndim=1,
        in_channels=2,
        out_channels=2,
        num_heads=2,
        boundary="periodic",
        minimum_scale=1e-4,
        scale_factor=0.05,
        key=jr.key(20),
    )
    values = jr.normal(jr.key(21), (7, 2))
    distribution = layer.distribution(values)

    mean_output = layer(values)
    mean_diagnostics = layer.diagnostics(values)
    expected_mean = layer.base.transport(values, distribution.mean)
    sample_key = jr.key(22)
    sampled_output = layer(values, key=sample_key)
    sampled_diagnostics = layer.diagnostics(values, key=sample_key)
    expected_sample = layer.base.transport(values, sampled_diagnostics.displacement)
    value_gradient = jax.grad(lambda field: jnp.mean(layer(field, key=sample_key) ** 2))(
        values
    )
    _, parameter_gradient = eqx.filter_value_and_grad(
        lambda current: jnp.mean(current(values, key=sample_key) ** 2)
    )(layer)
    gradient_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(parameter_gradient)
        if eqx.is_inexact_array(leaf)
    ]

    assert jnp.allclose(mean_diagnostics.displacement, distribution.mean)
    assert jnp.allclose(mean_output, expected_mean)
    assert jnp.allclose(sampled_output, expected_sample)
    assert jnp.array_equal(sampled_diagnostics.route_scale, distribution.scale)
    assert jnp.all(jnp.isfinite(value_gradient))
    assert jnp.linalg.norm(value_gradient) > 0.0
    assert gradient_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in gradient_leaves)
