#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
import pytest
from jax.scipy.special import sph_harm_y

import phydrax as phx


def _identity_basis_layer(basis, modes, axis):
    layer = phx.nn.operator.layers.BasisSpectralConvND(
        in_channels=1,
        out_channels=1,
        n_modes=modes,
        bases=basis,
        key=jr.key(modes),
    )
    layer = eqx.tree_at(lambda item: item.weight, layer, jnp.ones_like(layer.weight))
    return layer, axis


class _ValueFeature(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 2
        self.out_size = 1

    def __call__(self, value, *, key=None):
        del key
        return value[:1]


class _ConstantTrunk(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 1
        self.out_size = 1

    def __call__(self, value, *, key=None):
        del value, key
        return jnp.ones((1,))


class _SourceValueKernel(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 4
        self.out_size = 1

    def __call__(self, value, *, key=None):
        del key
        return value[:1]


class _ConstantDifferentialKernel(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 2
        self.out_size = 1

    def __call__(self, value, *, key=None):
        del value, key
        return jnp.ones((1,))


def test_fourier_spectral_conv_exactly_preserves_retained_mode():
    count = 32
    x = jnp.arange(count, dtype=float)
    signal = jnp.cos(2.0 * jnp.pi * 2.0 * x / count)[:, None]
    layer = phx.nn.operator.architectures.SpectralConvND(
        in_channels=1,
        out_channels=1,
        n_modes=4,
        key=jr.key(0),
    )
    layer = eqx.tree_at(lambda item: item.weight, layer, jnp.ones_like(layer.weight))

    output = layer(signal)
    assert jnp.allclose(output, signal, rtol=1e-11, atol=1e-11)


def test_fourier_spectral_conv_exactly_learns_negative_signed_block():
    nx, ny = 18, 20
    x = jnp.arange(nx, dtype=float)[:, None]
    y = jnp.arange(ny, dtype=float)[None, :]
    signal = jnp.cos(2.0 * jnp.pi * (-x / nx + y / ny))[..., None]
    layer = phx.nn.operator.architectures.SpectralConvND(
        in_channels=1,
        out_channels=1,
        n_modes=(3, 3),
        key=jr.key(0),
    )
    weight = jnp.zeros_like(layer.weight)
    weight = weight.at[1, 0, 0, 2, 1].set(1.0 + 0.0j)
    layer = eqx.tree_at(lambda item: item.weight, layer, weight)

    output = layer(signal)
    assert jnp.allclose(output, signal, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("basis", ("fourier", "sine", "cosine", "legendre"))
def test_basis_projection_reconstructs_representable_functions(basis):
    if basis == "fourier":
        nodes = jnp.linspace(0.0, 1.0, 40, endpoint=False)
        axis = phx.nn.operator.OperatorAxis(
            "x",
            nodes,
            quadrature_weights=jnp.full((40,), 1.0 / 40.0),
            basis="fourier",
            periodic=True,
        )
        modes = 5
        signal = (
            1.0
            + 0.3 * jnp.cos(2.0 * jnp.pi * nodes)
            - 0.2 * jnp.sin(4.0 * jnp.pi * nodes)
        )
    elif basis == "sine":
        nodes = jnp.linspace(0.0, 1.0, 41)
        axis = phx.nn.operator.OperatorAxis("x", nodes, basis="sine")
        modes = 3
        signal = jnp.sin(jnp.pi * nodes) + 0.4 * jnp.sin(3.0 * jnp.pi * nodes)
    elif basis == "cosine":
        nodes = jnp.linspace(0.0, 1.0, 41) ** 1.3
        axis = phx.nn.operator.OperatorAxis("x", nodes, basis="cosine")
        modes = 4
        signal = 1.0 + 0.2 * jnp.cos(jnp.pi * nodes) - 0.3 * jnp.cos(3.0 * jnp.pi * nodes)
    else:
        nodes = jnp.linspace(0.0, 1.0, 41) ** 1.2
        axis = phx.nn.operator.OperatorAxis("x", nodes, basis="legendre")
        modes = 4
        z = 2.0 * nodes - 1.0
        signal = 1.0 + 0.4 * z - 0.2 * 0.5 * (3.0 * z**2 - 1.0)

    layer, axis = _identity_basis_layer(basis, modes, axis)
    reconstructed = layer(signal[:, None], (axis,))[..., 0]
    assert jnp.allclose(reconstructed, signal, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("basis", ("fourier", "sine", "cosine", "legendre"))
def test_basis_projection_error_decreases_with_modes(basis):
    if basis == "fourier":
        nodes = jnp.linspace(0.0, 1.0, 80, endpoint=False)
        axis = phx.nn.operator.OperatorAxis(
            "x",
            nodes,
            quadrature_weights=jnp.full((80,), 1.0 / 80.0),
            periodic=True,
            basis="fourier",
        )
        signal = jnp.exp(0.4 * jnp.cos(2.0 * jnp.pi * nodes))
    elif basis == "sine":
        nodes = jnp.linspace(0.0, 1.0, 81)
        axis = phx.nn.operator.OperatorAxis("x", nodes, basis="sine")
        signal = nodes * (1.0 - nodes) * jnp.exp(nodes)
    elif basis == "cosine":
        nodes = jnp.linspace(0.0, 1.0, 81) ** 1.2
        axis = phx.nn.operator.OperatorAxis("x", nodes, basis="cosine")
        signal = jnp.exp(0.3 * jnp.cos(jnp.pi * nodes))
    else:
        nodes = jnp.linspace(0.0, 1.0, 81) ** 1.2
        axis = phx.nn.operator.OperatorAxis("x", nodes, basis="legendre")
        signal = jnp.exp(nodes)

    low, _ = _identity_basis_layer(basis, 3, axis)
    high, _ = _identity_basis_layer(basis, 10, axis)
    low_error = jnp.linalg.norm(low(signal[:, None], (axis,))[..., 0] - signal)
    high_error = jnp.linalg.norm(high(signal[:, None], (axis,))[..., 0] - signal)
    assert high_error < 0.25 * low_error


def test_basis_projection_is_jittable_and_differentiable():
    nodes = jnp.linspace(0.0, 1.0, 32) ** 1.2
    axis = phx.nn.operator.OperatorAxis("x", nodes, basis="legendre")
    layer, _ = _identity_basis_layer("legendre", 8, axis)
    values = jnp.exp(nodes)[:, None]

    evaluate = jax.jit(lambda x: layer(x, (axis,)))
    output = evaluate(values)
    gradient = jax.grad(lambda x: jnp.sum(evaluate(x) ** 2))(values)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.all(jnp.isfinite(gradient))


def _integral_encoder():
    return phx.nn.operator.architectures.IntegralBranchEncoder(
        feature_model=_ValueFeature(),
        latent_size=1,
        coord_dim=1,
    )


def test_integral_branch_padding_and_permutation_are_exact_invariances():
    coordinates = jnp.array([[0.1], [0.3], [0.6], [0.9]])
    values = jnp.array([1.0, 2.0, 4.0, 8.0])
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    encoder = _integral_encoder()
    samples = phx.nn.operator.FunctionSamples(
        values=values,
        coordinates=coordinates,
        quadrature_weights=weights,
    )
    reference = encoder(samples, case_ndim=0)

    permutation = jnp.array([2, 0, 3, 1])
    permuted = phx.nn.operator.FunctionSamples(
        values=values[permutation],
        coordinates=coordinates[permutation],
        quadrature_weights=weights[permutation],
    )
    padded = phx.nn.operator.FunctionSamples(
        values=jnp.concatenate((values, jnp.array([1e12, -1e12]))),
        coordinates=jnp.concatenate((coordinates, jnp.zeros((2, 1))), axis=0),
        quadrature_weights=jnp.concatenate((weights, jnp.ones((2,)))),
        mask=jnp.array([True, True, True, True, False, False]),
    )
    assert jnp.allclose(encoder(permuted, case_ndim=0), reference)
    assert jnp.allclose(encoder(padded, case_ndim=0), reference)


def test_integral_branch_has_midpoint_quadrature_convergence():
    encoder = _integral_encoder()

    def error(count):
        coordinates = (jnp.arange(count, dtype=float) + 0.5) / count
        samples = phx.nn.operator.FunctionSamples(
            values=coordinates**2,
            coordinates=coordinates[:, None],
            quadrature_weights=jnp.full((count,), 1.0 / count),
        )
        estimate = encoder(samples, case_ndim=0)[0]
        return jnp.abs(estimate - 1.0 / 3.0)

    assert error(64) < 0.02 * error(8)


def test_ragged_batched_deeponet_matches_individual_evaluations():
    encoder = _integral_encoder()
    model = phx.nn.operator.architectures.DeepONet(
        branch=encoder,
        trunk=_ConstantTrunk(),
        coord_dim=1,
        latent_size=1,
    )

    def make_batch(source_count, query_count):
        source_x = (jnp.arange(source_count, dtype=float) + 0.5) / source_count
        query_x = jnp.linspace(0.0, 1.0, query_count)
        return phx.nn.operator.OperatorBatch(
            inputs={
                "u": phx.nn.operator.FunctionSamples(
                    values=source_x**2,
                    coordinates=source_x[:, None],
                    quadrature_weights=jnp.full(
                        (source_count,),
                        1.0 / source_count,
                    ),
                )
            },
            queries={
                "query": phx.nn.operator.FunctionSamples(
                    values=None,
                    coordinates=query_x[:, None],
                )
            },
        )

    first = make_batch(7, 5)
    second = make_batch(13, 3)
    expected_first = model(first)
    expected_second = model(second)
    stacked = phx.nn.operator.stack_operator_batches((first, second), case_axis="case")
    actual = model(stacked)

    assert jnp.allclose(actual[0, :5], expected_first)
    assert jnp.allclose(actual[1, :3], expected_second)
    assert jnp.allclose(actual[1, 3:], 0.0)


def _local_integral_estimate(count):
    coordinates = (jnp.arange(count, dtype=float) + 0.5) / count
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "u": phx.nn.operator.FunctionSamples(
                values=coordinates**2,
                coordinates=coordinates[:, None],
                quadrature_weights=jnp.full((count,), 1.0 / count),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=jnp.array([[0.37]]),
            )
        },
    )
    operator = phx.nn.operator.architectures.LocalIntegralOperator(
        kernel_model=_SourceValueKernel(),
        coord_dim=1,
    )
    return operator(batch)[0]


def _graph_integral_estimate(count):
    coordinates = (jnp.arange(count, dtype=float) + 0.5) / count
    target = count
    graph = phx.graph.GraphIR(
        nodes={
            "u": jnp.concatenate((coordinates**2, jnp.zeros((1,)))),
            "quadrature_weight": jnp.concatenate(
                (jnp.full((count,), 1.0 / count), jnp.zeros((1,)))
            ),
            "type": jnp.concatenate(
                (
                    jnp.zeros((count,), dtype=jnp.int32),
                    jnp.ones((1,), dtype=jnp.int32),
                )
            ),
        },
        edges={},
        senders=jnp.arange(count, dtype=jnp.int32),
        receivers=jnp.full((count,), target, dtype=jnp.int32),
        n_node=jnp.array([count + 1], dtype=jnp.int32),
        n_edge=jnp.array([count], dtype=jnp.int32),
        validate=False,
    )
    operator = phx.graph.GraphNeuralOperator(
        input_key="u",
        output_key="integral",
        edge_weight_key=None,
        source_measure_key="quadrature_weight",
        normalize=False,
        target_node_type=1,
    )
    return operator(graph).nodes["integral"][target]


@pytest.mark.parametrize(
    "estimate",
    (_local_integral_estimate, _graph_integral_estimate),
)
def test_local_and_graph_integrals_have_midpoint_continuum_convergence(estimate):
    coarse = jnp.abs(estimate(8) - 1.0 / 3.0)
    fine = jnp.abs(estimate(64) - 1.0 / 3.0)
    assert fine < 0.02 * coarse


def test_graph_integral_is_invariant_to_source_permutation():
    count = 11
    coordinates = (jnp.arange(count, dtype=float) + 0.5) / count
    permutation = jnp.array([7, 1, 9, 0, 5, 10, 2, 6, 3, 8, 4])

    def evaluate(order):
        target = count
        graph = phx.graph.GraphIR(
            nodes={
                "u": jnp.concatenate((coordinates[order] ** 2, jnp.zeros((1,)))),
                "quadrature_weight": jnp.concatenate(
                    (jnp.full((count,), 1.0 / count), jnp.zeros((1,)))
                ),
                "type": jnp.concatenate(
                    (
                        jnp.zeros((count,), dtype=jnp.int32),
                        jnp.ones((1,), dtype=jnp.int32),
                    )
                ),
            },
            edges={},
            senders=jnp.arange(count, dtype=jnp.int32),
            receivers=jnp.full((count,), target, dtype=jnp.int32),
            n_node=jnp.array([count + 1], dtype=jnp.int32),
            n_edge=jnp.array([count], dtype=jnp.int32),
            validate=False,
        )
        operator = phx.graph.GraphNeuralOperator(
            input_key="u",
            output_key="integral",
            edge_weight_key=None,
            source_measure_key="quadrature_weight",
            normalize=False,
            target_node_type=1,
        )
        return operator(graph).nodes["integral"][target]

    assert jnp.allclose(evaluate(jnp.arange(count)), evaluate(permutation))


def _known_exponential_laplace_operator(decay=1.3):
    model = phx.nn.operator.architectures.LaplaceTemporalOperator(
        num_poles=1,
        max_initial_frequency=0.0,
        key=jr.key(0),
    )
    log_decay = jnp.log(jnp.expm1(decay - model.min_decay))
    model = eqx.tree_at(
        lambda item: (
            item.log_decay,
            item.frequency,
            item.residue,
            item.direct_weight,
            item.bias,
        ),
        model,
        (
            jnp.array([log_decay]),
            jnp.zeros((1,)),
            jnp.full((1, 1, 1), 0.5 + 0.0j),
            jnp.zeros((1, 1)),
            jnp.zeros((1,)),
        ),
    )
    return model


def _aligned_temporal_batch(count, values=None):
    time = jnp.linspace(0.0, 2.0, count)
    if values is None:
        values = jnp.ones((count,))
    axis = phx.nn.operator.OperatorAxis("t", time)
    return phx.nn.operator.OperatorBatch(
        inputs={"u": phx.nn.operator.FunctionSamples(values=values, axes=(axis,))},
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
    )


def test_laplace_recurrence_agrees_with_direct_aligned_evaluation():
    model = _known_exponential_laplace_operator()
    batch = _aligned_temporal_batch(37)
    assert jnp.allclose(
        model.recurrent(batch),
        model(batch),
        rtol=1e-11,
        atol=1e-11,
    )


def test_laplace_quadrature_converges_to_known_exponential_convolution():
    decay = 1.3
    model = _known_exponential_laplace_operator(decay)

    def error(count):
        batch = _aligned_temporal_batch(count)
        time = batch.require_single_query().axes[0].nodes
        exact = (1.0 - jnp.exp(-decay * time)) / decay
        return jnp.linalg.norm(model.recurrent(batch) - exact)

    assert error(129) < 0.08 * error(17)


def test_laplace_recurrence_is_stable_and_differentiable_for_long_sequences():
    count = 1025
    model = _known_exponential_laplace_operator()
    values = jnp.sin(jnp.linspace(0.0, 20.0, count))
    batch = _aligned_temporal_batch(count, values)

    output = eqx.filter_jit(model.recurrent)(batch)
    gradient = jax.grad(
        lambda source_values: jnp.sum(
            model.recurrent(_aligned_temporal_batch(count, source_values)) ** 2
        )
    )(values)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.max(jnp.abs(output)) < 2.0


def _sphere_axes(n_theta, n_phi):
    theta = phx.nn.operator.OperatorAxis(
        "theta",
        jnp.linspace(0.0, jnp.pi, n_theta),
        basis="sphere",
    )
    phi = phx.nn.operator.OperatorAxis(
        "phi",
        jnp.linspace(0.0, 2.0 * jnp.pi, n_phi, endpoint=False),
        quadrature_weights=jnp.full((n_phi,), 2.0 * jnp.pi / n_phi),
        basis="fourier",
        periodic=True,
    )
    return theta, phi


def _degree_filter(gains, n_theta=48, n_phi=64):
    axes = _sphere_axes(n_theta, n_phi)
    layer = phx.nn.operator.architectures.SphericalSpectralConv(
        in_channels=1,
        out_channels=1,
        max_degree=len(gains),
        key=jr.key(0),
    )
    weight = jnp.asarray(gains, dtype=float)[:, None, None]
    return eqx.tree_at(lambda item: item.weight, layer, weight), axes


def _harmonic(degree, order, axes, max_degree):
    theta, phi = jnp.meshgrid(axes[0].nodes, axes[1].nodes, indexing="ij")
    degrees = jnp.array([degree], dtype=jnp.int32)
    orders = jnp.array([order], dtype=jnp.int32)
    flattened = jax.vmap(
        lambda colatitude, longitude: sph_harm_y(
            degrees,
            orders,
            colatitude,
            longitude,
            n_max=max_degree,
        )
    )(theta.reshape((-1,)), phi.reshape((-1,)))
    return flattened[..., 0].reshape(theta.shape)


def test_spherical_constant_mode_has_quadrature_convergence():
    def error(n_theta):
        layer, axes = _degree_filter((1.0, 0.0, 0.0), n_theta, 2 * n_theta)
        values = jnp.ones((n_theta, 2 * n_theta, 1))
        return jnp.linalg.norm(layer(values, axes)[..., 0] - 1.0) / jnp.sqrt(values.size)

    assert error(65) < 0.3 * error(17)
    assert error(65) < 5e-4


def test_spherical_filter_applies_one_gain_per_harmonic_degree():
    layer, axes = _degree_filter((0.0, 0.0, 1.7, 0.0), 65, 96)
    mode = jnp.real(_harmonic(2, 1, axes, 3))
    output = layer(mode[..., None], axes)[..., 0]
    relative_error = jnp.linalg.norm(output - 1.7 * mode) / jnp.linalg.norm(mode)
    assert relative_error < 1e-3


def test_spherical_operator_is_equivariant_to_longitude_rotations():
    layer, axes = _degree_filter((0.8, -0.3, 1.2, 0.4), 33, 48)
    values = (
        jnp.real(_harmonic(1, 1, axes, 3)) + 0.3 * jnp.real(_harmonic(3, -2, axes, 3))
    )[..., None]
    shift = 7
    expected = jnp.roll(layer(values, axes), shift, axis=1)
    actual = layer(jnp.roll(values, shift, axis=1), axes)
    assert jnp.allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_spherical_degree_filter_is_equivariant_to_arbitrary_rotation():
    layer, axes = _degree_filter((0.0, 0.0, 1.7, 0.0), 65, 96)
    theta, phi = jnp.meshgrid(axes[0].nodes, axes[1].nodes, indexing="ij")
    points = jnp.stack(
        (
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ),
        axis=-1,
    )
    alpha, beta = 0.47, -0.81
    rotation_z = jnp.array(
        [
            [jnp.cos(alpha), -jnp.sin(alpha), 0.0],
            [jnp.sin(alpha), jnp.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation_y = jnp.array(
        [
            [jnp.cos(beta), 0.0, jnp.sin(beta)],
            [0.0, 1.0, 0.0],
            [-jnp.sin(beta), 0.0, jnp.cos(beta)],
        ]
    )
    rotated = oe.contract("...i,ij->...j", points, rotation_z @ rotation_y)
    x, y, z = rotated[..., 0], rotated[..., 1], rotated[..., 2]
    degree_two_field = x * y + 0.3 * (y**2 - z**2)
    output = layer(degree_two_field[..., None], axes)[..., 0]
    relative_error = jnp.linalg.norm(output - 1.7 * degree_two_field) / jnp.linalg.norm(
        degree_two_field
    )
    assert relative_error < 2e-3


def _attention_samples(weights, mask=None):
    count = len(weights)
    return phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=jnp.arange(count, dtype=float)[:, None],
        quadrature_weights=jnp.asarray(weights),
        mask=mask,
    )


def test_operator_attention_is_permutation_equivariant():
    values = jr.normal(jr.key(20), (7, 3))
    weights = jnp.array([0.05, 0.1, 0.15, 0.2, 0.1, 0.25, 0.15])
    samples = _attention_samples(weights)
    attention = phx.nn.operator.layers.OperatorAttention(
        source_channels=3,
        num_heads=2,
        head_dim=4,
        key=jr.key(21),
    )
    reference = attention(values, samples)
    permutation = jnp.array([4, 0, 6, 2, 1, 5, 3])
    permuted_samples = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=samples.coordinates[permutation],
        quadrature_weights=weights[permutation],
    )
    actual = attention(values[permutation], permuted_samples)
    assert jnp.allclose(actual, reference[permutation], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("kind", ("operator", "slice"))
def test_operator_attention_is_invariant_to_masked_padding(kind):
    values = jr.normal(jr.key(22), (5, 3))
    samples = _attention_samples(jnp.full((5,), 0.2))
    padded_values = jnp.concatenate((values, jnp.full((3, 3), 1e10)), axis=0)
    padded_samples = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=jnp.arange(8, dtype=float)[:, None],
        quadrature_weights=jnp.concatenate((jnp.full((5,), 0.2), jnp.ones((3,)))),
        mask=jnp.array([True, True, True, True, True, False, False, False]),
    )
    if kind == "operator":
        attention = phx.nn.operator.layers.OperatorAttention(
            source_channels=3,
            num_heads=2,
            head_dim=4,
            key=jr.key(23),
        )
    else:
        attention = phx.nn.operator.layers.SliceAttention(
            channels=3,
            num_slices=4,
            num_heads=2,
            head_dim=4,
            key=jr.key(23),
        )
    reference = attention(values, samples)
    actual = attention(padded_values, padded_samples)
    assert jnp.allclose(actual[:5], reference, rtol=1e-10, atol=1e-10)
    assert jnp.allclose(actual[5:], 0.0)


def test_cross_attention_is_continuum_consistent_under_measure_splitting():
    attention = phx.nn.operator.layers.OperatorAttention(
        source_channels=2,
        query_channels=2,
        num_heads=2,
        head_dim=3,
        key=jr.key(24),
    )
    source_values = jnp.array([[1.0, -0.5], [0.2, 2.0]])
    source = _attention_samples(jnp.array([0.3, 0.7]))
    query_values = jnp.array([[0.1, 0.2], [0.4, -0.3], [1.0, 0.0]])
    query = _attention_samples(jnp.ones((3,)))
    reference = attention.cross(source_values, query_values, source, query)

    split_values = jnp.array([[1.0, -0.5], [1.0, -0.5], [0.2, 2.0]])
    split_source = _attention_samples(jnp.array([0.15, 0.15, 0.7]))
    actual = attention.cross(split_values, query_values, split_source, query)
    assert jnp.allclose(actual, reference, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("kind", ("deeponet", "local"))
def test_quadrature_operator_value_gradients_equal_analytic_weights(kind):
    coordinates = jnp.array([[0.05], [0.2], [0.55], [0.9]])
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    values = jnp.array([0.7, -0.2, 1.3, 2.0])
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=jnp.array([[0.37]]),
    )
    if kind == "deeponet":
        model = phx.nn.operator.architectures.DeepONet(
            branch=_integral_encoder(),
            trunk=_ConstantTrunk(),
            coord_dim=1,
            latent_size=1,
        )
    else:
        model = phx.nn.operator.architectures.LocalIntegralOperator(
            kernel_model=_SourceValueKernel(),
            coord_dim=1,
        )

    def evaluate(source_values):
        batch = phx.nn.operator.OperatorBatch(
            inputs={
                "u": phx.nn.operator.FunctionSamples(
                    values=source_values,
                    coordinates=coordinates,
                    quadrature_weights=weights,
                )
            },
            queries={"query": query},
        )
        return model(batch)[0]

    gradient = jax.grad(evaluate)(values)
    assert jnp.allclose(gradient, weights, rtol=1e-11, atol=1e-11)


def test_laplace_terminal_gradient_matches_trapezoidal_convolution_weights():
    decay = 1.3
    count = 9
    model = _known_exponential_laplace_operator(decay)
    time = jnp.linspace(0.0, 2.0, count)
    values = jr.normal(jr.key(30), (count,))
    gradient = jax.grad(
        lambda source_values: model.recurrent(
            _aligned_temporal_batch(count, source_values)
        )[-1]
    )(values)

    delta = jnp.diff(time)
    expected = jnp.zeros((count,))
    expected = expected.at[0].set(0.5 * delta[0] * jnp.exp(-decay * (time[-1] - time[0])))
    expected = expected.at[-1].set(0.5 * delta[-1])
    expected = expected.at[1:-1].set(
        0.5 * (delta[:-1] + delta[1:]) * jnp.exp(-decay * (time[-1] - time[1:-1]))
    )
    assert jnp.allclose(gradient, expected, rtol=1e-11, atol=1e-11)


def test_retained_fourier_mode_has_identity_energy_gradient():
    count = 32
    x = jnp.arange(count, dtype=float)
    signal = (
        0.7 * jnp.cos(2.0 * jnp.pi * x / count)
        + 0.2 * jnp.sin(2.0 * jnp.pi * 3.0 * x / count)
    )[:, None]
    layer = phx.nn.operator.architectures.SpectralConvND(
        in_channels=1,
        out_channels=1,
        n_modes=5,
        key=jr.key(31),
    )
    layer = eqx.tree_at(lambda item: item.weight, layer, jnp.ones_like(layer.weight))
    gradient = jax.grad(lambda values: 0.5 * jnp.sum(layer(values) ** 2))(signal)
    assert jnp.allclose(gradient, signal, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("kind", ("integral", "differential"))
def test_sparse_neighbor_execution_matches_dense_radius_operator(kind):
    source_x = jnp.linspace(0.0, 1.0, 64)
    query_x = jnp.linspace(0.03, 0.97, 31)
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "u": phx.nn.operator.FunctionSamples(
                values=jnp.sin(2.0 * jnp.pi * source_x),
                coordinates=source_x[:, None],
                quadrature_weights=jnp.full((64,), 1.0 / 64.0),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=query_x[:, None],
            )
        },
    )
    if kind == "integral":
        dense = phx.nn.operator.architectures.LocalIntegralOperator(
            kernel_model=_SourceValueKernel(),
            coord_dim=1,
            radius=0.08,
        )
        sparse = phx.nn.operator.architectures.LocalIntegralOperator(
            kernel_model=_SourceValueKernel(),
            coord_dim=1,
            radius=0.08,
            max_neighbors=12,
        )
    else:
        dense = phx.nn.operator.architectures.LocalDifferentialOperator(
            kernel_model=_ConstantDifferentialKernel(),
            coord_dim=1,
            radius=0.08,
        )
        sparse = phx.nn.operator.architectures.LocalDifferentialOperator(
            kernel_model=_ConstantDifferentialKernel(),
            coord_dim=1,
            radius=0.08,
            max_neighbors=12,
        )
    assert jnp.allclose(sparse(batch), dense(batch), rtol=1e-11, atol=1e-11)


def test_basis_transform_plan_reuses_exact_projection_matrices():
    nodes = jnp.linspace(0.0, 1.0, 41) ** 1.3
    axis = phx.nn.operator.OperatorAxis("x", nodes, basis="legendre")
    layer, _ = _identity_basis_layer("legendre", 8, axis)
    values = jnp.exp(nodes)[:, None]
    plan = layer.plan((axis,))
    assert isinstance(plan, phx.nn.operator.layers.BasisTransformPlan)
    expected = layer(values, (axis,))
    actual = eqx.filter_jit(lambda x: layer(x, (axis,), plan=plan))(values)
    assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_spherical_transform_plan_reuses_basis_and_quadrature():
    layer, axes = _degree_filter((0.8, -0.3, 1.2, 0.4), 25, 36)
    values = jr.normal(jr.key(40), (25, 36, 1))
    plan = layer.plan(axes)
    assert isinstance(plan, phx.nn.operator.architectures.SphericalTransformPlan)
    expected = layer(values, axes)
    actual = eqx.filter_jit(lambda x: layer(x, axes, plan=plan))(values)
    assert jnp.allclose(actual, expected, rtol=1e-12, atol=1e-12)
