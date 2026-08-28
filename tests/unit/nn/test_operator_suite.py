#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import FrozenInstanceError
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._model import FrozenModel


class _FeatureMap(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, in_size, out_size):
        self.in_size = int(in_size)
        self.out_size = int(out_size)

    def __call__(self, value, *, key=None):
        del key
        if self.out_size == 1:
            return value[:1]
        return jnp.stack((value[0], value[0] * value[-1]))


class _ConstantDifferentialKernel(eqx.Module):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, coord_dim):
        self.in_size = int(coord_dim) + 1
        self.out_size = 1

    def __call__(self, value, *, key=None):
        del value, key
        return jnp.ones((1,), dtype=float)


def _axis(size, *, name="x", endpoint=True):
    nodes = jnp.linspace(0.0, 1.0, size, endpoint=endpoint)
    weights = jnp.ones((size,), dtype=float) / size
    return phx.nn.operator.OperatorAxis(
        name,
        nodes,
        quadrature_weights=weights,
        basis="fourier" if not endpoint else "uniform",
        periodic=not endpoint,
    )


def _grid_batch(values, axes, *, source="u", case_axes=()):
    samples = phx.nn.operator.FunctionSamples(values=values, axes=axes)
    query = phx.nn.operator.FunctionSamples(values=None, axes=axes)
    return phx.nn.operator.OperatorBatch(
        inputs={source: samples},
        queries={"query": query},
        case_axes=case_axes,
    )


def _parameter_count(model):
    return sum(
        int(leaf.size)
        for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    )


def test_operator_architecture_status_is_deeply_immutable():
    status = phx.nn.operator.operator_architecture_status("FNO")
    mutable_status: Any = status
    mutable_statuses: Any = phx.nn.operator.OPERATOR_ARCHITECTURE_STATUSES
    assert (
        phx.nn.operator.operator_architecture_status
        is phx.nn.operator.operator_architecture_status
    )
    assert (
        phx.nn.operator.OperatorArchitectureStatus
        is phx.nn.operator.OperatorArchitectureStatus
    )
    with pytest.raises(FrozenInstanceError):
        mutable_status.tier = "research"
    with pytest.raises(TypeError):
        mutable_statuses["FNO"] = status
    assert hash(status)


@pytest.mark.parametrize(
    ("alias", "canonical_name"),
    (
        ("fourier neural operator", "FNO"),
        ("higher-order Fourier neural operator", "HOFNO"),
        ("deep_operator_network", "DeepONet"),
        ("MIO Net", "MIONet"),
        ("local-integral", "LocalIntegralOperator"),
        ("operator attention", "OperatorAttention"),
        ("laplace", "LaplaceTemporalOperator"),
        ("implicit Fourier neural operator", "IFNO"),
        ("axial FNO", "AxialFactorizedFNO"),
        ("WNO", "WaveletNeuralOperator"),
        ("NOMAD", "CoordinateConditionedOperator"),
        ("universal physics transformer", "UPT"),
        ("equivariant GINO", "EqGINO"),
        ("geometry-informed flower", "GeometryInformedFlower"),
        ("domain-conditioned Flower", "GeometryInformedFlower"),
        ("conservative geometry-informed Flower", "GeometryInformedFlower"),
        ("scOT", "Poseidon"),
        ("Transolver++", "TransolverPlusPlus"),
        ("Koopman neural operator", "KoopmanTemporalOperator"),
        ("Green neural operator", "GreenKernelOperator"),
    ),
)
def test_operator_architecture_status_normalizes_aliases(alias, canonical_name):
    assert (
        phx.nn.operator.operator_architecture_status(alias)
        is phx.nn.operator.OPERATOR_ARCHITECTURE_STATUSES[canonical_name]
    )


def test_operator_architecture_tiers_and_recommendation_eligibility_are_exact():
    expected_tiers = {
        "stable": {
            "FNO",
            "TFNO",
            "DeepONet",
            "MIONet",
            "PODDeepONet",
        },
        "experimental": {
            "CNO",
            "HOFNO",
            "GraphNeuralOperator",
            "SFNO",
            "LocalDifferentialOperator",
            "LocalGlobalOperator",
            "LocalIntegralOperator",
            "OperatorAttention",
            "SliceAttention",
            "AxialOperatorAttention",
            "CodomainAttention",
            "IFNO",
            "AxialFactorizedFNO",
            "ConditionalFlowFunctionOperator",
            "LinearRecurrentOperator",
        },
        "research": {
            "Flower",
            "UNO",
            "LaplaceTemporalOperator",
            "GINO",
            "FunctionFrameReconstructor",
            "GeometryInformedFlower",
            "RIGNO",
            "GAOT",
            "WaveletNeuralOperator",
            "MultiwaveletOperator",
            "ManifoldSpectralOperator",
            "CoordinateConditionedOperator",
            "UPT",
            "CochainNeuralOperator",
            "ABUPT",
            "CoDANO",
            "EqGINO",
            "InContextOperator",
            "GaussianFunctionOperator",
            "Poseidon",
            "DPOT",
            "DiagonalStateSpaceMixer",
            "SelectiveStateSpaceMixer",
            "Transolver",
            "TransolverPlusPlus",
            "GNOT",
            "KoopmanTemporalOperator",
            "GreenKernelOperator",
            "LatticeEquivariantCNO",
            "WeightSpaceOperator",
        },
    }
    assert set(phx.nn.operator.OPERATOR_ARCHITECTURE_STATUSES) == set().union(
        *expected_tiers.values()
    )
    for tier, names in expected_tiers.items():
        statuses = {
            name: phx.nn.operator.operator_architecture_status(name) for name in names
        }
        assert all(status.tier == tier for status in statuses.values())
        assert all(
            status.recommendation_eligible == (tier == "stable")
            for status in statuses.values()
        )
        assert all(status.evidence for status in statuses.values())


def test_tfno_is_an_fno_tucker_configuration_not_an_architecture_class():
    status = phx.nn.operator.operator_architecture_status("tensorized FNO")
    assert status.name == "TFNO"
    assert status.architecture == "FNO"
    assert status.configuration == (("factorization", "tucker"),)
    assert not hasattr(phx.nn, "TFNO")


def test_mionet_is_a_deeponet_configuration():
    status = phx.nn.operator.operator_architecture_status(
        "multiple-input operator network"
    )
    assert status.architecture == "DeepONet"
    assert status.configuration == (("branch", "mapping"), ("fusion", "product"))


def test_transolver_plus_plus_is_an_overlap_configuration():
    status = phx.nn.operator.operator_architecture_status("Transolver++")
    assert status.architecture == "Transolver"
    assert status.configuration == (("slice_top_k", "greater_than_one"),)
    assert not status.recommendation_eligible


def test_pod_and_graph_operator_statuses_are_configured_explicitly():
    pod = phx.nn.operator.operator_architecture_status("POD DeepONet")
    assert pod.architecture == "DeepONet"
    assert pod.configuration == (("trunk", "pod_basis"),)
    graph = phx.nn.operator.operator_architecture_status("graph operator")
    assert graph.name == "GraphNeuralOperator"
    assert graph.tier == "experimental"
    assert not graph.recommendation_eligible


def test_operator_architecture_status_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown operator architecture"):
        phx.nn.operator.operator_architecture_status("not-an-operator")


def test_function_samples_combine_quadrature_and_mask():
    x = phx.nn.operator.OperatorAxis(
        "x", jnp.arange(3.0), quadrature_weights=jnp.array([0.2, 0.3, 0.5])
    )
    y = phx.nn.operator.OperatorAxis(
        "y", jnp.arange(2.0), quadrature_weights=jnp.array([0.4, 0.6])
    )
    mask = jnp.array([[True, False], [True, True], [False, True]])
    samples = phx.nn.operator.FunctionSamples(values=None, axes=(x, y), mask=mask)
    expected = jnp.multiply.outer(x.quadrature_weights, y.quadrature_weights) * mask
    assert jnp.allclose(samples.weights(), expected)
    assert jnp.allclose(jnp.sum(samples.weights(normalized=True)), 1.0)


def test_operator_metrics_are_per_case_and_quadrature_aware():
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.array([0.0, 0.25, 1.0]),
        quadrature_weights=jnp.array([0.1, 0.2, 0.7]),
    )
    query = phx.nn.operator.FunctionSamples(values=None, axes=(axis,))
    prediction = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    target = jnp.zeros_like(prediction)
    per_case = phx.nn.operator.operator_l2_loss(
        prediction, target, query, reduction="none"
    )
    assert jnp.allclose(per_case, jnp.array([jnp.sqrt(0.1), 2.0 * jnp.sqrt(0.7)]))
    assert jnp.allclose(
        phx.nn.operator.operator_conservation_error(
            prediction, target, query, reduction="none"
        ),
        jnp.array([0.1, 1.4]),
    )


@pytest.mark.parametrize("shape", ((9, 10), (10, 9), (9, 9), (10, 10)))
def test_spectral_conv_nd_preserves_odd_even_shapes(shape):
    layer = phx.nn.operator.architectures.SpectralConvND(
        in_channels=2,
        out_channels=3,
        n_modes=(4, 4),
        key=jr.key(sum(shape)),
    )
    output = layer(jr.normal(jr.key(0), shape + (2,)))
    assert output.shape == shape + (3,)
    assert jnp.all(jnp.isfinite(output))


def test_spectral_conv_nd_learns_negative_frequency_block():
    layer = phx.nn.operator.architectures.SpectralConvND(
        in_channels=1,
        out_channels=1,
        n_modes=(3, 3),
        key=jr.key(0),
    )
    weight = jnp.zeros_like(layer.weight)
    weight = weight.at[1, 0, 0, 2, 0].set(1.0 + 0.0j)
    layer = eqx.tree_at(lambda item: item.weight, layer, weight)
    x = jnp.arange(12.0)
    signal = jnp.cos(2.0 * jnp.pi * x / 12.0)[:, None] * jnp.ones((1, 10))
    output = layer(signal[..., None])
    assert jnp.linalg.norm(output) > 1e-6


@pytest.mark.parametrize("factorization", ("dense", "cp", "tucker"))
def test_spectral_factorizations_have_finite_gradients(factorization):
    layer = phx.nn.operator.architectures.SpectralConvND(
        in_channels=2,
        out_channels=2,
        n_modes=(3, 3),
        factorization=factorization,
        rank=2,
        key=jr.key(0),
    )
    values = jr.normal(jr.key(1), (7, 8, 2))
    gradient = eqx.filter_grad(lambda model: jnp.sum(model(values) ** 2))(layer)
    leaves = jax.tree_util.tree_leaves(eqx.filter(gradient, eqx.is_inexact_array))
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_fno_native_batch_matches_vmap_and_resolution_independent_parameters():
    model = phx.nn.operator.architectures.FNO(
        n_modes=(4, 4),
        in_channels=2,
        out_channels=2,
        width=6,
        depth=2,
        key=jr.key(0),
    )
    x_axis = jnp.linspace(0.0, 1.0, 9, endpoint=False)
    y_axis = jnp.linspace(0.0, 1.0, 10, endpoint=False)
    values = jr.normal(jr.key(1), (3, 9, 10, 2))
    native = model((values, x_axis, y_axis))
    mapped = jax.vmap(lambda value: model((value, x_axis, y_axis)))(values)
    assert jnp.allclose(native, mapped, rtol=1e-6, atol=1e-6)
    count = _parameter_count(model)
    assert model((values[:, :7, :8], x_axis[:7], y_axis[:8])).shape == (3, 7, 8, 2)
    assert _parameter_count(model) == count


def test_fno_prefers_explicit_channels_when_spatial_sizes_are_ambiguous():
    model = phx.nn.operator.architectures.FNO(
        n_modes=(2, 2),
        in_channels=4,
        out_channels=1,
        width=4,
        depth=1,
        key=jr.key(2),
    )
    axis = jnp.linspace(0.0, 1.0, 4, endpoint=False)
    values = jr.normal(jr.key(3), (2, 4, 4, 4))

    assert model((values, axis, axis)).shape == (2, 4, 4, 1)


def test_spectral_resampling_preserves_constants_and_multiscale_shape():
    values = jnp.ones((2, 15, 17, 3))
    resized = phx.nn.operator.architectures.spectral_resample(values, (8, 9))
    assert resized.shape == (2, 8, 9, 3)
    assert jnp.allclose(resized, 1.0)
    layer = phx.nn.operator.architectures.MultiScaleSpectralConvND(
        in_channels=3,
        out_channels=4,
        n_modes=(4, 4),
        scales=(1.0, 0.5),
        key=jr.key(0),
    )
    assert layer(values).shape == (2, 15, 17, 4)


@pytest.mark.parametrize("basis", ("fourier", "sine", "cosine", "legendre"))
def test_basis_spectral_policy_supports_nonuniform_nodes(basis):
    nodes = jnp.linspace(0.0, 1.0, 11) ** 2
    axis = phx.nn.operator.OperatorAxis(
        "x", nodes, quadrature_weights=jnp.gradient(nodes), basis=basis
    )
    layer = phx.nn.operator.layers.BasisSpectralConvND(
        in_channels=1,
        out_channels=2,
        n_modes=5,
        bases=basis,
        key=jr.key(0),
    )
    output = layer(jnp.sin(jnp.pi * nodes)[:, None], (axis,))
    assert output.shape == (11, 2)
    assert jnp.all(jnp.isfinite(output))


def test_integral_branch_is_permutation_invariant_and_mask_aware():
    coordinates = jnp.array([[0.0], [0.2], [0.7], [1.0]])
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])
    mask = jnp.array([True, True, False, True])
    values = jnp.array([1.0, 2.0, 1000.0, 4.0])
    encoder = phx.nn.operator.architectures.IntegralBranchEncoder(
        feature_model=_FeatureMap(2, 2),
        latent_size=2,
        coord_dim=1,
    )
    samples = phx.nn.operator.FunctionSamples(
        values=values,
        coordinates=coordinates,
        quadrature_weights=weights,
        mask=mask,
    )
    encoded = encoder(samples, case_ndim=0)
    permutation = jnp.array([3, 1, 0, 2])
    permuted = phx.nn.operator.FunctionSamples(
        values=values[permutation],
        coordinates=coordinates[permutation],
        quadrature_weights=weights[permutation],
        mask=mask[permutation],
    )
    assert jnp.allclose(encoded, encoder(permuted, case_ndim=0))
    changed_masked = phx.nn.operator.FunctionSamples(
        values=values.at[2].set(-999.0),
        coordinates=coordinates,
        quadrature_weights=weights,
        mask=mask,
    )
    assert jnp.allclose(encoded, encoder(changed_masked, case_ndim=0))


def test_mionet_product_fusion_and_pod_decode():
    sensor_axis = _axis(6, name="sensor")
    query_axis = _axis(5)
    encoder_a = phx.nn.operator.architectures.IntegralBranchEncoder(
        feature_model=_FeatureMap(2, 2), latent_size=2, coord_dim=1
    )
    encoder_b = phx.nn.operator.architectures.IntegralBranchEncoder(
        feature_model=_FeatureMap(2, 2), latent_size=2, coord_dim=1
    )
    basis = phx.nn.operator.architectures.PODBasis(jnp.ones((5, 2)), latent_size=2)
    model = phx.nn.operator.architectures.DeepONet(
        branch={"a": encoder_a, "b": encoder_b},
        trunk=basis,
        coord_dim=1,
        latent_size=2,
        fusion="product",
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "a": phx.nn.operator.FunctionSamples(
                values=jnp.ones((3, 6)), axes=(sensor_axis,)
            ),
            "b": phx.nn.operator.FunctionSamples(
                values=2.0 * jnp.ones((3, 6)), axes=(sensor_axis,)
            ),
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(values=None, axes=(query_axis,))
        },
        case_axes=("case",),
    )
    assert model(batch).shape == (3, 5)


def test_deeponet_chunked_and_unchunked_queries_agree():
    branch = phx.nn.models.MLP(
        in_size=4, out_size=5, width_size=8, depth=2, key=jr.key(0)
    )
    trunk = phx.nn.models.MLP(in_size=1, out_size=5, width_size=8, depth=2, key=jr.key(1))
    full = phx.nn.operator.architectures.DeepONet(
        branch=branch, trunk=trunk, coord_dim=1, latent_size=5
    )
    chunked = phx.nn.operator.architectures.DeepONet(
        branch=branch,
        trunk=trunk,
        coord_dim=1,
        latent_size=5,
        query_chunk_size=3,
    )
    inputs = (jnp.arange(4.0), jnp.linspace(0.0, 1.0, 11))
    assert jnp.allclose(full(inputs), chunked(inputs))


def test_deeponet_accepts_frozen_array_models():
    branch = phx.nn.models.MLP(
        in_size=4, out_size=3, width_size=8, depth=2, key=jr.key(2)
    )
    trunk = phx.nn.models.MLP(in_size=1, out_size=3, width_size=8, depth=2, key=jr.key(3))
    ordinary = phx.nn.operator.architectures.DeepONet(
        branch=branch,
        trunk=trunk,
        coord_dim=1,
        latent_size=3,
        use_bias=False,
    )
    frozen = phx.nn.operator.architectures.DeepONet(
        branch=FrozenModel(branch),
        trunk=FrozenModel(trunk),
        coord_dim=1,
        latent_size=3,
        use_bias=False,
    )
    inputs = (jnp.arange(4.0), jnp.linspace(0.0, 1.0, 7))

    assert jnp.allclose(ordinary(inputs), frozen(inputs))


def test_deeponet_bias_is_explicitly_optional():
    branch = phx.nn.models.MLP(
        in_size=4, out_size=3, width_size=8, depth=2, key=jr.key(4)
    )
    trunk = phx.nn.models.MLP(in_size=1, out_size=3, width_size=8, depth=2, key=jr.key(5))
    unbiased = phx.nn.operator.architectures.DeepONet(
        branch=branch,
        trunk=trunk,
        coord_dim=1,
        latent_size=3,
        use_bias=False,
    )
    biased = phx.nn.operator.architectures.DeepONet(
        branch=branch,
        trunk=trunk,
        coord_dim=1,
        latent_size=3,
    )
    biased = eqx.tree_at(lambda model: model.bias, biased, jnp.asarray([2.0]))
    inputs = (jnp.arange(4.0), jnp.linspace(0.0, 1.0, 7))

    assert unbiased.bias is None
    assert jnp.allclose(biased(inputs), unbiased(inputs) + 2.0)


def test_local_differential_operator_annihilates_constants():
    axis = _axis(12)
    batch = _grid_batch(jnp.ones((2, 12)), (axis,), case_axes=("case",))
    model = phx.nn.operator.architectures.LocalDifferentialOperator(
        kernel_model=_ConstantDifferentialKernel(1),
        coord_dim=1,
        radius=0.3,
    )
    assert jnp.allclose(model(batch), 0.0)


def test_laplace_operator_has_stable_poles_real_output_and_strict_causality():
    source_axis = phx.nn.operator.OperatorAxis("t", jnp.array([0.0, 0.2, 0.5, 0.8, 1.0]))
    query_axis = phx.nn.operator.OperatorAxis("t_query", jnp.array([0.05, 0.15]))
    values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = phx.nn.operator.architectures.LaplaceTemporalOperator(
        num_poles=5, key=jr.key(0)
    )

    def evaluate(source_values):
        return model(
            phx.nn.operator.OperatorBatch(
                inputs={
                    "u": phx.nn.operator.FunctionSamples(
                        values=source_values, axes=(source_axis,)
                    )
                },
                queries={
                    "query": phx.nn.operator.FunctionSamples(
                        values=None, axes=(query_axis,)
                    )
                },
            )
        )

    output = evaluate(values)
    changed_future = evaluate(values.at[1:].set(1000.0))
    assert jnp.all(jnp.real(model.poles()) < 0.0)
    assert jnp.all(jnp.isreal(output))
    assert jnp.allclose(output, changed_future)


def test_operator_attention_shapes_and_measure_aware_slice_pooling():
    axis = _axis(7)
    samples = phx.nn.operator.FunctionSamples(values=None, axes=(axis,))
    values = jr.normal(jr.key(0), (2, 7, 4))
    attention = phx.nn.operator.layers.OperatorAttention(
        source_channels=4, num_heads=2, head_dim=3, key=jr.key(1)
    )
    slices = phx.nn.operator.layers.SliceAttention(
        channels=4, num_slices=3, num_heads=2, head_dim=3, key=jr.key(2)
    )
    axial = phx.nn.operator.layers.AxialOperatorAttention(
        channels=4, num_heads=2, head_dim=3, key=jr.key(3)
    )
    assert attention(values, samples).shape == values.shape
    assert slices(values, samples).shape == values.shape
    assert axial(values, (axis,)).shape == values.shape


@pytest.mark.parametrize("model_name", ("cno", "uno"))
def test_cno_family_handles_odd_grids_and_native_batches(model_name):
    x_axis = jnp.linspace(0.0, 1.0, 15)
    y_axis = jnp.linspace(0.0, 1.0, 17)
    values = jr.normal(jr.key(0), (2, 15, 17))
    if model_name == "cno":
        model = phx.nn.operator.architectures.CNO(
            spatial_ndim=2, width=4, depth=2, key=jr.key(1)
        )
    else:
        model = phx.nn.operator.architectures.UNO(
            spatial_ndim=2, widths=(4, 6, 8), key=jr.key(1)
        )
    output = model((values, x_axis, y_axis))
    assert output.shape == values.shape
    assert jnp.all(jnp.isfinite(output))


def test_sfno_is_finite_on_its_exact_s2fft_sampling():
    space = phx.discretization.SphericalSpectralPlan(3).prepare()
    plan = space.transform
    axes = (
        phx.nn.operator.OperatorAxis(
            "theta",
            plan.theta,
            quadrature_weights=plan.theta_quadrature_weights,
        ),
        phx.nn.operator.OperatorAxis(
            "phi",
            plan.phi,
            quadrature_weights=plan.phi_quadrature_weights,
            periodic=True,
        ),
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "field": phx.nn.operator.FunctionSamples(
                values=jnp.ones((2, *plan.sample_shape)),
                axes=axes,
            )
        },
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=axes)},
        case_axes=("case",),
    )
    model = phx.nn.operator.architectures.SFNO(
        space, width=4, depth=2, source_key="field", key=jr.key(0)
    )
    count = _parameter_count(model)
    output = model(batch)
    assert output.shape == (2, *plan.sample_shape)
    assert jnp.all(jnp.isfinite(output))
    assert _parameter_count(model) == count


def test_operator_constraints_compose_data_and_physics_losses():
    axis = _axis(8)
    batch = _grid_batch(jnp.ones((2, 8)), (axis,), source="data", case_axes=("case",))
    model = phx.nn.operator.architectures.FNO(
        width=4, depth=1, n_modes=(3,), key=jr.key(0)
    )
    domain = phx.domain.DatasetDomain(jnp.ones((2, 8))) @ phx.domain.Interval1d(0.0, 1.0)
    function = domain.Model("data", "x")(model)
    data = phx.terms.OperatorDatasetTerm("u", batch, jnp.zeros((2, 8)), relative=False)
    physics = phx.terms.PhysicsInformedOperatorTerm(
        "u", batch, lambda prediction, _: prediction
    )
    suite = phx.terms.operator_term_suite(data, physics)
    assert len(suite) == 2
    assert jnp.isfinite(data.loss({"u": function}, key=jr.key(1)))
    assert jnp.isfinite(physics.loss({"u": function}, key=jr.key(1)))


def test_external_operator_manifest_roundtrip_and_adapter(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"operator-state")
    manifest = phx.nn.operator.adapters.OperatorCheckpointManifest(
        architecture="test-operator",
        model_version="1.2.0",
        source_uri="https://example.test/source",
        checkpoint_uri="https://example.test/checkpoint",
        revision="abc123",
        input_schema={"u": {"channels": 1}},
        output_schema={"y": {"channels": 1}},
        preprocessing={"layout": "case-query-channel"},
        normalization={"u": {"mean": 0.0, "std": 1.0}},
        dataset_provenance=("analytic",),
        code_license="test-only",
        weights_license="test-only",
        checkpoint_sha256=phx.nn.operator.adapters.checkpoint_sha256(checkpoint),
    )
    path = tmp_path / "operator.json"
    phx.nn.operator.adapters.save_operator_manifest(path, manifest)
    loaded = phx.nn.operator.adapters.load_operator_manifest(path)
    axis = _axis(5)
    batch = _grid_batch(jnp.arange(5.0), (axis,))
    adapter = phx.nn.operator.adapters.ExternalOperatorAdapter(
        runner=lambda payload, key: 2.0 * payload,
        input_adapter=lambda operator_batch, _: operator_batch.input("u").values,
        output_adapter=lambda output, operator_batch, _: output,
        manifest=loaded,
        in_size="scalar",
        out_size="scalar",
    )
    assert loaded.to_dict() == manifest.to_dict()
    assert "format_version" not in loaded.to_dict()
    assert jnp.allclose(adapter(batch), 2.0 * jnp.arange(5.0))
    verified = phx.nn.operator.adapters.load_external_operator_adapter(
        path,
        checkpoint,
        lambda operator_manifest, checkpoint_path: lambda payload, key: 2.0 * payload,
        input_adapter=lambda operator_batch, _: operator_batch.input("u").values,
        output_adapter=lambda output, operator_batch, _: output,
        in_size="scalar",
        out_size="scalar",
    )
    assert jnp.allclose(verified(batch), 2.0 * jnp.arange(5.0))
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        phx.nn.operator.adapters.load_external_operator_adapter(
            path,
            checkpoint,
            lambda operator_manifest, checkpoint_path: lambda payload, key: payload,
            input_adapter=lambda operator_batch, _: operator_batch.input("u").values,
            output_adapter=lambda output, operator_batch, _: output,
            in_size="scalar",
            out_size="scalar",
        )
