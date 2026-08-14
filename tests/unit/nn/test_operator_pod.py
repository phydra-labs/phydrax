#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _operator_pod_dataset():
    source_axis = phx.nn.operator.OperatorAxis("sensor", jnp.array([0.0]))
    query_axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.array([0.0, 0.2, 0.7, 1.0]),
        quadrature_weights=jnp.array([0.1, 0.3, 0.4, 0.2]),
    )
    coefficients = jnp.array(
        [[-2.0, 0.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 1.0], [2.0, -1.0]]
    )
    modes = jnp.array([[1.0, 0.5, -0.2, 0.8], [0.0, 1.0, 0.4, -0.5]])
    spatial_mean = jnp.array([3.0, -2.0, 1.0, 0.5])
    targets = coefficients @ modes + spatial_mean
    dataset = phx.nn.operator.training.operator_dataset_from_arrays(
        {"source": jnp.linspace(-1.0, 1.0, targets.shape[0])[:, None]},
        {"state": targets},
        source_axes={"source": (source_axis,)},
        query_axes=(query_axis,),
    )
    return dataset, source_axis, query_axis, targets


def _deeponet(branch, basis):
    return phx.nn.operator.architectures.DeepONet(
        branch=branch,
        trunk=basis,
        coord_dim=1,
        latent_size=2,
        out_size="scalar",
        in_size="scalar",
    )


def test_operator_pod_uses_physical_kernel_preserves_layout_and_reconstructs_centered_data():
    dataset, _source_axis, query_axis, targets = _operator_pod_dataset()
    fitted = phx.nn.operator.training.fit_operator_pod(
        dataset,
        "state",
        2,
        centered=True,
        differentiate="basis",
        require_physical_quadrature=True,
    )
    coefficients = fitted.transform(targets)
    reconstruction = fitted.inverse_transform(coefficients)
    metric = jnp.diag(query_axis.quadrature_weights)
    gram = fitted.components @ metric @ jnp.conj(fitted.components).T

    assert fitted.basis.has_offset
    assert fitted.basis.offset.shape == (4, 1)
    assert jnp.allclose(reconstruction, targets, atol=3e-5)
    assert jnp.allclose(gram, jnp.eye(2), atol=3e-5)
    assert fitted.query_name == "query"
    assert fitted.field_name == "state"
    assert fitted.sample_shape == (4,)
    assert (
        fitted.geometry_fingerprint == dataset.batch.query("query").geometry_fingerprint()
    )
    assert fitted.diagnostics.geometry_fingerprint == fitted.geometry_fingerprint
    assert fitted.diagnostics.query_layout_provenance[-1].endswith(
        fitted.geometry_fingerprint
    )
    assert fitted.diagnostics.centering_provenance == "fixed-spatial-snapshot-mean"
    assert fitted.diagnostics.weighted_orthogonality_error < 3e-5
    assert fitted.gradient_contract.fit_mode == "spectral"
    assert fitted.diagnostics.basis_gradient_supported == (
        fitted.valid & ~fitted.diagnostics.repeated_spectrum
    )


def test_centered_pod_deeponet_adds_fixed_spatial_mean_not_channel_bias():
    dataset, _source_axis, _query_axis, _targets = _operator_pod_dataset()
    fitted = phx.nn.operator.training.fit_pod_basis(dataset, "state", 2, centered=True)
    branch = phx.nn.models.MLP(
        in_size=1,
        out_size=2,
        width_size=4,
        depth=1,
        key=jax.random.key(0),
    )
    centered = _deeponet(branch, fitted.basis)
    legacy_basis = phx.nn.operator.architectures.PODBasis(
        fitted.basis.values,
        latent_size=2,
        out_size="scalar",
    )
    legacy = _deeponet(branch, legacy_basis)

    centered_output = centered(dataset.batch)
    legacy_output = legacy(dataset.batch)
    expected_mean = jnp.broadcast_to(
        fitted.spatial_mean.reshape((1, 4)), centered_output.shape
    )

    assert jnp.allclose(centered.bias, 0.0)
    assert jnp.allclose(legacy.bias, 0.0)
    assert jnp.allclose(centered_output - legacy_output, expected_mean, atol=1e-6)
    assert eqx.filter_jit(centered)(dataset.batch).shape == centered_output.shape


def test_legacy_uncentered_pod_basis_remains_shape_compatible_and_has_no_affine_offset():
    dataset, _source_axis, _query_axis, _targets = _operator_pod_dataset()
    fitted = phx.nn.operator.training.fit_operator_pod(
        dataset, "state", 2, centered=False
    )
    legacy = phx.nn.operator.architectures.PODBasis(jnp.ones((4, 2)), latent_size=2)

    assert not fitted.basis.has_offset
    assert jnp.allclose(fitted.spatial_mean, 0.0)
    assert not legacy.has_offset
    assert legacy.values.shape == (4, 1, 2)
    assert legacy.evaluate(dataset.batch.query("query")).shape == (4, 1, 2)
    assert jnp.allclose(legacy.evaluate_offset(dataset.batch.query("query")), 0.0)


def test_pod_basis_rejects_changed_fixed_query_nodes_weights_and_case_dependent_layout():
    dataset, _source_axis, query_axis, _targets = _operator_pod_dataset()
    fitted = phx.nn.operator.training.fit_operator_pod(dataset, "state", 2, centered=True)
    changed_nodes = phx.nn.operator.FunctionSamples(
        values=None,
        axes=(
            phx.nn.operator.OperatorAxis(
                "x",
                query_axis.nodes.at[1].set(0.25),
                quadrature_weights=query_axis.quadrature_weights,
            ),
        ),
    )
    changed_weights = phx.nn.operator.FunctionSamples(
        values=None,
        axes=(
            phx.nn.operator.OperatorAxis(
                "x",
                query_axis.nodes,
                quadrature_weights=query_axis.quadrature_weights.at[0].set(0.2),
            ),
        ),
    )
    with pytest.raises(Exception, match="axis nodes"):
        fitted.basis.evaluate(changed_nodes)
    with pytest.raises(Exception, match="quadrature or mask"):
        fitted.basis.evaluate(changed_weights)

    point_query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=jnp.stack(
            (
                jnp.linspace(0.0, 1.0, 4)[:, None],
                jnp.linspace(0.1, 1.1, 4)[:, None],
            ),
            axis=0,
        ),
    )
    with pytest.raises(ValueError, match="shared rather than case-dependent"):
        phx.nn.operator.architectures.PODBasis(
            jnp.ones((4, 1, 2)),
            latent_size=2,
            query_layout=point_query,
        )


def test_centered_pod_deeponet_prediction_and_operator_fit_gradients_are_finite():
    dataset, source_axis, query_axis, targets = _operator_pod_dataset()
    fitted = phx.nn.operator.training.fit_operator_pod(dataset, "state", 2, centered=True)
    branch = phx.nn.models.MLP(
        in_size=1,
        out_size=2,
        width_size=4,
        depth=1,
        key=jax.random.key(2),
    )
    model = _deeponet(branch, fitted.basis)

    def prediction_loss(source_values):
        batch = phx.nn.operator.OperatorBatch(
            inputs={
                "source": phx.nn.operator.FunctionSamples(
                    values=source_values[:, None], axes=(source_axis,)
                )
            },
            queries={
                "query": phx.nn.operator.FunctionSamples(values=None, axes=(query_axis,))
            },
            case_axes=("case",),
        )
        return jnp.sum(jnp.square(model(batch)))

    source_gradient = jax.grad(prediction_loss)(jnp.linspace(-1.0, 1.0, dataset.size))

    def fit_feature_loss(output_values, snapshot_weight):
        replaced_targets = phx.nn.operator.OperatorTargetBatch.from_arrays(
            {"state": output_values}, dataset.batch
        )
        replaced = phx.nn.operator.training.OperatorDataset(
            dataset.batch, replaced_targets
        )
        result = phx.nn.operator.training.fit_operator_pod(
            replaced,
            "state",
            2,
            centered=True,
            sample_weight=snapshot_weight,
        )
        return jnp.sum(jnp.square(result.transform(output_values[:2])))

    target_gradient, weight_gradient = jax.grad(fit_feature_loss, argnums=(0, 1))(
        targets, jnp.ones((targets.shape[0],))
    )
    assert jnp.all(jnp.isfinite(source_gradient))
    assert jnp.all(jnp.isfinite(target_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))
