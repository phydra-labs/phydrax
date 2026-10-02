#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array

import phydrax as phx
from phydrax.ml._contracts import ML_INFEASIBLE, ML_NONFINITE


def _operator_pod_dataset() -> tuple[
    phx.nn.operator.training.OperatorDataset,
    phx.nn.operator.OperatorAxis,
    phx.nn.operator.OperatorAxis,
    Array,
]:
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


def _deeponet(
    branch: phx.nn.models.MLP,
    basis: phx.nn.operator.architectures.PODBasis,
) -> phx.nn.operator.architectures.DeepONet:
    return phx.nn.operator.architectures.DeepONet(
        branch=branch,
        trunk=basis,
        coord_dim=1,
        latent_size=2,
        out_size="scalar",
        in_size="scalar",
    )


def test_operator_pod_scenario_1() -> None:
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
    assert query_axis.quadrature_weights is not None
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
    assert fitted.derivative_contract.route is phx.DerivativeRoute.SPECTRAL
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

    assert jnp.allclose(centered_output - legacy_output, expected_mean, atol=1e-6)
    assert eqx.filter_jit(centered)(dataset.batch).shape == centered_output.shape
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


def test_pod_basis_rejects_changed_fixed_query_nodes_weights_and_case_dependent_layout() -> (
    None
):
    dataset, _source_axis, query_axis, _targets = _operator_pod_dataset()
    fitted = phx.nn.operator.training.fit_operator_pod(dataset, "state", 2, centered=True)
    assert query_axis.quadrature_weights is not None
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


def test_centered_pod_deeponet_prediction_input_gradient_is_finite() -> None:
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

    def prediction_loss(source_values: Array) -> Array:
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

    source_values = jnp.linspace(-1.0, 1.0, dataset.size)
    direction = jnp.array([0.2, -0.4, 0.3, 0.5, -0.1])
    source_gradient = jax.grad(prediction_loss)(source_values)
    step = 1e-3
    finite_difference = (
        prediction_loss(source_values + step * direction)
        - prediction_loss(source_values - step * direction)
    ) / (2 * step)
    assert jnp.allclose(
        source_gradient @ direction, finite_difference, atol=2e-3, rtol=2e-3
    )


def _spectral_dataset(
    values: Array, weights: Array
) -> phx.nn.operator.training.OperatorDataset:
    source = phx.nn.operator.OperatorAxis("sensor", jnp.array([0.0]))
    query = phx.nn.operator.OperatorAxis(
        "x", jnp.arange(weights.size, dtype=weights.dtype), quadrature_weights=weights
    )
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"source": jnp.arange(values.shape[0], dtype=weights.dtype)[:, None]},
        {"state": values},
        source_axes={"source": (source,)},
        query_axes=(query,),
    )


def _replace_outputs(
    dataset: phx.nn.operator.training.OperatorDataset, values: Array
) -> phx.nn.operator.training.OperatorDataset:
    return phx.nn.operator.training.OperatorDataset(
        dataset.batch,
        phx.nn.operator.OperatorTargetBatch.from_arrays({"state": values}, dataset.batch),
    )


def test_repeated_retained_cluster_physical_projector_jvp_and_vjp() -> None:
    metric = jnp.array([0.5, 1.0, 2.0])
    rows = jnp.diag(jnp.array([3.0, 3.0, 1.0])) / jnp.sqrt(metric)
    values = jnp.concatenate((rows, -rows), axis=0)
    dataset = _spectral_dataset(values, metric)
    probe = jnp.array([[0.4, -0.7, 1.3], [1.1, 0.2, -0.5]])
    contraction = jnp.array([[0.8, 0.3, -0.6], [-0.4, 1.2, 0.9]])
    direction = jnp.zeros_like(values).at[0, 2].set(0.7).at[4, 0].set(-0.2)

    def loss(outputs: Array) -> Array:
        fit = phx.nn.operator.training.fit_operator_pod(
            _replace_outputs(dataset, outputs), "state", 2, centered=True
        )
        return jnp.sum(fit.project(probe) * contraction)

    fit = phx.nn.operator.training.fit_operator_pod(dataset, "state", 2, centered=True)
    assert fit.diagnostics.repeated_spectrum
    assert fit.diagnostics.projector_gradient_supported
    assert not fit.diagnostics.basis_gradient_supported
    assert jnp.allclose(
        fit.project(probe), fit.inverse_transform(fit.transform(probe)), atol=2e-6
    )
    assert jnp.allclose(
        (probe - fit.spatial_mean) @ fit.projector() + fit.spatial_mean,
        fit.project(probe),
        atol=2e-6,
    )
    request = phx.DifferentiationRequest((phx.DerivativeSurface.FIT_FEATURES,))
    admission = fit.require_derivative(request, operation="project")
    assert admission.runtime_valid is not None
    assert jnp.all(admission.runtime_valid)
    assert not fit.derivative_admission(request).supported
    assert not fit.derivative_admission(request, operation="decoder").supported
    tangent = jax.jvp(loss, (values,), (direction,))[1]
    reverse = jnp.sum(jax.grad(loss)(values) * direction)
    step = 1e-3
    difference = (loss(values + step * direction) - loss(values - step * direction)) / (
        2 * step
    )
    assert jnp.abs(difference) > 1e-3
    assert jnp.allclose(tangent, difference, atol=3e-3, rtol=3e-3)
    assert jnp.allclose(reverse, difference, atol=3e-3, rtol=3e-3)


def test_projection_fit_weight_derivative_and_projector_decoder_mode_split() -> None:
    dataset, _, _, values = _operator_pod_dataset()
    weights = jnp.array([0.8, 1.1, 0.9, 1.3, 0.7])
    direction = jnp.array([0.2, -0.4, 0.1, 0.3, -0.2])
    probe = jnp.array([0.4, -0.7, 1.1, 0.2])
    contraction = jnp.array([0.3, 0.8, -0.4, 1.2])

    def loss(w: Array) -> Array:
        fit = phx.nn.operator.training.fit_operator_pod(
            dataset, "state", 1, centered=True, sample_weight=w
        )
        return fit.project(probe) @ contraction

    tangent = jax.jvp(loss, (weights,), (direction,))[1]
    step = 1e-3
    difference = (loss(weights + step * direction) - loss(weights - step * direction)) / (
        2 * step
    )
    assert jnp.abs(difference) > 1e-3
    assert jnp.allclose(tangent, difference, atol=3e-3, rtol=3e-3)

    def decoder_loss(outputs: Array) -> Array:
        fit = phx.nn.operator.training.fit_operator_pod(
            _replace_outputs(dataset, outputs),
            "state",
            1,
            centered=False,
            differentiate="projector",
        )
        return fit.inverse_transform(jnp.array([0.7])) @ contraction

    assert jnp.allclose(jax.grad(decoder_loss)(values), 0.0)


def test_basis_decoder_fit_derivative_and_independent_current_basis_updates() -> None:
    rows = jnp.diag(jnp.array([4.0, 2.0, 0.5]))
    values = jnp.concatenate((rows, -rows), axis=0)
    dataset = _spectral_dataset(values, jnp.ones(3))
    direction = jnp.zeros_like(values).at[0, 2].set(0.4)
    probe = jnp.array([0.3, -0.8, 1.2])

    def loss(outputs: Array) -> Array:
        fit = phx.nn.operator.training.fit_operator_pod(
            _replace_outputs(dataset, outputs), "state", 1, differentiate="basis"
        )
        return fit.inverse_transform(jnp.array([0.7])) @ probe

    tangent = jax.jvp(loss, (values,), (direction,))[1]
    step = 1e-3
    difference = (loss(values + step * direction) - loss(values - step * direction)) / (
        2 * step
    )
    assert jnp.abs(difference) > 1e-3
    assert jnp.allclose(tangent, difference, atol=2e-3, rtol=2e-3)
    fit = phx.nn.operator.training.fit_operator_pod(
        dataset, "state", 1, differentiate="none"
    )
    replacement = fit.basis.weighted_values.at[0, 0, 0].set(0.8).at[2, 0, 0].set(0.4)
    updated = eqx.tree_at(lambda item: item.basis.weighted_values, fit, replacement)
    assert jnp.allclose(
        updated.project(probe),
        updated.inverse_transform(updated.transform(probe)),
        atol=2e-6,
    )
    assert jnp.allclose(fit.frame_correction, 0.0)

    def parameter_loss(basis_values: Array) -> Array:
        current = eqx.tree_at(lambda item: item.basis.weighted_values, fit, basis_values)
        decoded = current.basis.decode(jnp.array([0.7]), dataset.batch.query("query"))
        return decoded[..., 0] @ probe

    parameter_direction = jnp.ones_like(replacement) * 0.2
    tangent = jax.jvp(parameter_loss, (replacement,), (parameter_direction,))[1]
    difference = (
        parameter_loss(replacement + step * parameter_direction)
        - parameter_loss(replacement - step * parameter_direction)
    ) / (2 * step)
    assert jnp.abs(difference) > 1e-3
    assert jnp.allclose(tangent, difference, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    "weight", [jnp.nan, -1.0, 0.0], ids=["nonfinite", "negative", "zero-total"]
)
def test_operator_pod_invalid_snapshot_weights_preserve_wrapper_precedence(
    weight: float,
) -> None:
    dataset, _, _, _ = _operator_pod_dataset()
    weights = jnp.full((dataset.size,), weight)
    fit = phx.nn.operator.training.fit_operator_pod(
        dataset, "state", 1, sample_weight=weights
    )
    assert not fit.valid
    assert fit.status == ML_INFEASIBLE
    independent = phx.DifferentiationRequest(
        (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
    )
    assert fit.derivative_admission(independent).supported
    assert fit.derivative_admission(independent).runtime_valid is None


def test_operator_pod_nonfinite_active_samples_and_zero_physical_support() -> None:
    rows = jnp.diag(jnp.array([3.0, 2.0, 1.0]))
    values = jnp.concatenate((rows, -rows), axis=0)
    active = _spectral_dataset(values.at[0, 0].set(jnp.nan), jnp.ones(3))
    rejected = phx.nn.operator.training.fit_operator_pod(active, "state", 1)
    assert rejected.status == ML_NONFINITE
    masked_values = values.at[:, 2].set(jnp.nan)
    masked = _spectral_dataset(masked_values, jnp.array([0.5, 1.0, 0.0]))
    fit = phx.nn.operator.training.fit_operator_pod(masked, "state", 2)
    assert fit.valid
    projected = fit.project(jnp.array([0.2, 0.8, jnp.nan]))
    assert jnp.allclose(projected, jnp.array([0.2, 0.8, 0.0]), atol=2e-6)
    zero_metric = _spectral_dataset(values, jnp.zeros(3))
    invalid = phx.nn.operator.training.fit_operator_pod(zero_metric, "state", 1)
    assert invalid.status == ML_INFEASIBLE


def test_operator_pod_boundary_tie_refuses_projector_fit_derivative() -> None:
    rows = jnp.diag(jnp.array([3.0, 2.0, 2.0]))
    values = jnp.concatenate((rows, -rows), axis=0)
    fit = phx.nn.operator.training.fit_operator_pod(
        _spectral_dataset(values, jnp.ones(3)), "state", 2
    )
    request = phx.DifferentiationRequest((phx.DerivativeSurface.FIT_FEATURES,))
    admission = fit.derivative_admission(request, operation="project")
    assert admission.runtime_valid is not None
    assert not jnp.all(admission.runtime_valid)
    assert not fit.diagnostics.projector_gradient_supported
    with pytest.raises(Exception, match="DERIVATIVE|derivative"):
        fit.require_derivative(request, operation="project")
    with pytest.raises(ValueError, match="Unknown"):
        fit.derivative_admission(request, operation="evaluate_everywhere")


def test_operator_pod_randomized_options_and_fixed_key_projection() -> None:
    rows = jnp.diag(jnp.array([4.0, 2.0, 0.5]))
    values = jnp.concatenate((rows, -rows), axis=0)
    dataset = _spectral_dataset(values, jnp.array([0.5, 1.0, 2.0]))
    method = phx.linalg.svd.RandomizedSVD(oversampling=1, power_iterations=1)
    key = jax.random.key(42)
    fit = phx.nn.operator.training.fit_operator_pod(
        dataset, "state", 2, method=method, key=key
    )
    replay = phx.nn.operator.training.fit_operator_pod(
        dataset, "state", 2, method=method, key=key
    )
    exact = phx.nn.operator.training.fit_operator_pod(dataset, "state", 2)
    probe = jnp.array([0.3, -0.8, 1.2])
    assert fit.valid
    assert fit.diagnostics.leading_certified
    assert jnp.allclose(fit.project(probe), replay.project(probe), atol=2e-6)
    assert jnp.allclose(fit.project(probe), exact.project(probe), atol=3e-5)
    direction = jnp.zeros_like(values).at[0, 2].set(0.4)
    contraction = jnp.array([0.8, 0.2, -0.7])

    def loss(outputs: Array) -> Array:
        result = phx.nn.operator.training.fit_operator_pod(
            _replace_outputs(dataset, outputs), "state", 2, method=method, key=key
        )
        return result.project(probe) @ contraction

    tangent = jax.jvp(loss, (values,), (direction,))[1]
    step = 1e-3
    difference = (loss(values + step * direction) - loss(values - step * direction)) / (
        2 * step
    )
    assert jnp.abs(difference) > 1e-3
    assert jnp.allclose(tangent, difference, atol=3e-3, rtol=3e-3)
    with pytest.raises((TypeError, ValueError), match="key"):
        phx.nn.operator.training.fit_operator_pod(dataset, "state", 2, method=method)
    with pytest.raises((TypeError, ValueError)):
        phx.nn.operator.training.fit_operator_pod(
            dataset,
            "state",
            2,
            differentiate="singular-values",  # ty: ignore[invalid-argument-type]
        )


def test_physical_pod_multichannel_affine_layout_and_branch_parameter_response() -> None:
    source = phx.nn.operator.OperatorAxis("sensor", jnp.array([0.0]))
    query = phx.nn.operator.OperatorAxis(
        "x", jnp.array([0.0, 1.0]), quadrature_weights=jnp.array([0.2, 0.8])
    )
    coefficients = jnp.array(
        [[-2.0, 0.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 1.0], [2.0, -1.0]]
    )
    modes = jnp.array([[1.0, 0.2, -0.4, 0.8], [0.3, -0.7, 0.9, 0.1]])
    mean = jnp.array([0.3, 1.2, -0.8, 0.5])
    values = (coefficients @ modes + mean).reshape((5, 2, 2))
    dataset = phx.nn.operator.training.operator_dataset_from_arrays(
        {"source": jnp.linspace(-1.0, 1.0, 5)[:, None]},
        {"state": values},
        source_axes={"source": (source,)},
        query_axes=(query,),
        target_specs={"state": phx.nn.operator.OperatorOutputSpec(2)},
    )
    fit = phx.nn.operator.training.fit_operator_pod(dataset, "state", 2, centered=True)
    reconstructed = fit.inverse_transform(fit.transform(values))
    assert reconstructed.shape == values.shape
    assert jnp.allclose(reconstructed, values, atol=3e-5)
    assert jnp.allclose(fit.project(values), reconstructed, atol=3e-5)
    assert jnp.allclose(fit.physical_weights, jnp.array([0.2, 0.2, 0.8, 0.8]))
    assert jnp.allclose(
        fit.components @ jnp.diag(fit.physical_weights) @ jnp.conj(fit.components).T,
        jnp.eye(2),
        atol=3e-5,
    )
    branch = phx.nn.models.MLP(
        in_size=1, out_size=2, width_size=4, depth=1, key=jax.random.key(7)
    )
    contraction = (
        jnp.arange(values.size, dtype=values.dtype).reshape(values.shape) / values.size
    )

    def loss(scale: Array) -> Array:
        current = jax.tree.map(
            lambda leaf: leaf * scale if eqx.is_inexact_array(leaf) else leaf, branch
        )
        model = phx.nn.operator.architectures.DeepONet(
            branch=current,
            trunk=fit.basis,
            coord_dim=1,
            latent_size=2,
            out_size=2,
            in_size="scalar",
        )
        return jnp.sum(model(dataset.batch) * contraction)

    scale = jnp.asarray(1.0)
    derivative = jax.grad(loss)(scale)
    step = 1e-3
    difference = (loss(scale + step) - loss(scale - step)) / (2 * step)
    assert jnp.abs(difference) > 1e-3
    assert jnp.allclose(derivative, difference, atol=2e-3, rtol=2e-3)


def test_zero_snapshot_weight_excludes_nonfinite_outputs_without_success_sanitization() -> (
    None
):
    rows = jnp.diag(jnp.array([3.0, 2.0, 1.0]))
    values = jnp.concatenate((rows, -rows), axis=0).at[0].set(jnp.nan)
    dataset = _spectral_dataset(values, jnp.ones(3))
    weights = jnp.ones(6).at[0].set(0.0)
    fitted = phx.nn.operator.training.fit_operator_pod(
        dataset, "state", 1, sample_weight=weights, differentiate="none"
    )
    reference = phx.nn.operator.training.fit_operator_pod(
        _replace_outputs(dataset, values.at[0].set(0.0)),
        "state",
        1,
        sample_weight=weights,
        differentiate="none",
    )
    assert fitted.valid
    assert jnp.allclose(fitted.projector(), reference.projector(), atol=2e-6)


def test_query_mask_zero_extension_and_decoder_mask_geometry_refusal() -> None:
    rows = jnp.diag(jnp.array([3.0, 2.0, 1.0]))
    values = jnp.concatenate((rows, -rows), axis=0).at[:, 2].set(jnp.nan)
    source = phx.nn.operator.OperatorAxis("sensor", jnp.array([0.0]))
    axis = phx.nn.operator.OperatorAxis(
        "x", jnp.arange(3.0), quadrature_weights=jnp.array([0.2, 0.3, 0.5])
    )
    dataset = phx.nn.operator.training.operator_dataset_from_arrays(
        {"source": jnp.arange(6.0)[:, None]},
        {"state": values},
        source_axes={"source": (source,)},
        query_axes=(axis,),
        query_mask=jnp.array([True, True, False]),
    )
    fit = phx.nn.operator.training.fit_operator_pod(dataset, "state", 2)
    assert fit.valid
    assert jnp.allclose(
        fit.project(jnp.array([0.7, -0.2, jnp.nan])),
        jnp.array([0.7, -0.2, 0.0]),
        atol=2e-6,
    )
    changed = phx.nn.operator.FunctionSamples(
        values=None, axes=(axis,), mask=jnp.array([True, False, False])
    )
    with pytest.raises(Exception, match="quadrature or mask"):
        fit.basis.decode(jnp.array([0.2, 0.4]), changed)


@pytest.mark.parametrize("entry", [jnp.nan, -0.3], ids=["nonfinite", "negative"])
def test_invalid_physical_metric_retains_infeasible_wrapper_status(entry: float) -> None:
    dataset, _, _, targets = _operator_pod_dataset()
    invalid_batch = eqx.tree_at(
        lambda batch: batch.queries["query"].axes[0].quadrature_weights,
        dataset.batch,
        jnp.array([entry, 0.3, 0.4, 0.2]),
    )
    invalid = phx.nn.operator.training.OperatorDataset(
        invalid_batch,
        phx.nn.operator.OperatorTargetBatch.from_arrays(
            {"state": targets.at[0, 1].set(jnp.nan)}, invalid_batch
        ),
    )
    fit = phx.nn.operator.training.fit_operator_pod(invalid, "state", 1)
    assert not fit.valid
    assert fit.status == ML_INFEASIBLE
    assert not fit.diagnostics.projector_gradient_supported


def test_operator_weight_status_and_fit_admission_toggle_under_jit() -> None:
    dataset, _, _, _ = _operator_pod_dataset()
    request = phx.DifferentiationRequest((phx.DerivativeSurface.FIT_WEIGHTS,))

    @eqx.filter_jit
    def evidence(weights: Array) -> tuple[Array, Array]:
        fit = phx.nn.operator.training.fit_operator_pod(
            dataset, "state", 1, sample_weight=weights
        )
        admission = fit.derivative_admission(request, operation="project")
        assert admission.runtime_valid is not None
        return fit.status, admission.runtime_valid

    valid_status, valid_gate = evidence(jnp.ones(dataset.size))
    invalid_status, invalid_gate = evidence(jnp.zeros(dataset.size))
    assert valid_status == 0
    assert jnp.all(valid_gate)
    assert invalid_status == ML_INFEASIBLE
    assert not jnp.any(invalid_gate)


def test_projector_decoder_retains_only_centered_mean_fit_response_while_none_stops_it() -> (
    None
):
    dataset, _, _, values = _operator_pod_dataset()
    contraction = jnp.array([0.3, -0.8, 0.4, 1.1])

    def projector_decoder_loss(outputs: Array) -> Array:
        result = phx.nn.operator.training.fit_operator_pod(
            _replace_outputs(dataset, outputs), "state", 1, centered=True
        )
        return result.inverse_transform(jnp.array([0.7])) @ contraction

    def stopped_decoder_loss(outputs: Array) -> Array:
        result = phx.nn.operator.training.fit_operator_pod(
            _replace_outputs(dataset, outputs),
            "state",
            1,
            centered=True,
            differentiate="none",
        )
        return result.inverse_transform(jnp.array([0.7])) @ contraction

    expected = jnp.broadcast_to(contraction / values.shape[0], values.shape)
    assert jnp.allclose(jax.grad(projector_decoder_loss)(values), expected, atol=2e-6)
    assert jnp.allclose(jax.grad(stopped_decoder_loss)(values), 0.0)
