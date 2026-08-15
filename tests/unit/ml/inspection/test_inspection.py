#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
import pytest

from phydrax._model import AbstractArrayModel, ModelBinding
from phydrax.ml import FitDiagnostics, FitResult, GradientContract, ML_SUCCESS, MLBatch
from phydrax.ml.inspection import (
    gradient_sensitivity,
    hessian_sensitivity,
    individual_conditional_expectation,
    influence_functions,
    InfluenceFunctionResult,
    jacobian_sensitivity,
    leverage_and_cooks_distance,
    partial_dependence,
    PartialDependenceResult,
    permutation_importance,
    PermutationImportanceResult,
    RegressionInfluenceDiagnostics,
    SensitivityResult,
)


class _QuadraticModel(AbstractArrayModel):
    coefficients: jax.Array
    curvature: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, coefficients, curvature):
        self.coefficients = jnp.asarray(coefficients)
        self.curvature = jnp.asarray(curvature)
        self.in_size = int(self.coefficients.shape[0])
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        del key
        values = jnp.asarray(x)
        return oe.contract("...f,f->...", values, self.coefficients) + 0.5 * oe.contract(
            "...f,f->...", values * values, self.curvature
        )


class _LinearModel(AbstractArrayModel):
    coefficients: jax.Array
    intercept: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, coefficients, intercept=0.0):
        self.coefficients = jnp.asarray(coefficients)
        self.intercept = jnp.asarray(intercept)
        self.in_size = int(self.coefficients.shape[0])
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        del key
        return (
            oe.contract("...f,f->...", jnp.asarray(x), self.coefficients) + self.intercept
        )


class _HolomorphicSquare(AbstractArrayModel):
    in_size: int = 2
    out_size: str = "scalar"

    def __call__(self, x, /, *, key=None):
        del key
        return jnp.sum(jnp.asarray(x) ** 2, axis=-1)


class _BlockModel(AbstractArrayModel):
    in_size: int = 2
    out_size: str = "scalar"
    _input_binding = ModelBinding.blockwise()

    def __call__(self, x, /, *, key=None):
        del key
        return jnp.sum(jnp.asarray(x), axis=-1)


def _fit_result(model, contract):
    diagnostics = FitDiagnostics(valid=True, status=ML_SUCCESS, method="test-fit")
    return FitResult(
        model,
        diagnostics,
        valid=True,
        status=ML_SUCCESS,
        method="test-fit",
        gradient_contract=contract,
    )


def _batch(case=False):
    x = jnp.array([[-2.0, 0.0], [-1.0, 1.0], [0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
    if case:
        x = jnp.stack((x, x + jnp.array([0.25, -0.25])))
    model = _LinearModel(jnp.array([3.0, 0.2]), 1.0)
    y = model(x)
    return MLBatch(
        x,
        y,
        sample_mask=jnp.array([True, True, True, True, False]),
        sample_weight=jnp.array([1.0, 2.0, 1.0, 3.0, 10.0]),
        feature_mask=jnp.ones_like(x, dtype=bool).at[..., 4, 1].set(False),
    )


def test_partial_dependence_and_ice_are_structured_weighted_case_aware_and_differentiable():
    model = _QuadraticModel(jnp.array([2.0, -1.0]), jnp.array([0.5, 0.0]))
    batch = _batch(case=True)
    grid = jnp.array([-1.0, 0.0, 1.0])
    ice = individual_conditional_expectation(model, batch, (0,), grid)
    pdp = partial_dependence(model, batch, (0,), grid)
    assert isinstance(ice, PartialDependenceResult)
    assert ice.grid.shape == (3, 1)
    assert ice.ice.shape == (3, 2, 5)
    assert ice.average.shape == (3, 2)
    assert jnp.allclose(ice.average, pdp.average)
    assert jnp.allclose(
        jax.grad(
            lambda value: jnp.sum(partial_dependence(model, batch, (0,), value).average)
        )(grid),
        jnp.array([1.5, 2.0, 2.5]) * 2,
        atol=1e-5,
    )


def test_permutation_importance_is_keyed_deterministic_masked_and_structured():
    batch = _batch(case=True)
    model = _LinearModel(jnp.array([3.0, 0.2]), 1.0)
    first = permutation_importance(model, batch, key=jax.random.key(1), repeats=4)
    second = permutation_importance(model, batch, key=jax.random.key(1), repeats=4)
    assert isinstance(first, PermutationImportanceResult)
    assert first.permuted_scores.shape == (4, 2, 2)
    assert first.mean_importance.shape == (2, 2)
    assert jnp.allclose(first.importances, second.importances)
    assert jnp.all(first.mean_importance[0] > first.mean_importance[1])
    with pytest.raises(ValueError, match="explicit JAX key"):
        permutation_importance(model, batch, key=None)
    with pytest.raises(ValueError, match="greater than one"):
        permutation_importance(model, batch, key=jax.random.key(0), repeats=1)


def test_gradient_jacobian_hessian_jit_vmap_and_grad_contracts():
    model = _QuadraticModel(jnp.array([2.0, -1.0]), jnp.array([0.5, 3.0]))
    points = jnp.array([[1.0, 2.0], [2.0, -1.0]])
    gradient = gradient_sensitivity(model, points)
    jacobian = jacobian_sensitivity(model, points)
    hessian = hessian_sensitivity(model, points)
    expected_gradient = jnp.array([[2.5, 5.0], [3.0, -4.0]])
    assert isinstance(gradient, SensitivityResult)
    assert jnp.allclose(gradient.derivative, expected_gradient)
    assert jnp.allclose(jacobian.derivative, expected_gradient)
    assert jnp.allclose(
        hessian.derivative,
        jnp.broadcast_to(jnp.diag(jnp.array([0.5, 3.0])), (2, 2, 2)),
    )
    assert jnp.allclose(
        jax.jit(lambda value: jacobian_sensitivity(model, value).derivative)(points),
        expected_gradient,
    )
    assert jax.vmap(model)(points).shape == (2,)
    assert jax.grad(lambda x: jnp.sum(model(x)))(points).shape == points.shape


def test_complex_sensitivity_requires_explicit_holomorphic_semantics_and_blockwise_fails():
    points = jnp.array([[1.0 + 1.0j, 2.0 - 1.0j]])
    with pytest.raises(TypeError, match="holomorphic=True"):
        jacobian_sensitivity(_HolomorphicSquare(), points)
    result = jacobian_sensitivity(_HolomorphicSquare(), points, holomorphic=True)
    assert jnp.allclose(result.derivative, 2.0 * points)
    with pytest.raises(ValueError, match="pointwise"):
        jacobian_sensitivity(_BlockModel(), jnp.ones((2, 2)))
    with pytest.raises(TypeError, match="inexact-valued"):
        jacobian_sensitivity(
            _LinearModel(jnp.ones((2,))), jnp.ones((2, 2), dtype=jnp.int32)
        )


def test_leverage_and_cooks_diagnostics_preserve_cases_masks_and_complex_values():
    batch = _batch(case=True)
    model = _LinearModel(jnp.array([3.0, 0.2]), 1.0)
    diagnostics = leverage_and_cooks_distance(model, batch)
    assert isinstance(diagnostics, RegressionInfluenceDiagnostics)
    assert diagnostics.leverage.shape == (2, 5)
    assert diagnostics.cooks_distance.shape == (2, 5)
    assert diagnostics.mean_squared_error.shape == (2,)
    assert diagnostics.valid.shape == (2,)
    assert jnp.all(diagnostics.leverage >= 0.0)
    assert jnp.allclose(diagnostics.residual[..., :4], 0.0)

    complex_x = batch.dense_features().astype(jnp.complex64) * (1.0 + 0.2j)
    complex_model = _LinearModel(jnp.array([1.0 - 0.5j, 0.2 + 0.1j]))
    complex_batch = MLBatch(complex_x, complex_model(complex_x))
    complex_diagnostics = leverage_and_cooks_distance(complex_model, complex_batch)
    assert jnp.all(jnp.isfinite(complex_diagnostics.leverage))


def test_influence_functions_obey_gradient_contract_and_return_jax_arrays():
    model = _QuadraticModel(jnp.array([1.0, -0.5]), jnp.array([0.1, 0.2]))
    x = jnp.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, -1.0]])
    targets = model(x) + jnp.array([0.1, -0.1, 0.2, -0.2])
    batch = MLBatch(x, targets, sample_weight=jnp.array([1.0, 2.0, 1.0, 0.5]))
    result = _fit_result(model, GradientContract.direct())
    influence = influence_functions(result, batch, damping=1e-3)
    assert isinstance(influence, InfluenceFunctionResult)
    assert influence.parameter_influence.shape == (4, 4)
    assert influence.loss_influence.shape == (4, 4)
    assert influence.hessian.shape == (4, 4)
    assert influence.valid
    assert jnp.all(jnp.isfinite(influence.loss_influence))

    stopped = _fit_result(model, GradientContract())
    with pytest.raises(ValueError, match="does not permit"):
        influence_functions(stopped, batch)
    complex_result = _fit_result(
        _LinearModel(jnp.array([1.0 + 1.0j, 2.0 + 0.0j])),
        GradientContract.direct(),
    )
    with pytest.raises(TypeError, match="real parameterization"):
        influence_functions(complex_result, batch)
