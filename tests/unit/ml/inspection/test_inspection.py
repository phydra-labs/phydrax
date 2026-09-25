#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
import pytest

from phydrax import (
    DerivativeContract,
    DerivativeRoute,
    DerivativeSurface,
    fixed_field,
    GradientLevel,
    SurfaceDerivative,
)
from phydrax._model import AbstractArrayModel, ModelBinding
from phydrax.linalg import LinearSolveStatus
from phydrax.ml import FitDiagnostics, FitResult, ML_SUCCESS, MLBatch
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
        self.in_size = self.coefficients.shape[0]
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
        self.in_size = self.coefficients.shape[0]
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
        derivative_contract=contract,
    )


_DIRECT_CONTRACT = DerivativeContract(
    (
        SurfaceDerivative(DerivativeSurface.INPUT, GradientLevel.SMOOTH),
        SurfaceDerivative(DerivativeSurface.MODEL_PARAMETER, GradientLevel.SMOOTH),
        SurfaceDerivative(DerivativeSurface.FIT_FEATURES, GradientLevel.CONDITIONAL),
        SurfaceDerivative(DerivativeSurface.FIT_TARGETS, GradientLevel.CONDITIONAL),
        SurfaceDerivative(DerivativeSurface.FIT_WEIGHTS, GradientLevel.CONDITIONAL),
        SurfaceDerivative(
            DerivativeSurface.FIT_HYPERPARAMETERS, GradientLevel.CONDITIONAL
        ),
    ),
    route=DerivativeRoute.DIRECT,
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
        feature_mask=jnp.ones_like(x, dtype="bool").at[..., 4, 1].set(False),
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


def test_influence_functions_obey_derivative_contract_and_return_jax_arrays():
    model = _QuadraticModel(jnp.array([1.0, -0.5]), jnp.array([0.1, 0.2]))
    x = jnp.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, -1.0]])
    targets = model(x) + jnp.array([0.1, -0.1, 0.2, -0.2])
    batch = MLBatch(x, targets, sample_weight=jnp.array([1.0, 2.0, 1.0, 0.5]))
    result = _fit_result(model, _DIRECT_CONTRACT)
    influence = influence_functions(result, batch, damping=1e-3)
    assert isinstance(influence, InfluenceFunctionResult)
    assert influence.parameter_influence.shape == (4, 4)
    assert influence.loss_influence.shape == (4, 4)
    assert influence.hessian.shape == (4, 4)
    assert influence.valid
    assert jnp.all(jnp.isfinite(influence.loss_influence))

    stopped = _fit_result(
        model,
        DerivativeContract(
            (SurfaceDerivative(DerivativeSurface.MODEL_PARAMETER, GradientLevel.SMOOTH),),
            route=DerivativeRoute.STOPPED,
        ),
    )
    with pytest.raises(ValueError, match="does not permit"):
        influence_functions(stopped, batch)
    complex_result = _fit_result(
        _LinearModel(jnp.array([1.0 + 1.0j, 2.0 + 0.0j])),
        _DIRECT_CONTRACT,
    )
    with pytest.raises(TypeError, match="real parameterization"):
        influence_functions(complex_result, batch)


class _CenteredLinearModel(AbstractArrayModel):
    coefficients: jax.Array
    feature_mean: jax.Array = fixed_field()
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, coefficients, feature_mean):
        self.coefficients = jnp.asarray(coefficients)
        self.feature_mean = jnp.asarray(feature_mean)
        self.in_size = self.coefficients.shape[0]
        self.out_size = "scalar"

    def __call__(self, x, /, *, key=None):
        del key
        return oe.contract(
            "...f,f->...", jnp.asarray(x) - self.feature_mean, self.coefficients
        )


def test_influence_functions_perturb_only_parameters_and_carry_solve_evidence():
    x = jnp.array([[-1.0, 0.5], [0.0, 1.0], [1.0, 2.0], [2.0, -1.0], [0.5, 0.0]])
    targets = jnp.array([0.3, -0.2, 1.1, 0.4, -0.5])
    weights = jnp.array([1.0, 2.0, 1.0, 0.5, 1.5])
    model = _CenteredLinearModel(jnp.array([0.7, -0.3]), jnp.array([0.4, 0.5]))
    batch = MLBatch(x, targets, sample_weight=weights)
    damping = 1e-3
    influence = influence_functions(
        _fit_result(model, _DIRECT_CONTRACT), batch, damping=damping
    )

    # The fixed feature mean is a statistic of the fit, not a parameter.
    assert influence.parameter_paths == (".coefficients",)
    assert influence.parameter_influence.shape == (5, 2)
    assert influence.hessian.shape == (2, 2)
    centered = x - model.feature_mean
    normalizer = jnp.sum(weights)
    residual = centered @ model.coefficients - targets
    gradients = 2.0 * (weights * residual)[:, None] * centered
    hessian = 2.0 * (centered.T * weights) @ centered / normalizer
    expected = -jnp.linalg.solve(hessian + damping * jnp.eye(2), gradients.T).T
    assert jnp.allclose(influence.hessian, hessian, rtol=1e-12)
    assert jnp.allclose(influence.parameter_influence, expected / normalizer, rtol=1e-10)
    assert influence.valid
    assert influence.solve_status.tolist() == [int(LinearSolveStatus.SUCCESS)] * 5
    assert int(influence.hessian_rank) == 2
    singular = jnp.linalg.svd(hessian + damping * jnp.eye(2), compute_uv=False)
    assert float(influence.condition_estimate) == pytest.approx(
        float(singular[0] / singular[-1]), rel=1e-10
    )

    # A redundant parameterization without damping has a singular Hessian: the
    # solve reports the rank deficiency instead of returning plausible influence.
    redundant = _CenteredLinearModel(jnp.array([0.7, -0.3]), jnp.array([0.4, 0.4]))
    collinear = MLBatch(
        jnp.stack((x[:, 0], x[:, 0]), axis=-1), targets, sample_weight=weights
    )
    deficient = influence_functions(
        _fit_result(redundant, _DIRECT_CONTRACT), collinear, damping=0.0
    )
    assert not deficient.valid
    assert int(deficient.hessian_rank) == 1
    assert set(deficient.solve_status.tolist()) == {int(LinearSolveStatus.RANK_DEFICIENT)}
