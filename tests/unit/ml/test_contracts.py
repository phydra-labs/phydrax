#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel, FrozenModel, ModelBinding
from phydrax.ml._contracts import OperationDerivativeContract


_INPUT = phx.DerivativeSurface.INPUT
_PARAMETER = phx.DerivativeSurface.MODEL_PARAMETER
_FIT_TARGETS = phx.DerivativeSurface.FIT_TARGETS
_FIT_HYPERPARAMETERS = phx.DerivativeSurface.FIT_HYPERPARAMETERS


class _ScaleModel(AbstractArrayModel):
    scale: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, scale: Any) -> None:
        self.scale = jnp.asarray(scale)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, x: Any, /, *, key: Any = None) -> Any:
        del key
        return self.scale * x


class _BlockClassifier(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    _input_binding = ModelBinding.blockwise()

    def __init__(self) -> None:
        self.in_size = 1
        self.out_size = 2

    def __call__(self, x: Any, /, *, key: Any = None) -> Any:
        del key
        return self.decision_function(x)

    def decision_function(self, x: Any) -> Any:
        value = jnp.asarray(x)
        return jnp.concatenate((-value, value), axis=-1)

    def predict_proba(self, x: Any) -> Any:
        return jax.nn.softmax(self.decision_function(x), axis=-1)

    def predict(self, x: Any) -> Any:
        return jnp.argmax(self.decision_function(x), axis=-1)


class _ScaleRecipe(phx.ml.AbstractRecipe):
    def fit_batch(self, batch: Any, /, *, key: Any = None) -> Any:
        del key
        targets = batch.require_targets()
        scale = jnp.sum(batch.effective_weight() * targets) / jnp.sum(
            batch.effective_weight()
        )
        diagnostics = phx.ml.FitDiagnostics(
            valid=True,
            status=phx.ml.ML_SUCCESS,
            effective_samples=jnp.sum(batch.effective_weight() > 0),
            method="weighted-mean-scale",
        )
        return phx.ml.FitResult(
            _ScaleModel(scale),
            diagnostics,
            valid=True,
            status=phx.ml.ML_SUCCESS,
            method="weighted-mean-scale",
            derivative_contract=phx.DerivativeContract(
                (
                    phx.SurfaceDerivative(_INPUT, phx.GradientLevel.SMOOTH),
                    phx.SurfaceDerivative(_PARAMETER, phx.GradientLevel.SMOOTH),
                    phx.SurfaceDerivative(_FIT_TARGETS, phx.GradientLevel.CONDITIONAL),
                ),
                route=phx.DerivativeRoute.DIRECT,
            ),
        )


def test_contracts_scenario_1() -> None:
    features = jnp.arange(24.0).reshape(2, 4, 3)
    targets = jnp.arange(16.0).reshape(2, 4, 2)
    batch = phx.ml.MLBatch(
        features,
        targets,
        sample_mask=jnp.array([True, True, False, True]),
        sample_weight=jnp.array([1.0, 2.0, 7.0, 3.0]),
        measure_weight=jnp.array([0.5, 0.5, 0.5, 0.5]),
    )

    assert batch.case_shape == (2,)
    assert batch.target_shape == (2,)
    assert batch.feature_schema.names == ("feature_0", "feature_1", "feature_2")
    assert jnp.allclose(
        batch.effective_weight("product"),
        jnp.array([[0.5, 1.0, 0.0, 1.5], [0.5, 1.0, 0.0, 1.5]]),
    )
    selected = batch.take_samples(jnp.array([3, 0]))
    assert selected.features.shape == (2, 2, 3)
    # ty: ignore[invalid-argument-type]
    assert jnp.allclose(selected.targets, targets[:, jnp.array([3, 0])])
    sparse = phx.ml.SparseFeatures(
        jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        jnp.array([[0, 2], [1, 1]]),
        feature_count=3,
    )
    dense = sparse.to_dense()

    assert jnp.allclose(dense, jnp.array([[1.0, 0.0, 2.0], [0.0, 7.0, 0.0]]))
    assert jnp.allclose(
        sparse.right_matmul(jnp.arange(6.0).reshape(3, 2)),
        dense @ jnp.arange(6.0).reshape(3, 2),
    )
    with pytest.raises(ValueError, match="feature_mask is unsupported"):
        phx.ml.MLBatch(sparse, feature_mask=jnp.ones((2, 3), dtype="bool"))
    recipe = _ScaleRecipe()
    features = jnp.ones((3, 1))
    targets = jnp.array([1.0, 2.0, 5.0])
    result = phx.ml.fit(
        recipe,
        features,
        targets,
        sample_weight=jnp.array([1.0, 1.0, 0.0]),
    )

    assert isinstance(result.model, FrozenModel)
    assert jnp.allclose(result.model(jnp.array([3.0])), jnp.array([4.5]))
    assert result.as_trainable() is result.model.model
    gradient = jax.grad(lambda value: jnp.sum(result.model(value)))(jnp.array([2.0]))
    assert jnp.allclose(gradient, jnp.array([1.5]))
    result = phx.ml.fit(_ScaleRecipe(), jnp.ones((3, 1)), jnp.array([1.0, 2.0, 5.0]))

    mixed = phx.DifferentiationRequest((_FIT_TARGETS, _PARAMETER, _INPUT))
    admission = result.require_derivative(mixed)
    assert admission.status == phx.DERIVATIVE_SUPPORTED
    assert admission.route is phx.DerivativeRoute.DIRECT
    assert admission.level(_INPUT) is phx.GradientLevel.SMOOTH
    assert admission.level(_FIT_TARGETS) is phx.GradientLevel.CONDITIONAL
    assert "regularity-undeclared" in admission.conditions

    surrogate = phx.DifferentiationRequest(
        (_INPUT,), authority=phx.ComponentAuthority.SURROGATE
    )
    assert not result.derivative_admission(surrogate).supported
    permitted = result.derivative_admission(
        surrogate, policy=phx.RegularityPolicy(allow_undeclared=True)
    )
    assert permitted.level(_INPUT) is phx.GradientLevel.SMOOTH

    undeclared = phx.DifferentiationRequest((_PARAMETER, _FIT_HYPERPARAMETERS))
    rejected = result.derivative_admission(undeclared)
    assert rejected.status == phx.DERIVATIVE_UNSUPPORTED
    assert rejected.level(_PARAMETER) is phx.GradientLevel.SMOOTH
    assert rejected.level(_FIT_HYPERPARAMETERS) is phx.GradientLevel.NONE
    assert rejected.reasons == ("surface-unsupported:fit-hyperparameters",)
    with pytest.raises(ValueError, match="derivative-unsupported"):
        result.require_derivative(undeclared)
    with pytest.raises(ValueError, match="derivative-unsupported"):
        phx.ml.fit(
            _ScaleRecipe(),
            jnp.ones((3, 1)),
            jnp.array([1.0, 2.0, 5.0]),
            derivative_request=undeclared,
        )


def test_contracts_scenario_2() -> None:
    frozen = FrozenModel(_BlockClassifier())
    values = jnp.asarray(((1.0,), (-2.0,)))

    assert frozen.input_binding() == ModelBinding.blockwise()
    assert jnp.array_equal(frozen.predict(values), jnp.asarray((1, 0)))
    assert jnp.allclose(
        frozen.predict_proba(values),
        jax.nn.softmax(frozen.decision_function(values), axis=-1),
    )

    with pytest.raises(AttributeError):
        FrozenModel(_ScaleModel(1.0)).predict(values)
    batch = phx.ml.MLBatch(jnp.ones((3, 1)), jnp.ones((3,)))
    with pytest.raises(ValueError, match="cannot accompany"):
        phx.ml.fit(_ScaleRecipe(), batch, sample_weight=jnp.ones((3,)))


def _gated_operations(
    valid: jax.Array,
) -> tuple[OperationDerivativeContract, ...]:
    independent = (
        phx.SurfaceDerivative(_INPUT, phx.GradientLevel.SMOOTH),
        phx.SurfaceDerivative(_PARAMETER, phx.GradientLevel.SMOOTH),
    )
    encoder = phx.DerivativeContract(
        independent,
        route=phx.DerivativeRoute.DIRECT,
        regularity=phx.DerivativeRegularity.smooth(),
    )
    projection = phx.DerivativeContract(
        (
            *independent,
            phx.SurfaceDerivative(_FIT_TARGETS, phx.GradientLevel.CONDITIONAL),
        ),
        route=phx.DerivativeRoute.DIRECT,
        regularity=phx.DerivativeRegularity.smooth(),
        conditions=("nonzero-normalizer",),
    )
    return (
        OperationDerivativeContract("transform", encoder),
        OperationDerivativeContract(
            "project",
            projection,
            runtime_surfaces=(_FIT_TARGETS,),
            runtime_valid=valid[..., None],
            runtime_status=jnp.where(valid, 0, phx.ml.ML_RANK_DEFICIENT)[..., None],
        ),
    )


class _NormalizedScaleRecipe(phx.ml.AbstractRecipe):
    """An actual singular normalization supplies a dynamic conditional gate."""

    def fit_batch(self, batch: phx.ml.MLBatch, /, *, key: Any = None) -> phx.ml.FitResult:
        del key
        total = jnp.sum(batch.require_targets())
        valid = jnp.isfinite(total) & (total != 0)
        operations = _gated_operations(valid)
        return phx.ml.FitResult(
            _ScaleModel(1 / total),
            phx.ml.FitDiagnostics(
                valid=valid,
                status=jnp.where(valid, 0, phx.ml.ML_RANK_DEFICIENT),
                method="nonzero-normalization",
            ),
            valid=valid,
            status=jnp.where(valid, 0, phx.ml.ML_RANK_DEFICIENT),
            method="nonzero-normalization",
            derivative_contract=operations[0].contract,
            operation_derivatives=operations,
            default_operation="transform",
        )


def test_operation_gates_toggle_without_revoking_independent_surfaces() -> None:
    request = phx.DifferentiationRequest((_FIT_TARGETS,))
    independent = phx.DifferentiationRequest((_INPUT, _PARAMETER))

    @jax.jit
    def evidence(total: jax.Array) -> tuple[jax.Array, jax.Array]:
        result = phx.ml.fit(_NormalizedScaleRecipe(), jnp.ones((1, 1)), total[None])
        admission = result.derivative_admission(request, operation="project")
        assert admission.supported
        assert admission.runtime_valid is not None
        assert admission.runtime_status is not None
        assert result.require_derivative(independent, operation="project").supported
        assert result.derivative_admission(independent).runtime_valid is None
        return admission.runtime_valid, admission.runtime_status

    assert jnp.array_equal(evidence(jnp.array(2.0))[0], jnp.array([True]))
    valid, status = evidence(jnp.array(0.0))
    assert jnp.array_equal(valid, jnp.array([False]))
    assert jnp.array_equal(status, jnp.array([phx.ml.ML_RANK_DEFICIENT]))

    @jax.jit
    def require_gate(total: jax.Array) -> jax.Array:
        result = phx.ml.fit(_NormalizedScaleRecipe(), jnp.ones((1, 1)), total[None])
        admission = result.require_derivative(request, operation="project")
        assert admission.runtime_valid is not None
        return admission.runtime_valid

    assert jnp.array_equal(require_gate(jnp.array(2.0)), jnp.array([True]))
    with pytest.raises(Exception, match="derivative-unsupported"):
        require_gate(jnp.array(0.0)).block_until_ready()
    result = phx.ml.fit(_NormalizedScaleRecipe(), jnp.ones((1, 1)), jnp.array([2.0]))
    assert not result.derivative_admission(request).supported
    with pytest.raises(ValueError, match="derivative-unsupported"):
        result.require_derivative(request)
    with pytest.raises(ValueError, match="Unknown derivative operation"):
        result.derivative_admission(independent, operation="unknown")


def test_fit_checked_gate_is_retained_when_only_model_is_consumed() -> None:
    request = phx.DifferentiationRequest((_FIT_TARGETS,))

    @jax.jit
    def model_only(total: jax.Array) -> jax.Array:
        return phx.ml.fit(
            _NormalizedScaleRecipe(),
            jnp.ones((1, 1)),
            total[None],
            derivative_request=request,
            derivative_operation="project",
        ).model(jnp.array([3.0]))

    assert jnp.allclose(model_only(jnp.array(2.0)), jnp.array([1.5]))
    with pytest.raises(Exception, match="derivative-unsupported"):
        model_only(jnp.array(0.0)).block_until_ready()
    assert jnp.allclose(
        jax.grad(lambda total: jnp.sum(model_only(total)))(jnp.array(2.0)), -0.75
    )


def test_schema_binding_preserves_operation_gates_and_default_contract() -> None:
    original = phx.ml.fit(
        phx.ml.linear.RidgeRecipe(0.1),
        jnp.array([[1.0], [2.0], [3.0]]),
        jnp.array([2.0, 4.0, 6.0]),
    )
    operations = _gated_operations(jnp.array(False))
    result = phx.ml.FitResult(
        original.as_trainable(),
        original.diagnostics,
        valid=original.valid,
        status=original.status,
        method=original.method,
        derivative_contract=operations[0].contract,
        operation_derivatives=operations,
        default_operation="transform",
    )
    rebound = result.bind_schemas(phx.ml.FeatureSchema(("renamed",)))
    request = phx.DifferentiationRequest((_FIT_TARGETS,))
    assert not rebound.derivative_admission(request).supported
    admission = rebound.derivative_admission(request, operation="project")
    assert admission.runtime_valid is not None
    assert jnp.array_equal(admission.runtime_valid, jnp.array([False]))
    with pytest.raises(eqx.EquinoxRuntimeError, match="derivative-unsupported"):
        rebound.require_derivative(request, operation="project")
    unrelated = phx.ml.fit(
        _ScaleRecipe(), jnp.ones((2, 1)), jnp.array([1.0, 3.0])
    ).require_derivative(request)
    assert unrelated.level(_FIT_TARGETS) is phx.GradientLevel.CONDITIONAL
    assert unrelated.runtime_valid is None
    assert unrelated.runtime_status is None


def test_operation_metadata_rejects_misaligned_cases_and_independent_fit_gates() -> None:
    contract = phx.DerivativeContract.smooth((_INPUT,))
    with pytest.raises(ValueError, match="only FIT"):
        OperationDerivativeContract(
            "project",
            contract,
            runtime_surfaces=(_INPUT,),
            runtime_valid=jnp.array([False]),
            runtime_status=jnp.array([1]),
        )
    operations = _gated_operations(jnp.array([True, False]))
    with pytest.raises(ValueError, match="case shape"):
        phx.ml.FitResult(
            _ScaleModel(1.0),
            None,
            valid=True,
            status=0,
            method="invalid-cases",
            derivative_contract=contract,
            operation_derivatives=operations,
            default_operation="transform",
        )
