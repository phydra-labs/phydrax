#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal, overload, Protocol, runtime_checkable, TypeAlias, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._model import AbstractArrayModel, FrozenModel
from .._strict import StrictModule
from ._batch import MLBatch


GradientLevel: TypeAlias = Literal["smooth", "almost-everywhere", "conditional", "none"]
FitGradientMode: TypeAlias = Literal[
    "direct", "implicit", "unrolled", "spectral", "relaxed", "stopped"
]
GradientSurface: TypeAlias = Literal["prediction", "fit"]
GradientInput: TypeAlias = Literal[
    "inputs",
    "parameters",
    "features",
    "targets",
    "weights",
    "hyperparameters",
]
_ModelT = TypeVar("_ModelT", bound=AbstractArrayModel)

ML_SUCCESS = 0
ML_INSUFFICIENT_DATA = 1
ML_RANK_DEFICIENT = 2
ML_NONFINITE = 3
ML_NONCONVERGED = 4
ML_INFEASIBLE = 5
ML_CAPACITY_EXHAUSTED = 6
ML_UNSUPPORTED_GRADIENT = 7


@runtime_checkable
class DecisionFunctionModel(Protocol):
    """Structural contract for classifiers exposing unconstrained scores."""

    def decision_function(self, x: Any, /) -> Any: ...


@runtime_checkable
class LogProbabilityModel(Protocol):
    """Structural contract for classifiers exposing normalized log probabilities."""

    def predict_log_proba(self, x: Any, /) -> Any: ...


@runtime_checkable
class PredictionModel(Protocol):
    """Structural contract for models exposing consumer-facing predictions."""

    def predict(self, x: Any, /) -> Any: ...


@runtime_checkable
class ProbabilityModel(Protocol):
    """Structural contract for classifiers exposing normalized probabilities."""

    def predict_proba(self, x: Any, /) -> Any: ...


def _protocol_model(model: AbstractArrayModel, /) -> AbstractArrayModel:
    return model.as_trainable() if isinstance(model, FrozenModel) else model


class MLGradientRequest(StrictModule):
    """Explicit admission request for one prediction or fitting gradient surface."""

    surface: GradientSurface = eqx.field(static=True)
    inputs: tuple[GradientInput, ...] = eqx.field(static=True)

    def __init__(
        self,
        surface: GradientSurface,
        inputs: tuple[GradientInput, ...],
        /,
    ):
        if surface not in ("prediction", "fit"):
            raise ValueError("Gradient surface must be 'prediction' or 'fit'.")
        values = tuple(inputs)
        allowed = (
            {"inputs", "parameters"}
            if surface == "prediction"
            else {"features", "targets", "weights", "hyperparameters"}
        )
        if not values or any(value not in allowed for value in values):
            raise ValueError(
                f"Gradient inputs {values!r} are invalid for surface {surface!r}."
            )
        if len(set(values)) != len(values):
            raise ValueError("Gradient request inputs must be unique.")
        self.surface = surface
        self.inputs = values


class MLGradientAdmission(StrictModule):
    """Audited answer for one explicit ML gradient request."""

    request: MLGradientRequest
    levels: tuple[GradientLevel, ...] = eqx.field(static=True)
    fit_mode: FitGradientMode = eqx.field(static=True)
    supported: bool = eqx.field(static=True)
    status: int = eqx.field(static=True)
    conditions: tuple[str, ...] = eqx.field(static=True)
    nondifferentiable_outputs: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        request: MLGradientRequest,
        levels: tuple[GradientLevel, ...],
        /,
        *,
        fit_mode: FitGradientMode,
        conditions: tuple[str, ...],
        nondifferentiable_outputs: tuple[str, ...],
    ):
        supported = all(level != "none" for level in levels)
        self.request = request
        self.levels = tuple(levels)
        self.fit_mode = fit_mode
        self.supported = supported
        self.status = ML_SUCCESS if supported else ML_UNSUPPORTED_GRADIENT
        self.conditions = tuple(conditions)
        self.nondifferentiable_outputs = tuple(nondifferentiable_outputs)


class GradientContract(StrictModule):
    """Static declaration of the gradients an ML result mathematically supports."""

    prediction_inputs: GradientLevel = eqx.field(static=True)
    prediction_parameters: GradientLevel = eqx.field(static=True)
    fit_features: GradientLevel = eqx.field(static=True)
    fit_targets: GradientLevel = eqx.field(static=True)
    fit_weights: GradientLevel = eqx.field(static=True)
    fit_hyperparameters: GradientLevel = eqx.field(static=True)
    fit_mode: FitGradientMode = eqx.field(static=True)
    nondifferentiable_outputs: tuple[str, ...] = eqx.field(static=True)
    conditions: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        prediction_inputs: GradientLevel = "smooth",
        prediction_parameters: GradientLevel = "smooth",
        fit_features: GradientLevel = "none",
        fit_targets: GradientLevel = "none",
        fit_weights: GradientLevel = "none",
        fit_hyperparameters: GradientLevel = "none",
        fit_mode: FitGradientMode = "stopped",
        nondifferentiable_outputs: tuple[str, ...] = (),
        conditions: tuple[str, ...] = (),
    ):
        levels = {"smooth", "almost-everywhere", "conditional", "none"}
        mode_values = {"direct", "implicit", "unrolled", "spectral", "relaxed", "stopped"}
        declared = (
            prediction_inputs,
            prediction_parameters,
            fit_features,
            fit_targets,
            fit_weights,
            fit_hyperparameters,
        )
        if any(level not in levels for level in declared):
            raise ValueError("GradientContract contains an unsupported gradient level.")
        if fit_mode not in mode_values:
            raise ValueError("GradientContract contains an unsupported fit mode.")
        self.prediction_inputs = prediction_inputs
        self.prediction_parameters = prediction_parameters
        self.fit_features = fit_features
        self.fit_targets = fit_targets
        self.fit_weights = fit_weights
        self.fit_hyperparameters = fit_hyperparameters
        self.fit_mode = fit_mode
        self.nondifferentiable_outputs = tuple(nondifferentiable_outputs)
        self.conditions = tuple(conditions)

    @classmethod
    def direct(cls, /, *, conditions: tuple[str, ...] = ()) -> "GradientContract":
        return cls(
            fit_features="conditional",
            fit_targets="conditional",
            fit_weights="conditional",
            fit_hyperparameters="conditional",
            fit_mode="direct",
            conditions=conditions,
        )

    def admit(self, request: MLGradientRequest, /) -> MLGradientAdmission:
        if not isinstance(request, MLGradientRequest):
            raise TypeError("request must be an MLGradientRequest.")
        if request.surface == "prediction":
            declared = {
                "inputs": self.prediction_inputs,
                "parameters": self.prediction_parameters,
            }
        else:
            declared = {
                "features": self.fit_features,
                "targets": self.fit_targets,
                "weights": self.fit_weights,
                "hyperparameters": self.fit_hyperparameters,
            }
        levels = tuple(declared[value] for value in request.inputs)
        return MLGradientAdmission(
            request,
            levels,
            fit_mode=self.fit_mode,
            conditions=self.conditions,
            nondifferentiable_outputs=self.nondifferentiable_outputs,
        )

    def require(self, request: MLGradientRequest, /) -> MLGradientAdmission:
        admission = self.admit(request)
        if not admission.supported:
            unsupported = tuple(
                value
                for value, level in zip(
                    admission.request.inputs,
                    admission.levels,
                    strict=True,
                )
                if level == "none"
            )
            raise ValueError(
                "Gradient request is unsupported for inputs "
                f"{unsupported!r}; inspect GradientContract before transforming."
            )
        return admission


class FitDiagnostics(StrictModule):
    """Common numerical diagnostics shared by ML fit families."""

    valid: Array
    status: Array
    objective: Array
    iterations: Array
    effective_samples: Array
    rank: Array
    condition: Array
    method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        valid: Any,
        status: Any,
        objective: Any = jnp.nan,
        iterations: Any = 0,
        effective_samples: Any = 0.0,
        rank: Any = -1,
        condition: Any = jnp.nan,
        method: str,
    ):
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.objective = jnp.asarray(objective)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.effective_samples = jnp.asarray(effective_samples)
        self.rank = jnp.asarray(rank, dtype=jnp.int32)
        self.condition = jnp.asarray(condition)
        self.method = str(method)


class FitResult(StrictModule):
    """Frozen executable model and audited diagnostics from one pure fit."""

    model: FrozenModel
    diagnostics: Any
    valid: Array
    status: Array
    gradient_contract: GradientContract
    method: str = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractArrayModel,
        diagnostics: Any,
        /,
        *,
        valid: Any,
        status: Any,
        method: str,
        gradient_contract: GradientContract,
    ):
        self.model = model if isinstance(model, FrozenModel) else FrozenModel(model)
        self.diagnostics = diagnostics
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.gradient_contract = gradient_contract
        self.method = str(method)

    @overload
    def as_trainable(self, /) -> AbstractArrayModel: ...

    @overload
    def as_trainable(self, expected_type: type[_ModelT], /) -> _ModelT: ...

    def as_trainable(
        self,
        expected_type: type[_ModelT] | None = None,
        /,
    ) -> AbstractArrayModel | _ModelT:
        """Return the fitted executable, optionally enforcing its concrete type."""
        model = self.model.as_trainable()
        if expected_type is not None and not isinstance(model, expected_type):
            raise TypeError(
                f"Expected fitted model {expected_type.__name__}; got {type(model).__name__}."
            )
        return model

    def gradient_admission(
        self,
        request: MLGradientRequest,
        /,
    ) -> MLGradientAdmission:
        return self.gradient_contract.admit(request)

    def require_gradient(
        self,
        request: MLGradientRequest,
        /,
    ) -> MLGradientAdmission:
        return self.gradient_contract.require(request)


class AbstractRecipe(StrictModule):
    """Immutable configuration for a pure ML fitting operation."""

    @property
    def is_fresh_fit(self) -> bool:
        """Whether fitting starts without learned state from an earlier dataset."""
        return True

    @abstractmethod
    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        raise NotImplementedError


__all__ = [
    "ML_CAPACITY_EXHAUSTED",
    "ML_INFEASIBLE",
    "ML_INSUFFICIENT_DATA",
    "ML_NONCONVERGED",
    "ML_NONFINITE",
    "ML_RANK_DEFICIENT",
    "ML_SUCCESS",
    "ML_UNSUPPORTED_GRADIENT",
    "AbstractRecipe",
    "DecisionFunctionModel",
    "FitDiagnostics",
    "FitGradientMode",
    "FitResult",
    "GradientContract",
    "GradientInput",
    "GradientLevel",
    "GradientSurface",
    "LogProbabilityModel",
    "PredictionModel",
    "ProbabilityModel",
    "MLGradientAdmission",
    "MLGradientRequest",
]
