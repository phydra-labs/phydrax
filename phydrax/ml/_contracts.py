#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, overload, Protocol, runtime_checkable, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._differentiation import (
    DerivativeAdmission,
    DerivativeContract,
    DerivativeRoute,
    DifferentiationRequest,
    RegularityPolicy,
    SurfaceDerivative,
)
from .._model import AbstractArrayModel, FrozenModel, ModelPorts, PortProvider
from .._strict import StrictModule
from ._batch import MLBatch
from ._schema import AbstractFittedModel, FeatureSchema, TargetSchema


_ModelT = TypeVar("_ModelT", bound=AbstractArrayModel)

ML_SUCCESS = 0
ML_INSUFFICIENT_DATA = 1
ML_RANK_DEFICIENT = 2
ML_NONFINITE = 3
ML_NONCONVERGED = 4
ML_INFEASIBLE = 5
ML_CAPACITY_EXHAUSTED = 6


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
    """Frozen executable model and audited diagnostics from one pure fit.

    `derivative_contract` is the canonical declaration of the derivatives the
    fitted map supports: `INPUT` and `MODEL_PARAMETER` for predictions, and the
    `FIT_*` surfaces for the fit itself, with the fit route as its route.
    """

    model: FrozenModel
    diagnostics: Any
    valid: Array
    status: Array
    derivative_contract: DerivativeContract
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
        derivative_contract: DerivativeContract,
    ):
        if not isinstance(derivative_contract, DerivativeContract):
            raise TypeError("derivative_contract must be a DerivativeContract.")
        self.model = model if isinstance(model, FrozenModel) else FrozenModel(model)
        self.diagnostics = diagnostics
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.derivative_contract = derivative_contract
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

    def bind_schemas(
        self,
        feature_schema: FeatureSchema,
        /,
        target_schema: TargetSchema | None = None,
    ) -> FitResult:
        """Return this result with the schemas bound into its fitted executable."""
        model = self.as_trainable()
        if not isinstance(model, AbstractFittedModel):
            return self
        bound = model.bind_schemas(feature_schema, target_schema)
        if bound is model:
            return self
        return FitResult(
            bound,
            self.diagnostics,
            valid=self.valid,
            status=self.status,
            method=self.method,
            derivative_contract=self.derivative_contract,
        )

    def model_ports(self) -> ModelPorts:
        """Return the derived ports of the fitted executable."""
        model = self.as_trainable()
        if not isinstance(model, PortProvider):
            raise TypeError(f"{type(model).__name__} does not declare model ports.")
        return model.model_ports()

    def derivative_admission(
        self,
        request: DifferentiationRequest,
        /,
        *,
        policy: RegularityPolicy | None = None,
    ) -> DerivativeAdmission:
        """Admit `request` against the fitted map's derivative contract."""
        return self.derivative_contract.admit(request, policy=policy)

    def require_derivative(
        self,
        request: DifferentiationRequest,
        /,
        *,
        policy: RegularityPolicy | None = None,
    ) -> DerivativeAdmission:
        """Return the admission of `request`, raising `ValueError` if unsupported."""
        return self.derivative_contract.require(request, policy=policy)


def prediction_fit_contract(
    prediction: DerivativeContract,
    fit_surfaces: Iterable[SurfaceDerivative] = (),
    /,
    *,
    route: DerivativeRoute,
    conditions: Iterable[str] = (),
    nondifferentiable_outputs: Iterable[str] = (),
) -> DerivativeContract:
    """Fit contract extending a fitted model's prediction contract.

    The prediction surfaces, regularity, conditions, and nondifferentiable
    outputs of `prediction` (the model's `_prediction_contract()`) are kept, so a
    fit and its executable never disagree; `fit_surfaces`, the fit `route`, and
    fit `conditions` and `nondifferentiable_outputs` are added.
    """
    return DerivativeContract(
        (*prediction.surfaces, *fit_surfaces),
        route=route,
        regularity=prediction.regularity,
        conditions=(*prediction.conditions, *conditions),
        nondifferentiable_outputs=(
            *prediction.nondifferentiable_outputs,
            *nondifferentiable_outputs,
        ),
    )


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
    "AbstractRecipe",
    "DecisionFunctionModel",
    "FitDiagnostics",
    "FitResult",
    "LogProbabilityModel",
    "PredictionModel",
    "ProbabilityModel",
]
