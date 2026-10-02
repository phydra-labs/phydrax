#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, final, overload, Protocol, runtime_checkable, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .._differentiation import (
    _runtime_derivative_evidence,
    DerivativeAdmission,
    DerivativeContract,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    RegularityPolicy,
    SurfaceDerivative,
)
from .._model import AbstractArrayModel, FrozenModel, ModelPorts, PortProvider
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
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
    ) -> None:
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.objective = jnp.asarray(objective)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.effective_samples = jnp.asarray(effective_samples)
        self.rank = jnp.asarray(rank, dtype=jnp.int32)
        self.condition = jnp.asarray(condition)
        self.method = str(method)


_FIT_SURFACES = frozenset(
    {
        DerivativeSurface.FIT_FEATURES,
        DerivativeSurface.FIT_TARGETS,
        DerivativeSurface.FIT_WEIGHTS,
        DerivativeSurface.FIT_HYPERPARAMETERS,
    }
)


@final
class OperationDerivativeContract(StrictModule, NonTrainableState):
    """Immutable operation declaration with optional per-fit-surface evidence.

    Gate arrays have shape ``(*case_shape, len(runtime_surfaces))``. They resolve
    only the named fit surfaces, never independent prediction derivatives.
    Integer status codes retain the numerical owner's cause vocabulary.
    """

    operation: str = eqx.field(static=True)
    contract: DerivativeContract
    runtime_surfaces: tuple[DerivativeSurface, ...] = eqx.field(static=True)
    runtime_valid: Array | None
    runtime_status: Array | None

    @checked
    def __init__(
        self,
        operation: str,
        contract: DerivativeContract,
        /,
        *,
        runtime_surfaces: Iterable[DerivativeSurface] = (),
        runtime_valid: Array | None = None,
        runtime_status: Array | None = None,
    ) -> None:
        if not isinstance(operation, str):
            raise TypeError("operation must be an identifier string.")
        if not operation.isidentifier():
            raise ValueError("operation must be a nonempty identifier.")
        surfaces = tuple(runtime_surfaces)
        if any(not isinstance(surface, DerivativeSurface) for surface in surfaces):
            raise TypeError("runtime_surfaces must contain DerivativeSurface members.")
        if len(set(surfaces)) != len(surfaces):
            raise ValueError("Runtime derivative surfaces must be unique.")
        if not set(surfaces).issubset(_FIT_SURFACES):
            raise ValueError("Fit runtime gates may resolve only FIT_* surfaces.")
        if runtime_valid is not None and not surfaces:
            raise ValueError("Resolved runtime gates must name their fit surfaces.")
        valid, status = _runtime_derivative_evidence(
            runtime_valid, runtime_status, len(surfaces)
        )
        self.operation = operation
        self.contract = contract
        self.runtime_surfaces = surfaces
        self.runtime_valid = valid
        self.runtime_status = status

    def _attach_evidence(self, admission: DerivativeAdmission, /) -> DerivativeAdmission:
        if self.runtime_valid is None:
            return admission
        requested = admission.request.surfaces
        if not any(surface in self.runtime_surfaces for surface in requested):
            return admission
        if self.runtime_status is None:
            raise RuntimeError(
                "Resolved derivative gates are missing their status evidence."
            )
        valid_columns = []
        status_columns = []
        for surface in requested:
            if surface in self.runtime_surfaces:
                index = self.runtime_surfaces.index(surface)
                valid_columns.append(self.runtime_valid[..., index])
                status_columns.append(self.runtime_status[..., index])
            else:
                # Neutral columns do not prove unrelated declaration conditions.
                valid_columns.append(
                    jnp.ones(self.runtime_valid.shape[:-1], dtype=jnp.bool_)
                )
                status_columns.append(
                    jnp.zeros(
                        self.runtime_status.shape[:-1], dtype=self.runtime_status.dtype
                    )
                )
        return admission.with_runtime_evidence(
            jnp.stack(valid_columns, axis=-1), jnp.stack(status_columns, axis=-1)
        )

    def admit(
        self,
        request: DifferentiationRequest,
        /,
        *,
        policy: RegularityPolicy | None = None,
    ) -> DerivativeAdmission:
        """Return the canonical declaration-plus-requested-gate admission."""
        return self._attach_evidence(self.contract.admit(request, policy=policy))

    def require(
        self,
        request: DifferentiationRequest,
        /,
        *,
        policy: RegularityPolicy | None = None,
    ) -> DerivativeAdmission:
        """Preserve static rejection before guarding resolved numerical evidence."""
        admission = self.contract.require(request, policy=policy)
        return self._attach_evidence(admission).require_runtime()


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
    operation_derivatives: tuple[OperationDerivativeContract, ...]
    default_operation: str | None = eqx.field(static=True)

    @checked
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
        operation_derivatives: tuple[OperationDerivativeContract, ...] = (),
        default_operation: str | None = None,
    ) -> None:
        if not isinstance(operation_derivatives, tuple) or any(
            not isinstance(entry, OperationDerivativeContract)
            for entry in operation_derivatives
        ):
            raise TypeError(
                "operation_derivatives must be a tuple of operation contracts."
            )
        operations = tuple(entry.operation for entry in operation_derivatives)
        if len(set(operations)) != len(operations):
            raise ValueError("Derivative operation identifiers must be unique.")
        if operation_derivatives and default_operation not in operations:
            raise ValueError(
                "Operation metadata must name its default callable operation."
            )
        if not operation_derivatives and default_operation is not None:
            raise ValueError("default_operation requires operation metadata.")
        valid_ = jnp.asarray(valid, dtype=jnp.bool_)
        status_ = jnp.asarray(status, dtype=jnp.int32)
        for entry in operation_derivatives:
            if (
                entry.runtime_valid is not None
                and entry.runtime_valid.shape[:-1] != valid_.shape
            ):
                raise ValueError("Operation runtime gates must match the fit case shape.")
        self.model = model if isinstance(model, FrozenModel) else FrozenModel(model)
        self.diagnostics = diagnostics
        self.valid = valid_
        self.status = status_
        self.derivative_contract = derivative_contract
        self.method = str(method)
        self.operation_derivatives = operation_derivatives
        self.default_operation = default_operation

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
            operation_derivatives=self.operation_derivatives,
            default_operation=self.default_operation,
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
        operation: str | None = None,
    ) -> DerivativeAdmission:
        """Admit the default callable or an explicitly named fitted operation."""
        entry = self._operation_derivative(operation)
        if entry is None:
            return self.derivative_contract.admit(request, policy=policy)
        return entry.admit(request, policy=policy)

    def require_derivative(
        self,
        request: DifferentiationRequest,
        /,
        *,
        policy: RegularityPolicy | None = None,
        operation: str | None = None,
    ) -> DerivativeAdmission:
        """Require static eligibility and only requested resolved numerical gates."""
        entry = self._operation_derivative(operation)
        if entry is None:
            return self.derivative_contract.require(request, policy=policy)
        return entry.require(request, policy=policy)

    def _operation_derivative(
        self, operation: str | None, /
    ) -> OperationDerivativeContract | None:
        name = self.default_operation if operation is None else operation
        if name is None:
            return None
        if not isinstance(name, str):
            raise TypeError("operation must be an identifier string or None.")
        for entry in self.operation_derivatives:
            if entry.operation == name:
                return entry
        raise ValueError(f"Unknown derivative operation {name!r}.")


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
    def accepts_fit_key(self) -> bool:
        """Whether a workflow may route its addressed key to this recipe."""
        return True

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
    "OperationDerivativeContract",
    "PredictionModel",
    "ProbabilityModel",
]
