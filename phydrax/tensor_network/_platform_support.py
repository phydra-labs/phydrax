#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..backends._types import BackendAvailability
from ..qualification import SupportTuple
from ._core import LocallyPurifiedDensity, MatrixProductOperator, MatrixProductState


if TYPE_CHECKING:
    from ..solver._production_resources import ProductionResourceForecast


class TensorNetworkMaturity(StrEnum):
    EXPERIMENTAL = "experimental"
    QUALIFIED = "qualified"
    RELEASED = "released"


class TensorNetworkClaim(StrEnum):
    FINITE_EXECUTION = "finite-execution"
    RESOURCE_BOUNDEDNESS = "resource-boundedness"
    CHECKPOINT_INTEGRITY = "checkpoint-integrity"
    DETERMINISTIC_REPLAY = "deterministic-replay"
    INTERCHANGE_PORTABILITY = "interchange-portability"
    TELEMETRY_REDACTION = "telemetry-redaction"


class TensorNetworkFailure(StrEnum):
    NONE = "none"
    UNSUPPORTED_TUPLE = "unsupported-tuple"
    RESOURCE_REFUSED = "resource-refused"
    CHECKPOINT_NOT_ACCEPTED = "checkpoint-not-accepted"
    CHECKPOINT_MISMATCH = "checkpoint-mismatch"
    ARCHIVE_CORRUPTION = "archive-corruption"
    ARCHIVE_MISMATCH = "archive-mismatch"
    SECURITY_LIMIT = "security-limit"
    CANCELLED = "cancelled"
    REPLAY_MISMATCH = "replay-mismatch"
    EXECUTION_FAILED = "execution-failed"
    QUALIFICATION_FAILED = "qualification-failed"
    RELEASE_GATE_FAILED = "release-gate-failed"


def _identifier(value: object, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return value


def _positive_integer(value: object, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_integer(value: object, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


class TensorNetworkSupportTuple(StrictModule, NonTrainableState):
    """One exact tensor-network workflow coordinate tuple."""

    support_tuple: SupportTuple
    maturity: TensorNetworkMaturity = eqx.field(static=True)

    def __init__(
        self,
        *,
        representation: str,
        workflow: str,
        boundary: str,
        algorithm: str,
        backend: str,
        dtype: str,
        differentiation: str = "none",
        distribution: str = "single-process",
        maturity: TensorNetworkMaturity = TensorNetworkMaturity.EXPERIMENTAL,
    ):
        representation_ = _identifier(representation, "representation")
        if representation_ not in ("mps", "mpo", "lpdo", "array-pytree"):
            raise ValueError("Tensor-network representation is not supported.")
        dtype_ = np.dtype(_identifier(dtype, "dtype"))
        if dtype_.hasobject or dtype_.kind not in "biufc":
            raise ValueError("Tensor-network support requires a numerical dtype.")
        maturity_ = TensorNetworkMaturity(maturity)
        self.support_tuple = SupportTuple(
            "tensor-network.workflow",
            {
                "algorithm": _identifier(algorithm, "algorithm"),
                "backend": _identifier(backend, "backend"),
                "boundary": _identifier(boundary, "boundary"),
                "differentiation": _identifier(differentiation, "differentiation"),
                "distribution": _identifier(distribution, "distribution"),
                "dtype": dtype_.name,
                "maturity": maturity_.value,
                "representation": representation_,
                "workflow": _identifier(workflow, "workflow"),
            },
        )
        self.maturity = maturity_

    def _coordinate(self, name: str, /) -> str:
        for coordinate, value in self.support_tuple.attributes:
            if coordinate == name:
                return str(value)
        raise KeyError(f"No tensor-network support coordinate {name!r}.")

    @property
    def support_tuple_id(self) -> str:
        return self.support_tuple.support_tuple_id

    @property
    def representation(self) -> str:
        return self._coordinate("representation")

    @property
    def workflow(self) -> str:
        return self._coordinate("workflow")

    @property
    def boundary(self) -> str:
        return self._coordinate("boundary")

    @property
    def algorithm(self) -> str:
        return self._coordinate("algorithm")

    @property
    def backend(self) -> str:
        return self._coordinate("backend")

    @property
    def dtype(self) -> str:
        return self._coordinate("dtype")

    @property
    def differentiation(self) -> str:
        return self._coordinate("differentiation")

    @property
    def distribution(self) -> str:
        return self._coordinate("distribution")


class TensorNetworkResourcePolicy(StrictModule, NonTrainableState):
    production_budget: Any
    maximum_array_leaves: int = eqx.field(static=True)
    maximum_array_rank: int = eqx.field(static=True)
    maximum_elements: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_compile_units: int,
        maximum_host_bytes: int,
        maximum_device_bytes: int,
        maximum_output_queue_bytes: int,
        maximum_array_leaves: int = 100_000,
        maximum_array_rank: int = 16,
        maximum_elements: int = 1_000_000_000,
    ):
        from ..solver._production_resources import ProductionResourceBudget

        budget = ProductionResourceBudget(
            maximum_compile_units=_positive_integer(
                maximum_compile_units, "maximum_compile_units"
            ),
            maximum_host_bytes=_positive_integer(
                maximum_host_bytes, "maximum_host_bytes"
            ),
            maximum_device_bytes=_positive_integer(
                maximum_device_bytes, "maximum_device_bytes"
            ),
            maximum_output_queue_bytes=_positive_integer(
                maximum_output_queue_bytes, "maximum_output_queue_bytes"
            ),
        )
        leaves = _positive_integer(maximum_array_leaves, "maximum_array_leaves")
        rank = _positive_integer(maximum_array_rank, "maximum_array_rank")
        elements = _positive_integer(maximum_elements, "maximum_elements")
        self.production_budget = budget
        self.maximum_array_leaves = leaves
        self.maximum_array_rank = rank
        self.maximum_elements = elements
        self.policy_id = canonical_fingerprint(
            {
                "kind": "tensor-network-resource-policy",
                "production_budget": budget.budget_id,
                "maximum_array_leaves": leaves,
                "maximum_array_rank": rank,
                "maximum_elements": elements,
            }
        )


class TensorNetworkResourceForecast(StrictModule, NonTrainableState):
    support_tuple_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    array_count: int = eqx.field(static=True)
    total_elements: int = eqx.field(static=True)
    maximum_observed_rank: int = eqx.field(static=True)
    maximum_bond_dimension: int = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    production: Any
    shape_admitted: bool = eqx.field(static=True)
    forecast_id: str = eqx.field(static=True)

    def __init__(
        self,
        support_tuple_id: str,
        policy_id: str,
        array_count: int,
        total_elements: int,
        maximum_observed_rank: int,
        maximum_bond_dimension: int,
        storage_bytes: int,
        production: ProductionResourceForecast,
        shape_admitted: bool,
        /,
    ):
        from ..solver._production_resources import ProductionResourceForecast

        support_id = _identifier(support_tuple_id, "support_tuple_id")
        policy_id_ = _identifier(policy_id, "policy_id")
        values = tuple(
            _nonnegative_integer(value, name)
            for value, name in (
                (array_count, "array_count"),
                (total_elements, "total_elements"),
                (maximum_observed_rank, "maximum_observed_rank"),
                (maximum_bond_dimension, "maximum_bond_dimension"),
                (storage_bytes, "storage_bytes"),
            )
        )
        if values[0] == 0 or values[1] == 0 or values[4] == 0:
            raise ValueError("Resource forecasts require nonempty numerical storage.")
        if not isinstance(production, ProductionResourceForecast):
            raise TypeError("production must be ProductionResourceForecast.")
        if (
            any(
                value < 0
                for value in (
                    production.compile_units,
                    production.host_bytes,
                    production.device_bytes,
                    production.ad_bytes,
                    production.output_queue_bytes,
                )
            )
            or not production.forecast_id
        ):
            raise ValueError("Production resource forecast is invalid.")
        self.support_tuple_id = support_id
        self.policy_id = policy_id_
        (
            self.array_count,
            self.total_elements,
            self.maximum_observed_rank,
            self.maximum_bond_dimension,
            self.storage_bytes,
        ) = values
        self.production = production
        self.shape_admitted = bool(shape_admitted)
        self.forecast_id = canonical_fingerprint(
            {
                "kind": "tensor-network-resource-forecast",
                "support": support_id,
                "policy": policy_id_,
                "array_count": values[0],
                "total_elements": values[1],
                "maximum_observed_rank": values[2],
                "maximum_bond_dimension": values[3],
                "storage_bytes": values[4],
                "production": production.forecast_id,
                "shape_admitted": bool(shape_admitted),
            }
        )


class TensorNetworkResourceAdmission(StrictModule, NonTrainableState):
    forecast: TensorNetworkResourceForecast
    admitted: bool = eqx.field(static=True)
    failure: TensorNetworkFailure = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)

    def __init__(
        self,
        forecast: TensorNetworkResourceForecast,
        /,
        *,
        admitted: bool,
        failure: TensorNetworkFailure,
        reasons: Sequence[str],
    ):
        if not isinstance(forecast, TensorNetworkResourceForecast):
            raise TypeError("forecast must be TensorNetworkResourceForecast.")
        failure_ = TensorNetworkFailure(failure)
        reasons_ = tuple(_identifier(value, "admission reason") for value in reasons)
        admitted_ = bool(admitted)
        if admitted_ != (failure_ == TensorNetworkFailure.NONE):
            raise ValueError("Admission success and failure category are inconsistent.")
        if admitted_ == bool(reasons_):
            raise ValueError("Only refused admissions may contain reasons.")
        self.forecast = forecast
        self.admitted = admitted_
        self.failure = failure_
        self.reasons = reasons_
        self.admission_id = canonical_fingerprint(
            {
                "kind": "tensor-network-resource-admission",
                "forecast": forecast.forecast_id,
                "admitted": admitted_,
                "failure": failure_.value,
                "reasons": reasons_,
            }
        )

    def require_admitted(self) -> str:
        if not self.admitted:
            raise TensorNetworkAdmissionError(self.failure, self.reasons)
        return self.admission_id


class TensorNetworkAdmissionError(RuntimeError):
    failure: TensorNetworkFailure
    reasons: tuple[str, ...]

    def __init__(self, failure: TensorNetworkFailure, reasons: Sequence[str], /):
        failure_ = TensorNetworkFailure(failure)
        reasons_ = tuple(str(value) for value in reasons)
        if failure_ == TensorNetworkFailure.NONE or not reasons_:
            raise ValueError("Admission errors require a failure and reasons.")
        self.failure = failure_
        self.reasons = reasons_
        super().__init__(f"{failure_.value}: {'; '.join(reasons_)}")


def _array_leaves(value: Any, representation: str, /) -> tuple[np.ndarray, ...]:
    if representation == "mps":
        if not isinstance(value, MatrixProductState):
            raise TypeError("The mps support tuple requires MatrixProductState input.")
        leaves = value.tensors
    elif representation == "mpo":
        if not isinstance(value, MatrixProductOperator):
            raise TypeError("The mpo support tuple requires MatrixProductOperator input.")
        leaves = value.tensors
    elif representation == "lpdo":
        if not isinstance(value, LocallyPurifiedDensity):
            raise TypeError(
                "The lpdo support tuple requires LocallyPurifiedDensity input."
            )
        leaves = value.tensors
    else:
        leaves = tuple(jax.tree.leaves(value))
    arrays = tuple(np.asarray(leaf) for leaf in leaves)
    if not arrays or any(array.dtype.hasobject for array in arrays):
        raise TypeError("Tensor-network resources require a nonempty array-only PyTree.")
    if any(array.dtype.kind not in "biufc" for array in arrays):
        raise TypeError("Tensor-network resources require numerical array leaves.")
    return arrays


def forecast_tensor_network_resources(
    value: Any,
    support: TensorNetworkSupportTuple,
    policy: TensorNetworkResourcePolicy,
    /,
    *,
    device_workspace_copies: int = 2,
    differentiation_workspace_copies: int = 0,
    output_snapshots: int = 1,
) -> TensorNetworkResourceForecast:
    """Compute a finite conservative array-residency forecast without execution."""
    from ..solver._production_resources import ProductionResourceForecast

    if not isinstance(support, TensorNetworkSupportTuple) or not isinstance(
        policy, TensorNetworkResourcePolicy
    ):
        raise TypeError("Resource forecasting requires support and resource policy.")
    device_copies = _nonnegative_integer(
        device_workspace_copies, "device_workspace_copies"
    )
    differentiation_copies = _nonnegative_integer(
        differentiation_workspace_copies, "differentiation_workspace_copies"
    )
    snapshots = _nonnegative_integer(output_snapshots, "output_snapshots")
    arrays = _array_leaves(value, support.representation)
    if any(np.dtype(array.dtype).name != support.dtype for array in arrays):
        raise ValueError("Resource input dtype does not match the exact support tuple.")
    array_count = len(arrays)
    total_elements = sum(int(array.size) for array in arrays)
    maximum_rank = max(array.ndim for array in arrays)
    storage_bytes = sum(int(array.nbytes) for array in arrays)
    if support.representation in ("mps", "mpo", "lpdo"):
        maximum_bond = max(
            max(int(array.shape[0]), int(array.shape[-1])) for array in arrays
        )
    else:
        maximum_bond = 0
    compile_units = sum(array.ndim + 1 for array in arrays)
    host_bytes = storage_bytes
    device_bytes = storage_bytes * (1 + device_copies)
    differentiation_bytes = storage_bytes * differentiation_copies
    output_bytes = storage_bytes * snapshots
    budget = policy.production_budget
    production_admitted = bool(
        compile_units <= budget.maximum_compile_units
        and host_bytes <= budget.maximum_host_bytes
        and device_bytes + differentiation_bytes <= budget.maximum_device_bytes
        and output_bytes <= budget.maximum_output_queue_bytes
    )
    production_id = canonical_fingerprint(
        {
            "kind": "tensor-network-production-resource-forecast",
            "support": support.support_tuple_id,
            "policy": policy.policy_id,
            "compile_units": compile_units,
            "host_bytes": host_bytes,
            "device_bytes": device_bytes,
            "ad_bytes": differentiation_bytes,
            "output_queue_bytes": output_bytes,
        }
    )
    production = ProductionResourceForecast(
        compile_units,
        host_bytes,
        device_bytes,
        differentiation_bytes,
        output_bytes,
        production_admitted,
        production_id,
    )
    shape_admitted = bool(
        array_count <= policy.maximum_array_leaves
        and maximum_rank <= policy.maximum_array_rank
        and total_elements <= policy.maximum_elements
    )
    return TensorNetworkResourceForecast(
        support.support_tuple_id,
        policy.policy_id,
        array_count,
        total_elements,
        maximum_rank,
        maximum_bond,
        storage_bytes,
        production,
        shape_admitted,
    )


def admit_tensor_network_resources(
    forecast: TensorNetworkResourceForecast,
    supported_tuples: Sequence[TensorNetworkSupportTuple],
    /,
) -> TensorNetworkResourceAdmission:
    """Admit only an exact declared support tuple within every resource bound."""

    if not isinstance(forecast, TensorNetworkResourceForecast):
        raise TypeError("forecast must be TensorNetworkResourceForecast.")
    supported = tuple(supported_tuples)
    if not supported or any(
        not isinstance(value, TensorNetworkSupportTuple) for value in supported
    ):
        raise TypeError("supported_tuples must be a nonempty typed sequence.")
    identifiers = tuple(value.support_tuple_id for value in supported)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("supported_tuples contains duplicate exact tuples.")
    if forecast.support_tuple_id not in identifiers:
        return TensorNetworkResourceAdmission(
            forecast,
            admitted=False,
            failure=TensorNetworkFailure.UNSUPPORTED_TUPLE,
            reasons=("exact support tuple is not declared",),
        )
    reasons: list[str] = []
    production = forecast.production
    if not forecast.shape_admitted:
        reasons.append("array shape or count capacity exceeded")
    if not production.admitted:
        reasons.append("production byte or compile-unit budget exceeded")
    if reasons:
        return TensorNetworkResourceAdmission(
            forecast,
            admitted=False,
            failure=TensorNetworkFailure.RESOURCE_REFUSED,
            reasons=tuple(reasons),
        )
    return TensorNetworkResourceAdmission(
        forecast,
        admitted=True,
        failure=TensorNetworkFailure.NONE,
        reasons=(),
    )


def _backend_evidence_id(evidence: BackendAvailability, /) -> str:
    capabilities = evidence.capabilities
    return canonical_fingerprint(
        {
            "kind": "tensor-network-backend-evidence",
            "backend": evidence.backend,
            "available": evidence.available,
            "requirement": evidence.requirement,
            "reason": evidence.reason,
            "versions": evidence.versions,
            "problem_kinds": capabilities.problem_kinds,
            "execution": capabilities.execution,
            "host_only": capabilities.host_only,
            "matrix_free": capabilities.supports_matrix_free,
            "assembled": capabilities.supports_assembled,
            "lifecycle": capabilities.supports_plan_prepare_solve_refresh,
            "requires_release": capabilities.requires_explicit_release,
            "coordinate_dtypes": capabilities.coordinate_dtypes,
        }
    )


class TensorNetworkExecutionManifest(StrictModule, NonTrainableState):
    support: TensorNetworkSupportTuple
    admission: TensorNetworkResourceAdmission
    backend_evidence: BackendAvailability
    structure_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    input_id: str = eqx.field(static=True)
    backend_evidence_id: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: TensorNetworkSupportTuple,
        admission: TensorNetworkResourceAdmission,
        backend_evidence: BackendAvailability,
        /,
        *,
        structure_id: str,
        method_id: str,
        precision_policy_id: str,
        source_id: str,
        input_id: str,
    ):
        if not isinstance(support, TensorNetworkSupportTuple):
            raise TypeError("support must be TensorNetworkSupportTuple.")
        if not isinstance(admission, TensorNetworkResourceAdmission):
            raise TypeError("admission must be TensorNetworkResourceAdmission.")
        if not isinstance(backend_evidence, BackendAvailability):
            raise TypeError("backend_evidence must be BackendAvailability.")
        admission.require_admitted()
        if admission.forecast.support_tuple_id != support.support_tuple_id:
            raise TensorNetworkAdmissionError(
                TensorNetworkFailure.UNSUPPORTED_TUPLE,
                ("admission and execution support tuples differ",),
            )
        if backend_evidence.backend != support.backend:
            raise ValueError("Backend evidence does not match the exact support tuple.")
        if not backend_evidence.available:
            raise ValueError("Execution manifest requires available backend evidence.")
        if not (
            backend_evidence.capabilities.supports(support.workflow)
            or backend_evidence.capabilities.supports(support.representation)
        ):
            raise ValueError("Backend evidence does not declare this tensor workflow.")
        identifiers = tuple(
            _identifier(value, name)
            for value, name in (
                (structure_id, "structure_id"),
                (method_id, "method_id"),
                (precision_policy_id, "precision_policy_id"),
                (source_id, "source_id"),
                (input_id, "input_id"),
            )
        )
        if identifiers[1] != support.algorithm:
            raise ValueError(
                "Execution method_id must equal the exact support-tuple algorithm."
            )
        evidence_id = _backend_evidence_id(backend_evidence)
        self.support = support
        self.admission = admission
        self.backend_evidence = backend_evidence
        (
            self.structure_id,
            self.method_id,
            self.precision_policy_id,
            self.source_id,
            self.input_id,
        ) = identifiers
        self.backend_evidence_id = evidence_id
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "tensor-network-execution-manifest",
                "support": support.support_tuple_id,
                "admission": admission.admission_id,
                "backend_evidence": evidence_id,
                "structure": identifiers[0],
                "method": identifiers[1],
                "precision_policy": identifiers[2],
                "source": identifiers[3],
                "input": identifiers[4],
            }
        )


__all__ = [
    "TensorNetworkAdmissionError",
    "TensorNetworkClaim",
    "TensorNetworkExecutionManifest",
    "TensorNetworkFailure",
    "TensorNetworkMaturity",
    "TensorNetworkResourceAdmission",
    "TensorNetworkResourceForecast",
    "TensorNetworkResourcePolicy",
    "TensorNetworkSupportTuple",
    "admit_tensor_network_resources",
    "forecast_tensor_network_resources",
]
