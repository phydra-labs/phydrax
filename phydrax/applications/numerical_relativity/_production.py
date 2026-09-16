#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail-closed production bindings for fixed-grid numerical relativity."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from threading import RLock
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._execution_plan import ExecutionPlan
from ..._execution_resources import ExecutionPolicy, ResourceRequest
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...interchange._black_hole import (
    BlackHoleArtifactRights,
    BlackHoleArtifactUsePolicy,
    NeutralBlackHoleArtifact,
)
from ...lifecycle._resolved_run import ResolvedRunSpec
from ...qualification._evidence import SupportDependency
from ...qualification._registry import SupportTuple
from ...solver._fixed_step import AbstractFixedStepMethod, FixedStepResult
from ...solver._grrmhd_runtime import (
    FixedGridGRRMHDIMEXPlan,
    GRRMHDState,
)
from ...solver._production_runtime import (
    ArtifactCheckpointStore,
    CheckpointCommitReceipt,
    CheckpointGenerationPolicy,
    DurableCheckpointStore,
    PreparedProductionRun,
    ProductionCaseManifest,
    ProductionFailureRecord,
    ProductionRunPlan,
    ProductionRunResult,
    ProductionRunState,
)
from ...solver._relativistic_finite_volume import ValenciaFiniteVolumeStageGeometry
from ...solver._runtime_lifecycle import ByteBoundedAsyncPublisher
from ._status import NumericalRelativityStatus
from ._temporal import FixedGridZ4cRuntime, Z4cRuntimeState


SupportScope: TypeAlias = Literal["scientific", "deployment"]
FailureCategory: TypeAlias = Literal[
    "state-invalid",
    "step-rejected",
    "step-capacity-exhausted",
    "output-failed",
    "runtime-failed",
]
_FAILURE_CATEGORIES = frozenset(
    (
        "state-invalid",
        "step-rejected",
        "step-capacity-exhausted",
        "output-failed",
        "runtime-failed",
    )
)
_DOMAIN_COORDINATES = (
    "formulation_id",
    "chart_id",
    "gauge_id",
    "eos_id",
    "topology_id",
    "precision_id",
)


def _identifier(value: object, owner: str, /) -> str:
    if type(value) is not str or not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{owner} must be a non-empty canonical identifier.")
    return value


def _positive_integer(value: object, owner: str, /) -> int:
    if type(value) is not int:
        raise TypeError(f"{owner} must be an exact integer.")
    if value <= 0:
        raise ValueError(f"{owner} must be positive.")
    return value


def _index(value: object, owner: str, /) -> int:
    if type(value) is not int:
        raise TypeError(f"{owner} must be an exact integer.")
    if value < 0:
        raise ValueError(f"{owner} must be non-negative.")
    return value


def _finite_time(value: object, owner: str, /) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{owner} must be a real scalar.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{owner} must be finite and non-negative.")
    return result


def _host_bool(value: object, owner: str, /) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype(bool):
        raise TypeError(f"{owner} must be a scalar Boolean array.")
    return bool(array)


def _bounded_detail(value: object, owner: str, maximum_bytes: int, /) -> str:
    detail = _identifier(value, owner)
    if len(detail.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{owner} exceeds the configured UTF-8 byte bound.")
    return detail


class Z4cProductionState(StrictModule):
    """Fixed-shape committed Z4c state retaining independent scientific status."""

    runtime_state: Z4cRuntimeState
    status: Any
    finite: Any
    converged: Any
    physically_valid: Any
    qualified: Any
    derivative_valid: Any


class FixedGridZ4cProductionMethod(AbstractFixedStepMethod):
    """Generic fixed-step adapter preserving all Z4c runtime dispositions."""

    runtime: FixedGridZ4cRuntime
    method_id: str = eqx.field(static=True)

    def __init__(self, runtime: FixedGridZ4cRuntime, /):
        if not isinstance(runtime, FixedGridZ4cRuntime):
            raise TypeError("runtime must be FixedGridZ4cRuntime.")
        self.runtime = runtime
        self.method_id = canonical_fingerprint(
            {
                "kind": "fixed-grid-z4c-production-method",
                "runtime": runtime.runtime_id,
            }
        )

    @property
    def required_step_size(self) -> float:
        return self.runtime.time_step

    @property
    def allows_step_reduction(self) -> bool:
        return False

    def initialize(self, state: Z4cRuntimeState, /) -> Z4cProductionState:
        if (
            not isinstance(state, Z4cRuntimeState)
            or state.runtime_id != self.runtime.runtime_id
        ):
            raise ValueError(
                "Initial state does not belong to the fixed-grid Z4c runtime."
            )
        finite = jnp.all(jnp.isfinite(state.state.values)) & jnp.isfinite(state.time)
        status = jnp.where(
            finite,
            jnp.asarray(int(NumericalRelativityStatus.SUCCESS), dtype=jnp.int32),
            jnp.asarray(int(NumericalRelativityStatus.NONFINITE_STATE), dtype=jnp.int32),
        )
        unavailable = jnp.asarray(False)
        return Z4cProductionState(
            state,
            status,
            finite,
            unavailable,
            unavailable,
            unavailable,
            unavailable,
        )

    def step(
        self,
        step_index,
        time,
        state: Z4cProductionState,
        step_size,
        args: Any,
        /,
    ) -> FixedStepResult:
        if (
            not isinstance(state, Z4cProductionState)
            or state.runtime_state.runtime_id != self.runtime.runtime_id
        ):
            raise ValueError(
                "Production state does not belong to the fixed-grid Z4c runtime."
            )
        if args is not None and not callable(args):
            raise TypeError(
                "Z4c production args must be a snapshot-aware stress-energy provider or None."
            )
        source = state.runtime_state
        evaluation = self.runtime.evaluate(source, stress_energy_provider=args)
        dtype = source.time.dtype
        scale = jnp.maximum(jnp.abs(source.time), jnp.asarray(1.0, dtype=dtype))
        tolerance = jnp.asarray(64.0, dtype=dtype) * jnp.finfo(dtype).eps * scale
        consistent = (
            (jnp.asarray(step_index) == source.step_index)
            & (jnp.abs(jnp.asarray(time, dtype=dtype) - source.time) <= tolerance)
            & (
                jnp.abs(
                    jnp.asarray(step_size, dtype=dtype)
                    - jnp.asarray(self.runtime.time_step, dtype=dtype)
                )
                <= tolerance
            )
        )
        accepted_runtime = self.runtime.accept(evaluation, consistent)
        successful = evaluation.successful & consistent
        candidate = Z4cProductionState(
            evaluation.candidate,
            evaluation.status,
            evaluation.finite,
            evaluation.converged,
            evaluation.physically_valid,
            evaluation.qualified,
            evaluation.derivative_valid,
        )
        accepted = Z4cProductionState(
            accepted_runtime,
            jnp.where(successful, evaluation.status, state.status),
            jnp.where(successful, evaluation.finite, state.finite),
            jnp.where(successful, evaluation.converged, state.converged),
            jnp.where(successful, evaluation.physically_valid, state.physically_valid),
            jnp.where(successful, evaluation.qualified, state.qualified),
            jnp.where(successful, evaluation.derivative_valid, state.derivative_valid),
        )
        return FixedStepResult(
            candidate,
            accepted,
            successful,
            evaluation.constraints.maximum_norm,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(
                3 if self.runtime.integrator == "ssprk33" else 5, dtype=jnp.int32
            ),
            evaluation.enforcement.applied,
            evaluation.enforcement.correction_norm,
        )


class GRRMHDProductionArguments(StrictModule):
    stage_geometries: tuple[
        ValenciaFiniteVolumeStageGeometry,
        ValenciaFiniteVolumeStageGeometry,
    ]
    composition: Any
    transport_extinction: Any


class GRRMHDProductionState(StrictModule):
    runtime_state: GRRMHDState
    status: Any
    finite: Any
    converged: Any
    physically_valid: Any
    qualified: Any
    derivative_valid: Any


ProductionScientificState: TypeAlias = Z4cProductionState | GRRMHDProductionState


def _scientific_state_step_time(
    state: ProductionScientificState, /
) -> tuple[Array, Array]:
    if isinstance(state, Z4cProductionState):
        return state.runtime_state.step_index, state.runtime_state.time
    if isinstance(state, GRRMHDProductionState):
        return state.runtime_state.accepted_step, state.runtime_state.time
    raise TypeError("Unknown numerical-relativity production state.")


class FixedGridGRRMHDProductionMethod(AbstractFixedStepMethod):
    """Fixed-step production adapter preserving GRRMHD dispositions."""

    runtime: FixedGridGRRMHDIMEXPlan
    fixed_step_size: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: FixedGridGRRMHDIMEXPlan,
        /,
        *,
        fixed_step_size: float,
    ) -> None:
        if not isinstance(runtime, FixedGridGRRMHDIMEXPlan):
            raise TypeError("runtime must be FixedGridGRRMHDIMEXPlan.")
        step = float(fixed_step_size)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("GRRMHD production step size must be finite and positive.")
        self.runtime = runtime
        self.fixed_step_size = step
        self.method_id = canonical_fingerprint(
            {
                "kind": "fixed-grid-grrmhd-production-method",
                "runtime": runtime.plan_id,
                "step_size": step,
            }
        )

    @property
    def required_step_size(self) -> float:
        return self.fixed_step_size

    @property
    def allows_step_reduction(self) -> bool:
        return False

    def initialize(self, state: GRRMHDState, /) -> GRRMHDProductionState:
        if not isinstance(state, GRRMHDState):
            raise TypeError("Initial state must be GRRMHDState.")
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in jax.tree.leaves(state)
                    if jnp.issubdtype(value.dtype, jnp.inexact)
                )
            )
        )
        unavailable = jnp.asarray(False)
        return GRRMHDProductionState(
            state,
            state.status,
            finite,
            unavailable,
            unavailable,
            unavailable,
            unavailable,
        )

    def step(
        self,
        step_index,
        time,
        state: GRRMHDProductionState,
        step_size,
        args: Any,
        /,
    ) -> FixedStepResult:
        if not isinstance(state, GRRMHDProductionState):
            raise TypeError("Production state must be GRRMHDProductionState.")
        if not isinstance(args, GRRMHDProductionArguments):
            raise TypeError("GRRMHD production requires GRRMHDProductionArguments.")
        source = state.runtime_state
        dtype = source.time.dtype
        step = jnp.asarray(step_size, dtype=dtype).reshape(())
        time_ = jnp.asarray(time, dtype=dtype).reshape(())
        scale = jnp.maximum(jnp.abs(source.time), jnp.asarray(1.0, dtype=dtype))
        tolerance = 64.0 * jnp.finfo(dtype).eps * scale
        consistent = (
            (jnp.asarray(step_index) == source.accepted_step)
            & (jnp.abs(time_ - source.time) <= tolerance)
            & (
                jnp.abs(step - jnp.asarray(self.fixed_step_size, dtype=dtype))
                <= tolerance
            )
        )
        result = self.runtime.advance(
            source,
            time_,
            time_ + step,
            args.stage_geometries,
            args.composition,
            transport_extinction=args.transport_extinction,
        )
        successful = result.accepted & consistent
        candidate = GRRMHDProductionState(
            result.candidate,
            result.status,
            result.finite,
            result.converged,
            result.physically_valid,
            result.qualified,
            result.derivative_valid,
        )
        accepted = GRRMHDProductionState(
            jax.lax.cond(
                successful,
                lambda _: result.state,
                lambda _: source,
                operand=None,
            ),
            jnp.where(successful, result.status, state.status),
            jnp.where(successful, result.finite, state.finite),
            jnp.where(successful, result.converged, state.converged),
            jnp.where(successful, result.physically_valid, state.physically_valid),
            jnp.where(successful, result.qualified, state.qualified),
            jnp.where(successful, result.derivative_valid, state.derivative_valid),
        )
        residual = jnp.maximum(
            jnp.abs(result.attempted_ledger.combined_energy_defect),
            jnp.max(
                jnp.abs(result.attempted_ledger.combined_momentum_defect),
                initial=0.0,
            ),
        )
        iterations = sum(
            source_result.ledger.maximum_iterations
            for source_result in result.stages.source_results
        )
        return FixedStepResult(
            candidate,
            accepted,
            successful,
            residual,
            jnp.asarray(iterations, dtype=jnp.int32),
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(0.0, dtype=dtype),
        )


class NumericalRelativitySupportBinding(StrictModule, NonTrainableState):
    """One qualified support tuple bound to its exact resolved-run profile."""

    scope: SupportScope = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    support: SupportTuple
    dependency: SupportDependency
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        scope: SupportScope,
        profile_id: str,
        support: SupportTuple,
        /,
    ):
        if scope not in ("scientific", "deployment"):
            raise ValueError("Support scope must be scientific or deployment.")
        profile = _identifier(profile_id, "Support profile ID")
        if not isinstance(support, SupportTuple):
            raise TypeError("support must be a SupportTuple.")
        dependency = SupportDependency(profile, support.support_tuple_id)
        self.scope = scope
        self.profile_id = profile
        self.support = support
        self.dependency = dependency
        self.binding_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-support-binding",
                "scope": scope,
                "profile_id": profile,
                "support_tuple": support.support_tuple_id,
                "dependency": dependency.dependency_id,
            }
        )


class NumericalRelativityArtifactBinding(StrictModule, NonTrainableState):
    """Small immutable production identity projected from an admitted artifact."""

    rights: BlackHoleArtifactRights
    use_policy: BlackHoleArtifactUsePolicy
    artifact_kind: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)
    content_sha256: str = eqx.field(static=True)
    size_bytes: int = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    rights_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    producer: str = eqx.field(static=True)
    producer_version: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    coverage: str = eqx.field(static=True)
    source_uri: str = eqx.field(static=True)
    attribution_id: str = eqx.field(static=True)
    commercial_use: bool = eqx.field(static=True)
    training_use: bool = eqx.field(static=True)
    redistribution: bool = eqx.field(static=True)
    derivative_use: bool = eqx.field(static=True)
    model_execution: bool = eqx.field(static=True)
    export: bool = eqx.field(static=True)
    use_policy_id: str = eqx.field(static=True)
    intended_use: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(self, artifact: NeutralBlackHoleArtifact, /):
        if (
            not isinstance(artifact, NeutralBlackHoleArtifact)
            or not artifact.report.valid
        ):
            raise TypeError(
                "Production artifacts must be valid neutral black-hole artifacts."
            )
        artifact.rights.require(artifact.use_policy)
        self.artifact_kind = artifact.schema.artifact_kind
        self.artifact_id = artifact.artifact_id
        self.content_sha256 = artifact.resource.content_sha256
        self.size_bytes = artifact.resource.size_bytes
        self.license_id = artifact.rights.license_id
        self.rights_id = artifact.rights.rights_id
        self.rights = artifact.rights
        self.use_policy = artifact.use_policy
        self.producer = artifact.rights.producer
        self.producer_version = artifact.rights.producer_version
        self.model_id = artifact.rights.model_id
        self.coverage = artifact.rights.coverage
        self.source_uri = artifact.rights.source_uri
        self.attribution_id = artifact.rights.attribution_id
        self.commercial_use = artifact.rights.commercial_use
        self.training_use = artifact.rights.training_use
        self.redistribution = artifact.rights.redistribution
        self.derivative_use = artifact.rights.derivative_use
        self.model_execution = artifact.rights.model_execution
        self.export = artifact.rights.export
        self.use_policy_id = artifact.use_policy.policy_id
        self.intended_use = artifact.use_policy.intended_use
        self.schema_id = artifact.schema.schema_id
        self.binding_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-artifact-binding",
                "artifact_kind": self.artifact_kind,
                "artifact_id": self.artifact_id,
                "content_sha256": self.content_sha256,
                "size_bytes": self.size_bytes,
                "license_id": self.license_id,
                "rights_id": self.rights_id,
                "use_policy_id": self.use_policy_id,
                "producer": self.producer,
                "producer_version": self.producer_version,
                "model_id": self.model_id,
                "coverage": self.coverage,
                "source_uri": self.source_uri,
                "attribution_id": self.attribution_id,
                "permissions": [
                    self.commercial_use,
                    self.training_use,
                    self.redistribution,
                    self.derivative_use,
                    self.model_execution,
                    self.export,
                ],
                "intended_use": self.intended_use,
                "schema_id": self.schema_id,
            }
        )


class NumericalRelativityCommittedOutputReceipt(StrictModule, NonTrainableState):
    """Writer-issued receipt for artifacts committed from one runtime snapshot."""

    production_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    writer_id: str = eqx.field(static=True)
    event_id: str = eqx.field(static=True)
    output_cursor: int = eqx.field(static=True)
    step_index: int = eqx.field(static=True)
    time: float = eqx.field(static=True)
    state_id: str = eqx.field(static=True)
    artifacts: tuple[NumericalRelativityArtifactBinding, ...]
    total_bytes: int = eqx.field(static=True)
    status: int = eqx.field(static=True)
    finite: bool = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    physically_valid: bool = eqx.field(static=True)
    qualified: bool = eqx.field(static=True)
    derivative_valid: bool = eqx.field(static=True)
    receipt_id: str = eqx.field(static=True)

    def __init__(
        self,
        production_id: str,
        run_id: str,
        writer_id: str,
        event_id: str,
        output_cursor: int,
        state: ProductionScientificState,
        artifacts: Sequence[NumericalRelativityArtifactBinding],
        /,
    ):
        if not isinstance(state, (Z4cProductionState, GRRMHDProductionState)):
            raise TypeError("Committed output state has an unsupported runtime type.")
        bindings = tuple(artifacts)
        if not bindings or any(
            not isinstance(value, NumericalRelativityArtifactBinding)
            for value in bindings
        ):
            raise TypeError("Committed output requires typed artifact bindings.")
        bindings = tuple(sorted(bindings, key=lambda value: value.artifact_id))
        identifiers = (
            _identifier(production_id, "Receipt production ID"),
            _identifier(run_id, "Receipt run ID"),
            _identifier(writer_id, "Receipt writer ID"),
            _identifier(event_id, "Receipt event ID"),
        )
        runtime_step, runtime_time = _scientific_state_step_time(state)
        statuses = (
            _host_bool(state.finite, "finite"),
            _host_bool(state.converged, "converged"),
            _host_bool(state.physically_valid, "physically_valid"),
            _host_bool(state.qualified, "qualified"),
            _host_bool(state.derivative_valid, "derivative_valid"),
        )
        self.production_id, self.run_id, self.writer_id, self.event_id = identifiers
        self.output_cursor = _index(output_cursor, "Output cursor")
        self.step_index = _index(int(np.asarray(runtime_step)), "Receipt step index")
        self.time = _finite_time(float(np.asarray(runtime_time)), "Receipt time")
        status = np.asarray(state.status)
        if status.shape != () or not np.issubdtype(status.dtype, np.integer):
            raise TypeError("Receipt status must be a scalar integer array.")
        self.state_id = canonical_fingerprint(array_tree_fingerprint(state))
        self.artifacts = bindings
        self.total_bytes = sum(value.size_bytes for value in bindings)
        self.status = int(status)
        (
            self.finite,
            self.converged,
            self.physically_valid,
            self.qualified,
            self.derivative_valid,
        ) = statuses
        self.receipt_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-committed-output-receipt",
                "production": self.production_id,
                "run": self.run_id,
                "writer": self.writer_id,
                "event": self.event_id,
                "output_cursor": self.output_cursor,
                "step_index": self.step_index,
                "time": self.time,
                "state": self.state_id,
                "artifacts": [value.binding_id for value in bindings],
                "total_bytes": self.total_bytes,
                "status": self.status,
                "dispositions": list(statuses),
            }
        )


class NumericalRelativityOutputCommitter:
    """Bounded writer boundary retaining receipts only after successful return."""

    __slots__ = (
        "_lock",
        "_next_cursor",
        "_prepared",
        "_production",
        "_publisher",
        "_receipts",
        "_writer",
        "writer_id",
    )

    def __init__(
        self,
        production: NumericalRelativityProductionPlan,
        writer: Callable[
            [str, ProductionScientificState], Sequence[NeutralBlackHoleArtifact]
        ],
        writer_id: str,
        /,
    ):
        if not isinstance(production, NumericalRelativityProductionPlan):
            raise TypeError("production must be NumericalRelativityProductionPlan.")
        if not callable(writer):
            raise TypeError("Output writer must be callable.")
        self._production = production
        self._writer = writer
        self.writer_id = _identifier(writer_id, "Output writer ID")
        self._prepared: PreparedProductionRun | None = None
        self._publisher: ByteBoundedAsyncPublisher | None = None
        self._next_cursor = 0
        self._receipts: dict[str, NumericalRelativityCommittedOutputReceipt] = {}
        self._lock = RLock()

    def _bind(
        self,
        prepared: PreparedProductionRun,
        publisher: ByteBoundedAsyncPublisher,
        /,
    ) -> None:
        with self._lock:
            if self._prepared is not None:
                raise ValueError("Output committer is already bound.")
            self._production._require_prepared(prepared)
            if prepared.publisher is not publisher:
                raise ValueError("Prepared runtime does not use this output publisher.")
            self._prepared = prepared
            self._publisher = publisher

    def __call__(
        self, event_id: str, state: ProductionScientificState, /
    ) -> NumericalRelativityCommittedOutputReceipt:
        with self._lock:
            prepared = self._prepared
            if prepared is None:
                raise RuntimeError("Output committer is not bound to a prepared runtime.")
            event = _identifier(event_id, "Output event ID")
            if event in self._receipts:
                raise ValueError("Output event was already committed.")
            if not self._production._valid_output_event(prepared, event):
                raise ValueError(
                    "Output event is not generated by the bound runtime plan."
                )
            artifacts = tuple(self._writer(event, state))
            bindings = tuple(
                NumericalRelativityArtifactBinding(value) for value in artifacts
            )
            artifact_ids = tuple(value.artifact_id for value in bindings)
            limits = self._production.limits
            if (
                not bindings
                or len(bindings) > limits.maximum_output_artifacts
                or len(set(artifact_ids)) != len(artifact_ids)
                or self._next_cursor >= limits.maximum_output_manifests
                or sum(value.size_bytes for value in bindings)
                > limits.maximum_output_bytes
            ):
                raise ValueError("Committed output artifacts exceed their finite bounds.")
            receipt = NumericalRelativityCommittedOutputReceipt(
                self._production.production_id,
                prepared.run_id,
                self.writer_id,
                event,
                self._next_cursor,
                state,
                bindings,
            )
            self._receipts[event] = receipt
            self._next_cursor += 1
            return receipt

    def committed_receipts(
        self, /
    ) -> tuple[NumericalRelativityCommittedOutputReceipt, ...]:
        with self._lock:
            if self._publisher is None:
                raise RuntimeError("Output committer is not bound.")
            acknowledged = self._publisher.acknowledged_event_ids
            receipts = tuple(
                sorted(self._receipts.values(), key=lambda value: value.output_cursor)
            )
            if any(value.event_id not in acknowledged for value in receipts):
                raise ValueError("At least one output receipt is not acknowledged.")
            return receipts

    def require_receipt(
        self,
        receipt: NumericalRelativityCommittedOutputReceipt,
        result: ProductionRunResult,
        /,
    ) -> None:
        with self._lock:
            if (
                not isinstance(receipt, NumericalRelativityCommittedOutputReceipt)
                or not isinstance(result, ProductionRunResult)
                or self._prepared is None
                or self._publisher is None
            ):
                raise TypeError("Committed output lineage is incomplete.")
            stored = self._receipts.get(receipt.event_id)
            if stored is None or stored.receipt_id != receipt.receipt_id:
                raise ValueError("Output receipt was not issued by this committer.")
            if receipt.event_id not in self._publisher.acknowledged_event_ids:
                raise ValueError("Output receipt has no publisher acknowledgement.")
            final = result.state.accepted_state
            if not isinstance(final, (Z4cProductionState, GRRMHDProductionState)):
                raise TypeError(
                    "Production result does not retain scientific dispositions."
                )
            final_step_value, final_time_value = _scientific_state_step_time(final)
            final_step = int(np.asarray(final_step_value))
            final_time = float(np.asarray(final_time_value))
            terminal_state_id = canonical_fingerprint(array_tree_fingerprint(final))
            if (
                result.run_id != self._prepared.run_id
                or receipt.production_id != self._production.production_id
                or receipt.run_id != result.run_id
                or receipt.writer_id != self.writer_id
                or int(np.asarray(result.state.step_index)) != final_step
                or float(np.asarray(result.state.time)) != final_time
                or int(np.asarray(result.state.output_cursor)) != len(self._receipts)
                or receipt.output_cursor >= int(np.asarray(result.state.output_cursor))
                or receipt.step_index > final_step
                or receipt.time > final_time
                or (
                    receipt.step_index == final_step
                    and receipt.time == final_time
                    and receipt.state_id != terminal_state_id
                )
            ):
                raise ValueError(
                    "Output receipt is outside the authoritative run lineage."
                )


class NumericalRelativityProductionLimits(StrictModule, NonTrainableState):
    """Finite host metadata and artifact bounds for one production profile."""

    execution_policy: ExecutionPolicy = eqx.field(static=True)
    resource_request: ResourceRequest = eqx.field(static=True)
    maximum_input_artifacts: int = eqx.field(static=True)
    maximum_input_bytes: int = eqx.field(static=True)
    maximum_output_manifests: int = eqx.field(static=True)
    maximum_output_artifacts: int = eqx.field(static=True)
    maximum_output_bytes: int = eqx.field(static=True)
    maximum_cancellation_detail_bytes: int = eqx.field(static=True)
    resource_policy_id: str = eqx.field(static=True)
    output_policy_id: str = eqx.field(static=True)
    limits_id: str = eqx.field(static=True)

    def __init__(
        self,
        execution_policy: ExecutionPolicy,
        /,
        *,
        maximum_input_artifacts: int,
        maximum_input_bytes: int,
        maximum_output_manifests: int,
        maximum_output_artifacts: int,
        maximum_output_bytes: int,
        maximum_cancellation_detail_bytes: int,
    ):
        if not isinstance(execution_policy, ExecutionPolicy):
            raise TypeError("execution_policy must be ExecutionPolicy.")
        resource_request = execution_policy.resources
        if not isinstance(resource_request, ResourceRequest):
            raise ValueError("Execution policy must bind a ResourceRequest.")
        if (
            resource_request.maximum_checkpoint_staging_bytes is None
            or resource_request.maximum_checkpoint_staging_bytes <= 0
            or resource_request.maximum_output_backlog_bytes is None
            or resource_request.maximum_output_backlog_bytes <= 0
        ):
            raise ValueError(
                "Production resources require checkpoint-staging and output-backlog bounds."
            )
        values = tuple(
            _positive_integer(value, name)
            for value, name in (
                (maximum_input_artifacts, "maximum_input_artifacts"),
                (maximum_input_bytes, "maximum_input_bytes"),
                (maximum_output_manifests, "maximum_output_manifests"),
                (maximum_output_artifacts, "maximum_output_artifacts"),
                (maximum_output_bytes, "maximum_output_bytes"),
                (maximum_cancellation_detail_bytes, "maximum_cancellation_detail_bytes"),
            )
        )
        if values[4] > resource_request.maximum_output_backlog_bytes:
            raise ValueError(
                "One output manifest cannot exceed the execution output-backlog bound."
            )
        if (
            resource_request.maximum_host_bytes is not None
            and values[1] > resource_request.maximum_host_bytes
        ):
            raise ValueError(
                "Input artifact bytes cannot exceed the execution host-memory bound."
            )
        (
            self.maximum_input_artifacts,
            self.maximum_input_bytes,
            self.maximum_output_manifests,
            self.maximum_output_artifacts,
            self.maximum_output_bytes,
            self.maximum_cancellation_detail_bytes,
        ) = values
        self.execution_policy = execution_policy
        self.resource_request = resource_request
        self.resource_policy_id = resource_request.resource_id
        self.output_policy_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-production-output-policy",
                "maximum_output_manifests": values[2],
                "maximum_output_artifacts": values[3],
                "maximum_output_bytes": values[4],
            }
        )
        self.limits_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-production-limits",
                "execution_policy": execution_policy.policy_id,
                "resource_policy": self.resource_policy_id,
                "resource_request": resource_request.resource_id,
                "output_policy": self.output_policy_id,
            }
        )


class NumericalRelativityDomainBinding(StrictModule, NonTrainableState):
    """Exact formulation, coordinates, matter, topology, precision, and inputs."""

    formulation_id: str = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)
    eos_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    input_artifacts: tuple[NumericalRelativityArtifactBinding, ...]
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        formulation_id: str,
        chart_id: str,
        gauge_id: str,
        eos_id: str,
        topology_id: str,
        precision_id: str,
        input_artifacts: Sequence[NeutralBlackHoleArtifact] = (),
    ):
        identifiers = tuple(
            _identifier(value, name)
            for value, name in zip(
                (
                    formulation_id,
                    chart_id,
                    gauge_id,
                    eos_id,
                    topology_id,
                    precision_id,
                ),
                _DOMAIN_COORDINATES,
                strict=True,
            )
        )
        artifacts = tuple(
            sorted(
                (NumericalRelativityArtifactBinding(value) for value in input_artifacts),
                key=lambda value: value.artifact_id,
            )
        )
        artifact_ids = tuple(value.artifact_id for value in artifacts)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("Numerical-relativity input artifacts must be unique.")
        (
            self.formulation_id,
            self.chart_id,
            self.gauge_id,
            self.eos_id,
            self.topology_id,
            self.precision_id,
        ) = identifiers
        self.input_artifacts = artifacts
        self.binding_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-domain-binding",
                **dict(zip(_DOMAIN_COORDINATES, identifiers, strict=True)),
                "input_artifacts": [value.binding_id for value in artifacts],
            }
        )


class NumericalRelativityRestartManifest(StrictModule, NonTrainableState):
    """Exact restart admission record; no topology or artifact substitution."""

    production_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)
    domain_binding_id: str = eqx.field(static=True)
    case_manifest_id: str = eqx.field(static=True)
    execution_plan_id: str = eqx.field(static=True)
    execution_plan_fingerprint: str = eqx.field(static=True)
    resolved_run_spec_id: str = eqx.field(static=True)
    run_plan_id: str = eqx.field(static=True)
    input_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    restart_id: str = eqx.field(static=True)

    def __init__(
        self,
        production: NumericalRelativityProductionPlan,
        prepared: PreparedProductionRun,
        checkpoint_id: str,
        /,
    ):
        production._require_prepared(prepared)
        checkpoint = _identifier(checkpoint_id, "Restart checkpoint ID")
        self.production_id = production.production_id
        self.run_id = prepared.run_id
        self.checkpoint_id = checkpoint
        self.domain_binding_id = production.domain.binding_id
        self.case_manifest_id = production.case_manifest.manifest_id
        self.execution_plan_id = production.execution_plan.execution_plan_id
        self.execution_plan_fingerprint = production.execution_plan.plan_fingerprint
        self.resolved_run_spec_id = production.resolved_run_spec.spec_id
        self.run_plan_id = production.run_plan.plan_id
        self.input_artifact_ids = tuple(
            value.artifact_id for value in production.domain.input_artifacts
        )
        self.restart_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-restart",
                "production": self.production_id,
                "run": self.run_id,
                "checkpoint": checkpoint,
                "domain": self.domain_binding_id,
                "case": self.case_manifest_id,
                "execution_plan_id": self.execution_plan_id,
                "execution_plan_fingerprint": self.execution_plan_fingerprint,
                "resolved_run": self.resolved_run_spec_id,
                "run_plan": self.run_plan_id,
                "input_artifacts": list(self.input_artifact_ids),
            }
        )


class NumericalRelativityOutputManifest(StrictModule, NonTrainableState):
    """One bounded output transaction with independent scientific dispositions."""

    production_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    receipt_id: str = eqx.field(static=True)
    writer_id: str = eqx.field(static=True)
    event_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)
    output_index: int = eqx.field(static=True)
    step_index: int = eqx.field(static=True)
    time: float = eqx.field(static=True)
    artifacts: tuple[NumericalRelativityArtifactBinding, ...]
    total_bytes: int = eqx.field(static=True)
    finite: bool = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    physically_valid: bool = eqx.field(static=True)
    qualified: bool = eqx.field(static=True)
    derivative_valid: bool = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        production: NumericalRelativityProductionPlan,
        prepared: PreparedProductionRun,
        committer: NumericalRelativityOutputCommitter,
        receipt: NumericalRelativityCommittedOutputReceipt,
        result: ProductionRunResult,
        /,
    ):
        production._require_prepared(prepared)
        if not isinstance(committer, NumericalRelativityOutputCommitter):
            raise TypeError(
                "Output manifest requires NumericalRelativityOutputCommitter."
            )
        committer.require_receipt(receipt, result)
        if receipt.output_cursor >= production.limits.maximum_output_manifests:
            raise ValueError("Output cursor exceeds the configured manifest bound.")
        bindings = receipt.artifacts
        ids = tuple(value.artifact_id for value in bindings)
        if (
            not bindings
            or len(bindings) > production.limits.maximum_output_artifacts
            or len(set(ids)) != len(ids)
            or receipt.total_bytes > production.limits.maximum_output_bytes
        ):
            raise ValueError("Committed output artifacts violate their finite bounds.")
        statuses = (
            receipt.finite,
            receipt.converged,
            receipt.physically_valid,
            receipt.qualified,
            receipt.derivative_valid,
        )
        self.production_id = production.production_id
        self.run_id = prepared.run_id
        self.receipt_id = receipt.receipt_id
        self.writer_id = receipt.writer_id
        self.event_id = receipt.event_id
        self.state_id = receipt.state_id
        self.output_index = receipt.output_cursor
        self.step_index = receipt.step_index
        self.time = receipt.time
        self.artifacts = bindings
        self.total_bytes = receipt.total_bytes
        (
            self.finite,
            self.converged,
            self.physically_valid,
            self.qualified,
            self.derivative_valid,
        ) = statuses
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-output-manifest",
                "production": self.production_id,
                "run": self.run_id,
                "receipt": self.receipt_id,
                "writer": self.writer_id,
                "event": self.event_id,
                "state": self.state_id,
                "output_index": self.output_index,
                "step_index": self.step_index,
                "time": self.time,
                "artifacts": [value.binding_id for value in bindings],
                "total_bytes": self.total_bytes,
                "status": receipt.status,
                "dispositions": list(statuses),
            }
        )


class NumericalRelativityFailureManifest(StrictModule, NonTrainableState):
    """Bounded terminal failure preserving the generic runtime evidence."""

    production_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    runtime_failure_id: str = eqx.field(static=True)
    step_index: int = eqx.field(static=True)
    time: float = eqx.field(static=True)
    category: FailureCategory = eqx.field(static=True)
    error_code: str = eqx.field(static=True)
    last_checkpoint_id: str = eqx.field(static=True)
    finite: bool = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    physically_valid: bool = eqx.field(static=True)
    qualified: bool = eqx.field(static=True)
    derivative_valid: bool = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        production: NumericalRelativityProductionPlan,
        prepared: PreparedProductionRun,
        failure: ProductionFailureRecord,
        state: ProductionScientificState,
        /,
        *,
        terminal_checkpoint_id: str,
    ):
        production._require_prepared(prepared)
        if not isinstance(failure, ProductionFailureRecord):
            raise TypeError("failure must be a ProductionFailureRecord.")
        if not isinstance(state, (Z4cProductionState, GRRMHDProductionState)):
            raise TypeError("Failure result has an unsupported scientific state.")
        if type(terminal_checkpoint_id) is not str:
            raise TypeError("terminal_checkpoint_id must be a string.")
        category = _identifier(failure.category, "Failure category")
        if category not in _FAILURE_CATEGORIES:
            raise ValueError("Failure category is outside the production contract.")
        error_code = _identifier(failure.error_code, "Failure error code")
        checkpoint_value = (
            terminal_checkpoint_id
            if terminal_checkpoint_id
            else failure.last_checkpoint_id
        )
        checkpoint = (
            ""
            if checkpoint_value == ""
            else _identifier(checkpoint_value, "Failure checkpoint ID")
        )
        statuses = (
            _host_bool(state.finite, "finite"),
            _host_bool(state.converged, "converged"),
            _host_bool(state.physically_valid, "physically_valid"),
            _host_bool(state.qualified, "qualified"),
            _host_bool(state.derivative_valid, "derivative_valid"),
        )
        self.production_id = production.production_id
        self.run_id = prepared.run_id
        self.runtime_failure_id = failure.failure_id
        self.step_index = _index(
            int(np.asarray(failure.step_index)), "Failure step index"
        )
        self.time = _finite_time(float(np.asarray(failure.time)), "Failure time")
        self.category = category
        self.error_code = error_code
        self.last_checkpoint_id = checkpoint
        (
            self.finite,
            self.converged,
            self.physically_valid,
            self.qualified,
            self.derivative_valid,
        ) = statuses
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-failure-manifest",
                "production": self.production_id,
                "run": self.run_id,
                "runtime_failure": self.runtime_failure_id,
                "step_index": self.step_index,
                "time": self.time,
                "category": category,
                "error_code": error_code,
                "last_checkpoint": checkpoint,
                "status": list(statuses),
            }
        )


class NumericalRelativityCancellationManifest(StrictModule, NonTrainableState):
    """Bounded cancellation record requiring a durable preserved checkpoint."""

    production_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    step_index: int = eqx.field(static=True)
    time: float = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    preserved_checkpoint_id: str = eqx.field(static=True)
    checkpoint_receipt_id: str = eqx.field(static=True)
    checkpoint_content_digest: str = eqx.field(static=True)
    checkpoint_generation: int = eqx.field(static=True)
    checkpoint_accepted_step: int = eqx.field(static=True)
    checkpoint_commit_id: str = eqx.field(static=True)
    checkpoint_commit_locator: str = eqx.field(static=True)
    checkpoint_durable_size_bytes: int = eqx.field(static=True)
    checkpoint_durable_sha256: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        production: NumericalRelativityProductionPlan,
        prepared: PreparedProductionRun,
        state: ProductionRunState,
        receipt: CheckpointCommitReceipt,
        reason: str,
        /,
    ):
        production._require_prepared(prepared)
        if not isinstance(state, ProductionRunState) or state.status != "cancelled":
            raise ValueError(
                "Cancellation manifests require a cancelled production state."
            )
        if not isinstance(receipt, CheckpointCommitReceipt):
            raise TypeError("Cancellation requires a checkpoint commit receipt.")
        verified = prepared.checkpoint_store.verify_commit(receipt)
        checkpoint = _identifier(
            state.last_checkpoint_id, "Cancellation preserved checkpoint ID"
        )
        step_index = _index(int(np.asarray(state.step_index)), "Cancellation step index")
        if (
            verified.checkpoint_id != checkpoint
            or verified.runtime_id != prepared.run_id
            or verified.accepted_step != step_index
        ):
            raise ValueError(
                "Checkpoint commit receipt does not bind the cancelled runtime state."
            )
        reason_ = _bounded_detail(
            reason,
            "Cancellation reason",
            production.limits.maximum_cancellation_detail_bytes,
        )
        self.production_id = production.production_id
        self.run_id = prepared.run_id
        self.step_index = step_index
        self.time = _finite_time(float(np.asarray(state.time)), "Cancellation time")
        self.reason = reason_
        self.preserved_checkpoint_id = checkpoint
        self.checkpoint_receipt_id = verified.receipt_id

        self.checkpoint_content_digest = verified.content_digest
        self.checkpoint_generation = verified.generation
        self.checkpoint_accepted_step = verified.accepted_step
        self.checkpoint_commit_id = verified.commit_id
        self.checkpoint_commit_locator = verified.commit_locator
        self.checkpoint_durable_size_bytes = verified.durable_size_bytes
        self.checkpoint_durable_sha256 = verified.durable_sha256
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-cancellation-manifest",
                "production": self.production_id,
                "run": self.run_id,
                "step_index": self.step_index,
                "time": self.time,
                "reason": reason_,
                "preserved_checkpoint": checkpoint,
                "checkpoint_receipt": self.checkpoint_receipt_id,
                "checkpoint_content_digest": self.checkpoint_content_digest,
                "checkpoint_generation": self.checkpoint_generation,
                "checkpoint_accepted_step": self.checkpoint_accepted_step,
                "checkpoint_commit_id": self.checkpoint_commit_id,
                "checkpoint_commit_locator": self.checkpoint_commit_locator,
                "checkpoint_durable_size_bytes": self.checkpoint_durable_size_bytes,
                "checkpoint_durable_sha256": self.checkpoint_durable_sha256,
            }
        )


def _require_resolved_execution_resources(
    execution_plan: ExecutionPlan,
    limits: NumericalRelativityProductionLimits,
    /,
) -> None:
    evidence = execution_plan.resource_evidence
    if (
        execution_plan.policy_id != limits.execution_policy.policy_id
        or execution_plan.requirements_id is None
        or execution_plan.inventory_id is None
        or execution_plan.group is None
        or evidence is None
    ):
        raise ValueError(
            "Numerical-relativity production requires a resolver-produced execution plan."
        )
    request = limits.resource_request
    budget_pairs = (
        (
            request.maximum_device_bytes,
            None
            if evidence.per_device_peak_bytes is None
            or evidence.per_device_reserve_bytes is None
            else evidence.per_device_peak_bytes + evidence.per_device_reserve_bytes,
            "device memory",
        ),
        (
            request.maximum_host_bytes,
            None
            if evidence.per_host_peak_bytes is None
            or evidence.per_host_reserve_bytes is None
            else evidence.per_host_peak_bytes + evidence.per_host_reserve_bytes,
            "host memory",
        ),
        (
            request.maximum_compilation_cache_bytes,
            evidence.compilation_cache_bytes,
            "compilation cache",
        ),
        (
            request.maximum_halo_collective_bytes,
            evidence.halo_collective_bytes,
            "halo collective",
        ),
        (
            request.maximum_checkpoint_staging_bytes,
            evidence.checkpoint_staging_bytes,
            "checkpoint staging",
        ),
        (
            request.maximum_output_backlog_bytes,
            evidence.output_backlog_bytes,
            "output backlog",
        ),
    )
    for maximum, observed, name in budget_pairs:
        if maximum is not None and (observed is None or observed > maximum):
            raise ValueError(f"Execution resource evidence violates the {name} request.")
    for required, available, name in (
        (request.required_dtypes, evidence.dtypes, "dtype"),
        (request.required_backends, evidence.backends, "backend"),
        (request.required_collectives, evidence.collectives, "collective"),
    ):
        if not set(required).issubset(available):
            raise ValueError(
                f"Execution resource evidence lacks a required {name} capability."
            )


class NumericalRelativityProductionPlan(StrictModule, NonTrainableState):
    """Compiled exact-domain binding over generic execution and runtime plans."""

    domain: NumericalRelativityDomainBinding
    support_bindings: tuple[NumericalRelativitySupportBinding, ...]
    execution_plan: ExecutionPlan = eqx.field(static=True)
    resolved_run_spec: ResolvedRunSpec
    case_manifest: ProductionCaseManifest
    run_plan: ProductionRunPlan
    checkpoint_policy: CheckpointGenerationPolicy
    limits: NumericalRelativityProductionLimits
    production_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: NumericalRelativityDomainBinding,
        support_bindings: Sequence[NumericalRelativitySupportBinding],
        execution_plan: ExecutionPlan,
        resolved_run_spec: ResolvedRunSpec,
        case_manifest: ProductionCaseManifest,
        run_plan: ProductionRunPlan,
        checkpoint_policy: CheckpointGenerationPolicy,
        limits: NumericalRelativityProductionLimits,
        /,
    ):
        if not isinstance(domain, NumericalRelativityDomainBinding):
            raise TypeError("domain must be NumericalRelativityDomainBinding.")
        bindings = tuple(support_bindings)
        if not bindings or any(
            not isinstance(value, NumericalRelativitySupportBinding) for value in bindings
        ):
            raise TypeError("Production requires typed, non-empty support bindings.")
        bindings = tuple(sorted(bindings, key=lambda value: value.binding_id))
        if len({value.binding_id for value in bindings}) != len(bindings):
            raise ValueError("Production support bindings must be unique.")
        if not isinstance(execution_plan, ExecutionPlan):
            raise TypeError("execution_plan must be ExecutionPlan.")
        if not isinstance(resolved_run_spec, ResolvedRunSpec):
            raise TypeError("resolved_run_spec must be ResolvedRunSpec.")
        if not isinstance(case_manifest, ProductionCaseManifest):
            raise TypeError("case_manifest must be ProductionCaseManifest.")
        if not isinstance(run_plan, ProductionRunPlan):
            raise TypeError("run_plan must be ProductionRunPlan.")
        if not isinstance(checkpoint_policy, CheckpointGenerationPolicy):
            raise TypeError("checkpoint_policy must be CheckpointGenerationPolicy.")
        if not isinstance(limits, NumericalRelativityProductionLimits):
            raise TypeError("limits must be NumericalRelativityProductionLimits.")
        coordinates: dict[str, set[object]] = {
            name: set() for name in _DOMAIN_COORDINATES
        }
        for binding in bindings:
            attributes = dict(binding.support.attributes)
            for name in _DOMAIN_COORDINATES:
                if name in attributes:
                    coordinates[name].add(attributes[name])
        expected = dict(
            zip(
                _DOMAIN_COORDINATES,
                (
                    domain.formulation_id,
                    domain.chart_id,
                    domain.gauge_id,
                    domain.eos_id,
                    domain.topology_id,
                    domain.precision_id,
                ),
                strict=True,
            )
        )
        for name, value in expected.items():
            if coordinates[name] != {value}:
                raise ValueError(
                    f"Support tuples must bind exactly one production {name}: {value!r}."
                )
        scientific = tuple(
            sorted(
                (value.dependency for value in bindings if value.scope == "scientific"),
                key=lambda value: value.dependency_id,
            )
        )
        deployment = tuple(
            sorted(
                (value.dependency for value in bindings if value.scope == "deployment"),
                key=lambda value: value.dependency_id,
            )
        )
        if (
            scientific != resolved_run_spec.scientific_dependencies
            or deployment != resolved_run_spec.deployment_dependencies
        ):
            raise ValueError(
                "Resolved-run dependencies do not exactly match support bindings."
            )
        if resolved_run_spec.profile_ids != tuple(
            sorted(value.profile_id for value in bindings)
        ):
            raise ValueError(
                "Resolved-run profile IDs do not exactly match support bindings."
            )
        if resolved_run_spec.prepared_configuration_id != domain.binding_id:
            raise ValueError("Resolved run does not bind the exact domain configuration.")
        if (
            len(
                {
                    domain.precision_id,
                    execution_plan.precision_policy_id,
                    case_manifest.precision_id,
                }
            )
            != 1
        ):
            raise ValueError("Execution, case, and domain precision identities differ.")
        if resolved_run_spec.precision_policy_id != domain.precision_id:
            raise ValueError("Resolved run does not bind the domain precision policy.")
        if domain.topology_id != case_manifest.topology_id:
            raise ValueError("Production case topology differs from the domain binding.")
        if domain.chart_id != case_manifest.geometry_layout_id:
            raise ValueError(
                "Production case geometry layout differs from the bound chart."
            )
        if case_manifest.backend != execution_plan.backend:
            raise ValueError("Production case and execution-plan backends differ.")
        if case_manifest.method_id != run_plan.method.method_id:
            raise ValueError("Production case and runtime method identities differ.")
        if not isinstance(run_plan.method, FixedGridZ4cProductionMethod):
            raise TypeError(
                "Numerical-relativity production requires a typed supported method."
            )
        runtime = run_plan.method.runtime
        if (
            domain.formulation_id != runtime.system.system_id
            or domain.chart_id != runtime.system.chart_id
            or domain.gauge_id != runtime.gauge.gauge_id
            or domain.topology_id != runtime.grid.grid_id
        ):
            raise ValueError(
                "Fixed-grid Z4c runtime identities differ from the domain binding."
            )
        if execution_plan.solver_policy_id != run_plan.plan_id:
            raise ValueError(
                "Execution plan does not bind the production run-plan identity."
            )
        _require_resolved_execution_resources(execution_plan, limits)
        if resolved_run_spec.resource_policy_id != limits.resource_policy_id:
            raise ValueError("Resolved run does not bind the production resource limits.")
        if resolved_run_spec.output_policy_id != limits.output_policy_id:
            raise ValueError("Resolved run does not bind the production output limits.")
        if resolved_run_spec.checkpoint_policy_id != checkpoint_policy.policy_id:
            raise ValueError(
                "Resolved run does not bind the checkpoint-generation policy."
            )
        if len(domain.input_artifacts) > limits.maximum_input_artifacts:
            raise ValueError("Input artifact count exceeds the configured bound.")
        if (
            sum(value.size_bytes for value in domain.input_artifacts)
            > limits.maximum_input_bytes
        ):
            raise ValueError("Input artifact bytes exceed the configured bound.")
        self.domain = domain
        self.support_bindings = bindings
        self.execution_plan = execution_plan
        self.resolved_run_spec = resolved_run_spec
        self.case_manifest = case_manifest
        self.run_plan = run_plan
        self.checkpoint_policy = checkpoint_policy
        self.limits = limits
        self.production_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-production-plan",
                "domain": domain.binding_id,
                "support_bindings": [value.binding_id for value in bindings],
                "execution_plan": execution_plan.plan_fingerprint,
                "resolved_run": resolved_run_spec.spec_id,
                "case": case_manifest.manifest_id,
                "run_plan": run_plan.plan_id,
                "checkpoint_policy": checkpoint_policy.policy_id,
                "limits": limits.limits_id,
            }
        )

    def prepare(
        self,
        checkpoint_store: DurableCheckpointStore | ArtifactCheckpointStore,
        /,
        **kwargs,
    ) -> PreparedProductionRun:
        if not isinstance(
            checkpoint_store, (DurableCheckpointStore, ArtifactCheckpointStore)
        ):
            raise TypeError("checkpoint_store must be a production checkpoint store.")
        if (
            checkpoint_store.manifest.manifest_id != self.case_manifest.manifest_id
            or checkpoint_store.policy.policy_id != self.checkpoint_policy.policy_id
        ):
            raise ValueError(
                "Checkpoint store does not exactly bind this production plan."
            )
        if isinstance(checkpoint_store, ArtifactCheckpointStore):
            if (
                checkpoint_store.resolved_run_spec.spec_id
                != self.resolved_run_spec.spec_id
                or checkpoint_store.resource_request.resource_id
                != self.limits.resource_request.resource_id
            ):
                raise ValueError(
                    "Artifact checkpoint store binds another resolved run or resource policy."
                )
        return PreparedProductionRun(
            self.case_manifest,
            self.run_plan,
            checkpoint_store,
            resolved_run_spec=self.resolved_run_spec,
            **kwargs,
        )

    def _valid_output_event(
        self, prepared: PreparedProductionRun, event_id: str, /
    ) -> bool:
        self._require_prepared(prepared)
        candidates: set[str] = set()
        if self.run_plan.output_schedule is not None:
            for cursor in range(len(self.run_plan.output_schedule.targets)):
                candidates.add(
                    canonical_fingerprint(
                        {
                            "kind": "scheduled-production-output",
                            "run": prepared.run_id,
                            "schedule": self.run_plan.output_schedule.schedule_id,
                            "cursor": cursor,
                        }
                    )
                )
        for binding in self.run_plan.trigger_bindings:
            if binding.action != "publish":
                continue
            for fire_count in range(self.run_plan.maximum_steps + 1):
                candidates.add(
                    canonical_fingerprint(
                        {
                            "kind": "triggered-production-output",
                            "run": prepared.run_id,
                            "binding": binding.binding_id,
                            "action": binding.action_id,
                            "fire_count": fire_count,
                        }
                    )
                )
        return event_id in candidates

    def prepare_with_output_committer(
        self,
        checkpoint_store: DurableCheckpointStore | ArtifactCheckpointStore,
        writer: Callable[
            [str, ProductionScientificState], Sequence[NeutralBlackHoleArtifact]
        ],
        writer_id: str,
        /,
        *,
        maximum_pending: int = 2,
        **kwargs,
    ) -> tuple[PreparedProductionRun, NumericalRelativityOutputCommitter]:
        if "publisher" in kwargs:
            raise ValueError("Output committer preparation owns the publisher.")
        maximum_pending_ = _positive_integer(maximum_pending, "maximum_pending")
        backlog = self.limits.resource_request.maximum_output_backlog_bytes
        if backlog is None:
            raise ValueError("Output backlog bound is unavailable.")
        committer = NumericalRelativityOutputCommitter(self, writer, writer_id)
        publisher = ByteBoundedAsyncPublisher(
            committer,
            maximum_pending=maximum_pending_,
            maximum_pending_bytes=backlog,
        )
        prepared = self.prepare(
            checkpoint_store,
            publisher=publisher,
            **kwargs,
        )
        committer._bind(prepared, publisher)
        return prepared, committer

    def _require_prepared(self, prepared: PreparedProductionRun, /) -> None:
        if not isinstance(prepared, PreparedProductionRun):
            raise TypeError("prepared must be PreparedProductionRun.")
        if (
            prepared.manifest.manifest_id != self.case_manifest.manifest_id
            or prepared.plan.plan_id != self.run_plan.plan_id
            or prepared.checkpoint_store.policy.policy_id
            != self.checkpoint_policy.policy_id
            or prepared.resolved_run_spec is None
            or prepared.resolved_run_spec.spec_id != self.resolved_run_spec.spec_id
        ):
            raise ValueError(
                "Prepared runtime does not exactly bind this production plan."
            )

    def restart_manifest(
        self, prepared: PreparedProductionRun, state: ProductionRunState, /
    ) -> NumericalRelativityRestartManifest:
        self._require_prepared(prepared)
        if not isinstance(state, ProductionRunState):
            raise TypeError("state must be ProductionRunState.")
        return NumericalRelativityRestartManifest(
            self, prepared, _identifier(state.last_checkpoint_id, "Restart checkpoint ID")
        )

    def admit_restart(
        self,
        prepared: PreparedProductionRun,
        restart: NumericalRelativityRestartManifest,
        /,
    ) -> None:
        self._require_prepared(prepared)
        if not isinstance(restart, NumericalRelativityRestartManifest):
            raise TypeError("restart must be NumericalRelativityRestartManifest.")
        expected = (
            self.production_id,
            prepared.run_id,
            self.domain.binding_id,
            self.case_manifest.manifest_id,
            self.execution_plan.execution_plan_id,
            self.execution_plan.plan_fingerprint,
            self.resolved_run_spec.spec_id,
            self.run_plan.plan_id,
            tuple(value.artifact_id for value in self.domain.input_artifacts),
        )
        actual = (
            restart.production_id,
            restart.run_id,
            restart.domain_binding_id,
            restart.case_manifest_id,
            restart.execution_plan_id,
            restart.execution_plan_fingerprint,
            restart.resolved_run_spec_id,
            restart.run_plan_id,
            restart.input_artifact_ids,
        )
        if actual != expected:
            raise ValueError(
                "Restart identity is incompatible with this production plan."
            )

    def resume(
        self,
        prepared: PreparedProductionRun,
        template: ProductionRunState,
        restart: NumericalRelativityRestartManifest,
        /,
    ) -> ProductionRunState:
        self.admit_restart(prepared, restart)
        restored = prepared.resume(template)
        if restored.last_checkpoint_id != restart.checkpoint_id:
            raise ValueError(
                "Restored checkpoint differs from the admitted restart identity."
            )
        return restored

    def output_manifest(
        self,
        prepared: PreparedProductionRun,
        committer: NumericalRelativityOutputCommitter,
        receipt: NumericalRelativityCommittedOutputReceipt,
        result: ProductionRunResult,
        /,
    ) -> NumericalRelativityOutputManifest:
        return NumericalRelativityOutputManifest(
            self, prepared, committer, receipt, result
        )

    def failure_manifest(
        self,
        prepared: PreparedProductionRun,
        result: ProductionRunResult,
        /,
    ) -> NumericalRelativityFailureManifest:
        self._require_prepared(prepared)
        if (
            not isinstance(result, ProductionRunResult)
            or result.run_id != prepared.run_id
            or result.state.status != "failed"
            or result.failure is None
        ):
            raise ValueError("Failure manifest requires this runtime's failed result.")
        accepted = result.state.accepted_state
        if not isinstance(accepted, (Z4cProductionState, GRRMHDProductionState)):
            raise ValueError("Failure result has no accepted scientific state.")
        accepted_step, accepted_time = _scientific_state_step_time(accepted)
        if (
            int(np.asarray(result.failure.step_index))
            != int(np.asarray(result.state.step_index))
            or float(np.asarray(result.failure.time))
            != float(np.asarray(result.state.time))
            or int(np.asarray(accepted_step)) != int(np.asarray(result.state.step_index))
            or float(np.asarray(accepted_time)) != float(np.asarray(result.state.time))
        ):
            raise ValueError("Failure evidence is outside the authoritative run lineage.")
        return NumericalRelativityFailureManifest(
            self,
            prepared,
            result.failure,
            accepted,
            terminal_checkpoint_id=result.state.last_checkpoint_id,
        )

    def cancellation_manifest(
        self,
        prepared: PreparedProductionRun,
        result: ProductionRunResult,
        reason: str,
        /,
    ) -> NumericalRelativityCancellationManifest:
        self._require_prepared(prepared)
        if (
            not isinstance(result, ProductionRunResult)
            or result.run_id != prepared.run_id
            or result.state.status != "cancelled"
            or result.failure is not None
        ):
            raise ValueError(
                "Cancellation manifest requires this runtime's cancelled result."
            )
        preserved, receipt = prepared.commit_checkpoint(result.state)
        verified = prepared.checkpoint_store.verify_commit(receipt)
        return NumericalRelativityCancellationManifest(
            self, prepared, preserved, verified, reason
        )


def compile_numerical_relativity_production(
    domain: NumericalRelativityDomainBinding,
    support_bindings: Sequence[NumericalRelativitySupportBinding],
    execution_plan: ExecutionPlan,
    resolved_run_spec: ResolvedRunSpec,
    case_manifest: ProductionCaseManifest,
    run_plan: ProductionRunPlan,
    checkpoint_policy: CheckpointGenerationPolicy,
    limits: NumericalRelativityProductionLimits,
    /,
) -> NumericalRelativityProductionPlan:
    """Compile explicit domain support into existing execution/runtime structures."""

    return NumericalRelativityProductionPlan(
        domain,
        support_bindings,
        execution_plan,
        resolved_run_spec,
        case_manifest,
        run_plan,
        checkpoint_policy,
        limits,
    )


__all__ = [
    "FailureCategory",
    "FixedGridZ4cProductionMethod",
    "NumericalRelativityArtifactBinding",
    "NumericalRelativityCommittedOutputReceipt",
    "NumericalRelativityCancellationManifest",
    "NumericalRelativityDomainBinding",
    "NumericalRelativityFailureManifest",
    "NumericalRelativityOutputManifest",
    "NumericalRelativityOutputCommitter",
    "NumericalRelativityProductionLimits",
    "NumericalRelativityProductionPlan",
    "NumericalRelativityRestartManifest",
    "NumericalRelativitySupportBinding",
    "SupportScope",
    "FixedGridGRRMHDProductionMethod",
    "GRRMHDProductionArguments",
    "GRRMHDProductionState",
    "ProductionScientificState",
    "Z4cProductionState",
    "compile_numerical_relativity_production",
]
