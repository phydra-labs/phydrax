#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Thread-affine host sessions for deterministic native meshing execution."""

from __future__ import annotations

import threading
from typing import final, Self

from jax import Array

import phydrax.geometry.design._schema as schema_owner
from phydrax.execution import ResourceRequest

from ..._meshcore import MeshcoreError, MeshcoreStatus, NativeParallelExecution
from ...typing import checked
from .._contracts import MeshingExecutionMode
from .._failures import CutStageRecord, MeshingFailure, MeshingFailureCategory
from .._measurements import NativeMeshingPhaseRecorder
from .._result import CellMeshingResult
from .._session import AbstractMeshingSession, MeshingExecutionPolicy
from .._trace import MeshingStageKind
from ._native import NativeMeshingPlan


@final
class NativeMeshingSession(AbstractMeshingSession):
    """Execute prepared native plans under one admitted thread policy.

    A session belongs to its creating thread and has one context lifecycle.
    Multiple plans may execute before it closes; each execution installs a fresh
    native policy context and restores it on success or failure. An outer
    resource request supplies a CPU admission ceiling, not a thread request.
    """

    @checked
    def __init__(
        self,
        policy: MeshingExecutionPolicy,
        /,
        *,
        resources: ResourceRequest | None = None,
    ) -> None:
        if policy.execution_mode is not MeshingExecutionMode.IN_PROCESS:
            raise MeshingFailure(
                MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
                "NativeMeshingSession.policy.execution_mode must be in_process; "
                f"got {policy.execution_mode.value!r}",
                provider_code="execution_mode_unavailable",
            )
        if resources is not None and policy.parallelism > resources.cpu_cores:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "NativeMeshingSession.policy.parallelism must fit resources.cpu_cores; "
                f"got parallelism={policy.parallelism}, cpu_cores={resources.cpu_cores}",
                provider_code="cpu_ceiling_exceeded",
                requested=(("parallelism", policy.parallelism),),
                achieved=(("cpu_cores", resources.cpu_cores),),
            )
        self._policy = policy
        self._thread = threading.get_ident()
        self._closed = False
        self._entered = False
        self._executing = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_thread(self) -> None:
        caller = threading.get_ident()
        if caller != self._thread:
            raise RuntimeError(
                "NativeMeshingSession.thread must be the creating thread; "
                f"got owner_thread={self._thread}, caller_thread={caller}"
            )

    def __enter__(self) -> Self:
        self._require_thread()
        if self.closed:
            raise RuntimeError(
                "NativeMeshingSession.closed must be False on entry; got a closed session"
            )
        if self._entered:
            raise RuntimeError(
                "NativeMeshingSession.context must be single-use; got repeated entry"
            )
        self._entered = True
        return self

    @checked
    def execute(
        self,
        plan: NativeMeshingPlan,
        state: schema_owner.DesignState | None = None,
        /,
        *,
        record_phase: NativeMeshingPhaseRecorder | None = None,
    ) -> CellMeshingResult:
        """Return a plan's audited result after restoring its native thread policy."""
        self._require_thread()
        if self.closed:
            raise RuntimeError(
                "NativeMeshingSession.closed must be False for execution; got a closed session"
            )
        if self._executing:
            raise RuntimeError(
                "NativeMeshingSession.execute must be non-reentrant; got an active execution"
            )
        self._executing = True
        result: CellMeshingResult | None = None
        try:
            with NativeParallelExecution(
                self._policy.parallelism, self._policy.timeout_seconds
            ):
                result = plan._execute_in_session(state, record_phase=record_phase)
                return result
        except MeshcoreError as error:
            if error.status is not MeshcoreStatus.TIMEOUT:
                raise
            raise self._timeout_failure(error, result) from error
        finally:
            self._executing = False

    def _timeout_failure(
        self, error: MeshcoreError, result: CellMeshingResult | None, /
    ) -> MeshingFailure:
        """Retain ended scope evidence when the policy expires before publication."""
        stage = ""
        achieved: tuple[tuple[str, float], ...] = ()
        findings: tuple[tuple[str, Array], ...] = ()
        stages: tuple[CutStageRecord, ...] = ()
        if result is not None and result.execution_evidence is not None:
            record = result.execution_evidence
            ended = record.native_evidence()
            stage = MeshingStageKind.CANONICALIZATION.value
            stages = (CutStageRecord(stage, None, ended, str(error), None),)
            achieved = (
                ("native_scope:elapsed_seconds", ended.elapsed_seconds),
                (
                    "native_scope:geometry_queries",
                    float(ended.work_counters.source_queries),
                ),
                ("native_scope:work_units", float(ended.work_counters.construction_work)),
            )
            findings = (
                (
                    "native_execution.source_preparation_geometry_queries",
                    record.source_preparation_geometry_queries,
                ),
                (
                    "native_execution.source_preparation_work_units",
                    record.source_preparation_work_units,
                ),
            )
        return MeshingFailure(
            MeshingFailureCategory.TIMED_OUT,
            "NativeMeshingSession.policy.timeout_seconds must bound execution and "
            f"the pool lease; got an expired {self._policy.timeout_seconds}-second policy",
            provider_code=MeshcoreStatus.TIMEOUT.name.lower(),
            stage=stage,
            requested=(("timeout_seconds", self._policy.timeout_seconds),),
            achieved=achieved,
            logical_findings=findings,
            cut_stage_records=stages,
        )

    def close(self) -> None:
        """Close the session on its creating thread without suppressing failures."""
        self._require_thread()
        if self._executing:
            raise RuntimeError(
                "NativeMeshingSession.close must follow execution; got an active execution"
            )
        self._closed = True


__all__ = ["NativeMeshingSession"]
