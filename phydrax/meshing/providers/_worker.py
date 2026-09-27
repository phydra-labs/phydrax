#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Persistent native provider workers and their meshing failure semantics."""

from __future__ import annotations

import os
import shutil
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..._external_runtime import (
    NativeWorker,
    NativeWorkerCall,
    NativeWorkerError,
    NativeWorkerIdentity,
    NativeWorkerPolicy,
)
from ...logging import emit
from .._contracts import MeshingFailure, MeshingFailureCategory, MeshingLimits


_REJECTION_CATEGORIES = {
    "invalid_request": MeshingFailureCategory.INVALID_SPECIFICATION,
    "unsupported": MeshingFailureCategory.UNSUPPORTED_CAPABILITY,
    "resource_exhausted": MeshingFailureCategory.RESOURCE_EXHAUSTED,
    "library_failure": MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
}


def _category(error: NativeWorkerError, /) -> MeshingFailureCategory:
    match error.kind:
        case "unavailable":
            return MeshingFailureCategory.PROVIDER_UNAVAILABLE
        case "timeout":
            return MeshingFailureCategory.TIMED_OUT
        case "resource":
            return MeshingFailureCategory.RESOURCE_EXHAUSTED
        case "startup" | "exited" | "protocol":
            return MeshingFailureCategory.PROVIDER_EXECUTION_FAILED
        case "rejected":
            return _REJECTION_CATEGORIES[error.evidence["worker_kind"]]
        case _:
            raise ValueError(f"Unknown native worker failure kind {error.kind!r}.")


def _failure(provider: str, stage: str, error: NativeWorkerError, /) -> MeshingFailure:
    """Translate a worker failure; its evidence stays on ``__cause__``."""
    code = error.evidence.get("worker_kind", error.kind)
    returncode = error.evidence.get("returncode")
    log = str(error.evidence.get("log", "")).strip()
    message = f"{provider} {stage} failed ({code}): {error}"
    if returncode is not None:
        message += f" [exit status {returncode}]"
    if log:
        message += f"\n{log[-4000:]}"
    emit(
        "ERROR",
        "provider.worker.failed",
        "Native provider worker failed",
        provider=provider,
        stage=stage,
        failure_category=code,
        return_code=returncode,
    )
    return MeshingFailure(_category(error), message, provider_code=code, stage=stage)


class ProviderWorker:
    """One lazily launched native worker session reused by a provider.

    The executable is the explicit path, else ``environment_variable``, else
    ``default_executable`` on PATH. Runtime identity is probed once per session;
    a failed, closed, or call-exhausted session is replaced by a fresh launch at
    the next call and ``launches`` counts every session ever started. One
    reentrant lock serializes session creation, replacement, calls, and close
    across threads; it is always taken before the session's own lock.
    """

    def __init__(
        self,
        provider: str,
        /,
        *,
        executable: str | os.PathLike[str] | None,
        environment_variable: str,
        default_executable: str,
        build_hint: str,
        launcher: Sequence[str] = (),
        arguments: Sequence[str] = (),
        policy: NativeWorkerPolicy | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.provider = str(provider)
        self.executable = None if executable is None else str(executable)
        self.environment_variable = str(environment_variable)
        self.default_executable = str(default_executable)
        self.build_hint = str(build_hint)
        self.launcher = tuple(str(value) for value in launcher)
        self.arguments = tuple(str(value) for value in arguments)
        self.policy = NativeWorkerPolicy() if policy is None else policy
        if not isinstance(self.policy, NativeWorkerPolicy):
            raise TypeError("policy must be NativeWorkerPolicy or None.")
        self.environment = dict(environment or {})
        self.launches = 0
        self._lock = threading.RLock()
        self._worker: NativeWorker | None = None

    def _resolve(self) -> str:
        requested = self.executable or os.environ.get(
            self.environment_variable, self.default_executable
        )
        located = shutil.which(requested)
        if located is None:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_UNAVAILABLE,
                f"Cannot locate the {self.provider} worker {requested!r}. {self.build_hint}",
                stage="startup",
            )
        return located

    def session(self) -> NativeWorker:
        with self._lock:
            worker = self._worker
            if worker is not None and not worker.closed and not worker.exhausted:
                return worker
            if worker is not None:
                worker.close()
            executable = self._resolve()
            self.launches += 1
            try:
                self._worker = NativeWorker(
                    executable,
                    launcher=self.launcher,
                    arguments=self.arguments,
                    policy=self.policy,
                    environment=self.environment,
                )
            except NativeWorkerError as error:
                raise _failure(self.provider, "startup", error) from error
            return self._worker

    @property
    def identity(self) -> NativeWorkerIdentity:
        return self.session().identity

    def call(
        self,
        operation: str,
        parameters: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
        /,
        *,
        limits: MeshingLimits,
    ) -> NativeWorkerCall:
        if not isinstance(limits, MeshingLimits):
            raise TypeError("limits must be MeshingLimits.")
        # Holding the provider lock across the call keeps a concurrent caller from
        # replacing or closing this session between its selection and its use.
        with self._lock:
            worker = self.session()
            try:
                return worker.call(
                    operation,
                    parameters,
                    arrays,
                    timeout=limits.maximum_wall_seconds,
                    maximum_input_bytes=limits.maximum_data_bytes,
                    maximum_output_bytes=limits.maximum_data_bytes,
                )
            except NativeWorkerError as error:
                raise _failure(self.provider, operation, error) from error

    def memory_limit_evidence(self) -> str:
        """Enforced-limit label describing how worker memory is bounded."""
        return (
            "worker_address_space"
            if self.identity.memory_enforcement == "address-space-rlimit"
            else "worker_peak_resident_audit"
        )

    def close(self) -> None:
        with self._lock:
            if self._worker is not None:
                self._worker.close()
                self._worker = None


__all__ = ["ProviderWorker"]
