#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Self

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._contracts import MeshingExecutionMode


class MeshingExecutionPolicy(StrictModule, NonTrainableState):
    execution_mode: MeshingExecutionMode = eqx.field(static=True)
    timeout_seconds: float = eqx.field(static=True)
    maximum_transferred_bytes: int = eqx.field(static=True)
    parallelism: int = eqx.field(static=True)
    deterministic: bool = eqx.field(static=True)
    cleanup_workspace: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        execution_mode: MeshingExecutionMode = MeshingExecutionMode.IN_PROCESS,
        /,
        *,
        timeout_seconds: float = 3600.0,
        maximum_transferred_bytes: int = 4_000_000_000,
        parallelism: int = 1,
        deterministic: bool = True,
        cleanup_workspace: bool = True,
    ):
        if not isinstance(execution_mode, MeshingExecutionMode):
            raise TypeError("execution_mode must be MeshingExecutionMode.")
        timeout = float(timeout_seconds)
        transferred = int(maximum_transferred_bytes)
        workers = int(parallelism)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite.")
        if transferred <= 0 or workers <= 0:
            raise ValueError("Transfer and parallelism limits must be positive.")
        if deterministic and workers != 1:
            raise ValueError("Deterministic meshing requires parallelism=1.")
        self.execution_mode = execution_mode
        self.timeout_seconds = timeout
        self.maximum_transferred_bytes = transferred
        self.parallelism = workers
        self.deterministic = bool(deterministic)
        self.cleanup_workspace = bool(cleanup_workspace)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "meshing-execution-policy",
                "execution_mode": execution_mode.value,
                "timeout_seconds": timeout,
                "maximum_transferred_bytes": transferred,
                "parallelism": workers,
                "deterministic": bool(deterministic),
                "cleanup_workspace": bool(cleanup_workspace),
            }
        )


class AbstractMeshingSession(abc.ABC):
    """Ephemeral host-side provider session excluded from JAX state."""

    @property
    @abc.abstractmethod
    def closed(self) -> bool: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("Cannot enter a closed meshing session.")
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self.close()
        return False


__all__ = ["AbstractMeshingSession", "MeshingExecutionPolicy"]
