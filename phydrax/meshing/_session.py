#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from types import TracebackType
from typing import Literal, Self

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import parse
from ._contracts import MeshingExecutionMode


class MeshingExecutionPolicy(StrictModule, NonTrainableState):
    """Declare one provider execution policy and its requested thread count.

    Determinism is a provider admission requirement, including for parallel
    execution. Providers that cannot guarantee the requested pair refuse it.
    """

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
    ) -> None:
        mode = parse(
            execution_mode, MeshingExecutionMode, "MeshingExecutionPolicy.execution_mode"
        )
        timeout = float(timeout_seconds)
        transferred = int(maximum_transferred_bytes)
        workers = int(parallelism)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError(
                f"MeshingExecutionPolicy.timeout_seconds must be positive and finite; got {timeout}."
            )
        if transferred <= 0 or workers <= 0:
            raise ValueError(
                f"MeshingExecutionPolicy.limits must be positive; got transferred_bytes={transferred}, parallelism={workers}."
            )
        self.execution_mode = mode
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
            raise RuntimeError(
                "AbstractMeshingSession.closed must be False on entry; got a closed session."
            )
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False


__all__ = ["AbstractMeshingSession", "MeshingExecutionPolicy"]
