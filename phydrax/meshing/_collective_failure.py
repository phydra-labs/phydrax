#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Collective failure admission before publishing distributed mesh state."""

from __future__ import annotations
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import cast, TypeVar


import equinox as eqx
import numpy as np
from jax.experimental import multihost_utils

from .._meshcore import MeshcoreError, MeshcoreStatus
from ..discretization._coordinate_budget import CoordinateEnclosureResourceError
from .._strict import StrictModule
from ..typing import checked
from ._failures import MeshingFailure, MeshingFailureCategory


class CollectiveFailureDecision(StrictModule):
    """Process category codes and the minimum-priority rejected category."""

    code: int = eqx.field(static=True)
    rejected: np.ndarray
    category: MeshingFailureCategory | None = eqx.field(static=True)


def collective_failure_decision(
    category: MeshingFailureCategory | None, /
) -> CollectiveFailureDecision:
    """Gather category ordinals once while retaining all process verdicts."""
    categories = tuple(MeshingFailureCategory)
    code = 0 if category is None else categories.index(category) + 1
    rejected = np.asarray(multihost_utils.process_allgather(np.int32(code), tiled=False))
    failures = rejected[rejected > 0]
    selected = None if failures.size == 0 else categories[int(np.min(failures)) - 1]
    return CollectiveFailureDecision(code, rejected, selected)


@checked
def collective_failure_barrier(
    failure: MeshingFailure | None,
    /,
    *,
    message: str,
    stage: str,
) -> None:
    """Raise the minimum-priority category reported by any process.

    A matching local failure is raised unchanged, including its cause. Processes
    without that category receive the caller's remote refusal message and stage.
    """
    decision = collective_failure_decision(None if failure is None else failure.category)
    category = decision.category
    if category is not None:
        if failure is not None and failure.category is category:
            raise failure
        raise MeshingFailure(category, message, stage=stage)


_Value = TypeVar("_Value")


def agreed_local(
    action: Callable[[], _Value],
    /,
    *,
    stage: str = "distributed-device-commit",
) -> _Value:
    """Agree strictly local work before any process enters the next collective."""
    value: _Value | None = None
    failure: MeshingFailure | None = None
    try:
        value = action()
    except MeshingFailure as error:
        failure = error
    except (MeshcoreError, CoordinateEnclosureResourceError) as error:
        category = MeshingFailureCategory.RESOURCE_EXHAUSTED
        if isinstance(error, MeshcoreError) and error.status not in (
            MeshcoreStatus.CAPACITY_EXCEEDED,
            MeshcoreStatus.TIMEOUT,
        ):
            category = MeshingFailureCategory.PROVIDER_EXECUTION_FAILED
        failure = MeshingFailure(category, str(error), stage=stage)
        failure.__cause__ = error
    except ValueError as error:
        failure = MeshingFailure(
            MeshingFailureCategory.LINEAGE_FAILED, str(error), stage=stage
        )
        failure.__cause__ = error
    except Exception as error:  # noqa: BLE001 - refused collectively, cause retained
        failure = MeshingFailure(
            MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
            f"{type(error).__name__}: {error}",
            stage=stage,
        )
        failure.__cause__ = error
    collective_failure_barrier(
        failure,
        message="An owning process rejected the exact topology, lineage, or geometry "
        "resolution of the distributed epoch; no part was published.",
        stage=stage,
    )
    return cast(_Value, value)


@contextmanager
def agreed_local_step(
    *, stage: str = "distributed-device-commit"
) -> Iterator[None]:
    """Agree a strictly local block; never enclose a routed lookup or fingerprint."""
    failure: Exception | None = None
    try:
        yield
    except Exception as error:  # noqa: BLE001 - classified by agreed_local
        failure = error

    def finish() -> None:
        if failure is not None:
            raise failure

    agreed_local(finish, stage=stage)
