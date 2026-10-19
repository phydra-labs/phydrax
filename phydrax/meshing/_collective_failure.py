#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Collective failure admission before publishing distributed mesh state."""

from __future__ import annotations

import equinox as eqx
import numpy as np
from jax.experimental import multihost_utils

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
