#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared optimization iteration-evidence adapters."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

import equinox as eqx
import jax.numpy as jnp

from .._iteration import (
    bind_iteration_scope,
    finalize_iteration,
    initialize_iteration,
    IterationCapabilities,
    IterationCoordinates,
    IterationPhase,
    IterationPlan,
    IterationRecord,
)


class _OptimizationResultProtocol(Protocol):
    status: Any
    diagnostics: Any
    iteration_evidence: Any

    @property
    def successful(self) -> Any: ...


_OptimizationResultT = TypeVar("_OptimizationResultT", bound=_OptimizationResultProtocol)


def attach_terminal_optimization_iteration(
    result: _OptimizationResultT,
    iteration: IterationPlan | None,
    algorithm_id: str,
    /,
) -> _OptimizationResultT:
    """Attach honest terminal-only evidence to an opaque optimization result."""
    if iteration is None or result.iteration_evidence is not None:
        return result
    if not isinstance(iteration, IterationPlan):
        raise TypeError("iteration must be IterationPlan or None.")
    capabilities = IterationCapabilities.terminal_only()
    scope = bind_iteration_scope(iteration, capabilities, algorithm_id)
    diagnostics = result.diagnostics
    active = jnp.ones_like(result.status, dtype=bool)
    initial = IterationRecord(
        IterationCoordinates(IterationPhase.START, 0, active=active),
        result.status,
        diagnostics,
    )
    terminal = IterationRecord(
        IterationCoordinates(
            IterationPhase.TERMINAL,
            diagnostics.iterations,
            attempt=diagnostics.iterations,
            accepted=diagnostics.accepted_steps,
            rejected=diagnostics.rejected_steps,
            active=active,
            committed=result.successful,
            terminal=True,
        ),
        result.status,
        diagnostics,
    )
    evidence = finalize_iteration(
        iteration,
        scope,
        capabilities,
        initialize_iteration(iteration, initial),
        terminal,
    )
    return eqx.tree_at(
        lambda value: value.iteration_evidence,
        result,
        evidence,
        is_leaf=lambda value: value is None,
    )


__all__ = ["attach_terminal_optimization_iteration"]
