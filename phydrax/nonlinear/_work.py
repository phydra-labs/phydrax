#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule


_WORK_FIELDS = (
    "residual_evaluations",
    "validity_evaluations",
    "jvp_evaluations",
    "vjp_evaluations",
    "jacobian_preparations",
    "linear_setups",
    "linear_refreshes",
    "linear_solves",
    "linear_iterations",
    "preconditioner_applications",
    "local_updates",
)


def _count(value: Any, /, *, name: str) -> Array:
    array = jnp.asarray(value, dtype=jnp.int32)
    if array.ndim != 0:
        raise ValueError(f"{name} must be scalar.")
    return eqx.error_if(array, array < 0, f"{name} must be non-negative.")


class NonlinearWork(StrictModule):
    """Exact associative work evidence for nonlinear execution."""

    residual_evaluations: Array
    validity_evaluations: Array
    jvp_evaluations: Array
    vjp_evaluations: Array
    jacobian_preparations: Array
    linear_setups: Array
    linear_refreshes: Array
    linear_solves: Array
    linear_iterations: Array
    preconditioner_applications: Array
    local_updates: Array
    complete: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        residual_evaluations: Any = 0,
        validity_evaluations: Any = 0,
        jvp_evaluations: Any = 0,
        vjp_evaluations: Any = 0,
        jacobian_preparations: Any = 0,
        linear_setups: Any = 0,
        linear_refreshes: Any = 0,
        linear_solves: Any = 0,
        linear_iterations: Any = 0,
        preconditioner_applications: Any = 0,
        local_updates: Any = 0,
        complete: bool = True,
    ):
        values = {
            "residual_evaluations": residual_evaluations,
            "validity_evaluations": validity_evaluations,
            "jvp_evaluations": jvp_evaluations,
            "vjp_evaluations": vjp_evaluations,
            "jacobian_preparations": jacobian_preparations,
            "linear_setups": linear_setups,
            "linear_refreshes": linear_refreshes,
            "linear_solves": linear_solves,
            "linear_iterations": linear_iterations,
            "preconditioner_applications": preconditioner_applications,
            "local_updates": local_updates,
        }
        for name in _WORK_FIELDS:
            setattr(self, name, _count(values[name], name=name))
        self.complete = bool(complete)

    @classmethod
    def zero(cls, /, *, complete: bool = True) -> NonlinearWork:
        return cls(complete=complete)

    def __add__(self, other: NonlinearWork, /) -> NonlinearWork:
        if not isinstance(other, NonlinearWork):
            return NotImplemented
        values = {name: vars(self)[name] + vars(other)[name] for name in _WORK_FIELDS}
        return NonlinearWork(**values, complete=self.complete and other.complete)

    def scaled(self, count: int, /) -> NonlinearWork:
        count_ = int(count)
        if count_ < 0:
            raise ValueError("Work scale must be non-negative.")
        values = {name: vars(self)[name] * count_ for name in _WORK_FIELDS}
        return NonlinearWork(**values, complete=self.complete)


class NonlinearWorkBudget(StrictModule):
    """Traced remaining work; negative limits mean structurally unlimited."""

    residual_evaluations: Array
    validity_evaluations: Array
    jvp_evaluations: Array
    vjp_evaluations: Array
    jacobian_preparations: Array
    linear_setups: Array
    linear_refreshes: Array
    linear_solves: Array
    linear_iterations: Array
    preconditioner_applications: Array
    local_updates: Array

    def __init__(
        self,
        *,
        residual_evaluations: Any = -1,
        validity_evaluations: Any = -1,
        jvp_evaluations: Any = -1,
        vjp_evaluations: Any = -1,
        jacobian_preparations: Any = -1,
        linear_setups: Any = -1,
        linear_refreshes: Any = -1,
        linear_solves: Any = -1,
        linear_iterations: Any = -1,
        preconditioner_applications: Any = -1,
        local_updates: Any = -1,
    ):
        values = {
            "residual_evaluations": residual_evaluations,
            "validity_evaluations": validity_evaluations,
            "jvp_evaluations": jvp_evaluations,
            "vjp_evaluations": vjp_evaluations,
            "jacobian_preparations": jacobian_preparations,
            "linear_setups": linear_setups,
            "linear_refreshes": linear_refreshes,
            "linear_solves": linear_solves,
            "linear_iterations": linear_iterations,
            "preconditioner_applications": preconditioner_applications,
            "local_updates": local_updates,
        }
        for name in _WORK_FIELDS:
            array = jnp.asarray(values[name], dtype=jnp.int32)
            if array.ndim != 0:
                raise ValueError(f"{name} budget must be scalar.")
            setattr(self, name, array)

    @classmethod
    def unlimited(cls) -> NonlinearWorkBudget:
        return cls()

    def permits(self, work: NonlinearWork, /) -> Array:
        if not isinstance(work, NonlinearWork):
            raise TypeError("work must be NonlinearWork.")
        permitted = jnp.asarray(True)
        for name in _WORK_FIELDS:
            limit = vars(self)[name]
            need = vars(work)[name]
            permitted = permitted & ((limit < 0) | (need <= limit))
        if not work.complete:
            has_limit = jnp.asarray(False)
            for name in _WORK_FIELDS:
                has_limit = has_limit | (vars(self)[name] >= 0)
            permitted = permitted & ~has_limit
        return permitted

    def consume(self, work: NonlinearWork, /) -> NonlinearWorkBudget:
        if not isinstance(work, NonlinearWork):
            raise TypeError("work must be NonlinearWork.")
        values = {}
        for name in _WORK_FIELDS:
            limit = vars(self)[name]
            need = vars(work)[name]
            values[name] = jnp.where(limit < 0, limit, jnp.maximum(limit - need, 0))
        return NonlinearWorkBudget(**values)

    def split(
        self, count: int, /, *, reserve: NonlinearWork | None = None
    ) -> NonlinearWorkBudget:
        count_ = int(count)
        if count_ < 1:
            raise ValueError("Budget split count must be positive.")
        reserve_ = NonlinearWork.zero() if reserve is None else reserve
        if not isinstance(reserve_, NonlinearWork):
            raise TypeError("reserve must be NonlinearWork or None.")
        values = {}
        for name in _WORK_FIELDS:
            limit = vars(self)[name]
            kept = vars(reserve_)[name]
            values[name] = jnp.where(
                limit < 0,
                limit,
                jnp.maximum(limit - kept, 0) // count_,
            )
        return NonlinearWorkBudget(**values)


class NonlinearAttemptEvidence(StrictModule):
    """Fixed-topology evidence for one solver/update attempt."""

    status: Array
    accepted: Array
    skipped: Array
    input_residual_norm: Array
    output_residual_norm: Array
    work: NonlinearWork
    component_id: str = eqx.field(static=True)
    failure_origin: str = eqx.field(static=True)
    children: tuple[NonlinearAttemptEvidence, ...]

    def __init__(
        self,
        *,
        component_id: str,
        status: Any,
        accepted: Any,
        input_residual_norm: Any,
        output_residual_norm: Any,
        work: NonlinearWork,
        skipped: Any = False,
        failure_origin: str = "",
        children: tuple[NonlinearAttemptEvidence, ...] = (),
    ):
        identifier = str(component_id)
        if not identifier:
            raise ValueError("component_id must be non-empty.")
        if not isinstance(work, NonlinearWork):
            raise TypeError("work must be NonlinearWork.")
        children_ = tuple(children)
        if not all(isinstance(child, NonlinearAttemptEvidence) for child in children_):
            raise TypeError("children must contain NonlinearAttemptEvidence values.")
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.accepted = jnp.asarray(accepted, dtype=jnp.bool_)
        self.skipped = jnp.asarray(skipped, dtype=jnp.bool_)
        self.input_residual_norm = jnp.asarray(input_residual_norm)
        self.output_residual_norm = jnp.asarray(output_residual_norm)
        self.work = work
        self.component_id = identifier
        self.failure_origin = str(failure_origin)
        self.children = children_

    @classmethod
    def skipped_evidence(
        cls,
        component_id: str,
        /,
        *,
        status: int,
        children: tuple[NonlinearAttemptEvidence, ...] = (),
    ) -> NonlinearAttemptEvidence:
        nan = jnp.asarray(jnp.nan)
        return cls(
            component_id=component_id,
            status=status,
            accepted=False,
            skipped=True,
            input_residual_norm=nan,
            output_residual_norm=nan,
            work=NonlinearWork.zero(),
            failure_origin="update-status",
            children=children,
        )


def work_sum(values: tuple[NonlinearWork, ...], /) -> NonlinearWork:
    if not values:
        return NonlinearWork.zero()
    return sum(values[1:], values[0])


__all__ = [
    "NonlinearAttemptEvidence",
    "NonlinearWork",
    "NonlinearWorkBudget",
    "work_sum",
]
