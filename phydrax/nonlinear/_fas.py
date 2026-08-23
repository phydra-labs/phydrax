#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._tree_math import tree_add_scaled, tree_allfinite
from ..linalg import AbstractVectorSpace
from ._types import NonlinearProvenance, NonlinearStatus
from ._updates import (
    AbstractNonlinearUpdate,
    NonlinearUpdateCapabilities,
    NonlinearUpdateControl,
    NonlinearUpdateDiagnostics,
    NonlinearUpdateProvenance,
    NonlinearUpdateResult,
    NonlinearUpdateStatus,
    PreparedNonlinearUpdate,
)
from ._work import NonlinearWork


FASCycleKind: TypeAlias = Literal["v", "w", "f"]


class FASLevel(StrictModule):
    """One nonlinear FAS level and its transfer/smoothing semantics."""

    operator: Callable[[PyTree[Any], Any], PyTree[Array]]
    smoother: Callable[[PyTree[Any], PyTree[Any], Any], PyTree[Array]]
    restrict_state: Callable[[PyTree[Any]], PyTree[Array]] | None
    restrict_residual: Callable[[PyTree[Any]], PyTree[Array]] | None
    prolong_correction: Callable[[PyTree[Any]], PyTree[Array]] | None
    state_space: AbstractVectorSpace
    residual_space: AbstractVectorSpace
    level_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: Callable[[PyTree[Any], Any], PyTree[Array]],
        smoother: Callable[[PyTree[Any], PyTree[Any], Any], PyTree[Array]],
        /,
        *,
        state_space: AbstractVectorSpace,
        residual_space: AbstractVectorSpace,
        restrict_state: Callable[[PyTree[Any]], PyTree[Array]] | None = None,
        restrict_residual: Callable[[PyTree[Any]], PyTree[Array]] | None = None,
        prolong_correction: Callable[[PyTree[Any]], PyTree[Array]] | None = None,
        level_id: str,
    ):
        values = (operator, smoother)
        spaces = (state_space, residual_space)
        transfers = (restrict_state, restrict_residual, prolong_correction)
        if not all(callable(value) for value in values):
            raise TypeError("FAS operator and smoother must be callable.")
        if not all(isinstance(space, AbstractVectorSpace) for space in spaces):
            raise TypeError(
                "state_space and residual_space must be AbstractVectorSpace values."
            )
        if any(value is not None and not callable(value) for value in transfers):
            raise TypeError("FAS transfers must be callable or None.")
        identifier = str(level_id)
        if not identifier:
            raise ValueError("level_id must be non-empty.")
        self.operator = operator
        self.smoother = smoother
        self.state_space = state_space
        self.residual_space = residual_space
        self.restrict_state = restrict_state
        self.restrict_residual = restrict_residual
        self.prolong_correction = prolong_correction
        self.level_id = identifier

    @property
    def has_coarse_transfer(self) -> bool:
        return (
            self.restrict_state is not None
            and self.restrict_residual is not None
            and self.prolong_correction is not None
        )

    def evaluate(
        self,
        state: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        state_ = self.state_space.validate(state)
        return self.residual_space.validate(self.operator(state_, args))

    def smooth(
        self,
        state: PyTree[Any],
        right_hand_side: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        state_ = self.state_space.validate(state)
        right_hand_side_ = self.residual_space.validate(right_hand_side)
        return self.state_space.validate(self.smoother(state_, right_hand_side_, args))


class FASCyclePolicy(StrictModule):
    """Static smoothing counts and V/W/F coarse-recursion policy."""

    kind: FASCycleKind = eqx.field(static=True)
    pre_smoothing_steps: int = eqx.field(static=True)
    post_smoothing_steps: int = eqx.field(static=True)

    def __init__(
        self,
        kind: FASCycleKind = "v",
        /,
        *,
        pre_smoothing_steps: int = 1,
        post_smoothing_steps: int = 1,
    ):
        if kind not in ("v", "w", "f"):
            raise ValueError(f"Unknown FAS cycle kind {kind!r}.")
        pre = int(pre_smoothing_steps)
        post = int(post_smoothing_steps)
        if pre < 0 or post < 0 or pre + post == 0:
            raise ValueError(
                "FAS smoothing counts must be non-negative and not both zero."
            )
        self.kind = kind
        self.pre_smoothing_steps = pre
        self.post_smoothing_steps = post


class FASHierarchy(StrictModule):
    """Nonlinear hierarchy with explicit level-local operators and exact FAS transfers."""

    levels: tuple[FASLevel, ...]
    coarse_solver: Callable[[PyTree[Any], PyTree[Any], Any], PyTree[Array]]
    hierarchy_id: str = eqx.field(static=True)
    numeric_refreshes: Array

    def __init__(
        self,
        levels: tuple[FASLevel, ...],
        coarse_solver: Callable[[PyTree[Any], PyTree[Any], Any], PyTree[Array]],
        /,
        *,
        hierarchy_id: str = "nonlinear-fas",
        numeric_refreshes: int = 0,
    ):
        levels_ = tuple(levels)
        if len(levels_) < 2:
            raise ValueError("FAS requires at least one fine and one coarse level.")
        if not all(isinstance(level, FASLevel) for level in levels_):
            raise TypeError("Every FAS hierarchy entry must be an FASLevel.")
        if not all(level.has_coarse_transfer for level in levels_[:-1]):
            raise ValueError("Every non-coarsest FAS level requires all three transfers.")
        coarsest = levels_[-1]
        if any(
            transfer is not None
            for transfer in (
                coarsest.restrict_state,
                coarsest.restrict_residual,
                coarsest.prolong_correction,
            )
        ):
            raise ValueError("The coarsest FAS level must not declare coarse transfers.")
        if not callable(coarse_solver):
            raise TypeError("coarse_solver must be callable.")
        refreshes = int(numeric_refreshes)
        if refreshes < 0:
            raise ValueError("numeric_refreshes must be non-negative.")
        identifier = str(hierarchy_id)
        if not identifier:
            raise ValueError("hierarchy_id must be non-empty.")
        self.levels = levels_
        self.coarse_solver = coarse_solver
        self.hierarchy_id = identifier
        self.numeric_refreshes = jnp.asarray(refreshes, dtype=jnp.int32)


class FASDiagnostics(StrictModule):
    """Cycle work and physical residual-reduction evidence."""

    initial_residual_norm: Array
    final_residual_norm: Array
    residual_reduction: Array
    smoothing_steps: Array
    coarse_solves: Array
    level_visits: Array
    finite: Array


class FASResult(StrictModule):
    """One nonlinear FAS cycle result."""

    state: PyTree[Array]
    residual: PyTree[Array]
    diagnostics: FASDiagnostics
    hierarchy_id: str = eqx.field(static=True)
    cycle_kind: FASCycleKind = eqx.field(static=True)
    status: Array
    provenance: NonlinearProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(NonlinearStatus.SUCCESS)


def _subtract(left: PyTree[Any], right: PyTree[Any], /) -> PyTree[Array]:
    return tree_add_scaled(left, right, -1.0)


def _add(left: PyTree[Any], right: PyTree[Any], /) -> PyTree[Array]:
    return tree_add_scaled(left, right, 1.0)


def _space_norm(
    space: AbstractVectorSpace,
    vector: PyTree[Any],
    /,
) -> Array:
    squared = jnp.real(space.inner(vector, vector))
    return jnp.sqrt(jnp.maximum(squared, 0.0))


def _smooth(
    level: FASLevel,
    state: PyTree[Any],
    right_hand_side: PyTree[Any],
    args: Any,
    steps: int,
    /,
) -> PyTree[Array]:
    return jax.lax.fori_loop(
        0,
        steps,
        lambda _, current: level.smooth(current, right_hand_side, args),
        level.state_space.validate(state),
    )


def _fas_cycle_at(
    hierarchy: FASHierarchy,
    level_index: int,
    state: PyTree[Any],
    right_hand_side: PyTree[Any],
    args: Any,
    policy: FASCyclePolicy,
    kind: FASCycleKind,
    /,
) -> PyTree[Array]:
    level = hierarchy.levels[level_index]
    if level_index == len(hierarchy.levels) - 1:
        coarse_state = hierarchy.coarse_solver(
            level.state_space.validate(state),
            level.residual_space.validate(right_hand_side),
            args,
        )
        return level.state_space.validate(coarse_state)

    smoothed = _smooth(
        level,
        state,
        right_hand_side,
        args,
        policy.pre_smoothing_steps,
    )
    fine_value = level.evaluate(smoothed, args)
    fine_defect = _subtract(right_hand_side, fine_value)
    if (
        level.restrict_state is None
        or level.restrict_residual is None
        or level.prolong_correction is None
    ):
        raise ValueError("Non-coarsest FAS level is missing transfer operators.")
    coarse_level = hierarchy.levels[level_index + 1]
    coarse_initial = coarse_level.state_space.validate(level.restrict_state(smoothed))
    restricted_defect = coarse_level.residual_space.validate(
        level.restrict_residual(fine_defect)
    )
    coarse_right_hand_side = _add(
        coarse_level.evaluate(coarse_initial, args), restricted_defect
    )
    first_kind = kind
    coarse_state = _fas_cycle_at(
        hierarchy,
        level_index + 1,
        coarse_initial,
        coarse_right_hand_side,
        args,
        policy,
        first_kind,
    )
    if kind in ("w", "f"):
        second_kind: FASCycleKind = "w" if kind == "w" else "v"
        coarse_state = _fas_cycle_at(
            hierarchy,
            level_index + 1,
            coarse_state,
            coarse_right_hand_side,
            args,
            policy,
            second_kind,
        )
    coarse_correction = coarse_level.state_space.validate(
        _subtract(coarse_state, coarse_initial)
    )
    correction = level.state_space.validate(level.prolong_correction(coarse_correction))
    corrected = level.state_space.validate(_add(smoothed, correction))
    return _smooth(
        level,
        corrected,
        right_hand_side,
        args,
        policy.post_smoothing_steps,
    )


def _cycle_counts(
    level_count: int,
    policy: FASCyclePolicy,
    kind: FASCycleKind,
    level_index: int = 0,
) -> tuple[int, int, tuple[int, ...]]:
    visits = [0] * level_count

    def count(index: int, cycle_kind: FASCycleKind) -> tuple[int, int]:
        visits[index] += 1
        if index == level_count - 1:
            return 0, 1
        smooth = policy.pre_smoothing_steps + policy.post_smoothing_steps
        first_kind = cycle_kind
        child_smooth, child_coarse = count(index + 1, first_kind)
        if cycle_kind in ("w", "f"):
            second_kind: FASCycleKind = "w" if cycle_kind == "w" else "v"
            extra_smooth, extra_coarse = count(index + 1, second_kind)
            child_smooth += extra_smooth
            child_coarse += extra_coarse
        return smooth + child_smooth, child_coarse

    smoothing, coarse = count(level_index, kind)
    return smoothing, coarse, tuple(visits)


def fas_cycle(
    hierarchy: FASHierarchy,
    state: PyTree[Any],
    /,
    *,
    right_hand_side: PyTree[Any] | None = None,
    args: Any = None,
    policy: FASCyclePolicy | None = None,
) -> FASResult:
    """Execute an exact tau-corrected nonlinear FAS V, W, or F cycle."""
    if not isinstance(hierarchy, FASHierarchy):
        raise TypeError("hierarchy must be a FASHierarchy.")
    policy_ = FASCyclePolicy() if policy is None else policy
    if not isinstance(policy_, FASCyclePolicy):
        raise TypeError("policy must be FASCyclePolicy or None.")
    finest = hierarchy.levels[0]
    initial_state = finest.state_space.validate(state)
    initial_value = finest.evaluate(initial_state, args)
    rhs = (
        finest.residual_space.zeros()
        if right_hand_side is None
        else finest.residual_space.validate(right_hand_side)
    )
    initial_residual = _subtract(initial_value, rhs)
    solved = _fas_cycle_at(
        hierarchy,
        0,
        initial_state,
        rhs,
        args,
        policy_,
        policy_.kind,
    )
    final_residual = _subtract(finest.evaluate(solved, args), rhs)
    initial_norm = _space_norm(finest.residual_space, initial_residual)
    final_norm = _space_norm(finest.residual_space, final_residual)
    smoothing, coarse, visits = _cycle_counts(
        len(hierarchy.levels), policy_, policy_.kind
    )
    finite = jnp.isfinite(initial_norm) & jnp.isfinite(final_norm)
    diagnostics = FASDiagnostics(
        initial_residual_norm=initial_norm,
        final_residual_norm=final_norm,
        residual_reduction=final_norm / jnp.maximum(initial_norm, 1e-30),
        smoothing_steps=jnp.asarray(smoothing, dtype=jnp.int32),
        coarse_solves=jnp.asarray(coarse, dtype=jnp.int32),
        level_visits=jnp.asarray(visits, dtype=jnp.int32),
        finite=finite,
    )
    status = jnp.where(
        finite,
        int(NonlinearStatus.SUCCESS),
        int(NonlinearStatus.NONFINITE_EVALUATION),
    ).astype(jnp.int32)
    return FASResult(
        state=solved,
        residual=final_residual,
        diagnostics=diagnostics,
        hierarchy_id=hierarchy.hierarchy_id,
        cycle_kind=policy_.kind,
        status=status,
        provenance=NonlinearProvenance(
            problem_id=hierarchy.hierarchy_id,
            method_id=f"fas-{policy_.kind}-cycle",
            derivative_id="none",
            globalization_id="multilevel-correction",
            notes="explicit-state-residual-transfers",
        ),
    )


class FASNonlinearPreconditioner(AbstractNonlinearUpdate):
    """One prepared FAS cycle exposed as a finite nonlinear update."""

    hierarchy: FASHierarchy
    policy: FASCyclePolicy

    def __init__(
        self,
        hierarchy: FASHierarchy,
        /,
        *,
        policy: FASCyclePolicy | None = None,
    ):
        if not isinstance(hierarchy, FASHierarchy):
            raise TypeError("hierarchy must be a FASHierarchy.")
        policy_ = FASCyclePolicy() if policy is None else policy
        if not isinstance(policy_, FASCyclePolicy):
            raise TypeError("policy must be FASCyclePolicy or None.")
        self.hierarchy = hierarchy
        self.policy = policy_

    @property
    def update_id(self) -> str:
        return f"fas-{self.policy.kind}/{self.hierarchy.hierarchy_id}"

    @property
    def capabilities(self) -> NonlinearUpdateCapabilities:
        return NonlinearUpdateCapabilities(
            jit=True,
            prepared_refresh=True,
            differentiable_action=True,
            counts_complete=False,
        )

    @property
    def maximum_work(self) -> NonlinearWork:
        return NonlinearWork(
            residual_evaluations=2,
            local_updates=1,
            complete=False,
        )

    def _prepare_internal(self, problem, state, args, /):
        del problem, state, args
        return None

    def _refresh_internal(self, internal_state, problem, state, args, /):
        del problem, state, args
        return internal_state

    def __call__(self, state: PyTree[Any], args: Any = None, /) -> PyTree[Array]:
        return fas_cycle(
            self.hierarchy,
            state,
            args=args,
            policy=self.policy,
        ).state

    def _apply(
        self,
        prepared: PreparedNonlinearUpdate,
        state: PyTree[Any],
        args: Any,
        control: NonlinearUpdateControl,
        /,
    ):
        problem = prepared.problem
        state_ = prepared.plan.state_space.validate(state)

        def skipped(_):
            diagnostics = NonlinearUpdateDiagnostics(
                initial_residual_norm=jnp.asarray(jnp.nan),
                final_residual_norm=jnp.asarray(jnp.nan),
                step_norm=0.0,
                work=NonlinearWork.zero(complete=False),
            )
            return (
                NonlinearUpdateResult(
                    state=state_,
                    residual=prepared.plan.residual_space.zeros(),
                    auxiliary=prepared.reference_auxiliary,
                    status=NonlinearUpdateStatus.BUDGET_EXHAUSTED,
                    diagnostics=diagnostics,
                    provenance=NonlinearUpdateProvenance(
                        problem_id=problem.problem_id,
                        update_id=self.update_id,
                        plan_id=prepared.plan.plan_id,
                        notes="FAS work is incomplete under a finite budget.",
                    ),
                ),
                prepared.internal_state,
            )

        def execute(_):
            initial_residual, _ = problem.evaluate(state_, args)
            initial_norm = _space_norm(
                prepared.plan.residual_space,
                initial_residual,
            )
            cycle = fas_cycle(
                self.hierarchy,
                state_,
                args=args,
                policy=self.policy,
            )
            residual, auxiliary = problem.evaluate(cycle.state, args)
            final_norm = _space_norm(prepared.plan.residual_space, residual)
            finite = (
                tree_allfinite(cycle.state)
                & tree_allfinite(residual)
                & cycle.diagnostics.finite
            )
            valid = problem.valid(cycle.state, residual, auxiliary, args)
            status = jnp.where(
                ~finite,
                int(NonlinearUpdateStatus.NONFINITE_EVALUATION),
                jnp.where(
                    ~valid,
                    int(NonlinearUpdateStatus.DOMAIN_REJECTED),
                    int(NonlinearUpdateStatus.APPLIED),
                ),
            ).astype(jnp.int32)
            diagnostics = NonlinearUpdateDiagnostics(
                initial_residual_norm=initial_norm,
                final_residual_norm=final_norm,
                step_norm=_space_norm(
                    prepared.plan.state_space,
                    jax.tree.map(
                        lambda new, old: new - old,
                        cycle.state,
                        state_,
                    ),
                ),
                work=self.maximum_work,
                accepted_steps=(status == int(NonlinearUpdateStatus.APPLIED)).astype(
                    jnp.int32
                ),
                rejected_steps=(status != int(NonlinearUpdateStatus.APPLIED)).astype(
                    jnp.int32
                ),
                domain_failures=(finite & ~valid).astype(jnp.int32),
                nonfinite_trials=(~finite).astype(jnp.int32),
            )
            return (
                NonlinearUpdateResult(
                    state=cycle.state,
                    residual=residual,
                    auxiliary=auxiliary,
                    status=status,
                    diagnostics=diagnostics,
                    provenance=NonlinearUpdateProvenance(
                        problem_id=problem.problem_id,
                        update_id=self.update_id,
                        plan_id=prepared.plan.plan_id,
                        notes=(
                            f"cycle={self.policy.kind};"
                            f"hierarchy={self.hierarchy.hierarchy_id}"
                        ),
                    ),
                ),
                prepared.internal_state,
            )

        return jax.lax.cond(
            control.permits(self.maximum_work),
            execute,
            skipped,
            operand=None,
        )


__all__ = [
    "FASCycleKind",
    "FASCyclePolicy",
    "FASDiagnostics",
    "FASHierarchy",
    "FASLevel",
    "FASNonlinearPreconditioner",
    "FASResult",
    "fas_cycle",
]
