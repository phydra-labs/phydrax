#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._tree_math import tree_allfinite
from ..linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    solve as solve_linear,
)
from ._precision import NonlinearPrecisionPolicy
from ._updates import (
    AbstractNonlinearUpdate,
    apply_prepared_nonlinear_update,
    NonlinearUpdateCapabilities,
    NonlinearUpdateControl,
    NonlinearUpdateDiagnostics,
    NonlinearUpdateProvenance,
    NonlinearUpdateResult,
    NonlinearUpdateStatus,
    prepare_nonlinear_update,
    PreparedNonlinearUpdate,
    refresh_nonlinear_update,
    skipped_nonlinear_update_result,
)
from ._work import NonlinearWork, work_sum


NonlinearCompositionKind: TypeAlias = Literal[
    "multiplicative", "additive", "residual-optimal"
]


def _space_norm(
    space,
    value,
    precision: NonlinearPrecisionPolicy,
    /,
) -> Array:
    return precision.norm(space, value)


def _component_control(
    control: NonlinearUpdateControl,
    count: int,
    *,
    residual_reserve: int,
) -> NonlinearUpdateControl:
    return control.split(
        count,
        reserve=NonlinearWork(
            residual_evaluations=residual_reserve,
            validity_evaluations=1,
        ),
    )


def _sum_component_field(components, name: str, /) -> Array:
    values = [vars(component.diagnostics)[name] for component in components]
    return sum(values[1:], values[0])


class CompositeNonlinearUpdate(AbstractNonlinearUpdate):
    """Static additive, multiplicative, or residual-optimal update composition."""

    updates: tuple[AbstractNonlinearUpdate, ...]
    weights: tuple[float, ...] = eqx.field(static=True)
    kind: NonlinearCompositionKind = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    safeguard_factor: float = eqx.field(static=True)
    update_name: str = eqx.field(static=True)
    linear: LinearSolvePolicy
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        updates: tuple[AbstractNonlinearUpdate, ...],
        /,
        *,
        kind: NonlinearCompositionKind = "multiplicative",
        weights: tuple[float, ...] | None = None,
        regularization: float = 1e-10,
        safeguard_factor: float = 1.0,
        linear: LinearSolvePolicy | None = None,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        updates_ = tuple(updates)
        if not updates_ or not all(
            isinstance(update, AbstractNonlinearUpdate) for update in updates_
        ):
            raise TypeError(
                "updates must be a nonempty tuple of AbstractNonlinearUpdate values."
            )
        if kind not in ("multiplicative", "additive", "residual-optimal"):
            raise ValueError("Unknown nonlinear composition kind.")
        weights_ = (
            tuple(1.0 for _ in updates_)
            if weights is None
            else tuple(float(value) for value in weights)
        )
        if len(weights_) != len(updates_):
            raise ValueError("Composition weights must match the child count.")
        if any(not isfinite(value) for value in weights_):
            raise ValueError("Composition weights must be finite.")
        regularization_ = float(regularization)
        safeguard_ = float(safeguard_factor)
        if not isfinite(regularization_) or regularization_ < 0.0:
            raise ValueError("regularization must be finite and non-negative.")
        if not isfinite(safeguard_) or safeguard_ < 1.0:
            raise ValueError("safeguard_factor must be finite and at least one.")
        linear_ = LinearSolvePolicy(DenseSVD()) if linear is None else linear
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        identifier = canonical_fingerprint(
            {
                "kind": "nonlinear-composition",
                "composition": kind,
                "updates": [update.update_id for update in updates_],
                "weights": list(weights_),
                "regularization": regularization_,
                "safeguard": safeguard_,
            }
        )
        self.updates = updates_
        self.weights = weights_
        self.kind = kind
        self.regularization = regularization_
        self.safeguard_factor = safeguard_
        self.update_name = f"composite-{kind}/{identifier}"
        self.linear = linear_
        self.precision = precision_

    @property
    def update_id(self) -> str:
        return self.update_name

    @property
    def capabilities(self) -> NonlinearUpdateCapabilities:
        return NonlinearUpdateCapabilities(
            jit=all(update.capabilities.jit for update in self.updates),
            prepared_refresh=all(
                update.capabilities.prepared_refresh for update in self.updates
            ),
            differentiable_action=all(
                update.capabilities.differentiable_action for update in self.updates
            ),
            exposes_linearization=False,
            counts_complete=all(
                update.capabilities.counts_complete for update in self.updates
            ),
        )

    @property
    def maximum_work(self) -> NonlinearWork:
        reserve = NonlinearWork(
            residual_evaluations=(4 if self.kind == "residual-optimal" else 2),
            validity_evaluations=1,
        )
        return work_sum(tuple(update.maximum_work for update in self.updates)) + reserve

    def _prepare_internal(self, problem, state, args, /):
        return tuple(
            prepare_nonlinear_update(problem, state, update, args=args)
            for update in self.updates
        )

    def _refresh_internal(self, internal_state, problem, state, args, /):
        children = tuple(internal_state)
        if len(children) != len(self.updates) or not all(
            isinstance(child, PreparedNonlinearUpdate) for child in children
        ):
            raise TypeError("Prepared composite child state is invalid.")
        return tuple(
            refresh_nonlinear_update(child, problem, state, args=args)
            for child in children
        )

    def _apply(
        self,
        prepared: PreparedNonlinearUpdate,
        state: PyTree[Any],
        args: Any,
        control: NonlinearUpdateControl,
        /,
    ):
        children = tuple(prepared.internal_state)
        child_control = _component_control(
            control,
            len(children),
            residual_reserve=(4 if self.kind == "residual-optimal" else 2),
        )
        if self.kind == "multiplicative":
            components, next_children, candidate = self._apply_multiplicative(
                children,
                prepared.plan.state_space.validate(state),
                args,
                child_control,
            )
        else:
            components, next_children, candidate = self._apply_additive(
                prepared,
                children,
                prepared.plan.state_space.validate(state),
                args,
                child_control,
                residual_optimal=self.kind == "residual-optimal",
            )
        return self._package(
            prepared,
            state,
            candidate,
            args,
            components,
        ), next_children

    def _apply_multiplicative(self, children, state, args, control, /):
        current = state
        components = []
        next_children = []
        active = jnp.asarray(True)
        for child in children:
            child_dynamic, child_static = eqx.partition(child, eqx.is_array)

            def execute(_):
                combined = eqx.combine(child_dynamic, child_static)
                result, next_child = apply_prepared_nonlinear_update(
                    combined,
                    current,
                    args=args,
                    control=control,
                )
                next_dynamic, _ = eqx.partition(next_child, eqx.is_array)
                return result, next_dynamic

            def skip(_):
                combined = eqx.combine(child_dynamic, child_static)
                return (
                    skipped_nonlinear_update_result(
                        combined,
                        current,
                    ),
                    child_dynamic,
                )

            result, next_dynamic = jax.lax.cond(
                active,
                execute,
                skip,
                operand=None,
            )
            next_child = eqx.combine(next_dynamic, child_static)
            take = active & result.applied
            current = jax.tree.map(
                lambda proposed, old: jnp.where(take, proposed, old),
                result.state,
                current,
            )
            active = active & result.applied
            components.append(result)
            next_children.append(next_child)
        return tuple(components), tuple(next_children), current

    def _apply_additive(
        self,
        prepared,
        children,
        state,
        args,
        control,
        /,
        *,
        residual_optimal: bool,
    ):
        components = []
        next_children = []
        for child in children:
            result, next_child = apply_prepared_nonlinear_update(
                child,
                state,
                args=args,
                control=control,
            )
            components.append(result)
            next_children.append(next_child)
        components_ = tuple(components)
        if residual_optimal:
            candidate = self._residual_optimal_candidate(
                prepared,
                state,
                components_,
                args,
            )
        else:
            candidate = state
            for weight, component in zip(self.weights, components_, strict=True):
                candidate = jax.tree.map(
                    lambda value, proposed, base, scale=weight: (
                        value + scale * (proposed - base)
                    ),
                    candidate,
                    component.state,
                    state,
                )
        return components_, tuple(next_children), candidate

    def _residual_optimal_candidate(
        self,
        prepared,
        state,
        components,
        args,
        /,
    ):
        problem = prepared.problem
        residual_space = prepared.plan.residual_space
        base_residual, _ = problem.evaluate(state, args)
        residual_differences = tuple(
            jax.tree.map(
                lambda child, base: child - base,
                component.residual,
                base_residual,
            )
            for component in components
        )
        gram = jnp.stack(
            [
                jnp.stack(
                    [
                        jnp.real(self.precision.inner(residual_space, left, right))
                        for right in residual_differences
                    ]
                )
                for left in residual_differences
            ]
        )
        gram = gram + self.regularization * jnp.eye(
            len(components),
            dtype=gram.dtype,
        )
        right = -jnp.stack(
            [
                jnp.real(
                    self.precision.inner(
                        residual_space,
                        delta,
                        base_residual,
                    )
                )
                for delta in residual_differences
            ]
        )
        coefficients = solve_linear(
            LeastSquaresProblem(DenseLinearOperator(gram)),
            right,
            policy=self.precision.bind_linear(self.linear),
        ).value
        accelerated = state
        for coefficient, component in zip(coefficients, components, strict=True):
            accelerated = jax.tree.map(
                lambda value, proposed, base, scale=coefficient: jnp.asarray(
                    value + scale * (proposed - base),
                    dtype=value.dtype,
                ),
                accelerated,
                component.state,
                state,
            )
        accelerated_residual, accelerated_auxiliary = problem.evaluate(
            accelerated,
            args,
        )
        accelerated_norm = _space_norm(
            residual_space,
            accelerated_residual,
            self.precision,
        )
        child_norms = jnp.stack(
            [
                _space_norm(
                    residual_space,
                    component.residual,
                    self.precision,
                )
                for component in components
            ]
        )
        best_index = jnp.argmin(child_norms)
        best_state = components[0].state
        for index, component in enumerate(components):
            best_state = jax.tree.map(
                lambda selected, proposed, take=best_index == index: jnp.where(
                    take,
                    proposed,
                    selected,
                ),
                best_state,
                component.state,
            )
        all_applied = jnp.asarray(True)
        for component in components:
            all_applied = all_applied & component.applied
        base_norm = _space_norm(
            residual_space,
            base_residual,
            self.precision,
        )
        finite = (
            tree_allfinite(accelerated)
            & tree_allfinite(accelerated_residual)
            & jnp.all(jnp.isfinite(coefficients))
        )
        valid = problem.valid(
            accelerated,
            accelerated_residual,
            accelerated_auxiliary,
            args,
        )
        use_accelerated = (
            all_applied
            & finite
            & valid
            & (
                accelerated_norm
                <= self.safeguard_factor * jnp.minimum(base_norm, jnp.min(child_norms))
            )
        )
        selected = jax.tree.map(
            lambda accelerated_value, best_value: jnp.where(
                use_accelerated,
                accelerated_value,
                best_value,
            ),
            accelerated,
            best_state,
        )
        return jax.tree.map(
            lambda proposed, base: jnp.where(all_applied, proposed, base),
            selected,
            state,
        )

    def _package(self, prepared, initial_state, candidate, args, components, /):
        problem = prepared.problem
        initial_residual, _ = problem.evaluate(initial_state, args)
        candidate = prepared.plan.state_space.validate(candidate)
        residual, auxiliary = problem.evaluate(candidate, args)
        self.precision.validate_trees(initial_state, initial_residual)
        initial_norm = _space_norm(
            prepared.plan.residual_space,
            initial_residual,
            self.precision,
        )
        final_norm = _space_norm(
            prepared.plan.residual_space,
            residual,
            self.precision,
        )
        finite = tree_allfinite(candidate) & tree_allfinite(residual)
        valid = problem.valid(candidate, residual, auxiliary, args)
        children_applied = jnp.asarray(True)
        for component in components:
            children_applied = children_applied & component.applied
        status = jnp.where(
            ~children_applied,
            int(NonlinearUpdateStatus.INNER_FAILURE),
            jnp.where(
                ~finite,
                int(NonlinearUpdateStatus.NONFINITE_EVALUATION),
                jnp.where(
                    ~valid,
                    int(NonlinearUpdateStatus.DOMAIN_REJECTED),
                    int(NonlinearUpdateStatus.APPLIED),
                ),
            ),
        ).astype(jnp.int32)
        step = jax.tree.map(lambda new, old: new - old, candidate, initial_state)
        component_work = work_sum(
            tuple(component.diagnostics.work for component in components)
        )
        wrapper_work = NonlinearWork(
            residual_evaluations=(4 if self.kind == "residual-optimal" else 2),
            validity_evaluations=(2 if self.kind == "residual-optimal" else 1),
        )
        diagnostics = NonlinearUpdateDiagnostics(
            initial_residual_norm=initial_norm,
            final_residual_norm=final_norm,
            step_norm=_space_norm(
                prepared.plan.state_space,
                step,
                self.precision,
            ),
            work=component_work + wrapper_work,
            accepted_steps=_sum_component_field(components, "accepted_steps")
            + (status == int(NonlinearUpdateStatus.APPLIED)).astype(jnp.int32),
            rejected_steps=_sum_component_field(components, "rejected_steps")
            + (status != int(NonlinearUpdateStatus.APPLIED)).astype(jnp.int32),
            domain_failures=_sum_component_field(components, "domain_failures")
            + (finite & ~valid).astype(jnp.int32),
            nonfinite_trials=_sum_component_field(components, "nonfinite_trials")
            + (~finite).astype(jnp.int32),
        )
        return NonlinearUpdateResult(
            state=candidate,
            residual=residual,
            auxiliary=auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=NonlinearUpdateProvenance(
                problem_id=problem.problem_id,
                update_id=self.update_id,
                plan_id=prepared.plan.plan_id,
                notes=(
                    f"composition={self.kind};precision-policy={self.precision.policy_id}"
                ),
            ),
            components=components,
        )


__all__ = [
    "CompositeNonlinearUpdate",
    "NonlinearCompositionKind",
]
