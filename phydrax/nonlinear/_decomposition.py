#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._tree_math import tree_allfinite
from ..linalg import AbstractVectorSpace
from ._types import NonlinearSystemProblem
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


def _norm(space, value, /):
    squared = jnp.real(space.inner(value, value))
    return jnp.sqrt(jnp.maximum(squared, 0.0))


def _sum_field(results, field: str, /):
    values = [vars(result.diagnostics)[field] for result in results]
    return sum(values[1:], values[0])


def _child_control(
    control: NonlinearUpdateControl,
    count: int,
    /,
) -> NonlinearUpdateControl:
    return control.split(
        count,
        reserve=NonlinearWork(
            residual_evaluations=2,
            validity_evaluations=1,
        ),
    )


class NonlinearSubdomain(StrictModule):
    """One explicit nonlinear restriction, local problem, and correction route."""

    restrict_state: Any
    restrict_residual: Any
    prolong_correction: Any
    local_residual: Any
    update: AbstractNonlinearUpdate
    state_space: AbstractVectorSpace
    residual_space: AbstractVectorSpace
    weight: float = eqx.field(static=True)
    subdomain_id: str = eqx.field(static=True)

    def __init__(
        self,
        restrict_state: Any,
        restrict_residual: Any,
        prolong_correction: Any,
        local_residual: Any,
        update: AbstractNonlinearUpdate,
        /,
        *,
        state_space: AbstractVectorSpace,
        residual_space: AbstractVectorSpace,
        weight: float = 1.0,
        subdomain_id: str,
    ):
        callbacks = (
            restrict_state,
            restrict_residual,
            prolong_correction,
            local_residual,
        )
        if not all(callable(value) for value in callbacks):
            raise TypeError(
                "Nonlinear subdomain transfer and residual values must be callable."
            )
        if not isinstance(update, AbstractNonlinearUpdate):
            raise TypeError("update must be AbstractNonlinearUpdate.")
        if not isinstance(state_space, AbstractVectorSpace) or not isinstance(
            residual_space, AbstractVectorSpace
        ):
            raise TypeError("Subdomain state and residual spaces must be declared.")
        weight_ = float(weight)
        if not isfinite(weight_):
            raise ValueError("Subdomain weight must be finite.")
        identifier = str(subdomain_id)
        if not identifier:
            raise ValueError("subdomain_id must be non-empty.")
        self.restrict_state = restrict_state
        self.restrict_residual = restrict_residual
        self.prolong_correction = prolong_correction
        self.local_residual = local_residual
        self.update = update
        self.state_space = state_space
        self.residual_space = residual_space
        self.weight = weight_
        self.subdomain_id = identifier

    def local_problem(self, /) -> NonlinearSystemProblem:
        def residual(local_state, context):
            global_state, user_args = context
            return self.local_residual(local_state, global_state, user_args)

        return NonlinearSystemProblem(
            residual,
            state_space=self.state_space,
            residual_space=self.residual_space,
            problem_id=f"nonlinear-subdomain/{self.subdomain_id}",
        )


class AbstractNonlinearSchwarz(AbstractNonlinearUpdate):
    """Shared static nonlinear subspace-correction update."""

    subdomains: tuple[NonlinearSubdomain, ...]
    update_name: str = eqx.field(static=True)

    def __init__(self, subdomains: tuple[NonlinearSubdomain, ...], /):
        subdomains_ = tuple(subdomains)
        if not subdomains_ or not all(
            isinstance(subdomain, NonlinearSubdomain) for subdomain in subdomains_
        ):
            raise TypeError(
                "subdomains must be a nonempty tuple of NonlinearSubdomain values."
            )
        identifier = canonical_fingerprint(
            {
                "kind": self.schwarz_kind,
                "subdomains": [value.subdomain_id for value in subdomains_],
                "updates": [value.update.update_id for value in subdomains_],
                "weights": [value.weight for value in subdomains_],
            }
        )
        self.subdomains = subdomains_
        self.update_name = f"{self.schwarz_kind}/{identifier}"

    @property
    @abc.abstractmethod
    def schwarz_kind(self) -> str:
        raise NotImplementedError

    @property
    def update_id(self) -> str:
        return self.update_name

    @property
    def capabilities(self) -> NonlinearUpdateCapabilities:
        return NonlinearUpdateCapabilities(
            jit=all(value.update.capabilities.jit for value in self.subdomains),
            prepared_refresh=all(
                value.update.capabilities.prepared_refresh for value in self.subdomains
            ),
            differentiable_action=all(
                value.update.capabilities.differentiable_action
                for value in self.subdomains
            ),
            counts_complete=all(
                value.update.capabilities.counts_complete for value in self.subdomains
            ),
        )

    @property
    def maximum_work(self) -> NonlinearWork:
        children = work_sum(
            tuple(subdomain.update.maximum_work for subdomain in self.subdomains)
        )
        return children + NonlinearWork(
            residual_evaluations=2,
            validity_evaluations=1,
            local_updates=len(self.subdomains),
            complete=children.complete,
        )

    def _prepare_internal(self, problem, state, args, /):
        del problem
        return tuple(
            prepare_nonlinear_update(
                subdomain.local_problem(),
                subdomain.state_space.validate(subdomain.restrict_state(state)),
                subdomain.update,
                args=(state, args),
            )
            for subdomain in self.subdomains
        )

    def _refresh_internal(self, internal_state, problem, state, args, /):
        del problem
        children = tuple(internal_state)
        if len(children) != len(self.subdomains) or not all(
            isinstance(child, PreparedNonlinearUpdate) for child in children
        ):
            raise TypeError("Prepared nonlinear Schwarz child state is invalid.")
        return tuple(
            refresh_nonlinear_update(
                child,
                subdomain.local_problem(),
                subdomain.state_space.validate(subdomain.restrict_state(state)),
                args=(state, args),
            )
            for child, subdomain in zip(children, self.subdomains, strict=True)
        )

    def _apply(
        self,
        prepared: PreparedNonlinearUpdate,
        state: PyTree[Any],
        args: Any,
        control: NonlinearUpdateControl,
        /,
    ):
        state_ = prepared.plan.state_space.validate(state)
        initial_residual, _ = prepared.problem.evaluate(state_, args)
        child_control = _child_control(control, len(self.subdomains))
        if self.schwarz_kind == "nonlinear-additive-schwarz":
            candidate, components, children = self._apply_additive(
                prepared,
                state_,
                args,
                child_control,
            )
        else:
            candidate, components, children = self._apply_multiplicative(
                prepared,
                state_,
                args,
                child_control,
            )
        all_applied = jnp.asarray(True)
        for component in components:
            all_applied = all_applied & component.applied
        candidate = jax.tree.map(
            lambda proposed, base: jnp.where(all_applied, proposed, base),
            candidate,
            state_,
        )
        residual, auxiliary = prepared.problem.evaluate(candidate, args)
        finite = tree_allfinite(candidate) & tree_allfinite(residual)
        valid = prepared.problem.valid(candidate, residual, auxiliary, args)
        status = jnp.where(
            ~all_applied,
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
        step = jax.tree.map(lambda new, old: new - old, candidate, state_)
        component_work = work_sum(
            tuple(component.diagnostics.work for component in components)
        )
        diagnostics = NonlinearUpdateDiagnostics(
            initial_residual_norm=_norm(
                prepared.plan.residual_space,
                initial_residual,
            ),
            final_residual_norm=_norm(prepared.plan.residual_space, residual),
            step_norm=_norm(prepared.plan.state_space, step),
            work=component_work
            + NonlinearWork(
                residual_evaluations=2,
                validity_evaluations=1,
                local_updates=len(self.subdomains),
                complete=component_work.complete,
            ),
            accepted_steps=_sum_field(components, "accepted_steps")
            + (status == int(NonlinearUpdateStatus.APPLIED)).astype(jnp.int32),
            rejected_steps=_sum_field(components, "rejected_steps")
            + (status != int(NonlinearUpdateStatus.APPLIED)).astype(jnp.int32),
            domain_failures=_sum_field(components, "domain_failures")
            + (finite & ~valid).astype(jnp.int32),
            nonfinite_trials=_sum_field(components, "nonfinite_trials")
            + (~finite).astype(jnp.int32),
        )
        return (
            NonlinearUpdateResult(
                state=candidate,
                residual=residual,
                auxiliary=auxiliary,
                status=status,
                diagnostics=diagnostics,
                provenance=NonlinearUpdateProvenance(
                    problem_id=prepared.problem.problem_id,
                    update_id=self.update_id,
                    plan_id=prepared.plan.plan_id,
                    notes=f"subdomains={len(self.subdomains)}",
                ),
                components=components,
            ),
            children,
        )

    def _apply_additive(self, prepared, state, args, control, /):
        candidate = state
        components = []
        children = []
        for child, subdomain in zip(
            prepared.internal_state,
            self.subdomains,
            strict=True,
        ):
            local_state = subdomain.state_space.validate(subdomain.restrict_state(state))
            refreshed = refresh_nonlinear_update(
                child,
                subdomain.local_problem(),
                local_state,
                args=(state, args),
            )
            result, next_child = apply_prepared_nonlinear_update(
                refreshed,
                local_state,
                args=(state, args),
                control=control,
            )
            correction = jax.tree.map(
                lambda new, old: new - old,
                result.state,
                local_state,
            )
            prolonged = prepared.plan.state_space.validate(
                subdomain.prolong_correction(correction)
            )
            candidate = jax.tree.map(
                lambda value, delta, weight=subdomain.weight: value + weight * delta,
                candidate,
                prolonged,
            )
            components.append(result)
            children.append(next_child)
        return candidate, tuple(components), tuple(children)

    def _apply_multiplicative(self, prepared, state, args, control, /):
        current = state
        components = []
        children = []
        active = jnp.asarray(True)
        for child, subdomain in zip(
            prepared.internal_state,
            self.subdomains,
            strict=True,
        ):
            local_state = subdomain.state_space.validate(
                subdomain.restrict_state(current)
            )
            child_dynamic, child_static = eqx.partition(child, eqx.is_array)

            def execute(_):
                combined = eqx.combine(child_dynamic, child_static)
                refreshed = refresh_nonlinear_update(
                    combined,
                    subdomain.local_problem(),
                    local_state,
                    args=(current, args),
                )
                result, next_child = apply_prepared_nonlinear_update(
                    refreshed,
                    local_state,
                    args=(current, args),
                    control=control,
                )
                next_dynamic, _ = eqx.partition(next_child, eqx.is_array)
                return result, next_dynamic

            def skip(_):
                combined = eqx.combine(child_dynamic, child_static)
                return (
                    skipped_nonlinear_update_result(
                        combined,
                        local_state,
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
            correction = jax.tree.map(
                lambda new, old: new - old,
                result.state,
                local_state,
            )
            prolonged = prepared.plan.state_space.validate(
                subdomain.prolong_correction(correction)
            )
            proposal = jax.tree.map(
                lambda value, delta, weight=subdomain.weight: value + weight * delta,
                current,
                prolonged,
            )
            take = active & result.applied
            current = jax.tree.map(
                lambda proposed, old: jnp.where(take, proposed, old),
                proposal,
                current,
            )
            active = active & result.applied
            components.append(result)
            children.append(next_child)
        return current, tuple(components), tuple(children)


class NonlinearAdditiveSchwarz(AbstractNonlinearSchwarz):
    @property
    def schwarz_kind(self) -> str:
        return "nonlinear-additive-schwarz"


class NonlinearMultiplicativeSchwarz(AbstractNonlinearSchwarz):
    @property
    def schwarz_kind(self) -> str:
        return "nonlinear-multiplicative-schwarz"


class NonlinearGaussSeidel(AbstractNonlinearSchwarz):
    """Ordered nonlinear block sweep using multiplicative Schwarz semantics."""

    @property
    def schwarz_kind(self) -> str:
        return "nonlinear-gauss-seidel"


__all__ = [
    "AbstractNonlinearSchwarz",
    "NonlinearAdditiveSchwarz",
    "NonlinearGaussSeidel",
    "NonlinearMultiplicativeSchwarz",
    "NonlinearSubdomain",
]
