#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ._method import (
    AbstractLinearCombinatorialMethod,
    CombinatorialPlan,
    make_combinatorial_plan,
)
from ._problem import AbstractCombinatorialSpace, LinearCombinatorialProblem
from ._selection import relative_gap
from ._types import (
    CombinatorialCertificate,
    CombinatorialCertification,
    CombinatorialFeasibility,
    CombinatorialMethodCapabilities,
    CombinatorialProvenance,
    CombinatorialResult,
    CombinatorialStatus,
)


class SetPackingDecision(StrictModule):
    """Fixed-capacity mask of selected candidate sets."""

    selected: Array


class SetPackingSpace(AbstractCombinatorialSpace):
    """Weighted packing over a fixed candidate-by-resource incidence matrix."""

    incidence: Array
    capacities: Array
    valid: Array
    candidate_count: int = eqx.field(static=True)
    resource_count: int = eqx.field(static=True)
    minimum_selected: int = eqx.field(static=True)
    maximum_selected: int = eqx.field(static=True)
    _structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        incidence: Any,
        /,
        *,
        capacities: Any | None = None,
        valid: Any | None = None,
        minimum_selected: int = 0,
        maximum_selected: int | None = None,
    ):
        incidence_ = jnp.asarray(incidence, dtype=bool)
        if incidence_.ndim != 2:
            raise ValueError("incidence must be a rank-2 candidate-by-resource array.")
        candidates, resources = (int(size) for size in incidence_.shape)
        if candidates <= 0:
            raise ValueError("set packing requires at least one candidate.")
        if capacities is None:
            capacities_ = jnp.ones((resources,), dtype=jnp.int32)
        else:
            raw_capacities = jnp.asarray(capacities)
            if not jnp.issubdtype(raw_capacities.dtype, jnp.integer):
                raise TypeError("capacities must have an integer dtype.")
            capacities_ = raw_capacities.astype(jnp.int32)
            if capacities_.shape != (resources,):
                raise ValueError(
                    f"capacities must have shape {(resources,)}; got {capacities_.shape}."
                )
            if bool(jnp.any(capacities_ < 0)):
                raise ValueError("capacities must be non-negative.")
        if valid is None:
            valid_ = jnp.ones((candidates,), dtype=bool)
        else:
            valid_ = jnp.asarray(valid, dtype=bool)
            if valid_.shape != (candidates,):
                raise ValueError(
                    f"valid must have shape {(candidates,)}; got {valid_.shape}."
                )
        if isinstance(minimum_selected, bool) or not isinstance(
            minimum_selected, Integral
        ):
            raise TypeError("minimum_selected must be a non-negative integer.")
        minimum = int(minimum_selected)
        if maximum_selected is None:
            maximum = candidates
        else:
            if isinstance(maximum_selected, bool) or not isinstance(
                maximum_selected, Integral
            ):
                raise TypeError("maximum_selected must be a non-negative integer.")
            maximum = int(maximum_selected)
        if minimum < 0 or maximum < 0:
            raise ValueError("selection bounds must be non-negative.")
        if minimum > maximum:
            raise ValueError("minimum_selected cannot exceed maximum_selected.")
        if maximum > candidates:
            raise ValueError("maximum_selected cannot exceed the candidate count.")
        self.incidence = incidence_
        self.capacities = capacities_
        self.valid = valid_
        self.candidate_count = candidates
        self.resource_count = resources
        self.minimum_selected = minimum
        self.maximum_selected = maximum
        self._structure_id = canonical_fingerprint(
            {
                "kind": "set-packing-space",
                "incidence": array_tree_fingerprint(incidence_),
                "capacities": array_tree_fingerprint(capacities_),
                "valid": array_tree_fingerprint(valid_),
                "minimum_selected": minimum,
                "maximum_selected": maximum,
            }
        )

    @property
    def structure_id(self) -> str:
        return self._structure_id

    def decision_spec(self, /) -> SetPackingDecision:
        return SetPackingDecision(
            jax.ShapeDtypeStruct((self.candidate_count,), jnp.bool_)
        )

    def feature_spec(self, /) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.candidate_count,), jnp.float32)

    def canonicalize(self, decision: SetPackingDecision, /) -> SetPackingDecision:
        if not isinstance(decision, SetPackingDecision):
            raise TypeError("set-packing decisions must be SetPackingDecision values.")
        selected = jnp.asarray(decision.selected, dtype=bool)
        if selected.shape[-1:] != (self.candidate_count,):
            raise ValueError(
                f"selected must end with shape {(self.candidate_count,)}; got {selected.shape}."
            )
        return SetPackingDecision(selected & self.valid)

    def encode(self, decision: SetPackingDecision, /) -> Array:
        return self.canonicalize(decision).selected.astype(float)

    def audit(self, decision: SetPackingDecision, /) -> CombinatorialFeasibility:
        if not isinstance(decision, SetPackingDecision):
            raise TypeError("set-packing decisions must be SetPackingDecision values.")
        selected = jnp.asarray(decision.selected, dtype=bool)
        if selected.shape[-1:] != (self.candidate_count,):
            raise ValueError(
                f"selected must end with shape {(self.candidate_count,)}; got {selected.shape}."
            )
        invalid_residual = jnp.sum(selected & ~self.valid, axis=-1)
        count = jnp.sum(selected, axis=-1)
        lower_residual = jnp.maximum(self.minimum_selected - count, 0)
        upper_residual = jnp.maximum(count - self.maximum_selected, 0)
        if self.resource_count:
            usage = jnp.sum(
                selected[..., :, None] * self.incidence.astype(jnp.int32), axis=-2
            )
            capacity_residual = jnp.sum(jnp.maximum(usage - self.capacities, 0), axis=-1)
        else:
            capacity_residual = jnp.zeros(selected.shape[:-1], dtype=jnp.int32)
        residual = invalid_residual + lower_residual + upper_residual + capacity_residual
        return CombinatorialFeasibility(residual == 0, residual.astype(float))


def _prefer_selection(candidate: Array, incumbent: Array, /) -> Array:
    difference = candidate != incumbent
    first = jnp.argmax(difference.astype(jnp.int32))
    return jnp.any(difference) & candidate[first] & ~incumbent[first]


def _packing_bound(
    costs: Array,
    incidence: Array,
    capacities: Array,
    valid: Array,
    depth: Array,
    used: Array,
    count: Array,
    value: Array,
    minimum_selected: int,
    maximum_selected: int,
    /,
) -> Array:
    remaining = jnp.arange(costs.shape[0], dtype=jnp.int32) >= depth
    if incidence.shape[1]:
        capacity_ok = jnp.all(
            used[None, :] + incidence.astype(jnp.int32) <= capacities[None, :], axis=-1
        )
    else:
        capacity_ok = jnp.ones(costs.shape, dtype=bool)
    available = remaining & valid & capacity_ok
    enough = count + jnp.sum(available) >= minimum_selected
    relaxation = value + jnp.sum(jnp.where(available & (costs < 0.0), costs, 0.0))
    return jnp.where(enough & (count <= maximum_selected), relaxation, jnp.inf)


def _branch_and_bound_one(
    costs: Array,
    incidence: Array,
    capacities: Array,
    valid: Array,
    minimum_selected: int,
    maximum_selected: int,
    maximum_nodes: int,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    candidates = costs.shape[0]
    resources = incidence.shape[1]
    stack_capacity = candidates + 1
    root_selected = jnp.zeros((candidates,), dtype=bool)
    root_used = jnp.zeros((resources,), dtype=jnp.int32)
    root_bound = _packing_bound(
        costs,
        incidence,
        capacities,
        valid,
        jnp.asarray(0, dtype=jnp.int32),
        root_used,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=costs.dtype),
        minimum_selected,
        maximum_selected,
    )
    stack_selected = (
        jnp.zeros((stack_capacity, candidates), dtype=bool).at[0].set(root_selected)
    )
    stack_used = (
        jnp.zeros((stack_capacity, resources), dtype=jnp.int32).at[0].set(root_used)
    )
    stack_depth = jnp.zeros((stack_capacity,), dtype=jnp.int32)
    stack_count = jnp.zeros((stack_capacity,), dtype=jnp.int32)
    stack_value = jnp.zeros((stack_capacity,), dtype=costs.dtype)
    stack_bound = (
        jnp.full((stack_capacity,), jnp.inf, dtype=costs.dtype).at[0].set(root_bound)
    )
    initial = (
        stack_selected,
        stack_used,
        stack_depth,
        stack_count,
        stack_value,
        stack_bound,
        jnp.asarray(1, dtype=jnp.int32),
        jnp.zeros((candidates,), dtype=bool),
        jnp.asarray(jnp.inf, dtype=costs.dtype),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def condition(state):
        return (state[6] > 0) & (state[11] < maximum_nodes)

    def body(state):
        (
            selected_stack,
            used_stack,
            depth_stack,
            count_stack,
            value_stack,
            bound_stack,
            size,
            best_selected,
            best_value,
            has_best,
            tied,
            steps,
        ) = state
        slot = size - 1
        selected = selected_stack[slot]
        used = used_stack[slot]
        depth = depth_stack[slot]
        count = count_stack[slot]
        value = value_stack[slot]
        bound = bound_stack[slot]
        size = slot
        promising = jnp.isfinite(bound) & (~has_best | (bound <= best_value))
        leaf = depth == candidates

        def evaluate_leaf(values):
            incumbent_selected, incumbent_value, incumbent_valid, has_tie = values
            feasible = (count >= minimum_selected) & (count <= maximum_selected)
            equal_alternative = (
                feasible
                & incumbent_valid
                & (value == incumbent_value)
                & jnp.any(selected != incumbent_selected)
            )
            better = feasible & (
                ~incumbent_valid
                | (value < incumbent_value)
                | (
                    (value == incumbent_value)
                    & _prefer_selection(selected, incumbent_selected)
                )
            )
            return (
                jnp.where(better, selected, incumbent_selected),
                jnp.where(better, value, incumbent_value),
                incumbent_valid | feasible,
                has_tie | equal_alternative,
            )

        best_selected, best_value, has_best, tied = jax.lax.cond(
            promising & leaf,
            evaluate_leaf,
            lambda values: values,
            (best_selected, best_value, has_best, tied),
        )

        def expand(values):
            (
                selected_nodes,
                used_nodes,
                depth_nodes,
                count_nodes,
                value_nodes,
                bound_nodes,
                current_size,
            ) = values
            next_depth = depth + 1
            exclude_bound = _packing_bound(
                costs,
                incidence,
                capacities,
                valid,
                next_depth,
                used,
                count,
                value,
                minimum_selected,
                maximum_selected,
            )
            push_exclude = jnp.isfinite(exclude_bound) & (
                ~has_best | (exclude_bound <= best_value)
            )
            exclude_slot = current_size
            selected_nodes = selected_nodes.at[exclude_slot].set(selected)
            used_nodes = used_nodes.at[exclude_slot].set(used)
            depth_nodes = depth_nodes.at[exclude_slot].set(next_depth)
            count_nodes = count_nodes.at[exclude_slot].set(count)
            value_nodes = value_nodes.at[exclude_slot].set(value)
            bound_nodes = bound_nodes.at[exclude_slot].set(exclude_bound)
            current_size = current_size + push_exclude.astype(jnp.int32)
            if resources:
                include_used = used + incidence[depth].astype(jnp.int32)
                capacity_ok = jnp.all(include_used <= capacities)
            else:
                include_used = used
                capacity_ok = jnp.asarray(True)
            include_allowed = valid[depth] & capacity_ok & (count < maximum_selected)
            include_selected = selected.at[depth].set(True)
            include_value = value + costs[depth]
            include_count = count + 1
            include_bound = _packing_bound(
                costs,
                incidence,
                capacities,
                valid,
                next_depth,
                include_used,
                include_count,
                include_value,
                minimum_selected,
                maximum_selected,
            )
            push_include = (
                include_allowed
                & jnp.isfinite(include_bound)
                & (~has_best | (include_bound <= best_value))
            )
            include_slot = current_size
            selected_nodes = selected_nodes.at[include_slot].set(include_selected)
            used_nodes = used_nodes.at[include_slot].set(include_used)
            depth_nodes = depth_nodes.at[include_slot].set(next_depth)
            count_nodes = count_nodes.at[include_slot].set(include_count)
            value_nodes = value_nodes.at[include_slot].set(include_value)
            bound_nodes = bound_nodes.at[include_slot].set(include_bound)
            current_size = current_size + push_include.astype(jnp.int32)
            return (
                selected_nodes,
                used_nodes,
                depth_nodes,
                count_nodes,
                value_nodes,
                bound_nodes,
                current_size,
            )

        expanded = jax.lax.cond(
            promising & ~leaf,
            expand,
            lambda values: values,
            (
                selected_stack,
                used_stack,
                depth_stack,
                count_stack,
                value_stack,
                bound_stack,
                size,
            ),
        )
        return (*expanded, best_selected, best_value, has_best, tied, steps + 1)

    final = jax.lax.while_loop(condition, body, initial)
    size, best_selected, best_value, has_best, tied, steps = (
        final[6],
        final[7],
        final[8],
        final[9],
        final[10],
        final[11],
    )
    active_stack = jnp.arange(stack_capacity, dtype=jnp.int32) < size
    pending_bound = jnp.min(jnp.where(active_stack, final[5], jnp.inf))
    complete = size == 0
    lower_bound = jnp.where(
        complete & has_best, best_value, jnp.minimum(best_value, pending_bound)
    )
    return best_selected, best_value, lower_bound, has_best, complete, tied, steps


def _greedy_one(
    costs: Array,
    incidence: Array,
    capacities: Array,
    valid: Array,
    minimum_selected: int,
    maximum_selected: int,
    /,
) -> tuple[Array, Array, Array]:
    candidates = costs.shape[0]
    resources = incidence.shape[1]
    initial = (
        jnp.zeros((candidates,), dtype=bool),
        jnp.zeros((resources,), dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def choose(_, state):
        selected, used, count, steps = state
        if resources:
            capacity_ok = jnp.all(
                used[None, :] + incidence.astype(jnp.int32) <= capacities[None, :],
                axis=-1,
            )
        else:
            capacity_ok = jnp.ones((candidates,), dtype=bool)
        available = valid & ~selected & capacity_ok & (count < maximum_selected)
        masked = jnp.where(available, costs, jnp.inf)
        index = jnp.argmin(masked)
        best = masked[index]
        take = jnp.isfinite(best) & ((count < minimum_selected) | (best < 0.0))
        selected = selected.at[index].set(selected[index] | take)
        if resources:
            used = used + jnp.where(
                take, incidence[index].astype(jnp.int32), jnp.zeros_like(used)
            )
        return (
            selected,
            used,
            count + take.astype(jnp.int32),
            steps + take.astype(jnp.int32),
        )

    selected, _, _, steps = jax.lax.fori_loop(0, candidates, choose, initial)
    value = jnp.sum(jnp.where(selected, costs, 0.0))
    return selected, value, steps


class BranchAndBoundSetPacking(AbstractLinearCombinatorialMethod):
    """Deterministic exact set-packing search with a fixed node budget."""

    maximum_nodes: int = eqx.field(static=True)
    maximum_candidates: int = eqx.field(static=True)

    def __init__(self, *, maximum_nodes: int = 1_000_000, maximum_candidates: int = 1024):
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in (maximum_nodes, maximum_candidates)
        ):
            raise TypeError("set-packing resource limits must be positive integers.")
        if int(maximum_nodes) <= 0 or int(maximum_candidates) <= 0:
            raise ValueError("set-packing resource limits must be positive.")
        self.maximum_nodes = int(maximum_nodes)
        self.maximum_candidates = int(maximum_candidates)

    @property
    def method_id(self) -> str:
        return "native-branch-and-bound-set-packing"

    @property
    def capabilities(self) -> CombinatorialMethodCapabilities:
        return CombinatorialMethodCapabilities(
            exact=True,
            jax_native=True,
            jit=True,
            batched=True,
            signed_costs=True,
            deterministic_ties=True,
            optimality_certificate=True,
            surrogate_pullback=True,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (
            ("maximum_nodes", str(self.maximum_nodes)),
            ("maximum_candidates", str(self.maximum_candidates)),
        )

    def plan(
        self,
        problem: LinearCombinatorialProblem,
        certification: CombinatorialCertification,
        /,
    ) -> CombinatorialPlan:
        if not isinstance(problem.space, SetPackingSpace):
            raise TypeError("BranchAndBoundSetPacking requires SetPackingSpace.")
        if len(jax.tree_util.tree_leaves(problem.costs)) != 1:
            raise ValueError("SetPackingSpace requires one candidate-cost vector.")
        if problem.space.candidate_count > self.maximum_candidates:
            raise ValueError("set-packing candidate count exceeds maximum_candidates.")
        candidates = problem.space.candidate_count
        resources = problem.space.resource_count
        return make_combinatorial_plan(
            problem,
            self,
            certification,
            work_estimate=problem.batch_size * self.maximum_nodes,
            workspace_elements=problem.batch_size
            * (candidates + 1)
            * (candidates + resources + 5),
            certificate_kind="branch-and-bound-relaxation-gap",
        )

    def solve(
        self,
        problem: LinearCombinatorialProblem,
        plan: CombinatorialPlan,
        /,
    ) -> CombinatorialResult:
        space = problem.space
        if not isinstance(space, SetPackingSpace):
            raise TypeError("BranchAndBoundSetPacking requires SetPackingSpace.")
        raw_costs = jax.tree_util.tree_leaves(problem.costs)[0]
        batch_shape = problem.batch_shape
        flat_batch = problem.batch_size
        costs = raw_costs.reshape((flat_batch, space.candidate_count))
        finite = jnp.all(jnp.isfinite(costs), axis=-1)
        safe_costs = jnp.where(jnp.isfinite(costs), costs, 0.0)
        selected, best, lower, has_best, complete, tied, steps = jax.vmap(
            _branch_and_bound_one,
            in_axes=(0, None, None, None, None, None, None),
        )(
            safe_costs,
            space.incidence,
            space.capacities,
            space.valid,
            space.minimum_selected,
            space.maximum_selected,
            self.maximum_nodes,
        )
        decision = SetPackingDecision(
            selected.reshape(batch_shape + (space.candidate_count,))
        )
        features = space.encode(decision).astype(raw_costs.dtype)
        objective = problem.objective(features)
        feasibility = space.audit(decision)
        best_shaped = best.reshape(batch_shape)
        lower_shaped = lower.reshape(batch_shape)
        has_best_shaped = has_best.reshape(batch_shape)
        complete_shaped = complete.reshape(batch_shape)
        finite_shaped = finite.reshape(batch_shape)
        tolerance = plan.certification.threshold(objective, best_shaped)
        objective_consistent = has_best_shaped & (
            jnp.abs(objective - best_shaped) <= tolerance
        )
        optimality = (
            finite_shaped & complete_shaped & feasibility.feasible & objective_consistent
        )
        status = jnp.where(
            ~finite_shaped,
            int(CombinatorialStatus.NONFINITE_INPUT),
            jnp.where(
                complete_shaped & ~has_best_shaped,
                int(CombinatorialStatus.INFEASIBLE),
                jnp.where(
                    optimality,
                    int(CombinatorialStatus.OPTIMAL),
                    jnp.where(
                        ~complete_shaped,
                        int(CombinatorialStatus.MAXIMUM_STEPS_REACHED),
                        int(CombinatorialStatus.CERTIFICATION_FAILED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        valid_result = finite_shaped & feasibility.feasible & objective_consistent
        gap_available = has_best_shaped & jnp.isfinite(lower_shaped)
        absolute_gap = jnp.where(
            gap_available, jnp.maximum(best_shaped - lower_shaped, 0.0), jnp.nan
        )
        tie_available = complete_shaped & tied.reshape(batch_shape)
        zero = jnp.zeros(batch_shape, dtype=raw_costs.dtype)
        certificate = CombinatorialCertificate(
            finite=finite_shaped,
            feasible=feasibility.feasible,
            objective_consistent=objective_consistent,
            optimality_proven=optimality,
            primal_residual=feasibility.residual.astype(raw_costs.dtype),
            dual_residual=zero,
            absolute_gap=absolute_gap,
            relative_gap=relative_gap(absolute_gap, best_shaped, lower_shaped),
            tie_margin=jnp.where(tie_available, zero, jnp.nan),
            dual_available=jnp.zeros(batch_shape, dtype=bool),
            gap_available=gap_available,
            tie_available=tie_available,
        )
        provenance = CombinatorialProvenance(
            problem_id=problem.problem_id,
            structure_id=problem.structure_id,
            method_id=self.method_id,
            plan_id=plan.plan_id,
            implementation="phydrax-native-jax",
            tie_policy="include-lowest-candidate-first",
            certificate_kind=plan.certificate_kind,
            exact=True,
            signed_costs=True,
            configuration=plan.configuration,
        )
        decision = SetPackingDecision(
            jnp.where(valid_result[..., None], decision.selected, False)
        )
        features = jnp.where(valid_result[..., None], features, jnp.zeros_like(features))
        return CombinatorialResult(
            decision=decision,
            features=features,
            objective_value=jnp.where(valid_result, objective, jnp.nan),
            status=status,
            valid=valid_result,
            certificate=certificate,
            iterations=steps.reshape(batch_shape),
            work=(steps * (space.candidate_count + space.resource_count)).reshape(
                batch_shape
            ),
            provenance=provenance,
            batch_shape=batch_shape,
        )


class GreedySetPacking(AbstractLinearCombinatorialMethod):
    """Deterministic cost-ordered packing with a relaxation certificate."""

    maximum_candidates: int = eqx.field(static=True)

    def __init__(self, *, maximum_candidates: int = 1_000_000):
        if isinstance(maximum_candidates, bool) or not isinstance(
            maximum_candidates, Integral
        ):
            raise TypeError("maximum_candidates must be a positive integer.")
        if int(maximum_candidates) <= 0:
            raise ValueError("maximum_candidates must be positive.")
        self.maximum_candidates = int(maximum_candidates)

    @property
    def method_id(self) -> str:
        return "native-greedy-set-packing"

    @property
    def capabilities(self) -> CombinatorialMethodCapabilities:
        return CombinatorialMethodCapabilities(
            exact=False,
            jax_native=True,
            jit=True,
            batched=True,
            signed_costs=True,
            deterministic_ties=True,
            optimality_certificate=True,
            surrogate_pullback=True,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (("maximum_candidates", str(self.maximum_candidates)),)

    def plan(
        self,
        problem: LinearCombinatorialProblem,
        certification: CombinatorialCertification,
        /,
    ) -> CombinatorialPlan:
        if not isinstance(problem.space, SetPackingSpace):
            raise TypeError("GreedySetPacking requires SetPackingSpace.")
        if len(jax.tree_util.tree_leaves(problem.costs)) != 1:
            raise ValueError("SetPackingSpace requires one candidate-cost vector.")
        if problem.space.candidate_count > self.maximum_candidates:
            raise ValueError("set-packing candidate count exceeds maximum_candidates.")
        return make_combinatorial_plan(
            problem,
            self,
            certification,
            work_estimate=problem.batch_size * problem.space.candidate_count**2,
            workspace_elements=problem.batch_size
            * (problem.space.candidate_count + problem.space.resource_count),
            certificate_kind="independent-set-relaxation-bound",
        )

    def solve(
        self,
        problem: LinearCombinatorialProblem,
        plan: CombinatorialPlan,
        /,
    ) -> CombinatorialResult:
        space = problem.space
        if not isinstance(space, SetPackingSpace):
            raise TypeError("GreedySetPacking requires SetPackingSpace.")
        raw_costs = jax.tree_util.tree_leaves(problem.costs)[0]
        batch_shape = problem.batch_shape
        flat_batch = problem.batch_size
        costs = raw_costs.reshape((flat_batch, space.candidate_count))
        finite = jnp.all(jnp.isfinite(costs), axis=-1)
        safe_costs = jnp.where(jnp.isfinite(costs), costs, 0.0)
        selected, best, steps = jax.vmap(
            _greedy_one,
            in_axes=(0, None, None, None, None, None),
        )(
            safe_costs,
            space.incidence,
            space.capacities,
            space.valid,
            space.minimum_selected,
            space.maximum_selected,
        )
        decision = SetPackingDecision(
            selected.reshape(batch_shape + (space.candidate_count,))
        )
        features = space.encode(decision).astype(raw_costs.dtype)
        objective = problem.objective(features)
        feasibility = space.audit(decision)
        if space.resource_count:
            individually_admissible = space.valid & jnp.all(
                space.incidence.astype(jnp.int32) <= space.capacities[None, :], axis=-1
            )
        else:
            individually_admissible = space.valid
        lower = jnp.sum(
            jnp.where(
                individually_admissible[None, :] & (safe_costs < 0.0), safe_costs, 0.0
            ),
            axis=-1,
        ).reshape(batch_shape)
        best_shaped = best.reshape(batch_shape)
        finite_shaped = finite.reshape(batch_shape)
        tolerance = plan.certification.threshold(objective, best_shaped, lower)
        objective_consistent = jnp.abs(objective - best_shaped) <= tolerance
        absolute_gap = jnp.maximum(best_shaped - lower, 0.0)
        certified = (
            finite_shaped
            & feasibility.feasible
            & objective_consistent
            & (absolute_gap <= tolerance)
        )
        status = jnp.where(
            ~finite_shaped,
            int(CombinatorialStatus.NONFINITE_INPUT),
            jnp.where(
                ~feasibility.feasible,
                int(CombinatorialStatus.CERTIFICATION_FAILED),
                jnp.where(
                    certified,
                    int(CombinatorialStatus.OPTIMAL),
                    int(CombinatorialStatus.FEASIBLE),
                ),
            ),
        ).astype(jnp.int32)
        valid_result = finite_shaped & feasibility.feasible & objective_consistent
        certificate = CombinatorialCertificate(
            finite=finite_shaped,
            feasible=feasibility.feasible,
            objective_consistent=objective_consistent,
            optimality_proven=certified,
            primal_residual=feasibility.residual.astype(raw_costs.dtype),
            dual_residual=jnp.zeros(batch_shape, dtype=raw_costs.dtype),
            absolute_gap=absolute_gap,
            relative_gap=relative_gap(absolute_gap, best_shaped, lower),
            tie_margin=jnp.full(batch_shape, jnp.nan, dtype=raw_costs.dtype),
            dual_available=jnp.zeros(batch_shape, dtype=bool),
            gap_available=valid_result,
            tie_available=jnp.zeros(batch_shape, dtype=bool),
        )
        provenance = CombinatorialProvenance(
            problem_id=problem.problem_id,
            structure_id=problem.structure_id,
            method_id=self.method_id,
            plan_id=plan.plan_id,
            implementation="phydrax-native-jax",
            tie_policy="lowest-cost-then-lowest-candidate",
            certificate_kind=plan.certificate_kind,
            exact=False,
            signed_costs=True,
            configuration=plan.configuration,
        )
        decision = SetPackingDecision(
            jnp.where(valid_result[..., None], decision.selected, False)
        )
        features = jnp.where(valid_result[..., None], features, jnp.zeros_like(features))
        return CombinatorialResult(
            decision=decision,
            features=features,
            objective_value=jnp.where(valid_result, objective, jnp.nan),
            status=status,
            valid=valid_result,
            certificate=certificate,
            iterations=steps.reshape(batch_shape),
            work=jnp.full(batch_shape, space.candidate_count**2, dtype=jnp.int64),
            provenance=provenance,
            batch_shape=batch_shape,
        )


__all__ = [
    "BranchAndBoundSetPacking",
    "GreedySetPacking",
    "SetPackingDecision",
    "SetPackingSpace",
]
