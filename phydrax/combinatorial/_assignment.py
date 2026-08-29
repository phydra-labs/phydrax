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


class AssignmentDecision(StrictModule):
    """Selected zero-based column for every bipartite row."""

    columns: Array


class BipartiteAssignmentSpace(AbstractCombinatorialSpace):
    """Full-row, unit-column-capacity bipartite assignments."""

    valid: Array
    num_rows: int = eqx.field(static=True)
    num_columns: int = eqx.field(static=True)
    _structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        num_rows: int,
        num_columns: int,
        /,
        *,
        valid: Any | None = None,
    ):
        if isinstance(num_rows, bool) or not isinstance(num_rows, Integral):
            raise TypeError("num_rows must be a positive integer.")
        if isinstance(num_columns, bool) or not isinstance(num_columns, Integral):
            raise TypeError("num_columns must be a positive integer.")
        rows = int(num_rows)
        columns = int(num_columns)
        if rows <= 0 or columns <= 0:
            raise ValueError("assignment dimensions must be positive.")
        if valid is None:
            validity = jnp.ones((rows, columns), dtype=bool)
        else:
            validity = jnp.asarray(valid, dtype=bool)
            if validity.shape != (rows, columns):
                raise ValueError(
                    f"valid must have shape {(rows, columns)}; got {validity.shape}."
                )
        self.valid = validity
        self.num_rows = rows
        self.num_columns = columns
        self._structure_id = canonical_fingerprint(
            {
                "kind": "bipartite-assignment-space",
                "rows": rows,
                "columns": columns,
                "valid": array_tree_fingerprint(validity),
            }
        )

    @property
    def structure_id(self) -> str:
        return self._structure_id

    def decision_spec(self, /) -> AssignmentDecision:
        return AssignmentDecision(jax.ShapeDtypeStruct((self.num_rows,), jnp.int32))

    def feature_spec(self, /) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            (self.num_rows, self.num_columns),
            jnp.float32,
        )

    def canonicalize(self, decision: AssignmentDecision, /) -> AssignmentDecision:
        if not isinstance(decision, AssignmentDecision):
            raise TypeError("assignment decisions must be AssignmentDecision values.")
        columns = jnp.asarray(decision.columns, dtype=jnp.int32)
        if columns.shape[-1:] != (self.num_rows,):
            raise ValueError(
                f"assignment columns must end with shape {(self.num_rows,)}; "
                f"got {columns.shape}."
            )
        in_range = (columns >= 0) & (columns < self.num_columns)
        safe = jnp.clip(columns, 0, self.num_columns - 1)
        rows = jnp.arange(self.num_rows, dtype=jnp.int32)
        admissible = in_range & self.valid[rows, safe]
        return AssignmentDecision(jnp.where(admissible, columns, -1))

    def encode(self, decision: AssignmentDecision, /) -> Array:
        canonical = self.canonicalize(decision)
        columns = canonical.columns
        return jax.nn.one_hot(
            columns,
            self.num_columns,
            dtype=float,
            axis=-1,
        )

    def audit(self, decision: AssignmentDecision, /) -> CombinatorialFeasibility:
        canonical = self.canonicalize(decision)
        columns = canonical.columns
        assigned = columns >= 0
        keys = jnp.sort(jnp.where(assigned, columns, self.num_columns), axis=-1)
        if self.num_rows > 1:
            duplicate_count = jnp.sum(
                (keys[..., 1:] == keys[..., :-1]) & (keys[..., 1:] < self.num_columns),
                axis=-1,
            )
        else:
            duplicate_count = jnp.zeros(columns.shape[:-1], dtype=jnp.int32)
        missing_count = jnp.sum(~assigned, axis=-1)
        residual = missing_count + duplicate_count
        return CombinatorialFeasibility(
            residual == 0,
            residual.astype(float),
        )


def _hungarian_one(
    costs: Array, valid: Array, /
) -> tuple[Array, Array, Array, Array, Array]:
    rows, columns = costs.shape
    dtype = costs.dtype
    u = jnp.zeros((rows + 1,), dtype=dtype)
    v = jnp.zeros((columns + 1,), dtype=dtype)
    matching = jnp.zeros((columns + 1,), dtype=jnp.int32)
    feasible = jnp.asarray(rows <= columns)
    total_steps = jnp.asarray(0, dtype=jnp.int32)

    def add_row(row_index, outer):
        feasible_ = outer[3]

        def solve_row(state):
            u_current, v_current, matching_current, feasible_current, total = state
            row = row_index + 1
            matching_current = matching_current.at[0].set(row)
            minimum = jnp.full((columns + 1,), jnp.inf, dtype=dtype)
            used = jnp.zeros((columns + 1,), dtype=bool)
            way = jnp.zeros((columns + 1,), dtype=jnp.int32)
            inner_initial = (
                u_current,
                v_current,
                matching_current,
                minimum,
                used,
                way,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(True),
                jnp.asarray(True),
                jnp.asarray(0, dtype=jnp.int32),
            )

            def condition(inner):
                return inner[7] & (inner[9] <= columns)

            def augment_step(inner):
                (
                    u_inner,
                    v_inner,
                    matching_inner,
                    minimum_inner,
                    used_inner,
                    way_inner,
                    column_zero,
                    active,
                    row_feasible,
                    steps,
                ) = inner
                del active
                used_inner = used_inner.at[column_zero].set(True)
                row_zero = matching_inner[column_zero] - 1
                reduced = costs[row_zero] - u_inner[row_zero + 1] - v_inner[1:]
                candidate = (~used_inner[1:]) & valid[row_zero]
                better = candidate & (reduced < minimum_inner[1:])
                minimum_inner = minimum_inner.at[1:].set(
                    jnp.where(better, reduced, minimum_inner[1:])
                )
                way_inner = way_inner.at[1:].set(
                    jnp.where(better, column_zero, way_inner[1:])
                )
                selectable = (~used_inner[1:]) & jnp.isfinite(minimum_inner[1:])
                masked = jnp.where(selectable, minimum_inner[1:], jnp.inf)
                next_zero = jnp.argmin(masked).astype(jnp.int32) + 1
                delta = masked[next_zero - 1]
                found = jnp.isfinite(delta)
                safe_delta = jnp.where(found, delta, 0.0)
                additions = jnp.where(used_inner, safe_delta, 0.0)
                u_inner = u_inner.at[matching_inner].add(additions)
                v_inner = v_inner - additions
                minimum_inner = jnp.where(
                    ~used_inner,
                    minimum_inner - safe_delta,
                    minimum_inner,
                )
                free = matching_inner[next_zero] == 0
                return (
                    u_inner,
                    v_inner,
                    matching_inner,
                    minimum_inner,
                    used_inner,
                    way_inner,
                    next_zero,
                    found & ~free,
                    row_feasible & found,
                    steps + 1,
                )

            inner = jax.lax.while_loop(condition, augment_step, inner_initial)
            (
                u_current,
                v_current,
                matching_current,
                _,
                _,
                way,
                terminal_column,
                _,
                row_feasible,
                steps,
            ) = inner

            augment_initial = (
                matching_current,
                terminal_column,
                jnp.asarray(0, dtype=jnp.int32),
            )

            def augment_condition(augment):
                return row_feasible & (augment[1] != 0) & (augment[2] <= columns)

            def augment_body(augment):
                matching_aug, column, count = augment
                previous = way[column]
                matching_aug = matching_aug.at[column].set(matching_aug[previous])
                return matching_aug, previous, count + 1

            matching_current, final_column, _ = jax.lax.while_loop(
                augment_condition,
                augment_body,
                augment_initial,
            )
            row_feasible = row_feasible & (final_column == 0)
            return (
                u_current,
                v_current,
                matching_current,
                feasible_current & row_feasible,
                total + steps,
            )

        return jax.lax.cond(feasible_, solve_row, lambda state: state, outer)

    u, v, matching, feasible, total_steps = jax.lax.fori_loop(
        0,
        rows,
        add_row,
        (u, v, matching, feasible, total_steps),
    )
    assigned = jnp.full((rows,), -1, dtype=jnp.int32)

    def assign_column(column, current):
        row = matching[column] - 1
        return jax.lax.cond(
            row >= 0,
            lambda value: value.at[row].set(column - 1),
            lambda value: value,
            current,
        )

    assigned = jax.lax.fori_loop(1, columns + 1, assign_column, assigned)
    feasible = feasible & jnp.all(assigned >= 0)
    return assigned, u[1:], v[1:], feasible, total_steps


class HungarianAssignment(AbstractLinearCombinatorialMethod):
    """Native primal-dual shortest-augmenting-path assignment method."""

    maximum_dimension: int = eqx.field(static=True)

    def __init__(self, *, maximum_dimension: int = 4096):
        if isinstance(maximum_dimension, bool) or not isinstance(
            maximum_dimension, Integral
        ):
            raise TypeError("maximum_dimension must be a positive integer.")
        if int(maximum_dimension) <= 0:
            raise ValueError("maximum_dimension must be positive.")
        self.maximum_dimension = int(maximum_dimension)

    @property
    def method_id(self) -> str:
        return "native-hungarian-assignment"

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
        return (("maximum_dimension", str(self.maximum_dimension)),)

    def plan(
        self,
        problem: LinearCombinatorialProblem,
        certification: CombinatorialCertification,
        /,
    ) -> CombinatorialPlan:
        if not isinstance(problem.space, BipartiteAssignmentSpace):
            raise TypeError("HungarianAssignment requires BipartiteAssignmentSpace.")
        if len(jax.tree_util.tree_leaves(problem.costs)) != 1:
            raise ValueError("BipartiteAssignmentSpace requires one cost matrix.")
        rows = problem.space.num_rows
        columns = problem.space.num_columns
        if max(rows, columns) > self.maximum_dimension:
            raise ValueError(
                f"assignment dimension {max(rows, columns)} exceeds "
                f"maximum_dimension {self.maximum_dimension}."
            )
        return make_combinatorial_plan(
            problem,
            self,
            certification,
            work_estimate=problem.batch_size * rows * rows * columns,
            workspace_elements=problem.batch_size * (4 * columns + 2 * rows + 4),
            certificate_kind="assignment-primal-dual",
        )

    def solve(
        self,
        problem: LinearCombinatorialProblem,
        plan: CombinatorialPlan,
        /,
    ) -> CombinatorialResult:
        space = problem.space
        if not isinstance(space, BipartiteAssignmentSpace):
            raise TypeError("HungarianAssignment requires BipartiteAssignmentSpace.")
        raw_costs = jax.tree_util.tree_leaves(problem.costs)[0]
        batch_shape = problem.batch_shape
        rows = space.num_rows
        columns = space.num_columns
        flat_batch = problem.batch_size
        costs = raw_costs.reshape((flat_batch, rows, columns))
        finite = jnp.all(jnp.isfinite(costs), axis=(1, 2))
        effective_valid = space.valid[None, :, :] & jnp.isfinite(costs)
        assigned, row_dual, column_dual, solved, steps = jax.vmap(_hungarian_one)(
            costs,
            effective_valid,
        )
        assigned = jnp.where(solved[:, None], assigned, -1)
        assigned = assigned.reshape(batch_shape + (rows,))
        decision = AssignmentDecision(assigned)
        features = space.encode(decision).astype(raw_costs.dtype)
        objective = problem.objective(features)
        feasibility = space.audit(decision)

        safe_columns = jnp.clip(assigned.reshape((flat_batch, rows)), 0, columns - 1)
        row_index = jnp.arange(rows, dtype=jnp.int32)[None, :]
        primal = jnp.sum(
            costs[jnp.arange(flat_batch)[:, None], row_index, safe_columns], axis=-1
        )
        dual = jnp.sum(row_dual, axis=-1) + jnp.sum(column_dual, axis=-1)
        reduced_violation = row_dual[:, :, None] + column_dual[:, None, :] - costs
        reduced_violation = jnp.where(effective_valid, reduced_violation, -jnp.inf)
        dual_residual = jnp.maximum(jnp.max(reduced_violation, axis=(1, 2)), 0.0)
        dual_residual = jnp.maximum(
            dual_residual,
            jnp.maximum(jnp.max(column_dual, axis=-1), 0.0),
        )
        assigned_cost = costs[
            jnp.arange(flat_batch)[:, None],
            row_index,
            safe_columns,
        ]
        tightness = jnp.max(
            jnp.abs(
                assigned_cost
                - row_dual
                - jnp.take_along_axis(column_dual, safe_columns, axis=-1)
            ),
            axis=-1,
        )
        dual_residual = jnp.maximum(dual_residual, tightness)
        absolute_gap = jnp.abs(primal - dual)
        tolerance = plan.certification.threshold(primal, dual)
        objective_consistent = (
            jnp.abs(objective.reshape((flat_batch,)) - primal) <= tolerance
        )
        certified = (
            finite
            & solved
            & feasibility.feasible.reshape((flat_batch,))
            & objective_consistent
            & (dual_residual <= tolerance)
            & (absolute_gap <= tolerance)
        )
        status = jnp.where(
            ~finite,
            int(CombinatorialStatus.NONFINITE_INPUT),
            jnp.where(
                ~solved,
                int(CombinatorialStatus.INFEASIBLE),
                jnp.where(
                    certified,
                    int(CombinatorialStatus.OPTIMAL),
                    int(CombinatorialStatus.CERTIFICATION_FAILED),
                ),
            ),
        ).astype(jnp.int32)
        status = status.reshape(batch_shape)
        valid = status == int(CombinatorialStatus.OPTIMAL)
        objective_shaped = objective.reshape(batch_shape)
        certificate = CombinatorialCertificate(
            finite=finite.reshape(batch_shape),
            feasible=feasibility.feasible,
            objective_consistent=objective_consistent.reshape(batch_shape),
            optimality_proven=certified.reshape(batch_shape),
            primal_residual=feasibility.residual.astype(raw_costs.dtype),
            dual_residual=dual_residual.reshape(batch_shape),
            absolute_gap=absolute_gap.reshape(batch_shape),
            relative_gap=relative_gap(absolute_gap, primal, dual).reshape(batch_shape),
            tie_margin=jnp.full(batch_shape, jnp.nan, dtype=raw_costs.dtype),
            dual_available=solved.reshape(batch_shape),
            gap_available=solved.reshape(batch_shape),
            tie_available=jnp.zeros(batch_shape, dtype=bool),
        )
        provenance = CombinatorialProvenance(
            problem_id=problem.problem_id,
            structure_id=problem.structure_id,
            method_id=self.method_id,
            plan_id=plan.plan_id,
            implementation="phydrax-native-jax",
            tie_policy="row-order-then-lowest-column",
            certificate_kind=plan.certificate_kind,
            exact=True,
            signed_costs=True,
            configuration=plan.configuration,
        )
        decision = AssignmentDecision(jnp.where(valid[..., None], decision.columns, -1))
        features = jnp.where(
            valid[..., None, None],
            features,
            jnp.zeros_like(features),
        )
        return CombinatorialResult(
            decision=decision,
            features=features,
            objective_value=jnp.where(valid, objective_shaped, jnp.nan),
            status=status,
            valid=valid,
            certificate=certificate,
            iterations=steps.reshape(batch_shape),
            work=jnp.full(batch_shape, rows * columns, dtype=jnp.int64),
            provenance=provenance,
            batch_shape=batch_shape,
        )


__all__ = [
    "AssignmentDecision",
    "BipartiteAssignmentSpace",
    "HungarianAssignment",
]
