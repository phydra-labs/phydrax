#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array


def hungarian_assignment_one(
    costs: Array, valid: Array, /
) -> tuple[Array, Array, Array, Array, Array]:
    """Solve one finite rectangular full-row assignment with deterministic ties."""

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


__all__ = ["hungarian_assignment_one"]
