#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class AdditiveIMEXTableau(StrictModule, NonTrainableState):
    explicit_matrix: Array
    implicit_matrix: Array
    weights: Array
    nodes: Array
    tableau_id: str = eqx.field(static=True)

    def __init__(
        self,
        explicit_matrix: ArrayLike,
        implicit_matrix: ArrayLike,
        weights: ArrayLike,
        nodes: ArrayLike,
        /,
    ):
        explicit = np.asarray(explicit_matrix, dtype=float)
        implicit = np.asarray(implicit_matrix, dtype=float)
        weights_ = np.asarray(weights, dtype=float)
        nodes_ = np.asarray(nodes, dtype=float)
        if (
            explicit.ndim != 2
            or explicit.shape[0] != explicit.shape[1]
            or implicit.shape != explicit.shape
            or weights_.shape != (explicit.shape[0],)
            or explicit.shape[0] == 0
            or not np.all(np.isfinite(explicit))
            or not np.all(np.isfinite(implicit))
            or not np.all(np.isfinite(weights_))
            or not np.all(np.isfinite(nodes_))
            or nodes_.shape != weights_.shape
            or np.any(np.triu(explicit) != 0.0)
            or np.any(np.triu(implicit, k=1) != 0.0)
            or not np.isclose(np.sum(weights_), 1.0)
        ):
            raise ValueError("Additive IMEX tableau is invalid.")
        self.explicit_matrix = jnp.asarray(explicit)
        self.implicit_matrix = jnp.asarray(implicit)
        self.weights = jnp.asarray(weights_)
        self.nodes = jnp.asarray(nodes_)
        self.tableau_id = canonical_fingerprint(
            {
                "kind": "additive-imex-tableau",
                "explicit": explicit.tolist(),
                "implicit": implicit.tolist(),
                "weights": weights_.tolist(),
                "nodes": nodes_.tolist(),
            }
        )

    @property
    def stage_count(self) -> int:
        return int(self.weights.size)

    def step(
        self,
        state: Array,
        time: Array,
        step_size: Array,
        explicit_rhs,
        implicit_solve,
        args=None,
        /,
        *,
        implicit_rhs,
    ) -> Array:
        """Apply the tableau; RHS callbacks take ``(state, time, args)``.

        The implicit RHS is required even when a solve callback is supplied:
        a zero diagonal or zero step does not determine it from a state change.
        """
        state = jnp.asarray(state)
        structures = eqx.filter_eval_shape(
            lambda candidate: (
                jnp.asarray(explicit_rhs(candidate, time, args)),
                jnp.asarray(implicit_rhs(candidate, time, args)),
            ),
            state,
        )
        state = state.astype(
            jnp.result_type(
                state,
                step_size,
                self.weights,
                *(structure.dtype for structure in structures),
            )
        )
        solved_structure = eqx.filter_eval_shape(
            lambda candidate: jnp.asarray(
                implicit_solve(
                    candidate, time, step_size * self.implicit_matrix[0, 0], args
                )
            ),
            state,
        )
        state = state.astype(jnp.result_type(state, solved_structure.dtype))
        explicit_stages = []
        implicit_stages = []
        for stage in range(self.stage_count):
            provisional = state
            for previous in range(stage):
                provisional = provisional + step_size * (
                    self.explicit_matrix[stage, previous] * explicit_stages[previous]
                    + self.implicit_matrix[stage, previous] * implicit_stages[previous]
                )
            stage_time = time + self.nodes[stage] * step_size
            diagonal = self.implicit_matrix[stage, stage]
            coefficient = step_size * diagonal
            solved = jax.lax.cond(
                coefficient != 0.0,
                lambda provisional: jnp.asarray(
                    implicit_solve(provisional, stage_time, coefficient, args),
                    dtype=state.dtype,
                ),
                lambda provisional: provisional,
                provisional,
            )
            explicit_stages.append(explicit_rhs(solved, stage_time, args))
            safe_coefficient = jnp.where(coefficient != 0.0, coefficient, 1.0)
            implicit_stages.append(
                jnp.where(
                    coefficient != 0.0,
                    (solved - provisional) / safe_coefficient,
                    implicit_rhs(solved, stage_time, args),
                )
            )
        result = state
        for stage in range(self.stage_count):
            result = result + step_size * self.weights[stage] * (
                explicit_stages[stage] + implicit_stages[stage]
            )
        return result


class BalanceLawCompositionPlan(StrictModule, NonTrainableState):
    """Symmetric process subcycles; each process owns its finite-update method."""

    process_subcycles: tuple[int, ...] = eqx.field(static=True)
    composition_id: str = eqx.field(static=True)

    def __init__(
        self,
        process_subcycles: tuple[int, ...],
        /,
    ):
        subcycles = tuple(int(value) for value in process_subcycles)
        if not subcycles or any(value <= 0 for value in subcycles):
            raise ValueError("Balance-law multirate composition is invalid.")
        self.process_subcycles = subcycles
        self.composition_id = canonical_fingerprint(
            {
                "kind": "balance-law-symmetric-multirate-composition",
                "process_subcycles": list(subcycles),
            }
        )


__all__ = [
    "AdditiveIMEXTableau",
    "BalanceLawCompositionPlan",
]
