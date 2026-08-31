#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
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
    ) -> Array:
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
            solved = implicit_solve(
                provisional,
                stage_time,
                step_size * diagonal,
                args,
            )
            explicit_stages.append(explicit_rhs(solved, stage_time, args))
            implicit_stages.append(
                jnp.where(
                    diagonal != 0.0,
                    (solved - provisional) / (step_size * diagonal),
                    jnp.zeros_like(state),
                )
            )
        result = state
        for stage in range(self.stage_count):
            result = result + step_size * self.weights[stage] * (
                explicit_stages[stage] + implicit_stages[stage]
            )
        return result


BalanceLawIntegrationMode: TypeAlias = Literal[
    "explicit",
    "exact",
    "implicit",
    "stochastic_exact",
]


class BalanceLawCompositionPlan(StrictModule, NonTrainableState):
    """Static symmetric multirate composition for prepared source processes."""

    process_subcycles: tuple[int, ...] = eqx.field(static=True)
    integration_modes: tuple[BalanceLawIntegrationMode, ...] = eqx.field(static=True)
    composition_id: str = eqx.field(static=True)

    def __init__(
        self,
        process_subcycles: tuple[int, ...],
        /,
        *,
        integration_modes: tuple[BalanceLawIntegrationMode, ...] | None = None,
    ):
        subcycles = tuple(int(value) for value in process_subcycles)
        modes = (
            tuple("explicit" for _ in subcycles)
            if integration_modes is None
            else tuple(integration_modes)
        )
        if (
            not subcycles
            or any(value <= 0 for value in subcycles)
            or len(modes) != len(subcycles)
            or any(
                mode not in ("explicit", "exact", "implicit", "stochastic_exact")
                for mode in modes
            )
        ):
            raise ValueError("Balance-law multirate composition is invalid.")
        self.process_subcycles = subcycles
        self.integration_modes = modes
        self.composition_id = canonical_fingerprint(
            {
                "kind": "balance-law-symmetric-multirate-composition",
                "process_subcycles": list(subcycles),
                "integration_modes": list(modes),
            }
        )


__all__ = [
    "AdditiveIMEXTableau",
    "BalanceLawCompositionPlan",
    "BalanceLawIntegrationMode",
]
