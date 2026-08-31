#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._self_gravity import PreparedNewtonianSelfGravity


class CompositeAMRGravityDiagnostics(StrictModule):
    coarse_residual: Array
    fine_residual: Array
    interface_potential_defect: Array
    converged: Array


class CompositeAMRGravityPlan(StrictModule, NonTrainableState):
    coarse: PreparedNewtonianSelfGravity
    fine: PreparedNewtonianSelfGravity
    refinement_ratio: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse: PreparedNewtonianSelfGravity,
        fine: PreparedNewtonianSelfGravity,
        /,
        *,
        refinement_ratio: int = 2,
    ):
        ratio = int(refinement_ratio)
        if (
            not isinstance(coarse, PreparedNewtonianSelfGravity)
            or not isinstance(fine, PreparedNewtonianSelfGravity)
            or ratio <= 1
            or any(
                fine_count != ratio * coarse_count
                for fine_count, coarse_count in zip(
                    fine.cell_shape, coarse.cell_shape, strict=True
                )
            )
        ):
            raise ValueError("Composite AMR gravity hierarchy is invalid.")
        self.coarse = coarse
        self.fine = fine
        self.refinement_ratio = ratio
        self.plan_id = canonical_fingerprint(
            {
                "kind": "composite-amr-gravity",
                "coarse": coarse.process_id,
                "fine": fine.process_id,
                "refinement_ratio": ratio,
            }
        )

    def solve(
        self,
        fine_density: ArrayLike,
        args=None,
        /,
    ) -> tuple[Array, Array, CompositeAMRGravityDiagnostics]:
        density = jnp.asarray(fine_density)
        ratio = self.refinement_ratio
        coarse_density = density
        for axis in reversed(range(density.ndim)):
            shape = (
                coarse_density.shape[:axis]
                + (self.coarse.cell_shape[axis], ratio)
                + coarse_density.shape[axis + 1 :]
            )
            coarse_density = jnp.mean(coarse_density.reshape(shape), axis=axis + 1)
        coarse_potential, _, _, coarse_result = self.coarse.solve_density(
            coarse_density, args
        )
        fine_potential, _, _, fine_result = self.fine.solve_density(density, args)
        prolonged = coarse_potential
        for axis in range(density.ndim):
            prolonged = jnp.repeat(prolonged, ratio, axis=axis)
        interface_defect = jnp.max(jnp.abs(fine_potential - prolonged), initial=0.0)
        diagnostics = CompositeAMRGravityDiagnostics(
            coarse_residual=coarse_result.residual_norm,
            fine_residual=fine_result.residual_norm,
            interface_potential_defect=interface_defect,
            converged=coarse_result.converged & fine_result.converged,
        )
        return coarse_potential, fine_potential, diagnostics


class AMRTopologyEpoch(StrictModule, NonTrainableState):
    active_cells: Array
    parent_cells: Array
    level_ids: Array
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        active_cells: ArrayLike,
        parent_cells: ArrayLike,
        level_ids: ArrayLike,
        /,
    ):
        active = np.asarray(active_cells, dtype=bool)
        parents = np.asarray(parent_cells, dtype=np.int32)
        levels = np.asarray(level_ids, dtype=np.int32)
        if (
            active.ndim != 1
            or parents.shape != active.shape
            or levels.shape != active.shape
        ):
            raise ValueError("AMR topology epoch arrays are invalid.")
        self.active_cells = jnp.asarray(active)
        self.parent_cells = jnp.asarray(parents)
        self.level_ids = jnp.asarray(levels)
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "amr-topology-epoch",
                "active": array_tree_fingerprint(active),
                "parents": array_tree_fingerprint(parents),
                "levels": array_tree_fingerprint(levels),
            }
        )


class AMRTopologyReplayPlan(StrictModule, NonTrainableState):
    epochs: tuple[AMRTopologyEpoch, ...]
    transition_steps: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        epochs: tuple[AMRTopologyEpoch, ...],
        transition_steps: tuple[int, ...],
        /,
    ):
        epochs_ = tuple(epochs)
        transitions = tuple(int(value) for value in transition_steps)
        if (
            not epochs_
            or len(transitions) != len(epochs_)
            or any(value < 0 for value in transitions)
            or tuple(sorted(transitions)) != transitions
            or len(set(transitions)) != len(transitions)
        ):
            raise ValueError("AMR topology replay schedule is invalid.")
        self.epochs = epochs_
        self.transition_steps = transitions
        self.plan_id = canonical_fingerprint(
            {
                "kind": "amr-topology-replay",
                "epochs": [epoch.epoch_id for epoch in epochs_],
                "transition_steps": list(transitions),
            }
        )

    def epoch(self, accepted_step: int, /) -> AMRTopologyEpoch:
        index = max(
            position
            for position, transition in enumerate(self.transition_steps)
            if transition <= int(accepted_step)
        )
        return self.epochs[index]


__all__ = [
    "AMRTopologyEpoch",
    "AMRTopologyReplayPlan",
    "CompositeAMRGravityDiagnostics",
    "CompositeAMRGravityPlan",
]
