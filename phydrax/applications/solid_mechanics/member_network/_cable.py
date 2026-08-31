#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._blocks import (
    AxialMemberBlock,
    CorotationalFrameBlock,
    DiscreteRodBlock,
    TensionOnlyCableLaw,
)
from ._equilibrium import (
    MemberNetworkResult,
    MemberNetworkStatus,
    PreparedMemberNetworkSolve,
    refresh_member_network,
    solve_member_network,
)


class CableActiveSetPolicy(StrictModule, NonTrainableState):
    activation_tolerance: float = eqx.field(static=True)
    deactivation_tolerance: float = eqx.field(static=True)
    strict_complementarity_tolerance: float = eqx.field(static=True)
    maximum_active_set_changes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        activation_tolerance: float = 1.0e-10,
        deactivation_tolerance: float = 1.0e-10,
        strict_complementarity_tolerance: float = 1.0e-7,
        maximum_active_set_changes: int = 50,
    ):
        values = (
            float(activation_tolerance),
            float(deactivation_tolerance),
            float(strict_complementarity_tolerance),
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Cable active-set tolerances must be finite and nonnegative."
            )
        if values[2] <= max(values[:2]):
            raise ValueError(
                "Strict-complementarity tolerance must exceed switching tolerances."
            )
        if int(maximum_active_set_changes) <= 0:
            raise ValueError("maximum_active_set_changes must be positive.")
        (
            self.activation_tolerance,
            self.deactivation_tolerance,
            self.strict_complementarity_tolerance,
        ) = values
        self.maximum_active_set_changes = int(maximum_active_set_changes)


class CableSlacknessResult(StrictModule):
    equilibrium: MemberNetworkResult
    initial_active: Array
    final_active: Array
    activated: Array
    deactivated: Array
    active_set_changes: Array
    ambiguous: Array
    complementarity_residual: Array
    derivative_mode: Literal["fixed-active-set", "ambiguous-active-set"] = eqx.field(
        static=True
    )
    successful: Array


def solve_cable_slackness(
    prepared: PreparedMemberNetworkSolve,
    /,
    *,
    initial_active: ArrayLike | None = None,
    policy: CableActiveSetPolicy | None = None,
) -> CableSlacknessResult:
    """Solve fixed-mask smooth roots until the exact unilateral active set closes."""
    policy_ = CableActiveSetPolicy() if policy is None else policy
    problem = prepared.plan.problem
    unilateral = jnp.zeros((problem.definition.structure.member_count,), dtype=bool)
    for block in problem.assembly.blocks:
        if isinstance(block, AxialMemberBlock):
            law = block.law
        elif isinstance(block, (CorotationalFrameBlock, DiscreteRodBlock)):
            law = block.axial_law
        else:
            continue
        if isinstance(law, TensionOnlyCableLaw):
            unilateral = unilateral.at[block.member_indices].set(True)
    initial = (
        prepared.inputs.cable_active
        if initial_active is None
        else jnp.asarray(initial_active, dtype=bool)
    )
    if initial.shape != unilateral.shape:
        raise ValueError("initial_active must match the member axis.")
    applied_norm = jnp.sqrt(
        jnp.sum(prepared.inputs.nodal_forces**2)
        + jnp.sum(prepared.inputs.nodal_moments**2)
    )
    active = jnp.where(
        unilateral & (applied_norm > policy_.activation_tolerance),
        True,
        initial,
    )
    current = prepared
    kinematics = problem.definition.dofs.expand(
        prepared.initial_reduced,
        prepared.inputs.prescribed_positions,
        prepared.inputs.prescribed_rotations,
    )
    total_changes = jnp.asarray(0, dtype=jnp.int32)
    equilibrium = solve_member_network(current)
    cycle = False
    history: set[tuple[bool, ...]] = set()
    for _ in range(policy_.maximum_active_set_changes):
        inputs = eqx.tree_at(
            lambda selected: selected.cable_active,
            current.inputs,
            active,
        )
        current = refresh_member_network(current, inputs, kinematics)
        equilibrium = solve_member_network(current)
        kinematics = equilibrium.state.kinematics
        structure = problem.definition.structure
        vectors = (
            kinematics.positions[structure.receivers]
            - kinematics.positions[structure.senders]
        )
        lengths = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1))
        effective = inputs.rest_lengths * (1.0 + inputs.initial_strain)
        extension = lengths - effective
        next_active = jnp.where(
            unilateral,
            jnp.where(
                active,
                extension > policy_.deactivation_tolerance,
                extension > policy_.activation_tolerance,
            ),
            active,
        )
        changed = next_active != active
        total_changes = total_changes + jnp.sum(changed, dtype=jnp.int32)
        key = tuple(bool(value) for value in next_active.tolist())
        if key in history and bool(jnp.any(changed)):
            cycle = True
            break
        history.add(key)
        active = next_active
        if not bool(jnp.any(changed)):
            break
    final_active = active
    activated = ~initial & final_active
    deactivated = initial & ~final_active
    ambiguous = unilateral & (
        equilibrium.diagnostics.active_set.switching_margin
        <= policy_.strict_complementarity_tolerance
    )
    derivative_mode = (
        "ambiguous-active-set" if bool(jnp.any(ambiguous)) else "fixed-active-set"
    )
    successful = (
        equilibrium.status != int(MemberNetworkStatus.NONLINEAR_SOLVE_FAILED)
    ) & ~jnp.asarray(cycle)
    return CableSlacknessResult(
        equilibrium,
        initial,
        final_active,
        activated,
        deactivated,
        total_changes,
        ambiguous,
        equilibrium.diagnostics.active_set.complementarity_residual,
        derivative_mode,
        successful,
    )


__all__ = [
    "CableActiveSetPolicy",
    "CableSlacknessResult",
    "solve_cable_slackness",
]
