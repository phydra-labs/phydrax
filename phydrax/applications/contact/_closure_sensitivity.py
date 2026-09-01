#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ...discretization.contact._kinematics import ContactKinematicsEpoch
from ._closure import ContactClosurePlan, evaluate_contact_closure
from ._cone import (
    ContactConeProgram,
    ContactConeSolverPlan,
    solve_contact_cone,
)
from ._mortar import (
    evaluate_mortar_contact,
    MortarContactPlan,
    MortarContactState,
)
from ._route_state import ContactRouteState


class ContactDerivativeEvidence(StrictModule):
    branch_margin: Array
    primal_successful: Array
    derivative_finite: Array
    branch_qualified: Array
    successful: Array
    source_id: str = eqx.field(static=True)


class ContactClosureGapJVP(StrictModule):
    potential_tangent: Array
    traction_tangent: Array
    evidence: ContactDerivativeEvidence


class ContactClosureGapVJP(StrictModule):
    gap_cotangent: tuple[Array, ...]
    evidence: ContactDerivativeEvidence


def _replace_gaps(
    kinematics: ContactKinematicsEpoch, gaps: tuple[Array, ...], /
) -> ContactKinematicsEpoch:
    if len(gaps) != len(kinematics.batches):
        raise ValueError("Contact gap tuple does not match kinematics batches.")
    batches = tuple(
        eqx.tree_at(lambda value: value.gap, batch, gap)
        for batch, gap in zip(kinematics.batches, gaps, strict=True)
    )
    return eqx.tree_at(lambda value: value.batches, kinematics, batches)


def _closure_gap_outputs(
    plan,
    kinematics,
    state,
    gaps,
):
    replaced = _replace_gaps(kinematics, gaps)
    evaluation = evaluate_contact_closure(plan, replaced, state)
    traction = (
        jnp.concatenate(tuple(batch.normal.traction for batch in evaluation.batches))
        if evaluation.batches
        else jnp.empty((0,), dtype=state.accumulated_slip.dtype)
    )
    return evaluation.evidence.total_potential, traction


def _branch_evidence(
    kinematics: ContactKinematicsEpoch,
    derivative_values,
    primal_successful,
    source_id,
    margin_tolerance,
):
    margins = []
    for batch in kinematics.batches:
        margins.append(
            jnp.min(
                jnp.where(
                    batch.valid,
                    jnp.minimum(jnp.abs(batch.gap), batch.feature_margin),
                    jnp.inf,
                ),
                initial=jnp.inf,
            )
        )
    branch_margin = (
        jnp.min(jnp.stack(tuple(margins))) if margins else jnp.asarray(jnp.inf)
    )
    leaves = jax.tree.leaves(derivative_values)
    finite = jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))
    qualified = branch_margin > margin_tolerance
    return ContactDerivativeEvidence(
        branch_margin,
        jnp.asarray(primal_successful),
        finite,
        qualified,
        jnp.asarray(primal_successful) & finite & qualified,
        source_id,
    )


def contact_closure_gap_jvp(
    plan: ContactClosurePlan,
    kinematics: ContactKinematicsEpoch,
    state: ContactRouteState,
    tangent_gaps: tuple[Array, ...],
    /,
    *,
    margin_tolerance: float = 1.0e-10,
) -> ContactClosureGapJVP:
    gaps = tuple(batch.gap for batch in kinematics.batches)
    if len(tangent_gaps) != len(gaps):
        raise ValueError("Contact tangent gap tuple has invalid length.")
    primal = evaluate_contact_closure(plan, kinematics, state)
    _, tangent = jax.jvp(
        lambda values: _closure_gap_outputs(plan, kinematics, state, values),
        (gaps,),
        (tangent_gaps,),
    )
    evidence = _branch_evidence(
        kinematics,
        tangent,
        primal.evidence.successful,
        plan.closure_id,
        margin_tolerance,
    )
    return ContactClosureGapJVP(tangent[0], tangent[1], evidence)


def contact_closure_gap_vjp(
    plan: ContactClosurePlan,
    kinematics: ContactKinematicsEpoch,
    state: ContactRouteState,
    potential_cotangent: Array,
    traction_cotangent: Array,
    /,
    *,
    margin_tolerance: float = 1.0e-10,
) -> ContactClosureGapVJP:
    gaps = tuple(batch.gap for batch in kinematics.batches)
    primal = evaluate_contact_closure(plan, kinematics, state)
    _, pullback = jax.vjp(
        lambda values: _closure_gap_outputs(plan, kinematics, state, values),
        gaps,
    )
    cotangent = pullback((potential_cotangent, traction_cotangent))[0]
    evidence = _branch_evidence(
        kinematics,
        cotangent,
        primal.evidence.successful,
        plan.closure_id,
        margin_tolerance,
    )
    return ContactClosureGapVJP(cotangent, evidence)


class ContactConeJVP(StrictModule):
    impulse_tangent: Array
    cone_margin: Array
    evidence: ContactDerivativeEvidence


def contact_cone_solution_jvp(
    program: ContactConeProgram,
    tangent_free_velocity: Array,
    tangent_effective_mass: Array,
    /,
    *,
    solver: ContactConeSolverPlan | None = None,
    margin_tolerance: float = 1.0e-10,
) -> ContactConeJVP:
    solver_ = ContactConeSolverPlan() if solver is None else solver
    free = program.free_velocity
    matrix = program.effective_mass
    tangent_free = jnp.asarray(tangent_free_velocity, dtype=free.dtype)
    tangent_matrix = jnp.asarray(tangent_effective_mass, dtype=matrix.dtype)
    if tangent_free.shape != free.shape or tangent_matrix.shape != matrix.shape:
        raise ValueError("Contact cone tangent shapes are invalid.")

    def solution(free_value, matrix_value):
        changed = eqx.tree_at(
            lambda value: (
                value.free_velocity,
                value.effective_mass,
            ),
            program,
            (free_value, matrix_value),
        )
        return solve_contact_cone(changed, solver=solver_).impulse

    primal = solve_contact_cone(program, solver=solver_)
    _, tangent = jax.jvp(
        solution,
        (free, matrix),
        (tangent_free, tangent_matrix),
    )
    normal = primal.impulse[:, 0]
    tangent_impulse = primal.impulse[:, 1:]
    tangent_norm = jnp.sqrt(jnp.sum(tangent_impulse * tangent_impulse, axis=-1))
    cone_margin = jnp.min(
        jnp.where(
            program.valid,
            jnp.minimum(
                normal,
                program.friction * normal - tangent_norm,
            ),
            jnp.inf,
        ),
        initial=jnp.inf,
    )
    finite = jnp.all(jnp.isfinite(tangent))
    evidence = ContactDerivativeEvidence(
        cone_margin,
        primal.evidence.successful,
        finite,
        cone_margin > margin_tolerance,
        primal.evidence.successful & finite & (cone_margin > margin_tolerance),
        program.program_id,
    )
    return ContactConeJVP(tangent, cone_margin, evidence)


class MortarGapJVP(StrictModule):
    traction_tangent: Array
    evidence: ContactDerivativeEvidence


def mortar_gap_jvp(
    plan: MortarContactPlan,
    interface,
    kinematics,
    state: MortarContactState,
    tangent_gap: Array,
    /,
    *,
    margin_tolerance: float = 1.0e-10,
) -> MortarGapJVP:
    gap = kinematics.gap
    tangent = jnp.asarray(tangent_gap, dtype=gap.dtype)
    if tangent.shape != gap.shape:
        raise ValueError("Mortar tangent gap has invalid shape.")

    def traction(gap_value):
        changed = eqx.tree_at(lambda value: value.gap, kinematics, gap_value)
        return evaluate_mortar_contact(plan, interface, changed, state).traction

    primal = evaluate_mortar_contact(plan, interface, kinematics, state)
    _, traction_tangent = jax.jvp(traction, (gap,), (tangent,))
    margin = jnp.min(
        jnp.where(interface.valid, jnp.abs(gap), jnp.inf),
        initial=jnp.inf,
    )
    finite = jnp.all(jnp.isfinite(traction_tangent))
    evidence = ContactDerivativeEvidence(
        margin,
        primal.evidence.successful,
        finite,
        margin > margin_tolerance,
        primal.evidence.successful & finite & (margin > margin_tolerance),
        plan.plan_id,
    )
    return MortarGapJVP(traction_tangent, evidence)


__all__ = [
    "ContactClosureGapJVP",
    "ContactClosureGapVJP",
    "ContactConeJVP",
    "ContactDerivativeEvidence",
    "MortarGapJVP",
    "contact_closure_gap_jvp",
    "contact_closure_gap_vjp",
    "contact_cone_solution_jvp",
    "mortar_gap_jvp",
]
