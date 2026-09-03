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
    _contact_law_diagnostics,
    contact_cone_result_is_certified,
    ContactConeNumericRevision,
    ContactConeProgram,
    ContactConeResult,
    project_coulomb_cone,
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
    numeric_revision: ContactConeNumericRevision | None = None
    route_keys: Array | None = None
    route_mask: Array | None = None
    branch_classification: Array | None = None
    branch_margins: Array | None = None




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
    result: ContactConeResult,
    tangent_free_velocity: Array,
    tangent_effective_mass: Array,
    /,
    *,
    margin_tolerance: float = 1.0e-10,
) -> ContactConeJVP:
    """Differentiate a certified cone solution without changing its contact branch."""

    if not isinstance(program, ContactConeProgram):
        raise TypeError("program must be ContactConeProgram.")
    if not isinstance(result, ContactConeResult):
        raise TypeError("result must be ContactConeResult.")
    free = program.free_velocity
    matrix = program.effective_mass
    tangent_free = jnp.asarray(tangent_free_velocity, dtype=free.dtype)
    tangent_matrix = jnp.asarray(tangent_effective_mass, dtype=matrix.dtype)
    if tangent_free.shape != free.shape or tangent_matrix.shape != matrix.shape:
        raise ValueError("Contact cone tangent shapes are invalid.")

    revision = result.evidence.numeric_revision
    current = contact_cone_result_is_certified(program, result)
    zero_tangent = jnp.zeros_like(result.impulse)
    inactive_margin = jnp.where(program.valid, 0.0, jnp.inf)
    unclassified = jnp.zeros(program.valid.shape, dtype=jnp.int8)
    if not bool(current):
        evidence = ContactDerivativeEvidence(
            jnp.min(inactive_margin, initial=jnp.inf),
            current,
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(False),
            program.program_id,
            numeric_revision=revision,
            route_keys=program.route_keys,
            route_mask=program.valid,
            branch_classification=unclassified,
            branch_margins=inactive_margin,
        )
        return ContactConeJVP(
            zero_tangent,
            evidence.branch_margin,
            evidence,
        )

    certificate_tolerance = jnp.maximum(
        result.evidence.certificate_tolerance,
        jnp.asarray(0.0, dtype=free.dtype),
    )
    qualification_tolerance = jnp.maximum(
        certificate_tolerance,
        jnp.asarray(margin_tolerance, dtype=free.dtype),
    )
    normal_impulse = result.impulse[:, 0]
    tangent_impulse = result.impulse[:, 1:]
    normal_velocity = result.contact_law_velocity[:, 0]
    tangent_velocity = result.contact_law_velocity[:, 1:]
    tangent_impulse_norm = jnp.sqrt(
        jnp.sum(tangent_impulse * tangent_impulse, axis=-1)
    )
    tangent_velocity_norm = jnp.sqrt(
        jnp.sum(tangent_velocity * tangent_velocity, axis=-1)
    )
    static_slack = (
        program.static_friction * normal_impulse - tangent_impulse_norm
    )
    contacting = normal_impulse > certificate_tolerance
    separating = (
        (normal_impulse <= certificate_tolerance)
        & (normal_velocity > certificate_tolerance)
    )
    if program.tangent_dimension == 0:
        sticking = contacting
        sticking_margin = normal_impulse
    else:
        sticking = (
            contacting
            & (tangent_velocity_norm <= certificate_tolerance)
            & (static_slack > certificate_tolerance)
        )
        sticking_margin = jnp.minimum(normal_impulse, static_slack)
    sliding = contacting & (tangent_velocity_norm > certificate_tolerance)
    branch_classification = jnp.where(
        program.valid & separating,
        1,
        jnp.where(
            program.valid & sticking,
            2,
            jnp.where(program.valid & sliding, 3, 0),
        ),
    ).astype(jnp.int8)
    branch_margins = jnp.where(
        ~program.valid,
        jnp.inf,
        jnp.where(
            branch_classification == 1,
            normal_velocity,
            jnp.where(
                branch_classification == 2,
                sticking_margin,
                jnp.where(
                    branch_classification == 3,
                    jnp.minimum(normal_impulse, tangent_velocity_norm),
                    0.0,
                ),
            ),
        ),
    )
    cone_margin = jnp.min(branch_margins, initial=jnp.inf)
    branch_qualified = (
        jnp.all((~program.valid) | (branch_classification > 0))
        & (cone_margin > qualification_tolerance)
    )
    if not bool(branch_qualified):
        evidence = ContactDerivativeEvidence(
            cone_margin,
            current,
            jnp.asarray(False),
            branch_qualified,
            jnp.asarray(False),
            program.program_id,
            numeric_revision=revision,
            route_keys=program.route_keys,
            route_mask=program.valid,
            branch_classification=branch_classification,
            branch_margins=branch_margins,
        )
        return ContactConeJVP(zero_tangent, cone_margin, evidence)

    selected_friction = jnp.where(
        branch_classification == 2,
        program.static_friction,
        program.friction,
    )
    sliding_routes = branch_classification == 3
    sticking_routes = branch_classification == 2

    def fixed_branch_residual(flat_impulse, free_value, matrix_value):
        impulse = flat_impulse.reshape(result.impulse.shape)
        changed = eqx.tree_at(
            lambda value: (
                value.free_velocity,
                value.effective_mass,
            ),
            program,
            (free_value, matrix_value),
        )
        law_velocity = _contact_law_diagnostics(changed, impulse)[0]
        projection_argument = impulse - law_velocity
        safe_projection_argument = jnp.where(
            sliding_routes[:, None],
            projection_argument,
            jnp.ones_like(projection_argument),
        )
        sliding_residual = impulse - project_coulomb_cone(
            safe_projection_argument,
            selected_friction,
        )
        residual = jnp.where(sticking_routes[:, None], law_velocity, impulse)
        residual = jnp.where(
            sliding_routes[:, None],
            sliding_residual,
            residual,
        )
        return residual.reshape((-1,))

    flat_impulse = result.impulse.reshape((-1,))
    impulse_jacobian = jax.jacfwd(
        lambda value: fixed_branch_residual(value, free, matrix)
    )(flat_impulse)
    _, parameter_tangent = jax.jvp(
        lambda free_value, matrix_value: fixed_branch_residual(
            flat_impulse,
            free_value,
            matrix_value,
        ),
        (free, matrix),
        (tangent_free, tangent_matrix),
    )
    candidate_tangent = -jnp.linalg.solve(
        impulse_jacobian,
        parameter_tangent,
    ).reshape(result.impulse.shape)
    finite = jnp.all(jnp.isfinite(candidate_tangent))
    successful = current & finite & branch_qualified
    impulse_tangent = jnp.where(
        successful,
        candidate_tangent,
        jnp.zeros_like(candidate_tangent),
    )
    evidence = ContactDerivativeEvidence(
        cone_margin,
        current,
        finite,
        branch_qualified,
        successful,
        program.program_id,
        numeric_revision=revision,
        route_keys=program.route_keys,
        route_mask=program.valid,
        branch_classification=branch_classification,
        branch_margins=branch_margins,
    )
    return ContactConeJVP(impulse_tangent, cone_margin, evidence)


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
