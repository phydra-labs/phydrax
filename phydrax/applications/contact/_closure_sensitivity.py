#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

import phydrax.linalg as la
from phydrax.linalg.svd import SVDSolvePolicy

from ..._strict import StrictModule
from ...discretization.contact._interface import (
    ContactInterfaceKinematics,
    ContactInterfacePlan,
)
from ...discretization.contact._kinematics import ContactKinematicsEpoch
from ...typing import ConvertibleToArray
from ._closure import ContactClosurePlan, evaluate_contact_closure
from ._cone import (
    block_scale,
    certify_contact_cone_impulse,
    classify_contact_cone_branches,
    contact_cone_numeric_revision_matches,
    contact_cone_result_is_certified,
    ContactConeBranch,
    ContactConeNumericRevision,
    ContactConeProgram,
    ContactConeResult,
)
from ._mortar import (
    evaluate_mortar_contact,
    MortarContactPlan,
    MortarContactState,
)
from ._route_state import ContactRouteState


class ContactDerivativeEvidence(StrictModule):
    """Branch qualification and refusal evidence of one contact derivative.

    ``primal_successful`` is the current certification of the differentiated
    primal (including stale-revision refusal); ``successful`` additionally
    requires branch qualification, a finite derivative and, for cone
    derivatives, a natively certified nonsingular fixed-branch Jacobian
    (``jacobian_regular``), a successful implicit linear solve and no friction
    derivative across a frictionless Coulomb boundary
    (``friction_boundary_refused``).
    """

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
    linear_successful: Array | None = None
    jacobian_regular: Array | None = None
    regularity_margin: Array | None = None
    friction_boundary_refused: Array | None = None


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
    plan: ContactClosurePlan,
    kinematics: ContactKinematicsEpoch,
    state: ContactRouteState,
    gaps: tuple[Array, ...],
) -> tuple[Array, Array]:
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
    derivative_values: PyTree[Array],
    primal_successful: ConvertibleToArray,
    source_id: str,
    margin_tolerance: float,
) -> ContactDerivativeEvidence:
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


def _parameter_arrays(
    program: ContactConeProgram,
    response_action: ConvertibleToArray | None,
    free_velocity: ConvertibleToArray | None,
    normal_bias: ConvertibleToArray | None,
    compliance: ConvertibleToArray | None,
    static_friction: ConvertibleToArray | None,
    dynamic_friction: ConvertibleToArray | None,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Validate structured parameter/action arrays against one cone program."""

    if not isinstance(program, ContactConeProgram):
        raise TypeError("program must be ContactConeProgram.")
    dtype = program.free_velocity.dtype

    def value(argument: ConvertibleToArray | None, like: Array, name: str) -> Array:
        if argument is None:
            return jnp.zeros_like(like)
        array = jnp.asarray(argument, dtype=dtype)
        if array.shape != like.shape:
            raise ValueError(f"{name} has invalid shape.")
        return array

    return (
        value(response_action, program.free_velocity, "response_action"),
        value(free_velocity, program.free_velocity, "free_velocity"),
        value(normal_bias, program.normal_bias, "normal_bias"),
        value(compliance, program.compliance, "compliance"),
        value(static_friction, program.static_friction, "static_friction"),
        value(dynamic_friction, program.dynamic_friction, "dynamic_friction"),
    )


class ContactConeTangent(StrictModule):
    """Structured parameter/action tangent of one cone program.

    ``response_action`` is the response tangent applied to the solution
    impulse, ``(dW) p*``.  Geometry and inertia parameters enter through it and
    through the free velocity computed by their owners.
    """

    response_action: Array
    free_velocity: Array
    normal_bias: Array
    compliance: Array
    static_friction: Array
    dynamic_friction: Array

    def __init__(
        self,
        program: ContactConeProgram,
        /,
        *,
        response_action: ConvertibleToArray | None = None,
        free_velocity: ConvertibleToArray | None = None,
        normal_bias: ConvertibleToArray | None = None,
        compliance: ConvertibleToArray | None = None,
        static_friction: ConvertibleToArray | None = None,
        dynamic_friction: ConvertibleToArray | None = None,
    ) -> None:
        (
            self.response_action,
            self.free_velocity,
            self.normal_bias,
            self.compliance,
            self.static_friction,
            self.dynamic_friction,
        ) = _parameter_arrays(
            program,
            response_action,
            free_velocity,
            normal_bias,
            compliance,
            static_friction,
            dynamic_friction,
        )


class ContactConeCotangent(StrictModule):
    """Structured parameter/action cotangent returned by the cone VJP.

    The ``response_action`` cotangent ``m`` represents ``dW -> m · (dW p*)``.
    Static friction does not enter fixed-branch equations; its cotangent is
    exactly zero on every admitted branch.  When ``friction_computed`` is
    ``False`` the friction cotangents were not requested and are ``None``.
    """

    response_action: Array
    free_velocity: Array
    normal_bias: Array
    compliance: Array
    static_friction: Array | None
    dynamic_friction: Array | None
    friction_computed: bool = eqx.field(static=True)

    def __init__(
        self,
        program: ContactConeProgram,
        /,
        *,
        response_action: ConvertibleToArray | None = None,
        free_velocity: ConvertibleToArray | None = None,
        normal_bias: ConvertibleToArray | None = None,
        compliance: ConvertibleToArray | None = None,
        static_friction: ConvertibleToArray | None = None,
        dynamic_friction: ConvertibleToArray | None = None,
        friction_computed: bool = True,
    ) -> None:
        computed = bool(friction_computed)
        if not computed and (static_friction is not None or dynamic_friction is not None):
            raise ValueError("Uncomputed friction cotangents must be None.")
        (
            self.response_action,
            self.free_velocity,
            self.normal_bias,
            self.compliance,
            static,
            dynamic,
        ) = _parameter_arrays(
            program,
            response_action,
            free_velocity,
            normal_bias,
            compliance,
            static_friction,
            dynamic_friction,
        )
        self.static_friction = static if computed else None
        self.dynamic_friction = dynamic if computed else None
        self.friction_computed = computed


class ContactConeJVP(StrictModule):
    """Fixed-branch cone JVP: impulse tangent with derivative evidence.

    ``impulse_tangent`` is exactly zero unless ``evidence.successful``;
    ``cone_margin`` is the smallest block branch margin in certificate-tolerance
    units.
    """

    impulse_tangent: Array
    cone_margin: Array
    evidence: ContactDerivativeEvidence


class ContactConeVJP(StrictModule):
    """Fixed-branch cone VJP: structured parameter/action cotangent.

    ``cotangent`` is exactly zero unless ``evidence.successful``; ``cone_margin``
    is the smallest block branch margin in certificate-tolerance units.
    """

    cotangent: ContactConeCotangent
    cone_margin: Array
    evidence: ContactDerivativeEvidence


class ContactConeFixedBranch(StrictModule):
    """Fixed contact branch of one certified cone solution.

    Pure classification without linearization: per-block branch labels and
    margins, the congruent block scale, the active component mask, the
    contacting/sticking/sliding/frictionless-boundary block masks and the
    current certification of the solve.
    """

    branch: Array
    margins: Array
    cone_margin: Array
    scale: Array
    mask: Array
    contacting: Array
    sticking: Array
    sliding: Array
    frictionless_boundary: Array
    certified: Array
    program_id: str = eqx.field(static=True)


def fix_contact_cone_branch(
    program: ContactConeProgram, result: ContactConeResult, /
) -> ContactConeFixedBranch:
    """Classify the fixed branch of ``result`` against the current ``program``."""

    if not isinstance(program, ContactConeProgram):
        raise TypeError("program must be ContactConeProgram.")
    if not isinstance(result, ContactConeResult):
        raise TypeError("result must be ContactConeResult.")
    certificate = certify_contact_cone_impulse(program, result.impulse)
    branch, margins = classify_contact_cone_branches(program, result.impulse, certificate)
    sliding = branch == int(ContactConeBranch.SLIDING)
    sticking = branch == int(ContactConeBranch.STICKING)
    frictionless = branch == int(ContactConeBranch.FRICTIONLESS_BOUNDARY)
    return ContactConeFixedBranch(
        branch,
        margins,
        jnp.min(margins, initial=jnp.inf),
        block_scale(program),
        program.layout.component_mask & program.valid[:, None],
        sliding
        | sticking
        | (branch == int(ContactConeBranch.NORMAL_ACTIVE))
        | frictionless,
        sticking,
        sliding,
        frictionless,
        contact_cone_result_is_certified(program, result),
        program.program_id,
    )


def contact_cone_fixed_branch_residual(
    fixed: ContactConeFixedBranch,
    scaled_impulse: Array,
    scaled_law_velocity: Array,
    friction: Array,
    /,
) -> Array:
    """Fixed-branch block law residual in scaled impulse/law-velocity variables.

    Contacting blocks impose a zero normal law velocity, sticking blocks a zero
    tangential law velocity, sliding blocks the dynamic Coulomb boundary map
    ``p_t = μ a_n a_t / |a_t|`` with ``a = p - w``; every other component (and
    every padded or invalid component) imposes ``p = 0``.  The law velocity is
    supplied by the caller, so the same branch law serves the cone response and
    endpoint laws such as ``g(q1)/h``.
    """

    if not isinstance(fixed, ContactConeFixedBranch):
        raise TypeError("fixed must be ContactConeFixedBranch.")
    tiny = jnp.finfo(scaled_impulse.dtype).tiny
    mask = fixed.mask
    argument = scaled_impulse - scaled_law_velocity
    normal = argument[:, 0]
    tangent = jnp.where(mask[:, 1:], argument[:, 1:], 0.0)
    tangent_norm = jnp.sqrt(jnp.maximum(jnp.sum(tangent**2, axis=-1), tiny))
    boundary = (friction * normal / tangent_norm)[:, None] * tangent
    normal_residual = jnp.where(
        fixed.contacting, scaled_law_velocity[:, 0], scaled_impulse[:, 0]
    )
    tangent_residual = jnp.where(
        fixed.sticking[:, None],
        scaled_law_velocity[:, 1:],
        jnp.where(
            fixed.sliding[:, None],
            scaled_impulse[:, 1:] - boundary,
            scaled_impulse[:, 1:],
        ),
    )
    value = jnp.concatenate((normal_residual[:, None], tangent_residual), axis=-1)
    return jnp.where(mask, value, scaled_impulse)


class PreparedContactConeSensitivity(StrictModule):
    """One prepared fixed-branch linearization shared by every JVP and VJP.

    The preparation is bound to the exact numeric revision and impulse of the
    result it linearizes.  Reuse re-checks that binding and the current
    program revision on device; a stale preparation yields refused derivatives.
    Geometry admission and the margin requirement are reapplied from the
    arguments of every derivative call.

    ``regularity`` is the native rank certificate of the matrix-free
    fixed-branch Jacobian, computed only under the caller's explicit bounded
    ``SVDSolvePolicy``; ``jacobian_regular`` holds when that certificate proves
    full, locally fixed rank.  Without a certificate no derivative is admitted.
    """

    numeric_revision: ContactConeNumericRevision
    impulse: Array
    fixed_branch: ContactConeFixedBranch
    linearization: la.PreparedLinearization
    jacobian: la.AbstractLinearOperator
    regularity: la.RectangularRankCertificate | None
    jacobian_regular: Array
    regularity_margin: Array
    policy: la.LinearSolvePolicy
    program_id: str = eqx.field(static=True)


def prepare_contact_cone_sensitivity(
    program: ContactConeProgram,
    result: ContactConeResult,
    /,
    *,
    regularity: SVDSolvePolicy | None = None,
    linear_tolerance: float = 1.0e-12,
    linear_iterations: int = 400,
) -> PreparedContactConeSensitivity:
    """Classify the branch and prepare the fixed-branch residual linearization.

    ``regularity`` is an explicit bounded native SVD policy used by
    :func:`phydrax.linalg.certify_rectangular_rank` to prove that the
    fixed-branch Jacobian is nonsingular with a locally fixed rank.  With
    ``None``, or when the certificate does not prove full fixed rank (for
    example sticking blocks whose tangential response rows vanish, or
    duplicated support witnesses), every derivative is refused.  A successful
    single-right-hand-side Krylov solve is never taken as regularity evidence.
    """

    if not isinstance(program, ContactConeProgram):
        raise TypeError("program must be ContactConeProgram.")
    if not isinstance(result, ContactConeResult):
        raise TypeError("result must be ContactConeResult.")
    if regularity is not None and not isinstance(regularity, SVDSolvePolicy):
        raise TypeError("regularity must be SVDSolvePolicy or None.")
    fixed = fix_contact_cone_branch(program, result)
    scale = fixed.scale
    dtype = program.free_velocity.dtype

    def residual(point: tuple[Array, Array, Array]) -> Array:
        scaled_impulse, law_free, friction = point
        impulse = scale[:, None] * scaled_impulse
        law = scale[:, None] * (
            program.response.mv(impulse) + program.compliance * impulse + law_free
        )
        return contact_cone_fixed_branch_residual(fixed, scaled_impulse, law, friction)

    point = (
        result.impulse / scale[:, None],
        program.law_free_velocity,
        program.dynamic_friction,
    )
    linearization = la.prepare_linearization(residual, point)
    space = program.contact_space
    zero_law = jnp.zeros_like(program.law_free_velocity)
    zero_friction = jnp.zeros_like(program.dynamic_friction)

    def jacobian_action(direction: Array) -> Array:
        return linearization.jvp((direction, zero_law, zero_friction))

    def jacobian_transpose(direction: Array) -> Array:
        return linearization.vjp(direction)[0]

    jacobian = la.FunctionLinearOperator(
        jacobian_action,
        source=space,
        target=space,
        transpose_action=jacobian_transpose,
        operator_id=f"{program.program_id}/fixed-branch-jacobian",
    )
    if regularity is None:
        rank_certificate = None
        jacobian_regular = jnp.asarray(False)
        regularity_margin = jnp.asarray(jnp.nan, dtype=dtype)
    else:
        rank_certificate = la.certify_rectangular_rank(jacobian, regularity)
        jacobian_regular = rank_certificate.fixed_rank & (
            rank_certificate.rank == space.size
        )
        regularity_margin = (
            rank_certificate.gap_margin - rank_certificate.singular_value_error_bound
        ).astype(dtype)
    policy = la.LinearSolvePolicy(
        la.GMRES(),
        tolerance=la.TolerancePolicy(
            relative=float(linear_tolerance),
            absolute=0.0,
            max_steps=int(linear_iterations),
        ),
    )
    return PreparedContactConeSensitivity(
        result.evidence.numeric_revision,
        result.impulse,
        fixed,
        linearization,
        jacobian,
        rank_certificate,
        jacobian_regular,
        regularity_margin,
        policy,
        program.program_id,
    )


def contact_cone_fixed_branch_admitted(
    program: ContactConeProgram,
    fixed: ContactConeFixedBranch,
    /,
    *,
    margin_factor: float,
    friction_derivative: bool,
    geometry_admissible: ConvertibleToArray | None = None,
) -> Array:
    """Whether the fixed branch admits derivatives under current requirements.

    Requires the current certification of the solve, a classified branch for
    every valid block, geometry admission, a branch margin above
    ``margin_factor`` certificate tolerances, and, when ``friction_derivative``
    is requested, no frictionless Coulomb block in contact.  Jacobian
    regularity is a separate certificate.
    """

    if not isinstance(fixed, ContactConeFixedBranch):
        raise TypeError("fixed must be ContactConeFixedBranch.")
    if fixed.program_id != program.program_id:
        raise ValueError("Fixed branch belongs to another cone program.")
    admitted = fixed.certified & _qualified(
        program, fixed, margin_factor, geometry_admissible
    )
    if friction_derivative:
        admitted = admitted & ~jnp.any(_friction_boundary(program, fixed))
    return admitted


def _qualified(
    program: ContactConeProgram,
    fixed: ContactConeFixedBranch,
    margin_factor: float,
    geometry_admissible: ConvertibleToArray | None,
    /,
) -> Array:
    """Branch qualification from the current geometry and margin requirement."""

    factor = float(margin_factor)
    if not factor >= 0.0:
        raise ValueError("margin_factor must be nonnegative.")
    admissible = (
        jnp.ones(program.valid.shape, dtype=jnp.bool_)
        if geometry_admissible is None
        else jnp.asarray(geometry_admissible, dtype=jnp.bool_)
    )
    if admissible.shape != program.valid.shape:
        raise ValueError("geometry_admissible must have block shape.")
    classified = (~program.valid) | (fixed.branch != int(ContactConeBranch.UNCLASSIFIED))
    return (
        jnp.all(classified)
        & jnp.all((~program.valid) | admissible)
        & (fixed.cone_margin > factor)
    )


def _friction_boundary(
    program: ContactConeProgram, fixed: ContactConeFixedBranch, /
) -> Array:
    """Valid frictionless Coulomb blocks in contact (friction-parameter boundary)."""

    return program.valid & fixed.frictionless_boundary


def _derivative_evidence(
    program: ContactConeProgram,
    result: ContactConeResult,
    fixed: PreparedContactConeSensitivity,
    qualified: Array,
    finite: Array,
    linear_successful: Array,
    friction_boundary_refused: Array,
    /,
) -> ContactDerivativeEvidence:
    return ContactDerivativeEvidence(
        fixed.fixed_branch.cone_margin,
        fixed.fixed_branch.certified,
        finite,
        qualified,
        _admitted(fixed, qualified, friction_boundary_refused)
        & finite
        & linear_successful,
        program.program_id,
        numeric_revision=result.evidence.numeric_revision,
        route_keys=program.route_keys,
        route_mask=program.valid,
        branch_classification=fixed.fixed_branch.branch,
        branch_margins=fixed.fixed_branch.margins,
        linear_successful=linear_successful,
        jacobian_regular=fixed.jacobian_regular,
        regularity_margin=fixed.regularity_margin,
        friction_boundary_refused=friction_boundary_refused,
    )


def _admitted(
    fixed: PreparedContactConeSensitivity,
    qualified: Array,
    friction_boundary_refused: Array,
    /,
) -> Array:
    """Admission proven before any derivative action or linear solve runs."""

    return (
        fixed.fixed_branch.certified
        & qualified
        & fixed.jacobian_regular
        & ~friction_boundary_refused
    )


def _sensitivity(
    program: ContactConeProgram,
    result: ContactConeResult,
    sensitivity: PreparedContactConeSensitivity | None,
    regularity: SVDSolvePolicy | None,
    linear_tolerance: float,
    linear_iterations: int,
    /,
) -> PreparedContactConeSensitivity:
    if sensitivity is None:
        return prepare_contact_cone_sensitivity(
            program,
            result,
            regularity=regularity,
            linear_tolerance=linear_tolerance,
            linear_iterations=linear_iterations,
        )
    if not isinstance(sensitivity, PreparedContactConeSensitivity):
        raise TypeError("sensitivity must be PreparedContactConeSensitivity or None.")
    if regularity is not None:
        raise ValueError(
            "regularity is fixed when the sensitivity is prepared; omit it on reuse."
        )
    if sensitivity.program_id != program.program_id:
        raise ValueError("Prepared sensitivity belongs to another cone program.")
    revision_equal = eqx.tree_equal(
        sensitivity.numeric_revision, result.evidence.numeric_revision
    )
    impulse_comparable = (
        sensitivity.impulse.shape == result.impulse.shape
    ) and revision_equal is not False
    current = (
        contact_cone_numeric_revision_matches(program, result)
        & jnp.asarray(revision_equal)
        & (
            jnp.all(sensitivity.impulse == result.impulse)
            if impulse_comparable
            else jnp.asarray(False)
        )
    )
    return eqx.tree_at(
        lambda value: value.fixed_branch.certified,
        sensitivity,
        sensitivity.fixed_branch.certified & current,
    )


def contact_cone_solution_jvp(
    program: ContactConeProgram,
    result: ContactConeResult,
    tangent: ContactConeTangent,
    /,
    *,
    sensitivity: PreparedContactConeSensitivity | None = None,
    regularity: SVDSolvePolicy | None = None,
    margin_factor: float = 1.0,
    geometry_admissible: ConvertibleToArray | None = None,
    linear_tolerance: float = 1.0e-12,
    linear_iterations: int = 400,
) -> ContactConeJVP:
    """Differentiate a certified cone solution on its fixed contact branch.

    The implicit equation is the same block law as the primal; the Jacobian is
    applied matrix-free from one prepared linearization and solved with
    restarted GMRES.  A derivative is admitted only with a native regularity
    certificate of that Jacobian (``regularity`` when preparing here, or the
    one stored in ``sensitivity``).  Stale, uncertified, apex,
    stick/slip-boundary, geometrically inadmissible, insufficient-margin and
    unproven-regularity cases return a zero tangent with refusal evidence, as
    does a friction tangent on a frictionless Coulomb block in contact, whose
    friction derivative is one-sided.  ``margin_factor`` and
    ``geometry_admissible`` are applied on every call, including reuse.
    """

    if not isinstance(tangent, ContactConeTangent):
        raise TypeError("tangent must be ContactConeTangent.")
    fixed = _sensitivity(
        program,
        result,
        sensitivity,
        regularity,
        linear_tolerance,
        linear_iterations,
    )
    qualified = _qualified(
        program, fixed.fixed_branch, margin_factor, geometry_admissible
    )
    friction_refused = jnp.any(
        _friction_boundary(program, fixed.fixed_branch)
        & ((tangent.dynamic_friction != 0.0) | (tangent.static_friction != 0.0))
    )
    law_tangent = (
        (
            tangent.response_action
            + tangent.compliance * result.impulse
            + tangent.free_velocity
        )
        .at[:, 0]
        .add(tangent.normal_bias)
    )
    admitted = _admitted(fixed, qualified, friction_refused)

    def run(_: None) -> tuple[Array, Array]:
        parameter = fixed.linearization.jvp(
            (jnp.zeros_like(result.impulse), law_tangent, tangent.dynamic_friction)
        )
        solution = la.solve(
            la.LinearSystem(fixed.jacobian), -parameter, policy=fixed.policy
        )
        return (
            fixed.fixed_branch.scale[:, None] * solution.value,
            jnp.asarray(solution.successful),
        )

    def skip(_: None) -> tuple[Array, Array]:
        return jnp.zeros_like(result.impulse), jnp.asarray(False)

    candidate, linear_successful = jax.lax.cond(admitted, run, skip, None)
    finite = jnp.all(jnp.isfinite(candidate))
    evidence = _derivative_evidence(
        program,
        result,
        fixed,
        qualified,
        finite,
        linear_successful,
        friction_refused,
    )
    return ContactConeJVP(
        jnp.where(evidence.successful, candidate, 0.0),
        fixed.fixed_branch.cone_margin,
        evidence,
    )


def contact_cone_solution_vjp(
    program: ContactConeProgram,
    result: ContactConeResult,
    impulse_cotangent: ConvertibleToArray,
    /,
    *,
    sensitivity: PreparedContactConeSensitivity | None = None,
    regularity: SVDSolvePolicy | None = None,
    margin_factor: float = 1.0,
    geometry_admissible: ConvertibleToArray | None = None,
    friction_cotangent: bool = True,
    linear_tolerance: float = 1.0e-12,
    linear_iterations: int = 400,
) -> ContactConeVJP:
    """Transpose of :func:`contact_cone_solution_jvp` from the same linearization.

    With ``friction_cotangent=True`` the friction cotangents are returned and
    the VJP is refused when a frictionless Coulomb block is in contact (its
    friction derivative is one-sided).  With ``False`` the friction cotangents
    are not computed and are ``None``.
    """

    fixed = _sensitivity(
        program,
        result,
        sensitivity,
        regularity,
        linear_tolerance,
        linear_iterations,
    )
    qualified = _qualified(
        program, fixed.fixed_branch, margin_factor, geometry_admissible
    )
    friction_refused = (
        jnp.any(_friction_boundary(program, fixed.fixed_branch))
        if friction_cotangent
        else jnp.asarray(False)
    )
    cotangent = jnp.asarray(impulse_cotangent, dtype=program.free_velocity.dtype)
    if cotangent.shape != result.impulse.shape:
        raise ValueError("impulse_cotangent must have the impulse shape.")
    admitted = _admitted(fixed, qualified, friction_refused)

    def run(_: None) -> tuple[Array, Array, Array]:
        adjoint = la.solve(
            la.LinearSystem(la.transpose(fixed.jacobian)),
            fixed.fixed_branch.scale[:, None] * cotangent,
            policy=fixed.policy,
        )
        _, law_bar, friction_bar = fixed.linearization.vjp(-adjoint.value)
        return law_bar, friction_bar, jnp.asarray(adjoint.successful)

    def skip(_: None) -> tuple[Array, Array, Array]:
        return (
            jnp.zeros_like(program.law_free_velocity),
            jnp.zeros_like(program.dynamic_friction),
            jnp.asarray(False),
        )

    law_cotangent, friction_value, linear_successful = jax.lax.cond(
        admitted, run, skip, None
    )
    finite = jnp.all(jnp.isfinite(law_cotangent)) & (
        jnp.all(jnp.isfinite(friction_value)) if friction_cotangent else jnp.asarray(True)
    )
    evidence = _derivative_evidence(
        program,
        result,
        fixed,
        qualified,
        finite,
        linear_successful,
        friction_refused,
    )
    accepted = evidence.successful
    law_cotangent = jnp.where(accepted, law_cotangent, 0.0)
    return ContactConeVJP(
        ContactConeCotangent(
            program,
            response_action=law_cotangent,
            free_velocity=law_cotangent,
            normal_bias=law_cotangent[:, 0],
            compliance=law_cotangent * result.impulse,
            dynamic_friction=(
                jnp.where(accepted, friction_value, 0.0) if friction_cotangent else None
            ),
            friction_computed=friction_cotangent,
        ),
        fixed.fixed_branch.cone_margin,
        evidence,
    )


class MortarGapJVP(StrictModule):
    traction_tangent: Array
    evidence: ContactDerivativeEvidence


def mortar_gap_jvp(
    plan: MortarContactPlan,
    interface: ContactInterfacePlan,
    kinematics: ContactInterfaceKinematics,
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

    def traction(gap_value: Array) -> Array:
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
    "ContactConeCotangent",
    "ContactConeFixedBranch",
    "ContactConeJVP",
    "ContactConeTangent",
    "ContactConeVJP",
    "ContactDerivativeEvidence",
    "MortarGapJVP",
    "contact_closure_gap_jvp",
    "contact_closure_gap_vjp",
    "contact_cone_fixed_branch_admitted",
    "contact_cone_fixed_branch_residual",
    "contact_cone_solution_jvp",
    "contact_cone_solution_vjp",
    "mortar_gap_jvp",
    "fix_contact_cone_branch",
    "prepare_contact_cone_sensitivity",
    "PreparedContactConeSensitivity",
]
