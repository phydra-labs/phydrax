#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._kinematics import ContactKinematicsEpoch
from ._materials import ContactMaterialPairTable


class ContactConeProgram(StrictModule, NonTrainableState):
    """Fixed-route compliant Signorini--Coulomb impulse program."""

    free_velocity: Array
    effective_mass: Array
    compliance: Array
    friction: Array
    route_keys: Array
    valid: Array
    tangent_dimension: int = eqx.field(static=True)
    program_id: str = eqx.field(static=True)
    static_friction: Array
    restitution: Array
    mechanical_available: Array

    def __init__(
        self,
        free_velocity: ArrayLike,
        effective_mass: ArrayLike,
        compliance: ArrayLike,
        friction: ArrayLike,
        route_keys: ArrayLike,
        valid: ArrayLike,
        tangent_dimension: int,
        program_id: str,
        /,
        *,
        static_friction: ArrayLike | None = None,
        restitution: ArrayLike | None = None,
        mechanical_available: ArrayLike | None = None,
    ):
        free = jnp.asarray(free_velocity)
        effective = jnp.asarray(effective_mass, dtype=free.dtype)
        compliance_ = jnp.asarray(compliance, dtype=free.dtype)
        dynamic = jnp.asarray(friction, dtype=free.dtype)
        keys = jnp.asarray(route_keys)
        route_mask = jnp.asarray(valid, dtype=bool)
        tangent = int(tangent_dimension)
        local_dimension = 1 + tangent
        count = int(route_mask.size)
        size = count * local_dimension
        static = (
            dynamic
            if static_friction is None
            else jnp.asarray(static_friction, dtype=free.dtype)
        )
        restitution_ = (
            jnp.zeros_like(dynamic)
            if restitution is None
            else jnp.asarray(restitution, dtype=free.dtype)
        )
        available = (
            jnp.ones_like(route_mask)
            if mechanical_available is None
            else jnp.asarray(mechanical_available, dtype=bool)
        )
        if tangent < 0:
            raise ValueError("tangent_dimension must be nonnegative.")
        if free.shape != (count, local_dimension):
            raise ValueError("free_velocity has invalid fixed-route shape.")
        if effective.shape != (size, size):
            raise ValueError("effective_mass has invalid flattened shape.")
        if compliance_.shape != (size,):
            raise ValueError("compliance has invalid flattened shape.")
        if dynamic.shape != (count,) or static.shape != (count,):
            raise ValueError("Friction coefficients must match the route capacity.")
        if restitution_.shape != (count,) or available.shape != (count,):
            raise ValueError("Contact law data must match the route capacity.")
        if keys.shape[0] != count:
            raise ValueError("route_keys must match the route capacity.")
        identifier = str(program_id)
        if not identifier:
            raise ValueError("program_id must be non-empty.")
        self.free_velocity = free
        self.effective_mass = effective
        self.compliance = compliance_
        self.friction = dynamic
        self.route_keys = keys
        self.valid = route_mask
        self.tangent_dimension = tangent
        self.program_id = identifier
        self.static_friction = static
        self.restitution = restitution_
        self.mechanical_available = available

    @property
    def contact_count(self) -> int:
        return int(self.valid.size)

    @property
    def local_dimension(self) -> int:
        return 1 + self.tangent_dimension


class ContactConeSolverPlan(StrictModule, NonTrainableState):
    maximum_iterations: int = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int = 200,
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-8,
        relaxation: float = 1.0,
    ):
        iterations = int(maximum_iterations)
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        relaxation_ = float(relaxation)
        if iterations <= 0:
            raise ValueError("maximum_iterations must be positive.")
        if absolute <= 0.0 or relative < 0.0:
            raise ValueError("Cone solver tolerances are invalid.")
        if not 0.0 < relaxation_ <= 1.0:
            raise ValueError("relaxation must lie in (0, 1].")
        self.maximum_iterations = iterations
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.relaxation = relaxation_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "contact-cone-solver-plan",
                "maximum_iterations": iterations,
                "absolute_tolerance": absolute.hex(),
                "relative_tolerance": relative.hex(),
                "relaxation": relaxation_.hex(),
            }
        )


class ContactConeNumericRevision(StrictModule):
    """Exact numeric inputs and solver policy that produced one cone result."""

    free_velocity: Array
    effective_mass: Array
    compliance: Array
    static_friction: Array
    dynamic_friction: Array
    restitution: Array
    route_mask: Array
    mechanical_available: Array
    solver_parameters: Array
    program_id: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)


class ContactConeEvidence(StrictModule):
    converged: Array
    iterations: Array
    projected_residual: Array
    complementarity_defect: Array
    cone_defect: Array
    minimum_normal_impulse: Array
    dissipated_impulse_work: Array
    finite: Array
    successful: Array
    program_id: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    material_law_complete: Array
    minimum_normal_velocity: Array
    maximum_dissipation_defect: Array
    certificate_tolerance: Array
    dissipative: Array
    numeric_revision: ContactConeNumericRevision | None

    def __init__(
        self,
        converged,
        iterations,
        projected_residual,
        complementarity_defect,
        cone_defect,
        minimum_normal_impulse,
        dissipated_impulse_work,
        finite,
        successful,
        program_id,
        solver_id,
        /,
        *,
        material_law_complete=True,
        minimum_normal_velocity=0.0,
        maximum_dissipation_defect=0.0,
        certificate_tolerance=0.0,
        dissipative=True,
        numeric_revision: ContactConeNumericRevision | None = None,
    ):
        self.converged = jnp.asarray(converged, dtype=bool)
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32)
        self.projected_residual = jnp.asarray(projected_residual)
        self.complementarity_defect = jnp.asarray(complementarity_defect)
        self.cone_defect = jnp.asarray(cone_defect)
        self.minimum_normal_impulse = jnp.asarray(minimum_normal_impulse)
        self.dissipated_impulse_work = jnp.asarray(dissipated_impulse_work)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.successful = jnp.asarray(successful, dtype=bool)
        self.program_id = str(program_id)
        self.solver_id = str(solver_id)
        self.material_law_complete = jnp.asarray(material_law_complete, dtype=bool)
        self.minimum_normal_velocity = jnp.asarray(minimum_normal_velocity)
        self.maximum_dissipation_defect = jnp.asarray(maximum_dissipation_defect)
        self.certificate_tolerance = jnp.asarray(certificate_tolerance)
        self.dissipative = jnp.asarray(dissipative, dtype=bool)
        self.numeric_revision = numeric_revision


class ContactConeResult(StrictModule):
    """Accepted impulse plus the candidate retained for failure diagnosis."""

    impulse: Array
    post_relative_velocity: Array
    evidence: ContactConeEvidence
    candidate_impulse: Array
    candidate_post_relative_velocity: Array
    contact_law_velocity: Array
    candidate_contact_law_velocity: Array

    def __init__(
        self,
        impulse,
        post_relative_velocity,
        evidence,
        /,
        *,
        candidate_impulse=None,
        candidate_post_relative_velocity=None,
        contact_law_velocity=None,
        candidate_contact_law_velocity=None,
    ):
        accepted = jnp.asarray(impulse)
        post = jnp.asarray(post_relative_velocity, dtype=accepted.dtype)
        if not isinstance(evidence, ContactConeEvidence):
            raise TypeError("evidence must be ContactConeEvidence.")
        candidate = (
            accepted
            if candidate_impulse is None
            else jnp.asarray(candidate_impulse, dtype=accepted.dtype)
        )
        candidate_post = (
            post
            if candidate_post_relative_velocity is None
            else jnp.asarray(candidate_post_relative_velocity, dtype=accepted.dtype)
        )
        law = (
            post
            if contact_law_velocity is None
            else jnp.asarray(contact_law_velocity, dtype=accepted.dtype)
        )
        candidate_law = (
            candidate_post
            if candidate_contact_law_velocity is None
            else jnp.asarray(candidate_contact_law_velocity, dtype=accepted.dtype)
        )
        if (
            post.shape != accepted.shape
            or candidate.shape != accepted.shape
            or candidate_post.shape != accepted.shape
            or law.shape != accepted.shape
            or candidate_law.shape != accepted.shape
        ):
            raise ValueError("Cone result arrays must have identical shapes.")
        self.impulse = accepted
        self.post_relative_velocity = post
        self.evidence = evidence
        self.candidate_impulse = candidate
        self.candidate_post_relative_velocity = candidate_post
        self.contact_law_velocity = law
        self.candidate_contact_law_velocity = candidate_law


def project_signorini_coulomb_product(
    impulse: ArrayLike,
    friction: ArrayLike,
    /,
) -> Array:
    """Project the Signorini normal and Coulomb friction ball product."""

    value = jnp.asarray(impulse)
    coefficient = jnp.asarray(friction, dtype=value.dtype)
    normal = jnp.maximum(value[..., 0], 0.0)
    tangent = value[..., 1:]
    tangent_norm = jnp.sqrt(jnp.sum(tangent * tangent, axis=-1))
    limit = coefficient * normal
    safe_norm = jnp.maximum(tangent_norm, jnp.finfo(value.dtype).tiny)
    scale = jnp.minimum(1.0, limit / safe_norm)
    projected_tangent = scale[..., None] * tangent
    return jnp.concatenate((normal[..., None], projected_tangent), axis=-1)


def build_contact_cone_program(
    kinematics: ContactKinematicsEpoch,
    materials: ContactMaterialPairTable,
    effective_mass: ArrayLike,
    /,
    *,
    compliance: ArrayLike = 0.0,
) -> ContactConeProgram:
    if not isinstance(kinematics, ContactKinematicsEpoch):
        raise TypeError("kinematics must be ContactKinematicsEpoch.")
    if not isinstance(materials, ContactMaterialPairTable):
        raise TypeError("materials must be ContactMaterialPairTable.")
    if not kinematics.batches:
        raise ValueError("Contact cone program requires positive route capacity.")
    tangent_dimension = int(kinematics.batches[0].tangential_velocity.shape[-1])
    if any(
        batch.tangential_velocity.shape[-1] != tangent_dimension
        for batch in kinematics.batches
    ):
        raise ValueError("Contact cone tangent dimensions disagree.")
    normal_velocity = jnp.concatenate(
        tuple(batch.normal_velocity for batch in kinematics.batches)
    )
    tangential_velocity = jnp.concatenate(
        tuple(batch.tangential_velocity for batch in kinematics.batches)
    )
    left_material = jnp.concatenate(
        tuple(batch.left_material_ids for batch in kinematics.batches)
    )
    right_material = jnp.concatenate(
        tuple(batch.right_material_ids for batch in kinematics.batches)
    )
    route_keys = jnp.concatenate(tuple(batch.route_keys for batch in kinematics.batches))
    valid = jnp.concatenate(tuple(batch.valid for batch in kinematics.batches))
    parameters = materials.lookup(left_material, right_material)
    target_normal_velocity = -parameters.restitution * jnp.minimum(normal_velocity, 0.0)
    free = jnp.concatenate(
        (
            (normal_velocity - target_normal_velocity)[:, None],
            tangential_velocity,
        ),
        axis=-1,
    )
    contact_count = int(valid.size)
    local_dimension = 1 + tangent_dimension
    size = contact_count * local_dimension
    effective = jnp.asarray(effective_mass, dtype=free.dtype)
    if effective.shape == (contact_count, local_dimension, local_dimension):
        block = jnp.zeros((size, size), dtype=free.dtype)
        for index in range(contact_count):
            start = index * local_dimension
            block = block.at[
                start : start + local_dimension,
                start : start + local_dimension,
            ].set(effective[index])
        effective = block
    if effective.shape != (size, size):
        raise ValueError(
            "effective_mass must be flattened or one local block per contact."
        )
    compliance_ = jnp.asarray(compliance, dtype=free.dtype)
    if compliance_.shape == ():
        compliance_ = jnp.full((contact_count,), compliance_)
    if compliance_.shape == (contact_count,):
        compliance_ = jnp.repeat(compliance_, local_dimension)
    if compliance_.shape != (size,):
        raise ValueError("Cone compliance must match the flattened program.")
    program_id = canonical_fingerprint(
        {
            "kind": "contact-cone-program",
            "kinematics": kinematics.epoch_id,
            "materials": materials.table_id,
            "contacts": contact_count,
            "local_dimension": local_dimension,
        }
    )
    return ContactConeProgram(
        free,
        effective,
        compliance_,
        parameters.dynamic_friction,
        route_keys,
        valid,
        tangent_dimension,
        program_id,
        static_friction=parameters.static_friction,
        restitution=parameters.restitution,
        mechanical_available=parameters.mechanical_available,
    )


def _iterate_contact_law(
    matrix: Array,
    free: Array,
    initial_impulse: Array,
    friction: Array,
    valid: Array,
    solver: ContactConeSolverPlan,
    tolerance: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    row_bound = jnp.max(jnp.sum(jnp.abs(matrix), axis=-1), initial=0.0)
    step = jnp.where(
        row_bound > jnp.finfo(matrix.dtype).eps,
        1.0 / row_bound,
        jnp.asarray(1.0, dtype=matrix.dtype),
    )
    route_mask = valid[:, None]
    initial = jnp.where(
        route_mask,
        project_signorini_coulomb_product(initial_impulse, friction),
        0.0,
    )

    def body(index, state):
        value, converged, first_converged, residual_norm = state
        gradient = (matrix @ value.reshape((-1,)) + free).reshape(value.shape)
        trial = value - solver.relaxation * step * gradient
        projected = project_signorini_coulomb_product(trial, friction)
        projected = jnp.where(route_mask, projected, 0.0)
        residual = value - projected
        norm = jnp.sqrt(jnp.sum(residual * residual))
        now = norm <= tolerance
        first = jnp.where((~converged) & now, index + 1, first_converged)
        next_value = jnp.where(converged, value, projected)
        return next_value, converged | now, first, norm

    return jax.lax.fori_loop(
        0,
        solver.maximum_iterations,
        body,
        (
            initial,
            jnp.asarray(False),
            jnp.asarray(solver.maximum_iterations, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=free.dtype),
        ),
    )


def _contact_law_diagnostics(
    program: ContactConeProgram,
    impulse: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    matrix = program.effective_mass + jnp.diag(program.compliance)
    flat_impulse = impulse.reshape((-1,))
    law_velocity = (matrix @ flat_impulse + program.free_velocity.reshape((-1,))).reshape(
        impulse.shape
    )
    normal_impulse = impulse[:, 0]
    tangent_impulse = impulse[:, 1:]
    normal_velocity = law_velocity[:, 0]
    tangent_velocity = law_velocity[:, 1:]
    tangent_impulse_norm = jnp.sqrt(jnp.sum(tangent_impulse * tangent_impulse, axis=-1))
    tangent_velocity_norm = jnp.sqrt(
        jnp.sum(tangent_velocity * tangent_velocity, axis=-1)
    )
    static_limit = program.static_friction * jnp.maximum(normal_impulse, 0.0)
    cone_defect = jnp.max(
        jnp.where(
            program.valid,
            jnp.maximum(tangent_impulse_norm - static_limit, 0.0),
            0.0,
        ),
        initial=0.0,
    )
    complementarity = jnp.max(
        jnp.where(
            program.valid,
            jnp.abs(jnp.minimum(normal_impulse, normal_velocity)),
            0.0,
        ),
        initial=0.0,
    )
    safe_slip = jnp.maximum(tangent_velocity_norm, jnp.finfo(impulse.dtype).tiny)
    sliding_impulse = (-program.friction * jnp.maximum(normal_impulse, 0.0) / safe_slip)[
        :, None
    ] * tangent_velocity
    sliding_defect = jnp.sqrt(
        jnp.sum(
            (tangent_impulse - sliding_impulse) * (tangent_impulse - sliding_impulse),
            axis=-1,
        )
    )
    sticking_defect = jnp.maximum(
        tangent_velocity_norm,
        jnp.maximum(tangent_impulse_norm - static_limit, 0.0),
    )
    maximum_dissipation_defect = jnp.max(
        jnp.where(
            program.valid,
            jnp.minimum(sticking_defect, sliding_defect),
            0.0,
        ),
        initial=0.0,
    )
    active = jnp.any(program.valid)
    minimum_impulse = jnp.where(
        active,
        jnp.min(jnp.where(program.valid, normal_impulse, jnp.inf)),
        jnp.asarray(0.0, dtype=impulse.dtype),
    )
    minimum_velocity = jnp.where(
        active,
        jnp.min(jnp.where(program.valid, normal_velocity, jnp.inf)),
        jnp.asarray(0.0, dtype=impulse.dtype),
    )
    dissipated = -jnp.sum(
        jnp.where(program.valid[:, None], tangent_impulse * tangent_velocity, 0.0)
    )
    finite = (
        jnp.all(jnp.isfinite(impulse))
        & jnp.all(jnp.isfinite(law_velocity))
        & jnp.all(jnp.isfinite(program.free_velocity))
        & jnp.all(jnp.isfinite(program.effective_mass))
        & jnp.all(jnp.isfinite(program.compliance))
        & jnp.all(jnp.isfinite(program.static_friction))
        & jnp.all(jnp.isfinite(program.friction))
        & jnp.all(jnp.isfinite(program.restitution))
    )
    return (
        law_velocity,
        complementarity,
        cone_defect,
        minimum_impulse,
        minimum_velocity,
        maximum_dissipation_defect,
        dissipated,
        finite,
    )


def _numeric_revision(
    program: ContactConeProgram, solver: ContactConeSolverPlan, /
) -> ContactConeNumericRevision:
    parameters = jnp.asarray(
        (
            solver.maximum_iterations,
            solver.absolute_tolerance,
            solver.relative_tolerance,
            solver.relaxation,
        ),
        dtype=program.free_velocity.dtype,
    )
    return ContactConeNumericRevision(
        program.free_velocity,
        program.effective_mass,
        program.compliance,
        program.static_friction,
        program.friction,
        program.restitution,
        program.valid,
        program.mechanical_available,
        parameters,
        program.program_id,
        solver.plan_id,
    )


def contact_cone_numeric_revision_matches(
    program: ContactConeProgram, result: ContactConeResult, /
) -> Array:
    """Return exact, JAX-safe numeric provenance equality for a cone result."""

    if not isinstance(program, ContactConeProgram):
        raise TypeError("program must be ContactConeProgram.")
    if not isinstance(result, ContactConeResult):
        raise TypeError("result must be ContactConeResult.")
    revision = result.evidence.numeric_revision
    if not isinstance(revision, ContactConeNumericRevision):
        return jnp.asarray(False)
    expected = (
        program.free_velocity,
        program.effective_mass,
        program.compliance,
        program.static_friction,
        program.friction,
        program.restitution,
        program.valid,
        program.mechanical_available,
    )
    recorded = (
        revision.free_velocity,
        revision.effective_mass,
        revision.compliance,
        revision.static_friction,
        revision.dynamic_friction,
        revision.restitution,
        revision.route_mask,
        revision.mechanical_available,
    )
    if any(left.shape != right.shape for left, right in zip(expected, recorded)):
        return jnp.asarray(False)
    identifiers_match = (
        revision.program_id == program.program_id
        and revision.solver_id == result.evidence.solver_id
        and result.evidence.program_id == program.program_id
    )
    exact = jnp.all(
        jnp.stack(
            tuple(jnp.all(left == right) for left, right in zip(expected, recorded))
        )
    )
    policy_finite = revision.solver_parameters.shape == (4,) and identifiers_match
    return (
        jnp.asarray(policy_finite)
        & exact
        & jnp.all(jnp.isfinite(revision.solver_parameters))
    )


def contact_cone_result_is_certified(
    program: ContactConeProgram, result: ContactConeResult, /
) -> Array:
    """Re-certify a result against the program rather than trusting stale evidence."""

    current = contact_cone_numeric_revision_matches(program, result)
    revision = result.evidence.numeric_revision
    if not isinstance(revision, ContactConeNumericRevision):
        return jnp.asarray(False)
    (
        law_velocity,
        complementarity,
        cone_defect,
        minimum_impulse,
        minimum_velocity,
        maximum_dissipation,
        dissipated,
        finite,
    ) = _contact_law_diagnostics(program, result.impulse)
    expected_post = (
        program.effective_mass @ result.impulse.reshape((-1,))
        + program.free_velocity.reshape((-1,))
    ).reshape(result.impulse.shape)
    scale = jnp.maximum(
        1.0,
        jnp.sqrt(
            jnp.sum(
                jnp.where(program.valid[:, None], program.free_velocity, 0.0).reshape(
                    (-1,)
                )
                ** 2
            )
        ),
    )
    tolerance = revision.solver_parameters[1] + revision.solver_parameters[2] * scale
    post_residual = jnp.max(
        jnp.abs(expected_post - result.post_relative_velocity), initial=0.0
    )
    law_residual = jnp.max(
        jnp.abs(law_velocity - result.contact_law_velocity), initial=0.0
    )
    return (
        current
        & result.evidence.successful
        & finite
        & (complementarity <= tolerance)
        & (cone_defect <= tolerance)
        & (minimum_impulse >= -tolerance)
        & (minimum_velocity >= -tolerance)
        & (maximum_dissipation <= tolerance)
        & (dissipated >= -tolerance)
        & (post_residual <= tolerance)
        & (law_residual <= tolerance)
    )


def solve_contact_cone(
    program: ContactConeProgram,
    /,
    *,
    solver: ContactConeSolverPlan | None = None,
    initial_impulse: ArrayLike | None = None,
) -> ContactConeResult:
    """Solve the declared compliant Signorini and Coulomb impulse law."""

    if not isinstance(program, ContactConeProgram):
        raise TypeError("program must be ContactConeProgram.")
    solver_ = ContactConeSolverPlan() if solver is None else solver
    if not isinstance(solver_, ContactConeSolverPlan):
        raise TypeError("solver must be ContactConeSolverPlan or None.")
    count = program.contact_count
    local_dimension = program.local_dimension
    matrix = program.effective_mass + jnp.diag(program.compliance)
    free = program.free_velocity.reshape((-1,))
    initial = (
        jnp.zeros((count, local_dimension), dtype=free.dtype)
        if initial_impulse is None
        else jnp.asarray(initial_impulse, dtype=free.dtype)
    )
    if initial.shape != (count, local_dimension):
        raise ValueError("initial_impulse has invalid shape.")
    active_free = jnp.where(program.valid[:, None], program.free_velocity, 0.0)
    initial_scale = jnp.maximum(1.0, jnp.sqrt(jnp.sum(active_free.reshape((-1,)) ** 2)))
    tolerance = solver_.absolute_tolerance + solver_.relative_tolerance * initial_scale

    static_candidate, _, _, _ = _iterate_contact_law(
        matrix,
        free,
        initial,
        program.static_friction,
        program.valid,
        solver_,
        tolerance,
    )
    static_law_velocity = (matrix @ static_candidate.reshape((-1,)) + free).reshape(
        static_candidate.shape
    )
    static_slip = jnp.sqrt(jnp.sum(static_law_velocity[:, 1:] ** 2, axis=-1))
    static_impulse_norm = jnp.sqrt(jnp.sum(static_candidate[:, 1:] ** 2, axis=-1))
    sticking = (static_slip <= tolerance) & (
        static_impulse_norm
        <= program.static_friction * static_candidate[:, 0] + tolerance
    )
    selected_friction = jnp.where(sticking, program.static_friction, program.friction)
    candidate, converged, first_converged, residual_norm = _iterate_contact_law(
        matrix,
        free,
        initial,
        selected_friction,
        program.valid,
        solver_,
        tolerance,
    )
    (
        candidate_law_velocity,
        complementarity,
        cone_defect,
        minimum_impulse,
        minimum_velocity,
        maximum_dissipation,
        dissipated,
        finite,
    ) = _contact_law_diagnostics(program, candidate)
    candidate_post = (program.effective_mass @ candidate.reshape((-1,)) + free).reshape(
        candidate.shape
    )
    material_law_complete = jnp.all((~program.valid) | program.mechanical_available)
    numeric_inputs_valid = (
        jnp.all(program.compliance >= 0.0)
        & jnp.all(program.static_friction >= 0.0)
        & jnp.all(program.friction >= 0.0)
        & jnp.all(program.friction <= program.static_friction)
        & jnp.all((program.restitution >= 0.0) & (program.restitution <= 1.0))
    )
    dissipative = dissipated >= -tolerance
    successful = (
        converged
        & finite
        & numeric_inputs_valid
        & material_law_complete
        & (complementarity <= tolerance)
        & (cone_defect <= tolerance)
        & (minimum_impulse >= -tolerance)
        & (minimum_velocity >= -tolerance)
        & (maximum_dissipation <= tolerance)
        & dissipative
    )
    accepted = jnp.where(successful, candidate, jnp.zeros_like(candidate))
    accepted_post = (program.effective_mass @ accepted.reshape((-1,)) + free).reshape(
        accepted.shape
    )
    accepted_law_velocity = (matrix @ accepted.reshape((-1,)) + free).reshape(
        accepted.shape
    )
    revision = _numeric_revision(program, solver_)
    evidence = ContactConeEvidence(
        converged,
        first_converged,
        residual_norm,
        complementarity,
        cone_defect,
        minimum_impulse,
        dissipated,
        finite,
        successful,
        program.program_id,
        solver_.plan_id,
        material_law_complete=material_law_complete,
        minimum_normal_velocity=minimum_velocity,
        maximum_dissipation_defect=maximum_dissipation,
        certificate_tolerance=tolerance,
        dissipative=dissipative,
        numeric_revision=revision,
    )
    return ContactConeResult(
        accepted,
        accepted_post,
        evidence,
        candidate_impulse=candidate,
        candidate_post_relative_velocity=candidate_post,
        contact_law_velocity=accepted_law_velocity,
        candidate_contact_law_velocity=candidate_law_velocity,
    )


__all__ = [
    "ContactConeEvidence",
    "ContactConeNumericRevision",
    "ContactConeProgram",
    "ContactConeResult",
    "ContactConeSolverPlan",
    "build_contact_cone_program",
    "project_signorini_coulomb_product",
    "solve_contact_cone",
]
