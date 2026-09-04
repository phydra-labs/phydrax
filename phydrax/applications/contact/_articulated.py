#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._kinematics import ContactKinematicsEpoch
from ...discretization.contact._participant import AbstractContactParticipant
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    dual_transpose,
    DualSpace,
    eigen as eigen_api,
    FunctionLinearOperator,
    MaterializationPolicy,
    materialize,
    OperatorProperties,
)
from ._cone import (
    build_contact_cone_program,
    contact_cone_numeric_revision_matches,
    contact_cone_result_is_certified,
    ContactConeEvidence,
    ContactConeProgram,
    ContactConeResult,
    ContactConeSolverPlan,
    solve_contact_cone,
)
from ._materials import ContactMaterialPairTable


class ArticulatedContactPreparationEvidence(StrictModule):
    kinematics_successful: Array
    participant_routes_complete: Array
    material_law_complete: Array
    velocity_residual: Array
    velocity_scale: Array
    velocity_consistent: Array
    delassus_symmetry_residual: Array
    delassus_scale: Array
    minimum_delassus_diagonal: Array
    nonnegative_delassus_diagonal: Array
    minimum_delassus_eigenvalue: Array
    delassus_spectral_residual: Array
    delassus_spectral_valid: Array
    finite: Array
    successful: Array
    participant_id: str = eqx.field(static=True)
    kinematics_id: str = eqx.field(static=True)


class ArticulatedContactDualityEvidence(StrictModule):
    contact_power: Array
    generalized_power: Array
    residual: Array
    scale: Array
    finite: Array
    valid: Array
    operator_id: str = eqx.field(static=True)


class PreparedArticulatedContact(StrictModule, NonTrainableState):
    velocity_operator: AbstractLinearOperator
    inverse_mass_operator: AbstractLinearOperator
    delassus_operator: AbstractLinearOperator
    program: ContactConeProgram
    free_velocity: PyTree[Array]
    evidence: ArticulatedContactPreparationEvidence
    prepared_id: str = eqx.field(static=True)


class ArticulatedContactEvidence(StrictModule):
    preparation: ArticulatedContactPreparationEvidence
    cone: ContactConeEvidence
    duality: ArticulatedContactDualityEvidence
    minimum_post_normal_velocity: Array
    post_contact_feasible: Array
    certificate_tolerance: Array
    numeric_revision_matches: Array
    contact_equation_residual: Array
    contact_certificate_valid: Array
    finite: Array
    applied: Array
    fail_closed: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class ArticulatedContactResult(StrictModule):
    impulse: Array
    generalized_impulse: PyTree[Array]
    velocity_update: PyTree[Array]
    post_velocity: PyTree[Array]
    post_contact_velocity: Array
    cone_result: ContactConeResult
    evidence: ArticulatedContactEvidence


def _contact_layout(kinematics: ContactKinematicsEpoch, /) -> tuple[int, int]:
    if not isinstance(kinematics, ContactKinematicsEpoch):
        raise TypeError("kinematics must be ContactKinematicsEpoch.")
    if not kinematics.batches:
        raise ValueError("Articulated contact requires positive fixed-route capacity.")
    tangent_dimension = int(kinematics.batches[0].tangential_velocity.shape[-1])
    if any(
        batch.tangential_velocity.shape[-1] != tangent_dimension
        for batch in kinematics.batches
    ):
        raise ValueError("Contact route tangent dimensions disagree.")
    contact_count = sum(batch.capacity for batch in kinematics.batches)
    if contact_count <= 0:
        raise ValueError("Articulated contact requires positive fixed-route capacity.")
    return contact_count, 1 + tangent_dimension


def _participant_routes_complete(
    participant: AbstractContactParticipant,
    kinematics: ContactKinematicsEpoch,
    vertex_offset: int,
    /,
) -> Array:
    vertex_count = participant.surface_plan.vertex_count
    if vertex_offset < 0:
        raise ValueError("vertex_offset must be nonnegative.")
    complete = []
    for batch in kinematics.batches:
        indices = batch.vertex_indices - vertex_offset
        used = (jnp.abs(batch.coefficients) > 0.0) & (
            (indices >= 0) & (indices < vertex_count)
        )
        complete.append((~batch.valid) | jnp.any(used, axis=1))
    return jnp.all(jnp.concatenate(tuple(complete), axis=0))


def _route_velocity(
    participant: AbstractContactParticipant,
    configuration: PyTree[Any],
    kinematics: ContactKinematicsEpoch,
    rates: PyTree[Any],
    vertex_offset: int,
    /,
) -> Array:
    vertex_velocity = participant.velocities(configuration, rates)
    values = []
    for batch in kinematics.batches:
        local_indices = batch.vertex_indices - vertex_offset
        inside = (local_indices >= 0) & (
            local_indices < participant.surface_plan.vertex_count
        )
        safe = jnp.clip(local_indices, 0, participant.surface_plan.vertex_count - 1)
        gathered = jnp.where(inside[..., None], vertex_velocity[safe], 0.0)
        relative = jnp.sum(batch.coefficients[..., None] * gathered, axis=1)
        normal = jnp.sum(relative * batch.normal, axis=-1, keepdims=True)
        tangent = jnp.sum(batch.tangent_basis * relative[..., :, None], axis=-2)
        local = jnp.concatenate((normal, tangent), axis=-1)
        values.append(jnp.where(batch.valid[:, None], local, 0.0))
    return jnp.concatenate(tuple(values), axis=0)


def _surface_impulse(
    participant: AbstractContactParticipant,
    kinematics: ContactKinematicsEpoch,
    local_impulse: Array,
    vertex_offset: int,
    /,
) -> Array:
    dtype = local_impulse.dtype
    surface = jnp.zeros(
        (
            participant.surface_plan.vertex_count,
            participant.surface_plan.ambient_dimension,
        ),
        dtype=dtype,
    )
    start = 0
    for batch in kinematics.batches:
        stop = start + batch.capacity
        local = jnp.where(batch.valid[:, None], local_impulse[start:stop], 0.0)
        world = batch.normal * local[:, :1] + jnp.sum(
            batch.tangent_basis * local[:, None, 1:], axis=-1
        )
        route_impulse = batch.coefficients[..., None] * world[:, None, :]
        local_indices = batch.vertex_indices - vertex_offset
        inside = (local_indices >= 0) & (
            local_indices < participant.surface_plan.vertex_count
        )
        safe = jnp.clip(local_indices, 0, participant.surface_plan.vertex_count - 1)
        surface = surface.at[safe].add(jnp.where(inside[..., None], route_impulse, 0.0))
        start = stop
    return surface


def _recorded_contact_velocity(kinematics: ContactKinematicsEpoch, /) -> Array:
    return jnp.concatenate(
        tuple(
            jnp.concatenate(
                (batch.normal_velocity[:, None], batch.tangential_velocity), axis=-1
            )
            for batch in kinematics.batches
        ),
        axis=0,
    )


def _route_validity(kinematics: ContactKinematicsEpoch, /) -> Array:
    return jnp.concatenate(tuple(batch.valid for batch in kinematics.batches), axis=0)


def _tree_finite(value: PyTree[Array], /) -> Array:
    leaves = jax.tree.leaves(value)
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))


def _tree_where(
    condition: Array, accepted: PyTree[Array], rejected: PyTree[Array], /
) -> PyTree[Array]:
    return jax.tree.map(lambda yes, no: jnp.where(condition, yes, no), accepted, rejected)


def build_contact_velocity_operator(
    participant: AbstractContactParticipant,
    configuration: PyTree[Any],
    kinematics: ContactKinematicsEpoch,
    /,
    *,
    vertex_offset: int = 0,
) -> AbstractLinearOperator:
    """Build ``G``; scene vertices outside the participant range are stationary."""

    if not isinstance(participant, AbstractContactParticipant):
        raise TypeError("participant must be AbstractContactParticipant.")
    _contact_layout(kinematics)
    offset = int(vertex_offset)
    if offset < 0:
        raise ValueError("vertex_offset must be nonnegative.")
    configuration_ = participant.source_space.validate(configuration)
    positions = participant.positions(configuration_)
    contact_count, local_dimension = _contact_layout(kinematics)
    contact_space = ArraySpace((contact_count, local_dimension), dtype=positions.dtype)

    def action(rates):
        return _route_velocity(participant, configuration_, kinematics, rates, offset)

    def transpose_action(local_effort):
        surface_effort = _surface_impulse(participant, kinematics, local_effort, offset)
        return participant.effort_pullback(configuration_, surface_effort)

    return FunctionLinearOperator(
        action,
        source=participant.tangent_space,
        target=contact_space,
        transpose_action=transpose_action,
        operator_id=canonical_fingerprint(
            {
                "kind": "contact-velocity-operator",
                "participant": participant.participant_id,
                "kinematics": kinematics.epoch_id,
                "source": participant.tangent_space.space_id,
                "vertex_offset": offset,
            }
        ),
    )


def build_delassus_operator(
    velocity_operator: AbstractLinearOperator,
    inverse_mass_operator: AbstractLinearOperator,
    /,
) -> AbstractLinearOperator:
    """Compose ``W = G M^-1 G*`` without unconstrained-body response paths."""

    if not isinstance(velocity_operator, AbstractLinearOperator):
        raise TypeError("velocity_operator must be AbstractLinearOperator.")
    if not isinstance(inverse_mass_operator, AbstractLinearOperator):
        raise TypeError("inverse_mass_operator must be AbstractLinearOperator.")
    if velocity_operator.batch_shape or inverse_mass_operator.batch_shape:
        raise ValueError("Articulated contact operators cannot be operator-batched.")
    tangent = velocity_operator.source
    tangent_dual = DualSpace(tangent)
    if not inverse_mass_operator.source.compatible(tangent_dual) or not (
        inverse_mass_operator.target.compatible(tangent)
    ):
        raise ValueError(
            "inverse_mass_operator must map the contact tangent dual to its tangent."
        )
    return velocity_operator @ inverse_mass_operator @ dual_transpose(velocity_operator)


def contact_duality_evidence(
    velocity_operator: AbstractLinearOperator,
    generalized_velocity: PyTree[Any],
    contact_impulse: ArrayLike,
    /,
) -> ArticulatedContactDualityEvidence:
    """Certify contact/generalized force-power duality for one operator action."""

    if not isinstance(velocity_operator, AbstractLinearOperator):
        raise TypeError("velocity_operator must be AbstractLinearOperator.")
    velocity = velocity_operator.source.validate(generalized_velocity)
    contact_effort_space = DualSpace(velocity_operator.target)
    impulse = contact_effort_space.validate(contact_impulse)
    local_velocity = velocity_operator.mv(velocity)
    generalized_impulse = dual_transpose(velocity_operator).mv(impulse)
    contact_power = contact_effort_space.pair(impulse, local_velocity)
    generalized_power = DualSpace(velocity_operator.source).pair(
        generalized_impulse, velocity
    )
    residual = contact_power - generalized_power
    scale = jnp.maximum(
        1.0, jnp.maximum(jnp.abs(contact_power), jnp.abs(generalized_power))
    )
    dtype = jnp.result_type(contact_power, generalized_power)
    tolerance = jnp.finfo(dtype).eps * max(
        64, 8 * velocity_operator.source.size, 8 * velocity_operator.target.size
    )
    finite = jnp.all(
        jnp.isfinite(jnp.stack((contact_power, generalized_power, residual, scale)))
    )
    return ArticulatedContactDualityEvidence(
        contact_power,
        generalized_power,
        residual,
        scale,
        finite,
        finite & (jnp.abs(residual) <= tolerance * scale),
        velocity_operator.operator_id,
    )


def prepare_articulated_contact(
    participant: AbstractContactParticipant,
    configuration: PyTree[Any],
    free_velocity: PyTree[Any],
    kinematics: ContactKinematicsEpoch,
    materials: ContactMaterialPairTable,
    inverse_mass_operator: AbstractLinearOperator,
    /,
    *,
    vertex_offset: int = 0,
    compliance: ArrayLike = 0.0,
    materialization_policy: MaterializationPolicy | None = None,
) -> PreparedArticulatedContact:
    """Lower fixed-route articulated kinematics to the native cone program.

    ``vertex_offset`` locates the participant in a concatenated contact scene.
    Other scene vertices are treated as fixed by this reduced response.
    """

    if not isinstance(materials, ContactMaterialPairTable):
        raise TypeError("materials must be ContactMaterialPairTable.")
    velocity_operator = build_contact_velocity_operator(
        participant,
        configuration,
        kinematics,
        vertex_offset=vertex_offset,
    )
    free = participant.tangent_space.validate(free_velocity)
    delassus = build_delassus_operator(velocity_operator, inverse_mass_operator)
    policy = (
        MaterializationPolicy()
        if materialization_policy is None
        else materialization_policy
    )
    if not isinstance(policy, MaterializationPolicy):
        raise TypeError("materialization_policy must be MaterializationPolicy or None.")
    dense_delassus = materialize(delassus, policy)
    program = build_contact_cone_program(
        kinematics, materials, dense_delassus, compliance=compliance
    )

    local_free = velocity_operator.mv(free)
    recorded = _recorded_contact_velocity(kinematics)
    valid = _route_validity(kinematics)[:, None]
    velocity_difference = jnp.where(valid, local_free - recorded, 0.0)
    velocity_residual = jnp.max(jnp.abs(velocity_difference), initial=0.0)
    velocity_scale = jnp.maximum(
        1.0,
        jnp.maximum(
            jnp.max(jnp.abs(jnp.where(valid, local_free, 0.0)), initial=0.0),
            jnp.max(jnp.abs(jnp.where(valid, recorded, 0.0)), initial=0.0),
        ),
    )
    dtype = dense_delassus.dtype
    tolerance = jnp.finfo(dtype).eps * max(64, 8 * delassus.source.size)
    adjoint_dense = jnp.conj(jnp.swapaxes(dense_delassus, -1, -2))
    symmetry_residual = jnp.max(jnp.abs(dense_delassus - adjoint_dense), initial=0.0)
    delassus_scale = jnp.maximum(1.0, jnp.max(jnp.abs(dense_delassus), initial=0.0))
    minimum_diagonal = jnp.min(jnp.real(jnp.diag(dense_delassus)), initial=jnp.inf)
    hermitian_delassus = 0.5 * (dense_delassus + adjoint_dense)
    spectral_operator = DenseLinearOperator(
        hermitian_delassus,
        source=velocity_operator.target,
        target=velocity_operator.target,
        properties=OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "verified"},
        ),
        operator_id=f"{delassus.operator_id}/spectral-certificate",
    )
    spectral_result = eigen_api.eigensolve(
        eigen_api.Eigenproblem(spectral_operator),
        policy=eigen_api.EigenSolvePolicy(
            eigen_api.DenseEigh(),
            count=delassus.source.size,
            which="smallest-algebraic",
        ),
    )
    minimum_eigenvalue = jnp.min(
        jnp.where(spectral_result.mode_mask, spectral_result.eigenvalues, jnp.inf)
    )
    spectral_residual = jnp.max(
        jnp.where(
            spectral_result.mode_mask,
            spectral_result.diagnostics.relative_residuals,
            0.0,
        ),
        initial=0.0,
    )
    spectral_finite = jnp.all(jnp.isfinite(spectral_result.eigenvalues)) & jnp.isfinite(
        spectral_residual
    )
    spectral_valid = (
        spectral_result.successful
        & (spectral_result.effective_count == delassus.source.size)
        & spectral_finite
        & (minimum_eigenvalue >= -tolerance * delassus_scale)
    )
    participant_routes_complete = _participant_routes_complete(
        participant, kinematics, int(vertex_offset)
    )
    material_law_complete = jnp.all((~program.valid) | program.mechanical_available)
    velocity_consistent = velocity_residual <= tolerance * velocity_scale
    nonnegative_diagonal = minimum_diagonal >= -tolerance * delassus_scale
    finite = (
        jnp.all(jnp.isfinite(dense_delassus))
        & jnp.all(jnp.isfinite(local_free))
        & jnp.all(jnp.isfinite(recorded))
        & _tree_finite(free)
        & spectral_finite
    )
    successful = (
        kinematics.evidence.successful
        & participant_routes_complete
        & material_law_complete
        & velocity_consistent
        & (symmetry_residual <= tolerance * delassus_scale)
        & nonnegative_diagonal
        & spectral_valid
        & finite
    )
    evidence = ArticulatedContactPreparationEvidence(
        kinematics_successful=kinematics.evidence.successful,
        participant_routes_complete=participant_routes_complete,
        material_law_complete=material_law_complete,
        velocity_residual=velocity_residual,
        velocity_scale=velocity_scale,
        velocity_consistent=velocity_consistent,
        delassus_symmetry_residual=symmetry_residual,
        delassus_scale=delassus_scale,
        minimum_delassus_diagonal=minimum_diagonal,
        nonnegative_delassus_diagonal=nonnegative_diagonal,
        minimum_delassus_eigenvalue=minimum_eigenvalue,
        delassus_spectral_residual=spectral_residual,
        delassus_spectral_valid=spectral_valid,
        finite=finite,
        successful=successful,
        participant_id=participant.participant_id,
        kinematics_id=kinematics.epoch_id,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-articulated-contact",
            "participant": participant.participant_id,
            "kinematics": kinematics.epoch_id,
            "materials": materials.table_id,
            "velocity_operator": velocity_operator.operator_id,
            "inverse_mass_operator": inverse_mass_operator.operator_id,
        }
    )
    return PreparedArticulatedContact(
        velocity_operator,
        inverse_mass_operator,
        delassus,
        program,
        free,
        evidence,
        prepared_id,
    )


def apply_articulated_contact_impulse(
    prepared: PreparedArticulatedContact,
    cone_result: ContactConeResult,
    /,
) -> ArticulatedContactResult:
    """Apply a certified cone impulse through ``G*`` and constrained ``M^-1``."""

    if not isinstance(prepared, PreparedArticulatedContact):
        raise TypeError("prepared must be PreparedArticulatedContact.")
    if not isinstance(cone_result, ContactConeResult):
        raise TypeError("cone_result must be ContactConeResult.")
    if cone_result.evidence.program_id != prepared.program.program_id:
        raise ValueError("Cone result does not belong to the prepared contact program.")

    candidate_impulse = DualSpace(prepared.velocity_operator.target).validate(
        cone_result.impulse
    )
    candidate_generalized = dual_transpose(prepared.velocity_operator).mv(
        candidate_impulse
    )
    candidate_update = prepared.inverse_mass_operator.mv(candidate_generalized)
    candidate_post = jax.tree.map(
        lambda free, update: free + update, prepared.free_velocity, candidate_update
    )
    candidate_contact_velocity = prepared.velocity_operator.mv(candidate_post)
    contact_velocity_update = prepared.velocity_operator.mv(candidate_update)
    expected_contact_update = (
        prepared.program.effective_mass @ candidate_impulse.reshape((-1,))
    ).reshape(candidate_impulse.shape)
    contact_equation_residual = jnp.max(
        jnp.abs(contact_velocity_update - expected_contact_update), initial=0.0
    )
    law_velocity = (
        (prepared.program.effective_mass + jnp.diag(prepared.program.compliance))
        @ candidate_impulse.reshape((-1,))
        + prepared.program.free_velocity.reshape((-1,))
    ).reshape(candidate_impulse.shape)
    duality = contact_duality_evidence(
        prepared.velocity_operator, prepared.free_velocity, candidate_impulse
    )
    valid = prepared.program.valid
    active = jnp.any(valid)
    minimum_post_normal = jnp.where(
        active,
        jnp.min(jnp.where(valid, law_velocity[:, 0], jnp.inf)),
        jnp.asarray(0.0, dtype=law_velocity.dtype),
    )
    dtype = candidate_contact_velocity.dtype
    certificate_scale = jnp.maximum(
        1.0,
        jnp.maximum(
            jnp.max(jnp.abs(prepared.program.free_velocity), initial=0.0),
            jnp.max(jnp.abs(candidate_contact_velocity), initial=0.0),
        ),
    )
    tolerance = jnp.maximum(
        cone_result.evidence.certificate_tolerance,
        jnp.sqrt(jnp.finfo(dtype).eps) * certificate_scale,
    )
    post_feasible = minimum_post_normal >= -tolerance
    numeric_revision_matches = contact_cone_numeric_revision_matches(
        prepared.program, cone_result
    )
    cone_certificate_valid = contact_cone_result_is_certified(
        prepared.program, cone_result
    )
    candidate_finite = (
        jnp.all(jnp.isfinite(candidate_impulse))
        & _tree_finite(candidate_generalized)
        & _tree_finite(candidate_update)
        & _tree_finite(candidate_post)
        & jnp.all(jnp.isfinite(candidate_contact_velocity))
        & jnp.all(jnp.isfinite(law_velocity))
        & jnp.isfinite(contact_equation_residual)
    )
    finite = (
        prepared.evidence.finite
        & cone_result.evidence.finite
        & candidate_finite
        & duality.finite
    )
    contact_certificate_valid = (
        cone_certificate_valid
        & numeric_revision_matches
        & (contact_equation_residual <= tolerance)
        & duality.valid
        & post_feasible
        & finite
    )
    successful = prepared.evidence.successful & contact_certificate_valid

    zero_impulse = jnp.zeros_like(candidate_impulse)
    zero_generalized = DualSpace(prepared.velocity_operator.source).zeros()
    zero_update = prepared.velocity_operator.source.zeros()
    applied_impulse = jnp.where(successful, candidate_impulse, zero_impulse)
    generalized = _tree_where(successful, candidate_generalized, zero_generalized)
    update = _tree_where(successful, candidate_update, zero_update)
    post = jax.tree.map(
        lambda free, increment: free + increment, prepared.free_velocity, update
    )
    post_contact_velocity = prepared.velocity_operator.mv(post)
    zero_update = (
        jnp.max(jnp.abs(prepared.velocity_operator.source.flatten(update)), initial=0.0)
        == 0.0
    )
    zero_generalized_output = (
        jnp.max(
            jnp.abs(prepared.velocity_operator.source.flatten(generalized)),
            initial=0.0,
        )
        == 0.0
    )
    zero_applied = jnp.max(jnp.abs(applied_impulse), initial=0.0) == 0.0
    fail_closed = successful | (zero_update & zero_generalized_output & zero_applied)
    evidence = ArticulatedContactEvidence(
        preparation=prepared.evidence,
        cone=cone_result.evidence,
        duality=duality,
        minimum_post_normal_velocity=minimum_post_normal,
        post_contact_feasible=post_feasible,
        certificate_tolerance=tolerance,
        numeric_revision_matches=numeric_revision_matches,
        contact_equation_residual=contact_equation_residual,
        contact_certificate_valid=contact_certificate_valid,
        finite=finite,
        applied=successful,
        fail_closed=fail_closed,
        successful=successful & fail_closed,
        prepared_id=prepared.prepared_id,
    )
    return ArticulatedContactResult(
        applied_impulse,
        generalized,
        update,
        post,
        post_contact_velocity,
        cone_result,
        evidence,
    )


def solve_articulated_contact(
    prepared: PreparedArticulatedContact,
    /,
    *,
    solver: ContactConeSolverPlan | None = None,
    initial_impulse: ArrayLike | None = None,
) -> ArticulatedContactResult:
    """Solve and fail-closed apply one prepared articulated contact impulse."""

    if not isinstance(prepared, PreparedArticulatedContact):
        raise TypeError("prepared must be PreparedArticulatedContact.")
    cone_result = solve_contact_cone(
        prepared.program, solver=solver, initial_impulse=initial_impulse
    )
    return apply_articulated_contact_impulse(prepared, cone_result)


__all__ = [
    "ArticulatedContactDualityEvidence",
    "ArticulatedContactEvidence",
    "ArticulatedContactPreparationEvidence",
    "ArticulatedContactResult",
    "PreparedArticulatedContact",
    "apply_articulated_contact_impulse",
    "build_contact_velocity_operator",
    "build_delassus_operator",
    "contact_duality_evidence",
    "prepare_articulated_contact",
    "solve_articulated_contact",
]
