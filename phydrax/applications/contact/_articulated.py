#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-route articulated contact through the operator-valued cone program.

The contact response is the composed operator ``W = G M^-1 G*``.  It is
positive semidefinite when ``M^-1`` is a symmetric positive semidefinite map
from the generalized tangent dual to the tangent and ``G*`` is the algebraic
dual transpose of ``G``.  That hypothesis comes from a prepared articulated
mass (verified positive pivots; the matrix-free route) or from a dense property
verification of an explicit inverse-mass operator's represented bilinear
matrix, obtained from its stored dense matrix or under the caller's explicit
bounded materialization policy.  Preparation also records two deterministic
adjoint/nonnegativity probes of ``W`` as finite-precision evidence only.  No
dense Delassus matrix is formed.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._kinematics import ContactKinematicsEpoch
from ...discretization.contact._participant import AbstractContactParticipant
from ...discretization.particle._articulated_mass import PreparedArticulatedMass
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    DensePropertyEvidence,
    DensePropertyVerificationPolicy,
    dual_transpose,
    DualSpace,
    FunctionLinearOperator,
    MaterializationPolicy,
    materialize,
    verify_dense_properties,
)
from ...typing import ConvertibleToArray
from ._cone import (
    build_contact_cone_program,
    certify_contact_cone_impulse,
    contact_cone_numeric_revision_matches,
    contact_cone_result_is_certified,
    ContactCertificatePolicy,
    ContactConeEvidence,
    ContactConeProgram,
    ContactConeResult,
    ContactConeSolverPlan,
    solve_contact_cone,
)
from ._materials import ContactMaterialPairTable


class InverseMassOperatorCertificate(StrictModule):
    """Dense property certificate of an explicit true-dual inverse-mass operator.

    ``property_evidence`` is the native dense property evidence of the
    operator's represented bilinear matrix ``A`` (``⟨f, M⁻¹ g⟩ = f·A g`` under
    the algebraic dual pairing): Hermitian defect, eigenvalues, tolerance and
    the Hermitian/semidefinite flags.  ``route`` records whether ``A`` was the
    stored matrix of an explicit dense operator or a bounded materialization
    under the caller's policy.  A zero-dimensional map has no evidence and is
    trivially successful.
    """

    property_evidence: DensePropertyEvidence | None
    successful: Array
    operator_id: str = eqx.field(static=True)
    route: str = eqx.field(static=True)
    degrees_of_freedom: int = eqx.field(static=True)


class ArticulatedContactPreparationEvidence(StrictModule):
    """Preparation evidence; probe fields are finite-precision evidence only."""

    kinematics_successful: Array
    participant_routes_complete: Array
    material_law_complete: Array
    velocity_residual: Array
    velocity_scale: Array
    velocity_consistent: Array
    mass_successful: Array
    response_probe_adjoint_residual: Array
    response_probe_scale: Array
    response_probe_adjoint_consistent: Array
    response_probe_nonnegative: Array
    finite: Array
    successful: Array
    participant_id: str = eqx.field(static=True)
    kinematics_id: str = eqx.field(static=True)
    mass_hypothesis: str = eqx.field(static=True)
    inverse_mass_certificate: InverseMassOperatorCertificate | None = None


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
    program: ContactConeProgram
    free_velocity: PyTree[Array]
    evidence: ArticulatedContactPreparationEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def delassus_operator(self) -> AbstractLinearOperator:
        return self.program.response


class ArticulatedContactEvidence(StrictModule):
    """Application evidence; law and physical normal velocities are distinct.

    ``minimum_law_normal_velocity`` is the biased law residual (it includes
    restitution targets, compliance and any position-level gap bias);
    ``minimum_physical_normal_velocity`` is the post-impulse contact velocity
    ``G v_post``.
    """

    preparation: ArticulatedContactPreparationEvidence
    cone: ContactConeEvidence
    duality: ArticulatedContactDualityEvidence
    minimum_law_normal_velocity: Array
    minimum_physical_normal_velocity: Array
    law_feasible: Array
    numeric_revision_matches: Array
    contact_equation_residual: Array
    contact_equation_valid: Array
    physical_velocity_residual: Array
    physical_velocity_valid: Array
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
    tangent_dimension = kinematics.batches[0].tangential_velocity.shape[-1]
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
    if not leaves:
        return jnp.asarray(True)
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

    def action(rates: PyTree[Array]) -> Array:
        return _route_velocity(participant, configuration_, kinematics, rates, offset)

    def transpose_action(local_effort: Array) -> PyTree[Array]:
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


def build_mixed_contact_velocity_operator(
    contact_operator: AbstractLinearOperator,
    stop_operator: AbstractLinearOperator,
    /,
    *,
    width: int = 3,
) -> AbstractLinearOperator:
    """Stack contact rows and normal-only stop rows into one fixed-width ``G``.

    ``contact_operator`` maps the generalized tangent to ``(contacts, width)``
    local contact velocities; ``stop_operator`` maps it to ``(stops,)`` stop
    rates.  Stop rows occupy the normal component of trailing blocks and their
    padded components are identically zero, so the dual transpose pulls back
    only the stop normal impulse.
    """

    if not isinstance(contact_operator, AbstractLinearOperator) or not isinstance(
        stop_operator, AbstractLinearOperator
    ):
        raise TypeError("Mixed rows require AbstractLinearOperator values.")
    if contact_operator.batch_shape or stop_operator.batch_shape:
        raise ValueError("Mixed contact rows cannot be operator-batched.")
    tangent = contact_operator.source
    if not stop_operator.source.compatible(tangent):
        raise ValueError("Contact and stop rows must share the generalized tangent.")
    width_ = int(width)
    contact_target = contact_operator.target
    stop_target = stop_operator.target
    if (
        not isinstance(contact_target, ArraySpace)
        or len(contact_target.shape) != 2
        or contact_target.shape[1] != width_
    ):
        raise ValueError("contact_operator must target (contacts, width) blocks.")
    if not isinstance(stop_target, ArraySpace) or len(stop_target.shape) != 1:
        raise ValueError("stop_operator must target one normal row per stop.")
    contacts = contact_target.shape[0]
    stops = stop_target.shape[0]
    dtype = contact_target.dtype
    space = ArraySpace((contacts + stops, width_), dtype=dtype)
    contact_dual = dual_transpose(contact_operator)
    stop_dual = dual_transpose(stop_operator)

    def action(rates: PyTree[Array]) -> Array:
        stop_rows = (
            jnp.zeros((stops, width_), dtype=dtype).at[:, 0].set(stop_operator.mv(rates))
        )
        return jnp.concatenate((contact_operator.mv(rates), stop_rows), axis=0)

    def transpose_action(local_impulse: Array) -> PyTree[Array]:
        return jax.tree.map(
            lambda left, right: left + right,
            contact_dual.mv(local_impulse[:contacts]),
            stop_dual.mv(local_impulse[contacts:, 0]),
        )

    return FunctionLinearOperator(
        action,
        source=tangent,
        target=space,
        transpose_action=transpose_action,
        operator_id=canonical_fingerprint(
            {
                "kind": "mixed-contact-velocity-operator",
                "contact": contact_operator.operator_id,
                "stop": stop_operator.operator_id,
                "width": width_,
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
    contact_impulse: ConvertibleToArray,
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


def certify_inverse_mass_operator(
    operator: AbstractLinearOperator,
    policy: MaterializationPolicy | None,
    /,
    *,
    verification: DensePropertyVerificationPolicy | None = None,
) -> InverseMassOperatorCertificate:
    """Certify symmetric semidefiniteness of an explicit inverse-mass map.

    An explicit ``DenseLinearOperator`` is verified from its stored matrix
    (route ``explicit-dense-matrix``).  Any other operator is materialized only
    under the caller's explicit bounded ``policy`` (route
    ``bounded-materialization``); without one it is refused, and a policy too
    small for the operator raises ``LinearCapabilityError``.  The matrix-free
    route is a ``PreparedArticulatedMass``.  Verification is delegated to
    :func:`phydrax.linalg.verify_dense_properties`, which must require both
    Hermitian symmetry and positive semidefiniteness; the default policy uses
    its explicit relative tolerance ``64 eps max|λ|`` and zero absolute
    tolerance.  The original matrix remains the response; nothing is repaired.
    A zero-dimensional map is trivially symmetric semidefinite and carries no
    dense property evidence.
    """

    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("operator must be AbstractLinearOperator.")
    if operator.batch_shape:
        raise ValueError("Inverse mass cannot be operator-batched.")
    if not operator.source.compatible(DualSpace(operator.target)):
        raise ValueError("Inverse mass must map the tangent dual to the tangent.")
    verification_ = (
        DensePropertyVerificationPolicy(
            require_hermitian=True,
            require_positive_semidefinite=True,
            relative_tolerance=64.0,
            absolute_tolerance=0.0,
        )
        if verification is None
        else verification
    )
    if not isinstance(verification_, DensePropertyVerificationPolicy):
        raise TypeError("verification must be DensePropertyVerificationPolicy or None.")
    if not (
        verification_.require_hermitian and verification_.require_positive_semidefinite
    ):
        raise ValueError(
            "Inverse-mass verification must require Hermitian symmetry and "
            "positive semidefiniteness."
        )
    if isinstance(operator, DenseLinearOperator):
        matrix = operator.matrix
        route = "explicit-dense-matrix"
    elif policy is None:
        raise ValueError(
            "A matrix-free inverse-mass operator requires an explicit bounded "
            "inverse_mass_materialization policy; use PreparedArticulatedMass "
            "for the matrix-free route."
        )
    elif not isinstance(policy, MaterializationPolicy):
        raise TypeError("policy must be MaterializationPolicy or None.")
    else:
        matrix = materialize(operator, policy)
        route = "bounded-materialization"
    size = int(matrix.shape[-1])
    if size == 0:
        return InverseMassOperatorCertificate(
            None, jnp.asarray(True), operator.operator_id, route, size
        )
    evidence = verify_dense_properties(matrix, policy=verification_)
    return InverseMassOperatorCertificate(
        evidence, evidence.successful, operator.operator_id, route, size
    )


def _response_probes(
    response: AbstractLinearOperator, space: ArraySpace, /
) -> tuple[Array, Array, Array, Array]:
    """Two deterministic probes of ``W``: adjoint consistency and nonnegativity."""

    dtype = space.dtype
    first = jnp.ones(space.shape, dtype=dtype)
    second = jnp.where(
        (jnp.arange(space.size).reshape(space.shape) % 2) == 0, 1.0, -1.0
    ).astype(dtype)
    first_image = response.mv(first)
    second_image = response.mv(second)
    cross = jnp.sum(first * second_image) - jnp.sum(second * first_image)
    scale = jnp.maximum(
        jnp.sqrt(jnp.sum(first * first)) * jnp.sqrt(jnp.sum(second_image**2))
        + jnp.sqrt(jnp.sum(second * second)) * jnp.sqrt(jnp.sum(first_image**2)),
        jnp.finfo(dtype).tiny,
    )
    tolerance = jnp.finfo(dtype).eps * max(64, 8 * space.size)
    nonnegative = (jnp.sum(first * first_image) >= -tolerance * scale) & (
        jnp.sum(second * second_image) >= -tolerance * scale
    )
    return jnp.abs(cross), scale, jnp.abs(cross) <= tolerance * scale, nonnegative


def prepare_articulated_contact(
    participant: AbstractContactParticipant,
    configuration: PyTree[Any],
    free_velocity: PyTree[Any],
    kinematics: ContactKinematicsEpoch,
    materials: ContactMaterialPairTable,
    inverse_mass: AbstractLinearOperator | PreparedArticulatedMass,
    /,
    *,
    certificate: ContactCertificatePolicy,
    vertex_offset: int = 0,
    compliance: ConvertibleToArray = 0.0,
    inverse_mass_materialization: MaterializationPolicy | None = None,
    inverse_mass_verification: DensePropertyVerificationPolicy | None = None,
) -> PreparedArticulatedContact:
    """Lower fixed-route articulated kinematics to the operator cone program.

    ``inverse_mass`` is either a prepared articulated mass (the matrix-free
    route: its verified pivots supply the semidefinite hypothesis and its
    binding is bound into the cone revision) or an explicit true-dual
    inverse-mass operator certified by :func:`certify_inverse_mass_operator`:
    a ``DenseLinearOperator`` from its stored matrix, any other operator only
    under the caller's explicit bounded ``inverse_mass_materialization``, with
    the optional ``inverse_mass_verification`` dense property policy.
    ``vertex_offset`` locates the participant in a concatenated contact scene;
    other scene vertices are fixed by this response.
    """

    if not isinstance(materials, ContactMaterialPairTable):
        raise TypeError("materials must be ContactMaterialPairTable.")
    operator_certificate: InverseMassOperatorCertificate | None
    if isinstance(inverse_mass, PreparedArticulatedMass):
        inverse_mass_operator: AbstractLinearOperator = inverse_mass.inverse_mass_operator
        mass_successful = inverse_mass.successful
        binding: PyTree[Array] = inverse_mass.binding
        hypothesis = "prepared-articulated-mass"
        operator_certificate = None
    elif isinstance(inverse_mass, AbstractLinearOperator):
        inverse_mass_operator = inverse_mass
        operator_certificate = certify_inverse_mass_operator(
            inverse_mass,
            inverse_mass_materialization,
            verification=inverse_mass_verification,
        )
        mass_successful = operator_certificate.successful
        binding = ()
        hypothesis = f"certified-operator/{operator_certificate.route}"
    else:
        raise TypeError(
            "inverse_mass must be PreparedArticulatedMass or AbstractLinearOperator."
        )
    velocity_operator = build_contact_velocity_operator(
        participant,
        configuration,
        kinematics,
        vertex_offset=vertex_offset,
    )
    free = participant.tangent_space.validate(free_velocity)
    delassus = build_delassus_operator(velocity_operator, inverse_mass_operator)
    program = build_contact_cone_program(
        kinematics,
        materials,
        delassus,
        certificate=certificate,
        compliance=compliance,
        response_binding=binding,
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
    dtype = local_free.dtype
    tolerance = jnp.finfo(dtype).eps * max(64, 8 * delassus.source.size)
    probe_residual, probe_scale, probe_adjoint, probe_nonnegative = _response_probes(
        delassus, program.contact_space
    )
    participant_routes_complete = _participant_routes_complete(
        participant, kinematics, int(vertex_offset)
    )
    material_law_complete = jnp.all((~program.valid) | program.mechanical_available)
    velocity_consistent = velocity_residual <= tolerance * velocity_scale
    finite = (
        jnp.all(jnp.isfinite(local_free))
        & jnp.all(jnp.isfinite(recorded))
        & _tree_finite(free)
        & jnp.isfinite(probe_residual)
        & jnp.isfinite(probe_scale)
    )
    successful = (
        kinematics.evidence.successful
        & participant_routes_complete
        & material_law_complete
        & velocity_consistent
        & mass_successful
        & probe_adjoint
        & probe_nonnegative
        & finite
    )
    evidence = ArticulatedContactPreparationEvidence(
        kinematics_successful=kinematics.evidence.successful,
        participant_routes_complete=participant_routes_complete,
        material_law_complete=material_law_complete,
        velocity_residual=velocity_residual,
        velocity_scale=velocity_scale,
        velocity_consistent=velocity_consistent,
        mass_successful=jnp.asarray(mass_successful, dtype=jnp.bool_),
        response_probe_adjoint_residual=probe_residual,
        response_probe_scale=probe_scale,
        response_probe_adjoint_consistent=probe_adjoint,
        response_probe_nonnegative=probe_nonnegative,
        finite=finite,
        successful=successful,
        participant_id=participant.participant_id,
        kinematics_id=kinematics.epoch_id,
        mass_hypothesis=hypothesis,
        inverse_mass_certificate=operator_certificate,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-articulated-contact",
            "participant": participant.participant_id,
            "kinematics": kinematics.epoch_id,
            "materials": materials.table_id,
            "velocity_operator": velocity_operator.operator_id,
            "inverse_mass_operator": inverse_mass_operator.operator_id,
            "program": program.program_id,
        }
    )
    return PreparedArticulatedContact(
        velocity_operator,
        inverse_mass_operator,
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
    program = prepared.program
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
    certificate = certify_contact_cone_impulse(program, candidate_impulse)
    velocity_tolerance = jnp.concatenate(
        (
            certificate.law_velocity_tolerance[:, None],
            jnp.broadcast_to(
                certificate.slip_speed_tolerance[:, None],
                (program.contact_count, program.tangent_dimension),
            ),
        ),
        axis=-1,
    )
    valid = program.valid
    equation_difference = jnp.where(
        valid[:, None],
        contact_velocity_update - program.response_action(candidate_impulse),
        0.0,
    )
    contact_equation_residual = jnp.max(jnp.abs(equation_difference), initial=0.0)
    contact_equation_valid = jnp.all(jnp.abs(equation_difference) <= velocity_tolerance)
    physical_difference = jnp.where(
        valid[:, None], candidate_contact_velocity - certificate.post_velocity, 0.0
    )
    physical_velocity_residual = jnp.max(jnp.abs(physical_difference), initial=0.0)
    physical_velocity_valid = jnp.all(jnp.abs(physical_difference) <= velocity_tolerance)
    duality = contact_duality_evidence(
        prepared.velocity_operator, prepared.free_velocity, candidate_impulse
    )
    active = jnp.any(valid)
    dtype = candidate_impulse.dtype
    minimum_law_normal = jnp.where(
        active,
        jnp.min(jnp.where(valid, certificate.law_velocity[:, 0], jnp.inf)),
        jnp.asarray(0.0, dtype=dtype),
    )
    minimum_physical_normal = jnp.where(
        active,
        jnp.min(jnp.where(valid, candidate_contact_velocity[:, 0], jnp.inf)),
        jnp.asarray(0.0, dtype=dtype),
    )
    law_feasible = certificate.law_normal_valid
    numeric_revision_matches = contact_cone_numeric_revision_matches(program, cone_result)
    cone_certificate_valid = contact_cone_result_is_certified(program, cone_result)
    candidate_finite = (
        jnp.all(jnp.isfinite(candidate_impulse))
        & _tree_finite(candidate_generalized)
        & _tree_finite(candidate_update)
        & _tree_finite(candidate_post)
        & jnp.all(jnp.isfinite(candidate_contact_velocity))
        & certificate.finite
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
        & contact_equation_valid
        & physical_velocity_valid
        & duality.valid
        & law_feasible
        & finite
    )
    successful = prepared.evidence.successful & contact_certificate_valid

    zero_generalized = DualSpace(prepared.velocity_operator.source).zeros()
    zero_update = prepared.velocity_operator.source.zeros()
    applied_impulse = jnp.where(successful, candidate_impulse, 0.0)
    generalized = _tree_where(successful, candidate_generalized, zero_generalized)
    update = _tree_where(successful, candidate_update, zero_update)
    post = jax.tree.map(
        lambda free, increment: free + increment, prepared.free_velocity, update
    )
    post_contact_velocity = jnp.where(
        successful,
        candidate_contact_velocity,
        prepared.velocity_operator.mv(prepared.free_velocity),
    )
    zero_update_output = (
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
    fail_closed = successful | (
        zero_update_output & zero_generalized_output & zero_applied
    )
    evidence = ArticulatedContactEvidence(
        preparation=prepared.evidence,
        cone=cone_result.evidence,
        duality=duality,
        minimum_law_normal_velocity=minimum_law_normal,
        minimum_physical_normal_velocity=minimum_physical_normal,
        law_feasible=law_feasible,
        numeric_revision_matches=numeric_revision_matches,
        contact_equation_residual=contact_equation_residual,
        contact_equation_valid=contact_equation_valid,
        physical_velocity_residual=physical_velocity_residual,
        physical_velocity_valid=physical_velocity_valid,
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
    initial_impulse: ConvertibleToArray | None = None,
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
    "build_mixed_contact_velocity_operator",
    "certify_inverse_mass_operator",
    "InverseMassOperatorCertificate",
    "contact_duality_evidence",
    "prepare_articulated_contact",
    "solve_articulated_contact",
]
