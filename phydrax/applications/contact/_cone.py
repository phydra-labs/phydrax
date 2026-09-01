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
    free_velocity: Array
    effective_mass: Array
    compliance: Array
    friction: Array
    route_keys: Array
    valid: Array
    tangent_dimension: int = eqx.field(static=True)
    program_id: str = eqx.field(static=True)

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


class ContactConeResult(StrictModule):
    impulse: Array
    post_relative_velocity: Array
    evidence: ContactConeEvidence


def project_coulomb_cone(impulse: ArrayLike, friction: ArrayLike, /) -> Array:
    value = jnp.asarray(impulse)
    coefficient = jnp.asarray(friction, dtype=value.dtype)
    normal = value[..., 0]
    tangent = value[..., 1:]
    tangent_norm = jnp.sqrt(jnp.sum(tangent * tangent, axis=-1))
    inside = (normal >= 0.0) & (tangent_norm <= coefficient * normal)
    boundary_normal = (normal + coefficient * tangent_norm) / (
        1.0 + coefficient * coefficient
    )
    positive = boundary_normal > 0.0
    safe_norm = jnp.maximum(tangent_norm, jnp.finfo(value.dtype).eps)
    boundary_tangent = (coefficient * boundary_normal / safe_norm)[..., None] * tangent
    projected = jnp.concatenate((boundary_normal[..., None], boundary_tangent), axis=-1)
    zero = jnp.zeros_like(value)
    return jnp.where(
        inside[..., None],
        value,
        jnp.where(positive[..., None], projected, zero),
    )


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
    if compliance_.shape != (size,) or bool(jnp.any(compliance_ < 0.0)):
        raise ValueError("Cone compliance must be nonnegative and match the program.")
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
        valid & parameters.mechanical_available,
        tangent_dimension,
        program_id,
    )


def solve_contact_cone(
    program: ContactConeProgram,
    /,
    *,
    solver: ContactConeSolverPlan | None = None,
    initial_impulse: ArrayLike | None = None,
) -> ContactConeResult:
    if not isinstance(program, ContactConeProgram):
        raise TypeError("program must be ContactConeProgram.")
    solver_ = ContactConeSolverPlan() if solver is None else solver
    if not isinstance(solver_, ContactConeSolverPlan):
        raise TypeError("solver must be ContactConeSolverPlan or None.")
    count = program.contact_count
    local_dimension = program.local_dimension
    size = count * local_dimension
    matrix = program.effective_mass + jnp.diag(program.compliance)
    free = program.free_velocity.reshape((-1,))
    impulse = (
        jnp.zeros((count, local_dimension), dtype=free.dtype)
        if initial_impulse is None
        else jnp.asarray(initial_impulse, dtype=free.dtype)
    )
    if impulse.shape != (count, local_dimension):
        raise ValueError("initial_impulse has invalid shape.")
    impulse = project_coulomb_cone(impulse, program.friction)
    diagonal = jnp.diag(matrix).reshape((count, local_dimension))
    step = 1.0 / jnp.maximum(
        diagonal,
        jnp.finfo(matrix.dtype).eps,
    )
    valid = program.valid[:, None]
    initial_scale = jnp.maximum(1.0, jnp.sqrt(jnp.sum(free * free)))
    tolerance = solver_.absolute_tolerance + solver_.relative_tolerance * initial_scale

    def body(index, state):
        value, converged, first_converged, residual_norm = state
        flat = value.reshape((-1,))
        gradient = (matrix @ flat + free).reshape(value.shape)
        trial = value - solver_.relaxation * step * gradient
        projected = project_coulomb_cone(trial, program.friction)
        projected = jnp.where(valid, projected, 0.0)
        residual = value - projected
        norm = jnp.sqrt(jnp.sum(residual * residual))
        now = norm <= tolerance
        first = jnp.where((~converged) & now, index + 1, first_converged)
        next_value = jnp.where(converged, value, projected)
        return next_value, converged | now, first, norm

    impulse, converged, first_converged, residual_norm = jax.lax.fori_loop(
        0,
        solver_.maximum_iterations,
        body,
        (
            impulse,
            jnp.asarray(False),
            jnp.asarray(solver_.maximum_iterations, dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=free.dtype),
        ),
    )
    flat_impulse = impulse.reshape((-1,))
    post = (program.effective_mass @ flat_impulse + free).reshape(impulse.shape)
    normal_impulse = impulse[:, 0]
    tangent_impulse = impulse[:, 1:]
    tangent_norm = jnp.sqrt(jnp.sum(tangent_impulse * tangent_impulse, axis=-1))
    cone_defect = jnp.max(
        jnp.where(
            program.valid,
            jnp.maximum(tangent_norm - program.friction * normal_impulse, 0.0),
            0.0,
        ),
        initial=0.0,
    )
    complementarity = jnp.max(
        jnp.where(
            program.valid,
            jnp.abs(jnp.minimum(normal_impulse, post[:, 0])),
            0.0,
        ),
        initial=0.0,
    )
    dissipated = -jnp.sum(tangent_impulse * program.free_velocity[:, 1:])
    finite = jnp.all(jnp.isfinite(impulse)) & jnp.all(jnp.isfinite(post))
    successful = (
        converged
        & finite
        & (cone_defect <= tolerance)
        & (jnp.min(jnp.where(program.valid, normal_impulse, jnp.inf), initial=0.0) >= 0.0)
    )
    evidence = ContactConeEvidence(
        converged,
        first_converged,
        residual_norm,
        complementarity,
        cone_defect,
        jnp.min(jnp.where(program.valid, normal_impulse, jnp.inf), initial=0.0),
        dissipated,
        finite,
        successful,
        program.program_id,
        solver_.plan_id,
    )
    return ContactConeResult(impulse, post, evidence)


__all__ = [
    "ContactConeEvidence",
    "ContactConeProgram",
    "ContactConeResult",
    "ContactConeSolverPlan",
    "build_contact_cone_program",
    "project_coulomb_cone",
    "solve_contact_cone",
]
