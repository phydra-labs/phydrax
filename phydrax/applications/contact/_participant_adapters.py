#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._articulated_geometry import (
    ArticulatedCollisionGeometry,
    ArticulatedContactTrajectory,
    ArticulatedLiftedPath,
)
from ...discretization.contact._guarantee import (
    ContactCapability,
    ContactGuaranteeEvidence,
    ContactGuaranteeLevel,
)
from ...discretization.contact._iga import PreparedIGASplinePatchProxy
from ...discretization.contact._inversion import (
    simplex_inversion_step_limit,
    SimplexInversionStepPlan,
)
from ...discretization.contact._participant import (
    AbstractContactParticipant,
    ContactParticipantScene,
    FunctionContactParticipant,
    LinearContactParticipant,
    ParticipantTrajectoryBounds,
)
from ...discretization.contact._precision import ContactPrecisionPolicy
from ...discretization.contact._surface import (
    CollisionSurfacePlan,
    ContactPairPolicy,
    prepare_cell_mesh_collision_surface,
    PreparedCollisionSurface,
    selection_collision_operator,
)
from ...discretization.fem._generic import FiniteElementDiscretization
from ...discretization.mpm._transfer import gather_apic
from ...discretization.mpm._types import MPMRuntimeState
from ...discretization.particle._reduced_articulation import (
    PreparedReducedArticulation,
)
from ...equations._finite_element_material import MaterialTransaction
from ...equations._finite_element_variational import (
    CompiledFiniteElementProblem,
    FiniteElementExecutionContext,
)
from ...equations.fem._execution import FiniteElementMassPolicy
from ...linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    ArraySpace,
    BlockSpace,
    DualSpace,
    FunctionLinearOperator,
)
from ...solver._material_point_implicit import PreparedImplicitMPMResidual
from ...typing import checked
from ..solid_mechanics._fem_dynamics import (
    FiniteElementDynamicsExecutionContext,
    FiniteElementDynamicsState,
    ImplicitNewmarkMethod,
)
from ._implicit_participant import (
    AbstractImplicitContactParticipant,
    ContactParticipantEvaluation,
    ContactParticipantStepLimit,
)
from ._rod_capsule import ReducedRodCapsuleContactParticipant


def _rigid_rotation(rotation_vector: Array) -> Array:
    dimension = rotation_vector.shape[-1]
    if dimension == 1:
        angle = rotation_vector[..., 0]
        cosine = jnp.cos(angle)
        sine = jnp.sin(angle)
        return jnp.stack((cosine, -sine, sine, cosine), axis=-1).reshape(
            rotation_vector.shape[:-1] + (2, 2)
        )
    angle_squared = jnp.sum(rotation_vector * rotation_vector, axis=-1)
    angle = jnp.sqrt(jnp.maximum(angle_squared, jnp.finfo(rotation_vector.dtype).eps))
    safe_angle = jnp.where(angle > 0.0, angle, 1.0)
    axis = rotation_vector / safe_angle[..., None]
    x, y, z = (axis[..., index] for index in range(3))
    zero = jnp.zeros_like(x)
    skew = jnp.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1).reshape(
        rotation_vector.shape[:-1] + (3, 3)
    )
    identity = jnp.broadcast_to(jnp.eye(3, dtype=rotation_vector.dtype), skew.shape)
    sine_scale = jnp.where(
        angle_squared > jnp.finfo(rotation_vector.dtype).eps,
        jnp.sin(angle),
        angle - angle**3 / 6.0,
    )
    cosine_scale = jnp.where(
        angle_squared > jnp.finfo(rotation_vector.dtype).eps,
        1.0 - jnp.cos(angle),
        0.5 * angle_squared - angle_squared**2 / 24.0,
    )
    return (
        identity
        + sine_scale[..., None, None] * skew
        + cosine_scale[..., None, None] * contract("...ij,...jk->...ik", skew, skew)
    )


class RigidContactParticipant(AbstractContactParticipant):
    plan: CollisionSurfacePlan
    local_vertices: Array
    vertex_owner: Array
    body_count: int = eqx.field(static=True)
    contact_space: ArraySpace
    space: BlockSpace
    _participant_id: str = eqx.field(static=True)
    _capabilities: ContactCapability = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: CollisionSurfacePlan,
        local_vertices: ArrayLike,
        vertex_owner: ArrayLike,
        /,
        *,
        body_count: int,
    ) -> None:
        vertices = jnp.asarray(local_vertices)
        owner = np.asarray(vertex_owner)
        bodies = int(body_count)
        if vertices.shape != (
            plan.vertex_count,
            plan.ambient_dimension,
        ):
            raise ValueError("Rigid local vertices do not match collision topology.")
        if owner.shape != (plan.vertex_count,) or not np.issubdtype(
            owner.dtype, np.integer
        ):
            raise TypeError("Rigid vertex owners must be one integer vector.")
        for primitive in (np.asarray(plan.edges), np.asarray(plan.faces)):
            primitive_owner = owner[primitive]
            if np.any(primitive_owner != primitive_owner[:, :1]):
                raise ValueError(
                    "Rigid collision primitives cannot span distinct body owners."
                )
        if bodies <= 0 or np.any(owner < 0) or np.any(owner >= bodies):
            raise ValueError("Rigid vertex owner is invalid.")
        angular_dimension = 1 if plan.ambient_dimension == 2 else 3
        space = BlockSpace(
            (
                ArraySpace((bodies, plan.ambient_dimension), dtype=vertices.dtype),
                ArraySpace((bodies, angular_dimension), dtype=vertices.dtype),
            ),
            names=("translation", "rotation"),
        )
        self.plan = plan
        self.local_vertices = vertices
        self.vertex_owner = jnp.asarray(owner, dtype=jnp.int32)
        self.body_count = bodies
        self.contact_space = ArraySpace(vertices.shape, dtype=vertices.dtype)
        self.space = space
        self._participant_id = canonical_fingerprint(
            {
                "kind": "rigid-contact-participant",
                "surface": plan.topology_id,
                "body_count": bodies,
                "owner": owner.tolist(),
            }
        )
        self._capabilities = (
            ContactCapability.STATIC_DISTANCE
            | ContactCapability.NONLINEAR_TRAJECTORY
            | ContactCapability.DIFFERENTIABLE_KINEMATICS
            | ContactCapability.EFFORT_PULLBACK
        )

    @property
    def source_space(self) -> AbstractVectorSpace:
        return self.space

    @property
    def tangent_space(self) -> AbstractVectorSpace:
        return self.space

    @property
    def contact_velocity_space(self) -> ArraySpace:
        return self.contact_space

    @property
    def surface_plan(self) -> CollisionSurfacePlan:
        return self.plan

    @property
    def participant_id(self) -> str:
        return self._participant_id

    @property
    def capabilities(self) -> ContactCapability:
        return self._capabilities

    def positions(self, state: PyTree, /) -> Array:
        translation, rotation_vector = self.space.validate(state)
        rotation = _rigid_rotation(rotation_vector)
        world_offset = contract(
            "vij,vj->vi",
            rotation[self.vertex_owner],
            self.local_vertices,
        )
        return translation[self.vertex_owner] + world_offset

    def velocities(self, state: PyTree, rates: PyTree, /) -> Array:
        translation, rotation_vector = self.space.validate(state)
        linear_velocity, angular_velocity = self.space.validate(rates)
        rotation = _rigid_rotation(rotation_vector)
        world_offset = contract(
            "vij,vj->vi",
            rotation[self.vertex_owner],
            self.local_vertices,
        )
        if self.plan.ambient_dimension == 2:
            spin = angular_velocity[self.vertex_owner, 0]
            rotational = jnp.stack(
                (-spin * world_offset[:, 1], spin * world_offset[:, 0]),
                axis=-1,
            )
        else:
            rotational = jnp.cross(angular_velocity[self.vertex_owner], world_offset)
        return linear_velocity[self.vertex_owner] + rotational

    def effort_pullback(
        self, state: PyTree, surface_effort: ArrayLike, /
    ) -> PyTree[Array]:
        translation, rotation_vector = self.space.validate(state)
        del translation
        effort = self.contact_effort_space.validate(
            jnp.asarray(surface_effort, dtype=self.local_vertices.dtype)
        )
        rotation = _rigid_rotation(rotation_vector)
        arm = contract(
            "vij,vj->vi",
            rotation[self.vertex_owner],
            self.local_vertices,
        )
        body_force = (
            jnp.zeros((self.body_count, self.plan.ambient_dimension), dtype=effort.dtype)
            .at[self.vertex_owner]
            .add(effort)
        )
        if self.plan.ambient_dimension == 2:
            torque_value = arm[:, 0] * effort[:, 1] - arm[:, 1] * effort[:, 0]
            body_torque = (
                jnp.zeros((self.body_count, 1), dtype=effort.dtype)
                .at[self.vertex_owner, 0]
                .add(torque_value)
            )
        else:
            body_torque = (
                jnp.zeros((self.body_count, 3), dtype=effort.dtype)
                .at[self.vertex_owner]
                .add(jnp.cross(arm, effort))
            )
        return DualSpace(self.tangent_space).validate((body_force, body_torque))

    def trajectory_bounds(
        self, start_state: PyTree, end_state: PyTree, /
    ) -> ParticipantTrajectoryBounds:
        start_translation, _ = self.space.validate(start_state)
        end_translation, _ = self.space.validate(end_state)
        radius = (
            jnp.zeros((self.body_count,), dtype=self.local_vertices.dtype)
            .at[self.vertex_owner]
            .max(jnp.sqrt(jnp.sum(self.local_vertices * self.local_vertices, axis=-1)))
        )
        lower_center = jnp.minimum(start_translation, end_translation)
        upper_center = jnp.maximum(start_translation, end_translation)
        lower = lower_center[self.vertex_owner] - radius[self.vertex_owner, None]
        upper = upper_center[self.vertex_owner] + radius[self.vertex_owner, None]
        finite = jnp.all(jnp.isfinite(lower)) & jnp.all(jnp.isfinite(upper))
        return ParticipantTrajectoryBounds(
            lower,
            upper,
            jnp.asarray(
                int(ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE),
                dtype=jnp.int32,
            ),
            finite,
            finite,
            self.participant_id,
        )


def make_articulated_contact_participant(
    plan: CollisionSurfacePlan,
    source_space: AbstractVectorSpace,
    forward_kinematics: Callable[[PyTree], Array],
    /,
    *,
    tangent_space: AbstractVectorSpace | None = None,
    velocity_action: Callable[[PyTree, PyTree], Array] | None = None,
    effort_pullback_action: Callable[[PyTree, Array], PyTree] | None = None,
    bounds_action: Callable[[PyTree, PyTree], tuple[Array, Array]] | None = None,
    participant_id: str | None = None,
) -> FunctionContactParticipant:
    tangent = source_space if tangent_space is None else tangent_space
    return FunctionContactParticipant(
        plan,
        source_space,
        forward_kinematics,
        tangent_space=tangent,
        velocity_action=velocity_action,
        effort_pullback_action=effort_pullback_action,
        bounds_action=bounds_action,
        participant_id=participant_id,
    )


@final
class ArticulatedPrimitiveContactParticipant(AbstractContactParticipant):
    """Body-attached spheres/capsules of one prepared reduced articulation.

    The source state is the lifted generalized configuration; its tangent is
    the articulation's generalized velocity space shared with the mass and
    Jacobian operators. Contact vertices are sphere centers and capsule axis
    endpoints placed by native forward kinematics. Endpoint trajectory bounds
    use the lifted straight coordinate path ``q(s) = q0 + s (q1 - q0)``;
    contact steps bind their exact segment path through :meth:`trajectory`.
    """

    articulation: PreparedReducedArticulation
    geometry: ArticulatedCollisionGeometry
    configuration_space: ArraySpace
    contact_space: ArraySpace
    _participant_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        articulation: PreparedReducedArticulation,
        geometry: ArticulatedCollisionGeometry,
        /,
    ) -> None:
        if geometry.articulation_id != articulation.prepared_id:
            raise ValueError("Collision geometry belongs to another articulation.")
        dtype = articulation.reference_position.dtype
        self.articulation = articulation
        self.geometry = geometry
        self.configuration_space = ArraySpace(
            (articulation.nq,),
            dtype=dtype,
            space_id=f"{articulation.prepared_id}:lifted-configuration",
        )
        self.contact_space = ArraySpace((geometry.vertex_count, 3), dtype=dtype)
        self._participant_id = canonical_fingerprint(
            {
                "kind": "articulated-primitive-contact-participant",
                "geometry": geometry.geometry_id,
            }
        )

    @property
    def source_space(self) -> AbstractVectorSpace:
        return self.configuration_space

    @property
    def tangent_space(self) -> AbstractVectorSpace:
        return self.articulation.generalized_velocity_space

    @property
    def contact_velocity_space(self) -> ArraySpace:
        return self.contact_space

    @property
    def surface_plan(self) -> CollisionSurfacePlan:
        return self.geometry.surface_plan

    @property
    def participant_id(self) -> str:
        return self._participant_id

    @property
    def capabilities(self) -> ContactCapability:
        return (
            ContactCapability.STATIC_DISTANCE
            | ContactCapability.NONLINEAR_TRAJECTORY
            | ContactCapability.DIFFERENTIABLE_KINEMATICS
            | ContactCapability.EFFORT_PULLBACK
        )

    def positions(self, state: PyTree[Any], /) -> Array:
        configuration = self.source_space.validate(state)
        return self.geometry.vertex_positions(self.articulation, configuration)

    def velocities(self, state: PyTree[Any], rates: PyTree[Any], /) -> Array:
        configuration = self.source_space.validate(state)
        velocity = self.tangent_space.validate(rates)
        return self.contact_velocity_space.validate(
            self.geometry.vertex_velocities(self.articulation, configuration, velocity)
        )

    def effort_pullback(
        self, state: PyTree[Any], surface_effort: ArrayLike, /
    ) -> PyTree[Array]:
        configuration = self.source_space.validate(state)
        effort = self.contact_effort_space.validate(
            jnp.asarray(surface_effort, dtype=self.contact_velocity_space.dtype)
        )
        return self.effort_space.validate(
            self.geometry.vertex_force_pullback(self.articulation, configuration, effort)
        )

    def trajectory(
        self,
        start_configuration: ArrayLike,
        velocity: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        revision: ArrayLike = 0,
    ) -> ArticulatedContactTrajectory:
        """Bind the segment path ``q(s) = q0 + s h v`` used by every consumer."""
        return ArticulatedContactTrajectory(
            self.geometry,
            ArticulatedLiftedPath(
                self.articulation,
                start_configuration,
                velocity,
                step_size,
                revision=revision,
            ),
        )

    def trajectory_bounds(
        self, start_state: PyTree[Any], end_state: PyTree[Any], /
    ) -> ParticipantTrajectoryBounds:
        start = self.source_space.validate(start_state)
        end = self.source_space.validate(end_state)
        enclosure = self.trajectory(start, end - start, 1.0).enclosure(0.0, 1.0)
        return ParticipantTrajectoryBounds(
            enclosure.lower,
            enclosure.upper,
            enclosure.guarantee_level,
            enclosure.finite,
            enclosure.successful,
            self.participant_id,
        )


@checked
def prepare_articulated_contact_scene(
    articulation: PreparedReducedArticulation,
    geometry: ArticulatedCollisionGeometry,
    /,
) -> ContactParticipantScene:
    """One-participant contact scene with the geometry's static planes."""
    return ContactParticipantScene(
        (ArticulatedPrimitiveContactParticipant(articulation, geometry),),
        planes=geometry.planes,
    )


def prepare_point_contact_participant(
    source_space: ArraySpace,
    reference_positions: ArrayLike,
    /,
    *,
    vertex_ids: ArrayLike | None = None,
    participant_ids: ArrayLike | int = 0,
    body_ids: ArrayLike | int = 0,
    material_ids: ArrayLike | int | None = None,
    patch_ids: ArrayLike | int = 0,
    static_mask: ArrayLike | bool = False,
    physical_radius: ArrayLike | float = 0.0,
    solver_clearance: ArrayLike | float = 0.0,
    proxy_error: ArrayLike | float = 0.0,
    allowed_participant_pairs: ArrayLike | None = None,
    excluded_vertex_pairs: ArrayLike | None = None,
    precision: ContactPrecisionPolicy | None = None,
) -> LinearContactParticipant:
    reference = jnp.asarray(reference_positions)
    if reference.ndim != 2 or reference.shape != source_space.shape:
        raise ValueError(
            "Point contact reference positions must match the source ArraySpace."
        )
    count, dimension = reference.shape
    identifiers = (
        jnp.arange(count, dtype=jnp.int64)
        if vertex_ids is None
        else jnp.asarray(vertex_ids)
    )
    pair_policy = ContactPairPolicy(
        count,
        allowed_participant_pairs=allowed_participant_pairs,
        excluded_vertex_pairs=excluded_vertex_pairs,
    )
    plan = CollisionSurfacePlan(
        identifiers,
        ambient_dimension=dimension,
        edges=jnp.empty((0, 2), dtype=jnp.int32),
        codimensional_mask=jnp.ones((count,), dtype=jnp.bool_),
        pair_policy=pair_policy,
        participant_ids=participant_ids,
        body_ids=body_ids,
        material_ids=material_ids,
        patch_ids=patch_ids,
        static_mask=static_mask,
        physical_radius=physical_radius,
        solver_clearance=solver_clearance,
        proxy_error=proxy_error,
        allow_isolated_vertices=True,
    )
    surface = PreparedCollisionSurface(
        plan,
        reference,
        selection_collision_operator(source_space, jnp.arange(count, dtype=jnp.int32)),
        precision=precision,
    )
    return LinearContactParticipant(surface)


def prepare_mpm_contact_participant(
    plan: CollisionSurfacePlan,
    reference_positions: ArrayLike,
    grid_to_contact_operator: AbstractLinearOperator,
    /,
    *,
    precision: ContactPrecisionPolicy | None = None,
) -> LinearContactParticipant:
    surface = PreparedCollisionSurface(
        plan,
        reference_positions,
        grid_to_contact_operator,
        precision=precision,
    )
    return LinearContactParticipant(surface)


@final
class FiniteElementContactParticipant(AbstractImplicitContactParticipant):
    problem: CompiledFiniteElementProblem
    mass_problem: CompiledFiniteElementProblem
    method: ImplicitNewmarkMethod
    mass_policy: FiniteElementMassPolicy
    inversion: SimplexInversionStepPlan | None
    geometry: LinearContactParticipant
    _participant_id: str = eqx.field(static=True)

    @property
    def participant_id(self) -> str:
        return self._participant_id

    @property
    def dof_space(self) -> ArraySpace:
        space = self.problem.state_space
        if not isinstance(space, ArraySpace):
            raise TypeError("FEM contact requires one array field.")
        return space

    @property
    def has_potential(self) -> bool:
        return self.problem.potential_compatible

    def _history(self, history: PyTree[Array], /) -> FiniteElementDynamicsState:
        if not isinstance(history, FiniteElementDynamicsState):
            raise TypeError("FEM contact history must be FiniteElementDynamicsState.")
        self.dof_space.validate(history.velocity)
        self.dof_space.validate(history.acceleration)
        return history

    def _context(
        self,
        history: FiniteElementDynamicsState,
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> FiniteElementExecutionContext:
        base = self.problem._execution_context(args)
        dt = jnp.zeros_like(time) if step_size is None else step_size
        dynamics = FiniteElementDynamicsExecutionContext(
            history.materials, time, time + dt, dt, base.user_args
        )
        return FiniteElementExecutionContext(
            base.runtime,
            time=time + dt,
            lift=base.lift,
            lift_rate=base.lift_rate,
            lift_acceleration=base.lift_acceleration,
            metric_data=base.metric_data,
            user_args=dynamics,
        )

    def _rates(
        self,
        state: Array,
        history: FiniteElementDynamicsState,
        accepted_state: Array,
        step_size: Array,
        /,
    ) -> tuple[Array, Array]:
        predictor = (
            accepted_state
            + step_size * history.velocity
            + step_size**2 * (0.5 - self.method.beta) * history.acceleration
        )
        acceleration = (state - predictor) / (self.method.beta * step_size**2)
        velocity = history.velocity + step_size * (
            (1.0 - self.method.gamma) * history.acceleration
            + self.method.gamma * acceleration
        )
        return velocity, acceleration

    def evaluate(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> ContactParticipantEvaluation:
        value = self.dof_space.validate(state)
        accepted = self.dof_space.validate(accepted_state)
        history = self._history(committed_history)
        context = self._context(history, time, step_size, args)
        residual, auxiliary = self.problem.residual_with_auxiliary(value, context)
        residual = self.problem.residual_space.validate(residual)
        velocity, acceleration = history.velocity, history.acceleration
        valid = (
            auxiliary.valid & ~auxiliary.retry_requested & jnp.all(jnp.isfinite(residual))
        )
        if step_size is not None:
            velocity, acceleration = self._rates(value, history, accepted, step_size)
            _, mass = self.problem._mass_operators(
                context,
                jnp.ones((), dtype=value.dtype),
                self.mass_policy,
                self.mass_problem,
            )
            residual = residual + mass.mv(acceleration)
            valid = valid & jnp.isfinite(step_size) & (step_size > 0.0)
        materials = history.materials
        if auxiliary.trial_state is not None:
            if not isinstance(auxiliary.trial_state, MaterialTransaction):
                raise TypeError(
                    "FEM auxiliary trial history must be MaterialTransaction."
                )
            if (
                materials is None
                or materials.layout_id != auxiliary.trial_state.layout_id
            ):
                raise ValueError("FEM material trial changed committed history layout.")
            materials = auxiliary.trial_state
        trial = FiniteElementDynamicsState(
            value,
            velocity,
            acceleration,
            materials=materials,
            time=time if step_size is None else time + step_size,
            step=history.step + jnp.asarray(step_size is not None, dtype=jnp.int32),
            state_version=history.state_version + 1,
        )
        minimum = jnp.asarray(jnp.inf, dtype=value.dtype)
        if self.inversion is not None:
            from ...linalg import determinant_small_linear, SmallLinearSolvePlan

            full = self.problem.full_space.validate(self.problem.expand(value, context))
            if (
                not isinstance(full, Array)
                or full.shape != self.inversion.reference_positions.shape
            ):
                raise ValueError(
                    "FEM inversion requires nodal displacements matching reference positions."
                )
            current = self.inversion.reference_positions + full
            cells = current[self.inversion.cells]
            reference = self.inversion.reference_positions[self.inversion.cells]
            small = SmallLinearSolvePlan(self.inversion.dimension)
            ratio = determinant_small_linear(
                small, cells[:, 1:] - cells[:, :1]
            ) / determinant_small_linear(small, reference[:, 1:] - reference[:, :1])
            minimum = jnp.min(ratio)
            valid = valid & jnp.all(ratio > self.inversion.minimum_ratio)
        energy = self.potential(value, history, accepted, time, step_size, args)
        defect = jnp.zeros((), dtype=value.dtype)
        if self.has_potential:
            gradient = jax.grad(self._potential_value)(
                value, history, accepted, time, step_size, args
            )
            scale = jnp.maximum(
                jnp.asarray(1.0, dtype=value.dtype),
                jnp.sqrt(jnp.sum(residual * residual)),
            )
            defect = jnp.sqrt(jnp.sum((gradient - residual) ** 2)) / scale
        return ContactParticipantEvaluation(
            residual,
            trial,
            energy,
            valid,
            valid,
            valid,
            jnp.asarray(self.has_potential, dtype=jnp.bool_),
            defect,
            auxiliary.suggested_step,
            jnp.where(valid, jnp.int32(0), jnp.int32(1)),
            {"minimum_jacobian": minimum},
        )

    def _potential_value(
        self,
        state: Array,
        history: FiniteElementDynamicsState,
        accepted: Array,
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> Array:
        context = self._context(history, time, step_size, args)
        energy = self.problem.potential(state, context)
        if step_size is not None:
            predictor = (
                accepted
                + step_size * history.velocity
                + step_size**2 * (0.5 - self.method.beta) * history.acceleration
            )
            delta = state - predictor
            _, mass = self.problem._mass_operators(
                context,
                jnp.ones((), dtype=state.dtype),
                self.mass_policy,
                self.mass_problem,
            )
            energy = energy + jnp.sum(delta * mass.mv(delta)) / (
                2.0 * self.method.beta * step_size**2
            )
        return energy

    def potential(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> Array | None:
        if not self.has_potential:
            return None
        return self._potential_value(
            self.dof_space.validate(state),
            self._history(committed_history),
            self.dof_space.validate(accepted_state),
            time,
            step_size,
            args,
        )

    def residual_jvp(
        self,
        state: PyTree[Array],
        direction: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> Array:
        return jax.jvp(
            lambda q: (
                self.evaluate(
                    q, committed_history, accepted_state, time, step_size, args
                ).residual
            ),
            (self.dof_space.validate(state),),
            (self.dof_space.validate(direction),),
        )[1]

    def potential_hvp(
        self,
        state: PyTree[Array],
        direction: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> Array | None:
        if not self.has_potential:
            return None
        return jax.jvp(
            lambda q: jax.grad(self._potential_value)(
                q,
                self._history(committed_history),
                self.dof_space.validate(accepted_state),
                time,
                step_size,
                args,
            ),
            (self.dof_space.validate(state),),
            (self.dof_space.validate(direction),),
        )[1]

    def contact_velocity(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> Array:
        del time, args
        return self._rates(
            self.dof_space.validate(state),
            self._history(committed_history),
            self.dof_space.validate(accepted_state),
            step_size,
        )[0]

    def contact_velocity_scale(self, step_size: Array, /) -> Array:
        return self.method.position_to_velocity_scale(step_size, step_size.dtype)

    def step_limit(
        self,
        state: PyTree[Array],
        direction: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> ContactParticipantStepLimit:
        del accepted_state, time, step_size
        value = self.dof_space.validate(state)
        delta = self.dof_space.validate(direction)
        if self.inversion is None:
            finite = jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(delta))
            guarantee = ContactGuaranteeEvidence(
                ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE,
                finite=finite,
                backend_id=self.participant_id,
            )
            return ContactParticipantStepLimit(
                jnp.where(
                    finite,
                    jnp.ones((), dtype=value.dtype),
                    jnp.zeros((), dtype=value.dtype),
                ),
                finite,
                finite,
                guarantee,
                jnp.where(finite, jnp.int32(0), jnp.int32(1)),
                jnp.int32(-1),
                self.participant_id,
            )
        start = self.problem.full_space.validate(self.problem.expand(value, args))
        end = self.problem.full_space.validate(self.problem.expand(value + delta, args))
        if not isinstance(start, Array) or not isinstance(end, Array):
            raise TypeError("Simplex inversion requires array-valued nodal displacement.")
        evidence = simplex_inversion_step_limit(
            self.inversion,
            self.inversion.reference_positions + start,
            self.inversion.reference_positions + end,
        )
        guarantee = ContactGuaranteeEvidence(
            ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE,
            finite=evidence.finite,
            work_complete=evidence.successful,
            failure_code=evidence.status,
            margin=evidence.minimum_initial_ratio,
            backend_id=self.inversion.plan_id,
        )
        return ContactParticipantStepLimit(
            evidence.step_size,
            evidence.successful,
            evidence.conservative,
            guarantee,
            evidence.status,
            evidence.limiting_cell,
            self.participant_id,
        )


@final
class _FiniteElementSurfaceAction(StrictModule, NonTrainableState):
    problem: CompiledFiniteElementProblem
    surface: PreparedCollisionSurface
    template: Array

    def __call__(self, direction: Array, /) -> Array:
        full_direction = jax.jvp(self.problem.expand, (self.template,), (direction,))[1]
        return self.surface.displacement_operator.mv(full_direction)


@final
class _FiniteElementSurfaceTranspose(StrictModule, NonTrainableState):
    action: _FiniteElementSurfaceAction

    def __call__(self, effort: Array, /) -> Array:
        return jax.linear_transpose(self.action, self.action.template)(effort)[0]


@checked
def prepare_finite_element_contact_participant(
    problem: CompiledFiniteElementProblem,
    /,
    *,
    method: ImplicitNewmarkMethod | None = None,
    mass_policy: FiniteElementMassPolicy | None = None,
    inversion: SimplexInversionStepPlan | None = None,
) -> FiniteElementContactParticipant:
    if len(problem.form.field_names) != 1 or not isinstance(
        problem.state_space, ArraySpace
    ):
        raise ValueError("FEM contact mechanics requires one array-valued field.")
    selected_mass = FiniteElementMassPolicy() if mass_policy is None else mass_policy
    selected_method = ImplicitNewmarkMethod() if method is None else method
    identifier = canonical_fingerprint(
        {
            "kind": "fem-contact-mechanics",
            "compilation": problem.compilation_id,
            "mass": selected_mass.kind,
            "method": selected_method.method_id,
            "inversion": None if inversion is None else inversion.plan_id,
        }
    )
    discretization = problem.discretization
    full_space = problem.full_space
    if not isinstance(discretization, FiniteElementDiscretization) or not isinstance(
        full_space, ArraySpace
    ):
        raise ValueError(
            "Automatic FEM collision geometry requires a nodal cell-mesh discretization."
        )
    full_surface = prepare_cell_mesh_collision_surface(discretization.mesh, full_space)
    if problem.constraint is None:
        geometry = LinearContactParticipant(full_surface)
    else:
        template = jnp.zeros(problem.state_space.shape, dtype=problem.state_space.dtype)
        action = _FiniteElementSurfaceAction(problem, full_surface, template)
        operator = FunctionLinearOperator(
            action,
            source=problem.state_space,
            target=full_surface.contact_velocity_space,
            transpose_action=_FiniteElementSurfaceTranspose(action),
            operator_id=f"{identifier}:constrained-surface",
        )
        geometry = LinearContactParticipant(
            PreparedCollisionSurface(
                full_surface.plan,
                full_surface.positions(problem.expand(template)),
                operator,
                precision=full_surface.precision,
            )
        )
    return FiniteElementContactParticipant(
        problem,
        problem._compile_unit_mass_problem(selected_mass),
        selected_method,
        selected_mass,
        inversion,
        geometry,
        identifier,
    )


@final
class _IGAPositions(StrictModule, NonTrainableState):
    proxy: PreparedIGASplinePatchProxy

    def __call__(self, state: Array, /) -> Array:
        return self.proxy.positions(state)


@final
class _IGABounds(StrictModule, NonTrainableState):
    proxy: PreparedIGASplinePatchProxy

    def __call__(self, start: Array, end: Array, /) -> tuple[Array, Array]:
        first, second = self.proxy.positions(start), self.proxy.positions(end)
        inflation = self.proxy.proxy.approximation_error[:, None]
        return jnp.minimum(first, second) - inflation, jnp.maximum(
            first, second
        ) + inflation


@checked
def prepare_iga_contact_participant(
    proxy: PreparedIGASplinePatchProxy, source_space: ArraySpace, /
) -> FunctionContactParticipant:
    if not source_space.compatible(proxy.proxy.surface.source_space):
        raise ValueError("IGA source space must match certified proxy control space.")
    if not bool(np.asarray(proxy.proxy.evidence.certified)):
        raise ValueError("Implicit IGA contact requires a certified proxy.")
    return FunctionContactParticipant(
        proxy.proxy.surface.plan,
        source_space,
        _IGAPositions(proxy),
        tangent_space=source_space,
        bounds_action=_IGABounds(proxy),
        participant_id=f"{proxy.plan_id}:participant",
    )


@final
class _RodPositions(StrictModule, NonTrainableState):
    geometry: AbstractContactParticipant

    def __call__(self, state: PyTree[Array], /) -> Array:
        return self.geometry.positions(state)


@final
class _RodVelocity(StrictModule, NonTrainableState):
    geometry: AbstractContactParticipant

    def __call__(self, state: PyTree[Array], direction: PyTree[Array], /) -> Array:
        return self.geometry.velocities(state, direction)


@final
class _RodPullback(StrictModule, NonTrainableState):
    geometry: AbstractContactParticipant

    def __call__(self, state: PyTree[Array], effort: Array, /) -> PyTree[Array]:
        return self.geometry.effort_pullback(state, effort)


@checked
def prepare_implicit_reduced_rod_contact_geometry(
    geometry: ReducedRodCapsuleContactParticipant,
    /,
    *,
    bounds_action: Callable[[PyTree[Array], PyTree[Array]], tuple[Array, Array]]
    | None = None,
) -> FunctionContactParticipant:
    """Admit the existing exact rod map only with a conservative full-path bound.

    Endpoint capsule positions alone are not an enclosure of the nonlinear rod
    chart. The caller's bound owns every intermediate chart/segment state.
    """
    if bounds_action is None:
        raise ValueError(
            "Implicit reduced-rod contact requires a conservative nonlinear full-path bound provider."
        )
    return FunctionContactParticipant(
        geometry.surface_plan,
        geometry.source_space,
        _RodPositions(geometry),
        tangent_space=geometry.tangent_space,
        velocity_action=_RodVelocity(geometry),
        effort_pullback_action=_RodPullback(geometry),
        bounds_action=bounds_action,
        participant_id=f"{geometry.participant_id}:implicit-geometry",
    )


@final
class _MPMParticleMap(StrictModule, NonTrainableState):
    prepared: PreparedImplicitMPMResidual

    def __call__(self, displacement: Array, /) -> Array:
        dynamics = self.prepared.dynamics
        value = dynamics._axis_velocity(displacement, self.prepared.storage_state)
        gathered = gather_apic(
            self.prepared.execution_routes,
            value.reshape((dynamics.grid_count, dynamics.dimension)),
            dynamics.particles.active_mask,
            dynamics.method.transfer.maximum_condition,
        )
        return gathered.velocity


@final
class _MPMParticleTranspose(StrictModule, NonTrainableState):
    action: _MPMParticleMap
    template: Array

    def __call__(self, effort: Array, /) -> Array:
        return jax.linear_transpose(self.action, self.template)(effort)[0]


@final
class ImplicitMPMContactParticipant(AbstractImplicitContactParticipant):
    prepared: PreparedImplicitMPMResidual
    space: ArraySpace
    particle_displacement_operator: AbstractLinearOperator
    inversion: SimplexInversionStepPlan
    _participant_id: str = eqx.field(static=True)

    @property
    def dof_space(self) -> ArraySpace:
        return self.space

    @property
    def participant_id(self) -> str:
        return self._participant_id

    @property
    def has_potential(self) -> bool:
        return False

    def evaluate(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> ContactParticipantEvaluation:
        del accepted_state, time, args
        if not isinstance(committed_history, MPMRuntimeState):
            raise TypeError("Implicit MPM history must be MPMRuntimeState.")
        # Reprepare residual and participant at every accepted boundary. Frozen
        # routes/mass are never refreshed from a Newton trial or stale history.
        prepared = eqx.tree_at(
            lambda owner: owner.state, self.prepared, committed_history
        )
        value = self.space.validate(state)
        detailed = prepared.evaluate_detailed(
            value / prepared.step_size, include_inertia=step_size is not None
        )
        fresh = (
            (committed_history.time == self.prepared.state.time)
            & (committed_history.accepted_step == self.prepared.state.accepted_step)
            & (
                committed_history.topology_generation
                == self.prepared.state.topology_generation
            )
        )
        valid = detailed.successful & fresh
        if step_size is not None:
            valid = valid & (step_size == prepared.step_size)
        return ContactParticipantEvaluation(
            detailed.residual / prepared.step_size,
            detailed.candidate_state,
            None,
            valid,
            detailed.admissible & fresh,
            detailed.tangent_successful & fresh,
            jnp.asarray(False, dtype=jnp.bool_),
            jnp.asarray(jnp.inf, dtype=value.dtype),
            detailed.suggested_step,
            jnp.where(valid, jnp.int32(0), jnp.int32(1)),
            {
                "minimum_jacobian": detailed.minimum_jacobian,
                "prepared_history_fresh": fresh,
            },
        )

    def residual_jvp(
        self,
        state: PyTree[Array],
        direction: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> Array:
        del accepted_state, time, args
        if not isinstance(committed_history, MPMRuntimeState):
            raise TypeError("Implicit MPM history must be MPMRuntimeState.")
        prepared = eqx.tree_at(
            lambda owner: owner.state, self.prepared, committed_history
        )
        dt = prepared.step_size
        return (
            prepared.residual_jvp(
                self.space.validate(state) / dt,
                self.space.validate(direction) / dt,
                include_inertia=step_size is not None,
            )
            / dt
        )

    def potential(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> None:
        return None

    def potential_hvp(
        self,
        state: PyTree[Array],
        direction: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> None:
        return None

    def contact_velocity(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> Array:
        return self.space.validate(state) / self.prepared.step_size

    def contact_velocity_scale(self, step_size: Array, /) -> Array:
        return jnp.ones_like(step_size) / self.prepared.step_size

    def _frames(self, displacement: Array, /) -> Array:
        dynamics = self.prepared.dynamics
        velocity = dynamics._axis_velocity(
            displacement / self.prepared.step_size, self.prepared.storage_state
        )
        gathered = gather_apic(
            self.prepared.execution_routes,
            velocity.reshape((dynamics.grid_count, dynamics.dimension)),
            dynamics.particles.active_mask,
            dynamics.method.transfer.maximum_condition,
        )
        particle = self.prepared.state.particles
        deformation = dynamics._update_deformation(
            self.prepared.execution_routes,
            particle.deformation_gradient,
            gathered.velocity_gradient,
            self.prepared.step_size,
            self.prepared.mass,
            dynamics.particles.active_mask,
            reference_volume=particle.reference_volume,
            physical_deformation_gradient=particle.physical_deformation_gradient,
        )
        zero = jnp.zeros(
            (deformation.shape[0], 1, deformation.shape[-1]), dtype=deformation.dtype
        )
        return jnp.concatenate((zero, jnp.swapaxes(deformation, -1, -2)), axis=1).reshape(
            (-1, deformation.shape[-1])
        )

    def step_limit(
        self,
        state: PyTree[Array],
        direction: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> ContactParticipantStepLimit:
        del accepted_state, time, step_size, args
        value = self.space.validate(state)
        delta = self.space.validate(direction)
        evidence = simplex_inversion_step_limit(
            self.inversion, self._frames(value), self._frames(value + delta)
        )
        base = self.prepared.state.particles.position
        start = base + self.particle_displacement_operator.mv(value)
        move = self.particle_displacement_operator.mv(delta)
        dynamics = self.prepared.dynamics
        bound = jnp.ones((), dtype=value.dtype)
        domain = dynamics.particle_domain
        for axis, periodic in enumerate(domain.periodic):
            if not periodic:
                speed = move[:, axis]
                denominator = jnp.where(speed != 0.0, speed, jnp.ones_like(speed))
                ratio = jnp.where(
                    speed > 0.0,
                    (domain.bounds[1, axis] - start[:, axis]) / denominator,
                    (domain.bounds[0, axis] - start[:, axis]) / denominator,
                )
                ratio = jnp.where(
                    (speed == 0.0) | ~dynamics.particles.active_mask,
                    jnp.ones_like(ratio),
                    ratio,
                )
                bound = jnp.minimum(bound, jnp.min(ratio))
        initial_ok = jnp.all(~dynamics.particles.active_mask | domain.contains(start))
        # Rescale only an actual exit inside the unit path. A fully admitted
        # path (including zero motion or periodic axes) has no domain cap.
        domain_limit = jnp.where(
            bound >= 1.0,
            jnp.ones_like(bound),
            jnp.maximum(jnp.zeros_like(bound), 0.8 * bound),
        )
        maximum = jnp.minimum(evidence.step_size, domain_limit)
        valid = evidence.successful & initial_ok
        guarantee = ContactGuaranteeEvidence(
            ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE,
            finite=evidence.finite,
            work_complete=valid,
            failure_code=jnp.where(valid, jnp.int32(0), jnp.int32(1)),
            backend_id=self.participant_id,
        )
        return ContactParticipantStepLimit(
            jnp.where(valid, maximum, jnp.zeros_like(maximum)),
            valid,
            evidence.conservative,
            guarantee,
            jnp.where(valid, jnp.int32(0), jnp.int32(1)),
            evidence.limiting_cell,
            self.participant_id,
        )


@checked
def prepare_implicit_mpm_contact_participant(
    prepared: PreparedImplicitMPMResidual, /, *, participant_id: str | None = None
) -> ImplicitMPMContactParticipant:
    dynamics = prepared.dynamics
    if (
        dynamics.geometry.kind != "cartesian"
        or dynamics.method.volumetric_stabilization.kind == "f_bar"
    ):
        raise ValueError(
            "Implicit contact MPM requires Cartesian affine deformation paths; F-bar/axisymmetric paths need conservative nonlinear bounds."
        )
    dimension = dynamics.constitutive_dimension
    if dimension not in (2, 3):
        raise ValueError("Implicit MPM contact requires 2-D or 3-D constitutive frames.")
    active = np.asarray(dynamics.particles.active_mask)
    if not np.any(active):
        raise ValueError("Implicit MPM contact requires active particles.")
    count = active.shape[0]
    frames = np.broadcast_to(
        np.concatenate(
            (
                np.zeros((1, dimension), dtype=np.float64),
                np.eye(dimension, dtype=np.float64),
            ),
            axis=0,
        ),
        (count, dimension + 1, dimension),
    ).reshape((-1, dimension))
    cells = np.arange(count * (dimension + 1), dtype=np.int32).reshape(
        (count, dimension + 1)
    )[active]
    inversion = SimplexInversionStepPlan(
        cells, frames, minimum_ratio=0.0, maximum_iterations=128
    )
    source = ArraySpace(
        prepared.initial_velocity.shape, dtype=prepared.initial_velocity.dtype
    )
    target = ArraySpace(
        prepared.state.particles.position.shape, dtype=prepared.initial_velocity.dtype
    )
    action = _MPMParticleMap(prepared)
    identifier = (
        canonical_fingerprint(
            {
                "kind": "implicit-mpm-contact",
                "dynamics": dynamics.prepared_id,
                "shape": source.shape,
            }
        )
        if participant_id is None
        else participant_id
    )
    if not identifier:
        raise ValueError("participant_id must be nonempty.")
    from ...linalg import FunctionLinearOperator

    operator = FunctionLinearOperator(
        action,
        source=source,
        target=target,
        transpose_action=_MPMParticleTranspose(
            action, jnp.zeros_like(prepared.initial_velocity)
        ),
        operator_id=f"{identifier}:frozen-particle-map",
    )
    return ImplicitMPMContactParticipant(
        prepared, source, operator, inversion, identifier
    )


__all__ = [
    "ArticulatedPrimitiveContactParticipant",
    "RigidContactParticipant",
    "make_articulated_contact_participant",
    "prepare_articulated_contact_scene",
    "prepare_mpm_contact_participant",
    "prepare_point_contact_participant",
    "FiniteElementContactParticipant",
    "ImplicitMPMContactParticipant",
    "prepare_finite_element_contact_participant",
    "prepare_iga_contact_participant",
    "prepare_implicit_mpm_contact_participant",
    "prepare_implicit_reduced_rod_contact_geometry",
]
