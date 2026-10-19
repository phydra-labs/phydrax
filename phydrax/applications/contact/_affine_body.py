# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Four-control affine dynamics with reference-volume quadrature.

The orthogonality potential is VE/8 ||F.T F-I||², giving VE/2 F(F.T F-I).
Controls are displacement coordinates; geometry adds the reference controls.
Reflection has zero orthogonality energy but is never an admissible state.
"""

from __future__ import annotations

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._guarantee import (
    ContactGuaranteeEvidence,
    ContactGuaranteeLevel,
)
from ...discretization.contact._inversion import (
    simplex_inversion_step_limit,
    SimplexInversionStepPlan,
)
from ...discretization.contact._participant import LinearContactParticipant
from ...discretization.contact._precision import ContactPrecisionPolicy
from ...discretization.contact._surface import (
    CollisionSurfacePlan,
    PreparedCollisionSurface,
)
from ...ein import contract
from ...linalg import (
    ArraySpace,
    determinant_small_linear,
    FunctionLinearOperator,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ...typing import checked
from ..solid_mechanics._fem_dynamics import ImplicitNewmarkMethod
from ._implicit_participant import (
    AbstractImplicitContactParticipant,
    ContactParticipantEvaluation,
    ContactParticipantStepLimit,
)


@final
class AffineBodyPlan(StrictModule, NonTrainableState):
    """Reference controls and quadrature with explicit vertex ownership.

    ``vertex_body_ids`` indexes local control rows, while topology body IDs
    identify bodies globally in a collision scene. Their vertex bindings must
    establish a one-to-one relation; the numerical IDs need not be equal.

    The orientation enclosure's absolute error scales with the smallest reference
    frame determinant, preserving admissibility under uniform length rescaling.
    """

    topology: CollisionSurfacePlan
    reference_controls: Array
    material_coordinates: Array
    vertex_body_ids: Array
    quadrature_shape_values: Array
    quadrature_weights: Array
    density: Array
    rigidity_modulus: Array
    gravity: Array
    control_mass: Array
    solve_plan: SmallLinearSolvePlan
    inversion: SimplexInversionStepPlan
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        topology: CollisionSurfacePlan,
        reference_controls: ArrayLike,
        material_coordinates: ArrayLike,
        vertex_body_ids: ArrayLike,
        quadrature_shape_values: ArrayLike,
        quadrature_weights: ArrayLike,
        density: ArrayLike,
        rigidity_modulus: ArrayLike,
        /,
        *,
        gravity: ArrayLike | tuple[float, float, float] = (0.0, 0.0, -9.81),
        plan_id: str | None = None,
    ) -> None:
        controls = np.asarray(reference_controls, dtype=np.float64)
        coordinates = np.asarray(material_coordinates, dtype=np.float64)
        labels = np.asarray(vertex_body_ids)
        shapes = np.asarray(quadrature_shape_values, dtype=np.float64)
        weights = np.asarray(quadrature_weights, dtype=np.float64)
        rho = np.asarray(density, dtype=np.float64)
        modulus = np.asarray(rigidity_modulus, dtype=np.float64)
        gravity_ = np.asarray(gravity, dtype=np.float64)
        if (
            topology.ambient_dimension != 3
            or controls.ndim != 3
            or controls.shape[1:] != (4, 3)
            or controls.shape[0] == 0
        ):
            raise ValueError("Affine bodies require four 3-D controls per body.")
        bodies = controls.shape[0]
        if (
            coordinates.shape != (topology.vertex_count, 3)
            or labels.shape != (topology.vertex_count,)
            or not np.issubdtype(labels.dtype, np.integer)
        ):
            raise ValueError(
                "Affine material coordinates and integer labels must match topology."
            )
        if (
            np.any(labels < 0)
            or np.any(labels >= bodies)
            or set(labels.tolist()) != set(range(bodies))
        ):
            raise ValueError(
                "Every affine body requires surface samples with valid body labels."
            )
        global_body_ids = np.asarray(
            topology.feature_policy.body_ids[topology.feature_policy.vertex_slice]
        )
        # Control rows are local indices; collision body keys are scene identities.
        # Require one-to-one ownership, never equality of those distinct roles.
        body_keys = np.asarray(
            [global_body_ids[labels == body][0] for body in range(bodies)]
        )
        if np.unique(body_keys).size != bodies or not np.array_equal(
            global_body_ids, body_keys[labels]
        ):
            raise ValueError(
                "Affine control bodies require unique homogeneous collision body IDs."
            )
        for primitive in (np.asarray(topology.edges), np.asarray(topology.faces)):
            if np.any(labels[primitive] != labels[primitive[:, :1]]):
                raise ValueError("Affine collision primitives cannot span bodies.")
        if (
            shapes.ndim != 3
            or shapes.shape[0] != bodies
            or shapes.shape[2] != 4
            or shapes.shape[1] == 0
            or weights.shape != shapes.shape[:2]
        ):
            raise ValueError("Quadrature must have fixed (body, point, control) shape.")
        if rho.shape != (bodies,) or modulus.shape != (bodies,) or gravity_.shape != (3,):
            raise ValueError(
                "Density/modulus require one value per body; gravity requires three."
            )
        if any(
            np.any(~np.isfinite(value))
            for value in (controls, coordinates, shapes, weights, rho, modulus, gravity_)
        ):
            raise ValueError("Affine body data must be finite.")
        if np.any(weights <= 0.0) or np.any(rho <= 0.0) or np.any(modulus <= 0.0):
            raise ValueError(
                "Quadrature weights, density, and rigidity must be positive."
            )
        if np.any(np.abs(shapes.sum(axis=-1) - 1.0) > 64.0 * np.finfo(np.float64).eps):
            raise ValueError("Affine quadrature must reproduce constants.")
        frames = np.swapaxes(controls[:, 1:] - controls[:, :1], -1, -2)
        frame_determinants = np.linalg.det(frames)
        if np.any(frame_determinants <= 0.0):
            raise ValueError(
                "Reference controls must have strictly positive orientation."
            )
        mass = np.stack(
            [
                shape.T @ ((rho[b] * weights[b])[:, None] * shape)
                for b, shape in enumerate(shapes)
            ]
        )
        if np.any(np.linalg.eigvalsh(mass) <= 0.0):
            raise ValueError(
                "Control mass must be positive definite; free zero-mass controls refuse."
            )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "affine-body",
                    "topology": topology.topology_id,
                    "data": array_tree_fingerprint(
                        (
                            controls,
                            coordinates,
                            labels,
                            shapes,
                            weights,
                            rho,
                            modulus,
                            gravity_,
                        )
                    ),
                }
            )
            if plan_id is None
            else plan_id
        )
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.topology = topology
        self.reference_controls = jnp.asarray(controls, dtype=jnp.float64)
        self.material_coordinates = jnp.asarray(coordinates, dtype=jnp.float64)
        self.vertex_body_ids = jnp.asarray(labels, dtype=jnp.int32)
        self.quadrature_shape_values = jnp.asarray(shapes, dtype=jnp.float64)
        self.quadrature_weights = jnp.asarray(weights, dtype=jnp.float64)
        self.density = jnp.asarray(rho, dtype=jnp.float64)
        self.rigidity_modulus = jnp.asarray(modulus, dtype=jnp.float64)
        self.gravity = jnp.asarray(gravity_, dtype=jnp.float64)
        self.control_mass = jnp.asarray(mass, dtype=jnp.float64)
        self.solve_plan = SmallLinearSolvePlan(3)
        self.inversion = SimplexInversionStepPlan(
            np.arange(4 * bodies, dtype=np.int32).reshape((bodies, 4)),
            controls.reshape((-1, 3)),
            minimum_ratio=0.0,
            # The inversion owner subtracts an absolute determinant error.
            # Convert the relative roundoff allowance to reference volume units.
            numerical_error=1.0e-13 * float(np.min(np.abs(frame_determinants))),
            maximum_iterations=128,
        )
        self.plan_id = identifier


@final
class AffineBodyHistory(StrictModule, NonTrainableState):
    velocity: Array
    acceleration: Array

    def __init__(self, velocity: ArrayLike, acceleration: ArrayLike, /) -> None:
        value = jnp.asarray(velocity, dtype=jnp.float64)
        rate = jnp.asarray(acceleration, dtype=jnp.float64)
        if value.ndim != 3 or value.shape[1:] != (4, 3) or rate.shape != value.shape:
            raise ValueError(
                "Affine history must have (body,4,3) velocity and acceleration."
            )
        self.velocity = value
        self.acceleration = rate


@final
class _AffineMap(StrictModule, NonTrainableState):
    weights: Array
    owners: Array
    shape: tuple[int, ...] = eqx.field(static=True)

    def __call__(self, value: Array, /) -> Array:
        return contract("va,vai->vi", self.weights, value[self.owners])


@final
class _AffineTranspose(StrictModule, NonTrainableState):
    weights: Array
    owners: Array
    shape: tuple[int, ...] = eqx.field(static=True)

    def __call__(self, value: Array, /) -> Array:
        contributions = self.weights[:, :, None] * value[:, None, :]
        return jnp.zeros(self.shape, dtype=value.dtype).at[self.owners].add(contributions)


@final
class AffineBodyParticipant(AbstractImplicitContactParticipant):
    plan: AffineBodyPlan
    newmark: ImplicitNewmarkMethod
    geometry: LinearContactParticipant
    _participant_id: str = eqx.field(static=True)

    @property
    def participant_id(self) -> str:
        return self._participant_id

    @property
    def dof_space(self) -> ArraySpace:
        space = self.geometry.source_space
        if not isinstance(space, ArraySpace):
            raise TypeError("Affine controls require ArraySpace.")
        return space

    @property
    def has_potential(self) -> bool:
        return True

    def _displacement_gradient(self, state: Array, /) -> tuple[Array, Array]:
        frame = jnp.swapaxes(state[:, 1:] - state[:, :1], -1, -2)
        reference = jnp.swapaxes(
            self.plan.reference_controls[:, 1:] - self.plan.reference_controls[:, :1],
            -1,
            -2,
        )
        result = solve_small_linear(
            self.plan.solve_plan,
            jnp.swapaxes(reference, -1, -2),
            jnp.swapaxes(frame, -1, -2),
        )
        return jnp.swapaxes(result.value, -1, -2), jnp.all(result.successful)

    def potential(
        self,
        state: PyTree[Array],
        committed_history: PyTree[Array],
        accepted_state: PyTree[Array],
        time: Array,
        step_size: Array | None,
        args: Any,
        /,
    ) -> Array:
        del time, args
        value = self.dof_space.validate(state)
        gradient, _ = self._displacement_gradient(value)
        transpose = jnp.swapaxes(gradient, -1, -2)
        # F = I + G: expand F.T F - I without subtracting nearly equal tensors.
        strain = gradient + transpose + transpose @ gradient
        volume = jnp.sum(self.plan.quadrature_weights, axis=-1)
        energy = jnp.sum(
            volume
            * self.plan.rigidity_modulus
            * jnp.sum(strain * strain, axis=(-2, -1))
            / 8.0
        )
        gravity_force = (
            jnp.sum(self.plan.control_mass, axis=-1)[:, :, None]
            * self.plan.gravity[None, None, :]
        )
        energy = energy - jnp.sum(gravity_force * value)
        if step_size is not None:
            history = self._history(committed_history)
            predictor = (
                self.dof_space.validate(accepted_state)
                + step_size * history.velocity
                + step_size**2 * (0.5 - self.newmark.beta) * history.acceleration
            )
            delta = value - predictor
            energy = energy + jnp.sum(delta * (self.plan.control_mass @ delta)) / (
                2.0 * self.newmark.beta * step_size**2
            )
        return energy

    def _history(self, history: PyTree[Array], /) -> AffineBodyHistory:
        if not isinstance(history, AffineBodyHistory):
            raise TypeError("Affine mechanics requires AffineBodyHistory.")
        self.dof_space.validate(history.velocity)
        self.dof_space.validate(history.acceleration)
        return history

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
        history = self._history(committed_history)
        energy, residual = jax.value_and_grad(self.potential)(
            value, history, accepted_state, time, step_size, args
        )
        gradient, solve_ok = self._displacement_gradient(value)
        deformation = gradient + jnp.eye(3, dtype=value.dtype)[None, :, :]
        determinant = determinant_small_linear(self.plan.solve_plan, deformation)
        finite = jnp.all(jnp.isfinite(residual)) & jnp.isfinite(energy)
        admissible = solve_ok & jnp.all(determinant > 0.0) & finite
        trial = history
        if step_size is not None:
            predictor = (
                self.dof_space.validate(accepted_state)
                + step_size * history.velocity
                + step_size**2 * (0.5 - self.newmark.beta) * history.acceleration
            )
            acceleration = (value - predictor) / (self.newmark.beta * step_size**2)
            velocity = history.velocity + step_size * (
                (1.0 - self.newmark.gamma) * history.acceleration
                + self.newmark.gamma * acceleration
            )
            trial = AffineBodyHistory(velocity, acceleration)
            admissible = admissible & jnp.isfinite(step_size) & (step_size > 0.0)
        return ContactParticipantEvaluation(
            residual,
            trial,
            energy,
            admissible,
            admissible,
            admissible,
            jnp.asarray(True, dtype=jnp.bool_),
            jnp.zeros((), dtype=value.dtype),
            jnp.asarray(jnp.inf, dtype=value.dtype),
            jnp.where(admissible, jnp.int32(0), jnp.int32(1)),
            {"minimum_jacobian": jnp.min(determinant)},
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
        value = self.dof_space.validate(state)
        tangent = self.dof_space.validate(direction)
        return jax.jvp(
            lambda q: jax.grad(self.potential)(
                q, committed_history, accepted_state, time, step_size, args
            ),
            (value,),
            (tangent,),
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
    ) -> Array:
        return self.residual_jvp(
            state, direction, committed_history, accepted_state, time, step_size, args
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
        start = self.plan.reference_controls + self.dof_space.validate(state)
        end = start + self.dof_space.validate(direction)
        evidence = simplex_inversion_step_limit(
            self.plan.inversion, start.reshape((-1, 3)), end.reshape((-1, 3))
        )
        guarantee = ContactGuaranteeEvidence(
            ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE,
            finite=evidence.finite,
            work_complete=evidence.successful,
            failure_code=evidence.status,
            margin=evidence.minimum_initial_ratio,
            backend_id=self.plan.inversion.plan_id,
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
        history = self._history(committed_history)
        predictor = (
            self.dof_space.validate(accepted_state)
            + step_size * history.velocity
            + step_size**2 * (0.5 - self.newmark.beta) * history.acceleration
        )
        acceleration = (self.dof_space.validate(state) - predictor) / (
            self.newmark.beta * step_size**2
        )
        return history.velocity + step_size * (
            (1.0 - self.newmark.gamma) * history.acceleration
            + self.newmark.gamma * acceleration
        )

    def contact_velocity_scale(self, step_size: Array, /) -> Array:
        return self.newmark.position_to_velocity_scale(step_size, step_size.dtype)


@checked
def prepare_affine_body_participant(
    plan: AffineBodyPlan,
    /,
    *,
    method: ImplicitNewmarkMethod | None = None,
    precision: ContactPrecisionPolicy | None = None,
) -> AffineBodyParticipant:
    coordinates = plan.material_coordinates
    weights = jnp.concatenate(
        (1.0 - jnp.sum(coordinates, axis=-1, keepdims=True), coordinates), axis=-1
    )
    shape = plan.reference_controls.shape
    source = ArraySpace(shape, dtype=plan.reference_controls.dtype)
    action = _AffineMap(weights, plan.vertex_body_ids, shape)
    transpose = _AffineTranspose(weights, plan.vertex_body_ids, shape)
    reference = action(plan.reference_controls)
    operator = FunctionLinearOperator(
        action,
        source=source,
        target=ArraySpace(reference.shape, dtype=reference.dtype),
        transpose_action=transpose,
        operator_id=f"{plan.plan_id}:surface-map",
    )
    surface = PreparedCollisionSurface(
        plan.topology, reference, operator, precision=precision
    )
    return AffineBodyParticipant(
        plan,
        ImplicitNewmarkMethod() if method is None else method,
        LinearContactParticipant(surface),
        f"{plan.plan_id}:mechanics",
    )
