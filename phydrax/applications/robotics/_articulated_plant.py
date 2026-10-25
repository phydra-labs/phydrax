#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Complete-state native articulated robot plant.

`PreparedArticulatedPlant` implements the shared `AbstractDiscretePlant`
transaction over one `NativeRobotModel` and the native coupled contact/limit
step. The payload is the complete mechanical history: principal configuration
and integer winding (the lifted coordinate is their sum), generalized
velocity, mechanical contact manifold, stop reactions and fixed-branch modes,
the last applied command, transition/sensor epochs, sensor samples and the
geometry/material revisions the history is bound to. The runtime envelope owns
clock, step index and PRNG key; PlantCore commits the whole payload, clock and
key per case only when the step covered the full requested interval.

Cases are explicit leading axes (``plan.case_ndim``). Every lifecycle
operation evaluates the same pure single-case transition, mapped over the
flattened case axes; nothing is reconstructed from ``q``/``v`` alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from ..._array_tree import ArrayPyTreeSchema
from ..._fingerprint import canonical_fingerprint
from ..._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._tree_math import tree_where
from ...discretization.particle._reduced_articulation import (
    PreparedReducedArticulation,
)
from ...discretization.particle._rigid_joint_coordinates import (
    LiftedJointCoordinateState,
)
from ...dynamics import (
    AbstractDiscretePlant,
    PlantParameters,
    PlantProposal,
    PlantResetResult,
    PlantRuntimeState,
    PlantStepContext,
    PlantStepResult,
)
from ...typing import ConvertibleToArray, parse
from ..contact._articulated_step import (
    ArticulatedActuationMode,
    ArticulatedContactAdmissibility,
    ArticulatedContactSearchBackend,
    ArticulatedContactSource,
    ArticulatedContactStepPolicy,
    ArticulatedContactStepResult,
    PreparedArticulatedContactStep,
)
from ..contact._manifold import ContactManifoldState
from ._native_model import (
    NativeRobotModel,
    PreparedNativeRobotModel,
    RobotFramePlan,
)


REVISION_MISMATCH_STATUS = 64


class ArticulatedPlantStatus(IntEnum):
    """Plant-only statuses; step failures report `ArticulatedContactStepStatus`."""

    SUCCESS = 0
    REVISION_MISMATCH = REVISION_MISMATCH_STATUS


class NativeArticulatedPlantPlan(StrictModule, NonTrainableState):
    """Preparation of one native articulated plant.

    ``actuation_mode`` declares the command meaning explicitly; nothing is
    inferred from source bounds. ``initial_lifted_configuration`` is split into
    a principal chart point and integer winding by the model's lifted
    coordinate owner (nonzero winding is retained). ``activation_distance`` (m)
    and explicit ``route_capacities`` (witness rows per route kind: vertex-
    vertex, edge-vertex, edge-edge, face-vertex, vertex-plane, edge-plane) are
    required with collision geometry that can form any route; geometry whose
    legal-route bounds are all zero runs contact-free with zero capacity. The
    LBVH backend additionally requires ``maximum_traversal_visits``. Capacity
    or traversal exhaustion is an observable search failure. ``geometry_revision`` and
    ``material_revision`` bind contact history to the prepared numerics;
    histories carrying other revisions are refused.
    """

    step_policy: ArticulatedContactStepPolicy
    gravity: Array
    initial_lifted_configuration: Array | None
    initial_velocity: Array | None
    actuation_mode: ArticulatedActuationMode = eqx.field(static=True)
    case_ndim: int = eqx.field(static=True)
    search_backend: ArticulatedContactSearchBackend = eqx.field(static=True)
    activation_distance: float | None = eqx.field(static=True)
    envelope_radius: float = eqx.field(static=True)
    route_capacities: tuple[int, int, int, int, int, int] | None = eqx.field(static=True)
    history_capacity: int | None = eqx.field(static=True)
    maximum_traversal_visits: int | None = eqx.field(static=True)
    maximum_turns: int = eqx.field(static=True)
    geometry_revision: int = eqx.field(static=True)
    material_revision: int = eqx.field(static=True)

    def __init__(
        self,
        step_policy: ArticulatedContactStepPolicy,
        /,
        *,
        gravity: ConvertibleToArray,
        actuation_mode: ArticulatedActuationMode,
        initial_lifted_configuration: ConvertibleToArray | None = None,
        initial_velocity: ConvertibleToArray | None = None,
        case_ndim: int = 0,
        search_backend: ArticulatedContactSearchBackend = "exhaustive",
        activation_distance: float | None = None,
        envelope_radius: float = 0.0,
        route_capacities: tuple[int, int, int, int, int, int] | None = None,
        history_capacity: int | None = None,
        maximum_traversal_visits: int | None = None,
        maximum_turns: int = 1024,
        geometry_revision: int = 0,
        material_revision: int = 0,
    ) -> None:
        if not isinstance(step_policy, ArticulatedContactStepPolicy):
            raise TypeError("step_policy must be ArticulatedContactStepPolicy.")
        gravity_ = np.asarray(gravity, dtype=np.float64)
        if gravity_.shape != (3,) or not np.all(np.isfinite(gravity_)):
            raise ValueError("gravity must be one finite three-vector acceleration.")
        if isinstance(case_ndim, bool) or int(case_ndim) < 0:
            raise ValueError("case_ndim must be a nonnegative integer.")
        backend = parse(search_backend, ArticulatedContactSearchBackend, "search_backend")
        capacities: tuple[int, int, int, int, int, int] | None = None
        if route_capacities is not None:
            values = tuple(int(value) for value in route_capacities)
            if len(values) != 6:
                raise ValueError("route_capacities must contain six route-kind counts.")
            capacities = (
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
            )
        revisions = (int(geometry_revision), int(material_revision))
        if any(value < 0 for value in revisions):
            raise ValueError("Revisions must be nonnegative.")
        self.step_policy = step_policy
        self.gravity = jnp.asarray(gravity_)
        self.initial_lifted_configuration = (
            None
            if initial_lifted_configuration is None
            else jnp.asarray(initial_lifted_configuration)
        )
        self.initial_velocity = (
            None if initial_velocity is None else jnp.asarray(initial_velocity)
        )
        self.actuation_mode = ArticulatedActuationMode(actuation_mode)
        self.case_ndim = int(case_ndim)
        self.search_backend = backend
        self.activation_distance = (
            None if activation_distance is None else float(activation_distance)
        )
        self.envelope_radius = float(envelope_radius)
        self.route_capacities = capacities
        self.history_capacity = (
            None if history_capacity is None else int(history_capacity)
        )
        self.maximum_traversal_visits = (
            None if maximum_traversal_visits is None else int(maximum_traversal_visits)
        )
        self.maximum_turns = int(maximum_turns)
        self.geometry_revision, self.material_revision = revisions


class ArticulatedPlantSensors(StrictModule):
    """Sampled sensors of an accepted state.

    ``frame_transforms`` are the model's named source frames in world
    coordinates (name-ID order); ``contact_normal_impulse`` the recorded normal
    impulse (N s) per manifold slot, zero for inactive slots; ``stop_impulse``
    the last accepted stop reactions (N m s or N s).
    """

    frame_transforms: Array
    contact_normal_impulse: Array
    stop_impulse: Array


class ArticulatedPlantState(StrictModule):
    """Complete native plant payload of one case."""

    configuration: Array
    winding: Array
    velocity: Array
    contact_history: ContactManifoldState | None
    stop_impulse: Array
    stop_branch: Array
    command: Array
    epoch: Array
    sensor_epoch: Array
    sensors: ArticulatedPlantSensors
    geometry_revision: Array
    material_revision: Array

    @property
    def mechanics(self) -> ArticulatedContactSource:
        return ArticulatedContactSource(
            self.configuration,
            self.winding,
            self.velocity,
            self.contact_history,
            self.stop_impulse,
            self.stop_branch,
        )


class ArticulatedPlantCommand(StrictModule):
    """One command vector in the plant's declared actuation units.

    ``MAPPED_MOTORS``: actuator commands in model actuator-ID order.
    ``GENERALIZED_EFFORT``: generalized effort (N m hinge, N prismatic) in
    articulation velocity order.
    """

    values: Array

    def __init__(self, values: ConvertibleToArray, /) -> None:
        self.values = jnp.asarray(values)


class ArticulatedPlantParameters(StrictModule):
    """Dynamic plant parameters: the world gravity acceleration (m/s²)."""

    gravity: Array

    def __init__(self, gravity: ConvertibleToArray, /) -> None:
        self.gravity = jnp.asarray(gravity)


class ArticulatedPlantResetEvidence(StrictModule):
    admissibility: ArticulatedContactAdmissibility
    plant_id: str = eqx.field(static=True)


class ArticulatedPlantStepEvidence(StrictModule):
    """Every staged step result, retained even when the payload rolls back."""

    step: ArticulatedContactStepResult
    revision_valid: Array
    plant_id: str = eqx.field(static=True)


class ArticulatedPlantObservation(StrictModule):
    """Projection of one accepted runtime state; reading never advances it.

    Sensor samples are fresh only when ``sensor_epoch == epoch``.
    """

    time: Array
    step_index: Array
    configuration: Array
    lifted_configuration: Array
    velocity: Array
    command: Array
    frame_transforms: Array
    contact_normal_impulse: Array
    stop_impulse: Array
    epoch: Array
    sensor_epoch: Array
    fresh: Array
    finite: Array


class ArticulatedPlantRefreshResult(StrictModule):
    """Transactional sensor refresh of accepted states."""

    candidate_state: PlantRuntimeState
    accepted_state: PlantRuntimeState
    attempted: Array
    successful: Array
    status: Array
    observation: ArticulatedPlantObservation
    admissibility: ArticulatedContactAdmissibility


class ArticulatedPlantControlView(StrictModule):
    """Continuous domain view: lifted coordinates and generalized velocity."""

    lifted_configuration: Array
    velocity: Array


def _casewise(
    function: Callable[..., Any],
    case_shape: tuple[int, ...],
    batched: tuple[PyTree[Any], ...],
    shared: tuple[PyTree[Any], ...],
    /,
) -> Any:
    """Apply one pure single-case function over explicit leading case axes."""

    if case_shape == ():
        return function(*batched, *shared)
    ndim = len(case_shape)
    count = int(np.prod(case_shape, dtype=np.int64))
    flat = tuple(
        jax.tree.map(
            lambda leaf: jnp.reshape(leaf, (count,) + jnp.shape(leaf)[ndim:]), tree
        )
        for tree in batched
    )
    mapped = jax.vmap(
        function,
        in_axes=tuple(0 for _ in flat) + tuple(None for _ in shared),
    )(*flat, *shared)
    return jax.tree.map(
        lambda leaf: jnp.reshape(leaf, case_shape + jnp.shape(leaf)[1:]), mapped
    )


def _resolve_plant_model(
    model: NativeRobotModel | PreparedNativeRobotModel,
    plan: NativeArticulatedPlantPlan,
    /,
) -> PreparedNativeRobotModel:
    """Prepare (or accept) the model and check winding range and actuation mode."""

    prepared = (
        model.prepare(maximum_turns=plan.maximum_turns)
        if isinstance(model, NativeRobotModel)
        else model
    )
    if not isinstance(prepared, PreparedNativeRobotModel):
        raise TypeError("model must be NativeRobotModel or PreparedNativeRobotModel.")
    if prepared.coordinates.maximum_turns != plan.maximum_turns:
        raise ValueError("The prepared model's winding range differs from the plan.")
    mode = plan.actuation_mode
    if mode == ArticulatedActuationMode.MAPPED_MOTORS and prepared.motors is None:
        raise ValueError("MAPPED_MOTORS actuation requires declared model motors.")
    if mode != ArticulatedActuationMode.MAPPED_MOTORS and prepared.motors is not None:
        raise ValueError(
            "The model declares joint motors; select MAPPED_MOTORS rather than "
            "silently discarding them."
        )
    return prepared


def _prepare_plant_step(
    prepared: PreparedNativeRobotModel, plan: NativeArticulatedPlantPlan, /
) -> PreparedArticulatedContactStep:
    """Bind the model's owners and the plan's explicit search budgets to the step."""

    collision = prepared.model.collision
    if collision is None and (
        plan.route_capacities is not None
        or plan.activation_distance is not None
        or plan.maximum_traversal_visits is not None
    ):
        raise ValueError(
            "Route capacities, activation distance and traversal budgets "
            "require collision geometry."
        )
    materials = None if collision is None else collision.materials
    return PreparedArticulatedContactStep(
        prepared.articulation,
        prepared.coordinates,
        plan.step_policy,
        actuation_mode=plan.actuation_mode,
        geometry=prepared.collision,
        material_pairs=(
            None if materials is None else np.asarray(materials.material_pairs)
        ),
        static_friction=(
            None if materials is None else np.asarray(materials.static_friction)
        ),
        dynamic_friction=(
            None if materials is None else np.asarray(materials.dynamic_friction)
        ),
        stops=prepared.stops,
        motors=prepared.motors,
        damping=prepared.damping,
        search_backend=plan.search_backend,
        route_capacities=plan.route_capacities,
        activation_distance=plan.activation_distance,
        envelope_radius=plan.envelope_radius,
        maximum_traversal_visits=plan.maximum_traversal_visits,
        history_capacity=plan.history_capacity,
    )


def _initial_mechanics(
    prepared: PreparedNativeRobotModel,
    step: PreparedArticulatedContactStep,
    plan: NativeArticulatedPlantPlan,
    dtype: np.dtype,
    /,
) -> ArticulatedContactSource:
    """Initial mechanics from an explicit lifted coordinate (winding retained)."""

    articulation = prepared.articulation
    nq = articulation.nq
    lifted = (
        np.zeros((nq,), dtype=dtype)
        if plan.initial_lifted_configuration is None
        else np.asarray(plan.initial_lifted_configuration, dtype=dtype)
    )
    velocity = (
        np.zeros((articulation.nv,), dtype=dtype)
        if plan.initial_velocity is None
        else np.asarray(plan.initial_velocity, dtype=dtype)
    )
    if lifted.shape != (nq,) or velocity.shape != (articulation.nv,):
        raise ValueError("Initial lifted configuration/velocity shapes are invalid.")
    principal, winding, valid = prepared.coordinates.split(jnp.asarray(lifted))
    if not bool(valid):
        raise ValueError("The initial lifted configuration exceeds the winding range.")
    return step.initial_source(principal, winding.winding, velocity)


def _frame_body_indices(
    articulation: PreparedReducedArticulation, frames: RobotFramePlan, /
) -> np.ndarray:
    """Articulation body-capacity index of every named source frame."""

    body_ids = np.asarray(articulation.body_ids, dtype=np.int64)
    body_indices = np.asarray(articulation.body_indices, dtype=np.int64)
    index_by_body = {
        int(body): int(index) for body, index in zip(body_ids, body_indices, strict=True)
    }
    return np.asarray(
        [index_by_body[int(body)] for body in np.asarray(frames.body_ids)],
        dtype=np.int32,
    )


class PreparedArticulatedPlant(AbstractDiscretePlant, NonTrainableState):
    """Atomic complete-state native robot plant over the coupled contact step."""

    model: PreparedNativeRobotModel
    contact_step: PreparedArticulatedContactStep
    plan: NativeArticulatedPlantPlan
    frame_body_indices: Array
    frame_local_transforms: Array
    initial_state: ArticulatedPlantState
    default_parameters: ArticulatedPlantParameters
    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema | None
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: ArticulatedPlantState
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)
    plant_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: NativeRobotModel | PreparedNativeRobotModel,
        plan: NativeArticulatedPlantPlan,
        /,
    ) -> None:
        if not isinstance(plan, NativeArticulatedPlantPlan):
            raise TypeError("plan must be NativeArticulatedPlantPlan.")
        prepared = _resolve_plant_model(model, plan)
        source = prepared.model
        articulation = prepared.articulation
        dtype = np.dtype(articulation.reference_position.dtype)
        mode = plan.actuation_mode
        nq = articulation.nq
        step = _prepare_plant_step(prepared, plan)
        mechanics = _initial_mechanics(prepared, step, plan, dtype)
        frame_bodies = _frame_body_indices(articulation, source.frames)
        # The remaining body assembles the complete payload, schemas and the
        # semantic/numeric/executable identity records; it is one record
        # construction whose fields must stay visibly consistent with each other.
        self.model = prepared
        self.contact_step = step
        self.plan = plan
        self.frame_body_indices = jnp.asarray(frame_bodies)
        self.frame_local_transforms = source.frames.local_transforms().astype(dtype)
        command = jnp.zeros((step.actuation_size,), dtype=dtype)
        epoch = jnp.asarray(0, dtype=jnp.int32)
        initial = ArticulatedPlantState(
            mechanics.configuration,
            mechanics.winding,
            mechanics.velocity,
            mechanics.contact_history,
            mechanics.stop_impulse,
            mechanics.stop_branch,
            command,
            epoch,
            epoch,
            self._sensors(mechanics),
            jnp.asarray(plan.geometry_revision, dtype=jnp.int64),
            jnp.asarray(plan.material_revision, dtype=jnp.int64),
        )
        parameters = ArticulatedPlantParameters(jnp.asarray(plan.gravity, dtype=dtype))
        case_ndim = plan.case_ndim

        def expanded(tree: PyTree[Any]) -> PyTree[Any]:
            return jax.tree.map(
                lambda leaf: jnp.reshape(leaf, (1,) * case_ndim + jnp.shape(leaf)), tree
            )

        state_schema = ArrayPyTreeSchema.from_tree(expanded(initial), case_ndim=case_ndim)
        control_schema = (
            None
            if mode == ArticulatedActuationMode.AUTONOMOUS
            else ArrayPyTreeSchema.from_tree(
                expanded(ArticulatedPlantCommand(command)), case_ndim=case_ndim
            )
        )
        parameter_schema = ArrayPyTreeSchema.from_tree(parameters, case_ndim=0)
        semantic = SemanticProvenance(
            {
                "kind": "prepared-native-articulated-plant",
                "profile": "fixed-base-3d-scalar-joint-sphere-capsule-plane",
                "law": "position-level-inelastic-coulomb-hard-contact-coupled-stops",
                "actuation": int(mode),
                "time_policy": "full-requested-interval-or-reject",
                "state_schema": state_schema.content_id,
                "parameter_schema": parameter_schema.content_id,
                "case_ndim": case_ndim,
            }
        )
        numeric = NumericRevision(
            semantic,
            {
                "model": prepared.prepared_id,
                "step": step.step_id,
                "initial_state": initial,
                "gravity": parameters.gravity,
                "geometry_revision": plan.geometry_revision,
                "material_revision": plan.material_revision,
                "certificate": plan.step_policy.certificate,
            },
        )
        signature = ExecutableSignature(
            shapes={
                "configuration": (nq,),
                "velocity": (articulation.nv,),
                "command": (step.actuation_size,),
                "contact_blocks": (step.contact_capacity,),
                "stop_rows": (step.stop_capacity,),
                "history": (step.history_capacity,),
                "frames": tuple(self.frame_local_transforms.shape),
            },
            dtypes={"mechanics": dtype, "winding": np.dtype(np.int32)},
            space_ids={
                "generalized_velocity": articulation.generalized_velocity_space.space_id,
            },
            topology_ids={
                "model": prepared.prepared_id,
                "articulation": articulation.prepared_id,
                "contact_catalogue": step.catalogue_id,
            },
            capacities={
                "contact_routes": step.contact_capacity,
                "stop_rows": step.stop_capacity,
                "history": step.history_capacity,
                "root_iterations": plan.step_policy.maximum_root_iterations,
                "route_rebuilds": plan.step_policy.maximum_route_rebuilds,
                "subdivision_depth": plan.step_policy.maximum_subdivision_depth,
                "segment_attempts": plan.step_policy.maximum_segment_attempts,
                "cone_iterations": plan.step_policy.cone_solver.maximum_iterations,
                "path_intervals": plan.step_policy.path.maximum_intervals,
            },
            algorithm_facts={
                "step": step.step_id,
                "policy": plan.step_policy.policy_id,
                "case_ndim": case_ndim,
            },
        )
        self.initial_state = initial
        self.default_parameters = parameters
        self.state_schema = state_schema
        self.control_schema = control_schema
        self.parameter_schema = parameter_schema
        self.reset_fallback = initial
        self.semantic_provenance = semantic
        self.numeric_revision = numeric
        self.execution_signature = signature
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True
        self.plant_id = canonical_fingerprint(
            {
                "kind": "prepared-native-articulated-plant",
                "semantic": semantic.semantic_id,
                "numeric": numeric.revision_id,
                "execution": signature.signature_id,
            }
        )

    # ------------------------------------------------------------ utilities

    @property
    def actuation_mode(self) -> ArticulatedActuationMode:
        return self.plan.actuation_mode

    def _sensors(self, mechanics: ArticulatedContactSource, /) -> ArticulatedPlantSensors:
        articulation = self.model.articulation
        transforms = articulation.body_transforms(mechanics.configuration)
        frames = transforms[self.frame_body_indices] @ self.frame_local_transforms.astype(
            transforms.dtype
        )
        history = mechanics.contact_history
        normal = (
            jnp.zeros((0,), dtype=transforms.dtype)
            if history is None
            else jnp.where(history.active, history.impulse[:, 0], 0.0)
        )
        return ArticulatedPlantSensors(frames, normal, mechanics.stop_impulse)

    def bind_parameters(
        self, values: ArticulatedPlantParameters | None = None, /
    ) -> PlantParameters:
        resolved = self.default_parameters if values is None else values
        if not isinstance(resolved, ArticulatedPlantParameters):
            raise TypeError("values must be ArticulatedPlantParameters or None.")
        self.parameter_schema.validate(resolved)
        return PlantParameters(
            resolved, self.parameter_schema.schema_id, self.numeric_revision
        )

    def _runtime(self, state: PlantRuntimeState, /) -> tuple[int, ...]:
        if not isinstance(state, PlantRuntimeState):
            raise TypeError("state must be a PlantRuntimeState.")
        observed = (
            state.semantic_provenance_id,
            state.numeric_revision_id,
            state.state_schema_id,
            state.execution_signature_id,
        )
        expected = (
            self.semantic_provenance.semantic_id,
            self.numeric_revision.revision_id,
            self.state_schema.schema_id,
            self.execution_signature.signature_id,
        )
        if observed != expected:
            raise ValueError("Runtime state belongs to another plant identity.")
        return self.state_schema.validate(state.payload)

    def _gravity(self, parameters: Any, /) -> Array:
        if not isinstance(parameters, ArticulatedPlantParameters):
            raise TypeError("parameters must be ArticulatedPlantParameters.")
        if parameters.gravity.shape != (3,):
            raise ValueError("Native plant gravity parameters are shared across cases.")
        return parameters.gravity.astype(self.contact_step.dtype)

    # ----------------------------------------------------------- lifecycle

    def reset(
        self,
        keys: ArrayLike,
        parameters: PlantParameters,
        /,
        *,
        case_shape: Sequence[int] = (),
        initial_time: ArrayLike = 0,
    ) -> PlantResetResult:
        """Shared reset transaction executed through this plant's compiled body.

        Validation, fallback selection and statuses are exactly those of
        `AbstractDiscretePlant.reset`; the case shape is a static host value.
        Only typed JAX PRNG key arrays are admitted; raw ``uint32`` key data is
        refused. ``initial_time`` is normalized to a JAX array here (integers
        to ``float32`` as in the base contract) so its value never recompiles.
        """

        if not isinstance(keys, jax.Array) or not jnp.issubdtype(
            keys.dtype, jax.dtypes.prng_key
        ):
            raise TypeError(
                "The native articulated plant requires typed PRNG key arrays "
                "(jax.random.key); raw uint32 key data is not admitted."
            )
        time = jnp.asarray(initial_time)
        if not jnp.issubdtype(time.dtype, jnp.inexact):
            time = time.astype(jnp.float32)
        return _compiled_plant_reset(self, keys, parameters, tuple(case_shape), time)

    def step(
        self,
        context: PlantStepContext,
        source: PlantRuntimeState,
        commands: PyTree[Any] | None,
        parameters: PlantParameters,
        /,
    ) -> PlantStepResult:
        """Shared step transaction executed through this plant's compiled body.

        Repeated steps with the same plant structure and array shapes reuse
        one executable; source state, commands, parameters and the interval
        are dynamic, and validation/commit semantics are the base contract's.
        """

        return _compiled_plant_step(self, context, source, commands, parameters)

    def propose_reset(
        self,
        keys: Array,
        parameters: Any,
        /,
        *,
        case_shape: tuple[int, ...],
        initial_time: Array,
    ) -> PlantProposal:
        del keys, initial_time
        self._gravity(parameters)
        admissibility = self.contact_step.admissibility(self.initial_state.mechanics)

        def broadcast(tree: PyTree[Any]) -> PyTree[Any]:
            return jax.tree.map(
                lambda leaf: jnp.broadcast_to(leaf, case_shape + jnp.shape(leaf)), tree
            )

        payload = broadcast(self.initial_state)
        successful = jnp.broadcast_to(admissibility.successful, case_shape)
        status = jnp.broadcast_to(admissibility.status, case_shape).astype(jnp.int32)
        evidence = ArticulatedPlantResetEvidence(broadcast(admissibility), self.plant_id)
        return PlantProposal(
            payload,
            payload,
            jnp.ones(case_shape, dtype=jnp.bool_),
            successful,
            status,
            status,
            evidence,
        )

    def _transition(
        self,
        source: ArticulatedPlantState,
        command: Array,
        duration: Array,
        gravity: Array,
        /,
    ) -> tuple[
        ArticulatedPlantState,
        ArticulatedPlantState,
        Array,
        Array,
        Array,
        Array,
        ArticulatedPlantStepEvidence,
    ]:
        """Pure single-case transition of one complete payload."""

        plan = self.plan
        revision_valid = (source.geometry_revision == plan.geometry_revision) & (
            source.material_revision == plan.material_revision
        )
        result = self.contact_step.step(
            source.mechanics,
            duration,
            command,
            gravity,
            geometry_revision=source.geometry_revision,
            material_revision=source.material_revision,
        )
        mechanics = result.candidate
        epoch = source.epoch + 1
        candidate = ArticulatedPlantState(
            mechanics.configuration,
            mechanics.winding,
            mechanics.velocity,
            mechanics.contact_history,
            mechanics.stop_impulse,
            mechanics.stop_branch,
            command.astype(source.command.dtype),
            epoch,
            epoch,
            self._sensors(mechanics),
            source.geometry_revision,
            source.material_revision,
        )
        successful = result.successful & revision_valid
        status = jnp.where(
            revision_valid, result.status, REVISION_MISMATCH_STATUS
        ).astype(jnp.int32)
        accepted = tree_where(successful, candidate, source)
        evidence = ArticulatedPlantStepEvidence(result, revision_valid, self.plant_id)
        return (
            candidate,
            accepted,
            jnp.asarray(True),
            successful,
            status,
            result.status,
            evidence,
        )

    def propose_step(
        self,
        context: PlantStepContext,
        source: Any,
        commands: Any,
        parameters: Any,
        keys: Array,
        /,
    ) -> PlantProposal:
        del keys
        if not isinstance(source, ArticulatedPlantState):
            raise TypeError("source must be ArticulatedPlantState.")
        gravity = self._gravity(parameters)
        case_shape = self.state_schema.validate(source)
        dtype = self.contact_step.dtype
        if self.control_schema is None:
            if commands is not None:
                raise TypeError("An autonomous native plant accepts no commands.")
            command = jnp.zeros(case_shape + (0,), dtype=dtype)
        else:
            if not isinstance(commands, ArticulatedPlantCommand):
                raise TypeError("commands must be ArticulatedPlantCommand.")
            command = jnp.broadcast_to(
                commands.values.astype(dtype),
                case_shape + (self.contact_step.actuation_size,),
            )
        duration = jnp.broadcast_to(context.duration.astype(dtype), case_shape)
        candidate, accepted, attempted, successful, status, backend, evidence = _casewise(
            self._transition, case_shape, (source, command, duration), (gravity,)
        )
        return PlantProposal(
            candidate, accepted, attempted, successful, status, backend, evidence
        )

    # ---------------------------------------------------- observe / refresh

    def _observe_case(
        self,
        payload: ArticulatedPlantState,
        time: Array,
        step_index: Array,
    ) -> ArticulatedPlantObservation:
        lifted = self.model.coordinates.lift(
            payload.configuration,
            LiftedJointCoordinateState(payload.winding),
        )
        finite = (
            jnp.all(jnp.isfinite(payload.configuration))
            & jnp.all(jnp.isfinite(payload.velocity))
            & jnp.all(jnp.isfinite(payload.sensors.frame_transforms))
        )
        return ArticulatedPlantObservation(
            time,
            step_index,
            payload.configuration,
            lifted,
            payload.velocity,
            payload.command,
            payload.sensors.frame_transforms,
            payload.sensors.contact_normal_impulse,
            payload.sensors.stop_impulse,
            payload.epoch,
            payload.sensor_epoch,
            payload.sensor_epoch == payload.epoch,
            finite,
        )

    def observe(self, state: PlantRuntimeState, /) -> ArticulatedPlantObservation:
        """Project accepted states without advancing or mutating them."""

        case_shape = self._runtime(state)
        return _casewise(
            self._observe_case,
            case_shape,
            (state.payload, state.time, state.step_index),
            (),
        )

    def control_view(self, state: PlantRuntimeState, /) -> ArticulatedPlantControlView:
        """Lifted generalized coordinates and velocities of accepted states."""

        observation = self.observe(state)
        return ArticulatedPlantControlView(
            observation.lifted_configuration, observation.velocity
        )

    def _refresh_case(
        self, payload: ArticulatedPlantState
    ) -> tuple[
        ArticulatedPlantState,
        ArticulatedPlantState,
        Array,
        Array,
        ArticulatedContactAdmissibility,
    ]:
        admissibility = self.contact_step.admissibility(payload.mechanics)
        revision_valid = (payload.geometry_revision == self.plan.geometry_revision) & (
            payload.material_revision == self.plan.material_revision
        )
        candidate = ArticulatedPlantState(
            payload.configuration,
            payload.winding,
            payload.velocity,
            payload.contact_history,
            payload.stop_impulse,
            payload.stop_branch,
            payload.command,
            payload.epoch,
            payload.epoch,
            self._sensors(payload.mechanics),
            payload.geometry_revision,
            payload.material_revision,
        )
        successful = admissibility.successful & revision_valid
        status = jnp.where(
            revision_valid, admissibility.status, REVISION_MISMATCH_STATUS
        ).astype(jnp.int32)
        accepted = tree_where(successful, candidate, payload)
        return candidate, accepted, successful, status, admissibility

    def refresh(self, state: PlantRuntimeState, /) -> ArticulatedPlantRefreshResult:
        """Resample sensors at accepted states; failed cases keep their payload."""

        case_shape = self._runtime(state)
        candidate, accepted, successful, status, admissibility = _casewise(
            self._refresh_case, case_shape, (state.payload,), ()
        )
        ids = (
            state.semantic_provenance_id,
            state.numeric_revision_id,
            state.state_schema_id,
            state.execution_signature_id,
        )
        candidate_state = PlantRuntimeState(
            candidate, state.time, state.step_index, state.key, *ids
        )
        accepted_state = PlantRuntimeState(
            accepted, state.time, state.step_index, state.key, *ids
        )
        return ArticulatedPlantRefreshResult(
            candidate_state,
            accepted_state,
            jnp.ones(case_shape, dtype=jnp.bool_),
            successful,
            status,
            self.observe(accepted_state),
            admissibility,
        )


def prepare_articulated_plant(
    model: NativeRobotModel | PreparedNativeRobotModel,
    plan: NativeArticulatedPlantPlan,
    /,
) -> PreparedArticulatedPlant:
    """Prepare the complete-state native articulated plant of one robot model."""

    return PreparedArticulatedPlant(model, plan)


@eqx.filter_jit
def _compiled_plant_reset(
    plant: PreparedArticulatedPlant,
    keys: Array,
    parameters: PlantParameters,
    case_shape: tuple[int, ...],
    initial_time: Array,
    /,
) -> PlantResetResult:
    """Compiler boundary of the shared reset transaction for this plant."""

    return AbstractDiscretePlant.reset(
        plant, keys, parameters, case_shape=case_shape, initial_time=initial_time
    )


@eqx.filter_jit
def _compiled_plant_step(
    plant: PreparedArticulatedPlant,
    context: PlantStepContext,
    source: PlantRuntimeState,
    commands: PyTree[Any] | None,
    parameters: PlantParameters,
    /,
) -> PlantStepResult:
    """Compiler boundary of the shared step transaction for this plant."""

    return AbstractDiscretePlant.step(plant, context, source, commands, parameters)


__all__ = [
    "REVISION_MISMATCH_STATUS",
    "ArticulatedPlantCommand",
    "ArticulatedPlantControlView",
    "ArticulatedPlantObservation",
    "ArticulatedPlantParameters",
    "ArticulatedPlantRefreshResult",
    "ArticulatedPlantResetEvidence",
    "ArticulatedPlantSensors",
    "ArticulatedPlantState",
    "ArticulatedPlantStatus",
    "ArticulatedPlantStepEvidence",
    "NativeArticulatedPlantPlan",
    "PreparedArticulatedPlant",
    "prepare_articulated_plant",
]
