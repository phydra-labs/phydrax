#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

import phydrax.ein as ein

from ..._array_tree import ArrayPyTreeSchema
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import (
    ExecutableSignature,
    NumericRevision,
    SemanticProvenance,
    strict_module_payload,
)
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle._rigid_body import quaternion_rotation_matrix
from ...dynamics._plant import (
    AbstractDiscretePlant,
    PlantParameters,
    PlantProposal,
    PlantResetResult,
    PlantRuntimeState,
    PlantStepContext,
    PlantStepResult,
)
from ..solid_mechanics._rod_floating import (
    FloatingReducedRodPlant,
    FloatingReducedRodPlantControl,
    FloatingReducedRodPlantState,
)
from ..solid_mechanics._rod_plant import PreparedReducedRodPlant, ReducedRodPlantState
from ..solid_mechanics._rod_tendon_plant import (
    TendonDrivenRodPlant,
    TendonDrivenRodPlantCommand,
    TendonDrivenRodPlantState,
)


class HybridRigidSoftStatus(IntEnum):
    """Profile-owned statuses; child backend statuses remain in the evidence."""

    SUCCESS = 0
    INCOMPATIBLE_DURATION = 1
    INVALID_SOURCE_ATTACHMENT = 2
    INVALID_WRENCH_ROUTE = 3
    RIGID_STEP_FAILED = 4
    SOFT_STEP_FAILED = 5
    TOPOLOGY_CHANGED = 6
    INVALID_ACCEPTED_ATTACHMENT = 7
    RIGID_RESET_FAILED = 8
    SOFT_RESET_FAILED = 9


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be a non-empty string.")
    return value.strip()


def _nonnegative_tolerance(value: float, owner: str, /) -> float:
    tolerance = float(value)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"{owner} must be finite and nonnegative.")
    return tolerance


def _proper_rotation(value: ArrayLike, dimension: int, owner: str, /) -> Array:
    rotation = np.asarray(value)
    if rotation.shape != (dimension, dimension):
        raise ValueError(f"{owner} must have shape {(dimension, dimension)}.")
    if rotation.dtype.kind != "f" or not np.all(np.isfinite(rotation)):
        raise TypeError(f"{owner} must be a finite real floating-point array.")
    tolerance = np.finfo(rotation.dtype).eps * 128.0
    if not np.allclose(
        rotation.T @ rotation, np.eye(dimension), rtol=0.0, atol=tolerance
    ):
        raise ValueError(f"{owner} must be orthogonal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=tolerance):
        raise ValueError(f"{owner} must be a proper rotation.")
    return jnp.asarray(rotation)


def _local_position(value: ArrayLike, dimension: int, owner: str, /) -> Array:
    position = np.asarray(value)
    if position.shape != (dimension,):
        raise ValueError(f"{owner} must have shape {(dimension,)}.")
    if position.dtype.kind != "f" or not np.all(np.isfinite(position)):
        raise TypeError(f"{owner} must be a finite real floating-point array.")
    return jnp.asarray(position)


def _validate_frame(
    frame: AttachmentFrameState, dimension: int, owner: str, /
) -> tuple[int, ...]:
    if not isinstance(frame, AttachmentFrameState):
        raise TypeError(f"{owner} must return AttachmentFrameState.")
    position = jnp.asarray(frame.position)
    rotation = jnp.asarray(frame.rotation)
    linear = jnp.asarray(frame.linear_velocity)
    angular = jnp.asarray(frame.angular_velocity)
    case_shape = position.shape[:-1]
    angular_dimension = 1 if dimension == 2 else 3
    expected = (
        (owner + ".position", position.shape, case_shape + (dimension,)),
        (owner + ".rotation", rotation.shape, case_shape + (dimension, dimension)),
        (owner + ".linear_velocity", linear.shape, case_shape + (dimension,)),
        (owner + ".angular_velocity", angular.shape, case_shape + (angular_dimension,)),
    )
    for name, observed, required in expected:
        if observed != required:
            raise ValueError(f"{name} must have shape {required}; got {observed}.")
    dtypes = {np.dtype(array.dtype) for array in (position, rotation, linear, angular)}
    if len(dtypes) != 1 or next(iter(dtypes)).kind != "f":
        raise TypeError(f"{owner} frame arrays must share one real floating dtype.")
    return case_shape


def _mask(value: ArrayLike, case_shape: tuple[int, ...], owner: str, /) -> Array:
    mask = jnp.asarray(value)
    if np.dtype(mask.dtype) != np.dtype(bool):
        raise TypeError(f"{owner} must have boolean dtype.")
    if mask.shape == ():
        return jnp.broadcast_to(mask, case_shape)
    if mask.shape != case_shape:
        raise ValueError(f"{owner} must be scalar or have shape {case_shape}.")
    return mask


def _all_masks(values: tuple[Array, ...], case_shape: tuple[int, ...], /) -> Array:
    result = jnp.ones(case_shape, dtype=bool)
    for value in values:
        result = result & _mask(value, case_shape, "Hybrid evidence mask")
    return result


def _cross(dimension: int, left: Array, right: Array, /) -> Array:
    if dimension == 3:
        return jnp.cross(left, right)
    return (left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0])[..., None]


def _angular_cross(dimension: int, angular: Array, vector: Array, /) -> Array:
    if dimension == 3:
        return jnp.cross(angular, vector)
    omega = angular[..., 0]
    return jnp.stack((-omega * vector[..., 1], omega * vector[..., 0]), axis=-1)


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(jnp.square(value), axis=-1))


def _dot(left: Array, right: Array, /) -> Array:
    return jnp.sum(left * right, axis=-1)


def _branch_keys(keys: ArrayLike, case_shape: tuple[int, ...], /) -> tuple[Array, Array]:
    array = jnp.asarray(keys)
    typed = jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)
    if typed:
        if array.shape != case_shape:
            raise ValueError("Typed proposal keys do not match the hybrid case shape.")
        data = jax.random.key_data(array)
    else:
        if np.dtype(array.dtype) != np.dtype(jnp.uint32) or array.shape != case_shape + (
            2,
        ):
            raise ValueError("Legacy proposal keys do not match the hybrid case shape.")
        data = array
    flattened = jnp.reshape(jax.random.wrap_key_data(data), (-1,))
    split = jax.vmap(lambda key: jax.random.split(key, 2))(flattened)
    first = jnp.reshape(split[:, 0], case_shape)
    second = jnp.reshape(split[:, 1], case_shape)
    if typed:
        return first, second
    return jax.random.key_data(first), jax.random.key_data(second)


def _validate_intrinsic_template(
    schema: ArrayPyTreeSchema, template: PyTree[Any], owner: str, /
) -> None:
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    if treedef != schema.treedef:
        raise ValueError(f"{owner} PyTree structure does not match its schema.")
    paths = tuple(jax.tree_util.keystr(path) or "<root>" for path, _ in path_leaves)
    if paths != schema.leaf_paths:
        raise ValueError(f"{owner} leaf paths do not match its schema.")
    for (_, value), leaf in zip(path_leaves, schema.leaves, strict=True):
        array = jnp.asarray(value)
        if array.shape != leaf.shape:
            raise ValueError(f"{owner} leaf {leaf.path} has the wrong intrinsic shape.")
        if np.dtype(array.dtype) != leaf.dtype:
            raise TypeError(f"{owner} leaf {leaf.path} has the wrong dtype.")


def _broadcast_intrinsic(
    schema: ArrayPyTreeSchema,
    template: PyTree[Any],
    case_shape: tuple[int, ...],
    owner: str,
    /,
) -> PyTree[Array]:
    _validate_intrinsic_template(schema, template, owner)
    return schema.treedef.unflatten(
        tuple(
            jnp.broadcast_to(jnp.asarray(value), case_shape + leaf.shape)
            for value, leaf in zip(
                jax.tree_util.tree_leaves(template), schema.leaves, strict=True
            )
        )
    )


def _plant_ids(plant: AbstractDiscretePlant, /) -> tuple[str, str, str, str]:
    return (
        plant.semantic_provenance.semantic_id,
        plant.numeric_revision.revision_id,
        plant.state_schema.schema_id,
        plant.execution_signature.signature_id,
    )


def _runtime_state(
    plant: AbstractDiscretePlant,
    payload: PyTree[Any],
    context: PlantStepContext,
    key: Array,
    /,
) -> PlantRuntimeState:
    return PlantRuntimeState(
        payload,
        context.source_time,
        context.step_index,
        key,
        *_plant_ids(plant),
    )


def _parameters(plant: AbstractDiscretePlant, values: PyTree[Any], /) -> PlantParameters:
    return PlantParameters(
        values, plant.parameter_schema.schema_id, plant.numeric_revision
    )


def _validate_bound_parameters(
    plant: AbstractDiscretePlant, parameters: PlantParameters, owner: str, /
) -> None:
    if not isinstance(parameters, PlantParameters):
        raise TypeError(f"{owner} must be PlantParameters.")
    if parameters.schema_id != plant.parameter_schema.schema_id:
        raise ValueError(f"{owner} schema does not match its prepared plant.")
    if parameters.numeric_revision.revision_id != plant.numeric_revision.revision_id:
        raise ValueError(f"{owner} numeric revision does not match its prepared plant.")
    if parameters.numeric_revision.semantic_id != plant.semantic_provenance.semantic_id:
        raise ValueError(f"{owner} provenance does not match its prepared plant.")
    if plant.parameter_schema.case_ndim != 0:
        raise ValueError("Hybrid child parameter schemas must be scalar/shared.")
    plant.parameter_schema.validate(parameters.values)


class AttachmentFrameState(StrictModule):
    """World-frame pose and twist of one rigid or soft parent frame."""

    position: Array
    rotation: Array
    linear_velocity: Array
    angular_velocity: Array

    def __init__(
        self,
        position: ArrayLike,
        rotation: ArrayLike,
        linear_velocity: ArrayLike,
        angular_velocity: ArrayLike,
        /,
    ):
        self.position = jnp.asarray(position)
        self.rotation = jnp.asarray(rotation)
        self.linear_velocity = jnp.asarray(linear_velocity)
        self.angular_velocity = jnp.asarray(angular_velocity)


class RigidFrameAttachmentPlan(StrictModule, NonTrainableState):
    """Fixed transform from a named rigid parent frame to an attachment frame."""

    frame_id: str = eqx.field(static=True)
    local_position: Array
    local_rotation: Array
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        frame_id: str,
        local_position: ArrayLike,
        local_rotation: ArrayLike,
        /,
    ):
        position = np.asarray(local_position)
        if position.ndim != 1 or position.shape[0] not in (2, 3):
            raise ValueError("Rigid local_position must be a planar or spatial vector.")
        dimension = int(position.shape[0])
        position_ = _local_position(position, dimension, "Rigid local_position")
        rotation = _proper_rotation(local_rotation, dimension, "Rigid local_rotation")
        frame_id_ = _identifier(frame_id, "frame_id")
        plan_id = canonical_fingerprint(
            {
                "kind": "hybrid-rigid-frame-attachment",
                "frame_id": frame_id_,
                "local_position": array_tree_fingerprint(position_),
                "local_rotation": array_tree_fingerprint(rotation),
            }
        )
        self.frame_id = frame_id_
        self.local_position = position_
        self.local_rotation = rotation
        self.dimension = dimension
        self.plan_id = plan_id


class SoftEndpointAttachmentPlan(StrictModule, NonTrainableState):
    """Fixed transform from a named soft endpoint to an attachment frame."""

    endpoint_id: str = eqx.field(static=True)
    local_position: Array
    local_rotation: Array
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        endpoint_id: str,
        local_position: ArrayLike,
        local_rotation: ArrayLike,
        /,
    ):
        position = np.asarray(local_position)
        if position.ndim != 1 or position.shape[0] not in (2, 3):
            raise ValueError("Soft local_position must be a planar or spatial vector.")
        dimension = int(position.shape[0])
        position_ = _local_position(position, dimension, "Soft local_position")
        rotation = _proper_rotation(local_rotation, dimension, "Soft local_rotation")
        endpoint_id_ = _identifier(endpoint_id, "endpoint_id")
        plan_id = canonical_fingerprint(
            {
                "kind": "hybrid-soft-endpoint-attachment",
                "endpoint_id": endpoint_id_,
                "local_position": array_tree_fingerprint(position_),
                "local_rotation": array_tree_fingerprint(rotation),
            }
        )
        self.endpoint_id = endpoint_id_
        self.local_position = position_
        self.local_rotation = rotation
        self.dimension = dimension
        self.plan_id = plan_id


class RigidSoftAttachmentPlan(StrictModule, NonTrainableState):
    """One fixed rigid-frame/soft-endpoint coincidence constraint."""

    rigid: RigidFrameAttachmentPlan
    soft: SoftEndpointAttachmentPlan
    position_tolerance: float = eqx.field(static=True)
    rotation_tolerance: float = eqx.field(static=True)
    velocity_tolerance: float = eqx.field(static=True)
    balance_tolerance: float = eqx.field(static=True)
    power_tolerance: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    attachment_id: str = eqx.field(static=True)

    def __init__(
        self,
        rigid: RigidFrameAttachmentPlan,
        soft: SoftEndpointAttachmentPlan,
        /,
        *,
        position_tolerance: float = 1.0e-7,
        rotation_tolerance: float = 1.0e-7,
        velocity_tolerance: float = 1.0e-7,
        balance_tolerance: float = 1.0e-7,
        power_tolerance: float = 1.0e-7,
    ):
        if not isinstance(rigid, RigidFrameAttachmentPlan):
            raise TypeError("rigid must be RigidFrameAttachmentPlan.")
        if not isinstance(soft, SoftEndpointAttachmentPlan):
            raise TypeError("soft must be SoftEndpointAttachmentPlan.")
        if rigid.dimension != soft.dimension:
            raise ValueError("Rigid and soft attachment dimensions must match.")
        tolerances = (
            _nonnegative_tolerance(position_tolerance, "position_tolerance"),
            _nonnegative_tolerance(rotation_tolerance, "rotation_tolerance"),
            _nonnegative_tolerance(velocity_tolerance, "velocity_tolerance"),
            _nonnegative_tolerance(balance_tolerance, "balance_tolerance"),
            _nonnegative_tolerance(power_tolerance, "power_tolerance"),
        )
        attachment_id = canonical_fingerprint(
            {
                "kind": "hybrid-rigid-soft-attachment",
                "rigid": rigid.plan_id,
                "soft": soft.plan_id,
                "position_tolerance": tolerances[0],
                "rotation_tolerance": tolerances[1],
                "velocity_tolerance": tolerances[2],
                "balance_tolerance": tolerances[3],
                "power_tolerance": tolerances[4],
            }
        )
        self.rigid = rigid
        self.soft = soft
        self.position_tolerance = tolerances[0]
        self.rotation_tolerance = tolerances[1]
        self.velocity_tolerance = tolerances[2]
        self.balance_tolerance = tolerances[3]
        self.power_tolerance = tolerances[4]
        self.dimension = rigid.dimension
        self.attachment_id = attachment_id


class SynchronizedStepPolicy(StrictModule):
    """The only hybrid policy: one source-explicit step over one shared context."""

    fixed_duration: float | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, *, fixed_duration: float | None = None):
        if fixed_duration is None:
            duration = None
        else:
            duration = float(fixed_duration)
            if not isfinite(duration) or duration <= 0.0:
                raise ValueError("fixed_duration must be finite and positive.")
        self.fixed_duration = duration
        self.policy_id = canonical_fingerprint(
            {
                "kind": "hybrid-synchronized-step-policy",
                "coupling_stage": "source-explicit",
                "fixed_duration": duration,
            }
        )

    def duration_valid(self, context: PlantStepContext, /) -> Array:
        if self.fixed_duration is None:
            return jnp.ones(context.duration.shape, dtype=bool)
        working_dtype = jnp.result_type(context.duration, jnp.asarray(1.0))
        observed = jnp.asarray(context.duration, dtype=working_dtype)
        duration = jnp.asarray(self.fixed_duration, dtype=working_dtype)
        tolerance = jnp.finfo(working_dtype).eps * jnp.maximum(1.0, duration) * 16.0
        return jnp.isfinite(observed) & (jnp.abs(observed - duration) <= tolerance)


class AbstractHybridPlantPort(StrictModule):
    """Prepared state/frame/dual-effort adapter owned by one child plant profile."""

    semantic_provenance_id: AbstractAttribute[str]
    numeric_revision_id: AbstractAttribute[str]
    state_schema_id: AbstractAttribute[str]
    control_schema_id: AbstractAttribute[str | None]
    execution_signature_id: AbstractAttribute[str]
    topology_id: AbstractAttribute[str]
    step_policy_id: AbstractAttribute[str]
    frame_ids: AbstractAttribute[tuple[str, ...]]
    supports_external_wrenches: AbstractAttribute[bool]

    @abstractmethod
    def frame_state(self, payload: PyTree[Any], frame_id: str, /) -> AttachmentFrameState:
        """Extract one exact world-frame pose and twist from a complete payload."""
        raise NotImplementedError

    @abstractmethod
    def apply_frame_wrenches(
        self,
        payload: PyTree[Any],
        commands: PyTree[Any] | None,
        wrenches: tuple[FrameWrench, ...],
        /,
    ) -> PyTree[Any] | None:
        """Pull world-frame covectors into this plant's typed effort command."""
        raise NotImplementedError

    @abstractmethod
    def topology_unchanged(self, source: PyTree[Any], candidate: PyTree[Any], /) -> Array:
        """Return a case mask proving every fixed topology/mode leaf is unchanged."""
        raise NotImplementedError


class FrameWrench(StrictModule):
    """World-expressed force and moment about one named child parent-frame origin."""

    force: Array
    moment: Array
    frame_id: str = eqx.field(static=True)

    def __init__(self, force: ArrayLike, moment: ArrayLike, frame_id: str, /):
        self.force = jnp.asarray(force)
        self.moment = jnp.asarray(moment)
        self.frame_id = _identifier(frame_id, "FrameWrench frame_id")


def _endpoint_frame_ids(
    base_frame_id: str, tip_frame_id: str, owner: str, /
) -> tuple[str, str]:
    base = _identifier(base_frame_id, f"{owner} base_frame_id")
    tip = _identifier(tip_frame_id, f"{owner} tip_frame_id")
    if base == tip:
        raise ValueError(f"{owner} endpoint frame IDs must be distinct.")
    return base, tip


def _rod_frame_rotation(orientation: Array, dimension: int, /) -> Array:
    if dimension == 3:
        return quaternion_rotation_matrix(orientation)
    cosine = jnp.cos(orientation)
    sine = jnp.sin(orientation)
    return jnp.stack(
        (
            jnp.stack((cosine, -sine), axis=-1),
            jnp.stack((sine, cosine), axis=-1),
        ),
        axis=-2,
    )


def _reduced_rod_endpoint_frame(
    reduction: Any,
    state: Any,
    frame_id: str,
    frame_ids: tuple[str, str],
    /,
) -> AttachmentFrameState:
    if frame_id not in frame_ids:
        raise ValueError("Unknown reduced-rod endpoint frame.")
    native = reduction.lift(state)
    base = frame_id == frame_ids[0]
    node_index = int(np.asarray(reduction.path_node_ids[0 if base else -1]))
    segment_index = 0 if base else reduction.rod.plan.segment_count - 1
    dimension = reduction.rod.plan.dimension
    angular_velocity = native.angular_velocities[segment_index]
    return AttachmentFrameState(
        native.positions[node_index],
        _rod_frame_rotation(native.orientations[segment_index], dimension),
        native.velocities[node_index],
        angular_velocity if dimension == 3 else angular_velocity[None],
    )


def _floating_rod_endpoint_frame(
    plant: FloatingReducedRodPlant,
    state: Any,
    frame_id: str,
    frame_ids: tuple[str, str],
    /,
) -> AttachmentFrameState:
    if frame_id not in frame_ids:
        raise ValueError("Unknown floating reduced-rod endpoint frame.")
    native = plant.prepared.lift(state)
    reduction = plant.prepared.reduction
    base = frame_id == frame_ids[0]
    node_index = int(np.asarray(reduction.path_node_ids[0 if base else -1]))
    segment_index = 0 if base else reduction.rod.plan.segment_count - 1
    dimension = reduction.rod.plan.dimension
    angular_velocity = native.angular_velocities[segment_index]
    return AttachmentFrameState(
        native.positions[node_index],
        _rod_frame_rotation(native.orientations[segment_index], dimension),
        native.velocities[node_index],
        angular_velocity if dimension == 3 else angular_velocity[None],
    )


def _endpoint_native_wrenches(
    reduction: Any,
    native: Any,
    frame_ids: tuple[str, str],
    wrenches: tuple[FrameWrench, ...],
    /,
) -> tuple[Array, Array]:
    if not isinstance(wrenches, tuple) or any(
        not isinstance(wrench, FrameWrench) for wrench in wrenches
    ):
        raise TypeError("wrenches must be a tuple of FrameWrench values.")
    forces = jnp.zeros_like(native.velocities)
    moments = jnp.zeros_like(native.angular_velocities)
    dimension = reduction.rod.plan.dimension
    angular_dimension = 1 if dimension == 2 else 3
    for wrench in wrenches:
        if wrench.frame_id not in frame_ids:
            raise ValueError("Wrench names an unknown reduced-rod endpoint frame.")
        force = jnp.asarray(wrench.force)
        moment = jnp.asarray(wrench.moment)
        if force.shape != (dimension,) or moment.shape != (angular_dimension,):
            raise ValueError("Endpoint wrench has the wrong force or moment shape.")
        if np.dtype(force.dtype) != np.dtype(forces.dtype) or np.dtype(
            moment.dtype
        ) != np.dtype(moments.dtype):
            raise TypeError("Endpoint wrench dtype must match the prepared rod.")
        base = wrench.frame_id == frame_ids[0]
        node_index = int(np.asarray(reduction.path_node_ids[0 if base else -1]))
        segment_index = 0 if base else reduction.rod.plan.segment_count - 1
        forces = forces.at[node_index].add(force)
        moments = moments.at[segment_index].add(moment if dimension == 3 else moment[0])
    return forces, moments


class PreparedReducedRodPlantPort(AbstractHybridPlantPort, NonTrainableState):
    """Identity-bound endpoint adapter for the passive fixed-base rod plant."""

    plant: PreparedReducedRodPlant
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    control_schema_id: str | None = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    step_policy_id: str = eqx.field(static=True)
    frame_ids: tuple[str, str] = eqx.field(static=True)
    supports_external_wrenches: bool = eqx.field(static=True)

    def __init__(
        self,
        plant: PreparedReducedRodPlant,
        step_policy: SynchronizedStepPolicy,
        /,
        *,
        base_frame_id: str = "base",
        tip_frame_id: str = "tip",
    ):
        if not isinstance(plant, PreparedReducedRodPlant):
            raise TypeError("plant must be PreparedReducedRodPlant.")
        if not isinstance(step_policy, SynchronizedStepPolicy):
            raise TypeError("step_policy must be SynchronizedStepPolicy.")
        frame_ids = _endpoint_frame_ids(
            base_frame_id, tip_frame_id, "PreparedReducedRodPlantPort"
        )
        self.plant = plant
        self.semantic_provenance_id = plant.semantic_provenance.semantic_id
        self.numeric_revision_id = plant.numeric_revision.revision_id
        self.state_schema_id = plant.state_schema.schema_id
        self.control_schema_id = None
        self.execution_signature_id = plant.execution_signature.signature_id
        self.topology_id = canonical_fingerprint(
            {
                "kind": "prepared-reduced-rod-hybrid-port",
                "rod": plant.dynamics.reduction.rod.prepared_id,
                "reduction": plant.dynamics.reduction.prepared_id,
                "frame_ids": frame_ids,
            }
        )
        self.step_policy_id = step_policy.policy_id
        self.frame_ids = frame_ids
        self.supports_external_wrenches = False

    def frame_state(
        self, payload: ReducedRodPlantState, frame_id: str, /
    ) -> AttachmentFrameState:
        if not isinstance(payload, ReducedRodPlantState):
            raise TypeError("payload must be ReducedRodPlantState.")
        self.plant.state_schema.validate(payload)
        return _reduced_rod_endpoint_frame(
            self.plant.dynamics.reduction,
            payload.reduced_state,
            frame_id,
            self.frame_ids,
        )

    def apply_frame_wrenches(
        self,
        payload: ReducedRodPlantState,
        commands: None,
        wrenches: tuple[FrameWrench, ...],
        /,
    ) -> None:
        del payload, commands, wrenches
        raise ValueError(
            "PreparedReducedRodPlant is passive and has no external-wrench command."
        )

    def topology_unchanged(
        self, source: ReducedRodPlantState, candidate: ReducedRodPlantState, /
    ) -> Array:
        if not isinstance(source, ReducedRodPlantState) or not isinstance(
            candidate, ReducedRodPlantState
        ):
            raise TypeError("Reduced-rod payloads must be ReducedRodPlantState.")
        self.plant.state_schema.validate(source)
        self.plant.state_schema.validate(candidate)
        return jnp.asarray(True)


class FloatingReducedRodPlantPort(AbstractHybridPlantPort, NonTrainableState):
    """Identity-bound endpoint frame and covector adapter for a floating rod."""

    plant: FloatingReducedRodPlant
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    control_schema_id: str | None = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    step_policy_id: str = eqx.field(static=True)
    frame_ids: tuple[str, str] = eqx.field(static=True)
    supports_external_wrenches: bool = eqx.field(static=True)

    def __init__(
        self,
        plant: FloatingReducedRodPlant,
        step_policy: SynchronizedStepPolicy,
        /,
        *,
        base_frame_id: str = "base",
        tip_frame_id: str = "tip",
    ):
        if not isinstance(plant, FloatingReducedRodPlant):
            raise TypeError("plant must be FloatingReducedRodPlant.")
        if not isinstance(step_policy, SynchronizedStepPolicy):
            raise TypeError("step_policy must be SynchronizedStepPolicy.")
        frame_ids = _endpoint_frame_ids(
            base_frame_id, tip_frame_id, "FloatingReducedRodPlantPort"
        )
        self.plant = plant
        self.semantic_provenance_id = plant.semantic_provenance.semantic_id
        self.numeric_revision_id = plant.numeric_revision.revision_id
        self.state_schema_id = plant.state_schema.schema_id
        self.control_schema_id = plant.control_schema.schema_id
        self.execution_signature_id = plant.execution_signature.signature_id
        self.topology_id = canonical_fingerprint(
            {
                "kind": "floating-reduced-rod-hybrid-port",
                "rod": plant.prepared.reduction.rod.prepared_id,
                "reduction": plant.prepared.reduction.prepared_id,
                "floating": plant.prepared.prepared_id,
                "frame_ids": frame_ids,
            }
        )
        self.step_policy_id = step_policy.policy_id
        self.frame_ids = frame_ids
        self.supports_external_wrenches = True

    def frame_state(
        self, payload: FloatingReducedRodPlantState, frame_id: str, /
    ) -> AttachmentFrameState:
        if not isinstance(payload, FloatingReducedRodPlantState):
            raise TypeError("payload must be FloatingReducedRodPlantState.")
        self.plant.state_schema.validate(payload)
        return _floating_rod_endpoint_frame(
            self.plant, payload.rod_state, frame_id, self.frame_ids
        )

    def apply_frame_wrenches(
        self,
        payload: FloatingReducedRodPlantState,
        commands: FloatingReducedRodPlantControl | None,
        wrenches: tuple[FrameWrench, ...],
        /,
    ) -> FloatingReducedRodPlantControl:
        if not isinstance(payload, FloatingReducedRodPlantState):
            raise TypeError("payload must be FloatingReducedRodPlantState.")
        if not isinstance(commands, FloatingReducedRodPlantControl):
            raise TypeError("commands must be FloatingReducedRodPlantControl.")
        self.plant.state_schema.validate(payload)
        self.plant.control_schema.validate(commands)
        native = self.plant.prepared.lift(payload.rod_state)
        forces, moments = _endpoint_native_wrenches(
            self.plant.prepared.reduction,
            native,
            self.frame_ids,
            wrenches,
        )
        effort = self.plant.prepared.effort_pullback_operator(payload.rod_state).mv(
            (forces, moments)
        )
        result = FloatingReducedRodPlantControl(commands.effort + effort)
        self.plant.control_schema.validate(result)
        return result

    def topology_unchanged(
        self,
        source: FloatingReducedRodPlantState,
        candidate: FloatingReducedRodPlantState,
        /,
    ) -> Array:
        if not isinstance(source, FloatingReducedRodPlantState) or not isinstance(
            candidate, FloatingReducedRodPlantState
        ):
            raise TypeError("Floating rod payloads must be FloatingReducedRodPlantState.")
        self.plant.state_schema.validate(source)
        self.plant.state_schema.validate(candidate)
        return jnp.asarray(True)


class TendonDrivenRodPlantPort(AbstractHybridPlantPort, NonTrainableState):
    """Identity-bound endpoint adapter preserving every tendon command leaf."""

    plant: TendonDrivenRodPlant
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    control_schema_id: str | None = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    step_policy_id: str = eqx.field(static=True)
    frame_ids: tuple[str, str] = eqx.field(static=True)
    supports_external_wrenches: bool = eqx.field(static=True)

    def __init__(
        self,
        plant: TendonDrivenRodPlant,
        step_policy: SynchronizedStepPolicy,
        /,
        *,
        base_frame_id: str = "base",
        tip_frame_id: str = "tip",
    ):
        if not isinstance(plant, TendonDrivenRodPlant):
            raise TypeError("plant must be TendonDrivenRodPlant.")
        if not isinstance(step_policy, SynchronizedStepPolicy):
            raise TypeError("step_policy must be SynchronizedStepPolicy.")
        if plant.control_schema is None:
            raise ValueError(
                "TendonDrivenRodPlant must expose an external-wrench command."
            )
        frame_ids = _endpoint_frame_ids(
            base_frame_id, tip_frame_id, "TendonDrivenRodPlantPort"
        )
        reduction = plant.base_plant.dynamics.reduction
        self.plant = plant
        self.semantic_provenance_id = plant.semantic_provenance.semantic_id
        self.numeric_revision_id = plant.numeric_revision.revision_id
        self.state_schema_id = plant.state_schema.schema_id
        self.control_schema_id = plant.control_schema.schema_id
        self.execution_signature_id = plant.execution_signature.signature_id
        self.topology_id = canonical_fingerprint(
            {
                "kind": "tendon-driven-reduced-rod-hybrid-port",
                "rod": reduction.rod.prepared_id,
                "reduction": reduction.prepared_id,
                "plant": plant.plant_id,
                "frame_ids": frame_ids,
            }
        )
        self.step_policy_id = step_policy.policy_id
        self.frame_ids = frame_ids
        self.supports_external_wrenches = True

    def frame_state(
        self, payload: TendonDrivenRodPlantState, frame_id: str, /
    ) -> AttachmentFrameState:
        if not isinstance(payload, TendonDrivenRodPlantState):
            raise TypeError("payload must be TendonDrivenRodPlantState.")
        self.plant.state_schema.validate(payload)
        return _reduced_rod_endpoint_frame(
            self.plant.base_plant.dynamics.reduction,
            payload.reduced_state,
            frame_id,
            self.frame_ids,
        )

    def apply_frame_wrenches(
        self,
        payload: TendonDrivenRodPlantState,
        commands: TendonDrivenRodPlantCommand | None,
        wrenches: tuple[FrameWrench, ...],
        /,
    ) -> TendonDrivenRodPlantCommand:
        if not isinstance(payload, TendonDrivenRodPlantState):
            raise TypeError("payload must be TendonDrivenRodPlantState.")
        if not isinstance(commands, TendonDrivenRodPlantCommand):
            raise TypeError("commands must be TendonDrivenRodPlantCommand.")
        self.plant.state_schema.validate(payload)
        self.plant.control_schema.validate(commands)
        reduction = self.plant.base_plant.dynamics.reduction
        native = reduction.lift(payload.reduced_state)
        forces, moments = _endpoint_native_wrenches(
            reduction, native, self.frame_ids, wrenches
        )
        effort = reduction.pullback_loads(
            payload.reduced_state.coefficients, forces, moments
        )
        result = eqx.tree_at(
            lambda value: value.external_effort,
            commands,
            commands.external_effort + effort,
        )
        self.plant.control_schema.validate(result)
        return result

    def topology_unchanged(
        self,
        source: TendonDrivenRodPlantState,
        candidate: TendonDrivenRodPlantState,
        /,
    ) -> Array:
        if not isinstance(source, TendonDrivenRodPlantState) or not isinstance(
            candidate, TendonDrivenRodPlantState
        ):
            raise TypeError("Tendon-driven payloads must be TendonDrivenRodPlantState.")
        self.plant.state_schema.validate(source)
        self.plant.state_schema.validate(candidate)
        return jnp.asarray(True)


class AttachmentWrenchCommand(StrictModule):
    """Force and free moment on the soft side, expressed in attachment axes."""

    force: Array
    moment: Array
    attachment_id: str = eqx.field(static=True)

    def __init__(self, force: ArrayLike, moment: ArrayLike, attachment_id: str, /):
        self.force = jnp.asarray(force)
        self.moment = jnp.asarray(moment)
        self.attachment_id = _identifier(attachment_id, "attachment_id")


class AttachmentKinematics(StrictModule):
    """Parent/attachment frames and coincidence evidence for one fixed route."""

    rigid_parent: AttachmentFrameState
    soft_parent: AttachmentFrameState
    rigid_attachment: AttachmentFrameState
    soft_attachment: AttachmentFrameState
    position_residual: Array
    rotation_residual: Array
    linear_velocity_residual: Array
    angular_velocity_residual: Array
    finite: Array
    coincident: Array
    successful: Array
    attachment_id: str = eqx.field(static=True)


class AttachmentWrenchRoute(StrictModule):
    """Equal-opposite attachment wrench, shifted child covectors, and power proof."""

    rigid_at_attachment: FrameWrench
    soft_at_attachment: FrameWrench
    rigid_at_parent: FrameWrench
    soft_at_parent: FrameWrench
    force_balance_residual: Array
    moment_balance_residual: Array
    rigid_power: Array
    soft_power: Array
    power_residual: Array
    finite: Array
    balanced: Array
    power_conserving: Array
    successful: Array
    attachment_id: str = eqx.field(static=True)


class HybridRigidSoftState(StrictModule):
    """Complete fixed-schema payload of both prepared child plants."""

    rigid: Any
    soft: Any
    topology_id: str = eqx.field(static=True)

    def __init__(self, rigid: PyTree[Any], soft: PyTree[Any], topology_id: str, /):
        self.rigid = rigid
        self.soft = soft
        self.topology_id = _identifier(topology_id, "topology_id")


class HybridRigidSoftParameterValues(StrictModule):
    rigid: Any
    soft: Any
    topology_id: str = eqx.field(static=True)

    def __init__(self, rigid: PyTree[Any], soft: PyTree[Any], topology_id: str, /):
        self.rigid = rigid
        self.soft = soft
        self.topology_id = _identifier(topology_id, "topology_id")


class HybridRigidSoftCommands(StrictModule):
    """Child commands plus a fixed tuple of attachment-frame soft-side wrenches."""

    rigid: Any
    soft: Any
    attachment_wrenches: tuple[AttachmentWrenchCommand, ...]
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        rigid: PyTree[Any] | None,
        soft: PyTree[Any] | None,
        attachment_wrenches: tuple[AttachmentWrenchCommand, ...],
        topology_id: str,
        /,
    ):
        self.rigid = rigid
        self.soft = soft
        self.attachment_wrenches = tuple(attachment_wrenches)
        self.topology_id = _identifier(topology_id, "topology_id")


class HybridResetEvidence(StrictModule):
    rigid: PlantResetResult
    soft: PlantResetResult
    candidate_attachments: tuple[AttachmentKinematics, ...]
    accepted_attachments: tuple[AttachmentKinematics, ...]
    topology_unchanged: Array
    successful: Array
    status: Array
    topology_id: str = eqx.field(static=True)


class HybridStepEvidence(StrictModule):
    rigid: PlantStepResult
    soft: PlantStepResult
    source_attachments: tuple[AttachmentKinematics, ...]
    candidate_attachments: tuple[AttachmentKinematics, ...]
    accepted_attachments: tuple[AttachmentKinematics, ...]
    wrench_routes: tuple[AttachmentWrenchRoute, ...]
    duration_valid: Array
    topology_unchanged: Array
    successful: Array
    status: Array
    topology_id: str = eqx.field(static=True)


def transform_attachment_frame(
    parent: AttachmentFrameState,
    local_position: ArrayLike,
    local_rotation: ArrayLike,
    /,
) -> AttachmentFrameState:
    """Apply one fixed SE(2)/SE(3) frame transform and its exact tangent map."""
    position = jnp.asarray(local_position)
    rotation = jnp.asarray(local_rotation)
    dimension = int(position.shape[0])
    _validate_frame(parent, dimension, "parent")
    if position.shape != (dimension,) or rotation.shape != (dimension, dimension):
        raise ValueError("Local attachment transform has the wrong dimension.")
    world_offset = ein.contract("...ij,j->...i", parent.rotation, position)
    attachment_rotation = ein.contract("...ij,jk->...ik", parent.rotation, rotation)
    attachment_velocity = parent.linear_velocity + _angular_cross(
        dimension, parent.angular_velocity, world_offset
    )
    return AttachmentFrameState(
        parent.position + world_offset,
        attachment_rotation,
        attachment_velocity,
        parent.angular_velocity,
    )


def evaluate_attachment_kinematics(
    plan: RigidSoftAttachmentPlan,
    rigid_parent: AttachmentFrameState,
    soft_parent: AttachmentFrameState,
    /,
) -> AttachmentKinematics:
    """Evaluate exact fixed-frame coincidence without solving child mechanics."""
    if not isinstance(plan, RigidSoftAttachmentPlan):
        raise TypeError("plan must be RigidSoftAttachmentPlan.")
    rigid_case = _validate_frame(rigid_parent, plan.dimension, "rigid_parent")
    soft_case = _validate_frame(soft_parent, plan.dimension, "soft_parent")
    if rigid_case != soft_case:
        raise ValueError("Rigid and soft frame states must share one case shape.")
    frame_dtype = np.dtype(rigid_parent.position.dtype)
    attachment_dtypes = {
        frame_dtype,
        np.dtype(soft_parent.position.dtype),
        np.dtype(plan.rigid.local_position.dtype),
        np.dtype(plan.rigid.local_rotation.dtype),
        np.dtype(plan.soft.local_position.dtype),
        np.dtype(plan.soft.local_rotation.dtype),
    }
    if len(attachment_dtypes) != 1:
        raise TypeError("Rigid, soft, and attachment frame arrays must share one dtype.")
    rigid_attachment = transform_attachment_frame(
        rigid_parent, plan.rigid.local_position, plan.rigid.local_rotation
    )
    soft_attachment = transform_attachment_frame(
        soft_parent, plan.soft.local_position, plan.soft.local_rotation
    )
    position_residual = soft_attachment.position - rigid_attachment.position
    rotation_residual = soft_attachment.rotation - rigid_attachment.rotation
    linear_residual = soft_attachment.linear_velocity - rigid_attachment.linear_velocity
    angular_residual = (
        soft_attachment.angular_velocity - rigid_attachment.angular_velocity
    )
    arrays = (
        rigid_attachment.position,
        rigid_attachment.rotation,
        rigid_attachment.linear_velocity,
        rigid_attachment.angular_velocity,
        soft_attachment.position,
        soft_attachment.rotation,
        soft_attachment.linear_velocity,
        soft_attachment.angular_velocity,
    )
    finite = jnp.ones(rigid_case, dtype=bool)
    for value in arrays:
        axes = tuple(range(len(rigid_case), value.ndim))
        finite = finite & jnp.all(jnp.isfinite(value), axis=axes)
    position_scale = jnp.maximum(
        1.0,
        jnp.maximum(_norm(rigid_attachment.position), _norm(soft_attachment.position)),
    )
    rigid_rotation_norm = jnp.sqrt(
        jnp.sum(jnp.square(rigid_attachment.rotation), axis=(-2, -1))
    )
    soft_rotation_norm = jnp.sqrt(
        jnp.sum(jnp.square(soft_attachment.rotation), axis=(-2, -1))
    )
    rotation_scale = jnp.maximum(
        1.0, jnp.maximum(rigid_rotation_norm, soft_rotation_norm)
    )
    identity = jnp.eye(plan.dimension, dtype=rigid_attachment.rotation.dtype)
    rigid_gram = ein.contract(
        "...ji,...jk->...ik",
        rigid_attachment.rotation,
        rigid_attachment.rotation,
    )
    soft_gram = ein.contract(
        "...ji,...jk->...ik",
        soft_attachment.rotation,
        soft_attachment.rotation,
    )
    if plan.dimension == 2:
        rigid_determinant = (
            rigid_attachment.rotation[..., 0, 0] * rigid_attachment.rotation[..., 1, 1]
            - rigid_attachment.rotation[..., 0, 1] * rigid_attachment.rotation[..., 1, 0]
        )
        soft_determinant = (
            soft_attachment.rotation[..., 0, 0] * soft_attachment.rotation[..., 1, 1]
            - soft_attachment.rotation[..., 0, 1] * soft_attachment.rotation[..., 1, 0]
        )
    else:
        rigid_determinant = (
            rigid_attachment.rotation[..., 0, 0]
            * (
                rigid_attachment.rotation[..., 1, 1]
                * rigid_attachment.rotation[..., 2, 2]
                - rigid_attachment.rotation[..., 1, 2]
                * rigid_attachment.rotation[..., 2, 1]
            )
            - rigid_attachment.rotation[..., 0, 1]
            * (
                rigid_attachment.rotation[..., 1, 0]
                * rigid_attachment.rotation[..., 2, 2]
                - rigid_attachment.rotation[..., 1, 2]
                * rigid_attachment.rotation[..., 2, 0]
            )
            + rigid_attachment.rotation[..., 0, 2]
            * (
                rigid_attachment.rotation[..., 1, 0]
                * rigid_attachment.rotation[..., 2, 1]
                - rigid_attachment.rotation[..., 1, 1]
                * rigid_attachment.rotation[..., 2, 0]
            )
        )
        soft_determinant = (
            soft_attachment.rotation[..., 0, 0]
            * (
                soft_attachment.rotation[..., 1, 1] * soft_attachment.rotation[..., 2, 2]
                - soft_attachment.rotation[..., 1, 2]
                * soft_attachment.rotation[..., 2, 1]
            )
            - soft_attachment.rotation[..., 0, 1]
            * (
                soft_attachment.rotation[..., 1, 0] * soft_attachment.rotation[..., 2, 2]
                - soft_attachment.rotation[..., 1, 2]
                * soft_attachment.rotation[..., 2, 0]
            )
            + soft_attachment.rotation[..., 0, 2]
            * (
                soft_attachment.rotation[..., 1, 0] * soft_attachment.rotation[..., 2, 1]
                - soft_attachment.rotation[..., 1, 1]
                * soft_attachment.rotation[..., 2, 0]
            )
        )
    rigid_gram_residual = jnp.sqrt(
        jnp.sum(jnp.square(rigid_gram - identity), axis=(-2, -1))
    )
    soft_gram_residual = jnp.sqrt(
        jnp.sum(jnp.square(soft_gram - identity), axis=(-2, -1))
    )
    proper_rotations = (
        (rigid_gram_residual <= plan.rotation_tolerance * rotation_scale)
        & (soft_gram_residual <= plan.rotation_tolerance * rotation_scale)
        & (jnp.abs(rigid_determinant - 1.0) <= plan.rotation_tolerance * rotation_scale)
        & (jnp.abs(soft_determinant - 1.0) <= plan.rotation_tolerance * rotation_scale)
    )
    linear_scale = jnp.maximum(
        1.0,
        jnp.maximum(
            _norm(rigid_attachment.linear_velocity),
            _norm(soft_attachment.linear_velocity),
        ),
    )
    angular_scale = jnp.maximum(
        1.0,
        jnp.maximum(
            _norm(rigid_attachment.angular_velocity),
            _norm(soft_attachment.angular_velocity),
        ),
    )
    coincident = (
        proper_rotations
        & (_norm(position_residual) <= plan.position_tolerance * position_scale)
        & (
            jnp.sqrt(jnp.sum(jnp.square(rotation_residual), axis=(-2, -1)))
            <= plan.rotation_tolerance * rotation_scale
        )
        & (_norm(linear_residual) <= plan.velocity_tolerance * linear_scale)
        & (_norm(angular_residual) <= plan.velocity_tolerance * angular_scale)
    )
    successful = finite & coincident
    return AttachmentKinematics(
        rigid_parent,
        soft_parent,
        rigid_attachment,
        soft_attachment,
        position_residual,
        rotation_residual,
        linear_residual,
        angular_residual,
        finite,
        coincident,
        successful,
        plan.attachment_id,
    )


def route_attachment_wrench(
    plan: RigidSoftAttachmentPlan,
    kinematics: AttachmentKinematics,
    command: AttachmentWrenchCommand,
    /,
) -> AttachmentWrenchRoute:
    """Route one true covector with exact action-reaction and virtual power."""
    if not isinstance(plan, RigidSoftAttachmentPlan):
        raise TypeError("plan must be RigidSoftAttachmentPlan.")
    if not isinstance(kinematics, AttachmentKinematics):
        raise TypeError("kinematics must be AttachmentKinematics.")
    if kinematics.attachment_id != plan.attachment_id:
        raise ValueError("Kinematics belongs to a different attachment plan.")
    if not isinstance(command, AttachmentWrenchCommand):
        raise TypeError("command must be AttachmentWrenchCommand.")
    if command.attachment_id != plan.attachment_id:
        raise ValueError("Wrench command belongs to a different attachment plan.")
    case_shape = _validate_frame(
        kinematics.rigid_attachment, plan.dimension, "rigid_attachment"
    )
    angular_dimension = 1 if plan.dimension == 2 else 3
    force = jnp.asarray(command.force)
    moment = jnp.asarray(command.moment)
    if force.shape != case_shape + (plan.dimension,):
        raise ValueError("Attachment force has the wrong shape.")
    if moment.shape != case_shape + (angular_dimension,):
        raise ValueError("Attachment moment has the wrong shape.")
    frame_dtype = np.dtype(kinematics.rigid_attachment.position.dtype)
    if np.dtype(force.dtype) != frame_dtype or np.dtype(moment.dtype) != frame_dtype:
        raise TypeError("Attachment wrench and frame arrays must share one dtype.")
    force_on_soft = ein.contract(
        "...ij,...j->...i", kinematics.rigid_attachment.rotation, force
    )
    if plan.dimension == 3:
        moment_on_soft = ein.contract(
            "...ij,...j->...i", kinematics.rigid_attachment.rotation, moment
        )
    else:
        moment_on_soft = moment
    force_on_rigid = -force_on_soft
    moment_on_rigid = -moment_on_soft
    rigid_lever = kinematics.rigid_attachment.position - kinematics.rigid_parent.position
    soft_lever = kinematics.soft_attachment.position - kinematics.soft_parent.position
    rigid_parent_moment = moment_on_rigid + _cross(
        plan.dimension, rigid_lever, force_on_rigid
    )
    soft_parent_moment = moment_on_soft + _cross(
        plan.dimension, soft_lever, force_on_soft
    )
    rigid_at_attachment = FrameWrench(
        force_on_rigid, moment_on_rigid, plan.rigid.frame_id
    )
    soft_at_attachment = FrameWrench(force_on_soft, moment_on_soft, plan.soft.endpoint_id)
    rigid_at_parent = FrameWrench(
        force_on_rigid, rigid_parent_moment, plan.rigid.frame_id
    )
    soft_at_parent = FrameWrench(force_on_soft, soft_parent_moment, plan.soft.endpoint_id)
    force_balance = force_on_rigid + force_on_soft
    moment_balance = (
        rigid_parent_moment
        + _cross(plan.dimension, kinematics.rigid_parent.position, force_on_rigid)
        + soft_parent_moment
        + _cross(plan.dimension, kinematics.soft_parent.position, force_on_soft)
    )
    rigid_power = _dot(force_on_rigid, kinematics.rigid_parent.linear_velocity) + _dot(
        rigid_parent_moment, kinematics.rigid_parent.angular_velocity
    )
    soft_power = _dot(force_on_soft, kinematics.soft_parent.linear_velocity) + _dot(
        soft_parent_moment, kinematics.soft_parent.angular_velocity
    )
    power_residual = rigid_power + soft_power
    finite = (
        jnp.all(jnp.isfinite(force_on_rigid), axis=-1)
        & jnp.all(jnp.isfinite(moment_on_rigid), axis=-1)
        & jnp.all(jnp.isfinite(rigid_parent_moment), axis=-1)
        & jnp.all(jnp.isfinite(soft_parent_moment), axis=-1)
        & jnp.isfinite(rigid_power)
        & jnp.isfinite(soft_power)
        & jnp.isfinite(power_residual)
    )
    force_scale = jnp.maximum(
        1.0, jnp.maximum(_norm(force_on_rigid), _norm(force_on_soft))
    )
    moment_scale = jnp.maximum(
        1.0, jnp.maximum(_norm(rigid_parent_moment), _norm(soft_parent_moment))
    )
    power_scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(rigid_power), jnp.abs(soft_power)))
    balanced = (_norm(force_balance) <= plan.balance_tolerance * force_scale) & (
        _norm(moment_balance) <= plan.balance_tolerance * moment_scale
    )
    power_conserving = jnp.abs(power_residual) <= plan.power_tolerance * power_scale
    successful = kinematics.successful & finite & balanced & power_conserving
    return AttachmentWrenchRoute(
        rigid_at_attachment,
        soft_at_attachment,
        rigid_at_parent,
        soft_at_parent,
        force_balance,
        moment_balance,
        rigid_power,
        soft_power,
        power_residual,
        finite,
        balanced,
        power_conserving,
        successful,
        plan.attachment_id,
    )


class HybridRigidSoftPlant(AbstractDiscretePlant, NonTrainableState):
    """Atomic fixed-topology composition of prepared rigid and soft plants."""

    rigid_plant: AbstractDiscretePlant
    soft_plant: AbstractDiscretePlant
    rigid_parameters: PlantParameters
    soft_parameters: PlantParameters
    rigid_port: AbstractHybridPlantPort
    soft_port: AbstractHybridPlantPort
    attachments: tuple[RigidSoftAttachmentPlan, ...]
    step_policy: SynchronizedStepPolicy
    parameters: PlantParameters
    command_template: HybridRigidSoftCommands
    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: HybridRigidSoftState
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        rigid_plant: AbstractDiscretePlant,
        rigid_parameters: PlantParameters,
        rigid_port: AbstractHybridPlantPort,
        rigid_command_template: PyTree[Any] | None,
        soft_plant: AbstractDiscretePlant,
        soft_parameters: PlantParameters,
        soft_port: AbstractHybridPlantPort,
        soft_command_template: PyTree[Any] | None,
        attachments: tuple[RigidSoftAttachmentPlan, ...],
        step_policy: SynchronizedStepPolicy,
        /,
    ):
        if not isinstance(rigid_plant, AbstractDiscretePlant):
            raise TypeError("rigid_plant must be AbstractDiscretePlant.")
        if not isinstance(soft_plant, AbstractDiscretePlant):
            raise TypeError("soft_plant must be AbstractDiscretePlant.")
        if not isinstance(rigid_port, AbstractHybridPlantPort):
            raise TypeError("rigid_port must be AbstractHybridPlantPort.")
        if not isinstance(soft_port, AbstractHybridPlantPort):
            raise TypeError("soft_port must be AbstractHybridPlantPort.")
        if not isinstance(step_policy, SynchronizedStepPolicy):
            raise TypeError("step_policy must be SynchronizedStepPolicy.")
        attachments_ = tuple(attachments)
        if not attachments_ or any(
            not isinstance(plan, RigidSoftAttachmentPlan) for plan in attachments_
        ):
            raise ValueError("attachments must contain at least one attachment plan.")
        dimensions = {plan.dimension for plan in attachments_}
        if len(dimensions) != 1:
            raise ValueError("All hybrid attachments must share one dimension.")
        attachment_ids = tuple(plan.attachment_id for plan in attachments_)
        if len(set(attachment_ids)) != len(attachment_ids):
            raise ValueError("Hybrid attachment plans must be unique.")
        if rigid_plant.state_schema.case_ndim != soft_plant.state_schema.case_ndim:
            raise ValueError("Rigid and soft plants must use the same case-axis policy.")
        case_ndim = rigid_plant.state_schema.case_ndim
        for plant, port, owner in (
            (rigid_plant, rigid_port, "rigid_port"),
            (soft_plant, soft_port, "soft_port"),
        ):
            expected_control = (
                None if plant.control_schema is None else plant.control_schema.schema_id
            )
            observed = (
                port.semantic_provenance_id,
                port.numeric_revision_id,
                port.state_schema_id,
                port.control_schema_id,
                port.execution_signature_id,
            )
            plant_identity = _plant_ids(plant)
            expected = (
                plant_identity[0],
                plant_identity[1],
                plant_identity[2],
                expected_control,
                plant_identity[3],
            )
            if observed != expected:
                raise ValueError(f"{owner} identities do not match its prepared plant.")
            if port.step_policy_id != step_policy.policy_id:
                raise ValueError(f"{owner} has an incompatible step policy.")
            if not isinstance(port.supports_external_wrenches, bool):
                raise TypeError(f"{owner}.supports_external_wrenches must be bool.")
            if not port.supports_external_wrenches:
                raise ValueError(f"{owner} cannot accept attachment-frame wrenches.")
            _identifier(port.topology_id, f"{owner}.topology_id")
            if not isinstance(port.frame_ids, tuple) or any(
                not isinstance(frame_id, str) or not frame_id.strip()
                for frame_id in port.frame_ids
            ):
                raise TypeError(
                    f"{owner}.frame_ids must be a tuple of non-empty strings."
                )
            if len(set(port.frame_ids)) != len(port.frame_ids):
                raise ValueError(f"{owner}.frame_ids must be unique.")
        for plan in attachments_:
            if plan.rigid.frame_id not in rigid_port.frame_ids:
                raise ValueError("An attachment names an unknown rigid frame.")
            if plan.soft.endpoint_id not in soft_port.frame_ids:
                raise ValueError("An attachment names an unknown soft endpoint.")
        _validate_bound_parameters(rigid_plant, rigid_parameters, "rigid_parameters")
        _validate_bound_parameters(soft_plant, soft_parameters, "soft_parameters")
        for plant, template, owner in (
            (rigid_plant, rigid_command_template, "rigid_command_template"),
            (soft_plant, soft_command_template, "soft_command_template"),
        ):
            if plant.control_schema is None:
                if template is not None:
                    raise ValueError(f"{owner} must be None for an uncontrolled plant.")
            else:
                if plant.control_schema.case_ndim != case_ndim:
                    raise ValueError(
                        "Controlled hybrid children must share the state case-axis policy."
                    )
                _validate_intrinsic_template(plant.control_schema, template, owner)
        topology_id = canonical_fingerprint(
            {
                "kind": "fixed-hybrid-rigid-soft-topology",
                "rigid_topology": rigid_port.topology_id,
                "soft_topology": soft_port.topology_id,
                "attachments": attachment_ids,
            }
        )
        reset_fallback = HybridRigidSoftState(
            rigid_plant.reset_fallback, soft_plant.reset_fallback, topology_id
        )
        probe_shape = (1,) * case_ndim
        state_probe = HybridRigidSoftState(
            _broadcast_intrinsic(
                rigid_plant.state_schema,
                rigid_plant.reset_fallback,
                probe_shape,
                "rigid reset_fallback",
            ),
            _broadcast_intrinsic(
                soft_plant.state_schema,
                soft_plant.reset_fallback,
                probe_shape,
                "soft reset_fallback",
            ),
            topology_id,
        )
        state_schema = ArrayPyTreeSchema.from_tree(state_probe, case_ndim=case_ndim)
        dtype = np.result_type(
            *(np.dtype(plan.rigid.local_position.dtype) for plan in attachments_),
            *(np.dtype(plan.soft.local_position.dtype) for plan in attachments_),
        )
        angular_dimension = 1 if next(iter(dimensions)) == 2 else 3
        wrench_commands = tuple(
            AttachmentWrenchCommand(
                jnp.zeros((plan.dimension,), dtype=dtype),
                jnp.zeros((angular_dimension,), dtype=dtype),
                plan.attachment_id,
            )
            for plan in attachments_
        )
        command_template = HybridRigidSoftCommands(
            rigid_command_template,
            soft_command_template,
            wrench_commands,
            topology_id,
        )
        command_probe = jax.tree_util.tree_map(
            lambda value: jnp.broadcast_to(value, probe_shape + value.shape),
            command_template,
        )
        control_schema = ArrayPyTreeSchema.from_tree(command_probe, case_ndim=case_ndim)
        parameter_values = HybridRigidSoftParameterValues(
            rigid_parameters.values, soft_parameters.values, topology_id
        )
        parameter_schema = ArrayPyTreeSchema.from_tree(parameter_values, case_ndim=0)
        rigid_port_identity = strict_module_payload(rigid_port)
        soft_port_identity = strict_module_payload(soft_port)
        semantic = SemanticProvenance(
            {
                "kind": "hybrid-rigid-soft-plant",
                "rigid_plant": rigid_plant.semantic_provenance.semantic_id,
                "soft_plant": soft_plant.semantic_provenance.semantic_id,
                "rigid_port": rigid_port_identity["semantic_content_id"],
                "soft_port": soft_port_identity["semantic_content_id"],
                "topology": topology_id,
                "step_policy": step_policy.policy_id,
                "state_schema": state_schema.content_id,
                "control_schema": control_schema.content_id,
                "parameter_schema": parameter_schema.content_id,
            }
        )
        revision = NumericRevision(
            semantic,
            {
                "rigid_revision": rigid_plant.numeric_revision.revision_id,
                "soft_revision": soft_plant.numeric_revision.revision_id,
                "rigid_port": rigid_port_identity["numeric_content_id"],
                "soft_port": soft_port_identity["numeric_content_id"],
            },
        )
        shapes = tuple(
            (f"state:{leaf.path}", leaf.shape) for leaf in state_schema.leaves
        ) + tuple((f"control:{leaf.path}", leaf.shape) for leaf in control_schema.leaves)
        dtypes = tuple(
            (f"state:{leaf.path}", leaf.dtype) for leaf in state_schema.leaves
        ) + tuple((f"control:{leaf.path}", leaf.dtype) for leaf in control_schema.leaves)
        signature = ExecutableSignature(
            shapes=shapes,
            dtypes=dtypes,
            space_ids={
                "state": state_schema.schema_id,
                "control": control_schema.schema_id,
                "parameters": parameter_schema.schema_id,
            },
            topology_ids={
                "hybrid": topology_id,
                "rigid": rigid_port.topology_id,
                "soft": soft_port.topology_id,
            },
            capacities={"attachments": len(attachments_)},
            algorithm_facts={
                "step_policy": step_policy.policy_id,
                "rigid_executable": rigid_plant.execution_signature.signature_id,
                "soft_executable": soft_plant.execution_signature.signature_id,
            },
        )
        fallback_kinematics = self._attachment_kinematics_from(
            rigid_port,
            soft_port,
            attachments_,
            rigid_plant.reset_fallback,
            soft_plant.reset_fallback,
        )
        if not all(
            bool(np.asarray(jnp.all(item.successful))) for item in fallback_kinematics
        ):
            raise ValueError(
                "Prepared child reset fallbacks do not satisfy the attachments."
            )
        self.rigid_plant = rigid_plant
        self.soft_plant = soft_plant
        self.rigid_parameters = rigid_parameters
        self.soft_parameters = soft_parameters
        self.rigid_port = rigid_port
        self.soft_port = soft_port
        self.attachments = attachments_
        self.step_policy = step_policy
        self.parameters = PlantParameters(
            parameter_values, parameter_schema.schema_id, revision
        )
        self.command_template = command_template
        self.state_schema = state_schema
        self.control_schema = control_schema
        self.parameter_schema = parameter_schema
        self.reset_fallback = reset_fallback
        self.semantic_provenance = semantic
        self.numeric_revision = revision
        self.execution_signature = signature
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True
        self.topology_id = topology_id
        self.dimension = next(iter(dimensions))

    @staticmethod
    def _attachment_kinematics_from(
        rigid_port: AbstractHybridPlantPort,
        soft_port: AbstractHybridPlantPort,
        attachments: tuple[RigidSoftAttachmentPlan, ...],
        rigid_payload: PyTree[Any],
        soft_payload: PyTree[Any],
        /,
    ) -> tuple[AttachmentKinematics, ...]:
        return tuple(
            evaluate_attachment_kinematics(
                plan,
                rigid_port.frame_state(rigid_payload, plan.rigid.frame_id),
                soft_port.frame_state(soft_payload, plan.soft.endpoint_id),
            )
            for plan in attachments
        )

    def _attachment_kinematics(
        self, payload: HybridRigidSoftState, /
    ) -> tuple[AttachmentKinematics, ...]:
        if not isinstance(payload, HybridRigidSoftState):
            raise TypeError("Hybrid payload must be HybridRigidSoftState.")
        if payload.topology_id != self.topology_id:
            raise ValueError(
                "Hybrid payload topology identity does not match this plant."
            )
        return self._attachment_kinematics_from(
            self.rigid_port,
            self.soft_port,
            self.attachments,
            payload.rigid,
            payload.soft,
        )

    def attachment_kinematics(
        self, state: PlantRuntimeState, /
    ) -> tuple[AttachmentKinematics, ...]:
        """Evaluate attachments after exact prepared-state identity validation."""
        if not isinstance(state, PlantRuntimeState):
            raise TypeError("state must be PlantRuntimeState.")
        if (
            state.semantic_provenance_id,
            state.numeric_revision_id,
            state.state_schema_id,
            state.execution_signature_id,
        ) != _plant_ids(self):
            raise ValueError("Runtime state belongs to a different hybrid plant.")
        self.state_schema.validate(state.payload)
        return self._attachment_kinematics(state.payload)

    def commands(
        self,
        rigid: PyTree[Any] | None,
        soft: PyTree[Any] | None,
        attachment_wrenches: tuple[AttachmentWrenchCommand, ...],
        /,
    ) -> HybridRigidSoftCommands:
        """Bind child commands and the exact fixed attachment tuple to this plant."""
        commands = HybridRigidSoftCommands(
            rigid, soft, tuple(attachment_wrenches), self.topology_id
        )
        self.control_schema.validate(commands)
        expected = tuple(plan.attachment_id for plan in self.attachments)
        observed = tuple(item.attachment_id for item in commands.attachment_wrenches)
        if observed != expected:
            raise ValueError("Attachment wrench tuple does not match the fixed topology.")
        return commands

    def propose_reset(
        self,
        keys: Array,
        parameters: HybridRigidSoftParameterValues,
        /,
        *,
        case_shape: tuple[int, ...],
        initial_time: Array,
    ) -> PlantProposal:
        if not isinstance(parameters, HybridRigidSoftParameterValues):
            raise TypeError("Hybrid parameters have the wrong PyTree type.")
        if parameters.topology_id != self.topology_id:
            raise ValueError("Hybrid parameter topology identity does not match.")
        rigid_key, soft_key = _branch_keys(keys, case_shape)
        rigid = self.rigid_plant.reset(
            rigid_key,
            _parameters(self.rigid_plant, parameters.rigid),
            case_shape=case_shape,
            initial_time=initial_time,
        )
        soft = self.soft_plant.reset(
            soft_key,
            _parameters(self.soft_plant, parameters.soft),
            case_shape=case_shape,
            initial_time=initial_time,
        )
        candidate = HybridRigidSoftState(
            rigid.candidate_state.payload,
            soft.candidate_state.payload,
            self.topology_id,
        )
        accepted = HybridRigidSoftState(
            rigid.accepted_state.payload,
            soft.accepted_state.payload,
            self.topology_id,
        )
        candidate_kinematics = self._attachment_kinematics(candidate)
        accepted_kinematics = self._attachment_kinematics(accepted)
        accepted_valid = _all_masks(
            tuple(item.successful for item in accepted_kinematics), case_shape
        )
        topology = (
            _mask(
                self.rigid_port.topology_unchanged(
                    self.rigid_plant.reset_fallback, candidate.rigid
                ),
                case_shape,
                "rigid reset candidate topology",
            )
            & _mask(
                self.rigid_port.topology_unchanged(
                    self.rigid_plant.reset_fallback, accepted.rigid
                ),
                case_shape,
                "rigid reset accepted topology",
            )
            & _mask(
                self.soft_port.topology_unchanged(
                    self.soft_plant.reset_fallback, candidate.soft
                ),
                case_shape,
                "soft reset candidate topology",
            )
            & _mask(
                self.soft_port.topology_unchanged(
                    self.soft_plant.reset_fallback, accepted.soft
                ),
                case_shape,
                "soft reset accepted topology",
            )
        )
        attempted = rigid.attempted & soft.attempted
        successful = rigid.successful & soft.successful & accepted_valid & topology
        status = jnp.where(
            ~rigid.successful,
            int(HybridRigidSoftStatus.RIGID_RESET_FAILED),
            jnp.where(
                ~soft.successful,
                int(HybridRigidSoftStatus.SOFT_RESET_FAILED),
                jnp.where(
                    ~topology,
                    int(HybridRigidSoftStatus.TOPOLOGY_CHANGED),
                    jnp.where(
                        ~accepted_valid,
                        int(HybridRigidSoftStatus.INVALID_ACCEPTED_ATTACHMENT),
                        int(HybridRigidSoftStatus.SUCCESS),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        backend_status = jnp.where(
            ~rigid.successful, rigid.backend_status, soft.backend_status
        ).astype(jnp.int32)
        evidence = HybridResetEvidence(
            rigid,
            soft,
            candidate_kinematics,
            accepted_kinematics,
            topology,
            successful,
            status,
            self.topology_id,
        )
        return PlantProposal(
            candidate,
            accepted,
            attempted,
            successful,
            status,
            backend_status,
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
        if not isinstance(source, HybridRigidSoftState):
            raise TypeError("Hybrid source has the wrong PyTree type.")
        if not isinstance(commands, HybridRigidSoftCommands):
            raise TypeError("Hybrid commands have the wrong PyTree type.")
        if not isinstance(parameters, HybridRigidSoftParameterValues):
            raise TypeError("Hybrid parameters have the wrong PyTree type.")
        if source.topology_id != self.topology_id:
            raise ValueError("Hybrid source topology identity does not match.")
        if commands.topology_id != self.topology_id:
            raise ValueError("Hybrid command topology identity does not match.")
        if parameters.topology_id != self.topology_id:
            raise ValueError("Hybrid parameter topology identity does not match.")
        expected_attachment_ids = tuple(plan.attachment_id for plan in self.attachments)
        observed_attachment_ids = tuple(
            item.attachment_id for item in commands.attachment_wrenches
        )
        if observed_attachment_ids != expected_attachment_ids:
            raise ValueError("Attachment wrench tuple does not match the fixed topology.")
        case_shape = self.state_schema.validate(source)
        source_kinematics = self._attachment_kinematics(source)
        routes = tuple(
            route_attachment_wrench(plan, kinematics, command)
            for plan, kinematics, command in zip(
                self.attachments,
                source_kinematics,
                commands.attachment_wrenches,
                strict=True,
            )
        )
        rigid_wrenches = tuple(route.rigid_at_parent for route in routes)
        soft_wrenches = tuple(route.soft_at_parent for route in routes)
        rigid_commands = self.rigid_port.apply_frame_wrenches(
            source.rigid, commands.rigid, rigid_wrenches
        )
        soft_commands = self.soft_port.apply_frame_wrenches(
            source.soft, commands.soft, soft_wrenches
        )
        rigid_key, soft_key = _branch_keys(keys, case_shape)
        rigid = self.rigid_plant.step(
            context,
            _runtime_state(self.rigid_plant, source.rigid, context, rigid_key),
            rigid_commands,
            _parameters(self.rigid_plant, parameters.rigid),
        )
        soft = self.soft_plant.step(
            context,
            _runtime_state(self.soft_plant, source.soft, context, soft_key),
            soft_commands,
            _parameters(self.soft_plant, parameters.soft),
        )
        candidate = HybridRigidSoftState(
            rigid.candidate_state.payload,
            soft.candidate_state.payload,
            self.topology_id,
        )
        accepted = HybridRigidSoftState(
            rigid.accepted_state.payload,
            soft.accepted_state.payload,
            self.topology_id,
        )
        candidate_kinematics = self._attachment_kinematics(candidate)
        accepted_kinematics = self._attachment_kinematics(accepted)
        source_valid = _all_masks(
            tuple(item.successful for item in source_kinematics), case_shape
        )
        route_valid = _all_masks(tuple(item.successful for item in routes), case_shape)
        accepted_valid = _all_masks(
            tuple(item.successful for item in accepted_kinematics), case_shape
        )
        topology = (
            _mask(
                self.rigid_port.topology_unchanged(
                    source.rigid, rigid.candidate_state.payload
                ),
                case_shape,
                "rigid candidate topology",
            )
            & _mask(
                self.rigid_port.topology_unchanged(
                    source.rigid, rigid.accepted_state.payload
                ),
                case_shape,
                "rigid accepted topology",
            )
            & _mask(
                self.soft_port.topology_unchanged(
                    source.soft, soft.candidate_state.payload
                ),
                case_shape,
                "soft candidate topology",
            )
            & _mask(
                self.soft_port.topology_unchanged(
                    source.soft, soft.accepted_state.payload
                ),
                case_shape,
                "soft accepted topology",
            )
        )
        duration_valid = _mask(
            self.step_policy.duration_valid(context), case_shape, "duration_valid"
        )
        attempted = rigid.attempted & soft.attempted
        successful = (
            attempted
            & duration_valid
            & source_valid
            & route_valid
            & rigid.successful
            & soft.successful
            & topology
            & accepted_valid
        )
        status = jnp.where(
            ~duration_valid,
            int(HybridRigidSoftStatus.INCOMPATIBLE_DURATION),
            jnp.where(
                ~source_valid,
                int(HybridRigidSoftStatus.INVALID_SOURCE_ATTACHMENT),
                jnp.where(
                    ~route_valid,
                    int(HybridRigidSoftStatus.INVALID_WRENCH_ROUTE),
                    jnp.where(
                        ~rigid.successful,
                        int(HybridRigidSoftStatus.RIGID_STEP_FAILED),
                        jnp.where(
                            ~soft.successful,
                            int(HybridRigidSoftStatus.SOFT_STEP_FAILED),
                            jnp.where(
                                ~topology,
                                int(HybridRigidSoftStatus.TOPOLOGY_CHANGED),
                                jnp.where(
                                    ~accepted_valid,
                                    int(
                                        HybridRigidSoftStatus.INVALID_ACCEPTED_ATTACHMENT
                                    ),
                                    int(HybridRigidSoftStatus.SUCCESS),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        backend_status = jnp.where(
            ~rigid.successful, rigid.backend_status, soft.backend_status
        ).astype(jnp.int32)
        evidence = HybridStepEvidence(
            rigid,
            soft,
            source_kinematics,
            candidate_kinematics,
            accepted_kinematics,
            routes,
            duration_valid,
            topology,
            successful,
            status,
            self.topology_id,
        )
        return PlantProposal(
            candidate,
            accepted,
            attempted,
            successful,
            status,
            backend_status,
            evidence,
        )


__all__ = [
    "AbstractHybridPlantPort",
    "AttachmentFrameState",
    "AttachmentKinematics",
    "AttachmentWrenchCommand",
    "AttachmentWrenchRoute",
    "FrameWrench",
    "FloatingReducedRodPlantPort",
    "HybridResetEvidence",
    "HybridRigidSoftCommands",
    "HybridRigidSoftParameterValues",
    "HybridRigidSoftPlant",
    "HybridRigidSoftState",
    "HybridRigidSoftStatus",
    "HybridStepEvidence",
    "PreparedReducedRodPlantPort",
    "RigidFrameAttachmentPlan",
    "RigidSoftAttachmentPlan",
    "SoftEndpointAttachmentPlan",
    "SynchronizedStepPolicy",
    "TendonDrivenRodPlantPort",
    "evaluate_attachment_kinematics",
    "route_attachment_wrench",
    "transform_attachment_frame",
]
