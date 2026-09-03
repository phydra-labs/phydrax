#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, FunctionLinearOperator
from ...metrix._state_geometry import AbstractStateGeometry
from ._rigid_body import (
    _quaternion_conjugate,
    _quaternion_increment,
    _quaternion_multiply,
    PreparedRigidBodySet,
    quaternion_rotation_matrix,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ._rigid_joints import PreparedRigidJointGraph, RigidJointKind


if TYPE_CHECKING:
    from ...dynamics._layout import InputLayout, StateLayout


def _integer_vector(value: ArrayLike, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 array.")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer IDs.")
    return array.astype(np.int64, copy=False)


def _principal_angle(value: Array, /) -> Array:
    return jnp.arctan2(jnp.sin(value), jnp.cos(value))


def _normalize_preserving_sign(value: Array, /) -> Array:
    norm = jnp.sqrt(jnp.sum(value * value, axis=-1, keepdims=True))
    return value / jnp.maximum(norm, jnp.finfo(value.dtype).tiny)


def _configuration_increment(
    configuration: Array,
    velocity: Array,
    step_size: Array,
    hinge_dof_indices: tuple[int, ...],
    /,
) -> Array:
    point = configuration + step_size * velocity
    if hinge_dof_indices:
        indices = jnp.asarray(hinge_dof_indices, dtype=jnp.int32)
        point = point.at[indices].set(_principal_angle(point[indices]))
    return point


def _configuration_delta(
    reference: Array,
    point: Array,
    hinge_dof_indices: tuple[int, ...],
    /,
) -> Array:
    difference = point - reference
    if hinge_dof_indices:
        indices = jnp.asarray(hinge_dof_indices, dtype=jnp.int32)
        difference = difference.at[indices].set(_principal_angle(difference[indices]))
    return difference


class ReducedArticulationState(StrictModule):
    """Canonical generalized configuration and velocity before rank-1 packing."""

    configuration: Array
    velocity: Array


class ArticulationKinematics(StrictModule):
    """Full-capacity rigid-body kinematics and homogeneous transforms."""

    bodies: RigidBodyKinematics
    body_transforms: Array
    finite: Array
    successful: Array


class ArticulationDualityEvidence(StrictModule):
    """Measured equality of body-load and generalized-load power."""

    body_power: Array
    generalized_power: Array
    residual: Array
    scale: Array
    finite: Array
    valid: Array
    articulation_id: str = eqx.field(static=True)


class _ReducedArticulationStateGeometry(AbstractStateGeometry):
    state_size: int = eqx.field(static=True, default=0)
    nq: int = eqx.field(static=True, default=0)
    hinge_dof_indices: tuple[int, ...] = eqx.field(static=True, default=())
    geometry_id: str = eqx.field(static=True, default="")
    retraction_method: str = "reduced-articulation-scalar-joints"
    trivial: bool = False
    supports_exact_pullback: bool = True
    supports_commutator_free: bool = True

    def __init__(
        self,
        state_size: int,
        nq: int,
        hinge_dof_indices: tuple[int, ...],
        geometry_id: str,
        /,
    ):
        self.state_size = int(state_size)
        self.nq = int(nq)
        self.hinge_dof_indices = tuple(int(index) for index in hinge_dof_indices)
        self.geometry_id = str(geometry_id)

    def _state(self, value: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(value)
        if array.shape != (self.state_size,):
            raise ValueError(
                f"{name} must have packed articulation shape {(self.state_size,)}."
            )
        return array

    def contains(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != (self.state_size,):
            return jnp.asarray(False)
        return jnp.all(jnp.isfinite(value))

    def project_tangent(
        self, state: ArrayLike, vector: ArrayLike, /
    ) -> Array:
        self._state(state, "State")
        return self._state(vector, "Tangent")

    def to_local(self, state: ArrayLike, tangent: ArrayLike, /) -> Array:
        return self.project_tangent(state, tangent)

    def from_local(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        return self.project_tangent(state, local_tangent)

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        base = self._state(state, "State")
        local = self._state(local_tangent, "Local tangent")
        configuration = _configuration_increment(
            base[: self.nq],
            local[: self.nq],
            jnp.asarray(1.0, dtype=base.dtype),
            self.hinge_dof_indices,
        )
        return jnp.concatenate(
            (configuration, base[self.nq :] + local[self.nq :])
        )

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        base = self._state(state, "State")
        target = self._state(point, "Point")
        configuration = _configuration_delta(
            base[: self.nq], target[: self.nq], self.hinge_dof_indices
        )
        return jnp.concatenate((configuration, target[self.nq :] - base[self.nq :]))

    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        self._state(state, "State")
        self._state(local_tangent, "Local tangent")
        return self._state(tangent, "Retraction tangent")


class ReducedArticulationPlan(StrictModule, NonTrainableState):
    """An oriented rooted tree selecting existing rigid-joint graph edges."""

    joint_ids: Array
    parent_body_ids: Array
    child_body_ids: Array
    root_body_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        root_body_id: int,
        joint_ids: ArrayLike,
        parent_body_ids: ArrayLike,
        child_body_ids: ArrayLike,
        /,
        *,
        plan_id: str | None = None,
    ):
        if isinstance(root_body_id, bool) or not isinstance(root_body_id, Integral):
            raise TypeError("root_body_id must be an integer body ID.")
        joints = _integer_vector(joint_ids, "joint_ids")
        parents = _integer_vector(parent_body_ids, "parent_body_ids")
        children = _integer_vector(child_body_ids, "child_body_ids")
        if parents.shape != joints.shape or children.shape != joints.shape:
            raise ValueError(
                "joint_ids, parent_body_ids, and child_body_ids must have "
                "matching shapes."
            )
        if np.unique(joints).size != joints.size:
            raise ValueError("Reduced-articulation joint IDs must be unique.")
        if np.any(parents == children):
            raise ValueError("Every articulation edge must join distinct bodies.")
        root = int(root_body_id)
        generated = canonical_fingerprint(
            {
                "kind": "reduced-articulation-plan",
                "root_body_id": root,
                "values": array_tree_fingerprint(
                    {
                        "joint_ids": joints,
                        "parent_body_ids": parents,
                        "child_body_ids": children,
                    }
                ),
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.joint_ids = jnp.asarray(joints)
        self.parent_body_ids = jnp.asarray(parents)
        self.child_body_ids = jnp.asarray(children)
        self.root_body_id = root
        self.plan_id = identifier

    @property
    def edge_count(self) -> int:
        return int(self.joint_ids.shape[0])

    def prepare(
        self,
        graph: PreparedRigidJointGraph,
        reference: RigidBodyKinematics,
        /,
    ) -> PreparedReducedArticulation:
        return PreparedReducedArticulation(self, graph, reference)


class PreparedReducedArticulation(StrictModule, NonTrainableState):
    """Prepared fixed-base scalar-joint tree with pure JAX runtime actions."""

    plan: ReducedArticulationPlan
    graph: PreparedRigidJointGraph
    reference_position: Array
    reference_orientation: Array
    body_ids: Array
    body_indices: Array
    joint_ids: Array
    joint_kinds: Array
    parent_indices: Array
    child_indices: Array
    dof_body_indices: Array
    dof_joint_indices: Array
    configuration_offsets: Array
    velocity_offsets: Array
    parent_reference_translation: Array
    parent_reference_rotation: Array
    parent_reference_orientation: Array
    parent_axes: Array
    parent_anchors: Array
    state_layout: StateLayout | None
    input_layout: InputLayout | None
    root_index: int = eqx.field(static=True)
    nq: int = eqx.field(static=True)
    nv: int = eqx.field(static=True)
    configuration_slice: slice = eqx.field(static=True)
    velocity_slice: slice = eqx.field(static=True)
    joint_configuration_slices: tuple[slice, ...] = eqx.field(static=True)
    joint_velocity_slices: tuple[slice, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    _parent_order: tuple[int, ...] = eqx.field(static=True)
    _child_order: tuple[int, ...] = eqx.field(static=True)
    _kind_order: tuple[int, ...] = eqx.field(static=True)
    _edge_dof_order: tuple[int, ...] = eqx.field(static=True)
    _hinge_dof_indices: tuple[int, ...] = eqx.field(static=True)
    _body_id_order: tuple[int, ...] = eqx.field(static=True)
    _body_index_order: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        plan: ReducedArticulationPlan,
        graph: PreparedRigidJointGraph,
        reference: RigidBodyKinematics,
        /,
    ):
        if not isinstance(plan, ReducedArticulationPlan):
            raise TypeError("plan must be a ReducedArticulationPlan.")
        if not isinstance(graph, PreparedRigidJointGraph):
            raise TypeError("graph must be a PreparedRigidJointGraph.")
        if not isinstance(reference, RigidBodyKinematics):
            raise TypeError("reference must be RigidBodyKinematics.")
        bodies = graph.bodies
        if not isinstance(bodies, PreparedRigidBodySet):
            raise TypeError("graph.bodies must be a PreparedRigidBodySet.")
        if bodies.ambient_dimension != 3:
            raise ValueError(
                "Reduced articulation currently requires three dimensions."
            )
        expected_vector_shape = (bodies.capacity, 3)
        if (
            reference.position.shape != expected_vector_shape
            or reference.velocity.shape != expected_vector_shape
            or reference.orientation.shape != (bodies.capacity, 4)
            or reference.angular_velocity.shape != expected_vector_shape
        ):
            raise ValueError(
                "Reference rigid-body kinematics have incompatible shapes."
            )
        reference_leaves = (
            reference.position,
            reference.velocity,
            reference.orientation,
            reference.angular_velocity,
        )
        if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in reference_leaves):
            raise ValueError("Reference rigid-body kinematics must be finite.")
        orientation_norm = np.linalg.norm(np.asarray(reference.orientation), axis=-1)
        if not np.allclose(orientation_norm, 1.0, rtol=0.0, atol=1.0e-8):
            raise ValueError("Reference rigid-body orientations must have unit norm.")
        expected_graph_id = canonical_fingerprint(
            {
                "kind": "prepared-rigid-joint-graph",
                "plan": graph.plan.plan_id,
                "bodies": bodies.prepared_id,
                "reference": array_tree_fingerprint(
                    {
                        "position": np.asarray(reference.position),
                        "orientation": np.asarray(reference.orientation),
                    }
                ),
            }
        )
        if expected_graph_id != graph.prepared_id:
            raise ValueError("reference must be the pose used to prepare graph.")

        body_id_values = np.asarray(bodies.particles.particle_ids, dtype=np.int64)
        active = np.asarray(bodies.particles.active_mask, dtype=bool)
        fixed = np.asarray(bodies.fixed_mask, dtype=bool)
        body_index_by_id = {
            int(identifier): index for index, identifier in enumerate(body_id_values)
        }
        root_index = body_index_by_id.get(plan.root_body_id)
        if root_index is None or not active[root_index]:
            raise ValueError("Root body ID is absent from active rigid-body support.")
        if not fixed[root_index]:
            raise ValueError("Reduced articulation requires a fixed root body.")

        parent_ids = np.asarray(plan.parent_body_ids, dtype=np.int64)
        child_ids = np.asarray(plan.child_body_ids, dtype=np.int64)
        joint_ids = np.asarray(plan.joint_ids, dtype=np.int64)
        missing_body_ids = sorted(
            {
                int(identifier)
                for identifier in np.concatenate((parent_ids, child_ids))
                if int(identifier) not in body_index_by_id
                or not active[body_index_by_id[int(identifier)]]
            }
        )
        if missing_body_ids:
            raise ValueError(
                "Articulation body ID is absent from active rigid-body support."
            )
        parent_plan_indices = np.asarray(
            [body_index_by_id[int(identifier)] for identifier in parent_ids],
            dtype=np.int32,
        )
        child_plan_indices = np.asarray(
            [body_index_by_id[int(identifier)] for identifier in child_ids],
            dtype=np.int32,
        )

        joint_by_id: dict[int, tuple[int, int, int, int]] = {}
        joint_groups = (
            (
                graph.plan.fixed,
                RigidJointKind.FIXED,
                graph.fixed_left,
                graph.fixed_right,
            ),
            (
                graph.plan.ball,
                RigidJointKind.BALL,
                graph.ball_left,
                graph.ball_right,
            ),
            (
                graph.plan.hinge,
                RigidJointKind.HINGE,
                graph.hinge_left,
                graph.hinge_right,
            ),
            (
                graph.plan.prismatic,
                RigidJointKind.PRISMATIC,
                graph.prismatic_left,
                graph.prismatic_right,
            ),
            (
                graph.plan.distance,
                RigidJointKind.DISTANCE,
                graph.distance_left,
                graph.distance_right,
            ),
        )
        for joint_plan, kind, left_indices, right_indices in joint_groups:
            if joint_plan is None:
                continue
            identifiers = np.asarray(joint_plan.joint_ids, dtype=np.int64)
            left = np.asarray(left_indices, dtype=np.int32)
            right = np.asarray(right_indices, dtype=np.int32)
            for row, identifier in enumerate(identifiers):
                joint_by_id[int(identifier)] = (
                    int(kind),
                    row,
                    int(left[row]),
                    int(right[row]),
                )

        kinds_plan: list[int] = []
        rows_plan: list[int] = []
        for edge, identifier in enumerate(joint_ids):
            resolved = joint_by_id.get(int(identifier))
            if resolved is None:
                raise ValueError("Articulation joint ID is absent from prepared graph.")
            kind, row, left, right = resolved
            if kind in (int(RigidJointKind.BALL), int(RigidJointKind.DISTANCE)):
                raise ValueError(
                    "Reduced articulation supports only fixed, hinge, and "
                    "prismatic joints."
                )
            if left != int(parent_plan_indices[edge]) or right != int(
                child_plan_indices[edge]
            ):
                raise ValueError(
                    "Articulation edges must preserve each existing joint's "
                    "left-to-right orientation."
                )
            kinds_plan.append(kind)
            rows_plan.append(row)

        unconsumed_joint_ids = sorted(set(joint_by_id) - set(joint_ids.tolist()))
        if unconsumed_joint_ids:
            raise ValueError(
                "Prepared rigid-joint graph contains joints that are neither "
                f"selected tree edges nor represented closures: {unconsumed_joint_ids!r}."
            )

        active_indices = np.flatnonzero(active).astype(np.int32)
        declared_indices = set(parent_plan_indices.tolist()) | set(
            child_plan_indices.tolist()
        ) | {root_index}
        if declared_indices != set(active_indices.tolist()):
            raise ValueError(
                "Articulation tree is disconnected from active rigid-body support."
            )
        if plan.edge_count != active_indices.size - 1:
            raise ValueError(
                "A connected articulation tree must have one edge per non-root body."
            )
        if np.unique(child_plan_indices).size != child_plan_indices.size:
            raise ValueError(
                "Every non-root articulation body must have exactly one parent."
            )
        if root_index in set(child_plan_indices.tolist()):
            raise ValueError("The articulation root cannot be a child edge endpoint.")
        nonroot_fixed = fixed.copy()
        nonroot_fixed[root_index] = False
        if np.any(nonroot_fixed & active):
            raise ValueError("Only the articulation root body may be fixed.")

        remaining = list(range(plan.edge_count))
        reached = {root_index}
        body_order = [root_index]
        edge_order: list[int] = []
        while remaining:
            ready = [
                edge
                for edge in remaining
                if int(parent_plan_indices[edge]) in reached
                and int(child_plan_indices[edge]) not in reached
            ]
            if not ready:
                raise ValueError(
                    "Articulation edges contain a cycle or disconnected component."
                )
            for edge in ready:
                edge_order.append(edge)
                child = int(child_plan_indices[edge])
                reached.add(child)
                body_order.append(child)
                remaining.remove(edge)
        if reached != set(active_indices.tolist()):
            raise ValueError(
                "Articulation edges do not form one connected rooted tree."
            )

        parent_indices = parent_plan_indices[edge_order]
        child_indices = child_plan_indices[edge_order]
        ordered_joint_ids = joint_ids[edge_order]
        joint_kinds = np.asarray(
            [kinds_plan[edge] for edge in edge_order], dtype=np.int32
        )
        ordered_rows = [rows_plan[edge] for edge in edge_order]

        position_host = np.asarray(reference.position)
        orientation = reference.orientation
        rotation_host = np.asarray(quaternion_rotation_matrix(orientation))
        reference_translation = np.empty(
            (plan.edge_count, 3), dtype=position_host.dtype
        )
        reference_rotation = np.empty(
            (plan.edge_count, 3, 3), dtype=position_host.dtype
        )
        reference_orientation = np.empty(
            (plan.edge_count, 4), dtype=position_host.dtype
        )
        axes = np.zeros((plan.edge_count, 3), dtype=position_host.dtype)
        anchors = np.zeros((plan.edge_count, 3), dtype=position_host.dtype)
        for ordered_edge, (source_edge, row) in enumerate(
            zip(edge_order, ordered_rows)
        ):
            parent = int(parent_indices[ordered_edge])
            child = int(child_indices[ordered_edge])
            reference_translation[ordered_edge] = (
                rotation_host[parent].T @ (position_host[child] - position_host[parent])
            )
            reference_rotation[ordered_edge] = (
                rotation_host[parent].T @ rotation_host[child]
            )
            relative_orientation = _quaternion_multiply(
                _quaternion_conjugate(orientation[parent]), orientation[child]
            )
            reference_orientation[ordered_edge] = np.asarray(relative_orientation)
            kind = kinds_plan[source_edge]
            if kind == int(RigidJointKind.HINGE):
                axes[ordered_edge] = np.asarray(graph.hinge_axis_left[row])
                anchors[ordered_edge] = np.asarray(graph.hinge_anchor_left[row])
            elif kind == int(RigidJointKind.PRISMATIC):
                axes[ordered_edge] = np.asarray(graph.prismatic_axis_left[row])
                anchors[ordered_edge] = np.asarray(graph.prismatic_anchor_left[row])

        configuration_offsets = [0]
        velocity_offsets = [0]
        configuration_slices: list[slice] = []
        velocity_slices: list[slice] = []
        edge_dofs: list[int] = []
        hinge_dofs: list[int] = []
        dof_bodies: list[int] = []
        dof_joints: list[int] = []
        nq = 0
        nv = 0
        for edge, kind in enumerate(joint_kinds):
            width = 0 if int(kind) == int(RigidJointKind.FIXED) else 1
            configuration_slices.append(slice(nq, nq + width))
            velocity_slices.append(slice(nv, nv + width))
            edge_dofs.append(-1 if width == 0 else nq)
            if width:
                if int(kind) == int(RigidJointKind.HINGE):
                    hinge_dofs.append(nq)
                dof_bodies.append(int(child_indices[edge]))
                dof_joints.append(edge)
            nq += width
            nv += width
            configuration_offsets.append(nq)
            velocity_offsets.append(nv)

        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-reduced-articulation",
                "plan": plan.plan_id,
                "graph": graph.prepared_id,
                "topology": array_tree_fingerprint(
                    {
                        "body_ids": body_id_values[np.asarray(body_order)],
                        "joint_ids": ordered_joint_ids,
                        "joint_kinds": joint_kinds,
                        "parent_indices": parent_indices,
                        "child_indices": child_indices,
                        "parent_reference_translation": reference_translation,
                        "parent_reference_rotation": reference_rotation,
                        "parent_axes": axes,
                        "parent_anchors": anchors,
                    }
                ),
            }
        )
        from ...dynamics._layout import InputLayout, StateLayout

        state_size = nq + nv
        if state_size:
            state_components = tuple(
                [f"q:{int(ordered_joint_ids[edge])}" for edge in dof_joints]
                + [f"v:{int(ordered_joint_ids[edge])}" for edge in dof_joints]
            )
            geometry = _ReducedArticulationStateGeometry(
                state_size,
                nq,
                tuple(hinge_dofs),
                f"state-geometry:reduced-articulation:{prepared_id}",
            )
            state_layout: StateLayout | None = StateLayout(
                (state_size,),
                axes=("articulation_state",),
                component_names=state_components,
                geometry=geometry,
                layout_id=f"state-layout:reduced-articulation:{prepared_id}",
            )
            input_layout: InputLayout | None = InputLayout(
                (nv,),
                axes=("generalized_input",),
                component_names=tuple(
                    f"tau:{int(ordered_joint_ids[edge])}" for edge in dof_joints
                ),
                roles="control",
                layout_id=f"input-layout:reduced-articulation:{prepared_id}",
            )
        else:
            state_layout = None
            input_layout = None

        self.plan = plan
        self.graph = graph
        self.reference_position = jnp.asarray(reference.position)
        self.reference_orientation = jnp.asarray(reference.orientation)
        self.body_ids = jnp.asarray(body_id_values[np.asarray(body_order)])
        self.body_indices = jnp.asarray(body_order, dtype=jnp.int32)
        self.joint_ids = jnp.asarray(ordered_joint_ids)
        self.joint_kinds = jnp.asarray(joint_kinds)
        self.parent_indices = jnp.asarray(parent_indices)
        self.child_indices = jnp.asarray(child_indices)
        self.dof_body_indices = jnp.asarray(dof_bodies, dtype=jnp.int32)
        self.dof_joint_indices = jnp.asarray(dof_joints, dtype=jnp.int32)
        self.configuration_offsets = jnp.asarray(configuration_offsets, dtype=jnp.int32)
        self.velocity_offsets = jnp.asarray(velocity_offsets, dtype=jnp.int32)
        self.parent_reference_translation = jnp.asarray(reference_translation)
        self.parent_reference_rotation = jnp.asarray(reference_rotation)
        self.parent_reference_orientation = jnp.asarray(reference_orientation)
        self.parent_axes = jnp.asarray(axes)
        self.parent_anchors = jnp.asarray(anchors)
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.root_index = root_index
        self.nq = nq
        self.nv = nv
        self.configuration_slice = slice(0, nq)
        self.velocity_slice = slice(nq, nq + nv)
        self.joint_configuration_slices = tuple(configuration_slices)
        self.joint_velocity_slices = tuple(velocity_slices)
        self.prepared_id = prepared_id
        self._parent_order = tuple(int(index) for index in parent_indices)
        self._child_order = tuple(int(index) for index in child_indices)
        self._kind_order = tuple(int(kind) for kind in joint_kinds)
        self._edge_dof_order = tuple(edge_dofs)
        self._hinge_dof_indices = tuple(hinge_dofs)
        self._body_id_order = tuple(int(identifier) for identifier in self.body_ids)
        self._body_index_order = tuple(body_order)

    @property
    def edge_count(self) -> int:
        return len(self._kind_order)

    @property
    def state_size(self) -> int:
        return self.nq + self.nv

    @property
    def parent_reference_transforms(self) -> Array:
        upper = jnp.concatenate(
            (
                self.parent_reference_rotation,
                self.parent_reference_translation[..., None],
            ),
            axis=-1,
        )
        bottom = jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, 0.0, 1.0], dtype=upper.dtype),
            (self.edge_count, 1, 4),
        )
        return jnp.concatenate((upper, bottom), axis=-2)

    def _configuration(self, value: ArrayLike, name: str, /) -> Array:
        configuration = jnp.asarray(value, dtype=self.reference_position.dtype)
        if configuration.shape != (self.nq,):
            raise ValueError(f"{name} must have shape {(self.nq,)}.")
        return configuration

    def _velocity(self, value: ArrayLike, name: str, /) -> Array:
        velocity = jnp.asarray(value, dtype=self.reference_position.dtype)
        if velocity.shape != (self.nv,):
            raise ValueError(f"{name} must have shape {(self.nv,)}.")
        return velocity

    def zero_configuration(self, /) -> Array:
        return jnp.zeros((self.nq,), dtype=self.reference_position.dtype)

    def reference_configuration(self, /) -> Array:
        return self.zero_configuration()

    def reference_state(self, /) -> ReducedArticulationState:
        return ReducedArticulationState(
            self.zero_configuration(),
            jnp.zeros((self.nv,), dtype=self.reference_position.dtype),
        )

    def pack_state(
        self,
        state_or_configuration: ReducedArticulationState | ArrayLike,
        velocity: ArrayLike | None = None,
        /,
    ) -> Array:
        if isinstance(state_or_configuration, ReducedArticulationState):
            if velocity is not None:
                raise ValueError(
                    "velocity must be omitted when packing a state object."
                )
            configuration = self._configuration(
                state_or_configuration.configuration, "State configuration"
            )
            velocity_array = self._velocity(
                state_or_configuration.velocity, "State velocity"
            )
        else:
            if velocity is None:
                raise TypeError(
                    "velocity is required when packing a configuration array."
                )
            configuration = self._configuration(state_or_configuration, "Configuration")
            velocity_array = self._velocity(velocity, "Velocity")
        return jnp.concatenate((configuration, velocity_array))

    def unpack_state(self, packed: ArrayLike, /) -> ReducedArticulationState:
        value = jnp.asarray(packed, dtype=self.reference_position.dtype)
        if value.shape != (self.state_size,):
            raise ValueError(
                f"Packed articulation state must have shape {(self.state_size,)}."
            )
        return ReducedArticulationState(
            value[self.configuration_slice], value[self.velocity_slice]
        )

    def integrate_configuration(
        self,
        configuration: ArrayLike,
        velocity: ArrayLike,
        step_size: ArrayLike = 1.0,
        /,
    ) -> Array:
        point = self._configuration(configuration, "Configuration")
        tangent = self._velocity(velocity, "Generalized velocity")
        step = jnp.asarray(step_size, dtype=point.dtype)
        if step.shape != ():
            raise ValueError("step_size must be scalar.")
        return _configuration_increment(
            point, tangent, step, self._hinge_dof_indices
        )

    def configuration_difference(
        self, reference: ArrayLike, point: ArrayLike, /
    ) -> Array:
        reference_array = self._configuration(reference, "Reference configuration")
        point_array = self._configuration(point, "Point configuration")
        return _configuration_delta(
            reference_array, point_array, self._hinge_dof_indices
        )

    def _poses(self, configuration: Array, /) -> tuple[Array, Array]:
        position = self.reference_position
        orientation = self.reference_orientation
        for edge, (parent, child, kind, dof) in enumerate(
            zip(
                self._parent_order,
                self._child_order,
                self._kind_order,
                self._edge_dof_order,
            )
        ):
            parent_orientation = orientation[parent]
            parent_rotation = quaternion_rotation_matrix(parent_orientation)
            relative_translation = self.parent_reference_translation[edge]
            relative_orientation = self.parent_reference_orientation[edge]
            if kind == int(RigidJointKind.HINGE):
                joint_orientation = _quaternion_increment(
                    self.parent_axes[edge] * configuration[dof]
                )
                joint_rotation = quaternion_rotation_matrix(joint_orientation)
                arm = relative_translation - self.parent_anchors[edge]
                relative_translation = self.parent_anchors[edge] + contract(
                    "ij,j->i", joint_rotation, arm
                )
                relative_orientation = _quaternion_multiply(
                    joint_orientation, relative_orientation
                )
            elif kind == int(RigidJointKind.PRISMATIC):
                relative_translation = (
                    relative_translation + self.parent_axes[edge] * configuration[dof]
                )
            child_position = position[parent] + contract(
                "ij,j->i", parent_rotation, relative_translation
            )
            child_orientation = _normalize_preserving_sign(
                _quaternion_multiply(parent_orientation, relative_orientation)
            )
            position = position.at[child].set(child_position)
            orientation = orientation.at[child].set(child_orientation)
        return position, orientation

    def _body_velocity_from_poses(
        self,
        generalized_velocity: Array,
        position: Array,
        orientation: Array,
        /,
    ) -> Array:
        linear = jnp.zeros_like(position)
        angular = jnp.zeros_like(position)
        for edge, (parent, child, kind, dof) in enumerate(
            zip(
                self._parent_order,
                self._child_order,
                self._kind_order,
                self._edge_dof_order,
            )
        ):
            parent_linear = linear[parent]
            parent_angular = angular[parent]
            parent_position = position[parent]
            child_position = position[child]
            parent_rotation = quaternion_rotation_matrix(orientation[parent])
            if kind == int(RigidJointKind.HINGE):
                axis_world = contract(
                    "ij,j->i", parent_rotation, self.parent_axes[edge]
                )
                anchor_world = parent_position + contract(
                    "ij,j->i", parent_rotation, self.parent_anchors[edge]
                )
                child_angular = (
                    parent_angular + generalized_velocity[dof] * axis_world
                )
                anchor_velocity = parent_linear + jnp.cross(
                    parent_angular, anchor_world - parent_position
                )
                child_linear = anchor_velocity + jnp.cross(
                    child_angular, child_position - anchor_world
                )
            elif kind == int(RigidJointKind.PRISMATIC):
                axis_world = contract(
                    "ij,j->i", parent_rotation, self.parent_axes[edge]
                )
                child_angular = parent_angular
                child_linear = (
                    parent_linear
                    + jnp.cross(parent_angular, child_position - parent_position)
                    + generalized_velocity[dof] * axis_world
                )
            else:
                child_angular = parent_angular
                child_linear = parent_linear + jnp.cross(
                    parent_angular, child_position - parent_position
                )
            linear = linear.at[child].set(child_linear)
            angular = angular.at[child].set(child_angular)
        return jnp.concatenate((linear, angular), axis=-1)

    @staticmethod
    def _homogeneous_transforms(position: Array, orientation: Array, /) -> Array:
        rotation = quaternion_rotation_matrix(orientation)
        upper = jnp.concatenate((rotation, position[..., None]), axis=-1)
        bottom = jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, 0.0, 1.0], dtype=position.dtype),
            position.shape[:-1] + (1, 4),
        )
        return jnp.concatenate((upper, bottom), axis=-2)

    def forward_kinematics(
        self,
        configuration: ArrayLike,
        velocity: ArrayLike | None = None,
        /,
    ) -> ArticulationKinematics:
        configuration_array = self._configuration(configuration, "Configuration")
        velocity_array = (
            jnp.zeros((self.nv,), dtype=configuration_array.dtype)
            if velocity is None
            else self._velocity(velocity, "Generalized velocity")
        )
        position, orientation = self._poses(configuration_array)
        body_velocity = self._body_velocity_from_poses(
            velocity_array, position, orientation
        )
        bodies = RigidBodyKinematics(
            position,
            body_velocity[:, :3],
            orientation,
            body_velocity[:, 3:],
        )
        transforms = self._homogeneous_transforms(position, orientation)
        finite = (
            jnp.all(jnp.isfinite(configuration_array))
            & jnp.all(jnp.isfinite(velocity_array))
            & jnp.all(jnp.isfinite(position))
            & jnp.all(jnp.isfinite(orientation))
            & jnp.all(jnp.isfinite(body_velocity))
            & jnp.all(jnp.isfinite(transforms))
        )
        return ArticulationKinematics(bodies, transforms, finite, finite)

    def body_transforms(self, configuration: ArrayLike, /) -> Array:
        configuration_array = self._configuration(configuration, "Configuration")
        position, orientation = self._poses(configuration_array)
        return self._homogeneous_transforms(position, orientation)

    def body_velocity_action(
        self, configuration: ArrayLike, generalized_velocity: ArrayLike, /
    ) -> Array:
        configuration_array = self._configuration(configuration, "Configuration")
        velocity_array = self._velocity(
            generalized_velocity, "Generalized velocity"
        )
        position, orientation = self._poses(configuration_array)
        return self._body_velocity_from_poses(
            velocity_array, position, orientation
        )

    def _body_index(self, body_id: int, /) -> int:
        if isinstance(body_id, bool) or not isinstance(body_id, Integral):
            raise TypeError("body_id must be an integer body ID.")
        identifier = int(body_id)
        if identifier not in self._body_id_order:
            raise ValueError("body_id is not part of this articulation.")
        order = self._body_id_order.index(identifier)
        return self._body_index_order[order]

    def body_transform(self, configuration: ArrayLike, body_id: int, /) -> Array:
        index = self._body_index(body_id)
        return self.body_transforms(configuration)[index]

    def frame_transform(
        self,
        configuration: ArrayLike,
        body_id: int,
        local_transform: ArrayLike | None = None,
        /,
    ) -> Array:
        body = self.body_transform(configuration, body_id)
        if local_transform is None:
            return body
        local = jnp.asarray(local_transform, dtype=body.dtype)
        if local.shape != (4, 4):
            raise ValueError("local_transform must be a body-to-frame 4x4 matrix.")
        return contract("ij,jk->ik", body, local)

    def body_jacobian_operator(
        self, configuration: ArrayLike, /
    ) -> FunctionLinearOperator:
        configuration_array = self._configuration(configuration, "Configuration")
        position, orientation = self._poses(configuration_array)
        source = ArraySpace((self.nv,), dtype=configuration_array.dtype)
        target = ArraySpace(
            (self.graph.bodies.capacity, 6), dtype=configuration_array.dtype
        )

        def action(generalized_velocity):
            return self._body_velocity_from_poses(
                generalized_velocity, position, orientation
            )

        return FunctionLinearOperator(
            action,
            source=source,
            target=target,
            operator_id=f"{self.prepared_id}:body-jacobian",
        )

    def frame_jacobian_operator(
        self,
        configuration: ArrayLike,
        body_id: int,
        local_position: ArrayLike | None = None,
        /,
    ) -> FunctionLinearOperator:
        configuration_array = self._configuration(configuration, "Configuration")
        index = self._body_index(body_id)
        position, orientation = self._poses(configuration_array)
        local = (
            jnp.zeros((3,), dtype=configuration_array.dtype)
            if local_position is None
            else jnp.asarray(local_position, dtype=configuration_array.dtype)
        )
        if local.shape != (3,):
            raise ValueError("local_position must have body-frame shape (3,).")
        offset_world = contract(
            "ij,j->i", quaternion_rotation_matrix(orientation[index]), local
        )
        source = ArraySpace((self.nv,), dtype=configuration_array.dtype)
        target = ArraySpace((6,), dtype=configuration_array.dtype)

        def action(generalized_velocity):
            body_velocity = self._body_velocity_from_poses(
                generalized_velocity, position, orientation
            )[index]
            frame_linear = body_velocity[:3] + jnp.cross(
                body_velocity[3:], offset_world
            )
            return jnp.concatenate((frame_linear, body_velocity[3:]))

        return FunctionLinearOperator(
            action,
            source=source,
            target=target,
            operator_id=(
                f"{self.prepared_id}:frame-jacobian:{int(body_id)}"
            ),
        )

    def body_load_pullback(
        self,
        configuration: ArrayLike,
        load: RigidBodyLoad,
        generalized_velocity: ArrayLike,
        /,
    ) -> tuple[Array, ArticulationDualityEvidence]:
        if not isinstance(load, RigidBodyLoad):
            raise TypeError("load must be a RigidBodyLoad.")
        configuration_array = self._configuration(configuration, "Configuration")
        velocity_array = self._velocity(
            generalized_velocity, "Generalized velocity"
        )
        force = jnp.asarray(load.force, dtype=configuration_array.dtype)
        torque = jnp.asarray(load.torque, dtype=configuration_array.dtype)
        expected = (self.graph.bodies.capacity, 3)
        if force.shape != expected or torque.shape != expected:
            raise ValueError(
                "Rigid-body load arrays must have body-capacity shape (N,3)."
            )
        wrench = jnp.concatenate((force, torque), axis=-1)
        position, orientation = self._poses(configuration_array)

        def velocity_action(value):
            return self._body_velocity_from_poses(value, position, orientation)

        body_velocity = velocity_action(velocity_array)
        transpose_action = jax.linear_transpose(
            velocity_action,
            jnp.zeros((self.nv,), dtype=configuration_array.dtype),
        )
        generalized_load = transpose_action(wrench)[0]
        body_power = jnp.sum(body_velocity * wrench)
        generalized_power = jnp.sum(velocity_array * generalized_load)
        residual = body_power - generalized_power
        scale = jnp.maximum(
            1.0, jnp.maximum(jnp.abs(body_power), jnp.abs(generalized_power))
        )
        tolerance = jnp.finfo(configuration_array.dtype).eps * max(
            64, 8 * self.graph.bodies.capacity * 6 * max(self.nv, 1)
        )
        finite = (
            jnp.all(jnp.isfinite(configuration_array))
            & jnp.all(jnp.isfinite(velocity_array))
            & jnp.all(jnp.isfinite(wrench))
            & jnp.all(jnp.isfinite(body_velocity))
            & jnp.all(jnp.isfinite(generalized_load))
            & jnp.all(
                jnp.isfinite(
                    jnp.stack(
                        (body_power, generalized_power, residual, scale)
                    )
                )
            )
        )
        evidence = ArticulationDualityEvidence(
            body_power,
            generalized_power,
            residual,
            scale,
            finite,
            finite & (jnp.abs(residual) <= tolerance * scale),
            self.prepared_id,
        )
        return generalized_load, evidence


__all__ = [
    "ArticulationDualityEvidence",
    "ArticulationKinematics",
    "PreparedReducedArticulation",
    "ReducedArticulationPlan",
    "ReducedArticulationState",
]
