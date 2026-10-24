#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Coarse fixed-base articulated kinematics with a native device-resident schedule.

Four primitives own the kinematic tree traversals of a reduced articulation:

* ``_poses_p`` — forward pose recursion from the reference pose and scalar
  joint coordinates;
* ``_velocity_p`` — ``K`` independent world twist columns ``J(q) u`` at given
  poses, linear in ``u``;
* ``_velocity_transpose_p`` — the linear transpose of ``_velocity_p``: ``K``
  independent body wrench columns pulled back to generalized loads;
* ``_motion_p`` — the shared poses with ``K`` independent twist columns
  ``J(q) u_k`` and directional-derivative columns, column ``k`` being one
  ``jax.jvp`` of ``q -> J(q) u_k`` along ``d_k``.

The tree is static: a :class:`KinematicTopology` of per-edge joint kinds,
configuration indices, parents and children in topological storage order.
Hinge and prismatic edges move with their configuration coordinate; every
other kind is a fixed edge that never reads it. Twists are world COM linear
then angular velocity.

``_*_world`` functions are the canonical JAX meanings: the original
per-edge recursion unrolled over the static edges, so every derivative of
every order — including JAX's structural (symbolic) zeros and therefore its
``0 * NaN`` behaviour — is the derivative of that recursion. Each carry keeps
its own dtype: steps compute in the promoted type and round into the carry.
Numerical operands carry a leading world axis ``W`` of extent ``W`` or one
(shared and read in place); ``jax.vmap`` folds mapped operands into the world
axis so batched worlds bind one operation.

On an admitted CUDA target (``backends._mosaic.lower_mosaic_or_reference``)
every primitive lowers to one Mosaic GPU kernel in which each thread owns one
complete sequential traversal — one world for poses and one
``(world, column)`` lane otherwise. Per-edge arithmetic is not a second
implementation: the reference step functions are traced to jaxprs and
scalarized into the kernel. Motion derivative columns follow the reference's
JVP activity exactly: before tracing, a static schedule replays JAX's
symbolic-zero propagation through the unrolled recursion (carry arrays become
active only once a ``q``-dependent value is stored in them), and every edge
runs a JVP program in which exactly the inactive step inputs are constants,
so the kernel omits precisely the terms JAX omits. Scratch carries are
private kernel outputs that are discarded. Admission is static and shared
with the sweeps: one float32/float64 dtype for every floating operand, at
least one world, body, edge and column, the sweep body/edge/column bounds and
signed 32-bit buffer extents; anything else lowers the reference.
Derivatives of every primitive differentiate the reference. The direct
transpose of the twist action binds the native pullback; the transpose of the
pullback is its reference's linear transpose, the linear part ``J u`` of the
twist action without its ``u``-independent constants.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import cache, partial
from typing import Any, final, NamedTuple, Protocol, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax._src.core import ShapedArray
from jax.extend import core
from jax.interpreters import ad, batching
from jax.typing import ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ._articulated_sweeps import (
    _admitted,
    _Emitter,
    _launch,
    _map_worlds,
    _primitive,
    _reference_jvp,
    _require,
    _require_real,
    _Role,
    _spans_worlds,
    _world_batch,
    _world_count,
    _world_reader,
)
from ._rigid_body import (
    _quaternion_increment,
    _quaternion_multiply,
    quaternion_rotation_matrix,
)
from ._rigid_joints import RigidJointKind


if TYPE_CHECKING:
    from jax.extend.mlir import ir


type Tangent = Array | ad.Zero
type Primal = Array | ad.UndefinedPrimal

_KIND_HINGE = int(RigidJointKind.HINGE)
_KIND_PRISMATIC = int(RigidJointKind.PRISMATIC)
# Step branches; every kind other than hinge and prismatic is fixed.
_FIXED = 0
_HINGE = 1
_PRISMATIC = 2
# Per-body pose scratch of the motion kernel: position, orientation and
# their tangents.
_POSE_WIDTH = 14


def _branch(kind: int, /) -> int:
    if kind == _KIND_HINGE:
        return _HINGE
    if kind == _KIND_PRISMATIC:
        return _PRISMATIC
    return _FIXED


@final
class KinematicTopology(StrictModule):
    """Static fixed-base tree in topological edge storage order.

    ``kinds`` are rigid-joint kinds, ``coordinates`` the configuration (and
    velocity) index of each hinge or prismatic edge — ignored on fixed
    edges — and ``parents``/``children`` body storage indices.
    """

    kinds: tuple[int, ...] = eqx.field(static=True)
    coordinates: tuple[int, ...] = eqx.field(static=True)
    parents: tuple[int, ...] = eqx.field(static=True)
    children: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        kinds: tuple[int, ...],
        coordinates: tuple[int, ...],
        parents: tuple[int, ...],
        children: tuple[int, ...],
    ) -> None:
        values = (kinds, coordinates, parents, children)
        if not all(
            isinstance(group, tuple) and all(type(value) is int for value in group)
            for group in values
        ):
            raise TypeError("Kinematic topology entries must be tuples of Python ints.")
        if len({len(group) for group in values}) != 1:
            raise ValueError("Kinematic topology tuples must have one entry per edge.")
        self.kinds = kinds
        self.coordinates = coordinates
        self.parents = parents
        self.children = children

    @property
    def edge_count(self) -> int:
        return len(self.kinds)

    @property
    def branches(self) -> tuple[int, ...]:
        return tuple(_branch(kind) for kind in self.kinds)

    def require(self, bodies: int, edges: int, *sizes: int) -> None:
        """Refuse a tree that does not index the given body and coordinate extents."""

        if self.edge_count != edges:
            raise ValueError(
                f"Kinematic topology has {self.edge_count} edges; operands have {edges}."
            )
        for parent, child in zip(self.parents, self.children, strict=True):
            if not (0 <= parent < bodies and 0 <= child < bodies):
                raise ValueError("Kinematic topology body indices must be in range.")
        for branch, coordinate in zip(self.branches, self.coordinates, strict=True):
            if branch != _FIXED and not all(0 <= coordinate < size for size in sizes):
                raise ValueError(
                    "Moving kinematic edges must index an existing coordinate."
                )


# ---------------------------------------------------------------------------
# Canonical per-edge steps.
# ---------------------------------------------------------------------------


def _normalize_preserving_sign(value: Array, /) -> Array:
    norm = jnp.sqrt(jnp.sum(value * value, axis=-1, keepdims=True))
    return value / jnp.maximum(norm, jnp.finfo(value.dtype).tiny)


def _pose_step(
    branch: int,
    parent_position: Array,
    parent_orientation: Array,
    translation: Array,
    relative_orientation: Array,
    axis: Array,
    anchor: Array,
    coordinate: Array | None,
    /,
) -> tuple[Array, Array]:
    """Child pose from its parent pose and the edge's joint coordinate."""

    parent_rotation = quaternion_rotation_matrix(parent_orientation)
    if branch == _HINGE:
        assert coordinate is not None
        joint_orientation = _quaternion_increment(axis * coordinate)
        joint_rotation = quaternion_rotation_matrix(joint_orientation)
        arm = translation - anchor
        translation = anchor + contract("ij,j->i", joint_rotation, arm)
        relative_orientation = _quaternion_multiply(
            joint_orientation, relative_orientation
        )
    elif branch == _PRISMATIC:
        assert coordinate is not None
        translation = translation + axis * coordinate
    child_position = parent_position + contract("ij,j->i", parent_rotation, translation)
    child_orientation = _normalize_preserving_sign(
        _quaternion_multiply(parent_orientation, relative_orientation)
    )
    return child_position, child_orientation


def _velocity_step(
    branch: int,
    rate: Array | None,
    parent_linear: Array,
    parent_angular: Array,
    parent_position: Array,
    child_position: Array,
    parent_orientation: Array,
    axis: Array,
    anchor: Array,
    /,
) -> tuple[Array, Array]:
    """Child world twist from its parent twist, the poses and the joint rate."""

    parent_rotation = quaternion_rotation_matrix(parent_orientation)
    if branch == _HINGE:
        assert rate is not None
        axis_world = contract("ij,j->i", parent_rotation, axis)
        anchor_world = parent_position + contract("ij,j->i", parent_rotation, anchor)
        child_angular = parent_angular + rate * axis_world
        anchor_velocity = parent_linear + jnp.cross(
            parent_angular, anchor_world - parent_position
        )
        child_linear = anchor_velocity + jnp.cross(
            child_angular, child_position - anchor_world
        )
    elif branch == _PRISMATIC:
        assert rate is not None
        axis_world = contract("ij,j->i", parent_rotation, axis)
        child_angular = parent_angular
        child_linear = (
            parent_linear
            + jnp.cross(parent_angular, child_position - parent_position)
            + rate * axis_world
        )
    else:
        child_angular = parent_angular
        child_linear = parent_linear + jnp.cross(
            parent_angular, child_position - parent_position
        )
    return child_linear, child_angular


# ---------------------------------------------------------------------------
# Canonical single-world references: the unrolled recursion.
# ---------------------------------------------------------------------------


def _poses_world(
    reference_position: Array,
    reference_orientation: Array,
    translation: Array,
    relative_orientation: Array,
    axes: Array,
    anchors: Array,
    configuration: Array,
    /,
    *,
    topology: KinematicTopology,
) -> tuple[Array, Array]:
    """Forward pose recursion; bodies that are never a child keep their pose."""

    position = reference_position
    orientation = reference_orientation
    for edge, (branch, coordinate, parent, child) in enumerate(
        zip(
            topology.branches,
            topology.coordinates,
            topology.parents,
            topology.children,
            strict=True,
        )
    ):
        child_position, child_orientation = _pose_step(
            branch,
            position[parent],
            orientation[parent],
            translation[edge],
            relative_orientation[edge],
            axes[edge],
            anchors[edge],
            None if branch == _FIXED else configuration[coordinate],
        )
        position = position.at[child].set(child_position.astype(position.dtype))
        orientation = orientation.at[child].set(
            child_orientation.astype(orientation.dtype)
        )
    return position, orientation


def _velocity_column(
    axes: Array,
    anchors: Array,
    position: Array,
    orientation: Array,
    velocity: Array,
    /,
    *,
    topology: KinematicTopology,
) -> Array:
    """``(C, 6)`` world twists of one generalized-velocity column."""

    linear = jnp.zeros_like(position)
    angular = jnp.zeros_like(position)
    for edge, (branch, coordinate, parent, child) in enumerate(
        zip(
            topology.branches,
            topology.coordinates,
            topology.parents,
            topology.children,
            strict=True,
        )
    ):
        child_linear, child_angular = _velocity_step(
            branch,
            None if branch == _FIXED else velocity[coordinate],
            linear[parent],
            angular[parent],
            position[parent],
            position[child],
            orientation[parent],
            axes[edge],
            anchors[edge],
        )
        linear = linear.at[child].set(child_linear.astype(linear.dtype))
        angular = angular.at[child].set(child_angular.astype(angular.dtype))
    return jnp.concatenate((linear, angular), axis=-1)


def _velocity_world(
    axes: Array,
    anchors: Array,
    position: Array,
    orientation: Array,
    velocity: Array,
    /,
    *,
    topology: KinematicTopology,
) -> tuple[Array]:
    """``(K, C, 6)`` twists of ``(K, V)`` independent velocity columns."""

    if velocity.shape[0] == 0:
        return (jnp.zeros((0, position.shape[0], 6), dtype=position.dtype),)
    column = partial(
        _velocity_column, axes, anchors, position, orientation, topology=topology
    )
    return (jax.vmap(column)(velocity),)


def _velocity_transpose_world(
    axes: Array,
    anchors: Array,
    position: Array,
    orientation: Array,
    cotangent: Array,
    /,
    *,
    topology: KinematicTopology,
    velocity_size: int,
    velocity_dtype: np.dtype,
) -> tuple[Array]:
    """Linear transpose of ``_velocity_world`` in its velocity columns."""

    template = jax.ShapeDtypeStruct((cotangent.shape[0], velocity_size), velocity_dtype)

    def action(velocity: Array) -> Array:
        return _velocity_world(
            axes, anchors, position, orientation, velocity, topology=topology
        )[0]

    (result,) = jax.linear_transpose(action, template)(cotangent)
    return (result,)


def _motion_world(
    reference_position: Array,
    reference_orientation: Array,
    translation: Array,
    relative_orientation: Array,
    axes: Array,
    anchors: Array,
    configuration: Array,
    rates: Array,
    directions: Array,
    /,
    *,
    topology: KinematicTopology,
) -> tuple[Array, Array, Array, Array]:
    """Shared poses, twist columns ``J(q) u_k`` and their directional derivatives.

    Column ``k`` is one ``jax.jvp`` of ``q -> J(q) rates[k]`` (pose recursion
    followed by the twist recursion) along ``directions[k]``.
    """

    geometry = (
        reference_position,
        reference_orientation,
        translation,
        relative_orientation,
        axes,
        anchors,
    )
    position, orientation = _poses_world(*geometry, configuration, topology=topology)
    if rates.shape[0] == 0:
        empty = jnp.zeros((0, position.shape[0], 6), dtype=position.dtype)
        return position, orientation, empty, empty

    def column(rate: Array, direction: Array) -> tuple[Array, Array]:
        def action(point: Array) -> Array:
            position_, orientation_ = _poses_world(*geometry, point, topology=topology)
            return _velocity_column(
                axes, anchors, position_, orientation_, rate, topology=topology
            )

        return jax.jvp(action, (configuration,), (direction,))

    twists, derivatives = jax.vmap(column)(rates, directions)
    return position, orientation, twists, derivatives


# ---------------------------------------------------------------------------
# World-axis references bound by the primitives.
# ---------------------------------------------------------------------------

_POSE_ROLES = (_Role.WORLD,) * 7
_VELOCITY_ROLES = (_Role.WORLD,) * 5
_MOTION_ROLES = (_Role.WORLD,) * 9
# Position of the operand in which the velocity primitives are linear.
_LINEAR = 4


def _poses_reference(*operands: Array, topology: KinematicTopology) -> tuple[Array, ...]:
    return _map_worlds(partial(_poses_world, topology=topology), _POSE_ROLES, operands)


def _velocity_reference(
    *operands: Array, topology: KinematicTopology
) -> tuple[Array, ...]:
    return _map_worlds(
        partial(_velocity_world, topology=topology), _VELOCITY_ROLES, operands
    )


def _velocity_transpose_reference(
    *operands: Array,
    topology: KinematicTopology,
    velocity_size: int,
    velocity_dtype: np.dtype,
) -> tuple[Array, ...]:
    return _map_worlds(
        partial(
            _velocity_transpose_world,
            topology=topology,
            velocity_size=velocity_size,
            velocity_dtype=velocity_dtype,
        ),
        _VELOCITY_ROLES,
        operands,
    )


def _motion_reference(*operands: Array, topology: KinematicTopology) -> tuple[Array, ...]:
    return _map_worlds(partial(_motion_world, topology=topology), _MOTION_ROLES, operands)


# ---------------------------------------------------------------------------
# Abstract evaluation.
# ---------------------------------------------------------------------------


def _result_avals(
    reference: Callable[..., tuple[Array, ...]],
    operands: Sequence[ShapedArray],
    /,
    **params: object,
) -> tuple[ShapedArray, ...]:
    """Result types are those of the canonical reference, including promotion."""

    specs = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype, weak_type=value.weak_type)
        for value in operands
    )
    result = jax.eval_shape(partial(reference, **params), *specs)
    return tuple(
        ShapedArray(value.shape, value.dtype, weak_type=value.weak_type)
        for value in result
    )


def _require_rank(value: ShapedArray, rank: int, name: str, /) -> None:
    _require_real(value, name)
    if value.ndim != rank:
        raise ValueError(f"Articulated kinematic {name} must have rank {rank}.")


def _require_topology(topology: object, /) -> KinematicTopology:
    if not isinstance(topology, KinematicTopology):
        raise TypeError("Articulated kinematics require a KinematicTopology.")
    return topology


def _require_tree(operands: Sequence[ShapedArray], worlds: int, /) -> tuple[int, int]:
    """Validate the reference pose and edge geometry; return body and edge counts."""

    (
        reference_position,
        reference_orientation,
        translation,
        relative_orientation,
        axes,
        anchors,
    ) = operands[:6]
    _require_rank(reference_position, 3, "reference position")
    _require_rank(translation, 3, "edge translation")
    bodies = reference_position.shape[1]
    edges = translation.shape[1]
    _require(reference_position, worlds, (bodies, 3), "reference position")
    _require(reference_orientation, worlds, (bodies, 4), "reference orientation")
    _require(translation, worlds, (edges, 3), "edge translation")
    _require(relative_orientation, worlds, (edges, 4), "edge orientation")
    _require(axes, worlds, (edges, 3), "joint axes")
    _require(anchors, worlds, (edges, 3), "joint anchors")
    return bodies, edges


def _require_poses(operands: Sequence[ShapedArray], worlds: int, /) -> tuple[int, int]:
    axes, anchors, position, orientation = operands[:4]
    _require_rank(axes, 3, "joint axes")
    _require_rank(position, 3, "position")
    bodies = position.shape[1]
    edges = axes.shape[1]
    _require(axes, worlds, (edges, 3), "joint axes")
    _require(anchors, worlds, (edges, 3), "joint anchors")
    _require(position, worlds, (bodies, 3), "position")
    _require(orientation, worlds, (bodies, 4), "orientation")
    return bodies, edges


def _poses_abstract(
    *operands: ShapedArray, topology: KinematicTopology
) -> tuple[ShapedArray, ...]:
    topology = _require_topology(topology)
    configuration = operands[6]
    _require_rank(configuration, 2, "configuration")
    worlds = _world_count([value.shape for value in operands], _POSE_ROLES)
    bodies, edges = _require_tree(operands, worlds)
    _require(configuration, worlds, configuration.shape[1:], "configuration")
    topology.require(bodies, edges, configuration.shape[1])
    return _result_avals(_poses_reference, operands, topology=topology)


def _velocity_abstract(
    *operands: ShapedArray, topology: KinematicTopology
) -> tuple[ShapedArray, ...]:
    topology = _require_topology(topology)
    velocity = operands[_LINEAR]
    _require_rank(velocity, 3, "generalized velocity")
    worlds = _world_count([value.shape for value in operands], _VELOCITY_ROLES)
    bodies, edges = _require_poses(operands, worlds)
    _require(velocity, worlds, velocity.shape[1:], "generalized velocity")
    topology.require(bodies, edges, velocity.shape[2])
    return _result_avals(_velocity_reference, operands, topology=topology)


def _velocity_transpose_abstract(
    *operands: ShapedArray,
    topology: KinematicTopology,
    velocity_size: int,
    velocity_dtype: np.dtype,
) -> tuple[ShapedArray, ...]:
    topology = _require_topology(topology)
    cotangent = operands[_LINEAR]
    _require_rank(cotangent, 4, "body cotangent")
    if type(velocity_size) is not int or velocity_size < 0:
        raise ValueError("Articulated velocity size must be a non-negative integer.")
    if not jnp.issubdtype(velocity_dtype, jnp.floating):
        raise TypeError("Articulated velocity dtype must be real floating.")
    worlds = _world_count([value.shape for value in operands], _VELOCITY_ROLES)
    bodies, edges = _require_poses(operands, worlds)
    _require(cotangent, worlds, (cotangent.shape[1], bodies, 6), "body cotangent")
    if cotangent.dtype != operands[2].dtype:
        raise TypeError("Articulated body cotangents must have the position dtype.")
    topology.require(bodies, edges, velocity_size)
    return _result_avals(
        _velocity_transpose_reference,
        operands,
        topology=topology,
        velocity_size=velocity_size,
        velocity_dtype=velocity_dtype,
    )


def _motion_abstract(
    *operands: ShapedArray, topology: KinematicTopology
) -> tuple[ShapedArray, ...]:
    topology = _require_topology(topology)
    configuration, rates, directions = operands[6:]
    _require_rank(configuration, 2, "configuration")
    _require_rank(rates, 3, "rates")
    _require_rank(directions, 3, "directions")
    worlds = _world_count([value.shape for value in operands], _MOTION_ROLES)
    bodies, edges = _require_tree(operands, worlds)
    coordinates = configuration.shape[1]
    columns = rates.shape[1]
    _require(configuration, worlds, (coordinates,), "configuration")
    _require(rates, worlds, (columns, rates.shape[2]), "rates")
    _require(directions, worlds, (columns, coordinates), "directions")
    if directions.dtype != configuration.dtype:
        raise TypeError(
            "Articulated motion directions must have the configuration dtype."
        )
    topology.require(bodies, edges, coordinates, rates.shape[2])
    return _result_avals(_motion_reference, operands, topology=topology)


# ---------------------------------------------------------------------------
# Step programs and their JVP activity.
# ---------------------------------------------------------------------------

_POSE_SHAPES: tuple[tuple[int, ...], ...] = ((3,), (4,), (3,), (4,), (3,), (3,), ())
_VELOCITY_SHAPES: tuple[tuple[int, ...], ...] = (
    (),
    (3,),
    (3,),
    (3,),
    (3,),
    (4,),
    (3,),
    (3,),
)
_STEPS: dict[
    str, tuple[Callable[..., tuple[Array, Array]], tuple[tuple[int, ...], ...]]
] = {
    "pose": (_pose_step, _POSE_SHAPES),
    "velocity": (_velocity_step, _VELOCITY_SHAPES),
}
_CALLS = frozenset(("jit", "pjit", "closed_call", "core_call"))


def _call_jaxpr(eqn: core.JaxprEqn, /) -> core.ClosedJaxpr | None:
    if eqn.primitive.name not in _CALLS:
        return None
    closed = eqn.params.get("jaxpr", eqn.params.get("call_jaxpr"))
    if not isinstance(closed, core.ClosedJaxpr):
        closed = core.ClosedJaxpr(closed, ())
    return closed


def _activity(jaxpr: core.Jaxpr, active: Sequence[bool], /) -> tuple[bool, ...]:
    """Which outputs carry a structurally nonzero JVP tangent.

    Replays JAX's symbolic-zero propagation: a floating result is active when
    one of its operands is; calls propagate per output through their bodies.
    """

    marked = {
        variable for variable, flag in zip(jaxpr.invars, active, strict=True) if flag
    }

    def is_active(atom: object) -> bool:
        return isinstance(atom, core.Var) and atom in marked

    for eqn in jaxpr.eqns:
        flags = [is_active(atom) for atom in eqn.invars]
        if not any(flags):
            continue
        inner = _call_jaxpr(eqn)
        outputs = (
            _activity(inner.jaxpr, flags)
            if inner is not None
            else (True,) * len(eqn.outvars)
        )
        for variable, flag in zip(eqn.outvars, outputs, strict=True):
            if flag and jnp.issubdtype(_shaped(variable).dtype, jnp.inexact):
                marked.add(variable)
    return tuple(is_active(atom) for atom in jaxpr.outvars)


def _specs(
    shapes: Sequence[tuple[int, ...]], dtype: np.dtype, /
) -> tuple[jax.ShapeDtypeStruct, ...]:
    return tuple(jax.ShapeDtypeStruct(shape, dtype) for shape in shapes)


@cache
def _primal_program(name: str, branch: int, dtype: np.dtype, /) -> core.ClosedJaxpr:
    """The step itself; inputs and outputs in step order."""
    step, shapes = _STEPS[name]
    return jax.make_jaxpr(partial(step, branch))(*_specs(shapes, dtype))


class _Derivative(NamedTuple):
    """A step JVP with exactly the active inputs differentiated.

    ``closed`` takes every step input followed by the tangents of the active
    inputs and returns the step outputs followed by output tangents (when any
    input is active). ``outputs`` marks the structurally nonzero tangents.
    """

    closed: core.ClosedJaxpr
    inputs: tuple[bool, ...]
    outputs: tuple[bool, ...]


@cache
def _derivative(
    name: str, branch: int, active: tuple[bool, ...], dtype: np.dtype, /
) -> _Derivative:
    step, shapes = _STEPS[name]
    specs = _specs(shapes, dtype)
    primal = _primal_program(name, branch, dtype)
    outputs = _activity(primal.jaxpr, active)
    positions = tuple(position for position, flag in enumerate(active) if flag)
    if not positions:
        return _Derivative(primal, active, outputs)

    def run(*values: Array) -> object:
        primals = values[: len(specs)]
        tangents = values[len(specs) :]

        # Inactive inputs are closed-over constants: JAX keeps their tangents
        # symbolic and omits every term they would multiply.
        def differentiated(*inputs: Array) -> tuple[Array, Array]:
            arguments = list(primals)
            for position, value in zip(positions, inputs, strict=True):
                arguments[position] = value
            return step(branch, *arguments)

        return jax.jvp(
            differentiated, tuple(primals[position] for position in positions), tangents
        )

    closed = jax.make_jaxpr(run)(*specs, *(specs[position] for position in positions))
    return _Derivative(closed, active, outputs)


@cache
def _transpose_program(branch: int, dtype: np.dtype, /) -> core.ClosedJaxpr:
    """Linear transpose of the twist step in (rate, parent linear, parent angular)."""

    scalar, vector = jax.ShapeDtypeStruct((), dtype), jax.ShapeDtypeStruct((3,), dtype)
    quaternion = jax.ShapeDtypeStruct((4,), dtype)

    def run(*values: Array) -> object:
        parent_position, child_position, parent_orientation, axis, anchor = values[:5]
        return jax.linear_transpose(
            lambda rate, linear, angular: _velocity_step(
                branch,
                rate,
                linear,
                angular,
                parent_position,
                child_position,
                parent_orientation,
                axis,
                anchor,
            ),
            scalar,
            vector,
            vector,
        )(values[5:])

    return jax.make_jaxpr(run)(vector, vector, quaternion, vector, vector, vector, vector)


class _Stage(NamedTuple):
    """Per-edge program codes of one traversal and each program's carry activity."""

    programs: tuple[_Derivative, ...]
    carries: tuple[tuple[bool, bool], ...]
    branches: tuple[int, ...]
    codes: tuple[int, ...]


@cache
def _motion_schedule(
    topology: KinematicTopology, dtype: np.dtype, /
) -> tuple[_Stage, _Stage]:
    """Replay the reference JVP's activity through the unrolled recursion.

    Every carried array of the reference starts as a ``q``-independent
    constant and becomes active once an active value is stored in it; its
    gathered entries are then active (numerically zero where untouched).
    """

    def stage(
        name: str,
        inputs: Callable[[int, bool, bool], tuple[bool, ...]],
    ) -> tuple[_Stage, tuple[bool, bool]]:
        keys: dict[tuple[int, bool, bool], int] = {}
        programs: list[_Derivative] = []
        carries: list[tuple[bool, bool]] = []
        branches: list[int] = []
        codes: list[int] = []
        first = second = False
        for branch in topology.branches:
            key = (branch, first, second)
            if key not in keys:
                keys[key] = len(programs)
                programs.append(
                    _derivative(name, branch, inputs(branch, first, second), dtype)
                )
                carries.append((first, second))
                branches.append(branch)
            code = keys[key]
            codes.append(code)
            outputs = programs[code].outputs
            first = first or outputs[0]
            second = second or outputs[1]
        return (
            _Stage(tuple(programs), tuple(carries), tuple(branches), tuple(codes)),
            (first, second),
        )

    poses, (position, orientation) = stage(
        "pose",
        lambda branch, position_, orientation_: (
            position_,
            orientation_,
            False,
            False,
            False,
            False,
            branch != _FIXED,
        ),
    )
    twists, _ = stage(
        "velocity",
        lambda branch, linear, angular: (
            False,
            linear,
            angular,
            position,
            position,
            orientation,
            False,
            False,
        ),
    )
    return poses, twists


# ---------------------------------------------------------------------------
# Scalarization of step programs.
# ---------------------------------------------------------------------------

_SCALAR_DTYPES = (np.dtype(np.bool_), np.dtype(np.float32), np.dtype(np.float64))


class _Scalars(Protocol):
    """Scalar operations of one backend over values of the jaxpr's own dtypes."""

    def literal(self, value: bool | float, dtype: np.dtype, /) -> Any: ...

    def unary(self, name: str, value: Any, /) -> Any: ...

    def binary(self, name: str, left: Any, right: Any, /) -> Any: ...

    def compare(self, name: str, left: Any, right: Any, /) -> Any: ...

    def select(self, predicate: Any, on_false: Any, on_true: Any, /) -> Any: ...

    def convert(self, value: Any, source: np.dtype, target: np.dtype, /) -> Any: ...


class _MosaicScalars:
    """Scalar operations emitted as MLIR ``arith``/``math`` in one kernel lane."""

    def __init__(self) -> None:
        from jax.experimental.mosaic import gpu as mgpu
        from jax.extend.mlir import ir
        from jax.extend.mlir.dialects import arith, math as mlir_math

        self._mgpu = mgpu
        self._arith = arith
        self._math = mlir_math
        self._types = {
            np.dtype(np.bool_): ir.IntegerType.get_signless(1),
            np.dtype(np.float32): ir.F32Type.get(),
            np.dtype(np.float64): ir.F64Type.get(),
        }

    def literal(self, value: bool | float, dtype: np.dtype, /) -> ir.Value:
        if dtype == np.dtype(np.bool_):
            return self._mgpu.c(int(bool(value)), self._types[dtype])
        return self._mgpu.c(float(value), self._types[dtype])

    def unary(self, name: str, value: ir.Value, /) -> ir.Value:
        operation = {
            "neg": self._arith.negf,
            "sqrt": self._math.sqrt,
            "sin": self._math.sin,
            "cos": self._math.cos,
        }[name]
        return operation(value)

    def binary(self, name: str, left: ir.Value, right: ir.Value, /) -> ir.Value:
        operation = {
            "add": self._arith.addf,
            "sub": self._arith.subf,
            "mul": self._arith.mulf,
            "div": self._arith.divf,
            # NaN-propagating, as XLA maximum and minimum.
            "max": self._arith.maximumf,
            "min": self._arith.minimumf,
        }[name]
        return operation(left, right)

    def compare(self, name: str, left: ir.Value, right: ir.Value, /) -> ir.Value:
        predicate = self._arith.CmpFPredicate
        ordered = {
            "eq": predicate.OEQ,
            "ne": predicate.UNE,
            "lt": predicate.OLT,
            "le": predicate.OLE,
            "gt": predicate.OGT,
            "ge": predicate.OGE,
        }[name]
        return self._arith.cmpf(ordered, left, right)

    def select(
        self, predicate: ir.Value, on_false: ir.Value, on_true: ir.Value, /
    ) -> ir.Value:
        return self._arith.select(predicate, on_true, on_false)

    def convert(self, value: ir.Value, source: np.dtype, target: np.dtype, /) -> ir.Value:
        result = self._types[target]
        if source == np.dtype(np.bool_):
            return self._arith.uitofp(result, value)
        if source.itemsize < target.itemsize:
            return self._arith.extf(result, value)
        return self._arith.truncf(result, value)


type _Values = np.ndarray

_UNARY = frozenset(("neg", "sqrt", "sin", "cos"))
_BINARY = {"add": "add", "add_any": "add", "sub": "sub", "mul": "mul", "div": "div"}
_BINARY |= {"max": "max", "min": "min"}
_COMPARE = frozenset(("eq", "ne", "lt", "le", "gt", "ge"))


def _object_array(shape: tuple[int, ...], /) -> np.ndarray:
    return np.empty(shape, dtype=object)


def _shaped(atom: core.Var | core.Literal, /) -> ShapedArray:
    """The array type of one jaxpr atom; step programs hold arrays only."""

    aval = atom.aval
    if not isinstance(aval, ShapedArray):
        raise NotImplementedError(
            "Articulated kinematic kernels scalarize only array values."
        )
    return aval


def _scalar_dtype(atom: core.Var | core.Literal, /) -> np.dtype:
    resolved = np.dtype(_shaped(atom).dtype)
    if resolved not in _SCALAR_DTYPES:
        raise NotImplementedError(
            f"Articulated kinematic kernels cannot scalarize {resolved} values."
        )
    return resolved


def _constant(ops: _Scalars, value: object, dtype: np.dtype, /) -> _Values:
    host = np.asarray(value, dtype=dtype)
    result = _object_array(host.shape)
    for index in np.ndindex(host.shape):
        result[index] = ops.literal(
            bool(host[index]) if dtype == np.dtype(np.bool_) else float(host[index]),
            dtype,
        )
    return result


def _elementwise(function: Callable[..., object], *values: _Values) -> _Values:
    arrays = np.broadcast_arrays(*values)
    result = _object_array(arrays[0].shape)
    for index in np.ndindex(result.shape):
        result[index] = function(*(array[index] for array in arrays))
    return result


def _sum(ops: _Scalars, terms: Sequence[object], dtype: np.dtype, /) -> object:
    """Ascending sequential sum; an empty sum is an exact zero."""
    if not terms:
        return ops.literal(0.0, dtype)
    total = terms[0]
    for term in terms[1:]:
        total = ops.binary("add", total, term)
    return total


def _integer_power(
    ops: _Scalars, value: object, exponent: int, dtype: np.dtype, /
) -> object:
    """Square-and-multiply grouping of ``lax.integer_pow``."""

    if exponent == 0:
        return ops.literal(1.0, dtype)
    reciprocal = exponent < 0
    remaining = -exponent if reciprocal else exponent
    accumulated = None
    base = value
    while remaining > 0:
        if remaining & 1:
            accumulated = (
                base if accumulated is None else ops.binary("mul", accumulated, base)
            )
        remaining >>= 1
        if remaining > 0:
            base = ops.binary("mul", base, base)
    if reciprocal:
        return ops.binary("div", ops.literal(1.0, dtype), accumulated)
    return accumulated


def _pad(
    operand: _Values, fill: object, config: Sequence[tuple[int, int, int]], /
) -> _Values:
    result = operand
    for axis, (low, high, interior) in enumerate(config):
        extent = result.shape[axis]
        size = extent + max(extent - 1, 0) * interior
        shape = list(result.shape)
        shape[axis] = size
        spread = np.full(shape, fill, dtype=object)
        selector = [slice(None)] * result.ndim
        selector[axis] = slice(0, size, interior + 1)
        spread[tuple(selector)] = result
        for edge, amount in ((0, low), (1, high)):
            if amount >= 0:
                shape = list(spread.shape)
                shape[axis] = amount
                block = np.full(shape, fill, dtype=object)
                pieces = (block, spread) if edge == 0 else (spread, block)
                spread = np.concatenate(pieces, axis=axis)
            else:
                selector = [slice(None)] * spread.ndim
                selector[axis] = (
                    slice(-amount, None)
                    if edge == 0
                    else slice(None, spread.shape[axis] + amount)
                )
                spread = spread[tuple(selector)]
        result = spread
    return result


def _dot_general(
    ops: _Scalars, eqn: core.JaxprEqn, left: _Values, right: _Values, /
) -> _Values:
    (left_contract, right_contract), (left_batch, right_batch) = eqn.params[
        "dimension_numbers"
    ]
    dtype = _scalar_dtype(eqn.outvars[0])
    if any(_scalar_dtype(atom) != dtype for atom in eqn.invars):
        raise NotImplementedError("Mixed-precision contractions are not scalarized.")
    left_free = [
        d for d in range(left.ndim) if d not in left_contract and d not in left_batch
    ]
    right_free = [
        d for d in range(right.ndim) if d not in right_contract and d not in right_batch
    ]
    batch_shape = [left.shape[d] for d in left_batch]
    left_shape = [left.shape[d] for d in left_free]
    right_shape = [right.shape[d] for d in right_free]
    batch = math.prod(batch_shape)
    rows = math.prod(left_shape)
    columns = math.prod(right_shape)
    depth = math.prod(left.shape[d] for d in left_contract)
    lhs = np.transpose(left, list(left_batch) + left_free + list(left_contract)).reshape(
        (batch, rows, depth)
    )
    rhs = np.transpose(
        right, list(right_batch) + right_free + list(right_contract)
    ).reshape((batch, columns, depth))
    result = _object_array((batch, rows, columns))
    for b in range(batch):
        for i in range(rows):
            for j in range(columns):
                result[b, i, j] = _sum(
                    ops,
                    [ops.binary("mul", lhs[b, i, k], rhs[b, j, k]) for k in range(depth)],
                    dtype,
                )
    return result.reshape(tuple(batch_shape + left_shape + right_shape))


def _apply(ops: _Scalars, eqn: core.JaxprEqn, inputs: list[_Values], /) -> list[_Values]:
    """Scalarize one equation over object arrays of scalar values."""

    name = eqn.primitive.name
    params = eqn.params
    for atom in (*eqn.invars, *eqn.outvars):
        _scalar_dtype(atom)
    if name in _BINARY:
        operation = _BINARY[name]
        return [_elementwise(lambda a, b: ops.binary(operation, a, b), *inputs)]
    if name in _UNARY:
        return [_elementwise(lambda a: ops.unary(name, a), inputs[0])]
    if name in _COMPARE:
        return [_elementwise(lambda a, b: ops.compare(name, a, b), *inputs)]
    if name == "integer_pow":
        dtype = _scalar_dtype(eqn.outvars[0])
        return [
            _elementwise(lambda a: _integer_power(ops, a, params["y"], dtype), inputs[0])
        ]
    if name == "select_n":
        predicate, *cases = inputs
        if len(cases) != 2 or _scalar_dtype(eqn.invars[0]) != np.dtype(np.bool_):
            raise NotImplementedError("Only boolean two-way selects are scalarized.")
        return [_elementwise(ops.select, predicate, *cases)]
    if name == "convert_element_type":
        source = _scalar_dtype(eqn.invars[0])
        target = np.dtype(params["new_dtype"])
        if source == target:
            return [inputs[0]]
        if target == np.dtype(np.bool_):
            raise NotImplementedError("Conversions to bool are not scalarized.")
        return [_elementwise(lambda a: ops.convert(a, source, target), inputs[0])]
    if name in ("copy", "copy_p"):
        return [inputs[0]]
    if name == "broadcast_in_dim":
        operand = inputs[0]
        shape = tuple(params["shape"])
        expanded = [1] * len(shape)
        for position, dimension in enumerate(params["broadcast_dimensions"]):
            expanded[dimension] = operand.shape[position]
        view = np.broadcast_to(operand.reshape(expanded), shape)
        result = _object_array(shape)
        result[...] = view
        return [result]
    if name == "reshape":
        if params.get("dimensions") is not None:
            raise NotImplementedError("Transposing reshapes are not scalarized.")
        return [inputs[0].reshape(tuple(params["new_sizes"]))]
    if name == "squeeze":
        return [np.squeeze(inputs[0], axis=tuple(params["dimensions"]))]
    if name == "transpose":
        return [np.transpose(inputs[0], tuple(params["permutation"]))]
    if name == "slice":
        strides = params["strides"] or (1,) * inputs[0].ndim
        selector = tuple(
            slice(start, limit, stride)
            for start, limit, stride in zip(
                params["start_indices"], params["limit_indices"], strides, strict=True
            )
        )
        return [inputs[0][selector]]
    if name == "concatenate":
        return [np.concatenate(inputs, axis=params["dimension"])]
    if name == "stack":
        return [np.stack(inputs, axis=params["axis"])]
    if name == "split":
        boundaries = np.cumsum(params["sizes"])[:-1]
        return list(np.split(inputs[0], boundaries, axis=params["axis"]))
    if name == "pad":
        operand, fill = inputs
        return [_pad(operand, fill[()], params["padding_config"])]
    if name == "reduce_sum":
        (operand,) = inputs
        dtype = _scalar_dtype(eqn.outvars[0])
        axes = tuple(params["axes"])
        kept = [d for d in range(operand.ndim) if d not in axes]
        moved = np.transpose(operand, kept + list(axes))
        rows = moved.reshape(
            (
                math.prod(operand.shape[d] for d in kept),
                math.prod(operand.shape[d] for d in axes),
            )
        )
        result = _object_array((rows.shape[0],))
        for row in range(rows.shape[0]):
            result[row] = _sum(ops, list(rows[row]), dtype)
        return [result.reshape(tuple(operand.shape[d] for d in kept))]
    if name == "dot_general":
        return [_dot_general(ops, eqn, *inputs)]
    inner = _call_jaxpr(eqn)
    if inner is not None:
        return _evaluate(inner, ops, *inputs)
    raise NotImplementedError(
        f"Articulated kinematic kernels cannot scalarize primitive {name!r}."
    )


def _evaluate(
    closed: core.ClosedJaxpr, ops: _Scalars, *arguments: _Values
) -> list[_Values]:
    """Evaluate one step jaxpr elementwise through a scalar-operation backend."""

    jaxpr = closed.jaxpr
    environment: dict[core.Var, _Values] = {}

    def read(atom: core.Var | core.Literal) -> _Values:
        if isinstance(atom, core.Literal):
            return _constant(ops, atom.val, _scalar_dtype(atom))
        return environment[atom]

    for variable, value in zip(jaxpr.constvars, closed.consts, strict=True):
        environment[variable] = _constant(ops, value, _scalar_dtype(variable))
    for variable, value in zip(jaxpr.invars, arguments, strict=True):
        if value.shape != tuple(_shaped(variable).shape):
            raise ValueError("Scalarized step arguments have the wrong shape.")
        environment[variable] = value
    for eqn in jaxpr.eqns:
        outputs = _apply(ops, eqn, [read(atom) for atom in eqn.invars])
        for variable, value in zip(eqn.outvars, outputs, strict=True):
            if not isinstance(variable, core.DropVar):
                environment[variable] = value
    return [read(atom) for atom in jaxpr.outvars]


# ---------------------------------------------------------------------------
# Native Mosaic GPU kernels.
# ---------------------------------------------------------------------------


def _load(
    emit: _Emitter,
    ref: ir.Value,
    prefix: Sequence[ir.Value | int],
    start: int,
    size: int,
    /,
) -> _Values:
    result = _object_array((size,))
    for component in range(size):
        result[component] = emit.load(ref, *prefix, start + component)
    return result


def _ir_values(values: _Values, /) -> list[ir.Value]:
    """The MLIR scalars of one scalarized array, in row-major order."""

    from jax.extend.mlir import ir as mlir

    flat: list[ir.Value] = []
    for value in values.reshape(-1).tolist():
        if not isinstance(value, mlir.Value):
            raise TypeError("Scalarized kernel values must be MLIR values.")
        flat.append(value)
    return flat


def _store(
    emit: _Emitter,
    values: _Values,
    ref: ir.Value,
    prefix: Sequence[ir.Value | int],
    start: int,
    /,
) -> None:
    for component, value in enumerate(_ir_values(values)):
        emit.store(value, ref, *prefix, start + component)


def _zero(
    emit: _Emitter,
    ref: ir.Value,
    prefix: Sequence[ir.Value | int],
    start: int,
    size: int,
    /,
) -> None:
    for component in range(size):
        emit.store(emit.zero, ref, *prefix, start + component)


def _scalar(value: object, /) -> _Values:
    result = _object_array(())
    result[()] = value
    return result


def _int32(values: Sequence[int], /) -> Array:
    return jnp.asarray(np.asarray(values, dtype=np.int32).reshape(-1))


def _edge_arrays(
    topology: KinematicTopology, codes: Sequence[int], /
) -> tuple[Array, Array, Array, Array]:
    """Int32 parents, children, coordinates (zero when fixed) and program codes."""

    coordinates = tuple(
        coordinate if branch != _FIXED else 0
        for branch, coordinate in zip(
            topology.branches, topology.coordinates, strict=True
        )
    )
    return (
        _int32(topology.parents),
        _int32(topology.children),
        _int32(coordinates),
        _int32(codes),
    )


class _Edge(NamedTuple):
    parent: ir.Value
    child: ir.Value
    coordinate: ir.Value
    code: ir.Value


def _edge(emit: _Emitter, refs: Sequence[ir.Value], edge: ir.Value, /) -> _Edge:
    from jax.extend.mlir.dialects import arith

    parent, child, coordinate = (
        arith.index_cast(emit.index, emit.load(ref, edge)) for ref in refs[:3]
    )
    return _Edge(parent, child, coordinate, emit.load(refs[3], edge))


def _is(emit: _Emitter, code: ir.Value, value: int, /) -> ir.Value:
    from jax.experimental.mosaic import gpu as mgpu
    from jax.extend.mlir.dialects import arith

    return arith.cmpi(arith.CmpIPredicate.eq, code, mgpu.c(value, emit.integer))


def _is_zero(emit: _Emitter, value: ir.Value, /) -> ir.Value:
    from jax.extend.mlir.dialects import arith

    return arith.cmpi(arith.CmpIPredicate.eq, value, emit.constant(0))


def _poses_native(*operands: Array, topology: KinematicTopology) -> tuple[Array, ...]:
    configuration = operands[6]
    worlds = _world_count([operand.shape for operand in operands], _POSE_ROLES)
    bodies = operands[0].shape[1]
    edges = operands[2].shape[1]
    size = configuration.shape[1]
    dtype = np.dtype(operands[0].dtype)
    branches = topology.branches
    edge_arrays = _edge_arrays(topology, branches)
    if not _admitted(
        floating=operands,
        indices=edge_arrays,
        worlds=worlds,
        bodies=bodies,
        edges=edges,
        lanes=worlds,
        columns=1,
        buffers=(worlds * bodies * 4, worlds * edges * 4, worlds * size),
    ):
        return _poses_reference(*operands, topology=topology)
    spans = _spans_worlds(operands, _POSE_ROLES, worlds)
    numeric = operands[:6] + ((configuration,) if size else ())
    kernel_operands = numeric + edge_arrays

    def body(emit: _Emitter, world: ir.Value, refs: tuple[ir.Value, ...]) -> None:
        at = _world_reader(emit, world, spans)
        configuration_ref = refs[6] if size else None
        edge_refs = refs[len(numeric) : len(kernel_operands)]
        position_ref, orientation_ref = refs[len(kernel_operands) :]
        ops = _MosaicScalars()
        output = (world,)

        @emit.loop(bodies)
        def _(body_index: ir.Value) -> None:
            _store(
                emit,
                _load(emit, refs[0], (at(0), body_index), 0, 3),
                position_ref,
                (*output, body_index),
                0,
            )
            _store(
                emit,
                _load(emit, refs[1], (at(1), body_index), 0, 4),
                orientation_ref,
                (*output, body_index),
                0,
            )

        @emit.loop(edges)
        def _(edge_index: ir.Value) -> None:
            edge = _edge(emit, edge_refs, edge_index)
            inputs = (
                _load(emit, position_ref, (*output, edge.parent), 0, 3),
                _load(emit, orientation_ref, (*output, edge.parent), 0, 4),
                _load(emit, refs[2], (at(2), edge_index), 0, 3),
                _load(emit, refs[3], (at(3), edge_index), 0, 4),
                _load(emit, refs[4], (at(4), edge_index), 0, 3),
                _load(emit, refs[5], (at(5), edge_index), 0, 3),
            )
            for branch in sorted(set(branches)):
                with emit.when(_is(emit, edge.code, branch)):
                    if branch == _FIXED or configuration_ref is None:
                        coordinate = _scalar(ops.literal(0.0, dtype))
                    else:
                        coordinate = _scalar(
                            emit.load(configuration_ref, at(6), edge.coordinate)
                        )
                    child_position, child_orientation = _evaluate(
                        _primal_program("pose", branch, dtype), ops, *inputs, coordinate
                    )
                    _store(emit, child_position, position_ref, (*output, edge.child), 0)
                    _store(
                        emit, child_orientation, orientation_ref, (*output, edge.child), 0
                    )

    itemsize = dtype.itemsize
    out_type = (
        jax.ShapeDtypeStruct((worlds, bodies, 3), dtype),
        jax.ShapeDtypeStruct((worlds, bodies, 4), dtype),
    )
    return _launch(
        body,
        kernel_operands,
        out_type,
        dtype=dtype,
        index_dtype=np.dtype(np.int32),
        lanes=worlds,
        flops=worlds * edges * 160,
        bytes_accessed=worlds
        * (itemsize * (14 * bodies + edges * (14 + 14 + 1)) + edges * 16),
    )


def _velocity_native(*operands: Array, topology: KinematicTopology) -> tuple[Array, ...]:
    velocity = operands[_LINEAR]
    worlds = _world_count([operand.shape for operand in operands], _VELOCITY_ROLES)
    bodies = operands[2].shape[1]
    edges = operands[0].shape[1]
    _, columns, size = velocity.shape
    dtype = np.dtype(operands[2].dtype)
    branches = topology.branches
    edge_arrays = _edge_arrays(topology, branches)
    if not _admitted(
        floating=operands,
        indices=edge_arrays,
        worlds=worlds,
        bodies=bodies,
        edges=edges,
        lanes=worlds * columns,
        columns=columns,
        buffers=(
            worlds * columns * bodies * 6,
            worlds * columns * size,
            worlds * bodies * 4,
        ),
    ):
        return _velocity_reference(*operands, topology=topology)
    spans = _spans_worlds(operands, _VELOCITY_ROLES, worlds)
    numeric = operands[:4] + ((velocity,) if size else ())
    kernel_operands = numeric + edge_arrays

    def body(emit: _Emitter, lane: ir.Value, refs: tuple[ir.Value, ...]) -> None:
        world, column = emit.split(lane, columns)
        at = _world_reader(emit, world, spans)
        velocity_ref = refs[4] if size else None
        edge_refs = refs[len(numeric) : len(kernel_operands)]
        (twist_ref,) = refs[len(kernel_operands) :]
        ops = _MosaicScalars()
        lane_prefix = (world, column)

        @emit.loop(bodies)
        def _(body_index: ir.Value) -> None:
            _zero(emit, twist_ref, (*lane_prefix, body_index), 0, 6)

        @emit.loop(edges)
        def _(edge_index: ir.Value) -> None:
            edge = _edge(emit, edge_refs, edge_index)
            inputs = (
                _load(emit, twist_ref, (*lane_prefix, edge.parent), 0, 3),
                _load(emit, twist_ref, (*lane_prefix, edge.parent), 3, 3),
                _load(emit, refs[2], (at(2), edge.parent), 0, 3),
                _load(emit, refs[2], (at(2), edge.child), 0, 3),
                _load(emit, refs[3], (at(3), edge.parent), 0, 4),
                _load(emit, refs[0], (at(0), edge_index), 0, 3),
                _load(emit, refs[1], (at(1), edge_index), 0, 3),
            )
            for branch in sorted(set(branches)):
                with emit.when(_is(emit, edge.code, branch)):
                    if branch == _FIXED or velocity_ref is None:
                        rate = _scalar(ops.literal(0.0, dtype))
                    else:
                        rate = _scalar(
                            emit.load(velocity_ref, at(4), column, edge.coordinate)
                        )
                    child_linear, child_angular = _evaluate(
                        _primal_program("velocity", branch, dtype), ops, rate, *inputs
                    )
                    _store(emit, child_linear, twist_ref, (*lane_prefix, edge.child), 0)
                    _store(emit, child_angular, twist_ref, (*lane_prefix, edge.child), 3)

    itemsize = dtype.itemsize
    lanes = worlds * columns
    out_type = (jax.ShapeDtypeStruct((worlds, columns, bodies, 6), dtype),)
    return _launch(
        body,
        kernel_operands,
        out_type,
        dtype=dtype,
        index_dtype=np.dtype(np.int32),
        lanes=lanes,
        flops=lanes * edges * 90,
        bytes_accessed=lanes
        * (itemsize * (6 * bodies + edges * (6 + 6 + 3 + 3 + 4 + 6 + 1)) + edges * 16),
    )


def _velocity_transpose_native(
    *operands: Array,
    topology: KinematicTopology,
    velocity_size: int,
    velocity_dtype: np.dtype,
) -> tuple[Array, ...]:
    cotangent = operands[_LINEAR]
    worlds = _world_count([operand.shape for operand in operands], _VELOCITY_ROLES)
    bodies = operands[2].shape[1]
    edges = operands[0].shape[1]
    columns = cotangent.shape[1]
    dtype = np.dtype(operands[2].dtype)
    branches = topology.branches
    edge_arrays = _edge_arrays(topology, branches)
    if (
        np.dtype(velocity_dtype) != dtype
        or velocity_size == 0
        or not _admitted(
            floating=operands,
            indices=edge_arrays,
            worlds=worlds,
            bodies=bodies,
            edges=edges,
            lanes=worlds * columns,
            columns=columns,
            buffers=(
                worlds * columns * bodies * 6,
                worlds * columns * velocity_size,
                worlds * bodies * 4,
            ),
        )
    ):
        return _velocity_transpose_reference(
            *operands,
            topology=topology,
            velocity_size=velocity_size,
            velocity_dtype=velocity_dtype,
        )
    spans = _spans_worlds(operands, _VELOCITY_ROLES, worlds)
    kernel_operands = tuple(operands) + edge_arrays

    def body(emit: _Emitter, lane: ir.Value, refs: tuple[ir.Value, ...]) -> None:
        world, column = emit.split(lane, columns)
        at = _world_reader(emit, world, spans)
        edge_refs = refs[5:9]
        result_ref, workspace_ref = refs[9:]
        ops = _MosaicScalars()
        lane_prefix = (world, column)

        @emit.loop(bodies)
        def _(body_index: ir.Value) -> None:
            _store(
                emit,
                _load(emit, refs[4], (at(4), column, body_index), 0, 6),
                workspace_ref,
                (*lane_prefix, body_index),
                0,
            )

        @emit.loop(velocity_size)
        def _(index: ir.Value) -> None:
            emit.store(emit.zero, result_ref, world, column, index)

        @emit.loop(edges)
        def _(step: ir.Value) -> None:
            edge_index = emit.reversed_edge(step, edges)
            edge = _edge(emit, edge_refs, edge_index)
            # Transpose of the child store: release its cotangent and clear it.
            released = _load(emit, workspace_ref, (*lane_prefix, edge.child), 0, 6)
            _zero(emit, workspace_ref, (*lane_prefix, edge.child), 0, 6)
            inputs = (
                _load(emit, refs[2], (at(2), edge.parent), 0, 3),
                _load(emit, refs[2], (at(2), edge.child), 0, 3),
                _load(emit, refs[3], (at(3), edge.parent), 0, 4),
                _load(emit, refs[0], (at(0), edge_index), 0, 3),
                _load(emit, refs[1], (at(1), edge_index), 0, 3),
                released[:3],
                released[3:],
            )
            for branch in sorted(set(branches)):
                with emit.when(_is(emit, edge.code, branch)):
                    rate_bar, linear_bar, angular_bar = _evaluate(
                        _transpose_program(branch, dtype), ops, *inputs
                    )
                    parent_bar = _ir_values(linear_bar) + _ir_values(angular_bar)
                    for component, update in enumerate(parent_bar):
                        emit.store(
                            emit.add(
                                emit.load(
                                    workspace_ref, world, column, edge.parent, component
                                ),
                                update,
                            ),
                            workspace_ref,
                            world,
                            column,
                            edge.parent,
                            component,
                        )
                    # Fixed edges never read the velocity.
                    if branch != _FIXED:
                        (rate_update,) = _ir_values(rate_bar)
                        emit.store(
                            emit.add(
                                emit.load(result_ref, world, column, edge.coordinate),
                                rate_update,
                            ),
                            result_ref,
                            world,
                            column,
                            edge.coordinate,
                        )

    itemsize = dtype.itemsize
    lanes = worlds * columns
    out_type = (
        jax.ShapeDtypeStruct((worlds, columns, velocity_size), dtype),
        jax.ShapeDtypeStruct((worlds, columns, bodies, 6), dtype),
    )
    (result, _) = _launch(
        body,
        kernel_operands,
        out_type,
        dtype=dtype,
        index_dtype=np.dtype(np.int32),
        lanes=lanes,
        flops=lanes * edges * 120,
        bytes_accessed=lanes
        * (
            itemsize * (12 * bodies + velocity_size + edges * (6 * 3 + 3 + 3 + 4 + 6 + 2))
            + edges * 16
        ),
    )
    return (result,)


def _motion_native(*operands: Array, topology: KinematicTopology) -> tuple[Array, ...]:
    configuration, rates, directions = operands[6:]
    worlds = _world_count([operand.shape for operand in operands], _MOTION_ROLES)
    bodies = operands[0].shape[1]
    edges = operands[2].shape[1]
    size = configuration.shape[1]
    _, columns, rate_size = rates.shape
    dtype = np.dtype(operands[0].dtype)
    probe = _edge_arrays(topology, topology.branches)
    if not _admitted(
        floating=operands,
        indices=probe,
        worlds=worlds,
        bodies=bodies,
        edges=edges,
        lanes=worlds * columns,
        columns=columns,
        buffers=(
            worlds * columns * bodies * _POSE_WIDTH,
            worlds * columns * max(size, rate_size),
            worlds * edges * 4,
        ),
    ):
        return _motion_reference(*operands, topology=topology)
    pose_stage, twist_stage = _motion_schedule(topology, dtype)
    pose_arrays = _edge_arrays(topology, pose_stage.codes)
    twist_codes = jnp.asarray(np.asarray(twist_stage.codes, dtype=np.int32))
    spans = _spans_worlds(operands, _MOTION_ROLES, worlds)
    present = tuple(
        position for position, used in ((6, size), (7, rate_size), (8, size)) if used
    )
    numeric = operands[:6] + tuple(operands[position] for position in present)
    kernel_operands = numeric + pose_arrays + (twist_codes,)
    slot = {position: 6 + offset for offset, position in enumerate(present)}

    def body(emit: _Emitter, lane: ir.Value, refs: tuple[ir.Value, ...]) -> None:
        world, column = emit.split(lane, columns)
        at = _world_reader(emit, world, spans)
        edge_refs = refs[len(numeric) : len(numeric) + 4]
        twist_code_ref = refs[len(numeric) + 4]
        position_ref, orientation_ref, twist_ref, derivative_ref, pose_ref = refs[
            len(kernel_operands) :
        ]
        ops = _MosaicScalars()
        zero = _scalar(ops.literal(0.0, dtype))
        lane_prefix = (world, column)

        def operand(position: int) -> ir.Value:
            return refs[slot[position]]

        @emit.loop(bodies)
        def _(body_index: ir.Value) -> None:
            pose = (*lane_prefix, body_index)
            _store(
                emit, _load(emit, refs[0], (at(0), body_index), 0, 3), pose_ref, pose, 0
            )
            _store(
                emit, _load(emit, refs[1], (at(1), body_index), 0, 4), pose_ref, pose, 3
            )
            _zero(emit, pose_ref, pose, 7, _POSE_WIDTH - 7)

        # Pose recursion and its activity-specialized tangent.
        @emit.loop(edges)
        def _(edge_index: ir.Value) -> None:
            edge = _edge(emit, edge_refs, edge_index)
            parent = (*lane_prefix, edge.parent)
            child = (*lane_prefix, edge.child)
            primals = (
                _load(emit, pose_ref, parent, 0, 3),
                _load(emit, pose_ref, parent, 3, 4),
                _load(emit, refs[2], (at(2), edge_index), 0, 3),
                _load(emit, refs[3], (at(3), edge_index), 0, 4),
                _load(emit, refs[4], (at(4), edge_index), 0, 3),
                _load(emit, refs[5], (at(5), edge_index), 0, 3),
            )
            for code, (program, carries, branch) in enumerate(
                zip(
                    pose_stage.programs,
                    pose_stage.carries,
                    pose_stage.branches,
                    strict=True,
                )
            ):
                with emit.when(_is(emit, edge.code, code)):
                    if branch == _FIXED:
                        coordinate, coordinate_dot = zero, zero
                    else:
                        coordinate = _scalar(
                            emit.load(operand(6), at(6), edge.coordinate)
                        )
                        coordinate_dot = _scalar(
                            emit.load(operand(8), at(8), column, edge.coordinate)
                        )
                    tangents: list[_Values] = []
                    if program.inputs[0]:
                        tangents.append(_load(emit, pose_ref, parent, 7, 3))
                    if program.inputs[1]:
                        tangents.append(_load(emit, pose_ref, parent, 10, 4))
                    if program.inputs[6]:
                        tangents.append(coordinate_dot)
                    values = _evaluate(
                        program.closed, ops, *primals, coordinate, *tangents
                    )
                    _store(emit, values[0], pose_ref, child, 0)
                    _store(emit, values[1], pose_ref, child, 3)
                    for output, (offset, width) in enumerate(((7, 3), (10, 4))):
                        if program.outputs[output]:
                            _store(emit, values[2 + output], pose_ref, child, offset)
                        elif carries[output]:
                            _zero(emit, pose_ref, child, offset, width)

        # Every lane of a world traverses the same poses; column zero publishes.
        with emit.when(_is_zero(emit, column)):

            @emit.loop(bodies)
            def _(body_index: ir.Value) -> None:
                pose = (*lane_prefix, body_index)
                _store(
                    emit,
                    _load(emit, pose_ref, pose, 0, 3),
                    position_ref,
                    (world, body_index),
                    0,
                )
                _store(
                    emit,
                    _load(emit, pose_ref, pose, 3, 4),
                    orientation_ref,
                    (world, body_index),
                    0,
                )

        @emit.loop(bodies)
        def _(body_index: ir.Value) -> None:
            _zero(emit, twist_ref, (*lane_prefix, body_index), 0, 6)
            _zero(emit, derivative_ref, (*lane_prefix, body_index), 0, 6)

        # Twist recursion and its activity-specialized tangent.
        @emit.loop(edges)
        def _(edge_index: ir.Value) -> None:
            edge = _edge(emit, edge_refs, edge_index)
            code_value = emit.load(twist_code_ref, edge_index)
            parent = (*lane_prefix, edge.parent)
            child = (*lane_prefix, edge.child)
            primals = (
                _load(emit, twist_ref, parent, 0, 3),
                _load(emit, twist_ref, parent, 3, 3),
                _load(emit, pose_ref, parent, 0, 3),
                _load(emit, pose_ref, child, 0, 3),
                _load(emit, pose_ref, parent, 3, 4),
                _load(emit, refs[4], (at(4), edge_index), 0, 3),
                _load(emit, refs[5], (at(5), edge_index), 0, 3),
            )
            for code, (program, carries, branch) in enumerate(
                zip(
                    twist_stage.programs,
                    twist_stage.carries,
                    twist_stage.branches,
                    strict=True,
                )
            ):
                with emit.when(_is(emit, code_value, code)):
                    rate = (
                        zero
                        if branch == _FIXED
                        else _scalar(
                            emit.load(operand(7), at(7), column, edge.coordinate)
                        )
                    )
                    tangents: list[_Values] = []
                    for position, (source, prefix, first, width) in enumerate(
                        (
                            (derivative_ref, parent, 0, 3),
                            (derivative_ref, parent, 3, 3),
                            (pose_ref, parent, 7, 3),
                            (pose_ref, child, 7, 3),
                            (pose_ref, parent, 10, 4),
                        ),
                        start=1,
                    ):
                        if program.inputs[position]:
                            tangents.append(_load(emit, source, prefix, first, width))
                    values = _evaluate(program.closed, ops, rate, *primals, *tangents)
                    _store(emit, values[0], twist_ref, child, 0)
                    _store(emit, values[1], twist_ref, child, 3)
                    for output, offset in enumerate((0, 3)):
                        if program.outputs[output]:
                            _store(
                                emit, values[2 + output], derivative_ref, child, offset
                            )
                        elif carries[output]:
                            _zero(emit, derivative_ref, child, offset, 3)

    itemsize = dtype.itemsize
    lanes = worlds * columns
    out_type = (
        jax.ShapeDtypeStruct((worlds, bodies, 3), dtype),
        jax.ShapeDtypeStruct((worlds, bodies, 4), dtype),
        jax.ShapeDtypeStruct((worlds, columns, bodies, 6), dtype),
        jax.ShapeDtypeStruct((worlds, columns, bodies, 6), dtype),
        jax.ShapeDtypeStruct((worlds, columns, bodies, _POSE_WIDTH), dtype),
    )
    position, orientation, twists, derivatives, _ = _launch(
        body,
        kernel_operands,
        out_type,
        dtype=dtype,
        index_dtype=np.dtype(np.int32),
        lanes=lanes,
        flops=lanes * edges * 700,
        bytes_accessed=lanes
        * (
            itemsize * (bodies * (2 * _POSE_WIDTH + 24) + edges * (28 + 14 + 2 + 36 + 13))
            + edges * 20
        ),
    )
    return position, orientation, twists, derivatives


# ---------------------------------------------------------------------------
# Primitives, derivatives, transposes and batching.
# ---------------------------------------------------------------------------

_poses_p = _primitive(
    "phydrax_articulated_poses", _poses_abstract, _poses_reference, _poses_native
)
_velocity_p = _primitive(
    "phydrax_articulated_body_velocity",
    _velocity_abstract,
    _velocity_reference,
    _velocity_native,
)
_velocity_transpose_p = _primitive(
    "phydrax_articulated_body_velocity_transpose",
    _velocity_transpose_abstract,
    _velocity_transpose_reference,
    _velocity_transpose_native,
)
_motion_p = _primitive(
    "phydrax_articulated_motion", _motion_abstract, _motion_reference, _motion_native
)


def _jvp_rule(
    primitive: core.Primitive,
    reference: Callable[..., tuple[Array, ...]],
    roles: tuple[_Role, ...],
) -> Callable[..., tuple[tuple[Array, ...], tuple[Tangent, ...]]]:
    """Native primal; tangents differentiate the unrolled reference."""

    def rule(
        primals: tuple[Array, ...], tangents: tuple[Tangent, ...], **params: object
    ) -> tuple[tuple[Array, ...], tuple[Tangent, ...]]:
        return _reference_jvp(primitive, reference, roles, primals, tangents, **params)

    return rule


def _linear_operand(operands: Sequence[Primal], /) -> ShapedArray:
    """The array type of the one undefined (linear) operand of a transpose."""

    undefined = [
        position
        for position, value in enumerate(operands)
        if isinstance(value, ad.UndefinedPrimal)
    ]
    if undefined != [_LINEAR]:
        raise ValueError(
            "Articulated velocity actions are linear only in their velocity or "
            "cotangent columns; poses and geometry cannot be transposed."
        )
    linear = operands[_LINEAR]
    assert isinstance(linear, ad.UndefinedPrimal)
    aval = linear.aval
    if not isinstance(aval, ShapedArray):
        raise TypeError("Articulated velocity transposes need array operands.")
    return aval


def _world_sum(value: Array, aval: ShapedArray, /) -> Array:
    """Sum a per-world transpose over the worlds sharing one operand."""
    if value.shape != aval.shape:
        value = jnp.sum(value, axis=0, keepdims=True)
    return value


def _velocity_transpose_rule(
    cotangents: Sequence[Tangent], *operands: Primal, topology: KinematicTopology
) -> tuple[Tangent | None, ...]:
    velocity = _linear_operand(operands)
    (cotangent,) = cotangents
    if isinstance(cotangent, ad.Zero):
        result: Tangent = ad.Zero(velocity)
    else:
        (value,) = _velocity_transpose_p.bind(
            *operands[:_LINEAR],
            cotangent,
            topology=topology,
            velocity_size=velocity.shape[-1],
            velocity_dtype=np.dtype(velocity.dtype),
        )
        result = _world_sum(value, velocity)
    return (None,) * _LINEAR + (result,)


def _velocity_transpose_transpose(
    cotangents: Sequence[Tangent],
    *operands: Primal,
    topology: KinematicTopology,
    velocity_size: int,
    velocity_dtype: np.dtype,
) -> tuple[Tangent | None, ...]:
    """Transpose of the pullback: the linear part ``J u`` of the twist action.

    The twist action can carry ``u``-independent constants (``0 * NaN`` from
    structurally constant carries) that the pullback excludes, so its
    transpose is the owning reference's linear transpose rather than the
    primal action. Operands shared across worlds are reduced by that
    transpose.
    """

    body_cotangent = _linear_operand(operands)
    (cotangent,) = cotangents
    if isinstance(cotangent, ad.Zero):
        return (None,) * _LINEAR + (ad.Zero(body_cotangent),)
    fixed = tuple(
        value for value in operands[:_LINEAR] if not isinstance(value, ad.UndefinedPrimal)
    )

    def pullback(value: Array) -> Array:
        return _velocity_transpose_reference(
            *fixed,
            value,
            topology=topology,
            velocity_size=velocity_size,
            velocity_dtype=velocity_dtype,
        )[0]

    (result,) = jax.linear_transpose(
        pullback, jax.ShapeDtypeStruct(body_cotangent.shape, body_cotangent.dtype)
    )(cotangent)
    return (None,) * _LINEAR + (result,)


def _batch_rule(
    primitive: core.Primitive,
    reference: Callable[..., tuple[Array, ...]],
    roles: tuple[_Role, ...],
) -> Callable[..., tuple[tuple[Array, ...], tuple[int | None, ...]]]:
    def rule(
        operands: tuple[Array, ...], axes: tuple[int | None, ...], **params: object
    ) -> tuple[tuple[Array, ...], tuple[int | None, ...]]:
        return _world_batch(primitive, reference, roles, operands, axes, **params)

    return rule


for _p, _reference, _roles in (
    (_poses_p, _poses_reference, _POSE_ROLES),
    (_velocity_p, _velocity_reference, _VELOCITY_ROLES),
    (_velocity_transpose_p, _velocity_transpose_reference, _VELOCITY_ROLES),
    (_motion_p, _motion_reference, _MOTION_ROLES),
):
    ad.primitive_jvps[_p] = _jvp_rule(_p, _reference, _roles)
    batching.primitive_batchers[_p] = _batch_rule(_p, _reference, _roles)
ad.primitive_transposes[_velocity_p] = _velocity_transpose_rule
ad.primitive_transposes[_velocity_transpose_p] = _velocity_transpose_transpose
del _p, _reference, _roles


# ---------------------------------------------------------------------------
# Single-world interfaces.
# ---------------------------------------------------------------------------


def _real(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must be a real floating array.")
    return array


type _Geometry = tuple[Array, Array, Array, Array, Array, Array]


def _tree_geometry(
    reference_position: ArrayLike,
    reference_orientation: ArrayLike,
    translation: ArrayLike,
    relative_orientation: ArrayLike,
    axes: ArrayLike,
    anchors: ArrayLike,
    /,
) -> _Geometry:
    """The six numerical tree operands with a shared leading world axis."""

    return (
        _real(reference_position, "reference_position")[None],
        _real(reference_orientation, "reference_orientation")[None],
        _real(translation, "translation")[None],
        _real(relative_orientation, "relative_orientation")[None],
        _real(axes, "axes")[None],
        _real(anchors, "anchors")[None],
    )


def articulated_poses(
    topology: KinematicTopology,
    reference_position: ArrayLike,
    reference_orientation: ArrayLike,
    translation: ArrayLike,
    relative_orientation: ArrayLike,
    axes: ArrayLike,
    anchors: ArrayLike,
    configuration: ArrayLike,
    /,
) -> tuple[Array, Array]:
    """Forward pose recursion of one tree.

    Shapes: reference pose ``(C, 3)``/``(C, 4)``, parent-frame edge
    translation ``(E, 3)``, relative orientation ``(E, 4)``, joint axes and
    anchors ``(E, 3)`` and configuration ``(Q,)``. Returns body positions
    ``(C, 3)`` and orientations ``(C, 4)``; bodies that are never a child keep
    their reference pose.
    """

    point = _real(configuration, "configuration")
    if point.ndim != 1:
        raise ValueError("configuration must be a rank-1 coordinate array.")
    position, orientation = _poses_p.bind(
        *_tree_geometry(
            reference_position,
            reference_orientation,
            translation,
            relative_orientation,
            axes,
            anchors,
        ),
        point[None],
        topology=_require_topology(topology),
    )
    return position[0], orientation[0]


def articulated_body_velocity(
    topology: KinematicTopology,
    axes: ArrayLike,
    anchors: ArrayLike,
    position: ArrayLike,
    orientation: ArrayLike,
    generalized_velocity: ArrayLike,
    /,
) -> Array:
    """``(K, C, 6)`` world twists of ``(K, V)`` generalized-velocity columns.

    The action is linear in ``generalized_velocity``; ``jax.linear_transpose``
    binds the native transpose. Unvisited bodies have zero twist.
    """

    velocity = _real(generalized_velocity, "generalized_velocity")
    if velocity.ndim != 2:
        raise ValueError("generalized_velocity must have (column, velocity) shape.")
    (twists,) = _velocity_p.bind(
        _real(axes, "axes")[None],
        _real(anchors, "anchors")[None],
        _real(position, "position")[None],
        _real(orientation, "orientation")[None],
        velocity[None],
        topology=_require_topology(topology),
    )
    return twists[0]


def articulated_motion(
    topology: KinematicTopology,
    reference_position: ArrayLike,
    reference_orientation: ArrayLike,
    translation: ArrayLike,
    relative_orientation: ArrayLike,
    axes: ArrayLike,
    anchors: ArrayLike,
    configuration: ArrayLike,
    rates: ArrayLike,
    directions: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Shared poses with independent twist and directional-derivative columns.

    Operands are those of :func:`articulated_poses` plus ``(K, V)`` rate and
    ``(K, Q)`` direction columns in the configuration dtype. Returns positions
    ``(C, 3)``, orientations ``(C, 4)``, twists ``(K, C, 6)`` with
    ``twists[k] = J(q) rates[k]`` and derivatives ``(K, C, 6)`` equal to
    ``jax.jvp`` of ``q -> J(q) rates[k]`` along ``directions[k]``.
    """

    point = _real(configuration, "configuration")
    rate = _real(rates, "rates")
    direction = _real(directions, "directions")
    if point.ndim != 1 or rate.ndim != 2 or direction.ndim != 2:
        raise ValueError("configuration must be (Q,) and rates/directions (K, V)/(K, Q).")
    position, orientation, twists, derivatives = _motion_p.bind(
        *_tree_geometry(
            reference_position,
            reference_orientation,
            translation,
            relative_orientation,
            axes,
            anchors,
        ),
        point[None],
        rate[None],
        direction[None],
        topology=_require_topology(topology),
    )
    return position[0], orientation[0], twists[0], derivatives[0]


__all__ = [
    "KinematicTopology",
    "articulated_body_velocity",
    "articulated_motion",
    "articulated_poses",
]
