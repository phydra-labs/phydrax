#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Coarse articulated kinematics against the historical unrolled recursion.

The oracle is the original per-edge execution: a Python loop over static
edges evaluated and differentiated by JAX (``jax.jvp``/``jax.grad``), so its
derivatives carry JAX's structural zeros. Poses, twists, transposes and
directional-derivative columns are compared with it on every visible device,
including nonfinite geometry and rates whose ``0 * NaN`` behaviour depends on
which carries are structurally constant; batched worlds against their
members; mixed precision against the oracle's per-carry rounding; and the
scalarized kernel steps against JAX evaluation of the same steps.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax._src.core import ShapedArray
from jax.extend import core as jax_core
from jax.typing import DTypeLike

from phydrax import config as pxconfig
from phydrax.discretization.particle._articulated_kinematics import (
    _derivative,
    _evaluate,
    _primal_program,
    _transpose_program,
    articulated_body_velocity,
    articulated_motion,
    articulated_poses,
    KinematicTopology,
)
from phydrax.discretization.particle._rigid_body import (
    _quaternion_increment,
    _quaternion_multiply,
    quaternion_rotation_matrix,
)
from phydrax.discretization.particle._rigid_joints import RigidJointKind
from phydrax.execution import default_device


pytestmark = pytest.mark.strict_jax

_DTYPES = pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64), ids=("float32", "float64")
)
# (edges, unused body capacity)
_TREES = pytest.mark.parametrize(
    "edges,unused", ((1, 0), (6, 2), (17, 3)), ids=("single", "branch", "deep")
)

_FIXED = int(RigidJointKind.FIXED)
_HINGE = int(RigidJointKind.HINGE)
_PRISMATIC = int(RigidJointKind.PRISMATIC)

type Geometry = tuple[Array, Array, Array, Array, Array, Array]


class Tree:
    """Host tree in arbitrary body storage order with unused body capacity."""

    def __init__(
        self,
        seed: int,
        edges: int,
        unused: int,
        *,
        kinds: tuple[int, ...] | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        bodies = edges + 1 + unused
        storage = [int(value) for value in rng.permutation(bodies)]
        parent_nodes = [int(rng.integers(0, edge + 1)) for edge in range(edges)]
        if kinds is None:
            cycle = (_HINGE, _FIXED, _PRISMATIC)
            kinds = tuple(
                cycle[edge] if edge < 3 else int(rng.choice(cycle))
                for edge in range(edges)
            )
        coordinates = []
        count = 0
        for kind in kinds:
            if kind in (_HINGE, _PRISMATIC):
                coordinates.append(count)
                count += 1
            else:
                coordinates.append(-1)
        quaternions = rng.normal(size=(bodies, 4))
        quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
        relative = rng.normal(size=(edges, 4))
        relative /= np.linalg.norm(relative, axis=-1, keepdims=True)
        axes = rng.normal(size=(edges, 3))
        axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
        self.size = count
        self.geometry = [
            rng.normal(size=(bodies, 3)),
            quaternions,
            rng.normal(size=(edges, 3)),
            relative,
            axes,
            rng.normal(size=(edges, 3)),
        ]
        self.topology = KinematicTopology(
            tuple(int(kind) for kind in kinds),
            tuple(coordinates),
            tuple(storage[node] for node in parent_nodes),
            tuple(storage[1 : edges + 1]),
        )
        self.bodies = bodies

    def cast(self, dtype: DTypeLike) -> Geometry:
        position, orientation, translation, relative, axes, anchors = (
            jnp.asarray(value, dtype=dtype) for value in self.geometry
        )
        return position, orientation, translation, relative, axes, anchors


def _normalize(value: Array) -> Array:
    norm = jnp.sqrt(jnp.sum(value * value, axis=-1, keepdims=True))
    return value / jnp.maximum(norm, jnp.finfo(value.dtype).tiny)


def _edges(tree: Tree) -> Any:
    topology = tree.topology
    return enumerate(
        zip(topology.kinds, topology.coordinates, topology.parents, topology.children)
    )


def _oracle_poses(
    tree: Tree, geometry: Geometry, configuration: Array
) -> tuple[Array, Array]:
    position, orientation, translation, relative, axes, anchors = geometry
    for edge, (kind, dof, parent, child) in _edges(tree):
        parent_orientation = orientation[parent]
        parent_rotation = quaternion_rotation_matrix(parent_orientation)
        relative_translation = translation[edge]
        relative_orientation = relative[edge]
        if kind == _HINGE:
            joint_orientation = _quaternion_increment(axes[edge] * configuration[dof])
            joint_rotation = quaternion_rotation_matrix(joint_orientation)
            arm = relative_translation - anchors[edge]
            relative_translation = anchors[edge] + joint_rotation @ arm
            relative_orientation = _quaternion_multiply(
                joint_orientation, relative_orientation
            )
        elif kind == _PRISMATIC:
            relative_translation = relative_translation + axes[edge] * configuration[dof]
        child_position = position[parent] + parent_rotation @ relative_translation
        child_orientation = _normalize(
            _quaternion_multiply(parent_orientation, relative_orientation)
        )
        position = position.at[child].set(child_position.astype(position.dtype))
        orientation = orientation.at[child].set(
            child_orientation.astype(orientation.dtype)
        )
    return position, orientation


def _oracle_velocity(
    tree: Tree,
    geometry: Geometry,
    velocity: Array,
    position: Array,
    orientation: Array,
) -> Array:
    axes, anchors = geometry[4], geometry[5]
    linear = jnp.zeros_like(position)
    angular = jnp.zeros_like(position)
    for edge, (kind, dof, parent, child) in _edges(tree):
        parent_rotation = quaternion_rotation_matrix(orientation[parent])
        if kind == _HINGE:
            axis_world = parent_rotation @ axes[edge]
            anchor_world = position[parent] + parent_rotation @ anchors[edge]
            child_angular = angular[parent] + velocity[dof] * axis_world
            anchor_velocity = linear[parent] + jnp.cross(
                angular[parent], anchor_world - position[parent]
            )
            child_linear = anchor_velocity + jnp.cross(
                child_angular, position[child] - anchor_world
            )
        elif kind == _PRISMATIC:
            axis_world = parent_rotation @ axes[edge]
            child_angular = angular[parent]
            child_linear = (
                linear[parent]
                + jnp.cross(angular[parent], position[child] - position[parent])
                + velocity[dof] * axis_world
            )
        else:
            child_angular = angular[parent]
            child_linear = linear[parent] + jnp.cross(
                angular[parent], position[child] - position[parent]
            )
        linear = linear.at[child].set(child_linear.astype(linear.dtype))
        angular = angular.at[child].set(child_angular.astype(angular.dtype))
    return jnp.concatenate((linear, angular), axis=-1)


def _oracle_motion(
    tree: Tree,
    geometry: Geometry,
    configuration: Array,
    rates: Array,
    directions: Array,
) -> tuple[Array, Array, Array, Array]:
    """Poses plus the historical ``vmap`` of one unrolled ``jax.jvp`` per column."""

    position, orientation = _oracle_poses(tree, geometry, configuration)
    if rates.shape[0] == 0:
        empty = jnp.zeros((0, tree.bodies, 6), dtype=position.dtype)
        return position, orientation, empty, empty

    def column(rate: Array, direction: Array) -> tuple[Array, Array]:
        return jax.jvp(
            lambda point: _oracle_velocity(
                tree, geometry, rate, *_oracle_poses(tree, geometry, point)
            ),
            (configuration,),
            (direction,),
        )

    twists, derivatives = jax.vmap(column)(rates, directions)
    return position, orientation, twists, derivatives


def _assert_close(actual: Array, expected: Array, dtype: DTypeLike, name: str) -> None:
    """Equal nonfinite patterns and close finite values."""

    host = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(expected, dtype=np.float64)
    assert host.shape == reference.shape, name
    assert np.asarray(actual).dtype == np.dtype(dtype), name
    np.testing.assert_array_equal(np.isnan(host), np.isnan(reference), err_msg=name)
    finite = np.isfinite(reference)
    np.testing.assert_array_equal(np.isfinite(host), finite, err_msg=name)
    infinite = ~finite & ~np.isnan(reference)
    np.testing.assert_array_equal(host[infinite], reference[infinite], err_msg=name)
    scale = max(1.0, float(np.max(np.abs(reference[finite]), initial=0.0)))
    tolerance = 2e-3 if np.dtype(dtype) == np.dtype(np.float32) else 1e-9
    error = float(np.max(np.abs(host[finite] - reference[finite]), initial=0.0))
    assert error <= tolerance * scale, (name, error, scale)


def _columns(
    seed: int, tree: Tree, columns: int, dtype: DTypeLike
) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(size=(columns, tree.size)), dtype),
        jnp.asarray(rng.normal(size=(columns, tree.size)), dtype),
    )


def _devices() -> tuple[jax.Device, ...]:
    return (jax.devices("cpu")[0],) + tuple(
        device for device in jax.devices() if device.platform == "gpu"
    )


_DEVICES = pytest.mark.parametrize(
    "device", _devices(), ids=lambda device: f"{device.platform}-{device.id}"
)


def _checked_motion(
    device: jax.Device,
    tree: Tree,
    dtype: DTypeLike,
    configuration: Array,
    rates: Array,
    directions: Array,
    name: str,
) -> tuple[Array, Array, Array, Array]:
    """Check one backend's motion against the original arithmetic."""

    with default_device(device):
        geometry = jax.device_put(tree.cast(dtype), device)
        arguments = jax.device_put((configuration, rates, directions), device)
        motion = jax.jit(articulated_motion, static_argnums=0)(
            tree.topology, *geometry, *arguments
        )
        expected = _oracle_motion(tree, geometry, *arguments)
    for value, reference, part in zip(
        motion,
        expected,
        ("position", "orientation", "twists", "derivatives"),
        strict=True,
    ):
        assert value.devices() == {device}
        _assert_close(value, reference, dtype, f"{name} {part} {device}")
    return motion


@_DTYPES
@_TREES
@_DEVICES
def test_poses_twists_and_motion_match_the_unrolled_recursion(
    dtype: DTypeLike, edges: int, unused: int, device: jax.Device
) -> None:
    tree = Tree(edges + 10 * unused, edges, unused)
    geometry = tree.cast(dtype)
    rng = np.random.default_rng(3)
    configuration = jnp.asarray(rng.normal(size=tree.size), dtype)
    rates, directions = _columns(4, tree, 3, dtype)
    with default_device(device):
        operands = jax.device_put(geometry, device)
        configuration, rates = jax.device_put((configuration, rates), device)
        position, orientation = jax.jit(articulated_poses, static_argnums=0)(
            tree.topology, *operands, configuration
        )
        twists = jax.jit(articulated_body_velocity, static_argnums=0)(
            tree.topology, operands[4], operands[5], position, orientation, rates
        )
    expected_position, expected_orientation = _oracle_poses(tree, geometry, configuration)
    _assert_close(position, expected_position, dtype, f"position {device}")
    _assert_close(orientation, expected_orientation, dtype, f"orientation {device}")
    for column in range(3):
        _assert_close(
            twists[column],
            _oracle_velocity(tree, geometry, rates[column], position, orientation),
            dtype,
            f"twists {device}",
        )
    _checked_motion(device, tree, dtype, configuration, rates, directions, "motion")


@_DTYPES
@_DEVICES
def test_hinge_small_angle_series_and_zero_coordinates(
    dtype: DTypeLike, device: jax.Device
) -> None:
    tree = Tree(5, 4, 1, kinds=(_HINGE, _HINGE, _HINGE, _PRISMATIC))
    eps = float(jnp.finfo(dtype).eps)
    # Exact zero, below the series threshold, at it, and a prismatic zero.
    configuration = jnp.asarray([0.0, 0.25 * np.sqrt(eps), np.sqrt(eps), 0.0], dtype)
    rates = jnp.asarray([[0.0, 1.0, -2.0, 0.5], [0.0, 0.0, 0.0, 0.0]], dtype)
    directions = jnp.asarray([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], dtype)
    motion = _checked_motion(
        device, tree, dtype, configuration, rates, directions, "small angle"
    )
    assert not np.any(np.asarray(motion[2][1]))
    assert not np.any(np.asarray(motion[3][1]))


@_DTYPES
@_DEVICES
def test_prismatic_constant_orientation_omits_symbolic_zero_products(
    dtype: DTypeLike, device: jax.Device
) -> None:
    tree = Tree(1, 1, 0, kinds=(_PRISMATIC,))
    motion = _checked_motion(
        device,
        tree,
        dtype,
        jnp.asarray([0.3], dtype),
        jnp.asarray([[jnp.nan]], dtype),
        jnp.asarray([[1.0]], dtype),
        "prismatic",
    )
    child = tree.topology.children[0]
    np.testing.assert_array_equal(np.asarray(motion[3][0, child]), np.zeros(6))
    assert np.all(np.isnan(np.asarray(motion[2][0, child, :3])))


@_DTYPES
@_DEVICES
def test_hinge_active_orientation_preserves_numeric_zero_products(
    dtype: DTypeLike, device: jax.Device
) -> None:
    # The hinge makes the entire orientation carry active, including its root.
    tree = Tree(1, 1, 0, kinds=(_HINGE,))
    motion = _checked_motion(
        device,
        tree,
        dtype,
        jnp.asarray([0.3], dtype),
        jnp.asarray([[jnp.nan]], dtype),
        jnp.asarray([[1.0]], dtype),
        "hinge",
    )
    child = tree.topology.children[0]
    assert np.all(np.isnan(np.asarray(motion[3][0, child, 3:])))


@_DTYPES
@_DEVICES
def test_fixed_prefix_derivatives_stay_structurally_zero(
    dtype: DTypeLike, device: jax.Device
) -> None:
    tree = Tree(1, 2, 0, kinds=(_FIXED, _HINGE))
    tree.geometry[2] = tree.geometry[2].copy()
    tree.geometry[2][0, 0] = np.inf
    motion = _checked_motion(
        device,
        tree,
        dtype,
        jnp.asarray([0.3], dtype),
        jnp.asarray([[1.0]], dtype),
        jnp.asarray([[1.0]], dtype),
        "fixed prefix",
    )
    child = tree.topology.children[0]
    np.testing.assert_array_equal(np.asarray(motion[3][0, child]), np.zeros(6))


@_DTYPES
@_DEVICES
def test_fixed_tree_nonfinite_pose_keeps_zero_coordinate_derivatives(
    dtype: DTypeLike, device: jax.Device
) -> None:
    tree = Tree(2, 3, 1, kinds=(_FIXED, _FIXED, _FIXED))
    tree.geometry[2] = tree.geometry[2].copy()
    tree.geometry[2][1, 2] = np.inf
    motion = _checked_motion(
        device,
        tree,
        dtype,
        jnp.zeros((0,), dtype),
        jnp.zeros((2, 0), dtype),
        jnp.zeros((2, 0), dtype),
        "fixed",
    )
    np.testing.assert_array_equal(np.asarray(motion[3]), np.zeros((2, tree.bodies, 6)))


@_DEVICES
def test_zero_coordinate_tree_preserves_empty_dual_actions(device: jax.Device) -> None:
    dtype = jnp.float64
    tree = Tree(8, 5, 2, kinds=(_FIXED,) * 5)
    columns = jnp.zeros((2, 0), dtype)
    position, orientation, twists, derivatives = _checked_motion(
        device, tree, dtype, jnp.zeros((0,), dtype), columns, columns, "fixed tree"
    )
    assert not np.any(np.asarray(twists)) and not np.any(np.asarray(derivatives))
    with default_device(device):
        geometry = jax.device_put(tree.cast(dtype), device)
        columns = jax.device_put(columns, device)
        load = jnp.ones((2, tree.bodies, 6), dtype)
        (generalized,) = jax.linear_transpose(
            lambda value: articulated_body_velocity(
                tree.topology, geometry[4], geometry[5], position, orientation, value
            ),
            columns,
        )(load)
    assert generalized.shape == (2, 0)


@_DEVICES
def test_edge_free_tree_preserves_reference_poses(device: jax.Device) -> None:
    dtype = jnp.float64
    tree = Tree(9, 0, 2)
    with default_device(device):
        geometry = tree.cast(dtype)
        position, orientation, twists, derivatives = articulated_motion(
            tree.topology,
            *geometry,
            jnp.zeros((0,), dtype),
            jnp.zeros((1, 0), dtype),
            jnp.zeros((1, 0), dtype),
        )
    np.testing.assert_array_equal(position, geometry[0])
    np.testing.assert_array_equal(orientation, geometry[1])
    assert twists.shape == derivatives.shape == (1, tree.bodies, 6)
    assert not np.any(np.asarray(twists)) and not np.any(np.asarray(derivatives))


@_DEVICES
def test_zero_column_query_preserves_shared_poses(device: jax.Device) -> None:
    dtype = jnp.float64
    tree = Tree(10, 4, 1)
    configuration = jnp.ones((tree.size,), dtype)
    none = jnp.zeros((0, tree.size), dtype)
    _, _, twists, derivatives = _checked_motion(
        device, tree, dtype, configuration, none, none, "columnless"
    )
    assert twists.shape == derivatives.shape == (0, tree.bodies, 6)


@_DTYPES
@_DEVICES
def test_nonfinite_columns_never_mix_and_zero_products_propagate(
    dtype: DTypeLike, device: jax.Device
) -> None:
    tree = Tree(12, 6, 1)
    configuration = jnp.asarray(np.random.default_rng(13).normal(size=tree.size), dtype)
    rates, directions = _columns(14, tree, 4, dtype)
    rates = rates.at[1, 0].set(jnp.nan)
    directions = directions.at[2, 1].set(jnp.inf)
    # A zero rate along a nonfinite direction keeps the original zero-times-NaN.
    rates = rates.at[3].set(0.0)
    directions = directions.at[3, 0].set(jnp.nan)
    motion = _checked_motion(
        device, tree, dtype, configuration, rates, directions, "nonfinite"
    )
    twists, derivatives = np.asarray(motion[2]), np.asarray(motion[3])
    assert np.all(np.isfinite(twists[0])) and np.all(np.isfinite(derivatives[0]))
    assert not np.all(np.isfinite(twists[1]))
    assert np.all(np.isfinite(twists[2])) and not np.all(np.isfinite(derivatives[2]))
    assert np.all(np.isfinite(twists[3]))
    assert np.all(np.isfinite(np.asarray(motion[0])))


@_DTYPES
@_DEVICES
def test_nonfinite_configuration_reaches_zero_motion_columns(
    dtype: DTypeLike, device: jax.Device
) -> None:
    tree = Tree(12, 6, 1)
    configuration = (
        jnp.asarray(np.random.default_rng(13).normal(size=tree.size), dtype)
        .at[0]
        .set(jnp.nan)
    )
    rest = jnp.zeros((1, tree.size), dtype)
    motion = _checked_motion(device, tree, dtype, configuration, rest, rest, "poisoned")
    assert not np.all(np.isfinite(np.asarray(motion[0])))
    assert not np.all(np.isfinite(np.asarray(motion[2])))


@_DTYPES
@_DEVICES
def test_velocity_transpose_is_the_exact_adjoint(
    dtype: DTypeLike, device: jax.Device
) -> None:
    tree = Tree(21, 7, 2)
    geometry = tree.cast(dtype)
    topology = tree.topology
    rng = np.random.default_rng(23)
    configuration = jnp.asarray(rng.normal(size=tree.size), dtype)
    position, orientation = _oracle_poses(tree, geometry, configuration)
    load = jnp.asarray(rng.normal(size=(3, tree.bodies, 6)), dtype)
    velocity = jnp.asarray(rng.normal(size=(3, tree.size)), dtype)
    jacobian = jax.jacfwd(
        lambda value: _oracle_velocity(tree, geometry, value, position, orientation)
    )(jnp.zeros((tree.size,), dtype))
    expected = jnp.einsum("kcs,csv->kv", load, jacobian)

    def action(value: Array) -> Array:
        return articulated_body_velocity(
            topology, geometry[4], geometry[5], position, orientation, value
        )

    with default_device(device):
        (generalized,) = jax.jit(jax.linear_transpose(action, velocity))(
            jax.device_put(load, device)
        )
    _assert_close(generalized, expected, dtype, f"transpose {device}")
    # Virtual work: <w, J u> = <Jᵀ w, u> column by column.
    body_power = jnp.sum(load * action(velocity), axis=(1, 2))
    generalized_power = jnp.sum(generalized * velocity, axis=1)
    _assert_close(generalized_power, body_power, dtype, "virtual work")
    # The transpose of the transpose is the action again.
    (again,) = jax.linear_transpose(
        lambda value: jax.linear_transpose(action, velocity)(value)[0], load
    )(velocity)
    _assert_close(again, action(velocity), dtype, "double transpose")

    def pulled(q: Array) -> Array:
        pose = articulated_poses(topology, *geometry, q)
        (value,) = jax.linear_transpose(
            lambda rate: articulated_body_velocity(
                topology, geometry[4], geometry[5], *pose, rate
            ),
            velocity,
        )(load)
        return value

    def pulled_oracle(q: Array) -> Array:
        pose = _oracle_poses(tree, geometry, q)
        (value,) = jax.linear_transpose(
            lambda rate: jnp.stack(
                [_oracle_velocity(tree, geometry, column, *pose) for column in rate]
            ),
            velocity,
        )(load)
        return value

    # Derivatives of the pulled-back load in the configuration, forward and
    # reverse (the transpose primitive's own JVP and transpose).
    direction = jnp.asarray(rng.normal(size=tree.size), dtype)
    _, tangent = jax.jit(lambda q: jax.jvp(pulled, (q,), (direction,)))(configuration)
    _, expected_tangent = jax.jvp(pulled_oracle, (configuration,), (direction,))
    _assert_close(tangent, expected_tangent, dtype, "pullback tangent")
    weights = jnp.asarray(rng.normal(size=(3, tree.size)), dtype)
    reverse = jax.jit(jax.grad(lambda q: jnp.sum(weights * pulled(q))))(configuration)
    expected_reverse = jax.grad(lambda q: jnp.sum(weights * pulled_oracle(q)))(
        configuration
    )
    _assert_close(reverse, expected_reverse, dtype, "pullback reverse")


@_DTYPES
@_DEVICES
def test_pullback_transpose_excludes_velocity_independent_constants(
    dtype: DTypeLike, device: jax.Device
) -> None:
    """Transposing the pullback gives ``J u`` without the action's constants.

    A fixed edge whose child pose is NaN contributes the ``u``-independent
    ``0 * NaN`` constant to the twist action; the pullback and its transpose
    exclude it, leaving that body's linear-action twist exactly zero while a
    moving sibling still transports ``u``.
    """

    tree = Tree(97, 2, 0, kinds=(_FIXED, _HINGE))
    root = tree.topology.parents[0]
    tree.topology = KinematicTopology(
        (_FIXED, _HINGE), (-1, 0), (root, root), tree.topology.children
    )
    tree.geometry[2] = tree.geometry[2].copy()
    tree.geometry[2][0] = np.nan
    geometry = tree.cast(dtype)
    topology = tree.topology
    fixed_child, moving_child = topology.children
    rng = np.random.default_rng(98)
    configuration = jnp.asarray(rng.normal(size=tree.size), dtype)
    position, orientation = _oracle_poses(tree, geometry, configuration)
    velocity = jnp.asarray(rng.normal(size=(2, tree.size)), dtype)
    load = jnp.asarray(rng.normal(size=(2, tree.bodies, 6)), dtype)

    def action(value: Array) -> Array:
        return articulated_body_velocity(
            topology, geometry[4], geometry[5], position, orientation, value
        )

    def oracle_action(value: Array) -> Array:
        return jnp.stack(
            [
                _oracle_velocity(tree, geometry, column, position, orientation)
                for column in value
            ]
        )

    def double_transpose(
        function: Callable[[Array], Array],
    ) -> Callable[[Array], Array]:
        def linear_part(value: Array) -> Array:
            (result,) = jax.linear_transpose(
                lambda body: jax.linear_transpose(function, velocity)(body)[0], load
            )(value)
            return result

        return linear_part

    expected = double_transpose(oracle_action)(velocity)
    with default_device(device):
        primal = jax.jit(action)(jax.device_put(velocity, device))
        linear = jax.jit(double_transpose(action))(jax.device_put(velocity, device))
    assert np.all(np.isnan(np.asarray(primal[:, fixed_child, :3])))
    _assert_close(linear, expected, dtype, f"double transpose {device}")
    np.testing.assert_array_equal(np.asarray(linear[:, fixed_child]), np.zeros((2, 6)))
    assert np.all(np.isfinite(np.asarray(linear[:, moving_child])))


def test_higher_order_and_parameter_derivatives_follow_the_oracle() -> None:
    dtype = jnp.float64
    tree = Tree(31, 6, 1)
    geometry = tree.cast(dtype)
    topology = tree.topology
    rng = np.random.default_rng(32)
    configuration = jnp.asarray(rng.normal(size=tree.size), dtype)
    rates, directions = _columns(33, tree, 2, dtype)
    probe = jnp.asarray(rng.normal(size=(2, tree.bodies, 6)), dtype)
    pose_probe = jnp.asarray(rng.normal(size=(tree.bodies, 3)), dtype)

    def replaced(axes: Array, translation: Array) -> Geometry:
        position, orientation, _, relative, _, anchors = geometry
        return position, orientation, translation, relative, axes, anchors

    def native(q: Array, axes: Array, translation: Array) -> Array:
        parts = replaced(axes, translation)
        position, orientation, twists, derivatives = articulated_motion(
            topology, *parts, q, rates, directions
        )
        transported = articulated_body_velocity(
            topology, axes, parts[5], position, orientation, rates
        )
        return (
            jnp.sum(probe * (twists + derivatives + transported))
            + jnp.sum(pose_probe * position)
            + jnp.sum(orientation)
        )

    def oracle(q: Array, axes: Array, translation: Array) -> Array:
        parts = replaced(axes, translation)
        position, orientation, twists, derivatives = _oracle_motion(
            tree, parts, q, rates, directions
        )
        transported = jnp.stack(
            [_oracle_velocity(tree, parts, rate, position, orientation) for rate in rates]
        )
        return (
            jnp.sum(probe * (twists + derivatives + transported))
            + jnp.sum(pose_probe * position)
            + jnp.sum(orientation)
        )

    primals = (configuration, geometry[4], geometry[2])
    tangents = tuple(
        jnp.asarray(rng.normal(size=value.shape), dtype) for value in primals
    )
    value, tangent = jax.jvp(native, primals, tangents)
    expected_value, expected_tangent = jax.jvp(oracle, primals, tangents)
    np.testing.assert_allclose(value, expected_value, rtol=1e-10)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=1e-9)
    gradient = jax.jit(jax.grad(native, argnums=(0, 1, 2)))(*primals)
    expected_gradient = jax.grad(oracle, argnums=(0, 1, 2))(*primals)
    for left, right in zip(gradient, expected_gradient, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-11)
    hvp = jax.jvp(jax.grad(native), primals, tangents)[1]
    expected_hvp = jax.jvp(jax.grad(oracle), primals, tangents)[1]
    np.testing.assert_allclose(hvp, expected_hvp, rtol=1e-8, atol=1e-10)


@_DTYPES
def test_batched_worlds_share_geometry_and_match_members(dtype: DTypeLike) -> None:
    tree = Tree(41, 6, 2)
    geometry = tree.cast(dtype)
    topology = tree.topology
    rng = np.random.default_rng(42)
    configurations = jnp.asarray(rng.normal(size=(3, tree.size)), dtype)
    rates = jnp.asarray(rng.normal(size=(3, 2, tree.size)), dtype)
    directions = jnp.asarray(rng.normal(size=(3, 2, tree.size)), dtype)

    def motion(
        parts: Geometry, q: Array, rate: Array, direction: Array
    ) -> tuple[Array, Array, Array, Array]:
        return articulated_motion(topology, *parts, q, rate, direction)

    shared = jax.jit(jax.vmap(motion, in_axes=(None, 0, 0, 0)))(
        geometry, configurations, rates, directions
    )
    for member in range(3):
        single = motion(
            geometry, configurations[member], rates[member], directions[member]
        )
        for left, right in zip(shared, single, strict=True):
            _assert_close(left[member], right, dtype, "shared")

    # Mapped geometry: every world carries its own reference pose.
    scale = jnp.asarray([1.0, 2.0, -1.0], dtype)[:, None, None]

    def stacked(value: Array) -> Array:
        return jnp.stack((value, value, value))

    position_, orientation_, translation_, relative_, axes_, anchors_ = geometry
    mapped_geometry: Geometry = (
        stacked(position_) * scale,
        stacked(orientation_),
        stacked(translation_),
        stacked(relative_),
        stacked(axes_),
        stacked(anchors_),
    )
    mapped = jax.jit(jax.vmap(motion))(mapped_geometry, configurations, rates, directions)
    for member in range(3):
        first, second, third, fourth, fifth, sixth = (
            value[member] for value in mapped_geometry
        )
        parts: Geometry = (first, second, third, fourth, fifth, sixth)
        single = motion(parts, configurations[member], rates[member], directions[member])
        for left, right in zip(mapped, single, strict=True):
            _assert_close(left[member], right, dtype, "mapped")

    # Velocity columns batched over one shared pose (a dense Jacobian build).
    position, orientation = articulated_poses(topology, *geometry, configurations[0])
    jacobian = jax.jit(
        jax.jacfwd(
            lambda value: articulated_body_velocity(
                topology, geometry[4], geometry[5], position, orientation, value[None]
            )[0]
        )
    )(jnp.zeros((tree.size,), dtype))
    expected = jax.jacfwd(
        lambda value: _oracle_velocity(tree, geometry, value, position, orientation)
    )(jnp.zeros((tree.size,), dtype))
    _assert_close(jacobian, expected, dtype, "jacobian")

    # Nested maps fold into one world axis.
    nested = jax.jit(
        jax.vmap(jax.vmap(motion, in_axes=(None, 0, 0, 0)), in_axes=(None, 0, 0, 0))
    )(
        geometry,
        jnp.stack((configurations, configurations)),
        jnp.stack((rates, rates)),
        jnp.stack((directions, directions)),
    )
    for left, right in zip(nested, shared, strict=True):
        np.testing.assert_array_equal(left[0], left[1])
        _assert_close(left[1], right, dtype, "nested")


@_DTYPES
@_DEVICES
def test_arbitrary_storage_order_and_unknown_kinds(
    dtype: DTypeLike, device: jax.Device
) -> None:
    # Distance and ball kinds outside an articulation are fixed edges that
    # never read a coordinate.
    kinds = (
        _HINGE,
        int(RigidJointKind.DISTANCE),
        _PRISMATIC,
        int(RigidJointKind.BALL),
        _HINGE,
        _FIXED,
    )
    tree = Tree(51, 6, 3, kinds=kinds)
    rng = np.random.default_rng(52)
    configuration = jnp.asarray(rng.normal(size=tree.size), dtype)
    rates = jnp.asarray(rng.normal(size=(2, tree.size)), dtype)
    _checked_motion(device, tree, dtype, configuration, rates, rates, "storage order")


@_DEVICES
def test_cuda_columns_beyond_the_native_bound_keep_reference_values(
    device: jax.Device,
) -> None:
    dtype = jnp.float64
    tree = Tree(61, 5, 1)
    rng = np.random.default_rng(62)
    rates = jnp.asarray(rng.normal(size=(257, tree.size)), dtype)
    _checked_motion(
        device,
        tree,
        dtype,
        jnp.asarray(rng.normal(size=tree.size), dtype),
        rates,
        rates,
        "wide",
    )


@_DEVICES
def test_mixed_precision_keeps_carry_dtypes_and_rounding(device: jax.Device) -> None:
    tree = Tree(71, 6, 1)
    position, orientation, translation, relative, axes, anchors = tree.cast(jnp.float64)
    geometry: Geometry = (
        position,
        orientation.astype(jnp.float32),
        translation,
        relative,
        axes,
        anchors,
    )
    rng = np.random.default_rng(72)
    configuration = jnp.asarray(rng.normal(size=tree.size), jnp.float64)
    rates = jnp.asarray(rng.normal(size=(2, tree.size)), jnp.float64)
    with (
        default_device(device),
        pxconfig.context(pxconfig.NumericsConfig(dtype_promotion="standard")),
    ):
        operands = jax.device_put(geometry, device)
        motion = jax.jit(articulated_motion, static_argnums=0)(
            tree.topology, *operands, configuration, rates, rates
        )
        # Compare compiled recurrences: rounding through the float32 carry
        # affects later float64 values, so eager execution is not this oracle.
        expected = jax.jit(
            lambda geometry, q, u: _oracle_motion(tree, geometry, q, u, u)
        )(operands, configuration, rates)
    assert motion[0].dtype == np.dtype(np.float64)
    assert motion[1].dtype == np.dtype(np.float32)
    for value, reference in zip(motion, expected, strict=True):
        assert value.dtype == reference.dtype
        eps = float(jnp.finfo(value.dtype).eps)
        host = np.asarray(value, np.float64)
        target = np.asarray(reference, np.float64)
        scale = max(1.0, float(np.max(np.abs(target))))
        assert float(np.max(np.abs(host - target))) <= 64 * eps * scale


class _HostScalars:
    """NumPy scalar backend of the kernel scalarizer."""

    def literal(self, value: object, dtype: np.dtype) -> Any:
        return np.dtype(dtype).type(value)

    def unary(self, name: str, value: Any) -> Any:
        return {"neg": np.negative, "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos}[name](
            value
        )

    def binary(self, name: str, left: Any, right: Any) -> Any:
        return {
            "add": np.add,
            "sub": np.subtract,
            "mul": np.multiply,
            "div": np.divide,
            "max": np.maximum,
            "min": np.minimum,
        }[name](left, right)

    def compare(self, name: str, left: Any, right: Any) -> Any:
        return {
            "eq": np.equal,
            "ne": np.not_equal,
            "lt": np.less,
            "le": np.less_equal,
            "gt": np.greater,
            "ge": np.greater_equal,
        }[name](left, right)

    def select(self, predicate: Any, on_false: Any, on_true: Any) -> Any:
        return on_true if predicate else on_false

    def convert(self, value: Any, source: np.dtype, target: np.dtype) -> Any:
        del source
        return np.dtype(target).type(value)


@_DTYPES
def test_scalarized_kernel_steps_match_jax_evaluation(dtype: DTypeLike) -> None:
    rng = np.random.default_rng(81)
    resolved = np.dtype(dtype)
    tolerance = 1e-5 if resolved == np.dtype(np.float32) else 1e-13
    programs = []
    for branch in range(3):
        programs += [
            _primal_program("pose", branch, resolved),
            _primal_program("velocity", branch, resolved),
            _transpose_program(branch, resolved),
            _derivative(
                "pose", branch, (True, True) + (False,) * 4 + (branch != 0,), resolved
            ).closed,
            _derivative("pose", branch, (False,) * 6 + (branch != 0,), resolved).closed,
            _derivative(
                "velocity",
                branch,
                (False, True, False, True, True, False, False, False),
                resolved,
            ).closed,
            _derivative(
                "velocity", branch, (False,) + (True,) * 5 + (False, False), resolved
            ).closed,
        ]
    for closed in programs:
        # Scalars exercise exact zero, the small-angle series and the exact branch.
        for coordinate in (0.0, 1e-9, 0.7):
            arguments = []
            for variable in closed.jaxpr.invars:
                aval = variable.aval
                assert isinstance(aval, ShapedArray)
                shape = aval.shape
                value = (
                    np.asarray(coordinate) if shape == () else rng.normal(size=shape)
                ).astype(resolved)
                if shape == (4,):
                    value = value / np.linalg.norm(value)
                arguments.append(value)
            objects = []
            for value in arguments:
                holder = np.empty(value.shape, dtype=object)
                for index in np.ndindex(value.shape):
                    holder[index] = value[index]
                objects.append(holder)
            with np.errstate(all="ignore"):
                scalarized = _evaluate(closed, _HostScalars(), *objects)
            evaluated = jax_core.jaxpr_as_fun(closed)(*arguments)
            for left, right in zip(scalarized, evaluated, strict=True):
                host = np.asarray(left.tolist(), dtype=np.float64)
                reference = np.asarray(right, dtype=np.float64)
                scale = max(1.0, float(np.max(np.abs(reference), initial=0.0)))
                error = float(np.max(np.abs(host - reference), initial=0.0))
                assert error <= tolerance * scale, (coordinate, error)


def test_interfaces_refuse_invalid_layouts() -> None:
    tree = Tree(91, 3, 0)
    geometry = tree.cast(jnp.float64)
    topology = tree.topology
    configuration = jnp.zeros((tree.size,))
    with pytest.raises(TypeError):
        KinematicTopology((1, 3), (0, 0), (0, 1), (1, np.int32(2)))  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        KinematicTopology((1, 3), (0,), (0, 1), (1, 2))
    out_of_range = KinematicTopology(
        topology.kinds,
        topology.coordinates,
        topology.parents[:-1] + (tree.bodies,),
        topology.children,
    )
    with pytest.raises(ValueError):
        articulated_poses(out_of_range, *geometry, configuration)
    missing = KinematicTopology(
        (_HINGE,) * 3, (0, 1, tree.size + 4), topology.parents, topology.children
    )
    with pytest.raises(ValueError):
        articulated_poses(missing, *geometry, jnp.zeros((tree.size + 1,)))
    with pytest.raises(ValueError):
        articulated_poses(
            topology,
            geometry[0][:, :2],
            geometry[1],
            geometry[2],
            geometry[3],
            geometry[4],
            geometry[5],
            configuration,
        )
    with pytest.raises(ValueError):
        articulated_body_velocity(
            topology, geometry[4], geometry[5], geometry[0], geometry[1], configuration
        )
    with pytest.raises(TypeError):
        articulated_motion(
            topology,
            *geometry,
            configuration,
            jnp.zeros((1, tree.size)),
            jnp.zeros((1, tree.size), jnp.float32),
        )
