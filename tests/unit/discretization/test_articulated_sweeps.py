#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Coarse articulated tree sweeps against independent float64 NumPy oracles.

Factor, substitution and force-reduction values are compared with a host
recurrence, a dense ``JᵀGJ`` forward-dynamics model and the Jacobian
transpose; derivatives and direct transposes against the owning JAX
references; batched worlds against their unbatched members; mixed precision
against a host emulation of the per-carry rounding; and every visible device
against the same oracles.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import DTypeLike

from phydrax import config as pxconfig
from phydrax.discretization.particle._articulated_sweeps import (
    _factor_world,
    _reduce_world,
    _solve_world,
    factor_articulated_inertia,
    reduce_articulated_forces,
    solve_articulated_factor,
)
from phydrax.execution import default_device


pytestmark = pytest.mark.strict_jax


_DTYPES = pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64), ids=("float32", "float64")
)
# (edges, unused body capacity)
_TREES = pytest.mark.parametrize(
    "edges,unused", ((1, 0), (5, 2), (17, 3)), ids=("single", "branch", "deep")
)

type Tree = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = value
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _tree(seed: int, edges: int, unused: int) -> Tree:
    """Branched topological tree with fixed edges and unused body capacity.

    Edge ``e`` attaches body ``e + 1`` below an earlier body, so storage order
    is topological. Returns float64 spatial inertias, twist transforms,
    subspaces (zero on fixed edges), the moving mask and int32 topology.
    """

    rng = np.random.default_rng(seed)
    bodies = edges + 1 + unused
    parents = np.asarray(
        [rng.integers(0, edge + 1) for edge in range(edges)], dtype=np.int32
    )
    children = np.arange(1, edges + 1, dtype=np.int32)
    spatial = np.zeros((bodies, 6, 6))
    for body in range(bodies):
        mass = rng.uniform(0.5, 2.0)
        factor = rng.normal(size=(3, 3))
        spatial[body, :3, :3] = mass * np.eye(3)
        spatial[body, 3:, 3:] = factor @ factor.T + 0.5 * np.eye(3)
    transforms = np.broadcast_to(np.eye(6), (edges, 6, 6)).copy()
    for edge in range(edges):
        transforms[edge, :3, 3:] = -_skew(rng.normal(size=3))
    moving = rng.random(edges) > 0.3
    if edges > 1:
        moving[1] = False
        moving[0] = True
    subspaces = rng.normal(size=(edges, 6))
    subspaces[~moving] = 0.0
    return spatial, transforms, subspaces, moving, parents, children


type Operands = tuple[Array, Array, Array, Array, Array, Array]


def _cast(tree: Tree, dtype: DTypeLike) -> Operands:
    spatial, transforms, subspaces, moving, parents, children = tree
    return (
        jnp.asarray(spatial, dtype=dtype),
        jnp.asarray(transforms, dtype=dtype),
        jnp.asarray(subspaces, dtype=dtype),
        jnp.asarray(moving),
        jnp.asarray(parents),
        jnp.asarray(children),
    )


def _host_factor(tree: Tree) -> tuple[np.ndarray, ...]:
    """Independent float64 loop of the reverse articulated recurrence."""

    spatial, transforms, subspaces, moving, parents, children = tree
    inertia = np.array(spatial, dtype=np.float64)
    edges = parents.shape[0]
    projected = np.zeros((edges, 6))
    pivots = np.zeros(edges)
    scales = np.zeros(edges)
    reciprocal = np.zeros(edges)
    for edge in reversed(range(edges)):
        child = inertia[children[edge]].copy()
        subspace = subspaces[edge]
        column = child @ subspace
        pivot = subspace @ column
        inverse = (
            1.0 / pivot if moving[edge] and np.isfinite(pivot) and pivot != 0 else 0.0
        )
        reduced = child - inverse * np.outer(column, column)
        inertia[parents[edge]] += transforms[edge].T @ reduced @ transforms[edge]
        projected[edge] = column
        pivots[edge] = pivot
        scales[edge] = np.abs(subspace) @ np.abs(child) @ np.abs(subspace)
        reciprocal[edge] = inverse
    return inertia, projected, pivots, scales, reciprocal


def _jacobian(tree: Tree) -> np.ndarray:
    """Body twists per unit edge rate, ``(6C, E)``, by forward propagation."""

    spatial, transforms, subspaces, _, parents, children = tree
    bodies = spatial.shape[0]
    edges = parents.shape[0]
    jacobian = np.zeros((bodies * 6, edges))
    for column in range(edges):
        twist = np.zeros((bodies, 6))
        for edge in range(edges):
            twist[children[edge]] = transforms[edge] @ twist[parents[edge]] + subspaces[
                edge
            ] * float(edge == column)
        jacobian[:, column] = twist.reshape(-1)
    return jacobian


def _block_inertia(spatial: np.ndarray) -> np.ndarray:
    bodies = spatial.shape[0]
    dense = np.zeros((bodies * 6, bodies * 6))
    for body in range(bodies):
        dense[6 * body : 6 * body + 6, 6 * body : 6 * body + 6] = spatial[body]
    return dense


def _host_dynamics(
    tree: Tree,
    effort: np.ndarray,
    bias: np.ndarray | None,
    convective: np.ndarray | None,
) -> np.ndarray:
    """Dense model ``a = M⁻¹(τ - Jᵀ(G A_c + p))`` on moving edges, zero elsewhere."""

    spatial, transforms, _, moving, parents, children = tree
    bodies = spatial.shape[0]
    columns = effort.shape[1]
    jacobian = _jacobian(tree)
    inertia = _block_inertia(spatial)
    mass = jacobian.T @ inertia @ jacobian
    rhs = np.array(effort, dtype=np.float64)
    for column in range(columns):
        drift = np.zeros((bodies, 6))
        if convective is not None:
            for edge in range(parents.shape[0]):
                drift[children[edge]] = (
                    transforms[edge] @ drift[parents[edge]] + convective[edge, :, column]
                )
        force = inertia @ drift.reshape(-1)
        if bias is not None:
            force = force + bias[:, :, column].reshape(-1)
        rhs[:, column] -= jacobian.T @ force
    result = np.zeros_like(rhs)
    result[moving] = np.linalg.solve(mass[np.ix_(moving, moving)], rhs[moving])
    return result


def _assert_close(
    actual: Array, expected: np.ndarray, dtype: DTypeLike, name: str
) -> None:
    host = np.asarray(actual, dtype=np.float64)
    assert host.shape == expected.shape, name
    assert np.asarray(actual).dtype == np.dtype(dtype), name
    scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
    tolerance = 2e-3 if np.dtype(dtype) == np.dtype(np.float32) else 1e-9
    error = float(np.max(np.abs(host - expected), initial=0.0))
    assert error <= tolerance * scale, (name, error, scale)


def _factor(tree: Tree, dtype: DTypeLike) -> tuple[Array, ...]:
    return factor_articulated_inertia(*_cast(tree, dtype))


@_DTYPES
@_TREES
def test_factor_matches_host_recurrence_and_dense_determinant(
    edges: int, unused: int, dtype: DTypeLike
) -> None:
    tree = _tree(edges + unused, edges, unused)
    actual = jax.jit(factor_articulated_inertia)(*_cast(tree, dtype))
    expected = _host_factor(tree)
    for value, reference, name in zip(
        actual,
        expected,
        ("articulated", "projected", "pivots", "scales", "reciprocal"),
        strict=True,
    ):
        _assert_close(value, reference, dtype, name)
    moving = tree[3]
    reciprocal = np.asarray(actual[4])
    assert np.all(reciprocal[~moving] == 0.0)
    assert np.all(np.asarray(actual[2])[moving] > 0.0)
    if np.dtype(dtype) == np.dtype(np.float64):
        jacobian = _jacobian(tree)
        mass = jacobian.T @ _block_inertia(tree[0]) @ jacobian
        determinant = np.linalg.det(mass[np.ix_(moving, moving)])
        product = float(np.prod(np.asarray(actual[2])[moving]))
        assert abs(product - determinant) <= 1e-9 * abs(determinant)


@_DTYPES
@_TREES
@pytest.mark.parametrize("columns", (1, 3))
@pytest.mark.parametrize(
    "terms", ((False, False), (True, False), (True, True)), ids=("pure", "bias", "full")
)
def test_solve_matches_dense_forward_dynamics(
    edges: int, unused: int, columns: int, terms: tuple[bool, bool], dtype: DTypeLike
) -> None:
    tree = _tree(31 + edges, edges, unused)
    rng = np.random.default_rng(columns + 7 * edges)
    bodies = tree[0].shape[0]
    effort = rng.normal(size=(edges, columns))
    bias = rng.normal(size=(bodies, 6, columns)) if terms[0] else None
    convective = rng.normal(size=(edges, 6, columns)) if terms[1] else None
    articulated, projected, _, _, reciprocal = _factor(tree, dtype)
    _, transforms, subspaces, _, parents, children = _cast(tree, dtype)

    def solve(effort_: Array, bias_: Array | None, convective_: Array | None) -> Array:
        return solve_articulated_factor(
            articulated,
            transforms,
            subspaces,
            projected,
            reciprocal,
            parents,
            children,
            effort_,
            bias_,
            convective_,
        )

    actual = jax.jit(solve)(
        jnp.asarray(effort, dtype=dtype),
        None if bias is None else jnp.asarray(bias, dtype=dtype),
        None if convective is None else jnp.asarray(convective, dtype=dtype),
    )
    expected = _host_dynamics(tree, effort, bias, convective)
    _assert_close(actual, expected, dtype, "acceleration")
    assert np.all(np.asarray(actual)[~tree[3]] == 0.0)


@_DTYPES
@_TREES
@pytest.mark.parametrize("columns", (1, 4))
def test_force_reduction_is_the_jacobian_transpose(
    edges: int, unused: int, columns: int, dtype: DTypeLike
) -> None:
    tree = _tree(57 + edges, edges, unused)
    bodies = tree[0].shape[0]
    force = np.random.default_rng(columns).normal(size=(bodies, 6, columns))
    _, transforms, subspaces, _, parents, children = _cast(tree, dtype)
    actual = jax.jit(reduce_articulated_forces)(
        transforms, subspaces, parents, children, jnp.asarray(force, dtype=dtype)
    )
    expected = _jacobian(tree).T @ force.reshape(bodies * 6, columns)
    _assert_close(actual, expected, dtype, "effort")


@_DTYPES
def test_zero_edges_and_zero_columns_are_exact(dtype: DTypeLike) -> None:
    spatial = jnp.broadcast_to(jnp.eye(6, dtype=dtype), (2, 6, 6))
    empty = jnp.zeros((0,), dtype=jnp.int32)
    articulated, projected, pivots, scales, reciprocal = factor_articulated_inertia(
        spatial,
        jnp.zeros((0, 6, 6), dtype=dtype),
        jnp.zeros((0, 6), dtype=dtype),
        jnp.zeros((0,), dtype=jnp.bool_),
        empty,
        empty,
    )
    np.testing.assert_array_equal(articulated, spatial)
    assert projected.shape == (0, 6)
    assert pivots.shape == scales.shape == reciprocal.shape == (0,)
    force = jnp.ones((2, 6, 4), dtype=dtype)
    effort = reduce_articulated_forces(
        jnp.zeros((0, 6, 6), dtype=dtype),
        jnp.zeros((0, 6), dtype=dtype),
        empty,
        empty,
        force,
    )
    assert effort.shape == (0, 4) and effort.dtype == np.dtype(dtype)

    tree = _tree(3, 5, 1)
    factor = _factor(tree, dtype)
    _, transforms, subspaces, _, parents, children = _cast(tree, dtype)
    acceleration = solve_articulated_factor(
        factor[0],
        transforms,
        subspaces,
        factor[1],
        factor[4],
        parents,
        children,
        jnp.zeros((5, 0), dtype=dtype),
        None,
        None,
    )
    assert acceleration.shape == (5, 0)


@_DTYPES
def test_nonfinite_columns_never_mix_and_zero_products_propagate(
    dtype: DTypeLike,
) -> None:
    tree = _tree(5, 6, 1)
    articulated, projected, _, _, reciprocal = _factor(tree, dtype)
    _, transforms, subspaces, _, parents, children = _cast(tree, dtype)
    bodies = tree[0].shape[0]
    clean = np.random.default_rng(9).normal(size=(6, 3))
    poisoned = clean.copy()
    poisoned[2, 1] = np.nan
    solve = jax.jit(
        lambda effort: solve_articulated_factor(
            articulated,
            transforms,
            subspaces,
            projected,
            reciprocal,
            parents,
            children,
            effort,
            None,
            None,
        )
    )
    reference = np.asarray(solve(jnp.asarray(clean, dtype=dtype)))
    mixed = np.asarray(solve(jnp.asarray(poisoned, dtype=dtype)))
    np.testing.assert_array_equal(mixed[:, (0, 2)], reference[:, (0, 2)])
    assert np.any(np.isnan(mixed[:, 1]))

    force = np.random.default_rng(4).normal(size=(bodies, 6, 3))
    force[-2, 3, 2] = np.inf
    reduce = jax.jit(reduce_articulated_forces)
    effort = np.asarray(
        reduce(transforms, subspaces, parents, children, jnp.asarray(force, dtype=dtype))
    )
    finite = force.copy()
    finite[-2, 3, 2] = 0.0
    baseline = np.asarray(
        reduce(transforms, subspaces, parents, children, jnp.asarray(finite, dtype=dtype))
    )
    np.testing.assert_array_equal(effort[:, :2], baseline[:, :2])

    # A NaN child inertia below a fixed edge reaches its parent through the
    # zero reciprocal pivot: 0 * NaN is never skipped.
    spatial = np.array(tree[0])
    fixed_edge = 1
    spatial[tree[5][fixed_edge]] = np.nan
    nan_tree = (spatial,) + tree[1:]
    factored = jax.jit(factor_articulated_inertia)(*_cast(nan_tree, dtype))
    expected = jax.jit(_factor_world)(*_cast(nan_tree, dtype))
    for value, reference_value in zip(factored, expected, strict=True):
        np.testing.assert_array_equal(np.isnan(value), np.isnan(reference_value))
    assert np.all(np.isnan(np.asarray(factored[0])[tree[4][fixed_edge]]))


@_DTYPES
def test_zero_and_negative_pivots_follow_the_numerical_factor(dtype: DTypeLike) -> None:
    tree = _tree(13, 4, 0)
    spatial, transforms, subspaces, moving, parents, children = tree
    subspaces = np.array(subspaces)
    moving = np.array(moving)
    # A moving edge with a zero subspace has a zero pivot and no inverse.
    moving[2] = True
    subspaces[2] = 0.0
    # Body 4 is a leaf below edge 3: an indefinite inertia there yields a
    # negative pivot that the numerical factor still inverts.
    moving[3] = True
    subspaces[3] = np.asarray([1.0, -0.5, 0.25, 0.5, 0.0, 2.0])
    spatial = np.array(spatial)
    spatial[children[3]] = -np.eye(6)
    tree = (spatial, transforms, subspaces, moving, parents, children)
    _, _, pivots, _, reciprocal = jax.jit(factor_articulated_inertia)(*_cast(tree, dtype))
    _, _, host_pivots, _, host_reciprocal = _host_factor(tree)
    assert float(pivots[2]) == 0.0 and float(reciprocal[2]) == 0.0
    assert float(pivots[3]) < 0.0 and float(reciprocal[3]) < 0.0
    _assert_close(pivots, host_pivots, dtype, "pivots")
    _assert_close(reciprocal, host_reciprocal, dtype, "reciprocal")


def _devices() -> tuple[jax.Device, ...]:
    return (jax.devices("cpu")[0],) + tuple(
        device for device in jax.devices() if device.platform == "gpu"
    )


@_DTYPES
def test_safe_index_semantics_match_reference_on_every_device(dtype: DTypeLike) -> None:
    tree = _tree(21, 6, 1)
    bodies = tree[0].shape[0]
    # Negative indices wrap once; remaining out-of-range gathers clamp and
    # out-of-range scatters are dropped, exactly as jnp scalar indexing.
    parents = np.asarray([0, -bodies, 1, bodies + 3, -1, -bodies - 2], dtype=np.int32)
    children = np.asarray([1, 2, -3, 4, bodies + 5, -bodies - 4], dtype=np.int32)
    host = tree[:4] + (parents, children)
    force = np.random.default_rng(1).normal(size=(bodies, 6, 2))
    for device in _devices():
        with default_device(device):
            operands = jax.device_put(_cast(host, dtype), device)
            factor = jax.jit(factor_articulated_inertia)(*operands)
            expected = jax.jit(_factor_world)(*operands)
            for value, reference in zip(factor, expected, strict=True):
                assert value.devices() == {device}
                _assert_close(value, np.asarray(reference, np.float64), dtype, "factor")
            effort = jax.jit(reduce_articulated_forces)(
                operands[1],
                operands[2],
                operands[4],
                operands[5],
                jnp.asarray(force, dtype),
            )
            (reference_effort,) = jax.jit(_reduce_world)(
                operands[1],
                operands[2],
                operands[4],
                operands[5],
                jnp.asarray(force, dtype),
            )
            _assert_close(
                effort, np.asarray(reference_effort, np.float64), dtype, "forces"
            )
            rhs = jnp.asarray(np.random.default_rng(2).normal(size=(6, 2)), dtype)
            convective = jnp.ones((6, 6, 2), dtype)
            acceleration = jax.jit(solve_articulated_factor)(
                factor[0],
                operands[1],
                operands[2],
                factor[1],
                factor[4],
                operands[4],
                operands[5],
                rhs,
                jnp.asarray(force, dtype),
                convective,
            )
            (reference_acceleration,) = jax.jit(
                lambda *values: _solve_world(
                    *values, bodies=bodies, has_bias=True, has_convective=True
                )
            )(
                factor[0],
                operands[1],
                operands[2],
                factor[1],
                factor[4],
                operands[4],
                operands[5],
                rhs,
                jnp.asarray(force, dtype),
                convective,
            )
            _assert_close(
                acceleration,
                np.asarray(reference_acceleration, np.float64),
                dtype,
                "solve",
            )


type SolveOperands = tuple[Array, Array, Array, Array, Array, Array, Array]


def _solve_operands(tree: Tree, dtype: DTypeLike) -> SolveOperands:
    articulated, projected, _, _, reciprocal = _factor(tree, dtype)
    _, transforms, subspaces, _, parents, children = _cast(tree, dtype)
    return articulated, transforms, subspaces, projected, reciprocal, parents, children


def test_derivatives_follow_the_owning_references() -> None:
    dtype = jnp.float64
    tree = _tree(77, 7, 2)
    spatial, transforms, subspaces, moving, parents, children = _cast(tree, dtype)
    rng = np.random.default_rng(78)
    bodies = tree[0].shape[0]
    effort = jnp.asarray(rng.normal(size=(7, 2)))
    bias = jnp.asarray(rng.normal(size=(bodies, 6, 2)))
    convective = jnp.asarray(rng.normal(size=(7, 6, 2)))
    probe = jnp.asarray(rng.normal(size=(7, 2)))

    def native(spatial_: Array, transforms_: Array, subspaces_: Array) -> Array:
        articulated, projected, _, scales, reciprocal = factor_articulated_inertia(
            spatial_, transforms_, subspaces_, moving, parents, children
        )
        acceleration = solve_articulated_factor(
            articulated,
            transforms_,
            subspaces_,
            projected,
            reciprocal,
            parents,
            children,
            effort,
            bias,
            convective,
        )
        forces = reduce_articulated_forces(
            transforms_, subspaces_, parents, children, bias
        )
        return jnp.sum(probe * (acceleration + forces)) + jnp.sum(scales)

    def reference(spatial_: Array, transforms_: Array, subspaces_: Array) -> Array:
        articulated, projected, _, scales, reciprocal = _factor_world(
            spatial_, transforms_, subspaces_, moving, parents, children
        )
        (acceleration,) = _solve_world(
            articulated,
            transforms_,
            subspaces_,
            projected,
            reciprocal,
            parents,
            children,
            effort,
            bias,
            convective,
            bodies=bodies,
            has_bias=True,
            has_convective=True,
        )
        (forces,) = _reduce_world(transforms_, subspaces_, parents, children, bias)
        return jnp.sum(probe * (acceleration + forces)) + jnp.sum(scales)

    primals = (spatial, transforms, subspaces)
    directions = tuple(jnp.asarray(rng.normal(size=value.shape)) for value in primals)
    native_value, native_tangent = jax.jvp(native, primals, directions)
    reference_value, reference_tangent = jax.jvp(reference, primals, directions)
    np.testing.assert_allclose(native_value, reference_value, rtol=1e-10)
    np.testing.assert_allclose(native_tangent, reference_tangent, rtol=1e-9)
    native_gradient = jax.jit(jax.grad(native, argnums=(0, 1, 2)))(*primals)
    reference_gradient = jax.grad(reference, argnums=(0, 1, 2))(*primals)
    for left, right in zip(native_gradient, reference_gradient, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-9, atol=1e-11)
    native_hvp = jax.jvp(jax.grad(native), primals, directions)[1]
    reference_hvp = jax.jvp(jax.grad(reference), primals, directions)[1]
    np.testing.assert_allclose(native_hvp, reference_hvp, rtol=1e-8, atol=1e-10)


@_DTYPES
def test_direct_linear_transposes_preserve_virtual_work(dtype: DTypeLike) -> None:
    tree = _tree(91, 6, 2)
    bodies = tree[0].shape[0]
    operands = _solve_operands(tree, dtype)
    rng = np.random.default_rng(92)
    effort = jnp.asarray(rng.normal(size=(6, 3)), dtype)
    bias = jnp.asarray(rng.normal(size=(bodies, 6, 3)), dtype)
    convective = jnp.asarray(rng.normal(size=(6, 6, 3)), dtype)
    cotangent = jnp.asarray(rng.normal(size=(6, 3)), dtype)

    def solve(effort_: Array, bias_: Array, convective_: Array) -> Array:
        return solve_articulated_factor(*operands, effort_, bias_, convective_)

    transpose = jax.linear_transpose(solve, effort, bias, convective)
    effort_bar, bias_bar, convective_bar = jax.jit(transpose)(cotangent)
    work = float(jnp.vdot(cotangent, solve(effort, bias, convective)))
    dual = float(
        jnp.vdot(effort_bar, effort)
        + jnp.vdot(bias_bar, bias)
        + jnp.vdot(convective_bar, convective)
    )
    tolerance = 2e-3 if np.dtype(dtype) == np.dtype(np.float32) else 1e-10
    assert abs(work - dual) <= tolerance * max(1.0, abs(work))

    # M⁻¹ is symmetric: the effort transpose is the substitution itself.
    pure = jax.linear_transpose(
        lambda value: solve_articulated_factor(*operands, value, None, None), effort
    )
    (pure_bar,) = pure(cotangent)
    _assert_close(
        pure_bar,
        np.asarray(
            solve_articulated_factor(*operands, cotangent, None, None), np.float64
        ),
        dtype,
        "symmetric transpose",
    )

    _, transforms, subspaces, _, parents, children = _cast(tree, dtype)
    force = jnp.asarray(rng.normal(size=(bodies, 6, 3)), dtype)

    def reduce(force_: Array) -> Array:
        return reduce_articulated_forces(transforms, subspaces, parents, children, force_)

    (force_bar,) = jax.linear_transpose(reduce, force)(cotangent)
    expected = (_jacobian(tree) @ np.asarray(cotangent, np.float64)).reshape(bodies, 6, 3)
    _assert_close(force_bar, expected, dtype, "force transpose")


type Numerics = tuple[Array, Array, Array, Array]


def _numerics(values: Callable[[int], Array], /) -> Numerics:
    """Spatial inertia, transforms, subspaces and moving mask by position."""
    return values(0), values(1), values(2), values(3)


@_DTYPES
def test_shared_and_mapped_topology_batches_match_members(
    dtype: DTypeLike,
) -> None:
    trees = tuple(_tree(100 + seed, 5, 1) for seed in range(3))
    casts = tuple(_cast(tree, dtype) for tree in trees)
    stacked = _numerics(lambda position: jnp.stack([cast[position] for cast in casts]))
    parents, children = casts[0][4], casts[0][5]

    def pipeline(
        numerics: Numerics, parent: Array, child: Array, effort: Array
    ) -> tuple[Array, Array]:
        spatial, transforms, subspaces, moving = numerics
        articulated, projected, _, _, reciprocal = factor_articulated_inertia(
            spatial, transforms, subspaces, moving, parent, child
        )
        acceleration = solve_articulated_factor(
            articulated,
            transforms,
            subspaces,
            projected,
            reciprocal,
            parent,
            child,
            effort,
            None,
            None,
        )
        forces = reduce_articulated_forces(
            transforms, subspaces, parent, child, articulated[..., :1]
        )
        return acceleration, forces

    def member_of(member: int) -> Numerics:
        return _numerics(lambda position: stacked[position][member])

    efforts = jnp.asarray(np.random.default_rng(5).normal(size=(3, 5, 2)), dtype)
    shared_axes = (0, None, None, 0)
    batched = jax.jit(jax.vmap(pipeline, in_axes=shared_axes))(
        stacked, parents, children, efforts
    )
    for member in range(3):
        single = pipeline(member_of(member), parents, children, efforts[member])
        for left, right in zip(batched, single, strict=True):
            _assert_close(left[member], np.asarray(right, np.float64), dtype, "shared")

    # Mapped topology: every world follows its own tree.
    mapped_parents = jnp.stack([cast[4] for cast in casts])
    mapped_children = jnp.stack([cast[5] for cast in casts])
    mapped = jax.jit(jax.vmap(pipeline))(
        stacked, mapped_parents, mapped_children, efforts
    )
    for member in range(3):
        single = pipeline(
            member_of(member),
            mapped_parents[member],
            mapped_children[member],
            efforts[member],
        )
        for left, right in zip(mapped, single, strict=True):
            _assert_close(left[member], np.asarray(right, np.float64), dtype, "mapped")

    # Nested maps fold both axes into one world axis.
    nested = jax.jit(
        jax.vmap(jax.vmap(pipeline, in_axes=shared_axes), in_axes=shared_axes)
    )(
        _numerics(lambda position: jnp.stack((stacked[position], stacked[position]))),
        parents,
        children,
        jnp.stack((efforts, efforts)),
    )
    for left, right in zip(nested, batched, strict=True):
        np.testing.assert_array_equal(left[0], left[1])
        _assert_close(left[1], np.asarray(right, np.float64), dtype, "nested")


@_DTYPES
def test_shared_factor_batches_over_right_hand_sides_and_partial_nesting(
    dtype: DTypeLike,
) -> None:
    tree = _tree(81, 6, 1)
    bodies = tree[0].shape[0]
    operands = _solve_operands(tree, dtype)
    rng = np.random.default_rng(82)
    efforts = jnp.asarray(rng.normal(size=(4, 6, 2)), dtype)
    biases = jnp.asarray(rng.normal(size=(4, bodies, 6, 2)), dtype)
    forces = jnp.asarray(rng.normal(size=(4, bodies, 6, 2)), dtype)
    _, transforms, subspaces, _, parents, children = _cast(tree, dtype)

    def solve(effort: Array, bias: Array) -> Array:
        return solve_articulated_factor(*operands, effort, bias, None)

    def reduce(force: Array) -> Array:
        return reduce_articulated_forces(transforms, subspaces, parents, children, force)

    batched = jax.jit(jax.vmap(solve))(efforts, biases)
    reduced = jax.jit(jax.vmap(reduce))(forces)
    jacobian = _jacobian(tree)
    for member in range(4):
        _assert_close(
            batched[member],
            np.asarray(solve(efforts[member], biases[member]), np.float64),
            dtype,
            "shared factor",
        )
        _assert_close(
            reduced[member],
            jacobian.T @ np.asarray(forces[member], np.float64).reshape(bodies * 6, 2),
            dtype,
            "shared reduction",
        )

    # Inner map over right-hand sides, outer map over a shared bias with the
    # effort fixed: the factor stays shared, the bias spans the outer axis.
    nested = jax.jit(jax.vmap(jax.vmap(solve, in_axes=(0, None)), in_axes=(None, 0)))(
        efforts, biases
    )
    for outer in range(4):
        for inner in range(4):
            _assert_close(
                nested[outer, inner],
                np.asarray(solve(efforts[inner], biases[outer]), np.float64),
                dtype,
                "partial nesting",
            )


@_DTYPES
def test_cpu_and_cuda_sweeps_agree(dtype: DTypeLike) -> None:
    tree = _tree(66, 17, 2)
    bodies = tree[0].shape[0]
    rng = np.random.default_rng(67)
    effort = rng.normal(size=(17, 4))
    bias = rng.normal(size=(bodies, 6, 4))
    convective = rng.normal(size=(17, 6, 4))
    expected = _host_dynamics(tree, effort, bias, convective)
    for device in _devices():
        with default_device(device):
            operands = jax.device_put(_cast(tree, dtype), device)
            articulated, projected, _, _, reciprocal = jax.jit(
                factor_articulated_inertia
            )(*operands)
            acceleration = jax.jit(solve_articulated_factor)(
                articulated,
                operands[1],
                operands[2],
                projected,
                reciprocal,
                operands[4],
                operands[5],
                jnp.asarray(effort, dtype),
                jnp.asarray(bias, dtype),
                jnp.asarray(convective, dtype),
            )
        assert acceleration.devices() == {device}
        _assert_close(acceleration, expected, dtype, str(device))


def _host_mixed_factor(tree: Tree) -> tuple[np.ndarray, ...]:
    """Float32 inertia carry, float64 edge carries, float64 step arithmetic."""

    spatial, transforms, subspaces, moving, parents, children = tree
    inertia = np.asarray(spatial, np.float32).astype(np.float64)
    edges = parents.shape[0]
    projected = np.zeros((edges, 6))
    pivots = np.zeros(edges)
    scales = np.zeros(edges)
    reciprocal = np.zeros(edges)
    for edge in reversed(range(edges)):
        child = inertia[children[edge]].copy()
        subspace = subspaces[edge]
        column = child @ subspace
        pivot = subspace @ column
        inverse = 1.0 / pivot if moving[edge] and pivot != 0 else 0.0
        reduced = child - inverse * np.outer(column, column)
        inertia[parents[edge]] = (
            inertia[parents[edge]] + transforms[edge].T @ reduced @ transforms[edge]
        ).astype(np.float32)
        projected[edge] = column
        pivots[edge] = pivot
        scales[edge] = np.abs(subspace) @ np.abs(child) @ np.abs(subspace)
        reciprocal[edge] = inverse
    return inertia, projected, pivots, scales, reciprocal


def test_mixed_precision_keeps_reference_carry_dtypes_and_rounding() -> None:
    tree = _tree(44, 6, 1)
    spatial, transforms, subspaces, moving, parents, children = tree
    bodies = spatial.shape[0]
    force = np.random.default_rng(45).normal(size=(bodies, 6, 2))
    expected = _host_mixed_factor(tree)
    expected_effort = np.zeros((6, 2))
    carried = np.asarray(force, np.float32).astype(np.float64)
    for edge in reversed(range(6)):
        child_force = carried[children[edge]].copy()
        expected_effort[edge] = subspaces[edge] @ child_force
        carried[parents[edge]] = (
            carried[parents[edge]] + transforms[edge].T @ child_force
        ).astype(np.float32)
    eps = float(jnp.finfo(jnp.float32).eps)
    for device in _devices():
        with (
            default_device(device),
            pxconfig.context(pxconfig.NumericsConfig(dtype_promotion="standard")),
        ):
            operands = jax.device_put(
                (
                    jnp.asarray(spatial, jnp.float32),
                    jnp.asarray(transforms, jnp.float64),
                    jnp.asarray(subspaces, jnp.float64),
                    jnp.asarray(moving),
                    jnp.asarray(parents),
                    jnp.asarray(children),
                ),
                device,
            )
            factor = jax.jit(factor_articulated_inertia)(*operands)
            effort = jax.jit(reduce_articulated_forces)(
                operands[1],
                operands[2],
                operands[4],
                operands[5],
                jnp.asarray(force, jnp.float32),
            )
        assert factor[0].dtype == np.dtype(np.float32)
        assert all(value.dtype == np.dtype(np.float64) for value in factor[1:])
        assert effort.dtype == np.dtype(np.float64)
        for value, reference in zip(factor, expected, strict=True):
            scale = max(1.0, float(np.max(np.abs(reference))))
            error = float(np.max(np.abs(np.asarray(value, np.float64) - reference)))
            assert error <= 64 * eps * scale, (str(device), error)
        scale = max(1.0, float(np.max(np.abs(expected_effort))))
        error = float(np.max(np.abs(np.asarray(effort) - expected_effort)))
        assert error <= 64 * eps * scale, (str(device), error)


@_DTYPES
def test_columns_beyond_the_native_bound_keep_reference_values(dtype: DTypeLike) -> None:
    tree = _tree(71, 4, 1)
    bodies = tree[0].shape[0]
    columns = 257
    force = np.random.default_rng(72).normal(size=(bodies, 6, columns))
    _, transforms, subspaces, _, parents, children = _cast(tree, dtype)
    for device in _devices():
        with default_device(device):
            effort = jax.jit(reduce_articulated_forces)(
                *jax.device_put(
                    (transforms, subspaces, parents, children, jnp.asarray(force, dtype)),
                    device,
                )
            )
        expected = _jacobian(tree).T @ force.reshape(bodies * 6, columns)
        _assert_close(effort, expected, dtype, str(device))


def test_interfaces_refuse_invalid_layouts() -> None:
    tree = _tree(1, 3, 0)
    spatial, transforms, subspaces, moving, parents, children = _cast(tree, jnp.float64)
    with pytest.raises(TypeError):
        factor_articulated_inertia(
            spatial, transforms, subspaces.astype(jnp.int32), moving, parents, children
        )
    with pytest.raises(TypeError):
        factor_articulated_inertia(
            spatial, transforms, subspaces, moving.astype(jnp.int32), parents, children
        )
    with pytest.raises(TypeError):
        reduce_articulated_forces(
            transforms, subspaces, parents.astype(jnp.float64), children, spatial[..., :1]
        )
    with pytest.raises(ValueError):
        reduce_articulated_forces(
            transforms, subspaces, parents, children, spatial[..., 0]
        )
    with pytest.raises(ValueError):
        reduce_articulated_forces(
            transforms, subspaces, parents[:2], children, spatial[..., :1]
        )
