#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Streaming APIC moment primitive against an independent float64 NumPy oracle.

Every numerical comparison is through consumer-visible values: route sums
against a host oracle with a recursive-summation error bound, derivatives
against the owning pure-JAX reference, compiler identity through the metadata
payload, and structural sparsity through traced derivative patterns.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import DTypeLike

from phydrax.discretization.mpm._moments_native import (
    _moments_p,
    _reference,
    streaming_moments,
)
from phydrax.sparse._structural_trace import DerivativeKind, trace_sparse_pattern


pytestmark = pytest.mark.strict_jax

type Operands = tuple[Array, Array, Array, Array, Array, Array]
type Moments = tuple[Array, Array, Array]
type Shape = tuple[int, int, int, int, int, int]

_DTYPES = pytest.mark.parametrize(
    "dtype", (jnp.float32, jnp.float64), ids=("float32", "float64")
)
_ALL = (0, 1, 2)


def _operands(
    seed: int,
    particles: int,
    routes: int,
    nodes: int,
    components: int,
    gradient_width: int,
    offset_width: int,
    dtype: DTypeLike,
) -> Operands:
    """Dynamic routes with masked rows, garbage invalid indices and zero weights."""
    rng = np.random.default_rng(seed)
    valid = rng.random((particles, routes)) > 0.25
    if particles > 0:
        valid[0, : routes // 2] = False
    if particles > 2:
        valid[2] = False
    indices = rng.integers(0, nodes, size=(particles, routes))
    # Invalid routes may carry arbitrary out-of-range indices; none may be read.
    garbage = rng.choice(np.asarray((-7, nodes, nodes + 101)), size=indices.shape)
    indices = np.where(valid, indices, garbage).astype(np.int32)
    weights = rng.uniform(-0.5, 1.0, size=(particles, routes))
    weights[valid & (rng.random((particles, routes)) < 0.15)] = 0.0
    if valid.any():
        weights[tuple(np.argwhere(valid)[-1])] = 0.0
    return (
        jnp.asarray(indices),
        jnp.asarray(valid),
        jnp.asarray(weights, dtype=dtype),
        jnp.asarray(rng.normal(size=(particles, routes, gradient_width)), dtype=dtype),
        jnp.asarray(rng.normal(size=(particles, routes, offset_width)), dtype=dtype),
        jnp.asarray(rng.normal(size=(nodes, components)), dtype=dtype),
    )


def _host_moments(
    operands: Sequence[Array],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Float64 NumPy route sums and their absolute-term magnitudes."""
    indices, valid, weights, gradients, offsets, grid = (
        np.asarray(value) for value in operands
    )
    weights, gradients, offsets, grid = (
        value.astype(np.float64) for value in (weights, gradients, offsets, grid)
    )
    safe = np.where(valid, indices, 0)
    with np.errstate(invalid="ignore", over="ignore"):
        route = np.where(valid[..., None], grid[safe], 0.0)
        weight = np.where(valid, weights, 0.0)
        weighted = weight[..., None] * route
        values = (
            weighted.sum(axis=1),
            np.sum(route[..., :, None] * gradients[..., None, :], axis=1),
            np.sum(weighted[..., :, None] * offsets[..., None, :], axis=1),
        )
        magnitudes = (
            np.abs(weighted).sum(axis=1),
            np.sum(np.abs(route[..., :, None] * gradients[..., None, :]), axis=1),
            np.sum(np.abs(weighted[..., :, None] * offsets[..., None, :]), axis=1),
        )
    return values, magnitudes


def _assert_moment(
    actual: Array,
    expected: np.ndarray,
    magnitude: np.ndarray,
    routes: int,
    dtype: DTypeLike,
    name: str,
) -> None:
    """Exact NaN placement and a recursive-summation forward error bound."""
    actual_host = np.asarray(actual)
    assert actual_host.shape == expected.shape, name
    assert actual_host.dtype == np.dtype(dtype), name
    nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual_host), nan, err_msg=name)
    eps = float(jnp.finfo(dtype).eps)
    bound = (routes + 3) * eps * magnitude + np.finfo(np.dtype(dtype)).tiny
    error = np.abs(actual_host.astype(np.float64) - expected)
    assert np.all(error[~nan] <= bound[~nan]), (name, float(np.max(error[~nan])))


def _assert_moments(
    actual: Sequence[Array],
    operands: Sequence[Array],
    dtype: DTypeLike,
    outputs: Sequence[int] = _ALL,
) -> None:
    values, magnitudes = _host_moments(operands)
    routes = operands[0].shape[1]
    assert len(actual) == len(outputs)
    for value, output in zip(actual, outputs, strict=True):
        _assert_moment(
            value, values[output], magnitudes[output], routes, dtype, str(output)
        )


def _reference_all(*operands: Array) -> Moments:
    velocity, velocity_gradient, affine_moment = _reference(*operands, outputs=_ALL)
    return velocity, velocity_gradient, affine_moment


def _streaming_all(*operands: Array) -> Moments:
    return streaming_moments(*operands)


_SHAPES: dict[str, Shape] = {
    "single-particle": (1, 27, 5, 3, 3, 3),
    "tile-tail": (7, 27, 40, 3, 3, 3),
    "multi-tile-tail": (130, 27, 97, 3, 3, 3),
    "planar-quadratic": (33, 9, 20, 2, 2, 2),
    "planar-linear": (65, 4, 12, 2, 2, 2),
    "mixed-widths": (5, 8, 11, 1, 2, 3),
    "no-particles": (0, 27, 4, 3, 3, 3),
    "wide-stencil-reference": (7, 64, 97, 3, 3, 3),
    "wide-axis-reference": (7, 27, 97, 4, 4, 4),
}


@_DTYPES
@pytest.mark.parametrize("shape", tuple(_SHAPES.values()), ids=tuple(_SHAPES))
def test_streaming_moments_match_host_oracle_across_tails(
    shape: Shape, dtype: DTypeLike
) -> None:
    operands = _operands(sum(shape), *shape, dtype)
    _assert_moments(jax.jit(_streaming_all)(*operands), operands, dtype)
    _assert_moments(jax.jit(_reference_all)(*operands), operands, dtype)
    _assert_moments(_streaming_all(*operands), operands, dtype)


@_DTYPES
def test_int64_indices_preserve_reference_execution(dtype: DTypeLike) -> None:
    values = _operands(109, 7, 27, 97, 3, 3, 3, dtype)
    operands = (values[0].astype(jnp.int64),) + values[1:]
    _assert_moments(jax.jit(_streaming_all)(*operands), operands, dtype)


def test_streaming_refuses_noninteger_indices() -> None:
    values = _operands(117, 7, 27, 97, 3, 3, 3, jnp.float64)
    operands = (values[0].astype(jnp.float64),) + values[1:]
    with pytest.raises(TypeError):
        jax.jit(_streaming_all)(*operands)


def test_streaming_refuses_incompatible_route_extents() -> None:
    values = _operands(119, 7, 27, 97, 3, 3, 3, jnp.float64)
    operands = values[:2] + (values[2][:, :-1],) + values[3:]
    with pytest.raises(ValueError):
        jax.jit(_streaming_all)(*operands)


def test_mixed_precision_preserves_reference_promotion() -> None:
    operands = _operands(113, 7, 27, 97, 3, 3, 3, jnp.float64)
    mixed = operands[:-1] + (operands[-1].astype(jnp.float32),)
    with jax.numpy_dtype_promotion("standard"):
        actual = jax.jit(_streaming_all)(*mixed)
        expected = jax.jit(_reference_all)(*mixed)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)


@_DTYPES
def test_index_addressing_matches_native_reference(dtype: DTypeLike) -> None:
    indices = jnp.asarray([[-1], [6], [-6], [-1], [6], [-6]], dtype=jnp.int32)
    valid = jnp.asarray(
        [[True], [True], [True], [False], [False], [False]], dtype=jnp.bool_
    )
    weights = jnp.ones((6, 1), dtype=dtype)
    metadata = jnp.ones((6, 1, 1), dtype=dtype)
    grid = jnp.arange(5, dtype=dtype).reshape((5, 1))
    operands = (indices, valid, weights, metadata, metadata, grid)
    actual = jax.jit(_streaming_all)(*operands)
    expected = jax.jit(_reference_all)(*operands)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)
    np.testing.assert_array_equal(actual[0][0], grid[-1])
    np.testing.assert_array_equal(actual[0][3:], jnp.zeros((3, 1), dtype=dtype))


def _route_devices() -> tuple[jax.Device, ...]:
    return (jax.devices("cpu")[0],) + tuple(
        device for device in jax.devices() if device.platform == "gpu"
    )


@_DTYPES
def test_cpu_and_cuda_routes_agree(dtype: DTypeLike) -> None:
    host = _operands(5, 130, 27, 97, 3, 3, 3, dtype)
    results = []
    for device in _route_devices():
        with jax.default_device(device):
            operands = jax.device_put(host, device)
            result = jax.jit(_streaming_all)(*operands)
        assert all(value.devices() == {device} for value in result)
        _assert_moments(result, host, dtype)
        results.append(result)
    _, magnitudes = _host_moments(host)
    eps = float(jnp.finfo(dtype).eps)
    for result in results[1:]:
        for value, cpu, magnitude in zip(result, results[0], magnitudes, strict=True):
            # Two summation orders each within the host bound differ by at most twice it.
            np.testing.assert_array_less(
                np.abs(np.asarray(value, np.float64) - np.asarray(cpu, np.float64)),
                2 * 30 * eps * magnitude + 2 * np.finfo(np.dtype(dtype)).tiny,
            )


@_DTYPES
def test_masks_nonfinite_metadata_and_zero_weights(dtype: DTypeLike) -> None:
    indices, valid, weights, gradients, offsets, grid = (
        np.array(value) for value in _operands(11, 9, 27, 40, 3, 3, 3, dtype)
    )
    valid[:] = True
    valid[1, 4] = valid[2, 5] = valid[3, 6] = valid[5, 7] = False
    valid[6] = False
    indices[valid] = np.abs(indices[valid]) % 39
    # Node 39 is read only by invalid routes; its NaN and Inf must stay inert.
    indices[5, 7] = 39
    grid[39] = (np.nan, np.inf, -np.inf)
    gradients[1, 4] = np.nan  # 0 * NaN: metadata remains explicit.
    offsets[2, 5] = np.inf  # 0 * Inf: metadata remains explicit.
    weights[3, 6] = np.nan  # masked weight never multiplies.
    weights[4] = 0.0  # zero valid weights still route velocity gradients.
    operands = tuple(
        jnp.asarray(value)
        for value in (indices, valid, weights, gradients, offsets, grid)
    )
    velocity, velocity_gradient, affine_moment = jax.jit(_streaming_all)(*operands)
    _assert_moments((velocity, velocity_gradient, affine_moment), operands, dtype)
    _assert_moments(jax.jit(_reference_all)(*operands), operands, dtype)

    assert bool(jnp.all(jnp.isfinite(velocity)))
    assert bool(jnp.all(jnp.isnan(velocity_gradient[1])))
    assert bool(jnp.all(jnp.isfinite(jnp.delete(velocity_gradient, 1, axis=0))))
    assert bool(jnp.all(jnp.isnan(affine_moment[2])))
    assert bool(jnp.all(jnp.isfinite(jnp.delete(affine_moment, 2, axis=0))))
    np.testing.assert_array_equal(velocity[4], 0.0)
    np.testing.assert_array_equal(affine_moment[4], 0.0)
    assert float(jnp.max(jnp.abs(velocity_gradient[4]))) > 0.0
    for value in (velocity[6], velocity_gradient[6], affine_moment[6]):
        np.testing.assert_array_equal(value, jnp.zeros_like(value))


def _loss(
    moments: Callable[..., Moments],
    probes: tuple[Array, Array, Array],
    indices: Array,
    valid: Array,
    weights: Array,
    gradients: Array,
    offsets: Array,
    grid: Array,
) -> Array:
    velocity, velocity_gradient, affine_moment = moments(
        indices, valid, weights, gradients, offsets, grid
    )
    return (
        jnp.sum(probes[0] * velocity)
        + 0.5 * jnp.sum(velocity * velocity)
        + jnp.sum(probes[1] * velocity_gradient * velocity_gradient)
        + jnp.sum(probes[2] * affine_moment)
        + 0.5 * jnp.sum(affine_moment * affine_moment)
    )


def _derivatives(
    moments: Callable[..., Moments],
    probes: tuple[Array, Array, Array],
    operands: Operands,
    directions: tuple[Array, Array, Array, Array],
) -> tuple[object, ...]:
    indices, valid, *metadata = operands

    def loss(*values: Array) -> Array:
        return _loss(moments, probes, indices, valid, *values)

    argnums = (0, 1, 2, 3)
    value, gradient = jax.value_and_grad(loss, argnums=argnums)(*metadata)
    _, tangent = jax.jvp(loss, tuple(metadata), directions)
    _, hvp = jax.jvp(
        lambda *values: jax.grad(loss, argnums=argnums)(*values),
        tuple(metadata),
        directions,
    )
    return value, gradient, tangent, hvp


def _probes(operands: Operands, seed: int) -> tuple[Array, Array, Array]:
    rng = np.random.default_rng(seed)
    particles, components = operands[0].shape[0], operands[5].shape[1]
    gradient_width, offset_width = operands[3].shape[2], operands[4].shape[2]
    dtype = operands[5].dtype
    return (
        jnp.asarray(rng.normal(size=(particles, components)), dtype=dtype),
        jnp.asarray(
            rng.normal(size=(particles, components, gradient_width)), dtype=dtype
        ),
        jnp.asarray(rng.normal(size=(particles, components, offset_width)), dtype=dtype),
    )


@pytest.mark.parametrize(
    "shape",
    (_SHAPES["tile-tail"], _SHAPES["multi-tile-tail"], _SHAPES["mixed-widths"]),
    ids=("tile-tail", "multi-tile-tail", "mixed-widths"),
)
def test_gradient_jvp_and_hvp_through_metadata_match_reference(
    shape: Shape,
) -> None:
    operands = _operands(23, *shape, jnp.float64)
    probes = _probes(operands, 29)
    rng = np.random.default_rng(31)
    directions = tuple(
        jnp.asarray(rng.normal(size=value.shape), dtype=jnp.float64)
        for value in operands[2:]
    )
    actual = jax.jit(_derivatives, static_argnums=0)(
        _streaming_all, probes, operands, directions
    )
    expected = jax.jit(_derivatives, static_argnums=0)(
        _reference_all, probes, operands, directions
    )
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert bool(jnp.all(jnp.isfinite(left)))
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10)

    # Zero weights are not skipped: d(sum probe.v)/dw_pr = probe_p . u(i_pr).
    indices, valid, weights, gradients, offsets, grid = operands
    gradient = jax.grad(
        lambda w: jnp.sum(
            probes[0] * streaming_moments(indices, valid, w, gradients, offsets, grid)[0]
        )
    )(weights)
    route = jnp.where(
        valid[..., None], grid[jnp.where(valid, indices, 0)], jnp.zeros((), grid.dtype)
    )
    analytic = jnp.where(valid, jnp.sum(probes[0][:, None, :] * route, axis=-1), 0.0)
    zero = np.asarray(valid & (weights == 0.0))
    assert zero.any()
    np.testing.assert_allclose(gradient, analytic, rtol=1e-12, atol=1e-12)
    assert np.all(np.asarray(gradient)[zero] != 0.0)


_SUBSETS = tuple(
    subset for size in (1, 2, 3) for subset in itertools.combinations(_ALL, size)
)


@_DTYPES
@pytest.mark.parametrize("outputs", _SUBSETS, ids=lambda s: "-".join(map(str, s)))
def test_output_specific_consumption(outputs: tuple[int, ...], dtype: DTypeLike) -> None:
    operands = _operands(41, 130, 27, 97, 3, 3, 3, dtype)
    bound = jax.jit(lambda *ops: tuple(_moments_p.bind(*ops, outputs=outputs)))(*operands)
    _assert_moments(bound, operands, dtype, outputs)
    referenced = jax.jit(lambda *ops: _reference(*ops, outputs=outputs))(*operands)
    _assert_moments(referenced, operands, dtype, outputs)
    consumed = jax.jit(
        lambda *ops: tuple(streaming_moments(*ops)[output] for output in outputs)
    )(*operands)
    _assert_moments(consumed, operands, dtype, outputs)

    # Metadata a consumed subset cannot read has an exactly zero cotangent.
    def loss(weights: Array, gradients: Array, offsets: Array, grid: Array) -> Array:
        values = streaming_moments(
            operands[0], operands[1], weights, gradients, offsets, grid
        )
        total = jnp.zeros((), dtype=dtype)
        for output in outputs:
            total = total + jnp.sum(values[output] ** 2)
        return total

    weights_bar, gradients_bar, offsets_bar, grid_bar = jax.jit(
        jax.grad(loss, argnums=(0, 1, 2, 3))
    )(*operands[2:])
    if 1 not in outputs:
        np.testing.assert_array_equal(gradients_bar, jnp.zeros_like(gradients_bar))
    if 2 not in outputs:
        np.testing.assert_array_equal(offsets_bar, jnp.zeros_like(offsets_bar))
    if outputs == (1,):
        np.testing.assert_array_equal(weights_bar, jnp.zeros_like(weights_bar))
    assert float(jnp.max(jnp.abs(grid_bar))) > 0.0


_BATCH_AXES = {
    "grid": (None, None, None, None, None, 0),
    "metadata": (None, None, 0, 0, 0, None),
    "routes": (0, 0, None, None, None, None),
    "all": (0, 0, 0, 0, 0, 0),
}


@_DTYPES
@pytest.mark.parametrize("in_axes", tuple(_BATCH_AXES.values()), ids=tuple(_BATCH_AXES))
def test_batching_matches_host_oracle_and_reference_derivatives(
    in_axes: tuple[int | None, ...], dtype: DTypeLike
) -> None:
    batches = tuple(_operands(53 + seed, 7, 27, 40, 3, 3, 3, dtype) for seed in range(3))
    batched = tuple(
        jnp.stack([batch[position] for batch in batches])
        if axis == 0
        else batches[0][position]
        for position, axis in enumerate(in_axes)
    )
    result = jax.jit(jax.vmap(_streaming_all, in_axes=in_axes))(*batched)
    for member in range(len(batches)):
        operands = tuple(
            value[member] if axis == 0 else value
            for value, axis in zip(batched, in_axes, strict=True)
        )
        _assert_moments(tuple(value[member] for value in result), operands, dtype)

    if np.dtype(dtype) != np.dtype(np.float64):
        return
    probes = _probes(batches[0], 59)

    def gradient(moments: Callable[..., Moments], *operands: Array) -> tuple[Array, ...]:
        indices, valid, *metadata = operands
        return jax.grad(
            lambda *values: _loss(moments, probes, indices, valid, *values),
            argnums=(0, 1, 2, 3),
        )(*metadata)

    actual = jax.jit(
        jax.vmap(lambda *ops: gradient(_streaming_all, *ops), in_axes=in_axes)
    )(*batched)
    expected = jax.jit(
        jax.vmap(lambda *ops: gradient(_reference_all, *ops), in_axes=in_axes)
    )(*batched)
    for left, right in zip(actual, expected, strict=True):
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10)


@_DTYPES
def test_one_executable_follows_changing_routes_and_values(
    dtype: DTypeLike,
) -> None:
    compiled = jax.jit(_streaming_all)
    for seed in range(4):
        operands = _operands(71 + seed, 130, 27, 97, 3, 3, 3, dtype)
        _assert_moments(compiled(*operands), operands, dtype)


def _structural_operands() -> tuple[np.ndarray, np.ndarray, Operands]:
    indices = np.asarray(((0, 2, 5), (1, 1, 4), (3, 0, 2), (5, 4, 3)), dtype=np.int32)
    valid = np.asarray(
        (
            (True, True, False),
            (True, True, True),
            (False, False, False),
            (True, False, True),
        )
    )
    garbage = np.where(valid, indices, 17).astype(np.int32)
    rng = np.random.default_rng(83)
    operands = (
        jnp.asarray(garbage),
        jnp.asarray(valid),
        jnp.asarray(rng.uniform(0.25, 1.0, size=(4, 3)), dtype=jnp.float64),
        jnp.asarray(rng.normal(size=(4, 3, 2)), dtype=jnp.float64),
        jnp.asarray(rng.normal(size=(4, 3, 2)), dtype=jnp.float64),
        jnp.asarray(rng.normal(size=(6, 2)), dtype=jnp.float64),
    )
    return indices, valid, operands


def _pattern(
    function: Callable[[Array, None], Array],
    coordinates: Array,
    target_size: int,
    derivative_kind: DerivativeKind,
) -> set[tuple[int, int]]:
    pattern = trace_sparse_pattern(
        function,
        coordinates,
        None,
        source_size=coordinates.size,
        target_size=target_size,
        derivative_kind=derivative_kind,
    )
    return set(zip(np.asarray(pattern.rows).tolist(), np.asarray(pattern.cols).tolist()))


def test_structural_jacobians_follow_valid_routes_only() -> None:
    indices, valid, operands = _structural_operands()
    route_indices, route_valid, weights, gradients, offsets, grid = operands
    components = 2

    def velocity_of_grid(values: Array, args: None) -> Array:
        del args
        moments = streaming_moments(
            route_indices, route_valid, weights, gradients, offsets, values.reshape(6, 2)
        )
        return moments[0].reshape(-1)

    expected = {
        (
            particle * components + component,
            int(indices[particle, route]) * components + component,
        )
        for particle, route in zip(*np.nonzero(valid), strict=True)
        for component in range(components)
    }
    assert _pattern(velocity_of_grid, grid.reshape(-1), 8, "jacobian") == expected

    def velocity_of_weights(values: Array, args: None) -> Array:
        del args
        moments = streaming_moments(
            route_indices, route_valid, values.reshape(4, 3), gradients, offsets, grid
        )
        return moments[0].reshape(-1)

    expected = {
        (particle * components + component, particle * 3 + route)
        for particle, route in zip(*np.nonzero(valid), strict=True)
        for component in range(components)
    }
    assert _pattern(velocity_of_weights, weights.reshape(-1), 8, "jacobian") == expected

    def gradient_moment_of_grid(values: Array, args: None) -> Array:
        del args
        moments = streaming_moments(
            route_indices, route_valid, weights, gradients, offsets, values.reshape(6, 2)
        )
        return moments[1].reshape(-1)

    expected = {
        (
            (particle * components + component) * 2 + axis,
            int(indices[particle, route]) * components + component,
        )
        for particle, route in zip(*np.nonzero(valid), strict=True)
        for component in range(components)
        for axis in range(2)
    }
    assert _pattern(gradient_moment_of_grid, grid.reshape(-1), 16, "jacobian") == expected


def test_structural_hessian_couples_only_shared_particle_routes() -> None:
    indices, valid, operands = _structural_operands()
    route_indices, route_valid, weights, gradients, offsets, _ = operands
    components = 2

    def energy(values: Array, args: None) -> Array:
        del args
        velocity, velocity_gradient, _ = streaming_moments(
            route_indices, route_valid, weights, gradients, offsets, values.reshape(6, 2)
        )
        return 0.5 * jnp.sum(velocity * velocity) + jnp.sum(
            velocity_gradient * velocity_gradient
        )

    expected = set()
    for particle in range(indices.shape[0]):
        nodes = indices[particle][valid[particle]].tolist()
        for left, right in itertools.product(nodes, nodes):
            for component in range(components):
                expected.add(
                    (left * components + component, right * components + component)
                )
    coordinates = operands[5].reshape(-1)
    assert _pattern(energy, coordinates, coordinates.size, "hessian") == expected


def test_structural_batching_keeps_members_independent() -> None:
    indices, valid, operands = _structural_operands()
    route_indices, route_valid, weights, gradients, offsets, grid = operands
    members, components = 3, 2

    def velocity(values: Array, args: None) -> Array:
        del args
        moments = jax.vmap(_streaming_all, in_axes=(None, None, None, None, None, 0))(
            route_indices,
            route_valid,
            weights,
            gradients,
            offsets,
            values.reshape(3, 6, 2),
        )
        return moments[0].reshape(-1)

    expected = {
        (
            (member * 4 + particle) * components + component,
            (member * 6 + int(indices[particle, route])) * components + component,
        )
        for member in range(members)
        for particle, route in zip(*np.nonzero(valid), strict=True)
        for component in range(components)
    }
    coordinates = jnp.tile(grid.reshape(-1), members)
    assert _pattern(velocity, coordinates, members * 8, "jacobian") == expected


def test_foreign_same_name_primitive_is_refused() -> None:
    from jax.extend import core

    from phydrax._identity import execution_metadata_payload

    foreign = core.Primitive(_moments_p.name)
    foreign.def_impl(lambda value: value)
    foreign.def_abstract_eval(lambda value: value)
    values = jnp.ones(12, dtype=jnp.float64)
    with pytest.raises(TypeError, match="Foreign compiler primitive"):
        execution_metadata_payload(jax.make_jaxpr(foreign.bind)(values))
    with pytest.raises(NotImplementedError, match="Foreign APIC streaming moment"):
        _pattern(lambda value, args: foreign.bind(value), values, 12, "jacobian")


def test_compiler_identity_tracks_signature_not_values() -> None:
    from phydrax._identity import execution_metadata_payload

    def payload(operands: Operands, outputs: tuple[int, ...]) -> object:
        return execution_metadata_payload(
            jax.make_jaxpr(lambda *ops: _moments_p.bind(*ops, outputs=outputs))(*operands)
        )

    first = _operands(89, 7, 27, 40, 3, 3, 3, jnp.float64)
    second = _operands(97, 7, 27, 40, 3, 3, 3, jnp.float64)
    baseline = payload(first, _ALL)
    assert baseline == payload(second, _ALL)
    assert baseline != payload(first, (0, 2))
    assert baseline != payload(_operands(89, 9, 27, 40, 3, 3, 3, jnp.float64), _ALL)
    assert baseline != payload(_operands(89, 7, 27, 40, 3, 3, 3, jnp.float32), _ALL)
