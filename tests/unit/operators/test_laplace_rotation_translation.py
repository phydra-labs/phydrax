from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.operators.integral.multipole import (
    LaplaceMultipolePlan3D,
    PreparedLaplaceMultipole3D,
    TranslationRoute3D,
)
from tests._support.differentiation import assert_coordinate_transpose_duality


_POINTS = jnp.asarray(
    [[0.02, -0.015, 0.01], [-0.01, 0.02, -0.015], [0.005, 0.01, 0.025]],
    dtype=jnp.float64,
)
_STRENGTHS = jnp.asarray(
    [
        [[0.8 + 0.3j, -0.2 + 0.1j], [0.4 - 0.2j, 0.3j]],
        [[-0.3 + 0.1j, 0.4 - 0.2j], [0.1 + 0.4j, 0.2]],
        [[0.5 - 0.2j, 0.2 + 0.5j], [-0.2 + 0.1j, 0.6j]],
    ],
    dtype=jnp.complex128,
)
_ZERO = jnp.zeros((3,), dtype=jnp.float64)


def _prepared(
    order: int, route: TranslationRoute3D = "rotation"
) -> PreparedLaplaceMultipole3D:
    return LaplaceMultipolePlan3D(
        _POINTS,
        jnp.full((3,), -4.0, dtype=jnp.float64),
        jnp.full((3,), 4.0, dtype=jnp.float64),
        reference_targets=jnp.asarray([[1.7, 0.2, -0.1]], dtype=jnp.float64),
        depth=2,
        expansion_order=order,
        translation_route=route,
    ).prepare()


def _modal(prepared: PreparedLaplaceMultipole3D) -> Array:
    rng = np.random.default_rng(410)
    shape = prepared.layout.coefficient_shape + (2,)
    values = jnp.asarray(
        rng.normal(size=shape) + 1j * rng.normal(size=shape), dtype=jnp.complex128
    )
    return (
        prepared.layout.valid_mask[..., None]
        * values
        / 3.0 ** jnp.arange(prepared.layout.bandlimit, dtype=jnp.float64)[:, None, None]
    )


def _realify(value: Array) -> Array:
    return jnp.concatenate((jnp.real(value).reshape(-1), jnp.imag(value).reshape(-1)))


def _direct(points: Array, strengths: Array, targets: Array) -> Array:
    radius = jnp.linalg.norm(targets[:, None, :] - points[None, :, :], axis=-1)
    return jnp.tensordot(1.0 / (4.0 * jnp.pi * radius), strengths, axes=((1,), (0,)))


@pytest.mark.parametrize("order", [0, 1, 2, 4, 8])
def test_native_moments_polynomials_and_composition(order: int) -> None:
    prepared = _prepared(order)
    dense = _prepared(order, "dense")
    parent = jnp.asarray([0.17, -0.11, 0.09], dtype=jnp.float64)
    grandparent = jnp.asarray([-0.13, 0.04, -0.08], dtype=jnp.float64)
    modal = prepared.p2m(_POINTS, _STRENGTHS, _ZERO)
    translated = prepared.m2m(modal, _ZERO, parent)
    expected = prepared.p2m(_POINTS, _STRENGTHS, parent)
    np.testing.assert_allclose(translated, expected, rtol=3e-12, atol=3e-15)
    np.testing.assert_allclose(
        translated, dense.m2m(modal, _ZERO, parent), rtol=3e-12, atol=3e-15
    )
    np.testing.assert_allclose(
        prepared.m2m(translated, parent, grandparent),
        prepared.m2m(modal, _ZERO, grandparent),
        rtol=4e-12,
        atol=5e-15,
    )
    np.testing.assert_array_equal(prepared.m2m(modal, _ZERO, _ZERO), modal)
    local = _modal(prepared)
    shifted = prepared.l2l(local, _ZERO, parent)
    points = parent + _POINTS
    np.testing.assert_allclose(
        prepared.l2p(shifted, parent, points),
        prepared.l2p(local, _ZERO, points),
        rtol=4e-12,
        atol=3e-14,
    )
    np.testing.assert_allclose(
        prepared.l2l(shifted, parent, grandparent),
        prepared.l2l(local, _ZERO, grandparent),
        rtol=4e-12,
        atol=3e-14,
    )
    np.testing.assert_array_equal(prepared.l2l(local, _ZERO, _ZERO), local)
    assert prepared.resources.quadrature_node_count == 0
    assert prepared.resources.translation_table_bytes > 0
    assert prepared.plan.plan_id != dense.plan.plan_id
    assert prepared.prepared_id != dense.prepared_id


def test_low_degree_native_normalization() -> None:
    prepared = _prepared(2)
    center = jnp.asarray([0.12, -0.07, 0.04], dtype=jnp.float64)
    relative = _POINTS - center
    x, y, z = relative.T
    expected = jnp.zeros((3, 5, 2, 2), dtype=jnp.complex128)
    # Explicit orthonormal Condon--Shortley polynomials, independent of the
    # translation tables and the harmonic recurrence used by p2m.
    regular = {
        (0, 0): jnp.ones_like(x) / jnp.sqrt(4.0 * jnp.pi),
        (1, 0): jnp.sqrt(3.0 / (4.0 * jnp.pi)) * z,
        (1, 1): -jnp.sqrt(3.0 / (8.0 * jnp.pi)) * (x + 1j * y),
        (2, 0): jnp.sqrt(5.0 / (16.0 * jnp.pi)) * (2.0 * z * z - x * x - y * y),
        (2, 1): -jnp.sqrt(15.0 / (8.0 * jnp.pi)) * (x + 1j * y) * z,
        (2, 2): jnp.sqrt(15.0 / (32.0 * jnp.pi)) * (x + 1j * y) ** 2,
    }
    for (ell, m), harmonic in regular.items():
        value = jnp.tensordot(
            jnp.conj(harmonic) / (2 * ell + 1), _STRENGTHS, axes=((0,), (0,))
        )
        expected = expected.at[ell, 2 + m].set(value)
        if m > 0:
            negative = jnp.tensordot(
                (-1.0) ** m * harmonic / (2 * ell + 1), _STRENGTHS, axes=((0,), (0,))
            )
            expected = expected.at[ell, 2 - m].set(negative)
    original = prepared.p2m(_POINTS, _STRENGTHS, _ZERO)
    np.testing.assert_allclose(
        prepared.m2m(original, _ZERO, center), expected, rtol=3e-13, atol=2e-15
    )


@pytest.mark.parametrize("order", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_axial_m2l_monopole_and_direct_sum(order: int, sign: float) -> None:
    prepared = _prepared(order)
    center = jnp.asarray([0.0, 0.0, sign * 1.7], dtype=jnp.float64)
    strengths = jnp.sum(_STRENGTHS, axis=0, keepdims=True)
    monopole = prepared.p2m(_ZERO[None, :], strengths, _ZERO)
    np.testing.assert_allclose(
        prepared.m2l(monopole, _ZERO, center),
        prepared.p2l(_ZERO[None, :], strengths, center),
        rtol=3e-12,
        atol=4e-15,
    )
    multipole = prepared.p2m(_POINTS, _STRENGTHS, _ZERO)
    local = prepared.m2l(multipole, _ZERO, center)
    targets = center + 0.3 * _POINTS
    expected = _direct(_POINTS, _STRENGTHS, targets)
    actual = prepared.l2p(local, center, targets)
    absolute_scale = _direct(_POINTS, jnp.abs(_STRENGTHS), targets)
    # Independent geometric envelope: source radius plus evaluation radius is
    # below 0.04*R, with both moment and local tails included.
    np.testing.assert_array_less(
        jnp.abs(actual - expected), 3.0 * 0.04 ** (order + 1) * absolute_scale + 2e-15
    )


@pytest.mark.parametrize("order", [0, 1, 2, 4])
def test_dense_m2l_and_l2l_controls(order: int) -> None:
    prepared, dense = _prepared(order), _prepared(order, "dense")
    modal = _modal(prepared)
    center = jnp.asarray([0.23, -0.31, 1.9], dtype=jnp.float64)
    np.testing.assert_allclose(
        prepared.m2l(modal, _ZERO, center),
        dense.m2l(modal, _ZERO, center),
        rtol=8e-12,
        atol=3e-14,
    )
    shift = center / 5.0
    np.testing.assert_allclose(
        prepared.l2l(modal, _ZERO, shift),
        dense.l2l(modal, _ZERO, shift),
        rtol=8e-12,
        atol=3e-14,
    )


@pytest.mark.parametrize("order", [2, 8])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_axis_approach_derivatives_and_rotational_covariance(
    order: int, sign: float
) -> None:
    prepared = _prepared(order)
    modal = _modal(prepared)
    direction = jnp.asarray([0.31, -0.23, 0.17], dtype=jnp.float64)
    for transverse in (0.1, 1e-3, 1e-6, 1e-10, 0.0):
        shift = jnp.asarray(
            [transverse, -0.7 * transverse, sign * 2.1], dtype=jnp.float64
        )
        function = lambda displacement: prepared.m2l(modal, _ZERO, displacement)
        _, tangent = jax.jvp(function, (shift,), (direction,))
        step = 2e-5
        finite_difference = (
            function(shift + step * direction) - function(shift - step * direction)
        ) / (2.0 * step)
        np.testing.assert_allclose(tangent, finite_difference, rtol=3e-7, atol=2e-10)
        monopole = prepared.p2m(
            _ZERO[None, :], jnp.asarray([0.7 + 0.3j], dtype=jnp.complex128), _ZERO
        )
        reference = lambda displacement: prepared.p2l(
            _ZERO[None, :], jnp.asarray([0.7 + 0.3j], dtype=jnp.complex128), displacement
        )
        _, expected = jax.jvp(reference, (shift,), (direction,))
        _, actual = jax.jvp(
            lambda displacement: prepared.m2l(monopole, _ZERO, displacement),
            (shift,),
            (direction,),
        )
        np.testing.assert_allclose(actual, expected, rtol=8e-12, atol=3e-14)
    rotation = phx.discretization.SphericalRotationPlan(prepared.layout).prepare()
    angles = jnp.asarray([0.4, 0.6, -0.2], dtype=jnp.float64)
    alpha, beta, gamma = np.asarray(angles)

    def rz(angle: float) -> np.ndarray:
        return np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    ry = np.asarray(
        [
            [np.cos(beta), 0.0, np.sin(beta)],
            [0.0, 1.0, 0.0],
            [-np.sin(beta), 0.0, np.cos(beta)],
        ],
        dtype=np.float64,
    )
    matrix = jnp.asarray(rz(float(alpha)) @ ry @ rz(float(gamma)), dtype=jnp.float64)
    shift = jnp.asarray([0.2, -0.3, sign * 2.1], dtype=jnp.float64)
    rotated = rotation.apply(modal, angles)
    for kind in ("m2m", "m2l", "l2l"):
        translate = {"m2m": prepared.m2m, "m2l": prepared.m2l, "l2l": prepared.l2l}[kind]
        if kind == "m2m":
            original = translate(modal, shift, _ZERO)
            actual = translate(rotated, matrix @ shift, _ZERO)
        else:
            original = translate(modal, _ZERO, shift)
            actual = translate(rotated, _ZERO, matrix @ shift)
        np.testing.assert_allclose(
            actual, rotation.apply(original, angles), rtol=2e-11, atol=2e-12
        )


@pytest.mark.parametrize("kind", ["m2m", "l2l"])
def test_zero_shift_nonzero_first_and_mixed_displacement_actions(kind: str) -> None:
    prepared, dense = _prepared(2), _prepared(2, "dense")
    modal = _modal(prepared)
    first = jnp.asarray([0.3, -0.2, 0.4], dtype=jnp.float64)
    second = jnp.asarray([-0.1, 0.5, 0.2], dtype=jnp.float64)
    if kind == "m2m":
        function = lambda shift: prepared.m2m(modal, shift, _ZERO)
        reference = lambda shift: dense.m2m(modal, shift, _ZERO)
    else:
        function = lambda shift: prepared.l2l(modal, _ZERO, shift)
        reference = lambda shift: dense.l2l(modal, _ZERO, shift)
    _, tangent = jax.jvp(function, (_ZERO,), (first,))
    _, expected = jax.jvp(reference, (_ZERO,), (first,))
    assert float(jnp.linalg.norm(tangent)) > 1e-3
    np.testing.assert_allclose(tangent, expected, rtol=8e-12, atol=3e-14)

    def directional(shift: Array) -> Array:
        return jax.jvp(function, (shift,), (first,))[1]

    def reference_directional(shift: Array) -> Array:
        return jax.jvp(reference, (shift,), (first,))[1]

    np.testing.assert_allclose(
        jax.jvp(directional, (_ZERO,), (second,))[1],
        jax.jvp(reference_directional, (_ZERO,), (second,))[1],
        rtol=8e-12,
        atol=3e-14,
    )


@pytest.mark.parametrize(
    ("kind", "order", "sign"),
    [
        (kind, order, sign)
        for kind, order in (("m2m", 4), ("m2l", 4), ("l2l", 4), ("m2l", 8))
        for sign in (-1.0, 1.0)
    ]
    + [("m2m", 4, 0.0), ("l2l", 4, 0.0)],
)
def test_geometry_strength_duality_and_mixed_actions(
    kind: str, order: int, sign: float
) -> None:
    prepared = _prepared(order)
    shift = jnp.asarray([0.0, 0.0, sign * 1.9], dtype=jnp.float64)
    state = jnp.concatenate(
        (_ZERO, jnp.asarray([0.8, 0.3], dtype=jnp.float64), _ZERO, shift)
    )

    def function(value: Array) -> Array:
        points = _POINTS + value[:3]
        strength = (value[3] + 1j * value[4]) * _STRENGTHS
        modal = prepared.p2m(points, strength, value[5:8])
        match kind:
            case "m2m":
                translated = prepared.m2m(modal, value[5:8], value[8:11])
            case "m2l":
                translated = prepared.m2l(modal, value[5:8], value[8:11])
            case "l2l":
                translated = prepared.l2l(modal, value[5:8], value[8:11])
            case _:
                raise ValueError(kind)
        return _realify(translated)

    output, pushforward = jax.linearize(function, state)
    _, pullback = jax.vjp(function, state)
    tangent = (
        jnp.asarray(np.random.default_rng(11).normal(size=state.shape), dtype=jnp.float64)
        / 10.0
    )
    covector = jnp.asarray(
        np.random.default_rng(12).normal(size=output.shape), dtype=jnp.float64
    )

    def transpose(value: Array) -> Array:
        return pullback(value)[0]

    operator = phx.linalg.FunctionLinearOperator(
        pushforward,
        source=phx.linalg.ArraySpace(state.shape, dtype=jnp.float64),
        target=phx.linalg.ArraySpace(output.shape, dtype=jnp.float64),
        transpose_action=transpose,
        closure_convert=False,
    )
    assert_coordinate_transpose_duality(
        operator, tangent, covector, rtol=3e-12, atol=3e-13
    )
    objective: Callable[[Array], Array] = lambda value: jnp.sum(
        function(value) * covector
    )
    gradient = jax.grad(objective)
    mixed = jax.jvp(gradient, (state,), (tangent,))[1]
    step = 2e-5
    expected = (gradient(state + step * tangent) - gradient(state - step * tangent)) / (
        2.0 * step
    )
    np.testing.assert_allclose(mixed, expected, rtol=3e-7, atol=2e-9)
    reverse_mixed = jax.grad(lambda value: jax.jvp(objective, (value,), (tangent,))[1])(
        state
    )
    np.testing.assert_allclose(mixed, reverse_mixed, rtol=3e-12, atol=3e-13)


def test_complete_rotation_evaluation_and_geometry_gradient() -> None:
    sources = jnp.concatenate((_POINTS - 0.7, _POINTS + 0.6), axis=0)
    targets = jnp.asarray(
        [[-0.62, -0.65, -0.61], [0.68, 0.64, 0.67], [0.1, -0.15, 0.2]], dtype=jnp.float64
    )
    strengths = jnp.concatenate((_STRENGTHS, 0.3j * _STRENGTHS), axis=0)

    def prepare(route: TranslationRoute3D) -> PreparedLaplaceMultipole3D:
        return LaplaceMultipolePlan3D(
            sources,
            jnp.full((3,), -1.0, dtype=jnp.float64),
            jnp.full((3,), 1.0, dtype=jnp.float64),
            reference_targets=targets,
            depth=3,
            expansion_order=4,
            source_leaf_occupancy=1,
            maximum_reference_displacement=0.02,
            translation_route=route,
        ).prepare()

    rotation, dense = prepare("rotation"), prepare("dense")
    result = eqx.filter_jit(rotation.evaluate)(sources, strengths, targets)
    control = dense.evaluate(sources, strengths, targets)
    assert bool(result.successful)
    assert int(result.m2m_count) > 0
    assert int(result.m2l_count) > 0
    np.testing.assert_allclose(result.values, control.values, rtol=8e-12, atol=2e-13)
    expected = _direct(sources, strengths, targets)
    scale = _direct(sources, jnp.abs(strengths), targets)
    np.testing.assert_array_less(jnp.abs(result.values - expected), 1e-3 * scale)
    objective = lambda points: jnp.real(
        jnp.sum(rotation.evaluate(points, strengths, targets).values)
    )
    direct_objective = lambda points: jnp.real(
        jnp.sum(_direct(points, strengths, targets))
    )
    np.testing.assert_allclose(
        jax.grad(objective)(sources),
        jax.grad(direct_objective)(sources),
        rtol=3e-3,
        atol=2e-4,
    )


def test_rotation_resource_and_kernel_refusals() -> None:
    with pytest.raises(ValueError, match="at most 8"):
        _prepared(9)
    with pytest.raises(ValueError, match="maximum_coefficient_bytes"):
        LaplaceMultipolePlan3D(
            _POINTS,
            jnp.full((3,), -4.0, dtype=jnp.float64),
            jnp.full((3,), 4.0, dtype=jnp.float64),
            depth=2,
            expansion_order=4,
            translation_route="rotation",
            maximum_coefficient_bytes=64,
        ).prepare()
    prepared = _prepared(2)
    modal = prepared.p2m(_POINTS, _STRENGTHS, _ZERO)
    for center in (_ZERO, jnp.asarray([0.0, 0.0, 1e-9], dtype=jnp.float64)):
        with pytest.raises(eqx.EquinoxRuntimeError, match="noncoincident"):
            prepared.m2l(modal, _ZERO, center)
    with pytest.raises(eqx.EquinoxRuntimeError, match="radius envelope"):
        prepared.m2m(modal, _ZERO, jnp.asarray([1e9, 0.0, 0.0], dtype=jnp.float64))
    for plan_type, keyword in (
        (phx.operators.HelmholtzMultipolePlan3D, {"wavenumber": 0.7}),
        (phx.operators.ModifiedHelmholtzMultipolePlan3D, {"decay": 0.7}),
    ):
        with pytest.raises(ValueError, match="do not support.*rotation"):
            plan_type(
                _POINTS,
                (-1.0, -1.0, -1.0),
                (1.0, 1.0, 1.0),
                translation_route="rotation",
                **keyword,
            )


@pytest.mark.parametrize("order", [4, 8])
def test_rotation_vector_workspace_admits_benchmark_cloud(order: int) -> None:
    rng = np.random.default_rng(390174)
    sources = np.asarray(rng.uniform(-0.9, 0.9, size=(256, 3)), dtype=np.float64)
    targets = np.asarray(rng.uniform(-0.9, 0.9, size=(12, 3)), dtype=np.float64)
    prepared = LaplaceMultipolePlan3D(
        sources,
        jnp.full((3,), -1.0, dtype=jnp.float64),
        jnp.full((3,), 1.0, dtype=jnp.float64),
        reference_targets=targets,
        depth=2,
        expansion_order=order,
        translation_route="rotation",
    ).prepare()
    resources = prepared.resources
    assert resources.within_budget
    assert resources.quadrature_node_count == 0
    assert resources.required_coefficient_bytes >= (
        resources.far_interaction_capacity * resources.coefficient_bytes_per_expansion
    )
    assert (
        resources.required_coefficient_bytes
        + resources.translation_table_bytes
        + resources.translation_workspace_bytes
        <= resources.maximum_coefficient_bytes
    )
