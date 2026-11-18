# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from phydrax import config
from phydrax.linalg import ArraySpace, OperatorProperties
from phydrax.optim import cone_barrier_oracle, ConicProgram, NonnegativeCone
from phydrax.optim._programming._native_hsd import _embedding_residual
from phydrax.optim._programming._problem import _conic_quadratic_perspective
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


pytestmark = pytest.mark.strict_jax

_Q = np.asarray([[2.0, 0.5], [0.5, 1.5]], dtype=np.float64)
_X = np.asarray([1.0, -0.5], dtype=np.float64)
_TAU = 0.75
_DX = np.asarray([0.3, -0.2], dtype=np.float64)
_DTAU = 0.1


def _quadratic(kind: str) -> Array | SparseCoordinateOperator | None:
    if kind == "none":
        return None
    if kind == "dense":
        return jnp.asarray(_Q, dtype=jnp.float64)
    space = ArraySpace((2,), dtype=jnp.float64, space_id="perspective:variables")
    relation = EdgeRelation(
        jnp.asarray([0, 1, 0, 1], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        source_size=2,
        target_size=2,
    )
    return SparseCoordinateOperator(
        relation,
        jnp.asarray(_Q.reshape(-1), dtype=jnp.float64),
        source=space,
        target=space,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )


def _oracle(
    kind: str,
) -> tuple[
    NDArray[np.float64], float, NDArray[np.float64], float, NDArray[np.float64], float
]:
    q = np.zeros_like(_Q) if kind == "none" else _Q
    qx = q @ _X
    energy = float(_X @ qx)
    gradient_x = 2.0 * qx / _TAU
    gradient_tau = -energy / _TAU**2
    hvp_x = 2.0 * (q @ _DX / _TAU - qx * _DTAU / _TAU**2)
    hvp_tau = float(-2.0 * (_DX @ qx) / _TAU**2 + 2.0 * energy * _DTAU / _TAU**3)
    return qx, energy / _TAU, gradient_x, gradient_tau, hvp_x, hvp_tau


def _assert_close(actual: ArrayLike, expected: ArrayLike) -> None:
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("kind", ["dense", "sparse", "none"])
@pytest.mark.parametrize("scale", [1.0, 1e-155, 1e155])
def test_perspective_value_and_compiled_derivatives(kind: str, scale: float) -> None:
    quadratic = _quadratic(kind)
    x = jnp.asarray(_X * scale, dtype=jnp.float64)
    tau = jnp.asarray(_TAU * scale, dtype=jnp.float64)
    dx = jnp.asarray(_DX * scale, dtype=jnp.float64)
    dtau = jnp.asarray(_DTAU * scale, dtype=jnp.float64)
    qx, value, gx, gt, hx, ht = _oracle(kind)

    def perspective(x: Array, tau: Array) -> Array:
        return _conic_quadratic_perspective(quadratic, x, tau)[1]

    image, actual, declared_gx, declared_gt = jax.jit(
        lambda x, tau: _conic_quadratic_perspective(quadratic, x, tau)
    )(x, tau)
    _assert_close(image / scale, qx)
    _assert_close(actual / scale, value)
    _assert_close(declared_gx, gx)
    _assert_close(declared_gt, gt)

    _, tangent = jax.jit(
        lambda x, tau, dx, dtau: jax.jvp(perspective, (x, tau), (dx, dtau))
    )(x, tau, dx, dtau)
    _assert_close(tangent / scale, gx @ _DX + gt * _DTAU)

    # Compile the pullback itself: scaling the scalar perspective after its
    # reduction used to silently produce an all-zero x cotangent here.
    vjp_x, vjp_tau = jax.jit(jax.grad(perspective, argnums=(0, 1)))(x, tau)
    _assert_close(vjp_x, gx)
    _assert_close(vjp_tau, gt)
    _, (actual_hx, actual_ht) = jax.jit(
        lambda x, tau, dx, dtau: jax.jvp(
            jax.grad(perspective, argnums=(0, 1)), (x, tau), (dx, dtau)
        )
    )(x, tau, dx, dtau)
    _assert_close(actual_hx, hx)
    _assert_close(actual_ht, ht)


@pytest.mark.parametrize("kind", ["dense", "sparse", "none"])
@pytest.mark.parametrize("scale", [1.0, 1e-155, 1e155])
def test_embedding_gap_and_compiled_derivatives(kind: str, scale: float) -> None:
    quadratic = _quadratic(kind)
    # Constructor dtype canonicalization uses its native promotion policy;
    # the residual and every derivative below execute under strict_jax.
    with config.context(config.NumericsConfig(dtype_promotion="standard")):
        program = ConicProgram(
            quadratic,
            jnp.zeros((2,), dtype=jnp.float64),
            jnp.zeros((1, 2), dtype=jnp.float64),
            jnp.zeros((1,), dtype=jnp.float64),
            NonnegativeCone(1),
            convexity_evidence="construction",
        )
    barrier = cone_barrier_oracle(program.cone)
    one = jnp.ones((1,), dtype=jnp.float64)
    zero = jnp.zeros((1,), dtype=jnp.float64)

    def gap(x: Array, tau: Array) -> Array:
        vector = jnp.concatenate((x, one, one, tau[None], zero))
        return _embedding_residual(program, barrier, vector, one[0])[-2]

    x = jnp.asarray(_X * scale, dtype=jnp.float64)
    tau = jnp.asarray(_TAU * scale, dtype=jnp.float64)
    dx = jnp.asarray(_DX * scale, dtype=jnp.float64)
    dtau = jnp.asarray(_DTAU * scale, dtype=jnp.float64)
    _, value, gx, gt, hx, ht = _oracle(kind)
    actual, tangent = jax.jit(
        lambda x, tau, dx, dtau: jax.jvp(gap, (x, tau), (dx, dtau))
    )(x, tau, dx, dtau)
    _assert_close(actual / scale, value)
    _assert_close(tangent / scale, gx @ _DX + gt * _DTAU)
    grad = jax.grad(gap, argnums=(0, 1))
    actual_gx, actual_gt = jax.jit(grad)(x, tau)
    _assert_close(actual_gx, gx)
    _assert_close(actual_gt, gt)
    _, (actual_hx, actual_ht) = jax.jit(
        lambda x, tau, dx, dtau: jax.jvp(grad, (x, tau), (dx, dtau))
    )(x, tau, dx, dtau)
    _assert_close(actual_hx, hx)
    _assert_close(actual_ht, ht)


@pytest.mark.parametrize("kind", ["dense", "sparse", "none"])
def test_zero_x_at_positive_tau(kind: str) -> None:
    quadratic = _quadratic(kind)
    x = jnp.zeros((2,), dtype=jnp.float64)
    tau = jnp.asarray(2.0, dtype=jnp.float64)
    result = jax.jit(lambda x, tau: _conic_quadratic_perspective(quadratic, x, tau))(
        x, tau
    )
    for part in result:
        _assert_close(part, np.zeros_like(part))
    gx, gt = jax.jit(
        jax.grad(
            lambda x, tau: _conic_quadratic_perspective(quadratic, x, tau)[1],
            argnums=(0, 1),
        )
    )(x, tau)
    _assert_close(gx, np.zeros(2))
    _assert_close(gt, 0.0)


@pytest.mark.parametrize("kind", ["dense", "sparse", "none"])
def test_infinite_tau_does_not_produce_finite_embedding_residual(kind: str) -> None:
    with config.context(config.NumericsConfig(dtype_promotion="standard")):
        program = ConicProgram(
            _quadratic(kind),
            jnp.zeros((2,), dtype=jnp.float64),
            jnp.zeros((1, 2), dtype=jnp.float64),
            jnp.zeros((1,), dtype=jnp.float64),
            NonnegativeCone(1),
            convexity_evidence="construction",
        )
    barrier = cone_barrier_oracle(program.cone)
    vector = jnp.asarray([1.0, -0.5, 1.0, 1.0, np.inf, 1.0], dtype=jnp.float64)
    residual = jax.jit(
        lambda vector: _embedding_residual(
            program, barrier, vector, jnp.asarray(1.0, dtype=jnp.float64)
        )
    )(vector)
    assert not np.all(np.isfinite(np.asarray(residual)))


@pytest.mark.parametrize("kind", ["dense", "sparse", "none"])
@pytest.mark.parametrize("tau_value", [0.0, float("nan")])
def test_invalid_tau_is_not_repaired(kind: str, tau_value: float) -> None:
    quadratic = _quadratic(kind)
    result = jax.jit(lambda x, tau: _conic_quadratic_perspective(quadratic, x, tau))(
        jnp.asarray(_X, dtype=jnp.float64), jnp.asarray(tau_value, dtype=jnp.float64)
    )
    assert not np.isfinite(np.asarray(result[1]))


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1e-155, 1e155])
def test_dense_quadratic_coefficient_vjp(zero: bool, scale: float) -> None:
    quadratic = jnp.asarray(np.zeros_like(_Q) if zero else _Q, dtype=jnp.float64)
    x = jnp.asarray(_X * scale, dtype=jnp.float64)
    tau = jnp.asarray(_TAU * scale, dtype=jnp.float64)
    grad_q = jax.jit(
        jax.grad(lambda q, x, tau: _conic_quadratic_perspective(q, x, tau)[1], argnums=0)
    )(quadratic, x, tau)
    _assert_close(grad_q / scale, np.outer(_X, _X) / _TAU)


@pytest.mark.parametrize("kind", ["dense", "sparse", "none"])
@pytest.mark.parametrize("scale", [1e-155, 1e155])
def test_stationarity_image_retains_linear_derivatives(kind: str, scale: float) -> None:
    quadratic = _quadratic(kind)
    q = np.zeros_like(_Q) if kind == "none" else _Q
    x = jnp.asarray(_X * scale, dtype=jnp.float64)
    tau = jnp.asarray(_TAU * scale, dtype=jnp.float64)
    direction = jnp.asarray(_DX * scale, dtype=jnp.float64)

    def image(point: Array) -> Array:
        return _conic_quadratic_perspective(quadratic, point, tau)[0]

    _, tangent = jax.jit(lambda point, delta: jax.jvp(image, (point,), (delta,)))(
        x, direction
    )
    _assert_close(tangent / scale, q @ _DX)

    def pullback(point: Array) -> Array:
        _, backward = jax.vjp(image, point)
        return backward(jnp.ones_like(point))[0]

    _assert_close(jax.jit(pullback)(x), q.T @ np.ones(2, dtype=np.float64))
