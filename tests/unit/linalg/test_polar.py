"""Admission evidence and all-order derivatives of the 3D polar rotation.

Independent controls: orthogonality, determinant and stretch symmetry of the
value; central differences of the value against forward and reverse
Jacobians; the closed identity tangent ``dR = skew(dF)`` at ``F = I``; and
central differences of the reverse gradient against forward-over-reverse
curvature at isotropic and uniaxial frames, where SVD factor derivatives are
singular.  Inadmissible frames report their status with a NaN value.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.linalg import (
    polar_rotation3,
    PolarRotation3Result,
    PolarRotation3Status,
    SmallLinearSolvePlan,
)


pytestmark = pytest.mark.strict_jax
pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))

_STEP = 1.0e-6


def _rotation(seed: int, angle: float) -> np.ndarray:
    axis = np.random.default_rng(seed).normal(size=3)
    generator = np.cross(np.eye(3), axis / np.linalg.norm(axis))
    return (
        np.eye(3)
        + np.sin(angle) * generator
        + (1.0 - np.cos(angle)) * generator @ generator
    )


def _frames() -> tuple[np.ndarray, ...]:
    general = np.diag((1.3, 0.9, 1.1)) + 0.1 * np.random.default_rng(19).normal(
        size=(3, 3)
    )
    assert np.linalg.det(general) > 0.0
    return (
        np.eye(3),
        1.7 * _rotation(3, 1.1),
        _rotation(5, 0.4) @ np.diag((1.2, 1.2, 0.8)),
        _rotation(7, 2.0) @ general,
    )


def _value(frame: jax.Array) -> jax.Array:
    return polar_rotation3(frame).value


def test_admitted_value_is_the_proper_orthogonal_factor() -> None:
    for frame in _frames():
        result = polar_rotation3(jnp.asarray(frame))
        assert isinstance(result, PolarRotation3Result)
        assert bool(result.successful)
        assert int(result.status) == PolarRotation3Status.SUCCESS
        rotation = np.asarray(result.value)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-13)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-13)
        stretch = rotation.T @ frame
        np.testing.assert_allclose(stretch, stretch.T, atol=1e-13)
        assert np.min(np.linalg.eigvalsh(0.5 * (stretch + stretch.T))) > 0.0
        singular = np.linalg.svd(frame, compute_uv=False)
        np.testing.assert_allclose(np.asarray(result.singular_values), singular)
        assert float(result.determinant) == pytest.approx(np.linalg.det(frame))
        assert float(result.condition_estimate) == pytest.approx(
            singular[0] / singular[2]
        )
        # The derivative operator tr(U) I - U has eigenvalues s_i + s_j.
        derivative_condition = float(result.derivative_condition_estimate)
        assert (
            1.0 - 1e-12
            <= derivative_condition
            <= 3.0 * float((singular[0] + singular[1]) / (singular[1] + singular[2]))
        )


def test_first_derivatives_match_central_differences() -> None:
    for frame in _frames():
        point = jnp.asarray(frame)
        forward = np.asarray(jax.jacfwd(_value)(point))
        reverse = np.asarray(jax.jacrev(_value)(point))
        assert np.all(np.isfinite(forward)) and np.all(np.isfinite(reverse))
        np.testing.assert_allclose(reverse, forward, rtol=0.0, atol=1e-12)
        finite = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                offset = np.zeros((3, 3))
                offset[i, j] = _STEP
                plus = np.asarray(_value(jnp.asarray(frame + offset)))
                minus = np.asarray(_value(jnp.asarray(frame - offset)))
                finite[:, :, i, j] = (plus - minus) / (2.0 * _STEP)
        np.testing.assert_allclose(forward, finite, atol=1e-8)


def test_tangent_at_identity_is_the_skew_frame_change() -> None:
    direction = np.random.default_rng(11).normal(size=(3, 3))
    _, tangent = jax.jvp(_value, (jnp.eye(3),), (jnp.asarray(direction),))
    np.testing.assert_allclose(
        np.asarray(tangent), 0.5 * (direction - direction.T), atol=1e-14
    )


def test_curvature_matches_gradient_differences() -> None:
    rng = np.random.default_rng(13)
    weights = jnp.asarray(rng.normal(size=(3, 3)))

    def gradient(frame: jax.Array) -> jax.Array:
        return jax.grad(lambda value: jnp.sum(weights * _value(value)))(frame)

    step = 1.0e-5
    for frame in _frames():
        direction = rng.normal(size=(3, 3))
        _, curvature = jax.jvp(gradient, (jnp.asarray(frame),), (jnp.asarray(direction),))
        curvature = np.asarray(curvature)
        assert np.all(np.isfinite(curvature))
        finite = (
            np.asarray(gradient(jnp.asarray(frame + step * direction)))
            - np.asarray(gradient(jnp.asarray(frame - step * direction)))
        ) / (2.0 * step)
        np.testing.assert_allclose(curvature, finite, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("scale", [1e-110, 1e110], ids=["tiny", "huge"])
def test_proper_frame_admission_and_tangent_are_scale_invariant(scale: float) -> None:
    rotation = jnp.asarray(_rotation(3, 0.5), dtype=jnp.float64)
    perturbation = jnp.asarray(
        [[0.1, -0.2, 0.3], [0.4, 0.2, -0.1], [-0.3, 0.5, 0.2]],
        dtype=jnp.float64,
    )
    frame = scale * rotation
    result = polar_rotation3(frame)
    assert bool(result.successful)
    assert int(result.status) == PolarRotation3Status.SUCCESS
    np.testing.assert_allclose(result.value, rotation, atol=1e-12)
    _, tangent = jax.jvp(_value, (frame,), (scale * (rotation @ perturbation),))
    expected = rotation @ (0.5 * (perturbation - perturbation.T))
    np.testing.assert_allclose(tangent, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("frame", "status"),
    [
        (np.full((3, 3), np.nan), PolarRotation3Status.NONFINITE),
        (np.diag((1.0, 1.0, 0.0)), PolarRotation3Status.SINGULAR),
        (np.diag((1.0, 1.0, 1.0e-13)), PolarRotation3Status.ILL_CONDITIONED),
        (np.diag((1.0, 1.0, -1.0)), PolarRotation3Status.IMPROPER),
        (
            1e-110 * np.diag((1.0, 1.0, -1.0)),
            PolarRotation3Status.IMPROPER,
        ),
    ],
)
def test_inadmissible_frames_report_status_and_nan(
    frame: np.ndarray, status: PolarRotation3Status
) -> None:
    result = polar_rotation3(
        jnp.asarray(frame), plan=SmallLinearSolvePlan(3, singular_tolerance=1e-14)
    )
    assert not bool(result.successful)
    assert int(result.status) == status
    assert np.all(np.isnan(np.asarray(result.value)))


def test_rejects_wrong_frame_shape() -> None:
    with pytest.raises(ValueError):
        polar_rotation3(jnp.ones((2, 3), dtype=jnp.float64))


def test_rejects_nonfloating_frame() -> None:
    with pytest.raises(TypeError):
        polar_rotation3(jnp.eye(3, dtype=jnp.int32))


def test_rejects_incompatible_plan_dimension() -> None:
    with pytest.raises(ValueError):
        polar_rotation3(jnp.eye(3, dtype=jnp.float64), plan=SmallLinearSolvePlan(2))


def test_rejects_wrong_plan_kind() -> None:
    with pytest.raises(TypeError):
        polar_rotation3(jnp.eye(3, dtype=jnp.float64), plan=object())  # ty: ignore[invalid-argument-type]
