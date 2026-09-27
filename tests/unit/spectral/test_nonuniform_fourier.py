#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# References are dense NumPy sums of the defining exponentials, independent of
# the spreading kernel, FFT and deconvolution under test.

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._spectral._nonuniform_fourier import (
    NonuniformFourierPlan,
    NonuniformFourierResourceError,
    NonuniformFourierType,
    NonuniformFourierType3Plan,
    PreparedNonuniformFourier,
    PreparedNonuniformFourierType3,
)


pytestmark = pytest.mark.strict_jax


def _modes(size: int, centered: bool) -> np.ndarray:
    return (
        np.arange(size) - size // 2 if centered else np.fft.fftfreq(size) * size
    ).astype(np.float64)


def _phase_matrix(
    points: np.ndarray, mode_shape: tuple[int, ...], sign: int, centered: bool
) -> np.ndarray:
    grids = np.meshgrid(*(_modes(size, centered) for size in mode_shape), indexing="ij")
    wavevectors = np.stack([grid.reshape(-1) for grid in grids], axis=-1)
    return np.exp(1j * sign * points @ wavevectors.T)


def _relative_error(value: jax.Array, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(value) - reference) / np.linalg.norm(reference)
    )


def _gridded(
    mode_shape: tuple[int, ...],
    transform_type: NonuniformFourierType,
    *,
    sign: int,
    centered: bool,
    tolerance: float,
) -> PreparedNonuniformFourier:
    plan = NonuniformFourierPlan(
        mode_shape,
        transform_type,
        sign=sign,
        centered=centered,
        route="gridded",
        chunk_size=37,
        tolerance=tolerance,
    )
    return PreparedNonuniformFourier(plan, dtype=jnp.float64)


CASES = [
    pytest.param((31,), 1, False, 1e-6, id="1d-uncentered-plus"),
    pytest.param((24,), -1, True, 1e-11, id="1d-centered-minus"),
    pytest.param((12, 9), 1, True, 1e-3, id="2d-centered-plus"),
    pytest.param((10, 7), -1, False, 1e-9, id="2d-uncentered-minus"),
    pytest.param((6, 8, 5), 1, False, 1e-6, id="3d-uncentered-plus"),
    pytest.param((5, 4, 6), -1, True, 1e-12, id="3d-centered-minus"),
]


@pytest.mark.parametrize(("mode_shape", "sign", "centered", "tolerance"), CASES)
def test_gridded_type2_matches_direct_sum_within_tolerance(
    mode_shape: tuple[int, ...], sign: int, centered: bool, tolerance: float
) -> None:
    rng = np.random.default_rng(11)
    points = rng.uniform(-7.0, 7.0, (173, len(mode_shape)))
    coefficients = rng.normal(size=(*mode_shape, 2)) + 1j * rng.normal(
        size=(*mode_shape, 2)
    )
    reference = _phase_matrix(points, mode_shape, sign, centered) @ coefficients.reshape(
        (-1, 2)
    )

    values = _gridded(
        mode_shape, 2, sign=sign, centered=centered, tolerance=tolerance
    ).type2(points, coefficients)

    assert values.shape == (173, 2)
    assert values.dtype == jnp.complex128
    assert _relative_error(values, reference) <= tolerance


@pytest.mark.parametrize(("mode_shape", "sign", "centered", "tolerance"), CASES)
def test_gridded_type1_matches_direct_sum_within_tolerance(
    mode_shape: tuple[int, ...], sign: int, centered: bool, tolerance: float
) -> None:
    rng = np.random.default_rng(12)
    points = rng.uniform(-np.pi, np.pi, (211, len(mode_shape)))
    strengths = rng.normal(size=211) + 1j * rng.normal(size=211)
    reference = (_phase_matrix(points, mode_shape, sign, centered).T @ strengths).reshape(
        mode_shape
    )

    values = _gridded(
        mode_shape, 1, sign=sign, centered=centered, tolerance=tolerance
    ).type1(points, strengths)

    assert values.shape == mode_shape
    assert _relative_error(values, reference) <= tolerance


@pytest.mark.parametrize(
    "mode_shape", [(19,), (8, 11), (4, 6, 5)], ids=["1d", "2d", "3d"]
)
def test_gridded_type1_is_the_exact_adjoint_of_opposite_sign_type2(
    mode_shape: tuple[int, ...],
) -> None:
    rng = np.random.default_rng(13)
    points = rng.uniform(-np.pi, np.pi, (57, len(mode_shape)))
    coefficients = rng.normal(size=mode_shape) + 1j * rng.normal(size=mode_shape)
    samples = rng.normal(size=57) + 1j * rng.normal(size=57)
    forward = _gridded(mode_shape, 2, sign=1, centered=False, tolerance=1e-4)
    adjoint = _gridded(mode_shape, 1, sign=-1, centered=False, tolerance=1e-4)

    left = np.vdot(np.asarray(forward.type2(points, coefficients)), samples)
    right = np.vdot(coefficients, np.asarray(adjoint.type1(points, samples)))

    # Both routes share one real kernel, grid and deconvolution, so duality
    # holds to roundoff even though each route only meets 1e-4.
    assert abs(left - right) <= 1e-12 * abs(left)


def test_gridded_type2_jvp_and_vjp_match_analytic_derivatives() -> None:
    rng = np.random.default_rng(14)
    mode_shape = (9, 6)
    tolerance = 1e-10
    points = rng.uniform(-np.pi, np.pi, (41, 2))
    coefficients = rng.normal(size=mode_shape) + 1j * rng.normal(size=mode_shape)
    point_tangent = rng.normal(size=points.shape)
    coefficient_tangent = rng.normal(size=mode_shape) + 1j * rng.normal(size=mode_shape)
    cotangent = rng.normal(size=41) + 1j * rng.normal(size=41)
    transform = _gridded(mode_shape, 2, sign=1, centered=True, tolerance=tolerance)

    grids = np.meshgrid(*(_modes(size, True) for size in mode_shape), indexing="ij")
    wavevectors = np.stack([grid.reshape(-1) for grid in grids], axis=-1)
    phases = _phase_matrix(points, mode_shape, 1, True)
    flat = coefficients.reshape(-1)
    point_jacobian = 1j * (phases * flat) @ wavevectors
    reference_tangent = phases @ coefficient_tangent.reshape(-1) + np.sum(
        point_jacobian * point_tangent, axis=-1
    )

    _, tangent = jax.jvp(
        transform.type2,
        (jnp.asarray(points), jnp.asarray(coefficients)),
        (jnp.asarray(point_tangent), jnp.asarray(coefficient_tangent)),
    )
    _, pullback = jax.vjp(transform.type2, jnp.asarray(points), jnp.asarray(coefficients))
    point_cotangent, coefficient_cotangent = pullback(jnp.asarray(cotangent))

    # A coordinate derivative multiplies each mode by at most |k| <= 5.
    assert _relative_error(tangent, reference_tangent) <= 10.0 * tolerance
    assert (
        _relative_error(point_cotangent, np.real(cotangent[:, None] * point_jacobian))
        <= 10.0 * tolerance
    )
    assert (
        _relative_error(coefficient_cotangent, (phases.T @ cotangent).reshape(mode_shape))
        <= tolerance
    )


def test_gridded_type1_jvp_matches_analytic_derivative() -> None:
    rng = np.random.default_rng(15)
    mode_shape = (13,)
    tolerance = 1e-10
    points = rng.uniform(-np.pi, np.pi, (29, 1))
    strengths = rng.normal(size=29) + 1j * rng.normal(size=29)
    point_tangent = rng.normal(size=points.shape)
    transform = _gridded(mode_shape, 1, sign=-1, centered=False, tolerance=tolerance)
    modes = _modes(13, False)
    phases = _phase_matrix(points, mode_shape, -1, False)
    reference = (phases * (-1j * modes[None, :])).T @ (strengths * point_tangent[:, 0])

    _, tangent = jax.jvp(
        lambda coordinates: transform.type1(coordinates, jnp.asarray(strengths)),
        (jnp.asarray(points),),
        (jnp.asarray(point_tangent),),
    )

    assert _relative_error(tangent, reference) <= 10.0 * tolerance


TYPE3_CASES = [
    pytest.param((0.4,), (3.0,), (120.0,), (25.0,), 1, 1e-8, id="1d"),
    pytest.param((0.0, -1.0), (1.5, 2.0), (3.0, -40.0), (9.0, 6.0), -1, 1e-5, id="2d"),
    pytest.param(
        (0.2, 0.1, -0.3),
        (1.0, 2.0, 1.5),
        (-2.0, 5.0, 0.0),
        (6.0, 3.0, 4.0),
        1,
        1e-10,
        id="3d",
    ),
]


@pytest.mark.parametrize(
    (
        "source_center",
        "source_half_width",
        "target_center",
        "target_half_width",
        "sign",
        "tolerance",
    ),
    TYPE3_CASES,
)
def test_type3_matches_direct_sum_within_tolerance(
    source_center: tuple[float, ...],
    source_half_width: tuple[float, ...],
    target_center: tuple[float, ...],
    target_half_width: tuple[float, ...],
    sign: int,
    tolerance: float,
) -> None:
    rng = np.random.default_rng(16)
    dimension = len(source_center)
    sources = np.asarray(source_center) + np.asarray(source_half_width) * rng.uniform(
        -1.0, 1.0, (143, dimension)
    )
    targets = np.asarray(target_center) + np.asarray(target_half_width) * rng.uniform(
        -1.0, 1.0, (67, dimension)
    )
    strengths = rng.normal(size=(143, 2)) + 1j * rng.normal(size=(143, 2))
    reference = np.exp(1j * sign * targets @ sources.T) @ strengths
    plan = NonuniformFourierType3Plan(
        source_center,
        source_half_width,
        target_center,
        target_half_width,
        sign=sign,
        tolerance=tolerance,
        chunk_size=31,
    )

    result = PreparedNonuniformFourierType3(plan, dtype=jnp.float64).apply(
        sources, strengths, targets
    )

    assert result.values.shape == (67, 2)
    assert bool(jnp.all(result.supported))
    assert _relative_error(result.values, reference) <= tolerance


def test_type3_reports_targets_and_sources_outside_their_declared_boxes() -> None:
    plan = NonuniformFourierType3Plan((0.0,), (1.0,), (0.0,), (10.0,), tolerance=1e-6)
    prepared = PreparedNonuniformFourierType3(plan, dtype=jnp.float64)
    sources = jnp.asarray([[0.5], [-0.25]])
    strengths = jnp.asarray([1.0 + 0.0j, 2.0 + 0.0j])

    inside = prepared.apply(sources, strengths, jnp.asarray([[3.0], [12.0], [-10.0]]))
    outside = prepared.apply(
        jnp.asarray([[0.5], [1.5]]), strengths, jnp.asarray([[3.0], [-1.0]])
    )

    assert np.array_equal(np.asarray(inside.supported), [True, False, True])
    assert not bool(jnp.any(outside.supported))


def test_type3_jvp_and_vjp_match_analytic_derivatives() -> None:
    rng = np.random.default_rng(17)
    tolerance = 1e-10
    sources = rng.uniform(-2.0, 2.0, (23, 1))
    targets = 50.0 + rng.uniform(-8.0, 8.0, (19, 1))
    strengths = rng.normal(size=23) + 1j * rng.normal(size=23)
    source_tangent = rng.normal(size=sources.shape)
    target_tangent = rng.normal(size=targets.shape)
    cotangent = rng.normal(size=19) + 1j * rng.normal(size=19)
    plan = NonuniformFourierType3Plan(
        (0.0,), (2.0,), (50.0,), (8.0,), tolerance=tolerance
    )
    prepared = PreparedNonuniformFourierType3(plan, dtype=jnp.float64)
    phases = np.exp(1j * targets @ sources.T)
    # d f_k = sum_j c_j i (s_k dx_j + x_j ds_k) exp(i s_k x_j).
    reference = (1j * phases * targets) @ (strengths * source_tangent[:, 0]) + (
        1j * phases * target_tangent
    ) @ (strengths * sources[:, 0])

    def values(source_points: jax.Array, target_points: jax.Array) -> jax.Array:
        return prepared.apply(source_points, jnp.asarray(strengths), target_points).values

    _, tangent = jax.jvp(
        values,
        (jnp.asarray(sources), jnp.asarray(targets)),
        (jnp.asarray(source_tangent), jnp.asarray(target_tangent)),
    )
    _, pullback = jax.vjp(
        lambda weights: prepared.apply(sources, weights, targets).values,
        jnp.asarray(strengths),
    )
    (strength_cotangent,) = pullback(jnp.asarray(cotangent))

    # Derivatives scale by |s| <= 58 and |x| <= 2 relative to the transform.
    assert _relative_error(tangent, reference) <= 100.0 * tolerance
    assert _relative_error(strength_cotangent, phases.T @ cotangent) <= tolerance


def test_gridded_evidence_reports_tolerance_kernel_and_resources() -> None:
    plan = NonuniformFourierPlan(
        (33, 20), 2, route="gridded", tolerance=4e-6, chunk_size=64
    )
    prepared = PreparedNonuniformFourier(plan, dtype=jnp.float64)
    evidence = prepared.evidence

    # Axis budget 0.5 * 4e-6 / 2 = 1e-6: width ceil(log10(1e6)) + 1 = 7 and
    # beta 2.30 * 7 (Barnett, Magland, af Klinteberg 2019).
    assert evidence is not None
    assert evidence.requested_tolerance == 4e-6
    assert evidence.kernel_width == 7
    assert evidence.kernel_beta == pytest.approx(16.1)
    assert evidence.oversampling == 2.0
    assert evidence.fine_shape == (72, 40)
    assert evidence.grid_points == 72 * 40
    assert evidence.grid_bytes == 72 * 40 * 16
    assert evidence.working_entries == 64 * 7**2
    assert (
        PreparedNonuniformFourier(
            NonuniformFourierPlan((33, 20), 2), dtype=jnp.float64
        ).evidence
        is None
    )


def test_gridded_plans_refuse_grids_above_capacity() -> None:
    with pytest.raises(NonuniformFourierResourceError, match="maximum_grid_points"):
        NonuniformFourierPlan(
            (64, 64), 1, route="gridded", tolerance=1e-6, maximum_grid_points=128**2 - 1
        )
    with pytest.raises(NonuniformFourierResourceError, match="maximum_grid_points"):
        NonuniformFourierType3Plan(
            (0.0,), (100.0,), (0.0,), (1.0e4,), tolerance=1e-6, maximum_grid_points=10**5
        )


def test_gridded_route_refuses_tolerances_the_prepared_precision_cannot_meet() -> None:
    plan = NonuniformFourierPlan((8,), 2, route="gridded", tolerance=1e-9)

    with pytest.raises(ValueError, match="resolution of prepared dtype float32"):
        PreparedNonuniformFourier(plan, dtype=jnp.float32)
    with pytest.raises(TypeError, match="real floating dtype"):
        PreparedNonuniformFourier(plan, dtype=jnp.complex128)


def test_tolerance_is_required_exactly_by_the_gridded_route() -> None:
    with pytest.raises(ValueError, match="needs tolerance"):
        NonuniformFourierPlan((8,), 2, route="gridded")
    with pytest.raises(ValueError, match="do not accept a tolerance"):
        NonuniformFourierPlan((8,), 2, route="chunked", tolerance=1e-6)
    with pytest.raises(ValueError, match="route"):
        NonuniformFourierPlan((8,), 2, route="fast")  # ty: ignore[invalid-argument-type]
