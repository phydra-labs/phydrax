from __future__ import annotations

from dataclasses import replace
from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

from phydrax.discretization.spatial import (
    MortonAddressPlan,
    PreparedSparseVoxelGrid,
    SparseVoxelField,
    SparseVoxelGridPlan,
)
from phydrax.geometry import (
    DistanceSemantics,
    ExactSDFEnclosureCertificate,
    FieldRegularity,
    level_set_interpolation_certificate,
    SignReliability,
    Sphere,
    VoxelGeometrySamplingPlan,
    ZeroSetAccuracy,
)


def _grid(
    dimension: int = 3, *, missing_corner: int | None = None
) -> PreparedSparseVoxelGrid:
    lower = (-1.0, 2.0, -0.5)[:dimension]
    upper = (3.0, 8.0, 1.5)[:dimension]
    address = MortonAddressPlan(lower, upper, 2)
    coordinates = np.asarray(tuple(product(range(4), repeat=dimension)), dtype=np.int64)
    if missing_corner is not None:
        corner = np.asarray(tuple(product((0, 1), repeat=dimension))[missing_corner])
        coordinates = coordinates[np.any(coordinates != corner[None, :], axis=1)]
    return SparseVoxelGridPlan(
        address, brick_size=2, brick_capacity=2**dimension
    ).prepare(coordinates)


def _polynomial(points: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y = points[..., 0], points[..., 1]
    value = 1.2 + 2.1 * x - 0.7 * y + 0.9 * x * y
    if points.shape[-1] == 3:
        z = points[..., 2]
        value = value + 0.4 * z - 0.3 * x * z + 0.6 * y * z + 1.1 * x * y * z
    return value


def _polynomial_gradient(points: NDArray[np.float64]) -> NDArray[np.float64]:
    x, y = points[..., 0], points[..., 1]
    dx = 2.1 + 0.9 * y
    dy = -0.7 + 0.9 * x
    if points.shape[-1] == 2:
        return np.stack((dx, dy), axis=-1)
    z = points[..., 2]
    dx = dx - 0.3 * z + 1.1 * y * z
    dy = dy + 0.6 * z + 1.1 * x * z
    dz = 0.4 - 0.3 * x + 0.6 * y + 1.1 * x * y
    return np.stack((dx, dy, dz), axis=-1)


def _points(grid: PreparedSparseVoxelGrid, *, single: bool = False) -> Array:
    lower = jnp.asarray(grid.address_plan.lower, dtype=jnp.float64)
    spacing = (
        jnp.asarray(grid.address_plan.upper, dtype=jnp.float64) - lower
    ) / grid.address_plan.resolution
    indices = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.25, 2.6, 0.7), (2.75, 1.4, 2.1)),
        dtype=jnp.float64,
    )[:, : grid.dimension]
    return lower[None, :] + spacing[None, :] * (
        indices[:1] + 0.5 if single else indices + 0.5
    )


@pytest.mark.parametrize("dimension", (2, 3), ids=("bilinear", "trilinear"))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64), ids=("float32", "float64"))
def test_analytic_gradient_matches_independent_multilinear_polynomial(
    dimension: int, dtype: type[np.float32] | type[np.float64]
) -> None:
    """Anisotropic physical spacing and mixed terms detect index-space derivatives."""
    grid = _grid(dimension)
    centers = np.asarray(grid.voxel_centers(), dtype=np.float64)
    field = SparseVoxelField(grid, jnp.asarray(_polynomial(centers), dtype=dtype))
    points = _points(grid).astype(dtype)
    result = eqx.filter_jit(field.sample_multilinear_with_gradient)(points)
    host_points = np.asarray(points, dtype=np.float64)
    tolerance = 2.0e-5 if dtype is jnp.float32 else 1.0e-12
    assert bool(jnp.all(result.supported))
    assert bool(jnp.all(result.stencil_complete))
    np.testing.assert_allclose(
        result.values, _polynomial(host_points), rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        result.gradient, _polynomial_gradient(host_points), rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        result.values,
        field.sample_multilinear(points).values,
        rtol=tolerance,
        atol=tolerance,
    )


def test_vector_field_gradient_preserves_component_axes() -> None:
    grid = _grid()
    scalar = jnp.asarray(_polynomial(np.asarray(grid.voxel_centers(), dtype=np.float64)))
    field = SparseVoxelField(grid, jnp.stack((scalar, -2.0 * scalar), axis=-1))
    points = _points(grid)
    result = field.sample_multilinear_with_gradient(points)
    expected = _polynomial_gradient(np.asarray(points, dtype=np.float64))
    assert result.gradient.shape == (3, 2, 3)
    assert bool(jnp.all(result.supported))
    np.testing.assert_allclose(
        result.gradient,
        np.stack((expected, -2.0 * expected), axis=1),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("corner", range(8), ids=tuple(f"corner-{i}" for i in range(8)))
@pytest.mark.parametrize("background_mode", ("unsupported", "constant"))
def test_missing_corner_refuses_even_at_zero_interpolation_weight(
    corner: int, background_mode: str
) -> None:
    """No sparse background or zero corner weight can complete a derivative stencil."""
    grid = _grid(missing_corner=corner)
    field = SparseVoxelField(
        grid,
        jnp.ones((grid.brick_capacity, grid.voxels_per_brick), dtype=jnp.float64),
        background_mode=background_mode,
        background_value=9.0,
    )
    points = _points(grid, single=True)
    result = eqx.filter_jit(field.sample_multilinear_with_gradient)(points)
    assert not bool(result.stencil_complete[0])
    assert not bool(result.supported[0])
    assert bool(jnp.all(jnp.isnan(result.values)))
    assert bool(jnp.all(jnp.isnan(result.gradient)))
    if background_mode == "constant":
        ordinary = field.sample_multilinear(points)
        assert bool(ordinary.supported[0])
        assert not bool(ordinary.stencil_complete[0])
        assert bool(jnp.all(jnp.isfinite(ordinary.values)))


@pytest.mark.parametrize("bad_value", (np.nan, np.inf), ids=("nan", "infinite"))
def test_nonfinite_field_refuses_without_erasing_topology_evidence(
    bad_value: float,
) -> None:
    grid = _grid()
    field = SparseVoxelField(
        grid,
        jnp.full(
            (grid.brick_capacity, grid.voxels_per_brick), bad_value, dtype=jnp.float64
        ),
    )
    result = field.sample_multilinear_with_gradient(_points(grid, single=True))
    assert bool(result.stencil_complete[0])
    assert not bool(result.supported[0])
    assert bool(jnp.all(jnp.isnan(result.values)))
    assert bool(jnp.all(jnp.isnan(result.gradient)))


@pytest.mark.parametrize(
    "point",
    ((-1.1, 3.0, 0.0), (-0.9, 3.0, 0.0), (np.nan, 3.0, 0.0)),
    ids=("outside-domain", "missing-boundary-corner", "nonfinite-point"),
)
def test_invalid_point_has_no_background_gradient(
    point: tuple[float, float, float],
) -> None:
    grid = _grid()
    field = SparseVoxelField(
        grid,
        jnp.ones((grid.brick_capacity, grid.voxels_per_brick), dtype=jnp.float64),
        background_mode="constant",
        background_value=9.0,
    )
    result = field.sample_multilinear_with_gradient(
        jnp.asarray((point,), dtype=jnp.float64)
    )
    assert not bool(result.supported[0])
    assert not bool(result.stencil_complete[0])
    assert bool(jnp.all(jnp.isnan(result.gradient)))


def test_piecewise_gradient_jumps_at_cell_face() -> None:
    grid = _grid()
    centers = grid.voxel_centers()
    field = SparseVoxelField(grid, centers[..., 0] ** 2)
    points = jnp.asarray(
        ((0.5 - 1.0e-6, 3.5, 0.0), (0.5 + 1.0e-6, 3.5, 0.0)), dtype=jnp.float64
    )
    result = field.sample_multilinear_with_gradient(points)
    assert bool(jnp.all(result.supported))
    np.testing.assert_allclose(
        result.gradient, ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), atol=1.0e-12
    )


@pytest.mark.parametrize("sign", tuple(SignReliability), ids=lambda sign: sign.value)
def test_certificate_downgrades_without_strengthening_source_sign(
    sign: SignReliability,
) -> None:
    source = replace(
        Sphere((0.0, 0.0, 0.0), 0.5).compile().field_certificate,
        sign_reliability=sign,
        topology_identity="analytic-sphere-region",
    )
    certificate = level_set_interpolation_certificate(source)
    assert certificate.distance_semantics is DistanceSemantics.LEVEL_SET
    assert certificate.zero_set_accuracy is ZeroSetAccuracy.APPROXIMATE
    assert certificate.regularity is FieldRegularity.PIECEWISE_SMOOTH
    assert certificate.sign_reliability is (
        SignReliability.UNRELIABLE
        if sign is SignReliability.UNRELIABLE
        else SignReliability.LOCAL
    )
    assert certificate.safe_step_factor is None
    assert certificate.validity_region == "sparse_voxel_support"
    assert certificate.parameter_differentiable == source.parameter_differentiable
    assert certificate.lipschitz_upper_bound is None
    assert certificate.evaluation_error is None
    assert certificate.topology_identity is None
    with pytest.raises(ValueError, match="globally reliable exact signed-distance"):
        ExactSDFEnclosureCertificate(certificate)


@pytest.mark.parametrize(
    "enclosed", (False, True), ids=("uncertified", "source-enclosed")
)
def test_level_set_sampling_preserves_existing_samples_and_enclosure(
    enclosed: bool,
) -> None:
    geometry = Sphere((0.0, 4.0, 0.5), 0.9, feature_id="voxel-level-set").compile()
    grid = _grid()
    enclosure = (
        ExactSDFEnclosureCertificate(geometry.field_certificate) if enclosed else None
    )
    plan = VoxelGeometrySamplingPlan(
        grid,
        enclosure=enclosure,
        enclosure_geometry=geometry if enclosed else None,
        narrow_band_width=0.3,
    )
    ordinary = plan.sample(geometry)
    level_set = eqx.filter_jit(plan.sample_level_set)(geometry)
    assert ordinary.certificate.distance_semantics is DistanceSemantics.APPROXIMATE
    assert level_set.certificate.distance_semantics is DistanceSemantics.LEVEL_SET
    assert level_set.certificate.zero_set_accuracy is ZeroSetAccuracy.APPROXIMATE
    assert level_set.certificate.regularity is FieldRegularity.PIECEWISE_SMOOTH
    assert level_set.certificate.safe_step_factor is None
    assert level_set.certificate.sign_reliability is (
        SignReliability.LOCAL if enclosed else SignReliability.UNRELIABLE
    )
    assert level_set.sampling_id != ordinary.sampling_id
    assert bool(level_set.evidence.successful)
    np.testing.assert_array_equal(level_set.field.values, ordinary.field.values)
    np.testing.assert_array_equal(level_set.lower_bounds, ordinary.lower_bounds)
    np.testing.assert_array_equal(level_set.upper_bounds, ordinary.upper_bounds)
    np.testing.assert_array_equal(level_set.sign_certified, ordinary.sign_certified)
    np.testing.assert_array_equal(level_set.narrow_band, ordinary.narrow_band)
    np.testing.assert_array_equal(
        level_set.evidence.certified_sign_samples,
        ordinary.evidence.certified_sign_samples,
    )
    assert bool(jnp.any(level_set.sign_certified)) is enclosed


def test_level_set_sampling_retains_enclosure_owner_refusal() -> None:
    owner = Sphere((0.0, 4.0, 0.5), 0.9, feature_id="enclosure-owner").compile()
    unrelated = Sphere((0.0, 4.0, 0.5), 0.9, feature_id="unrelated-owner").compile()
    plan = VoxelGeometrySamplingPlan(
        _grid(),
        enclosure=ExactSDFEnclosureCertificate(owner.field_certificate),
        enclosure_geometry=owner,
    )
    with pytest.raises(RuntimeError, match="does not belong"):
        plan.sample_level_set(unrelated)
