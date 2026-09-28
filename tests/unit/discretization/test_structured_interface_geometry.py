#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization.finite_volume import (
    CurvatureEvidence,
    CurvatureStatus,
    HeightFunctionCurvaturePlan,
    LinearSurfaceTensionLaw,
    MACBalancedCapillaryOperator,
    plane_volume_fraction,
    StructuredPLICPlan,
    SurfaceTensionEvaluation,
    SurfaceTensionPolicy,
    VariableSurfaceTensionPolicy,
)


def _grid(
    count: int,
    dimension: int,
    *,
    periodic: bool = True,
    dtype: Any = jnp.float64,
) -> Any:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic)
            for _ in range(dimension)
        ),
        axis_names=("x", "y", "z")[:dimension],
    ).prepare(
        jnp.asarray(
            ((0.0,) * dimension, (1.0,) * dimension),
            dtype=dtype,
        )
    )
    return phx.discretization.FiniteVolumePlan(grid, component_names=("alpha",)).prepare()


def _ball_fraction(
    count: int, dimension: int, center: Any, radius: float, *, inside: bool = True
) -> Any:
    """Cell volume fractions of a ball: exact chords, composite Gauss elsewhere."""

    nodes, weights = np.polynomial.legendre.leggauss(8)
    sub = 16 if dimension == 2 else 4
    points = ((np.arange(sub)[:, None] + 0.5 * (nodes[None, :] + 1.0)) / sub).ravel()
    point_weights = np.tile(0.5 * weights, sub) / sub
    width = 1.0 / count
    edges = np.arange(count) * width
    tangential = edges[:, None] + points[None, :] * width
    center_ = np.asarray(center, dtype=np.float64)
    if dimension == 2:
        squared = radius**2 - (tangential - center_[1]) ** 2
        grids = (squared,)
    else:
        squared = (
            radius**2
            - (tangential[:, None, :, None] - center_[1]) ** 2
            - (tangential[None, :, None, :] - center_[2]) ** 2
        )
        grids = (squared,)
    half = np.sqrt(np.maximum(grids[0], 0.0))
    lower = center_[0] - half
    upper = center_[0] + half
    shape = (count,) + (1,) * half.ndim
    cell_lower = edges.reshape(shape)
    chord = np.clip(
        np.minimum(upper[None], cell_lower + width) - np.maximum(lower[None], cell_lower),
        0.0,
        None,
    ) * (grids[0][None] > 0.0)
    fraction = chord @ point_weights
    if dimension == 3:
        fraction = fraction @ point_weights
    fraction = fraction / width
    fraction = np.clip(fraction, 0.0, 1.0)
    return jnp.asarray(fraction if inside else 1.0 - fraction)


def _ellipse_fraction(
    count: int, center: tuple[float, float], radii: tuple[float, float], /
) -> Any:
    """Subcell volume fractions of the liquid outside a two-dimensional ellipse."""

    subcells = 12
    fine = (np.arange(count * subcells) + 0.5) / (count * subcells)
    x, y = np.meshgrid(fine, fine, indexing="ij")
    gas = ((x - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2 < 1.0
    fraction = gas.reshape(count, subcells, count, subcells).mean(axis=(1, 3))
    return jnp.asarray(1.0 - fraction, dtype=jnp.float64)


def _cusp_fraction(count: int, /) -> Any:
    """Subcell liquid fractions above a sharp V-shaped gas skirt."""

    subcells = 16
    fine = (np.arange(count * subcells) + 0.5) / (count * subcells)
    x, y = np.meshgrid(fine, fine, indexing="ij")
    gas = y < 0.12 + 2.0 * np.abs(x - 0.5)
    fraction = gas.reshape(count, subcells, count, subcells).mean(axis=(1, 3))
    return jnp.asarray(1.0 - fraction, dtype=jnp.float64)


def _square_below_line(normal: np.ndarray, offset: float) -> float:
    """Independent exact area of {n . x <= d} in [0, 1]^2 by polygon clipping."""

    corners = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    values = offset - corners @ normal
    kept = []
    for index in range(4):
        current, following = corners[index], corners[(index + 1) % 4]
        a, b = values[index], values[(index + 1) % 4]
        if a >= 0.0:
            kept.append(current)
        if (a >= 0.0) != (b >= 0.0):
            kept.append(current + a / (a - b) * (following - current))
    if len(kept) < 3:
        return 0.0
    polygon = np.asarray(kept)
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _line_segment_centroid(normal: np.ndarray, offset: float) -> np.ndarray:
    """Independent centroid of a line clipped by the unit square."""

    points = []
    for axis, value in ((0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0)):
        other = 1 - axis
        point = np.zeros((2,), dtype=np.float64)
        point[axis] = value
        point[other] = (offset - normal[axis] * value) / normal[other]
        if -1.0e-12 <= point[other] <= 1.0 + 1.0e-12:
            points.append(point)
    unique = np.unique(np.round(np.asarray(points), decimals=13), axis=0)
    if unique.shape != (2, 2):
        raise ValueError("The clipped line must have exactly two endpoints.")
    return np.mean(unique, axis=0)


def test_plane_volume_fraction_matches_exact_polygon_areas_and_complement() -> None:
    generator = np.random.default_rng(3)
    for _ in range(200):
        normal = generator.normal(size=2)
        if generator.random() < 0.25:
            normal[generator.integers(2)] *= 1.0e-9
        low, high = np.sum(np.minimum(normal, 0.0)), np.sum(np.maximum(normal, 0.0))
        offset = generator.uniform(low - 0.1, high + 0.1)
        value = plane_volume_fraction(jnp.asarray(normal), jnp.asarray(offset))
        np.testing.assert_allclose(value, _square_below_line(normal, offset), atol=1e-13)
    normal3 = jnp.asarray(generator.normal(size=(64, 3)))
    offset3 = jnp.asarray(generator.normal(size=64))
    np.testing.assert_allclose(
        plane_volume_fraction(normal3, offset3)
        + plane_volume_fraction(-normal3, -offset3),
        1.0,
        atol=1e-13,
    )


def test_plane_offset_inverts_volume_in_two_and_three_dimensions() -> None:
    plan = StructuredPLICPlan(_grid(8, 3))
    generator = np.random.default_rng(5)
    fraction = jnp.asarray(generator.uniform(0.0, 1.0, size=512))
    for dimension in (2, 3):
        normal = jnp.asarray(generator.normal(size=(512, dimension)))
        normal = normal.at[:64, 0].multiply(1.0e-8)
        offset, converged = plan.offset(normal, fraction)
        assert bool(jnp.all(converged))
        np.testing.assert_allclose(
            plane_volume_fraction(normal, offset), fraction, atol=1e-13
        )


def test_reconstruction_facets_converge_to_circle_perimeter() -> None:
    errors = []
    for count in (16, 48):
        plan = StructuredPLICPlan(_grid(count, 2))
        alpha = _ball_fraction(count, 2, (0.513, 0.507), 0.3)
        reconstruction = plan.reconstruct(alpha)
        assert bool(reconstruction.valid)
        errors.append(abs(float(jnp.sum(reconstruction.facet_measure)) - 0.6 * np.pi))
    assert errors[1] < errors[0] / 4.0
    assert errors[1] < 4.0e-3


def test_oblique_plic_reconstruction_returns_exact_facet_centroids() -> None:
    count = 12
    plic = StructuredPLICPlan(_grid(count, 2, periodic=False))
    angle = 0.37
    normal = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float64)
    centers = np.asarray(plic.discretization.cell_centers)
    width = 1.0 / count
    lower = centers - 0.5 * width
    scaled = np.broadcast_to(normal * width, (*centers.shape[:-1], 2))
    offset = 0.61 - np.sum(lower * normal, axis=-1)
    alpha = plane_volume_fraction(jnp.asarray(scaled), jnp.asarray(offset))
    normals = jnp.broadcast_to(jnp.asarray(normal), (*alpha.shape, 2))

    reconstruction = plic.reconstruct(alpha, normals)
    active = np.asarray(reconstruction.facet_valid)
    local_centroid = (np.asarray(reconstruction.interface_point) - lower) / width
    expected = np.zeros_like(local_centroid)
    for index in zip(*np.nonzero(active), strict=True):
        expected[index] = _line_segment_centroid(scaled[index], float(offset[index]))

    assert bool(reconstruction.valid)
    assert int(np.count_nonzero(active)) > 0
    np.testing.assert_allclose(local_centroid[active], expected[active], atol=2.0e-12)
    np.testing.assert_allclose(
        np.sum(scaled[active] * local_centroid[active], axis=-1),
        offset[active],
        atol=2.0e-13,
    )

    curvature = HeightFunctionCurvaturePlan(plic).evaluate(alpha, reconstruction)
    indices = jnp.indices(alpha.shape)
    interior = jnp.all((indices >= 3) & (indices < count - 3), axis=0)
    active_interior = curvature.evidence.interface_active & interior
    assert bool(jnp.all(curvature.evidence.usable_mask[active_interior]))
    np.testing.assert_allclose(
        curvature.evidence.curvature[active_interior],
        0.0,
        atol=2.0e-10,
    )


@pytest.mark.parametrize(
    ("dimension", "axis"),
    ((2, 0), (2, 1), (3, 0), (3, 1), (3, 2)),
    ids=("2d-x", "2d-y", "3d-x", "3d-y", "3d-z"),
)
@pytest.mark.parametrize(
    "orientation",
    (-1.0, 1.0),
    ids=("alpha-above", "alpha-below"),
)
def test_height_function_rejects_a_second_interface_band(
    dimension: int, axis: int, orientation: float
) -> None:
    count = 7
    plic = StructuredPLICPlan(_grid(count, dimension, periodic=False))
    # The one-cell liquid film creates two extra crossings hidden by valid endpoints.
    profile = jnp.asarray(
        (1.0, 1.0, 1.0, 0.5, 0.0, 1.0, 0.0), dtype=jnp.float64
    )
    if orientation < 0.0:
        profile = profile[::-1]
    profile_shape = tuple(
        count if other == axis else 1 for other in range(dimension)
    )
    alpha = jnp.broadcast_to(profile.reshape(profile_shape), (count,) * dimension)
    normal = jnp.zeros((*alpha.shape, dimension), dtype=alpha.dtype)
    normal = normal.at[..., axis].set(orientation)

    result = HeightFunctionCurvaturePlan(plic, fallback_radius=1).evaluate(
        alpha, plic.reconstruct(alpha, normal)
    )
    target = (count // 2,) * dimension
    status = int(result.evidence.status[target])

    assert bool(result.fallback_attempted[target])
    assert status in (
        int(CurvatureStatus.FALLBACK),
        int(CurvatureStatus.UNDERRESOLVED),
    )


def test_height_function_curvature_converges_for_circle_and_sphere() -> None:
    cases = (
        (2, (16, 48), 1.0e-2, 2.0e-2),
        (3, (12, 20), 2.0e-2, 3.5e-1),
    )
    for dimension, counts, primary_bound, complete_bound in cases:
        primary_errors = []
        complete_errors = []
        for count in counts:
            plan = StructuredPLICPlan(_grid(count, dimension))
            height = HeightFunctionCurvaturePlan(plan)
            alpha = _ball_fraction(
                count, dimension, (0.513, 0.507, 0.511)[:dimension], 0.3
            )
            result = height.evaluate(alpha, plan.reconstruct(alpha))
            evidence = result.evidence
            primary = evidence.status == int(CurvatureStatus.VALID)
            usable = evidence.usable_mask
            assert int(result.underresolved_count) == 0
            assert bool(jnp.all(usable == evidence.interface_active))
            exact = (dimension - 1) / 0.3
            primary_error = jnp.where(primary, evidence.curvature - exact, 0.0)
            complete_error = jnp.where(usable, evidence.curvature - exact, 0.0)
            primary_errors.append(
                float(jnp.sqrt(jnp.sum(primary_error**2) / jnp.sum(primary))) / exact
            )
            complete_errors.append(
                float(jnp.sqrt(jnp.sum(complete_error**2) / jnp.sum(usable))) / exact
            )
        assert primary_errors[-1] < primary_errors[0] / 2.0
        assert primary_errors[-1] < primary_bound
        assert complete_errors[-1] < complete_errors[0]
        assert complete_errors[-1] < complete_bound


def test_parabolic_fallback_resolves_skirt_without_chaining() -> None:
    plic = StructuredPLICPlan(_grid(20, 2))
    alpha = _ellipse_fraction(20, (0.5, 0.53), (0.3, 0.14))
    reconstruction = plic.reconstruct(alpha)
    plan = HeightFunctionCurvaturePlan(plic)
    result = plan.evaluate(alpha, reconstruction)
    fallback = result.evidence.status == int(CurvatureStatus.FALLBACK)

    assert int(result.fallback_count) > 0
    assert int(result.underresolved_count) == 0
    assert bool(jnp.all(result.fallback_support_count[fallback] >= 3))
    assert bool(jnp.all(result.fallback_rank[fallback] == 3))
    assert bool(
        jnp.all(result.fallback_condition[fallback] <= plan.fallback_condition_limit)
    )
    assert bool(jnp.all(result.position_valid[fallback]))
    assert bool(jnp.all(result.evidence.usable_mask == result.evidence.interface_active))

    without_primary_facets = eqx.tree_at(
        lambda selected: selected.facet_valid,
        reconstruction,
        jnp.zeros_like(reconstruction.facet_valid),
    )
    refused = plan.evaluate(alpha, without_primary_facets)
    attempted = result.fallback_attempted
    assert int(refused.fallback_count) == 0
    assert bool(
        jnp.all(refused.evidence.status[attempted] == int(CurvatureStatus.UNDERRESOLVED))
    )


def test_subcell_skirt_curvature_refuses_resolution_bound() -> None:
    plic = StructuredPLICPlan(_grid(20, 2))
    alpha = _ellipse_fraction(20, (0.5, 0.53), (0.3, 0.1))
    plan = HeightFunctionCurvaturePlan(plic)
    result = plan.evaluate(alpha, plic.reconstruct(alpha))
    underresolved = result.evidence.status == int(CurvatureStatus.UNDERRESOLVED)

    assert int(result.underresolved_count) > 0
    assert bool(jnp.all(result.fallback_support_count[underresolved] >= 3))
    assert bool(jnp.all(result.fallback_rank[underresolved] == 3))
    assert bool(
        jnp.any(
            jnp.abs(result.fallback_candidate_curvature[underresolved])
            > plan.curvature_bound / min(plan.spacing)
        )
    )


def test_bubble_curvature_is_negative_under_alpha_phase_convention() -> None:
    plan = StructuredPLICPlan(_grid(32, 2))
    height = HeightFunctionCurvaturePlan(plan)
    drop = _ball_fraction(32, 2, (0.5, 0.5), 0.25)
    bubble = 1.0 - drop
    drop_result = height.evaluate(drop, plan.reconstruct(drop)).evidence
    bubble_result = height.evaluate(bubble, plan.reconstruct(bubble)).evidence
    drop_mean = jnp.sum(jnp.where(drop_result.usable_mask, drop_result.curvature, 0.0))
    bubble_mean = jnp.sum(
        jnp.where(bubble_result.usable_mask, bubble_result.curvature, 0.0)
    )
    assert float(drop_mean) > 0.0
    np.testing.assert_allclose(bubble_mean, -drop_mean, rtol=1e-10)


def test_height_function_statuses_refuse_underresolved_and_mark_missing() -> None:
    plan = StructuredPLICPlan(_grid(16, 2))
    height = HeightFunctionCurvaturePlan(plan)
    speck = jnp.zeros((16, 16)).at[8, 8].set(0.3)
    result = height.evaluate(speck, plan.reconstruct(speck))
    status = result.evidence.status
    assert int(result.underresolved_count) > 0
    assert bool(
        jnp.all(status[result.evidence.interface_active] != CurvatureStatus.VALID)
    )
    empty = jnp.zeros((16, 16))
    missing = height.evaluate(empty, plan.reconstruct(empty)).evidence
    assert bool(jnp.all(missing.status == int(CurvatureStatus.MISSING_INTERFACE)))
    assert bool(jnp.all(missing.curvature == 0.0))


def test_cusp_fallback_refuses_unresolved_curvature() -> None:
    plic = StructuredPLICPlan(_grid(24, 2, periodic=False))
    alpha = _cusp_fraction(24)
    plan = HeightFunctionCurvaturePlan(plic)
    result = plan.evaluate(alpha, plic.reconstruct(alpha))
    underresolved = result.evidence.status == int(CurvatureStatus.UNDERRESOLVED)

    assert int(result.underresolved_count) > 0
    assert bool(jnp.all(result.fallback_support_count[underresolved] >= 3))
    assert bool(jnp.all(result.fallback_rank[underresolved] == 3))
    assert bool(
        jnp.all(
            jnp.abs(result.fallback_candidate_curvature[underresolved])
            > plan.curvature_bound / min(plan.spacing)
        )
    )


def test_parabolic_fallback_refuses_rank_deficient_primary_facets() -> None:
    plic = StructuredPLICPlan(_grid(16, 2, periodic=False))
    profile = jnp.concatenate(
        (
            jnp.ones((7,), dtype=jnp.float64),
            jnp.asarray((0.8, 0.5, 0.2), dtype=jnp.float64),
            jnp.zeros((6,), dtype=jnp.float64),
        )
    )
    diffuse_plane = jnp.broadcast_to(profile[:, None], (16, 16))
    reconstruction = plic.reconstruct(diffuse_plane)
    one_row = reconstruction.facet_valid & (jnp.arange(16, dtype=jnp.int32)[None, :] == 8)
    rank_deficient = eqx.tree_at(
        lambda selected: selected.facet_valid,
        reconstruction,
        one_row,
    )
    query = jnp.zeros((16, 16), dtype=jnp.float64).at[8, 8].set(0.3)

    result = HeightFunctionCurvaturePlan(plic).evaluate(query, rank_deficient)

    assert bool(result.fallback_attempted[8, 8])
    assert int(result.fallback_support_count[8, 8]) == 3
    assert int(result.fallback_rank[8, 8]) < 3
    assert int(result.evidence.status[8, 8]) == int(CurvatureStatus.UNDERRESOLVED)


@pytest.mark.parametrize(
    ("x64_enabled", "dtype"),
    ((False, jnp.float32), (True, jnp.float64)),
)
def test_parabolic_fallback_jit_preserves_dtype(x64_enabled: bool, dtype: Any) -> None:
    with jax.enable_x64(x64_enabled):
        plic = StructuredPLICPlan(_grid(8, 2, dtype=dtype))
        alpha = _ball_fraction(8, 2, (0.513, 0.507), 0.28).astype(dtype)
        reconstruction = plic.reconstruct(alpha)
        plan = HeightFunctionCurvaturePlan(plic)

        eager = plan.evaluate(alpha, reconstruction)
        compiled = eqx.filter_jit(plan.evaluate)(alpha, reconstruction)

        assert compiled.evidence.curvature.dtype == dtype
        assert compiled.evidence.residual.dtype == dtype
        assert compiled.evidence.interface_delta.dtype == dtype
        assert compiled.fallback_condition.dtype == dtype
        np.testing.assert_array_equal(compiled.evidence.status, eager.evidence.status)
        np.testing.assert_array_equal(
            compiled.evidence.interface_delta_supported,
            eager.evidence.interface_delta_supported,
        )
        np.testing.assert_allclose(
            compiled.evidence.interface_delta,
            eager.evidence.interface_delta,
            rtol=2.0e-6 if dtype == jnp.float32 else 1.0e-12,
            atol=2.0e-6 if dtype == jnp.float32 else 1.0e-12,
        )
        np.testing.assert_allclose(
            compiled.evidence.curvature,
            eager.evidence.curvature,
            rtol=2.0e-6 if dtype == jnp.float32 else 1.0e-12,
            atol=2.0e-6 if dtype == jnp.float32 else 1.0e-12,
        )


def test_height_function_refuses_grids_without_column_support() -> None:
    with pytest.raises(ValueError, match="stencil support"):
        HeightFunctionCurvaturePlan(StructuredPLICPlan(_grid(6, 2)))
    with pytest.raises(ValueError, match="fallback_radius"):
        HeightFunctionCurvaturePlan(
            StructuredPLICPlan(_grid(8, 2)),
            fallback_radius=4,
        )


def _planar_marangoni_case(
    angle: float, *, delta_supported: bool = True
) -> tuple[
    MACBalancedCapillaryOperator,
    Array,
    CurvatureEvidence,
    SurfaceTensionEvaluation,
    tuple[Array, ...],
]:
    discretization = _grid(24, 2, periodic=False)
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    plic = StructuredPLICPlan(discretization)
    theta = jnp.asarray(angle, dtype=jnp.float64)
    normal = jnp.asarray((-jnp.sin(theta), jnp.cos(theta)))
    tangent = jnp.asarray((jnp.cos(theta), jnp.sin(theta)))
    widths = plic.cell_widths()
    lower = discretization.cell_centers - 0.5 * widths
    point = jnp.asarray((0.503, 0.491), dtype=widths.dtype)
    alpha = plane_volume_fraction(
        normal * widths,
        jnp.sum(normal * (point - lower), axis=-1),
    )
    reconstruction = plic.reconstruct(
        alpha, jnp.broadcast_to(normal, alpha.shape + (2,))
    )
    delta, supported = plic.interface_delta(alpha, reconstruction)
    if not delta_supported:
        delta = jnp.zeros_like(delta)
        supported = jnp.zeros_like(supported)
    active = reconstruction.facet_valid
    status = jnp.where(
        active,
        int(CurvatureStatus.VALID),
        int(CurvatureStatus.MISSING_INTERFACE),
    ).astype(jnp.int8)
    curvature = CurvatureEvidence(
        jnp.zeros_like(alpha),
        jnp.zeros_like(alpha),
        status,
        interface_active=active,
        interface_delta=delta,
        interface_delta_supported=supported,
        geometry_id=discretization.prepared_id,
        reconstruction_id=plic.plan_id,
        evidence_id=f"planar-marangoni-{angle}-{delta_supported}",
    )
    policy = VariableSurfaceTensionPolicy(
        LinearSurfaceTensionLaw(1.0, 1.0, 0.0),
        density_floor=1.0e-6,
        capillary_cfl=0.4,
        law_id="planar-marangoni",
    )
    capillarity = MACBalancedCapillaryOperator(operators, policy)
    variable = SurfaceTensionEvaluation(
        jnp.ones_like(alpha),
        jnp.broadcast_to(tangent, alpha.shape + (2,)),
        jnp.ones_like(alpha, dtype=jnp.bool_),
    )
    return capillarity, alpha, curvature, variable, operators.face_dual_measures


@pytest.mark.parametrize("angle", (0.0, np.pi / 6.0, np.pi / 4.0))
def test_marangoni_force_uses_rotation_invariant_scalar_interface_delta(
    angle: float,
) -> None:
    capillarity, alpha, curvature, variable, dual_measures = _planar_marangoni_case(
        angle
    )

    result = capillarity.evaluate(
        alpha, curvature, variable_surface_tension=variable
    )

    integrated = jnp.stack(
        tuple(
            jnp.sum(component * measure)
            for component, measure in zip(
                result.marangoni_face_force, dual_measures, strict=True
            )
        )
    )
    tangent = jnp.asarray((np.cos(angle), np.sin(angle)), dtype=integrated.dtype)
    interface_measure = jnp.sum(
        curvature.interface_delta
        * capillarity.operators.discretization.cell_volumes
    )
    assert bool(result.valid)
    assert int(result.unsupported_face_count) == 0
    np.testing.assert_allclose(
        integrated, tangent * interface_measure, rtol=2.0e-13, atol=2.0e-13
    )
    np.testing.assert_allclose(
        jnp.linalg.norm(integrated) / interface_measure,
        1.0,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    normal = jnp.asarray((-np.sin(angle), np.cos(angle)), dtype=integrated.dtype)
    np.testing.assert_allclose(jnp.dot(integrated, normal), 0.0, atol=2.0e-13)
    if angle == 0.0:
        assert float(jnp.max(result.marangoni_face_force[0])) > 0.0
        assert jnp.array_equal(
            result.marangoni_face_force[1],
            jnp.zeros_like(result.marangoni_face_force[1]),
        )


def test_marangoni_force_refuses_missing_interface_delta_geometry() -> None:
    capillarity, alpha, curvature, variable, _ = _planar_marangoni_case(
        0.0, delta_supported=False
    )

    result = capillarity.evaluate(
        alpha, curvature, variable_surface_tension=variable
    )

    assert not bool(result.valid)
    assert int(result.unsupported_face_count) > 0
    assert jnp.array_equal(
        result.marangoni_face_force[0],
        jnp.zeros_like(result.marangoni_face_force[0]),
    )


def _drop_marangoni_force_error(count: int, /) -> tuple[float, float]:
    radius = 0.2
    discretization = _grid(count, 2)
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    plic = StructuredPLICPlan(discretization)
    alpha = _ball_fraction(count, 2, (0.503, 0.497), radius)
    reconstruction = plic.reconstruct(alpha)
    curvature = HeightFunctionCurvaturePlan(plic).evaluate(
        alpha, reconstruction
    ).evidence
    policy = VariableSurfaceTensionPolicy(
        LinearSurfaceTensionLaw(1.0, 1.0, 0.0),
        density_floor=1.0e-6,
        capillary_cfl=0.4,
        law_id="drop-marangoni-refinement",
    )
    scalar_gradient = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0), dtype=alpha.dtype),
        alpha.shape + (1, 2),
    )
    surface = policy.evaluate(
        discretization.cell_centers,
        jnp.zeros(alpha.shape + (1,), dtype=alpha.dtype),
        reconstruction.normal,
        scalar_gradient,
    )
    capillarity = MACBalancedCapillaryOperator(operators, policy)
    result = capillarity.evaluate(
        alpha, curvature, variable_surface_tension=surface
    )
    integrated = jnp.stack(
        tuple(
            jnp.sum(component * measure)
            for component, measure in zip(
                result.marangoni_face_force,
                operators.face_dual_measures,
                strict=True,
            )
        )
    )
    reference = np.pi * radius
    assert bool(result.valid)
    assert int(result.unsupported_face_count) == 0
    return (
        abs(float(integrated[0]) - reference) / reference,
        abs(float(integrated[1])) / reference,
    )


def test_bounded_drop_ygb_force_balance_improves_under_refinement() -> None:
    coarse_error, _ = _drop_marangoni_force_error(16)
    fine_error, fine_cross_force = _drop_marangoni_force_error(32)

    assert fine_error < coarse_error
    assert fine_error < 0.05
    assert fine_cross_force < 0.01


def test_balanced_force_with_uniform_potential_is_removed_exactly_by_projection() -> None:
    discretization = _grid(24, 2)
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    alpha = _ball_fraction(24, 2, (0.5, 0.5), 0.3)
    active = jnp.ones(alpha.shape, dtype=jnp.bool_)
    curvature = CurvatureEvidence(
        jnp.full(alpha.shape, 1.0 / 0.3),
        jnp.zeros(alpha.shape),
        jnp.zeros(alpha.shape, dtype=jnp.int8),
        interface_active=active,
        geometry_id="uniform",
        reconstruction_id="prescribed",
        evidence_id="uniform-curvature",
    )
    capillarity = MACBalancedCapillaryOperator(
        operators, SurfaceTensionPolicy(0.7, 1.0e-6, 0.4, "balanced-test")
    )
    force = capillarity.evaluate(alpha, curvature)
    assert bool(force.valid)
    projection = phx.solver.MACVariableDensityProjectionPlan(operators, tolerance=1e-12)
    step = 1.0e-2
    result = projection.project(
        tuple(step * value for value in force.face_force),
        tuple(jnp.ones_like(value) for value in force.face_force),
        step,
    )
    speed = max(float(jnp.max(jnp.abs(value))) for value in result.velocity)
    # Without balance the residual acceleration would be O(sigma kappa / h);
    # balance leaves only the pressure-solve tolerance.
    assert speed <= 1.0e-12 * step * (0.7 / 0.3) * 24.0
    inside = float(jnp.mean(result.pressure_increment[alpha == 1.0]))
    outside = float(jnp.mean(result.pressure_increment[alpha == 0.0]))
    np.testing.assert_allclose(inside - outside, 0.7 / 0.3, rtol=1e-10)


def test_capillary_force_jvp_matches_finite_difference_away_from_branches() -> None:
    discretization = _grid(16, 2)
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    plan = StructuredPLICPlan(discretization)
    height = HeightFunctionCurvaturePlan(plan)
    capillarity = MACBalancedCapillaryOperator(
        operators, SurfaceTensionPolicy(1.0, 1.0e-6, 0.4, "jvp-test")
    )
    alpha = _ball_fraction(16, 2, (0.51, 0.49), 0.3)
    mixed = plan.mixed_mask(alpha)
    direction = jnp.where(
        mixed & (alpha > 0.1) & (alpha < 0.9),
        jnp.sin(7.0 * discretization.cell_centers[..., 0]),
        0.0,
    )

    def force(value: Any) -> Any:
        reconstruction = plan.reconstruct(value, plan.interface_normal(alpha))
        curvature = height.evaluate(value, reconstruction).evidence
        return jnp.concatenate(
            tuple(
                face.ravel() for face in capillarity.evaluate(value, curvature).face_force
            )
        )

    _, tangent = jax.jvp(force, (alpha,), (direction,))
    epsilon = 1.0e-6
    difference = (
        force(alpha + epsilon * direction) - force(alpha - epsilon * direction)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(tangent, difference, rtol=1e-5, atol=1e-5)
