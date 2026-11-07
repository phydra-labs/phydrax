from fractions import Fraction
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._exact_algebraic_geometry import (
    coefficient_payload,
    RadicalElement,
)
from phydrax.geometry.brep import (
    brep_box,
    BRepEntityDimension,
    BRepProjectionStatus,
    BRepSource,
    OffsetCurve,
    OffsetSurface,
    prepare_brep_projection,
    RuledSurface,
)
from phydrax.geometry.brep._boolean_core import AffinePCurve, PeriodicPCurve
from phydrax.geometry.brep._conic_gaps import endpoint_gap_bounds
from phydrax.geometry.brep._correspondence import exact_source_lift, native_edge_pcurve
from phydrax.geometry.brep._intersection import TrimIntersectionRoot
from phydrax.geometry.brep._intersection_curve import SurfaceRegion
from phydrax.geometry.brep._patches import (
    _validate_curve_source_shadows,
    _validate_surface_source_shadows,
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    EllipseCurve,
    ExactLineCurve,
    ExtrusionSurface,
    LineCurve,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
    SurfaceMappedCurve,
    TorusPatch,
)
from phydrax.geometry.brep._root_bindings import BRepConicVertex, BRepRootSupport
from phydrax.units import MILLIMETER


@pytest.mark.parametrize("weights", ((1.0, 1.0), (2.0, 2.0), (1.0, 2.0)))
def test_boolean_chart_recognizes_exact_affine_spline_geometry(
    weights: tuple[float, float],
) -> None:
    from phydrax.geometry.brep._boolean_core import line_chart_coefficients

    curve = BSplineCurve(
        np.asarray(((1.0, 2.0), (5.0, -2.0))),
        np.asarray(weights),
        np.asarray((2.0, 2.0, 4.0, 4.0)),
        1,
    )
    coefficients = line_chart_coefficients(curve)
    if weights[0] != weights[1]:
        assert coefficients is None
    else:
        assert coefficients == (
            (Fraction(-3), Fraction(6)),
            (Fraction(2), Fraction(-2)),
        )
        patch = CylinderPatch(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
        )
        shifted = PeriodicPCurve(curve, patch, (1, 0))
        shifted_coefficients = line_chart_coefficients(shifted)
        assert shifted_coefficients is not None
        assert shifted_coefficients[1] == coefficients[1]
        assert shifted_coefficients[0][1] == coefficients[0][1]


def test_affine_pcurve_identity_preserves_exact_rational_coefficients() -> None:
    curve = AffinePCurve(
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        np.asarray(
            ((Fraction(1, 3), Fraction(0)), (Fraction(0), Fraction(1))),
            dtype=object,
        ),
        np.asarray((Fraction(1, 7), Fraction(-2, 5)), dtype=object),
    )

    identity = curve.identity_payload()

    assert identity["matrix"] == (
        (coefficient_payload(Fraction(1, 3)), coefficient_payload(Fraction(0))),
        (coefficient_payload(Fraction(0)), coefficient_payload(Fraction(1))),
    )
    assert identity["offset"] == (
        coefficient_payload(Fraction(1, 7)),
        coefficient_payload(Fraction(-2, 5)),
    )
    canonical_fingerprint(identity)


@pytest.mark.parametrize(
    "value",
    (Fraction(0), Fraction(1, 3), Fraction(-2, 5)),
    ids=("zero", "positive-rational", "negative-rational"),
)
def test_curve_identity_preserves_rational_field_embedding(value: Fraction) -> None:
    rational = LineCurve(
        (float(value), 0.0),
        (1.0, 0.0),
        origin_exact=(value, Fraction(0)),
        direction_exact=(Fraction(1), Fraction(0)),
    )
    embedded = LineCurve(
        (float(value), 0.0),
        (1.0, 0.0),
        origin_exact=(RadicalElement(value), RadicalElement(0)),
        direction_exact=(RadicalElement(1), RadicalElement(0)),
    )
    assert canonical_fingerprint(rational.identity_payload()) == canonical_fingerprint(
        embedded.identity_payload()
    )


def test_curve_identity_distinguishes_exact_radical_from_rounded_value() -> None:
    exact = RadicalElement.sqrt(Fraction(2))
    algebraic = LineCurve(
        (float(exact), 0.0),
        (1.0, 0.0),
        origin_exact=(exact, Fraction(0)),
    )
    rounded = LineCurve((float(exact), 0.0), (1.0, 0.0))
    np.testing.assert_array_equal(algebraic.origin, rounded.origin)
    assert canonical_fingerprint(algebraic.identity_payload()) != canonical_fingerprint(
        rounded.identity_payload()
    )


@final
class _CuspSurface(AbstractSurfacePatch):
    def evaluate(self, parameters: Array, /) -> Array:
        u, v = parameters[..., 0], parameters[..., 1]
        return jnp.stack((u, v, jnp.abs(u)), axis=-1)


@final
class _CuspCurve(AbstractCurve):
    @property
    def ambient_dimension(self) -> int:
        return 3

    def evaluate(self, parameters: Array, /) -> Array:
        return jnp.stack(
            (parameters, jnp.zeros_like(parameters), jnp.abs(parameters)), axis=-1
        )


def test_unknown_surface_does_not_inherit_analytic_continuity() -> None:
    box = np.asarray(((-0.5, -0.5), (0.5, 0.5)), dtype=np.float64)
    cusp = _CuspSurface()
    plane = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    assert not cusp.is_c1_on(box)
    assert plane.is_c1_on(box)


def test_native_sweep_carriers_evaluate_authored_source_parameters() -> None:
    extrusion_profile = LineCurve((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    revolution_profile = LineCurve((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    extrusion = ExtrusionSurface(extrusion_profile, (0.0, 2.0, 0.0))
    revolution = RevolutionSurface(
        revolution_profile,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(
        extrusion.evaluate(jnp.asarray((0.5, 2.0), dtype=jnp.float64)),
        (1.5, 4.0, 0.0),
    )
    np.testing.assert_allclose(
        revolution.evaluate(jnp.asarray((np.pi / 2.0, 2.0), dtype=jnp.float64)),
        (0.0, 1.0, 2.0),
        atol=1.0e-15,
    )
    assert extrusion.is_c1_on(np.asarray(((0.0, 0.0), (1.0, 1.0))))
    assert revolution.is_c1_on(np.asarray(((0.0, 0.0), (1.0, 1.0))))


def test_surface_isoparametric_curve_retains_period_and_evaluation() -> None:
    circle = CircleCurve(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        2.0,
    )
    extrusion = ExtrusionSurface(circle, (0.0, 0.0, 1.0))
    seam = SurfaceIsoparametricCurve(extrusion, 1, 0.25)

    assert seam.period == pytest.approx(2.0 * np.pi)
    np.testing.assert_allclose(
        seam.evaluate(jnp.asarray(0.0, dtype=jnp.float64)), (2.0, 0.0, 0.25)
    )
    np.testing.assert_allclose(
        seam.evaluate(jnp.asarray(np.pi / 2.0, dtype=jnp.float64)),
        (0.0, 2.0, 0.25),
        atol=1.0e-15,
    )
    assert seam.is_c1_on(0.0, np.pi)


def test_zero_angle_cylinder_generator_line_has_exact_source_lift() -> None:
    patch = CylinderPatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        5.0,
    )
    source = LineCurve((5.0, 0.0, 2.0), (0.0, 0.0, 1.0))
    pcurve = LineCurve((0.0, 2.0), (0.0, 1.0))
    region = SurfaceRegion(patch, np.asarray(((0.0, 0.0), (1.0, 3.0))))

    assert exact_source_lift(source, pcurve, region, 0.0, 1.0)
    assert not exact_source_lift(
        source,
        LineCurve((0.25, 2.0), (0.0, 1.0)),
        region,
        0.0,
        1.0,
    )


def test_exact_source_lift_covers_direct_ruled_rails_and_spline_boundary() -> None:
    first_rail = ExactLineCurve((0, 0, 0), (1, 0, 0))
    second_rail = ExactLineCurve((0, 1, 0), (1, 1, 0))
    ruled = RuledSurface(first_rail, second_rail)
    ruled_region = SurfaceRegion(ruled, np.asarray(((0.0, 0.0), (1.0, 1.0))))

    assert exact_source_lift(
        first_rail,
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        ruled_region,
        0.0,
        1.0,
    )

    exact_points = tuple(
        tuple(
            (
                Fraction(i, 2),
                Fraction(j, 2),
                Fraction(int(i == 1 and j == 1)),
            )
            for j in range(3)
        )
        for i in range(3)
    )
    exact_weights = tuple(
        tuple(Fraction(2 if i == 1 and j == 1 else 1) for j in range(3)) for i in range(3)
    )
    exact_knots = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1),
        Fraction(1),
        Fraction(1),
    )
    patch = BSplineSurfacePatch(
        np.asarray(exact_points, dtype=np.float64),
        np.asarray(exact_weights, dtype=np.float64),
        np.asarray(exact_knots, dtype=np.float64),
        np.asarray(exact_knots, dtype=np.float64),
        2,
        2,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
        exact_u_knots=exact_knots,
        exact_v_knots=exact_knots,
    )
    region = SurfaceRegion(patch, np.asarray(((0.0, 0.0), (1.0, 1.0))))
    reversed_top = ExactLineCurve((1, 1, 0), (0, 1, 0))
    reversed_chart = LineCurve((1.0, 1.0), (-1.0, 0.0))

    assert exact_source_lift(reversed_top, reversed_chart, region, 0.0, 1.0)
    assert not exact_source_lift(
        reversed_top,
        LineCurve((1.0, 0.0), (-1.0, 0.0)),
        region,
        0.0,
        1.0,
    )
    boundary_points = tuple(row[0] for row in exact_points)
    boundary_weights = tuple(row[0] for row in exact_weights)
    spline_edge = BSplineCurve(
        np.asarray(boundary_points, dtype=np.float64),
        np.asarray(boundary_weights, dtype=np.float64),
        np.asarray(exact_knots, dtype=np.float64),
        2,
        exact_control_points=boundary_points,
        exact_weights=boundary_weights,
        exact_knots=exact_knots,
    )
    spline_chart_points = (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 2), Fraction(0)),
        (Fraction(1), Fraction(0)),
    )
    spline_chart = BSplineCurve(
        np.asarray(spline_chart_points, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        np.asarray(exact_knots, dtype=np.float64),
        2,
        exact_control_points=spline_chart_points,
        exact_weights=tuple(Fraction(1) for _ in range(3)),
        exact_knots=exact_knots,
    )
    assert exact_source_lift(spline_edge, spline_chart, region, 0.0, 1.0)

    scaled_knots = (Fraction(0), Fraction(0), Fraction(1), Fraction(1))
    scaled_edge = BSplineCurve(
        np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))),
        np.ones(2, dtype=np.float64),
        np.asarray(scaled_knots, dtype=np.float64),
        1,
        exact_control_points=(
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(2), Fraction(0), Fraction(0)),
        ),
        exact_weights=(Fraction(1), Fraction(1)),
        exact_knots=scaled_knots,
    )
    scaled_chart_points = ((Fraction(0), Fraction(0)), (Fraction(2), Fraction(0)))
    scaled_chart = BSplineCurve(
        np.asarray(scaled_chart_points, dtype=np.float64),
        np.ones(2, dtype=np.float64),
        np.asarray(scaled_knots, dtype=np.float64),
        1,
        exact_control_points=scaled_chart_points,
        exact_weights=(Fraction(1), Fraction(1)),
        exact_knots=scaled_knots,
    )
    assert exact_source_lift(scaled_edge, scaled_chart, region, 0.0, 0.5)


def test_spline_boundary_gap_bounds_pointwise_parameterization() -> None:
    from phydrax.geometry.brep._conic_gaps import native_face_gap_squared

    exact_points = tuple(
        tuple(
            (
                Fraction(i, 2),
                Fraction(j, 2),
                Fraction(int(i == 1 and j == 1)),
            )
            for j in range(3)
        )
        for i in range(3)
    )
    exact_weights = tuple(tuple(Fraction(i + 1) for _ in range(3)) for i in range(3))
    exact_knots = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1),
        Fraction(1),
        Fraction(1),
    )

    def patch_from_controls(
        controls: tuple[tuple[tuple[Fraction, ...], ...], ...], /
    ) -> BSplineSurfacePatch:
        return BSplineSurfacePatch(
            np.asarray(controls, dtype=np.float64),
            np.asarray(exact_weights, dtype=np.float64),
            np.asarray(exact_knots, dtype=np.float64),
            np.asarray(exact_knots, dtype=np.float64),
            2,
            2,
            exact_control_points=controls,
            exact_weights=exact_weights,
            exact_u_knots=exact_knots,
            exact_v_knots=exact_knots,
        )

    patch = patch_from_controls(exact_points)
    region = SurfaceRegion(patch, np.asarray(((0.0, 0.0), (1.0, 1.0))))
    edge = ExactLineCurve((0, 0, 0), (1, 0, 0))
    pcurve = LineCurve((0.0, 0.0), (1.0, 0.0))

    assert not exact_source_lift(edge, pcurve, region, 0.0, 1.0)
    gap_squared = native_face_gap_squared(edge, pcurve, region, 0.0, 1.0)
    assert gap_squared is not None and gap_squared >= Fraction(1, 64)

    moved_points = tuple(
        tuple(
            (
                point[0],
                point[1] + (Fraction(1, 10) if i == 1 and j == 0 else 0),
                point[2],
            )
            for j, point in enumerate(row)
        )
        for i, row in enumerate(exact_points)
    )
    moved_region = SurfaceRegion(
        patch_from_controls(moved_points), np.asarray(((0.0, 0.0), (1.0, 1.0)))
    )
    moved_gap_squared = native_face_gap_squared(edge, pcurve, moved_region, 0.0, 1.0)
    assert moved_gap_squared is not None
    assert moved_gap_squared >= Fraction(29, 1600)


def test_torus_rotational_circle_requires_exact_profile_period_gauge() -> None:
    profile = CircleCurve(
        (2.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        0.5,
    )
    patch = RevolutionSurface(profile, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    ring = CircleCurve(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        2.5,
    )
    region = SurfaceRegion(patch, np.asarray(((0.0, 0.0), (2.0 * np.pi, 2.0 * np.pi))))
    source_chart = LineCurve((0.0, 0.0), (1.0, 0.0))
    periodic_chart = PeriodicPCurve(source_chart, patch, (0, 1))

    assert exact_source_lift(ring, periodic_chart, region, 0.0, 2.0 * np.pi)
    assert not exact_source_lift(
        ring,
        LineCurve((0.0, 2.0 * np.pi), (1.0, 0.0)),
        region,
        0.0,
        2.0 * np.pi,
    )
    from phydrax.geometry.brep._conic_gaps import native_face_gap_squared

    quarter_parameter = float(np.pi / 2.0)
    quarter_region = SurfaceRegion(
        patch,
        np.asarray(
            (
                (0.0, quarter_parameter),
                (2.0 * np.pi, quarter_parameter + 2.0 * np.pi),
            )
        ),
    )
    quarter_ring = CircleCurve(
        (0.0, 0.0, 0.5),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        2.0,
    )
    quarter_chart = LineCurve((0.0, quarter_parameter), (1.0, 0.0))
    quarter_gap = native_face_gap_squared(
        quarter_ring, quarter_chart, quarter_region, 0.0, 2.0 * np.pi
    )
    periodic_gap = native_face_gap_squared(
        quarter_ring,
        PeriodicPCurve(quarter_chart, patch, (0, 1)),
        quarter_region,
        0.0,
        2.0 * np.pi,
    )

    assert not exact_source_lift(
        quarter_ring, quarter_chart, quarter_region, 0.0, 2.0 * np.pi
    )
    periodic_quarter = PeriodicPCurve(quarter_chart, patch, (0, 1))
    assert not exact_source_lift(
        quarter_ring, periodic_quarter, quarter_region, 0.0, 2.0 * np.pi
    )
    assert quarter_gap is not None and quarter_gap > 0
    assert periodic_gap == quarter_gap
    quarter_root = ConicParameterRoot.from_exact_point(
        profile,
        (Fraction(2), Fraction(0), Fraction(1, 2)),
        quarter_parameter,
    )
    assert quarter_root is not None
    rooted_chart = AffinePCurve(
        ConicRootLinePCurve(quarter_root, 0, 1, parameter_range=(0.0, 2.0 * np.pi)),
        ((0, 1), (1, 0)),
        (0, 0),
    )
    rooted_periodic_chart = PeriodicPCurve(rooted_chart, patch, (0, 1))
    assert exact_source_lift(quarter_ring, rooted_chart, quarter_region, 0.0, 2.0 * np.pi)
    assert exact_source_lift(
        quarter_ring, rooted_periodic_chart, quarter_region, 0.0, 2.0 * np.pi
    )
    assert (
        native_face_gap_squared(
            quarter_ring, rooted_chart, quarter_region, 0.0, 2.0 * np.pi
        )
        == 0
    )
    assert (
        native_face_gap_squared(
            quarter_ring,
            rooted_periodic_chart,
            quarter_region,
            0.0,
            2.0 * np.pi,
        )
        == 0
    )


def test_revolution_one_vertex_conic_uses_rooted_turn_for_nonzero_start() -> None:
    from phydrax.geometry.brep._constructors import (
        brep_revolution,
        PlanarProfile,
        ProfileArc,
        ProfileLoop,
        ProfilePlane,
    )

    profile = PlanarProfile(
        ProfilePlane(),
        ProfileLoop(((2.0, 0.5),), (ProfileArc((2.0, 0.0)),)),
    )
    assert profile.endpoint_gap_bounds is None
    model = brep_revolution(
        profile,
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    geometry = model.geometry

    # Profile-turn uses close the one-vertex ring; the angular-turn wrapper is the
    # forward seam use on the closing native-period sheet.
    periodic_indices = tuple(
        index
        for index, pcurve in enumerate(geometry.pcurves)
        if isinstance(pcurve, PeriodicPCurve) and pcurve.period_shifts[1] != 0
    )
    assert periodic_indices
    for index in periodic_indices:
        pcurve = geometry.pcurves[index]
        assert isinstance(pcurve, PeriodicPCurve)
        assert isinstance(pcurve.source_curve, AffinePCurve)
        assert isinstance(pcurve.source_curve.curve, ConicRootLinePCurve)
        profile_root = pcurve.source_curve.curve.root
        assert any(
            isinstance(vertex_root.primary, BRepConicVertex)
            and vertex_root.primary.point_exact == profile_root.exact_point()
            for vertex_root in geometry.vertex_roots
            if vertex_root is not None
        )
    if geometry.edge_face_gap_bounds is not None:
        assert len(geometry.edge_face_gap_bounds) == len(geometry.coedge_edges)
        assert all(
            geometry.edge_face_gap_bounds[index] == 0.0 for index in periodic_indices
        )


def test_source_shadow_validation_replays_nested_roots_and_offset_witness() -> None:
    spatial_conic = CircleCurve(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        1.0,
    )
    root = ConicParameterRoot(spatial_conic, Fraction(1), Fraction(0))
    root_line = ConicRootLinePCurve(root, Fraction(0), Fraction(1))
    _validate_curve_source_shadows(root_line)
    original_lower = root.parameter_lower
    object.__setattr__(root, "parameter_lower", original_lower + 0.25)
    try:
        with pytest.raises(ValueError, match="ConicParameterRoot derived bounds"):
            _validate_curve_source_shadows(root_line)
    finally:
        object.__setattr__(root, "parameter_lower", original_lower)

    phase_conic = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)
    phase_root = ConicParameterRoot(phase_conic, Fraction(1), Fraction(0))
    section = PlaneCylinderSectionPCurve(
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
        parameter_range=(0.0, 1.0),
        angle_offset_root=phase_root,
    )
    _validate_curve_source_shadows(section)
    original_upper = phase_root.parameter_upper
    object.__setattr__(phase_root, "parameter_upper", original_upper + 0.25)
    try:
        with pytest.raises(ValueError, match="ConicParameterRoot derived bounds"):
            _validate_curve_source_shadows(section)
    finally:
        object.__setattr__(phase_root, "parameter_upper", original_upper)

    profile = CircleCurve(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        1.0,
    )
    revolution = RevolutionSurface(profile, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    offset = OffsetSurface(revolution, Fraction(1, 10))
    _validate_surface_source_shadows(offset)
    witness = offset._sphere_revolution_witness
    assert witness is not None
    object.__setattr__(
        offset,
        "_sphere_revolution_witness",
        (witness[0], witness[1] + Fraction(1), witness[2], witness[3]),
    )
    try:
        with pytest.raises(ValueError, match="spherical revolution witness"):
            _validate_surface_source_shadows(offset)
    finally:
        object.__setattr__(offset, "_sphere_revolution_witness", witness)


def test_exact_line_root_support_admits_only_proved_closed_surface_boundary() -> None:
    profile = ExactLineCurve((1, 0, 0), (1, 0, 1))
    patch = RevolutionSurface(profile, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    first = LineCurve(
        (0.0, 0.0),
        (1.0, 0.0),
        origin_exact=(Fraction(0), Fraction(0)),
        direction_exact=(Fraction(1), Fraction(0)),
    )
    second = LineCurve(
        (0.5, 0.0),
        (0.0, 1.0),
        origin_exact=(Fraction(1, 2), Fraction(0)),
        direction_exact=(Fraction(0), Fraction(1)),
    )
    root = TrimIntersectionRoot(
        first,
        second,
        parameter_lower=np.asarray((-1.0, -1.0)),
        parameter_upper=np.asarray((1.0, 1.0)),
    )

    support = BRepRootSupport(patch, root)
    enclosure = support.point_enclosure()

    assert enclosure.shape == (2, 3)
    assert np.all(np.isfinite(enclosure))
    assert not np.any(enclosure[0] > enclosure[1])

    outside_first = LineCurve(
        (0.0, -0.01),
        (1.0, 0.0),
        origin_exact=(Fraction(0), Fraction(-1, 100)),
        direction_exact=(Fraction(1), Fraction(0)),
    )
    outside_second = LineCurve(
        (0.5, -0.01),
        (0.0, 1.0),
        origin_exact=(Fraction(1, 2), Fraction(-1, 100)),
        direction_exact=(Fraction(0), Fraction(1)),
    )
    outside_root = TrimIntersectionRoot(
        outside_first,
        outside_second,
        parameter_lower=np.asarray((-1.0, -1.0)),
        parameter_upper=np.asarray((1.0, 1.0)),
    )
    with pytest.raises(ValueError):
        BRepRootSupport(patch, outside_root)


def test_conic_root_generator_lifts_to_exact_plane_line() -> None:
    conic = CircleCurve(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        5.0,
    )
    root = ConicParameterRoot(conic, Fraction(3, 5), Fraction(4, 5))
    cylinder = CylinderPatch(
        conic.center,
        conic.first_axis,
        conic.second_axis,
        (0.0, 0.0, 1.0),
        conic.radius,
    )
    source = SurfaceMappedCurve(
        cylinder,
        ConicRootLinePCurve(root, 0, 2, angle_realization=float(np.arctan2(4.0, 3.0))),
    )
    plane = PlanePatch((0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (0.0, 0.0, 1.0))
    region = SurfaceRegion(plane, np.asarray(((0.0, -1.0), (2.0, 3.0))))
    plane_line = LineCurve((1.0, 0.0), (0.0, 2.0))

    assert exact_source_lift(source, plane_line, region, 0.0, 1.0)
    assert not exact_source_lift(
        source, LineCurve((1.0, 0.25), (0.0, 2.0)), region, 0.0, 1.0
    )
    wrong_radius = CylinderPatch(
        conic.center,
        conic.first_axis,
        conic.second_axis,
        (0.0, 0.0, 1.0),
        21.0 / 4.0,
    )
    assert not exact_source_lift(
        SurfaceMappedCurve(wrong_radius, source.pcurve),
        plane_line,
        region,
        0.0,
        1.0,
    )


def test_rational_conic_parameter_root_lifts_a_parallel_generator() -> None:
    conic = CircleCurve(
        (-3.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        5.0,
    )
    root = ConicParameterRoot(conic, Fraction(3, 5), Fraction(4, 5))
    angle = float(np.arctan2(4.0, 3.0))
    pcurve = ConicRootLinePCurve(root, 0, 1, angle_realization=angle)
    source = LineCurve((0.0, 4.0, 0.0), (0.0, 0.0, 2.0))
    patch = ExtrusionSurface(conic, (0.0, 0.0, 2.0))
    region = SurfaceRegion(patch, np.asarray(((0.0, 0.0), (2.0, 1.0))))
    cylinder = CylinderPatch(
        conic.center,
        conic.first_axis,
        conic.second_axis,
        (0.0, 0.0, 2.0),
        conic.radius,
    )
    cylinder_region = SurfaceRegion(cylinder, np.asarray(((0.0, 0.0), (2.0, 1.0))))

    assert root.exact_point() == (Fraction(0), Fraction(4), Fraction(0))
    assert (
        pcurve.bounding_box(0.0, 1.0)[0, 0]
        <= angle
        <= pcurve.bounding_box(0.0, 1.0)[1, 0]
    )
    assert exact_source_lift(source, pcurve, region, 0.0, 1.0)
    assert exact_source_lift(source, pcurve, cylinder_region, 0.0, 1.0)
    assert not exact_source_lift(
        source, LineCurve((angle, 0.0), (0.0, 1.0)), region, 0.0, 1.0
    )


def test_conic_parameter_roots_isolate_every_quadrant() -> None:
    conic = CircleCurve(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        1.0,
    )
    diagonal = RadicalElement.sqrt(Fraction(1, 2))
    for cosine, sine in (
        (Fraction(3, 5), Fraction(4, 5)),
        (Fraction(-3, 5), Fraction(4, 5)),
        (Fraction(-3, 5), Fraction(-4, 5)),
        (Fraction(3, 5), Fraction(-4, 5)),
        (diagonal, diagonal),
        (-diagonal, diagonal),
        (-diagonal, -diagonal),
        (diagonal, -diagonal),
    ):
        root = ConicParameterRoot(conic, cosine, sine)
        angle = float(np.arctan2(float(sine), float(cosine)))
        if angle < 0.0:
            angle += 2.0 * np.pi
        lower, upper = root.parameter_enclosure()
        assert lower <= angle <= upper


def test_conic_vertex_retains_exact_radical_point_and_offset() -> None:
    diagonal = RadicalElement.sqrt(Fraction(1, 2))
    conic = CircleCurve(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        1.0,
    )
    root = ConicParameterRoot(conic, diagonal, diagonal)
    vertex = BRepConicVertex(
        root,
        offset_exact=(Fraction(0), Fraction(0), Fraction(5, 4)),
    )
    expected = (diagonal, diagonal, Fraction(5, 4))
    assert vertex.point_exact == expected
    lower, upper = vertex.point_enclosure_exact()
    for value, lower_bound, upper_bound in zip(expected, lower, upper, strict=True):
        assert lower_bound <= value <= upper_bound


def test_isoparametric_c1_query_detects_tensor_spline_knot_break() -> None:
    u_knots = np.asarray((0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0))
    v_knots = np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0))
    controls = np.zeros((5, 3, 3), dtype=np.float64)
    controls[:, :, 0] = np.linspace(0.0, 1.0, 5)[:, None]
    controls[:, :, 1] = np.linspace(0.0, 1.0, 3)[None, :]
    patch = BSplineSurfacePatch(
        controls,
        np.ones((5, 3), dtype=np.float64),
        u_knots,
        v_knots,
        2,
        2,
    )
    isoline = SurfaceIsoparametricCurve(
        patch,
        1,
        0.5,
        parameter_range=(0.0, 1.0),
    )

    assert isoline.is_c1_on(0.1, 0.4)
    assert not isoline.is_c1_on(0.1, 0.9)
    assert patch.is_c1_on(np.asarray(((0.1, 0.1), (0.4, 0.9))))
    assert not patch.is_c1_on(np.asarray(((0.1, 0.1), (0.9, 0.9))))


def test_sphere_isoparametric_seam_is_c1_and_periodic() -> None:
    sphere = SpherePatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        1.0,
    )
    seam = SurfaceIsoparametricCurve(sphere, 1, 0.0)

    assert seam.period == pytest.approx(2.0 * np.pi)
    assert seam.is_c1_on(0.0, 2.0 * np.pi)
    np.testing.assert_allclose(
        seam.evaluate(jnp.asarray((0.0, 2.0 * np.pi), dtype=jnp.float64)),
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        atol=1.0e-15,
    )


def test_source_native_projection_needs_no_tessellation_or_optional_engine() -> None:
    coordinate_contract = SpatialCoordinateContract(MILLIMETER)
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=coordinate_contract,
    )
    assert model.tessellation is None
    source = BRepSource(model)
    assert source.model.source_revision == model.source_revision

    projection = prepare_brep_projection(model)
    locations = projection.locate_solids(
        np.asarray(((0.5, 0.5, 0.5), (1.5, 0.5, 0.5)), dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(locations.status),
        (BRepProjectionStatus.UNIQUE, BRepProjectionStatus.FAILED),
    )

    closest = projection.classify(
        np.asarray(((1.25, 0.5, 0.5),), dtype=np.float64),
        tolerance=0.26,
    )
    assert int(np.asarray(closest.dimensions)[0]) == BRepEntityDimension.FACE
    face_index = int(np.asarray(closest.indices)[0])
    np.testing.assert_allclose(
        model.patches[face_index].evaluate(closest.parameters[0]),
        (1.0, 0.5, 0.5),
    )
    assert int(np.asarray(closest.status)[0]) == BRepProjectionStatus.UNIQUE
    lower = float(np.asarray(closest.distance_lower_bounds)[0])
    upper = float(np.asarray(closest.residuals)[0])
    assert lower <= 0.25 <= upper


def test_brep_source_bounds_retain_occurrence_and_edge_lineage() -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 4.0),
        coordinate_contract=SpatialCoordinateContract(MILLIMETER),
    )
    geometry = model.geometry
    vertex = 0
    exact_point = geometry.vertex_exact_points[vertex]
    assert exact_point is not None
    assert geometry.source_vertex_point_bounds(vertex) == (exact_point, exact_point)
    occurrence = next(geometry.assembly_graph.iter_occurrences())
    assert model.source_vertex_point_bounds(vertex, occurrence_path=occurrence.path) == (
        exact_point,
        exact_point,
    )

    edge = next(index for index, curve in enumerate(geometry.edge_curves) if curve >= 0)
    domain = model.source_curve_parameter_domain(edge, occurrence_path=occurrence.path)
    lower, upper = model.source_curve_point_bounds(
        edge, *domain, occurrence_path=occurrence.path
    )
    endpoints = np.asarray(geometry.vertex_points)[list(geometry.edge_vertices[edge])]
    assert np.all(lower <= endpoints)
    assert np.all(endpoints <= upper)

    with pytest.raises(ValueError, match="exactly one source assembly instance"):
        model.source_vertex_point_bounds(vertex, occurrence_path=("missing",))


def test_bspline_curve_binds_exact_source_bank_to_numeric_shadow() -> None:
    exact_control_points = (
        (Fraction(1, 3), Fraction(0)),
        (Fraction(4, 3), Fraction(0)),
    )
    exact_weights = (Fraction(1), Fraction(1))
    exact_knots = (Fraction(0), Fraction(0), Fraction(1), Fraction(1))
    curve = BSplineCurve(
        np.asarray(
            tuple(tuple(float(value) for value in row) for row in exact_control_points)
        ),
        np.asarray(tuple(float(value) for value in exact_weights)),
        np.asarray(tuple(float(value) for value in exact_knots)),
        1,
        exact_control_points=exact_control_points,
        exact_weights=exact_weights,
        exact_knots=exact_knots,
    )

    assert curve.exact_control_points == exact_control_points
    assert curve.exact_weights == exact_weights
    assert curve.exact_knots == exact_knots
    curve_identity = curve.identity_payload()
    curve_control_points = curve_identity["control_points"]
    assert isinstance(curve_control_points, tuple)
    curve_first_point = curve_control_points[0]
    assert isinstance(curve_first_point, tuple)
    assert curve_first_point[0] == ("rational", 1, 3)
    box = curve.bounding_box(0.0, 1.0)
    assert Fraction.from_float(float(box[0, 0])) <= Fraction(1, 3)
    assert Fraction.from_float(float(box[1, 0])) >= Fraction(4, 3)


def test_bspline_surface_binds_exact_control_grid_and_knot_banks() -> None:
    exact_control_points = (
        (
            (Fraction(1, 3), Fraction(0), Fraction(0)),
            (Fraction(1, 3), Fraction(1), Fraction(0)),
        ),
        (
            (Fraction(4, 3), Fraction(0), Fraction(0)),
            (Fraction(4, 3), Fraction(1), Fraction(0)),
        ),
    )
    exact_weights = ((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1)))
    exact_u_knots = (Fraction(0), Fraction(0), Fraction(1), Fraction(1))
    exact_v_knots = exact_u_knots
    patch = BSplineSurfacePatch(
        np.asarray(
            tuple(
                tuple(tuple(float(value) for value in point) for point in row)
                for row in exact_control_points
            )
        ),
        np.asarray(tuple(tuple(float(value) for value in row) for row in exact_weights)),
        np.asarray(tuple(float(value) for value in exact_u_knots)),
        np.asarray(tuple(float(value) for value in exact_v_knots)),
        1,
        1,
        exact_control_points=exact_control_points,
        exact_weights=exact_weights,
        exact_u_knots=exact_u_knots,
        exact_v_knots=exact_v_knots,
        control_point_ids=((17, 23), (41, 59)),
    )

    assert patch.exact_control_points == exact_control_points
    assert patch.exact_weights == exact_weights
    assert patch.exact_u_knots == exact_u_knots
    assert patch.exact_v_knots == exact_v_knots
    assert patch.control_point_ids == ((17, 23), (41, 59))
    surface_identity = patch.identity_payload()
    surface_control_grid = surface_identity["control_points"]
    assert isinstance(surface_control_grid, tuple)
    surface_first_strip = surface_control_grid[0]
    assert isinstance(surface_first_strip, tuple)
    surface_first_point = surface_first_strip[0]
    assert isinstance(surface_first_point, tuple)
    assert surface_first_point[0] == ("rational", 1, 3)


def test_unequal_radius_cylinder_rootlines_share_exact_axial_curve() -> None:
    first_patch = CylinderPatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        5.0,
    )
    second_patch = CylinderPatch(
        (4.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        3.0,
    )
    first_conic = CircleCurve(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        5.0,
    )
    second_conic = CircleCurve(
        (4.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        3.0,
    )
    first_root = ConicParameterRoot(first_conic, Fraction(4, 5), Fraction(3, 5))
    second_root = ConicParameterRoot(second_conic, Fraction(0), Fraction(1))
    first_pcurve = ConicRootLinePCurve(
        first_root, 0, 1, angle_realization=np.arctan2(3.0, 4.0)
    )
    second_pcurve = ConicRootLinePCurve(second_root, 0, 1, angle_realization=np.pi / 2.0)
    curve = SurfaceMappedCurve(first_patch, first_pcurve)
    region = SurfaceRegion(second_patch, np.asarray(((0.0, 0.0), (2.0 * np.pi, 1.0))))

    assert (
        first_root.exact_point()
        == second_root.exact_point()
        == (Fraction(4), Fraction(3), Fraction(0))
    )
    assert exact_source_lift(curve, second_pcurve, region, 0.0, 1.0)


def test_offset_curve_preserves_exact_affine_surface_correspondence() -> None:
    curve = OffsetCurve(
        LineCurve((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        Fraction(1, 3),
        (0.0, 0.0, 1.0),
    )
    plane = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    region = SurfaceRegion(
        plane, np.asarray(((-2.0, -2.0), (3.0, 2.0)), dtype=np.float64)
    )
    pcurve = native_edge_pcurve(curve, region.patch)
    assert pcurve is not None
    assert exact_source_lift(curve, pcurve, region, 0.0, 1.0)
    assert curve.is_c1_on(0.0, 1.0)
    parameters = jnp.asarray((0.0, 0.5, 1.0), dtype=jnp.float64)
    expected = np.asarray(
        ((0.0, -1.0 / 3.0, 0.0), (1.0, -1.0 / 3.0, 0.0), (2.0, -1.0 / 3.0, 0.0)),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(curve.evaluate(parameters), expected)
    np.testing.assert_array_equal(plane.evaluate(pcurve.evaluate(parameters)), expected)


def test_offset_distance_is_differentiable_but_source_edits_require_reauthoring() -> None:
    curve = OffsetCurve(LineCurve((0.0, 0.0), (1.0, 0.0)), Fraction(1, 3))
    parameter = jnp.asarray(0.5, dtype=jnp.float64)

    def distance_leaf(value: OffsetCurve) -> Array:
        return value.distance

    def height(distance: Array) -> Array:
        return eqx.tree_at(distance_leaf, curve, distance).evaluate(parameter)[1]

    np.testing.assert_allclose(jax.grad(height)(curve.distance), -1.0, rtol=0.0, atol=0.0)

    edited = eqx.tree_at(distance_leaf, curve, jnp.asarray(0.5, dtype=jnp.float64))
    with pytest.raises(ValueError):
        edited.identity_payload()
    with pytest.raises(ValueError):
        edited.bounding_box(0.0, 1.0)
    np.testing.assert_array_equal(curve.evaluate(parameter), (0.5, -1.0 / 3.0))


def test_space_curve_parallel_direction_refuses_offset_normal_certificate() -> None:
    curve = OffsetCurve(
        LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        0.25,
        (1.0, 0.0, 0.0),
    )
    assert not curve.is_c1_on(0.0, 1.0)
    with pytest.raises(NotImplementedError):
        curve.bounding_box(0.0, 1.0)


def test_ruled_surface_retains_rail_parameterization_and_directional_derivative() -> None:
    surface = RuledSurface(
        LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        LineCurve((0.0, 1.0, 2.0), (2.0, 0.0, 0.0)),
    )
    uv = jnp.asarray((0.2, 0.3), dtype=jnp.float64)
    direction = jnp.asarray((0.5, -0.25), dtype=jnp.float64)
    point, tangent = jax.jvp(surface.evaluate, (uv,), (direction,))
    np.testing.assert_allclose(point, (0.26, 0.3, 0.6), rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(tangent, (0.6, -0.25, -0.5), rtol=0.0, atol=1.0e-15)
    assert surface.is_c1_on(((0.0, 0.0), (1.0, 1.0)))
    assert surface.is_c1_along(0, 0.2, 0.0, 1.0)
    with pytest.raises(ValueError):
        surface.validate_parameter_box(((0.0, -0.1), (1.0, 1.0)))


def test_ruled_surface_does_not_infer_continuity_of_custom_rails() -> None:
    cusp = _CuspCurve()
    surface = RuledSurface(cusp, LineCurve((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)))
    assert not cusp.is_c1_on(-0.5, 0.5)
    assert not surface.is_c1_on(((-0.5, 0.0), (0.5, 1.0)))
    assert not surface.is_c1_along(1, 0.5, -0.5, 0.5)


@pytest.mark.parametrize(
    ("family", "radius", "support_families"),
    (
        ("cylinder", 3.0, {"cylinder", "plane"}),
        ("cone", 2.0, {"cone", "plane"}),
        ("sphere", 4.0, {"sphere", "plane"}),
    ),
)
def test_authored_revolution_seam_coedges_have_exact_source_lifts(
    family: str, radius: float, support_families: set[str]
) -> None:
    from phydrax.geometry.brep._constructors import (
        brep_cone,
        brep_cylinder,
        brep_revolution,
        PlanarProfile,
        ProfileArc,
        ProfileLine,
        ProfileLoop,
        ProfilePlane,
    )

    contract = SpatialCoordinateContract.si()
    if family == "cylinder":
        model = brep_cylinder(3.0, 3.0, coordinate_contract=contract)
    elif family == "cone":
        model = brep_cone(3.0, 2.0, 3.0, coordinate_contract=contract)
    else:
        profile = PlanarProfile(
            ProfilePlane(
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            ProfileLoop(
                ((0.0, -5.0), (4.0, 3.0), (0.0, 3.0)),
                (ProfileArc((0.0, 0.0)), ProfileLine(), ProfileLine()),
            ),
        )
        model = brep_revolution(
            profile, (0.0, 0.0), (0.0, 1.0), coordinate_contract=contract
        )

    geometry = model.geometry
    points = np.asarray(geometry.vertex_points)

    seam_candidates = []
    for edge, (start, end) in enumerate(geometry.edge_vertices):
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0:
            continue
        source_curve = geometry.curves[curve_index]
        if (
            start == end
            and points[start, 2] == 3.0
            and isinstance(source_curve, CircleCurve)
            and float(source_curve.radius_exact) == radius
            and sum(
                geometry.coedge_edges[coedge] == edge
                for loops in geometry.face_loops
                for loop in loops
                for coedge in loop
            )
            == 2
        ):
            seam_candidates.append(edge)
    assert len(seam_candidates) == 1
    seam_edge = seam_candidates[0]
    from phydrax.geometry.brep._intersection import NativePeriodEndpoint
    from phydrax.geometry.brep._root_bindings import BRepConicVertex

    seam_vertex = geometry.edge_vertices[seam_edge][0]
    source_curve = geometry.curves[geometry.edge_curves[seam_edge]]
    vertex_root = geometry.vertex_roots[seam_vertex]
    assert isinstance(source_curve, CircleCurve)
    assert vertex_root is not None and isinstance(vertex_root.primary, BRepConicVertex)
    assert vertex_root.primary.root.conic is source_curve
    assert geometry.vertex_exact_points[seam_vertex] == vertex_root.primary.point_exact
    assert all(
        isinstance(root, NativePeriodEndpoint)
        for root in geometry.edge_endpoint_roots[seam_edge]
    )
    incident = tuple(
        (face, coedge)
        for face, loops in enumerate(geometry.face_loops)
        for loop in loops
        for coedge in loop
        if geometry.coedge_edges[coedge] == seam_edge
    )
    assert len(incident) == 2

    lifts: list[bool] = []
    observed_families: set[str] = set()
    for face, coedge in incident:
        patch = model.patches[face]
        assert isinstance(patch, RevolutionSurface)
        bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
        witness = patch.exact_quadric_witness((float(bounds[0, 1]), float(bounds[1, 1])))
        assert witness is not None
        observed_families.add(witness.family)
        region = SurfaceRegion(patch, bounds)
        curve = geometry.curves[geometry.edge_curves[seam_edge]]
        pcurve = geometry.pcurves[coedge]
        first, last = map(float, np.asarray(geometry.edge_ranges[seam_edge]))
        lifts.append(exact_source_lift(curve, pcurve, region, first, last))
    assert observed_families == support_families
    assert lifts == [True, True]

    # A sphere ring uses its rooted profile turn; the negative controls below
    # perturb the plain angular line chart of the other incident use.
    face, coedge = next(
        (face, coedge)
        for face, coedge in incident
        if isinstance(geometry.pcurves[coedge], LineCurve)
    )
    patch = model.patches[face]
    assert isinstance(patch, RevolutionSurface)
    region = SurfaceRegion(
        patch, np.asarray(model.parameter_bounds[face], dtype=np.float64)
    )
    curve = geometry.curves[geometry.edge_curves[seam_edge]]
    assert isinstance(curve, CircleCurve)
    pcurve = geometry.pcurves[coedge]
    assert isinstance(pcurve, LineCurve)
    first, last = map(float, np.asarray(geometry.edge_ranges[seam_edge]))

    from phydrax.geometry.brep._placed import PlacedCurve, PlacedSurface

    pose = np.eye(3, dtype=np.float64)
    placed_curve = PlacedCurve(curve, pose, np.zeros((3,), dtype=np.float64))
    placed_patch = PlacedSurface(patch, pose, np.zeros((3,), dtype=np.float64))
    placed_region = SurfaceRegion(placed_patch, np.asarray(region.parameter_box))
    assert not exact_source_lift(placed_curve, pcurve, placed_region, first, last)
    wrong_radius = curve.radius_exact + Fraction(1, 4)
    wrong_curve = CircleCurve(
        tuple(float(value) for value in curve.center_exact),
        tuple(float(value) for value in curve.first_axis_exact),
        tuple(float(value) for value in curve.second_axis_exact),
        float(wrong_radius),
        center_exact=curve.center_exact,
        first_axis_exact=curve.first_axis_exact,
        second_axis_exact=curve.second_axis_exact,
        radius_exact=wrong_radius,
    )
    assert not exact_source_lift(wrong_curve, pcurve, region, first, last)

    period = float(np.asarray(region.parameter_box)[1, 0])
    reversed_pcurve = LineCurve(
        (period, float(pcurve.origin_exact[1])),
        (-1.0, 0.0),
        origin_exact=(Fraction.from_float(period), pcurve.origin_exact[1]),
        direction_exact=(Fraction(-1), Fraction(0)),
    )
    assert not exact_source_lift(curve, reversed_pcurve, region, first, last)

    shifted_box = np.asarray(region.parameter_box, dtype=np.float64).copy()
    shifted_period = float(shifted_box[1, 0] - shifted_box[0, 0])
    shifted_box[:, 0] += shifted_period
    shifted_region = SurfaceRegion(patch, shifted_box)
    shifted_pcurve = LineCurve(
        (shifted_period, float(pcurve.origin_exact[1])),
        (1.0, 0.0),
        origin_exact=(
            Fraction.from_float(shifted_period),
            pcurve.origin_exact[1],
        ),
        direction_exact=(Fraction(1), Fraction(0)),
    )
    assert not exact_source_lift(curve, shifted_pcurve, shifted_region, first, last)

    profile_lower, profile_upper = np.asarray(region.parameter_box)[:, 1]
    alternate_profile = float(
        profile_upper
        if pcurve.origin_exact[1] == Fraction.from_float(float(profile_lower))
        else profile_lower
    )
    changed_composition = LineCurve(
        (0.0, alternate_profile),
        (1.0, 0.0),
        origin_exact=(Fraction(0), Fraction.from_float(alternate_profile)),
        direction_exact=(Fraction(1), Fraction(0)),
    )
    assert not exact_source_lift(curve, changed_composition, region, first, last)
    assert not exact_source_lift(curve, pcurve, region, last, first)
    assert not exact_source_lift(curve, pcurve, region, first, last + 0.25)


def test_extruded_mixed_profile_cap_classifies_and_gaps_remain_visible() -> None:
    from math import pi

    from phydrax.geometry.brep._constructors import (
        brep_extrusion,
        PlanarProfile,
        ProfileCurveUse,
        ProfileLine,
        ProfileLoop,
        ProfilePlane,
    )
    from phydrax.geometry.brep._patches import CircleCurve

    circle = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)

    def profile(end_y: float) -> PlanarProfile:
        return PlanarProfile(
            ProfilePlane(),
            ProfileLoop(
                ((1.0, 0.0), (0.0, end_y), (0.0, 0.0)),
                (ProfileCurveUse(circle, 0.0, pi / 2), ProfileLine(), ProfileLine()),
            ),
        )

    exact = profile(1.0)
    assert exact.endpoint_gap_bounds is None
    model = brep_extrusion(
        exact, (0.0, 0.0, 2.0), coordinate_contract=SpatialCoordinateContract.si()
    )
    trim = model.trim_domains[model.physical_tags.index("cap:bottom")]
    assert trim is not None
    decision = trim.classify(np.asarray(((0.25, 0.25), (0.9, 0.9))))
    np.testing.assert_array_equal(decision.resolved, (True, True))
    np.testing.assert_array_equal(decision.inside, (True, False))

    gapped = profile(1.000001)
    assert gapped.endpoint_gap_bounds is not None
    assert any(
        bound > 0 for row in gapped.endpoint_gap_bounds for edge in row for bound in edge
    )
    with pytest.raises(ValueError, match="No exact source face-gap proof"):
        brep_extrusion(
            gapped, (0.0, 0.0, 2.0), coordinate_contract=SpatialCoordinateContract.si()
        )


def test_parallel_cylinder_generator_coedges_retain_exact_conic_sources() -> None:
    from phydrax.geometry.brep._constructors import (
        brep_extrusion,
        PlanarProfile,
        ProfileArc,
        ProfileLoop,
        ProfilePlane,
    )
    from phydrax.geometry.brep._intersection import ConicParameterEndpoint
    from phydrax.geometry.brep._patches import ConicRootLinePCurve

    profile = PlanarProfile(
        ProfilePlane(),
        ProfileLoop(
            ((0.0, -4.0), (0.0, 4.0)),
            (ProfileArc((-3.0, 0.0)), ProfileArc((3.0, 0.0))),
        ),
    )
    model = brep_extrusion(
        profile, (0.0, 0.0, 2.0), coordinate_contract=SpatialCoordinateContract.si()
    )
    geometry = model.geometry
    edge_candidates = []
    for index, curve_index in enumerate(geometry.edge_curves):
        if curve_index < 0:
            continue
        curve_candidate = geometry.curves[curve_index]
        if (
            isinstance(curve_candidate, LineCurve)
            and curve_candidate.origin_exact == (Fraction(0), Fraction(4), Fraction(0))
            and curve_candidate.direction_exact == (Fraction(0), Fraction(0), Fraction(2))
        ):
            edge_candidates.append(index)
    assert len(edge_candidates) == 1
    edge = edge_candidates[0]
    endpoints = (
        (Fraction(0), Fraction(4), Fraction(0)),
        (Fraction(0), Fraction(4), Fraction(2)),
    )
    edge_roots = geometry.edge_endpoint_roots[edge]
    assert all(isinstance(root, ConicParameterEndpoint) for root in edge_roots)
    assert (
        tuple(
            root.point_exact
            for root in edge_roots
            if isinstance(root, ConicParameterEndpoint)
        )
        == endpoints
    )
    for vertex, point_exact in zip(geometry.edge_vertices[edge], endpoints, strict=True):
        vertex_root = geometry.vertex_roots[vertex]
        assert vertex_root is not None
        assert isinstance(vertex_root.primary, BRepConicVertex)
        assert vertex_root.primary.point_exact == point_exact

    incident = tuple(
        (face, coedge)
        for face, loops in enumerate(geometry.face_loops)
        for loop in loops
        for coedge in loop
        if geometry.coedge_edges[coedge] == edge
    )
    assert len(incident) == 2
    curve = geometry.curves[geometry.edge_curves[edge]]
    assert isinstance(curve, LineCurve)
    first, last = map(float, np.asarray(geometry.edge_ranges[edge]))
    for face, coedge in incident:
        patch = model.patches[face]
        assert isinstance(patch, CylinderPatch)
        pcurve = geometry.pcurves[coedge]
        assert isinstance(pcurve, ConicRootLinePCurve)
        roots = geometry.coedge_endpoint_roots[coedge]
        assert all(isinstance(root, ConicParameterEndpoint) for root in roots)
        assert (
            tuple(
                root.point_exact
                for root in roots
                if isinstance(root, ConicParameterEndpoint)
            )
            == endpoints
        )
        region = SurfaceRegion(
            patch, np.asarray(model.parameter_bounds[face], dtype=np.float64)
        )
        assert exact_source_lift(curve, pcurve, region, first, last)


def test_revolution_profile_line_lift_accepts_exact_affine_and_periodic_maps() -> None:
    from phydrax.geometry.brep._boolean_core import PeriodicPCurve
    from phydrax.geometry.brep._constructors import brep_cylinder

    model = brep_cylinder(3.0, 3.0, coordinate_contract=SpatialCoordinateContract.si())
    face = next(
        index
        for index, patch in enumerate(model.patches)
        if isinstance(patch, RevolutionSurface) and isinstance(patch.curve, LineCurve)
    )
    patch = model.patches[face]
    assert isinstance(patch, RevolutionSurface)
    assert isinstance(patch.curve, LineCurve)
    profile = patch.curve
    lower, upper = Fraction(1, 4), Fraction(3, 4)
    scale = upper - lower
    source_origin = tuple(
        profile.origin_exact[axis] + lower * profile.direction_exact[axis]
        for axis in range(3)
    )
    source_direction = tuple(scale * profile.direction_exact[axis] for axis in range(3))
    source = LineCurve(
        tuple(float(value) for value in source_origin),
        tuple(float(value) for value in source_direction),
        origin_exact=source_origin,
        direction_exact=source_direction,
    )
    pcurve = LineCurve(
        (0.0, float(lower)),
        (0.0, float(scale)),
        origin_exact=(Fraction(0), lower),
        direction_exact=(Fraction(0), scale),
    )
    region = SurfaceRegion(
        patch, np.asarray(model.parameter_bounds[face], dtype=np.float64)
    )
    assert exact_source_lift(source, pcurve, region, 0.0, 1.0)
    assert exact_source_lift(
        source, PeriodicPCurve(pcurve, patch, (1, 0)), region, 0.0, 1.0
    )
    shifted = LineCurve(
        (float(np.pi * 2.0), float(lower)),
        (0.0, float(scale)),
        origin_exact=(Fraction.from_float(float(np.pi * 2.0)), lower),
        direction_exact=(Fraction(0), scale),
    )
    assert not exact_source_lift(source, shifted, region, 0.0, 1.0)


def test_torus_meridian_circle_lift_uses_exact_source_harmonics() -> None:
    from phydrax.geometry.brep._boolean_core import PeriodicPCurve

    torus = TorusPatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        3.0,
        0.5,
    )
    curve = CircleCurve(
        (3.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        0.5,
    )
    pcurve = LineCurve((0.0, 0.0), (0.0, 1.0))
    period = float(2.0 * np.pi)
    region = SurfaceRegion(
        torus, np.asarray(((0.0, 0.0), (period, period)), dtype=np.float64)
    )
    assert exact_source_lift(curve, pcurve, region, 0.0, period)
    assert exact_source_lift(
        curve, PeriodicPCurve(pcurve, torus, (1, 0)), region, 0.0, period
    )
    assert exact_source_lift(
        curve, PeriodicPCurve(pcurve, torus, (0, 1)), region, 0.0, period
    )
    assert not exact_source_lift(
        curve,
        LineCurve((period, 0.0), (0.0, 1.0)),
        region,
        0.0,
        period,
    )
    wrong_curve = CircleCurve(
        (3.25, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        0.5,
    )
    assert not exact_source_lift(wrong_curve, pcurve, region, 0.0, period)


def test_revolution_and_circle_rails_lift_to_ruled_surface() -> None:
    profile = CircleCurve(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        1.0,
    )
    revolution = RevolutionSurface(profile, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    ring = SurfaceIsoparametricCurve(revolution, 1, 0.5)
    first_rail = CircleCurve(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        1.0,
    )
    ruled = RuledSurface(first_rail, ring)
    period = float(2.0 * np.pi)
    region = SurfaceRegion(
        ruled, np.asarray(((0.0, 0.0), (period, 1.0)), dtype=np.float64)
    )
    pcurve = LineCurve((0.0, 1.0), (1.0, 0.0))
    first_pcurve = LineCurve((0.0, 0.0), (1.0, 0.0))

    assert exact_source_lift(first_rail, first_pcurve, region, 0.0, period)
    assert not exact_source_lift(first_rail, pcurve, region, 0.0, period)
    assert not exact_source_lift(
        CircleCurve(
            (0.25, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            1.0,
        ),
        first_pcurve,
        region,
        0.0,
        period,
    )
    assert not exact_source_lift(
        first_rail,
        LineCurve((period, 0.0), (-1.0, 0.0)),
        region,
        0.0,
        period,
    )

    assert exact_source_lift(ring, pcurve, region, 0.0, period)
    assert not exact_source_lift(
        SurfaceIsoparametricCurve(revolution, 1, 0.75),
        pcurve,
        region,
        0.0,
        period,
    )
    assert not exact_source_lift(
        ring, LineCurve((0.0, 0.0), (1.0, 0.0)), region, 0.0, period
    )
    assert not exact_source_lift(
        ring,
        LineCurve((period, 1.0), (-1.0, 0.0)),
        region,
        0.0,
        period,
    )


def test_plane_cylinder_section_factory_lifts_exact_conic_edges() -> None:
    import math

    from phydrax.geometry.brep._boolean_core import PeriodicPCurve
    from phydrax.geometry.brep._intersection import exact_plane_cylinder_conic_section

    unit_circle = CircleCurve(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        1.0,
        center_exact=(Fraction(0), Fraction(0), Fraction(0)),
        first_axis_exact=(Fraction(1), Fraction(0), Fraction(0)),
        second_axis_exact=(Fraction(0), Fraction(1), Fraction(0)),
        radius_exact=Fraction(1),
    )
    plane = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, -1.0), (0.0, 1.0, 0.0))
    cylinder = CylinderPatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        3.0,
    )
    plane_region = SurfaceRegion(
        plane, np.asarray(((-4.0, -4.0), (4.0, 4.0)), dtype=np.float64)
    )
    cylinder_region = SurfaceRegion(
        cylinder,
        np.asarray(((0.0, -4.0), (math.pi / 2.0, 4.0)), dtype=np.float64),
    )
    angle_roots = (
        ConicParameterRoot(unit_circle, Fraction(1), Fraction(0)),
        ConicParameterRoot(unit_circle, Fraction(0), Fraction(1)),
    )
    section = exact_plane_cylinder_conic_section(
        plane_region, cylinder_region, angle_roots
    )
    assert section is not None
    assert section.cylinder_pcurve.angle_offset_root is not None
    first, last = section.parameter_range
    assert exact_source_lift(
        section.edge_curve, section.plane_pcurve, plane_region, first, last
    )
    assert exact_source_lift(
        section.edge_curve, section.cylinder_pcurve, cylinder_region, first, last
    )
    assert exact_source_lift(
        section.edge_curve,
        PeriodicPCurve(section.cylinder_pcurve, cylinder, (1, 0)),
        cylinder_region,
        first,
        last,
    )
    wrong_center = (
        section.edge_curve.center_exact[0] + Fraction(1, 4),
        section.edge_curve.center_exact[1],
        section.edge_curve.center_exact[2],
    )
    wrong_curve = EllipseCurve(
        tuple(float(value) for value in wrong_center),
        tuple(float(value) for value in section.edge_curve.first_axis_exact),
        tuple(float(value) for value in section.edge_curve.second_axis_exact),
        float(section.edge_curve.first_radius_exact),
        float(section.edge_curve.second_radius_exact),
        center_exact=wrong_center,
        first_axis_exact=section.edge_curve.first_axis_exact,
        second_axis_exact=section.edge_curve.second_axis_exact,
        first_radius_exact=section.edge_curve.first_radius_exact,
        second_radius_exact=section.edge_curve.second_radius_exact,
    )
    assert not exact_source_lift(
        wrong_curve, section.cylinder_pcurve, cylinder_region, first, last
    )


@pytest.mark.parametrize("k", (1, 2, 3))
def test_endpoint_gap_bounds_honor_exact_quarter_turn_conic_roots(k: int) -> None:
    curve = EllipseCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 2.0, 1.0)
    cosine, sine = ((0, 1), (-1, 0), (0, -1))[k - 1]
    root = ConicParameterRoot(curve, Fraction(cosine), Fraction(sine))
    point = (2.0 * cosine, float(sine))
    parameter = k * float(np.pi) / 2
    vertices = np.asarray((point, point))
    unrooted = endpoint_gap_bounds(curve, (parameter, parameter), vertices)
    rooted = endpoint_gap_bounds(
        curve, (parameter, parameter), vertices, endpoint_roots=(root, None)
    )
    # The binary64 parameter only approximates k pi / 2; the root is exact.
    assert rooted == (0.0, unrooted[1]) and unrooted[1] > 0.0
    other = ConicParameterRoot(curve, Fraction(-cosine), Fraction(-sine))
    with pytest.raises(ValueError, match="realize its exact conic root"):
        endpoint_gap_bounds(
            curve, (parameter, parameter), vertices, endpoint_roots=(other, None)
        )


def test_nonconvex_ellipse_profile_refusal_names_its_missing_proof() -> None:
    from phydrax.geometry.brep._constructors import (
        PlanarProfile,
        ProfileCurveUse,
        ProfileLine,
        ProfileLoop,
        ProfilePlane,
        UnsupportedProfileLoop,
    )

    curve = EllipseCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 2.0, 1.0)
    loop = ProfileLoop(
        ((2.0, 0.0), (0.0, -1.0), (0.0, 0.0)),
        (ProfileCurveUse(curve, 0.0, 3 * np.pi / 2), ProfileLine(), ProfileLine()),
    )
    with pytest.raises(UnsupportedProfileLoop) as refusal:
        PlanarProfile(ProfilePlane(), loop)
    assert refusal.value.feature == "nonconvex-curved-loop-simplicity"
    assert isinstance(refusal.value, NotImplementedError)
