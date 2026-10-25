"""Construction, codec, and symbolic-endpoint contracts for B-Rep carriers.

Codec tests exercise the interchange boundary directly. All numerical equality
checks here are bitwise: reconstruction retains the same source coefficients.
"""

import math
from fractions import Fraction
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.geometry.brep._patches as patch_owner
from phydrax import geometry
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._intersection import NativePeriodEndpoint
from phydrax.geometry.brep._intersection_curve import (
    affine_curve_coefficients,
    AffinePCurve,
    decode_geometry,
    encode_geometry,
    GeometryDefinition,
    intersection_derivative_bounds,
    IntersectionCurveSide,
    PeriodicPCurve,
)
from phydrax.geometry.brep._root_bindings import BRepPlacedVertex


class _CoefficientPair(StrictModule):
    first: AffinePCurve
    second: AffinePCurve


def _plane() -> geometry.PlanePatch:
    return geometry.PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0))


def _space_line() -> geometry.LineCurve:
    return geometry.LineCurve((1, 0, 0), (0, 0, 1))


def _circle() -> geometry.CircleCurve:
    return geometry.CircleCurve((0, 0), (1, 0), (0, 1), 1)


def _sphere() -> geometry.SpherePatch:
    return geometry.SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1)


def _definition(case: str) -> GeometryDefinition:
    match case:
        case "plane":
            return _plane()
        case "cylinder":
            return geometry.CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1)
        case "cone":
            return geometry.ConePatch(
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1, 0.125
            )
        case "sphere":
            return _sphere()
        case "torus":
            return geometry.TorusPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 2, 0.5)
        case "offset-surface":
            return geometry.OffsetSurface(_sphere(), 0.125)
        case "isoline":
            return patch_owner.SurfaceIsoparametricCurve(
                _plane(), 0, 0.25, parameter_range=(0, 1)
            )
        case "spline-surface":
            return geometry.BSplineSurfacePatch(
                [[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]],
                [[1, 1], [1, 1]],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                1,
                1,
            )
        case "extrusion":
            return geometry.ExtrusionSurface(_space_line(), (0, 1, 0))
        case "revolution":
            return geometry.RevolutionSurface(_space_line(), (0, 0, 0), (0, 0, 1))
        case "ruled":
            return geometry.RuledSurface(
                _space_line(), geometry.LineCurve((2, 0, 0), (0, 0, 1))
            )
        case "line":
            return geometry.LineCurve((0, 0), (1, 0))
        case "circle":
            return _circle()
        case "ellipse":
            return geometry.EllipseCurve((0, 0), (1, 0), (0, 1), 2, 1)
        case "parabola":
            return geometry.ParabolaCurve((0, 0), (1, 0), (0, 1), 1)
        case "hyperbola":
            return geometry.HyperbolaCurve((0, 0), (1, 0), (0, 1), 2, 1)
        case "offset-curve":
            return geometry.OffsetCurve(geometry.LineCurve((0, 0), (1, 0)), 0.125)
        case "spline-curve":
            return geometry.BSplineCurve([[0, 0], [1, 1]], [1, 1], [0, 0, 1, 1], 1)
        case "region":
            return geometry.SurfaceRegion(_plane(), [[0, 0], [1, 1]])
        case "affine":
            return AffinePCurve(
                _circle(),
                ((Fraction(2), Fraction(0)), (Fraction(0), Fraction(3))),
                (Fraction(1, 3), Fraction(-1, 7)),
            )
        case "periodic":
            return PeriodicPCurve(geometry.LineCurve((0, 0), (1, 0)), _sphere(), (1, 0))
        case "segment":
            return geometry.CurveTrimSegment(_circle(), 0, 1, reversed=True)
        case _:
            raise ValueError(f"test.case must identify a carrier fixture; got {case!r}")


# These ordered fields are the persisted construction contract, not dataclass
# layout or implementation wiring. Derived caches must not enter the payload.
@pytest.mark.parametrize(
    ("case", "fields"),
    [
        ("plane", ("origin", "first_axis", "second_axis")),
        ("cylinder", ("origin", "first_axis", "second_axis", "axis", "radius")),
        (
            "cone",
            (
                "origin",
                "first_axis",
                "second_axis",
                "axis",
                "reference_radius",
                "semi_angle",
            ),
        ),
        ("sphere", ("center", "first_axis", "second_axis", "axis", "radius")),
        (
            "torus",
            (
                "center",
                "first_axis",
                "second_axis",
                "axis",
                "major_radius",
                "minor_radius",
            ),
        ),
        ("offset-surface", ("base", "distance")),
        ("isoline", ("surface", "fixed_axis", "fixed_value", "parameter_range")),
        (
            "spline-surface",
            ("control_points", "weights", "u_knots", "v_knots", "u_degree", "v_degree"),
        ),
        ("extrusion", ("curve", "direction")),
        ("revolution", ("curve", "axis_origin", "axis_direction")),
        ("ruled", ("first", "second")),
        ("line", ("origin", "direction")),
        ("circle", ("center", "first_axis", "second_axis", "radius")),
        (
            "ellipse",
            ("center", "first_axis", "second_axis", "first_radius", "second_radius"),
        ),
        ("parabola", ("vertex", "first_axis", "second_axis", "focal_length")),
        (
            "hyperbola",
            ("center", "first_axis", "second_axis", "first_radius", "second_radius"),
        ),
        ("offset-curve", ("base", "distance", "direction")),
        ("spline-curve", ("control_points", "weights", "knots", "degree")),
        ("region", ("patch", "parameter_box")),
        ("affine", ("curve", "matrix", "offset")),
        ("periodic", ("source_curve", "patch", "period_shifts")),
        ("segment", ("curve", "first", "last", "reversed", "first_root", "last_root")),
    ],
    ids=[
        "plane",
        "cylinder",
        "cone",
        "sphere",
        "torus",
        "offset-surface",
        "isoline",
        "spline-surface",
        "extrusion",
        "revolution",
        "ruled",
        "line",
        "circle",
        "ellipse",
        "parabola",
        "hyperbola",
        "offset-curve",
        "spline-curve",
        "region",
        "affine",
        "periodic",
        "segment",
    ],
)
def test_construction_codec_retains_field_order_and_payload(
    case: str, fields: tuple[str, ...]
) -> None:
    definition = _definition(case)
    payload = encode_geometry(definition)
    assert tuple(payload["fields"]) == fields
    restored = decode_geometry(payload)
    assert type(restored) is type(definition)
    assert encode_geometry(restored) == payload


def test_line_codec_retains_binary64_scalar_representation() -> None:
    payload = encode_geometry(geometry.LineCurve((-0.0, 0.5), (1, 0)))
    assert payload == {
        "type": "LineCurve",
        "fields": {
            "origin": {
                "dtype": "float64",
                "shape": [2],
                "hex": ["-0x0.0p+0", "0x1.0000000000000p-1"],
            },
            "direction": {
                "dtype": "float64",
                "shape": [2],
                "hex": ["0x1.0000000000000p+0", "0x0.0p+0"],
            },
        },
    }
    assert encode_geometry(decode_geometry(payload)) == payload


@pytest.mark.parametrize(
    "payload",
    [{"fraction": [2, 4]}, {"fraction": [1, 0]}, {"fraction": [True, 1]}],
    ids=["unreduced", "zero-denominator", "boolean-numerator"],
)
def test_fraction_codec_refuses_invalid_construction(
    payload: dict[str, list[int]],
) -> None:
    with pytest.raises(ValueError, match="geometry.fraction"):
        decode_geometry(payload)


def test_decoder_refuses_extra_construction_fields() -> None:
    payload = encode_geometry(_circle())
    payload["fields"]["derived_cache"] = 1
    with pytest.raises(ValueError, match="geometry.fields"):
        decode_geometry(payload)


def test_placed_vertex_codec_retains_positional_source_and_identity() -> None:
    vertex = BRepPlacedVertex(
        (1, 2, 3), np.eye(3, dtype=np.float64), (0, 0, 0), "source:vertex:0"
    )
    payload = encode_geometry(vertex)
    restored = decode_geometry(payload)
    assert isinstance(restored, BRepPlacedVertex)
    assert restored.root_id == vertex.root_id
    assert encode_geometry(restored) == payload


@pytest.mark.parametrize("reversed", [False, True], ids=["forward", "reversed"])
def test_trim_source_parameters_preserve_scalar_and_array_mapping(reversed: bool) -> None:
    trim = geometry.CurveTrimSegment(
        geometry.LineCurve((0, 0), (1, 0)), 2, 6, reversed=reversed
    )
    local = jnp.asarray([0.0, 0.25, 1.0], dtype=jnp.float64)
    expected = 6 - 4 * local if reversed else 2 + 4 * local
    np.testing.assert_array_equal(trim.source_parameters(local), expected)
    assert trim.source_parameters(0.25) == float(expected[1])


def test_affine_and_periodic_continuity_use_the_source_contract() -> None:
    line = geometry.LineCurve((0, 0), (1, 0))
    affine = AffinePCurve(
        line,
        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
        (Fraction(0), Fraction(0)),
    )
    periodic = PeriodicPCurve(affine, _sphere(), (1, 0))
    assert affine.is_c1_on(0, 1)
    assert periodic.is_c1_on(0, 1)


def test_symbolic_quarter_turn_joins_the_rational_endpoint() -> None:
    circle = _circle()
    endpoint = NativePeriodEndpoint(circle, turns=Fraction(1, 4))
    arc = geometry.CurveTrimSegment(circle, 0, endpoint.parameter, last_root=endpoint)
    line = geometry.CurveTrimSegment(geometry.LineCurve((0, 1), (1, 0)), 0, 1)
    assert arc.shares_endpoint(line)
    assert decode_geometry(encode_geometry(arc)).shares_endpoint(line)


def test_binary_quarter_turn_is_not_reinterpreted_as_a_symbolic_turn() -> None:
    arc = geometry.CurveTrimSegment(_circle(), 0, math.pi / 2)
    line = geometry.CurveTrimSegment(geometry.LineCurve((0, 1), (1, 0)), 0, 1)
    assert not arc.shares_endpoint(line)


@pytest.fixture(scope="module")
def branch() -> geometry.IntersectionCurve:
    plane = geometry.SurfaceRegion(
        geometry.PlanePatch((-2, -2, 0.5), (4, 0, 0), (0, 4, 0)), [[0, 0], [1, 1]]
    )
    sphere = geometry.SurfaceRegion(
        _sphere(), [[0, -math.pi / 2], [math.tau, math.pi / 2]]
    )
    return geometry.intersect_surface_regions(plane, sphere).curves[0]


@pytest.fixture(scope="module")
def expanding_branch() -> geometry.IntersectionCurve:
    # This source construction also owns the live-source differentiation
    # scenario. Positive rational weights bound the sheet in its control hull;
    # the independent plane equation fixes every intersection point's y to 0.4.
    heights = [
        [0.0, 0.1, 0.0],
        [0.2, 0.3, 0.1],
        [0.1, 0.4, 0.2],
        [0.0, 0.2, 0.1],
    ]
    xs, ys = (0.0, 0.3, 0.7, 1.0), (0.0, 0.5, 1.0)
    points = [[[x, y, heights[i][j]] for j, y in enumerate(ys)] for i, x in enumerate(xs)]
    sheet = geometry.SurfaceRegion(
        geometry.BSplineSurfacePatch(
            points,
            [[1.0, 1.3, 1.0], [0.9, 1.2, 1.1], [1.1, 0.8, 1.0], [1.0, 1.1, 0.9]],
            [0, 0, 0, 0.5, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            2,
            2,
        ),
        [[0, 0], [1, 1]],
    )
    plane = geometry.PlanePatch((-0.5, 0.4, -2.0), (2, 0, 0), (0, 0, 4))
    transverse = geometry.SurfaceRegion(plane, [[0, 0], [1, 1]])
    result = geometry.intersect_surface_regions(sheet, transverse)
    # The test owns a locally certified point and its necessary source expansion;
    # global event discovery at parameter-domain walls is a separate contract.
    assert len(result.curves) == 1
    curve = result.curves[0]
    assert curve.fully_certified
    point = curve.evaluate(jnp.asarray(0.25, dtype=jnp.float64))
    assert math.isfinite(float(point.parameter_bound))
    np.testing.assert_array_equal(
        np.asarray(plane.evaluate(point.second_parameters))[1],
        np.asarray(0.4, dtype=np.float64),
    )
    return curve


def test_branch_constructor_preserves_identity_payload(
    branch: geometry.IntersectionCurve,
) -> None:
    expected = canonical_fingerprint(
        {
            "kind": "intersection-curve",
            "first": encode_geometry(branch.first.patch),
            "first_box": [list(branch.first.lower), list(branch.first.upper)],
            "second": encode_geometry(branch.second.patch),
            "second_box": [list(branch.second.lower), list(branch.second.upper)],
            "atlas": array_tree_fingerprint(
                (
                    branch.chart_axes,
                    branch.chart_start,
                    branch.chart_end,
                    branch.chart_pieces,
                    branch.box_lower,
                    branch.box_upper,
                    branch.preconditioners,
                    branch.contraction,
                    branch.certified,
                    branch.transition_shifts,
                    branch.node_axes,
                    branch.node_values,
                    branch.node_pieces,
                    branch.node_references,
                    branch.node_period_shifts,
                )
            ),
            "closed": branch.closed,
            "endpoints": [branch.start_kind, branch.end_kind],
        }
    )
    assert branch.branch_id == expected
    restored = geometry.IntersectionCurve.from_payload(branch.payload())
    assert restored.branch_id == expected
    assert restored.payload() == branch.payload()


@pytest.mark.parametrize(
    "side,dimension",
    [(None, 3), ("first", 2), ("second", 2), ("all", 7)],
    ids=["spatial", "first-uv", "second-uv", "coupled"],
)
def test_second_jet_across_chart_transition_retains_unbounded_dimension(
    branch: geometry.IntersectionCurve,
    side: IntersectionCurveSide | Literal["all"] | None,
    dimension: int,
) -> None:
    with coordinate_source_request((branch,)) as budget:
        lower, upper = intersection_derivative_bounds(
            branch, 0, min(2, branch.num_charts), 2, side, budget=budget
        )
    np.testing.assert_array_equal(lower, np.full(dimension, -np.inf))
    np.testing.assert_array_equal(upper, np.full(dimension, np.inf))


def test_parameter_shift_bounds_retain_symbolic_period_enclosure(
    branch: geometry.IntersectionCurve,
) -> None:
    shifts = np.asarray([0, 0, 1, 0], dtype=np.int32)
    lower, upper = branch.shift_parameter_bounds(shifts)
    mathematical_tau = Fraction("6.28318530717958647692528676655900576839433879875021")
    assert Fraction(float(lower[2])) <= mathematical_tau <= Fraction(float(upper[2]))
    np.testing.assert_array_equal(lower[[0, 1, 3]], np.zeros(3, dtype=np.float64))
    np.testing.assert_array_equal(upper[[0, 1, 3]], np.zeros(3, dtype=np.float64))


def test_coefficient_merge_encloses_distinct_rationals_with_equal_binary_values() -> None:
    first_scale = Fraction(1) + Fraction(1, 2**60)
    second_scale = Fraction(1) - Fraction(1, 2**60)
    source = geometry.LineCurve((0, 0), (1, 0))
    first = AffinePCurve(
        source,
        ((first_scale, Fraction(0)), (Fraction(0), Fraction(1))),
        (Fraction(0), Fraction(0)),
    )
    second = AffinePCurve(
        source,
        ((second_scale, Fraction(0)), (Fraction(0), Fraction(1))),
        (Fraction(0), Fraction(0)),
    )
    enclosures = patch_owner.coefficient_enclosures(_CoefficientPair(first, second))
    matrices = tuple(entry for entry in enclosures if entry[0].shape == (2, 2))
    assert len(matrices) == 1
    nominal, lower, upper = matrices[0]
    np.testing.assert_array_equal(nominal, np.eye(2, dtype=np.float64))
    assert Fraction(float(lower[0, 0])) <= second_scale < first_scale
    assert Fraction(float(upper[0, 0])) >= first_scale > second_scale


def test_live_source_without_a_chart_root_refuses_native_nonconvergence(
    branch: geometry.IntersectionCurve,
) -> None:
    # The shifted plane has z=50 while the generating sphere has |z|<=1.
    # This independently proves that the revised coupled equations have no root.
    plane = branch.first.patch
    assert isinstance(plane, geometry.PlanePatch)
    shifted = eqx.tree_at(
        lambda patch: patch.origin,
        plane,
        jnp.asarray((-2.0, -2.0, 50.0), dtype=jnp.float64),
    )
    revised = eqx.tree_at(lambda curve: curve.first.patch, branch, shifted)
    with pytest.raises(eqx.EquinoxRuntimeError, match="IntersectionCurve.chart_root"):
        jax.block_until_ready(
            revised.evaluate(jnp.asarray(0.25, dtype=jnp.float64)).point
        )


def test_surface_region_stored_projection_round_trips() -> None:
    region = geometry.SurfaceRegion(_plane(), [[0, 0], [1, 1]])
    restored = geometry.SurfaceRegion.from_stored_fields(region)
    assert encode_geometry(restored) == encode_geometry(region)
    assert restored.periodic == region.periodic


def test_surface_region_stored_projection_refuses_incompatible_patch_domain() -> None:
    region = geometry.SurfaceRegion(_plane(), [[0, 0], [1, 2]])
    corrupted = eqx.tree_at(lambda value: value.patch, region, _sphere())
    with pytest.raises(ValueError, match="SpherePatch.latitude"):
        geometry.SurfaceRegion.from_stored_fields(corrupted)


def test_curve_range_stored_projection_round_trips() -> None:
    curve_range = geometry.CurveRange(_circle(), 0, math.tau)
    restored = geometry.CurveRange.from_stored_fields(curve_range)
    assert restored.first == curve_range.first
    assert restored.last == curve_range.last
    assert restored.periodic == curve_range.periodic
    assert encode_geometry(restored.curve) == encode_geometry(curve_range.curve)


def test_curve_range_stored_projection_refuses_incompatible_carrier_range() -> None:
    curve_range = geometry.CurveRange(geometry.LineCurve((0, 0), (1, 0)), 0, 10)
    corrupted = eqx.tree_at(lambda value: value.curve, curve_range, _circle())
    with pytest.raises(ValueError, match="curve.parameter_range"):
        geometry.CurveRange.from_stored_fields(corrupted)


def test_periodic_pcurve_stored_projection_round_trips() -> None:
    pcurve = PeriodicPCurve(geometry.LineCurve((0, 0), (1, 0)), _sphere(), (1, 0))
    restored = PeriodicPCurve.from_stored_fields(pcurve)
    assert encode_geometry(restored) == encode_geometry(pcurve)
    np.testing.assert_array_equal(restored.offset_value, pcurve.offset_value)


def test_periodic_pcurve_stored_projection_refuses_wrong_source_dimension() -> None:
    pcurve = PeriodicPCurve(geometry.LineCurve((0, 0), (1, 0)), _sphere(), (1, 0))
    corrupted = eqx.tree_at(lambda value: value.source_curve, pcurve, _space_line())
    with pytest.raises(TypeError, match="PeriodicPCurve.source_curve"):
        PeriodicPCurve.from_stored_fields(corrupted)


def test_intersection_stored_projection_round_trips(
    branch: geometry.IntersectionCurve,
) -> None:
    restored = geometry.IntersectionCurve.from_stored_fields(branch)
    assert restored.branch_id == branch.branch_id
    assert restored.payload() == branch.payload()
    np.testing.assert_array_equal(restored.nodes_certified, branch.nodes_certified)
    np.testing.assert_array_equal(
        restored.transitions_certified, branch.transitions_certified
    )


def test_intersection_banks_are_immutable_snapshots_of_caller_storage(
    branch: geometry.IntersectionCurve,
) -> None:
    from phydrax.geometry.brep._carrier_pieces import admit_carrier_fingerprint

    caller = np.array(branch.chart_start, copy=True)
    rebuilt = geometry.IntersectionCurve.from_stored_fields(
        eqx.tree_at(lambda value: value.chart_start, branch, caller)
    )
    expected = caller.copy()
    caller += 1.0
    np.testing.assert_array_equal(rebuilt.chart_start, expected)
    assert rebuilt.branch_id == branch.branch_id
    assert array_tree_fingerprint(rebuilt) == array_tree_fingerprint(branch)
    for bank in (rebuilt.chart_start, rebuilt.preconditioners, rebuilt.node_lower):
        with pytest.raises(ValueError):
            bank.setflags(write=True)
    # A carrier is reusable only when every host bank is immutable; a caller's
    # writable bank keeps its owner reauthenticated on every use.
    foreign = eqx.tree_at(lambda value: value.chart_start, rebuilt, caller)
    with coordinate_source_request((rebuilt, foreign)) as budget:
        assert admit_carrier_fingerprint(rebuilt, budget)
        assert not admit_carrier_fingerprint(foreign, budget)


def test_intersection_stored_projection_refuses_invalid_chart_axis(
    branch: geometry.IntersectionCurve,
) -> None:
    corrupted = eqx.tree_at(
        lambda value: value.chart_axes,
        branch,
        np.full_like(branch.chart_axes, 4),
    )
    with pytest.raises(ValueError, match="IntersectionCurve.chart_axes"):
        geometry.IntersectionCurve.from_stored_fields(corrupted)


def test_affine_spline_coefficients_retain_nonbinary_rational_slope() -> None:
    curve = geometry.BSplineCurve([[0, 0], [1, 1]], [2, 2], [0, 0, 3, 3], 1)
    coefficients = affine_curve_coefficients(curve, np.zeros(2, dtype=np.float64))
    assert coefficients == (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(1, 3)),
    )


@pytest.mark.parametrize(
    "knots",
    [(1.0, 1.0, 1.0, 1.0), (2.0, 2.0, 1.0, 1.0)],
    ids=["empty-active-span", "reversed-active-span"],
)
def test_affine_spline_coefficients_refuse_nonpositive_active_span(
    knots: tuple[float, float, float, float],
) -> None:
    curve = geometry.BSplineCurve([[0, 0], [1, 1]], [1, 1], [0, 0, 1, 1], 1)
    corrupted = eqx.tree_at(
        lambda value: value.knots,
        curve,
        jnp.asarray(knots, dtype=jnp.float64),
    )
    assert affine_curve_coefficients(corrupted, np.zeros(2, dtype=np.float64)) is None


def _context_trim(
    case: str,
    branch: geometry.IntersectionCurve,
) -> (
    geometry.IntersectionPCurve
    | AffinePCurve
    | PeriodicPCurve
    | geometry.CurveTrimSegment
):
    source = branch.p_curve("first")
    match case:
        case "intersection":
            return source
        case "affine":
            return AffinePCurve(
                source,
                ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))),
                (Fraction(1, 7), Fraction(0)),
            )
        case "periodic":
            return PeriodicPCurve(source, branch.first.patch, (0, 0))
        case "segment":
            return geometry.CurveTrimSegment(source, 0, 1)
        case _:
            raise ValueError(
                f"test.case must identify a trim evaluation variant; got {case!r}"
            )


@pytest.mark.parametrize(
    "case",
    ["intersection", "affine", "periodic", "segment"],
    ids=["coupled-pcurve", "affine-transport", "period-transport", "oriented-segment"],
)
def test_evaluation_with_caller_context_preserves_standalone_values(
    case: str,
    branch: geometry.IntersectionCurve,
) -> None:
    curve = _context_trim(case, branch)
    parameters = jnp.asarray([0.125, 0.25], dtype=jnp.float64)
    expected = curve.evaluate(parameters)
    with coordinate_source_request((curve, parameters)) as budget:
        observed = curve.evaluate(parameters, budget=budget)
    np.testing.assert_array_equal(observed, expected)


def test_rich_curve_evaluation_refuses_an_exhausted_caller_certification_stage(
    expanding_branch: geometry.IntersectionCurve,
) -> None:
    parameter = jnp.asarray(0.25, dtype=jnp.float64)
    with coordinate_source_request((expanding_branch, parameter)) as budget:
        # Source Bernstein-span extraction is necessary for this spline-backed
        # point certificate. A zero-work caller stage cannot fund that expansion.
        with budget.bound_stage(0, budget.maximum_memory_bytes):
            with pytest.raises(
                CoordinateEnclosureResourceError,
                match="CoordinateEnclosureBudget.coefficient_work",
            ) as refused:
                with budget.measurement():
                    expanding_branch.evaluate(parameter, budget=budget)
        assert refused.value.resource == "coefficient_work"
        assert refused.value.limit == 0
        assert refused.value.completed == 0
        assert refused.value.requested > 0
        assert budget.work_units == 0
        assert budget.required_work_units >= refused.value.requested
        phase = refused.value.phase_evidence
        assert phase is not None
        assert phase.work_units == 0
