"""Patch enclosure and offset-expression contracts through their public owners."""

from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._resource_ticket import HostBatchTicket
from phydrax._strict import StrictModule
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry._domain_numerics import bucketed
from phydrax.geometry.brep._carrier_pieces import (
    CurveCoordinateResult,
    RootCurveCoordinateSource,
)
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._patches import (
    AbstractSurfacePatch,
    BernsteinCurvePiece,
    BernsteinSurfacePiece,
    BSplineCurve,
    BSplineSurfacePatch,
    coefficient_enclosures,
    ConePatch,
    curve_evaluate_host,
    curve_value_and_derivative,
    curve_value_and_derivative_host,
    CylinderPatch,
    exact_frame_orientation,
    ExtrusionSurface,
    LineCurve,
    offset_chain,
    OffsetSurface,
    patch_evaluation_error,
    PlanePatch,
    prepare_curve_host,
    RevolutionSurface,
    RuledSurface,
    sphere_source_equivalence,
    sphere_source_radius_terms,
    SpherePatch,
    surface_evaluate_host,
    surface_value_and_differential,
    surface_value_and_differential_host,
    TorusPatch,
)
from phydrax.nonlinear import VectorLocalRootPlan


def _sphere(radius: float = 3.0) -> SpherePatch:
    return SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), radius)


def _bilinear_patch() -> BSplineSurfacePatch:
    return BSplineSurfacePatch(
        (((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 1))),
        np.ones((2, 2), dtype=np.float64),
        (0, 0, 1, 1),
        (0, 0, 1, 1),
        1,
        1,
    )


def test_offset_chain_retains_nodes_and_fraction_operation_order() -> None:
    basis = _sphere()
    first = OffsetSurface(basis, 0.5)
    final = OffsetSurface(first, -0.25)
    retained, nodes = offset_chain(final)
    assert retained is basis
    assert nodes[0] is final and nodes[1] is first
    assert sphere_source_equivalence(final) == (basis, Fraction(13, 4))
    assert sphere_source_radius_terms(final) == (
        basis,
        (Fraction(3), Fraction(1, 2), Fraction(-1, 4)),
    )


def test_offset_chain_can_be_traced_without_reading_distance_leaves() -> None:
    source = OffsetSurface(_sphere(), 0.5)

    def radius_value(distance: Array) -> Array:
        # Functional updates keep the constructor's static normal-depth proof.
        patch = eqx.tree_at(lambda value: value.distance, source, distance)
        basis, nodes = offset_chain(patch)
        return basis.evaluate(jnp.zeros((2,), dtype=jnp.float64))[0] + nodes[0].distance

    np.testing.assert_array_equal(jax.jit(radius_value)(jnp.asarray(0.25)), 3.25)


def test_exact_frame_refuses_tolerance_only_orthogonality() -> None:
    assert exact_frame_orientation((1, 0, 0), (0, 1, 0), (0, 0, 1)) == 1
    assert exact_frame_orientation((1, 0, 0), (0, 1, 0), (0, 0, -1)) == -1
    assert exact_frame_orientation((1, 1.0e-12, 0), (0, 1, 0), (0, 0, 1)) is None


@pytest.mark.parametrize("order", (1, 2), ids=("first-jets", "mixed-second-jets"))
def test_tensor_derivative_bounds_enclose_bilinear_closed_form(order: int) -> None:
    patch = _bilinear_patch()
    boxes = np.asarray((((0.25, 0.125), (0.75, 0.875)),) * 2, dtype=np.float64)
    lower, upper = patch.derivative_bounds_batch(boxes, order=order)
    if order == 1:
        expected = np.asarray(((1, 0), (0, 1), (0.5, 0.5)), dtype=np.float64)
    else:
        expected = np.zeros((3, 2, 2), dtype=np.float64)
        expected[2, 0, 1] = expected[2, 1, 0] = 1
    assert np.all(lower[0] <= expected) and np.all(expected <= upper[0])
    np.testing.assert_array_equal(lower[0], lower[1])
    np.testing.assert_array_equal(upper[0], upper[1])


def test_rational_surface_bounds_reuse_only_current_full_source_and_range() -> None:
    patch = _bilinear_patch()
    box = np.asarray(((0.125, 0.25), (0.875, 0.75)), dtype=np.float64)
    ticket = HostBatchTicket(100_000, bytes_upper=16_000_000)
    owner = CoordinateEnclosureBudget(
        ticket, maximum_work_units=100_000, maximum_memory_bytes=16_000_000
    )
    with ticket, owner.activate():
        original = patch.bounding_box(box, budget=owner)
        cold_work = owner.work_units
        repeated = patch.bounding_box(box, budget=owner)
        assert 0 < owner.work_units - cold_work < cold_work
        np.testing.assert_array_equal(repeated.view(np.uint64), original.view(np.uint64))
        repeated[0, 0] = 99
        np.testing.assert_array_equal(
            patch.bounding_box(box, budget=owner).view(np.uint64),
            original.view(np.uint64),
        )
        points = np.asarray(patch.control_points).copy()
        points[1, 1, 2] += 1
        changed = BSplineSurfacePatch(
            points, patch.weights, patch.u_knots, patch.v_knots, 1, 1
        )
        changed_result = changed.bounding_box(box, budget=owner)
        assert not np.array_equal(changed_result, original)
        other_box = box.copy()
        other_box[1, 0] = 0.5
        other_result = patch.bounding_box(other_box, budget=owner)
        assert not np.array_equal(other_result, original)
    with coordinate_source_request((changed, box)) as independent:
        np.testing.assert_array_equal(
            changed.bounding_box(box, budget=independent).view(np.uint64),
            changed_result.view(np.uint64),
        )


def test_rational_curve_bounds_reuse_only_current_full_source_and_range() -> None:
    curve = BSplineCurve.bezier(((0, 0), (0.5, 1), (1, 0)), (1, 0.75, 1))
    ticket = HostBatchTicket(100_000, bytes_upper=16_000_000)
    owner = CoordinateEnclosureBudget(
        ticket, maximum_work_units=100_000, maximum_memory_bytes=16_000_000
    )
    with ticket, owner.activate():
        original = curve.bounding_box(0.125, 0.875, budget=owner)
        cold_work = owner.work_units
        repeated = curve.bounding_box(0.125, 0.875, budget=owner)
        assert 0 < owner.work_units - cold_work < cold_work
        np.testing.assert_array_equal(repeated.view(np.uint64), original.view(np.uint64))
        repeated[0, 0] = 99
        np.testing.assert_array_equal(
            curve.bounding_box(0.125, 0.875, budget=owner).view(np.uint64),
            original.view(np.uint64),
        )
        changed = BSplineCurve.bezier(((0, 0), (0.5, 2), (1, 0)), (1, 0.75, 1))
        changed_result = changed.bounding_box(0.125, 0.875, budget=owner)
        assert not np.array_equal(changed_result, original)
        other_result = curve.bounding_box(0.125, 0.5, budget=owner)
        assert not np.array_equal(other_result, original)
    with coordinate_source_request((changed, 0.125, 0.875)) as independent:
        np.testing.assert_array_equal(
            changed.bounding_box(0.125, 0.875, budget=independent).view(np.uint64),
            changed_result.view(np.uint64),
        )


def test_tensor_piece_jets_reuse_identical_clipped_ranges_without_changing_bounds() -> (
    None
):
    patch = BSplineSurfacePatch(
        (
            ((0, 0, 0), (0, 1, 0)),
            ((0.5, 0, 0.25), (0.5, 1, 0.75)),
            ((1, 0, 0.5), (1, 1, 1)),
        ),
        ((1, 0.9), (1.1, 1), (0.95, 1.05)),
        (0, 0, 0.5, 1, 1),
        (0, 0, 1, 1),
        1,
        1,
    )
    boxes = np.asarray((((0, 0.25), (0.75, 0.75)), ((0, 0.25), (1, 0.75))))
    # First derivatives remain defined across the degree-one internal knot.
    with coordinate_source_request((patch, boxes)) as owner:
        actual = patch.derivative_bounds_batch(boxes, order=1, budget=owner)
        shared_work = owner.work_units
    independent_work = 0
    for row, box in enumerate(boxes):
        with coordinate_source_request((patch, box)) as independent:
            expected = patch.derivative_bounds_batch(
                box[None], order=1, budget=independent
            )
            independent_work += independent.work_units
        for result, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(
                result[row].view(np.uint64), reference[0].view(np.uint64)
            )
    assert 0 < shared_work < independent_work


def test_extrusion_second_jets_keep_affine_sweep_exactly_zero() -> None:
    curve = BSplineCurve(((0, 0, 0), (1, 0, 1)), (1, 1), (0, 0, 1, 1), 1)
    patch = ExtrusionSurface(curve, (0, 1, 0))
    lower, upper = patch.derivative_bounds_batch(
        np.asarray((((0.25, 0.125), (0.75, 0.875)),), dtype=np.float64), order=2
    )
    assert np.all(lower[..., 0, 0] <= 0) and np.all(upper[..., 0, 0] >= 0)
    np.testing.assert_array_equal(lower[..., 1], 0)
    np.testing.assert_array_equal(upper[..., 1], 0)


def test_compiled_curve_boundary_preserves_linear_values_and_derivatives() -> None:
    curve = BSplineCurve(((0, 0, 0), (1, 0, 1)), (1, 1), (0, 0, 1, 1), 1)
    parameters = jnp.asarray((0.25, 0.5, 0.75), dtype=jnp.float64)
    values, derivatives = curve_value_and_derivative(curve, parameters)
    np.testing.assert_array_equal(
        values, np.stack((parameters, np.zeros(3), parameters), axis=1)
    )
    np.testing.assert_array_equal(derivatives, np.broadcast_to((1, 0, 1), (3, 3)))


def test_compiled_surface_boundary_preserves_cylinder_closed_form() -> None:
    patch = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 2)
    values, differential = surface_value_and_differential(
        patch, jnp.asarray(((0, 0.5),), dtype=jnp.float64)
    )
    np.testing.assert_array_equal(values, ((2, 0, 0.5),))
    np.testing.assert_array_equal(differential, (((0, 0), (2, 0), (0, 1)),))


@pytest.mark.parametrize("family", ("spline", "revolution", "offset"))
def test_second_jets_are_derivatives_of_first_jets(family: str) -> None:
    patch: AbstractSurfacePatch
    match family:
        case "spline":
            patch = _bilinear_patch()
        case "revolution" | "offset":
            curve = BSplineCurve(((1, 0, 0), (1, 0, 1)), (1, 1), (0, 0, 1, 1), 1)
            revolution = RevolutionSurface(curve, (0, 0, 0), (0, 0, 1))
            patch = (
                revolution if family == "revolution" else OffsetSurface(revolution, 0.25)
            )
        case _:
            raise ValueError(family)
    point = jnp.asarray((0.3, 0.4), dtype=jnp.float64)
    first_jet = jax.jacfwd(patch.evaluate)
    second_jet = np.asarray(jax.jacfwd(first_jet)(point))
    # Central differencing has O(h²) truncation and O(eps/h) roundoff.
    # Unit-domain bilinear/cylinder maps have third-jet magnitudes <= 1.25.
    # h=eps^(1/3) balances those terms; 64 covers the first-jet operation count.
    eps = np.finfo(np.float64).eps
    step = eps ** (1.0 / 3.0)
    tolerance = 64 * eps ** (2.0 / 3.0)
    for axis in range(2):
        delta = jnp.zeros((2,), dtype=jnp.float64).at[axis].set(step)
        reference = np.asarray(
            (first_jet(point + delta) - first_jet(point - delta)) / (2 * step)
        )
        np.testing.assert_allclose(
            second_jet[..., axis], reference, rtol=0, atol=tolerance
        )
    if family == "spline":
        expected = np.zeros((3, 2, 2), dtype=np.float64)
        expected[2, 0, 1] = expected[2, 1, 0] = 1
    else:
        radius = 1.0 if family == "revolution" else 1.25
        expected = np.zeros((3, 2, 2), dtype=np.float64)
        expected[:2, 0, 0] = -radius * np.asarray((np.cos(0.3), np.sin(0.3)))
    np.testing.assert_allclose(second_jet, expected, rtol=0, atol=64 * eps)


def _evaluation_patch(family: str) -> AbstractSurfacePatch:
    frame = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    profile = BSplineCurve(((1, 0, 0), (1, 0, 1)), (1, 1), (0, 0, 1, 1), 1)
    match family:
        case "plane":
            return PlanePatch(frame[0], frame[1], frame[2])
        case "sphere":
            return _sphere()
        case "cylinder":
            return CylinderPatch(*frame, 2)
        case "cone":
            return ConePatch(*frame, 2, 0.25)
        case "torus":
            return TorusPatch(*frame, 3, 1)
        case "extrusion":
            return ExtrusionSurface(profile, (0, 1, 0))
        case "revolution":
            return RevolutionSurface(profile, (0, 0, 0), (0, 0, 1))
        case "ruled":
            other = BSplineCurve(((2, 1, 0), (2, 1, 1)), (1, 1), (0, 0, 1, 1), 1)
            return RuledSurface(profile, other)
        case "spline":
            return _bilinear_patch()
        case "offset":
            return OffsetSurface(_sphere(), 0.25)
        case _:
            raise ValueError(family)


@pytest.mark.parametrize(
    "family",
    (
        "plane",
        "sphere",
        "cylinder",
        "cone",
        "torus",
        "extrusion",
        "revolution",
        "ruled",
        "spline",
        "offset",
    ),
)
def test_host_and_device_patch_evaluation_agree_with_certified_roundoff(
    family: str,
) -> None:
    patch = _evaluation_patch(family)
    parameters = jnp.asarray(((0.3, 0.4), (0.6, 0.7)), dtype=jnp.float64)
    host = surface_evaluate_host(patch, parameters)
    device = np.asarray(patch.evaluate(parameters))
    error = np.asarray(eqx.filter_jit(patch_evaluation_error)(patch, parameters))
    assert error.shape == device.shape
    assert np.all(np.isfinite(error)) and np.all(error >= 0)
    assert np.all(np.abs(host - device) <= error)


@pytest.mark.parametrize("exponent", (-30, 0, 30))
def test_patch_roundoff_enclosure_scales_with_coordinate_units(exponent: int) -> None:
    scale = 2.0**exponent
    patch = PlanePatch((3 * scale, 5 * scale, 7 * scale), (scale, 0, 0), (0, scale, 0))
    parameters = jnp.asarray((0.3, 0.4), dtype=jnp.float64)
    value = np.asarray(patch.evaluate(parameters))
    error = np.asarray(patch_evaluation_error(patch, parameters))
    reference = (
        Fraction(3 * scale) + Fraction(0.3) * Fraction(scale),
        Fraction(5 * scale) + Fraction(0.4) * Fraction(scale),
        Fraction(7 * scale),
    )
    for coordinate, expected, bound in zip(value, reference, error, strict=True):
        assert abs(Fraction(float(coordinate)) - expected) <= Fraction(float(bound))
    # Each coordinate has at most two products/additions and their directed
    # enclosures. 64 eps times the largest coordinate encloses that operation
    # count and rules out an absolute model-unit tolerance.
    assert np.max(error) <= 64 * np.finfo(np.float64).eps * 7 * scale


def test_host_curve_jet_plan_matches_linear_closed_form() -> None:
    curve = BSplineCurve(((0, 0, 0), (1, 0, 1)), (1, 1), (0, 0, 1, 1), 1)
    parameters = np.asarray((0.25, 0.5, 0.75), dtype=np.float64)
    value, derivative = curve_value_and_derivative_host(curve, parameters)
    np.testing.assert_array_equal(curve_evaluate_host(curve, parameters), value)
    np.testing.assert_array_equal(
        value, np.stack((parameters, np.zeros(3), parameters), axis=1)
    )
    np.testing.assert_array_equal(derivative, np.broadcast_to((1, 0, 1), (3, 3)))


def test_host_surface_jet_plan_matches_bilinear_closed_form() -> None:
    parameters = np.asarray(((0.25, 0.5),), dtype=np.float64)
    value, differential = surface_value_and_differential_host(
        _bilinear_patch(), parameters
    )
    np.testing.assert_array_equal(value, ((0.25, 0.5, 0.125),))
    np.testing.assert_array_equal(differential, (((1, 0), (0, 1), (0.5, 0.25)),))


def test_native_curve_piece_point_query_retains_both_closed_source_spans() -> None:
    curve = BSplineCurve(
        ((0, 0, 0), (1, 0, 0), (1, 1, 0)), (1, 1, 1), (0, 0, 0.5, 1, 1), 1
    )
    with coordinate_source_request((curve, 0.5, 0.5)) as budget:
        pieces = curve.source_pieces(0.5, 0.5, coordinate_budget=budget)
    assert tuple(piece.index for piece in pieces) == (0, 1)
    assert tuple((piece.lower, piece.upper) for piece in pieces) == ((0.5, 0.5),) * 2
    for piece in pieces:
        assert isinstance(piece.evaluator, BernsteinCurvePiece)
        np.testing.assert_array_equal(
            piece.evaluator.evaluate(jnp.asarray(0.5)), (1, 0, 0)
        )


def test_native_surface_piece_query_retains_a_degenerate_isoline_box() -> None:
    surface = _bilinear_patch()
    box = np.asarray(((0.5, 0.25), (0.5, 0.75)), dtype=np.float64)
    with coordinate_source_request((surface, box)) as budget:
        pieces = surface.source_pieces(box, budget=budget)
    assert len(pieces) == 1
    piece = pieces[0]
    assert isinstance(piece.evaluator, BernsteinSurfacePiece)
    assert piece.index == 0
    np.testing.assert_array_equal(piece.lower, box[0])
    np.testing.assert_array_equal(piece.upper, box[1])
    np.testing.assert_array_equal(
        piece.evaluator.evaluate(jnp.asarray((0.5, 0.5))), (0.5, 0.5, 0.25)
    )


def test_coefficient_enclosures_merge_distinct_constructions_of_one_constant() -> None:
    controls = np.asarray(((0, 0, 0, 1), (1, 0, 0, 1)), dtype=np.float64)
    first_lower = np.nextafter(controls, -np.inf)
    second_upper = np.nextafter(np.nextafter(controls, np.inf), np.inf)
    first = BernsteinCurvePiece(
        controls, 0, 1, controls_lower=first_lower, controls_upper=controls
    )
    second = BernsteinCurvePiece(
        controls, 0, 1, controls_lower=controls, controls_upper=second_upper
    )
    bounds = coefficient_enclosures(RuledSurface(first, second))
    assert len(bounds) == 1
    np.testing.assert_array_equal(bounds[0][0], controls)
    np.testing.assert_array_equal(bounds[0][1], first_lower)
    np.testing.assert_array_equal(bounds[0][2], second_upper)


def test_native_period_proofs_survive_sphere_offset_composition() -> None:
    sphere = _sphere()
    assert sphere.native_period_symbols() == ("two_pi", None)
    assert OffsetSurface(sphere, 0.25).native_period_symbols() == ("two_pi", None)


class _RootPoint(CurveCoordinateResult):
    point: Array
    residual_norm: Array


class _SquareRootCurve(StrictModule, RootCurveCoordinateSource):
    plan: VectorLocalRootPlan

    def evaluate(self, parameters: Array, /) -> _RootPoint:
        def residual(value: Array) -> Array:
            return value**2 - (parameters + 2.0)

        value, diagnostics = self.plan.solve_with_diagnostics(
            residual, jnp.ones((1,), dtype=jnp.float64)
        )
        value = eqx.error_if(
            value,
            ~diagnostics.converged,
            "test_curve.root must converge; got native root nonconvergence.",
        )
        return _RootPoint(
            point=jnp.stack((value[0], parameters, jnp.ones_like(parameters))),
            residual_norm=diagnostics.residual_norm,
        )


def test_host_coordinates_execute_native_root_source_and_preserve_implicit_jet() -> None:
    tolerance = 1.0e-13
    curve = _SquareRootCurve(
        plan=VectorLocalRootPlan(
            1,
            maximum_steps=16,
            tolerance=tolerance,
            plan_id="brep-host-square-root",
        )
    )
    parameters = np.asarray((0.25, 0.5, 0.75), dtype=np.float64)
    prepared = prepare_curve_host(curve, with_derivative=True, batch_capacity=2)
    actual = prepared.evaluate_host_points(parameters[:, None])
    radius = np.sqrt(parameters + 2.0)
    expected_value = np.stack((radius, parameters, np.ones_like(parameters)), axis=-1)
    expected_jet = np.stack(
        (0.5 / radius, np.ones_like(parameters), np.zeros_like(parameters)), axis=-1
    )
    # |r-sqrt(a)| = |r²-a|/(r+sqrt(a)); this positive Newton branch stays r>=1.
    # The implicit derivative 1/(2r) has Lipschitz constant <= 1/2 there.
    # The root residual bound plus eight elementary-operation ulps encloses both.
    bound = tolerance / 2 + 8 * np.finfo(np.float64).eps
    np.testing.assert_allclose(actual[..., 0], expected_value, rtol=0, atol=bound)
    np.testing.assert_allclose(actual[..., 1], expected_jet, rtol=0, atol=bound)
    np.testing.assert_allclose(
        curve_evaluate_host(curve, parameters), expected_value, rtol=0, atol=bound
    )


def test_host_native_root_source_propagates_its_convergence_refusal() -> None:
    curve = _SquareRootCurve(
        plan=VectorLocalRootPlan(
            1,
            maximum_steps=1,
            tolerance=1.0e-13,
            plan_id="brep-host-square-root-refusal",
        )
    )
    with pytest.raises(RuntimeError, match="test_curve.root must converge"):
        curve_evaluate_host(curve, np.asarray((0.25,), dtype=np.float64))


def test_host_native_root_chunks_match_rowwise_execution() -> None:
    curve = _SquareRootCurve(
        plan=VectorLocalRootPlan(
            1,
            maximum_steps=16,
            tolerance=1.0e-13,
            plan_id="brep-host-square-root-chunks",
        )
    )
    parameters = np.linspace(0.0, 1.0, 131, dtype=np.float64)[:, None]
    prepared = prepare_curve_host(curve, with_derivative=True, batch_capacity=131)
    batched = prepared.evaluate_host_points(parameters)
    assert batched.shape == (131, 3, 2)
    rowwise = np.concatenate(
        [prepared.evaluate_host_points(parameters[row : row + 1]) for row in range(131)]
    )
    np.testing.assert_array_equal(batched, rowwise)


def test_host_line_coordinates_preserve_closed_endpoint_values() -> None:
    curve = LineCurve((0, 0, 0), (1, 2, 3))
    parameters = np.asarray((-0.5, 0, 1, 2), dtype=np.float64)
    actual = curve_evaluate_host(curve, parameters)
    np.testing.assert_array_equal(actual, parameters[:, None] * np.asarray((1, 2, 3)))


def test_host_spline_preparation_uses_the_authored_nonunit_parameter_domain() -> None:
    curve = BSplineCurve(((0, 0, 0), (1, 0, 1)), (1, 1), (2, 2, 3, 3), 1)
    parameters = np.asarray((2, 2.25, 2.75, 3), dtype=np.float64)
    local = parameters - 2
    expected = np.stack((local, np.zeros_like(local), local), axis=-1)
    prepared = prepare_curve_host(curve, batch_capacity=2)
    np.testing.assert_array_equal(
        prepared.evaluate_host_points(parameters[:, None]), expected
    )
    np.testing.assert_array_equal(curve_evaluate_host(curve, parameters), expected)


def test_spline_surface_source_pieces_refuse_unadmitted_preparation_work() -> None:
    surface = _bilinear_patch()
    box = np.asarray(((0.25, 0.25), (0.75, 0.75)), dtype=np.float64)
    ticket = HostBatchTicket(0, bytes_upper=16_000_000)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=0, maximum_memory_bytes=16_000_000
    )
    with ticket, budget.activate(), budget.measurement():
        with pytest.raises(CoordinateEnclosureResourceError):
            surface.source_pieces(box, budget=budget)


@pytest.mark.parametrize("batch_shape", [(4,), (2, 2)])
def test_bernstein_patch_batches_equal_scalar_evaluations_bitwise(
    batch_shape: tuple[int, ...],
) -> None:
    controls = jnp.asarray(
        (
            ((0.125, 0.25, 0.5, 1.0), (0.75, 1.125, 0.25, 2.0)),
            ((1.25, 0.375, 0.625, 1.5), (2.25, 1.75, 1.125, 3.0)),
        ),
        dtype=jnp.float64,
    )
    patch = BernsteinSurfacePiece(
        controls,
        jnp.zeros(2, dtype=jnp.float64),
        jnp.ones(2, dtype=jnp.float64),
        controls_lower=controls,
        controls_upper=controls,
    )
    points = jnp.asarray(
        ((0.125, 0.25), (0.375, 0.875), (0.75, 0.125), (0.9375, 0.625)), dtype=jnp.float64
    )
    expected = jnp.stack([patch.evaluate(point) for point in points]).reshape(
        (*batch_shape, 3)
    )
    result = patch.evaluate(points.reshape((*batch_shape, 2)))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("count", (1, 3, 64, 65))
def test_source_bucket_host_admission_preserves_query_and_padding_bits(
    count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries = np.arange(count * 2, dtype=np.float64).reshape((count, 2))
    queries[0, 0] = -0.0
    capacity = max(64, 1 << (count - 1).bit_length())
    expected = np.concatenate(
        (queries, np.broadcast_to(queries[0], (capacity - count, 2)))
    )

    with monkeypatch.context() as host_only:
        host_only.setattr(jnp, "asarray", None)
        actual = np.asarray(bucketed(queries))

    assert actual.shape == expected.shape
    assert actual.dtype == np.dtype(np.float64)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("ambient", (2, 3))
def test_line_curve_host_admission_preserves_coefficient_bits(
    ambient: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = np.asarray((-0.0, 0.25, -0.125), dtype=np.float64)[:ambient]
    direction = np.asarray((1.0, -0.0, 0.5), dtype=np.float64)[:ambient]
    with monkeypatch.context() as host_only:
        host_only.setattr(jnp, "asarray", None)
        curve = LineCurve(origin, direction)

    assert np.asarray(curve.origin).dtype == origin.dtype
    assert np.asarray(curve.direction).dtype == direction.dtype
    assert np.asarray(curve.origin).tobytes() == origin.tobytes()
    assert np.asarray(curve.direction).tobytes() == direction.tobytes()


def test_bernstein_surface_host_admission_preserves_bank_bits_and_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = np.asarray(
        (
            ((-0.0, 0.25, 0.5, 1.0), (0.75, 1.125, 0.25, 2.0)),
            ((1.25, 0.375, 0.625, 1.5), (2.25, 1.75, 1.125, 3.0)),
        ),
        dtype=np.float64,
    )
    lower = np.asarray(((-0.0, 0.125),), dtype=np.float64)
    upper = np.asarray(((0.75,), (0.875,)), dtype=np.float64)
    controls_lower = np.nextafter(controls, -np.inf)
    controls_upper = np.nextafter(controls, np.inf)
    with monkeypatch.context() as host_only:
        host_only.setattr(jnp, "asarray", None)
        patch = BernsteinSurfacePiece(
            controls,
            lower,
            upper,
            controls_lower=controls_lower,
            controls_upper=controls_upper,
        )

    for actual, expected in (
        (patch.controls, controls),
        (patch.lower, lower.reshape((2,))),
        (patch.upper, upper.reshape((2,))),
        (patch.controls_lower, controls_lower),
        (patch.controls_upper, controls_upper),
    ):
        host = np.asarray(actual)
        assert host.dtype == expected.dtype
        assert host.shape == expected.shape
        assert host.tobytes() == expected.tobytes()


def test_bernstein_surface_constructor_keeps_traced_coefficients_differentiable() -> None:
    controls = jax.device_put(np.asarray((((0.25, 0.5, 0.75, 1.0),),), dtype=np.float64))
    lower = jax.device_put(np.zeros(2, dtype=np.float64))
    upper = jax.device_put(np.ones(2, dtype=np.float64))
    parameter = jax.device_put(np.asarray((0.25, 0.75), dtype=np.float64))

    def evaluate(coefficients: Array) -> Array:
        patch = BernsteinSurfacePiece(
            coefficients,
            lower,
            upper,
            controls_lower=coefficients,
            controls_upper=coefficients,
        )
        return patch.evaluate(parameter)

    tangent = jax.device_put(np.asarray((((1.0, 2.0, 3.0, 0.0),),), dtype=np.float64))

    def value_and_derivative(
        coefficients: Array, perturbation: Array
    ) -> tuple[Array, Array]:
        return jax.jvp(evaluate, (coefficients,), (perturbation,))

    value, derivative = jax.jit(value_and_derivative)(controls, tangent)
    np.testing.assert_array_equal(value, (0.25, 0.5, 0.75))
    np.testing.assert_array_equal(derivative, (1.0, 2.0, 3.0))
