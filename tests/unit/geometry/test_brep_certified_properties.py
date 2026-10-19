from ast import literal_eval
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from fractions import Fraction
from hashlib import sha256
from itertools import permutations, product
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ellipe

import phydrax.geometry.brep._measure as measure_module
from phydrax import config as pxconfig
from phydrax._physical import SpatialCoordinateContract
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization.particle._rigid_body import RigidBodySetPlan
from phydrax.geometry._interval_enclosure import interval_add, prepare_interval_function
from phydrax.geometry.brep._boolean import boolean_brep
from phydrax.geometry.brep._constructors import (
    assemble_brep_model,
    brep_box,
    brep_extrusion,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._edge_classification import brep_edge_classification
from phydrax.geometry.brep._fillet import brep_fillet
from phydrax.geometry.brep._intersection import NativePeriodEndpoint
from phydrax.geometry.brep._intersection_curve import AffinePCurve
from phydrax.geometry.brep._interval_derivatives import (
    _program_has_arrays,
    prepare_derivative_interval,
    prepare_taylor_derivative_interval,
)
from phydrax.geometry.brep._measure import (
    brep_mass_properties,
    brep_signed_shell_volume_enclosures,
)
from phydrax.geometry.brep._model import (
    BRepEntityId,
    BRepGeometry,
    BRepModel,
    BRepTopology,
)
from phydrax.geometry.brep._patches import (
    AbstractSurfacePatch,
    BernsteinSurfacePiece,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    coefficient_enclosures,
    CylinderPatch,
    EllipseCurve,
    LineCurve,
    PlanePatch,
)
from phydrax.meshing.qualification import trimmed_polynomial_nurbs_box
from phydrax.units._catalog import KILOGRAM, METER
from phydrax.units._unit import derived_unit
from tests._support.cad_models import native_solid


@contextmanager
def _budget() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100_000_000, bytes_upper=1_000_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000_000, maximum_memory_bytes=1_000_000_000
        )
        with budget.activate():
            yield budget


def _contains(bounds: np.ndarray, reference: np.ndarray) -> None:
    assert np.all(bounds[0] <= reference)
    assert np.all(reference <= bounds[1])


@pytest.mark.parametrize(
    "name,volume,inertia",
    [
        ("box", 6.0, np.diag((6 * 13 / 12, 6 * 10 / 12, 6 * 5 / 12))),
        ("cylinder", 2 * np.pi, np.diag((2 * np.pi * 7 / 12, 2 * np.pi * 7 / 12, np.pi))),
        ("sphere", 4 * np.pi / 3, np.eye(3) * 8 * np.pi / 15),
        (
            "torus",
            np.pi**2,
            np.diag(
                (
                    np.pi**2 * (2 + 5 / 32),
                    np.pi**2 * (2 + 5 / 32),
                    np.pi**2 * (4 + 3 / 16),
                )
            ),
        ),
    ],
)
def test_mass_closed_form_enclosures(
    name: str, volume: float, inertia: np.ndarray
) -> None:
    model = native_solid(name, catalog="geometry")
    geometry = model.geometry
    assert geometry is not None
    with _budget() as budget:
        result = brep_mass_properties(
            model, geometry, subdivisions=12, coordinate_budget=budget
        )
    _contains(np.asarray(result.volume_enclosures)[0], np.asarray(volume))
    _contains(np.asarray(result.inertia_enclosures)[0], inertia)
    assert result.model_id == model.model_id


def test_box_edge_convexity_and_lengths() -> None:
    model = native_solid("box", catalog="geometry")
    geometry = model.geometry
    assert geometry is not None
    with _budget() as budget:
        result = brep_edge_classification(
            model, geometry, maximum_cells=4, coordinate_budget=budget
        )
    assert tuple(record.edge_id for record in result) == model.edge_ids
    for edge, record in enumerate(result):
        assert record.status == "convex"
        vertices = np.asarray(geometry.vertex_points)[
            np.asarray(geometry.edge_vertices[edge])
        ]
        _contains(
            np.asarray(record.length_enclosure),
            np.asarray(np.linalg.norm(vertices[1] - vertices[0])),
        )


def test_insufficient_mass_cells_refuse_zero_volume() -> None:
    model = native_solid("cone", catalog="geometry")
    geometry = model.geometry
    assert geometry is not None
    with (
        _budget() as budget,
        pytest.raises(ValueError, match="volume enclosure containing zero"),
    ):
        brep_mass_properties(model, geometry, subdivisions=1, coordinate_budget=budget)


@pytest.mark.parametrize("removed_face", range(6))
def test_mass_refuses_box_missing_face_with_three_axis_closure_evidence(
    removed_face: int,
) -> None:
    source = native_solid("box", catalog="geometry")
    original = source.require_geometry()
    faces = tuple(face for face in range(6) if face != removed_face)
    coedges = tuple(
        coedge for face in faces for loop in original.face_loops[face] for coedge in loop
    )
    coedge_indices = {coedge: index for index, coedge in enumerate(coedges)}
    geometry = BRepGeometry(
        vertex_points=original.vertex_points,
        curves=original.curves,
        edge_curves=original.edge_curves,
        edge_ranges=original.edge_ranges,
        edge_vertices=original.edge_vertices,
        pcurves=tuple(original.pcurves[coedge] for coedge in coedges),
        coedge_edges=tuple(original.coedge_edges[coedge] for coedge in coedges),
        coedge_senses=tuple(original.coedge_senses[coedge] for coedge in coedges),
        face_loops=tuple(
            tuple(
                tuple(coedge_indices[coedge] for coedge in loop)
                for loop in original.face_loops[face]
            )
            for face in faces
        ),
        shell_faces=(tuple(range(5)),),
        shell_orientations=(
            tuple(
                sign
                for face, sign in zip(
                    original.shell_faces[0], original.shell_orientations[0], strict=True
                )
                if face != removed_face
            ),
        ),
        solid_shells=(),
        vertex_roots=original.vertex_roots,
        edge_endpoint_roots=original.edge_endpoint_roots,
        coedge_endpoint_roots=tuple(
            original.coedge_endpoint_roots[coedge] for coedge in coedges
        ),
        occurrences=(),
    )
    model = _publish(
        geometry,
        tuple(source.patches[face] for face in faces),
        np.asarray(source.parameter_bounds)[np.asarray(faces)],
        f"box-missing-face-{removed_face}",
        np.asarray(source.orientation)[np.asarray(faces)],
    )
    # Publication admits this honest open shell, not a closed solid. Corrupt its
    # solid incidence through a functional update to exercise the mass query's
    # independent closure refusal without relaxing assembly validation.
    assert model.topology.solid_faces == ()
    topology = model.topology
    adversarial_topology = BRepTopology(
        face_edges=topology.face_edges,
        edge_faces=topology.edge_faces,
        face_wires=topology.face_wires,
        solid_faces=(tuple(range(5)),),
        solid_face_orientations=geometry.shell_orientations,
        num_vertices=topology.num_vertices,
    )
    model = eqx.tree_at(lambda value: value.topology, model, adversarial_topology)
    geometry = model.require_geometry()
    assert len(model.topology.solid_faces[0]) == 5
    with _budget() as budget, pytest.raises(ValueError, match="closure") as refusal:
        brep_mass_properties(model, geometry, subdivisions=8, coordinate_budget=budget)
    message, separator, evidence = str(refusal.value).partition("axis enclosures=")
    assert separator, message
    bounds = np.asarray(literal_eval(evidence), dtype=np.float64)
    assert bounds.shape == (3, 2)
    assert np.all(np.isfinite(bounds))
    assert np.all(bounds[:, 0] <= bounds[:, 1])
    assert np.max(bounds[:, 0]) > np.min(bounds[:, 1])


def _publish(
    geometry: BRepGeometry,
    patches: tuple[AbstractSurfacePatch, ...],
    bounds: np.ndarray,
    name: str,
    orientation: np.ndarray | None = None,
) -> BRepModel:
    return assemble_brep_model(
        geometry,
        patches,
        bounds,
        np.ones(len(patches)) if orientation is None else orientation,
        ("face",) * len(patches),
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id=name,
        source_format="native",
        source_digest=sha256(name.encode()).hexdigest(),
        import_policy_id=sha256(b"certified-property-test").hexdigest(),
    )


def test_ellipse_arc_length_enclosure() -> None:
    curve = EllipseCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 2.0, 1.0)
    pcurve = EllipseCurve((0, 0), (1, 0), (0, 1), 2.0, 1.0)
    edge_roots = (
        NativePeriodEndpoint(curve, turns=0),
        NativePeriodEndpoint(curve, turns=Fraction(1, 4)),
    )
    pcurve_roots = (
        NativePeriodEndpoint(pcurve, turns=0),
        NativePeriodEndpoint(pcurve, turns=Fraction(1, 4)),
    )
    geometry = BRepGeometry(
        vertex_points=np.asarray(((2, 0, 0), (0, 1, 0), (0, 0, 0)), dtype=np.float64),
        curves=(curve, LineCurve((0, 1, 0), (0, -1, 0)), LineCurve((0, 0, 0), (2, 0, 0))),
        edge_curves=(0, 1, 2),
        edge_ranges=np.asarray(((0, np.pi / 2), (0, 1), (0, 1))),
        edge_vertices=((0, 1), (1, 2), (2, 0)),
        pcurves=(pcurve, LineCurve((0, 1), (0, -1)), LineCurve((0, 0), (2, 0))),
        coedge_edges=(0, 1, 2),
        coedge_senses=(1, 1, 1),
        face_loops=(((0, 1, 2),),),
        shell_faces=((0,),),
        shell_orientations=((1,),),
        solid_shells=(),
        edge_endpoint_roots=(edge_roots, (None, None), (None, None)),
        coedge_endpoint_roots=(pcurve_roots, (None, None), (None, None)),
        occurrences=(),
    )
    model = _publish(
        geometry,
        (PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)),),
        np.asarray((((0, 0), (2, 1)),)),
        "ellipse-sector",
    )
    with _budget() as budget:
        result = brep_edge_classification(
            model, geometry, maximum_cells=32, coordinate_budget=budget
        )
    _contains(
        np.asarray(result[0].length_enclosure), np.asarray(2 * ellipe(np.float64(0.75)))
    )
    assert result[0].status == "unresolved"  # A lamina has no material-side dihedral.


def test_circular_lengths_and_pocket_edges() -> None:
    cylinder = native_solid("cylinder", catalog="geometry")
    cylinder_geometry = cylinder.geometry
    assert cylinder_geometry is not None
    with _budget() as budget:
        edges = brep_edge_classification(
            cylinder, cylinder_geometry, maximum_cells=16, coordinate_budget=budget
        )
    circles = [
        record
        for edge, record in enumerate(edges)
        if isinstance(
            cylinder_geometry.curves[cylinder_geometry.edge_curves[edge]], CircleCurve
        )
    ]
    assert len(circles) == 2
    for record in circles:
        _contains(np.asarray(record.length_enclosure), np.asarray(2 * np.pi))
    profile = PlanarProfile(
        ProfilePlane(),
        ProfileLoop.polygon(((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2))),
    )
    pocket = brep_extrusion(
        profile, (0, 0, 1), coordinate_contract=SpatialCoordinateContract.si()
    )
    geometry = pocket.geometry
    assert geometry is not None
    with _budget() as budget:
        edges = brep_edge_classification(
            pocket, geometry, maximum_cells=4, coordinate_budget=budget
        )
    vertical = [
        record
        for edge, record in enumerate(edges)
        if np.all(
            np.asarray(geometry.vertex_points)[
                np.asarray(geometry.edge_vertices[edge]), :2
            ]
            == (1, 1)
        )
    ]
    assert len(vertical) == 1
    assert vertical[0].status == "concave"


def test_sign_change_parameter_enclosed() -> None:
    points = np.asarray(
        ((-1, -1, 0), (1, -1, 0), (1, 0, 0), (-1, 0, 0), (1, 1, 1), (-1, 1, -1)),
        dtype=np.float64,
    )
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 5), (5, 3))
    curves = tuple(LineCurve(points[a], points[b] - points[a]) for a, b in pairs)
    use_edges = (0, 1, 2, 3, 2, 4, 5, 6)
    pcurves = tuple(
        LineCurve(
            points[pairs[edge][0], :2],
            points[pairs[edge][1], :2] - points[pairs[edge][0], :2],
        )
        for edge in use_edges
    )
    geometry = BRepGeometry(
        vertex_points=points,
        curves=curves,
        edge_curves=tuple(range(7)),
        edge_ranges=np.asarray(((0, 1),) * 7),
        edge_vertices=pairs,
        pcurves=pcurves,
        coedge_edges=use_edges,
        coedge_senses=(1, 1, 1, 1, -1, 1, 1, 1),
        face_loops=(((0, 1, 2, 3),), ((4, 5, 6, 7),)),
        shell_faces=((0, 1),),
        shell_orientations=((1, 1),),
        solid_shells=(),
        occurrences=(),
    )
    controls = np.asarray(
        (((-1, 0, 0), (-1, 1, -1)), ((1, 0, 0), (1, 1, 1))), dtype=np.float64
    )
    patch = BSplineSurfacePatch(
        controls,
        np.ones((2, 2)),
        np.asarray((-1, -1, 1, 1)),
        np.asarray((0, 0, 1, 1)),
        1,
        1,
    )
    model = _publish(
        geometry,
        (PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0)), patch),
        np.asarray((((-1, -1), (1, 0)), ((-1, 0), (1, 1)))),
        "sign-changing-hinge",
    )
    with _budget() as budget:
        result = brep_edge_classification(
            model, geometry, maximum_cells=16, coordinate_budget=budget
        )[2]
    assert result.status == "sign_change"
    changes = np.asarray(result.change_parameters)
    assert np.any((changes[:, 0] <= 0.5) & (changes[:, 1] >= 0.5))


def test_trimmed_nurbs_solid_contains_exact_box_moments() -> None:
    model = trimmed_polynomial_nurbs_box()
    geometry = model.require_geometry()
    with _budget() as budget:
        result = brep_mass_properties(
            model, geometry, subdivisions=8, coordinate_budget=budget
        )
    # A 200-cell midpoint volume cubature underestimates each variance by
    # size**2 / (12 * 200**2); it is not a certified point reference for tight bounds.
    # Use the independently known box moments, not wider production tolerances.
    sizes = np.asarray((1, 2, 3), dtype=np.float64)
    variances = sizes**2 / 12
    reference = 6 * np.diag(np.sum(variances) - variances)
    _contains(np.asarray(result.volume_enclosures)[0], np.asarray(6.0))
    _contains(np.asarray(result.inertia_enclosures)[0], reference)
    _contains(np.asarray(result.centroid_enclosures)[0], sizes / 2)


def test_trimmed_polynomial_boundary_moments_reduce_constant_cell_evaluations(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, int], None],
) -> None:
    model = trimmed_polynomial_nurbs_box()
    geometry = model.require_geometry()
    subdivisions = 8
    integrands = measure_module._mass_integrands
    calls = 0

    def counted_integrands(
        patch: AbstractSurfacePatch,
        box: np.ndarray,
        center: np.ndarray,
        orientation: float,
        budget: CoordinateEnclosureBudget,
        /,
        *,
        moment_set: Literal["volume", "mass_properties"],
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        nonlocal calls
        calls += 1
        return integrands(patch, box, center, orientation, budget, moment_set=moment_set)

    monkeypatch.setattr(measure_module, "_mass_integrands", counted_integrands)
    with _budget() as budget:
        boundary = brep_mass_properties(
            model, geometry, subdivisions=subdivisions, coordinate_budget=budget
        )
        boundary_work = budget.work_units
    boundary_calls = calls
    calls = 0
    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(
            measure_module,
            "_face_moment_enclosures",
            measure_module._face_cell_moment_enclosures,
        )
        with _budget() as budget:
            constant_cells = brep_mass_properties(
                model, geometry, subdivisions=subdivisions, coordinate_budget=budget
            )
            constant_cell_work = budget.work_units
    expected_cell_calls = subdivisions**2 * sum(
        len(loop) for loops in geometry.face_loops for loop in loops
    )
    assert calls == expected_cell_calls == 1536
    assert boundary_calls < calls
    record_property("constant_cell_integrand_evaluations", calls)
    record_property("boundary_integrand_evaluations", boundary_calls)
    record_property("constant_cell_work_units", constant_cell_work)
    record_property("boundary_work_units", boundary_work)
    for reduced, baseline in zip(
        (
            boundary.volume_enclosures,
            boundary.first_moment_enclosures,
            boundary.second_moment_enclosures,
            boundary.centroid_enclosures,
            boundary.inertia_enclosures,
        ),
        (
            constant_cells.volume_enclosures,
            constant_cells.first_moment_enclosures,
            constant_cells.second_moment_enclosures,
            constant_cells.centroid_enclosures,
            constant_cells.inertia_enclosures,
        ),
        strict=True,
    ):
        reduced_bounds, baseline_bounds = np.asarray(reduced), np.asarray(baseline)
        assert np.all(
            reduced_bounds[:, 1] - reduced_bounds[:, 0]
            <= baseline_bounds[:, 1] - baseline_bounds[:, 0]
        )
    reference = np.diag((6 * 13 / 12, 6 * 10 / 12, 6 * 5 / 12))
    _contains(np.asarray(boundary.inertia_enclosures)[0], reference)
    _contains(np.asarray(constant_cells.inertia_enclosures)[0], reference)


def _generic_green_moment_fixture(
    kind: Literal["polynomial", "rational"],
) -> tuple[measure_module._GreenMomentDensity, np.ndarray]:
    match kind:
        case "polynomial":
            # Bernstein controls represent exactly S(u,v) = (u,v,u**2).
            controls = np.asarray(
                (
                    ((0, 0, 0), (0, 1, 0)),
                    ((0.5, 0, 0), (0.5, 1, 0)),
                    ((1, 0, 1), (1, 1, 1)),
                ),
                dtype=np.float64,
            )
            weights = np.ones((3, 2), dtype=np.float64)
            knots = np.asarray((0, 0, 0, 1, 1, 1), dtype=np.float64)
            u_degree, polynomial_degree = 2, 5
            # x dot n = -u**2, x_i*n_i = (-2*u**2, 0, u**2).
            reference = (Fraction(-1, 9), Fraction(-2, 3), Fraction(0), Fraction(1, 3))
        case "rational":
            # Dyadic controls give S(u,v) = (u/(1+u),v,1).
            controls = np.asarray(
                (((0, 0, 1), (0, 1, 1)), ((0.5, 0, 1), (0.5, 1, 1))),
                dtype=np.float64,
            )
            weights = np.asarray(((1, 1), (2, 2)), dtype=np.float64)
            knots = np.asarray((0, 0, 1, 1), dtype=np.float64)
            u_degree, polynomial_degree = 1, -1
            # n_z = 1/(1+u)**2; its unit-rectangle integral is exactly 1/2.
            reference = (Fraction(1, 6), Fraction(0), Fraction(0), Fraction(1, 2))
    patch = BSplineSurfacePatch(
        controls,
        weights,
        knots,
        np.asarray((0, 0, 1, 1), dtype=np.float64),
        u_degree,
        1,
    )
    pieces = patch.bezier_pieces()
    assert len(pieces) == 1
    piece = pieces[0]
    bounds = np.asarray(piece.parameter_bounds, dtype=np.float64).T
    source = BernsteinSurfacePiece(
        piece.homogeneous_controls,
        bounds[0],
        bounds[1],
        controls_lower=piece.homogeneous_lower,
        controls_upper=piece.homogeneous_upper,
    )
    density = measure_module._MomentDensity(
        source,
        jnp.zeros(3, dtype=jnp.float64),
        jnp.asarray(1, dtype=jnp.float64),
        "volume",
    )
    evaluator = measure_module._GreenMomentDensity(
        density,
        LineCurve((1, 0), (0, 1)),
        jnp.asarray(0, dtype=jnp.float64),
        jnp.asarray(1, dtype=jnp.float64),
        jnp.asarray(1, dtype=jnp.float64),
        True,
        polynomial_degree,
    )
    return evaluator, np.asarray(
        tuple(float(value) for value in reference), dtype=np.float64
    )


@pytest.mark.parametrize("kind", ("polynomial", "rational"))
def test_generic_green_gauss_moments_contain_exact_chart_integrals(
    kind: Literal["polynomial", "rational"],
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, int], None],
) -> None:
    evaluator, reference = _generic_green_moment_fixture(kind)
    calls = 0
    evaluate = measure_module._GaussGreenRule.evaluate

    def counted_rule(
        rule: measure_module._GaussGreenRule,
        box: np.ndarray,
        budget: CoordinateEnclosureBudget,
        /,
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        nonlocal calls
        calls += 1
        return evaluate(rule, box, budget)

    monkeypatch.setattr(measure_module._GaussGreenRule, "evaluate", counted_rule)
    with _budget() as budget:
        rule = measure_module._prepare_gauss_green_rule(evaluator, budget)
        assert (rule.inner_derivative is None) == (kind == "polynomial")
        coarse = measure_module._refine_gauss_boundary(rule, 0, 1, 1, budget)
        assert calls == 1
        calls = 0
        refined = measure_module._refine_gauss_boundary(rule, 0, 1, 8, budget)
        assert calls == 15  # One initial piece, then two evaluations per split.
        record_property("gauss_rule_evaluations_one_piece", 1)
        record_property("gauss_rule_evaluations_eight_pieces", calls)
        record_property("gauss_rule_work_units", budget.work_units)
    assert np.all(np.isfinite(np.stack(coarse)))
    assert np.all(np.isfinite(np.stack(refined)))
    assert np.all(coarse[0] <= coarse[1])
    assert np.all(refined[0] <= refined[1])
    _contains(np.stack(coarse), reference)
    _contains(np.stack(refined), reference)
    if kind == "rational":
        assert np.max(refined[1] - refined[0]) < np.max(coarse[1] - coarse[0])


@pytest.mark.parametrize("order", (1, 2, 4))
def test_taylor_pure_derivative_of_polynomial_moment_density(order: int) -> None:
    evaluator, _ = _generic_green_moment_fixture("polynomial")
    density = evaluator.density
    parameter = np.asarray(((0.5, 0.25),), dtype=np.float64)
    derivative = prepare_taylor_derivative_interval(
        density,
        2,
        0,
        order,
        batch_capacity=1,
        constant_bounds=coefficient_enclosures(density),
    )
    lower, upper = derivative.evaluate(parameter, parameter)
    # Derivatives of (-u**2/3, -2*u**2, 0, u**2) at u=1/2.
    factor = 1 if order == 1 else 2 if order == 2 else 0
    reference = factor * np.asarray((-1 / 3, -2, 0, 1), dtype=np.float64)
    _contains(np.stack((lower[0], upper[0])), reference)


@pytest.mark.parametrize("mapped", (False, True))
def test_green_gauss_spline_trim_spans_contain_exact_rational_chart_integral(
    mapped: bool,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, int], None],
) -> None:
    evaluator, reference = _generic_green_moment_fixture("rational")
    controls = ((0, 0), (0, 0.25), (0, 0.5)) if mapped else ((1, 0), (1, 0.5), (1, 1))
    spline = BSplineCurve(
        np.asarray(controls, dtype=np.float64),
        np.ones(3, dtype=np.float64),
        np.asarray((0, 0, 0.5, 1, 1), dtype=np.float64),
        1,
    )
    pcurve = AffinePCurve(spline, ((1, 0), (0, 2)), (1, 0)) if mapped else spline
    # Both native trims are exactly (1,t), but their internal knot must split
    # the Taylor source before differentiating either restricted span.
    calls = 0
    evaluate = measure_module._GaussGreenRule.evaluate

    def counted_rule(
        rule: measure_module._GaussGreenRule,
        box: np.ndarray,
        budget: CoordinateEnclosureBudget,
        /,
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        nonlocal calls
        calls += 1
        return evaluate(rule, box, budget)

    monkeypatch.setattr(measure_module._GaussGreenRule, "evaluate", counted_rule)
    total = np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64)
    with _budget() as budget:
        cuts = np.unique(measure_module._trim_native_knots(pcurve, budget))
        np.testing.assert_array_equal(cuts, np.asarray((0, 0.5, 1), dtype=np.float64))
        assert measure_module._taylor_trim_source(pcurve, 0, 1, budget) is None
        for first, last in zip(cuts[:-1], cuts[1:], strict=True):
            source = measure_module._taylor_trim_source(
                pcurve, float(first), float(last), budget
            )
            assert source is not None
            span = measure_module._GreenMomentDensity(
                evaluator.density,
                source,
                evaluator.lower_u,
                evaluator.upper_u,
                evaluator.sense,
                evaluator.final_span,
                evaluator.polynomial_degree,
            )
            rule = measure_module._prepare_gauss_green_rule(span, budget)
            value = measure_module._refine_gauss_boundary(
                rule, float(first), float(last), 8, budget
            )
            assert np.all(np.isfinite(np.stack(value)))
            assert np.all(value[0] <= value[1])
            total = interval_add(total, value)
        assert calls == 30  # Two spans, each with one initial and seven splits.
        record_property("spline_trim_boundary_spans", 2)
        record_property("spline_trim_gauss_rule_evaluations", calls)
        record_property("spline_trim_work_units", budget.work_units)
    assert np.all(np.isfinite(np.stack(total)))
    assert np.all(total[0] <= total[1])
    _contains(np.stack(total), reference)


@pytest.mark.parametrize(
    "permutation,signs",
    [(p, s) for p in permutations((0, 1, 2)) for s in product((-1, 1), repeat=3)],
)
@pytest.mark.parametrize("exponent", (-20, 0, 20))
def test_exact_binary_transform_edge_and_mass_decisions(
    permutation: tuple[int, ...], signs: tuple[int, ...], exponent: int
) -> None:
    scale = np.ldexp(np.float64(1), exponent)
    sizes = np.asarray((1, 2, 3), dtype=np.float64)[np.asarray(permutation)]
    upper = scale * sizes * np.asarray(signs)
    model = brep_box(
        np.minimum(0, upper),
        np.maximum(0, upper),
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    geometry = model.geometry
    assert geometry is not None
    with _budget() as budget:
        edges = brep_edge_classification(
            model, geometry, maximum_cells=1, coordinate_budget=budget
        )
        mass = brep_mass_properties(
            model, geometry, subdivisions=2, coordinate_budget=budget
        )
    assert all(edge.status == "convex" for edge in edges)
    _contains(np.asarray(mass.volume_enclosures)[0], np.asarray(6 * scale**3))
    for edge, record in enumerate(edges):
        a, b = geometry.edge_vertices[edge]
        length = np.linalg.norm(
            np.asarray(geometry.vertex_points)[a] - np.asarray(geometry.vertex_points)[b]
        )
        _contains(np.asarray(record.length_enclosure), np.asarray(length))


def test_explicit_density_units_and_rigid_body_consumer() -> None:
    model = native_solid("box", catalog="geometry")
    geometry = model.geometry
    assert geometry is not None
    density_unit = derived_unit("kg/m^3", ((KILOGRAM, 1), (METER, -3)))
    with _budget() as budget:
        result = brep_mass_properties(
            model,
            geometry,
            subdivisions=8,
            coordinate_budget=budget,
            density=2.0,
            density_unit=density_unit,
            coordinate_unit=METER,
        )
    assert result.mass_enclosures is not None
    assert result.mass_inertia is not None
    _contains(np.asarray(result.mass_enclosures)[0], np.asarray(12.0))
    body = RigidBodySetPlan(np.asarray((0,), dtype=np.int32), result.mass_inertia)
    assert body.inertia_com.shape == (1, 3, 3)
    with _budget() as budget, pytest.raises(ValueError, match="density_units"):
        brep_mass_properties(
            model,
            geometry,
            subdivisions=1,
            coordinate_budget=budget,
            density=2.0,
            density_unit=METER,
            coordinate_unit=METER,
        )


def test_fillet_boundary_angle_matches_g1_certificate() -> None:
    box = native_solid("box", catalog="geometry")
    source = box.require_geometry()
    edge = next(
        index
        for index, (a, b) in enumerate(source.edge_vertices)
        if np.all(np.asarray(source.vertex_points[b] - source.vertex_points[a])[:2] == 0)
    )
    with _budget() as budget:
        blend = brep_fillet(
            box,
            (box.edge_ids[edge],),
            0.125,
            maximum_cells=1,
            tessellation=BRepTessellationPolicy(realize=False),
            budget=budget,
        )
        assert blend.status == "certified", blend.reason
        model = blend.model
        assert model is not None
        geometry = model.require_geometry()
        records = brep_edge_classification(
            model, geometry, maximum_cells=1, coordinate_budget=budget
        )
    cylinder_face = next(
        face
        for face, patch in enumerate(model.patches)
        if isinstance(patch, CylinderPatch)
    )
    cylinder = model.patches[cylinder_face]
    assert isinstance(cylinder, CylinderPatch)
    boundary = sorted(
        edge
        for edge in model.topology.face_edges[cylinder_face]
        if isinstance(geometry.curves[geometry.edge_curves[edge]], LineCurve)
    )
    assert len(boundary) == 2
    for row, index in enumerate(boundary):
        curve = geometry.curves[geometry.edge_curves[index]]
        assert isinstance(curve, LineCurve)
        other = next(
            face for face in model.topology.edge_faces[index] if face != cylinder_face
        )
        plane = model.patches[other]
        assert isinstance(plane, PlanePatch)
        axis = np.asarray(cylinder.axis)
        delta = np.asarray(curve.origin - cylinder.origin)
        radial = delta - axis * (axis @ delta)
        normal = np.cross(np.asarray(plane.first_axis), np.asarray(plane.second_axis))
        # Dyadic source coefficients prove tangency along the entire line.
        assert np.all(np.cross(np.asarray(curve.direction), axis) == 0)
        assert np.all(np.cross(radial, normal) == 0)
        assert radial @ radial == float(cylinder.radius) ** 2
        angle = np.asarray(records[index].dihedral_range)
        _contains(angle, np.asarray(0.0))
        np.testing.assert_array_equal(angle, np.asarray(blend.g1_angle_bounds)[row])


@pytest.mark.parametrize("capability", ("edge", "mass"))
def test_capability_work_refuses_before_unadmitted_evaluation(capability: str) -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError

    model = native_solid("box", catalog="geometry")
    geometry = model.require_geometry()
    with HostBatchTicket(0, bytes_upper=0) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=0, maximum_memory_bytes=0
        )
        with budget.activate(), pytest.raises(CoordinateEnclosureResourceError):
            if capability == "edge":
                brep_edge_classification(
                    model, geometry, maximum_cells=1, coordinate_budget=budget
                )
            else:
                brep_mass_properties(
                    model, geometry, subdivisions=1, coordinate_budget=budget
                )


@pytest.mark.parametrize("capability", ("edge", "mass"))
def test_certified_records_refuse_binary32_publication(capability: str) -> None:
    model = native_solid("box", catalog="geometry")
    geometry = model.require_geometry()
    with (
        _budget() as budget,
        pxconfig.context(pxconfig.NumericsConfig(enable_x64=False)),
        pytest.raises(ValueError, match="precision"),
    ):
        if capability == "edge":
            brep_edge_classification(
                model, geometry, maximum_cells=1, coordinate_budget=budget
            )
        else:
            brep_mass_properties(
                model, geometry, subdivisions=1, coordinate_budget=budget
            )


def test_default_orientation_audit_uses_reduced_signed_moments(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    model = native_solid("box", catalog="geometry")
    geometry = model.require_geometry()
    integrands = measure_module._mass_integrands
    calls = 0

    def counted_integrands(
        patch: AbstractSurfacePatch,
        box: np.ndarray,
        center: np.ndarray,
        orientation: float,
        budget: CoordinateEnclosureBudget,
        /,
        *,
        moment_set: Literal["volume", "mass_properties"],
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        nonlocal calls
        calls += 1
        return integrands(patch, box, center, orientation, budget, moment_set=moment_set)

    monkeypatch.setattr(measure_module, "_mass_integrands", counted_integrands)
    with _budget() as budget:
        reduced = brep_signed_shell_volume_enclosures(
            model, geometry, subdivisions=1, coordinate_budget=budget
        )
        reduced_work = budget.work_units
    assert calls == 0
    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(
            measure_module,
            "_face_moment_enclosures",
            measure_module._face_cell_moment_enclosures,
        )
        with _budget() as budget:
            baseline = brep_signed_shell_volume_enclosures(
                model, geometry, subdivisions=1, coordinate_budget=budget
            )
            baseline_work = budget.work_units
    assert calls == 24
    assert reduced.model_id == baseline.model_id
    assert reduced.shell_ids == baseline.shell_ids
    assert reduced_work < baseline_work
    _contains(np.asarray(reduced.volume_enclosures)[0], np.asarray(6.0))
    _contains(np.asarray(baseline.volume_enclosures)[0], np.asarray(6.0))
    record_property("signed_constant_cell_evaluations", calls)
    record_property("signed_boundary_evaluations", 0)
    record_property("signed_constant_cell_work_units", baseline_work)
    record_property("signed_boundary_work_units", reduced_work)
    record_property("model_id", reduced.model_id)


def test_signed_box_shell_volume_is_positive_and_source_keyed() -> None:
    model = native_solid("box", catalog="geometry")
    geometry = model.require_geometry()
    with _budget() as budget:
        result = brep_signed_shell_volume_enclosures(
            model, geometry, subdivisions=4, coordinate_budget=budget
        )
    assert result.model_id == model.model_id
    assert result.shell_ids == (BRepEntityId(model.source_revision, "shell", 0),)
    enclosure = np.asarray(result.volume_enclosures)[0]
    assert enclosure[0] > 0
    _contains(enclosure, np.asarray(6.0))


def test_reversed_box_shell_volume_enclosure_is_negated() -> None:
    source = native_solid("box", catalog="geometry")
    geometry = source.require_geometry()
    reversed_model = assemble_brep_model(
        geometry,
        source.patches,
        source.parameter_bounds,
        -source.orientation,
        source.physical_tags,
        coordinate_contract=source.coordinate_contract,
        source_id="inverted-shell-box",
        source_format="native",
        source_digest=sha256(b"inverted-shell-box").hexdigest(),
        import_policy_id=sha256(b"signed-shell-volume-test").hexdigest(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    with _budget() as budget:
        positive = brep_signed_shell_volume_enclosures(
            source, geometry, subdivisions=4, coordinate_budget=budget
        )
        negative = brep_signed_shell_volume_enclosures(
            reversed_model, geometry, subdivisions=4, coordinate_budget=budget
        )
    np.testing.assert_array_equal(
        np.asarray(negative.volume_enclosures),
        -np.asarray(positive.volume_enclosures)[:, ::-1],
    )
    assert np.asarray(negative.volume_enclosures)[0, 1] < 0


def test_box_cavity_shell_volumes_keep_material_side_signs() -> None:
    policy = BRepTessellationPolicy(realize=False)
    outer = brep_box(
        (0, 0, 0),
        (3, 3, 3),
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=policy,
    )
    inner = brep_box(
        (1, 1, 1),
        (2, 2, 2),
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=policy,
    )
    with _budget() as budget:
        model = boolean_brep(outer, inner, "difference", budget=budget).model
        geometry = model.require_geometry()
        result = brep_signed_shell_volume_enclosures(
            model, geometry, subdivisions=4, coordinate_budget=budget
        )
    assert result.shell_ids == tuple(
        BRepEntityId(model.source_revision, "shell", index)
        for index in range(len(geometry.shell_faces))
    )
    enclosures = np.asarray(result.volume_enclosures)
    assert enclosures.shape == (2, 2)
    positive = enclosures[enclosures[:, 0] > 0]
    negative = enclosures[enclosures[:, 1] < 0]
    assert positive.shape == negative.shape == (1, 2)
    _contains(positive[0], np.asarray(27.0))
    _contains(negative[0], np.asarray(-1.0))


def test_signed_shell_query_publishes_zero_containing_bounds() -> None:
    model = native_solid("cone", catalog="geometry")
    with _budget() as budget:
        result = brep_signed_shell_volume_enclosures(
            model, model.require_geometry(), subdivisions=1, coordinate_budget=budget
        )
    enclosure = np.asarray(result.volume_enclosures)[0]
    assert enclosure[0] <= 0 <= enclosure[1]


def test_signed_shell_query_keeps_precision_and_budget_refusals() -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError

    model = native_solid("box", catalog="geometry")
    geometry = model.require_geometry()
    with (
        _budget() as budget,
        pxconfig.context(pxconfig.NumericsConfig(enable_x64=False)),
        pytest.raises(ValueError, match="precision"),
    ):
        brep_signed_shell_volume_enclosures(
            model, geometry, subdivisions=1, coordinate_budget=budget
        )
    with HostBatchTicket(0, bytes_upper=0) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=0, maximum_memory_bytes=0
        )
        with budget.activate(), pytest.raises(CoordinateEnclosureResourceError):
            brep_signed_shell_volume_enclosures(
                model, geometry, subdivisions=1, coordinate_budget=budget
            )


def test_cached_derivative_program_rebinds_face_coefficients() -> None:
    boxes = np.asarray(((0.0, 0.0),), dtype=np.float64)
    first = PlanePatch((0, 0, 0), (1, 0, 0), (0, 1, 0))
    second = PlanePatch((4, 5, 6), (2, 0, 0), (0, 3, 0))
    one = prepare_derivative_interval(first, 2, 1, batch_capacity=1, constant_bounds=())
    two = prepare_derivative_interval(second, 2, 1, batch_capacity=1, constant_bounds=())
    again = prepare_derivative_interval(first, 2, 1, batch_capacity=1, constant_bounds=())
    for program, expected in (
        (one, np.asarray(((1, 0), (0, 1), (0, 0)))),
        (two, np.asarray(((2, 0), (0, 3), (0, 0)))),
        (again, np.asarray(((1, 0), (0, 1), (0, 0)))),
    ):
        lower, upper = program.evaluate(boxes, boxes)
        _contains(np.stack((lower[0], upper[0])), expected)


def test_template_admission_walks_nested_shared_jaxprs() -> None:
    @jax.jit
    def inner(value: jax.Array) -> jax.Array:
        return value * 2.0

    def outer(value: jax.Array) -> jax.Array:
        return inner(value) + inner(value)

    prepared = prepare_interval_function(outer, 1, batch_capacity=1)
    assert not _program_has_arrays((prepared.closed_jaxpr, prepared.closed_jaxpr))
    lower, upper = prepared.evaluate(np.asarray(((1.0,),)), np.asarray(((1.0,),)))
    assert np.all(lower <= 4.0) and np.all(upper >= 4.0)
    graph: dict[str, object] = {"program": prepared.closed_jaxpr}
    graph["self"] = graph
    assert not _program_has_arrays(graph)
    graph["request_coefficients"] = np.asarray((3.0,))
    assert _program_has_arrays(graph)
