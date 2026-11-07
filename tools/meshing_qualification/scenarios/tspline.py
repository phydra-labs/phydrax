from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from tempfile import mkdtemp

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._interpolation import TMesh2D
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import iga
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry._source_fidelity import (
    certify_source_fidelity,
    validate_source_fidelity_witness,
)
from phydrax.geometry.brep._model import BRepGeometry, BRepImportReport, BRepModel
from phydrax.geometry.brep._patches import LineCurve
from phydrax.geometry.brep._tspline import TSplineSurfacePatch
from phydrax.interchange import (
    CadInterchangeError,
    load_brep_archive,
    save_brep_archive,
    write_iges,
    write_step,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import UniformSizeControl
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.variational import LocalFieldJet, LocalGeometry

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


def _source_patch() -> TSplineSurfacePatch:
    breaks = tuple(Fraction(index, 8) for index in range(9))
    source_mesh = TMesh2D.tensor_product(breaks, breaks)
    marked = next(
        cell
        for cell in source_mesh.cells
        if cell.parameter_bounds
        == (Fraction(3, 8), Fraction(4, 8), Fraction(3, 8), Fraction(4, 8))
    )
    refinement = source_mesh.refine((marked.cell_id,))
    if not refinement.certificate.passed:
        raise RuntimeError("T-spline source T-mesh failed its ASTS certificate.")
    exact_control_points = tuple(
        (u, v, u * v)
        for u, v in (
            anchor.local_knots.greville_parameter for anchor in source_mesh.anchors
        )
    )
    exact_weights = tuple(
        Fraction(1) + Fraction(2) * u * (1 - u) * v * (1 - v)
        for u, v, _ in exact_control_points
    )
    control_points = np.asarray(
        [[float(value) for value in row] for row in exact_control_points],
        dtype=np.float64,
    )
    weights = np.asarray([float(value) for value in exact_weights], dtype=np.float64)
    if len(set(exact_weights)) == 1:
        raise RuntimeError(
            "T-spline qualification requires nonuniform exact rational weights."
        )
    orientations = {
        (extension.axis, extension.missing_direction)
        for extension in refinement.refined_mesh.t_junction_extensions
    }
    expected = {("u", "left"), ("u", "right"), ("v", "down"), ("v", "up")}
    if not orientations.issuperset(expected):
        raise RuntimeError(
            "T-spline qualification lost an ASTS T-junction extension orientation."
        )
    return TSplineSurfacePatch(
        refinement.refined_mesh,
        control_points,
        weights,
        exact_control_points=exact_control_points,
        exact_weights=exact_weights,
        source_mesh=source_mesh,
        transfer_plans=(refinement.transfer_plan,),
        transfer_meshes=(source_mesh, refinement.refined_mesh),
    )


def _brep_model(
    patch: TSplineSurfacePatch, contract: SpatialCoordinateContract, /
) -> BRepModel:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    curves = tuple(
        LineCurve(points[first], points[last] - points[first]) for first, last in edges
    )
    pcurves = (
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        LineCurve((1.0, 0.0), (0.0, 1.0)),
        LineCurve((1.0, 1.0), (-1.0, 0.0)),
        LineCurve((0.0, 1.0), (0.0, -1.0)),
    )
    geometry = BRepGeometry(
        vertex_points=points,
        curves=curves,
        edge_curves=(0, 1, 2, 3),
        edge_ranges=np.tile(np.asarray((0.0, 1.0)), (4, 1)),
        edge_vertices=edges,
        pcurves=pcurves,
        coedge_edges=(0, 1, 2, 3),
        coedge_senses=(1, 1, 1, 1),
        face_loops=(((0, 1, 2, 3),),),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    source_digest = canonical_fingerprint(
        {
            "kind": "native-tspline-qualification-source",
            "surface_definition": patch.source_definition_id,
            "boundary_points": points.tolist(),
        }
    )
    policy_id = canonical_fingerprint(
        {"kind": "native-tspline-qualification-policy", "revision": 1}
    )
    report = BRepImportReport(
        source_id="qualification:native-tspline-cad",
        source_digest=source_digest,
        source_format="native",
        coordinate_contract=contract,
        import_policy_id=policy_id,
        num_solids=0,
        num_faces=1,
        num_edges=4,
        num_vertices=4,
    )
    return BRepModel(
        geometry=geometry,
        patches=(patch,),
        parameter_bounds=np.asarray((((0.0, 0.0), (1.0, 1.0)),)),
        orientation=np.asarray((1.0,)),
        trim_domains=(
            phx.geometry.TrimDomain(
                np.asarray(
                    ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                    dtype=np.float64,
                )
            ),
        ),
        physical_tags=("analysis-surface",),
        coordinate_contract=contract,
        report=report,
    )


def _unit_source(points: jax.Array, args: object) -> jax.Array:
    del args
    return jnp.ones(points.shape[:-1], dtype=points.dtype)


def _manufactured_gradient_load(
    fields: Mapping[str, LocalFieldJet], geometry: LocalGeometry, context: object
) -> jax.Array:
    del context
    points = geometry.points
    x, y = points[..., 0], points[..., 1]
    gradient = jnp.stack(
        (
            jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y),
            jnp.pi * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y),
            jnp.zeros_like(x),
        ),
        axis=-1,
    )
    field_gradient = fields["u"].gradient
    if field_gradient is None:
        raise ValueError("The manufactured load requires the requested field gradient.")
    return -jnp.sum(field_gradient * gradient, axis=-1)


def _solve_manufactured(
    prepared: iga.PreparedTSplineIsogeometricDiscretization,
    route: iga.TSplineQueryRoute,
) -> tuple[float, list[float]]:
    form = phx.equations.FiniteElementForm(
        "native-tspline-manufactured-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.LocalFunctionalAction(
                phx.variational.LocalIntegralTerm(
                    "manufactured-gradient",
                    region="body",
                    fields=(phx.variational.FieldJetSpec("u", gradient=True),),
                    density=_manufactured_gradient_load,
                    density_id="native-tspline-manufactured-gradient-density",
                ),
                {"u": "u"},
                ("u",),
                action_id="native-tspline-manufactured-gradient-load",
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        prepared,
        constraint=prepared.homogeneous_trace_constraint("u"),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free", local_kernel="dense"
        ),
    )
    operator, rhs = compiled.linear_system()
    result = phx.linalg.solve(operator, rhs)
    residual = jnp.linalg.norm(compiled.residual(result.value))
    if not bool(jnp.all(result.successful)) or float(residual) > float(
        2e-8 * jnp.maximum(jnp.linalg.norm(rhs), 1.0)
    ):
        raise RuntimeError("Manufactured T-spline Poisson residual failed.")
    solution = compiled.expand(result.value)
    actual = prepared.evaluate_field_route("u", route, solution)
    points = prepared.evaluate_geometry_route(route).values
    expected = jnp.sin(jnp.pi * points[..., 0]) * jnp.sin(jnp.pi * points[..., 1])
    max_error = float(jnp.max(jnp.abs(actual - expected)))
    if max_error > 2e-3:
        raise RuntimeError(
            f"Manufactured T-spline physical-field error {max_error} exceeds 0.002."
        )
    return max_error, np.asarray(solution).tolist()


def _solve_poisson(
    prepared: iga.PreparedTSplineIsogeometricDiscretization,
) -> tuple[float, list[float]]:
    source_term = phx.equations.coefficient(
        _unit_source, coefficient_id="native-tspline-qualification-unit-source"
    )
    form = phx.equations.FiniteElementForm(
        "native-tspline-qualification-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction("u", source_term),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        prepared,
        constraint=prepared.homogeneous_trace_constraint("u"),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free", local_kernel="dense"
        ),
    )
    operator, right_hand_side = compiled.linear_system()
    solve = phx.linalg.solve(operator, right_hand_side)
    residual = jnp.linalg.norm(compiled.residual(solve.value))
    rhs_norm = jnp.linalg.norm(right_hand_side)
    solution = compiled.expand(solve.value)
    if (
        not bool(jnp.all(solve.successful))
        or float(residual) > float(2.0e-8 * jnp.maximum(rhs_norm, 1.0))
        or float(jnp.linalg.norm(solution)) == 0.0
    ):
        raise RuntimeError(
            "T-spline CAD-derived Poisson solve failed its independent residual oracle."
        )
    return float(residual / jnp.maximum(rhs_norm, 1.0)), np.asarray(solution).tolist()


def _produce(archive: Path) -> dict[str, object]:
    meshcore = require_meshcore()
    contract = SpatialCoordinateContract.si()
    surface = _source_patch()
    model = _brep_model(surface, contract)
    save_brep_archive(model, archive)
    projection = phx.geometry.prepare_brep_projection(model)
    vertex_query = projection.project(
        np.asarray(((0.0, 0.0, 0.0),), dtype=np.float64),
        np.asarray((0,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
    )
    if int(vertex_query.status[0]) != 0 or float(vertex_query.residuals[0]) > 1e-12:
        raise RuntimeError("T-spline source projection failed at the boundary vertex.")
    domain = MeshingDomain.from_brep(model)
    scope = MeshingScope.whole(domain, 2)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(UniformSizeControl(scope, 0.25),),
        maximum_deviation=0.025,
    )
    plan = NativeMeshingProvider(NativeMeshingOptions("parametric_surface")).plan(
        domain, specification, coordinate_contract=contract
    )
    native_result = plan.execute()
    accepted = accepted_result_record(native_result)
    surface_source = native_result.surface_source
    certification = native_result.certification
    if surface_source is None or certification is None or certification.fidelity is None:
        raise RuntimeError(
            "T-spline CAD meshing omitted source-chart fidelity acceptance."
        )
    if (
        certification.inputs.source is None
        or certification.inputs.fidelity_tolerance is None
        or certification.inputs.fidelity_witness is None
    ):
        raise RuntimeError("T-spline acceptance omitted bound source-fidelity premises.")
    if (
        certification.inputs.fidelity_witness.witness_id
        != surface_source.chart_witness.witness_id
    ):
        raise RuntimeError(
            "T-spline acceptance and retained chart use different witnesses."
        )
    validate_source_fidelity_witness(
        native_result.mesh,
        native_result.geometry,
        certification.inputs.source,
        surface_source.chart_witness,
    )
    fidelity = certify_source_fidelity(
        native_result.mesh,
        native_result.geometry,
        certification.inputs.source,
        tolerance=certification.inputs.fidelity_tolerance,
        maximum_normal_deviation=certification.inputs.maximum_normal_deviation,
        witness=surface_source.chart_witness,
    )
    if (
        fidelity.status != "certified"
        or fidelity.certificate_id != certification.fidelity.certificate_id
    ):
        raise RuntimeError(
            "Independent T-spline source-fidelity revalidation disagreed with acceptance."
        )
    if any(len(piece) != 3 for piece in surface_source.source_piece_ids):
        raise RuntimeError("T-spline meshing lost patch/cell/extraction-piece ancestry.")

    trim = iga.UVTrimmedSurface(
        iga.UVTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    )
    iga_plan = iga.TSplineIsogeometricPlan(
        iga.TSplineGeometryState(surface),
        trim=trim,
        quadrature_policy=iga.IsogeometricQuadraturePolicy(6),
    )
    marked = min(
        surface.mesh.cells,
        key=lambda cell: (
            abs(float((cell.parameter_bounds[0] + cell.parameter_bounds[1]) / 2) - 0.5)
            + abs(float((cell.parameter_bounds[2] + cell.parameter_bounds[3]) / 2) - 0.5)
        ),
    )
    refinement = iga_plan.refine((marked.cell_id,))
    target_plan = refinement.target_plan
    distant = {
        cell.cell_id: cell.parameter_bounds
        for cell in surface.mesh.cells
        if float(cell.parameter_bounds[1]) <= 0.25
        and float(cell.parameter_bounds[3]) <= 0.25
    }
    target_cells = {
        cell.cell_id: cell.parameter_bounds for cell in target_plan.geometry.mesh.cells
    }
    if not distant or any(
        target_cells.get(key) != bounds for key, bounds in distant.items()
    ):
        raise RuntimeError("T-spline local refinement changed distant source topology.")
    target_mesh = target_plan.geometry.mesh
    full_tensor_count = (len(target_mesh.axis_breaks[0]) - 1) * (
        len(target_mesh.axis_breaks[1]) - 1
    )
    if target_mesh.t_junction_count == 0 or target_mesh.cell_count >= full_tensor_count:
        raise RuntimeError("T-spline IGA refinement lost its local non-tensor topology.")
    if (
        target_plan.geometry.surface_patch.source_mesh.mesh_id
        != surface.source_mesh.mesh_id
    ):
        raise RuntimeError("T-spline IGA refinement lost source T-mesh ancestry.")
    if (
        not target_plan.geometry.surface_patch.basis_certificate.passed
        or target_plan.geometry.surface_patch.basis_certificate.continuity_order != 2
        or not target_plan.exact_geometry_certificate.passed
        or not target_plan.geometry_certificate.passed
    ):
        raise RuntimeError(
            "T-spline IGA refinement failed basis, exact-map, or numeric-map certification."
        )
    source_prepared = iga_plan.prepare(numeric_version="native-tspline-source-field")
    exact_coefficients = tuple(
        Fraction((index % 7) - 3, 11) for index in range(surface.mesh.coefficient_count)
    )
    source_coefficients = jnp.asarray([float(value) for value in exact_coefficients])
    target_coefficients, target_weights = refinement.prolong_field(
        "u", source_coefficients
    )
    exact_target, exact_target_weights = refinement.prolong_exact_field(
        "u", exact_coefficients
    )
    if target_weights is None or exact_target_weights is None:
        raise RuntimeError("Weighted T-spline field lost its rational weights.")
    np.testing.assert_allclose(
        target_coefficients,
        [float(value) for value in exact_target],
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        target_weights,
        [float(value) for value in exact_target_weights],
        rtol=2e-12,
        atol=2e-12,
    )
    np.savez(
        archive.with_suffix(".transfer.npz"),
        source_coefficients=np.asarray(source_coefficients),
        target_coefficients=np.asarray(target_coefficients),
        target_weights=np.asarray(target_weights),
        marked_cell_id=marked.cell_id,
    )
    prepared = target_plan.prepare(numeric_version="native-tspline-qualification-poisson")
    query_points = np.asarray(((0.117, 0.223), (0.529, 0.681), (0.883, 0.347)))
    source_route = source_prepared.prepare_query_route(query_points)
    target_route = prepared.prepare_query_route(query_points)
    source_geometry = source_prepared.evaluate_geometry_route(source_route).values
    target_geometry = prepared.evaluate_geometry_route(target_route).values
    np.testing.assert_allclose(target_geometry, source_geometry, rtol=2e-12, atol=2e-12)
    source_values = source_prepared.evaluate_field_route(
        "u", source_route, source_coefficients
    )
    target_values = prepared.evaluate_field_route("u", target_route, target_coefficients)
    np.testing.assert_allclose(target_values, source_values, rtol=2e-12, atol=2e-12)
    if float(jnp.ptp(source_values)) < 1e-4:
        raise RuntimeError("T-spline transferred field is not meaningfully nonconstant.")
    runtime = prepared.default_runtime
    source_points = runtime.surface_patch.control_points
    source_weights = runtime.surface_patch.weights
    direction = (
        jnp.zeros_like(source_points)
        .at[:, 2]
        .set(jnp.linspace(0.0, 1.0, source_points.shape[0]))
    )

    def mapped(scale: jax.Array) -> jax.Array:
        points, weights = runtime.realize_parameter_bank(
            source_points + scale * direction,
            source_weights,
            source_authority_id=runtime.source_authority_id,
        )
        return prepared.basis_realization.evaluate_rational(
            target_route.parameters,
            points,
            weights,
            cell_ids=target_route.cell_ids,
            maximum_order=0,
        ).values

    zero = jnp.asarray(0.0)
    tangent = jax.jvp(mapped, (zero,), (jnp.ones_like(zero),))[1]

    def independent_value(scale: float) -> jax.Array:
        points = source_points + scale * direction
        weights = source_weights
        for transfer in runtime.surface_patch.transfer_plans:
            points, weights = transfer.apply_homogeneous_geometry(points, weights)
        return prepared.basis_realization.evaluate_rational(
            target_route.parameters,
            points,
            weights,
            cell_ids=target_route.cell_ids,
            maximum_order=0,
        ).values

    step = 1e-5
    difference = (independent_value(step) - independent_value(-step)) / (2 * step)
    np.testing.assert_allclose(tangent, difference, rtol=2e-8, atol=2e-10)
    cotangent = jnp.arange(tangent.size, dtype=tangent.dtype).reshape(tangent.shape) + 1
    adjoint = jax.vjp(mapped, zero)[1](cotangent)[0]
    np.testing.assert_allclose(
        adjoint, jnp.vdot(tangent, cotangent), rtol=2e-12, atol=2e-12
    )
    if float(jnp.linalg.norm(tangent)) == 0.0:
        raise RuntimeError("T-spline fixed-route derivative is zero.")
    check_axis = np.linspace(0.037, 0.963, 11)
    check_points = np.asarray([(u, v) for u in check_axis for v in check_axis])
    source_check_route = source_prepared.prepare_query_route(check_points)
    target_check_route = prepared.prepare_query_route(check_points)
    coarse_error, _ = _solve_manufactured(source_prepared, source_check_route)
    normalized_residual, solution = _solve_poisson(prepared)
    manufactured_error, manufactured_solution = _solve_manufactured(
        prepared, target_check_route
    )
    refusals = {}
    for name, writer in (("step", write_step), ("iges", write_iges)):
        try:
            writer(model, archive.with_suffix("." + name))
        except CadInterchangeError as error:
            if (
                error.refusal.reason != "unsupported-entity"
                or "multi-piece T-spline" not in error.refusal.message
            ):
                raise
            refusals[name] = error.refusal.message
        else:
            raise RuntimeError(f"{name} unexpectedly accepted a multi-piece T-spline.")
    return {
        "producer_pid": os.getpid(),
        "status": "passed",
        "route": plan.route,
        "cad_source_revision": model.source_revision,
        "source_t_mesh_id": surface.source_mesh.mesh_id,
        "native_result": accepted,
        "source_fidelity_certificate_id": certification.fidelity.certificate_id,
        "source_fidelity_revalidation_id": fidelity.certificate_id,
        "iga_cell_count": target_mesh.cell_count,
        "iga_t_junction_count": target_mesh.t_junction_count,
        "full_tensor_cell_count": full_tensor_count,
        "exact_geometry_certificate_id": target_plan.exact_geometry_certificate.certificate_id,
        "source_model_id": model.model_id,
        "source_definition_id": surface.source_definition_id,
        "refinement_id": refinement.refinement_id,
        "marked_cell_id": marked.cell_id,
        "target_mesh_id": target_mesh.mesh_id,
        "distant_cell_count": len(distant),
        "weighted_field_values": np.asarray(source_values).tolist(),
        "weighted_field_transfer_max_error": float(
            jnp.max(jnp.abs(target_values - source_values))
        ),
        "fixed_route_derivative_max_difference": float(
            jnp.max(jnp.abs(tangent - difference))
        ),
        "fixed_route_derivative_norm": float(jnp.linalg.norm(tangent)),
        "external_refusals": refusals,
        "normalized_poisson_residual": normalized_residual,
        "manufactured_max_point_error": manufactured_error,
        "manufactured_coarse_max_point_error": coarse_error,
        "manufactured_solution": manufactured_solution,
        "poisson_solution": solution,
        "solve_successful": True,
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }


def _restored_field_values(evidence: Mapping[str, object], name: str) -> np.ndarray:
    field = evidence[name]
    if not isinstance(field, list):
        raise TypeError(f"Archived {name} must be a list of scalar field values.")
    values: list[float] = []
    for value in field:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"Archived {name} must contain numeric scalar field values.")
        values.append(float(value))
    return np.asarray(values, dtype=np.float64)


def _continue_archive(archive: Path, evidence: dict[str, object]) -> dict[str, object]:
    if os.getpid() == evidence["producer_pid"]:
        raise RuntimeError("Archive continuation did not run in a fresh process.")
    model = load_brep_archive(archive)
    surface = model.patches[0]
    if not isinstance(surface, TSplineSurfacePatch):
        raise RuntimeError("Native archive lost the T-spline patch.")
    if (
        model.model_id != evidence["source_model_id"]
        or surface.source_definition_id != evidence["source_definition_id"]
    ):
        raise RuntimeError("Fresh-process T-spline archive changed source identity.")
    surface.validate_exact_transfer_chain()
    plan = iga.TSplineIsogeometricPlan(
        iga.TSplineGeometryState(surface),
        trim=iga.UVTrimmedSurface(
            iga.UVTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        ),
        quadrature_policy=iga.IsogeometricQuadraturePolicy(6),
    )
    marked_cell_id = evidence["marked_cell_id"]
    if not isinstance(marked_cell_id, int) or isinstance(marked_cell_id, bool):
        raise TypeError("The archived marked cell identifier must be an integer.")
    refinement = plan.refine((marked_cell_id,))
    if (
        refinement.refinement_id != evidence["refinement_id"]
        or refinement.target_plan.geometry.mesh.mesh_id != evidence["target_mesh_id"]
    ):
        raise RuntimeError(
            "Fresh-process refinement changed topology or transfer identity."
        )
    coefficients = jnp.asarray(
        [
            float(Fraction((index % 7) - 3, 11))
            for index in range(surface.mesh.coefficient_count)
        ]
    )
    transferred, _ = refinement.prolong_field("u", coefficients)
    prepared = refinement.target_plan.prepare(
        numeric_version="native-tspline-archive-continuation"
    )
    route = prepared.prepare_query_route(
        np.asarray(((0.117, 0.223), (0.529, 0.681), (0.883, 0.347)))
    )
    values = prepared.evaluate_field_route("u", route, transferred)
    normalized_residual, solution = _solve_poisson(prepared)
    check_axis = np.linspace(0.037, 0.963, 11)
    check_points = np.asarray([(u, v) for u in check_axis for v in check_axis])
    manufactured_route = prepared.prepare_query_route(check_points)
    manufactured_error, manufactured_solution = _solve_manufactured(
        prepared, manufactured_route
    )
    np.testing.assert_allclose(
        manufactured_solution,
        _restored_field_values(evidence, "manufactured_solution"),
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        solution,
        _restored_field_values(evidence, "poisson_solution"),
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        values,
        _restored_field_values(evidence, "weighted_field_values"),
        rtol=2e-12,
        atol=2e-12,
    )
    return {
        "status": "passed",
        "model_id": model.model_id,
        "target_mesh_id": refinement.target_plan.geometry.mesh.mesh_id,
        "normalized_poisson_residual": normalized_residual,
        "manufactured_max_point_error": manufactured_error,
        "reader_pid": os.getpid(),
    }


def native_tspline_cad_analysis() -> dict[str, object]:
    # Separate interpreters ensure the archive reader cannot retain producer objects.
    root = Path(mkdtemp(prefix="phydrax-tspline-qualification-"))
    archive = root / "source.phx"
    producer_result = root / "producer.json"
    reader_result = root / "reader.json"
    module = "tools.meshing_qualification.scenarios.tspline"
    producer = (
        "import json, pathlib; from " + module + " import _produce; "
        f"result = _produce(pathlib.Path({str(archive)!r})); "
        f"pathlib.Path({str(producer_result)!r}).write_text(json.dumps(result))"
    )
    subprocess.run((sys.executable, "-c", producer), check=True)
    reader = (
        "import json, pathlib; from " + module + " import _continue_archive; "
        f"evidence = json.loads(pathlib.Path({str(producer_result)!r}).read_text()); "
        f"result = _continue_archive(pathlib.Path({str(archive)!r}), evidence); "
        f"pathlib.Path({str(reader_result)!r}).write_text(json.dumps(result))"
    )
    subprocess.run((sys.executable, "-c", reader), check=True)
    result = json.loads(producer_result.read_text())
    result["fresh_process_continuation"] = json.loads(reader_result.read_text())
    result["continuation_artifacts"] = str(root)
    return result
