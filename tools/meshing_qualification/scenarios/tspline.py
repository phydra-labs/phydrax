from __future__ import annotations

from fractions import Fraction

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
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import UniformSizeControl
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider

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
        trim_domains=None,
        physical_tags=("analysis-surface",),
        coordinate_contract=contract,
        report=report,
    )


def _unit_source(points: jax.Array, args: object) -> jax.Array:
    del args
    return jnp.ones(points.shape[:-1], dtype=points.dtype)


def native_tspline_cad_analysis() -> dict[str, object]:
    meshcore = require_meshcore()
    contract = SpatialCoordinateContract.si()
    surface = _source_patch()
    model = _brep_model(surface, contract)
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
    prepared = target_plan.prepare(numeric_version="native-tspline-qualification-poisson")
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
    return {
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
        "normalized_poisson_residual": float(residual / jnp.maximum(rhs_norm, 1.0)),
        "solve_successful": bool(jnp.all(solve.successful)),
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
