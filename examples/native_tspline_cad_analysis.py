#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import SpatialCoordinateContract
from phydrax.discretization import iga
from phydrax.discretization.iga import TMesh2D
from phydrax.geometry.brep import (
    BRepGeometry,
    BRepImportReport,
    BRepModel,
    LineCurve,
    TSplineSurfacePatch,
)
from phydrax.meshing import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingScope,
    SurfaceMeshingSpec,
    UniformSizeControl,
)
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider


def _example_digest(payload: dict[str, object]) -> str:
    """Content digest of this example's authored source description."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _source_patch() -> TSplineSurfacePatch:
    breaks = tuple(Fraction(index, 8) for index in range(9))
    source_mesh = TMesh2D.tensor_product(breaks, breaks)
    marked = next(
        cell
        for cell in source_mesh.cells
        if cell.parameter_bounds
        == (
            Fraction(3, 8),
            Fraction(4, 8),
            Fraction(3, 8),
            Fraction(4, 8),
        )
    )
    refinement = source_mesh.refine((marked.cell_id,))
    if not refinement.certificate.passed:
        raise RuntimeError("The CAD source T-mesh lacks passing ASTS evidence.")
    exact_control_points = tuple(
        (u, v, u * v)
        for u, v in (
            anchor.local_knots.greville_parameter for anchor in source_mesh.anchors
        )
    )
    exact_weights = tuple(
        Fraction(1) + Fraction(2) * u * (1 - u) * v * (1 - v)
        for u, v, _z in exact_control_points
    )
    control_points = np.asarray(
        [[float(value) for value in row] for row in exact_control_points],
        dtype=np.float64,
    )
    weights = np.asarray([float(value) for value in exact_weights], dtype=np.float64)
    if len(set(exact_weights)) == 1:
        raise RuntimeError("The CAD example requires nonuniform exact rational weights.")
    orientations = {
        (extension.axis, extension.missing_direction)
        for extension in refinement.refined_mesh.t_junction_extensions
    }
    expected = {
        ("u", "left"),
        ("u", "right"),
        ("v", "down"),
        ("v", "up"),
    }
    if not orientations.issuperset(expected):
        raise RuntimeError("The source refinement lost an ASTS T-junction orientation.")
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
    patch: TSplineSurfacePatch,
    coordinate_contract: SpatialCoordinateContract,
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
    source_digest = _example_digest(
        {
            "kind": "native-tspline-cad-example-source",
            "surface_definition": patch.source_definition_id,
            "boundary_points": points.tolist(),
        }
    )
    import_policy_id = _example_digest(
        {"kind": "native-tspline-cad-example-import-policy"}
    )
    report = BRepImportReport(
        source_id="example:native-tspline-surface",
        source_digest=source_digest,
        source_format="native",
        coordinate_contract=coordinate_contract,
        import_policy_id=import_policy_id,
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
        coordinate_contract=coordinate_contract,
        report=report,
    )


def _unit_source(points: jax.Array, args: object) -> jax.Array:
    del args
    return jnp.ones(points.shape[:-1], dtype=points.dtype)


def main() -> None:
    coordinate_contract = SpatialCoordinateContract.si()
    surface = _source_patch()
    model = _brep_model(surface, coordinate_contract)
    scope = MeshingScope.whole(model, 2)
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(
            2,
            3,
            CellFamilyPolicy(required=("triangle",)),
        ),
        scope,
        size_controls=(UniformSizeControl(scope, 0.25),),
        maximum_deviation=0.025,
    )
    native_result = (
        NativeMeshingProvider(NativeMeshingOptions("parametric_surface"))
        .plan(model, specification, coordinate_contract=coordinate_contract)
        .execute()
    )
    native_mesh_cell_count = sum(block.cell_count for block in native_result.mesh.blocks)
    surface_source = native_result.surface_source
    acceptance = native_result.certification
    if (
        surface_source is None
        or acceptance is None
        or acceptance.fidelity is None
        or acceptance.inputs.source is None
        or acceptance.inputs.fidelity_tolerance is None
        or acceptance.inputs.fidelity_witness is None
    ):
        raise RuntimeError(
            "Accepted native CAD meshing discarded its source-chart proof."
        )
    acceptance.require_passed()
    witness = acceptance.inputs.fidelity_witness
    if witness is None or witness.witness_id != surface_source.chart_witness.witness_id:
        raise RuntimeError(
            "Accepted fidelity and retained source chart use different witnesses."
        )
    fidelity = acceptance.fidelity
    if fidelity.status != "certified":
        raise RuntimeError(
            "Accepted native CAD surface failed exact chart-bound source fidelity: "
            + "; ".join(
                f"{finding.check}: {finding.detail}" for finding in fidelity.findings
            )
        )
    if any(len(piece) != 3 for piece in surface_source.source_piece_ids):
        raise RuntimeError(
            "T-spline CAD ancestry must retain patch, control-cell, and extraction-piece IDs."
        )

    trim = iga.UVTrimmedSurface(
        iga.UVTrimLoop(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    )
    plan = iga.TSplineIsogeometricPlan(
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
    refinement = plan.refine((marked.cell_id,))
    target_plan = refinement.target_plan
    target_mesh = target_plan.geometry.mesh
    full_tensor_count = (len(target_mesh.axis_breaks[0]) - 1) * (
        len(target_mesh.axis_breaks[1]) - 1
    )
    if target_mesh.t_junction_count == 0 or target_mesh.cell_count >= full_tensor_count:
        raise RuntimeError("IGA refinement did not retain a local non-tensor T-mesh.")
    if (
        target_plan.geometry.surface_patch.source_mesh.mesh_id
        != surface.source_mesh.mesh_id
    ):
        raise RuntimeError("CAD-to-IGA transfer lost the exact source T-mesh ancestry.")
    if (
        not target_plan.geometry.surface_patch.basis_certificate.passed
        or target_plan.geometry.surface_patch.basis_certificate.continuity_order != 2
        or not target_plan.exact_geometry_certificate.passed
        or not target_plan.geometry_certificate.passed
    ):
        raise RuntimeError("IGA refinement failed ASTS or exact rational-map proofs.")

    prepared = target_plan.prepare(numeric_version="native-tspline-cad-poisson")
    source = phx.equations.coefficient(
        _unit_source,
        coefficient_id="native-tspline-cad-poisson-unit-source",
    )
    form = phx.equations.FiniteElementForm(
        "native-tspline-cad-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction("u", source),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        prepared,
        constraint=prepared.homogeneous_trace_constraint("u"),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="dense",
        ),
    )
    operator, right_hand_side = compiled.linear_system()
    result = phx.linalg.solve(operator, right_hand_side)
    residual_norm = jnp.linalg.norm(compiled.residual(result.value))
    rhs_norm = jnp.linalg.norm(right_hand_side)
    solution = compiled.expand(result.value)
    residual_bound = 2.0e-8 * jnp.maximum(rhs_norm, 1.0)
    if (
        not bool(jnp.all(result.successful))
        or float(residual_norm) > float(residual_bound)
        or float(jnp.linalg.norm(solution)) == 0.0
    ):
        raise RuntimeError(
            "The CAD-derived T-spline Poisson solve failed its residual gate."
        )

    print(
        json.dumps(
            {
                "cad_source_revision": model.source_revision,
                "exact_geometry_certificate_id": (
                    target_plan.exact_geometry_certificate.certificate_id
                ),
                "exact_geometry_passed": target_plan.exact_geometry_certificate.passed,
                "iga_cell_count": target_mesh.cell_count,
                "iga_t_junction_count": target_mesh.t_junction_count,
                "native_mesh_cell_count": native_mesh_cell_count,
                "native_provider_name": native_result.provider.name,
                "native_runtime_actual_version": (native_result.runtime.actual_version),
                "native_acceptance_report_id": acceptance.report_id,
                "native_source_fidelity_certificate_id": (
                    acceptance.fidelity.certificate_id
                ),
                "native_source_fidelity_status": acceptance.fidelity.status,
                "native_source_piece_keys": sorted(
                    {
                        tuple(int(value) for value in piece)
                        for piece in surface_source.source_piece_ids
                    }
                ),
                "native_source_root_count": len(
                    {int(value) for value in surface_source.source_root_cell_ids}
                ),
                "normalized_iga_residual": float(
                    residual_norm / jnp.maximum(rhs_norm, 1.0)
                ),
                "source_t_mesh_id": surface.source_mesh.mesh_id,
                "source_transfer_id": surface.transfer_plans[0].plan_id,
                "iga_transfer_id": refinement.basis_refinement.transfer_plan.plan_id,
                "numeric_geometry_passed": target_plan.geometry_certificate.passed,
                "successful": bool(jnp.all(result.successful)),
                "target_tensor_cell_count": full_tensor_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
