from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._constructors import brep_sphere
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers import NativeMeshingProvider

from .._runtime import (
    accepted_result_record,
    require_meshcore,
    runtime_identity,
    single_cell_kind,
)


def _mapped_volume(
    mesh: phx.discretization.CellMesh, geometry: phx.discretization.CellGeometrySpec, /
) -> float:
    field = phx.discretization.FiniteElementFieldSpec(
        "x",
        phx.discretization.lagrange_element(single_cell_kind(mesh, "tetrahedron"), 1),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh, field, coordinate_spec=geometry
    ).prepare()
    blocks = discretization.evaluate_geometry("x", geometry.coordinates)
    return float(sum(jnp.sum(block.measure) for block in blocks))


def _solve_linear_field(result: phx.meshing.CellMeshingResult, /) -> dict[str, object]:
    # u=x belongs to the isoparametric field space: x composed with the
    # coordinate map has its degree, not necessarily degree one.
    degree = max(
        int(getattr(element, "degree", 1)) for element in result.geometry.elements
    )
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element(
            single_cell_kind(result.mesh, "tetrahedron"), degree
        ),
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    form = phx.equations.FiniteElementForm(
        "native-curved-sphere-harmonic",
        "u",
        (phx.equations.DiffusionAction("u", 1.0),),
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={"positive_definite": "asserted"},
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=lambda points: points[..., 0],
    )
    system, rhs = problem.linear_system(storage="sparse")
    prepared = phx.linalg.prepare(
        system,
        phx.linalg.LinearSolvePolicy(
            phx.linalg.PCG(),
            tolerance=phx.linalg.TolerancePolicy(relative=1.0e-14, absolute=0.0),
            preconditioning=phx.linalg.PreconditioningPolicy(
                phx.linalg.SparseFactorizationPreconditionerBuilder(
                    phx.linalg.SparseFactorizationPolicy(
                        "cholesky", ordering="nested-dissection"
                    )
                )
            ),
        ),
    )
    state = prepared.preconditioning_state
    if state is None or not isinstance(
        state.action, phx.linalg.SparseFactorizationPreconditioner
    ):
        raise RuntimeError("The sparse solve did not retain its native factorization.")
    factorization = state.action.factorization
    factor_status = int(factorization.status)
    minimum_pivot = float(factorization.diagnostics.minimum_pivot)
    solved = phx.linalg.solve(prepared, rhs)
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    exact = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)[:, 0]
    error = float(np.max(np.abs(values - exact), initial=0.0))
    residual = float(jnp.linalg.norm(system.operator.mv(solved.value) - rhs))
    rhs_norm = float(jnp.linalg.norm(rhs))
    relative_residual = float(solved.diagnostics.relative_residual)
    successful = bool(jnp.all(solved.successful))
    if (
        not successful
        or factor_status != int(phx.linalg.SparseFactorizationStatus.SUCCESS)
        or not minimum_pivot > 0.0
        or values.shape != exact.shape
        or error > 2.0e-9
        or residual > 2.0e-9 * max(1.0, rhs_norm)
        or relative_residual > 2.0e-9
    ):
        raise RuntimeError(
            "Curved-mesh FE solve failed its exact harmonic-field oracle: "
            f"{successful=}, {factor_status=}, {minimum_pivot=}, "
            f"{error=}, {residual=}, {rhs_norm=}, {relative_residual=}."
        )
    return {
        "form": form.form_id,
        "field_degree": degree,
        "successful": successful,
        "linear_solver": {
            "method": solved.provenance.method,
            "backend": solved.provenance.backend,
            "preconditioner": solved.provenance.preconditioner_id,
            "status": int(solved.status),
            "iterations": int(solved.diagnostics.iterations),
            "relative_residual": relative_residual,
            "factorization_status": factor_status,
            "minimum_pivot": minimum_pivot,
        },
        "field_dofs": int(values.size),
        "free_dofs": int(problem.state_space.size),
        "maximum_exact_solution_error": error,
        "residual_norm": residual,
        "normalized_residual": residual / max(1.0, rhs_norm),
        "exact_field": "u=x on the unit ball",
        "tolerance": 2.0e-9,
    }


def native_high_order_curving() -> dict[str, object]:
    meshcore = require_meshcore()
    contract = SpatialCoordinateContract.si()
    radius = 1.0
    exact_volume = (4.0 / 3.0) * np.pi * radius**3
    model = brep_sphere(
        radius,
        coordinate_contract=contract,
        source_id="qualification:native-high-order-sphere",
    )
    scope = MeshingScope.whole(model, 2)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("tetrahedron",)),
            geometry_order=1,
        ),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            UniformSizeControl(scope, 0.7, strength=SizeControlStrength.SOFT),
        ),
        maximum_deviation=0.08,
    )
    plan = NativeMeshingProvider().plan(
        model, specification, coordinate_contract=contract
    )
    straight_result = plan.execute()
    straight_accepted = accepted_result_record(straight_result)
    projection = phx.geometry.prepare_brep_projection(model)
    association_policy = phx.meshing.AssociationPropagationPolicy(
        classification_tolerance=1.0e-8
    )
    association = phx.meshing.associate_mesh_vertices(
        straight_result.mesh,
        projection,
        policy=association_policy,
    )
    policy = phx.meshing.HighOrderCurvingPolicy(degree=2)
    curved = phx.meshing.curve_cell_mesh(
        straight_result.mesh,
        association,
        projection,
        policy=policy,
        source_result=straight_result,
    )
    if curved.status is not phx.meshing.HighOrderCurvingStatus.CURVED:
        raise RuntimeError(
            f"Native P2 sphere curving did not accept a source-faithful map: {curved.status.value}."
        )
    if curved.cell_result is None:
        raise RuntimeError(
            "Native P2 curving did not publish a target-bound accepted cell result."
        )
    curved_result = curved.cell_result
    curved_accepted = accepted_result_record(curved_result)
    independent = phx.discretization.certify_cell_geometry_validity(
        curved.geometry, mesh=straight_result.mesh
    )
    straight_cell_count = sum(block.cell_count for block in straight_result.mesh.blocks)
    if (
        independent.certified_valid_count != straight_cell_count
        or independent.invalid_count != 0
        or independent.unresolved_count != 0
        or curved.evidence.maximum_residual > policy.residual_tolerance
    ):
        raise RuntimeError(
            "Independent high-order geometry certification or CAD residual failed."
        )
    straight_volume = _mapped_volume(straight_result.mesh, curved.straight)
    curved_volume = _mapped_volume(straight_result.mesh, curved.geometry)
    straight_error = abs(straight_volume - exact_volume) / exact_volume
    curved_error = abs(curved_volume - exact_volume) / exact_volume
    if not curved_error < 0.25 * straight_error:
        raise RuntimeError(
            f"P2 curving did not materially reduce exact sphere-volume error: {straight_error=}, {curved_error=}."
        )
    solve = _solve_linear_field(curved_result)

    first_cell_id = int(np.asarray(curved_result.mesh.blocks[0].global_ids)[0])
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            curved_result,
            phx.meshing.MarkedMeshAdaptation(
                np.asarray((first_cell_id,), dtype=np.int64)
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
                association_transfer=phx.meshing.BRepAssociationTransfer(
                    projection, policy=association_policy
                ),
            ),
        )
    )
    adapted = adaptation.target
    adapted_accepted = accepted_result_record(adapted)
    adapted_solve = _solve_linear_field(adapted)
    degree_values: list[int] = []
    for element in adapted.geometry.elements:
        degree = getattr(element, "degree", None)
        if not isinstance(degree, (int, np.integer)):
            raise RuntimeError(
                "Native bisection omitted degree-bearing geometry elements."
            )
        degree_values.append(int(degree))
    degrees = tuple(degree_values)
    if not degrees or max(degrees) < 2:
        raise RuntimeError(
            "Native bisection dropped the accepted high-order geometry map."
        )
    return {
        "status": "passed",
        "workflow": "native exact BRep sphere -> P1 tetrahedra -> P2 curving -> certified FE solve -> bisection continuation",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#high-order-curving-and-continuation",
        "source_id": model.source_id,
        "source_revision": model.source_revision,
        "route": plan.route,
        "geometry_order": policy.degree,
        "straight_result": straight_accepted,
        "curved_result": curved_accepted,
        "curving_status": curved.status.value,
        "curving_evidence_id": curved.evidence.evidence_id,
        "independent_certificate_id": independent.certificate_id,
        "maximum_source_residual": curved.evidence.maximum_residual,
        "straight_volume_m3": straight_volume,
        "curved_volume_m3": curved_volume,
        "exact_sphere_volume_m3": exact_volume,
        "straight_relative_volume_error": straight_error,
        "curved_relative_volume_error": curved_error,
        "initial_fe_solve": solve,
        "adaptation_status": adaptation.status.value,
        "adapted_result": adapted_accepted,
        "adapted_geometry_degrees": degrees,
        "continued_fe_solve": adapted_solve,
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
