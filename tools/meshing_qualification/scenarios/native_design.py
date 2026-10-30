from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
    TensorGridPlan,
    UniformAxisSpec,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SurfaceMeshingSpec,
)
from phydrax.meshing._proposals import (
    LearnedMeshProposer,
    mesh_proposal_scope,
    MeshMarkingProposal,
    MeshProposalFeatures,
    MeshProposalSafetyPolicy,
    prepare_mesh_proposal,
    project_mesh_proposal,
)
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import SizeControlStrength, UniformSizeControl
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_sources import NativeImplicitSource

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


class _FittedSizeModel(phx.AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)
    weights: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, weights: Sequence[float], bias: float, /) -> None:
        self.in_size = len(weights)
        self.out_size = "scalar"
        self.weights = jnp.asarray(weights, dtype=jnp.float64)
        self.bias = jnp.asarray(bias, dtype=jnp.float64)

    def __call__(
        self,
        features: jax.Array,
        /,
        *,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del key
        return jnp.dot(jnp.asarray(features), self.weights) + self.bias



class _PhysicalOracleRecord(TypedDict):
    represented_fe_qoi: float
    represented_exact_field_qoi: float
    continuous_sphere_qoi: float
    field_l2_error: float
    field_qoi_error_bound: float
    geometry_qoi_error: float
    maximum_radial_deviation_m: float
    surface_area_m2: float


class _PhysicalSolveRecord(TypedDict):
    space: phx.discretization.FiniteElementDiscretization
    problem: phx.equations.CompiledFiniteElementProblem
    solve_check: phx.linalg.LinearSolveCheckEvidence
    values: np.ndarray
    residual: np.ndarray
    form: str
    successful: bool
    dofs: int
    nodal_l2_error: float
    nodal_maximum_error: float
    normalized_residual: float
    independent_exact_solution: str
    physical_oracle: _PhysicalOracleRecord
    residual_tolerance: float

def _sphere_source(
    lattice_factor: int,
    radius: float = 0.75,
) -> tuple[
    phx.geometry.CompiledGeometry,
    phx.discretization.PreparedTensorGrid,
    NativeImplicitSource,
    phx.meshing.CellMeshingResult,
    dict[str, object],
]:
    half_width = 1.4
    resolution = 4
    count = lattice_factor * resolution + 1
    bounds = np.asarray(((-half_width,) * 3, (half_width,) * 3), dtype=np.float64)
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), radius, feature_id="native-design-sphere"
    ).compile()
    grid = TensorGridPlan(
        tuple(UniformAxisSpec(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(bounds)
    source_revision = canonical_fingerprint(
        {
            "kind": "native-design-sphere-source",
            "radius": radius,
            "lattice_count": count,
            "bounds": bounds,
        }
    )
    source = NativeImplicitSource(
        geometry, "native-design-sphere", source_revision, grid=grid
    )
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "native-design-sphere-boundary",
        np.asarray((0,), dtype=np.int64),
    )
    specification = SurfaceMeshingSpec(
        CellMeshingTarget(2, 3, CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            UniformSizeControl(
                scope,
                2.0 * half_width / (count - 1),
                strength=SizeControlStrength.SOFT,
            ),
        ),
    )
    options = NativeMeshingOptions(
        "implicit_surface",
        implicit_policy=phx.geometry.ImplicitSurfacePolicy(
            projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45)
        ),
    )
    plan = NativeMeshingProvider(options).plan(
        source, specification, coordinate_contract=SpatialCoordinateContract.si()
    )
    result = plan.execute()
    accepted = accepted_result_record(result)
    return (
        geometry,
        grid,
        source,
        result,
        {
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "radius_m": radius,
            "lattice_bounds_m": bounds.tolist(),
            "lattice_points_per_axis": count,
            "lattice_factor": lattice_factor,
            "native_route": plan.route,
            "accepted_result": accepted,
        },
    )


def _sphere_load(points: jnp.ndarray, args: object) -> jnp.ndarray:
    del args
    return points[..., 0] ** 2


def _exact(points: np.ndarray, /) -> np.ndarray:
    return np.square(points[:, 0])


def _duffy_rule(order: int, /) -> tuple[np.ndarray, np.ndarray]:
    axis, weights = np.polynomial.legendre.leggauss(order)
    axis, weights = (axis + 1.0) * 0.5, weights * 0.5
    first, second = np.meshgrid(axis, axis, indexing="ij")
    barycentric = np.stack(
        (1.0 - first - (1.0 - first) * second, first, (1.0 - first) * second),
        axis=-1,
    ).reshape((-1, 3))
    return barycentric, (np.outer(weights, weights) * (1.0 - first)).reshape(-1)


def _physical_oracle(
    result: phx.meshing.CellMeshingResult,
    space: phx.discretization.FiniteElementDiscretization,
    values: np.ndarray,
    radius: float,
    order: int,
    /,
) -> _PhysicalOracleRecord:
    barycentric, reference_weights = _duffy_rule(order)
    coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64)
    squared_error = 0.0
    represented_qoi = 0.0
    represented_exact_qoi = 0.0
    surface_area = 0.0
    maximum_radial_deviation = 0.0
    for index, (block, dof_map) in enumerate(
        zip(result.mesh.blocks, space.dof_maps, strict=True)
    ):
        if block.cell_kind != "triangle":
            raise ValueError("The native design sphere requires triangle surface cells.")
        triangles = coordinates[np.asarray(block.vertices, dtype=np.int64)]
        cell_values = values[np.asarray(dof_map.cell_dofs[index], dtype=np.int64)]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        measures = 0.5 * np.linalg.norm(cross, axis=1)
        if np.any(measures <= 0.0) or not np.all(np.isfinite(measures)):
            raise ValueError("The native design surface contains a degenerate triangle.")
        points = np.einsum("qi,cid->cqd", barycentric, triangles)
        discrete = cell_values @ barycentric.T
        physical_weights = 2.0 * measures[:, None] * reference_weights
        exact = np.square(points[..., 0])
        squared_error += float(np.sum(physical_weights * np.square(discrete - exact)))
        represented_qoi += float(np.sum(physical_weights * discrete))
        x = triangles[..., 0]
        represented_exact_qoi += float(
            np.sum(
                measures
                * (
                    np.sum(x * x, axis=1)
                    + x[:, 0] * x[:, 1]
                    + x[:, 0] * x[:, 2]
                    + x[:, 1] * x[:, 2]
                )
                / 6.0
            )
        )
        surface_area += float(np.sum(measures))
        maximum_radial_deviation = max(
            maximum_radial_deviation,
            float(
                np.max(
                    np.abs(np.linalg.norm(points, axis=-1) - radius),
                    initial=0.0,
                )
            ),
        )
    continuous_qoi = 4.0 * np.pi * radius**4 / 3.0
    field_l2 = float(np.sqrt(max(0.0, squared_error)))
    return {
        "represented_fe_qoi": represented_qoi,
        "represented_exact_field_qoi": represented_exact_qoi,
        "continuous_sphere_qoi": continuous_qoi,
        "field_l2_error": field_l2,
        "field_qoi_error_bound": field_l2 * np.sqrt(surface_area),
        "geometry_qoi_error": abs(represented_exact_qoi - continuous_qoi),
        "maximum_radial_deviation_m": maximum_radial_deviation,
        "surface_area_m2": surface_area,
    }


def _physical_solve(
    result: phx.meshing.CellMeshingResult, radius: float = 0.75, /
) -> _PhysicalSolveRecord:
    field = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    space = FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    if not isinstance(space, phx.discretization.FiniteElementDiscretization):
        raise TypeError("Native design solve requires the prepared P1 FE space.")
    source = phx.equations.coefficient(
        _sphere_load, coefficient_id="native-design-sphere-x2-load"
    )
    form = phx.equations.FiniteElementForm(
        "native-design-sphere-reaction",
        "u",
        (
            phx.equations.MassAction("u", 1.0),
            phx.equations.SourceAction("u", source),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(form, space)
    operator, rhs = problem.linear_system()
    solved, solve_check = phx.linalg.solve_checked(operator, rhs)
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    coordinates = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    exact = _exact(coordinates)
    nodal_l2 = float(np.sqrt(np.mean(np.square(values - exact))))
    nodal_max = float(np.max(np.abs(values - exact), initial=0.0))
    residual = np.asarray(problem.residual(solved.value), dtype=np.float64)
    rhs_norm = float(jnp.linalg.norm(rhs))
    normalized_residual = float(jnp.linalg.norm(residual)) / max(1.0, rhs_norm)
    oracle = _physical_oracle(result, space, values, radius, 10)
    if (
        not bool(np.all(np.asarray(solve_check.valid)))
        or values.shape != exact.shape
        or normalized_residual > 2.0e-8
    ):
        raise RuntimeError(
            f"Native sphere FE solve failed its numerical/exact-field oracle: {normalized_residual=}."
        )
    return {
        "space": space,
        "problem": problem,
        "solve_check": solve_check,
        "values": values,
        "residual": residual,
        "form": form.form_id,
        "successful": bool(np.all(np.asarray(solve_check.valid))),
        "dofs": int(values.size),
        "nodal_l2_error": nodal_l2,
        "nodal_maximum_error": nodal_max,
        "normalized_residual": normalized_residual,
        "independent_exact_solution": "u=x^2 on the analytic sphere for the reaction equation u=x^2",
        "physical_oracle": oracle,
        "residual_tolerance": 2.0e-8,
    }


def _vertex_dof_rows(
    mesh: phx.discretization.CellMesh,
    space: phx.discretization.FiniteElementDiscretization,
    /,
) -> np.ndarray:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    dof_points = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    row_by_point = {tuple(point.tolist()): row for row, point in enumerate(dof_points)}
    return np.asarray(
        [row_by_point[tuple(point.tolist())] for point in points], dtype=np.int64
    )


def _proposal_features(
    source: phx.meshing.CellMeshingResult,
    source_solve: _PhysicalSolveRecord,
    radius: float,
    /,
) -> tuple[MeshingScope, np.ndarray, np.ndarray]:
    scope = mesh_proposal_scope(source, source.mesh.topological_dimension)
    coordinates = np.asarray(source.mesh.coordinates, dtype=np.float64)
    barycentric, reference_weights = _duffy_rule(6)
    cell_errors: list[np.ndarray] = []
    cell_ids: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    values = np.asarray(source_solve["values"], dtype=np.float64)
    space = source_solve["space"]
    for block_index, block in enumerate(source.mesh.blocks):
        if block.cell_kind != "triangle":
            raise ValueError("Learned sphere marking requires native triangle blocks.")
        triangles = coordinates[np.asarray(block.vertices, dtype=np.int64)]
        dof_rows = np.asarray(space.dof_maps[0].cell_dofs[block_index], dtype=np.int64)
        cell_values = values[dof_rows]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        measures = 0.5 * np.linalg.norm(cross, axis=1)
        points = np.einsum("qi,cid->cqd", barycentric, triangles)
        discrete = cell_values @ barycentric.T
        exact = np.square(points[..., 0])
        errors = np.sum(
            2.0
            * measures[:, None]
            * reference_weights[None, :]
            * np.square(discrete - exact),
            axis=1,
        )
        cell_errors.append(np.sqrt(np.maximum(errors, 0.0)))
        cell_ids.append(np.asarray(block.global_ids, dtype=np.int64))
        centroids.append(np.mean(triangles, axis=1))
    all_ids = np.concatenate(cell_ids)
    row_by_id = {int(identifier): row for row, identifier in enumerate(all_ids.tolist())}
    rows = np.asarray(
        [row_by_id[int(identifier)] for identifier in np.asarray(scope.entity_ids)],
        dtype=np.int64,
    )
    observed_error = np.concatenate(cell_errors)[rows]
    points = np.concatenate(centroids)[rows]
    entity_ids = np.asarray(scope.entity_ids, dtype=np.int64)
    if points.shape[0] != entity_ids.size:
        raise ValueError("Cell-error rows do not match the native proposal scope.")
    threshold = float(np.median(observed_error))
    analytic_scores = np.where(observed_error >= threshold, 1.0, -1.0)
    feature_rows = np.column_stack(
        (
            points / radius,
            observed_error / max(float(np.max(observed_error, initial=0.0)), 1.0e-14),
        )
    )
    return scope, feature_rows, analytic_scores


def _qoi_error(
    result: phx.meshing.CellMeshingResult,
    solve: _PhysicalSolveRecord,
    radius: float,
    estimator_id: str,
    qoi_certificate_id: str,
    /,
    *,
    transfer_error: float = 0.0,
    quadrature_order: int = 10,
) -> phx.meshing.PhysicalErrorEvidence:
    oracle = _physical_oracle(
        result,
        solve["space"],
        np.asarray(solve["values"], dtype=np.float64),
        radius,
        quadrature_order,
    )
    return phx.meshing.PhysicalErrorEvidence(
        result.result_id,
        "authored-continuous-sphere-x2-moment",
        estimator_id,
        field_error=float(oracle["field_qoi_error_bound"]),
        geometry_error=float(oracle["geometry_qoi_error"]),
        algebraic_error=abs(
            float(np.sum(np.asarray(solve["residual"], dtype=np.float64)))
        ),
        transfer_error=transfer_error,
        quantity="qoi",
        qoi_certificate_id=qoi_certificate_id,
    )


def _execute_proposal(
    source: phx.meshing.CellMeshingResult,
    source_solve: _PhysicalSolveRecord,
    name: str,
    proposal: MeshMarkingProposal,
    safety: MeshProposalSafetyPolicy,
    radius: float,
    qoi_certificate_id: str,
    /,
) -> tuple[
    dict[str, object], phx.meshing.MeshAdaptationCandidate, phx.meshing.CellMeshingResult
]:
    preparation_start = perf_counter()
    transaction = prepare_mesh_proposal(source, proposal, safety)
    preparation_seconds = perf_counter() - preparation_start
    if transaction.adaptation is None:
        raise RuntimeError(
            f"{name} proposal did not prepare native mesh adaptation evidence."
        )
    target = transaction.commit(source)
    if (
        not target.audit.passed
        or not target.compliance.passed
        or target.result_id != transaction.trusted_result.result_id
    ):
        raise RuntimeError(
            f"{name} proposal did not publish its independently audited accepted mesh."
        )

    solve_start = perf_counter()
    target_solve = _physical_solve(target, radius)
    reanalysis_seconds = perf_counter() - solve_start
    transition = transaction.adaptation.transition
    stencil = None if transition is None else transition.vertex_stencil
    if stencil is None:
        raise RuntimeError(
            f"{name} native topology transition omitted its FE field transfer stencil."
        )
    transfer_start = perf_counter()
    source_vertex_dofs = _vertex_dof_rows(source.mesh, source_solve["space"])
    source_vertex_values = np.asarray(source_solve["values"], dtype=np.float64)[
        source_vertex_dofs
    ]
    transferred = np.asarray(
        stencil.apply(source.mesh.vertex_global_ids, source_vertex_values),
        dtype=np.float64,
    )
    target_vertex_dofs = _vertex_dof_rows(target.mesh, target_solve["space"])
    target_vertex_values = np.asarray(target_solve["values"], dtype=np.float64)[
        target_vertex_dofs
    ]
    transfer_difference = transferred - target_vertex_values
    transfer_l2 = float(np.sqrt(np.mean(np.square(transfer_difference))))
    transfer_area = float(target_solve["physical_oracle"]["surface_area_m2"])
    transfer_error = transfer_l2 * np.sqrt(transfer_area)
    transfer_seconds = perf_counter() - transfer_start

    candidate_error = _qoi_error(
        target,
        target_solve,
        radius,
        f"{name}-independent-duffy-qoi",
        qoi_certificate_id,
        transfer_error=transfer_error,
        quadrature_order=8,
    )
    solve_check = target_solve["solve_check"]
    feasibility = phx.meshing.RouteFeasibility(
        source.result_id,
        target.result_id,
        transaction.adaptation.route.value,
        target.result_id,
        cell_families=tuple(block.cell_kind for block in target.mesh.blocks),
        geometry_layout_id=target.geometry.geometry_layout_id,
        field_layouts=(("u", target_solve["space"].dof_maps[0].dof_map_id),),
        solver_operator_id=solve_check.operator_id,
        transfer_plan_id=stencil.stencil_id,
        compiled_layout_id=target_solve["problem"].compilation_id,
        geometry_certificate_id=(
            None if target.certification is None else target.certification.report_id
        ),
        topology_certificate_id=target.audit.validity.certificate_id,
    )
    cost = phx.meshing.MeasuredAdaptationCost(
        target.result_id,
        preparation_seconds=preparation_seconds,
        compilation_seconds=0.0,
        adaptation_seconds=0.0,
        transfer_seconds=transfer_seconds,
        reanalysis_seconds=reanalysis_seconds,
    )
    candidate = phx.meshing.MeshAdaptationCandidate(
        phx.meshing.AdaptationAction.H,
        target.result_id,
        feasibility,
        candidate_error,
        cost,
        solve_check,
    )
    return (
        {
            "status": "observed",
            "proposal_id": proposal.proposal_id,
            "proposer_id": proposal.proposer_id,
            "source_result_id": source.result_id,
            "target_result_id": target.result_id,
            "source_cells": sum(block.cell_count for block in source.mesh.blocks),
            "target_cells": sum(block.cell_count for block in target.mesh.blocks),
            "adaptation_status": transaction.adaptation.status.value,
            "candidate_error_bound": candidate_error.bound,
            "candidate_components": {
                "field_error": candidate_error.field_error,
                "geometry_error": candidate_error.geometry_error,
                "algebraic_error": candidate_error.algebraic_error,
                "transfer_error": candidate_error.transfer_error,
            },
            "transfer_maximum_error": float(
                np.max(np.abs(transfer_difference), initial=0.0)
            ),
            "solver_operator_id": solve_check.operator_id,
            "candidate_evidence_id": candidate.candidate_evidence_id,
        },
        candidate,
        target,
    )


def _sphere_moment_quadrature(radius: jnp.ndarray) -> jnp.ndarray:
    z, z_weights = np.polynomial.legendre.leggauss(12)
    azimuth = np.arange(24, dtype=np.float64) * (2.0 * np.pi / 24.0)
    z_values = jnp.asarray(z, dtype=jnp.float64)
    z_weights_array = jnp.asarray(z_weights, dtype=jnp.float64)
    azimuth_values = jnp.asarray(azimuth, dtype=jnp.float64)
    return (
        radius**4
        * jnp.sum(z_weights_array * (1.0 - z_values**2))
        * jnp.sum(jnp.cos(azimuth_values) ** 2)
        * (2.0 * jnp.pi / 24.0)
    )


def _derivative_oracle(radius: float, /) -> dict[str, object]:
    parameter = jnp.asarray(radius, dtype=jnp.float64)
    automatic = float(jax.grad(_sphere_moment_quadrature)(parameter))
    step = 1.0e-5 * radius
    finite_difference = float(
        (
            _sphere_moment_quadrature(parameter + step)
            - _sphere_moment_quadrature(parameter - step)
        )
        / (2.0 * step)
    )
    exact = 16.0 * np.pi * radius**3 / 3.0
    relative_error = max(abs(automatic - exact), abs(finite_difference - exact)) / abs(
        exact
    )
    return {
        "automatic_derivative": automatic,
        "central_difference_derivative": finite_difference,
        "analytic_derivative": exact,
        "relative_error": relative_error,
        "relative_tolerance": 1.0e-6,
        "passed": relative_error <= 1.0e-6,
    }


def _recordable_solve(value: dict[str, object], /) -> dict[str, object]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"space", "problem", "solve_check", "values", "residual"}
    }


def _profile_case(
    name: str,
    lattice_factor: int,
    /,
    *,
    combined_target: bool,
    meshcore: str,
) -> dict[str, object]:
    profile: dict[str, object] = {
        "profile_id": {
            "routine-sphere": "native-design-lifecycle",
            "larger-sphere": "native-design-lifecycle:larger-sphere",
            "combined-design-target": "native-design-lifecycle:combined-design-target",
        }[name],
        "case": name,
        "positive_intent": True,
        "source_registry": "tools/meshing_qualification/native_case_migration.json#design-learned-loop",
        "provenance": {
            "source_branch": "feature/native-meshing-20260929",
            "source_commit": "c56614a9fe52c4a8f56484874e91fde6498ec9ff",
            "source_file": "tools/meshing_qualification/scenarios/design.py::_design_qualification_source",
            "source_id": "native-design-sphere",
            "lattice_half_width_m": 1.4,
            "resolution": 4,
            "lattice_factor": lattice_factor,
            "lattice_points_per_axis": lattice_factor * 4 + 1,
            "radius_m": 0.75,
            "load": "x^2",
            "objective": "authored-continuous-sphere-x2-moment",
        },
        "meshcore_identity": meshcore,
        "status": "failed",
        "failures": [],
        "proposal_attempts": [],
        "missing_profile_contracts": [],
    }
    failures = profile["failures"]
    try:
        _, _, _, source, source_record = _sphere_source(lattice_factor)
        profile["source"] = source_record
    except Exception as error:
        failures.append(f"native implicit sphere source: {type(error).__name__}: {error}")
        return profile

    try:
        source_solve = _physical_solve(source, 0.75)
    except Exception as error:
        failures.append(f"source FE solve: {type(error).__name__}: {error}")
        return profile
    profile["source_physical_reanalysis"] = _recordable_solve(source_solve)
    profile["source_result_id"] = source.result_id
    profile["source_cell_count"] = sum(block.cell_count for block in source.mesh.blocks)
    derivative = _derivative_oracle(0.75)
    profile["independent_derivative_oracle"] = derivative
    if not derivative["passed"]:
        failures.append(
            "independent analytic sphere moment derivative exceeded 1e-6 relative tolerance"
        )

    physical = source_solve["physical_oracle"]
    qoi_certificate_id = canonical_fingerprint(
        {
            "kind": "continuous-sphere-x2-moment",
            "radius": 0.75,
            "value": physical["continuous_sphere_qoi"],
        }
    )
    objective_tolerance = 1.0e-8
    objective = phx.meshing.MeshAdaptationObjective(
        "authored-continuous-sphere-x2-moment",
        objective_tolerance,
        quantity="qoi",
    )
    baseline = _qoi_error(
        source,
        source_solve,
        0.75,
        "native-source-duffy-order-10",
        qoi_certificate_id,
        quadrature_order=10,
    )
    profile["objective"] = {
        "objective_id": objective.objective_id,
        "quantity": objective.quantity,
        "qoi_certificate_id": qoi_certificate_id,
        "absolute_tolerance": objective_tolerance,
        "baseline_error_bound": baseline.bound,
        "baseline_components": {
            "field_error": baseline.field_error,
            "geometry_error": baseline.geometry_error,
            "algebraic_error": baseline.algebraic_error,
            "transfer_error": baseline.transfer_error,
        },
    }

    spacing = 2.8 / (lattice_factor * 4)
    proposals: list[tuple[str, MeshMarkingProposal]] = []
    try:
        scope, features, analytic_scores = _proposal_features(source, source_solve, 0.75)
        safety = MeshProposalSafetyPolicy(
            source,
            minimum_size=0.04,
            maximum_size=spacing,
            maximum_displacement=0.0,
            maximum_gradation=1.5,
        )
    except Exception as error:
        safety = None
        setup_error = f"{type(error).__name__}: {error}"
        for proposal_name in ("uniform", "analytic", "learned"):
            profile["proposal_attempts"].append(
                {
                    "proposal": proposal_name,
                    "status": "failed",
                    "stage": "source-bound feature/safety setup",
                    "error": setup_error,
                }
            )
        failures.append(f"source-bound proposal setup: {setup_error}")

    if safety is not None:
        try:
            uniform = MeshMarkingProposal(
                source,
                scope,
                np.ones(scope.entity_ids.size, dtype=np.float64),
                proposer_id="uniform-native-sphere-marking",
            )
            proposals.append(("uniform", uniform))
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            profile["proposal_attempts"].append(
                {
                    "proposal": "uniform",
                    "status": "failed",
                    "stage": "proposal creation",
                    "error": detail,
                }
            )
            failures.append(f"uniform proposal creation: {detail}")
        try:
            analytic = MeshMarkingProposal(
                source,
                scope,
                analytic_scores,
                proposer_id="analytic-sphere-fe-error-marking",
            )
            proposals.append(("analytic", analytic))
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            profile["proposal_attempts"].append(
                {
                    "proposal": "analytic",
                    "status": "failed",
                    "stage": "proposal creation",
                    "error": detail,
                }
            )
            failures.append(f"analytic proposal creation: {detail}")
        try:
            design_matrix = np.column_stack((np.ones(features.shape[0]), features))
            regression, _, _, _ = np.linalg.lstsq(
                design_matrix, analytic_scores, rcond=None
            )
            model = _FittedSizeModel(regression[1:], float(regression[0]))
            proposer = LearnedMeshProposer(
                model,
                kind="marking",
                proposer_id="learned-observed-sphere-fe-error-marking",
            )
            feature_record = MeshProposalFeatures(
                source,
                scope,
                features,
                feature_ids=(
                    "x-over-radius",
                    "y-over-radius",
                    "z-over-radius",
                    "observed-fe-error",
                ),
                feature_owner_id="native-design-fe-error-estimator",
            )
            learned = proposer.propose(source, feature_record, scope=scope)
            if not isinstance(learned, MeshMarkingProposal):
                raise TypeError("The learned sphere proposer did not produce a marking proposal.")
            learned_fit_error = float(
                np.max(
                    np.abs(np.asarray(learned.values) - analytic_scores),
                    initial=0.0,
                )
            )
            profile["learned_training"] = {
                "observations": int(features.shape[0]),
                "feature_id": feature_record.feature_id,
                "features": ["x/r", "y/r", "z/r", "observed_cell_FE_error"],
                "target": "native cell markings supervised by independent per-cell FE error",
                "maximum_fit_error": learned_fit_error,
                "proposal_id": learned.proposal_id,
            }
            proposals.append(("learned", learned))
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            profile["proposal_attempts"].append(
                {
                    "proposal": "learned",
                    "status": "failed",
                    "stage": "source-bound learning",
                    "error": detail,
                }
            )
            failures.append(f"learned proposal creation: {detail}")

    candidates: list[phx.meshing.MeshAdaptationCandidate] = []
    targets: dict[str, phx.meshing.CellMeshingResult] = {}
    if safety is not None:
        for proposal_name, proposal in proposals:
            try:
                record, candidate, target = _execute_proposal(
                    source,
                    source_solve,
                    proposal_name,
                    proposal,
                    safety,
                    0.75,
                    qoi_certificate_id,
                )
                profile["proposal_attempts"].append(record)
                candidates.append(candidate)
                targets[candidate.candidate_id] = target
            except Exception as error:
                profile["proposal_attempts"].append(
                    {
                        "proposal": proposal_name,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                failures.append(
                    f"{proposal_name} source-bound proposal: {type(error).__name__}: {error}"
                )

    decision = objective.decide(baseline, candidates)
    profile["decision"] = {
        "decision_id": decision.decision_id,
        "selected_candidate_id": (
            None if decision.selected is None else decision.selected.candidate_id
        ),
        "dispositions": [
            {"candidate_id": candidate_id, "issues": list(issues)}
            for candidate_id, issues in decision.dispositions
        ],
        "satisfied_without_adaptation": decision.satisfied_without_adaptation,
    }
    if decision.selected is not None:
        target = targets[decision.selected.candidate_id]
        try:
            reanalysis_start = perf_counter()
            fresh_solve = _physical_solve(target, 0.75)
            fresh_error = _qoi_error(
                target,
                fresh_solve,
                0.75,
                "fresh-independent-duffy-order-14",
                qoi_certificate_id,
                quadrature_order=14,
            )
            decision.require_reanalysis(fresh_error, fresh_solve["solve_check"])
            profile["fresh_physical_reanalysis"] = {
                "solve": _recordable_solve(fresh_solve),
                "error_bound": fresh_error.bound,
                "independent_estimator_id": fresh_error.estimator_id,
                "elapsed_seconds": perf_counter() - reanalysis_start,
            }
        except Exception as error:
            failures.append(
                f"selected candidate fresh FE reanalysis: {type(error).__name__}: {error}"
            )
    elif not decision.satisfied_without_adaptation:
        failures.append(
            "MeshAdaptationObjective found no candidate meeting the authored sphere QoI target"
        )

    try:
        if safety is None:
            raise RuntimeError("Source-bound proposal safety policy is unavailable.")
        changed_radius = 0.68
        _, _, _, event_source, event_record = _sphere_source(
            lattice_factor, radius=changed_radius
        )
        axes = np.linspace(-1.4, 1.4, lattice_factor * 4 + 1, dtype=np.float64)
        lattice = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), axis=-1)
        initial_signs = np.sum(np.square(lattice), axis=-1) < 0.75**2
        changed_signs = np.sum(np.square(lattice), axis=-1) < changed_radius**2
        event_crossed = not np.array_equal(initial_signs, changed_signs)
        refused = []
        for proposal_name, proposal in proposals:
            try:
                project_mesh_proposal(event_source, proposal, safety)
            except ValueError:
                refused.append(proposal_name)
        profile["event_boundary_invalidation"] = {
            "changed_radius_m": changed_radius,
            "source_revision_changed": event_source.result_id != source.result_id,
            "lattice_sign_event_crossed": event_crossed,
            "native_event_source": event_record,
            "stale_proposals_refused": refused,
            "required_refusals": [name for name, _ in proposals],
        }
        if not event_crossed or not proposals or len(refused) != len(proposals):
            failures.append(
                "sphere lattice-event change did not refuse every source-bound proposal"
            )
    except Exception as error:
        failures.append(f"source event invalidation: {type(error).__name__}: {error}")

    profile_gaps = []
    profile_gaps.append(
        "phydrax/meshing/_proposals.py::prepare_mesh_proposal rejects boundary/association metadata and has no implicit source-chart transfer owner; the authored sphere marking proposals cannot safely remap this native implicit source"
    )
    if combined_target:
        profile_gaps.append(
            "the combined profile still requires a fixed-epoch FE-adjoint solve of the represented-surface moment target and republication/requalification at the designed radius"
        )
    if len(profile["proposal_attempts"]) != 3:
        profile_gaps.append(
            "all three source-bound uniform/analytic/learned proposals were not executed"
        )
    profile["missing_profile_contracts"] = profile_gaps
    if failures or profile_gaps:
        profile["status"] = "failed"
    else:
        profile["status"] = "passed"
    return profile


def native_design_loop() -> dict[str, object]:
    meshcore = require_meshcore()
    profiles = (
        _profile_case("routine-sphere", 2, combined_target=False, meshcore=meshcore),
        _profile_case("larger-sphere", 4, combined_target=False, meshcore=meshcore),
        _profile_case(
            "combined-design-target", 2, combined_target=True, meshcore=meshcore
        ),
    )
    all_passed = all(profile["status"] == "passed" for profile in profiles)
    missing_contracts: set[str] = set()
    for profile in profiles:
        profile_contracts = profile.get("missing_profile_contracts")
        if not isinstance(profile_contracts, list):
            raise TypeError("Design profile omitted its explicit missing-contract record.")
        for contract in profile_contracts:
            if not isinstance(contract, str):
                raise TypeError("Design missing-contract entries must be strings.")
            missing_contracts.add(contract)
    return {
        "status": "passed" if all_passed else "failed",
        "workflow": "native analytic sphere x^2 surface reaction -> independent QoI/error objective -> uniform/analytic/learned proposals -> fresh FE reanalysis and event invalidation",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#design-learned-loop",
        "profile_order": [profile["case"] for profile in profiles],
        "profile_count": len(profiles),
        "profiles": list(profiles),
        "missing_production_contracts": sorted(missing_contracts),
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
