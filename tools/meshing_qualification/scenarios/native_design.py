from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
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
from phydrax.geometry.design import (
    DerivativeTier,
    DesignQualificationEvidence,
    DesignState,
)
from phydrax.geometry.implicit import AnalyticImplicitProfile
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
from phydrax.meshing.providers import (
    NativeMeshingOptions,
    NativeMeshingPlan,
    NativeMeshingProvider,
)
from phydrax.meshing.providers._native_implicit import _PreparedFixedImplicitRoute
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
    solve_result: phx.linalg.LinearSolveResult
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
    *,
    fixed_plan: NativeMeshingPlan | None = None,
) -> tuple[
    phx.geometry.CompiledGeometry,
    phx.discretization.PreparedTensorGrid,
    NativeImplicitSource,
    NativeMeshingPlan,
    phx.meshing.CellMeshingResult,
    dict[str, object],
]:
    half_width = 1.4
    resolution = 4
    count = lattice_factor * resolution + 1
    bounds = np.asarray(((-half_width,) * 3, (half_width,) * 3), dtype=np.float64)
    if fixed_plan is None:
        geometry = phx.geometry.Sphere(
            (0.0, 0.0, 0.0), radius, feature_id="design-sphere"
        ).compile()
    else:
        if not isinstance(fixed_plan.source, NativeImplicitSource):
            raise ValueError("Fixed sphere realization requires its original implicit source.")
        original = fixed_plan.source.geometry
        index = original.schema.index(phx.geometry.ParameterId("design-sphere", "radius"))
        geometry = original.with_state(
            original.state.replace_at(index, jnp.asarray(radius, dtype=jnp.float64))
        )
    grid = TensorGridPlan(
        tuple(UniformAxisSpec(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(bounds)
    implicit_policy = phx.geometry.ImplicitSurfacePolicy(
        projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45)
    )
    coordinate_contract = SpatialCoordinateContract.si()
    source_revision = canonical_fingerprint(
        {
            "kind": "design-qualification-source",
            "parameters": tuple(np.asarray(value) for value in geometry.state.values),
            "parameter_ids": tuple(str(value) for value in geometry.schema.parameter_ids),
            "parameter_schema_id": geometry.schema.schema_id,
            "coordinate_contract": coordinate_contract.spatial_id,
            "lattice_count": count,
            "bounds": bounds,
            "fixed_lattice_policy": asdict(implicit_policy),
        }
    )
    profile = AnalyticImplicitProfile(
        geometry,
        coordinate_contract,
        source_id="design-sphere",
        source_revision=source_revision,
    )
    source = NativeImplicitSource(
        geometry,
        "design-sphere",
        source_revision,
        grid=grid,
        authoring_profile=profile,
    )
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
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
        implicit_policy=implicit_policy,
    )
    plan = NativeMeshingProvider(options).plan(
        source, specification, coordinate_contract=coordinate_contract, fixed_plan=fixed_plan
    )
    result = plan.execute()
    accepted = accepted_result_record(result)
    vertices = np.asarray(result.mesh.coordinates, dtype=np.float64)
    triangles = np.concatenate(
        [vertices[np.asarray(block.vertices, dtype=np.int64)] for block in result.mesh.blocks]
    )
    centers = np.mean(triangles, axis=1)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    orientation = np.einsum("ij,ij->i", normals, centers)
    if not np.all(np.isfinite(vertices)) or np.any(orientation <= 0.0):
        raise ValueError("Independent sphere audit found nonfinite or non-outward facets.")
    independent_source_audit = {
        "maximum_vertex_radial_error_m": float(
            np.max(np.abs(np.linalg.norm(vertices, axis=1) - radius))
        ),
        "maximum_centroid_radial_error_m": float(
            np.max(np.abs(np.linalg.norm(centers, axis=1) - radius))
        ),
        "minimum_outward_orientation": float(np.min(orientation)),
        "scope": "independent samples; global fidelity remains with the source certificate",
    }
    return (
        geometry,
        grid,
        source,
        plan,
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
            "independent_source_audit": independent_source_audit,
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
    from phydrax.solver._finite_element_adaptivity import _surface_p1_corners
    squared_error = 0.0
    represented_qoi = 0.0
    represented_exact_qoi = 0.0
    surface_area = 0.0
    maximum_radial_deviation = 0.0
    for index, (block, triangles) in enumerate(
        zip(result.mesh.blocks, _surface_p1_corners(result.mesh, result.geometry), strict=True)
    ):
        dof_map = space.dof_maps[0]
        if block.cell_kind != "triangle":
            raise ValueError("The native design sphere requires triangle surface cells.")
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
        _sphere_load, coefficient_id="ambient-x-squared-reaction"
    )
    form = phx.equations.FiniteElementForm(
        "native-design-surface-reaction",
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
        "solve_result": solved,
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
    dof_map = space.dof_maps[0]
    count = int(mesh.coordinates.shape[0])
    if dof_map.association != "vertex" or dof_map.global_dof_count != count:
        raise ValueError("Sphere P1 transfer requires the canonical vertex DOF layout.")
    return np.arange(count, dtype=np.int64)


def _proposal_features(
    source: phx.meshing.CellMeshingResult,
    source_solve: _PhysicalSolveRecord,
    radius: float,
    /,
) -> tuple[MeshingScope, np.ndarray, np.ndarray, np.ndarray]:
    del radius
    scope = mesh_proposal_scope(source, source.mesh.topological_dimension)
    from phydrax.solver._finite_element_adaptivity import _surface_p1_corners
    barycentric, reference_weights = _duffy_rule(8)
    cell_errors: list[np.ndarray] = []
    cell_ids: list[np.ndarray] = []
    features: list[np.ndarray] = []
    jensen: list[np.ndarray] = []
    values = np.asarray(source_solve["values"], dtype=np.float64)
    space = source_solve["space"]
    for block_index, (block, triangles) in enumerate(
        zip(source.mesh.blocks, _surface_p1_corners(source.mesh, source.geometry), strict=True)
    ):
        if block.cell_kind != "triangle":
            raise ValueError("Learned sphere marking requires native triangle blocks.")
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
        cell_errors.append(np.maximum(errors, 0.0))
        cell_ids.append(np.asarray(block.global_ids, dtype=np.int64))
        edges = triangles[:, (1, 2, 0)] - triangles
        diameter = np.max(np.linalg.norm(edges, axis=2), axis=1)
        energy = np.sum(np.square(edges[..., 0]), axis=1)
        features.append(
            np.stack(
                (measures * energy**2, measures * energy * diameter**2,
                 measures * diameter**4),
                axis=1,
            )
        )
        x = triangles[..., 0]
        jensen.append(np.mean(x * x, axis=1) - np.square(np.mean(x, axis=1)))
    all_ids = np.concatenate(cell_ids)
    row_by_id = {int(identifier): row for row, identifier in enumerate(all_ids.tolist())}
    rows = np.asarray(
        [row_by_id[int(identifier)] for identifier in np.asarray(scope.entity_ids)],
        dtype=np.int64,
    )
    observed_error = np.concatenate(cell_errors)[rows]
    feature_rows = np.concatenate(features)[rows]
    defects = np.concatenate(jensen)[rows]
    entity_ids = np.asarray(scope.entity_ids, dtype=np.int64)
    order = np.lexsort((entity_ids, -defects))
    analytic_scores = -np.ones(entity_ids.size, dtype=np.float64)
    analytic_scores[order[: (entity_ids.size + 1) // 2]] = 1.0
    return scope, feature_rows, analytic_scores, observed_error


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ordered = values[order]
        starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        ends = np.r_[starts[1:], values.size]
        result = np.empty(values.size, dtype=np.float64)
        result[order] = np.repeat(0.5 * (starts + ends - 1), ends - starts)
        return result

    a, b = ranks(first), ranks(second)
    if first.size < 2 or np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _learned_from_uniform(
    source: phx.meshing.CellMeshingResult,
    scope: MeshingScope,
    source_features: np.ndarray,
    held_out: np.ndarray,
    training: phx.meshing.CellMeshingResult,
    training_solve: _PhysicalSolveRecord,
    /,
) -> tuple[MeshMarkingProposal, dict[str, object]]:
    if source.result_id == training.result_id:
        raise ValueError("The learned comparator needs a distinct solved training mesh.")
    _, train_features, _, teacher = _proposal_features(training, training_solve, 0.75)
    scales = np.sqrt(np.mean(train_features**2, axis=0))
    teacher_scale = float(np.sqrt(np.mean(teacher**2)))
    if (
        np.any(scales <= 0.0)
        or not np.all(np.isfinite(scales))
        or not np.isfinite(teacher_scale)
        or teacher_scale <= 0.0
    ):
        raise ValueError("Learned comparator training scales must be finite and positive.")
    design = train_features / scales
    target = teacher / teacher_scale
    weights, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = float(np.linalg.norm(design @ weights - target) / np.linalg.norm(target))
    feature_record = MeshProposalFeatures(
        source,
        scope,
        source_features / scales,
        feature_ids=(
            "area-times-load-hessian-edge-energy-squared",
            "area-times-load-hessian-edge-energy-times-diameter-squared",
            "area-times-diameter-fourth",
        ),
        feature_owner_id="declared-load-hessian-flat-triangle-geometry",
    )
    proposer = LearnedMeshProposer(
        _FittedSizeModel(weights, 0.0),
        kind="marking",
        proposer_id="supervised-native-design-observed-error-marker",
    )
    proposal = proposer.propose(source, feature_record, scope=scope)
    if not isinstance(proposal, MeshMarkingProposal):
        raise TypeError("The learned comparator did not produce a marking proposal.")
    scores = np.asarray(proposal.values, dtype=np.float64)
    correlation = _rank_correlation(scores, held_out)
    if (
        rank != 3 or not np.all(np.isfinite(scores))
        or not np.isfinite(residual) or residual > 0.5
        or not np.isfinite(correlation) or correlation < 0.5
    ):
        raise ValueError(
            f"Learned comparator trust failed: {rank=}, {residual=}, {correlation=}."
        )
    try:
        bad = LearnedMeshProposer(
            _FittedSizeModel(np.full(weights.shape, np.nan), 0.0),
            kind="marking",
            proposer_id="nonfinite-native-design-marker",
        )
        bad.propose(source, feature_record, scope=scope)
    except (TypeError, ValueError) as error:
        nonfinite_refusal = f"{type(error).__name__}: {error}"
    else:
        raise RuntimeError("Nonfinite learned scores were admitted.")
    # The authored comparison requests the highest positive third, not a
    # resource cap.  Preserve stable source-ID ties before trusted projection.
    eligible = np.flatnonzero(scores > 0.0)
    order = np.lexsort((np.asarray(scope.entity_ids)[eligible], -scores[eligible]))
    selected = eligible[order[: (scores.size + 2) // 3]]
    selected_scores = -np.ones(scores.size, dtype=np.float64)
    selected_scores[selected] = scores[selected]
    selected_proposal = MeshMarkingProposal(
        source, scope, selected_scores,
        proposer_id="supervised-native-design-observed-error-marker-top-third",
    )
    return selected_proposal, {
        "training_result_id": training.result_id,
        "application_source_result_id": source.result_id,
        "training_samples": int(teacher.size),
        "application_samples": int(held_out.size),
        "feature_id": feature_record.feature_id,
        "feature_scales": scales.tolist(),
        "teacher_scale": teacher_scale,
        "weights": weights.tolist(),
        "fit_rank": int(rank),
        "training_relative_residual": residual,
        "training_relative_residual_bound": 0.5,
        "held_out_rank_correlation": correlation,
        "held_out_rank_correlation_bound": 0.5,
        "raw_proposal_id": proposal.proposal_id,
        "selected_positive_marks": int(selected.size),
        "nonfinite_prediction_refusal": nonfinite_refusal,
        "proposal_id": selected_proposal.proposal_id,
    }


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


def _field_error(
    result: phx.meshing.CellMeshingResult,
    solve: _PhysicalSolveRecord,
    radius: float,
    estimator_id: str,
    /,
    *,
    quadrature_order: int,
) -> phx.meshing.PhysicalErrorEvidence:
    oracle = _physical_oracle(
        result, solve["space"], solve["values"], radius, quadrature_order
    )
    # The directly integrated physical field error already includes the
    # algebraic and solution-refresh defects; do not add them twice.
    return phx.meshing.PhysicalErrorEvidence(
        result.result_id,
        "represented-surface-reaction-L2",
        estimator_id,
        field_error=oracle["field_l2_error"],
        geometry_error=0.0,
        algebraic_error=0.0,
        transfer_error=0.0,
    )


def _fresh_candidate_solve(
    target: phx.meshing.CellMeshingResult,
    radius: float,
    state_problem: phx.equations.CompiledFiniteElementProblem,
    candidate_mesh: phx.discretization.CellMesh,
    fields: tuple[jax.Array, ...],
    materials: object,
    adaptation: phx.meshing.MeshAdaptationResult,
    args: object,
) -> phx.solver.FiniteElementReanalysisResult:
    del fields, materials, args
    if candidate_mesh.mesh_id != target.mesh.mesh_id or adaptation.target.result_id != target.result_id:
        raise ValueError("Reanalysis must bind the selected target result.")
    fresh = _physical_solve(target, radius)
    return phx.solver.FiniteElementReanalysisResult(
        fresh["problem"], fresh["solve_result"], fresh["solve_check"], "u",
        state_problem=state_problem, check_policy=phx.linalg.LinearSolveCheckPolicy(),
    )


def _certify_candidate_fields(
    target: phx.meshing.CellMeshingResult,
    expected_u: jax.Array,
    source_inventory: float,
    candidate_mesh: phx.discretization.CellMesh,
    fields: tuple[jax.Array, ...],
    materials: object,
    lineage: object,
    args: object,
) -> bool:
    from phydrax.solver._finite_element_adaptivity import _surface_p1_measures

    del materials, lineage, args
    if candidate_mesh.mesh_id != target.mesh.mesh_id or len(fields) != 3:
        return False
    u, density, history = (np.asarray(field, dtype=np.float64) for field in fields)
    measures = _surface_p1_measures(candidate_mesh, target.geometry)
    return bool(
        np.all(np.isfinite(u))
        and np.max(np.abs(u - np.asarray(expected_u)), initial=0.0) <= 1.0e-10
        and np.max(np.abs(density - 2.0), initial=0.0) <= 1.0e-10
        and np.max(np.abs(history - (np.asarray(candidate_mesh.coordinates)[:, 1] + 0.5)), initial=0.0) <= 1.0e-10
        and abs(float(np.dot(measures, density)) - source_inventory) <= 1.0e-9
    )


def _stage_all_fields(
    source: phx.meshing.CellMeshingResult,
    source_solve: _PhysicalSolveRecord,
    target: phx.meshing.CellMeshingResult,
    target_solve: _PhysicalSolveRecord,
    adaptation: phx.meshing.MeshAdaptationResult,
    radius: float,
    /,
) -> tuple[dict[str, object], phx.discretization.FiniteElementDiscretization]:
    from phydrax.solver._finite_element_adaptivity import _surface_p1_measures

    semantics = phx.solver.FiniteElementFieldTransferSemantics
    fields = tuple(
        FiniteElementFieldSpec(name, lagrange_element("triangle", 1))
        for name in ("u", "density", "history")
    )
    layouts = tuple(
        phx.solver.FiniteElementAcceptedFieldLayout(field, transfer_semantics=mode)
        for field, mode in zip(
            fields,
            (
                semantics.NONCONSERVATIVE_INTERPOLATION,
                semantics.CONSERVATIVE_SURFACE_CONTENT,
                semantics.NONCONSERVATIVE_INTERPOLATION,
            ),
            strict=True,
        )
    )
    source_all = FiniteElementPlan(
        source.mesh, fields, coordinate_spec=source.geometry
    ).prepare()
    all_space = FiniteElementPlan(
        target.mesh, fields, coordinate_spec=target.geometry
    ).prepare()
    if not isinstance(source_all, phx.discretization.FiniteElementDiscretization) or not isinstance(
        all_space, phx.discretization.FiniteElementDiscretization
    ):
        raise TypeError("All-field reanalysis requires finite-element discretizations.")
    state_form = phx.equations.FiniteElementForm(
        "native-design-surface-reaction",
        tuple(field.name for field in fields),
        source_solve["problem"].form.actions,
    )
    source_state_problem = phx.equations.compile_finite_element_problem(state_form, source_all)
    target_state_problem = phx.equations.compile_finite_element_problem(state_form, all_space)
    source_points = np.asarray(source.mesh.coordinates, dtype=np.float64)
    accepted = phx.solver.FiniteElementAcceptedState(
        (
            jnp.asarray(source_solve["values"]),
            jnp.full((len(source_points),), 2.0),
            jnp.asarray(source_points[:, 1] + 0.5),
        ),
        0.0,
        0,
        source.mesh.topology_id,
        source_all.prepared_id,
        source_state_problem.compilation_id,
        field_layouts=layouts,
    )
    source_inventory = float(2.0 * np.sum(_surface_p1_measures(source.mesh, source.geometry)))
    certify = eqx.Partial(
        _certify_candidate_fields, target, jnp.asarray(target_solve["values"]), source_inventory
    )
    rollback = phx.solver.FiniteElementTopologyTransaction(certify).execute(
        accepted, source.mesh, adaptation
    )
    if (
        bool(np.asarray(rollback.committed))
        or rollback.state is not accepted
        or rollback.mesh is not source.mesh
        or rollback.diagnostics != "candidate-certification-rejected"
    ):
        raise RuntimeError(f"Unrefreshed reanalysis did not exercise physical rollback: {rollback.diagnostics}")
    executor = phx.solver.FiniteElementTopologyTransaction(certify)
    staged = executor.execute(
        accepted, source.mesh, adaptation,
        reanalysis_action=eqx.Partial(
            _fresh_candidate_solve, target, radius, target_state_problem
        ),
    )
    if not bool(np.asarray(staged.committed)) or staged.reanalysis is None:
        raise RuntimeError(f"All-field physical reanalysis refused: {staged.diagnostics}")
    if len(staged.surface_conservative_transfers) != 1:
        raise RuntimeError("Density transfer omitted its surface conservation proof.")
    evidence = staged.surface_conservative_transfers[0]
    if adaptation.transfer is None:
        raise ValueError("The fixed transfer derivative requires the actual native transfer.")
    transfer = adaptation.transfer
    linearization = phx.linalg.prepare_linearization(transfer.apply, accepted.fields[0])
    direction = jnp.linspace(-0.5, 0.5, accepted.fields[0].size)
    tangent = linearization.jvp(direction)
    cotangent = jnp.linspace(-0.3, 0.8, tangent.size)
    duality = abs(float(jnp.vdot(tangent, cotangent) - jnp.vdot(direction, linearization.vjp(cotangent))))
    finite = (transfer.apply(accepted.fields[0] + 1.0e-5 * direction)
              - transfer.apply(accepted.fields[0] - 1.0e-5 * direction)) / 2.0e-5
    transfer_fd = float(jnp.max(jnp.abs(tangent - finite)))
    if transfer_fd > 2.0e-6 or duality > 2.0e-8:
        raise ValueError("Fixed native transfer failed its finite-difference or duality oracle.")
    return {
        "source_accepted_id": accepted.accepted_id,
        "target_accepted_id": staged.state.accepted_id,
        "field_names": [layout.name for layout in layouts],
        "density_transfer_id": evidence.transfer.transfer_id,
        "transfer_plan_id": transfer.transfer_id,
        "density_conservation_evidence_id": evidence.evidence_id,
        "source_inventory": float(evidence.source_integral),
        "target_inventory": float(evidence.target_integral),
        "inventory_error": abs(float(evidence.conservation_defect)),
        "source_retained_after_unrefreshed_refusal": rollback.state is accepted and rollback.mesh is source.mesh,
        "unrefreshed_refusal": rollback.diagnostics,
        "reanalysis_operator_id": staged.reanalysis.solve_check.operator_id,
        "reanalysis_evidence_id": staged.reanalysis.evidence_id,
        "transfer_finite_difference_error": transfer_fd,
        "transfer_jvp_vjp_duality_error": duality,
        "reanalysis_valid": bool(np.all(np.asarray(staged.reanalysis.solve_check.valid))),
    }, all_space




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
    dict[str, object], phx.meshing.MeshAdaptationCandidate,
    phx.meshing.CellMeshingResult, _PhysicalSolveRecord,
    phx.meshing.MeshAdaptationResult, phx.meshing.PhysicalErrorEvidence,
]:
    preparation_start = perf_counter()
    transaction = prepare_mesh_proposal(
        source,
        proposal,
        safety,
        compatibility=phx.meshing.BisectionCompatibility.UNIFORM_REFINEMENT,
    )
    preparation_seconds = perf_counter() - preparation_start
    if transaction.adaptation is None:
        raise RuntimeError(
            f"{name} proposal did not prepare native mesh adaptation evidence."
        )
    if transaction.commit(source, accept=False) is not source:
        raise RuntimeError("Explicit proposal rejection failed to retain its source.")
    target = transaction.commit(source)
    try:
        project_mesh_proposal(target, proposal, safety)
    except ValueError as error:
        stale_refusal = f"{type(error).__name__}: {error}"
    else:
        raise RuntimeError("A proposal bound to the old source was accepted on its target.")
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
    staging_start = perf_counter()
    staged_record, all_space = _stage_all_fields(
        source, source_solve, target, target_solve, transaction.adaptation, radius
    )
    reanalysis_seconds += perf_counter() - staging_start

    candidate_error = _field_error(
        target, target_solve, radius, f"{name}-independent-duffy-field",
        quadrature_order=8,
    )
    candidate_qoi = _qoi_error(
        target, target_solve, radius, f"{name}-independent-duffy-qoi",
        qoi_certificate_id, quadrature_order=8,
    )
    solve_check = target_solve["solve_check"]
    feasibility = phx.meshing.RouteFeasibility(
        source.result_id,
        target.result_id,
        transaction.adaptation.route.value,
        target.result_id,
        cell_families=tuple(block.cell_kind for block in target.mesh.blocks),
        geometry_layout_id=target.geometry.geometry_layout_id,
        field_layouts=tuple(
            (field.name, field.layout.layout_id) for field in all_space.field_spaces
        ),
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
            "bisection_compatibility": "uniform_refinement",
            "stale_source_refusal": stale_refusal,
            "all_field_transaction": staged_record,
            "candidate_error_bound": candidate_error.bound,
            "qoi_components": {
                "field_error": candidate_qoi.field_error,
                "geometry_error": candidate_qoi.geometry_error,
                "algebraic_error": candidate_qoi.algebraic_error,
            },
            "transfer_rms_area_scaled": transfer_error,
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
        target_solve,
        transaction.adaptation,
        candidate_qoi,
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
        "scope": (
            "continuous analytic sphere moment only; not a derivative of the "
            "represented geometry, FE solve, transfer, or adaptation"
        ),
        "automatic_derivative": automatic,
        "central_difference_derivative": finite_difference,
        "analytic_derivative": exact,
        "relative_error": relative_error,
        "relative_tolerance": 1.0e-6,
        "passed": relative_error <= 1.0e-6,
    }


def _fixed_epoch_derivative(
    geometry: phx.geometry.CompiledGeometry,
    native_plan: NativeMeshingPlan,
    result: phx.meshing.CellMeshingResult,
    solve: _PhysicalSolveRecord,
    /,
    *,
    design_target: float | None = None,
) -> dict[str, object]:
    """Differentiate the actual represented FE problem on its fixed lattice branch."""
    prepared = native_plan.prepared
    if not isinstance(prepared, _PreparedFixedImplicitRoute):
        raise TypeError("Design derivatives require the actual fixed implicit plan.")
    surface = prepared.surface_plan
    radius_index = geometry.schema.index(
        phx.geometry.ParameterId("design-sphere", "radius")
    )
    radius = geometry.state.values[radius_index]
    lattice_radii = np.linalg.norm(np.asarray(surface.grid_points), axis=1)
    delta = lattice_radii - float(radius)
    lower = float(np.max(delta[delta < 0.0]))
    upper = float(np.min(delta[delta > 0.0]))
    event_distance = min(-lower, upper)
    direction = 0.7
    step = 1.0e-5 * direction
    if step >= event_distance / 32.0:
        raise ValueError("The derivative probe is not inside the fixed lattice event.")
    runtime = phx.discretization.FiniteElementRuntimeData(
        result.mesh,
        result.mesh.coordinates,
        numeric_version=result.mesh.numeric_version,
        geometry_layout_id=result.geometry.geometry_layout_id,
    )
    problem = solve["problem"]
    cells = tuple(jnp.asarray(block.vertices, dtype=jnp.int32) for block in result.mesh.blocks)
    dofs = tuple(jnp.asarray(rows, dtype=jnp.int32) for rows in solve["space"].dof_maps[0].cell_dofs)
    policy = phx.linalg.LinearSolvePolicy(
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(
            relative_tolerance=1.0e-12, absolute_tolerance=1.0e-14
        )
    )

    def coordinates(value: jax.Array) -> jax.Array:
        return surface.realize(geometry.state.replace_at(radius_index, value)).vertices

    def solved_values(value: jax.Array) -> tuple[jax.Array, jax.Array]:
        points = coordinates(value)
        current = eqx.tree_at(lambda data: data.coordinates, runtime, points)
        context = phx.equations.FiniteElementExecutionContext(current)
        operator, rhs = problem.linear_system(context)
        solution = phx.linalg.solve(operator, rhs, policy=policy)
        return points, problem.expand(solution.value, context)

    def represented(value: jax.Array) -> jax.Array:
        points, values = solved_values(value)
        total = jnp.zeros((), dtype=points.dtype)
        for triangles, rows in zip(cells, dofs, strict=True):
            corners = points[triangles]
            areas = 0.5 * jnp.linalg.norm(
                jnp.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                axis=1,
            )
            total = total + jnp.sum(areas * jnp.mean(values[rows], axis=1))
        return total

    taylor_distance = max(-lower, upper)
    taylor_sign = 1.0 if upper >= -lower else -1.0
    offsets = (
        0.0, -step, step,
        taylor_sign * taylor_distance / 8.0,
        taylor_sign * taylor_distance / 16.0,
        taylor_sign * taylor_distance / 32.0,
    )
    for offset in offsets:
        evidence = surface.realize(
            geometry.state.replace_at(radius_index, radius + offset)
        ).evidence
        if int(np.asarray(evidence.status)) != 0:
            raise ValueError(f"Fixed implicit realization refused derivative offset {offset}.")
    base, tangent = jax.jvp(represented, (radius,), (jnp.ones_like(radius),))
    adjoint = jax.grad(represented)(radius)
    minus = represented(radius - step)
    plus = represented(radius + step)
    finite_difference = (plus - minus) / (2.0 * step)

    def derivative_probe(
        function: Callable[[jax.Array], jax.Array],
    ) -> tuple[float, float]:
        linearization = phx.linalg.prepare_linearization(function, radius)
        primal = linearization.primal
        derivative = linearization.jvp(jnp.asarray(direction, dtype=radius.dtype))
        cotangent = jnp.linspace(-0.3, 0.8, primal.size, dtype=primal.dtype).reshape(
            primal.shape
        )
        duality = abs(
            float(jnp.vdot(derivative, cotangent) - direction * linearization.vjp(cotangent))
        )
        finite = (function(radius + step) - function(radius - step)) / 2.0e-5
        return float(jnp.max(jnp.abs(derivative - finite))), duality

    geometry_fd, geometry_duality = derivative_probe(coordinates)
    pde_fd, pde_duality = derivative_probe(lambda value: solved_values(value)[1])
    points = np.asarray(coordinates(radius), dtype=np.float64)
    _, point_tangent = jax.jvp(coordinates, (radius,), (jnp.ones_like(radius),))
    gradient = np.zeros_like(points)
    moment = 0.0
    for cell in cells:
        indices = np.asarray(cell, dtype=np.int64)
        corners = points[indices]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        lengths = np.linalg.norm(normal, axis=1)
        if np.any(lengths <= 0.0):
            raise ValueError("Independent moment oracle requires nondegenerate triangles.")
        area = 0.5 * lengths
        unit = normal / lengths[:, None]
        x = corners[..., 0]
        local_moment = (
            np.sum(x * x, axis=1) + x[:, 0] * x[:, 1]
            + x[:, 0] * x[:, 2] + x[:, 1] * x[:, 2]
        ) / 6.0
        moment += float(np.sum(area * local_moment))
        for local in range(3):
            opposite = corners[:, (local + 2) % 3] - corners[:, (local + 1) % 3]
            contribution = 0.5 * np.cross(unit, opposite) * local_moment[:, None]
            contribution[:, 0] += area * (x[:, local] + np.sum(x, axis=1)) / 6.0
            np.add.at(gradient, indices[:, local], contribution)
    analytic = float(np.sum(gradient * np.asarray(point_tangent)))
    adjoint_value = float(adjoint)
    base_value = float(base)
    remainder = [
        abs(float(represented(radius + offset)) - base_value - offset * adjoint_value)
        for offset in offsets[3:]
    ]
    if any(not np.isfinite(value) or value <= 0.0 for value in remainder):
        raise ValueError("The fixed-event Taylor remainders are not resolved.")
    slopes = [
        float(np.log(remainder[i] / remainder[i + 1]) / np.log(2.0))
        for i in range(2)
    ]
    roundoff_bound = (
        sum(block.cell_count for block in result.mesh.blocks)
        * np.finfo(np.float64).eps * abs(base_value)
    )
    checks = {
        "geometry_finite_difference": geometry_fd,
        "geometry_jvp_vjp_duality": geometry_duality,
        "pde_finite_difference": pde_fd,
        "pde_jvp_vjp_duality": pde_duality,
        "adjoint_tangent": abs(adjoint_value - float(tangent)),
        "adjoint_finite_difference": abs(adjoint_value - float(finite_difference)),
        "adjoint_analytic_moment": abs(adjoint_value - analytic),
        "represented_moment_identity": abs(base_value - moment),
        "taylor_maximum_slope_deviation": max(abs(value - 2.0) for value in slopes),
        "taylor_roundoff_ratio": roundoff_bound / min(remainder),
        "finite_difference_step_event_fraction": step / event_distance,
    }
    passed = (
        checks["geometry_finite_difference"] <= 2.0e-6
        and checks["geometry_jvp_vjp_duality"] <= 2.0e-8
        and checks["pde_finite_difference"] <= 2.0e-6
        and checks["pde_jvp_vjp_duality"] <= 2.0e-8
        and checks["adjoint_tangent"] <= 2.0e-8
        and checks["adjoint_finite_difference"] <= 2.0e-6
        and checks["adjoint_analytic_moment"] <= 2.0e-6
        and checks["represented_moment_identity"] <= 1.0e-8
        and checks["taylor_maximum_slope_deviation"] <= 0.2
        and checks["taylor_roundoff_ratio"] <= 1.0e-3
        and checks["finite_difference_step_event_fraction"] <= 1.0 / 32.0
    )
    record: dict[str, object] = {
        "scope": "fixed lattice, accepted projection/QEF branch and represented FE solve; no topology or acceptance derivative",
        "surface_plan_id": surface.plan_id,
        "source_result_id": result.result_id,
        "event_interval": [lower, upper],
        "central_difference_step": step,
        "finite_difference_direction": direction,
        "adjoint": adjoint_value,
        "tangent": float(tangent),
        "central_difference": float(finite_difference),
        "analytic_moment": analytic,
        "represented_qoi": base_value,
        "independent_moment": moment,
        "taylor_remainders": remainder,
        "taylor_roundoff_bound": roundoff_bound,
        "taylor_slopes": slopes,
        "checks": checks,
        "passed": passed,
    }
    if design_target is not None:
        bracket = (float(radius) + 0.5 * lower, float(radius) + 0.5 * upper)
        compiled_qoi = eqx.filter_jit(represented)
        root_problem = phx.nonlinear.ScalarRootProblem(
            lambda value, _: compiled_qoi(value) - design_target,
            bracket=bracket,
            validity=lambda value, _residual, _args: surface.realize(
                geometry.state.replace_at(radius_index, value)
            ).accepted,
            problem_id="fixed-epoch-represented-sphere-moment",
        )
        solved = phx.nonlinear.scalar_root(
            root_problem,
            method=phx.nonlinear.SafeguardedNewton(),
            termination=phx.nonlinear.NonlinearTermination(
                absolute_residual=1.0e-9, relative_residual=0.0, maximum_steps=32
            ),
        )
        root = float(solved.root)
        if (
            not bool(np.asarray(solved.successful))
            or not bool(np.asarray(solved.bracket_valid))
            or not bracket[0] < root < bracket[1]
            or abs(float(solved.value)) > 1.0e-9
        ):
            raise ValueError("The represented FE design root did not converge inside its event.")
        root_points = np.asarray(coordinates(jnp.asarray(root)), dtype=np.float64)
        independent = 0.0
        for cell in cells:
            corners = root_points[np.asarray(cell, dtype=np.int64)]
            area = 0.5 * np.linalg.norm(
                np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                axis=1,
            )
            x = corners[..., 0]
            independent += float(np.sum(area * (
                np.sum(x * x, axis=1) + x[:, 0] * x[:, 1]
                + x[:, 0] * x[:, 2] + x[:, 1] * x[:, 2]
            ) / 6.0))
        if abs(independent - design_target) > 1.0e-8:
            raise ValueError("The designed root failed its independent represented-moment oracle.")
        record["design_step"] = {
            "target": design_target,
            "initial_radius": float(radius),
            "root_radius": root,
            "event_free_bracket": list(bracket),
            "nonlinear_status": int(np.asarray(solved.status)),
            "residual": float(solved.value),
            "independent_moment": independent,
            "independent_error": abs(independent - design_target),
        }
    return record


def _fixed_epoch_margins(
    surface: phx.geometry.ImplicitSurfacePlan,
    state: DesignState,
    /,
) -> tuple[tuple[str, float], ...]:
    realization = surface.realize(state)
    values = surface.kernel.boundary_field(state, surface.grid_points)
    signed = jnp.where(surface.inside_pattern, -values, values)
    return (
        ("lattice-sign", float(jnp.min(signed)) - surface.policy.lattice_zero_tolerance),
        ("face-area", float(realization.evidence.minimum_face_area) - surface.policy.minimum_face_area),
        ("orientation", float(realization.evidence.minimum_orientation_margin)),
        ("gradient", float(realization.evidence.projection.minimum_gradient_norm) - surface.policy.projection.minimum_gradient_norm),
        ("projection-trust", 1.0 - float(realization.evidence.projection.maximum_displacement_ratio)),
    )


def _fixed_epoch_evidence(
    geometry: phx.geometry.CompiledGeometry,
    native_plan: NativeMeshingPlan,
    source: phx.meshing.CellMeshingResult,
    solve: _PhysicalSolveRecord,
    derivative: dict[str, object],
    transaction: dict[str, object],
    /,
) -> tuple[phx.meshing.FixedEpochDerivativeEvidence, tuple[tuple[str, float], ...]]:
    prepared = native_plan.prepared
    if not isinstance(prepared, _PreparedFixedImplicitRoute) or not derivative["passed"]:
        raise ValueError("Fixed-epoch evidence requires a qualified fixed plan.")
    surface = prepared.surface_plan
    realization = surface.realize(geometry.state)
    margins = _fixed_epoch_margins(surface, geometry.state)
    if not bool(np.asarray(realization.accepted)) or any(value <= 0.0 for _, value in margins):
        raise ValueError("Fixed-epoch evidence requires positive branch and regularity margins.")
    transfer_id = transaction.get("transfer_plan_id")
    if not isinstance(transfer_id, str) or not transaction.get("reanalysis_valid"):
        raise ValueError("Fixed-epoch transfer requires actual accepted transfer evidence.")
    routes = (
        ("geometry", surface.plan_id),
        ("pde", solve["problem"].compilation_id),
        ("transfer", transfer_id),
    )
    event_ids = tuple(name for name, _ in margins)
    event_margins = tuple(value for _, value in margins)
    qualifications = tuple(
        DesignQualificationEvidence(
            tier=DerivativeTier.Q2,
            analysis_plan_id=native_plan.plan_id,
            numeric_revision_id=source.result_id,
            execution_plan_id=route_id,
            event_ids=event_ids,
            event_margins=event_margins,
            valid=True,
        )
        for _, route_id in routes
    )
    evidence = phx.meshing.FixedEpochDerivativeEvidence(
        source.result_id,
        phx.geometry.ParameterId("design-sphere", "radius"),
        geometry=qualifications[0],
        pde=qualifications[1],
        transfer=qualifications[2],
        routes=routes,
    )
    return evidence, margins


def _recordable_solve(value: _PhysicalSolveRecord, /) -> dict[str, object]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"space", "problem", "solve_check", "solve_result", "values", "residual"}
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
            "source_id": "design-sphere",
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
        geometry, _, _, native_plan, source, source_record = _sphere_source(lattice_factor)
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
    current_radius = 0.75
    derivative = _derivative_oracle(0.75)
    profile["independent_derivative_oracle"] = derivative
    if not derivative["passed"]:
        failures.append(
            "independent analytic sphere moment derivative exceeded 1e-6 relative tolerance"
        )
    fixed_derivative: dict[str, object] | None = None
    try:
        fixed_derivative = _fixed_epoch_derivative(
            geometry, native_plan, source, source_solve,
            design_target=(4.0 * np.pi * 0.75**4 / 3.0 if combined_target else None),
        )
        profile["fixed_epoch_derivative"] = fixed_derivative
        if not fixed_derivative["passed"]:
            failures.append("Fixed-epoch represented FE derivative checks failed.")
        if combined_target:
            design_step = fixed_derivative["design_step"]
            if not isinstance(design_step, dict):
                raise TypeError("Combined design must retain its nonlinear root evidence.")
            current_radius = float(design_step["root_radius"])
            initial_result_id = source.result_id
            geometry, _, _, native_plan, source, source_record = _sphere_source(
                lattice_factor, radius=current_radius, fixed_plan=native_plan
            )
            if source.result_id == initial_result_id:
                raise ValueError("Designed geometry must publish a new source revision.")
            source_solve = _physical_solve(source, current_radius)
            published_qoi = source_solve["physical_oracle"]["represented_fe_qoi"]
            published_target_error = abs(published_qoi - float(design_step["target"]))
            profile["designed_republication_check"] = {
                "result_id": source.result_id,
                "represented_fe_qoi": published_qoi,
                "target": design_step["target"],
                "absolute_error": published_target_error,
                "tolerance": 1.0e-8,
            }
            if published_target_error > 1.0e-8:
                raise ValueError(
                    f"Fresh designed publication misses the represented-QoI target: {published_target_error}."
                )
            requalified = _fixed_epoch_derivative(
                geometry, native_plan, source, source_solve
            )
            if not requalified["passed"]:
                raise ValueError("Designed source did not requalify its fixed derivatives.")
            fixed_derivative = requalified
            profile["designed_state"] = {
                "root": design_step,
                "source": source_record,
                "source_result_id": source.result_id,
                "fresh_physical_reanalysis": _recordable_solve(source_solve),
                "fixed_epoch_derivative": requalified,
            }
            profile["initial_source"] = profile["source"]
            profile["initial_source_result_id"] = initial_result_id
            profile["source"] = source_record
            profile["source_result_id"] = source.result_id
            profile["source_cell_count"] = sum(
                block.cell_count for block in source.mesh.blocks
            )
            profile["source_physical_reanalysis"] = _recordable_solve(source_solve)
    except Exception as error:
        failures.append(
            f"fixed-epoch represented FE derivative: {type(error).__name__}: {error}"
        )
        if combined_target:
            return profile

    physical = source_solve["physical_oracle"]
    qoi_certificate_id = canonical_fingerprint(
        {
            "kind": "continuous-sphere-x2-moment",
            "radius": current_radius,
            "value": physical["continuous_sphere_qoi"],
        }
    )
    objective_tolerance = 0.05
    objective = phx.meshing.MeshAdaptationObjective(
        "represented-surface-reaction-L2", objective_tolerance, intent="improve"
    )
    baseline = _field_error(
        source, source_solve, current_radius, "native-source-duffy-order-10",
        quadrature_order=10,
    )
    baseline_qoi = _qoi_error(
        source, source_solve, current_radius, "native-source-duffy-order-10-qoi",
        qoi_certificate_id, quadrature_order=10,
    )
    profile["objective"] = {
        "objective_id": objective.objective_id,
        "quantity": objective.quantity,
        "intent": objective.intent,
        "qoi_certificate_id": qoi_certificate_id,
        "absolute_tolerance": objective_tolerance,
        "baseline_error_bound": baseline.bound,
        "baseline_components": {
            "field_error": baseline.field_error,
            "geometry_error": baseline.geometry_error,
            "algebraic_error": baseline.algebraic_error,
            "transfer_error": baseline.transfer_error,
        },
        "continuous_qoi_components": {
            "field_error": baseline_qoi.field_error,
            "geometry_error": baseline_qoi.geometry_error,
            "algebraic_error": baseline_qoi.algebraic_error,
        },
    }

    proposals: list[tuple[str, MeshMarkingProposal]] = []
    try:
        scope, features, analytic_scores, held_out = _proposal_features(
            source, source_solve, current_radius
        )
        safety = MeshProposalSafetyPolicy(
            source,
            minimum_size=1.0e-6,
            maximum_size=2.8,
            maximum_displacement=0.0,
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

    candidates: list[phx.meshing.MeshAdaptationCandidate] = []
    targets: dict[str, tuple[
        phx.meshing.CellMeshingResult, _PhysicalSolveRecord,
        phx.meshing.MeshAdaptationResult,
    ]] = {}
    qoi_candidates: list[phx.meshing.MeshAdaptationCandidate] = []
    if safety is not None:
        for proposal_name, proposal in proposals:
            try:
                record, candidate, target, target_solve, adaptation, qoi = _execute_proposal(
                    source,
                    source_solve,
                    proposal_name,
                    proposal,
                    safety,
                    current_radius,
                    qoi_certificate_id,
                )
                profile["proposal_attempts"].append(record)
                candidates.append(candidate)
                targets[candidate.candidate_id] = (target, target_solve, adaptation)
                qoi_candidates.append(phx.meshing.MeshAdaptationCandidate(
                    candidate.action, candidate.candidate_id, candidate.feasibility,
                    qoi, candidate.cost, candidate.solve_check,
                ))
                if proposal_name == "uniform":
                    try:
                        learned, training = _learned_from_uniform(
                            source, scope, features, held_out, target, target_solve
                        )
                        profile["learned_training"] = training
                        proposals.append(("learned", learned))
                    except Exception as error:
                        detail = f"{type(error).__name__}: {error}"
                        profile["proposal_attempts"].append(
                            {
                                "proposal": "learned",
                                "status": "failed",
                                "stage": "held-out learning",
                                "error": detail,
                            }
                        )
                        failures.append(f"learned proposal creation: {detail}")
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
    qoi_objective = phx.meshing.MeshAdaptationObjective(
        "authored-continuous-sphere-x2-moment", 1.0e-8, quantity="qoi",
        intent="improve",
    )
    qoi_decision = qoi_objective.decide(baseline_qoi, qoi_candidates)
    profile["geometry_vs_field_decision"] = {
        "field_decision_id": decision.decision_id,
        "qoi_decision_id": qoi_decision.decision_id,
        "diagnostic_only": True,
        "qoi_absolute_target": qoi_objective.tolerance,
        "baseline_field_component": baseline_qoi.field_error,
        "baseline_geometry_component": baseline_qoi.geometry_error,
        "candidate_components": [
            {
                "candidate_id": candidate.candidate_id,
                "field_error": candidate.error.field_error,
                "geometry_error": candidate.error.geometry_error,
            }
            for candidate in qoi_candidates
        ],
        "qoi_selected_candidate_id": (
            None if qoi_decision.selected is None else qoi_decision.selected.candidate_id
        ),
        "qoi_dispositions": [
            {"candidate_id": candidate_id, "issues": list(issues)}
            for candidate_id, issues in qoi_decision.dispositions
        ],
        "scope": (
            "diagnostic continuous-sphere QoI comparison, not the original "
            "0.05 field-L2 acceptance target or combined design-root check; "
            "field refinement does not remesh the analytic source geometry"
        ),
    }
    derivative_evidence = None
    derivative_margins: tuple[tuple[str, float], ...] = ()
    if decision.selected is not None:
        target, target_solve, selected_adaptation = targets[decision.selected.candidate_id]
        try:
            reanalysis_start = perf_counter()
            selected_state, _ = _stage_all_fields(
                source, source_solve, target, target_solve, selected_adaptation,
                current_radius,
            )
            profile["selected_all_field_transaction"] = selected_state
            if fixed_derivative is None:
                raise ValueError("The source fixed-epoch derivative was not qualified.")
            derivative_evidence, derivative_margins = _fixed_epoch_evidence(
                geometry, native_plan, source, source_solve, fixed_derivative, selected_state
            )
            profile["fixed_epoch_qualification"] = {
                "evidence_id": derivative_evidence.evidence_id,
                "epoch_id": derivative_evidence.epoch_id,
                "routes": list(derivative_evidence.routes),
                "event_margins": list(derivative_margins),
            }
            fresh_solve = _physical_solve(target, current_radius)
            fresh_error = _field_error(
                target, fresh_solve, current_radius, "fresh-independent-duffy-order-14",
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
            "MeshAdaptationObjective found no candidate improving to the authored field L2 target"
        )

    try:
        if safety is None:
            raise RuntimeError("Source-bound proposal safety policy is unavailable.")
        changed_radius = 0.68
        _, _, _, _, event_source, event_record = _sphere_source(
            lattice_factor, radius=changed_radius
        )
        event_solve = _physical_solve(event_source, changed_radius)
        axes = np.linspace(-1.4, 1.4, lattice_factor * 4 + 1, dtype=np.float64)
        lattice = np.stack(np.meshgrid(axes, axes, axes, indexing="ij"), axis=-1)
        initial_signs = np.sum(np.square(lattice), axis=-1) < current_radius**2
        changed_signs = np.sum(np.square(lattice), axis=-1) < changed_radius**2
        event_crossed = not np.array_equal(initial_signs, changed_signs)
        if not isinstance(native_plan.prepared, _PreparedFixedImplicitRoute):
            raise TypeError("Event invalidation requires the published fixed implicit plan.")
        surface_plan = native_plan.prepared.surface_plan
        radius_index = geometry.schema.index(
            phx.geometry.ParameterId("design-sphere", "radius")
        )
        initial_status = int(np.asarray(surface_plan.realize(geometry.state).evidence.status))
        changed_state = geometry.state.replace_at(radius_index, jnp.asarray(changed_radius))
        changed_status = int(np.asarray(surface_plan.realize(changed_state).evidence.status))
        sign_bit = int(phx.geometry.ImplicitSurfaceStatus.SIGN_PATTERN_CHANGED)
        if derivative_evidence is None or decision.selected is None:
            raise ValueError("Event invalidation requires the selected derivative evidence.")
        parameter_id = phx.geometry.ParameterId("design-sphere", "radius")
        initial_issues = derivative_evidence.invalidation_issues(
            source.result_id, parameter_id, routes=derivative_evidence.routes,
            event_margins=derivative_margins,
        )
        changed_issues = derivative_evidence.invalidation_issues(
            source.result_id, parameter_id, routes=derivative_evidence.routes,
            event_margins=_fixed_epoch_margins(surface_plan, changed_state),
            geometry_realization_id=event_source.result_id,
            topology_event_id=decision.selected.candidate_id,
        )
        if initial_issues or not {
            "event-boundary", "stopped-geometry-realization-event", "stopped-topology-event"
        }.issubset(changed_issues):
            raise ValueError("Fixed-epoch derivative invalidation omitted a stopped event.")
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
            "lattice_sign_event_scope": "published fixed-lattice surface plan",
            "fixed_plan_initial_status": initial_status,
            "fixed_plan_changed_status": changed_status,
            "initial_derivative_issues": list(initial_issues),
            "changed_derivative_issues": list(changed_issues),
            "native_event_source": event_record,
            "fresh_physical_reanalysis": _recordable_solve(event_solve),
            "stale_proposals_refused": refused,
            "required_refusals": [name for name, _ in proposals],
        }
        if (
            not event_crossed
            or initial_status != 0
            or not changed_status & sign_bit
            or event_source.result_id == source.result_id
            or not proposals
            or len(refused) != len(proposals)
        ):
            failures.append(
                "sphere lattice-event change did not refuse every source-bound proposal"
            )
    except Exception as error:
        failures.append(f"source event invalidation: {type(error).__name__}: {error}")

    profile_gaps = []
    for key, detail in (
        ("fixed_epoch_qualification", "fixed geometry/PDE/transfer derivative qualification"),
        ("selected_all_field_transaction", "selected all-field physical transaction"),
        ("fresh_physical_reanalysis", "independent selected physical reanalysis"),
        ("learned_training", "held-out learned comparator"),
        ("event_boundary_invalidation", "stopped-event invalidation"),
        ("geometry_vs_field_decision", "geometry-versus-field decision"),
    ):
        if key not in profile:
            profile_gaps.append(detail)
    if combined_target and "designed_state" not in profile:
        profile_gaps.append("combined represented-QoI design and fresh republication")
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
            raise TypeError(
                "Design profile omitted its explicit missing-contract record."
            )
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
