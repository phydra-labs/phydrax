# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite, pi
from time import perf_counter
from typing import Any, final, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import array_tree_fingerprint, canonical_fingerprint, model_array_bytes
from phydrax.discretization import (
    FiniteElementRuntimeData,
)
from phydrax.discretization.fem import FiniteElementTopologyTransfer
from phydrax.geometry import ImplicitProjectionBoundarySource
from phydrax.geometry.design import (
    DerivativeTier,
    DesignQualificationEvidence,
    DesignState,
)
from phydrax.geometry.implicit import AnalyticImplicitProfile, ImplicitSurfaceStatus
from phydrax.lifecycle import CompositionRebind
from phydrax.meshing import (
    AdaptationAction,
    DecisionBudget,
    FixedEpochDerivativeEvidence,
    LearnedMeshProposer,
    MeasuredAdaptationCost,
    mesh_proposal_scope,
    MeshProposalFeatures,
    MeshProposalSafetyPolicy,
    NativeDesignSourceState,
    NativeMeshingPhaseMeasurement,
    PhysicalErrorEvidence,
    prepare_mesh_proposal,
    RouteFeasibility,
    SolverAwareCandidate,
    SolverAwareDecision,
)
from phydrax.meshing.qualification import affine_triangle_corner_basis
from phydrax.meshing.qualification._design_oracle import design_qualification_cases
from phydrax.nonlinear import (
    NonlinearStatus,
    NonlinearTermination,
    SafeguardedNewton,
    scalar_root,
    ScalarRootProblem,
)
from phydrax.units import METER
from tools.compile_counter import CompileCounter
from tools.meshing_qualification._controls import (
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._physics import (
    _compiled_solver_stages,
    _prepare_corpus_epoch_system,
    _solve_corpus_epoch_system,
    CorpusCompilerEvidence,
    CorpusPreconditioning,
)
from tools.meshing_qualification._records import (
    _phase_recordings,
    _result_identity,
    FailureRecord,
)
from tools.meshing_qualification._runtime import (
    _measured,
    _runtime_identity,
    Deadline,
    epoch_stage_name,
    measured_peak_resident_bytes,
    peak_resident_bytes,
    PEAK_RESIDENT_SCOPE,
    qualification_failure,
    StageLedger,
    StageMeasurement,
)
from tools.meshing_qualification.scenarios.curved_native_transfer import LinearMarker
from tools.meshing_qualification.scenarios.triangulation import _simplex_measures


type _DesignQualificationReanalysis = Callable[
    [
        phx.discretization.CellMesh,
        tuple[jax.Array, ...],
        phx.equations.MaterialTransaction | None,
        object,
    ],
    phx.meshing.PhysicalErrorEvidence,
]


class _DesignQualificationReanalysisRecord(TypedDict):
    field_L2_error: float
    integral: float
    density_max_error: float
    history_max_error: float
    material_inventory: float
    material_inventory_error: float
    recomputed_solution_defect: float
    evidence_id: str
    wall_seconds: float
    assembly_compilation_count: int
    assembly_compile_seconds: float
    solve_compilation_count: int
    solve_compile_seconds: float


class _DesignQualificationTrial(TypedDict):
    candidate: phx.meshing.SolverAwareCandidate
    adaptation: phx.meshing.MeshAdaptationResult
    executor: phx.solver.FiniteElementTopologyTransaction
    proposal: str
    reanalysis: _DesignQualificationReanalysis
    problem: phx.equations.CompiledFiniteElementProblem
    scalar: phx.discretization.FiniteElementDiscretization
    values: jax.Array
    staged: phx.solver.FiniteElementTopologyResult
    physical: phx.meshing.PhysicalErrorEvidence
    all_fields: phx.discretization.FiniteElementDiscretization
    learned_proposer: phx.meshing.LearnedMeshProposer | None
    learned_features: phx.meshing.MeshProposalFeatures | None
    proposal_transaction: phx.meshing.MeshProposalTransaction | None


def _design_qualification_array(value: object) -> jax.Array:
    """Validate the scalar FE consumer's array-valued solver boundary."""
    if not isinstance(value, jax.Array):
        raise TypeError(
            f"_design_qualification_array.value must be an array-valued FE coefficient; got {type(value).__name__!r}."
        )
    return value


def _design_qualification_warm_seconds(compiler: CorpusCompilerEvidence) -> float:
    """Validate and sum the compiler's observed warm execution samples."""
    samples = compiler["warm_samples_seconds"]
    if not isinstance(samples, list):
        raise TypeError(
            f"_design_qualification_warm_seconds.samples must be a list of measured execution seconds; got {type(samples).__name__!r}."
        )
    total = 0.0
    for sample in samples:
        if not isinstance(sample, float) or not np.isfinite(sample) or sample < 0.0:
            raise ValueError(
                f"_design_qualification_warm_seconds.sample must be finite nonnegative floating-point seconds; got {sample!r}."
            )
        total += sample
    return total


def _design_qualification_source_acceptance(
    result: phx.meshing.CellMeshingResult,
    geometry: phx.geometry.CompiledGeometry,
) -> dict[str, object]:
    """Expose strict owner fidelity alongside an independent analytic sphere audit."""

    certification = result.certification
    if (
        certification is None
        or not certification.passed
        or certification.fidelity is None
    ):
        raise qualification_failure(
            "_design_qualification_source_acceptance",
            ("Native design publication lacks its required source-fidelity certificate."),
            {"certification": certification},
        )
    radius = float(
        geometry.state.values[
            geometry.schema.index(phx.geometry.ParameterId("design-sphere", "radius"))
        ]
    )
    points = np.asarray(result.mesh.coordinates)
    cells = np.asarray(result.mesh.blocks[0].vertices)
    corners = points[cells]
    centers = np.mean(corners, axis=1)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    orientation = np.sum(normals * centers, axis=1)
    if not np.all(np.isfinite(points)) or np.min(orientation) <= 0.0:
        raise qualification_failure(
            "_design_qualification_source_acceptance",
            (
                "Independent sphere audit found nonfinite coordinates or inward/degenerate facets."
            ),
            {"orientation": orientation, "points": points},
        )
    vertex_error = float(np.max(np.abs(np.linalg.norm(points, axis=1) - radius)))
    centroid_error = float(np.max(np.abs(np.linalg.norm(centers, axis=1) - radius)))
    return {
        "certification_report_id": certification.report_id,
        "source_fidelity_certificate_id": certification.fidelity.certificate_id,
        "requested": dict(certification.requested),
        "achieved": dict(certification.achieved),
        "analytic_radius": radius,
        "vertex_radial_error": vertex_error,
        "facet_centroid_radial_error": centroid_error,
        "minimum_outward_orientation": float(np.min(orientation)),
        "analytic_probe_scope": "independent samples; two-sided bounds remain with owner certificate",
    }


def _design_qualification_case(args: QualificationControls) -> dict[str, Any]:
    """Return the canonical source and workflow contract of the selected case."""
    case = design_qualification_cases().get(args.design_case)
    if case is None:
        raise ValueError(
            f"_design_qualification_case.design_case must name a registered design case; got {args.design_case!r}."
        )
    return {"name": args.design_case, **case}


def _design_qualification_lattice_count(
    args: QualificationControls, case: Mapping[str, Any]
) -> int:
    """Return the case's frozen lattice point count per axis."""
    return int(case["lattice_factor"]) * args.resolution + 1


def _design_qualification_source(
    args: QualificationControls,
    case: Mapping[str, Any],
) -> tuple[
    phx.geometry.CompiledGeometry,
    phx.meshing.NativeImplicitSource,
    phx.meshing.SurfaceMeshingSpec,
    phx.meshing.NativeMeshingOptions,
]:
    """Bind the case's fixed-lattice sphere source and immutable request budgets."""
    half_width = float(case["lattice_half_width"])
    bounds = ((-half_width,) * 3, (half_width,) * 3)
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), float(case["radius"]), feature_id="design-sphere"
    ).compile(coordinate_unit=METER)
    count = _design_qualification_lattice_count(args, case)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(np.asarray(bounds, dtype=np.float64))
    revision = canonical_fingerprint(
        {
            "kind": "design-qualification-source",
            "parameters": tuple(np.asarray(value) for value in geometry.state.values),
            "parameter_ids": tuple(str(item) for item in geometry.schema.parameter_ids),
            "lattice_count": count,
            "bounds": bounds,
        }
    )
    source = phx.meshing.NativeImplicitSource(geometry, grid, "design-sphere", revision)
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "design-sphere-boundary",
        np.asarray((0,), dtype=np.int64),
    )
    limits = capacity_meshing_limits(
        args.capacity,
        args.timeout,
        maximum_work_units=args.maximum_work_units,
        maximum_geometry_queries=args.maximum_geometry_queries,
        edges=3,
        faces=1,
        cells=1,
        connectivity_entries=12,
        data_bytes_limit=args.maximum_memory_bytes,
        scratch_bytes_limit=args.maximum_memory_bytes,
        work_units=16,
        cavity_cells=1,
        geometry_queries=64,
    )
    request = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope,
                2.0 * half_width / (count - 1),
                strength=phx.meshing.SizeControlStrength.SOFT,
            ),
        ),
        limits=limits,
    )
    options = phx.meshing.NativeMeshingOptions(
        "implicit_surface",
        implicit_policy=phx.geometry.ImplicitSurfacePolicy(
            projection=phx.geometry.ImplicitProjectionPolicy(trust_fraction=0.45),
            maximum_intersection_pairs=args.capacity * 32,
        ),
    )
    return geometry, source, request, options


def _design_qualification_load(points: jax.Array, context: object) -> jax.Array:
    del context
    return points[..., 0] ** 2


def design_field_templates() -> tuple[
    tuple[
        phx.discretization.SourceRealizationFieldSemantics,
        phx.discretization.FiniteElementFieldSpec,
    ],
    ...,
]:
    """Return the design fields and their surface-chart transfer semantics."""
    return tuple(
        (
            role,
            phx.discretization.FiniteElementFieldSpec(
                name, phx.discretization.lagrange_element("triangle", 1)
            ),
        )
        for name, role in (
            ("u", "intensive"),
            ("density", "conservative-density"),
            ("history", "intensive"),
        )
    )


def _design_qualification_prepare(
    result: phx.meshing.CellMeshingResult,
    *,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> tuple[
    phx.discretization.FiniteElementDiscretization,
    phx.discretization.FiniteElementDiscretization,
    phx.equations.CompiledFiniteElementProblem,
    tuple[phx.discretization.FiniteElementFieldSpec, ...],
]:
    """Prepare the embedded scalar reaction PDE and all declared FE fields."""
    fields = tuple(field for _, field in design_field_templates())
    scalar = phx.discretization.FiniteElementPlan(
        result.mesh, fields[0], coordinate_spec=result.geometry, budget=budget
    ).prepare(budget=budget)
    all_fields = phx.discretization.FiniteElementPlan(
        result.mesh, fields, coordinate_spec=result.geometry, budget=budget
    ).prepare(budget=budget)
    form = phx.equations.FiniteElementForm(
        "native-design-surface-reaction",
        "u",
        (
            phx.equations.MassAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    _design_qualification_load,
                    coefficient_id="ambient-x-squared-reaction",
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(form, scalar, budget=budget)
    return scalar, all_fields, problem, fields


def _design_qualification_physical_blocks(
    result: phx.meshing.CellMeshingResult,
    space: phx.discretization.FiniteElementDiscretization,
    values: jax.Array,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return affine-proved physical corners, measures and P1 values per block."""
    if any(block.cell_kind != "triangle" for block in result.mesh.blocks):
        raise ValueError(
            f"_design_qualification_physical.blocks must select triangle cell blocks; got {tuple(block.cell_kind for block in result.mesh.blocks)!r}"
        )
    coordinates = np.asarray(space.default_runtime.coordinates)
    coefficients = np.asarray(values)
    physical_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index, (element, routes) in enumerate(
        zip(space.coordinate_elements, space.coordinate_dofs, strict=True)
    ):
        basis = np.asarray(affine_triangle_corner_basis(element))
        local = coordinates[np.asarray(routes)]
        corners = np.einsum("ql,cld->cqd", basis, local)
        measures = _simplex_measures(
            corners.reshape(-1, corners.shape[-1]),
            np.arange(corners.shape[0] * 3, dtype=np.int32).reshape(-1, 3),
        )
        cell_values = coefficients[np.asarray(space.dof_maps[0].cell_dofs[index])]
        physical_blocks.append((corners, measures, cell_values))
    return physical_blocks


def _design_qualification_duffy(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return collapsed Gauss-Legendre barycentric points and reference weights."""
    axis, weight = np.polynomial.legendre.leggauss(order)
    axis, weight = (axis + 1.0) * 0.5, weight * 0.5
    first, second = np.meshgrid(axis, axis, indexing="ij")
    barycentric = np.stack(
        (1.0 - first - (1.0 - first) * second, first, (1.0 - first) * second), axis=-1
    ).reshape(-1, 3)
    weights = (np.outer(weight, weight) * (1.0 - first)).reshape(-1)
    return barycentric, weights


def _design_qualification_physical(
    result: phx.meshing.CellMeshingResult,
    space: phx.discretization.FiniteElementDiscretization,
    values: jax.Array,
    count: int,
) -> dict[str, float]:
    """Integrate the represented-surface QoI independently with host Duffy quadrature."""

    physical_blocks = _design_qualification_physical_blocks(result, space, values)

    def quadrature(order: int) -> tuple[float, float]:
        barycentric, weights = _design_qualification_duffy(order)
        squared, integral = 0.0, 0.0
        for corners, measures, cell_values in physical_blocks:
            points = np.einsum("ql,cld->cqd", barycentric, corners)
            discrete = cell_values @ barycentric.T
            physical_weights = 2.0 * measures[:, None] * weights
            squared += float(
                np.sum(physical_weights * (discrete - points[..., 0] ** 2) ** 2)
            )
            integral += float(np.sum(physical_weights * discrete))
        return float(np.sqrt(max(0.0, squared))), integral

    error, integral = quadrature(count)
    refined_error, refined_integral = quadrature(count + 2)
    exact_integral = 0.0
    for corners, measures, _ in physical_blocks:
        x = corners[..., 0]
        # Triangle moments: E(lambda_i^2)=1/6, E(lambda_i lambda_j)=1/12.
        exact_integral += float(
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
    return {
        "field_L2_error": error,
        "integral": integral,
        "exact_integral": exact_integral,
        "QoI_absolute_error": abs(integral - exact_integral),
        "L2_quadrature_refinement_delta": abs(refined_error - error),
        "QoI_quadrature_refinement_delta": abs(refined_integral - integral),
    }


def _design_qualification_cell_errors(
    result: phx.meshing.CellMeshingResult,
    space: phx.discretization.FiniteElementDiscretization,
    values: jax.Array,
    count: int,
) -> np.ndarray:
    """Return each cell's independent squared L2 error against x², in block order."""
    barycentric, weights = _design_qualification_duffy(count)
    errors = []
    for corners, measures, cell_values in _design_qualification_physical_blocks(
        result, space, values
    ):
        points = np.einsum("ql,cld->cqd", barycentric, corners)
        discrete = cell_values @ barycentric.T
        errors.append(
            np.sum(
                2.0
                * measures[:, None]
                * weights
                * (discrete - points[..., 0] ** 2) ** 2,
                axis=1,
            )
        )
    return np.concatenate(errors)


def _design_qualification_moment_gradient(
    points: np.ndarray, cells: tuple[np.ndarray, ...]
) -> tuple[float, np.ndarray]:
    """Return ∫x² over flat triangles and its closed-form vertex gradient.

    The solve-free reference uses the affine-triangle moments E(λᵢ²)=1/6 and
    E(λᵢλⱼ)=1/12 with the area gradient ½ n̂ × (opposite edge) of each corner.
    """
    total = 0.0
    gradient = np.zeros_like(points, dtype=np.float64)
    for triangles in cells:
        corners = points[triangles]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        length = np.linalg.norm(normal, axis=1)
        if not np.all(np.isfinite(length)) or np.any(length <= 0.0):
            raise ValueError(
                f"_design_qualification_moment_gradient.cells must be finite nondegenerate triangles; got minimum doubled area {float(np.min(length))!r}."
            )
        area = 0.5 * length
        unit = normal / length[:, None]
        x = corners[..., 0]
        moment = (
            np.sum(x * x, axis=1)
            + x[:, 0] * x[:, 1]
            + x[:, 0] * x[:, 2]
            + x[:, 1] * x[:, 2]
        ) / 6.0
        total += float(np.sum(area * moment))
        moment_gradient = (x + np.sum(x, axis=1, keepdims=True)) / 6.0
        for local in range(3):
            opposite = corners[:, (local + 2) % 3] - corners[:, (local + 1) % 3]
            contribution = 0.5 * np.cross(unit, opposite) * moment[:, None]
            contribution[:, 0] += area * moment_gradient[:, local]
            np.add.at(gradient, triangles[:, local], contribution)
    return total, gradient


def _design_qualification_host_qoi(
    points: np.ndarray,
    values: np.ndarray,
    cells: tuple[np.ndarray, ...],
    dofs: tuple[np.ndarray, ...],
) -> float:
    """Integrate a P1 field exactly over flat triangles on the host."""
    total = 0.0
    for triangles, cell_dofs in zip(cells, dofs, strict=True):
        corners = points[triangles]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        area = 0.5 * np.linalg.norm(normal, axis=1)
        total += float(np.sum(area * np.mean(values[cell_dofs], axis=1)))
    return total


def _design_qualification_cell_features(
    result: phx.meshing.CellMeshingResult,
) -> np.ndarray:
    """Return declared-load Hessian-weighted cell sizes, in mesh block order.

    For a flat triangle with area A, longest edge h and declared load Hessian H
    at its centroid, s = Σₑ eᵀHe/2 over its edges. The columns A·s², A·s·h² and
    A·h⁴ read the load and geometry, never an observed or exact error.
    """
    points = np.asarray(result.mesh.coordinates)
    hessian = jax.vmap(
        jax.hessian(lambda point: _design_qualification_load(point, None))
    )
    rows = []
    for block in result.mesh.blocks:
        corners = points[np.asarray(block.vertices)]
        edges = np.stack(
            (
                corners[:, 1] - corners[:, 0],
                corners[:, 2] - corners[:, 1],
                corners[:, 0] - corners[:, 2],
            ),
            axis=1,
        )
        area = 0.5 * np.linalg.norm(np.cross(edges[:, 0], -edges[:, 2]), axis=1)
        size = np.max(np.linalg.norm(edges, axis=2), axis=1)
        curvature = np.asarray(
            hessian(jax.device_put(np.mean(corners, axis=1).astype(np.float64)))
        )
        weighted = 0.5 * np.einsum("cei,cij,cej->c", edges, curvature, edges)
        rows.append(
            np.stack(
                (area * weighted**2, area * weighted * size**2, area * size**4), axis=1
            )
        )
    return np.concatenate(rows)


def _design_qualification_average_ranks(value: np.ndarray) -> np.ndarray:
    """Return zero-based ranks with tied observations sharing their mean rank."""
    order = np.argsort(value, kind="stable")
    sorted_value = value[order]
    starts = np.flatnonzero(np.r_[True, sorted_value[1:] != sorted_value[:-1]])
    ends = np.r_[starts[1:], value.size]
    ranks = np.empty(value.size, dtype=np.float64)
    ranks[order] = np.repeat(0.5 * (starts + ends - 1), ends - starts)
    return ranks


def _design_qualification_rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return the Spearman rank correlation of two equally long observations.

    Ties take their average rank: ordinal tie-breaking by cell order would
    rank equal scores and equal errors identically and inflate agreement.
    """
    if first.shape != second.shape or first.ndim != 1 or first.size < 2:
        raise ValueError(
            f"_design_qualification_rank_correlation.values must be equally long vectors of at least two entries; got shapes {(first.shape, second.shape)!r}."
        )
    ranks = [_design_qualification_average_ranks(value) for value in (first, second)]
    if np.ptp(ranks[0]) == 0.0 or np.ptp(ranks[1]) == 0.0:
        # A constant ranking carries no ordering information.
        return float("nan")
    return float(np.corrcoef(ranks[0], ranks[1])[0, 1])


def _design_qualification_surface_area(result: phx.meshing.CellMeshingResult) -> float:
    """Sum represented triangle measures across all mesh blocks."""
    coordinates = np.asarray(result.mesh.coordinates)
    return sum(
        float(np.sum(_simplex_measures(coordinates, np.asarray(block.vertices))))
        for block in result.mesh.blocks
    )


def _design_qualification_transaction(
    source: phx.meshing.CellMeshingResult,
    target: phx.meshing.CellMeshingResult,
    scalar: phx.discretization.FiniteElementDiscretization,
    all_fields: phx.discretization.FiniteElementDiscretization,
    problem: phx.equations.CompiledFiniteElementProblem,
    fields: tuple[phx.discretization.FiniteElementFieldSpec, ...],
    values: jax.Array,
) -> tuple[
    phx.solver.FiniteElementTopologyTransaction,
    _DesignQualificationReanalysis,
    list[phx.lifecycle.CompositionRebind],
    list[_DesignQualificationReanalysisRecord],
]:
    """Bind transports to a fresh physical reanalysis, sharing only compiled code.

    Each invocation reexecutes assembly and the zero-seed residual solve against
    its current problem. No trial matrix, solution, or acceptance verdict is reused.
    """

    rebinds: list[CompositionRebind] = []
    reanalyses: list[_DesignQualificationReanalysisRecord] = []
    source_area = _design_qualification_surface_area(source)
    target_area = _design_qualification_surface_area(target)
    expected_history = np.asarray(all_fields.dof_maps[2].dof_coordinates)[:, 1] + 0.5
    seed = jax.device_put(
        np.zeros((scalar.field_spaces[0].layout.size,), dtype=np.float64)
    )

    def capture(rebind: CompositionRebind) -> CompositionRebind:
        rebinds.append(rebind)
        return rebind

    def refresh(
        carried: tuple[jax.Array, ...],
        candidate: phx.meshing.MeshAdaptationResult,
        context: object,
    ) -> tuple[jax.Array, ...]:
        del candidate, context
        return (values, *carried[1:])

    def certify(
        mesh: phx.discretization.CellMesh,
        carried: tuple[jax.Array, ...],
        material: phx.equations.MaterialTransaction | None,
        lineage: phx.meshing.MeshLineage,
        context: object,
    ) -> bool:
        del mesh, material, lineage, context
        scalar_values, density_values, history_values = jax.device_get(carried)
        density_error = float(np.max(np.abs(np.asarray(density_values) - 2.0)))
        history_error = float(
            np.max(np.abs(np.asarray(history_values) - expected_history))
        )
        return max(
            density_error, history_error, abs(target_area - source_area)
        ) < 1.0e-10 and bool(np.all(np.isfinite(scalar_values)))

    def reanalysis(
        mesh: phx.discretization.CellMesh,
        carried: tuple[jax.Array, ...],
        material: phx.equations.MaterialTransaction | None,
        context: object,
    ) -> phx.meshing.PhysicalErrorEvidence:
        del mesh, material, context
        started = perf_counter()
        # Re-run both numerical kernels from the zero seed. Reuse only their
        # module-level lowering, never the trial's system, solution or verdict.
        with CompileCounter(stage="design_reanalysis_assembly") as assembly_compiler:
            epoch, _ = jax.block_until_ready(
                _prepare_corpus_epoch_system(problem, CorpusPreconditioning.NONE)
            )
        with CompileCounter(stage="design_reanalysis_solve") as solve_compiler:
            independent, solved = jax.block_until_ready(
                _solve_corpus_epoch_system(epoch, seed, None)
            )
        independent = _design_qualification_array(independent)
        if not bool(np.all(np.asarray(solved.successful))):
            raise qualification_failure(
                "reanalysis",
                ("Independent target reaction solve failed."),
                {"solved": solved},
            )
        physical = _design_qualification_physical(target, scalar, carried[0], 10)
        physical["density_max_error"] = float(jnp.max(jnp.abs(carried[1] - 2.0)))
        physical["history_max_error"] = float(
            jnp.max(
                jnp.abs(carried[2] - (all_fields.dof_maps[2].dof_coordinates[:, 1] + 0.5))
            )
        )
        inventory = 0.0
        for index, block in enumerate(target.mesh.blocks):
            measures = _simplex_measures(
                np.asarray(target.mesh.coordinates), np.asarray(block.vertices)
            )
            nodal_density = np.asarray(carried[1])[
                np.asarray(all_fields.dof_maps[1].cell_dofs[index])
            ]
            inventory += float(np.sum(measures * np.mean(nodal_density, axis=1)))
        source_inventory = 2.0 * source_area
        physical["material_inventory"] = inventory
        physical["material_inventory_error"] = abs(inventory - source_inventory)
        defect = float(jnp.max(jnp.abs(independent - carried[0])))
        if defect > 1.0e-10:
            raise qualification_failure(
                "reanalysis",
                ("Staged scalar field does not solve the independently recomputed PDE."),
                {"defect": defect},
            )
        if max(
            physical["density_max_error"], physical["history_max_error"]
        ) > 1.0e-10 or (physical["material_inventory_error"] > 1.0e-9):
            raise qualification_failure(
                "reanalysis",
                ("Independent reanalysis rejected a declared material/history field."),
                {"physical": physical},
            )
        evidence = PhysicalErrorEvidence(
            target.result_id,
            "represented-surface-reaction-L2",
            "independent-host-Duffy-10",
            field_error=physical["field_L2_error"],
            geometry_error=0.0,
            algebraic_error=0.0,
            transfer_error=0.0,
        )
        reanalyses.append(
            {
                "field_L2_error": physical["field_L2_error"],
                "integral": physical["integral"],
                "density_max_error": physical["density_max_error"],
                "history_max_error": physical["history_max_error"],
                "material_inventory": physical["material_inventory"],
                "material_inventory_error": physical["material_inventory_error"],
                "recomputed_solution_defect": defect,
                "evidence_id": evidence.evidence_id,
                "wall_seconds": perf_counter() - started,
                "assembly_compilation_count": assembly_compiler.count,
                "assembly_compile_seconds": assembly_compiler.seconds,
                "solve_compilation_count": solve_compiler.count,
                "solve_compile_seconds": solve_compiler.seconds,
            }
        )
        return evidence

    executor = phx.solver.FiniteElementTopologyTransaction(
        certify,
        fields=fields,
        field_transfer=refresh,
        composition_rebind=capture,
        callback_owners=(
            source,
            target,
            scalar,
            all_fields,
            problem,
            values,
            expected_history,
            seed,
        ),
        surface_chart_semantics={
            field.name: role for role, field in design_field_templates()
        },
    )
    return executor, reanalysis, rebinds, reanalyses


def _design_qualification_expected_refusal(
    call: Callable[[], object],
    phase: str,
) -> dict[str, object]:
    """Observe an adversarial owner refusal; unexpected admission fails the caller."""

    started = perf_counter()
    try:
        call()
    except (phx.meshing.MeshingFailure, ValueError) as failure:
        return {
            **{
                **FailureRecord.from_exception(failure, phase).to_record(),
                "status": "failed",
                "safe_refusal_is_positive": False,
            },
            "expected_refusal": True,
            "wall_seconds": perf_counter() - started,
        }
    raise qualification_failure(
        "_design_qualification_expected_refusal",
        (f"The trusted owner admitted the adversarial {phase} case."),
        {},
    )


def _design_qualification_source_transfer(
    source: phx.meshing.CellMeshingResult,
    limits: phx.meshing.MeshingLimits,
) -> phx.meshing.ImplicitAssociationTransfer:

    if source.certification is None:
        raise ValueError(
            f"_design_qualification_source_transfer.certification must contain source certification; got {source.certification!r}."
        )
    query = source.certification.request.source
    profile = (
        query.profile if isinstance(query, ImplicitProjectionBoundarySource) else query
    )
    if not isinstance(profile, AnalyticImplicitProfile):
        raise TypeError(
            f"_design_qualification_source_transfer.profile must be an AnalyticImplicitProfile; got {type(profile).__name__!r}."
        )
    return phx.meshing.ImplicitAssociationTransfer(
        profile, maximum_queries=limits.maximum_geometry_queries
    )


# Preregistered scientific trust bounds of the learned marker. A model that does
# not reproduce its observed supervision, or does not rank the held-out observed
# source errors, is refused before native admission.
_LEARNED_TRAINING_RESIDUAL_BOUND = 0.5
_LEARNED_RANK_CORRELATION_BOUND = 0.5
_LEARNED_FEATURE_IDS = (
    "area-times-load-hessian-edge-energy-squared",
    "area-times-load-hessian-edge-energy-times-diameter-squared",
    "area-times-diameter-fourth",
)


@final
class _DesignLearnedObservations(phx.StrictModule):
    """Observed solved field of a completed comparator and the source field."""

    training: phx.meshing.CellMeshingResult
    training_space: phx.discretization.FiniteElementDiscretization
    training_values: jax.Array
    source_space: phx.discretization.FiniteElementDiscretization
    source_values: jax.Array
    training_proposal: str = eqx.field(static=True)


def _design_qualification_trust_check(
    observed: float | int | bool | None, bound: float | int | bool, passed: bool
) -> dict[str, object]:
    return {"observed": observed, "bound": bound, "passed": bool(passed)}


def _design_qualification_learned(
    source: phx.meshing.CellMeshingResult,
    limits: phx.meshing.MeshingLimits,
    trial: dict[str, Any],
    observations: _DesignLearnedObservations | None,
) -> tuple[
    phx.meshing.MeshAdaptationResult,
    phx.meshing.MeshProposalTransaction,
    dict[str, float],
    phx.meshing.LearnedMeshProposer,
    phx.meshing.MeshProposalFeatures,
]:
    """Fit a marker to observed comparator errors; topology and proof stay native."""

    trial["active_phase"] = "learned_training_observation"
    if observations is None:
        raise qualification_failure(
            "_design_qualification_learned",
            (
                "The learned marker requires the completed comparator's observed solved field as supervision."
            ),
            {},
        )
    start = perf_counter()
    teacher = _design_qualification_cell_errors(
        observations.training,
        observations.training_space,
        observations.training_values,
        8,
    )
    training_features = _design_qualification_cell_features(observations.training)
    held_out = _design_qualification_cell_errors(
        source, observations.source_space, observations.source_values, 8
    )
    source_features = _design_qualification_cell_features(source)
    feature_scales = np.sqrt(np.mean(training_features**2, axis=0))
    teacher_scale = float(np.sqrt(np.mean(teacher**2)))
    if (
        not np.all(np.isfinite(feature_scales))
        or np.any(feature_scales <= 0.0)
        or not isfinite(teacher_scale)
        or teacher_scale <= 0.0
    ):
        raise qualification_failure(
            "_design_qualification_learned",
            (
                "Learned supervision requires finite positive feature and observed-error scales."
            ),
            {"feature_scales": feature_scales, "teacher_scale": teacher_scale},
        )
    phases = {"learned_training_observation": perf_counter() - start}
    trial["phase_seconds"] = phases
    trial["active_phase"] = "learned_features_safety_preparation"
    start = perf_counter()
    ids = np.concatenate([np.asarray(block.global_ids) for block in source.mesh.blocks])
    order = np.argsort(ids, kind="stable")
    scope = mesh_proposal_scope(source, source.mesh.topological_dimension)
    bound = MeshProposalFeatures(
        source,
        scope,
        (source_features / feature_scales)[order],
        feature_ids=_LEARNED_FEATURE_IDS,
        feature_owner_id="declared-load-hessian-flat-triangle-geometry",
    )
    native_policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
        limits=limits,
        compatibility=phx.meshing.BisectionCompatibility.CONFORMING_CLOSURE,
        association_transfer=_design_qualification_source_transfer(source, limits),
    )
    safety = MeshProposalSafetyPolicy(
        source,
        minimum_size=1.0e-6,
        maximum_size=2.8,
        maximum_displacement=0.0,
        maximum_marked_cells=max(1, (ids.size + 2) // 3),
        limits=limits,
    )
    phases["learned_features_safety_preparation"] = perf_counter() - start
    trial["active_phase"] = "supervised_marker_training"

    def train() -> tuple[phx.meshing.LearnedMeshProposer, np.ndarray, float, int]:
        normalized = training_features / feature_scales
        target = teacher / teacher_scale
        weights, _, rank, _ = np.linalg.lstsq(normalized, target, rcond=None)
        model = LinearMarker(jax.device_put(np.asarray(weights, dtype=np.float64)))
        proposer = LearnedMeshProposer(
            model,
            kind="marking",
            proposer_id="supervised-native-design-observed-error-marker",
        )
        residual = float(
            np.linalg.norm(normalized @ weights - target) / np.linalg.norm(target)
        )
        return proposer, weights, residual, int(rank)

    training, timing = _measured(train)
    proposer, weights, residual, rank = training
    phases["supervised_marker_training"] = timing["wall_seconds"]
    trial["training"] = {
        "algorithm": "deterministic-host-least-squares",
        "model_type": "LinearMarker",
        "teacher": "observed per-cell squared L2 error of the comparator's solved P1 field (independent Duffy-8)",
        "teacher_digest": canonical_fingerprint(teacher),
        "training_proposal": observations.training_proposal,
        "training_result_id": observations.training.result_id,
        "training_samples": int(teacher.size),
        "application_source_result_id": source.result_id,
        "application_samples": int(ids.size),
        "feature_id": bound.feature_id,
        "scope_id": scope.scope_id,
        "feature_ids": list(bound.feature_ids),
        "feature_scales": feature_scales.tolist(),
        "teacher_scale": teacher_scale,
        "weights": weights.tolist(),
        "model_id": proposer.model_id,
        "fit_rank": rank,
        "training_relative_residual": residual,
        "selection_policy": "native safety projects highest positive scores, at most ceil(cells/3)",
        "evaluation_scope": "one bounded campaign; held-out rank agreement is not generalization evidence",
    }
    trial["active_phase"] = "learned_marker_evaluation"
    raw, timing = _measured(lambda: proposer.propose(source, bound))
    phases["learned_marker_evaluation"] = timing["wall_seconds"]
    scores = np.asarray(raw.values, dtype=np.float64)
    trial["raw_proposal_id"] = raw.proposal_id
    trial["raw_proposer_evaluation_id"] = raw.proposer_id
    finite = bool(np.all(np.isfinite(scores)))
    trial["raw_scores"] = scores.tolist() if finite else None
    correlation = (
        _design_qualification_rank_correlation(scores, held_out[order])
        if finite
        else None
    )
    rank_correlation = (
        correlation if correlation is not None and isfinite(correlation) else None
    )
    trust = {
        "distinct_observed_training_mesh": _design_qualification_trust_check(
            observations.training.result_id != source.result_id,
            True,
            observations.training.result_id != source.result_id,
        ),
        "full_feature_rank": _design_qualification_trust_check(
            rank, len(_LEARNED_FEATURE_IDS), rank == len(_LEARNED_FEATURE_IDS)
        ),
        "training_relative_residual": _design_qualification_trust_check(
            residual if isfinite(residual) else None,
            _LEARNED_TRAINING_RESIDUAL_BOUND,
            isfinite(residual) and residual <= _LEARNED_TRAINING_RESIDUAL_BOUND,
        ),
        "finite_source_scores": _design_qualification_trust_check(finite, True, finite),
        "held_out_observed_error_rank_correlation": _design_qualification_trust_check(
            rank_correlation,
            _LEARNED_RANK_CORRELATION_BOUND,
            rank_correlation is not None
            and rank_correlation >= _LEARNED_RANK_CORRELATION_BOUND,
        ),
    }
    trial["trust_bounds"] = trust
    if not all(check["passed"] for check in trust.values()):
        raise qualification_failure(
            "_design_qualification_learned",
            ("The learned marker violates its preregistered scientific trust bounds."),
            {"trust": trust},
        )
    trial["active_phase"] = "trusted_learned_projection_native_certification"
    native_records: list[NativeMeshingPhaseMeasurement] = []
    try:
        trusted, timing = _measured(
            lambda: prepare_mesh_proposal(
                source,
                raw,
                safety,
                native_policy=native_policy,
                record_phase=native_records.append,
            )
        )
    finally:
        trial["trusted_native_phases"] = _phase_recordings(native_records)
    phases["trusted_learned_projection_native_certification"] = timing["wall_seconds"]
    trial["trusted_proposal"] = {
        "transaction_id": trusted.transaction_id,
        "projection_id": trusted.projection.projection_id,
        "policy_id": safety.policy_id,
        "safety_audit_id": trusted.safety_audit.report_id,
        "safety_audit_passed": trusted.safety_audit.passed,
        "compliance_issues": list(trusted.compliance.issues),
        "phase_scope": "inclusive owner interval; actual recorded inner phases must not be double-summed",
    }
    if not trusted.admissible or trusted.adaptation is None:
        raise qualification_failure(
            "_design_qualification_learned",
            ("The learned candidate failed its trusted native admission."),
            {"trusted": trusted},
        )
    if trusted.commit(source, accept=False) is not source:
        raise qualification_failure(
            "_design_qualification_learned",
            ("Explicit proposal rejection failed to retain the source."),
            {"source": source, "trusted": trusted},
        )
    trial["marked_cell_ids"] = np.asarray(trusted.projection.marked_cell_ids).tolist()
    trial["active_phase"] = "adversarial_learned_binding"
    bad = LearnedMeshProposer(
        LinearMarker(jax.device_put(np.full(weights.shape, np.nan, dtype=np.float64))),
        kind="marking",
        proposer_id="nonfinite-native-design-marker",
    )
    refusals = [
        _design_qualification_expected_refusal(
            lambda: bad.propose(source, bound), "nonfinite_learned_prediction"
        ),
        _design_qualification_expected_refusal(
            lambda: prepare_mesh_proposal(
                trusted.trusted_result, raw, safety, native_policy=native_policy
            ),
            "stale_learned_source_scope",
        ),
    ]
    trial["adversarial_refusals"] = refusals
    refusal_seconds = 0.0
    for refusal in refusals:
        seconds = refusal["wall_seconds"]
        if not isinstance(seconds, float) or not np.isfinite(seconds) or seconds < 0.0:
            raise ValueError(
                f"_design_qualification_learned.seconds must be finite nonnegative floating-point seconds; got {seconds!r}."
            )
        refusal_seconds += seconds
    phases["adversarial_learned_refusals"] = refusal_seconds
    trial["phase_seconds"] = phases
    return trusted.adaptation, trusted, dict(phases), proposer, bound


def _design_qualification_native_event(
    source: phx.meshing.CellMeshingResult,
    mode: str,
    limits: phx.meshing.MeshingLimits,
    trial: dict[str, Any],
    observations: _DesignLearnedObservations | None,
) -> tuple[
    phx.meshing.MeshAdaptationResult,
    phx.meshing.PreparedMeshAdaptation | phx.meshing.MeshProposalTransaction,
    dict[str, float],
    phx.meshing.LearnedMeshProposer | None,
    phx.meshing.MeshProposalFeatures | None,
]:
    """Execute each preregistered proposal through its trusted native owner."""

    if mode == "learned":
        return _design_qualification_learned(source, limits, trial, observations)
    trial["active_phase"] = "trusted_analytic_proposal"
    start = perf_counter()
    corners = np.asarray(source.mesh.coordinates)[
        np.asarray(source.mesh.blocks[0].vertices)
    ]
    ids = np.asarray(source.mesh.blocks[0].global_ids)
    match mode:
        case "uniform":
            marks = ids
        case "analytic":
            scores = (
                np.mean(corners[..., 0] ** 2, axis=1)
                - np.mean(corners[..., 0], axis=1) ** 2
            )
            order = np.lexsort((ids, -scores))
            marks = ids[order[: max(1, (ids.size + 1) // 2)]]
        case _:
            raise ValueError(
                f"_design_qualification_native_event.mode must be 'uniform', 'analytic' or 'learned'; got {mode!r}."
            )
    phases = {"proposal": perf_counter() - start}
    trial["marked_cell_ids"] = marks.tolist()
    trial["phase_seconds"] = phases
    trial["active_phase"] = "native_adaptation_preparation"
    prepared, timing = _measured(
        lambda: phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(marks),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
                limits=limits,
                compatibility=phx.meshing.BisectionCompatibility.CONFORMING_CLOSURE,
                association_transfer=_design_qualification_source_transfer(
                    source, limits
                ),
            ),
            observe_managed_admission=True,
        )
    )
    trial["prepared_id"] = prepared.prepared_id
    phases["adaptation_preparation"] = timing["wall_seconds"]
    trial["active_phase"] = "native_event"
    adaptation, timing = _measured(
        lambda: phx.meshing.execute_mesh_adaptation(
            prepared, observe_managed_admission=True
        )
    )
    phases["native_event"] = timing["wall_seconds"]
    return adaptation, prepared, dict(phases), None, None


def _design_qualification_changed_load(points: jax.Array, context: object) -> jax.Array:
    del context
    return 2.0 * points[..., 0] ** 2


def _design_qualification_rollback(
    source: phx.meshing.CellMeshingResult,
    accepted: phx.solver.FiniteElementAcceptedState,
    adaptation: phx.meshing.MeshAdaptationResult,
    scalar: phx.discretization.FiniteElementDiscretization,
    fields: tuple[phx.discretization.FiniteElementFieldSpec, ...],
    values: jax.Array,
) -> dict[str, object]:
    """Change the PDE load and refuse all staged learned-route state atomically."""

    changed = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "native-design-altered-load-rollback",
            "u",
            (
                phx.equations.MassAction("u", 1.0),
                phx.equations.SourceAction(
                    "u",
                    phx.equations.coefficient(
                        _design_qualification_changed_load,
                        coefficient_id="altered-double-x-squared-load",
                    ),
                ),
            ),
        ),
        scalar,
    )
    diagnostics: dict[str, object] = {
        "experiment": "actual doubled scalar load; all accepted owners must remain unchanged",
        "changed_compilation_id": changed.compilation_id,
    }

    def refresh(
        carried: tuple[jax.Array, ...],
        candidate: phx.meshing.MeshAdaptationResult,
        context: object,
    ) -> tuple[jax.Array, ...]:
        del candidate, context
        return (values, *carried[1:])

    def reanalyze(
        mesh: phx.discretization.CellMesh,
        carried: tuple[jax.Array, ...],
        material: phx.equations.MaterialTransaction | None,
        lineage: phx.meshing.MeshLineage,
        context: object,
    ) -> bool:
        del mesh, material, lineage, context
        with CompileCounter(stage="design_rollback_assembly") as assembly_compiler:
            epoch, _ = jax.block_until_ready(
                _prepare_corpus_epoch_system(changed, CorpusPreconditioning.NONE)
            )
        with CompileCounter(stage="design_rollback_solve") as solve_compiler:
            independent, solved = jax.block_until_ready(
                _solve_corpus_epoch_system(epoch, jnp.zeros_like(values))
            )
        independent = _design_qualification_array(independent)
        if not bool(jnp.all(solved.successful)):
            raise qualification_failure(
                "reanalyze",
                ("The altered-load rollback oracle did not converge."),
                {"solved": solved},
            )
        defect = float(jnp.max(jnp.abs(independent - carried[0])))
        diagnostics.update(
            {
                "actual_staged_solution_defect": defect,
                "changed_solver_status": np.asarray(solved.status).tolist(),
                "changed_solver_residual": np.asarray(
                    solved.diagnostics.residual_norm
                ).tolist(),
                "assembly_compilation_count": assembly_compiler.count,
                "assembly_compile_seconds": assembly_compiler.seconds,
                "solve_compilation_count": solve_compiler.count,
                "solve_compile_seconds": solve_compiler.seconds,
            }
        )
        return defect <= 1.0e-10

    executor = phx.solver.FiniteElementTopologyTransaction(
        reanalyze,
        fields=fields,
        field_transfer=refresh,
        surface_chart_semantics={
            field.name: role for role, field in design_field_templates()
        },
    )
    refused = executor.execute(accepted, source.mesh, adaptation)
    retained = {
        "committed": bool(refused.committed),
        "source_state_identically_retained": refused.state is accepted,
        "source_mesh_identically_retained": refused.mesh is source.mesh,
        "receipt_present": refused.receipt is not None,
        "receipt_published": refused.receipt is not None and refused.receipt.published,
        "accepted_state_id": accepted.accepted_id,
    }
    if (
        retained["committed"]
        or not retained["source_state_identically_retained"]
        or not retained["source_mesh_identically_retained"]
        or refused.receipt is None
        or refused.receipt.published
    ):
        raise qualification_failure(
            "_design_qualification_rollback",
            (
                "Failed physical reanalysis did not retain identical scalar/material/history owners."
            ),
            {"accepted": accepted, "refused": refused, "source": source},
        )
    diagnostics.update(
        {
            **retained,
            "receipt_id": refused.receipt.receipt_id,
            "diagnostics": refused.diagnostics,
        }
    )
    return diagnostics


def _design_qualification_admit_learner(
    source: phx.meshing.CellMeshingResult,
    accepted: phx.solver.FiniteElementAcceptedState,
    adaptation: phx.meshing.MeshAdaptationResult,
    executor: phx.solver.FiniteElementTopologyTransaction,
    problem: phx.equations.CompiledFiniteElementProblem,
    candidate: phx.meshing.SolverAwareCandidate,
    baseline: phx.meshing.PhysicalErrorEvidence,
    budget: phx.meshing.DecisionBudget,
    reanalysis: _DesignQualificationReanalysis,
    reanalyses: list[_DesignQualificationReanalysisRecord],
    deadline: float,
) -> tuple[phx.meshing.SolverAwareCandidate, dict[str, Any]]:
    """Bind the learned route to its own accepted physical decision."""

    if perf_counter() >= deadline:
        raise qualification_failure(
            "_design_qualification_admit_learner",
            ("The campaign wall budget expired before learned decision admission."),
            {"deadline": deadline, "perf_counter": perf_counter},
        )
    decision, selection = _measured(
        lambda: SolverAwareDecision(baseline, budget, (candidate,))
    )
    if decision.selected is None:
        raise qualification_failure(
            "_design_qualification_admit_learner",
            (
                "The learned native candidate did not independently attain the frozen objective/budgets."
            ),
            {"decision": decision},
        )
    before = sum(row["wall_seconds"] for row in reanalyses)
    if perf_counter() >= deadline:
        raise qualification_failure(
            "_design_qualification_admit_learner",
            ("The campaign wall budget expired before learned state publication."),
            {"deadline": deadline, "perf_counter": perf_counter},
        )
    committed, timing = _measured(
        lambda: executor.execute(
            accepted,
            source.mesh,
            adaptation,
            decision=decision,
            reanalysis=reanalysis,
            compiled_layout_id=problem.compilation_id,
        )
    )
    if (
        not bool(committed.committed)
        or committed.receipt is None
        or not committed.receipt.published
    ):
        raise qualification_failure(
            "_design_qualification_admit_learner",
            (committed.diagnostics),
            {"committed": committed},
        )
    extra_reanalysis = sum(row["wall_seconds"] for row in reanalyses) - before
    extra_transfer = timing["wall_seconds"] - extra_reanalysis
    if extra_transfer < 0.0:
        raise qualification_failure(
            "_design_qualification_admit_learner",
            ("Observed learned acceptance intervals have inconsistent containment."),
            {"extra_transfer": extra_transfer},
        )
    prior = candidate.cost.phase_seconds
    cost = MeasuredAdaptationCost(
        adaptation.result_id,
        candidate.candidate_id,
        preparation_seconds=prior[0],
        compilation_seconds=prior[1],
        solve_seconds=prior[2],
        transfer_seconds=prior[3] + extra_transfer,
        reanalysis_seconds=prior[4] + extra_reanalysis,
        decision_seconds=prior[5] + selection["wall_seconds"],
        peak_memory_bytes=measured_peak_resident_bytes(),
    )
    updated = SolverAwareCandidate(
        candidate.action,
        candidate.candidate_id,
        candidate.feasibility,
        candidate.error,
        cost,
    )
    return updated, {
        "decision_id": decision.decision_id,
        "source_accepted_state_id": accepted.accepted_id,
        "accepted_state_id": committed.state.accepted_id,
        "receipt_id": committed.receipt.receipt_id,
        "receipt_published": committed.receipt.published,
        "actual_reanalysis": dict(reanalyses[-1]),
        "selection_seconds": selection["wall_seconds"],
        "inclusive_acceptance_seconds": timing["wall_seconds"],
        "contained_reanalysis_seconds": extra_reanalysis,
        "transfer_rebind_excluding_reanalysis_seconds": extra_transfer,
    }


def _design_qualification_observed_cost(
    candidate: phx.meshing.SolverAwareCandidate,
    elapsed: float,
) -> tuple[phx.meshing.SolverAwareCandidate, float]:
    """Price the observed trial interval, including unattributed caller work."""

    prior = candidate.cost
    orchestration = elapsed - prior.total_seconds
    if orchestration < 0.0:
        raise qualification_failure(
            "_design_qualification_observed_cost",
            (
                "Recorded candidate intervals overlap or exceed the observed trial interval."
            ),
            {"orchestration": orchestration},
        )
    phases = prior.phase_seconds
    cost = MeasuredAdaptationCost(
        prior.sample_id,
        candidate.candidate_id,
        preparation_seconds=phases[0] + orchestration,
        compilation_seconds=phases[1],
        solve_seconds=phases[2],
        transfer_seconds=phases[3],
        reanalysis_seconds=phases[4],
        decision_seconds=phases[5],
        peak_memory_bytes=measured_peak_resident_bytes(),
    )
    return SolverAwareCandidate(
        candidate.action,
        candidate.candidate_id,
        candidate.feasibility,
        candidate.error,
        cost,
    ), orchestration


def _design_qualification_comparison(rows: list[dict[str, Any]]) -> dict[str, object]:
    """Report observed bounded differences and regressions without declaring superiority."""
    learned = next(row for row in rows if row["proposal"] == "learned")
    comparisons = []
    for baseline in rows:
        if baseline["proposal"] == "learned" or baseline["status"] != "completed":
            continue
        time_difference = (
            learned["observed_attempt_seconds"] - baseline["observed_attempt_seconds"]
        )
        error_difference = (
            learned["physical"]["field_L2_error"] - baseline["physical"]["field_L2_error"]
        )
        memory_difference = (
            learned["peak_process_resident_bytes"]
            - baseline["peak_process_resident_bytes"]
        )
        comparisons.append(
            {
                "baseline": baseline["proposal"],
                "baseline_candidate_id": baseline["candidate_id"],
                "learned_candidate_id": learned["candidate_id"],
                "observed_cost_difference_seconds": time_difference,
                "priced_route_difference_seconds": learned["measured_cost_seconds"]
                - baseline["measured_cost_seconds"],
                "observed_L2_error_difference": error_difference,
                "observed_process_peak_difference_bytes": memory_difference,
                "observed_dof_difference": learned["dofs"] - baseline["dofs"],
                "regressions": [
                    name
                    for name, difference in (
                        ("cost", time_difference),
                        ("physical_error", error_difference),
                        ("process_peak", memory_difference),
                    )
                    if difference > 0.0
                ],
            }
        )
    return {
        "totals": [
            {
                "proposal": row["proposal"],
                "status": row["status"],
                "observed_attempt_seconds": row["observed_attempt_seconds"],
                **(
                    {
                        name: row[name]
                        for name in (
                            "measured_cost_seconds",
                            "dofs",
                            "condition",
                            "peak_process_resident_bytes",
                            "physical",
                        )
                    }
                    if row["status"] == "completed"
                    else {}
                ),
            }
            for row in rows
        ],
        "uncertainty": {
            "physical": "Reported quadrature-refinement deltas are numerical diagnostics, not rigorous error bounds.",
            "cost": "Single ordered campaign; sampling uncertainty and cache/order effects are not estimated.",
            "memory": "Shared process-lifetime peaks cannot isolate per-route allocation.",
        },
        "comparisons": comparisons,
        "failed_baselines": [
            row["proposal"]
            for row in rows
            if row["proposal"] != "learned" and row["status"] != "completed"
        ],
        "scope": "same actual source/objective/limits; bounded observed routes, no generalization/leadership",
        "cost_scope": "complete observed trials include training, trust, solve, transfer, orchestration and adversarial work",
        "memory_scope": "process-lifetime peak, not independently isolated per-route allocation",
    }


@final
class _DesignTrialAnalysis(phx.StrictModule):
    scalar: phx.discretization.FiniteElementDiscretization
    all_fields: phx.discretization.FiniteElementDiscretization
    problem: phx.equations.CompiledFiniteElementProblem
    fields: tuple[phx.discretization.FiniteElementFieldSpec, ...]
    values: jax.Array
    compiler: CorpusCompilerEvidence
    solver_stages: dict[str, float]
    preparation: StageMeasurement
    condition: float
    condition_stage: StageMeasurement
    error: dict[str, float]
    error_stage: StageMeasurement


@final
class _DesignTrialStaging(phx.StrictModule):
    executor: phx.solver.FiniteElementTopologyTransaction
    reanalysis: _DesignQualificationReanalysis
    rebind: phx.lifecycle.CompositionRebind
    reanalyses: list[_DesignQualificationReanalysisRecord]
    staged: phx.solver.FiniteElementTopologyResult
    receipt_id: str = eqx.field(static=True)
    actual: phx.meshing.PhysicalErrorEvidence
    transfer_stage: StageMeasurement
    reanalysis_seconds: float
    rollback_seconds: float


@final
class _DesignTrialPricing(phx.StrictModule):
    candidate: phx.meshing.SolverAwareCandidate
    feasibility: phx.meshing.RouteFeasibility
    phases: dict[str, float]


@eqx.filter_jit
def _design_qualification_condition_matrix(
    problem: phx.equations.CompiledFiniteElementProblem,
    basis: jax.Array,
    /,
) -> jax.Array:
    """Materialize the condition oracle in the declared FE state-capacity bucket."""
    system, _ = problem.linear_system()

    def scalar_action(vector: jax.Array) -> jax.Array:
        return _design_qualification_array(system.operator.mv(vector))

    return jax.vmap(scalar_action)(basis).T


def _design_qualification_analyze_trial(
    target: phx.meshing.CellMeshingResult,
    args: QualificationControls,
    trial: dict[str, Any],
) -> _DesignTrialAnalysis:
    """Solve the candidate and measure its operator condition and physical error."""

    trial["target"] = _result_identity(target)
    trial["active_phase"] = "target_solver_preparation"
    bundle, preparation = _measured(lambda: _design_qualification_prepare(target))
    scalar, all_fields, problem, fields = bundle
    trial["phase_seconds"]["solver_preparation"] = preparation["wall_seconds"]
    seed = jax.device_put(
        np.zeros((scalar.field_spaces[0].layout.size,), dtype=np.float64)
    )
    trial["active_phase"] = "target_solver"
    values, compiler, solver_stages = _compiled_solver_stages(problem, seed, args.repeats)
    values = _design_qualification_array(values)
    trial["compiler"] = compiler
    trial["phase_seconds"].update(solver_stages)
    trial["active_phase"] = "actual_operator_condition"
    condition, condition_stage = _measured(
        lambda: _design_qualification_operator_condition(problem, seed.size)
    )
    trial["phase_seconds"]["actual_operator_condition"] = condition_stage["wall_seconds"]
    trial["active_phase"] = "independent_candidate_physical_error"
    error, error_stage = _measured(
        lambda: _design_qualification_physical(target, scalar, values, 8)
    )
    trial["phase_seconds"]["physical_error"] = error_stage["wall_seconds"]
    return _DesignTrialAnalysis(
        scalar=scalar,
        all_fields=all_fields,
        problem=problem,
        fields=fields,
        values=values,
        compiler=compiler,
        solver_stages=solver_stages,
        preparation=preparation,
        condition=condition,
        condition_stage=condition_stage,
        error=error,
        error_stage=error_stage,
    )


def _design_qualification_stage_trial(
    source: phx.meshing.CellMeshingResult,
    accepted: phx.solver.FiniteElementAcceptedState,
    adaptation: phx.meshing.MeshAdaptationResult,
    analysis: _DesignTrialAnalysis,
    mode: str,
    trial: dict[str, Any],
    deadline: float,
) -> _DesignTrialStaging:
    """Stage all fields atomically, reanalyze them, and exercise learned rollback."""

    target = adaptation.target
    scalar, all_fields, problem, fields, values = (
        analysis.scalar,
        analysis.all_fields,
        analysis.problem,
        analysis.fields,
        analysis.values,
    )
    error_stage = analysis.error_stage
    executor, reanalysis, rebinds, reanalyses = _design_qualification_transaction(
        source, target, scalar, all_fields, problem, fields, values
    )
    trial["active_phase"] = "canonical_all_field_transfer"
    if perf_counter() >= deadline:
        raise qualification_failure(
            "_design_qualification_trial",
            ("The campaign wall budget expired before all-field staging."),
            {"deadline": deadline, "perf_counter": perf_counter},
        )
    staged, transfer_stage = _measured(
        lambda: executor.execute(
            accepted, source.mesh, adaptation, compiled_layout_id=problem.compilation_id
        )
    )
    receipt = staged.receipt
    if not bool(staged.committed) or receipt is None or not receipt.published:
        raise qualification_failure(
            "_design_qualification_trial", (staged.diagnostics), {"staged": staged}
        )
    actual = reanalysis(target.mesh, staged.state.fields, None, None)
    reanalysis_seconds = error_stage["wall_seconds"] + sum(
        row["wall_seconds"] for row in reanalyses
    )
    trial["active_phase"] = "solver_aware_feasibility"
    rollback_seconds = 0.0
    if mode == "learned":
        trial["active_phase"] = "failed_physical_reanalysis_rollback"
        rollback, rollback_stage = _measured(
            lambda: _design_qualification_rollback(
                source, accepted, adaptation, scalar, fields, values
            )
        )
        trial["physical_reanalysis_rollback"] = rollback
        rollback_seconds = rollback_stage["wall_seconds"]
        reanalysis_seconds += rollback_seconds
        trial["active_phase"] = "solver_aware_feasibility"
    return _DesignTrialStaging(
        executor=executor,
        reanalysis=reanalysis,
        rebind=rebinds[-1],
        reanalyses=reanalyses,
        staged=staged,
        receipt_id=receipt.receipt_id,
        actual=actual,
        transfer_stage=transfer_stage,
        reanalysis_seconds=reanalysis_seconds,
        rollback_seconds=rollback_seconds,
    )


def _design_qualification_price_trial(
    source: phx.meshing.CellMeshingResult,
    adaptation: phx.meshing.MeshAdaptationResult,
    analysis: _DesignTrialAnalysis,
    staging: _DesignTrialStaging,
    native_phases: dict[str, float],
) -> _DesignTrialPricing:
    """Bind route feasibility to measured preparation, solve, and transport costs."""

    target = adaptation.target
    all_fields, problem = analysis.all_fields, analysis.problem
    preparation, condition_stage, error_stage = (
        analysis.preparation,
        analysis.condition_stage,
        analysis.error_stage,
    )
    condition, error = analysis.condition, analysis.error
    compiler, solver_stages = analysis.compiler, analysis.solver_stages
    transfer_stage = staging.transfer_stage
    reanalysis_seconds, rollback_seconds = (
        staging.reanalysis_seconds,
        staging.rollback_seconds,
    )
    feasibility_start = perf_counter()
    rebind = staging.rebind
    feasibility = RouteFeasibility(
        source.result_id,
        target.result_id,
        adaptation.route.value,
        adaptation.result_id,
        cell_families=tuple(block.cell_kind for block in target.mesh.blocks),
        geometry_layout_id=target.geometry.geometry_layout_id,
        field_layouts=tuple(
            (space.name, space.layout.layout_id) for space in all_fields.field_spaces
        ),
        compiled_layout_id=problem.compilation_id,
        geometry_certificate_id=target.audit.report_id,
        topology_certificate_id=target.audit.report_id,
        required_state_ids=rebind.source.entry_ids,
        state_dispositions=RouteFeasibility.rebind_dispositions(rebind),
        dofs=sum(space.layout.size for space in all_fields.field_spaces),
        condition_estimate=condition,
    )
    phases = {
        **native_phases,
        "solver_preparation": preparation["wall_seconds"],
        "condition": condition_stage["wall_seconds"],
        **solver_stages,
        "all_field_transfer": transfer_stage["wall_seconds"],
        "reanalysis": reanalysis_seconds,
        "physical_error_included_in_reanalysis": error_stage["wall_seconds"],
        "rollback_included_in_reanalysis": rollback_seconds,
    }
    cost = MeasuredAdaptationCost(
        adaptation.result_id,
        adaptation.result_id,
        preparation_seconds=sum(native_phases.values())
        + preparation["wall_seconds"]
        + condition_stage["wall_seconds"]
        + solver_stages["solver_assembly"],
        compilation_seconds=solver_stages["solver_lowering"]
        + solver_stages["solver_compile"],
        solve_seconds=solver_stages["solver_first_execution"]
        + _design_qualification_warm_seconds(compiler),
        transfer_seconds=transfer_stage["wall_seconds"],
        reanalysis_seconds=reanalysis_seconds,
        decision_seconds=perf_counter() - feasibility_start,
        peak_memory_bytes=measured_peak_resident_bytes(),
    )
    candidate = SolverAwareCandidate(
        AdaptationAction.H,
        adaptation.result_id,
        feasibility,
        PhysicalErrorEvidence(
            target.result_id,
            "represented-surface-reaction-L2",
            "independent-host-Duffy-8",
            field_error=error["field_L2_error"],
            geometry_error=0.0,
            algebraic_error=0.0,
            transfer_error=0.0,
        ),
        cost,
    )
    return _DesignTrialPricing(
        candidate=candidate, feasibility=feasibility, phases=phases
    )


def _design_qualification_trial(
    source: phx.meshing.CellMeshingResult,
    accepted: phx.solver.FiniteElementAcceptedState,
    mode: str,
    args: QualificationControls,
    limits: phx.meshing.MeshingLimits,
    trial: dict[str, Any],
    baseline: phx.meshing.PhysicalErrorEvidence,
    budget: phx.meshing.DecisionBudget,
    deadline: float,
    observations: _DesignLearnedObservations | None,
) -> _DesignQualificationTrial:
    """Measure a real native event and all-field transaction before selection."""

    if perf_counter() >= deadline:
        raise qualification_failure(
            "_design_qualification_trial",
            ("The campaign wall budget expired before this preregistered trial."),
            {"deadline": deadline, "perf_counter": perf_counter},
        )
    adaptation, prepared, native_phases, learned_proposer, learned_features = (
        _design_qualification_native_event(source, mode, limits, trial, observations)
    )
    if not adaptation.status.converged or adaptation.transition is None:
        raise qualification_failure(
            "_design_qualification_trial",
            (f"Native positive adaptation was not accepted: {adaptation.status.value}."),
            {"adaptation": adaptation},
        )
    target = adaptation.target
    analysis = _design_qualification_analyze_trial(target, args, trial)
    staging = _design_qualification_stage_trial(
        source, accepted, adaptation, analysis, mode, trial, deadline
    )
    pricing = _design_qualification_price_trial(
        source, adaptation, analysis, staging, native_phases
    )
    candidate, feasibility, phases = (
        pricing.candidate,
        pricing.feasibility,
        pricing.phases,
    )
    cost = candidate.cost
    executor, reanalysis, reanalyses = (
        staging.executor,
        staging.reanalysis,
        staging.reanalyses,
    )
    scalar, all_fields, problem, values = (
        analysis.scalar,
        analysis.all_fields,
        analysis.problem,
        analysis.values,
    )
    error, condition = analysis.error, analysis.condition
    staged, actual, rebind = staging.staged, staging.actual, staging.rebind
    if mode == "learned":
        trial["active_phase"] = "solver_aware_learned_acceptance"
        candidate, admission = _design_qualification_admit_learner(
            source,
            accepted,
            adaptation,
            executor,
            problem,
            candidate,
            baseline,
            budget,
            reanalysis,
            reanalyses,
            deadline,
        )
        cost = candidate.cost
        trial["learned_acceptance"] = admission
        phases["learned_selection"] = admission["selection_seconds"]
        phases["learned_acceptance_including_reanalysis"] = admission[
            "inclusive_acceptance_seconds"
        ]
        phases["learned_acceptance_contained_reanalysis"] = admission[
            "contained_reanalysis_seconds"
        ]
    trial.update(
        {
            "status": "completed",
            "candidate_id": candidate.candidate_id,
            "physical": error,
            "reanalyses": reanalyses,
            "phase_seconds": phases,
            "measured_cost_seconds": cost.total_seconds,
            "retained_bytes": model_array_bytes((prepared, adaptation, staged.transfers)),
            "peak_process_resident_bytes": cost.peak_memory_bytes,
            "state_entries": list(rebind.source.entry_ids),
            "transfer_ids": [item.transfer_id for item in staged.transfers],
            "dofs": feasibility.dofs,
            "feasibility_issues": list(feasibility.issues),
            "native_event": {
                "adaptation_result_id": adaptation.result_id,
                "route": adaptation.route.value,
                "status": adaptation.status.value,
                "converged": bool(adaptation.status.converged),
                "transition_present": adaptation.transition is not None,
                "source_cells": int(source.audit.entity_counts[-1]),
                "target_cells": int(adaptation.target.audit.entity_counts[-1]),
            },
            "feasibility_id": feasibility.feasibility_id,
            "physical_evidence_id": candidate.error.evidence_id,
            "cost_id": cost.cost_id,
            "trial_receipt": staging.receipt_id,
            "condition": condition,
        }
    )
    return {
        "candidate": candidate,
        "adaptation": adaptation,
        "executor": executor,
        "proposal": mode,
        "reanalysis": reanalysis,
        "problem": problem,
        "scalar": scalar,
        "values": values,
        "staged": staged,
        "physical": actual,
        "all_fields": all_fields,
        "learned_proposer": learned_proposer,
        "learned_features": learned_features,
        "proposal_transaction": prepared
        if isinstance(prepared, phx.meshing.MeshProposalTransaction)
        else None,
    }


@eqx.filter_jit
def _design_qualification_coordinates(
    surface_plan: phx.geometry.ImplicitSurfacePlan,
    state: DesignState,
    radius_index: int,
    radius: jax.Array,
) -> jax.Array:
    return surface_plan.realize(state.replace_at(radius_index, radius)).vertices


@eqx.filter_jit
def _design_qualification_solve_coordinates(
    problem: phx.equations.CompiledFiniteElementProblem,
    mesh: phx.discretization.CellMesh,
    geometry_layout_id: str,
    solve_policy: phx.linalg.LinearSolvePolicy,
    points: jax.Array,
) -> tuple[jax.Array, phx.linalg.LinearSolveResult]:
    runtime = FiniteElementRuntimeData(
        mesh,
        points,
        numeric_version=mesh.numeric_version,
        geometry_layout_id=geometry_layout_id,
    )
    context = phx.equations.FiniteElementExecutionContext(runtime)
    system, rhs = problem.linear_system(context)
    solved = phx.linalg.solve(system, rhs, policy=solve_policy)
    return _design_qualification_array(problem.expand(solved.value, context)), solved


def _design_qualification_pde(
    solve_coordinates: Callable[
        [jax.Array], tuple[jax.Array, phx.linalg.LinearSolveResult]
    ],
    coordinates: Callable[[jax.Array], jax.Array],
    radius: jax.Array,
) -> jax.Array:
    return solve_coordinates(coordinates(radius))[0]


@eqx.filter_jit
def _design_qualification_linearizations(
    coordinates: Callable[[jax.Array], jax.Array],
    pde: Callable[[jax.Array], jax.Array],
    parameter: jax.Array,
) -> tuple[phx.linalg.PreparedLinearization, phx.linalg.PreparedLinearization]:
    """Retain the actual geometry/PDE tapes for their fixed accepted parameter."""
    return (
        phx.linalg.prepare_linearization(coordinates, parameter),
        phx.linalg.prepare_linearization(pde, parameter),
    )


@eqx.filter_jit
def _design_qualification_transfer_linearization(
    transfer: FiniteElementTopologyTransfer,
    values: jax.Array,
) -> phx.linalg.PreparedLinearization:
    return phx.linalg.prepare_linearization(transfer.apply, values)


@eqx.filter_jit
def _design_qualification_tape_jvp(
    linearization: phx.linalg.PreparedLinearization, tangent: jax.Array
) -> jax.Array:
    """Compile one retained action, independently of its downstream consumer."""
    return _design_qualification_array(linearization.jvp(tangent))


@eqx.filter_jit
def _design_qualification_tape_vjp(
    linearization: phx.linalg.PreparedLinearization, cotangent: jax.Array
) -> jax.Array:
    return _design_qualification_array(linearization.vjp(cotangent))


@eqx.filter_jit
def _design_qualification_derivative_residuals(
    tangent: jax.Array,
    cotangent: jax.Array,
    direction: jax.Array,
    pullback: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    step: float,
) -> tuple[jax.Array, jax.Array]:
    duality = jnp.abs(jnp.vdot(tangent, cotangent) - direction * pullback)
    central = (upper - lower) / (2.0 * step)
    return jnp.max(jnp.abs(tangent - central)), duality


def _design_qualification_derivative_values(
    linearizations: tuple[phx.linalg.PreparedLinearization, ...],
    direction: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    step: float,
) -> tuple[jax.Array, jax.Array]:
    # Dispatch the same compiled actions for FD, adjoint and transfer probes.
    # Compiling their composition here would duplicate the entire PDE tape
    # for each consumer despite retaining the correct linearization.
    value = _design_qualification_array(linearizations[-1].primal)
    tangent = direction
    for linearization in linearizations:
        tangent = _design_qualification_tape_jvp(linearization, tangent)
    cotangent = jnp.linspace(-0.3, 0.8, value.size, dtype=jnp.float64).reshape(
        value.shape
    )
    pullback = cotangent
    for linearization in reversed(linearizations):
        pullback = _design_qualification_tape_vjp(linearization, pullback)
    return _design_qualification_derivative_residuals(
        tangent, cotangent, direction, pullback, lower, upper, step
    )


def _design_qualification_derivative_probe(
    linearizations: tuple[phx.linalg.PreparedLinearization, ...],
    direction: jax.Array,
    lower: jax.Array | np.ndarray,
    upper: jax.Array | np.ndarray,
    step: float,
) -> dict[str, float]:
    """Check one fixed-route derivative against its independently solved perturbations."""

    discrepancy_array, duality_array = _design_qualification_derivative_values(
        linearizations, direction, jnp.asarray(lower), jnp.asarray(upper), step
    )
    discrepancy, duality = float(discrepancy_array), float(duality_array)
    if (
        not np.isfinite(discrepancy + duality)
        or discrepancy > _DERIVATIVE_FD_BOUND
        or duality > _DERIVATIVE_DUALITY_BOUND
    ):
        raise qualification_failure(
            "_design_qualification_derivative_probe",
            (
                f"Native fixed-epoch derivative failed: FD={discrepancy}, duality={duality}."
            ),
            {"discrepancy": discrepancy, "duality": duality},
        )
    return {"finite_difference_error": discrepancy, "JVP_VJP_duality_error": duality}


# Preregistered fixed-epoch derivative contracts. Every probe stays inside the
# qualified event margins; crossings are refused by the realization owner and
# invalidate derivatives instead of being smoothed.
_DERIVATIVE_FD_BOUND = 2.0e-6
_DERIVATIVE_DUALITY_BOUND = 2.0e-8
_DERIVATIVE_STEP = 1.0e-5
_DERIVATIVE_DIRECTION = 0.7
_TAYLOR_FRACTIONS = (2.0**-3, 2.0**-4, 2.0**-5)
_TAYLOR_SLOPE_DEVIATION_BOUND = 0.2
# Worst-case host summation roundoff (cells·eps·|Q|) must stay three orders of
# magnitude below the smallest observed Taylor remainder.
_TAYLOR_ROUNDOFF_RATIO_BOUND = 1.0e-3
# Central-difference probes stay nearer the epoch than the smallest Taylor step.
_DERIVATIVE_STEP_EVENT_FRACTION_BOUND = _TAYLOR_FRACTIONS[-1]
_EVENT_PROBE_FRACTION = 1.0e-3
_DESIGN_BRACKET_FRACTION = 0.5
_DESIGN_QOI_RESIDUAL = 1.0e-9
_DESIGN_INDEPENDENT_QOI_BOUND = 1.0e-8

# Every derivative check and its preregistered bound; completion recomputes
# each observation against these constants rather than any recorded verdict.
_DESIGN_SOURCE_DERIVATIVE_BOUNDS: dict[str, float] = {
    "finite_difference_step_event_fraction": _DERIVATIVE_STEP_EVENT_FRACTION_BOUND,
    "geometry_finite_difference": _DERIVATIVE_FD_BOUND,
    "geometry_JVP_VJP_duality": _DERIVATIVE_DUALITY_BOUND,
    "pde_finite_difference": _DERIVATIVE_FD_BOUND,
    "pde_JVP_VJP_duality": _DERIVATIVE_DUALITY_BOUND,
    "qoi_solve_free_moment_identity": _DESIGN_INDEPENDENT_QOI_BOUND,
    "qoi_adjoint_tangent_duality": _DERIVATIVE_DUALITY_BOUND,
    "qoi_adjoint_finite_difference": _DERIVATIVE_FD_BOUND,
    "qoi_adjoint_analytic_reference": _DERIVATIVE_FD_BOUND,
    "qoi_taylor_roundoff_ratio": _TAYLOR_ROUNDOFF_RATIO_BOUND,
    "qoi_taylor_slope_deviation": _TAYLOR_SLOPE_DEVIATION_BOUND,
    "qoi_taylor_pairwise_slope_deviation": _TAYLOR_SLOPE_DEVIATION_BOUND,
}
_DESIGN_DERIVATIVE_BOUNDS: dict[str, float] = {
    **_DESIGN_SOURCE_DERIVATIVE_BOUNDS,
    "transfer_finite_difference": _DERIVATIVE_FD_BOUND,
    "transfer_JVP_VJP_duality": _DERIVATIVE_DUALITY_BOUND,
}


def _design_qualification_check(observed: float, bound: float) -> dict[str, object]:
    return {
        "observed": observed,
        "bound": bound,
        "passed": isfinite(observed) and observed <= bound,
    }


def _design_qualification_taylor_slopes(
    fractions: tuple[float, ...], remainders: list[float]
) -> tuple[float, list[float]]:
    """Return the least-squares log-log Taylor order and its pairwise slopes."""
    if len(fractions) != len(remainders) or len(fractions) < 2:
        raise ValueError(
            f"_design_qualification_taylor_slopes.samples must pair at least two step fractions with remainders; got {(fractions, remainders)!r}."
        )
    steps = np.log(np.asarray(fractions, dtype=np.float64))
    values = np.asarray(remainders, dtype=np.float64)
    if not np.all(np.isfinite(steps)) or not np.all(np.isfinite(values)) or np.any(
        values <= 0.0
    ):
        raise ValueError(
            f"_design_qualification_taylor_slopes.remainders must be finite positive observations; got {remainders!r}."
        )
    logs = np.log(values)
    centered = steps - np.mean(steps)
    fitted = float(np.sum(centered * (logs - np.mean(logs))) / np.sum(centered**2))
    pairwise = [
        float((logs[index] - logs[index + 1]) / (steps[index] - steps[index + 1]))
        for index in range(len(fractions) - 1)
    ]
    return fitted, pairwise


def _design_qualification_represented_qoi(
    cells: tuple[jax.Array, ...],
    dofs: tuple[jax.Array, ...],
    points: jax.Array,
    values: jax.Array,
) -> jax.Array:
    """Integrate the P1 field exactly over the represented flat triangles."""
    total = jnp.zeros((), dtype=points.dtype)
    for triangles, cell_dofs in zip(cells, dofs, strict=True):
        corners = points[triangles]
        normal = jnp.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        area = 0.5 * jnp.sqrt(jnp.sum(normal * normal, axis=-1))
        total = total + jnp.sum(area * jnp.mean(values[cell_dofs], axis=1))
    return total


def _design_qualification_qoi(
    solve_coordinates: Callable[
        [jax.Array], tuple[jax.Array, phx.linalg.LinearSolveResult]
    ],
    coordinates: Callable[[jax.Array], jax.Array],
    cells: tuple[jax.Array, ...],
    dofs: tuple[jax.Array, ...],
    radius: jax.Array,
) -> jax.Array:
    points = coordinates(radius)
    return _design_qualification_represented_qoi(
        cells, dofs, points, solve_coordinates(points)[0]
    )


@final
class _DesignFixedRoute(phx.StrictModule):
    """Fixed-topology realization, PDE and represented-QoI routes of one epoch."""

    surface_plan: phx.geometry.ImplicitSurfacePlan
    state: DesignState
    parameter: jax.Array
    coordinates: Callable[[jax.Array], jax.Array]
    solve_coordinates: Callable[
        [jax.Array], tuple[jax.Array, phx.linalg.LinearSolveResult]
    ]
    pde: Callable[[jax.Array], jax.Array]
    qoi: Callable[[jax.Array], jax.Array]
    cells: tuple[jax.Array, ...]
    dofs: tuple[jax.Array, ...]
    radius_index: int = eqx.field(static=True)
    dof_count: int = eqx.field(static=True)


def _design_qualification_fixed_route(
    geometry: phx.geometry.CompiledGeometry,
    plan: phx.meshing.NativeMeshingPlan,
    source: phx.meshing.CellMeshingResult,
    scalar: phx.discretization.FiniteElementDiscretization,
    problem: phx.equations.CompiledFiniteElementProblem,
) -> _DesignFixedRoute:
    """Bind the epoch's fixed realization, PDE solve and QoI to one parameter."""
    surface_plan = plan.prepared
    if not isinstance(surface_plan, phx.geometry.ImplicitSurfacePlan):
        raise TypeError(
            f"_design_qualification_derivatives.surface_plan must be an ImplicitSurfacePlan; got {type(surface_plan).__name__!r}."
        )
    radius_index = geometry.schema.index(
        phx.geometry.ParameterId("design-sphere", "radius")
    )
    solve_policy = phx.linalg.LinearSolvePolicy(
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(
            relative_tolerance=1.0e-12, absolute_tolerance=1.0e-14
        )
    )
    coordinates = eqx.Partial(
        _design_qualification_coordinates,
        surface_plan,
        geometry.state,
        radius_index,
    )
    solve_coordinates = eqx.Partial(
        _design_qualification_solve_coordinates,
        problem,
        source.mesh,
        source.geometry.geometry_layout_id,
        solve_policy,
    )
    cells = tuple(
        jnp.asarray(np.asarray(block.vertices), dtype=jnp.int32)
        for block in source.mesh.blocks
    )
    dofs = tuple(
        jnp.asarray(np.asarray(cell_dofs), dtype=jnp.int32)
        for cell_dofs in scalar.dof_maps[0].cell_dofs
    )
    return _DesignFixedRoute(
        surface_plan=surface_plan,
        state=geometry.state,
        parameter=geometry.state.values[radius_index],
        coordinates=coordinates,
        solve_coordinates=solve_coordinates,
        pde=eqx.Partial(_design_qualification_pde, solve_coordinates, coordinates),
        qoi=eqx.Partial(
            _design_qualification_qoi, solve_coordinates, coordinates, cells, dofs
        ),
        cells=cells,
        dofs=dofs,
        radius_index=radius_index,
        dof_count=scalar.field_spaces[0].layout.size,
    )


@eqx.filter_jit
def _design_qualification_realization_margins(
    surface_plan: phx.geometry.ImplicitSurfacePlan,
    state: DesignState,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Realize one state with its signed fixed-topology event margins."""
    realization = surface_plan.realize(state)
    field = surface_plan.kernel.boundary_field(state, surface_plan.grid_points)
    # Positive exactly while every lattice sign agrees with the prepared topology.
    signed = jnp.where(surface_plan.inside_pattern, -field, field)
    return (
        realization.vertices,
        realization.evidence.status,
        jnp.min(signed),
        realization.evidence.minimum_face_area,
        realization.evidence.minimum_orientation_margin,
    )


def _design_qualification_event_margins(
    surface_plan: phx.geometry.ImplicitSurfacePlan,
    state: DesignState,
) -> tuple[tuple[tuple[str, float], ...], np.ndarray, int]:
    """Return the owner's event margins, safe vertices and realization status."""
    vertices, status, lattice, area, orientation = jax.device_get(
        _design_qualification_realization_margins(surface_plan, state)
    )
    margins = (
        ("lattice-sign", float(lattice) - surface_plan.policy.lattice_zero_tolerance),
        ("positive-face-area", float(area) - surface_plan.policy.minimum_face_area),
        ("positive-orientation", float(orientation)),
    )
    return margins, np.asarray(vertices, dtype=np.float64), int(status)


@eqx.filter_jit
def _design_qualification_lattice_rates(
    surface_plan: phx.geometry.ImplicitSurfacePlan,
    state: DesignState,
    radius_index: int,
    radius: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    def field(value: jax.Array) -> jax.Array:
        return surface_plan.kernel.boundary_field(
            state.replace_at(radius_index, value), surface_plan.grid_points
        )

    return jax.jvp(field, (radius,), (jnp.ones_like(radius),))


def _design_qualification_parameter_events(
    route: _DesignFixedRoute,
) -> dict[str, Any]:
    """Predict the nearest lattice sign events on each side of the parameter."""
    values, rates = jax.device_get(
        _design_qualification_lattice_rates(
            route.surface_plan, route.state, route.radius_index, route.parameter
        )
    )
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    rates = np.asarray(rates, dtype=np.float64).reshape(-1)
    moving = rates != 0.0
    offsets = np.full(values.shape, np.inf)
    offsets[moving] = -values[moving] / rates[moving]
    upper = float(np.min(offsets[offsets > 0.0], initial=np.inf))
    lower = float(np.max(offsets[offsets < 0.0], initial=-np.inf))
    if not (isfinite(upper) or isfinite(lower)):
        raise qualification_failure(
            "_design_qualification_parameter_events",
            ("The design parameter moves no lattice sign; no fixed-epoch event exists."),
            {"moving": int(np.count_nonzero(moving))},
        )
    return {
        "lower": lower if isfinite(lower) else None,
        "upper": upper if isfinite(upper) else None,
        "moving_lattice_nodes": int(np.count_nonzero(moving)),
        "prediction": "first-order zero crossing of each lattice boundary-field value",
    }


def _design_qualification_solved_realization(
    route: _DesignFixedRoute, radius: jax.Array | float
) -> tuple[np.ndarray, np.ndarray]:
    """Realize and solve one admitted state of the fixed epoch."""
    state = route.state.replace_at(
        route.radius_index, jnp.asarray(radius, dtype=jnp.float64)
    )
    _, vertices, status = _design_qualification_event_margins(route.surface_plan, state)
    if status != int(ImplicitSurfaceStatus.SUCCESS):
        raise qualification_failure(
            "_design_qualification_solved_realization",
            (
                "A fixed-epoch derivative probe crosses the implicit realization boundary."
            ),
            {"radius": float(radius), "status": status},
        )
    values, solved = route.solve_coordinates(jnp.asarray(vertices))
    if not bool(jnp.all(solved.successful)):
        raise qualification_failure(
            "_design_qualification_solved_realization",
            ("A fixed-epoch derivative probe failed the scalar PDE solve."),
            {"solved": solved},
        )
    return vertices, np.asarray(values, dtype=np.float64)


def _design_qualification_perturbations(
    route: _DesignFixedRoute,
    direction: jax.Array,
    step: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Retain both admitted geometry/PDE states for the common FD oracle."""
    return (
        _design_qualification_solved_realization(
            route, route.parameter - step * direction
        ),
        _design_qualification_solved_realization(
            route, route.parameter + step * direction
        ),
    )


def _design_qualification_qoi_fields(
    cells: tuple[jax.Array, ...],
    dofs: tuple[jax.Array, ...],
    fields: tuple[jax.Array, jax.Array],
) -> jax.Array:
    return _design_qualification_represented_qoi(cells, dofs, *fields)


@eqx.filter_jit
def _design_qualification_qoi_derivative_values(
    cells: tuple[jax.Array, ...],
    dofs: tuple[jax.Array, ...],
    points: jax.Array,
    values: jax.Array,
    geometry_tangent: jax.Array,
    pde_tangent: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    linearization = phx.linalg.prepare_linearization(
        eqx.Partial(_design_qualification_qoi_fields, cells, dofs), (points, values)
    )
    value = _design_qualification_array(linearization.primal)
    geometry_cotangent, pde_cotangent = linearization.vjp(jnp.ones_like(value))
    tangent = _design_qualification_array(
        linearization.jvp((geometry_tangent, pde_tangent))
    )
    return value, geometry_cotangent, pde_cotangent, tangent


def _design_qualification_adjoint_values(
    geometry: phx.linalg.PreparedLinearization,
    pde: phx.linalg.PreparedLinearization,
    cells: tuple[jax.Array, ...],
    dofs: tuple[jax.Array, ...],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    direction = jnp.ones_like(_design_qualification_array(pde.point))
    value, geometry_cotangent, pde_cotangent, tangent = (
        _design_qualification_qoi_derivative_values(
            cells,
            dofs,
            _design_qualification_array(geometry.primal),
            _design_qualification_array(pde.primal),
            _design_qualification_tape_jvp(geometry, direction),
            _design_qualification_tape_jvp(pde, direction),
        )
    )
    gradient = _design_qualification_tape_vjp(geometry, geometry_cotangent) + (
        _design_qualification_tape_vjp(pde, pde_cotangent)
    )
    return value, gradient, tangent


def _design_qualification_adjoint(
    route: _DesignFixedRoute,
    perturbations: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    direction: jax.Array,
    step: float,
    events: Mapping[str, Any],
    linearizations: tuple[phx.linalg.PreparedLinearization, phx.linalg.PreparedLinearization],
) -> dict[str, Any]:
    """Check the reverse-mode QoI adjoint against independent references.

    References are the forward tangent, central differences of independently
    realized and solved states, the solve-free closed-form moment gradient on
    independently realized vertex tangents, and Taylor remainder convergence.
    """
    value, gradient, tangent = jax.device_get(
        _design_qualification_adjoint_values(*linearizations, route.cells, route.dofs)
    )
    adjoint, forward = float(gradient), float(tangent)
    cells = tuple(np.asarray(item) for item in route.cells)
    dofs = tuple(np.asarray(item) for item in route.dofs)
    (lower_points, lower_values), (upper_points, upper_values) = perturbations
    width = 2.0 * step * float(direction)
    finite_difference = (
        _design_qualification_host_qoi(upper_points, upper_values, cells, dofs)
        - _design_qualification_host_qoi(lower_points, lower_values, cells, dofs)
    ) / width
    points, values = _design_qualification_solved_realization(route, route.parameter)
    moment, moment_gradient = _design_qualification_moment_gradient(points, cells)
    analytic = float(np.sum(moment_gradient * ((upper_points - lower_points) / width)))
    base = _design_qualification_host_qoi(points, values, cells, dofs)
    upper, lower = events["upper"], events["lower"]
    sign, distance = (
        (1.0, float(upper))
        if upper is not None and (lower is None or upper >= -lower)
        else (-1.0, -float(lower))
    )
    remainders = []
    for fraction in _TAYLOR_FRACTIONS:
        offset = sign * distance * fraction
        probe_points, probe_values = _design_qualification_solved_realization(
            route, route.parameter + offset
        )
        remainders.append(
            abs(
                _design_qualification_host_qoi(probe_points, probe_values, cells, dofs)
                - base
                - offset * adjoint
            )
        )
    if any(not isfinite(remainder) or remainder <= 0.0 for remainder in remainders):
        raise qualification_failure(
            "_design_qualification_adjoint",
            ("Taylor remainders must be finite and resolved above zero."),
            {"remainders": remainders},
        )
    fitted, slopes = _design_qualification_taylor_slopes(_TAYLOR_FRACTIONS, remainders)
    cell_count = sum(int(triangles.shape[0]) for triangles in cells)
    roundoff = float(np.finfo(np.float64).eps) * cell_count * abs(base)
    checks = {
        # The solve-free reference is the gradient of ∫x² over the represented
        # triangles; it references the adjoint only while 1ᵀ(Mu - b) vanishes.
        "qoi_solve_free_moment_identity": _design_qualification_check(
            abs(base - moment), _DESIGN_INDEPENDENT_QOI_BOUND
        ),
        "qoi_adjoint_tangent_duality": _design_qualification_check(
            abs(adjoint - forward), _DERIVATIVE_DUALITY_BOUND
        ),
        "qoi_adjoint_finite_difference": _design_qualification_check(
            abs(adjoint - finite_difference), _DERIVATIVE_FD_BOUND
        ),
        "qoi_adjoint_analytic_reference": _design_qualification_check(
            abs(adjoint - analytic), _DERIVATIVE_FD_BOUND
        ),
        "qoi_taylor_roundoff_ratio": _design_qualification_check(
            roundoff / min(remainders), _TAYLOR_ROUNDOFF_RATIO_BOUND
        ),
        "qoi_taylor_slope_deviation": _design_qualification_check(
            abs(fitted - 2.0), _TAYLOR_SLOPE_DEVIATION_BOUND
        ),
        "qoi_taylor_pairwise_slope_deviation": _design_qualification_check(
            max(abs(slope - 2.0) for slope in slopes), _TAYLOR_SLOPE_DEVIATION_BOUND
        ),
    }
    failed = [name for name, check in checks.items() if not check["passed"]]
    if failed:
        raise qualification_failure(
            "_design_qualification_adjoint",
            (
                "Native fixed-epoch QoI adjoint failed its independent references: "
                + ", ".join(failed)
            ),
            {"checks": checks},
        )
    return {
        "objective": "represented-surface integral of the solved P1 field",
        "value": float(value),
        "host_value": base,
        "solve_free_moment": moment,
        "moment_identity_error": abs(base - moment),
        "adjoint_gradient": adjoint,
        "tangent_gradient": forward,
        "finite_difference_gradient": finite_difference,
        "analytic_reference_gradient": analytic,
        "analytic_reference": "closed-form affine-triangle x^2 moment gradient on independently realized FD vertex tangents; no PDE solve",
        "taylor": {
            "direction": sign,
            "event_distance": distance,
            "fractions": list(_TAYLOR_FRACTIONS),
            "offsets": [sign * distance * fraction for fraction in _TAYLOR_FRACTIONS],
            "remainders": remainders,
            "least_squares_log_log_slope": fitted,
            "pairwise_slopes": slopes,
            "roundoff_bound": roundoff,
        },
        "taylor_slope": fitted,
        "checks": checks,
    }


def _design_qualification_lattice_crossing(
    route: _DesignFixedRoute, events: Mapping[str, Any]
) -> dict[str, Any]:
    """Observe the realization owner's refusal across the nearest lattice event."""
    upper, lower = events["upper"], events["lower"]
    offset = (
        float(upper)
        if upper is not None and (lower is None or upper <= -lower)
        else float(lower)
    )
    observed: dict[
        str, tuple[float, tuple[tuple[str, float], ...], np.ndarray, int]
    ] = {}
    for name, factor in (
        ("before", 1.0 - _EVENT_PROBE_FRACTION),
        ("after", 1.0 + _EVENT_PROBE_FRACTION),
    ):
        radius = float(route.parameter) + factor * offset
        margins, vertices, status = _design_qualification_event_margins(
            route.surface_plan,
            route.state.replace_at(
                route.radius_index, jnp.asarray(radius, dtype=jnp.float64)
            ),
        )
        observed[name] = (radius, margins, vertices, status)
    sign_bit = int(ImplicitSurfaceStatus.SIGN_PATTERN_CHANGED)
    before, after = observed["before"], observed["after"]
    crossing = {
        "predicted_event_offset": offset,
        "probe_fraction": _EVENT_PROBE_FRACTION,
        "before": {"radius": before[0], "status": before[3], "margins": before[1]},
        "after": {"radius": after[0], "status": after[3], "margins": after[1]},
        "event_free_before": not before[3] & sign_bit,
        "owner_refused_crossing": bool(after[3] & sign_bit)
        and after[3] != int(ImplicitSurfaceStatus.SUCCESS),
        "realized_vertex_jump": float(np.max(np.abs(after[2] - before[2]))),
        "event_id": canonical_fingerprint(
            {
                "kind": "design-lattice-event-crossing",
                "plan": route.surface_plan.plan_id,
                "radius": after[0],
                "status": after[3],
            }
        ),
    }
    if not crossing["event_free_before"] or not crossing["owner_refused_crossing"]:
        raise qualification_failure(
            "_design_qualification_lattice_crossing",
            (
                "The predicted lattice event was not refused by the fixed-topology realization owner."
            ),
            {"crossing": crossing},
        )
    return crossing


def _design_qualification_operator_condition(
    problem: phx.equations.CompiledFiniteElementProblem, size: int
) -> float:
    """Measure the 2-norm condition of the compiled scalar operator."""
    matrix = np.asarray(
        _design_qualification_condition_matrix(
            problem, jax.device_put(np.eye(size, dtype=np.float64))
        )
    )
    return float(np.linalg.cond(matrix))


def _design_qualification_source_derivatives(
    route: _DesignFixedRoute,
    source: phx.meshing.CellMeshingResult,
    problem: phx.equations.CompiledFiniteElementProblem,
) -> dict[str, Any]:
    """Qualify geometry, PDE and adjoint QoI routes before any native proposal."""
    margins, _, status = _design_qualification_event_margins(
        route.surface_plan, route.state
    )
    if status != int(ImplicitSurfaceStatus.SUCCESS) or any(
        value <= 0.0 for _, value in margins
    ):
        raise qualification_failure(
            "_design_qualification_derivatives",
            ("Fixed-lattice realization has no positive derivative event margins."),
            {"margins": margins, "status": status},
        )
    events = _design_qualification_parameter_events(route)
    direction = jnp.asarray(_DERIVATIVE_DIRECTION, dtype=jnp.float64)
    nearest = min(
        abs(float(offset))
        for offset in (events["lower"], events["upper"])
        if offset is not None
    )
    step_check = _design_qualification_check(
        _DERIVATIVE_STEP * abs(_DERIVATIVE_DIRECTION) / nearest,
        _DERIVATIVE_STEP_EVENT_FRACTION_BOUND,
    )
    if not step_check["passed"]:
        raise qualification_failure(
            "_design_qualification_derivatives",
            ("Central-difference probes would leave the event-free parameter interval."),
            {"events": events, "step_check": step_check},
        )
    perturbations = _design_qualification_perturbations(
        route, direction, _DERIVATIVE_STEP
    )
    (lower_points, lower_values), (upper_points, upper_values) = perturbations
    geometry_linearization, pde_linearization = _design_qualification_linearizations(
        route.coordinates, route.pde, route.parameter
    )
    diagnostics = {
        "geometry": _design_qualification_derivative_probe(
            (geometry_linearization,),
            direction,
            lower_points,
            upper_points,
            _DERIVATIVE_STEP,
        ),
        "pde": _design_qualification_derivative_probe(
            (pde_linearization,),
            direction,
            lower_values,
            upper_values,
            _DERIVATIVE_STEP,
        ),
    }
    adjoint = _design_qualification_adjoint(
        route, perturbations, direction, _DERIVATIVE_STEP, events,
        (geometry_linearization, pde_linearization),
    )
    _, solved = route.solve_coordinates(route.coordinates(route.parameter))
    checks = {
        "finite_difference_step_event_fraction": step_check,
        "geometry_finite_difference": _design_qualification_check(
            diagnostics["geometry"]["finite_difference_error"], _DERIVATIVE_FD_BOUND
        ),
        "geometry_JVP_VJP_duality": _design_qualification_check(
            diagnostics["geometry"]["JVP_VJP_duality_error"], _DERIVATIVE_DUALITY_BOUND
        ),
        "pde_finite_difference": _design_qualification_check(
            diagnostics["pde"]["finite_difference_error"], _DERIVATIVE_FD_BOUND
        ),
        "pde_JVP_VJP_duality": _design_qualification_check(
            diagnostics["pde"]["JVP_VJP_duality_error"], _DERIVATIVE_DUALITY_BOUND
        ),
        **adjoint["checks"],
    }
    return {
        "source_result_id": source.result_id,
        "execution_ids": (route.surface_plan.plan_id, problem.compilation_id),
        "event_margins": margins,
        "parameter_events": events,
        "diagnostics": diagnostics,
        "adjoint": adjoint,
        "primal_successful": bool(np.all(np.asarray(solved.successful))),
        "primal_residual": float(np.max(np.asarray(solved.diagnostics.residual_norm))),
        "operator_condition": _design_qualification_operator_condition(
            problem, route.dof_count
        ),
        "lattice_crossing": _design_qualification_lattice_crossing(route, events),
        "checks": checks,
        "scope": "fixed source geometry/PDE/adjoint QoI before any native proposal",
    }


def _design_qualification_derivative_evidence(
    source: phx.meshing.CellMeshingResult,
    scalar: phx.discretization.FiniteElementDiscretization,
    trial: _DesignQualificationTrial,
    ids: tuple[str, str, str],
    margins: tuple[tuple[str, float], ...],
    baseline: Mapping[str, Any],
    transfer_diagnostic: dict[str, float],
    retained_evidence: list[phx.meshing.FixedEpochDerivativeEvidence] | None,
) -> dict[str, object]:
    """Qualify measured fixed routes and prove invalidation at declared events."""
    diagnostics = {**baseline["diagnostics"], "transfer": transfer_diagnostic}
    checks = {
        **baseline["checks"],
        "transfer_finite_difference": _design_qualification_check(
            transfer_diagnostic["finite_difference_error"], _DERIVATIVE_FD_BOUND
        ),
        "transfer_JVP_VJP_duality": _design_qualification_check(
            transfer_diagnostic["JVP_VJP_duality_error"], _DERIVATIVE_DUALITY_BOUND
        ),
    }

    def passed(*names: str) -> bool:
        return all(checks[name]["passed"] for name in names)

    adjoint = baseline["adjoint"]
    event_ids = tuple(name for name, _ in margins)
    event_values = tuple(value for _, value in margins)
    geometry = DesignQualificationEvidence(
        DerivativeTier.Q1,
        scalar.prepared_id,
        source.result_id,
        ids[0],
        event_ids=event_ids,
        event_margins=event_values,
        gradient_error=diagnostics["geometry"]["finite_difference_error"],
        valid=passed(
            "finite_difference_step_event_fraction",
            "geometry_finite_difference",
            "geometry_JVP_VJP_duality",
        ),
    )
    pde = DesignQualificationEvidence(
        DerivativeTier.Q2,
        scalar.prepared_id,
        source.result_id,
        ids[1],
        primal_residual=baseline["primal_residual"],
        condition_estimate=baseline["operator_condition"],
        event_ids=event_ids,
        event_margins=event_values,
        gradient_error=max(
            diagnostics["pde"]["finite_difference_error"],
            adjoint["checks"]["qoi_adjoint_finite_difference"]["observed"],
            adjoint["checks"]["qoi_adjoint_analytic_reference"]["observed"],
        ),
        taylor_slope=adjoint["taylor_slope"],
        valid=baseline["primal_successful"]
        and passed(
            "finite_difference_step_event_fraction",
            "pde_finite_difference",
            "pde_JVP_VJP_duality",
            "qoi_solve_free_moment_identity",
            "qoi_adjoint_tangent_duality",
            "qoi_adjoint_finite_difference",
            "qoi_adjoint_analytic_reference",
            "qoi_taylor_roundoff_ratio",
            "qoi_taylor_slope_deviation",
            "qoi_taylor_pairwise_slope_deviation",
        ),
    )
    transfer = DesignQualificationEvidence(
        DerivativeTier.Q1,
        scalar.prepared_id,
        source.result_id,
        ids[2],
        event_ids=event_ids,
        event_margins=event_values,
        gradient_error=transfer_diagnostic["finite_difference_error"],
        valid=passed(
            "finite_difference_step_event_fraction",
            "transfer_finite_difference",
            "transfer_JVP_VJP_duality",
        ),
    )
    routes = tuple(zip(("geometry", "pde", "transfer"), ids, strict=True))
    evidence = FixedEpochDerivativeEvidence(
        source.result_id,
        "design-sphere.radius",
        geometry=geometry,
        pde=pde,
        transfer=transfer,
        routes=routes,
    )
    binding = evidence.invalidation_issues(
        source.result_id, "design-sphere.radius", routes=routes, event_margins=margins
    )
    adaptation = trial["adaptation"]
    topology = evidence.invalidation_issues(
        adaptation.target.result_id,
        "design-sphere.radius",
        routes=routes,
        event_margins=margins,
        topology_event_id=adaptation.result_id,
    )
    crossing = baseline["lattice_crossing"]
    crossed = evidence.invalidation_issues(
        source.result_id,
        "design-sphere.radius",
        routes=routes,
        event_margins=crossing["after"]["margins"],
        geometry_realization_id=crossing["event_id"],
    )
    transition = DesignQualificationEvidence(
        DerivativeTier.Q6,
        scalar.prepared_id,
        source.result_id,
        adaptation.result_id,
        event_ids=event_ids,
        event_margins=event_values,
        transfer_plan_id=ids[2],
        transition_event_id=adaptation.result_id,
        valid=bool(adaptation.status.converged) and adaptation.transition is not None,
    )
    if (
        binding
        or "stopped-topology-event" not in topology
        or "event-boundary" not in crossed
        or "stopped-geometry-realization-event" not in crossed
        or not crossing["owner_refused_crossing"]
        or transition.tier.is_derivative
        or not transition.valid
    ):
        raise qualification_failure(
            "_design_qualification_derivatives",
            (
                "Fixed-epoch derivatives must be valid on their epoch and stopped at every declared event."
            ),
            {"binding": binding, "topology": topology, "crossing": crossed},
        )
    if retained_evidence is not None:
        retained_evidence.append(evidence)
    return {
        "evidence_id": evidence.evidence_id,
        "epoch_id": evidence.epoch_id,
        "event_margins": margins,
        "diagnostics": diagnostics,
        "adjoint": adjoint,
        "parameter_events": baseline["parameter_events"],
        "checks": checks,
        "qualifications": {
            stage: {
                "tier": qualification.tier.value,
                "evidence_id": qualification.evidence_id,
                "valid": qualification.valid,
            }
            for stage, qualification in zip(
                ("geometry", "pde", "transfer"),
                (geometry, pde, transfer),
                strict=True,
            )
        },
        "binding_issues": list(binding),
        "event_nondifferentiability": {
            "topology_event": {
                "event_id": adaptation.result_id,
                "invalidation": list(topology),
                "transition_evidence_id": transition.evidence_id,
                "transition_tier": transition.tier.value,
                "transition_is_derivative": transition.tier.is_derivative,
                "transition_valid": transition.valid,
            },
            "lattice_crossing": {**crossing, "invalidation": list(crossed)},
        },
        "event_invalidation": list(topology),
        "scope": "fixed discovery, PDE and donor routes only; events stop derivatives",
    }


def _design_qualification_derivatives(
    geometry: phx.geometry.CompiledGeometry,
    plan: phx.meshing.NativeMeshingPlan,
    source: phx.meshing.CellMeshingResult,
    scalar: phx.discretization.FiniteElementDiscretization,
    problem: phx.equations.CompiledFiniteElementProblem,
    trial: _DesignQualificationTrial | None,
    baseline: Mapping[str, Any] | None = None,
    *,
    retained_evidence: list[phx.meshing.FixedEpochDerivativeEvidence] | None = None,
) -> dict[str, Any]:
    """Measure fixed-route geometry, PDE, adjoint QoI and transfer contracts."""

    route = _design_qualification_fixed_route(geometry, plan, source, scalar, problem)
    if trial is None:
        return _design_qualification_source_derivatives(route, source, problem)
    margins, _, status = _design_qualification_event_margins(
        route.surface_plan, route.state
    )
    if status != int(ImplicitSurfaceStatus.SUCCESS) or any(
        value <= 0.0 for _, value in margins
    ):
        raise qualification_failure(
            "_design_qualification_derivatives",
            ("Fixed-lattice realization has no positive derivative event margins."),
            {"margins": margins, "status": status},
        )
    if (
        baseline is None
        or baseline["source_result_id"] != source.result_id
        or baseline["execution_ids"] != (route.surface_plan.plan_id, problem.compilation_id)
    ):
        raise ValueError(
            f"_design_qualification_derivatives.baseline must contain measured geometry and PDE evidence for this source and execution epoch; got {baseline!r}."
        )
    transfers = trial["staged"].transfers
    if not transfers:
        raise qualification_failure(
            "_design_qualification_derivatives",
            ("The transfer derivative requires a prepared native field transfer."),
            {"transfers": transfers},
        )
    transfer = transfers[0]
    direction = jnp.asarray(_DERIVATIVE_DIRECTION, dtype=jnp.float64)
    (_, lower_values), (_, upper_values) = _design_qualification_perturbations(
        route, direction, _DERIVATIVE_STEP
    )
    _, pde_linearization = _design_qualification_linearizations(
        route.coordinates, route.pde, route.parameter
    )
    transferred_linearization = _design_qualification_transfer_linearization(
        transfer.transfer, _design_qualification_array(pde_linearization.primal)
    )
    transfer_diagnostic = _design_qualification_derivative_probe(
        (pde_linearization, transferred_linearization),
        direction,
        _design_qualification_array(transfer.transfer.apply(jnp.asarray(lower_values))),
        _design_qualification_array(transfer.transfer.apply(jnp.asarray(upper_values))),
        _DERIVATIVE_STEP,
    )
    return _design_qualification_derivative_evidence(
        source,
        scalar,
        trial,
        (route.surface_plan.plan_id, problem.compilation_id, transfer.transfer_id),
        margins,
        baseline,
        transfer_diagnostic,
        retained_evidence,
    )


def _design_qualification_root_residual(
    qoi: Callable[[jax.Array], jax.Array],
    target: jax.Array,
    radius: jax.Array,
    args: object,
) -> jax.Array:
    del args
    return qoi(radius) - target


def _design_qualification_root_derivative(
    qoi: Callable[[jax.Array], jax.Array],
    radius: jax.Array,
    args: object,
) -> jax.Array:
    # The design step consumes the qualified reverse-mode adjoint of the route.
    del args
    linearization = phx.linalg.prepare_linearization(qoi, radius)
    return linearization.vjp(jnp.ones_like(linearization.primal))


def _design_qualification_root_validity(
    surface_plan: phx.geometry.ImplicitSurfacePlan,
    state: DesignState,
    radius_index: int,
    radius: jax.Array,
    value: jax.Array,
    args: object,
) -> jax.Array:
    del value, args
    return surface_plan.realize(state.replace_at(radius_index, radius)).evidence.accepted


@eqx.filter_jit
def _design_qualification_root(
    problem: ScalarRootProblem,
) -> phx.nonlinear.ScalarRootResult:
    return scalar_root(
        problem,
        method=SafeguardedNewton(),
        termination=NonlinearTermination(
            absolute_residual=_DESIGN_QOI_RESIDUAL,
            relative_residual=0.0,
            maximum_steps=32,
        ),
    )


def _design_qualification_candidates(
    current: phx.meshing.CellMeshingResult,
    accepted: phx.solver.FiniteElementAcceptedState,
    scalar: phx.discretization.FiniteElementDiscretization,
    request: phx.meshing.SurfaceMeshingSpec,
    baseline: phx.meshing.PhysicalErrorEvidence,
    budget: phx.meshing.DecisionBudget,
    args: QualificationControls,
    epoch_record: dict[str, Any],
    record: dict[str, Any],
    started: float,
) -> tuple[
    dict[str, phx.meshing.SolverAwareCandidate], dict[str, _DesignQualificationTrial]
]:
    """Retain every observed trial and choose the cheaper proof for coincident native routes.

    The learned marker is supervised by the completed uniform comparator's
    observed solved field; without it the learned trial fails.
    """
    alternatives: dict[str, phx.meshing.SolverAwareCandidate] = {}
    trials: dict[str, _DesignQualificationTrial] = {}
    observations: _DesignLearnedObservations | None = None
    for mode in ("uniform", "analytic", "learned"):
        trial_record: dict[str, Any] = {
            "proposal": mode,
            "status": "failed",
            "active_phase": "proposal",
        }
        epoch_record["trials"].append(trial_record)
        attempt_started = perf_counter()
        try:
            trial = _design_qualification_trial(
                current,
                accepted,
                mode,
                args,
                request.limits,
                trial_record,
                baseline,
                budget,
                started + args.timeout,
                observations if mode == "learned" else None,
            )
            if mode == "uniform":
                observations = _DesignLearnedObservations(
                    training=trial["adaptation"].target,
                    training_space=trial["scalar"],
                    training_values=trial["values"],
                    source_space=scalar,
                    source_values=accepted.fields[0],
                    training_proposal=mode,
                )
            candidate, orchestration = _design_qualification_observed_cost(
                trial["candidate"], perf_counter() - attempt_started
            )
            trial["candidate"] = candidate
            trial_record.update(
                {
                    "measured_cost_seconds": candidate.cost.total_seconds,
                    "peak_process_resident_bytes": candidate.cost.peak_memory_bytes,
                    "cost_id": candidate.cost.cost_id,
                }
            )
            trial_record["phase_seconds"]["caller_orchestration"] = orchestration
            previous = alternatives.get(candidate.candidate_id)
            if previous is not None:
                trial_record["coincident_realized_route"] = candidate.candidate_id
            if (
                previous is None
                or candidate.cost.total_seconds < previous.cost.total_seconds
            ):
                alternatives[candidate.candidate_id] = candidate
                trials[candidate.candidate_id] = trial
        except (
            phx.meshing.MeshingFailure,
            RuntimeError,
            ValueError,
            TypeError,
        ) as failure:
            refusal = {
                **FailureRecord.from_exception(
                    failure, trial_record["active_phase"]
                ).to_record(),
                "status": "failed",
                "safe_refusal_is_positive": False,
            }
            trial_record.update(refusal)
            record["failed_positive_attempts"].append(
                {"epoch": epoch_record["epoch"], "proposal": mode, **refusal}
            )
        finally:
            trial_record["observed_attempt_seconds"] = perf_counter() - attempt_started
            del trial_record["active_phase"]
    return alternatives, trials


@final
class _DesignInitialEpoch(phx.StrictModule):
    geometry: phx.geometry.CompiledGeometry
    plan: phx.meshing.NativeMeshingPlan
    request: phx.meshing.SurfaceMeshingSpec
    result: phx.meshing.CellMeshingResult
    accepted: phx.solver.FiniteElementAcceptedState
    fields: phx.discretization.FiniteElementDiscretization


@final
class _DesignEpochSelection(phx.StrictModule):
    epoch_record: dict[str, Any]
    trials: dict[str, _DesignQualificationTrial]
    decision: phx.meshing.SolverAwareDecision
    candidate: phx.meshing.SolverAwareCandidate
    selected: _DesignQualificationTrial


@final
class _DesignContinuedEpoch(phx.StrictModule):
    accepted: phx.solver.FiniteElementAcceptedState
    current: phx.meshing.CellMeshingResult
    scalar: phx.discretization.FiniteElementDiscretization
    physical: dict[str, float]
    values: jax.Array


@final
class _DesignRetainedSource(phx.StrictModule):
    """Archivable design source plus its uninterrupted in-process continued solve.

    ``uninterrupted_continuation`` is the warm continued solve from
    ``state.accepted_state.fields[0]``; a cold consumer continuing the restored
    accepted state under the same solver policy must reproduce it.
    """

    state: NativeDesignSourceState
    uninterrupted_continuation: jax.Array


def _design_qualification_select_epoch(
    current: phx.meshing.CellMeshingResult,
    accepted: phx.solver.FiniteElementAcceptedState,
    scalar: phx.discretization.FiniteElementDiscretization,
    request: phx.meshing.SurfaceMeshingSpec,
    budget: phx.meshing.DecisionBudget,
    args: QualificationControls,
    epoch: int,
    record: dict[str, Any],
    started: float,
    deadline: Deadline,
) -> _DesignEpochSelection:
    """Execute all preregistered trials and select a target-reaching measured route."""

    _ledger = StageLedger(record, deadline=deadline)
    baseline, baseline_stage = _ledger.run(
        "independent_baseline_physical_error",
        lambda: _measured(
            lambda: _design_qualification_physical(
                current, scalar, accepted.fields[0], 10
            )
        ),
    )
    baseline_evidence = PhysicalErrorEvidence(
        current.result_id,
        "represented-surface-reaction-L2",
        "baseline-host-Duffy-10",
        field_error=baseline["field_L2_error"],
        geometry_error=0.0,
        algebraic_error=0.0,
        transfer_error=0.0,
    )
    epoch_record: dict[str, Any] = {
        "epoch": epoch,
        "source": _result_identity(current),
        "baseline": baseline,
        "baseline_physical_error_seconds": baseline_stage["wall_seconds"],
        "trials": [],
    }
    record["epochs"].append(epoch_record)
    alternatives, trials = _design_qualification_candidates(
        current,
        accepted,
        scalar,
        request,
        baseline_evidence,
        budget,
        args,
        epoch_record,
        record,
        started,
    )

    epoch_record["observed_campaign_seconds"] = _ledger.run(
        "solver_aware_selection",
        lambda: sum(row["observed_attempt_seconds"] for row in epoch_record["trials"]),
    )
    if perf_counter() - started >= args.timeout:
        raise qualification_failure(
            "_design_qualification_epochs",
            (
                "Native design campaign, including failed attempts, exhausted its wall budget."
            ),
            {"args": args, "perf_counter": perf_counter, "started": started},
        )
    incomplete = tuple(
        row["proposal"] for row in epoch_record["trials"] if row["status"] != "completed"
    )
    if incomplete:
        raise qualification_failure(
            "_design_qualification_epochs",
            (
                "Preregistered native design comparison omitted measured trials: "
                + ", ".join(incomplete)
            ),
            {"incomplete": incomplete},
        )
    epoch_record["observed_comparison"] = _design_qualification_comparison(
        epoch_record["trials"]
    )
    decision, timing = _ledger.run(
        record.get("active_phase", "SolverAwareDecision"),
        lambda: _measured(
            lambda: SolverAwareDecision(
                baseline_evidence,
                budget,
                tuple(alternatives.values()),
                observed_campaign_seconds=epoch_record["observed_campaign_seconds"],
            )
        ),
    )
    epoch_record["selection_seconds"] = timing["wall_seconds"]
    epoch_record["dispositions"] = decision.dispositions
    if decision.selected is None:
        raise qualification_failure(
            "_design_qualification_epochs",
            (
                "No measured native candidate improves and attains the immutable physical target."
            ),
            {"decision": decision},
        )
    selected = trials[decision.selected.candidate_id]
    epoch_record["selected_proposal"] = selected["proposal"]
    return _DesignEpochSelection(
        epoch_record=epoch_record,
        trials=trials,
        decision=decision,
        candidate=decision.selected,
        selected=selected,
    )


def _design_qualification_continue_epoch(
    current: phx.meshing.CellMeshingResult,
    accepted: phx.solver.FiniteElementAcceptedState,
    selection: _DesignEpochSelection,
    args: QualificationControls,
    record: dict[str, Any],
    deadline: Deadline,
) -> _DesignContinuedEpoch:
    """Commit the selected all-field transaction and continue its carried solution."""

    _ledger = StageLedger(record, deadline=deadline)
    selected, decision, epoch_record = (
        selection.selected,
        selection.decision,
        selection.epoch_record,
    )
    committed, timing = _ledger.run(
        "accepted_all_field_reanalysis",
        lambda: _measured(
            lambda: selected["executor"].execute(
                accepted,
                current.mesh,
                selected["adaptation"],
                decision=decision,
                reanalysis=selected["reanalysis"],
                compiled_layout_id=selected["problem"].compilation_id,
            )
        ),
    )
    if (
        not bool(committed.committed)
        or committed.receipt is None
        or not committed.receipt.published
    ):
        raise qualification_failure(
            "_design_qualification_epochs",
            (committed.diagnostics),
            {"committed": committed},
        )
    epoch_record["acceptance_seconds"] = timing["wall_seconds"]
    epoch_record["receipt_id"] = committed.receipt.receipt_id
    epoch_record["receipt_published"] = committed.receipt.published
    epoch_record["source_accepted_state_id"] = accepted.accepted_id
    epoch_record["selected_candidate_id"] = selection.candidate.candidate_id
    epoch_record["result"] = _result_identity(selected["adaptation"].target)
    accepted, current, scalar = (
        committed.state,
        selected["adaptation"].target,
        selected["scalar"],
    )
    record["accepted_state"] = {
        "accepted_id": accepted.accepted_id,
        "topology_id": accepted.topology_id,
        "transition_id": accepted.transition_id,
        "state_version": accepted.state_version,
        "prepared_id": accepted.prepared_id,
        "compilation_id": accepted.compilation_id,
    }

    continued, compiler, continued_stages = _ledger.run(
        "continued_solve",
        lambda: _compiled_solver_stages(
            selected["problem"], accepted.fields[0], args.repeats
        ),
    )
    continued = _design_qualification_array(continued)
    continued_physical = _design_qualification_physical(current, scalar, continued, 10)
    epoch_record["uninterrupted_continuation"] = {
        "seed_fingerprint": array_tree_fingerprint(accepted.fields[0]),
        "fingerprint": array_tree_fingerprint(continued),
        "change_from_accepted": float(jnp.max(jnp.abs(continued - accepted.fields[0]))),
    }
    actual = selected["reanalysis"](current.mesh, accepted.fields, None, None)
    reanalysis_issues = decision.reanalysis_issues(actual)
    epoch_record["independent_reanalysis"] = {
        "evidence_id": actual.evidence_id,
        "estimator_id": actual.estimator_id,
        "selected_estimator_id": selection.candidate.error.estimator_id,
        "bound": actual.bound,
        "issues": list(reanalysis_issues),
    }
    decision.require_reanalysis(actual)
    # The accepted epoch's solver evidence uses the shared epoch attribution keys.
    record["stages_seconds"].update(
        {
            epoch_stage_name(name, epoch_record["epoch"]): seconds
            for name, seconds in continued_stages.items()
        }
    )
    epoch_record.update(
        {
            "continued_physical": continued_physical,
            "compiler": compiler,
            "retained_bytes": model_array_bytes(
                (current, scalar, selected["problem"], accepted, continued)
            ),
        }
    )
    return _DesignContinuedEpoch(
        accepted=accepted,
        current=current,
        scalar=scalar,
        physical=continued_physical,
        values=continued,
    )


def _design_qualification_retain_source(
    initial: _DesignInitialEpoch,
    selection: _DesignEpochSelection,
    continued: _DesignContinuedEpoch,
    derivative_owners: list[phx.meshing.FixedEpochDerivativeEvidence],
    retained_source_states: list[_DesignRetainedSource],
) -> None:
    """Retain the admitted learned and derivative owners for cold source continuation."""

    selected, trials, decision = selection.selected, selection.trials, selection.decision
    accepted, current = continued.accepted, continued.current
    learned = next(trial for trial in trials.values() if trial["proposal"] == "learned")
    proposer, features, transaction = (
        learned["learned_proposer"],
        learned["learned_features"],
        learned["proposal_transaction"],
    )
    if (
        proposer is None
        or features is None
        or transaction is None
        or not derivative_owners
    ):
        raise qualification_failure(
            "_design_qualification_epochs",
            ("Source cold state lost its learned/derivative owners."),
            {
                "derivative_owners": derivative_owners,
                "features": features,
                "proposer": proposer,
                "transaction": transaction,
            },
        )
    retained_source = initial.plan.source
    if not isinstance(retained_source, phx.meshing.NativeImplicitSource):
        raise TypeError(
            f"_design_qualification_retain_source.retained_source must be a NativeImplicitSource; got {type(retained_source).__name__!r}."
        )
    retained_source_states.append(
        _DesignRetainedSource(
            state=NativeDesignSourceState(
                geometry=initial.geometry,
                source=retained_source,
                specification=initial.request,
                options=initial.plan.options,
                plan=initial.plan,
                initial=initial.result,
                initial_state=initial.accepted,
                source_fields=initial.fields,
                accepted=current,
                accepted_state=accepted,
                target_fields=selected["all_fields"],
                field_specs=selected["executor"].fields,
                learned_proposer=proposer,
                learned_features=features,
                learned_transaction=transaction,
                decision=decision,
                adaptation=selected["adaptation"],
                derivative=derivative_owners[-1],
                physical=selected["reanalysis"](current.mesh, accepted.fields, None, None),
            ),
            uninterrupted_continuation=continued.values,
        )
    )


def _design_qualification_epochs(
    geometry: phx.geometry.CompiledGeometry,
    plan: phx.meshing.NativeMeshingPlan,
    request: phx.meshing.SurfaceMeshingSpec,
    current: phx.meshing.CellMeshingResult,
    accepted: phx.solver.FiniteElementAcceptedState,
    scalar: phx.discretization.FiniteElementDiscretization,
    all_fields: phx.discretization.FiniteElementDiscretization,
    source_problem: phx.equations.CompiledFiniteElementProblem,
    args: QualificationControls,
    budget: phx.meshing.DecisionBudget,
    record: dict[str, Any],
    started: float,
    *,
    deadline: Deadline,
    retained_source_states: list[_DesignRetainedSource] | None = None,
) -> None:
    """Compare all preregistered native proposals, publish one decision and continue its solve."""

    _ledger = StageLedger(record, deadline=deadline)

    initial = _DesignInitialEpoch(
        geometry=geometry,
        plan=plan,
        request=request,
        result=current,
        accepted=accepted,
        fields=all_fields,
    )
    derivative_owners: list[phx.meshing.FixedEpochDerivativeEvidence] = []
    for epoch in range(args.adaptation_rounds):
        if perf_counter() - started >= args.timeout:
            raise qualification_failure(
                "_design_qualification_epochs",
                ("Native design campaign exhausted its immutable wall budget."),
                {"args": args, "perf_counter": perf_counter, "started": started},
            )

        selection = _design_qualification_select_epoch(
            current,
            accepted,
            scalar,
            request,
            budget,
            args,
            epoch,
            record,
            started,
            deadline,
        )
        selected = selection.selected
        if epoch == 0:
            derivatives, timing = _ledger.run(
                "fixed_epoch_derivatives",
                lambda: _measured(
                    lambda: _design_qualification_derivatives(
                        geometry,
                        plan,
                        current,
                        scalar,
                        source_problem,
                        selected,
                        record["fixed_epoch_geometry_pde"],
                        retained_evidence=derivative_owners
                        if retained_source_states is not None
                        else None,
                    )
                ),
            )
            record["fixed_epoch_derivatives"] = derivatives
            record["stages_seconds"]["fixed_epoch_derivatives"] = timing["wall_seconds"]
        if perf_counter() - started >= args.timeout:
            raise qualification_failure(
                "_design_qualification_epochs",
                (
                    "Fixed-epoch derivative work exhausted the immutable wall budget before acceptance."
                ),
                {"args": args, "perf_counter": perf_counter, "started": started},
            )

        continued = _design_qualification_continue_epoch(
            current, accepted, selection, args, record, deadline
        )
        accepted, current, scalar = (
            continued.accepted,
            continued.current,
            continued.scalar,
        )
        continued_physical = continued.physical
        if continued_physical["field_L2_error"] <= args.target_error:
            record["time_to_target_error_seconds"] = perf_counter() - started
            record["physical_error"] = continued_physical["field_L2_error"]
            record["independent_scientific_certification"] = "passed"
            record["status"] = "passed"
            if retained_source_states is not None:
                _design_qualification_retain_source(
                    initial,
                    selection,
                    continued,
                    derivative_owners,
                    retained_source_states,
                )
            return
    raise qualification_failure(
        "_design_qualification_epochs",
        (
            "Bounded native design epochs did not attain the immutable physical error target."
        ),
        {},
    )


@final
class _DesignSourcePublication(phx.StrictModule):
    geometry: phx.geometry.CompiledGeometry
    plan: phx.meshing.NativeMeshingPlan
    request: phx.meshing.SurfaceMeshingSpec
    current: phx.meshing.CellMeshingResult


@final
class _DesignBaseline(phx.StrictModule):
    scalar: phx.discretization.FiniteElementDiscretization
    all_fields: phx.discretization.FiniteElementDiscretization
    accepted: phx.solver.FiniteElementAcceptedState
    problem: phx.equations.CompiledFiniteElementProblem


def _design_qualification_publish_source(
    args: QualificationControls,
    case: Mapping[str, Any],
    native: list[NativeMeshingPhaseMeasurement],
    record: dict[str, Any],
    deadline: Deadline,
) -> _DesignSourcePublication:
    """Publish the design's native source and independently certify its realization."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    fixture, timing = _ledger.run(
        record.get("active_phase", "_design_qualification_source"),
        lambda: _measured(lambda: _design_qualification_source(args, case)),
    )
    geometry, source, request, options = fixture
    stages["source_preparation"] = timing["wall_seconds"]
    record["source"] = {
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "design_state": {
            str(parameter_id): np.asarray(value).tolist()
            for parameter_id, value in zip(
                geometry.schema.parameter_ids, geometry.state.values, strict=True
            )
        },
    }
    record["request"] = {
        "specification_id": request.specification_id,
        "scope_id": request.scope.scope_id,
        "limits_id": request.limits.limits_id,
        "options_id": options.options_id,
    }

    plan, timing = _ledger.run(
        "native_geometry_preparation",
        lambda: _measured(
            lambda: phx.meshing.NativeMeshingProvider(options).plan(
                source,
                request,
                coordinate_contract=phx.SpatialCoordinateContract.si(),
                record_phase=native.append,
            )
        ),
    )
    stages["native_geometry_preparation"] = timing["wall_seconds"]
    record["plan_id"] = plan.plan_id
    record["source_retained_bytes"] = model_array_bytes(
        (geometry, source, request, options, plan)
    )

    current, timing = _ledger.run(
        "native_design_realization_publication",
        lambda: _measured(
            lambda: plan.execute(geometry.state, record_phase=native.append)
        ),
    )
    stages["native_design_realization_publication"] = timing["wall_seconds"]
    record["generation"] = _result_identity(current)
    record["case_observation"] = {
        "lattice_points_per_axis": [
            int(np.asarray(axis.point_coordinates).shape[0])
            for axis in source.grid.structured_axes
        ],
        "published_vertices": int(current.mesh.coordinates.shape[0]),
        "published_cells": int(current.audit.entity_counts[-1]),
    }

    acceptance, timing = _ledger.run(
        "independent_source_acceptance",
        lambda: _measured(
            lambda: _design_qualification_source_acceptance(current, geometry)
        ),
    )
    record["independent_source_acceptance"] = acceptance
    stages["independent_source_acceptance"] = timing["wall_seconds"]
    return _DesignSourcePublication(
        geometry=geometry, plan=plan, request=request, current=current
    )


def _design_qualification_baseline(
    publication: _DesignSourcePublication,
    args: QualificationControls,
    record: dict[str, Any],
    deadline: Deadline,
    *,
    stage_prefix: str = "",
) -> _DesignBaseline:
    """Solve and probe one published design epoch, then author its physical fields."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    geometry, plan, current = publication.geometry, publication.plan, publication.current
    bundle, timing = _ledger.run(
        f"{stage_prefix}baseline_solver_preparation",
        lambda: _measured(lambda: _design_qualification_prepare(current)),
    )
    scalar, all_fields, problem, _ = bundle
    stages[f"{stage_prefix}baseline_solver_preparation"] = timing["wall_seconds"]
    seed = jax.device_put(
        np.zeros((scalar.field_spaces[0].layout.size,), dtype=np.float64)
    )

    values, compiler, solver_stages = _ledger.run(
        f"{stage_prefix}baseline_solver",
        lambda: _compiled_solver_stages(problem, seed, args.repeats),
    )
    values = _design_qualification_array(values)
    stages.update(
        {f"{stage_prefix}{name}": seconds for name, seconds in solver_stages.items()}
    )
    record["baseline_compiler"] = compiler

    derivatives, timing = _ledger.run(
        f"{stage_prefix}fixed_epoch_geometry_pde",
        lambda: _measured(
            lambda: _design_qualification_derivatives(
                geometry, plan, current, scalar, problem, None
            )
        ),
    )
    record["fixed_epoch_geometry_pde"] = derivatives
    stages[f"{stage_prefix}fixed_epoch_geometry_pde"] = timing["wall_seconds"]
    condition = derivatives["operator_condition"]
    record["baseline_operator_condition"] = condition
    if not isfinite(condition) or condition > args.maximum_condition:
        raise qualification_failure(
            "_design_qualification_baseline",
            ("The baseline scalar operator exceeds the frozen condition budget."),
            {"condition": condition, "maximum": args.maximum_condition},
        )
    density = jax.device_put(
        np.full((all_fields.field_spaces[1].layout.size,), 2.0, dtype=np.float64)
    )
    history = jax.device_put(
        np.asarray(all_fields.dof_maps[2].dof_coordinates)[:, 1] + 0.5
    )
    accepted = phx.solver.FiniteElementAcceptedState(
        (values, density, history),
        0.0,
        0,
        current.mesh.topology_id,
        all_fields.prepared_id,
        problem.compilation_id,
    )
    return _DesignBaseline(
        scalar=scalar, all_fields=all_fields, accepted=accepted, problem=problem
    )


def _design_qualification_design_step(
    publication: _DesignSourcePublication,
    baseline: _DesignBaseline,
    case: Mapping[str, Any],
    native: list[NativeMeshingPhaseMeasurement],
    record: dict[str, Any],
    deadline: Deadline,
) -> _DesignSourcePublication:
    """Solve the represented-QoI design equation inside the event-free epoch.

    The preregistered target is the authored continuous sphere moment
    ∫_S x² dA = 4πR⁴/3. The safeguarded Newton solve uses the qualified
    reverse-mode adjoint, is bracketed strictly inside the predicted lattice
    events, and the designed state is republished through the same native plan.
    """

    _ledger = StageLedger(record, deadline=deadline)
    geometry, plan, current = publication.geometry, publication.plan, publication.current
    initial = record["fixed_epoch_geometry_pde"]
    route = _design_qualification_fixed_route(
        geometry, plan, current, baseline.scalar, baseline.problem
    )
    radius = float(route.parameter)
    target = 4.0 * pi * radius**4 / 3.0
    events = initial["parameter_events"]
    if events["lower"] is None or events["upper"] is None:
        raise qualification_failure(
            "_design_qualification_design_step",
            ("The design bracket requires a predicted lattice event on both sides."),
            {"events": events},
        )
    bracket = (
        radius + _DESIGN_BRACKET_FRACTION * float(events["lower"]),
        radius + _DESIGN_BRACKET_FRACTION * float(events["upper"]),
    )
    cells = tuple(np.asarray(item) for item in route.cells)
    endpoints: list[dict[str, Any]] = []
    for value in bracket:
        margins, vertices, status = _design_qualification_event_margins(
            route.surface_plan,
            route.state.replace_at(
                route.radius_index, jnp.asarray(value, dtype=jnp.float64)
            ),
        )
        endpoints.append({"radius": value, "status": status, "margins": margins})
        if status != int(ImplicitSurfaceStatus.SUCCESS) or any(
            margin <= 0.0 for _, margin in margins
        ):
            raise qualification_failure(
                "_design_qualification_design_step",
                ("The design bracket endpoint crosses a fixed-epoch realization event."),
                {"endpoint": endpoints[-1]},
            )
        # The represented polyhedron differs from the smooth sphere, so the
        # equation is posed only if the solve-free moment straddles the target.
        endpoints[-1]["solve_free_residual"] = (
            _design_qualification_moment_gradient(vertices, cells)[0] - target
        )
    if not endpoints[0]["solve_free_residual"] < 0.0 < endpoints[1]["solve_free_residual"]:
        raise qualification_failure(
            "_design_qualification_design_step",
            (
                "The represented-surface moment does not straddle the authored sphere target inside the event-free bracket."
            ),
            {"endpoints": endpoints, "target": target},
        )
    problem = ScalarRootProblem(
        eqx.Partial(
            _design_qualification_root_residual,
            route.qoi,
            jnp.asarray(target, dtype=jnp.float64),
        ),
        bracket=(
            jnp.asarray(bracket[0], dtype=jnp.float64),
            jnp.asarray(bracket[1], dtype=jnp.float64),
        ),
        derivative=eqx.Partial(_design_qualification_root_derivative, route.qoi),
        validity=eqx.Partial(
            _design_qualification_root_validity,
            route.surface_plan,
            route.state,
            route.radius_index,
        ),
        problem_id="design-sphere-radius-represented-QoI",
    )
    solved, timing = _ledger.run(
        "fixed_epoch_design_root",
        lambda: _measured(
            lambda: jax.block_until_ready(_design_qualification_root(problem))
        ),
    )
    record["stages_seconds"]["fixed_epoch_design_root"] = timing["wall_seconds"]
    status = int(np.asarray(solved.status))
    root = float(np.asarray(solved.root))
    diagnostics = solved.nonlinear_result.diagnostics
    step = {
        "objective": "represented-surface integral of the solved P1 field equals the authored continuous sphere moment 4*pi*R^4/3",
        "design_target": case["design_target"],
        "target_QoI": target,
        "initial_radius": radius,
        "initial_QoI": initial["adjoint"]["value"],
        "event_free_bracket": list(bracket),
        "bracket_fraction": _DESIGN_BRACKET_FRACTION,
        "bracket_endpoints": endpoints,
        "method": SafeguardedNewton().method_id,
        "derivative": "qualified reverse-mode adjoint of the fixed-epoch route",
        "status": NonlinearStatus(status).name,
        "bracket_valid": bool(np.asarray(solved.bracket_valid)),
        "root_radius": root,
        "residual": float(np.asarray(solved.value)),
        "residual_bound": _DESIGN_QOI_RESIDUAL,
        "iterations": int(np.asarray(diagnostics.iterations)),
        "residual_evaluations": int(np.asarray(diagnostics.residual_evaluations)),
        "derivative_evaluations": int(np.asarray(diagnostics.jvp_evaluations)),
        "root_inside_bracket": bracket[0] < root < bracket[1],
        "initial_epoch": {
            "result_id": current.result_id,
            "derivatives": initial,
            "baseline_compiler": record["baseline_compiler"],
            "baseline_operator_condition": record["baseline_operator_condition"],
        },
    }
    record["design_step"] = step
    if (
        status != int(NonlinearStatus.SUCCESS)
        or not step["bracket_valid"]
        or not step["root_inside_bracket"]
    ):
        raise qualification_failure(
            "_design_qualification_design_step",
            (
                "The represented-QoI design equation did not converge inside its event-free bracket."
            ),
            {"design_step": step},
        )
    designed_state = geometry.state.replace_at(
        route.radius_index, jnp.asarray(root, dtype=jnp.float64)
    )
    margins, vertices, realization_status = _design_qualification_event_margins(
        route.surface_plan, designed_state
    )
    moment, _ = _design_qualification_moment_gradient(
        vertices, tuple(np.asarray(item) for item in route.cells)
    )
    step.update(
        {
            "root_event_margins": margins,
            "root_realization_status": realization_status,
            "independent_solve_free_QoI": moment,
            "independent_QoI_error": abs(moment - target),
            "independent_QoI_bound": _DESIGN_INDEPENDENT_QOI_BOUND,
        }
    )
    if (
        realization_status != int(ImplicitSurfaceStatus.SUCCESS)
        or any(margin <= 0.0 for _, margin in margins)
        or not abs(moment - target) <= _DESIGN_INDEPENDENT_QOI_BOUND
    ):
        raise qualification_failure(
            "_design_qualification_design_step",
            (
                "The designed state failed its solve-free QoI or fixed-epoch realization check."
            ),
            {"design_step": step},
        )
    designed = geometry.with_state(designed_state)
    republished, timing = _ledger.run(
        "native_designed_realization_publication",
        lambda: _measured(
            lambda: plan.execute(designed_state, record_phase=native.append)
        ),
    )
    record["stages_seconds"]["native_designed_realization_publication"] = timing[
        "wall_seconds"
    ]
    acceptance, timing = _ledger.run(
        "independent_designed_source_acceptance",
        lambda: _measured(
            lambda: _design_qualification_source_acceptance(republished, designed)
        ),
    )
    record["stages_seconds"]["independent_designed_source_acceptance"] = timing[
        "wall_seconds"
    ]
    step.update(
        {
            "published": _result_identity(republished),
            "source_acceptance": acceptance,
            "geometry_realization_event": canonical_fingerprint(
                {
                    "kind": "design-geometry-realization-event",
                    "from": current.result_id,
                    "to": republished.result_id,
                    "radius": root,
                }
            ),
        }
    )
    return _DesignSourcePublication(
        geometry=designed, plan=plan, request=publication.request, current=republished
    )


def _design_qualification_record(
    args: QualificationControls,
    budget: phx.meshing.DecisionBudget,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Initialize the preregistered targets, comparators, and external evidence."""

    record: dict[str, Any] = {
        "scenario": "native-design-lifecycle",
        "status": "failed",
        "runtime": _runtime_identity(),
        "case": dict(case),
        "decision_budget_id": budget.budget_id,
        "targets": {
            "physical_L2_error": args.target_error,
            "wall_seconds": args.timeout,
            "maximum_memory_bytes": args.maximum_memory_bytes,
            "maximum_condition": args.maximum_condition,
            "derivative_finite_difference_error": _DERIVATIVE_FD_BOUND,
            "derivative_duality_error": _DERIVATIVE_DUALITY_BOUND,
            "derivative_step": _DERIVATIVE_STEP,
            "adjoint_analytic_reference_error": _DERIVATIVE_FD_BOUND,
            "taylor_fractions_of_event_distance": list(_TAYLOR_FRACTIONS),
            "taylor_slope_deviation": _TAYLOR_SLOPE_DEVIATION_BOUND,
            "event_probe_fraction": _EVENT_PROBE_FRACTION,
            "learned_training_relative_residual": _LEARNED_TRAINING_RESIDUAL_BOUND,
            "learned_held_out_rank_correlation": _LEARNED_RANK_CORRELATION_BOUND,
            "design_bracket_fraction_of_event_distance": _DESIGN_BRACKET_FRACTION,
            "design_QoI_residual": _DESIGN_QOI_RESIDUAL,
            "design_independent_QoI_error": _DESIGN_INDEPENDENT_QOI_BOUND,
            "recomputed_solution_defect": 1.0e-10,
            "declared_field_error": 1.0e-10,
            "material_inventory_error": 1.0e-9,
        },
        "controls": {
            "design_case": args.design_case,
            "resolution": args.resolution,
            "capacity": args.capacity,
            "adaptation_rounds": args.adaptation_rounds,
            "repeats": args.repeats,
            "maximum_work_units": args.maximum_work_units,
            "timeout": args.timeout,
        },
        "stages_seconds": {},
        "epochs": [],
        "failed_positive_attempts": [],
        "time_to_target_error_seconds": None,
        "mandatory_workflow_completion": "incomplete",
        "case_contract_completion": "incomplete",
        "independent_scientific_certification": "unassessed",
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
        "PDE": "u=x^2, coercive scalar reaction on the actual embedded represented surface",
        "declared_fields": {
            "u": "scalar solution",
            "density": "nodal constant material density",
            "history": "advected affine history",
        },
        "comparators": [
            {"proposal": "uniform", "marks": "all actual source cell global IDs"},
            {
                "proposal": "analytic",
                "marks": "highest exact Jensen defects, ceil(cells/2), stable global-ID ties",
            },
            {
                "proposal": "learned",
                "teacher": "observed per-cell squared L2 errors of the uniform comparator's solved field on its refined mesh",
                "features": list(_LEARNED_FEATURE_IDS),
                "model": "deterministic least-squares LinearMarker",
                "trust": "training residual and held-out observed-error rank bounds before native admission",
                "marks": "trusted safety projection of positive model scores, at most ceil(cells/3)",
            },
        ],
        "physical_error_accounting": (
            "Independent total field L2 error includes algebraic and scalar refresh defects; "
            "do not add those terms again. Geometry is the declared represented surface; "
            "analytic-source fidelity is separately certified and independently probed."
        ),
        "nonclaims": [
            "Discovery, lattice-sign, retopology and acceptance are stopped events, never differentiated.",
            "The represented-surface PDE is not the exact-sphere diffusion PDE.",
            "No constitutive material history or learned-model generalization is claimed.",
            "No estimator superiority, full corpus or release claim.",
        ],
        "active_phase": "source_preparation",
    }
    return record


def _design_qualification_number(value: object) -> float | None:
    """Return a finite recorded numeric observation, refusing booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if isfinite(number) else None


def _design_qualification_derivative_observations(
    derivatives: Mapping[str, Any],
) -> dict[str, float | None]:
    """Recompute every derivative check observation from recorded raw values.

    Fixed-route probe errors are the measured JVP-vs-central and duality
    residuals; adjoint, moment-identity and Taylor observations are rebuilt
    from the recorded gradients, QoI values and remainders.
    """
    checks = derivatives.get("checks") or {}
    observed = {
        name: _design_qualification_number((checks.get(name) or {}).get("observed"))
        for name in _DESIGN_DERIVATIVE_BOUNDS
    }
    events = derivatives.get("parameter_events") or {}
    distances = [
        abs(value)
        for value in (
            _design_qualification_number(events.get("lower")),
            _design_qualification_number(events.get("upper")),
        )
        if value is not None and value != 0.0
    ]
    observed["finite_difference_step_event_fraction"] = (
        _DERIVATIVE_STEP * abs(_DERIVATIVE_DIRECTION) / min(distances)
        if distances
        else None
    )
    adjoint = derivatives.get("adjoint") or {}
    values = {
        name: _design_qualification_number(adjoint.get(name))
        for name in (
            "adjoint_gradient",
            "tangent_gradient",
            "finite_difference_gradient",
            "analytic_reference_gradient",
            "host_value",
            "solve_free_moment",
        )
    }
    gradient = values["adjoint_gradient"]
    for name, reference in (
        ("qoi_adjoint_tangent_duality", "tangent_gradient"),
        ("qoi_adjoint_finite_difference", "finite_difference_gradient"),
        ("qoi_adjoint_analytic_reference", "analytic_reference_gradient"),
    ):
        other = values[reference]
        observed[name] = (
            None if gradient is None or other is None else abs(gradient - other)
        )
    host, moment = values["host_value"], values["solve_free_moment"]
    observed["qoi_solve_free_moment_identity"] = (
        None if host is None or moment is None else abs(host - moment)
    )
    taylor = adjoint.get("taylor") or {}
    remainders = taylor.get("remainders")
    roundoff = _design_qualification_number(taylor.get("roundoff_bound"))
    for name in (
        "qoi_taylor_roundoff_ratio",
        "qoi_taylor_slope_deviation",
        "qoi_taylor_pairwise_slope_deviation",
    ):
        observed[name] = None
    if (
        isinstance(remainders, list)
        and tuple(taylor.get("fractions") or ()) == _TAYLOR_FRACTIONS
        and len(remainders) == len(_TAYLOR_FRACTIONS)
        and all(
            (number := _design_qualification_number(item)) is not None and number > 0.0
            for item in remainders
        )
    ):
        fitted, pairwise = _design_qualification_taylor_slopes(
            _TAYLOR_FRACTIONS, [float(item) for item in remainders]
        )
        observed["qoi_taylor_slope_deviation"] = abs(fitted - 2.0)
        observed["qoi_taylor_pairwise_slope_deviation"] = max(
            abs(slope - 2.0) for slope in pairwise
        )
        if roundoff is not None:
            observed["qoi_taylor_roundoff_ratio"] = roundoff / min(
                float(item) for item in remainders
            )
    return observed


def _design_qualification_derivatives_within_bounds(
    derivatives: Mapping[str, Any],
) -> bool:
    """Hold every recomputed derivative observation to its preregistered bound."""
    observed = _design_qualification_derivative_observations(derivatives)
    return set(derivatives.get("checks") or {}) == set(_DESIGN_DERIVATIVE_BOUNDS) and all(
        (value := observed[name]) is not None and value <= bound
        for name, bound in _DESIGN_DERIVATIVE_BOUNDS.items()
    )


def _design_qualification_named_margin(margins: object, name: str) -> float | None:
    """Return one recorded named event margin."""
    if not isinstance(margins, (list, tuple)):
        return None
    values = [
        _design_qualification_number(item[1])
        for item in margins
        if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == name
    ]
    return values[0] if len(values) == 1 else None


def _design_qualification_crossing_observed(crossing: Mapping[str, Any]) -> bool:
    """Recompute the lattice-event refusal from recorded owner statuses and margins."""
    sign_bit = int(ImplicitSurfaceStatus.SIGN_PATTERN_CHANGED)
    before = crossing.get("before") or {}
    after = crossing.get("after") or {}
    before_status, after_status = before.get("status"), after.get("status")
    if not isinstance(before_status, int) or not isinstance(after_status, int):
        return False
    before_margin = _design_qualification_named_margin(before.get("margins"), "lattice-sign")
    after_margin = _design_qualification_named_margin(after.get("margins"), "lattice-sign")
    return (
        not before_status & sign_bit
        and before_margin is not None
        and before_margin > 0.0
        and bool(after_status & sign_bit)
        and after_margin is not None
        and after_margin <= 0.0
    )


def _design_qualification_learner_trusted(trial: Mapping[str, Any]) -> bool:
    """Recompute the learned marker's trust bounds from its recorded observations."""
    trust = trial.get("trust_bounds") or {}
    training = trial.get("training") or {}
    scores = trial.get("raw_scores")
    residual = _design_qualification_number(
        (trust.get("training_relative_residual") or {}).get("observed")
    )
    correlation = _design_qualification_number(
        (trust.get("held_out_observed_error_rank_correlation") or {}).get("observed")
    )
    return (
        training.get("fit_rank") == len(_LEARNED_FEATURE_IDS)
        and isinstance(training.get("training_samples"), int)
        and training["training_samples"] > len(_LEARNED_FEATURE_IDS)
        and residual is not None
        and residual == _design_qualification_number(
            training.get("training_relative_residual")
        )
        and residual <= _LEARNED_TRAINING_RESIDUAL_BOUND
        and correlation is not None
        and correlation >= _LEARNED_RANK_CORRELATION_BOUND
        and isinstance(scores, list)
        and len(scores) == training.get("application_samples")
        and all(_design_qualification_number(score) is not None for score in scores)
    )


def _design_qualification_design_step_observed(
    design: Mapping[str, Any], case: Mapping[str, Any]
) -> bool:
    """Recompute the combined case's posed, bracketed and solved design equation."""
    radius = _design_qualification_number(design.get("initial_radius"))
    root = _design_qualification_number(design.get("root_radius"))
    moment = _design_qualification_number(design.get("independent_solve_free_QoI"))
    bracket = design.get("event_free_bracket")
    endpoints = design.get("bracket_endpoints")
    events = (
        ((design.get("initial_epoch") or {}).get("derivatives") or {}).get(
            "parameter_events"
        )
        or {}
    )
    lower = _design_qualification_number(events.get("lower"))
    upper = _design_qualification_number(events.get("upper"))
    if (
        radius is None
        or root is None
        or moment is None
        or lower is None
        or upper is None
        or radius != case.get("radius")
        or not isinstance(bracket, list)
        or len(bracket) != 2
        or not isinstance(endpoints, list)
        or len(endpoints) != 2
    ):
        return False
    target = 4.0 * pi * radius**4 / 3.0
    expected = (
        radius + _DESIGN_BRACKET_FRACTION * lower,
        radius + _DESIGN_BRACKET_FRACTION * upper,
    )
    residuals = [
        _design_qualification_number(endpoint.get("solve_free_residual"))
        for endpoint in endpoints
    ]
    return (
        lower < 0.0 < upper
        and tuple(bracket) == expected
        and [endpoint.get("radius") for endpoint in endpoints] == list(expected)
        and residuals[0] is not None
        and residuals[1] is not None
        and residuals[0] < 0.0 < residuals[1]
        and design.get("status") == NonlinearStatus.SUCCESS.name
        and expected[0] < root < expected[1]
        and abs(moment - target) <= _DESIGN_INDEPENDENT_QOI_BOUND
    )


def _design_qualification_selected_trial(epoch: Mapping[str, Any]) -> dict[str, Any]:
    selected = epoch.get("selected_candidate_id")
    return next(
        (
            trial
            for trial in epoch.get("trials", ())
            if selected is not None and trial.get("candidate_id") == selected
        ),
        {},
    )


def _design_qualification_learned_gates(
    epoch: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Return the epoch's trusted-learned and rollback-reuse observations."""
    learned = [
        trial for trial in epoch.get("trials", ()) if trial.get("proposal") == "learned"
    ]
    if len(learned) != 1 or learned[0].get("status") != "completed":
        return False, False
    trial = learned[0]
    proposal = trial.get("trusted_proposal") or {}
    refusals = trial.get("adversarial_refusals") or []
    training = trial.get("training") or {}
    acceptance = trial.get("learned_acceptance") or {}
    rollback = trial.get("physical_reanalysis_rollback") or {}
    defect = rollback.get("actual_staged_solution_defect")
    trusted = (
        _design_qualification_learner_trusted(trial)
        and proposal.get("safety_audit_passed") is True
        and proposal.get("compliance_issues") == []
        and len(refusals) == 2
        and all(refusal.get("expected_refusal") is True for refusal in refusals)
        and training.get("training_result_id") is not None
        and training.get("training_result_id")
        != training.get("application_source_result_id")
        and acceptance.get("receipt_published") is True
    )
    reused = (
        rollback.get("committed") is False
        and rollback.get("source_state_identically_retained") is True
        and rollback.get("source_mesh_identically_retained") is True
        and rollback.get("receipt_present") is True
        and rollback.get("receipt_published") is False
        and isinstance(defect, float)
        and defect > 1.0e-10
        and acceptance.get("source_accepted_state_id") == rollback.get("accepted_state_id")
        and epoch.get("source_accepted_state_id") == rollback.get("accepted_state_id")
        and epoch.get("receipt_published") is True
    )
    return trusted, reused


def _design_qualification_completion_gates(
    record: Mapping[str, Any], args: QualificationControls
) -> dict[str, bool]:
    """Derive every mandatory design-loop gate from recorded oracle observations."""
    epochs = list(record.get("epochs", ()))
    derivatives = record.get("fixed_epoch_derivatives") or {}
    events = derivatives.get("event_nondifferentiability") or {}
    crossing = events.get("lattice_crossing") or {}
    topology = events.get("topology_event") or {}
    qualifications = derivatives.get("qualifications") or {}
    selected = [_design_qualification_selected_trial(epoch) for epoch in epochs]
    learned = [_design_qualification_learned_gates(epoch) for epoch in epochs]
    completed = [
        trial
        for epoch in epochs
        for trial in epoch.get("trials", ())
        if trial.get("status") == "completed"
    ]
    case = _design_qualification_case(args)
    observed = record.get("case_observation") or {}
    lattice = _design_qualification_lattice_count(args, case)
    case_observed = (
        record.get("case") == case
        and observed.get("lattice_points_per_axis") == [lattice] * 3
    )
    if case["design_target"] is not None:
        design = record.get("design_step") or {}
        published = (design.get("published") or {}).get("result_id")
        case_observed = (
            case_observed
            and _design_qualification_design_step_observed(design, case)
            and published is not None
            and published != (design.get("initial_epoch") or {}).get("result_id")
            and derivatives.get("epoch_id") == published
            and bool(epochs)
            and epochs[0].get("source", {}).get("result_id") == published
        )
    final = epochs[-1].get("continued_physical", {}) if epochs else {}
    final_error = final.get("field_L2_error")
    initial_error = epochs[0].get("baseline", {}).get("field_L2_error") if epochs else None
    condition = record.get("baseline_operator_condition")
    end = record.get("end_to_end_seconds")
    peak = record.get("peak_process_resident_bytes")
    return {
        "fixed_epoch_derivatives_qualified": (
            _design_qualification_derivatives_within_bounds(derivatives)
            and set(qualifications) == {"geometry", "pde", "transfer"}
            and all(item.get("valid") is True for item in qualifications.values())
            and derivatives.get("binding_issues") == []
        ),
        "event_nondifferentiability_observed": (
            _design_qualification_crossing_observed(crossing)
            and {"event-boundary", "stopped-geometry-realization-event"}
            <= set(crossing.get("invalidation", ()))
            and "stopped-topology-event" in topology.get("invalidation", ())
            and topology.get("transition_is_derivative") is False
            and topology.get("transition_valid") is True
        ),
        "trusted_learned_proposal_admitted": bool(learned)
        and all(trusted for trusted, _ in learned),
        "native_events_accepted": bool(selected)
        and all(
            (trial.get("native_event") or {}).get("converged") is True
            and (trial.get("native_event") or {}).get("transition_present") is True
            for trial in selected
        ),
        "all_state_transfer_published": bool(epochs)
        and all(
            epoch.get("receipt_published") is True
            and trial.get("feasibility_issues") == []
            and len(trial.get("state_entries", ())) >= len(design_field_templates())
            for epoch, trial in zip(epochs, selected, strict=True)
        ),
        "independent_reanalysis_accepted": bool(epochs)
        and all(
            (epoch.get("independent_reanalysis") or {}).get("issues") == []
            and (epoch.get("independent_reanalysis") or {}).get("estimator_id")
            != (epoch.get("independent_reanalysis") or {}).get("selected_estimator_id")
            for epoch in epochs
        ),
        "failed_reanalysis_rollback_reused": bool(learned)
        and all(reused for _, reused in learned),
        "physical_objective_improved_to_target": (
            isinstance(final_error, float)
            and isinstance(initial_error, float)
            and final_error <= args.target_error
            and final_error < initial_error
            and record.get("physical_error") == final_error
        ),
        "resource_budgets_within": (
            isinstance(end, float)
            and end <= args.timeout
            and isinstance(peak, int)
            and not isinstance(peak, bool)
            and peak <= args.maximum_memory_bytes
            and isinstance(condition, float)
            and condition <= args.maximum_condition
            and bool(completed)
            and all(
                isinstance(trial.get("condition"), float)
                and trial["condition"] <= args.maximum_condition
                for trial in completed
            )
        ),
        "case_contract_observed": case_observed,
    }


def _design_qualification_completion(
    args: QualificationControls,
    native: list[NativeMeshingPhaseMeasurement],
    record: dict[str, Any],
    started: float,
) -> None:
    """Record process resources and enforce the campaign's frozen time and memory budgets."""

    record["native_phases"] = _phase_recordings(native)
    record["end_to_end_seconds"] = perf_counter() - started
    record["peak_process_resident_bytes"] = peak_resident_bytes()
    record["memory_scope"] = (
        f"{PEAK_RESIDENT_SCOPE}; retained artifact bytes reported separately"
    )
    record["device_memory"] = [
        {
            "device_id": device.id,
            "status": "unavailable" if statistics is None else "backend-reported",
            "statistics": statistics,
        }
        for device in jax.local_devices()
        for statistics in (device.memory_stats(),)
    ]
    if record["end_to_end_seconds"] > args.timeout:
        record["status"] = "timeout"
        record["failed_positive_attempts"].append(
            {
                "status": "timeout",
                **FailureRecord(
                    "end_to_end_wall_budget",
                    "ResourceBudgetExceeded",
                    "Frozen wall budget exceeded.",
                ).to_record(),
                "requested": {"maximum_wall_seconds": args.timeout},
                "achieved": {"wall_seconds": record["end_to_end_seconds"]},
                "safe_refusal_is_positive": False,
            }
        )
    peak = record["peak_process_resident_bytes"]
    if peak is None:
        # An unavailable peak cannot satisfy the frozen memory gate.
        record["memory_budget_status"] = "unassessed"
        if record["status"] == "passed":
            record["status"] = "failed"
        record["failed_positive_attempts"].append(
            {
                "status": "failed",
                **FailureRecord(
                    "process_peak_memory_budget",
                    "ResourceBudgetUnassessed",
                    f"Frozen memory budget unassessed: {PEAK_RESIDENT_SCOPE}.",
                ).to_record(),
                "requested": {"maximum_memory_bytes": args.maximum_memory_bytes},
                "achieved": {"peak_process_resident_bytes": None},
                "safe_refusal_is_positive": False,
            }
        )
    elif peak > args.maximum_memory_bytes:
        record["memory_budget_status"] = "exceeded"
        record["status"] = "failed"
        record["memory_budget_exceeded"] = True
        record["failed_positive_attempts"].append(
            {
                "status": "failed",
                **FailureRecord(
                    "process_peak_memory_budget",
                    "ResourceBudgetExceeded",
                    "Frozen memory budget exceeded.",
                ).to_record(),
                "requested": {"maximum_memory_bytes": args.maximum_memory_bytes},
                "achieved": {"peak_process_resident_bytes": peak},
                "safe_refusal_is_positive": False,
            }
        )
    else:
        record["memory_budget_status"] = "within"
    gates = _design_qualification_completion_gates(record, args)
    record["completion_gates"] = gates
    completion = (
        "complete"
        if record["status"] == "passed" and all(gates.values())
        else "incomplete"
    )
    record["case_contract_completion"] = completion
    record["mandatory_workflow_completion"] = completion
    del record["active_phase"]


def _design_qualification_lifecycle(
    args: QualificationControls,
    *,
    retained_source_states: list[_DesignRetainedSource] | None = None,
) -> dict[str, object]:
    """Execute DesignState → derivatives → trusted native event → accepted objective."""

    if (
        min(
            args.resolution,
            args.capacity,
            args.repeats,
            args.adaptation_rounds,
            args.maximum_memory_bytes,
        )
        < 1
    ):
        raise ValueError(
            f"_design_qualification_lifecycle.bounds must have positive resolution, capacity, repeats, adaptation rounds and maximum memory bytes; got {(args.resolution, args.capacity, args.repeats, args.adaptation_rounds, args.maximum_memory_bytes)!r}."
        )
    if not np.isfinite(args.timeout + args.target_error + args.maximum_condition) or (
        args.timeout <= 0.0 or args.target_error <= 0.0 or args.maximum_condition < 1.0
    ):
        raise ValueError(
            f"_design_qualification_lifecycle.bounds must have finite positive timeout and target error and maximum condition at least one; got {(args.timeout, args.target_error, args.maximum_condition)!r}."
        )
    case = _design_qualification_case(args)
    budget = DecisionBudget(
        tolerance=args.target_error,
        maximum_wall_seconds=args.timeout,
        maximum_memory_bytes=args.maximum_memory_bytes,
        maximum_dofs=3 * args.capacity,
        maximum_condition=args.maximum_condition,
    )
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    native: list[NativeMeshingPhaseMeasurement] = []
    record = _design_qualification_record(args, budget, case)
    try:
        publication = _design_qualification_publish_source(
            args, case, native, record, deadline
        )
        baseline = _design_qualification_baseline(publication, args, record, deadline)
        if case["design_target"] is not None:
            publication = _design_qualification_design_step(
                publication, baseline, case, native, record, deadline
            )
            # The realization event stops the initial epoch's derivatives; the
            # designed epoch is solved and requalified before any proposal.
            baseline = _design_qualification_baseline(
                publication, args, record, deadline, stage_prefix="designed_"
            )
        _design_qualification_epochs(
            publication.geometry,
            publication.plan,
            publication.request,
            publication.current,
            baseline.accepted,
            baseline.scalar,
            baseline.all_fields,
            baseline.problem,
            args,
            budget,
            record,
            started,
            deadline=deadline,
            retained_source_states=retained_source_states,
        )
    except (
        phx.meshing.MeshingFailure,
        RuntimeError,
        ValueError,
        TypeError,
        TimeoutError,
    ) as failure:
        detail = {
            **FailureRecord.from_exception(failure, record["active_phase"]).to_record(),
            "status": "failed",
            "safe_refusal_is_positive": False,
        }
        record.update(detail)
        record["failed_positive_attempts"].append(detail)
    finally:
        _design_qualification_completion(args, native, record, started)
    return record


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return _design_qualification_lifecycle(args)
