# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import resource
from collections.abc import Callable
from time import perf_counter
from typing import Any, final, TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import canonical_fingerprint, model_array_bytes
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
from phydrax.geometry.implicit import AnalyticImplicitProfile
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
from phydrax.units import METER
from tools.meshing_qualification._controls import (
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._physics import (
    _compiled_solver_stages,
    _solve_corpus_problem,
    CorpusCompilerEvidence,
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
    peak_resident_bytes,
    qualification_failure,
    StageLedger,
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


def _design_qualification_source(
    args: QualificationControls,
) -> tuple[
    phx.geometry.CompiledGeometry,
    phx.meshing.NativeImplicitSource,
    phx.meshing.SurfaceMeshingSpec,
    phx.meshing.NativeMeshingOptions,
]:
    """Bind the fixed-lattice sphere source and immutable request budgets."""
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), 0.75, feature_id="design-sphere"
    ).compile(coordinate_unit=METER)
    count = 2 * args.resolution + 1
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformAxisSpec(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(np.asarray(((-1.4,) * 3, (1.4,) * 3), dtype=np.float64))
    revision = canonical_fingerprint(
        {
            "kind": "design-qualification-source",
            "parameters": tuple(np.asarray(value) for value in geometry.state.values),
            "parameter_ids": tuple(str(item) for item in geometry.schema.parameter_ids),
            "lattice_count": count,
            "bounds": ((-1.4,) * 3, (1.4,) * 3),
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
                scope, 2.8 / (count - 1), strength=phx.meshing.SizeControlStrength.SOFT
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
) -> tuple[
    phx.discretization.FiniteElementDiscretization,
    phx.discretization.FiniteElementDiscretization,
    phx.equations.CompiledFiniteElementProblem,
    tuple[phx.discretization.FiniteElementFieldSpec, ...],
]:
    """Prepare the embedded scalar reaction PDE and all declared FE fields."""
    fields = tuple(field for _, field in design_field_templates())
    scalar = phx.discretization.FiniteElementPlan(
        result.mesh, fields[0], coordinate_spec=result.geometry
    ).prepare()
    all_fields = phx.discretization.FiniteElementPlan(
        result.mesh, fields, coordinate_spec=result.geometry
    ).prepare()
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
    problem = phx.equations.compile_finite_element_problem(form, scalar)
    return scalar, all_fields, problem, fields


def _design_qualification_physical(
    result: phx.meshing.CellMeshingResult,
    space: phx.discretization.FiniteElementDiscretization,
    values: jax.Array,
    count: int,
) -> dict[str, float]:
    """Integrate the represented-surface QoI independently with host Duffy quadrature."""

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

    def quadrature(order: int) -> tuple[float, float]:
        axis, weight = np.polynomial.legendre.leggauss(order)
        axis, weight = (axis + 1.0) * 0.5, weight * 0.5
        first, second = np.meshgrid(axis, axis, indexing="ij")
        barycentric = np.stack(
            (1.0 - first - (1.0 - first) * second, first, (1.0 - first) * second), axis=-1
        ).reshape(-1, 3)
        weights = (np.outer(weight, weight) * (1.0 - first)).reshape(-1)
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
    """Bind field transports to the solved candidate and physical oracle."""

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
        independent, solved = _solve_corpus_problem(problem, seed)
        jax.block_until_ready((independent, solved))
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


def _design_qualification_learned(
    source: phx.meshing.CellMeshingResult,
    limits: phx.meshing.MeshingLimits,
    trial: dict[str, Any],
) -> tuple[
    phx.meshing.MeshAdaptationResult,
    phx.meshing.MeshProposalTransaction,
    dict[str, float],
    phx.meshing.LearnedMeshProposer,
    phx.meshing.MeshProposalFeatures,
]:
    """Fit a real supervised marker; all topology and proof remain native-owned."""

    trial["active_phase"] = "learned_feature_preparation"
    start = perf_counter()
    ids = np.asarray(source.mesh.blocks[0].global_ids)
    corners = np.asarray(source.mesh.coordinates)[
        np.asarray(source.mesh.blocks[0].vertices)
    ]
    scores = np.mean(corners[..., 0] ** 2, axis=1) - np.mean(corners[..., 0], axis=1) ** 2
    order = np.argsort(ids, kind="stable")
    features = np.stack((np.ones_like(scores), scores), axis=1)[order]
    teacher = scores[order]
    scope = mesh_proposal_scope(source, source.mesh.topological_dimension)
    bound = MeshProposalFeatures(
        source,
        scope,
        features,
        feature_ids=("constant-feature", "physical-simplex-x-squared-Jensen-defect"),
        feature_owner_id="analytic-ambient-x-squared-reaction",
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
    phases = {"learned_features_safety_preparation": perf_counter() - start}
    trial["phase_seconds"] = phases
    trial["active_phase"] = "supervised_marker_training"

    def train() -> tuple[phx.meshing.LearnedMeshProposer, np.ndarray, float, int]:
        weights, _, rank, _ = np.linalg.lstsq(features, teacher, rcond=None)
        model = LinearMarker(jax.device_put(np.asarray(weights, dtype=np.float64)))
        proposer = LearnedMeshProposer(
            model, kind="marking", proposer_id="supervised-native-design-Jensen-marker"
        )
        fit_error = float(np.max(np.abs(features @ weights - teacher)))
        return proposer, weights, fit_error, int(rank)

    training, timing = _measured(train)
    proposer, weights, fit_error, rank = training
    phases["supervised_marker_training"] = timing["wall_seconds"]
    trial["training"] = {
        "algorithm": "deterministic-host-least-squares",
        "model_type": "LinearMarker",
        "teacher": "analytic simplex Jensen defect of ambient x squared",
        "teacher_digest": canonical_fingerprint(teacher),
        "training_source_result_id": source.result_id,
        "feature_id": bound.feature_id,
        "scope_id": scope.scope_id,
        "feature_ids": list(bound.feature_ids),
        "weights": weights.tolist(),
        "model_id": proposer.model_id,
        "fit_rank": rank,
        "maximum_training_error": fit_error,
        "selection_policy": "native safety projects highest positive scores, at most ceil(cells/3)",
        "evaluation_scope": "same bounded source; teacher imitation, not generalization evidence",
    }
    trial["active_phase"] = "learned_marker_evaluation"
    raw, timing = _measured(lambda: proposer.propose(source, bound))
    phases["learned_marker_evaluation"] = timing["wall_seconds"]
    trial["raw_proposal_id"] = raw.proposal_id
    trial["raw_proposer_evaluation_id"] = raw.proposer_id
    trial["raw_scores"] = np.asarray(raw.values).tolist()
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
) -> tuple[
    phx.meshing.MeshAdaptationResult,
    phx.meshing.PreparedMeshAdaptation | phx.meshing.MeshProposalTransaction,
    dict[str, float],
    phx.meshing.LearnedMeshProposer | None,
    phx.meshing.MeshProposalFeatures | None,
]:
    """Execute each preregistered proposal through its trusted native owner."""

    if mode == "learned":
        return _design_qualification_learned(source, limits, trial)
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
        )
    )
    trial["prepared_id"] = prepared.prepared_id
    phases["adaptation_preparation"] = timing["wall_seconds"]
    trial["active_phase"] = "native_event"
    adaptation, timing = _measured(lambda: phx.meshing.execute_mesh_adaptation(prepared))
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
        independent, solved = _solve_corpus_problem(changed, jnp.zeros_like(values))
        jax.block_until_ready((independent, solved))
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
    if (
        bool(refused.committed)
        or refused.state is not accepted
        or refused.mesh is not source.mesh
        or (refused.receipt is None or refused.receipt.published)
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
            "receipt_id": refused.receipt.receipt_id,
            "receipt_published": refused.receipt.published,
            "source_state_identically_retained": True,
            "source_mesh_identically_retained": True,
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
        peak_memory_bytes=round(timing["peak_resident_bytes"]),
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
        peak_memory_bytes=peak_resident_bytes(resource.RUSAGE_SELF),
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
    preparation: dict[str, float]
    condition: float
    condition_stage: dict[str, float]
    error: dict[str, float]
    error_stage: dict[str, float]


@final
class _DesignTrialStaging(phx.StrictModule):
    executor: phx.solver.FiniteElementTopologyTransaction
    reanalysis: _DesignQualificationReanalysis
    rebind: phx.lifecycle.CompositionRebind
    reanalyses: list[_DesignQualificationReanalysisRecord]
    staged: phx.solver.FiniteElementTopologyResult
    receipt_id: str = eqx.field(static=True)
    actual: phx.meshing.PhysicalErrorEvidence
    transfer_stage: dict[str, float]
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

    def condition_number() -> float:
        matrix = np.asarray(
            _design_qualification_condition_matrix(
                problem, jax.device_put(np.eye(seed.size, dtype=np.float64))
            )
        )
        return float(np.linalg.cond(matrix))

    condition, condition_stage = _measured(condition_number)
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
        peak_memory_bytes=peak_resident_bytes(resource.RUSAGE_SELF),
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
) -> _DesignQualificationTrial:
    """Measure a real native event and all-field transaction before selection."""

    if perf_counter() >= deadline:
        raise qualification_failure(
            "_design_qualification_trial",
            ("The campaign wall budget expired before this preregistered trial."),
            {"deadline": deadline, "perf_counter": perf_counter},
        )
    adaptation, prepared, native_phases, learned_proposer, learned_features = (
        _design_qualification_native_event(
            source,
            mode,
            limits,
            trial,
        )
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


def _design_qualification_transported(
    transfer: FiniteElementTopologyTransfer,
    pde: Callable[[jax.Array], jax.Array],
    radius: jax.Array,
) -> jax.Array:
    return _design_qualification_array(transfer.apply(pde(radius)))


@eqx.filter_jit
def _design_qualification_derivative_values(
    function: Callable[[jax.Array], jax.Array],
    parameter: jax.Array,
    direction: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    step: float,
) -> tuple[jax.Array, jax.Array]:
    # Callable PyTrees expose changing geometry, PDE and transfer arrays to one
    # stable executable; only the prepared route and numerical signature specialize.
    linearization = phx.linalg.prepare_linearization(function, parameter)
    value = _design_qualification_array(linearization.primal)
    tangent = _design_qualification_array(linearization.jvp(direction))
    cotangent = jnp.linspace(-0.3, 0.8, value.size, dtype=jnp.float64).reshape(
        value.shape
    )
    pullback = _design_qualification_array(linearization.vjp(cotangent))
    duality = jnp.abs(jnp.vdot(tangent, cotangent) - direction * pullback)
    central = (upper - lower) / (2.0 * step)
    discrepancy = jnp.max(jnp.abs(tangent - central))
    return discrepancy, duality


def _design_qualification_derivative_probe(
    function: Callable[[jax.Array], jax.Array],
    parameter: jax.Array,
    direction: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    step: float,
) -> dict[str, float]:
    """Check one fixed-route derivative against its independently solved perturbations."""

    discrepancy_array, duality_array = _design_qualification_derivative_values(
        function, parameter, direction, lower, upper, step
    )
    discrepancy, duality = float(discrepancy_array), float(duality_array)
    if not np.isfinite(discrepancy + duality) or discrepancy > 2.0e-6 or duality > 2.0e-8:
        raise qualification_failure(
            "_design_qualification_derivative_probe",
            (
                f"Native fixed-epoch derivative failed: FD={discrepancy}, duality={duality}."
            ),
            {"discrepancy": discrepancy, "duality": duality},
        )
    return {"finite_difference_error": discrepancy, "JVP_VJP_duality_error": duality}


def _design_qualification_perturbations(
    geometry: phx.geometry.CompiledGeometry,
    surface_plan: phx.geometry.ImplicitSurfacePlan,
    radius_index: int,
    parameter: jax.Array,
    direction: jax.Array,
    step: float,
    solve_coordinates: Callable[
        [jax.Array], tuple[jax.Array, phx.linalg.LinearSolveResult]
    ],
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Retain both admitted geometry/PDE states for the common FD oracle."""

    perturbations: list[tuple[jax.Array, jax.Array]] = []
    for changed in (parameter - step * direction, parameter + step * direction):
        realization = surface_plan.realize(
            geometry.state.replace_at(radius_index, changed)
        )
        if not bool(realization.evidence.accepted):
            raise qualification_failure(
                "_design_qualification_perturbations",
                (
                    "Finite-difference perturbation crosses the implicit realization boundary."
                ),
                {"realization": realization},
            )
        values, solved = solve_coordinates(realization.vertices)
        if not bool(jnp.all(solved.successful)):
            raise qualification_failure(
                "_design_qualification_perturbations",
                ("A derivative perturbation failed the scalar PDE solve."),
                {"solved": solved},
            )
        perturbations.append((realization.vertices, values))
    return perturbations[0], perturbations[1]


@final
class _DesignDerivativeDiagnostics(phx.StrictModule):
    event_margins: tuple[tuple[str, float], ...]
    diagnostics: list[dict[str, float]]


def _design_qualification_derivative_admission(
    geometry: phx.geometry.CompiledGeometry,
    surface_plan: phx.geometry.ImplicitSurfacePlan,
    source: phx.meshing.CellMeshingResult,
    problem: phx.equations.CompiledFiniteElementProblem,
    trial: _DesignQualificationTrial | None,
    baseline: dict[str, Any] | None,
) -> _DesignDerivativeDiagnostics:
    """Require positive realization margins and source-epoch baseline evidence."""
    realization = surface_plan.realize(geometry.state)
    lattice_field = surface_plan.kernel.boundary_field(
        geometry.state, surface_plan.grid_points
    )
    margins = (
        (
            "lattice-sign",
            float(jnp.min(jnp.abs(lattice_field)))
            - surface_plan.policy.lattice_zero_tolerance,
        ),
        ("positive-face-area", float(realization.evidence.minimum_face_area)),
        ("positive-orientation", float(realization.evidence.minimum_orientation_margin)),
    )
    if not bool(realization.evidence.accepted) or any(
        value <= 0.0 for _, value in margins
    ):
        raise qualification_failure(
            "_design_qualification_derivatives",
            ("Fixed-lattice realization has no positive derivative event margins."),
            {"margins": margins, "realization": realization},
        )
    if trial is None:
        diagnostics = []
    else:
        if (
            baseline is None
            or baseline["source_result_id"] != source.result_id
            or (
                baseline["execution_ids"]
                != (surface_plan.plan_id, problem.compilation_id)
            )
        ):
            raise ValueError(
                f"_design_qualification_derivative_admission.baseline must contain measured geometry and PDE evidence for this source and execution epoch; got {baseline!r}."
            )
        diagnostics = [
            baseline["diagnostics"]["geometry"],
            baseline["diagnostics"]["pde"],
        ]
    return _DesignDerivativeDiagnostics(event_margins=margins, diagnostics=diagnostics)


def _design_qualification_derivative_evidence(
    source: phx.meshing.CellMeshingResult,
    scalar: phx.discretization.FiniteElementDiscretization,
    trial: _DesignQualificationTrial,
    ids: tuple[str, str, str],
    analysis: _DesignDerivativeDiagnostics,
    retained_evidence: list[phx.meshing.FixedEpochDerivativeEvidence] | None,
) -> dict[str, object]:
    """Qualify measured fixed routes and prove invalidation after topology acceptance."""
    margins, diagnostics = analysis.event_margins, analysis.diagnostics
    qualifications = tuple(
        DesignQualificationEvidence(
            DerivativeTier.Q1,
            scalar.prepared_id,
            source.result_id,
            execution_id,
            event_ids=tuple(name for name, _ in margins),
            event_margins=tuple(value for _, value in margins),
            gradient_error=diagnostic["finite_difference_error"],
            valid=True,
        )
        for execution_id, diagnostic in zip(ids, diagnostics, strict=True)
    )
    routes = tuple(zip(("geometry", "pde", "transfer"), ids, strict=True))
    evidence = FixedEpochDerivativeEvidence(
        source.result_id,
        "design-sphere.radius",
        geometry=qualifications[0],
        pde=qualifications[1],
        transfer=qualifications[2],
        routes=routes,
    )
    evidence.require_valid(
        source.result_id, "design-sphere.radius", routes=routes, event_margins=margins
    )
    invalidation = evidence.invalidation_issues(
        trial["adaptation"].target.result_id,
        "design-sphere.radius",
        routes=routes,
        event_margins=margins,
        topology_event_id=trial["adaptation"].result_id,
    )
    if "stopped-topology-event" not in invalidation:
        raise qualification_failure(
            "_design_qualification_derivatives",
            ("Native topology acceptance failed to invalidate fixed-epoch derivatives."),
            {"invalidation": invalidation},
        )
    if retained_evidence is not None:
        retained_evidence.append(evidence)
    return {
        "evidence_id": evidence.evidence_id,
        "event_margins": margins,
        "diagnostics": dict(
            zip(("geometry", "pde", "transfer"), diagnostics, strict=True)
        ),
        "event_invalidation": invalidation,
        "scope": "fixed discovery and donor routes only",
    }


def _design_qualification_derivatives(
    geometry: phx.geometry.CompiledGeometry,
    plan: phx.meshing.NativeMeshingPlan,
    source: phx.meshing.CellMeshingResult,
    scalar: phx.discretization.FiniteElementDiscretization,
    problem: phx.equations.CompiledFiniteElementProblem,
    trial: _DesignQualificationTrial | None,
    baseline: dict[str, Any] | None = None,
    *,
    retained_evidence: list[phx.meshing.FixedEpochDerivativeEvidence] | None = None,
) -> dict[str, object]:
    """Measure fixed-route geometry, PDE and transfer JVP/VJP contracts."""

    surface_plan = plan.prepared
    if not isinstance(surface_plan, phx.geometry.ImplicitSurfacePlan):
        raise TypeError(
            f"_design_qualification_derivatives.surface_plan must be an ImplicitSurfacePlan; got {type(surface_plan).__name__!r}."
        )
    radius_index = geometry.schema.index(
        phx.geometry.ParameterId("design-sphere", "radius")
    )
    parameter = geometry.state.values[radius_index]
    transfer = None if trial is None else trial["staged"].transfers[0]
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
    pde = eqx.Partial(_design_qualification_pde, solve_coordinates, coordinates)

    analysis = _design_qualification_derivative_admission(
        geometry, surface_plan, source, problem, trial, baseline
    )
    margins, diagnostics = analysis.event_margins, analysis.diagnostics
    direction = jnp.asarray(0.7, dtype=jnp.float64)
    step = 1.0e-5
    perturbations = _design_qualification_perturbations(
        geometry,
        surface_plan,
        radius_index,
        parameter,
        direction,
        step,
        solve_coordinates,
    )
    if trial is None:
        functions = (
            (coordinates, perturbations[0][0], perturbations[1][0]),
            (pde, perturbations[0][1], perturbations[1][1]),
        )
    else:
        if transfer is None:
            raise qualification_failure(
                "_design_qualification_derivatives",
                ("The transfer derivative requires a prepared native field transfer."),
                {"transfer": transfer},
            )
        transported = eqx.Partial(
            _design_qualification_transported, transfer.transfer, pde
        )
        functions = (
            (
                transported,
                _design_qualification_array(transfer.transfer.apply(perturbations[0][1])),
                _design_qualification_array(transfer.transfer.apply(perturbations[1][1])),
            ),
        )
    for function, lower, upper in functions:
        diagnostics.append(
            _design_qualification_derivative_probe(
                function, parameter, direction, lower, upper, step
            )
        )
    if trial is None:
        return {
            "source_result_id": source.result_id,
            "execution_ids": (surface_plan.plan_id, problem.compilation_id),
            "event_margins": margins,
            "diagnostics": dict(zip(("geometry", "pde"), diagnostics, strict=True)),
            "scope": "fixed source geometry/PDE before any native proposal",
        }
    if transfer is None:
        raise qualification_failure(
            "_design_qualification_derivatives",
            (
                "A native transfer is required for geometry, PDE and transfer derivative evidence."
            ),
            {"transfer": transfer},
        )
    ids = (surface_plan.plan_id, problem.compilation_id, transfer.transfer_id)
    return _design_qualification_derivative_evidence(
        source, scalar, trial, ids, analysis, retained_evidence
    )


def _design_qualification_candidates(
    current: phx.meshing.CellMeshingResult,
    accepted: phx.solver.FiniteElementAcceptedState,
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
    """Retain every observed trial and choose the cheaper proof for coincident native routes."""
    alternatives: dict[str, phx.meshing.SolverAwareCandidate] = {}
    trials: dict[str, _DesignQualificationTrial] = {}
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
    epoch_record["selected_candidate_id"] = selection.candidate.candidate_id
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
    decision.require_reanalysis(
        selected["reanalysis"](current.mesh, accepted.fields, None, None)
    )
    epoch_record.update(
        {
            "continued_physical": continued_physical,
            "continued_compiler": compiler,
            "continued_stages_seconds": continued_stages,
        }
    )
    return _DesignContinuedEpoch(
        accepted=accepted, current=current, scalar=scalar, physical=continued_physical
    )


def _design_qualification_retain_source(
    initial: _DesignInitialEpoch,
    selection: _DesignEpochSelection,
    continued: _DesignContinuedEpoch,
    derivative_owners: list[phx.meshing.FixedEpochDerivativeEvidence],
    retained_source_states: list[NativeDesignSourceState],
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
        NativeDesignSourceState(
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
    retained_source_states: list[NativeDesignSourceState] | None = None,
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
    native: list[NativeMeshingPhaseMeasurement],
    record: dict[str, Any],
    deadline: Deadline,
) -> _DesignSourcePublication:
    """Publish the design's native source and independently certify its realization."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    fixture, timing = _ledger.run(
        record.get("active_phase", "_design_qualification_source"),
        lambda: _measured(lambda: _design_qualification_source(args)),
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
) -> _DesignBaseline:
    """Solve and probe the initial design, then author its accepted physical fields."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    geometry, plan, current = publication.geometry, publication.plan, publication.current
    bundle, timing = _ledger.run(
        "baseline_solver_preparation",
        lambda: _measured(lambda: _design_qualification_prepare(current)),
    )
    scalar, all_fields, problem, _ = bundle
    stages["baseline_solver_preparation"] = timing["wall_seconds"]
    seed = jax.device_put(
        np.zeros((scalar.field_spaces[0].layout.size,), dtype=np.float64)
    )

    values, compiler, solver_stages = _ledger.run(
        "baseline_solver",
        lambda: _compiled_solver_stages(problem, seed, args.repeats),
    )
    values = _design_qualification_array(values)
    stages.update(solver_stages)
    record["baseline_compiler"] = compiler

    derivatives, timing = _ledger.run(
        "fixed_epoch_geometry_pde",
        lambda: _measured(
            lambda: _design_qualification_derivatives(
                geometry, plan, current, scalar, problem, None
            )
        ),
    )
    record["fixed_epoch_geometry_pde"] = derivatives
    stages["fixed_epoch_geometry_pde"] = timing["wall_seconds"]
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


def _design_qualification_record(
    args: QualificationControls,
    budget: phx.meshing.DecisionBudget,
) -> dict[str, Any]:
    """Initialize the preregistered targets, comparators, and external evidence."""

    record: dict[str, Any] = {
        "scenario": "native-design-lifecycle",
        "status": "failed",
        "runtime": _runtime_identity(),
        "decision_budget_id": budget.budget_id,
        "targets": {
            "physical_L2_error": args.target_error,
            "wall_seconds": args.timeout,
            "maximum_memory_bytes": args.maximum_memory_bytes,
            "maximum_condition": args.maximum_condition,
            "derivative_finite_difference_error": 2.0e-6,
            "derivative_duality_error": 2.0e-8,
            "derivative_step": 1.0e-5,
            "recomputed_solution_defect": 1.0e-10,
            "declared_field_error": 1.0e-10,
            "material_inventory_error": 1.0e-9,
        },
        "controls": {
            "resolution": args.resolution,
            "capacity": args.capacity,
            "adaptation_rounds": args.adaptation_rounds,
            "repeats": args.repeats,
        },
        "stages_seconds": {},
        "epochs": [],
        "failed_positive_attempts": [],
        "time_to_target_error_seconds": None,
        "mandatory_workflow_completion": "incomplete",
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
                "teacher": "same-source exact Jensen defect",
                "model": "deterministic least-squares LinearMarker",
                "marks": "trusted safety projection of positive model scores, at most ceil(cells/3)",
            },
        ],
        "physical_error_accounting": (
            "Independent total field L2 error includes algebraic and scalar refresh defects; "
            "do not add those terms again. Geometry is the declared represented surface; "
            "analytic-source fidelity is separately certified and independently probed."
        ),
        "nonclaims": [
            "No differentiation through discovery, topology or acceptance.",
            "The represented-surface PDE is not the exact-sphere diffusion PDE.",
            "No constitutive material history or learned-model generalization is claimed.",
            "No estimator superiority, full corpus or release claim.",
        ],
        "active_phase": "source_preparation",
    }
    return record


def _design_qualification_completion(
    args: QualificationControls,
    native: list[NativeMeshingPhaseMeasurement],
    record: dict[str, Any],
    started: float,
) -> None:
    """Record process resources and enforce the campaign's frozen time and memory budgets."""

    record["native_phases"] = _phase_recordings(native)
    record["end_to_end_seconds"] = perf_counter() - started
    record["peak_process_resident_bytes"] = peak_resident_bytes(resource.RUSAGE_SELF)
    record["memory_scope"] = (
        "process-lifetime peak; retained artifact bytes reported separately"
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
    if record["peak_process_resident_bytes"] > args.maximum_memory_bytes:
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
                "achieved": {
                    "peak_process_resident_bytes": record["peak_process_resident_bytes"]
                },
                "safe_refusal_is_positive": False,
            }
        )
    record["mandatory_workflow_completion"] = (
        "complete" if record["status"] == "passed" else "incomplete"
    )
    del record["active_phase"]


def _design_qualification_lifecycle(
    args: QualificationControls,
    *,
    retained_source_states: list[NativeDesignSourceState] | None = None,
) -> dict[str, object]:
    """Execute DesignState → PDE → measured trusted event → accepted continued solve."""

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
    record = _design_qualification_record(args, budget)
    _ledger = StageLedger(record, deadline=deadline)
    try:
        publication = _design_qualification_publish_source(args, native, record, deadline)
        baseline = _design_qualification_baseline(publication, args, record, deadline)
        geometry, plan, request, current = (
            publication.geometry,
            publication.plan,
            publication.request,
            publication.current,
        )
        accepted, scalar, all_fields = (
            baseline.accepted,
            baseline.scalar,
            baseline.all_fields,
        )
        _design_qualification_epochs(
            geometry,
            plan,
            request,
            current,
            accepted,
            scalar,
            all_fields,
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
