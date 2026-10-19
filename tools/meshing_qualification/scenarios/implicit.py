# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
import os
import resource
import sys
from dataclasses import (
    asdict,
)
from pathlib import (
    Path,
)
from tempfile import (
    TemporaryDirectory,
)
from time import (
    perf_counter,
)
from typing import (
    Any,
    final,
)

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import (
    array_tree_fingerprint,
    ArrayArchiveLimits,
    canonical_fingerprint,
    model_array_bytes,
)
from phydrax.discretization.fem import (
    prepare_finite_element_field_reconstruction,
)
from phydrax.geometry import (
    CommonRefinementPolicy,
    ImplicitProjectionBoundarySource,
)
from phydrax.geometry.implicit import (
    AnalyticImplicitProfile,
    discover_adaptive_implicit_surface,
)
from phydrax.lifecycle import (
    MeshingAcceptedEpoch,
    read_meshing_source_closure,
    recertify_restored_meshing_source,
    validate_meshing_accepted_epochs,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    NativeMeshingPhaseMeasurement,
    NativeVolumeSchedule,
    PreparedImplicitVolume,
)
from phydrax.meshing.qualification import (
    prepare_envelope_accepted_epoch,
    transition_envelope_state,
)
from phydrax.units import METER
from tools.meshing_qualification._controls import (
    _implicit_qualification_requirements,
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._physics import (
    _compiled_solver_stages,
    _manufactured_source,
    _manufactured_value,
    _physical_error,
)
from tools.meshing_qualification._records import (
    _phase_recordings,
    _quality_summary,
    _result_identity,
    FailureRecord,
)
from tools.meshing_qualification._runtime import (
    _measured,
    _runtime_identity,
    ColdWorkerTimeout,
    Deadline,
    peak_resident_bytes,
    qualification_failure,
    run_cold_worker,
    StageLedger,
)


def _implicit_qualification_fixture(
    args: QualificationControls,
) -> tuple[
    phx.geometry.CompiledGeometry,
    phx.meshing.NativeImplicitSource,
    phx.meshing.VolumeMeshingSpec,
    phx.meshing.NativeMeshingOptions,
    np.ndarray,
    phx.meshing.MeshQualityTarget,
]:
    """Freeze the analytic source and every hard physical and resource request."""

    M, D = phx.meshing, phx.discretization
    radius = args.implicit_radius
    domain = np.asarray(((-1.2 * radius,) * 3, (1.2 * radius,) * 3), dtype=np.float64)
    geometry = phx.geometry.Sphere(
        (0.0, 0.0, 0.0), radius, feature_id="qualification-implicit-sphere"
    ).compile(coordinate_unit=METER)
    revision = canonical_fingerprint(
        {
            "kind": "qualification-analytic-sphere",
            "radius": radius,
            "center": (0.0, 0.0, 0.0),
            "state": tuple(np.asarray(value) for value in geometry.state.values),
            "domain": domain,
            "coordinates": phx.SpatialCoordinateContract.si().spatial_id,
        }
    )
    grid = D.TensorGridPlan(
        tuple(D.UniformAxisSpec(args.resolution + 1) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(domain, dtype=jnp.float64))
    source = M.NativeImplicitSource(
        geometry, grid, "qualification-implicit-sphere", revision
    )
    scope = M.MeshingScope(
        source.source_id,
        revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "implicit-zero-set",
        np.asarray((0,), dtype=np.int64),
    )
    h = 2.0 * radius / args.resolution
    limits = capacity_meshing_limits(
        args.capacity,
        args.timeout,
        edges=6,
        faces=4,
        cells=1,
        connectivity_entries=16,
        data_bytes_limit=args.maximum_memory_bytes,
        scratch_bytes_limit=args.maximum_memory_bytes,
        work_units=256,
        cavity_cells=1,
        geometry_queries=256,
    )
    request = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(
                scope, h, maximum_size=1.5 * h, strength=M.SizeControlStrength.HARD
            ),
        ),
        size_compliance=M.SizeCompliancePolicy(relative_tolerance=0.5),
        protected_features=(
            M.ProtectedFeature(
                scope,
                M.FeatureKind.SURFACE,
                maximum_deviation=args.implicit_fidelity,
                hard=True,
            ),
        ),
        region_seeds=(
            M.RegionSeed(
                np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
                "interior",
                "uniform-material",
                M.RegionRole.USER,
            ),
        ),
        limits=limits,
    )
    options = M.NativeMeshingOptions(
        "implicit_restricted_delaunay",
        volume_schedule=NativeVolumeSchedule(
            radius_edge_bound=2.0,
            minimum_dihedral_degrees=10.0,
            refinement_rounds=args.implicit_refinement_rounds,
            improvement_passes=args.implicit_improvement_passes,
        ),
    )
    quality = M.MeshQualityTarget(minimum_angle=np.deg2rad(1.0), hard=True)
    return geometry, source, request, options, domain, quality


def _implicit_qualification_acceptance(
    result: phx.meshing.CellMeshingResult,
    original: phx.meshing.CellMeshingResult,
    geometry: phx.geometry.CompiledGeometry,
    request: phx.meshing.VolumeMeshingSpec,
    quality: phx.meshing.MeshQualityTarget,
) -> dict[str, object]:
    """Require fresh source/projection/global/domain proofs and independent volume."""

    certificate = result.certification
    if (
        certificate is None
        or not certificate.passed
        or not result.audit.passed
        or not result.compliance.passed
    ):
        raise qualification_failure(
            "_implicit_qualification_acceptance",
            ("Implicit volume lacks strict passing publication."),
            {"certificate": certificate, "result": result},
        )
    embedding, coverage, fidelity = (
        certificate.embedding,
        certificate.coverage,
        certificate.fidelity,
    )
    if (
        embedding is None
        or coverage is None
        or fidelity is None
        or embedding.status != "certified"
        or coverage.status != "certified"
        or fidelity.status != "certified"
    ):
        raise qualification_failure(
            "_implicit_qualification_acceptance",
            (
                "Implicit volume lacks source fidelity, global embedding, or declared-domain coverage."
            ),
            {"coverage": coverage, "embedding": embedding, "fidelity": fidelity},
        )
    retained = certificate.request
    if original.certification is None:
        raise qualification_failure(
            "_implicit_qualification_acceptance",
            ("The initial implicit result has no certification request."),
            {"original": original},
        )
    reference = original.certification.request
    retained.validate_source_integrity()
    query = retained.source
    if not isinstance(query, ImplicitProjectionBoundarySource) or not isinstance(
        query.profile, AnalyticImplicitProfile
    ):
        raise qualification_failure(
            "_implicit_qualification_acceptance",
            ("Implicit successor discarded the analytic projection query."),
            {
                "AnalyticImplicitProfile": AnalyticImplicitProfile,
                "ImplicitProjectionBoundarySource": ImplicitProjectionBoundarySource,
                "query": query,
            },
        )
    profile = query.profile
    profile.require_bound(geometry, result.coordinate_contract)
    original_query = reference.source
    if (
        not isinstance(original_query, ImplicitProjectionBoundarySource)
        or not isinstance(original_query.profile, AnalyticImplicitProfile)
        or profile.profile_id != original_query.profile.profile_id
        or retained.source_id != reference.source_id
        or retained.source_revision != reference.source_revision
        or retained.domain is None
        or coverage.domain_id != retained.domain_id
        or retained.fidelity_tolerance != reference.fidelity_tolerance
        or retained.mesh_id != result.mesh.mesh_id
        or retained.topology_id != result.mesh.topology_id
    ):
        raise qualification_failure(
            "_implicit_qualification_acceptance",
            (
                "Implicit successor changed source/query/domain/request identity or retained a stale binding."
            ),
            {
                "AnalyticImplicitProfile": AnalyticImplicitProfile,
                "ImplicitProjectionBoundarySource": ImplicitProjectionBoundarySource,
                "coverage": coverage,
                "original_query": original_query,
                "profile": profile,
                "reference": reference,
                "result": result,
                "retained": retained,
            },
        )
    projection = fidelity.projection_coverage
    if (
        projection is None
        or projection.status != "certified"
        or projection.fiber_crossing_classes.count("interior") != profile.component_count
    ):
        raise qualification_failure(
            "_implicit_qualification_acceptance",
            ("Source-enclosed component projection coverage is incomplete."),
            {"profile": profile, "projection": projection},
        )
    epsilon = projection.forward_upper
    if (
        not np.isfinite(epsilon)
        or epsilon > request.protected_features[0].maximum_deviation
        or epsilon >= profile.radius
    ):
        raise qualification_failure(
            "_implicit_qualification_acceptance",
            ("Implicit source fidelity exceeds the frozen hard request."),
            {"epsilon": epsilon, "profile": profile, "request": request},
        )
    mesh = result.mesh
    if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind != "tetrahedron":
        raise qualification_failure(
            "_implicit_qualification_acceptance",
            ("Implicit diffusion requires one tetrahedral body."),
            {"mesh": mesh},
        )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int64)
    corners = points[cells]
    signed = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    volume = float(np.sum(signed))
    lower = 4.0 * np.pi / 3.0 * (profile.radius - epsilon) ** 3
    upper = 4.0 * np.pi / 3.0 * (profile.radius + epsilon) ** 3
    connectivity = mesh.connectivity
    if not isinstance(connectivity, phx.discretization.TetrahedralConnectivity):
        raise TypeError(
            f"implicit.connectivity must be TetrahedralConnectivity; got {type(connectivity).__name__}"
        )
    edges = np.asarray(connectivity.edges)
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    control = request.size_controls[0]
    if (
        not isinstance(control, phx.meshing.UniformSizeControl)
        or control.maximum_size is None
    ):
        raise ValueError(
            f"implicit.size_control must be UniformSizeControl with a maximum size; got {control!r}"
        )
    maximum = control.maximum_size * (1.0 + request.size_compliance.relative_tolerance)
    if (
        np.any(signed <= 0.0)
        or not lower <= volume <= upper
        or float(np.max(lengths)) > maximum
        or result.quality.minimum_angle < quality.minimum_angle
    ):
        raise qualification_failure(
            "_implicit_qualification_acceptance",
            (
                "Implicit tetrahedra fail independent signed-volume, hard size, or hard quality acceptance."
            ),
            {
                "lengths": lengths,
                "lower": lower,
                "maximum": maximum,
                "quality": quality,
                "result": result,
                "signed": signed,
                "upper": upper,
                "volume": volume,
            },
        )
    return {
        "passed": True,
        "certificate_id": certificate.report_id,
        "embedding_id": embedding.certificate_id,
        "coverage_id": coverage.certificate_id,
        "fidelity_id": fidelity.certificate_id,
        "projection_id": projection.evidence_id,
        "profile_id": profile.profile_id,
        "source_id": retained.source_id,
        "source_revision": retained.source_revision,
        "domain_id": retained.domain_id,
        "original_approximation_domain_id": reference.domain_id,
        "target_approximation_domain_id": retained.domain_id,
        "source_request_id": retained.request_id,
        "original_source_request_id": reference.request_id,
        "source_state_id": profile.state_id,
        "source_query_preserved": True,
        "source_component_count": profile.component_count,
        "projection_fiber_classes": list(projection.fiber_crossing_classes),
        "projection_query_count": projection.query_count,
        "projection_ray_test_count": projection.ray_test_count,
        "coverage_semantics": "declared-restricted-approximation-domain",
        "source_fidelity_semantics": "original-analytic-source-degree-one-projection",
        "source_fidelity_upper": epsilon,
        "source_fidelity_tolerance": retained.fidelity_tolerance,
        "mesh_volume": volume,
        "analytic_volume": 4.0 * np.pi / 3.0 * profile.radius**3,
        "independent_volume_bounds": [lower, upper],
        "minimum_signed_cell_volume": float(np.min(signed)),
        "maximum_edge": float(np.max(lengths)),
        "boundary_edge_quantiles": {
            name: float(value)
            for name, value in zip(
                ("p50", "p95"),
                np.quantile(
                    lengths[np.asarray(connectivity.boundary_edges)], (0.5, 0.95)
                ),
                strict=True,
            )
        },
        "compliance_requested": dict(result.compliance.requested),
        "compliance_achieved": dict(result.compliance.achieved),
        "quality": _quality_summary(result.quality),
    }


def implicit_field_templates() -> tuple[
    tuple[str, phx.discretization.FiniteElementFieldSpec], ...
]:
    """Declare the scalar H1 field for implicit-domain diffusion."""
    return (
        (
            "intensive",
            phx.discretization.FiniteElementFieldSpec(
                "u", phx.discretization.lagrange_element("tetrahedron", 1)
            ),
        ),
    )


def _implicit_qualification_prepare_solver(
    result: phx.meshing.CellMeshingResult,
) -> tuple[
    phx.discretization.FiniteElementDiscretization,
    phx.equations.CompiledFiniteElementProblem,
    jax.Array,
    phx.discretization.FiniteElementFieldSpec,
]:
    """Configure the one-region FE diffusion formulation."""

    ((_, field),) = implicit_field_templates()
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    if not constraint.free_dofs.size:
        raise qualification_failure(
            "_implicit_qualification_prepare_solver",
            (
                "Implicit manufactured diffusion has no interior scalar H1 degrees of freedom."
            ),
            {"constraint": constraint},
        )
    form = phx.equations.FiniteElementForm(
        "analytic-implicit-manufactured-diffusion",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    _manufactured_source, coefficient_id="uniform-quadratic-laplacian-3D"
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint, dirichlet_values=_manufactured_value
    )
    return space, problem, constraint.free_dofs, field


def _implicit_qualification_cold_process(
    path: str,
    content_id: str,
    repeats: int,
    target_error: float,
    minimum_angle: float,
    hard_quality: bool,
) -> dict[str, Any]:
    """Reopen and reanalyze accepted source state in a fresh interpreter."""

    restored = read_meshing_source_closure(path, expected_content_id=content_id)
    history = validate_meshing_accepted_epochs(restored)
    source = restored["generation_source"]
    geometry = source.geometry
    request = restored["generation_specification"]
    quality = phx.meshing.MeshQualityTarget(
        minimum_angle=minimum_angle, hard=hard_quality
    )
    cold = history[-1].carrier
    fields = tuple(history[-1].fields[name] for name in ("u", "density", "history"))
    renewed = []
    for accepted in history:
        carrier = accepted.carrier
        certificate = carrier.certification
        if certificate is None:
            raise ValueError(
                "implicit.cold_epoch.certification must retain its source request; got None"
            )
        report = recertify_restored_meshing_source(
            certificate.request,
            carrier.mesh,
            carrier.geometry,
            carrier.audit,
            archive_limits=ArrayArchiveLimits(),
        )
        report.require_passed()
        renewed.append(
            _implicit_qualification_acceptance(
                carrier, history[0].carrier, geometry, request, quality
            )
        )
    acceptance = _implicit_qualification_acceptance(
        cold, cold, geometry, request, quality
    )
    space, problem, free, _ = _implicit_qualification_prepare_solver(cold)
    continued, compiler, timings = _compiled_solver_stages(
        problem,
        jnp.asarray(fields[0])[free],
        repeats,
    )
    error = _physical_error(cold.mesh, continued)
    density_defect = float(jnp.max(jnp.abs(fields[1] - 2.0)))
    history_defect = float(
        jnp.max(jnp.abs(fields[2] - (space.dof_maps[0].dof_coordinates[:, 0] + 0.5)))
    )
    if error > target_error or density_defect > 1.0e-12 or history_defect > 1.0e-12:
        raise qualification_failure(
            "_implicit_qualification_cold_process",
            ("Cold continued PDE or restored allfield/history oracle failed."),
            {
                "density_defect": density_defect,
                "error": error,
                "history_defect": history_defect,
                "target_error": target_error,
            },
        )
    peak = peak_resident_bytes(resource.RUSAGE_SELF)
    return {
        "source_fingerprint": array_tree_fingerprint(geometry),
        "field_fingerprint": array_tree_fingerprint(fields),
        "request_id": request.specification_id,
        "quality_id": quality.target_id,
        "restored_result": _result_identity(cold),
        "source_acceptance": acceptance,
        "prepared_field_id": space.prepared_id,
        "accepted_epoch_ids": [accepted.epoch_id for accepted in history],
        "renewed_epochs": renewed,
        "compiler": compiler,
        "continued_physical_L2_error": error,
        "density_max_defect": density_defect,
        "history_max_defect": history_defect,
        "stages_seconds": timings,
        "peak_process_resident_bytes": peak,
        "interpreter": sys.executable,
        "process_id": os.getpid(),
    }


def _implicit_qualification_cold_reanalysis(
    geometry: phx.geometry.CompiledGeometry,
    result: phx.meshing.CellMeshingResult,
    request: phx.meshing.VolumeMeshingSpec,
    quality: phx.meshing.MeshQualityTarget,
    args: QualificationControls,
    record: dict[str, Any],
    values: jax.Array,
    auxiliary: tuple[jax.Array, jax.Array],
    started: float,
    *,
    deadline: Deadline,
    generation_source: phx.meshing.NativeImplicitSource,
    generation_options: phx.meshing.NativeMeshingOptions,
    history: tuple[MeshingAcceptedEpoch, ...],
) -> None:
    """Restore scientific source, accepted carrier and every field without FE caches."""

    _ledger = StageLedger(record, deadline=deadline)

    limits = ArrayArchiveLimits()
    fields = (values, *auxiliary)
    certification = result.certification
    if certification is None:
        raise ValueError(
            "implicit.checkpoint.certification must retain its source request; got None"
        )
    records = {
        "certification_inputs": certification.request,
        "report": certification,
        "associations": result.associations,
        "generation_source": generation_source,
        "generation_specification": request,
        "generation_options": generation_options,
        "accepted_target": result,
        "accepted_data": {"accepted_epochs": history},
    }
    with TemporaryDirectory(prefix="phydrax-native-implicit-source-") as directory:
        receipt, timing = _ledger.run(
            "allfield_history_checkpoint_publication",
            lambda: _measured(
                lambda: write_meshing_source_closure(
                    Path(directory) / "implicit.checkpoint.zip", records, limits=limits
                )
            ),
        )
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]

        remaining = _ledger.run(
            "fresh_process_allfield_history_restore_and_continued_pde",
            lambda: deadline.expires_at - perf_counter(),
        )
        if remaining <= 0.0:
            raise phx.meshing.MeshingFailure(
                phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"implicit.remaining_seconds must be positive before cold reanalysis; got {remaining!r}",
                stage=record["active_phase"],
            )
        # The scenario module owns its worker command.
        cold_started = perf_counter()
        try:
            restored = run_cold_worker(
                "scenarios.implicit",
                (
                    str(receipt.path),
                    receipt.content_id,
                    str(args.repeats),
                    str(args.target_error),
                    repr(quality.minimum_angle),
                    json.dumps(quality.hard),
                ),
                deadline=deadline,
                ledger=StageLedger(
                    record,
                    deadline=deadline,
                    cold_phase="fresh_process_allfield_history_restore_and_continued_pde",
                ),
            )
            timing = {"wall_seconds": perf_counter() - cold_started}
        except ColdWorkerTimeout as failure:
            raise phx.meshing.MeshingFailure(
                phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"implicit.cold_reanalysis must finish within {remaining!r} seconds; got {type(failure).__name__}",
                stage=record["active_phase"],
            ) from failure
        # Parsed and process-validated by run_cold_worker.
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
        if (
            restored["restored_result"] != _result_identity(result)
            or restored["request_id"] != request.specification_id
            or restored["quality_id"] != quality.target_id
            or restored["source_fingerprint"] != array_tree_fingerprint(geometry)
            or restored["field_fingerprint"] != array_tree_fingerprint(fields)
            or restored["process_id"] == os.getpid()
            or restored["accepted_epoch_ids"]
            != [accepted.epoch_id for accepted in history]
        ):
            raise qualification_failure(
                "_implicit_qualification_cold_reanalysis",
                (
                    "Fresh implicit restore changed scientific source, controls or accepted fields."
                ),
                {
                    "array_tree_fingerprint": array_tree_fingerprint,
                    "fields": fields,
                    "geometry": geometry,
                    "quality": quality,
                    "request": request,
                    "restored": restored,
                    "result": result,
                },
            )
        if restored["peak_process_resident_bytes"] > args.maximum_memory_bytes:
            raise phx.meshing.MeshingFailure(
                phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"implicit.peak_process_resident_bytes must not exceed {args.maximum_memory_bytes!r}; got {restored['peak_process_resident_bytes']!r}",
                stage=record["active_phase"],
                achieved=(
                    (
                        "peak_process_resident_bytes",
                        restored["peak_process_resident_bytes"],
                    ),
                ),
            )
        record["checkpoint"] = {
            "content_id": receipt.content_id,
            "container_bytes": receipt.path.stat().st_size,
            "archive_limits": {
                "members": limits.max_members,
                "nesting": limits.max_manifest_nesting,
                "rank": limits.max_array_rank,
            },
            "source_authority_stripped": False,
            "prepared_fe_caches_serialized": False,
            "field_names": ["u", "density", "history"],
            "accepted_fields_bitwise_preserved": True,
            "fresh_interpreter": restored,
        }


@final
class _ImplicitPreparedSource(phx.StrictModule):
    """Frozen source and discovery request shared by the publication phases."""

    geometry: phx.geometry.CompiledGeometry
    source: phx.meshing.NativeImplicitSource
    request: phx.meshing.VolumeMeshingSpec
    options: phx.meshing.NativeMeshingOptions
    domain: np.ndarray
    quality: phx.meshing.MeshQualityTarget
    discovery_policy: phx.geometry.AdaptiveImplicitSurfacePolicy
    control: phx.meshing.UniformSizeControl


@final
class _ImplicitSolvedEpoch(phx.StrictModule):
    """Accepted diffusion state and its prepared owners for the next transition."""

    space: phx.discretization.FiniteElementDiscretization
    problem: phx.equations.CompiledFiniteElementProblem
    values: jax.Array
    error: float
    row: dict[str, Any]


@final
class _ImplicitTransferredState(phx.StrictModule):
    """Transferred physical and auxiliary state for the next source-aware epoch."""

    current: phx.meshing.CellMeshingResult
    carried: jax.Array
    auxiliary: tuple[jax.Array, jax.Array]


def _implicit_solve_epoch(
    geometry: phx.geometry.CompiledGeometry,
    original: phx.meshing.CellMeshingResult,
    request: phx.meshing.VolumeMeshingSpec,
    quality: phx.meshing.MeshQualityTarget,
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    epoch: int,
    current: phx.meshing.CellMeshingResult,
    carried: jax.Array | None,
    *,
    deadline: Deadline,
) -> _ImplicitSolvedEpoch:
    """Certify, prepare, solve, and independently measure one accepted diffusion epoch."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    acceptance, timing = _ledger.run(
        f"source_certification_epoch_{epoch}",
        lambda: _measured(
            lambda: _implicit_qualification_acceptance(
                current, original, geometry, request, quality
            )
        ),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]

    bundle, timing = _ledger.run(
        f"solver_preparation_epoch_{epoch}",
        lambda: _measured(lambda: _implicit_qualification_prepare_solver(current)),
    )
    space, problem, free, _ = bundle
    stages[record["active_phase"]] = timing["wall_seconds"]
    seed = (
        jnp.zeros((free.size,), dtype=jnp.float64)
        if carried is None
        else jnp.asarray(carried)[free]
    )
    row: dict[str, Any] = {
        "epoch": epoch,
        "result": _result_identity(current),
        "acceptance": acceptance,
        "prepared_field_id": space.prepared_id,
        "compiled_problem_id": problem.compilation_id,
        "seed_origin": "zero"
        if carried is None
        else "actual-transferred-prior-diffusion-solution",
        "transferred_state_L2_error": None
        if carried is None
        else _physical_error(current.mesh, carried),
    }
    record["epochs"].append(row)

    values, compiler, timings = _ledger.run(
        f"diffusion_solve_epoch_{epoch}",
        lambda: _compiled_solver_stages(problem, seed, args.repeats),
    )
    row["compiler"] = compiler
    stages.update({f"{name}_epoch_{epoch}": value for name, value in timings.items()})

    error, timing = _ledger.run(
        f"independent_physical_error_epoch_{epoch}",
        lambda: _measured(lambda: _physical_error(current.mesh, values)),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    row.update(
        {
            "physical_L2_error": error,
            "free_dofs": free.size,
            "dofs": space.dof_maps[0].global_dof_count,
            "retained_bytes": model_array_bytes(
                (current, space, problem, values, carried)
            ),
        }
    )
    if error <= args.target_error and record["time_to_target_error_seconds"] is None:
        record["time_to_target_error_seconds"] = perf_counter() - started
    record["physical_error"] = error
    return _ImplicitSolvedEpoch(
        space=space, problem=problem, values=values, error=error, row=row
    )


def _implicit_adapt_epoch(
    geometry: phx.geometry.CompiledGeometry,
    original: phx.meshing.CellMeshingResult,
    request: phx.meshing.VolumeMeshingSpec,
    quality: phx.meshing.MeshQualityTarget,
    args: QualificationControls,
    record: dict[str, Any],
    epoch: int,
    current: phx.meshing.CellMeshingResult,
    row: dict[str, Any],
    uniform_control: phx.meshing.UniformSizeControl,
    *,
    deadline: Deadline,
) -> phx.meshing.MeshAdaptationResult:
    """Prepare and accept one source-aware metric edit before transporting fields."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    control = _ledger.run(f"metric_preparation_epoch_{epoch}", lambda: uniform_control)
    h = control.target_size / (2.0 ** ((epoch + 1) / 3.0))
    vertices = current.mesh.entity_set(0)
    scope = phx.meshing.MeshingScope(
        current.mesh.mesh_id,
        current.mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    tensors = np.broadcast_to(
        np.eye(3, dtype=np.float64) / h**2, (current.mesh.coordinates.shape[0], 3, 3)
    ).copy()
    metric = phx.meshing.MeshMetricField(
        scope, tensors, minimum_size=h, maximum_size=h, maximum_anisotropy=1.0
    )
    certificate = current.certification
    if certificate is None or not isinstance(
        certificate.request.source, ImplicitProjectionBoundarySource
    ):
        raise ValueError(
            "implicit.adaptation.source must retain its analytic projection source; "
            f"got {None if certificate is None else type(certificate.request.source).__name__!r}"
        )
    profile = certificate.request.source.profile
    profile.require_bound(geometry, current.coordinate_contract)
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_3D,
        maximum_passes=args.implicit_metric_passes,
        relocation=True,
        limits=request.limits,
        association_transfer=phx.meshing.ImplicitAssociationTransfer(
            profile, maximum_queries=request.limits.maximum_geometry_queries
        ),
    )
    prepared, timing = _ledger.run(
        record.get("active_phase", "prepare_mesh_adaptation"),
        lambda: _measured(
            lambda: phx.meshing.prepare_mesh_adaptation(
                current, phx.meshing.MetricMeshAdaptation(metric), policy=policy
            )
        ),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    row["metric"] = {
        "metric_id": metric.metric_id,
        "target_size": h,
        "policy_id": policy.policy_id,
        "prepared_id": prepared.prepared_id,
    }

    adaptation, timing = _ledger.run(
        f"native_metric_execution_epoch_{epoch}",
        lambda: _measured(lambda: phx.meshing.execute_mesh_adaptation(prepared)),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    row["adaptation"] = {
        "result_id": adaptation.result_id,
        "status": adaptation.status.value,
        "target": _result_identity(adaptation.target),
        "compliance_passed": adaptation.compliance.passed,
        "retained_bytes": model_array_bytes((prepared, adaptation)),
    }
    metric_evidence = adaptation.evidence
    if isinstance(metric_evidence, phx.meshing.MetricRemeshingEvidence):
        row["adaptation"]["metric_evidence"] = {
            "passes": metric_evidence.passes,
            "splits": metric_evidence.splits,
            "collapses": metric_evidence.collapses,
            "flips": metric_evidence.flips,
            "relocations": metric_evidence.relocations,
            "out_of_range_edges": metric_evidence.out_of_range_edges,
            "minimum_metric_length": metric_evidence.minimum_metric_length,
            "maximum_metric_length": metric_evidence.maximum_metric_length,
            "minimum_metric_quality": metric_evidence.minimum_metric_quality,
            "work_units": metric_evidence.work_units,
        }
    if (
        not adaptation.status.converged
        or adaptation.transition is None
        or not adaptation.compliance.passed
    ):
        raise qualification_failure(
            "_implicit_qualification_epochs",
            (f"Positive source-aware metric cycle failed: {adaptation.status.value}."),
            {"adaptation": adaptation},
        )

    row["adaptation"]["acceptance"] = _ledger.run(
        f"target_source_acceptance_epoch_{epoch}",
        lambda: _implicit_qualification_acceptance(
            adaptation.target, original, geometry, request, quality
        ),
    )
    return adaptation


def _implicit_transfer_epoch(
    current: phx.meshing.CellMeshingResult,
    adaptation: phx.meshing.MeshAdaptationResult,
    solved: _ImplicitSolvedEpoch,
    auxiliary: tuple[jax.Array, jax.Array] | None,
    epoch: int,
    args: QualificationControls,
    record: dict[str, Any],
    *,
    deadline: Deadline,
) -> _ImplicitTransferredState:
    """Transport all fields/history, then query the prior prepared scalar owner."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    space, problem, values, row = solved.space, solved.problem, solved.values, solved.row
    transferred, timing = _ledger.run(
        f"allfield_history_composition_rebind_epoch_{epoch}",
        lambda: _measured(
            lambda: transition_envelope_state(
                current,
                adaptation,
                problem,
                values,
                epoch,
                auxiliary,
                projection_policy=CommonRefinementPolicy(
                    overlap_simplices=True,
                    maximum_candidate_pairs=256 * args.capacity,
                    maximum_accepted_pairs=256 * args.capacity,
                    maximum_memory_bytes=args.maximum_memory_bytes,
                ),
                failure_factory=qualification_failure,
                physical_error=_physical_error,
            )
        ),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    carried, auxiliary, row["transfer"] = transferred
    row["transfer"]["state_L2_error"] = _physical_error(adaptation.target.mesh, carried)

    reconstruction, timing = _ledger.run(
        f"prepared_scalar_query_epoch_{epoch}",
        lambda: _measured(
            lambda: prepare_finite_element_field_reconstruction(space, "u")
        ),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    sites = jnp.asarray(((0.0, 0.0, 0.0),), dtype=jnp.float64)
    query, timing = _ledger.run(
        record.get("active_phase", "prepare_query"),
        lambda: _measured(
            lambda: reconstruction.prepare_query(sites, coverage="complete")
        ),
    )
    stages[f"scalar_query_location_epoch_{epoch}"] = timing["wall_seconds"]
    sample, timing = _ledger.run(
        record.get("active_phase", "block_until_ready"),
        lambda: _measured(lambda: jax.block_until_ready(query.apply(values))),
    )
    stages[f"scalar_query_execution_epoch_{epoch}"] = timing["wall_seconds"]
    row["prepared_field_query"] = {
        "query_id": query.query_id,
        "values": np.asarray(sample).tolist(),
        "retained_bytes": model_array_bytes((reconstruction, query)),
        "origin": "actual-source-diffusion-state",
        "site": [0.0, 0.0, 0.0],
    }
    return _ImplicitTransferredState(
        current=adaptation.target, carried=carried, auxiliary=auxiliary
    )


def _implicit_qualification_epochs(
    geometry: phx.geometry.CompiledGeometry,
    original: phx.meshing.CellMeshingResult,
    request: phx.meshing.VolumeMeshingSpec,
    quality: phx.meshing.MeshQualityTarget,
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    *,
    deadline: Deadline,
    uniform_control: phx.meshing.UniformSizeControl,
    generation_source: phx.meshing.NativeImplicitSource,
    generation_options: phx.meshing.NativeMeshingOptions,
) -> None:
    """Carry all accepted fields/history through metric edits and continue diffusion."""

    current, carried = original, None
    auxiliary: tuple[jax.Array, jax.Array] | None = None
    accepted_epochs: list[MeshingAcceptedEpoch] = []
    transition = None
    errors: list[float] = []
    for epoch in range(args.adaptation_rounds + 1):
        solved = _implicit_solve_epoch(
            geometry,
            original,
            request,
            quality,
            args,
            record,
            started,
            epoch,
            current,
            carried,
            deadline=deadline,
        )
        errors.append(solved.error)
        if auxiliary is None:
            auxiliary = (
                jnp.full(solved.values.shape, 2.0, dtype=solved.values.dtype),
                solved.space.dof_maps[0].dof_coordinates[:, 0] + 0.5,
            )
        accepted_epochs.append(
            StageLedger(record, deadline=deadline).run(
                f"accepted_epoch_declaration_{epoch}",
                lambda: prepare_envelope_accepted_epoch(
                    current,
                    solved.space,
                    solved.values,
                    auxiliary,
                    epoch,
                    part_name="implicit",
                    transition=transition,
                ),
            )
        )
        if epoch == args.adaptation_rounds:
            break
        adaptation = _implicit_adapt_epoch(
            geometry,
            original,
            request,
            quality,
            args,
            record,
            epoch,
            current,
            solved.row,
            uniform_control,
            deadline=deadline,
        )
        transferred = _implicit_transfer_epoch(
            current,
            adaptation,
            solved,
            auxiliary,
            epoch,
            args,
            record,
            deadline=deadline,
        )
        current, carried, auxiliary = (
            transferred.current,
            transferred.carried,
            transferred.auxiliary,
        )
        transition = adaptation
    record["physical_error_convergence"] = {
        "errors": errors,
        "strictly_decreasing": all(b < a for a, b in zip(errors, errors[1:])),
        "target_met": errors[-1] <= args.target_error,
    }
    if (
        not record["physical_error_convergence"]["strictly_decreasing"]
        or errors[-1] > args.target_error
    ):
        raise qualification_failure(
            "_implicit_qualification_epochs",
            (
                "Metric/transfer/diffusion lifecycle failed physical-error convergence or target."
            ),
            {"args": args, "errors": errors, "record": record},
        )
    if auxiliary is None:
        raise qualification_failure(
            "_implicit_qualification_epochs",
            (
                "Implicit allfield/history continuation lost its accepted auxiliary fields."
            ),
            {"auxiliary": auxiliary},
        )
    _implicit_qualification_cold_reanalysis(
        geometry,
        current,
        request,
        quality,
        args,
        record,
        solved.values,
        auxiliary,
        started,
        deadline=deadline,
        generation_source=generation_source,
        generation_options=generation_options,
        history=tuple(accepted_epochs),
    )


def _implicit_prepare_source(
    args: QualificationControls,
    record: dict[str, Any],
    *,
    deadline: Deadline,
) -> _ImplicitPreparedSource:
    """Validate frozen controls and record the authored analytic source request."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    positive = (
        args.timeout,
        args.target_error,
        args.implicit_radius,
        args.implicit_fidelity,
        args.implicit_root_tolerance,
    )
    if (
        any(not np.isfinite(value) or value <= 0.0 for value in positive)
        or args.resolution < 4
        or args.capacity < 64
        or args.repeats < 1
        or args.adaptation_rounds < 1
        or args.maximum_memory_bytes < 1
        or not 3 <= args.implicit_maximum_level <= 16
        or args.implicit_edge_isolation_depth < 0
        or min(
            args.implicit_refinement_rounds,
            args.implicit_improvement_passes,
            args.implicit_metric_passes,
        )
        < 1
    ):
        raise ValueError(
            f"implicit.controls must have finite positive targets and bounded capacities; got targets={positive!r}, resolution={args.resolution!r}, capacity={args.capacity!r}, repeats={args.repeats!r}, rounds={args.adaptation_rounds!r}, memory_bytes={args.maximum_memory_bytes!r}, maximum_level={args.implicit_maximum_level!r}, edge_isolation_depth={args.implicit_edge_isolation_depth!r}, refinement_rounds={args.implicit_refinement_rounds!r}, improvement_passes={args.implicit_improvement_passes!r}, metric_passes={args.implicit_metric_passes!r}"
        )

    bundle, timing = _ledger.run(
        "analytic_source_request_preparation",
        lambda: _measured(lambda: _implicit_qualification_fixture(args)),
    )
    geometry, source, request, options, domain, quality = bundle
    stages[record["active_phase"]] = timing["wall_seconds"]
    discovery_policy = phx.geometry.AdaptiveImplicitSurfacePolicy(
        enclosure="interval",
        initial_level=2,
        minimum_surface_level=3,
        maximum_level=args.implicit_maximum_level,
        maximum_boxes=args.capacity,
        maximum_evaluations=64 * args.capacity,
        maximum_faces=args.capacity,
        maximum_intersection_pairs=32 * args.capacity,
        edge_isolation_depth=args.implicit_edge_isolation_depth,
        root_tolerance=args.implicit_root_tolerance,
        flatness_tolerance=args.implicit_fidelity,
    )
    record["source"] = {
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "radius": args.implicit_radius,
        "domain": domain.tolist(),
        "state_values": [np.asarray(value).tolist() for value in geometry.state.values],
    }
    control = request.size_controls[0]
    schedule = options.volume_schedule
    if not isinstance(control, phx.meshing.UniformSizeControl) or schedule is None:
        raise qualification_failure(
            "_implicit_qualification_lifecycle",
            (
                "The prepared implicit qualification lost its physical sizing or numerical schedule."
            ),
            {"control": control, "schedule": schedule},
        )
    record["request"] = {
        "specification_id": request.specification_id,
        "limits_id": request.limits.limits_id,
        "options_id": options.options_id,
        "target_size": control.target_size,
        "size_control_id": control.control_id,
        "size_compliance_id": request.size_compliance.policy_id,
        "quality_target_id": quality.target_id,
        "hard_quality_minimum_angle": quality.minimum_angle,
        "hard_quality_owner": "canonical-MeshQualityTarget-independent-qualification-gate",
        "volume_spec_quality_enforcement": "not-declared-by-current-VolumeMeshingSpec",
        "qualification_request_id": canonical_fingerprint(
            {
                "volume_request": request.specification_id,
                "hard_quality": quality.target_id,
            }
        ),
        "fidelity_feature_id": request.protected_features[0].feature_id,
        "discovery_policy": asdict(discovery_policy),
        "source_authority_stripped": False,
    }
    record["request"]["resource_budgets"] = {
        "vertices": args.capacity,
        "cells": args.capacity,
        "edges": 6 * args.capacity,
        "faces": 4 * args.capacity,
        "connectivity_entries": 16 * args.capacity,
        "work_units": 256 * args.capacity,
        "geometry_queries": 256 * args.capacity,
        "cavity_cells": args.capacity,
        "data_bytes": args.maximum_memory_bytes,
        "scratch_bytes": args.maximum_memory_bytes,
        "wall_seconds": args.timeout,
    }
    record["request"]["volume_schedule"] = {
        "schedule_id": schedule.schedule_id,
        "radius_edge_bound": 2.0,
        "minimum_dihedral_degrees": 10.0,
        "refinement_rounds": args.implicit_refinement_rounds,
        "improvement_passes": args.implicit_improvement_passes,
        "restricted_dual_maximum_depth": 32,
        "restricted_dual_spatial_tolerance": args.implicit_fidelity * 1.0e-3,
    }
    return _ImplicitPreparedSource(
        geometry=geometry,
        source=source,
        request=request,
        options=options,
        domain=domain,
        quality=quality,
        discovery_policy=discovery_policy,
        control=control,
    )


def _implicit_discover_source(
    args: QualificationControls,
    prepared: _ImplicitPreparedSource,
    record: dict[str, Any],
    *,
    deadline: Deadline,
) -> None:
    """Certify discovery and independent inside/outside domain probes."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    geometry, domain, source = prepared.geometry, prepared.domain, prepared.source
    discovery_policy = prepared.discovery_policy
    discovery, timing = _ledger.run(
        "certified_adaptive_discovery",
        lambda: _measured(
            lambda: discover_adaptive_implicit_surface(
                geometry,
                domain=domain,
                policy=discovery_policy,
                source_id=source.source_id,
            )
        ),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    evidence = discovery.evidence
    record["discovery"] = {
        "evidence_id": evidence.evidence_id,
        "accuracy": evidence.accuracy,
        "status": evidence.status,
        "unresolved_boxes": np.asarray(evidence.unresolved_boxes).tolist(),
        "unresolved_issues": np.asarray(evidence.unresolved_issues).tolist(),
        "domain_query_id": discovery.volume.query_id,
        "leaf_count": evidence.leaf_count,
        "box_evaluations": evidence.box_evaluations,
        "point_evaluations": evidence.point_evaluations,
        "extraction_evaluations": evidence.extraction_evaluations,
        "retained_bytes": model_array_bytes(discovery),
        "extracted_surface_used_for_fill": False,
    }
    if (
        evidence.accuracy != "certified"
        or discovery.topology is None
        or evidence.unresolved_boxes.shape[0]
    ):
        raise qualification_failure(
            "_implicit_qualification_lifecycle",
            (
                "Positive certified discovery remained uncertain; no workset or extraction substitutes for volume."
            ),
            {"discovery": discovery, "evidence": evidence},
        )

    classified, timing = _ledger.run(
        "certified_domain_queries",
        lambda: _measured(
            lambda: discovery.volume.classify_boxes(
                np.asarray(
                    ((0.0, 0.0, 0.0), (1.1 * args.implicit_radius,) * 3),
                    dtype=np.float64,
                ),
                np.asarray(
                    ((0.0, 0.0, 0.0), (1.1 * args.implicit_radius,) * 3),
                    dtype=np.float64,
                ),
            )
        ),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    classes = np.asarray(classified.classes)
    record["discovery"]["domain_probe_classes"] = classes.tolist()
    if not np.array_equal(
        classes,
        (
            int(phx.geometry.ImplicitVolumeClass.INSIDE),
            int(phx.geometry.ImplicitVolumeClass.OUTSIDE),
        ),
    ):
        raise qualification_failure(
            "_implicit_qualification_lifecycle",
            ("Source-enclosed interior/exterior domain probes were not certified."),
            {"classes": classes},
        )


def _implicit_publish_volume(
    prepared: _ImplicitPreparedSource,
    record: dict[str, Any],
    native: list[NativeMeshingPhaseMeasurement],
    *,
    deadline: Deadline,
) -> phx.meshing.CellMeshingResult:
    """Prepare and publish the restricted native volume with its retained source proof."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    source, request, options = prepared.source, prepared.request, prepared.options
    plan, timing = _ledger.run(
        "native_restricted_volume_preparation",
        lambda: _measured(
            lambda: phx.meshing.NativeMeshingProvider(options).plan(
                source,
                request,
                coordinate_contract=phx.SpatialCoordinateContract.si(),
                record_phase=native.append,
            )
        ),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    record["plan_id"] = plan.plan_id

    prepared_volume = plan.prepared
    if not isinstance(prepared_volume, PreparedImplicitVolume):
        raise TypeError(
            f"implicit.plan.prepared must be PreparedImplicitVolume; got {type(prepared_volume).__name__}"
        )

    current, timing = _ledger.run(
        "native_restricted_volume_publication",
        lambda: _measured(lambda: plan.execute(record_phase=native.append)),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    record["generation"] = _result_identity(current)
    record["retained_plan_bytes"] = model_array_bytes((source, plan))
    record["restricted_source"] = {
        "domain_query_id": prepared_volume.query.query_id,
        "profile_id": prepared_volume.profile.profile_id,
        "prepared_id": prepared_volume.prepared_id,
        "query_certified": prepared_volume.query.certified,
        "seed_cover_radius": prepared_volume.seed_cover_radius,
        "seed_kind": prepared_volume.seed_kind.value,
        "actual_fidelity_tolerance": prepared_volume.fidelity_tolerance,
        "surface_extraction_used_as_source": False,
    }
    return current


def _implicit_record_failure(failure: Exception, record: dict[str, Any]) -> None:
    """Record the failed positive attempt with native evidence when available."""
    detail = FailureRecord.from_exception(failure, record["active_phase"]).to_record()
    record.update(detail)
    record["failed_positive_attempts"].append(detail)


def _implicit_finish_record(
    args: QualificationControls,
    record: dict[str, Any],
    native: list[NativeMeshingPhaseMeasurement],
    started: float,
) -> None:
    """Finalize resource evidence and failure counts after every lifecycle exit."""

    record["native_phases"] = _phase_recordings(native)
    record["end_to_end_seconds"] = perf_counter() - started
    record["peak_process_resident_bytes"] = peak_resident_bytes(resource.RUSAGE_SELF)
    record["memory_scope"] = (
        "process-lifetime-peak; per-artifact distinct-leaf retained bytes separate"
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
    for exceeded, status, label in (
        (record["end_to_end_seconds"] > args.timeout, "timeout", "wall"),
        (
            record["peak_process_resident_bytes"] > args.maximum_memory_bytes,
            "failed",
            "memory",
        ),
    ):
        if exceeded:
            record["status"] = status
            detail = {
                **FailureRecord(
                    record["active_phase"],
                    "ResourceBudgetExceeded",
                    f"Frozen {label} budget exceeded.",
                ).to_record(),
                "failure_category": "resource_exhausted",
                "requested": record["targets"],
                "achieved": {
                    "wall_seconds": record["end_to_end_seconds"],
                    "peak_process_resident_bytes": record["peak_process_resident_bytes"],
                },
            }
            record.update(detail)
            record["failed_positive_attempts"].append(detail)
    if record["status"] != "passed":
        record["failure_phase"] = record["active_phase"]
        record["time_to_target_error_seconds"] = None
    else:
        record["mandatory_workflow_completion"] = "complete"
        record["case_contract_completion"] = "complete"
        record["independent_scientific_certification"] = "passed"
    record["counts"] = {
        "passed": int(record["status"] == "passed"),
        "failed": int(record["status"] == "failed"),
        "timeout": int(record["status"] == "timeout"),
        "uncertainty_or_refusal_is_positive": False,
    }
    del record["active_phase"]


def _implicit_qualification_lifecycle(args: QualificationControls) -> dict[str, object]:
    """Execute certified discovery → strict native volume → metric/transfer/solve."""

    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    native: list[NativeMeshingPhaseMeasurement] = []
    record: dict[str, Any] = {
        "scenario": "native-implicit-lifecycle",
        "engine": "native",
        "status": "failed",
        "runtime": _runtime_identity(),
        "expected_admission": "positive",
        "requirements": _implicit_qualification_requirements(),
        "targets": {
            "physical_L2_error": args.target_error,
            "wall_seconds": args.timeout,
            "maximum_memory_bytes": args.maximum_memory_bytes,
        },
        "epochs": [],
        "stages_seconds": {},
        "failed_positive_attempts": [],
        "physical_error": None,
        "time_to_target_error_seconds": None,
        "mandatory_workflow_completion": "incomplete",
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
        "active_phase": "frozen_request_validation",
        "manufactured_diffusion": {
            "exact": "|x|^2",
            "forcing": -6.0,
            "conductivity": 1.0,
            "norm": "independent-exact-affine-P1-quadratic-L2",
        },
    }
    try:
        prepared = _implicit_prepare_source(args, record, deadline=deadline)
        _implicit_discover_source(args, prepared, record, deadline=deadline)
        current = _implicit_publish_volume(prepared, record, native, deadline=deadline)
        _implicit_qualification_epochs(
            prepared.geometry,
            current,
            prepared.request,
            prepared.quality,
            args,
            record,
            started,
            deadline=deadline,
            uniform_control=prepared.control,
            generation_source=prepared.source,
            generation_options=prepared.options,
        )
        record["status"] = "passed"
        record["implemented_admitted_workflow"] = (
            "analytic-sphere-restricted-tetra-metric-allfield-history-rollback-diffusion-cold-reanalysis"
        )
    except (
        phx.meshing.MeshingFailure,
        RuntimeError,
        ValueError,
        TypeError,
        TimeoutError,
    ) as failure:
        _implicit_record_failure(failure, record)
    finally:
        _implicit_finish_record(args, record, native, started)
    return record


def worker(argv: tuple[str, ...]) -> dict[str, Any]:
    """Decode process arguments and run the scenario-owned cold consumer."""
    return _implicit_qualification_cold_process(
        argv[0],
        argv[1],
        int(argv[2]),
        float(argv[3]),
        float(argv[4]),
        json.loads(argv[5]),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "worker":
        raise ValueError(
            "qualification.subcommand must equal worker; got " + repr(sys.argv[1:])
        )
    print(json.dumps(worker(tuple(sys.argv[2:]))))


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return _implicit_qualification_lifecycle(args)
