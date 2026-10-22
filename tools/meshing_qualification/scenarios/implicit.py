# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import math
import os
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
    Literal,
    NamedTuple,
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
    DomainCoverageCertificate,
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
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _measured,
    _runtime_identity,
    ColdControlPacket,
    ColdWorkerReport,
    ColdWorkerTimeout,
    Deadline,
    peak_resident_bytes,
    PEAK_RESIDENT_SCOPE,
    prepare_cold_worker,
    qualification_failure,
    StageLedger,
)


class ImplicitQualificationCase(NamedTuple):
    """Declared analytic source, evidence role and resolution schedule of one profile.

    ``center`` and ``major_ratio`` are in units of ``implicit_radius`` (the minor
    radius of a ring torus). Ladder level ``k`` requests hard fidelity
    ``implicit_fidelity * 2**-k`` from a fresh native generation of the same
    certified source; the lifecycle continues from the finest level. Every
    metric epoch requests the ``z`` size ``h / metric_anisotropy``.
    """

    role: Literal["routine", "larger", "combined"]
    family: Literal["sphere", "ring-torus"]
    center: tuple[float, float, float]
    major_ratio: float
    fidelity_levels: int
    metric_anisotropy: float
    minimum_adaptation_rounds: int


IMPLICIT_QUALIFICATION_CASES: dict[str, ImplicitQualificationCase] = {
    "sphere": ImplicitQualificationCase(
        "routine", "sphere", (0.0, 0.0, 0.0), 0.0, 1, 1.0, 1
    ),
    "sphere-refinement": ImplicitQualificationCase(
        "larger", "sphere", (0.0, 0.0, 0.0), 0.0, 3, 1.0, 2
    ),
    "offset-ring-torus-anisotropic": ImplicitQualificationCase(
        "combined", "ring-torus", (0.0331, -0.0191, 0.0466), 2.0, 2, 2.0, 2
    ),
}


def _implicit_case(args: QualificationControls) -> ImplicitQualificationCase:
    """Resolve the declared profile; unknown names are refused, not defaulted."""
    case = IMPLICIT_QUALIFICATION_CASES.get(args.implicit_case)
    if case is None:
        raise ValueError(
            f"implicit.implicit_case must name a declared profile {tuple(IMPLICIT_QUALIFICATION_CASES)!r}; got {args.implicit_case!r}"
        )
    return case


def _declared_topology(case: ImplicitQualificationCase) -> tuple[int, int, int]:
    """Return the authored boundary components, Euler characteristic and genus."""
    return (1, 2, 0) if case.family == "sphere" else (1, 0, 1)


# The P1 L2 error of the smooth manufactured state decays at second order in
# the carrier size. Larger and combined profiles must observe at least first
# order between consecutive metric epochs; stagnating edits are not convergence.
IMPLICIT_MINIMUM_PHYSICAL_ORDER = 1.0


def _realized_carrier_size(acceptance: dict[str, Any]) -> float:
    """Volume-equivalent cell size (|Omega_h| / cells)^(1/3) of an accepted carrier.

    Measured from the independently summed signed tetrahedron volumes, so the
    order uses the refinement each carrier actually achieved; for anisotropic
    carriers it is the geometric mean of the directional sizes (dof-based h).
    """
    return (float(acceptance["mesh_volume"]) / int(acceptance["cells"])) ** (1.0 / 3.0)


def _physical_error_convergence(
    errors: list[float],
    sizes: list[float],
    minimum_order: float | None,
    target_error: float,
) -> dict[str, Any]:
    """Observed orders log(e_k / e_k+1) / log(h_k / h_k+1) over realized sizes.

    A pair whose carrier did not refine (h_k+1 >= h_k) or whose error is not
    positive has no observable order and fails any required minimum.
    """
    orders = [
        math.log(previous / following) / math.log(coarse / fine)
        if previous > 0.0 and following > 0.0 and coarse > fine
        else None
        for previous, following, coarse, fine in zip(
            errors, errors[1:], sizes, sizes[1:], strict=False
        )
    ]
    return {
        "errors": list(errors),
        "realized_carrier_sizes": list(sizes),
        "observed_orders": orders,
        "required_minimum_order": minimum_order,
        "strictly_decreasing": all(b < a for a, b in zip(errors, errors[1:])),
        "target_met": bool(errors) and errors[-1] <= target_error,
        "order_met": minimum_order is None
        or (
            bool(orders)
            and all(order is not None and order >= minimum_order for order in orders)
        ),
    }


def _interior_witness(
    args: QualificationControls, case: ImplicitQualificationCase
) -> np.ndarray:
    """Return the authored interior point: the ball center or the tube center."""
    radius = args.implicit_radius
    witness = radius * np.asarray(case.center, dtype=np.float64)
    if case.family == "ring-torus":
        witness[0] += case.major_ratio * radius
    return witness


def _implicit_fidelity_ladder(
    args: QualificationControls, case: ImplicitQualificationCase
) -> tuple[float, ...]:
    """Hard fidelity tolerances of the declared native generation ladder."""
    return tuple(
        args.implicit_fidelity * 2.0**-level for level in range(case.fidelity_levels)
    )


def _domain_probes(
    args: QualificationControls,
    case: ImplicitQualificationCase,
    domain: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, tuple[int, ...]]:
    """Authored volume boxes whose classes follow from the analytic source alone."""
    radius = args.implicit_radius
    center = radius * np.asarray(case.center, dtype=np.float64)
    half = 0.25 * radius
    inside = int(phx.geometry.ImplicitVolumeClass.INSIDE)
    outside = int(phx.geometry.ImplicitVolumeClass.OUTSIDE)
    rows: list[tuple[str, np.ndarray, int]] = []
    if case.family == "sphere":
        rows.append(("interior", center, inside))
    else:
        offset = np.asarray((case.major_ratio * radius, 0.0, 0.0), dtype=np.float64)
        rows.extend(
            (
                ("tube-interior", center + offset, inside),
                ("opposite-tube-interior", center - offset, inside),
                ("genus-hole", center, outside),
            )
        )
    span = domain[1] - domain[0]
    lower = [witness - half for _, witness, _ in rows]
    upper = [witness + half for _, witness, _ in rows]
    lower.append(domain[1] - 0.1 * span)
    upper.append(domain[1].copy())
    names = (*(name for name, _, _ in rows), "domain-corner")
    classes = (*(value for _, _, value in rows), outside)
    return names, np.stack(lower), np.stack(upper), classes


def _implicit_qualification_source(
    args: QualificationControls, case: ImplicitQualificationCase
) -> tuple[
    phx.geometry.CompiledGeometry,
    phx.meshing.NativeImplicitSource,
    np.ndarray,
    np.ndarray,
]:
    """Author the analytic source, its enclosing domain and an interior witness."""

    M, D = phx.meshing, phx.discretization
    radius = args.implicit_radius
    center = tuple(float(value) * radius for value in case.center)
    origin = np.asarray(center, dtype=np.float64)
    if case.family == "sphere":
        source_id = "qualification-implicit-sphere"
        extent = 1.2 * radius
        geometry = phx.geometry.Sphere(center, radius, feature_id=source_id).compile(
            coordinate_unit=METER
        )
        description: dict[str, object] = {
            "kind": "qualification-analytic-sphere",
            "radius": radius,
            "center": center,
        }
    else:
        source_id = "qualification-implicit-ring-torus"
        major = case.major_ratio * radius
        extent = 1.2 * (major + radius)
        geometry = phx.geometry.Torus(
            center, major - radius, major + radius, feature_id=source_id
        ).compile(coordinate_unit=METER)
        description = {
            "kind": "qualification-analytic-ring-torus",
            "minor_radius": radius,
            "major_radius": major,
            "center": center,
        }
    domain = np.stack((origin - extent, origin + extent))
    revision = canonical_fingerprint(
        {
            **description,
            "state": tuple(np.asarray(value) for value in geometry.state.values),
            "domain": domain,
            "coordinates": phx.SpatialCoordinateContract.si().spatial_id,
        }
    )
    grid = D.TensorGridPlan(
        tuple(D.UniformAxisSpec(args.resolution + 1) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(domain, dtype=jnp.float64))
    source = M.NativeImplicitSource(geometry, grid, source_id, revision)
    return geometry, source, domain, _interior_witness(args, case)


def _implicit_volume_request(
    args: QualificationControls,
    source: phx.meshing.NativeImplicitSource,
    witness: np.ndarray,
    fidelity: float,
) -> phx.meshing.VolumeMeshingSpec:
    """Freeze every hard physical and resource request at one fidelity level."""

    M = phx.meshing
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "implicit-zero-set",
        np.asarray((0,), dtype=np.int64),
    )
    h = 2.0 * args.implicit_radius / args.resolution
    limits = capacity_meshing_limits(
        args.capacity,
        args.timeout,
        maximum_work_units=args.maximum_work_units,
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
    return M.VolumeMeshingSpec(
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
                maximum_deviation=fidelity,
                hard=True,
            ),
        ),
        region_seeds=(
            M.RegionSeed(
                np.asarray(witness, dtype=np.float64),
                "interior",
                "uniform-material",
                M.RegionRole.USER,
            ),
        ),
        limits=limits,
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
    """Freeze the declared analytic source and its coarsest hard request."""

    M = phx.meshing
    geometry, source, domain, witness = _implicit_qualification_source(
        args, _implicit_case(args)
    )
    request = _implicit_volume_request(args, source, witness, args.implicit_fidelity)
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


def _analytic_volume(family: str, radius: float, major_radius: float) -> float:
    """Exact enclosed volume of the authored ball or ring torus.

    Ball: 4/3 pi r^3. Ring torus (Pappus): pi r^2 * 2 pi R = 2 pi^2 R r^2.
    The epsilon-offsets of either body below its reach are the same family
    with ``radius`` replaced by ``radius -/+ epsilon`` and ``R`` unchanged.
    """
    if family == "sphere":
        return 4.0 * math.pi / 3.0 * radius**3
    if family == "ring-torus":
        return 2.0 * math.pi**2 * major_radius * radius**2
    raise ValueError(
        f"implicit.analytic_volume.family must be sphere or ring-torus; got {family!r}"
    )


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
        or not isinstance(coverage, DomainCoverageCertificate)
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
    # The epsilon-offsets used for the volume bounds below are the exact
    # inner/outer parallel bodies only while epsilon stays below the reach
    # (the radius of a ball, min(r, R - r) for a ring torus).
    if (
        not np.isfinite(epsilon)
        or epsilon > request.protected_features[0].maximum_deviation
        or epsilon >= profile.reach_lower
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
    analytic = _analytic_volume(profile.family, profile.radius, profile.major_radius)
    lower = _analytic_volume(
        profile.family, profile.radius - epsilon, profile.major_radius
    )
    upper = _analytic_volume(
        profile.family, profile.radius + epsilon, profile.major_radius
    )
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
        "source_component_count": profile.component_count,
        "source_family": profile.family,
        "source_genus": profile.genus,
        "source_boundary_euler_characteristic": profile.boundary_euler_characteristic,
        "source_center": np.asarray(profile.center, dtype=np.float64).tolist(),
        "source_radius": profile.radius,
        "source_major_radius": profile.major_radius,
        "source_reach_lower": profile.reach_lower,
        "projection_fiber_classes": list(projection.fiber_crossing_classes),
        "projection_query_count": projection.query_count,
        "projection_ray_test_count": projection.ray_test_count,
        "coverage_semantics": "declared-restricted-approximation-domain",
        "source_fidelity_semantics": "original-analytic-source-degree-one-projection",
        "source_fidelity_upper": epsilon,
        "source_fidelity_tolerance": retained.fidelity_tolerance,
        "mesh_volume": volume,
        "analytic_volume": analytic,
        "independent_volume_error": abs(volume - analytic),
        "independent_volume_bounds": [lower, upper],
        "vertices": int(points.shape[0]),
        "cells": int(cells.shape[0]),
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
    path: Path,
    content_id: str,
    repeats: int,
    target_error: float,
    minimum_angle: float,
    hard_quality: bool,
) -> ColdWorkerReport:
    """Reopen and reanalyze accepted source state in a fresh interpreter."""

    started = perf_counter()
    restored = read_meshing_source_closure(path, expected_content_id=content_id)
    history = validate_meshing_accepted_epochs(restored)
    read_seconds = perf_counter() - started
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
    peak = peak_resident_bytes()
    return ColdWorkerReport(
        record={
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
        },
        read_seconds=read_seconds,
    )


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
    with (
        TemporaryDirectory(prefix="phydrax-native-implicit-source-") as directory,
        prepare_cold_worker(
            "tools.meshing_qualification.scenarios.implicit",
            deadline=deadline,
            ledger=StageLedger(
                record,
                deadline=deadline,
                cold_phase="fresh_process_allfield_history_restore_and_continued_pde",
            ),
        ) as cold_worker,
    ):
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
        cold_started = perf_counter()
        try:
            restored = cold_worker.execute(
                receipt.path,
                receipt.content_id,
                {
                    "repeats": args.repeats,
                    "target_error": args.target_error,
                    "minimum_angle": quality.minimum_angle,
                    "hard_quality": quality.hard,
                },
            )
            timing = {"wall_seconds": perf_counter() - cold_started}
        except ColdWorkerTimeout as failure:
            raise phx.meshing.MeshingFailure(
                phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED,
                f"implicit.cold_reanalysis must finish within {remaining!r} seconds; got {type(failure).__name__}",
                stage=record["active_phase"],
            ) from failure
        # Parsed and process-bound by ColdWorker.execute.
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
    """Frozen source and discovery request shared by the publication phases.

    ``ladder_requests`` holds one hard request per declared fidelity level;
    ``request`` is its finest member, which the metric/PDE lifecycle continues.
    """

    geometry: phx.geometry.CompiledGeometry
    source: phx.meshing.NativeImplicitSource
    request: phx.meshing.VolumeMeshingSpec
    ladder_requests: tuple[phx.meshing.VolumeMeshingSpec, ...]
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
    anisotropy = _implicit_case(args).metric_anisotropy
    sizes = np.asarray((h, h, h / anisotropy), dtype=np.float64)
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
        np.diag(1.0 / sizes**2), (current.mesh.coordinates.shape[0], 3, 3)
    ).copy()
    metric = phx.meshing.MeshMetricField(
        scope,
        tensors,
        minimum_size=float(np.min(sizes)),
        maximum_size=h,
        maximum_anisotropy=anisotropy,
    )
    eigenvalues = np.linalg.eigvalsh(tensors)
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
        "axis_sizes": sizes.tolist(),
        "anisotropy": anisotropy,
        # Measured on the tensors handed to adaptation: sqrt(lambda_max/lambda_min).
        "tensor_anisotropy": float(
            np.max(np.sqrt(eigenvalues[:, -1] / eigenvalues[:, 0]))
        ),
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
    target_connectivity = adaptation.target.mesh.connectivity
    if isinstance(target_connectivity, phx.discretization.TetrahedralConnectivity):
        target_points = np.asarray(adaptation.target.mesh.coordinates, dtype=np.float64)
        target_edges = np.asarray(target_connectivity.edges)
        # Mean |dx|, |dy|, |dz| of the accepted carrier's edges: the realized
        # directional resolution the anisotropic metric was meant to produce.
        row["adaptation"]["realized_axis_edge_extent"] = np.mean(
            np.abs(target_points[target_edges[:, 1]] - target_points[target_edges[:, 0]]),
            axis=0,
        ).tolist()
    metric_evidence = adaptation.evidence
    if isinstance(metric_evidence, phx.meshing.MetricRemeshingEvidence):
        row["adaptation"]["metric_evidence"] = {
            "passes": metric_evidence.passes,
            "splits": metric_evidence.splits,
            "collapses": metric_evidence.collapses,
            "flips": metric_evidence.flips,
            "relocations": metric_evidence.relocations,
            "out_of_range_edges": metric_evidence.out_of_range_edges,
            "unmet_metric_edges": metric_evidence.unmet_metric_edges,
            "source_lower_exemptions": metric_evidence.source_lower_exemptions,
            "source_lower_binding": metric_evidence.source_lower_binding,
            "maximum_fidelity_bound": metric_evidence.maximum_fidelity_bound,
            "rejected_operations": metric_evidence.rejected_operations,
            "operation_attempts": metric_evidence.operation_attempts,
            "size_issues": metric_evidence.size_issues,
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
    witness = _interior_witness(args, _implicit_case(args))
    sites = jnp.asarray(witness[None, :], dtype=jnp.float64)
    query, timing = _ledger.run(
        record.get("active_phase", "prepare_query"),
        lambda: _measured(
            lambda: reconstruction.prepare_query(
                sites,
                coverage="complete",
                maximum_location_pairs=256 * args.capacity,
            )
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
        "site": witness.tolist(),
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
    carrier_sizes: list[float] = []
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
        carrier_sizes.append(_realized_carrier_size(solved.row["acceptance"]))
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
    role = _implicit_case(args).role
    minimum_order = None if role == "routine" else IMPLICIT_MINIMUM_PHYSICAL_ORDER
    convergence = _physical_error_convergence(
        errors, carrier_sizes, minimum_order, args.target_error
    )
    # The requested metric sizes are recorded beside the realized ones; the
    # order is measured against what each accepted carrier actually achieved.
    convergence["metric_target_sizes"] = [
        uniform_control.target_size * 2.0 ** (-epoch / 3.0)
        for epoch in range(len(errors))
    ]
    record["physical_error_convergence"] = convergence
    if (
        not convergence["strictly_decreasing"]
        or not convergence["target_met"]
        or not convergence["order_met"]
    ):
        raise qualification_failure(
            "_implicit_qualification_epochs",
            (
                "Metric/transfer/diffusion lifecycle failed physical-error convergence, order or target."
            ),
            {"args": args, "convergence": convergence, "record": record},
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

    case = _implicit_case(args)

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
        or args.adaptation_rounds < max(1, case.minimum_adaptation_rounds)
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
            f"implicit.controls must have finite positive targets and bounded capacities; got case={args.implicit_case!r}, minimum_rounds={case.minimum_adaptation_rounds!r}, targets={positive!r}, resolution={args.resolution!r}, capacity={args.capacity!r}, repeats={args.repeats!r}, rounds={args.adaptation_rounds!r}, memory_bytes={args.maximum_memory_bytes!r}, maximum_level={args.implicit_maximum_level!r}, edge_isolation_depth={args.implicit_edge_isolation_depth!r}, refinement_rounds={args.implicit_refinement_rounds!r}, improvement_passes={args.implicit_improvement_passes!r}, metric_passes={args.implicit_metric_passes!r}"
        )

    bundle, timing = _ledger.run(
        "analytic_source_request_preparation",
        lambda: _measured(lambda: _implicit_qualification_fixture(args)),
    )
    geometry, source, coarse, options, domain, quality = bundle
    witness = _interior_witness(args, case)
    tolerances = _implicit_fidelity_ladder(args, case)
    ladder = (
        coarse,
        *(
            _implicit_volume_request(args, source, witness, tolerance)
            for tolerance in tolerances[1:]
        ),
    )
    request = ladder[-1]
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
    components, euler, genus = _declared_topology(case)
    record["source"] = {
        "implicit_case": args.implicit_case,
        "qualification_role": case.role,
        "family": case.family,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "radius": args.implicit_radius,
        "major_radius": case.major_ratio * args.implicit_radius,
        "center": (args.implicit_radius * np.asarray(case.center)).tolist(),
        "interior_witness": witness.tolist(),
        "declared_topology": {
            "components": components,
            "boundary_euler_characteristic": euler,
            "genus": genus,
        },
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
        "fidelity_ladder": [
            {
                "level": level,
                "hard_fidelity_tolerance": tolerance,
                "specification_id": level_request.specification_id,
            }
            for level, (tolerance, level_request) in enumerate(
                zip(tolerances, ladder, strict=True)
            )
        ],
        "metric_anisotropy": case.metric_anisotropy,
        "discovery_policy": asdict(discovery_policy),
        "source_authority_stripped": False,
    }
    record["request"]["resource_budgets"] = {
        "vertices": args.capacity,
        "cells": args.capacity,
        "edges": 6 * args.capacity,
        "faces": 4 * args.capacity,
        "connectivity_entries": 16 * args.capacity,
        "work_units": request.limits.maximum_work_units,
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
        ladder_requests=ladder,
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

    # The topology claim comes from the certified enclosure cover and its
    # extracted complex; the authored source fixes the expected invariants.
    case = _implicit_case(args)
    components, euler, genus = _declared_topology(case)
    topology, surface = discovery.topology, discovery.mesh
    record["discovery"].update(
        {
            "topology_result_id": topology.result_id,
            "topology_certified": topology.certified,
            "topology_established": topology.established,
            "topology_premise": str(topology.premise),
            "topology_unresolved_boxes": topology.unresolved_box_count,
            "surface_components": None
            if surface is None
            else surface.topology.num_face_components,
            "surface_euler_characteristic": None
            if surface is None
            else surface.topology.euler_characteristic,
            "surface_watertight": None
            if surface is None
            else surface.topology.watertight,
            "declared_components": components,
            "declared_euler_characteristic": euler,
            "declared_genus": genus,
        }
    )
    if (
        surface is None
        or not topology.certified
        or not topology.established
        or topology.unresolved_box_count
        or not surface.topology.watertight
        or surface.topology.num_face_components != components
        or surface.topology.euler_characteristic != euler
    ):
        raise qualification_failure(
            "_implicit_qualification_lifecycle",
            (
                "Certified discovery topology misses or adds source components, or changes the declared genus."
            ),
            {"discovery": discovery, "record": record["discovery"]},
        )

    names, lower, upper, expected = _domain_probes(args, case, domain)
    classified, timing = _ledger.run(
        "certified_domain_queries",
        lambda: _measured(lambda: discovery.volume.classify_boxes(lower, upper)),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    classes = np.asarray(classified.classes)
    record["discovery"].update(
        {
            "domain_probe_names": list(names),
            "domain_probe_boxes": np.stack((lower, upper), axis=1).tolist(),
            "domain_probe_classes": classes.tolist(),
            "expected_probe_classes": list(expected),
            "domain_probes_certified": bool(classified.certified),
        }
    )
    if not classified.certified or not np.array_equal(classes, expected):
        raise qualification_failure(
            "_implicit_qualification_lifecycle",
            ("Source-enclosed interior/exterior/genus volume probes were not certified."),
            {"classes": classes, "expected": expected, "names": names},
        )


def _implicit_publish_level(
    prepared: _ImplicitPreparedSource,
    request: phx.meshing.VolumeMeshingSpec,
    record: dict[str, Any],
    native: list[NativeMeshingPhaseMeasurement],
    *,
    deadline: Deadline,
    suffix: str,
) -> tuple[
    phx.meshing.CellMeshingResult, PreparedImplicitVolume, phx.meshing.NativeMeshingPlan
]:
    """Prepare and publish one restricted native volume with its retained source proof."""

    _ledger = StageLedger(record, deadline=deadline)
    stages = record["stages_seconds"]
    source, options = prepared.source, prepared.options
    plan, timing = _ledger.run(
        f"native_restricted_volume_preparation{suffix}",
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
    prepared_volume = plan.prepared
    if not isinstance(prepared_volume, PreparedImplicitVolume):
        raise TypeError(
            f"implicit.plan.prepared must be PreparedImplicitVolume; got {type(prepared_volume).__name__}"
        )
    current, timing = _ledger.run(
        f"native_restricted_volume_publication{suffix}",
        lambda: _measured(lambda: plan.execute(record_phase=native.append)),
    )
    stages[record["active_phase"]] = timing["wall_seconds"]
    return current, prepared_volume, plan


def _implicit_publish_volume(
    args: QualificationControls,
    prepared: _ImplicitPreparedSource,
    record: dict[str, Any],
    native: list[NativeMeshingPhaseMeasurement],
    *,
    deadline: Deadline,
) -> phx.meshing.CellMeshingResult:
    """Publish every declared fidelity level; the finest continues the lifecycle.

    Each level is a fresh native generation of the same certified source. Its
    profile must carry the authored component count, Euler characteristic and
    genus. Multi-level profiles require the certified projection bound and the
    independent exact-volume error to decrease strictly with the tolerance.
    """

    _ledger = StageLedger(record, deadline=deadline)
    case = _implicit_case(args)
    declared = _declared_topology(case)
    ladder = prepared.ladder_requests
    rows: list[dict[str, Any]] = []
    current: phx.meshing.CellMeshingResult | None = None
    for level, request in enumerate(ladder):
        final = level == len(ladder) - 1
        current, prepared_volume, plan = _implicit_publish_level(
            prepared,
            request,
            record,
            native,
            deadline=deadline,
            suffix="" if final else f"_level_{level}",
        )
        profile = prepared_volume.profile
        achieved = (
            profile.component_count,
            profile.boundary_euler_characteristic,
            profile.genus,
        )
        if profile.family != case.family or achieved != declared:
            raise qualification_failure(
                "_implicit_publish_volume",
                (
                    "Restricted native volume profile changed the authored family, components or genus."
                ),
                {"achieved": achieved, "declared": declared, "profile": profile},
            )
        if len(ladder) > 1:
            acceptance = _ledger.run(
                f"ladder_source_acceptance_level_{level}",
                lambda result=current, level_request=request: (
                    _implicit_qualification_acceptance(
                        result,
                        result,
                        prepared.geometry,
                        level_request,
                        prepared.quality,
                    )
                ),
            )
            rows.append(
                {
                    "level": level,
                    "hard_fidelity_tolerance": request.protected_features[
                        0
                    ].maximum_deviation,
                    "result": _result_identity(current),
                    "source_fidelity_upper": acceptance["source_fidelity_upper"],
                    "independent_volume_error": acceptance["independent_volume_error"],
                    "mesh_volume": acceptance["mesh_volume"],
                    "analytic_volume": acceptance["analytic_volume"],
                    "vertices": acceptance["vertices"],
                    "cells": acceptance["cells"],
                    "acceptance": acceptance,
                }
            )
        if final:
            record["plan_id"] = plan.plan_id
            record["generation"] = _result_identity(current)
            record["retained_plan_bytes"] = model_array_bytes((prepared.source, plan))
            record["restricted_source"] = {
                "domain_query_id": prepared_volume.query.query_id,
                "profile_id": profile.profile_id,
                "prepared_id": prepared_volume.prepared_id,
                "query_certified": prepared_volume.query.certified,
                "seed_cover_radius": prepared_volume.seed_cover_radius,
                "seed_kind": prepared_volume.seed_kind.value,
                "actual_fidelity_tolerance": prepared_volume.fidelity_tolerance,
                "profile_family": profile.family,
                "profile_component_count": profile.component_count,
                "profile_boundary_euler_characteristic": profile.boundary_euler_characteristic,
                "profile_genus": profile.genus,
                "surface_extraction_used_as_source": False,
            }
    if current is None:
        raise qualification_failure(
            "_implicit_publish_volume",
            ("The declared implicit profile has no fidelity level."),
            {"ladder": ladder},
        )
    if len(ladder) > 1:
        uppers = [row["source_fidelity_upper"] for row in rows]
        errors = [row["independent_volume_error"] for row in rows]
        tolerances = [row["hard_fidelity_tolerance"] for row in rows]
        record["source_fidelity_convergence"] = {
            "levels": rows,
            "hard_fidelity_tolerances": tolerances,
            "certified_fidelity_uppers": uppers,
            "independent_volume_errors": errors,
            "observed_volume_orders": [
                math.log(previous / following) / math.log(coarse / fine)
                if previous > 0.0 and following > 0.0
                else None
                for previous, following, coarse, fine in zip(
                    errors, errors[1:], tolerances, tolerances[1:], strict=False
                )
            ],
            "fidelity_strictly_decreasing": all(
                b < a for a, b in zip(uppers, uppers[1:])
            ),
            "volume_error_strictly_decreasing": all(
                b < a for a, b in zip(errors, errors[1:])
            ),
            "lifecycle_level": len(ladder) - 1,
        }
        convergence = record["source_fidelity_convergence"]
        if not (
            convergence["fidelity_strictly_decreasing"]
            and convergence["volume_error_strictly_decreasing"]
        ):
            raise qualification_failure(
                "_implicit_publish_volume",
                (
                    "Native generations failed certified source-fidelity or independent exact-volume convergence."
                ),
                {"convergence": convergence},
            )
    return current


def _implicit_record_failure(failure: Exception, record: dict[str, Any]) -> None:
    """Record the failed positive attempt with native evidence when available."""
    detail = record_failure(record, failure, record["active_phase"])
    record["failed_positive_attempts"].append(detail)


def _source_fidelity_holds(
    value: dict[str, Any],
    tolerance: float,
    args: QualificationControls,
    case: ImplicitQualificationCase,
) -> bool:
    """Recompute one carrier's fidelity and exact-volume bracket from its numbers.

    The certified projection bound must meet the hard tolerance and stay below
    the reach; the summed signed volume must lie between the exact volumes of
    the inner and outer epsilon-offsets of the authored body, whose radii must
    be the authored ones.
    """
    epsilon = float(value.get("source_fidelity_upper", math.inf))
    radius = float(value.get("source_radius", math.nan))
    major = float(value.get("source_major_radius", math.nan))
    family = value.get("source_family")
    if (
        family != case.family
        or not math.isclose(radius, args.implicit_radius, rel_tol=1.0e-12)
        or not math.isclose(
            major, case.major_ratio * args.implicit_radius, rel_tol=1.0e-12
        )
        or not math.isfinite(epsilon)
        or epsilon > tolerance
        or epsilon >= float(value.get("source_reach_lower", 0.0))
    ):
        return False
    volume = float(value.get("mesh_volume", math.nan))
    return float(value.get("minimum_signed_cell_volume", 0.0)) > 0.0 and _analytic_volume(
        family, radius - epsilon, major
    ) <= volume <= _analytic_volume(family, radius + epsilon, major)


def _completed_native_metric_epochs(epochs: list[dict[str, Any]]) -> int:
    """Count audited metric transitions whose transported state continued diffusion."""
    completed = 0
    for row, continued in zip(epochs, epochs[1:]):
        adaptation = row.get("adaptation", {})
        target = adaptation.get("target", {})
        evidence = adaptation.get("metric_evidence", {})
        transfer = row.get("transfer", {})
        field_transfers = transfer.get("field_transfers", [])
        # Raw out-of-range edges include certified source-lower exemptions.
        # Native compliance uses the unmet count; keep both observations intact.
        if (
            adaptation.get("status") == "complete"
            and adaptation.get("compliance_passed") is True
            and evidence.get("unmet_metric_edges") == 0
            and target.get("audit_passed") is True
            and bool(target.get("result_id"))
            and target == continued.get("result")
            and bool(adaptation.get("acceptance"))
            and bool(continued.get("acceptance"))
            and bool(transfer.get("receipt_id"))
            and transfer.get("field_names") == ["u", "density", "history"]
            and transfer.get("transport_accepted") == [True, True, True]
            and len(field_transfers) == 3
            and all(value.get("passed") is True for value in field_transfers)
            and continued.get("seed_origin")
            == "actual-transferred-prior-diffusion-solution"
            and all(
                isinstance(continued.get(key), (int, float))
                and math.isfinite(continued[key])
                for key in ("physical_L2_error", "transferred_state_L2_error")
            )
        ):
            completed += 1
    return completed


def _implicit_qualification_obligations(
    args: QualificationControls, record: dict[str, Any]
) -> dict[str, bool]:
    """Recompute every declared-role obligation from the recorded observations.

    Expected classes, tolerances, exact volumes, orders and topology come from
    the authored case; observed values come from library evidence and
    independent measurements, never from flags this scenario set itself.
    """

    case = _implicit_case(args)
    discovery = record.get("discovery", {})
    restricted = record.get("restricted_source", {})
    epochs = record.get("epochs", [])
    adapted = [row["adaptation"] for row in epochs if "adaptation" in row]
    metrics = [row.get("metric", {}) for row in epochs if "adaptation" in row]
    acceptances = [row["acceptance"] for row in epochs if "acceptance" in row] + [
        value["acceptance"] for value in adapted if "acceptance" in value
    ]
    checkpoint = record.get("checkpoint")
    fresh = checkpoint.get("fresh_interpreter") if isinstance(checkpoint, dict) else None
    components, euler, genus = _declared_topology(case)
    ladder_tolerances = _implicit_fidelity_ladder(args, case)
    tolerance = ladder_tolerances[-1]
    outside = int(phx.geometry.ImplicitVolumeClass.OUTSIDE)

    observed_components = discovery.get("surface_components")
    observed_euler = discovery.get("surface_euler_characteristic")
    # Closed orientable surfaces: chi = 2 * components - 2 * total genus.
    observed_genus = (
        None
        if observed_components is None or observed_euler is None
        else (2 * observed_components - observed_euler) / 2
    )
    domain = record.get("source", {}).get("domain")
    probe_names, _, _, expected_classes = (
        _domain_probes(args, case, np.asarray(domain, dtype=np.float64))
        if domain is not None
        else ((), None, None, ())
    )
    probes = dict(
        zip(
            discovery.get("domain_probe_names", ()),
            discovery.get("domain_probe_classes", ()),
            strict=False,
        )
    )

    minimum_order = None if case.role == "routine" else IMPLICIT_MINIMUM_PHYSICAL_ORDER
    solved = [row for row in epochs if "physical_L2_error" in row and "acceptance" in row]
    physical = _physical_error_convergence(
        [float(row["physical_L2_error"]) for row in solved],
        [_realized_carrier_size(row["acceptance"]) for row in solved],
        minimum_order,
        args.target_error,
    )

    obligations = {
        "certified_adaptive_discovery": discovery.get("accuracy") == "certified"
        and discovery.get("unresolved_boxes") == [],
        "certified_topology_has_no_missed_components": discovery.get("topology_certified")
        is True
        and discovery.get("topology_established") is True
        and discovery.get("topology_unresolved_boxes") == 0
        and discovery.get("surface_watertight") is True
        and observed_components == components
        and observed_euler == euler
        and observed_genus == genus,
        "certified_volume_domain_probes": bool(probe_names)
        and discovery.get("domain_probes_certified") is True
        and [probes.get(name) for name in probe_names] == list(expected_classes),
        "native_volume_profile_matches_source_topology": (
            restricted.get("profile_family"),
            restricted.get("profile_component_count"),
            restricted.get("profile_boundary_euler_characteristic"),
            restricted.get("profile_genus"),
        )
        == (case.family, components, euler, genus),
        "continuous_source_fidelity_every_carrier": bool(acceptances)
        and all(
            _source_fidelity_holds(value, tolerance, args, case) for value in acceptances
        ),
        "native_metric_adaptation_epochs": _completed_native_metric_epochs(epochs)
        == len(adapted)
        == max(args.adaptation_rounds, case.minimum_adaptation_rounds),
        "physical_error_convergence": len(solved) == args.adaptation_rounds + 1
        and physical["strictly_decreasing"]
        and physical["target_met"]
        and physical["order_met"],
        "cold_restored_continuation": isinstance(fresh, dict)
        and fresh.get("process_id") != os.getpid()
        and bool(epochs)
        and fresh.get("restored_result") == epochs[-1].get("result")
        and float(fresh.get("continued_physical_L2_error", math.inf)) <= args.target_error
        and float(fresh.get("density_max_defect", math.inf)) <= 1.0e-12
        and float(fresh.get("history_max_defect", math.inf)) <= 1.0e-12
        and len(fresh.get("renewed_epochs", ())) == len(epochs)
        and all(
            _source_fidelity_holds(value, tolerance, args, case)
            for value in fresh.get("renewed_epochs", ())
        ),
    }
    if case.fidelity_levels > 1:
        rows = record.get("source_fidelity_convergence", {}).get("levels", [])
        uppers = [float(row["acceptance"]["source_fidelity_upper"]) for row in rows]
        volume_errors = [
            abs(
                float(row["acceptance"]["mesh_volume"])
                - _analytic_volume(
                    case.family,
                    args.implicit_radius,
                    case.major_ratio * args.implicit_radius,
                )
            )
            for row in rows
        ]
        obligations["source_fidelity_convergence"] = (
            len(rows) == case.fidelity_levels
            and all(
                _source_fidelity_holds(row["acceptance"], level_tolerance, args, case)
                for row, level_tolerance in zip(rows, ladder_tolerances, strict=False)
            )
            and all(b < a for a, b in zip(uppers, uppers[1:]))
            and all(b < a for a, b in zip(volume_errors, volume_errors[1:]))
        )
    if case.role == "combined":
        authored_center = args.implicit_radius * np.asarray(case.center, dtype=np.float64)
        obligations["nonzero_genus_certified"] = (
            observed_genus is not None
            and observed_genus >= 1
            and probes.get("genus-hole") == outside
        )
        obligations["anisotropic_metric_executed"] = (
            bool(adapted)
            and len(metrics) == len(adapted)
            and all(
                float(metric.get("tensor_anisotropy", 1.0)) > 1.0
                and math.isclose(
                    float(metric.get("tensor_anisotropy", 1.0)),
                    case.metric_anisotropy,
                    rel_tol=1.0e-12,
                )
                for metric in metrics
            )
            and all(
                len(value.get("realized_axis_edge_extent", ())) == 3
                and value["realized_axis_edge_extent"][2]
                < min(value["realized_axis_edge_extent"][:2])
                for value in adapted
            )
        )
        obligations["off_center_source"] = bool(acceptances) and all(
            np.any(np.asarray(value.get("source_center", (0.0,) * 3)) != 0.0)
            and np.allclose(
                value.get("source_center", (math.nan,) * 3),
                authored_center,
                rtol=1.0e-12,
                atol=0.0,
            )
            for value in acceptances
        )
    return obligations


def _implicit_finish_record(
    args: QualificationControls,
    record: dict[str, Any],
    native: list[NativeMeshingPhaseMeasurement],
    started: float,
) -> None:
    """Finalize resource evidence and failure counts after every lifecycle exit."""

    record["native_phases"] = _phase_recordings(native)
    record["end_to_end_seconds"] = perf_counter() - started
    peak = peak_resident_bytes()
    record["peak_process_resident_bytes"] = peak
    record["memory_scope"] = (
        f"{PEAK_RESIDENT_SCOPE}; per-artifact distinct-leaf retained bytes separate"
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
    record["memory_budget_status"] = (
        "unassessed"
        if peak is None
        else "exceeded"
        if peak > args.maximum_memory_bytes
        else "within"
    )
    if peak is None:
        detail = {
            **FailureRecord(
                "process_peak_memory_budget",
                "ResourceBudgetUnassessed",
                f"Frozen memory budget unassessed: {PEAK_RESIDENT_SCOPE}.",
            ).to_record(),
            "requested": {"maximum_memory_bytes": args.maximum_memory_bytes},
            "achieved": {"peak_process_resident_bytes": None},
            "safe_refusal_is_positive": False,
        }
        if record["status"] == "passed":
            record["status"] = "failed"
            record.update(detail)
        record["failed_positive_attempts"].append(detail)
    for exceeded, status, label in (
        (record["end_to_end_seconds"] > args.timeout, "timeout", "wall"),
        (
            peak is not None and peak > args.maximum_memory_bytes,
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
    if record["status"] == "passed":
        obligations = _implicit_qualification_obligations(args, record)
        record["qualification_obligations"] = obligations
        failed = sorted(name for name, passed in obligations.items() if not passed)
        if failed:
            detail = {
                **FailureRecord(
                    "qualification_obligations",
                    "QualificationObligationUnmet",
                    f"Declared {record['qualification_role']} obligations unmet: {failed!r}.",
                ).to_record(),
                "safe_refusal_is_positive": False,
            }
            record["status"] = "failed"
            record.update(detail)
            record["failed_positive_attempts"].append(detail)
    if record["status"] != "passed":
        if record.get("failure_phase") is None:
            record["failure_phase"] = record["active_phase"]
        record["time_to_target_error_seconds"] = None
    else:
        # Every declared obligation of this profile's role holds; the
        # campaign separately requires each registered role of the family.
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
    declared = IMPLICIT_QUALIFICATION_CASES.get(args.implicit_case)
    record: dict[str, Any] = {
        "scenario": "native-implicit-lifecycle",
        "implicit_case": args.implicit_case,
        "qualification_role": None if declared is None else declared.role,
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
        current = _implicit_publish_volume(
            args, prepared, record, native, deadline=deadline
        )
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
            f"analytic-{_implicit_case(args).family}-certified-discovery-"
            "restricted-tetra-fidelity-ladder-metric-allfield-history-rollback-"
            "diffusion-cold-reanalysis"
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


def cold_worker(packet: ColdControlPacket) -> ColdWorkerReport:
    """Run the scenario-owned cold consumer on its one control packet."""
    controls = packet.controls
    return _implicit_qualification_cold_process(
        packet.archive_path,
        packet.content_id,
        int(controls["repeats"]),
        float(controls["target_error"]),
        float(controls["minimum_angle"]),
        bool(controls["hard_quality"]),
    )


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return _implicit_qualification_lifecycle(args)
