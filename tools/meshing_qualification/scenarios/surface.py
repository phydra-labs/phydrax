# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, final, TypedDict

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from examples.native_surface_meshing import sphere_error, zero_mean_constraint
from phydrax import array_tree_fingerprint, ArrayArchiveLimits, model_array_bytes
from phydrax.geometry import PreparedSurfaceSourceSupport
from phydrax.lifecycle import (
    CompositionRebind,
    MeshingAcceptedEpoch,
    read_meshing_source_closure,
    read_meshing_source_execution_controls,
    recertify_restored_meshing_source,
    validate_meshing_accepted_epochs,
    write_meshing_source_closure,
)
from phydrax.lifecycle._meshing_field_records import (
    rebind_meshing_adaptation_execution_endpoints,
)
from phydrax.meshing import (
    NativeMeshingPhaseMeasurement,
    qualification as originals,
    SurfaceAssociationTransfer,
)
from tools._meshing_cases import prepare_sphere_case
from tools.meshing_qualification._controls import (
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._physics import (
    _compiled_solver_stages,
    CorpusCompilerEvidence,
)
from tools.meshing_qualification._records import (
    _phase_recordings,
    _result_identity,
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _measured,
    _runtime_identity,
    ColdControlPacket,
    ColdWorker,
    ColdWorkerReport,
    ColdWorkerTimeout,
    Deadline,
    peak_resident_bytes,
    prepare_cold_worker,
    qualification_failure,
    StageLedger,
)
from tools.meshing_qualification.scenarios.native_lifecycle import (
    _corpus_native_consumer,
    _corpus_native_request,
)


def _surface_forcing(points: Any, context: Any) -> Any:
    del context
    return 2.0 * points[..., 0] / jnp.linalg.norm(points, axis=-1)


def _prepare_surface_solver(
    result: phx.meshing.CellMeshingResult,
    *,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> tuple[Any, Any, Any]:

    space = phx.discretization.FiniteElementPlan(
        result.mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
        coordinate_spec=result.geometry,
        budget=budget,
    ).prepare(budget=budget)
    constraint, weights = zero_mean_constraint(space)
    form = phx.equations.FiniteElementForm(
        "sphere-laplace-beltrami",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    _surface_forcing, coefficient_id="unit-sphere-degree-one-harmonic"
                ),
            ),
        ),
        # Unit diffusion is SPD on the connected mesh's positive-mass mean-zero subspace.
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint, budget=budget
    )
    return space, problem, weights


class _SurfaceLaw(TypedDict):
    name: str
    area: float
    curvature: float
    closed: bool
    euler: int


_COLD_SEED_ORIGIN = "actual-cold-restored-accepted-u-free"
_ROUTINE_SURFACE_RESOLUTION = 4
_COMBINED_SURFACE_CASE = "capped_cylinder_features"
# Authored sizes belong to the routine resolution; larger campaign resolutions
# refine them proportionally, like the sphere's 1.6/resolution.
_SURFACE_FACTORIES: dict[str, tuple[Callable[[], Any], float, float]] = {
    "torus": (originals.torus_surface, 0.45, 1.0 / 0.6),
    "capped_cylinder": (originals.capped_cylinder_surface, 0.3, 1.0 / 0.8),
    "annulus": (originals.annulus_surface, 0.2, 1.0 / 0.4),
    "cylinder_sheet": (originals.cylinder_sheet_surface, 0.25, 1.0),
    "folded_plates": (originals.folded_plates_surface, 0.3, 0.0),
}
# The frozen can fixture's defaults: rim circles of this radius at z=0 and z=height.
_CAN_RADIUS, _CAN_HEIGHT = 0.8, 1.5


def _combined_feature_request(resolution: int) -> dict[str, float]:
    """Preregistered combined-profile request values at one resolution."""

    size = _SURFACE_FACTORIES["capped_cylinder"][1] * _ROUTINE_SURFACE_RESOLUTION / resolution
    return {
        "target_size": size,
        "maximum_size": 1.5 * size,
        "relative_tolerance": 0.5,
        "maximum_deviation": 0.25 * size * size,
    }


def _surface_qualification_prepare(
    args: QualificationControls,
) -> tuple[Any, Any, Any, _SurfaceLaw]:

    name = args.surface_case
    if not isinstance(name, str):
        raise TypeError(
            f"qualification.surface_case must be a registered profile name string; got {type(name).__name__}"
        )
    if name == "sphere":
        source, request, options = prepare_sphere_case(
            args.resolution, args.capacity, args.timeout,
            maximum_work_units=args.maximum_work_units,
            maximum_geometry_queries=args.maximum_geometry_queries,
            maximum_scratch_bytes=args.maximum_scratch_bytes,
        )
        law = _SurfaceLaw(
            name=name, area=float(4.0 * np.pi), curvature=1.0, closed=True, euler=2
        )
        return source, request, options, law
    # The combined profile reuses the frozen can source and its PDE under hard
    # size, size-compliance, surface-deviation and rim-curve-deviation controls.
    combined = name == _COMBINED_SURFACE_CASE
    physics = "capped_cylinder" if combined else name
    factory, authored_size, curvature = _SURFACE_FACTORIES[physics]
    size = authored_size * _ROUTINE_SURFACE_RESOLUTION / args.resolution
    original = factory()
    domain = original.domain
    source = phx.meshing.NativeSurfaceSource(domain)
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.scope_indices(2), dtype=np.int64),
    )
    limits = capacity_meshing_limits(
        args.capacity,
        args.timeout,
        maximum_work_units=args.maximum_work_units,
        maximum_geometry_queries=args.maximum_geometry_queries,
        scratch_bytes_limit=args.maximum_scratch_bytes,
        cells=8,
        edges=8,
        faces=8,
        connectivity_entries=64,
        data_bytes=1024,
        work_units=32,
        cavity_cells=1,
        geometry_queries=64,
        scratch_bytes=4096,
    )
    options = phx.meshing.NativeMeshingOptions("parametric_surface")
    target = phx.meshing.CellMeshingTarget(
        2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
    )
    if combined:
        curves = phx.meshing.MeshingScope(
            source.source_id,
            source.source_revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            1,
            domain.entity_set_id(1),
            np.asarray(domain.scope_indices(1), dtype=np.int64),
        )
        controls = _combined_feature_request(args.resolution)
        deviation = controls["maximum_deviation"]
        request = phx.meshing.SurfaceMeshingSpec(
            target,
            scope,
            size_controls=(
                phx.meshing.UniformSizeControl(
                    scope,
                    size,
                    maximum_size=controls["maximum_size"],
                    strength=phx.meshing.SizeControlStrength.HARD,
                ),
            ),
            size_compliance=phx.meshing.SizeCompliancePolicy(
                relative_tolerance=controls["relative_tolerance"],
                target_statistics=("p50",),
            ),
            protected_features=(
                phx.meshing.ProtectedFeature(
                    scope, phx.meshing.FeatureKind.SURFACE, maximum_deviation=deviation
                ),
                phx.meshing.ProtectedFeature(
                    curves, phx.meshing.FeatureKind.CURVE, maximum_deviation=deviation
                ),
            ),
            limits=limits,
        )
    else:
        request = phx.meshing.SurfaceMeshingSpec(
            target,
            scope,
            size_controls=(
                phx.meshing.UniformSizeControl(
                    scope, size, strength=phx.meshing.SizeControlStrength.SOFT
                ),
            ),
            limits=limits,
        )
    euler = original.euler_characteristic
    if euler is None:
        raise ValueError(
            f"surface.euler_characteristic must provide an Euler reference; got {euler!r}"
        )
    law = _SurfaceLaw(
        name=physics,
        area=original.area,
        curvature=curvature,
        closed=original.closed,
        euler=euler,
    )
    return source, request, options, law


def _surface_qualification_value(points: Any, name: str) -> Any:
    if name == "sphere":
        return points[..., 0] / jnp.linalg.norm(points, axis=-1)
    if name in ("torus", "cylinder_sheet"):
        return points[..., 2]
    if name == "capped_cylinder":
        z = points[..., 2]
        return z * z * (1.5 - z) ** 2
    if name == "annulus":
        return points[..., 0]
    if name == "folded_plates":
        return points[..., 1]
    raise ValueError(
        f"surface.profile must be sphere, torus, cylinder_sheet, capped_cylinder, annulus, or folded_plates; got {name!r}"
    )


def _surface_qualification_solver(
    result: Any, law: _SurfaceLaw, *,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> tuple[Any, Any, np.ndarray, Any]:
    name = str(law["name"])
    if name == "sphere":
        space, problem, weights = _prepare_surface_solver(result, budget=budget)
        free = np.flatnonzero(np.arange(weights.size) != int(np.argmax(weights)))
        return space, problem, free, weights
    space = phx.discretization.FiniteElementPlan(
        result.mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
        coordinate_spec=result.geometry,
        budget=budget,
    ).prepare(budget=budget)

    def exact(points: Any) -> Any:
        return _surface_qualification_value(points, name)

    def forcing(points: Any, context: Any) -> Any:
        del context
        if name == "torus":
            rho = jnp.linalg.norm(points[..., :2], axis=-1)
            z = points[..., 2]
            return z * (1.0 + (2.0 * rho - 2.0) / (0.6**2 * rho))
        if name == "capped_cylinder":
            z = points[..., 2]
            second = 2.0 * 1.5**2 - 12.0 * 1.5 * z + 12.0 * z * z
            # Caps carry u=0. The lateral quartic has zero conormal derivative
            # at each authored rim, so no unrepresented interface load is used.
            return jnp.where((z == 0.0) | (z == 1.5), 0.0, exact(points) - second)
        return jnp.zeros(points.shape[:-1], dtype=points.dtype)

    closed = bool(law["closed"])
    actions = [phx.equations.DiffusionAction("u", 1.0)]
    if closed:
        actions.append(phx.equations.MassAction("u", 1.0))
    actions.append(
        phx.equations.SourceAction(
            "u",
            phx.equations.coefficient(
                forcing, coefficient_id=f"original-{name}-manufactured-source"
            ),
        )
    )
    form = phx.equations.FiniteElementForm(
        f"original-{name}-physical-surface", "u", tuple(actions)
    )
    if closed:
        problem = phx.equations.compile_finite_element_problem(form, space, budget=budget)
        free = np.arange(space.field_spaces[0].layout.size)
    else:
        constraint = phx.discretization.dirichlet_constraint(space, "u")
        problem = phx.equations.compile_finite_element_problem(
            form, space, constraint=constraint, dirichlet_values=exact, budget=budget
        )
        free = np.asarray(constraint.free_dofs)
    return space, problem, free, None


def _surface_qualification_error(space: Any, values: Any, law: _SurfaceLaw) -> float:
    if law["name"] == "sphere":
        return sphere_error(space, values)
    rule = phx.integration.ReferenceTriangleRule(phx.integration.GaussLegendreRule(5))
    data = phx.integration.reference_rule_data(rule)
    total = 0.0
    for index, _ in enumerate(space.mesh.blocks):
        geometry = space.evaluate_block_geometry(
            "u", index, space.default_runtime.coordinates, data.points, data.weights
        )
        routes = np.asarray(space.dof_maps[0].cell_dofs[index])
        approximate = np.einsum(
            "ql,cl->cq", np.asarray(geometry.basis_values), np.asarray(values)[routes]
        )
        exact = np.asarray(
            _surface_qualification_value(geometry.physical_points, str(law["name"]))
        )
        total += float(
            np.sum(np.asarray(geometry.physical_weights) * (approximate - exact) ** 2)
        )
    return float(np.sqrt(total))


def _surface_qualification_content(space: Any, name: str, values: Any) -> float:
    index = next(
        i for i, field in enumerate(space.construction_plan.fields) if field.name == name
    )
    total = 0.0
    for block, geometry in enumerate(
        space.evaluate_geometry(name, space.default_runtime.coordinates)
    ):
        routes = np.asarray(space.dof_maps[index].cell_dofs[block])
        sampled = np.einsum(
            "ql,cl->cq", np.asarray(geometry.basis_values), np.asarray(values)[routes]
        )
        total += float(np.sum(sampled * np.asarray(geometry.physical_weights)))
    return total


_SURFACE_ORIGINALS: dict[str, Callable[[], originals.AnalyticSurface]] = {
    "sphere": originals.sphere_surface,
    "torus": originals.torus_surface,
    "capped_cylinder": originals.capped_cylinder_surface,
    "annulus": originals.annulus_surface,
    "cylinder_sheet": originals.cylinder_sheet_surface,
    "folded_plates": originals.folded_plates_surface,
}


@cache
def _surface_qualification_distance(
    name: str,
) -> Callable[[np.ndarray], np.ndarray]:
    """Closed-form point-to-source distance, independent of meshing and certificates."""
    return _SURFACE_ORIGINALS[name]().distance


def _surface_qualification_measure(
    space: Any, coordinates: np.ndarray, name: str
) -> tuple[float, float]:
    """Physical area and the largest analytic source distance of the actual map.

    The distance is sampled at every mesh vertex and every physical quadrature
    point of the published coordinate map, so a map leaving its source
    surface cannot hide between vertices.
    """
    distance = _surface_qualification_distance(name)
    area = 0.0
    deviation = float(np.max(distance(coordinates)))
    for geometry in space.evaluate_geometry("u", space.default_runtime.coordinates):
        area += float(np.sum(np.asarray(geometry.physical_weights)))
        points = np.asarray(geometry.physical_points, dtype=np.float64).reshape((-1, 3))
        deviation = max(deviation, float(np.max(distance(points))))
    return area, deviation


def _surface_qualification_vector_norm(space: Any, name: str, values: Any) -> float:
    data = phx.integration.reference_rule_data(
        phx.integration.ReferenceTriangleRule(phx.integration.GaussLegendreRule(5))
    )
    total = 0.0
    for index, block in enumerate(space.mesh.blocks):
        geometry = space.evaluate_block_geometry(
            name, index, space.default_runtime.coordinates, data.points, data.weights
        )
        physical = np.asarray(space.reconstruct(name, values, block.name, data.points))
        total += float(
            np.sum(
                np.asarray(geometry.physical_weights)
                * np.sum(physical * physical, axis=-1)
            )
        )
    return float(np.sqrt(total))


def _surface_qualification_curl_seed(points: Any, context: Any) -> Any:
    del context
    return jnp.stack(
        (
            1.0 + 0.2 * points[..., 0],
            -0.4 + 0.1 * points[..., 1],
            0.3 + 0.15 * points[..., 2],
        ),
        axis=-1,
    )


def _surface_qualification_flux_seed(points: Any, context: Any) -> Any:
    del context
    return jnp.stack(
        (
            0.7 + 0.1 * points[..., 1],
            0.8 + 0.2 * points[..., 2],
            -0.2 + 0.05 * points[..., 0],
        ),
        axis=-1,
    )


def _surface_qualification_initial_fields(
    result: Any, scalar: Any, *,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> tuple[Any, ...]:
    space = phx.meshing.qualification.prepare_surface_fields(result, budget=budget)
    return (
        scalar,
        jnp.full((space.field_spaces[1].layout.size,), 2.0, dtype=scalar.dtype),
        space.dof_maps[2].dof_coordinates[:, 0] + 0.5,
        space.project("curl", _surface_qualification_curl_seed),
        space.project("flux", _surface_qualification_flux_seed),
    )


def _surface_qualification_material_initial(
    result: Any,
    *,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> phx.equations.MaterialTransaction:
    """Initialize nonzero cell history on the accepted physical FV owner."""
    owner = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        result.mesh,
        field_name="surface-history",
        component_names=("surface-history",),
        budget=budget,
    ).prepare(cell_geometry=result.geometry, budget=budget)
    history = (1.0 + owner.cell_centers[:, 0] ** 2)[:, None]
    return phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                phx.equations.MaterialSiteId("surface-history"),
                "surface-history",
                history,
            ),
        )
    )


def _surface_qualification_material_values(
    remap: Any, source_history: Any,
) -> tuple[Any, dict[str, object]]:
    """Apply the real FV operator and check the carried bank's physical inventory."""
    plan = remap.plan
    expected = (len(plan.source_volumes), 1)
    if source_history.shape != expected:
        raise ValueError(
            f"surface.material_history must have cell-by-one Kelvin shape {expected}; got {source_history.shape}"
        )
    target_history = plan.apply(source_history)
    defect = np.asarray(plan.conservation_defect(source_history, target_history))
    source_terms = np.asarray(plan.source_volumes)[:, None] * np.asarray(source_history)
    target_terms = np.asarray(plan.target_volumes)[:, None] * np.asarray(target_history)
    coverage_error = float(
        np.asarray(plan.report.total_source_coverage_error_bound)
    ) * np.max(np.abs(np.asarray(source_history)), axis=0)
    publication_roundoff = (
        64.0
        * np.finfo(np.asarray(target_history).dtype).eps
        * (np.sum(np.abs(source_terms), axis=0) + np.sum(np.abs(target_terms), axis=0))
    )
    allowance = coverage_error + publication_roundoff
    if not np.all(np.isfinite(defect)) or np.any(np.abs(defect) > allowance):
        raise qualification_failure(
            "_surface_qualification_material_transport",
            "Surface material remap exceeds declared physical coverage and publication roundoff.",
            {"allowance": allowance, "defect": defect},
        )
    return target_history, {
        "source_physical_inventory": np.sum(source_terms, axis=0).tolist(),
        "target_physical_inventory": np.sum(target_terms, axis=0).tolist(),
        "conservation_defect": defect.tolist(),
        "source_coverage_error_bound": coverage_error.tolist(),
        "publication_roundoff_bound": publication_roundoff.tolist(),
        "passed": True,
    }


def _surface_qualification_material_transport(
    sourceResult: Any,
    adaptation: Any,
    materials: phx.equations.MaterialTransaction,
    limits: Any,
    *,
    source_workspace: Any,
    scope: Any,
    budget: phx.discretization.CoordinateEnclosureBudget,
    prepared: tuple[Any, Any, Any] | None = None,
) -> tuple[phx.solver.MaterialTopologyTransferResult, dict[str, object], tuple[Any, Any, Any]]:
    """Carry committed Kelvin history through the native chart remap."""

    if prepared is None:
        source = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
            sourceResult.mesh,
            field_name="surface-history",
            component_names=("surface-history",),
            budget=budget,
        ).prepare(cell_geometry=sourceResult.geometry, budget=budget)
        target = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
            adaptation.target.mesh,
            field_name="surface-history",
            component_names=("surface-history",),
            budget=budget,
        ).prepare(cell_geometry=adaptation.target.geometry, budget=budget)
        source_workspace.retain_owner((source, target))
        remap = phx.discretization.prepare_unstructured_conservative_remap(
            source,
            target,
            provenance="surface-history",
            policy=phx.geometry.CommonRefinementPolicy(
                maximum_exact_work=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            ),
            source_geometry=sourceResult.geometry,
            target_geometry=adaptation.target.geometry,
            surface_chart_deformation=adaptation.geometry_transition.chart_deformation,
            adaptation=adaptation,
            source_workspace=source_workspace,
            scope=scope,
            budget=budget,
        )
    else:
        source, target, remap = prepared
    source_workspace.retain_owner(
        (source, target, remap, materials, materials.state("surface-history").committed)
    )
    evidence = remap.evidence
    if (
        not remap.succeeded
        or remap.plan is None
        or not isinstance(
            evidence, phx.discretization.finite_volume.MappedSurfaceChartRemapEvidence
        )
        or not evidence.passed
    ):
        raise qualification_failure(
            "_surface_qualification_material_transport",
            (f"Surface material remap refused publication: {remap.reason}"),
            {"evidence": evidence, "remap": remap},
        )
    state = materials.state("surface-history")
    target_history, committed_evidence = _surface_qualification_material_values(
        remap, state.committed
    )
    if state.trial is state.committed:
        target_trial, trial_evidence = target_history, committed_evidence
    else:
        target_trial, trial_evidence = _surface_qualification_material_values(
            remap, state.trial
        )
    carried = phx.equations.MaterialTransaction(
        (
            phx.equations.MaterialState(
                state.site_id,
                state.model_id,
                target_history,
                trial=target_trial,
                state_version=state.state_version + 1,
            ),
        )
    )
    return phx.solver.MaterialTopologyTransferResult(carried, (remap,)), {
        "site": "surface-history",
        "value_units": ["Kelvin"],
        "source_result_id": sourceResult.result_id,
        "source_owner_id": source.prepared_id,
        "target_owner_id": target.prepared_id,
        "source_material_state_id": state.state_id,
        "target_material_state_id": carried.state("surface-history").state_id,
        "remap_id": remap.remap_id,
        "plan_id": remap.plan.plan_id,
        "evidence_id": array_tree_fingerprint(evidence),
        "chart_deformation_id": array_tree_fingerprint(
            adaptation.geometry_transition.chart_deformation
        ),
        **committed_evidence,
        "trial_transport": trial_evidence,
    }, (source, target, remap)


def _can_rim_distance(points: np.ndarray, /) -> np.ndarray:
    """Closed-form distance to the union of the frozen can's two rim circles."""

    radial = np.hypot(points[:, 0], points[:, 1]) - _CAN_RADIUS
    return np.minimum(
        np.hypot(radial, points[:, 2]), np.hypot(radial, points[:, 2] - _CAN_HEIGHT)
    )


def _requested_feature_values(request: Any) -> dict[str, float]:
    """The combined request's size and protected-deviation values."""

    control, policy = request.size_controls[0], request.size_compliance
    deviations = {float(feature.maximum_deviation) for feature in request.protected_features}
    if policy is None or control.maximum_size is None or len(deviations) != 1:
        raise qualification_failure(
            "_requested_feature_values",
            "Combined surface request lacks its size policy, maximum, or one deviation bound.",
            {"request": request},
        )
    return {
        "target_size": float(control.target_size),
        "maximum_size": float(control.maximum_size),
        "relative_tolerance": float(policy.relative_tolerance),
        "maximum_deviation": deviations.pop(),
    }


_FEATURE_FLOATS = (
    "target_size",
    "maximum_size",
    "relative_tolerance",
    "absolute_tolerance",
    "surface_deviation_request",
    "curve_deviation_request",
    "measured_maximum_edge",
    "measured_p50_edge",
    "sampled_surface_deviation",
    "sampled_rim_curve_deviation",
    "certified_surface_deviation",
    "certified_curve_deviation",
    "distance_roundoff",
)


def _surface_feature_measurements(
    result: Any,
    charts: Any,
    request: Any,
    law: _SurfaceLaw,
    sampled_surface: float,
    roundoff: float,
    *,
    combined: bool,
) -> dict[str, object] | None:
    """Measure the combined request's size and feature controls on the published mesh.

    Edge lengths come from the affine triangles' coordinates. Rim edges are
    the mesh edges whose two incident triangles lie on different source
    patches; on an affine chord of a circle the distance to the circle peaks
    at the midpoint, which the five sampled edge fractions include. Native
    continuous deviation bounds are the certified evidence; the sampled
    distances must not exceed them.
    """

    if (
        not combined
        or not request.protected_features
        or result.compliance.specification_id != request.specification_id
    ):
        return None
    if law["name"] != "capped_cylinder":
        raise qualification_failure(
            "_surface_feature_measurements",
            "Protected rim-curve measurement is defined only on the frozen can source.",
            {"law": law},
        )
    points = np.asarray(charts.mesh.coordinates, dtype=np.float64)
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in charts.mesh.blocks]
    )
    patches = np.asarray(charts.root_patches, dtype=np.int64)
    if patches.shape != (cells.shape[0],):
        raise qualification_failure(
            "_surface_feature_measurements",
            "Surface source charts do not assign one source patch per triangle.",
            {"cells": cells.shape, "patches": patches.shape},
        )
    uses = np.sort(cells[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2)), axis=1)
    edges, inverse = np.unique(uses, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    use_patches = np.repeat(patches, 3)
    lowest = np.full((edges.shape[0],), np.iinfo(np.int64).max, dtype=np.int64)
    highest = np.full((edges.shape[0],), np.iinfo(np.int64).min, dtype=np.int64)
    np.minimum.at(lowest, inverse, use_patches)
    np.maximum.at(highest, inverse, use_patches)
    rim = edges[lowest != highest]
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    fractions = np.linspace(0.0, 1.0, 5)[None, :, None]
    along = (
        points[rim[:, 0]][:, None, :] * (1.0 - fractions)
        + points[rim[:, 1]][:, None, :] * fractions
    ).reshape((-1, 3))
    achieved = dict(result.compliance.achieved)
    requested: dict[object, float] = {}
    certified: dict[object, float] = {}
    for feature in request.protected_features:
        key = f"protected:{feature.feature_id}:maximum_deviation:continuous_interpolation_bound"
        if key not in achieved:
            raise qualification_failure(
                "_surface_feature_measurements",
                "Native compliance lacks a protected feature's continuous deviation bound.",
                {"achieved": achieved, "key": key},
            )
        requested[feature.feature_kind] = float(feature.maximum_deviation)
        certified[feature.feature_kind] = float(achieved[key])
    surface, curve = phx.meshing.FeatureKind.SURFACE, phx.meshing.FeatureKind.CURVE
    control, policy = request.size_controls[0], request.size_compliance
    return {
        "target_size": float(control.target_size),
        "maximum_size": float(control.maximum_size),
        "relative_tolerance": float(policy.relative_tolerance),
        "absolute_tolerance": float(policy.absolute_tolerance),
        "surface_deviation_request": requested[surface],
        "curve_deviation_request": requested[curve],
        "measured_edges": int(edges.shape[0]),
        "measured_maximum_edge": float(np.max(lengths)),
        "measured_p50_edge": float(np.quantile(lengths, 0.5)),
        "rim_curve_edges": int(rim.shape[0]),
        "sampled_surface_deviation": float(sampled_surface),
        "sampled_rim_curve_deviation": float(
            np.max(_can_rim_distance(along), initial=0.0)
        ),
        "certified_surface_deviation": certified[surface],
        "certified_curve_deviation": certified[curve],
        "distance_roundoff": float(roundoff),
    }


def _combined_feature_verdict(features: object, expected: dict[str, float]) -> bool:
    """Recheck measured size and feature evidence against the request values.

    The hard maximum is the declared ``1.5 h`` itself (not widened by the
    statistic tolerance); p50 must lie within the declared relative tolerance
    of ``h``; certified surface and rim-curve deviation bounds must meet
    ``0.25 h^2`` and dominate the sampled distances.
    """

    if not isinstance(features, dict):
        return False
    try:
        values = {name: float(features[name]) for name in _FEATURE_FLOATS}
        edges, rim = int(features["measured_edges"]), int(features["rim_curve_edges"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(np.isfinite(value) for value in values.values()):
        return False
    size, deviation = expected["target_size"], expected["maximum_deviation"]
    roundoff = values["distance_roundoff"]
    return bool(
        values["target_size"] == size
        and values["maximum_size"] == expected["maximum_size"]
        and values["relative_tolerance"] == expected["relative_tolerance"]
        and values["absolute_tolerance"] == 0.0
        and values["surface_deviation_request"] == deviation
        and values["curve_deviation_request"] == deviation
        and edges > 0
        and rim > 0
        and values["measured_maximum_edge"] <= expected["maximum_size"]
        and abs(values["measured_p50_edge"] - size)
        <= expected["relative_tolerance"] * size
        and 0.0 <= values["certified_surface_deviation"] <= deviation
        and 0.0 <= values["certified_curve_deviation"] <= deviation
        and values["sampled_surface_deviation"]
        <= values["certified_surface_deviation"] + roundoff
        and values["sampled_rim_curve_deviation"]
        <= values["certified_curve_deviation"] + roundoff
    )


def _surface_qualification_acceptance(
    source: Any, result: Any, request: Any, law: _SurfaceLaw, *, combined: bool,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> dict[str, object]:

    certification = result.certification
    if certification is None or certification.fidelity is None:
        raise qualification_failure(
            "_surface_qualification_acceptance",
            ("Surface publication lost its source fidelity graph."),
            {"certification": certification},
        )
    fidelity = certification.fidelity
    fidelity.binding.require(result.mesh, result.geometry)
    charts = result.surface_source
    if charts is None:
        raise qualification_failure(
            "_surface_qualification_acceptance",
            ("Surface publication lost its SCI-ordered native source chart binding."),
            {"charts": charts},
        )
    charts.require_bound(result.mesh, result.geometry, certification.request.source)
    if charts.domain.content_domain_id() != source.domain.content_domain_id():
        raise qualification_failure(
            "_surface_qualification_acceptance",
            ("Surface publication changed source coefficient/stratum/control banks."),
            {"charts": charts, "source": source},
        )
    if (fidelity.binding.source_id, fidelity.binding.source_revision) != (
        source.source_id,
        source.source_revision,
    ):
        raise qualification_failure(
            "_surface_qualification_acceptance",
            ("Surface publication source certification changed source identity."),
            {"fidelity": fidelity, "source": source},
        )
    embedding = certification.embedding
    if embedding is None:
        raise qualification_failure(
            "_surface_qualification_acceptance",
            ("Surface publication lost its global-embedding certificate."),
            {"certification": certification},
        )
    embedding.binding.require(result.mesh, result.geometry)
    coordinates = np.asarray(result.mesh.coordinates, dtype=np.float64)
    cells = np.concatenate([np.asarray(block.vertices) for block in result.mesh.blocks])
    edges, uses = np.unique(
        np.sort(
            np.concatenate((cells[:, (0, 1)], cells[:, (1, 2)], cells[:, (2, 0)])), axis=1
        ),
        axis=0,
        return_counts=True,
    )
    euler = np.unique(cells).size - edges.shape[0] + cells.shape[0]
    space = phx.meshing.qualification.prepare_surface_fields(result, budget=budget)
    area, source_distance = _surface_qualification_measure(
        space, coordinates, str(law["name"])
    )
    size = request.size_controls[0].target_size
    bound = float(law["area"]) * max(
        0.25 * (float(law["curvature"]) * size) ** 2, 1.0e-12
    )
    # The analytic distance is evaluated in binary64 at rounded physical points.
    distance_roundoff = 64.0 * np.finfo(np.float64).eps * (
        1.0 + float(np.max(np.abs(coordinates)))
    )
    # The certified bound is the fidelity evidence; the closed-form distance at
    # vertices and quadrature points can only falsify it, never replace it.
    distance_bound = min(fidelity.mesh_to_source_upper, fidelity.tolerance)
    features = _surface_feature_measurements(
        result,
        charts,
        request,
        law,
        source_distance,
        distance_roundoff,
        combined=combined,
    )
    passed = bool(
        result.audit.passed
        and result.compliance.passed
        and certification.passed
        and embedding.status == "certified"
        and fidelity.status == "certified"
        and np.isfinite(fidelity.mesh_to_source_upper)
        and 0.0 <= fidelity.mesh_to_source_upper <= fidelity.tolerance
        and fidelity.chart_coverage is not None
        and fidelity.chart_coverage.complete
        and all(item.complete for item in result.associations)
        and np.all((uses == 1) | (uses == 2))
        and (not law["closed"] or np.all(uses == 2))
        and euler == law["euler"]
        and abs(area - float(law["area"])) <= bound
        and source_distance <= distance_bound + distance_roundoff
        and (
            features is None
            or _combined_feature_verdict(features, _requested_feature_values(request))
        )
    )
    return {
        "passed": passed,
        "source_binding_id": source.binding_id,
        "original_source_chart_id": charts.source_chart_id,
        "original_source_mesh_topology_id": charts.mesh.topology_id,
        "original_source_boundary_scope_id": charts.boundary_source.source_scope_id,
        "fidelity_certificate_id": fidelity.certificate_id,
        "global_embedding_certificate_id": embedding.certificate_id,
        "global_embedding_status": embedding.status,
        "fidelity_status": fidelity.status,
        "chart_coverage_id": None
        if fidelity.chart_coverage is None
        else fidelity.chart_coverage.cover_id,
        "physical_area": area,
        "independent_source_area": law["area"],
        "area_error_bound": bound,
        "independent_analytic_source_distance": source_distance,
        "certified_mesh_to_source_upper": fidelity.mesh_to_source_upper,
        "fidelity_tolerance": fidelity.tolerance,
        "analytic_distance_roundoff": distance_roundoff,
        "euler_characteristic": int(euler),
        "feature_controls": features,
        "complete_source_associations": all(
            item.complete for item in result.associations
        ),
        "result": _result_identity(result),
    }


def _surface_qualification_transition(
    source: Any,
    adaptation: Any,
    problem: Any,
    values: Any,
    auxiliary: tuple[Any, ...],
    epoch: int,
    law: _SurfaceLaw,
    materials: Any,
    limits: Any,
    *,
    budget: phx.discretization.CoordinateEnclosureBudget,
) -> tuple[Any, tuple[Any, ...], Any, dict[str, object]]:

    full = phx.meshing.qualification.prepare_surface_fields(source, budget=budget)
    fields = full.construction_plan.fields
    accepted = phx.solver.FiniteElementAcceptedState(
        (values, *auxiliary),
        0.0,
        epoch,
        source.mesh.topology_id,
        full.prepared_id,
        problem.compilation_id,
        state_version=epoch,
        materials=materials,
    )
    source_density = _surface_qualification_content(full, "density", auxiliary[0])
    target_space = phx.meshing.qualification.prepare_surface_fields(
        adaptation.target, budget=budget
    )
    rebinds: list[CompositionRebind] = []
    material_evidence: list[dict[str, object]] = []
    material_preparations: list[
        tuple[Any, phx.discretization.CoordinateEnclosureBudget, tuple[Any, Any, Any]]
    ] = []

    def transfer_material(
        original: Any, lineage: object, context: object, *, source_workspace: Any,
        scope: Any, budget: phx.discretization.CoordinateEnclosureBudget,
    ) -> Any:
        del lineage, context
        if original.transaction_id != materials.transaction_id:
            raise ValueError(
                f"surface.material.transaction_id must equal accepted source transaction {materials.transaction_id!r}; got {original.transaction_id!r}"
            )
        prepared = next(
            (
                value for prepared_scope, prepared_budget, value in material_preparations
                if prepared_scope is scope and prepared_budget is budget
            ),
            None,
        )
        transfer, evidence, prepared = _surface_qualification_material_transport(
            source, adaptation, original, limits, source_workspace=source_workspace,
            scope=scope, budget=budget, prepared=prepared,
        )
        if not material_preparations:
            budget.retain_owner(prepared)
            material_preparations.append((scope, budget, prepared))
        material_evidence.append(evidence)
        return transfer

    def capture(rebind: CompositionRebind) -> CompositionRebind:
        rebinds.append(rebind)
        return rebind

    def certify(
        mesh: Any,
        moved: tuple[Any, ...],
        candidate_materials: object,
        lineage: object,
        context: object,
    ) -> bool:
        del lineage, context
        if mesh.topology_id != adaptation.target.mesh.topology_id:
            return False
        if (
            not isinstance(candidate_materials, phx.equations.MaterialTransaction)
            or candidate_materials.state("surface-history").state_version
            != materials.state("surface-history").state_version + 1
        ):
            return False
        content = _surface_qualification_content(target_space, "density", moved[1])
        return (
            all(bool(jnp.all(jnp.isfinite(item))) for item in moved)
            and abs(content - source_density) <= 1.0e-10 * max(1.0, abs(source_density))
            and _surface_qualification_vector_norm(target_space, "curl", moved[3]) > 0.0
            and _surface_qualification_vector_norm(target_space, "flux", moved[4]) > 0.0
        )

    def impossible_density(
        mesh: Any,
        moved: tuple[Any, ...],
        materials: object,
        lineage: object,
        context: object,
    ) -> bool:
        del mesh, materials, lineage, context
        return _surface_qualification_content(target_space, "density", moved[1]) <= 0.0

    semantics = {
        field.name: role
        for role, field in phx.meshing.qualification.surface_field_templates()
    }
    rejected = phx.solver.FiniteElementTopologyTransaction(
        impossible_density,
        fields=fields,
        surface_chart_semantics=semantics,
        material_transfer=transfer_material,
        callback_owners=(full, target_space, materials, accepted.fields),
    ).execute(accepted, source.mesh, adaptation, budget=budget)
    if (
        bool(rejected.committed)
        or rejected.receipt is None
        or rejected.receipt.published
        or rejected.state is not accepted
        or rejected.mesh is not source.mesh
        or rejected.state.materials is not accepted.materials
        or any(
            not np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(rejected.state.fields, accepted.fields, strict=True)
        )
    ):
        raise qualification_failure(
            "_surface_qualification_transition",
            ("Rejected physical surface transaction changed its accepted source epoch."),
            {"accepted": accepted, "rejected": rejected, "source": source},
        )
    transaction = phx.solver.FiniteElementTopologyTransaction(
        certify,
        fields=fields,
        surface_chart_semantics=semantics,
        composition_rebind=capture,
        material_transfer=transfer_material,
        callback_owners=(full, target_space, materials, accepted.fields),
    ).execute(accepted, source.mesh, adaptation, budget=budget)
    if (
        not bool(transaction.committed)
        or transaction.receipt is None
        or not transaction.receipt.published
        or not rebinds
        or not all(item.evidence.passed for item in transaction.transfers)
    ):
        raise qualification_failure(
            "_surface_qualification_transition",
            (f"Physical surface CompositionRebind refused: {transaction.diagnostics}"),
            {"rebinds": rebinds, "transaction": transaction},
        )
    target_density = _surface_qualification_content(
        target_space, "density", transaction.state.fields[1]
    )
    scalar, _, _, _ = _surface_qualification_solver(
        adaptation.target, law, budget=budget
    )
    return (
        transaction.state.fields[0],
        transaction.state.fields[1:],
        transaction.state.materials,
        {
            "receipt_id": transaction.receipt.receipt_id,
            "source_composition_id": rebinds[-1].source.composition_id,
            "target_composition_id": transaction.receipt.composition.composition_id,
            "field_names": [field.name for field in fields],
            "surface_chart_semantics": semantics,
            "source_density_content": source_density,
            "target_density_content": target_density,
            "density_content_defect": target_density - source_density,
            "material_transport": material_evidence[-1],
            "remapped_physical_error": _surface_qualification_error(
                scalar, transaction.state.fields[0], law
            ),
            "field_transfers": [
                {
                    "transfer_id": item.transfer_id,
                    "passed": item.evidence.passed,
                    "defects": dict(item.evidence.defects),
                    "bounds": dict(item.evidence.bounds),
                }
                for item in transaction.transfers
            ],
            "rollback": {
                "passed": True,
                "requested_density_content_upper": 0.0,
                "receipt_id": rejected.receipt.receipt_id,
                "diagnostics": rejected.diagnostics,
            },
        },
    )


@final
class _SurfaceSolvedEpoch(phx.StrictModule):
    problem: phx.equations.CompiledFiniteElementProblem
    fields: tuple[Array, ...]
    compiler: CorpusCompilerEvidence
    timings: dict[str, float]
    error: float
    mean: float | None


def _surface_qualification_restart(
    cold_worker: ColdWorker,
    publications: list[dict[str, Any]],
    law: _SurfaceLaw,
    args: QualificationControls,
    record: dict[str, Any],
    evidence: dict[str, Any],
    started: float,
    deadline: Deadline,
) -> None:
    """Restart each accepted archive in a fresh process within the campaign deadline."""

    remaining = deadline.expires_at - perf_counter()
    if remaining <= 0.0:
        raise TimeoutError(
            f"surface.remaining_wall_seconds must be positive for publication; got {remaining!r}"
        )
    # The scenario module owns its cold consumer; the final epoch is its archive.
    try:
        cold = cold_worker.execute(
            Path(publications[-1]["path"]),
            publications[-1]["content_id"],
            {
                "publications": publications,
                "law": law,
                "repeats": args.repeats,
                "target_error": args.target_error,
            },
        )
    except ColdWorkerTimeout as failure:
        evidence.update(
            {
                "timeout_seconds": remaining,
                "stdout": failure.stdout,
                "stderr": failure.stderr,
            }
        )
        raise TimeoutError(
            f"surface.cold_restart must finish within {remaining!r} remaining wall seconds; got {type(failure).__name__}"
        ) from failure
    evidence.update(
        {
            "returncode": 0,
            "stdout": json.dumps(cold),
            "stderr": "",
        }
    )
    # Worker return-code evidence is retained by StageLedger.
    # Parsed and process-bound by ColdWorker.execute.
    if cold["process_id"] == os.getpid():
        raise qualification_failure(
            "_surface_qualification_checkpoint",
            ("Surface restart did not use an independent process."),
            {"cold": cold},
        )
    evidence["cold_restart"] = cold
    continuation = cold["continued_solver_epoch"]
    continuation["epoch"] = len(record["epochs"])
    record["epochs"].append(continuation)
    if (
        continuation["l2_error"] <= args.target_error
        and record["time_to_target_error_seconds"] is None
    ):
        record["time_to_target_error_seconds"] = perf_counter() - started


def _surface_qualification_checkpoint(
    source: Any,
    result: Any,
    request: Any,
    options: Any,
    law: Any,
    history: tuple[phx.meshing.qualification.SurfaceAcceptedEpoch, ...],
    adaptations: tuple[phx.meshing.MeshAdaptationResult, ...],
    checkpoint_states: tuple[phx.meshing.qualification.SurfaceCheckpointState, ...],
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    deadline: Deadline,
) -> None:
    """Publish each atomic accepted epoch with its native source closure."""

    _ledger = StageLedger(record, deadline=deadline)

    if not history or history[-1].result.result_id != result.result_id:
        raise qualification_failure(
            "_surface_qualification_checkpoint",
            (
                "Surface checkpoint requires all accepted history through the current result."
            ),
            {"history": history, "result": result},
        )
    limits = ArrayArchiveLimits()
    policy = asdict(limits)
    policy["allowed_dtype_kinds"] = sorted(limits.allowed_dtype_kinds)
    evidence = {
        "archive_limits": policy,
        "prepared_fe_caches_serialized": False,
        "archives": [],
    }
    record["checkpoint"] = evidence
    with (
        TemporaryDirectory(prefix="phydrax-native-surface-epochs-") as directory,
        prepare_cold_worker(
            "tools.meshing_qualification.scenarios.surface",
            deadline=deadline,
            ledger=StageLedger(
                record,
                deadline=deadline,
                cold_phase="surface_checkpoint_fresh_process_restart",
            ),
        ) as cold_worker,
    ):
        publications = []
        retained_epochs: list[MeshingAcceptedEpoch] = []
        for epoch, (accepted, state) in enumerate(
            zip(history, checkpoint_states, strict=True)
        ):
            carrier = accepted.result
            certification = carrier.certification
            if certification is None:
                raise qualification_failure(
                    "_surface_qualification_checkpoint",
                    "Surface checkpoint carrier.certification must retain source request evidence; got None",
                    {"carrier": carrier},
                )
            transition = (
                None if epoch == 0
                else rebind_meshing_adaptation_execution_endpoints(
                    adaptations[epoch - 1], retained_epochs[-1].carrier, carrier
                )
            )
            retained_epochs.append(
                MeshingAcceptedEpoch(
                    carrier, (state.declaration,), state.fields,
                    transition=transition,
                )
            )
            records = {
                "certification_inputs": certification.request,
                "report": certification,
                "associations": carrier.associations,
                "accepted_target": retained_epochs[-1].carrier,
                "generation_source": source,
                "generation_specification": request,
                "generation_options": options,
                "accepted_data": {"accepted_epochs": tuple(retained_epochs)},
            }
            owner, values, material_identity = (
                state.owner,
                state.values,
                state.material_identity,
            )

            receipt = _ledger.run(
                f"surface_checkpoint_epoch_{epoch}",
                lambda: write_meshing_source_closure(
                    Path(directory) / f"epoch-{epoch}.zip", records, limits=limits
                ),
            )
            physical = {
                "error": accepted.error,
                "density_content": accepted.density_content,
                "history_content": _surface_qualification_content(
                    owner, "history", values["history"]
                ),
                "curl_norm": _surface_qualification_vector_norm(
                    owner, "curl", values["curl"]
                ),
                "flux_norm": _surface_qualification_vector_norm(
                    owner, "flux", values["flux"]
                ),
            }
            publication = {
                "path": str(receipt.path),
                "content_id": receipt.content_id,
                "fingerprint": array_tree_fingerprint(records),
                "physical": physical,
                "material": material_identity,
            }
            publications.append(publication)
            evidence["archives"].append(
                {
                    "epoch": epoch,
                    "content_id": receipt.content_id,
                    "container_bytes": receipt.path.stat().st_size,
                    "fingerprint": publication["fingerprint"],
                }
            )
        _surface_qualification_restart(
            cold_worker, publications, law, args, record, evidence, started, deadline
        )


def _surface_qualification_renew_epoch(
    restored: dict[str, Any],
    accepted: MeshingAcceptedEpoch,
    publication: dict[str, Any],
    law: _SurfaceLaw,
    epoch: int,
    repeats: int,
    target_error: float,
    *,
    continue_solve: bool,
    budget: phx.discretization.CoordinateEnclosureBudget | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Re-earn geometry and continue only from the archived accepted coefficients."""
    if budget is None:
        raise ValueError("surface.cold_budget must be the actual native request ledger; got None")
    carrier = accepted.carrier
    report = recertify_restored_meshing_source(
        restored["certification_inputs"],
        carrier.mesh,
        carrier.geometry,
        carrier.audit,
        archive_limits=ArrayArchiveLimits(),
        budget=budget,
    )
    report.require_passed()
    if report.report_id != restored["report"].report_id:
        raise qualification_failure(
            "_surface_qualification_cold_worker",
            "Surface source renewal changed the accepted certification theorem.",
            {"report": report, "restored": restored},
        )
    arrays = accepted.fields
    if int(np.asarray(arrays["surface:accepted-epoch"])) != epoch:
        raise qualification_failure(
            "_surface_qualification_cold_worker",
            "Surface restored accepted epoch changed.",
            {"arrays": arrays, "epoch": epoch},
        )
    names = ("u", "density", "history", "curl", "flux")
    values = tuple(jnp.asarray(arrays[f"surface:{name}"]) for name in names)
    owner = phx.meshing.qualification.prepare_surface_fields(carrier, budget=budget)
    scalar, problem, free, weights = _surface_qualification_solver(
        carrier, law, budget=budget
    )
    physical = {
        "error": _surface_qualification_error(scalar, values[0], law),
        "density_content": _surface_qualification_content(owner, "density", values[1]),
        "history_content": _surface_qualification_content(owner, "history", values[2]),
        "curl_norm": _surface_qualification_vector_norm(owner, "curl", values[3]),
        "flux_norm": _surface_qualification_vector_norm(owner, "flux", values[4]),
    }
    if physical != publication["physical"]:
        raise qualification_failure(
            "_surface_qualification_cold_worker",
            "Surface restored physical error, inventory or mapped vector norm changed.",
            {"physical": physical, "publication": publication},
        )
    material = publication["material"]
    if material is not None:
        transaction = phx.equations.MaterialTransaction(
            (
                phx.equations.MaterialState(
                    phx.equations.MaterialSiteId("surface-history"),
                    "surface-history",
                    arrays["surface:material"],
                    trial=arrays["surface:material-trial"],
                    state_version=material["state_version"],
                ),
            )
        )
        if transaction.transaction_id != material["transaction_id"]:
            raise qualification_failure(
                "_surface_qualification_cold_worker",
                "Surface cold material primitive state changed.",
                {"material": material, "transaction": transaction},
            )
    renewed = {
        "epoch": epoch,
        "result": _result_identity(carrier),
        "report_id": report.report_id,
        "physical": physical,
    }
    if not continue_solve:
        return renewed, None
    solved, compiler, stages = _compiled_solver_stages(problem, values[0][free], repeats)
    error = _surface_qualification_error(scalar, solved, law)
    mean = None if weights is None else float(weights @ np.asarray(solved) / np.sum(weights))
    if not np.isfinite(error) or (mean is not None and abs(mean) > 1.0e-10):
        raise qualification_failure(
            "_surface_qualification_cold_worker",
            "Surface cold continued PDE failed its independent error/zero-mean oracle.",
            {"error": error, "mean": mean},
        )
    return renewed, {
        "result": _result_identity(carrier),
        "l2_error": error,
        "surface_mean": mean,
        "target_error": target_error,
        "target_error_met": error <= target_error,
        "seed_origin": _COLD_SEED_ORIGIN,
        "compiler": compiler,
        "retained_bytes": model_array_bytes((carrier, owner, scalar, problem, solved)),
        "stages_seconds": stages,
    }


def _surface_qualification_restore_epoch(
    publication: dict[str, Any],
    law: _SurfaceLaw,
    epoch: int,
    repeats: int,
    target_error: float,
    *,
    continue_solve: bool,
    budget: phx.discretization.CoordinateEnclosureBudget,
) -> tuple[phx.meshing.CellMeshingResult, dict[str, Any], dict[str, Any] | None, float]:
    """Restore and consume one publication inside its already-open cold request."""
    started = perf_counter()
    restored = read_meshing_source_closure(
        Path(publication["path"]),
        expected_content_id=publication["content_id"],
        limits=ArrayArchiveLimits(),
        budget=budget,
    )
    if array_tree_fingerprint(restored) != publication["fingerprint"]:
        raise qualification_failure(
            "_surface_qualification_cold_worker",
            "Surface cold closure changed source, controls, result or field bits.",
            {"publication": publication, "restored": restored},
        )
    history = validate_meshing_accepted_epochs(restored)
    if len(history) != epoch + 1:
        raise qualification_failure(
            "_surface_qualification_cold_worker",
            "Surface archive must retain every accepted epoch through its publication.",
            {"history": history, "epoch": epoch},
        )
    read_seconds = perf_counter() - started
    renewed, continued = _surface_qualification_renew_epoch(
        restored, history[-1], publication, law, epoch, repeats, target_error,
        continue_solve=continue_solve,
        budget=budget,
    )
    return history[-1].carrier, renewed, continued, read_seconds


def _surface_qualification_cold_worker(
    path: Path,
    content_id: str,
    publications: list[dict[str, Any]],
    law: _SurfaceLaw,
    repeats: int,
    target_error: float,
    deadline: Deadline,
) -> ColdWorkerReport:
    """Renew every retained native map and continue the final accepted PDE state."""

    if not publications or (
        Path(publications[-1]["path"]).resolve(),
        publications[-1]["content_id"],
    ) != (path, content_id):
        raise qualification_failure(
            "_surface_qualification_cold_worker",
            ("Surface cold packet archive is not the final accepted publication."),
            {"content_id": content_id, "path": path, "publications": publications},
        )
    renewed = []
    read_seconds = 0.0
    continuation = None
    for epoch, publication in enumerate(publications):
        control_started = perf_counter()
        controls = read_meshing_source_execution_controls(
            Path(publication["path"]),
            expected_content_id=publication["content_id"],
            limits=ArrayArchiveLimits(),
        )
        read_seconds += perf_counter() - control_started
        restored_request = originals.execute_native_request(
            controls.execution_evidence,
            controls.limits,
            lambda budget: _surface_qualification_restore_epoch(
                publication, law, epoch, repeats, target_error,
                continue_solve=epoch == len(publications) - 1,
                budget=budget,
            ),
            target=lambda value: value[0],
            failure_factory=qualification_failure,
            observe_managed_admission=True,
            workflow_remaining_seconds=deadline.remaining(),
        )
        _, renewed_epoch, continued, epoch_read_seconds = restored_request.value
        read_seconds += epoch_read_seconds
        renewed_epoch["execution"] = (
            None if restored_request.result.execution_evidence is None
            else restored_request.result.execution_evidence.to_record()
        )
        renewed.append(renewed_epoch)
        if continued is not None:
            continuation = continued
    if continuation is None:
        raise qualification_failure(
            "_surface_qualification_cold_worker",
            "Surface cold continuation has no accepted epoch.",
            {"renewed": renewed},
        )
    return ColdWorkerReport(
        record={
            "runtime": _runtime_identity(),
            "accepted_fields_bitwise_preserved": True,
            "prepared_fe_caches_serialized": False,
            "renewed_epochs": renewed,
            "continued_solver_epoch": continuation,
        },
        read_seconds=read_seconds,
    )


def _surface_qualification_adapt(
    result: Any, initial: Any, request: Any, record: dict[str, Any], *,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> Any:

    charts = initial.surface_source
    if (
        charts is None
        or initial.certification is None
        or initial.certification.fidelity is None
    ):
        raise qualification_failure(
            "_surface_qualification_adapt",
            ("Surface adaptation requires retained native charts and source fidelity."),
            {"charts": charts, "initial": initial},
        )
    charts.require_root(initial.mesh, initial.geometry)
    support = PreparedSurfaceSourceSupport(
        charts.domain,
        initial,
        charts.root_parameters,
        charts.boundary_source,
        maximum_support_queries=charts.maximum_support_queries,
        budget=budget,
    )
    vertices = result.mesh.entity_set(0)
    scope = phx.meshing.MeshingScope(
        result.mesh.mesh_id,
        result.mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    size = request.size_controls[0].target_size / 2.0
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(np.eye(3) / size**2, (result.mesh.coordinates.shape[0], 3, 3)),
        minimum_size=size,
        maximum_size=size,
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_SURFACE_METRIC,
        limits=request.limits,
        association_transfer=SurfaceAssociationTransfer(support),
        geometry_transition=phx.discretization.CellGeometryTransitionPolicy(
            reconstruction="bounded_chart_deformation",
            reconstruction_tolerance=initial.certification.fidelity.tolerance,
            maximum_evaluations=request.limits.maximum_work_units,
        ),
        audit_policy=phx.meshing.CellMeshAuditPolicy(require_complete_association=True),
    )
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            result,
            phx.meshing.MetricMeshAdaptation(
                metric, coordinate_contract=result.coordinate_contract
            ),
            policy=policy,
        )
    )
    if (
        adaptation.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
        or adaptation.target.mesh.topology_id == result.mesh.topology_id
    ):
        evidence: Any = adaptation.evidence
        record["adaptation_refusal"] = {
            "status": adaptation.status.value,
            "source_result_id": adaptation.source.result_id,
            "target_result_id": adaptation.target.result_id,
            "resource_message": evidence.resource_message,
            "resource_requested": dict(evidence.resource_requested),
            "resource_achieved": dict(evidence.resource_achieved),
            "work_units": evidence.work_units,
            "native_work_units": evidence.native_work_units,
            "passes": evidence.passes,
            "splits": evidence.splits,
            "collapses": evidence.collapses,
            "flips": evidence.flips,
            "relocations": evidence.relocations,
            "execution": (
                None
                if evidence.execution_evidence is None
                else evidence.execution_evidence.to_record()
            ),
        }
        raise qualification_failure(
            "_surface_qualification_adapt",
            (
                f"Source surface adaptation did not finish a topology transition: {adaptation.status.value}."
            ),
            {"adaptation": adaptation, "result": result},
        )
    return adaptation


def _surface_qualification_adapt_state(
    result: phx.meshing.CellMeshingResult,
    initial: phx.meshing.CellMeshingResult,
    request: phx.meshing.SurfaceMeshingSpec,
    record: dict[str, Any],
    solved: _SurfaceSolvedEpoch,
    epoch: int,
    law: _SurfaceLaw,
    material: phx.equations.MaterialTransaction | None,
    *,
    budget: phx.discretization.CoordinateEnclosureBudget | None,
) -> tuple[Any, tuple[Array, ...], Any, dict[str, object]]:
    """Execute the complete adaptation and physical transfer on one request ledger."""
    if budget is None or material is None:
        raise ValueError(
            "surface.adaptation must retain its actual request ledger and accepted material; got absent state"
        )
    adaptation = _surface_qualification_adapt(
        result, initial, request, record, budget=budget
    )
    moved, auxiliary, carried_material, transition = _surface_qualification_transition(
        result,
        adaptation,
        solved.problem,
        solved.fields[0],
        solved.fields[1:],
        epoch,
        law,
        material,
        request.limits,
        budget=budget,
    )
    return adaptation, (moved, *auxiliary), carried_material, transition


def _surface_qualification_state_inventory(
    result: Any, fields: tuple[Any, ...], material: Any, *,
    epoch: int, error: float,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> tuple[Any, float, dict[str, float], Any]:
    actual_material = (
        _surface_qualification_material_initial(result, budget=budget)
        if material is None else material
    )
    owner = phx.meshing.qualification.prepare_surface_fields(result, budget=budget)
    density = _surface_qualification_content(owner, "density", fields[1])
    norms = {
        name: _surface_qualification_vector_norm(owner, name, fields[index])
        for name, index in (("curl", 3), ("flux", 4))
    }
    accepted = phx.meshing.qualification.SurfaceAcceptedEpoch(
        result=result,
        fields=fields,
        material=actual_material,
        error=error,
        density_content=density,
    )
    checkpoint = phx.meshing.qualification.prepare_surface_checkpoint(
        accepted, epoch, failure_factory=qualification_failure, budget=budget,
        reuse_prepared=owner,
    )
    return actual_material, density, norms, checkpoint


def _surface_qualification_solve_epoch(
    result: phx.meshing.CellMeshingResult,
    law: _SurfaceLaw,
    previous: tuple[Array, ...] | None,
    repeats: int,
    *,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> _SurfaceSolvedEpoch:
    """Solve from authored or carried coefficients and evaluate the physical oracle."""

    space, problem, free, weights = _surface_qualification_solver(
        result, law, budget=budget
    )
    seed = (
        problem.state_space.zeros()
        if previous is None
        else jnp.asarray(previous[0])[free]
    )
    values, compiler, timings = _compiled_solver_stages(problem, seed, repeats)
    fields = (
        _surface_qualification_initial_fields(result, values, budget=budget)
        if previous is None
        else (values, *previous[1:])
    )
    error = _surface_qualification_error(space, values, law)
    mean = (
        None if weights is None else float(weights @ np.asarray(values) / np.sum(weights))
    )
    return _SurfaceSolvedEpoch(
        problem=problem,
        fields=fields,
        compiler=compiler,
        timings=timings,
        error=error,
        mean=mean,
    )


def _surface_qualification_accept_epoch(
    result: phx.meshing.CellMeshingResult,
    request: phx.meshing.SurfaceMeshingSpec,
    solved: _SurfaceSolvedEpoch,
    previous_material: phx.equations.MaterialTransaction | None,
    acceptance: dict[str, object],
    epoch: int,
    authored_initial: bool,
    record: dict[str, Any],
) -> tuple[phx.meshing.qualification.SurfaceAcceptedEpoch, Any]:
    """Require nonzero physical inventories and retain one accepted epoch."""

    fields, error, mean = solved.fields, solved.error, solved.mean
    compiler, timings = solved.compiler, solved.timings
    result, inventory = _corpus_native_consumer(
        result,
        request,
        lambda budget: _surface_qualification_state_inventory(
            result, fields, previous_material, epoch=epoch, error=error, budget=budget
        ),
        native=True,
    )
    material, density, vector_norms, checkpoint = inventory
    if not np.isfinite(density) or density <= 0.0 or any(
        value <= 0.0 or not np.isfinite(value) for value in vector_norms.values()
    ):
        raise qualification_failure(
            "_surface_qualification_lifecycle",
            (
                "Physical surface fields lost their authored nonzero density or compatible moments."
            ),
            {"density": density, "vector_norms": vector_norms},
        )
    record["epochs"].append(
        {
            "epoch": epoch,
            "result": _result_identity(result),
            "source_acceptance": acceptance,
            "physical_error": error,
            "surface_mean": mean,
            "density_content": density,
            "compatible_physical_norms": vector_norms,
            "compiler": compiler,
            "retained_bytes": model_array_bytes(
                (result, solved.problem, fields, material)
            ),
            "seed_origin": "authored-initial-state"
            if authored_initial
            else "actual-carried-u",
        }
    )
    record["stages_seconds"].update(
        {f"{name}_epoch_{epoch}": value for name, value in timings.items()}
    )
    return phx.meshing.qualification.SurfaceAcceptedEpoch(
        result=result,
        fields=fields,
        material=material,
        error=error,
        density_content=density,
    ), checkpoint


def _surface_qualification_record(args: QualificationControls) -> dict[str, Any]:
    """Initialize the surface qualification's external evidence record."""

    record: dict[str, Any] = {
        "scenario": "native-surface-pde",
        "surface_case": args.surface_case,
        "status": "failed",
        "runtime": _runtime_identity(),
        "targets": {"physical_error": args.target_error, "wall_seconds": args.timeout},
        "control_capacities": {
            "resolution": args.resolution,
            "entity_capacity": args.capacity,
        },
        "qualification_scale": {
            "resolution": args.resolution,
            "routine_resolution": _ROUTINE_SURFACE_RESOLUTION,
            "role": "routine"
            if args.resolution == _ROUTINE_SURFACE_RESOLUTION
            else "larger"
            if args.resolution > _ROUTINE_SURFACE_RESOLUTION
            else "below-routine",
            "combined_controls": [
                "curved-and-planar-patches",
                "periodic-rim-curves",
                "hard-size",
                "p50-size-compliance",
                "surface-deviation",
                "rim-curve-deviation",
                "metric-adaptation",
                "all-field-material-transfer",
                "checkpoint-cold-restart",
            ]
            if args.surface_case == _COMBINED_SURFACE_CASE
            else [],
            "cells": None,
            "dofs": None,
        },
        "stages_seconds": {},
        "epochs": [],
        "mandatory_workflow_completion": "incomplete",
        "case_contract_completion": "incomplete",
        "independent_scientific_certification": "unassessed",
        "time_to_target_error_seconds": None,
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
    }
    return record


def _source_fidelity_obligation(
    acceptance: object, law: object, size: object
) -> bool:
    """Recheck one epoch's certified fidelity and its sampled falsification check.

    The certified ``mesh_to_source_upper`` must meet the fidelity tolerance;
    the closed-form source distance at vertices and quadrature points only
    supplements it and must not exceed the certified bound.
    """

    if (
        not isinstance(acceptance, dict) or not isinstance(law, dict)
        or not isinstance(size, (int, float))
    ):
        return False
    try:
        upper = float(acceptance["certified_mesh_to_source_upper"])
        tolerance = float(acceptance["fidelity_tolerance"])
        sampled = float(acceptance["independent_analytic_source_distance"])
        roundoff = float(acceptance["analytic_distance_roundoff"])
        area = float(acceptance["physical_area"])
        source_area, curvature = float(law["area"]), float(law["curvature"])
        bound = source_area * max(0.25 * (curvature * float(size)) ** 2, 1.0e-12)
        euler = int(acceptance["euler_characteristic"])
    except (KeyError, TypeError, ValueError):
        return False
    if not all(np.isfinite((upper, tolerance, sampled, roundoff, area, bound))):
        return False
    return bool(
        acceptance.get("passed") is True
        and acceptance.get("fidelity_status") == "certified"
        and acceptance.get("global_embedding_status") == "certified"
        and acceptance.get("complete_source_associations") is True
        and 0.0 <= upper <= tolerance
        and 0.0 <= sampled <= upper + roundoff
        and abs(area - source_area) <= bound
        and euler == law.get("euler")
    )


def _surface_qualification_obligations(
    record: dict[str, Any], args: QualificationControls, elapsed: float
) -> dict[str, bool]:
    """Derive every profile obligation from the emitted evidence alone."""

    epochs = record.get("epochs", [])
    accepted = [epoch for epoch in epochs if "source_acceptance" in epoch]
    cold = [epoch for epoch in epochs if epoch.get("seed_origin") == _COLD_SEED_ORIGIN]
    transitions = [epoch.get("state_transition") for epoch in accepted[:-1]]
    checkpoint = record.get("checkpoint", {})
    restart = checkpoint.get("cold_restart", {}) if isinstance(checkpoint, dict) else {}
    fields = [field.name for _, field in phx.meshing.qualification.surface_field_templates()]
    errors = [epoch.get("physical_error") for epoch in accepted]
    measured = ("temporary_bytes", "output_bytes", "generated_code_bytes")
    law = record.get("profile_law")
    size = record.get("qualification_scale", {}).get("requested_target_size")
    obligations = {
        "original_source_fidelity_and_global_embedding": len(accepted)
        == args.adaptation_rounds + 1
        and all(
            _source_fidelity_obligation(epoch["source_acceptance"], law, size)
            for epoch in accepted
        ),
        "adaptation_all_field_material_transfer_and_rollback": args.adaptation_rounds
        >= 1
        and len(transitions) == args.adaptation_rounds
        and all(
            isinstance(item, dict)
            and item.get("receipt_id") is not None
            and item.get("field_names") == fields
            and bool(item.get("field_transfers"))
            and all(entry.get("passed") is True for entry in item["field_transfers"])
            and item.get("material_transport", {}).get("passed") is True
            and item.get("rollback", {}).get("passed") is True
            for item in transitions
        ),
        "independent_manufactured_pde": bool(errors)
        and all(
            isinstance(value, (int, float)) and np.isfinite(value) for value in errors
        )
        and errors[-1] <= args.target_error,
        "checkpoint_cold_restart_continuation": isinstance(checkpoint, dict)
        and len(checkpoint.get("archives", ())) == len(accepted)
        and len(restart.get("renewed_epochs", ())) == len(accepted)
        and len(cold) == 1
        and cold[0].get("target_error_met") is True,
        "runtime_phase_and_deadline_evidence": elapsed <= args.timeout
        and all(
            isinstance(epoch.get("compiler"), dict)
            and all(epoch["compiler"].get(name) is not None for name in measured)
            and bool(epoch["compiler"].get("warm_samples_seconds"))
            and epoch.get("retained_bytes") is not None
            for epoch in epochs
        ),
    }
    if args.surface_case == _COMBINED_SURFACE_CASE:
        # The request governs the generated epoch; its values are rederived
        # from the resolution, not read back from the record.
        obligations["combined_size_and_feature_controls_measured"] = bool(
            accepted
        ) and _combined_feature_verdict(
            accepted[0]["source_acceptance"].get("feature_controls"),
            _combined_feature_request(args.resolution),
        )
    return obligations


def _surface_qualification_completion(
    accepted: phx.meshing.qualification.SurfaceAcceptedEpoch,
    acceptance: dict[str, object],
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
) -> None:
    """Publish the final physical verdict and evidence-derived profile completion."""
    elapsed = perf_counter() - started
    final_error = accepted.error
    passed = final_error <= args.target_error and elapsed <= args.timeout
    obligations = _surface_qualification_obligations(record, args, elapsed)
    complete = passed and all(obligations.values())
    record.update(
        {
            "status": "passed"
            if complete
            else "timeout"
            if elapsed > args.timeout
            else "failed",
            "physical_error": final_error,
            "target_error_reached": final_error <= args.target_error,
            "independent_scientific_certification": (
                "passed" if final_error <= args.target_error else "failed"
            ),
            "case_contract_completion": "complete" if complete else "incomplete",
            "profile_obligations": obligations,
            "mandatory_workflow_completion": "complete" if complete else "incomplete",
            "independent_acceptance": acceptance,
            "source_acceptance": acceptance,
            "time_to_target_error_seconds": (
                record["time_to_target_error_seconds"] or elapsed if passed else None
            ),
            "field_names": ["u", "density", "history", "curl", "flux"],
        }
    )
    record["qualification_scale"].update(
        {
            "cells": sum(block.cell_count for block in accepted.result.mesh.blocks),
            "dofs": int(np.asarray(accepted.fields[0]).shape[0]),
        }
    )


def _surface_qualification_lifecycle(args: QualificationControls) -> dict[str, object]:

    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    record = _surface_qualification_record(args)
    _ledger = StageLedger(record, deadline=deadline)
    phases: list[NativeMeshingPhaseMeasurement] = []
    phase = "original_source_request_preparation"
    try:
        prepared, timing = _ledger.run(
            phase, lambda: _measured(lambda: _surface_qualification_prepare(args))
        )
        source, request, options, law = prepared
        record["stages_seconds"][phase] = timing["wall_seconds"]
        record.update(
            {
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "source_binding_id": source.binding_id,
                "request_id": request.specification_id,
                "limits_id": request.limits.limits_id,
                "profile_law": law,
            }
        )
        target = request.size_controls[0].target_size
        record["qualification_scale"].update(
            {
                "requested_target_size": target,
                # Non-routine resolutions request a different size, so their
                # request (not source) identity differs from the routine case.
                "request_identity": "routine-authored"
                if args.resolution == _ROUTINE_SURFACE_RESOLUTION
                else "resolution-scaled-size",
            }
        )
        phase = "original_native_source_preparation"
        provider = phx.meshing.NativeMeshingProvider(options)
        plan, timing = _ledger.run(
            phase,
            lambda: _measured(
                lambda: provider.plan(
                    source,
                    request,
                    coordinate_contract=phx.SpatialCoordinateContract.si(),
                    record_phase=phases.append,
                )
            ),
        )
        record["stages_seconds"][phase] = timing["wall_seconds"]
        phase = "original_public_surface_publication"
        result, timing = _ledger.run(
            phase, lambda: _measured(lambda: plan.execute(record_phase=phases.append))
        )
        record["stages_seconds"][phase] = timing["wall_seconds"]
        initial = result
        record["generation"] = _result_identity(initial)
        accepted_history: list[phx.meshing.qualification.SurfaceAcceptedEpoch] = []
        checkpoint_states = []
        adaptations = []
        previous = None
        previous_material = None
        for epoch in range(args.adaptation_rounds + 1):
            phase = f"source_acceptance_epoch_{epoch}"
            result, acceptance = _corpus_native_consumer(
                result,
                request,
                lambda budget: _surface_qualification_acceptance(
                    source,
                    result,
                    request,
                    law,
                    combined=args.surface_case == _COMBINED_SURFACE_CASE,
                    budget=budget,
                ),
                native=True,
            )
            if not acceptance["passed"]:
                raise qualification_failure(
                    "_surface_qualification_lifecycle",
                    (
                        "Surface publication failed the source, physical measure or topology oracle."
                    ),
                    {"acceptance": acceptance},
                )
            phase = f"physical_solver_epoch_{epoch}"

            result, solved = _corpus_native_consumer(
                result,
                request,
                lambda budget: _surface_qualification_solve_epoch(
                    result, law, previous, args.repeats, budget=budget
                ),
                native=True,
            )
            error, mean = solved.error, solved.mean
            if not np.isfinite(error) or (
                law["name"] == "sphere" and (mean is None or abs(mean) > 1.0e-10)
            ):
                raise qualification_failure(
                    "_surface_qualification_lifecycle",
                    (
                        "Physical surface PDE failed its independent error/zero-mean oracle."
                    ),
                    {"error": error, "law": law, "mean": mean},
                )
            accepted, checkpoint_state = _surface_qualification_accept_epoch(
                result,
                request,
                solved,
                previous_material,
                acceptance,
                epoch,
                previous is None,
                record,
            )
            result, material = accepted.result, accepted.material
            accepted_history.append(accepted)
            checkpoint_states.append(checkpoint_state)
            if epoch == args.adaptation_rounds:
                break
            phase = f"original_source_metric_adaptation_epoch_{epoch}"
            # Each adaptation is a new complete request: original limits, the
            # accepted epoch as history, and the unchanged workflow deadline.
            adapted, adapted_state = _corpus_native_request(
                result,
                request,
                lambda budget: _surface_qualification_adapt_state(
                    result, initial, request, record, solved, epoch, law, material,
                    budget=budget,
                ),
                native=True,
                target=lambda value: value[0].target,
                deadline=deadline,
            )
            adaptation, previous, previous_material, transition = adapted_state
            adaptations.append(adaptation)
            transition["execution"] = (
                None if adapted.execution_evidence is None
                else adapted.execution_evidence.to_record()
            )
            record["epochs"][-1].update(
                {
                    "state_transition": transition,
                    "adaptation": {
                        "status": adaptation.status.value,
                        "source_result_id": adaptation.source.result_id,
                        "target": _result_identity(adaptation.target),
                    },
                }
            )
            result = adapted
        phase = "default_allfield_accepted_history_checkpoint_cold_continuation"
        _surface_qualification_checkpoint(
            source,
            result,
            request,
            options,
            law,
            tuple(accepted_history),
            tuple(adaptations),
            tuple(checkpoint_states),
            args,
            record,
            started,
            deadline,
        )
        _surface_qualification_completion(
            accepted_history[-1], acceptance, args, record, started
        )
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, phase)
    except (RuntimeError, ValueError, TypeError, TimeoutError) as failure:
        record_failure(record, failure, phase)
    finally:
        record["native_phases"] = _phase_recordings(phases)
        record["end_to_end_seconds"] = perf_counter() - started
        record["peak_process_resident_bytes"] = peak_resident_bytes()
    return record


def qualify_native_surface(args: QualificationControls) -> dict[str, object]:
    """Exercise the source, all physical states and cold continued PDE."""
    return _surface_qualification_lifecycle(args)


def cold_worker(packet: ColdControlPacket) -> ColdWorkerReport:
    """Run the scenario-owned cold consumer on its one control packet."""
    controls = packet.controls
    return _surface_qualification_cold_worker(
        packet.archive_path,
        packet.content_id,
        controls["publications"],
        controls["law"],
        int(controls["repeats"]),
        float(controls["target_error"]),
        packet.deadline,
    )


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host qualification controls."""
    return qualify_native_surface(args)
