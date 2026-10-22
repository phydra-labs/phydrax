# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import os
from fractions import Fraction
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import phydrax as phx
from phydrax import model_array_bytes, StrictModule
from phydrax._meshcore import current_native_execution_scope
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization.fem._topology_transfer import (
    prepare_shared_nested_field_transfer,
)
from phydrax.meshing import NativeMeshingPhaseMeasurement
from phydrax.meshing.qualification import (
    family_finite_volume_field_templates,
    family_oracle_content_id,
    family_solver_field_templates,
    family_transport_field_templates,
    FamilyAssociationTransfer,
    FamilyPreparationError,
    prepare_family_annular_block_source,
    prepare_family_association_transfer,
    prepare_family_finite_element_bindings,
    prepare_family_finite_volume_bindings,
    prepare_family_frame_sweep_source,
    prepare_family_quality_measurements,
    prepare_family_reversed_multiblock_source,
    read_family_oracle,
    write_family_oracle,
)
from phydrax.units import METER
from tools.meshing_qualification._controls import (
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._family_progress import (
    record_family_native_phase,
    run_with_family_progress,
)
from tools.meshing_qualification._records import FailureRecord, record_failure


@final
class _FamilyFixtureParts(StrictModule):
    source: (
        phx.meshing.NativePlanarSource
        | phx.meshing.NativeSurfaceSource
        | phx.meshing.NativePlcSource
        | phx.meshing.NativeSweepSource
        | phx.meshing.NativeStructuredSource
        | phx.meshing.NativeMappedHexSource
    )
    domain: (
        phx.geometry.PiecewiseLinearDomain
        | phx.geometry.MappedReferenceDomain
        | phx.geometry.MeshingDomain
        | None
    )
    expected: dict[str, float] = eqx.field(static=True)
    scope_ids: np.ndarray
    regions: tuple[phx.meshing.RegionControl, ...]
    patches: tuple[phx.meshing.PatchControl, ...]
    holes: tuple[phx.meshing.HoleSeed, ...]
    embedding: phx.geometry.PlanarEmbedding | None = eqx.field(static=True)


@final
class _FamilyRequestProfile(StrictModule):
    name: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    geometry_order: int = eqx.field(static=True)
    family: str = eqx.field(static=True)
    route: phx.meshing.NativeMeshingRoute = eqx.field(static=True)


def _family_request_for_profile(
    profile: _FamilyRequestProfile,
    parts: _FamilyFixtureParts,
    case: dict[str, Any],
    args: QualificationControls,
    limits: phx.meshing.MeshingLimits,
) -> phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec:
    """Bind authored source scopes and freeze the profile's hard family and size request."""
    n = args.resolution
    source, domain, scope_ids, regions, patches, holes, embedding = (
        parts.source,
        parts.domain,
        parts.scope_ids,
        parts.regions,
        parts.patches,
        parts.holes,
        parts.embedding,
    )
    dimension = profile.dimension
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2)
        if isinstance(
            domain, (phx.geometry.MappedReferenceDomain, phx.geometry.MeshingDomain)
        )
        else "family-boundary",
        scope_ids,
    )
    family_policy = (
        phx.meshing.CellFamilyPolicy(**case["family_policy"])
        if profile.name == "hex-dominant"
        else phx.meshing.CellFamilyPolicy(required=(profile.family,))
    )
    target = phx.meshing.CellMeshingTarget(
        dimension,
        profile.ambient_dimension,
        family_policy,
        geometry_order=profile.geometry_order,
    )
    sizes = (
        phx.meshing.UniformSizeControl(
            scope,
            1.0 / n,
            maximum_size=2.0 / n,
            strength=phx.meshing.SizeControlStrength.HARD,
        ),
    )
    compliance = phx.meshing.SizeCompliancePolicy(
        relative_tolerance=0.95, target_statistics=("p50",)
    )
    if dimension == 2:
        request = phx.meshing.SurfaceMeshingSpec(
            target,
            scope,
            planar_embedding=embedding,
            size_controls=sizes,
            size_compliance=compliance,
            limits=limits,
            protected_features=(
                phx.meshing.ProtectedFeature(
                    scope,
                    phx.meshing.FeatureKind.SURFACE,
                    maximum_deviation=0.0,
                ),
            )
            if args.family_case == "curved-quarter-extrusion-quad"
            else (),
            quality_target=phx.meshing.MeshQualityTarget(minimum_angle=0.01, hard=True),
        )
    else:
        request = phx.meshing.VolumeMeshingSpec(
            target,
            scope,
            phx.meshing.VolumeFillStrategy.SWEEP
            if profile.route == "sweep"
            else phx.meshing.VolumeFillStrategy.HEX_DOMINANT_SIMPLEX_TRANSITION
            if profile.name == "hex-dominant"
            else phx.meshing.VolumeFillStrategy.MULTIZONE,
            size_controls=sizes,
            size_compliance=compliance,
            limits=limits,
            region_controls=regions,
            patch_controls=patches,
            hole_seeds=holes,
        )
    return request


def _family_qualification_fixture(
    args: QualificationControls,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Author independent sources; use the structured and PLC fixtures."""

    profiles = _family_qualification_requirements()["profiles"]
    if args.family_case not in profiles:
        raise ValueError(
            f"_family_qualification_fixture.family_case must identify a registered source/family profile; got {args.family_case!r}."
        )
    declared = profiles[args.family_case]
    case: dict[str, Any] = dict(declared)
    case["name"] = args.family_case
    capacity = args.capacity
    limits = capacity_meshing_limits(
        capacity,
        args.timeout,
        maximum_work_units=args.maximum_work_units,
        maximum_geometry_queries=args.maximum_geometry_queries,
        scratch_bytes_limit=args.maximum_scratch_bytes,
        data_bytes_limit=args.maximum_data_bytes,
        edges=12,
        faces=12,
        cells=1,
        connectivity_entries=64,
        data_bytes=1024,
        work_units=100,
        geometry_queries=100,
        scratch_bytes=4096,
    )
    profile = _FamilyRequestProfile(
        args.family_case,
        declared["dimension"],
        declared.get("ambient_dimension", declared["dimension"]),
        declared.get("geometry_order", 1),
        declared["family"],
        declared["route"],
    )
    match args.family_case:
        case "dual-quad":
            parts = _family_build_dual_quad(args, case, limits)
        case "curved-quarter-extrusion-quad":
            parts = _family_build_quarter_extrusion_quad(args, case, limits)
        case "dual-hex" | "hex-dominant":
            parts = _family_build_plc_hex(args, case, limits)
        case "nonsweepable-dual-hex":
            parts = _family_build_nonsweepable_dual_hex(args, case, limits)
        case "curved-frame-sweep-hex":
            parts = _family_build_frame_sweep_hex(args, case, limits)
        case "multiblock-reversed-hex" | "multiblock-material-hex":
            parts = _family_build_reversed_multiblock_hex(args, case, limits)
        case "multiblock-cycle-quad":
            parts = _family_build_annular_quad(args, case, limits)
        case "curved-nonsweepable-hex" | "curved-cavity-hex" | "curved-material-hex":
            parts = _family_build_curved_mapped_hex(args, case, limits)
        case "transfinite-quad" | "multiblock-quad":
            parts = _family_build_structured_quad(args, case, limits)
        case "sweep-hex" | "sweep-prism":
            parts = _family_build_sweep(args, case, limits)
        case _:
            raise ValueError(
                f"qualification.family_case must name a declared source profile; got {args.family_case!r}"
            )
    request = _family_request_for_profile(profile, parts, case, args, limits)
    source, domain, expected = parts.source, parts.domain, parts.expected
    case.update(
        {
            "domain": domain,
            "expected_regions": expected,
            "source_id": source.source_id,
            "source_revision": source.source_revision,
        }
    )
    options = (
        phx.meshing.NativeMeshingOptions(
            case["route"], hex_core_fraction=case["core_fraction"]
        )
        if case["name"] == "hex-dominant"
        else phx.meshing.NativeMeshingOptions(case["route"])
    )
    return source, request, options, case


@final
class _FamilySourceAcceptance(StrictModule):
    renewed: phx.meshing.MeshCertificationReport
    mesh: phx.discretization.CellMesh
    regions: np.ndarray
    coverage_record: dict[str, Any]
    measured: dict[str, float | None] | None
    embedding_id: str = eqx.field(static=True)


def _family_qualification_acceptance(
    result: phx.meshing.CellMeshingResult,
    request: phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec,
    case: dict[str, Any],
    authority: phx.meshing.MeshCertificationInputs,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> dict[str, object]:
    """Certify publication, source coverage, material topology, and quality independently."""
    family_record, certificate = _family_publication_acceptance(result, request, case)
    source = _family_source_acceptance(
        authority, result, case, certificate, budget=budget
    )
    renewed, mesh, regions = source.renewed, source.mesh, source.regions
    coverage_record, measured, embedding_id = (
        source.coverage_record,
        source.measured,
        source.embedding_id,
    )
    material_record = _family_material_acceptance(
        result, case, mesh, regions, family_record
    )
    quality_record = _family_quality_acceptance(
        result, request, case, mesh, budget=budget
    )
    obligation = _family_obligation_class(case)
    class_record = (
        {"topology": _family_topology_acceptance(mesh, regions, case)}
        if obligation == "general-nonsweepable-all-hex"
        else {"feature_topology": _family_quad_feature_acceptance(mesh, case)}
        if obligation == "feature-conforming-quad"
        else {}
    )
    return {
        "passed": True,
        "certification_id": renewed.report_id,
        "obligation_class": obligation,
        **family_record,
        **class_record,
        "embedding_id": embedding_id,
        **coverage_record,
        "expected_region_measures": case["expected_regions"],
        "achieved_region_measures": measured,
        "source_fidelity": "exact-PL-domain-coverage-not-two-sided-fidelity"
        if renewed.fidelity is None
        else {
            "status": renewed.fidelity.status,
            "certificate_id": renewed.fidelity.certificate_id,
        },
        "source_authority": {
            "source_id": case["source_id"],
            "source_revision": case["source_revision"],
            "domain_id": authority.domain_id,
            "query_id": authority.source_id,
        },
        "material_organization": material_record,
        **quality_record,
    }


def _quarter_extrusion_manufactured_value(points: Any) -> Any:
    """Physical axial quadratic on the straight-extruded surface."""
    return points[..., 0] ** 2


def _quarter_extrusion_manufactured_source(points: Any, parameters: Any) -> Any:
    # The authored surface is a Cartesian product with its unit axial interval.
    # Thus its intrinsic Laplacian of x² is exactly 2, regardless of arc weights.
    del parameters
    return jnp.full(points.shape[:-1], -2.0, dtype=points.dtype)


def _family_qualification_solver(
    result: Any,
    case: dict[str, Any],
    *,
    budget: CoordinateEnclosureBudget | None = None,
    prepared_space: phx.discretization.FiniteElementDiscretization | None = None,
) -> tuple[Any, Any, Any]:
    """Join the tensor Q1 consumer, retaining its coordinate maps."""

    square = case["name"] in ("transfinite-quad", "multiblock-quad")
    curved_quad = case["name"] == "curved-quarter-extrusion-quad"
    if case["name"] == "hex-dominant":
        templates = {
            kind: family_solver_field_templates(kind)[0][1]
            for kind in {block.cell_kind for block in result.mesh.blocks}
        }
        template = next(iter(templates.values()))
        field = phx.discretization.FiniteElementFieldSpec(
            template.name,
            {
                block.name: templates[block.cell_kind].elements[0]
                for block in result.mesh.blocks
            },
            component_shape=template.component_shape,
        )
    else:
        field = family_solver_field_templates(case["family"])[0][1]
    plan = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry, budget=budget
    )
    if prepared_space is None:
        space = plan.prepare(budget=budget)
    else:
        prepared_space.require_exact_plan(
            plan, numeric_version=prepared_space.numeric_version, budget=budget
        )
        space = prepared_space
    form = phx.equations.FiniteElementForm(
        "exact-family-manufactured-diffusion",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    _quarter_extrusion_manufactured_source
                    if curved_quad
                    else manufactured_source
                    if square
                    else _manufactured_source,
                    coefficient_id="original-extrusion-axial-quadratic-source"
                    if curved_quad
                    else "square-polynomial-source"
                    if square
                    else "quadratic-source",
                ),
            ),
        ),
    )
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=_quarter_extrusion_manufactured_value
        if curved_quad
        else manufactured_solution
        if square
        else _manufactured_value,
        budget=budget,
    )
    return space, problem, constraint.free_dofs


def _family_qualification_physical(
    space: Any, values: Any, case: dict[str, Any]
) -> dict[str, float]:
    """Independent higher-order physical quadrature; not an area-only consumer."""

    line = phx.integration.GaussLegendreRule(6)
    rule = (
        phx.integration.ReferenceQuadrilateralRule(line)
        if case["dimension"] == 2
        else phx.integration.ReferencePrismRule(line)
        if case["family"] == "prism"
        else phx.integration.ReferenceHexahedronRule(line)
    )
    quadrature = phx.integration.reference_rule_data(rule)
    if case["name"] == "hex-dominant":
        quadratures = {
            "hexahedron": quadrature,
            "tetrahedron": phx.integration.reference_rule_data(
                phx.integration.ReferenceTetrahedronRule(line)
            ),
            "pyramid": phx.integration.reference_rule_data(
                phx.integration.ReferencePyramidRule(line)
            ),
        }
    squared, inventory = 0.0, 0.0
    exact = (
        _quarter_extrusion_manufactured_value
        if case["name"] == "curved-quarter-extrusion-quad"
        else manufactured_solution
        if case["name"] in ("transfinite-quad", "multiblock-quad")
        else _manufactured_value
    )
    for index in range(len(space.mesh.blocks)):
        if case["name"] == "hex-dominant":
            quadrature = quadratures[space.mesh.blocks[index].cell_kind]
        geometry = space.evaluate_block_geometry(
            "u",
            index,
            space.default_runtime.coordinates,
            quadrature.points,
            quadrature.weights,
        )
        dofs = space.dof_maps[0].cell_dofs[index]
        discrete = phx.ein.contract("ql,cl->cq", geometry.basis_values, values[dofs])
        weights = np.asarray(geometry.physical_weights)
        squared += float(
            np.sum(weights * np.asarray(discrete - exact(geometry.physical_points)) ** 2)
        )
        inventory += float(np.sum(weights * np.asarray(discrete)))
    return {"l2_error": float(np.sqrt(squared)), "inventory": inventory}


def _family_mapped_fv(
    publication: Any,
    case: dict[str, Any],
    *,
    content: Any,
    budget: CoordinateEnclosureBudget | None = None,
    fv: phx.discretization.UnstructuredFiniteVolumeDiscretization | None = None,
) -> dict[str, Any]:
    """Measure mapped FV volumes and the stationary Euler equation on each epoch."""

    dimension = publication.mesh.ambient_dimension
    system = phx.equations.EulerSystem(dimension)
    template = family_finite_volume_field_templates(dimension)[0]
    prepared_fv = fv
    fv, measured_preparation = _measured(
        lambda: _family_fv(publication, budget=budget) if prepared_fv is None else prepared_fv
    )
    preparation = dict(measured_preparation)
    preparation["reused_prepared_owner"] = prepared_fv is not None
    volumes = np.asarray(fv.cell_volumes)
    source_measure = sum(case["expected_regions"].values())
    measured_volume = float(np.sum(volumes))
    if (
        not np.all(np.isfinite(volumes))
        or np.any(volumes <= 0.0)
        or abs(measured_volume - source_measure) > 1e-10 * source_measure
    ):
        raise qualification_failure(
            "_family_mapped_fv",
            ("Structured FV volumes disagree with the independent source domain."),
            {
                "measured_volume": measured_volume,
                "source_measure": source_measure,
                "volumes": volumes,
            },
        )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        fv.boundary_patch_names,
        {name: phx.discretization.SlipWallBoundary() for name in fv.boundary_patch_names},
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    dynamics = phx.discretization.PreparedUnstructuredFiniteVolumeDynamics(
        system, fv, method, boundaries
    )
    content = jnp.asarray(content)
    if content.shape != (fv.cell_count, *template["component_shape"]):
        raise qualification_failure(
            "_family_mapped_fv",
            ("Continued family FV state lost a conserved component."),
            {"content": content, "dimension": dimension, "fv": fv},
        )
    state = content / fv.cell_volumes[:, None]
    rate, execution = _measured(
        lambda: dynamics(jnp.asarray(0.0, dtype=state.dtype), state)
    )
    defect = float(jnp.max(jnp.abs(rate)))
    if not np.isfinite(defect) or defect > 1e-10:
        raise qualification_failure(
            "_family_mapped_fv",
            ("Structured mapped FV stationary Euler content-rate theorem failed."),
            {"defect": defect},
        )
    return {
        "source_measure": source_measure,
        "mapped_cell_volume_sum": measured_volume,
        "stationary_euler_maximum_content_rate": defect,
        "preparation": preparation,
        "execution": execution,
        "seed_origin": "actual-accepted-material-content",
    }


def _family_qualification_lineage(adaptation: Any) -> dict[str, object]:
    """Account for entity dimensions independently of family corner count."""

    source, target, lineage = (
        adaptation.source.mesh,
        adaptation.target.mesh,
        adaptation.lineage,
    )
    if lineage is None:
        raise qualification_failure(
            "_family_qualification_lineage",
            "retain lineage for every entity dimension",
            {"lineage": lineage},
        )
    return lineage.validate_complete(source, target).to_record()


def _family_quad_fv(
    publication: Any,
    *,
    accepted_content: Any,
    budget: CoordinateEnclosureBudget | None = None,
    fv: phx.discretization.UnstructuredFiniteVolumeDiscretization | None = None,
) -> dict[str, float]:
    """Measure embedded FV content against the independently authored source."""

    source = publication.certification.request.source
    if not isinstance(source, SurfaceNativeRestrictionBoundarySource) or not isinstance(
        source.support.original, SurfaceSourceRootAtlas
    ):
        raise TypeError(
            f"_family_quad_fv.source must be a SurfaceNativeRestrictionBoundarySource with a SurfaceSourceRootAtlas; got {type(source).__name__!r}."
        )
    original = source.support.original
    original.require_current()
    template = family_finite_volume_field_templates(
        publication.mesh.topological_dimension, scalar_quadrature=True
    )[0]
    source_fv = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        original.mesh,
        field_name=template["name"],
        component_names=template["component_names"],
        budget=budget,
    ).prepare(cell_geometry=original.geometry, budget=budget)
    source_area = float(np.sum(np.asarray(source_fv.cell_volumes)))
    if fv is None:
        fv = _family_fv(publication, budget=budget)
    weights = np.where(
        np.asarray(fv.cell_quadrature_valid), np.asarray(fv.cell_quadrature_weights), 0.0
    )
    area = float(np.sum(weights))
    accepted = np.asarray(accepted_content)
    if accepted.shape != (fv.cell_count, 1) or not np.all(np.isfinite(accepted)):
        raise qualification_failure(
            "_family_quad_fv",
            ("Continued surface FV content lost its accepted scalar layout."),
            {"accepted_content": accepted, "fv": fv},
        )
    content = float(np.sum(accepted))
    defect = abs(content - source_area / 3.0)
    if (
        not np.all(np.isfinite((area, source_area, content)))
        or area <= 0.0
        or source_area <= 0.0
        or np.any(np.asarray(fv.cell_volumes) <= 0.0)
        or defect > 1.0e-10 * source_area
        or abs(area - source_area) > 1.0e-10 * source_area
    ):
        raise qualification_failure(
            "_family_quad_fv",
            ("Curved quad embedded FV source-area/axial-content theorem failed."),
            {
                "area": area,
                "content": content,
                "defect": defect,
                "fv": fv,
                "source_area": source_area,
            },
        )
    return {
        "surface_area": area,
        "original_source_area": source_area,
        "absolute_source_area_defect": abs(area - source_area),
        "axial_quadratic_content": content,
        "exact_product_content": source_area / 3.0,
        "absolute_content_defect": defect,
    }


def _family_fv(
    result: Any, *, budget: CoordinateEnclosureBudget | None = None
) -> Any:
    template = family_finite_volume_field_templates(
        result.mesh.ambient_dimension,
        scalar_quadrature=result.mesh.topological_dimension
        != result.mesh.ambient_dimension,
    )[0]
    return phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        result.mesh,
        field_name=template["name"],
        component_names=template["component_names"],
        budget=budget,
    ).prepare(cell_geometry=result.geometry, budget=budget)


def _family_allfield_spaces(
    result: Any, *, budget: CoordinateEnclosureBudget | None = None
) -> dict[str, Any]:
    """Prepare the family spaces, never infer from array width."""

    spaces = {}
    for kind in LAYER_CORE_FIELD_KINDS:
        templates = {
            cell_kind: {
                field.name: field
                for _, field in family_transport_field_templates(cell_kind, kind=kind)
            }
            for cell_kind in {block.cell_kind for block in result.mesh.blocks}
        }
        fields = tuple(
            phx.discretization.FiniteElementFieldSpec(
                field.name,
                {
                    block.name: templates[block.cell_kind][field.name].elements[0]
                    for block in result.mesh.blocks
                },
                component_shape=field.component_shape,
            )
            for field in next(iter(templates.values())).values()
        )
        spaces[kind] = phx.discretization.FiniteElementPlan(
            result.mesh,
            fields,
            coordinate_spec=result.geometry,
            budget=budget,
        ).prepare(budget=budget)
    return spaces


def _family_scalar_content(space: Any, values: np.ndarray) -> np.ndarray:
    """Use the physical surface measure, not a determinant of a rectangular map."""
    if space.mesh.topological_dimension == space.mesh.ambient_dimension:
        return layer_core_scalar_content(space, values)
    rule = phx.integration.ReferenceQuadrilateralRule(
        phx.integration.GaussLegendreRule(8)
    ).materialize()
    total = np.zeros(values.shape[1:], dtype=np.float64)
    for block_index, (element, routes) in enumerate(
        zip(space.elements[0], space.dof_maps[0].cell_dofs, strict=True)
    ):
        geometry = space.evaluate_block_geometry(
            "current",
            block_index,
            space.default_runtime.coordinates,
            rule.points,
            rule.weights,
        )
        basis = np.asarray(element.tabulate(rule.points)[0])
        local = np.asarray(values)[np.asarray(routes)]
        total += np.sum(
            np.asarray(geometry.physical_weights)[..., None] * (basis @ local),
            axis=(0, 1),
        )
    return total


def _family_constant_vector_projection(points: Any, vector: Any) -> Any:
    """Project the declared vector data without recompiling its coefficient law."""
    return jnp.broadcast_to(vector, points.shape)


def _family_allfield_state(
    result: Any,
    *,
    spaces: dict[str, Any],
    budget: CoordinateEnclosureBudget | None = None,
    fv: phx.discretization.UnstructuredFiniteVolumeDiscretization | None = None,
) -> dict[str, Any]:

    dimension = result.mesh.topological_dimension
    ambient = result.mesh.ambient_dimension
    embedded = dimension != ambient
    fields = {}
    for kind, space in spaces.items():
        if kind in ("H1", "DG"):
            if dimension == 3:
                fields[kind] = jnp.asarray(layer_core_initial_values(space, kind))
            else:
                first = space.project(
                    "current",
                    lambda points, args: (
                        1.0 + points[:, 0]
                        if embedded
                        else 1.0 + points[:, 0] + points[:, 1]
                    ),
                )
                second = space.project(
                    "history",
                    lambda points, args: (
                        0.5 + 0.125 * points[:, 0]
                        if embedded
                        else 0.5 + 0.125 * points[:, 0] + 0.1 * points[:, 1]
                    ),
                )
                fields[kind] = jnp.stack((first, second), axis=-1)
        else:
            # Physical moments of the two scalar/history gradients;
            # no random coefficients or alternate form basis.
            if dimension == 3:
                first = space.project(
                    "current",
                    lambda points, args: jnp.broadcast_to(
                        jnp.asarray((0.0, 1.0, 1.0)), points.shape
                    ),
                )
                second = space.project(
                    "history",
                    lambda points, args: jnp.stack(
                        (
                            jnp.zeros_like(points[:, 0]),
                            0.25 * points[:, 1],
                            0.2 * points[:, 2],
                        ),
                        axis=-1,
                    ),
                )
            else:
                current = (1.0, 0.0, 0.0) if embedded else (1.0, 1.0)
                history = (0.125, 0.0, 0.0) if embedded else (0.125, 0.1)
                first = space.project(
                    "current",
                    _family_constant_vector_projection,
                    args=jnp.asarray(current),
                )
                second = space.project(
                    "history",
                    _family_constant_vector_projection,
                    args=jnp.asarray(history),
                )
            fields[kind] = jnp.stack((first, second), axis=-1)
    if fv is None:
        fv = _family_fv(result, budget=budget)
    if embedded:
        weights = jnp.where(fv.cell_quadrature_valid, fv.cell_quadrature_weights, 0.0)
        fields["material_content"] = jnp.sum(
            weights * fv.cell_quadrature_points[..., 0] ** 2, axis=1
        )[:, None]
    else:
        system = phx.equations.EulerSystem(ambient)
        template = family_finite_volume_field_templates(ambient)[0]
        primitive = jnp.zeros(
            (fv.cell_count, *template["component_shape"]),
            dtype=result.mesh.coordinates.dtype,
        )
        primitive = primitive.at[:, 0].set(1.0).at[:, -1].set(1.0)
        fields["material_content"] = fv.cell_volumes[
            :, None
        ] * system.primitive_to_conserved(primitive)
    return fields


def _family_allfield_transfer(
    adaptation: Any,
    fields: dict[str, Any],
    *,
    refine: bool,
    source_spaces: dict[str, Any],
    target_spaces: dict[str, Any],
    budget: CoordinateEnclosureBudget | None = None,
    source_fv: phx.discretization.UnstructuredFiniteVolumeDiscretization | None = None,
    target_fv: phx.discretization.UnstructuredFiniteVolumeDiscretization | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Carry both components of every form/scalar role through the epoch."""

    source, target = source_spaces, target_spaces
    carried, evidence = {}, {}
    prepared_references = []
    for kind in source:
        values = fields[kind]
        names = tuple(field.name for field in source[kind].construction_plan.fields)
        parts, field_evidence = [], []
        for index, name in enumerate(names):
            transfer, reference = prepare_shared_nested_field_transfer(
                source[kind],
                target[kind],
                adaptation.parent_cells if refine else None,
                field_name=name,
                parent_reference_vertices=adaptation.parent_reference_vertices
                if refine
                else None,
                parent_reference_homogeneous=adaptation.parent_reference_homogeneous
                if refine
                else None,
                source_geometry=adaptation.source.geometry,
                target_geometry=adaptation.target.geometry,
                geometry_transition=adaptation.geometry_transition,
                coarsening_witnesses=None if refine else adaptation.coarsening_witnesses,
                prepared=tuple(prepared_references),
                budget=budget,
            )
            if reference is not None and all(
                reference is not previous for previous in prepared_references
            ):
                prepared_references.append(reference)
            if not transfer.evidence.passed:
                raise qualification_failure(
                    "_family_allfield_transfer",
                    (f"Family {kind}/{name} transfer failed its certificate."),
                    {"transfer": transfer},
                )
            state = values if len(names) == 1 else values[:, index]
            mapped = jax.block_until_ready(transfer.transfer.apply(state))
            tangent = jnp.ones_like(state)
            _, jvp = jax.jvp(transfer.transfer.apply, (state,), (tangent,))
            dual = jnp.ones_like(mapped)
            _, vjp = jax.vjp(transfer.transfer.apply, state)
            expected_jvp = transfer.transfer.apply(tangent)
            expected_vjp = transfer.transfer.pullback(dual)
            differential_defect = max(
                float(jnp.max(jnp.abs(jvp - expected_jvp))),
                float(jnp.max(jnp.abs(vjp(dual)[0] - expected_vjp))),
            )
            adjoint_defect = float(
                jnp.abs(jnp.vdot(mapped, dual) - jnp.vdot(state, expected_vjp))
            )
            if differential_defect > 2.0e-11 or adjoint_defect > 2.0e-11 * max(
                1.0, float(jnp.sum(jnp.abs(mapped)))
            ):
                raise qualification_failure(
                    "_family_allfield_transfer",
                    (f"Family {kind}/{name} JVP/VJP or adjoint action failed."),
                    {
                        "adjoint_defect": adjoint_defect,
                        "differential_defect": differential_defect,
                        "mapped": mapped,
                    },
                )
            parts.append(mapped)
            field_evidence.append(
                {
                    "field_name": name,
                    "transfer_id": transfer.transfer_id,
                    "defects": dict(transfer.evidence.defects),
                    "semantics": transfer.semantics,
                    "jvp_vjp_defect": differential_defect,
                    "adjoint_defect": adjoint_defect,
                }
            )
            field_evidence[-1].update(
                {
                    "source_field_space_id": source[kind]
                    .field_spaces[index]
                    .field_space_id,
                    "target_field_space_id": target[kind]
                    .field_spaces[index]
                    .field_space_id,
                }
            )
        carried[kind] = parts[0] if len(parts) == 1 else jnp.stack(parts, axis=-1)
        physical = None
        if kind == "DG":
            physical = float(
                np.max(
                    np.abs(
                        _family_scalar_content(target[kind], np.asarray(carried[kind]))
                        - _family_scalar_content(source[kind], np.asarray(values))
                    )
                )
            )
        elif kind == "Hdiv":
            physical = float(
                np.max(
                    np.abs(
                        layer_core_divergence_content(
                            target[kind], np.asarray(carried[kind])
                        )
                        - layer_core_divergence_content(source[kind], np.asarray(values))
                    )
                )
            )
        elif kind == "Hcurl" and not refine:
            physical = layer_core_edge_circulation_defect(
                source[kind],
                target[kind],
                np.asarray(values),
                np.asarray(carried[kind]),
                adaptation,
            )
        if physical is not None and physical > 3.0e-11:
            raise qualification_failure(
                "_family_allfield_transfer",
                (f"Family {kind} physical continuation defect {physical}."),
                {"physical": physical},
            )
        evidence[kind] = {
            "fields": field_evidence,
            "physical_defect": physical,
            "components": int(values.shape[1]),
        }

    first = _family_fv(adaptation.source, budget=budget) if source_fv is None else source_fv
    last = _family_fv(adaptation.target, budget=budget) if target_fv is None else target_fv
    remap = prepare_layer_core_remap(
        first,
        last,
        adaptation,
        "family-refinement" if refine else "family-coarsening",
        plans=LayerCoreLifecyclePlans(budget=budget),
    )
    carried["material_content"] = jax.block_until_ready(
        remap.apply_content(fields["material_content"])
    )
    old_inventory = _family_material_inventory(
        adaptation.source, fields["material_content"], budget=budget, fv=first
    )
    new_inventory = _family_material_inventory(
        adaptation.target, carried["material_content"], budget=budget, fv=last
    )
    defect = float(np.max(np.abs(np.asarray(new_inventory) - np.asarray(old_inventory))))
    if defect > 1.0e-13 * max(
        1.0, float(np.sum(np.abs(np.asarray(fields["material_content"]))))
    ):
        raise qualification_failure(
            "_family_allfield_transfer",
            (f"Family material FV remap lost a region inventory: {defect}."),
            {"defect": defect, "fields": fields},
        )
    evidence["material_content"] = {
        "region_inventory_defect": defect,
        "source": old_inventory,
        "target": new_inventory,
    }
    return carried, evidence


def _family_material_inventory(
    result: Any, content: Any, *, budget: CoordinateEnclosureBudget | None = None,
    fv: phx.discretization.UnstructuredFiniteVolumeDiscretization | None = None,
) -> list[list[float]]:

    if fv is None:
        fv = _family_fv(result, budget=budget)
    request = result.certification.request
    if request.cell_regions is None and result.mesh.topological_dimension == 2:
        labels = np.zeros(fv.cell_count, dtype=np.int64)
        region_count = 1
    else:
        regions = layer_core_cell_regions(result)
        labels = np.asarray([regions[int(gid)] for gid in np.asarray(fv.cell_global_ids)])
        region_count = len(request.domain.region_ids)
    values = np.asarray(content)
    component_count = (
        1
        if result.mesh.topological_dimension != result.mesh.ambient_dimension
        else result.mesh.ambient_dimension + 2
    )
    if values.shape != (labels.size, component_count) or not np.all(np.isfinite(values)):
        raise qualification_failure(
            "_family_material_inventory",
            ("Family FV state lost declared material/component roles."),
            {"labels": labels, "result": result, "values": values},
        )
    return [
        [
            float(np.sum(values[labels == region, component]))
            for component in range(values.shape[1])
        ]
        for region in range(region_count)
    ]


def _family_allfield_inventory(
    result: Any,
    fields: dict[str, Any],
    *,
    spaces: dict[str, Any],
    budget: CoordinateEnclosureBudget | None = None,
    fv: phx.discretization.UnstructuredFiniteVolumeDiscretization | None = None,
) -> dict[str, Any]:
    """Reconstruct restored scalar/form state with mapped family elements."""

    if set(fields) != {*spaces, "material_content"}:
        raise qualification_failure(
            "_family_allfield_inventory",
            ("Family accepted history lost a declared scientific field."),
            {"fields": fields, "spaces": spaces},
        )
    inventory = {}
    for kind, space in spaces.items():
        values = np.asarray(fields[kind])
        if values.shape != (space.dof_maps[0].global_dof_count, 2) or not np.all(
            np.isfinite(values)
        ):
            raise qualification_failure(
                "_family_allfield_inventory",
                (f"Family {kind} state lost its declared two-component layout."),
                {"space": space, "values": values},
            )
        for index, owner in enumerate(space.field_spaces):
            owner.vector_space.validate(fields[kind][:, index])
        content = (
            _family_scalar_content(space, values)
            if kind in ("H1", "DG")
            else layer_core_divergence_content(space, values)
            if kind == "Hdiv"
            else None
        )
        inventory[kind] = {
            "dofs": values.shape[0],
            "components": 2,
            "field_space_ids": [owner.field_space_id for owner in space.field_spaces],
            "physical_content": None if content is None else content.tolist(),
        }
    inventory["material_content"] = {
        "region_inventories": _family_material_inventory(
            result, fields["material_content"], budget=budget, fv=fv
        )
    }
    return inventory


def _family_accepted_epoch(
    result: phx.meshing.CellMeshingResult,
    space: phx.discretization.FiniteElementDiscretization,
    problem: phx.equations.CompiledFiniteElementProblem,
    values: jax.Array,
    fields: dict[str, jax.Array],
    epoch: int,
    transition: phx.meshing.MeshAdaptationResult | None,
    *,
    field_spaces: dict[str, Any],
    budget: CoordinateEnclosureBudget | None = None,
    fv: phx.discretization.UnstructuredFiniteVolumeDiscretization | None = None,
) -> phx.lifecycle.MeshingAcceptedEpoch:
    """Bind every primitive array to its physical owner."""

    spaces = {"Q1": space, **(field_spaces if fields else {})}
    try:
        bindings = prepare_family_finite_element_bindings(
            result, spaces, problem, values, fields, epoch
        )
    except FamilyPreparationError as error:
        raise qualification_failure(
            error.field, error.expectation, error.values
        ) from error
    declarations, arrays = bindings.declarations, bindings.arrays
    if fields:
        volume = prepare_family_finite_volume_bindings(
            result,
            _family_fv(result, budget=budget) if fv is None else fv,
            fields["material_content"],
            epoch,
            scalar_quadrature=result.mesh.topological_dimension
            != result.mesh.ambient_dimension,
        )
        declarations += volume.declarations
        arrays.update(volume.arrays)
    return phx.lifecycle.MeshingAcceptedEpoch(
        result, declarations, arrays, transition=transition
    )


def _family_epoch_state(epoch: Any) -> tuple[Any, Any, dict[str, Any]]:
    fields = {}
    for kind in ("H1", "DG", "Hcurl", "Hdiv"):
        entry = f"{kind}/current/coefficients"
        if entry in epoch.fields:
            fields[kind] = jnp.stack(
                (epoch.fields[entry], epoch.fields[f"{kind}/history/coefficients"]),
                axis=-1,
            )
    if "FV/state/content" in epoch.fields:
        fields["material_content"] = epoch.fields["FV/state/content"]
    return epoch.carrier, epoch.fields["Q1/u/coefficients"], fields


from dataclasses import replace
from itertools import product

from examples.native_multiblock_meshing import (
    manufactured_solution,
    manufactured_source,
    rectangle,
)
from examples.native_tetrahedral_meshing import complex_with_cavity
from phydrax import array_tree_fingerprint, ArrayArchiveLimits, canonical_fingerprint
from phydrax.geometry import (
    ImplicitBoundarySource,
    MappedDomainBoundarySource,
    ParametricCurveBoundarySource,
    SurfaceNativeRestrictionBoundarySource,
    SurfaceSourceRootAtlas,
)
from phydrax.geometry.brep import LineCurve
from phydrax.lifecycle import (
    MeshingFiniteVolumeFieldDeclaration,
    prepare_meshing_field_owner,
    read_meshing_source_closure,
    read_meshing_source_execution_controls,
    write_meshing_source_closure,
)
from phydrax.lifecycle._meshing_field_records import (
    _prepare_field_dynamics,
    rebind_meshing_accepted_epoch_execution,
    rebind_meshing_adaptation_execution_endpoints,
)
from phydrax.meshing import (
    certify_meshing_acceptance,
    declared_plc_domain,
    generate_structured_block,
    SweepControl,
    SweepMapKind,
)
from phydrax.meshing.qualification import (
    curved_occupied_source,
    layer_core_cell_regions,
    layer_core_divergence_content,
    layer_core_edge_circulation_defect,
    LAYER_CORE_FIELD_KINDS,
    layer_core_initial_values,
    layer_core_scalar_content,
    LayerCoreLifecyclePlans,
    prepare_layer_core_remap,
    source_quarter_extrusion,
)
from phydrax.meshing.qualification._native_lifecycle import (
    execute_native_consumer,
    execute_native_request,
)
from tools.meshing_qualification._controls import (
    _family_qualification_requirements,
    MeshingLimitControls,
    with_meshing_limits,
)
from tools.meshing_qualification._physics import (
    _compiled_solver_stages,
    _manufactured_source,
    _manufactured_value,
    CorpusCompilerEvidence,
)
from tools.meshing_qualification._records import _phase_recordings, _result_identity
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


@final
class _FamilyEpochInputs(StrictModule):
    request: phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec
    authority: phx.meshing.MeshCertificationInputs
    audit_policy: phx.meshing.CellMeshAuditPolicy
    association_transfer: FamilyAssociationTransfer
    args: QualificationControls
    case: dict[str, Any]
    record: dict[str, Any]
    ledger: StageLedger
    started: float


@final
class _FamilyEpochOwners(StrictModule):
    """Actual private numerical owners of one accepted carrier, never a ledger."""

    field_spaces: dict[str, Any]
    solver: tuple[Any, Any, Any]
    fv: phx.discretization.UnstructuredFiniteVolumeDiscretization


def _family_prepare_epoch_owners(
    inputs: _FamilyEpochInputs,
    result: phx.meshing.CellMeshingResult,
    epoch: int,
    *,
    budget: CoordinateEnclosureBudget,
) -> _FamilyEpochOwners:
    ledger, stages = inputs.ledger, inputs.record["stages_seconds"]
    label = (
        "original_allfield_owner_preparation" if epoch == 0
        else f"target_allfield_owner_preparation_epoch_{epoch - 1}"
    )
    field_spaces, measured = ledger.run(
        label, lambda: _measured(lambda: _family_allfield_spaces(result, budget=budget))
    )
    stages[label] = measured["wall_seconds"]
    fv, measured = ledger.run(
        f"finite_volume_owner_preparation_epoch_{epoch}",
        lambda: _measured(lambda: _family_fv(result, budget=budget)),
    )
    stages[inputs.record["active_phase"]] = measured["wall_seconds"]
    solver, measured = ledger.run(
        f"solver_prepare_epoch_{epoch}",
        lambda: _measured(lambda: _family_qualification_solver(result, inputs.case, budget=budget)),
    )
    stages[inputs.record["active_phase"]] = measured["wall_seconds"]
    return _FamilyEpochOwners(field_spaces, solver, fv)


@final
class _FamilySolvedEpoch(StrictModule):
    space: phx.discretization.FiniteElementDiscretization
    problem: phx.equations.CompiledFiniteElementProblem
    values: jax.Array
    physical: dict[str, float]
    compiler: CorpusCompilerEvidence
    accepted: phx.lifecycle.MeshingAcceptedEpoch


@final
class _FamilyContinuation(StrictModule):
    result: phx.meshing.CellMeshingResult
    carried: jax.Array
    fields: dict[str, jax.Array]
    adaptation: phx.meshing.MeshAdaptationResult
    owners: _FamilyEpochOwners


def _family_solve_epoch(
    inputs: _FamilyEpochInputs,
    result: phx.meshing.CellMeshingResult,
    carried: jax.Array | None,
    fields: dict[str, jax.Array],
    accepted_transition: phx.meshing.MeshAdaptationResult | None,
    epoch: int,
    owners: _FamilyEpochOwners,
    *,
    budget: CoordinateEnclosureBudget,
) -> _FamilySolvedEpoch:
    """Solve from the accepted seed, check physical error, and bind durable state."""
    case, args, record = (inputs.case, inputs.args, inputs.record)
    _ledger, stages = inputs.ledger, inputs.record["stages_seconds"]
    field_spaces = owners.field_spaces
    space, problem, free = owners.solver
    seed = (
        jnp.zeros((free.size,), dtype=result.mesh.coordinates.dtype)
        if carried is None
        else jnp.asarray(carried)[free]
    )

    values, compiler, timings = _ledger.run(
        f"solver_execution_epoch_{epoch}",
        lambda: _compiled_solver_stages(problem, seed, args.repeats),
    )
    stages.update({f"{key}_epoch_{epoch}": value for key, value in timings.items()})
    physical, measured = _ledger.run(
        f"physical_error_epoch_{epoch}",
        lambda: _measured(lambda: _family_qualification_physical(space, values, case)),
    )
    stages[f"physical_error_epoch_{epoch}"] = measured["wall_seconds"]
    if not np.isfinite(physical["l2_error"]) or physical["l2_error"] > args.target_error:
        raise qualification_failure(
            "_family_qualification_epochs",
            (
                "An accepted family epoch missed the physical PDE error target: "
                f"l2_error={physical['l2_error']!r}, target_error={args.target_error!r}."
            ),
            {"args": args, "physical": physical},
        )

    accepted_epoch, measured = _ledger.run(
        f"canonical_field_epoch_binding_epoch_{epoch}",
        lambda: _measured(
            lambda: _family_accepted_epoch(
                result,
                space,
                problem,
                values,
                fields,
                epoch,
                accepted_transition,
                field_spaces=field_spaces,
                budget=budget,
                fv=owners.fv,
            )
        ),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    return _FamilySolvedEpoch(
        space=space,
        problem=problem,
        values=values,
        physical=physical,
        compiler=compiler,
        accepted=accepted_epoch,
    )


def _family_epoch_evidence(
    inputs: _FamilyEpochInputs,
    result: phx.meshing.CellMeshingResult,
    solved: _FamilySolvedEpoch,
    carried: jax.Array | None,
    fields: dict[str, jax.Array],
    epoch: int,
    owners: _FamilyEpochOwners,
    *,
    budget: CoordinateEnclosureBudget,
) -> dict[str, Any]:
    """Record reconstructed all-field and finite-volume evidence for the accepted epoch."""
    case, args, record = (inputs.case, inputs.args, inputs.record)
    field_spaces = owners.field_spaces
    _ledger, stages = inputs.ledger, inputs.record["stages_seconds"]
    space, physical, values, compiler = (
        solved.space,
        solved.physical,
        solved.values,
        solved.compiler,
    )
    started = inputs.started
    if fields:
        inventory, measured = _ledger.run(
            f"allfield_physical_reconstruction_epoch_{epoch}",
            lambda: _measured(
                lambda: _family_allfield_inventory(
                    result, fields, spaces=field_spaces, budget=budget, fv=owners.fv
                )
            ),
        )
        stages[record["active_phase"]] = measured["wall_seconds"]
        record.setdefault("allfield_history", []).append(inventory)
    if case["name"] == "curved-quarter-extrusion-quad":
        fv_evidence, fv_timing = _ledger.run(
            f"embedded_fv_content_epoch_{epoch}",
            lambda: _measured(
                lambda: _family_quad_fv(
                    result, accepted_content=fields["material_content"],
                    budget=budget, fv=owners.fv,
                )
            ),
        )
        stages[record["active_phase"]] = fv_timing["wall_seconds"]
    else:
        fv_evidence, fv_timing = _ledger.run(
            f"source_mapped_fv_epoch_{epoch}",
            lambda: _measured(
                lambda: _family_mapped_fv(
                    result, case, content=fields["material_content"],
                    budget=budget, fv=owners.fv,
                )
            ),
        )
        stages[record["active_phase"]] = fv_timing["wall_seconds"]
        stages[f"source_mapped_fv_prepare_epoch_{epoch}"] = fv_evidence["preparation"][
            "wall_seconds"
        ]
        stages[f"source_mapped_fv_execute_epoch_{epoch}"] = fv_evidence["execution"][
            "wall_seconds"
        ]
    if (
        physical["l2_error"] <= args.target_error
        and record["time_to_target_error_seconds"] is None
    ):
        record["time_to_target_error_seconds"] = perf_counter() - started
    entry = {
        "epoch": epoch,
        "result": _result_identity(result),
        "physical": physical,
        "embedded_fv_content": fv_evidence,
        "compiler": compiler,
        "dofs": space.dof_maps[0].global_dof_count,
        "resource_guarantees": {
            "enforced": list(result.runtime.enforced_limits),
            "unenforced": list(result.runtime.unenforced_limits),
        },
        "seed_origin": "zero"
        if carried is None
        else "actual-transferred-accepted-scalar",
        "transferred_state_physical": None
        if carried is None
        else _family_qualification_physical(space, carried, case),
        "retained_bytes": model_array_bytes(
            (result, space, solved.problem, values, carried)
        ),
    }
    record["epochs"].append(entry)
    record["physical_error"] = physical["l2_error"]
    return entry


def _family_adapt_epoch(
    inputs: _FamilyEpochInputs,
    result: phx.meshing.CellMeshingResult,
    solved: _FamilySolvedEpoch,
    hierarchy: phx.meshing.MixedAdaptationHierarchy | None,
    epoch: int,
) -> phx.meshing.MeshAdaptationResult:
    """Execute every original mark in its genuine new adaptation request."""
    request, record = inputs.request, inputs.record
    _ledger, stages = inputs.ledger, inputs.record["stages_seconds"]
    audit_policy = inputs.audit_policy
    association_transfer = inputs.association_transfer
    refine = epoch % 2 == 0
    identifiers = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in result.mesh.blocks]
    )
    empty = np.empty((0,), dtype=np.int64)
    marks = phx.meshing.MarkedMeshAdaptation(
        identifiers if refine else empty,
        empty if refine else identifiers,
        hierarchy=hierarchy,
    )
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
        limits=request.limits,
        association_transfer=association_transfer,
        audit_policy=audit_policy,
    )

    prepared, measured = _ledger.run(
        f"adaptation_prepare_epoch_{epoch}",
        lambda: _measured(
            lambda: phx.meshing.prepare_mesh_adaptation(result, marks, policy=policy)
        ),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]

    adaptation, measured = _ledger.run(
        f"adaptation_execute_epoch_{epoch}",
        lambda: _measured(lambda: phx.meshing.execute_mesh_adaptation(prepared)),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    if adaptation.status is not phx.meshing.MeshAdaptationStatus.COMPLETE:
        raise qualification_failure(
            "_family_qualification_epochs",
            ("Mandatory family refine/coarsen marks were not all executed."),
            {"adaptation": adaptation},
        )
    return adaptation


def _family_transfer_epoch(
    inputs: _FamilyEpochInputs,
    adaptation: phx.meshing.MeshAdaptationResult,
    solved: _FamilySolvedEpoch,
    fields: dict[str, jax.Array],
    source_owners: _FamilyEpochOwners,
    epoch: int,
    entry: dict[str, Any],
    *,
    budget: CoordinateEnclosureBudget,
) -> _FamilyContinuation:
    """Reanalyze and transfer on the target request's same live allowance."""
    request, case, record = inputs.request, inputs.case, inputs.record
    _ledger, stages = inputs.ledger, inputs.record["stages_seconds"]
    authority = inputs.authority
    result = adaptation.source
    space, values, physical = solved.space, solved.values, solved.physical
    field_spaces = source_owners.field_spaces
    refine = epoch % 2 == 0
    target = adaptation.target

    acceptance, measured = _ledger.run(
        f"independent_acceptance_epoch_{epoch + 1}",
        lambda: _measured(
            lambda: _family_qualification_acceptance(
                target, request, case, authority, budget=budget
            )
        ),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    owners = _family_prepare_epoch_owners(inputs, target, epoch + 1, budget=budget)
    target_spaces = owners.field_spaces
    if fields:
        field_result, measured = _ledger.run(
            f"allfield_transfer_epoch_{epoch}",
            lambda: _measured(
                lambda: _family_allfield_transfer(
                    adaptation,
                    fields,
                    refine=refine,
                    source_spaces=field_spaces,
                    target_spaces=target_spaces,
                    budget=budget,
                    source_fv=source_owners.fv,
                    target_fv=owners.fv,
                )
            ),
        )
        fields, field_evidence = field_result
        stages[record["active_phase"]] = measured["wall_seconds"]
        entry["allfield_transfer"] = field_evidence

    target_space, _, _ = owners.solver

    transfer, measured = _ledger.run(
        f"field_transfer_prepare_epoch_{epoch}",
        lambda: _measured(
            lambda: phx.discretization.prepare_nested_field_transfer(
                space,
                target_space,
                adaptation.parent_cells if refine else None,
                field_name="u",
                parent_reference_vertices=adaptation.parent_reference_vertices
                if refine
                else None,
                parent_reference_homogeneous=adaptation.parent_reference_homogeneous
                if refine
                else None,
                source_geometry=result.geometry,
                target_geometry=target.geometry,
                geometry_transition=adaptation.geometry_transition,
                coarsening_witnesses=None if refine else adaptation.coarsening_witnesses,
                budget=budget,
            )
        ),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    if not transfer.evidence.passed:
        raise qualification_failure(
            "_family_qualification_epochs",
            ("Family H1 state transfer failed its certificate."),
            {"transfer": transfer},
        )

    carried, measured = _ledger.run(
        f"field_transfer_execute_epoch_{epoch}",
        lambda: _measured(lambda: jax.block_until_ready(transfer.transfer.apply(values))),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    transferred = _family_qualification_physical(target_space, carried, case)
    entry.update(
        {
            "operation": "refine" if refine else "coarsen",
            "target_acceptance": acceptance,
            "adaptation_id": adaptation.result_id,
            "lineage": _family_qualification_lineage(adaptation),
            "transfer_id": transfer.transfer_id,
            "transfer_semantics": transfer.semantics,
            "transfer_defects": dict(transfer.evidence.defects),
            "state_inventory": {
                "source": physical["inventory"],
                "target": transferred["inventory"],
                "defect": transferred["inventory"] - physical["inventory"],
                "conservation_claim": False,
            },
            "adaptation_retained_bytes": model_array_bytes(
                (adaptation, transfer)
            ),
        }
    )
    return _FamilyContinuation(target, carried, fields, adaptation, owners)


@final
class _FamilyPaidEpoch(StrictModule):
    solved: _FamilySolvedEpoch
    fields: dict[str, jax.Array]
    owners: _FamilyEpochOwners
    entry: dict[str, Any]


def _family_consume_epoch(
    inputs: _FamilyEpochInputs,
    result: phx.meshing.CellMeshingResult,
    epoch: int,
    previous: _FamilyPaidEpoch | None,
    adaptation: phx.meshing.MeshAdaptationResult | None,
    *,
    budget: CoordinateEnclosureBudget,
) -> _FamilyPaidEpoch:
    """Perform all actual owners, history, solve and evidence in one paid epoch."""
    if adaptation is None:
        if previous is not None or epoch != 0:
            raise ValueError("Family initial consumer must precede every adaptation")
        owners = _family_prepare_epoch_owners(inputs, result, epoch, budget=budget)
        fields, measured = inputs.ledger.run(
            "original_allfield_history_preparation",
            lambda: _measured(
                lambda: _family_allfield_state(
                    result, spaces=owners.field_spaces, fv=owners.fv, budget=budget
                )
            ),
        )
        inputs.record["stages_seconds"][inputs.record["active_phase"]] = measured["wall_seconds"]
        carried = None
    else:
        if previous is None:
            raise ValueError("Family adaptation consumer must retain its accepted source owners")
        continuation = _family_transfer_epoch(
            inputs, adaptation, previous.solved, previous.fields, previous.owners,
            epoch - 1, previous.entry, budget=budget,
        )
        owners, fields, carried = continuation.owners, continuation.fields, continuation.carried
    solved = _family_solve_epoch(
        inputs, result, carried, fields, adaptation, epoch, owners, budget=budget
    )
    entry = _family_epoch_evidence(
        inputs, result, solved, carried, fields, epoch, owners, budget=budget
    )
    return _FamilyPaidEpoch(solved, fields, owners, entry)


def _family_qualification_epochs(
    result: phx.meshing.CellMeshingResult,
    request: phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec,
    case: dict[str, Any],
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    *,
    deadline: Deadline,
    authority: phx.meshing.MeshCertificationInputs,
    association_transfer: FamilyAssociationTransfer = None,
) -> tuple[
    phx.meshing.CellMeshingResult,
    jax.Array,
    tuple[phx.lifecycle.MeshingAcceptedEpoch, ...],
]:
    """Consume transferred state after both refinement and coarsening."""

    _ledger = StageLedger(record, deadline=deadline)
    represented = bool(result.associations)
    audit_policy = phx.meshing.CellMeshAuditPolicy(
        require_complete_association=represented,
        watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT
        if (represented and case["name"] != "curved-quarter-extrusion-quad")
        or case["dimension"] == 3
        else phx.meshing.CellMeshAuditDisposition.SKIP,
    )
    if audit_policy.policy_id != result.audit.policy_id:
        raise qualification_failure(
            "_family_qualification_epochs",
            ("Family adaptation must preserve the source audit policy."),
            {"audit_policy": audit_policy, "result": result},
        )
    record["source_audit_policy_id"] = audit_policy.policy_id
    inputs = _FamilyEpochInputs(
        request,
        authority,
        audit_policy,
        association_transfer,
        args,
        case,
        record,
        _ledger,
        started,
    )
    hierarchy = None
    adaptation = None
    previous = None
    history = []
    for epoch in range(2 * args.adaptation_rounds + 1):
        deadline.remaining()
        consumed = execute_native_consumer(
            result, request.limits,
            lambda budget: _family_consume_epoch(
                inputs, result, epoch, previous, adaptation, budget=budget
            ),
            failure_factory=qualification_failure,
        )
        result, paid = consumed.result, consumed.value
        sealed_transition = (
            None if adaptation is None
            else rebind_meshing_adaptation_execution_endpoints(
                adaptation, adaptation.source, result
            )
        )
        accepted = rebind_meshing_accepted_epoch_execution(
            paid.solved.accepted, result, transition=sealed_transition
        )
        solved = replace(paid.solved, accepted=accepted)
        paid = replace(paid, solved=solved)
        paid.entry["result"] = _result_identity(result)
        receipt = result.execution_evidence
        if receipt is None:
            raise ValueError("Family paid epoch must publish its actual ended native receipt")
        paid.entry["native_consumer"] = receipt.to_record()
        history.append(accepted)
        previous = paid
        if epoch == 2 * args.adaptation_rounds:
            break
        adaptation = _family_adapt_epoch(inputs, result, solved, hierarchy, epoch)
        result = adaptation.target
        match adaptation.hierarchy:
            case phx.meshing.MixedAdaptationHierarchy() as next_hierarchy:
                hierarchy = next_hierarchy
            case next_hierarchy:
                raise qualification_failure(
                    "family.mixed_hierarchy",
                    "Retain the accepted family sibling records",
                    {"hierarchy": next_hierarchy},
                )
    return result, paid.solved.values, tuple(history)


def _family_oracle_payload(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain independent authored oracle data outside physical epoch roles."""
    domain = case["domain"]
    metadata = {
        "case": {name: value for name, value in case.items() if name != "domain"},
        "domain_id": domain.domain_id,
    }
    if isinstance(domain, phx.geometry.PiecewiseLinearDomain):
        metadata["domain_role"] = "piecewise_linear"
        metadata["region_ids"] = domain.region_ids
        metadata["source_id"] = domain.source_id
        return metadata, {
            "vertices": domain.vertices,
            "facets": domain.facets,
            "facet_regions": domain.facet_regions,
        }
    metadata["domain_role"] = "generation_source_domain"
    return metadata, {}


@final
class _FamilyColdCheckpoint(StrictModule):
    result: phx.meshing.CellMeshingResult
    specification: phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec
    case: dict[str, Any]
    history: tuple[phx.lifecycle.MeshingAcceptedEpoch, ...]
    authority: phx.meshing.MeshCertificationInputs


@final
class _FamilyColdHistory(StrictModule):
    acceptance: list[dict[str, object]]
    physical: list[dict[str, float]]
    embedded_fv: list[dict[str, float]]
    mapped_fv: list[dict[str, Any]]
    allfields: list[dict[str, Any]]
    replay: list[dict[str, Any]]
    owners: _FamilyEpochOwners


@final
class _FamilyColdContinuation(StrictModule):
    physical: dict[str, float]
    fv: dict[str, Any] | None
    compiler: CorpusCompilerEvidence
    timings: dict[str, float]


def _family_restore_checkpoint(
    path: str,
    content_id: str,
    oracle_path: str,
    oracle_content_id: str,
    expected: tuple[str, ...],
    *,
    budget: CoordinateEnclosureBudget,
) -> _FamilyColdCheckpoint:
    """Restore the bounded archive and authenticate source, controls, and accepted history."""

    restored = read_meshing_source_closure(
        Path(path),
        expected_content_id=content_id,
        limits=ArrayArchiveLimits(),
        budget=budget,
    )
    source = restored["generation_source"]
    result = restored["accepted_target"]
    specification = restored["generation_specification"]
    options = restored["generation_options"]
    try:
        case = read_family_oracle(
            oracle_path, oracle_content_id, source, limits=phx.ArrayArchiveLimits()
        )
    except FamilyPreparationError as error:
        raise qualification_failure(
            error.field, error.expectation, error.values
        ) from error
    history = restored["accepted_data"]["accepted_epochs"]
    if (
        not isinstance(result, phx.meshing.CellMeshingResult)
        or not isinstance(case, dict)
        or not isinstance(history, tuple)
        or not history
        or (
            source.binding_id,
            result.result_id,
            specification.specification_id,
            options.options_id,
            canonical_fingerprint(
                array_tree_fingerprint((source, specification, options, case))
            ),
            canonical_fingerprint(array_tree_fingerprint(history)),
        )
        != expected
    ):
        raise qualification_failure(
            "_family_qualification_cold_worker",
            ("Cold family archive changed source, controls or accepted state histories."),
            {
                "array_tree_fingerprint": array_tree_fingerprint,
                "canonical_fingerprint": canonical_fingerprint,
                "case": case,
                "expected": expected,
                "history": history,
                "options": options,
                "result": result,
                "source": source,
                "specification": specification,
            },
        )
    certificate = history[0].carrier.certification
    if certificate is None:
        raise AttributeError(
            f"_family_restore_checkpoint.certificate must contain a request-bearing certificate; got {certificate!r}."
        )
    return _FamilyColdCheckpoint(
        result, specification, case, history, certificate.request
    )


def _family_reanalyze_accepted_history(
    checkpoint: _FamilyColdCheckpoint,
    target_error: float,
    *,
    budget: CoordinateEnclosureBudget,
) -> _FamilyColdHistory:
    """Replay declared owners and transfers, then independently recertify every accepted epoch."""

    history, specification, case, authority = (
        checkpoint.history,
        checkpoint.specification,
        checkpoint.case,
        checkpoint.authority,
    )
    acceptance = []
    physical_history = []
    fv_history = []
    mapped_fv_history = []
    allfield_history = []
    replay_evidence = []
    previous_fields = None
    previous_spaces = None
    previous_fv = None
    last_owners = None
    for epoch in history:
        publication, values, fields = _family_epoch_state(epoch)
        field_spaces = {}
        solver_space = None
        fv_owner = None
        for declaration in epoch.declarations:
            owner, reconstruction = prepare_meshing_field_owner(
                declaration, publication, budget=budget
            )
            if isinstance(owner, phx.discretization.FiniteElementDiscretization):
                kinds = {
                    role.entry_name.partition("/")[0] for role in declaration.state_roles
                }
                if len(kinds) != 1:
                    raise qualification_failure(
                        "_family_qualification_cold_worker",
                        (
                            "Cold family declaration mixes different physical field owners."
                        ),
                        {"declaration": declaration},
                    )
                kind = next(iter(kinds))
                if kind != "Q1":
                    if kind not in LAYER_CORE_FIELD_KINDS or kind in field_spaces:
                        raise qualification_failure(
                            "_family_qualification_cold_worker",
                            (
                                "Cold family declaration has an unknown or duplicate field owner."
                            ),
                            {"declaration": declaration},
                        )
                    field_spaces[kind] = owner
                else:
                    if solver_space is not None:
                        raise ValueError("Cold family history must have exactly one Q1 owner")
                    solver_space = owner
            elif isinstance(owner, phx.discretization.UnstructuredFiniteVolumeDiscretization):
                if fv_owner is not None:
                    raise ValueError("Cold family history must have exactly one FV owner")
                fv_owner = owner
            match declaration:
                case MeshingFiniteVolumeFieldDeclaration() if (
                    declaration.finite_volume_system is not None
                ):
                    dynamics = _prepare_field_dynamics(declaration, owner, reconstruction)
                    rate = jax.block_until_ready(
                        dynamics(
                            jnp.asarray(0.0, dtype=values.dtype),
                            jnp.asarray(epoch.fields["FV/state/coefficients"]),
                        )
                    )
                    if float(jnp.max(jnp.abs(rate))) > 1.0e-10:
                        raise qualification_failure(
                            "_family_qualification_cold_worker",
                            (
                                "Cold declared Euler owner failed carried-state continuation."
                            ),
                            {"rate": rate},
                        )
                case _:
                    continue
        if solver_space is None or fv_owner is None:
            raise ValueError("Cold family history must retain actual Q1 and FV declarations")
        if epoch.transition is not None and fields:
            if not isinstance(epoch.transition, phx.meshing.MeshAdaptationResult):
                raise ValueError("Cold family history must retain its original refine/coarsen transitions")
            if previous_fields is None or previous_spaces is None:
                raise qualification_failure(
                    "_family_qualification_cold_worker",
                    ("Cold family transition lacks its previous accepted field epoch."),
                    {"previous_fields": previous_fields},
                )
            replayed, proof = _family_allfield_transfer(
                epoch.transition,
                previous_fields,
                refine=epoch.transition.coarsening_witnesses is None,
                source_spaces=previous_spaces,
                target_spaces=field_spaces,
                budget=budget,
                source_fv=previous_fv,
                target_fv=fv_owner,
            )
            for name in fields:
                np.testing.assert_allclose(
                    replayed[name], fields[name], rtol=2.0e-11, atol=2.0e-11
                )
            replay_evidence.append(proof)
        previous_fields = fields
        previous_spaces = field_spaces
        previous_fv = fv_owner
        if fields:
            allfield_history.append(
                _family_allfield_inventory(
                    publication, fields, spaces=field_spaces, budget=budget, fv=fv_owner
                )
            )
        acceptance.append(
            _family_qualification_acceptance(
                publication, specification, case, authority, budget=budget
            )
        )
        solver = _family_qualification_solver(
            publication, case, budget=budget, prepared_space=solver_space
        )
        space, epoch_problem, _ = solver
        last_owners = _FamilyEpochOwners(field_spaces, solver, fv_owner)
        mask = (
            jnp.zeros(values.shape, dtype=jnp.bool_)
            .at[epoch_problem.dirichlet_rows[0]]
            .set(True)
        )
        if not np.array_equal(
            np.asarray(mask), np.asarray(epoch.fields["Q1/u/boundary-mask"])
        ) or not np.array_equal(
            np.asarray(epoch_problem.lift),
            np.asarray(epoch.fields["Q1/u/boundary-values"]),
        ):
            raise qualification_failure(
                "_family_qualification_cold_worker",
                ("Cold Q1 source PDE changed its essential mask or boundary lift."),
                {"epoch": epoch, "epoch_problem": epoch_problem, "mask": mask},
            )
        physical = _family_qualification_physical(space, values, case)
        if not np.isfinite(physical["l2_error"]) or physical["l2_error"] > target_error:
            raise qualification_failure(
                "_family_qualification_cold_worker",
                (
                    "Cold family accepted history missed the physical PDE error target: "
                    f"l2_error={physical['l2_error']!r}, target_error={target_error!r}."
                ),
                {"physical": physical, "target_error": target_error},
            )
        physical_history.append(physical)
        if case["name"] == "curved-quarter-extrusion-quad":
            fv_history.append(
                _family_quad_fv(
                    publication,
                    accepted_content=fields["material_content"],
                    budget=budget,
                    fv=fv_owner,
                )
            )
        else:
            mapped_fv_history.append(
                _family_mapped_fv(
                    publication, case, content=fields["material_content"], budget=budget,
                    fv=fv_owner,
                )
            )
    if last_owners is None:
        raise ValueError("Cold family history must include a physically reanalyzed epoch")
    return _FamilyColdHistory(
        acceptance,
        physical_history,
        fv_history,
        mapped_fv_history,
        allfield_history,
        replay_evidence,
        last_owners,
    )


def _family_continue_cold_checkpoint(
    checkpoint: _FamilyColdCheckpoint,
    repeats: int,
    target_error: float,
    *,
    budget: CoordinateEnclosureBudget,
    owners: _FamilyEpochOwners,
) -> _FamilyColdContinuation:
    """Continue the final accepted scalar state and evaluate its independent PDE and FV evidence."""

    history, result, case = checkpoint.history, checkpoint.result, checkpoint.case
    if history[-1].carrier.result_id != result.result_id:
        raise qualification_failure(
            "_family_qualification_cold_worker",
            ("Cold family final state does not belong to its archived publication."),
            {"history": history, "result": result},
        )
    space, problem, free = owners.solver
    _, last_values, last_fields = _family_epoch_state(history[-1])
    values, compiler, timings = _compiled_solver_stages(
        problem, jnp.asarray(last_values)[free], repeats
    )
    physical = _family_qualification_physical(space, values, case)
    if not np.isfinite(physical["l2_error"]) or physical["l2_error"] > target_error:
        raise qualification_failure(
            "_family_qualification_cold_worker",
            ("Cold family continued PDE missed its physical error target."),
            {"physical": physical, "target_error": target_error},
        )
    fv = (
        _family_quad_fv(
            result, accepted_content=last_fields["material_content"], budget=budget,
            fv=owners.fv,
        )
        if case["name"] == "curved-quarter-extrusion-quad"
        else _family_mapped_fv(
            result, case, content=last_fields["material_content"], budget=budget,
            fv=owners.fv,
        )
    )
    return _FamilyColdContinuation(physical, fv, compiler, timings)


def _family_qualification_cold_worker(
    path: str,
    content_id: str,
    oracle_path: str,
    oracle_content_id: str,
    expected: tuple[str, ...],
    repeats: int,
    target_error: float,
    budget: CoordinateEnclosureBudget,
) -> tuple[phx.meshing.CellMeshingResult, ColdWorkerReport]:
    """Fresh-process source recertification of every accepted epoch and PDE."""

    started = perf_counter()
    checkpoint = _family_restore_checkpoint(
        path, content_id, oracle_path, oracle_content_id, expected, budget=budget
    )
    read_seconds = perf_counter() - started
    analyzed = _family_reanalyze_accepted_history(
        checkpoint, target_error, budget=budget
    )
    continued = _family_continue_cold_checkpoint(
        checkpoint, repeats, target_error, budget=budget, owners=analyzed.owners
    )
    acceptance, physical_history, fv_history = (
        analyzed.acceptance,
        analyzed.physical,
        analyzed.embedded_fv,
    )
    mapped_fv_history, allfield_history, replay_evidence = (
        analyzed.mapped_fv,
        analyzed.allfields,
        analyzed.replay,
    )
    physical, fv, compiler, timings = (
        continued.physical,
        continued.fv,
        continued.compiler,
        continued.timings,
    )
    history = checkpoint.history
    return checkpoint.result, ColdWorkerReport(
        record={
            "passed": True,
            "runtime": _runtime_identity(),
            "independent_oracle_content_id": oracle_content_id,
            "epoch_acceptance": acceptance,
            "accepted_history_physical": physical_history,
            "accepted_history_embedded_fv": fv_history,
            "accepted_history_mapped_fv": mapped_fv_history,
            "cold_native_field_transfer_replay": replay_evidence,
            "accepted_history_allfields": allfield_history,
            "accepted_fields_bitwise_preserved": True,
            "prepared_fe_caches_serialized": False,
            "continued_physical": physical,
            "continued_fv": fv,
            "compiler": compiler,
            "solver_seconds": timings,
            "field_names": [
                role.entry_name
                for declaration in history[-1].declarations
                for role in declaration.state_roles
                if role.role == "coefficients"
            ],
            "history_epochs": len(history),
            "peak_process_resident_bytes": peak_resident_bytes(),
        },
        read_seconds=read_seconds,
    )


def _family_qualification_cold_reanalysis(
    source: Any,
    result: Any,
    specification: Any,
    options: Any,
    case: dict[str, Any],
    history: tuple[Any, ...],
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    *,
    deadline: Deadline,
) -> None:

    _ledger = StageLedger(record, deadline=deadline)

    limits = ArrayArchiveLimits()
    records = {
        "certification_inputs": result.certification.request,
        "report": result.certification,
        "associations": result.associations,
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": options,
        "accepted_target": result,
        "accepted_data": {"accepted_epochs": history},
    }
    with (
        TemporaryDirectory(prefix="phydrax-native-family-source-") as directory,
        prepare_cold_worker(
            "tools.meshing_qualification.scenarios.family",
            deadline=deadline,
            ledger=StageLedger(
                record,
                deadline=deadline,
                cold_phase="fresh_process_source_allfield_accepted_history_continued_PDE",
            ),
        ) as cold_worker,
    ):
        oracle_metadata, oracle_arrays = _family_oracle_payload(case)
        oracle_content_id = family_oracle_content_id(oracle_metadata, oracle_arrays)

        oracle_path, oracle_timing = _ledger.run(
            "independent_authored_source_oracle_publication",
            lambda: _measured(
                lambda: write_family_oracle(
                    Path(directory) / "family.oracle.zip",
                    metadata=oracle_metadata,
                    arrays=oracle_arrays,
                    limits=limits,
                )
            ),
        )
        record["stages_seconds"][record["active_phase"]] = oracle_timing["wall_seconds"]

        receipt, timing = _ledger.run(
            "allfield_accepted_history_checkpoint_publication",
            lambda: _measured(
                lambda: write_meshing_source_closure(
                    Path(directory) / "family.checkpoint.zip", records, limits=limits
                )
            ),
        )
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
        expected = (
            source.binding_id,
            result.result_id,
            specification.specification_id,
            options.options_id,
            canonical_fingerprint(
                array_tree_fingerprint((source, specification, options, case))
            ),
            canonical_fingerprint(array_tree_fingerprint(history)),
        )
        # The scenario module owns its cold consumer.
        remaining = deadline.expires_at - perf_counter()
        if remaining <= 0.0:
            raise TimeoutError(
                f"_family_qualification_cold_reanalysis.remaining must be positive before cold continued solve; got {remaining!r}."
            )

        cold_started = perf_counter()
        try:
            cold = cold_worker.execute(
                receipt.path,
                receipt.content_id,
                {
                    "oracle_path": str(oracle_path),
                    "oracle_content_id": oracle_content_id,
                    "expected": list(expected),
                    "repeats": args.repeats,
                    "target_error": args.target_error,
                },
            )
            timing = {"wall_seconds": perf_counter() - cold_started}
        except ColdWorkerTimeout as error:
            raise TimeoutError(
                f"_family_qualification_cold_reanalysis.remaining must cover the cold process wall time; got {remaining!r}."
            ) from error
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
        # Worker return-code evidence is retained by StageLedger.
        # Parsed and process-bound by ColdWorker.execute.
        if (
            not isinstance(cold, dict)
            or cold.get("passed") is not True
            or cold.get("process_id") == os.getpid()
        ):
            raise qualification_failure(
                "_family_qualification_cold_reanalysis",
                ("Family cold certificate did not originate in an independent process."),
                {"cold": cold},
            )
        record["checkpoint"] = {
            **cold,
            "content_id": receipt.content_id,
            "container_bytes": receipt.path.stat().st_size,
            "independent_oracle_container_bytes": oracle_path.stat().st_size,
            "archive_limits": {
                "members": limits.max_members,
                "nesting": limits.max_manifest_nesting,
                "rank": limits.max_array_rank,
            },
            "source_authority_stripped": False,
        }


def _family_qualification_resource_refusals(
    source: Any, request: Any, options: Any, result: Any
) -> list[dict[str, object]]:
    """Lower one observed allowance, retaining every other authored control."""

    original = source.binding_id, array_tree_fingerprint(source)
    outcomes = []
    for name, value in (
        ("maximum_cells", sum(block.cell_count for block in result.mesh.blocks) - 1),
        ("maximum_geometry_queries", 1),
        ("maximum_scratch_bytes", 1),
        ("maximum_work_units", 1),
    ):
        limit_controls = MeshingLimitControls.from_limits(request.limits)
        limit_controls = replace(limit_controls, **{name: value})
        bounded = limit_controls.to_limits()
        reduced = with_meshing_limits(request, bounded)
        try:
            phx.meshing.NativeMeshingProvider(options).plan(
                source, reduced, coordinate_contract=result.coordinate_contract
            ).execute()
        except phx.meshing.MeshingFailure as failure:
            if (
                failure.category
                is not phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
            ):
                raise qualification_failure(
                    "_family_qualification_resource_refusals",
                    ("Controlled family budget refusal has the wrong scientific status."),
                    {"failure": failure},
                ) from failure
            outcomes.append(
                {
                    "control": name,
                    "limit": value,
                    "original_limits_id": request.limits.limits_id,
                    "refusal_limits_id": bounded.limits_id,
                    **FailureRecord.from_exception(
                        failure, failure.evidence.stage
                    ).to_record(),
                }
            )
        else:
            raise qualification_failure(
                "_family_qualification_resource_refusals",
                ("A controlled family request bypassed its resource allowance."),
                {},
            )
        if (source.binding_id, array_tree_fingerprint(source)) != original:
            raise qualification_failure(
                "_family_qualification_resource_refusals",
                ("Controlled family refusal mutated the source."),
                {
                    "array_tree_fingerprint": array_tree_fingerprint,
                    "original": original,
                    "source": source,
                },
            )
    return outcomes


@final
class _FamilyLifecycleStart(StrictModule):
    record: dict[str, Any]
    started: float
    deadline: Deadline
    phases: list[NativeMeshingPhaseMeasurement]


@final
class _FamilyPreparedPublication(StrictModule):
    source: phx.meshing.qualification.FamilySource
    request: phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec
    options: phx.meshing.NativeMeshingOptions
    case: dict[str, Any]
    plan: phx.meshing.NativeMeshingPlan


@final
class _FamilyPublication(StrictModule):
    result: phx.meshing.CellMeshingResult
    association_transfer: FamilyAssociationTransfer
    authority: phx.meshing.MeshCertificationInputs


def _family_lifecycle_start(args: QualificationControls) -> _FamilyLifecycleStart:
    """Validate the frozen controls and initialize the bounded lifecycle evidence."""

    requirements = _family_qualification_requirements()
    controls = requirements["controls"]
    if (
        args.resolution < 2
        or args.capacity < 1
        or args.repeats < 1
        or args.adaptation_rounds < 1
        or not np.isfinite(args.timeout)
        or args.timeout <= 0.0
        or not np.isfinite(args.target_error)
        or args.target_error <= 0.0
    ):
        raise ValueError(
            f"_family_lifecycle_start.bounds must have resolution at least two, positive capacity, repeats, finite timeout and target error, and at least one adaptation round; got {(args.resolution, args.capacity, args.repeats, args.adaptation_rounds, args.timeout, args.target_error)!r}."
        )
    started = perf_counter()
    phases: list[NativeMeshingPhaseMeasurement] = []
    record: dict[str, Any] = {
        "scenario": "native-family-lifecycle",
        "status": "failed",
        "case": args.family_case,
        "runtime": _runtime_identity(),
        "requirements": requirements,
        "frozen_controls": {
            "resolution": args.resolution,
            "capacity": args.capacity,
            "adaptation_rounds": args.adaptation_rounds,
            **controls,
        },
        "targets": {"l2_error": args.target_error, "wall_seconds": args.timeout},
        "epochs": [],
        "stages_seconds": {},
        "time_to_target_error_seconds": None,
        "mandatory_workflow_completion": "incomplete",
        "case_contract_completion": "incomplete",
        "independent_scientific_certification": "unassessed",
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
        "nonclaims": requirements["nonclaims"],
        "active_phase": "source_preparation",
    }
    return _FamilyLifecycleStart(
        record, started, Deadline(started + args.timeout), phases
    )


def _family_prepare_publication(
    args: QualificationControls,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    *,
    deadline: Deadline,
) -> _FamilyPreparedPublication:
    """Prepare authored source controls and the admitted native publication plan."""

    _ledger, stages = StageLedger(record, deadline=deadline), record["stages_seconds"]
    fixture, measured = _ledger.run(
        record.get("active_phase", "_family_qualification_fixture"),
        lambda: _measured(lambda: _family_qualification_fixture(args)),
    )
    source, request, options, case = fixture
    if case["dimension"] == 3:
        case["manufactured_consumer_fields"] = {
            "u": "original Q1 manufactured diffusion, actual accepted solution",
            "H1": ("Q2", "1+y+z", ".5+.125*y+.1*z"),
            "DG": ("DG1", "1+y+z", ".5+.125*y+.1*z"),
            "Hcurl": (
                "trimmed order2 circulation",
                "grad(1+y+z)",
                "grad(.5+.125*y**2+.1*z**2)",
            ),
            "Hdiv": (
                "trimmed order2 flux",
                "grad(1+y+z)",
                "grad(.5+.125*y**2+.1*z**2)",
            ),
            "material_content": (
                "original stationary Euler",
                "rho=1",
                "velocity=0",
                "pressure=1",
                "all conserved components; original region labels",
            ),
        }
    else:
        embedded = case["name"] == "curved-quarter-extrusion-quad"
        case["manufactured_consumer_fields"] = {
            "u": "original Q1 manufactured diffusion, actual accepted solution",
            "H1": ("Q2", "1+x", ".5+.125*x")
            if embedded
            else ("Q2", "1+x+y", ".5+.125*x+.1*y"),
            "DG": ("DG1", "1+x", ".5+.125*x")
            if embedded
            else ("DG1", "1+x+y", ".5+.125*x+.1*y"),
            "Hcurl": (
                "trimmed order2 circulation",
                "axial tangent" if embedded else "(1,1)",
                "(.125,0,0)" if embedded else "(.125,.1)",
            ),
            "Hdiv": (
                "trimmed order2 intrinsic flux",
                "axial tangent" if embedded else "(1,1)",
                "(.125,0,0)" if embedded else "(.125,.1)",
            ),
            "material_content": (
                "accepted intrinsic scalar FV content of x**2"
                if embedded
                else "accepted stationary 2D Euler conserved content",
                "conservative refinement/coarsening; no per-epoch reseeding",
            ),
        }
    record["frozen_controls"].update(case.get("frozen_controls", {}))
    stages[record["active_phase"]] = measured["wall_seconds"]
    record.update(
        {
            "source_binding_id": source.binding_id,
            "request_id": request.specification_id,
            "limits_id": request.limits.limits_id,
            "source_revision": source.source_revision,
            "source_family_profile": {
                key: value for key, value in case.items() if key != "domain"
            },
        }
    )

    provider = _ledger.run(
        "geometry_constraint_preparation",
        lambda: phx.meshing.NativeMeshingProvider(options),
    )
    plan, measured = _ledger.run(
        record.get("active_phase", "plan"),
        lambda: _measured(
            lambda: provider.plan(
                source,
                request,
                coordinate_contract=phx.SpatialCoordinateContract.si(),
                record_phase=partial(record_family_native_phase, phases, record),
            )
        ),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    record["prepared_id"] = plan.plan_id
    record["prepared_retained_bytes"] = model_array_bytes(plan)
    return _FamilyPreparedPublication(source, request, options, case, plan)


def _family_execute_publication(
    prepared: _FamilyPreparedPublication,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    *,
    deadline: Deadline,
) -> _FamilyPublication:
    """Execute and independently accept the publication, then bind source association transfer."""

    request, plan = prepared.request, prepared.plan
    _ledger, stages = StageLedger(record, deadline=deadline), record["stages_seconds"]
    result, measured = _ledger.run(
        "generation_publication_certification",
        lambda: _measured(lambda: plan.execute(
            record_phase=partial(record_family_native_phase, phases, record)
        )),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    certification = result.certification
    if certification is None:
        raise qualification_failure(
            "_family_qualification_lifecycle",
            ("Purefamily result omitted source certification."),
            {"certification": certification},
        )

    consumed = execute_native_consumer(
        result, request.limits,
        lambda budget: _family_publication_consumers(
            prepared, result, record, phases, deadline=deadline, budget=budget
        ),
        failure_factory=qualification_failure,
    )
    return _FamilyPublication(consumed.result, consumed.value, certification.request)


def _family_publication_consumers(
    prepared: _FamilyPreparedPublication,
    result: phx.meshing.CellMeshingResult,
    record: dict[str, Any],
    phases: list[NativeMeshingPhaseMeasurement],
    *,
    deadline: Deadline,
    budget: CoordinateEnclosureBudget,
) -> FamilyAssociationTransfer:
    """Charge independent acceptance and actual source transfer to the producer."""
    source, request, case = prepared.source, prepared.request, prepared.case
    certification = result.certification
    if certification is None:
        raise ValueError("Family publication consumers require current source certification")
    scope = current_native_execution_scope()
    if scope is None:
        raise ValueError("Family publication consumers require their actual live native scope")
    ledger, stages = StageLedger(record, deadline=deadline), record["stages_seconds"]
    acceptance, measured = ledger.run(
        "independent_acceptance",
        lambda: _measured(
            lambda: _family_qualification_acceptance(
                result, request, case, certification.request, budget=budget
            )
        ),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    record["initial_acceptance"] = acceptance
    association_transfer, measured = ledger.run(
        "source_association_transfer_preparation",
        lambda: _measured(
            lambda: prepare_family_association_transfer(
                source, result, request, domain=case["domain"], limits=request.limits,
                record_phase=partial(record_family_native_phase, phases, record),
                budget=budget, scope=scope,
            )
        ),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    record["association_transfer_retained_bytes"] = model_array_bytes(association_transfer)
    return association_transfer


def _family_obligation(passed: bool, **evidence: object) -> dict[str, object]:
    return {"passed": bool(passed), **evidence}


def _family_completion_obligations(
    record: dict[str, Any],
    case: dict[str, Any],
    args: QualificationControls,
    accepted_history_count: int,
    elapsed: float,
) -> dict[str, dict[str, object]]:
    """Derive every family workflow obligation from recorded evidence only.

    Completion is the conjunction of these items; reaching the end of the
    lifecycle or a passing status is never sufficient by itself.
    """
    rounds = args.adaptation_rounds
    expected = 2 * rounds + 1
    epochs = record["epochs"]
    transitions = epochs[:-1]
    acceptances = [
        record.get("initial_acceptance"),
        *(entry.get("target_acceptance") for entry in transitions),
    ]
    accepted = [
        isinstance(item, dict) and item.get("passed") is True for item in acceptances
    ]
    accepted_records = [item for item in acceptances if isinstance(item, dict)]
    counts = [
        item.get("actual_family_counts", {}) if isinstance(item, dict) else {}
        for item in acceptances
    ]
    obligation = _family_obligation_class(case)
    checkpoint = record.get("checkpoint") or {}
    refusals = record.get("resource_refusals") or []
    errors = [entry["physical"]["l2_error"] for entry in epochs]
    items: dict[str, dict[str, object]] = {
        "accepted_epochs": _family_obligation(
            len(epochs) == expected and accepted_history_count == expected,
            expected=expected,
            epochs=len(epochs),
            accepted_history=accepted_history_count,
        ),
        "refine_then_coarsen_each_round": _family_obligation(
            [entry.get("operation") for entry in transitions]
            == ["refine", "coarsen"] * rounds
            and all(
                entry.get("adaptation_id") is not None
                and isinstance(entry.get("lineage"), dict)
                for entry in transitions
            ),
            operations=[entry.get("operation") for entry in transitions],
        ),
        "independent_acceptance_each_publication": _family_obligation(
            len(accepted) == expected and all(accepted),
            publications=len(accepted),
        ),
        "global_embedding_coverage_fidelity": _family_obligation(
            all(accepted)
            and all(
                item.get("embedding_id") is not None
                and (
                    item.get("coverage_id") is not None
                    or item.get("original_uv_coverage_certificate_id") is not None
                )
                and item.get("source_fidelity") is not None
                for item in accepted_records
            )
        ),
        "positive_maps_and_frozen_quality": _family_obligation(
            all(accepted)
            and all(
                item["minimum_scaled_jacobian"] > 0.0 and item["size_passed"] is True
                for item in accepted_records
            ),
            minimum_scaled_jacobian=min(
                (item["minimum_scaled_jacobian"] for item in accepted_records),
                default=None,
            ),
        ),
        "physical_error_each_epoch": _family_obligation(
            bool(errors)
            and all(np.isfinite(error) and error <= args.target_error for error in errors)
            and all(
                entry["seed_origin"] == "actual-transferred-accepted-scalar"
                for entry in epochs[1:]
            ),
            errors=errors,
            target=args.target_error,
        ),
        "h1_state_transfer_each_transition": _family_obligation(
            all(entry.get("transfer_id") is not None for entry in transitions)
        ),
        "fv_consumer_each_epoch": _family_obligation(
            all(entry.get("embedded_fv_content") is not None for entry in epochs)
        ),
        "cold_fresh_process_reanalysis": _family_obligation(
            checkpoint.get("passed") is True
            and checkpoint.get("process_id") != os.getpid()
            and checkpoint.get("history_epochs") == expected
            and len(checkpoint.get("epoch_acceptance", ())) == expected
            and checkpoint.get("continued_physical", {}).get("l2_error", np.inf)
            <= args.target_error
            and checkpoint.get("continued_fv") is not None,
            continued_physical=checkpoint.get("continued_physical"),
        ),
        "controlled_resource_refusals": _family_obligation(
            len(refusals) == 4 and len({item["control"] for item in refusals}) == 4,
            controls=[item["control"] for item in refusals],
        ),
        "wall_time": _family_obligation(
            elapsed <= args.timeout, elapsed=elapsed, timeout=args.timeout
        ),
    }
    items["all_field_transfer_each_transition"] = _family_obligation(
        all(entry.get("allfield_transfer") is not None for entry in transitions)
        and len(checkpoint.get("cold_native_field_transfer_replay", ()))
        == len(transitions)
    )
    if obligation == "declared-mixed-family-policy":
        policy = case["family_policy"]
        permitted = {*policy["required"], *policy["allowed_transitions"]}
        items["declared_family_policy"] = _family_obligation(
            all(accepted)
            and all(
                set(policy["required"]) <= set(count) and set(count) <= permitted
                for count in counts
            ),
            actual_family_counts=counts,
        )
    else:
        items["exact_required_family"] = _family_obligation(
            all(accepted)
            and all(
                set(count) == {case["family"]} and count[case["family"]] > 0
                for count in counts
            ),
            actual_family_counts=counts,
        )
    if obligation == "general-nonsweepable-all-hex":
        items["nonsweepable_topology_each_publication"] = _family_obligation(
            all(accepted)
            and all(
                item.get("topology", {}).get("nonsweepability_proved") is True
                and item.get("topology", {}).get("authored_topology_match") is True
                for item in accepted_records
            ),
            witness=case["authored_topology"]["nonsweepable_witness"]["witness"],
        )
    if obligation == "feature-conforming-quad":
        items["feature_conformity_each_publication"] = _family_obligation(
            all(accepted)
            and all(
                item.get("feature_topology", {}).get("authored_feature_match") is True
                for item in accepted_records
            )
        )
    return items


def _family_demanding_controls(
    case: dict[str, Any], args: QualificationControls
) -> list[str]:
    """Report the demanding controls actually present in this authored case."""
    topology = case.get("authored_topology")
    controls = [
        name
        for name, present in (
            ("curved-geometry", case.get("geometry_order", 1) >= 2),
            ("multiple-materials", len(case["expected_regions"]) >= 2),
            (
                "enclosed-cavity",
                topology is not None and topology["boundary"]["enclosed_cavities"] > 0,
            ),
            (
                "nonsweepable",
                topology is not None
                and topology["nonsweepable_witness"]["witness"] is not None,
            ),
            (
                "multiply-connected-feature-loops",
                case.get("authored_boundary", {}).get("holes", 0) > 0,
            ),
            ("refine-coarsen-adaptation", args.adaptation_rounds >= 1),
        )
        if present
    ]
    return controls


def _family_qualification_scale(
    record: dict[str, Any], args: QualificationControls
) -> dict[str, object]:
    """Actual published cell counts per accepted epoch, for routine/larger aggregation."""
    cells = [entry["result"]["cells"] for entry in record["epochs"]]
    initial = record.get("initial_acceptance", {}).get("actual_family_counts", {})
    return {
        "resolution": args.resolution,
        "capacity": args.capacity,
        "initial_cells": sum(initial.values()),
        "epoch_cells": cells,
    }


def _family_qualification_lifecycle(args: QualificationControls) -> dict[str, object]:
    """Bounded source → strict pure family → transferred-state FEM lifecycle."""

    initial = _family_lifecycle_start(args)
    record, started, phases = initial.record, initial.started, initial.phases
    deadline = initial.deadline
    _ledger, stages = StageLedger(record, deadline=deadline), record["stages_seconds"]
    try:
        prepared = _family_prepare_publication(args, record, phases, deadline=deadline)
        source, request, options, case = (
            prepared.source,
            prepared.request,
            prepared.options,
            prepared.case,
        )
        publication = _family_execute_publication(
            prepared, record, phases, deadline=deadline
        )
        result, association_transfer = (
            publication.result,
            publication.association_transfer,
        )
        generation_result = result
        result, values, history = _family_qualification_epochs(
            result,
            request,
            case,
            args,
            record,
            started,
            deadline=deadline,
            authority=publication.authority,
            association_transfer=association_transfer,
        )
        refusals, timing = _ledger.run(
            "original_source_controlled_resource_refusals",
            lambda: _measured(
                lambda: _family_qualification_resource_refusals(
                    source, request, options, generation_result
                )
            ),
        )
        stages[record["active_phase"]] = timing["wall_seconds"]
        record["resource_refusals"] = refusals
        _family_qualification_cold_reanalysis(
            source,
            result,
            request,
            options,
            case,
            history,
            args,
            record,
            started,
            deadline=deadline,
        )
        elapsed = perf_counter() - started
        obligations = _family_completion_obligations(record, case, args, len(history), elapsed)
        complete = all(item["passed"] for item in obligations.values())
        scientific = all(
            item["passed"] for name, item in obligations.items() if name != "wall_time"
        )
        mixed = _family_obligation_class(case) == "declared-mixed-family-policy"
        record.update(
            {
                "status": "timeout"
                if elapsed > args.timeout
                else "passed"
                if scientific
                else "failed",
                "target_error_reached": record["physical_error"] <= args.target_error,
                "independent_scientific_certification": "passed"
                if scientific
                else "failed",
                "completion_obligations": obligations,
                "case_contract_completion": "complete" if complete else "incomplete",
                "mandatory_workflow_completion": "complete" if complete else "incomplete",
                "obligation_class": _family_obligation_class(case),
                "decomposition_provenance": _DECOMPOSITION_PROVENANCE[case["route"]],
                "demanding_controls": _family_demanding_controls(case, args),
                "qualification_scale": _family_qualification_scale(record, args),
                "invariants": {
                    **(
                        {
                            "declared_family_policy": obligations[
                                "declared_family_policy"
                            ]["passed"],
                            "exact_pure_family": False,
                        }
                        if mixed
                        else {
                            "exact_family": obligations["exact_required_family"]["passed"]
                        }
                    ),
                    "original_source_recertified_each_epoch": obligations[
                        "independent_acceptance_each_publication"
                    ]["passed"]
                    and obligations["cold_fresh_process_reanalysis"]["passed"],
                    "all_declared_fields": [
                        role.entry_name
                        for declaration in history[-1].declarations
                        for role in declaration.state_roles
                        if role.role == "coefficients"
                    ],
                    "continued_physics_after_coarsen": obligations[
                        "cold_fresh_process_reanalysis"
                    ]["passed"],
                },
            }
        )
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, record["active_phase"])
    except (RuntimeError, ValueError, TypeError, TimeoutError) as failure:
        record_failure(record, failure, record["active_phase"])
    finally:
        record["native_phases"] = _phase_recordings(phases)
        record["end_to_end_seconds"] = perf_counter() - started
        record["peak_process_resident_bytes"] = peak_resident_bytes()
        record["process_peak_scope"] = PEAK_RESIDENT_SCOPE
        record["device_memory"] = [
            {
                "device_id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
                "memory_statistics": statistics,
                "peak_bytes": None
                if statistics is None
                else statistics.get("peak_bytes_in_use"),
                "status": "unavailable" if statistics is None else "backend-reported",
            }
            for device in jax.local_devices()
            for statistics in (device.memory_stats(),)
        ]
    return record


def _family_build_dual_quad(
    args: QualificationControls, case: dict[str, Any], limits: phx.meshing.MeshingLimits
) -> _FamilyFixtureParts:

    regions: tuple[phx.meshing.RegionControl, ...] = ()
    patches: tuple[phx.meshing.PatchControl, ...] = ()
    holes: tuple[phx.meshing.HoleSeed, ...] = ()
    embedding = (
        phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        )
        if case["dimension"] == 2
        else None
    )
    vertices = np.asarray(
        (
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.8, 0.8),
            (0.8, 1.2),
            (1.2, 1.2),
            (1.2, 0.8),
        ),
        dtype=np.float64,
    )
    loops = ((0, 1, 2, 3), (4, 5, 6, 7))
    region = phx.geometry.PlanarMeshRegion(vertices, loops, feature_id="quad-plate")
    source = phx.meshing.NativePlanarSource(region, "plate-revision")
    facets = tuple(
        (loop[index], loop[(index + 1) % len(loop)])
        for loop in loops
        for index in range(len(loop))
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        vertices,
        np.asarray(facets, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (8, 1)),
        ("plate",),
        source_id=source.source_id,
    )
    expected = {"plate": 3.84}
    case["authored_boundary"] = _family_authored_planar_boundary(vertices, loops)
    scope_ids = np.asarray((0,), dtype=np.int64)
    return _FamilyFixtureParts(
        source, domain, expected, scope_ids, regions, patches, holes, embedding
    )


def _family_build_quarter_extrusion_quad(
    args: QualificationControls, case: dict[str, Any], limits: phx.meshing.MeshingLimits
) -> _FamilyFixtureParts:

    regions: tuple[phx.meshing.RegionControl, ...] = ()
    patches: tuple[phx.meshing.PatchControl, ...] = ()
    holes: tuple[phx.meshing.HoleSeed, ...] = ()
    embedding = (
        phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        )
        if case["dimension"] == 2
        else None
    )

    source = source_quarter_extrusion()
    domain = source.domain
    (patch,) = domain.patches
    # One authored patch: every loop is a source boundary loop of the surface.
    case["authored_boundary"] = {"loops": len(patch.loops)}
    expected = {"original-UV-patch:0": 1.0}
    scope_ids = np.asarray((0,), dtype=np.int64)
    embedding = None
    return _FamilyFixtureParts(
        source, domain, expected, scope_ids, regions, patches, holes, embedding
    )


def _family_build_plc_hex(
    args: QualificationControls, case: dict[str, Any], limits: phx.meshing.MeshingLimits
) -> _FamilyFixtureParts:

    regions: tuple[phx.meshing.RegionControl, ...] = ()
    patches: tuple[phx.meshing.PatchControl, ...] = ()
    holes: tuple[phx.meshing.HoleSeed, ...] = ()
    embedding = (
        phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        )
        if case["dimension"] == 2
        else None
    )
    complex_ = complex_with_cavity()
    source = phx.meshing.NativePlcSource(complex_, "material-cavity", "cavity-revision")
    domain = declared_plc_domain(complex_, source.source_id)
    expected = {"left": 1.0, "right": 0.875}
    if _family_obligation_class(case) == "general-nonsweepable-all-hex":
        case["authored_topology"] = _family_authored_topology(complex_)
    scope_ids = np.arange(complex_.facet_count, dtype=np.int64)
    region_scopes = tuple(
        phx.meshing.MeshingScope(
            source.source_id,
            source.source_revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            3,
            "material-cavity-regions",
            np.asarray((index,), dtype=np.int64),
        )
        for index in range(2)
    )
    regions = tuple(
        phx.meshing.RegionControl(
            region_scopes[index],
            name,
            f"material-{name}",
            phx.meshing.RegionRole.SOLID,
        )
        for index, name in enumerate(("left", "right"))
    )
    interface = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "family-boundary",
        np.asarray((2,), dtype=np.int64),
    )
    cavity = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "family-boundary",
        np.asarray((3,), dtype=np.int64),
    )
    patches = (
        phx.meshing.PatchControl("material-interface", interface, ("left", "right")),
        phx.meshing.PatchControl("cavity-wall", cavity, ("right",)),
    )
    holes = (phx.meshing.HoleSeed(np.asarray((1.5, 0.5, 0.5), dtype=np.float64), cavity),)
    return _FamilyFixtureParts(
        source, domain, expected, scope_ids, regions, patches, holes, embedding
    )


def _family_nonsweepable_plc_source() -> phx.meshing.NativePlcSource:
    """A unit tetrahedral ball with two strictly enclosed disjoint dyadic voids.

    The connected solid has three spherical boundary components, excluding both
    open and closed globally embedded manifold sweep products. No volume mesh,
    sweep, Cartesian decomposition, or mapped reference atlas is authored here.
    All twelve triangles have distinct source facet identities; their normals
    point from material to void, including inward-facing cavity boundaries.
    """
    tetrahedron = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64
    )
    vertices = np.concatenate(
        (
            tetrahedron,
            (2 * tetrahedron + (2, 2, 2)) / 16,
            (2 * tetrahedron + (7, 2, 2)) / 16,
        )
    )
    outward = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    polygons = list(outward)
    for offset in (4, 8):
        polygons.extend(
            (offset + face[2], offset + face[1], offset + face[0]) for face in outward
        )
    complex_ = phx.meshing.PiecewiseLinearComplex(
        vertices,
        polygons,
        np.arange(12, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (12, 1)),
        ("solid",),
    )
    return phx.meshing.NativePlcSource(
        complex_, "qualification-nonsweepable-dual-hex", "dyadic-two-cavities-r1"
    )


def _family_nonsweepable_source_proof(
    complex_: phx.meshing.PiecewiseLinearComplex,
) -> dict[str, Any]:
    """Check exact source premises without using the mesh topology gate.

    A nondegenerate tetrahedron is a closed PL 3-ball. Strict barycentric
    containment and a strict separating plane make the two removed balls
    disjoint interior balls. Their complement in the outer ball is connected
    and has precisely the three checked spherical boundary components.
    Fractions represent the actual binary64 source coordinates, not rounded
    measurements or a tolerance-based proxy for these premises.
    """
    points = tuple(
        tuple(Fraction(float(value)) for value in point) for point in complex_.vertices
    )

    def determinant(
        a: tuple[Fraction, ...],
        b: tuple[Fraction, ...],
        c: tuple[Fraction, ...],
        d: tuple[Fraction, ...],
    ) -> Fraction:
        u, v, w = tuple(
            tuple(point[axis] - a[axis] for axis in range(3)) for point in (b, c, d)
        )
        return (
            u[0] * (v[1] * w[2] - v[2] * w[1])
            - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])
        )

    if (
        len(points) != 12
        or complex_.region_ids != ("solid",)
        or not np.array_equal(complex_.polygon_offsets, np.arange(13) * 3)
        or not np.array_equal(complex_.polygon_facets, np.arange(12))
        or not np.array_equal(complex_.facet_regions, np.tile((-1, 0), (12, 1)))
        or complex_.segments.size
    ):
        raise ValueError(
            "Nonsweepable source must retain its twelve oriented solid/void facets."
        )
    tetrahedra = tuple(points[offset : offset + 4] for offset in (0, 4, 8))
    determinants = tuple(determinant(*tetrahedron) for tetrahedron in tetrahedra)
    if any(value <= 0 for value in determinants):
        raise ValueError(
            "Nonsweepable source tetrahedra must have positive exact volume."
        )
    outer = tetrahedra[0]
    barycentric_minima = []
    for cavity in tetrahedra[1:]:
        weights = []
        for point in cavity:
            for vertex in range(4):
                replaced = list(outer)
                replaced[vertex] = point
                weights.append(determinant(*replaced) / determinants[0])
        barycentric_minima.append(min(weights))
    separation = min(point[0] for point in tetrahedra[2]) - max(
        point[0] for point in tetrahedra[1]
    )
    if min(barycentric_minima) <= 0 or separation <= 0:
        raise ValueError(
            "Nonsweepable cavities must be strictly enclosed and strictly disjoint."
        )
    polygons = np.asarray(complex_.polygon_vertices).reshape(12, 3)
    origin = (Fraction(0),) * 3
    shells = []
    signed_volumes = []
    for shell, offset in enumerate((0, 4, 8)):
        faces = polygons[offset : offset + 4]
        vertices = set(range(offset, offset + 4))
        edges: dict[tuple[int, int], list[int]] = {}
        volume = Fraction(0)
        for face in faces:
            ids = tuple(map(int, face))
            if len(set(ids)) != 3 or not set(ids) < vertices:
                raise ValueError(
                    "Nonsweepable shell faces must bound their own tetrahedron."
                )
            opposite = next(iter(vertices - set(ids)))
            a, b, c = (points[index] for index in ids)
            side = determinant(a, b, c, points[opposite])
            if side >= 0 if shell == 0 else side <= 0:
                raise ValueError(
                    "Nonsweepable boundary orientation must point from material to void."
                )
            volume += determinant(origin, a, b, c) / 6
            for u, v in zip(ids, (*ids[1:], ids[0]), strict=True):
                edges.setdefault((min(u, v), max(u, v)), []).append(1 if u < v else -1)
        euler = len(vertices) - len(edges) + len(faces)
        if euler != 2 or any(
            len(sides) != 2 or sum(sides) != 0 for sides in edges.values()
        ):
            raise ValueError(
                "Nonsweepable boundary shells must be closed oriented tetrahedral spheres."
            )
        expected = determinants[shell] / (6 if shell == 0 else -6)
        if volume != expected:
            raise ValueError(
                "Nonsweepable shell signed volume must match its exact tetrahedron."
            )
        signed_volumes.append(volume)
        shells.append(
            {
                "shell": "outer" if shell == 0 else f"cavity-{shell - 1}",
                "euler_characteristic": euler,
                "signed_volume": str(volume),
                "orientation": "outward" if shell == 0 else "inward",
            }
        )
    solid_volume = sum(signed_volumes, Fraction(0))
    if solid_volume <= 0:
        raise ValueError("Nonsweepable solid must have positive exact volume.")
    return {
        "arithmetic": "exact-rational-of-authored-binary64-coordinates",
        "solid_connected": True,
        "connectedness_basis": "PL-ball-minus-two-disjoint-strictly-interior-PL-balls",
        "strict_cavity_containment": True,
        "minimum_outer_barycentric_coordinates": tuple(map(str, barycentric_minima)),
        "strict_cavity_disjointness": True,
        "separating_axis": 0,
        "separation": str(separation),
        "boundary_components": shells,
        "solid_volume": str(solid_volume),
    }


_NONSWEEPABLE_CAVITY_PATCH_NAMES = ("cavity-wall-0", "cavity-wall-1")


def _family_build_nonsweepable_dual_hex(
    args: QualificationControls, case: dict[str, Any], limits: phx.meshing.MeshingLimits
) -> _FamilyFixtureParts:
    source = _family_nonsweepable_plc_source()
    proof = _family_nonsweepable_source_proof(source.complex)
    case["source_construction_proof"] = proof
    case["authored_topology"] = _family_authored_topology(source.complex)
    domain = declared_plc_domain(source.complex, source.source_id)
    region_scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        3,
        "nonsweepable-solid-regions",
        np.asarray((0,), dtype=np.int64),
    )
    regions = (
        phx.meshing.RegionControl(
            region_scope, "solid", "material-solid", phx.meshing.RegionRole.SOLID
        ),
    )
    scopes = tuple(
        phx.meshing.MeshingScope(
            source.source_id,
            source.source_revision,
            phx.meshing.MeshingEntityKind.GEOMETRY,
            2,
            "family-boundary",
            np.arange(offset, offset + 4, dtype=np.int64),
        )
        for offset in (0, 4, 8)
    )
    patches = tuple(
        phx.meshing.PatchControl(name, scope, ("solid",))
        for name, scope in zip(
            ("outer-wall", *_NONSWEEPABLE_CAVITY_PATCH_NAMES), scopes, strict=True
        )
    )
    holes = tuple(
        phx.meshing.HoleSeed(
            np.asarray(source.complex.vertices)[offset : offset + 4].mean(axis=0),
            scopes[index + 1],
        )
        for index, offset in enumerate((4, 8))
    )
    return _FamilyFixtureParts(
        source,
        domain,
        {"solid": float(Fraction(proof["solid_volume"]))},
        np.arange(source.complex.facet_count, dtype=np.int64),
        regions,
        patches,
        holes,
        None,
    )


def _family_build_frame_sweep_hex(
    args: QualificationControls, case: dict[str, Any], limits: phx.meshing.MeshingLimits
) -> _FamilyFixtureParts:

    n = args.resolution
    regions: tuple[phx.meshing.RegionControl, ...] = ()
    patches: tuple[phx.meshing.PatchControl, ...] = ()
    holes: tuple[phx.meshing.HoleSeed, ...] = ()
    embedding = (
        phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        )
        if case["dimension"] == 2
        else None
    )
    source = prepare_family_frame_sweep_source(n, limits=limits)
    match source.domain:
        case phx.geometry.MappedReferenceDomain() as domain:
            expected = dict(
                zip(
                    domain.region_ids,
                    map(float, domain.exact_region_measures()),
                    strict=True,
                )
            )
            scope_ids = np.asarray(
                domain.reference_mesh.topology.entities(2).entity_ids, dtype=np.int64
            )
        case domain:
            raise qualification_failure(
                "family.frame_sweep_domain",
                "retain the authored mapped reference domain",
                {"source": source, "domain": domain},
            )
    return _FamilyFixtureParts(
        source, domain, expected, scope_ids, regions, patches, holes, embedding
    )


def _family_build_reversed_multiblock_hex(
    args: QualificationControls, case: dict[str, Any], limits: phx.meshing.MeshingLimits
) -> _FamilyFixtureParts:

    n = args.resolution
    regions: tuple[phx.meshing.RegionControl, ...] = ()
    patches: tuple[phx.meshing.PatchControl, ...] = ()
    holes: tuple[phx.meshing.HoleSeed, ...] = ()
    embedding = (
        phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        )
        if case["dimension"] == 2
        else None
    )
    materials = args.family_case == "multiblock-material-hex"
    source, interface_ids = prepare_family_reversed_multiblock_source(n, materials)
    expected = {"left": 1.0, "right": 1.0} if materials else {"solid": 1.0}
    match source.domain:
        case phx.geometry.PiecewiseLinearDomain() as domain:
            scope_ids = np.arange(domain.facets.shape[0], dtype=np.int64)
        case domain:
            raise qualification_failure(
                "family.reversed_multiblock_domain",
                "retain the authored piecewise-linear material domain",
                {"source": source, "domain": domain},
            )
    if materials:
        case["material_split"] = 1.0
        regions = tuple(
            phx.meshing.RegionControl(
                phx.meshing.MeshingScope(
                    source.source_id,
                    source.source_revision,
                    phx.meshing.MeshingEntityKind.GEOMETRY,
                    3,
                    "reversed-source-materials",
                    np.asarray((index,), dtype=np.int64),
                ),
                name,
                f"material-{name}",
                phx.meshing.RegionRole.SOLID,
            )
            for index, name in enumerate(("left", "right"))
        )
        patches = (
            phx.meshing.PatchControl(
                "material-interface",
                phx.meshing.MeshingScope(
                    source.source_id,
                    source.source_revision,
                    phx.meshing.MeshingEntityKind.GEOMETRY,
                    2,
                    "family-boundary",
                    interface_ids,
                ),
                ("left", "right"),
            ),
        )
    return _FamilyFixtureParts(
        source, domain, expected, scope_ids, regions, patches, holes, embedding
    )


def _family_build_annular_quad(
    args: QualificationControls, case: dict[str, Any], limits: phx.meshing.MeshingLimits
) -> _FamilyFixtureParts:

    n = args.resolution
    regions: tuple[phx.meshing.RegionControl, ...] = ()
    patches: tuple[phx.meshing.PatchControl, ...] = ()
    holes: tuple[phx.meshing.HoleSeed, ...] = ()
    embedding = (
        phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        )
        if case["dimension"] == 2
        else None
    )
    source, domain, area = prepare_family_annular_block_source(n)
    expected, scope_ids = (
        {"annulus": area},
        np.arange(domain.facets.shape[0], dtype=np.int64),
    )
    return _FamilyFixtureParts(
        source, domain, expected, scope_ids, regions, patches, holes, embedding
    )


# Independently authored unit root-cell fixtures for the registered mapped sources.
# The routes are generic over any root-chart atlas; these sets are inputs only.
_CAVITY_ROOT_CELLS = tuple(
    (x, y, z) for x, y, z in product(range(3), repeat=3) if (x, y, z) != (1, 1, 1)
)
_AUTHORED_ROOT_CELLS: dict[str, tuple[tuple[tuple[int, int, int], ...], bool]] = {
    "NativeMappedHexSource:curved-six-arm-junction": (
        ((1, 1, 1), (0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)),
        False,
    ),
    "NativeMappedHexSource:curved-enclosed-cavity": (_CAVITY_ROOT_CELLS, False),
    "NativeMappedHexSource:curved-two-material-cavity": (_CAVITY_ROOT_CELLS, True),
}


def _family_build_curved_mapped_hex(
    args: QualificationControls, case: dict[str, Any], limits: phx.meshing.MeshingLimits
) -> _FamilyFixtureParts:

    regions: tuple[phx.meshing.RegionControl, ...] = ()
    patches: tuple[phx.meshing.PatchControl, ...] = ()
    holes: tuple[phx.meshing.HoleSeed, ...] = ()
    embedding = (
        phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        )
        if case["dimension"] == 2
        else None
    )

    occupied, materials = _AUTHORED_ROOT_CELLS[case["source"]]
    source = curved_occupied_source(set(occupied), materials=materials, limits=limits)
    case["authored_topology"] = _family_authored_topology(source.reference.complex)
    domain = source.domain
    expected = dict(
        zip(domain.region_ids, map(float, domain.exact_region_measures()), strict=True)
    )
    root = domain.reference_mesh
    scope_ids = np.asarray(root.topology.entities(2).entity_ids, dtype=np.int64)
    cell_ids = np.asarray(root.topology.entities(3).entity_ids, dtype=np.int64)
    regions = tuple(
        phx.meshing.RegionControl(
            phx.meshing.MeshingScope(
                source.source_id,
                source.source_revision,
                phx.meshing.MeshingEntityKind.GEOMETRY,
                3,
                domain.entity_set_id(3),
                cell_ids[domain.cell_regions == index],
            ),
            name,
            f"material-{name}",
            phx.meshing.RegionRole.SOLID,
        )
        for index, name in enumerate(domain.region_ids)
    )
    return _FamilyFixtureParts(
        source, domain, expected, scope_ids, regions, patches, holes, embedding
    )


def _family_build_structured_quad(
    args: QualificationControls, case: dict[str, Any], limits: phx.meshing.MeshingLimits
) -> _FamilyFixtureParts:

    n = args.resolution
    regions: tuple[phx.meshing.RegionControl, ...] = ()
    patches: tuple[phx.meshing.PatchControl, ...] = ()
    holes: tuple[phx.meshing.HoleSeed, ...] = ()
    embedding = (
        phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        )
        if case["dimension"] == 2
        else None
    )
    curves = (
        LineCurve((0.0, 0.0), (0.0, 1.0)),
        LineCurve((1.0, 0.0), (0.0, 1.0)),
        LineCurve((0.0, 0.0), (1.0, 0.0)),
        LineCurve((0.0, 1.0), (1.0, 0.0)),
    )
    revision = canonical_fingerprint(
        {"kind": "unit-square-boundary", "curves": array_tree_fingerprint(curves)}
    )
    boundary = ParametricCurveBoundarySource(
        curves,
        ((0.0, 1.0),) * 4,
        source_id="unit-square",
        source_revision=revision,
        covering_radius=0.01,
    )
    if args.family_case == "multiblock-quad":
        blocks = (rectangle("left", 0.0, 0.5, n), rectangle("right", 0.5, 1.0, n))
        interfaces = (
            phx.meshing.BlockInterfaceControl("left", 1, "right", 0, (0,), (False,)),
        )
    else:
        blocks, interfaces = (rectangle("square", 0.0, 1.0, n),), ()
    source = phx.meshing.NativeStructuredSource(
        blocks,
        interfaces,
        "unit-square",
        revision,
        fidelity_source=boundary,
        maximum_deviation=0.08,
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (4, 1)),
        ("square",),
        source_id=source.source_id,
    )
    expected, scope_ids = {"square": 1.0}, np.asarray((0,), dtype=np.int64)
    return _FamilyFixtureParts(
        source, domain, expected, scope_ids, regions, patches, holes, embedding
    )


def _family_build_sweep(
    args: QualificationControls, case: dict[str, Any], limits: phx.meshing.MeshingLimits
) -> _FamilyFixtureParts:

    n = args.resolution
    regions: tuple[phx.meshing.RegionControl, ...] = ()
    patches: tuple[phx.meshing.PatchControl, ...] = ()
    holes: tuple[phx.meshing.HoleSeed, ...] = ()
    embedding = (
        phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        )
        if case["dimension"] == 2
        else None
    )
    geometry = phx.geometry.Box(
        (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), feature_id="box"
    ).compile(coordinate_unit=METER)
    query = ImplicitBoundarySource(geometry, source_id="box", spacing=0.025)
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    facets = tuple(
        triangle for a, b, c, d in faces for triangle in ((a, b, c), (a, c, d))
    )
    domain = phx.geometry.PiecewiseLinearDomain(
        vertices,
        np.asarray(facets, dtype=np.int64),
        np.tile(np.asarray((0, -1), dtype=np.int64), (12, 1)),
        ("solid",),
        source_id="box",
    )
    if args.family_case == "sweep-prism":
        axis = np.linspace(0.0, 1.0, n + 1)
        points = np.asarray(tuple((u, v) for u in axis for v in axis))
        triangles = []
        for i in range(n):
            for j in range(n):
                a, b = i * (n + 1) + j, (i + 1) * (n + 1) + j
                triangles.extend(((a, b, b + 1), (a, b + 1, a + 1)))
        profile = phx.discretization.CellMesh(
            points,
            (
                phx.discretization.CellBlock(
                    "profile",
                    "triangle",
                    np.asarray(triangles, dtype=np.int32),
                    global_ids=np.arange(len(triangles), dtype=np.int64) + 500,
                ),
            ),
            vertex_global_ids=np.arange(points.shape[0], dtype=np.int64) + 40,
        )
    else:
        profile = generate_structured_block(rectangle("profile", 0.0, 1.0, n)).mesh
    control = SweepControl(
        SweepMapKind.EXTRUSION,
        phx.meshing.LayerSchedule(np.full(n, 1.0 / n, dtype=np.float64)),
        "box-extrusion",
    )
    source = phx.meshing.NativeSweepSource(
        profile,
        control,
        "box",
        query.source_revision,
        fidelity_source=query,
        maximum_deviation=0.2,
        domain=domain,
    )
    expected, scope_ids = {"solid": 1.0}, np.arange(12, dtype=np.int64)
    return _FamilyFixtureParts(
        source, domain, expected, scope_ids, regions, patches, holes, embedding
    )


def _family_publication_acceptance(
    result: phx.meshing.CellMeshingResult,
    request: phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec,
    case: dict[str, Any],
) -> tuple[dict[str, object], phx.meshing.MeshCertificationReport]:

    certificate = result.certification
    if (
        certificate is None
        or not certificate.passed
        or not result.audit.passed
        or not result.compliance.passed
    ):
        raise qualification_failure(
            "_family_publication_acceptance",
            ("Family publication lacks strict independent certification."),
            {"certificate": certificate, "result": result},
        )
    counts: dict[str, int] = {}
    for block in result.mesh.blocks:
        counts[block.cell_kind] = counts.get(block.cell_kind, 0) + block.cell_count
    family_record: dict[str, object] = {"actual_family_counts": counts}
    if case["name"] == "hex-dominant":
        policy = request.target.cell_families
        permitted = set(
            (*policy.required, *policy.preferred, *policy.allowed_transitions)
        )
        if (
            not policy.allow_mixed
            or not set(policy.required) <= set(counts)
            or not set(counts) <= permitted
            or not any(kind in policy.allowed_transitions for kind in counts)
        ):
            raise qualification_failure(
                "_family_publication_acceptance",
                (
                    "Declared mixed family closure omitted its core/transitions or introduced an undeclared family."
                ),
                {"counts": counts, "permitted": permitted, "policy": policy},
            )
        family_record["declared_family_policy"] = case["family_policy"]
    elif set(counts) != {case["family"]} or counts[case["family"]] <= 0:
        raise qualification_failure(
            "_family_publication_acceptance",
            ("Pure-family request admitted a transition or a different family."),
            {"case": case, "result": result},
        )
    if case["name"] in (
        "dual-quad",
        "dual-hex",
        "nonsweepable-dual-hex",
        "hex-dominant",
        "curved-quarter-extrusion-quad",
    ) and (
        not result.associations
        or any(
            item.source_id != case["source_id"]
            or item.source_revision != case["source_revision"]
            for item in result.associations
        )
    ):
        raise qualification_failure(
            "_family_publication_acceptance",
            ("Family result lost its source/revision associations."),
            {"case": case, "result": result},
        )
    return family_record, certificate


def _family_source_embedding(
    authority: phx.meshing.MeshCertificationInputs,
    result: phx.meshing.CellMeshingResult,
    *,
    budget: CoordinateEnclosureBudget,
) -> phx.geometry.GlobalEmbeddingCertificate | None:
    """Reprove mapped source roots before importing their exact target partition."""
    from phydrax.discretization._cell_geometry_predecessor import (
        prepare_cell_geometry_predecessor,
    )
    from phydrax.geometry._mapped_reference_coverage import (
        certify_mapped_reference_source,
        retained_mapped_reference_source_evidence,
    )

    domain = authority.domain
    if (
        not isinstance(domain, phx.geometry.MappedReferenceDomain)
        or result.geometry.restriction_source is None
    ):
        return None
    limits = (
        phx.geometry.MeshCertificateLimits(maximum_scratch_bytes=budget.maximum_memory_bytes)
        if authority.limits is None
        else authority.limits
    )
    # This is a new source proof, not the publication's embedding verdict.
    # Its actual operand preparation, target reproof and fidelity all consume
    # the same request ledger, including independent cold reconstruction.
    with budget.polynomial_preparation_scope():
        certify_mapped_reference_source(domain, limits, budget=budget)
    evidence = retained_mapped_reference_source_evidence(domain, limits, budget=budget)
    if evidence is None:
        raise qualification_failure(
            "_family_source_acceptance",
            "Independent mapped source roots lack certified reference coverage or physical embedding.",
            {"domain": domain},
        )
    predecessor = prepare_cell_geometry_predecessor(
        evidence.physical_mesh,
        domain.source_geometry,
        evidence.physical_validity,
        evidence.physical_embedding,
        result.mesh,
        result.geometry,
        source_limits=limits,
        budget=budget,
    )
    return phx.geometry.certify_global_embedding(
        result.mesh,
        result.geometry,
        result.audit.validity,
        limits=limits,
        predecessor=predecessor,
        budget=budget,
    )


def _family_source_acceptance(
    authority: phx.meshing.MeshCertificationInputs,
    result: phx.meshing.CellMeshingResult,
    case: dict[str, Any],
    certificate: phx.meshing.MeshCertificationReport,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> _FamilySourceAcceptance:
    """Renew source science with one ledger for root, target and coverage proofs."""
    from phydrax.geometry._embedding_state import coordinate_certificate_request

    limits = (
        (
            phx.geometry.MeshCertificateLimits(maximum_scratch_bytes=budget.maximum_memory_bytes)
            if budget is not None else phx.geometry.MeshCertificateLimits()
        )
        if authority.limits is None
        else authority.limits
    )
    with coordinate_certificate_request(limits, budget=budget) as request_budget:
        return _family_source_acceptance_with_budget(
            authority, result, case, certificate, budget=request_budget
        )


def _family_source_acceptance_with_budget(
    authority: phx.meshing.MeshCertificationInputs,
    result: phx.meshing.CellMeshingResult,
    case: dict[str, Any],
    certificate: phx.meshing.MeshCertificationReport,
    *,
    budget: CoordinateEnclosureBudget,
) -> _FamilySourceAcceptance:
    inputs = certificate.request
    inputs.validate_source_integrity(budget=budget)
    if (
        inputs.domain_id != authority.domain_id
        or inputs.source_id != authority.source_id
        or inputs.source_revision != authority.source_revision
        or inputs.fidelity_tolerance != authority.fidelity_tolerance
    ):
        raise qualification_failure(
            "_family_source_acceptance",
            ("Adapted publication changed the source/domain model."),
            {"authority": authority, "inputs": inputs},
        )
    equality = (
        None if certificate.fidelity is None else certificate.fidelity.boundary_equality
    )
    embedding = _family_source_embedding(authority, result, budget=budget)
    renewed = certify_meshing_acceptance(
        result.mesh,
        result.geometry,
        result.audit,
        schedule=authority.schedule,
        domain=authority.domain,
        cell_regions=inputs.cell_regions,
        source=inputs.source
        if isinstance(
            inputs.source,
            (MappedDomainBoundarySource, SurfaceNativeRestrictionBoundarySource),
        )
        else authority.source,
        fidelity_tolerance=authority.fidelity_tolerance,
        fidelity_sample_order=authority.fidelity_sample_order,
        limits=authority.limits,
        junction_vertices=inputs.junction_vertices,
        prepared=None
        if equality is None and embedding is None
        else phx.meshing.PreparedBoundaryAcceptance(
            embedding=embedding,
            fidelity_predecessor=None if equality is None else equality.predecessor,
        ),
        budget=budget,
    )
    if not renewed.passed or renewed.embedding is None:
        raise qualification_failure(
            "_family_source_acceptance",
            "Source scientific recertification failed: "
            + "; ".join(
                f"{outcome.check} {outcome.status}: {', '.join(outcome.reasons)}"
                for outcome in renewed.failing_outcomes
            ),
            {"renewed": renewed},
        )
    mesh = result.mesh
    centers = np.concatenate(
        [
            np.asarray(mesh.coordinates)[np.asarray(block.vertices)].mean(axis=1)
            for block in mesh.blocks
        ]
    )
    regions = (
        (centers[:, 0] > 1.0).astype(np.int64)
        if case["name"] in ("dual-hex", "hex-dominant", "multiblock-material-hex")
        else (centers[:, 0] > 2.0).astype(np.int64)
        if case["name"] == "curved-material-hex"
        else np.zeros(centers.shape[0], dtype=np.int64)
    )
    if inputs.cell_regions is not None and not np.array_equal(
        regions, np.asarray(inputs.cell_regions)
    ):
        raise qualification_failure(
            "_family_source_acceptance",
            ("Published cell-material assignments disagree with independent geometry."),
            {"inputs": inputs, "regions": regions},
        )
    if case["name"] == "curved-quarter-extrusion-quad":
        current_source, original_source = inputs.source, authority.source
        if not isinstance(
            current_source, SurfaceNativeRestrictionBoundarySource
        ) or not isinstance(original_source, SurfaceNativeRestrictionBoundarySource):
            raise TypeError(
                f"_family_source_acceptance.sources must be SurfaceNativeRestrictionBoundarySource objects; got {(type(current_source).__name__, type(original_source).__name__)!r}."
            )
        current_atlas, original_atlas = (
            current_source.support.original,
            original_source.support.original,
        )
        if not isinstance(current_atlas, SurfaceSourceRootAtlas) or not isinstance(
            original_atlas, SurfaceSourceRootAtlas
        ):
            raise TypeError(
                f"_family_source_acceptance.atlases must be SurfaceSourceRootAtlas objects; got {(type(current_atlas).__name__, type(original_atlas).__name__)!r}."
            )
        if current_atlas.atlas_id != original_atlas.atlas_id:
            raise qualification_failure(
                "_family_source_acceptance",
                ("Curved quad adaptation changed the spline control/knot/weight atlas."),
                {"current_atlas": current_atlas, "original_atlas": original_atlas},
            )
        fidelity = renewed.fidelity
        if (
            fidelity is None
            or fidelity.status != "certified"
            or fidelity.mesh_to_source_upper != 0.0
            or fidelity.source_to_mesh_upper != 0.0
        ):
            raise qualification_failure(
                "_family_source_acceptance",
                ("Whole-map equality and independent coverage of every UV span failed."),
                {"fidelity": fidelity},
            )
        measured = None
        coverage_record = {
            "coverage_id": None,
            "source_facets": None,
            "covered_source_facets": None,
            "coverage_semantics": "whole-original-UV-domain-coverage-in-source-fidelity",
            "original_uv_coverage_certificate_id": fidelity.certificate_id,
        }
    else:
        coverage = phx.geometry.certify_domain_coverage(
            mesh,
            result.geometry,
            case["domain"],
            regions,
            embedding=renewed.embedding,
            limits=authority.limits,
            budget=budget,
        )
        if not isinstance(coverage, phx.geometry.DomainCoverageCertificate):
            raise qualification_failure(
                "_family_source_acceptance",
                (
                    "Family material acceptance requires exact declared-domain coverage, not approximate CAD source coverage."
                ),
                {"coverage": coverage},
            )
        measured = dict(
            zip(coverage.region_ids, coverage.achieved_region_measures, strict=True)
        )
        if (
            coverage.status != "certified"
            or set(measured) != set(case["expected_regions"])
            or any(
                measured[name] is None or abs(measured[name] - expected) > 1.0e-10
                for name, expected in case["expected_regions"].items()
            )
        ):
            raise qualification_failure(
                "_family_source_acceptance",
                ("Independent source/material domain measures failed."),
                {"case": case, "coverage": coverage, "measured": measured},
            )
        coverage_record = {
            "coverage_id": coverage.certificate_id,
            "source_facets": coverage.source_facet_count,
            "covered_source_facets": coverage.covered_source_facet_count,
        }
    return _FamilySourceAcceptance(
        renewed,
        mesh,
        regions,
        coverage_record,
        measured,
        renewed.embedding.certificate_id,
    )


def _family_material_acceptance(
    result: phx.meshing.CellMeshingResult,
    case: dict[str, Any],
    mesh: phx.discretization.CellMesh,
    regions: np.ndarray,
    family_record: dict[str, object],
) -> dict[str, object]:

    material_record: dict[str, object] = {"status": "single-domain"}
    if case["name"] in (
        "dual-hex",
        "hex-dominant",
        "curved-material-hex",
        "nonsweepable-dual-hex",
        "multiblock-material-hex",
    ):
        ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
        zones = {
            zone.name: zone
            for zone in result.zones
            if zone.role is phx.meshing.MeshZoneRole.REGION
        }
        region_names = tuple(case["domain"].region_ids)
        if set(zones) != set(region_names):
            raise qualification_failure(
                "_family_material_acceptance",
                ("Family publication lost a material."),
                {"zones": zones},
            )
        for index, name in enumerate(region_names):
            zone = zones[name]
            if (
                zone.material_id != f"material-{name}"
                or zone.region_role is not phx.meshing.RegionRole.SOLID
                or not np.array_equal(
                    np.sort(np.asarray(zone.scope.entity_ids)),
                    np.sort(ids[regions == index]),
                )
            ):
                raise qualification_failure(
                    "_family_material_acceptance",
                    (
                        "Material zones do not own exactly their independently classified cells."
                    ),
                    {
                        "ids": ids,
                        "index": index,
                        "name": name,
                        "regions": regions,
                        "zone": zone,
                    },
                )
        if case["name"] == "nonsweepable-dual-hex":
            patches = {patch.name: patch for patch in result.patches}
            cavity_names = _NONSWEEPABLE_CAVITY_PATCH_NAMES
            if not all(name in patches for name in cavity_names):
                raise qualification_failure(
                    "_family_material_acceptance",
                    "The two authored cavities must retain distinct source patches.",
                    {"patches": tuple(patches)},
                )
            first, second = (
                np.asarray(patches[name].scope.entity_ids) for name in cavity_names
            )
            if first.size == 0 or second.size == 0 or np.intersect1d(first, second).size:
                raise qualification_failure(
                    "_family_material_acceptance",
                    "Distinct cavity patches must own disjoint, represented face sets.",
                    {"first": first, "second": second},
                )
        if case["name"] in ("dual-hex", "hex-dominant") and not {
            "material-interface",
            "cavity-wall",
        } <= {patch.name for patch in result.patches}:
            raise qualification_failure(
                "_family_material_acceptance",
                ("Family result lost its declared interface or cavity-wall patch."),
                {"case": case, "result": result},
            )
        if case["name"] == "multiblock-material-hex" and "material-interface" not in {
            patch.name for patch in result.patches
        }:
            raise qualification_failure(
                "_family_material_acceptance",
                ("Reversed multiblock publication lost its material interface."),
                {"case": case, "result": result},
            )
        material_record = {
            "status": "exclusive-source-material-coverage",
            "material_ids": [zones[name].material_id for name in region_names],
        }
    if case["name"] == "hex-dominant":
        match mesh.connectivity:
            case phx.discretization.PolyhedralConnectivity() as connectivity:
                pass
            case _:
                raise qualification_failure(
                    "family.material_connectivity",
                    "retain packed mixed-cell face incidence",
                    {"connectivity": mesh.connectivity},
                )
        face_ids = np.asarray(mesh.entity_set(2).entity_ids)
        owners = np.asarray(connectivity.face_owner, dtype=np.int64)
        neighbors = np.asarray(connectivity.face_neighbor, dtype=np.int64)
        incidence = np.asarray(connectivity.face_cell_counts)
        if np.any((incidence != 1) & (incidence != 2)):
            raise qualification_failure(
                "_family_material_acceptance",
                ("Mixed transitions have nonconforming or nonmanifold face incidence."),
                {"incidence": incidence},
            )
        internal = neighbors >= 0
        material_faces = internal & (regions[owners] != regions[np.maximum(neighbors, 0)])
        interface = next(
            patch for patch in result.patches if patch.name == "material-interface"
        )
        if not np.array_equal(
            np.sort(np.asarray(interface.scope.entity_ids)),
            np.sort(face_ids[material_faces]),
        ):
            raise qualification_failure(
                "_family_material_acceptance",
                (
                    "Declared material interface does not cover exactly the conforming mixed-cell transition faces."
                ),
                {
                    "face_ids": face_ids,
                    "interface": interface,
                    "material_faces": material_faces,
                },
            )
        family_record["conforming_material_interface_face_count"] = int(
            np.count_nonzero(material_faces)
        )
    return material_record


# Obligation classes derive from registered route/family fields; no profile names.
_FEATURE_QUAD_ROUTES = frozenset(("planar_dual_quad", "parametric_surface"))
_DECOMPOSITION_PROVENANCE = {
    "plc_dual_hex": "constructed: native constrained PLC tetrahedralization, then the four-hexahedron tetrahedral-dual split of every tetrahedron",
    "plc_hex_dominant": "constructed: tetrahedral-dual hexahedral core plus declared pyramid/tetrahedron transition closure",
    "mapped_frame_grid_hex": "authored root-chart atlas (polycube map); route-constructed source-frame integer-grid extraction",
    "mapped_balanced_grid_hex": "authored root-chart atlas (polycube map); route-constructed balanced 1:8 face-parity grid",
    "planar_dual_quad": "constructed: native constrained planar triangulation, then the three-quadrilateral barycentric dual split",
    "parametric_surface": "constructed: original rational chart triangle decomposition, then quadrilateral extraction",
    "sweep": "declared sweep profile and station schedule",
    "structured_transfinite": "declared structured block topology and interface permutations",
}


def general_all_hex_family_cases() -> tuple[str, ...]:
    """Registered family cases whose positive completion closes the general all-hex gate."""
    profiles = _family_qualification_requirements()["profiles"]
    return tuple(
        sorted(
            name
            for name, profile in profiles.items()
            if _family_obligation_class(dict(profile)) == "general-nonsweepable-all-hex"
        )
    )


def _family_obligation_class(case: dict[str, Any]) -> str:
    """Classify declared obligations, never infer nonsweepability from a route."""
    if "family_policy" in case:
        return "declared-mixed-family-policy"
    if case.get("nonsweepability_required") is True:
        return "general-nonsweepable-all-hex"
    if case["family"] == "quadrilateral" and case["route"] in _FEATURE_QUAD_ROUTES:
        return "feature-conforming-quad"
    if case["route"] == "sweep":
        return "declared-sweep"
    if case["family"] == "hexahedron":
        return "declared-exact-all-hex"
    return "declared-structured-multiblock"


def _family_loop_arrays(
    loops: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.asarray([loop.shape[0] for loop in loops], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    flat = (
        np.concatenate(loops).astype(np.int64)
        if loops
        else np.empty((0,), dtype=np.int64)
    )
    return offsets, flat


def _family_face_components(
    offsets: np.ndarray, flat: np.ndarray
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Label polygon components through shared undirected edges."""
    count = offsets.shape[0] - 1
    if count == 0:
        empty = np.empty((0,), dtype=np.int64)
        return 0, empty, empty, empty, np.empty((0, 2), dtype=np.int64)
    sizes = np.diff(offsets)
    face = np.repeat(np.arange(count, dtype=np.int64), sizes)
    successor = np.arange(flat.shape[0], dtype=np.int64) + 1
    successor[offsets[1:] - 1] = offsets[:-1]
    directed = np.stack((flat, flat[successor]), axis=1)
    edges, edge_index = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
    edge_index = edge_index.reshape(-1)
    graph = coo_matrix(
        (np.ones(face.shape[0]), (face, count + edge_index)),
        shape=(count + edges.shape[0], count + edges.shape[0]),
    )
    components, labels = connected_components(graph, directed=False)
    return int(components), labels[:count], edge_index, successor, directed


def _family_surface_vertex_links(flat: np.ndarray, successor: np.ndarray) -> bool:
    """Every boundary vertex must have one circular link, not pinched sheets."""
    if flat.size == 0:
        return True
    predecessor = np.empty_like(successor)
    predecessor[successor] = np.arange(flat.size, dtype=np.int64)
    endpoints = np.concatenate(
        (
            np.stack((flat, flat[predecessor]), axis=1),
            np.stack((flat, flat[successor]), axis=1),
        ),
        axis=0,
    )
    link_nodes, inverse = np.unique(endpoints, axis=0, return_inverse=True)
    first, second = inverse[: flat.size], inverse[flat.size :]
    degree = np.bincount(inverse, minlength=link_nodes.shape[0])
    graph = coo_matrix(
        (np.ones(flat.size, dtype=np.int8), (first, second)),
        shape=(link_nodes.shape[0], link_nodes.shape[0]),
    )
    _, labels = connected_components(graph, directed=False)
    vertices, vertex_index = np.unique(link_nodes[:, 0], return_inverse=True)
    lower = np.full(vertices.size, link_nodes.shape[0], dtype=np.int64)
    upper = np.full(vertices.size, -1, dtype=np.int64)
    np.minimum.at(lower, vertex_index, labels)
    np.maximum.at(upper, vertex_index, labels)
    return bool(np.all(degree == 2) and np.all(lower == upper))


def _family_surface_topology(
    points: np.ndarray, offsets: np.ndarray, flat: np.ndarray
) -> dict[str, Any]:
    """Classify an outward-oriented closed polygon surface independently of its producer."""
    components, labels, edge_index, successor, directed = _family_face_components(
        offsets, flat
    )
    edge_count = int(edge_index.max(initial=-1)) + 1
    uses = np.bincount(edge_index, minlength=edge_count)
    forward = np.bincount(
        edge_index,
        weights=(directed[:, 0] < directed[:, 1]).astype(np.float64),
        minlength=edge_count,
    )
    manifold = bool(
        np.all(uses == 2)
        and np.all(forward == 1.0)
        and _family_surface_vertex_links(flat, successor)
    )
    face = np.repeat(np.arange(offsets.shape[0] - 1), np.diff(offsets))
    face_label = labels[face]
    edge_label = np.zeros((edge_count,), dtype=np.int64)
    edge_label[edge_index] = face_label
    vertex_pairs = np.unique(np.stack((face_label, flat), axis=1), axis=0)
    vertices = np.bincount(vertex_pairs[:, 0], minlength=components)
    edges = np.bincount(edge_label, minlength=components)
    faces = np.bincount(labels, minlength=components)
    # Fan triangles about each loop's first vertex give the enclosed signed volume.
    first = np.repeat(flat[offsets[:-1]], np.diff(offsets))
    interior = np.ones(flat.shape[0], dtype=np.bool_)
    interior[offsets[:-1]] = False
    interior[offsets[1:] - 1] = False
    a, b, c = (
        points[first[interior]],
        points[flat[interior]],
        points[flat[successor[interior]]],
    )
    volume = np.bincount(
        face_label[interior],
        weights=np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0,
        minlength=components,
    )
    shells = sorted(
        (
            "outer" if volume[index] > 0.0 else "cavity",
            int(vertices[index] - edges[index] + faces[index]),
        )
        for index in range(components)
    )
    return {
        "closed_oriented_manifold": manifold,
        "components": [
            {"shell": shell, "euler_characteristic": euler} for shell, euler in shells
        ],
        "outer_shells": sum(shell == "outer" for shell, _ in shells),
        "enclosed_cavities": sum(shell == "cavity" for shell, _ in shells),
    }


def _family_authored_topology(
    complex_: phx.meshing.PiecewiseLinearComplex,
) -> dict[str, Any]:
    """Derive solid, cavity, interface and nonsweepability facts from the authored PLC."""
    points = np.asarray(complex_.vertices, dtype=np.float64)
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    flat = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    incidence = np.asarray(complex_.facet_regions, dtype=np.int64)[
        np.asarray(complex_.polygon_facets, dtype=np.int64)
    ]
    loops = [flat[offsets[index] : offsets[index + 1]] for index in range(len(incidence))]
    void = incidence < 0
    # The right-hand normal points into incidence[:, 0]; outward means toward void.
    boundary = [
        loop if void[index, 0] else loop[::-1]
        for index, loop in enumerate(loops)
        if void[index, 0] != void[index, 1]
    ]
    interface = [
        loop
        for index, loop in enumerate(loops)
        if not void[index].any() and incidence[index, 0] != incidence[index, 1]
    ]
    topology = _family_surface_topology(points, *_family_loop_arrays(boundary))
    boundary_points = points[np.concatenate(boundary)]
    lower, upper = boundary_points.min(axis=0), boundary_points.max(axis=0)
    excluded = [
        axis
        for axis in range(3)
        if any(
            np.all(points[loop, axis] == points[loop[0], axis])
            and lower[axis] < points[loop[0], axis] < upper[axis]
            for loop in boundary
        )
    ]
    euler = tuple(
        component["euler_characteristic"] for component in topology["components"]
    )
    interval_product_excluded = len(euler) > 2 or (
        len(euler) == 2 and euler[0] != euler[1]
    )
    circle_product_excluded = any(value != 0 for value in euler)
    product_obstruction = (
        topology["closed_oriented_manifold"]
        and interval_product_excluded
        and circle_product_excluded
    )
    return {
        "boundary": topology,
        "interface_components": _family_face_components(*_family_loop_arrays(interface))[
            0
        ],
        "nonsweepable_witness": {
            "axes_with_intermediate_caps": excluded,
            "interval_product_excluded": interval_product_excluded,
            "circle_product_excluded": circle_product_excluded,
            "requires_connected_solid": True,
            "witness": "boundary-component-product-obstruction"
            if product_obstruction
            else None,
            "scope": (
                "For a connected compact surface S, the boundary of S x I is connected "
                "or consists of two homeomorphic closed surfaces. The boundary of "
                "S x S1 is empty or consists only of tori. With independently proved "
                "solid connectivity, these boundary invariants exclude both regular sweeps."
            ),
        },
    }


def _family_published_topology(
    mesh: phx.discretization.CellMesh, regions: np.ndarray
) -> dict[str, Any]:
    """Recompute boundary/interface topology from published hexahedral connectivity only."""
    table = np.asarray(
        phx.discretization.reference_cell_topology("hexahedron").entities[2],
        dtype=np.int64,
    )
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    loops = cells[:, table].reshape((-1, table.shape[1]))
    owner = np.repeat(np.arange(cells.shape[0], dtype=np.int64), table.shape[0])
    _, inverse, counts = np.unique(
        np.sort(loops, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    once = counts[inverse] == 1
    twice = np.flatnonzero(counts[inverse] == 2)
    order = twice[np.argsort(inverse[twice], kind="stable")].reshape((-1, 2))
    different = regions[owner[order[:, 0]]] != regions[owner[order[:, 1]]]
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    boundary = _family_surface_topology(
        points,
        np.arange(0, 4 * (int(np.count_nonzero(once)) + 1), 4, dtype=np.int64),
        loops[once].reshape(-1),
    )
    interface = loops[order[different, 0]]
    adjacent = owner[order]
    cell_graph = coo_matrix(
        (
            np.ones(adjacent.shape[0], dtype=np.int8),
            (adjacent[:, 0], adjacent[:, 1]),
        ),
        shape=(cells.shape[0], cells.shape[0]),
    )
    solid_components = connected_components(
        cell_graph, directed=False, return_labels=False
    )
    return {
        "boundary": boundary,
        "interface_components": _family_face_components(
            np.arange(0, 4 * (interface.shape[0] + 1), 4, dtype=np.int64),
            interface.reshape(-1),
        )[0],
        "nonconforming_faces": int(np.count_nonzero(counts > 2)),
        "solid_components": int(solid_components),
    }


def _family_topology_acceptance(
    mesh: phx.discretization.CellMesh, regions: np.ndarray, case: dict[str, Any]
) -> dict[str, object]:
    """Compare published topology with the independently authored source topology."""
    authored = case["authored_topology"]
    published = _family_published_topology(mesh, regions)
    required = case.get("nonsweepability_required") is True
    construction = case.get("source_construction_proof")
    nonsweepability_proved = (
        authored["nonsweepable_witness"]["witness"] is not None
        and isinstance(construction, dict)
        and construction.get("solid_connected") is True
        and published["solid_components"] == 1
    )
    if (
        not authored["boundary"]["closed_oriented_manifold"]
        or (required and not nonsweepability_proved)
        or not published["boundary"]["closed_oriented_manifold"]
        or published["nonconforming_faces"] != 0
        or published["boundary"] != authored["boundary"]
        or published["interface_components"] != authored["interface_components"]
    ):
        raise qualification_failure(
            "_family_topology_acceptance",
            "Published all-hex topology disagrees with its authored topology or "
            "required product-topology obstruction.",
            {"authored": authored, "published": published},
        )
    return {
        "authored_topology_match": True,
        "nonsweepability_proved": nonsweepability_proved,
        "published_topology": published,
        "nonsweepable_witness": authored["nonsweepable_witness"]["witness"],
    }


def _family_authored_planar_boundary(
    vertices: np.ndarray, loops: tuple[tuple[int, ...], ...]
) -> dict[str, Any]:
    """Authored feature segments and loop orientation of a planar source."""
    areas = [
        0.5
        * float(
            np.sum(
                vertices[list(loop), 0] * np.roll(vertices[list(loop), 1], -1)
                - np.roll(vertices[list(loop), 0], -1) * vertices[list(loop), 1]
            )
        )
        for loop in loops
    ]
    return {
        "loops": len(loops),
        "outer_loops": sum(area > 0.0 for area in areas),
        "holes": sum(area < 0.0 for area in areas),
        "segments": [
            [
                [float(value) for value in vertices[loop[index]]],
                [float(value) for value in vertices[loop[(index + 1) % len(loop)]]],
            ]
            for loop in loops
            for index in range(len(loop))
        ],
    }


def _family_quad_feature_acceptance(
    mesh: phx.discretization.CellMesh, case: dict[str, Any]
) -> dict[str, object]:
    """Recompute closed boundary loops and feature-segment coverage from published quads."""
    authored = case["authored_boundary"]
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    directed = np.stack(
        (cells.reshape(-1), np.roll(cells, -1, axis=1).reshape(-1)), axis=1
    )
    _, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    boundary = directed[counts[inverse] == 1]
    undirected = np.unique(np.sort(directed, axis=1), axis=0)
    vertex_count = int(np.asarray(mesh.coordinates).shape[0])
    valence = np.bincount(undirected.reshape(-1), minlength=vertex_count)
    degree = np.bincount(boundary.reshape(-1), minlength=vertex_count)
    on_boundary = degree > 0
    used = np.zeros((vertex_count,), dtype=np.bool_)
    used[cells.reshape(-1)] = True
    irregular = int(np.count_nonzero(used & ~on_boundary & (valence != 4)))
    graph = coo_matrix(
        (np.ones(boundary.shape[0]), (boundary[:, 0], boundary[:, 1])),
        shape=(vertex_count, vertex_count),
    )
    _, labels = connected_components(graph, directed=False)
    loop_labels = np.unique(labels[on_boundary])
    closed = bool(np.all(degree[on_boundary] == 2))
    record: dict[str, object] = {
        "closed_boundary_loops": closed,
        "boundary_loops": int(loop_labels.shape[0]),
        "irregular_interior_vertices": irregular,
    }
    passed = closed and loop_labels.shape[0] == authored["loops"]
    if "segments" in authored:
        points = np.asarray(mesh.coordinates, dtype=np.float64)
        start, end = points[boundary[:, 0]], points[boundary[:, 1]]
        area = np.bincount(
            labels[boundary[:, 0]],
            weights=0.5 * (start[:, 0] * end[:, 1] - end[:, 0] * start[:, 1]),
            minlength=int(labels.max(initial=-1)) + 1,
        )[loop_labels]
        # The loop areas sum to the signed domain area; normalize the cell orientation.
        area = area if float(np.sum(area)) > 0.0 else -area
        segments = np.asarray(authored["segments"], dtype=np.float64)
        origin, span = segments[:, 0], segments[:, 1] - segments[:, 0]
        lengths = np.linalg.norm(span, axis=1)
        scale = float(np.max(lengths))
        tolerance = 64.0 * np.finfo(np.float64).eps * scale

        def offsets(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            relative = values[:, None, :] - origin[None, :, :]
            cross = (
                span[None, :, 0] * relative[..., 1] - span[None, :, 1] * relative[..., 0]
            ) / lengths[None, :]
            along = np.einsum("psk,sk->ps", relative, span) / lengths[None, :] ** 2
            return cross, along

        cross_start, along_start = offsets(start)
        cross_end, along_end = offsets(end)
        slack = tolerance / scale
        on_segment = (
            (np.abs(cross_start) <= tolerance)
            & (np.abs(cross_end) <= tolerance)
            & (np.minimum(along_start, along_end) >= -slack)
            & (np.maximum(along_start, along_end) <= 1.0 + slack)
        )
        carried = on_segment.sum(axis=1) >= 1
        owner = np.argmax(on_segment, axis=1)
        covered = np.bincount(
            owner[carried],
            weights=np.linalg.norm(end - start, axis=1)[carried],
            minlength=segments.shape[0],
        )
        coverage_defect = float(np.max(np.abs(covered - lengths)))
        record.update(
            {
                "outer_loops": int(np.count_nonzero(area > 0.0)),
                "holes": int(np.count_nonzero(area < 0.0)),
                "boundary_edges_on_authored_features": bool(np.all(carried)),
                "maximum_feature_coverage_defect": coverage_defect,
            }
        )
        passed = (
            passed
            and bool(np.all(carried))
            and coverage_defect <= tolerance * segments.shape[0]
            and record["outer_loops"] == authored["outer_loops"]
            and record["holes"] == authored["holes"]
            and (authored["holes"] > 0 or irregular > 0)
        )
    if not passed:
        raise qualification_failure(
            "_family_quad_feature_acceptance",
            ("Published quadrilaterals do not close onto the authored feature loops."),
            {"authored": authored, "published": record},
        )
    return {**record, "authored_feature_match": True}


def _family_quality_bounds(
    request: phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec,
) -> tuple[float, float, float, float | None]:

    match request.size_controls[0]:
        case phx.meshing.UniformSizeControl() as control:
            maximum_size = control.maximum_size
            if maximum_size is None:
                raise qualification_failure(
                    "family.maximum_size",
                    "retain the authored hard upper bound",
                    {"maximum_size": maximum_size},
                )
            target_size = control.target_size
        case control:
            raise qualification_failure(
                "family.size_control",
                "retain an authored uniform size control",
                {"control": control},
            )
    match request:
        case phx.meshing.SurfaceMeshingSpec():
            target = request.quality_target
            if target is None:
                raise qualification_failure(
                    "family.quality_target",
                    "retain the authored minimum angle",
                    {"quality_target": target},
                )
            minimum_angle = target.minimum_angle
        case phx.meshing.VolumeMeshingSpec():
            minimum_angle = None
        case _:
            raise qualification_failure(
                "family.request",
                "be a surface or volume specification",
                {"request": request},
            )
    return (
        target_size,
        maximum_size,
        request.size_compliance.relative_tolerance,
        minimum_angle,
    )


def _family_quality_acceptance(
    result: phx.meshing.CellMeshingResult,
    request: phx.meshing.SurfaceMeshingSpec | phx.meshing.VolumeMeshingSpec,
    case: dict[str, Any],
    mesh: phx.discretization.CellMesh,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> dict[str, object]:

    try:
        measured = prepare_family_quality_measurements(
            result,
            request,
            profile=case["name"],
            limits=request.limits,
            budget=budget,
        )
    except FamilyPreparationError as error:
        raise qualification_failure(
            error.field, error.expectation, error.values
        ) from error
    values = measured.values
    minimum_scaled, minimum_mean, maximum_aspect = (
        measured.minimum_scaled,
        measured.minimum_mean,
        measured.maximum_aspect,
    )
    lengths, edge_lower, edge_upper = (
        measured.lengths,
        measured.edge_lower,
        measured.edge_upper,
    )
    if case["name"] in ("curved-quarter-extrusion-quad", "curved-frame-sweep-hex"):
        if edge_lower is None or edge_upper is None:
            raise qualification_failure(
                "_family_quality_acceptance",
                "Curved quad physical edge integration is unresolved.",
                {"edge_lower": edge_lower, "edge_upper": edge_upper},
            )
        if case["name"] == "curved-quarter-extrusion-quad" and (
            not np.all(np.isfinite(edge_lower)) or not np.all(np.isfinite(edge_upper))
        ):
            raise qualification_failure(
                "_family_quality_acceptance",
                "Curved quad physical edge integration is unresolved.",
                {"edge_lower": edge_lower, "edge_upper": edge_upper},
            )
    target_size, maximum_size, relative_tolerance, minimum_angle = _family_quality_bounds(
        request
    )
    median, maximum = (float(np.median(lengths)), float(np.max(lengths)))
    size_passed = abs(
        median - target_size
    ) <= relative_tolerance * target_size and maximum <= maximum_size * (
        1.0 + relative_tolerance
    )
    if case["name"] in (
        "curved-quarter-extrusion-quad",
        "curved-frame-sweep-hex",
    ):
        if edge_lower is None or edge_upper is None:
            raise qualification_failure(
                "_family_quality_acceptance.edge_bounds",
                "contain lower and upper edge bounds",
                {"edge_lower": edge_lower, "edge_upper": edge_upper},
            )
        size_passed = size_passed and (
            float(np.median(edge_lower)) >= target_size * (1.0 - relative_tolerance)
            and float(np.median(edge_upper)) <= target_size * (1.0 + relative_tolerance)
            and (float(np.max(edge_upper)) <= maximum_size * (1.0 + relative_tolerance))
        )
    angle_passed = (
        minimum_angle is None
        or float(np.min(np.asarray(values.minimum_angle))) >= minimum_angle
    )
    if (
        not np.all(
            np.isfinite((minimum_scaled, minimum_mean, maximum_aspect, median, maximum))
        )
        or minimum_scaled < 1e-08
        or minimum_mean < 0.01
        or (maximum_aspect > 100.0)
        or (not size_passed)
        or (not angle_passed)
    ):
        raise qualification_failure(
            "_family_quality_acceptance",
            "Frozen independent family quality or hard size budget failed.",
            {
                "angle_passed": angle_passed,
                "maximum": maximum,
                "maximum_aspect": maximum_aspect,
                "median": median,
                "minimum_mean": minimum_mean,
                "minimum_scaled": minimum_scaled,
                "size_passed": size_passed,
            },
        )
    return {
        "minimum_scaled_jacobian": minimum_scaled,
        "minimum_mean_ratio": minimum_mean,
        "maximum_aspect_ratio": maximum_aspect,
        "median_edge": median,
        "maximum_edge": maximum,
        "size_passed": size_passed,
    }


def _cold_worker(packet: ColdControlPacket) -> ColdWorkerReport:
    """Run the scenario-owned cold consumer on its one control packet."""
    controls = packet.controls
    started = perf_counter()
    raw = read_meshing_source_execution_controls(
        packet.archive_path,
        expected_content_id=packet.content_id,
        limits=ArrayArchiveLimits(),
    )
    control_read_seconds = perf_counter() - started
    requested = execute_native_request(
        raw.execution_evidence,
        raw.limits,
        lambda budget: _family_qualification_cold_worker(
            str(packet.archive_path),
            packet.content_id,
            str(controls["oracle_path"]),
            str(controls["oracle_content_id"]),
            tuple(controls["expected"]),
            int(controls["repeats"]),
            float(controls["target_error"]),
            budget,
        ),
        target=lambda value: value[0],
        failure_factory=qualification_failure,
        workflow_remaining_seconds=packet.deadline.remaining(),
    )
    _, report = requested.value
    receipt = requested.result.execution_evidence
    if receipt is None:
        raise RuntimeError(
            "Family cold continuation must retain its complete request receipt."
        )
    return ColdWorkerReport(
        record={
            **report.record,
            "cold_request_result": _result_identity(requested.result),
            "cold_request": {
                "history_owner_id": raw.execution_evidence.owner_id,
                "owner_id": receipt.owner_id,
                "work_units": int(np.asarray(receipt.total_work_units)),
                "geometry_queries": int(np.asarray(receipt.total_geometry_queries)),
                "receipt_fingerprint": array_tree_fingerprint(receipt),
                "limits_id": raw.limits.limits_id,
            },
        },
        read_seconds=control_read_seconds + report.read_seconds,
    )


def cold_worker(packet: ColdControlPacket) -> ColdWorkerReport:
    return run_with_family_progress(
        {"scenario": "native-family-lifecycle", "role": "cold"},
        lambda: _cold_worker(packet), cold=True,
    )


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return run_with_family_progress(
        args.to_record(), lambda: _family_qualification_lifecycle(args)
    )
