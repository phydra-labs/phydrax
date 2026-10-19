# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
import os
import resource
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import model_array_bytes, StrictModule
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
) -> dict[str, object]:
    """Certify publication, source coverage, material topology, and quality independently."""
    family_record, certificate = _family_publication_acceptance(result, request, case)
    source = _family_source_acceptance(authority, result, case, certificate)
    renewed, mesh, regions = source.renewed, source.mesh, source.regions
    coverage_record, measured, embedding_id = (
        source.coverage_record,
        source.measured,
        source.embedding_id,
    )
    material_record = _family_material_acceptance(
        result, case, mesh, regions, family_record
    )
    quality_record = _family_quality_acceptance(result, request, case, mesh)
    return {
        "passed": True,
        "certification_id": renewed.report_id,
        **family_record,
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
    result: Any, case: dict[str, Any]
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
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
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
    return {"l2_error": float(np.sqrt(max(0.0, squared))), "inventory": inventory}


def _family_mapped_fv(
    publication: Any, case: dict[str, Any], *, content: Any = None
) -> dict[str, Any]:
    """Measure mapped FV volumes and the stationary Euler equation on each epoch."""

    dimension = publication.mesh.ambient_dimension
    system = phx.equations.EulerSystem(dimension)
    template = family_finite_volume_field_templates(dimension)[0]
    fv, preparation = _measured(
        lambda: phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
            publication.mesh,
            field_name=template["name"],
            component_names=template["component_names"],
        ).prepare(cell_geometry=publication.geometry)
    )
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
    if content is None:
        primitive = jnp.zeros(
            (fv.cell_count, *template["component_shape"]),
            dtype=publication.mesh.coordinates.dtype,
        )
        primitive = primitive.at[:, 0].set(1.0).at[:, -1].set(1.0)
        state = system.primitive_to_conserved(primitive)
    else:
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
        "seed_origin": "original-stationary-Euler"
        if content is None
        else "actual-transferred-material-content",
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


def _family_quad_fv(publication: Any) -> dict[str, float]:
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
    ).prepare(cell_geometry=original.geometry)
    source_area = float(np.sum(np.asarray(source_fv.cell_volumes)))
    fv = phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        publication.mesh,
        field_name=template["name"],
        component_names=template["component_names"],
    ).prepare(cell_geometry=publication.geometry)
    points = np.asarray(fv.cell_quadrature_points)
    weights = np.where(
        np.asarray(fv.cell_quadrature_valid), np.asarray(fv.cell_quadrature_weights), 0.0
    )
    area = float(np.sum(weights))
    content = float(np.sum(weights * points[..., 0] ** 2))
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


def _family_euler_fv(result: Any) -> Any:
    template = family_finite_volume_field_templates(result.mesh.ambient_dimension)[0]
    return phx.discretization.UnstructuredFiniteVolumePlan.from_cell_mesh(
        result.mesh,
        field_name=template["name"],
        component_names=template["component_names"],
    ).prepare(cell_geometry=result.geometry)


def _family_allfield_spaces(result: Any) -> dict[str, Any]:
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
        ).prepare()
    return spaces


def _family_allfield_state(result: Any) -> dict[str, Any]:

    fields = {}
    for kind, space in _family_allfield_spaces(result).items():
        if kind in ("H1", "DG"):
            fields[kind] = jnp.asarray(layer_core_initial_values(space, kind))
        else:
            # Physical moments of the two scalar/history gradients;
            # no random coefficients or alternate form basis.
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
            fields[kind] = jnp.stack((first, second), axis=-1)
    fv = _family_euler_fv(result)
    system = phx.equations.EulerSystem(result.mesh.ambient_dimension)
    template = family_finite_volume_field_templates(result.mesh.ambient_dimension)[0]
    primitive = jnp.zeros(
        (fv.cell_count, *template["component_shape"]),
        dtype=result.mesh.coordinates.dtype,
    )
    primitive = primitive.at[:, 0].set(1.0).at[:, -1].set(1.0)
    fields["material_content"] = fv.cell_volumes[:, None] * system.primitive_to_conserved(
        primitive
    )
    return fields


def _family_allfield_transfer(
    adaptation: Any,
    fields: dict[str, Any],
    *,
    refine: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Carry both components of every form/scalar role through the epoch."""

    source = _family_allfield_spaces(adaptation.source)
    target = _family_allfield_spaces(adaptation.target)
    carried, evidence = {}, {}
    for kind in source:
        values = fields[kind]
        names = tuple(field.name for field in source[kind].construction_plan.fields)
        parts, field_evidence = [], []
        for index, name in enumerate(names):
            transfer = phx.discretization.prepare_nested_field_transfer(
                source[kind],
                target[kind],
                adaptation.parent_cells if refine else None,
                field_name=name,
                parent_reference_vertices=adaptation.parent_reference_vertices
                if refine
                else None,
                source_geometry=adaptation.source.geometry,
                target_geometry=adaptation.target.geometry,
                geometry_transition=adaptation.geometry_transition,
                coarsening_witnesses=None if refine else adaptation.coarsening_witnesses,
            )
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
                        layer_core_scalar_content(target[kind], np.asarray(carried[kind]))
                        - layer_core_scalar_content(source[kind], np.asarray(values))
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

    first, last = _family_euler_fv(adaptation.source), _family_euler_fv(adaptation.target)
    remap = prepare_layer_core_remap(
        first, last, adaptation, "family-refinement" if refine else "family-coarsening"
    )
    carried["material_content"] = jax.block_until_ready(
        remap.apply_content(fields["material_content"])
    )
    old_inventory = _family_material_inventory(
        adaptation.source, fields["material_content"]
    )
    new_inventory = _family_material_inventory(
        adaptation.target, carried["material_content"]
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


def _family_material_inventory(result: Any, content: Any) -> list[list[float]]:

    fv = _family_euler_fv(result)
    regions = layer_core_cell_regions(result)
    labels = np.asarray([regions[int(gid)] for gid in np.asarray(fv.cell_global_ids)])
    values = np.asarray(content)
    if values.shape != (labels.size, result.mesh.ambient_dimension + 2) or not np.all(
        np.isfinite(values)
    ):
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
        for region in range(len(result.certification.request.domain.region_ids))
    ]


def _family_allfield_inventory(result: Any, fields: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct restored scalar/form state with mapped family elements."""

    spaces = _family_allfield_spaces(result)
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
            layer_core_scalar_content(space, values)
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
            result, fields["material_content"]
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
) -> phx.lifecycle.MeshingAcceptedEpoch:
    """Bind every primitive array to its physical owner."""

    spaces = {"Q1": space, **(_family_allfield_spaces(result) if fields else {})}
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
            result, _family_euler_fv(result), fields["material_content"], epoch
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
    prepare_meshing_field_dynamics,
    prepare_meshing_field_owner,
    read_meshing_source_closure,
    write_meshing_source_closure,
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
    prepare_layer_core_remap,
    source_quarter_extrusion,
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
    ColdWorkerTimeout,
    Deadline,
    peak_resident_bytes,
    qualification_failure,
    run_cold_worker,
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


def _family_solve_epoch(
    inputs: _FamilyEpochInputs,
    result: phx.meshing.CellMeshingResult,
    carried: jax.Array | None,
    fields: dict[str, jax.Array],
    accepted_transition: phx.meshing.MeshAdaptationResult | None,
    epoch: int,
) -> _FamilySolvedEpoch:
    """Solve from the accepted seed, check physical error, and bind durable state."""
    case, args, record = (inputs.case, inputs.args, inputs.record)
    _ledger, stages = inputs.ledger, inputs.record["stages_seconds"]
    prepared_solver, measured = _ledger.run(
        f"solver_prepare_epoch_{epoch}",
        lambda: _measured(lambda: _family_qualification_solver(result, case)),
    )
    space, problem, free = prepared_solver
    stages[record["active_phase"]] = measured["wall_seconds"]
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
        record.get("active_phase", "_family_qualification_physical"),
        lambda: _measured(lambda: _family_qualification_physical(space, values, case)),
    )
    stages[f"physical_error_epoch_{epoch}"] = measured["wall_seconds"]
    if not np.isfinite(physical["l2_error"]) or physical["l2_error"] > args.target_error:
        raise qualification_failure(
            "_family_qualification_epochs",
            ("An accepted family epoch missed the physical PDE error target."),
            {"args": args, "physical": physical},
        )

    accepted_epoch, measured = _ledger.run(
        f"canonical_field_epoch_binding_epoch_{epoch}",
        lambda: _measured(
            lambda: _family_accepted_epoch(
                result, space, problem, values, fields, epoch, accepted_transition
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
) -> dict[str, Any]:
    """Record reconstructed all-field and finite-volume evidence for the accepted epoch."""
    case, args, record = (inputs.case, inputs.args, inputs.record)
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
            lambda: _measured(lambda: _family_allfield_inventory(result, fields)),
        )
        stages[record["active_phase"]] = measured["wall_seconds"]
        record.setdefault("allfield_history", []).append(inventory)
    fv_evidence = None
    if case["name"] == "curved-quarter-extrusion-quad":
        fv_evidence, fv_timing = _ledger.run(
            f"embedded_fv_content_epoch_{epoch}",
            lambda: _measured(lambda: _family_quad_fv(result)),
        )
        stages[record["active_phase"]] = fv_timing["wall_seconds"]
    elif case["dimension"] == 3 or case["route"] in (
        "sweep",
        "structured_transfinite",
    ):
        fv_evidence, fv_timing = _ledger.run(
            f"source_mapped_fv_epoch_{epoch}",
            lambda: _measured(
                lambda: _family_mapped_fv(
                    result, case, content=fields.get("material_content")
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
    fields: dict[str, jax.Array],
    hierarchy: phx.meshing.MixedAdaptationHierarchy | None,
    epoch: int,
    entry: dict[str, Any],
) -> _FamilyContinuation:
    """Accept all required marks and transfer every accepted field to the certified target."""
    request, case, record = (inputs.request, inputs.case, inputs.record)
    _ledger, stages = inputs.ledger, inputs.record["stages_seconds"]
    authority, audit_policy = inputs.authority, inputs.audit_policy
    association_transfer = inputs.association_transfer
    space, values, physical = solved.space, solved.values, solved.physical
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
    target = adaptation.target

    acceptance, measured = _ledger.run(
        f"independent_acceptance_epoch_{epoch + 1}",
        lambda: _measured(
            lambda: _family_qualification_acceptance(target, request, case, authority)
        ),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    if fields:
        field_result, measured = _ledger.run(
            f"allfield_transfer_epoch_{epoch}",
            lambda: _measured(
                lambda: _family_allfield_transfer(adaptation, fields, refine=refine)
            ),
        )
        fields, field_evidence = field_result
        stages[record["active_phase"]] = measured["wall_seconds"]
        entry["allfield_transfer"] = field_evidence

    target_solver, measured = _ledger.run(
        f"target_field_prepare_epoch_{epoch}",
        lambda: _measured(lambda: _family_qualification_solver(target, case)),
    )
    target_space, _, _ = target_solver
    stages[record["active_phase"]] = measured["wall_seconds"]

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
                source_geometry=result.geometry,
                target_geometry=target.geometry,
                geometry_transition=adaptation.geometry_transition,
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
                (prepared, adaptation, transfer)
            ),
        }
    )
    return _FamilyContinuation(target, carried, fields, adaptation)


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
    carried, hierarchy = None, None
    stages = record["stages_seconds"]
    history = []
    accepted_transition = None
    fields = {}
    if case["dimension"] == 3:
        fields, measured = _ledger.run(
            "original_allfield_history_preparation",
            lambda: _measured(lambda: _family_allfield_state(result)),
        )
        stages[record["active_phase"]] = measured["wall_seconds"]
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
    for epoch in range(2 * args.adaptation_rounds + 1):
        if perf_counter() - started >= args.timeout:
            raise TimeoutError(
                f"_family_qualification_epochs.elapsed must be less than the frozen timeout; got {perf_counter() - started!r}."
            )
        solved = _family_solve_epoch(
            inputs, result, carried, fields, accepted_transition, epoch
        )
        values = solved.values
        history.append(solved.accepted)
        entry = _family_epoch_evidence(inputs, result, solved, carried, fields, epoch)
        if epoch == 2 * args.adaptation_rounds:
            break
        continuation = _family_adapt_epoch(
            inputs, result, solved, fields, hierarchy, epoch, entry
        )
        result, carried, fields = (
            continuation.result,
            continuation.carried,
            continuation.fields,
        )
        accepted_transition = continuation.adaptation
        match accepted_transition.hierarchy:
            case phx.meshing.MixedAdaptationHierarchy() as next_hierarchy:
                hierarchy = next_hierarchy
            case next_hierarchy:
                raise qualification_failure(
                    "family.mixed_hierarchy",
                    "Retain the accepted family sibling records",
                    {"hierarchy": next_hierarchy},
                )
    return result, values, tuple(history)


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
) -> _FamilyColdCheckpoint:
    """Restore the bounded archive and authenticate source, controls, and accepted history."""

    restored = read_meshing_source_closure(
        Path(path), expected_content_id=content_id, limits=ArrayArchiveLimits()
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
    checkpoint: _FamilyColdCheckpoint, target_error: float
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
    for epoch in history:
        publication, values, fields = _family_epoch_state(epoch)
        for declaration in epoch.declarations:
            owner, _ = prepare_meshing_field_owner(declaration, publication)
            match declaration:
                case MeshingFiniteVolumeFieldDeclaration():
                    dynamics = prepare_meshing_field_dynamics(declaration, publication)
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
        if epoch.transition is not None and fields:
            if previous_fields is None:
                raise qualification_failure(
                    "_family_qualification_cold_worker",
                    ("Cold family transition lacks its previous accepted field epoch."),
                    {"previous_fields": previous_fields},
                )
            replayed, proof = _family_allfield_transfer(
                epoch.transition,
                previous_fields,
                refine=epoch.transition.coarsening_witnesses is None,
            )
            for name in fields:
                np.testing.assert_allclose(
                    replayed[name], fields[name], rtol=2.0e-11, atol=2.0e-11
                )
            replay_evidence.append(proof)
        previous_fields = fields
        if fields:
            allfield_history.append(_family_allfield_inventory(publication, fields))
        acceptance.append(
            _family_qualification_acceptance(publication, specification, case, authority)
        )
        space, epoch_problem, _ = _family_qualification_solver(publication, case)
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
                ("Cold family accepted history missed the physical PDE error target."),
                {"physical": physical, "target_error": target_error},
            )
        physical_history.append(physical)
        if case["name"] == "curved-quarter-extrusion-quad":
            fv_history.append(_family_quad_fv(publication))
        elif case["dimension"] == 3 or case["route"] in (
            "sweep",
            "structured_transfinite",
        ):
            mapped_fv_history.append(
                _family_mapped_fv(
                    publication, case, content=fields.get("material_content")
                )
            )
    return _FamilyColdHistory(
        acceptance,
        physical_history,
        fv_history,
        mapped_fv_history,
        allfield_history,
        replay_evidence,
    )


def _family_continue_cold_checkpoint(
    checkpoint: _FamilyColdCheckpoint,
    repeats: int,
    target_error: float,
) -> _FamilyColdContinuation:
    """Continue the final accepted scalar state and evaluate its independent PDE and FV evidence."""

    history, result, case = checkpoint.history, checkpoint.result, checkpoint.case
    if history[-1].carrier.result_id != result.result_id:
        raise qualification_failure(
            "_family_qualification_cold_worker",
            ("Cold family final state does not belong to its archived publication."),
            {"history": history, "result": result},
        )
    space, problem, free = _family_qualification_solver(result, case)
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
        _family_quad_fv(result)
        if case["name"] == "curved-quarter-extrusion-quad"
        else _family_mapped_fv(result, case, content=last_fields.get("material_content"))
        if case["dimension"] == 3 or case["route"] in ("sweep", "structured_transfinite")
        else None
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
) -> dict[str, object]:
    """Fresh-process source recertification of every accepted epoch and PDE."""

    checkpoint = _family_restore_checkpoint(
        path, content_id, oracle_path, oracle_content_id, expected
    )
    analyzed = _family_reanalyze_accepted_history(checkpoint, target_error)
    continued = _family_continue_cold_checkpoint(checkpoint, repeats, target_error)
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
    return {
        "passed": True,
        "process_id": os.getpid(),
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
        "peak_process_resident_bytes": peak_resident_bytes(resource.RUSAGE_SELF),
    }


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
    with TemporaryDirectory(prefix="phydrax-native-family-source-") as directory:
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
        # The scenario module owns its worker command.
        remaining = deadline.expires_at - perf_counter()
        if remaining <= 0.0:
            raise TimeoutError(
                f"_family_qualification_cold_reanalysis.remaining must be positive before cold continued solve; got {remaining!r}."
            )

        cold_started = perf_counter()
        try:
            cold = run_cold_worker(
                "scenarios.family",
                (
                    str(receipt.path),
                    receipt.content_id,
                    str(oracle_path),
                    oracle_content_id,
                    json.dumps(expected),
                    str(args.repeats),
                    str(args.target_error),
                ),
                deadline=deadline,
                ledger=StageLedger(
                    record,
                    deadline=deadline,
                    cold_phase="fresh_process_source_allfield_accepted_history_continued_PDE",
                ),
            )
            timing = {"wall_seconds": perf_counter() - cold_started}
        except ColdWorkerTimeout as error:
            raise TimeoutError(
                f"_family_qualification_cold_reanalysis.remaining must cover the cold process wall time; got {remaining!r}."
            ) from error
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
        # Worker return-code evidence is retained by StageLedger.
        # Parsed and process-validated by run_cold_worker.
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
            "H1": ("Q2", "1+y+z", ".5+.125*y**2+.1*z**2"),
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
                record_phase=phases.append,
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

    source, request, case, plan = (
        prepared.source,
        prepared.request,
        prepared.case,
        prepared.plan,
    )
    _ledger, stages = StageLedger(record, deadline=deadline), record["stages_seconds"]
    result, measured = _ledger.run(
        "generation_publication_certification",
        lambda: _measured(lambda: plan.execute(record_phase=phases.append)),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    certification = result.certification
    if certification is None:
        raise qualification_failure(
            "_family_qualification_lifecycle",
            ("Purefamily result omitted source certification."),
            {"certification": certification},
        )

    acceptance, measured = _ledger.run(
        "independent_acceptance",
        lambda: _measured(
            lambda: _family_qualification_acceptance(
                result, request, case, certification.request
            )
        ),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    record["initial_acceptance"] = acceptance

    association_transfer, measured = _ledger.run(
        "source_association_transfer_preparation",
        lambda: _measured(
            lambda: prepare_family_association_transfer(
                source,
                result,
                request,
                domain=case["domain"],
                limits=request.limits,
                record_phase=phases.append,
            )
        ),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    record["association_transfer_retained_bytes"] = model_array_bytes(
        association_transfer
    )
    return _FamilyPublication(result, association_transfer, certification.request)


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
        if case["dimension"] == 3 or case["route"] in ("sweep", "structured_transfinite"):
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
        record.update(
            {
                "status": "passed"
                if record["physical_error"] <= args.target_error
                and elapsed <= args.timeout
                else "timeout"
                if elapsed > args.timeout
                else "failed",
                "target_error_reached": record["physical_error"] <= args.target_error,
                "independent_scientific_certification": "passed",
                "mandatory_workflow_completion": "complete"
                if elapsed <= args.timeout
                else "incomplete",
                "invariants": {
                    **(
                        {"declared_family_policy": True, "exact_pure_family": False}
                        if case["name"] == "hex-dominant"
                        else {"exact_family": True}
                    ),
                    "original_source_recertified_each_epoch": True,
                    "all_declared_fields": [
                        role.entry_name
                        for declaration in history[-1].declarations
                        for role in declaration.state_roles
                        if role.role == "coefficients"
                    ],
                    "continued_physics_after_coarsen": True,
                },
            }
        )
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, record["active_phase"])
        record.update(
            {
                "status": "timeout"
                if failure.category is phx.meshing.MeshingFailureCategory.TIMED_OUT
                else "failed"
            }
        )
    except (RuntimeError, ValueError, TypeError, TimeoutError) as failure:
        record_failure(record, failure, record["active_phase"])
        record.update(
            {"status": "timeout" if isinstance(failure, TimeoutError) else "failed"}
        )
    finally:
        record["native_phases"] = _phase_recordings(phases)
        record["end_to_end_seconds"] = perf_counter() - started
        record["peak_process_resident_bytes"] = peak_resident_bytes(resource.RUSAGE_SELF)
        record["process_peak_scope"] = "process-lifetime"
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

    if args.family_case == "curved-nonsweepable-hex":
        occupied: set[tuple[int, int, int]] = {
            (1, 1, 1),
            (0, 1, 1),
            (2, 1, 1),
            (1, 0, 1),
            (1, 2, 1),
            (1, 1, 0),
            (1, 1, 2),
        }
    else:
        occupied = {(x, y, z) for x, y, z in product(range(3), repeat=3)} - {(1, 1, 1)}
    source = curved_occupied_source(
        occupied, materials=args.family_case == "curved-material-hex"
    )
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
    family_record: dict[str, object] = {}
    if case["name"] == "hex-dominant":
        policy = request.target.cell_families
        counts: dict[str, int] = {}
        for block in result.mesh.blocks:
            counts[block.cell_kind] = counts.get(block.cell_kind, 0) + block.cell_count
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
        family_record = {
            "declared_family_policy": case["family_policy"],
            "actual_family_counts": counts,
        }
    elif any(block.cell_kind != case["family"] for block in result.mesh.blocks):
        raise qualification_failure(
            "_family_publication_acceptance",
            ("Pure-family request admitted a transition or a different family."),
            {"case": case, "result": result},
        )
    if case["name"] in (
        "dual-quad",
        "dual-hex",
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


def _family_source_acceptance(
    authority: phx.meshing.MeshCertificationInputs,
    result: phx.meshing.CellMeshingResult,
    case: dict[str, Any],
    certificate: phx.meshing.MeshCertificationReport,
) -> _FamilySourceAcceptance:

    inputs = certificate.request
    inputs.validate_source_integrity()
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
    )
    if not renewed.passed or renewed.embedding is None:
        raise qualification_failure(
            "_family_source_acceptance",
            ("Source scientific recertification failed."),
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
        "multiblock-material-hex",
    ):
        ids = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
        zones = {
            zone.name: zone
            for zone in result.zones
            if zone.role is phx.meshing.MeshZoneRole.REGION
        }
        if set(zones) != {"left", "right"}:
            raise qualification_failure(
                "_family_material_acceptance",
                ("Family publication lost a material."),
                {"zones": zones},
            )
        for index, name in enumerate(("left", "right")):
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
            "material_ids": [zones[name].material_id for name in ("left", "right")],
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
) -> dict[str, object]:

    try:
        measured = prepare_family_quality_measurements(
            result, request, profile=case["name"], limits=request.limits
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


def worker(argv: tuple[str, ...]) -> dict[str, Any]:
    """Decode process arguments and run the scenario-owned cold consumer."""
    return _family_qualification_cold_worker(
        argv[0],
        argv[1],
        argv[2],
        argv[3],
        tuple(json.loads(argv[4])),
        int(argv[5]),
        float(argv[6]),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "worker":
        raise ValueError(
            "qualification.subcommand must equal worker; got " + repr(sys.argv[1:])
        )
    print(json.dumps(worker(tuple(sys.argv[2:]))))


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return _family_qualification_lifecycle(args)
