# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded preparation of repaired-envelope sources and their rational associations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any, final, Protocol

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx

from ..._differentiation import BranchDifferentiationPolicy
from ..._frozendict import frozendict
from ..._meshcore import NativeExecutionScope
from ..._trainable import NonTrainableState
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._coordinate_budget import CoordinateEnclosureResourceError
from ...lifecycle import (
    CompositionRebind,
    CompositionRebindReceipt,
    MeshingAcceptedEpoch,
    MeshingFieldStateRole,
    MeshingFiniteElementFieldDeclaration,
    prepare_meshing_field_index_binding,
)
from ...typing import checked
from .._association import PlcAssociationTransfer
from .._measurements import NativeExecutionRecord, NativeMeshingPhaseRecorder
from .._surface_envelope import (
    _require_surface_envelope_repair_provenance,
    RawTriangleSoup,
    surface_envelope_source_entities,
    SurfaceEnvelope,
    SurfaceEnvelopePolicy,
    wrap_surface_envelope,
)
from .._volume_generation import (
    declared_plc_domain,
    native_volume_execution_budget,
    prepare_plc_source,
)


class EnvelopePreparationStages(Protocol):
    def run[T](self, name: str, call: Callable[[], T]) -> T:
        """Execute and retain caller-owned stage evidence without changing the result."""
        ...


@final
class EnvelopeSourceRequest(phx.StrictModule):
    source: phx.meshing.NativeSurfaceEnvelopeSource
    specification: phx.meshing.VolumeMeshingSpec
    options: phx.meshing.NativeMeshingOptions


@final
class PreparedEnvelopeSource(phx.StrictModule):
    envelope: SurfaceEnvelope
    request: EnvelopeSourceRequest
    receipt: NativeExecutionRecord


def _prepare_envelope_request(
    envelope: SurfaceEnvelope,
    limits: phx.meshing.MeshingLimits,
) -> EnvelopeSourceRequest:
    """Bind the volume request to the separately repaired source identity."""
    source = phx.meshing.NativeSurfaceEnvelopeSource(envelope, "wrapped-solid")
    scope = phx.meshing.MeshingScope(
        source.source_id,
        source.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        "envelope-facets",
        np.arange(source.plc_source.complex.facet_count, dtype=np.int64),
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 2.0, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        limits=limits,
    )
    options = phx.meshing.NativeMeshingOptions(
        "surface_envelope_tetrahedral",
        volume_schedule=phx.meshing.NativeVolumeSchedule(
            refinement_rounds=1, improvement_passes=1, minimum_dihedral_degrees=0.0
        ),
    )
    return EnvelopeSourceRequest(
        source=source, specification=specification, options=options
    )


@checked
def prepare_envelope_source(
    soup: RawTriangleSoup,
    policy: SurfaceEnvelopePolicy,
    limits: phx.meshing.MeshingLimits,
    /,
    *,
    failure_factory: Callable[[str, str, dict[str, Any]], RuntimeError],
    record_phase: NativeMeshingPhaseRecorder | None = None,
    stages: EnvelopePreparationStages | None = None,
) -> PreparedEnvelopeSource:
    """Wrap and bind a repaired source under one ended root execution record."""
    with native_volume_execution_budget(
        limits, stage=phx.meshing.MeshingStageKind.SOURCE_INSPECTION
    ) as budget:
        if stages is None:
            envelope = wrap_surface_envelope(soup, policy, record_phase=record_phase)
            request = _prepare_envelope_request(envelope, limits)
        else:
            envelope = stages.run(
                "native_envelope_construction",
                lambda: wrap_surface_envelope(soup, policy, record_phase=record_phase),
            )
            request = stages.run(
                "repaired_request_preparation",
                lambda: _prepare_envelope_request(envelope, limits),
            )
    if budget.evidence is None:
        raise failure_factory(
            "envelope.source.execution_evidence",
            "retain the ended root execution record",
            {"budget": budget},
        )
    receipt = NativeExecutionRecord(budget.evidence, owner_id=envelope.envelope_id)
    return PreparedEnvelopeSource(envelope=envelope, request=request, receipt=receipt)


@checked
def prepare_envelope_association_transfer(
    source: phx.meshing.NativeSurfaceEnvelopeSource | phx.meshing.NativePlcSource,
    result: phx.meshing.CellMeshingResult,
    limits: phx.meshing.MeshingLimits,
    /,
) -> PlcAssociationTransfer:
    """Bind rational repaired-carrier identity tables in their source order."""
    with NativeExecutionScope(limits.native_limits()) as scope:
        match source:
            case phx.meshing.NativePlcSource():
                transfer = prepare_plc_source(
                    source.complex,
                    source.source_id,
                    source.source_revision,
                    result.coordinate_contract,
                    limits=limits,
                    scope=scope,
                ).association_transfer
            case phx.meshing.NativeSurfaceEnvelopeSource():
                complex_ = source.plc_source.complex
                triangles, polygons, edges = surface_envelope_source_entities(
                    source, scope=scope
                )
                transfer = PlcAssociationTransfer(
                    phx.meshing.declared_plc_domain(complex_, source.source_id),
                    result.coordinate_contract,
                    source.source_revision,
                    edge_vertices=edges,
                    triangle_vertices=triangles,
                    triangle_facets=complex_.polygon_facets[polygons],
                    facet_regions=complex_.facet_regions,
                    maximum_support_queries=min(
                        limits.maximum_geometry_queries, limits.maximum_work_units
                    ),
                )
            case _:
                raise TypeError(
                    f"qualification.envelope_source must be a PLC or envelope source; got {type(source).__qualname__}"
                )
        transfer.source_associations(result)
    return transfer


@final
class _EnvelopeCertificates(phx.StrictModule):
    certificate: phx.meshing.MeshCertificationReport
    embedding: phx.geometry.GlobalEmbeddingCertificate
    coverage: phx.geometry.DomainCoverageCertificate


@final
class _EnvelopeVolumeClosure(phx.StrictModule):
    reference_volume: float
    volume: float
    volume_defect: float
    measures: np.ndarray
    skin: np.ndarray


def _envelope_certificates(
    source: phx.meshing.NativeSurfaceEnvelopeSource,
    result: phx.meshing.CellMeshingResult,
    *,
    failure_factory: Callable[[str, str, dict[str, object]], RuntimeError],
) -> _EnvelopeCertificates:
    """Require strict repaired-domain certification before consuming repair evidence."""
    certificate = result.certification
    if certificate is None or not certificate.passed:
        raise failure_factory(
            "envelope.acceptance.certification",
            "retain passed native certification",
            {"certificate": certificate},
        )
    embedding, coverage = (certificate.embedding, certificate.coverage)
    if embedding is None or coverage is None:
        raise failure_factory(
            "envelope.acceptance.certificates",
            "retain embedding and domain coverage",
            {"coverage": coverage, "embedding": embedding},
        )
    if embedding.status != "certified" or coverage.status != "certified":
        raise failure_factory(
            "envelope.acceptance.certificates",
            "certify embedding and domain coverage",
            {"coverage": coverage, "embedding": embedding},
        )
    domain = certificate.request.domain
    expected_domain = declared_plc_domain(source.plc_source.complex, source.source_id)
    if (
        domain is None
        or domain.domain_id != expected_domain.domain_id
        or coverage.domain_id != expected_domain.domain_id
        or (domain.source_id != source.source_id)
    ):
        raise failure_factory(
            "envelope.acceptance.domain",
            "retain the repaired source domain",
            {
                "coverage": coverage,
                "domain": domain,
                "expected_domain": expected_domain,
                "source": source,
            },
        )
    if coverage.covered_source_facet_count != coverage.source_facet_count:
        raise failure_factory(
            "envelope.acceptance.coverage",
            "cover every source facet",
            {"coverage": coverage},
        )
    if not result.audit.passed or not result.compliance.passed:
        raise failure_factory(
            "envelope.acceptance.result",
            "pass audit and request compliance",
            {"result": result},
        )
    return _EnvelopeCertificates(
        certificate=certificate, embedding=embedding, coverage=coverage
    )


def _envelope_repair_ancestry(
    source: phx.meshing.NativeSurfaceEnvelopeSource,
    result: phx.meshing.CellMeshingResult,
    coverage: phx.geometry.DomainCoverageCertificate,
    *,
    limits: phx.meshing.MeshingLimits,
    failure_factory: Callable[[str, str, dict[str, object]], RuntimeError],
) -> None:
    """Require the directed repair evidence, source ancestry, and repaired region."""
    envelope = source.envelope
    evidence = envelope.evidence
    achieved = dict(result.compliance.achieved)
    for name, expected_bound in (
        ("source_to_repaired_upper", evidence.source_to_repaired_upper),
        ("repaired_to_source_upper", evidence.repaired_to_source_upper),
        ("source_inclusion_margin", evidence.source_inclusion_margin),
        ("interpolation_error", evidence.interpolation_error),
    ):
        if achieved.get(f"envelope:{name}") != expected_bound:
            raise failure_factory(
                f"envelope.acceptance.{name}",
                "retain the directed bound evidence",
                {"achieved": achieved, "evidence": evidence, "name": name},
            )
    repair_inputs = (
        evidence.source_id,
        evidence.source_revision,
        evidence.source_geometry_id,
        evidence.policy_id,
    )
    repair_outputs = (
        source.source_id,
        source.source_revision,
        evidence.repaired_model_id,
        evidence.evidence_id,
    )
    if not any(
        (
            stage.stage is phx.meshing.MeshingStageKind.TOPOLOGY_REPAIR
            and stage.status is phx.meshing.MeshingStageStatus.PASSED
            and (stage.input_ids == repair_inputs)
            and (stage.output_ids == repair_outputs)
            for stage in result.trace.stages
        )
    ):
        raise failure_factory(
            "envelope.acceptance.repair_ancestry",
            "retain the source repair transition",
            {
                "repair_inputs": repair_inputs,
                "repair_outputs": repair_outputs,
                "result": result,
            },
        )
    try:
        _require_surface_envelope_repair_provenance(source, result, limits=limits)
    except CoordinateEnclosureResourceError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise failure_factory(
            "envelope.acceptance.repair_provenance",
            "authenticate the retained authored repair and independently certified PLC",
            {"result": result, "source": source, "error": str(error)},
        ) from error
    if (
        evidence.source_inclusion_margin <= 0.0
        or max(evidence.source_to_repaired_upper, evidence.repaired_to_source_upper)
        > envelope.policy.maximum_deviation
        or evidence.repaired_topology.boundary_edges
        or evidence.repaired_topology.nonmanifold_edges
        or evidence.repaired_topology.degenerate_faces
        or evidence.repaired_topology.duplicate_faces
    ):
        raise failure_factory(
            "envelope.acceptance.repair_evidence",
            "certify wrapping policy and repaired topology",
            {"envelope": envelope, "evidence": evidence},
        )
    if not result.associations or any(
        (
            association.source_id != source.source_id
            or association.source_revision != source.source_revision
            for association in result.associations
        )
    ):
        raise failure_factory(
            "envelope.acceptance.associations",
            "retain current repaired source associations",
            {"result": result, "source": source},
        )
    zones = {
        zone.name for zone in result.zones if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    if zones != {"wrapped-solid"} or coverage.region_ids != ("wrapped-solid",):
        raise failure_factory(
            "envelope.acceptance.regions",
            "retain the declared repaired region",
            {"coverage": coverage, "zones": zones},
        )


def _envelope_volume_closure(
    source: phx.meshing.NativeSurfaceEnvelopeSource,
    result: phx.meshing.CellMeshingResult,
    *,
    failure_factory: Callable[[str, str, dict[str, object]], RuntimeError],
    measure_cells: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> _EnvelopeVolumeClosure:
    """Compare positive affine cell volumes with the oriented repaired boundary."""
    envelope = source.envelope
    mesh = result.mesh
    if (
        mesh.topological_dimension != 3
        or mesh.ambient_dimension != 3
        or any((block.cell_kind != "tetrahedron" for block in mesh.blocks))
    ):
        raise failure_factory(
            "envelope.acceptance.mesh",
            "contain full-dimensional affine tetrahedra",
            {"mesh": mesh},
        )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    measures = measure_cells(points, cells)
    surface = envelope.repaired.mesh
    skin = np.asarray(surface.coordinates, dtype=np.float64)[
        np.asarray(surface.blocks[0].vertices, dtype=np.int64)
    ]
    reference_volume = float(
        np.sum(np.sum(skin[:, 0] * np.cross(skin[:, 1], skin[:, 2]), axis=1)) / 6.0
    )
    volume = float(np.sum(measures))
    volume_defect = volume - reference_volume
    if np.any(measures <= 0.0) or reference_volume <= 0.0 or abs(volume_defect) > 1e-12:
        raise failure_factory(
            "envelope.acceptance.volume",
            "have positive cells and close the signed-volume reference",
            {
                "measures": measures,
                "reference_volume": reference_volume,
                "volume_defect": volume_defect,
            },
        )
    return _EnvelopeVolumeClosure(
        reference_volume=reference_volume,
        volume=volume,
        volume_defect=volume_defect,
        measures=measures,
        skin=skin,
    )


@checked
def validate_envelope_acceptance(
    source: phx.meshing.NativeSurfaceEnvelopeSource,
    result: phx.meshing.CellMeshingResult,
    *,
    limits: phx.meshing.MeshingLimits,
    failure_factory: Callable[[str, str, dict[str, object]], RuntimeError],
    measure_cells: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, object]:
    """Check repaired fill under independently retained caller request limits."""
    envelope = source.envelope
    evidence = envelope.evidence
    certificates = _envelope_certificates(source, result, failure_factory=failure_factory)
    certificate, embedding, coverage = (
        certificates.certificate,
        certificates.embedding,
        certificates.coverage,
    )
    _envelope_repair_ancestry(
        source, result, coverage, limits=limits, failure_factory=failure_factory
    )
    closure = _envelope_volume_closure(
        source, result, failure_factory=failure_factory, measure_cells=measure_cells
    )
    reference_volume, volume, volume_defect, measures, skin = (
        closure.reference_volume,
        closure.volume,
        closure.volume_defect,
        closure.measures,
        closure.skin,
    )
    return {
        "passed": True,
        "repair_evidence": asdict(evidence),
        "interpretation": "sampled_unsigned_distance_sublevel",
        "source_solid_volume": None,
        "source_solid_volume_status": "undefined-open-sheet-no-interior-inference",
        "original_source_id": evidence.source_id,
        "original_source_revision": evidence.source_revision,
        "repaired_source_id": source.source_id,
        "repaired_source_revision": source.source_revision,
        "repaired_model_id": evidence.repaired_model_id,
        "carrier_id": envelope.carrier_id,
        "policy_id": envelope.policy.policy_id,
        "feature_permission": envelope.policy.feature_permission.value,
        "topology_permission": envelope.policy.topology_permission.value,
        "maximum_deviation": envelope.policy.maximum_deviation,
        "offset": envelope.policy.offset,
        "repaired_boundary_volume": reference_volume,
        "mesh_volume": volume,
        "volume_defect": volume_defect,
        "minimum_cell_volume": float(np.min(measures)),
        "boundary_coordinates_min": np.min(skin.reshape((-1, 3)), axis=0).tolist(),
        "boundary_coordinates_max": np.max(skin.reshape((-1, 3)), axis=0).tolist(),
        "topology_comparison_semantics": "incidence-counts-not-homeomorphism",
        "embedding_id": embedding.certificate_id,
        "embedding_checks": list(embedding.evaluated_checks),
        "coverage_id": coverage.certificate_id,
        "domain_id": coverage.domain_id,
        "covered_source_facets": coverage.covered_source_facet_count,
        "source_facets": coverage.source_facet_count,
        "requested_region_measures": list(coverage.requested_region_measures),
        "achieved_region_measures": list(coverage.achieved_region_measures),
        "requested_region_measure_bounds": list(coverage.requested_region_measure_bounds),
        "achieved_region_measure_bounds": list(coverage.achieved_region_measure_bounds),
        "integration_error_bounds": list(coverage.integration_error_bounds),
        "repair_fidelity_semantics": "rigorous-two-directed-envelope-bounds",
        "repaired_domain_fidelity_semantics": "exact-represented-PLC-domain-coverage",
        "fidelity_certificate_id": None
        if certificate.fidelity is None
        else certificate.fidelity.certificate_id,
        "association_ids": [
            association.association_id for association in result.associations
        ],
        "quality": {
            "minimum_mean_ratio": result.quality.minimum_mean_ratio,
            "maximum_aspect_ratio": result.quality.maximum_aspect_ratio,
            "minimum_scaled_jacobian": result.quality.minimum_scaled_jacobian,
            "minimum_angle": result.quality.minimum_angle,
            "maximum_angle": result.quality.maximum_angle,
            "sampled_invalid_count": result.quality.sampled_invalid_count,
        },
        "compliance_requested": dict(result.compliance.requested),
        "compliance_achieved": dict(result.compliance.achieved),
        "soft_aim_semantics": "Native radius-edge and dihedral aims retain explicit unmet evidence.",
    }


def envelope_solver_field_templates() -> tuple[
    tuple[str, phx.discretization.FiniteElementFieldSpec], ...
]:
    """Return the repaired-volume scalar diffusion declaration."""
    return (
        (
            "solve:intensive",
            phx.discretization.FiniteElementFieldSpec(
                "u", phx.discretization.lagrange_element("tetrahedron", 1)
            ),
        ),
    )


def envelope_transport_field_templates(
    cell_kind: str, /
) -> tuple[tuple[str, phx.discretization.FiniteElementFieldSpec], ...]:
    """Return the transported solution, material density, and accepted history."""
    element = phx.discretization.lagrange_element(cell_kind, 1)
    return tuple(
        (role, phx.discretization.FiniteElementFieldSpec(name, element))
        for role, name in (
            ("transport:intensive", "u"),
            ("transport:conservative-density", "density"),
            ("transport:material-history", "history"),
        )
    )


@checked
def prepare_envelope_accepted_epoch(
    result: phx.meshing.CellMeshingResult,
    scalar_space: phx.discretization.FiniteElementDiscretization,
    values: jax.Array,
    auxiliary: tuple[jax.Array, jax.Array],
    epoch: int,
    *,
    part_name: str,
    transition: phx.meshing.MeshAdaptationResult | None = None,
) -> MeshingAcceptedEpoch:
    """Bind transported solution, continuous density, and material history.

    The coefficient layouts and field templates are those used by the topology
    transaction. The scalar solve must use the same carrier and coordinate map.
    """
    fields = tuple(
        field
        for _, field in envelope_transport_field_templates(
            result.mesh.blocks[0].cell_kind
        )
    )
    plan = phx.discretization.FiniteElementPlan(
        result.mesh, fields, coordinate_spec=result.geometry
    )
    owner = plan.prepare()
    if (
        scalar_space.mesh.mesh_id != result.mesh.mesh_id
        or scalar_space.numeric_version != owner.numeric_version
        or cell_geometry_id(scalar_space.coordinate_spec)
        != cell_geometry_id(result.geometry)
        or len(scalar_space.dof_maps) != 1
        or scalar_space.dof_maps[0].dof_map_id != owner.dof_maps[0].dof_map_id
    ):
        raise ValueError(
            "envelope.accepted_epoch.scalar_space must match the accepted scalar coefficient owner; got a foreign carrier, geometry, or layout"
        )
    state = dict(zip(("u", "density", "history"), (values, *auxiliary), strict=True))
    spaces = {field.name: field for field in owner.field_spaces}
    bindings = {
        name: prepare_meshing_field_index_binding(
            result, "field/global_coefficient_ids", field_space=space
        )
        for name, space in spaces.items()
    }
    roles = tuple(
        MeshingFieldStateRole(
            name,
            name,
            "coefficients",
            tuple(array.shape),
            np.dtype(array.dtype),
            epoch,
            index_binding=bindings[name],
        )
        for name, array in state.items()
    ) + (
        MeshingFieldStateRole(
            "accepted-history",
            "history",
            "material-history",
            tuple(auxiliary[1].shape),
            np.dtype(auxiliary[1].dtype),
            epoch,
            value_units=(phx.units.ONE,),
            index_binding=bindings["history"],
        ),
    )
    declaration = MeshingFiniteElementFieldDeclaration(
        part_name=part_name,
        topology_id=result.mesh.topology_id,
        geometry_layout_id=result.geometry.geometry_layout_id,
        field_space_ids=frozendict(
            {name: space.field_space_id for name, space in spaces.items()}
        ),
        value_units=frozendict({name: (phx.units.ONE,) for name in state}),
        maximum_derivative_orders=frozendict({name: 1 for name in state}),
        branch_policy=BranchDifferentiationPolicy.FROZEN_DECISION,
        trace_policy=phx.discretization.FieldTracePolicy("cell-sided"),
        state_roles=roles,
        history_policy="material-history",
        numeric_version=owner.numeric_version,
        finite_element_fields=plan.fields,
        precision_policy=plan.precision_policy,
    )
    return MeshingAcceptedEpoch(
        result,
        (declaration,),
        {**state, "accepted-history": auxiliary[1]},
        transition=transition,
    )


@checked
def prepare_envelope_solver(
    result: phx.meshing.CellMeshingResult,
    *,
    failure_factory: Callable[[str, str, dict[str, object]], RuntimeError],
    source_coefficient: Callable[[jax.Array, object], jax.Array],
    dirichlet_values: Callable[[jax.Array], jax.Array],
) -> tuple[
    phx.discretization.FiniteElementDiscretization,
    phx.equations.CompiledFiniteElementProblem,
    jax.Array,
]:
    """Scalar H1 diffusion on the repaired domain, not the open sheet."""
    space = phx.discretization.FiniteElementPlan(
        result.mesh,
        envelope_solver_field_templates()[0][1],
        coordinate_spec=result.geometry,
    ).prepare()
    form = phx.equations.FiniteElementForm(
        "repaired-envelope-manufactured-diffusion",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    source_coefficient,
                    coefficient_id="repaired-domain-quadratic-laplacian",
                ),
            ),
        ),
    )
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    if constraint.free_dofs.size == 0:
        raise failure_factory(
            "envelope.solver.free_dofs",
            "contain interior scalar H1 degrees of freedom",
            {"constraint": constraint},
        )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint, dirichlet_values=dirichlet_values
    )
    return (space, problem, constraint.free_dofs)


class _EnvelopeFieldPhase(phx.StrictModule):
    fields: tuple[phx.discretization.FiniteElementFieldSpec, ...]
    source_fields: tuple[jax.Array, ...]
    accepted: phx.solver.FiniteElementAcceptedState


class _EnvelopeRollbackPhase(phx.StrictModule):
    result: phx.solver.FiniteElementTopologyResult
    receipt: CompositionRebindReceipt


class _EnvelopeCommitPhase(phx.StrictModule):
    result: phx.solver.FiniteElementTopologyResult
    receipt: CompositionRebindReceipt


def _prepare_envelope_fields(
    result: phx.meshing.CellMeshingResult,
    problem: phx.equations.CompiledFiniteElementProblem,
    values: jax.Array,
    epoch: int,
    auxiliary: tuple[jax.Array, jax.Array] | None,
) -> _EnvelopeFieldPhase:
    dimension = result.mesh.topological_dimension
    if dimension not in (2, 3):
        raise ValueError(
            f"envelope.state_transport.mesh must use triangle or tetrahedron cells; got topological dimension {dimension}"
        )
    cell_kind = "triangle" if dimension == 2 else "tetrahedron"
    fields = tuple(field for _, field in envelope_transport_field_templates(cell_kind))
    all_fields = phx.discretization.FiniteElementPlan(
        result.mesh, fields, coordinate_spec=result.geometry
    ).prepare()
    if auxiliary is None:
        auxiliary = (
            jnp.full((all_fields.field_spaces[1].layout.size,), 2.0, dtype=values.dtype),
            all_fields.dof_maps[2].dof_coordinates[:, 0] + 0.5,
        )
    source_fields = (values, *auxiliary)
    accepted = phx.solver.FiniteElementAcceptedState(
        source_fields,
        0.0,
        epoch,
        result.mesh.topology_id,
        all_fields.prepared_id,
        problem.compilation_id,
        state_version=epoch,
    )
    return _EnvelopeFieldPhase(
        fields=fields, source_fields=source_fields, accepted=accepted
    )


@final
class _EnvelopeErrorGate(phx.StrictModule, NonTrainableState):
    error: Callable[[phx.discretization.CellMesh, jax.Array], float]

    def __call__(
        self,
        mesh: phx.discretization.CellMesh,
        fields: tuple[jax.Array, ...],
        materials: object,
        lineage: object,
        context: object,
    ) -> bool:
        del materials, lineage, context
        return self.error(mesh, fields[0]) <= 0.0


@final
class _EnvelopeRebindCapture(phx.StrictModule, NonTrainableState):
    records: list[CompositionRebind]

    def __call__(self, rebind: CompositionRebind) -> CompositionRebind:
        self.records.append(rebind)
        return rebind


def _finite_envelope_fields(
    mesh: phx.discretization.CellMesh,
    fields: tuple[jax.Array, ...],
    materials: object,
    lineage: object,
    context: object,
) -> bool:
    del mesh, materials, lineage, context
    return all(bool(jnp.all(jnp.isfinite(value))) for value in fields)


def _check_envelope_rollback(
    result: phx.meshing.CellMeshingResult,
    adaptation: phx.meshing.MeshAdaptationResult,
    phase: _EnvelopeFieldPhase,
    projection_policy: phx.geometry.CommonRefinementPolicy | None,
    physical_error: Callable[[phx.discretization.CellMesh, jax.Array], float],
    failure_factory: Callable[[str, str, dict[str, object]], RuntimeError],
) -> _EnvelopeRollbackPhase:
    fields, source_fields, accepted = phase.fields, phase.source_fields, phase.accepted
    rejected = phx.solver.FiniteElementTopologyTransaction(
        _EnvelopeErrorGate(error=physical_error),
        fields=fields,
        projection_policy=projection_policy,
    ).execute(accepted, result.mesh, adaptation)
    if (
        bool(rejected.committed)
        or rejected.receipt is None
        or rejected.receipt.published
        or (rejected.state is not accepted)
        or (rejected.mesh is not result.mesh)
        or any(
            (
                not np.array_equal(np.asarray(actual), np.asarray(expected))
                for actual, expected in zip(
                    rejected.state.fields, source_fields, strict=True
                )
            )
        )
    ):
        raise failure_factory(
            "envelope.state.rollback",
            "preserve the accepted source epoch and field arrays",
            {
                "accepted": accepted,
                "rejected": rejected,
                "result": result,
                "source_fields": source_fields,
            },
        )
    return _EnvelopeRollbackPhase(result=rejected, receipt=rejected.receipt)


def _check_envelope_commit(
    transaction: phx.solver.FiniteElementTopologyResult,
    rebinds: list[CompositionRebind],
    failure_factory: Callable[[str, str, dict[str, object]], RuntimeError],
) -> _EnvelopeCommitPhase:
    if (
        not bool(transaction.committed)
        or transaction.receipt is None
        or (not transaction.receipt.published)
        or (not rebinds)
        or (not all((item.evidence.passed for item in transaction.transfers)))
    ):
        raise failure_factory(
            "envelope.state.commit",
            "publish the rebind with passed field-transfer evidence",
            {"rebinds": rebinds, "transaction": transaction},
        )
    return _EnvelopeCommitPhase(result=transaction, receipt=transaction.receipt)


@checked
def transition_envelope_state(
    result: phx.meshing.CellMeshingResult,
    adaptation: phx.meshing.MeshAdaptationResult,
    problem: phx.equations.CompiledFiniteElementProblem,
    values: jax.Array,
    epoch: int,
    auxiliary: tuple[jax.Array, jax.Array] | None = None,
    *,
    projection_policy: phx.geometry.CommonRefinementPolicy | None = None,
    failure_factory: Callable[[str, str, dict[str, object]], RuntimeError],
    physical_error: Callable[[phx.discretization.CellMesh, jax.Array], float],
) -> tuple[jax.Array, tuple[jax.Array, jax.Array], dict[str, object]]:
    """Atomically carry solved, material-density and history fields without reseeding."""
    phase = _prepare_envelope_fields(result, problem, values, epoch, auxiliary)
    fields, accepted = phase.fields, phase.accepted
    capture = _EnvelopeRebindCapture(records=[])
    rebinds = capture.records
    rollback = _check_envelope_rollback(
        result, adaptation, phase, projection_policy, physical_error, failure_factory
    )
    rejected = rollback.result
    transaction = phx.solver.FiniteElementTopologyTransaction(
        _finite_envelope_fields,
        fields=fields,
        composition_rebind=capture,
        projection_policy=projection_policy,
    ).execute(accepted, result.mesh, adaptation)
    committed = _check_envelope_commit(transaction, rebinds, failure_factory)
    target_fields = phx.discretization.FiniteElementPlan(
        adaptation.target.mesh, fields, coordinate_spec=adaptation.target.geometry
    ).prepare()
    density_defect = float(jnp.max(jnp.abs(transaction.state.fields[1] - 2.0)))
    history_defect = float(
        jnp.max(
            jnp.abs(
                transaction.state.fields[2]
                - (target_fields.dof_maps[2].dof_coordinates[:, 0] + 0.5)
            )
        )
    )
    if density_defect > 1e-12 or history_defect > 1e-12:
        raise failure_factory(
            "envelope.state.fields",
            "satisfy the affine density and history references",
            {"density_defect": density_defect, "history_defect": history_defect},
        )
    return (
        transaction.state.fields[0],
        (transaction.state.fields[1], transaction.state.fields[2]),
        {
            "receipt_id": committed.receipt.receipt_id,
            "source_composition_id": rebinds[-1].source.composition_id,
            "target_composition_id": committed.receipt.composition.composition_id,
            "transport_accepted": list(committed.receipt.transport_accepted),
            "field_names": [field.name for field in fields],
            "density_max_defect": density_defect,
            "history_max_defect": history_defect,
            "auxiliary_seed_origin": "authored-initial-fields"
            if epoch == 0
            else "actual-carried-fields",
            "field_transfers": [
                {
                    "transfer_id": item.transfer_id,
                    "evidence_id": item.evidence.evidence_id,
                    "passed": item.evidence.passed,
                    "defects": dict(item.evidence.defects),
                    "bounds": dict(item.evidence.bounds),
                }
                for item in transaction.transfers
            ],
            "rollback": {
                "passed": True,
                "requested_l2_error": 0.0,
                "diagnostics": rejected.diagnostics,
                "receipt_id": rollback.receipt.receipt_id,
                "accepted_id": accepted.accepted_id,
                "topology_id": rejected.mesh.topology_id,
                "state_version": rejected.state.state_version,
            },
        },
    )
