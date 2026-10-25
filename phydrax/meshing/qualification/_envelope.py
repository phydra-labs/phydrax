# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded preparation of repaired-envelope sources and their rational associations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any, final, NamedTuple, Protocol, TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx

from ..._differentiation import BranchDifferentiationPolicy
from ..._frozendict import frozendict
from ..._meshcore import NativeExecutionScope
from ..._resource_ticket import HostBatchTicket
from ..._trainable import NonTrainableState
from ...discretization._cell_geometry_validity import cell_geometry_id
from ...discretization._coordinate_budget import CoordinateEnclosureResourceError
from ...geometry._domain_coverage import DomainCoverageCertificate
from ...lifecycle import (
    CompositionRebind,
    CompositionRebindReceipt,
    MeshingAcceptedEpoch,
    MeshingFieldStateRole,
    MeshingFiniteElementFieldDeclaration,
    prepare_meshing_field_index_binding,
)
from ...solver._finite_element_adaptivity import (
    PreparedFiniteElementTopologyTransition,
)
from ...sparse import EdgeRelation, SparseLinearMap
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


if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from ...discretization.fem._topology_transfer import SourceRealizationFieldSemantics
    from ...geometry._volume_material_map import VolumeMaterialChain, VolumeMaterialEdit


class _MappedMaterialError(TypedDict):
    field: str
    maximum_local_L2: list[float]
    edits: int


class _MappedMaterialReference(TypedDict):
    density_content_defect: float
    weak_defects: list[float]
    source_density_content: float
    target_density_content: float
    material_L2_errors: list[_MappedMaterialError]
    material_error_tolerance: float
    reference_coverage_ratio: float
    coefficient_excess: list[float]
    coefficient_bounds: list[float]
    coefficient_L2_bounds: list[float]


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
    """Bind rational repaired-carrier identity tables in their source order.

    The source-support proof is not duplicated here: adaptation preparation
    proves ``source_associations`` for this transfer and result on its own
    request ledger before any association is consumed.
    """
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
    if embedding is None or not isinstance(coverage, DomainCoverageCertificate):
        raise failure_factory(
            "envelope.acceptance.certificates",
            "retain embedding and exact repaired-domain coverage",
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
    cell_kind: str, /, *, solution_degree: int = 1
) -> tuple[tuple[str, phx.discretization.FiniteElementFieldSpec], ...]:
    """Keep density P1 while matching solution and material-history degrees."""
    return tuple(
        (
            role,
            phx.discretization.FiniteElementFieldSpec(
                name, phx.discretization.lagrange_element(cell_kind, degree)
            ),
        )
        for role, name, degree in (
            ("transport:intensive", "u", solution_degree),
            ("transport:conservative-density", "density", 1),
            ("transport:material-history", "history", solution_degree),
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
    transport_space: phx.discretization.FiniteElementDiscretization | None = None,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> MeshingAcceptedEpoch:
    """Bind transported solution, continuous density, and material history.

    The coefficient layouts and field templates are those used by the topology
    transaction. The scalar solve must use the same carrier and coordinate map.
    A caller-retained transported-field owner is admitted only if it realizes
    this plan bitwise.
    """
    fields = tuple(
        field
        for _, field in envelope_transport_field_templates(
            result.mesh.blocks[0].cell_kind,
            solution_degree=scalar_space.elements[0][0].degree,
        )
    )
    plan = phx.discretization.FiniteElementPlan(
        result.mesh, fields, coordinate_spec=result.geometry, budget=budget,
    )
    owner = plan.prepare(reuse_prepared=transport_space, budget=budget)
    if (
        scalar_space.mesh.mesh_id != result.mesh.mesh_id
        or scalar_space.numeric_version != owner.numeric_version
        or cell_geometry_id(scalar_space.coordinate_spec)
        != cell_geometry_id(result.geometry)
        or len(scalar_space.dof_maps) != 1
        or scalar_space.dof_maps[0].dof_map_id != owner.dof_maps[0].dof_map_id
        or values.shape != (scalar_space.dof_maps[0].global_dof_count,)
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
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
) -> tuple[
    phx.discretization.FiniteElementDiscretization,
    phx.equations.CompiledFiniteElementProblem,
    jax.Array,
]:
    """Prepare scalar H1 diffusion, sharing an explicit cold request budget."""
    space = phx.discretization.FiniteElementPlan(
        result.mesh,
        envelope_solver_field_templates()[0][1],
        coordinate_spec=result.geometry,
        budget=budget,
    ).prepare(budget=budget)
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
        # Real unit diffusion uses the same scalar H1 trial/test space.
        # The strong Dirichlet Galerkin restriction preserves symmetry;
        # the source and inhomogeneous lift affect only the right-hand side.
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            evidence={"self_adjoint": "construction"},
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
        form, space, constraint=constraint, dirichlet_values=dirichlet_values, budget=budget
    )
    return (space, problem, constraint.free_dofs)


class _EnvelopeFieldPhase(phx.StrictModule):
    fields: tuple[phx.discretization.FiniteElementFieldSpec, ...]
    source_fields: tuple[jax.Array, ...]
    accepted: phx.solver.FiniteElementAcceptedState
    space: phx.discretization.FiniteElementDiscretization


class _EnvelopeRollbackPhase(phx.StrictModule):
    result: phx.solver.FiniteElementTopologyResult
    receipt: CompositionRebindReceipt


class _EnvelopeCommitPhase(phx.StrictModule):
    result: phx.solver.FiniteElementTopologyResult
    receipt: CompositionRebindReceipt


def _envelope_transport_fields(
    mesh: phx.discretization.CellMesh, /, *, solution_degree: int
) -> tuple[phx.discretization.FiniteElementFieldSpec, ...]:
    dimension = mesh.topological_dimension
    if dimension not in (2, 3):
        raise ValueError(
            f"envelope.state_transport.mesh must use triangle or tetrahedron cells; got topological dimension {dimension}"
        )
    cell_kind = "triangle" if dimension == 2 else "tetrahedron"
    return tuple(
        field
        for _, field in envelope_transport_field_templates(
            cell_kind, solution_degree=solution_degree
        )
    )


def _prepare_envelope_fields(
    result: phx.meshing.CellMeshingResult,
    problem: phx.equations.CompiledFiniteElementProblem,
    values: jax.Array,
    epoch: int,
    auxiliary: tuple[jax.Array, jax.Array] | None,
    space: phx.discretization.FiniteElementDiscretization | None,
    budget: phx.discretization.CoordinateEnclosureBudget | None,
) -> _EnvelopeFieldPhase:
    scalar_space = problem.discretization
    if not isinstance(scalar_space, phx.discretization.FiniteElementDiscretization):
        raise TypeError("envelope.state_transport.problem must retain its scalar FE space")
    fields = _envelope_transport_fields(
        result.mesh, solution_degree=scalar_space.elements[0][0].degree
    )
    # A caller-retained owner is admitted only if it realizes this plan bitwise.
    all_fields = phx.discretization.FiniteElementPlan(
        result.mesh, fields, coordinate_spec=result.geometry, budget=budget
    ).prepare(reuse_prepared=space, budget=budget)
    if auxiliary is None:
        # Authored material fields seed only the first epoch; later epochs
        # must carry their accepted material and history, never a reseed.
        if epoch != 0:
            raise ValueError(
                "envelope.state_transport.auxiliary must carry the accepted density and history after epoch 0; got None"
            )
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
        fields=fields, source_fields=source_fields, accepted=accepted, space=all_fields
    )


@final
class _EnvelopeRollbackGate(phx.StrictModule, NonTrainableState):
    """Force a rejected transaction independently of solution representability."""

    def __call__(
        self,
        mesh: phx.discretization.CellMesh,
        fields: tuple[jax.Array, ...],
        materials: object,
        lineage: object,
        context: object,
    ) -> bool:
        del mesh, fields, materials, lineage, context
        return False


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
    gate: phx.solver.FiniteElementTopologyTransaction,
    preparation: PreparedFiniteElementTopologyTransition,
    failure_factory: Callable[[str, str, dict[str, object]], RuntimeError],
) -> _EnvelopeRollbackPhase:
    source_fields, accepted = phase.source_fields, phase.accepted
    # The forced rejection applies the prepared operators and gates the
    # candidate fields in its own transaction; it never reaches publication.
    rejected = gate.execute(accepted, result.mesh, adaptation, preparation=preparation)
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


def _mapped_p1_cells(mesh: phx.discretization.CellMesh) -> tuple[np.ndarray, np.ndarray]:
    if any(block.cell_kind != "tetrahedron" for block in mesh.blocks):
        raise ValueError("envelope.material_map requires affine tetrahedral P1 fields")
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    points = np.asarray(mesh.coordinates)[cells]
    volumes = np.linalg.det(points[:, 1:] - points[:, :1]) / 6.0
    if np.any(volumes <= 0.0):
        raise ValueError("envelope.material_map requires positive physical cells")
    return cells, volumes


def _mapped_p1_mass(mesh: phx.discretization.CellMesh) -> csr_matrix:
    from scipy.sparse import coo_matrix

    cells, volumes = _mapped_p1_cells(mesh)
    local = volumes[:, None, None] * (np.ones((4, 4)) + np.eye(4)) / 20.0
    rows = np.broadcast_to(cells[:, :, None], local.shape).ravel()
    columns = np.broadcast_to(cells[:, None, :], local.shape).ravel()
    count = mesh.coordinates.shape[0]
    return coo_matrix((local.ravel(), (rows, columns)), shape=(count, count)).tocsr()


class MaterialChainReference(NamedTuple):
    """Independent edit-by-edit values, local L2 errors and certificates.

    ``coverage_ratio`` is the largest per-cell piece-volume defect divided by
    its binary64 Hadamard roundoff bound (at most one when every piece,
    however thin, is resolved). ``value_bounds`` are certified forward-error
    bounds of every replayed value, propagated through each local solve from
    its inputs' bounds, its residual and binary64 accumulation.
    ``condition_sum`` reports the coefficient
    amplification of all local solves.
    """

    values: dict[str, dict[int, np.ndarray]]
    l2_errors: dict[str, np.ndarray]
    coverage_ratio: float
    condition_sum: float
    value_bounds: dict[str, dict[int, np.ndarray]]


def _reference_hadamard(tets: np.ndarray) -> np.ndarray:
    """Hadamard bound ``prod |edge|`` of each tetrahedron's volume determinant."""
    return np.prod(np.linalg.norm(tets[:, 1:] - tets[:, :1], axis=2), axis=1)


def _reference_halfspaces(tet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outward normals ``n`` and offsets ``c`` with ``n @ x <= c`` on one tetrahedron."""
    normals, offsets = [], []
    for opposite in range(4):
        face = np.delete(tet, opposite, axis=0)
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        if normal @ (tet[opposite] - face[0]) > 0.0:
            normal = -normal
        normals.append(normal)
        offsets.append(normal @ face[0])
    return np.asarray(normals), np.asarray(offsets)


def _reference_clip(tets: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Clip tetrahedra to ``normal @ x <= offset`` by the standard tetrahedral cases.

    Every case is an explicit tetrahedral split of the retained corner,
    wedge or truncated prism, so thin positive-volume slivers keep their
    volume and a pure contact contributes only its (vanishing) volume. No
    thickness threshold or hull reconstruction decides membership.
    """
    pieces: list[np.ndarray] = []
    for tet in tets:
        distance = tet @ normal - offset
        inside: list[int] = [int(index) for index in np.flatnonzero(distance <= 0.0)]
        outside: list[int] = [int(index) for index in np.flatnonzero(distance > 0.0)]

        def cut(a: int, b: int) -> np.ndarray:
            ratio = distance[a] / (distance[a] - distance[b])
            return tet[a] + ratio * (tet[b] - tet[a])

        match len(inside):
            case 4:
                pieces.append(tet)
            case 1:
                corner = inside[0]
                pieces.append(np.stack((tet[corner], *(cut(corner, o) for o in outside))))
            case 2:
                a, b = inside
                o, q = outside
                ao, aq, bo, bq = cut(a, o), cut(a, q), cut(b, o), cut(b, q)
                # Prism (a, ao, aq) -> (b, bo, bq).
                pieces.extend((
                    np.stack((tet[a], ao, aq, tet[b])),
                    np.stack((ao, aq, tet[b], bo)),
                    np.stack((aq, tet[b], bo, bq)),
                ))
            case 3:
                a, b, c = inside
                o = outside[0]
                ao, bo, co = cut(a, o), cut(b, o), cut(c, o)
                # Prism (a, b, c) -> (ao, bo, co).
                pieces.extend((
                    np.stack((tet[a], tet[b], tet[c], ao)),
                    np.stack((tet[b], tet[c], ao, bo)),
                    np.stack((tet[c], ao, bo, co)),
                ))
            case _:
                pass
    return np.asarray(pieces, dtype=np.float64).reshape((-1, 4, 3))


def _reference_barycentric(tet: np.ndarray) -> np.ndarray:
    """``X`` with ``lambda_i(x) = [x, 1] @ X[:, i]``."""
    return np.linalg.solve(np.hstack((tet, np.ones((4, 1)))), np.eye(4))


def _reference_volume(tets: np.ndarray) -> np.ndarray:
    return np.abs(np.linalg.det(tets[:, 1:] - tets[:, :1])) / 6.0


def _reference_cone_map(origin: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    first, second = (origin[1:] - origin[:1]).T, (image[1:] - image[:1]).T
    linear = np.linalg.solve(first.T, second.T).T
    return linear, image[0] - linear @ origin[0], float(np.linalg.det(linear))


def material_chain_reference(
    chain: VolumeMaterialChain,
    values: Mapping[str, Mapping[int, np.ndarray]],
    semantics: Mapping[str, SourceRealizationFieldSemantics],
    *,
    budget: phx.discretization.CoordinateEnclosureBudget,
) -> MaterialChainReference:
    """Replay a material chain on actual P1 vertex values with independent NumPy algebra.

    Pieces are re-derived by halfspace vertex enumeration of
    ``s ∩ C_k ∩ A_k^{-1}(t)`` from the declared cones (never the chain's exact
    piece bank), integrated exactly for affine products, and every local
    projection is solved directly (density by its content KKT system). Each
    edit's L2 error integrates the squared affine defect, never a cancelling
    quadratic form.

    Every host bank is admitted on ``budget`` before allocation: retained
    vertex values as they are created, and each edit's clipping banks, pieces
    and dense local forms inside one temporary scope.
    """
    mesh = chain.source_mesh
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    incidence: dict[int, set[int]] = {}
    for block in mesh.blocks:
        for identity, row in zip(
            np.asarray(block.global_ids).tolist(), np.asarray(block.vertices), strict=True,
        ):
            cells[int(identity)] = (vertex_ids[row], coordinates[row])
            for vertex in vertex_ids[row].tolist():
                incidence.setdefault(vertex, set()).add(int(identity))
    state = {name: {int(k): np.atleast_1d(np.asarray(v, dtype=np.float64)) for k, v in field.items()}
             for name, field in values.items()}
    errors: dict[str, list[np.ndarray]] = {name: [] for name in values}
    bounds = {name: {vertex: np.zeros_like(value) for vertex, value in field.items()} for name, field in state.items()}
    width = max((len(next(iter(field.values()), ())) for field in values.values()), default=1)
    budget.reserve(0, 64 * width * sum(len(field) for field in values.values()))
    coverage = 0.0
    condition_sum = 0.0
    for edit in chain.edits:
        # Only vertices new to the replay add retained values (one per field);
        # admit them before the edit allocates and stores any of them.
        created = np.unique(np.asarray(edit.target_vertex_ids, dtype=np.int64)).tolist()
        fresh = sum(int(vertex) not in incidence for vertex in created)
        budget.reserve(0, 128 * width * len(state) * fresh)
        with budget.temporary_scope():
            ratio, condition = _reference_edit(
                edit, cells, incidence, state, bounds, errors, semantics, width, budget,
            )
        coverage = max(coverage, ratio)
        condition_sum += condition
    return MaterialChainReference(
        state,
        {name: np.asarray(rows) for name, rows in errors.items()},
        coverage,
        condition_sum,
        bounds,
    )


def _reference_edit(
    edit: VolumeMaterialEdit,
    cells: dict[int, tuple[np.ndarray, np.ndarray]],
    incidence: dict[int, set[int]],
    state: dict[str, dict[int, np.ndarray]],
    bounds: dict[str, dict[int, np.ndarray]],
    errors: dict[str, list[np.ndarray]],
    semantics: Mapping[str, SourceRealizationFieldSemantics],
    width: int,
    budget: phx.discretization.CoordinateEnclosureBudget,
    /,
) -> tuple[float, float]:
    """Replay one edit; return its coverage ratio and local mass condition."""
    from ...discretization.fem._topology_transfer import CLAIM_ULPS

    epsilon = float(np.finfo(np.float64).eps)
    removed = [int(value) for value in np.asarray(edit.source_cell_ids)]
    created = [int(value) for value in np.asarray(edit.target_cell_ids)]
    old_vertices = np.asarray(edit.source_vertex_ids, dtype=np.int64)
    new_vertices = np.asarray(edit.target_vertex_ids, dtype=np.int64)
    old_points = np.asarray(edit.source_points, dtype=np.float64)
    new_points = np.asarray(edit.target_points, dtype=np.float64)
    created_vertices = sorted(set(new_vertices.ravel().tolist()))
    ring = sorted(
        {cell for vertex in created_vertices for cell in incidence.get(vertex, ())} - set(removed)
    )
    target_cells = [(new_vertices[i], new_points[i]) for i in range(len(created))]
    target_cells += [cells[cell] for cell in ring]
    source_cells = [(old_vertices[i], old_points[i]) for i in range(len(removed))]
    source_cells += [cells[cell] for cell in ring]
    # A patch vertex stays fixed only if a surviving cell outside the patch keeps it.
    enclosed = set(removed) | set(ring)
    vertices = sorted({int(v) for vids, _ in target_cells for v in vids})
    free = [v for v in vertices if not incidence.get(v, set()) - enclosed]
    patch = free + [v for v in vertices if v not in set(free)]
    inputs = sorted({int(v) for vids, _ in source_cells for v in vids})
    row = {vertex: index for index, vertex in enumerate(patch)}
    column = {vertex: index for index, vertex in enumerate(inputs)}
    charts = [
        _reference_cone_map(
            np.asarray(origin, dtype=np.float64), np.asarray(image, dtype=np.float64),
        )
        for origin, image in zip(edit.source_charts, edit.target_charts, strict=True)
    ] or [(np.eye(3), np.zeros(3), 1.0)]
    cones = [np.asarray(tet, dtype=np.float64) for tet in edit.source_charts]
    # (target cell, source cell, chart, sub-tetrahedra in source coordinates)
    pieces: list[tuple[int, int, int, np.ndarray]] = []
    for t in range(len(created)):
        image_planes = _reference_halfspaces(new_points[t])
        for s in range(len(removed)):
            for k, (linear, offset, _) in enumerate(charts):
                normals = [image_planes[0] @ linear]
                offsets = [image_planes[1] - image_planes[0] @ offset]
                if cones:
                    cone = _reference_halfspaces(cones[k])
                    normals.append(cone[0])
                    offsets.append(cone[1])
                tets = old_points[s][None]
                for normal, bound in zip(np.vstack(normals), np.concatenate(offsets), strict=True):
                    # One plane splits a tetrahedron into at most three.
                    budget.reserve(0, 3 * tets.nbytes)
                    tets = _reference_clip(tets, normal, float(bound))
                if tets.shape[0]:
                    pieces.append((t, s, k, tets))
    for index in range(len(ring)):
        pieces.append((len(created) + index, len(removed) + index, -1, target_cells[len(created) + index][1][None]))
    # Pieces plus their target/source barycentric evaluations.
    budget.reserve(0, 4 * sum(tets.nbytes for *_, tets in pieces) + 64 * len(pieces))
    source_volume = np.zeros(len(source_cells))
    target_volume = np.zeros(len(target_cells))
    source_bound = np.zeros(len(source_cells))
    target_bound = np.zeros(len(target_cells))
    for t, s, k, tets in pieces:
        jacobian = 1.0 if k < 0 else charts[k][2]
        volume = float(np.sum(_reference_volume(tets)))
        hadamard = float(np.sum(_reference_hadamard(tets)))
        source_volume[s] += volume
        target_volume[t] += volume * jacobian
        source_bound[s] += hadamard
        target_bound[t] += hadamard * jacobian
    source_points = np.asarray([points for _, points in source_cells])
    target_points = np.asarray([points for _, points in target_cells])
    # Every piece, however thin, must be resolved: the per-cell defect stays
    # within the binary64 Hadamard bound of all determinants that formed it.
    ratio = max(
        float(np.max(
            np.abs(volume - _reference_volume(points))
            / (CLAIM_ULPS * epsilon * (bound + _reference_hadamard(points)))
        ))
        for volume, points, bound in (
            (source_volume, source_points, source_bound),
            (target_volume, target_points, target_bound),
        )
    )
    # Simultaneously live doubles: mass and both mixed forms; per field the
    # KKT system with LAPACK's matrix copy and SVD condition workspace (three
    # more), right sides with their copy, and old/fixed/patch value banks.
    size, free_size = len(patch), len(free) + 1
    budget.reserve(0, 8 * (
        size * size + 2 * size * len(inputs) + size + len(inputs)
        + 5 * free_size * free_size + 4 * free_size * width
        + 2 * len(inputs) * width + 2 * size * width
    ))
    mass = np.zeros((len(patch), len(patch)))
    for vids, points in target_cells:
        local = _reference_volume(points[None])[0] / 20.0 * (np.ones((4, 4)) + np.eye(4))
        index = [row[int(v)] for v in vids]
        mass[np.ix_(index, index)] += local
    mixed = (np.zeros((len(patch), len(inputs))), np.zeros((len(patch), len(inputs))))
    evaluations = []
    for t, s, k, tets in pieces:
        linear, offset, jacobian = (np.eye(3), np.zeros(3), 1.0) if k < 0 else charts[k]
        mapped = tets @ linear.T + offset
        ones = np.ones(tets.shape[:2] + (1,))
        at_target = np.concatenate((mapped, ones), axis=2) @ _reference_barycentric(target_cells[t][1])
        at_source = np.concatenate((tets, ones), axis=2) @ _reference_barycentric(source_cells[s][1])
        volumes = _reference_volume(tets)[:, None, None]
        local = np.sum(volumes / 20.0 * (
            at_target.sum(axis=1)[:, :, None] * at_source.sum(axis=1)[:, None, :]
            + np.swapaxes(at_target, -1, -2) @ at_source
        ), axis=0)
        rows = [row[int(v)] for v in target_cells[t][0]]
        columns = [column[int(v)] for v in source_cells[s][0]]
        mixed[0][np.ix_(rows, columns)] += jacobian * local
        mixed[1][np.ix_(rows, columns)] += local
        evaluations.append((rows, columns, jacobian, at_target, at_source, _reference_volume(tets)))
    count = len(free)
    measures = mass.sum(axis=1)
    condition = float(np.linalg.cond(mass[:count, :count])) if count else 1.0
    if any(semantics[name] == "conservative-density" for name in state):
        kkt = np.zeros((count + 1, count + 1))
        kkt[:count, :count] = mass[:count, :count]
        kkt[:count, count] = kkt[count, :count] = measures[:count]
        condition = max(condition, float(np.linalg.cond(kkt)))
    source_measures = np.zeros(len(inputs))
    for vids, points in source_cells:
        for v in vids:
            source_measures[column[int(v)]] += _reference_volume(points[None])[0] / 4.0
    gamma = (len(inputs) + len(patch)) * epsilon / (1.0 - (len(inputs) + len(patch)) * epsilon)
    for name, field in state.items():
        density = semantics[name] == "conservative-density"
        old = np.stack([field[v] for v in inputs])
        old_bound = np.stack([bounds[name][v] for v in inputs])
        fixed = np.stack([field[v] for v in patch[count:]]) if len(patch) > count else np.zeros((0, old.shape[1]))
        fixed_bound = (
            np.stack([bounds[name][v] for v in patch[count:]]) if len(patch) > count else np.zeros_like(fixed)
        )
        source_form = mixed[1 if density else 0][:count]
        coupling = mass[:count, count:]
        rhs = source_form @ old - coupling @ fixed
        # Inherited bounds plus binary64 accumulation of the right side.
        rhs_bound = (
            np.abs(source_form) @ old_bound + np.abs(coupling) @ fixed_bound
            + gamma * (np.abs(source_form) @ np.abs(old) + np.abs(coupling) @ np.abs(fixed))
        )
        if density:
            system = np.zeros((count + 1, count + 1))
            system[:count, :count] = mass[:count, :count]
            system[:count, count] = measures[:count]
            system[count, :count] = measures[:count]
            content = source_measures @ old - measures[count:] @ fixed
            content_bound = (
                np.abs(source_measures) @ old_bound + np.abs(measures[count:]) @ fixed_bound
                + gamma * (np.abs(source_measures) @ np.abs(old) + np.abs(measures[count:]) @ np.abs(fixed))
            )
            right = np.vstack((rhs, content[None]))
            right_bound = np.vstack((rhs_bound, content_bound[None]))
        else:
            system = mass[:count, :count]
            right, right_bound = rhs, rhs_bound
        solution = np.linalg.solve(system, right)
        new = solution[:count]
        # ||dx||_2 <= (||dr||_2 + ||A x - r||_2 + gamma (||A|| ||x|| + ||r||)) / sigma_min.
        singular = np.linalg.svd(system, compute_uv=False)
        residual = np.linalg.norm(system @ solution - right, axis=0)
        forward = (
            np.linalg.norm(right_bound, axis=0) + residual
            + gamma * (singular[0] * np.linalg.norm(solution, axis=0) + np.linalg.norm(right, axis=0))
        ) / singular[-1]
        patch_values = np.vstack((new, fixed))
        # This edit's certified local L2 error: the computed defect plus the
        # same edit's value uncertainty and its binary64 evaluation. The
        # affine interpolant of vertex bounds dominates |affine defect|.
        patch_bound = np.vstack((np.broadcast_to(forward, new.shape), fixed_bound))
        squared = np.zeros(old.shape[1])
        uncertain = np.zeros(old.shape[1])
        magnitude = np.zeros(old.shape[1])

        def tet_square(values: np.ndarray, volumes: np.ndarray) -> np.ndarray:
            return np.sum(
                volumes[:, None] / 20.0 * (values.sum(axis=1) ** 2 + np.sum(values**2, axis=1)),
                axis=0,
            )

        for rows, columns, jacobian, at_target, at_source, volumes in evaluations:
            weight = jacobian if density else 1.0
            defect = at_target @ patch_values[rows] - at_source @ old[columns] / weight
            spread = np.abs(at_target) @ patch_bound[rows] + np.abs(at_source) @ old_bound[columns] / weight
            scale = np.abs(at_target) @ np.abs(patch_values[rows]) + np.abs(at_source) @ np.abs(old[columns]) / weight
            squared += jacobian * tet_square(defect, volumes)
            uncertain += jacobian * tet_square(spread, volumes)
            magnitude += jacobian * tet_square(scale, volumes)
        errors[name].append(
            np.sqrt(np.maximum(squared, 0.0)) + np.sqrt(uncertain) + gamma * np.sqrt(magnitude)
        )
        for position, vertex in enumerate(free):
            field[vertex] = np.asarray(new[position], dtype=np.float64)
            bounds[name][vertex] = np.asarray(forward, dtype=np.float64)
    for vertex in set(old_vertices.ravel().tolist()):
        incidence[vertex].difference_update(removed)
    for cell in removed:
        del cells[cell]
    for position, cell in enumerate(created):
        vids = np.asarray(new_vertices[position], dtype=np.int64)
        cells[cell] = (vids, np.asarray(new_points[position], dtype=np.float64))
        for vertex in vids.tolist():
            incidence.setdefault(int(vertex), set()).add(cell)
    return ratio, condition


def _mapped_envelope_field_reference(
    chain: VolumeMaterialChain,
    semantics: dict[str, SourceRealizationFieldSemantics],
    phase: _EnvelopeFieldPhase,
    target_space: phx.discretization.FiniteElementDiscretization,
    candidate_fields: tuple[jax.Array, ...],
    *,
    transfers: tuple[phx.discretization.FiniteElementFieldTransfer, ...],
    material_error_tolerance: float,
    budget: phx.discretization.CoordinateEnclosureBudget,
) -> _MappedMaterialReference:
    """Check carried P1 fields against an independent edit-by-edit NumPy replay.

    :func:`material_chain_reference` re-derives every material piece and local
    projection from the declared chain and the actual source values. The
    composed production transfers must reproduce its coefficients within the
    binary64 amplification of both replays' local solves (so thin cells cannot
    hide a wrong value behind a small mass) and in the weak sense. Density
    content is measured independently from both endpoint meshes.
    """
    from ...discretization.fem._topology_transfer import CLAIM_ULPS

    if any(
        element.degree != 1
        for elements in phase.space.elements for element in elements
    ):
        raise ValueError("envelope.material_map independent oracle requires declared P1 fields")
    names = tuple(field.name for field in phase.fields)
    density = next(
        (index for index, name in enumerate(names) if semantics[name] == "conservative-density"),
        None,
    )
    if density is None:
        raise ValueError("envelope.material_map requires one conservative density field")

    def vertex_values(
        mesh: phx.discretization.CellMesh,
        space: phx.discretization.FiniteElementDiscretization,
        fields: tuple[jax.Array | np.ndarray, ...],
    ) -> np.ndarray:
        cells, _ = _mapped_p1_cells(mesh)
        values = np.zeros((mesh.coordinates.shape[0], len(fields)))
        for index, field in enumerate(fields):
            routes = np.concatenate([np.asarray(row) for row in space.dof_maps[index].cell_dofs])
            values[cells.ravel(), index] = np.asarray(field)[routes].ravel()
        return values

    first, final = chain.source_mesh, chain.target_mesh
    source_cells, source_volumes = _mapped_p1_cells(first)
    source_values = vertex_values(first, phase.space, phase.source_fields)
    source_ids = np.asarray(first.vertex_global_ids, dtype=np.int64).tolist()
    reference = material_chain_reference(
        chain,
        {
            name: {vertex: source_values[row, index : index + 1] for row, vertex in enumerate(source_ids)}
            for index, name in enumerate(names)
        },
        semantics,
        budget=budget,
    )
    initial_content = float(source_volumes @ source_values[source_cells, density].mean(axis=1))
    cells, volumes = _mapped_p1_cells(final)
    actual = vertex_values(final, target_space, candidate_fields)
    expected = np.zeros_like(actual)
    for row, vertex in enumerate(np.asarray(final.vertex_global_ids, dtype=np.int64).tolist()):
        for index, name in enumerate(names):
            expected[row, index] = reference.values[name][vertex][0]
    mass = _mapped_p1_mass(final)
    weak = np.max(np.abs(mass @ (actual - expected)), axis=0)
    # Per-vertex certified uncertainty of both sides: the production row's
    # propagated forward error (its action condition at that row's claim
    # scale, times the source field scale) plus the replay's own bound.
    unit = CLAIM_ULPS * float(np.finfo(np.float64).eps)
    row_scales = []
    for transfer in transfers:
        primal = transfer.transfer.primal
        if not isinstance(primal, SparseLinearMap) or not isinstance(primal.relation, EdgeRelation):
            raise ValueError("envelope.material_map transfers must expose coalesced sparse rows")
        row_scales.append(np.maximum(np.bincount(
            np.asarray(primal.relation.target_indices, dtype=np.int64),
            weights=np.abs(np.asarray(primal.coefficients, dtype=np.float64)),
            minlength=transfer.transfer.target_size,
        ), 1.0))
    rows = vertex_values(final, target_space, tuple(row_scales))
    field_scale = np.max(np.abs(source_values), axis=0, initial=0.0)
    conditions = np.asarray([transfer.transfer.action_condition for transfer in transfers])
    replay_bound = np.zeros_like(expected)
    for row, vertex in enumerate(np.asarray(final.vertex_global_ids, dtype=np.int64).tolist()):
        for index, name in enumerate(names):
            replay_bound[row, index] = reference.value_bounds[name][vertex][0]
    coefficient_bound = unit * conditions * rows * field_scale + replay_bound
    coefficient_defect = np.max(np.abs(actual - expected) - coefficient_bound, axis=0, initial=-np.inf)
    # Endpoint matching only, in field units: |delta| <= b entrywise and
    # M >= 0 entrywise give ||delta||_L2^2 = delta^T M delta <= b^T M b.
    coefficient_l2 = np.sqrt(np.sum(coefficient_bound * (mass @ coefficient_bound), axis=0))
    density_content = float(volumes @ actual[cells, density].mean(axis=1))
    content_defect = abs(density_content - initial_content)
    errors: list[_MappedMaterialError] = [
        {
            "field": name,
            "maximum_local_L2": np.max(reference.l2_errors[name], axis=0, initial=0.0).tolist(),
            "edits": int(reference.l2_errors[name].shape[0]),
        }
        for name in names
    ]
    # The independent floating clipping must resolve every piece of every edit.
    if not reference.coverage_ratio <= 1.0:
        raise ValueError("envelope.material_map independent pieces must cover every edit within roundoff")
    if np.any(coefficient_defect > 0.0):
        raise ValueError("envelope.material_map coefficients must match the independent replay within solve amplification")
    if any(
        np.any(reference.l2_errors[name] > material_error_tolerance)
        for name in names
    ):
        raise ValueError("envelope.material_map exceeds the declared material L2 error budget")
    if np.any(weak > 1e-12) or content_defect > 1e-12:
        raise ValueError("envelope.material_map must satisfy mapped weak equations and conserve density content")
    return {
        "density_content_defect": content_defect,
        "weak_defects": weak.tolist(),
        "source_density_content": initial_content,
        "target_density_content": density_content,
        "material_L2_errors": errors,
        "material_error_tolerance": material_error_tolerance,
        "reference_coverage_ratio": reference.coverage_ratio,
        "coefficient_excess": coefficient_defect.tolist(),
        "coefficient_bounds": np.max(coefficient_bound, axis=0, initial=0.0).tolist(),
        "coefficient_L2_bounds": coefficient_l2.tolist(),
    }


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
    source_space: phx.discretization.FiniteElementDiscretization | None = None,
    target_space: phx.discretization.FiniteElementDiscretization | None = None,
    budget: phx.discretization.CoordinateEnclosureBudget | None = None,
    material_error_tolerance: float | None = None,
    solution_inventory: Callable[
        [phx.discretization.FiniteElementDiscretization, jax.Array], float
    ] | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array], dict[str, object]]:
    """Atomically carry solved, material-density and history fields without reseeding.

    Optional transported-field spaces are caller-retained owners of the source
    and target carriers; each is admitted only if it realizes its plan bitwise.
    """
    chain = None if adaptation.transition is None else adaptation.transition.volume_material_chain
    if chain is not None:
        if material_error_tolerance is None or not np.isfinite(material_error_tolerance) or material_error_tolerance <= 0.0:
            raise ValueError("envelope.material_map requires the declared positive material L2 error budget")
        if budget is None:
            limits = adaptation.policy.limits
            with HostBatchTicket(
                limits.maximum_work_units, bytes_upper=limits.maximum_scratch_bytes,
            ) as ticket:
                ledger = phx.discretization.CoordinateEnclosureBudget(
                    ticket, maximum_work_units=limits.maximum_work_units,
                    maximum_memory_bytes=limits.maximum_scratch_bytes,
                )
                return transition_envelope_state(
                    result, adaptation, problem, values, epoch, auxiliary,
                    projection_policy=projection_policy, failure_factory=failure_factory,
                    source_space=source_space, target_space=target_space, budget=ledger,
                    solution_inventory=solution_inventory,
                    material_error_tolerance=material_error_tolerance,
                )
    semantics: dict[str, SourceRealizationFieldSemantics] | None = (
        {"u": "intensive", "density": "conservative-density", "history": "intensive"}
        if chain is not None else None
    )
    phase = _prepare_envelope_fields(
        result, problem, values, epoch, auxiliary, source_space, budget
    )
    fields, accepted = phase.fields, phase.accepted
    capture = _EnvelopeRebindCapture(records=[])
    rebinds = capture.records
    gate = phx.solver.FiniteElementTopologyTransaction(
        _EnvelopeRollbackGate(),
        fields=fields,
        projection_policy=projection_policy,
        surface_chart_semantics=semantics,
    )
    commit = phx.solver.FiniteElementTopologyTransaction(
        _finite_envelope_fields,
        fields=fields,
        composition_rebind=capture,
        projection_policy=projection_policy,
        surface_chart_semantics=semantics,
    )
    target_fields = phx.discretization.FiniteElementPlan(
        adaptation.target.mesh,
        fields,
        coordinate_spec=adaptation.target.geometry,
        budget=budget,
    ).prepare(reuse_prepared=target_space, budget=budget)
    # Spaces and field operators are immutable: both transactions bind the same
    # prepared transfer, while rejection and commit still execute separately.
    gated = gate.prepare_transition(
        accepted,
        result.mesh,
        adaptation,
        source=phase.space,
        target=target_fields,
        budget=budget,
    )
    rollback = _check_envelope_rollback(
        result, adaptation, phase, gate, gated, failure_factory
    )
    rejected = rollback.result
    mapped_reference = None
    if chain is not None:
        if budget is None or material_error_tolerance is None or semantics is None:
            raise ValueError("envelope.material_map requires its budget, error budget and semantics")
        candidate_fields = tuple(
            transfer.transfer.apply(field)
            for transfer, field in zip(gated.transfers, phase.source_fields, strict=True)
        )
        # Endpoint banks are admitted here; the replay admits every
        # edit-dependent bank (clips, pieces, local forms, new values) itself.
        nodes = max(chain.source_mesh.coordinates.shape[0], chain.target_mesh.coordinates.shape[0])
        cells = max(
            sum(block.cell_count for block in mesh.blocks)
            for mesh in (chain.source_mesh, chain.target_mesh)
        )
        with budget.temporary_scope():
            budget.reserve(0, 256 * nodes * len(fields) + 1024 * cells)
            mapped_reference = _mapped_envelope_field_reference(
                chain, semantics, phase, target_fields, candidate_fields,
                transfers=tuple(gated.transfers),
                material_error_tolerance=material_error_tolerance,
                budget=budget,
            )
    transaction = commit.execute(
        accepted,
        result.mesh,
        adaptation,
        preparation=commit.prepare_transition(
            accepted,
            result.mesh,
            adaptation,
            source=gated.source,
            target=gated.target,
            field_transfers=gated.transfers,
            budget=budget,
        ),
    )
    committed = _check_envelope_commit(transaction, rebinds, failure_factory)
    if mapped_reference is None:
        density_defect = float(jnp.max(jnp.abs(transaction.state.fields[1] - 2.0)))
        history_defect = float(
            jnp.max(jnp.abs(
                transaction.state.fields[2]
                - (target_fields.dof_maps[2].dof_coordinates[:, 0] + 0.5)
            ))
        )
    else:
        density_defect = mapped_reference["density_content_defect"]
        history_defect = mapped_reference["weak_defects"][2]
    if density_defect > 1e-12 or history_defect > 1e-12:
        raise failure_factory(
            "envelope.state.fields",
            "satisfy the declared density and material-history references",
            {"density_defect": density_defect, "history_defect": history_defect},
        )
    return (
        transaction.state.fields[0],
        (transaction.state.fields[1], transaction.state.fields[2]),
        {
            "mapped_material_reference": mapped_reference,
            "receipt_id": committed.receipt.receipt_id,
            "source_composition_id": rebinds[-1].source.composition_id,
            "target_composition_id": committed.receipt.composition.composition_id,
            "transport_accepted": list(committed.receipt.transport_accepted),
            "field_names": [field.name for field in fields],
            "density_max_defect": density_defect,
            "history_max_defect": history_defect,
            "solution_inventory": (
                None
                if solution_inventory is None
                else {
                    "source": solution_inventory(phase.space, values),
                    "target": solution_inventory(target_fields, transaction.state.fields[0]),
                }
            ),
            "auxiliary_seed_origin": "authored-initial-fields"
            if auxiliary is None
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
                "gate": "forced-rejection",
                "diagnostics": rejected.diagnostics,
                "receipt_id": rollback.receipt.receipt_id,
                "accepted_id": accepted.accepted_id,
                "topology_id": rejected.mesh.topology_id,
                "state_version": rejected.state.state_version,
                "transfer_ids": [item.transfer_id for item in rejected.transfers],
            },
        },
    )
