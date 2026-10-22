#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Moving-mesh monitoring and escalation: accept, relocate, remesh, or reject.

:class:`MeshMotionMonitor` compares a fixed-topology coordinate proposal with
its reference mesh through the host-epoch Bernstein certificate
(:func:`certify_cell_geometry_validity`), sampled cell quality, vertex
displacement, and the caller's boundary residual. :func:`advance_mesh_motion`
escalates deterministically: an accepted proposal is certified as is; a
relocation decision first untangles and optimizes the free coordinates through
:mod:`phydrax.meshing._optimization`; a proposal that still crosses a threshold
is remeshed through a metric adaptation whose transition the solver transaction
consumes. Nothing is accepted silently: every outcome carries its assessments.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from enum import StrEnum
from typing import assert_never

import equinox as eqx
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._cell_geometry as cell_geometry_owner
import phydrax.discretization._cell_geometry_validity as cell_geometry_validity_owner
import phydrax.discretization._cell_mesh as cell_mesh_owner

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._meshcore import NativeExecutionScope
from .._resource_ticket import HostBatchTicket
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_finite_float, positive_finite_float
from ..discretization._cell_geometry_transfer import (
    transition_displaced_cell_geometry,
)
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..optim import OptimizationTermination
from ..typing import checked
from ._adaptation import (
    adaptation_phase,
    execute_mesh_adaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
    region_boundary_transition,
)
from ._association import (
    BRepAssociationTransfer,
    MappedReferenceAssociationTransfer,
    PlcAssociationTransfer,
    retain_association_scopes,
    SurfaceAssociationTransfer,
)
from ._association_composition import ComposedAssociationTransfer
from ._association_state import GeometryAssociation
from ._audit import CellMeshAuditDisposition, CellMeshAuditPolicy
from ._canonical import certify_cell_mesh
from ._compartments import revalidate_region_evidence
from ._contracts import MeshingLimits
from ._failures import MeshingFailure, MeshingFailureCategory
from ._implicit_association_transfer import ImplicitAssociationTransfer
from ._lineage import CellMeshTransition, identity_lineage, inherit_mesh_organization
from ._lineage_records import MeshLineage
from ._measurements import NativeExecutionImports, NativeExecutionRecord
from ._metric import MeshMetricField
from ._native_preparation import native_preparation_ledger
from ._optimization import (
    MeshOptimizationResult,
    MeshQualityObjective,
    MeshUntanglingPolicy,
    optimize_cell_mesh,
    TargetMatrixOptimizationPlan,
)
from ._quality import evaluate_cell_quality
from ._result import CellMeshingResult
from ._scope import MeshingEntityKind, MeshingScope
from .providers._native_periodic import PeriodicAssociationTransfer


class MeshMotionDecision(StrEnum):
    """Outcome of one moving-mesh assessment or advance."""

    ACCEPT_MOTION = "accept_motion"
    RELOCATE = "relocate"
    REMESH = "remesh"
    REJECT = "reject"


def _ratio(value: float, name: str, /) -> float:
    number = positive_finite_float(value, f"MeshMotionMonitorPolicy.{name}")
    if number > 1.0:
        raise ValueError(
            f"{'MeshMotionMonitorPolicy.' + name} must lie in (0, 1]; got {number!r}."
        )
    return number


def _optional_fraction(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    return positive_finite_float(value, f"MeshMotionMonitorPolicy.{name}")


class MeshMotionMonitorPolicy(StrictModule, NonTrainableState):
    """Relocation and remesh thresholds of a moving mesh.

    Jacobian ratios compare each cell's certified determinant lower bound with
    the reference certificate; quality ratios compare each cell's mean ratio with
    its reference value. Crossing a ``relocation_*`` threshold requests
    fixed-topology relocation, crossing a ``remesh_*`` threshold requests a
    metric remesh; displacement is measured in units of the smallest reference
    cell size and ``None`` disables that criterion. A boundary residual above
    ``maximum_boundary_residual`` rejects the proposal, since neither relocation
    nor remeshing can realize a boundary the motion failed to reach.
    ``relocation_termination`` bounds the relocation solve (``None``: the
    :class:`TargetMatrixOptimizationPlan` default). A relocation whose optimizer
    did not converge is a relocation failure unless
    ``accept_valid_nonconverged_relocation`` explicitly admits its inversion-free,
    audited iterate.
    """

    relocation_jacobian_ratio: float = eqx.field(static=True)
    remesh_jacobian_ratio: float = eqx.field(static=True)
    relocation_quality_ratio: float = eqx.field(static=True)
    remesh_quality_ratio: float = eqx.field(static=True)
    relocation_displacement_fraction: float | None = eqx.field(static=True)
    remesh_displacement_fraction: float | None = eqx.field(static=True)
    maximum_boundary_residual: float = eqx.field(static=True)
    relocation_objective: MeshQualityObjective = eqx.field(static=True)
    accept_valid_nonconverged_relocation: bool = eqx.field(static=True)
    relocation_termination: OptimizationTermination | None
    validity: cell_geometry_validity_owner.CellValidityPolicy
    policy_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        relocation_jacobian_ratio: float = 0.3,
        remesh_jacobian_ratio: float = 0.05,
        relocation_quality_ratio: float = 0.5,
        remesh_quality_ratio: float = 0.2,
        relocation_displacement_fraction: float | None = None,
        remesh_displacement_fraction: float | None = None,
        maximum_boundary_residual: float = 1.0e-8,
        relocation_objective: MeshQualityObjective = MeshQualityObjective.SHAPE,
        validity: cell_geometry_validity_owner.CellValidityPolicy | None = None,
        relocation_termination: OptimizationTermination | None = None,
        accept_valid_nonconverged_relocation: bool = False,
    ) -> None:
        relocation_jacobian = _ratio(
            relocation_jacobian_ratio, "relocation_jacobian_ratio"
        )
        remesh_jacobian = _ratio(remesh_jacobian_ratio, "remesh_jacobian_ratio")
        relocation_quality = _ratio(relocation_quality_ratio, "relocation_quality_ratio")
        remesh_quality = _ratio(remesh_quality_ratio, "remesh_quality_ratio")
        relocation_displacement = _optional_fraction(
            relocation_displacement_fraction, "relocation_displacement_fraction"
        )
        remesh_displacement = _optional_fraction(
            remesh_displacement_fraction, "remesh_displacement_fraction"
        )
        if remesh_jacobian > relocation_jacobian or remesh_quality > relocation_quality:
            raise ValueError(
                "MeshMotionMonitorPolicy.remesh_jacobian_ratio must not exceed relocation_jacobian_ratio, and remesh_quality_ratio must not exceed relocation_quality_ratio; got a remesh threshold above its relocation threshold."
            )
        if (
            relocation_displacement is not None
            and remesh_displacement is not None
            and remesh_displacement < relocation_displacement
        ):
            raise ValueError(
                "MeshMotionMonitorPolicy.remesh_displacement_fraction must not be below relocation_displacement_fraction when both are set; got a remesh threshold below the relocation threshold."
            )
        residual = nonnegative_finite_float(
            maximum_boundary_residual,
            "MeshMotionMonitorPolicy.maximum_boundary_residual",
        )
        if not isinstance(relocation_objective, MeshQualityObjective):
            raise TypeError(
                "MeshMotionMonitorPolicy.relocation_objective must be a MeshQualityObjective; got a non-MeshQualityObjective value."
            )
        if relocation_objective is MeshQualityObjective.METRIC_ALIGNMENT:
            raise ValueError(
                "MeshMotionMonitorPolicy.relocation_objective must target reference shapes rather than metric alignment; got MeshQualityObjective.METRIC_ALIGNMENT."
            )
        validity_ = (
            cell_geometry_validity_owner.CellValidityPolicy()
            if validity is None
            else validity
        )
        if not isinstance(accept_valid_nonconverged_relocation, bool):
            raise TypeError(
                "MeshMotionMonitorPolicy.accept_valid_nonconverged_relocation must be bool; got a non-bool value."
            )
        self.relocation_jacobian_ratio = relocation_jacobian
        self.remesh_jacobian_ratio = remesh_jacobian
        self.relocation_quality_ratio = relocation_quality
        self.remesh_quality_ratio = remesh_quality
        self.relocation_displacement_fraction = relocation_displacement
        self.remesh_displacement_fraction = remesh_displacement
        self.maximum_boundary_residual = residual
        self.relocation_objective = relocation_objective
        self.accept_valid_nonconverged_relocation = accept_valid_nonconverged_relocation
        self.relocation_termination = relocation_termination
        self.validity = validity_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mesh-motion-monitor-policy",
                "jacobian": [relocation_jacobian, remesh_jacobian],
                "quality": [relocation_quality, remesh_quality],
                "displacement": [relocation_displacement, remesh_displacement],
                "maximum_boundary_residual": residual,
                "relocation_objective": relocation_objective.value,
                "accept_valid_nonconverged_relocation": (
                    accept_valid_nonconverged_relocation
                ),
                "relocation_termination": None
                if relocation_termination is None
                else [
                    relocation_termination.absolute_optimality,
                    relocation_termination.relative_optimality,
                    relocation_termination.absolute_step,
                    relocation_termination.relative_step,
                    relocation_termination.maximum_steps,
                    relocation_termination.maximum_evaluations,
                ],
                "validity": validity_.policy_id,
            }
        )


class MeshMotionAssessment(StrictModule, NonTrainableState):
    """Host evidence and the threshold decision for one coordinate proposal.

    ``reasons`` lists every crossed criterion in a fixed order; the certificate
    is ``None`` only for rejected non-finite proposals.
    """

    decision: MeshMotionDecision = eqx.field(static=True)
    certificate: cell_geometry_validity_owner.CellValidityCertificate | None
    minimum_jacobian_ratio: float = eqx.field(static=True)
    minimum_quality_ratio: float = eqx.field(static=True)
    maximum_displacement_fraction: float = eqx.field(static=True)
    boundary_residual: float = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    assessment_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        decision: MeshMotionDecision,
        certificate: cell_geometry_validity_owner.CellValidityCertificate | None,
        /,
        *,
        minimum_jacobian_ratio: float,
        minimum_quality_ratio: float,
        maximum_displacement_fraction: float,
        boundary_residual: float,
        reasons: tuple[str, ...],
        coordinates_id: str,
    ) -> None:
        self.decision = decision
        self.certificate = certificate
        self.minimum_jacobian_ratio = float(minimum_jacobian_ratio)
        self.minimum_quality_ratio = float(minimum_quality_ratio)
        self.maximum_displacement_fraction = float(maximum_displacement_fraction)
        self.boundary_residual = float(boundary_residual)
        self.reasons = tuple(reasons)
        self.assessment_id = canonical_fingerprint(
            {
                "kind": "mesh-motion-assessment",
                "decision": decision.value,
                "coordinates": coordinates_id,
                "certificate": None
                if certificate is None
                else certificate.certificate_id,
                "metrics": [
                    repr(self.minimum_jacobian_ratio),
                    repr(self.minimum_quality_ratio),
                    repr(self.maximum_displacement_fraction),
                    repr(self.boundary_residual),
                ],
                "reasons": list(self.reasons),
            }
        )


class MeshMotionMonitor(StrictModule, NonTrainableState):
    """Reference certificate, quality, and size scale of one moving mesh."""

    reference: cell_mesh_owner.CellMesh
    policy: MeshMotionMonitorPolicy
    reference_certificate: cell_geometry_validity_owner.CellValidityCertificate
    reference_quality: Array
    length_scale: float = eqx.field(static=True)
    monitor_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        reference: cell_mesh_owner.CellMesh,
        /,
        *,
        policy: MeshMotionMonitorPolicy | None = None,
    ) -> None:
        policy_ = MeshMotionMonitorPolicy() if policy is None else policy
        if reference.topological_dimension != reference.ambient_dimension:
            raise ValueError(
                "MeshMotionMonitor.reference must have equal topology and ambient dimensions; got topology and ambient dimensions that differ."
            )
        certificate = cell_geometry_validity_owner.certify_cell_geometry_validity(
            reference, policy=policy_.validity
        )
        if not certificate.all_certified:
            raise ValueError(
                "MeshMotionMonitor.reference must have every required validity predicate certified; got one or more uncertified predicates."
            )
        quality = evaluate_cell_quality(reference)
        measures = np.asarray(quality.measures, dtype=np.float64)
        self.reference = reference
        self.policy = policy_
        self.reference_certificate = certificate
        self.reference_quality = quality.mean_ratios
        self.length_scale = float(
            np.min(measures ** (1.0 / reference.topological_dimension))
        )
        self.monitor_id = canonical_fingerprint(
            {
                "kind": "mesh-motion-monitor",
                "reference": reference.mesh_id,
                "policy": policy_.policy_id,
            }
        )

    def moved_mesh(self, coordinates: ArrayLike, /) -> cell_mesh_owner.CellMesh:
        """The reference topology at ``coordinates`` with a content-derived version."""

        points = np.asarray(coordinates, dtype=np.float64)
        return self.reference.with_coordinates(
            points, numeric_version=f"motion:{array_tree_fingerprint(points)}"
        )

    def assess(
        self,
        coordinates: ArrayLike,
        /,
        *,
        boundary_residual: float,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> MeshMotionAssessment:
        """Certify and measure a proposal, optionally on its owning request ledger."""

        points = np.asarray(coordinates, dtype=np.float64)
        if points.shape != self.reference.coordinates.shape:
            raise ValueError(
                "MeshMotionMonitor.coordinates must have the same shape as MeshMotionMonitor.reference.coordinates; got a mismatched coordinate shape."
            )
        residual = float(boundary_residual)
        if math.isnan(residual) or residual < 0.0:
            raise ValueError(
                "MeshMotionMonitor.boundary_residual must be non-negative and not NaN; got NaN or a negative value."
            )
        coordinates_id = array_tree_fingerprint(points)
        if not np.all(np.isfinite(points)):
            return MeshMotionAssessment(
                MeshMotionDecision.REJECT,
                None,
                minimum_jacobian_ratio=-math.inf,
                minimum_quality_ratio=0.0,
                maximum_displacement_fraction=math.inf,
                boundary_residual=residual,
                reasons=("nonfinite-coordinates",),
                # ty: ignore[invalid-argument-type]
                coordinates_id=coordinates_id,
            )
        policy = self.policy
        certificate = cell_geometry_validity_owner.certify_cell_geometry_validity(
            self.moved_mesh(points), policy=policy.validity, budget=budget
        )
        jacobian = np.asarray(certificate.determinant_lower) / np.asarray(
            self.reference_certificate.determinant_lower
        )
        jacobian_ratio = float(np.min(np.where(np.isnan(jacobian), -np.inf, jacobian)))
        quality = np.asarray(
            evaluate_cell_quality(self.reference, points, budget=budget).mean_ratios
        ) / np.asarray(self.reference_quality)
        quality_ratio = float(np.min(np.where(np.isnan(quality), 0.0, quality)))
        displacement = float(
            np.max(
                np.linalg.norm(points - np.asarray(self.reference.coordinates), axis=1)
            )
            / self.length_scale
        )
        remesh = (
            ("jacobian-below-remesh", jacobian_ratio < policy.remesh_jacobian_ratio),
            ("quality-below-remesh", quality_ratio < policy.remesh_quality_ratio),
            (
                "displacement-above-remesh",
                policy.remesh_displacement_fraction is not None
                and displacement > policy.remesh_displacement_fraction,
            ),
        )
        relocation = (
            ("uncertified-cells", not certificate.all_certified),
            (
                "jacobian-below-relocation",
                jacobian_ratio < policy.relocation_jacobian_ratio,
            ),
            ("quality-below-relocation", quality_ratio < policy.relocation_quality_ratio),
            (
                "displacement-above-relocation",
                policy.relocation_displacement_fraction is not None
                and displacement > policy.relocation_displacement_fraction,
            ),
        )
        rejected = residual > policy.maximum_boundary_residual
        reasons = tuple(
            name
            for name, crossed in (
                ("boundary-residual", rejected),
                *relocation,
                *remesh,
            )
            if crossed
        )
        # Uncertified cells cannot be remeshed; untangling must come first.
        if rejected:
            decision = MeshMotionDecision.REJECT
        elif not certificate.all_certified:
            decision = MeshMotionDecision.RELOCATE
        elif any(crossed for _, crossed in remesh):
            decision = MeshMotionDecision.REMESH
        elif any(crossed for _, crossed in relocation):
            decision = MeshMotionDecision.RELOCATE
        else:
            decision = MeshMotionDecision.ACCEPT_MOTION
        return MeshMotionAssessment(
            decision,
            certificate,
            minimum_jacobian_ratio=jacobian_ratio,
            minimum_quality_ratio=quality_ratio,
            maximum_displacement_fraction=displacement,
            boundary_residual=residual,
            reasons=reasons,
            # ty: ignore[invalid-argument-type]
            coordinates_id=coordinates_id,
        )


class MeshMotionAdvance(StrictModule, NonTrainableState):
    """Executed escalation of one proposal and the result the solver commits.

    ``result`` keeps the reference topology for ``ACCEPT_MOTION`` and
    ``RELOCATE``. For ``REMESH`` it is the adapted target, whose ``transition``
    and ``adaptation.transfer`` feed the field-transfer owners; without an
    adaptation policy the remesh is only requested, ``result`` is the best valid
    same-topology mesh, and ``remesh_metric`` is the requested target metric.
    ``accepted`` is false for ``REJECT``, an unexecuted remesh request, and an
    adaptation that did not converge.

    ``execution_evidence`` is the actual ended nonperiodic policy request.
    Its totals include preparation and nested work once. An adapted target
    retains its own narrower receipt through ``adaptation``; it is not added to
    the outer totals. Earlier source receipts remain historical provenance,
    not charged preparation. Execution evidence is excluded from ``advance_id``.
    """

    decision: MeshMotionDecision = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    assessments: tuple[MeshMotionAssessment, ...]
    result: CellMeshingResult | None
    relocation: MeshOptimizationResult | None
    remesh_metric: MeshMetricField | None
    adaptation: MeshAdaptationResult | None
    transition: CellMeshTransition | None
    execution_evidence: NativeExecutionRecord | None
    advance_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        decision: MeshMotionDecision,
        assessments: tuple[MeshMotionAssessment, ...],
        /,
        *,
        result: CellMeshingResult | None = None,
        relocation: MeshOptimizationResult | None = None,
        remesh_metric: MeshMetricField | None = None,
        adaptation: MeshAdaptationResult | None = None,
        execution_evidence: NativeExecutionRecord | None = None,
    ) -> None:
        remesh = decision is MeshMotionDecision.REMESH
        if (decision is MeshMotionDecision.REJECT) != (result is None):
            raise ValueError(
                "MeshMotionAdvance.result must be None if and only if MeshMotionAdvance.decision is REJECT; got a decision/result pairing that violates the rejection bijection."
            )
        if remesh != (remesh_metric is not None) or (
            adaptation is not None and not remesh
        ):
            raise ValueError(
                "MeshMotionAdvance.remesh_metric must be present exactly for REMESH decisions, with adaptation absent for other decisions; got an inconsistent metric or adaptation state."
            )
        if execution_evidence is not None:
            execution_evidence.require_valid()
        self.decision = decision
        self.accepted = decision is not MeshMotionDecision.REJECT and (
            not remesh or (adaptation is not None and adaptation.status.converged)
        )
        self.assessments = tuple(assessments)
        self.result = result
        self.relocation = relocation
        self.remesh_metric = remesh_metric
        self.adaptation = adaptation
        self.transition = None if adaptation is None else adaptation.transition
        self.execution_evidence = execution_evidence
        self.advance_id = canonical_fingerprint(
            {
                "kind": "mesh-motion-advance",
                "decision": decision.value,
                "assessments": [value.assessment_id for value in self.assessments],
                "result": None if result is None else result.result_id,
                "relocation": None if relocation is None else relocation.optimization_id,
                "remesh_metric": None
                if remesh_metric is None
                else remesh_metric.metric_id,
                "adaptation": None if adaptation is None else adaptation.result_id,
            }
        )
        if (
            execution_evidence is not None
            and execution_evidence.owner_id != self.advance_id
        ):
            raise ValueError(
                "MeshMotionAdvance.execution_evidence must bind this advance's identity; got another request owner."
            )


def _reference_size_metric(
    reference: cell_mesh_owner.CellMesh,
    candidate: CellMeshingResult,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> MeshMetricField:
    """Isotropic metric restoring the reference cell size around every vertex."""

    dimension = reference.topological_dimension
    sizes = np.asarray(
        evaluate_cell_quality(reference, budget=budget).measures, dtype=np.float64
    ) ** (1.0 / dimension)
    totals = np.zeros((reference.coordinates.shape[0],), dtype=np.float64)
    counts = np.zeros_like(totals)
    offset = 0
    for block in reference.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        cells = np.broadcast_to(
            np.arange(offset, offset + vertices.shape[0])[:, None], vertices.shape
        )
        np.add.at(totals, vertices[valid], sizes[cells[valid]])
        np.add.at(counts, vertices[valid], 1.0)
        offset += vertices.shape[0]
    vertex_size = totals / counts
    mesh = candidate.mesh
    entities = mesh.entity_set(0)
    order = np.argsort(np.asarray(mesh.vertex_global_ids, dtype=np.int64))
    values = (vertex_size[order] ** -2)[:, None, None] * np.eye(dimension)
    return MeshMetricField(
        MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            0,
            entities.entity_set_id,
            entities.entity_ids,
        ),
        values,
        minimum_size=float(np.min(vertex_size)),
        maximum_size=float(np.max(vertex_size)),
    )


def _boundary_mask(mesh: cell_mesh_owner.CellMesh, /) -> np.ndarray:
    return np.asarray(mesh.topology.entities(0).subset("boundary").mask, dtype=np.bool_)


@contextmanager
def _motion_request(
    limits: MeshingLimits, /
) -> Iterator[tuple[NativeExecutionScope, CoordinateEnclosureBudget]]:
    """Open one request scope and ledger under the caller's adaptation limits."""
    with NativeExecutionScope(limits.native_limits()) as scope:
        initial = HostBatchTicket(0)
        budget = CoordinateEnclosureBudget(
            initial,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        initial.settle()
        with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
            yield scope, budget


def _propagated_associations(
    transfer: BRepAssociationTransfer
    | PlcAssociationTransfer
    | ComposedAssociationTransfer
    | SurfaceAssociationTransfer
    | MappedReferenceAssociationTransfer
    | ImplicitAssociationTransfer,
    source: CellMeshingResult,
    lineage: MeshLineage,
    moved: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[GeometryAssociation, ...]:
    """Revalidate fixed-topology associations on one request budget."""
    match transfer:
        case BRepAssociationTransfer():
            return transfer.propagate(source, lineage, moved, budget=budget)
        case (
            PlcAssociationTransfer()
            | ComposedAssociationTransfer()
            | MappedReferenceAssociationTransfer()
            | ImplicitAssociationTransfer()
            | SurfaceAssociationTransfer()
        ):
            return transfer.propagate(
                source, lineage, moved, geometry=geometry, budget=budget
            )
        case _ as unknown:
            assert_never(unknown)


def _carried(
    source: CellMeshingResult,
    moved: cell_mesh_owner.CellMesh,
    transfer: BRepAssociationTransfer
    | PlcAssociationTransfer
    | ComposedAssociationTransfer
    | SurfaceAssociationTransfer
    | MappedReferenceAssociationTransfer
    | ImplicitAssociationTransfer
    | PeriodicAssociationTransfer
    | None,
    /,
    *,
    limits: MeshingLimits | None = None,
    audit_policy: CellMeshAuditPolicy | None = None,
    scope: NativeExecutionScope | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> CellMeshingResult:
    """Certify moved coordinates using the source's region evidence and geometry.

    Patches, zones, and labels are rebound through the identity lineage.
    Associations, boundary models, moved coordinate maps, region evidence and
    certification are re-derived together on the supplied request ledger;
    classification evidence is not copied. Unassociated standalone motion
    retains its ordinary host contract. Uncarriable evidence refuses the motion.
    """

    if (scope is None) != (budget is None):
        raise RuntimeError(
            "_carried.scope must be supplied together with budget; got a missing request resource."
        )
    if transfer is not None and (scope is None or budget is None):
        raise RuntimeError(
            "_carried.scope and budget must own association-bearing motion; got no request resource."
        )
    phase = (
        adaptation_phase(scope, budget)
        if scope is not None and budget is not None
        else nullcontext()
    )
    with phase:
        return _carried_in_scope(
            source,
            moved,
            transfer,
            limits=limits,
            audit_policy=audit_policy,
            scope=scope,
            budget=budget,
        )


def _carried_in_scope(
    source: CellMeshingResult,
    moved: cell_mesh_owner.CellMesh,
    transfer: BRepAssociationTransfer
    | PlcAssociationTransfer
    | ComposedAssociationTransfer
    | SurfaceAssociationTransfer
    | MappedReferenceAssociationTransfer
    | ImplicitAssociationTransfer
    | PeriodicAssociationTransfer
    | None,
    /,
    *,
    limits: MeshingLimits | None,
    audit_policy: CellMeshAuditPolicy | None,
    scope: NativeExecutionScope | None,
    budget: CoordinateEnclosureBudget | None,
) -> CellMeshingResult:
    if source.attributes or (
        source.boundary is not None
        and not isinstance(transfer, ImplicitAssociationTransfer)
    ):
        raise ValueError(
            "advance_mesh_motion.source must carry no mesh attributes and must pair any boundary model with ImplicitAssociationTransfer; got mesh attributes or a boundary model without ImplicitAssociationTransfer."
        )
    if source.associations and transfer is None:
        raise ValueError(
            "advance_mesh_motion.association_transfer must revalidate retained geometry associations; got None"
        )
    lineage = identity_lineage(source.mesh, moved)
    patches, zones, labels = inherit_mesh_organization(source, moved, lineage)
    geometry = (
        cell_geometry_owner.CellGeometrySpec.affine(moved, budget=budget)
        if (
            source.geometry.exact_source is None
            and source.geometry.periodic_source is None
            and source.geometry.restriction_source is None
            and source.geometry.storage is None
            and source.geometry.geometry_layout_id
            == cell_geometry_owner.CellGeometrySpec.affine(
                source.mesh, budget=budget
            ).geometry_layout_id
            and np.array_equal(
                np.asarray(source.geometry.coordinates),
                np.asarray(source.mesh.coordinates),
            )
        )
        else transition_displaced_cell_geometry(
            source.mesh, source.geometry, moved, budget=budget
        ).geometry
    )
    boundary = None
    if source.boundary is not None and isinstance(transfer, ImplicitAssociationTransfer):
        if budget is None:
            raise RuntimeError(
                "_carried.budget must own implicit boundary remapping; got no request ledger."
            )
        boundary = transfer.remap_boundary(
            source, lineage, moved, geometry=geometry, budget=budget
        )
    certification_inputs = (
        None if source.certification is None else source.certification.request
    )
    policy = CellMeshAuditPolicy() if audit_policy is None else audit_policy
    if (
        certification_inputs is not None
        and "open_boundary" in certification_inputs.schedule.required_audit_checks
        and policy.watertight_boundary is CellMeshAuditDisposition.SKIP
    ):
        policy = CellMeshAuditPolicy(
            unused_entities=policy.unused_entities,
            require_complete_association=policy.require_complete_association,
            minimum_measure=policy.minimum_measure,
            minimum_mean_ratio=policy.minimum_mean_ratio,
            maximum_aspect_ratio=policy.maximum_aspect_ratio,
            maximum_connectivity_entries=policy.maximum_connectivity_entries,
            coincident_vertices=policy.coincident_vertices,
            coincident_vertex_tolerance=policy.coincident_vertex_tolerance,
            maximum_coincidence_candidates=policy.maximum_coincidence_candidates,
            maximum_intersection_candidates=policy.maximum_intersection_candidates,
            duplicate_cells=policy.duplicate_cells,
            nonmanifold=policy.nonmanifold,
            inconsistent_orientation=policy.inconsistent_orientation,
            watertight_boundary=CellMeshAuditDisposition.REJECT,
            self_intersection=policy.self_intersection,
            invalid_geometry=policy.invalid_geometry,
            unresolved=policy.unresolved,
            validity_policy=policy.validity_policy,
        )
    region_boundary_evidence = region_boundary_transition(
        source,
        moved,
        lineage,
        patches,
        zones,
        labels,
    )
    region_evidence = None
    if source.region_evidence is not None:
        if limits is None:
            raise ValueError(
                "advance_mesh_motion.limits must be supplied for material-bearing motion; got no adaptation work limits."
            )
        renewal = revalidate_region_evidence(
            source,
            moved,
            geometry,
            zones,
            patches,
            lineage=lineage,
            certificate_limits=source.region_evidence.certificate_limits,
            limits=limits,
            budget=budget,
        )
        zones, patches, region_evidence = (
            renewal.zones,
            renewal.patches,
            renewal.region_evidence,
        )
    if transfer is None or not source.associations:
        associations = ()
    elif isinstance(transfer, PeriodicAssociationTransfer):
        if scope is None or budget is None:
            raise RuntimeError(
                "_carried.scope must be supplied together with budget for periodic motion; got a missing request resource."
            )
        associations = transfer.propagate(
            source, lineage, moved, geometry=geometry, scope=scope, budget=budget
        )
    else:
        if budget is None:
            raise RuntimeError(
                "_carried.budget must own association revalidation; got no request ledger."
            )
        associations = _propagated_associations(
            transfer, source, lineage, moved, geometry, budget
        )
    associations = retain_association_scopes(source.associations, associations)
    return certify_cell_mesh(
        moved,
        source.coordinate_contract,
        geometry=geometry,
        audit_policy=policy,
        boundary=boundary,
        patches=patches,
        zones=zones,
        labels=labels,
        associations=associations,
        region_evidence=region_evidence,
        region_boundary_evidence=region_boundary_evidence,
        certification_inputs=certification_inputs,
        lineage=None if certification_inputs is None else lineage,
        budget=budget,
    )


def _relocate(
    monitor: MeshMotionMonitor,
    source: CellMeshingResult,
    moved: cell_mesh_owner.CellMesh,
    fixed: np.ndarray,
    numeric_version: str,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> MeshOptimizationResult:
    plan = TargetMatrixOptimizationPlan(
        moved,
        objective=monitor.policy.relocation_objective,
        target_coordinates=np.asarray(monitor.reference.coordinates),
        fixed_vertices=fixed,
        termination=monitor.policy.relocation_termination,
        untangling=MeshUntanglingPolicy(),
        accept_valid_nonconverged=monitor.policy.accept_valid_nonconverged_relocation,
    )
    return optimize_cell_mesh(
        plan,
        source.coordinate_contract,
        numeric_version=f"{numeric_version}:relocated",
        budget=budget,
    )


def _remesh(
    candidate: CellMeshingResult,
    metric: MeshMetricField,
    policy: MeshAdaptationPolicy,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> MeshAdaptationResult:
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(
            candidate, MetricMeshAdaptation(metric), policy=policy, budget=budget
        )
    )


@checked
def advance_mesh_motion(
    monitor: MeshMotionMonitor,
    source: CellMeshingResult,
    coordinates: ArrayLike,
    /,
    *,
    boundary_residual: float,
    adaptation_policy: MeshAdaptationPolicy | None = None,
    fixed_vertices: ArrayLike | None = None,
    remesh_metric: Callable[[CellMeshingResult], MeshMetricField] | None = None,
    numeric_version: str = "mesh-motion",
) -> MeshMotionAdvance:
    """Advance motion with one genuine caller-policy request.

    Policy-bearing nonperiodic assessment, coordinate preparation, boundary and
    association transport, region renewal and certification share one live
    ledger and the original limits/deadline, including nested native work.
    Earlier source execution is history, never new preparation. The ended
    request is published only after successful settlement. Periodic motion
    retains its existing request/phase semantics; policy-free ordinary motion
    retains its standalone host contract.
    """
    if adaptation_policy is None:
        return _advance_mesh_motion(
            monitor,
            source,
            coordinates,
            boundary_residual=boundary_residual,
            adaptation_policy=adaptation_policy,
            fixed_vertices=fixed_vertices,
            remesh_metric=remesh_metric,
            numeric_version=numeric_version,
            scope=None,
            budget=None,
        )
    if not isinstance(
        adaptation_policy.association_transfer, PeriodicAssociationTransfer
    ):
        return _advance_nonperiodic_motion(
            monitor,
            source,
            coordinates,
            boundary_residual=boundary_residual,
            adaptation_policy=adaptation_policy,
            fixed_vertices=fixed_vertices,
            remesh_metric=remesh_metric,
            numeric_version=numeric_version,
        )
    with _motion_request(adaptation_policy.limits) as (scope, budget):
        return _advance_mesh_motion(
            monitor,
            source,
            coordinates,
            boundary_residual=boundary_residual,
            adaptation_policy=adaptation_policy,
            fixed_vertices=fixed_vertices,
            remesh_metric=remesh_metric,
            numeric_version=numeric_version,
            scope=scope,
            budget=budget,
        )


def _advance_nonperiodic_motion(
    monitor: MeshMotionMonitor,
    source: CellMeshingResult,
    coordinates: ArrayLike,
    /,
    *,
    boundary_residual: float,
    adaptation_policy: MeshAdaptationPolicy,
    fixed_vertices: ArrayLike | None,
    remesh_metric: Callable[[CellMeshingResult], MeshMetricField] | None,
    numeric_version: str,
) -> MeshMotionAdvance:
    """Prepare and execute motion on the original request, then bind settlement."""
    history = source.execution_evidence
    with _motion_request(adaptation_policy.limits) as (scope, budget):
        if history is not None:
            ledger = native_preparation_ledger(scope)
            # A completed source request may already have settled into the
            # still-live workflow parent. It remains historical to this new
            # motion, without replaying or relabeling the parent's charge.
            if not ledger.receipt_is_accounted(scope, history):
                ledger.record_history(scope, history)
        # This is a view of the actual allowance, not a second reservation.
        # Native owners and nested requests retain headroom and share this
        # request's original deadline; all explicit science uses this ledger.
        with budget.bind_ticket(scope.ticket_view()):
            advance = _advance_mesh_motion(
                monitor,
                source,
                coordinates,
                boundary_residual=boundary_residual,
                adaptation_policy=adaptation_policy,
                fixed_vertices=fixed_vertices,
                remesh_metric=remesh_metric,
                numeric_version=numeric_version,
                scope=scope,
                budget=budget,
            )
        imports = NativeExecutionImports(scope)
        expires_at = time.monotonic() + scope.remaining().remaining_seconds
    # Nested adaptation remains reachable through advance.adaptation/result.
    # Its raw work is already in this scope, not a predecessor import
    # authorizing a cross-owner consumer receipt edge.
    record = NativeExecutionRecord(
        imports.evidence(),
        owner_id=advance.advance_id,
        imports=imports,
        history_evidence=() if history is None else (history,),
    )
    result = advance.result
    if result is not None and advance.adaptation is None:
        result = result.with_execution_evidence(record)
    publication = MeshMotionAdvance(
        advance.decision,
        advance.assessments,
        result=result,
        relocation=advance.relocation,
        remesh_metric=advance.remesh_metric,
        adaptation=advance.adaptation,
        execution_evidence=record,
    )
    if scope.parent is not None:
        native_preparation_ledger(scope.parent).remember_receipt(scope.parent, record)
    if time.monotonic() >= expires_at:
        raise MeshingFailure(
            MeshingFailureCategory.TIMED_OUT,
            "MeshMotionAdvance must finish receipt publication before its "
            "original request or earlier parent deadline",
            stage="motion-publication",
            achieved=(("remaining_seconds", expires_at - time.monotonic()),),
        )
    return publication


def _advance_mesh_motion(
    monitor: MeshMotionMonitor,
    source: CellMeshingResult,
    coordinates: ArrayLike,
    /,
    *,
    boundary_residual: float,
    adaptation_policy: MeshAdaptationPolicy | None = None,
    fixed_vertices: ArrayLike | None = None,
    remesh_metric: Callable[[CellMeshingResult], MeshMetricField] | None = None,
    numeric_version: str = "mesh-motion",
    scope: NativeExecutionScope | None,
    budget: CoordinateEnclosureBudget | None,
) -> MeshMotionAdvance:
    """Accept, relocate, remesh, or reject one moved coordinate proposal.

    ``source`` is the certified result of the monitor's reference mesh.
    ``fixed_vertices`` (default: every boundary vertex) stay bit-identical during
    relocation. A remesh targets ``remesh_metric(candidate)`` (default: the
    isotropic metric of the reference cell sizes) on the best valid
    same-topology candidate and executes through ``adaptation_policy``'s explicit
    route; without a policy the remesh is returned as an unexecuted request.
    Every published result keeps the source's patches, zones, and labels, its
    B-Rep associations revalidated by ``adaptation_policy.association_transfer``
    (required when the source has associations), and a curved coordinate map
    moved with its vertices; evidence that cannot be carried refuses the advance.
    """

    if source.mesh.mesh_id != monitor.reference.mesh_id:
        raise ValueError(
            "advance_mesh_motion.source.mesh.mesh_id must match advance_mesh_motion.monitor.reference.mesh_id; got a stale source for another mesh."
        )
    version = str(numeric_version)
    if not version:
        raise ValueError(
            "advance_mesh_motion.numeric_version must be non-empty; got an empty value."
        )
    fixed = (
        _boundary_mask(source.mesh)
        if fixed_vertices is None
        else np.asarray(fixed_vertices, dtype=np.bool_)
    )
    if fixed.shape != (source.mesh.coordinates.shape[0],):
        raise ValueError(
            "advance_mesh_motion.fixed_vertices must contain one marker for every source mesh vertex; got a shape that does not cover every source mesh vertex."
        )
    transfer = (
        None if adaptation_policy is None else adaptation_policy.association_transfer
    )
    limits = None if adaptation_policy is None else adaptation_policy.limits
    audit_policy = None if adaptation_policy is None else adaptation_policy.audit_policy
    # Periodic requests retain their existing per-carry phase contract.
    assessment_budget = (
        budget if budget is not None and not budget.ticket.settled else None
    )
    assessment = monitor.assess(
        coordinates, boundary_residual=boundary_residual, budget=assessment_budget
    )
    assessments = [assessment]
    match assessment.decision:
        case MeshMotionDecision.REJECT:
            return MeshMotionAdvance(MeshMotionDecision.REJECT, tuple(assessments))
        case MeshMotionDecision.ACCEPT_MOTION:
            return MeshMotionAdvance(
                MeshMotionDecision.ACCEPT_MOTION,
                tuple(assessments),
                result=_carried(
                    source,
                    monitor.moved_mesh(coordinates),
                    transfer,
                    limits=limits,
                    audit_policy=audit_policy,
                    scope=scope,
                    budget=budget,
                ),
            )
        case MeshMotionDecision.RELOCATE | MeshMotionDecision.REMESH:
            pass
        case decision:
            raise ValueError(
                f"advance_mesh_motion.decision must be a supported MeshMotionDecision; got {decision!r}."
            )
    # Every crossing first tries fixed-topology relocation (untangling included);
    # only an accepted relocation (converged, or explicitly admitted valid
    # non-convergence) seeds the relocated assessment.
    moved = monitor.moved_mesh(coordinates)
    relocation = _relocate(
        monitor, source, moved, fixed, version, budget=assessment_budget
    )
    if relocation.accepted:
        relocated_result = relocation.result
        if relocated_result is None:
            raise RuntimeError(
                "advance_mesh_motion.relocated_result must be present for an accepted relocation; got no certified relocation result."
            )
        relocated = monitor.assess(
            relocation.coordinates,
            boundary_residual=boundary_residual,
            budget=assessment_budget,
        )
        assessments.append(relocated)
        if relocated.decision is MeshMotionDecision.ACCEPT_MOTION:
            return MeshMotionAdvance(
                MeshMotionDecision.RELOCATE,
                tuple(assessments),
                result=_carried(
                    source,
                    relocated_result.mesh,
                    transfer,
                    limits=limits,
                    audit_policy=audit_policy,
                    scope=scope,
                    budget=budget,
                ),
                relocation=relocation,
            )
        candidate = _carried(
            source,
            relocated_result.mesh,
            transfer,
            limits=limits,
            audit_policy=audit_policy,
            scope=scope,
            budget=budget,
        )
    # ty: ignore[unresolved-attribute]
    elif assessment.certificate.all_certified:
        candidate = _carried(
            source,
            moved,
            transfer,
            limits=limits,
            audit_policy=audit_policy,
            scope=scope,
            budget=budget,
        )
    else:
        # Untangling failed: an uncertified mesh cannot seed a remesh.
        return MeshMotionAdvance(
            MeshMotionDecision.REJECT, tuple(assessments), relocation=relocation
        )
    metric = (
        _reference_size_metric(monitor.reference, candidate, budget=assessment_budget)
        if remesh_metric is None
        else remesh_metric(candidate)
    )
    if not isinstance(metric, MeshMetricField):
        raise TypeError(
            "advance_mesh_motion.remesh_metric must return MeshMetricField; got another metric result kind."
        )
    if adaptation_policy is None:
        return MeshMotionAdvance(
            MeshMotionDecision.REMESH,
            tuple(assessments),
            result=candidate,
            relocation=relocation,
            remesh_metric=metric,
        )
    adaptation = _remesh(
        candidate, metric, adaptation_policy, budget=assessment_budget
    )
    return MeshMotionAdvance(
        MeshMotionDecision.REMESH,
        tuple(assessments),
        result=adaptation.target,
        relocation=relocation,
        remesh_metric=metric,
        adaptation=adaptation,
    )


__all__ = [
    "MeshMotionAdvance",
    "MeshMotionAssessment",
    "MeshMotionDecision",
    "MeshMotionMonitor",
    "MeshMotionMonitorPolicy",
    "advance_mesh_motion",
]
