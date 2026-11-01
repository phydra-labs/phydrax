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
from collections.abc import Callable
from enum import StrEnum
from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ..discretization._cell_geometry_transition_types import CellGeometryTransition
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..optim import OptimizationTermination
from ..typing import checked, ConvertibleToArray
from ._adaptation import (
    _extend_prior_volume_material_chain,
    _identity_lineage,
    _identity_reference_maps_for_transition,
    _inherited_organization,
    _inherited_scope,
    _piecewise_linear_geometry,
    _renew_transition_certification_inputs,
    _retarget_source_boundary_chart_witness,
    _transition_geometry_record,
    execute_mesh_adaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
)
from ._audit import CellMeshManifoldScope
from ._canonical import certify_cell_mesh
from ._certification import MeshCertificationReport
from ._contracts import MeshingFailure
from ._lineage import CellMeshTransition, MeshTransitionKind
from ._metric import MeshMetricField
from ._native_construction import NativeMeshOrganization
from ._optimization import (
    MeshOptimizationResult,
    MeshQualityObjective,
    MeshUntanglingPolicy,
    optimize_cell_mesh,
    TargetMatrixOptimizationPlan,
)
from ._organization import MeshAttribute
from ._quality import evaluate_cell_quality
from ._result import CellMeshingResult
from ._scope import MeshingEntityKind, MeshingScope


class MeshMotionDecision(StrEnum):
    """Outcome of one moving-mesh assessment or advance."""

    ACCEPT_MOTION = "accept_motion"
    RELOCATE = "relocate"
    REMESH = "remesh"
    REJECT = "reject"


def _ratio(value: float, name: str, /) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1].")
    return number


def _optional_fraction(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite or None.")
    return number


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
    validity: CellValidityPolicy
    policy_id: str = eqx.field(static=True)

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
        validity: CellValidityPolicy | None = None,
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
            raise ValueError("Remesh thresholds must not exceed relocation thresholds.")
        if (
            relocation_displacement is not None
            and remesh_displacement is not None
            and remesh_displacement < relocation_displacement
        ):
            raise ValueError(
                "remesh_displacement_fraction must not precede the relocation bound."
            )
        residual = float(maximum_boundary_residual)
        if not math.isfinite(residual) or residual < 0.0:
            raise ValueError("maximum_boundary_residual must be finite and non-negative.")
        if not isinstance(relocation_objective, MeshQualityObjective):
            raise TypeError("relocation_objective must be MeshQualityObjective.")
        if relocation_objective is MeshQualityObjective.METRIC_ALIGNMENT:
            raise ValueError("Relocation targets the reference shapes, not a metric.")
        validity_ = CellValidityPolicy() if validity is None else validity
        if not isinstance(validity_, CellValidityPolicy):
            raise TypeError("validity must be CellValidityPolicy or None.")
        if relocation_termination is not None and not isinstance(
            relocation_termination, OptimizationTermination
        ):
            raise TypeError(
                "relocation_termination must be OptimizationTermination or None."
            )
        if not isinstance(accept_valid_nonconverged_relocation, bool):
            raise TypeError("accept_valid_nonconverged_relocation must be bool.")
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
    certificate: CellValidityCertificate | None
    minimum_jacobian_ratio: float = eqx.field(static=True)
    minimum_quality_ratio: float = eqx.field(static=True)
    maximum_displacement_fraction: float = eqx.field(static=True)
    boundary_residual: float = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    source_certification: MeshCertificationReport | None
    failure_detail: str | None = eqx.field(static=True)
    assessment_id: str = eqx.field(static=True)

    def __init__(
        self,
        decision: MeshMotionDecision,
        certificate: CellValidityCertificate | None,
        /,
        *,
        minimum_jacobian_ratio: float,
        minimum_quality_ratio: float,
        maximum_displacement_fraction: float,
        boundary_residual: float,
        reasons: tuple[str, ...],
        coordinates_id: str,
        source_certification: MeshCertificationReport | None = None,
        failure_detail: str | None = None,
    ) -> None:
        self.decision = decision
        self.certificate = certificate
        self.minimum_jacobian_ratio = float(minimum_jacobian_ratio)
        self.minimum_quality_ratio = float(minimum_quality_ratio)
        self.maximum_displacement_fraction = float(maximum_displacement_fraction)
        self.boundary_residual = float(boundary_residual)
        self.reasons = tuple(reasons)
        self.source_certification = source_certification
        self.failure_detail = failure_detail
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
                "source_certification": None
                if source_certification is None
                else source_certification.report_id,
                "failure_detail": failure_detail,
            }
        )


class MeshMotionMonitor(StrictModule, NonTrainableState):
    """Reference certificate, quality, and size scale of one moving mesh."""

    reference: CellMesh
    policy: MeshMotionMonitorPolicy
    reference_certificate: CellValidityCertificate
    reference_quality: Array
    length_scale: float = eqx.field(static=True)
    monitor_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        reference: CellMesh,
        /,
        *,
        policy: MeshMotionMonitorPolicy | None = None,
    ) -> None:
        policy_ = MeshMotionMonitorPolicy() if policy is None else policy
        if not isinstance(policy_, MeshMotionMonitorPolicy):
            raise TypeError("policy must be MeshMotionMonitorPolicy or None.")
        if reference.topological_dimension != reference.ambient_dimension:
            raise ValueError("Mesh motion monitoring requires full-dimensional meshes.")
        certificate = certify_cell_geometry_validity(reference, policy=policy_.validity)
        if not certificate.all_certified:
            raise ValueError("The reference mesh must be certified valid.")
        quality = evaluate_cell_quality(reference)
        measures = np.asarray(quality.measures, dtype=np.float64)
        self.reference = reference
        self.policy = policy_
        self.reference_certificate = certificate
        self.reference_quality = jnp.asarray(quality.mean_ratios)
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

    def moved_mesh(self, coordinates: ConvertibleToArray, /) -> CellMesh:
        """The reference topology at ``coordinates`` with a content-derived version."""

        points = np.asarray(coordinates, dtype=np.float64)
        return self.reference.with_coordinates(
            points, numeric_version=f"motion:{array_tree_fingerprint(points)}"
        )

    def assess(
        self, coordinates: ConvertibleToArray, /, *, boundary_residual: float
    ) -> MeshMotionAssessment:
        """Certify and measure one proposal against every policy threshold."""

        points = np.asarray(coordinates, dtype=np.float64)
        if points.shape != self.reference.coordinates.shape:
            raise ValueError("coordinates must match the reference coordinate shape.")
        residual = float(boundary_residual)
        if math.isnan(residual) or residual < 0.0:
            raise ValueError("boundary_residual must be non-negative.")
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
        certificate = certify_cell_geometry_validity(
            self.moved_mesh(points), policy=policy.validity
        )
        jacobian = np.asarray(certificate.determinant_lower) / np.asarray(
            self.reference_certificate.determinant_lower
        )
        jacobian_ratio = float(np.min(np.where(np.isnan(jacobian), -np.inf, jacobian)))
        quality = np.asarray(
            evaluate_cell_quality(self.reference, points).mean_ratios
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
    and ``adaptation.transfer`` feed the field-transfer owners only under their
    declared semantics. ``transfer`` may be absent; the accepted-field
    transaction then applies an explicit action or retains the accepted state.
    Without an adaptation policy the remesh is only requested, ``result`` is the
    best valid same-topology mesh, and ``remesh_metric`` is the requested target
    metric. ``accepted`` is false for ``REJECT``, an unexecuted remesh request,
    and an adaptation that did not converge.
    ``source_mesh_id``/``source_result_id`` bind the exact input accepted state.
    ``geometry_transition`` records source-to-motion endpoints; for remeshes,
    ``transition`` separately records the motion-candidate-to-remesh target map.
    With an explicit material-bearing `prior_transition`, `material_transition`
    is the source-to-motion edge. An executed remesh retains its distinct
    candidate-to-target `transition`; both edges bind the same declaration.
    ``local_manifold_scope`` is the declared manifold scope under which an
    owner-local target patch was certified (``None``: every resident vertex
    star was checked).
    """

    decision: MeshMotionDecision = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    source_mesh_id: str = eqx.field(static=True)
    source_result_id: str = eqx.field(static=True)
    local_source_view_id: str | None = eqx.field(static=True)
    local_target_view_id: str | None = eqx.field(static=True)
    local_target_mesh_id: str | None = eqx.field(static=True)
    local_fixed_vertex_ids: tuple[int, ...] | None = eqx.field(static=True)
    motion_driver_id: str | None = eqx.field(static=True)
    assessments: tuple[MeshMotionAssessment, ...]
    result: CellMeshingResult | None
    local_result: Any
    relocation: MeshOptimizationResult | None
    remesh_metric: MeshMetricField | None
    adaptation: MeshAdaptationResult | None
    transition: CellMeshTransition | None
    geometry_transition: CellGeometryTransition | None
    material_transition: CellMeshTransition | None
    local_manifold_scope: CellMeshManifoldScope | None
    advance_id: str = eqx.field(static=True)

    def __init__(
        self,
        decision: MeshMotionDecision,
        assessments: tuple[MeshMotionAssessment, ...],
        /,
        *,
        source_mesh_id: str,
        source_result_id: str,
        result: CellMeshingResult | None = None,
        local_result: Any = None,
        local_source_view_id: str | None = None,
        local_target_mesh_id: str | None = None,
        local_fixed_vertex_ids: ArrayLike | None = None,
        motion_driver_id: str | None = None,
        relocation: MeshOptimizationResult | None = None,
        remesh_metric: MeshMetricField | None = None,
        adaptation: MeshAdaptationResult | None = None,
        geometry_transition: CellGeometryTransition | None = None,
        material_transition: CellMeshTransition | None = None,
        local_manifold_scope: CellMeshManifoldScope | None = None,
    ) -> None:
        remesh = decision is MeshMotionDecision.REMESH
        source_mesh_id_ = str(source_mesh_id).strip()
        source_result_id_ = str(source_result_id).strip()
        if not source_mesh_id_ or not source_result_id_:
            raise ValueError(
                "Motion advances must retain their exact source mesh/result IDs."
            )
        if motion_driver_id is not None and (
            not isinstance(motion_driver_id, str) or not motion_driver_id
        ):
            raise ValueError("motion_driver_id must be a non-empty string or None.")
        local_fixed_vertex_ids_: tuple[int, ...] | None = None
        if local_fixed_vertex_ids is not None:
            fixed_ids = np.asarray(local_fixed_vertex_ids)
            if (
                fixed_ids.ndim != 1
                or fixed_ids.dtype.kind not in "iu"
                or np.unique(fixed_ids).size != fixed_ids.size
                or np.any(fixed_ids < 0)
            ):
                raise ValueError(
                    "local_fixed_vertex_ids must be unique non-negative integer vertex IDs."
                )
            local_fixed_vertex_ids_ = tuple(
                int(value) for value in np.sort(fixed_ids.astype(np.int64))
            )
        local_target_view_id = None
        local_empty = False
        if local_result is not None:
            from ..discretization._view_support import OwnerLocalCellGeometry

            if (
                not isinstance(local_result, OwnerLocalCellGeometry)
                or result is not None
                or adaptation is not None
                or remesh_metric is not None
                or not isinstance(local_source_view_id, str)
                or not local_source_view_id
                or local_source_view_id != source_result_id_
            ):
                raise ValueError(
                    "Owner-local motion requires exact source/target views and no global remesh result."
                )
            local_target_view_id = local_result.local_view_id
            local_empty = int(np.asarray(local_result.cell_ids).size) == 0
            if local_fixed_vertex_ids_ is None or not set(
                local_fixed_vertex_ids_
            ).issubset(set(int(value) for value in np.asarray(local_result.vertex_ids))):
                raise ValueError(
                    "Owner-local motion must retain the exact owner-authorized fixed-vertex request."
                )
            if local_empty:
                if local_target_mesh_id is not None or geometry_transition is not None:
                    raise ValueError(
                        "Empty local motion shards must not claim a cell transition."
                    )
            elif (
                not isinstance(local_target_mesh_id, str)
                or not local_target_mesh_id
                or geometry_transition is None
                or geometry_transition.target_mesh.mesh_id != local_target_mesh_id
            ):
                raise ValueError(
                    "Owner-local motion must bind its target view to the local geometry transition."
                )
        elif (
            local_source_view_id is not None
            or local_target_mesh_id is not None
            or motion_driver_id is not None
            or local_fixed_vertex_ids is not None
        ):
            raise ValueError(
                "Local motion identities require an owner-local target result."
            )
        if local_manifold_scope is not None and (
            not isinstance(local_manifold_scope, CellMeshManifoldScope)
            or local_result is None
            or local_empty
        ):
            raise ValueError(
                "A local manifold scope belongs to a non-empty owner-local target."
            )
        if (
            result is not None or (local_result is not None and not local_empty)
        ) and geometry_transition is None:
            raise ValueError(
                "Every non-empty accepted motion result must retain its coordinate transition."
            )
        if geometry_transition is not None:
            if (
                not isinstance(geometry_transition, CellGeometryTransition)
                or geometry_transition.source_mesh.mesh_id != source_mesh_id_
            ):
                raise ValueError(
                    "Motion coordinate transition must bind the accepted source mesh."
                )
            if adaptation is None:
                if (
                    result is not None
                    and local_result is None
                    and (
                        geometry_transition.target_mesh.mesh_id != result.mesh.mesh_id
                        or geometry_transition.target_topology_id
                        != result.mesh.topology_id
                        or geometry_transition.target_geometry_id
                        != cell_geometry_id(result.geometry)
                    )
                ):
                    raise ValueError(
                        "Motion coordinate transition must bind its returned target."
                    )
            else:
                mesh_transition = adaptation.transition
                if mesh_transition is None:
                    if result is None or (
                        geometry_transition.target_mesh.mesh_id != result.mesh.mesh_id
                        or geometry_transition.target_geometry_id
                        != cell_geometry_id(result.geometry)
                    ):
                        raise ValueError(
                            "Unchanged remesh must retain the motion candidate endpoint."
                        )
                elif (
                    mesh_transition.source_mesh_id
                    != geometry_transition.target_mesh.mesh_id
                    or mesh_transition.source_topology_id
                    != geometry_transition.target_topology_id
                    or mesh_transition.target.result_id
                    != (None if result is None else result.result_id)
                ):
                    raise ValueError(
                        "Motion and remesh transition endpoints do not compose."
                    )
        if material_transition is not None:
            if not isinstance(material_transition, CellMeshTransition):
                raise TypeError("material_transition must be CellMeshTransition or None.")
            if (
                material_transition.source_mesh_id != source_mesh_id_
                or material_transition.volume_material_chain is None
                or geometry_transition is None
                or material_transition.geometry_transition is None
                or material_transition.geometry_transition.transition_id
                != geometry_transition.transition_id
                or material_transition.target.mesh.mesh_id
                != geometry_transition.target_mesh.mesh_id
            ):
                raise ValueError(
                    "Motion material transition does not bind the source/motion endpoints."
                )
            if adaptation is None:
                if (
                    result is None
                    or material_transition.target.result_id != result.result_id
                ):
                    raise ValueError(
                        "Motion material transition target differs from its returned result."
                    )
            elif adaptation.transition is not None and (
                adaptation.transition.volume_material_chain is None
                or adaptation.transition.material_reference is None
                or material_transition.material_reference is None
                or adaptation.transition.material_reference.declaration_id
                != material_transition.material_reference.declaration_id
            ):
                raise ValueError(
                    "Motion/remesh material transitions lost their shared reference."
                )
        if (
            adaptation is not None
            and adaptation.transition is None
            and material_transition is not None
            and (
                result is None or material_transition.target.result_id != result.result_id
            )
        ):
            raise ValueError(
                "Motion material transition target differs from unchanged remesh result."
            )
        has_result = result is not None or local_result is not None
        if (decision is MeshMotionDecision.REJECT) != (not has_result):
            raise ValueError("Exactly the rejected advances carry no result.")
        if remesh != (remesh_metric is not None) or (
            adaptation is not None and not remesh
        ):
            raise ValueError("Only remesh advances carry a metric and an adaptation.")
        self.decision = decision
        self.accepted = decision is not MeshMotionDecision.REJECT and (
            not remesh or (adaptation is not None and adaptation.status.converged)
        )
        self.source_mesh_id = source_mesh_id_
        self.source_result_id = source_result_id_
        self.local_source_view_id = local_source_view_id
        self.local_target_view_id = local_target_view_id
        self.local_target_mesh_id = local_target_mesh_id
        self.local_fixed_vertex_ids = local_fixed_vertex_ids_
        self.motion_driver_id = motion_driver_id
        self.assessments = tuple(assessments)
        self.result = result
        self.local_result = local_result
        self.relocation = relocation
        self.remesh_metric = remesh_metric
        self.adaptation = adaptation
        self.transition = None if adaptation is None else adaptation.transition
        self.geometry_transition = geometry_transition
        self.material_transition = material_transition
        self.local_manifold_scope = local_manifold_scope
        self.advance_id = canonical_fingerprint(
            {
                "kind": "mesh-motion-advance",
                "source_mesh": source_mesh_id_,
                "source_result": source_result_id_,
                "local_source_view": local_source_view_id,
                "local_target_view": local_target_view_id,
                "local_target_mesh": local_target_mesh_id,
                "local_fixed_vertices": local_fixed_vertex_ids_,
                "motion_driver": motion_driver_id,
                "decision": decision.value,
                "assessments": [value.assessment_id for value in self.assessments],
                "result": (
                    None
                    if result is None and local_result is None
                    else (
                        result.result_id
                        if result is not None
                        else local_result.local_view_id
                    )
                ),
                "relocation": None if relocation is None else relocation.optimization_id,
                "remesh_metric": None
                if remesh_metric is None
                else remesh_metric.metric_id,
                "adaptation": None if adaptation is None else adaptation.result_id,
                "geometry_transition": None
                if geometry_transition is None
                else geometry_transition.transition_id,
                "material_transition": (
                    None
                    if material_transition is None
                    else material_transition.transition_id
                ),
                **(
                    {}
                    if local_manifold_scope is None
                    else {"local_manifold_scope": local_manifold_scope.scope_id}
                ),
            }
        )


def _reference_size_metric(
    reference: CellMesh, candidate: CellMeshingResult, /
) -> MeshMetricField:
    """Isotropic metric restoring the reference cell size around every vertex."""

    dimension = reference.topological_dimension
    sizes = np.asarray(evaluate_cell_quality(reference).measures, dtype=np.float64) ** (
        1.0 / dimension
    )
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


def _boundary_mask(mesh: CellMesh, /) -> np.ndarray:
    return np.asarray(mesh.topology.entities(0).subset("boundary").mask, dtype=np.bool_)


def _motion_target(
    source: CellMeshingResult,
    mesh: CellMesh,
    /,
    *,
    prior_transition: CellMeshTransition | None = None,
) -> tuple[
    CellMeshingResult | None,
    MeshCertificationReport | None,
    str | None,
    CellGeometryTransition | None,
    CellMeshTransition | None,
]:
    """Recompute source claims and optional material correspondence on one motion epoch."""
    inputs = None
    validated_witnesses = ()
    audit_policy = None
    patches = zones = labels = boundaries = ()
    attributes = ()
    transition = None
    material_transition = None
    try:
        if not _piecewise_linear_geometry(source.mesh, source.geometry):
            raise ValueError(
                "Fixed-topology motion requires an exact P1 source coordinate map."
            )
        geometry = CellGeometrySpec.affine(mesh)
        lineage = _identity_lineage(source.mesh, mesh)
        transition = _transition_geometry_record(source, mesh, geometry)
        patches, zones, labels, boundaries = _inherited_organization(
            source, mesh, lineage
        )
        attributes = tuple(
            MeshAttribute(
                attribute.name,
                attribute.role,
                _inherited_scope(attribute.scope, lineage, mesh, attribute.name),
                np.asarray(attribute.values),
                unit=attribute.unit,
            )
            for attribute in source.attributes
        )
        material_requested = prior_transition is not None and (
            prior_transition.volume_material_chain is not None
            or prior_transition.material_reference is not None
        )
        needs_reference_maps = (
            material_requested or source.layer_curving_evidence is not None
        )
        identity_maps = (
            _identity_reference_maps_for_transition(source.mesh, mesh)
            if needs_reference_maps
            else ()
        )
        layer_curving_evidence = None
        if source.layer_curving_evidence is not None:
            target_chart_witness = _retarget_source_boundary_chart_witness(
                source,
                mesh,
                geometry,
                lineage,
                source.layer_curving_evidence.chart_witness,
                reference_maps=identity_maps,
            )
            from ._layer_curving import renew_layer_curving_source_evidence

            layer_curving_evidence = renew_layer_curving_source_evidence(
                source.layer_curving_evidence,
                source.mesh,
                source.geometry,
                mesh,
                geometry,
                target_chart_witness,
                lineage,
                identity_maps,
            )
        renewed_inputs = _renew_transition_certification_inputs(
            source,
            mesh,
            geometry,
            lineage,
            reference_maps=identity_maps,
            transition=transition,
        )
        if renewed_inputs is not None:
            inputs, validated_witnesses = renewed_inputs
        audit_policy = (
            None if source.certification is None else source.certification.audit_policy
        )
        result = certify_cell_mesh(
            mesh,
            source.coordinate_contract,
            geometry=geometry,
            audit_policy=audit_policy,
            certification_inputs=inputs,
            validated_witnesses=validated_witnesses,
            boundary=source.boundary,
            patches=patches,
            zones=zones,
            labels=labels,
            attributes=attributes,
            associations=source.associations,
            surface_source=source.surface_source,
            region_boundary_evidence=boundaries,
            planar_embedding=source.planar_embedding,
            layer_curving_evidence=layer_curving_evidence,
        )
        if prior_transition is not None:
            material_chain = _extend_prior_volume_material_chain(
                prior_transition,
                source,
                result,
                lineage,
                reference_maps=identity_maps,
                coordinate_transition=transition,
            )
            if material_chain is not None:
                material_transition = CellMeshTransition(
                    source.mesh.mesh_id,
                    source.mesh.topology_id,
                    result,
                    lineage,
                    MeshTransitionKind.REPAIR,
                    geometry_transition=transition,
                    volume_material_chain=material_chain,
                )
    except (MeshingFailure, ValueError) as error:
        proof = None
        if source.certification is not None and inputs is not None:
            try:
                proof = MeshCertificationReport(
                    inputs,
                    audit_policy=audit_policy,
                    organization=NativeMeshOrganization(
                        boundary=source.boundary,
                        patches=patches,
                        zones=zones,
                        labels=labels,
                        attributes=attributes,
                        associations=source.associations,
                        surface_source=source.surface_source,
                        region_boundary_evidence=boundaries,
                    ),
                )
            except (MeshingFailure, ValueError):
                pass
        return None, proof, str(error), transition, None
    return result, result.certification, None, transition, material_transition


def _motion_assessment_with_acceptance(
    assessment: MeshMotionAssessment,
    decision: MeshMotionDecision,
    coordinates: ConvertibleToArray,
    *,
    certification: MeshCertificationReport | None,
    failure_detail: str | None,
) -> MeshMotionAssessment:
    reasons = assessment.reasons
    if failure_detail is not None:
        reasons = (*reasons, "source-acceptance")
    return MeshMotionAssessment(
        decision,
        assessment.certificate,
        minimum_jacobian_ratio=assessment.minimum_jacobian_ratio,
        minimum_quality_ratio=assessment.minimum_quality_ratio,
        maximum_displacement_fraction=assessment.maximum_displacement_fraction,
        boundary_residual=assessment.boundary_residual,
        reasons=reasons,
        coordinates_id=canonical_fingerprint(
            array_tree_fingerprint(np.asarray(coordinates, dtype=np.float64))
        ),
        source_certification=certification,
        failure_detail=failure_detail,
    )


def _relocate(
    monitor: MeshMotionMonitor,
    source: CellMeshingResult,
    moved: CellMesh,
    fixed: np.ndarray,
    numeric_version: str,
    /,
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
        plan, source.coordinate_contract, numeric_version=f"{numeric_version}:relocated"
    )


def _remesh(
    candidate: CellMeshingResult,
    metric: MeshMetricField,
    policy: MeshAdaptationPolicy,
    /,
    *,
    prior_transition: CellMeshTransition | None = None,
) -> MeshAdaptationResult:
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(
            candidate,
            MetricMeshAdaptation(metric),
            policy=policy,
            prior_transition=prior_transition,
        )
    )


def advance_mesh_motion(
    monitor: MeshMotionMonitor,
    source: CellMeshingResult,
    coordinates: ConvertibleToArray,
    /,
    *,
    boundary_residual: float,
    adaptation_policy: MeshAdaptationPolicy | None = None,
    fixed_vertices: ConvertibleToArray | None = None,
    remesh_metric: Callable[[CellMeshingResult], MeshMetricField] | None = None,
    numeric_version: str = "mesh-motion",
    prior_transition: CellMeshTransition | None = None,
) -> MeshMotionAdvance:
    """Accept, relocate, remesh, or reject one moved coordinate proposal.

    ``source`` is the certified result of the monitor's reference mesh. An
    optional `prior_transition` must target that exact result; when it carries a
    material declaration or chain, accepted motion returns a source-to-motion
    `material_transition` with a freshly computed Lagrangian map.
    `fixed_vertices` (default: every boundary vertex) stay bit-identical during
    relocation. A remesh targets `remesh_metric(candidate)` (default: the
    isotropic metric of the reference cell sizes) on the best valid
    same-topology candidate and executes through `adaptation_policy`'s explicit
    route; without a policy the remesh is returned as an unexecuted request.
    """

    if not isinstance(monitor, MeshMotionMonitor):
        raise TypeError("monitor must be MeshMotionMonitor.")
    if not isinstance(source, CellMeshingResult):
        raise TypeError("source must be CellMeshingResult.")
    if source.mesh.mesh_id != monitor.reference.mesh_id:
        raise ValueError("source must be the certified result of the monitor reference.")
    if prior_transition is not None:
        if not isinstance(prior_transition, CellMeshTransition):
            raise TypeError("prior_transition must be CellMeshTransition or None.")
        if (
            prior_transition.target.result_id != source.result_id
            or prior_transition.target.mesh.mesh_id != source.mesh.mesh_id
            or prior_transition.target.mesh.topology_id != source.mesh.topology_id
            or cell_geometry_id(prior_transition.target.geometry)
            != cell_geometry_id(source.geometry)
        ):
            raise ValueError(
                "prior_transition target must bind the exact motion source result."
            )
    if remesh_metric is not None and not callable(remesh_metric):
        raise TypeError("remesh_metric must be callable or None.")
    version = str(numeric_version)
    if not version:
        raise ValueError("numeric_version must be non-empty.")
    fixed = (
        _boundary_mask(source.mesh)
        if fixed_vertices is None
        else np.asarray(fixed_vertices, dtype=np.bool_)
    )
    if fixed.shape != (source.mesh.coordinates.shape[0],):
        raise ValueError("fixed_vertices must mark every mesh vertex.")
    assessment = monitor.assess(coordinates, boundary_residual=boundary_residual)
    assessments = [assessment]
    match assessment.decision:
        case MeshMotionDecision.REJECT:
            return MeshMotionAdvance(
                MeshMotionDecision.REJECT,
                tuple(assessments),
                source_mesh_id=source.mesh.mesh_id,
                source_result_id=source.result_id,
            )
        case MeshMotionDecision.ACCEPT_MOTION:
            mesh = monitor.moved_mesh(coordinates)
            candidate, proof, failure, motion_transition, material_transition = (
                _motion_target(source, mesh, prior_transition=prior_transition)
            )
            assessments[0] = _motion_assessment_with_acceptance(
                assessment,
                MeshMotionDecision.REJECT if failure is not None else assessment.decision,
                coordinates,
                certification=proof,
                failure_detail=failure,
            )
            if failure is not None or candidate is None:
                return MeshMotionAdvance(
                    MeshMotionDecision.REJECT,
                    tuple(assessments),
                    source_mesh_id=source.mesh.mesh_id,
                    source_result_id=source.result_id,
                    geometry_transition=motion_transition,
                )
            return MeshMotionAdvance(
                MeshMotionDecision.ACCEPT_MOTION,
                tuple(assessments),
                source_mesh_id=source.mesh.mesh_id,
                source_result_id=source.result_id,
                result=candidate,
                geometry_transition=motion_transition,
                material_transition=material_transition,
            )
        case MeshMotionDecision.RELOCATE | MeshMotionDecision.REMESH:
            pass
        case decision:
            raise ValueError(f"Unsupported mesh motion decision {decision!r}.")
    # Every crossing first tries fixed-topology relocation (untangling included);
    # only a source-accepted relocation seeds the remesh route.
    moved = monitor.moved_mesh(coordinates)
    relocation = _relocate(monitor, source, moved, fixed, version)
    if relocation.accepted:
        if relocation.result is None:
            raise ValueError("An accepted relocation must retain its target mesh result.")
        relocated_coordinates = np.asarray(relocation.coordinates, dtype=np.float64)
        relocated = monitor.assess(
            relocated_coordinates, boundary_residual=boundary_residual
        )
        candidate, proof, failure, motion_transition, material_transition = (
            _motion_target(
                source, relocation.result.mesh, prior_transition=prior_transition
            )
        )
        relocated = _motion_assessment_with_acceptance(
            relocated,
            MeshMotionDecision.REJECT if failure is not None else relocated.decision,
            relocated_coordinates,
            certification=proof,
            failure_detail=failure,
        )
        assessments.append(relocated)
        if failure is not None or candidate is None:
            return MeshMotionAdvance(
                MeshMotionDecision.REJECT,
                tuple(assessments),
                source_mesh_id=source.mesh.mesh_id,
                source_result_id=source.result_id,
                relocation=relocation,
                geometry_transition=motion_transition,
            )
        if relocated.decision is MeshMotionDecision.ACCEPT_MOTION:
            return MeshMotionAdvance(
                MeshMotionDecision.RELOCATE,
                tuple(assessments),
                source_mesh_id=source.mesh.mesh_id,
                source_result_id=source.result_id,
                result=candidate,
                relocation=relocation,
                geometry_transition=motion_transition,
                material_transition=material_transition,
            )
    elif assessment.certificate is not None and assessment.certificate.all_certified:
        candidate, proof, failure, motion_transition, material_transition = (
            _motion_target(source, moved, prior_transition=prior_transition)
        )
        assessments[0] = _motion_assessment_with_acceptance(
            assessment,
            MeshMotionDecision.REJECT if failure is not None else assessment.decision,
            coordinates,
            certification=proof,
            failure_detail=failure,
        )
        if failure is not None or candidate is None:
            return MeshMotionAdvance(
                MeshMotionDecision.REJECT,
                tuple(assessments),
                source_mesh_id=source.mesh.mesh_id,
                source_result_id=source.result_id,
                relocation=relocation,
                geometry_transition=motion_transition,
            )
    else:
        return MeshMotionAdvance(
            MeshMotionDecision.REJECT,
            tuple(assessments),
            source_mesh_id=source.mesh.mesh_id,
            source_result_id=source.result_id,
            relocation=relocation,
        )
    metric = (
        _reference_size_metric(monitor.reference, candidate)
        if remesh_metric is None
        else remesh_metric(candidate)
    )
    if not isinstance(metric, MeshMetricField):
        raise TypeError("remesh_metric must return MeshMetricField.")
    if adaptation_policy is None:
        return MeshMotionAdvance(
            MeshMotionDecision.REMESH,
            tuple(assessments),
            source_mesh_id=source.mesh.mesh_id,
            source_result_id=source.result_id,
            result=candidate,
            relocation=relocation,
            remesh_metric=metric,
            geometry_transition=motion_transition,
            material_transition=material_transition,
        )
    adaptation = _remesh(
        candidate,
        metric,
        adaptation_policy,
        prior_transition=material_transition,
    )
    return MeshMotionAdvance(
        MeshMotionDecision.REMESH,
        tuple(assessments),
        source_mesh_id=source.mesh.mesh_id,
        source_result_id=source.result_id,
        result=adaptation.target,
        relocation=relocation,
        remesh_metric=metric,
        adaptation=adaptation,
        geometry_transition=motion_transition,
        material_transition=material_transition,
    )


def _owner_local_periodic_motion_source_inputs(
    view: Any,
    execution_group: Any,
    axis_name: str,
    /,
) -> tuple[Any, Any, np.ndarray]:
    from ..discretization._exact_power_geometry import ExactPowerCellGeometrySource
    from ._periodic import PeriodicPointOrbits
    from .providers._native_periodic import NativePeriodicSource

    source = view.source_value
    authority = view.source_geometry_authority
    if (
        not isinstance(source, NativePeriodicSource)
        or not isinstance(source.domain, PeriodicPointOrbits)
        or source.source_id != view.source_id
        or source.source_revision != view.source_revision
    ):
        raise TypeError("Periodic motion view lacks its authored root-site bank.")
    if isinstance(authority, NativePeriodicSource):
        support_domain = authority.domain
        support_weights = authority.site_weights
        support_map_id = None
        support_context_id = authority.binding_id
        authority_source_id = authority.source_id
        authority_source_revision = authority.source_revision
    elif type(authority) is ExactPowerCellGeometrySource:
        context = authority.periodic_source_context
        if context is None or not isinstance(
            context.periodic_source, PeriodicPointOrbits
        ):
            raise TypeError(
                "Periodic power motion lacks its exact support source context."
            )
        support_domain = context.periodic_source
        support_weights = context.source_weights
        support_map_id = authority.source_map_id
        support_context_id = context.context_id
        authority_source_id = authority.source_id
        authority_source_revision = authority.source_revision
    else:
        raise TypeError("Periodic motion lacks its concrete support geometry authority.")
    if (
        not isinstance(support_domain, PeriodicPointOrbits)
        or authority_source_id != source.source_id
        or authority_source_revision != source.source_revision
        or support_domain.cell.cell_id != source.domain.cell.cell_id
        or support_map_id != view.source_map_id
        or support_context_id != view.source_context_id
    ):
        raise ValueError(
            "Periodic motion source/support identities differ from the current view."
        )
    source_ids = np.asarray(source.domain.point_ids, dtype=np.int64)
    source_points = np.asarray(source.domain.points, dtype=np.float64)
    support_ids = np.asarray(support_domain.point_ids, dtype=np.int64)
    support_points = np.asarray(support_domain.points, dtype=np.float64)
    if (
        source_ids.ndim != 1
        or np.unique(source_ids).size != source_ids.size
        or source_points.shape != (source_ids.size, source.domain.ambient_dimension)
        or support_ids.ndim != 1
        or np.unique(support_ids).size != support_ids.size
        or support_points.shape != (support_ids.size, support_domain.ambient_dimension)
        or source.domain.ambient_dimension != support_domain.ambient_dimension
    ):
        raise ValueError("Periodic motion authored/support source banks are malformed.")
    support_rows = {int(identifier): row for row, identifier in enumerate(support_ids)}
    try:
        matching_rows = np.asarray(
            [support_rows[int(identifier)] for identifier in source_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            "Expanded periodic support omits an authored source site."
        ) from error
    if not np.array_equal(source_points, support_points[matching_rows]):
        raise ValueError(
            "Expanded periodic support changes authored root-site coordinates."
        )
    source_weights = (
        None
        if source.site_weights is None
        else np.asarray(source.site_weights, dtype=np.float64)
    )
    expanded_weights = (
        None if support_weights is None else np.asarray(support_weights, dtype=np.float64)
    )
    if (
        source_weights is not None
        and source_weights.shape != source_ids.shape
        or expanded_weights is not None
        and expanded_weights.shape != support_ids.shape
    ):
        raise ValueError("Periodic motion authored/support site weights are malformed.")
    if (
        source_weights is not None
        and expanded_weights is not None
        and not np.array_equal(source_weights, expanded_weights[matching_rows])
    ):
        raise ValueError("Expanded periodic support changes authored site weights.")
    if source_weights is None and source_ids.size and expanded_weights is not None:
        raise ValueError("Periodic support weights lack an authored source weight bank.")
    owners = view.resolve_owner_ranks(
        source_ids,
        execution_group,
        directory_kind="source-points",
        axis_name=f"{axis_name}_authored_source_owners",
    )
    if source.point_owner_ranks is not None:
        source_owner_rows = np.asarray(source.point_owner_ranks, dtype=np.int32)
        if source_owner_rows.shape != source_ids.shape or not np.array_equal(
            source_owner_rows, owners
        ):
            raise ValueError(
                "Authored periodic source owners differ from the current directory."
            )
    return source, source.domain, owners


def _owner_local_motion_coordinate_bank(
    view: Any,
    execution_group: Any,
    local_rank: int,
    axis_name: str,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if view.periodic_topology_id is not None:
        _, domain, owners = _owner_local_periodic_motion_source_inputs(
            view, execution_group, axis_name
        )
        point_ids = np.asarray(domain.point_ids, dtype=np.int64)
        points = np.asarray(domain.points, dtype=np.float64)
        if (
            owners.shape != point_ids.shape
            or points.ndim != 2
            or points.shape[0] != point_ids.size
            or np.unique(point_ids).size != point_ids.size
        ):
            raise ValueError("Periodic authored root point rows are malformed.")
        return points, point_ids, owners
    coordinate_rows = np.asarray(view.coordinate_rows, dtype=np.int64)
    source_ids = np.asarray(view.source_point_ids, dtype=np.int64)
    source_owners = np.asarray(view.source_owner_ranks, dtype=np.int32)
    if (
        coordinate_rows.ndim != 1
        or np.any(coordinate_rows < 0)
        or np.any(coordinate_rows >= source_ids.size)
    ):
        raise ValueError("Owner-local motion coordinate rows are malformed.")
    return (
        np.asarray(view.coordinates, dtype=np.float64),
        source_ids[coordinate_rows],
        source_owners[coordinate_rows],
    )


def _owner_local_sync_coordinates(
    view: Any,
    proposed: np.ndarray,
    execution_group: Any,
    local_rank: int,
    axis_name: str,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace shared local coordinate proposals with their owner-authored values."""
    from ._distribution import _owner_exchange_message_banks, _owner_group_fail_if_any

    parts = len(execution_group.spec.process_indices)
    coordinate_ids = np.empty((0,), dtype=np.int64)
    coordinate_owners = np.empty((0,), dtype=np.int32)
    local_error: str | None = None
    try:
        coordinate_bank, coordinate_ids, coordinate_owners = (
            _owner_local_motion_coordinate_bank(
                view, execution_group, local_rank, axis_name
            )
        )
        if (
            proposed.ndim != 2
            or proposed.shape != coordinate_bank.shape
            or coordinate_owners.shape != coordinate_ids.shape
            or np.any(~np.isfinite(proposed))
        ):
            raise ValueError(
                "Owner-local motion proposal does not match its exact source coordinate bank."
            )
        if (
            np.unique(coordinate_ids).size != coordinate_ids.size
            or np.any(coordinate_owners < 0)
            or np.any(coordinate_owners >= parts)
        ):
            raise ValueError("Owner-local motion coordinate ownership is malformed.")
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "local motion proposal admission"
    )
    view.require_owner_directory_current(
        execution_group,
        axis_name=f"{axis_name}_directory",
    )
    dimension = proposed.shape[1]
    row_width = dimension + 1
    empty = np.empty((0, row_width), dtype=np.int64)
    outgoing: list[np.ndarray] = [empty.copy() for _ in range(parts)]
    proposal_bits = np.ascontiguousarray(proposed, dtype=np.float64).view(np.int64)
    for owner in range(parts):
        if owner == local_rank:
            continue
        selected = np.flatnonzero(coordinate_owners == owner)
        if selected.size:
            outgoing[owner] = np.concatenate(
                (
                    coordinate_ids[selected, None],
                    proposal_bits[selected],
                ),
                axis=1,
            )
    incoming = _owner_exchange_message_banks(
        tuple(outgoing), local_rank, execution_group, f"{axis_name}_proposal"
    )
    owner_rows = {
        int(identifier): row
        for row, identifier in enumerate(coordinate_ids)
        if coordinate_owners[row] == local_rank
    }
    responses: list[np.ndarray] = [empty.copy() for _ in range(parts)]
    local_error = None
    for requester, rows in incoming.items():
        replies = np.empty((rows.shape[0], row_width), dtype=np.int64)
        for index, message in enumerate(rows):
            identifier = int(message[0])
            row = owner_rows.get(identifier)
            if row is None:
                local_error = "Motion coordinate reached a rank without its owner row."
                replies[index] = message
                continue
            if not np.array_equal(message[1:], proposal_bits[row]):
                local_error = (
                    "Shared motion proposals disagree with their coordinate owner."
                )
            replies[index, 0] = identifier
            replies[index, 1:] = proposal_bits[row]
        responses[requester] = replies
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "shared motion proposal validation"
    )
    received = _owner_exchange_message_banks(
        tuple(responses), local_rank, execution_group, f"{axis_name}_authority"
    )
    result = np.array(proposed, dtype=np.float64, copy=True)
    reply_rows: dict[int, np.ndarray] = {}
    local_error = None
    for owner, rows in received.items():
        for message in rows:
            identifier = int(message[0])
            if identifier in reply_rows:
                local_error = "A shared coordinate received multiple owner replies."
            reply_rows[identifier] = np.asarray(message[1:], dtype=np.int64)
    for row, (identifier, owner) in enumerate(
        zip(coordinate_ids, coordinate_owners, strict=True)
    ):
        if owner == local_rank:
            continue
        reply = reply_rows.get(int(identifier))
        if reply is None:
            local_error = "A shared coordinate is missing its owner-authored proposal."
            continue
        result[row] = reply.view(np.float64)
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "shared motion owner update"
    )
    return result, coordinate_ids, coordinate_owners


def _owner_local_motion_mesh(
    view: Any,
    coordinate_bank: np.ndarray,
    fixed_vertices: np.ndarray,
    /,
) -> tuple[CellMesh, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the compact per-cell geometry used by the canonical motion monitor."""
    coordinate_rows = np.asarray(view.coordinate_rows, dtype=np.int64)
    source_point_ids = np.asarray(view.source_point_ids, dtype=np.int64)
    coordinate_source_ids = source_point_ids[coordinate_rows]
    vertex_source_ids = np.asarray(view.vertex_source_ids, dtype=np.int64)
    fixed_by_source = {}
    for row, identifier in enumerate(vertex_source_ids):
        flag = bool(fixed_vertices[row])
        previous = fixed_by_source.get(int(identifier))
        if previous is not None and previous != flag:
            raise ValueError(
                "Periodic copies of one source vertex have conflicting boundary flags."
            )
        fixed_by_source[int(identifier)] = flag

    points: list[np.ndarray] = []
    point_routes: list[int] = []
    point_shifts: list[np.ndarray] = []
    point_fixed: list[bool] = []
    point_by_key: dict[tuple[int, tuple[float, ...]], int] = {}
    standard_blocks: list[CellBlock] = []
    polygon_blocks: list[CellBlock] = []
    polyhedral_blocks: dict[str, tuple[tuple[np.ndarray, ...], ...]] = {}
    polyhedral_ids: dict[str, np.ndarray] = {}
    periodic_rank = (
        0
        if view.periodic_lift_matrix is None
        else int(np.asarray(view.periodic_lift_matrix).shape[0])
    )
    lattice = (
        None
        if view.periodic_lift_matrix is None
        else np.asarray(view.periodic_lift_matrix, dtype=np.float64)
    )
    for block in view.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int32)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        routes = np.asarray(block.coordinate_routes, dtype=np.int32)
        cell_ids = np.asarray(block.cell_ids, dtype=np.int64)
        if (
            vertices.shape != valid.shape
            or routes.shape != vertices.shape
            or cell_ids.shape != (vertices.shape[0],)
            or np.any(routes[valid] < 0)
            or np.any(routes[valid] >= coordinate_bank.shape[0])
            or np.any(vertices[valid] >= np.asarray(view.vertex_ids).size)
        ):
            raise ValueError("Owner-local motion cell routes are malformed.")
        element = block.element
        if getattr(element, "local_dof_count", routes.shape[1]) != routes.shape[1]:
            raise ValueError(
                "Owner-local motion requires exact vertex-aligned coordinate maps."
            )
        if getattr(element, "degree", 1) > 1:
            raise ValueError(
                "Owner-local mesh motion does not admit curved/high-order coordinate maps."
            )
        raw_shifts = block.cell_vertex_shifts
        shifts = None if raw_shifts is None else np.asarray(raw_shifts, dtype=np.int64)
        if shifts is not None and shifts.shape != (
            cell_ids.size,
            vertices.shape[1],
            periodic_rank,
        ):
            raise ValueError(
                "Owner-local motion winding rows do not match cell incidence."
            )
        use_cell_shifts = (
            not bool(view.coordinates_are_lifted)
            and periodic_rank > 0
            and shifts is not None
        )
        local_rows = np.full(vertices.shape, -1, dtype=np.int32)
        faces_by_cell: list[tuple[np.ndarray, ...]] = []
        for cell in range(cell_ids.size):
            cell_vertices = vertices[cell, valid[cell]]
            cell_routes = routes[cell, valid[cell]]
            cell_coordinates = np.asarray(
                coordinate_bank[cell_routes], dtype=np.float64
            ).copy()
            cell_shifts = (
                np.zeros((cell_routes.size, coordinate_bank.shape[1]), dtype=np.float64)
                if not use_cell_shifts
                else shifts[cell, valid[cell]] @ lattice
            )
            cell_coordinates += cell_shifts
            # Vertex alignment is a property of the accepted source view; the
            # proposed bank moves those same coordinate rows.
            if not np.array_equal(
                np.asarray(view.coordinates, dtype=np.float64)[cell_routes],
                np.asarray(view.vertex_coordinates, dtype=np.float64)[cell_vertices],
            ):
                raise ValueError(
                    "Owner-local motion requires exact vertex-aligned affine geometry."
                )
            cell_points = []
            for local, (route, shift, coordinate) in enumerate(
                zip(cell_routes, cell_shifts, cell_coordinates, strict=True)
            ):
                # One mesh vertex per coordinate route and lattice image, so
                # neighboring cells share their vertices as in the source view.
                key = (int(route), tuple(float(value) for value in shift))
                point = point_by_key.get(key)
                if point is None:
                    point = len(points)
                    point_by_key[key] = point
                    points.append(coordinate)
                    point_routes.append(int(route))
                    point_shifts.append(shift)
                    source_id = int(coordinate_source_ids[int(route)])
                    point_fixed.append(fixed_by_source.get(source_id, False))
                local_rows[cell, np.flatnonzero(valid[cell])[local]] = point
                cell_points.append(point)
            if block.cell_kind == "polyhedron":
                local_vertex_to_point = {
                    int(vertex_row): point
                    for vertex_row, point in zip(cell_vertices, cell_points, strict=True)
                }
                faces_by_cell.append(
                    tuple(
                        np.asarray(
                            [
                                local_vertex_to_point[int(vertex)]
                                for vertex in np.asarray(face, dtype=np.int32)
                            ],
                            dtype=np.int32,
                        )
                        for face in block.polyhedron_faces[cell]
                    )
                )
        if block.cell_kind == "polyhedron":
            polyhedral_blocks[block.name] = tuple(faces_by_cell)
            polyhedral_ids[block.name] = cell_ids
        else:
            block_ = CellBlock(
                block.name,
                block.cell_kind,
                local_rows,
                vertex_valid=valid,
                global_ids=cell_ids,
            )
            if block.cell_kind == "polygon":
                polygon_blocks.append(block_)
            else:
                standard_blocks.append(block_)
    if not points or not (standard_blocks or polygon_blocks or polyhedral_blocks):
        raise ValueError("Owner-local motion needs at least one resident cell.")
    coordinates = np.asarray(points, dtype=np.float64)
    local_vertex_ids = np.arange(coordinates.shape[0], dtype=np.int64)
    blocks = (*standard_blocks, *polygon_blocks)
    if polyhedral_blocks:
        if view.topological_dimension != 3 or polygon_blocks:
            raise ValueError("Mixed owner-local polyhedral motion is unsupported.")
        mesh = CellMesh.from_mixed_3d(
            coordinates,
            standard_blocks,
            polyhedra=polyhedral_blocks,
            vertex_global_ids=local_vertex_ids,
            polyhedral_cell_global_ids=polyhedral_ids,
        )
    else:
        mesh = CellMesh(
            coordinates,
            blocks,
            vertex_global_ids=local_vertex_ids,
        )
    return (
        mesh,
        np.asarray(point_routes, dtype=np.int32),
        np.asarray(point_shifts, dtype=np.float64),
        np.asarray(point_fixed, dtype=np.bool_),
        coordinate_source_ids,
    )


class _OwnerLocalPeriodicMotionRefresh(NamedTuple):
    source_entry: Any
    source_composition: Any
    source_content_id: str
    candidate: dict[str, Any]
    final_source: Any
    source_authority_value: Any
    source_map: Any
    route: str
    source_context_id: str
    exact_root_coordinates: tuple[Any, ...]
    specification_id: str


def _prepare_owner_local_periodic_motion_target(
    view: Any,
    specification: Any,
    coordinates: np.ndarray,
    coordinate_ids: np.ndarray,
    coordinate_owners: np.ndarray,
    execution_group: Any,
    local_rank: int,
    axis_name: str,
    /,
) -> tuple[Any, Any] | None:
    if view.periodic_topology_id is None:
        return None
    from fractions import Fraction

    from ..discretization._exact_power_geometry import (
        ExactPowerCellGeometrySource,
    )
    from ..discretization._view_support import (
        _materialize_owner_local_periodic_target,
    )
    from ._distribution import _owner_group_fail_if_any
    from ._periodic import PeriodicPointOrbits
    from .providers._native_periodic import (
        _owner_local_periodic_source_authority,
        _refresh_owner_local_periodic_delaunay_source,
        NativePeriodicSource,
        VolumeMeshingSpec,
    )
    from .providers._native_periodic_power import (
        _owner_local_periodic_power_candidate,
    )

    authority = view.source_geometry_authority
    route = None
    source_shard = None
    source_map = None
    source_error: Exception | None = None
    source_error_text: str | None = None
    try:
        from .providers._native_periodic import (
            SurfaceMeshingSpec,
            VolumeFillStrategy,
            VolumeMeshingSpec,
        )

        if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
            raise TypeError("Periodic motion requires its exact typed specification.")
        specification_id = specification.specification_id
        if specification_id != view.specification_id or not isinstance(
            authority, (NativePeriodicSource, ExactPowerCellGeometrySource)
        ):
            raise ValueError("Periodic motion source or specification is stale.")
        power_request = (
            isinstance(specification, VolumeMeshingSpec)
            and specification.fill_strategy is VolumeFillStrategy.POLYHEDRAL
            and set(specification.target.cell_families.required) == {"polyhedron"}
        )
        if type(authority) is ExactPowerCellGeometrySource:
            route = "periodic_power"
            if not power_request:
                raise ValueError(
                    "An exact power map requires its typed polyhedral power specification."
                )
        else:
            route = "periodic_power" if power_request else "periodic_delaunay"
            if power_request and (view.source_map_id is not None or view.vertex_ids.size):
                raise ValueError(
                    "A nonempty periodic power view lacks its exact power map."
                )
        authored_source, domain, point_owners = (
            _owner_local_periodic_motion_source_inputs(
                view, execution_group, f"{axis_name}_authored_root"
            )
        )
        point_ids = np.asarray(domain.point_ids, dtype=np.int64)
        points = np.asarray(domain.points, dtype=np.float64).copy()
        if (
            point_owners.shape != point_ids.shape
            or points.shape != (point_ids.size, domain.ambient_dimension)
            or coordinate_ids.shape != coordinates.shape[:1]
            or coordinate_owners.shape != coordinate_ids.shape
        ):
            raise ValueError("Periodic motion root/proposal banks are misaligned.")
        point_rows = {int(identifier): row for row, identifier in enumerate(point_ids)}
        if route == "periodic_delaunay":
            for row, (identifier, owner) in enumerate(
                zip(coordinate_ids, coordinate_owners, strict=True)
            ):
                point_row = point_rows.get(int(identifier))
                if point_row is not None and int(owner) == local_rank:
                    points[point_row] = coordinates[row]
            owned_rows = np.flatnonzero(point_owners == local_rank)
            local_domain = PeriodicPointOrbits(
                domain.cell,
                points[owned_rows],
                point_ids[owned_rows],
            )
            local_weights = (
                None
                if authored_source.site_weights is None
                else np.asarray(authored_source.site_weights, dtype=np.float64)[
                    np.flatnonzero(point_owners == local_rank)
                ]
            )
            source_shard = NativePeriodicSource(
                local_domain,
                authored_source.source_id,
                authored_source.source_revision,
                point_owner_ranks=point_owners[owned_rows],
                site_weights=local_weights,
            )
        else:
            if (
                coordinate_ids.shape != point_ids.shape
                or set(int(value) for value in coordinate_ids)
                != set(int(value) for value in point_ids)
                or coordinates.shape != points.shape
            ):
                raise ValueError("Power motion root/proposal banks are misaligned.")
            if authored_source.site_weights is None:
                if point_ids.size:
                    raise ValueError(
                        "Power motion requires its exact authored site weights."
                    )
                weights = np.empty((0,), dtype=np.float64)
            else:
                weights = np.asarray(authored_source.site_weights, dtype=np.float64)
            source_id = authored_source.source_id
            source_revision = authored_source.source_revision
            if weights.shape != point_ids.shape:
                raise ValueError("Power motion exact source weights are misaligned.")
            for row, (identifier, owner) in enumerate(
                zip(coordinate_ids, coordinate_owners, strict=True)
            ):
                point_row = point_rows[int(identifier)]
                if int(owner) == local_rank:
                    points[point_row] = coordinates[row]
            source_shard = NativePeriodicSource(
                PeriodicPointOrbits(domain.cell, points, point_ids),
                source_id,
                source_revision,
                point_owner_ranks=point_owners,
                site_weights=weights,
            )
        # The native power refresh below creates a fresh concrete target map.
    except Exception as error:
        source_error = error
        source_error_text = f"{type(error).__name__}: {error}"
    try:
        _owner_group_fail_if_any(
            source_error_text,
            execution_group,
            f"{axis_name}_source_preflight",
            "periodic motion source rows",
        )
    except Exception:
        if source_error is not None:
            raise ValueError(
                "Owner-local periodic motion source refresh failed: " + str(source_error)
            ) from source_error
        raise
    if source_shard is None or route is None:
        raise ValueError("Periodic source preflight accepted no typed source.")

    if route == "periodic_delaunay":
        refreshed_source = _refresh_owner_local_periodic_delaunay_source(
            source_shard,
            execution_group,
            local_rank,
            axis_name=f"{axis_name}_delaunay_refresh",
        )
        final_source = refreshed_source.final_source
        if not isinstance(final_source.domain, PeriodicPointOrbits):
            raise TypeError("Refreshed Delaunay source lost its periodic point bank.")
        final_domain = final_source.domain
        source_entry = refreshed_source.source_entry
        source_composition = refreshed_source.source_composition
        source_content_id = refreshed_source.source_content_id
        candidate = refreshed_source.candidate
        exact_root_coordinates = refreshed_source.exact_root_coordinates
        source_authority_value = final_source
        source_map = None
        source_context_id = final_source.binding_id
    else:
        (
            source_entry,
            source_composition,
            source_content_id,
        ) = _owner_local_periodic_source_authority(
            source_shard,
            execution_group,
            f"{axis_name}_power_source",
        )
        power = _owner_local_periodic_power_candidate(
            source_shard,
            local_rank,
            execution_group,
            source_content_id,
            axis_name=f"{axis_name}_power_refresh",
        )
        candidate = power["candidate"]
        source_map = power["source_map"]
        if not isinstance(source_shard, NativePeriodicSource):
            raise TypeError("Power motion source shard is not NativePeriodicSource.")
        final_source = source_shard
        if not isinstance(final_source.domain, PeriodicPointOrbits):
            raise TypeError("Power motion lost its typed periodic point bank.")
        final_domain = final_source.domain
        if source_map is None:
            source_authority_value = source_shard
            source_context_id = source_shard.binding_id
        else:
            power_context = source_map.periodic_source_context
            if power_context is None:
                raise ValueError("Periodic power motion lacks its exact source context.")
            source_authority_value = source_map
            source_context_id = power_context.context_id
        point_ids = np.asarray(final_domain.point_ids, dtype=np.int64)
        points = np.asarray(final_domain.points, dtype=np.float64)
        point_order = np.argsort(point_ids, kind="stable")
        exact_root_coordinates = tuple(
            (
                int(point_ids[row]),
                tuple(Fraction.from_float(float(value)) for value in points[row]),
            )
            for row in point_order
        )

    refreshed = _OwnerLocalPeriodicMotionRefresh(
        source_entry,
        source_composition,
        source_content_id,
        candidate,
        final_source,
        source_authority_value,
        source_map,
        route,
        source_context_id,
        exact_root_coordinates,
        specification_id,
    )
    target_root_ids = np.asarray(final_domain.point_ids, dtype=np.int64)
    target_root_coordinates = np.asarray(final_domain.points, dtype=np.float64)
    root_order = np.argsort(target_root_ids, kind="stable")
    target_source_entity_set_id = final_source.scope_entity_set_id(0)
    local_mesh, local_geometry = view.reconstruct_local_geometry()
    if local_mesh is not None and local_geometry is None:
        raise ValueError("Periodic motion source mesh lost its exact geometry.")
    target = _materialize_owner_local_periodic_target(
        view,
        final_domain.cell,
        exact_root_coordinates,
        target_root_coordinates[root_order],
        target_source_entity_set_id,
        np.asarray(view.vertex_source_ids, dtype=np.int64),
        np.asarray(view.vertex_absolute_windings, dtype=np.int64),
        source_map=source_map,
    )
    return refreshed, target


def _periodic_motion_block_records(
    source: Any, materialization: Any, /
) -> tuple[
    Any,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Any,
    tuple[Any, ...],
    np.ndarray,
    list[dict[str, Any]],
]:
    """Validate one periodic motion materialization against its source view rows."""
    target_mesh = materialization.mesh
    source_cell_ids = np.asarray(source.cell_ids, dtype=np.int64)
    source_vertex_ids = np.asarray(source.vertex_ids, dtype=np.int64)
    cell_owner_ranks = np.asarray(source.cell_owner_ranks, dtype=np.int32)
    vertex_owner_ranks = np.asarray(source.vertex_owner_ranks, dtype=np.int32)
    owned = np.asarray(source.owned, dtype=np.bool_)
    vertex_coordinates = np.asarray(materialization.vertex_coordinates, dtype=np.float64)
    exact_vertex_coordinates = materialization.exact_vertex_coordinates
    blocks = tuple(materialization.blocks)
    if (
        vertex_coordinates.shape != np.asarray(source.vertex_coordinates).shape
        or not np.array_equal(
            np.asarray(materialization.vertex_source_ids, dtype=np.int64),
            np.asarray(source.vertex_source_ids, dtype=np.int64),
        )
        or not np.array_equal(
            np.asarray(materialization.vertex_absolute_windings, dtype=np.int64),
            np.asarray(source.vertex_absolute_windings, dtype=np.int64),
        )
    ):
        raise ValueError(
            "Periodic motion materialization changed quotient vertex ancestry."
        )
    if source_cell_ids.size and (target_mesh is None or not blocks):
        raise ValueError("Nonempty periodic motion lost its materialized cell mesh.")
    if target_mesh is not None and (
        not np.array_equal(
            np.asarray(target_mesh.vertex_global_ids, dtype=np.int64),
            source_vertex_ids,
        )
        or not np.array_equal(
            np.asarray(
                target_mesh.entity_set(source.topological_dimension).entity_ids,
                dtype=np.int64,
            ),
            source_cell_ids,
        )
    ):
        raise ValueError(
            "Periodic motion materialization changed stable local topology IDs."
        )

    exact_cell_measures = tuple(materialization.exact_cell_measures)
    if len(exact_cell_measures) != source_cell_ids.size:
        raise ValueError("Periodic exact motion measures omit local cells.")
    cell_measures = np.asarray(
        [float(value) for value in exact_cell_measures], dtype=np.float64
    )
    block_records: list[dict[str, Any]] = []
    for block in blocks:
        block_cell_ids = np.asarray(block.cell_ids, dtype=np.int64)
        record: dict[str, Any] = {
            "name": block.name,
            "cell_kind": block.cell_kind,
            "cell_rows": np.asarray(block.cell_rows, dtype=np.int64),
            "cell_ids": block_cell_ids,
            "source_cell_ids": np.asarray(block.source_cell_ids, dtype=np.int64),
            "vertices": np.asarray(block.vertices, dtype=np.int32),
            "vertex_valid": np.asarray(block.vertex_valid, dtype=np.bool_),
            "coordinate_routes": np.asarray(block.coordinate_routes, dtype=np.int32),
            "element": block.element,
            "element_rows": np.asarray(block.element_rows, dtype=np.int32),
            "chart_arguments": block.chart_arguments,
            "cell_measures": np.asarray(block.cell_measures, dtype=np.float64),
            "cell_vertex_shifts": (
                None
                if block.cell_vertex_shifts is None
                else np.asarray(block.cell_vertex_shifts, dtype=np.int64)
            ),
        }
        if block.cell_kind == "polyhedron":
            face_ids = np.asarray(block.polyhedron_face_ids, dtype=np.int64)
            face_loops = tuple(
                np.asarray(loop, dtype=np.int32)
                for loop in block.polyhedron_face_vertex_loops
            )
            face_index = {
                tuple(sorted(int(value) for value in loop)): index
                for index, loop in enumerate(face_loops)
            }
            face_offsets = np.zeros((len(face_loops) + 1,), dtype=np.int64)
            for index, loop in enumerate(face_loops):
                face_offsets[index + 1] = face_offsets[index] + loop.size
            face_values = (
                np.concatenate(face_loops).astype(np.int32, copy=False)
                if face_loops
                else np.empty((0,), dtype=np.int32)
            )
            cell_face_offsets = np.zeros((block_cell_ids.size + 1,), dtype=np.int64)
            cell_face_values: list[int] = []
            cell_face_signs: list[int] = []
            for cell, cell_faces in enumerate(block.polyhedron_faces):
                signs = np.asarray(block.polyhedron_face_signs[cell], dtype=np.int8)
                if signs.shape != (len(cell_faces),):
                    raise ValueError("Periodic power face signs differ from face loops.")
                for face, sign in zip(cell_faces, signs, strict=True):
                    key = tuple(sorted(int(value) for value in np.asarray(face)))
                    face_id = face_index.get(key)
                    if face_id is None:
                        raise ValueError("Periodic power motion lost an authored face.")
                    cell_face_values.append(face_id)
                    cell_face_signs.append(int(sign))
                cell_face_offsets[cell + 1] = len(cell_face_values)
            record.update(
                {
                    "face_ids": face_ids,
                    "cell_face_offsets": cell_face_offsets,
                    "cell_face_values": np.asarray(cell_face_values, dtype=np.int64),
                    "cell_face_signs": np.asarray(cell_face_signs, dtype=np.int8),
                    "face_vertex_offsets": face_offsets,
                    "face_vertex_values": face_values,
                }
            )
        block_records.append(record)
    if target_mesh is not None and not block_records:
        raise ValueError("Periodic target mesh has no owner-local cell blocks.")
    return (
        target_mesh,
        source_cell_ids,
        source_vertex_ids,
        cell_owner_ranks,
        vertex_owner_ranks,
        owned,
        vertex_coordinates,
        exact_vertex_coordinates,
        exact_cell_measures,
        cell_measures,
        block_records,
    )


def _owner_local_periodic_motion_view(
    source: Any,
    specification: Any,
    periodic_target: tuple[Any, Any],
    candidate_revision_id: str,
    periodic_margin: float,
    periodic_shell_certificate: Any,
    support_fractional_lower: np.ndarray | None,
    support_fractional_upper: np.ndarray | None,
    execution_group: Any,
    local_rank: int,
    axis_name: str,
    /,
) -> Any:
    from ..discretization._view_support import OwnerLocalCellGeometry
    from ._distribution import _owner_group_fail_if_any
    from .providers._native_periodic import (
        _owner_local_periodic_assembly_from_state,
        _owner_local_periodic_mesh_authority,
        SurfaceMeshingSpec,
        VolumeMeshingSpec,
    )

    specification_id = None
    specification_error: str | None = None
    try:
        if not isinstance(specification, (SurfaceMeshingSpec, VolumeMeshingSpec)):
            raise TypeError(
                "Periodic motion requires its exact typed meshing specification."
            )
        specification_id = specification.specification_id
        if specification_id != source.specification_id:
            raise ValueError(
                "Periodic motion specification differs from its source view."
            )
    except Exception as error:
        specification_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        specification_error,
        execution_group,
        f"{axis_name}_specification",
        "periodic motion specification identity",
    )
    if specification_id is None:
        raise ValueError("Periodic motion specification preflight accepted no identity.")

    refreshed, materialization = periodic_target
    final_source = refreshed.final_source
    source_authority_value = refreshed.source_authority_value
    block_error: str | None = None
    try:
        (
            target_mesh,
            source_cell_ids,
            source_vertex_ids,
            cell_owner_ranks,
            vertex_owner_ranks,
            owned,
            vertex_coordinates,
            exact_vertex_coordinates,
            exact_cell_measures,
            cell_measures,
            block_records,
        ) = _periodic_motion_block_records(source, materialization)
    except Exception as error:
        block_error = f"{type(error).__name__}: {error}"
    # Local refusals join the group before its next collective.
    _owner_group_fail_if_any(
        block_error,
        execution_group,
        f"{axis_name}_materialization",
        "periodic motion materialization",
    )

    point_ids = np.asarray(refreshed.candidate["point_ids"], dtype=np.int64)
    coordinate_error: str | None = None
    try:
        # Periodic coordinate rows address the vertex bank. A coordinate whose
        # quotient root is an authored site moves with that site; an adapted
        # (self-keyed) root moves only through the exact coordinate ancestry.
        coordinate_rows = np.asarray(source.coordinate_rows, dtype=np.int64)
        roots = np.asarray(source.vertex_source_ids, dtype=np.int64)[coordinate_rows]
        if coordinate_rows.shape != vertex_coordinates.shape[:1]:
            raise ValueError(
                "Periodic motion coordinate routes do not match target vertices."
            )
        if (
            not np.all(np.isin(roots, point_ids))
            and materialization.source_coordinate_stencil is None
        ):
            raise ValueError(
                "Periodic motion coordinates lack an authored site or exact ancestry."
            )
    except Exception as error:
        coordinate_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        coordinate_error,
        execution_group,
        f"{axis_name}_coordinate_routes",
        "periodic motion coordinate routes",
    )

    if target_mesh is None:
        vertices = np.empty((0, source.topological_dimension + 1), dtype=np.int32)
        vertex_valid = np.empty(vertices.shape, dtype=np.bool_)
    else:
        vertices = np.asarray(block_records[0]["vertices"], dtype=np.int32)
        vertex_valid = np.asarray(block_records[0]["vertex_valid"], dtype=np.bool_)
    mesh_data: dict[str, Any] = {
        "cell_ids": source_cell_ids,
        "cell_owner_ranks": cell_owner_ranks,
        "owned": owned,
        "cell_measures": cell_measures,
        "exact_cell_measures": exact_cell_measures,
        "support_cell_mask": np.asarray(source.support_cell_mask, dtype=np.bool_),
        "vertices": vertices,
        "vertex_valid": vertex_valid,
        "vertex_ids": source_vertex_ids,
        "vertex_owner_ranks": vertex_owner_ranks,
        "vertex_source_ids": np.asarray(
            materialization.vertex_source_ids, dtype=np.int64
        ),
        "vertex_absolute_windings": np.asarray(
            materialization.vertex_absolute_windings, dtype=np.int64
        ),
        "vertex_coordinates": vertex_coordinates,
        "exact_vertex_coordinates": exact_vertex_coordinates,
        "cell_key_offsets": np.asarray(materialization.cell_key_offsets, dtype=np.int64),
        "cell_key_source_ids": np.asarray(
            materialization.cell_key_source_ids, dtype=np.int64
        ),
        "cell_key_relative_windings": np.asarray(
            materialization.cell_key_relative_windings, dtype=np.int64
        ),
        "cell_windings": np.asarray(materialization.cell_windings, dtype=np.int64),
        "blocks": tuple(block_records),
        # Fixed topology: the row, entity and coordinate-route layout of the
        # source view carries over; only geometry and its ancestry move.
        "cell_rows": np.arange(source_cell_ids.size, dtype=np.int64),
        "vertex_rows": np.asarray(source.vertex_rows, dtype=np.int64),
        "coordinate_rows": np.asarray(source.coordinate_rows, dtype=np.int64),
        "coordinates": vertex_coordinates.copy(),
        "exact_coordinates": exact_vertex_coordinates,
        "entity_set_ids": dict(source.entity_set_ids),
        "local_entity_ids": dict(source.local_entity_ids),
        "entity_vertex_offsets": dict(source.entity_vertex_offsets),
        "entity_vertex_values": dict(source.entity_vertex_values),
        "periodic_cell_ids": np.asarray(source.periodic_cell_ids, dtype=np.int64),
        "vertex_representative_ids": np.asarray(
            materialization.vertex_source_ids, dtype=np.int64
        ).copy(),
        "vertex_windings": np.asarray(
            materialization.vertex_absolute_windings, dtype=np.int64
        ).copy(),
        # The periodic simplex layout is a group-wide identity (cell kind and
        # element), unchanged by a fixed-topology motion.
        "geometry_layout_id": source.geometry_layout_id,
        "support_fractional_lower": support_fractional_lower,
        "support_fractional_upper": support_fractional_upper,
        "source_coordinate_stencil": materialization.source_coordinate_stencil,
        "exact_vertex_coordinate_offsets": (
            materialization.exact_vertex_coordinate_offsets
        ),
        "local_mesh": target_mesh,
    }
    if refreshed.route == "periodic_delaunay":
        dimension = source.topological_dimension
        windings = np.asarray(materialization.vertex_absolute_windings, dtype=np.int64)
        mesh_data["cell_kind"] = "triangle" if dimension == 2 else "tetrahedron"
        mesh_data["cell_vertex_shifts"] = (
            windings[vertices]
            if vertices.size
            else np.zeros((0, dimension + 1, dimension), dtype=np.int64)
        )
    source_map = materialization.source_map
    source_map_id = None if source_map is None else source_map.source_map_id
    candidate = dict(refreshed.candidate)
    candidate["candidate_revision_id"] = candidate_revision_id
    candidate["periodic_shell_certificate"] = dict(periodic_shell_certificate)
    mesh_authority = _owner_local_periodic_mesh_authority(
        mesh_data,
        final_source,
        candidate,
        refreshed.source_entry,
        refreshed.source_content_id,
        source.coordinate_contract_id,
        specification.specification_id,
        refreshed.route,
        source_map_id,
        refreshed.source_context_id,
        local_rank,
        execution_group,
        f"{axis_name}_authority",
    )
    target_assembly = _owner_local_periodic_assembly_from_state(
        final_source,
        specification,
        source.coordinate_contract,
        refreshed.source_entry,
        refreshed.source_composition,
        source_authority_value,
        refreshed.route,
        candidate,
        mesh_data,
        mesh_authority,
        source_map_id,
        refreshed.source_context_id,
        local_rank,
        execution_group,
        materialization.source_coordinate_stencil,
        materialization.exact_vertex_coordinate_offsets,
        None,
        None,
        axis_name=f"{axis_name}_publication",
    )
    matches = tuple(
        part for part in target_assembly.parts if part.name == source.part_name
    )
    if len(matches) != 1 or not isinstance(matches[0].carrier, OwnerLocalCellGeometry):
        raise ValueError(
            "Periodic motion materializer returned no exact local target view."
        )
    return matches[0].carrier


def _owner_local_motion_view(
    source: Any,
    coordinates: np.ndarray,
    coordinate_ids: np.ndarray,
    coordinate_owners: np.ndarray,
    measure_by_id: dict[int, float],
    candidate_revision_id: str,
    periodic_margin: float | None,
    periodic_shell_certificate: Any,
    support_fractional_lower: np.ndarray | None,
    support_fractional_upper: np.ndarray | None,
    /,
    *,
    periodic_target: tuple[Any, Any] | None = None,
    periodic_specification: Any | None = None,
    execution_group: Any = None,
    local_rank: int | None = None,
    axis_name: str = "owner_local_motion_target",
) -> Any:
    if periodic_target is not None:
        if execution_group is None or local_rank is None:
            raise ValueError("Periodic motion target authority requires its owner group.")
        if periodic_margin is None:
            raise ValueError("Periodic motion requires its certified shell margin.")
        return _owner_local_periodic_motion_view(
            source,
            periodic_specification,
            periodic_target,
            candidate_revision_id,
            float(periodic_margin),
            periodic_shell_certificate,
            support_fractional_lower,
            support_fractional_upper,
            execution_group,
            local_rank,
            axis_name,
        )
    from fractions import Fraction

    from ..discretization._view_support import OwnerLocalCellGeometry
    from ..lifecycle import Composition, CompositionEntry
    from ._assembly import MeshPartManifest
    from ._distribution import (
        _owner_group_allgather,
        _owner_local_multiset_identity,
        OwnerLocalCoordinateMap,
        ShardedOwnerDirectoryWitness,
    )

    if execution_group is None or local_rank is None:
        raise ValueError("Owner-local motion target geometry requires its current group.")
    source_cell_ids = np.asarray(source.cell_ids, dtype=np.int64)
    source_vertex_ids = np.asarray(source.vertex_ids, dtype=np.int64)
    coordinate_rows = np.asarray(source.coordinate_rows, dtype=np.int64)
    source_point_ids = np.asarray(source.source_point_ids, dtype=np.int64)
    coordinate_source_ids = source_point_ids[coordinate_rows]
    target_coordinate_ids = np.asarray(coordinate_ids, dtype=np.int64)
    target_coordinate_owners = np.asarray(coordinate_owners, dtype=np.int32)
    target_coordinate_bank = np.asarray(coordinates, dtype=np.float64)
    if (
        target_coordinate_ids.ndim != 1
        or np.unique(target_coordinate_ids).size != target_coordinate_ids.size
        or target_coordinate_owners.shape != target_coordinate_ids.shape
        or target_coordinate_bank.ndim != 2
        or target_coordinate_bank.shape[0] != target_coordinate_ids.size
        or np.any(~np.isfinite(target_coordinate_bank))
        or np.any(target_coordinate_owners < 0)
        or np.any(target_coordinate_owners >= len(execution_group.spec.process_indices))
    ):
        raise ValueError("Owner-local motion target coordinate rows are malformed.")
    current_coordinate_owners = source.resolve_owner_ranks(
        target_coordinate_ids,
        execution_group,
        directory_kind="source-points",
        axis_name=f"{axis_name}_target_source_owners",
    )
    if not np.array_equal(target_coordinate_owners, current_coordinate_owners):
        raise ValueError("Owner-local motion target source ownership changed.")
    coordinate_row_by_source = {
        int(identifier): row for row, identifier in enumerate(target_coordinate_ids)
    }
    if any(
        int(identifier) not in coordinate_row_by_source
        for identifier in (*coordinate_source_ids, *np.asarray(source.vertex_source_ids))
    ):
        raise ValueError("Moved owner-local vertices lack exact source coordinate rows.")
    local_coordinate_order = np.asarray(
        [
            coordinate_row_by_source[int(identifier)]
            for identifier in coordinate_source_ids
        ],
        dtype=np.int64,
    )
    local_coordinates = target_coordinate_bank[local_coordinate_order]
    vertex_source_ids = np.asarray(source.vertex_source_ids, dtype=np.int64)
    vertex_coordinate_order = np.asarray(
        [coordinate_row_by_source[int(identifier)] for identifier in vertex_source_ids],
        dtype=np.int64,
    )
    vertex_coordinates = target_coordinate_bank[vertex_coordinate_order].reshape(
        (-1, target_coordinate_bank.shape[1])
    )
    exact_coordinates = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in local_coordinates
    )
    exact_vertex_coordinates = (
        None
        if vertex_coordinates.shape[0] == 0
        else tuple(
            tuple(Fraction.from_float(float(value)) for value in row)
            for row in vertex_coordinates
        )
    )
    target_measures = np.asarray(
        [measure_by_id[int(identifier)] for identifier in source_cell_ids],
        dtype=np.float64,
    )
    if local_coordinates.shape != np.asarray(source.coordinates).shape:
        raise ValueError("Owner-local motion changed its source coordinate-bank shape.")
    local_changed = not np.array_equal(
        local_coordinates, np.asarray(source.coordinates, dtype=np.float64)
    )
    changed_flags = _owner_group_allgather(
        np.asarray((local_changed,), dtype=np.uint8),
        execution_group,
        f"{axis_name}_coordinate_change",
    )
    if not any(bool(np.asarray(flags, dtype=np.uint8)[0]) for flags in changed_flags):
        if not np.array_equal(target_measures, np.asarray(source.cell_measures)):
            raise ValueError(
                "Unchanged motion coordinates changed accepted cell measures."
            )
        return source
    if source.source_context_id is not None:
        raise ValueError("Non-periodic motion cannot retain a periodic source context.")
    local_records = tuple(
        (
            int(target_coordinate_ids[row]),
            tuple(
                int(bits)
                for bits in np.asarray(target_coordinate_bank[row], np.float64).view(
                    np.uint64
                )
            ),
        )
        for row in np.flatnonzero(target_coordinate_owners == local_rank)
    )
    target_map_id = _owner_local_multiset_identity(
        "owner-local-motion-coordinate-map",
        {
            "source_topology": source.source_topology_id,
            "coordinate_contract": source.coordinate_contract_id,
            "geometry_layout": source.geometry_layout_id,
            "ambient_dimension": int(target_coordinate_bank.shape[1]),
        },
        local_records,
        source.source_id,
        source.source_revision,
        execution_group,
        f"{axis_name}_coordinate_map",
    )
    target_geometry_id = canonical_fingerprint(
        {
            "kind": "owner-local-motion-global-geometry",
            "source_topology": source.source_topology_id,
            "coordinate_map": target_map_id,
            "geometry_layout": source.geometry_layout_id,
            "coordinate_contract": source.coordinate_contract_id,
        }
    )
    target_source_revision = target_map_id
    target_part_id = canonical_fingerprint(
        {
            "kind": "owner-local-motion-part",
            "source": source.source_id,
            "revision": target_source_revision,
            "topology": source.source_topology_id,
            "geometry": target_geometry_id,
        }
    )
    target_map = OwnerLocalCoordinateMap(
        source.source_id,
        target_source_revision,
        target_map_id,
        tuple(int(row) for row in coordinate_rows),
        exact_coordinates,
    )
    source_manifest = MeshPartManifest(
        source.part_name,
        target_part_id,
        source.coordinate_contract,
        source.topological_dimension,
        int(target_coordinate_bank.shape[1]),
        source_topology_id=source.source_topology_id,
        source_geometry_id=target_geometry_id,
    )
    unbound_directory = ShardedOwnerDirectoryWitness(
        execution_group,
        local_rank,
        source_manifest.manifest_id,
        f"{source.part_name}/owner-directory",
        source_point_ids,
        np.asarray(source.source_owner_ranks, dtype=np.int32),
        np.asarray(source.source_owned, dtype=np.bool_),
        source_cell_ids,
        np.asarray(source.cell_owner_ranks, dtype=np.int32),
        source_vertex_ids,
        np.asarray(source.vertex_owner_ranks, dtype=np.int32),
        source_map_id=target_map_id,
        source_context_id=None,
        axis_name=f"{axis_name}_target_directory",
    )
    # The moved coordinate map is the accepted scientific source and part
    # authority of the successor view; its directory binds that authority.
    source_entry = CompositionEntry(
        target_map,
        entry_id=f"{source.part_name}/motion-source",
        role="topology",
        owner_id="phydrax.meshing.motion.owner_local",
        structure_id=source.source_mesh_id,
        revision_id=target_source_revision,
        semantics_id=target_map_id,
    )
    part_authority = CompositionEntry(
        target_map,
        entry_id=f"{source.part_name}/motion-part",
        role="topology",
        owner_id="phydrax.meshing.motion.owner_local",
        structure_id=source.source_topology_id,
        revision_id=target_part_id,
        semantics_id=target_geometry_id,
        dependencies=(
            source_entry.binding("structure"),
            source_entry.binding("revision"),
            source_entry.binding("semantics"),
            unbound_directory.composition_entry().binding("structure"),
        ),
    )
    target_directory = unbound_directory.bind_source_authority(
        part_authority.record_id,
        execution_group,
        axis_name=f"{axis_name}_target_directory_binding",
    )
    accepted_composition = Composition(
        (source_entry, target_directory.composition_entry(), part_authority),
        boundary_id=target_part_id,
    )
    block_records = []
    for block in source.blocks:
        block_cell_ids = np.asarray(block.cell_ids, dtype=np.int64)
        record: dict[str, Any] = {
            "name": block.name,
            "cell_kind": block.cell_kind,
            "cell_rows": np.asarray(block.cell_rows, dtype=np.int64),
            "cell_ids": block_cell_ids,
            "source_cell_ids": np.asarray(block.source_cell_ids, dtype=np.int64),
            "vertices": np.asarray(block.vertices, dtype=np.int32),
            "vertex_valid": np.asarray(block.vertex_valid, dtype=np.bool_),
            "coordinate_routes": np.asarray(block.coordinate_routes, dtype=np.int32),
            "element": block.element,
            "element_rows": np.asarray(block.element_rows, dtype=np.int32),
            "chart_arguments": block.chart_arguments,
            "cell_measures": np.asarray(
                [measure_by_id[int(identifier)] for identifier in block_cell_ids],
                dtype=np.float64,
            ),
            "cell_vertex_shifts": (
                None
                if block.cell_vertex_shifts is None
                else np.asarray(block.cell_vertex_shifts, dtype=np.int64)
            ),
        }
        if block.cell_kind == "polyhedron":
            face_ids = np.asarray(block.polyhedron_face_ids, dtype=np.int64)
            face_loops = tuple(
                np.asarray(loop, dtype=np.int32)
                for loop in block.polyhedron_face_vertex_loops
            )
            face_index = {
                tuple(sorted(int(value) for value in loop)): index
                for index, loop in enumerate(face_loops)
            }
            face_offsets = np.zeros((len(face_loops) + 1,), dtype=np.int64)
            for index, loop in enumerate(face_loops):
                face_offsets[index + 1] = face_offsets[index] + loop.size
            face_values = (
                np.concatenate(face_loops).astype(np.int32, copy=False)
                if face_loops
                else np.empty((0,), dtype=np.int32)
            )
            cell_face_offsets = np.zeros((block_cell_ids.size + 1,), dtype=np.int64)
            cell_face_values: list[int] = []
            cell_face_signs: list[int] = []
            for cell, cell_faces in enumerate(block.polyhedron_faces):
                signs = np.asarray(block.polyhedron_face_signs[cell], dtype=np.int8)
                if signs.shape != (len(cell_faces),):
                    raise ValueError("Owner-local polyhedron face signs are malformed.")
                for face, sign in zip(cell_faces, signs, strict=True):
                    key = tuple(sorted(int(value) for value in np.asarray(face)))
                    face_id = face_index.get(key)
                    if face_id is None:
                        raise ValueError(
                            "Owner-local polyhedron incidence lost an authored face."
                        )
                    cell_face_values.append(face_id)
                    cell_face_signs.append(int(sign))
                cell_face_offsets[cell + 1] = len(cell_face_values)
            record.update(
                {
                    "face_ids": face_ids,
                    "cell_face_offsets": cell_face_offsets,
                    "cell_face_values": np.asarray(cell_face_values, dtype=np.int64),
                    "cell_face_signs": np.asarray(cell_face_signs, dtype=np.int8),
                    "face_vertex_offsets": face_offsets,
                    "face_vertex_values": face_values,
                }
            )
        block_records.append(record)
    record = {
        "part_name": source.part_name,
        "part_id": target_part_id,
        "distribution_id": target_directory.directory_id,
        "source_id": source.source_id,
        "source_revision": target_source_revision,
        "source_mesh_id": source.source_mesh_id,
        "source_topology_id": source.source_topology_id,
        "source_geometry_id": target_geometry_id,
        "source_map_id": target_map_id,
        "geometry_layout_id": source.geometry_layout_id,
        "coordinate_contract_id": source.coordinate_contract_id,
        "coordinate_contract": source.coordinate_contract,
        "topological_dimension": source.topological_dimension,
        "cell_rows": np.asarray(source.cell_rows, dtype=np.int64),
        "cell_ids": source_cell_ids,
        "owned": np.asarray(source.owned, dtype=np.bool_),
        "cell_measures": target_measures,
        "cell_owner_ranks": np.asarray(source.cell_owner_ranks, dtype=np.int32),
        "support_cell_mask": np.asarray(source.support_cell_mask, dtype=np.bool_),
        "vertex_rows": np.asarray(source.vertex_rows, dtype=np.int64),
        "vertex_ids": source_vertex_ids,
        "vertex_coordinates": vertex_coordinates,
        "exact_vertex_coordinates": exact_vertex_coordinates,
        "coordinate_rows": coordinate_rows,
        "coordinates": local_coordinates,
        "exact_coordinates": exact_coordinates,
        "entity_set_ids": dict(source.entity_set_ids),
        "local_entity_ids": {
            int(dimension): np.asarray(ids, dtype=np.int64)
            for dimension, ids in source.local_entity_ids
        },
        "entity_vertex_offsets": {
            int(dimension): np.asarray(offsets, dtype=np.int64)
            for dimension, offsets in source.entity_vertex_offsets
        },
        "entity_vertex_values": {
            int(dimension): np.asarray(values, dtype=np.int32)
            for dimension, values in source.entity_vertex_values
        },
        "cell_key_offsets": np.asarray(source.cell_key_offsets, dtype=np.int64),
        "cell_key_source_ids": np.asarray(source.cell_key_source_ids, dtype=np.int64),
        "cell_key_relative_windings": np.asarray(
            source.cell_key_relative_windings, dtype=np.int64
        ),
        "vertex_source_ids": vertex_source_ids,
        "source_point_ids": np.asarray(source.source_point_ids, dtype=np.int64),
        "source_owner_ranks": np.asarray(source.source_owner_ranks, dtype=np.int32),
        "source_owned": np.asarray(source.source_owned, dtype=np.bool_),
        "vertex_owner_ranks": np.asarray(source.vertex_owner_ranks, dtype=np.int32),
        "vertex_absolute_windings": np.asarray(
            source.vertex_absolute_windings, dtype=np.int64
        ),
        "candidate_revision_id": candidate_revision_id,
        "certified_periodic_margin": periodic_margin,
        "periodic_shell_certificate": periodic_shell_certificate,
        "periodic_topology_id": source.periodic_topology_id,
        "periodic_cell_ids": (
            None
            if source.periodic_cell_ids is None
            else np.asarray(source.periodic_cell_ids, dtype=np.int64)
        ),
        "vertex_representative_ids": (
            None
            if source.vertex_representative_ids is None
            else np.asarray(source.vertex_representative_ids, dtype=np.int64)
        ),
        "vertex_windings": (
            None
            if source.vertex_windings is None
            else np.asarray(source.vertex_windings, dtype=np.int64)
        ),
        "cell_windings": (
            None
            if source.cell_windings is None
            else np.asarray(source.cell_windings, dtype=np.int64)
        ),
        "source_context_id": None,
        "periodic_lift_id": source.periodic_lift_id,
        "periodic_lift_matrix": (
            None
            if source.periodic_lift_matrix is None
            else np.asarray(source.periodic_lift_matrix, dtype=np.float64)
        ),
        "coordinates_are_lifted": source.coordinates_are_lifted,
        "support_fractional_lower": support_fractional_lower,
        "support_fractional_upper": support_fractional_upper,
        "blocks": tuple(block_records),
        "rank": local_rank,
        "serial_source_view": False,
        "part_authority": part_authority,
        "accepted_composition": accepted_composition,
        "owner_directory_witness": target_directory,
        "owned_cell_ids": np.asarray(source.owned_cell_ids, dtype=np.int64),
        "owned_cell_owner_ranks": np.asarray(
            source.owned_cell_owner_ranks, dtype=np.int32
        ),
        "owned_cell_weights": np.ones(len(source.owned_cell_ids), dtype=np.int64),
        "cell_neighbor_offsets": np.asarray(source.cell_neighbor_offsets, dtype=np.int64),
        "cell_neighbor_ids": np.asarray(source.cell_neighbor_ids, dtype=np.int64),
        "cell_neighbor_owner_ranks": np.asarray(
            source.cell_neighbor_owner_ranks, dtype=np.int32
        ),
        "cell_neighbor_weights": np.asarray(source.cell_neighbor_weights, dtype=np.int64),
    }
    return OwnerLocalCellGeometry.from_owner_local_result(record)


def _owner_local_owned_star_scope(
    view: Any,
    execution_group: Any,
    local_rank: int,
    axis_name: str,
    /,
) -> CellMeshManifoldScope:
    """Collect the owned-star manifold scope of one owner-local patch.

    A face-halo patch is pinched where its resident cells end, and even an
    owned vertex may miss star cells that only touch it. Every vertex is
    certified by its owner on its complete star: the owner's resident cells
    plus the incident cells other owners send it. Halo vertices are left to
    their owners. Collective over ``execution_group``; empty owners join.
    """
    from ..discretization._reference_cell import REFERENCE_TOPOLOGIES
    from ._distribution import (
        _owner_exchange_message_banks,
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )

    kinds = tuple(sorted(REFERENCE_TOPOLOGIES))
    parts = len(execution_group.spec.process_indices)
    local_error: str | None = None
    vertex_ids = np.empty((0,), dtype=np.int64)
    owned_vertex_ids = np.empty((0,), dtype=np.int64)
    resident: dict[int, tuple[int, np.ndarray]] = {}
    owned_cells: list[tuple[int, int, np.ndarray]] = []
    local_width = 0
    try:
        vertex_ids = np.asarray(view.vertex_ids, dtype=np.int64)
        vertex_owners = np.asarray(view.vertex_owner_ranks, dtype=np.int32)
        cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
        cell_owners = np.asarray(view.cell_owner_ranks, dtype=np.int32)
        if (
            vertex_owners.shape != vertex_ids.shape
            or cell_owners.shape != cell_ids.shape
            or np.any((vertex_owners < 0) | (vertex_owners >= parts))
        ):
            raise ValueError("Owner-local star ownership rows are malformed.")
        owner_by_cell = dict(zip(cell_ids.tolist(), cell_owners.tolist(), strict=True))
        for block in view.blocks:
            identifiers = np.asarray(block.cell_ids, dtype=np.int64)
            if not identifiers.size:
                continue
            vertices = np.asarray(block.vertices, dtype=np.int64)
            if block.cell_kind not in kinds or not np.all(
                np.asarray(block.vertex_valid, dtype=np.bool_)
            ):
                raise ValueError(
                    "Owned-star certification requires dense reference cells."
                )
            code = kinds.index(block.cell_kind)
            local_width = max(local_width, vertices.shape[1])
            for identifier, row in zip(identifiers.tolist(), vertices, strict=True):
                global_vertices = vertex_ids[row]
                resident[identifier] = (code, global_vertices)
                if owner_by_cell[identifier] == local_rank:
                    owned_cells.append((identifier, code, global_vertices))
        owned_vertex_ids = vertex_ids[vertex_owners == local_rank]
        if owned_vertex_ids.size and not resident:
            raise ValueError("An owner without resident cells cannot certify its stars.")
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "owned-star scope admission"
    )
    width = int(
        np.max(
            _owner_group_allgather(
                np.asarray((local_width,), dtype=np.int64),
                execution_group,
                f"{axis_name}_width",
            )
        )
    )
    owner_by_vertex = dict(
        zip(
            vertex_ids.tolist(),
            np.asarray(view.vertex_owner_ranks, dtype=np.int32).tolist(),
            strict=True,
        )
    )
    outgoing: list[list[np.ndarray]] = [[] for _ in range(parts)]
    for identifier, code, global_vertices in owned_cells:
        message = np.full((width + 3,), -1, dtype=np.int64)
        message[:3] = (identifier, code, global_vertices.size)
        message[3 : 3 + global_vertices.size] = global_vertices
        for owner in sorted(
            {owner_by_vertex[int(vertex)] for vertex in global_vertices} - {local_rank}
        ):
            outgoing[owner].append(message)
    incoming = _owner_exchange_message_banks(
        tuple(
            np.stack(rows) if rows else np.empty((0, width + 3), dtype=np.int64)
            for rows in outgoing
        ),
        local_rank,
        execution_group,
        f"{axis_name}_cells",
    )
    local_error = None
    owned = set(owned_vertex_ids.tolist())
    star_cells: dict[int, dict[int, np.ndarray]] = {}
    for sender in sorted(incoming):
        for message in incoming[sender]:
            identifier, code, arity = (int(value) for value in message[:3])
            global_vertices = message[3 : 3 + arity]
            if not owned.intersection(global_vertices.tolist()):
                local_error = "A star cell reached a rank owning none of its vertices."
            elif identifier in resident:
                if resident[identifier][0] != code or not np.array_equal(
                    resident[identifier][1], global_vertices
                ):
                    local_error = "Owner-local star cells disagree with their owner."
            else:
                star_cells.setdefault(code, {})[identifier] = global_vertices
    _owner_group_fail_if_any(
        local_error, execution_group, axis_name, "owned-star scope completion"
    )
    return CellMeshManifoldScope(
        "owned-star",
        owned_vertex_ids,
        star_cells=tuple(
            (
                kinds[code],
                np.asarray(sorted(cells), dtype=np.int64),
                np.stack([cells[identifier] for identifier in sorted(cells)]),
            )
            for code, cells in sorted(star_cells.items())
        ),
    )


def _owner_local_rejected(
    source: Any,
    coordinates: Any,
    boundary_residual: float,
    reason: str,
    /,
) -> MeshMotionAdvance:
    assessment = MeshMotionAssessment(
        MeshMotionDecision.REJECT,
        None,
        minimum_jacobian_ratio=0.0,
        minimum_quality_ratio=0.0,
        maximum_displacement_fraction=0.0,
        boundary_residual=boundary_residual,
        reasons=("owner-local-motion-rejected",),
        coordinates_id=canonical_fingerprint(
            array_tree_fingerprint(np.asarray(coordinates))
        ),
        failure_detail=str(reason),
    )
    return MeshMotionAdvance(
        MeshMotionDecision.REJECT,
        (assessment,),
        source_mesh_id=source.source_mesh_id,
        source_result_id=source.local_view_id,
    )


def advance_owner_local_mesh_motion(
    source: Any,
    coordinates: ConvertibleToArray,
    execution_group: Any,
    /,
    *,
    boundary_residual: float,
    fixed_vertices: ConvertibleToArray,
    driver_id: str,
    policy: MeshMotionMonitorPolicy | None = None,
    periodic_specification: Any | None = None,
    axis_name: str = "owner_local_mesh_motion",
) -> MeshMotionAdvance:
    """Collectively accept a source-bound owner-local motion proposal.

    Periodic views take coordinates from the current native source point bank;
    ``fixed_vertices`` always indexes the local lifted mesh vertex bank.
    """
    import jax

    from .._execution_runtime import ExecutionGroup
    from ..discretization._view_support import OwnerLocalCellGeometry
    from ._distribution import (
        _owner_group_allgather,
        _owner_group_fail_if_any,
    )

    if not isinstance(source, OwnerLocalCellGeometry):
        raise TypeError("source must be OwnerLocalCellGeometry.")
    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup.")
    if source.periodic_topology_id is not None:
        from .providers._native_periodic import SurfaceMeshingSpec, VolumeMeshingSpec

        if (
            not isinstance(
                periodic_specification, (SurfaceMeshingSpec, VolumeMeshingSpec)
            )
            or periodic_specification.specification_id != source.specification_id
        ):
            raise ValueError(
                "Periodic owner-local motion requires its exact typed current specification."
            )
    elif periodic_specification is not None:
        raise ValueError(
            "Non-periodic owner-local motion cannot take a periodic specification."
        )
    name = str(driver_id).strip()
    axis = str(axis_name).strip()
    if not name or not axis:
        raise ValueError("driver_id and axis_name must be non-empty.")
    members = execution_group.spec.process_indices
    if (
        not execution_group.is_member
        or len(execution_group.devices) != len(members)
        or tuple(device.process_index for device in execution_group.devices) != members
        or jax.process_index() not in members
    ):
        raise ValueError(
            "Owner-local motion requires one registered group device per member."
        )
    local_rank = members.index(jax.process_index())
    proposed = np.asarray(coordinates, dtype=np.float64)
    fixed_raw = np.asarray(fixed_vertices)
    fixed = fixed_raw.astype(np.bool_, copy=False)
    local_error = None
    try:
        source.require_owner_directory_current(
            execution_group,
            axis_name=f"{axis}_source_directory",
        )
        coordinate_bank, _, _ = _owner_local_motion_coordinate_bank(
            source, execution_group, local_rank, f"{axis}_admission_bank"
        )
        if (
            source.owner_rank != local_rank
            or proposed.shape != coordinate_bank.shape
            or fixed_raw.shape != np.asarray(source.vertex_ids).shape
            or fixed_raw.dtype != np.dtype(np.bool_)
            or np.any(~np.isfinite(proposed))
        ):
            raise ValueError(
                "Owner-local motion does not bind the accepted source view and coordinate request."
            )
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    try:
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis}_admission",
            "owner-local motion admission",
        )
    except Exception as error:
        return _owner_local_rejected(source, proposed, boundary_residual, str(error))
    try:
        synced_coordinates, coordinate_ids, coordinate_owners = (
            _owner_local_sync_coordinates(
                source, proposed, execution_group, local_rank, axis
            )
        )
    except Exception as error:
        return _owner_local_rejected(source, proposed, boundary_residual, str(error))

    motion: MeshMotionAdvance | None = None
    target_coordinates = np.array(synced_coordinates, dtype=np.float64, copy=True)
    source_mesh = None
    source_local_result = None
    source_route_rows = np.empty((0,), dtype=np.int32)
    try:
        periodic_target = None
        owned_star_scope = None
        if source.periodic_topology_id is not None:
            periodic_target = _prepare_owner_local_periodic_motion_target(
                source,
                periodic_specification,
                synced_coordinates,
                coordinate_ids,
                coordinate_owners,
                execution_group,
                local_rank,
                axis,
            )
            # The resident patch is pinched at its halo; each owner certifies
            # the complete stars of its own vertices (fixed topology, so the
            # source scope also covers the target).
            owned_star_scope = _owner_local_owned_star_scope(
                source, execution_group, local_rank, f"{axis}_owned_star"
            )
        if np.asarray(source.cell_ids).size:
            if periodic_target is not None:
                _, target_materialization = periodic_target
                source_mesh, source_geometry = source.reconstruct_local_geometry()
                target_mesh = target_materialization.mesh
                target_geometry = target_materialization.geometry
                if (
                    source_mesh is None
                    or source_geometry is None
                    or target_mesh is None
                    or target_geometry is None
                ):
                    raise ValueError(
                        "Nonempty owner-local periodic motion lacks exact source/target geometry."
                    )
                if source_mesh.topology_id != target_mesh.topology_id:
                    raise ValueError("Owner-local periodic motion changed cell topology.")
                if not np.array_equal(
                    np.asarray(source_mesh.coordinates)[fixed],
                    np.asarray(target_mesh.coordinates)[fixed],
                ):
                    raise ValueError("Owner-local periodic motion moved a fixed vertex.")
                source_local_result = certify_cell_mesh(
                    source_mesh,
                    source.coordinate_contract,
                    geometry=source_geometry,
                    manifold_scope=owned_star_scope,
                )
                monitor = MeshMotionMonitor(source_mesh, policy=policy)
                assessment = monitor.assess(
                    np.asarray(target_mesh.coordinates),
                    boundary_residual=boundary_residual,
                )
                if assessment.decision is not MeshMotionDecision.ACCEPT_MOTION:
                    raise ValueError(
                        "Owner-local periodic motion did not pass its fixed-topology monitor."
                    )
                target_result = certify_cell_mesh(
                    target_mesh,
                    source.coordinate_contract,
                    geometry=target_geometry,
                    manifold_scope=owned_star_scope,
                )
                transition = _transition_geometry_record(
                    source_local_result, target_mesh, target_geometry
                )
                assessment = _motion_assessment_with_acceptance(
                    assessment,
                    MeshMotionDecision.ACCEPT_MOTION,
                    np.asarray(target_mesh.coordinates),
                    certification=target_result.certification,
                    failure_detail=None,
                )
                motion = MeshMotionAdvance(
                    MeshMotionDecision.ACCEPT_MOTION,
                    (assessment,),
                    source_mesh_id=source_mesh.mesh_id,
                    source_result_id=source_local_result.result_id,
                    result=target_result,
                    geometry_transition=transition,
                )
                target_coordinates = np.array(
                    synced_coordinates, dtype=np.float64, copy=True
                )
            else:
                source_mesh, _, _, source_fixed, _ = _owner_local_motion_mesh(
                    source,
                    np.asarray(source.coordinates),
                    fixed,
                )
                target_mesh, source_route_rows, target_route_shifts, _, _ = (
                    _owner_local_motion_mesh(source, synced_coordinates, fixed)
                )
                if source_mesh.topology_id != target_mesh.topology_id:
                    raise ValueError("Owner-local motion changed local cell topology.")
                source_geometry = CellGeometrySpec.affine(source_mesh)
                source_local_result = certify_cell_mesh(
                    source_mesh,
                    source.coordinate_contract,
                    geometry=source_geometry,
                )
                monitor = MeshMotionMonitor(source_mesh, policy=policy)
                local_motion = advance_mesh_motion(
                    monitor,
                    source_local_result,
                    np.asarray(target_mesh.coordinates),
                    boundary_residual=boundary_residual,
                    fixed_vertices=source_fixed,
                    numeric_version=f"owner-local-motion:{name}",
                )
                if (
                    not local_motion.accepted
                    or local_motion.result is None
                    or local_motion.adaptation is not None
                    or local_motion.decision is MeshMotionDecision.REMESH
                    or local_motion.geometry_transition is None
                ):
                    raise ValueError(
                        "Owner-local motion did not produce an accepted fixed-topology transition."
                    )
                if local_motion.result.mesh.topology_id != source_mesh.topology_id:
                    raise ValueError(
                        "Owner-local motion result changed its source topology."
                    )
                recovered = np.zeros_like(synced_coordinates)
                assigned = np.zeros((synced_coordinates.shape[0],), dtype=np.bool_)
                moved_points = np.asarray(local_motion.result.mesh.coordinates)
                for point, (route, shift) in enumerate(
                    zip(source_route_rows, target_route_shifts, strict=True)
                ):
                    value = moved_points[point] - shift
                    if assigned[route] and not np.array_equal(recovered[route], value):
                        raise ValueError(
                            "Local motion copies of one source coordinate diverged."
                        )
                    recovered[route] = value
                    assigned[route] = True
                target_coordinates[assigned] = recovered[assigned]
                target_coordinates, _, _ = _owner_local_sync_coordinates(
                    source,
                    target_coordinates,
                    execution_group,
                    local_rank,
                    f"{axis}_accepted",
                )
                target_mesh_check, _, _, _, _ = _owner_local_motion_mesh(
                    source, target_coordinates, fixed
                )
                if not np.array_equal(
                    np.asarray(target_mesh_check.coordinates),
                    np.asarray(local_motion.result.mesh.coordinates),
                ):
                    raise ValueError(
                        "Owner-authored motion coordinates differ from the accepted transition."
                    )
                motion = local_motion
        elif periodic_target is None and (
            # The periodic coordinate bank is collective; only the local
            # non-periodic bank may be rechecked on an empty owner alone.
            synced_coordinates.shape
            != _owner_local_motion_coordinate_bank(
                source, execution_group, local_rank, f"{axis}_empty_bank"
            )[0].shape
        ):
            raise ValueError("Empty owner-local motion coordinate bank changed shape.")
    except Exception as error:
        local_error = str(error)
    try:
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis}_geometry",
            "owner-local motion geometry",
        )
    except Exception as error:
        return _owner_local_rejected(source, proposed, boundary_residual, str(error))

    accepted_flag = np.asarray(
        [1 if motion is None or motion.accepted else 0], dtype=np.int32
    )
    collective_flags = _owner_group_allgather(
        accepted_flag, execution_group, f"{axis}_acceptance"
    )
    if not np.all(collective_flags == 1):
        return _owner_local_rejected(
            source,
            target_coordinates,
            boundary_residual,
            "another rank rejected the owner-local motion proposal",
        )

    coordinate_source_ids = np.asarray(coordinate_ids, dtype=np.int64)
    coordinate_owner_ranks = np.asarray(coordinate_owners, dtype=np.int32)
    digest_rows = np.flatnonzero(coordinate_owner_ranks == local_rank)
    order = np.argsort(coordinate_source_ids[digest_rows], kind="stable")
    digest = canonical_fingerprint(
        {
            "rank": local_rank,
            "source_ids": coordinate_source_ids[digest_rows][order],
            "coordinates": target_coordinates[digest_rows][order],
        }
    )
    digests = _owner_group_allgather(
        np.frombuffer(bytes.fromhex(digest), dtype=np.uint8),
        execution_group,
        f"{axis}_candidate_identity",
    )
    periodic_source_refresh = None if periodic_target is None else periodic_target[0]
    source_candidate_id = (
        source.candidate_revision_id
        if periodic_source_refresh is None
        else periodic_source_refresh.candidate["candidate_revision_id"]
    )
    candidate_revision_id = canonical_fingerprint(
        {
            "kind": "owner-local-motion-candidate",
            "source_candidate": source_candidate_id,
            "driver": name,
            "rank_digests": tuple(
                np.asarray(row, dtype=np.uint8).tobytes().hex() for row in digests
            ),
        }
    )

    periodic_margin = source.certified_periodic_margin
    periodic_shell_certificate = source.periodic_shell_certificate
    support_lower = (
        None
        if source.support_fractional_lower is None
        else np.asarray(source.support_fractional_lower, dtype=np.float64)
    )
    support_upper = (
        None
        if source.support_fractional_upper is None
        else np.asarray(source.support_fractional_upper, dtype=np.float64)
    )
    if source.periodic_topology_id is not None:
        local_error = None
        try:
            if periodic_source_refresh is None:
                raise ValueError(
                    "Periodic motion did not refresh its native source closure."
                )
            shell = dict(periodic_source_refresh.candidate["periodic_shell_certificate"])
            periodic_margin = float(
                periodic_source_refresh.candidate["certified_periodic_margin"]
            )
            shell["certified_periodic_margin"] = periodic_margin
            if not np.isfinite(periodic_margin) or periodic_margin <= 0.0:
                raise ValueError("Refreshed periodic motion shell margin is invalid.")
            shell["candidate_revision_id"] = candidate_revision_id
            shell["motion_driver_id"] = name

            periodic_shell_certificate = shell
            if support_lower is not None and np.asarray(source.cell_ids).size:
                if motion is None:
                    target_mesh = source_mesh
                else:
                    if motion.result is None:
                        raise ValueError("Periodic motion has no accepted target mesh.")
                    target_mesh = motion.result.mesh
                if target_mesh is None:
                    raise ValueError("Periodic motion lost its accepted target mesh.")
                lower_by_id: dict[int, np.ndarray] = {}
                upper_by_id: dict[int, np.ndarray] = {}
                cell_ids = np.asarray(
                    target_mesh.entity_set(target_mesh.topological_dimension).entity_ids,
                    dtype=np.int64,
                )
                quality_coordinates = np.asarray(target_mesh.coordinates)
                cell_index = 0
                for block in target_mesh.blocks:
                    cells = np.asarray(block.vertices, dtype=np.int64)
                    valid = np.asarray(block.vertex_valid, dtype=np.bool_)
                    for row_index in range(cells.shape[0]):
                        cell_id = int(cell_ids[cell_index])
                        cell_vertices = quality_coordinates[
                            cells[row_index, valid[row_index]]
                        ]
                        fractional = np.linalg.lstsq(
                            np.asarray(source.periodic_lift_matrix, dtype=np.float64).T,
                            cell_vertices.T,
                            rcond=None,
                        )[0].T
                        lower_by_id[cell_id] = np.min(fractional, axis=0)
                        upper_by_id[cell_id] = np.max(fractional, axis=0)
                        cell_index += 1
                if cell_index != cell_ids.size:
                    raise ValueError("Periodic motion support bounds omit target cells.")
                support_lower = np.asarray(
                    [
                        lower_by_id[int(identifier)]
                        for identifier in np.asarray(source.cell_ids)
                    ],
                    dtype=np.float64,
                )
                support_upper = np.asarray(
                    [
                        upper_by_id[int(identifier)]
                        for identifier in np.asarray(source.cell_ids)
                    ],
                    dtype=np.float64,
                )
        except Exception as error:
            local_error = str(error)
        try:
            _owner_group_fail_if_any(
                local_error,
                execution_group,
                f"{axis}_periodic_admission",
                "owner-local periodic motion",
            )
        except Exception as error:
            return _owner_local_rejected(
                source, target_coordinates, boundary_residual, str(error)
            )

    measure_by_id: dict[int, float] = {}
    local_transition = None
    local_target_mesh_id = None
    decision = MeshMotionDecision.ACCEPT_MOTION
    assessments: tuple[MeshMotionAssessment, ...] = ()
    local_manifold_scope = None
    local_error = None
    if motion is not None:
        if motion.adaptation is not None or motion.result is None:
            local_error = (
                "Owner-local motion currently requires a fixed-topology accepted result."
            )
        else:
            target_mesh = motion.result.mesh
            target_quality = evaluate_cell_quality(target_mesh)
            target_cell_ids = np.asarray(
                target_mesh.entity_set(target_mesh.topological_dimension).entity_ids,
                dtype=np.int64,
            )
            target_measures = np.asarray(target_quality.measures, dtype=np.float64)
            if target_cell_ids.shape != target_measures.shape:
                local_error = (
                    "Local motion measures do not align with stable target cell IDs."
                )
            measure_by_id = {
                int(identifier): float(value)
                for identifier, value in zip(
                    target_cell_ids.tolist(), target_measures.tolist(), strict=False
                )
            }
            local_transition = motion.geometry_transition
            local_target_mesh_id = target_mesh.mesh_id
            decision = motion.decision
            assessments = motion.assessments
            local_manifold_scope = motion.result.audit.manifold_scope
    # Every owner joins the target admission before the collective view build.
    try:
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis}_target_admission",
            "owner-local motion target admission",
        )
    except Exception as error:
        return _owner_local_rejected(
            source, target_coordinates, boundary_residual, str(error)
        )

    target_view: Any = None
    local_error = None
    try:
        if motion is None:
            support_lower_target, support_upper_target = support_lower, support_upper
        else:
            support_lower_target, support_upper_target = support_lower, support_upper
        target_view = _owner_local_motion_view(
            source,
            target_coordinates,
            coordinate_ids,
            coordinate_owners,
            measure_by_id,
            candidate_revision_id,
            periodic_margin,
            periodic_shell_certificate,
            support_lower_target,
            support_upper_target,
            periodic_target=periodic_target,
            periodic_specification=periodic_specification,
            execution_group=execution_group,
            local_rank=local_rank,
            axis_name=f"{axis}_target_authority",
        )
        target_view.require_owner_directory_current(
            execution_group,
            axis_name=f"{axis}_target_directory",
        )
    except Exception as error:
        local_error = str(error)
    try:
        _owner_group_fail_if_any(
            local_error,
            execution_group,
            f"{axis}_target_view",
            "owner-local motion target view",
        )
    except Exception as error:
        return _owner_local_rejected(
            source, target_coordinates, boundary_residual, str(error)
        )
    advance_source_mesh_id = (
        source.source_mesh_id if motion is None else motion.source_mesh_id
    )
    if source.periodic_topology_id is None:
        # The monitor ran on its compact motion mesh; the accepted transition
        # binds the canonical local meshes the source and target views publish.
        local_error = None
        try:
            if motion is not None:
                source_local_mesh, source_local_geometry = (
                    source.reconstruct_local_geometry()
                )
                target_local_mesh, target_local_geometry = (
                    target_view.reconstruct_local_geometry()
                )
                if (
                    source_local_mesh is None
                    or source_local_geometry is None
                    or target_local_mesh is None
                    or target_local_geometry is None
                ):
                    raise ValueError("Nonempty motion views lost their local carriers.")
                local_transition = _transition_geometry_record(
                    certify_cell_mesh(
                        source_local_mesh,
                        source.coordinate_contract,
                        geometry=source_local_geometry,
                    ),
                    target_local_mesh,
                    target_local_geometry,
                )
                local_target_mesh_id = target_local_mesh.mesh_id
                advance_source_mesh_id = source_local_mesh.mesh_id
        except Exception as error:
            local_error = str(error)
        try:
            _owner_group_fail_if_any(
                local_error,
                execution_group,
                f"{axis}_canonical_transition",
                "owner-local motion canonical transition",
            )
        except Exception as error:
            return _owner_local_rejected(
                source, target_coordinates, boundary_residual, str(error)
            )
    return MeshMotionAdvance(
        decision,
        tuple(assessments),
        source_mesh_id=advance_source_mesh_id,
        source_result_id=source.local_view_id,
        local_result=target_view,
        local_source_view_id=source.local_view_id,
        local_target_mesh_id=local_target_mesh_id,
        motion_driver_id=name,
        local_fixed_vertex_ids=np.asarray(source.vertex_ids, dtype=np.int64)[fixed],
        geometry_transition=local_transition,
        relocation=None if motion is None else motion.relocation,
        local_manifold_scope=local_manifold_scope,
    )


__all__ = [
    "MeshMotionAdvance",
    "MeshMotionAssessment",
    "MeshMotionDecision",
    "MeshMotionMonitor",
    "MeshMotionMonitorPolicy",
    "advance_mesh_motion",
    "advance_owner_local_mesh_motion",
]
