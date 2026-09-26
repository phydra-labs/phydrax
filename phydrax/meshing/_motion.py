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

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    CellMesh,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ..optim import OptimizationTermination
from ._adaptation import (
    execute_mesh_adaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
)
from ._canonical import certify_cell_mesh
from ._lineage import CellMeshTransition
from ._metric import MeshMetricField
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
    ):
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
    ):
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

    reference: CellMesh
    policy: MeshMotionMonitorPolicy
    reference_certificate: CellValidityCertificate
    reference_quality: Array
    length_scale: float = eqx.field(static=True)
    monitor_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference: CellMesh,
        /,
        *,
        policy: MeshMotionMonitorPolicy | None = None,
    ):
        if not isinstance(reference, CellMesh):
            raise TypeError("reference must be CellMesh.")
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

    def moved_mesh(self, coordinates: ArrayLike, /) -> CellMesh:
        """The reference topology at ``coordinates`` with a content-derived version."""

        points = np.asarray(coordinates, dtype=np.float64)
        return self.reference.with_coordinates(
            points, numeric_version=f"motion:{array_tree_fingerprint(points)}"
        )

    def assess(
        self, coordinates: ArrayLike, /, *, boundary_residual: float
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
    """

    decision: MeshMotionDecision = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    assessments: tuple[MeshMotionAssessment, ...]
    result: CellMeshingResult | None
    relocation: MeshOptimizationResult | None
    remesh_metric: MeshMetricField | None
    adaptation: MeshAdaptationResult | None
    transition: CellMeshTransition | None
    advance_id: str = eqx.field(static=True)

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
    ):
        remesh = decision is MeshMotionDecision.REMESH
        if (decision is MeshMotionDecision.REJECT) != (result is None):
            raise ValueError("Exactly the rejected advances carry no result.")
        if remesh != (remesh_metric is not None) or (
            adaptation is not None and not remesh
        ):
            raise ValueError("Only remesh advances carry a metric and an adaptation.")
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
) -> MeshAdaptationResult:
    return execute_mesh_adaptation(
        prepare_mesh_adaptation(candidate, MetricMeshAdaptation(metric), policy=policy)
    )


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
    """Accept, relocate, remesh, or reject one moved coordinate proposal.

    ``source`` is the certified result of the monitor's reference mesh.
    ``fixed_vertices`` (default: every boundary vertex) stay bit-identical during
    relocation. A remesh targets ``remesh_metric(candidate)`` (default: the
    isotropic metric of the reference cell sizes) on the best valid
    same-topology candidate and executes through ``adaptation_policy``'s explicit
    route; without a policy the remesh is returned as an unexecuted request.
    """

    if not isinstance(monitor, MeshMotionMonitor):
        raise TypeError("monitor must be MeshMotionMonitor.")
    if not isinstance(source, CellMeshingResult):
        raise TypeError("source must be CellMeshingResult.")
    if source.mesh.mesh_id != monitor.reference.mesh_id:
        raise ValueError("source must be the certified result of the monitor reference.")
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
            return MeshMotionAdvance(MeshMotionDecision.REJECT, tuple(assessments))
        case MeshMotionDecision.ACCEPT_MOTION:
            return MeshMotionAdvance(
                MeshMotionDecision.ACCEPT_MOTION,
                tuple(assessments),
                result=certify_cell_mesh(
                    monitor.moved_mesh(coordinates), source.coordinate_contract
                ),
            )
        case MeshMotionDecision.RELOCATE | MeshMotionDecision.REMESH:
            pass
        case decision:
            raise ValueError(f"Unsupported mesh motion decision {decision!r}.")
    # Every crossing first tries fixed-topology relocation (untangling included);
    # only an accepted relocation (converged, or explicitly admitted valid
    # non-convergence) seeds the relocated assessment.
    moved = monitor.moved_mesh(coordinates)
    relocation = _relocate(monitor, source, moved, fixed, version)
    if relocation.accepted:
        relocated = monitor.assess(
            relocation.coordinates, boundary_residual=boundary_residual
        )
        assessments.append(relocated)
        if relocated.decision is MeshMotionDecision.ACCEPT_MOTION:
            return MeshMotionAdvance(
                MeshMotionDecision.RELOCATE,
                tuple(assessments),
                result=relocation.result,
                relocation=relocation,
            )
        candidate = relocation.result
    elif assessment.certificate.all_certified:
        candidate = certify_cell_mesh(moved, source.coordinate_contract)
    else:
        # Untangling failed: an uncertified mesh cannot seed a remesh.
        return MeshMotionAdvance(
            MeshMotionDecision.REJECT, tuple(assessments), relocation=relocation
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
            result=candidate,
            relocation=relocation,
            remesh_metric=metric,
        )
    adaptation = _remesh(candidate, metric, adaptation_policy)
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
