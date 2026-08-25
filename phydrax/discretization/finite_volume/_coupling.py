#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._capillarity import BalancedCapillaryOperator
from ._contact_angle import EmbeddedBoundaryContactAngleSet
from ._embedded_dynamics import UnstructuredEmbeddedBoundarySet
from ._unstructured import UnstructuredFiniteVolumeDiscretization
from ._unstructured_amr import UnstructuredAMRHierarchyPlan
from ._unstructured_embedded_boundary import (
    EmbeddedBoundaryMetrics,
    EmbeddedBoundaryPlan,
    EmbeddedBoundaryStabilizationPolicy,
    EmbeddedBoundaryStatus,
)
from ._unstructured_motion import FixedConnectivityMotionPlan
from ._unstructured_overset import (
    PeriodicSlidingCoupling,
    PeriodicSlidingInterfacePlan,
    UnstructuredOversetPlan,
)
from ._unstructured_vof import UnstructuredVOFPlan


_TOPOLOGY_EVENT_POLICIES = frozenset(("disabled", "accepted_step"))


def _optional_plan(value: Any, expected_type: type, name: str, /):
    if value is not None and not isinstance(value, expected_type):
        raise TypeError(f"{name} must be {expected_type.__name__} or None.")
    return value


def _topology_event_configuration(capacity: Any, policy: Any, /) -> tuple[int, str]:
    if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)):
        raise TypeError("topology_event_capacity must be an integer.")
    capacity_ = int(capacity)
    if capacity_ < 0 or capacity_ > np.iinfo(np.int32).max:
        raise ValueError("topology_event_capacity must be a nonnegative int32 value.")
    if not isinstance(policy, str):
        raise TypeError("topology_event_policy must be a string.")
    if policy not in _TOPOLOGY_EVENT_POLICIES:
        raise ValueError("topology_event_policy must be 'disabled' or 'accepted_step'.")
    if policy == "disabled" and capacity_ != 0:
        raise ValueError("Disabled topology events require zero event capacity.")
    if policy == "accepted_step" and capacity_ == 0:
        raise ValueError("Accepted-step topology events require positive event capacity.")
    return capacity_, policy


def _validate_current_geometry(
    name: str,
    artifact_topology_id: str,
    artifact_geometry_id: str,
    discretization: UnstructuredFiniteVolumeDiscretization,
    /,
) -> None:
    if artifact_topology_id != discretization.topology_id:
        raise ValueError(f"{name} belongs to a different unstructured topology.")
    if artifact_geometry_id != discretization.geometry_id:
        raise ValueError(f"{name} belongs to stale unstructured geometry.")


def _validate_prepared_geometry(
    name: str,
    artifact: UnstructuredFiniteVolumeDiscretization,
    discretization: UnstructuredFiniteVolumeDiscretization,
    /,
) -> None:
    _validate_current_geometry(
        name,
        artifact.topology_id,
        artifact.geometry_id,
        discretization,
    )
    if artifact.prepared_id != discretization.prepared_id:
        raise ValueError(f"{name} belongs to a different prepared geometry.")


def _overset_masks(
    overset: UnstructuredOversetPlan,
    receptor_count: int,
    donor_count: int,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return validated donor/receptor ownership masks for one immutable map."""

    def mask(
        value: Any,
        count: int,
        name: str,
        default: np.ndarray,
    ) -> np.ndarray:
        result = default if value is None else np.asarray(value)
        if result.shape != (count,) or result.dtype != np.dtype(bool):
            raise ValueError(f"{name} must be a boolean mask with shape ({count},).")
        return result

    donor_hole = mask(
        getattr(overset, "donor_hole_mask", None),
        donor_count,
        "overset.donor_hole_mask",
        np.zeros((donor_count,), dtype=bool),
    )
    receptor_hole = mask(
        getattr(overset, "receptor_hole_mask", getattr(overset, "hole_mask", None)),
        receptor_count,
        "overset.receptor_hole_mask",
        np.zeros((receptor_count,), dtype=bool),
    )
    donor_fringe = mask(
        getattr(overset, "donor_fringe_mask", None),
        donor_count,
        "overset.donor_fringe_mask",
        np.zeros((donor_count,), dtype=bool),
    )
    receptor_fringe = mask(
        getattr(overset, "receptor_fringe_mask", None),
        receptor_count,
        "overset.receptor_fringe_mask",
        np.asarray(overset.receptor_mask, dtype=bool),
    )
    donor_active = mask(
        getattr(overset, "donor_active_mask", None),
        donor_count,
        "overset.donor_active_mask",
        ~donor_hole,
    )
    receptor_active = mask(
        getattr(overset, "receptor_active_mask", None),
        receptor_count,
        "overset.receptor_active_mask",
        ~receptor_hole,
    )
    donor_active = donor_active & ~donor_hole
    receptor_active = receptor_active & ~receptor_hole
    for prefix, active, hole, fringe in (
        ("donor", donor_active, donor_hole, donor_fringe),
        ("receptor", receptor_active, receptor_hole, receptor_fringe),
    ):
        if np.any(active & hole) or np.any(fringe & hole) or np.any(fringe & ~active):
            raise ValueError(
                f"Overset {prefix} active/hole/fringe ownership is inconsistent."
            )
        if np.any(~active & ~hole):
            raise ValueError(
                f"Overset {prefix} ownership must classify every non-hole cell as active."
            )
    if np.any(receptor_hole & np.asarray(overset.receptor_mask)):
        raise ValueError("Overset receptor cells cannot also be holes.")
    return (
        donor_active,
        donor_hole,
        donor_fringe,
        receptor_active,
        receptor_hole,
        receptor_fringe,
    )


def _validate_overset_epoch(
    overset: UnstructuredOversetPlan,
    discretization: UnstructuredFiniteVolumeDiscretization,
    /,
) -> None:
    """Reject maps compiled for a different receiver epoch."""

    for name in ("receptor_epoch_id", "receiver_epoch_id"):
        value = getattr(overset, name, None)
        if value is not None and value not in (
            discretization.prepared_id,
            discretization.geometry_id,
            discretization.topology_id,
        ):
            raise ValueError("Overset receptor map belongs to a stale geometry epoch.")


def _overset_identity(overset: UnstructuredOversetPlan, name: str, /) -> str:
    value = getattr(overset, name, None)
    if value is None and name == "policy_id":
        value = canonical_fingerprint(
            {
                "kind": "overset-interpolation-policy",
                "policy": getattr(overset, "interpolation_policy", "conservative"),
                "bounded": bool(getattr(overset, "bounded_interpolation", False)),
                "tolerance_id": getattr(overset, "tolerance_id", None),
            }
        )
    if value is None and name == "mapping_id":
        value = getattr(overset, "identity", None)
    if value is None:
        value = overset.plan_id
    if not isinstance(value, str) or not value:
        raise ValueError(f"Overset {name} must be a non-empty identifier.")
    return value


def _has_certified_receptor_face_routes(
    overset: UnstructuredOversetPlan | None,
    /,
) -> bool:
    if overset is None or overset.face_artifact_id is None:
        return False
    artifacts = (
        overset.receptor_face_ids,
        overset.receptor_face_points,
        overset.receptor_face_normals,
        overset.receptor_face_measures,
        overset.receptor_face_cells,
    )
    return all(artifact is not None for artifact in artifacts)


class UnstructuredFiniteVolumeCouplingPlan(StrictModule, NonTrainableState):
    """Identity-only coupling inputs for one unstructured finite-volume epoch."""

    motion: FixedConnectivityMotionPlan | None
    embedded_boundary: EmbeddedBoundaryPlan | None
    embedded_boundaries: UnstructuredEmbeddedBoundarySet | None
    vof: UnstructuredVOFPlan | None
    capillarity: BalancedCapillaryOperator | None
    contact_angles: EmbeddedBoundaryContactAngleSet | None
    amr: UnstructuredAMRHierarchyPlan | None
    overset: UnstructuredOversetPlan | None
    sliding: PeriodicSlidingInterfacePlan | None
    topology_event_capacity: int = eqx.field(static=True)
    topology_event_policy: str = eqx.field(static=True)
    topology_event_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        motion: FixedConnectivityMotionPlan | None = None,
        embedded_boundary: EmbeddedBoundaryPlan | None = None,
        embedded_boundaries: UnstructuredEmbeddedBoundarySet | None = None,
        vof: UnstructuredVOFPlan | None = None,
        capillarity: BalancedCapillaryOperator | None = None,
        contact_angles: EmbeddedBoundaryContactAngleSet | None = None,
        amr: UnstructuredAMRHierarchyPlan | None = None,
        overset: UnstructuredOversetPlan | None = None,
        sliding: PeriodicSlidingInterfacePlan | None = None,
        topology_event_capacity: int = 0,
        topology_event_policy: str = "disabled",
    ):
        motion_ = _optional_plan(motion, FixedConnectivityMotionPlan, "motion")
        embedded_boundary_ = _optional_plan(
            embedded_boundary, EmbeddedBoundaryPlan, "embedded_boundary"
        )
        embedded_boundaries_ = _optional_plan(
            embedded_boundaries,
            UnstructuredEmbeddedBoundarySet,
            "embedded_boundaries",
        )
        if (embedded_boundary_ is None) != (embedded_boundaries_ is None):
            raise ValueError(
                "Embedded-boundary coupling requires both an EmbeddedBoundaryPlan "
                "and UnstructuredEmbeddedBoundarySet."
            )
        vof_ = _optional_plan(vof, UnstructuredVOFPlan, "vof")
        capillarity_ = _optional_plan(
            capillarity, BalancedCapillaryOperator, "capillarity"
        )
        contact_angles_ = _optional_plan(
            contact_angles, EmbeddedBoundaryContactAngleSet, "contact_angles"
        )
        if capillarity_ is not None and vof_ is None:
            raise ValueError("Capillarity coupling requires a VOF plan.")
        if contact_angles_ is not None and (vof_ is None or embedded_boundary_ is None):
            raise ValueError(
                "Contact-angle coupling requires both VOF and embedded geometry."
            )
        amr_ = _optional_plan(amr, UnstructuredAMRHierarchyPlan, "amr")
        overset_ = _optional_plan(overset, UnstructuredOversetPlan, "overset")
        sliding_ = _optional_plan(sliding, PeriodicSlidingInterfacePlan, "sliding")
        capacity, policy = _topology_event_configuration(
            topology_event_capacity, topology_event_policy
        )
        topology_event_id = (
            None
            if policy == "disabled"
            else canonical_fingerprint(
                {
                    "kind": "unstructured-finite-volume-topology-events",
                    "schema_version": 1,
                    "capacity": capacity,
                    "policy": policy,
                }
            )
        )
        if sliding_ is not None and motion_ is None:
            raise ValueError("Sliding coupling requires a current motion identity.")
        if sliding_ is not None and policy != "accepted_step":
            raise ValueError("Sliding coupling requires accepted-step topology events.")

        self.motion = motion_
        self.embedded_boundary = embedded_boundary_
        self.embedded_boundaries = embedded_boundaries_
        self.vof = vof_
        self.capillarity = capillarity_
        self.contact_angles = contact_angles_
        self.amr = amr_
        self.overset = overset_
        self.sliding = sliding_
        self.topology_event_capacity = capacity
        self.topology_event_id = topology_event_id
        self.topology_event_policy = policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-finite-volume-coupling-plan",
                "schema_version": 2,
                "motion": None if motion_ is None else motion_.plan_id,
                "embedded_boundary": (
                    None if embedded_boundary_ is None else embedded_boundary_.plan_id
                ),
                "embedded_boundaries": (
                    None
                    if embedded_boundaries_ is None
                    else embedded_boundaries_.boundary_set_id
                ),
                "vof": None if vof_ is None else vof_.plan_id,
                "capillarity": (
                    None if capillarity_ is None else capillarity_.operator_id
                ),
                "contact_angles": (
                    None
                    if contact_angles_ is None
                    else contact_angles_.contact_angle_set_id
                ),
                "amr": None if amr_ is None else amr_.plan_id,
                "overset": None if overset_ is None else overset_.plan_id,
                "sliding": None if sliding_ is None else sliding_.plan_id,
                "topology_event_capacity": capacity,
                "topology_event_policy": policy,
            }
        )

    def coupled_component_ids(self, /) -> tuple[tuple[str, str], ...]:
        """Return enabled non-embedded subsystem identities in stable order."""
        components = (
            ("motion", None if self.motion is None else self.motion.plan_id),
            ("vof", None if self.vof is None else self.vof.plan_id),
            ("amr", None if self.amr is None else self.amr.plan_id),
            ("overset", None if self.overset is None else self.overset.plan_id),
            ("sliding", None if self.sliding is None else self.sliding.plan_id),
            ("topology_events", self.topology_event_id),
        )
        return tuple(
            (name, identifier)
            for name, identifier in components
            if identifier is not None
        )

    def validate_execution_support(self, /) -> None:
        """Reject subsystem combinations without a coupled execution path."""
        if self.amr is not None:
            raise ValueError(
                "AMR coupling requires PreparedUnstructuredAMRRuntime dispatch; "
                f"amr={self.amr.plan_id} cannot be ignored by ordinary FV."
            )
        moved_sliding_overset = (
            self.motion is not None
            and self.overset is not None
            and self.sliding is not None
            and _has_certified_receptor_face_routes(self.overset)
        )
        if self.sliding is not None and not moved_sliding_overset:
            raise ValueError(
                "Sliding coupling requires motion, overset, and a fully certified "
                "receptor-face artifact with explicit face IDs."
            )
        if (
            self.motion is not None
            and self.overset is not None
            and not moved_sliding_overset
        ):
            raise ValueError(
                "Motion plus overset requires stage-bound moved receptor artifacts "
                "with explicit physical face IDs and sliding interpolation."
            )
        embedded_boundary = self.embedded_boundary
        if embedded_boundary is None:
            return
        if (
            self.vof is not None
            and self.contact_angles is not None
            and self.amr is None
            and self.overset is None
            and self.sliding is None
            and self.motion is None
            and self.topology_event_id is None
        ):
            return
        conflicts = self.coupled_component_ids()
        if not conflicts:
            return
        conflict_ids = ", ".join(
            f"{component}={identifier}" for component, identifier in conflicts
        )
        raise ValueError(
            "Stationary embedded-boundary finite-volume execution does not support "
            "coupled subsystem execution "
            f"(embedded_boundary={embedded_boundary.plan_id}; {conflict_ids})."
        )

    def prepare(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        /,
        *,
        sliding_coupling: PeriodicSlidingCoupling | None = None,
    ) -> PreparedUnstructuredFiniteVolumeCoupling:
        prepared = PreparedUnstructuredFiniteVolumeCoupling(
            self,
            discretization,
            sliding_coupling=sliding_coupling,
        )
        self.validate_execution_support()
        return prepared


class PreparedUnstructuredFiniteVolumeCoupling(StrictModule, NonTrainableState):
    """A coupling plan certified against one prepared unstructured geometry."""

    motion: FixedConnectivityMotionPlan | None
    embedded_boundary: EmbeddedBoundaryPlan | None
    embedded_metrics: EmbeddedBoundaryMetrics | None
    embedded_stabilization_policy: EmbeddedBoundaryStabilizationPolicy | None
    embedded_boundaries: UnstructuredEmbeddedBoundarySet | None
    vof: UnstructuredVOFPlan | None
    capillarity: BalancedCapillaryOperator | None
    contact_angles: EmbeddedBoundaryContactAngleSet | None
    amr: UnstructuredAMRHierarchyPlan | None
    overset: UnstructuredOversetPlan | None
    sliding: PeriodicSlidingInterfacePlan | None
    sliding_coupling: PeriodicSlidingCoupling | None
    overset_policy_id: str | None = eqx.field(static=True)
    overset_mapping_id: str | None = eqx.field(static=True)
    overset_epoch_id: str | None = eqx.field(static=True)
    topology_event_capacity: int = eqx.field(static=True)
    topology_event_policy: str = eqx.field(static=True)
    topology_event_id: str | None = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    cut_boundary_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: UnstructuredFiniteVolumeCouplingPlan,
        discretization: UnstructuredFiniteVolumeDiscretization,
        /,
        *,
        sliding_coupling: PeriodicSlidingCoupling | None = None,
    ):
        if not isinstance(plan, UnstructuredFiniteVolumeCouplingPlan):
            raise TypeError("plan must be UnstructuredFiniteVolumeCouplingPlan.")
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError(
                "discretization must be UnstructuredFiniteVolumeDiscretization."
            )

        motion = plan.motion
        if motion is not None:
            base = motion.base_plan
            _validate_current_geometry(
                "Motion plan",
                base.topology_id,
                base.geometry_id,
                discretization,
            )

        embedded_boundary = plan.embedded_boundary
        embedded_boundaries = plan.embedded_boundaries
        embedded_metrics = None
        embedded_stabilization_policy = None
        cut_boundary_id = None
        if embedded_boundary is not None:
            if embedded_boundaries is None:
                raise ValueError(
                    "Embedded-boundary coupling requires cut-boundary ownership."
                )
            _validate_prepared_geometry(
                "Embedded-boundary plan",
                embedded_boundary.discretization,
                discretization,
            )
            metrics = embedded_boundary.prepare()
            if not isinstance(metrics, EmbeddedBoundaryMetrics):
                raise TypeError(
                    "EmbeddedBoundaryPlan.prepare() must return EmbeddedBoundaryMetrics."
                )
            if (
                metrics.prepared_id != discretization.prepared_id
                or metrics.topology_id != discretization.topology_id
                or metrics.geometry_id != discretization.geometry_id
            ):
                raise ValueError(
                    "Embedded-boundary metrics belong to stale prepared geometry."
                )
            if (
                metrics.field_id != embedded_boundary.field_id
                or metrics.body_tag != embedded_boundary.body_tag
            ):
                raise ValueError(
                    "Embedded-boundary metrics do not match the field/body identity."
                )
            stabilization_policy = embedded_boundary.stabilization_policy
            if metrics.stabilization_policy_id != stabilization_policy.policy_id:
                raise ValueError(
                    "Embedded-boundary metrics do not match the stabilization policy."
                )
            evidence_passed = bool(np.asarray(metrics.evidence.passed))
            evidence_status = int(np.asarray(metrics.evidence.status))
            if not evidence_passed or evidence_status != int(
                EmbeddedBoundaryStatus.SUCCESS
            ):
                raise ValueError(
                    "Embedded-boundary compilation requires SUCCESS metric evidence."
                )
            metric_body_tags = frozenset(
                int(value) for value in np.asarray(metrics.body_tags).reshape(-1)
            )
            policy_body_tags = frozenset(embedded_boundaries.body_tags)
            if metric_body_tags != policy_body_tags:
                metric_tags = ", ".join(str(value) for value in sorted(metric_body_tags))
                policy_tags = ", ".join(str(value) for value in sorted(policy_body_tags))
                missing_tags = ", ".join(
                    str(value)
                    for value in sorted(metric_body_tags.difference(policy_body_tags))
                )
                extra_tags = ", ".join(
                    str(value)
                    for value in sorted(policy_body_tags.difference(metric_body_tags))
                )
                raise ValueError(
                    "Embedded-boundary metric and cut-boundary policy body-tag sets "
                    "must match exactly "
                    f"(metric_body_tags={{{metric_tags}}}, "
                    f"cut_boundary_policy_body_tags={{{policy_tags}}}); "
                    "no cut-boundary policy for metric body tag(s): "
                    f"{missing_tags or 'none'}; extra cut-boundary policy body tag(s): "
                    f"{extra_tags or 'none'}."
                )
            embedded_metrics = metrics
            embedded_stabilization_policy = stabilization_policy
            cut_boundary_id = embedded_boundaries.boundary_set_id

        vof = plan.vof
        if vof is not None:
            _validate_prepared_geometry(
                "VOF plan",
                vof.discretization,
                discretization,
            )
            _validate_prepared_geometry(
                "VOF gradient",
                vof.gradient.discretization,
                discretization,
            )
        capillarity = plan.capillarity
        contact_angles = plan.contact_angles
        if capillarity is not None:
            if vof is None:
                raise ValueError("Prepared capillarity requires a VOF plan.")
            if capillarity.discretization.prepared_id != discretization.prepared_id:
                raise ValueError("Capillary operator belongs to stale geometry.")
        if contact_angles is not None:
            if embedded_metrics is None or vof is None:
                raise ValueError(
                    "Prepared contact angles require embedded metrics and VOF."
                )
            contact_angles.validate_bindings(
                embedded_metrics.geometry_id,
                vof.plan_id,
            )
            contact_angles.validate_body_tags(embedded_metrics.body_tags)

        amr = plan.amr
        overset = plan.overset
        overset_policy_id = None
        overset_mapping_id = None
        overset_epoch_id = None
        if overset is not None:
            _validate_current_geometry(
                "Overset receptor",
                overset.receptor_topology_id,
                overset.receptor_geometry_id,
                discretization,
            )
            _validate_overset_epoch(overset, discretization)
            (
                donor_active,
                donor_hole,
                donor_fringe,
                receptor_active,
                receptor_hole,
                receptor_fringe,
            ) = _overset_masks(
                overset,
                int(np.asarray(overset.donor_covered_measures).size),
                int(discretization.cell_count),
            )
            del donor_active, donor_hole, donor_fringe, receptor_fringe
            if overset.donor_topology_id == discretization.topology_id:
                _validate_current_geometry(
                    "Overset donor",
                    overset.donor_topology_id,
                    overset.donor_geometry_id,
                    discretization,
                )
            elif overset.donor_covered_measures.shape[0] != discretization.cell_count:
                raise ValueError(
                    "Single-device overset donors must share the compiled cell layout."
                )
            if receptor_active.shape != (discretization.cell_count,) or np.any(
                receptor_active & receptor_hole
            ):
                raise ValueError("Overset receptor active ownership must exclude holes.")
            overset_policy_id = _overset_identity(overset, "policy_id")
            overset_mapping_id = _overset_identity(overset, "mapping_id")
            overset_epoch_id = getattr(overset, "epoch_id", None)
            if overset_epoch_id is not None and not isinstance(overset_epoch_id, str):
                raise ValueError("Overset epoch_id must be a string.")
        sliding_plan = plan.sliding
        if sliding_plan is None:
            if sliding_coupling is not None:
                raise ValueError(
                    "An explicit sliding coupling requires a sliding interface plan."
                )
            current_sliding_coupling = None
        else:
            current_sliding_coupling = (
                sliding_plan.coupling(0.0)
                if sliding_coupling is None
                else sliding_coupling
            )
            if not isinstance(current_sliding_coupling, PeriodicSlidingCoupling):
                raise TypeError(
                    "sliding_coupling must be PeriodicSlidingCoupling or None."
                )
            certified_coupling = sliding_plan.coupling(
                current_sliding_coupling.normalized_shift
            )
            if (
                current_sliding_coupling.coupling_id != certified_coupling.coupling_id
                or current_sliding_coupling.evidence_id != certified_coupling.evidence_id
                or current_sliding_coupling.shift_precision
                != sliding_plan.shift_precision
                or not bool(np.asarray(current_sliding_coupling.coverage_passed))
                or int(np.asarray(current_sliding_coupling.status)) != 0
            ):
                raise ValueError(
                    "Explicit sliding coupling is stale or does not belong to the "
                    "prepared sliding interface plan."
                )
            if overset is None or not _has_certified_receptor_face_routes(overset):
                raise ValueError(
                    "Sliding coupling requires a fully certified overset "
                    "receptor-face artifact with explicit physical face IDs."
                )
            if int(np.asarray(current_sliding_coupling.left_measures).size) != int(
                np.asarray(overset.donor_indices).size
            ) or int(np.asarray(current_sliding_coupling.right_measures).size) != int(
                np.asarray(overset.receptor_face_ids).size
            ):
                raise ValueError(
                    "Sliding left/right interval routes must match overset donor "
                    "routes and certified receptor faces."
                )

        self.motion = motion
        self.embedded_boundary = embedded_boundary
        self.embedded_metrics = embedded_metrics
        self.embedded_stabilization_policy = embedded_stabilization_policy
        self.embedded_boundaries = embedded_boundaries
        self.vof = vof
        self.capillarity = capillarity
        self.contact_angles = contact_angles
        self.amr = amr
        self.overset = overset
        self.sliding = sliding_plan
        self.sliding_coupling = current_sliding_coupling
        self.overset_policy_id = overset_policy_id
        self.overset_mapping_id = overset_mapping_id
        self.overset_epoch_id = overset_epoch_id
        self.topology_event_capacity = plan.topology_event_capacity
        self.topology_event_policy = plan.topology_event_policy
        self.topology_event_id = plan.topology_event_id
        self.topology_id = discretization.topology_id
        self.geometry_id = discretization.geometry_id
        self.discretization_id = discretization.prepared_id
        self.cut_boundary_id = cut_boundary_id
        self.plan_id = plan.plan_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-finite-volume-coupling",
                "schema_version": 4,
                "plan": plan.plan_id,
                "topology": discretization.topology_id,
                "geometry": discretization.geometry_id,
                "discretization": discretization.prepared_id,
                "motion": None if motion is None else motion.plan_id,
                "embedded_boundary": (
                    None if embedded_boundary is None else embedded_boundary.plan_id
                ),
                "embedded_metrics": (
                    None if embedded_metrics is None else embedded_metrics.metrics_id
                ),
                "embedded_stabilization_policy": (
                    None
                    if embedded_stabilization_policy is None
                    else embedded_stabilization_policy.policy_id
                ),
                "cut_boundaries": cut_boundary_id,
                "vof": None if vof is None else vof.plan_id,
                "capillarity": (None if capillarity is None else capillarity.operator_id),
                "contact_angles": (
                    None
                    if contact_angles is None
                    else contact_angles.contact_angle_set_id
                ),
                "amr": None if amr is None else amr.plan_id,
                "overset": None if overset is None else overset.plan_id,
                "overset_policy": overset_policy_id,
                "overset_mapping": overset_mapping_id,
                "overset_epoch": overset_epoch_id,
                "sliding": None if plan.sliding is None else plan.sliding.plan_id,
                "sliding_coupling": (
                    None
                    if current_sliding_coupling is None
                    else current_sliding_coupling.coupling_id
                ),
                "topology_event_capacity": plan.topology_event_capacity,
                "topology_event_policy": plan.topology_event_policy,
            }
        )

    def with_sliding_coupling(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        sliding_coupling: PeriodicSlidingCoupling,
        /,
    ) -> "PreparedUnstructuredFiniteVolumeCoupling":
        """Reprepare this coupling plan with one accepted frozen overlap map."""

        plan = UnstructuredFiniteVolumeCouplingPlan(
            motion=self.motion,
            embedded_boundary=self.embedded_boundary,
            embedded_boundaries=self.embedded_boundaries,
            vof=self.vof,
            amr=self.amr,
            overset=self.overset,
            sliding=self.sliding,
            topology_event_capacity=self.topology_event_capacity,
            topology_event_policy=self.topology_event_policy,
        )
        if plan.plan_id != self.plan_id:
            raise RuntimeError("Prepared coupling plan identity cannot be reconstructed.")
        return PreparedUnstructuredFiniteVolumeCoupling(
            plan,
            discretization,
            sliding_coupling=sliding_coupling,
        )


__all__ = [
    "PreparedUnstructuredFiniteVolumeCoupling",
    "UnstructuredFiniteVolumeCouplingPlan",
]
