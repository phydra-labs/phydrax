#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume._distributed_marker_transfer import (
    DistributedMACMarkerTransfer,
)
from ...discretization.lattice_boltzmann._immersed import ImmersedBoundaryForcingPlan
from ...discretization.particle._resolved_lubrication import (
    ResolvedLubricationCorrectionPlan,
)
from ...equations._mac_penalty_ib_cfd_dem import MACPenaltyIBCFDEMCouplingPlan
from ...qualification._registry import SupportTuple
from ...solver._mac_immersed_boundary import MACImmersedBoundaryProjectionPlan
from ...solver._mac_immersed_contact import MACRigidImmersedContactMethod
from ...solver._mac_immersed_deformable import MACDeformableImmersedBackwardEulerMethod
from ...solver._mac_immersed_rigid import (
    MACRigidImmersedBackwardEulerMethod,
    MACRigidImmersedEulerMethod,
    MACRigidImmersedMidpointMethod,
    MACRigidImmersedProjectionPlan,
)
from ...solver._mac_sharp_interface import (
    MACMovingSharpInterfaceEpochPlan,
    MACSharpInterfaceProjectionPlan,
)
from ...solver._marker_flow_runtime import HydrodynamicLoadPlan


ImmersedRegime: TypeAlias = Literal[
    "prescribed-marker",
    "free-rigid-marker",
    "fixed-topology-sharp",
    "deformable-contact",
    "lbm-body",
    "resolved-cfd-dem",
]
ImmersedDerivativeMode: TypeAlias = Literal["none", "jvp", "vjp"]
ImmersedOwnerPlan: TypeAlias = (
    MACImmersedBoundaryProjectionPlan
    | MACRigidImmersedProjectionPlan
    | MACRigidImmersedEulerMethod
    | MACRigidImmersedBackwardEulerMethod
    | MACRigidImmersedMidpointMethod
    | MACSharpInterfaceProjectionPlan
    | MACDeformableImmersedBackwardEulerMethod
    | MACRigidImmersedContactMethod
    | ImmersedBoundaryForcingPlan
    | MACPenaltyIBCFDEMCouplingPlan
)


_PRESCRIBED_MARKER_ATTRIBUTES = {
    "regime": "prescribed-marker",
    "discretization": "staggered-mac",
    "enforcement": "pressure-marker-kkt",
    "motion": "prescribed",
    "geometry": "regularized-lagrangian-markers",
    "topology": "fixed-marker-identities",
    "contact": "none",
    "load_path": "marker-reaction",
    "derivative_scope": "fixed-route-jvp-vjp",
    "distributed": "owner-computes-explicit-reduction",
    "qualification_state": "candidate",
}
_FREE_RIGID_MARKER_ATTRIBUTES = {
    "regime": "free-rigid-marker",
    "discretization": "staggered-mac",
    "enforcement": "simultaneous-fluid-rigid-marker-kkt",
    "motion": "free-rigid",
    "geometry": "prepared-rigid-marker-map",
    "topology": "fixed-marker-identities",
    "contact": "optional-hard-contact",
    "load_path": "marker-reaction",
    "derivative_scope": "fixed-route-jvp-vjp-without-contact",
    "distributed": "owner-computes-explicit-reduction",
    "qualification_state": "candidate",
}
_FIXED_TOPOLOGY_SHARP_ATTRIBUTES = {
    "regime": "fixed-topology-sharp",
    "discretization": "sharp-cut-cell-mac",
    "enforcement": "weighted-compatible-projection",
    "motion": "fixed-or-epoch-frozen",
    "geometry": "qualified-absolute-sharp-measures",
    "topology": "fixed-active-set",
    "contact": "none",
    "load_path": "pressure-viscous-traction",
    "derivative_scope": "fixed-topology-jvp-vjp",
    "distributed": "not-admitted",
    "qualification_state": "candidate",
}
_DEFORMABLE_CONTACT_ATTRIBUTES = {
    "regime": "deformable-contact",
    "discretization": "staggered-mac-finite-element-markers",
    "enforcement": "accepted-time-monolithic-fsi",
    "motion": "deformable-or-rigid-contact",
    "geometry": "prepared-material-marker-map",
    "topology": "fixed-marker-identities",
    "contact": "explicit-qualified-contact-state",
    "load_path": "marker-contact-lubrication",
    "derivative_scope": "fixed-route-without-active-contact",
    "distributed": "owner-computes-explicit-reduction",
    "qualification_state": "candidate",
}
_LBM_BODY_ATTRIBUTES = {
    "regime": "lbm-body",
    "discretization": "lattice-boltzmann-cartesian",
    "enforcement": "iterated-direct-forcing",
    "motion": "prescribed-or-rigid",
    "geometry": "regularized-lagrangian-markers",
    "topology": "fixed-marker-count",
    "contact": "none",
    "load_path": "direct-forcing-ledger",
    "derivative_scope": "none",
    "distributed": "not-admitted",
    "qualification_state": "candidate",
}
_RESOLVED_CFD_DEM_ATTRIBUTES = {
    "regime": "resolved-cfd-dem",
    "discretization": "staggered-mac-soft-sphere-dem",
    "enforcement": "resolved-penalty-marker-coupling",
    "motion": "free-rigid-spheres",
    "geometry": "prepared-sphere-marker-map",
    "topology": "fixed-capacity-contact-graph",
    "contact": "soft-sphere-contact",
    "load_path": "marker-contact-lubrication-ledger",
    "derivative_scope": "none",
    "distributed": "owner-computes-explicit-reduction",
    "qualification_state": "candidate",
}

PRESCRIBED_MARKER_SUPPORT_TUPLE = SupportTuple(
    "immersed-dns", _PRESCRIBED_MARKER_ATTRIBUTES
)
FREE_RIGID_MARKER_SUPPORT_TUPLE = SupportTuple(
    "immersed-dns", _FREE_RIGID_MARKER_ATTRIBUTES
)
FIXED_TOPOLOGY_SHARP_SUPPORT_TUPLE = SupportTuple(
    "immersed-dns", _FIXED_TOPOLOGY_SHARP_ATTRIBUTES
)
DEFORMABLE_CONTACT_SUPPORT_TUPLE = SupportTuple(
    "immersed-dns", _DEFORMABLE_CONTACT_ATTRIBUTES
)
LBM_BODY_SUPPORT_TUPLE = SupportTuple("immersed-dns", _LBM_BODY_ATTRIBUTES)
RESOLVED_CFD_DEM_SUPPORT_TUPLE = SupportTuple(
    "immersed-dns", _RESOLVED_CFD_DEM_ATTRIBUTES
)
IMMERSED_DNS_SUPPORT_TUPLES = (
    PRESCRIBED_MARKER_SUPPORT_TUPLE,
    FREE_RIGID_MARKER_SUPPORT_TUPLE,
    FIXED_TOPOLOGY_SHARP_SUPPORT_TUPLE,
    DEFORMABLE_CONTACT_SUPPORT_TUPLE,
    LBM_BODY_SUPPORT_TUPLE,
    RESOLVED_CFD_DEM_SUPPORT_TUPLE,
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _rigid_projection(
    owner: MACRigidImmersedProjectionPlan
    | MACRigidImmersedEulerMethod
    | MACRigidImmersedBackwardEulerMethod
    | MACRigidImmersedMidpointMethod
    | MACRigidImmersedContactMethod,
    /,
) -> MACRigidImmersedProjectionPlan:
    if isinstance(owner, MACRigidImmersedProjectionPlan):
        return owner
    if isinstance(owner, MACRigidImmersedEulerMethod):
        return owner.projection
    if isinstance(owner, MACRigidImmersedBackwardEulerMethod):
        return owner.base.projection
    if isinstance(owner, MACRigidImmersedMidpointMethod):
        return owner.backward_euler.base.projection
    immersed = owner.immersed
    if isinstance(immersed, MACRigidImmersedMidpointMethod):
        return immersed.backward_euler.base.projection
    return immersed.base.projection


def _owner_contract(
    owner: ImmersedOwnerPlan,
    /,
) -> tuple[ImmersedRegime, str, str | None, str | None, bool]:
    if isinstance(owner, MACImmersedBoundaryProjectionPlan):
        return (
            "prescribed-marker",
            owner.plan_id,
            owner.transfer.markers.prepared_id,
            None,
            False,
        )
    if isinstance(
        owner,
        (
            MACRigidImmersedProjectionPlan,
            MACRigidImmersedEulerMethod,
            MACRigidImmersedBackwardEulerMethod,
            MACRigidImmersedMidpointMethod,
        ),
    ):
        projection = _rigid_projection(owner)
        owner_id = (
            owner.plan_id
            if isinstance(owner, MACRigidImmersedProjectionPlan)
            else owner.method_id
        )
        return (
            "free-rigid-marker",
            owner_id,
            projection.transfer.markers.prepared_id,
            None,
            False,
        )
    if isinstance(owner, MACSharpInterfaceProjectionPlan):
        return (
            "fixed-topology-sharp",
            owner.plan_id,
            None,
            owner.geometry.realization_id,
            False,
        )
    if isinstance(owner, MACDeformableImmersedBackwardEulerMethod):
        return (
            "deformable-contact",
            owner.method_id,
            owner.projection.transfer.markers.prepared_id,
            None,
            owner.structural_contact_residual is not None,
        )
    if isinstance(owner, MACRigidImmersedContactMethod):
        projection = _rigid_projection(owner)
        return (
            "deformable-contact",
            owner.method_id,
            projection.transfer.markers.prepared_id,
            None,
            True,
        )
    if isinstance(owner, ImmersedBoundaryForcingPlan):
        return "lbm-body", owner.plan_id, None, None, False
    if isinstance(owner, MACPenaltyIBCFDEMCouplingPlan):
        return (
            "resolved-cfd-dem",
            owner.plan_id,
            owner.transfer.markers.prepared_id,
            None,
            True,
        )
    raise TypeError("owner must be a supported prepared immersed-flow plan.")


def _support_tuple(regime: ImmersedRegime, /) -> SupportTuple:
    support_by_regime = {
        "prescribed-marker": PRESCRIBED_MARKER_SUPPORT_TUPLE,
        "free-rigid-marker": FREE_RIGID_MARKER_SUPPORT_TUPLE,
        "fixed-topology-sharp": FIXED_TOPOLOGY_SHARP_SUPPORT_TUPLE,
        "deformable-contact": DEFORMABLE_CONTACT_SUPPORT_TUPLE,
        "lbm-body": LBM_BODY_SUPPORT_TUPLE,
        "resolved-cfd-dem": RESOLVED_CFD_DEM_SUPPORT_TUPLE,
    }
    return support_by_regime[regime]


def _owner_transfer(owner: ImmersedOwnerPlan, /):
    if isinstance(owner, MACImmersedBoundaryProjectionPlan):
        return owner.transfer
    if isinstance(
        owner,
        (
            MACRigidImmersedProjectionPlan,
            MACRigidImmersedEulerMethod,
            MACRigidImmersedBackwardEulerMethod,
            MACRigidImmersedMidpointMethod,
            MACRigidImmersedContactMethod,
        ),
    ):
        return _rigid_projection(owner).transfer
    if isinstance(owner, MACDeformableImmersedBackwardEulerMethod):
        return owner.projection.transfer
    if isinstance(owner, MACPenaltyIBCFDEMCouplingPlan):
        return owner.transfer
    return None


class ImmersedNearGapRegime(IntEnum):
    RESOLVED_GRID = 0
    LUBRICATION = 1
    CONTACT = 2
    INADMISSIBLE = 3


class ImmersedNearGapDecision(StrictModule):
    regime: Array
    resolved_grid: Array
    lubrication: Array
    contact: Array
    finite: Array
    admissible: Array
    plan_id: str = eqx.field(static=True)


class ImmersedBodyRegimePlan(StrictModule, NonTrainableState):
    """Bind one existing immersed owner to exact support and runtime identities.

    This object never evaluates the owner. It only carries the already-prepared plan
    and validates that qualification/admission evidence names that same plan.
    """

    owner: ImmersedOwnerPlan
    sharp_epoch_owner: MACMovingSharpInterfaceEpochPlan | None
    support_tuple: SupportTuple
    distributed_transfer: DistributedMACMarkerTransfer | None
    lubrication: ResolvedLubricationCorrectionPlan | None
    regime: ImmersedRegime = eqx.field(static=True)
    owner_plan_id: str = eqx.field(static=True)
    marker_set_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    topology_epoch_id: str = eqx.field(static=True)
    motion_epoch_id: str = eqx.field(static=True)
    geometry_epoch: int = eqx.field(static=True)
    moving: bool = eqx.field(static=True)
    fixed_topology: bool = eqx.field(static=True)
    contact_capable: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        owner: ImmersedOwnerPlan,
        /,
        *,
        marker_set_id: str,
        geometry_id: str,
        route_id: str,
        topology_epoch_id: str,
        geometry_epoch: int,
        moving: bool,
        motion_epoch_id: str | None = None,
        sharp_epoch_owner: MACMovingSharpInterfaceEpochPlan | None = None,
        fixed_topology: bool = True,
        distributed_transfer: DistributedMACMarkerTransfer | None = None,
        lubrication: ResolvedLubricationCorrectionPlan | None = None,
    ):
        regime, owner_id, bound_markers, bound_geometry, contact_capable = (
            _owner_contract(owner)
        )
        marker_id = _identifier(marker_set_id, "marker_set_id")
        geometry = _identifier(geometry_id, "geometry_id")
        route = _identifier(route_id, "route_id")
        topology = _identifier(topology_epoch_id, "topology_epoch_id")
        epoch = int(geometry_epoch)
        if epoch < 0:
            raise ValueError("geometry_epoch must be non-negative.")
        fixed = bool(fixed_topology)
        moving_ = bool(moving)
        if moving_ and motion_epoch_id is None:
            raise ValueError("Moving bodies require an explicit motion_epoch_id.")
        motion_epoch = (
            topology
            if motion_epoch_id is None
            else _identifier(motion_epoch_id, "motion_epoch_id")
        )
        if bound_markers is not None and marker_id != bound_markers:
            raise ValueError("marker_set_id does not match the prepared immersed owner.")
        if bound_geometry is not None and geometry != bound_geometry:
            raise ValueError("geometry_id does not match the prepared sharp geometry.")
        if regime == "fixed-topology-sharp" and not fixed:
            raise ValueError("Sharp cut-cell admission is limited to fixed topology.")
        if sharp_epoch_owner is not None:
            if not isinstance(sharp_epoch_owner, MACMovingSharpInterfaceEpochPlan):
                raise TypeError(
                    "sharp_epoch_owner must be MACMovingSharpInterfaceEpochPlan or None."
                )
            if regime != "fixed-topology-sharp":
                raise ValueError("A sharp epoch owner requires the sharp regime.")
            if (
                sharp_epoch_owner.operators.prepared_id != owner.operators.prepared_id
                or sharp_epoch_owner.boundaries.prepared_id
                != owner.boundaries.prepared_id
            ):
                raise ValueError(
                    "Sharp projection and moving epoch owner bind different operators."
                )
        if regime == "fixed-topology-sharp" and moving_ and sharp_epoch_owner is None:
            raise ValueError("Moving sharp bodies require an explicit epoch owner.")
        if distributed_transfer is not None:
            if not isinstance(distributed_transfer, DistributedMACMarkerTransfer):
                raise TypeError(
                    "distributed_transfer must be DistributedMACMarkerTransfer or None."
                )
            transfer = _owner_transfer(owner)
            if transfer is None or (
                distributed_transfer.local.prepared_id != transfer.prepared_id
            ):
                raise ValueError(
                    "Distributed transfer and immersed owner must share local transfer."
                )
        if lubrication is not None and not isinstance(
            lubrication, ResolvedLubricationCorrectionPlan
        ):
            raise TypeError(
                "lubrication must be ResolvedLubricationCorrectionPlan or None."
            )
        if lubrication is not None and regime not in (
            "free-rigid-marker",
            "deformable-contact",
            "resolved-cfd-dem",
        ):
            raise ValueError(
                "Lubrication is only valid for moving material-body regimes."
            )
        self.owner = owner
        self.sharp_epoch_owner = sharp_epoch_owner
        self.support_tuple = _support_tuple(regime)
        self.distributed_transfer = distributed_transfer
        self.lubrication = lubrication
        self.regime = regime
        self.owner_plan_id = owner_id
        self.marker_set_id = marker_id
        self.geometry_id = geometry
        self.route_id = route
        self.topology_epoch_id = topology
        self.motion_epoch_id = motion_epoch
        self.geometry_epoch = epoch
        self.moving = moving_
        self.fixed_topology = fixed
        self.contact_capable = contact_capable
        self.plan_id = canonical_fingerprint(
            {
                "kind": "immersed-body-regime-plan",
                "owner": owner_id,
                "support_tuple": self.support_tuple.support_tuple_id,
                "marker_set_id": marker_id,
                "geometry_id": geometry,
                "route_id": route,
                "topology_epoch_id": topology,
                "motion_epoch_id": motion_epoch,
                "geometry_epoch": epoch,
                "moving": moving_,
                "fixed_topology": fixed,
                "sharp_epoch_owner": (
                    None if sharp_epoch_owner is None else sharp_epoch_owner.plan_id
                ),
                "distributed_transfer": (
                    None if distributed_transfer is None else distributed_transfer.plan_id
                ),
                "lubrication": None if lubrication is None else lubrication.plan_id,
            }
        )

    @property
    def marker_constraint_count(self) -> int:
        transfer = _owner_transfer(self.owner)
        return 0 if transfer is None else transfer.markers.active_velocity_space.size

    @property
    def estimated_marker_resource_bytes(self) -> int:
        transfer = _owner_transfer(self.owner)
        if transfer is None:
            return 0
        marker_count = transfer.markers.active_count
        dimension = transfer.dimension
        itemsize = np.dtype(transfer.operators.pressure_space.dtype).itemsize
        relation = dimension * marker_count * transfer.route_width
        constraint_count = transfer.markers.active_velocity_space.size
        rank_workspace = 3 * constraint_count * constraint_count * itemsize
        return int(
            relation * (np.dtype(np.int32).itemsize + (2 * dimension + 2) * itemsize + 1)
            + sum(value.size * itemsize for value in transfer.flattened_dual_measures)
            + rank_workspace
        )

    def load_plan(
        self,
        body_ids: ArrayLike,
        ambient_dimension: int,
        /,
        *,
        reference_point_id: str,
        tolerance: float = 1.0e-9,
    ) -> HydrodynamicLoadPlan:
        """Create provenance-bound load assembly without re-running a fluid solve."""

        return HydrodynamicLoadPlan(
            body_ids,
            ambient_dimension,
            marker_set_id=self.marker_set_id,
            geometry_id=self.geometry_id,
            route_id=self.route_id,
            topology_epoch_id=self.topology_epoch_id,
            reference_point_id=reference_point_id,
            tolerance=tolerance,
        )

    def classify_gap(self, gap: ArrayLike, /) -> ImmersedNearGapDecision:
        """Classify resolved/lubrication/contact crossover without applying forces."""

        values = jnp.asarray(gap)
        if values.size == 0:
            raise ValueError("gap must contain at least one body-pair separation.")
        finite = jnp.isfinite(values)
        if self.lubrication is None:
            resolved = finite & (values > 0.0)
            lubrication = jnp.zeros_like(resolved)
            contact = finite & (values <= 0.0) & self.contact_capable
        else:
            cutoff = jnp.broadcast_to(self.lubrication.cutoff, values.shape)
            minimum = jnp.broadcast_to(self.lubrication.minimum_gap, values.shape)
            resolved = finite & (values >= cutoff)
            lubrication = finite & (values > minimum) & (values < cutoff)
            contact = finite & (values <= minimum) & self.contact_capable
        admissible = resolved | lubrication | contact
        regime = jnp.where(
            resolved,
            int(ImmersedNearGapRegime.RESOLVED_GRID),
            jnp.where(
                lubrication,
                int(ImmersedNearGapRegime.LUBRICATION),
                jnp.where(
                    contact,
                    int(ImmersedNearGapRegime.CONTACT),
                    int(ImmersedNearGapRegime.INADMISSIBLE),
                ),
            ),
        )
        return ImmersedNearGapDecision(
            regime,
            resolved,
            lubrication,
            contact,
            jnp.all(finite),
            jnp.all(admissible),
            self.plan_id,
        )


__all__ = [
    "DEFORMABLE_CONTACT_SUPPORT_TUPLE",
    "FIXED_TOPOLOGY_SHARP_SUPPORT_TUPLE",
    "FREE_RIGID_MARKER_SUPPORT_TUPLE",
    "IMMERSED_DNS_SUPPORT_TUPLES",
    "ImmersedBodyRegimePlan",
    "ImmersedDerivativeMode",
    "ImmersedNearGapDecision",
    "ImmersedNearGapRegime",
    "ImmersedOwnerPlan",
    "ImmersedRegime",
    "LBM_BODY_SUPPORT_TUPLE",
    "PRESCRIBED_MARKER_SUPPORT_TUPLE",
    "RESOLVED_CFD_DEM_SUPPORT_TUPLE",
]
