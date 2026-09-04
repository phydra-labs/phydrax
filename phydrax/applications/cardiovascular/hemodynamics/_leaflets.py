#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.finite_volume._mac_cut_cell import (
    MACDiffuseSDFGeometryPlan,
    MACDiffuseSDFGeometryState,
)
from ....solver._mac_deformable_contact import (
    DeformableContactResidualEvaluation,
    DeformableContactResidualPlan,
)
from ._immersed_fsi import PreparedSparseMarkerTransfer, SparseMarkerRelation


class LeafletTransitionStatus(IntEnum):
    SUCCESS = 0
    STRUCTURAL_FAILURE = 1
    CONTACT_FAILURE = 2
    LEAKAGE_FAILURE = 3
    REFINEMENT_REQUIRED = 4
    NONFINITE = 5


class ImmersedLeafletFluidState(StrictModule):
    relation: SparseMarkerRelation


class CutCellLeafletFluidState(StrictModule):
    geometry: MACDiffuseSDFGeometryState


LeafletFluidState: TypeAlias = ImmersedLeafletFluidState | CutCellLeafletFluidState
LeafletKinematics = Callable[[Array, Array, Any], tuple[ArrayLike, ArrayLike]]
ImmersedLeakageProbe = Callable[[SparseMarkerRelation, Array, Array, Any], ArrayLike]
CutCellGeometryArguments = Callable[[Array, Array, Any], Any]


class LeafletFluidEvidence(StrictModule):
    """Route coverage, leakage proxy, GCL, and refinement qualification.

    ``leakage_proxy`` is a resolved open-area or user-supplied surrogate. It is
    evidence at the prepared resolution, never an exact sealing certificate.
    """

    minimum_coverage: Array
    leakage_proxy: Array
    geometric_conservation_residual: Array
    small_cell_fraction: Array
    leakage_qualified: Array
    refinement_required: Array
    finite: Array
    successful: Array
    exact_sealing_certified: bool = eqx.field(static=True)
    route_id: str = eqx.field(static=True)


class ImmersedLeafletRoute(StrictModule, NonTrainableState):
    """Fixed-stencil immersed leaflet geometry and resolved leakage probe."""

    transfer: PreparedSparseMarkerTransfer
    kinematics: LeafletKinematics = eqx.field(static=True)
    leakage_probe: ImmersedLeakageProbe = eqx.field(static=True)
    maximum_leakage_proxy: float = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedSparseMarkerTransfer,
        kinematics: LeafletKinematics,
        leakage_probe: ImmersedLeakageProbe,
        /,
        *,
        maximum_leakage_proxy: float,
        route_id: str = "cardiovascular-immersed-leaflet",
    ):
        if not isinstance(transfer, PreparedSparseMarkerTransfer):
            raise TypeError("transfer must be PreparedSparseMarkerTransfer.")
        if not callable(kinematics) or not callable(leakage_probe):
            raise TypeError("Immersed leaflet adapters must be callable.")
        leakage = float(maximum_leakage_proxy)
        identifier = str(route_id)
        if not np.isfinite(leakage) or leakage < 0.0:
            raise ValueError("maximum_leakage_proxy must be finite and non-negative.")
        if not identifier:
            raise ValueError("route_id must be non-empty.")
        self.transfer = transfer
        self.kinematics = kinematics
        self.leakage_probe = leakage_probe
        self.maximum_leakage_proxy = leakage
        self.route_id = identifier

    def initialize(
        self,
        configuration: Array,
        velocity: Array,
        time: ArrayLike,
        args: Any,
        /,
    ) -> ImmersedLeafletFluidState:
        del time
        position, marker_velocity = self.kinematics(configuration, velocity, args)
        position_ = jnp.asarray(position)
        velocity_ = jnp.asarray(marker_velocity)
        if position_.shape != velocity_.shape:
            raise ValueError("Immersed leaflet position and velocity shapes differ.")
        return ImmersedLeafletFluidState(self.transfer.relation(position_))

    def evaluate(
        self,
        configuration: Array,
        velocity: Array,
        time: ArrayLike,
        step_size: ArrayLike,
        previous: ImmersedLeafletFluidState,
        args: Any,
        /,
    ) -> tuple[ImmersedLeafletFluidState, LeafletFluidEvidence]:
        del time, step_size
        if not isinstance(previous, ImmersedLeafletFluidState):
            raise TypeError("Immersed route requires ImmersedLeafletFluidState.")
        position, marker_velocity = self.kinematics(configuration, velocity, args)
        position_ = jnp.asarray(position)
        velocity_ = jnp.asarray(marker_velocity)
        if position_.shape != velocity_.shape:
            raise ValueError("Immersed leaflet position and velocity shapes differ.")
        relation = self.transfer.relation(position_)
        leakage = jnp.asarray(
            self.leakage_probe(relation, configuration, velocity, args),
            dtype=position_.dtype,
        ).reshape(())
        active_coverage = jnp.where(
            relation.active,
            relation.evidence.coverage_fraction,
            jnp.inf,
        )
        minimum_coverage = jnp.min(active_coverage)
        leakage_qualified = (
            jnp.isfinite(leakage)
            & (leakage >= 0.0)
            & (leakage <= self.maximum_leakage_proxy)
        )
        refinement = ~relation.evidence.successful | (
            minimum_coverage < self.transfer.plan.minimum_coverage
        )
        finite = jnp.all((~relation.active) | relation.evidence.finite) & jnp.isfinite(
            leakage
        )
        successful = (
            finite & relation.evidence.successful & leakage_qualified & ~refinement
        )
        evidence = LeafletFluidEvidence(
            minimum_coverage,
            leakage,
            jnp.asarray(0.0, dtype=position_.dtype),
            jnp.asarray(0.0, dtype=position_.dtype),
            leakage_qualified,
            refinement,
            finite,
            successful,
            False,
            self.route_id,
        )
        return ImmersedLeafletFluidState(relation), evidence


class CutCellLeafletRoute(StrictModule, NonTrainableState):
    """Diffuse cut-cell leaflet route with open-area and small-cell evidence."""

    geometry_plan: MACDiffuseSDFGeometryPlan
    geometry_arguments: CutCellGeometryArguments = eqx.field(static=True)
    leakage_face_masks: tuple[Array, ...]
    maximum_leakage_proxy: float = eqx.field(static=True)
    maximum_gcl_residual: float = eqx.field(static=True)
    maximum_small_cell_fraction: float = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry_plan: MACDiffuseSDFGeometryPlan,
        geometry_arguments: CutCellGeometryArguments,
        leakage_face_masks: tuple[ArrayLike, ...],
        /,
        *,
        maximum_leakage_proxy: float,
        maximum_gcl_residual: float = 1.0e-6,
        maximum_small_cell_fraction: float = 0.05,
        route_id: str = "cardiovascular-cut-cell-leaflet",
    ):
        if not isinstance(geometry_plan, MACDiffuseSDFGeometryPlan):
            raise TypeError("geometry_plan must be MACDiffuseSDFGeometryPlan.")
        if not callable(geometry_arguments):
            raise TypeError("geometry_arguments must be callable.")
        masks = tuple(np.asarray(mask, dtype=bool) for mask in leakage_face_masks)
        face_shapes = tuple(
            tuple(value.shape) for value in geometry_plan.operators.face_dual_measures
        )
        if len(masks) != len(face_shapes) or any(
            mask.shape != shape for mask, shape in zip(masks, face_shapes, strict=True)
        ):
            raise ValueError("leakage_face_masks must match every MAC face layout.")
        if not any(np.any(mask) for mask in masks):
            raise ValueError("At least one leakage face must be selected.")
        leakage = float(maximum_leakage_proxy)
        gcl = float(maximum_gcl_residual)
        small = float(maximum_small_cell_fraction)
        identifier = str(route_id)
        if (
            not np.isfinite(leakage)
            or leakage < 0.0
            or not np.isfinite(gcl)
            or gcl < 0.0
            or not np.isfinite(small)
            or not 0.0 <= small <= 1.0
        ):
            raise ValueError("Cut-cell leaflet qualification thresholds are invalid.")
        if not identifier:
            raise ValueError("route_id must be non-empty.")
        self.geometry_plan = geometry_plan
        self.geometry_arguments = geometry_arguments
        self.leakage_face_masks = tuple(jnp.asarray(mask) for mask in masks)
        self.maximum_leakage_proxy = leakage
        self.maximum_gcl_residual = gcl
        self.maximum_small_cell_fraction = small
        self.route_id = identifier

    def initialize(
        self,
        configuration: Array,
        velocity: Array,
        time: ArrayLike,
        args: Any,
        /,
    ) -> CutCellLeafletFluidState:
        geometry_args = self.geometry_arguments(configuration, velocity, args)
        return CutCellLeafletFluidState(
            self.geometry_plan.evaluate(time, args=geometry_args)
        )

    def evaluate(
        self,
        configuration: Array,
        velocity: Array,
        time: ArrayLike,
        step_size: ArrayLike,
        previous: CutCellLeafletFluidState,
        args: Any,
        /,
    ) -> tuple[CutCellLeafletFluidState, LeafletFluidEvidence]:
        if not isinstance(previous, CutCellLeafletFluidState):
            raise TypeError("Cut-cell route requires CutCellLeafletFluidState.")
        geometry_args = self.geometry_arguments(configuration, velocity, args)
        geometry = self.geometry_plan.evaluate(
            time,
            args=geometry_args,
            previous=previous.geometry,
            step_size=step_size,
        )
        numerator = jnp.asarray(0.0, dtype=geometry.cell_fluid_fraction.dtype)
        denominator = jnp.asarray(0.0, dtype=geometry.cell_fluid_fraction.dtype)
        for mask, fraction, measure in zip(
            self.leakage_face_masks,
            geometry.face_open_fraction,
            self.geometry_plan.operators.discretization.face_measures,
            strict=True,
        ):
            selected_measure = jnp.where(mask, measure, 0.0)
            numerator = numerator + jnp.sum(selected_measure * fraction)
            denominator = denominator + jnp.sum(selected_measure)
        leakage = numerator / jnp.maximum(denominator, jnp.finfo(numerator.dtype).tiny)
        small_fraction = jnp.mean(geometry.small_cell_mask.astype(numerator.dtype))
        leakage_qualified = leakage <= self.maximum_leakage_proxy
        refinement = (small_fraction > self.maximum_small_cell_fraction) | (
            geometry.geometric_conservation_residual > self.maximum_gcl_residual
        )
        finite = (
            geometry.finite
            & jnp.isfinite(leakage)
            & jnp.isfinite(small_fraction)
            & jnp.isfinite(geometry.geometric_conservation_residual)
        )
        successful = geometry.successful & finite & leakage_qualified & ~refinement
        evidence = LeafletFluidEvidence(
            jnp.asarray(1.0, dtype=numerator.dtype),
            leakage,
            geometry.geometric_conservation_residual,
            small_fraction,
            leakage_qualified,
            refinement,
            finite,
            successful,
            False,
            self.route_id,
        )
        return CutCellLeafletFluidState(geometry), evidence


LeafletFluidRoute: TypeAlias = ImmersedLeafletRoute | CutCellLeafletRoute


class LeafletContactEvidence(StrictModule):
    gap: Array
    minimum_gap: Array
    maximum_penetration: Array
    active_contact_count: Array
    normal_pressure: Array
    force_balance_residual: Array
    normal_power: Array
    dissipation_rate: Array
    native_successful: Array
    finite: Array
    successful: Array
    contact_plan_id: str = eqx.field(static=True)


class LeafletStructuralAdvanceResult(StrictModule):
    candidate_configuration: Array
    candidate_velocity: Array
    successful: Array
    status: Array
    residual_norm: Array
    iterations: Array


LeafletStructuralAdvance = Callable[
    [Array, Array, Array, Array, Array, Any], LeafletStructuralAdvanceResult
]


class LeafletFSIState(StrictModule):
    """Accepted leaflet structural and fluid-geometry state."""

    configuration: Array
    velocity: Array
    fluid_state: LeafletFluidState


class LeafletTransitionEvidence(StrictModule):
    contact_before: LeafletContactEvidence
    contact_candidate: LeafletContactEvidence
    fluid: LeafletFluidEvidence
    structural_residual_norm: Array
    structural_iterations: Array
    finite: Array
    successful: Array


class LeafletContactTransition(StrictModule):
    candidate_state: LeafletFSIState
    accepted_state: LeafletFSIState
    evidence: LeafletTransitionEvidence
    contact_before: DeformableContactResidualEvaluation
    contact_candidate: DeformableContactResidualEvaluation
    structural: LeafletStructuralAdvanceResult
    successful: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class LeafletContactWorkflowPlan(StrictModule, NonTrainableState):
    """Plan true leaflet contact coupled to an immersed or cut-cell fluid route."""

    contact: DeformableContactResidualPlan
    fluid_route: LeafletFluidRoute
    structural_advance: LeafletStructuralAdvance = eqx.field(static=True)
    maximum_penetration: float = eqx.field(static=True)
    force_balance_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        contact: DeformableContactResidualPlan,
        fluid_route: LeafletFluidRoute,
        structural_advance: LeafletStructuralAdvance,
        /,
        *,
        maximum_penetration: float,
        force_balance_tolerance: float = 1.0e-9,
    ):
        if not isinstance(contact, DeformableContactResidualPlan):
            raise TypeError("contact must be DeformableContactResidualPlan.")
        if not isinstance(fluid_route, (ImmersedLeafletRoute, CutCellLeafletRoute)):
            raise TypeError(
                "fluid_route must be an explicit immersed or cut-cell leaflet route."
            )
        if not callable(structural_advance):
            raise TypeError("structural_advance must be callable.")
        penetration = float(maximum_penetration)
        tolerance = float(force_balance_tolerance)
        if (
            not np.isfinite(penetration)
            or penetration < 0.0
            or not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError(
                "Leaflet contact tolerances must be finite and non-negative."
            )
        self.contact = contact
        self.fluid_route = fluid_route
        self.structural_advance = structural_advance
        self.maximum_penetration = penetration
        self.force_balance_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-leaflet-contact-workflow",
                "contact": contact.plan_id,
                "fluid_route": fluid_route.route_id,
                "maximum_penetration": penetration,
                "force_balance_tolerance": tolerance,
                "exact_sealing_claim": False,
            }
        )

    def prepare(self, /) -> PreparedLeafletContactWorkflow:
        return PreparedLeafletContactWorkflow(self)


class PreparedLeafletContactWorkflow(StrictModule, NonTrainableState):
    """Prepared fixed-topology leaflet transaction with atomic rollback."""

    plan: LeafletContactWorkflowPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: LeafletContactWorkflowPlan, /):
        if not isinstance(plan, LeafletContactWorkflowPlan):
            raise TypeError("plan must be LeafletContactWorkflowPlan.")
        self.plan = plan
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-cardio-leaflet-contact", "plan": plan.plan_id}
        )

    def initialize(
        self,
        configuration: ArrayLike,
        velocity: ArrayLike,
        time: ArrayLike = 0.0,
        args: Any = None,
        /,
    ) -> LeafletFSIState:
        configuration_ = jnp.asarray(configuration)
        velocity_ = jnp.asarray(velocity, dtype=configuration_.dtype)
        if configuration_.shape != velocity_.shape:
            raise ValueError("Leaflet configuration and velocity shapes must match.")
        if not jnp.issubdtype(configuration_.dtype, jnp.inexact):
            raise TypeError("Leaflet state must use an inexact dtype.")
        fluid_state = self.plan.fluid_route.initialize(
            configuration_, velocity_, time, args
        )
        return LeafletFSIState(configuration_, velocity_, fluid_state)

    def _contact_evidence(
        self,
        result: DeformableContactResidualEvaluation,
        configuration: Array,
        /,
    ) -> LeafletContactEvidence:
        contact = result.contact
        gap = jnp.asarray(configuration)[..., -1].reshape((-1,))
        pressure = jnp.concatenate(
            tuple(
                jnp.where(kinematics.valid, response.normal.traction, 0.0)
                for kinematics, response in zip(
                    contact.kinematics.batches,
                    contact.closure.batches,
                    strict=True,
                )
            )
        )
        balance = contact.assembly.action_reaction_residual
        minimum_gap = jnp.min(gap)
        maximum_penetration = jnp.max(jnp.maximum(-gap, 0.0))
        force_balance_residual = jnp.sqrt(jnp.sum(balance**2))
        finite = (
            result.finite
            & jnp.all(jnp.isfinite(gap))
            & jnp.all(jnp.isfinite(pressure))
            & jnp.isfinite(force_balance_residual)
        )
        activation_distance = self.plan.contact.activation_distance
        native_successful = jnp.asarray(result.successful) & (
            True if activation_distance is None else minimum_gap < activation_distance
        )
        successful = (
            native_successful
            & finite
            & (maximum_penetration <= self.plan.maximum_penetration)
            & (force_balance_residual <= self.plan.force_balance_tolerance)
        )
        return LeafletContactEvidence(
            gap,
            minimum_gap,
            maximum_penetration,
            jnp.sum((pressure > 0.0).astype(jnp.int32)),
            pressure,
            force_balance_residual,
            result.normal_power,
            result.dissipation_rate,
            native_successful,
            finite,
            successful,
            self.plan.contact.plan_id,
        )

    def advance(
        self,
        state: LeafletFSIState,
        start_time: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> LeafletContactTransition:
        if not isinstance(state, LeafletFSIState):
            raise TypeError("state must be LeafletFSIState.")
        fluid_route = self.plan.fluid_route
        if isinstance(fluid_route, ImmersedLeafletRoute):
            if not isinstance(state.fluid_state, ImmersedLeafletFluidState):
                raise TypeError("Leaflet state does not match its immersed route.")
        elif not isinstance(state.fluid_state, CutCellLeafletFluidState):
            raise TypeError("Leaflet state does not match its cut-cell route.")
        start = jnp.asarray(start_time, dtype=state.configuration.dtype).reshape(())
        step = jnp.asarray(step_size, dtype=state.configuration.dtype).reshape(())
        contact_before = self.plan.contact.evaluate(
            state.configuration, state.velocity, args
        )
        before_evidence = self._contact_evidence(
            contact_before,
            state.configuration,
        )
        structural = self.plan.structural_advance(
            start,
            step,
            state.configuration,
            state.velocity,
            contact_before.residual,
            args,
        )
        if not isinstance(structural, LeafletStructuralAdvanceResult):
            raise TypeError(
                "structural_advance must return LeafletStructuralAdvanceResult."
            )
        candidate_configuration = jnp.asarray(
            structural.candidate_configuration, dtype=state.configuration.dtype
        )
        candidate_velocity = jnp.asarray(
            structural.candidate_velocity, dtype=state.velocity.dtype
        )
        if (
            candidate_configuration.shape != state.configuration.shape
            or candidate_velocity.shape != state.velocity.shape
        ):
            raise ValueError(
                "Leaflet structural advance changed the prepared state shape."
            )
        contact_candidate = self.plan.contact.evaluate(
            candidate_configuration, candidate_velocity, args
        )
        candidate_contact_evidence = self._contact_evidence(
            contact_candidate,
            candidate_configuration,
        )
        if isinstance(fluid_route, ImmersedLeafletRoute):
            assert isinstance(state.fluid_state, ImmersedLeafletFluidState)
            candidate_fluid_state, fluid_evidence = fluid_route.evaluate(
                candidate_configuration,
                candidate_velocity,
                start + step,
                step,
                state.fluid_state,
                args,
            )
        else:
            assert isinstance(state.fluid_state, CutCellLeafletFluidState)
            candidate_fluid_state, fluid_evidence = fluid_route.evaluate(
                candidate_configuration,
                candidate_velocity,
                start + step,
                step,
                state.fluid_state,
                args,
            )
        finite_state = (
            jnp.isfinite(start)
            & jnp.isfinite(step)
            & (step > 0.0)
            & jnp.all(jnp.isfinite(candidate_configuration))
            & jnp.all(jnp.isfinite(candidate_velocity))
            & before_evidence.finite
        )
        successful = (
            jnp.asarray(structural.successful)
            & before_evidence.successful
            & candidate_contact_evidence.successful
            & fluid_evidence.successful
            & finite_state
        )
        candidate_state = LeafletFSIState(
            candidate_configuration, candidate_velocity, candidate_fluid_state
        )
        accepted_fluid_state = jax.lax.cond(
            successful,
            lambda _: candidate_fluid_state,
            lambda _: state.fluid_state,
            operand=None,
        )
        accepted_state = LeafletFSIState(
            jnp.where(successful, candidate_configuration, state.configuration),
            jnp.where(successful, candidate_velocity, state.velocity),
            accepted_fluid_state,
        )
        status = jnp.where(
            successful,
            int(LeafletTransitionStatus.SUCCESS),
            jnp.where(
                ~finite_state
                | ~candidate_contact_evidence.finite
                | ~fluid_evidence.finite,
                int(LeafletTransitionStatus.NONFINITE),
                jnp.where(
                    ~jnp.asarray(structural.successful),
                    int(LeafletTransitionStatus.STRUCTURAL_FAILURE),
                    jnp.where(
                        ~before_evidence.successful
                        | ~candidate_contact_evidence.successful,
                        int(LeafletTransitionStatus.CONTACT_FAILURE),
                        jnp.where(
                            fluid_evidence.refinement_required,
                            int(LeafletTransitionStatus.REFINEMENT_REQUIRED),
                            int(LeafletTransitionStatus.LEAKAGE_FAILURE),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        evidence = LeafletTransitionEvidence(
            before_evidence,
            candidate_contact_evidence,
            fluid_evidence,
            jnp.asarray(structural.residual_norm),
            jnp.asarray(structural.iterations, dtype=jnp.int32),
            finite_state & candidate_contact_evidence.finite & fluid_evidence.finite,
            successful,
        )
        return LeafletContactTransition(
            candidate_state,
            accepted_state,
            evidence,
            contact_before,
            contact_candidate,
            structural,
            successful,
            status,
            self.prepared_id,
        )


__all__ = [
    "CutCellGeometryArguments",
    "CutCellLeafletFluidState",
    "CutCellLeafletRoute",
    "ImmersedLeafletFluidState",
    "ImmersedLeafletRoute",
    "ImmersedLeakageProbe",
    "LeafletContactEvidence",
    "LeafletContactTransition",
    "LeafletContactWorkflowPlan",
    "LeafletFSIState",
    "LeafletFluidEvidence",
    "LeafletFluidRoute",
    "LeafletFluidState",
    "LeafletKinematics",
    "LeafletStructuralAdvance",
    "LeafletStructuralAdvanceResult",
    "LeafletTransitionEvidence",
    "LeafletTransitionStatus",
    "PreparedLeafletContactWorkflow",
]
