#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prescribed-trajectory Shockley--Ramo detector response."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import apply_gather_stencil, rectilinear_stencil
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import StateLayout, TrajectoryData
from ...units import (
    conversion_factor,
    COULOMB,
    METER,
    SECOND,
    UnitDefinition,
)
from ._detector import DetectorWeightingFieldPlan, DetectorWeightingFieldResult
from ._quantities import _si


class DetectorTrajectoryRoute(StrictModule, NonTrainableState):
    """Explicit position and time route from generic ``TrajectoryData``.

    Position components are flattened ``StateLayout`` component indices.  The
    declared units convert those components and the trajectory coordinate to SI;
    no state name, velocity, carrier species, or dynamics is inferred.
    """

    state_layout: StateLayout
    position_components: tuple[int, ...] = eqx.field(static=True)
    position_scale_to_meter: float = eqx.field(static=True)
    time_scale_to_second: float = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_layout: StateLayout,
        position_components: Sequence[int],
        /,
        *,
        position_unit: UnitDefinition = METER,
        time_unit: UnitDefinition = SECOND,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be StateLayout.")
        indices = tuple(int(index) for index in position_components)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or min(indices) < 0
            or max(indices) >= state_layout.size
        ):
            raise ValueError(
                "position_components must be distinct valid flattened state indices."
            )
        position_scale = float(conversion_factor(position_unit, METER))
        time_scale = float(conversion_factor(time_unit, SECOND))
        self.state_layout = state_layout
        self.position_components = indices
        self.position_scale_to_meter = position_scale
        self.time_scale_to_second = time_scale
        self.route_id = canonical_fingerprint(
            {
                "kind": "semiconductor-detector-trajectory-route",
                "state_layout": state_layout.layout_id,
                "position_components": list(indices),
                "position_unit": position_unit.unit_id,
                "time_unit": time_unit.unit_id,
            }
        )


class ShockleyRamoResourceEvidence(StrictModule, NonTrainableState):
    """Exact fixed-shape work admitted before interpolation allocation."""

    case_count: int = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    interpolation_route_count: int = eqx.field(static=True)
    electrode_count: int = eqx.field(static=True)


class ShockleyRamoResponseResult(StrictModule):
    """Induced electrode charge and interval-average current.

    For carrier charge ``q`` and weighting potential ``phi_w``, induced charge
    on the external electrode is ``Q = -q * phi_w``.  Current is positive into
    that electrode and equals ``dQ/dt``.  Each interval current is the exact
    secant ``(Q[n+1] - Q[n]) / (t[n+1] - t[n])``; its discrete time integral
    therefore closes the retained endpoint charges without a velocity estimate.
    """

    induced_charge: Array
    interval_current: Array
    integrated_current: Array
    endpoint_charge_change: Array
    current_integral_closure_defect: Array
    interpolation_support: Array
    sample_valid: Array
    transition_valid: Array
    route_valid: Array
    finite: Array
    closure_valid: Array
    successful: Array
    resources: ShockleyRamoResourceEvidence
    trajectory_dataset_id: str = eqx.field(static=True)
    weighting_plan_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    sign_convention: str = eqx.field(static=True)


class PrescribedShockleyRamoPlan(StrictModule, NonTrainableState):
    """Evaluate response only; this plan never advances or modifies carriers."""

    weighting_plan: DetectorWeightingFieldPlan
    weighting: DetectorWeightingFieldResult
    route: DetectorTrajectoryRoute
    closure_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        weighting_plan: DetectorWeightingFieldPlan,
        weighting: DetectorWeightingFieldResult,
        route: DetectorTrajectoryRoute,
        /,
        *,
        closure_tolerance: float = 1.0e-12,
    ):
        if not isinstance(weighting_plan, DetectorWeightingFieldPlan):
            raise TypeError("weighting_plan must be DetectorWeightingFieldPlan.")
        if not isinstance(weighting, DetectorWeightingFieldResult):
            raise TypeError(
                "weighting must be DetectorWeightingFieldResult; bias results cannot substitute."
            )
        if weighting.plan_id != weighting_plan.plan_id:
            raise ValueError("Weighting result belongs to another weighting plan.")
        if not bool(weighting.evidence.certified):
            raise ValueError(
                "Shockley--Ramo response requires a certified complete-electrode weighting basis."
            )
        if not isinstance(route, DetectorTrajectoryRoute):
            raise TypeError("route must be DetectorTrajectoryRoute.")
        if len(route.position_components) != weighting_plan.detector.bridge.dimension:
            raise ValueError(
                "Trajectory position dimension must match the detector cochain dimension."
            )
        tolerance = float(closure_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("closure_tolerance must be finite and positive.")
        self.weighting_plan = weighting_plan
        self.weighting = weighting
        self.route = route
        self.closure_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prescribed-shockley-ramo-plan",
                "weighting": weighting.plan_id,
                "route": route.route_id,
                "closure_tolerance": tolerance,
                "dynamics": "prescribed-trajectory-no-carrier-advance",
                "sign": "Q_induced=-q*phi_w;i_into_electrode=dQ_induced/dt",
            }
        )

    def evaluate(
        self,
        trajectory: TrajectoryData,
        carrier_charge: ArrayLike,
        /,
        *,
        charge_unit: UnitDefinition = COULOMB,
    ) -> ShockleyRamoResponseResult:
        if not isinstance(trajectory, TrajectoryData):
            raise TypeError("trajectory must be TrajectoryData.")
        if trajectory.state_layout.layout_id != self.route.state_layout.layout_id:
            raise ValueError("Trajectory state layout does not match its detector route.")
        if trajectory.coordinate_kind != "continuous":
            raise ValueError(
                "Shockley--Ramo response requires a continuous time coordinate."
            )
        detector = self.weighting_plan.detector
        case_count = trajectory.num_cases
        sample_count = case_count * trajectory.capacity
        interpolation_routes = sample_count * (1 << detector.bridge.dimension)
        detector.resources.admit_trajectory(
            case_count, sample_count, interpolation_routes
        )
        resources = ShockleyRamoResourceEvidence(
            case_count,
            sample_count,
            interpolation_routes,
            detector.electrode_count,
        )
        event_rank = len(trajectory.state_layout.shape)
        leading = (
            trajectory.states.shape[:-event_rank]
            if event_rank
            else trajectory.states.shape
        )
        flat_states = trajectory.states.reshape(leading + (trajectory.state_layout.size,))
        positions = jnp.take(
            flat_states,
            jnp.asarray(self.route.position_components, dtype=jnp.int32),
            axis=-1,
        )
        positions = positions * jnp.asarray(
            self.route.position_scale_to_meter, dtype=positions.dtype
        )
        sample_valid = trajectory.sample_valid
        lower = jnp.asarray(
            tuple(
                axis.point_coordinates[0] for axis in detector.bridge.grid.structured_axes
            ),
            dtype=positions.dtype,
        )
        safe_positions = jnp.where(sample_valid[..., None], positions, lower)
        stencil = rectilinear_stencil(
            tuple(
                axis.point_coordinates for axis in detector.bridge.grid.structured_axes
            ),
            safe_positions,
            boundary=("constant",) * detector.bridge.dimension,
        )
        interpolated = apply_gather_stencil(self.weighting.potentials.T, stencil)
        interpolation_support = interpolated.support & sample_valid
        weighting_potential = interpolated.values

        charges = _si(carrier_charge, charge_unit, COULOMB)
        if charges.shape == ():
            charges = jnp.broadcast_to(charges, trajectory.case_shape)
        if charges.shape != trajectory.case_shape:
            raise ValueError(
                "carrier_charge must be scalar or contain one fixed charge per trajectory case."
            )
        charges = eqx.error_if(
            charges,
            jnp.any(~jnp.isfinite(charges)),
            "carrier_charge must be finite.",
        )
        induced = -charges[..., None, None] * weighting_potential
        induced = jnp.where(sample_valid[..., None], induced, 0.0)
        times = trajectory.coordinates * jnp.asarray(
            self.route.time_scale_to_second, dtype=trajectory.coordinates.dtype
        )
        dt = times[..., 1:] - times[..., :-1]

        lengths = jnp.sum(sample_valid, axis=-1, dtype=jnp.int32)
        sample_index = jnp.arange(trajectory.capacity, dtype=jnp.int32)
        transition_index = jnp.arange(trajectory.capacity - 1, dtype=jnp.int32)
        expected_samples = sample_index < lengths[..., None]
        expected_transitions = transition_index < jnp.maximum(lengths - 1, 0)[..., None]
        transition_valid = trajectory.transition_valid
        prefix_valid = jnp.all(sample_valid == expected_samples, axis=-1)
        transitions_complete = jnp.all(transition_valid == expected_transitions, axis=-1)
        positions_supported = jnp.all(~sample_valid | interpolation_support, axis=-1)
        route_valid = (
            (lengths >= 2) & prefix_valid & transitions_complete & positions_supported
        )

        safe_dt = jnp.where(transition_valid, dt, 1.0)
        interval_current = (induced[..., 1:, :] - induced[..., :-1, :]) / safe_dt[
            ..., None
        ]
        interval_current = jnp.where(transition_valid[..., None], interval_current, 0.0)
        integrated = jnp.sum(interval_current * dt[..., None], axis=-2)
        last = jnp.maximum(lengths - 1, 0)
        endpoint = (
            jnp.take_along_axis(
                induced,
                last[..., None, None],
                axis=-2,
            )[..., 0, :]
            - induced[..., 0, :]
        )
        closure = jnp.abs(integrated - endpoint)
        # A nearly unchanged electrode may have an endpoint delta near zero even
        # though both endpoint charges are O(q). Normalize roundoff by the fixed
        # carrier charge, not by that cancellation-sized delta.
        charge_scale = jnp.abs(charges)[..., None]
        scale = jnp.maximum(charge_scale, jnp.finfo(induced.dtype).tiny)
        closure_valid = (
            jnp.all(closure <= self.closure_tolerance * scale, axis=-1) & route_valid
        )
        finite = (
            jnp.all(jnp.isfinite(induced), axis=(-2, -1))
            & jnp.all(jnp.isfinite(interval_current), axis=(-2, -1))
            & jnp.all(jnp.isfinite(closure), axis=-1)
        )
        successful = jnp.all(route_valid & closure_valid & finite)
        return ShockleyRamoResponseResult(
            induced,
            interval_current,
            integrated,
            endpoint,
            closure,
            interpolation_support,
            sample_valid,
            transition_valid,
            route_valid,
            finite,
            closure_valid,
            successful,
            resources,
            trajectory.dataset_id,
            self.weighting.plan_id,
            self.route.route_id,
            "Q_induced=-q*phi_w; i_into_electrode=dQ_induced/dt",
        )


__all__ = [
    "DetectorTrajectoryRoute",
    "PrescribedShockleyRamoPlan",
    "ShockleyRamoResourceEvidence",
    "ShockleyRamoResponseResult",
]
