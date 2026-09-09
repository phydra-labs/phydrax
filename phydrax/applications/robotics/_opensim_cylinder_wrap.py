#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Single-cylinder lateral geodesics from pinned OpenSim tangency equations.

This independent realization solves the common axial slope exactly on the
unrolled cylinder, rather than reproducing OpenSim's display-segment iteration.
It is not a finite-solid cap solver or a multi-obstacle route authority. The
existing planar-cylinder source family is deliberately unchanged.
"""

from __future__ import annotations

from enum import IntFlag
from math import isfinite
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._identity import NumericRevision
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import ArraySpace, FunctionLinearOperator
from ._analytic_wrap import _dot, _norm, _positive, _unit, _vector3


_SOURCE_REVISION = "86b30588374650fbaf012a345a836a64f6855522"
_SOURCE_SHA256 = "ce01766de755cd78ae2b21a271809d5c988082e87a50e9ebf5302c9282662580"
_SOURCE_URL = (
    "https://raw.githubusercontent.com/opensim-org/opensim-core/"
    + _SOURCE_REVISION
    + "/OpenSim/Simulation/Wrap/WrapCylinder.cpp"
)
CylinderWrapSide = Literal["shortest", "positive", "negative"]


class OpenSimCylinderWrapStatus(IntFlag):
    SUCCESS = 0
    ENDPOINT_INSIDE_RADIUS = 1
    DEGENERATE = 2
    CONTACT_EVENT = 4
    TOPOLOGY_TIE = 8
    CAP_OR_RIM_UNSUPPORTED = 16
    NONFINITE = 32
    STALE_STATE = 64
    BRANCH_CHANGED = 128
    RESIDUAL_FAILURE = 256


class OpenSimCylinderWrapEvidence(StrictModule, NonTrainableState):
    status: Array
    applied: Array
    mode_changed: Array
    fixed_branch_gradient_supported: Array
    candidate_lengths_m: Array
    candidate_feasible: Array
    selected_branch: Array
    shortest_lateral_branch: Array
    shortest_lateral_gap_m: Array
    selected_excess_length_m: Array
    contact_margin_m: Array
    rim_margin_m: Array
    tangent_direction_residual: Array
    surface_residual_m: Array
    source_revision: str = eqx.field(static=True, default=_SOURCE_REVISION)
    source_sha256: str = eqx.field(static=True, default=_SOURCE_SHA256)

    @property
    def successful(self) -> Array:
        return self.status == 0


class OpenSimCylinderWrapEvaluation(StrictModule, NonTrainableState):
    tangent_points_m: Array
    surface_points_m: Array
    surface_mask: Array
    total_length_m: Array
    surface_length_m: Array
    signed_surface_angle_rad: Array
    evidence: OpenSimCylinderWrapEvidence
    prepared_id: str = eqx.field(static=True)


class OpenSimCylinderWrapState(StrictModule, NonTrainableState):
    """Accepted discrete branch: -1 uninitialized, 0 direct, 1 +axis, 2 -axis."""

    branch: Array
    endpoints_m: Array
    geometry: Array
    length_m: Array
    initialized: Array
    accepted_steps: Array
    prepared_id: str = eqx.field(static=True)


class OpenSimCylinderWrapCandidate(StrictModule, NonTrainableState):
    source_state: OpenSimCylinderWrapState
    proposed_state: OpenSimCylinderWrapState
    evaluation: OpenSimCylinderWrapEvaluation
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evaluation.evidence.successful


class OpenSimCylinderWrapPullbackEvidence(StrictModule, NonTrainableState):
    tensile_force_N: Array
    length_rate_m_per_s: Array
    route_power_W: Array
    endpoint_power_W: Array
    power_residual_W: Array
    successful: Array
    force_owner: str = eqx.field(static=True, default="native-tension")


class OpenSimCylinderRouteWrapPlan(StrictModule, NonTrainableState):
    """Two zero-extra-winding lateral branches with explicit outer selection.

    ``side='shortest'`` selects the shorter of the two complete lateral paths;
    ``positive``/``negative`` prescribe travel around the oriented axis. All
    policies use a direct route when the chord is clear. Ties fail closed, and
    there is no invented hysteresis. Length bounds certify lateral support only:
    shortest-path evidence does not compare unimplemented cap/rim paths.
    """

    sample_count: int = eqx.field(static=True)
    side: CylinderWrapSide = eqx.field(static=True)
    event_tolerance_m: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_count: int = 32,
        /,
        *,
        side: CylinderWrapSide = "shortest",
        event_tolerance_m: float = 1.0e-8,
        residual_tolerance: float = 1.0e-5,
    ):
        if isinstance(sample_count, bool) or int(sample_count) != sample_count:
            raise ValueError("sample_count must be an integer.")
        if sample_count < 2:
            raise ValueError("sample_count must be at least two.")
        if side not in ("shortest", "positive", "negative"):
            raise ValueError("side must be shortest, positive, or negative.")
        for value in (event_tolerance_m, residual_tolerance):
            if not isfinite(value) or value <= 0.0:
                raise ValueError("Tolerances must be positive and finite.")
        self.sample_count = int(sample_count)
        self.side = side
        self.event_tolerance_m = float(event_tolerance_m)
        self.residual_tolerance = float(residual_tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "opensim-cylinder-lateral-route-plan",
                "source_revision": _SOURCE_REVISION,
                "source_sha256": _SOURCE_SHA256,
                "source_url": _SOURCE_URL,
                "source_license": "Apache-2.0",
                "numerical_realization": "exact-unrolled-common-axial-slope",
                "candidate_order": ["direct", "positive", "negative"],
                "extra_windings": 0,
                "selection": side,
                "hysteresis": "none-ties-fail-closed",
                "length_support": "lateral-only-no-cap-or-rim",
                "frame": "world-origin-oriented-axis",
                "length_unit": "m",
                "sample_count": self.sample_count,
                "event_tolerance_m": self.event_tolerance_m.hex(),
                "residual_tolerance": self.residual_tolerance.hex(),
            }
        )

    def prepare(
        self,
        origin_m: ArrayLike,
        axis: ArrayLike,
        radius_m: ArrayLike,
        length_m: ArrayLike,
        /,
    ) -> PreparedOpenSimCylinderRouteWrap:
        origin = _vector3(origin_m, "origin_m")
        direction, magnitude = _unit(_vector3(axis, "axis"), 0.0)
        if not np.all(np.isfinite(np.asarray(origin))):
            raise ValueError("origin_m must be finite.")
        if not np.isfinite(float(magnitude)) or float(magnitude) <= 0.0:
            raise ValueError("axis must be finite and nonzero.")
        radius = _positive(radius_m, "radius_m")
        length = _positive(length_m, "length_m")
        prepared_id = NumericRevision(
            self.plan_id,
            {
                "origin_m": origin,
                "axis": direction,
                "radius_m": radius,
                "length_m": length,
            },
        ).revision_id
        return PreparedOpenSimCylinderRouteWrap(
            self, origin, direction, radius, length, prepared_id
        )


def _endpoints(value: ArrayLike) -> Array:
    points = jnp.asarray(value)
    if points.shape != (2, 3):
        raise ValueError("endpoints_m must have shape (2, 3).")
    if not jnp.issubdtype(points.dtype, jnp.inexact):
        points = points.astype(float)
    return points


class PreparedOpenSimCylinderRouteWrap(StrictModule):
    """Pure fixed-shape geometry; owns no independent mechanical force."""

    plan: OpenSimCylinderRouteWrapPlan
    origin_m: Array
    axis: Array
    radius_m: Array
    length_m: Array
    prepared_id: str = eqx.field(static=True)

    def _geometry(self) -> Array:
        return jnp.concatenate(
            (self.origin_m, self.axis, jnp.stack((self.radius_m, self.length_m)))
        )

    def initial_state(self) -> OpenSimCylinderWrapState:
        dtype = self.origin_m.dtype
        return OpenSimCylinderWrapState(
            jnp.asarray(-1, jnp.int32),
            jnp.zeros((2, 3), dtype=dtype),
            self._geometry(),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(False),
            jnp.asarray(0, jnp.int32),
            self.prepared_id,
        )

    def _branches(self, points: Array) -> tuple[Array, ...]:
        """Analytic source stationary solution; no display-sample quadrature."""
        tolerance = self.plan.event_tolerance_m
        offset = points - self.origin_m
        axial = contract("ni,i->n", offset, self.axis)
        radial = offset - axial[:, None] * self.axis
        square = jnp.sum(radial * radial, axis=-1)
        distance = jnp.sqrt(jnp.maximum(square, tolerance**2))
        unit = radial / distance[:, None]
        cosine = self.radius_m / jnp.maximum(distance, self.radius_m)
        # Keep inactive/invalid branch derivatives finite. Admitted endpoints
        # lie strictly beyond this guard, so it does not change valid geometry.
        free = jnp.sqrt(jnp.maximum(square - self.radius_m**2, tolerance**2))
        sine = free / distance
        signs = jnp.asarray((1.0, -1.0), dtype=points.dtype)
        cross = jnp.cross(self.axis, unit)
        first = self.radius_m * (
            cosine[0] * unit[0] + signs[:, None] * sine[0] * cross[0]
        )
        second = self.radius_m * (
            cosine[1] * unit[1] - signs[:, None] * sine[1] * cross[1]
        )
        principal = jnp.arctan2(
            contract("ni,i->n", jnp.cross(first, second), self.axis),
            jnp.sum(first * second, axis=-1),
        )
        angle = signs * jnp.mod(signs * principal, 2.0 * jnp.pi)
        arc = self.radius_m * jnp.abs(angle)
        unrolled = free[0] + arc + free[1]
        slope = (axial[1] - axial[0]) / unrolled
        z_first = axial[0] + slope * free[0]
        z_second = axial[1] - slope * free[1]
        tangents = self.origin_m + jnp.stack(
            (
                first + z_first[:, None] * self.axis,
                second + z_second[:, None] * self.axis,
            ),
            axis=1,
        )
        stretch = jnp.sqrt(1.0 + slope**2)
        lengths = unrolled * stretch
        surface_lengths = arc * stretch
        rim = 0.5 * self.length_m - jnp.maximum(jnp.abs(z_first), jnp.abs(z_second))
        chord = radial[1] - radial[0]
        parameter = jnp.clip(
            -_dot(radial[0], chord) / jnp.maximum(_dot(chord, chord), tolerance**2),
            0.0,
            1.0,
        )
        closest = radial[0] + parameter * chord
        clearance = (
            jnp.sqrt(jnp.maximum(_dot(closest, closest), tolerance**2)) - self.radius_m
        )
        return tangents, lengths, surface_lengths, angle, rim, clearance, distance

    def _evaluate(
        self, points: Array, state: OpenSimCylinderWrapState, *, fixed: bool
    ) -> OpenSimCylinderWrapEvaluation:
        tangents, lengths, wall_lengths, angles, rims, clearance, distance = (
            self._branches(points)
        )
        tolerance = self.plan.event_tolerance_m
        axial = contract("ni,i->n", points - self.origin_m, self.axis)
        # This is a certified free chord above/below an end, not a cap path.
        end_clearance = jnp.maximum(jnp.min(axial), -jnp.max(axial)) - 0.5 * self.length_m
        direct_margin = jnp.maximum(clearance, end_clearance)
        direct = direct_margin > tolerance
        shortest = jnp.argmin(lengths).astype(jnp.int32) + 1
        chosen = shortest
        if self.plan.side != "shortest":
            chosen = jnp.asarray(1 if self.plan.side == "positive" else 2, jnp.int32)
        selected = jnp.where(direct, 0, chosen)
        branch = state.branch if fixed else selected
        index = jnp.clip(branch - 1, 0, 1)
        applied = branch > 0
        tangent = tangents[index]
        signed_angle = angles[index]
        fraction = jnp.linspace(0.0, 1.0, self.plan.sample_count, dtype=points.dtype)
        first_offset = tangent[0] - self.origin_m
        first_axial = _dot(first_offset, self.axis)
        second_axial = _dot(tangent[1] - self.origin_m, self.axis)
        first_radial = first_offset - first_axial * self.axis
        phase = fraction * signed_angle
        surface = (
            self.origin_m
            + jnp.cos(phase)[:, None] * first_radial
            + jnp.sin(phase)[:, None] * jnp.cross(self.axis, first_radial)
            + (first_axial + fraction * (second_axial - first_axial))[:, None] * self.axis
        )
        radial_tangent = (
            tangent
            - self.origin_m
            - contract("ni,i->n", tangent - self.origin_m, self.axis)[:, None] * self.axis
        )
        helix_derivative = (
            signed_angle * jnp.cross(self.axis, radial_tangent)
            + (second_axial - first_axial) * self.axis
        )
        line = jnp.stack((tangent[0] - points[0], points[1] - tangent[1]))
        line_unit = (
            line / jnp.maximum(jnp.sqrt(jnp.sum(line**2, axis=-1)), tolerance)[:, None]
        )
        helix_unit = helix_derivative / jnp.maximum(wall_lengths[index], tolerance)
        tangent_residual = jnp.max(jnp.abs(line_unit - helix_unit))
        surface_radial = (
            surface
            - self.origin_m
            - contract("ni,i->n", surface - self.origin_m, self.axis)[:, None] * self.axis
        )
        surface_residual = jnp.max(
            jnp.abs(jnp.sqrt(jnp.sum(surface_radial**2, axis=-1)) - self.radius_m)
        )
        gap = jnp.abs(lengths[1] - lengths[0])
        changed = state.initialized & (selected != state.branch)
        current = (
            (state.prepared_id == self.prepared_id)
            & jnp.all(state.geometry == self._geometry())
            & (state.accepted_steps >= 0)
            & jnp.where(
                state.initialized,
                (state.branch >= 0) & (state.branch <= 2),
                state.branch == -1,
            )
        )
        finite = (
            jnp.all(jnp.isfinite(points))
            & jnp.all(jnp.isfinite(self._geometry()))
            & jnp.all(jnp.isfinite(lengths))
            & jnp.all(jnp.isfinite(surface))
        )
        inside = jnp.any(distance - self.radius_m <= tolerance)
        contact_event = jnp.abs(direct_margin) <= tolerance
        tie = (~direct) & (gap <= tolerance) & (self.plan.side == "shortest")
        rim_failure = (~direct) & (rims[index] <= tolerance)
        degenerate = (
            (_norm(points[1] - points[0]) <= tolerance)
            | (self.radius_m <= tolerance)
            | (self.length_m <= tolerance)
            | (jnp.abs(_dot(self.axis, self.axis) - 1.0) > self.plan.residual_tolerance)
        )
        residual_failure = applied & (
            (tangent_residual > self.plan.residual_tolerance)
            | (surface_residual > self.plan.residual_tolerance * self.radius_m)
        )
        status = jnp.asarray(0, jnp.int32)
        for failed, flag in (
            (inside, OpenSimCylinderWrapStatus.ENDPOINT_INSIDE_RADIUS),
            (degenerate, OpenSimCylinderWrapStatus.DEGENERATE),
            (contact_event, OpenSimCylinderWrapStatus.CONTACT_EVENT),
            (tie, OpenSimCylinderWrapStatus.TOPOLOGY_TIE),
            (rim_failure, OpenSimCylinderWrapStatus.CAP_OR_RIM_UNSUPPORTED),
            (~finite, OpenSimCylinderWrapStatus.NONFINITE),
            (~current, OpenSimCylinderWrapStatus.STALE_STATE),
            (residual_failure, OpenSimCylinderWrapStatus.RESIDUAL_FAILURE),
            (
                fixed & (~state.initialized | (selected != branch)),
                OpenSimCylinderWrapStatus.BRANCH_CHANGED,
            ),
        ):
            status |= jnp.where(failed, int(flag), 0).astype(jnp.int32)
        successful = status == 0
        direct_length = _norm(points[1] - points[0])
        feasible_lateral = (~direct) & (~inside) & finite & (rims > tolerance)
        evidence = OpenSimCylinderWrapEvidence(
            status,
            applied & successful,
            changed,
            successful & state.initialized & ~changed,
            jnp.concatenate((direct_length[None], lengths)),
            jnp.concatenate(((direct & ~inside & finite)[None], feasible_lateral)),
            selected,
            shortest,
            gap,
            jnp.where(applied, lengths[index] - jnp.min(lengths), 0.0),
            jnp.abs(direct_margin),
            jnp.where(applied, rims[index], end_clearance),
            jnp.where(applied, tangent_residual, 0.0),
            jnp.where(applied, surface_residual, 0.0),
        )
        return OpenSimCylinderWrapEvaluation(
            jnp.where(applied, tangent, points),
            jnp.where(applied, surface, jnp.zeros_like(surface)),
            jnp.full((self.plan.sample_count,), applied & successful),
            jnp.where(applied, lengths[index], direct_length),
            jnp.where(applied, wall_lengths[index], 0.0),
            jnp.where(applied, signed_angle, 0.0),
            evidence,
            self.prepared_id,
        )

    def propose(
        self, state: OpenSimCylinderWrapState, endpoints_m: ArrayLike, /
    ) -> OpenSimCylinderWrapCandidate:
        points = _endpoints(endpoints_m)
        evaluation = self._evaluate(points, state, fixed=False)
        proposed = OpenSimCylinderWrapState(
            evaluation.evidence.selected_branch,
            points,
            self._geometry(),
            evaluation.total_length_m,
            jnp.asarray(True),
            state.accepted_steps + 1,
            self.prepared_id,
        )
        return OpenSimCylinderWrapCandidate(state, proposed, evaluation, self.prepared_id)

    def commit(
        self, candidate: OpenSimCylinderWrapCandidate, state: OpenSimCylinderWrapState, /
    ) -> OpenSimCylinderWrapState:
        """Commit every state leaf together; failed/stale/foreign candidates roll back."""
        matching = (
            (candidate.prepared_id == self.prepared_id)
            & (state.prepared_id == self.prepared_id)
            & eqx.tree_equal(candidate.source_state, state)
            & jnp.all(state.geometry == self._geometry())
        )
        accepted = matching & candidate.successful
        # Retain the caller's static provenance even for a foreign candidate.
        return eqx.tree_at(
            lambda item: (
                item.branch,
                item.endpoints_m,
                item.geometry,
                item.length_m,
                item.initialized,
                item.accepted_steps,
            ),
            state,
            tuple(
                jnp.where(accepted, proposed, previous)
                for proposed, previous in zip(
                    (
                        candidate.proposed_state.branch,
                        candidate.proposed_state.endpoints_m,
                        candidate.proposed_state.geometry,
                        candidate.proposed_state.length_m,
                        candidate.proposed_state.initialized,
                        candidate.proposed_state.accepted_steps,
                    ),
                    (
                        state.branch,
                        state.endpoints_m,
                        state.geometry,
                        state.length_m,
                        state.initialized,
                        state.accepted_steps,
                    ),
                )
            ),
        )

    def evaluate_fixed_branch(
        self, state: OpenSimCylinderWrapState, endpoints_m: ArrayLike, /
    ) -> OpenSimCylinderWrapEvaluation:
        """Evaluate accepted topology, never differentiate candidate selection."""
        return self._evaluate(_endpoints(endpoints_m), state, fixed=True)

    def _fixed_length(self, state: OpenSimCylinderWrapState, points: Array) -> Array:
        _, lengths, _, _, _, _, _ = self._branches(points)
        index = jnp.clip(state.branch - 1, 0, 1)
        return jnp.where(state.branch == 0, _norm(points[1] - points[0]), lengths[index])[
            None
        ]

    def length_jacobian_operator(
        self, state: OpenSimCylinderWrapState, endpoints_m: ArrayLike, /
    ) -> FunctionLinearOperator:
        """Endpoint velocity -> length rate, only on an evidenced fixed branch."""
        points = _endpoints(endpoints_m)
        supported = self.evaluate_fixed_branch(
            state, points
        ).evidence.fixed_branch_gradient_supported

        def length(value):
            return self._fixed_length(state, value)

        def action(velocity):
            rate = jax.jvp(length, (points,), (velocity,))[1]
            return jnp.where(supported, rate, jnp.zeros_like(rate))

        def transpose_action(cotangent):
            load = jax.vjp(length, points)[1](cotangent)[0]
            return jnp.where(supported, load, jnp.zeros_like(load))

        return FunctionLinearOperator(
            action,
            source=ArraySpace((2, 3), dtype=points.dtype),
            target=ArraySpace((1,), dtype=points.dtype),
            transpose_action=transpose_action,
            operator_id=f"{self.prepared_id}:fixed-branch-length-jacobian",
        )

    def tensile_force_pullback(
        self,
        state: OpenSimCylinderWrapState,
        endpoints_m: ArrayLike,
        endpoint_velocity_m_per_s: ArrayLike,
        tensile_force_N: ArrayLike,
        /,
        *,
        force_owner: Literal["native-tension"],
    ) -> tuple[Array, OpenSimCylinderWrapPullbackEvidence]:
        """Apply the sole native tension as endpoint loads -J_L^T T.

        These endpoint loads may be pulled through body kinematics once. A
        provider-native raw force must instead stay on its provider force path.
        Uncommitted transitions, invalid geometry and non-tensile inputs load zero.
        """
        if force_owner != "native-tension":
            raise ValueError(
                "Provider-native force cannot enter a native route pullback."
            )
        points = _endpoints(endpoints_m)
        velocity = _endpoints(endpoint_velocity_m_per_s)
        tension = jnp.asarray(tensile_force_N, dtype=points.dtype)
        if tension.shape != ():
            raise ValueError("tensile_force_N must be scalar.")
        evaluation = self.evaluate_fixed_branch(state, points)
        eligible = (
            evaluation.evidence.fixed_branch_gradient_supported
            & jnp.isfinite(tension)
            & (tension >= 0.0)
            & jnp.all(jnp.isfinite(velocity))
        )
        operator = self.length_jacobian_operator(state, points)
        rate = operator.mv(velocity)[0]
        loads = operator.transpose_mv(-tension[None])
        route_power = -tension * rate
        endpoint_power = jnp.sum(loads * velocity)
        residual = endpoint_power - route_power
        scale = jnp.maximum(
            1.0, jnp.maximum(jnp.abs(route_power), jnp.abs(endpoint_power))
        )
        eligible &= (
            jnp.all(jnp.isfinite(loads))
            & jnp.isfinite(rate)
            & jnp.isfinite(residual)
            & (jnp.abs(residual) <= self.plan.residual_tolerance * scale)
        )
        evidence = OpenSimCylinderWrapPullbackEvidence(
            tension,
            jnp.where(eligible, rate, 0.0),
            jnp.where(eligible, route_power, 0.0),
            jnp.where(eligible, endpoint_power, 0.0),
            jnp.where(eligible, residual, 0.0),
            eligible,
        )
        return jnp.where(eligible, loads, jnp.zeros_like(loads)), evidence


__all__ = [
    "CylinderWrapSide",
    "OpenSimCylinderRouteWrapPlan",
    "PreparedOpenSimCylinderRouteWrap",
    "OpenSimCylinderWrapCandidate",
    "OpenSimCylinderWrapEvaluation",
    "OpenSimCylinderWrapEvidence",
    "OpenSimCylinderWrapPullbackEvidence",
    "OpenSimCylinderWrapState",
    "OpenSimCylinderWrapStatus",
]
