#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any, Literal, TypeAlias

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ...metrix._connection import (
    connection_geodesic_rhs,
    connection_parallel_transport_rhs,
    LeviCivitaConnection,
)
from ...metrix._metric import LorentzianMetric
from ...metrix._metric_domain import MetricDomainEvidence
from ...metrix._spacetime_conventions import RelativityConvention
from ...solver._differential import DifferentialProblem
from ...solver._diffrax_backend import solve_diffrax
from ...units import UnitDefinition
from ._gr_bundles import (
    AbstractGRConstantOfMotion,
    build_gr_ray_bundle_evidence,
    gr_chart_identity,
    gr_metric_identity,
    GRJacobiEvidence,
    GRRayBundleEvidence,
)
from ._gr_events import (
    GRRayEventCode,
    GRRayEventLedger,
    GRRayEventMargin,
    GRRayEventSurfaces,
    ordered_gr_ray_event_code,
)
from ._gr_screens import GRObserverScreenResult
from ._gr_status import GRRayStatus, GRRayStatusEvidence


GRRayKind: TypeAlias = Literal["null", "timelike"]


class GRRayState(StrictModule):
    """Fixed-capacity batched initial states for null or timelike trajectories."""

    coordinates: Array
    tangents: Array
    screen_basis: Array | None
    active: Array

    def __init__(
        self,
        coordinates: ArrayLike,
        tangents: ArrayLike,
        /,
        *,
        screen_basis: ArrayLike | None = None,
        active: ArrayLike | None = None,
    ):
        points = jnp.asarray(coordinates)
        velocities = jnp.asarray(tangents, dtype=points.dtype)
        if points.ndim != 2 or points.shape[-1] != 4 or velocities.shape != points.shape:
            raise ValueError(
                "GR ray coordinates and tangents must have shape (num_rays, 4)."
            )
        if not jnp.issubdtype(points.dtype, jnp.inexact):
            points = points.astype(float)
            velocities = velocities.astype(float)
        basis = (
            None
            if screen_basis is None
            else jnp.asarray(screen_basis, dtype=points.dtype)
        )
        if basis is not None and basis.shape != (points.shape[0], 2, 4):
            raise ValueError("screen_basis must have shape (num_rays, 2, 4).")
        mask = (
            jnp.ones((points.shape[0],), dtype=bool)
            if active is None
            else jnp.asarray(active, dtype=bool)
        )
        if mask.shape != (points.shape[0],):
            raise ValueError("active must have shape (num_rays,).")
        self.coordinates = points
        self.tangents = velocities
        self.screen_basis = basis
        self.active = mask

    @property
    def num_rays(self) -> int:
        return int(self.coordinates.shape[0])


class GRRayPlan(StrictModule):
    """Batched relativistic trajectory plan with static history and work caps."""

    metric: LorentzianMetric
    scale: RelativityScaleContract
    convention: RelativityConvention
    coordinate_unit: UnitDefinition
    affine_parameter_unit: UnitDefinition
    initial_state: GRRayState
    affine_parameter: Array
    events: GRRayEventSurfaces
    constants_of_motion: tuple[AbstractGRConstantOfMotion, ...]
    solver: Any | None
    stepsize_controller: Any | None
    adjoint: Any | None
    dt0: Array | None
    ray_kind: GRRayKind = eqx.field(static=True)
    transport_screen_basis: bool = eqx.field(static=True)
    track_jacobi: bool = eqx.field(static=True)
    affine_budget_is_event: bool = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    event_tolerance: float = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    coordinate_unit_id: str = eqx.field(static=True)
    affine_parameter_unit_id: str = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        metric: LorentzianMetric,
        initial_state: GRRayState,
        affine_parameter: ArrayLike,
        /,
        *,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        coordinate_unit: UnitDefinition,
        affine_parameter_unit: UnitDefinition,
        metric_semantic_id: str | None = None,
        metric_numeric_id: str | None = None,
        ray_kind: GRRayKind = "null",
        events: GRRayEventSurfaces | None = None,
        capture_margin: GRRayEventMargin | None = None,
        escape_margin: GRRayEventMargin | None = None,
        domain_margin: GRRayEventMargin | None = None,
        transport_screen_basis: bool = False,
        track_jacobi: bool = False,
        constants_of_motion: Sequence[AbstractGRConstantOfMotion] = (),
        affine_budget_is_event: bool = True,
        solver: Any | None = None,
        stepsize_controller: Any | None = None,
        adjoint: Any | None = None,
        dt0: ArrayLike | None = None,
        relative_tolerance: float = 1.0e-7,
        absolute_tolerance: float = 1.0e-9,
        constraint_tolerance: float = 1.0e-5,
        event_tolerance: float = 1.0e-8,
        maximum_steps: int = 4096,
        plan_id: str | None = None,
    ):
        if not isinstance(metric, LorentzianMetric) or metric.chart.dimension != 4:
            raise TypeError("GR ray plans require a four-dimensional LorentzianMetric.")
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        if not isinstance(convention, RelativityConvention):
            raise TypeError("convention must be a RelativityConvention.")
        if convention.metric_signature != metric.convention:
            raise ValueError(
                "RelativityConvention metric signature must match the metric."
            )
        if not isinstance(coordinate_unit, UnitDefinition):
            raise TypeError("coordinate_unit must be a UnitDefinition.")
        if not isinstance(affine_parameter_unit, UnitDefinition):
            raise TypeError("affine_parameter_unit must be a UnitDefinition.")
        if not isinstance(initial_state, GRRayState):
            raise TypeError("initial_state must be a GRRayState.")
        affine = jnp.asarray(affine_parameter, dtype=initial_state.coordinates.dtype)
        if affine.ndim != 1 or affine.shape[0] < 2:
            raise ValueError(
                "affine_parameter must contain at least two fixed history nodes."
            )
        if bool(jnp.any(~jnp.isfinite(affine))) or bool(jnp.any(jnp.diff(affine) <= 0.0)):
            raise ValueError("affine_parameter must be finite and strictly increasing.")
        if ray_kind not in ("null", "timelike"):
            raise ValueError("ray_kind must be 'null' or 'timelike'.")
        if events is not None and any(
            margin is not None
            for margin in (capture_margin, escape_margin, domain_margin)
        ):
            raise ValueError("Pass events or individual event margins, not both.")
        surfaces = (
            GRRayEventSurfaces(
                capture_margin=capture_margin,
                escape_margin=escape_margin,
                domain_margin=domain_margin,
            )
            if events is None
            else events
        )
        if not isinstance(surfaces, GRRayEventSurfaces):
            raise TypeError("events must be a GRRayEventSurfaces.")
        transport = bool(transport_screen_basis)
        jacobi = bool(track_jacobi)
        if (transport or jacobi) and initial_state.screen_basis is None:
            raise ValueError(
                "Screen-basis transport and Jacobi tracking require initial screen_basis."
            )
        quantities = tuple(constants_of_motion)
        if any(not isinstance(value, AbstractGRConstantOfMotion) for value in quantities):
            raise TypeError(
                "constants_of_motion must contain AbstractGRConstantOfMotion values."
            )
        if len({value.name for value in quantities}) != len(quantities):
            raise ValueError("constants_of_motion names must be unique.")
        rtol = float(relative_tolerance)
        atol = float(absolute_tolerance)
        constraint = float(constraint_tolerance)
        event_tol = float(event_tolerance)
        if any(
            not isfinite(value) or value <= 0.0
            for value in (rtol, atol, constraint, event_tol)
        ):
            raise ValueError("Ray tolerances must be finite and positive.")
        steps = int(maximum_steps)
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        step = None if dt0 is None else jnp.asarray(dt0, dtype=affine.dtype)
        if step is not None and step.shape != ():
            raise ValueError("dt0 must be scalar or None.")
        metric_id = gr_metric_identity(
            metric,
            semantic_id=metric_semantic_id,
            numeric_id=metric_numeric_id,
        )
        chart_id = gr_chart_identity(metric)
        plan_id_ = (
            canonical_fingerprint(
                {
                    "kind": "gr-ray-plan",
                    "metric_id": metric_id,
                    "chart_id": chart_id,
                    "convention_id": convention.convention_id,
                    "scale_id": scale.scale_id,
                    "coordinate_unit_id": coordinate_unit.unit_id,
                    "affine_parameter_unit_id": affine_parameter_unit.unit_id,
                    "coordinates": initial_state.coordinates,
                    "tangents": initial_state.tangents,
                    "screen_basis": initial_state.screen_basis,
                    "active": initial_state.active,
                    "affine_parameter": affine,
                    "ray_kind": ray_kind,
                    "events": surfaces.event_id,
                    "transport_screen_basis": transport,
                    "track_jacobi": jacobi,
                    "constants": tuple(value.constant_id for value in quantities),
                    "affine_budget_is_event": bool(affine_budget_is_event),
                    "rtol": rtol,
                    "atol": atol,
                    "constraint_tolerance": constraint,
                    "event_tolerance": event_tol,
                    "maximum_steps": steps,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not plan_id_:
            raise ValueError("plan_id must be non-empty.")
        self.metric = metric
        self.scale = scale
        self.convention = convention
        self.coordinate_unit = coordinate_unit
        self.affine_parameter_unit = affine_parameter_unit
        self.initial_state = initial_state
        self.affine_parameter = affine
        self.events = surfaces
        self.constants_of_motion = quantities
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.adjoint = adjoint
        self.dt0 = step
        self.ray_kind = ray_kind
        self.transport_screen_basis = transport
        self.track_jacobi = jacobi
        self.affine_budget_is_event = bool(affine_budget_is_event)
        self.relative_tolerance = rtol
        self.absolute_tolerance = atol
        self.constraint_tolerance = constraint
        self.event_tolerance = event_tol
        self.metric_id = metric_id
        self.chart_id = chart_id
        self.convention_id = convention.convention_id
        self.scale_id = scale.scale_id
        self.coordinate_unit_id = coordinate_unit.unit_id
        self.affine_parameter_unit_id = affine_parameter_unit.unit_id
        self.maximum_steps = steps
        self.plan_id = plan_id_

    @classmethod
    def from_screen(
        cls,
        metric: LorentzianMetric,
        screen: GRObserverScreenResult,
        affine_parameter: ArrayLike,
        /,
        **kwargs: Any,
    ) -> GRRayPlan:
        """Construct a null-ray plan from a materialized observer screen."""

        if not isinstance(screen, GRObserverScreenResult):
            raise TypeError("screen must be a GRObserverScreenResult.")
        if "ray_kind" in kwargs and kwargs["ray_kind"] != "null":
            raise ValueError("Observer screen rays are null trajectories.")
        state = GRRayState(
            screen.ray_coordinates,
            screen.ray_tangents,
            screen_basis=screen.screen_basis,
            active=screen.valid,
        )
        plan = cls(
            metric,
            state,
            affine_parameter,
            ray_kind="null",
            **{key: value for key, value in kwargs.items() if key != "ray_kind"},
        )
        screen_context = (
            screen.metric_id,
            screen.chart_id,
            screen.convention_id,
            screen.scale_id,
            screen.coordinate_unit_id,
            screen.affine_parameter_unit_id,
        )
        plan_context = (
            plan.metric_id,
            plan.chart_id,
            plan.convention_id,
            plan.scale_id,
            plan.coordinate_unit_id,
            plan.affine_parameter_unit_id,
        )
        if screen_context != plan_context:
            raise ValueError("Observer screen and ray plan contexts must match exactly.")
        return plan

    def trace(self) -> GRRayResult:
        return trace_gr_rays(self)


class _GRGeodesicDrift(StrictModule):
    connection: LeviCivitaConnection
    transport_screen_basis: bool = eqx.field(static=True)
    track_jacobi: bool = eqx.field(static=True)

    def __init__(
        self,
        metric: LorentzianMetric,
        /,
        *,
        transport_screen_basis: bool,
        track_jacobi: bool,
    ):
        self.connection = LeviCivitaConnection(metric)
        self.transport_screen_basis = transport_screen_basis
        self.track_jacobi = track_jacobi

    def _phase_rhs(self, phase: Array, /) -> Array:
        return connection_geodesic_rhs(self.connection, phase)

    def __call__(self, affine: Array, state: Array, args: Any, /) -> Array:
        del affine, args
        phase = state[:8]
        parts = [self._phase_rhs(phase)]
        offset = 8
        if self.transport_screen_basis or self.track_jacobi:
            basis = state[offset : offset + 8].reshape((2, 4))
            basis_rhs = jax.vmap(
                lambda vector: connection_parallel_transport_rhs(
                    self.connection,
                    phase[:4],
                    phase[4:],
                    vector,
                )
            )(basis)
            parts.append(basis_rhs.reshape((8,)))
            offset += 8
        if self.track_jacobi:
            variations = state[offset : offset + 16].reshape((2, 8))
            variation_rhs = jax.vmap(
                lambda variation: jax.jvp(
                    self._phase_rhs,
                    (phase,),
                    (variation,),
                )[1]
            )(variations)
            parts.append(variation_rhs.reshape((16,)))
        return jnp.concatenate(parts)


class _GREventCondition(StrictModule):
    margin: GRRayEventMargin | None

    def __init__(self, margin: GRRayEventMargin | None, /):
        self.margin = margin

    def __call__(
        self,
        t: Array,
        y: Array,
        args: Any,
        **kwargs: Any,
    ) -> Array:
        del args, kwargs
        if self.margin is None:
            return jnp.ones((), dtype=y.dtype)
        return jnp.asarray(self.margin(t, y[:4], y[4:8]))


class GRRayResult(StrictModule):
    """Fixed-shape batched trajectories and their independent scientific evidence."""

    coordinates: Array
    tangents: Array
    affine_parameter: Array
    active: Array
    valid: Array
    status: Array
    null_residual: Array
    transported_screen_basis: Array | None
    domain_evidence: MetricDomainEvidence
    event_ledger: GRRayEventLedger
    status_evidence: GRRayStatusEvidence
    bundle_evidence: GRRayBundleEvidence
    ray_kind: GRRayKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    coordinate_unit_id: str = eqx.field(static=True)
    affine_parameter_unit_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: ArrayLike,
        tangents: ArrayLike,
        affine_parameter: ArrayLike,
        active: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        null_residual: ArrayLike,
        transported_screen_basis: ArrayLike | None,
        domain_evidence: MetricDomainEvidence,
        event_ledger: GRRayEventLedger,
        status_evidence: GRRayStatusEvidence,
        bundle_evidence: GRRayBundleEvidence,
        /,
        *,
        ray_kind: GRRayKind,
        plan_id: str,
        result_id: str,
        metric_id: str,
        chart_id: str,
        convention_id: str,
        scale_id: str,
        coordinate_unit_id: str,
        affine_parameter_unit_id: str,
    ):
        points = jnp.asarray(coordinates)
        velocities = jnp.asarray(tangents, dtype=points.dtype)
        affine = jnp.asarray(affine_parameter, dtype=points.dtype)
        active_ = jnp.asarray(active, dtype=bool)
        valid_ = jnp.asarray(valid, dtype=bool)
        status_ = jnp.asarray(status, dtype=jnp.int32)
        residual = jnp.asarray(null_residual, dtype=points.dtype)
        if points.ndim != 3 or points.shape[-1] != 4 or velocities.shape != points.shape:
            raise ValueError(
                "GR ray histories must have shape (num_rays, num_history, 4)."
            )
        history_shape = points.shape[:2]
        if any(
            value.shape != history_shape for value in (affine, active_, valid_, residual)
        ):
            raise ValueError(
                "Affine parameters, masks, and residuals must match ray/history axes."
            )
        if status_.shape != (points.shape[0],):
            raise ValueError("status must have shape (num_rays,).")
        basis = (
            None
            if transported_screen_basis is None
            else jnp.asarray(transported_screen_basis, dtype=points.dtype)
        )
        if basis is not None and basis.shape != history_shape + (2, 4):
            raise ValueError(
                "Transported screen bases must have shape (rays, history, 2, 4)."
            )
        if not isinstance(domain_evidence, MetricDomainEvidence):
            raise TypeError("domain_evidence must be MetricDomainEvidence.")
        if domain_evidence.margin.shape != history_shape:
            raise ValueError("Domain evidence must match the ray/history axes.")
        if not isinstance(event_ledger, GRRayEventLedger):
            raise TypeError("event_ledger must be a GRRayEventLedger.")
        if not isinstance(status_evidence, GRRayStatusEvidence):
            raise TypeError("status_evidence must be GRRayStatusEvidence.")
        if not isinstance(bundle_evidence, GRRayBundleEvidence):
            raise TypeError("bundle_evidence must be GRRayBundleEvidence.")
        if ray_kind not in ("null", "timelike"):
            raise ValueError("ray_kind must be 'null' or 'timelike'.")
        identities = (
            plan_id,
            result_id,
            metric_id,
            chart_id,
            convention_id,
            scale_id,
            coordinate_unit_id,
            affine_parameter_unit_id,
        )
        if any(not isinstance(value, str) or not value for value in identities):
            raise ValueError("GR ray result identities must be non-empty strings.")
        self.coordinates = points
        self.tangents = velocities
        self.affine_parameter = affine
        self.active = active_
        self.valid = valid_
        self.status = status_
        self.null_residual = residual
        self.transported_screen_basis = basis
        self.domain_evidence = domain_evidence
        self.event_ledger = event_ledger
        self.status_evidence = status_evidence
        self.bundle_evidence = bundle_evidence
        self.ray_kind = ray_kind
        self.plan_id = plan_id
        self.result_id = result_id
        self.metric_id = metric_id
        self.chart_id = chart_id
        self.convention_id = convention_id
        self.scale_id = scale_id
        self.coordinate_unit_id = coordinate_unit_id
        self.affine_parameter_unit_id = affine_parameter_unit_id

    @property
    def event_code(self) -> Array:
        return self.event_ledger.event_code

    @property
    def jacobi_evidence(self) -> GRJacobiEvidence | None:
        return self.bundle_evidence.jacobi

    @property
    def transport_residual(self) -> Array:
        return self.bundle_evidence.transport_residual

    @property
    def terminal_transported_screen_basis(self) -> Array | None:
        if self.transported_screen_basis is None:
            return None
        return self.event_ledger.state[..., 8:16].reshape(
            self.coordinates.shape[0],
            2,
            4,
        )

    @property
    def constants_of_motion(self) -> Array:
        return self.bundle_evidence.constant_values

    @property
    def constant_relative_drift(self) -> Array:
        return self.bundle_evidence.constant_relative_drift

    @property
    def finite(self) -> Array:
        return self.status_evidence.finite

    @property
    def converged(self) -> Array:
        return self.status_evidence.converged

    @property
    def physically_valid(self) -> Array:
        return self.status_evidence.physically_valid

    @property
    def qualified(self) -> Array:
        return self.status_evidence.qualified

    @property
    def derivative_valid(self) -> Array:
        return self.status_evidence.derivative_valid


def _initial_packed_state(plan: GRRayPlan, /) -> Array:
    parts = (plan.initial_state.coordinates, plan.initial_state.tangents)
    packed = jnp.concatenate(parts, axis=-1)
    if plan.transport_screen_basis or plan.track_jacobi:
        assert plan.initial_state.screen_basis is not None
        packed = jnp.concatenate(
            (packed, plan.initial_state.screen_basis.reshape((packed.shape[0], 8))),
            axis=-1,
        )
    if plan.track_jacobi:
        assert plan.initial_state.screen_basis is not None
        zero = jnp.zeros_like(plan.initial_state.screen_basis)
        variations = jnp.concatenate((zero, plan.initial_state.screen_basis), axis=-1)
        packed = jnp.concatenate(
            (packed, variations.reshape((packed.shape[0], 16))), axis=-1
        )
    return packed


def _mass_shell_target(plan: GRRayPlan, /) -> float:
    if plan.ray_kind == "null":
        return 0.0
    return float(-1 if plan.metric.convention == "mostly_plus" else 1)


def _initial_validity(plan: GRRayPlan, /) -> tuple[Array, Array, Array]:
    points = plan.initial_state.coordinates
    tangents = plan.initial_state.tangents
    margins = jax.vmap(
        lambda point, tangent: plan.events.margins(
            plan.affine_parameter[0], point, tangent
        )
    )(points, tangents)
    domain_safe = jnp.isfinite(margins[:, 2]) & (margins[:, 2] > 0.0)
    domain_safe = domain_safe | jnp.isposinf(margins[:, 2])
    finite_input = (
        jnp.all(jnp.isfinite(points), axis=-1)
        & jnp.all(jnp.isfinite(tangents), axis=-1)
        & jnp.all(jnp.isfinite(margins) | jnp.isposinf(margins), axis=-1)
    )
    nonzero_tangent = jnp.max(jnp.abs(tangents), axis=-1) > jnp.finfo(points.dtype).tiny
    target = _mass_shell_target(plan)

    def one(point: Array, tangent: Array, enabled: Array) -> Array:
        return jax.lax.cond(
            enabled,
            lambda operands: jnp.abs(
                plan.metric.quadratic_form(operands[1], operands[0]) - target
            ),
            lambda operands: jnp.asarray(jnp.inf, dtype=operands[0].dtype),
            (point, tangent),
        )

    residual = jax.vmap(one)(points, tangents, domain_safe & finite_input)
    valid = (
        plan.initial_state.active
        & finite_input
        & domain_safe
        & nonzero_tangent
        & jnp.isfinite(residual)
        & (residual <= plan.constraint_tolerance)
    )
    return valid, residual, finite_input


def _diffrax_event(plan: GRRayPlan, /) -> dfx.Event | None:
    if not plan.events.enabled:
        return None
    conditions = (
        _GREventCondition(plan.events.capture_margin),
        _GREventCondition(plan.events.escape_margin),
        _GREventCondition(plan.events.domain_margin),
    )
    root_finder = optx.Newton(
        rtol=plan.event_tolerance,
        atol=plan.event_tolerance,
    )
    return dfx.Event(
        conditions,
        root_finder=root_finder,
        direction=(False, False, False),
    )


def _solve_batched(
    plan: GRRayPlan,
    packed_initial: Array,
    initial_valid: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    drift = _GRGeodesicDrift(
        plan.metric,
        transport_screen_basis=plan.transport_screen_basis,
        track_jacobi=plan.track_jacobi,
    )
    event = _diffrax_event(plan)
    history_size = plan.affine_parameter.shape[0]

    def solve_one(
        initial: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        problem = DifferentialProblem(
            drift,
            initial,
            t0=plan.affine_parameter[0],
            t1=plan.affine_parameter[-1],
            problem_id=f"{plan.plan_id}:lane",
        )
        solution = solve_diffrax(
            problem,
            save_times=plan.affine_parameter,
            solver=plan.solver,
            stepsize_controller=plan.stepsize_controller,
            adjoint=plan.adjoint,
            dt0=plan.dt0,
            event=event,
            rtol=plan.relative_tolerance,
            atol=plan.absolute_tolerance,
            max_steps=plan.maximum_steps,
            throw=False,
            solver_configuration_id=f"{plan.plan_id}:solver",
        )
        triggered = (
            jnp.zeros((3,), dtype=bool)
            if event is None
            else jnp.stack(
                tuple(jnp.asarray(value, dtype=bool) for value in solution.event_mask)
            )
        )
        return (
            solution.states,
            solution.valid,
            solution.backend_successful,
            triggered,
            jnp.asarray(solution.stats["num_steps"], dtype=jnp.int32),
            solution.terminal_time,
            solution.terminal_state,
            solution.terminal_valid,
        )

    def inactive_one(
        initial: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
        return (
            jnp.broadcast_to(initial, (history_size, initial.shape[0])),
            jnp.zeros((history_size,), dtype=bool),
            jnp.asarray(False),
            jnp.zeros((3,), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
            plan.affine_parameter[0],
            initial,
            jnp.asarray(False),
        )

    return jax.vmap(
        lambda initial, enabled: jax.lax.cond(
            enabled,
            solve_one,
            inactive_one,
            initial,
        )
    )(packed_initial, initial_valid)


def _fill_inactive_history(states: Array, active: Array, initial: Array, /) -> Array:
    def one(ray_states: Array, ray_active: Array, initial_state: Array) -> Array:
        def step(previous: Array, operands: tuple[Array, Array]) -> tuple[Array, Array]:
            candidate, enabled = operands
            current = jnp.where(enabled, candidate, previous)
            return current, current

        _, filled = jax.lax.scan(step, initial_state, (ray_states, ray_active))
        return filled

    return jax.vmap(one)(states, active, initial)


def _history_diagnostics(
    plan: GRRayPlan,
    coordinates: Array,
    tangents: Array,
    affine_parameter: Array,
    active: Array,
    /,
) -> tuple[Array, Array, Array]:
    target = _mass_shell_target(plan)

    def one(
        affine: Array,
        point: Array,
        tangent: Array,
        enabled: Array,
    ) -> tuple[Array, Array, Array]:
        def evaluate(
            operands: tuple[Array, Array, Array],
        ) -> tuple[Array, Array, Array]:
            affine_, point_, tangent_ = operands
            matrix = plan.metric(point_)
            value = ein.contract("i,ij,j->", tangent_, matrix, tangent_)
            margin = plan.events.margins(affine_, point_, tangent_)[2]
            finite = (
                jnp.all(jnp.isfinite(point_))
                & jnp.all(jnp.isfinite(tangent_))
                & jnp.all(jnp.isfinite(matrix))
                & jnp.isfinite(value)
            )
            finite_margin = jnp.where(
                jnp.isposinf(margin),
                jnp.asarray(jnp.finfo(point_.dtype).max, dtype=point_.dtype),
                margin,
            )
            return jnp.abs(value - target), finite, finite_margin

        return jax.lax.cond(
            enabled,
            evaluate,
            lambda operands: (
                jnp.asarray(0.0, dtype=operands[1].dtype),
                jnp.asarray(True),
                jnp.asarray(
                    jnp.finfo(operands[1].dtype).max,
                    dtype=operands[1].dtype,
                ),
            ),
            (affine, point, tangent),
        )

    return jax.vmap(jax.vmap(one))(
        affine_parameter,
        coordinates,
        tangents,
        active,
    )


def trace_gr_rays(plan: GRRayPlan, /) -> GRRayResult:
    """Trace every fixed ray lane with independent ordered terminal events."""

    if not isinstance(plan, GRRayPlan):
        raise TypeError("plan must be a GRRayPlan.")
    packed_initial = _initial_packed_state(plan)
    initial_valid, _, initial_finite = _initial_validity(plan)
    (
        raw_states,
        solver_valid,
        backend_successful,
        triggered,
        steps,
        terminal_time,
        terminal_state,
        terminal_valid,
    ) = _solve_batched(plan, packed_initial, initial_valid)
    active = solver_valid & initial_valid[:, None]
    states = _fill_inactive_history(raw_states, active, packed_initial)
    coordinates = states[..., :4]
    tangents = states[..., 4:8]
    offset = 8
    if plan.transport_screen_basis or plan.track_jacobi:
        transported_basis_internal = states[..., offset : offset + 8].reshape(
            states.shape[:2] + (2, 4)
        )
        offset += 8
    else:
        transported_basis_internal = None
    if plan.track_jacobi:
        jacobi_variations = states[..., offset : offset + 16].reshape(
            states.shape[:2] + (2, 8)
        )
    else:
        jacobi_variations = None

    affine = jnp.broadcast_to(plan.affine_parameter, active.shape)
    residual, history_finite, domain_margin = _history_diagnostics(
        plan, coordinates, tangents, affine, active
    )
    domain_evidence = MetricDomainEvidence.from_margin(
        domain_margin,
        chart=plan.metric.chart,
        domain_id=plan.events.event_id,
        boundary_tolerance=plan.event_tolerance,
        extra_valid=active & history_finite,
    )
    history_domain = domain_evidence.inside
    valid = (
        active
        & history_finite
        & history_domain
        & jnp.isfinite(residual)
        & (residual <= plan.constraint_tolerance)
    )
    has_event = jnp.any(triggered, axis=-1)
    maximum_step_work = (~backend_successful) & (steps >= plan.maximum_steps)
    planned_work = backend_successful & ~has_event & plan.affine_budget_is_event
    work = initial_valid & ~has_event & (maximum_step_work | planned_work)
    event_code = ordered_gr_ray_event_code(triggered, work)
    simultaneous = jnp.sum(triggered.astype(jnp.int32), axis=-1) > 1
    history_before_terminal = plan.affine_parameter[None, :] <= terminal_time[:, None]
    preceding_index = jnp.clip(
        jnp.sum(history_before_terminal, axis=1, dtype=jnp.int32) - 1,
        0,
        active.shape[1] - 1,
    )
    recorded = initial_valid & (has_event | work)
    ledger = GRRayEventLedger(
        event_code,
        terminal_time,
        preceding_index,
        terminal_state,
        terminal_valid,
        recorded,
        simultaneous,
        ledger_id=canonical_fingerprint(
            {"kind": "gr-ray-event-ledger", "plan": plan.plan_id}
        ),
    )

    (
        terminal_residual,
        terminal_finite,
        _,
    ) = _history_diagnostics(
        plan,
        terminal_state[:, None, :4],
        terminal_state[:, None, 4:8],
        terminal_time[:, None],
        initial_valid[:, None] & terminal_valid[:, None],
    )
    terminal_residual = terminal_residual[:, 0]
    terminal_finite = terminal_finite[:, 0]

    active_finite = jnp.all(jnp.where(active, history_finite, True), axis=1)
    finite = (
        initial_valid
        & jnp.any(active, axis=1)
        & active_finite
        & terminal_valid
        & terminal_finite
    )
    constraint_valid = jnp.all(
        jnp.where(active, residual <= plan.constraint_tolerance, True),
        axis=1,
    ) & (terminal_residual <= plan.constraint_tolerance)
    domain_exit = event_code == int(GRRayEventCode.DOMAIN)
    retained_domain_valid = jnp.all(
        jnp.where(active, domain_evidence.physically_valid, True), axis=1
    )
    physically_valid = finite & constraint_valid & retained_domain_valid & ~domain_exit
    reached_success_endpoint = (
        initial_valid
        & backend_successful
        & ~has_event
        & (not plan.affine_budget_is_event)
    )
    status = jnp.where(
        event_code == int(GRRayEventCode.CAPTURE),
        int(GRRayStatus.CAPTURED),
        jnp.where(
            event_code == int(GRRayEventCode.ESCAPE),
            int(GRRayStatus.ESCAPED),
            jnp.where(
                domain_exit,
                int(GRRayStatus.DOMAIN_EXIT),
                jnp.where(
                    event_code == int(GRRayEventCode.WORK),
                    int(GRRayStatus.WORK_EXHAUSTED),
                    jnp.where(
                        reached_success_endpoint,
                        int(GRRayStatus.SUCCESS),
                        int(GRRayStatus.NUMERICAL_FAILURE),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    status = jnp.where(
        ~initial_valid,
        int(GRRayStatus.INVALID_INITIAL_STATE),
        status,
    )
    status = jnp.where(
        ~initial_finite,
        int(GRRayStatus.NONFINITE),
        status,
    )
    status = jnp.where(
        initial_valid & ~finite,
        int(GRRayStatus.NONFINITE),
        status,
    )
    status = jnp.where(
        finite & ~constraint_valid,
        int(GRRayStatus.CONSTRAINT_VIOLATION),
        status,
    )
    status = jnp.where(
        ~plan.initial_state.active,
        int(GRRayStatus.INACTIVE),
        status,
    )

    bundle = build_gr_ray_bundle_evidence(
        plan.metric,
        coordinates,
        tangents,
        active,
        transported_basis_internal,
        jacobi_variations,
        plan.constants_of_motion,
        metric_id=plan.metric_id,
        tolerance=plan.constraint_tolerance,
        bundle_id=canonical_fingerprint({"kind": "gr-ray-bundle", "plan": plan.plan_id}),
    )
    terminal_outcome = (
        (status == int(GRRayStatus.CAPTURED))
        | (status == int(GRRayStatus.ESCAPED))
        | (status == int(GRRayStatus.SUCCESS))
    )
    qualified = physically_valid & backend_successful & terminal_outcome & bundle.valid
    smooth_domain = jnp.all(
        jnp.where(active, domain_evidence.derivative_valid, True), axis=1
    )
    derivative_valid = qualified & ~simultaneous & smooth_domain
    evidence = GRRayStatusEvidence(
        finite,
        backend_successful & initial_valid,
        physically_valid,
        qualified,
        derivative_valid,
    )
    result_id = canonical_fingerprint({"kind": "gr-ray-result", "plan": plan.plan_id})
    return GRRayResult(
        coordinates,
        tangents,
        affine,
        active,
        valid,
        status,
        residual,
        transported_basis_internal if plan.transport_screen_basis else None,
        domain_evidence,
        ledger,
        evidence,
        bundle,
        ray_kind=plan.ray_kind,
        plan_id=plan.plan_id,
        result_id=result_id,
        metric_id=plan.metric_id,
        chart_id=plan.chart_id,
        convention_id=plan.convention_id,
        scale_id=plan.scale_id,
        coordinate_unit_id=plan.coordinate_unit_id,
        affine_parameter_unit_id=plan.affine_parameter_unit_id,
    )


__all__ = [
    "GRRayKind",
    "gr_chart_identity",
    "gr_metric_identity",
    "GRRayPlan",
    "GRRayResult",
    "GRRayState",
    "trace_gr_rays",
]
