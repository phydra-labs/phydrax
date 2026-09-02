#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix import AbstractStateGeometry
from ..solver._dae_events import DAEResetMap
from ..solver._hybrid_event import HybridEventTape
from ..solver._hybrid_schedule import ScheduledHybridEvent
from ..solver._radau_iia import RadauIIAMethod
from ._direct_collocation import (
    DirectCollocationBounds,
    DirectCollocationDecision,
    DirectCollocationPlan,
)
from ._trajectory_optimization import TrajectoryOptimizationProblem


class RadauCollocationDefects(StrictModule, NonTrainableState):
    stage_times: Array
    stage_states: Array
    stage_defects: Array
    endpoint_defects: Array
    finite: Array
    method_id: str = eqx.field(static=True)


def radau_collocation_defects(
    method: RadauIIAMethod,
    dynamics: Callable[..., Array],
    times: ArrayLike,
    states: ArrayLike,
    stage_rates: ArrayLike,
    controls: ArrayLike,
    /,
    *,
    args: Any = None,
    implicit: bool = False,
) -> RadauCollocationDefects:
    """Evaluate fixed-topology explicit or DAE Radau IIA transcription defects."""

    if not isinstance(method, RadauIIAMethod):
        raise TypeError("method must be a RadauIIAMethod.")
    if not callable(dynamics):
        raise TypeError("dynamics must be callable.")
    times_ = jnp.asarray(times)
    states_ = jnp.asarray(states)
    rates = jnp.asarray(stage_rates)
    controls_ = jnp.asarray(controls)
    if times_.ndim != 1 or states_.shape[0] != times_.size or times_.size < 2:
        raise ValueError("Radau states must align with a rank-one interval time grid.")
    intervals = times_.size - 1
    if rates.shape[:2] != (intervals, method.stage_count):
        raise ValueError("stage_rates must have leading (intervals,stage_count) axes.")
    if controls_.shape[0] != intervals or rates.shape[2:] != states_.shape[1:]:
        raise ValueError("Radau rate/control/state interval shapes do not align.")
    widths = times_[1:] - times_[:-1]
    if np.any(np.asarray(widths) <= 0):
        raise ValueError("Radau interval times must be strictly increasing.")
    state_shape = states_.shape[1:]
    flat_rates = rates.reshape((intervals, method.stage_count, -1))
    increments = oe.contract("ij,njd->nid", method.A, flat_rates)
    stage_states = (
        states_[:-1, None].reshape((intervals, 1, -1))
        + widths[:, None, None] * increments
    )
    stage_states = stage_states.reshape((intervals, method.stage_count) + state_shape)
    stage_times = times_[:-1, None] + widths[:, None] * method.c[None, :]

    def interval_defects(t, y, k, u):
        if implicit:
            return jax.vmap(
                lambda ti, yi, ki: jnp.asarray(dynamics(ti, yi, ki, u, args))
            )(t, y, k)
        return k - jax.vmap(lambda ti, yi: jnp.asarray(dynamics(ti, yi, u, args)))(t, y)

    stage_defects = jax.vmap(interval_defects)(
        stage_times, stage_states, rates, controls_
    )
    endpoint_increment = oe.contract("j,njd->nd", method.b, flat_rates).reshape(
        (intervals,) + state_shape
    )
    endpoint_defects = (
        states_[1:]
        - states_[:-1]
        - widths.reshape((intervals,) + (1,) * len(state_shape)) * endpoint_increment
    )
    finite = jnp.all(jnp.isfinite(stage_defects)) & jnp.all(
        jnp.isfinite(endpoint_defects)
    )
    return RadauCollocationDefects(
        stage_times,
        stage_states,
        stage_defects,
        endpoint_defects,
        finite,
        method.method_id,
    )


class DirectCollocationPhase(StrictModule, NonTrainableState):
    problem: TrajectoryOptimizationProblem
    plan: DirectCollocationPlan | RadauIIAMethod
    initial_decision: DirectCollocationDecision | Array
    bounds: DirectCollocationBounds | None
    phase_id: str = eqx.field(static=True)

    def __init__(self, problem, plan, initial_decision, bounds=None, /, *, phase_id: str):
        if not isinstance(problem, TrajectoryOptimizationProblem):
            raise TypeError("phase problem must be a TrajectoryOptimizationProblem.")
        if not isinstance(plan, (DirectCollocationPlan, RadauIIAMethod)):
            raise TypeError("phase plan must be DirectCollocationPlan or RadauIIAMethod.")
        if bounds is not None and not isinstance(bounds, DirectCollocationBounds):
            raise TypeError("phase bounds must be DirectCollocationBounds or None.")
        self.problem = problem
        self.plan = plan
        self.initial_decision = initial_decision
        self.bounds = bounds
        self.phase_id = phase_id


class DirectCollocationLink(StrictModule, NonTrainableState):
    left_phase: int = eqx.field(static=True)
    right_phase: int = eqx.field(static=True)
    residual: Callable[[Array, Array, Any], Array]
    lower: Array
    upper: Array
    event: ScheduledHybridEvent | None
    dae_reset: DAEResetMap | None
    link_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_phase: int,
        right_phase: int,
        residual: Callable[[Array, Array, Any], Array],
        bounds: tuple[ArrayLike, ArrayLike],
        /,
        *,
        event: ScheduledHybridEvent | None = None,
        dae_reset: DAEResetMap | None = None,
        link_id: str,
    ):
        if not callable(residual):
            raise TypeError("link residual must be callable.")
        lower, upper = (jnp.asarray(value) for value in bounds)
        if lower.shape != upper.shape:
            raise ValueError("link bounds must have identical shapes.")
        if (event is None) != (dae_reset is None):
            if dae_reset is not None:
                raise ValueError("A DAE reset requires a scheduled event link.")
        if event is not None and not isinstance(event, ScheduledHybridEvent):
            raise TypeError("event must be a ScheduledHybridEvent or None.")
        self.left_phase = int(left_phase)
        self.right_phase = int(right_phase)
        self.residual = residual
        self.lower = lower
        self.upper = upper
        self.event = event
        self.dae_reset = dae_reset
        self.link_id = link_id


class MultiphaseDirectCollocationProblem(StrictModule, NonTrainableState):
    phases: tuple[DirectCollocationPhase, ...]
    links: tuple[DirectCollocationLink, ...]
    shared_parameter_space: Any
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        phases: Sequence[DirectCollocationPhase],
        links: Sequence[DirectCollocationLink],
        shared_parameter_space: Any = None,
        /,
        *,
        problem_id: str,
    ):
        phases_ = tuple(phases)
        links_ = tuple(links)
        if not phases_ or any(
            not isinstance(value, DirectCollocationPhase) for value in phases_
        ):
            raise ValueError("multiphase collocation requires fixed declared phases.")
        if any(not isinstance(value, DirectCollocationLink) for value in links_):
            raise TypeError("links must be DirectCollocationLink values.")
        for link in links_:
            if (
                link.left_phase < 0
                or link.right_phase >= len(phases_)
                or link.left_phase >= link.right_phase
            ):
                raise ValueError("phase links must follow the fixed forward phase order.")
        self.phases = phases_
        self.links = links_
        self.shared_parameter_space = shared_parameter_space
        self.problem_id = canonical_fingerprint(
            {
                "kind": "multiphase-direct-collocation",
                "user_id": problem_id,
                "phases": [value.phase_id for value in phases_],
                "links": [value.link_id for value in links_],
            }
        )


class MultiphaseDirectCollocationEvidence(StrictModule, NonTrainableState):
    link_residuals: tuple[Array, ...]
    link_valid: Array
    topology_valid: Array
    event_tape: HybridEventTape | None
    valid: Array
    problem_id: str = eqx.field(static=True)


def audit_multiphase_links(
    problem: MultiphaseDirectCollocationProblem,
    endpoints: Sequence[tuple[ArrayLike, ArrayLike]],
    /,
    *,
    args: Any = None,
    event_tape: HybridEventTape | None = None,
    tolerance: float = 1.0e-8,
) -> MultiphaseDirectCollocationEvidence:
    """Audit all fixed phase links independently of the NLP solver."""

    if not isinstance(problem, MultiphaseDirectCollocationProblem):
        raise TypeError("problem must be a MultiphaseDirectCollocationProblem.")
    endpoints_ = tuple(
        (jnp.asarray(left), jnp.asarray(right)) for left, right in endpoints
    )
    if len(endpoints_) != len(problem.phases):
        raise ValueError("endpoints must provide one initial/final pair per phase.")
    residuals = tuple(
        jnp.asarray(
            link.residual(
                endpoints_[link.left_phase][1], endpoints_[link.right_phase][0], args
            )
        )
        for link in problem.links
    )
    link_valid = jnp.asarray(
        tuple(
            jnp.all(jnp.isfinite(value))
            & jnp.all(value >= link.lower - tolerance)
            & jnp.all(value <= link.upper + tolerance)
            for value, link in zip(residuals, problem.links, strict=True)
        )
    )
    event_links = sum(link.event is not None for link in problem.links)
    topology_valid = jnp.asarray(
        event_links == 0
        or (event_tape is not None and event_tape.event_count == event_links)
    )
    valid = (
        jnp.all(link_valid)
        & topology_valid
        & (event_tape is None or ~event_tape.capacity_exceeded)
    )
    return MultiphaseDirectCollocationEvidence(
        residuals, link_valid, topology_valid, event_tape, valid, problem.problem_id
    )


class ComplementarityConstraint(StrictModule, NonTrainableState):
    pair: Callable[[Any], tuple[Array, Array]]
    scale: float = eqx.field(static=True)
    form: Literal["product", "fischer-burmeister"] = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        pair: Callable[[Any], tuple[Array, Array]],
        scale: float = 1.0,
        /,
        *,
        form: Literal["product", "fischer-burmeister"] = "product",
        constraint_id: str,
    ):
        if not callable(pair):
            raise TypeError("complementarity pair must be callable.")
        scale_ = float(scale)
        if not np.isfinite(scale_) or scale_ <= 0:
            raise ValueError("complementarity scale must be positive and finite.")
        if form not in ("product", "fischer-burmeister"):
            raise ValueError("unknown complementarity residual form.")
        self.pair = pair
        self.scale = scale_
        self.form = form
        self.constraint_id = constraint_id

    def residual(
        self, decision: Any, mu: ArrayLike = 0.0, /
    ) -> tuple[Array, Array, Array]:
        a, b = (jnp.asarray(value) for value in self.pair(decision))
        if a.shape != b.shape:
            raise ValueError("complementarity pair arrays must have identical shapes.")
        mu_ = jnp.asarray(mu)
        residual = (
            (a * b - mu_) / self.scale
            if self.form == "product"
            else (jnp.sqrt(a * a + b * b + 2 * mu_) - a - b) / self.scale
        )
        return a, b, residual


class ComplementarityHomotopyPolicy(StrictModule, NonTrainableState):
    mu_values: tuple[float, ...] = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)

    def __init__(self, mu_values: Sequence[float], residual_tolerance: float, /):
        values = tuple(float(value) for value in mu_values)
        tolerance = float(residual_tolerance)
        if (
            not values
            or any(not np.isfinite(value) or value < 0 for value in values)
            or any(values[index + 1] > values[index] for index in range(len(values) - 1))
        ):
            raise ValueError(
                "complementarity mu_values must be finite, nonnegative, and nonincreasing."
            )
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("residual_tolerance must be finite and nonnegative.")
        self.mu_values = values
        self.residual_tolerance = tolerance


class ComplementarityEvidence(StrictModule, NonTrainableState):
    minimum_a: Array
    minimum_b: Array
    maximum_residual: Array
    final_mu: Array
    strict_complementarity: Array
    exact: Array
    valid: Array
    constraint_id: str = eqx.field(static=True)


def audit_complementarity(
    constraint: ComplementarityConstraint,
    decision: Any,
    /,
    *,
    final_mu: ArrayLike = 0.0,
    tolerance: float = 1.0e-8,
) -> ComplementarityEvidence:
    a, b, residual = constraint.residual(decision, final_mu)
    mu = jnp.asarray(final_mu)
    strict = jnp.all((a > tolerance) | (b > tolerance))
    valid = (
        jnp.all(a >= -tolerance)
        & jnp.all(b >= -tolerance)
        & (jnp.max(jnp.abs(residual)) <= tolerance)
    )
    exact = valid & (mu == 0)
    return ComplementarityEvidence(
        jnp.min(a),
        jnp.min(b),
        jnp.max(jnp.abs(residual)),
        mu,
        strict,
        exact,
        valid,
        constraint.constraint_id,
    )


class StochasticDirectTranscription(StrictModule, NonTrainableState):
    realization: Any
    method: Literal["ito-euler", "stratonovich-heun"] = eqx.field(static=True)
    scenario_weights: Array
    nonanticipativity: str = eqx.field(static=True)
    transcription_id: str = eqx.field(static=True)

    def __init__(
        self,
        realization: Any,
        method: Literal["ito-euler", "stratonovich-heun"],
        scenario_weights: ArrayLike,
        /,
        *,
        nonanticipativity: str = "shared-open-loop",
        transcription_id: str,
    ):
        weights = jnp.asarray(scenario_weights)
        if (
            weights.ndim != 1
            or weights.size == 0
            or np.any(np.asarray(weights) < 0)
            or not np.isclose(float(np.asarray(jnp.sum(weights))), 1.0)
        ):
            raise ValueError("scenario weights must be a nonnegative normalized vector.")
        if method not in ("ito-euler", "stratonovich-heun"):
            raise ValueError("unknown stochastic direct transcription method.")
        if nonanticipativity != "shared-open-loop":
            raise ValueError(
                "initial stochastic transcription supports shared open-loop controls only."
            )
        self.realization = realization
        self.method = method
        self.scenario_weights = weights
        self.nonanticipativity = nonanticipativity
        self.transcription_id = transcription_id


class StochasticDirectCollocationEvidence(StrictModule, NonTrainableState):
    scenario_defects: Array
    scenario_valid: Array
    weighted_residual: Array
    path_ids: tuple[str, ...] = eqx.field(static=True)
    valid: Array
    transcription_id: str = eqx.field(static=True)


class ManifoldCollocationStages(StrictModule, NonTrainableState):
    stages: Array
    endpoint: Array
    contained: Array
    chart_valid: Array
    method_id: str = eqx.field(static=True)


def manifold_radau_stages(
    method: RadauIIAMethod,
    geometry: AbstractStateGeometry,
    anchor: ArrayLike,
    stage_tangents: ArrayLike,
    /,
) -> ManifoldCollocationStages:
    """Map every stage/endpoint through one declared fixed retraction chart."""

    if not isinstance(method, RadauIIAMethod) or not isinstance(
        geometry, AbstractStateGeometry
    ):
        raise TypeError(
            "method and geometry must be RadauIIAMethod/AbstractStateGeometry."
        )
    anchor_ = jnp.asarray(anchor)
    tangents = jnp.asarray(stage_tangents)
    if tangents.shape[0] != method.stage_count:
        raise ValueError("stage_tangents leading axis must equal stage_count.")
    flat = tangents.reshape((method.stage_count, -1))
    local_stages = oe.contract("ij,jd->id", method.A, flat).reshape(tangents.shape)
    local_endpoint = oe.contract("j,jd->d", method.b, flat).reshape(anchor_.shape)
    stages = jax.vmap(lambda local: geometry.retract(anchor_, local))(local_stages)
    endpoint = geometry.retract(anchor_, local_endpoint)
    contained = jnp.concatenate(
        (
            jax.vmap(geometry.contains)(stages).reshape((-1,)),
            jnp.asarray(geometry.contains(endpoint)).reshape((1,)),
        )
    )
    chart_valid = (
        jnp.all(contained)
        & jnp.all(jnp.isfinite(stages))
        & jnp.all(jnp.isfinite(endpoint))
    )
    return ManifoldCollocationStages(
        stages, endpoint, contained, chart_valid, method.method_id
    )


__all__ = [
    "ComplementarityConstraint",
    "ComplementarityEvidence",
    "ComplementarityHomotopyPolicy",
    "DirectCollocationLink",
    "DirectCollocationPhase",
    "ManifoldCollocationStages",
    "MultiphaseDirectCollocationEvidence",
    "MultiphaseDirectCollocationProblem",
    "RadauCollocationDefects",
    "StochasticDirectCollocationEvidence",
    "StochasticDirectTranscription",
    "audit_complementarity",
    "audit_multiphase_links",
    "manifold_radau_stages",
    "radau_collocation_defects",
]
