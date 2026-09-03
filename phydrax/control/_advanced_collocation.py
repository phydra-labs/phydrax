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
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

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
    increments = ein.contract("ij,njd->nid", method.A, flat_rates)
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
    endpoint_increment = ein.contract("j,njd->nd", method.b, flat_rates).reshape(
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
    local_stages = ein.contract("ij,jd->id", method.A, flat).reshape(tangents.shape)
    local_endpoint = ein.contract("j,jd->d", method.b, flat).reshape(anchor_.shape)
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


class ManifoldCollocationEvidence(StrictModule, NonTrainableState):
    """Interval-local evidence for one manifold collocation evaluation.

    ``contained`` has columns for the interval anchor, every Radau stage, the
    retracted collocation endpoint, and the supplied right endpoint, in that
    order. The remaining arrays have one entry per interval; ``valid`` is their
    global conjunction.
    """

    finite: Array
    contained: Array
    chart_valid: Array
    equation_valid: Array
    valid: Array
    chart_tolerance: float = eqx.field(static=True)
    equation_tolerance: float = eqx.field(static=True)


class ManifoldRadauCollocationDefects(StrictModule, NonTrainableState):
    """Geometry-aware Radau defects expressed in shared local coordinates."""

    stage_times: Array
    stage_states: Array
    predicted_endpoints: Array
    stage_defects: Array
    endpoint_defects: Array
    evidence: ManifoldCollocationEvidence
    configuration_convention: Literal["retraction"] = eqx.field(static=True)
    tangent_convention: Literal["shared-local"] = eqx.field(static=True)
    implicit: bool = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def finite(self) -> Array:
        return jnp.all(self.evidence.finite)

    @property
    def contained(self) -> Array:
        return self.evidence.contained

    @property
    def chart_valid(self) -> Array:
        return jnp.all(self.evidence.chart_valid)

    @property
    def equation_valid(self) -> Array:
        return jnp.all(self.evidence.equation_valid)

    @property
    def valid(self) -> Array:
        return self.evidence.valid


def manifold_radau_collocation_defects(
    method: RadauIIAMethod,
    geometry: AbstractStateGeometry,
    dynamics: Callable[..., Array],
    times: ArrayLike,
    states: ArrayLike,
    stage_local_rates: ArrayLike,
    controls: ArrayLike,
    /,
    *,
    configuration_convention: Literal["retraction"],
    tangent_convention: Literal["shared-local"],
    args: Any = None,
    implicit: bool = False,
    chart_tolerance: float = 1.0e-6,
    equation_tolerance: float = 1.0e-8,
) -> ManifoldRadauCollocationDefects:
    """Evaluate fixed-grid Radau defects in one anchored local chart.

    Node states and every local rate have the same array shape. On each
    interval, ``stage_local_rates`` are derivatives of coordinates in the
    retraction anchored at the left node. Radau combinations are evaluated in
    that chart, and endpoint differences use ``inverse_retract`` at the same
    anchor.

    Explicit ``dynamics(time, state, control, args)`` returns a state-shaped
    physical rate. The inverse differential declared by ``geometry.pullback``
    maps it into the anchored stage coordinates. Implicit
    ``dynamics(time, state, state_rate, control, args)`` receives the physical
    rate produced by the anchored retraction JVP; its state-shaped tangent
    residual is pulled back through that same differential.

    Exact inverse-differential capability is required. If it is unavailable,
    the evaluator returns non-finite sentinel stage defects and invalid typed
    evidence rather than substituting the unrelated ``to_local`` or
    ``from_local`` maps. Chart evidence also checks the endpoint round-trip and
    JVP/pullback consistency. Equation evidence requires finite stage and
    endpoint defects bounded by ``equation_tolerance``.
    """

    if not isinstance(method, RadauIIAMethod):
        raise TypeError("method must be a RadauIIAMethod.")
    if not isinstance(geometry, AbstractStateGeometry):
        raise TypeError("geometry must be an AbstractStateGeometry.")
    if not callable(dynamics):
        raise TypeError("dynamics must be callable.")
    if configuration_convention != "retraction":
        raise ValueError("configuration_convention must be 'retraction'.")
    if tangent_convention != "shared-local":
        raise ValueError("tangent_convention must be 'shared-local'.")
    if not isinstance(implicit, bool):
        raise TypeError("implicit must be a boolean.")
    chart_tolerance_ = float(chart_tolerance)
    equation_tolerance_ = float(equation_tolerance)
    if not np.isfinite(chart_tolerance_) or chart_tolerance_ < 0.0:
        raise ValueError("chart_tolerance must be finite and nonnegative.")
    if not np.isfinite(equation_tolerance_) or equation_tolerance_ < 0.0:
        raise ValueError("equation_tolerance must be finite and nonnegative.")

    times_ = jnp.asarray(times)
    states_ = jnp.asarray(states)
    rates = jnp.asarray(stage_local_rates)
    controls_ = jnp.asarray(controls)
    if times_.ndim != 1 or times_.size < 2:
        raise ValueError("Manifold Radau times must be a rank-one interval grid.")
    if states_.ndim < 1 or states_.shape[0] != times_.size:
        raise ValueError("Manifold states must provide one point per grid time.")
    intervals = times_.size - 1
    state_shape = states_.shape[1:]
    if any(size == 0 for size in state_shape):
        raise ValueError("Manifold state points must be nonempty.")
    expected_rate_shape = (intervals, method.stage_count) + state_shape
    if rates.shape != expected_rate_shape:
        raise ValueError(
            "stage_local_rates must have shape "
            f"{expected_rate_shape}; got {rates.shape}."
        )
    if controls_.ndim < 1 or controls_.shape[0] != intervals:
        raise ValueError("controls must provide one value per Radau interval.")
    widths = times_[1:] - times_[:-1]
    times_ = eqx.error_if(
        times_,
        jnp.any(~jnp.isfinite(times_)) | jnp.any(widths <= 0),
        "Manifold Radau interval times must be finite and increasing.",
    )

    flat_rates = rates.reshape((intervals, method.stage_count, -1))
    stage_local = (
        widths[:, None, None] * ein.contract("ij,njd->nid", method.A, flat_rates)
    ).reshape(rates.shape)
    endpoint_local = (
        widths[:, None] * ein.contract("j,njd->nd", method.b, flat_rates)
    ).reshape((intervals,) + state_shape)
    stage_times = times_[:-1, None] + widths[:, None] * method.c[None, :]
    anchors = states_[:-1]
    right_endpoints = states_[1:]

    def retract_stages(anchor, local):
        return jax.vmap(lambda value: jnp.asarray(geometry.retract(anchor, value)))(
            local
        )

    stage_states = jax.vmap(retract_stages)(anchors, stage_local)
    if stage_states.shape != expected_rate_shape:
        raise ValueError(
            "geometry.retract must preserve the state shape at every Radau stage."
        )
    predicted_endpoints = jax.vmap(
        lambda anchor, local: jnp.asarray(geometry.retract(anchor, local))
    )(anchors, endpoint_local)
    if predicted_endpoints.shape != (intervals,) + state_shape:
        raise ValueError("geometry.retract must preserve endpoint state shape.")
    observed_endpoint_local = jax.vmap(
        lambda anchor, point: jnp.asarray(geometry.inverse_retract(anchor, point))
    )(anchors, right_endpoints)
    if observed_endpoint_local.shape != (intervals,) + state_shape:
        raise ValueError(
            "geometry.inverse_retract must return state-shaped coordinates."
        )

    observed_endpoint_reconstructions = jax.vmap(
        lambda anchor, local: jnp.asarray(geometry.retract(anchor, local))
    )(anchors, observed_endpoint_local)
    if observed_endpoint_reconstructions.shape != (intervals,) + state_shape:
        raise ValueError(
            "geometry.retract must preserve reconstructed endpoint state shape."
        )

    def membership(point):
        contained = jnp.asarray(geometry.contains(point), dtype=bool)
        if contained.shape != ():
            raise ValueError("geometry.contains must return one scalar boolean.")
        return contained

    node_contained = jax.vmap(membership)(states_)
    stage_contained = jax.vmap(jax.vmap(membership))(stage_states)
    predicted_contained = jax.vmap(membership)(predicted_endpoints)
    contained = jnp.concatenate(
        (
            node_contained[:-1, None],
            stage_contained,
            predicted_contained[:, None],
            node_contained[1:, None],
        ),
        axis=1,
    )

    def projected_physical(point, ambient):
        projected = jnp.asarray(geometry.project_tangent(point, ambient))
        if projected.shape != point.shape:
            raise ValueError("geometry.project_tangent must preserve state shape.")
        return projected

    def anchored_pullback(anchor, local, tangent):
        coordinate_rate = jnp.asarray(geometry.pullback(anchor, local, tangent))
        if coordinate_rate.shape != anchor.shape:
            raise ValueError(
                "geometry.pullback must return state-shaped stage-coordinate rates."
            )
        return coordinate_rate

    def anchored_jvp(anchor, local, coordinate_rate):
        _, physical_rate = jax.jvp(
            lambda coordinates: jnp.asarray(geometry.retract(anchor, coordinates)),
            (local,),
            (coordinate_rate,),
        )
        if physical_rate.shape != anchor.shape:
            raise ValueError(
                "The anchored retraction JVP must return state-shaped physical rates."
            )
        return physical_rate

    def pullback_interval(anchor, local, tangent):
        return jax.vmap(
            lambda coordinates, vector: anchored_pullback(
                anchor, coordinates, vector
            )
        )(local, tangent)

    def jvp_interval(anchor, local, coordinate_rate):
        return jax.vmap(
            lambda coordinates, velocity: anchored_jvp(
                anchor, coordinates, velocity
            )
        )(local, coordinate_rate)

    def interval_finite(value):
        return jnp.all(jnp.isfinite(value.reshape((intervals, -1))), axis=1)

    def interval_error(left, right):
        return jnp.max(
            jnp.abs(left - right).reshape((intervals, -1)),
            axis=1,
        )

    tangent_equation_valid = jnp.ones((intervals,), dtype=bool)
    dynamics_finite = jnp.ones((intervals,), dtype=bool)
    differential_finite = jnp.ones((intervals,), dtype=bool)
    differential_valid = jnp.ones((intervals,), dtype=bool)

    if not geometry.supports_exact_pullback:
        sentinel_dtype = jnp.result_type(rates.dtype, jnp.float32)
        stage_defects = jnp.full(
            expected_rate_shape,
            jnp.nan,
            dtype=sentinel_dtype,
        )
        tangent_equation_valid = jnp.zeros((intervals,), dtype=bool)
        dynamics_finite = jnp.zeros((intervals,), dtype=bool)
        differential_finite = jnp.zeros((intervals,), dtype=bool)
        differential_valid = jnp.zeros((intervals,), dtype=bool)
    elif implicit:
        ambient_rates = jax.vmap(jvp_interval)(anchors, stage_local, rates)
        if ambient_rates.shape != expected_rate_shape:
            raise ValueError(
                "The anchored retraction JVP must preserve the stage-rate shape."
            )
        recovered_rates = jax.vmap(pullback_interval)(
            anchors, stage_local, ambient_rates
        )
        if recovered_rates.shape != expected_rate_shape:
            raise ValueError(
                "geometry.pullback must preserve the stage-rate shape."
            )

        def interval_equations(t, y, y_dot, control):
            return jax.vmap(
                lambda ti, yi, y_dot_i: jnp.asarray(
                    dynamics(ti, yi, y_dot_i, control, args)
                )
            )(t, y, y_dot)

        ambient_equations = jax.vmap(interval_equations)(
            stage_times, stage_states, ambient_rates, controls_
        )
        if ambient_equations.shape != expected_rate_shape:
            raise ValueError(
                "Implicit dynamics must return one state-shaped tangent residual "
                "per stage."
            )
        projected_equations = jax.vmap(jax.vmap(projected_physical))(
            stage_states, ambient_equations
        )
        stage_defects = jax.vmap(pullback_interval)(
            anchors, stage_local, projected_equations
        )
        reconstructed_equations = jax.vmap(jvp_interval)(
            anchors, stage_local, stage_defects
        )
        if reconstructed_equations.shape != expected_rate_shape:
            raise ValueError(
                "The anchored retraction JVP must preserve the equation shape."
            )
        rate_inverse_error = interval_error(recovered_rates, rates)
        equation_inverse_error = interval_error(
            reconstructed_equations, projected_equations
        )
        differential_finite = (
            interval_finite(ambient_rates)
            & interval_finite(recovered_rates)
            & interval_finite(projected_equations)
            & interval_finite(stage_defects)
            & interval_finite(reconstructed_equations)
        )
        differential_valid = (
            differential_finite
            & (rate_inverse_error <= chart_tolerance_)
            & (equation_inverse_error <= chart_tolerance_)
        )
        tangent_equation_valid = (
            interval_finite(ambient_equations)
            & interval_finite(reconstructed_equations)
            & (equation_inverse_error <= equation_tolerance_)
        )
        dynamics_finite = (
            interval_finite(ambient_rates)
            & interval_finite(ambient_equations)
            & interval_finite(projected_equations)
            & interval_finite(reconstructed_equations)
        )
    else:

        def interval_fields(t, y, control):
            return jax.vmap(
                lambda ti, yi: jnp.asarray(dynamics(ti, yi, control, args))
            )(t, y)

        ambient_fields = jax.vmap(interval_fields)(
            stage_times, stage_states, controls_
        )
        if ambient_fields.shape != expected_rate_shape:
            raise ValueError(
                "Explicit dynamics must return one state-shaped vector per stage."
            )
        projected_fields = jax.vmap(jax.vmap(projected_physical))(
            stage_states, ambient_fields
        )
        local_fields = jax.vmap(pullback_interval)(
            anchors, stage_local, projected_fields
        )
        reconstructed_fields = jax.vmap(jvp_interval)(
            anchors, stage_local, local_fields
        )
        if reconstructed_fields.shape != expected_rate_shape:
            raise ValueError(
                "The anchored retraction JVP must preserve the vector-field shape."
            )
        stage_defects = rates - local_fields
        field_inverse_error = interval_error(
            reconstructed_fields, projected_fields
        )
        differential_finite = (
            interval_finite(projected_fields)
            & interval_finite(local_fields)
            & interval_finite(reconstructed_fields)
        )
        differential_valid = differential_finite & (
            field_inverse_error <= chart_tolerance_
        )
        dynamics_finite = (
            interval_finite(ambient_fields)
            & interval_finite(projected_fields)
            & interval_finite(local_fields)
            & interval_finite(reconstructed_fields)
        )

    endpoint_defects = observed_endpoint_local - endpoint_local

    chart_finite = (
        interval_finite(stage_local)
        & interval_finite(endpoint_local)
        & interval_finite(observed_endpoint_local)
        & interval_finite(stage_states)
        & interval_finite(predicted_endpoints)
        & interval_finite(observed_endpoint_reconstructions)
    )
    equation_finite = interval_finite(stage_defects) & interval_finite(
        endpoint_defects
    )
    input_finite = (
        jnp.isfinite(times_[:-1])
        & jnp.isfinite(times_[1:])
        & interval_finite(anchors)
        & interval_finite(right_endpoints)
        & interval_finite(rates)
        & interval_finite(controls_)
    )
    finite = (
        input_finite
        & chart_finite
        & dynamics_finite
        & differential_finite
        & equation_finite
    )
    chart_error = jnp.max(
        jnp.abs(observed_endpoint_reconstructions - right_endpoints).reshape(
            (intervals, -1)
        ),
        axis=1,
    )
    chart_valid = (
        jnp.all(contained, axis=1)
        & chart_finite
        & differential_valid
        & (chart_error <= chart_tolerance_)
    )
    maximum_stage_defect = jnp.max(
        jnp.abs(stage_defects).reshape((intervals, -1)), axis=1
    )
    maximum_endpoint_defect = jnp.max(
        jnp.abs(endpoint_defects).reshape((intervals, -1)), axis=1
    )
    equation_valid = (
        tangent_equation_valid
        & differential_valid
        & equation_finite
        & (maximum_stage_defect <= equation_tolerance_)
        & (maximum_endpoint_defect <= equation_tolerance_)
    )
    valid = jnp.all(finite & chart_valid & equation_valid)
    evidence = ManifoldCollocationEvidence(
        finite,
        contained,
        chart_valid,
        equation_valid,
        valid,
        chart_tolerance_,
        equation_tolerance_,
    )
    return ManifoldRadauCollocationDefects(
        stage_times,
        stage_states,
        predicted_endpoints,
        stage_defects,
        endpoint_defects,
        evidence,
        configuration_convention,
        tangent_convention,
        implicit,
        geometry.geometry_id,
        method.method_id,
    )


__all__ = [
    "ComplementarityConstraint",
    "ComplementarityEvidence",
    "ComplementarityHomotopyPolicy",
    "DirectCollocationLink",
    "DirectCollocationPhase",
    "ManifoldCollocationEvidence",
    "ManifoldCollocationStages",
    "ManifoldRadauCollocationDefects",
    "MultiphaseDirectCollocationEvidence",
    "MultiphaseDirectCollocationProblem",
    "RadauCollocationDefects",
    "StochasticDirectCollocationEvidence",
    "StochasticDirectTranscription",
    "audit_complementarity",
    "audit_multiphase_links",
    "manifold_radau_stages",
    "manifold_radau_collocation_defects",
    "radau_collocation_defects",
]
