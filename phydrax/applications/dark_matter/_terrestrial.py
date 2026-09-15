#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._numerics import gauss_legendre_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import CompiledGeometry, Sphere
from ...integration import WeightedSampleBatch
from ...solver import (
    HybridSchedulePlan,
    JumpDifferentialSolution,
    solve_jump_differential,
)
from ...stochastic import PoissonClockRealization
from ._profiles import LayeredTerrestrialProfile
from ._rates import spherical_surface_crossings, SurfaceCrossingMeasure
from ._scattering import ElasticScatteringTable
from ._transport import (
    jump_differential_problem,
    ProfiledElasticJumpProcess,
    radial_guard_schedule,
    RadialGravityDrift,
    transport_collision_evidence,
    transport_path_evidence,
    TransportCollisionEvidence,
    TransportNumericalStatus,
    TransportOutcome,
    TransportPathEvidence,
)


class HazardQuadraturePlan(StrictModule, NonTrainableState):
    """Fixed Gauss--Legendre rule for a bounded transparent path."""

    nodes: Array
    weights: Array
    order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, order: int = 32, /):
        if not isinstance(order, int) or isinstance(order, bool) or order < 2:
            raise ValueError(
                "Hazard quadrature order must be an integer of at least two."
            )
        rule = gauss_legendre_data(order)
        self.nodes = rule.nodes
        self.weights = rule.weights
        self.order = order
        self.plan_id = canonical_fingerprint(
            {"kind": "dark-matter-hazard-quadrature", "order": order}
        )


class HazardIntegrationResult(StrictModule):
    """Per-target optical depth and bounded-integration evidence."""

    partial_optical_depths: Array
    total_optical_depth: Array
    interaction_cdf: Array
    distance_m: Array
    finite: Array
    successful: Array
    status: Array
    method: str = eqx.field(static=True)


class TerrestrialTransportResult(StrictModule):
    """Guarded trajectories, detector measure, physics, and numerical evidence."""

    solution: JumpDifferentialSolution
    collisions: TransportCollisionEvidence
    detector_crossings: SurfaceCrossingMeasure
    outcomes: Array
    outcome_one_hot: Array
    evidence: TransportPathEvidence
    plan_id: str = eqx.field(static=True)


def _ray(position_m: ArrayLike, direction: ArrayLike, distance_m: ArrayLike, /):
    position = jnp.asarray(position_m, dtype=float)
    direction_ = jnp.asarray(direction, dtype=position.dtype)
    distance = jnp.asarray(distance_m, dtype=position.dtype).reshape(())
    if position.shape != (3,) or direction_.shape != (3,):
        raise ValueError("Ray position and direction must have shape (3,).")
    direction_norm = jnp.sqrt(ein.contract("i,i->", direction_, direction_))
    invalid = (
        jnp.any(~jnp.isfinite(position))
        | jnp.any(~jnp.isfinite(direction_))
        | ~jnp.isfinite(distance)
        | (direction_norm <= 0.0)
        | (distance < 0.0)
    )
    position = eqx.error_if(
        position,
        invalid,
        "Ray inputs must be finite with nonzero direction and distance >= 0.",
    )
    return position, direction_ / direction_norm, distance


def _layer_breakpoints(
    profile: LayeredTerrestrialProfile,
    position: Array,
    direction: Array,
    distance: Array,
    /,
) -> Array:
    projection = ein.contract("i,i->", position, direction)
    squared_position = ein.contract("i,i->", position, position)
    discriminants = projection**2 - (squared_position - profile.outer_radii_m**2)
    roots = jnp.sqrt(jnp.maximum(discriminants, 0.0))
    near = -projection - roots
    far = -projection + roots
    valid_near = (discriminants >= 0.0) & (near > 0.0) & (near < distance)
    valid_far = (discriminants >= 0.0) & (far > 0.0) & (far < distance)
    candidates = jnp.concatenate(
        (
            jnp.asarray((0.0,), dtype=position.dtype),
            jnp.where(valid_near, near, distance),
            jnp.where(valid_far, far, distance),
            distance.reshape((1,)),
        )
    )
    return jnp.sort(candidates)


def quadrature_optical_depth(
    profile: LayeredTerrestrialProfile,
    scattering: ElasticScatteringTable,
    position_m: ArrayLike,
    direction: ArrayLike,
    distance_m: ArrayLike,
    projectile_speed_m_s: ArrayLike,
    /,
    *,
    quadrature: HazardQuadraturePlan | None = None,
) -> HazardIntegrationResult:
    """Integrate generic profiled hazards along a bounded straight SI path."""
    if not isinstance(profile, LayeredTerrestrialProfile) or not isinstance(
        scattering, ElasticScatteringTable
    ):
        raise TypeError("Optical depth requires layered profile and scattering tables.")
    if profile.target_ids != scattering.target_ids:
        raise ValueError("Profile and scattering target axes must match exactly.")
    plan = HazardQuadraturePlan() if quadrature is None else quadrature
    if not isinstance(plan, HazardQuadraturePlan):
        raise TypeError("quadrature must be a HazardQuadraturePlan or None.")
    position, direction_, distance = _ray(position_m, direction, distance_m)
    speed = jnp.asarray(projectile_speed_m_s, dtype=position.dtype).reshape(())
    speed = eqx.error_if(
        speed,
        ~jnp.isfinite(speed) | (speed <= 0.0),
        "projectile_speed_m_s must be finite and positive.",
    )
    breakpoints = _layer_breakpoints(profile, position, direction_, distance)
    left = breakpoints[:-1]
    lengths = jnp.diff(breakpoints)
    active_intervals = lengths > 0.0
    coordinates = left[:, None] + 0.5 * lengths[:, None] * (plan.nodes[None, :] + 1.0)
    weights = 0.5 * lengths[:, None] * plan.weights[None, :]
    points = position[None, None, :] + coordinates[..., None] * direction_
    evaluated = profile.evaluate(points.reshape((-1, 3)))
    interval_active = jnp.broadcast_to(
        active_intervals[:, None], coordinates.shape
    ).reshape((-1,))
    number_density = jnp.where(
        interval_active[:, None], evaluated.target_number_densities_m3, 0.0
    )
    velocity = speed * direction_
    rates = jax.vmap(
        lambda number, temperature: scattering.partial_rates(
            number, velocity, temperature
        )
    )(number_density, evaluated.temperature_K)
    hazards_per_m = rates.reshape(coordinates.shape + (profile.num_targets,)) / speed
    partial = ein.contract("iq,iqt->t", weights, hazards_per_m)
    total = jnp.sum(partial)
    finite = (
        jnp.all(evaluated.finite)
        & jnp.all(jnp.isfinite(partial))
        & jnp.all(partial >= 0.0)
    )
    return HazardIntegrationResult(
        partial,
        total,
        -jnp.expm1(-total),
        distance,
        finite,
        finite,
        jnp.where(
            finite,
            int(TransportNumericalStatus.SUCCESS),
            int(TransportNumericalStatus.INVALID_RATE),
        ).astype(jnp.int32),
        "layer-split-gauss-legendre",
    )


def layered_analytic_optical_depth(
    profile: LayeredTerrestrialProfile,
    scattering: ElasticScatteringTable,
    position_m: ArrayLike,
    direction: ArrayLike,
    distance_m: ArrayLike,
    projectile_speed_m_s: ArrayLike,
    /,
) -> HazardIntegrationResult:
    """Exactly sum piecewise-constant layer hazards between sphere intersections."""
    if not isinstance(profile, LayeredTerrestrialProfile) or not isinstance(
        scattering, ElasticScatteringTable
    ):
        raise TypeError("Analytic optical depth requires layered profile and tables.")
    if profile.target_ids != scattering.target_ids:
        raise ValueError("Profile and scattering target axes must match exactly.")
    position, direction_, distance = _ray(position_m, direction, distance_m)
    speed = jnp.asarray(projectile_speed_m_s, dtype=position.dtype).reshape(())
    speed = eqx.error_if(
        speed,
        ~jnp.isfinite(speed) | (speed <= 0.0),
        "projectile_speed_m_s must be finite and positive.",
    )
    breakpoints = _layer_breakpoints(profile, position, direction_, distance)
    lengths = jnp.diff(breakpoints)
    active_intervals = lengths > 0.0
    midpoints = (
        position[None, :]
        + (0.5 * (breakpoints[:-1] + breakpoints[1:]))[:, None] * direction_[None, :]
    )
    evaluated = profile.evaluate(midpoints)
    number_density = jnp.where(
        active_intervals[:, None], evaluated.target_number_densities_m3, 0.0
    )
    velocity = speed * direction_
    rates = jax.vmap(
        lambda number, temperature: scattering.partial_rates(
            number, velocity, temperature
        )
    )(number_density, evaluated.temperature_K)
    partial = ein.contract("i,it->t", lengths, rates / speed)
    total = jnp.sum(partial)
    finite = (
        jnp.all(evaluated.finite)
        & jnp.all(jnp.isfinite(partial))
        & jnp.all(partial >= 0.0)
    )
    return HazardIntegrationResult(
        partial,
        total,
        -jnp.expm1(-total),
        distance,
        finite,
        finite,
        jnp.where(
            finite,
            int(TransportNumericalStatus.SUCCESS),
            int(TransportNumericalStatus.INVALID_RATE),
        ).astype(jnp.int32),
        "piecewise-analytic",
    )


def sample_target_from_partial_rates(key: Array, partial_rates: ArrayLike, /) -> Array:
    """Draw the target mark from explicit partial hazards, without total-rate bias."""
    rates = jnp.asarray(partial_rates, dtype=float)
    if rates.ndim != 1 or rates.size == 0:
        raise ValueError("partial_rates must be a non-empty vector.")
    valid = jnp.all(jnp.isfinite(rates) & (rates >= 0.0)) & (jnp.sum(rates) > 0.0)
    logits = jnp.where(rates > 0.0, jnp.log(rates), -jnp.inf)
    selected = jr.categorical(key, logits).astype(jnp.int32)
    return jnp.where(valid, selected, -1).astype(jnp.int32)


class TerrestrialTransportPlan(StrictModule, NonTrainableState):
    """Bounded guarded transport through a qualified concentric body profile."""

    profile: LayeredTerrestrialProfile
    scattering: ElasticScatteringTable
    process: ProfiledElasticJumpProcess
    detector_depth_m: Array
    detector: CompiledGeometry
    schedule: HybridSchedulePlan
    maximum_jump_events: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    detector_in_event_index: int = eqx.field(static=True)
    detector_out_event_index: int = eqx.field(static=True)
    surface_out_event_index: int = eqx.field(static=True)

    def __init__(
        self,
        profile: LayeredTerrestrialProfile,
        scattering: ElasticScatteringTable,
        projectile_mass_kg: ArrayLike,
        /,
        *,
        detector_depth_m: float,
        maximum_jump_events: int = 64,
        maximum_guard_events: int = 64,
    ):
        if not isinstance(profile, LayeredTerrestrialProfile) or not isinstance(
            scattering, ElasticScatteringTable
        ):
            raise TypeError("Terrestrial transport requires layered profile and tables.")
        depth = float(detector_depth_m)
        radius = float(profile.radius_m)
        if not np.isfinite(depth) or not 0.0 <= depth < radius:
            raise ValueError("detector_depth_m must lie in [0, body radius).")
        if not isinstance(maximum_jump_events, int) or maximum_jump_events <= 0:
            raise ValueError("maximum_jump_events must be positive.")
        detector_radius = radius - depth
        detector = Sphere(
            (0.0, 0.0, 0.0),
            detector_radius,
            feature_id=f"{profile.frame_id}:detector-depth:{depth:.17g}",
        ).compile()
        specifications: list[tuple[str, CompiledGeometry, int, bool, int]] = []
        for index, boundary in enumerate(profile.boundaries[:-1]):
            specifications.append((f"layer:{index}", boundary, 0, False, 0))
        specifications.extend(
            (
                ("surface:in", profile.boundaries[-1], -1, False, 10),
                ("surface:out", profile.boundaries[-1], 1, True, 10),
            )
        )
        internal_count = profile.num_layers - 1
        surface_in_event_index = internal_count
        surface_out_event_index = internal_count + 1
        if depth > 0.0:
            specifications.extend(
                (
                    ("detector:in", detector, -1, False, 20),
                    ("detector:out", detector, 1, False, 20),
                )
            )
            detector_in_event_index = internal_count + 2
            detector_out_event_index = internal_count + 3
        else:
            detector_in_event_index = surface_in_event_index
            detector_out_event_index = surface_out_event_index
        self.profile = profile
        self.scattering = scattering
        self.process = ProfiledElasticJumpProcess(profile, scattering, projectile_mass_kg)
        self.detector_depth_m = jnp.asarray(depth)
        self.detector = detector
        self.schedule = radial_guard_schedule(
            tuple(specifications),
            RadialGravityDrift(profile),
            frame_id=profile.frame_id,
            maximum_events=maximum_guard_events,
        )
        self.maximum_jump_events = maximum_jump_events
        self.surface_out_event_index = surface_out_event_index
        self.detector_in_event_index = detector_in_event_index
        self.detector_out_event_index = detector_out_event_index
        self.plan_id = canonical_fingerprint(
            {
                "kind": "terrestrial-dark-matter-transport",
                "profile": profile.profile_id,
                "process": self.process.process_id,
                "detector_depth_m": depth,
                "schedule": self.schedule.plan_id,
                "maximum_jump_events": maximum_jump_events,
            }
        )

    @property
    def detector_radius_m(self) -> Array:
        return self.profile.radius_m - self.detector_depth_m

    def simulate(
        self,
        initial_paths: WeightedSampleBatch,
        poisson: PoissonClockRealization,
        save_times_s: ArrayLike,
        /,
    ) -> TerrestrialTransportResult:
        """Execute guarded transport with exact stochastic replay.

        Sample columns are ``(x_m, y_m, z_m, vx_m_s, vy_m_s, vz_m_s)`` in
        ``profile.frame_id``. Event histories are not pathwise gradients.
        """
        if not isinstance(initial_paths, WeightedSampleBatch):
            raise TypeError("initial_paths must be a WeightedSampleBatch.")
        if not isinstance(poisson, PoissonClockRealization):
            raise TypeError("poisson must be a PoissonClockRealization.")
        raw_states = jnp.asarray(initial_paths.samples, dtype=float)
        if raw_states.ndim != 2 or raw_states.shape[0] == 0 or raw_states.shape[1] != 6:
            raise ValueError("initial path samples must have nonempty shape (path, 6).")
        if initial_paths.sample_axes != (0,):
            raise ValueError("initial_paths must use one leading trajectory sample axis.")
        if jnp.asarray(initial_paths.log_weights).shape != (raw_states.shape[0],):
            raise ValueError("initial path log weights must match the trajectory axis.")
        path_mask = (
            jnp.ones((raw_states.shape[0],), dtype=bool)
            if initial_paths.mask is None
            else jnp.asarray(initial_paths.mask, dtype=bool)
        )
        path_support = (
            jnp.ones((raw_states.shape[0],), dtype=bool)
            if initial_paths.support_valid is None
            else jnp.broadcast_to(
                jnp.asarray(initial_paths.support_valid, dtype=bool),
                (raw_states.shape[0],),
            )
        )
        active_paths = path_mask & path_support
        if not bool(jnp.any(active_paths)):
            raise ValueError("Terrestrial transport requires at least one active path.")
        fallback = raw_states[int(jnp.argmax(active_paths))]
        states = jnp.where(active_paths[:, None], raw_states, fallback)
        if poisson.sample_shape != (states.shape[0],):
            raise ValueError("Poisson sample_shape must match the trajectory path axis.")
        available = poisson.num_channels * poisson.max_events_per_channel
        if self.maximum_jump_events > available:
            raise ValueError(
                "Poisson clocks do not provide the plan's jump-event capacity."
            )
        times = jnp.asarray(save_times_s, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise ValueError("save_times_s must contain at least two times.")
        problem = jump_differential_problem(
            self.process,
            states[0],
            t0_s=float(times[0]),
            t1_s=float(times[-1]),
        )
        solution = solve_jump_differential(
            problem,
            poisson,
            save_times=times,
            initial_states=states,
            hybrid_schedule=self.schedule,
            max_events=self.maximum_jump_events,
        )
        deterministic = solution.deterministic_events
        if deterministic is None:
            raise RuntimeError(
                "Guarded terrestrial solve returned no deterministic evidence."
            )
        if solution.events.pre_states is None:
            raise RuntimeError("Guarded terrestrial solve returned no collision states.")
        collisions = transport_collision_evidence(
            self.process,
            solution.events.pre_states,
            solution.events.channels,
            solution.events.marks,
            solution.events.valid,
        )
        finite = jnp.all(jnp.isfinite(solution.states), axis=(-1, -2))
        evidence = transport_path_evidence(
            solution.events.status,
            deterministic.event_count,
            deterministic.capacity_exceeded,
            finite,
            solver_successful=solution.numerical_successful,
            collision_invariants_valid=jnp.all(collisions.successful, axis=-1),
            initial_valid=active_paths,
        )
        detector_event = (deterministic.event_indices == self.detector_in_event_index) | (
            deterministic.event_indices == self.detector_out_event_index
        )
        detector_valid = deterministic.valid & detector_event
        qualified_detector = detector_valid & evidence.successful[..., None]
        crossings = spherical_surface_crossings(
            deterministic.event_states_before,
            qualified_detector,
            initial_paths,
            self.detector_radius_m,
            direction="both",
        )
        transmitted = jnp.any(
            deterministic.valid
            & (deterministic.event_indices == self.surface_out_event_index),
            axis=-1,
        )
        outcomes = jnp.where(
            evidence.successful & transmitted,
            int(TransportOutcome.TRANSMITTED),
            int(TransportOutcome.UNRESOLVED),
        ).astype(jnp.int32)
        one_hot = jax.nn.one_hot(outcomes, len(TransportOutcome), dtype=bool)
        return TerrestrialTransportResult(
            solution,
            collisions,
            crossings,
            outcomes,
            one_hot,
            evidence,
            self.plan_id,
        )


__all__ = [
    "HazardIntegrationResult",
    "HazardQuadraturePlan",
    "TerrestrialTransportPlan",
    "TerrestrialTransportResult",
    "layered_analytic_optical_depth",
    "quadrature_optical_depth",
    "sample_target_from_partial_rates",
]
