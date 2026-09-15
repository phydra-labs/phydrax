#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
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
from ...units import KILOGRAM, METER, SECOND
from ..astrodynamics import (
    AstrodynamicsContext,
    CartesianOrbitState,
    propagate_universal_kepler,
    UniversalKeplerPolicy,
)
from ._profiles import SmoothStellarRadialProfile
from ._rates import spherical_surface_crossings, SurfaceCrossingMeasure
from ._scattering import ElasticScatteringTable
from ._transport import (
    BodyFrameTransportState,
    jump_differential_problem,
    ProfiledElasticJumpProcess,
    radial_guard_schedule,
    RadialGravityDrift,
    transport_collision_evidence,
    transport_path_evidence,
    TransportCollisionEvidence,
    TransportOutcome,
    TransportPathEvidence,
)


_GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.67430e-11


def _require_si_physical_inertial_context(context: AstrodynamicsContext, /) -> None:
    scale = context.scale
    if (
        scale.length_unit.unit_id != METER.unit_id
        or scale.mass_unit.unit_id != KILOGRAM.unit_id
        or scale.time_unit.unit_id != SECOND.unit_id
        or scale.length_coordinate_kind != "physical"
        or not context.epoch.continuous
        or not context.frame.pseudo_inertial
    ):
        raise ValueError(
            "Solar transport requires continuous, pseudo-inertial physical SI context."
        )


class ExteriorKeplerTransportResult(StrictModule):
    """Analytic exterior propagation with specific-energy evidence."""

    state: BodyFrameTransportState
    specific_energy_before_m2_s2: Array
    specific_energy_after_m2_s2: Array
    energy_defect_m2_s2: Array
    iterations: Array
    residual: Array
    valid: Array
    status: Array


class SolarOutcomeClassification(StrictModule):
    """Exclusive solar physical outcomes after numerical qualification."""

    outcomes: Array
    one_hot: Array


class StellarSpecificEnergyEvidence(StrictModule):
    """Knot-split interior energy with quadrature and sign evidence."""

    specific_energy_m2_s2: Array
    potential_m2_s2: Array
    quadrature_error_m2_s2: Array
    finite: Array
    sign_qualified: Array


class SolarTransportResult(StrictModule):
    """Guarded stellar paths with exclusive physics and numerical evidence."""

    solution: JumpDifferentialSolution
    collisions: TransportCollisionEvidence
    observation_crossings: SurfaceCrossingMeasure
    outcomes: Array
    final_states: Array
    outcome_one_hot: Array
    final_specific_energy_m2_s2: Array
    energy_error_m2_s2: Array
    energy_sign_qualified: Array
    scattering_count: Array
    evidence: TransportPathEvidence
    plan_id: str = eqx.field(static=True)


def kepler_specific_energy(
    state: BodyFrameTransportState | ArrayLike,
    central_mass_kg: ArrayLike,
    /,
) -> Array:
    """Specific Newtonian energy ``v^2/2 - G M/r`` in body-frame SI units."""
    packed = (
        state.packed()
        if isinstance(state, BodyFrameTransportState)
        else jnp.asarray(state)
    )
    if packed.shape[-1:] != (6,):
        raise ValueError("Kepler energy requires a trailing packed state axis of 6.")
    mass = jnp.asarray(central_mass_kg, dtype=packed.dtype).reshape(())
    radius = jnp.sqrt(jnp.sum(packed[..., :3] ** 2, axis=-1))
    speed_squared = jnp.sum(packed[..., 3:] ** 2, axis=-1)
    return 0.5 * speed_squared - _GRAVITATIONAL_CONSTANT_M3_KG_S2 * mass / radius


def propagate_exterior_kepler(
    state: BodyFrameTransportState,
    elapsed_s: ArrayLike,
    central_mass_kg: ArrayLike,
    context: AstrodynamicsContext,
    /,
    *,
    policy: UniversalKeplerPolicy | None = None,
) -> ExteriorKeplerTransportResult:
    """Use the native universal-variable Kepler propagator outside the star."""
    if not isinstance(state, BodyFrameTransportState):
        raise TypeError("state must be a BodyFrameTransportState.")
    if not isinstance(context, AstrodynamicsContext):
        raise TypeError("context must be an AstrodynamicsContext.")
    _require_si_physical_inertial_context(context)
    if state.frame_id != context.frame.frame_id:
        raise ValueError("Transport and astrodynamics body-frame identities differ.")
    mass = jnp.asarray(central_mass_kg, dtype=state.position_m.dtype).reshape(())
    mass = eqx.error_if(
        mass,
        ~jnp.isfinite(mass) | (mass <= 0.0),
        "central_mass_kg must be finite and positive.",
    )
    before = kepler_specific_energy(state, mass)
    native = propagate_universal_kepler(
        CartesianOrbitState(state.position_m, state.velocity_m_s, context),
        elapsed_s,
        _GRAVITATIONAL_CONSTANT_M3_KG_S2 * mass,
        policy=policy,
    )
    after_state = BodyFrameTransportState(
        native.state.position,
        native.state.velocity,
        frame_id=state.frame_id,
    )
    after = kepler_specific_energy(after_state, mass)
    return ExteriorKeplerTransportResult(
        after_state,
        before,
        after,
        after - before,
        native.iterations,
        native.residual,
        native.valid,
        native.status,
    )


def gravitational_focusing_speed(
    speed_at_observation_m_s: ArrayLike,
    observation_radius_m: ArrayLike,
    body_radius_m: ArrayLike,
    central_mass_kg: ArrayLike,
    /,
) -> Array:
    """Exact zero-scattering speed at the surface from Kepler energy conservation."""
    speed = jnp.asarray(speed_at_observation_m_s, dtype=float).reshape(())
    observation = jnp.asarray(observation_radius_m, dtype=speed.dtype).reshape(())
    body = jnp.asarray(body_radius_m, dtype=speed.dtype).reshape(())
    mass = jnp.asarray(central_mass_kg, dtype=speed.dtype).reshape(())
    invalid = (
        ~jnp.isfinite(speed)
        | ~jnp.isfinite(observation)
        | ~jnp.isfinite(body)
        | ~jnp.isfinite(mass)
        | (speed < 0.0)
        | (body <= 0.0)
        | (observation <= body)
        | (mass <= 0.0)
    )
    speed = eqx.error_if(
        speed,
        invalid,
        "Focusing inputs must be finite with observation radius > body radius.",
    )
    return jnp.sqrt(
        speed**2
        + 2.0 * _GRAVITATIONAL_CONSTANT_M3_KG_S2 * mass * (1.0 / body - 1.0 / observation)
    )


def stellar_specific_energy_evidence(
    profile: SmoothStellarRadialProfile,
    states: ArrayLike,
    /,
    *,
    quadrature_order: int = 32,
) -> StellarSpecificEnergyEvidence:
    """Evaluate energy by splitting every interior integral at profile knots."""
    if not isinstance(profile, SmoothStellarRadialProfile):
        raise TypeError("profile must be a SmoothStellarRadialProfile.")
    packed = jnp.asarray(states, dtype=float)
    if packed.shape[-1:] != (6,):
        raise ValueError("states must have a trailing packed axis of 6.")
    if not isinstance(quadrature_order, int) or quadrature_order < 4:
        raise ValueError("quadrature_order must be an integer of at least four.")
    radius = jnp.sqrt(jnp.sum(packed[..., :3] ** 2, axis=-1))
    speed_squared = jnp.sum(packed[..., 3:] ** 2, axis=-1)
    surface_potential = (
        -_GRAVITATIONAL_CONSTANT_M3_KG_S2 * profile.total_mass_kg / profile.radius_m
    )
    high_rule = gauss_legendre_data(quadrature_order)
    low_rule = gauss_legendre_data(max(2, quadrature_order // 2))
    knot_left = profile.radii_m[:-1]
    knot_right = profile.radii_m[1:]

    def integrate_rule(value, nodes, weights):
        left = jnp.maximum(value, knot_left)
        lengths = jnp.maximum(knot_right - left, 0.0)
        coordinates = left[:, None] + 0.5 * lengths[:, None] * (nodes[None, :] + 1.0)
        positions = jnp.stack(
            (
                coordinates,
                jnp.zeros_like(coordinates),
                jnp.zeros_like(coordinates),
            ),
            axis=-1,
        )
        gravity = profile.evaluate(
            positions.reshape((-1, 3))
        ).gravitational_acceleration_m_s2.reshape(coordinates.shape)
        return ein.contract(
            "kq,kq->",
            0.5 * lengths[:, None] * weights[None, :],
            gravity,
        )

    def potential_one(value):
        high_integral = integrate_rule(value, high_rule.nodes, high_rule.weights)
        low_integral = integrate_rule(value, low_rule.nodes, low_rule.weights)
        interior = surface_potential - high_integral
        exterior = -_GRAVITATIONAL_CONSTANT_M3_KG_S2 * profile.total_mass_kg / value
        potential = jnp.where(value <= profile.radius_m, interior, exterior)
        error = jnp.where(
            value <= profile.radius_m,
            jnp.abs(high_integral - low_integral),
            0.0,
        )
        return potential, error

    flat_radius = radius.reshape((-1,))
    flat_potential, flat_error = jax.vmap(potential_one)(flat_radius)
    potential = flat_potential.reshape(radius.shape)
    error = flat_error.reshape(radius.shape)
    energy = 0.5 * speed_squared + potential
    finite = jnp.isfinite(energy) & jnp.isfinite(error)
    sign_qualified = finite & (jnp.abs(energy) > error)
    return StellarSpecificEnergyEvidence(
        energy,
        potential,
        error,
        finite,
        sign_qualified,
    )


def stellar_specific_energy(
    profile: SmoothStellarRadialProfile,
    states: ArrayLike,
    /,
    *,
    quadrature_order: int = 32,
) -> Array:
    """Return knot-split energy; use the evidence API for physical classification."""
    return stellar_specific_energy_evidence(
        profile, states, quadrature_order=quadrature_order
    ).specific_energy_m2_s2


def classify_solar_outcomes(
    specific_energy_m2_s2: ArrayLike,
    scattering_count: ArrayLike,
    observation_outward: ArrayLike,
    surface_outward: ArrayLike,
    numerical_successful: ArrayLike,
    /,
    *,
    energy_sign_qualified: ArrayLike | None = None,
) -> SolarOutcomeClassification:
    """Classify escaped, captured, and reflected paths as exclusive outcomes."""
    energy = jnp.asarray(specific_energy_m2_s2)
    count = jnp.asarray(scattering_count, dtype=jnp.int32)
    observed = jnp.asarray(observation_outward, dtype=bool)
    emerged = jnp.asarray(surface_outward, dtype=bool)
    numerical = jnp.asarray(numerical_successful, dtype=bool)
    sign_qualified = (
        jnp.isfinite(energy)
        if energy_sign_qualified is None
        else jnp.broadcast_to(
            jnp.asarray(energy_sign_qualified, dtype=bool), energy.shape
        )
    )
    escaped = observed & sign_qualified & (energy >= 0.0)
    captured = sign_qualified & (energy < 0.0)
    reflected = emerged & (count > 0) & sign_qualified & (energy >= 0.0)
    outcomes = jnp.where(
        numerical,
        jnp.where(
            escaped,
            int(TransportOutcome.ESCAPED),
            jnp.where(
                captured,
                int(TransportOutcome.CAPTURED),
                jnp.where(
                    reflected,
                    int(TransportOutcome.REFLECTED),
                    int(TransportOutcome.UNRESOLVED),
                ),
            ),
        ),
        int(TransportOutcome.UNRESOLVED),
    ).astype(jnp.int32)
    return SolarOutcomeClassification(
        outcomes,
        jax.nn.one_hot(outcomes, len(TransportOutcome), dtype=bool),
    )


class SolarTransportPlan(StrictModule, NonTrainableState):
    """Bounded stellar transport with analytic exterior and guarded interior paths."""

    profile: SmoothStellarRadialProfile
    scattering: ElasticScatteringTable
    context: AstrodynamicsContext
    process: ProfiledElasticJumpProcess
    observation_radius_m: Array
    observation_sphere: CompiledGeometry
    schedule: HybridSchedulePlan
    maximum_jump_events: int = eqx.field(static=True)
    surface_out_event_index: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: SmoothStellarRadialProfile,
        scattering: ElasticScatteringTable,
        projectile_mass_kg: ArrayLike,
        context: AstrodynamicsContext,
        /,
        *,
        observation_radius_m: float,
        maximum_jump_events: int = 128,
        maximum_guard_events: int = 1,
    ):
        if not isinstance(profile, SmoothStellarRadialProfile) or not isinstance(
            scattering, ElasticScatteringTable
        ):
            raise TypeError("Solar transport requires a stellar profile and tables.")
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        _require_si_physical_inertial_context(context)
        if profile.frame_id != context.frame.frame_id:
            raise ValueError("Profile and astrodynamics body-frame identities differ.")
        observation = float(observation_radius_m)
        if not np.isfinite(observation) or observation <= float(profile.radius_m):
            raise ValueError("observation_radius_m must exceed the stellar radius.")
        if not isinstance(maximum_jump_events, int) or maximum_jump_events <= 0:
            raise ValueError("maximum_jump_events must be positive.")
        if (
            not isinstance(maximum_guard_events, int)
            or isinstance(maximum_guard_events, bool)
            or maximum_guard_events != 1
        ):
            raise ValueError(
                "Solar transport has exactly one reachable terminal guard event."
            )
        observation_sphere = Sphere(
            (0.0, 0.0, 0.0),
            observation,
            feature_id=f"{profile.frame_id}:observation:{observation:.17g}",
        ).compile()
        specifications = (("surface:out", profile.boundary, 1, True, 20),)
        self.profile = profile
        self.scattering = scattering
        self.context = context
        self.process = ProfiledElasticJumpProcess(profile, scattering, projectile_mass_kg)
        self.observation_radius_m = jnp.asarray(observation)
        self.observation_sphere = observation_sphere
        self.schedule = radial_guard_schedule(
            specifications,
            RadialGravityDrift(profile),
            frame_id=profile.frame_id,
            maximum_events=maximum_guard_events,
        )
        self.maximum_jump_events = maximum_jump_events
        self.surface_out_event_index = 0
        self.plan_id = canonical_fingerprint(
            {
                "kind": "solar-dark-matter-transport",
                "profile": profile.profile_id,
                "process": self.process.process_id,
                "context": context.context_id,
                "observation_radius_m": observation,
                "schedule": self.schedule.plan_id,
                "maximum_jump_events": maximum_jump_events,
            }
        )

    def exterior_propagate(
        self,
        state: BodyFrameTransportState,
        elapsed_s: ArrayLike,
        /,
        *,
        policy: UniversalKeplerPolicy | None = None,
    ) -> ExteriorKeplerTransportResult:
        radius = jnp.sqrt(jnp.sum(state.position_m**2))
        checked_radius = eqx.error_if(
            radius,
            radius < self.profile.radius_m,
            "Analytic Kepler propagation is restricted to the exterior.",
        )
        checked_state = eqx.tree_at(
            lambda value: value.position_m,
            state,
            state.position_m + jnp.zeros_like(state.position_m) * checked_radius,
        )
        return propagate_exterior_kepler(
            checked_state,
            elapsed_s,
            self.profile.total_mass_kg,
            self.context,
            policy=policy,
        )

    def _kepler_packed(self, packed: Array, elapsed_s: Array, /) -> tuple[Array, Array]:
        native = propagate_universal_kepler(
            CartesianOrbitState(packed[:3], packed[3:], self.context),
            elapsed_s,
            _GRAVITATIONAL_CONSTANT_M3_KG_S2 * self.profile.total_mass_kg,
        )
        return native.state.packed(), native.valid

    def _observation_crossing(
        self,
        surface_states: Array,
        available_times_s: Array,
        active: Array,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        def one(surface_state, available, enabled):
            surface_energy = kepler_specific_energy(
                surface_state, self.profile.total_mass_kg
            )
            unbound = enabled & (surface_energy >= 0.0)
            propagation_time = jnp.where(unbound, available, 0.0)
            propagated_state, propagated_valid = self._kepler_packed(
                surface_state, propagation_time
            )
            final_state = jnp.where(unbound, propagated_state, surface_state)
            final_valid = jnp.where(
                enabled & ~unbound,
                jnp.all(jnp.isfinite(surface_state)),
                propagated_valid,
            )
            final_radius = jnp.sqrt(jnp.sum(final_state[:3] ** 2))
            bracketed = (
                unbound & final_valid & (final_radius >= self.observation_radius_m)
            )
            left = jnp.asarray(0.0, dtype=available.dtype)
            right = propagation_time
            for _ in range(56):
                middle = 0.5 * (left + right)
                middle_state, _ = self._kepler_packed(surface_state, middle)
                middle_radius = jnp.sqrt(jnp.sum(middle_state[:3] ** 2))
                before = middle_radius < self.observation_radius_m
                left = jnp.where(before, middle, left)
                right = jnp.where(before, right, middle)
            crossing_state, crossing_valid = self._kepler_packed(surface_state, right)
            return (
                crossing_state,
                bracketed & crossing_valid,
                final_state,
                final_valid,
            )

        return jax.vmap(one)(surface_states, available_times_s, active)

    def simulate(
        self,
        initial_paths: WeightedSampleBatch,
        poisson: PoissonClockRealization,
        save_times_s: ArrayLike,
        /,
    ) -> SolarTransportResult:
        """Execute guarded transport with exact stochastic replay.

        Samples start on the stellar surface with inward velocity; exterior entry
        preparation uses :meth:`exterior_propagate`. Columns are
        ``(x_m, y_m, z_m, vx_m_s, vy_m_s, vz_m_s)`` in ``profile.frame_id``.
        Surface egress to the observation radius is propagated analytically.
        This orchestration method requires concrete host time endpoints; its
        propagation, rate, mark, and crossing kernels remain JAX traceable.
        Event histories are not pathwise gradients.
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
            raise ValueError("Solar transport requires at least one active path.")
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
        initial_radius = jnp.sqrt(jnp.sum(states[:, :3] ** 2, axis=-1))
        initial_radial_speed = jnp.sum(states[:, :3] * states[:, 3:], axis=-1)
        surface_tolerance = jnp.sqrt(jnp.finfo(states.dtype).eps) * jnp.maximum(
            self.profile.radius_m, 1.0
        )
        if bool(
            jnp.any(jnp.abs(initial_radius - self.profile.radius_m) > surface_tolerance)
            | jnp.any(initial_radial_speed >= 0.0)
        ):
            raise ValueError(
                "Solar interior transport starts on the stellar surface with inward velocity."
            )
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
            raise RuntimeError("Guarded solar solve returned no deterministic evidence.")
        if solution.events.pre_states is None:
            raise RuntimeError("Guarded solar solve returned no collision states.")
        collisions = transport_collision_evidence(
            self.process,
            solution.events.pre_states,
            solution.events.channels,
            solution.events.marks,
            solution.events.valid,
        )
        surface_outward = jnp.any(
            deterministic.valid
            & (deterministic.event_indices == self.surface_out_event_index),
            axis=-1,
        )
        surface_states = deterministic.event_states_after[..., 0, :]
        surface_times = deterministic.event_times[..., 0]
        remaining = jnp.where(
            surface_outward, jnp.maximum(times[-1] - surface_times, 0.0), 0.0
        )
        (
            observation_states,
            observation_valid,
            exterior_final_states,
            exterior_valid,
        ) = self._observation_crossing(surface_states, remaining, surface_outward)
        interior_final_states = solution.states[..., -1, :]
        final_states = jnp.where(
            surface_outward[:, None], exterior_final_states, interior_final_states
        )
        interior_energy_evidence = stellar_specific_energy_evidence(
            self.profile, interior_final_states
        )
        exterior_energy = kepler_specific_energy(
            exterior_final_states, self.profile.total_mass_kg
        )
        energy = jnp.where(
            surface_outward,
            exterior_energy,
            interior_energy_evidence.specific_energy_m2_s2,
        )
        energy_error = jnp.where(
            surface_outward,
            0.0,
            interior_energy_evidence.quadrature_error_m2_s2,
        )
        energy_sign_qualified = jnp.where(
            surface_outward,
            jnp.isfinite(exterior_energy) & (jnp.abs(exterior_energy) > 0.0),
            interior_energy_evidence.sign_qualified,
        )
        finite = (
            jnp.all(jnp.isfinite(solution.states), axis=(-1, -2))
            & jnp.isfinite(energy)
            & (~surface_outward | exterior_valid)
            & (surface_outward | interior_energy_evidence.finite)
        )
        evidence = transport_path_evidence(
            solution.events.status,
            deterministic.event_count,
            deterministic.capacity_exceeded,
            finite,
            solver_successful=solution.numerical_successful,
            collision_invariants_valid=jnp.all(collisions.successful, axis=-1),
            initial_valid=active_paths,
        )
        qualified_observation = observation_valid & evidence.successful
        crossings = spherical_surface_crossings(
            observation_states[:, None, :],
            qualified_observation[:, None],
            initial_paths,
            self.observation_radius_m,
            direction="outward",
        )
        scattering_count = jnp.sum(solution.events.valid, axis=-1, dtype=jnp.int32)
        classification = classify_solar_outcomes(
            energy,
            scattering_count,
            qualified_observation,
            surface_outward,
            evidence.successful,
            energy_sign_qualified=energy_sign_qualified,
        )
        return SolarTransportResult(
            solution,
            collisions,
            crossings,
            classification.outcomes,
            final_states,
            classification.one_hot,
            energy,
            energy_error,
            energy_sign_qualified,
            scattering_count,
            evidence,
            self.plan_id,
        )


__all__ = [
    "ExteriorKeplerTransportResult",
    "StellarSpecificEnergyEvidence",
    "SolarTransportPlan",
    "SolarTransportResult",
    "SolarOutcomeClassification",
    "classify_solar_outcomes",
    "gravitational_focusing_speed",
    "kepler_specific_energy",
    "propagate_exterior_kepler",
    "stellar_specific_energy",
    "stellar_specific_energy_evidence",
]
