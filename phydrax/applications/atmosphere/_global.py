#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-step hydrostatic spectral primitive equations on a rotating sphere.

Pressure interfaces are top-to-bottom, and the last axis is vertical. Nonlinear
terms are evaluated on a padded spherical grid, not in independent columns.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization.spectral._spherical import (
    SphericalSpectralDiscretization,
    SphericalSpectralPlan,
)
from ...discretization.spectral._spherical_vector import PreparedSphericalVectorOperators
from ...ein import contract
from ...enforcement import SimplexProjection
from ...linalg._local_blocks import (
    LocalBlockFactorization,
    prepare_local_block_factorization,
    solve_local_blocks,
)
from ..geophysics._vertical import HybridPressureCoordinate
from ._processes import GlobalAtmosphereProcesses, GlobalHeldForcing, GlobalProcessRates


class GlobalAtmosphereState(StrictModule):
    """Modal prognostics, plus physical surface/environment reservoirs.

    Water inventories are kg/m² per layer of TOTAL moist hydrostatic mass;
    the tuple is vapor, liquid, ice. Surface reservoirs use the padded grid.
    """

    vorticity: Array
    divergence: Array
    temperature: Array
    surface_pressure: Array
    water: tuple[Array, Array, Array]
    surface_water: Array
    surface_energy: Array
    environment_energy: Array


class GlobalAtmosphereLedger(StrictModule):
    mass_residual: Array
    water_residual: Array
    energy_residual: Array
    process_energy: Array
    filter_energy: Array
    filter_kinetic_energy: Array
    water_total_redistribution_mass: Array
    water_phase_repartition_mass: Array
    physical_phase_conversion_mass: Array
    angular_momentum_projection_impulse: Array
    absolute_angular_momentum_projection_impulse: Array


class GlobalAtmosphereContinuation(StrictModule):
    state: GlobalAtmosphereState
    time: Array
    accepted_steps: Array
    rejected_steps: Array
    held_forcing: GlobalHeldForcing
    forcing_age: Array
    ledger: GlobalAtmosphereLedger


class GlobalStepEvidence(StrictModule):
    accepted: Array
    admissible: Array
    process_successful: Array
    linear_solve_successful: Array
    linear_residual: Array
    advective_courant: Array
    mass_residual: Array
    water_residual: Array
    energy_residual: Array
    filter_energy: Array
    filter_kinetic_energy: Array
    filter_angular_momentum: Array
    water_projection_active: Array
    water_total_redistribution_mass: Array
    water_total_redistribution_fraction: Array
    water_phase_repartition_mass: Array
    water_phase_repartition_fraction: Array
    physical_phase_conversion_mass: Array
    maximum_water_projection_energy_residual: Array
    water_projection_successful: Array
    raw_torque_residual_nm: Array
    angular_momentum_projection_torque_nm: Array
    angular_momentum_projection_energy_power_w: Array
    maximum_angular_momentum_projection_acceleration: Array
    angular_momentum_projection_fraction: Array
    angular_momentum_projection_successful: Array
    angular_momentum_projection_evaluated: Array


class GlobalStepResult(StrictModule):
    continuation: GlobalAtmosphereContinuation
    evidence: GlobalStepEvidence


class GlobalAtmosphereView(StrictModule):
    east: Array
    north: Array
    temperature: Array
    surface_pressure: Array
    interfaces: Array
    pressure: Array
    layer_mass: Array
    water: tuple[Array, Array, Array]
    gas_constant: Array
    heat_capacity: Array
    geopotential: Array


class GlobalTendencyEvidence(StrictModule):
    raw_torque_residual_nm: Array
    projection_torque_nm: Array
    corrected_torque_residual_nm: Array
    projection_energy_power_w: Array
    maximum_projection_acceleration_m_per_s2: Array
    projection_fraction: Array
    successful: Array
    evaluated: Array


class GlobalAtmosphereBudgetRates(StrictModule):
    """Independent instantaneous budgets, with dimensional physical scales.

    Angular momentum is atmospheric only: the existing surface/environment
    reservoirs have energy and water, not a resolved velocity or moment of
    inertia. Their momentum exchange is consequently an external torque here.
    """

    energy_residual_w_per_m2: Array
    process_energy_w_per_m2: Array
    angular_momentum_kg_m2_per_s: Array
    angular_momentum_tendency_nm: Array
    terrain_torque_nm: Array
    process_torque_nm: Array
    torque_residual_nm: Array
    raw_torque_residual_nm: Array
    projection_torque_nm: Array
    projection_energy_power_w_per_m2: Array
    maximum_projection_acceleration_m_per_s2: Array
    projection_fraction: Array


def _interfaces(value: Array) -> Array:
    return jnp.concatenate(
        (value[..., :1], 0.5 * (value[..., :-1] + value[..., 1:]), value[..., -1:]),
        axis=-1,
    )


def _add(
    state: GlobalAtmosphereState, rate: GlobalAtmosphereState, scale: ArrayLike
) -> GlobalAtmosphereState:
    return jax.tree_util.tree_map(lambda x, y: x + scale * y, state, rate)


class GlobalPrimitiveEquationPlan(StrictModule):
    """Resolved horizontal space, hybrid pressure layers, and fixed-step policy.

    Terrain is geopotential in m²/s² on the *retained* scalar space's grid.
    Its discrete isothermal-rest acceleration must pass ``terrain_rest_tolerance``.
    Filtering is explicit exponential hyperdiffusion of wind and temperature.
    Moist runs may opt into joint nonnegative phase projection that preserves the
    represented total-water field and moist enthalpy while measuring phase
    repartition separately. Energy-neutral angular-momentum projection retains
    its raw correction evidence. Unresolved input still rejects rather than
    being repaired during initialization.
    """

    space: SphericalSpectralDiscretization
    vertical: HybridPressureCoordinate
    processes: GlobalAtmosphereProcesses
    terrain: Array
    dt: float = eqx.field(static=True)
    rotation_rate: float = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
    gas_constant: float = eqx.field(static=True)
    heat_capacity: float = eqx.field(static=True)
    reference_temperature: float = eqx.field(static=True)
    reference_pressure: float = eqx.field(static=True)
    padding_factor: float = eqx.field(static=True)
    filter_rate: float = eqx.field(static=True)
    filter_order: int = eqx.field(static=True)
    terrain_rest_tolerance: float = eqx.field(static=True)
    budget_tolerance: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    maximum_courant: float = eqx.field(static=True)
    water_limiter: str = eqx.field(static=True)
    maximum_water_phase_repartition_fraction: float = eqx.field(static=True)
    water_projection_iterations: int = eqx.field(static=True)
    water_projection_tolerance: float = eqx.field(static=True)
    angular_momentum_projection: str = eqx.field(static=True)
    maximum_angular_momentum_projection_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        space,
        vertical,
        /,
        *,
        dt=60.0,
        rotation_rate=7.292115e-5,
        gravity=9.80665,
        gas_constant=287.05,
        heat_capacity=1004.0,
        reference_temperature=288.0,
        reference_pressure=100000.0,
        terrain=0.0,
        processes=None,
        padding_factor=1.5,
        filter_rate=0.0,
        filter_order=4,
        terrain_rest_tolerance=1e-6,
        budget_tolerance=1e-9,
        energy_tolerance=1e-4,
        maximum_courant=0.8,
        water_limiter="reject",
        maximum_water_phase_repartition_fraction=1e-4,
        water_projection_iterations=4,
        water_projection_tolerance=1e-12,
        angular_momentum_projection="none",
        maximum_angular_momentum_projection_fraction=1e-5,
    ):
        if (
            not isinstance(space, SphericalSpectralDiscretization)
            or space.layout.spin != 0
            or not space.layout.reality
        ):
            raise TypeError(
                "Global atmosphere requires a prepared real scalar spherical space."
            )
        if not isinstance(vertical, HybridPressureCoordinate):
            raise TypeError("vertical must be a HybridPressureCoordinate.")
        values = (
            dt,
            gravity,
            gas_constant,
            heat_capacity,
            reference_temperature,
            reference_pressure,
            terrain_rest_tolerance,
            budget_tolerance,
            energy_tolerance,
            maximum_courant,
            maximum_water_phase_repartition_fraction,
            water_projection_tolerance,
            maximum_angular_momentum_projection_fraction,
        )
        if (
            any(not np.isfinite(v) or v <= 0 for v in values)
            or heat_capacity <= gas_constant
        ):
            raise ValueError(
                "Atmospheric thermodynamic and numerical scales must be positive and cp > R."
            )
        if (
            not np.isfinite(rotation_rate)
            or not np.isfinite(filter_rate)
            or filter_rate < 0
            or filter_order < 1
        ):
            raise ValueError("Invalid rotation or filter parameters.")
        if not np.isfinite(padding_factor) or padding_factor < 1.5:
            raise ValueError(
                "Nonlinear reference dynamics require padding_factor >= 1.5."
            )
        a, b = np.asarray(vertical.a), np.asarray(vertical.b)
        if b[0] != 0 or b[-1] != 1 or a[-1] != 0:
            raise ValueError(
                "Atmosphere requires fixed-pressure top and bottom interface equal to surface pressure."
            )
        interfaces = np.asarray(vertical.interfaces(reference_pressure))
        if interfaces[0] <= 0 or np.any(np.diff(interfaces) <= 0):
            raise ValueError(
                "Atmosphere requires strictly positive, increasing pressure interfaces (no zero-pressure log)."
            )
        self.space, self.vertical = space, vertical
        self.processes = GlobalAtmosphereProcesses() if processes is None else processes
        if not isinstance(self.processes, GlobalAtmosphereProcesses):
            raise TypeError("processes must be GlobalAtmosphereProcesses.")
        if water_limiter not in ("reject", "conservative"):
            raise ValueError("water_limiter must be 'reject' or 'conservative'.")
        if water_limiter == "conservative" and self.processes.thermodynamics is None:
            raise ValueError("Conservative water limiting requires moist thermodynamics.")
        if (
            int(water_projection_iterations) != water_projection_iterations
            or water_projection_iterations < 1
        ):
            raise ValueError("water_projection_iterations must be a positive integer.")
        if angular_momentum_projection not in ("none", "energy-neutral"):
            raise ValueError(
                "angular_momentum_projection must be 'none' or 'energy-neutral'."
            )
        self.processes.admit_step(float(dt))
        self.terrain = jnp.broadcast_to(
            jnp.asarray(terrain, dtype=jnp.float64), space.sample_shape
        )
        if not np.all(np.isfinite(np.asarray(self.terrain))):
            raise ValueError("Terrain geopotential must be finite.")
        self.dt, self.rotation_rate, self.gravity = (
            float(dt),
            float(rotation_rate),
            float(gravity),
        )
        self.gas_constant, self.heat_capacity = float(gas_constant), float(heat_capacity)
        self.reference_temperature, self.reference_pressure = (
            float(reference_temperature),
            float(reference_pressure),
        )
        self.padding_factor, self.filter_rate, self.filter_order = (
            float(padding_factor),
            float(filter_rate),
            int(filter_order),
        )
        self.terrain_rest_tolerance = float(terrain_rest_tolerance)
        self.budget_tolerance, self.energy_tolerance = (
            float(budget_tolerance),
            float(energy_tolerance),
        )
        self.maximum_courant = float(maximum_courant)
        self.water_limiter = water_limiter
        self.maximum_water_phase_repartition_fraction = float(
            maximum_water_phase_repartition_fraction
        )
        self.water_projection_iterations = int(water_projection_iterations)
        self.water_projection_tolerance = float(water_projection_tolerance)
        self.angular_momentum_projection = angular_momentum_projection
        self.maximum_angular_momentum_projection_fraction = float(
            maximum_angular_momentum_projection_fraction
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "global-hydrostatic-primitive-equations",
                "space": space.prepared_id,
                "vertical": vertical.coordinate_id,
                "processes": self.processes.process_id,
                "terrain": array_tree_fingerprint(np.asarray(self.terrain)),
                "scales": list(values),
                "rotation": rotation_rate,
                "padding": padding_factor,
                "filter": [filter_rate, filter_order],
                "water_limiter": [
                    self.water_limiter,
                    self.maximum_water_phase_repartition_fraction,
                    self.water_projection_iterations,
                    self.water_projection_tolerance,
                ],
                "angular_momentum_projection": [
                    self.angular_momentum_projection,
                    self.maximum_angular_momentum_projection_fraction,
                ],
            }
        )

    def prepare(self) -> "PreparedGlobalAtmosphere":
        return PreparedGlobalAtmosphere(self)


class PreparedGlobalAtmosphere(StrictModule):
    plan: GlobalPrimitiveEquationPlan
    work_space: SphericalSpectralDiscretization
    vectors: PreparedSphericalVectorOperators
    terrain: Array
    coriolis: Array
    fast_matrix: Array
    fast_factorization: LocalBlockFactorization
    terrain_east: Array
    filter_multiplier: Array
    constant_mode: Array
    phase_simplex: SimplexProjection
    terrain_rest_acceleration: Array
    levels: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: GlobalPrimitiveEquationPlan):
        self.plan = plan
        original = plan.space.plan
        limit = int(np.ceil(plan.padding_factor * original.bandlimit))
        self.work_space = SphericalSpectralPlan(
            limit,
            sampling=original.sampling,
            execution=original.execution,
            precision=original.precision,
            max_precompute_bytes=original.max_precompute_bytes,
            max_dense_operator_bytes=original.max_dense_operator_bytes,
            max_explicit_eigenbasis_bytes=original.max_explicit_eigenbasis_bytes,
        ).prepare(radius=plan.space.radius)
        self.vectors = PreparedSphericalVectorOperators(
            self.work_space, mean_policy="project"
        )
        self.phase_simplex = SimplexProjection(tolerance=plan.water_projection_tolerance)
        self.levels = int(plan.vertical.a.size) - 1
        self.constant_mode = self.project(
            jnp.ones(self.work_space.sample_shape, dtype=jnp.float64)
        )
        self.terrain = self.reconstruct(plan.space.project(plan.terrain))
        self.terrain_east = self.gradient(plan.space.project(plan.terrain))[0]
        self.coriolis = (
            2
            * plan.rotation_rate
            * jnp.cos(self.work_space.transform.theta)[:, None, None]
        )
        self.fast_matrix = self._prepare_fast_matrix()
        identity = jnp.eye(2 * self.levels + 1, dtype=jnp.complex128)
        self.fast_factorization = prepare_local_block_factorization(
            identity[None] - 0.5 * plan.dt * self.fast_matrix
        )
        if original.bandlimit == 1:
            # A monopole has no nonconstant mode on which diffusion can act.
            self.filter_multiplier = jnp.ones((1, 1, 1), dtype=jnp.float64)
        else:
            degree = jnp.arange(original.bandlimit, dtype=jnp.float64)
            scale = (
                degree * (degree + 1) / (original.bandlimit * (original.bandlimit - 1))
            )
            self.filter_multiplier = jnp.exp(
                -plan.dt * plan.filter_rate * scale**plan.filter_order
            )[:, None, None]
        # Gate the actual projected rest state, rather than labelling terrain "balanced".
        rest_ps = plan.reference_pressure * jnp.exp(
            -self.terrain / (plan.gas_constant * plan.reference_temperature)
        )
        represented_ps = self.reconstruct(self.project(rest_ps))
        rest_potential = (
            self.terrain
            + plan.gas_constant
            * plan.reference_temperature
            * jnp.log(represented_ps / plan.reference_pressure)
        )
        east, north = self.gradient(self.project(rest_potential))
        self.terrain_rest_acceleration = jnp.max(jnp.hypot(east, north))
        if (
            not bool(plan.vertical.valid(represented_ps).all())
            or float(self.terrain_rest_acceleration) > plan.terrain_rest_tolerance
        ):
            raise ValueError(
                "Terrain fails the discrete hydrostatic-rest gate; refine horizontally or reduce terrain."
            )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-global-atmosphere",
                "plan": plan.plan_id,
                "work_space": self.work_space.prepared_id,
                "partition": "isothermal-degreewise-imex-midpoint",
            }
        )

    def lift(self, coefficients: Array) -> Array:
        old, new = self.plan.space.layout.bandlimit, self.work_space.layout.bandlimit
        result = jnp.zeros(
            self.work_space.coefficient_shape + coefficients.shape[2:],
            dtype=coefficients.dtype,
        )
        return result.at[:old, new - old : new + old - 1].set(coefficients)

    def restrict(self, coefficients: Array) -> Array:
        old, new = self.plan.space.layout.bandlimit, self.work_space.layout.bandlimit
        return coefficients[:old, new - old : new + old - 1]

    def project(self, physical: ArrayLike) -> Array:
        return self.restrict(self.work_space.project(physical))

    def reconstruct(self, coefficients: Array) -> Array:
        return self.work_space.reconstruct(self.lift(coefficients))

    def gradient(self, coefficients: Array) -> tuple[Array, Array]:
        return self.vectors.gradient(self.lift(coefficients))

    def divergence(self, east: Array, north: Array) -> Array:
        return self.restrict(self.vectors.divergence(east, north))

    def curl(self, east: Array, north: Array) -> Array:
        return self.restrict(self.vectors.curl(east, north))

    def _hydrostatic_anomaly(self, thermal: Array, interfaces: Array) -> Array:
        middle = 0.5 * (interfaces[..., :-1] + interfaces[..., 1:])
        whole = thermal * jnp.log(interfaces[..., 1:] / interfaces[..., :-1])
        below = jnp.flip(jnp.cumsum(jnp.flip(whole, axis=-1), axis=-1), axis=-1) - whole
        return below + thermal * jnp.log(interfaces[..., 1:] / middle)

    def view(self, state: GlobalAtmosphereState) -> GlobalAtmosphereView:
        east, north = self.vectors.wind(
            self.lift(state.vorticity), self.lift(state.divergence)
        )
        temperature, ps = (
            self.reconstruct(state.temperature),
            self.reconstruct(state.surface_pressure),
        )
        interfaces = self.plan.vertical.interfaces(ps)
        pressure = 0.5 * (interfaces[..., :-1] + interfaces[..., 1:])
        mass = jnp.diff(interfaces, axis=-1) / self.plan.gravity
        water = tuple(self.reconstruct(q) / mass for q in state.water)
        gas, cp = self.plan.processes.thermodynamic_coefficients(
            water, self.plan.gas_constant, self.plan.heat_capacity
        )
        anomaly = (
            gas * temperature - self.plan.gas_constant * self.plan.reference_temperature
        )
        phi = self.terrain[
            ..., None
        ] + self.plan.gas_constant * self.plan.reference_temperature * jnp.log(
            ps[..., None] / pressure
        )
        phi = phi + self._hydrostatic_anomaly(anomaly, interfaces)
        return GlobalAtmosphereView(
            east, north, temperature, ps, interfaces, pressure, mass, water, gas, cp, phi
        )

    def initialize(
        self,
        *,
        temperature=None,
        surface_pressure=None,
        east=0.0,
        north=0.0,
        vapor=0.0,
        liquid=0.0,
        ice=0.0,
        surface_water=1000.0,
        surface_temperature=290.0,
        time=0.0,
    ) -> GlobalAtmosphereContinuation:
        """Initialize physical data on the padded grid (scalars/vertical profiles broadcast)."""
        shape = self.work_space.sample_shape + (self.levels,)
        temp = jnp.broadcast_to(
            jnp.asarray(
                self.plan.reference_temperature if temperature is None else temperature
            ),
            shape,
        )
        ps = (
            self.plan.reference_pressure
            * jnp.exp(
                -self.terrain / (self.plan.gas_constant * self.plan.reference_temperature)
            )
            if surface_pressure is None
            else jnp.broadcast_to(
                jnp.asarray(surface_pressure), self.work_space.sample_shape
            )
        )
        ps_modal = self.project(ps)
        mass = self.plan.vertical.layer_mass(
            self.reconstruct(ps_modal), gravity=self.plan.gravity
        )
        u, v = (
            jnp.broadcast_to(jnp.asarray(east), shape),
            jnp.broadcast_to(jnp.asarray(north), shape),
        )
        water = tuple(
            self.project(mass * jnp.broadcast_to(jnp.asarray(q), shape))
            for q in (vapor, liquid, ice)
        )
        zero = jnp.zeros(self.work_space.sample_shape, dtype=jnp.float64)
        surface_mass = jnp.broadcast_to(
            jnp.asarray(surface_water, dtype=jnp.float64), zero.shape
        )
        surface_energy = zero
        if self.plan.processes.surface_physics is not None:
            slab = self.plan.processes.surface_physics.initialize(
                jnp.broadcast_to(
                    jnp.asarray(surface_temperature, dtype=jnp.float64), zero.shape
                ),
                surface_mass,
            )
            surface_mass, surface_energy = slab.water_mass, slab.energy
        state = GlobalAtmosphereState(
            self.curl(u, v),
            self.divergence(u, v),
            self.project(temp),
            ps_modal,
            water,
            surface_mass,
            surface_energy,
            zero,
        )
        if not bool(self.admissible(state)):
            raise ValueError(
                "Initial global atmosphere is outside the admitted pressure/temperature/water domain."
            )
        held = self.plan.processes.forcing(
            self.view(state), self.work_space.transform.theta
        )
        scalar = jnp.asarray(0.0)
        return GlobalAtmosphereContinuation(
            state,
            jnp.asarray(time),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            held,
            jnp.asarray(0, jnp.int32),
            GlobalAtmosphereLedger(
                scalar,
                scalar,
                scalar,
                scalar,
                scalar,
                scalar,
                scalar,
                scalar,
                scalar,
                scalar,
                scalar,
            ),
        )

    def admissible(self, state: GlobalAtmosphereState) -> Array:
        view = self.view(state)
        finite = jnp.all(
            jnp.stack(
                [jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(state)]
            )
        )
        water = jnp.stack(view.water)
        composition = (
            jnp.all(water == 0.0)
            if self.plan.processes.thermodynamics is None
            else jnp.asarray(True)
        )
        surface_admissible = (
            jnp.asarray(True)
            if self.plan.processes.surface_physics is None
            else jnp.all(
                self.plan.processes.surface_physics.admissible(
                    state.surface_water,
                    state.surface_energy,
                    self.plan.processes.thermodynamics,
                )
            )
        )
        return (
            finite
            & composition
            & surface_admissible
            & jnp.all(self.plan.vertical.valid(view.surface_pressure))
            & jnp.all((view.temperature > 100.0) & (view.temperature < 500.0))
            & jnp.all(water >= -1e-13)
            & jnp.all(jnp.sum(water, axis=0) < 1.0)
            & jnp.all(state.surface_water >= 0.0)
        )

    def _limit_nonnegative_water(
        self, water: tuple[Array, Array, Array]
    ) -> tuple[tuple[Array, Array, Array], Array, Array, Array, Array]:
        """Project phase composition while preserving represented total water.

        Pointwise projection uses the native simplex map with the original total
        inventory. The affine spectral projection keeps every phase represented
        and restores the original total-water coefficients exactly. Individual
        phase means may change because phase mass is not a conserved inventory;
        total water may not move between represented columns.
        """
        coefficients = jnp.stack(water, axis=-1)

        def reconstruct_phases(value):
            flattened = value.reshape(value.shape[:2] + (-1,))
            return self.reconstruct(flattened).reshape(
                self.work_space.sample_shape + (self.levels, 3)
            )

        def project_phases(value):
            flattened = value.reshape(self.work_space.sample_shape + (-1,))
            return self.project(flattened).reshape(coefficients.shape)

        physical = reconstruct_phases(coefficients)
        total_coefficients = jnp.sum(coefficients, axis=-1)
        total = self.reconstruct(total_coefficients)
        magnitude = jnp.maximum(
            jnp.max(jnp.abs(physical), axis=(0, 1, 3)),
            jnp.max(jnp.abs(total), axis=(0, 1)),
        )
        active = jnp.any(physical < 0)

        def project_simplex(value):
            nonnegative_total = jnp.maximum(total, 0.0)
            normalized = value / jnp.where(total[..., None] > 0, total[..., None], 1.0)
            return self.phase_simplex.apply(normalized) * nonnegative_total[..., None]

        def project_affine(value):
            projected = project_phases(value)
            discrepancy = total_coefficients - jnp.sum(projected, axis=-1)
            projected = projected + discrepancy[..., None] / 3.0
            return projected, reconstruct_phases(projected)

        def solve(_):
            zeros = jnp.zeros_like(physical)

            def iteration(_, carry):
                current, simplex_correction, spectral_correction, _ = carry
                shifted = current + simplex_correction
                simplex = project_simplex(shifted)
                simplex_correction = shifted - simplex
                shifted = simplex + spectral_correction
                projected, represented = project_affine(shifted)
                spectral_correction = shifted - represented
                return (
                    represented,
                    simplex_correction,
                    spectral_correction,
                    projected,
                )

            return jax.lax.fori_loop(
                0,
                self.plan.water_projection_iterations,
                iteration,
                (physical, zeros, zeros, coefficients),
            )

        represented, _, _, projected = jax.lax.cond(
            active,
            solve,
            lambda _: (physical, physical * 0, physical * 0, coefficients),
            operand=None,
        )
        reference = total[..., None] / 3.0
        reference_coefficients = total_coefficients[..., None] / 3.0
        minimum = jnp.min(represented, axis=(0, 1, 3))
        margin = 64 * jnp.finfo(represented.dtype).eps * magnitude
        margin_field = margin[None, None, :, None]
        denominator = jnp.where(reference > represented, reference - represented, 1.0)
        ratio = jnp.where(
            represented < 0,
            jnp.clip(
                jnp.maximum(reference - margin_field, 0.0) / denominator,
                0.0,
                1.0,
            ),
            1.0,
        )
        closure_factor = jnp.min(ratio, axis=(0, 1, 3))
        closed = reference_coefficients + closure_factor[None, None, :, None] * (
            projected - reference_coefficients
        )
        projected = jnp.where(
            (minimum < 0)[None, None, :, None],
            closed,
            projected,
        )
        represented = reconstruct_phases(projected)
        phase_difference = represented - physical
        total_difference = jnp.sum(phase_difference, axis=-1)
        phase_repartition = 0.5 * jnp.sum(
            self.work_space.integral(jnp.abs(phase_difference))
        )
        total_redistribution = 0.5 * jnp.sum(
            self.work_space.integral(jnp.abs(total_difference))
        )
        tolerance = self.plan.water_projection_tolerance * jnp.maximum(magnitude, 1.0)
        total_tolerance = tolerance[None, None, :]
        phase_tolerance = total_tolerance[..., None]
        successful = (
            jnp.all(total >= -total_tolerance)
            & jnp.all(represented >= -phase_tolerance)
            & jnp.all(jnp.abs(jnp.sum(represented, axis=-1) - total) <= total_tolerance)
        )
        limited = tuple(projected[..., index] for index in range(3))
        return (
            limited,
            active,
            total_redistribution,
            phase_repartition,
            successful,
        )

    def limit_water_inventories(
        self, state: GlobalAtmosphereState
    ) -> tuple[
        GlobalAtmosphereState,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ]:
        """Project phase composition and conserve represented water and enthalpy.

        Total-water coefficients remain unchanged. Vapor/liquid/ice repartition
        jointly on the native simplex, so negative trace condensate is not
        redistributed between columns merely to preserve a nonphysical phase
        mean. Temperature retains local moist enthalpy before one uniform
        roundoff correction closes global energy.
        """
        zero = jnp.asarray(0.0)
        if self.plan.water_limiter == "reject":
            return (
                state,
                jnp.asarray(False),
                zero,
                zero,
                zero,
                zero,
                zero,
                jnp.asarray(True),
            )
        (
            water,
            active,
            total_redistribution,
            phase_repartition,
            represented,
        ) = self._limit_nonnegative_water(state.water)
        atmospheric_water = jnp.real(
            sum(
                jnp.sum(self.plan.space.modal_integral(component))
                for component in state.water
            )
        )
        water_scale = jnp.maximum(jnp.abs(atmospheric_water), 1.0)
        total_relative = total_redistribution / water_scale
        phase_relative = phase_repartition / water_scale

        def correct(_):
            thermodynamics = self.plan.processes.thermodynamics
            before = self.view(state)
            replaced = eqx.tree_at(lambda value: value.water, state, water)
            after = self.view(replaced)
            old_enthalpy = thermodynamics.enthalpy(before.temperature, *before.water)
            reference = jnp.asarray(
                thermodynamics.reference_temperature, dtype=old_enthalpy.dtype
            )
            reference_enthalpy = thermodynamics.enthalpy(reference, *after.water)
            heat_capacity = thermodynamics.heat_capacity(
                *after.water, at_constant_pressure=True
            )
            temperature = reference + (old_enthalpy - reference_enthalpy) / heat_capacity
            replaced = eqx.tree_at(
                lambda value: (value.water, value.temperature),
                state,
                (water, self.project(temperature)),
            )
            energy_before = self.inventories(state)[2]
            energy_after = self.inventories(replaced)[2]
            represented_after = self.view(replaced)
            total_capacity = jnp.sum(
                self.work_space.integral(
                    represented_after.layer_mass
                    * thermodynamics.heat_capacity(
                        *represented_after.water, at_constant_pressure=True
                    )
                )
            )
            correction = (energy_before - energy_after) / total_capacity
            corrected_temperature = replaced.temperature + self.project(
                jnp.full_like(represented_after.temperature, correction)
            )
            corrected = eqx.tree_at(
                lambda value: value.temperature,
                replaced,
                corrected_temperature,
            )
            return corrected, self.inventories(corrected)[2] - energy_before

        limited, energy_residual = jax.lax.cond(
            active,
            correct,
            lambda _: (state, jnp.asarray(0.0)),
            operand=None,
        )
        energy_scale = jnp.maximum(jnp.abs(self.inventories(state)[2]), 1.0)
        successful = (
            represented
            & (phase_relative <= self.plan.maximum_water_phase_repartition_fraction)
            & (total_relative <= self.plan.water_projection_tolerance)
            & (jnp.abs(energy_residual) <= self.plan.energy_tolerance * energy_scale)
        )
        return (
            limited,
            active,
            total_redistribution,
            total_relative,
            phase_repartition,
            phase_relative,
            energy_residual,
            successful,
        )

    def inventories(
        self, state: GlobalAtmosphereState
    ) -> tuple[Array, Array, Array, Array]:
        """Closed mass/water/energy and kinetic energy.

        Energy includes the analytic constant-pressure upper-lid work reservoir:
        its addition to atmospheric internal+gravitational energy equals column
        enthalpy+surface-geopotential energy (an irrelevant constant is omitted).
        """
        view = self.view(state)
        integral = self.work_space.integral
        atmospheric_mass = jnp.sum(integral(view.layer_mass))
        water = sum(
            jnp.sum(jnp.real(self.plan.space.modal_integral(q))) for q in state.water
        )
        kinetic = 0.5 * (view.east**2 + view.north**2)
        internal = self.plan.processes.internal_energy(
            view.temperature, view.water, self.plan.gas_constant, self.plan.heat_capacity
        )
        atmospheric_energy = jnp.sum(
            integral(
                view.layer_mass
                * (
                    internal
                    + view.gas_constant * view.temperature
                    + kinetic
                    + self.terrain[..., None]
                )
            )
        )
        return (
            atmospheric_mass + integral(state.surface_water),
            water + integral(state.surface_water),
            atmospheric_energy
            + integral(state.surface_energy + state.environment_energy),
            jnp.sum(integral(view.layer_mass * kinetic)),
        )

    def angular_momentum(self, state: GlobalAtmosphereState) -> Array:
        """Atmospheric axial absolute angular momentum in kg m²/s.

        Uses the owner's shallow-sphere radius and total hydrostatic layer mass;
        it is not a deep-atmosphere or unresolved surface angular momentum.
        """
        view = self.view(state)
        lever = (
            self.plan.space.radius
            * jnp.sin(self.work_space.transform.theta)[:, None, None]
        )
        specific = lever * (view.east + self.plan.rotation_rate * lever)
        return jnp.sum(self.work_space.integral(view.layer_mass * specific))

    def step_energy_flux(self, evidence: GlobalStepEvidence) -> Array:
        """Signed unclosed step energy in W/m², excluding measured filter work."""
        return evidence.energy_residual / (
            4 * jnp.pi * self.plan.space.radius**2 * self.plan.dt
        )

    def _physical_torques(
        self, state: GlobalAtmosphereState, process: GlobalProcessRates
    ) -> tuple[Array, Array]:
        view = self.view(state)
        lever = (
            self.plan.space.radius
            * jnp.sin(self.work_space.transform.theta)[:, None, None]
        )
        specific = lever * (view.east + self.plan.rotation_rate * lever)
        terrain_torque = -self.work_space.integral(
            view.surface_pressure / self.plan.gravity * lever[..., 0] * self.terrain_east
        )
        process_torque = jnp.sum(
            self.work_space.integral(
                view.layer_mass * lever * process.east + specific * process.mass
            )
        )
        return terrain_torque, process_torque

    def _apply_angular_momentum_projection(
        self,
        state: GlobalAtmosphereState,
        rate: GlobalAtmosphereState,
        process: GlobalProcessRates,
        *,
        evaluate: bool,
    ) -> tuple[GlobalAtmosphereState, GlobalTendencyEvidence]:
        if self.plan.angular_momentum_projection == "none" and not evaluate:
            zero = jnp.asarray(0.0)
            return rate, GlobalTendencyEvidence(
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                jnp.asarray(True),
                jnp.asarray(False),
            )
        terrain_torque, process_torque = self._physical_torques(state, process)
        raw_tendency = jax.jvp(self.angular_momentum, (state,), (rate,))[1]
        raw_residual = raw_tendency - terrain_torque - process_torque
        momentum = self.angular_momentum(state)
        torque_scale = jnp.maximum(
            jnp.maximum(
                jnp.abs(momentum) / 86400.0,
                jnp.abs(terrain_torque) + jnp.abs(process_torque),
            ),
            1.0,
        )
        if self.plan.angular_momentum_projection == "none":
            return rate, GlobalTendencyEvidence(
                raw_residual,
                jnp.asarray(0.0),
                raw_residual,
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.asarray(True),
                jnp.asarray(True),
            )

        view = self.view(state)
        lever = (
            self.plan.space.radius
            * jnp.sin(self.work_space.transform.theta)[:, None, None]
        )
        zero = jax.tree.map(jnp.zeros_like, state)
        east_basis = jnp.broadcast_to(lever, view.east.shape)
        wind_basis = eqx.tree_at(
            lambda value: (value.vorticity, value.divergence),
            zero,
            (
                self.curl(east_basis, jnp.zeros_like(east_basis)),
                self.divergence(east_basis, jnp.zeros_like(east_basis)),
            ),
        )
        represented_east, represented_north = self.vectors.wind(
            self.lift(wind_basis.vorticity),
            self.lift(wind_basis.divergence),
        )
        momentum_response = jnp.sum(
            self.work_space.integral(view.layer_mass * lever * represented_east)
        )
        safe_momentum_response = jnp.where(momentum_response != 0, momentum_response, 1.0)
        wind_scale = -raw_residual / safe_momentum_response
        wind_correction = jax.tree.map(lambda value: wind_scale * value, wind_basis)
        wind_power = jnp.sum(
            self.work_space.integral(
                view.layer_mass
                * (
                    view.east * (wind_scale * represented_east)
                    + view.north * (wind_scale * represented_north)
                )
            )
        )
        thermal_basis = eqx.tree_at(
            lambda value: value.temperature,
            zero,
            self.project(jnp.ones_like(view.temperature)),
        )
        thermal_response = jnp.sum(
            self.work_space.integral(view.layer_mass * view.heat_capacity)
        )
        safe_thermal_response = jnp.where(thermal_response != 0, thermal_response, 1.0)
        thermal_scale = -wind_power / safe_thermal_response
        correction = _add(wind_correction, thermal_basis, thermal_scale)
        corrected = _add(rate, correction, 1.0)
        projection_torque = wind_scale * momentum_response
        corrected_residual = raw_residual + projection_torque
        projection_power = wind_power + thermal_scale * thermal_response
        maximum_acceleration = jnp.abs(wind_scale) * jnp.max(
            jnp.hypot(represented_east, represented_north)
        )
        fraction = jnp.abs(projection_torque) / torque_scale
        energy_scale = jnp.maximum(jnp.abs(self.inventories(state)[2]) / 86400.0, 1.0)
        eps = jnp.finfo(view.temperature.dtype).eps
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        raw_residual,
                        projection_torque,
                        corrected_residual,
                        projection_power,
                        maximum_acceleration,
                        fraction,
                    )
                )
            )
        )
        successful = (
            finite
            & (momentum_response != 0)
            & (thermal_response != 0)
            & (fraction <= self.plan.maximum_angular_momentum_projection_fraction)
            & (jnp.abs(corrected_residual) <= 1024 * eps * torque_scale)
            & (jnp.abs(projection_power) <= 1024 * eps * energy_scale)
        )
        return corrected, GlobalTendencyEvidence(
            raw_residual,
            projection_torque,
            corrected_residual,
            projection_power,
            maximum_acceleration,
            fraction,
            successful,
            jnp.asarray(True),
        )

    def budget_rates(
        self, state: GlobalAtmosphereState, held: GlobalHeldForcing
    ) -> GlobalAtmosphereBudgetRates:
        """Compare the actual unfiltered RHS with physical and numerical torques.

        Mountain torque is -integral(ps/g * dPhi_surface/dlongitude). The
        constant-pressure upper lid has zero integrated axial torque. Co-moving
        mass injection/removal carries its local absolute angular momentum;
        prescribed drag and conservative mixing contribute their resolved zonal
        force. Raw discretization residual and the optional explicit constrained
        projection remain distinct.
        """
        rate, process, _, constraint = self.tendency(
            state, held, evaluate_angular_momentum=True
        )
        _, changes = jax.jvp(
            lambda value: (
                self.inventories(value)[2],
                self.angular_momentum(value),
            ),
            (state,),
            (rate,),
        )
        terrain_torque, process_torque = self._physical_torques(state, process)
        area = 4 * jnp.pi * self.plan.space.radius**2
        corrected_torque_residual = changes[1] - terrain_torque - process_torque
        return GlobalAtmosphereBudgetRates(
            changes[0] / area,
            process.energy_power / area,
            self.angular_momentum(state),
            changes[1],
            terrain_torque,
            process_torque,
            corrected_torque_residual,
            constraint.raw_torque_residual_nm,
            constraint.projection_torque_nm,
            constraint.projection_energy_power_w / area,
            constraint.maximum_projection_acceleration_m_per_s2,
            constraint.projection_fraction,
        )

    def _prepare_fast_matrix(self) -> Array:
        n, plan = self.levels, self.plan
        p = plan.vertical.interfaces(plan.reference_pressure)
        dp = jnp.diff(p)
        # The compressional map is the mass-weighted negative hydrostatic adjoint.
        hydro = jax.jacfwd(lambda t: self._hydrostatic_anomaly(plan.gas_constant * t, p))(
            jnp.zeros((n,))
        )
        temperature_from_div = (
            -plan.reference_temperature
            / plan.heat_capacity
            * hydro.T
            * dp[None, :]
            / dp[:, None]
        )
        k2 = plan.space.negative_laplacian_levels()
        matrix = jnp.zeros(
            (plan.space.layout.bandlimit, 2 * n + 1, 2 * n + 1), dtype=jnp.complex128
        )
        matrix = matrix.at[:, :n, n : 2 * n].set(k2[:, None, None] * hydro[None])
        matrix = matrix.at[:, :n, -1].set(
            k2[:, None]
            * plan.gas_constant
            * plan.reference_temperature
            / plan.reference_pressure
        )
        matrix = matrix.at[:, n : 2 * n, :n].set(temperature_from_div[None])
        return matrix.at[:, -1, :n].set(-dp[None])

    def fast_tendency(self, state: GlobalAtmosphereState) -> GlobalAtmosphereState:
        packed = jnp.concatenate(
            (state.divergence, state.temperature, state.surface_pressure[..., None]),
            axis=-1,
        )
        result = contract("lij,lmj->lmi", self.fast_matrix, packed)
        zero = jax.tree_util.tree_map(jnp.zeros_like, state)
        return eqx.tree_at(
            lambda s: (s.divergence, s.temperature, s.surface_pressure),
            zero,
            (
                result[..., : self.levels],
                result[..., self.levels : 2 * self.levels],
                result[..., -1],
            ),
        )

    def _solve_fast(
        self, rhs: GlobalAtmosphereState
    ) -> tuple[GlobalAtmosphereState, Array, Array]:
        packed = jnp.concatenate(
            (rhs.divergence, rhs.temperature, rhs.surface_pressure[..., None]), axis=-1
        )
        solved, failed = solve_local_blocks(
            self.fast_factorization, jnp.swapaxes(packed, 1, 2)
        )
        solved = jnp.swapaxes(solved, 1, 2)
        residual = (
            solved
            - 0.5 * self.plan.dt * contract("lij,lmj->lmi", self.fast_matrix, solved)
            - packed
        )
        relative = jnp.max(jnp.abs(residual)) / jnp.maximum(jnp.max(jnp.abs(packed)), 1.0)
        state = eqx.tree_at(
            lambda s: (s.divergence, s.temperature, s.surface_pressure),
            rhs,
            (
                solved[..., : self.levels],
                solved[..., self.levels : 2 * self.levels],
                solved[..., -1],
            ),
        )
        return state, (~failed) & (relative < 1e-10), relative

    def continuity(
        self, view: GlobalAtmosphereView, mass_source: Array
    ) -> tuple[Array, Array, Array]:
        """Return surface-pressure tendency, interface relative mass flux (Pa/s), omega."""
        dp = view.layer_mass * self.plan.gravity
        horizontal = self.reconstruct(self.divergence(dp * view.east, dp * view.north))
        psdot = -jnp.sum(horizontal, axis=-1) + self.plan.gravity * jnp.sum(
            mass_source, axis=-1
        )
        residual = (
            horizontal
            + jnp.diff(self.plan.vertical.b) * psdot[..., None]
            - self.plan.gravity * mass_source
        )
        flux = jnp.concatenate(
            (jnp.zeros_like(psdot[..., None]), -jnp.cumsum(residual, axis=-1)), axis=-1
        )
        ps_e, ps_n = self.gradient(self.project(view.surface_pressure))
        bmid = 0.5 * (self.plan.vertical.b[:-1] + self.plan.vertical.b[1:])
        omega = 0.5 * (flux[..., :-1] + flux[..., 1:]) + bmid * (
            psdot[..., None] + view.east * ps_e[..., None] + view.north * ps_n[..., None]
        )
        return psdot, flux, omega

    def _pressure_heating(
        self,
        view: GlobalAtmosphereView,
        mass_balance: Array,
        log_pressure_advection: ArrayLike,
    ) -> Array:
        """Compressional work paired with the logarithmic hydrostatic map.

        This finite-layer approximation converges to (R/cp) T omega/p without
        mixing midpoint pressure work with a different log-pressure force.
        ``mass_balance`` is horizontal layer-mass divergence minus sources.
        """
        p = view.interfaces
        whole = jnp.log(p[..., 1:] / p[..., :-1])
        partial = jnp.log(p[..., 1:] / view.pressure)
        above = jnp.cumsum(mass_balance, axis=-1) - mass_balance
        adjoint = whole * above + partial * mass_balance
        thermal = view.gas_constant * view.temperature
        return (
            thermal
            / view.heat_capacity
            * (-adjoint / view.layer_mass + log_pressure_advection)
        )

    def tendency(
        self,
        state: GlobalAtmosphereState,
        held: GlobalHeldForcing,
        *,
        evaluate_angular_momentum: bool = False,
    ) -> tuple[
        GlobalAtmosphereState,
        GlobalProcessRates,
        Array,
        GlobalTendencyEvidence,
    ]:
        view = self.view(state)
        process = self.plan.processes.tendencies(
            view, state.surface_water, state.surface_energy, held
        )
        psdot, flux, _ = self.continuity(view, process.mass)
        dp = self.plan.gravity * view.layer_mass
        anomaly = (
            view.gas_constant * view.temperature
            - self.plan.gas_constant * self.plan.reference_temperature
        )
        base = (
            self.terrain
            + self.plan.gas_constant
            * self.plan.reference_temperature
            * jnp.log(view.surface_pressure / self.plan.reference_pressure)
        )
        potential = (
            base[..., None]
            + self._hydrostatic_anomaly(anomaly, view.interfaces)
            + 0.5 * (view.east**2 + view.north**2)
        )
        ge, gn = self.gradient(self.project(potential))
        pe, pn = self.gradient(self.project(jnp.log(view.pressure)))
        absolute = self.reconstruct(state.vorticity) + self.coriolis

        def vertical_adv(value):
            trace = _interfaces(value)
            return (
                -(
                    flux[..., 1:] * (trace[..., 1:] - value)
                    - flux[..., :-1] * (trace[..., :-1] - value)
                )
                / dp
            )

        udot = (
            absolute * view.north
            - ge
            - anomaly * pe
            + vertical_adv(view.east)
            + process.east
        )
        vdot = (
            -absolute * view.east
            - gn
            - anomaly * pn
            + vertical_adv(view.north)
            + process.north
        )

        def transport_inventory(inventory, specific):
            horizontal = self.reconstruct(
                self.divergence(view.east * inventory, view.north * inventory)
            )
            return (
                -horizontal
                - jnp.diff(flux * _interfaces(specific), axis=-1) / self.plan.gravity
            )

        temp_inventory = transport_inventory(
            view.layer_mass * view.temperature, view.temperature
        )
        massdot = jnp.diff(self.plan.vertical.b) * psdot[..., None] / self.plan.gravity
        pressure_heating = self._pressure_heating(
            view,
            -jnp.diff(flux, axis=-1) / self.plan.gravity - massdot,
            view.east * pe + view.north * pn,
        )
        tdot = (
            (
                temp_inventory
                - view.temperature * massdot
                + view.temperature * process.mass
            )
            / view.layer_mass
            + pressure_heating
            + process.temperature
        )
        waterdot = tuple(
            self.project(transport_inventory(view.layer_mass * q, q) + source)
            for q, source in zip(view.water, process.water, strict=True)
        )
        rate = GlobalAtmosphereState(
            self.curl(udot, vdot),
            self.divergence(udot, vdot),
            self.project(tdot),
            self.project(psdot),
            waterdot,
            process.surface_water,
            process.surface_energy,
            process.environment_energy,
        )
        # Isolate the process contribution INCLUDING the mass-source-induced
        # moving-coordinate flux and compression. Direct heating alone misses
        # pressure work and redistribution when evaporation changes total mass.
        if self.plan.processes.active:
            source_ps = self.plan.gravity * jnp.sum(process.mass, axis=-1)
            source_massdot = (
                jnp.diff(self.plan.vertical.b) * source_ps[..., None] / self.plan.gravity
            )
            source_flux = jnp.concatenate(
                (
                    jnp.zeros_like(source_ps[..., None]),
                    -self.plan.gravity
                    * jnp.cumsum(source_massdot - process.mass, axis=-1),
                ),
                axis=-1,
            )

            def source_adv(value):
                trace = _interfaces(value)
                return (
                    -(
                        source_flux[..., 1:] * (trace[..., 1:] - value)
                        - source_flux[..., :-1] * (trace[..., :-1] - value)
                    )
                    / dp
                )

            source_t = (
                source_adv(view.temperature)
                + self._pressure_heating(view, -process.mass, 0.0)
                + process.temperature
            )
            source_water = tuple(
                self.project(
                    -jnp.diff(source_flux * _interfaces(q), axis=-1) / self.plan.gravity
                    + s
                )
                for q, s in zip(view.water, process.water, strict=True)
            )
            source_u, source_v = (
                source_adv(view.east) + process.east,
                source_adv(view.north) + process.north,
            )
            process_rate = GlobalAtmosphereState(
                self.curl(source_u, source_v),
                self.divergence(source_u, source_v),
                self.project(source_t),
                self.project(source_ps),
                source_water,
                process.surface_water,
                process.surface_energy,
                process.environment_energy,
            )
            power = jax.jvp(lambda s: self.inventories(s)[2], (state,), (process_rate,))[
                1
            ]
        else:
            power = jnp.asarray(0.0)
        courant = self.plan.dt * (
            self.plan.space.layout.bandlimit
            / self.plan.space.radius
            * jnp.max(jnp.hypot(view.east, view.north))
            + jnp.max((jnp.abs(flux[..., :-1]) + jnp.abs(flux[..., 1:])) / dp)
        )
        process = eqx.tree_at(lambda value: value.energy_power, process, power)
        rate, constraint = self._apply_angular_momentum_projection(
            state,
            rate,
            process,
            evaluate=evaluate_angular_momentum,
        )
        return rate, process, courant, constraint

    def advance(self, continuation: GlobalAtmosphereContinuation) -> GlobalStepResult:
        """One IMEX midpoint attempt; rejection preserves state, time, forcing and ledgers."""
        old, dt = continuation.state, self.plan.dt
        refresh = continuation.forcing_age >= self.plan.processes.cadence
        held = jax.lax.cond(
            refresh,
            lambda _: self.plan.processes.forcing(
                self.view(old), self.work_space.transform.theta
            ),
            lambda _: continuation.held_forcing,
            operand=None,
        )
        first, process0, courant0, constraint0 = self.tendency(old, held)
        nonlinear = _add(first, self.fast_tendency(old), -1.0)
        stage, solved, residual = self._solve_fast(_add(old, nonlinear, 0.5 * dt))
        (
            stage,
            stage_projection_active,
            stage_total_redistribution,
            stage_total_relative,
            stage_phase_repartition,
            stage_phase_relative,
            stage_projection_energy,
            stage_projection_successful,
        ) = self.limit_water_inventories(stage)
        midpoint, process1, courant1, constraint1 = self.tendency(stage, held)
        candidate = _add(old, midpoint, dt)
        (
            candidate,
            candidate_projection_active,
            candidate_total_redistribution,
            candidate_total_relative,
            candidate_phase_repartition,
            candidate_phase_relative,
            candidate_projection_energy,
            candidate_projection_successful,
        ) = self.limit_water_inventories(candidate)
        projection_active = stage_projection_active | candidate_projection_active
        total_redistribution = stage_total_redistribution + candidate_total_redistribution
        total_relative = stage_total_relative + candidate_total_relative
        phase_repartition = stage_phase_repartition + candidate_phase_repartition
        phase_relative = stage_phase_relative + candidate_phase_relative
        maximum_projection_energy_residual = jnp.maximum(
            jnp.abs(stage_projection_energy),
            jnp.abs(candidate_projection_energy),
        )
        water_projection_successful = (
            stage_projection_successful & candidate_projection_successful
        )
        before_filter = self.inventories(candidate)
        before_filter_angular = (
            self.angular_momentum(candidate)
            if self.plan.filter_rate > 0
            else jnp.asarray(0.0)
        )
        candidate = eqx.tree_at(
            lambda s: (s.vorticity, s.divergence, s.temperature),
            candidate,
            tuple(
                x * self.filter_multiplier
                for x in (
                    candidate.vorticity,
                    candidate.divergence,
                    candidate.temperature,
                )
            ),
        )
        initial, final = self.inventories(old), self.inventories(candidate)
        filter_energy, filter_kinetic = (
            final[2] - before_filter[2],
            final[3] - before_filter[3],
        )
        filter_angular = (
            self.angular_momentum(candidate) - before_filter_angular
            if self.plan.filter_rate > 0
            else jnp.asarray(0.0)
        )
        mass_residual, water_residual = final[0] - initial[0], final[1] - initial[1]
        process_energy = dt * process1.energy_power
        # All prescribed exchanges have paired real reservoirs; independent
        # external power is zero. The JVP diagnostic is NOT subtracted here.
        energy_residual = final[2] - initial[2] - filter_energy
        admissible = (
            self.admissible(old) & self.admissible(stage) & self.admissible(candidate)
        )
        process_ok = process0.successful & process1.successful
        constraint_ok = constraint0.successful & constraint1.successful
        courant = jnp.maximum(courant0, courant1)
        budgets = (
            (
                jnp.abs(mass_residual)
                <= self.plan.budget_tolerance * jnp.maximum(jnp.abs(initial[0]), 1.0)
            )
            & (
                jnp.abs(water_residual)
                <= self.plan.budget_tolerance * jnp.maximum(jnp.abs(initial[1]), 1.0)
            )
            & (
                jnp.abs(energy_residual)
                <= self.plan.energy_tolerance * jnp.maximum(jnp.abs(initial[2]), 1.0)
            )
        )
        physical_phase_conversion_mass = dt * (
            0.5 * self.work_space.integral(process0.phase_conversion_mass_rate)
            + self.work_space.integral(process1.phase_conversion_mass_rate)
        )
        accepted = (
            admissible
            & process_ok
            & solved
            & budgets
            & water_projection_successful
            & constraint_ok
            & (courant <= self.plan.maximum_courant)
        )
        ledger = GlobalAtmosphereLedger(
            continuation.ledger.mass_residual + mass_residual,
            continuation.ledger.water_residual + water_residual,
            continuation.ledger.energy_residual + energy_residual,
            continuation.ledger.process_energy + process_energy,
            continuation.ledger.filter_energy + filter_energy,
            continuation.ledger.filter_kinetic_energy + filter_kinetic,
            continuation.ledger.water_total_redistribution_mass + total_redistribution,
            continuation.ledger.water_phase_repartition_mass + phase_repartition,
            continuation.ledger.physical_phase_conversion_mass
            + physical_phase_conversion_mass,
            continuation.ledger.angular_momentum_projection_impulse
            + dt * constraint1.projection_torque_nm,
            continuation.ledger.absolute_angular_momentum_projection_impulse
            + dt * jnp.abs(constraint1.projection_torque_nm),
        )
        proposed = GlobalAtmosphereContinuation(
            candidate,
            continuation.time + dt,
            continuation.accepted_steps + 1,
            continuation.rejected_steps,
            held,
            jnp.where(refresh, 1, continuation.forcing_age + 1),
            ledger,
        )
        rejected = eqx.tree_at(
            lambda c: c.rejected_steps, continuation, continuation.rejected_steps + 1
        )
        chosen = jax.lax.cond(
            accepted, lambda _: proposed, lambda _: rejected, operand=None
        )
        evidence = GlobalStepEvidence(
            accepted,
            admissible,
            process_ok,
            solved,
            residual,
            courant,
            mass_residual,
            water_residual,
            energy_residual,
            filter_energy,
            filter_kinetic,
            filter_angular,
            projection_active,
            total_redistribution,
            total_relative,
            phase_repartition,
            phase_relative,
            physical_phase_conversion_mass,
            maximum_projection_energy_residual,
            water_projection_successful,
            constraint1.raw_torque_residual_nm,
            constraint1.projection_torque_nm,
            constraint1.projection_energy_power_w,
            constraint1.maximum_projection_acceleration_m_per_s2,
            constraint1.projection_fraction,
            constraint_ok,
            constraint1.evaluated,
        )
        return GlobalStepResult(chosen, evidence)


def write_global_atmosphere_checkpoint(
    path: str | Path,
    model: PreparedGlobalAtmosphere,
    continuation: GlobalAtmosphereContinuation,
    /,
) -> Path:
    arrays: dict[str, object] = {}
    specification = pack_array_tree("continuation", continuation, arrays)
    return write_array_archive(
        path,
        manifest={
            "kind": "global-atmosphere-checkpoint",
            "model_id": model.prepared_id,
            "physical_parameters": array_tree_fingerprint(
                eqx.filter(model.plan.processes, eqx.is_array)
            ),
            "continuation": specification,
        },
        arrays=arrays,
    )


def read_global_atmosphere_checkpoint(
    path: str | Path,
    model: PreparedGlobalAtmosphere,
    template: GlobalAtmosphereContinuation,
    /,
) -> GlobalAtmosphereContinuation:
    manifest, arrays = read_array_archive(path)
    if (
        manifest.get("kind") != "global-atmosphere-checkpoint"
        or manifest.get("model_id") != model.prepared_id
        or manifest.get("physical_parameters")
        != array_tree_fingerprint(eqx.filter(model.plan.processes, eqx.is_array))
    ):
        raise ValueError("Global atmosphere checkpoint model identity mismatch.")
    return unpack_array_tree(manifest["continuation"], arrays, template)


__all__ = [
    "GlobalPrimitiveEquationPlan",
    "PreparedGlobalAtmosphere",
    "GlobalAtmosphereState",
    "GlobalAtmosphereView",
    "GlobalAtmosphereContinuation",
    "GlobalAtmosphereLedger",
    "GlobalAtmosphereBudgetRates",
    "GlobalStepEvidence",
    "GlobalTendencyEvidence",
    "GlobalStepResult",
    "write_global_atmosphere_checkpoint",
    "read_global_atmosphere_checkpoint",
]
