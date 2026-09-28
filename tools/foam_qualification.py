"""Qualification campaigns of explicit foam equilibrium and dynamics.

Scenarios cover Laplace-pressure and double-bubble refinement, constrained
volume dynamics, deterministic accepted-film rupture, vortex-sheet bubble
modes, gravity-driven Plateau-border drainage, and pinned Kelvin/Weaire-Phelan
references. The periodic references are recorded as refused because
multiregion surfaces do not represent unwrapped periodic coordinates. Output
is JSON with runtime identity.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax._meshcore import meshcore_available
from phydrax.applications.foams import (
    apply_foam_rupture,
    foam_reference_values,
    FoamDynamicsPlan,
    FoamDynamicsState,
    FoamEquilibriumPlan,
    FoamMaterialPlan,
    FoamRupturePlan,
    PlateauBorderPlan,
    PreparedFoamDynamics,
    PreparedFoamEquilibrium,
    PreparedVortexSheetAir,
    RegionPressureAirPlan,
    StandardDoubleBubble,
    VortexSheetAirPlan,
    VortexSheetAirState,
)
from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceSeed,
    MultiRegionSurfaceState,
    PreparedMultiRegionSurface,
    seed_double_bubble,
    seed_sphere,
)
from phydrax.interfacial_transport import (
    FilmStepStatus,
    prepare_film_sheet_slots,
    SurfaceFilmEvidence,
)
from phydrax.qualification import QualificationRuntimeIdentity


SIGMA = 0.025


def _runtime() -> dict[str, Any]:
    identity = QualificationRuntimeIdentity(
        f"phydrax-{importlib.metadata.version('phydrax')}",
        f"python-{platform.python_version()}-jax-{jax.__version__}-numpy-{np.__version__}",
        jax.default_backend(),
        f"devices-{jax.device_count()}",
        "float64",
    )
    return {**dict(identity.to_record()), "meshcore": meshcore_available()}


def _solve(seed: MultiRegionSurfaceSeed, targets: tuple[float, ...]) -> dict[str, Any]:
    topology = seed.topology(seed.capacity_plan(resource_id="foam-qualification"))
    state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, state)
    material = FoamMaterialPlan.soap_film(topology.region_ids, SIGMA)
    equilibrium = PreparedFoamEquilibrium(FoamEquilibriumPlan(), surface, material, state)
    start = time.perf_counter()
    result = equilibrium.solve(state, equilibrium.parameters(jnp.asarray(targets)))
    evidence = result.evidence
    return {
        "vertices": topology.vertex_count,
        "seconds": time.perf_counter() - start,
        "status": int(evidence.status),
        "kkt_status": int(evidence.kkt_status),
        "stationarity": float(evidence.stationarity_residual),
        "virial_residual": float(evidence.virial_residual),
        "pressures": np.asarray(result.pressures[: len(targets)]).tolist(),
        "energy": float(result.energy),
        "evidence_prepared_id": evidence.prepared_id,
    }


def laplace_refinement() -> dict[str, Any]:
    volume = 4.0 * np.pi / 3.0
    rows: list[dict[str, Any]] = [
        {"subdivisions": level, **_solve(seed_sphere(1.0, subdivisions=level), (volume,))}
        for level in (1, 2, 3)
    ]
    errors = [row["pressures"][0] / (4.0 * SIGMA) - 1.0 for row in rows]
    orders = [float(np.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)]
    successful = all(
        row["status"] == 0 and row["kkt_status"] == 0 for row in rows
    )
    passed = successful and all(order > 1.5 for order in orders)
    return {
        "reference": "Delta p = 4 sigma / R (soap film, gamma = 2 sigma)",
        "rows": rows,
        "relative_errors": errors,
        "observed_orders": orders,
        "status": "passed" if passed else "failed",
        "successful": successful,
        "passed": passed,
    }


def double_bubble_refinement() -> dict[str, Any]:
    reference = StandardDoubleBubble(1.0, 0.8, 2.0 * SIGMA)
    targets = (reference.volume_first, reference.volume_second)
    rows: list[dict[str, Any]] = [
        {
            "ring_points": count,
            **_solve(seed_double_bubble(1.0, 0.8, ring_points=count), targets),
        }
        for count in (12, 18, 24)
    ]
    for row in rows:
        row["pressure_errors"] = [
            value / exact - 1.0
            for value, exact in zip(row["pressures"], reference.pressures, strict=True)
        ]
        row["energy_error"] = row["energy"] / reference.energy - 1.0
    decreasing = all(
        abs(rows[i + 1]["pressure_errors"][1]) < abs(rows[i]["pressure_errors"][1])
        for i in range(len(rows) - 1)
    )
    successful = all(
        row["status"] == 0 and row["kkt_status"] == 0 for row in rows
    )
    passed = decreasing and successful
    return {
        "reference": "standard double bubble (Hutchings-Morgan-Ritore-Ros 2002)",
        "rows": rows,
        "status": "passed" if passed else "failed",
        "successful": successful,
        "passed": passed,
    }


def dynamics_and_rupture_controls() -> dict[str, Any]:
    sphere_seed = seed_sphere(1.0, subdivisions=1)
    sphere_topology = sphere_seed.topology(
        sphere_seed.capacity_plan(resource_id="foam-dynamics-qualification")
    )
    sphere_state = sphere_seed.state(sphere_topology)
    sphere_surface = PreparedMultiRegionSurface(sphere_topology, sphere_state)
    finite_slots = jnp.asarray(sphere_topology.finite_region_indices, dtype=jnp.int32)
    target_volumes = sphere_surface.region_volumes(sphere_state.positions)[finite_slots]
    air = RegionPressureAirPlan.incompressible(target_volumes)
    dynamics_state = FoamDynamicsState(sphere_state)
    prepared = PreparedFoamDynamics(
        FoamDynamicsPlan(
            route="overdamped",
            time_step=1.0e-4,
            friction=5.0,
        ),
        sphere_surface,
        FoamMaterialPlan.soap_film(sphere_topology.region_ids, SIGMA),
        air,
        dynamics_state,
    )
    start = time.perf_counter()
    dynamics = prepared.advance(dynamics_state)
    dynamics_seconds = time.perf_counter() - start

    burst_seed = seed_double_bubble(1.0, 0.8, ring_points=12)
    burst_topology = burst_seed.topology(
        burst_seed.capacity_plan(resource_id="foam-burst-qualification")
    )
    base = burst_seed.state(burst_topology)
    sheet_fields = np.zeros(
        (burst_topology.vertex_capacity, burst_topology.slot_width, 1),
        dtype=np.float64,
    )
    sheet_fields[np.asarray(burst_topology.slot_active), 0] = 1.0e-10
    region_fields = np.zeros((burst_topology.region_capacity, 2), dtype=np.float64)
    region_fields[: burst_topology.region_count, 0] = (1.0, 0.8, 0.0)
    region_fields[: burst_topology.region_count, 1] = (3.0, 2.4, 0.0)
    burst_state = MultiRegionSurfaceState(
        burst_topology,
        base.positions,
        sheet_fields=sheet_fields,
        region_fields=region_fields,
        sheet_field_names=("film_liquid_volume",),
        region_field_names=("gas_amount_mol", "gas_internal_energy_j"),
    )
    finite = np.flatnonzero(np.asarray(burst_topology.region_finite))
    pairs = np.asarray(burst_topology.region_pairs[: burst_topology.region_pair_count])
    pair_index = int(np.flatnonzero(np.all(pairs == np.sort(finite)[None, :], axis=1))[0])
    pair_slots = np.asarray(burst_topology.vertex_pair_slots) == pair_index
    thickness = np.full(
        (burst_topology.vertex_capacity, burst_topology.slot_width),
        1.0e-6,
        dtype=np.float64,
    )
    thickness[pair_slots] = 5.0e-8
    film = SurfaceFilmEvidence(
        liquid_volume_residual_m3=jnp.asarray(0.0),
        boundary_exchange_m3=jnp.asarray(0.0),
        minimum_thickness_m=jnp.asarray(5.0e-8),
        rupture_mask=jnp.asarray(pair_slots),
        energy_change_j=jnp.asarray(-1.0),
        dissipation_guaranteed=jnp.asarray(True),
        positivity_guaranteed=jnp.asarray(True),
        conductance_admissible=jnp.asarray(True),
        nonlinear_status=jnp.asarray(0, dtype=jnp.int32),
        nonlinear_iterations=jnp.asarray(4, dtype=jnp.int32),
        nonlinear_residual_norm=jnp.asarray(1.0e-12),
        converged=jnp.asarray(True),
        finite=jnp.asarray(True),
        geometry_revision=jnp.asarray(1, dtype=jnp.int32),
    )
    burst = apply_foam_rupture(
        FoamRupturePlan(5.0e-8, minimum_trigger_slots=2),
        burst_topology,
        burst_state,
        thickness,
        FilmStepStatus.ACCEPTED,
        film,
        1,
        0.0,
    )
    successful = bool(dynamics.successful and burst.successful)
    passed = (
        successful
        and float(dynamics.evidence.volume_residual) < 1.0e-9
        and abs(float(burst.evidence.liquid_conservation_residual)) < 1.0e-18
        and abs(float(burst.evidence.gas_amount_residual)) < 1.0e-14
        and abs(float(burst.evidence.gas_energy_residual)) < 1.0e-14
    )
    return {
        "dynamics_status": int(dynamics.evidence.status),
        "dynamics_seconds": dynamics_seconds,
        "volume_residual": float(dynamics.evidence.volume_residual),
        "constraint_rank": int(dynamics.evidence.constraint_rank),
        "constraint_condition": float(dynamics.evidence.constraint_condition),
        "energy_nonincreasing": bool(dynamics.evidence.energy_nonincreasing),
        "rupture_status": int(burst.evidence.status),
        "liquid_residual": float(burst.evidence.liquid_conservation_residual),
        "gas_amount_residual": float(burst.evidence.gas_amount_residual),
        "gas_energy_residual": float(burst.evidence.gas_energy_residual),
        "source_regions": burst_topology.region_count,
        "target_regions": burst.topology.region_count,
        "status": "passed" if passed else "failed",
        "successful": successful,
        "passed": bool(passed),
    }


def _prepare_vortex_mode(
    subdivisions: int,
    core_radius_fraction: float,
    step: float,
    amplitude: float,
    /,
) -> tuple[
    PreparedVortexSheetAir,
    FoamDynamicsState,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    float,
    float,
]:
    radius = 0.0237
    density = 1.2
    mode = 2
    seed = seed_sphere(radius, subdivisions=subdivisions)
    topology = seed.topology(
        seed.capacity_plan(resource_id=f"vortex-mode-{subdivisions}")
    )
    unscaled = seed.state(topology)
    finite = jnp.asarray(topology.finite_region_indices, dtype=jnp.int32)
    unscaled_surface = PreparedMultiRegionSurface(topology, unscaled)
    unscaled_volume = unscaled_surface.region_volumes(unscaled.positions)[finite]
    physical_volume = 4.0 * jnp.pi * radius**3 / 3.0
    base_scale = (physical_volume / unscaled_volume[0]) ** (1.0 / 3.0)
    base = unscaled.with_positions(base_scale * unscaled.positions)
    base_surface = PreparedMultiRegionSurface(topology, base)
    targets = base_surface.region_volumes(base.positions)[finite]
    base_radius = jnp.linalg.norm(base.positions, axis=1)
    direction = jnp.where(
        base_radius[:, None] > 0.0,
        base.positions / jnp.maximum(base_radius[:, None], 1.0e-300),
        0.0,
    )
    cosine = direction[:, 2]
    shape = 0.5 * (3.0 * cosine * cosine - 1.0)
    raw_positions = base.positions * (1.0 + amplitude * shape)[:, None]
    raw_volume = base_surface.region_volumes(raw_positions)[finite]
    volume_scale = (targets[0] / raw_volume[0]) ** (1.0 / 3.0)
    positions = raw_positions * volume_scale
    surface_state = base.with_positions(positions)
    surface = PreparedMultiRegionSurface(topology, surface_state)
    mode_weight = jnp.sum(surface.slot_areas(positions), axis=1)
    radial_displacement = jnp.linalg.norm(positions, axis=1) / radius - 1.0
    weight_sum = jnp.sum(jnp.where(topology.vertex_active, mode_weight, 0.0))
    radial_mean = (
        jnp.sum(
            jnp.where(
                topology.vertex_active,
                mode_weight * radial_displacement,
                0.0,
            )
        )
        / weight_sum
    )
    shape_mean = (
        jnp.sum(jnp.where(topology.vertex_active, mode_weight * shape, 0.0)) / weight_sum
    )
    centered_radial = radial_displacement - radial_mean
    centered_shape = shape - shape_mean
    initial_mode_amplitude = jnp.sum(
        jnp.where(
            topology.vertex_active,
            mode_weight * centered_radial * centered_shape,
            0.0,
        )
    ) / jnp.sum(
        jnp.where(
            topology.vertex_active,
            mode_weight * centered_shape**2,
            0.0,
        )
    )
    mode_residual = centered_radial - initial_mode_amplitude * centered_shape
    initial_mode_purity = 1.0 - jnp.sum(
        jnp.where(
            topology.vertex_active,
            mode_weight * mode_residual**2,
            0.0,
        )
    ) / jnp.maximum(
        jnp.sum(
            jnp.where(
                topology.vertex_active,
                mode_weight * centered_radial**2,
                0.0,
            )
        ),
        jnp.finfo(positions.dtype).tiny,
    )
    reference_radius = float((3.0 * targets[0] / (4.0 * jnp.pi)) ** (1.0 / 3.0))
    dynamics_state = FoamDynamicsState(surface_state)
    dynamics = PreparedFoamDynamics(
        FoamDynamicsPlan(
            route="film-inertia",
            time_step=step,
            areal_mass=density * radius,
            volume_tolerance=1.0e-14,
        ),
        surface,
        FoamMaterialPlan.soap_film(topology.region_ids, SIGMA),
        RegionPressureAirPlan.incompressible(targets),
        dynamics_state,
    )
    prepared: PreparedVortexSheetAir = VortexSheetAirPlan(
        air_density=density,
        time_step=step,
        core_radius_fraction=core_radius_fraction,
        fmm_depth=min(4, subdivisions + 2),
        fmm_leaf_capacity=16,
        maximum_fmm_relative_error=0.15,
    ).prepare(dynamics, dynamics_state)
    reference_squared = (
        2.0
        * SIGMA
        * (mode - 1)
        * mode
        * (mode + 1)
        * (mode + 2)
        / (density * reference_radius**3 * (2 * mode + 1))
    )
    return (
        prepared,
        dynamics_state,
        direction,
        base.positions,
        shape,
        positions,
        mode_weight,
        initial_mode_amplitude,
        initial_mode_purity,
        reference_radius,
        reference_squared,
    )


def _vortex_mode_row(
    subdivisions: int,
    core_radius_fraction: float,
    /,
    *,
    step: float = 1.0e-6,
    use_direct: bool = True,
) -> dict[str, Any]:
    radius = 0.0237
    amplitude = 1.0e-4
    (
        prepared,
        dynamics_state,
        direction,
        _base_positions,
        shape,
        positions,
        mode_weight,
        initial_mode_amplitude,
        initial_mode_purity,
        reference_radius,
        reference_squared,
    ) = _prepare_vortex_mode(
        subdivisions,
        core_radius_fraction,
        step,
        amplitude,
    )
    topology = prepared.topology
    dynamics = prepared.dynamics
    initial = prepared.initialize(dynamics_state)
    source = prepared.circulation_source(initial)
    circulation, means = prepared.gauge(initial.circulation + step * source)
    start = time.perf_counter()
    (
        induced_velocity,
        _,
        _,
        core,
        evaluator_successful,
        direct_evaluated,
        relative_error,
        interactions,
        backend,
    ) = prepared._evaluate_velocity(
        circulation,
        positions,
        use_direct=use_direct,
    )
    kinematic = dynamics.constrained_kinematic_step(
        dynamics_state,
        induced_velocity,
        jnp.asarray(step, dtype=positions.dtype),
    )
    seconds = time.perf_counter() - start
    velocity = kinematic.state.surface.velocities
    evaluator_successful = evaluator_successful & kinematic.accepted
    targets = dynamics.air.target_volumes
    if targets is None:
        raise RuntimeError("Vortex mode row lost its target volume.")
    volumes = dynamics.surface.region_volumes(kinematic.state.surface.positions)[
        dynamics.finite_slots
    ]
    volume_residual = jnp.max(
        jnp.abs(volumes - targets)
        / jnp.maximum(
            jnp.abs(targets),
            jnp.finfo(volumes.dtype).tiny,
        )
    )
    radial_acceleration = jnp.sum(velocity * direction, axis=1) / step
    active = topology.vertex_active
    weight_sum = jnp.sum(jnp.where(active, mode_weight, 0.0))
    shape_mean = jnp.sum(jnp.where(active, mode_weight * shape, 0.0)) / weight_sum
    projection_shape = shape - shape_mean
    numerator = jnp.sum(
        jnp.where(
            active,
            mode_weight * radial_acceleration * projection_shape,
            0.0,
        )
    )
    denominator = jnp.sum(jnp.where(active, mode_weight * projection_shape**2, 0.0))
    acceleration_mode = numerator / denominator
    measured_squared = -acceleration_mode / (initial_mode_amplitude * radius)
    successful = bool(evaluator_successful) and bool(jnp.isfinite(relative_error))
    return {
        "subdivisions": subdivisions,
        "vertices": topology.vertex_count,
        "faces": topology.face_count,
        "core_radius_fraction": core_radius_fraction,
        "time_step_s": step,
        "velocity_evaluator": "bounded-direct" if use_direct else "fmm",
        "seconds": seconds,
        "status_code": 0 if successful else 1,
        "measured_frequency_hz": float(
            jnp.sqrt(jnp.maximum(measured_squared, 0.0)) / (2.0 * jnp.pi)
        ),
        "reference_radius_m": reference_radius,
        "reference_frequency_hz": float(np.sqrt(reference_squared) / (2.0 * np.pi)),
        "relative_frequency_error": float(
            jnp.sqrt(jnp.maximum(measured_squared, 0.0) / reference_squared) - 1.0
        ),
        "relative_squared_frequency_error": float(
            measured_squared / reference_squared - 1.0
        ),
        "initial_mode_purity": float(initial_mode_purity),
        "initial_mode_amplitude": float(initial_mode_amplitude),
        "gauge_residual": float(jnp.max(jnp.abs(means))),
        "volume_residual": float(volume_residual),
        "direct_reference_evaluated": bool(direct_evaluated),
        "fmm_relative_l2_error": float(relative_error),
        "fmm_interactions": int(interactions),
        "fmm_geometric_tail_bound": float(backend.geometric_tail_bound),
        "fmm_reference_displacement": float(backend.maximum_reference_displacement),
        "minimum_core_radius": float(jnp.min(core[: prepared.source_count])),
        "maximum_core_radius": float(jnp.max(core[: prepared.source_count])),
        "regularization_policy": prepared.plan.core_policy,
        "successful": successful,
    }


def _harmonic_frequency(
    times: np.ndarray,
    amplitudes: np.ndarray,
    estimated_frequency: float,
    /,
) -> tuple[float, float]:
    transient_end = 0.5 / estimated_frequency
    selected = times >= transient_end
    fit_times = times[selected]
    fit_values = amplitudes[selected]
    centered_time = fit_times - fit_times[0]
    frequencies = np.linspace(
        0.5 * estimated_frequency,
        1.5 * estimated_frequency,
        2001,
        dtype=np.float64,
    )
    residuals = np.empty(frequencies.shape, dtype=np.float64)
    for index, frequency in enumerate(frequencies):
        phase = 2.0 * np.pi * frequency * centered_time
        design = np.column_stack(
            (
                np.cos(phase),
                np.sin(phase),
                np.ones_like(phase),
                centered_time,
            )
        )
        coefficients = np.linalg.lstsq(design, fit_values, rcond=None)[0]
        residuals[index] = np.sum((fit_values - design @ coefficients) ** 2)
    best = int(np.argmin(residuals))
    frequency = frequencies[best]
    if 0 < best < frequencies.size - 1:
        left, center, right = residuals[best - 1 : best + 2]
        curvature = left - 2.0 * center + right
        if curvature > 0.0:
            offset = 0.5 * (left - right) / curvature
            frequency += offset * (frequencies[1] - frequencies[0])
    phase = 2.0 * np.pi * frequency * centered_time
    design = np.column_stack(
        (
            np.cos(phase),
            np.sin(phase),
            np.ones_like(phase),
            centered_time,
        )
    )
    coefficients = np.linalg.lstsq(design, fit_values, rcond=None)[0]
    residual = fit_values - design @ coefficients
    scale = np.sum((fit_values - np.mean(fit_values)) ** 2)
    purity = 1.0 - np.sum(residual**2) / max(scale, np.finfo(np.float64).tiny)
    return float(frequency), float(purity)


def _vortex_transient_row(
    subdivisions: int,
    core_radius_fraction: float,
    step: float,
    estimated_frequency: float,
    /,
) -> dict[str, Any]:
    radius = 0.0237
    amplitude = 1.0e-4
    cycles = 4.0
    steps = int(np.ceil(cycles / (estimated_frequency * step)))
    (
        prepared,
        dynamics_state,
        _,
        base_positions,
        shape,
        _,
        mode_weight,
        initial_mode_amplitude,
        initial_mode_purity,
        reference_radius,
        reference_squared,
    ) = _prepare_vortex_mode(
        subdivisions,
        core_radius_fraction,
        step,
        amplitude,
    )
    topology = prepared.topology
    plus = prepared.initialize(dynamics_state)
    base_positions = jnp.asarray(base_positions)
    minus_positions = base_positions * (1.0 - amplitude * shape)[:, None]
    targets = prepared.dynamics.air.target_volumes
    if targets is None:
        raise RuntimeError("Vortex transient lost its target volume.")
    minus_volume = prepared.dynamics.surface.region_volumes(minus_positions)[
        prepared.dynamics.finite_slots
    ]
    minus_scale = (targets[0] / minus_volume[0]) ** (1.0 / 3.0)
    minus_dynamics = FoamDynamicsState(
        dynamics_state.surface.with_positions(minus_scale * minus_positions)
    )
    minus = prepared.initialize(minus_dynamics)
    active = topology.vertex_active
    weight_sum = jnp.sum(jnp.where(active, mode_weight, 0.0))
    shape_mean = jnp.sum(jnp.where(active, mode_weight * shape, 0.0)) / weight_sum
    projection_shape = shape - shape_mean
    denominator = jnp.sum(
        jnp.where(
            active,
            mode_weight * projection_shape**2,
            0.0,
        )
    )
    step_array = jnp.asarray(step, dtype=dynamics_state.surface.positions.dtype)

    def advance_state(
        state: VortexSheetAirState, /
    ) -> tuple[VortexSheetAirState, Array, Array, Array]:
        source = prepared.circulation_source(state)
        circulation, means = prepared.gauge(state.circulation + step_array * source)
        velocity, successful = prepared.bounded_direct_velocity(
            circulation,
            state.surface.positions,
        )
        kinematic = prepared.dynamics.constrained_kinematic_step(
            state.dynamics,
            velocity,
            step_array,
        )
        accepted = successful & kinematic.accepted
        next_state = VortexSheetAirState(
            kinematic.state,
            jnp.where(accepted, circulation, state.circulation),
            removed_circulation=state.removed_circulation,
        )
        radial = jnp.linalg.norm(next_state.surface.positions, axis=1) / radius - 1.0
        mode_amplitude = (
            jnp.sum(
                jnp.where(
                    active,
                    mode_weight * radial * projection_shape,
                    0.0,
                )
            )
            / denominator
        )
        return next_state, mode_amplitude, accepted, jnp.max(jnp.abs(means))

    def body(
        states: tuple[VortexSheetAirState, VortexSheetAirState],
        _: None,
    ) -> tuple[
        tuple[VortexSheetAirState, VortexSheetAirState],
        tuple[Array, Array, Array],
    ]:
        next_plus, plus_amplitude, plus_accepted, plus_gauge = advance_state(states[0])
        next_minus, minus_amplitude, minus_accepted, minus_gauge = advance_state(
            states[1]
        )
        return (next_plus, next_minus), (
            0.5 * (plus_amplitude - minus_amplitude),
            plus_accepted & minus_accepted,
            jnp.maximum(plus_gauge, minus_gauge),
        )

    @jax.jit
    def run(
        states: tuple[VortexSheetAirState, VortexSheetAirState],
        /,
    ) -> tuple[
        tuple[VortexSheetAirState, VortexSheetAirState],
        tuple[Array, Array, Array],
    ]:
        return jax.lax.scan(body, states, None, length=steps)

    start = time.perf_counter()
    final, outputs = run((plus, minus))
    jax.block_until_ready(outputs)
    seconds = time.perf_counter() - start
    amplitudes = np.asarray(outputs[0], dtype=np.float64)
    times = step * np.arange(1, steps + 1, dtype=np.float64)
    frequency, temporal_purity = _harmonic_frequency(
        times,
        amplitudes,
        estimated_frequency,
    )
    plus_radius = np.asarray(
        jnp.linalg.norm(final[0].surface.positions, axis=1) / radius - 1.0,
        dtype=np.float64,
    )
    minus_radius = np.asarray(
        jnp.linalg.norm(final[1].surface.positions, axis=1) / radius - 1.0,
        dtype=np.float64,
    )
    final_response = 0.5 * (plus_radius - minus_radius)
    weights = np.asarray(mode_weight, dtype=np.float64)
    basis = np.asarray(shape, dtype=np.float64)
    active_host = np.asarray(topology.vertex_active)
    weights = np.where(active_host, weights, 0.0)
    mean = np.sum(weights * final_response) / np.sum(weights)
    centered = final_response - mean
    centered_basis = basis - np.sum(weights * basis) / np.sum(weights)
    coefficient = np.sum(weights * centered * centered_basis) / np.sum(
        weights * centered_basis**2
    )
    spatial_residual = centered - coefficient * centered_basis
    final_mode_purity = 1.0 - np.sum(weights * spatial_residual**2) / max(
        np.sum(weights * centered**2),
        np.finfo(np.float64).tiny,
    )
    plus_volume = prepared.dynamics.surface.region_volumes(final[0].surface.positions)[
        prepared.dynamics.finite_slots
    ]
    minus_volume = prepared.dynamics.surface.region_volumes(final[1].surface.positions)[
        prepared.dynamics.finite_slots
    ]
    volume_residual = jnp.maximum(
        jnp.max(
            jnp.abs(plus_volume - targets)
            / jnp.maximum(jnp.abs(targets), jnp.finfo(plus_volume.dtype).tiny)
        ),
        jnp.max(
            jnp.abs(minus_volume - targets)
            / jnp.maximum(jnp.abs(targets), jnp.finfo(minus_volume.dtype).tiny)
        ),
    )
    reference_frequency = np.sqrt(reference_squared) / (2.0 * np.pi)
    successful = bool(jnp.all(outputs[1])) and bool(jnp.all(jnp.isfinite(outputs[0])))
    return {
        "subdivisions": subdivisions,
        "vertices": topology.vertex_count,
        "faces": topology.face_count,
        "core_radius_fraction": core_radius_fraction,
        "time_step_s": step,
        "steps": steps,
        "elapsed_time_s": steps * step,
        "seconds": seconds,
        "velocity_evaluator": "bounded-direct",
        "mode_isolation": "antisymmetric-plus-minus",
        "measured_frequency_hz": frequency,
        "reference_radius_m": reference_radius,
        "reference_frequency_hz": float(reference_frequency),
        "relative_frequency_error": frequency / reference_frequency - 1.0,
        "initial_mode_purity": float(initial_mode_purity),
        "initial_mode_amplitude": float(initial_mode_amplitude),
        "final_mode_purity": float(final_mode_purity),
        "temporal_fit_purity": temporal_purity,
        "maximum_gauge_residual": float(jnp.max(outputs[2])),
        "volume_residual": float(volume_residual),
        "successful": successful,
    }


def vortex_sheet_bubble_modes() -> dict[str, Any]:
    levels = (0, 1, 2, 3)
    refinement = [
        _vortex_mode_row(
            level,
            0.35,
            step=8.0e-6 / (2**level),
            use_direct=True,
        )
        for level in levels
    ]
    regularization: list[dict[str, Any]] = []
    core_sensitivity: list[dict[str, Any]] = []
    for level in (2, 3):
        central = refinement[level]
        rows = [
            _vortex_mode_row(
                level,
                fraction,
                step=8.0e-6 / (2**level),
                use_direct=True,
            )
            for fraction in (0.25, 0.5)
        ]
        regularization.extend(rows)
        frequencies = [
            central["measured_frequency_hz"],
            *(row["measured_frequency_hz"] for row in rows),
        ]
        core_sensitivity.append(
            {
                "subdivisions": level,
                "relative_frequency_spread": (max(frequencies) - min(frequencies))
                / central["reference_frequency_hz"],
            }
        )
    transients = [
        _vortex_transient_row(
            level,
            0.35,
            8.0e-4 / (2**level),
            refinement[level]["measured_frequency_hz"],
        )
        for level in (0, 1)
    ]
    time_refinement = [
        transients[0],
        _vortex_transient_row(
            0,
            0.35,
            4.0e-4,
            refinement[0]["measured_frequency_hz"],
        ),
    ]
    spatial_errors = [abs(row["relative_frequency_error"]) for row in refinement]
    transient_errors = [abs(row["relative_frequency_error"]) for row in transients]
    spatial_converged = all(
        spatial_errors[index + 1] < spatial_errors[index]
        for index in range(len(spatial_errors) - 1)
    )
    transient_converged = all(
        transient_errors[index + 1] < transient_errors[index]
        for index in range(len(transient_errors) - 1)
    )
    regularization_converged = (
        core_sensitivity[1]["relative_frequency_spread"]
        < core_sensitivity[0]["relative_frequency_spread"]
    )
    time_integration_error = abs(
        time_refinement[0]["measured_frequency_hz"]
        / time_refinement[1]["measured_frequency_hz"]
        - 1.0
    )
    mode_pure = all(
        row["initial_mode_purity"] > 1.0 - 1.0e-12
        and row["final_mode_purity"] > 0.9
        and row["temporal_fit_purity"] > 0.9
        for row in transients
    )
    native_successful = all(
        row["successful"] for row in (*refinement, *regularization)
    ) and all(row["successful"] for row in (*transients, *time_refinement[1:]))
    evidence_ok = native_successful and all(
        row["gauge_residual"] < 1.0e-12
        and row["volume_residual"] < 1.0e-10
        and row["fmm_relative_l2_error"] < 0.05
        for row in (*refinement, *regularization)
    ) and all(
        row["maximum_gauge_residual"] < 1.0e-12
        and row["volume_residual"] < 1.0e-10
        for row in (*transients, *time_refinement[1:])
    )
    target_met = (
        spatial_errors[-1] < 0.05
        and core_sensitivity[-1]["relative_frequency_spread"] < 0.05
        and transient_errors[-1] < 0.05
    )
    passed = (
        spatial_converged
        and transient_converged
        and regularization_converged
        and time_integration_error < 0.01
        and mode_pure
        and evidence_ok
        and target_met
    )
    next_level = 2
    next_step = 8.0e-4 / (2**next_level)
    next_steps = int(
        np.ceil(4.0 / (refinement[next_level]["measured_frequency_hz"] * next_step))
    )
    next_interactions = (
        2
        * refinement[next_level]["vertices"]
        * refinement[next_level]["faces"]
        * next_steps
    )
    return {
        "reference": (
            "Kornek et al., Oscillations of soap bubbles, New J. Phys. 12 "
            "073031 (2010), eqs. (2),(9): gamma_film=2 sigma, "
            "rho_i=rho_o=rho_a, R is fixed by equal-volume rescaling, "
            "omega_l^2 = 2 sigma (l-1)l(l+1)(l+2)/(rho_a R^3(2l+1))"
        ),
        "source_audit": (
            "Da et al. (2015), eqs. (1),(2),(8)-(12): signed edge curvature, "
            "piecewise-linear sheet strength, 1/(4 pi) Biot-Savart; declared "
            "Gaussian core differs from their Rosenhead core"
        ),
        "unit_audit": {
            "circulation_source": "(N / (kg m^-3 m^2)) = m^2 s^-2",
            "integrated_vorticity": "(m s^-1)(m^2) = m^3 s^-1",
            "biot_savart": "(m^3 s^-1)(m)/(m^3) = m s^-1",
            "kornek_frequency_squared": ("(N m^-1)/(kg m^-3 m^3) = s^-2"),
        },
        "initialization": (
            "volume-corrected +/-l=2 pair; antisymmetric response removes "
            "the discrete base-mesh transient before harmonic fitting"
        ),
        "mode": 2,
        "refinement": refinement,
        "regularization": regularization,
        "core_sensitivity": core_sensitivity,
        "transient_frequency": transients,
        "time_refinement": time_refinement,
        "time_integration_relative_error": time_integration_error,
        "error_separation": {
            "finest_spatial_relative_frequency_error": spatial_errors[-1],
            "finest_core_relative_frequency_spread": core_sensitivity[-1][
                "relative_frequency_spread"
            ],
            "finest_fmm_relative_l2_error": refinement[-1]["fmm_relative_l2_error"],
            "time_integration_relative_frequency_error": time_integration_error,
            "finest_extraction_vs_acceleration_relative_difference": abs(
                transients[-1]["measured_frequency_hz"]
                / refinement[1]["measured_frequency_hz"]
                - 1.0
            ),
        },
        "remaining_error_source": (
            "After bounded-direct and time/extraction controls, the remaining "
            "error is the regularized face-centroid sheet quadrature and "
            "piecewise-linear spatial curvature/strength discretization."
        ),
        "spatial_converged": spatial_converged,
        "transient_converged": transient_converged,
        "regularization_converged": regularization_converged,
        "mode_pure": mode_pure,
        "evidence_ok": evidence_ok,
        "target_met": target_met,
        "resource_refusal": None
        if passed
        else {
            "next_subdivisions": next_level,
            "next_transient_time_step_s": next_step,
            "next_transient_steps": next_steps,
            "direct_pair_interactions": next_interactions,
            "reason": (
                "The next post-transient bounded-direct run exceeds the "
                "declared practical CPU pair-evaluation bound."
            ),
        },
        "status": "passed" if passed else "candidate-resource-limited",
        "successful": native_successful,
        "passed": bool(passed),
    }


def _plateau_border_row(ring_points: int, /) -> dict[str, Any]:
    seed = seed_double_bubble(1.0, 0.8, ring_points=ring_points)
    topology = seed.topology(
        seed.capacity_plan(resource_id=f"plateau-border-{ring_points}")
    )
    state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, state)
    film_slots = prepare_film_sheet_slots(surface, state)
    valence = np.sum(np.asarray(topology.edge_faces[: topology.edge_count]) >= 0, axis=1)
    physical_edges = np.asarray(topology.edges[: topology.edge_count])[valence == 3]
    points = np.asarray(state.positions)
    physical_midpoint = 0.5 * (
        points[physical_edges[:, 0]] + points[physical_edges[:, 1]]
    )
    gravity_axis = int(np.argmax(np.ptp(physical_midpoint, axis=0)))
    gravity = np.zeros((3,), dtype=np.float64)
    gravity[gravity_axis] = -9.81
    prepared = PlateauBorderPlan(
        border_edge_capacity=ring_points,
        quad_point_capacity=0,
        density_kg_m3=1000.0,
        viscosity_pa_s=1.0e-3,
        surface_tension_n_m=SIGMA,
        gravity_m_s2=gravity,
        hydraulic_shape_factor=50.0,
        time_step_s=1.0e-5,
        resource_id=f"plateau-border-{ring_points}",
    ).prepare(surface, film_slots, state)
    slot_area = surface.slot_areas(state.positions)
    border_state = prepared.initial_state(
        jnp.where(topology.slot_active, 2.0e-6 * slot_area, 0.0),
        jnp.where(topology.slot_active, 1.0e-7 * slot_area, 0.0),
        1.0e-8,
        border_surfactant_concentration_mol_m3=2.0e-4,
    )
    start = time.perf_counter()
    result = prepared.step(border_state, prepared.zero_boundary_flux())
    seconds = time.perf_counter() - start
    change = np.asarray(result.state.border_liquid_m3 - border_state.border_liquid_m3)
    order = np.argsort(physical_midpoint[:, gravity_axis], kind="stable")
    lower_gain = float(np.mean(change[order[: ring_points // 2]]))
    upper_gain = float(np.mean(change[order[ring_points // 2 :]]))
    liquid_residual = float(result.evidence.liquid_conservation_residual_m3)
    surfactant_residual = float(result.evidence.surfactant_conservation_residual_mol)
    liquid_relative = abs(liquid_residual) / float(border_state.total_liquid_m3())
    surfactant_relative = abs(surfactant_residual) / float(
        border_state.total_surfactant_mol()
    )
    return {
        "ring_points": ring_points,
        "border_edges": prepared.border_count,
        "sheet_boundary_routes": film_slots.evidence.boundary_route_count,
        "seconds": seconds,
        "status_code": int(result.evidence.status),
        "gravity_axis": gravity_axis,
        "lower_mean_volume_change_m3": lower_gain,
        "upper_mean_volume_change_m3": upper_gain,
        "liquid_residual_m3": liquid_residual,
        "liquid_relative_residual": liquid_relative,
        "surfactant_residual_mol": surfactant_residual,
        "surfactant_relative_residual": surfactant_relative,
        "maximum_courant_number": float(result.evidence.maximum_courant_number),
        "gravity_drainage": lower_gain > upper_gain,
    }


def plateau_border_drainage() -> dict[str, Any]:
    rows = [_plateau_border_row(count) for count in (6, 12, 24)]
    successful = all(row["status_code"] == 0 for row in rows)
    passed = successful and all(
        row["gravity_drainage"]
        and row["liquid_relative_residual"] < 1.0e-14
        and row["surfactant_relative_residual"] < 1.0e-14
        for row in rows
    )
    return {
        "reference": (
            "Koehler-Hilgenfeldt-Stone, Langmuir 16, 6327 (2000), "
            "gravity/viscous Plateau-border network balance"
        ),
        "rows": rows,
        "status": "passed" if passed else "failed",
        "successful": successful,
        "passed": passed,
    }


def periodic_references() -> dict[str, Any]:
    values = {
        value.name: (value.value, value.source) for value in foam_reference_values()
    }
    return {
        "kelvin": values["kelvin.relaxed-cell-area"],
        "weaire_phelan": values["weaire-phelan.relaxed-cell-area"],
        "flat_truncated_octahedron": values["kelvin.flat-truncated-octahedron-area"],
        "status": "refused",
        "detail": "periodic foam volumes need unwrapped coordinates (nonclaim)",
        "successful": False,
        "passed": False,
    }


_SCENARIOS: dict[str, Callable[[], dict[str, Any]]] = {
    "laplace-refinement": laplace_refinement,
    "double-bubble-refinement": double_bubble_refinement,
    "dynamics-and-rupture-controls": dynamics_and_rupture_controls,
    "periodic-references": periodic_references,
    "vortex-sheet-bubble-modes": vortex_sheet_bubble_modes,
    "plateau-border-drainage": plateau_border_drainage,
}


def run_qualification(names: tuple[str, ...], /) -> dict[str, Any]:
    scenarios = {name: _SCENARIOS[name]() for name in names}
    return {
        "runtime": _runtime(),
        "scenarios": scenarios,
        "successful": bool(scenarios)
        and all(
            record["successful"] is True and record["passed"] is True
            for record in scenarios.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=(*_SCENARIOS, "all"), default="all")
    arguments = parser.parse_args()
    names = tuple(_SCENARIOS) if arguments.scenario == "all" else (arguments.scenario,)
    report = run_qualification(names)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
