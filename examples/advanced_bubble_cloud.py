"""Interacting bubble clouds: coupled pulsation, Bjerknes forces, emission and routes.

1. A heterogeneous two-species cloud (clean air bubbles and smooth
   Marmottant–Gompertz coated microbubbles) under a 200 kHz Keller–Miksis drive,
   solved with the implicit dense coupling: the far-field monopole pressure at
   two observers and the time-averaged secondary Bjerknes forces.
2. Bjerknes's sign rule for two widely separated two-bubble pairs below/above resonance.
3. The FMM matrix-free route against the dense route on a 125-bubble lattice.
4. Retarded (finite sound speed) versus incompressible coupling of two bubbles.

Each section reports preparation, lowering, compilation and synchronized
execution separately, so the example also exposes where its smoke time is spent.
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import numpy as np

import phydrax.bubble_dynamics as bd


jax.config.update("jax_enable_x64", True)

AMBIENT = 101325.0
DENSITY = 998.0
SOUND_SPEED = 1481.0


def _prepare(plan: bd.BubbleCloudPlan, /) -> tuple[bd.PreparedBubbleCloud, float]:
    started = time.perf_counter()
    prepared = plan.prepare()
    jax.block_until_ready(prepared)
    return prepared, time.perf_counter() - started


def _solve(
    prepared: bd.PreparedBubbleCloud, /
) -> tuple[bd.BubbleCloudResult, dict[str, float]]:
    started = time.perf_counter()
    lowered = bd.solve_bubble_cloud.lower(prepared)  # ty: ignore[unresolved-attribute]
    lowering = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compilation = time.perf_counter() - started
    started = time.perf_counter()
    result = executable(prepared)
    jax.block_until_ready(result)
    execution = time.perf_counter() - started
    return result, {
        "lowering": lowering,
        "compilation": compilation,
        "execution": execution,
    }


@eqx.filter_jit
def _rates(prepared: bd.PreparedBubbleCloud, /) -> bd.BubbleCloudRates:
    return prepared.rates(prepared.initial_state, 2.5e-7)


def _evaluate_rates(
    prepared: bd.PreparedBubbleCloud, /
) -> tuple[bd.BubbleCloudRates, dict[str, float]]:
    started = time.perf_counter()
    lowered = _rates.lower(prepared)  # ty: ignore[unresolved-attribute]
    lowering = time.perf_counter() - started
    started = time.perf_counter()
    executable = lowered.compile()
    compilation = time.perf_counter() - started
    started = time.perf_counter()
    rates = executable(prepared)
    jax.block_until_ready(rates)
    execution = time.perf_counter() - started
    return rates, {
        "lowering": lowering,
        "compilation": compilation,
        "execution": execution,
    }


def _model(
    interface: bd.AbstractBubbleInterfaceLaw,
    equation: bd.RadialBubbleEquation = "keller_miksis",
    *,
    sound_speed: float = SOUND_SPEED,
) -> bd.RadialBubbleModel:
    return bd.RadialBubbleModel(
        equation,
        bd.PolytropicBubbleGasLaw(1.07),
        bd.NewtonianBubbleLiquidLaw(1.0e-3),
        interface,
        bd.BubbleEnvironment(AMBIENT, 293.15),
        liquid_density=DENSITY,
        liquid_sound_speed=sound_speed,
    )


def _heterogeneous_cloud() -> dict[str, object]:
    rng = np.random.default_rng(3)
    clean_radii = np.array([4.0e-6, 5.0e-6, 6.0e-6])
    coated_radii = np.array([1.5e-6, 2.0e-6, 2.5e-6, 1.8e-6])
    positions = rng.uniform(0.0, 120.0e-6, size=(7, 3))
    positions[:, 2] *= 0.25
    clean = bd.BubbleSpeciesGroup(
        _model(bd.CleanBubbleInterfaceLaw(0.072)),
        clean_radii,
        positions[:3],
        bubble_ids=(0, 1, 2),
    )
    coated = bd.BubbleSpeciesGroup(
        [
            _model(bd.GompertzMarmottantShell(0.5 + 0.1 * member, 0.02, 0.072, 1.0e-8))
            for member in range(4)
        ],
        coated_radii,
        positions[3:],
        bubble_ids=(10, 11, 12, 13),
    )
    frequency = 2.0e5
    # Dense interpolation serves emission; 20 rows per cycle resolve saved averages.
    times = np.linspace(0.0, 30.0e-6, 121)[1:]
    observers = np.array([[5.0e-3, 0.0, 0.0], [0.0, 0.0, 1.0e-2]])
    emission = bd.FarFieldEmissionPlan(observers, np.linspace(15.0e-6, 25.0e-6, 101))
    plan = bd.BubbleCloudPlan(
        (clean, coated),
        bd.HarmonicPressureDrive(3.0e4, 2.0 * np.pi * frequency),
        times,
        emission=emission,
    )
    prepared, preparation = _prepare(plan)
    result, timing = _solve(prepared)
    forces = bd.mean_bjerknes_forces(result, 10.0e-6, 30.0e-6)
    radius = np.asarray(result.trajectory.radius)
    equilibrium = np.concatenate([clean_radii, coated_radii])
    evidence = result.evidence
    if result.emission is None:
        raise RuntimeError("The plan requested far-field emission.")
    return {
        "status": bd.BubbleDynamicsStatus(int(result.status)).name,
        "route": evidence.route,
        "bubble_ids": result.bubble_ids,
        "max_radius_ratio": np.round(np.max(radius / equilibrium, axis=0), 4).tolist(),
        "max_condition_number": float(evidence.maximum_condition_number),
        "min_contact_ratio": float(evidence.minimum_contact_ratio),
        "max_wall_mach": float(evidence.validity.max_wall_mach),
        "emission_successful": bool(result.emission.successful),
        "emission_peak_pa": np.max(
            np.abs(np.asarray(result.emission.pressure)), axis=0
        ).tolist(),
        "emission_distance_ratio": float(result.emission.evidence.minimum_distance_ratio),
        "mean_secondary_bjerknes_n": np.asarray(forces.secondary)[:, 0].tolist(),
        "timing_seconds": {"preparation": preparation, **timing},
    }


def _bjerknes_sign_rule() -> dict[str, object]:
    frequency = 4.0e5
    drive = bd.HarmonicPressureDrive(1.0e4, 2.0 * np.pi * frequency)
    # Ten saved rows per cycle resolve the mean-force sign without oversampling.
    times = np.linspace(0.0, 60.0e-6, 241)[1:]
    distance = 200.0e-6
    pair_offset = 1.0
    radii = np.array([2.0e-6, 3.0e-6, 2.0e-6, 30.0e-6])
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [distance, 0.0, 0.0],
            [pair_offset, 0.0, 0.0],
            [pair_offset + distance, 0.0, 0.0],
        ]
    )
    group = bd.BubbleSpeciesGroup(
        _model(bd.CleanBubbleInterfaceLaw(0.072)),
        radii,
        positions,
        bubble_ids=(0, 1, 2, 3),
    )
    plan = bd.BubbleCloudPlan((group,), drive, times)
    prepared, preparation = _prepare(plan)
    result, timing = _solve(prepared)
    forces = bd.mean_bjerknes_forces(result, 20.0e-6, 60.0e-6)
    secondary = np.asarray(forces.secondary)
    return {
        "status": bd.BubbleDynamicsStatus(int(result.status)).name,
        "both_below": float(secondary[0, 0]),
        "below_above": float(secondary[2, 0]),
        "timing_seconds": {"preparation": preparation, **timing},
    }


def _lattice(count_per_side: int, /) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    axis = np.arange(count_per_side) * 60.0e-6
    lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    points = lattice + rng.uniform(-10.0e-6, 10.0e-6, size=lattice.shape)
    radii = rng.uniform(6.0e-6, 12.0e-6, size=lattice.shape[0])
    return points, radii


def _routes() -> dict[str, object]:
    points, radii = _lattice(5)
    count = radii.shape[0]
    velocities = np.random.default_rng(1).uniform(-1.0, 1.0, size=count)
    group = bd.BubbleSpeciesGroup(
        _model(bd.CleanBubbleInterfaceLaw(0.072), "rayleigh_plesset"),
        radii,
        points,
        bubble_ids=tuple(range(count)),
        initial_radii=1.05 * radii,
        initial_wall_velocities=velocities,
    )
    corrections: dict[str, np.ndarray] = {}
    timings: dict[str, dict[str, float]] = {}
    iterations = 0
    for route in ("dense", "fmm"):
        plan = bd.BubbleCloudPlan(
            (group,),
            bd.HarmonicPressureDrive(2.0e4, 2.0 * np.pi * 1.0e5),
            np.array([1.0e-6]),
            route=route,
            resources=bd.BubbleCloudResourcePolicy(
                maximum_dense_entries=count * count, fmm_order=3
            ),
        )
        prepared, preparation = _prepare(plan)
        rates, timing = _evaluate_rates(prepared)
        corrections[route] = np.asarray(rates.acceleration - rates.uncoupled_acceleration)
        timings[route] = {"preparation": preparation, **timing}
        if route == "fmm":
            iterations = int(rates.coupling_iterations)
    error = np.max(np.abs(corrections["fmm"] - corrections["dense"]))
    return {
        "bubble_count": count,
        "fmm_relative_coupling_error": float(
            error / np.max(np.abs(corrections["dense"]))
        ),
        "fmm_cg_iterations": iterations,
        "timing_seconds": timings,
    }


def _retarded_limit() -> dict[str, object]:
    times = np.linspace(0.0, 6.0e-6, 61)[1:]
    drive = bd.HarmonicPressureDrive(3.0e4, 2.0 * np.pi * 2.0e5)
    sound_speed = SOUND_SPEED
    group = bd.BubbleSpeciesGroup(
        _model(
            bd.CleanBubbleInterfaceLaw(0.072),
            "rayleigh_plesset",
            sound_speed=sound_speed,
        ),
        np.array([10.0e-6, 8.0e-6]),
        np.array([[0.0, 0.0, 0.0], [80.0e-6, 0.0, 0.0]]),
        bubble_ids=(0, 1),
    )
    results: dict[str, bd.BubbleCloudResult] = {}
    timings: dict[str, dict[str, float]] = {}
    for coupling in ("incompressible", "retarded"):
        plan = bd.BubbleCloudPlan((group,), drive, times, coupling=coupling)
        prepared, preparation = _prepare(plan)
        result, timing = _solve(prepared)
        results[coupling] = result
        timings[coupling] = {"preparation": preparation, **timing}
    incompressible = results["incompressible"]
    retarded = results["retarded"]
    return {
        "sound_speed_m_per_s": sound_speed,
        "max_radius_difference_m": float(
            np.max(
                np.abs(
                    np.asarray(
                        retarded.trajectory.radius - incompressible.trajectory.radius
                    )
                )
            )
        ),
        "retardation_ratio": float(retarded.evidence.retardation_ratio),
        "statuses": {
            coupling: bd.BubbleDynamicsStatus(int(result.status)).name
            for coupling, result in results.items()
        },
        "timing_seconds": timings,
    }


def run() -> dict[str, object]:
    return {
        "heterogeneous_cloud": _heterogeneous_cloud(),
        "bjerknes_sign_rule": _bjerknes_sign_rule(),
        "routes": _routes(),
        "retarded_minus_incompressible_max_radius_m": _retarded_limit(),
    }


if __name__ == "__main__":
    print(run())
