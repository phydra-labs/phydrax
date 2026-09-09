# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Small scientific qualification; no climatology or forecast-skill claims.

Run with JAX_ENABLE_X64=1. --refine compares dt, dt/2, dt/4 at equal duration.
The JSON records measured values and fails on any rejected step. It does not
turn a short stable run into a Held--Suarez or aquaplanet validation claim.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from phydrax.applications.atmosphere._global import (
    GlobalPrimitiveEquationPlan,
    read_global_atmosphere_checkpoint,
    write_global_atmosphere_checkpoint,
)
from phydrax.applications.atmosphere._moist import MoistThermodynamicPlan
from phydrax.applications.atmosphere._processes import GlobalAtmosphereProcesses
from phydrax.applications.geophysics._vertical import HybridPressureCoordinate
from phydrax.discretization.spectral._spherical import SphericalSpectralPlan


CASES = (
    "rest",
    "wave",
    "solid_rotation",
    "baroclinic",
    "held_suarez",
    "moist",
    "aquaplanet",
)


def make_case(case, *, bandlimit=4, levels=2, dt=20.0):
    space = SphericalSpectralPlan(bandlimit, sampling="gl").prepare(radius=6.371e6)
    sigma = np.linspace(0.0, 1.0, levels + 1)
    vertical = HybridPressureCoordinate(0.1 * (1.0 - sigma), sigma)
    processes = GlobalAtmosphereProcesses()
    if case == "held_suarez":
        processes = GlobalAtmosphereProcesses(held_suarez=True, cadence=3)
    if case in ("moist", "aquaplanet"):
        processes = GlobalAtmosphereProcesses(
            thermodynamics=MoistThermodynamicPlan(),
            cadence=3,
            mixing_rate=1e-5,
            radiative_timescale=30 * 86400.0,
            sensible_heat_flux=10.0,
            evaporation_flux=1e-5 if case == "aquaplanet" else 0.0,
        )
    model = GlobalPrimitiveEquationPlan(
        space,
        vertical,
        dt=dt,
        processes=processes,
        rotation_rate=0.0 if case == "wave" else 7.292115e-5,
        filter_rate=1.0 / (10 * 86400)
        if case in ("baroclinic", "held_suarez", "aquaplanet")
        else 0.0,
    ).prepare()
    theta = model.work_space.transform.theta[:, None, None]
    phi = model.work_space.transform.phi[None, :, None]
    layer = jnp.asarray(0.5 * (sigma[:-1] + sigma[1:]))
    if case == "rest":
        initial = model.initialize()
    elif case == "wave":
        initial = model.initialize(
            temperature=288.0 + 1e-3 * jnp.sin(theta) * jnp.cos(phi) * (1.0 - layer)
        )
    elif case == "solid_rotation":
        speed = 20.0
        amplitude = (speed**2 + 2 * model.plan.rotation_rate * space.radius * speed) / (
            2 * model.plan.gas_constant * 288.0
        )
        ps = model.plan.reference_pressure * jnp.exp(
            -amplitude * jnp.cos(theta[..., 0]) ** 2
        )
        initial = model.initialize(east=speed * jnp.sin(theta), surface_pressure=ps)
    elif case in ("baroclinic", "held_suarez"):
        initial = model.initialize(
            temperature=250
            + 35 * layer
            + 12 * jnp.sin(theta) ** 2
            + 0.2 * jnp.sin(theta) ** 2 * jnp.cos(phi),
            east=25 * jnp.sin(theta) * (1 - layer),
        )
    elif case == "moist":
        initial = model.initialize(
            temperature=280.0, vapor=0.007, liquid=0.002, surface_water=1000.0
        )
    else:
        initial = model.initialize(
            temperature=265 + 20 * layer + 8 * jnp.sin(theta) ** 2,
            vapor=0.002 + 0.006 * layer,
            liquid=0.0002,
            east=5 * jnp.sin(theta) * (1 - layer),
            surface_water=1000.0,
        )
    return model, initial


def integrate(model, initial, steps):
    current = initial
    evidence = []
    # Explicit host stepping keeps small qualification runs inspectable and
    # avoids claiming any compilation-throughput benchmark.
    for _ in range(steps):
        result = model.advance(current)
        evidence.append(result.evidence)
        if not bool(result.evidence.accepted):
            break
        current = result.continuation
    return current, evidence


def state_distance(left, right):
    # Dimensionless retained-modal comparison; reservoirs are diagnosed in
    # budgets rather than mixed into a norm with unrelated units.
    scales = (1e-5, 1e-5, 300.0, 1e5)
    a = (left.vorticity, left.divergence, left.temperature, left.surface_pressure)
    b = (right.vorticity, right.divergence, right.temperature, right.surface_pressure)
    return float(
        jnp.sqrt(
            sum(
                jnp.mean(jnp.abs((x - y) / scale) ** 2)
                for x, y, scale in zip(a, b, scales, strict=True)
            )
        )
    )


def linear_wave_error(model, initial, final):
    n = model.levels
    packed = np.concatenate(
        (
            np.asarray(initial.state.divergence),
            np.asarray(initial.state.temperature),
            np.asarray(initial.state.surface_pressure)[..., None],
        ),
        axis=-1,
    )
    duration = float(final.time - initial.time)
    expected = np.stack(
        [
            packed[degree] @ expm(duration * np.asarray(model.fast_matrix[degree])).T
            for degree in range(packed.shape[0])
        ]
    )
    exact = eqx.tree_at(
        lambda s: (s.divergence, s.temperature, s.surface_pressure),
        initial.state,
        (
            jnp.asarray(expected[..., :n]),
            jnp.asarray(expected[..., n : 2 * n]),
            jnp.asarray(expected[..., -1]),
        ),
    )
    return state_distance(final.state, exact)


def qualify(case, *, bandlimit, levels, dt, steps, refine=False):
    model, initial = make_case(case, bandlimit=bandlimit, levels=levels, dt=dt)
    final, evidence = integrate(model, initial, steps)
    before, after = model.inventories(initial.state), model.inventories(final.state)
    view = model.view(final.state)
    result = {
        "case": case,
        "bandlimit": bandlimit,
        "work_bandlimit": model.work_space.layout.bandlimit,
        "levels": levels,
        "dt_seconds": dt,
        "duration_seconds": float(final.time - initial.time),
        "accepted_steps": int(final.accepted_steps),
        "requested_steps": steps,
        "all_accepted": len(evidence) == steps
        and all(bool(e.accepted) for e in evidence),
        "maximum_courant": max(float(e.advective_courant) for e in evidence),
        "maximum_linear_residual": max(float(e.linear_residual) for e in evidence),
        "relative_mass_change": float((after[0] - before[0]) / before[0]),
        "relative_water_change": float(
            (after[1] - before[1]) / jnp.maximum(jnp.abs(before[1]), 1.0)
        ),
        "relative_total_energy_change": float(
            (after[2] - before[2]) / jnp.maximum(jnp.abs(before[2]), 1.0)
        ),
        "energy_residual_joule": float(final.ledger.energy_residual),
        "process_energy_joule": float(final.ledger.process_energy),
        "filter_energy_joule": float(final.ledger.filter_energy),
        "filter_kinetic_energy_joule": float(final.ledger.filter_kinetic_energy),
        "temperature_range_kelvin": [
            float(jnp.min(view.temperature)),
            float(jnp.max(view.temperature)),
        ],
        "maximum_wind_m_per_s": float(jnp.max(jnp.hypot(view.east, view.north))),
        "terrain_rest_acceleration": float(model.terrain_rest_acceleration),
        "state_change": state_distance(initial.state, final.state),
    }
    if evidence and not bool(evidence[-1].accepted):
        rejected = evidence[-1]
        result["rejection"] = {
            "admissible": bool(rejected.admissible),
            "process_successful": bool(rejected.process_successful),
            "linear_solve_successful": bool(rejected.linear_solve_successful),
            "mass_residual": float(rejected.mass_residual),
            "water_residual": float(rejected.water_residual),
            "energy_residual": float(rejected.energy_residual),
        }
    if case == "wave":
        result["linear_wave_error"] = linear_wave_error(model, initial, final)
    if case in ("rest", "solid_rotation"):
        result["analytic_steady_state_error"] = state_distance(initial.state, final.state)
    with tempfile.TemporaryDirectory() as directory:
        path = write_global_atmosphere_checkpoint(
            Path(directory) / "global.zip", model, final
        )
        restored = read_global_atmosphere_checkpoint(path, model, initial)
        # Exercise beyond the next forcing refresh, not just archive equality.
        uninterrupted, un_evidence = integrate(
            model, final, model.plan.processes.cadence + 1
        )
        resumed, re_evidence = integrate(
            model, restored, model.plan.processes.cadence + 1
        )
        result["restart_bitwise_equal"] = all(
            np.array_equal(np.asarray(a), np.asarray(b))
            for a, b in zip(
                jax.tree_util.tree_leaves(uninterrupted),
                jax.tree_util.tree_leaves(resumed),
                strict=True,
            )
        )
        result["restart_steps_accepted"] = all(
            bool(e.accepted) for e in un_evidence + re_evidence
        )
    if refine:
        fine_model, fine_initial = make_case(
            case, bandlimit=bandlimit, levels=levels, dt=dt / 2
        )
        finer_model, finer_initial = make_case(
            case, bandlimit=bandlimit, levels=levels, dt=dt / 4
        )
        fine, fine_evidence = integrate(fine_model, fine_initial, 2 * steps)
        finer, finer_evidence = integrate(finer_model, finer_initial, 4 * steps)
        coarse_error, fine_error = (
            state_distance(final.state, finer.state),
            state_distance(fine.state, finer.state),
        )
        result["refinement"] = {
            "coarse_error": coarse_error,
            "fine_error": fine_error,
            "error_ratio": coarse_error / fine_error if fine_error > 0 else None,
            "all_accepted": all(bool(e.accepted) for e in fine_evidence + finer_evidence),
        }
    result["claim_boundary"] = {
        "rest": "Discrete dry isothermal rest, flat terrain; no general terrain-balance claim.",
        "wave": "Weak gravity wave versus the independent matrix exponential of the linear partition.",
        "solid_rotation": "Analytic gradient-wind isothermal equilibrium, with reported pressure projection error.",
        "baroclinic": "Unbalanced sheared thermal perturbation, not a published balanced benchmark.",
        "held_suarez": "Standard-form Newtonian cooling/Rayleigh drag; short runs do not qualify climate.",
        "moist": "Mixed-phase relaxation, precipitation, radiation and real reservoirs with global PDE.",
        "aquaplanet": "Prescribed flux/grey cooling idealization; no ocean, convection or forecast skill.",
    }[case]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", choices=CASES, default=["rest", "wave", "moist"]
    )
    parser.add_argument("--bandlimit", type=int, default=4)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--dt", type=float, default=20.0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.levels < 1:
        parser.error("steps and levels must be positive")
    if not jax.config.x64_enabled:
        parser.error("Spherical transforms require JAX_ENABLE_X64=1")
    results = [
        qualify(
            case,
            bandlimit=args.bandlimit,
            levels=args.levels,
            dt=args.dt,
            steps=args.steps,
            refine=args.refine,
        )
        for case in args.cases
    ]
    print(json.dumps(results, indent=2, allow_nan=False))
    if not all(
        r["all_accepted"]
        and r["restart_bitwise_equal"]
        and r["restart_steps_accepted"]
        and (not args.refine or r["refinement"]["all_accepted"])
        for r in results
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
