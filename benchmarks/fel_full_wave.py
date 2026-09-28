#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated full-wave (boosted-frame PIC) FEL benchmark.

The seeded small-signal case of the full-wave FEL guide (γ = 20, planar K = 1,
``--periods`` undulator periods, a 22-wavelength one-dimensional flat-top
electron–positron beam) runs at each ``--resolutions`` value of cells per
boosted radiation wavelength, the controlling capacity: it sets the boost-axis
cell count and, through the unaliased PSATD step, the step count. Each case
records host preparation (grid sizing, schedule, species, seed antenna), the
lowering, compilation, and warm execution of one NCI-guarded boosted PIC step,
and the cold (compiling) and warm complete run with its lab-frame extraction,
together with compiler memory, logical bytes of the prepared run and state, and
the physics evidence (status, steady gain, seed amplitude, ledger defect).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

from phydrax import ElectromagneticScaleContract
from phydrax.applications.accelerator import fel, InsertionDeviceField
from phydrax.discretization.pic import PIC_CODE_RELATIVITY
from phydrax.units import CHARGE, UnitDefinition


_GAMMA = 20.0
_DEFLECTION = 1.0
_PERIOD = 1.0
_CURRENT = 4.0e-3
_SEED = 0.3
_WIDTH = math.sqrt(2.0 * math.pi) * 0.01


def _compiler_record(compiled: Any) -> dict[str, object]:
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="The selected JAX backend did not report compiler analysis.",
    )
    record = asdict(evidence)
    record["estimated_device_memory_bytes"] = evidence.estimated_device_memory_bytes
    return record


def _plan(periods: int, resolution: int) -> fel.FELFullWavePlan:
    scale = ElectromagneticScaleContract.code_units(
        PIC_CODE_RELATIVITY.dimensional_scale,
        UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
        gravitational_constant=1,
        speed_of_light=1,
        reduced_planck_constant=1,
        boltzmann_constant=1,
        elementary_charge=1,
        electron_mass=1,
        vacuum_permittivity=1,
        constant_set_id="fel-full-wave-benchmark",
    )
    device = InsertionDeviceField(
        _DEFLECTION * 2.0 * math.pi / _PERIOD,
        _PERIOD,
        periods,
        polarization="planar",
        center=0.0,
        aperture=(0.3, 0.09),
        ramp_periods=0.125,
    )
    lattice = fel.FELUndulatorLattice(
        scale, (fel.FELUndulatorSegment(device),), step_length=_PERIOD / 16.0
    )
    return fel.FELFullWavePlan(
        lattice,
        lattice.resonant_wavelength(_GAMMA) * 1.07,
        boost_lorentz_factor=_GAMMA / math.sqrt(1.0 + 0.5 * _DEFLECTION**2),
        transverse_size=(_WIDTH, _WIDTH),
        cells_per_wavelength=resolution,
        steps_per_period=2 * resolution,
        seed=fel.FELFullWaveSeed(_SEED),
    )


def _case(periods: int, resolution: int, warmup: int, repeats: int) -> dict[str, object]:
    plan = _plan(periods, resolution)
    beam = plan.flat_top_beam(_GAMMA, _CURRENT, wavelengths=22, taper_wavelengths=3.0)
    prepared, preparation_seconds = measure_synchronized(lambda: plan.prepare(beam))
    start, dt, count = prepared.schedule
    boosted = prepared.boosted
    c = plan.speed_of_light
    proper = prepared.initial_proper_velocities
    velocities = proper / jnp.sqrt(1.0 + jnp.sum(proper**2, axis=-1) / (c * c))[:, None]
    # The electron–positron pair beam: both species share the initial phase space.
    state = boosted.initialize(
        (prepared.initial_positions, prepared.initial_positions),
        (velocities, velocities),
        dt,
        time=jnp.asarray(start, dtype=jnp.float64),
        masses=(prepared.masses, prepared.masses),
    )
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(lambda value: boosted.step_detailed(value, dt)).lower(state),
        lambda lowered: lowered.compile(),
    )
    stepped, step_execution = measure_repeated(
        lambda: compiled(state), warmup=warmup, repeats=repeats
    )
    _, cold_seconds = measure_synchronized(prepared.run)
    result, run_execution = measure_repeated(prepared.run, warmup=warmup, repeats=repeats)
    frame = prepared.frame_evidence
    return {
        "cells_per_wavelength": resolution,
        "periods": periods,
        "grid_shape": list(frame.grid_shape),
        "boosted_steps": count,
        "macroparticles_per_species": beam.macroparticle_count,
        "identity": prepared.prepared_id,
        "host_preparation_seconds": preparation_seconds,
        "boosted_step": {
            "compilation": asdict(compilation),
            "execution": step_execution.to_seconds_dict(),
            "compiler": _compiler_record(compiled),
        },
        "run": {
            "cold_seconds": cold_seconds,
            "execution": run_execution.to_seconds_dict(),
        },
        "memory": {
            "prepared_bytes": logical_array_bytes(prepared),
            "state_bytes": logical_array_bytes(state),
        },
        "physics": {
            "boosted_step_successful": bool(stepped.successful),
            "status": int(result.evidence.status),
            "steady_gain": None
            if result.steady_gain is None
            else float(result.steady_gain),
            "seed_amplitude": None
            if result.seed_amplitude is None
            else float(result.seed_amplitude),
            "relative_ledger_defect": float(result.ledger.relative_defect),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=str, default="12,24")
    parser.add_argument("--periods", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    resolutions = tuple(int(value) for value in arguments.resolutions.split(","))
    # The seed antenna sheet needs more than eight cells per emitted wavelength.
    if (
        min(resolutions) <= 8
        or arguments.periods < 1
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError(
            "resolutions above eight cells per wavelength, positive periods, "
            "nonnegative warmup and positive repeats are required"
        )
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "resolutions": list(resolutions),
            "periods": arguments.periods,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
        },
        "cases": [
            _case(arguments.periods, value, arguments.warmup, arguments.repeats)
            for value in resolutions
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
