#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated quasi-cylindrical PSATD Hankel, field-advance, and PIC step cost.

The controlling capacities are the radial count ``N_r`` and the mode count
``M + 1``: the shared-grid Hankel preparation forms and pseudo-inverts
``3(M + 1)`` dense ``N_r × N_r`` matrices on the host, and every field advance
applies them as ``(M + 1)``-batched dense radial contractions around one
axial FFT of ``N_z`` points per mode, radius, and component. Each case records
lowering, compilation, warm execution, compiler memory, and the logical bytes
of the prepared objects and states. The Hankel case reports its rank and
pseudoinverse evidence; the vacuum case advances a resolved pulse (an
azimuthal ``E_θ`` in mode 0 and, with two or more modes, an ``x``-polarized
Gaussian in mode 1, eight axial cells per wavelength) and reports the relative
change of its node-quadrature field energy over one step, which the exact
spectral propagator leaves at the pulse's resolution level; the PIC step
reports its acceptance, Gauss residual relative to the deposited charge, and
deposit↔Gauss pairing defect.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx
from phydrax.solver.maxwell import spectral


_RADIUS = 4.0
_LENGTH = 8.0
_PLASMA_RADIUS = 2.0


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


def _measure(
    function: Any, arguments: tuple[Any, ...], warmup: int, repeats: int
) -> tuple[Any, dict[str, object]]:
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(function).lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(*arguments), warmup=warmup, repeats=repeats
    )
    return result, {
        "compilation": asdict(compilation),
        "execution": execution.to_seconds_dict(),
        "compiler": _compiler_record(compiled),
    }


def _hankel_case(
    grid: phx.discretization.pic.QuasiCylindricalGrid, warmup: int, repeats: int
) -> dict[str, object]:
    plan = phx.discretization.SharedGridHankelPlan(
        grid.radius, grid.radial_count, grid.mode_count
    )
    prepared, preparation = measure_repeated(plan.prepare, warmup=warmup, repeats=repeats)
    evidence = prepared.evidence
    return {
        "preparation": preparation.to_seconds_dict(),
        "memory": {
            "prepared_bytes": logical_array_bytes(prepared),
            "retained_bytes": evidence.retained_bytes,
            "matrix_elements": evidence.matrix_elements,
        },
        "evidence": {
            "successful": bool(evidence.successful),
            "rank_matches_expected": bool(
                jnp.all(evidence.rank == evidence.expected_rank)
            ),
            "maximum_condition": float(jnp.max(evidence.condition)),
            "maximum_pseudoinverse_residual": float(
                jnp.max(evidence.pseudoinverse_residual)
            ),
        },
    }


def _vacuum_case(
    grid: phx.discretization.pic.QuasiCylindricalGrid, warmup: int, repeats: int
) -> dict[str, object]:
    solver = spectral.QuasiCylindricalMaxwellPlan(grid).prepare()
    shape = (grid.mode_count, grid.radial_count, grid.axial_count)
    r = grid.radial_coordinates[:, None]
    z = grid.axial_coordinates[None, :] - 0.5 * (grid.lower + grid.upper)
    width = 0.25 * grid.radius
    envelope = np.exp(-((r / width) ** 2) - (z / (0.125 * grid.length)) ** 2)
    carrier = np.cos(2.0 * np.pi * z / (8.0 * grid.axial_spacing))
    pulse = np.zeros((*shape, 3), dtype=np.complex128)
    # Mode 0: E_θ = (r/w) g, i.e. F_± = ∓(i/2)E_θ; mode 1: E_x = g, i.e. F_+ = g.
    pulse[0, :, :, 0] = -0.5j * (r / width) * envelope * carrier
    pulse[0, :, :, 1] = 0.5j * (r / width) * envelope * carrier
    if grid.mode_count > 1:
        pulse[1, :, :, 0] = envelope * carrier
    field = solver.add_propagating_field(
        solver.field_with_charge(jnp.zeros(shape, dtype=jnp.complex128)), pulse
    )
    source = spectral.QuasiCylindricalSource(
        jnp.zeros((*shape, 3), dtype=jnp.complex128),
        jnp.zeros(shape, dtype=jnp.complex128),
    )
    dt = 0.5 * solver.stable_step
    advance, record = _measure(
        lambda state, current, step: solver.advance(
            jnp.asarray(0.0), state, current, step
        ),
        (field, source, dt),
        warmup,
        repeats,
    )
    before = solver.field_energy(field)
    record["memory"] = {
        "prepared_solver_bytes": logical_array_bytes(solver),
        "field_state_bytes": logical_array_bytes(field),
        "source_bytes": logical_array_bytes(source),
    }
    record["physics"] = {
        "successful": bool(advance.successful),
        "relative_energy_change": float(jnp.abs(advance.energy - before) / before),
        "electric_constraint": float(advance.electric_constraint),
        "step_over_stable_step": 0.5,
    }
    return record


def _species(
    grid: phx.discretization.pic.QuasiCylindricalGrid, particles: int
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    transfer = phx.discretization.pic.AzimuthalTransferPlan(grid, shape_order=1)
    species, transfers = [], []
    for offset, specific, name, mass in (
        (0, -1.0, "electrons", 1.0),
        (10**7, 0.01, "ions", 100.0),
    ):
        support = phx.discretization.ParticleSetPlan(
            jnp.arange(offset, offset + particles),
            mass * jnp.ones((particles,)),
            ambient_dimension=3,
        ).prepare()
        species.append(
            phx.discretization.pic.PICSpeciesPlan(
                phx.discretization.ParticlePopulationPlan(support),
                phx.discretization.pic.PICChargeModelPlan(
                    specific,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
        transfers.append(transfer.prepare(support))
    return tuple(species), tuple(transfers)


def _pic_case(
    grid: phx.discretization.pic.QuasiCylindricalGrid,
    particles: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    species, transfers = _species(grid, particles)
    solver = spectral.QuasiCylindricalMaxwellPlan(grid).prepare(transfers)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=species)
    generator = np.random.default_rng(11)
    radius = _PLASMA_RADIUS * np.sqrt(generator.uniform(0.0, 1.0, particles))
    angle = generator.uniform(0.0, 2.0 * np.pi, particles)
    position = jnp.asarray(
        np.stack(
            (
                radius * np.cos(angle),
                radius * np.sin(angle),
                generator.uniform(grid.lower, grid.upper, particles),
            ),
            axis=-1,
        )
    )
    velocity = jnp.asarray(generator.normal(0.0, 0.1, (particles, 3)))
    dt = 0.5 * solver.stable_step
    state = pic.initialize((position, position), (velocity, jnp.zeros_like(velocity)), dt)
    result, record = _measure(
        lambda value, step: pic.step_detailed(value, step),
        (state, dt),
        warmup,
        repeats,
    )
    charge = jnp.max(jnp.abs(result.accepted_state.field.charge))
    record["configuration"] = {"particles_per_species": particles, "species": 2}
    record["memory"] = {
        "prepared_solver_bytes": logical_array_bytes(solver),
        "pic_state_bytes": logical_array_bytes(state),
    }
    record["physics"] = {
        "successful": bool(result.successful),
        "relative_gauss_residual": float(result.diagnostics.electric_constraint / charge),
        "pairing_defect": pic.pairing_defect,
    }
    return record


def _size_case(
    radial: int, modes: int, axial: int, particles: int, warmup: int, repeats: int
) -> dict[str, object]:
    grid = phx.discretization.pic.QuasiCylindricalGrid(
        _RADIUS, radial, 0.0, _LENGTH, axial, modes
    )
    return {
        "radial_count": radial,
        "mode_count": modes,
        "axial_count": axial,
        "shared_grid_hankel_prepare": _hankel_case(grid, warmup, repeats),
        "vacuum_advance": _vacuum_case(grid, warmup, repeats),
        "pic_step": _pic_case(grid, particles, warmup, repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radial-counts", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--mode-counts", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--axial-count", type=int, default=64)
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--small",
        action="store_true",
        help="smoke sizes: N_r = 16, M + 1 in (1, 2), N_z = 16, 32 particles",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.small:
        args.radial_counts, args.mode_counts = [16], [1, 2]
        args.axial_count, args.particles = 16, 32
    if (
        min(args.radial_counts) < 4
        or min(args.mode_counts) < 1
        or args.axial_count < 4
        or args.particles < 1
        or args.repeats <= 0
    ):
        raise ValueError(
            "radial and axial counts must be at least 4, mode counts and particles "
            "positive, and repeats positive"
        )
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "radius": _RADIUS,
            "length": _LENGTH,
            "radial_counts": args.radial_counts,
            "mode_counts": args.mode_counts,
            "axial_count": args.axial_count,
            "particles_per_species": args.particles,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "sizes": [
            _size_case(
                radial, modes, args.axial_count, args.particles, args.warmup, args.repeats
            )
            for radial in args.radial_counts
            for modes in args.mode_counts
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
