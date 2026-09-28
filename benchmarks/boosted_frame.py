#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated boosted-frame PIC step cost against the boost factor.

The controlling capacity is the boost Lorentz factor ``γ_b`` of one fixed
one-dimensional laser-wakefield stage (laser wavelength 1, 16-long plasma of
density 0.01 n_c, 16 cells per Doppler-stretched wavelength). The boosted grid
covers the contracted plasma and the Doppler-stretched laser on the initial
slice, and the number of steps to cross the stage shrinks roughly like
``γ_b²(1 + β_b)``. Each case records the boosted cell and step counts against
the lab-frame counts of the same stage, the lowering, compilation, and warm
execution of one plain PIC step and of one NCI-guarded boosted step with a
back-transformed snapshot ring (their difference is the guard and snapshot
overhead), compiler memory, and the logical bytes of the prepared run, the
state, and the snapshot ring.
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
from phydrax.discretization.pic import ExternalFieldSample


D = phx.discretization
_WAVENUMBER = 2.0 * np.pi
_PLASMA_FREQUENCY2 = 0.01 * _WAVENUMBER**2
_PLASMA = (26.0, 42.0)
_LAB_CELLS_PER_WAVELENGTH = 16
_LAB_STEP = 0.45 / _LAB_CELLS_PER_WAVELENGTH


class _PlaneLaser(phx.StrictModule):
    @property
    def source_id(self) -> str:
        return "boosted-frame-benchmark-laser"

    def external_fields(
        self, positions: jax.Array, times: jax.Array, /
    ) -> ExternalFieldSample:
        xi = positions[:, 2] - times - 20.0
        envelope = jnp.where(jnp.abs(xi) < 5.0, jnp.cos(0.1 * jnp.pi * xi) ** 2, 0.0)
        value = _WAVENUMBER * envelope * jnp.cos(_WAVENUMBER * xi)
        zero = jnp.zeros_like(value)
        return ExternalFieldSample(
            jnp.stack((value, zero, zero), axis=-1),
            jnp.stack((zero, value, zero), axis=-1),
            jnp.ones(value.shape, dtype=jnp.bool_),
        )


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


def _species(
    offset: int, count: int, specific: float, name: str
) -> tuple[D.pic.PICSpeciesPlan, Any]:
    support = D.ParticleSetPlan(
        jnp.arange(offset, offset + count), jnp.ones((count,)), ambient_dimension=3
    ).prepare()
    charged = D.ChargedParticlePlan(specific * jnp.ones((count,)), name).prepare(support)
    plan = D.pic.PICSpeciesPlan(
        D.ParticlePopulationPlan(support),
        D.pic.PICChargeModelPlan(
            specific,
            name,
            minimum_charge_number=1,
            maximum_charge_number=1,
            initial_charge_number=1,
        ),
    )
    return plan, charged


def _case(gamma: float, warmup: int, repeats: int) -> dict[str, object]:
    beta = float(np.sqrt(1.0 - 1.0 / gamma**2))
    frame = phx.solver.BoostedFramePlan(
        phx.LorentzFrame(phx.boost_matrix(jnp.asarray([0.0, 0.0, beta]))),
        phx.geometry.Box([1.5, 1.5, 24.0], [3.0, 3.0, 48.0]),
    )
    start, _ = frame.from_lab(0.0, jnp.asarray([1.5, 1.5, _PLASMA[0]]))
    stop, _ = frame.from_lab(_PLASMA[1] + 10.0, jnp.asarray([1.5, 1.5, _PLASMA[1]]))
    entrance, exit_ = (value / gamma for value in _PLASMA)
    cells = round((exit_ - entrance) * _LAB_CELLS_PER_WAVELENGTH / (gamma * (1.0 + beta)))
    spacing = (exit_ - entrance) / cells
    # On the slice t = β(z − 26) the laser |z − t − 20| < 5 trails back to
    # z = (15 − 26β)/(1 − β); the grid holds it and the contracted plasma.
    laser_tail = (15.0 - 26.0 * beta) / ((1.0 - beta) * gamma)
    lower = (
        entrance
        - np.ceil((entrance - min(laser_tail, entrance) + 4.0) / spacing) * spacing
    )
    count = int(np.ceil((exit_ + 8.0 - lower) / spacing))
    upper = lower + count * spacing
    column = entrance + (np.arange(4 * cells) + 0.5) * spacing / 4
    plasma = np.stack(
        (np.full(column.size, 1.5), np.full(column.size, 1.5), gamma * column), axis=-1
    )
    weight = _PLASMA_FREQUENCY2 * 9.0 * gamma * spacing / 4
    electrons, electron_charge = _species(0, plasma.shape[0], -1.0, "electrons")
    ions, ion_charge = _species(10**6, plasma.shape[0], 1.0 / 1836.0, "ions")
    grid = D.TensorGridPlan(
        (
            D.UniformCellAxisSpec(3, periodic=True),
            D.UniformCellAxisSpec(3, periodic=True),
            D.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, lower], [3.0, 3.0, upper]]))
    bridge = D.StructuredCochainBridge(grid)
    transfer = D.pic.PICParticleCochainTransferPlan(bridge, shape_order=1)
    transfers = tuple(transfer.prepare(value) for value in (electron_charge, ion_charge))
    currents = tuple(D.pic.ChargeConservingCurrentPlan(value) for value in transfers)
    solver = phx.solver.maxwell.spectral.SpectralMaxwellPlan(
        bridge,
        variant="galilean",
        galilean_velocity=frame.galilean_velocity,
        charge_conservation="update-with-rho",
    ).prepare(transfers, currents)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=(electrons, ions))
    lab_bounds = (frame.lab_lower[2], frame.lab_upper[2])
    snapshot_times = tuple(float(value) for value in np.linspace(10.0, 30.0, 4))
    boosted = frame.prepare(
        pic, snapshots=phx.solver.BoostedSnapshotPlan(snapshot_times, species=(0,))
    )
    loaded = frame.boost_particles(plasma, np.zeros_like(plasma), boosted_time=start)
    dt = 0.45 * spacing
    state = boosted.initialize(
        (loaded.positions, loaded.positions),
        (loaded.velocities, loaded.velocities),
        dt,
        time=start,
        masses=(np.full(column.size, weight), np.full(column.size, 1836.0 * weight)),
        vacuum_fields=(_PlaneLaser(),),
    )
    plain, plain_record = _measure(
        lambda value, step: pic.step_detailed(value, step),
        (state.pic, dt),
        warmup,
        repeats,
    )
    guarded, guarded_record = _measure(
        lambda value, step: boosted.step_detailed(value, step),
        (state, dt),
        warmup,
        repeats,
    )
    lab_cells = int(round((lab_bounds[1] - lab_bounds[0]) * _LAB_CELLS_PER_WAVELENGTH))
    lab_steps = int(np.ceil((_PLASMA[1] + 10.0) / _LAB_STEP))
    return {
        "lorentz_factor": gamma,
        "boosted_cells": count,
        "boosted_steps": int(np.ceil((float(stop) - float(start)) / dt)),
        "lab_cells": lab_cells,
        "lab_steps": lab_steps,
        "particles_per_species": int(column.size),
        "plain_pic_step": plain_record,
        "guarded_boosted_step": guarded_record,
        "memory": {
            "prepared_boosted_bytes": logical_array_bytes(boosted),
            "state_bytes": logical_array_bytes(state),
            "snapshot_ring_bytes": logical_array_bytes(state.snapshots),
        },
        "physics": {
            "pic_step_successful": bool(plain.successful),
            "boosted_step_successful": bool(guarded.successful),
            "nci_admissible": bool(guarded.nci_admissible),
            "vacuum_divergence_removed": float(state.evidence.vacuum_divergence),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gammas", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.gammas) <= 1.0 or args.repeats <= 0:
        raise ValueError("gammas must exceed one and repeats must be positive")
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "gammas": args.gammas,
            "lab_cells_per_wavelength": _LAB_CELLS_PER_WAVELENGTH,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "cases": [_case(gamma, args.warmup, args.repeats) for gamma in args.gammas],
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
