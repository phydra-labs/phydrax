#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated Cartesian PSATD field-advance and spectral PIC step cost.

The controlling capacity is the grid size ``N³``: one global-FFT infinite-order
advance costs one batched C2C transform pair of ``N³`` points per field and
source channel, and one local-guarded finite-order advance transforms
``b³`` guard-extended blocks of ``(N/b + 2g)³`` points. Each case records
lowering, compilation, warm execution, compiler memory, and the logical bytes
of the prepared solver and its state. The vacuum cases report the relative
field-energy change of one PSATD step: roundoff for the global transform, the
guard-cell truncation of the propagator tails for local blocks. The PIC step
reports its acceptance, Gauss residual, and deposit↔Gauss pairing defect.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
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


_SUBDOMAINS = (2, 2, 2)
_STENCIL_ORDER = 4
_GUARD_CELLS = (4, 4, 4)
_PARTICLES = 4


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


def _bridge(cells: int) -> phx.discretization.StructuredCochainBridge:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(cells, periodic=True) for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0] * 3, [1.0] * 3]))
    return phx.discretization.StructuredCochainBridge(grid)


def _vacuum_case(
    plan: spectral.SpectralMaxwellPlan, warmup: int, repeats: int
) -> dict[str, object]:
    solver = plan.prepare()
    cells = plan.counts[0]
    x = (jnp.arange(cells, dtype=jnp.float64) + 0.5) / cells
    wave = jnp.broadcast_to(jnp.cos(2.0 * jnp.pi * x)[:, None, None], plan.counts).astype(
        jnp.float64
    )
    zero = jnp.zeros(plan.counts, dtype=jnp.float64)
    field = eqx.tree_at(
        lambda state: (state.electric, state.magnetic),
        solver.field_with_charge(zero),
        (
            jnp.stack((zero, wave, zero), axis=-1),
            jnp.stack((zero, zero, wave), axis=-1),
        ),
    )
    intervals = plan.current_intervals
    source = spectral.SpectralMaxwellSource(
        jnp.zeros((intervals, *plan.counts, 3), dtype=jnp.float64),
        jnp.zeros((intervals, *plan.counts), dtype=jnp.float64),
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
        "step_over_stable_step": 0.5,
    }
    return record


def _species(
    bridge: phx.discretization.StructuredCochainBridge,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    species, charged = [], []
    for offset, charge, name in ((0, -1.0, "electrons"), (100, 1.0, "ions")):
        support = phx.discretization.ParticleSetPlan(
            jnp.arange(offset, offset + _PARTICLES),
            jnp.ones((_PARTICLES,)),
            ambient_dimension=3,
        ).prepare()
        charged.append(
            phx.discretization.ChargedParticlePlan(
                charge * jnp.ones((_PARTICLES,)), name
            ).prepare(support)
        )
        species.append(
            phx.discretization.pic.PICSpeciesPlan(
                phx.discretization.ParticlePopulationPlan(support),
                phx.discretization.pic.PICChargeModelPlan(
                    charge,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    transfer = phx.discretization.pic.PICParticleCochainTransferPlan(
        bridge, shape_order=2
    )
    transfers = tuple(transfer.prepare(value) for value in charged)
    currents = tuple(
        phx.discretization.pic.ChargeConservingCurrentPlan(value) for value in transfers
    )
    return tuple(species), transfers, currents


def _pic_case(
    bridge: phx.discretization.StructuredCochainBridge, warmup: int, repeats: int
) -> dict[str, object]:
    species, transfers, currents = _species(bridge)
    solver = spectral.SpectralMaxwellPlan(bridge).prepare(transfers, currents)
    pic = phx.solver.ElectromagneticPICPlan(solver, species=species)
    generator = np.random.default_rng(11)
    position = jnp.asarray(generator.uniform(0.0, 1.0, (_PARTICLES, 3)))
    velocity = jnp.asarray(generator.uniform(-0.1, 0.1, (_PARTICLES, 3)))
    dt = 0.5 * solver.stable_step
    state = pic.initialize((position, position), (velocity, -velocity), dt)
    result, record = _measure(
        lambda value, step: pic.step_detailed(value, step),
        (state, dt),
        warmup,
        repeats,
    )
    record["configuration"] = {"particles_per_species": _PARTICLES, "species": 2}
    record["memory"] = {
        "prepared_solver_bytes": logical_array_bytes(solver),
        "pic_state_bytes": logical_array_bytes(state),
    }
    record["physics"] = {
        "successful": bool(result.successful),
        "gauss_residual": float(result.diagnostics.electric_constraint),
        "pairing_defect": pic.pairing_defect,
    }
    return record


def _size_case(cells: int, warmup: int, repeats: int) -> dict[str, object]:
    bridge = _bridge(cells)
    global_plan = spectral.SpectralMaxwellPlan(bridge)
    local_plan = spectral.SpectralMaxwellPlan(
        bridge,
        charge_conservation="vay-deposition",
        stencil="finite-order",
        stencil_order=_STENCIL_ORDER,
        decomposition="local-guarded",
        grid="staggered",
        subdomains=_SUBDOMAINS,
        guard_cells=_GUARD_CELLS,
    )
    return {
        "cells": cells,
        "global_fft_infinite_order": _vacuum_case(global_plan, warmup, repeats),
        "local_guarded_finite_order": _vacuum_case(local_plan, warmup, repeats),
        "pic_step": _pic_case(bridge, warmup, repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(args.cells) < 2 * max(_GUARD_CELLS)
        or any(cells % _SUBDOMAINS[0] for cells in args.cells)
        or args.repeats <= 0
    ):
        raise ValueError(
            "cells must be even and at least 8, and repeats must be positive"
        )
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "cells": args.cells,
            "stencil_order": _STENCIL_ORDER,
            "subdomains": list(_SUBDOMAINS),
            "guard_cells": list(_GUARD_CELLS),
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "sizes": [_size_case(cells, args.warmup, args.repeats) for cells in args.cells],
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
