#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated spline-Whitney deposition, PIC filter, and cell-binning cost.

The controlling capacity is the particle count: every shape order deposits the
same paths at each capacity, so compile/execution time and compiler memory
scale with the route count ``N · 12 · p(p+1)²`` (order ``p ≥ 2``) or ``N · 48``
(order one). The filter and binning phases run on the same grid and paths.
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
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx
from phydrax.discretization.pic import PICShapeOrder


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


def _measure(function: Any, arguments: Any, warmup: int, repeats: int) -> Any:
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


def _bridge(cells: int) -> Any:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(cells, periodic=True) for _ in range(3)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0] * 3, [1.0] * 3]))
    return phx.discretization.StructuredCochainBridge(grid)


def _transfer(bridge: Any, order: PICShapeOrder, particles: int) -> Any:
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(particles), jnp.ones((particles,)), ambient_dimension=3
    ).prepare()
    charged = phx.discretization.ChargedParticlePlan(
        -jnp.ones((particles,)), "benchmark"
    ).prepare(support)
    return phx.discretization.pic.PICParticleCochainTransferPlan(
        bridge, shape_order=order
    ).prepare(charged)


def _paths(cells: int, particles: int) -> tuple[Any, Any]:
    generator = np.random.default_rng(17)
    start = generator.uniform(0.0, 1.0, (particles, 3))
    end = start + generator.uniform(-0.4, 0.4, (particles, 3)) / cells
    return jnp.asarray(start), jnp.asarray(end)


def _deposit_case(
    bridge: Any,
    cells: int,
    order: PICShapeOrder,
    particles: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    current = phx.discretization.pic.ChargeConservingCurrentPlan(
        _transfer(bridge, order, particles)
    )
    start, end = _paths(cells, particles)
    result, record = _measure(
        lambda left, right: current.deposit(left, right, jnp.asarray(1.0e-3)),
        (start, end),
        warmup,
        repeats,
    )
    rate = jnp.max(jnp.abs(result.end_charge.cochain - result.start_charge.cochain))
    record["configuration"] = {"shape_order": order, "particles": particles}
    record["physics"] = {
        "successful": bool(result.successful),
        "relative_continuity_defect": float(
            result.maximum_continuity_defect / (rate / 1.0e-3)
        ),
        "segments": int(result.segment_count),
    }
    return record


def _filter_case(bridge: Any, warmup: int, repeats: int) -> dict[str, object]:
    transfers = (_transfer(bridge, 2, 1),)
    maxwell = phx.solver.CompatibleMaxwellPlan(
        bridge, sources=(phx.solver.PICMaxwellCurrentSourcePlan(),)
    ).prepare()
    solver = phx.solver.CochainMaxwellPICFieldSolver(
        maxwell,
        phx.solver.CochainElectrostaticPlan(
            bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
        ),
        transfers,
        tuple(phx.discretization.pic.ChargeConservingCurrentPlan(v) for v in transfers),
    )
    plan = phx.solver.PICFilterPlan(passes=2)
    generator = np.random.default_rng(3)
    counts = bridge.cochain.cell_counts
    charge = jnp.asarray(generator.standard_normal(counts[0]))
    current = jnp.asarray(generator.standard_normal(counts[1]))
    field = solver.field_with_charge(charge)

    def apply(rho: Any, flux: Any, state: Any) -> Any:
        return (
            plan.filter_charge(solver, rho),
            plan.filter_current(solver, flux),
            plan.filter_field(solver, state),
        )

    _, record = _measure(apply, (charge, current, field), warmup, repeats)
    report = plan.continuity_report(solver)
    record["physics"] = {
        "interior_commutation_defect": report.interior_commutation_defect,
        "gauss_initialization_defect": report.gauss_initialization_defect,
    }
    return record


def _binning_case(
    cells: int, particles: int, warmup: int, repeats: int
) -> dict[str, object]:
    plan = phx.discretization.pic.PICCellBinningPlan(
        (0.0,) * 3, (1.0,) * 3, (cells,) * 3, (True,) * 3
    )
    start, _ = _paths(cells, particles)
    active = jnp.ones((particles,), dtype=jnp.bool_)
    identity = (
        jnp.zeros((particles,), dtype=jnp.uint32),
        jnp.arange(particles, dtype=jnp.uint32)[::-1],
    )
    bins, record = _measure(
        lambda position, mask, hi, lo: plan.bin(position, mask, identity=(hi, lo)),
        (start, active, *identity),
        warmup,
        repeats,
    )
    record["configuration"] = {"particles": particles}
    record["physics"] = {
        "successful": bool(bins.successful),
        "occupied_cells": int(bins.occupied_cells),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=8)
    parser.add_argument("--particles", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cells < 4 or min(args.particles) <= 0 or args.repeats <= 0:
        raise ValueError("cells >= 4 and positive particle counts/repeats are required")
    bridge = _bridge(args.cells)
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "cells": args.cells,
            "particles": args.particles,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "deposition": [
            _deposit_case(bridge, args.cells, order, count, args.warmup, args.repeats)
            for order in (1, 2, 3)
            for count in args.particles
        ],
        "filter": _filter_case(bridge, args.warmup, args.repeats),
        "binning": [
            _binning_case(args.cells, count, args.warmup, args.repeats)
            for count in args.particles
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
