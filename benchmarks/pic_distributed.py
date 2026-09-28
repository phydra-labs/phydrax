#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated weak and strong scaling of distributed electromagnetic PIC.

Routes:

- ``reduced-slabs``: reduced 1-D solver, one-axis mesh (slabs);
- ``spectral-global-fft-blocks``: finite-order global-FFT PSATD on a two-axis
  mesh (``1×1``, ``2×1``, ``2×2``, ``4×2`` blocks; pencil transforms) with
  diagonal migration;
- ``spectral-local-guarded-blocks``: the same stencil with local-guarded
  transforms, one subdomain per device, guards exchanged per step.

The controlling sizes are the cells per device along each decomposed axis and
the species slots per device: each device deposits and gathers its ``C/P``
slots on its block window, exchanges ``guard_cells`` guards per side of every
decomposed axis, and migrates particles through fixed ``packet_capacity``
packets. Weak scaling holds the per-device cells and slots fixed; strong
scaling holds the totals fixed. Each case records host preparation (including
the locality and pairing probes), initialization, lowering and compilation,
warm step execution, compiler memory, migration counts, the local-guarded
stencil-truncation evidence, and, for strong scaling, the largest field
difference from the one-device run (reduction-order roundoff).

Run with forced host devices, e.g.
``XLA_FLAGS=--xla_force_host_platform_device_count=8 python benchmarks/pic_distributed.py``.
"""

from __future__ import annotations

import argparse
import json
import time
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
from jax.sharding import Mesh

import phydrax as phx


_D = phx.discretization
_PIC = _D.pic
_ROUTES = (
    "reduced-slabs",
    "spectral-global-fft-blocks",
    "spectral-local-guarded-blocks",
)


def _species(capacity: int, dimension: int) -> tuple[Any, ...]:
    values = []
    for offset, sign, name in ((0, -1.0, "electrons"), (10**7, 1.0, "ions")):
        support = _D.ParticleSetPlan(
            jnp.arange(offset, offset + capacity),
            jnp.ones((capacity,)),
            ambient_dimension=dimension,
        ).prepare()
        values.append(
            _PIC.PICSpeciesPlan(
                _D.ParticlePopulationPlan(support),
                _PIC.PICChargeModelPlan(
                    sign,
                    name,
                    minimum_charge_number=1,
                    maximum_charge_number=1,
                    initial_charge_number=1,
                ),
            )
        )
    return tuple(values)


def _block_shape(route: str, parts: int) -> tuple[int, ...]:
    """Blocks per decomposed axis for ``parts`` devices (``1×1``, ``2×1``, ``2×2``, …)."""
    if route == "reduced-slabs":
        return (parts,)
    first = 1 << ((parts.bit_length() - 1 + 1) // 2)
    return (first, parts // first)


def _mesh_shape(blocks: tuple[int, ...], /) -> tuple[int, ...]:
    """Device mesh of a block shape: size-one axes are dropped."""
    return tuple(value for value in blocks if value > 1) or (1,)


def _base(
    route: str, shape: tuple[int, ...], cells: int, capacity: int, mesh: Mesh
) -> tuple[Any, tuple[Any, ...], int, tuple[float, ...]]:
    """Solver, species, particle dimension, and domain lengths of one case."""
    match route:
        case "reduced-slabs":
            grid = _D.TensorGridPlan(
                (_D.UniformCellAxisSpec(cells * shape[0], periodic=True),),
                axis_names=("x",),
            ).prepare(jnp.asarray([[0.0], [1.0]]))
            base = phx.solver.ReducedMaxwellPICFieldSolver(
                phx.solver.CompatibleMaxwell1DPlan(grid),
                _PIC.ReducedPICTransferPlan(grid),
            )
            return base, _species(capacity, 1), 1, (1.0,)
        case "spectral-global-fft-blocks" | "spectral-local-guarded-blocks":
            h = 0.125
            counts = (cells * shape[0], cells * shape[1], 8)
            grid = _D.TensorGridPlan(
                tuple(_D.UniformCellAxisSpec(n, periodic=True) for n in counts),
                axis_names=("x", "y", "z"),
            ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [n * h for n in counts]]))
            bridge = _D.StructuredCochainBridge(grid)
            species = _species(capacity, 3)
            transfer = _PIC.PICParticleCochainTransferPlan(bridge, shape_order=2)
            transfers = tuple(
                transfer.prepare(
                    _D.ChargedParticlePlan(
                        plan.charge_model.base_specific_charge * jnp.ones((capacity,)),
                        plan.species_id,
                    ).prepare(plan.population.particles)
                )
                for plan in species
            )
            local = route == "spectral-local-guarded-blocks"
            blocks: dict[str, Any] = (
                {
                    "decomposition": "local-guarded",
                    "subdomains": (shape[0], shape[1], 1),
                    "guard_cells": (6, 6, 4),
                }
                if local
                else {"decomposition": "global-fft"}
            )
            base = phx.solver.maxwell.spectral.SpectralMaxwellPlan(
                bridge,
                grid="staggered",
                charge_conservation="vay-deposition",
                stencil="finite-order",
                stencil_order=8,
                topology=_D.spectral.SpectralMeshTopology(mesh),
                **blocks,
            ).prepare(
                transfers,
                tuple(_PIC.ChargeConservingCurrentPlan(value) for value in transfers),
            )
            return base, species, 3, tuple(n * h for n in counts)
        case _:
            raise ValueError(f"Unknown route {route!r}.")


def _particles(
    capacity: int, dimension: int, lengths: tuple[float, ...]
) -> tuple[np.ndarray, ...]:
    """Half-filled slots on a jittered lattice, so every block has room."""
    rng = np.random.default_rng(7)
    active = np.arange(capacity) < capacity // 2
    position = rng.uniform(0.0, 1.0, (capacity, dimension)) * np.asarray(lengths)
    velocity = np.zeros((capacity, 3))
    velocity[:, :dimension] = rng.uniform(-0.3, 0.3, (capacity, dimension))
    return position, velocity, active, active.astype(np.float64)


def _case(
    route: str,
    parts: int,
    scale: int,
    cells: int,
    slots: int,
    packet: int,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, object], Any]:
    """One case; ``parts`` devices, grid and slots sized for ``scale`` devices."""
    start = time.perf_counter()
    shape = _mesh_shape(_block_shape(route, parts))
    grid_shape = _block_shape(route, scale)
    mesh = Mesh(
        np.asarray(jax.devices()[:parts], dtype=object).reshape(shape),
        ("x", "y")[: len(shape)],
    )
    capacity = slots * scale
    base, species, dimension, lengths = _base(route, grid_shape, cells, capacity, mesh)
    solver = phx.solver.DistributedPICFieldSolver(base, mesh)
    # Local-guarded PSATD keeps Gauss's law to its stencil truncation.
    tolerance = {"constraint_tolerance": 1e-2} if route.endswith("guarded-blocks") else {}
    pic = phx.solver.ElectromagneticPICPlan(solver, species=species, **tolerance)
    run = phx.solver.DistributedElectromagneticPICPlan(pic, packet_capacity=packet)
    preparation = time.perf_counter() - start
    dt = 0.4 * float(base.stable_step)
    position, velocity, active, mass = _particles(capacity, dimension, lengths)
    start = time.perf_counter()
    state = jax.block_until_ready(
        run.initialize(
            (position, position[::-1]),
            (velocity, np.zeros((capacity, 3))),
            dt,
            active_masks=(active, active),
            masses=(mass, mass),
        )
    )
    initialization = time.perf_counter() - start
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(lambda value: run.step_detailed(value, dt)).lower(state),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(state), warmup=warmup, repeats=repeats
    )
    analysis = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="The selected JAX backend did not report compiler analysis.",
    )
    evidence = run.evidence
    truncation = (
        base.guard_truncation(dt)
        if isinstance(base, phx.solver.maxwell.spectral.PreparedSpectralMaxwell)
        and base.plan.decomposition == "local-guarded"
        else None
    )
    record: dict[str, object] = {
        "route": route,
        "parts": parts,
        "mesh_shape": list(shape),
        "cells_per_part_axis": cells,
        "slots_per_species": capacity,
        "packet_capacity": packet,
        "host_preparation_seconds": preparation,
        "initialization_seconds": initialization,
        "compilation": asdict(compilation),
        "step_execution": execution.to_seconds_dict(),
        "compiler": asdict(analysis),
        "memory": {"pic_state_bytes": logical_array_bytes(state)},
        "evidence": {
            "halo_phases": evidence.halo_phases,
            "halo_message_capacity": evidence.halo_message_capacity,
            "field_sharded": evidence.field_sharded,
            "spectral_guard_cells": evidence.spectral_guard_cells,
            "guard_truncation": truncation,
        },
        "physics": {
            "successful": bool(result.successful),
            "migrated": int(result.migration.migrated),
            "rejection_reason": int(result.rejection_reason),
        },
    }
    return record, result.accepted_state


def _field_difference(left: Any, right: Any) -> float:
    return max(
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(
            jax.tree.leaves(left.field), jax.tree.leaves(right.field), strict=True
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", nargs="+", choices=_ROUTES, default=list(_ROUTES))
    parser.add_argument("--parts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--cells-per-part", type=int, default=16)
    parser.add_argument("--slots-per-part", type=int, default=512)
    parser.add_argument("--packet-capacity", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    parts = sorted(set(args.parts))
    if (
        parts[0] != 1
        or parts[-1] > len(jax.devices())
        or any(value & (value - 1) for value in parts)
        or args.repeats <= 0
    ):
        raise ValueError(
            "parts must be powers of two starting at 1 that fit the visible devices; "
            "repeats must be positive."
        )
    options = (args.cells_per_part, args.slots_per_part, args.packet_capacity)
    timing = (args.warmup, args.repeats)
    routes = {}
    for route in args.routes:
        weak = [_case(route, count, count, *options, *timing)[0] for count in parts]
        strong = []
        reference = None
        for count in parts:
            record, state = _case(route, count, parts[-1], *options, *timing)
            reference = state if reference is None else reference
            record["field_difference_from_one_device"] = _field_difference(
                state, reference
            )
            strong.append(record)
        routes[route] = {"weak_scaling": weak, "strong_scaling": strong}
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "routes": args.routes,
            "parts": parts,
            "cells_per_part": args.cells_per_part,
            "slots_per_part": args.slots_per_part,
            "packet_capacity": args.packet_capacity,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "routes": routes,
    }
    encoded = json.dumps(payload, indent=2)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
