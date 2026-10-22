#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated weak and strong scaling of distributed electromagnetic PIC.

Routes:

- ``reduced-slabs``: reduced 1-D solver, one-axis mesh (slabs);
- ``spectral-global-fft-blocks``: finite-order global-FFT PSATD on an exactly
  factored mesh (prime counts use slabs, composite counts use pencils) with
  diagonal migration;
- ``spectral-local-guarded-blocks``: the same stencil with local-guarded
  transforms, one subdomain per device, guards exchanged per step.

The controlling sizes are the cells per device along each decomposed axis and
the species slots per device. Weak scaling rounds the requested cell width to
one common FFT-compatible width, keeping the actual cells and slots per device
fixed. Strong scaling rounds the largest requested grid and slot count up to
common divisibility, not to a whole least-common-multiple of device problems.
Local-guarded strong runs share one fixed subdomain tiling as well as the grid,
particles, timestep, and stencil. Each case records actual sizes alongside
the requested sizes. Each device exchanges guards and migrates particles through
fixed ``packet_capacity`` packets. Each case records host preparation (including
the locality and pairing probes), initialization, lowering and compilation,
warm step execution, compiler memory, migration counts, the local-guarded
stencil-truncation evidence, and, for strong scaling, the largest field
difference from the one-device run (reduction-order roundoff).

The default campaign explicitly selects 1/2/4/8 devices. Every warmup and
measured attempt must commit exactly one finite, advancing, unrejected step.
Acceptance audits run outside execution timing; rejected attempts remain in
``sample_outcomes``. Strong-scaling field errors must be finite and at most
1e-11 for reduced/global-FFT routes or 1e-2 for the explicitly approximate
local-guarded route. Any failed gate retains its evidence and exits nonzero.

Run on the visible accelerator devices. For CPU-only functional coverage use
``JAX_NUM_CPU_DEVICES=8 python -m benchmarks.pic_distributed``; forced host
devices do not establish GPU performance.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

import phydrax as phx
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    DurationDistribution,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_synchronized,
)
from tools._accelerator_process import JSON_LIMIT, write_json_evidence


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
    """Exact near-square factorization, with the larger factor first."""
    if parts <= 0:
        raise ValueError("parts must be positive.")
    if route == "reduced-slabs":
        return (parts,)
    if route not in _ROUTES:
        raise ValueError(f"Unknown route {route!r}.")
    second = math.isqrt(parts)
    while parts % second:
        second -= 1
    return (parts // second, second)


def _mesh_shape(blocks: tuple[int, ...], /) -> tuple[int, ...]:
    """Device mesh of a block shape: size-one axes are dropped."""
    return tuple(value for value in blocks if value > 1) or (1,)


@dataclass(frozen=True)
class _Problem:
    counts: tuple[int, ...]
    capacity: int
    subdomains: tuple[int, ...] | None = None


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _problem_sizes(
    route: str, parts: list[int], cells: int, slots: int
) -> tuple[list[_Problem], _Problem]:
    """Size both campaigns against PIC and canonical slab/pencil FFT layouts."""
    if not parts or any(count <= 0 for count in parts) or cells <= 0 or slots <= 0:
        raise ValueError("parts, cells, and slots must be positive.")
    blocks = [_block_shape(route, count) for count in parts]
    capacity = _round_up(slots * max(parts), math.lcm(*parts))
    if route == "reduced-slabs":
        weak = [_Problem((cells * count,), slots * count) for count in parts]
        strong = _Problem((_round_up(cells * max(parts), math.lcm(*parts)),), capacity)
        return weak, strong

    # Slab transposes partition y by the whole mesh; pencil transposes
    # partition z by the whole mesh. These are native FFT layout constraints.
    slab_multiple = math.lcm(
        *(count for count, block in zip(parts, blocks, strict=True) if block[1] == 1)
    )
    pencil_multiple = math.lcm(
        *(count for count, block in zip(parts, blocks, strict=True) if block[1] > 1)
    )
    width = _round_up(cells, slab_multiple)
    depth = _round_up(8, pencil_multiple)
    local = route == "spectral-local-guarded-blocks"
    weak = [
        _Problem(
            (width * block[0], width * block[1], depth),
            slots * count,
            (*block, 1) if local else None,
        )
        for count, block in zip(parts, blocks, strict=True)
    ]
    tiling = tuple(math.lcm(*(block[axis] for block in blocks)) for axis in range(2))
    counts = (
        _round_up(cells * max(block[0] for block in blocks), tiling[0]),
        _round_up(
            cells * max(block[1] for block in blocks),
            math.lcm(tiling[1], slab_multiple),
        ),
        depth,
    )
    return weak, _Problem(counts, capacity, (*tiling, 1) if local else None)


def _base(
    route: str, problem: _Problem, mesh: Mesh
) -> tuple[Any, tuple[Any, ...], int, tuple[float, ...]]:
    """Solver, species, particle dimension, and domain lengths of one case."""
    counts, capacity = problem.counts, problem.capacity
    match route:
        case "reduced-slabs":
            grid = _D.TensorGridPlan(
                (_D.UniformCellAxisSpec(counts[0], periodic=True),),
                axis_names=("x",),
            ).prepare(jnp.asarray([[0.0], [1.0]]))
            base = phx.solver.ReducedMaxwellPICFieldSolver(
                phx.solver.CompatibleMaxwell1DPlan(grid),
                _PIC.ReducedPICTransferPlan(grid),
            )
            return base, _species(capacity, 1), 1, (1.0,)
        case "spectral-global-fft-blocks" | "spectral-local-guarded-blocks":
            h = 0.125
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
                    "subdomains": problem.subdomains,
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
    problem: _Problem,
    packet: int,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, object], Any]:
    """One case with explicit scientific sizes, independent of device count."""
    start = time.perf_counter()
    shape = _mesh_shape(_block_shape(route, parts))
    mesh = Mesh(
        np.asarray(jax.devices()[:parts], dtype=object).reshape(shape),
        ("x", "y")[: len(shape)],
    )
    capacity = problem.capacity
    base, species, dimension, lengths = _base(route, problem, mesh)
    solver = phx.solver.distribute_pic_field_solver(base, mesh)
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
    result, execution, sample_outcomes = _measure_steps(
        lambda: compiled(state), state, warmup=warmup, repeats=repeats
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
        "grid_shape": list(problem.counts),
        "cells_per_part": math.prod(problem.counts) // parts,
        "local_cells": [
            count // blocks
            for count, blocks in zip(
                problem.counts,
                (*shape, *((1,) * (len(problem.counts) - len(shape)))),
                strict=True,
            )
        ],
        "spectral_subdomains": problem.subdomains,
        "slots_per_part": capacity // parts,
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
        "acceptance": {
            "all_samples_accepted": all(row["passed"] for row in sample_outcomes),
            "finite_candidate": _finite_state(result.candidate_state),
            "finite_accepted": _finite_state(result.accepted_state),
            "accepted_step_delta": int(result.accepted_state.accepted_step)
            - int(state.accepted_step),
            "time_advanced": bool(result.accepted_state.time > state.time),
            "field_absolute_tolerance": (
                1e-2 if route.endswith("guarded-blocks") else 1e-11
            ),
        },
        "sample_outcomes": sample_outcomes,
    }
    return record, result.accepted_state


def _finite_state(state: Any) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree.leaves(state)
        if np.issubdtype(np.asarray(leaf).dtype, np.inexact)
    )


def _measure_steps(
    operation: Callable[[], Any], previous: Any, *, warmup: int, repeats: int
) -> tuple[Any, DurationDistribution, list[dict[str, Any]]]:
    """Audit every attempted step outside its synchronized execution timing."""
    if warmup < 0 or repeats <= 0 or warmup + repeats > 1024:
        raise ValueError(
            "warmup must be nonnegative; repeats must be positive; at most 1024 samples"
        )
    samples = []
    outcomes = []
    result = None
    for index in range(warmup + repeats):
        result, seconds = measure_synchronized(operation)
        successful = bool(result.successful)
        reason = int(result.rejection_reason)
        advanced = int(result.accepted_state.accepted_step) == int(
            previous.accepted_step
        ) + 1 and bool(result.accepted_state.time > previous.time)
        finite = _finite_state(result.candidate_state) and _finite_state(
            result.accepted_state
        )
        outcomes.append(
            {
                "index": index,
                "warmup": index < warmup,
                "successful": successful,
                "rejection_reason": reason,
                "advanced": advanced,
                "finite": finite,
                "passed": successful and reason == 0 and advanced and finite,
            }
        )
        if index >= warmup:
            samples.append(seconds)
    if result is None:
        raise RuntimeError("PIC measurement produced no step")
    return result, DurationDistribution(tuple(samples)), outcomes


def _case_errors(record: dict[str, Any], *, reference: bool = False) -> list[str]:
    physics, acceptance = record["physics"], record["acceptance"]
    errors = []
    if physics["successful"] is not True or physics["rejection_reason"] != 0:
        errors.append(f"PIC rejected the step: reason={physics['rejection_reason']}")
    if acceptance["all_samples_accepted"] is not True:
        errors.append("At least one measured or warmup PIC step failed acceptance")
    if (
        acceptance["finite_candidate"] is not True
        or acceptance["finite_accepted"] is not True
        or acceptance["accepted_step_delta"] != 1
        or acceptance["time_advanced"] is not True
    ):
        errors.append("PIC did not commit one finite advancing step")
    if reference:
        difference = record["field_difference_from_one_device"]
        if (
            difference is None
            or not math.isfinite(difference)
            or difference > acceptance["field_absolute_tolerance"]
            or difference < 0
        ):
            errors.append("PIC field difference exceeds the finite scientific tolerance")
    return errors


def _field_difference(left: Any, right: Any) -> float:
    differences = [
        float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        for a, b in zip(
            jax.tree.leaves(left.field), jax.tree.leaves(right.field), strict=True
        )
    ]
    return max(differences) if all(map(math.isfinite, differences)) else math.inf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", nargs="+", choices=_ROUTES, default=list(_ROUTES))
    parser.add_argument("--parts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--cells-per-part", type=int, default=16)
    parser.add_argument("--slots-per-part", type=int, default=512)
    parser.add_argument("--packet-capacity", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    parts = sorted(args.parts)
    if (
        parts[0] != 1
        or len(set(parts)) != len(parts)
        or parts[-1] > len(jax.devices())
        or args.cells_per_part <= 0
        or args.slots_per_part <= 0
        or args.packet_capacity <= 0
        or args.warmup < 0
        or args.repeats <= 0
        or args.warmup + args.repeats > 1024
    ):
        raise ValueError(
            "parts must be positive, start at 1, and fit the visible devices; cells, "
            "slots, packet capacity, and repeats must be positive; warmup must be nonnegative."
        )
    timing = (args.warmup, args.repeats)
    routes = {}
    errors = []
    for route in args.routes:
        weak_problems, strong_problem = _problem_sizes(
            route, parts, args.cells_per_part, args.slots_per_part
        )
        weak = [
            _case(route, count, problem, args.packet_capacity, *timing)[0]
            for count, problem in zip(parts, weak_problems, strict=True)
        ]
        for record in weak:
            failures = _case_errors(record)
            record.update(passed=not failures, errors=failures)
            errors.extend(
                f"{route}/weak/{record['parts']}: {error}" for error in failures
            )
        strong = []
        reference = None
        for count in parts:
            record, state = _case(
                route, count, strong_problem, args.packet_capacity, *timing
            )
            reference = state if reference is None else reference
            difference = _field_difference(state, reference)
            record["field_difference_from_one_device"] = (
                difference if math.isfinite(difference) else None
            )
            failures = _case_errors(record, reference=True)
            record.update(passed=not failures, errors=failures)
            errors.extend(f"{route}/strong/{count}: {error}" for error in failures)
            strong.append(record)
        routes[route] = {"weak_scaling": weak, "strong_scaling": strong}
    payload = {
        "passed": not errors,
        "errors": errors,
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
    encoded = json.dumps(payload, indent=2, allow_nan=False)
    if len(encoded.encode("utf-8")) > JSON_LIMIT:
        raise ValueError("PIC benchmark evidence exceeds the bounded JSON limit")
    if args.output is None:
        print(encoded)
    else:
        write_json_evidence(args.output, payload)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
