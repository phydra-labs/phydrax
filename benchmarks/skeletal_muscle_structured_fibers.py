#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Per-block compiled timing/memory of the actual structured Shorten response."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax.applications.skeletal_muscle.cellular import ShortenFastTwitchModel
from phydrax.applications.skeletal_muscle.fibers import (
    PrescribedFiberStimulusSchedule,
    Shorten2007FiberReaction,
    StructuredFiberResponsePlan,
)


def _measure(function, args, repeats):
    compiled, compilation = measure_lower_and_compile(
        lambda: eqx.filter_jit(function).lower(*args),
        lambda lowered: lowered.compile(),
    )
    result, timing = measure_repeated(lambda: compiled(*args), warmup=1, repeats=repeats)
    evidence = compiler_evidence(
        compiled.compiled.cost_analysis(),
        compiled.compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="Backend did not provide compiler estimates.",
    )
    return result, {
        "compilation": asdict(compilation),
        "steady_ms": timing.to_dict(unit="milliseconds"),
        "compiler_estimates": asdict(evidence),
    }


def _case(fibers: int, nodes: int, repeats: int) -> dict[str, object]:
    positions = (
        jnp.zeros((fibers, nodes, 3)).at[:, :, 0].set(jnp.linspace(0.0, 10.0, nodes))
    )
    mask = jnp.zeros((1, fibers, nodes), dtype=bool).at[0, :, 0].set(True)
    stimulus = PrescribedFiberStimulusSchedule(
        jnp.asarray([0.0]),
        jnp.asarray([0.1]),
        jnp.asarray([150.0]),
        mask,
    )
    runtime = StructuredFiberResponsePlan(
        tuple(f"fiber-{index}" for index in range(fibers)),
        positions,
        stimulus,
        jnp.asarray([0.0, 0.5, 1.0]),
        geometry_source_id="manufactured-10mm-parallel-world-x-benchmark",
        maximum_step_ms=0.02,
    ).prepare(
        Shorten2007FiberReaction(ShortenFastTwitchModel()),
        jnp.full((fibers, nodes - 1), 0.1),
    )
    source = runtime.initialize()
    path = runtime.linear_geometry_path(source, 1.02 * source.node_positions_mm)
    candidate, total = _measure(
        runtime.candidate, (source, jnp.asarray(0.02), path), repeats
    )
    g0 = runtime.geometry(path[0])
    g1 = runtime.geometry(path[1])
    gm = runtime.geometry(0.5 * (path[0] + path[1]))
    reaction, reaction_timing = _measure(
        runtime._reaction_step,
        (
            source.values,
            jnp.asarray(0.0),
            jnp.asarray(0.005),
            jnp.asarray(0.0),
            stimulus.current(0.005),
            g0.node_jacobian,
            (g1.node_jacobian - g0.node_jacobian) / 0.01,
        ),
        repeats,
    )
    diffusion, diffusion_timing = _measure(
        runtime._diffusion_step,
        (reaction[0][..., 0], gm, jnp.asarray(0.01)),
        repeats,
    )
    return {
        "fiber_count": fibers,
        "node_count": nodes,
        "cell_state_count": runtime.reaction.state_count,
        "local_cell_count": fibers * nodes,
        "prepared_id": runtime.prepared_id,
        "logical_state_payload_bytes": logical_array_bytes(source),
        "candidate": total,
        "one_local_reaction_half_step": reaction_timing,
        "one_diffusion_step": diffusion_timing,
        "reaction_solver_steps_per_substep": candidate.evidence.reaction_solver_steps.tolist(),
        "maximum_diffusion_relative_residual": float(
            jnp.max(candidate.evidence.diffusion_relative_residual)
        ),
        "successful": bool(candidate.evidence.successful)
        and bool(reaction[1])
        and bool(reaction[2])
        and bool(diffusion[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fibers", type=int)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.fibers is None) != (args.nodes is None):
        parser.error("--fibers and --nodes must be supplied together")
    if args.repeats < 1 or (
        args.fibers is not None and (args.fibers < 1 or args.nodes < 3)
    ):
        parser.error("repeats/fibers must be positive and nodes must be at least three")
    jax.config.update("jax_enable_x64", True)
    shapes = (
        [(args.fibers, args.nodes)]
        if args.fibers is not None
        else ([(1, 3)] if args.smoke else [(1, 17), (8, 65)])
    )
    cases = [_case(fibers, nodes, args.repeats) for fibers, nodes in shapes]
    payload = {
        "environment": capture_environment().to_dict(),
        "solver_structure": {
            "reaction": "vmap independent local 56-state Kvaerno5/VeryChord dense local LU",
            "diffusion": "native batched tridiagonal line solve; no full-bundle Jacobian",
            "splitting": "Strang reaction/diffusion/reaction with midpoint moving geometry",
            "reaction_jacobian_storage_order": "fiber_count * node_count * 56^2",
            "diffusion_storage_order": "fiber_count * node_count",
            "numerical_fallback": False,
        },
        "scope": "Local-device timings; compiler memory estimates, not peaks. Block timings are nonadditive.",
        "cases": cases,
        "all_successful": all(case["successful"] for case in cases),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")
    if not payload["all_successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
