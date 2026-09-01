#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp

import phydrax as phx


def _time(function, *arguments, repetitions=3):
    compiled = jax.jit(function)
    started = perf_counter()
    result = compiled(*arguments)
    jax.block_until_ready(jax.tree.leaves(result)[0])
    first = (perf_counter() - started) * 1e3
    started = perf_counter()
    for _ in range(repetitions):
        result = compiled(*arguments)
    jax.block_until_ready(jax.tree.leaves(result)[0])
    steady = (perf_counter() - started) * 1e3 / repetitions
    return first, steady


def _base(schedule=None, assignment=None):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(16, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    x, y = jnp.meshgrid(
        (jnp.arange(8) + 0.37) / 8.0,
        (jnp.arange(8) + 0.37) / 8.0,
        indexing="ij",
    )
    position = jnp.stack((x, y), axis=-1).reshape((-1, 2))
    volume = jnp.full((position.shape[0],), 1.0 / position.shape[0])
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0]), volume, ambient_dimension=2
    ).prepare()
    assignment_ = (
        phx.discretization.TensorBSplineSplatAssignment(2)
        if assignment is None
        else assignment
    )
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=assignment_
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "benchmark",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(schedule=schedule),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    velocity = 1.0e-3 * jnp.sin(2.0 * jnp.pi * position)
    state = compiled.initialize_state(position, velocity, volume, arguments)
    return compiled, arguments, state


def run(output: Path):
    metrics = {}
    for schedule in (
        phx.discretization.USLMPMSchedule(),
        phx.discretization.USFMPMSchedule(),
        phx.discretization.MUSLMPMSchedule(),
    ):
        compiled, arguments, state = _base(schedule)
        first, steady = _time(
            lambda value, compiled=compiled, arguments=arguments: (
                compiled.dynamics.step_detailed(value, 2e-4, arguments)
            ),
            state,
        )
        metrics[f"schedule_{schedule.common_name}"] = {
            "compile_and_first_ms": first,
            "steady_ms": steady,
        }
    widths = jnp.full((64, 2), 0.02)
    for name, assignment in (
        ("gimp", phx.discretization.UniformGIMPSplatAssignment(widths)),
        (
            "cpdi",
            phx.discretization.AffineCPDISplatAssignment(
                jnp.broadcast_to(0.02 * jnp.eye(2), (64, 2, 2))
            ),
        ),
    ):
        compiled, _, state = _base(assignment=assignment)
        first, steady = _time(
            lambda position, deformation, assignment_input, compiled=compiled: (
                compiled.dynamics.splat.build(
                    position,
                    assignment_input=compiled.dynamics.splat.plan.assignment.update_input(
                        position, deformation, assignment_input
                    ),
                )
            ),
            state.particles.position,
            state.particles.deformation_gradient,
            state.assignment_input,
        )
        metrics[f"assignment_{name}"] = {
            "compile_and_first_ms": first,
            "steady_ms": steady,
            "routes": compiled.dynamics.splat.route_count,
        }
    contact = phx.discretization.RigidMPMContactPlan(
        phx.geometry.Circle((0.0, 0.0), 0.5).compile(),
        phx.discretization.SharpCoulombMPMFrictionPlan(0.25),
        contact_band=0.02,
    )
    points = jnp.stack((jnp.linspace(0.45, 0.55, 1024), jnp.zeros((1024,))), axis=-1)
    velocity = jnp.broadcast_to(jnp.asarray((-0.2, 0.1)), points.shape)
    mass = jnp.ones((1024,))
    first, steady = _time(lambda v: contact.apply(points, v, mass, 0.0, 0.01), velocity)
    metrics["rigid_contact"] = {
        "compile_and_first_ms": first,
        "steady_ms": steady,
        "nodes": 1024,
    }
    compiled, arguments, state = _base()
    implicit = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics)
    started = perf_counter()
    result = implicit.step_detailed(state, 2e-4, arguments)
    jax.block_until_ready(result.accepted_state.particles.position)
    metrics["implicit"] = {
        "solve_ms": (perf_counter() - started) * 1e3,
        "nonlinear_steps": int(result.diagnostics.nonlinear_steps),
        "linear_iterations": int(result.diagnostics.linear_iterations),
    }
    routes = compiled.dynamics.splat.build(state.particles.position)
    blocks = phx.discretization.MPMActiveBlockPlan((16, 16), (4, 4), 16)
    first, steady = _time(lambda _: blocks.build(routes), jnp.asarray(0))
    active = blocks.build(routes)
    storage = phx.discretization.BlockSparseMPMNodalStoragePlan(blocks)
    dense = jnp.ones((16, 16, 4))
    _, pack_steady = _time(lambda value: storage.pack(value, active), dense)
    compact = storage.pack(dense, active)
    _, unpack_steady = _time(lambda value: storage.unpack(value, active), compact)
    metrics["sparse"] = {
        "activation_compile_and_first_ms": first,
        "activation_steady_ms": steady,
        "pack_steady_ms": pack_steady,
        "unpack_steady_ms": unpack_steady,
        "active_blocks": int(active.active_block_count),
        "dense_values": int(dense.size),
        "compact_values": int(compact.size),
    }
    passed = all(
        all(
            isinstance(value, (int, float)) and jnp.isfinite(value)
            for key, value in case.items()
            if key.endswith("_ms")
        )
        for case in metrics.values()
    )
    payload = {"maturity": "experimental", "cases": metrics, "passed": bool(passed)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Benchmark advanced MPM capabilities.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/material_point_advanced.json"),
    )
    arguments = parser.parse_args()
    run(arguments.output)


if __name__ == "__main__":
    main()
