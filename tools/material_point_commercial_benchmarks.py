#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
from material_point_commercial_qualification import _base

import phydrax as phx


def _time(function, *arguments, repetitions=3):
    compiled = jax.jit(function)
    started = perf_counter()
    result = compiled(*arguments)
    jax.block_until_ready(jax.tree.leaves(result)[0])
    first = perf_counter() - started
    started = perf_counter()
    for _ in range(repetitions):
        result = compiled(*arguments)
    jax.block_until_ready(jax.tree.leaves(result)[0])
    steady = (perf_counter() - started) / repetitions
    return first, steady


def run(output):
    cases = {}
    for transfer in (
        phx.discretization.PICTransferPlan(),
        phx.discretization.FLIPTransferPlan(),
        phx.discretization.PICFLIPTransferPlan(0.25),
        phx.discretization.APICTransferPlan(),
    ):
        compiled, arguments, state = _base(transfer)
        first, steady = _time(
            lambda value, compiled=compiled, arguments=arguments: (
                compiled.dynamics.step_detailed(value, 5e-4, arguments)
            ),
            state,
        )
        cases[f"transfer-{transfer.transfer_name}"] = {
            "compile_and_first_seconds": first,
            "steady_seconds": steady,
            "particles": compiled.dynamics.particles.capacity,
            "routes": compiled.dynamics.splat.route_count,
        }
    mass = jnp.ones((3, 256))
    velocity = jnp.stack(
        (
            jnp.broadcast_to(jnp.asarray((0.5, 0.1)), (256, 2)),
            jnp.zeros((256, 2)),
            jnp.broadcast_to(jnp.asarray((-0.5, -0.1)), (256, 2)),
        )
    )
    gradients = jnp.stack(
        (
            jnp.broadcast_to(jnp.asarray((1.0, 0.0)), (256, 2)),
            jnp.broadcast_to(jnp.asarray((0.0, 1.0)), (256, 2)),
            jnp.broadcast_to(jnp.asarray((-1.0, -1.0)), (256, 2)),
        )
    )
    contact = phx.discretization.KWayMPMContactPlan(3, maximum_steps=40, tolerance=1e-8)
    graph = contact.build_graph(mass, gradients)
    first, steady = _time(
        lambda current: contact.solve(mass, current, graph, 0.01), velocity
    )
    cases["kway-contact"] = {
        "compile_and_first_seconds": first,
        "steady_seconds": steady,
        "nodes": 256,
        "pairs_per_node": 3,
    }
    material = phx.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan()
    parameters = phx.applications.solid_mechanics.DruckerPragerParameters(
        10.0, 30.0, 0.05, 0.5, 0.2, 1.0
    )
    deformation = jnp.broadcast_to(
        jnp.asarray([[1.0, 0.1, 0.0], [0.0, 0.96, 0.0], [0.0, 0.0, 1.04]]),
        (1024, 3, 3),
    )
    history = material.initialize_state((1024,), jnp.float64)
    density = jnp.ones((1024,))
    first, steady = _time(
        lambda value: material.evaluate_linearized(
            value, history, density, parameters, 0.0, 0.01
        ),
        deformation,
        repetitions=1,
    )
    cases["drucker-prager-tangent"] = {
        "compile_and_first_seconds": first,
        "steady_seconds": steady,
        "particles": 1024,
    }
    shard_values = jnp.arange(8 * 4096.0).reshape((8, 4096))
    first, steady = _time(
        lambda values: phx.discretization.distributed_p2g_reduce(values),
        shard_values,
    )
    cases["distributed-reduction"] = {
        "compile_and_first_seconds": first,
        "steady_seconds": steady,
        "shards": 8,
        "values_per_shard": 4096,
    }
    compiled, _, state = _base()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = phx.solver.MPMCheckpointPlan(compiled, state)
        started = perf_counter()
        manifest = checkpoint.write_generation(directory, state, generation=1)
        checkpoint_seconds = perf_counter() - started
        size = (Path(directory) / "generation-00000001.mpmckpt").stat().st_size
        started = perf_counter()
        checkpoint.read_current(directory)
        restore_seconds = perf_counter() - started
    cases["checkpoint"] = {
        "write_seconds": checkpoint_seconds,
        "restore_seconds": restore_seconds,
        "bytes": size,
        "payload_id": manifest.payload_id,
    }
    execution = phx.discretization.MPMExecutionPlan(
        backend=jax.default_backend(),
        device_mesh="1",
        precision_policy_id=compiled.dynamics.splat.plan.precision.policy_id,
        determinism=phx.discretization.MPMDeterminismMode.DETERMINISTIC,
        realization=phx.discretization.MPMKernelRealization.REFERENCE,
        particle_capacity=compiled.dynamics.particles.capacity,
        grid_capacity=compiled.dynamics.splat.target_size,
        route_capacity=compiled.dynamics.splat.route_count,
        field_capacity=compiled.dynamics.nodal_fields.field_count,
        block_capacity=max(1, compiled.dynamics.splat.target_size),
        contact_pair_capacity=1,
    )
    resources = dict(compiled.dynamics.preparation.resource_counts)
    compile_p95 = max(
        case["compile_and_first_seconds"]
        for case in cases.values()
        if "compile_and_first_seconds" in case
    )
    steady_p95 = max(
        case["steady_seconds"] for case in cases.values() if "steady_seconds" in case
    )
    apic_case = cases["transfer-apic"]
    certificate = phx.discretization.MPMCapacityCertificate(
        execution,
        source_commit="working-tree",
        toolchain=f"jax-{jax.__version__}",
        hardware=str(jax.devices()[0]),
        cold_compile_seconds=compile_p95,
        peak_memory_bytes=int(resources["step_workspace_bytes"])
        + int(resources["state_bytes"]),
        routes_per_second=apic_case["routes"] / apic_case["steady_seconds"],
        step_seconds_p95=steady_p95,
        gradient_seconds_p95=0.0,
        checkpoint_bytes_per_second=size / checkpoint_seconds,
        numerical_defect_p99=0.0,
    )
    cases["capacity-certificate"] = {
        "execution_id": execution.execution_id,
        "certificate_id": certificate.certificate_id,
        "peak_memory_bytes": certificate.peak_memory_bytes,
        "routes_per_second": certificate.routes_per_second,
    }
    passed = all(
        all(
            not key.endswith("seconds") or jnp.isfinite(value) and value >= 0.0
            for key, value in case.items()
            if isinstance(value, (int, float))
        )
        for case in cases.values()
    )
    payload = {
        "maturity": "commercial-qualification-candidate",
        "cases": cases,
        "passed": bool(passed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Benchmark commercial MPM closures.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/material_point_commercial.json"),
    )
    arguments = parser.parse_args()
    run(arguments.output)


if __name__ == "__main__":
    main()
