#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Deterministic performance and invariant benchmarks for neurofluid substrates.

``--scenario synthetic`` retains image sampling, voxel transfer, and H(div)
microbenchmarks. ``--scenario native-generated-transport`` runs the shared real
generated compartment/Neurofluid admission and independently checked transport
campaign, retaining every warmed execution sample and phase-separated memory.
Each ``--attempts`` run uses a fresh spawned process with a hard ``--timeout``
deadline; passed, failed, and forcibly cancelled attempts all remain in JSON.
Use ``JAX_ENABLE_X64=1 python -m tools.neurofluid_benchmarks
--scenario native-generated-transport --size 2 --repeats 5 --target-error 0.05``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from multiprocessing.connection import Connection
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tools.neurofluid_qualification import (
    add_native_arguments,
    native_transport,
    validate_native_arguments,
)


def image_sampling(size: int, query_count: int, repeats: int) -> dict[str, float | int]:
    contract = phx.SpatialCoordinateContract(
        phx.units.MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame="benchmark",
    )
    affine = phx.imaging.ImageIndexAffine(
        np.eye(4), "voxels", contract, phx.imaging.ImageAxisConvention.LPS
    )
    axis = np.linspace(0.1, size - 1.1, query_count)
    points = np.column_stack(
        (axis, np.mod(2.0 * axis, size - 1), np.mod(3.0 * axis, size - 1))
    )
    start = perf_counter()
    prepared = phx.spatial_sampling.VoxelObservationPlan(
        (size, size, size), affine, points, require_complete_coverage=True
    ).prepare()
    preparation = perf_counter() - start
    values = jnp.asarray(
        np.fromfunction(lambda i, j, k: i + 2.0 * j + 3.0 * k, (size, size, size))
    )
    execute = jax.jit(lambda operator, data: operator.apply(data).values)
    start = perf_counter()
    first_values = execute(prepared, values)
    jax.block_until_ready(first_values)
    compilation = perf_counter() - start
    start = perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(execute(prepared, values))
    elapsed = (perf_counter() - start) / repeats
    first = prepared.apply(values)
    expected = points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2]
    error = float(np.max(np.abs(np.asarray(first_values) - expected)))
    if error > 1.0e-10 or not bool(first.evidence.successful):
        raise RuntimeError("Image sampling benchmark violated affine-field exactness.")
    route_bytes = sum(value.nbytes for value in jax.tree.leaves(prepared.stencil))
    return {
        "size": size,
        "query_count": query_count,
        "preparation_seconds": preparation,
        "compile_seconds": compilation,
        "execution_seconds": elapsed,
        "route_bytes": route_bytes,
        "maximum_error": error,
    }


def conservative_transfer(count: int, repeats: int) -> dict[str, float | int]:
    indices = np.arange(count, dtype=np.int32)
    measures = np.ones((count,))
    transfer = phx.imaging.ConservativeVoxelCellTransfer(
        indices,
        indices,
        measures,
        measures,
        measures,
        source_id="benchmark-voxels",
        target_id="benchmark-cells",
    )
    values = jnp.linspace(0.0, 1.0, count)
    apply = jax.jit(lambda operator, data: operator.apply(data).values)
    start = perf_counter()
    first = apply(transfer, values)
    jax.block_until_ready(first)
    compilation = perf_counter() - start
    start = perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(apply(transfer, values))
    elapsed = (perf_counter() - start) / repeats
    error = float(jnp.max(jnp.abs(first - values)))
    evidence = transfer.apply(values).evidence
    if error > 1.0e-12 or not bool(evidence.successful):
        raise RuntimeError("Conservative transfer benchmark violated identity and mass.")
    storage = sum(value.nbytes for value in jax.tree.leaves(transfer))
    return {
        "entity_count": count,
        "compile_seconds": compilation,
        "execution_seconds": elapsed,
        "storage_bytes": storage,
        "maximum_error": error,
    }


def hdiv_tabulation(repeats: int) -> dict[str, float | int]:
    element = phx.discretization.fem.form_element(
        "tetrahedron", 2, 2, family="full", twist="twisted", proxy="flux"
    )
    points = jnp.asarray(((0.1, 0.2, 0.3), (0.25, 0.25, 0.25)))
    if element.tabulator is None:
        raise RuntimeError("BDM benchmark requires its native reference tabulator.")
    tabulate = jax.jit(element.tabulator)
    values, gradients = tabulate(points)
    jax.block_until_ready(values)
    start = perf_counter()
    for _ in range(repeats):
        values, gradients = tabulate(points)
        jax.block_until_ready(gradients)
    elapsed = (perf_counter() - start) / repeats
    divergence = jnp.trace(gradients, axis1=-2, axis2=-1)
    if not bool(jnp.all(jnp.isfinite(divergence))):
        raise RuntimeError("BDM benchmark produced non-finite divergence.")
    return {
        "dofs": element.local_dof_count,
        "point_count": len(points),
        "execution_seconds": elapsed,
    }


def _transport_worker(connection: Connection, args: argparse.Namespace) -> None:
    """Send lifecycle phases and one complete native record from a fresh process."""

    def phase(name: str) -> None:
        connection.send({"kind": "phase", "phase": name})

    try:
        record = native_transport(
            size=args.size,
            repeats=args.repeats,
            steps=args.steps,
            dt=args.dt,
            target_error=args.target_error,
            balance_tolerance=args.balance_tolerance,
            maximum_cells=args.maximum_cells,
            maximum_vertices=args.maximum_vertices,
            oracle_capacity=args.oracle_capacity,
            phase_callback=phase,
        )
        connection.send({"kind": "result", "record": record})
    finally:
        connection.close()


def _native_attempt(args: argparse.Namespace, attempt: int) -> dict[str, Any]:
    """Cancel the native process at the deadline; never discard a failed attempt."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_transport_worker, args=(sender, args))
    started = perf_counter()
    last_phase = "process-startup"
    timed_out = False
    record: dict[str, Any] | None = None
    process_error: dict[str, object] | None = None
    try:
        worker.start()
        sender.close()
        while record is None:
            remaining = args.timeout - (perf_counter() - started)
            if remaining <= 0.0 or not receiver.poll(remaining):
                timed_out = True
                break
            try:
                message = receiver.recv()
            except EOFError:
                break
            if message["kind"] == "phase":
                last_phase = message["phase"]
            elif message["kind"] == "result":
                record = message["record"]
            else:
                raise RuntimeError(
                    "Native campaign received an invalid lifecycle message."
                )
    except Exception as error:
        process_error = {"exception_type": type(error).__name__, "message": str(error)}
    finally:
        receiver.close()
        sender.close()
        if worker.pid is not None:
            if record is not None:
                worker.join(timeout=0.1)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.1)
            if worker.is_alive():
                worker.kill()
            worker.join()
    if record is None:
        record = {
            "scenario": "native-generated-transport",
            "successful": False,
            "status": "timeout" if timed_out else "failed",
            "failure": {
                "phase": last_phase,
                "exception_type": "TimeoutError" if timed_out else "WorkerExit",
                "message": (
                    "Native process was forcibly cancelled at the declared deadline."
                    if timed_out
                    else "Native process exited without a complete result record."
                ),
                "process_exitcode": worker.exitcode,
                **(process_error or {}),
            },
            "controls": {
                "size": args.size,
                "steps": args.steps,
                "dt_seconds": args.dt,
                "repeats": args.repeats,
                "maximum_cells": args.maximum_cells,
                "maximum_vertices": args.maximum_vertices,
                "oracle_capacity": args.oracle_capacity,
            },
            "targets": {
                "physical_error": args.target_error,
                "balance_tolerance": args.balance_tolerance,
            },
            "comparison": {"status": "not-requested"},
        }
    else:
        record["status"] = "passed" if record["successful"] else "failed"
    record["attempt"] = attempt
    record["deadline_seconds"] = args.timeout
    record["attempt_wall_seconds"] = perf_counter() - started
    return record


def native_campaign(args: argparse.Namespace) -> dict[str, Any]:
    records = [_native_attempt(args, index) for index in range(args.attempts)]
    passed = sum(record["status"] == "passed" for record in records)
    timed_out = sum(record["status"] == "timeout" for record in records)
    return {
        "scenario": "native-generated-transport",
        "attempts": records,
        "counts": {
            "attempted": len(records),
            "successful": passed,
            "failed": len(records) - passed - timed_out,
            "timed_out": timed_out,
        },
        "timeout_seconds": args.timeout,
        "successful": passed == len(records),
        "comparison": {"status": "not-requested"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_native_arguments(parser, default_scenario="synthetic")
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Independent cold native campaign attempts; every result is retained.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Hard native process deadline in seconds per attempt.",
    )
    args = parser.parse_args()
    validate_native_arguments(parser, args)
    if args.attempts < 1 or not np.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("attempts must be positive; timeout must be finite and positive")
    if args.queries < 1:
        parser.error("queries must be positive")
    if args.scenario == "native-generated-transport":
        record = native_campaign(args)
        print(json.dumps(record, indent=2))
        if not record["successful"]:
            raise SystemExit(1)
        return
    print(
        json.dumps(
            {
                "scenario": "synthetic",
                "image_sampling": image_sampling(args.size, args.queries, args.repeats),
                "conservative_transfer": conservative_transfer(
                    args.queries, args.repeats
                ),
                "hdiv_tabulation": hdiv_tabulation(args.repeats),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
