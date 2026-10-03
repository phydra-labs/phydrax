# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Durable, restartable production run of a meshfree advection-diffusion evolution.

The command advances a periodic GMLS reaction-diffusion evolution with the
native additive IMEX method inside bounded compiled production segments,
publishes scheduled outputs through a byte-bounded asynchronous publisher
(an intentionally slow archive exercises output backpressure), and commits
crash-consistent checkpoints whose identity inventory binds build source,
program, geometry, support, measure, capacity and precision. Kill it at any
point and rerun with ``--resume``: the run continues from the last committed
checkpoint and ends in the same accepted state, evidence cursors and outputs.

    python -m examples.meshfree_production_restart --root /tmp/run
    python -m examples.meshfree_production_restart --root /tmp/run --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from phydrax import solver
from phydrax.discretization import PointCloudPlan, PreparedPointCloudDiscretization
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    meshfree_runtime_inventory,
    MeshfreeDiffusionLaw,
    MeshfreeEvolutionPlan,
    MeshfreeReactionLaw,
    PreparedMeshfreeEvolution,
)
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.domain import HyperRectangle, PeriodicIdentification


_COUNT = 10
_STEP = 0.002
_END = 0.4
_OUTPUT_EVERY = 10
_UNIT_SQUARE = HyperRectangle(np.zeros(2), np.ones(2))
_PERIODIC = MortonAddressPlan.from_periodic_identifications(
    tuple(PeriodicIdentification(_UNIT_SQUARE, "x", component=axis) for axis in range(2)),
    maximum_depth=10,
)


def _reaction(
    time: jax.Array, points: jax.Array, value: jax.Array, args: Any
) -> jax.Array:
    del time, points, args
    return -0.5 * value


def prepare_evolution() -> PreparedMeshfreeEvolution:
    spacing = 1.0 / _COUNT
    axis = (np.arange(_COUNT) + 0.5) * spacing
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    jitter = np.random.default_rng(3).uniform(-1.0, 1.0, points.shape)
    cloud = PointCloudPlan(
        np.mod(points + 0.005 * spacing * jitter, 1.0),
        np.full(points.shape[0], spacing**2),
        stencil=LocalStencilPolicy(polynomial_degree=3),
        neighbors=21,
        address=_PERIODIC,
    ).prepare()
    return MeshfreeEvolutionPlan(
        cloud,
        diffusion=MeshfreeDiffusionLaw(0.05, law_id="diffusivity:0.05"),
        reaction=MeshfreeReactionLaw(_reaction, law_id="linear-decay:0.5"),
        plan_id="production-restart-example",
    ).prepare()


def _source_id() -> str:
    """Build identity of this driver: a changed program source refuses restart."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _digest(tree: Any) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        digest.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--archive-seconds", type=float, default=0.25)
    arguments = parser.parse_args(argv)
    root = arguments.root
    root.mkdir(mode=0o700, parents=True, exist_ok=True)

    evolution = prepare_evolution()
    spatial = evolution.plan.spatial
    if not isinstance(spatial, PreparedPointCloudDiscretization):
        raise RuntimeError(
            "The restart example requires a prepared point-cloud evolution."
        )
    method = evolution.imex_method("ars-222")
    retry = solver.RobustRetryPolicy(maximum_retries=1, reduction_factor=0.5)
    targets = np.arange(1, _END / (_OUTPUT_EVERY * _STEP) + 0.5) * _OUTPUT_EVERY * _STEP
    plan = solver.ProductionRunPlan(
        method,
        retry,
        step_size=_STEP,
        end_time=_END,
        maximum_steps=int(round(_END / _STEP)) + 10,
        checkpoint_interval=_OUTPUT_EVERY,
        segment_steps=5,
        output_schedule=solver.ExactTimeSchedule(jnp.asarray(targets)),
        moments=(
            solver.StreamingMomentPlan(
                lambda time, state, args: jnp.sum(evolution.fields(state).content),
                value_shape=(),
                plan_id="total-content",
            ),
        ),
    )
    inventory = meshfree_runtime_inventory(
        evolution,
        source=_source_id(),
        program=plan.plan_id,
        method=method,
        controller=retry.policy_id,
        precision="float64",
    )
    manifest = solver.ProductionCaseManifest.from_inventory(
        inventory, problem_id="meshfree-production-restart", dtype="float64"
    )
    store = solver.DurableCheckpointStore(
        root / "checkpoints", manifest, solver.CheckpointGenerationPolicy(2)
    )

    def archive(event_id: str, snapshot: Any) -> None:
        # A slow durable archive: the bounded publisher applies backpressure.
        time.sleep(arguments.archive_seconds)
        content = float(np.sum(np.asarray(evolution.fields(snapshot).content)))
        print(json.dumps({"output": event_id[:16], "content": content}), flush=True)

    publisher = solver.ByteBoundedAsyncPublisher(
        archive, maximum_pending=1, maximum_pending_bytes=1 << 20
    )
    prepared = solver.PreparedProductionRun(manifest, plan, store, publisher=publisher)
    initial = prepared.initial_state(
        evolution.initial_state(1.0 + 0.5 * jnp.sin(2.0 * jnp.pi * spatial.points[:, 0]))
    )
    state = prepared.resume(initial) if arguments.resume else initial
    print(
        json.dumps(
            {
                "start": "resume" if arguments.resume else "fresh",
                "step": int(state.step_index),
                "inventory": inventory.inventory_id,
                "pid": os.getpid(),
            }
        ),
        flush=True,
    )
    result = prepared.run(state, memory_sampling_interval=0.05)
    publisher.close()
    evidence = result.state.evidence
    if evidence is None:
        raise RuntimeError("Terminal evidence retention was requested.")
    memory = result.memory
    print(
        json.dumps(
            {
                "status": result.state.status,
                "step": int(result.state.step_index),
                "time": float(result.state.time),
                "state_sha256": _digest(result.state.accepted_state),
                "moments_sha256": _digest(result.state.moment_states),
                "evidence_sha256": _digest(evidence),
                "accepted_step": int(evidence.accepted_step),
                "refused_attempts": int(evidence.refused_attempts),
                "output_cursor": int(result.state.output_cursor),
                "sampled_peak_resident_bytes": None
                if memory is None
                else memory.sampled_peak_resident.value_bytes,
            }
        ),
        flush=True,
    )
    return 0 if bool(result.successful) else 1


if __name__ == "__main__":
    sys.exit(main())
