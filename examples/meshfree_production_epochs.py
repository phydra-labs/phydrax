# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Durable multi-epoch meshfree production run across support-epoch rebases.

A uniform ALE drift exhausts the trust of each fixed GMLS support after a few
SSPRK33 steps. The exhausted step is refused, the run ends ``step-rejected``
and checkpoints the held state; the driver then rebases the support at that
held state and continues in the next epoch through ``support_epoch_relation``.
That restore commits the epoch receipt and the held anchor with the repository
checkpoint lineage, atomically with the first checkpoint of the new epoch.
Kill the command at any point and rerun it with ``--resume``: a fresh process
re-derives the current epoch with ``resume_support_epochs`` and continues
through the identity relation to the same accepted state, moments, evidence
and outputs as an uninterrupted run.

    python -m examples.meshfree_production_epochs --root /tmp/run
    python -m examples.meshfree_production_epochs --root /tmp/run --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from phydrax import execution, solver
from phydrax.discretization import PointCloudPlan, PreparedPointCloudDiscretization
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    meshfree_runtime_inventory,
    MeshfreeDiffusionLaw,
    MeshfreeEvolutionPlan,
    MeshfreeMotion,
    PreparedMeshfreeEvolution,
    resume_support_epochs,
    support_epoch_relation,
)
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.domain import HyperRectangle, PeriodicIdentification
from phydrax.lifecycle import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
    ResolvedRunSpec,
)
from phydrax.qualification import SupportDependency


_COUNT = 8
_END_STEPS = 16
_OUTPUT_EVERY = 2
_RNG = "threefry:fold-in(accepted-step)"
_RETRY = solver.RobustRetryPolicy(maximum_retries=2, reduction_factor=0.5)
_ARTIFACT = "meshfree-production-epochs"
_UNIT_SQUARE = HyperRectangle(np.zeros(2), np.ones(2))
_PERIODIC = MortonAddressPlan.from_periodic_identifications(
    tuple(PeriodicIdentification(_UNIT_SQUARE, "x", component=axis) for axis in range(2)),
    maximum_depth=10,
)


def prepare_cloud() -> PreparedPointCloudDiscretization:
    spacing = 1.0 / _COUNT
    axis = (np.arange(_COUNT) + 0.5) * spacing
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    jitter = np.random.default_rng(3).uniform(-1.0, 1.0, points.shape)
    return PointCloudPlan(
        np.mod(points + 0.005 * spacing * jitter, 1.0),
        np.full(points.shape[0], spacing**2),
        stencil=LocalStencilPolicy(polynomial_degree=2),
        neighbors=13,
        address=_PERIODIC,
    ).prepare()


def _drift(time: jax.Array, points: jax.Array, args: Any) -> jax.Array:
    del time, args
    return jnp.broadcast_to(jnp.asarray([1.0, 0.0]), points.shape)


def prepare_evolution(
    cloud: PreparedPointCloudDiscretization,
) -> PreparedMeshfreeEvolution:
    return MeshfreeEvolutionPlan(
        cloud,
        diffusion=MeshfreeDiffusionLaw(0.05, law_id="diffusivity:0.05"),
        motion=MeshfreeMotion("ale", mesh_velocity=_drift, law_id="uniform-drift"),
        plan_id="production-epochs-example",
    ).prepare()


def _source_id() -> str:
    """Build identity of this driver: a changed program source refuses restart."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _digest(tree: Any) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        digest.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class _Epoch:
    """One support epoch's prepared run: plan, identity inventory and manifest."""

    evolution: PreparedMeshfreeEvolution
    plan: solver.ProductionRunPlan
    inventory: solver.RuntimeIdentityInventory
    manifest: solver.ProductionCaseManifest


def _epoch(evolution: PreparedMeshfreeEvolution, step: float, source: str) -> _Epoch:
    method = evolution.ssprk_method("ssprk33")
    targets = jnp.arange(1, _END_STEPS // _OUTPUT_EVERY + 1) * _OUTPUT_EVERY * step
    plan = solver.ProductionRunPlan(
        method,
        _RETRY,
        step_size=step,
        end_time=_END_STEPS * step,
        maximum_steps=4 * _END_STEPS,
        checkpoint_interval=_OUTPUT_EVERY,
        segment_steps=_OUTPUT_EVERY,
        output_schedule=solver.ExactTimeSchedule(targets),
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
        source=source,
        program=plan.plan_id,
        method=method,
        controller=_RETRY.policy_id,
        precision="float64",
        rng=_RNG,
    )
    manifest = solver.ProductionCaseManifest.from_inventory(
        inventory, problem_id="meshfree-production-epochs", dtype="float64"
    )
    return _Epoch(evolution, plan, inventory, manifest)


@dataclass(frozen=True)
class _Repository:
    repository: POSIXArtifactRepository
    resolved: ResolvedRunSpec
    request: execution.ResourceRequest
    policy: solver.CheckpointGenerationPolicy


def _repository(root: Path, /) -> _Repository:
    repository = POSIXArtifactRepository(
        root.resolve(),
        POSIXRepositoryPolicy(
            HPCFilesystemProfile(
                "meshfree-production-epochs-posix",
                "local-filesystem",
                atomic_rename_same_filesystem=True,
                file_fsync=True,
                directory_fsync=True,
                advisory_locking=True,
                attempt_private_staging=True,
            ),
            maximum_chunk_bytes=1 << 16,
            maximum_metadata_bytes=1 << 20,
        ),
    )
    request = execution.ResourceRequest(
        cpu_cores=1,
        memory_bytes=64 << 20,
        maximum_checkpoint_staging_bytes=16 << 20,
        maximum_output_backlog_bytes=8 << 20,
    )
    policy = solver.CheckpointGenerationPolicy(3)
    dependency = SupportDependency(
        "meshfree-production-epochs-repository",
        repository.support_tuple.support_tuple_id,
    )
    resolved = ResolvedRunSpec(
        (),
        (dependency,),
        release_index_id="release-index",
        profile_ids=(dependency.profile_id,),
        trust_policy_id="trust-policy",
        valid_at=10,
        valid_from=0,
        valid_until=20,
        prepared_configuration_id="meshfree-production-epochs-configuration",
        precision_policy_id="float64",
        resource_policy_id=request.resource_id,
        checkpoint_policy_id=policy.policy_id,
        output_policy_id="ordered-outbox",
        repository_id=repository.provider_id,
        scheduler_id="scheduler",
        auth_policy_id="auth-policy",
    )
    return _Repository(repository, resolved, request, policy)


def _store(repository: _Repository, epoch: _Epoch, /) -> solver.ArtifactCheckpointStore:
    """A fresh store over the run's one checkpoint artifact."""
    return solver.ArtifactCheckpointStore(
        repository.repository,
        epoch.manifest,
        repository.policy,
        repository.resolved,
        writer_id="meshfree-production-epochs",
        resource_request=repository.request,
        artifact_id=_ARTIFACT,
    )


def _runtime(
    repository: _Repository,
    epoch: _Epoch,
    publisher: solver.ByteBoundedAsyncPublisher,
    /,
    *,
    relation: solver.RuntimeRestartRelation | None = None,
) -> tuple[solver.PreparedProductionRun, solver.ArtifactCheckpointStore]:
    store = _store(repository, epoch)
    runtime = solver.PreparedProductionRun(
        epoch.manifest,
        epoch.plan,
        store,
        publisher=publisher,
        resolved_run_spec=repository.resolved,
        restart_relation=relation,
    )
    return runtime, store


def _initial(runtime: solver.PreparedProductionRun, state: jax.Array) -> Any:
    return runtime.initial_state(
        state,
        controller_state=jnp.asarray(0, dtype=jnp.int64),
        rng_state=jax.random.key_data(jax.random.key(11)),
    )


@dataclass
class _Progress:
    """Support epoch the driver is currently advancing (labels outputs)."""

    epoch: int


def _emit(record: dict[str, Any]) -> None:
    print(json.dumps(record), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--archive-seconds", type=float, default=0.25)
    arguments = parser.parse_args(argv)
    root = arguments.root
    root.mkdir(mode=0o700, parents=True, exist_ok=True)

    cloud = prepare_cloud()
    step = float(jnp.min(cloud.trust_radius)) / 5.5
    source = _source_id()
    repository = _repository(root / "repository")
    first = _epoch(prepare_evolution(cloud), step, source)
    progress = _Progress(0)

    def archive(event_id: str, snapshot: Any) -> None:
        # A slow durable archive: outputs reach it only after their checkpoint.
        time.sleep(arguments.archive_seconds)
        content = float(np.sum(np.asarray(first.evolution.fields(snapshot).content)))
        _emit({"output": event_id[:16], "epoch": progress.epoch, "content": content})

    publisher = solver.ByteBoundedAsyncPublisher(
        archive, maximum_pending=1, maximum_pending_bytes=1 << 20
    )
    current = first
    if arguments.resume:
        chain = resume_support_epochs(
            first.evolution,
            _store(repository, first),
            lambda evolution: _epoch(evolution, step, source).inventory,
        )
        current = _epoch(chain.evolution, step, source)
        progress.epoch = chain.epoch
    runtime, store = _runtime(repository, current, publisher)
    initial = _initial(
        runtime,
        current.evolution.initial_state(
            1.0 + 0.5 * jnp.sin(2.0 * jnp.pi * cloud.points[:, 0])
        ),
    )
    state = runtime.resume(initial) if arguments.resume else initial
    # A checkpoint whose committed terminal is the support-exhaustion refusal is
    # held at an epoch boundary: rebase instead of repeating the refused step.
    terminal = store.terminal_record
    held = (
        state
        if terminal is not None and terminal["failure_category"] == "step-rejected"
        else None
    )
    _emit(
        {
            "start": "resume" if arguments.resume else "fresh",
            "epoch": progress.epoch,
            "step": int(state.step_index),
            "inventory": current.inventory.inventory_id,
            "pid": os.getpid(),
        }
    )
    while True:
        if held is None:
            result = runtime.run(state)
            failure = result.failure
            if failure is None or failure.category != "step-rejected":
                break
            held = result.state
        anchor = held.accepted_state
        successor = _epoch(current.evolution.rebase(anchor), step, source)
        relation = support_epoch_relation(
            current.inventory,
            successor.inventory,
            current.evolution,
            successor.evolution,
            anchor,
            source_template=anchor,
        )
        runtime, store = _runtime(repository, successor, publisher, relation=relation)
        state = runtime.resume(_initial(runtime, current.evolution.repacked(anchor)))
        current = successor
        progress.epoch += 1
        held = None
        _emit({"epoch": progress.epoch, "anchor_step": int(state.step_index)})
    publisher.close()
    evidence = result.state.evidence
    if evidence is None:
        raise RuntimeError("Terminal evidence retention was requested.")
    _emit(
        {
            "status": result.state.status,
            "epoch": progress.epoch,
            "step": int(result.state.step_index),
            "time": float(result.state.time),
            "state_sha256": _digest(result.state.accepted_state),
            "moments_sha256": _digest(result.state.moment_states),
            "evidence_sha256": _digest(evidence),
            "accepted_step": int(evidence.accepted_step),
            "refused_attempts": int(evidence.refused_attempts),
            "output_cursor": int(result.state.output_cursor),
        }
    )
    return 0 if bool(result.successful) else 1


if __name__ == "__main__":
    sys.exit(main())
