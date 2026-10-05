# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Interrupted meshfree production runs resume exactly where they were committed.

Every scenario compares an uninterrupted production run with one interrupted
after accepted steps and resumed from durable checkpoints through fresh owners
and stores. Reproducibility is declared bitwise for same-topology and
support-epoch restarts (the same compiled program on identical values) and
for the transport of an ownership migration; continued evolution on a changed
owner partition is compared under a declared reduction-order tolerance.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.meshfree_bulk_surface_exchange import (
    prepare_workflow as prepare_bulk_surface,
)
from examples.meshfree_moving_surface_reaction_diffusion import (
    prepare_workflow as prepare_moving_surface,
)
from phydrax import ArrayArchiveCorruptionError, execution, solver
from phydrax.discretization import PointCloudPlan, PreparedPointCloudDiscretization
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    meshfree_runtime_inventory,
    MeshfreeDiffusionLaw,
    MeshfreeEvolutionPlan,
    MeshfreeEvolutionStatus,
    MeshfreeMotion,
    moving_archive_policy,
    MovingSurfaceFixedStepMethod,
    MovingSurfacePlan,
    MovingSurfaceStatus,
    PreparedMeshfreeEvolution,
    resume_support_epochs,
    support_epoch_relation,
)
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.domain import HyperRectangle, PeriodicIdentification
from phydrax.interfacial_transport import FilmStepStatus
from phydrax.lifecycle import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
    ResolvedRunSpec,
)
from phydrax.qualification import SupportDependency
from phydrax.solver.coupling import MeshfreeBulkSurfaceMethod


_ROOT = Path(__file__).resolve().parents[2]
_PYTHON = sys.executable
_SOURCE = "meshfree-runtime-restart-test-build"
_RNG = "threefry:fold-in(accepted-step)"
_UNIT_SQUARE = HyperRectangle(np.zeros(2), np.ones(2))
_PERIODIC = MortonAddressPlan.from_periodic_identifications(
    tuple(PeriodicIdentification(_UNIT_SQUARE, "x", component=axis) for axis in range(2)),
    maximum_depth=10,
)
_RETRY = solver.RobustRetryPolicy(maximum_retries=2, reduction_factor=0.5)


def _cloud(count: int = 8, *, seed: int = 3) -> PreparedPointCloudDiscretization:
    spacing = 1.0 / count
    axis = (np.arange(count) + 0.5) * spacing
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    jitter = np.random.default_rng(seed).uniform(-1.0, 1.0, points.shape)
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


def _evolution(cloud: PreparedPointCloudDiscretization) -> PreparedMeshfreeEvolution:
    """ALE drift: the fixed support exhausts its trust after a few steps."""
    return MeshfreeEvolutionPlan(
        cloud,
        diffusion=MeshfreeDiffusionLaw(0.05, law_id="diffusivity:0.05"),
        motion=MeshfreeMotion("ale", mesh_velocity=_drift, law_id="uniform-drift"),
        plan_id="meshfree-runtime-restart",
    ).prepare()


def _step(cloud: PreparedPointCloudDiscretization) -> float:
    return float(jnp.min(cloud.trust_radius)) / 5.5


@dataclass(frozen=True)
class _Leg:
    evolution: PreparedMeshfreeEvolution
    plan: solver.ProductionRunPlan
    inventory: solver.RuntimeIdentityInventory
    manifest: solver.ProductionCaseManifest


def _leg(
    evolution: PreparedMeshfreeEvolution,
    step: float,
    /,
    *,
    end_steps: int = 10,
    source: str = _SOURCE,
    precision: str = "float64",
) -> _Leg:
    method = evolution.ssprk_method("ssprk33")
    end = end_steps * step
    plan = solver.ProductionRunPlan(
        method,
        _RETRY,
        step_size=step,
        end_time=end,
        maximum_steps=4 * end_steps,
        checkpoint_interval=2,
        segment_steps=2,
        output_schedule=solver.ExactTimeSchedule(
            jnp.arange(1, end_steps // 4 + 1) * 4 * step
        ),
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
        precision=precision,
        rng=_RNG,
    )
    manifest = solver.ProductionCaseManifest.from_inventory(
        inventory, problem_id="meshfree-ale-drift", dtype="float64"
    )
    return _Leg(evolution, plan, inventory, manifest)


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
                "meshfree-runtime-posix",
                "test-filesystem",
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
        "meshfree-runtime-repository", repository.support_tuple.support_tuple_id
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
        prepared_configuration_id="meshfree-runtime-configuration",
        precision_policy_id="float64",
        resource_policy_id=request.resource_id,
        checkpoint_policy_id=policy.policy_id,
        output_policy_id="ordered-outbox",
        repository_id=repository.provider_id,
        scheduler_id="scheduler",
        auth_policy_id="auth-policy",
    )
    return _Repository(repository, resolved, request, policy)


class _Archive:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.publisher = solver.ByteBoundedAsyncPublisher(
            lambda event_id, snapshot: self.events.append(event_id),
            maximum_pending=2,
            maximum_pending_bytes=1 << 20,
        )


def _runtime(
    repository: _Repository,
    leg: _Leg,
    archive: _Archive,
    /,
    *,
    relation: solver.RuntimeRestartRelation | None = None,
) -> solver.PreparedProductionRun:
    """Fresh store and runtime objects, as a restarted process builds them."""
    store = solver.ArtifactCheckpointStore(
        repository.repository,
        leg.manifest,
        repository.policy,
        repository.resolved,
        writer_id="meshfree-runtime",
        resource_request=repository.request,
        artifact_id="meshfree-ale-drift",
    )
    return solver.PreparedProductionRun(
        leg.manifest,
        leg.plan,
        store,
        publisher=archive.publisher,
        resolved_run_spec=repository.resolved,
        restart_relation=relation,
    )


def _initial(runtime: solver.PreparedProductionRun, state: Any) -> Any:
    return runtime.initial_state(
        state,
        controller_state=jnp.asarray(0, dtype=jnp.int64),
        rng_state=jax.random.key_data(jax.random.key(11)),
    )


def _interrupt(runtime: solver.PreparedProductionRun, state: Any, steps: int, /) -> None:
    """Advance accepted steps, commit, and abandon every live object."""
    for _ in range(steps):
        state, transition = runtime.step(state)
        assert bool(transition.successful)
    runtime.checkpoint(state)


@dataclass(frozen=True)
class _Campaign:
    held: Any
    held_failure: solver.ProductionFailureRecord
    final: solver.ProductionRunResult
    replay_classification: str | None


def _campaign(root: Path, *, interrupt: bool) -> _Campaign:
    """Support-exhaustion refusal ends epoch 0; a rebased epoch completes the run."""
    cloud = _cloud()
    step = _step(cloud)
    repository = _repository(root)
    first = _leg(_evolution(cloud), step)
    runtime = _runtime(repository, first, _Archive())
    start = _initial(
        runtime,
        first.evolution.initial_state(
            1.0 + 0.5 * jnp.sin(2.0 * jnp.pi * cloud.points[:, 0])
        ),
    )
    if interrupt:
        _interrupt(runtime, start, 3)
        runtime = _runtime(repository, first, _Archive())
        start = runtime.resume(start)
        assert int(start.step_index) == 3
    held = runtime.run(start)
    assert held.failure is not None
    successor = first.evolution.rebase(held.state.accepted_state)
    second = _leg(successor, step)
    relation = support_epoch_relation(
        first.inventory,
        second.inventory,
        first.evolution,
        successor,
        held.state.accepted_state,
        source_template=held.state.accepted_state,
    )
    runtime = _runtime(repository, second, _Archive(), relation=relation)
    template = _initial(runtime, first.evolution.repacked(held.state.accepted_state))
    resumed = runtime.resume(template)
    classification = runtime.last_replay_classification
    if interrupt:
        _interrupt(runtime, resumed, 2)
        # Same-epoch restart binds the identity relation of the second epoch.
        runtime = _runtime(repository, second, _Archive())
        resumed = runtime.resume(template)
    final = runtime.run(resumed)
    return _Campaign(held.state, held.failure, final, classification)


def test_support_epoch_relation_refuses_a_checkpoint_other_than_its_anchor(
    tmp_path: Path,
) -> None:
    cloud = _cloud()
    step = _step(cloud)
    first = _leg(_evolution(cloud), step)
    repository = _repository(tmp_path)
    original = _runtime(repository, first, _Archive())
    original_store = original.checkpoint_store
    assert isinstance(original_store, solver.ArtifactCheckpointStore)
    state = _initial(
        original,
        first.evolution.initial_state(jnp.ones(cloud.points.shape[0], dtype=jnp.float64)),
    )
    for _ in range(2):
        state, transition = original.step(state)
        assert bool(transition.successful)
    archived = original.checkpoint(state)
    anchor, transition = original.step(archived)
    assert bool(transition.successful)
    assert int(archived.step_index) == 2
    assert int(anchor.step_index) == 3

    successor = first.evolution.rebase(anchor.accepted_state)
    second = _leg(successor, step)
    relation = support_epoch_relation(
        first.inventory,
        second.inventory,
        first.evolution,
        successor,
        anchor.accepted_state,
        source_template=archived.accepted_state,
    )
    target = _runtime(repository, second, _Archive(), relation=relation)
    target_store = target.checkpoint_store
    assert isinstance(target_store, solver.ArtifactCheckpointStore)
    template = _initial(target, first.evolution.repacked(anchor.accepted_state))
    source_manifest = repository.repository.get_manifest(original_store.artifact_id)
    with pytest.raises(
        ValueError, match="support-epoch source anchor differs from the migration receipt"
    ):
        target.resume(template)

    assert (
        repository.repository.get_manifest(original_store.artifact_id).manifest_id
        == source_manifest.manifest_id
    )
    assert target_store.migration_lineage() == ()
    source_restart = _runtime(repository, first, _Archive())
    restored_source = source_restart.resume(state)
    assert int(restored_source.step_index) == 2
    _bitwise(restored_source.accepted_state, archived.accepted_state)

    # The same admitted relation succeeds once its actual anchor is committed.
    original.checkpoint(anchor)
    resumed = target.resume(template)
    assert int(resumed.step_index) == 3
    assert target.last_replay_classification == "bitwise"
    _bitwise(resumed.accepted_state, first.evolution.repacked(anchor.accepted_state))

    fresh = _runtime(repository, first, _Archive())
    fresh_store = fresh.checkpoint_store
    assert isinstance(fresh_store, solver.ArtifactCheckpointStore)
    chain = resume_support_epochs(
        first.evolution,
        fresh_store,
        lambda evolution: _leg(evolution, step).inventory,
    )
    assert chain.epoch == 1
    assert chain.inventory.inventory_id == second.inventory.inventory_id
    restarted = _runtime(repository, _leg(chain.evolution, step), _Archive())
    replayed = restarted.resume(_initial(restarted, resumed.accepted_state))
    assert restarted.last_replay_classification == "bitwise"
    _equal_runs(resumed, replayed)
    _bitwise(
        (
            resumed.controller_state,
            resumed.trigger_states,
            resumed.output_cursor,
            resumed.rng_state,
        ),
        (
            replayed.controller_state,
            replayed.trigger_states,
            replayed.output_cursor,
            replayed.rng_state,
        ),
    )


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory: pytest.TempPathFactory) -> _Campaign:
    return _campaign(tmp_path_factory.mktemp("uninterrupted"), interrupt=False)


@pytest.fixture(scope="module")
def interrupted(tmp_path_factory: pytest.TempPathFactory) -> _Campaign:
    return _campaign(tmp_path_factory.mktemp("interrupted"), interrupt=True)


def test_support_exhaustion_is_a_refused_epoch_with_retained_refusal_evidence(
    uninterrupted: _Campaign,
) -> None:
    held, failure = uninterrupted.held, uninterrupted.held_failure
    assert failure.category == "step-rejected"
    evidence = held.evidence
    assert evidence is not None
    # Every retry of the refused step is a refusal; earlier steps were accepted.
    assert int(evidence.refused_step) == int(held.step_index)
    assert int(evidence.refused_attempt) == _RETRY.maximum_retries
    assert int(evidence.refused_attempts) >= _RETRY.maximum_retries + 1
    assert int(evidence.accepted_step) == int(held.step_index) - 1
    # The retained records are the native admissions of both outcomes.
    exceeded = int(MeshfreeEvolutionStatus.SUPPORT_EXCEEDED)
    assert int(evidence.refused.admission.status) == exceeded
    assert not bool(evidence.refused.admission.support_accepted)
    assert int(evidence.accepted.admission.status) == int(
        MeshfreeEvolutionStatus.ACCEPTED
    )
    assert uninterrupted.replay_classification == "bitwise"
    assert uninterrupted.final.state.status == "completed"
    # The refusal record survives the epoch migration into the completed run.
    final = uninterrupted.final.state.evidence
    assert final is not None
    assert int(final.refused_step) == int(held.step_index)
    assert int(final.refused_attempts) == int(evidence.refused_attempts)
    assert int(final.refused.admission.status) == exceeded
    assert int(final.accepted.admission.status) == int(MeshfreeEvolutionStatus.ACCEPTED)


def _bitwise(reference: Any, restarted: Any, /) -> None:
    """Equal trees, leaf by leaf; a fail-closed candidate's NaN evidence included."""
    assert jax.tree.structure(restarted) == jax.tree.structure(reference)
    jax.tree.map(np.testing.assert_array_equal, restarted, reference)


def test_interrupted_run_matches_uninterrupted_across_the_refused_epoch(
    uninterrupted: _Campaign, interrupted: _Campaign
) -> None:
    np.testing.assert_array_equal(
        interrupted.held.accepted_state, uninterrupted.held.accepted_state
    )
    _bitwise(uninterrupted.held.evidence, interrupted.held.evidence)
    reference, restarted = uninterrupted.final.state, interrupted.final.state
    assert restarted.status == reference.status == "completed"
    assert int(restarted.step_index) == int(reference.step_index)
    np.testing.assert_array_equal(restarted.time, reference.time)
    np.testing.assert_array_equal(restarted.accepted_state, reference.accepted_state)
    assert eqx.tree_equal(restarted.moment_states, reference.moment_states)
    _bitwise(reference.evidence, restarted.evidence)
    assert eqx.tree_equal(restarted.trigger_states, reference.trigger_states)
    assert int(restarted.schedule_cursor) == int(reference.schedule_cursor)
    assert int(restarted.output_cursor) == int(reference.output_cursor)
    np.testing.assert_array_equal(restarted.rng_state, reference.rng_state)
    np.testing.assert_array_equal(restarted.controller_state, reference.controller_state)
    assert interrupted.replay_classification == "bitwise"


def _durable(
    root: Path, plan: solver.ProductionRunPlan, manifest: solver.ProductionCaseManifest
) -> solver.PreparedProductionRun:
    return solver.PreparedProductionRun(
        manifest,
        plan,
        solver.DurableCheckpointStore(
            root, manifest, solver.CheckpointGenerationPolicy(2)
        ),
        publisher=_Archive().publisher,
    )


@pytest.fixture(scope="module")
def committed(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, _Leg, Any]:
    root = tmp_path_factory.mktemp("stale") / "checkpoints"
    cloud = _cloud()
    leg = _leg(_evolution(cloud), _step(cloud))
    runtime = _durable(root, leg.plan, leg.manifest)
    state = runtime.initial_state(
        leg.evolution.initial_state(jnp.ones(cloud.points.shape[0]))
    )
    runtime.checkpoint(state)
    return root, leg, state


def _stale_leg(change: str) -> _Leg:
    match change:
        case "geometry":
            cloud = _cloud(seed=4)
            return _leg(_evolution(cloud), _step(_cloud()))
        case "capacity":
            cloud = _cloud(7)
            return _leg(_evolution(cloud), _step(_cloud()))
        case "precision":
            cloud = _cloud()
            return _leg(_evolution(cloud), _step(cloud), precision="float64-mixed-policy")
        case "source":
            cloud = _cloud()
            return _leg(_evolution(cloud), _step(cloud), source="another-build")
        case "program":
            cloud = _cloud()
            return _leg(_evolution(cloud), _step(cloud), end_steps=12)
        case _:
            raise ValueError(change)


@pytest.mark.parametrize(
    "change", ("geometry", "capacity", "precision", "source", "program")
)
def test_stale_identity_refuses_before_any_checkpoint_value_is_used(
    committed: tuple[Path, _Leg, Any], change: str
) -> None:
    root, _, _ = committed
    before = sorted((path.name, path.stat().st_mtime_ns) for path in root.iterdir())
    leg = _stale_leg(change)
    runtime = _durable(root, leg.plan, leg.manifest)
    template = runtime.initial_state(
        leg.evolution.initial_state(jnp.ones(leg.evolution.point_count))
    )
    with pytest.raises(solver.StaleRuntimeCheckpointError) as refused:
        runtime.resume(template)
    assert change in refused.value.roles
    assert (
        sorted((path.name, path.stat().st_mtime_ns) for path in root.iterdir()) == before
    )


def test_truncated_and_foreign_payloads_refuse(
    committed: tuple[Path, _Leg, Any], tmp_path: Path
) -> None:
    root, leg, state = committed
    truncated = tmp_path / "truncated"
    truncated.mkdir(mode=0o700)
    for path in root.iterdir():
        (truncated / path.name).write_bytes(path.read_bytes())
    archive = next(truncated.glob("generation-*.phx"))
    archive.write_bytes(archive.read_bytes()[: archive.stat().st_size // 2])
    with pytest.raises(ArrayArchiveCorruptionError):
        _durable(truncated, leg.plan, leg.manifest).resume(state)

    # A checkpoint of another case with the same discretization identities.
    foreign_manifest = solver.ProductionCaseManifest.from_inventory(
        leg.inventory, problem_id="another-case", dtype="float64"
    )
    foreign = tmp_path / "foreign"
    _durable(foreign, leg.plan, foreign_manifest).checkpoint(state)
    with pytest.raises(ValueError, match="another store"):
        _durable(foreign, leg.plan, leg.manifest).resume(state)


def _equal_runs(reference: Any, restarted: Any) -> None:
    assert restarted.status == reference.status
    assert int(restarted.step_index) == int(reference.step_index)
    np.testing.assert_array_equal(restarted.time, reference.time)
    assert eqx.tree_equal(restarted.accepted_state, reference.accepted_state)
    assert eqx.tree_equal(restarted.evidence, reference.evidence)
    assert eqx.tree_equal(restarted.moment_states, reference.moment_states)


def _same_topology_restart(
    root: Path,
    plan: solver.ProductionRunPlan,
    manifest: solver.ProductionCaseManifest,
    initial: Any,
    steps: int,
    /,
) -> tuple[solver.ProductionRunResult, solver.ProductionRunResult]:
    reference = _durable(root / "reference", plan, manifest)
    uninterrupted = reference.run(reference.initial_state(initial))
    runtime = _durable(root / "restarted", plan, manifest)
    start = runtime.initial_state(initial)
    _interrupt(runtime, start, steps)
    restarted = _durable(root / "restarted", plan, manifest)
    return uninterrupted, restarted.run(restarted.resume(start))


def test_coupled_bulk_surface_restart_keeps_refused_newton_evidence(
    tmp_path: Path,
) -> None:
    workflow = prepare_bulk_surface(size=128, dimension=3, seed=0)
    original = workflow.method
    kinetics = original.transport.structure.kinetics
    if kinetics is None:
        raise TypeError("Langmuir exchange requires its actual kinetic owner.")
    # A tight native Newton budget refuses long steps; reduced steps converge.
    method = MeshfreeBulkSurfaceMethod(
        original.query,
        workflow.graph,
        original.bulk_volumes,
        kinetics,
        surface_diffusivity=0.02,
        maximum_iterations=6,
    )
    retry = solver.RobustRetryPolicy(maximum_retries=4, reduction_factor=0.125)
    plan = solver.ProductionRunPlan(
        method,
        retry,
        step_size=8.0,
        end_time=4.0,
        maximum_steps=16,
        checkpoint_interval=2,
        segment_steps=2,
    )
    inventory = meshfree_runtime_inventory(
        method,
        source=_SOURCE,
        program=plan.plan_id,
        method=method,
        controller=retry.policy_id,
        precision="float64",
    )
    manifest = solver.ProductionCaseManifest.from_inventory(
        inventory, problem_id="bulk-surface-langmuir", dtype="float64"
    )
    reference, restarted = _same_topology_restart(
        tmp_path, plan, manifest, workflow.initial, 2
    )
    assert reference.state.status == "completed"
    evidence = reference.state.evidence
    assert evidence is not None and int(evidence.refused_attempts) > 0
    assert int(evidence.refused.status) == int(FilmStepStatus.SOLVE_FAILED)
    assert int(evidence.accepted.status) == int(FilmStepStatus.ACCEPTED)
    _equal_runs(reference.state, restarted.state)


def test_moving_surface_restart_keeps_live_history_and_archive_cursor(
    tmp_path: Path,
) -> None:
    rolling, initial = prepare_moving_surface(size=48, dimension=3, seed=0)
    plan = MovingSurfacePlan(
        rolling.geometry,
        rolling.motion,
        rolling.reaction,
        method=rolling.tableau,
        epoch=rolling.epoch,
        capacity=rolling.capacity,
        history_capacity=5,
        archive="acknowledged",
        plan_id="growing-sphere:acknowledged-archive",
    )
    # The live-history ring is part of the state layout: initialize on this plan.
    state = plan.initialize(initial.points, initial.concentration)
    method = MovingSurfaceFixedStepMethod(plan)
    archive = moving_archive_policy(plan)
    retry = solver.RobustRetryPolicy(maximum_retries=0)
    with pytest.raises(ValueError, match="history window"):
        solver.ProductionRunPlan(
            method,
            retry,
            step_size=0.01,
            end_time=0.12,
            maximum_steps=12,
            checkpoint_interval=5,
            archive=archive,
        )
    run_plan = solver.ProductionRunPlan(
        method,
        retry,
        step_size=0.01,
        end_time=0.12,
        maximum_steps=12,
        checkpoint_interval=4,
        segment_steps=2,
        archive=archive,
    )
    inventory = meshfree_runtime_inventory(
        plan,
        source=_SOURCE,
        program=run_plan.plan_id,
        method=method,
        controller=archive.policy_id,
        precision="float64",
    )
    manifest = solver.ProductionCaseManifest.from_inventory(
        inventory, problem_id="growing-sphere", dtype="float64"
    )
    reference, restarted = _same_topology_restart(tmp_path, run_plan, manifest, state, 3)
    final = reference.state.accepted_state
    refused = reference.state.evidence
    assert reference.state.status == "completed", (
        reference.failure,
        None if refused is None else int(refused.refused.status),
        int(reference.state.step_index),
    )
    # Twelve steps through a five-slot ring: only acknowledgement admits them.
    assert int(final.accepted_steps) == 12
    assert int(final.history_count) == 5
    assert int(final.archive_cursor) == 13
    evidence = reference.state.evidence
    assert evidence is not None
    assert int(evidence.accepted.status) == int(MovingSurfaceStatus.ACCEPTED)
    _equal_runs(reference.state, restarted.state)


def test_ownership_migration_restart_in_a_four_device_process(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    environment["JAX_ENABLE_X64"] = "1"
    completed = subprocess.run(
        [_PYTHON, "-m", "tests.integration._meshfree_runtime_migration", str(tmp_path)],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["devices"] == 4
    assert report["migration_committed"] is True
    assert report["source_partition"] != report["target_partition"]
    # Ownership transport moves placement only: values, IDs and geometry stay.
    assert "partition" in report["migrated_roles"]
    assert set(report["migrated_roles"]) <= {"partition", "program", "capacity"}
    assert report["replay_classification"] == "bitwise"
    assert report["restored_bitwise"] is True
    assert report["pending_output_bitwise"] is True
    assert report["retained_output_bitwise"] is True
    assert report["identity_retained_output_bitwise"] is True
    assert report["identity_restart_no_reack"] is True
    assert report["identity_checkpoint_phase"] == "restart-lineage"
    assert report["identity_repeat_manifest_unchanged"] is True
    assert report["identity_lineage_unchanged"] is True
    assert report["source_published_events"] == ["delivered-initial"]
    assert report["destination_published_events"] == ["pending-interruption"]
    assert report["identity_published_events"] == []
    assert report["identity_output_cursor"] == 2
    assert "partition" in report["identity_restart_refused_roles"]
    assert report["final_status"] == "completed"
    assert report["final_maximum_difference"] <= 1e-12
    assert report["step_index"] == report["reference_step_index"]


def _cli(
    root: Path, *extra: str, module: str = "examples.meshfree_production_restart"
) -> subprocess.Popen[str]:
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "1"
    return subprocess.Popen(
        [_PYTHON, "-m", module, "--root", str(root), *extra],
        cwd=_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _records(lines: list[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in lines if line.startswith("{")]


def test_cli_killed_mid_run_resumes_to_the_uninterrupted_result(tmp_path: Path) -> None:
    reference = _cli(tmp_path / "reference", "--archive-seconds", "0.0")
    out, err = reference.communicate(timeout=1800)
    assert reference.returncode == 0, err[-4000:]
    expected = _records(out.splitlines())[-1]
    assert expected["status"] == "completed"

    killed = _cli(tmp_path / "killed")
    observed: list[str] = []
    stdout = killed.stdout
    if stdout is None:
        raise RuntimeError("The CLI subprocess has no stdout pipe.")
    deadline = time.monotonic() + 1800
    while sum('"output"' in line for line in observed) < 3:
        line = stdout.readline()
        assert line, killed.stderr.read() if killed.stderr is not None else ""
        observed.append(line)
        assert time.monotonic() < deadline
    killed.send_signal(signal.SIGKILL)
    killed.wait(timeout=60)
    assert killed.returncode == -signal.SIGKILL
    assert not any('"status"' in line for line in observed)

    resumed = _cli(tmp_path / "killed", "--resume", "--archive-seconds", "0.0")
    out, err = resumed.communicate(timeout=1800)
    assert resumed.returncode == 0, err[-4000:]
    records = _records(out.splitlines())
    start = records[0]
    assert start["start"] == "resume" and start["step"] > 0
    final = records[-1]
    for name in (
        "status",
        "step",
        "time",
        "state_sha256",
        "moments_sha256",
        "evidence_sha256",
        "accepted_step",
        "refused_attempts",
        "output_cursor",
    ):
        assert final[name] == expected[name], name
    assert final["sampled_peak_resident_bytes"] is not None


def test_epoch_cli_killed_inside_a_rebased_epoch_resumes_to_the_uninterrupted_result(
    tmp_path: Path,
) -> None:
    module = "examples.meshfree_production_epochs"
    reference = _cli(tmp_path / "reference", "--archive-seconds", "0.0", module=module)
    out, err = reference.communicate(timeout=1800)
    assert reference.returncode == 0, err[-4000:]
    expected = _records(out.splitlines())[-1]
    assert expected["status"] == "completed" and expected["epoch"] >= 2

    killed = _cli(tmp_path / "killed", module=module)
    stdout = killed.stdout
    if stdout is None:
        raise RuntimeError("The CLI subprocess has no stdout pipe.")
    observed: list[dict[str, Any]] = []
    deadline = time.monotonic() + 1800
    # An output labeled with a rebased epoch is dispatched only after the
    # checkpoint of an accepted step in that epoch committed.
    while not any("output" in record and record["epoch"] >= 1 for record in observed):
        line = stdout.readline()
        assert line, killed.stderr.read() if killed.stderr is not None else ""
        observed.extend(_records([line]))
        assert time.monotonic() < deadline
    killed.send_signal(signal.SIGKILL)
    killed.wait(timeout=60)
    assert killed.returncode == -signal.SIGKILL
    anchors = {
        record["epoch"]: record["anchor_step"]
        for record in observed
        if "anchor_step" in record
    }
    assert 1 in anchors and not any("status" in record for record in observed)

    resumed = _cli(
        tmp_path / "killed", "--resume", "--archive-seconds", "0.0", module=module
    )
    out, err = resumed.communicate(timeout=1800)
    assert resumed.returncode == 0, err[-4000:]
    records = _records(out.splitlines())
    # Committed but unacknowledged outputs are redelivered while resuming; the
    # fresh process found its post-rebase epoch from the committed lineage.
    start = next(record for record in records if "start" in record)
    assert start["start"] == "resume" and start["epoch"] >= 1
    assert start["step"] > anchors[1]
    final = records[-1]
    for name in (
        "status",
        "epoch",
        "step",
        "time",
        "state_sha256",
        "moments_sha256",
        "evidence_sha256",
        "accepted_step",
        "refused_attempts",
        "output_cursor",
    ):
        assert final[name] == expected[name], name
