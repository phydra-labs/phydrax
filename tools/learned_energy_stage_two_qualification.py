#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._identity import SemanticProvenance
from phydrax._model import AbstractArrayModel
from phydrax._numerics._checkpointed_scan import (
    AdaptiveReplayPreparationPolicy,
    prepare_replay_schedule,
)
from phydrax.closure_data._dataset import (
    ClosureSample,
    ClosureSampleKey,
    LeakageSafePartitionPlan,
)
from phydrax.closure_data._kinetic_equilibrium import (
    energy_equilibrium_numeric_revision,
    LearnedEnergyEquilibriumBindingPlan,
)
from phydrax.closure_data._kinetic_equilibrium_artifact import (
    read_learned_energy_equilibrium_artifact,
    write_learned_energy_equilibrium_artifact,
)
from phydrax.closure_data._kinetic_rollout import (
    prepare_smooth_compressible_rollout_dataset,
    SmoothCompressibleRolloutSchema,
    SmoothCompressibleRolloutTrajectory,
    SmoothCompressibleRolloutWindowPlan,
)
from phydrax.closure_data._kinetic_rollout_checkpoint import (
    read_kinetic_rollout_checkpoint,
    write_kinetic_rollout_checkpoint,
)
from phydrax.closure_data._kinetic_rollout_training import (
    initialize_kinetic_rollout_training,
    kinetic_rollout_objective,
    KineticRolloutTrainingPlan,
    train_kinetic_rollout,
)
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
)
from phydrax.discretization.discrete_velocity._spatial import (
    D2V17PeriodicTransportPlan,
    PreparedSmoothCompressibleD2V17SpatialDynamics,
)
from phydrax.equations._transport_closures import ConstantTransport


STAGE_ONE_ARTIFACT_PATH = Path("benchmarks/learned_energy_equilibrium.phxml")
REPORT_PATH = Path("benchmarks/learned_energy_stage_two.json")
STAGE_TWO_ARTIFACT_PATH = Path("benchmarks/learned_energy_stage_two.phxml")
CURRICULUM = (1, 2, 4, 8, 16, 25)
REPLAY_LOSS_TOLERANCE = 1.0e-12
REPLAY_GRADIENT_RELATIVE_TOLERANCE = 1.0e-10


def _stage_two_binding(
    stage_one,
    *,
    semantic_id=None,
    training_preparation_id=None,
):
    source = stage_one.binding.plan
    return LearnedEnergyEquilibriumBindingPlan(
        source.equilibrium_plan,
        source.schema,
        source.material,
        source.normalizer,
        source.support,
        input_component_names=source.input_component_names,
        semantic_id=source.semantic_id if semantic_id is None else semantic_id,
        training_preparation_id=(
            source.training_preparation_id
            if training_preparation_id is None
            else training_preparation_id
        ),
        parent_artifact_id=stage_one.artifact_id,
    )


def _runtime(binding):
    method = SmoothCompressibleD2VKineticMethod(
        binding.equilibrium_plan.quadrature,
        binding.material,
        ConstantTransport(0.03, 0.04),
    )
    transport = D2V17PeriodicTransportPlan(method.quadrature, (5, 5), (0.01, 0.01), 0.01)
    return PreparedSmoothCompressibleD2V17SpatialDynamics(
        method,
        binding.equilibrium_plan,
        transport,
        conservation_tolerance=1.0e-10,
    )


def _sample_key(trajectory_id, *, time_index=0):
    return ClosureSampleKey(
        case_id=f"case-{trajectory_id}",
        trajectory_id=trajectory_id,
        realization_id="deterministic",
        time_block_id="qualification",
        time_index=time_index,
    )


def _trajectory_id_for_split(schema_id, partition, split):
    for index in range(10_000):
        trajectory_id = f"{split}-{index}"
        sample = ClosureSample(
            jnp.zeros((1,), dtype=jnp.float64),
            _sample_key(trajectory_id),
            schema_id=schema_id,
        )
        if partition.assign((sample,)).assignments[0].split == split:
            return trajectory_id
    raise RuntimeError(f"Could not construct deterministic {split} trajectory.")


def _rollout_schema(runtime, binding):
    return SmoothCompressibleRolloutSchema(
        runtime.transport.spatial_shape,
        jnp.float64,
        runtime_id=runtime.prepared_id,
        topology_id=runtime.transport.plan_id,
        method_id=runtime.method.method_id,
        quadrature_id=runtime.method.quadrature.quadrature_id,
        material_id=runtime.method.material.material_id,
        stage_one_artifact_id=binding.parent_artifact_id,
        support_id=binding.support.support_id,
        oracle_id=runtime.energy_plan.plan_id,
        dt=runtime.required_step_size,
    )


def _reference_conserved(runtime, density, phase):
    nx, ny = runtime.transport.spatial_shape
    x = (jnp.arange(nx, dtype=jnp.float64) + 0.5) / nx
    y = (jnp.arange(ny, dtype=jnp.float64) + 0.5) / ny
    x_grid, y_grid = jnp.meshgrid(x, y, indexing="ij")
    angle_x = 2.0 * jnp.pi * x_grid + phase
    angle_y = 2.0 * jnp.pi * y_grid - 0.5 * phase
    rho = density + 0.02 * jnp.sin(angle_x) * jnp.cos(angle_y)
    velocity = jnp.stack(
        (
            0.02 * jnp.sin(angle_y),
            -0.015 * jnp.cos(angle_x),
        ),
        axis=-1,
    )
    temperature = 0.5 + 0.015 * jnp.cos(angle_x) * jnp.cos(angle_y)
    pressure = rho * runtime.method.material.gas_constant * temperature
    internal_energy = runtime.method.material.specific_internal_energy(rho, pressure)
    total_energy = rho * internal_energy + 0.5 * rho * jnp.sum(velocity**2, axis=-1)
    return jnp.concatenate(
        (rho[..., None], rho[..., None] * velocity, total_energy[..., None]),
        axis=-1,
    )


def _oracle_trajectory(runtime, binding, trajectory_id, density, phase):
    schema = _rollout_schema(runtime, binding)
    conserved = _reference_conserved(runtime, density, phase)
    velocity = conserved[..., 1:3] / conserved[..., 0, None]
    kinetic_energy = 0.5 * jnp.sum(conserved[..., 1:3] * velocity, axis=-1)
    internal_energy = (conserved[..., -1] - kinetic_energy) / conserved[..., 0]
    pressure = runtime.method.material.pressure(conserved[..., 0], internal_energy)
    target_flux = (conserved[..., -1] + pressure)[..., None] * velocity
    oracle = runtime.energy_plan.solve(conserved[..., -1], target_flux)
    equilibrium, equilibrium_evidence = (
        runtime.method.equilibrium_from_energy_dual_with_evidence(
            conserved, oracle.dual, runtime.energy_plan
        )
    )
    if not bool(np.asarray(equilibrium_evidence.successful)):
        raise RuntimeError("Deterministic oracle initial state failed.")
    state = equilibrium
    f_states = []
    g_states = []
    U_states = []
    valid = []
    for _ in range(31):
        f_states.append(state.particle_populations)
        g_states.append(state.total_energy_populations)
        U_states.append(runtime.method.moments(state).conserved)
        result, oracle_evidence = runtime.step_oracle(
            state, jnp.asarray(runtime.required_step_size, dtype=jnp.float64)
        )
        step_ok = bool(
            np.asarray(result.successful & jnp.all(oracle_evidence.successful))
        )
        valid.append(step_ok)
        if not step_ok:
            raise RuntimeError("Deterministic oracle rollout failed.")
        state = result.accepted_state
    return SmoothCompressibleRolloutTrajectory(
        jnp.stack(f_states),
        jnp.stack(g_states),
        jnp.stack(U_states),
        jnp.asarray(valid, dtype="bool"),
        schema,
        case_id=f"case-{trajectory_id}",
        trajectory_id=trajectory_id,
        realization_id="deterministic",
        time_block_id="qualification",
    )


def _dataset(runtime, binding):
    partition = LeakageSafePartitionPlan(
        "trajectory",
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        salt="learned-energy-stage-two-qualification",
    )
    schema = _rollout_schema(runtime, binding)
    trajectory_ids = tuple(
        _trajectory_id_for_split(schema.schema_id, partition, split)
        for split in ("train", "validation", "test")
    )
    rho_min, rho_max = binding.support.rho_bounds
    densities = tuple(
        rho_min + fraction * (rho_max - rho_min) for fraction in (0.25, 0.5, 0.75)
    )
    phases = (0.0, 0.7, 1.4)
    trajectories = tuple(
        _oracle_trajectory(runtime, binding, trajectory_id, density, phase)
        for trajectory_id, density, phase in zip(
            trajectory_ids, densities, phases, strict=True
        )
    )
    dataset = prepare_smooth_compressible_rollout_dataset(
        trajectories,
        partition,
        SmoothCompressibleRolloutWindowPlan(1, 25),
    )
    observed = {
        dataset.partition.assignment_for(sample.sample_id).split
        for sample in dataset.parent_samples
    }
    if observed != {"train", "validation", "test"}:
        raise RuntimeError("Qualification parent partition did not cover all splits.")
    return dataset


def _schedules():
    policy = AdaptiveReplayPreparationPolicy(2_000_000, 2_000_000)
    return tuple(prepare_replay_schedule(horizon, 4096, policy) for horizon in CURRICULUM)


def _training_plan(runtime, binding, replay_mode):
    options = {}
    if replay_mode == "block":
        options["replay_block_size"] = 4
    if replay_mode == "scheduled":
        options["replay_schedules"] = _schedules()
    return KineticRolloutTrainingPlan(
        runtime,
        binding,
        accepted_updates_per_horizon=1,
        batch_size=2,
        guard_batch_size=2,
        learning_rate=1.0e-5,
        maximum_scaled_flux_error=6.0e-2,
        replay_mode=replay_mode,
        **options,
    )


def _gradient_measurement(model, plan, dataset, horizon):
    def loss(candidate):
        return kinetic_rollout_objective(
            candidate, plan, dataset.statistics, dataset.train_windows, horizon
        )[0]

    return eqx.filter_grad(loss)(model), float(np.asarray(loss(model)))


def _maximum_relative_gradient_difference(left, right):
    left_leaves = tuple(
        np.asarray(value) for value in jax.tree.leaves(left) if eqx.is_array(value)
    )
    right_leaves = tuple(
        np.asarray(value) for value in jax.tree.leaves(right) if eqx.is_array(value)
    )
    if len(left_leaves) != len(right_leaves):
        raise ValueError("Replay gradients do not have the same trainable tree.")
    maximum = 0.0
    for observed, expected in zip(left_leaves, right_leaves, strict=True):
        scale = max(
            float(np.max(np.abs(observed))),
            float(np.max(np.abs(expected))),
            1.0,
        )
        maximum = max(
            maximum,
            float(np.max(np.abs(observed - expected))) / scale,
        )
    return maximum


def _atomic_write_report(report):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_name(f".{REPORT_PATH.name}.tmp")
    temporary.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(REPORT_PATH)


def main() -> None:
    stage_one = read_learned_energy_equilibrium_artifact(STAGE_ONE_ARTIFACT_PATH)
    source_binding = _stage_two_binding(stage_one)
    model = stage_one.binding.model.as_trainable()
    if not isinstance(model, AbstractArrayModel):
        raise TypeError("The stage-one artifact did not restore an array model.")
    runtime = _runtime(source_binding)
    dataset = _dataset(runtime, source_binding)
    semantic = SemanticProvenance(
        {
            "kind": "learned-positive-total-energy-equilibrium-rollout",
            "parent_semantic": source_binding.semantic_id,
            "objective": {
                "macroscopic_rollout": 1.0,
                "energy_population_rollout": 1.0,
                "on_policy_oracle_flux": 0.1,
            },
            "curriculum": list(CURRICULUM),
            "optimizer": "Adam",
            "learning_rate": 1.0e-5,
            "replay": "step",
        },
        resource_ids={
            "stage_one_artifact": stage_one.artifact_id,
            "rollout_preparation": dataset.preparation_id,
            "runtime": runtime.prepared_id,
            "support": source_binding.support.support_id,
            "normalizer": source_binding.normalizer.normalizer_id,
        },
    )
    binding = _stage_two_binding(
        stage_one,
        semantic_id=semantic.semantic_id,
        training_preparation_id=dataset.preparation_id,
    )
    step_plan = _training_plan(runtime, binding, "step")
    guard_windows = dataset.validation_windows
    baseline_loss, baseline_evidence = kinetic_rollout_objective(
        model, step_plan, dataset.statistics, guard_windows, 25
    )
    initial = initialize_kinetic_rollout_training(
        model, step_plan, dataset, key=jr.key(1701)
    )
    uninterrupted = train_kinetic_rollout(initial, step_plan, dataset, 6)
    first_half = train_kinetic_rollout(initial, step_plan, dataset, 3)
    with tempfile.TemporaryDirectory(prefix="phydrax-stage-two-") as directory:
        checkpoint_path = Path(directory) / "accepted.ckpt"
        write_kinetic_rollout_checkpoint(
            checkpoint_path, first_half.state, step_plan, dataset
        )
        restored = read_kinetic_rollout_checkpoint(
            checkpoint_path, step_plan, dataset, model
        )
        resumed = train_kinetic_rollout(restored, step_plan, dataset, 3)

    trained_loss, trained_evidence = kinetic_rollout_objective(
        resumed.state.best_model,
        step_plan,
        dataset.statistics,
        guard_windows,
        25,
    )
    replay_raw = {
        mode: _gradient_measurement(
            model, _training_plan(runtime, binding, mode), dataset, 4
        )
        for mode in ("full", "step", "block", "scheduled")
    }
    reference_gradient, reference_loss = replay_raw["full"]
    replay = {
        mode: {
            "gradient_fingerprint": array_tree_fingerprint(gradient),
            "loss": loss,
            "absolute_loss_error": abs(loss - reference_loss),
            "maximum_relative_gradient_error": _maximum_relative_gradient_difference(
                gradient, reference_gradient
            ),
        }
        for mode, (gradient, loss) in replay_raw.items()
    }
    late_window = max(dataset.test_windows, key=lambda value: value.anchor)
    _, late_evidence = kinetic_rollout_objective(
        resumed.state.best_model,
        step_plan,
        dataset.statistics,
        (late_window,),
        25,
    )
    out_of_support = eqx.tree_at(
        lambda value: value.f_history,
        late_window,
        late_window.f_history.at[-1, 0, 0, 0].set(-1.0),
    )
    _, out_of_support_evidence = kinetic_rollout_objective(
        resumed.state.best_model,
        step_plan,
        dataset.statistics,
        (out_of_support,),
        25,
    )
    gates = {
        "stage_one_baseline_physical": bool(
            np.asarray(jnp.all(baseline_evidence.successful))
        ),
        "trained_guard_physical": bool(np.asarray(jnp.all(trained_evidence.successful))),
        "trained_no_regression": float(np.asarray(trained_loss))
        <= float(np.asarray(baseline_loss)) + 1.0e-14,
        "replay_objective_and_gradient_agree": all(
            value["absolute_loss_error"] <= REPLAY_LOSS_TOLERANCE
            and value["maximum_relative_gradient_error"]
            <= REPLAY_GRADIENT_RELATIVE_TOLERANCE
            for value in replay.values()
        ),
        "deterministic_resume_exact": resumed.state.state_id
        == uninterrupted.state.state_id,
        "curriculum_complete": resumed.termination == "curriculum_complete"
        and tuple(value.horizon for value in (*first_half.attempts, *resumed.attempts))
        == CURRICULUM,
        "late_start_physical": late_window.anchor > 1
        and bool(np.asarray(jnp.all(late_evidence.successful))),
        "out_of_support_refused": not bool(
            np.asarray(jnp.all(out_of_support_evidence.successful))
        ),
        "checkpoint_preserved_exact_accepted_boundary": (
            first_half.state.last_update_accepted
            and restored.state_id == first_half.state.state_id
        ),
    }
    if not all(gates.values()):
        failed = tuple(name for name, passed in gates.items() if not passed)
        raise RuntimeError(f"Stage-two qualification gates failed: {failed}")

    revision = energy_equilibrium_numeric_revision(
        binding.semantic_id, resumed.state.best_model
    )
    prepared_binding = binding.prepare(resumed.state.best_model, revision)
    artifact_path = write_learned_energy_equilibrium_artifact(
        STAGE_TWO_ARTIFACT_PATH, prepared_binding
    )
    restored_stage_two = read_learned_energy_equilibrium_artifact(artifact_path)
    report = {
        "kind": "learned-energy-stage-two-qualification",
        "passed": True,
        "gates": gates,
        "identities": {
            "training_plan_id": step_plan.plan_id,
            "rollout_preparation_id": dataset.preparation_id,
            "runtime_id": runtime.prepared_id,
            "stage_one_artifact_id": stage_one.artifact_id,
            "stage_one_model_revision_id": (
                stage_one.binding.numeric_revision.revision_id
            ),
            "normalizer_id": binding.normalizer.normalizer_id,
            "support_id": binding.support.support_id,
            "final_model_revision_id": revision.revision_id,
            "stage_two_semantic_id": semantic.semantic_id,
            "stage_two_binding_plan_id": binding.plan_id,
            "stage_two_artifact_id": restored_stage_two.artifact_id,
            "stage_two_prepared_binding_id": restored_stage_two.binding.prepared_id,
            "checkpoint_state_id": resumed.state.state_id,
        },
        "curriculum": list(CURRICULUM),
        "attempt_count": resumed.state.attempt_count,
        "accepted_update_count": resumed.state.accepted_update_count,
        "rejection_count": resumed.state.rejection_count,
        "stage_one_baseline_loss": float(np.asarray(baseline_loss)),
        "selected_stage_two_loss": float(np.asarray(trained_loss)),
        "maximum_scaled_flux_error": float(
            np.asarray(jnp.max(trained_evidence.maximum_scaled_flux_error))
        ),
        "late_start_anchor": late_window.anchor,
        "replay": replay,
        "replay_tolerances": {
            "absolute_loss": REPLAY_LOSS_TOLERANCE,
            "maximum_relative_gradient": REPLAY_GRADIENT_RELATIVE_TOLERANCE,
        },
        "artifact": {
            "path": str(artifact_path),
            "artifact_id": restored_stage_two.artifact_id,
            "prepared_binding_id": restored_stage_two.binding.prepared_id,
            "numeric_revision_id": restored_stage_two.binding.numeric_revision.revision_id,
        },
    }
    report["artifact_id"] = canonical_fingerprint(report)
    _atomic_write_report(report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
