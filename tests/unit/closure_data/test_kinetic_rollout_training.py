from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint
from phydrax._numerics._checkpointed_scan import (
    AdaptiveReplayPreparationPolicy,
    prepare_replay_schedule,
)
from phydrax.closure_data._dataset import (
    ClosureSample,
    ClosureSampleKey,
    LeakageSafePartitionPlan,
    NormalizerProvenance,
    TrainOnlyNormalizer,
)
from phydrax.closure_data._kinetic_equilibrium import (
    EnergyEquilibriumSupportEnvelope,
    LearnedEnergyEquilibriumBindingPlan,
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
    attempt_kinetic_rollout_update,
    initialize_kinetic_rollout_training,
    kinetic_rollout_objective,
    KineticRolloutTrainingPlan,
    train_kinetic_rollout,
)
from phydrax.closure_data._state import FlowStateSchema
from phydrax.discretization.discrete_velocity._energy_equilibrium import (
    PositiveEnergyEquilibriumPlan,
)
from phydrax.discretization.discrete_velocity._quadrature import d2v17_quadrature
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)
from phydrax.discretization.discrete_velocity._spatial import (
    D2V17PeriodicTransportPlan,
    PreparedSmoothCompressibleD2V17SpatialDynamics,
)
from phydrax.equations._materials import IdealGasMaterial
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.nn.layers import Linear


def _model(*, offset=0.0):
    model = Linear(
        in_size=4,
        out_size=2,
        rwf=False,
        key=jr.key(101),
    )
    return eqx.tree_at(
        lambda value: (value.weight, value.bias),
        model,
        (
            jnp.zeros((2, 4), dtype=jnp.float64),
            jnp.full((2,), offset, dtype=jnp.float64),
        ),
    )


def _runtime():
    quadrature = d2v17_quadrature()
    material = IdealGasMaterial(1.4, 1.0)
    method = SmoothCompressibleD2VKineticMethod(
        quadrature, material, ConstantTransport(0.03, 0.04)
    )
    energy = PositiveEnergyEquilibriumPlan(quadrature, residual_tolerance=1.0e-8)
    transport = D2V17PeriodicTransportPlan(quadrature, (5, 5), (0.01, 0.01), 0.01)
    return PreparedSmoothCompressibleD2V17SpatialDynamics(
        method, energy, transport, conservation_tolerance=1.0e-10
    )


def _flow_schema():
    return FlowStateSchema(
        ("density", "momentum_x", "momentum_y", "total_energy"),
        ("kg/m^3", "kg/(m^2*s)", "kg/(m^2*s)", "J/m^3"),
        (1.0, 1.0, 1.0, 1.0),
        density_name="density",
        total_energy_name="total_energy",
    )


def _binding(runtime, *, stage_one_artifact_id="stage-one-artifact"):
    flow_schema = _flow_schema()
    provenance = NormalizerProvenance(
        partition_id="stage-one-partition",
        training_assignment_ids=("stage-one-assignment",),
        training_sample_ids=("stage-one-sample",),
        feature_name="conserved-state",
        schema_id=flow_schema.schema_id,
    )
    normalizer = TrainOnlyNormalizer(
        jnp.zeros((4,), dtype=jnp.float64),
        jnp.ones((4,), dtype=jnp.float64),
        provenance,
        epsilon=1.0e-12,
    )
    support = EnergyEquilibriumSupportEnvelope(
        rho_bounds=(0.5, 2.0),
        u_x_bounds=(-0.2, 0.2),
        u_y_bounds=(-0.2, 0.2),
        temperature_bounds=(0.45, 0.55),
        maximum_mach=0.5,
        minimum_hull_margin=1.0e-8,
        minimum_particle_equilibrium_margin=0.0,
        schema_id=flow_schema.schema_id,
        material_id=runtime.method.material.material_id,
        normalizer_id=normalizer.normalizer_id,
        quadrature_id=runtime.method.quadrature.quadrature_id,
        equilibrium_plan_id=runtime.energy_plan.plan_id,
        training_preparation_id="stage-one-preparation",
    )
    return LearnedEnergyEquilibriumBindingPlan(
        runtime.energy_plan,
        flow_schema,
        runtime.method.material,
        normalizer,
        support,
        input_component_names=flow_schema.component_names,
        semantic_id="stage-one-semantic",
        training_preparation_id="stage-one-preparation",
        parent_artifact_id=stage_one_artifact_id,
    )


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


def _uniform_state(runtime):
    conserved = jnp.asarray((1.0, 0.0, 0.0, 1.25), dtype=jnp.float64)
    target_flux = jnp.zeros((2,), dtype=jnp.float64)
    oracle = runtime.energy_plan.solve(conserved[-1], target_flux)
    equilibrium, evidence = runtime.method.equilibrium_from_energy_dual_with_evidence(
        conserved, oracle.dual, runtime.energy_plan
    )
    assert bool(evidence.successful)
    shape = runtime.transport.spatial_shape + (17,)
    return SmoothCompressibleKineticState(
        jnp.broadcast_to(equilibrium.particle_populations, shape),
        jnp.broadcast_to(equilibrium.total_energy_populations, shape),
    )


def _trajectory(runtime, binding, trajectory_id, *, perturbation=0.0):
    schema = _rollout_schema(runtime, binding)
    state = _uniform_state(runtime)
    f = jnp.broadcast_to(state.particle_populations, (26, *schema.f_shape))
    g = jnp.broadcast_to(state.total_energy_populations, (26, *schema.g_shape))
    U0 = runtime.method.moments(state).conserved
    U = jnp.broadcast_to(U0, (26, *schema.U_shape))
    if perturbation:
        g = g + perturbation
        U = U.at[..., -1].add(17.0 * perturbation)
    return SmoothCompressibleRolloutTrajectory(
        f,
        g,
        U,
        jnp.ones((26,), dtype=bool),
        schema,
        case_id=f"case-{trajectory_id}",
        trajectory_id=trajectory_id,
        realization_id="realization",
        time_block_id="block",
    )


def _parent_sample(schema, trajectory_id):
    return ClosureSample(
        jnp.zeros((1,), dtype=jnp.float64),
        ClosureSampleKey(
            case_id=f"case-{trajectory_id}",
            trajectory_id=trajectory_id,
            realization_id="realization",
            time_block_id="block",
            time_index=0,
        ),
        schema_id=schema.schema_id,
    )


def _trajectory_id_for_split(schema, partition, split):
    for index in range(10_000):
        trajectory_id = f"{split}-{index}"
        assignment = partition.assign((_parent_sample(schema, trajectory_id),))
        if assignment.assignments[0].split == split:
            return trajectory_id
    raise AssertionError(f"No deterministic {split} identity found.")


def _dataset(runtime, binding, *, validation_perturbation=0.0):
    schema = _rollout_schema(runtime, binding)
    partition = LeakageSafePartitionPlan(
        "trajectory",
        train_fraction=0.5,
        validation_fraction=0.5,
        test_fraction=0.0,
        salt="stage-two-training",
    )
    train_id = _trajectory_id_for_split(schema, partition, "train")
    validation_id = _trajectory_id_for_split(schema, partition, "validation")
    trajectories = (
        _trajectory(runtime, binding, train_id),
        _trajectory(
            runtime,
            binding,
            validation_id,
            perturbation=validation_perturbation,
        ),
    )
    return prepare_smooth_compressible_rollout_dataset(
        trajectories,
        partition,
        SmoothCompressibleRolloutWindowPlan(1, 25, maximum_windows_per_trajectory=1),
    )


def _schedules():
    policy = AdaptiveReplayPreparationPolicy(1_000_000, 1_000_000)
    return tuple(
        prepare_replay_schedule(horizon, 1024, policy) for horizon in (1, 2, 4, 8, 16, 25)
    )


def _plan(
    runtime,
    binding,
    dataset,
    *,
    replay_mode="step",
    learning_rate=1.0e-12,
):
    options = {}
    if replay_mode == "block":
        options["replay_block_size"] = 2
    if replay_mode == "scheduled":
        options["replay_schedules"] = _schedules()
    stage_two_binding = LearnedEnergyEquilibriumBindingPlan(
        binding.equilibrium_plan,
        binding.schema,
        binding.material,
        binding.normalizer,
        binding.support,
        input_component_names=binding.input_component_names,
        semantic_id="stage-two-semantic",
        training_preparation_id=dataset.preparation_id,
        parent_artifact_id=binding.parent_artifact_id,
    )
    return KineticRolloutTrainingPlan(
        runtime,
        stage_two_binding,
        accepted_updates_per_horizon=1,
        batch_size=1,
        guard_batch_size=1,
        learning_rate=learning_rate,
        replay_mode=replay_mode,
        **options,
    )


def _array_leaves(tree):
    return tuple(
        np.asarray(value) for value in jax.tree.leaves(tree) if eqx.is_array(value)
    )


def test_training_objective_has_no_holdout_split_dependence():
    runtime = _runtime()
    binding = _binding(runtime)
    clean = _dataset(runtime, binding)
    changed_holdout = _dataset(runtime, binding, validation_perturbation=0.01)
    plan = _plan(runtime, binding, clean)
    model = _model()

    clean_value, _ = kinetic_rollout_objective(
        model, plan, clean.statistics, clean.train_windows, 1
    )
    changed_value, _ = kinetic_rollout_objective(
        model,
        plan,
        changed_holdout.statistics,
        changed_holdout.train_windows,
        1,
    )

    np.testing.assert_array_equal(
        clean.statistics.f_mean, changed_holdout.statistics.f_mean
    )
    np.testing.assert_array_equal(
        clean.statistics.g_mean, changed_holdout.statistics.g_mean
    )
    np.testing.assert_array_equal(
        clean.statistics.U_mean, changed_holdout.statistics.U_mean
    )
    np.testing.assert_array_equal(clean_value, changed_value)


def test_short_window_objective_and_gradient_are_exact_across_replay_modes():
    runtime = _runtime()
    binding = _binding(runtime)
    dataset = _dataset(runtime, binding)
    model = _model()
    references = []
    for mode in ("full", "step", "block", "scheduled"):
        plan = _plan(runtime, binding, dataset, replay_mode=mode)

        def loss(candidate):
            return kinetic_rollout_objective(
                candidate, plan, dataset.statistics, dataset.train_windows, 2
            )[0]

        references.append((loss(model), eqx.filter_grad(loss)(model)))

    value, gradient = references[0]
    for replay_value, replay_gradient in references[1:]:
        np.testing.assert_array_equal(replay_value, value)
        for observed, expected in zip(
            _array_leaves(replay_gradient), _array_leaves(gradient), strict=True
        ):
            np.testing.assert_array_equal(observed, expected)


def test_vmap_isolates_one_failed_trajectory_without_shortening_other_scans():
    runtime = _runtime()
    binding = _binding(runtime)
    dataset = _dataset(runtime, binding)
    plan = _plan(runtime, binding, dataset)
    good = dataset.train_windows[0]
    bad = eqx.tree_at(
        lambda value: value.f_history,
        good,
        good.f_history.at[-1, 0, 0, 0].set(-1.0),
    )

    _, evidence = kinetic_rollout_objective(
        _model(), plan, dataset.statistics, (good, bad), 4
    )

    np.testing.assert_array_equal(evidence.successful, np.asarray((True, False)))
    assert evidence.total_loss.shape == (2,)
    assert evidence.maximum_conservation_residual.shape == (2,)


def test_rejected_proposal_rolls_back_model_and_optimizer_together():
    runtime = _runtime()
    binding = _binding(runtime)
    dataset = _dataset(runtime, binding)
    plan = _plan(runtime, binding, dataset)
    state = initialize_kinetic_rollout_training(
        _model(offset=0.2), plan, dataset, key=jr.key(7)
    )
    before_model = array_tree_fingerprint(state.model)
    before_optimizer = array_tree_fingerprint(state.optimizer_state)

    update = attempt_kinetic_rollout_update(state, plan, dataset)

    assert not update.accepted
    assert update.state.attempt_count == 1
    assert update.state.rejection_count == 1
    assert update.state.accepted_update_count == 0
    assert array_tree_fingerprint(update.state.model) == before_model
    assert array_tree_fingerprint(update.state.optimizer_state) == before_optimizer


def test_checkpoint_resume_matches_uninterrupted_next_update(tmp_path):
    runtime = _runtime()
    binding = _binding(runtime)
    dataset = _dataset(runtime, binding)
    plan = _plan(runtime, binding, dataset)
    initial = initialize_kinetic_rollout_training(_model(), plan, dataset, key=jr.key(11))
    first = attempt_kinetic_rollout_update(initial, plan, dataset).state
    checkpoint = tmp_path / "stage-two.ckpt"
    write_kinetic_rollout_checkpoint(checkpoint, first, plan, dataset)
    restored = read_kinetic_rollout_checkpoint(checkpoint, plan, dataset, _model())

    uninterrupted = attempt_kinetic_rollout_update(first, plan, dataset).state
    resumed = attempt_kinetic_rollout_update(restored, plan, dataset).state

    assert restored.state_id == first.state_id
    assert resumed.state_id == uninterrupted.state_id
    assert array_tree_fingerprint(resumed.model) == array_tree_fingerprint(
        uninterrupted.model
    )
    assert array_tree_fingerprint(resumed.optimizer_state) == array_tree_fingerprint(
        uninterrupted.optimizer_state
    )


def test_best_model_snapshot_is_immutable_across_rejection():
    runtime = _runtime()
    binding = _binding(runtime)
    dataset = _dataset(runtime, binding)
    plan = _plan(runtime, binding, dataset)
    state = initialize_kinetic_rollout_training(
        _model(offset=0.2), plan, dataset, key=jr.key(19)
    )
    best_before = array_tree_fingerprint(state.best_model)

    rejected = attempt_kinetic_rollout_update(state, plan, dataset)

    assert not rejected.accepted
    assert array_tree_fingerprint(state.best_model) == best_before
    assert array_tree_fingerprint(rejected.state.best_model) == best_before
    np.testing.assert_array_equal(rejected.state.best_loss, state.best_loss)


def test_curriculum_advances_only_after_each_accepted_guard():
    runtime = _runtime()
    binding = _binding(runtime)
    dataset = _dataset(runtime, binding)
    plan = _plan(runtime, binding, dataset)
    initial = initialize_kinetic_rollout_training(_model(), plan, dataset, key=jr.key(23))

    result = train_kinetic_rollout(initial, plan, dataset, 6)

    assert result.termination == "curriculum_complete"
    assert tuple(value.horizon for value in result.attempts) == (1, 2, 4, 8, 16, 25)
    assert all(value.accepted for value in result.attempts)
    assert result.state.accepted_update_count == 6
    assert result.state.rejection_count == 0


def test_checkpoint_refuses_plan_dataset_and_model_identity_mismatch(tmp_path):
    runtime = _runtime()
    binding = _binding(runtime)
    dataset = _dataset(runtime, binding)
    plan = _plan(runtime, binding, dataset)
    state = initialize_kinetic_rollout_training(_model(), plan, dataset, key=jr.key(29))
    checkpoint = tmp_path / "identity.ckpt"
    write_kinetic_rollout_checkpoint(checkpoint, state, plan, dataset)

    with pytest.raises(ValueError, match="identity mismatch"):
        read_kinetic_rollout_checkpoint(
            checkpoint,
            _plan(runtime, binding, dataset, learning_rate=2.0e-12),
            dataset,
            _model(),
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        read_kinetic_rollout_checkpoint(
            checkpoint,
            plan,
            _dataset(runtime, binding, validation_perturbation=0.001),
            _model(),
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        read_kinetic_rollout_checkpoint(
            checkpoint,
            plan,
            dataset,
            eqx.tree_at(
                lambda value: value.weight,
                _model(),
                jnp.zeros((2, 5), dtype=jnp.float64),
            ),
        )
