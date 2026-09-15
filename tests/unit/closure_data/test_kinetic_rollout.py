from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.closure_data._dataset import (
    ClosureSample,
    ClosureSampleKey,
    LeakageSafePartitionPlan,
)
from phydrax.closure_data._kinetic_rollout import (
    prepare_smooth_compressible_rollout_dataset,
    SmoothCompressibleRolloutSchema,
    SmoothCompressibleRolloutTrajectory,
    SmoothCompressibleRolloutWindow,
    SmoothCompressibleRolloutWindowPlan,
)


def _schema(
    *,
    runtime_id: str = "runtime",
    stage_one_artifact_id: str = "stage-one",
) -> SmoothCompressibleRolloutSchema:
    return SmoothCompressibleRolloutSchema(
        (2, 3),
        jnp.float32,
        runtime_id=runtime_id,
        topology_id="periodic-topology",
        method_id="atomic-collide-kick-stream",
        quadrature_id="d2v17",
        material_id="calorically-perfect-gas",
        stage_one_artifact_id=stage_one_artifact_id,
        support_id="cell-centers",
        oracle_id="native-d2v17-oracle",
        dt=0.125,
        boundary_schedule_id="periodic-boundary-schedule",
        force_schedule_id="body-force-schedule",
    )


def _trajectory(
    *,
    schema: SmoothCompressibleRolloutSchema | None = None,
    case_id: str = "case",
    trajectory_id: str = "trajectory",
    time_block_id: str = "block",
    count: int = 7,
    valid: np.ndarray | None = None,
) -> SmoothCompressibleRolloutTrajectory:
    schema_ = _schema() if schema is None else schema
    time = jnp.arange(count, dtype=jnp.float32).reshape((count, 1, 1, 1))
    velocity = jnp.arange(17, dtype=jnp.float32).reshape((1, 1, 1, 17))
    components = jnp.arange(4, dtype=jnp.float32).reshape((1, 1, 1, 4))
    f = jnp.broadcast_to(time + velocity, (count, 2, 3, 17))
    g = jnp.broadcast_to(2.0 * time - velocity, (count, 2, 3, 17))
    U = jnp.broadcast_to(jnp.square(time) + components, (count, 2, 3, 4))
    validity = np.ones((count,), dtype=bool) if valid is None else valid
    return SmoothCompressibleRolloutTrajectory(
        f,
        g,
        U,
        validity,
        schema_,
        case_id=case_id,
        trajectory_id=trajectory_id,
        realization_id="realization",
        time_block_id=time_block_id,
    )


def _parent_sample(case_id: str, trajectory_id: str) -> ClosureSample:
    return ClosureSample(
        jnp.zeros((1,), dtype=jnp.float32),
        ClosureSampleKey(
            case_id=case_id,
            trajectory_id=trajectory_id,
            realization_id="realization",
            time_block_id="block",
            time_index=0,
        ),
        schema_id="selection-schema",
    )


def _one_group_per_split(
    plan: LeakageSafePartitionPlan,
) -> dict[str, tuple[str, str]]:
    selected: dict[str, tuple[str, str]] = {}
    for index in range(10_000):
        identity = (f"case-{index}", f"trajectory-{index}")
        sample = _parent_sample(*identity)
        split = plan.assign((sample,)).assignments[0].split
        selected.setdefault(split, identity)
        if len(selected) == 3:
            return selected
    raise AssertionError("Could not construct deterministic split fixtures.")


def test_complete_parents_are_partitioned_before_any_overlapping_windows():
    partition_plan = LeakageSafePartitionPlan(
        "trajectory",
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        salt="kinetic-parent-splits",
    )
    groups = _one_group_per_split(partition_plan)
    parents = tuple(
        _trajectory(
            case_id=case_id,
            trajectory_id=trajectory_id,
            time_block_id=f"block-{block}",
        )
        for case_id, trajectory_id in groups.values()
        for block in range(2)
    )
    window_plan = SmoothCompressibleRolloutWindowPlan(2, 2)
    dataset = prepare_smooth_compressible_rollout_dataset(
        parents, partition_plan, window_plan
    )

    for parent in parents:
        parent_windows = tuple(
            window for window in dataset.windows if window.parent_id == parent.parent_id
        )
        assert len(parent_windows) == 4
        assert len({window.split for window in parent_windows}) == 1
    for case_id, trajectory_id in groups.values():
        group_windows = tuple(
            window
            for parent in parents
            if (parent.case_id, parent.trajectory_id) == (case_id, trajectory_id)
            for window in dataset.windows
            if window.parent_id == parent.parent_id
        )
        assert len({window.split for window in group_windows}) == 1

    train_groups = {
        (parent.case_id, parent.trajectory_id)
        for parent, sample in zip(
            dataset.trajectories, dataset.parent_samples, strict=True
        )
        if dataset.partition.assignment_for(sample.sample_id).split == "train"
    }
    holdout_groups = {
        (parent.case_id, parent.trajectory_id)
        for parent, sample in zip(
            dataset.trajectories, dataset.parent_samples, strict=True
        )
        if dataset.partition.assignment_for(sample.sample_id).split != "train"
    }
    assert train_groups.isdisjoint(holdout_groups)
    assert {
        dataset.partition.assignment_for(sample.sample_id).split
        for sample in dataset.parent_samples
    } == {"train", "validation", "test"}


def test_windows_are_contiguous_deterministic_and_reject_bad_anchor_or_horizon():
    parent = _trajectory(count=9)
    plan = SmoothCompressibleRolloutWindowPlan(
        3,
        2,
        stride=1,
        maximum_windows_per_trajectory=2,
        selection_salt="fixed-selection",
    )
    anchors = plan.anchors(parent)
    assert anchors == plan.anchors(parent)
    assert len(anchors) == 2
    windows = tuple(
        SmoothCompressibleRolloutWindow(parent, plan, anchor, split="train")
        for anchor in anchors
    )
    for window in windows:
        assert window.history_indices == tuple(range(window.anchor - 3, window.anchor))
        assert window.target_indices == tuple(range(window.anchor, window.anchor + 2))
        assert window.time_indices == tuple(range(window.anchor - 3, window.anchor + 2))
        assert window.f_history.shape == (3, 2, 3, 17)
        assert window.g_targets.shape == (2, 2, 3, 17)
        assert window.U_targets.shape == (2, 2, 3, 4)

    repeat = prepare_smooth_compressible_rollout_dataset(
        (parent,),
        LeakageSafePartitionPlan(
            "trajectory",
            train_fraction=1.0,
            validation_fraction=0.0,
            test_fraction=0.0,
            salt="deterministic-preparation",
        ),
        plan,
    )
    duplicate = prepare_smooth_compressible_rollout_dataset(
        (_trajectory(count=9),), repeat.partition.plan, plan
    )
    assert tuple(value.window_id for value in repeat.windows) == tuple(
        value.window_id for value in duplicate.windows
    )
    assert repeat.preparation_id == duplicate.preparation_id

    with pytest.raises(ValueError, match="horizon_steps"):
        SmoothCompressibleRolloutWindowPlan(2, 0)
    with pytest.raises(ValueError, match="anchor"):
        SmoothCompressibleRolloutWindow(parent, plan, 2, split="train")
    with pytest.raises(ValueError, match="too short"):
        plan.anchors(_trajectory(count=4))


def test_native_shapes_dtype_and_bound_identities_are_exact():
    schema = _schema()
    parent = _trajectory(schema=schema)
    assert schema.f_shape == (2, 3, 17)
    assert schema.g_shape == (2, 3, 17)
    assert schema.U_shape == (2, 3, 4)
    assert schema.conserved_component_names == (
        "density",
        "momentum_x",
        "momentum_y",
        "total_energy",
    )
    assert schema.boundary_schedule_id == "periodic-boundary-schedule"
    assert schema.force_schedule_id == "body-force-schedule"
    assert parent.f.dtype == parent.g.dtype == parent.U.dtype == jnp.dtype(schema.dtype)

    with pytest.raises(ValueError, match="f history"):
        SmoothCompressibleRolloutTrajectory(
            parent.f[..., :-1],
            parent.g,
            parent.U,
            parent.valid,
            schema,
            case_id="case",
            trajectory_id="bad-f",
            realization_id="realization",
            time_block_id="block",
        )
    with pytest.raises(TypeError, match="dtypes"):
        SmoothCompressibleRolloutTrajectory(
            parent.f,
            parent.g.astype(jnp.float16),
            parent.U,
            parent.valid,
            schema,
            case_id="case",
            trajectory_id="bad-g-dtype",
            realization_id="realization",
            time_block_id="block",
        )
    with pytest.raises(ValueError, match="U history"):
        SmoothCompressibleRolloutTrajectory(
            parent.f,
            parent.g,
            parent.U[..., :3],
            parent.valid,
            schema,
            case_id="case",
            trajectory_id="bad-U",
            realization_id="realization",
            time_block_id="block",
        )
    with pytest.raises(ValueError, match="validity"):
        SmoothCompressibleRolloutTrajectory(
            parent.f,
            parent.g,
            parent.U,
            np.ones((parent.sample_count - 1,), dtype=bool),
            schema,
            case_id="case",
            trajectory_id="bad-validity",
            realization_id="realization",
            time_block_id="block",
        )

    foreign = _trajectory(
        schema=_schema(runtime_id="foreign-runtime"),
        case_id="foreign-case",
        trajectory_id="foreign-trajectory",
    )
    with pytest.raises(ValueError, match="schemas"):
        prepare_smooth_compressible_rollout_dataset(
            (parent, foreign),
            LeakageSafePartitionPlan(
                "case",
                train_fraction=1.0,
                validation_fraction=0.0,
                test_fraction=0.0,
                salt="schema-rejection",
            ),
            SmoothCompressibleRolloutWindowPlan(2, 2),
        )


def test_failed_oracle_parent_is_rejected_whole_and_never_partially_windowed():
    failed_validity = np.ones((8,), dtype=bool)
    failed_validity[-1] = False
    good = _trajectory(trajectory_id="good", count=8)
    failed = _trajectory(trajectory_id="failed", count=8, valid=failed_validity)
    with pytest.raises(ValueError, match="rejected in full"):
        prepare_smooth_compressible_rollout_dataset(
            (good, failed),
            LeakageSafePartitionPlan(
                "trajectory",
                train_fraction=1.0,
                validation_fraction=0.0,
                test_fraction=0.0,
                salt="oracle-validity",
            ),
            SmoothCompressibleRolloutWindowPlan(2, 2, maximum_windows_per_trajectory=1),
        )


def test_statistics_count_each_training_parent_state_once_not_each_window():
    parent = _trajectory(count=6)
    dataset = prepare_smooth_compressible_rollout_dataset(
        (parent,),
        LeakageSafePartitionPlan(
            "trajectory",
            train_fraction=1.0,
            validation_fraction=0.0,
            test_fraction=0.0,
            salt="statistics",
        ),
        SmoothCompressibleRolloutWindowPlan(2, 2),
    )
    direct_U_mean = jnp.mean(parent.U, axis=(0, 1, 2))
    overlapping_U = jnp.concatenate(
        tuple(
            jnp.concatenate((window.U_history, window.U_targets), axis=0)
            for window in dataset.train_windows
        ),
        axis=0,
    )
    overlapping_U_mean = jnp.mean(overlapping_U, axis=(0, 1, 2))
    np.testing.assert_allclose(dataset.statistics.U_mean, direct_U_mean)
    assert not np.allclose(dataset.statistics.U_mean, overlapping_U_mean)
    assert dataset.statistics.training_parent_ids == (parent.parent_id,)
    assert len(dataset.train_windows) > 1


def test_parent_artifact_and_runtime_ids_each_change_preparation_identity():
    partition_plan = LeakageSafePartitionPlan(
        "trajectory",
        train_fraction=1.0,
        validation_fraction=0.0,
        test_fraction=0.0,
        salt="identity",
    )
    window_plan = SmoothCompressibleRolloutWindowPlan(2, 2)

    def prepared(
        *,
        trajectory_id: str = "trajectory",
        stage_one_artifact_id: str = "stage-one",
        runtime_id: str = "runtime",
    ):
        schema = _schema(
            runtime_id=runtime_id,
            stage_one_artifact_id=stage_one_artifact_id,
        )
        return prepare_smooth_compressible_rollout_dataset(
            (_trajectory(schema=schema, trajectory_id=trajectory_id),),
            partition_plan,
            window_plan,
        )

    baseline = prepared()
    identities = {
        baseline.preparation_id,
        prepared(trajectory_id="other-parent").preparation_id,
        prepared(stage_one_artifact_id="other-stage-one").preparation_id,
        prepared(runtime_id="other-runtime").preparation_id,
    }
    assert len(identities) == 4
