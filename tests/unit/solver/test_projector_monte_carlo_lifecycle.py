#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Persistence preserves full histories, scientific bindings, and logical draws."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from jax import Array

from phydrax._array_archive import ArrayArchiveLimits
from phydrax.lifecycle import open as open_lifecycle, query, ResultRevision
from phydrax.solver import (
    analyze_projector_monte_carlo,
    initialize_projector_monte_carlo,
    prepare_projector_monte_carlo,
    PreparedProjectorMonteCarlo,
    ProjectorEstimatorPolicy,
    ProjectorMonteCarloPlan,
    ProjectorMonteCarloProblem,
    ProjectorMonteCarloState,
    ProjectorMonteCarloStatus,
    read_projector_monte_carlo_checkpoint,
    solve_projector_monte_carlo,
    step_projector_monte_carlo,
    transport_projector_monte_carlo_resources,
    write_projector_monte_carlo_checkpoint,
    write_projector_monte_carlo_result,
)
from phydrax.solver._projector_monte_carlo_contracts import SpawnPolicy
from phydrax.typing import PRNGKey
from phydrax.units import derived_unit, HARTREE
from phydrax.uq import CorrelatedRatioPolicy
from tests._support.projector_monte_carlo import (
    control_keys,
    control_operator,
    two_boson_two_site_control,
)


pytestmark = pytest.mark.strict_jax


def _prepared(
    *,
    support: int = 3,
    groups: int = 8,
    events: int = 2,
    history: int = 24,
    dt: float = 0.02,
    spawning: SpawnPolicy = "sampled",
) -> PreparedProjectorMonteCarlo:
    control = two_boson_two_site_control()
    operator = control_operator(control)
    keys = control_keys(operator, control)
    problem = ProjectorMonteCarloProblem(
        operator,
        keys,
        np.asarray((0.0, 1.0, 0.0), dtype=np.complex128),
        dt=dt,
        energy_unit=HARTREE,
        inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
        provenance_id=control.provenance,
    )
    plan = ProjectorMonteCarloPlan(
        replicas=2,
        support_capacity=support,
        group_capacity=groups,
        event_capacity=events,
        attempt_capacity=32,
        source_capacity=1,
        history_capacity=history,
        maximum_retained_bytes=20_000_000,
        maximum_workspace_bytes=20_000_000,
        spawn_policy=spawning,
    )
    return prepare_projector_monte_carlo(problem, plan)


def _leaf_value(value: Array, /) -> Array:
    return (
        jax.random.key_data(value)
        if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key)
        else value
    )


def _assert_state_values(
    actual: ProjectorMonteCarloState,
    expected: ProjectorMonteCarloState,
    /,
) -> None:
    left = jax.tree.leaves(actual)
    right = jax.tree.leaves(expected)
    for actual_leaf, expected_leaf in zip(left, right, strict=True):
        if not isinstance(actual_leaf, Array) or not isinstance(expected_leaf, Array):
            raise TypeError("Committed state dynamic leaves must be canonical arrays.")
        np.testing.assert_array_equal(
            _leaf_value(actual_leaf), _leaf_value(expected_leaf)
        )
    assert actual.scientific_id == expected.scientific_id
    assert actual.plan_id == expected.plan_id


def test_checkpoint_preserves_stream_and_uninterrupted_history(tmp_path: Path) -> None:
    prepared = _prepared()
    initial = initialize_projector_monte_carlo(prepared, jax.random.key(17))
    first = solve_projector_monte_carlo(prepared, initial, steps=3)
    assert int(first.status) == ProjectorMonteCarloStatus.SUCCESS
    path = write_projector_monte_carlo_checkpoint(
        tmp_path / "committed.phx", prepared, first.state
    )
    template = initialize_projector_monte_carlo(prepared, jax.random.key(999))
    restored = read_projector_monte_carlo_checkpoint(path, prepared, template)
    _assert_state_values(restored, first.state)
    resumed = solve_projector_monte_carlo(prepared, restored, steps=4)
    uninterrupted = solve_projector_monte_carlo(prepared, initial, steps=7)
    assert int(resumed.status) == int(uninterrupted.status) == 0
    _assert_state_values(resumed.state, uninterrupted.state)
    np.testing.assert_array_equal(initial.step, np.asarray(0, dtype=np.int64))


@pytest.mark.parametrize("implementation", ("threefry2x32", "rbg"))
def test_checkpoint_restores_declared_typed_key_implementation(
    tmp_path: Path, implementation: str
) -> None:
    prepared = _prepared(spawning="exact")
    key: PRNGKey = jax.random.key(31, impl=implementation)
    initial = initialize_projector_monte_carlo(prepared, key)
    result = solve_projector_monte_carlo(prepared, initial, steps=2)
    path = write_projector_monte_carlo_checkpoint(
        tmp_path / "implementation.phx", prepared, result.state
    )
    restored = read_projector_monte_carlo_checkpoint(path, prepared, initial)
    assert str(jax.random.key_impl(restored.root_key)) == implementation
    _assert_state_values(restored, result.state)


def test_checkpoint_refuses_changed_physical_step(tmp_path: Path) -> None:
    source = _prepared(dt=0.02)
    state = initialize_projector_monte_carlo(source, jax.random.key(4))
    path = write_projector_monte_carlo_checkpoint(tmp_path / "source.phx", source, state)
    other = _prepared(dt=0.025)
    template = initialize_projector_monte_carlo(other, jax.random.key(4))
    with pytest.raises(ValueError, match="compatibility identities"):
        read_projector_monte_carlo_checkpoint(path, other, template)


def test_resource_enlargement_replays_last_committed_step_without_lost_history(
    tmp_path: Path,
) -> None:
    source = _prepared(support=1, groups=1, events=1, history=8, spawning="exact")
    state = initialize_projector_monte_carlo(source, jax.random.key(11))
    failed = step_projector_monte_carlo(source, state)
    assert int(failed.status) == ProjectorMonteCarloStatus.INTERMEDIATE_GROUP_OVERFLOW
    _assert_state_values(failed.state, state)
    path = write_projector_monte_carlo_checkpoint(
        tmp_path / "last-commit.phx", source, failed.state
    )
    restored_source = read_projector_monte_carlo_checkpoint(path, source, state)
    target = _prepared(support=3, groups=8, events=3, history=24, spawning="exact")
    transported, relation = transport_projector_monte_carlo_resources(
        source, target, restored_source
    )
    assert relation.classification == "bitwise"
    np.testing.assert_array_equal(
        jax.random.key_data(transported.root_key), jax.random.key_data(state.root_key)
    )
    assert int(transported.step) == int(transported.history.count) == 0
    resumed = solve_projector_monte_carlo(target, transported, steps=5)
    uninterrupted = solve_projector_monte_carlo(
        target, initialize_projector_monte_carlo(target, state.root_key), steps=5
    )
    assert int(resumed.status) == 0
    _assert_state_values(resumed.state, uninterrupted.state)


def test_resource_transport_preserves_existing_samples_and_cursor() -> None:
    source = _prepared(history=8)
    original = initialize_projector_monte_carlo(source, jax.random.key(63))
    partial = solve_projector_monte_carlo(source, original, steps=3)
    assert int(partial.status) == 0
    target = _prepared(support=5, groups=12, events=4, history=24)
    transferred, _ = transport_projector_monte_carlo_resources(
        source, target, partial.state
    )
    assert int(transferred.step) == int(transferred.history.count) == 3
    np.testing.assert_array_equal(
        transferred.history.applied_shifts[:, :8], partial.state.history.applied_shifts
    )
    np.testing.assert_array_equal(
        transferred.history.pair_numerators[:, :8], partial.state.history.pair_numerators
    )
    resumed = solve_projector_monte_carlo(target, transferred, steps=3)
    full = solve_projector_monte_carlo(
        target, initialize_projector_monte_carlo(target, original.root_key), steps=6
    )
    _assert_state_values(resumed.state, full.state)


def test_resource_transport_refuses_a_scientific_change() -> None:
    source = _prepared()
    state = initialize_projector_monte_carlo(source, jax.random.key(9))
    changed = _prepared(history=32, dt=0.04)
    with pytest.raises(ValueError, match="scientific bindings"):
        transport_projector_monte_carlo_resources(source, changed, state)


def test_result_archive_retains_raw_statistics_and_explicit_scientific_scope(
    tmp_path: Path,
) -> None:
    prepared = _prepared(spawning="exact", history=32)
    initial = initialize_projector_monte_carlo(prepared, jax.random.key(5))
    result = solve_projector_monte_carlo(prepared, initial, steps=24)
    analysis = analyze_projector_monte_carlo(
        prepared,
        result,
        policy=ProjectorEstimatorPolicy(
            ratio_policy=CorrelatedRatioPolicy(minimum_draws=2, minimum_blocks=2),
            deterministic_records=True,
            history_depths=(0, 1),
        ),
    )
    archive = write_projector_monte_carlo_result(
        tmp_path / "result.phx",
        prepared,
        result,
        run_id="finite-projector-control",
        analysis=analysis,
    )
    reopened = open_lifecycle(
        archive.path,
        limits=ArrayArchiveLimits(
            max_members=len(archive.arrays) + 1,
            max_container_bytes=prepared.plan.maximum_retained_bytes,
            max_aggregate_bytes=prepared.plan.maximum_retained_bytes,
            max_member_bytes=prepared.plan.maximum_retained_bytes,
        ),
    )
    assert isinstance(reopened.manifest, ResultRevision)
    np.testing.assert_array_equal(
        reopened.arrays["history.projected_numerator"],
        result.state.history.projected_numerator,
    )
    np.testing.assert_array_equal(
        reopened.arrays["history.pair_denominators"],
        result.state.history.pair_denominators,
    )
    selected = query(reopened, fields=("applied-shift", "projected-denominator"))
    assert selected.fields[0].unit == HARTREE.unit_id
    semantics = dict(reopened.manifest.manifest.sampled_semantics)
    assert semantics["completion"] == "operational-not-scientific-release"
    assert semantics["scientific_id"] == prepared.scientific_id
    np.testing.assert_array_equal(
        reopened.arrays["analysis/.projected.mean_covariance"],
        analysis.projected.mean_covariance,
    )


def test_archive_distinguishes_equal_numeric_histories_with_different_weight_horizons(
    tmp_path: Path,
) -> None:
    prepared = _prepared(spawning="exact", history=32)
    initial = initialize_projector_monte_carlo(prepared, jax.random.key(5))
    result = solve_projector_monte_carlo(prepared, initial, steps=24)
    common = CorrelatedRatioPolicy(minimum_draws=2, minimum_blocks=2)
    zero = analyze_projector_monte_carlo(
        prepared,
        result,
        policy=ProjectorEstimatorPolicy(
            ratio_policy=common,
            burn_in=1,
            deterministic_records=True,
            reference_energy=prepared.plan.initial_shift,
            history_depths=(0,),
        ),
    )
    one = analyze_projector_monte_carlo(
        prepared,
        result,
        policy=ProjectorEstimatorPolicy(
            ratio_policy=common,
            burn_in=1,
            deterministic_records=True,
            reference_energy=prepared.plan.initial_shift,
            history_depths=(1,),
        ),
    )
    np.testing.assert_array_equal(
        zero.reweighted[0].projected_weights.effective_samples,
        one.reweighted[0].projected_weights.effective_samples,
    )
    first = write_projector_monte_carlo_result(
        tmp_path / "zero-history.phx",
        prepared,
        result,
        run_id="same-finite-run",
        analysis=zero,
    )
    second = write_projector_monte_carlo_result(
        tmp_path / "one-history.phx",
        prepared,
        result,
        run_id="same-finite-run",
        analysis=one,
    )
    limits = ArrayArchiveLimits(max_members=1000)
    reopened_zero = open_lifecycle(first.path, limits=limits)
    reopened_one = open_lifecycle(second.path, limits=limits)
    if not isinstance(reopened_zero.manifest, ResultRevision) or not isinstance(
        reopened_one.manifest, ResultRevision
    ):
        raise TypeError("Projector result archives require canonical result revisions.")
    left = dict(reopened_zero.manifest.manifest.sampled_semantics)
    right = dict(reopened_one.manifest.manifest.sampled_semantics)
    assert json.loads(left["analysis_interpretation"])["history_depths"] == [0]
    assert json.loads(right["analysis_interpretation"])["history_depths"] == [1]
    assert json.loads(right["analysis_interpretation"])["history_horizons"] == [
        prepared.problem.dt
    ]
    assert (
        reopened_zero.manifest.manifest.result_id
        != reopened_one.manifest.manifest.result_id
    )
