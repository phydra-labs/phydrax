#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.statistical_dynamics._cumulants import (
    cumulants_from_ensemble,
    DenseCumulantState,
    factorize_cumulant,
    ForcingCovariance,
    RankAdaptationPolicy,
    require_valid_state,
    SecondCumulantLayout,
    solve_stationary_covariance,
)
from phydrax.statistical_dynamics._distributed import (
    DistributedBatchLayout,
    DistributedCovarianceLayout,
    DistributedRestartRelation,
    DistributedStatisticalLayout,
)
from phydrax.statistical_dynamics._interactions import InteractionContinuationSchedule
from phydrax.statistical_dynamics._plan import (
    execute_interaction_continuation,
    QuadraticDynamics,
    StatisticalDynamicsPlan,
)


def _coupled_plan(*, closure="ce2", interaction_model="ql", time_step=0.01):
    layout = SecondCumulantLayout(2, [0])
    quadratic = jnp.zeros((2, 2, 2))
    quadratic = quadratic.at[0, 1, 1].set(1.0)
    quadratic = quadratic.at[1, 0, 1].set(0.5)
    quadratic = quadratic.at[1, 1, 0].set(0.5)
    dynamics = QuadraticDynamics(
        jnp.zeros(2),
        jnp.diag(jnp.asarray([-1.0, -2.0])),
        quadratic,
    )
    forcing = ForcingCovariance(jnp.asarray([[0.2]]))
    plan = StatisticalDynamicsPlan(
        layout,
        dynamics,
        forcing,
        closure=closure,
        interaction_model=interaction_model,
        time_step=time_step,
    )
    return layout, plan.prepare()


def test_analytical_ou_lyapunov_covariance_uses_native_matrix_equation():
    forcing = ForcingCovariance(jnp.diag(jnp.asarray([2.0, 8.0])))
    result = solve_stationary_covariance(
        jnp.diag(jnp.asarray([-1.0, -2.0])),
        forcing,
    )

    assert bool(result.successful)
    np.testing.assert_allclose(result.covariance, jnp.diag(jnp.asarray([1.0, 2.0])))
    np.testing.assert_allclose(result.residual, 0.0, atol=1e-12)


def test_ce2_and_gce2_equal_the_corresponding_symmetric_ensemble_moments():
    mean = jnp.asarray([0.3])
    covariance = jnp.asarray([[0.7]])
    members = jnp.asarray(
        [[mean[0], jnp.sqrt(covariance[0, 0])], [mean[0], -jnp.sqrt(covariance[0, 0])]]
    )

    for closure, model in (("ce2", "ql"), ("gce2", "gql")):
        layout, prepared = _coupled_plan(closure=closure, interaction_model=model)
        state = cumulants_from_ensemble(layout, members)
        np.testing.assert_allclose(state.mean, mean)
        np.testing.assert_allclose(state.covariance, covariance)
        tendency = prepared.rhs(state)
        member_tendencies = jnp.stack(
            tuple(prepared.plan.dynamics(member) for member in members)
        )
        ensemble_mean_tendency = jnp.mean(member_tendencies[:, 0])
        centered = members[:, 1]
        ensemble_covariance_tendency = (
            jnp.mean(2.0 * centered * member_tendencies[:, 1])
            + prepared.plan.forcing.covariance[0, 0]
        )

        np.testing.assert_allclose(tendency.mean[0], ensemble_mean_tendency, atol=1e-12)
        np.testing.assert_allclose(
            tendency.covariance[0, 0], ensemble_covariance_tendency, atol=1e-12
        )


def test_psd_hermitian_and_rank_gates_are_fail_closed_without_repair():
    layout, _ = _coupled_plan()
    wrong_layout = DenseCumulantState(
        jnp.zeros(1), jnp.eye(1), layout_id="another-layout"
    )
    with pytest.raises(ValueError, match="another layout"):
        require_valid_state(layout, wrong_layout)

    negative = DenseCumulantState(jnp.zeros(1), -jnp.eye(1), layout_id=layout.layout_id)
    with pytest.raises(ValueError, match="positive-semidefinite"):
        require_valid_state(layout, negative)

    with pytest.raises(ValueError, match="PSD gate"):
        ForcingCovariance(-jnp.eye(1))

    complex_layout = SecondCumulantLayout(3, [0])
    nonhermitian = DenseCumulantState(
        jnp.zeros(1),
        jnp.asarray([[1.0, 1.0j], [0.0, 1.0]]),
        layout_id=complex_layout.layout_id,
    )
    with pytest.raises(ValueError, match="Hermitian"):
        require_valid_state(complex_layout, nonhermitian)

    full = DenseCumulantState(jnp.zeros(1), jnp.eye(1), layout_id=layout.layout_id)
    with pytest.raises(ValueError, match="exceeds the eddy covariance dimension"):
        factorize_cumulant(layout, full, RankAdaptationPolicy(0, 2))


def test_dense_and_factor_paths_match_before_explicit_rank_adaptation():
    layout, prepared = _coupled_plan()
    dense = DenseCumulantState(
        jnp.asarray([0.2]), jnp.asarray([[0.5]]), layout_id=layout.layout_id
    )
    policy = RankAdaptationPolicy(1, 1)
    factor = factorize_cumulant(layout, dense, policy).state

    dense_step = prepared.step(dense)
    factor_step = prepared.step(factor, rank_policy=policy)

    np.testing.assert_allclose(
        dense_step.pre_truncation_state.mean,
        factor_step.pre_truncation_state.mean,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        dense_step.pre_truncation_state.covariance,
        factor_step.pre_truncation_state.covariance,
        atol=1e-12,
    )
    assert factor_step.rank_event is not None
    assert bool(factor_step.rank_event.accepted)
    np.testing.assert_allclose(
        factor_step.rank_event.pre_truncation_error, 0.0, atol=1e-12
    )


def test_factor_execution_records_explicit_rank_growth():
    layout = SecondCumulantLayout(3, [0])
    dynamics = QuadraticDynamics(
        jnp.zeros(3),
        jnp.diag(jnp.asarray([0.0, -1.0, -1.0])),
        jnp.zeros((3, 3, 3)),
    )
    prepared = StatisticalDynamicsPlan(
        layout,
        dynamics,
        ForcingCovariance(jnp.diag(jnp.asarray([0.0, 1.0]))),
        closure="ce2",
        interaction_model="ql",
        time_step=0.01,
    ).prepare()
    dense = DenseCumulantState(
        jnp.zeros(1),
        jnp.diag(jnp.asarray([1.0, 0.0])),
        layout_id=layout.layout_id,
    )
    policy = RankAdaptationPolicy(1, 2)
    factor = factorize_cumulant(layout, dense, policy).state

    result = prepared.step(factor, rank_policy=policy)

    assert result.rank_event is not None
    assert bool(result.rank_event.triggered)
    assert int(result.rank_event.old_rank) == 1
    assert int(result.rank_event.new_rank) == 2


def test_continuation_restart_and_distributed_topology_relation():
    layout, prepared = _coupled_plan()
    initial = DenseCumulantState(
        jnp.zeros(1), jnp.asarray([[0.1]]), layout_id=layout.layout_id
    )
    schedule = InteractionContinuationSchedule([0.0, 0.5])
    continued = execute_interaction_continuation(
        schedule,
        (prepared, prepared),
        initial,
        steps_per_stage=2,
    )
    assert bool(continued.completed)
    assert continued.evidence[1].start_state_id == continued.evidence[0].end_state_id
    checkpoint = prepared.checkpoint(continued.state, 0.04, 4)
    restarted, time, step = prepared.restart(checkpoint)
    np.testing.assert_allclose(restarted.covariance, continued.state.covariance)
    np.testing.assert_allclose(time, 0.04)
    assert int(step) == 4

    source = DistributedStatisticalLayout(
        DistributedBatchLayout(8, 2, item_bytes=16, maximum_local_bytes=64),
        DistributedCovarianceLayout(4, 2, maximum_local_bytes=64),
    )
    target = DistributedStatisticalLayout(
        DistributedBatchLayout(8, 4, item_bytes=16, maximum_local_bytes=64),
        DistributedCovarianceLayout(4, 4, maximum_local_bytes=64),
    )
    relation = DistributedRestartRelation(source, target)
    assert relation.accepted
    assert relation.topology_changed
    relation.require()
    array = jnp.arange(32.0).reshape((8, 4))
    redistributed = relation.redistribute_batch(source.batch.shard(array))
    np.testing.assert_array_equal(target.batch.assemble(redistributed), array)

    with pytest.raises(MemoryError, match="maximum_local_bytes"):
        DistributedCovarianceLayout(100, 1, maximum_local_bytes=100)
