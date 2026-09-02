from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

import phydrax as phx
from phydrax._trainable import partition_trainable
from phydrax.linalg import MaterializationPolicy, materialize
from phydrax.solver._functional_surrogate import prepare_functional_update


def test_functional_sharding_places_named_sample_axes_and_replicates_events():
    policy = phx.solver.FunctionalShardingPolicy({"sample": "data"})
    field = cx.Field(
        jnp.arange(8.0).reshape((4, 2)),
        dims=("sample", None),
    )
    placed = policy.place_field(field)

    assert jnp.array_equal(placed.data, field.data)
    assert placed.dims == field.dims
    assert placed.data.sharding.mesh == policy.mesh
    assert policy.field_sharding(field).spec == jax.sharding.PartitionSpec(
        "data", None
    )
    assert jnp.allclose(jnp.sum(placed.data), 28.0)


def _scalar_solver(value=1.0):
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Parameter(jnp.asarray(value))
    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda current: current)
    batch = component.points(
        {"x": jnp.asarray([[0.1], [0.3], [0.7], [0.9]])}
    )
    term = phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(
            phx.integration.from_samples(
                phx.integration.mean_over(component), batch
            )
        ),
    )
    return phx.solver.FunctionalSolver(functions={"u": field}, terms=(term,))


def test_sharded_functional_ntk_matches_unsharded_global_kernel():
    solver = _scalar_solver()
    params, non_trainable = partition_trainable(solver.functions)
    prepared = solver.objective.prepare_training(
        (0,),
        scale=1.0,
        evaluation_key=jax.random.key(4),
        sampling_key=jax.random.key(5),
        iteration=1,
    )
    policy = phx.solver.FunctionalShardingPolicy(
        {"__phydra_blk__x": "data"}
    )
    sharded = policy.place_prepared(prepared)
    unsharded_update = prepare_functional_update(
        prepared,
        params,
        non_trainable,
        solver.enforcement,
    )
    sharded_update = prepare_functional_update(
        sharded,
        policy.place_parameters(params),
        policy.place_tree(non_trainable),
        solver.enforcement,
    )
    unsharded_ntk = phx.solver.prepare_functional_ntk(
        solver,
        prepared_update=unsharded_update,
    )
    sharded_ntk = phx.solver.prepare_functional_ntk(
        solver,
        prepared_update=sharded_update,
    )
    materialization = MaterializationPolicy(max_entries=64, max_bytes=4096)

    assert jnp.allclose(
        materialize(sharded_ntk.kernel, materialization),
        materialize(unsharded_ntk.kernel, materialization),
    )

class _WindowAdapter(phx.solver.FunctionalWindowAdapter):
    adapter_id: str = eqx.field(static=True, default="test-window-adapter")


    def build_solver(
        self,
        previous_solver: Any,
        window_index: int,
        bounds,
        previous_terminal,
        /,
    ):
        del window_index, bounds, previous_terminal
        return eqx.tree_at(
            lambda solver: solver.training_state,
            previous_solver,
            None,
            is_leaf=lambda value: value is None,
        )

    def terminal_fields(self, solver, window_index, bounds, /):
        del window_index, bounds
        return {"u": solver.functions["u"]}

    def seam_metrics(
        self, previous_terminal, current_solver, window_index, bounds, /
    ):
        del window_index, bounds
        previous = previous_terminal["u"].func()
        current = current_solver.functions["u"].func()
        return {"u": jnp.abs(current - previous)}


def test_functional_time_windows_train_and_route_physical_query():
    schedule = phx.sampling.collocation.CausalTimeSlabSchedule((0.0, 0.5, 1.0))
    plan = phx.solver.FunctionalTimeWindowPlan(
        schedule,
        _WindowAdapter(),
        lambda index: optax.sgd(0.05),
        steps=1,
    )
    result = phx.solver.train_functional_time_windows(_scalar_solver(), plan)

    assert len(result.solvers) == 2
    assert len(result.terminal_fields) == 2
    assert len(result.seam_metrics) == 1
    assert result.solver_at(jnp.asarray(0.25)) is result.solvers[0]
    assert result.solver_at(jnp.asarray(0.75)) is result.solvers[1]
    assert bool(result.successful)


def test_functional_time_windows_transfer_optimizer_state_independently():
    schedule = phx.sampling.collocation.CausalTimeSlabSchedule((0.0, 0.5, 1.0))
    plan = phx.solver.FunctionalTimeWindowPlan(
        schedule,
        _WindowAdapter(),
        lambda index: optax.adam(0.01),
        steps=1,
        training=phx.solver.FunctionalTrainingPlan(),
        transfer_optimizer_state=True,
    )
    result = phx.solver.train_functional_time_windows(_scalar_solver(), plan)
    final_state = result.solvers[-1].training_state
    assert final_state is not None
    integer_scalars = tuple(
        int(value)
        for value in jax.tree.leaves(final_state.optimizer_state)
        if hasattr(value, "dtype")
        and jnp.issubdtype(value.dtype, jnp.integer)
        and value.shape == ()
    )
    assert 2 in integer_scalars
