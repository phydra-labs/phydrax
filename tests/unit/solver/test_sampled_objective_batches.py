from collections.abc import Callable, Mapping

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, Key

import phydrax as phx
import phydrax.solver._functional_objective as functional_objective


class _NestedSampledObjective(phx.terms.AbstractSamplingTerm):
    recorder: Callable[[Key[Array, ""]], None] = eqx.field(static=True)
    target_shift: float = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def sample(self, *, key):
        self.recorder(key)
        return {
            "target": (
                jnp.asarray(self.target_shift),
                {"scale": jnp.asarray(1.0)},
            )
        }

    def loss(self, functions: Mapping, /, *, key, iter_=None, batch=None, **kwargs):
        del key, iter_, kwargs
        if batch is None:
            raise AssertionError("Sampled objective batch was not materialized.")
        target, metadata = batch["target"]
        value = jnp.asarray(functions["u"].func()).reshape(())
        return metadata["scale"] * (value - target) ** 2


def _solver(*objectives):
    domain = phx.domain.Interval1d(0.0, 1.0)
    return phx.solver.FunctionalSolver(functions={"u": domain.Parameter(1.0)}, terms=objectives, )


def _key_recorder(store):
    def record(key):
        store.append(tuple(np.asarray(jr.key_data(key), dtype=np.uint32).tolist()))

    return record


def test_optax_materializes_each_sampled_objective_once_per_update():
    sampled_keys = []
    objective = _NestedSampledObjective(
        label="sampled",
        recorder=_key_recorder(sampled_keys),
        target_shift=0.0,
    )

    _solver(objective).solve(
        num_iter=4,
        optim=optax.sgd(0.1),
        jit=True,
        keep_best=False,
        log_every=0,
    )

    assert len(sampled_keys) == 4
    assert len(set(sampled_keys)) == 4

def test_optax_materializes_each_integration_realization_once_per_update(
    monkeypatch,
):
    domain = phx.domain.Interval1d(0.0, 1.0)
    component = domain.component()
    condition = phx.conditions.Residual("u", component, lambda value: value)
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(
            8,
            layout=phx.domain.SampleLayout((("x",),)),
        ),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": domain.Parameter(1.0)},
        terms=(phx.terms.ResidualPenalty(condition, source),),
    )
    materialization_keys = []
    resolve_integration = functional_objective.resolve_integration

    def record_materialization(source_, /, **kwargs):
        materialization_keys.append(
            tuple(np.asarray(jr.key_data(kwargs["key"]), dtype=np.uint32).tolist())
        )
        return resolve_integration(source_, **kwargs)

    monkeypatch.setattr(
        functional_objective,
        "resolve_integration",
        record_materialization,
    )

    solver.solve(
        num_iter=3,
        optim=optax.sgd(0.1),
        jit=True,
        keep_best=True,
        log_every=0,
    )

    assert len(materialization_keys) == 3
    assert len(set(materialization_keys)) == 3


def test_selection_reuses_the_optimizer_update_batch():
    sampled_keys = []
    objective = _NestedSampledObjective(
        label="sampled",
        recorder=_key_recorder(sampled_keys),
        target_shift=0.0,
    )

    _solver(objective).solve(
        num_iter=3,
        optim=optax.sgd(0.1),
        evaluation_parameters=lambda _state, params: params,
        jit=False,
        keep_best=True,
        log_every=0,
    )

    assert len(sampled_keys) == 3


def test_nested_batches_and_multiple_objectives_use_distinct_subkeys():
    first_keys = []
    second_keys = []
    objectives = (
        _NestedSampledObjective(
            label="first",
            recorder=_key_recorder(first_keys),
            target_shift=0.0,
        ),
        _NestedSampledObjective(
            label="second",
            recorder=_key_recorder(second_keys),
            target_shift=0.5,
        ),
    )

    trained = _solver(*objectives).solve(
        num_iter=2,
        optim=optax.sgd(0.05),
        jit=False,
        keep_best=False,
        log_every=0,
    )

    assert len(first_keys) == len(second_keys) == 2
    assert all(left != right for left, right in zip(first_keys, second_keys, strict=True))
    assert jnp.isfinite(jnp.asarray(trained["u"].func()).reshape(()))
