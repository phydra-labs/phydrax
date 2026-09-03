#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel


class _LinearEncoder(AbstractArrayModel):
    weight: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.weight = jnp.asarray([[0.2], [1.0]], dtype=jnp.float64)
        self.in_size = 2
        self.out_size = 1

    def __call__(self, value, /, *, key=None):
        del key
        return value @ self.weight


def _data(steps=300):
    state = jnp.asarray([1.0, -0.3], dtype=jnp.float64)
    values = []
    for index in range(steps):
        values.append(state)
        state = jnp.asarray([0.98 * state[0], 0.45 * state[1]]) + 0.02 * jnp.asarray(
            [jnp.sin(index * 0.3), jnp.cos(index * 0.7)]
        )
    return phx.dynamics.TrajectoryData(
        jnp.arange(steps, dtype=jnp.float64) * 0.1,
        jnp.stack(values),
        state_layout=phx.dynamics.StateLayout((2,)),
        source_id="kinetic-training-data",
    )


def test_variational_training_selects_executable_canonical_coordinate():
    data = _data()
    policy = phx.dynamics.identification.VariationalKineticTrainingPolicy(
        maximum_steps=4,
        learning_rate=5.0e-3,
        regularization=1.0e-5,
        validation_interval=1,
        maximum_transitions=1000,
    )

    result = phx.dynamics.identification.fit_variational_kinetic_model(
        _LinearEncoder(),
        data,
        jax.random.key(12),
        model_id="linear-slow-encoder",
        policy=policy,
        n_modes=1,
    )

    assert bool(result.valid)
    assert result.history.steps.shape == (5,)
    assert jnp.all(jnp.isfinite(result.history.training_scores))
    assert result.coordinate_model(jnp.asarray([0.4, -0.2])).shape == (1,)
    assert result.transform(data.states[:8]).shape == (8, 1)
    gradient = jax.grad(lambda value: result.coordinate_model(value)[0])(
        jnp.asarray([0.4, -0.2])
    )
    assert jnp.all(jnp.isfinite(gradient))


def test_training_capacity_and_shape_fail_closed():
    data = _data(20)
    policy = phx.dynamics.identification.VariationalKineticTrainingPolicy(
        maximum_steps=0, maximum_transitions=5
    )

    with pytest.raises(ValueError, match="capacity"):
        phx.dynamics.identification.fit_variational_kinetic_model(
            _LinearEncoder(),
            data,
            jax.random.key(1),
            model_id="capacity-test",
            policy=policy,
            n_modes=1,
        )


def test_variational_training_checkpoint_roundtrip(tmp_path):
    data = _data(80)
    policy = phx.dynamics.identification.VariationalKineticTrainingPolicy(
        maximum_steps=1,
        validation_interval=1,
        maximum_transitions=100,
    )
    checkpoint = tmp_path / "kinetic-checkpoint"
    first = phx.dynamics.identification.fit_variational_kinetic_model(
        _LinearEncoder(),
        data,
        jax.random.key(5),
        model_id="checkpointed-encoder",
        policy=policy,
        n_modes=1,
        checkpoint_path=checkpoint,
    )
    resumed = phx.dynamics.identification.fit_variational_kinetic_model(
        _LinearEncoder(),
        data,
        jax.random.key(5),
        model_id="checkpointed-encoder",
        policy=policy,
        n_modes=1,
        checkpoint_path=checkpoint,
        resume=True,
    )

    point = jnp.asarray([0.3, -0.1])
    assert resumed.resumed_from_step == 1
    assert jnp.allclose(first.coordinate_model(point), resumed.coordinate_model(point))
    assert jnp.array_equal(first.history.steps, resumed.history.steps)
