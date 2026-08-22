import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.transport import AbstractBalancedTransportPlan


class _FixedPlan(AbstractBalancedTransportPlan):
    matrix: jnp.ndarray
    regularized_cost: jnp.ndarray
    convergence: jnp.ndarray

    def __init__(self, matrix, *, converged=True):
        self.matrix = jnp.asarray(matrix, dtype=float)
        self.regularized_cost = jnp.asarray(0.0)
        self.convergence = jnp.asarray(converged)

    @property
    def converged(self):
        return self.convergence

    def regularized_objective(self):
        return self.regularized_cost

    def source_marginal(self):
        return jnp.sum(self.matrix, axis=1)

    def target_marginal(self):
        return jnp.sum(self.matrix, axis=0)

    def apply_source_to_target(self, values):
        return self.matrix.T @ jnp.asarray(values)

    def apply_target_to_source(self, values):
        return self.matrix @ jnp.asarray(values)

    def barycentric_source_to_target(self, values):
        marginal = self.target_marginal()
        applied = self.apply_source_to_target(values)
        return applied / marginal.reshape((-1,) + (1,) * (applied.ndim - 1))

    def barycentric_target_to_source(self, values):
        marginal = self.source_marginal()
        applied = self.apply_target_to_source(values)
        return applied / marginal.reshape((-1,) + (1,) * (applied.ndim - 1))

    def dense_plan(self):
        return self.matrix


def test_independent_endpoint_coupling_replays_indices_and_gathers_context():
    source = jnp.arange(12.0).reshape((6, 2))
    target = 100.0 + jnp.arange(8.0).reshape((4, 2))
    context = {"condition": jnp.arange(4.0)[:, None]}

    first = phx.transport.independent_endpoint_coupling(
        source,
        target,
        jr.key(3),
        num_pairs=9,
        source_probabilities=jnp.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
        target_context=context,
    )
    second = phx.transport.independent_endpoint_coupling(
        source,
        target,
        jr.key(3),
        num_pairs=9,
        source_probabilities=jnp.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
        target_context=context,
    )

    assert jnp.array_equal(first.source_indices, second.source_indices)
    assert jnp.array_equal(first.target_indices, second.target_indices)
    assert jnp.all(first.source_indices != 1)
    assert jnp.all(first.source_indices != 3)
    assert jnp.array_equal(
        first.context["condition"], context["condition"][first.target_indices]
    )
    assert jnp.allclose(jnp.sum(first.probabilities), 1.0)


def test_transport_plan_endpoint_coupling_samples_joint_not_barycentric_pairs():
    source = jnp.asarray([[0.0], [1.0]])
    target = jnp.asarray([[10.0], [20.0]])
    plan = _FixedPlan(jnp.asarray([[0.0, 0.5], [0.5, 0.0]]))

    pairs = phx.transport.transport_plan_endpoint_coupling(
        plan,
        source,
        target,
        jr.key(4),
        num_pairs=32,
    )

    assert jnp.all(pairs.source_indices + pairs.target_indices == 1)
    assert jnp.array_equal(pairs.source, source[pairs.source_indices])
    assert jnp.array_equal(pairs.target, target[pairs.target_indices])
    assert jnp.all(jnp.isin(pairs.target[:, 0], jnp.asarray([10.0, 20.0])))


def test_transport_plan_endpoint_coupling_rejects_failure_and_shape_mismatch():
    failed = _FixedPlan(jnp.eye(2), converged=False)
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        phx.transport.transport_plan_endpoint_coupling(
            failed,
            jnp.zeros((2, 1)),
            jnp.ones((2, 1)),
            jr.key(0),
            num_pairs=2,
        )

    plan = _FixedPlan(jnp.ones((2, 3)) / 6.0)
    with pytest.raises(ValueError, match="atom counts"):
        phx.transport.transport_plan_endpoint_coupling(
            plan,
            jnp.zeros((2, 1)),
            jnp.ones((2, 1)),
            jr.key(0),
            num_pairs=2,
        )
