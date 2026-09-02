import jax.numpy as jnp
import jax.random as jr

from phydrax._probability import DiagonalNormalLaw
from phydrax.stochastic._path_diffusion import TrajectoryEventLayout
from phydrax.transport._assignment import solve_multidimensional_assignment
from phydrax.transport.continuous._field_density import (
    FiniteFieldFlowLaw,
    HybridFlowLaw,
    prepare_field_query,
    TrajectoryFlowLaw,
)
from phydrax.transport.continuous._injective_density import InjectiveContinuousFlowLaw


def test_rectangular_assignment_is_exact_and_lexicographic():
    costs = jnp.asarray([[0.0, 2.0, 2.0], [2.0, 0.0, 1.0]])
    result = solve_multidimensional_assignment(
        jnp.asarray([[0.0], [1.0]]),
        jnp.asarray([[0.0], [1.0], [2.0]]),
        cost=costs,
        forbidden=jnp.asarray([[False, False, False], [False, True, False]]),
        cardinality=2,
    )

    assert result.valid
    assert jnp.array_equal(result.source_indices, jnp.asarray([0, 1]))
    assert jnp.array_equal(result.target_indices, jnp.asarray([0, 2]))
    assert jnp.allclose(result.total_cost, 1.0)


def test_injective_flow_uses_hausdorff_gram_jacobian():
    latent = DiagonalNormalLaw(jnp.zeros((1,)), jnp.ones((1,)), event_shape=(1,))
    law = InjectiveContinuousFlowLaw(
        latent,
        lambda z: jnp.asarray([z[0], 2.0 * z[0]]),
        lambda y: jnp.asarray([(y[0] + 2.0 * y[1]) / 5.0]),
        event_shape=(2,),
    )
    point = jnp.asarray([0.3, 0.6])
    result = law.log_prob_with_diagnostics(point)

    assert result.valid
    assert law.density_measure_kind == "hausdorff"
    assert jnp.allclose(
        result.log_prob,
        latent.log_prob(jnp.asarray([0.3])) - 0.5 * jnp.log(5.0),
    )
    assert law.log_prob(jnp.asarray([0.3, 0.5])) == -jnp.inf


def test_hybrid_flow_combines_counting_mass_and_conditional_density():
    left = DiagonalNormalLaw(jnp.asarray([-1.0]), jnp.asarray([0.5]), event_shape=(1,))
    right = DiagonalNormalLaw(jnp.asarray([1.0]), jnp.asarray([0.5]), event_shape=(1,))
    law = HybridFlowLaw(jnp.asarray([0.25, 0.75]), (left, right), mode_id="two-mode")

    assert jnp.allclose(
        law.log_prob(jnp.asarray(1), jnp.asarray([1.0])),
        jnp.log(0.75) + right.log_prob(jnp.asarray([1.0])),
    )
    sample = law.sample(jr.key(3), (16,))
    assert sample.mode.shape == (16,)
    assert sample.value.shape == (16, 1)


def test_trajectory_flow_is_a_finite_coefficient_density():
    layout = TrajectoryEventLayout.from_increments(jnp.asarray([0.0, 0.5, 1.0]), (1,))
    rank = layout.coefficient_layout.rank
    coefficients = DiagonalNormalLaw(
        jnp.zeros((rank,)),
        jnp.ones((rank,)),
        event_shape=(rank,),
    )
    law = TrajectoryFlowLaw(coefficients, layout)
    value = law.sample(jr.key(4))
    recovered, residual = layout.coefficients(value)

    assert law.density_measure_kind == "trajectory"
    assert residual < 1.0e-10
    assert jnp.allclose(
        law.log_prob(value),
        coefficients.log_prob(recovered) - layout.coefficient_layout.log_volume,
    )


def test_field_coefficient_log_prob_is_query_independent():
    coefficients = DiagonalNormalLaw(
        jnp.zeros((2,)),
        jnp.ones((2,)),
        event_shape=(2,),
    )

    def decoder(values, points):
        return values[..., 0, None] + values[..., 1, None] * points[:, 0]

    law = FiniteFieldFlowLaw(
        coefficients,
        decoder,
        field_space_id="linear-functions",
    )
    first = prepare_field_query(law, jnp.asarray([[0.0], [1.0]]), capacity=2)
    second = prepare_field_query(
        law,
        jnp.asarray([[0.0], [0.5], [1.0]]),
        capacity=3,
    )
    coefficient = jnp.asarray([0.2, -0.7])
    first_sample = law.sample_field(jr.key(5), first)
    second_sample = law.sample_field(jr.key(5), second)

    assert jnp.allclose(
        law.coefficient_log_prob(coefficient), coefficients.log_prob(coefficient)
    )
    assert jnp.allclose(first_sample.coefficients, second_sample.coefficients)
    assert jnp.allclose(first_sample.values[0], second_sample.values[0])
    assert jnp.allclose(first_sample.values[-1], second_sample.values[-1])
