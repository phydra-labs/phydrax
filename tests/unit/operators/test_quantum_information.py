#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _density_from_state(time, state):
    return phx.operators.density_from_factor(time.Function()(state[:, None]))


def test_purity_and_entropy_distinguish_pure_and_mixed_states():
    time = phx.domain.TimeInterval(0.0, 1.0)
    pure = _density_from_state(time, jnp.asarray([1.0, 0.0], dtype=complex))
    mixed = time.Function()(0.5 * jnp.eye(2, dtype=complex))

    assert jnp.allclose(phx.operators.purity(pure).func(), 1.0)
    assert jnp.allclose(phx.operators.purity(mixed).func(), 0.5)
    assert jnp.allclose(phx.operators.von_neumann_entropy(pure).func(), 0.0)
    assert jnp.allclose(phx.operators.von_neumann_entropy(mixed).func(), 1.0)
    assert jnp.allclose(
        phx.operators.von_neumann_entropy(mixed, base=jnp.e).func(),
        jnp.log(2.0),
    )


def test_bell_state_has_one_bit_of_entanglement_entropy():
    time = phx.domain.TimeInterval(0.0, 1.0)
    zero = time.Function()(jnp.asarray([1.0, 0.0], dtype=complex))
    one = time.Function()(jnp.asarray([0.0, 1.0], dtype=complex))
    bell = (
        phx.operators.tensor_product(zero, zero)
        + phx.operators.tensor_product(one, one)
    ) / jnp.sqrt(2.0)
    density = _density_from_state(time, bell.func())
    reduced = phx.operators.partial_trace(
        density,
        subsystem_dims=(2, 2),
        trace_out=1,
    )

    assert jnp.allclose(phx.operators.von_neumann_entropy(reduced).func(), 1.0)


def test_state_fidelity_preserves_domain_dependencies_and_known_values():
    time = phx.domain.TimeInterval(0.0, 1.0)

    @time.Function("t")
    def state(t):
        return jnp.asarray([jnp.cos(t), jnp.sin(t)], dtype=complex)

    zero = time.Function()(jnp.asarray([1.0, 0.0], dtype=complex))
    one = time.Function()(jnp.asarray([0.0, 1.0], dtype=complex))
    fidelity = phx.operators.state_fidelity(state, zero)

    assert fidelity.deps == ("t",)
    assert jnp.allclose(fidelity.func(0.37), jnp.cos(0.37) ** 2)
    assert jnp.allclose(phx.operators.state_fidelity(zero, zero).func(), 1.0)
    assert jnp.allclose(phx.operators.state_fidelity(zero, one).func(), 0.0)


def test_density_fidelity_matches_pure_and_commuting_state_formulas():
    time = phx.domain.TimeInterval(0.0, 1.0)
    zero = jnp.asarray([1.0, 0.0], dtype=complex)
    plus = jnp.asarray([1.0, 1.0], dtype=complex) / jnp.sqrt(2.0)
    zero_state = time.Function()(zero)
    plus_state = time.Function()(plus)
    zero_density = _density_from_state(time, zero)
    plus_density = _density_from_state(time, plus)

    state_value = phx.operators.state_fidelity(zero_state, plus_state).func()
    density_value = phx.operators.density_fidelity(zero_density, plus_density).func()
    assert jnp.allclose(density_value, state_value, atol=1e-12)

    p = 0.7
    q = 0.2
    left = time.Function()(jnp.diag(jnp.asarray([p, 1.0 - p])))
    right = time.Function()(jnp.diag(jnp.asarray([q, 1.0 - q])))
    expected = (jnp.sqrt(p * q) + jnp.sqrt((1.0 - p) * (1.0 - q))) ** 2
    assert jnp.allclose(phx.operators.density_fidelity(left, right).func(), expected)


def test_trace_distance_matches_orthogonal_and_commuting_state_formulas():
    time = phx.domain.TimeInterval(0.0, 1.0)
    zero = _density_from_state(time, jnp.asarray([1.0, 0.0], dtype=complex))
    one = _density_from_state(time, jnp.asarray([0.0, 1.0], dtype=complex))

    assert jnp.allclose(phx.operators.trace_distance(zero, zero).func(), 0.0)
    assert jnp.allclose(phx.operators.trace_distance(zero, one).func(), 1.0)

    p = 0.73
    q = 0.19
    left = time.Function()(jnp.diag(jnp.asarray([p, 1.0 - p])))
    right = time.Function()(jnp.diag(jnp.asarray([q, 1.0 - q])))
    assert jnp.allclose(phx.operators.trace_distance(left, right).func(), abs(p - q))


def test_information_measures_are_jittable_and_parameter_differentiable():
    time = phx.domain.TimeInterval(0.0, 1.0)
    q = 0.8
    target = time.Function()(jnp.diag(jnp.asarray([q, 1.0 - q])))

    def measures(p):
        density = time.Function()(jnp.diag(jnp.asarray([p, 1.0 - p])))
        return (
            phx.operators.purity(density).func(),
            phx.operators.von_neumann_entropy(density).func(),
            phx.operators.density_fidelity(density, target).func(),
        )

    def expected_fidelity(p):
        return (jnp.sqrt(p * q) + jnp.sqrt((1.0 - p) * (1.0 - q))) ** 2

    p = 0.37
    purity_gradient, entropy_gradient, fidelity_gradient = jax.jit(
        jax.jacrev(measures)
    )(p)

    assert jnp.allclose(purity_gradient, 4.0 * p - 2.0, atol=1e-12)
    assert jnp.allclose(
        entropy_gradient,
        jnp.log2((1.0 - p) / p),
        atol=1e-12,
    )
    assert jnp.allclose(
        fidelity_gradient,
        jax.grad(expected_fidelity)(p),
        atol=1e-11,
    )


def test_information_operators_reject_invalid_values():
    time = phx.domain.TimeInterval(0.0, 1.0)
    vector = time.Function()(jnp.ones((2,)))
    matrix = time.Function()(0.5 * jnp.eye(2))
    larger_vector = time.Function()(jnp.ones((3,)))
    larger_matrix = time.Function()(jnp.eye(3) / 3.0)
    nonhermitian = time.Function()(jnp.asarray([[0.5, 1.0], [0.0, 0.5]]))
    indefinite = time.Function()(jnp.diag(jnp.asarray([1.1, -0.1])))
    empty = time.Function()(jnp.empty((0, 0)))

    with pytest.raises(ValueError, match="square matrix"):
        phx.operators.purity(vector).func()
    with pytest.raises(ValueError, match="must be nonempty"):
        phx.operators.von_neumann_entropy(empty).func()
    with pytest.raises(eqx.EquinoxRuntimeError, match="must be Hermitian"):
        phx.operators.von_neumann_entropy(nonhermitian).func()
    with pytest.raises(eqx.EquinoxRuntimeError, match="positive semidefinite"):
        eqx.filter_jit(phx.operators.von_neumann_entropy(indefinite).func)()
    with pytest.raises(ValueError, match="positive and unequal to one"):
        phx.operators.von_neumann_entropy(matrix, base=1.0)
    with pytest.raises(ValueError, match="dimensions must match"):
        phx.operators.state_fidelity(vector, larger_vector).func()
    with pytest.raises(ValueError, match="dimensions must match"):
        phx.operators.density_fidelity(matrix, larger_matrix).func()
    with pytest.raises(ValueError, match="dimensions must match"):
        phx.operators.trace_distance(matrix, larger_matrix).func()
