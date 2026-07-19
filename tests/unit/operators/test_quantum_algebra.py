#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


SIGMA_X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = jnp.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _pauli_fields():
    time = phx.domain.TimeInterval(0.0, 1.0)
    return (
        time,
        time.Function()(SIGMA_X),
        time.Function()(SIGMA_Y),
        time.Function()(SIGMA_Z),
    )


def test_pauli_commutation_and_anticommutation_relations():
    _time, sigma_x, sigma_y, sigma_z = _pauli_fields()

    commutator = phx.operators.commutator(sigma_x, sigma_y)
    anticommutator = phx.operators.anticommutator(sigma_x, sigma_y)
    bracket = phx.operators.quantum_bracket(sigma_x, sigma_y)
    assert jnp.allclose(commutator.func(), 2.0j * SIGMA_Z)
    assert jnp.allclose(anticommutator.func(), jnp.zeros((2, 2)))
    assert jnp.allclose(bracket.func(), 2.0 * SIGMA_Z)
    assert jnp.allclose(phx.operators.hermiticity_residual(bracket).func(), 0.0)


def test_commutator_lie_and_leibniz_identities():
    _time, a, b, c = _pauli_fields()
    jacobi = (
        phx.operators.commutator(a, phx.operators.commutator(b, c))
        + phx.operators.commutator(b, phx.operators.commutator(c, a))
        + phx.operators.commutator(c, phx.operators.commutator(a, b))
    )
    leibniz_left = phx.operators.commutator(a, b @ c)
    leibniz_right = phx.operators.commutator(a, b) @ c + b @ phx.operators.commutator(a, c)

    assert jnp.allclose(phx.operators.commutator(a, b).func(), -phx.operators.commutator(b, a).func())
    assert jnp.allclose(jacobi.func(), jnp.zeros((2, 2)), atol=1e-12)
    assert jnp.allclose(leibniz_left.func(), leibniz_right.func(), atol=1e-12)
    assert jnp.allclose(phx.operators.commutator(a, a).func(), jnp.zeros((2, 2)))


def test_density_structure_residuals():
    time = phx.domain.TimeInterval(0.0, 1.0)
    density = time.Function()(0.5 * (jnp.eye(2) + SIGMA_X))

    assert jnp.allclose(phx.operators.hermiticity_residual(density).func(), 0.0)
    assert jnp.allclose(phx.operators.unit_trace_residual(density).func(), 0.0)


def test_quantum_bracket_is_jittable_and_parameter_differentiable():
    time = phx.domain.TimeInterval(0.0, 1.0)
    sigma_y = time.Function()(SIGMA_Y)

    def bracket_entry(theta):
        operator = time.Function()(theta * SIGMA_X)
        value = phx.operators.quantum_bracket(operator, sigma_y).func()
        return jnp.real(value[0, 0])

    assert jnp.allclose(jax.jit(jax.grad(bracket_entry))(1.7), 2.0, atol=1e-12)


def test_quantum_algebra_validates_shapes_and_hbar():
    time = phx.domain.TimeInterval(0.0, 1.0)
    square = time.Function()(jnp.eye(2))
    rectangular = time.Function()(jnp.ones((2, 3)))
    larger = time.Function()(jnp.eye(3))

    with pytest.raises(ValueError, match="square matrix"):
        phx.operators.commutator(square, rectangular).func()
    with pytest.raises(ValueError, match="dimensions must match"):
        phx.operators.commutator(square, larger).func()
    with pytest.raises(ValueError, match="hbar must be positive"):
        phx.operators.quantum_bracket(square, square, hbar=0.0)
    with pytest.raises(TypeError, match="hbar must be real"):
        phx.operators.quantum_bracket(square, square, hbar=1.0j)
    with pytest.raises(ValueError, match="hbar must be a scalar"):
        phx.operators.quantum_bracket(square, square, hbar=jnp.ones((2,)))
