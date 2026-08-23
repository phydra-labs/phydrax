import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


class _TableAmplitude(eqx.Module):
    values: jax.Array

    def __call__(self, configuration):
        bits = (configuration > 0).astype(jnp.int32)
        index = 2 * bits[0] + bits[1]
        value = self.values[index]
        return phx.operators.LogAmplitude(jnp.log(jnp.abs(value)), value / jnp.abs(value))


def _z2(characters):
    return phx.operators.FiniteSignedPermutationSymmetry(
        jnp.asarray([[0, 1], [0, 1]]),
        jnp.asarray([[1, 1], [-1, -1]]),
        jnp.asarray(characters),
        symmetry_id=f"z2-{characters[1]}",
    )


def _value(amplitude):
    return jnp.where(
        amplitude.nonzero,
        jnp.exp(amplitude.log_abs) * amplitude.phase,
        0.0j,
    )


def test_even_and_odd_projection_obey_sector_characters():
    model = _TableAmplitude(
        jnp.asarray([1.0 + 0.2j, 0.5 - 0.1j, -0.25 + 0.8j, 1.3 + 0.4j])
    )
    configuration = jnp.asarray([1, -1])
    flipped = -configuration
    even = phx.operators.SymmetryProjectedAmplitude(model, _z2([1.0, 1.0]))
    odd = phx.operators.SymmetryProjectedAmplitude(model, _z2([1.0, -1.0]))

    assert jnp.allclose(_value(even(configuration)), _value(even(flipped)))
    assert jnp.allclose(_value(odd(configuration)), -_value(odd(flipped)))
    expected_even = 0.5 * (_value(model(configuration)) + _value(model(flipped)))
    expected_odd = 0.5 * (_value(model(configuration)) - _value(model(flipped)))
    assert jnp.allclose(_value(even(configuration)), expected_even)
    assert jnp.allclose(_value(odd(configuration)), expected_odd)


def test_symmetry_projection_preserves_finite_parameter_gradients():
    configuration = jnp.asarray([1, -1])
    symmetry = _z2([1.0, 1.0])

    def objective(values):
        projected = phx.operators.SymmetryProjectedAmplitude(
            _TableAmplitude(values), symmetry
        )(configuration)
        return projected.log_abs

    values = jnp.asarray([1.0 + 0.1j, 0.7 - 0.2j, 0.5 + 0.3j, 1.2 - 0.4j])
    gradient = jax.grad(objective)(values)

    assert jnp.all(jnp.isfinite(gradient))


def test_signed_permutation_group_and_character_laws_are_validated():
    with pytest.raises(ValueError, match="closed"):
        phx.operators.FiniteSignedPermutationSymmetry(
            jnp.asarray([[0, 1], [1, 0]]),
            jnp.asarray([[1, 1], [-1, 1]]),
            jnp.asarray([1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="one-dimensional representation"):
        _z2([1.0, 1.0j])
    with pytest.raises(ValueError, match="site permutation"):
        phx.operators.FiniteSignedPermutationSymmetry(
            jnp.asarray([[0, 0]]),
            jnp.ones((1, 2)),
            jnp.ones((1,)),
        )
