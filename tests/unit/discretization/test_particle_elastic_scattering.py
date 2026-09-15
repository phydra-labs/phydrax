import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


scatter_elastic_pairs = phx.discretization.particle.scatter_elastic_pairs


def test_unequal_mass_elastic_scattering_reports_exact_pair_invariants():
    first = jnp.asarray(((2.0, -1.0, 0.5), (-0.4, 0.2, 0.8)))
    second = jnp.asarray(((-1.0, 0.5, -0.25), (0.3, -0.7, 0.1)))
    first_mass = jnp.asarray((2.0, 7.0))
    second_mass = jnp.asarray((5.0, 3.0))
    direction = jnp.asarray(((0.0, 3.0, 0.0), (1.0, 1.0, -1.0)))

    result = jax.jit(scatter_elastic_pairs)(
        first, second, first_mass, second_mass, direction
    )

    assert jnp.all(result.successful)
    assert jnp.all(result.scattered)
    np.testing.assert_allclose(result.momentum_after, result.momentum_before, atol=2e-14)
    np.testing.assert_allclose(
        result.kinetic_energy_after, result.kinetic_energy_before, atol=2e-14
    )
    np.testing.assert_allclose(result.momentum_defect, 0.0, atol=2e-14)
    np.testing.assert_allclose(result.kinetic_energy_defect, 0.0, atol=2e-14)
    center_before = (first_mass[:, None] * first + second_mass[:, None] * second) / (
        first_mass + second_mass
    )[:, None]
    center_after = (
        first_mass[:, None] * result.first_velocity
        + second_mass[:, None] * result.second_velocity
    ) / (first_mass + second_mass)[:, None]
    np.testing.assert_allclose(center_after, center_before, atol=2e-14)


def test_zero_relative_speed_and_masks_are_exact_no_injection_paths():
    first = jnp.asarray(((1.0, 2.0, 3.0), (4.0, 0.0, -1.0)))
    second = jnp.asarray(((1.0, 2.0, 3.0), (-2.0, 1.0, 0.0)))
    result = scatter_elastic_pairs(
        first,
        second,
        jnp.asarray((2.0, 3.0)),
        jnp.asarray((5.0, 7.0)),
        jnp.asarray(((0.0, 0.0, 0.0), (jnp.nan, 0.0, 0.0))),
        mask=jnp.asarray((True, False)),
    )

    assert result.successful.tolist() == [True, True]
    assert result.scattered.tolist() == [True, False]
    np.testing.assert_array_equal(result.first_velocity, first)
    np.testing.assert_array_equal(result.second_velocity, second)
    np.testing.assert_array_equal(result.momentum_defect, 0.0)
    np.testing.assert_array_equal(result.kinetic_energy_defect, 0.0)


def test_selected_nonpositive_mass_fails_closed_without_mutating_pair():
    first = jnp.asarray(((1.0, 0.0, 0.0),))
    second = jnp.asarray(((-1.0, 0.0, 0.0),))
    result = scatter_elastic_pairs(
        first,
        second,
        jnp.asarray((0.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray(((0.0, 1.0, 0.0),)),
    )

    assert not bool(result.successful[0])
    assert not bool(result.mass_valid[0])
    assert not bool(result.scattered[0])
    np.testing.assert_array_equal(result.first_velocity, first)
    np.testing.assert_array_equal(result.second_velocity, second)
