import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._dispersion_ts import (
    SOURCE_BOHR_ANGSTROM,
    source_clamp,
    SOURCE_HARTREE_ELECTRONVOLT,
    TSDispersionPotential,
    TSDispersionReference,
)


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
SYNTHETIC_R4R2 = np.asarray(
    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.1, 0.0, 2.8], dtype=np.float64
)


def _reference_table() -> np.ndarray:
    table = np.zeros((10, 2))
    table[0] = (0.0, 1.0)
    table[1] = (6.5, 4.5)
    table[6] = (46.6, 12.0)
    table[8] = (15.6, 5.4)
    return table


def test_dispparam_bounds_the_log_multiplier_and_keeps_padding_dummy() -> None:
    reference = TSDispersionReference(_reference_table(), implemented_species=(1, 6, 8))
    raw = jnp.asarray([[0.5, -0.2], [5.0, -6.0], [0.0, 0.0]])
    species = jnp.asarray([6, 8, 0])
    observed = np.asarray(reference.coefficients(raw, species))
    expected = _reference_table()[[6, 8, 0]] * np.exp(np.clip(np.asarray(raw), -4, 4))
    np.testing.assert_allclose(observed, expected, rtol=1e-14)
    np.testing.assert_array_equal(observed[2], (0.0, 1.0))
    gradient = jax.grad(lambda value: reference.coefficients(value, species).sum())(raw)
    np.testing.assert_array_equal(np.asarray(gradient)[1], 0.0)
    assert float(jax.grad(lambda x: source_clamp(x, -4.0, 4.0))(jnp.asarray(4.0))) == 1.0


def test_reference_coverage_and_dummy_are_validated() -> None:
    table = _reference_table()
    with pytest.raises(ValueError, match="positive reference"):
        TSDispersionReference(table, implemented_species=(1, 7))
    bad = table.copy()
    bad[0] = (0.0, 0.0)
    with pytest.raises(ValueError, match="padding dummy"):
        TSDispersionReference(bad, implemented_species=(1,))
    potential = TSDispersionPotential(
        a1=0.4, a2=3.0, s8=0.8, pair_cutoff=10.0, r4r2=SYNTHETIC_R4R2
    )
    with pytest.raises(ValueError, match="r4r2"):
        potential.require_coverage((1, 100))


def test_ts_combination_damping_units_and_route_counting() -> None:
    potential = TSDispersionPotential(
        a1=0.4, a2=3.0, s8=0.8, pair_cutoff=4.0, r4r2=SYNTHETIC_R4R2
    )
    coefficients = jnp.asarray([[20.0, 6.0], [8.0, 4.0]])
    species = jnp.asarray([6, 1])
    distance = 2.3
    observed = potential.route_energies(
        coefficients,
        species,
        jnp.asarray([0, 1, 0]),
        jnp.asarray([1, 0, 1]),
        jnp.asarray([distance, distance, 5.0]),
        jnp.asarray([True, True, True]),
    )
    r4r2 = SYNTHETIC_R4R2
    c6 = 2 * 20.0 * 8.0 / (20.0 * 4.0 / 6.0 + 8.0 * 6.0 / 4.0)
    rr = 3 * r4r2[6] * r4r2[1]
    r0 = 0.4 * np.sqrt(rr) + 3.0
    bohr = distance / SOURCE_BOHR_ANGSTROM
    pair = c6 * (1.0 / (bohr**6 + r0**6) + 0.8 * rr / (bohr**8 + r0**8))
    expected = -0.5 * SOURCE_HARTREE_ELECTRONVOLT * pair
    np.testing.assert_allclose(np.asarray(observed[:2]), [expected, expected], rtol=1e-6)
    # Beyond the hard pair radius a route contributes nothing.
    assert float(observed[2]) == 0.0
    doubled = potential.route_energies(
        coefficients,
        species,
        jnp.asarray([0]),
        jnp.asarray([1]),
        jnp.asarray([distance]),
        jnp.asarray([True]),
        multiplicity=jnp.asarray([2]),
    )
    np.testing.assert_allclose(doubled[0], observed[:2].sum(), rtol=1e-14)


def test_connected_coefficient_and_distance_response() -> None:
    potential = TSDispersionPotential(
        a1=0.4, a2=3.0, s8=0.8, pair_cutoff=8.0, r4r2=SYNTHETIC_R4R2
    )
    species = (jnp.asarray(6), jnp.asarray(8))

    def energy(c6: jnp.ndarray, distance: jnp.ndarray) -> jnp.ndarray:
        return potential.pair_energy(
            (jnp.stack((c6, 6.0)), species[0]),
            (jnp.asarray([15.0, 5.0]), species[1]),
            distance,
        )

    c6, distance = jnp.asarray(30.0), jnp.asarray(3.1)
    gradient = jax.grad(energy, argnums=(0, 1))(c6, distance)
    step = 1e-6
    np.testing.assert_allclose(
        gradient[0],
        (energy(c6 + step, distance) - energy(c6 - step, distance)) / (2 * step),
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        gradient[1],
        (energy(c6, distance + step) - energy(c6, distance - step)) / (2 * step),
        rtol=1e-7,
    )
