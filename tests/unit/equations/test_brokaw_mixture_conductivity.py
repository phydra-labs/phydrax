"""Brokaw mixture rule for dilute-gas thermal conductivity (MATPRO GTHCON).

Independent reference: MATPRO-11 Rev. 1 (NUREG/CR-0497, 1979), GTHCON Fig. C-1.1,
"Thermal conductivity of xenon-helium mixtures at 520 C", GTHCON curve read from
the figure at xenon mole fractions 0.3, 0.5, 0.7 (reading uncertainty about
+-0.002 W/m/K). The pure-gas inputs are MATPRO-11 Eqs. C-1.4 (helium) and C-1.6
(xenon); the molecular weights are those of the GTHCON listing.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.equations import brokaw_mixture_conductivity


pytestmark = pytest.mark.strict_jax

# MATPRO-11 GTHCON listing molecular weights of He, Ar, Kr, Xe, N2 [g/mol].
_MASSES = np.asarray([4.003, 39.944, 83.80, 131.30, 28.8])
_CONDUCTIVITY = np.asarray([0.29, 0.042, 0.024, 0.0136, 0.058])


def _matpro11_helium_xenon(temperature: float) -> np.ndarray:
    helium = 3.366e-3 * temperature**0.668
    xenon = 4.0288e-5 * temperature**0.872
    return np.asarray([helium, xenon])


@pytest.mark.parametrize("species", range(5), ids=["He", "Ar", "Kr", "Xe", "N2"])
def test_single_species_mixture_is_the_pure_gas(species: int) -> None:
    mole = np.zeros(5)
    mole[species] = 1.0
    value = brokaw_mixture_conductivity(mole, _CONDUCTIVITY, _MASSES)
    np.testing.assert_allclose(value, _CONDUCTIVITY[species], rtol=1e-15)


def test_species_order_does_not_change_the_mixture() -> None:
    mole = np.asarray([0.55, 0.05, 0.1, 0.25, 0.05])
    reference = brokaw_mixture_conductivity(mole, _CONDUCTIVITY, _MASSES)
    for permutation in ([4, 3, 2, 1, 0], [3, 0, 4, 1, 2], [1, 2, 0, 4, 3]):
        permuted = brokaw_mixture_conductivity(
            mole[permutation], _CONDUCTIVITY[permutation], _MASSES[permutation]
        )
        np.testing.assert_allclose(permuted, reference, rtol=1e-14)


def test_absent_species_contribute_exactly_nothing() -> None:
    binary = brokaw_mixture_conductivity(
        np.asarray([0.8, 0.2]), _CONDUCTIVITY[[0, 3]], _MASSES[[0, 3]]
    )
    padded = brokaw_mixture_conductivity(
        np.asarray([0.8, 0.0, 0.0, 0.2, 0.0]), _CONDUCTIVITY, _MASSES
    )
    np.testing.assert_allclose(padded, binary, rtol=1e-15)
    gradient = jax.grad(
        lambda mole: brokaw_mixture_conductivity(mole, _CONDUCTIVITY, _MASSES)
    )(jnp.asarray([0.8, 0.0, 0.0, 0.2, 0.0]))
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize(
    ("xenon_fraction", "figure_value"),
    [(0.3, 0.1215), (0.5, 0.0700), (0.7, 0.0410)],
    ids=["x_Xe=0.3", "x_Xe=0.5", "x_Xe=0.7"],
)
def test_helium_xenon_matches_matpro_gthcon_figure(
    xenon_fraction: float, figure_value: float
) -> None:
    temperature = 520.0 + 273.15
    value = brokaw_mixture_conductivity(
        np.asarray([1.0 - xenon_fraction, xenon_fraction]),
        _matpro11_helium_xenon(temperature),
        np.asarray([4.003, 131.30]),
    )
    # Figure reading tolerance; Wilke's rule without the Brokaw mass factor is
    # 16-20 % higher at these compositions.
    np.testing.assert_allclose(value, figure_value, rtol=0.04)


def test_batched_compositions_broadcast_against_shared_species_data() -> None:
    xenon = np.linspace(0.0, 1.0, 7)
    mole = np.stack([1.0 - xenon, xenon], axis=-1)
    values = brokaw_mixture_conductivity(
        mole, _matpro11_helium_xenon(793.15), np.asarray([4.003, 131.30])
    )
    assert values.shape == (7,)
    assert np.all(np.diff(np.asarray(values)) < 0.0)


def test_species_axis_mismatch_is_refused() -> None:
    with pytest.raises(ValueError, match="species axis"):
        brokaw_mixture_conductivity(np.ones(4) / 4, _CONDUCTIVITY, _MASSES)


@pytest.mark.parametrize(
    "composition",
    [np.zeros(5), np.asarray([1.1, 0.0, 0.0, -0.1, 0.0]), np.full(5, np.nan)],
    ids=["empty-mixture", "negative-fraction", "nonfinite-fraction"],
)
def test_invalid_composition_is_not_a_zero_conductivity(composition: np.ndarray) -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="sum to one"):
        brokaw_mixture_conductivity(composition, _CONDUCTIVITY, _MASSES)
