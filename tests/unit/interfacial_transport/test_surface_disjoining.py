#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.interfacial_transport as it


def _double_layer() -> it.DoubleLayerDisjoiningPressure:
    return it.DoubleLayerDisjoiningPressure(1.0, 0.05, 298.15, 78.5)


def test_double_layer_matches_debye_length_and_analytic_black_film() -> None:
    law = _double_layer()
    # 1 mM 1:1 electrolyte in water at 25 C has a Debye length of 9.6 nm.
    np.testing.assert_allclose(1.0 / float(law.debye_parameter_m_inv), 9.62e-9, rtol=2e-3)
    result = it.black_film_equilibrium(law, 1000.0, (1e-9, 1e-6))
    expected = np.log(float(law.contact_pressure_pa) / 1000.0) / float(
        law.debye_parameter_m_inv
    )
    assert int(result.status) == it.BlackFilmStatus.CONVERGED
    np.testing.assert_allclose(float(result.thickness_m), expected, rtol=1e-10)
    assert bool(result.stable)


def test_dlvo_branches_have_opposite_stability_and_missing_root_is_reported() -> None:
    law = it.CompositeDisjoiningPressure(
        (it.VanDerWaalsDisjoiningPressure(4e-20), _double_layer())
    )
    thickness = np.geomspace(1e-9, 1e-7, 4000)
    barrier = float(thickness[np.argmax(np.asarray(law.pressure(thickness)))])
    thick = it.black_film_equilibrium(law, 1000.0, (barrier, 1e-6))
    thin = it.black_film_equilibrium(law, 1000.0, (1e-9, barrier))
    for result in (thick, thin):
        assert int(result.status) == it.BlackFilmStatus.CONVERGED
        np.testing.assert_allclose(
            float(law.pressure(result.thickness_m)), 1000.0, rtol=1e-9
        )
    assert bool(thick.stable)
    assert not bool(thin.stable)
    assert float(thin.thickness_m) < barrier < float(thick.thickness_m)
    missing = it.black_film_equilibrium(law, 1e9, (barrier, 1e-6))
    assert int(missing.status) == it.BlackFilmStatus.NO_ROOT_IN_BRACKET
    assert not bool(missing.successful)


@pytest.mark.parametrize(
    "law",
    [
        it.VanDerWaalsDisjoiningPressure(-3e-20),
        _double_layer(),
        it.ShortRangeRepulsionPressure(1e5, 1e-9, exponent=9),
    ],
)
def test_disjoining_derivative_and_energy_are_consistent(
    law: it.AbstractDisjoiningPressure,
) -> None:
    thickness = jnp.geomspace(2e-9, 5e-8, 7)
    np.testing.assert_allclose(
        law.derivative(thickness), jax.vmap(jax.grad(law.pressure))(thickness), rtol=1e-10
    )
    np.testing.assert_allclose(
        jax.vmap(jax.grad(law.energy))(thickness), -law.pressure(thickness), rtol=1e-10
    )
    assert bool(law.monotone_decreasing())


def test_attractive_van_der_waals_is_not_monotone() -> None:
    assert not bool(it.VanDerWaalsDisjoiningPressure(1e-20).monotone_decreasing())
