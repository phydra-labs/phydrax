import numpy as np
import pytest

from phydrax.atomistic import AtomisticUnitSystem
from phydrax.chemistry.periodic._lattice_thermodynamics import (
    HarmonicThermodynamicsPlan,
    QuasiHarmonicPlan,
)


def test_stable_mode_thermodynamics_obeys_identity_and_classical_cv_limit():
    units = AtomisticUnitSystem.reduced()
    plan = HarmonicThermodynamicsPlan([0.4, 0.6], [1.0, 1000.0], units)
    result = plan.evaluate([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])

    assert bool(result.successful)
    assert float(result.thermodynamic_identity_residual) < 1.0e-10
    np.testing.assert_allclose(result.heat_capacity[-1], 3.0, rtol=2.0e-6)
    with pytest.raises(ValueError, match="strictly positive stable"):
        plan.evaluate([[0.0, 2.0, 3.0], [1.5, 2.5, 3.5]])


def test_qha_requires_and_recovers_strict_interior_minimum():
    units = AtomisticUnitSystem.reduced()
    thermodynamics = HarmonicThermodynamicsPlan([1.0], [1.0, 2.0, 3.0], units)
    volumes = np.arange(8.0, 13.0)
    frequencies = np.ones((5, 1, 1))
    result = QuasiHarmonicPlan(
        volumes,
        (volumes - 10.0) ** 2,
        frequencies,
        thermodynamics,
    ).evaluate()
    np.testing.assert_allclose(result.equilibrium_volumes, 10.0, atol=1.0e-12)
    np.testing.assert_allclose(result.volumetric_thermal_expansion, 0.0, atol=1.0e-12)

    with pytest.raises(ValueError, match="strictly interior"):
        QuasiHarmonicPlan(
            volumes,
            (volumes - 8.0) ** 2,
            frequencies,
            thermodynamics,
        ).evaluate()
