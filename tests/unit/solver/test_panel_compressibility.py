import numpy as np
import pytest

import phydrax as phx


def test_subsonic_panel_pressure_corrections_match_declared_relations():
    pressure = np.asarray((0.2, -0.4))
    prandtl_glauert = phx.solver.PanelCompressibilityPolicy("prandtl-glauert", 0.8)
    karman_tsien = phx.solver.PanelCompressibilityPolicy("karman-tsien", 0.8)

    np.testing.assert_allclose(
        prandtl_glauert.correct_pressure_coefficient(pressure),
        pressure / 0.6,
        rtol=1.0e-7,
    )
    expected = pressure / (0.6 + 0.8**2 * pressure / (2.0 * (1.0 + 0.6)))
    np.testing.assert_allclose(
        karman_tsien.correct_pressure_coefficient(pressure),
        expected,
        rtol=1.0e-7,
    )


def test_panel_compressibility_rejects_sonic_and_supersonic_postprocessing():
    with pytest.raises(ValueError, match="strictly subsonic"):
        phx.solver.PanelCompressibilityPolicy("prandtl-glauert", 1.0)
    with pytest.raises(ValueError, match="strictly subsonic"):
        phx.solver.PanelCompressibilityPolicy("karman-tsien", 1.2)
