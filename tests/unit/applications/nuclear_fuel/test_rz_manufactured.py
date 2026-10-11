#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Manufactured RZ thermomechanics through the production weak forms."""

from __future__ import annotations

from tools.nuclear_fuel_qualification import rz_manufactured


def test_rz_manufactured_body_force_preserves_coupled_displacement() -> None:
    # Independent strong-form forcing b=-Div(P) is applied to the production
    # weak form. Reversing its sign produces a >1e-7 m displacement defect.
    measurement = rz_manufactured(4, 2)
    displacement_error = measurement["displacement_l2_m"]
    temperature_error = measurement["temperature_l2_K"]
    assert isinstance(displacement_error, float)
    assert isinstance(temperature_error, float)
    assert displacement_error < 1e-8
    assert temperature_error < 1.0
