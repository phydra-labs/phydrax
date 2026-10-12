"""Intragranular fission gas diffusion of UO2.

Independent reference: the Booth equivalent-sphere release fraction
``f(tau) = 1 - 6/(pi^4 tau) sum_m (1 - exp(-m^2 pi^2 tau)) / m^4`` (Booth,
AECL-496, 1957; Speight, Nucl. Sci. Eng. 37 (1969) 180, as used by the BISON
Sifgrs first-stage regression) summed with 10^5 terms.
"""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.applications.nuclear_fuel._fission_gas import (
    AthermalRelease,
    FissionGasModel,
)
from phydrax.applications.nuclear_fuel._grain_boundary import AinscoughGrainGrowth
from phydrax.applications.nuclear_fuel._intragranular import (
    SpeightTrapping,
    TurnbullDiffusivity,
)
from tests._support.nuclear_fuel_gas import (
    advance_gas,
    PRODUCTION,
)


pytestmark = pytest.mark.strict_jax


def _booth_release_fraction(reduced_time: float) -> float:
    modes = np.arange(1, 100_001, dtype=np.float64)
    retained = np.sum(-np.expm1(-(modes**2) * np.pi**2 * reduced_time) / modes**4)
    return 1.0 - 6.0 / (np.pi**4 * reduced_time) * retained


@pytest.mark.parametrize(
    ("reduced_time", "steps"),
    [(1.0e-3, 1), (1.0e-2, 1), (1.0e-1, 4), (1.0, 7)],
    ids=["tau-1e-3", "tau-1e-2-one-step", "tau-1e-1-four-steps", "tau-1-seven-steps"],
)
def test_constant_diffusivity_release_matches_booth_series(
    reduced_time: float, steps: int
) -> None:
    diffusivity = 1.0e-18
    radius = 5.0e-6
    model = FissionGasModel(
        athermal=AthermalRelease(
            geometric_surface_to_volume=0.0, total_surface_to_volume=0.0
        ),
        mode_count=20,
        diffusivity=TurnbullDiffusivity(
            intrinsic_prefactor=diffusivity,
            intrinsic_activation_energy=0.0,
            irradiation_prefactor=0.0,
            athermal_coefficient=0.0,
        ),
        # No intragranular bubbles: Speight's D b / (b + g) reduces to D.
        trapping=SpeightTrapping(
            bubble_density_coefficient=0.0, bubble_density_offset=0.0
        ),
        grain_growth=AinscoughGrainGrowth(rate_prefactor=0.0),
    )
    state = model.initial_state((), grain_radius=radius)
    total_time = reduced_time * radius**2 / diffusivity
    boundary_arrival = 0.0
    bound = 0.0
    for _ in range(steps):
        result = advance_gas(
            model,
            state,
            temperature=1500.0,
            previous_temperature=1500.0,
            burnup=0.0,
            time_step=total_time / steps,
        )
        boundary_arrival += float(result.transfers.diffusion_to_boundary)
        bound += float(result.diagnostics.modal_truncation_bound)
        state = result.state
    produced = float(state.produced)
    model_fraction = boundary_arrival / produced
    exact = _booth_release_fraction(reduced_time)
    # Slowest-unresolved-mode closure retains at least the exact Booth tail.
    assert exact - model_fraction >= -1.0e-12
    assert exact - model_fraction <= bound / produced + 1.0e-12
    # The bound is informative: small against the released fraction itself.
    assert bound / produced < 0.1 * exact


def test_zero_diffusivity_does_not_release_unresolved_source() -> None:
    model = FissionGasModel(
        athermal=AthermalRelease(
            geometric_surface_to_volume=0.0, total_surface_to_volume=0.0
        ),
        mode_count=3,
        diffusivity=TurnbullDiffusivity(
            intrinsic_prefactor=0.0, irradiation_prefactor=0.0, athermal_coefficient=0.0
        ),
        grain_growth=AinscoughGrainGrowth(rate_prefactor=0.0),
    )
    result = advance_gas(
        model,
        model.initial_state((), grain_radius=5.0e-6),
        temperature=1500.0,
        previous_temperature=1500.0,
        burnup=0.0,
        time_step=1.0e4,
    )
    assert float(result.transfers.diffusion_to_boundary) <= 1.0e-14 * PRODUCTION * 1.0e4
    np.testing.assert_allclose(
        float(result.retained_total), PRODUCTION * 1.0e4, rtol=1.0e-14
    )
    assert bool(result.admissible)
