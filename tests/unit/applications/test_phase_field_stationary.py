#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.applications.phase_field import (
    analytic_double_well_kink,
    DoubleWellKinkPlan,
    solve_double_well_kink,
)
from phydrax.equations import DoubleWellFreeEnergy


def test_stationary_double_well_kink_solves_sector_energy_and_stability():
    plan = DoubleWellKinkPlan(
        jnp.linspace(-8.0, 8.0, 129),
        DoubleWellFreeEnergy(1.0),
        gradient_coefficient=1.0,
        maximum_newton_steps=24,
        residual_tolerance=1e-10,
    )
    result = solve_double_well_kink(plan)
    assert bool(result.converged)
    assert bool(result.evidence.accepted)
    np.testing.assert_allclose(result.evidence.topological_sector, 1.0, atol=1e-12)
    np.testing.assert_allclose(
        result.evidence.energy,
        2.0 * np.sqrt(2.0) / 3.0,
        rtol=3e-3,
    )
    assert int(result.evidence.negative_mode_count) == 0
    assert float(result.evidence.maximum_residual) < 1e-10
    np.testing.assert_allclose(result.field, analytic_double_well_kink(plan), atol=2e-3)


def test_kink_resolution_refines_toward_analytic_profile():
    errors = []
    for count in (65, 129):
        plan = DoubleWellKinkPlan(
            jnp.linspace(-8.0, 8.0, count),
            DoubleWellFreeEnergy(1.0),
            gradient_coefficient=1.0,
            residual_tolerance=1e-9,
        )
        result = solve_double_well_kink(plan)
        errors.append(
            float(jnp.max(jnp.abs(result.field - analytic_double_well_kink(plan))))
        )
    assert errors[1] < errors[0]
