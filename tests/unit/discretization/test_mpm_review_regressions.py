from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.mpm._types import MPMRejectionReason, MPMRunStatus
from tests.unit.discretization.test_mpm_predictor_split import _case


def test_split_increment_work_reaches_energy_ledger() -> None:
    compiled, arguments, state = _case()
    dynamics = compiled.dynamics
    baseline = dynamics.predict_grid(state, 0.001, arguments)
    acceleration = jnp.asarray((0.3, -0.2), dtype=jnp.float64)
    predictor = dynamics.predict_grid(
        state,
        0.001,
        arguments,
        nodal_force_increment=baseline.grid.mass[..., None] * acceleration,
    )
    result = dynamics.finalize_grid_velocity(predictor, predictor.grid.velocity)
    assert result.successful
    energy = result.diagnostics.energy
    expected = (
        jnp.sum(
            dynamics.particles.safe_masses
            * jnp.sum(
                acceleration * (state.particles.velocity + 0.0005 * acceleration), axis=-1
            )
        )
        * 0.001
    )
    np.testing.assert_allclose(energy.external_work, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(energy.balance_defect, 0.0, atol=1e-12)


def test_split_stress_increment_work_reaches_energy_ledger() -> None:
    compiled, arguments, state = _case()
    dynamics = compiled.dynamics
    baseline = dynamics.predict_grid(state, 0.001, arguments)
    increment = jnp.broadcast_to(
        jnp.asarray(((0.4, 0.1), (0.1, -0.2)), dtype=jnp.float64),
        state.particles.first_piola.shape,
    )
    predictor = dynamics.predict_grid(
        state, 0.001, arguments, first_piola_increment=increment
    )
    result = dynamics.finalize_grid_velocity(predictor, predictor.grid.velocity)
    assert result.successful
    increment_force = predictor.grid.internal_force - baseline.grid.internal_force
    expected = 0.001 * jnp.sum(
        increment_force * 0.5 * (predictor.grid.velocity_before + predictor.grid.velocity)
    )
    np.testing.assert_allclose(
        result.diagnostics.energy.external_work, expected, rtol=1e-12, atol=1e-15
    )


def test_split_boundary_failure_is_predictor_failure() -> None:
    compiled, arguments, state = _case(clamp_x=True)
    dynamics = compiled.dynamics
    assert dynamics.boundary is not None
    dynamics = eqx.tree_at(
        lambda value: value.boundary.values,
        dynamics,
        jnp.full_like(dynamics.boundary.values, jnp.nan),
    )
    predictor = dynamics.predict_grid(state, 0.001, arguments)
    assert not predictor.successful
    assert predictor.status == int(MPMRunStatus.NONFINITE_STATE)
    assert predictor.rejection_reasons & int(MPMRejectionReason.NONFINITE)
    result = dynamics.finalize_grid_velocity(predictor, predictor.grid.velocity)
    assert not result.successful
    np.testing.assert_array_equal(
        result.accepted_state.particles.position, state.particles.position
    )
