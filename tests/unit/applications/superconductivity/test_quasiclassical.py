#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _plans():
    velocities = 1.0e6 * jnp.asarray(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)))
    fermi = phx.applications.superconductivity.FermiSurfacePlan(
        velocities,
        jnp.full((4,), 0.25),
        jnp.ones((1, 4), dtype=jnp.complex128),
        ("s-wave",),
    )
    trajectories = phx.applications.superconductivity.RiccatiTrajectoryPlan(
        fermi, jnp.full((4, 3), 1.0e-9), tolerance=1.0e-9
    )
    matsubara = phx.applications.superconductivity.MatsubaraQuadraturePlan(1.0e-23, 8)
    gap = 2.0e-22
    omega = matsubara.frequencies
    pair = (
        2.0
        * jnp.pi
        * matsubara.temperature_energy
        * jnp.sum(gap / jnp.sqrt(omega**2 + gap**2))
    )
    coupling = gap / pair
    equilibrium = phx.applications.superconductivity.QuasiclassicalSuperconductivityPlan(
        trajectories,
        matsubara,
        jnp.asarray(((coupling,),)),
        damping=1.0,
        iterations=2,
        tolerance=1.0e-8,
    )
    return trajectories, matsubara, equilibrium, gap


def test_uniform_riccati_solution_normalizes_and_closes_gap_equation():
    trajectories, matsubara, equilibrium, gap = _plans()
    gap_field = gap * jnp.ones((4, 3), dtype=jnp.complex128)
    propagator = trajectories.evaluate(matsubara, gap_field)
    result = equilibrium.solve(jnp.asarray((gap + 0.0j,)))

    assert bool(propagator.evidence.successful)
    assert propagator.evidence.normalization_residual < 1.0e-12
    assert bool(result.evidence.successful)
    np.testing.assert_allclose(result.channel_amplitudes, (gap,), rtol=1.0e-10)
    np.testing.assert_allclose(result.current_density, 0.0, atol=1.0e-12)


def test_retarded_spectroscopy_is_separate_causal_real_axis_profile():
    _, _, equilibrium_plan, gap = _plans()
    equilibrium = equilibrium_plan.solve(jnp.asarray((gap + 0.0j,)))
    spectroscopy = phx.applications.superconductivity.RetardedSpectroscopyPlan(
        equilibrium_plan,
        jnp.asarray((-3.0 * gap, 0.0, 3.0 * gap)),
        broadening=0.01 * gap,
    ).evaluate(equilibrium)

    assert bool(spectroscopy.successful)
    assert spectroscopy.density_of_states[1] < 0.02
    assert spectroscopy.density_of_states[0] > 0.9
    assert spectroscopy.density_of_states[2] > 0.9
