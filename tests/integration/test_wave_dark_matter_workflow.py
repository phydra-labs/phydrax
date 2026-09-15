import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.cosmology._wave_dark_matter import (
    WaveDarkMatterPlan,
    WaveDarkMatterStepPolicy,
)


def test_small_soliton_like_wave_dark_matter_workflow_is_bounded():
    shape = (8, 8, 8)
    space = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for count in shape),
        axis_names=("x", "y", "z"),
        field_name="psi",
    ).prepare(tuple(phx.discretization.AxisDomain.periodic(0.0, 1.0) for _ in shape))
    background = phx.applications.cosmology.FLRWBackground(1.0, 1.0)
    policy = WaveDarkMatterStepPolicy(
        maximum_phase_radians=2.0,
        minimum_de_broglie_cells=2.0,
        norm_relative_tolerance=1.0e-7,
    )
    prepared = WaveDarkMatterPlan(
        1.0,
        jnp.asarray((0.5, 0.5001, 0.5002)),
        gravitational_constant=0.05,
        reduced_planck_constant=0.03,
        step_policy=policy,
    ).prepare(space, background)

    coordinates = jnp.meshgrid(
        *(axis.nodes for axis in space.axes),
        indexing="ij",
    )
    periodic_radius_squared = sum(
        jnp.minimum(jnp.abs(axis - 0.5), 1.0 - jnp.abs(axis - 0.5)) ** 2
        for axis in coordinates
    )
    psi = jnp.exp(-periodic_radius_squared / (2.0 * 0.16**2)).astype(jnp.complex128)
    psi = psi / jnp.sqrt(jnp.sum(space.quadrature_weights * jnp.abs(psi) ** 2))
    state = prepared.initialize(psi)

    result = eqx.filter_jit(prepared.solve)(state)

    assert bool(result.successful)
    assert bool(result.diagnostics.completed)
    assert int(result.diagnostics.accepted_steps) == 2
    assert int(result.diagnostics.first_failed_step) == -1
    assert bool(jnp.all(result.diagnostics.attempted))
    assert bool(jnp.all(result.diagnostics.accepted))
    assert bool(jnp.all(result.diagnostics.finite))
    assert bool(jnp.all(result.diagnostics.poisson_closed))
    assert bool(jnp.all(result.diagnostics.zero_mode_removed))
    assert bool(jnp.all(result.diagnostics.de_broglie_resolved))
    assert (
        jnp.max(result.diagnostics.maximum_kinetic_phase) < policy.maximum_phase_radians
    )
    assert (
        jnp.max(result.diagnostics.maximum_potential_phase) < policy.maximum_phase_radians
    )
    assert (
        jnp.max(result.diagnostics.norm_relative_error) < policy.norm_relative_tolerance
    )
    assert (
        jnp.max(result.diagnostics.poisson_relative_residual)
        < policy.poisson_relative_tolerance
    )
    assert jnp.max(jnp.abs(result.diagnostics.total_energy)) < 1.0
    np.testing.assert_allclose(
        result.diagnostics.mass,
        result.diagnostics.mass[0],
        rtol=policy.norm_relative_tolerance,
    )
