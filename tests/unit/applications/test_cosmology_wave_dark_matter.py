import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.cosmology._wave_dark_matter import (
    WaveDarkMatterPlan,
    WaveDarkMatterStepPolicy,
)


def _fourier_space(count=6):
    return phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(3)),
        axis_names=("x", "y", "z"),
        field_name="psi",
    ).prepare(tuple(phx.discretization.AxisDomain.periodic(0.0, 1.0) for _ in range(3)))


def _prepared(*, schedule=(1.0, 1.001), policy=None, gravity=0.1, hbar=0.05):
    space = _fourier_space()
    background = phx.applications.cosmology.FLRWBackground(1.0, 1.0)
    selected = (
        WaveDarkMatterStepPolicy(
            maximum_phase_radians=2.0,
            minimum_de_broglie_cells=2.0,
        )
        if policy is None
        else policy
    )
    prepared = WaveDarkMatterPlan(
        1.0,
        jnp.asarray(schedule),
        gravitational_constant=gravity,
        reduced_planck_constant=hbar,
        step_policy=selected,
    ).prepare(space, background)
    return space, background, prepared


def test_uniform_density_has_zero_mean_zero_potential_and_conserved_mass():
    space, _, prepared = _prepared(schedule=(1.0, 1.001, 1.002))
    state = prepared.initialize(jnp.ones(space.physical_shape, dtype=jnp.complex128))

    poisson = prepared.poisson(state)
    result = eqx.filter_jit(prepared.solve)(state)

    np.testing.assert_allclose(poisson.source, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(poisson.potential, 0.0, atol=2.0e-13)
    assert bool(poisson.successful)
    assert bool(result.successful)
    np.testing.assert_allclose(result.diagnostics.norm, 1.0, rtol=2.0e-12)
    np.testing.assert_allclose(result.diagnostics.mass, 1.0, rtol=2.0e-12)
    assert bool(jnp.all(result.diagnostics.accepted))
    assert result.diagnostics.dealiasing.kind == "oversampling"
    assert not result.diagnostics.dealiasing.exact


def test_free_plane_wave_has_analytic_cosmological_phase_and_norm():
    space, background, prepared = _prepared(schedule=(1.0, 1.0005, 1.001))
    x = space.axes[0].nodes[:, None, None]
    initial_psi = jnp.broadcast_to(
        jnp.exp(2j * jnp.pi * x),
        space.physical_shape,
    )
    state = prepared.initialize(initial_psi)

    result = eqx.filter_jit(prepared.solve)(state)

    drift = sum(
        background.drift_factor(start, end)
        for start, end in zip(
            prepared.scale_factors[:-1],
            prepared.scale_factors[1:],
            strict=True,
        )
    )
    expected_phase = (
        prepared.reduced_planck_constant
        * (2.0 * jnp.pi) ** 2
        * drift
        / (2.0 * prepared.boson_mass)
    )
    expected = initial_psi * jnp.exp(-1j * expected_phase)
    np.testing.assert_allclose(result.state.psi, expected, rtol=3.0e-11, atol=3.0e-11)
    np.testing.assert_allclose(
        result.diagnostics.norm,
        result.diagnostics.norm[0],
        rtol=3.0e-11,
        atol=3.0e-11,
    )
    assert bool(jnp.all(result.diagnostics.phase_resolved))
    assert bool(jnp.all(result.diagnostics.zero_mode_removed))


def test_manufactured_poisson_residual_closes_on_retained_modes():
    space, _, prepared = _prepared()
    x = space.axes[0].nodes[:, None, None]
    density = 1.0 + 0.1 * jnp.cos(2.0 * jnp.pi * x)
    psi = jnp.broadcast_to(jnp.sqrt(density).astype(jnp.complex128), space.physical_shape)

    poisson = prepared.poisson(prepared.initialize(psi))
    weighted_source_mean = jnp.sum(space.quadrature_weights * poisson.source)

    np.testing.assert_allclose(
        poisson.laplacian, poisson.source, rtol=2.0e-11, atol=2.0e-11
    )
    np.testing.assert_allclose(weighted_source_mean, 0.0, atol=2.0e-12)
    assert poisson.relative_residual < prepared.step_policy.poisson_relative_tolerance
    assert poisson.zero_mode_absolute < prepared.step_policy.zero_mode_absolute_tolerance
    assert bool(poisson.successful)


def test_fixed_grid_jvp_matches_free_wave_tangent_and_can_be_disabled():
    space, background, prepared = _prepared()
    x = space.axes[0].nodes[:, None, None]
    psi = jnp.broadcast_to(jnp.exp(2j * jnp.pi * x), space.physical_shape)
    tangent = 0.1j * psi
    state = prepared.initialize(psi)

    action = eqx.filter_jit(prepared.jvp)(state, tangent)
    drift = background.drift_factor(1.0, 1.001)
    phase = (
        prepared.reduced_planck_constant
        * (2.0 * jnp.pi) ** 2
        * drift
        / (2.0 * prepared.boson_mass)
    )
    np.testing.assert_allclose(
        action,
        tangent * jnp.exp(-1j * phase),
        rtol=4.0e-10,
        atol=4.0e-10,
    )

    disabled_policy = WaveDarkMatterStepPolicy(
        maximum_phase_radians=2.0,
        minimum_de_broglie_cells=2.0,
        differentiability="none",
    )
    _, _, disabled = _prepared(policy=disabled_policy)
    with pytest.raises(ValueError, match="disabled"):
        disabled.jvp(disabled.initialize(psi), tangent)


def test_phase_resolution_rejection_rolls_back_and_masks_later_steps():
    policy = WaveDarkMatterStepPolicy(
        maximum_phase_radians=1.0e-6,
        minimum_de_broglie_cells=2.0,
    )
    space, _, prepared = _prepared(
        schedule=(1.0, 1.001, 1.002),
        policy=policy,
    )
    x = space.axes[0].nodes[:, None, None]
    psi = jnp.broadcast_to(jnp.exp(2j * jnp.pi * x), space.physical_shape)
    state = prepared.initialize(psi)

    result = eqx.filter_jit(prepared.solve)(state)

    assert not bool(result.successful)
    np.testing.assert_allclose(result.state.psi, state.psi, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result.state.scale_factor, state.scale_factor)
    np.testing.assert_array_equal(result.diagnostics.attempted, (True, False))
    np.testing.assert_array_equal(result.diagnostics.accepted, (False, False))
    np.testing.assert_array_equal(result.diagnostics.step_status, (3, 1))
    assert int(result.diagnostics.first_failed_step) == 0


def test_unsupported_geometry_curvature_state_and_schedule_fail_closed():
    background = phx.applications.cosmology.FLRWBackground(1.0, 1.0)
    plan = WaveDarkMatterPlan(1.0, (1.0, 1.001))
    bounded = phx.discretization.TensorSpectralPlan(
        (phx.discretization.ChebyshevBasisPlan(6),),
        axis_names=("x",),
    ).prepare((phx.discretization.AxisDomain.interval(-1.0, 1.0),))
    with pytest.raises(ValueError, match="tensor Fourier"):
        plan.prepare(bounded, background)

    curved = phx.applications.cosmology.FLRWBackground(
        1.0,
        0.9,
        curvature_density=0.1,
        dark_energy_density=0.0,
    )
    with pytest.raises(ValueError, match="zero spatial curvature"):
        plan.prepare(_fourier_space(), curved)

    _, _, prepared = _prepared()
    with pytest.raises(TypeError, match="complex dtype"):
        prepared.initialize(jnp.ones(prepared.discretization.physical_shape))
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="first scheduled"):
        state = prepared.initialize(
            jnp.ones(prepared.discretization.physical_shape, dtype=jnp.complex128),
            0.9,
        )
        jax.block_until_ready(state.psi)
    with pytest.raises(ValueError, match="increasing"):
        WaveDarkMatterPlan(1.0, (1.0, 1.0))
