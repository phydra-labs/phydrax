import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cosmology = phx.applications.cosmology


def _case(shape, *, order=1, dealiasing="none"):
    dimension = len(shape)
    capacity = int(np.prod(shape))
    scale = cosmology.CosmologyScaleContract("L", "M", "T")
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(capacity),
        jnp.full((capacity,), 1.0 / capacity),
        ambient_dimension=dimension,
    ).prepare()
    background = cosmology.FLRWBackground(1.0, 1.0, scale=scale)
    growth = cosmology.FLRWGrowthPlan(jnp.geomspace(1.0e-2, 1.0, 24)).solve(background)
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="test-power-request",
        numerical_policy_id="test-power",
        physics_policy_id="linear-cold-baryon-power",
        scale_id=scale.scale_id,
        source_kind="external",
        differentiability="constant",
    )
    maximum_k = np.sqrt(dimension) * np.pi * max(shape)
    k = jnp.linspace(1.0, maximum_k + 2.0, 128)
    base = 1.0e-8 / (1.0 + k**2)
    first_growth = growth.evaluate(0.1)[0]
    power = cosmology.MatterPowerTable(
        [0.1, 1.0],
        k,
        jnp.stack((first_growth**2 * base, base)),
        cosmology.MatterPowerDescriptor(
            "cold_baryon",
            "cold_baryon",
            spatial_dimension=dimension,
        ),
        scale,
        provenance,
        background.realization,
    )
    plan = cosmology.LagrangianPerturbationInitialConditionPlan(
        particles,
        shape,
        tuple(1.0 for _ in shape),
        order=order,
        dealiasing=dealiasing,
        scale=scale,
    )
    return plan, background, growth, power


def test_zero_noise_produces_lattice_and_zero_momentum():
    plan, background, growth, power = _case((4, 4, 4), order=2)
    result = plan.realize(background, growth, power, jnp.zeros(plan.shape), 0.1)
    axes = tuple((jnp.arange(4) + 0.5) / 4.0 for _ in range(3))
    lattice = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
    assert bool(result.successful)
    np.testing.assert_allclose(result.positions, lattice, atol=1e-14)
    np.testing.assert_allclose(result.canonical_momenta, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.second_order_displacement, 0.0, atol=1e-14)


def test_plane_wave_has_zero_second_order_source_and_edS_momentum():
    plan, background, growth, power = _case((8,), order=2)
    x = (jnp.arange(8) + 0.5) / 8.0
    noise = jnp.cos(2.0 * jnp.pi * x)
    result = plan.realize(background, growth, power, noise, 0.1)
    np.testing.assert_allclose(result.second_order_displacement, 0.0, atol=1e-12)
    _, first_rate, _, _ = growth.evaluate(0.1)
    expected = (
        plan.particles.safe_masses[:, None]
        * 0.1**2
        * background.hubble(0.1)
        * first_rate
        * result.first_order_displacement.reshape((-1, 1))
    )
    np.testing.assert_allclose(result.canonical_momenta, expected, rtol=1e-10)


def test_nonparallel_modes_generate_finite_second_order_displacement():
    plan, background, growth, power = _case((4, 4, 4), order=2, dealiasing="three_halves")
    coordinates = tuple((jnp.arange(4) + 0.5) / 4.0 for _ in range(3))
    x, y, _ = jnp.meshgrid(*coordinates, indexing="ij")
    noise = jnp.cos(2.0 * jnp.pi * x) + jnp.cos(2.0 * jnp.pi * y)
    result = plan.realize(background, growth, power, noise, 0.1)
    assert bool(result.successful)
    assert jnp.max(jnp.abs(result.second_order_displacement)) > 0.0
    assert jnp.all(jnp.isfinite(result.canonical_momenta))
    modes = jnp.fft.fftn(result.density_contrast)
    recovered = jnp.abs(modes) ** 2 / np.prod(plan.shape) ** 2
    np.testing.assert_allclose(result.power_spectrum, recovered, rtol=1e-10, atol=1e-40)


def test_lpt_rejects_unsupported_order_and_dimension_mismatch():
    plan, _, _, _ = _case((4,), order=1)
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        cosmology.LagrangianPerturbationInitialConditionPlan(
            plan.particles, (4,), (1.0,), order=3
        )
    scale = plan.scale
    background = cosmology.FLRWBackground(1.0, 1.0, scale=scale)
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="test",
        numerical_policy_id="test",
        physics_policy_id="test",
        scale_id=scale.scale_id,
        source_kind="external",
        differentiability="constant",
    )
    mismatched = cosmology.MatterPowerTable(
        [0.1, 1.0],
        [1.0, 10.0],
        [[1.0, 1.0], [1.0, 1.0]],
        cosmology.MatterPowerDescriptor(
            "cold_baryon", "cold_baryon", spatial_dimension=2
        ),
        scale,
        provenance,
        background.realization,
    )
    with pytest.raises(ValueError, match="dimensions disagree"):
        plan.realize(
            background,
            cosmology.FLRWGrowthPlan([0.1, 1.0]).solve(background),
            mismatched,
            jnp.ones((4,)),
            0.1,
        )
