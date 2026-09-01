import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cosmology = phx.applications.cosmology


def _context(shells, *, fields=("total_matter", "total_matter"), stage="linear"):
    background = cosmology.FLRWBackground(1.0, 0.3)
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="measured-field",
        numerical_policy_id=shells.plan_id,
        physics_policy_id="measured-density-contrast",
        scale_id=background.scale.scale_id,
        source_kind="native",
        differentiation=cosmology.DifferentiationContract.native(),
    )
    descriptor = cosmology.MatterPowerDescriptor(
        fields[0], fields[1], stage=stage, spatial_dimension=len(shells.source_shape)
    )
    return background, provenance, descriptor


def test_auto_estimate_content_identity_and_table_stacking():
    shape = (8, 8)
    x, _ = jnp.meshgrid(
        (jnp.arange(8) + 0.5) / 8.0,
        (jnp.arange(8) + 0.5) / 8.0,
        indexing="ij",
    )
    field = jnp.cos(2.0 * jnp.pi * x)
    shells = phx.discretization.PeriodicFourierShellPlan(
        shape,
        (1.0, 1.0),
        jnp.linspace(0.0, jnp.sqrt(2.0) * 8.0 * jnp.pi, 10),
    )
    background, provenance, descriptor = _context(shells)
    plan = cosmology.CosmologicalFieldSpectrumPlan(shells, descriptor)
    first = plan.estimate_auto(
        0.5 * field,
        0.5,
        background.scale,
        background.realization,
        provenance,
        "field-0.5",
    )
    second = plan.estimate_auto(
        field,
        1.0,
        background.scale,
        background.realization,
        provenance,
        "field-1",
    )
    assert bool(first.successful) and bool(second.successful)
    assert first.content_id != second.content_id
    table = cosmology.stack_matter_power_estimates((first, second))
    assert table.power_values.shape[0] == 2
    np.testing.assert_allclose(
        table.power_values[0], 0.25 * table.power_values[1], rtol=1e-12
    )


def test_density_content_and_cross_power_contracts():
    shape = (8,)
    x = (jnp.arange(8) + 0.5) / 8.0
    contrast = 0.1 * jnp.cos(2.0 * jnp.pi * x)
    density = 2.0 * (1.0 + contrast)
    shells = phx.discretization.PeriodicFourierShellPlan(
        shape, (1.0,), [0.0, np.pi, 3.0 * np.pi, 8.0 * np.pi]
    )
    background, provenance, auto_descriptor = _context(shells)
    density_plan = cosmology.CosmologicalFieldSpectrumPlan(
        shells, auto_descriptor, density_convention="density"
    )
    estimate = density_plan.estimate_auto(
        density,
        1.0,
        background.scale,
        background.realization,
        provenance,
        "density-field",
    )
    direct = cosmology.CosmologicalFieldSpectrumPlan(
        shells, auto_descriptor
    ).estimate_auto(
        contrast,
        1.0,
        background.scale,
        background.realization,
        provenance,
        "contrast-field",
    )
    np.testing.assert_allclose(
        estimate.power_values[estimate.valid_shells],
        direct.power_values[direct.valid_shells],
        rtol=1e-12,
        atol=1e-30,
    )

    _, _, cross_descriptor = _context(
        shells, fields=("cold_baryon", "massive_neutrino_total")
    )
    cross = cosmology.CosmologicalFieldSpectrumPlan(
        shells, cross_descriptor
    ).estimate_cross(
        contrast,
        2.0 * contrast,
        1.0,
        background.scale,
        background.realization,
        provenance,
        ("cb", "nu"),
    )
    assert bool(cross.successful)
    np.testing.assert_allclose(
        cross.power_values[cross.valid_shells],
        2.0 * direct.power_values[direct.valid_shells],
        rtol=1e-12,
    )


def test_spectral_discrepancy_is_phase_sensitive_and_parseval_consistent():
    shape = (8, 8)
    x, _ = jnp.meshgrid(
        (jnp.arange(8) + 0.5) / 8.0,
        (jnp.arange(8) + 0.5) / 8.0,
        indexing="ij",
    )
    field = jnp.cos(2.0 * jnp.pi * x)
    shifted = jnp.roll(field, 1, axis=0)
    shells = phx.discretization.PeriodicFourierShellPlan(
        shape,
        (1.0, 1.0),
        jnp.linspace(0.0, jnp.sqrt(2.0) * 8.0 * jnp.pi, 10),
    )
    result = cosmology.SpectralFieldDiscrepancyPlan(shells).evaluate(
        field, shifted, "field", "shifted"
    )
    assert bool(result.successful)
    assert result.total_discrepancy > 0.0
    np.testing.assert_allclose(result.parseval_residual, 0.0, atol=1e-12)
    vector = result.as_theory_vector()
    assert vector.values.shape == result.shell_discrepancy.shape
    gradient = jax.grad(
        lambda amplitude: (
            cosmology.SpectralFieldDiscrepancyPlan(shells)
            .evaluate(amplitude * field, shifted, "field", "shifted")
            .total_discrepancy
        )
    )(jnp.asarray(1.0))
    assert jnp.isfinite(gradient)


def test_invalid_density_mean_and_estimate_stack_fail_closed():
    shells = phx.discretization.PeriodicFourierShellPlan(
        (4,), (1.0,), [0.0, np.pi, 3.0 * np.pi, 4.0 * np.pi]
    )
    background, provenance, descriptor = _context(shells)
    plan = cosmology.CosmologicalFieldSpectrumPlan(
        shells, descriptor, density_convention="density"
    )
    with pytest.raises((ValueError, RuntimeError), match="mean"):
        jax.block_until_ready(
            plan.estimate_auto(
                jnp.zeros((4,)),
                1.0,
                background.scale,
                background.realization,
                provenance,
                "zero-density",
            ).power_values
        )
