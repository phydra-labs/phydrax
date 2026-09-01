"""Periodic isotropic power, cross-power, and phase-sensitive discrepancy."""

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    cosmo = phx.applications.cosmology
    shape = (8, 8, 8)
    coordinate = tuple((jnp.arange(count) + 0.5) / count for count in shape)
    x, y, _ = jnp.meshgrid(*coordinate, indexing="ij")
    field = jnp.cos(2.0 * jnp.pi * x) + 0.5 * jnp.cos(4.0 * jnp.pi * y)
    shifted = jnp.roll(field, 1, axis=0)
    maximum_k = jnp.sqrt(3.0) * jnp.pi * shape[0]
    shells = phx.discretization.PeriodicFourierShellPlan(
        shape,
        (1.0, 1.0, 1.0),
        jnp.linspace(0.0, maximum_k, 12),
    )
    background = cosmo.FLRWBackground(1.0, 0.3)
    provenance = cosmo.CosmologyProductProvenance(
        producer="fourier-statistics-example",
        producer_version="native",
        model_form_id=background.model_form_id,
        request_id="example-realized-field",
        numerical_policy_id=shells.plan_id,
        physics_policy_id="measured-total-matter-density-contrast",
        scale_id=background.scale.scale_id,
        source_kind="native",
        differentiation=cosmo.DifferentiationContract.native(),
    )
    estimator = cosmo.CosmologicalFieldSpectrumPlan(
        shells,
        cosmo.MatterPowerDescriptor("total_matter", "total_matter"),
    )
    estimate_0 = estimator.estimate_auto(
        0.5 * field,
        0.5,
        background.scale,
        background.realization,
        provenance,
        "field-a0.5",
    )
    estimate_1 = estimator.estimate_auto(
        field,
        1.0,
        background.scale,
        background.realization,
        provenance,
        "field-a1",
    )
    table = cosmo.stack_matter_power_estimates((estimate_0, estimate_1))
    discrepancy = cosmo.SpectralFieldDiscrepancyPlan(shells).evaluate(
        field,
        shifted,
        "field",
        "shifted-field",
    )
    print("valid_shells", int(jnp.sum(estimate_1.valid_shells)))
    print("mode_count", int(jnp.sum(estimate_1.mode_counts)))
    print("table_shape", table.power_values.shape)
    print("phase_sensitive_discrepancy", float(discrepancy.total_discrepancy))
    print("parseval_residual", float(discrepancy.parseval_residual))


if __name__ == "__main__":
    main()
