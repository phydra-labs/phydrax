"""Analytic and algebraic qualification for advanced cosmology contracts."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    cosmo = phx.applications.cosmology
    distance = cosmo.FLRWDistancePlan(light_speed=1.0, order=96)
    milne = cosmo.FLRWBackground(1.0, 0.0, curvature_density=1.0)
    milne_result = distance.evaluate(milne, 1.0)
    logarithm = jnp.log(2.0)
    de_sitter = cosmo.FLRWBackground(1.0, 0.0)
    de_sitter_result = distance.evaluate(de_sitter, 1.0)
    eds_nodes = jnp.geomspace(1.0e-2, 1.0, 48)
    eds_growth = cosmo.FLRWGrowthPlan(eds_nodes).solve(cosmo.FLRWBackground(1.0, 1.0))

    background = cosmo.FLRWBackground(1.0, 0.3)
    provenance = cosmo.CosmologyProductProvenance(
        producer="advanced-qualification",
        producer_version="native",
        model_form_id=background.model_form_id,
        request_id="qualification-power",
        numerical_policy_id="qualification-grid",
        physics_policy_id="linear-components",
        scale_id=background.scale.scale_id,
        source_kind="native",
        differentiability="native-parameter",
    )
    descriptor = lambda left, right: cosmo.MatterPowerDescriptor(left, right)
    common = (
        [0.5, 1.0],
        [1.0, 2.0],
        background.scale,
        provenance,
        background.realization,
    )
    cb = cosmo.MatterPowerTable(
        common[0],
        common[1],
        [[4.0, 4.0], [4.0, 4.0]],
        descriptor("cold_baryon", "cold_baryon"),
        common[2],
        common[3],
        common[4],
    )
    nu = cosmo.MatterPowerTable(
        common[0],
        common[1],
        [[1.0, 1.0], [1.0, 1.0]],
        descriptor("massive_neutrino_total", "massive_neutrino_total"),
        common[2],
        common[3],
        common[4],
    )
    cross = cosmo.MatterPowerTable(
        common[0],
        common[1],
        [[2.0, 2.0], [2.0, 2.0]],
        descriptor("cold_baryon", "massive_neutrino_total"),
        common[2],
        common[3],
        common[4],
    )
    total = cosmo.reconstruct_total_matter_power(cb, nu, cross, 0.8, 0.2)

    primordial = cosmo.PrimordialPowerLaw(2.1e-9, 0.965, 0.05)
    primordial_gradient = jax.grad(
        lambda amplitude: cosmo.PrimordialPowerLaw(amplitude, 0.965, 0.05).scalar_power(
            0.1
        )
    )(jnp.asarray(2.1e-9))
    force = cosmo.PeriodicImageForcePlan(
        (1.0, 1.0, 1.0), 1.0, softening=0.02, image_shells=1
    )
    positions = jnp.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    masses = jnp.ones((2,))
    reference = force.acceleration(positions, masses)
    force_result = force.qualify(positions, masses, reference)

    report = {
        "milne_radial_error": float(
            jnp.abs(milne_result.radial_comoving_distance - logarithm)
        ),
        "milne_transverse_error": float(
            jnp.abs(milne_result.transverse_comoving_distance - jnp.sinh(logarithm))
        ),
        "de_sitter_radial_error": float(
            jnp.abs(de_sitter_result.radial_comoving_distance - 1.0)
        ),
        "distance_duality_error": float(
            jnp.abs(
                milne_result.luminosity_distance
                - 4.0 * milne_result.angular_diameter_distance
            )
        ),
        "eds_first_growth_error": float(
            jnp.max(jnp.abs(eds_growth.first_order_growth - eds_nodes))
        ),
        "eds_second_growth_error": float(
            jnp.max(jnp.abs(eds_growth.second_order_growth - (3.0 / 7.0) * eds_nodes**2))
        ),
        "neutrino_total_power_error": float(jnp.max(jnp.abs(total.power_values - 3.24))),
        "primordial_at_pivot": float(primordial.scalar_power(0.05)),
        "primordial_amplitude_gradient": float(primordial_gradient),
        "force_qualification": bool(force_result.successful),
        "force_net_momentum_defect": float(
            jnp.max(jnp.abs(force_result.reference_net_force))
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
