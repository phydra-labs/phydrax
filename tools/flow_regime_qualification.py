#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    species = phx.discretization.dsmc.DSMCSpeciesPlan(
        ("A",),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray((0.5,)),
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
    )
    collision = phx.discretization.dsmc.DSMCVHSCollisionPlan(
        species,
        phx.discretization.dsmc.DSMCPairCollisionParameters(
            jnp.asarray(((1.0,),)),
            jnp.asarray(((0.5,),)),
            boltzmann_constant=1.0,
        ),
    )
    particles = phx.discretization.dsmc.DSMCParticleState(
        jnp.zeros((2, 3)),
        jnp.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.zeros((2,)),
        jnp.zeros((2,)),
        jnp.ones((2,)),
        jnp.zeros((2,), dtype=jnp.int32),
        jnp.ones((2,), dtype="bool"),
        jnp.zeros((2,), dtype=jnp.int32),
    )
    required = (
        collision.collision_cross_section(
            jnp.asarray((0,)), jnp.asarray((0,)), jnp.asarray((2.0,))
        )
        * 2.0
    )
    collided = collision.collide(
        particles,
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1,), dtype=jnp.int32),
        jnp.asarray((True,)),
        jnp.asarray(((0.0, 0.25, 0.5),)),
        1.01 * required,
    )

    conversion = phx.solver.ContinuumToDSMCConversionPlan(8, 3).convert(
        jax.random.key(0), 10.0, jnp.asarray((1.0, 0.0, 0.0)), 5.0, 2.0
    )
    thin = phx.solver.ThinEDLElectroosmoticSlipPlan(
        permittivity=7.0e-10,
        dynamic_viscosity=1.0e-3,
        zeta_potential=-0.05,
        characteristic_length=1.0e-4,
        bulk_conductivity=1.0,
    ).evaluate(
        jnp.asarray((100.0, 100.0)),
        jnp.asarray((1.0, -1.0)),
        300.0,
        jnp.asarray((1000.0, 0.0)),
        jnp.asarray((0.0, 1.0)),
    )
    fluid = phx.applications.thermofluids.HydraulicFluidProperties(
        density=1000.0,
        dynamic_viscosity=1.0e-3,
        vapor_pressure=2000.0,
        temperature=300.0,
        provenance="synthetic-qualification-fluid",
    )
    hydraulic = phx.applications.thermofluids.HydraulicChannelPlan.circular(
        fluid, radius=1.0e-3, length=0.1
    ).evaluate(1.0e-7, 101425.0, 101325.0)
    mf = phx.applications.microfluidics
    topology = mf.DLDTopology(
        row_count=4,
        column_count=1,
        period_rows=4,
        outlet_count=2,
        particle_capacity=2,
    )
    design = mf.DLDDesign(
        post_radius=0.1,
        axial_pitch=1.0,
        lateral_pitch=1.0,
        row_shift=0.25,
        channel_lower=0.0,
        channel_upper=3.0,
        first_row_x=1.0,
        outlet_x=5.0,
        depth=0.1,
        length_unit_id="mm",
    )
    geometry = mf.DLDGeometryPlan(topology, design).evaluate(
        jnp.asarray(((1.0, 0.4),)), jnp.asarray((0.05,))
    )
    screen = mf.DLDEmpiricalScreenPlan(
        coefficient=1.4,
        exponent=0.48,
        minimum_shift_fraction=0.1,
        maximum_shift_fraction=0.4,
        source_id="qualification-coefficient-set",
    ).evaluate(topology, design)

    checks = {
        "dsmc_pair_conservation": bool(collided.successful),
        "continuum_to_dsmc_moments": bool(conversion.header.globally_eligible),
        "thin_edl_admission": bool(thin.header.globally_eligible),
        "hydraulic_admission": bool(hydraulic.header.globally_eligible),
        "dld_geometry_admission": bool(geometry.header.globally_eligible),
        "dld_screen_admission": bool(screen.header.globally_eligible),
    }
    successful = all(checks.values())
    print(
        json.dumps(
            {
                "kind": "flow-regime-invariant-qualification",
                "scientific_status": "inconclusive-without-reference-manifest",
                "checks": checks,
                "maximum_dsmc_energy_defect": float(
                    jnp.max(jnp.abs(collided.energy_defect))
                ),
                "conversion_mass_defect": float(conversion.mass_defect),
                "successful": successful,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
