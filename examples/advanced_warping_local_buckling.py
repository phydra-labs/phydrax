#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


mn = phx.applications.solid_mechanics.member_network
base = mn.BeamSection(1.0, 2.0, 1.0, 0.5, 0.8, 0.8)
section = mn.WarpingBeamSection(base, 0.1, 0.0, 3.0)
warping = mn.evaluate_warping_beam(
    5.0,
    0.01,
    jnp.asarray(((0.0, 0.0, 0.0), (0.1, 0.03, 0.05))),
    jnp.asarray((0.0, 0.02)),
    200_000.0,
    80_000.0,
    section,
    load_height_force=5.0,
)
geometry = mn.FiberSectionGeometry(
    jnp.asarray(((-0.5, 0.0), (0.5, 0.0))),
    jnp.asarray((0.5, 0.5)),
    jnp.asarray((0, 0)),
)
material = mn.BilinearFiberMaterial(200_000.0, 250.0, isotropic_hardening=2_000.0)
history = mn.FiberMaterialHistory.zeros(2, jnp.float64)
fiber, transaction = mn.evaluate_fiber_section(
    geometry,
    (material,),
    jnp.asarray((0.002, 0.001, 0.0)),
    mn.FiberSectionTransaction(history, history),
)
thin = mn.ThinWalledSection(
    jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 0.2))),
    jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
    jnp.asarray((0.01, 0.01)),
)
strip = mn.solve_finite_strip_buckling(
    mn.FiniteStripBucklingProblem(
        thin,
        200_000.0,
        0.3,
        jnp.asarray((-100.0, -100.0)),
        jnp.geomspace(0.1, 10.0, 40),
    )
)
collapse = mn.classify_collapse(
    1.0,
    jnp.asarray((1.0, 2.0)),
    yielded=fiber.yielded,
    fractured=fiber.fractured,
)
print("bimoment", warping.bimoment)
print("fiber axial force", fiber.axial_force)
print("plastic dissipation", fiber.plastic_dissipation)
print("finite-strip factor", strip.critical_stress)
print("mode family", int(strip.family))
print("collapse event", int(collapse.event))
print("committed plastic strain", transaction.commit().committed.plastic_strain)
