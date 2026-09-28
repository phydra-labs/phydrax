#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Discrete dispersion of a magnetized plasma and Cherenkov regimes of the Yee grid.

The audit linearizes the executed compatible leapfrog step inside a homogeneous
periodic region and eigen-analyses the one-step Bloch map ``M(k)``. At ``k = 0`` the
magnetized-plasma branches sit at the cold-plasma cutoffs ``ω_R``, ``ω_L``, ``ω_p``.
A charge moving at ``0.95c`` through vacuum shows numerical-only Cherenkov
resonances near the grid cutoff; in an ``ε = 4`` dielectric it shows the physical cone
``cos θ = 1/(βn)``.
"""

import jax.numpy as jnp
import numpy as np

import phydrax as phx


mx = phx.solver.maxwell
cells = 8
grid = phx.discretization.TensorGridPlan(
    tuple(phx.discretization.UniformCellAxisSpec(cells, periodic=True) for _ in range(3)),
    axis_names=("x", "y", "z"),
).prepare(jnp.asarray([[0.0] * 3, [1.0] * 3]))
bridge = phx.discretization.StructuredCochainBridge(grid)
region = mx.MaxwellMaterialRegion((0, 0, 0), (cells, cells, cells))


def audit(
    constitutive: mx.AbstractMaxwellConstitutivePlan, fraction: float
) -> mx.CompatibleMaxwellDispersionAudit:
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge, constitutive=constitutive
    ).prepare()
    return mx.CompatibleMaxwellDispersionAudit(
        runtime, region, fraction * float(runtime.cfl_limit)
    )


cyclotron = -2.0
plasma = audit(
    mx.MagnetizedColdPlasmaMaxwellConstitutivePlan(
        jnp.asarray([1.0]), jnp.asarray([[0.0, 0.0, cyclotron]])
    ),
    0.05,
)
at_rest = np.asarray(plasma.dispersion(np.zeros((1, 3))).angular_frequencies)[0].real
branches = np.unique(np.round(at_rest[at_rest > 1e-6], 6))
root = np.sqrt(cyclotron**2 + 4.0)
cutoffs = np.sort([0.5 * (root - abs(cyclotron)), 1.0, 0.5 * (root + abs(cyclotron))])
print("k = 0 plasma branches:", branches)
print("cold-plasma cutoffs:  ", np.round(cutoffs, 6))
print("cyclotron shift:", float(plasma.cyclotron_resonance_shift[0]))

velocity = jnp.asarray([0.95, 0.0, 0.0])
frequencies = jnp.asarray([1.0, 16.0])
for permittivity in (1.0, 4.0):
    dielectric = audit(mx.DiagonalMaxwellConstitutivePlan(permittivity=permittivity), 0.9)
    evidence = mx.CherenkovRegimePlan(dielectric, velocity, frequencies, 4).evaluate()
    print(f"ε = {permittivity}:")
    print("  physical emission:", np.asarray(evidence.physical_emission[:, 0]))
    print("  numerical only:   ", np.asarray(evidence.numerical_only[:, 0]))
    print("  physical cone:    ", np.asarray(evidence.physical_cone_angle[:, 0, 0]))
