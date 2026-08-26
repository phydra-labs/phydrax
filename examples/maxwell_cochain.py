#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


grid = phx.discretization.TensorGridPlan(
    tuple(phx.discretization.UniformCellAxisSpec(4) for _ in range(3)),
    axis_names=("x", "y", "z"),
).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
bridge = phx.discretization.StructuredCochainBridge(grid)
n0, n1, n2, _ = bridge.cochain.cell_counts
probe = phx.solver.maxwell.FieldProbePlan("electric", jnp.asarray([0, 1, 2]))
dft = phx.solver.maxwell.DFTObserverPlan(probe, jnp.asarray([2.0, 4.0]))
material = phx.solver.maxwell.DiagonalMaxwellConstitutivePlan(
    permittivity=1.0 + 0.2 * jnp.arange(n1) / n1,
    permeability=1.0 + 0.1 * jnp.arange(n2) / n2,
)
runtime = phx.solver.CompatibleMaxwellPlan(
    bridge,
    constitutive=material,
    observers=(probe, dft),
    pml=phx.solver.maxwell.MaxwellCPMLPlan(1),
).prepare()
electric = jnp.sin(jnp.arange(n1, dtype=float) / 13.0)
displacement = runtime.constitutive.electric_displacement(electric, None)
magnetic = bridge.exterior_derivative(1, electric)
charge = bridge.codifferential(1, displacement)
state = runtime.pack(displacement, magnetic, charge)
dt = 0.05 * runtime.stable_dt
for step in range(40):
    state = runtime.leapfrog_step(step * dt, state, dt)
report = runtime.diagnostics(40 * dt, state, step_size=dt)
probe_value, dft_value = runtime.observe(state)
print("energy", float(report.energy))
print("electric constraint", float(report.electric_constraint_linf))
print("magnetic constraint", float(report.magnetic_constraint_linf))
print("probe", probe_value)
print("DFT", dft_value)
