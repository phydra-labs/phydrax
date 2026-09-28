#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-domain Cherenkov radiation of a prescribed line charge in a dielectric.

A charge moves along the periodic ``z`` axis at ``β = 0.9`` through a medium of
index ``n = 1.5`` (``βn = 1.35 > 1``); ``y`` is two periodic cells with the charge midway, so the lattice
charge is a line charge ``λ = q/L_y``, and CPML absorbs along ``x``. The run
reports the continuity/Gauss/power-ledger evidence and compares the measured
cone angle ``tan θ = k_x/k_z`` of the first harmonic ``ω₁ = 2π v/L_z`` with
``cos θ = 1/(βn)``.
"""

from __future__ import annotations

from typing import Any

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import phydrax as phx


D = phx.discretization
PIC = phx.discretization.pic
mx = phx.solver.maxwell

cells_x, cells_z, spacing = 48, 24, 0.05
beta, index = 0.9, 1.5
grid = D.TensorGridPlan(
    (
        D.UniformCellAxisSpec(cells_x),
        D.UniformCellAxisSpec(2, periodic=True),
        D.UniformCellAxisSpec(cells_z, periodic=True),
    ),
    axis_names=("x", "y", "z"),
).prepare(
    jnp.asarray([[0.0, 0.0, 0.0], [cells_x * spacing, 2 * spacing, cells_z * spacing]])
)
bridge = D.StructuredCochainBridge(grid)
medium: dict[str, Any] = {
    "constitutive": mx.DiagonalMaxwellConstitutivePlan(permittivity=index**2),
    "pml": mx.MaxwellCPMLPlan((8, 0, 0)),
}
stable = float(phx.solver.CompatibleMaxwellPlan(bridge, **medium).prepare().stable_dt)
length_z = cells_z * spacing
period = length_z / beta
# CFL ≤ 0.45: the CPML power ledger is an O(Δt²) residual that stays below its
# 1e-2 default tolerance here (it is 3e-2 at CFL 0.9).
step = period / int(np.ceil(period / (0.45 * stable)))
periods = 8
times = step * np.arange(int(round(periods * period / step)) + 1)
x0 = 0.5 * cells_x * spacing
positions = np.stack(
    [np.asarray([[x0, 0.5 * spacing, 0.3 * spacing + beta * time]]) for time in times]
)

particles = D.ParticleSetPlan(
    jnp.arange(1), jnp.ones((1,)), ambient_dimension=3
).prepare()
charged = D.ChargedParticlePlan(jnp.ones((1,)), "cherenkov").prepare(particles)
current = PIC.ChargeConservingCurrentPlan(
    PIC.PICParticleCochainTransferPlan(bridge).prepare(charged)
)
trajectory = mx.PrescribedChargeTrajectory(times, positions)
omega = 2.0 * np.pi * beta / length_z
layout = mx.MaxwellCochainLayout(bridge)
edge_shape = bridge.orientation_shapes[1][2]
probe_x = np.arange(cells_x // 2 + 4, cells_x - 10)
probe = bridge.orientation_offsets[1][2] + np.ravel_multi_index(
    (probe_x, np.zeros_like(probe_x), np.zeros_like(probe_x)), edge_shape
)
acquisition = mx.MaxwellSpectralAcquisition(
    jnp.asarray([omega]),
    sign="positive",
    measure="sample-mean",
    start_time=float(times[-1] - 3 * period),
    stop_time=float(times[-1]),
)
runtime = phx.solver.CompatibleMaxwellPlan(
    bridge,
    sources=(mx.PrescribedChargeCurrentSourcePlan(trajectory, current),),
    observers=(
        mx.DFTObserverPlan(
            mx.FieldProbePlan("electric", jnp.asarray(probe)), acquisition
        ),
    ),
    **medium,
).prepare()
plan = mx.PrescribedChargeMaxwellPlan(runtime, current, trajectory, np.asarray([1.0]))
result = mx.solve_prescribed_charge_maxwell(plan)
evidence = result.evidence
phasor = np.asarray(result.observations[0]).reshape(-1)
phase = np.unwrap(np.angle(phasor))
wavenumber_x = np.polyfit(probe_x * spacing, phase, 1)[0]
measured = np.degrees(np.arctan2(abs(wavenumber_x), omega / (beta)))
expected = np.degrees(np.arccos(1.0 / (beta * index)))
print(f"successful={bool(evidence.successful)} status={int(evidence.status)}")
print(
    f"continuity={float(evidence.maximum_continuity_defect):.1e} "
    f"gauss={float(evidence.maximum_gauss_defect):.1e} "
    f"ledger={float(evidence.relative_ledger_defect):.1e}"
)
print(f"cone angle measured={measured:.2f} deg  continuum={expected:.2f} deg")
