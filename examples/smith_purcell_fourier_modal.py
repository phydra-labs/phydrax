#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Smith–Purcell radiation from a lamellar grating with the Fourier-modal solver.

A line charge at ``β = 0.328`` passes 100 nm above a fused-silica lamellar grating
(period 300 nm, depth 200 nm, fill 0.5). Its field is one zeroth-harmonic sheet with
Bloch wavevector ``ω/v``; the grating's ``n = −1`` order radiates at the angle of
the Smith–Purcell relation ``λ = d(1/β − cos θ)``. The energy per unit length of
path and of charge line is ``(2/π)·P/d`` for the outgoing port power ``P``.
Lengths are in grating periods and ``c = ε₀ = μ₀ = 1``.
"""

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


beta, gap, depth, silica = 0.328, 1.0 / 3.0, 2.0 / 3.0, 2.107 + 0j
harmonics = LatticeHarmonicPlan.parallelogramic((21,), (168,)).prepare(
    jnp.asarray(((1.0, 0.0),))
)
vacuum = fm.FrequencyMaxwellMaterial(1.0 + 0j, material_id="vacuum")
fraction = harmonics.fractional_coordinates[..., 0]
teeth = fm.FrequencyMaxwellMaterial(
    jnp.where(fraction < 0.5, silica, 1.0 + 0j), material_id="lamellar"
)
walls = fm.VectorFourierFactorizationPlan(
    fm.AnalyticInterfaceFramePlan(
        jnp.broadcast_to(jnp.asarray((0.0, 1.0)), harmonics.sample_shape + (2,)),
        frame_id="lamellar-walls",
    )
)
elements = (
    fm.FourierModalLayer(
        vacuum, 0.05, fm.DirectFourierFactorizationPlan(), layer_id="cap"
    ),
    fm.FourierModalSourcePlane("beam"),
    fm.FourierModalLayer(
        vacuum, gap, fm.DirectFourierFactorizationPlan(), layer_id="gap"
    ),
    fm.FourierModalLayer(teeth, depth, walls, layer_id="grating"),
)
source = fm.MovingLineChargeSource("beam", charge=1.0, speed=beta)
orders = np.asarray(harmonics.plan.layout.coefficients[:, 0])
minus_one = int(np.flatnonzero(orders == -1)[0])

print("  λ/d    θ(solved)  θ(Smith–Purcell)  d²W/(dω dl dy) up, down")
for wavelength in (2.9, 3.05, 3.2):
    omega = 2.0 * np.pi / wavelength
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        omega,
        source.bloch_wavevector(omega),
        fm.HomogeneousMaxwellPort(vacuum, port_id="vacuum"),
        elements,
        fm.HomogeneousMaxwellPort(
            fm.FrequencyMaxwellMaterial(silica, material_id="substrate"),
            port_id="substrate",
        ),
    )
    prepared = fm.prepare_fourier_modal_maxwell(problem)
    excitation = fm.moving_line_charge_excitation(
        problem, source, prepared.interface_scattering.block_size
    )
    result = fm.solve_fourier_modal_maxwell(prepared, excitation)
    far = fm.diffraction_order_far_field(prepared, result, side="left")
    solved = np.degrees(np.arccos(float(far.directions[minus_one, 0])))
    relation = np.degrees(np.arccos(1.0 / beta - wavelength))
    up = 2.0 / np.pi * float(result.left_outgoing_power[0])
    down = 2.0 / np.pi * float(result.right_outgoing_power[0])
    print(
        f"  {wavelength:.2f}   {solved:8.3f}   {relation:8.3f}          {up:.3e}, {down:.3e}"
    )
