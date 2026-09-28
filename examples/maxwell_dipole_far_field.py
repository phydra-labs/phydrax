#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hertzian-dipole far field from a compatible Maxwell run and a Huygens box.

A pulsed z-directed edge current radiates in vacuum; a closed Huygens box samples the
transient tangential spectra ``∫ f(t) e^{+iωt} dt`` (``exp(-iωt)`` phasors), and the
far-field transform is compared with the analytic dipole
``F_θ = −iω m sin θ e^{−ik r̂·r₀}/(4π)`` and radiated energy ``ω²|m|²/(6π²)``.
"""

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


mx = phx.solver.maxwell
count, length = 36, 2.0
omega0, tau = 4.0 * np.pi, 0.15
t0, stop = 3.0 * tau, 1.6
omegas = np.asarray([0.8 * omega0, omega0, 1.2 * omega0])


def envelope(time: jax.Array, args: object) -> jax.Array:
    del args
    return jnp.exp(-(((time - t0) / tau) ** 2)) * jnp.sin(omega0 * (time - t0))


grid = phx.discretization.TensorGridPlan(
    tuple(phx.discretization.UniformCellAxisSpec(count) for _ in range(3)),
    axis_names=("x", "y", "z"),
).prepare(jnp.asarray([[0.0] * 3, [length] * 3]))
bridge = phx.discretization.StructuredCochainBridge(grid)
spacing = length / count
center = count // 2
edge = bridge.orientation_offsets[1][2] + int(
    np.ravel_multi_index((center, center, center), bridge.orientation_shapes[1][2])
)
position = np.asarray([center, center, center + 0.5]) * spacing

acquisition = mx.MaxwellSpectralAcquisition(
    jnp.asarray(omegas), sign="positive", measure="time-integral", stop_time=stop
)
exterior = mx.HomogeneousMaxwellExterior()
box = mx.MaxwellHuygensBoxPlan(
    bridge, (center - 5,) * 3, (center + 5,) * 3, acquisition, exterior
)
runtime = phx.solver.CompatibleMaxwellPlan(
    bridge,
    observers=(box,),
    sources=(
        mx.MaxwellElectricCurrentSourcePlan(
            jnp.asarray([edge]), jnp.asarray([1.0]), envelope=envelope
        ),
    ),
).prepare()
dt = 0.9 * float(runtime.stable_dt)
steps = int(np.ceil(stop / dt))
result = mx.solve_compatible_maxwell(runtime, runtime.initialize(), 0.0, dt, steps)
sampler = runtime.observers[0]
if not isinstance(sampler, mx.PreparedMaxwellHuygensBox):
    raise TypeError("The prepared runtime must carry the Huygens box sampler.")
phasors = sampler.surface_phasors(result.final_state.observations[0])

nodes, weights = np.polynomial.legendre.leggauss(16)
phi = 2.0 * np.pi * (np.arange(32) + 0.5) / 32
cos_theta, phi_grid = np.meshgrid(nodes, phi, indexing="ij")
sin_theta = np.sqrt(1.0 - cos_theta**2)
directions = np.stack(
    (sin_theta * np.cos(phi_grid), sin_theta * np.sin(phi_grid), cos_theta), axis=-1
).reshape(-1, 3)
quadrature = (weights[:, None] * np.full((1, 32), 2.0 * np.pi / 32)).reshape(-1)
far = mx.MaxwellFarFieldPlan(directions, jnp.asarray([0.0, 0.0, 1.0]), exterior).evaluate(
    phasors
)

# Current cochain j = J·ℓ, so the dipole moment I·ℓ is j·h² on a uniform grid.
moment = (
    np.exp(1j * omegas * t0)
    * (tau * np.sqrt(np.pi) / 2j)
    * (
        np.exp(-(tau**2) * (omegas + omega0) ** 2 / 4.0)
        - np.exp(-(tau**2) * (omegas - omega0) ** 2 / 4.0)
    )
    * spacing**2
)
exact_theta = (
    -1j
    * omegas[:, None]
    * moment[:, None]
    * sin_theta.reshape(1, -1)
    * np.exp(-1j * omegas[:, None] * (directions @ position)[None])
    / (4.0 * np.pi)
)
field_error = np.linalg.norm(
    np.asarray(far.field_spectrum[..., 0]) - exact_theta, axis=1
) / np.linalg.norm(exact_theta, axis=1)
radiated = np.asarray(far.spectral_energy) @ quadrature
surface = np.asarray(mx.spectral_poynting_energy(phasors))
exact = omegas**2 * np.abs(moment) ** 2 / (6.0 * np.pi**2)
for index, omega in enumerate(omegas):
    print(
        f"omega={omega:6.3f}  |F_theta| rel. error={field_error[index]:.3e}  "
        f"far-field energy/exact={radiated[index] / exact[index]:.4f}  "
        f"surface Poynting/exact={surface[index] / exact[index]:.4f}"
    )
