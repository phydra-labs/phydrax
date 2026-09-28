#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Radiative (Sokolov–Ternov) polarization of electrons held at χ = 1 across a
uniform magnetic field, and the linear polarization of the photons they emit,
with the spin-resolved nonlinear Compton Monte Carlo."""

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.discretization import pic
from phydrax.units import CHARGE, UnitDefinition


# Code units c = ε₀ = 1, time 1/ω₀ and fields m ω₀ c/e for a 1 μm laser:
# a_S = mc²/(ħω₀) = 4.1e5 and α = 1/137.036 with q = m = ħ a_S.
schwinger = 410000
hbar = 4 * Fraction(math.pi) * Fraction(1000, 137036) / schwinger**2
mass = float(hbar) * schwinger
scale = phx.ElectromagneticScaleContract.code_units(
    pic.PIC_CODE_RELATIVITY.dimensional_scale,
    UnitDefinition("code_charge", CHARGE, "phydrax:pic-code"),
    gravitational_constant=1,
    speed_of_light=1,
    reduced_planck_constant=hbar,
    boltzmann_constant=1,
    elementary_charge=Fraction(mass),
    electron_mass=Fraction(mass),
    vacuum_permittivity=1,
    constant_set_id="polarized-qed-example",
)

tables = {
    polarization: pic.QEDTable(
        "nonlinear-compton", maximum_chi=10.0, polarization=polarization
    )
    for polarization in ("averaged", "positive", "negative")
}
compton = pic.NonlinearComptonPlan(
    "lcfa",
    scale,
    -mass,
    mass,
    tables["averaged"],
    maximum_chi=10.0,
    minimum_gamma=1.0,
    maximum_event_probability=0.2,
    maximum_subcycles=16,
    polarization="spin-and-photon-polarized",
    spin_tables=(tables["positive"], tables["negative"]),
)

count, gamma, dt, steps = 2048, 1000.0, 0.4, 80
u = jnp.zeros((count, 3)).at[:, 0].set(math.sqrt(gamma**2 - 1.0))
electric = jnp.zeros((count, 3))
magnetic = jnp.zeros((count, 3)).at[:, 2].set(schwinger / gamma)  # χ = 1
active = jnp.ones((count,), dtype=bool)
ids = jnp.arange(count, dtype=jnp.uint32)
high = jnp.zeros_like(ids)
key = jax.random.key(0)


@jax.jit
def run(depth: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Emission steps with the lepton energy restored after each step."""

    def body(
        carry: tuple[jax.Array, jax.Array], step: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
        spin, depth = carry
        result = compton.apply(
            u,
            electric,
            magnetic,
            dt,
            active,
            depth,
            compton.uniforms(jax.random.fold_in(key, step), high, ids),
            spin=spin,
        )
        if result.spin is None or result.photon_stokes is None:
            raise ValueError("The spin model reports spins and photon Stokes vectors.")
        return (result.spin, result.optical_depth), (
            jnp.mean(result.spin[:, 2]),
            jnp.sum(jnp.where(result.emitted, result.photon_stokes, 0.0)),
            jnp.sum(result.emitted),
        )

    _, history = jax.lax.scan(body, (jnp.zeros((count, 3)), depth), jnp.arange(steps))
    return history


polarization, stokes, emitted = run(compton.initial_optical_depth(key, high, ids))
axis = compton.spin_axis(u, electric, magnetic)[0]
print(f"spin quantization axis ê = v̂ × F̂ = {np.asarray(axis)} (along B)")
for step in range(9, steps, 10):
    print(f"t = {dt * (step + 1):5.1f}: mean S·ê = {float(polarization[step]):+.3f}")
print(
    "Electron spins polarize antiparallel to B; the χ → 0 limit is "
    f"−8/(5√3) = {-8.0 / (5.0 * math.sqrt(3.0)):.3f}."
)
print(
    "Mean linear Stokes parameter of emitted photons along the force: "
    f"{float(jnp.sum(stokes) / jnp.sum(emitted)):.3f}"
)
