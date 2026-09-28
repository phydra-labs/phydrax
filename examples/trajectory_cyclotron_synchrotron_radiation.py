#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cyclotron to synchrotron radiation of one circular orbit.

An electron on a circular orbit of angular frequency ω₀ radiates harmonics
m ω₀. At β = 0.01 the fundamental carries the Larmor power and the second
harmonic is (12/5) β² of it; at γ = 3 tens of harmonics are Schott-weighted;
at γ = 30 the harmonic comb follows the synchrotron spectrum
dP/dω = √3 q² γ F(ω/ω_c) / (8π² ε₀ R) with ω_c = (3/2) γ³ ω₀.

The orbit is sampled at half steps over exactly one period, so the spectrum
at m ω₀ is the Fourier coefficient of the periodic received field and
dP_m/dΩ = 2π S(m ω₀) / T₀².
"""

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.electromagnetics import (
    ChargedTrajectory,
    RadiationObserverPlan,
    TrajectoryRadiationPlan,
)


scale = phx.ElectromagneticScaleContract.si()
c = float(scale.speed_of_light)
epsilon0 = float(scale.vacuum_permittivity)
charge = float(scale.elementary_charge)
omega0 = 1.0e10
period = 2.0 * np.pi / omega0


def orbit(gamma: float, samples: int) -> tuple[ChargedTrajectory, float, float]:
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    times = (np.arange(samples + 2) - 0.5) * period / samples
    phase = omega0 * times
    zeros = np.zeros_like(phase)
    radius = beta * c / omega0
    positions = radius * np.stack((np.cos(phase), np.sin(phase), zeros), -1)
    proper = gamma * beta * c * np.stack((-np.sin(phase), np.cos(phase), zeros), -1)
    trajectory = ChargedTrajectory(
        times,
        positions[:, None],
        proper[:, None],
        np.array([-charge]),
        np.array([1.0]),
        np.ones((times.size, 1), dtype=bool),
        (np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint32)),
    )
    return trajectory, beta, radius


def harmonic_power(
    trajectory: ChargedTrajectory,
    harmonics: np.ndarray,
    elevation_edges: list[float],
    order: int,
) -> tuple[np.ndarray, int]:
    """Angle-integrated power per harmonic, both hemispheres by symmetry."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    elevation = np.concatenate(
        [
            0.5 * (b - a) * nodes + 0.5 * (a + b)
            for a, b in zip(elevation_edges[:-1], elevation_edges[1:])
        ]
    )
    elevation_weights = np.concatenate(
        [
            0.5 * (b - a) * weights
            for a, b in zip(elevation_edges[:-1], elevation_edges[1:])
        ]
    )
    directions = np.stack(
        (np.cos(elevation), np.zeros_like(elevation), np.sin(elevation)), axis=-1
    )
    plan = TrajectoryRadiationPlan(
        scale,
        RadiationObserverPlan(directions, np.array([0.0, 0.0, 1.0])),
        omega0 * harmonics,
        coherence="coherent",
        route="segment-exact",
        emission="truncated",
    )
    result = plan.prepare().evaluate(trajectory)
    density = 2.0 * np.pi * np.asarray(result.spectral_energy) / period**2
    solid_angle = 4.0 * np.pi * np.cos(elevation) * elevation_weights
    return density @ solid_angle, int(result.evidence.status)


def lienard_power(gamma: float, beta: float) -> float:
    return charge**2 * gamma**4 * (beta * omega0) ** 2 / (6.0 * np.pi * epsilon0 * c)


# β = 0.01: cyclotron line, Larmor power, and polarization on the axis.
trajectory, beta, _ = orbit(1.0 / np.sqrt(1.0 - 0.01**2), 512)
powers, status = harmonic_power(
    trajectory, np.array([1.0, 2.0, 3.0]), [0.0, np.pi / 2.0], 16
)
print("beta = 0.01 (cyclotron)")
print(
    f"  sum of harmonics / Lienard power     = {powers.sum() / lienard_power(1.0 / np.sqrt(1.0 - beta**2), beta):.8f}"
)
print(
    f"  P2 / P1 / (12/5 beta^2)              = {powers[1] / powers[0] / (2.4 * beta**2):.5f}"
)
axis = (
    TrajectoryRadiationPlan(
        scale,
        RadiationObserverPlan(np.array([[0.0, 0.0, 1.0]]), np.array([1.0, 0.0, 0.0])),
        np.array([omega0]),
        coherence="coherent",
        route="segment-exact",
        emission="truncated",
    )
    .prepare()
    .evaluate(trajectory)
)
stokes = np.asarray(axis.stokes)[0, 0]
print(f"  on-axis circular polarization V / I  = {stokes[3] / stokes[0]:+.6f}")
print(f"  status                               = {status}")

# γ = 3: Schott harmonics and the Liénard total.
gamma = 3.0
trajectory, beta, _ = orbit(gamma, 2048)
harmonics = np.arange(1.0, 301.0)
powers, status = harmonic_power(
    trajectory, harmonics, [0.0, 0.1, 0.3, 0.7, np.pi / 2.0], 12
)
theta = np.pi / 2.0
m = harmonics[:5]
bessel_argument = jnp.asarray(m * beta)
derivative = 0.5 * np.asarray(
    phx.special.jv(jnp.asarray(m - 1.0), bessel_argument)
    - phx.special.jv(jnp.asarray(m + 1.0), bessel_argument)
)
in_plane_schott = (
    charge**2
    * omega0**2
    * m**2
    * beta**2
    * derivative**2
    / (8.0 * np.pi**2 * epsilon0 * c)
)
axis = (
    TrajectoryRadiationPlan(
        scale,
        RadiationObserverPlan(np.array([[1.0, 0.0, 0.0]]), np.array([0.0, 0.0, 1.0])),
        omega0 * m,
        coherence="coherent",
        route="segment-exact",
        emission="truncated",
    )
    .prepare()
    .evaluate(trajectory)
)
in_plane = 2.0 * np.pi * np.asarray(axis.spectral_energy)[:, 0] / period**2
print("gamma = 3")
print(
    f"  sum of harmonics 1..300 / Lienard    = {powers.sum() / lienard_power(gamma, beta):.6f}"
)
print(
    "  in-plane dP_m/dOmega / Schott (m=1..5) =",
    np.array2string(in_plane / in_plane_schott, precision=6),
)
print(f"  status                               = {status}")

# γ = 30: the harmonic comb follows the synchrotron function.
gamma = 30.0
trajectory, beta, radius = orbit(gamma, 96000)
critical = 1.5 * gamma**3
harmonics = np.round(critical * np.array([0.03, 0.1, 0.3, 1.0]))
powers, status = harmonic_power(
    trajectory, harmonics, [0.0, 0.01, 0.03, 0.06, 0.12, 0.3], 8
)
x = harmonics / critical
continuum = (
    np.sqrt(3.0)
    * charge**2
    * gamma
    * np.asarray(phx.special.synchrotron_f(jnp.asarray(x)))
    / (8.0 * np.pi**2 * epsilon0 * radius)
    * omega0
)
print("gamma = 30 (synchrotron)")
print("  x = omega / omega_c                  =", np.array2string(x, precision=4))
print(
    "  P_m / (dP/domega * omega0)           =",
    np.array2string(powers / continuum, precision=4),
)
print(f"  status                               = {status}")
