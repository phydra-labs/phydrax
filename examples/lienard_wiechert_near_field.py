#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Near-zone Liénard–Wiechert fields of a uniform and a hyperbolic worldline.

A uniformly moving charge (γ = 3) is compared with Heaviside's field written in
its present position, and a charge with constant proper acceleration
``g = c²/α`` with Born's closed form. The hyperbolic field is split into its
velocity and acceleration parts, and the resolution evidence is printed.
"""

import numpy as np

import phydrax as phx
from phydrax.electromagnetics import ChargedTrajectory, LienardWiechertFieldPlan


scale = phx.ElectromagneticScaleContract.si()
c = float(scale.speed_of_light)
charge = float(scale.elementary_charge)
coulomb = charge / (4.0 * np.pi * float(scale.vacuum_permittivity))


def lane(
    times: np.ndarray, positions: np.ndarray, proper: np.ndarray, rates: np.ndarray
) -> ChargedTrajectory:
    return ChargedTrajectory(
        times,
        positions[:, None],
        proper[:, None],
        np.array([charge]),
        np.array([1.0]),
        np.ones((times.shape[0], 1), dtype=bool),
        (np.zeros(1, dtype=np.uint32), np.zeros(1, dtype=np.uint32)),
        proper_accelerations=rates[:, None],
    )


prepared = LienardWiechertFieldPlan(
    scale,
    history="refuse",
    exclusion_radius=1.0e-6,
    interpolation="hermite-quintic",
).prepare()

# Uniform motion along x at γ = 3: Heaviside's present-position field.
gamma = 3.0
proper_speed = c * np.sqrt(gamma**2 - 1.0)
beta = proper_speed / (gamma * c)
times = np.linspace(-10.0e-9, 2.0e-9, 121)
zeros = np.zeros_like(times)
uniform = lane(
    times,
    np.stack((beta * c * times, zeros, zeros), axis=-1),
    np.stack((np.full_like(times, proper_speed), zeros, zeros), axis=-1),
    np.zeros((times.shape[0], 3)),
)
separations = np.array([[0.0, 0.1, 0.0], [-0.2, 0.05, 0.05], [0.1, 0.0, 0.3]])
events = np.concatenate((np.zeros((3, 1)), separations), axis=1)
result = prepared.evaluate(uniform, events)
distance = np.linalg.norm(separations, axis=1)
shape = 1.0 / gamma**2 + (beta * separations[:, 0] / distance) ** 2
heaviside = (
    coulomb * separations / (gamma**2 * distance[:, None] ** 3 * shape[:, None] ** 1.5)
)
error = np.max(np.abs(np.asarray(result.electric_field) - heaviside)) / np.max(
    np.abs(heaviside)
)
print("uniform motion, gamma = 3")
print(f"  max |E - E_Heaviside| / max |E|     = {error:.2e}")
print(f"  status                              = {np.asarray(result.evidence.status)}")

# Hyperbolic motion z² − c² t² = α² with α = 1 m: Born's field.
alpha = 1.0
times = np.linspace(-3.0 * alpha / c, 3.0 * alpha / c, 481)
zeros = np.zeros_like(times)
hyperbolic = lane(
    times,
    np.stack((zeros, zeros, np.sqrt(alpha**2 + (c * times) ** 2)), axis=-1),
    np.stack((zeros, zeros, c**2 * times / alpha), axis=-1),
    np.stack((zeros, zeros, np.full_like(times, c**2 / alpha)), axis=-1),
)
events = np.array(
    [
        [0.0, 0.5, 0.0, 1.2],
        [0.4 / c, 0.3, 0.4, 0.8],
        [1.1 / c, -0.6, 0.2, 2.0],
    ]
)
result = prepared.evaluate(hyperbolic, events)
ct = c * events[:, 0]
x, y, z = events[:, 1], events[:, 2], events[:, 3]
rho = np.hypot(x, y)
xi = np.sqrt((alpha**2 + ct**2 - rho**2 - z**2) ** 2 + 4.0 * alpha**2 * rho**2)
radial = coulomb * 8.0 * alpha**2 * rho * z / xi**3
born = np.stack(
    (
        radial * x / rho,
        radial * y / rho,
        -coulomb * 4.0 * alpha**2 * (alpha**2 + ct**2 + rho**2 - z**2) / xi**3,
    ),
    axis=-1,
)
error = np.max(np.abs(np.asarray(result.electric_field) - born)) / np.max(np.abs(born))
velocity = np.linalg.norm(np.asarray(result.velocity_field), axis=1)
acceleration = np.linalg.norm(np.asarray(result.acceleration_field), axis=1)
print("hyperbolic motion, alpha = 1 m")
print(f"  max |E - E_Born| / max |E|          = {error:.2e}")
print(
    "  |E_acceleration| / |E_velocity|     =",
    np.array2string(acceleration / velocity, precision=4),
)
print(
    "  retarded times [ns]                 =",
    np.array2string(1.0e9 * np.asarray(result.retarded_times)[:, 0], precision=4),
)
print(
    f"  max root residual [s]               = {float(np.max(np.asarray(result.evidence.maximum_root_residual))):.2e}"
)
print(f"  status                              = {np.asarray(result.evidence.status)}")
