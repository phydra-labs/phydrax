#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Acoustically driven air microbubble in water: Keller–Miksis with thermal gas."""

from __future__ import annotations

from typing import Any

import numpy as np

import phydrax.bubble_dynamics as bd


def run() -> dict[str, Any]:
    radius = 5.0e-6
    environment = bd.BubbleEnvironment(101325.0, 293.15, vapor_pressure=2.33e3)
    model = bd.RadialBubbleModel(
        "keller_miksis",
        bd.BoundaryLayerThermalBubbleGasLaw(1.4, 0.0262, core_radius_ratio=1.0 / 8.86),
        bd.NewtonianBubbleLiquidLaw(1.0e-3),
        bd.CleanBubbleInterfaceLaw(0.072),
        environment,
        liquid_density=998.0,
        liquid_sound_speed=1481.0,
    )
    linear = bd.linear_bubble_response(model, radius, 2.0 * np.pi * np.array([2.0e5, 6.0e5]))
    frequency = 2.0e5
    plan = bd.SingleBubblePlan(
        model,
        bd.PulsedPressureDrive(1.2e5, 2.0 * np.pi * frequency, 4.0),
        np.linspace(0.0, 4.0 / frequency, 401),
        validity=bd.BubbleValidityPolicy(molecular_diameter=3.7e-10),
    )
    result = bd.solve_single_bubble(plan.prepare(radius))
    if not bool(result.successful) or not bool(linear.successful):
        raise RuntimeError(f"Acoustic bubble solve failed with status {int(result.status)}.")
    evidence = result.evidence
    validity = evidence.validity
    return {
        "status": bd.BubbleDynamicsStatus(int(result.status)).name,
        "max_radius_ratio": float(validity.max_radius_ratio),
        "min_radius_ratio": float(validity.min_radius_ratio),
        "max_gas_temperature_k": float(validity.max_gas_temperature),
        "max_wall_mach": float(validity.max_wall_mach),
        "within_support": bool(validity.within_support),
        "accepted_steps": int(evidence.accepted_steps),
        "radiated_energy_j": float(-evidence.work_residual),
        "gas_heat_j": float(evidence.gas_heat),
        "resonance_hz": float(linear.resonance_frequency) / (2.0 * np.pi),
        "effective_polytropic_index": [float(value) for value in linear.effective_polytropic_index],
        "thermal_damping_per_s": [float(value) for value in linear.thermal_damping],
    }


if __name__ == "__main__":
    print(run())
