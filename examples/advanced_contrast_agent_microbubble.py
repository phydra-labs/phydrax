#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Lipid-coated contrast-agent microbubble: forward response and shell inference.

The forward part contrasts a clean bubble with a Marmottant shell that starts
at its buckling radius, the mechanism behind compression-only oscillations.
The inverse part recovers the shell elasticity and viscosity of the smooth
Marmottant–Gompertz law from a synthetic radius–time curve by bounded L-BFGS on
the reverse-mode gradient of the full Keller–Miksis solve.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.bubble_dynamics as bd
from phydrax.optim import Bounds, minimize, OptimizationTermination, ProjectedLBFGS


RADIUS = 2.0e-6
FREQUENCY = 2.5e6
ENVIRONMENT = bd.BubbleEnvironment(101325.0, 293.15)


def _model(interface: bd.AbstractBubbleInterfaceLaw) -> bd.RadialBubbleModel:
    return bd.RadialBubbleModel(
        "keller_miksis",
        bd.PolytropicBubbleGasLaw(1.07),
        bd.NewtonianBubbleLiquidLaw(1.0e-3),
        interface,
        ENVIRONMENT,
        liquid_density=1000.0,
        liquid_sound_speed=1500.0,
    )


def _expansion_to_compression(result: bd.SingleBubbleResult) -> float:
    radius = np.asarray(result.trajectory.radius)
    return float((radius.max() - RADIUS) / (RADIUS - radius.min()))


def run() -> dict[str, Any]:
    drive = bd.PulsedPressureDrive(8.0e4, 2.0 * np.pi * FREQUENCY, 6.0)
    times = np.linspace(0.0, 8.0 / FREQUENCY, 321)
    clean = bd.solve_single_bubble(
        bd.SingleBubblePlan(_model(bd.CleanBubbleInterfaceLaw(0.072)), drive, times).prepare(RADIUS)
    )
    buckled_shell = bd.MarmottantShell(1.0, 0.0, 0.072, 1.0e-8)
    coated = bd.solve_single_bubble(
        bd.SingleBubblePlan(_model(buckled_shell), drive, times).prepare(RADIUS)
    )
    if not (bool(clean.successful) and bool(coated.successful)):
        raise RuntimeError("Forward contrast-agent solves failed.")
    tape = coated.evidence.regime_tape
    linear = bd.linear_bubble_response(
        _model(bd.MarmottantShell(0.5, 0.02, 0.072, 1.0e-8)), RADIUS, 2.0 * np.pi * FREQUENCY
    )

    truth = (0.5, 1.0e-8)
    fit_times = np.linspace(0.0, 3.0 / FREQUENCY, 61)
    fit_drive = bd.HarmonicPressureDrive(3.0e4, 2.0 * np.pi * FREQUENCY)
    template = bd.SingleBubblePlan(
        _model(bd.GompertzMarmottantShell(truth[0], 0.02, 0.072, truth[1])),
        fit_drive,
        fit_times,
        relative_tolerance=1.0e-9,
        absolute_tolerance=1.0e-11,
    )
    observed = bd.solve_single_bubble(template.prepare(RADIUS)).trajectory.radius

    def objective(normalized: Array, args: object) -> Array:
        del args
        plan = eqx.tree_at(
            lambda current: (
                current.model.interface.shell_elasticity,
                current.model.interface.shell_viscosity,
            ),
            template,
            (normalized[0] * truth[0], normalized[1] * truth[1]),
        )
        predicted = bd.solve_single_bubble(plan.prepare(RADIUS)).trajectory.radius
        residual = (predicted - observed) / (1.0e-2 * RADIUS)
        return 0.5 * jnp.mean(residual**2)

    fit = minimize(
        objective,
        jnp.asarray([1.6, 0.5]),
        method=ProjectedLBFGS(),
        bounds=Bounds(jnp.asarray([0.2, 0.1]), jnp.asarray([5.0, 5.0])),
        termination=OptimizationTermination(
            absolute_optimality=1.0e-12, relative_optimality=1.0e-10, maximum_steps=100
        ),
    )
    recovered = np.asarray(fit.parameters) * np.asarray(truth)
    if not bool(fit.successful):
        raise RuntimeError(f"Shell inference failed with status {int(fit.status)}.")
    return {
        "clean_expansion_to_compression": _expansion_to_compression(clean),
        "buckled_shell_expansion_to_compression": _expansion_to_compression(coated),
        "regime_transitions": int(tape.count),
        "coated_resonance_mhz": float(linear.resonance_frequency) / (2.0e6 * np.pi),
        "shell_damping_per_s": float(linear.shell_damping[0]),
        "recovered_shell_elasticity_n_per_m": float(recovered[0]),
        "recovered_shell_viscosity_kg_per_s": float(recovered[1]),
        "relative_error": [float(value) for value in recovered / np.asarray(truth) - 1.0],
        "fit_objective": float(fit.objective),
    }


if __name__ == "__main__":
    print(run())
