#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run delayed regional oscillators with persistent Balloon–Windkessel state."""

import diffrax as dfx
import jax.numpy as jnp

from phydrax.applications import neuroscience as ns


def main() -> None:
    connectivity = ns.RegionalConnectivity(
        ("stimulated", "downstream"),
        [[0.0, 0.2], [1.0, 0.0]],
        [[0.0, 0.137], [0.211, 0.0]],
    )

    def history(time_s, args):
        del time_s, args
        return jnp.zeros((2, 2))

    def drive(time_s, neural, args):
        del args
        pulse = 0.3 * jnp.exp(-jnp.square((time_s - 0.5) / 0.2))
        return jnp.zeros_like(neural).at[0, 0].set(pulse)

    problem = ns.regional_bold_problem(
        connectivity,
        ns.Hopf(a_per_s=-0.4, frequency_hz=[0.08, 0.11], coupling_per_s=0.6),
        history,
        ns.BalloonWindkessel(),
        ns.NeuralBOLDDrive([1.0, 0.0], [0.0, 0.0], gain=0.5),
        t0=0.0,
        t1=8.0,
        drive=drive,
    )
    solution = ns.solve_regional(
        problem,
        save_times=jnp.linspace(0.0, 8.0, 65),
        solver=dfx.Heun(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.025,
    )
    if solution.bold is None:
        raise RuntimeError("Joint regional/BOLD problem did not return BOLD samples.")
    if not bool(jnp.all(solution.bold.sample_valid)):
        raise RuntimeError("Regional/BOLD solve produced invalid active samples.")
    print("peak fractional BOLD:", jnp.max(solution.bold.values, axis=0))
    print("final neural state:", solution.neural.values[-1])


if __name__ == "__main__":
    main()
