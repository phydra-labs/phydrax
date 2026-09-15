# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Advance a small periodic tensor perturbation with the fixed-grid Z4c runtime."""

from __future__ import annotations

from jax import config


config.update("jax_enable_x64", True)

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    nr = phx.applications.numerical_relativity
    shape = (9, 5, 5)
    spacing = (0.25, 1.0, 1.0)
    grid = nr.FixedGridGeometry(shape, (0.0, 0.0, 0.0), spacing, periodic=True)
    derivatives = nr.FourthOrderDerivatives(grid.shape, grid.spacing)
    system = nr.Z4cSystem(
        phx.RelativityScaleContract.si(),
        phx.metrix.RelativityConvention.canonical(),
        chart_id="cartesian-periodic",
        constraint_damping=0.02,
        constraint_tolerance=1.0,
    )

    flat = nr.flat_z4c_state(shape, grid_id=grid.grid_id)
    tensor_wave = 1.0e-4 * jnp.sin(
        2.0 * jnp.pi * grid.coordinates[0] / (shape[0] * spacing[0])
    )
    conformal_metric = (
        flat.conformal_metric.at[1, 1].add(tensor_wave).at[2, 2].add(-tensor_wave)
    )
    initial_state = nr.make_z4c_state(
        flat.chi,
        conformal_metric,
        2.0e-4 * jnp.ones(shape),
        flat.conformal_extrinsic_curvature,
        flat.theta,
        flat.conformal_connection,
        flat.lapse,
        flat.shift,
        flat.shift_driver,
        grid_id=grid.grid_id,
    )
    runtime = nr.FixedGridZ4cRuntime(
        system,
        grid,
        derivatives,
        nr.HarmonicGauge(),
        nr.PeriodicBoundary(),
        nr.Z4cAlgebraicEnforcement(),
        time_step=0.01,
        integrator="ssprk33",
    )

    state = runtime.initialize(initial_state)
    for _ in range(2):
        latest = runtime.evaluate(state)
        if not bool(latest.successful):
            raise RuntimeError(
                f"Z4c proposal failed with status bits {int(latest.status)}."
            )
        state = runtime.accept(latest)

    evolved_wave = state.state.conformal_metric[1, 1] - 1.0
    print("steps", int(state.step_index), "time", float(state.time))
    print("courant_number", runtime.courant_number)
    print("max_constraint_norm", float(latest.constraints.maximum_norm))
    print("metric_wave_amplitude", float(jnp.max(jnp.abs(evolved_wave))))
    print(
        "status_bits",
        int(latest.status),
        "qualified",
        bool(latest.qualified),
        "derivative_valid",
        bool(latest.derivative_valid),
    )


if __name__ == "__main__":
    main()
