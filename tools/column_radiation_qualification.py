#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measure grey-column analytic limits, W/m² budgets, and derivative epsilon sweeps.

PYTHONPATH=. python tools/column_radiation_qualification.py --layers 12
This is numerical evidence for declared synthetic grey coefficients, not a
comparison to atmospheric observations or a spectral reference radiation model.
"""

import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.atmosphere._radiation import (
    ColumnOpticalProperties,
    ColumnRadiationPlan,
)


SIGMA = 5.670374419e-8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=12)
    args = parser.parse_args()
    if args.layers < 1:
        parser.error("layers must be positive")
    jax.config.update("jax_enable_x64", True)
    optics = ColumnOpticalProperties(
        shortwave_absorption=(2e-6, 0.003, 0.15, 0.1),
        shortwave_scattering=(1e-5, 0.0, 50.0, 30.0),
        shortwave_asymmetry=(0.0, 0.0, 0.8, 0.7),
        longwave_absorption=(5e-5, 0.07, 20.0, 15.0),
        reference_id="declared-synthetic-grey-qualification-coefficients-not-measured",
    )
    plan = ColumnRadiationPlan(optics, surface_albedo=0.19, surface_emissivity=0.94)
    z = jnp.linspace(0.0, 1.0, args.layers)
    mass = jnp.full(args.layers, 10000.0 / args.layers)
    vapor = (5.0 + 25.0 * z) / args.layers
    liquid = 0.025 * jnp.exp(-(((z - 0.6) / 0.2) ** 2)) / args.layers
    ice = 0.015 * jnp.exp(-(((z - 0.3) / 0.2) ** 2)) / args.layers
    temperature = 220.0 + 65.0 * z

    def evaluate(controls):
        calibrated = eqx.tree_at(
            lambda p: (p.longwave_absorption_scale, p.shortwave_scattering_scale),
            plan,
            (jnp.full(4, controls[3]), jnp.full(4, controls[4])),
        )
        return calibrated.evaluate(
            temperature,
            mass,
            vapor * controls[0],
            liquid * controls[1],
            ice,
            controls[2],
            controls[5],
        )

    def observable(controls):
        result = evaluate(controls)
        return jnp.stack(
            (
                result.shortwave_upward_flux[0],
                result.longwave_upward_flux[0],
                result.surface_heating,
                jnp.sum(result.heating),
            )
        )

    controls = jnp.array([1.0, 1.0, 295.0, 1.0, 1.0, 350.0])
    observed = eqx.filter_jit(observable)
    derivative = np.asarray(jax.jacfwd(observed)(controls))
    baseline = evaluate(controls)
    if not bool(baseline.successful):
        raise RuntimeError("Baseline column was rejected.")
    epsilon_rows = []
    control_scale = jnp.array([1.0, 1.0, 295.0, 1.0, 1.0, 350.0])
    budgets = [abs(float(baseline.budget_residual))]
    for epsilon in np.logspace(-2, -7, 6):
        columns = []
        for i in range(controls.size):
            step = float(epsilon * control_scale[i])
            plus, minus = controls.at[i].add(step), controls.at[i].add(-step)
            plus_result, minus_result = evaluate(plus), evaluate(minus)
            if not bool(plus_result.successful & minus_result.successful):
                raise RuntimeError(
                    "A derivative perturbation left the declared input domain."
                )
            budgets.extend(
                (
                    abs(float(plus_result.budget_residual)),
                    abs(float(minus_result.budget_residual)),
                )
            )
            columns.append(np.asarray((observed(plus) - observed(minus)) / (2.0 * step)))
        finite_difference = np.column_stack(columns)
        error = np.abs(finite_difference - derivative)
        epsilon_rows.append(
            {
                "relative_control_epsilon": float(epsilon),
                "finite_difference_W_m2_per_control_unit": finite_difference.tolist(),
                "absolute_error_W_m2_per_control_unit": error.tolist(),
                "max_scaled_derivative_error": float(
                    np.max(error / np.maximum(1.0, np.abs(derivative)))
                ),
            }
        )

    # Independent scalar slab solve through its 2x2 fundamental ODE solution.
    absorption, scattering, albedo = 0.23, 0.71, 0.27
    analytic_optics = ColumnOpticalProperties(
        shortwave_absorption=(absorption, 0, 0, 0),
        shortwave_scattering=(scattering, 0, 0, 0),
        longwave_absorption=(0.42, 0, 0, 0),
        reference_id="analytic-homogeneous-slab",
    )
    slab = ColumnRadiationPlan(
        analytic_optics, surface_albedo=albedo, surface_emissivity=1.0
    )
    one = slab.evaluate([260.0], [1.0], [0.0], [0.0], [0.0], 300.0, 400.0)
    a, b = 2 * absorption + scattering, scattering
    k = np.sqrt(a * a - b * b)
    transfer = np.cosh(k) * np.eye(2) + np.sinh(k) / k * np.array([[-a, b], [-b, a]])
    u0 = (
        400
        * (albedo * transfer[0, 0] - transfer[1, 0])
        / (transfer[1, 1] - albedo * transfer[0, 1])
    )
    dn, un = transfer @ np.array([400.0, u0])
    analytic_error = max(
        float(np.max(np.abs(np.asarray(one.shortwave_upward_flux) - [u0, un]))),
        float(np.max(np.abs(np.asarray(one.shortwave_downward_flux) - [400.0, dn]))),
    )
    composition = {}
    for count in (2, 8, 32):
        piece = jnp.full(count, 1.0 / count)
        split = slab.evaluate(
            jnp.full(count, 260.0),
            piece,
            jnp.zeros(count),
            jnp.zeros(count),
            jnp.zeros(count),
            300.0,
            400.0,
        )
        if not bool(split.successful):
            raise RuntimeError("Homogeneous layer composition was rejected.")
        composition[count] = max(
            float(
                jnp.max(
                    jnp.abs(split.upward_flux[jnp.array([0, count])] - one.upward_flux)
                )
            ),
            float(
                jnp.max(
                    jnp.abs(
                        split.downward_flux[jnp.array([0, count])] - one.downward_flux
                    )
                )
            ),
            float(jnp.abs(jnp.sum(split.heating) - one.heating[0])),
        )
        budgets.append(abs(float(split.budget_residual)))

    transparent = eqx.tree_at(
        lambda p: (
            p.shortwave_absorption_scale,
            p.shortwave_scattering_scale,
            p.longwave_absorption_scale,
        ),
        slab,
        (jnp.zeros(4), jnp.zeros(4), jnp.zeros(4)),
    ).evaluate([260.0], [1.0], [0.0], [0.0], [0.0], 300.0, 400.0)
    transparent_error = max(
        float(
            jnp.max(jnp.abs(transparent.upward_flux - (albedo * 400 + SIGMA * 300.0**4)))
        ),
        float(jnp.max(jnp.abs(transparent.downward_flux - 400.0))),
        float(jnp.max(jnp.abs(transparent.heating))),
    )
    thick = eqx.tree_at(
        lambda p: p.longwave_absorption_scale, slab, jnp.full(4, 1e6)
    ).evaluate([260.0], [1.0], [0.0], [0.0], [0.0], 300.0, 0.0)
    blackbody_error = float(jnp.abs(thick.longwave_upward_flux[0] - SIGMA * 260.0**4))
    if not bool(one.successful & transparent.successful & thick.successful):
        raise RuntimeError("An analytic limit was rejected.")
    if (
        max(analytic_error, transparent_error, blackbody_error, *composition.values())
        > 1e-9
    ):
        raise RuntimeError("Analytic/composition flux discrepancy exceeds 1e-9 W/m².")
    if max(budgets) > 1e-9:
        raise RuntimeError("Unnormalized budget residual exceeds 1e-9 W/m².")
    if min(row["max_scaled_derivative_error"] for row in epsilon_rows) > 2e-6:
        raise RuntimeError(
            "The derivative epsilon sweep did not converge to the native derivative."
        )
    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "layers": args.layers,
                "optical_reference_id": optics.reference_id,
                "scope": "diffuse hemispheric SW scattering; grey LTE LW; no spectral or observational accuracy claim",
                "control_names": [
                    "vapor_mass_multiplier",
                    "liquid_mass_multiplier",
                    "surface_temperature_K",
                    "longwave_absorption_scale",
                    "shortwave_scattering_scale",
                    "solar_down_W_m2",
                ],
                "observable_names": [
                    "TOA_SW_up_W_m2",
                    "TOA_LW_up_W_m2",
                    "surface_heating_W_m2",
                    "atmosphere_heating_W_m2",
                ],
                "baseline_observables_W_m2": np.asarray(observed(controls)).tolist(),
                "native_derivative_W_m2_per_control_unit": derivative.tolist(),
                "derivative_epsilon_sweep": epsilon_rows,
                "scattering_slab_max_flux_error_W_m2": analytic_error,
                "transparent_max_flux_error_W_m2": transparent_error,
                "opaque_blackbody_flux_error_W_m2": blackbody_error,
                "homogeneous_composition_flux_error_W_m2": composition,
                "maximum_budget_residual_W_m2": max(budgets),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
