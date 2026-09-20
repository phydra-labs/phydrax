# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Solve a QNM, computed scalar scattering, and a bounded Hawking spectrum.

The Schwarzschild QNM and every real-frequency radial solve retain their numerical
qualification. The deliberately unqualified finite-mode/frequency tail bounds keep
the illustrative Hawking quadrature from claiming a complete physical spectrum.
"""

from __future__ import annotations

from jax import config


config.update("jax_enable_x64", True)

import jax.numpy as jnp

import phydrax as phx


def qnm_plan(compact):
    mode = compact.SeparatedMode(
        -2,
        2,
        2,
        overtone=0,
        sector="regge-wheeler",
        family="qnm",
        background_id="schwarzschild-M1",
    )
    return compact.QnmSolvePlan(
        mode,
        compact.SpheroidalAngularPlan(
            mode,
            5,
            residual_tolerance=1.0e-9,
            isolation_tolerance=1.0e-9,
            minimum_target_overlap=1.0e-6,
            maximum_condition=1.0e10,
        ),
        compact.SchwarzschildRadialPlan(
            mode,
            1.0,
            node_count=65,
            outer_radius=30.0,
            residual_tolerance=1.0e-5,
            matching_tolerance=1.0e-7,
            asymptotic_tolerance=5.0e-2,
            maximum_dimension=96,
            integration_substeps=32,
            infinity_asymptotic_order=12,
        ),
        jnp.asarray(0.0),
        phx.nonlinear.NewtonKrylov(),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=1.0e-10,
            relative_residual=1.0e-10,
            maximum_steps=20,
        ),
        phx.nonlinear.SensitivityPolicy("implicit-forward", condition_limit=1.0e12),
        compact.BoundedContinuedFractionPlan(48, 96, 0, 1.0e-9, 1.0e-9),
        compact.BoundedContinuedFractionPlan(96, 192, 0, 1.0e-9, 1.0e-9),
        1.0e-8,
        1.0e12,
        (-0.5, 0.5),
        branch_id="schwarzschild-s-2-l2-n0",
    )


def main() -> None:
    compact = phx.applications.compact_objects

    mode_plan = qnm_plan(compact)
    reference = compact.schwarzschild_qnm_reference(mode_plan.mode, 2.0e-8, 2.0e-8)
    qnm = compact.solve_qnm(
        mode_plan,
        reference.angular_frequency * (1.0 + 1.0e-4),
        jnp.asarray(4.001 + 0.0j),
        qualification=reference,
        continuation_active=True,
    )
    if not bool(qnm.qualified):
        raise RuntimeError("The bounded Schwarzschild QNM solve was not qualified.")

    scale = phx.RelativityScaleContract.geometric(phx.units.SOLAR_MASS)
    kerr_input = compact.KerrInput(1.0, 0.0)
    horizon = compact.evaluate_stationary_kerr_horizon(kerr_input)
    thermal = compact.evaluate_kerr_entropy_temperature(horizon, scale)
    source_state = compact.KerrEvaporationState(
        kerr_input.mass,
        kerr_input.angular_momentum,
        elapsed_time=0.0,
        step_index=0,
        state_id=kerr_input.input_id,
    )

    scatter_mode = compact.SeparatedMode(
        0,
        0,
        0,
        sector="scalar",
        family="scattering",
        background_id="computed-schwarzschild-M1",
    )
    radial_scattering = compact.SchwarzschildRadialPlan(
        scatter_mode,
        1.0,
        node_count=65,
        inner_radius=2.00002,
        outer_radius=80.0,
        residual_tolerance=1.0e-5,
        matching_tolerance=1.0e-7,
        asymptotic_tolerance=5.0e-2,
        maximum_dimension=96,
        integration_substeps=16,
        infinity_asymptotic_order=3,
    )
    flux_plan = compact.BlackHoleScatteringPlan(
        scatter_mode,
        horizon.angular_velocity,
        1.0e-7,
        1.0e-10,
    )
    scatter_plan = compact.SchwarzschildScatteringSolvePlan(
        radial_scattering,
        flux_plan,
        phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU()),
        refined_integration_substeps=32,
        frequency_step=1.0e-3,
        decomposition_tolerance=1.0e-10,
        refinement_tolerance=1.0e-5,
        slope_refinement_tolerance=5.0e-3,
        absolute_flux_tolerance=1.0e-8,
        incident_amplitude_tolerance=1.0e-12,
        low_frequency_maximum=5.0e-2,
        low_frequency_relative_tolerance=2.5e-1,
    )
    frequencies = jnp.asarray((0.015, 0.020, 0.025))
    scattering = [
        compact.solve_schwarzschild_scattering(scatter_plan, frequency)
        for frequency in frequencies
    ]
    if not bool(jnp.all(jnp.stack(tuple(item.qualified for item in scattering)))):
        statuses = [int(item.status) for item in scattering]
        raise RuntimeError(f"Computed Schwarzschild scattering failed: {statuses}.")

    graybody = jnp.stack(tuple(item.graybody_factor for item in scattering))[None, :]
    finite = jnp.asarray([jnp.all(jnp.stack(tuple(item.finite for item in scattering)))])
    converged = jnp.asarray(
        [jnp.all(jnp.stack(tuple(item.converged for item in scattering)))]
    )
    physically_valid = jnp.asarray(
        [jnp.all(jnp.stack(tuple(item.physically_valid for item in scattering)))]
    )
    qualified = jnp.asarray(
        [jnp.all(jnp.stack(tuple(item.qualified for item in scattering)))]
    )
    derivative_valid = jnp.asarray(
        [jnp.all(jnp.stack(tuple(item.derivative_valid for item in scattering)))]
    )
    species = compact.QuantumFieldSpecies("massless-scalar", 0.0, "boson")
    spectrum_plan = compact.HawkingSpectrumPlan(
        scale,
        (species,),
        frequencies,
        (species.species_id,),
        (0,),
        (0,),
        mode_ids=(scatter_mode.mode_id,),
    )
    tails = compact.HawkingTailEvidence(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        qualified=False,
        derivative_valid=False,
        qualification_id="illustrative-tail-bounds-not-qualified",
    )
    scattering_data = compact.HawkingScatteringData(
        spectrum_plan,
        graybody,
        jnp.asarray((scattering[0].corotation_slope,)),
        finite=finite,
        converged=converged,
        physically_valid=physically_valid,
        qualified=qualified,
        derivative_valid=derivative_valid,
        tail_evidence=tails,
        source_ids=(tuple(item.source_id for item in scattering),),
        qualification_id="computed-schwarzschild-radial-scattering",
    )
    hawking = compact.evaluate_hawking_spectrum(
        spectrum_plan,
        source_state,
        scattering_data,
        thermal.temperature,
        horizon.angular_velocity,
        horizon_source_id=source_state.state_id,
        horizon_finite=horizon.finite & thermal.finite,
        horizon_converged=horizon.converged & thermal.converged,
        horizon_physically_valid=horizon.physically_valid & thermal.physically_valid,
        horizon_qualified=horizon.qualified & thermal.qualified,
        horizon_derivative_valid=horizon.derivative_valid & thermal.derivative_valid,
    )

    qnm_status = compact.QnmStatus(int(qnm.status)).name.lower()
    print("qnm_status", qnm_status, "M_omega", complex(qnm.angular_frequency))
    print("graybody_factors", [float(value) for value in graybody[0]])
    print(
        "computed_corotation_slopes",
        [float(item.corotation_slope) for item in scattering],
    )
    print(
        "max_flux_residual",
        float(
            jnp.max(jnp.abs(jnp.stack(tuple(item.flux_residual for item in scattering))))
        ),
    )
    print("hawking_number_flux", float(hawking.number_flux))
    print("hawking_energy_flux", float(hawking.energy_flux))
    print("hawking_state_bound", bool(hawking.bound_to(source_state)))
    print(
        "hawking_evidence",
        "finite",
        bool(hawking.finite),
        "physical",
        bool(hawking.physically_valid),
        "qualified",
        bool(hawking.qualified),
        "tail_qualified",
        bool(tails.qualified),
    )


if __name__ == "__main__":
    main()
