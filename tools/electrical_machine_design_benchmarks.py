#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run ``JAX_ENABLE_X64=1 python tools/electrical_machine_design_benchmarks.py``.

Actual rotating Az FEM solves, energy/air-stress torque checks, a circular
magnet reference, and native bounded design. No analytic replacement residuals.
The full angle grid uses synchronous impressed currents; the mechanical torque
partial derivative holds each sample's current fixed.
"""

import argparse
import json
import platform

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import measure_repeated, measure_synchronized
from phydrax.applications.electrical_machines import (
    optimize_machine_design,
    polar_machine,
    polar_machine_study,
    scan_machine_angles,
    solve_planar_machine,
)
from phydrax.optim import Bounds, OptimizationTermination


def _evidence(scan):
    contour_defects = [
        abs(float(field.contour_torque - field.torque))
        for field in scan.fields
        if field.contour_torque is not None
    ]
    return {
        "mean_torque_nm": float(scan.average_torque),
        "rms_torque_ripple_nm": float(scan.rms_torque_ripple),
        "peak_to_peak_torque_ripple_nm": float(scan.peak_to_peak_torque_ripple),
        "maximum_equation_relative_residual": max(
            float(field.relative_residual) for field in scan.fields
        ),
        "maximum_energy_air_stress_defect_nm": max(
            float(field.torque_discrepancy) for field in scan.fields
        ),
        "maximum_contour_torque_defect_nm": max(contour_defects, default=0.0),
        "field_and_torque_accepted": bool(scan.accepted),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sectors", type=int, nargs="+", default=(16, 32))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--optimization-steps", type=int, default=24)
    options = parser.parse_args()
    if options.samples < 1 or options.repeats < 1 or options.optimization_steps < 1:
        parser.error("samples, repeats, and optimization steps must be positive")
    if not jax.config.jax_enable_x64:
        parser.error("this certification benchmark requires JAX_ENABLE_X64=1")
    rows = []
    angles = np.linspace(0, 2 * np.pi, options.samples, endpoint=False) + 0.13
    currents = -8.0 * np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    for sectors in options.sectors:
        layers = max(1, sectors // 16)
        study, prepare_seconds = measure_synchronized(
            lambda: polar_machine_study(
                angles,
                currents,
                sectors=sectors,
                rotor_layers=layers,
                airgap_layers=max(2, layers),
                winding_layers=layers,
                stator_layers=layers,
                salient=True,
            )
        )
        evaluate = eqx.filter_jit(lambda design: scan_machine_angles(study, design))
        initial = jnp.asarray((0.028, 0.9, 0.9))
        scan, first_seconds = measure_synchronized(lambda: evaluate(initial))
        _, warm = measure_repeated(
            lambda: evaluate(initial),
            warmup=1,
            repeats=options.repeats,
        )
        machine = study.machines[0]
        fixed_topology = (
            len({model.discretization.mesh.topology_id for model in study.machines}) == 1
        )
        reluctance_design = initial.at[1].set(0.0)
        reluctance_positive = solve_planar_machine(
            machine, (-8.0, 0.0), design=reluctance_design
        )
        reluctance_negative = solve_planar_machine(
            machine, (8.0, 0.0), design=reluctance_design
        )
        reluctance_scale = max(
            abs(float(reluctance_positive.torque)),
            abs(float(reluctance_negative.torque)),
            np.finfo(float).tiny,
        )
        reluctance_even_defect = (
            abs(float(reluctance_positive.torque - reluctance_negative.torque))
            / reluctance_scale
        )
        step = 2e-6
        plus, minus = (
            solve_planar_machine(
                machine,
                currents[0],
                design=initial,
                angle_delta=sign * step,
            )
            for sign in (1, -1)
        )
        resolved_energy_torque = (plus.coenergy - minus.coenergy) / (2 * step)
        energy_fd_defect = float(abs(resolved_energy_torque - scan.fields[0].torque))
        rows.append(
            {
                "kind": "rotating-linear-magnetic-h1-p1",
                "sectors": sectors,
                "angle_samples": options.samples,
                "nodes_per_angle": int(machine.discretization.mesh.coordinates.shape[0]),
                "cells_per_angle": int(machine.cell_regions.size),
                "prepare_seconds": prepare_seconds,
                "first_scan_seconds": first_seconds,
                "warm_scan_seconds": warm.median_seconds,
                "resolved_energy_finite_difference_defect_nm": energy_fd_defect,
                "fixed_topology_across_angles": fixed_topology,
                "reluctance_current_even_relative_defect": reluctance_even_defect,
                **_evidence(scan),
            }
        )

    # A separate all-mu0 model has a closed-form circular magnet/current
    # interaction torque. The actual evaluated response remains the FEM solve.
    reference_model = polar_machine(
        0.21,
        sectors=32,
        rotor_layers=3,
        airgap_layers=4,
        winding_layers=4,
        stator_layers=4,
        rotor_relative_permeability=1.0,
        stator_relative_permeability=1.0,
    )
    plus = solve_planar_machine(reference_model, (-4.0, 0.0))
    minus = solve_planar_machine(reference_model, (4.0, 0.0))
    odd = 0.5 * (plus.torque - minus.torque)
    contour_odd = 0.5 * (plus.contour_torque - minus.contour_torque)
    a, b, outer = 0.04, 0.05, 0.065
    field_per_mu = (
        50.0 / (b * b - a * a) * ((b - a) - (b**3 - a**3) / (3 * outer * outer))
    )
    exact = 0.1 * np.pi * 0.03**2 * 0.8 * 4.0 * field_per_mu * np.cos(0.21)
    reference_error = float(abs(odd / exact - 1))
    reference_accepted = bool(plus.accepted & minus.accepted)
    rows.append(
        {
            "kind": "permanent-magnet-circular-reference",
            "fem_current_odd_torque_nm": float(odd),
            "contour_current_odd_torque_nm": float(contour_odd),
            "circular_reference_torque_nm": exact,
            "relative_discretization_error": reference_error,
            "accepted": reference_accepted and reference_error < 0.035,
        }
    )

    design_study = polar_machine_study(
        angles,
        currents,
        sectors=16,
        rotor_layers=1,
        airgap_layers=2,
        winding_layers=1,
        stator_layers=1,
        rotor_relative_permeability=1.0,
        stator_relative_permeability=1.0,
    )
    bounds = Bounds(
        jnp.asarray((0.026, 0.6, 0.6)),
        jnp.asarray((0.032, 1.2, 1.2)),
    )
    initial = jnp.asarray((0.028, 0.8, 0.8))
    baseline = scan_machine_angles(design_study, initial)
    optimized, optimize_seconds = measure_synchronized(
        lambda: optimize_machine_design(
            design_study,
            initial,
            bounds,
            termination=OptimizationTermination(
                maximum_steps=options.optimization_steps,
                absolute_optimality=1e-5,
            ),
        )
    )
    final = optimized.final_evaluation
    improved = float(final.average_torque) > float(baseline.average_torque)
    rows.append(
        {
            "kind": "native-bounded-machine-design",
            "initial_design_radius_remanence_turns": np.asarray(initial).tolist(),
            "final_design_radius_remanence_turns": np.asarray(optimized.design).tolist(),
            "initial_mean_torque_nm": float(baseline.average_torque),
            "optimization_seconds": optimize_seconds,
            "native_optimizer_status": int(optimized.optimization.status),
            "native_optimizer_success": bool(optimized.optimization.successful),
            "radius_and_parameter_bounds_satisfied": bool(
                bounds.contains(optimized.design)
            ),
            "physical_torque_improved": improved,
            "accepted": bool(optimized.accepted),
            **_evidence(final),
        }
    )
    rotating_rows = [
        row for row in rows if row["kind"] == "rotating-linear-magnetic-h1-p1"
    ]
    passed = (
        all(row["field_and_torque_accepted"] for row in rotating_rows)
        and all(row["fixed_topology_across_angles"] for row in rotating_rows)
        and all(
            row["resolved_energy_finite_difference_defect_nm"] < 1e-5
            for row in rotating_rows
        )
        and all(
            row["reluctance_current_even_relative_defect"] < 1e-8 for row in rotating_rows
        )
        and reference_accepted
        and reference_error < 0.035
        and bool(optimized.accepted)
        and improved
    )
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "jax_backend": jax.default_backend(),
                "jax_enable_x64": bool(jax.config.jax_enable_x64),
                "fidelity": (
                    "2D prescribed-angle linear isotropic magnetostatics; no "
                    "nonlinear B-H, motion-induced voltage, electrical or "
                    "magnetic losses, thermal coupling, or end effects"
                ),
                "passed": passed,
                "results": rows,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
