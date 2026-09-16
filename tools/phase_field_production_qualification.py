#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _rectangle_mesh(
    x_cells: int,
    y_cells: int,
    /,
    *,
    x_bounds: tuple[float, float] = (0.0, 1.0),
    y_bounds: tuple[float, float] = (0.0, 1.0),
):
    nx = int(x_cells)
    ny = int(y_cells)
    if nx < 1 or ny < 1:
        raise ValueError("Structured triangle counts must be positive.")
    xs = np.linspace(x_bounds[0], x_bounds[1], nx + 1)
    ys = np.linspace(y_bounds[0], y_bounds[1], ny + 1)
    vertices = np.asarray([(x, y) for y in ys for x in xs], dtype=float)
    cells = []
    for row in range(ny):
        for column in range(nx):
            lower_left = row * (nx + 1) + column
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            cells.extend(
                (
                    (lower_left, lower_right, upper_left),
                    (lower_right, upper_right, upper_left),
                )
            )
    return phx.discretization.CellMesh.from_triangles(
        jnp.asarray(vertices, dtype=jnp.float64),
        jnp.asarray(cells, dtype=jnp.int32),
    )


def _model(gradient_coefficient: float):
    return phx.applications.phase_field.BinaryPhaseFieldModel(
        phx.equations.BinaryThermodynamicParameters(
            jnp.asarray(1.0, dtype=jnp.float64),
            jnp.asarray(gradient_coefficient, dtype=jnp.float64),
        )
    )


def _allen_cahn(
    mesh,
    model,
    /,
    *,
    execution_policy=None,
    termination=None,
):
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec("eta", element),
    ).prepare()
    return phx.applications.phase_field.AllenCahnFEMPlan(
        model,
        jnp.asarray(1.0, dtype=jnp.float64),
        execution_policy=execution_policy,
        termination=termination,
    ).prepare(discretization, "eta")


def _cahn_hilliard(
    mesh,
    model,
    /,
    *,
    execution_policy=None,
    termination=None,
):
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        (
            phx.discretization.FiniteElementFieldSpec("c", element),
            phx.discretization.FiniteElementFieldSpec("mu", element),
        ),
    ).prepare()
    return phx.applications.phase_field.CahnHilliardFEMPlan(
        model,
        jnp.asarray(1.0, dtype=jnp.float64),
        execution_policy=execution_policy,
        termination=termination,
    ).prepare(discretization, "c", "mu")


def _advance_allen_cahn(method, initial, step_size: float, end_time: float, /):
    count = int(round(end_time / step_size))
    state = initial
    successful = True
    maximum_energy_excess = 0.0
    stepper = eqx.filter_jit(method.step_detailed)
    for index in range(count):
        result = stepper(
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(index * step_size, dtype=jnp.float64),
            state,
            jnp.asarray(step_size, dtype=jnp.float64),
            None,
        )
        successful = successful and bool(result.successful)
        maximum_energy_excess = max(
            maximum_energy_excess,
            float(
                np.asarray(
                    result.evidence.energy_balance_defect
                    - result.evidence.energy_tolerance
                )
            ),
        )
        state = result.accepted_state
    return state, successful, maximum_energy_excess


def _advance_cahn_hilliard(method, initial, step_size: float, end_time: float, /):
    count = int(round(end_time / step_size))
    state = initial
    successful = True
    maximum_energy_excess = 0.0
    maximum_mass_defect = 0.0
    stepper = eqx.filter_jit(method.step_detailed)
    for index in range(count):
        result = stepper(
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(index * step_size, dtype=jnp.float64),
            state,
            jnp.asarray(step_size, dtype=jnp.float64),
            None,
        )
        successful = successful and bool(result.successful)
        maximum_energy_excess = max(
            maximum_energy_excess,
            float(
                np.asarray(
                    result.evidence.energy_balance_defect
                    - result.evidence.energy_tolerance
                )
            ),
        )
        maximum_mass_defect = max(
            maximum_mass_defect,
            float(np.asarray(result.evidence.mass_defect)),
        )
        state = result.accepted_state
    return (
        state,
        successful,
        maximum_energy_excess,
        maximum_mass_defect,
    )


def _rms_error(left, right, /) -> float:
    difference = np.asarray(left) - np.asarray(right)
    return float(np.sqrt(np.mean(difference * difference)))


def _temporal_orders(errors: list[float], /) -> list[float]:
    orders = []
    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        orders.append(math.log(coarse / fine, 2.0))
    return orders


def _planar_interface_qualification() -> dict[str, object]:
    model = _model(0.125)
    width = float(
        np.asarray(model.closure.characteristic_interface_width(model.thermodynamics))
    )
    analytic_tension = float(
        np.asarray(model.closure.planar_surface_tension(model.thermodynamics))
    )
    levels = []
    relative_errors = []
    for x_cells, y_cells in ((16, 4), (24, 6), (32, 8)):
        mesh = _rectangle_mesh(
            x_cells,
            y_cells,
            x_bounds=(-2.0, 2.0),
        )
        method = _allen_cahn(mesh, model)
        coordinates = method.discretization.dof_maps[
            method.field_index
        ].evaluate_coordinates(
            method.discretization.mesh,
            method.discretization.default_runtime.coordinates,
        )
        profile = jnp.tanh(coordinates[:, 0] / width)
        energy = float(np.asarray(method.energy(profile)))
        relative_error = abs(energy - analytic_tension) / analytic_tension
        relative_errors.append(relative_error)
        levels.append(
            {
                "x_cells": x_cells,
                "y_cells": y_cells,
                "degrees_of_freedom": int(profile.size),
                "cells_across_transition": float(
                    np.asarray(method.resolution.cells_across_transition)
                ),
                "energy_per_unit_length": energy,
                "relative_surface_tension_error": relative_error,
                "method_id": method.method_id,
            }
        )
    monotone = all(
        following < previous
        for previous, following in zip(
            relative_errors[:-1], relative_errors[1:], strict=True
        )
    )
    passed = monotone and relative_errors[-1] < 0.03
    return {
        "status": "pass" if passed else "fail",
        "characteristic_width": width,
        "analytic_surface_tension": analytic_tension,
        "levels": levels,
        "monotone_error_reduction": monotone,
        "finest_relative_error": relative_errors[-1],
    }


def _temporal_qualification() -> dict[str, object]:
    mesh = _rectangle_mesh(1, 1)
    model = _model(1.0)
    termination = phx.nonlinear.NonlinearTermination(
        absolute_residual=1.0e-10,
        relative_residual=1.0e-10,
        maximum_steps=100,
    )
    allen_cahn = _allen_cahn(mesh, model, termination=termination)
    cahn_hilliard = _cahn_hilliard(mesh, model, termination=termination)
    initial_values = jnp.asarray((-0.2, 0.1, -0.1, 0.3), dtype=jnp.float64)

    ac_initial = allen_cahn.initialize(initial_values)
    ac_sizes = (0.01, 0.005, 0.0025)
    (
        ac_reference,
        ac_reference_success,
        _,
    ) = _advance_allen_cahn(
        allen_cahn,
        ac_initial,
        0.00125,
        0.02,
    )
    ac_states = []
    ac_success = ac_reference_success
    ac_energy_excess = 0.0
    for step_size in ac_sizes:
        state, successful, energy_excess = _advance_allen_cahn(
            allen_cahn,
            ac_initial,
            step_size,
            0.02,
        )
        ac_states.append(state)
        ac_success = ac_success and successful
        ac_energy_excess = max(ac_energy_excess, energy_excess)
    ac_errors = [_rms_error(state.phase, ac_reference.phase) for state in ac_states]
    ac_orders = _temporal_orders(ac_errors)

    ch_initial = cahn_hilliard.initialize(initial_values)
    ch_sizes = (0.005, 0.0025, 0.00125)
    (
        ch_reference,
        ch_reference_success,
        _,
        ch_reference_mass_defect,
    ) = _advance_cahn_hilliard(
        cahn_hilliard,
        ch_initial,
        0.000625,
        0.01,
    )
    ch_states = []
    ch_success = ch_reference_success
    ch_energy_excess = 0.0
    ch_mass_defect = ch_reference_mass_defect
    for step_size in ch_sizes:
        state, successful, energy_excess, mass_defect = _advance_cahn_hilliard(
            cahn_hilliard,
            ch_initial,
            step_size,
            0.01,
        )
        ch_states.append(state)
        ch_success = ch_success and successful
        ch_energy_excess = max(ch_energy_excess, energy_excess)
        ch_mass_defect = max(ch_mass_defect, mass_defect)
    ch_errors = [
        _rms_error(state.concentration, ch_reference.concentration) for state in ch_states
    ]
    ch_orders = _temporal_orders(ch_errors)

    ac_passed = (
        ac_success
        and all(fine < coarse for coarse, fine in zip(ac_errors[:-1], ac_errors[1:]))
        and min(ac_orders) > 0.7
        and ac_energy_excess <= 0.0
    )
    ch_passed = (
        ch_success
        and all(fine < coarse for coarse, fine in zip(ch_errors[:-1], ch_errors[1:]))
        and min(ch_orders) > 0.7
        and ch_energy_excess <= 0.0
        and ch_mass_defect
        <= float(
            np.asarray(
                cahn_hilliard.plan.acceptance.mass_tolerance(
                    ch_initial.reference_mass,
                    cahn_hilliard.domain_measure,
                )
            )
        )
    )
    return {
        "status": "pass" if ac_passed and ch_passed else "fail",
        "allen_cahn": {
            "status": "pass" if ac_passed else "fail",
            "step_sizes": list(ac_sizes),
            "errors_against_fine_reference": ac_errors,
            "observed_orders": ac_orders,
            "maximum_energy_gate_excess": ac_energy_excess,
            "terminal_energy": float(np.asarray(ac_states[-1].energy)),
        },
        "cahn_hilliard": {
            "status": "pass" if ch_passed else "fail",
            "step_sizes": list(ch_sizes),
            "errors_against_fine_reference": ch_errors,
            "observed_orders": ch_orders,
            "maximum_energy_gate_excess": ch_energy_excess,
            "maximum_mass_defect": ch_mass_defect,
            "terminal_energy": float(np.asarray(ch_states[-1].energy)),
        },
    }


def _backend_qualification() -> dict[str, object]:
    mesh = _rectangle_mesh(1, 1)
    model = _model(1.0)
    policies = {
        name: phx.equations.FiniteElementExecutionPolicy(
            realization=name,
            local_kernel="auto",
            accumulation="deterministic",
        )
        for name in ("matrix_free", "sparse")
    }
    initial_values = jnp.asarray((-0.2, 0.1, -0.1, 0.3), dtype=jnp.float64)

    ac_results = {}
    ch_results = {}
    for name, policy in policies.items():
        ac_method = _allen_cahn(mesh, model, execution_policy=policy)
        ac_results[name] = ac_method.step_detailed(
            jnp.asarray(0),
            jnp.asarray(0.0),
            ac_method.initialize(initial_values),
            jnp.asarray(0.005),
        )
        ch_method = _cahn_hilliard(mesh, model, execution_policy=policy)
        ch_results[name] = ch_method.step_detailed(
            jnp.asarray(0),
            jnp.asarray(0.0),
            ch_method.initialize(initial_values),
            jnp.asarray(0.005),
        )

    ac_difference = float(
        np.max(
            np.abs(
                np.asarray(ac_results["matrix_free"].candidate_state.phase)
                - np.asarray(ac_results["sparse"].candidate_state.phase)
            )
        )
    )
    ch_difference = float(
        np.max(
            np.abs(
                np.asarray(ch_results["matrix_free"].candidate_state.concentration)
                - np.asarray(ch_results["sparse"].candidate_state.concentration)
            )
        )
    )
    accepted = all(
        bool(result.successful)
        for result in tuple(ac_results.values()) + tuple(ch_results.values())
    )
    passed = accepted and max(ac_difference, ch_difference) < 1.0e-10
    return {
        "status": "pass" if passed else "fail",
        "all_steps_accepted": accepted,
        "allen_cahn_maximum_difference": ac_difference,
        "cahn_hilliard_maximum_difference": ch_difference,
    }


def qualify() -> dict[str, object]:
    if not bool(jax.config.read("jax_enable_x64")):
        raise ValueError("Phase-field qualification requires JAX float64 support.")
    planar = _planar_interface_qualification()
    temporal = _temporal_qualification()
    backends = _backend_qualification()
    passed = all(section["status"] == "pass" for section in (planar, temporal, backends))
    return {
        "status": "pass" if passed else "fail",
        "capability": "binary-closed-phase-field-fem",
        "support": {
            "free_energy": "symmetric-quartic-double-well",
            "time_discretization": "first-order-convex-splitting",
            "element": "conforming-P1-triangle",
            "cell_blocks": 1,
            "boundary": "closed-natural",
            "precision": "float64",
            "execution": "single-device-deterministic",
        },
        "planar_interface": planar,
        "temporal_and_balance": temporal,
        "backend_agreement": backends,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qualify the closed binary phase-field production route."
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualify()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
        print(arguments.output)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
