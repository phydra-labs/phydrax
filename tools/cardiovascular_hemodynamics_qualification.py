"""Deterministic qualification for fixed-wall cardiovascular hemodynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.cardiovascular.circulation._components import Resistance
from phydrax.applications.cardiovascular.hemodynamics._domain import (
    compare_lbm_mac,
    FixedWallLumenRegion,
    HemodynamicsScaling,
    HemodynamicsValidityLimits,
    PoiseuillePipeReference,
    WomersleyPipeReference,
)
from phydrax.applications.cardiovascular.hemodynamics._fixed_wall_lbm import (
    FixedWallLBMPlan,
)
from phydrax.applications.cardiovascular.hemodynamics._ports import (
    CirculationPortBinding,
    FlowTerminalPort,
    PressureTerminalPort,
    terminal_balance_evidence,
    TerminalDirection,
    TerminalFace,
    TerminalPortValues,
)
from phydrax.applications.cardiovascular.hemodynamics._rheology import (
    CarreauYasudaRheology,
    NewtonianRheology,
)


def _grid(shape, periodic):
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic[axis])
            for axis, count in enumerate(shape)
        ),
        axis_names=("x", "y", "z"),
    ).prepare(
        jnp.asarray(
            (
                (0.0, 0.0, 0.0),
                tuple(float(count) for count in shape),
            )
        )
    )


def _workflow(shape=(6, 8, 4)):
    # Nonperiodic transverse faces let the production plan install explicit
    # terminal-over-stationary-wall ownership at every D3Q19 edge link.
    grid = _grid(shape, (False, False, False))
    discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D3Q19()
    ).prepare()
    scaling = HemodynamicsScaling(
        1.0,
        1.0,
        1.06,
        reference_velocity_mm_per_ms=0.02,
    )
    component = Resistance("qualification_terminal", 1.0)
    terminals = (
        FlowTerminalPort(
            "inlet",
            TerminalFace("x", "lower", TerminalDirection.INTO_LUMEN),
            CirculationPortBinding(component, "inlet"),
        ),
        PressureTerminalPort(
            "outlet",
            TerminalFace("x", "upper", TerminalDirection.OUT_OF_LUMEN),
            CirculationPortBinding(component, "outlet"),
        ),
    )
    plan = FixedWallLBMPlan(
        discretization,
        scaling,
        FixedWallLumenRegion(np.ones(shape, dtype=bool)),
        terminals,
        NewtonianRheology(0.004, maximum_shear_rate_per_ms=2.0),
        limits=HemodynamicsValidityLimits(
            maximum_relative_mass_balance_defect=1.0e-5,
            maximum_terminal_flow_relative_defect=1.0e-6,
            maximum_terminal_power_relative_defect=1.0e-8,
        ),
    )
    return plan.prepare()


def _scaling_case():
    scaling = HemodynamicsScaling(
        0.25,
        0.01,
        1.06,
        reference_velocity_mm_per_ms=0.5,
        maximum_lattice_mach=0.05,
    )
    pressure = jnp.asarray((-2.0, 0.0, 12.0))
    flow = jnp.asarray((-5.0, 7.0))
    velocity = jnp.asarray((-0.2, 0.1, 0.3))
    errors = {
        "pressure": float(
            jnp.max(
                jnp.abs(
                    scaling.density_gauge_pressure(scaling.pressure_density(pressure))
                    - pressure
                )
            )
        ),
        "flow": float(
            jnp.max(
                jnp.abs(
                    scaling.physical_flow_rate(scaling.lattice_flow_rate(flow)) - flow
                )
            )
        ),
        "velocity": float(
            jnp.max(
                jnp.abs(
                    scaling.physical_velocity(scaling.lattice_velocity(velocity))
                    - velocity
                )
            )
        ),
    }
    return {
        "reference_lattice_mach": scaling.reference_lattice_mach,
        "maximum_roundtrip_error": max(errors.values()),
        "errors": errors,
        "passed": bool(
            scaling.reference_lattice_mach <= scaling.maximum_lattice_mach
            and max(errors.values()) <= 2.0e-12
        ),
    }


def _rheology_case():
    shear = jnp.asarray((0.0, 1.0e-3, 1.0e-2, 1.0))
    newtonian = NewtonianRheology(0.004)
    limit = CarreauYasudaRheology(0.004, 0.004, 10.0, 0.5, 2.0)
    thinning = CarreauYasudaRheology(0.056, 0.0035, 3313.0, 0.3568, 2.0)
    newtonian_values = np.asarray(newtonian.dynamic_viscosity(shear))
    limit_values = np.asarray(limit.dynamic_viscosity(shear))
    thinning_values = np.asarray(thinning.dynamic_viscosity(shear))
    limit_error = float(np.max(np.abs(limit_values - newtonian_values)))
    monotone = bool(np.all(np.diff(thinning_values) <= 0.0))
    return {
        "newtonian_limit_maximum_error_kpa_ms": limit_error,
        "carreau_yasuda_viscosity_kpa_ms": thinning_values.tolist(),
        "shear_thinning_monotone": monotone,
        "passed": bool(limit_error <= 1.0e-14 and monotone),
    }


def _reference_case():
    pipe = PoiseuillePipeReference(1.5, 20.0, 0.8, 0.004)
    radius = np.linspace(0.0, 1.5, 4001)
    velocity = np.asarray(pipe.axial_velocity(radius))
    integrated = float(np.trapezoid(2.0 * np.pi * radius * velocity, radius))
    exact = float(pipe.flow_rate_mm3_per_ms)
    poiseuille_error = abs(integrated - exact) / exact

    frequency = 1.0e-6
    gradient = 0.02
    womersley = WomersleyPipeReference(1.0, gradient, 0.004, 1.0, frequency)
    quasi_steady = PoiseuillePipeReference(1.0, 1.0, gradient, 0.004)
    samples = jnp.linspace(0.0, 1.0, 65)
    pulsatile = np.asarray(womersley.axial_velocity(samples, 0.0))
    steady = np.asarray(quasi_steady.axial_velocity(samples))
    womersley_error = float(np.linalg.norm(pulsatile - steady) / np.linalg.norm(steady))
    return {
        "poiseuille_flow_relative_error": poiseuille_error,
        "womersley_number": womersley.womersley_number,
        "womersley_quasi_steady_relative_error": womersley_error,
        "passed": bool(
            poiseuille_error <= 2.0e-7
            and womersley.womersley_number < 0.02
            and womersley_error <= 2.0e-5
        ),
    }


def _terminal_and_candidate_case():
    prepared = _workflow()
    shape = prepared.discretization.grid.shape
    velocity = jnp.zeros(shape + (3,)).at[..., 0].set(0.01)
    pressure = jnp.zeros(shape).at[0, :, :].set(12.0).at[-1, :, :].set(10.0)
    measurements = prepared.terminal_measurements.measure(pressure, velocity)
    area = float(prepared.terminal_measurements.areas_mm2[0])
    directed_flow = 0.01 * area
    values = TerminalPortValues(
        jnp.asarray((12.0, 10.0)),
        jnp.asarray((directed_flow, directed_flow)),
    )
    balance = terminal_balance_evidence(
        prepared.terminal_measurements,
        measurements,
        values,
        storage_volume_change_mm3=0.0,
        time_step_ms=1.0,
        flow_relative_tolerance=1.0e-12,
        pressure_absolute_tolerance_kpa=1.0e-12,
        volume_relative_tolerance=1.0e-12,
        power_relative_tolerance=1.0e-12,
    )
    state = prepared.initialize_state()
    candidate = prepared.candidate(
        state, TerminalPortValues(jnp.zeros((2,)), jnp.zeros((2,)))
    )
    committed = prepared.commit(state, candidate)
    return {
        "outward_flow_mm3_per_ms": np.asarray(
            measurements.outward_flow_mm3_per_ms
        ).tolist(),
        "pressure_residual_kpa": np.asarray(balance.pressure_residual_kpa).tolist(),
        "volume_relative_defect": float(balance.volume_relative_defect),
        "power_relative_defect": float(balance.power_relative_defect),
        "candidate_status": int(candidate.evidence.status),
        "candidate_maximum_lattice_mach": float(candidate.evidence.maximum_lattice_mach),
        "committed_step_index": int(committed.step_index),
        "scope": prepared.scope.statement,
        "passed": bool(balance.passed and candidate.evidence.successful),
    }


def _lbm_mac_case():
    """Evolve native D3Q19 and MAC states under the same body acceleration."""

    shape = (8, 8, 4)
    grid = _grid(shape, (True, False, True))
    viscosity = 0.02
    time_step = 0.1
    step_count = 12
    acceleration = jnp.asarray((1.0e-3, 0.0, 0.0))

    lbm_discretization = phx.discretization.LatticeBoltzmannPlan(
        grid, phx.discretization.D3Q19()
    ).prepare()
    lbm_problem = phx.equations.LatticeBoltzmannProblem(
        "hemodynamics-lbm-mac-comparison",
        3,
        reference_density=1.0,
        acceleration=lambda time, coordinates, parameters: parameters,
        acceleration_id="uniform-comparison-acceleration",
    )
    lbm = phx.equations.compile_lattice_boltzmann_problem(
        lbm_problem,
        lbm_discretization,
        phx.discretization.LatticeBoltzmannMethodPlan(
            phx.discretization.TRTCollisionPlan(),
            forcing=phx.discretization.GuoForcingPlan(),
        ),
        phx.discretization.LatticeBoltzmannBoundaryPlan(),
        time_step=time_step,
    )
    lbm_parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(
        viscosity, force_parameters=acceleration
    )
    lbm_state = lbm.initialize_state(1.0, jnp.zeros((3,)), lbm_parameters)
    lbm_method = phx.solver.LatticeBoltzmannFixedStepMethod(lbm.dynamics)
    lbm_successful = True
    for step_index in range(step_count):
        result = lbm_method.step(
            jnp.asarray(step_index, dtype=jnp.int32),
            jnp.asarray(step_index * time_step),
            lbm_state,
            jnp.asarray(time_step),
            lbm_parameters,
        )
        lbm_state = result.accepted_state
        lbm_successful = lbm_successful and bool(result.successful)
    final_time = step_count * time_step
    lbm_macroscopic = lbm.macroscopic_state(final_time, lbm_state, lbm_parameters)

    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide(
                "y",
                "lower",
                "no-slip",
                provider=phx.discretization.MACBoundaryProvider(jnp.zeros(3)),
            ),
            phx.discretization.MACBoundarySide(
                "y",
                "upper",
                "no-slip",
                provider=phx.discretization.MACBoundaryProvider(jnp.zeros(3)),
            ),
        ),
    )
    momentum = phx.discretization.MACMomentumPlan(
        operators, boundaries=boundaries
    ).prepare()

    def mac_forcing(time, velocity, parameters):
        del time
        return tuple(
            jnp.ones_like(component) * parameters[axis]
            for axis, component in enumerate(velocity)
        )

    mac = phx.equations.compile_mac_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(
            3,
            viscosity,
            forcing=mac_forcing,
            forcing_id="uniform-comparison-acceleration",
        ),
        momentum,
        phx.solver.MACPressureProjectionPlan(
            operators,
            boundaries=boundaries,
            solve_method="transform",
            tolerance=1.0e-11,
        ),
    )
    mac_state = mac.project_state(
        tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts),
        args=acceleration,
    )
    mac_method = phx.solver.SSPRK33FixedStepMethod(mac)
    mac_successful = True
    for step_index in range(step_count):
        result = mac_method.step(
            jnp.asarray(step_index, dtype=jnp.int32),
            jnp.asarray(step_index * time_step),
            mac_state,
            jnp.asarray(time_step),
            acceleration,
        )
        mac_state = result.accepted_state
        mac_successful = mac_successful and bool(result.successful)
    mac_faces = mac.physical_state(final_time, mac_state, acceleration)
    mac_cells = []
    for axis, component in enumerate(mac_faces):
        moved = jnp.moveaxis(component, axis, 0)
        centered = (
            0.5 * (moved + jnp.roll(moved, -1, axis=0))
            if grid.structured_axes[axis].periodic
            else 0.5 * (moved[:-1] + moved[1:])
        )
        mac_cells.append(jnp.moveaxis(centered, 0, axis))
    mac_velocity = jnp.stack(tuple(mac_cells), axis=-1)
    mac_pressure = mac.pressure_field(final_time, mac_state, acceleration)
    lbm_pressure = lbm_macroscopic.pressure - jnp.mean(lbm_macroscopic.pressure)
    mac_pressure = mac_pressure - jnp.mean(mac_pressure)
    evidence = compare_lbm_mac(
        lbm_macroscopic.velocity,
        mac_velocity,
        lbm_pressure,
        mac_pressure,
        jnp.ones(shape),
        velocity_relative_tolerance=0.25,
        pressure_relative_tolerance=0.25,
    )
    return {
        "shape": list(shape),
        "steps": step_count,
        "time_step_ms": time_step,
        "forcing_mm_per_ms2": np.asarray(acceleration).tolist(),
        "velocity_relative_l2": float(evidence.velocity_relative_l2),
        "pressure_relative_l2": float(evidence.pressure_relative_l2),
        "lbm_route_id": evidence.lbm_route_id,
        "mac_route_id": evidence.mac_route_id,
        "lbm_successful": lbm_successful,
        "mac_successful": mac_successful,
        "passed": bool(evidence.passed and lbm_successful and mac_successful),
    }


def qualification_report():
    cases = {
        "scaling": _scaling_case(),
        "rheology": _rheology_case(),
        "poiseuille_womersley": _reference_case(),
        "terminal_candidate": _terminal_and_candidate_case(),
        "lbm_mac": _lbm_mac_case(),
    }
    return {
        "scope": "fixed-wall numerical hemodynamics; no FSI, curved-wall accuracy, or clinical claim",
        "cases": cases,
        "passed": all(case["passed"] for case in cases.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualification_report()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
