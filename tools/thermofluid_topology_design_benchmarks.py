#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.thermofluids._topology_design import (
    ThermofluidMaterial,
    ThermofluidTopologyDesign,
)
from phydrax.optim._pde_constrained_mma import ReducedMMA
from phydrax.optim._state_design_linearization import (
    prepare_state_design_linearization,
    state_design_response_vjp,
)


def _drive(_time, velocity, _args):
    return jnp.ones_like(velocity[0]), jnp.zeros_like(velocity[1])


def heated_channel(count, resistance, *, smoke=False):
    """Pressure-gradient-driven periodic channel with cold stationary side walls."""
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(2 * count, periodic=True),
            phx.discretization.UniformCellAxisSpec(count, periodic=False),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [2.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators).prepare()
    projection = phx.solver.MACPressureProjectionPlan(
        operators,
        solve_method="transform",
        tolerance=1e-9,
    )
    scalar_problem = phx.discretization.MACScalarProblem(
        (phx.discretization.MACScalarTransport("temperature", 0.01, advection="upwind"),)
    )
    layout = phx.discretization.MACScalarLayout(operators, ("temperature",))
    cold = phx.discretization.MACScalarBoundaryCondition("dirichlet", 0.0)
    scalar_boundaries = phx.discretization.MACScalarBoundarySet(
        layout,
        walls={"temperature": {"y": (cold, cold)}},
    )
    transport = scalar_problem.prepare(operators, boundaries=scalar_boundaries)
    dynamics = phx.equations.compile_mac_scalar_buoyancy(
        phx.equations.IncompressibleFlowProblem(
            2,
            0.02,
            forcing=_drive,
            forcing_id="heated-channel-unit-pressure-gradient",
        ),
        momentum,
        projection,
        scalar_problem,
        transport,
        phx.equations.MACBuoyancyLaw([0.0, -1.0], {"temperature": -0.05}),
    )
    velocity = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    initial = dynamics.pack_state(
        velocity, {"temperature": jnp.zeros(finite_volume.cell_shape)}
    )
    points = finite_volume.cell_centers
    active = points[..., 0] >= 0.2
    transform = phx.optim.DensityTransformPlan(
        phx.optim.ConicDensityFilterPlan(
            points.reshape((-1, 2)),
            0.20,
            active.reshape((-1,)),
            jnp.zeros((points[..., 0].size,)),
            finite_volume.cell_volumes.reshape((-1,)),
        ),
        phx.optim.TanhDensityProjectionPlan(0.5),
    ).prepare()
    source = 4.0 * jnp.exp(
        -(((points[..., 0] - 0.65) / 0.25) ** 2) - ((points[..., 1] - 0.5) / 0.22) ** 2
    )
    workflow = ThermofluidTopologyDesign(
        dynamics,
        transform,
        ThermofluidMaterial(
            solid_resistance=resistance,
            fluid_conductivity=0.01,
            solid_conductivity=0.10,
            heat_capacity=1.0,
        ),
        initial,
        heat_source=source,
        final_time=0.02 if smoke else 0.60,
        step_size=0.01 if smoke else 0.0025,
        maximum_solid_fraction=0.40,
        beta=2.0,
        resistance_weight=0.002,
    )
    design = jnp.where(active, 0.22 + 0.02 * jnp.cos(2.0 * jnp.pi * points[..., 1]), 0.0)
    return workflow, design.reshape((-1,))


def _ledger(evidence):
    return {
        "objective": float(evidence.objective),
        "solid_fraction": float(evidence.solid_fraction),
        "maximum_temperature": float(evidence.maximum_temperature),
        "heat_content": float(evidence.heat_content),
        "heat_content_rate": float(evidence.heat_content_rate),
        "heat_source_power": float(evidence.heat_source_power),
        "boundary_heat_outflow": float(evidence.boundary_heat_outflow),
        "thermal_balance_defect": float(evidence.thermal_balance_defect),
        "divergence_norm": float(evidence.divergence_norm),
        "boundary_mass_flux": float(evidence.boundary_mass_flux),
        "kinetic_energy": float(evidence.kinetic_energy),
        "kinetic_energy_rate": float(evidence.kinetic_energy_rate),
        "external_power": float(evidence.external_power),
        "buoyancy_power": float(evidence.buoyancy_power),
        "viscous_dissipation": float(evidence.viscous_dissipation),
        "resistance_power": float(evidence.resistance_power),
        "resistance_work_defect": float(evidence.resistance_work_defect),
        "kinetic_balance_defect": float(evidence.kinetic_balance_defect),
        "solid_velocity_energy_fraction": float(evidence.solid_velocity_energy_fraction),
        "endpoint_rate_norm_not_steady_certificate": float(evidence.endpoint_rate_norm),
        "successful": bool(evidence.successful),
    }


def _balanced(evidence):
    defects = jnp.asarray(
        (
            evidence.thermal_balance_defect,
            evidence.divergence_norm,
            evidence.boundary_mass_flux,
            evidence.resistance_work_defect,
            evidence.kinetic_balance_defect,
        )
    )
    return bool(
        evidence.successful
        & jnp.all(jnp.isfinite(defects))
        & (jnp.max(jnp.abs(defects)) < 2e-7)
        & (evidence.resistance_power >= 0.0)
    )


def _feasible_transfer(workflow, design):
    """Preserve the material bound after grid transfer by monotone raw scaling."""
    if bool(
        workflow.solid_fraction(workflow.density(design))
        <= workflow.maximum_solid_fraction
    ):
        return design, 1.0
    plan = workflow.transform.plan.filter

    def scaled(scale):
        return jnp.where(plan.design_mask, scale * design, plan.fixed_density)

    def bisect(_index, interval):
        lower, upper = interval
        midpoint = 0.5 * (lower + upper)
        fraction = workflow.solid_fraction(workflow.density(scaled(midpoint)))
        feasible = fraction <= workflow.maximum_solid_fraction - 1e-8
        return jnp.where(feasible, midpoint, lower), jnp.where(feasible, upper, midpoint)

    lower, _ = jax.lax.fori_loop(
        0,
        40,
        bisect,
        (jnp.asarray(0.0), jnp.asarray(1.0)),
    )
    return scaled(lower), float(lower)


def run_thermofluid_topology_benchmark(*, smoke=False):
    records = []
    previous_design = None
    previous_shape = None
    grids = (3,) if smoke else (6, 8)
    resistances = (8.0,) if smoke else (20.0, 80.0)
    for count in grids:
        for resistance in resistances:
            workflow, seed = heated_channel(count, resistance, smoke=smoke)
            shape = workflow.heat_source.shape
            if previous_design is None:
                initial_design = seed
            else:
                prolonged = jax.image.resize(
                    previous_design.reshape(previous_shape),
                    shape,
                    method="linear",
                ).reshape((-1,))
                plan = workflow.transform.plan.filter
                initial_design = jnp.where(
                    plan.design_mask, jnp.clip(prolonged, 0.0, 1.0), plan.fixed_density
                )
            initial_design, transfer_scale = _feasible_transfer(workflow, initial_design)
            problem = workflow.state_design_problem()
            initial_result = workflow.integrate(initial_design)
            initial_value = workflow.objective(initial_result.final_state, initial_design)
            # Derivative check uses an interior point, not a threshold or a bound kink.
            direction = jnp.sin(jnp.arange(seed.size, dtype=seed.dtype) + 0.3)
            direction = jnp.where(
                workflow.transform.plan.filter.design_mask, direction, 0.0
            )
            direction = direction / jnp.sqrt(jnp.sum(direction * direction))
            linearization = prepare_state_design_linearization(
                problem, seed, workflow.initial_state
            )
            sensitivity = state_design_response_vjp(linearization)
            derivative = jnp.sum(sensitivity.design_cotangent * direction)
            epsilon = 2e-4

            def terminal_objective(design):
                run = workflow.integrate(design)
                return workflow.objective(run.final_state, design)

            finite_difference = (
                terminal_objective(seed + epsilon * direction)
                - terminal_objective(seed - epsilon * direction)
            ) / (2.0 * epsilon)
            gradient_error = jnp.abs(derivative - finite_difference) / jnp.maximum(
                jnp.maximum(jnp.abs(derivative), jnp.abs(finite_difference)),
                1e-8,
            )
            start = perf_counter()
            result = phx.optim.solve_state_design(
                problem,
                workflow.initial_state,
                initial_design,
                method=ReducedMMA(policy=phx.optim.MMAPolicy(move_limit=0.12)),
                termination=phx.optim.OptimizationTermination(
                    maximum_steps=2 if smoke else 24,
                    absolute_optimality=1e-6,
                    relative_optimality=0.0,
                    absolute_step=1e-10,
                    relative_step=0.0,
                ),
            )
            result.objective.block_until_ready()
            elapsed = perf_counter() - start
            density = workflow.density(result.design)
            relaxed = workflow.evidence(result.state, density)
            binary = workflow.binary_reanalysis(result.design, eta=0.3)
            # Same binary geometry at a higher drag isolates permeability leakage
            # from geometric/optimizer changes. It is not another optimization.
            stronger, _ = heated_channel(count, 2.0 * resistance, smoke=smoke)
            stronger_run = stronger.integrate_density(binary.density)
            stronger_evidence = stronger.evidence(
                stronger_run.final_state, binary.density
            )
            physical_improvement = float(initial_value - relaxed.objective)
            passed = bool(
                initial_result.successful
                & result.state_acceptance.accepted
                & sensitivity.accepted
                & (gradient_error < 3e-3)
                & (relaxed.solid_fraction <= workflow.maximum_solid_fraction + 1e-6)
                & (relaxed.objective <= initial_value + 1e-9)
                & binary.integration_successful
                & stronger_run.successful
                & (
                    stronger_evidence.solid_velocity_energy_fraction
                    <= binary.evidence.solid_velocity_energy_fraction + 1e-12
                )
            ) and all(_balanced(e) for e in (relaxed, binary.evidence, stronger_evidence))
            records.append(
                {
                    "cell_shape": list(shape),
                    "solid_resistance": resistance,
                    "final_time": workflow.final_time,
                    "step_size": workflow.step_size,
                    "derivative_contract": "fixed SSPRK33 finite-horizon algorithm; not steady root",
                    "initial_objective": float(initial_value),
                    "objective_improvement": physical_improvement,
                    "grid_transfer_raw_density_scale": transfer_scale,
                    "optimization_status": int(result.status),
                    "optimization_converged": bool(result.successful),
                    "optimization_iterations": int(result.diagnostics.iterations),
                    "state_accepted": bool(result.state_acceptance.accepted),
                    "gradient_accepted": bool(sensitivity.accepted),
                    "directional_derivative": float(derivative),
                    "finite_difference": float(finite_difference),
                    "gradient_relative_error": float(gradient_error),
                    "relaxed": _ledger(relaxed),
                    "binary": _ledger(binary.evidence),
                    "binary_volume_feasible": bool(binary.material_bound_satisfied),
                    "same_binary_double_resistance": _ledger(stronger_evidence),
                    "double_resistance_leakage_change": float(
                        stronger_evidence.solid_velocity_energy_fraction
                        - binary.evidence.solid_velocity_energy_fraction
                    ),
                    "optimized_density": np.asarray(density).tolist(),
                    "binary_density": np.asarray(binary.density).tolist(),
                    "optimized_temperature": np.asarray(
                        workflow.dynamics.unpack_state(result.state)[1]["temperature"]
                    ).tolist(),
                    "optimization_seconds": elapsed,
                    "passed": passed,
                }
            )
            previous_design, previous_shape = result.design, shape
    return {
        "case": "heated-periodic-channel-material-and-resistance-design",
        "envelope": (
            "2D fixed uniform grid; unit-density Newtonian/Boussinesq; "
            "constant heat capacity; no dissipation heating"
        ),
        "continuation": (
            "raw design prolonged and volume-feasibly scaled between grids; "
            "resistance continuation within each grid"
        ),
        "records": records,
        "passed": all(record["passed"] for record in records)
        and any(record["objective_improvement"] > 1e-9 for record in records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_thermofluid_topology_benchmark(smoke=args.smoke)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
