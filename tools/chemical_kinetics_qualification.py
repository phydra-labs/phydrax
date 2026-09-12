"""Invariant, physics, differentiation, and execution evidence for chemistry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import capture_environment


def _mechanism():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (
            phx.equations.ChemicalPhaseKind.GAS,
            phx.equations.ChemicalPhaseKind.GAS,
        ),
        jnp.asarray((1.0, 1.0)),
        ("X",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.asarray((0, 0), dtype=jnp.int32),
        gas_standard_pressure=101325.0,
    )
    thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((10.0, 10.0)),
        jnp.asarray((0.0, 0.0)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=1000.0,
    )
    return phx.equations.ChemicalMechanismIR(
        "qualification",
        schema,
        thermodynamics,
        (
            phx.equations.ChemicalReactionSpec(
                "A->B",
                {"A": 1.0},
                {"B": 1.0},
                phx.equations.ArrheniusRatePlan(2.0),
            ),
        ),
    ).prepare()


def qualification():
    mechanism = _mechanism()
    fields = mechanism.evaluate(jnp.asarray((1.0, 0.0)), 500.0, 101325.0)
    rate_gradient = jax.grad(
        lambda concentration: mechanism.evaluate(
            jnp.stack((concentration, 1.0 - concentration)), 500.0, 101325.0
        ).forward_progress_rates[0]
    )(jnp.asarray(0.7))
    reactor = phx.solver.ChemicalReactorPlan(
        mechanism,
        phx.solver.ChemicalReactorKind.ISOTHERMAL_CONSTANT_VOLUME,
        fixed_temperature=500.0,
        fixed_volume=1.0,
    )
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 1.0, 17), time_id="chemistry")
    solution = reactor.solve(jnp.asarray((1.0, 0.0)), grid)
    analytic_error = float(jnp.abs(solution.states[-1, 0] - jnp.exp(-2.0)))
    bdf_solution = reactor.solve_bdf(jnp.asarray((1.0, 0.0)), grid, maximum_order=2)
    bdf_error = float(jnp.abs(bdf_solution.states[-1, 0] - jnp.exp(-2.0)))
    initial_state = reactor.initial_state(jnp.asarray((1.0, 0.0)))
    dynamics = phx.solver.PreparedChemicalReactorDynamics(reactor)
    replay_problem = phx.solver.DifferentialProblem(
        dynamics,
        initial_state,
        t0=grid.t0,
        t1=grid.t1,
        args=None,
        problem_id=f"chemical-reactor-replay:{reactor.reactor_id}",
    )
    replay_prepared = phx.solver.prepare_rosenbrock(
        replay_problem,
        grid,
        adaptive=phx.solver.RosenbrockAdaptivePolicy(
            relative_tolerance=1e-6,
            absolute_tolerance=1e-9,
            initial_step=0.02,
            maximum_accepted_steps=1024,
            maximum_attempts=2048,
        ),
    )
    replay_source = phx.solver.solve_rosenbrock(replay_prepared)
    replay_schedule = phx.solver.schedule_rosenbrock(
        replay_prepared,
        replay_source,
    )
    replayed = phx.solver.solve_scheduled_rosenbrock(replay_schedule)
    replay_difference = float(jnp.max(jnp.abs(replayed.states - replay_source.states)))
    replay_gradient = jax.grad(
        lambda amount: phx.solver.solve_scheduled_rosenbrock(
            replay_schedule,
            initial_state=reactor.initial_state(jnp.stack((amount, jnp.asarray(0.0)))),
        ).states[-1, 0]
    )(jnp.asarray(1.0))
    replay_gradient_error = float(jnp.abs(replay_gradient - jnp.exp(-2.0)))
    replay_adequacy = replayed.stats["replay_adequacy"]
    jump = phx.stochastic.ChemicalJumpProcess(mechanism, 1.0)
    jump_intensity = jump.intensities(
        0.0,
        jnp.asarray((10.0, 0.0)),
        phx.stochastic.ChemicalJumpRuntime(500.0, 101325.0),
    )
    cases = {
        "conservation": {
            "element_residual": float(jnp.max(jnp.abs(fields.element_residual))),
            "charge_residual": float(jnp.max(jnp.abs(fields.charge_residual))),
            "passed": bool(fields.successful),
        },
        "differentiation": {
            "rate_gradient": float(rate_gradient),
            "passed": bool(
                jnp.isfinite(rate_gradient) & (jnp.abs(rate_gradient - 2.0) < 1e-10)
            ),
        },
        "stiff_reactor": {
            "rosenbrock_analytic_error": analytic_error,
            "bdf_analytic_error": bdf_error,
            "passed": bool(
                solution.successful
                & bdf_solution.successful
                & (analytic_error < 5e-4)
                & (bdf_error < 2e-2)
            ),
        },
        "rosenbrock_replay": {
            "accepted_steps": int(replay_source.stats["accepted_steps"]),
            "attempts": int(replay_source.stats["attempts"]),
            "source_schedule_id": replay_schedule.source_realization_id,
            "schedule_id": replay_schedule.schedule_id,
            "source_replay_max_difference": replay_difference,
            "maximum_error_ratio": float(replay_adequacy.maximum_error_ratio),
            "maximum_stage_residual": float(jnp.max(replay_adequacy.stage_residual_norm)),
            "initial_amount_gradient": float(replay_gradient),
            "gradient_error": replay_gradient_error,
            "passed": bool(
                replay_source.successful
                & replayed.successful
                & (replay_difference < 1e-10)
                & (replay_adequacy.maximum_error_ratio <= 1.0)
                & (replay_gradient_error < 2e-3)
            ),
        },
        "stochastic": {
            "first_channel_intensity": float(jump_intensity[0]),
            "passed": bool(
                jnp.isfinite(jump_intensity[0])
                & (jnp.abs(jump_intensity[0] - 20.0) < 1e-12)
            ),
        },
    }
    return {
        "environment": capture_environment().to_dict(),
        "evidence_levels": {
            "invariant_complete": bool(cases["conservation"]["passed"]),
            "physics_qualified": bool(cases["stiff_reactor"]["passed"]),
            "differentiation_qualified": bool(cases["differentiation"]["passed"]),
            "execution_qualified": bool(cases["stiff_reactor"]["passed"]),
            "deployment_qualified": False,
        },
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/chemical_kinetics_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = qualification()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
