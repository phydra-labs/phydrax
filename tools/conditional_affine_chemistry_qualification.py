"""Scientific qualification for conditional-affine chemical transitions."""

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
        ("A", "B", "C"),
        (
            phx.equations.ChemicalPhaseKind.GAS,
            phx.equations.ChemicalPhaseKind.GAS,
            phx.equations.ChemicalPhaseKind.GAS,
        ),
        jnp.asarray((1.0, 1.0, 3.0)),
        ("X", "Y"),
        jnp.asarray(((1, 0, 2), (0, 1, 1)), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
        gas_standard_pressure=101325.0,
    )
    thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.full((3,), 10.0),
        jnp.zeros((3,)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=2000.0,
    )
    return phx.equations.ChemicalMechanismIR(
        "conditional-affine-qualification",
        schema,
        thermodynamics,
        (
            phx.equations.ChemicalReactionSpec(
                "2A+B<->C",
                {"A": 2.0, "B": 1.0},
                {"C": 1.0},
                phx.equations.ArrheniusRatePlan(2.0),
                reverse_rate=phx.equations.ArrheniusRatePlan(0.5),
            ),
        ),
    ).prepare()


def _drivers(value):
    return phx.equations.ChemicalConditionalAffineDrivers(
        jnp.asarray((value,)),
        jnp.asarray(500.0),
        jnp.asarray(101325.0),
    )


def qualification():
    mechanism = _mechanism()
    plan = phx.equations.ChemicalConditionalAffinePlan(("B", "C"), ("A",))
    certificate = plan.analyze(mechanism)
    prepared = plan.prepare(mechanism)
    initial = jnp.asarray((2.0, 1.0, 0.0))
    horizon = 1.0e-2
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, horizon, 101), time_id="conditional-affine-reference"
    )
    reactor = phx.solver.ChemicalReactorPlan(
        mechanism,
        phx.solver.ChemicalReactorKind.ISOTHERMAL_CONSTANT_VOLUME,
        fixed_temperature=500.0,
        fixed_volume=1.0,
    )
    reference = reactor.solve_bdf(initial, grid)
    midpoint_driver = reference.states[50, 0]
    midpoint = prepared.advance(initial, _drivers(midpoint_driver), horizon)
    endpoint_error = float(
        jnp.max(jnp.abs(midpoint.candidate_state - reference.states[-1]))
    )

    first = prepared.advance(initial, _drivers(reference.states[25, 0]), horizon / 2.0)
    second = prepared.advance(
        first.candidate_state,
        _drivers(reference.states[75, 0]),
        horizon / 2.0,
    )
    refined_error = float(jnp.max(jnp.abs(second.candidate_state - reference.states[-1])))
    semigroup_defect = float(
        jnp.max(jnp.abs(second.candidate_state - midpoint.candidate_state))
    )

    def differentiated(driver):
        batched_drivers = phx.equations.ChemicalConditionalAffineDrivers(
            driver.reshape((1, 1)),
            jnp.asarray((500.0,)),
            jnp.asarray((101325.0,)),
        )
        return jnp.sum(
            prepared.advance(
                initial[None, :],
                batched_drivers,
                jnp.asarray((horizon,)),
            ).candidate_state
            ** 2
        )

    gradient = jax.grad(differentiated)(jnp.asarray(midpoint_driver))
    maximum_invariant = float(
        jnp.max(
            jnp.asarray(
                (
                    jnp.max(jnp.abs(midpoint.element_residual)),
                    jnp.abs(midpoint.charge_residual),
                    midpoint.affine_consistency_residual,
                )
            )
        )
    )
    cases = {
        "structural_certificate": {
            "channel_count": prepared.channel_count,
            "affine_size": prepared.affine_size,
            "driver_size": prepared.driver_size,
            "passed": bool(certificate.certified),
        },
        "midpoint_reference": {
            "endpoint_max_abs_error": endpoint_error,
            "two_half_steps_max_abs_error": refined_error,
            "semigroup_defect": semigroup_defect,
            "reference_successful": bool(reference.successful),
            "passed": bool(
                reference.successful
                & midpoint.successful
                & first.successful
                & second.successful
                & (endpoint_error < 2.0e-2)
                & (refined_error <= endpoint_error)
            ),
        },
        "physical_invariants": {
            "maximum_residual": maximum_invariant,
            "minimum_species": float(midpoint.minimum_species),
            "passed": bool(
                midpoint.successful
                & (maximum_invariant < 1.0e-11)
                & (midpoint.minimum_species >= 0.0)
            ),
        },
        "differentiation": {
            "driver_gradient": float(gradient),
            "passed": bool(jnp.isfinite(gradient)),
        },
    }
    return {
        "environment": capture_environment().to_dict(),
        "mechanism_id": mechanism.mechanism_id,
        "certificate_id": certificate.certificate_id,
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/conditional_affine_chemistry_qualification.json"),
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
