"""Structural, variational, relaxation, and flow evidence for nematics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import capture_environment


def qualification():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(16, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    finite_difference = phx.discretization.periodic_finite_difference(grid)
    basis = phx.equations.NematicTensorBasis(3)
    closure = phx.equations.LandauDeGennesClosure(basis)
    parameters = phx.equations.LandauDeGennesParameters(-1.0, 0.0, 1.0, 0.05)
    dynamics = phx.solver.PreparedNematicDynamics(
        finite_difference,
        closure,
        parameters,
        phx.equations.BerisEdwardsParameters(0.5, 0.7),
        energy_tolerance=1e-8,
    )
    compact = jnp.zeros((16, 5)).at[:, 0].set(0.1)
    before = dynamics.evaluate(compact)
    advanced = dynamics.step(compact, 1e-3)
    semi_implicit = phx.solver.PreparedNematicSemiImplicitStepPlan(dynamics, 1e-3).step(
        compact
    )
    tensor = basis.decode(compact[0])
    gradient = jax.grad(
        lambda value: (
            closure.evaluate(
                value,
                jnp.zeros((1, 3, 5))[0],
                jnp.zeros((5,)),
                parameters,
            ).bulk_energy_density
        )
    )(compact[0])
    local = closure.evaluate(
        compact[0],
        jnp.zeros((1, 5)),
        jnp.zeros((5,)),
        parameters,
    )
    variational_error = float(jnp.max(jnp.abs(gradient + local.molecular_field)))
    cases = {
        "tensor_structure": {
            "trace_residual": float(jnp.abs(jnp.trace(tensor))),
            "symmetry_residual": float(jnp.max(jnp.abs(tensor - tensor.T))),
            "passed": bool(jnp.abs(jnp.trace(tensor)) < 1e-14),
        },
        "variational_derivative": {
            "maximum_error": variational_error,
            "passed": variational_error < 1e-10,
        },
        "passive_relaxation": {
            "energy_change": float(
                advanced.evaluation.total_free_energy - before.total_free_energy
            ),
            "passed": bool(advanced.successful),
        },
        "semi_implicit_relaxation": {
            "energy_change": float(
                semi_implicit.evaluation.total_free_energy - before.total_free_energy
            ),
            "passed": bool(semi_implicit.successful),
        },
    }
    return {
        "environment": capture_environment().to_dict(),
        "evidence_levels": {
            "invariant_complete": bool(cases["tensor_structure"]["passed"]),
            "physics_qualified": bool(cases["passive_relaxation"]["passed"]),
            "differentiation_qualified": bool(cases["variational_derivative"]["passed"]),
            "execution_qualified": bool(
                cases["passive_relaxation"]["passed"]
                and cases["semi_implicit_relaxation"]["passed"]
            ),
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
        default=Path("benchmarks/liquid_crystal_qualification.json"),
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
