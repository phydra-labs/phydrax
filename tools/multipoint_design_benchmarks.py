#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic shared/local and native axial-member multi-load optimization.

Run from the repository root with
``python -m tools.multipoint_design_benchmarks --case all --repeats 3``.
Every timing synchronizes device execution; failed physical acceptance or an
incorrect analytic design/gradient fails the command instead of producing a
successful performance record.
"""

from __future__ import annotations

import argparse
import json
import platform

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import measure_repeated, measure_synchronized
from phydrax.optim._multipoint_state_design import (
    MultipointStateDesignProblem,
    StateDesignCase,
)
from phydrax.optim._state_design_linearization import (
    prepare_state_design_linearization,
    state_design_response_vjp,
)


opt = phx.optim


def _policy():
    return opt.StateAcceptancePolicy(
        state_relative_tolerance=1e-9,
        state_absolute_tolerance=1e-8,
        adjoint_relative_tolerance=1e-9,
        adjoint_absolute_tolerance=1e-8,
    )


def _termination():
    return opt.OptimizationTermination(
        absolute_optimality=1e-6,
        relative_optimality=0.0,
        maximum_steps=200,
    )


def _inner(left, right):
    return phx.ein.contract("i,i->", left, right, backend="jax")


def _vector(design):
    return jnp.stack((design["shared"], *design["local"]))


def _block_state_evidence(evidence):
    return {
        name: {
            "accepted": bool(block.accepted),
            "status_accepted": bool(block.status_accepted),
            "residual_norm": float(block.residual_norm),
            "threshold": float(block.threshold),
            "reference_norm": float(block.reference_norm),
        }
        for name, block in zip(evidence.block_ids, evidence.blocks, strict=True)
    }


def _block_adjoint_evidence(evidence):
    return {
        name: {
            "accepted": bool(block.accepted),
            "status_accepted": bool(block.status_accepted),
            "transpose_defect_norm": float(block.transpose_defect_norm),
            "threshold": float(block.threshold),
        }
        for name, block in zip(evidence.block_ids, evidence.blocks, strict=True)
    }


def _solve_pair(multipoint, design, repeats):
    problem = multipoint.to_state_design_problem()
    reduced, reduced_time = measure_repeated(
        lambda: opt.solve_state_design(
            problem,
            multipoint.initial_state,
            design,
            method=opt.ReducedAdjoint(),
            termination=_termination(),
        ),
        warmup=1,
        repeats=repeats,
    )
    compilation, compile_seconds = measure_synchronized(
        lambda: opt.compile_structured_state_design(
            problem,
            multipoint.initial_state,
            design,
        )
    )
    structured, structured_time = measure_repeated(
        lambda: opt.solve_structured_state_design(
            compilation,
            method=opt.PrimalDualInteriorPoint(mode="sparse-augmented"),
            termination=_termination(),
        ),
        warmup=1,
        repeats=repeats,
    )
    assert bool(reduced.successful), f"Reduced solve failed: {int(reduced.status)}"
    assert bool(structured.successful), "Structured all-at-once solve failed."
    # Independently recertify every all-at-once physical block at its returned
    # state, rather than treating a global NLP residual as physical evidence.
    structured_evidence = problem.state_evidence(
        structured.state,
        structured.design,
        problem.residual(structured.state, structured.design),
        structured.optimization.optimization.status,
        reference_norm=0.0,
    )
    assert bool(structured_evidence.accepted), (
        "Structured physical recertification failed."
    )
    np.testing.assert_allclose(
        reduced.objective, structured.objective, atol=1e-5, rtol=1e-5
    )
    return (
        problem,
        reduced,
        structured,
        {
            "reduced_status": int(reduced.status),
            "objective": float(reduced.objective),
            "structured_objective": float(structured.objective),
            "reduced_time": reduced_time.to_seconds_dict(),
            "structured_compile_seconds": compile_seconds,
            "structured_time": structured_time.to_seconds_dict(),
            "reduced_states": _block_state_evidence(reduced.state_acceptance),
            "reduced_adjoints": _block_adjoint_evidence(reduced.adjoint_acceptance),
            "structured_states": _block_state_evidence(structured_evidence),
        },
    )


def analytic_shared_local(repeats):
    """Heterogeneous state sizes, exact reduced Hessian, and all-at-once agreement."""
    operating_points = (
        {
            "a": jnp.asarray((0.8,)),
            "b": jnp.asarray((0.6,)),
            "load": jnp.asarray((0.2,)),
            "target": jnp.asarray((1.1,)),
        },
        {
            "a": jnp.asarray((1.2, -0.5)),
            "b": jnp.asarray((-0.3, 0.9)),
            "load": jnp.asarray((-0.1, 0.3)),
            "target": jnp.asarray((0.7, -0.2)),
        },
    )
    shared_regularization, local_regularization = 0.5, 0.4

    def residual(state, design, args):
        return state - (
            args["a"] * design["shared"] + args["b"] * design["local"] + args["load"]
        )

    def objective(state, design, args):
        error = state - args["target"]
        return (
            0.5 * _inner(error, error)
            + 0.5 * local_regularization * design["local"] ** 2
            + 0.25 * shared_regularization * design["shared"] ** 2
        )

    child = opt.StateDesignProblem(residual, objective, acceptance_policy=_policy())
    multipoint = MultipointStateDesignProblem(
        tuple(
            StateDesignCase(
                f"operating-point-{index}",
                child,
                jnp.zeros_like(args["a"]),
                lambda shared, local, _: {"shared": shared, "local": local},
                args=args,
            )
            for index, args in enumerate(operating_points)
        ),
        problem_id="analytic-shared-local",
    )
    initial_design = {
        "shared": jnp.asarray(0.2),
        "local": (jnp.asarray(-0.1), jnp.asarray(0.3)),
    }
    hessian = jnp.diag(
        jnp.asarray((shared_regularization, local_regularization, local_regularization))
    )
    right = jnp.zeros((3,))
    for index, args in enumerate(operating_points):
        a, b, target = args["a"], args["b"], args["target"] - args["load"]
        hessian = hessian.at[0, 0].add(_inner(a, a))
        hessian = hessian.at[index + 1, index + 1].add(_inner(b, b))
        hessian = hessian.at[0, index + 1].set(_inner(a, b))
        hessian = hessian.at[index + 1, 0].set(_inner(a, b))
        right = right.at[0].add(_inner(a, target))
        right = right.at[index + 1].set(_inner(b, target))
    exact = phx.linalg.solve(
        phx.linalg.LinearSystem(phx.linalg.DenseLinearOperator(hessian)),
        right,
        policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU()),
    )
    assert int(exact.status) == int(phx.linalg.LinearSolveStatus.SUCCESS)
    problem, reduced, structured, report = _solve_pair(
        multipoint, initial_design, repeats
    )
    np.testing.assert_allclose(_vector(reduced.design), exact.value, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(
        _vector(structured.design), exact.value, atol=2e-5, rtol=2e-5
    )

    point, preparation_seconds = measure_synchronized(
        lambda: prepare_state_design_linearization(
            problem, initial_design, multipoint.initial_state
        )
    )
    derivative, derivative_time = measure_repeated(
        lambda: state_design_response_vjp(point),
        warmup=1,
        repeats=repeats,
    )
    assert bool(derivative.accepted), "Shared/local response pullback was rejected."
    expected_gradient = (
        phx.ein.contract("ij,j->i", hessian, _vector(initial_design), backend="jax")
        - right
    )
    gradient = _vector(derivative.design_cotangent)
    np.testing.assert_allclose(gradient, expected_gradient, atol=2e-7, rtol=2e-7)
    report.update(
        {
            "case": "analytic-shared-local",
            "state_sizes": [1, 2],
            "design": np.asarray(_vector(reduced.design)).tolist(),
            "analytic_design": np.asarray(exact.value).tolist(),
            "structured_design": np.asarray(_vector(structured.design)).tolist(),
            "shared_gradient_error": float(jnp.abs(gradient[0] - expected_gradient[0])),
            "local_gradient_errors": np.asarray(
                jnp.abs(gradient[1:] - expected_gradient[1:])
            ).tolist(),
            "response_preparation_seconds": preparation_seconds,
            "response_vjp_time": derivative_time.to_seconds_dict(),
            "response_adjoints": _block_adjoint_evidence(derivative.adjoint_acceptance),
        }
    )
    return report


def native_axial_multiload(repeats):
    """One shared member area, independent load states, native axial constitutive law.

    For loads 1 and 2, E=10, L=1, and mass weight 0.5, minimized
    compliance plus mass is 0.5/A + 0.5*A. The exact area is 1, with
    elongations 0.1 and 0.2. The residual evaluates native member mechanics;
    the analytic expression is used only as the independent benchmark oracle.
    """
    law = phx.applications.solid_mechanics.member_network.LinearAxialLaw()

    def constitutive(extension, area):
        return law.evaluate(
            1.0 + extension,
            jnp.asarray(1.0),
            10.0 * area,
            jnp.asarray(0.0),
            jnp.asarray(0.0),
        )

    child = opt.StateDesignProblem(
        lambda state, area, load: constitutive(state, area).axial_force - load,
        lambda state, area, load: load * state + 0.25 * area,
        state_admissibility=lambda state, area, load: constitutive(state, area).valid,
        acceptance_policy=_policy(),
        problem_id="native-axial-equilibrium",
    )
    multipoint = MultipointStateDesignProblem(
        tuple(
            StateDesignCase(
                f"axial-load-{index}",
                child,
                jnp.asarray(0.0),
                lambda shared, local, _: shared,
                args=jnp.asarray(load),
            )
            for index, load in enumerate((1.0, 2.0))
        ),
        design_bounds=opt.Bounds(0.1, 3.0),
        problem_id="native-axial-multiload",
    )
    initial_design = {"shared": jnp.asarray(0.6), "local": ((), ())}
    problem, reduced, structured, report = _solve_pair(
        multipoint, initial_design, repeats
    )
    np.testing.assert_allclose(reduced.design["shared"], 1.0, atol=3e-5)
    np.testing.assert_allclose(structured.design["shared"], 1.0, atol=3e-5)
    np.testing.assert_allclose(jnp.stack(reduced.state), (0.1, 0.2), atol=3e-6)
    np.testing.assert_allclose(reduced.objective, 1.0, atol=1e-6)
    point = prepare_state_design_linearization(
        problem, initial_design, multipoint.initial_state
    )
    derivative, derivative_time = measure_repeated(
        lambda: state_design_response_vjp(point),
        warmup=1,
        repeats=repeats,
    )
    assert bool(derivative.accepted)
    expected_gradient = 0.5 - 0.5 / initial_design["shared"] ** 2
    np.testing.assert_allclose(
        derivative.design_cotangent["shared"], expected_gradient, atol=2e-7
    )
    report.update(
        {
            "case": "native-axial-multiload",
            "area": float(reduced.design["shared"]),
            "structured_area": float(structured.design["shared"]),
            "elongations": [float(value) for value in reduced.state],
            "analytic_area": 1.0,
            "shared_gradient_error": float(
                jnp.abs(derivative.design_cotangent["shared"] - expected_gradient)
            ),
            "response_vjp_time": derivative_time.to_seconds_dict(),
        }
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("analytic", "native", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    jax.config.update("jax_enable_x64", True)
    rows = []
    if args.case in ("analytic", "all"):
        rows.append(analytic_shared_local(args.repeats))
    if args.case in ("native", "all"):
        rows.append(native_axial_multiload(args.repeats))
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "jax_version": jax.__version__,
                "devices": [str(device) for device in jax.devices()],
                "results": rows,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
