#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Executable physical target, model rejection, coupled-state and FE benchmarks."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from phydrax.optim._anchored_target import (
    AnchoredResponseModel,
    AnchoredTargetMethod,
    AnchoredTargetProblem,
    solve_anchored_target,
)


def _measure(problem, initial, method, repeats, *, execution="native"):
    def action(design):
        return solve_anchored_target(problem, design, method=method, execution=execution)

    run = eqx.filter_jit(action) if execution == "native" else action
    start = perf_counter()
    result = run(initial)
    jax.block_until_ready(result)
    first_seconds = perf_counter() - start
    start = perf_counter()
    for _ in range(repeats):
        result = run(initial)
        jax.block_until_ready(result)
    steady_seconds = (perf_counter() - start) / repeats
    return result, {
        "first_solve_seconds": first_seconds,
        "steady_solve_seconds": steady_seconds,
        "successful": bool(result.successful),
        "status": int(result.status),
        "design": result.design.tolist(),
        "responses": result.values.tolist(),
        "scaled_target_error": float(result.target_error),
        "constraint_violation": float(result.constraint_violation),
        "fine_evaluations": int(result.evaluations),
        "inner_residual_evaluations": int(result.inner_residual_evaluations),
        "accepted_steps": int(result.accepted_steps),
        "rejected_steps": int(result.rejected_steps),
    }


def exact_scale(repeats):
    scales = jnp.asarray([1.0e3, 1.0e-3])

    def predictor(design, args):
        del args
        return design, jnp.asarray(True), {"response": design}

    def fine(design, args):
        values, valid, evidence = predictor(design, args)
        return scales * values, valid, evidence

    problem = AnchoredTargetProblem(
        fine,
        AnchoredResponseModel(
            predictor,
            correction="multiplicative",
            minimum_denominator=1e-8,
        ),
        targets=scales * jnp.asarray([1.5, 2.0]),
        scales=scales,
        response_names=("force", "displacement"),
        bounds=phx.optim.Bounds(0.1, 3.0),
    )
    result, report = _measure(
        problem,
        jnp.asarray([1.0, 1.0]),
        AnchoredTargetMethod(maximum_evaluations=2, target_tolerance=1e-7),
        repeats,
    )
    report["first_trial_reduction_ratio"] = float(result.history["ratio"][0])
    report["qualified"] = bool(result.successful) and int(result.evaluations) == 2
    return report


def wrong_model(repeats):
    def fine(design, args):
        del args
        return design, jnp.asarray(True), {"design": design}

    def predictor(design, args):
        del args
        return -design, jnp.asarray(True), {"design": design}

    problem = AnchoredTargetProblem(
        fine,
        AnchoredResponseModel(predictor),
        targets=jnp.asarray([1.0]),
        response_names=("signed-load",),
        scales=1.0,
        bounds=phx.optim.Bounds(-2.0, 2.0),
    )
    result, report = _measure(
        problem,
        jnp.asarray([0.0]),
        AnchoredTargetMethod(maximum_evaluations=4),
        repeats,
    )
    report["retained_anchor"] = float(result.design[0])
    report["qualified"] = (
        not bool(result.successful)
        and int(result.evaluations) == 4
        and int(result.rejected_steps) == 3
        and float(result.design[0]) == 0.0
    )
    return report


def coupled_implicit(repeats):
    # Both state equations depend on both unknowns; the approximate model is
    # solved, not an independent algebraic response fit.
    def residual(state, design):
        return jnp.asarray(
            [
                state[0] ** 2 + 0.2 * state[1] - design[0],
                0.3 * state[0] + state[1] ** 2 - design[1],
            ]
        )

    def predictor(design, args):
        del args
        state = phx.optim.implicit_least_squares(
            residual,
            jnp.full((2,), 0.8),
            args=design,
            termination=phx.optim.OptimizationTermination(
                absolute_optimality=1e-10,
                relative_optimality=0.0,
                maximum_steps=32,
            ),
        )
        defect = jnp.max(jnp.abs(residual(state, design)))
        return state, defect <= 1e-8, {"state": state, "defect": defect}

    def fine(design, args):
        state, valid, evidence = predictor(design, args)
        return 1.4 * state, valid, evidence

    problem = AnchoredTargetProblem(
        fine,
        AnchoredResponseModel(
            predictor,
            correction="multiplicative",
            minimum_denominator=1e-8,
        ),
        targets=1.4 * jnp.asarray([1.2, 0.8]),
        scales=jnp.ones((2,)),
        response_names=("coupled-state-a", "coupled-state-b"),
        bounds=phx.optim.Bounds(0.8, 2.0),
    )
    result, report = _measure(
        problem,
        jnp.asarray([1.2, 1.3]),
        AnchoredTargetMethod(target_tolerance=1e-7, maximum_evaluations=4),
        repeats,
    )
    report["coupled_state_defect"] = float(result.model_evidence["defect"])
    report["known_design_error"] = float(
        jnp.max(jnp.abs(result.design - jnp.asarray([1.6, 1.0])))
    )
    report["qualified"] = bool(result.successful) and report["known_design_error"] < 1e-6
    return report


def _finite_element_response(cells):
    """Native P1 FE diffusion-reaction compliance with natural boundaries."""
    coordinates = jnp.linspace(0.0, 1.0, cells + 1)
    vertices = jnp.asarray([(x, y) for y in coordinates for x in coordinates])
    triangles = []
    for row in range(cells):
        for column in range(cells):
            lower = row * (cells + 1) + column
            upper = lower + cells + 1
            triangles.extend(((lower, lower + 1, upper), (lower + 1, upper + 1, upper)))
    mesh = phx.discretization.CellMesh.from_triangles(
        vertices,
        jnp.asarray(triangles, dtype=jnp.int32),
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "temperature",
            phx.discretization.lagrange_element("triangle", 1),
        ),
    ).prepare()
    source = discretization.project(
        "temperature",
        lambda points, args: (
            jnp.cos(jnp.pi * points[:, 0]) * jnp.cos(jnp.pi * points[:, 1])
        ),
    )
    diffusion = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            f"target-diffusion-{cells}",
            "temperature",
            (phx.equations.DiffusionAction("temperature"),),
        ),
        discretization,
    )
    mass = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            f"target-mass-{cells}",
            "temperature",
            (phx.equations.MassAction("temperature"),),
        ),
        discretization,
    )
    # Fixed-mesh FE assembly is preparation work, outside every target solve.
    zero = jnp.zeros_like(source)
    stiffness = jax.jacfwd(diffusion.full_residual)(zero)
    mass_matrix = jax.jacfwd(mass.full_residual)(zero)
    load = mass.full_residual(source)
    policy = phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())

    def evaluate(design, args):
        del args
        matrix = design[0] * stiffness + 0.1 * mass_matrix
        operator = phx.linalg.DenseLinearOperator(matrix)
        result = phx.linalg.solve(phx.linalg.LinearSystem(operator), load, policy=policy)
        defect = jnp.max(jnp.abs(operator.mv(result.value) - load))
        accepted = (
            jnp.all(result.diagnostics.converged)
            & jnp.all(result.diagnostics.finite)
            & (defect <= 1e-9)
            & jnp.all(jnp.isfinite(result.value))
        )
        compliance = phx.ein.contract("i,i->", load, result.value)
        return (
            jnp.reshape(compliance, (1,)),
            accepted,
            {"defect": defect, "state": result.value},
        )

    return evaluate, int(source.size)


def fine_fe(repeats, fine_cells, coarse_cells):
    start = perf_counter()
    fine, fine_dofs = _finite_element_response(fine_cells)
    approximate, approximate_dofs = _finite_element_response(coarse_cells)
    target, target_valid, _ = fine(jnp.asarray([2.0]), None)
    jax.block_until_ready(target)
    preparation_seconds = perf_counter() - start
    if not bool(target_valid):
        raise RuntimeError(
            "Reference native finite-element target failed its state acceptance."
        )
    problem = AnchoredTargetProblem(
        fine,
        AnchoredResponseModel(
            approximate,
            correction="multiplicative",
            minimum_denominator=1e-12,
        ),
        targets=target,
        scales=target,
        response_names=("thermal-compliance",),
        bounds=phx.optim.Bounds(0.2, 4.0),
    )
    result, report = _measure(
        problem,
        jnp.asarray([0.75]),
        AnchoredTargetMethod(target_tolerance=1e-6, maximum_evaluations=12),
        repeats,
    )
    report.update(
        {
            "preparation_seconds": preparation_seconds,
            "fine_degrees_of_freedom": fine_dofs,
            "approximate_degrees_of_freedom": approximate_dofs,
            "fine_state_defect": float(result.fine_evidence["defect"]),
            "approximate_state_defect": float(result.model_evidence["defect"]),
            "known_design_error": abs(float(result.design[0]) - 2.0),
            "qualified": bool(result.successful)
            and abs(float(result.design[0]) - 2.0) < 1e-4,
        }
    )
    return report


def host_beam(repeats):
    calls = []
    force, length, modulus, width = 100.0, 1.0, 70.0e9, 0.05

    def evaluate(design, args):
        del args
        # Real host-only physical evaluation: float conversion deliberately
        # forbids tracing through this evaluator (as with an external solver).
        height = float(design[0])
        calls.append(height)
        inertia = width * height**3 / 12.0
        displacement = force * length**3 / (3.0 * modulus * inertia)
        return (
            jnp.asarray([displacement]),
            jnp.asarray(height > 0),
            {"height": jnp.asarray(height)},
        )

    def approximate(design, args):
        del args
        inertia = width * design[0] ** 3 / 12.0
        displacement = force * length**3 / (3.0 * (0.8 * modulus) * inertia)
        return jnp.reshape(displacement, (1,)), jnp.asarray(True), {"height": design[0]}

    target_height = 0.03
    target = force * length**3 / (3.0 * modulus * width * target_height**3 / 12.0)
    problem = AnchoredTargetProblem(
        evaluate,
        AnchoredResponseModel(
            approximate,
            correction="multiplicative",
            minimum_denominator=1e-10,
        ),
        targets=jnp.asarray([target]),
        scales=jnp.asarray([target]),
        response_names=("tip-displacement",),
        bounds=phx.optim.Bounds(0.01, 0.06),
    )
    result, report = _measure(
        problem,
        jnp.asarray([0.02]),
        AnchoredTargetMethod(
            design_scales=0.02, maximum_evaluations=4, target_tolerance=1e-6
        ),
        repeats,
        execution="host",
    )
    report["host_calls_including_repeats"] = len(calls)
    report["qualified"] = (
        bool(result.successful)
        and abs(float(result.design[0]) - target_height) < 1e-7
        and len(calls) == (repeats + 1) * int(result.evaluations)
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "all",
            "exact-scale",
            "wrong-model",
            "coupled-implicit",
            "fine-fe",
            "host-beam",
        ),
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--fine-cells", type=int, default=6)
    parser.add_argument("--coarse-cells", type=int, default=2)
    options = parser.parse_args()
    if (
        options.repeats < 1
        or options.coarse_cells < 1
        or options.fine_cells <= options.coarse_cells
    ):
        parser.error(
            "Positive repeats/cell counts and fine-cells > coarse-cells are required."
        )
    cases = {
        "exact-scale": lambda: exact_scale(options.repeats),
        "wrong-model": lambda: wrong_model(options.repeats),
        "coupled-implicit": lambda: coupled_implicit(options.repeats),
        "fine-fe": lambda: fine_fe(
            options.repeats, options.fine_cells, options.coarse_cells
        ),
        "host-beam": lambda: host_beam(options.repeats),
    }
    selected = cases if options.case == "all" else {options.case: cases[options.case]}
    reports = {name: run() for name, run in selected.items()}
    print(json.dumps(reports, indent=2, sort_keys=True))
    if not all(report["qualified"] for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
