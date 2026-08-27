#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

import phydrax as phx


def _run(function):
    started = time.perf_counter()
    result = function()
    jax.block_until_ready(result.points[-1].state)
    elapsed = time.perf_counter() - started
    return result, elapsed


def _report(result, elapsed):
    diagnostics = result.diagnostics
    return {
        "seconds": elapsed,
        "status": int(result.status),
        "points": len(result.points),
        "attempted_steps": int(diagnostics.attempted_steps),
        "accepted_steps": int(diagnostics.accepted_steps),
        "rejected_steps": int(diagnostics.rejected_steps),
        "curvature_rejections": int(diagnostics.curvature_rejections),
        "corrector_iterations": int(diagnostics.corrector_iterations),
        "residual_evaluations": int(diagnostics.corrector_residual_evaluations),
        "jacobian_preparations": int(diagnostics.corrector_jacobian_preparations),
        "linear_solves": int(diagnostics.corrector_linear_solves),
        "linear_iterations": int(diagnostics.corrector_linear_iterations),
        "tangent_failures": int(diagnostics.tangent_failures),
        "geometry_id": result.provenance.geometry_id,
        "representation_id": result.provenance.representation_id,
    }


def _affine_case(predictor):
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, args: state - 2.0 * coordinate,
        problem_id=f"benchmark-affine-{predictor}",
    )
    return phx.continuation.continue_branch(
        problem,
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        num_steps=20,
        method=phx.continuation.NaturalParameterContinuation(
            predictor=predictor,
            initial_step=0.05,
            maximum_step=0.1,
        ),
    )


def _fold_case(tangent_update, minimum_alignment):
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, args: state**3 - state + coordinate,
        problem_id=f"benchmark-fold-{tangent_update}",
    )
    return phx.continuation.continue_branch(
        problem,
        jnp.asarray(-1.0, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        num_steps=20,
        method=phx.continuation.PseudoArclengthContinuation(
            initial_step=0.35,
            maximum_step=0.5,
            tangent_update=tangent_update,
            minimum_tangent_alignment=minimum_alignment,
        ),
    )


def _complex_case():
    public_space = phx.linalg.ArraySpace((1,), dtype=jnp.complex128)
    coordinates = phx.linalg.ComplexCartesianCoordinates(public_space)
    representation = phx.continuation.ContinuationRepresentationPolicy(
        state_coordinates=coordinates,
        residual_coordinates=coordinates,
    )
    problem = phx.continuation.ParameterContinuationProblem(
        lambda state, coordinate, args: (
            state + 0.2 * jnp.conj(state) - (1.0 + 1.0j) * coordinate
        ),
        representation=representation,
        problem_id="benchmark-complex-realification",
    )
    return phx.continuation.continue_branch(
        problem,
        jnp.asarray([0.0 + 0.0j], dtype=jnp.complex128),
        jnp.asarray(0.0, dtype=jnp.float64),
        num_steps=20,
        method=phx.continuation.NaturalParameterContinuation(
            predictor="tangent",
            initial_step=0.05,
            maximum_step=0.1,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    cases = {
        "natural_constant": lambda: _affine_case("constant"),
        "natural_tangent": lambda: _affine_case("tangent"),
        "fold_secant": lambda: _fold_case("secant", None),
        "fold_bordered": lambda: _fold_case("bordered", None),
        "fold_curvature": lambda: _fold_case("secant", 0.999),
        "complex_real_coordinates": _complex_case,
    }
    report = {}
    for name, function in cases.items():
        measurements = []
        for _ in range(arguments.repeats):
            result, elapsed = _run(function)
            measurements.append(_report(result, elapsed))
        report[name] = measurements
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
