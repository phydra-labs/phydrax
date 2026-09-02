#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
import phydrax.ein as ein


def _weighted_moments(solution):
    mask = solution.mask[-1]
    weights = jnp.where(mask, jnp.exp(solution.log_weights[-1]), 0.0)
    points = solution.points[-1]
    mean = ein.contract("p,pi->i", weights, points)
    centered = points - mean
    covariance = ein.contract("p,pi,pj->ij", weights, centered, centered)
    return mean, covariance


def _case(state_dimension, intervals, repeats):
    rate = 0.15
    diffusion_scale = 0.3
    initial = jnp.linspace(0.5, 1.5, state_dimension)
    problem = phx.solver.DifferentialProblem(
        lambda time, state, value: -value * state,
        initial,
        t0=0.0,
        t1=1.0,
        args=rate,
        wiener_terms=(
            phx.solver.WienerTerm(
                "isotropic-noise",
                lambda time, state, args: diffusion_scale * jnp.eye(state_dimension),
                (state_dimension,),
                structure="additive",
            ),
        ),
    )
    preparation_started = time.perf_counter()
    plan = phx.solver.MarkovCubaturePlan(
        phx.discretization.TemporalMesh.uniform(
            0.0,
            1.0,
            intervals,
            role="driver",
        ),
        phx.integration.GaussianCubatureRule(state_dimension, 3),
    )
    preparation_seconds = time.perf_counter() - preparation_started
    solve = eqx.filter_jit(lambda: phx.solver.solve_markov_cubature(problem, plan))

    compilation_started = time.perf_counter()
    solution = solve()
    jax.block_until_ready(solution.points)
    compilation_seconds = time.perf_counter() - compilation_started

    steady_started = time.perf_counter()
    for _ in range(repeats):
        solution = solve()
        jax.block_until_ready(solution.points)
    steady_seconds = (time.perf_counter() - steady_started) / repeats

    mean, covariance = _weighted_moments(solution)
    step = 1.0 / intervals
    multiplier = 1.0 - rate * step
    expected_mean = multiplier**intervals * initial
    expected_variance = (
        diffusion_scale**2
        * step
        * sum(multiplier ** (2 * power) for power in range(intervals))
    )
    expected_covariance = expected_variance * jnp.eye(state_dimension)
    return {
        "state_dimension": state_dimension,
        "noise_dimension": state_dimension,
        "intervals": intervals,
        "feature_count": solution.diagnostics.feature_count,
        "retained_capacity": solution.diagnostics.retained_capacity,
        "expanded_capacity": solution.diagnostics.expanded_capacity,
        "maximum_retained_points": int(jnp.max(solution.diagnostics.retained_points)),
        "preparation_seconds": preparation_seconds,
        "compilation_and_first_solve_seconds": compilation_seconds,
        "steady_solve_seconds": steady_seconds,
        "maximum_mean_error": float(jnp.max(jnp.abs(mean - expected_mean))),
        "maximum_covariance_error": float(
            jnp.max(jnp.abs(covariance - expected_covariance))
        ),
        "status": int(solution.status),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dimension", type=int, default=4)
    parser.add_argument("--intervals", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.state_dimension < 1:
        raise ValueError("state-dimension must be positive.")
    if arguments.intervals < 1:
        raise ValueError("intervals must be positive.")
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")

    report = {
        "markov_cubature": _case(
            arguments.state_dimension,
            arguments.intervals,
            arguments.repeats,
        )
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
