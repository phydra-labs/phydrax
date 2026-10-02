# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Reusable exact contractions and a fourth-order randomized PDE optimizer update.

Run from the repository root with ``python -m examples.taylor_derivative_contractions --smoke``.
The validation objective is analytic, fixed, and independent of training probes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

import phydrax as phx
from phydrax.equations import (
    compile_pde_randomized_term,
    PDECoordinate,
    PDEEquation,
    PDEExpression,
    PDEField,
    PDEProblemIR,
    RandomizedDifferentialPlan,
)
from phydrax.operators import (
    dimension_sum_samples,
    DimensionSamplingPolicy,
    evaluate_taylor_contractions,
    plan_taylor_contractions,
    TaylorContractionPolicy,
    TaylorContractionRequest,
    TaylorContractionResources,
)


def _prepared_contractions(dimension: int) -> dict[str, Any]:
    plan = plan_taylor_contractions(
        (TaylorContractionRequest(("v", "w"), (2, 2)),),
        policy=TaylorContractionPolicy(strategy="single"),
    )
    coefficients = jnp.linspace(0.8, 1.2, dimension, dtype=jnp.float64)
    point = jnp.linspace(0.2, 0.6, dimension, dtype=jnp.float64)
    v = jnp.full((dimension,), 0.3, dtype=jnp.float64)
    w = jnp.linspace(0.1, 0.4, dimension, dtype=jnp.float64)

    def evaluate(weights: Array, position: Array, first: Array, second: Array) -> Array:
        def field(value: Array) -> Array:
            return jnp.sum(weights * value**6)

        return evaluate_taylor_contractions(
            field, (position,), {"v": (first,), "w": (second,)}, plan
        ).values[0]

    prepared = jax.jit(evaluate)
    initial = prepared(coefficients, point, v, w)
    changed_point = prepared(coefficients, point + 0.1, v, w)
    changed_coefficients = prepared(1.1 * coefficients, point + 0.1, v, w)

    def objective(weights: Array, first: Array, second: Array) -> Array:
        return evaluate(weights, point, first, second)

    parameter_gradient, v_gradient, w_gradient = jax.grad(objective, argnums=(0, 1, 2))(
        coefficients, v, w
    )
    # Independent closed form of D_v^2 D_w^2 sum_i a_i x_i^6.
    host_a, host_x, host_v, host_w = (
        np.asarray(jax.device_get(value)) for value in (coefficients, point, v, w)
    )
    exact = 360 * np.sum(host_a * host_x**2 * host_v**2 * host_w**2)
    parameter_reference = 360 * host_x**2 * host_v**2 * host_w**2
    v_reference = 720 * host_a * host_x**2 * host_v * host_w**2
    w_reference = 720 * host_a * host_x**2 * host_v**2 * host_w
    return {
        "plan_id": plan.plan_id,
        "prepared_schedule_count": len(plan.schedules),
        "required_regularity_order": plan.required_regularity_order,
        "initial": float(initial),
        "analytic_reference": float(exact),
        "changed_point": float(changed_point),
        "changed_point_and_coefficients": float(changed_coefficients),
        "absolute_error": abs(float(initial) - float(exact)),
        "parameter_gradient": np.asarray(jax.device_get(parameter_gradient)).tolist(),
        "v_gradient": np.asarray(jax.device_get(v_gradient)).tolist(),
        "w_gradient": np.asarray(jax.device_get(w_gradient)).tolist(),
        "maximum_gradient_error": max(
            float(
                np.max(
                    np.abs(
                        np.asarray(jax.device_get(parameter_gradient))
                        - parameter_reference
                    )
                )
            ),
            float(np.max(np.abs(np.asarray(jax.device_get(v_gradient)) - v_reference))),
            float(np.max(np.abs(np.asarray(jax.device_get(w_gradient)) - w_reference))),
        ),
    }


def _randomized_pde(dimension: int, points: int, probes: int) -> dict[str, Any]:
    domain = phx.domain.HyperRectangle(
        jnp.full((dimension,), -0.5, dtype=jnp.float64),
        jnp.full((dimension,), 0.5, dtype=jnp.float64),
        label="x",
    )
    expression = PDEExpression.field("u").laplacian("x").laplacian("x")
    problem = PDEProblemIR(
        coordinates=(PDECoordinate("x", "space", size=dimension, bounds=(-0.5, 0.5)),),
        fields=(PDEField("u", coordinates=("x",)),),
        equations=(PDEEquation("biharmonic", expression, PDEExpression.constant(1.0)),),
    )
    count = min(probes, dimension * dimension)
    plan = RandomizedDifferentialPlan(
        "dimension",
        backend="jet",
        dimension_policy=DimensionSamplingPolicy(dimension * dimension, count),
        loss_mode="independent_product",
        taylor_policy=TaylorContractionPolicy(
            strategy="linear", resources=TaylorContractionResources(workset_size=2)
        ),
    )
    compiled = compile_pde_randomized_term(
        problem,
        "biharmonic",
        plan,
        component=domain.component(),
        sampling=phx.domain.PointSampling(
            points, layout=phx.domain.SampleLayout((("x",),))
        ),
        sampling_mode="fixed",
        fixed_batch_key=jax.random.key(19),
    )

    def unit_solution(position: Array) -> Array:
        return jnp.sum(position**2) ** 2 / jnp.asarray(
            8 * dimension * (dimension + 2), dtype=jnp.float64
        )

    initial_coefficient = jnp.asarray(0.2, dtype=jnp.float64)
    basis = domain.Function("x")(unit_solution)
    functions = {"u": domain.Parameter(initial_coefficient) * basis}
    batch = compiled.term.sample(key=jax.random.key(7))
    diagnostics = compiled.term.diagnostics(functions, batch=batch)

    def training_loss(coefficient: Array) -> Array:
        return compiled.term.loss(
            {"u": domain.Parameter(coefficient) * basis}, batch=batch
        )

    parameter_gradient = jax.grad(training_loss)(initial_coefficient)
    solver = phx.solver.FunctionalSolver(functions=functions, terms=(compiled.term,))
    trained = solver.solve(
        num_iter=1,
        optim=optax.adam(0.02),
        seed=7,
        jit=True,
        keep_best=False,
        log_every=0,
    )
    # The unit polynomial has bilaplacian one. Recover its trained scalar
    # coefficient at a fixed point without reading implementation-specific leaves.
    validation_point = jnp.full((dimension,), 0.4, dtype=jnp.float64)
    fitted_coefficient = trained.functions["u"].func(validation_point) / unit_solution(
        validation_point
    )
    validation_before = (initial_coefficient - 1) ** 2
    validation_after = (fitted_coefficient - 1) ** 2
    after = compiled.term.diagnostics(trained.functions, batch=batch)
    return {
        "equation": "bilaplacian(u) = 1",
        "backend": compiled.report.execution_backend,
        "population": compiled.report.population,
        "population_size": compiled.report.population_size,
        "sampling_design": diagnostics.sampling_design,
        "contraction_certificates": compiled.report.contraction_certificates,
        "num_realizations": diagnostics.num_realizations,
        "training_objective_before": float(diagnostics.objective),
        "training_objective_after": float(after.objective),
        "residual_norm": float(diagnostics.plug_in_residual_norm),
        "residual_standard_error": float(diagnostics.mean_probe_standard_error),
        "uncertainty_available": diagnostics.uncertainty_available,
        "parameter_gradient": float(parameter_gradient),
        "initial_coefficient": float(initial_coefficient),
        "updated_coefficient": float(fitted_coefficient),
        "fixed_analytic_validation_before": float(validation_before),
        "fixed_analytic_validation_after": float(validation_after),
        "finite": bool(diagnostics.finite) and bool(after.finite),
    }


def _sampling_boundaries() -> dict[str, Any]:
    def contribution(index: Array) -> Array:
        return index.astype(jnp.float64) + 1

    singleton = dimension_sum_samples(
        contribution, jax.random.key(3), DimensionSamplingPolicy(1, 1)
    )
    fpc = dimension_sum_samples(
        contribution, jax.random.key(4), DimensionSamplingPolicy(5, 3)
    )
    sparse = dimension_sum_samples(
        contribution, jax.random.key(4), DimensionSamplingPolicy(5, 1)
    )
    if sparse.uncertainty_available or not bool(jnp.isnan(sparse.standard_error)):
        raise RuntimeError("A one-of-five sample cannot identify population uncertainty.")
    scaled_values = np.asarray(jax.device_get(fpc.values))
    fpc_reference = np.sqrt(np.var(scaled_values, ddof=1) / 3 * (1 - 3 / 5))
    refusal = ""
    try:
        RandomizedDifferentialPlan(
            "dimension",
            dimension_policy=DimensionSamplingPolicy(5, 3),
            loss_mode="u_statistic",
        )
    except ValueError as error:
        refusal = str(error)
    if not refusal:
        raise RuntimeError(
            "Dependent FPC realizations unexpectedly admitted an IID U-statistic."
        )
    return {
        "exact_singleton": {
            "mean": float(singleton.mean),
            "standard_error": float(singleton.standard_error),
            "design": singleton.sampling_design,
            "uncertainty_available": singleton.uncertainty_available,
        },
        "finite_population": {
            "mean": float(fpc.mean),
            "standard_error": float(fpc.standard_error),
            "population_size": fpc.population_size,
            "sample_count": fpc.subset_size,
            "fpc_factor": 1 - fpc.subset_size / 5,
            "independent_standard_error_reference": float(fpc_reference),
            "standard_error_absolute_error": abs(
                float(fpc.standard_error) - float(fpc_reference)
            ),
        },
        "one_of_five_uncertainty_available": sparse.uncertainty_available,
        "one_of_five_standard_error": None,
        "dependent_u_statistic_refusal": refusal,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--probes", type=int, default=8)
    arguments = parser.parse_args(argv)
    dimension, points, probes = arguments.dimension, arguments.points, arguments.probes
    if arguments.smoke:
        dimension, points, probes = 2, 2, 2
    if dimension < 2 or points < 1 or probes < 2:
        raise ValueError(
            "dimension/probes must be at least two; points must be positive."
        )
    jax.config.update("jax_enable_x64", True)
    payload = {
        "dimension": dimension,
        "reused_plan": _prepared_contractions(dimension),
        "randomized_pde_and_solver": _randomized_pde(dimension, points, probes),
        "sampling_boundaries": _sampling_boundaries(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
