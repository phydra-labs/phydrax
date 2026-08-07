#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx
from phydrax._interpolation import bspline_stencil
from phydrax._trainable import partition_trainable


def _block(tree: Any) -> Any:
    return jax.tree.map(
        lambda leaf: leaf.block_until_ready() if eqx.is_array(leaf) else leaf,
        tree,
    )


def _benchmark(
    function: Callable[..., Any],
    *arguments: Any,
    repeats: int,
) -> dict[str, float]:
    compiled = eqx.filter_jit(function)
    started = time.perf_counter()
    _block(compiled(*arguments))
    compile_and_first_ms = 1e3 * (time.perf_counter() - started)

    started = time.perf_counter()
    for _ in range(repeats):
        _block(compiled(*arguments))
    steady_ms = 1e3 * (time.perf_counter() - started) / repeats
    return {
        "compile_and_first_ms": compile_and_first_ms,
        "steady_ms": steady_ms,
    }


def _parameter_count(model: phx.nn.KAN) -> int:
    trainable, _ = partition_trainable(model)
    return sum(int(leaf.size) for leaf in jax.tree.leaves(trainable))


def _basis_matrix(grid: phx.nn.BSplineGrid, query: jax.Array) -> jax.Array:

    stencil = bspline_stencil(
        grid.knots,
        query,
        degree=grid.degree,
        bounds="error",
    )
    rows = jnp.arange(query.size, dtype=jnp.int32)[:, None]
    return (
        jnp.zeros((query.size, grid.coefficient_count), dtype=query.dtype)
        .at[rows, stencil.indices]
        .add(stencil.weights)
    )


def _fit_relative_error(
    grid: phx.nn.BSplineGrid,
    target: Callable[[jax.Array], jax.Array],
) -> float:
    fitting_points = jnp.linspace(-1.0, 1.0, 2048)
    evaluation_points = jnp.linspace(-1.0, 1.0, 8192)
    fitting_basis = _basis_matrix(grid, fitting_points)
    coefficients = jnp.linalg.lstsq(fitting_basis, target(fitting_points))[0]
    residual = _basis_matrix(grid, evaluation_points) @ coefficients - target(
        evaluation_points
    )
    return float(jnp.linalg.norm(residual) / jnp.linalg.norm(target(evaluation_points)))


def _adaptation_quality_records() -> dict[str, dict[str, float | int]]:
    basis = phx.nn.BSplineEdgeBasis(degree=3, num_intervals=8)
    uniform_grid = basis.grid
    profiles = {
        "boundary_layer": (
            -1.0 + 2.0 * jnp.linspace(0.0, 1.0, 512) ** 3,
            lambda values: jnp.exp(-30.0 * (values + 1.0)),
        ),
        "narrow_gaussian": (
            jnp.clip(
                0.2 + 0.16 * jr.normal(jr.key(101), (512,)),
                -1.0,
                1.0,
            ),
            lambda values: jnp.exp(-(((values - 0.2) / 0.07) ** 2)),
        ),
        "localized_oscillation": (
            jnp.linspace(-0.45, 0.45, 512),
            lambda values: (
                jnp.sin(28.0 * values) * jnp.exp(-(((values + 0.05) / 0.32) ** 8))
            ),
        ),
    }
    records: dict[str, dict[str, float | int]] = {}
    for name, (calibration, target) in profiles.items():
        model = phx.nn.KAN(
            in_size="scalar",
            out_size="scalar",
            hidden_sizes=(),
            edge_basis=basis,
            scale_mode="none",
            skip_connection=False,
            key=jr.key(102),
        )
        started = time.perf_counter()
        adapted, report = phx.nn.adapt_kan_grids(
            model,
            calibration,
            plan=phx.nn.KANGridAdaptationPlan(
                blend=0.05,
                minimum_span=1.0e-3,
            ),
        )
        adaptation_ms = 1e3 * (time.perf_counter() - started)
        adapted_grid = adapted.layers[0].edge_basis.grid
        uniform_error = _fit_relative_error(uniform_grid, target)
        adapted_error = _fit_relative_error(adapted_grid, target)
        records[name] = {
            "coefficient_count": uniform_grid.coefficient_count,
            "adaptation_ms": adaptation_ms,
            "uniform_relative_l2": uniform_error,
            "adapted_relative_l2": adapted_error,
            "error_ratio": adapted_error / uniform_error,
            "transfer_condition": report.transfer_conditioning[0],
        }
    return records


def _sampled_fit_relative_error(
    grid: phx.nn.BSplineGrid,
    target: Callable[[jax.Array], jax.Array],
    fitting_points: jax.Array,
    evaluation_points: jax.Array,
) -> float:
    fitting_basis = _basis_matrix(grid, fitting_points)
    coefficients = jnp.linalg.lstsq(fitting_basis, target(fitting_points))[0]
    residual = _basis_matrix(grid, evaluation_points) @ coefficients - target(
        evaluation_points
    )
    return float(jnp.linalg.norm(residual) / jnp.linalg.norm(target(evaluation_points)))


def _per_input_grid_record(repeats: int) -> dict[str, Any]:
    basis = phx.nn.BSplineEdgeBasis(degree=3, num_intervals=8)
    model = phx.nn.KAN(
        in_size=2,
        out_size="scalar",
        hidden_sizes=(),
        edge_basis=basis,
        scale_mode="none",
        skip_connection=False,
        key=jr.key(110),
    )
    fitting = jnp.stack(
        (
            jnp.clip(-0.67 + 0.14 * jr.normal(jr.key(111), (1024,)), -1.0, 1.0),
            jnp.clip(0.59 + 0.16 * jr.normal(jr.key(112), (1024,)), -1.0, 1.0),
        ),
        axis=1,
    )
    evaluation = jnp.stack(
        (
            jnp.clip(-0.67 + 0.14 * jr.normal(jr.key(113), (4096,)), -1.0, 1.0),
            jnp.clip(0.59 + 0.16 * jr.normal(jr.key(114), (4096,)), -1.0, 1.0),
        ),
        axis=1,
    )
    targets = (
        lambda values: jnp.sin(34.0 * (values + 0.67)),
        lambda values: jnp.cos(31.0 * (values - 0.59)),
    )
    shared, _ = phx.nn.adapt_kan_grids(
        model,
        fitting,
        plan=phx.nn.KANGridAdaptationPlan(blend=0.0),
    )
    per_input, _ = phx.nn.adapt_kan_grids(
        model,
        fitting,
        plan=phx.nn.KANGridAdaptationPlan(blend=0.0, per_input=True),
    )
    shared_grid = shared.layers[0].edge_basis.grid
    grid_bank = per_input.layers[0].edge_basis.grid
    shared_errors = tuple(
        _sampled_fit_relative_error(
            shared_grid,
            target,
            fitting[:, index],
            evaluation[:, index],
        )
        for index, target in enumerate(targets)
    )
    per_input_errors = tuple(
        _sampled_fit_relative_error(
            grid_bank.grids[index],
            target,
            fitting[:, index],
            evaluation[:, index],
        )
        for index, target in enumerate(targets)
    )
    coefficients = jr.normal(
        jr.key(115), (1, 2, shared.layers[0].edge_basis.coefficient_count)
    )

    def evaluate_basis(edge_basis, edge_coefficients, values):
        return jax.vmap(
            lambda row: edge_basis.evaluate(
                edge_coefficients, jnp.broadcast_to(row, (1, 2))
            )
        )(values)

    shared_timing = _benchmark(
        evaluate_basis,
        shared.layers[0].edge_basis,
        coefficients,
        evaluation,
        repeats=repeats,
    )
    per_input_timing = _benchmark(
        evaluate_basis,
        per_input.layers[0].edge_basis,
        coefficients,
        evaluation,
        repeats=repeats,
    )
    shared_mean = sum(shared_errors) / len(shared_errors)
    per_input_mean = sum(per_input_errors) / len(per_input_errors)
    error_ratio = per_input_mean / shared_mean
    return {
        "coefficient_count_per_edge": shared_grid.coefficient_count,
        "shared_relative_l2": shared_mean,
        "per_input_relative_l2": per_input_mean,
        "error_ratio": error_ratio,
        "graduated": error_ratio < 0.75,
        "shared_grid_bytes": int(
            shared_grid.knots.size * shared_grid.knots.dtype.itemsize
        ),
        "per_input_grid_bytes": int(
            grid_bank.knots.size * grid_bank.knots.dtype.itemsize
        ),
        "shared_evaluation": shared_timing,
        "per_input_evaluation": per_input_timing,
    }


def _trainable_grid_record() -> dict[str, float | int | bool]:
    degree = 3
    num_intervals = 8
    calibration = jnp.linspace(-1.0, 1.0, 1024)
    target = lambda values: jnp.exp(-(((values - 0.2) / 0.065) ** 2))
    model = phx.nn.KAN(
        in_size="scalar",
        out_size="scalar",
        hidden_sizes=(),
        edge_basis=phx.nn.BSplineEdgeBasis(
            degree=degree,
            num_intervals=num_intervals,
        ),
        scale_mode="none",
        skip_connection=False,
        key=jr.key(120),
    )
    adapted, _ = phx.nn.adapt_kan_grids(
        model,
        calibration,
        plan=phx.nn.KANGridAdaptationPlan(blend=0.0),
    )
    fixed_grid = adapted.layers[0].edge_basis.grid
    trainable_grid = phx.nn.TrainableBSplineGrid.from_grid(
        fixed_grid,
        minimum_span=2.0e-3,
    )
    fitting_basis = _basis_matrix(fixed_grid, calibration)
    target_values = target(calibration)
    initial_coefficients = jnp.linalg.lstsq(fitting_basis, target_values)[0]
    optimizer = optax.adam(3.0e-2)
    parameters = (trainable_grid.raw_span_logits, initial_coefficients)
    optimizer_state = optimizer.init(parameters)

    def objective(candidate):
        logits, coefficients = candidate
        grid = eqx.tree_at(
            lambda value: value.raw_span_logits,
            trainable_grid,
            logits,
        )
        residual = _basis_matrix(grid, calibration) @ coefficients - target_values
        return jnp.mean(residual**2) + 1.0e-6 * grid.regularization()

    @jax.jit
    def step(candidate, state):
        loss, gradient = jax.value_and_grad(objective)(candidate)
        updates, new_state = optimizer.update(gradient, state, candidate)
        return optax.apply_updates(candidate, updates), new_state, loss

    started = time.perf_counter()
    for _ in range(400):
        parameters, optimizer_state, loss = step(parameters, optimizer_state)
    _block((parameters, optimizer_state, loss))
    optimization_ms = 1e3 * (time.perf_counter() - started)
    optimized_grid = eqx.tree_at(
        lambda value: value.raw_span_logits,
        trainable_grid,
        parameters[0],
    )
    evaluation = jnp.linspace(-1.0, 1.0, 8192)
    expected = target(evaluation)
    fixed_residual = (
        _basis_matrix(fixed_grid, evaluation) @ initial_coefficients - expected
    )
    optimized_residual = (
        _basis_matrix(optimized_grid, evaluation) @ parameters[1] - expected
    )
    fixed_error = float(jnp.linalg.norm(fixed_residual) / jnp.linalg.norm(expected))
    optimized_error = float(
        jnp.linalg.norm(optimized_residual) / jnp.linalg.norm(expected)
    )
    error_ratio = optimized_error / fixed_error
    return {
        "coefficient_count": fixed_grid.coefficient_count,
        "optimization_steps": 400,
        "optimization_ms": optimization_ms,
        "fixed_quantile_relative_l2": fixed_error,
        "trainable_relative_l2": optimized_error,
        "error_ratio": error_ratio,
        "minimum_live_span": float(jnp.min(optimized_grid.span_widths)),
        "graduated": error_ratio < 0.75,
    }


def _rational_edge_records() -> dict[str, Any]:
    fitting = jnp.linspace(-1.0, 1.0, 2048)
    evaluation = jnp.linspace(-1.0, 1.0, 8192)
    rational_grid = phx.nn.BSplineGrid.open_uniform(3, 3)
    rational_basis = phx.nn.RationalBSplineEdgeBasis(grid=rational_grid)
    polynomial_plan = phx.operators.BSplineInterpolationPlan(
        degree=3,
        num_intervals=8,
        mode="least_squares",
    )
    rational_plan = phx.operators.BSplineInterpolationPlan(
        degree=3,
        mode="least_squares",
    )
    profiles = {
        "near_pole_reciprocal": (
            lambda values: jnp.ones_like(values),
            lambda values: 1.0 + 0.98 * values,
        ),
        "saturating_response": (
            lambda values: values + 1.0,
            lambda values: 1.0 + 3.0 * (values + 1.0),
        ),
        "rational_wave": (
            lambda values: 2.0 * values,
            lambda values: 1.0 + values**2,
        ),
    }
    records: dict[str, Any] = {}
    approved = False
    for name, (numerator, denominator) in profiles.items():
        expected_fitting = numerator(fitting) / denominator(fitting)
        expected = numerator(evaluation) / denominator(evaluation)
        polynomial = phx.operators.fit_bspline(
            fitting,
            expected_fitting,
            plan=polynomial_plan,
        )
        numerator_coefficients = phx.operators.fit_bspline(
            fitting,
            numerator(fitting),
            plan=rational_plan,
            grid=rational_grid,
        ).coefficients
        denominator_coefficients = phx.operators.fit_bspline(
            fitting,
            denominator(fitting),
            plan=rational_plan,
            grid=rational_grid,
        ).coefficients
        centered_log_weights = jnp.log(denominator_coefficients)
        centered_log_weights = centered_log_weights - jnp.mean(centered_log_weights)
        parameters = phx.nn.RationalBSplineEdgeParameters(
            (numerator_coefficients / denominator_coefficients)[None, None, :],
            jnp.arctanh(centered_log_weights / rational_basis.maximum_log_weight)[
                None, None, :
            ],
        )
        rational_values = jax.vmap(
            lambda value: rational_basis.evaluate(parameters, jnp.asarray([[value]]))[
                0, 0
            ]
        )(evaluation)
        polynomial_error = float(
            jnp.linalg.norm(polynomial(evaluation) - expected) / jnp.linalg.norm(expected)
        )
        rational_error = float(
            jnp.linalg.norm(rational_values - expected) / jnp.linalg.norm(expected)
        )
        error_ratio = rational_error / polynomial_error
        approved = approved or (polynomial_error > 1.0e-4 and error_ratio < 0.1)
        records[name] = {
            "effective_parameter_count": 2 * rational_grid.coefficient_count - 1,
            "polynomial_parameter_count": polynomial.coefficients.size,
            "polynomial_relative_l2": polynomial_error,
            "rational_relative_l2": rational_error,
            "error_ratio": error_ratio,
        }
    records["approved"] = approved
    return records


def _adaptive_capacity_records(repeats: int) -> dict[str, Any]:
    fitting = jnp.linspace(-1.0, 1.0, 2048)
    evaluation = jnp.linspace(-1.0, 1.0, 8192)
    fixed_grid = phx.nn.BSplineGrid.open_uniform(3, 3)
    fine_grid = phx.nn.BSplineGrid.open_uniform(3, 16)
    coarse_grid = phx.nn.BSplineGrid.open_uniform(3, 1)
    targets = [
        lambda values: jnp.sin(22.0 * values),
        lambda values: jnp.cos(19.0 * values),
        *[
            lambda values, weight=weight: weight * values**3 + (1.0 - weight) * values
            for weight in jnp.linspace(0.1, 0.9, 14)
        ],
    ]

    def fit_residual(
        grid: phx.nn.BSplineGrid,
        target: Callable[[jax.Array], jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        coefficients = jnp.linalg.lstsq(
            _basis_matrix(grid, fitting),
            target(fitting),
        )[0]
        expected = target(evaluation)
        residual = _basis_matrix(grid, evaluation) @ coefficients - expected
        return jnp.sum(residual**2), jnp.sum(expected**2)

    fixed_residuals = [fit_residual(fixed_grid, target) for target in targets]
    heterogeneous_residuals = [
        fit_residual(fine_grid if index < 2 else coarse_grid, target)
        for index, target in enumerate(targets)
    ]
    fixed_error = float(
        jnp.sqrt(
            sum(residual for residual, _ in fixed_residuals)
            / sum(reference for _, reference in fixed_residuals)
        )
    )
    heterogeneous_error = float(
        jnp.sqrt(
            sum(residual for residual, _ in heterogeneous_residuals)
            / sum(reference for _, reference in heterogeneous_residuals)
        )
    )
    fixed_parameter_count = len(targets) * fixed_grid.coefficient_count
    heterogeneous_parameter_count = (
        2 * fine_grid.coefficient_count
        + (len(targets) - 2) * coarse_grid.coefficient_count
    )

    model = phx.nn.KAN(
        in_size=8,
        out_size=8,
        hidden_sizes=(),
        edge_basis=phx.nn.BSplineEdgeBasis(grid=fixed_grid),
        skip_connection=False,
        scan=True,
        key=jr.key(160),
    )
    adapted, report = phx.nn.refine_kan_edges(
        model,
        {(0, 0, 0): jnp.asarray([0.0, 1.0, 0.0])},
        budget=1,
    )
    inputs = jnp.linspace(-0.8, 0.8, 8)
    exact_error = float(jnp.max(jnp.abs(model(inputs) - adapted(inputs))))
    parameter_delta = _parameter_count(adapted) - _parameter_count(model)
    quality_ratio = heterogeneous_error / fixed_error
    return {
        "equal_budget_fixed_relative_l2": fixed_error,
        "heterogeneous_relative_l2": heterogeneous_error,
        "error_ratio": quality_ratio,
        "fixed_parameter_count": fixed_parameter_count,
        "heterogeneous_parameter_count": heterogeneous_parameter_count,
        "dense_fine_parameter_count": len(targets) * fine_grid.coefficient_count,
        "single_edge_parameter_delta": parameter_delta,
        "single_edge_block_count": len(adapted.layers[0].edge_blocks),
        "refinement_exact_max_error": exact_error,
        "refinement_report_paths": report.paths,
        "dense_forward": _benchmark(
            lambda candidate, values: candidate(values),
            model,
            inputs,
            repeats=repeats,
        ),
        "block_forward": _benchmark(
            lambda candidate, values: candidate(values),
            adapted,
            inputs,
            repeats=repeats,
        ),
        "graduated": (
            heterogeneous_parameter_count <= fixed_parameter_count
            and quality_ratio < 0.3
            and parameter_delta == 1
            and exact_error < 1.0e-12
        ),
    }


def _parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare equal-width orthogonal-polynomial and B-spline KANs."
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    return parser


def main() -> None:
    arguments = _parse().parse_args()
    coefficient_count = 11
    bases = {
        "orthogonal_degree_10": phx.nn.OrthogonalPolynomialEdgeBasis(
            degree=coefficient_count - 1
        ),
        "bspline_degree_3_intervals_8": phx.nn.BSplineEdgeBasis(
            degree=3,
            num_intervals=coefficient_count - 3,
        ),
    }
    inputs = jnp.linspace(-0.8, 0.8, 8)
    records: dict[str, Any] = {}

    for name, basis in bases.items():
        model = phx.nn.KAN(
            in_size=8,
            out_size=8,
            width_size=arguments.width,
            depth=arguments.depth,
            edge_basis=basis,
            scale_mode="none",
            skip_connection=False,
            scan=True,
            key=jr.key(0),
        )
        records[name] = {
            "coefficient_count_per_edge": basis.coefficient_count,
            "active_coefficients_per_query": (
                basis.degree + 1
                if isinstance(basis, phx.nn.BSplineEdgeBasis)
                else basis.coefficient_count
            ),
            "parameter_count": _parameter_count(model),
            "forward": _benchmark(
                lambda candidate, values: candidate(values),
                model,
                inputs,
                repeats=arguments.repeats,
            ),
            "input_jacobian": _benchmark(
                lambda candidate, values: jax.jacrev(candidate)(values),
                model,
                inputs,
                repeats=arguments.repeats,
            ),
            "input_hessian": _benchmark(
                lambda candidate, values: jax.hessian(
                    lambda argument: jnp.sum(candidate(argument))
                )(values),
                model,
                inputs,
                repeats=arguments.repeats,
            ),
            "parameter_gradient": _benchmark(
                eqx.filter_value_and_grad(
                    lambda candidate, values: jnp.sum(candidate(values) ** 2)
                ),
                model,
                inputs,
                repeats=arguments.repeats,
            ),
        }

    records["fixed_count_adaptation_gate"] = _adaptation_quality_records()
    records["per_input_grid_gate"] = _per_input_grid_record(arguments.repeats)
    records["trainable_grid_gate"] = _trainable_grid_record()
    records["rational_edge_gate"] = _rational_edge_records()
    records["adaptive_capacity_gate"] = _adaptive_capacity_records(arguments.repeats)

    print(
        json.dumps(
            {
                "backend": jax.default_backend(),
                "jax_version": jax.__version__,
                "repeats": arguments.repeats,
                "width": arguments.width,
                "depth": arguments.depth,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
