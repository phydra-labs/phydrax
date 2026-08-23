#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx


def _block(tree: Any) -> Any:
    return jax.tree.map(jax.block_until_ready, tree)


def _output_bytes(tree: Any) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if isinstance(leaf, jax.Array)
    )


def _benchmark(
    name: str,
    function: Callable[[jax.Array], Any],
    argument: jax.Array,
    /,
    *,
    repeats: int,
) -> dict[str, Any]:
    compiled = jax.jit(function)
    started = time.perf_counter()
    output = _block(compiled(argument))
    first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        output = _block(compiled(argument))
    steady_seconds = (time.perf_counter() - started) / repeats
    return {
        "name": name,
        "input_shape": tuple(argument.shape),
        "compile_and_first_seconds": first_seconds,
        "steady_seconds": steady_seconds,
        "output_bytes": _output_bytes(output),
    }


def _metric_jet_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"metric-{dimension}", tuple(f"q{index}" for index in range(dimension))
    )
    metric = phx.metrix.diagonal_metric(
        lambda q: 1.0 + q**2,
        chart=chart,
    )
    jet = phx.metrix.metric_jet(metric, points, order=2)
    return jet.matrix, jet.inverse, jet.first_derivative, jet.second_derivative


def _form_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"forms-{dimension}", tuple(f"q{index}" for index in range(dimension))
    )
    metric = phx.metrix.RiemannianMetric(lambda q: jnp.eye(dimension), chart=chart)
    form = phx.metrix.DifferentialForm(
        lambda q: q,
        chart=chart,
        degree=1,
    )
    exterior = phx.metrix.exterior_derivative(form)
    dual = phx.metrix.hodge_star(form, metric)
    return exterior(points), dual(points)


def _poisson_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"phase-{dimension}", tuple(f"z{index}" for index in range(dimension))
    )
    symplectic = phx.metrix.canonical_symplectic_form(chart)
    poisson = phx.metrix.symplectic_to_poisson(symplectic)

    def hamiltonian(point):
        return 0.5 * jnp.dot(point, point)

    return phx.metrix.hamiltonian_vector_field(hamiltonian, poisson, points)


def _horizontal_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"horizontal-{dimension}", tuple(f"q{index}" for index in range(dimension))
    )
    rank = dimension - 1
    cometric = phx.metrix.HorizontalCometric(
        lambda q: jnp.eye(dimension)[:, :rank],
        chart,
        rank,
    )

    def field(point):
        return jnp.dot(point, point)

    return phx.metrix.sub_laplacian(field, cometric, points)


def _lorentzian_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"spacetime-{dimension}", tuple(f"q{index}" for index in range(dimension))
    )
    metric = phx.metrix.minkowski_metric(chart)

    def field(point):
        return -(point[0] ** 2) + jnp.sum(point[1:] ** 2)

    return phx.metrix.dalembertian(field, metric, points)


def _map_geometry_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"map-{dimension}", tuple(f"q{index}" for index in range(dimension))
    )
    metric = phx.metrix.euclidean_metric(chart)
    map = phx.metrix.DifferentiableMap(chart, chart, lambda q: q + 0.05 * q**2)
    geometry = phx.metrix.RiemannianMapGeometry(map, metric, metric)
    return geometry.energy_density(points), geometry.tension_field(points)


def _weighted_measure_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"measure-{dimension}", tuple(f"q{index}" for index in range(dimension))
    )
    measure = phx.metrix.WeightedRiemannianMeasure(
        phx.metrix.euclidean_metric(chart),
        lambda q: -0.5 * jnp.dot(q, q),
    )
    return measure.laplacian(lambda q: jnp.dot(q, q), points)


def _kahler_case(points: jax.Array):
    dimension = points.shape[-1]
    chart = phx.metrix.CoordinateChart(
        f"kahler-{dimension}", tuple(f"q{index}" for index in range(dimension))
    )
    convention = phx.metrix.ComplexCoordinateConvention(chart)
    structure = phx.metrix.KahlerStructure(
        phx.metrix.HermitianStructure(
            phx.metrix.euclidean_metric(chart),
            phx.metrix.standard_complex_structure(convention),
        )
    )
    report = phx.metrix.validate_kahler_structure(
        structure,
        points,
        raise_on_error=False,
    )
    return (
        report.compatibility_residual,
        report.closure_residual,
        report.covariant_complex_residual,
    )


def run_benchmarks(
    *,
    batch_size: int = 256,
    dimension: int = 4,
    repeats: int = 10,
) -> dict[str, Any]:
    """Benchmark representative geometric kernels without external comparisons."""
    if batch_size <= 0 or repeats <= 0:
        raise ValueError("batch_size and repeats must be positive.")
    if dimension < 2:
        raise ValueError("dimension must be at least two.")
    even_dimension = dimension if dimension % 2 == 0 else dimension + 1
    points = jnp.linspace(0.1, 0.8, batch_size * dimension).reshape(batch_size, dimension)
    phase_points = jnp.linspace(0.1, 0.8, batch_size * even_dimension).reshape(
        batch_size, even_dimension
    )
    records = [
        _benchmark("metric_jet", _metric_jet_case, points, repeats=repeats),
        _benchmark("differential_forms", _form_case, points, repeats=repeats),
        _benchmark("riemannian_map", _map_geometry_case, points, repeats=repeats),
        _benchmark("weighted_measure", _weighted_measure_case, points, repeats=repeats),
        _benchmark("horizontal_sub_laplacian", _horizontal_case, points, repeats=repeats),
        _benchmark("lorentzian_dalembertian", _lorentzian_case, points, repeats=repeats),
        _benchmark("poisson_hamiltonian", _poisson_case, phase_points, repeats=repeats),
        _benchmark("kahler_validation", _kahler_case, phase_points, repeats=repeats),
    ]
    return {
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "batch_size": batch_size,
        "dimension": dimension,
        "repeats": repeats,
        "records": records,
    }


def _csg_setup(grid_size: int):
    Sphere = phx.geometry.analytic.Sphere
    target_source = Sphere(
        (-0.45, 0.0, 0.0),
        0.75,
        feature_id="left",
    ) | Sphere((0.45, 0.0, 0.0), 0.75, feature_id="right")
    sharp_source = Sphere(
        (-0.85, 0.0, 0.0),
        0.75,
        feature_id="left",
    ) | Sphere((0.8, 0.0, 0.0), 0.75, feature_id="right")
    blend_source = phx.geometry.analytic.BlendUnion(
        Sphere((-0.85, 0.0, 0.0), 0.75, feature_id="left"),
        Sphere((0.8, 0.0, 0.0), 0.75, feature_id="right"),
        width=0.3,
    )
    target = target_source.compile()
    sharp = sharp_source.compile()
    blend = blend_source.compile()
    left_id = phx.geometry.ParameterId("left", "center")
    right_id = phx.geometry.ParameterId("right", "center")
    width_id = phx.geometry.ParameterId(blend_source.feature_id, "width")
    sharp_indices = (
        sharp.schema.index(left_id),
        sharp.schema.index(right_id),
    )
    blend_indices = (
        blend.schema.index(left_id),
        blend.schema.index(right_id),
        blend.schema.index(width_id),
    )

    axis = jnp.linspace(-1.4, 1.4, grid_size)
    xx, yy = jnp.meshgrid(axis, axis, indexing="ij")
    training_points = jnp.stack(
        (xx.reshape((-1,)), yy.reshape((-1,)), jnp.zeros((grid_size**2,))),
        axis=-1,
    )
    target_training_field = target.boundary_field(training_points)

    boundary_count = max(64, 4 * grid_size)
    index = jnp.arange(boundary_count, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / boundary_count
    angle = jnp.pi * (3.0 - jnp.sqrt(5.0)) * index
    radius = jnp.sqrt(jnp.maximum(1.0 - z**2, 0.0))
    directions = jnp.stack(
        (radius * jnp.cos(angle), radius * jnp.sin(angle), z),
        axis=-1,
    )
    target_centers = jnp.asarray([[-0.45, 0.0, 0.0], [0.45, 0.0, 0.0]])
    boundary_points = (
        target_centers[:, None, :] + 0.75 * directions[None, :, :]
    ).reshape((-1, 3))
    other_center = jnp.repeat(target_centers[::-1], boundary_count, axis=0)
    boundary_mask = (
        jnp.linalg.norm(boundary_points - other_center, axis=-1) >= 0.75 - 1e-10
    )

    pde_points = jnp.asarray(
        [
            [-1.1, 0.15, 0.1],
            [-0.8, -0.25, 0.2],
            [-0.55, 0.3, -0.15],
            [-0.3, -0.45, 0.1],
            [0.3, 0.4, -0.1],
            [0.55, -0.3, 0.15],
            [0.8, 0.2, -0.2],
            [1.1, -0.1, 0.1],
        ]
    )

    def state_with_centers(compiled, indices, parameters, width=None):
        left = jnp.stack((parameters[0], jnp.asarray(0.0), jnp.asarray(0.0)))
        right = jnp.stack((parameters[1], jnp.asarray(0.0), jnp.asarray(0.0)))
        state = compiled.state.replace_at(indices[0], left)
        state = state.replace_at(indices[1], right)
        if width is not None:
            state = state.replace_at(indices[2], width)
        return state

    def sharp_field(parameters, points):
        state = state_with_centers(sharp, sharp_indices, parameters)
        return sharp.kernel.boundary_field(state, points)

    def blend_field(parameters, width, points):
        state = state_with_centers(blend, blend_indices, parameters, width)
        return blend.kernel.boundary_field(state, points)

    def target_field(point):
        return target.kernel.boundary_field(target.state, point)

    def target_trial(point):
        value = target_field(point)
        return value**2

    target_forcing = jax.vmap(lambda point: jnp.trace(jax.hessian(target_trial)(point)))(
        pde_points
    )
    return {
        "target": target,
        "sharp_field": sharp_field,
        "blend_field": blend_field,
        "training_points": training_points,
        "target_training_field": target_training_field,
        "boundary_points": boundary_points,
        "boundary_mask": boundary_mask,
        "pde_points": pde_points,
        "target_forcing": target_forcing,
        "target_parameters": jnp.asarray([-0.45, 0.45]),
    }


def _csg_loss(setup, strategy: str):
    def loss(parameters, width):
        if strategy == "sharp":
            field = setup["sharp_field"](parameters, setup["training_points"])
        else:
            field = setup["blend_field"](
                parameters,
                width,
                setup["training_points"],
            )
        residual = field - setup["target_training_field"]
        return jnp.mean(residual**2) + 1e-5 * jnp.sum(parameters**2)

    return loss


def _fit_csg(setup, strategy: str, seed: int, steps: int):
    key = jax.random.key(seed)
    parameters = jnp.asarray([-0.85, 0.8]) + 0.08 * jax.random.normal(key, (2,))
    state = (
        parameters,
        jnp.zeros_like(parameters),
        jnp.zeros_like(parameters),
        jnp.asarray(0, dtype=jnp.int32),
    )
    loss = _csg_loss(setup, strategy)

    def step(state_, width):
        parameters_, first_moment, second_moment, iteration = state_
        value, gradient = jax.value_and_grad(loss)(parameters_, width)
        iteration = iteration + 1
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        corrected_first = first_moment / (1.0 - 0.9**iteration)
        corrected_second = second_moment / (1.0 - 0.999**iteration)
        parameters_ = parameters_ - 0.04 * corrected_first / (
            jnp.sqrt(corrected_second) + 1e-8
        )
        return (
            (
                parameters_,
                first_moment,
                second_moment,
                iteration,
            ),
            value,
            gradient,
        )

    if strategy == "sharp":
        widths = jnp.zeros((steps,))
    elif strategy == "fixed":
        widths = jnp.full((steps,), 0.3)
    else:
        widths = jnp.geomspace(0.4, 0.02, steps)
    compiled = jax.jit(step)
    started = time.perf_counter()
    state, value, gradient = _block(compiled(state, widths[0]))
    compile_and_first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for width in widths[1:]:
        state, value, gradient = _block(compiled(state, width))
    remaining_seconds = time.perf_counter() - started
    return {
        "parameters": state[0],
        "final_training_loss": value,
        "final_gradient_norm": jnp.linalg.norm(gradient),
        "compile_and_first_seconds": compile_and_first_seconds,
        "remaining_training_seconds": remaining_seconds,
        "final_width": widths[-1],
    }


def _csg_metric_functions(setup, geometry: str, width: jax.Array):
    if geometry == "sharp":
        return lambda parameters, points: setup["sharp_field"](parameters, points)
    return lambda parameters, points: setup["blend_field"](parameters, width, points)


def _csg_record(setup, strategy: str, seed: int, fit, *, geometry: str):
    width = fit["final_width"]
    evaluated = _block(
        _csg_record_metrics(
            setup,
            fit["parameters"],
            geometry=geometry,
            width=width,
        )
    )
    if geometry == "sharp":
        final_sharp = evaluated
    else:
        final_sharp = _block(
            _csg_record_metrics(
                setup,
                fit["parameters"],
                geometry="sharp",
                width=width,
            )
        )
    success = (final_sharp["parameter_recovery_error"] < 0.08) & (
        final_sharp["containment_error"] < 0.02
    )
    return {
        "strategy": strategy,
        "evaluation_geometry": geometry,
        "seed": seed,
        "parameters": [float(value) for value in fit["parameters"]],
        "final_width": float(width),
        "final_training_loss": float(fit["final_training_loss"]),
        "final_gradient_norm": float(fit["final_gradient_norm"]),
        "compile_and_first_seconds": fit["compile_and_first_seconds"],
        "remaining_training_seconds": fit["remaining_training_seconds"],
        "metrics": {name: float(value) for name, value in evaluated.items()},
        "final_sharp_metrics": {
            name: float(value) for name, value in final_sharp.items()
        },
        "success": bool(success),
    }


def _csg_record_metrics(setup, parameters, *, geometry: str, width: jax.Array):
    field_for = _csg_metric_functions(setup, geometry, width)

    def field(points):
        return field_for(parameters, points)

    mask = setup["boundary_mask"]
    count = jnp.maximum(jnp.sum(mask), 1)
    boundary_values = field(setup["boundary_points"])
    zero_set = jnp.sum(jnp.where(mask, jnp.abs(boundary_values), 0.0)) / count
    boundary_trial = boundary_values * (
        1.0 + 0.2 * jnp.sum(setup["boundary_points"] ** 2, axis=-1)
    )
    boundary_error = jnp.sum(jnp.where(mask, jnp.abs(boundary_trial), 0.0)) / count
    containment = jnp.mean(
        (field(setup["training_points"]) <= 0.0)
        != (setup["target_training_field"] <= 0.0)
    )

    def trial(point):
        return field(point) ** 2

    forcing = jax.vmap(lambda point: jnp.trace(jax.hessian(trial)(point)))(
        setup["pde_points"]
    )
    pde_residual = forcing - setup["target_forcing"]
    target = setup["target_parameters"]
    parameter_error = jnp.minimum(
        jnp.linalg.norm(parameters - target),
        jnp.linalg.norm(parameters - target[::-1]),
    )
    delta = 1e-5
    switch = 0.5 * (parameters[0] + parameters[1])
    transverse = jnp.linspace(-0.5, 0.5, 9)
    left_points = jnp.stack(
        (
            jnp.full_like(transverse, switch - delta),
            transverse,
            jnp.zeros_like(transverse),
        ),
        axis=-1,
    )
    right_points = left_points.at[:, 0].set(switch + delta)
    left_jacobian = jax.jacfwd(lambda candidate: field_for(candidate, left_points))(
        parameters
    )
    right_jacobian = jax.jacfwd(lambda candidate: field_for(candidate, right_points))(
        parameters
    )
    switch_jacobian = jnp.concatenate((left_jacobian, right_jacobian), axis=0)
    singular_values = jnp.linalg.svd(switch_jacobian, compute_uv=False)
    condition = singular_values[0] / jnp.maximum(
        singular_values[-1],
        jnp.finfo(singular_values.dtype).eps,
    )
    jump = jnp.linalg.norm(right_jacobian - left_jacobian) / jnp.sqrt(right_jacobian.size)
    return {
        "sharp_zero_set_error": zero_set,
        "containment_error": containment,
        "boundary_condition_error": boundary_error,
        "pde_residual_l2": jnp.sqrt(jnp.mean(pde_residual**2)),
        "pde_residual_linf": jnp.max(jnp.abs(pde_residual)),
        "parameter_recovery_error": parameter_error,
        "switch_gradient_condition": condition,
        "switch_gradient_jump": jump,
    }


def run_csg_continuation(
    *,
    steps: int = 100,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    grid_size: int = 31,
) -> dict[str, Any]:
    """Compare sharp fitting with fixed and annealed smooth CSG continuation."""
    if steps < 1 or grid_size < 5 or not seeds:
        raise ValueError("steps, grid_size, and seeds must define nonempty work.")
    setup = _csg_setup(grid_size)
    records = []
    for seed in seeds:
        sharp = _fit_csg(setup, "sharp", seed, steps)
        fixed = _fit_csg(setup, "fixed", seed, steps)
        annealed = _fit_csg(setup, "annealed", seed, steps)
        records.extend(
            (
                _csg_record(
                    setup,
                    "sharp-csg",
                    seed,
                    sharp,
                    geometry="sharp",
                ),
                _csg_record(
                    setup,
                    "fixed-width-blend-csg",
                    seed,
                    fixed,
                    geometry="blend",
                ),
                _csg_record(
                    setup,
                    "width-annealed-blend-csg",
                    seed,
                    annealed,
                    geometry="blend",
                ),
                _csg_record(
                    setup,
                    "annealed-sharp-final",
                    seed,
                    annealed,
                    geometry="sharp",
                ),
            )
        )
    success_by_strategy = {
        strategy: sum(
            record["success"] for record in records if record["strategy"] == strategy
        )
        for strategy in (
            "sharp-csg",
            "fixed-width-blend-csg",
            "width-annealed-blend-csg",
            "annealed-sharp-final",
        )
    }
    return {
        "schema": "phydrax.csg-continuation.v1",
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "steps": steps,
        "seeds": list(seeds),
        "grid_size": grid_size,
        "success_by_strategy": success_by_strategy,
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Phydrax differentiable-geometric kernels."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--csg-continuation", action="store_true")
    parser.add_argument("--csg-steps", type=int, default=100)
    parser.add_argument("--csg-seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--csg-grid-size", type=int, default=31)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.csg_continuation:
        report = run_csg_continuation(
            steps=3 if arguments.smoke else arguments.csg_steps,
            seeds=(0,) if arguments.smoke else tuple(arguments.csg_seeds),
            grid_size=7 if arguments.smoke else arguments.csg_grid_size,
        )
    else:
        report = run_benchmarks(
            batch_size=8 if arguments.smoke else arguments.batch_size,
            dimension=4 if arguments.smoke else arguments.dimension,
            repeats=1 if arguments.smoke else arguments.repeats,
        )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
