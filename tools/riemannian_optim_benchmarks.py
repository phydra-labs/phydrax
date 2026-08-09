#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
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

import phydrax as phx


def _block(tree: Any) -> Any:
    return jax.tree.map(jax.block_until_ready, tree)


def _tree_bytes(tree: Any) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if eqx.is_array(leaf)
    )


def _parameter_count(tree: Any) -> int:
    return sum(
        int(leaf.size) for leaf in jax.tree.leaves(tree) if eqx.is_inexact_array(leaf)
    )


def _orthogonality_residual(matrix: jax.Array) -> float:
    identity = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    return float(jnp.max(jnp.abs(jnp.swapaxes(matrix, -1, -2) @ matrix - identity)))


def _benchmark_case(
    name: str,
    parameters: Any,
    geometry: phx.optim.ParameterGeometry,
    objective: Callable[[Any], jax.Array],
    diagnostics: Callable[[Any], dict[str, float]],
    /,
    *,
    repeats: int,
) -> dict[str, Any]:
    optimizer = phx.optim.riemannian_sgd(geometry, learning_rate=0.05)
    state = optimizer.init(parameters)
    value_and_grad = jax.value_and_grad(objective)

    def step(parameters_, state_):
        value, gradients = value_and_grad(parameters_)
        destination, destination_state = optimizer.update(
            gradients,
            state_,
            parameters_,
        )
        return destination, destination_state, value

    compiled_step = eqx.filter_jit(step)
    initial_objective = float(objective(parameters))
    started = time.perf_counter()
    parameters, state, _ = _block(compiled_step(parameters, state))
    first_step_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(int(repeats)):
        parameters, state, _ = compiled_step(parameters, state)
    parameters, state = _block((parameters, state))
    steady_step_seconds = (time.perf_counter() - started) / int(repeats)

    metrics = optimizer.step_metrics(state)
    record = {
        "name": name,
        "first_step_seconds": first_step_seconds,
        "steady_step_seconds": steady_step_seconds,
        "output_bytes": _tree_bytes((parameters, state)),
        "parameter_count": _parameter_count(parameters),
        "num_manifold_leaves": geometry.num_manifold_leaves,
        "initial_objective": initial_objective,
        "final_objective": float(objective(parameters)),
        "gradient_norm": float(metrics.gradient_norm),
        "tangent_step_norm": float(metrics.tangent_step_norm),
        "constraint_residual_max": float(
            geometry.maximum_constraint_residual(parameters)
        ),
        "manifold_ids": geometry.manifold_ids,
    }
    return record | diagnostics(parameters)


def _cases():
    sphere_parameters = {"point": jnp.ones((8,)) / jnp.sqrt(8.0)}
    sphere_geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        sphere_parameters,
        {"['point']": phx.metrix.SphereManifold(8)},
    )
    sphere_target = jnp.eye(8)[0]

    stiefel_seed = jnp.arange(1.0, 25.0).reshape(8, 3)
    stiefel_point, _ = jnp.linalg.qr(stiefel_seed)
    stiefel_parameters = {"point": stiefel_point}
    stiefel_geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        stiefel_parameters,
        {"['point']": phx.metrix.StiefelManifold(8, 3)},
    )
    stiefel_target = jnp.eye(8, 3)

    so_parameters = {"point": jnp.eye(3)}
    so_geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        so_parameters,
        {"['point']": phx.metrix.SpecialOrthogonalManifold(3)},
    )
    angle = jnp.asarray(0.4)
    so_target = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    spd_parameters = {"point": jnp.eye(3)}
    spd_geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        spd_parameters,
        {"['point']": phx.metrix.AffineInvariantSPDManifold(3)},
    )
    spd_target = jnp.array([[1.8, 0.1, 0.0], [0.1, 1.3, -0.05], [0.0, -0.05, 0.8]])

    mixed_parameters = {
        "offset": jnp.asarray(2.0),
        "point": jnp.array([1.0, 0.0, 0.0]),
    }
    mixed_geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        mixed_parameters,
        {"['point']": phx.metrix.SphereManifold(3)},
        weights={"['point']": 2.0},
    )
    mixed_target = jnp.array([0.0, 1.0, 0.0])

    return (
        (
            "sphere",
            sphere_parameters,
            sphere_geometry,
            lambda tree: jnp.sum((tree["point"] - sphere_target) ** 2),
            lambda tree: {
                "unit_norm_error": float(jnp.abs(jnp.linalg.norm(tree["point"]) - 1.0))
            },
        ),
        (
            "stiefel",
            stiefel_parameters,
            stiefel_geometry,
            lambda tree: jnp.sum((tree["point"] - stiefel_target) ** 2),
            lambda tree: {"orthogonality_error": _orthogonality_residual(tree["point"])},
        ),
        (
            "special_orthogonal",
            so_parameters,
            so_geometry,
            lambda tree: jnp.sum((tree["point"] - so_target) ** 2),
            lambda tree: {
                "orthogonality_error": _orthogonality_residual(tree["point"]),
                "determinant": float(jnp.linalg.det(tree["point"])),
            },
        ),
        (
            "affine_invariant_spd",
            spd_parameters,
            spd_geometry,
            lambda tree: jnp.sum((tree["point"] - spd_target) ** 2),
            lambda tree: {
                "minimum_eigenvalue": float(jnp.min(jnp.linalg.eigvalsh(tree["point"])))
            },
        ),
        (
            "mixed_pytree",
            mixed_parameters,
            mixed_geometry,
            lambda tree: (
                jnp.sum((tree["point"] - mixed_target) ** 2) + (tree["offset"] - 0.5) ** 2
            ),
            lambda tree: {
                "unit_norm_error": float(jnp.abs(jnp.linalg.norm(tree["point"]) - 1.0)),
                "offset": float(tree["offset"]),
            },
        ),
    )


def _line_search_case(
    name: str,
    optimizer_factory: Callable[[phx.optim.ParameterGeometry], Any],
    /,
    *,
    repeats: int,
) -> dict[str, Any]:
    point = jnp.array([1.0, 0.0, 0.0])
    target = jnp.array([0.0, 1.0, 0.0])
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        point,
        {"<root>": phx.metrix.SphereManifold(3)},
    )
    optimizer = optimizer_factory(geometry)
    state = optimizer.init(point)

    def objective(candidate):
        return 1.0 - jnp.dot(candidate, target)

    value_and_grad = jax.value_and_grad(objective)

    def step(point_, state_):
        value, gradient = value_and_grad(point_)
        return optimizer.update(
            gradient,
            state_,
            point_,
            value=value,
            value_fn=objective,
        )

    compiled_step = eqx.filter_jit(step)
    initial_objective = float(objective(point))
    started = time.perf_counter()
    point, state = _block(compiled_step(point, state))
    first_step_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        point, state = compiled_step(point, state)
    point, state = _block((point, state))
    steady_step_seconds = (time.perf_counter() - started) / repeats

    tangent = geometry.project_tangent(point, jnp.array([0.1, -0.2, 0.3]))
    tangent_step = 1e-2 * tangent
    destination = geometry.retract(point, tangent_step)
    compiled_transport = eqx.filter_jit(geometry.transport)
    started = time.perf_counter()
    transported = _block(compiled_transport(point, tangent_step, destination, tangent))
    transport_first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        transported = compiled_transport(point, tangent_step, destination, tangent)
    _block(transported)
    transport_steady_seconds = (time.perf_counter() - started) / repeats

    return {
        "name": name,
        "first_step_seconds": first_step_seconds,
        "steady_step_seconds": steady_step_seconds,
        "transport_first_seconds": transport_first_seconds,
        "transport_steady_seconds": transport_steady_seconds,
        "output_bytes": _tree_bytes((point, state)),
        "initial_objective": initial_objective,
        "final_objective": float(objective(point)),
        "line_search_evaluations": int(state.line_search_evaluations),
        "line_search_accepted": bool(state.line_search_accepted),
        "constraint_residual_max": float(geometry.maximum_constraint_residual(point)),
    }


def _line_search_cases(repeats: int, /) -> list[dict[str, Any]]:
    return [
        _line_search_case(
            "riemannian_conjugate_gradient",
            phx.optim.riemannian_conjugate_gradient,
            repeats=repeats,
        ),
        _line_search_case(
            "riemannian_lbfgs",
            lambda geometry: phx.optim.riemannian_lbfgs(
                geometry,
                history_size=5,
            ),
            repeats=repeats,
        ),
    ]


def run_benchmarks(*, repeats: int = 5) -> dict[str, Any]:
    """Run invariant-aware first-order Riemannian optimizer benchmarks."""
    count = int(repeats)
    if count <= 0:
        raise ValueError("repeats must be positive.")
    records = [
        _benchmark_case(
            name,
            parameters,
            geometry,
            objective,
            diagnostics,
            repeats=count,
        )
        for name, parameters, geometry, objective, diagnostics in _cases()
    ]
    return {
        "repeats": count,
        "records": records,
        "line_search_records": _line_search_cases(count),
    }


def run_smoke_benchmarks() -> dict[str, Any]:
    """Execute every benchmark once and enforce basic progress and invariants."""
    report = run_benchmarks(repeats=1)
    for record in report["records"] + report["line_search_records"]:
        if record["final_objective"] >= record["initial_objective"]:
            raise RuntimeError(f"{record['name']} objective did not decrease.")
        if record["constraint_residual_max"] > 1e-8:
            raise RuntimeError(f"{record['name']} violated its manifold constraint.")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Phydrax product-manifold first-order optimization."
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = (
        run_smoke_benchmarks()
        if arguments.smoke
        else run_benchmarks(repeats=arguments.repeats)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
