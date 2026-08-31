#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import synchronize


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
    optimizer_name: str = "riemannian_sgd",
) -> dict[str, Any]:
    if optimizer_name == "riemannian_sgd":
        optimizer = phx.optim.riemannian_sgd(geometry, learning_rate=0.05)
    elif optimizer_name == "riemannian_adam":
        optimizer = phx.optim.riemannian_adam(
            geometry,
            learning_rate=0.03,
            first_moment_decay=0.8,
            second_moment_decay=0.9,
            amsgrad=True,
        )
    else:
        raise ValueError(f"Unknown benchmark optimizer {optimizer_name!r}.")
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
    parameters, state, _ = synchronize(compiled_step(parameters, state))
    first_step_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for _ in range(int(repeats)):
        parameters, state, _ = compiled_step(parameters, state)
    parameters, state = synchronize((parameters, state))
    steady_step_seconds = (time.perf_counter() - started) / int(repeats)

    metrics = optimizer.step_metrics(state)
    record = {
        "name": name,
        "optimizer": optimizer_name,
        "first_step_seconds": first_step_seconds,
        "steady_step_seconds": steady_step_seconds,
        "output_bytes": _tree_bytes((parameters, state)),
        "parameter_count": _parameter_count(parameters),
        "num_manifold_leaves": geometry.num_manifold_leaves,
        "initial_objective": initial_objective,
        "final_objective": float(objective(parameters)),
        "gradient_norm": float(metrics.gradient_norm),
        "tangent_step_norm": float(metrics.tangent_step_norm),
        "adaptive_denominator_minimum": float(metrics.adaptive_denominator_minimum),
        "adaptive_denominator_maximum": float(metrics.adaptive_denominator_maximum),
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
    point, state = synchronize(compiled_step(point, state))
    first_step_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        point, state = compiled_step(point, state)
    point, state = synchronize((point, state))
    steady_step_seconds = (time.perf_counter() - started) / repeats

    tangent = geometry.project_tangent(point, jnp.array([0.1, -0.2, 0.3]))
    tangent_step = 1e-2 * tangent
    destination = geometry.retract(point, tangent_step)
    compiled_transport = eqx.filter_jit(geometry.transport)
    started = time.perf_counter()
    transported = synchronize(
        compiled_transport(point, tangent_step, destination, tangent)
    )
    transport_first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        transported = compiled_transport(point, tangent_step, destination, tangent)
    synchronize(transported)
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


def _qualification_cases():
    sphere_initial = jnp.array([1.0, 0.0, 0.0])
    sphere_target = jnp.array([0.0, 1.0, 0.0])

    qualification_angle = jnp.asarray(0.4)
    stiefel_initial = jnp.eye(3, 2)
    stiefel_target = jnp.array(
        [
            [jnp.cos(qualification_angle), -jnp.sin(qualification_angle)],
            [jnp.sin(qualification_angle), jnp.cos(qualification_angle)],
            [0.0, 0.0],
        ],
    )

    grassmann_initial = jnp.eye(3, 2)
    grassmann_target = jnp.array(
        [
            [1.0, 0.0],
            [0.0, jnp.cos(qualification_angle)],
            [0.0, jnp.sin(qualification_angle)],
        ],
    )
    grassmann_projector = grassmann_target @ grassmann_target.T

    oblique_initial = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
    )
    oblique_target = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )

    fixed_rank_initial = jnp.diag(jnp.array([1.0, 0.0, 0.0]))
    fixed_rank_target = jnp.diag(jnp.array([2.0, 0.0, 0.0]))

    angle = jnp.asarray(0.35)
    rotation_target = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle), 0.0],
            [jnp.sin(angle), jnp.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    spd_initial = jnp.eye(2)
    spd_target = jnp.array([[1.5, 0.1], [0.1, 0.8]])

    poincare_initial = jnp.zeros((2,))
    poincare_target = jnp.array([0.2, -0.1])

    hyperboloid_initial = jnp.array([1.0, 0.0, 0.0])
    hyperboloid_spatial_target = jnp.array([0.2, -0.1])
    hyperboloid_target = jnp.concatenate(
        (
            jnp.sqrt(1.0 + jnp.sum(hyperboloid_spatial_target**2))[None],
            hyperboloid_spatial_target,
        )
    )

    simplex_initial = jnp.ones((3,)) / 3.0
    simplex_target = jnp.array([0.6, 0.3, 0.1])

    return (
        (
            "sphere",
            sphere_initial,
            phx.metrix.SphereManifold(3),
            lambda point: jnp.sum((point - sphere_target) ** 2),
        ),
        (
            "stiefel",
            stiefel_initial,
            phx.metrix.StiefelManifold(3, 2),
            lambda point: jnp.sum((point - stiefel_target) ** 2),
        ),
        (
            "grassmann",
            grassmann_initial,
            phx.metrix.GrassmannManifold(3, 2),
            lambda point: jnp.sum(
                (point @ jnp.swapaxes(point, -1, -2) - grassmann_projector) ** 2
            ),
        ),
        (
            "oblique",
            oblique_initial,
            phx.metrix.ObliqueManifold(3, 2),
            lambda point: jnp.sum((point - oblique_target) ** 2),
        ),
        (
            "fixed_rank",
            fixed_rank_initial,
            phx.metrix.FixedRankManifold(3, 3, 1),
            lambda point: jnp.sum((point - fixed_rank_target) ** 2),
        ),
        (
            "special_orthogonal",
            jnp.eye(3),
            phx.metrix.SpecialOrthogonalManifold(3),
            lambda point: jnp.sum((point - rotation_target) ** 2),
        ),
        (
            "affine_invariant_spd",
            spd_initial,
            phx.metrix.AffineInvariantSPDManifold(2),
            lambda point: jnp.sum((point - spd_target) ** 2),
        ),
        (
            "poincare_ball",
            poincare_initial,
            phx.metrix.PoincareBallManifold(2),
            lambda point: jnp.sum((point - poincare_target) ** 2),
        ),
        (
            "hyperboloid",
            hyperboloid_initial,
            phx.metrix.HyperboloidManifold(2),
            lambda point: jnp.sum((point - hyperboloid_target) ** 2),
        ),
        (
            "probability_simplex",
            simplex_initial,
            phx.metrix.ProbabilitySimplexManifold(3),
            lambda point: jnp.sum(
                simplex_target * (jnp.log(simplex_target) - jnp.log(point))
            ),
        ),
    )


def _qualification_case(
    geometry_name: str,
    initial: jax.Array,
    manifold: phx.metrix.AbstractRiemannianManifold,
    objective: Callable[[jax.Array], jax.Array],
    optimizer_name: str,
    /,
    *,
    steps: int,
) -> dict[str, Any]:
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        initial,
        {"<root>": manifold},
    )
    optimizer = cast(
        Any,
        phx.optim.riemannian_conjugate_gradient(geometry)
        if optimizer_name == "riemannian_conjugate_gradient"
        else phx.optim.riemannian_lbfgs(geometry, history_size=5),
    )
    state = optimizer.init(initial)
    value_and_grad = jax.value_and_grad(objective)
    point = initial
    accepted_count = 0
    restart_count = 0
    pair_count = 0
    for _ in range(steps):
        value, gradient = value_and_grad(point)
        point, state = optimizer.update(
            gradient,
            state,
            point,
            value=value,
            value_fn=objective,
        )
        point, state = synchronize((point, state))
        accepted_count += int(state.line_search_accepted)
        restart_count += int(state.restarted)
        if optimizer_name == "riemannian_lbfgs":
            pair_count += int(state.pair_accepted)
    metrics = optimizer.step_metrics(state)
    return {
        "geometry": geometry_name,
        "optimizer": optimizer_name,
        "initial_objective": float(objective(initial)),
        "final_objective": float(objective(point)),
        "constraint_residual_max": float(geometry.maximum_constraint_residual(point)),
        "gradient_norm": float(metrics.gradient_norm),
        "accepted_step_count": accepted_count,
        "restart_count": restart_count,
        "accepted_pair_count": pair_count,
    }


def run_qualification_benchmarks(*, steps: int = 12) -> dict[str, Any]:
    """Qualify line-search optimizers across every built-in manifold family."""
    count = int(steps)
    if count <= 0:
        raise ValueError("steps must be positive.")
    records = [
        _qualification_case(
            geometry_name,
            initial,
            manifold,
            objective,
            optimizer_name,
            steps=count,
        )
        for geometry_name, initial, manifold, objective in _qualification_cases()
        for optimizer_name in (
            "riemannian_conjugate_gradient",
            "riemannian_lbfgs",
        )
    ]
    for record in records:
        if record["final_objective"] >= record["initial_objective"]:
            raise RuntimeError(
                f"{record['optimizer']} did not improve {record['geometry']}."
            )
        if record["constraint_residual_max"] > 1e-6:
            raise RuntimeError(
                f"{record['optimizer']} violated {record['geometry']} constraints."
            )
        if record["accepted_step_count"] == 0:
            raise RuntimeError(
                f"{record['optimizer']} accepted no {record['geometry']} step."
            )
    return {"steps": count, "records": records}


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
    adaptive_records = [
        _benchmark_case(
            name,
            parameters,
            geometry,
            objective,
            diagnostics,
            repeats=count,
            optimizer_name="riemannian_adam",
        )
        for name, parameters, geometry, objective, diagnostics in _cases()
    ]
    return {
        "repeats": count,
        "records": records,
        "line_search_records": _line_search_cases(count),
        "adaptive_records": adaptive_records,
    }


def run_smoke_benchmarks() -> dict[str, Any]:
    """Execute every benchmark once and enforce basic progress and invariants."""
    report = run_benchmarks(repeats=1)
    for record in (
        report["records"] + report["adaptive_records"] + report["line_search_records"]
    ):
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
    parser.add_argument("--qualification", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.smoke:
        report = run_smoke_benchmarks()
    elif arguments.qualification:
        report = run_qualification_benchmarks(steps=arguments.repeats)
    else:
        report = run_benchmarks(repeats=arguments.repeats)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
