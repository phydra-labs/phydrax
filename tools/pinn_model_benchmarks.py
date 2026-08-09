#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import phydrax as phx
from phydrax._trainable import combine_trainable, partition_trainable


@dataclass(frozen=True, slots=True)
class PINNBenchmarkScenario:
    """One manufactured one-dimensional pointwise PDE benchmark."""

    name: str
    equation: Literal["poisson", "helmholtz", "allen_cahn"]
    frequency: int
    train_points: int
    evaluation_points: int
    equation_parameter: float = 1.0
    boundary_weight: float = 10.0


@dataclass(frozen=True, slots=True)
class PINNBenchmarkRecord:
    """Training and differential-error evidence for one model and seed."""

    scenario: str
    equation: str
    architecture: str
    seed: int
    width: int
    depth: int
    parameter_count: int
    target_parameter_count: int
    training_steps: int
    learning_rate: float
    initial_loss: float
    initial_residual_loss: float
    initial_boundary_loss: float
    initial_gradient_norm: float
    initial_first_derivative_std: float
    initial_second_derivative_std: float
    final_loss: float
    relative_l2: float
    relative_h1: float
    compile_seconds: float
    training_seconds: float
    inference_seconds: float
    peak_memory_bytes: int | None
    losses: tuple[float, ...]


_SCENARIOS = {
    "poisson-smooth": PINNBenchmarkScenario(
        "poisson-smooth",
        "poisson",
        frequency=1,
        train_points=64,
        evaluation_points=257,
    ),
    "helmholtz-oscillatory": PINNBenchmarkScenario(
        "helmholtz-oscillatory",
        "helmholtz",
        frequency=4,
        train_points=128,
        evaluation_points=513,
        equation_parameter=10.0,
    ),
    "allen-cahn-nonlinear": PINNBenchmarkScenario(
        "allen-cahn-nonlinear",
        "allen_cahn",
        frequency=1,
        train_points=96,
        evaluation_points=385,
        equation_parameter=0.05,
    ),
    "poisson-depth-stress": PINNBenchmarkScenario(
        "poisson-depth-stress",
        "poisson",
        frequency=3,
        train_points=96,
        evaluation_points=385,
    ),
}


def _parameter_count(model: eqx.Module, /) -> int:
    trainable, _ = partition_trainable(model)
    return sum(
        int(leaf.size)
        for leaf in jax.tree_util.tree_leaves(trainable)
        if isinstance(leaf, jax.Array)
    )


def _build_model(
    architecture: str,
    /,
    *,
    width: int,
    depth: int,
    key: jax.Array,
):
    if architecture == "mlp":
        return phx.nn.models.MLP(
            in_size=1,
            out_size="scalar",
            width_size=int(width),
            depth=int(depth),
            activation=jnp.tanh,
            rwf=False,
            key=key,
        )
    if architecture == "modified_mlp":
        return phx.nn.models.ModifiedMLP(
            in_size=1,
            out_size="scalar",
            width_size=int(width),
            depth=int(depth),
            activation=jnp.tanh,
            rwf=False,
            key=key,
        )
    if architecture == "piratenet":
        return phx.nn.models.PirateNet(
            in_size=1,
            out_size="scalar",
            width_size=int(width),
            depth=int(depth),
            activation=jnp.tanh,
            rwf=False,
            key=key,
        )
    if architecture == "siren":
        return phx.nn.models.SIREN(
            in_size=1,
            out_size="scalar",
            width_size=int(width),
            depth=int(depth),
            key=key,
        )
    raise ValueError(f"Unknown PINN benchmark architecture {architecture!r}.")


def _matched_width(
    architecture: str,
    /,
    *,
    requested_width: int,
    depth: int,
    target_parameters: int,
    key: jax.Array,
) -> int:
    maximum = max(8, 2 * int(requested_width))
    candidates = range(4, maximum + 1)
    return min(
        candidates,
        key=lambda candidate: abs(
            _parameter_count(
                _build_model(
                    architecture,
                    width=candidate,
                    depth=depth,
                    key=key,
                )
            )
            - int(target_parameters)
        ),
    )


def _manufactured_solution(points: jax.Array, frequency: int, /) -> jax.Array:
    return jnp.sin(float(frequency) * jnp.pi * points[..., 0])


def _pointwise_derivatives(model, points: jax.Array, /) -> tuple[jax.Array, ...]:
    def scalar_value(point):
        return jnp.asarray(model(point)).reshape(())

    values = jax.vmap(scalar_value)(points)
    gradients = jax.vmap(jax.grad(scalar_value))(points)[..., 0]
    hessians = jax.vmap(jax.hessian(scalar_value))(points)[..., 0, 0]
    return values, gradients, hessians


def _pde_residual(
    values: jax.Array,
    second_derivatives: jax.Array,
    points: jax.Array,
    scenario: PINNBenchmarkScenario,
    /,
) -> jax.Array:
    truth = _manufactured_solution(points, scenario.frequency)
    omega = float(scenario.frequency) * jnp.pi
    if scenario.equation == "poisson":
        forcing = omega**2 * truth
        return -second_derivatives - forcing
    if scenario.equation == "helmholtz":
        wave_number = float(scenario.equation_parameter)
        forcing = (wave_number**2 - omega**2) * truth
        return second_derivatives + wave_number**2 * values - forcing
    if scenario.equation == "allen_cahn":
        diffusivity = float(scenario.equation_parameter)
        forcing = diffusivity * omega**2 * truth + truth**3 - truth
        return -diffusivity * second_derivatives + values**3 - values - forcing
    raise ValueError(f"Unknown PINN benchmark equation {scenario.equation!r}.")


def _loss_components(
    model,
    scenario: PINNBenchmarkScenario,
    interior: jax.Array,
    boundary: jax.Array,
    /,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    values, _, second = _pointwise_derivatives(model, interior)
    residual = _pde_residual(values, second, interior, scenario)
    residual_loss = jnp.mean(residual**2)
    boundary_values = jax.vmap(model)(boundary)
    boundary_loss = jnp.mean(jnp.asarray(boundary_values) ** 2)
    total = residual_loss + float(scenario.boundary_weight) * boundary_loss
    return total, residual_loss, boundary_loss


def _tree_l2_norm(tree, /) -> jax.Array:
    leaves = tuple(
        leaf for leaf in jax.tree_util.tree_leaves(tree) if isinstance(leaf, jax.Array)
    )
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in leaves))


def _memory_bytes() -> int | None:
    statistics = jax.devices()[0].memory_stats()
    if statistics is None:
        return None
    for name in ("peak_bytes_in_use", "bytes_in_use"):
        if name in statistics:
            return int(statistics[name])
    return None


def run_pinn_model_benchmark(
    architecture: str,
    scenario: PINNBenchmarkScenario,
    /,
    *,
    seed: int,
    width: int,
    depth: int,
    steps: int,
    learning_rate: float,
    match_parameters: bool = True,
) -> PINNBenchmarkRecord:
    """Train one capacity-controlled model on a manufactured PDE residual."""
    if int(width) <= 0 or int(depth) <= 0:
        raise ValueError("width and depth must be positive.")
    if int(steps) < 0 or float(learning_rate) <= 0.0:
        raise ValueError("steps must be non-negative and learning_rate positive.")

    root_key = jr.PRNGKey(int(seed))
    baseline_key, model_key, sample_key = jr.split(root_key, 3)
    baseline = _build_model("mlp", width=width, depth=depth, key=baseline_key)
    target_parameters = _parameter_count(baseline)
    selected_width = (
        _matched_width(
            architecture,
            requested_width=width,
            depth=depth,
            target_parameters=target_parameters,
            key=model_key,
        )
        if match_parameters
        else int(width)
    )
    model = _build_model(
        architecture,
        width=selected_width,
        depth=depth,
        key=model_key,
    )

    interior = jr.uniform(
        sample_key,
        (int(scenario.train_points), 1),
        minval=-1.0,
        maxval=1.0,
    )
    boundary = jnp.asarray([[-1.0], [1.0]])
    initial_total, initial_residual, initial_boundary = _loss_components(
        model, scenario, interior, boundary
    )
    parameters, fixed = partition_trainable(model)

    def objective(candidate):
        current = combine_trainable(candidate, fixed)
        return _loss_components(current, scenario, interior, boundary)[0]

    initial_gradient = eqx.filter_grad(objective)(parameters)
    _, initial_first, initial_second = _pointwise_derivatives(model, interior)

    optimizer = optax.adam(float(learning_rate))
    optimizer_state = optimizer.init(parameters)

    @eqx.filter_jit
    def train_step(current_parameters, current_state):
        value, gradient = eqx.filter_value_and_grad(objective)(current_parameters)
        updates, next_state = optimizer.update(
            gradient, current_state, current_parameters
        )
        return eqx.apply_updates(current_parameters, updates), next_state, value

    losses: list[float] = []
    compile_seconds = 0.0
    training_started = time.perf_counter()
    if int(steps) > 0:
        first_started = time.perf_counter()
        parameters, optimizer_state, loss = train_step(parameters, optimizer_state)
        jax.block_until_ready(loss)
        compile_seconds = time.perf_counter() - first_started
        losses.append(float(loss))
        for _ in range(1, int(steps)):
            parameters, optimizer_state, loss = train_step(parameters, optimizer_state)
            jax.block_until_ready(loss)
            losses.append(float(loss))
    training_seconds = time.perf_counter() - training_started

    trained = combine_trainable(parameters, fixed)
    final_total = _loss_components(trained, scenario, interior, boundary)[0]
    evaluation = jnp.linspace(-1.0, 1.0, int(scenario.evaluation_points), dtype=float)[
        :, None
    ]

    inference_started = time.perf_counter()
    prediction, prediction_gradient, _ = _pointwise_derivatives(trained, evaluation)
    jax.block_until_ready(prediction)
    inference_seconds = time.perf_counter() - inference_started

    truth = _manufactured_solution(evaluation, scenario.frequency)
    omega = float(scenario.frequency) * jnp.pi
    truth_gradient = omega * jnp.cos(omega * evaluation[:, 0])
    relative_l2 = jnp.linalg.norm(prediction - truth) / jnp.maximum(
        jnp.linalg.norm(truth), jnp.finfo(truth.dtype).eps
    )
    h1_error = jnp.sqrt(
        jnp.sum((prediction - truth) ** 2)
        + jnp.sum((prediction_gradient - truth_gradient) ** 2)
    )
    h1_truth = jnp.sqrt(jnp.sum(truth**2) + jnp.sum(truth_gradient**2))
    relative_h1 = h1_error / jnp.maximum(h1_truth, jnp.finfo(truth.dtype).eps)

    return PINNBenchmarkRecord(
        scenario=scenario.name,
        equation=scenario.equation,
        architecture=architecture,
        seed=int(seed),
        width=int(selected_width),
        depth=int(depth),
        parameter_count=_parameter_count(model),
        target_parameter_count=int(target_parameters),
        training_steps=int(steps),
        learning_rate=float(learning_rate),
        initial_loss=float(initial_total),
        initial_residual_loss=float(initial_residual),
        initial_boundary_loss=float(initial_boundary),
        initial_gradient_norm=float(_tree_l2_norm(initial_gradient)),
        initial_first_derivative_std=float(jnp.std(initial_first)),
        initial_second_derivative_std=float(jnp.std(initial_second)),
        final_loss=float(final_total),
        relative_l2=float(relative_l2),
        relative_h1=float(relative_h1),
        compile_seconds=float(compile_seconds),
        training_seconds=float(training_seconds),
        inference_seconds=float(inference_seconds),
        peak_memory_bytes=_memory_bytes(),
        losses=tuple(losses),
    )


def run_pinn_model_benchmarks(
    *,
    architectures: tuple[str, ...] = ("mlp", "modified_mlp", "piratenet", "siren"),
    scenarios: tuple[str, ...] = tuple(_SCENARIOS),
    seeds: tuple[int, ...] = (0, 1, 2),
    width: int = 32,
    depth: int = 4,
    steps: int = 500,
    learning_rate: float = 1e-3,
    match_parameters: bool = True,
) -> dict[str, object]:
    records = tuple(
        run_pinn_model_benchmark(
            architecture,
            _SCENARIOS[scenario_name],
            seed=seed,
            width=width,
            depth=depth,
            steps=steps,
            learning_rate=learning_rate,
            match_parameters=match_parameters,
        )
        for scenario_name in scenarios
        for architecture in architectures
        for seed in seeds
    )
    return {
        "environment": {
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
        },
        "records": [asdict(record) for record in records],
    }


def _comma_tuple(value: str, /) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run capacity-controlled pointwise SciML model benchmarks."
    )
    parser.add_argument("--architectures", default="mlp,modified_mlp,piratenet,siren")
    parser.add_argument("--scenarios", default=",".join(_SCENARIOS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-match-parameters", action="store_true")
    arguments = parser.parse_args()
    result = run_pinn_model_benchmarks(
        architectures=_comma_tuple(arguments.architectures),
        scenarios=_comma_tuple(arguments.scenarios),
        seeds=tuple(int(seed) for seed in _comma_tuple(arguments.seeds)),
        width=arguments.width,
        depth=arguments.depth,
        steps=1 if arguments.quick else arguments.steps,
        learning_rate=arguments.learning_rate,
        match_parameters=not arguments.no_match_parameters,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
