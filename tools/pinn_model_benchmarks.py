#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    DurationDistribution,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_synchronized,
    synchronize,
)
from phydrax._dtype_names import real_precision_dtype_name
from phydrax._jax_config import default_device
from phydrax._trainable import (
    combine_parameters,
    partition_parameters,
    require_parameter_roles,
)
from phydrax.optim import adam


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
    dtype: str
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
    preparation_seconds: float
    lowering_seconds: float
    compile_seconds: float
    first_execution_seconds: float | None
    warm_train_step: dict[str, Any]
    training_seconds: float
    inference_seconds: float
    peak_memory_bytes: int | None
    backend: str
    devices: tuple[str, ...]
    parameter_dtypes: tuple[str, ...]
    input_dtype: str
    train_points: int
    evaluation_points: int
    preparation_estimate_bytes: int
    logical_bytes: dict[str, int]
    compiler_estimates: dict[str, Any]
    input_fingerprint: str
    initial_model_fingerprint: str
    derivative_samples: dict[str, tuple[float, ...]]
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

# Architectures whose constructors build every parameter in an explicit dtype.
_EXPLICIT_DTYPE_ARCHITECTURES = ("mlp",)


def _workload_dtype(dtype: str | None, architectures: tuple[str, ...], /) -> str | None:
    """Resolve an explicit workload dtype, refusing architectures that ignore it."""
    if dtype is None:
        return None
    resolved = real_precision_dtype_name(dtype)
    unsupported = tuple(
        name for name in architectures if name not in _EXPLICIT_DTYPE_ARCHITECTURES
    )
    if unsupported:
        raise ValueError(
            f"Explicit dtype {resolved!r} is supported only for architectures "
            f"{_EXPLICIT_DTYPE_ARCHITECTURES}; got {unsupported}."
        )
    return resolved


def _parameter_count(model: eqx.Module, /) -> int:
    parameters, _, _ = partition_parameters(model)
    return sum(
        leaf.size
        for leaf in jax.tree_util.tree_leaves(parameters)
        if isinstance(leaf, jax.Array)
    )


def _build_model(
    architecture: str,
    /,
    *,
    width: int,
    depth: int,
    key: jax.Array,
    dtype: str | None = None,
) -> Any:
    if architecture == "mlp":
        return phx.nn.models.MLP(
            in_size=1,
            out_size="scalar",
            width_size=int(width),
            depth=int(depth),
            activation=jnp.tanh,
            rwf=False,
            dtype=dtype,
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
    dtype: str | None = None,
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
                    dtype=dtype,
                )
            )
            - int(target_parameters)
        ),
    )


def _manufactured_solution(points: jax.Array, frequency: int, /) -> jax.Array:
    omega = jnp.asarray(float(frequency) * jnp.pi, dtype=points.dtype)
    return jnp.sin(omega * points[..., 0])


def _pointwise_derivatives(model: Any, points: jax.Array, /) -> tuple[jax.Array, ...]:
    def scalar_value(point: Any) -> Any:
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
        forcing = jnp.asarray(omega**2, dtype=truth.dtype) * truth
        return -second_derivatives - forcing
    if scenario.equation == "helmholtz":
        wave_number = float(scenario.equation_parameter)
        forcing = jnp.asarray(wave_number**2 - omega**2, dtype=truth.dtype) * truth
        return (
            second_derivatives
            + jnp.asarray(wave_number**2, dtype=values.dtype) * values
            - forcing
        )
    if scenario.equation == "allen_cahn":
        diffusivity = float(scenario.equation_parameter)
        forcing = (
            jnp.asarray(diffusivity * omega**2, dtype=truth.dtype) * truth
            + truth**3
            - truth
        )
        return (
            jnp.asarray(-diffusivity, dtype=second_derivatives.dtype) * second_derivatives
            + values**3
            - values
            - forcing
        )
    raise ValueError(f"Unknown PINN benchmark equation {scenario.equation!r}.")


def _loss_components(
    model: Any,
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
    total = (
        residual_loss
        + jnp.asarray(float(scenario.boundary_weight), dtype=boundary_loss.dtype)
        * boundary_loss
    )
    return total, residual_loss, boundary_loss


def _tree_l2_norm(tree: Any, /) -> float:
    """Reporting-only float64 host norm of the actual rounded gradient leaves."""
    squared = np.float64(0.0)
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.Array):
            host = np.asarray(jax.device_get(leaf))
            dtype = np.complex128 if np.iscomplexobj(host) else np.float64
            values = host.astype(dtype)
            squared += np.sum(np.abs(values) ** 2, dtype=np.float64)
    return float(np.sqrt(squared))


def _memory_bytes() -> int | None:
    statistics = jax.devices()[0].memory_stats()
    if statistics is None:
        return None
    if "peak_bytes_in_use" in statistics:
        return int(statistics["peak_bytes_in_use"])
    return None


def _array_fingerprint(tree: Any, /) -> str:
    """Hash exact host payloads, shapes and dtypes in PyTree leaf order."""
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, (jax.Array, np.ndarray)):
            host = np.asarray(jax.device_get(leaf))
            digest.update(str((host.shape, host.dtype.str)).encode())
            digest.update(host.tobytes(order="C"))
    return digest.hexdigest()


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
    dtype: str | None = None,
    max_preparation_bytes: int = 512 * 1024 * 1024,
) -> PINNBenchmarkRecord:
    """Train one capacity-controlled model on a manufactured PDE residual.

    An explicit `dtype` builds the model parameters, training points, boundary
    points and evaluation grid in that dtype; `None` keeps the default route.
    """
    if int(width) <= 0 or int(depth) <= 0:
        raise ValueError("width and depth must be positive.")
    if int(steps) < 0 or float(learning_rate) <= 0.0:
        raise ValueError("steps must be non-negative and learning_rate positive.")
    workload_dtype = _workload_dtype(dtype, (architecture,))
    if jax.default_backend().lower() in ("metal", "mps") and (
        architecture != "mlp" or workload_dtype != "float32"
    ):
        raise ValueError(
            "Metal PINN benchmarks require --architectures mlp --dtype float32."
        )
    if scenario.train_points < 1 or scenario.evaluation_points < 2:
        raise ValueError(
            "Scenarios require positive training and at least two evaluation points."
        )
    # Conservative logical preparation budget, not a compiler/GPU-memory bound.
    # Include parameter matching candidates, Adam state and derivative activations.
    itemsize = np.dtype(workload_dtype or "float64").itemsize
    preparation_estimate = itemsize * (
        64 * (int(depth) + 2) * (2 * int(width) + 2) ** 2
        + 64
        * max(scenario.train_points, scenario.evaluation_points)
        * (2 * int(width) + 2)
        * (int(depth) + 1)
        + int(steps)
    )
    if max_preparation_bytes <= 0 or preparation_estimate > max_preparation_bytes:
        raise ValueError(
            f"Preparation estimate {preparation_estimate} exceeds budget "
            f"{max_preparation_bytes}; reduce width, depth, points or steps."
        )
    preparation_started = time.perf_counter()

    def prepare() -> tuple[Any, int, int, jax.Array, jax.Array, jax.Array]:
        root_key = jr.key(int(seed))
        baseline_key, model_key, sample_key = jr.split(root_key, 3)
        baseline = _build_model(
            "mlp", width=width, depth=depth, key=baseline_key, dtype=workload_dtype
        )
        target_parameters = _parameter_count(baseline)
        selected_width = (
            _matched_width(
                architecture,
                requested_width=width,
                depth=depth,
                target_parameters=target_parameters,
                key=model_key,
                dtype=workload_dtype,
            )
            if match_parameters
            else int(width)
        )
        model = _build_model(
            architecture,
            width=selected_width,
            depth=depth,
            key=model_key,
            dtype=workload_dtype,
        )
        if workload_dtype is None:
            lower, upper = -1.0, 1.0
        else:
            lower = jnp.asarray(-1.0, dtype=workload_dtype)
            upper = jnp.asarray(1.0, dtype=workload_dtype)
        interior = jr.uniform(
            sample_key,
            (int(scenario.train_points), 1),
            dtype=workload_dtype,
            minval=lower,
            maxval=upper,
        )
        boundary = jnp.asarray([[-1.0], [1.0]], dtype=workload_dtype)
        evaluation = jnp.linspace(
            lower,
            upper,
            int(scenario.evaluation_points),
            dtype="float64" if workload_dtype is None else workload_dtype,
        )[:, None]
        return model, target_parameters, selected_width, interior, boundary, evaluation

    # Explicit-dtype comparisons construct identical CPU inputs, then transfer
    # those payloads to the requested backend. This does not run a CPU reference.
    if workload_dtype is not None:
        with default_device(jax.devices("cpu")[0]):
            prepared = synchronize(prepare())
        model, target_parameters, selected_width, interior, boundary, evaluation = (
            prepared
        )
        model_arrays, model_static = eqx.partition(model, eqx.is_array)
        model = eqx.combine(jax.device_put(model_arrays, jax.devices()[0]), model_static)
        interior, boundary, evaluation = jax.device_put(
            (interior, boundary, evaluation), jax.devices()[0]
        )
    else:
        model, target_parameters, selected_width, interior, boundary, evaluation = (
            prepare()
        )
    input_fingerprint = _array_fingerprint((interior, boundary, evaluation))
    initial_model_fingerprint = _array_fingerprint(eqx.filter(model, eqx.is_array))
    initial_total, initial_residual, initial_boundary = _loss_components(
        model, scenario, interior, boundary
    )
    require_parameter_roles(model, context="run_pinn_model_benchmark")
    parameters, model_state, fixed = partition_parameters(model)

    def objective(candidate: Any) -> Any:
        current = combine_parameters(candidate, model_state, fixed)
        return _loss_components(current, scenario, interior, boundary)[0]

    initial_gradient = eqx.filter_grad(objective)(parameters)
    _, initial_first, initial_second = _pointwise_derivatives(model, interior)

    optimizer = adam(float(learning_rate))
    optimizer_state = optimizer.init(parameters)

    @eqx.filter_jit
    def train_step(current_parameters: Any, current_state: Any) -> Any:
        value, gradient = eqx.filter_value_and_grad(objective)(current_parameters)
        updates, next_state = optimizer.update(
            gradient, current_state, current_parameters
        )
        return eqx.apply_updates(current_parameters, updates), next_state, value

    synchronize(
        (
            parameters,
            optimizer_state,
            initial_total,
            initial_residual,
            initial_boundary,
            initial_gradient,
            initial_first,
            initial_second,
        )
    )
    preparation_seconds = time.perf_counter() - preparation_started
    logical_bytes = {
        "model": logical_array_bytes(model),
        "inputs": logical_array_bytes((interior, boundary, evaluation)),
        "optimizer_state": logical_array_bytes(optimizer_state),
        "initial_gradient": logical_array_bytes(initial_gradient),
        "train_step_arguments": logical_array_bytes((parameters, optimizer_state)),
    }
    compiled, timing = measure_lower_and_compile(
        lambda: train_step.lower(parameters, optimizer_state),  # ty: ignore[unresolved-attribute]
        lambda lowered: lowered.compile(),
    )
    cost_analysis = memory_analysis = None
    analysis_reasons: list[str] = []
    try:
        cost_analysis = compiled.compiled.cost_analysis()
    except (AttributeError, NotImplementedError, RuntimeError) as error:
        analysis_reasons.append(f"cost analysis: {type(error).__name__}: {error}")
    try:
        memory_analysis = compiled.compiled.memory_analysis()
    except (AttributeError, NotImplementedError, RuntimeError) as error:
        analysis_reasons.append(f"memory analysis: {type(error).__name__}: {error}")
    estimates = compiler_evidence(
        cost_analysis,
        memory_analysis,
        source="xla",
        unavailable_reason="; ".join(analysis_reasons)
        or "Backend omitted compiler estimates.",
    )
    losses_device: list[jax.Array] = []
    first_seconds: float | None = None
    warm_samples: list[float] = []
    training_started = time.perf_counter()
    if int(steps) > 0:
        (parameters, optimizer_state, loss), first_seconds = measure_synchronized(
            lambda: compiled(parameters, optimizer_state)
        )
        losses_device.append(loss)
        for _ in range(1, int(steps)):
            (parameters, optimizer_state, loss), elapsed = measure_synchronized(
                lambda: compiled(parameters, optimizer_state)
            )
            warm_samples.append(elapsed)
            losses_device.append(loss)
    training_seconds = time.perf_counter() - training_started
    # Keep all per-step scientific losses; scalar host conversion is outside timing.
    losses = tuple(float(value) for value in jax.device_get(tuple(losses_device)))

    trained = combine_parameters(parameters, model_state, fixed)
    final_total = _loss_components(trained, scenario, interior, boundary)[0]
    synchronize(final_total)
    (prediction, prediction_gradient, prediction_second), inference_seconds = (
        measure_synchronized(lambda: _pointwise_derivatives(trained, evaluation))
    )

    truth = _manufactured_solution(evaluation, scenario.frequency)
    omega = jnp.asarray(float(scenario.frequency) * jnp.pi, dtype=evaluation.dtype)
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
        dtype=str(prediction.dtype),
        parameter_count=_parameter_count(model),
        target_parameter_count=int(target_parameters),
        training_steps=int(steps),
        learning_rate=float(learning_rate),
        initial_loss=float(initial_total),
        initial_residual_loss=float(initial_residual),
        initial_boundary_loss=float(initial_boundary),
        initial_gradient_norm=_tree_l2_norm(initial_gradient),
        initial_first_derivative_std=float(jnp.std(initial_first)),
        initial_second_derivative_std=float(jnp.std(initial_second)),
        final_loss=float(final_total),
        relative_l2=float(relative_l2),
        relative_h1=float(relative_h1),
        preparation_seconds=preparation_seconds,
        lowering_seconds=timing.lowering_seconds,
        compile_seconds=timing.compilation_seconds,
        first_execution_seconds=first_seconds,
        warm_train_step=DurationDistribution(tuple(warm_samples)).to_seconds_dict(),
        training_seconds=float(training_seconds),
        inference_seconds=float(inference_seconds),
        peak_memory_bytes=_memory_bytes(),
        backend=jax.default_backend(),
        devices=tuple(sorted(str(device) for device in prediction.devices())),
        parameter_dtypes=tuple(
            sorted(
                {
                    str(leaf.dtype)
                    for leaf in jax.tree_util.tree_leaves(parameters)
                    if isinstance(leaf, jax.Array)
                }
            )
        ),
        input_dtype=str(interior.dtype),
        train_points=int(scenario.train_points),
        evaluation_points=int(scenario.evaluation_points),
        preparation_estimate_bytes=preparation_estimate,
        logical_bytes=logical_bytes,
        compiler_estimates={
            **asdict(estimates),
            "estimated_device_memory_bytes": estimates.estimated_device_memory_bytes,
        },
        input_fingerprint=input_fingerprint,
        initial_model_fingerprint=initial_model_fingerprint,
        derivative_samples={
            name: tuple(float(value) for value in np.asarray(jax.device_get(array[:8])))
            for name, array in (
                ("initial_first", initial_first),
                ("initial_second", initial_second),
                ("final_value", prediction),
                ("final_first", prediction_gradient),
                ("final_second", prediction_second),
            )
        },
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
    dtype: str | None = None,
    max_preparation_bytes: int = 512 * 1024 * 1024,
) -> dict[str, object]:
    _workload_dtype(dtype, architectures)
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
            dtype=dtype,
            max_preparation_bytes=max_preparation_bytes,
        )
        for scenario_name in scenarios
        for architecture in architectures
        for seed in seeds
    )
    return {
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            Path(__file__).resolve().parents[1],
            Path(__file__).resolve(),
            tuple(field.name for field in fields(PINNBenchmarkRecord)),
        ).to_dict(),
        "reference_policy": {
            "cpu_reference_executed_in_this_report": jax.default_backend() == "cpu",
            "comparison": "Separate CPU and Metal processes; no automatic reference run.",
            "inputs": "Explicit dtype uses CPU initialization and exact SHA256 payload fingerprints.",
            "required_matches": [
                "input_fingerprint",
                "initial_model_fingerprint",
                "scenario",
                "equation",
                "architecture",
                "seed",
                "width",
                "depth",
                "training_steps",
                "learning_rate",
                "dtype",
                "parameter_dtypes",
                "train_points",
                "evaluation_points",
            ],
            "float32_comparison": {
                "rtol": 1e-4,
                "atol": 1e-5,
                "equal_nan": False,
                "evidence": (
                    "Compare all losses, initial loss components, gradient norm, "
                    "derivative standard deviations/samples, final loss, relative_l2 "
                    "and relative_h1; require finite evidence and positive gradient "
                    "norm. For nonzero training require final_loss < initial_loss. "
                    "No automatic pass is asserted by this benchmark."
                ),
            },
            "gradient_norm": "Reporting-only host float64 accumulation over rounded gradient leaves.",
            "timing": (
                "Preparation includes initialization, transfers and initial diagnostics; "
                "lowering and compilation exclude execution. First execution is training "
                "step one; warm samples are steps two onward. No extra optimizer updates. "
                "Training excludes compilation and host scalar conversion. Inference "
                "includes eager dispatch of values and both derivatives, fully synchronized."
            ),
            "memory": (
                "Preparation estimate is a preallocation guard, not a device-memory bound; "
                "logical bytes are unique payloads per category (categories overlap). "
                "Compiler estimates are backend-reported; peak memory is process-lifetime."
            ),
        },
        "records": [asdict(record) for record in records],
    }


def run_multifidelity_pinn_benchmark(
    *,
    seed: int = 0,
    low_steps: int = 80,
    target_steps: int = 100,
    learning_rate: float = 0.06,
) -> dict[str, object]:
    """Benchmark staged low-model training and target-physics correction."""
    started = time.perf_counter()
    low = phx.fidelity.FidelityLevelSpec(
        "low",
        problem_id="poisson",
        observable_id="u",
        model_id="biased-poisson",
        approximation_id="low",
        observable_contract_id="scalar-field",
    )
    high = phx.fidelity.FidelityLevelSpec(
        "high",
        problem_id="poisson",
        observable_id="u",
        model_id="target-poisson",
        approximation_id="high",
        observable_contract_id="scalar-field",
    )
    hierarchy = phx.fidelity.FidelityHierarchy(
        (low, high),
        (phx.fidelity.FidelityRelation("low", "high"),),
        target_level_id="high",
    )
    points = jnp.linspace(-1.0, 1.0, 21)
    cases = tuple(
        phx.fidelity.FidelityCaseSpec(
            {"x": point},
            case_id=f"case-{index}",
            split_group_id=f"physical-{index}",
        )
        for index, point in enumerate(points)
    )
    evaluations = []
    for index, point in enumerate(points):
        evaluations.append(
            phx.fidelity.FidelityEvaluation(
                0.8 * point * point,
                case_id=f"case-{index}",
                pair_id=f"pair-{index}",
                level_id="low",
                evaluator_id="pinn-benchmark",
                valid=True,
                cost=1.0,
                cost_unit="relative",
            )
        )
        if index % 2 == 0:
            evaluations.append(
                phx.fidelity.FidelityEvaluation(
                    point * point,
                    case_id=f"case-{index}",
                    pair_id=f"pair-{index}",
                    level_id="high",
                    evaluator_id="pinn-benchmark",
                    valid=True,
                    cost=20.0,
                    cost_unit="relative",
                )
            )
    dataset = phx.fidelity.FidelityDataset(hierarchy, cases, tuple(evaluations))
    split = phx.fidelity.split_fidelity_dataset(
        dataset,
        train_fraction=0.6,
        validation_fraction=0.2,
        seed=seed,
        requirements=phx.fidelity.FidelitySplitRequirements(
            train={"low": 2, "high": 2},
            validation={"low": 1, "high": 1},
            test={"high": 1},
        ),
    )
    geometry = phx.domain.Interval1d(-1.0, 1.0)
    component = geometry.component()
    low_train = phx.terms.prepare_fidelity_observation_penalty(
        split.train,
        "low",
        field="u",
        component=component,
    )
    low_validation = phx.terms.prepare_fidelity_observation_penalty(
        split.validation,
        "low",
        field="u",
        component=component,
    )
    high_train = phx.terms.prepare_fidelity_observation_penalty(
        split.train,
        "high",
        field="u",
        component=component,
    )
    high_validation = phx.terms.prepare_fidelity_observation_penalty(
        split.validation,
        "high",
        field="u",
        component=component,
    )
    high_test = phx.terms.prepare_fidelity_observation_penalty(
        split.test,
        "high",
        field="u",
        component=component,
    )
    x_squared = geometry.Function("x")(lambda x: x * x)
    low_started = time.perf_counter()
    low_solver = phx.solver.FunctionalSolver(
        functions={"u": geometry.Parameter(0.4) * x_squared},
        terms=(low_train.term,),
        evaluation_terms=(low_validation.term,),
    ).solve(
        num_iter=int(low_steps),
        optim=adam(float(learning_rate)),
        seed=seed,
        log_every=0,
    )
    low_seconds = time.perf_counter() - low_started
    parent = phx.solver.bind_fidelity_pinn_level(
        hierarchy.linear_path(),
        "low",
        low_solver,
        training_observations=(low_train,),
        validation_observations=(low_validation,),
    )
    pde_condition = phx.conditions.Residual(
        "u",
        component,
        lambda u: phx.operators.differential.laplacian(u, var="x") - 2.0,
        label="target-poisson",
    )
    pde_realization = phx.integration.materialize(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(48),
        key=jr.key(seed + 1),
    )
    pde = phx.terms.ResidualPenalty(
        pde_condition,
        phx.integration.fixed(pde_realization),
    )
    parent_before = parent.functions["u"](high_test.batch).data
    target_started = time.perf_counter()
    stage = phx.solver.prepare_fidelity_pinn_stage(
        parent,
        "high",
        {"u": geometry.Parameter(0.0) * x_squared},
        (pde,),
        training_observations=(high_train,),
        validation_observations=(high_validation,),
        epsilon=1.0,
    )
    trained = stage.training_solver.solve(
        num_iter=int(target_steps),
        optim=adam(float(learning_rate)),
        seed=seed + 2,
        log_every=0,
    )
    result = stage.finalize(trained)
    target_seconds = time.perf_counter() - target_started
    evaluation = phx.solver.evaluate_fidelity_pinn(
        result,
        (high_test,),
        physics_terms=(pde,),
    )
    parent_after = parent.functions["u"](high_test.batch).data
    parent_rmse = float(jnp.sqrt(jnp.mean((parent_before - high_test.targets) ** 2)))
    target_rmse = float(evaluation.target_data_rmse)
    physics_loss = float(evaluation.physics_losses[0])
    parent_unchanged = bool(jnp.array_equal(parent_before, parent_after))
    finite = bool(
        np.isfinite(target_rmse)
        and np.isfinite(physics_loss)
        and np.isfinite(parent_rmse)
    )
    return {
        "workflow": "multifidelity",
        "seed": int(seed),
        "low_steps": int(low_steps),
        "target_steps": int(target_steps),
        "low_seconds": low_seconds,
        "target_seconds": target_seconds,
        "wall_seconds": time.perf_counter() - started,
        "parent_target_rmse": parent_rmse,
        "target_rmse": target_rmse,
        "target_physics_loss": physics_loss,
        "parent_unchanged": parent_unchanged,
        "hierarchy_id": hierarchy.hierarchy_id,
        "stage_id": stage.stage_id,
        "result_id": result.result_id,
        "passed": bool(
            finite
            and parent_unchanged
            and target_rmse < parent_rmse
            and target_rmse < 0.05
            and physics_loss < 0.05
        ),
    }


def _comma_tuple(value: str, /) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run capacity-controlled pointwise SciML model benchmarks."
    )
    parser.add_argument("--architectures", default="mlp,modified_mlp,piratenet,siren")
    parser.add_argument("--scenarios", default=",".join(_SCENARIOS))
    parser.add_argument(
        "--workflow",
        choices=("architecture", "multifidelity"),
        default="architecture",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--max-preparation-bytes",
        type=int,
        default=512 * 1024 * 1024,
        help="Reject estimated logical preparation workloads above this budget before allocation.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-match-parameters", action="store_true")
    parser.add_argument(
        "--dtype",
        default=None,
        help=(
            "Explicit real floating workload dtype (for example float32) for "
            f"architectures {','.join(_EXPLICIT_DTYPE_ARCHITECTURES)}; "
            "omit to keep the default route."
        ),
    )
    arguments = parser.parse_args()
    if arguments.workflow == "multifidelity":
        if arguments.dtype is not None:
            parser.error("--dtype applies only to the architecture workflow.")
        result = run_multifidelity_pinn_benchmark(
            seed=int(_comma_tuple(arguments.seeds)[0]),
            low_steps=1 if arguments.quick else arguments.steps,
            target_steps=1 if arguments.quick else arguments.steps,
            learning_rate=arguments.learning_rate,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    result = run_pinn_model_benchmarks(
        architectures=_comma_tuple(arguments.architectures),
        scenarios=_comma_tuple(arguments.scenarios),
        seeds=tuple(int(seed) for seed in _comma_tuple(arguments.seeds)),
        width=arguments.width,
        depth=arguments.depth,
        steps=1 if arguments.quick else arguments.steps,
        learning_rate=arguments.learning_rate,
        match_parameters=not arguments.no_match_parameters,
        dtype=arguments.dtype,
        max_preparation_bytes=arguments.max_preparation_bytes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
