#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import phydrax as phx
from benchmarks._runtime import logical_array_bytes, measure_repeated, synchronize


def _measure(
    operation: Callable[[], Any],
    /,
    *,
    repeats: int,
) -> tuple[Any, float, float]:
    result, distribution = measure_repeated(
        operation,
        warmup=1,
        repeats=repeats,
    )
    samples = 1_000.0 * np.asarray(distribution.samples_seconds)
    return result, float(np.mean(samples)), float(np.std(samples))


def _parameter_count(tree: Any, /) -> int:
    return sum(
        int(leaf.size)
        for leaf in jax.tree_util.tree_leaves(tree)
        if eqx.is_inexact_array(leaf)
    )


def _compile_and_train(
    model: Any,
    loss: Callable[[Any], jax.Array],
    /,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[Any, float, float, float, float]:
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def update(candidate, state):
        value, gradient = eqx.filter_value_and_grad(loss)(candidate)
        updates, state = optimizer.update(gradient, state, candidate)
        return eqx.apply_updates(candidate, updates), state, value

    compiled_update = eqx.filter_jit(update)
    started = time.perf_counter()
    model, optimizer_state, initial_loss = compiled_update(model, optimizer_state)
    synchronize((model, optimizer_state, initial_loss))
    compilation_ms = 1e3 * (time.perf_counter() - started)
    started = time.perf_counter()
    final_loss = initial_loss
    for _ in range(max(steps - 1, 0)):
        model, optimizer_state, final_loss = compiled_update(model, optimizer_state)
    synchronize((model, optimizer_state, final_loss))
    training_ms = 1e3 * (time.perf_counter() - started)
    return (
        model,
        compilation_ms,
        training_ms,
        float(initial_loss),
        float(final_loss),
    )


def _finite_record(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _lorenz_derivative(states: jax.Array, /) -> jax.Array:
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    return jnp.stack(
        (
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ),
        axis=-1,
    )


def _lorenz_coefficients(feature_names: tuple[str, ...], /) -> jax.Array:
    index = {name: position for position, name in enumerate(feature_names)}
    coefficients = jnp.zeros((3, len(feature_names)))
    coefficients = coefficients.at[0, index["state:x"]].set(-10.0)
    coefficients = coefficients.at[0, index["state:y"]].set(10.0)
    coefficients = coefficients.at[1, index["state:x"]].set(28.0)
    coefficients = coefficients.at[1, index["state:y"]].set(-1.0)
    coefficients = coefficients.at[1, index["state:x * state:z"]].set(-1.0)
    coefficients = coefficients.at[2, index["state:z"]].set(-8.0 / 3.0)
    coefficients = coefficients.at[2, index["state:x * state:y"]].set(1.0)
    return coefficients


def _sparse_recovery_benchmark(
    *,
    samples: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    states = jnp.asarray(
        generator.uniform(
            low=np.asarray((-20.0, -30.0, 0.0)),
            high=np.asarray((20.0, 30.0, 50.0)),
            size=(samples, 3),
        )
    )
    layout = phx.dynamics.StateLayout(
        (3,),
        component_names=("x", "y", "z"),
    )
    data = phx.dynamics.TrajectoryData(
        jnp.arange(samples, dtype=states.dtype),
        states,
        state_layout=layout,
        derivatives=_lorenz_derivative(states),
        coordinate_id="sample",
        source_id="benchmark:lorenz-exact",
    )
    library = phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=2)
    problem = phx.dynamics.identification.SINDyProblem(
        data=data,
        library=library,
        formulation=phx.dynamics.identification.StrongSINDyFormulation(),
    )
    regressor = phx.dynamics.identification.SequentialThresholdedLeastSquares(
        1e-8,
        scale_features=True,
        threshold_space="physical",
        unbiased_refit=True,
    )
    result, mean_ms, standard_deviation_ms = _measure(
        lambda: phx.dynamics.identification.fit_sindy(problem, regressor),
        repeats=repeats,
    )
    expected = _lorenz_coefficients(result.design.feature_names)
    expected_support = expected != 0.0
    recovered_support = result.support
    true_positive = int(jnp.sum(expected_support & recovered_support))
    false_positive = int(jnp.sum(~expected_support & recovered_support))
    false_negative = int(jnp.sum(expected_support & ~recovered_support))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    coefficient_error = float(jnp.max(jnp.abs(result.coefficients - expected)))
    target = result.design.target
    prediction = result.predict_design()
    relative_residual = float(
        jnp.linalg.norm(prediction - target) / jnp.maximum(jnp.linalg.norm(target), 1.0)
    )
    passed = bool(result.valid) and precision == 1.0 and recall == 1.0
    passed = passed and coefficient_error <= 1e-8 and relative_residual <= 1e-10
    return {
        "problem_id": "lorenz-exact-strong-sindy",
        "samples": samples,
        "features": result.design.num_features,
        "outputs": result.design.output_size,
        "mean_ms": mean_ms,
        "standard_deviation_ms": standard_deviation_ms,
        "working_set_bytes": logical_array_bytes((data, result)),
        "valid": bool(result.valid),
        "status": np.asarray(result.status).tolist(),
        "support_size": int(jnp.sum(recovered_support)),
        "support_precision": precision,
        "support_recall": recall,
        "maximum_coefficient_error": coefficient_error,
        "relative_residual": relative_residual,
        "passed": passed,
    }


def _diagonal_map(coordinate, state, rates):
    del coordinate
    return jnp.exp(rates) * state


def _matrix_free_benchmark(
    *,
    dimension: int,
    leading_k: int,
    num_steps: int,
    repeats: int,
) -> dict[str, Any]:
    layout = phx.dynamics.StateLayout((dimension,))
    system = phx.dynamics.DiscreteSystem(
        _diagonal_map,
        state_layout=layout,
        system_id=f"benchmark:diagonal-map:{dimension}",
    )
    evolution = phx.dynamics.DiscreteEvolution(system)
    grid = phx.dynamics.IterationGrid.from_steps(
        num_steps,
        iteration_id=f"benchmark:diagonal-map:{dimension}:{num_steps}",
    )
    rates = jnp.linspace(0.5, -3.0, leading_k)
    if leading_k < dimension:
        rates = jnp.concatenate((rates, jnp.linspace(-3.5, -4.0, dimension - leading_k)))
    initial_state = jnp.ones((dimension,))
    initial_basis = jnp.eye(dimension, leading_k)
    cadence = max(1, min(4, num_steps))
    burn_in = cadence * max(1, num_steps // (4 * cadence))
    accumulation_interval = cadence * max(1, num_steps // (8 * cadence))
    num_intervals = (num_steps + cadence - 1) // cadence
    backward_discard = max(1, num_intervals // 2)

    spectrum, spectrum_mean_ms, spectrum_standard_deviation_ms = _measure(
        lambda: phx.dynamics.analysis.finite_time_lyapunov_spectrum(
            evolution,
            initial_state,
            grid,
            args=rates,
            leading_k=leading_k,
            qr_interval=cadence,
            burn_in=burn_in,
            accumulation_interval=accumulation_interval,
            initial_basis=initial_basis,
        ),
        repeats=repeats,
    )
    directions, directions_mean_ms, directions_standard_deviation_ms = _measure(
        lambda: phx.dynamics.analysis.covariant_directions(
            evolution,
            initial_state,
            grid,
            args=rates,
            leading_k=leading_k,
            initial_basis=initial_basis,
            memory_mode="store",
            qr_interval=cadence,
            save_every=2,
            backward_discard=backward_discard,
            convergence_tolerance=1e-6,
        ),
        repeats=repeats,
    )
    expected = rates[:leading_k]
    exponent_error = float(jnp.max(jnp.abs(spectrum.exponents - expected)))
    valid_covariance = jnp.where(
        directions.direction_valid[..., None],
        directions.covariance_error,
        jnp.nan,
    )
    maximum_covariance_error = float(jnp.nanmax(valid_covariance))
    valid_drift = jnp.where(
        directions.direction_valid[..., None],
        directions.backward_convergence_drift,
        jnp.nan,
    )
    maximum_convergence_drift = float(jnp.nanmax(valid_drift))
    passed = bool(spectrum.valid) and exponent_error <= 1e-10
    passed = passed and bool(directions.valid) and bool(directions.converged)
    passed = passed and maximum_covariance_error <= 1e-10
    return {
        "problem_id": "high-dimensional-diagonal-map",
        "dimension": dimension,
        "leading_k": leading_k,
        "num_steps": num_steps,
        "dense_jacobian_materialized": False,
        "tangent_method": spectrum.tangent_method,
        "spectrum": {
            "mean_ms": spectrum_mean_ms,
            "standard_deviation_ms": spectrum_standard_deviation_ms,
            "working_set_bytes": logical_array_bytes(spectrum),
            "valid": bool(spectrum.valid),
            "status": int(spectrum.status),
            "maximum_exponent_error": exponent_error,
        },
        "covariant_directions": {
            "mean_ms": directions_mean_ms,
            "standard_deviation_ms": directions_standard_deviation_ms,
            "working_set_bytes": logical_array_bytes(directions),
            "valid": bool(directions.valid),
            "converged": bool(directions.converged),
            "maximum_backward_convergence_drift": maximum_convergence_drift,
            "status": int(directions.status),
            "maximum_covariance_error": maximum_covariance_error,
            "stored_frame_count": directions.stored_frame_count,
            "tangent_evaluations": directions.tangent_evaluations,
        },
        "passed": passed,
    }


def _oscillator_derivative(state: jax.Array, /) -> jax.Array:
    position, momentum = state
    damping = 0.12 + 0.08 * position**2
    return jnp.asarray(
        [
            momentum,
            -position - 0.4 * position**3 - damping * momentum,
        ]
    )


def _rk4_rollout(
    vector_field: Callable[[jax.Array], jax.Array],
    initial_state: jax.Array,
    /,
    *,
    dt: float,
    num_steps: int,
) -> jax.Array:
    def step(state, _):
        k1 = vector_field(state)
        k2 = vector_field(state + 0.5 * dt * k1)
        k3 = vector_field(state + 0.5 * dt * k2)
        k4 = vector_field(state + dt * k3)
        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return next_state, next_state

    _, states = jax.lax.scan(step, initial_state, xs=None, length=num_steps)
    return jnp.concatenate((initial_state[None, :], states), axis=0)


def _deterministic_architecture(
    architecture: str,
    /,
    *,
    key: jax.Array,
    quick: bool,
) -> Any:
    width = 10 if quick else 20
    depth = 1 if quick else 2
    if architecture == "unstructured_mlp":
        return phx.nn.models.MLP(
            in_size=2,
            out_size=2,
            width_size=width,
            depth=depth,
            key=key,
        )
    if architecture == "constant_port_hamiltonian":
        return phx.nn.models.PortHamiltonianVectorField(
            state_size=2,
            energy_width=width,
            energy_depth=depth,
            dissipation_structure="positive_semidefinite",
            initial_damping=0.1,
            key=key,
        )
    if architecture == "state_dependent_port_hamiltonian":
        interconnection_key, dissipation_key, field_key = jr.split(key, 3)
        return phx.nn.models.PortHamiltonianVectorField(
            state_size=2,
            energy_width=width,
            energy_depth=depth,
            interconnection_model=phx.nn.models.MLP(
                in_size=2,
                out_size=1,
                width_size=width,
                depth=depth,
                key=interconnection_key,
            ),
            dissipation_model=phx.nn.models.MLP(
                in_size=2,
                out_size=3,
                width_size=width,
                depth=depth,
                key=dissipation_key,
            ),
            dissipation_structure="positive_semidefinite",
            key=field_key,
        )
    raise ValueError(f"Unsupported deterministic architecture {architecture!r}.")


def _deterministic_thermodynamic_record(
    architecture: str,
    /,
    *,
    seed: int,
    repeats: int,
    quick: bool,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    train_count = 96 if quick else 768
    test_count = 64 if quick else 256
    train_states = jnp.asarray(generator.uniform(-1.25, 1.25, size=(train_count, 2)))
    test_states = jnp.asarray(generator.uniform(-1.25, 1.25, size=(test_count, 2)))
    train_targets = jax.vmap(_oscillator_derivative)(train_states)
    test_targets = jax.vmap(_oscillator_derivative)(test_states)
    model = _deterministic_architecture(
        architecture,
        key=jr.key(seed),
        quick=quick,
    )

    def derivative_loss(candidate):
        predictions = jax.vmap(candidate)(train_states)
        return jnp.mean((predictions - train_targets) ** 2)

    model, compilation_ms, training_ms, initial_loss, final_loss = _compile_and_train(
        model,
        derivative_loss,
        steps=12 if quick else 180,
        learning_rate=4e-3,
    )
    predict = eqx.filter_jit(lambda candidate, values: jax.vmap(candidate)(values))
    predictions, inference_ms, inference_std_ms = _measure(
        lambda: predict(model, test_states),
        repeats=repeats,
    )
    derivative_mse = float(jnp.mean((predictions - test_targets) ** 2))
    initial_state = jnp.asarray([0.7, -0.15])
    short_steps = 20 if quick else 50
    long_steps = 80 if quick else 250
    dt = 0.02
    reference = _rk4_rollout(
        _oscillator_derivative,
        initial_state,
        dt=dt,
        num_steps=long_steps,
    )
    rollout = eqx.filter_jit(
        lambda candidate: _rk4_rollout(
            candidate,
            initial_state,
            dt=dt,
            num_steps=long_steps,
        )
    )(model)
    short_rollout_mse = float(
        jnp.mean((rollout[: short_steps + 1] - reference[: short_steps + 1]) ** 2)
    )
    long_rollout_mse = float(jnp.mean((rollout - reference) ** 2))
    if architecture == "unstructured_mlp":
        maximum_energy_balance_residual = None
        positive_energy_drift_fraction = None
    else:
        source_states = rollout[:-1]
        residuals = jax.vmap(model.energy_balance_residual)(source_states)
        energy_rates = jax.vmap(model.energy_rate)(source_states)
        maximum_energy_balance_residual = float(jnp.max(jnp.abs(residuals)))
        positive_energy_drift_fraction = float(jnp.mean(energy_rates > 1e-10))
    passed = _finite_record(
        initial_loss,
        final_loss,
        derivative_mse,
        short_rollout_mse,
        long_rollout_mse,
        compilation_ms,
        training_ms,
        inference_ms,
    )
    if (
        maximum_energy_balance_residual is not None
        and positive_energy_drift_fraction is not None
    ):
        passed = passed and _finite_record(
            maximum_energy_balance_residual,
            positive_energy_drift_fraction,
        )
    return {
        "scenario_id": "deterministic-nonlinear-oscillator",
        "architecture": architecture,
        "seed": seed,
        "parameter_count": _parameter_count(model),
        "compilation_ms": compilation_ms,
        "training_ms": training_ms,
        "inference_mean_ms": inference_ms,
        "inference_standard_deviation_ms": inference_std_ms,
        "initial_training_loss": initial_loss,
        "final_training_loss": final_loss,
        "held_out_derivative_mse": derivative_mse,
        "short_horizon_rollout_mse": short_rollout_mse,
        "long_horizon_rollout_mse": long_rollout_mse,
        "maximum_energy_balance_residual": maximum_energy_balance_residual,
        "positive_energy_drift_fraction": positive_energy_drift_fraction,
        "passed": passed,
    }


def _deterministic_thermodynamic_benchmark(
    *,
    architectures: tuple[str, ...],
    seeds: tuple[int, ...],
    repeats: int,
    quick: bool,
) -> dict[str, Any]:
    supported = (
        "unstructured_mlp",
        "constant_port_hamiltonian",
        "state_dependent_port_hamiltonian",
    )
    selected = tuple(name for name in supported if name in architectures)
    records = [
        _deterministic_thermodynamic_record(
            architecture,
            seed=seed,
            repeats=repeats,
            quick=quick,
        )
        for seed in seeds
        for architecture in selected
    ]
    return {
        "scenario_id": "deterministic-nonlinear-oscillator",
        "records": records,
        "passed": bool(records) and all(record["passed"] for record in records),
    }


class _ConstantDiffusionCoefficient(eqx.Module):
    value: float = eqx.field(static=True)

    def __call__(self, time, state, args):
        del time, state, args
        return jnp.asarray([[self.value]])


class _PositiveDiffusionCoefficient(eqx.Module):
    model: Any
    minimum: float = eqx.field(static=True)

    def __init__(self, model: Any, /, *, minimum: float = 1e-4):
        self.model = model
        self.minimum = float(minimum)

    def __call__(self, time, state, args):
        del time, args
        value = jnp.asarray(self.model(state)).reshape(())
        return (jax.nn.softplus(value) + self.minimum).reshape((1, 1))


def _variable_mobility(state: jax.Array, /) -> jax.Array:
    return 0.35 + 0.15 * state**2


def _variable_mobility_drift(
    state: jax.Array,
    /,
    *,
    temperature: float,
) -> jax.Array:
    mobility = _variable_mobility(state)
    mobility_derivative = 0.3 * state
    return -mobility * state + temperature * mobility_derivative


def _stochastic_kernel(
    architecture: str,
    /,
    *,
    key: jax.Array,
    quick: bool,
    temperature: float,
) -> Any:
    width = 8 if quick else 16
    depth = 1 if quick else 2
    layout = phx.dynamics.StateLayout((1,))
    if architecture in ("drift_only", "unstructured_diffusion"):
        drift_key, diffusion_key = jr.split(key)
        drift_model = phx.nn.models.MLP(
            in_size=1,
            out_size=1,
            width_size=width,
            depth=depth,
            key=drift_key,
        )
        system = phx.dynamics.continuous_model_system(
            drift_model,
            state_layout=layout,
            system_id=f"benchmark:{architecture}:drift",
        )
        if architecture == "drift_only":
            coefficient = _ConstantDiffusionCoefficient(
                float(jnp.sqrt(2.0 * temperature * 0.35))
            )
        else:
            coefficient = _PositiveDiffusionCoefficient(
                phx.nn.models.MLP(
                    in_size=1,
                    out_size=1,
                    width_size=width,
                    depth=depth,
                    key=diffusion_key,
                )
            )
        term = phx.solver.WienerTerm(
            "benchmark-noise",
            coefficient,
            (1,),
        )
        return phx.stochastic.EulerMaruyamaTransitionKernel(
            system,
            (term,),
            state_shape=(1,),
            noise_shape=(1,),
            process_id=f"benchmark:{architecture}",
        )
    if architecture == "structured_isothermal":
        dissipation_key, field_key = jr.split(key)
        field = phx.nn.models.PortHamiltonianVectorField(
            state_size=1,
            energy_width=width,
            energy_depth=depth,
            dissipation_model=phx.nn.models.MLP(
                in_size=1,
                out_size=1,
                width_size=width,
                depth=depth,
                key=dissipation_key,
            ),
            dissipation_structure="positive_semidefinite",
            key=field_key,
        )
        dynamics = phx.stochastic.IsothermalPortHamiltonianDynamics(
            field,
            temperature=temperature,
            process_id="benchmark:structured-isothermal",
        )
        return dynamics.transition_kernel()
    raise ValueError(f"Unsupported stochastic architecture {architecture!r}.")


def _transition_negative_log_likelihood(
    kernel: Any,
    source: jax.Array,
    target: jax.Array,
    intervals: jax.Array,
    /,
) -> jax.Array:
    context = phx.stochastic.StateSpaceStepContext.empty()
    log_density = jax.vmap(
        lambda source_, target_, interval_: kernel.log_prob(
            target_,
            source_,
            jnp.asarray(0.0),
            interval_,
            context,
        )
    )(source, target, intervals)
    return -jnp.mean(log_density)


def _stochastic_thermodynamic_record(
    architecture: str,
    /,
    *,
    seed: int,
    repeats: int,
    quick: bool,
) -> dict[str, Any]:
    temperature = 0.5
    generator = np.random.default_rng(seed)
    train_count = 80 if quick else 512
    test_count = 64 if quick else 256

    def generate(count: int, *, equilibrium: bool) -> tuple[jax.Array, ...]:
        source_scale = np.sqrt(temperature) if equilibrium else 0.9
        source = generator.normal(0.0, source_scale, size=(count, 1))
        intervals = generator.uniform(0.02, 0.08, size=(count,))
        drift = np.asarray(
            _variable_mobility_drift(
                jnp.asarray(source),
                temperature=temperature,
            )
        )
        diffusion = np.sqrt(
            2.0 * temperature * np.asarray(_variable_mobility(jnp.asarray(source)))
        )
        target = (
            source
            + intervals[:, None] * drift
            + np.sqrt(intervals)[:, None] * diffusion * generator.normal(size=(count, 1))
        )
        return jnp.asarray(source), jnp.asarray(target), jnp.asarray(intervals)

    train_source, train_target, train_intervals = generate(
        train_count,
        equilibrium=False,
    )
    test_source, test_target, test_intervals = generate(
        test_count,
        equilibrium=True,
    )
    kernel = _stochastic_kernel(
        architecture,
        key=jr.key(seed),
        quick=quick,
        temperature=temperature,
    )
    if architecture == "drift_only":
        derivative_targets = (train_target - train_source) / train_intervals[:, None]

        def training_loss(candidate):
            context = phx.stochastic.StateSpaceStepContext.empty()
            predictions = jax.vmap(
                lambda state: (
                    candidate.mean(
                        state,
                        0.0,
                        1.0,
                        context,
                    )
                    - state
                )
            )(train_source)
            return jnp.mean((predictions - derivative_targets) ** 2)

        loss_kind = "increment-regression-mse"
    else:

        def training_loss(candidate):
            return _transition_negative_log_likelihood(
                candidate,
                train_source,
                train_target,
                train_intervals,
            )

        loss_kind = "euler-maruyama-negative-log-likelihood"
    kernel, compilation_ms, training_ms, initial_loss, final_loss = _compile_and_train(
        kernel,
        training_loss,
        steps=8 if quick else 120,
        learning_rate=3e-3,
    )
    evaluate_nll = eqx.filter_jit(
        lambda candidate: _transition_negative_log_likelihood(
            candidate,
            test_source,
            test_target,
            test_intervals,
        )
    )
    held_out_nll, inference_ms, inference_std_ms = _measure(
        lambda: evaluate_nll(kernel),
        repeats=repeats,
    )
    context = phx.stochastic.StateSpaceStepContext.empty()
    unit_means = jax.vmap(lambda state: kernel.mean(state, 0.0, 1.0, context))(
        test_source
    )
    predicted_drift = unit_means - test_source
    predicted_covariance_rate = jax.vmap(
        lambda state: kernel.covariance(state, 0.0, 1.0, context)[0, 0]
    )(test_source)
    true_drift = _variable_mobility_drift(
        test_source,
        temperature=temperature,
    )
    true_covariance_rate = 2.0 * temperature * _variable_mobility(test_source[:, 0])
    drift_mse = float(jnp.mean((predicted_drift - true_drift) ** 2))
    covariance_mse = float(
        jnp.mean((predicted_covariance_rate - true_covariance_rate) ** 2)
    )
    stationary_interval = 0.04
    next_means = jax.vmap(
        lambda state: kernel.mean(
            state,
            0.0,
            stationary_interval,
            context,
        )
    )(test_source)[:, 0]
    next_variances = jax.vmap(
        lambda state: kernel.covariance(
            state,
            0.0,
            stationary_interval,
            context,
        )[0, 0]
    )(test_source)
    stationary_moment_discrepancy = float(
        jnp.abs(jnp.mean(next_means))
        + jnp.abs(jnp.mean(next_means**2 + next_variances) - temperature)
    )
    if architecture == "structured_isothermal":
        grid = jnp.linspace(-1.5, 1.5, 81)[:, None]
        energy = jax.vmap(kernel.dynamics.field.energy)(grid)
        origin = kernel.dynamics.field.energy(jnp.zeros((1,)))
        aligned_energy = energy - origin
        energy_mse = float(jnp.mean((aligned_energy - 0.5 * grid[:, 0] ** 2) ** 2))
    else:
        energy_mse = None
    held_out_nll_value = float(held_out_nll)
    passed = _finite_record(
        initial_loss,
        final_loss,
        held_out_nll_value,
        drift_mse,
        covariance_mse,
        stationary_moment_discrepancy,
        compilation_ms,
        training_ms,
        inference_ms,
    )
    if energy_mse is not None:
        passed = passed and math.isfinite(energy_mse)
    return {
        "scenario_id": "stochastic-variable-mobility",
        "architecture": architecture,
        "seed": seed,
        "loss_kind": loss_kind,
        "parameter_count": _parameter_count(kernel),
        "compilation_ms": compilation_ms,
        "training_ms": training_ms,
        "inference_mean_ms": inference_ms,
        "inference_standard_deviation_ms": inference_std_ms,
        "initial_training_loss": initial_loss,
        "final_training_loss": final_loss,
        "held_out_transition_nll": held_out_nll_value,
        "held_out_drift_mse": drift_mse,
        "held_out_covariance_rate_mse": covariance_mse,
        "stationary_moment_discrepancy": stationary_moment_discrepancy,
        "energy_mse_after_additive_alignment": energy_mse,
        "passed": passed,
    }


def _stochastic_thermodynamic_benchmark(
    *,
    architectures: tuple[str, ...],
    seeds: tuple[int, ...],
    repeats: int,
    quick: bool,
) -> dict[str, Any]:
    supported = (
        "drift_only",
        "unstructured_diffusion",
        "structured_isothermal",
    )
    selected = tuple(name for name in supported if name in architectures)
    records = [
        _stochastic_thermodynamic_record(
            architecture,
            seed=seed,
            repeats=repeats,
            quick=quick,
        )
        for seed in seeds
        for architecture in selected
    ]
    return {
        "scenario_id": "stochastic-variable-mobility",
        "records": records,
        "passed": bool(records) and all(record["passed"] for record in records),
    }


def _learned_discrete_benchmark(
    *,
    repeats: int,
    seed: int,
    quick: bool,
) -> dict[str, Any]:
    cases = 8 if quick else 32
    capacity = 6
    layout = phx.dynamics.StateLayout((1,), component_names=("x",))
    initial = jnp.linspace(-1.0, 1.0, cases, dtype=jnp.float64)[:, None]
    snapshots = [initial]
    for _ in range(capacity - 1):
        snapshots.append(0.9 * snapshots[-1] + 0.1)
    states = jnp.stack(tuple(snapshots), axis=1)
    coordinates = jnp.broadcast_to(
        jnp.arange(capacity, dtype=jnp.float64),
        (cases, capacity),
    )
    data = phx.dynamics.TrajectoryData(
        coordinates,
        states,
        state_layout=layout,
        source_id="learned-discrete-benchmark",
    )
    model = phx.nn.models.MLP(
        in_size=1,
        out_size=1,
        width_size=8 if quick else 16,
        depth=1,
        key=jax.random.key(seed),
    )
    policy = phx.dynamics.identification.DiscreteModelRolloutPolicy(
        max_horizon=3,
    )
    fit = phx.dynamics.identification.fit_discrete_model(
        model,
        data,
        state_layout=layout,
        system_id="learned-affine-map",
        step_size=1.0,
        rollout_policy=policy,
        objectives=(phx.dynamics.identification.SupervisedDiscreteModelObjective(),),
        learning_rate=2e-2,
        steps=4 if quick else 100,
        batch_size=cases,
        key=jax.random.key(seed + 1),
    )
    predicted = jax.vmap(lambda value: fit.system.evaluate(0.0, value, None))(
        states[:, 0]
    )
    one_step_error = float(
        jnp.linalg.norm(predicted - states[:, 1])
        / jnp.maximum(jnp.linalg.norm(states[:, 1]), 1e-12)
    )

    evolution = phx.dynamics.DiscreteEvolution(fit.system)
    whole = phx.dynamics.evolve(
        evolution,
        states[0, 0],
        phx.dynamics.IterationGrid(
            jnp.arange(5),
            iteration_id="learned-map-whole",
        ),
    )
    prefix = phx.dynamics.evolve(
        evolution,
        states[0, 0],
        phx.dynamics.IterationGrid(
            jnp.arange(3),
            iteration_id="learned-map-prefix",
        ),
    )
    suffix = phx.dynamics.evolve(
        evolution,
        prefix.final_state,
        phx.dynamics.IterationGrid(
            jnp.arange(2, 5),
            iteration_id="learned-map-suffix",
        ),
    )
    chunked = jnp.concatenate((prefix.states, suffix.states[1:]), axis=0)
    chunk_error = float(jnp.max(jnp.abs(whole.states - chunked)))
    _, inference_mean, inference_std = _measure(
        lambda: phx.dynamics.evolve(
            evolution,
            states[0, 0],
            phx.dynamics.IterationGrid(
                jnp.arange(5),
                iteration_id="learned-map-timing",
            ),
        ),
        repeats=repeats,
    )
    finite = _finite_record(
        fit.initial_loss,
        fit.final_loss,
        fit.training_seconds,
        one_step_error,
        chunk_error,
        inference_mean,
        inference_std,
    )
    return {
        "initial_loss": fit.initial_loss,
        "final_loss": fit.final_loss,
        "training_seconds": fit.training_seconds,
        "one_step_relative_l2": one_step_error,
        "chunk_equivalence_max_error": chunk_error,
        "inference_seconds_mean": inference_mean,
        "inference_seconds_std": inference_std,
        "completed_steps": fit.completed_steps,
        "passed": bool(finite and chunk_error <= 1e-12),
    }


def run_benchmarks(
    *,
    sparse_samples: int,
    dimension: int,
    leading_k: int,
    num_steps: int,
    repeats: int,
    seed: int,
    scenarios: Sequence[str] = ("baseline",),
    architectures: Sequence[str] = ("all",),
    seeds: Sequence[int] = (0, 1, 2),
    quick: bool = False,
) -> dict[str, Any]:
    """Benchmark baseline and opt-in thermodynamic dynamics scenarios."""
    requested_scenarios = tuple(str(value) for value in scenarios)
    if "all" in requested_scenarios:
        resolved_scenarios = ("baseline", "deterministic", "stochastic", "learned")
    elif "thermodynamic" in requested_scenarios:
        resolved_scenarios = tuple(
            dict.fromkeys(
                value
                for scenario in requested_scenarios
                for value in (
                    ("deterministic", "stochastic")
                    if scenario == "thermodynamic"
                    else (scenario,)
                )
            )
        )
    else:
        resolved_scenarios = requested_scenarios
    valid_scenarios = {"baseline", "deterministic", "stochastic", "learned"}
    if not resolved_scenarios or any(
        scenario not in valid_scenarios for scenario in resolved_scenarios
    ):
        raise ValueError(
            "scenarios must select baseline, deterministic, stochastic, learned, "
            "thermodynamic, or all."
        )
    supported_architectures = (
        "unstructured_mlp",
        "constant_port_hamiltonian",
        "state_dependent_port_hamiltonian",
        "drift_only",
        "unstructured_diffusion",
        "structured_isothermal",
    )
    requested_architectures = tuple(str(value) for value in architectures)
    resolved_architectures = (
        supported_architectures
        if "all" in requested_architectures
        else requested_architectures
    )
    if not resolved_architectures or any(
        architecture not in supported_architectures
        for architecture in resolved_architectures
    ):
        raise ValueError("architectures contains an unsupported thermodynamic model.")
    resolved_seeds = tuple(int(value) for value in seeds)
    if not resolved_seeds or any(value < 0 for value in resolved_seeds):
        raise ValueError("seeds must contain nonnegative integers.")
    device = jax.devices()[0]
    configuration: dict[str, Any] = {
        "sparse_samples": sparse_samples,
        "dimension": dimension,
        "leading_k": leading_k,
        "num_steps": num_steps,
        "repeats": repeats,
        "seed": seed,
    }
    result: dict[str, Any] = {
        "configuration": configuration,
        "environment": {
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "device_kind": device.device_kind,
            "machine": platform.machine(),
            "system": platform.system(),
            "system_release": platform.release(),
            "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        },
    }
    passed = True
    if "baseline" in resolved_scenarios:
        sparse = _sparse_recovery_benchmark(
            samples=sparse_samples,
            repeats=repeats,
            seed=seed,
        )
        matrix_free = _matrix_free_benchmark(
            dimension=dimension,
            leading_k=leading_k,
            num_steps=num_steps,
            repeats=repeats,
        )
        result["sparse_recovery"] = sparse
        result["matrix_free_analysis"] = matrix_free
        passed = passed and sparse["passed"] and matrix_free["passed"]
    if "learned" in resolved_scenarios:
        learned = _learned_discrete_benchmark(
            repeats=repeats,
            seed=seed,
            quick=quick,
        )
        result["learned_discrete_map"] = learned
        passed = passed and learned["passed"]
    thermodynamic: dict[str, Any] = {}
    if "deterministic" in resolved_scenarios:
        deterministic = _deterministic_thermodynamic_benchmark(
            architectures=resolved_architectures,
            seeds=resolved_seeds,
            repeats=repeats,
            quick=quick,
        )
        thermodynamic["deterministic_nonlinear_oscillator"] = deterministic
        passed = passed and deterministic["passed"]
    if "stochastic" in resolved_scenarios:
        stochastic = _stochastic_thermodynamic_benchmark(
            architectures=resolved_architectures,
            seeds=resolved_seeds,
            repeats=repeats,
            quick=quick,
        )
        thermodynamic["stochastic_variable_mobility"] = stochastic
        passed = passed and stochastic["passed"]
    if thermodynamic:
        configuration["scenarios"] = list(resolved_scenarios)
        configuration["architectures"] = list(resolved_architectures)
        configuration["seeds"] = list(resolved_seeds)
        configuration["quick"] = bool(quick)
        thermodynamic["passed"] = all(
            scenario["passed"]
            for scenario in thermodynamic.values()
            if isinstance(scenario, dict)
        )
        result["thermodynamic_dynamics"] = thermodynamic
    configuration["scenarios"] = list(resolved_scenarios)
    result["passed"] = bool(passed)
    return result


def _comma_values(value: str, /) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected one or more comma-separated values")
    return values


def _comma_integers(value: str, /) -> tuple[int, ...]:
    return tuple(int(item) for item in _comma_values(value))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the Phydrax nonlinear-dynamics substrate."
    )
    parser.add_argument("--sparse-samples", type=int, default=2_048)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--leading-k", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenarios",
        type=_comma_values,
        default=("baseline",),
        help=(
            "Comma-separated: baseline, deterministic, stochastic, learned, "
            "thermodynamic, all."
        ),
    )
    parser.add_argument(
        "--architectures",
        type=_comma_values,
        default=("all",),
        help="Comma-separated thermodynamic architecture names or all.",
    )
    parser.add_argument(
        "--seeds",
        type=_comma_integers,
        default=(0, 1, 2),
        help="Comma-separated thermodynamic benchmark seeds.",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(
            args.sparse_samples,
            args.dimension,
            args.leading_k,
            args.num_steps,
            args.repeats,
        )
        <= 0
    ):
        parser.error("benchmark sizes and repeats must be positive")
    if args.leading_k > args.dimension:
        parser.error("leading-k cannot exceed dimension")
    print(
        json.dumps(
            run_benchmarks(
                sparse_samples=args.sparse_samples,
                dimension=args.dimension,
                leading_k=args.leading_k,
                num_steps=args.num_steps,
                repeats=args.repeats,
                seed=args.seed,
                scenarios=args.scenarios,
                architectures=args.architectures,
                seeds=args.seeds,
                quick=args.quick,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
