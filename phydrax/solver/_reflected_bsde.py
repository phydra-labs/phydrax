#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite, prod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._strict import StrictModule
from ..stochastic._bsde import BSDEPathBatch
from ..stochastic._path_dependent_bsde import ReflectedPathDependentBSDEProblem
from ._regression_bsde import (
    _basis_matrix,
    _normal_equation_error,
    _normalise_design,
    _regress,
    AbstractBSDERegressionBasis,
    BSDERegressionScheme,
)


class ReflectedPathDependentBSDEResult(StrictModule):
    """Constrained pathwise values, reflection measures, and fitted continuations."""

    problem: ReflectedPathDependentBSDEProblem
    paths: BSDEPathBatch
    basis: AbstractBSDERegressionBasis
    values: Array
    continuation_values: Array
    controls: Array
    generator_values: Array
    lower_obstacles: Array
    upper_obstacles: Array
    lower_reflection_increments: Array
    upper_reflection_increments: Array
    local_residuals: Array
    continuation_coefficients: Array
    control_coefficients: Array
    feature_means: Array
    feature_scales: Array
    regression_masks: Array
    control_regression_masks: Array
    sample_counts: Array
    design_ranks: Array
    condition_numbers: Array
    continuation_normal_equation_errors: Array
    control_normal_equation_errors: Array
    picard_iterations: Array
    picard_errors: Array
    picard_converged: Array
    valid_steps: Array
    valid_paths: Array
    terminal_compatible: Array
    scheme: BSDERegressionScheme = eqx.field(static=True)
    ridge: float = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (
            jnp.any(self.valid_paths)
            & jnp.all(self.valid_steps)
            & jnp.all(self.picard_converged)
        )


class ReflectedPathDependentBSDEDiagnostics(StrictModule):
    """Obstacle, complementarity, regression, and equation diagnostics."""

    local_equation_rmse: Array
    lower_constraint_violation: Array
    upper_constraint_violation: Array
    lower_complementarity_error: Array
    upper_complementarity_error: Array
    mean_lower_reflection: Array
    mean_upper_reflection: Array
    valid_path_fraction: Array
    terminal_compatibility_fraction: Array
    all_regressions_valid: Array
    all_picard_steps_converged: Array
    finite: Array

    @property
    def passed(self) -> bool:
        return bool(
            self.finite
            & self.all_regressions_valid
            & self.all_picard_steps_converged
            & (self.valid_path_fraction > 0.0)
            & (self.lower_constraint_violation == 0.0)
            & (self.upper_constraint_violation == 0.0)
            & (self.lower_complementarity_error == 0.0)
            & (self.upper_complementarity_error == 0.0)
        )


def _path_features(
    problem: ReflectedPathDependentBSDEProblem,
    prefix_times: Array,
    histories: Array,
    /,
) -> Array:
    def evaluate(history):
        features = jnp.asarray(problem.path_features(prefix_times, history, problem.args))
        if features.shape != problem.regression_state_shape:
            raise ValueError(
                "path_features returned an incompatible regression state shape."
            )
        return features

    return jax.vmap(evaluate)(histories)


def _path_values(
    function: Callable,
    problem: ReflectedPathDependentBSDEProblem,
    time: Array,
    prefix_times: Array,
    histories: Array,
    /,
    *,
    owner: str,
) -> Array:
    def evaluate(history):
        value = jnp.asarray(function(time, prefix_times, history, problem.args))
        if value.shape != problem.output_shape:
            raise ValueError(f"{owner} returned an incompatible output shape.")
        return value

    return jax.vmap(evaluate)(histories)


def _terminal_values(
    problem: ReflectedPathDependentBSDEProblem,
    times: Array,
    histories: Array,
    /,
) -> Array:
    def evaluate(history):
        value = jnp.asarray(problem.terminal(times, history, problem.args))
        if value.shape != problem.output_shape:
            raise ValueError("terminal returned an incompatible output shape.")
        return value

    return jax.vmap(evaluate)(histories)


def _generator_values(
    problem: ReflectedPathDependentBSDEProblem,
    time: Array,
    prefix_times: Array,
    histories: Array,
    values: Array,
    controls: Array,
    /,
) -> Array:
    def evaluate(history, value, control):
        output = jnp.asarray(
            problem.generator(
                time,
                prefix_times,
                history,
                value,
                control,
                problem.args,
            )
        )
        if output.shape != problem.output_shape:
            raise ValueError("generator returned an incompatible output shape.")
        return output

    return jax.vmap(evaluate)(histories, values, controls)


def _project_obstacles(
    continuation: Array,
    lower: Array,
    upper: Array,
    /,
    *,
    has_lower: bool,
    has_upper: bool,
) -> tuple[Array, Array, Array]:
    constrained = continuation
    if has_lower:
        lower_push = jnp.maximum(lower - constrained, 0.0)
        constrained = jnp.maximum(constrained, lower)
    else:
        lower_push = jnp.zeros_like(continuation)
    if has_upper:
        upper_push = jnp.maximum(constrained - upper, 0.0)
        constrained = jnp.minimum(constrained, upper)
    else:
        upper_push = jnp.zeros_like(continuation)
    return constrained, lower_push, upper_push


def _obstacle_data(
    problem: ReflectedPathDependentBSDEProblem,
    time: Array,
    prefix_times: Array,
    histories: Array,
    /,
) -> tuple[Array, Array, Array]:
    count = histories.shape[0]
    output_shape = (count,) + problem.output_shape
    if problem.lower_obstacle is None:
        lower = jnp.zeros(output_shape, dtype=histories.dtype)
        lower_finite = jnp.ones((count,), dtype=bool)
    else:
        lower = _path_values(
            problem.lower_obstacle,
            problem,
            time,
            prefix_times,
            histories,
            owner="lower_obstacle",
        )
        lower_finite = jnp.all(jnp.isfinite(lower), axis=tuple(range(1, lower.ndim)))
    if problem.upper_obstacle is None:
        upper = jnp.zeros(output_shape, dtype=histories.dtype)
        upper_finite = jnp.ones((count,), dtype=bool)
    else:
        upper = _path_values(
            problem.upper_obstacle,
            problem,
            time,
            prefix_times,
            histories,
            owner="upper_obstacle",
        )
        upper_finite = jnp.all(jnp.isfinite(upper), axis=tuple(range(1, upper.ndim)))
    ordered = (
        jnp.all(lower <= upper, axis=tuple(range(1, lower.ndim)))
        if problem.has_lower_obstacle and problem.has_upper_obstacle
        else jnp.ones((count,), dtype=bool)
    )
    return lower, upper, lower_finite & upper_finite & ordered


def solve_reflected_path_dependent_bsde(
    problem: ReflectedPathDependentBSDEProblem,
    basis: AbstractBSDERegressionBasis,
    /,
    *,
    paths: BSDEPathBatch | None = None,
    key: Key[Array, ""] | None = None,
    scheme: BSDERegressionScheme = "explicit",
    ridge: float = 1e-8,
    standardize: bool = True,
    rcond: float = 1e-10,
    min_samples: int | None = None,
    max_picard_steps: int = 32,
    picard_tolerance: float = 1e-8,
    picard_damping: float = 1.0,
    raise_on_failure: bool = False,
) -> ReflectedPathDependentBSDEResult:
    """Solve a singly or doubly reflected path-dependent BSDE by regression.

    Obstacle projection is applied inside the implicit Picard map. The returned lower
    and upper reflection increments obey the discrete complementarity conditions by
    construction; terminal data is never silently projected onto an incompatible
    obstacle.
    """
    if not isinstance(problem, ReflectedPathDependentBSDEProblem):
        raise TypeError("problem must be a ReflectedPathDependentBSDEProblem.")
    if not isinstance(basis, AbstractBSDERegressionBasis):
        raise TypeError("basis must implement AbstractBSDERegressionBasis.")
    if basis.state_shape != problem.regression_state_shape:
        raise ValueError("basis state_shape must match regression_state_shape.")
    if scheme not in ("explicit", "implicit"):
        raise ValueError("scheme must be 'explicit' or 'implicit'.")
    ridge_value = float(ridge)
    rcond_value = float(rcond)
    tolerance = float(picard_tolerance)
    damping = float(picard_damping)
    if not isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError("ridge must be finite and nonnegative.")
    if not isfinite(rcond_value) or rcond_value <= 0.0:
        raise ValueError("rcond must be finite and positive.")
    if not isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("picard_tolerance must be finite and positive.")
    if not isfinite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError("picard_damping must lie in (0, 1].")
    picard_steps = int(max_picard_steps)
    if picard_steps < 1:
        raise ValueError("max_picard_steps must be positive.")
    required_samples = basis.num_features if min_samples is None else int(min_samples)
    if required_samples < 1:
        raise ValueError("min_samples must be positive.")

    resolved_paths = (
        problem.sample(jr.key(0) if key is None else key) if paths is None else paths
    )
    if not isinstance(resolved_paths, BSDEPathBatch):
        raise TypeError("paths must be a BSDEPathBatch or None.")
    if (
        resolved_paths.state_shape != problem.state_shape
        or resolved_paths.noise_shape != problem.noise_shape
        or resolved_paths.process_id != problem.process_id
    ):
        raise ValueError("paths do not match the reflected BSDE contract.")
    if resolved_paths.jump_events:
        raise ValueError("Reflected Brownian BSDE schemes do not consume jump events.")

    num_paths = resolved_paths.num_paths
    num_steps = resolved_paths.num_steps
    output_size = prod(problem.output_shape)
    noise_size = prod(problem.noise_shape)
    states = resolved_paths.states.reshape(
        (num_paths, num_steps + 1) + problem.state_shape
    )
    increments = resolved_paths.wiener_increments.reshape(
        (num_paths, num_steps, noise_size)
    )
    path_valid = resolved_paths.path_valid.reshape((num_paths,))

    regression_states: list[Array] = []
    designs: list[Array] = []
    lower_obstacles: list[Array] = []
    upper_obstacles: list[Array] = []
    obstacle_validity: list[Array] = []
    for node in range(num_steps + 1):
        prefix_times = resolved_paths.times[: node + 1]
        histories = states[:, : node + 1]
        regression_state = _path_features(problem, prefix_times, histories)
        regression_states.append(regression_state)
        designs.append(_basis_matrix(basis, regression_state))
        lower, upper, obstacle_valid = _obstacle_data(
            problem,
            resolved_paths.times[node],
            prefix_times,
            histories,
        )
        lower_obstacles.append(lower.reshape((num_paths, output_size)))
        upper_obstacles.append(upper.reshape((num_paths, output_size)))
        obstacle_validity.append(obstacle_valid)

    terminal_target = _terminal_values(problem, resolved_paths.times, states).reshape(
        (num_paths, output_size)
    )
    terminal_finite = jnp.all(jnp.isfinite(terminal_target), axis=-1)
    terminal_compatible = terminal_finite & obstacle_validity[-1]
    if problem.has_lower_obstacle:
        terminal_compatible = terminal_compatible & jnp.all(
            terminal_target >= lower_obstacles[-1], axis=-1
        )
    if problem.has_upper_obstacle:
        terminal_compatible = terminal_compatible & jnp.all(
            terminal_target <= upper_obstacles[-1], axis=-1
        )
    all_obstacles_valid = jnp.all(jnp.stack(obstacle_validity, axis=-1), axis=-1)
    eligible = path_valid & all_obstacles_valid & terminal_compatible

    node_values: list[Array | None] = [None] * (num_steps + 1)
    continuations: list[Array | None] = [None] * (num_steps + 1)
    controls: list[Array | None] = [None] * num_steps
    generators: list[Array | None] = [None] * num_steps
    lower_reflections: list[Array | None] = [None] * num_steps
    upper_reflections: list[Array | None] = [None] * num_steps
    local_residuals: list[Array | None] = [None] * num_steps
    continuation_coefficients: list[Array | None] = [None] * (num_steps + 1)
    control_coefficients: list[Array | None] = [None] * num_steps
    feature_means: list[Array | None] = [None] * (num_steps + 1)
    feature_scales: list[Array | None] = [None] * (num_steps + 1)
    regression_masks: list[Array | None] = [None] * (num_steps + 1)
    control_masks: list[Array | None] = [None] * num_steps
    sample_counts: list[Array | None] = [None] * (num_steps + 1)
    design_ranks: list[Array | None] = [None] * (num_steps + 1)
    condition_numbers: list[Array | None] = [None] * (num_steps + 1)
    continuation_normal_errors: list[Array | None] = [None] * (num_steps + 1)
    control_normal_errors: list[Array | None] = [None] * num_steps
    picard_iterations: list[Array | None] = [None] * num_steps
    picard_errors: list[Array | None] = [None] * num_steps
    picard_converged: list[Array | None] = [None] * num_steps
    valid_steps: list[Array | None] = [None] * (num_steps + 1)

    (
        terminal_normalized,
        terminal_mean,
        terminal_scale,
        terminal_base_mask,
        _terminal_design_count,
        terminal_rank,
        terminal_condition,
    ) = _normalise_design(
        designs[-1],
        eligible,
        standardize=standardize,
        rcond=rcond_value,
    )
    (
        terminal_coefficients,
        _terminal_prediction,
        _terminal_residual,
        terminal_mask,
        terminal_count,
        terminal_normal_error,
        terminal_valid,
    ) = _regress(
        terminal_normalized,
        terminal_target,
        terminal_base_mask,
        ridge=ridge_value,
        rcond=rcond_value,
        min_samples=required_samples,
    )
    node_values[-1] = terminal_target
    continuations[-1] = terminal_target
    continuation_coefficients[-1] = terminal_coefficients
    feature_means[-1] = terminal_mean
    feature_scales[-1] = terminal_scale
    regression_masks[-1] = terminal_mask
    sample_counts[-1] = terminal_count
    design_ranks[-1] = terminal_rank
    condition_numbers[-1] = terminal_condition
    continuation_normal_errors[-1] = terminal_normal_error
    valid_steps[-1] = terminal_valid

    for step in range(num_steps - 1, -1, -1):
        next_value = node_values[step + 1]
        if next_value is None:
            raise RuntimeError("Backward reflected recursion lost its next-node value.")
        time = resolved_paths.times[step]
        dt = resolved_paths.times[step + 1] - time
        prefix_times = resolved_paths.times[: step + 1]
        histories = states[:, : step + 1]
        lower = lower_obstacles[step]
        upper = upper_obstacles[step]
        (
            normalized,
            mean,
            scale,
            base_mask,
            design_count,
            rank,
            condition,
        ) = _normalise_design(
            designs[step],
            eligible,
            standardize=standardize,
            rcond=rcond_value,
        )
        z_target = (
            next_value.reshape((num_paths, output_size, 1))
            * increments[:, step, None, :]
            / dt
        ).reshape((num_paths, output_size * noise_size))
        (
            z_coefficients,
            z_flat,
            _z_residual,
            z_mask,
            z_count,
            z_normal_error,
            z_valid,
        ) = _regress(
            normalized,
            z_target,
            base_mask,
            ridge=ridge_value,
            rcond=rcond_value,
            min_samples=required_samples,
        )
        z_value = z_flat.reshape(
            (num_paths,) + problem.output_shape + problem.noise_shape
        )

        if scheme == "explicit":
            generator = _generator_values(
                problem,
                time,
                prefix_times,
                histories,
                next_value.reshape((num_paths,) + problem.output_shape),
                z_value,
            ).reshape((num_paths, output_size))
            continuation_target = next_value + dt * generator
            (
                y_coefficients,
                continuation,
                _continuation_residual,
                y_mask,
                y_count,
                y_normal_error,
                y_valid,
            ) = _regress(
                normalized,
                continuation_target,
                base_mask,
                ridge=ridge_value,
                rcond=rcond_value,
                min_samples=required_samples,
            )
            value, lower_push, upper_push = _project_obstacles(
                continuation,
                lower,
                upper,
                has_lower=problem.has_lower_obstacle,
                has_upper=problem.has_upper_obstacle,
            )
            iterations = jnp.asarray(1, dtype=jnp.int32)
            iteration_error = jnp.asarray(0.0, dtype=value.dtype)
            converged = jnp.asarray(True)
        else:
            (
                y_coefficients,
                continuation,
                _,
                _,
                _,
                _,
                initial_valid,
            ) = _regress(
                normalized,
                next_value,
                base_mask,
                ridge=ridge_value,
                rcond=rcond_value,
                min_samples=required_samples,
            )
            value, lower_push, upper_push = _project_obstacles(
                continuation,
                lower,
                upper,
                has_lower=problem.has_lower_obstacle,
                has_upper=problem.has_upper_obstacle,
            )
            converged = jnp.asarray(False)
            iterations = jnp.asarray(picard_steps, dtype=jnp.int32)
            iteration_error = jnp.asarray(jnp.inf, dtype=value.dtype)
            continuation_target = next_value
            y_mask = base_mask
            y_count = design_count
            y_normal_error = jnp.asarray(jnp.inf, dtype=value.dtype)
            y_valid = initial_valid
            generator = jnp.zeros_like(next_value)
            for iteration in range(picard_steps):
                candidate_generator = _generator_values(
                    problem,
                    time,
                    prefix_times,
                    histories,
                    value.reshape((num_paths,) + problem.output_shape),
                    z_value,
                ).reshape((num_paths, output_size))
                candidate_target = next_value + dt * candidate_generator
                (
                    candidate_coefficients,
                    _candidate_continuation,
                    _,
                    candidate_mask,
                    candidate_count,
                    candidate_normal_error,
                    candidate_valid,
                ) = _regress(
                    normalized,
                    candidate_target,
                    base_mask,
                    ridge=ridge_value,
                    rcond=rcond_value,
                    min_samples=required_samples,
                )
                damped_coefficients = (
                    1.0 - damping
                ) * y_coefficients + damping * candidate_coefficients
                damped_continuation = normalized @ damped_coefficients
                candidate_value, candidate_lower_push, candidate_upper_push = (
                    _project_obstacles(
                        damped_continuation,
                        lower,
                        upper,
                        has_lower=problem.has_lower_obstacle,
                        has_upper=problem.has_upper_obstacle,
                    )
                )
                difference = jnp.where(
                    candidate_mask[:, None], candidate_value - value, 0.0
                )
                denominator = jnp.maximum(
                    jnp.sqrt(
                        jnp.sum(
                            jnp.where(candidate_mask[:, None], candidate_value**2, 0.0)
                        )
                        / jnp.maximum(jnp.sum(candidate_mask) * output_size, 1)
                    ),
                    1.0,
                )
                candidate_error = (
                    jnp.sqrt(
                        jnp.sum(difference**2)
                        / jnp.maximum(jnp.sum(candidate_mask) * output_size, 1)
                    )
                    / denominator
                )
                active = ~converged
                newly_converged = active & (candidate_error <= tolerance)
                iterations = jnp.where(
                    newly_converged,
                    jnp.asarray(iteration + 1, dtype=jnp.int32),
                    iterations,
                )
                y_coefficients = jnp.where(active, damped_coefficients, y_coefficients)
                continuation = jnp.where(active, damped_continuation, continuation)
                value = jnp.where(active, candidate_value, value)
                lower_push = jnp.where(active, candidate_lower_push, lower_push)
                upper_push = jnp.where(active, candidate_upper_push, upper_push)
                continuation_target = jnp.where(
                    active, candidate_target, continuation_target
                )
                y_mask = jnp.where(active, candidate_mask, y_mask)
                y_count = jnp.where(active, candidate_count, y_count)
                y_normal_error = jnp.where(active, candidate_normal_error, y_normal_error)
                y_valid = jnp.where(active, candidate_valid, y_valid)
                generator = jnp.where(active, candidate_generator, generator)
                iteration_error = jnp.where(active, candidate_error, iteration_error)
                converged = converged | newly_converged
            generator = _generator_values(
                problem,
                time,
                prefix_times,
                histories,
                value.reshape((num_paths,) + problem.output_shape),
                z_value,
            ).reshape((num_paths, output_size))
            continuation_target = next_value + dt * generator
            y_normal_error = _normal_equation_error(
                normalized,
                continuation_target,
                continuation,
                y_mask,
                y_coefficients,
                ridge_value,
            )

        martingale = jnp.einsum(
            "pon,pn->po",
            z_value.reshape((num_paths, output_size, noise_size)),
            increments[:, step],
        )
        equation_residual = (
            next_value - value + dt * generator + lower_push - upper_push - martingale
        )
        node_values[step] = value
        continuations[step] = continuation
        controls[step] = z_value.reshape((num_paths, output_size * noise_size))
        generators[step] = generator
        lower_reflections[step] = lower_push
        upper_reflections[step] = upper_push
        local_residuals[step] = equation_residual
        continuation_coefficients[step] = y_coefficients
        control_coefficients[step] = z_coefficients
        feature_means[step] = mean
        feature_scales[step] = scale
        regression_masks[step] = y_mask
        control_masks[step] = z_mask
        sample_counts[step] = jnp.minimum(y_count, z_count)
        design_ranks[step] = rank
        condition_numbers[step] = condition
        continuation_normal_errors[step] = y_normal_error
        control_normal_errors[step] = z_normal_error
        picard_iterations[step] = iterations
        picard_errors[step] = iteration_error
        picard_converged[step] = converged
        valid_steps[step] = y_valid & z_valid

    def stack_paths(items: list[Array | None], *, owner: str, axis: int = 1) -> Array:
        arrays = [item for item in items if item is not None]
        if len(arrays) != len(items):
            raise RuntimeError(f"Incomplete reflected recursion for {owner}.")
        return jnp.stack(arrays, axis=axis)

    values_array = stack_paths(node_values, owner="values").reshape(
        resolved_paths.sample_shape + (num_steps + 1,) + problem.output_shape
    )
    continuation_array = stack_paths(continuations, owner="continuation values").reshape(
        resolved_paths.sample_shape + (num_steps + 1,) + problem.output_shape
    )
    controls_array = stack_paths(controls, owner="controls").reshape(
        resolved_paths.sample_shape
        + (num_steps,)
        + problem.output_shape
        + problem.noise_shape
    )
    generators_array = stack_paths(generators, owner="generators").reshape(
        resolved_paths.sample_shape + (num_steps,) + problem.output_shape
    )
    lower_reflection_array = stack_paths(
        lower_reflections, owner="lower reflection increments"
    ).reshape(resolved_paths.sample_shape + (num_steps,) + problem.output_shape)
    upper_reflection_array = stack_paths(
        upper_reflections, owner="upper reflection increments"
    ).reshape(resolved_paths.sample_shape + (num_steps,) + problem.output_shape)
    local_array = stack_paths(local_residuals, owner="local residuals").reshape(
        resolved_paths.sample_shape + (num_steps,) + problem.output_shape
    )
    regression_mask_array = stack_paths(
        regression_masks, owner="regression masks"
    ).reshape(resolved_paths.sample_shape + (num_steps + 1,))
    control_mask_array = stack_paths(control_masks, owner="control masks").reshape(
        resolved_paths.sample_shape + (num_steps,)
    )
    lower_array = jnp.stack(lower_obstacles, axis=1).reshape(
        resolved_paths.sample_shape + (num_steps + 1,) + problem.output_shape
    )
    upper_array = jnp.stack(upper_obstacles, axis=1).reshape(
        resolved_paths.sample_shape + (num_steps + 1,) + problem.output_shape
    )
    result = ReflectedPathDependentBSDEResult(
        problem=problem,
        paths=resolved_paths,
        basis=basis,
        values=values_array,
        continuation_values=continuation_array,
        controls=controls_array,
        generator_values=generators_array,
        lower_obstacles=lower_array,
        upper_obstacles=upper_array,
        lower_reflection_increments=lower_reflection_array,
        upper_reflection_increments=upper_reflection_array,
        local_residuals=local_array,
        continuation_coefficients=stack_paths(
            continuation_coefficients, owner="continuation coefficients", axis=0
        ),
        control_coefficients=stack_paths(
            control_coefficients, owner="control coefficients", axis=0
        ),
        feature_means=stack_paths(feature_means, owner="feature means", axis=0),
        feature_scales=stack_paths(feature_scales, owner="feature scales", axis=0),
        regression_masks=regression_mask_array,
        control_regression_masks=control_mask_array,
        sample_counts=stack_paths(sample_counts, owner="sample counts", axis=0),
        design_ranks=stack_paths(design_ranks, owner="design ranks", axis=0),
        condition_numbers=stack_paths(
            condition_numbers, owner="condition numbers", axis=0
        ),
        continuation_normal_equation_errors=stack_paths(
            continuation_normal_errors,
            owner="continuation normal-equation errors",
            axis=0,
        ),
        control_normal_equation_errors=stack_paths(
            control_normal_errors, owner="control normal-equation errors", axis=0
        ),
        picard_iterations=stack_paths(
            picard_iterations, owner="Picard iterations", axis=0
        ),
        picard_errors=stack_paths(picard_errors, owner="Picard errors", axis=0),
        picard_converged=stack_paths(
            picard_converged, owner="Picard convergence", axis=0
        ),
        valid_steps=stack_paths(valid_steps, owner="valid steps", axis=0),
        valid_paths=eligible.reshape(resolved_paths.sample_shape),
        terminal_compatible=terminal_compatible.reshape(resolved_paths.sample_shape),
        scheme=scheme,
        ridge=ridge_value,
    )
    if raise_on_failure and not bool(result.successful):
        raise RuntimeError("Reflected path-dependent BSDE regression failed validation.")
    return result


def _prediction_histories(
    result: ReflectedPathDependentBSDEResult,
    step: int,
    histories: ArrayLike,
    /,
) -> tuple[Array, tuple[int, ...]]:
    values = jnp.asarray(histories)
    state_rank = len(result.problem.state_shape)
    required_suffix = (step + 1,) + result.problem.state_shape
    if values.ndim < state_rank + 1 or values.shape[-state_rank - 1 :] != required_suffix:
        raise ValueError("histories must end with (step + 1,) + state_shape.")
    batch_shape = values.shape[: -state_rank - 1]
    return values.reshape((-1,) + required_suffix), batch_shape


def predict_reflected_path_dependent_value(
    result: ReflectedPathDependentBSDEResult,
    step: int,
    histories: ArrayLike,
    /,
) -> Array:
    """Evaluate the constrained fitted value for one or more path prefixes."""
    if not isinstance(result, ReflectedPathDependentBSDEResult):
        raise TypeError("result must be a ReflectedPathDependentBSDEResult.")
    index = int(step)
    if index < 0 or index > result.paths.num_steps:
        raise ValueError("step is outside the reflected BSDE node range.")
    history_values, batch_shape = _prediction_histories(result, index, histories)
    prefix_times = result.paths.times[: index + 1]
    if index == result.paths.num_steps:
        terminal = _terminal_values(result.problem, prefix_times, history_values)
        return terminal.reshape(batch_shape + result.problem.output_shape)
    regression_states = _path_features(result.problem, prefix_times, history_values)
    design = _basis_matrix(result.basis, regression_states)
    normalized = (design - result.feature_means[index]) / result.feature_scales[index]
    continuation = normalized @ result.continuation_coefficients[index]
    lower, upper, valid = _obstacle_data(
        result.problem,
        result.paths.times[index],
        prefix_times,
        history_values,
    )
    if not bool(jnp.all(valid)):
        raise ValueError("Prediction histories produced invalid or crossed obstacles.")
    constrained, _, _ = _project_obstacles(
        continuation,
        lower.reshape(continuation.shape),
        upper.reshape(continuation.shape),
        has_lower=result.problem.has_lower_obstacle,
        has_upper=result.problem.has_upper_obstacle,
    )
    return constrained.reshape(batch_shape + result.problem.output_shape)


def predict_reflected_path_dependent_control(
    result: ReflectedPathDependentBSDEResult,
    step: int,
    histories: ArrayLike,
    /,
) -> Array:
    """Evaluate the fitted Brownian control from path-prefix features."""
    if not isinstance(result, ReflectedPathDependentBSDEResult):
        raise TypeError("result must be a ReflectedPathDependentBSDEResult.")
    index = int(step)
    if index < 0 or index >= result.paths.num_steps:
        raise ValueError("step is outside the reflected BSDE interval range.")
    history_values, batch_shape = _prediction_histories(result, index, histories)
    regression_states = _path_features(
        result.problem, result.paths.times[: index + 1], history_values
    )
    design = _basis_matrix(result.basis, regression_states)
    normalized = (design - result.feature_means[index]) / result.feature_scales[index]
    controls = normalized @ result.control_coefficients[index]
    return controls.reshape(
        batch_shape + result.problem.output_shape + result.problem.noise_shape
    )


def _masked_local_rmse(result: ReflectedPathDependentBSDEResult, /) -> Array:
    mask = result.control_regression_masks & result.regression_masks[..., :-1]
    sample_rank = mask.ndim - 1
    event_axes = tuple(
        range(
            result.local_residuals.ndim - len(result.problem.output_shape),
            result.local_residuals.ndim,
        )
    )
    squared = jnp.abs(result.local_residuals) ** 2
    if event_axes:
        squared = jnp.mean(squared, axis=event_axes)
    sample_axes = tuple(range(sample_rank))
    count = jnp.maximum(jnp.sum(mask, axis=sample_axes), 1)
    return jnp.sqrt(jnp.sum(jnp.where(mask, squared, 0.0), axis=sample_axes) / count)


def reflected_path_dependent_bsde_diagnostics(
    result: ReflectedPathDependentBSDEResult,
    /,
) -> ReflectedPathDependentBSDEDiagnostics:
    """Check hard constraints, reflection complementarity, and solver validity."""
    if not isinstance(result, ReflectedPathDependentBSDEResult):
        raise TypeError("result must be a ReflectedPathDependentBSDEResult.")
    valid = result.valid_paths
    event_rank = len(result.problem.output_shape)
    node_mask = valid[..., None].reshape(valid.shape + (1,) * (event_rank + 1))
    interval_mask = valid[..., None].reshape(valid.shape + (1,) * (event_rank + 1))
    prefix = (slice(None),) * len(result.paths.sample_shape)
    suffix = (slice(None),) * event_rank
    current = prefix + (slice(None, -1),) + suffix
    current_values = result.values[current]
    current_lower = result.lower_obstacles[current]
    current_upper = result.upper_obstacles[current]
    if result.problem.has_lower_obstacle:
        lower_violation = jnp.max(
            jnp.where(
                node_mask,
                jnp.maximum(result.lower_obstacles - result.values, 0.0),
                0.0,
            ),
            initial=0.0,
        )
        lower_contact = jnp.max(
            jnp.where(
                interval_mask,
                jnp.abs(
                    result.lower_reflection_increments * (current_values - current_lower)
                ),
                0.0,
            ),
            initial=0.0,
        )
    else:
        lower_violation = jnp.asarray(0.0)
        lower_contact = jnp.asarray(0.0)
    if result.problem.has_upper_obstacle:
        upper_violation = jnp.max(
            jnp.where(
                node_mask,
                jnp.maximum(result.values - result.upper_obstacles, 0.0),
                0.0,
            ),
            initial=0.0,
        )
        upper_contact = jnp.max(
            jnp.where(
                interval_mask,
                jnp.abs(
                    result.upper_reflection_increments * (current_upper - current_values)
                ),
                0.0,
            ),
            initial=0.0,
        )
    else:
        upper_violation = jnp.asarray(0.0)
        upper_contact = jnp.asarray(0.0)
    finite = (
        jnp.all(jnp.isfinite(result.values))
        & jnp.all(jnp.isfinite(result.continuation_values))
        & jnp.all(jnp.isfinite(result.controls))
        & jnp.all(jnp.isfinite(result.lower_reflection_increments))
        & jnp.all(jnp.isfinite(result.upper_reflection_increments))
        & jnp.all(result.lower_reflection_increments >= 0.0)
        & jnp.all(result.upper_reflection_increments >= 0.0)
    )
    path_valid = result.paths.path_valid
    path_count = jnp.maximum(jnp.sum(path_valid), 1)
    terminal_fraction = jnp.sum(result.terminal_compatible & path_valid) / path_count
    return ReflectedPathDependentBSDEDiagnostics(
        local_equation_rmse=_masked_local_rmse(result),
        lower_constraint_violation=lower_violation,
        upper_constraint_violation=upper_violation,
        lower_complementarity_error=lower_contact,
        upper_complementarity_error=upper_contact,
        mean_lower_reflection=jnp.mean(result.lower_reflection_increments),
        mean_upper_reflection=jnp.mean(result.upper_reflection_increments),
        valid_path_fraction=jnp.mean(result.valid_paths),
        terminal_compatibility_fraction=terminal_fraction,
        all_regressions_valid=jnp.all(result.valid_steps),
        all_picard_steps_converged=jnp.all(result.picard_converged),
        finite=finite,
    )


__all__ = [
    "predict_reflected_path_dependent_control",
    "predict_reflected_path_dependent_value",
    "reflected_path_dependent_bsde_diagnostics",
    "ReflectedPathDependentBSDEDiagnostics",
    "ReflectedPathDependentBSDEResult",
    "solve_reflected_path_dependent_bsde",
]
