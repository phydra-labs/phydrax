#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from itertools import product
from math import isfinite, prod
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, Key

from .._numerics import (
    normalize_least_squares_design,
    solve_normalized_least_squares,
)
from .._strict import StrictModule
from ..stochastic._bsde import BSDEPathBatch, BSDEProblem


BSDERegressionScheme: TypeAlias = Literal["explicit", "implicit"]


def _shape(value: Sequence[int], /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(f"{owner} must contain positive dimensions.")
    return shape


def _name(value: str, /, *, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


class AbstractBSDERegressionBasis(StrictModule):
    """Finite feature map used for conditional least-squares projections."""

    state_shape: tuple[int, ...] = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def __call__(self, state: ArrayLike, /) -> Array:
        raise NotImplementedError


class CallableBSDERegressionBasis(AbstractBSDERegressionBasis):
    """Shape-checked adapter for a user-defined state feature map."""

    function: Callable[[Array], Array]

    def __init__(
        self,
        function: Callable[[Array], Array],
        /,
        *,
        state_shape: Sequence[int],
        num_features: int,
        basis_id: str,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        shape = _shape(state_shape, owner="state_shape")
        count = int(num_features)
        if count < 1:
            raise ValueError("num_features must be positive.")
        probe = jnp.asarray(function(jnp.zeros(shape)))
        if probe.shape != (count,):
            raise ValueError(f"function must return shape {(count,)}; got {probe.shape}.")
        self.function = function
        self.state_shape = shape
        self.num_features = count
        self.basis_id = _name(basis_id, owner="basis_id")

    def __call__(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError("state must have exactly the basis state_shape.")
        features = jnp.asarray(self.function(value))
        if features.shape != (self.num_features,):
            raise ValueError("The callable basis returned an incompatible feature shape.")
        return features


class PolynomialBSDERegressionBasis(AbstractBSDERegressionBasis):
    """Total-degree monomial basis over a flattened finite-dimensional state."""

    degree: int = eqx.field(static=True)
    exponents: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        state_shape: Sequence[int],
        degree: int,
        /,
        *,
        basis_id: str | None = None,
    ):
        shape = _shape(state_shape, owner="state_shape")
        resolved_degree = int(degree)
        if resolved_degree < 0:
            raise ValueError("degree must be nonnegative.")
        state_size = prod(shape)
        exponents = tuple(
            powers
            for powers in product(range(resolved_degree + 1), repeat=state_size)
            if sum(powers) <= resolved_degree
        )
        self.state_shape = shape
        self.degree = resolved_degree
        self.exponents = exponents
        self.num_features = len(exponents)
        self.basis_id = (
            f"total-degree-{resolved_degree}:{'x'.join(map(str, shape))}"
            if basis_id is None
            else _name(basis_id, owner="basis_id")
        )

    def __call__(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError("state must have exactly the basis state_shape.")
        flat = value.reshape((-1,))
        powers = jnp.asarray(self.exponents, dtype=jnp.int32)
        factors = jnp.where(powers == 0, 1.0, flat[None, :] ** powers)
        return jnp.prod(factors, axis=-1)


class LeastSquaresBSDEResult(StrictModule):
    """Pathwise backward solution and auditable conditional regressions."""

    paths: BSDEPathBatch
    basis: AbstractBSDERegressionBasis
    values: Array
    controls: Array
    generator_values: Array
    local_residuals: Array
    value_targets: Array
    control_targets: Array
    value_regression_residuals: Array
    control_regression_residuals: Array
    value_coefficients: Array
    control_coefficients: Array
    feature_means: Array
    feature_scales: Array
    regression_masks: Array
    control_regression_masks: Array
    sample_counts: Array
    design_ranks: Array
    condition_numbers: Array
    value_normal_equation_errors: Array
    control_normal_equation_errors: Array
    picard_iterations: Array
    picard_errors: Array
    picard_converged: Array
    valid_steps: Array
    valid_paths: Array
    problem_id: str = eqx.field(static=True)
    scheme: BSDERegressionScheme = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return (
            jnp.any(self.valid_paths)
            & jnp.all(self.valid_steps)
            & jnp.all(self.picard_converged)
        )


class LeastSquaresBSDEDiagnostics(StrictModule):
    """Accuracy, conditioning, and nonlinear-iteration diagnostics."""

    value_regression_rmse: Array
    control_regression_rmse: Array
    local_equation_rmse: Array
    valid_path_fraction: Array
    max_value_normal_equation_error: Array
    max_control_normal_equation_error: Array
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
        )


def _basis_matrix(
    basis: AbstractBSDERegressionBasis,
    states: Array,
    /,
) -> Array:
    flat = states.reshape((-1,) + basis.state_shape)
    matrix = jax.vmap(basis)(flat)
    expected = (flat.shape[0], basis.num_features)
    if matrix.shape != expected:
        raise ValueError(f"basis returned shape {matrix.shape}; expected {expected}.")
    return matrix


def _normalise_design(
    design: Array,
    mask: Array,
    /,
    *,
    standardize: bool,
    rcond: float,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    normalized = normalize_least_squares_design(
        design,
        mask=mask,
        center=standardize,
        scale=standardize,
        rcond=rcond,
    )
    return (
        normalized.values,
        normalized.offset,
        normalized.scale,
        normalized.valid_rows,
        normalized.sample_count,
        normalized.rank,
        normalized.condition_number,
    )


def _regress(
    normalized_design: Array,
    target: Array,
    base_mask: Array,
    /,
    *,
    ridge: float,
    rcond: float,
    min_samples: int,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    design = normalize_least_squares_design(
        normalized_design,
        mask=base_mask,
        rcond=rcond,
    )
    result = solve_normalized_least_squares(
        design,
        target,
        ridge=ridge,
        rcond=rcond,
        min_samples=min_samples,
    )
    return (
        result.coefficients,
        result.prediction,
        result.residual,
        result.valid_rows,
        result.sample_count,
        result.normal_equation_error,
        result.valid,
    )


def _generator_batch(
    problem: BSDEProblem,
    time: Array,
    states: Array,
    values: Array,
    controls: Array,
    /,
) -> Array:
    def evaluate(state, value, control):
        result = jnp.asarray(problem.generator(time, state, value, control, problem.args))
        if result.shape != problem.output_shape:
            raise ValueError("BSDE generator returned an incompatible output shape.")
        return result

    return jax.vmap(evaluate)(states, values, controls)


def _terminal_batch(problem: BSDEProblem, states: Array, /) -> Array:
    def evaluate(state):
        result = jnp.asarray(problem.terminal(state, problem.args))
        if result.shape != problem.output_shape:
            raise ValueError(
                "BSDE terminal function returned an incompatible output shape."
            )
        return result

    return jax.vmap(evaluate)(states)


def _normal_equation_error(
    design: Array,
    target: Array,
    prediction: Array,
    mask: Array,
    coefficients: Array,
    ridge: float,
    /,
) -> Array:
    count = jnp.maximum(jnp.sum(mask), 1)
    residual = jnp.where(mask[:, None], target - prediction, 0.0)
    moment = design.T @ residual / count - ridge * coefficients
    return jnp.max(jnp.abs(moment), initial=0.0)


def solve_bsde_least_squares(
    problem: BSDEProblem,
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
) -> LeastSquaresBSDEResult:
    """Solve a Markovian Brownian BSDE by backward conditional regression.

    ``explicit`` evaluates the generator at the next pathwise value. ``implicit``
    solves the current-value backward Euler equation by damped Picard iteration.
    In both schemes, controls are projected from the conditional moment
    ``Y[n + 1] * dW[n] / dt``. No path resampling or hidden noise is introduced.
    """
    if not isinstance(problem, BSDEProblem):
        raise TypeError("problem must be a BSDEProblem.")
    if not isinstance(basis, AbstractBSDERegressionBasis):
        raise TypeError("basis must implement AbstractBSDERegressionBasis.")
    if basis.state_shape != problem.state_shape:
        raise ValueError("basis state_shape must match the BSDE problem.")
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
        raise ValueError("paths do not match the BSDE state, noise, or process contract.")
    if resolved_paths.jump_events:
        raise ValueError(
            "Least-squares Brownian BSDE schemes do not consume jump events."
        )

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

    node_values: list[Array | None] = [None] * (num_steps + 1)
    controls: list[Array | None] = [None] * num_steps
    generators: list[Array | None] = [None] * num_steps
    local_residuals: list[Array | None] = [None] * num_steps
    value_targets: list[Array | None] = [None] * (num_steps + 1)
    control_targets: list[Array | None] = [None] * num_steps
    value_residuals: list[Array | None] = [None] * (num_steps + 1)
    control_residuals: list[Array | None] = [None] * num_steps
    value_coefficients: list[Array | None] = [None] * (num_steps + 1)
    control_coefficients: list[Array | None] = [None] * num_steps
    feature_means: list[Array | None] = [None] * (num_steps + 1)
    feature_scales: list[Array | None] = [None] * (num_steps + 1)
    regression_masks: list[Array | None] = [None] * (num_steps + 1)
    control_masks: list[Array | None] = [None] * num_steps
    sample_counts: list[Array | None] = [None] * (num_steps + 1)
    design_ranks: list[Array | None] = [None] * (num_steps + 1)
    condition_numbers: list[Array | None] = [None] * (num_steps + 1)
    value_normal_errors: list[Array | None] = [None] * (num_steps + 1)
    control_normal_errors: list[Array | None] = [None] * num_steps
    picard_iterations: list[Array | None] = [None] * num_steps
    picard_errors: list[Array | None] = [None] * num_steps
    picard_converged: list[Array | None] = [None] * num_steps
    valid_steps: list[Array | None] = [None] * (num_steps + 1)

    terminal_target = _terminal_batch(problem, states[:, -1]).reshape(
        (num_paths, output_size)
    )
    terminal_design = _basis_matrix(basis, states[:, -1])
    (
        terminal_normalized,
        terminal_mean,
        terminal_scale,
        terminal_base_mask,
        terminal_design_count,
        terminal_rank,
        terminal_condition,
    ) = _normalise_design(
        terminal_design,
        path_valid,
        standardize=standardize,
        rcond=rcond_value,
    )
    (
        terminal_coefficients,
        _terminal_prediction,
        terminal_residual,
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
    del terminal_design_count
    node_values[-1] = terminal_target
    value_targets[-1] = terminal_target
    value_residuals[-1] = terminal_residual
    value_coefficients[-1] = terminal_coefficients
    feature_means[-1] = terminal_mean
    feature_scales[-1] = terminal_scale
    regression_masks[-1] = terminal_mask
    sample_counts[-1] = terminal_count
    design_ranks[-1] = terminal_rank
    condition_numbers[-1] = terminal_condition
    value_normal_errors[-1] = terminal_normal_error
    valid_steps[-1] = terminal_valid

    for step in range(num_steps - 1, -1, -1):
        next_value = node_values[step + 1]
        if next_value is None:
            raise RuntimeError("Backward BSDE recursion lost its next-node value.")
        time = resolved_paths.times[step]
        dt = resolved_paths.times[step + 1] - time
        current_states = states[:, step]
        design = _basis_matrix(basis, current_states)
        (
            normalized,
            mean,
            scale,
            base_mask,
            design_count,
            rank,
            condition,
        ) = _normalise_design(
            design,
            path_valid,
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
            z_residual,
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
            generator = _generator_batch(
                problem,
                time,
                current_states,
                next_value.reshape((num_paths,) + problem.output_shape),
                z_value,
            ).reshape((num_paths, output_size))
            y_target = next_value + dt * generator
            (
                y_coefficients,
                y_value,
                y_residual,
                y_mask,
                y_count,
                y_normal_error,
                y_valid,
            ) = _regress(
                normalized,
                y_target,
                base_mask,
                ridge=ridge_value,
                rcond=rcond_value,
                min_samples=required_samples,
            )
            iterations = jnp.asarray(1, dtype=jnp.int32)
            iteration_error = jnp.asarray(0.0, dtype=y_value.dtype)
            converged = jnp.asarray(True)
        else:
            (
                y_coefficients,
                y_value,
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
            converged = jnp.asarray(False)
            iterations = jnp.asarray(picard_steps, dtype=jnp.int32)
            iteration_error = jnp.asarray(jnp.inf, dtype=y_value.dtype)
            y_target = next_value
            y_residual = next_value - y_value
            y_mask = base_mask
            y_count = design_count
            y_normal_error = jnp.asarray(jnp.inf, dtype=y_value.dtype)
            y_valid = initial_valid
            generator = jnp.zeros_like(next_value)
            for iteration in range(picard_steps):
                candidate_generator = _generator_batch(
                    problem,
                    time,
                    current_states,
                    y_value.reshape((num_paths,) + problem.output_shape),
                    z_value,
                ).reshape((num_paths, output_size))
                candidate_target = next_value + dt * candidate_generator
                (
                    candidate_coefficients,
                    _candidate_value,
                    candidate_residual,
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
                damped_value = normalized @ damped_coefficients
                difference = jnp.where(
                    candidate_mask[:, None], damped_value - y_value, 0.0
                )
                denominator = jnp.maximum(
                    jnp.sqrt(
                        jnp.sum(jnp.where(candidate_mask[:, None], damped_value**2, 0.0))
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
                y_value = jnp.where(active, damped_value, y_value)
                y_target = jnp.where(active, candidate_target, y_target)
                y_residual = jnp.where(active, candidate_residual, y_residual)
                y_mask = jnp.where(active, candidate_mask, y_mask)
                y_count = jnp.where(active, candidate_count, y_count)
                y_normal_error = jnp.where(active, candidate_normal_error, y_normal_error)
                y_valid = jnp.where(active, candidate_valid, y_valid)
                generator = jnp.where(active, candidate_generator, generator)
                iteration_error = jnp.where(active, candidate_error, iteration_error)
                converged = converged | newly_converged
            generator = _generator_batch(
                problem,
                time,
                current_states,
                y_value.reshape((num_paths,) + problem.output_shape),
                z_value,
            ).reshape((num_paths, output_size))
            y_target = next_value + dt * generator
            y_residual = y_target - y_value
            y_normal_error = _normal_equation_error(
                normalized,
                y_target,
                y_value,
                y_mask,
                y_coefficients,
                ridge_value,
            )

        martingale = oe.contract(
            "pon,pn->po",
            z_value.reshape((num_paths, output_size, noise_size)),
            increments[:, step],
        )
        equation_residual = next_value - y_value + dt * generator - martingale
        node_values[step] = y_value
        controls[step] = z_value.reshape((num_paths, output_size * noise_size))
        generators[step] = generator
        local_residuals[step] = equation_residual
        value_targets[step] = y_target
        control_targets[step] = z_target
        value_residuals[step] = y_residual
        control_residuals[step] = z_residual
        value_coefficients[step] = y_coefficients
        control_coefficients[step] = z_coefficients
        feature_means[step] = mean
        feature_scales[step] = scale
        regression_masks[step] = y_mask
        control_masks[step] = z_mask
        sample_counts[step] = jnp.minimum(y_count, z_count)
        design_ranks[step] = rank
        condition_numbers[step] = condition
        value_normal_errors[step] = y_normal_error
        control_normal_errors[step] = z_normal_error
        picard_iterations[step] = iterations
        picard_errors[step] = iteration_error
        picard_converged[step] = converged
        valid_steps[step] = y_valid & z_valid

    def stack(items: list[Array | None], *, owner: str, axis: int = 1) -> Array:
        arrays = [item for item in items if item is not None]
        if len(arrays) != len(items):
            raise RuntimeError(f"Incomplete backward recursion for {owner}.")
        return jnp.stack(arrays, axis=axis)

    values_array = stack(node_values, owner="values").reshape(
        resolved_paths.sample_shape + (num_steps + 1,) + problem.output_shape
    )
    controls_array = stack(controls, owner="controls").reshape(
        resolved_paths.sample_shape
        + (num_steps,)
        + problem.output_shape
        + problem.noise_shape
    )
    generator_array = stack(generators, owner="generators").reshape(
        resolved_paths.sample_shape + (num_steps,) + problem.output_shape
    )
    local_array = stack(local_residuals, owner="local residuals").reshape(
        resolved_paths.sample_shape + (num_steps,) + problem.output_shape
    )
    value_target_array = stack(value_targets, owner="value targets").reshape(
        resolved_paths.sample_shape + (num_steps + 1,) + problem.output_shape
    )
    control_target_array = stack(control_targets, owner="control targets").reshape(
        resolved_paths.sample_shape
        + (num_steps,)
        + problem.output_shape
        + problem.noise_shape
    )
    value_residual_array = stack(
        value_residuals, owner="value regression residuals"
    ).reshape(resolved_paths.sample_shape + (num_steps + 1,) + problem.output_shape)
    control_residual_array = stack(
        control_residuals, owner="control regression residuals"
    ).reshape(
        resolved_paths.sample_shape
        + (num_steps,)
        + problem.output_shape
        + problem.noise_shape
    )
    regression_mask_array = stack(regression_masks, owner="regression masks").reshape(
        resolved_paths.sample_shape + (num_steps + 1,)
    )
    control_mask_array = stack(control_masks, owner="control masks").reshape(
        resolved_paths.sample_shape + (num_steps,)
    )
    valid_path_array = path_valid.reshape(resolved_paths.sample_shape)
    result = LeastSquaresBSDEResult(
        paths=resolved_paths,
        basis=basis,
        values=values_array,
        controls=controls_array,
        generator_values=generator_array,
        local_residuals=local_array,
        value_targets=value_target_array,
        control_targets=control_target_array,
        value_regression_residuals=value_residual_array,
        control_regression_residuals=control_residual_array,
        value_coefficients=stack(value_coefficients, owner="value coefficients", axis=0),
        control_coefficients=stack(
            control_coefficients, owner="control coefficients", axis=0
        ),
        feature_means=stack(feature_means, owner="feature means", axis=0),
        feature_scales=stack(feature_scales, owner="feature scales", axis=0),
        regression_masks=regression_mask_array,
        control_regression_masks=control_mask_array,
        sample_counts=stack(sample_counts, owner="sample counts", axis=0),
        design_ranks=stack(design_ranks, owner="design ranks", axis=0),
        condition_numbers=stack(condition_numbers, owner="condition numbers", axis=0),
        value_normal_equation_errors=stack(
            value_normal_errors, owner="value normal-equation errors", axis=0
        ),
        control_normal_equation_errors=stack(
            control_normal_errors, owner="control normal-equation errors", axis=0
        ),
        picard_iterations=stack(picard_iterations, owner="Picard iterations", axis=0),
        picard_errors=stack(picard_errors, owner="Picard errors", axis=0),
        picard_converged=stack(picard_converged, owner="Picard convergence", axis=0),
        valid_steps=stack(valid_steps, owner="valid steps", axis=0),
        valid_paths=valid_path_array,
        problem_id=problem.problem_id,
        scheme=scheme,
        ridge=ridge_value,
        output_shape=problem.output_shape,
        noise_shape=problem.noise_shape,
    )
    if raise_on_failure and not bool(result.successful):
        raise RuntimeError("Least-squares BSDE regression failed validation.")
    return result


def _predict(
    result: LeastSquaresBSDEResult,
    step: int,
    states: ArrayLike,
    coefficients: Array,
    event_shape: tuple[int, ...],
    /,
) -> Array:
    if not isinstance(result, LeastSquaresBSDEResult):
        raise TypeError("result must be a LeastSquaresBSDEResult.")
    state_values = jnp.asarray(states)
    state_rank = len(result.basis.state_shape)
    if (
        state_values.ndim < state_rank
        or state_values.shape[-state_rank:] != result.basis.state_shape
    ):
        raise ValueError("states must end with the BSDE state_shape.")
    batch_shape = state_values.shape[:-state_rank]
    design = _basis_matrix(result.basis, state_values)
    normalized = (design - result.feature_means[step]) / result.feature_scales[step]
    prediction = normalized @ coefficients
    return prediction.reshape(batch_shape + event_shape)


def predict_bsde_least_squares_value(
    result: LeastSquaresBSDEResult,
    step: int,
    states: ArrayLike,
    /,
) -> Array:
    """Evaluate the fitted conditional value at one stored time index."""
    index = int(step)
    if index < 0 or index >= result.paths.num_steps + 1:
        raise ValueError("step is outside the BSDE node range.")
    return _predict(
        result,
        index,
        states,
        result.value_coefficients[index],
        result.output_shape,
    )


def predict_bsde_least_squares_control(
    result: LeastSquaresBSDEResult,
    step: int,
    states: ArrayLike,
    /,
) -> Array:
    """Evaluate the fitted martingale control on one stored interval."""
    index = int(step)
    if index < 0 or index >= result.paths.num_steps:
        raise ValueError("step is outside the BSDE interval range.")
    return _predict(
        result,
        index,
        states,
        result.control_coefficients[index],
        result.output_shape + result.noise_shape,
    )


def _masked_node_rmse(values: Array, mask: Array, event_rank: int, /) -> Array:
    sample_rank = mask.ndim - 1
    sample_axes = tuple(range(sample_rank))
    event_axes = tuple(range(values.ndim - event_rank, values.ndim))
    squared = jnp.abs(values) ** 2
    if event_axes:
        squared = jnp.mean(squared, axis=event_axes)
    count = jnp.maximum(jnp.sum(mask, axis=sample_axes), 1)
    total = jnp.sum(jnp.where(mask, squared, 0.0), axis=sample_axes)
    return jnp.sqrt(total / count)


def least_squares_bsde_diagnostics(
    result: LeastSquaresBSDEResult,
    /,
) -> LeastSquaresBSDEDiagnostics:
    """Summarize regression fit, discrete BSDE residuals, and solver validity."""
    if not isinstance(result, LeastSquaresBSDEResult):
        raise TypeError("result must be a LeastSquaresBSDEResult.")
    value_rmse = _masked_node_rmse(
        result.value_regression_residuals,
        result.regression_masks,
        len(result.output_shape),
    )
    control_rmse = _masked_node_rmse(
        result.control_regression_residuals,
        result.control_regression_masks,
        len(result.output_shape) + len(result.noise_shape),
    )
    local_mask = result.control_regression_masks & result.regression_masks[..., :-1]
    local_rmse = _masked_node_rmse(
        result.local_residuals,
        local_mask,
        len(result.output_shape),
    )
    finite = (
        jnp.all(jnp.isfinite(result.values))
        & jnp.all(jnp.isfinite(result.controls))
        & jnp.all(jnp.isfinite(value_rmse))
        & jnp.all(jnp.isfinite(control_rmse))
        & jnp.all(jnp.isfinite(local_rmse))
        & jnp.all(jnp.isfinite(result.value_coefficients))
        & jnp.all(jnp.isfinite(result.control_coefficients))
    )
    return LeastSquaresBSDEDiagnostics(
        value_regression_rmse=value_rmse,
        control_regression_rmse=control_rmse,
        local_equation_rmse=local_rmse,
        valid_path_fraction=jnp.mean(result.valid_paths),
        max_value_normal_equation_error=jnp.max(
            result.value_normal_equation_errors, initial=0.0
        ),
        max_control_normal_equation_error=jnp.max(
            result.control_normal_equation_errors, initial=0.0
        ),
        all_regressions_valid=jnp.all(result.valid_steps),
        all_picard_steps_converged=jnp.all(result.picard_converged),
        finite=finite,
    )


__all__ = [
    "AbstractBSDERegressionBasis",
    "BSDERegressionScheme",
    "CallableBSDERegressionBasis",
    "least_squares_bsde_diagnostics",
    "LeastSquaresBSDEDiagnostics",
    "LeastSquaresBSDEResult",
    "PolynomialBSDERegressionBasis",
    "predict_bsde_least_squares_control",
    "predict_bsde_least_squares_value",
    "solve_bsde_least_squares",
]
