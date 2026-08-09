#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from .._layout import StateLayout


ContinuationMethod: TypeAlias = Literal["natural", "pseudo_arclength"]
ContinuationSpectrumKind: TypeAlias = Literal["none", "flow", "map", "floquet"]

STABILITY_STABLE = 0
STABILITY_UNSTABLE = 1
STABILITY_MARGINAL = 2
STABILITY_UNKNOWN = 3

CONTINUATION_POINT_SUCCESS = 0
CONTINUATION_POINT_CORRECTION_FAILED = 1
CONTINUATION_POINT_NONFINITE = 2

CONTINUATION_CAPACITY_REACHED = 0
CONTINUATION_PARAMETER_BOUND_REACHED = 1
CONTINUATION_CORRECTION_FAILED = 2
CONTINUATION_INITIAL_FAILED = 3
CONTINUATION_TANGENT_FAILED = 4
CONTINUATION_STATE_LIMIT_REACHED = 5
CONTINUATION_NONFINITE = 6


class NormalFormEvaluation(StrictModule):
    values: Array
    valid: Array
    names: tuple[str, ...] = eqx.field(static=True)
    hook_id: str = eqx.field(static=True)


class AbstractNormalFormHook(StrictModule):
    names: AbstractAttribute[tuple[str, ...]]
    hook_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        state: Array,
        parameter: Array,
        args: Any = None,
        /,
    ) -> NormalFormEvaluation:
        raise NotImplementedError


class CallableNormalFormHook(AbstractNormalFormHook):
    """Fixed-width user diagnostics evaluated at every accepted branch point."""

    function: Callable[[Array, Array, Any], Array]
    names: tuple[str, ...] = eqx.field(static=True)
    hook_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[Array, Array, Any], Array],
        /,
        *,
        names: Sequence[str],
        hook_id: str,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        resolved_names = tuple(str(name) for name in names)
        if (
            not resolved_names
            or any(not name for name in resolved_names)
            or len(set(resolved_names)) != len(resolved_names)
        ):
            raise ValueError("names must be non-empty and unique.")
        if not isinstance(hook_id, str) or not hook_id:
            raise ValueError("hook_id must be a non-empty string.")
        self.function = function
        self.names = resolved_names
        self.hook_id = hook_id

    def evaluate(
        self,
        state: Array,
        parameter: Array,
        args: Any = None,
        /,
    ) -> NormalFormEvaluation:
        values = jnp.asarray(self.function(state, parameter, args))
        if values.shape != (len(self.names),):
            raise ValueError(f"Normal-form hook must return shape ({len(self.names)},).")
        valid = jnp.all(jnp.isfinite(values))
        return NormalFormEvaluation(
            values=jnp.where(valid, values, jnp.nan),
            valid=valid,
            names=self.names,
            hook_id=self.hook_id,
        )


class ContinuationProblem(StrictModule):
    """Square parameterized residual with optional physical spectrum and hooks."""

    residual_function: Callable[[Array, Array, Any], Array]
    spectrum_function: Callable[[Array, Array, Any], Array] | None
    normal_form_hook: AbstractNormalFormHook | None
    state_layout: StateLayout
    spectrum_kind: ContinuationSpectrumKind = eqx.field(static=True)
    neutral_multipliers: int = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        residual_function: Callable[[Array, Array, Any], Array],
        /,
        *,
        state_layout: StateLayout,
        parameter_id: str,
        spectrum_kind: ContinuationSpectrumKind = "flow",
        spectrum_function: Callable[[Array, Array, Any], Array] | None = None,
        neutral_multipliers: int = 0,
        normal_form_hook: AbstractNormalFormHook | None = None,
        problem_id: str,
    ):
        if not callable(residual_function):
            raise TypeError("residual_function must be callable.")
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if spectrum_kind not in ("none", "flow", "map", "floquet"):
            raise ValueError("Unsupported spectrum_kind.")
        if spectrum_function is not None and not callable(spectrum_function):
            raise TypeError("spectrum_function must be callable or None.")
        if spectrum_kind == "floquet" and spectrum_function is None:
            raise ValueError("Floquet continuation requires spectrum_function.")
        neutral = int(neutral_multipliers)
        if neutral < 0:
            raise ValueError("neutral_multipliers must be nonnegative.")
        if spectrum_kind != "floquet" and neutral:
            raise ValueError("neutral_multipliers only applies to Floquet spectra.")
        if normal_form_hook is not None and not isinstance(
            normal_form_hook, AbstractNormalFormHook
        ):
            raise TypeError("normal_form_hook must be an AbstractNormalFormHook or None.")
        if not isinstance(parameter_id, str) or not parameter_id:
            raise ValueError("parameter_id must be non-empty.")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError("problem_id must be non-empty.")
        self.residual_function = residual_function
        self.spectrum_function = spectrum_function
        self.normal_form_hook = normal_form_hook
        self.state_layout = state_layout
        self.spectrum_kind = spectrum_kind
        self.neutral_multipliers = neutral
        self.parameter_id = parameter_id
        self.problem_id = problem_id

    def residual(
        self, state: ArrayLike, parameter: ArrayLike, args: Any = None, /
    ) -> Array:
        state_value = jnp.asarray(state)
        parameter_value = jnp.asarray(parameter)
        if state_value.shape != self.state_layout.shape:
            raise ValueError("state must have the continuation state layout shape.")
        if parameter_value.shape != ():
            raise ValueError("parameter must be scalar.")
        value = jnp.asarray(self.residual_function(state_value, parameter_value, args))
        if value.shape != self.state_layout.shape:
            raise ValueError("Continuation residual must have the state layout shape.")
        return value


class BifurcationIndicators(StrictModule):
    fold_distance: Array
    hopf_distance: Array
    flip_distance: Array
    torus_distance: Array
    branch_distance: Array
    fold_test: Array
    hopf_test: Array
    flip_test: Array
    torus_test: Array
    critical_real: Array
    critical_imaginary: Array


class BranchBifurcationFlags(StrictModule):
    fold: Array
    hopf: Array
    flip: Array
    torus: Array
    branch_point: Array


class ContinuationBranch(StrictModule):
    """Fixed-capacity branch states, tangents, spectra, indicators, and provenance."""

    states: Array
    parameters: Array
    valid: Array
    residual_norm: Array
    newton_iterations: Array
    retry_count: Array
    point_status: Array
    step_size: Array
    tangent_states: Array
    tangent_parameters: Array
    spectra: Array
    spectrum_valid: Array
    stability: Array
    indicators: BifurcationIndicators
    bifurcations: BranchBifurcationFlags
    hook_values: Array
    hook_valid: Array
    count: Array
    termination_status: Array
    problem: ContinuationProblem
    capacity: int = eqx.field(static=True)
    method: ContinuationMethod = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    parent_branch_id: str | None = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.count > 0


class BranchSwitchSeed(StrictModule):
    state: Array
    parameter: Array
    tangent_state: Array
    tangent_parameter: Array
    valid: Array
    source_index: int = eqx.field(static=True)
    source_branch_id: str = eqx.field(static=True)
    switch_id: str = eqx.field(static=True)


class AbstractBranchSwitchHook(StrictModule):
    switch_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        state: Array,
        parameter: Array,
        tangent_state: Array,
        tangent_parameter: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        raise NotImplementedError


class CallableBranchSwitchHook(AbstractBranchSwitchHook):
    """Pluggable branch-switch seed without prescribed normal-form machinery."""

    function: Callable[
        [Array, Array, Array, Array, Any], tuple[Array, Array, Array, Array]
    ]
    switch_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[
            [Array, Array, Array, Array, Any], tuple[Array, Array, Array, Array]
        ],
        /,
        *,
        switch_id: str,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        if not isinstance(switch_id, str) or not switch_id:
            raise ValueError("switch_id must be non-empty.")
        self.function = function
        self.switch_id = switch_id

    def evaluate(
        self,
        state: Array,
        parameter: Array,
        tangent_state: Array,
        tangent_parameter: Array,
        args: Any = None,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        return self.function(state, parameter, tangent_state, tangent_parameter, args)


class _CorrectorResult(StrictModule):
    state: Array
    parameter: Array
    residual_norm: Array
    iterations: Array
    valid: Array
    status: Array


def _newton_correct(
    residual_function: Callable[[Array], Array],
    initial: Array,
    /,
    *,
    max_iterations: int,
    rtol: float,
    atol: float,
    max_line_search: int,
) -> tuple[Array, Array, int, bool, int]:
    values = initial
    residual = residual_function(values)
    norm = jnp.linalg.norm(residual)
    threshold = atol + rtol * norm
    if not bool(jnp.all(jnp.isfinite(residual))):
        return values, norm, 0, False, CONTINUATION_POINT_NONFINITE
    if bool(norm <= threshold):
        return values, norm, 0, True, CONTINUATION_POINT_SUCCESS
    for iteration in range(1, max_iterations + 1):
        jacobian = jax.jacfwd(residual_function)(values)
        step = jnp.linalg.lstsq(jacobian, -residual, rcond=None)[0]
        if not bool(jnp.all(jnp.isfinite(jacobian)) & jnp.all(jnp.isfinite(step))):
            return (
                values,
                norm,
                iteration - 1,
                False,
                CONTINUATION_POINT_NONFINITE,
            )
        scale = 1.0
        accepted = False
        for _ in range(max_line_search):
            candidate = values + scale * step
            candidate_residual = residual_function(candidate)
            candidate_norm = jnp.linalg.norm(candidate_residual)
            if bool(jnp.all(jnp.isfinite(candidate_residual)) & (candidate_norm < norm)):
                values = candidate
                residual = candidate_residual
                norm = candidate_norm
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            return (
                values,
                norm,
                iteration - 1,
                False,
                CONTINUATION_POINT_CORRECTION_FAILED,
            )
        if bool(norm <= threshold):
            return values, norm, iteration, True, CONTINUATION_POINT_SUCCESS
    return (
        values,
        norm,
        max_iterations,
        False,
        CONTINUATION_POINT_CORRECTION_FAILED,
    )


def _natural_corrector(
    problem: ContinuationProblem,
    state: Array,
    parameter: Array,
    args: Any,
    /,
    **newton_options,
) -> _CorrectorResult:
    def residual(flat_state: Array) -> Array:
        return problem.residual(
            flat_state.reshape(problem.state_layout.shape), parameter, args
        ).reshape((-1,))

    values, norm, iterations, valid, status = _newton_correct(
        residual, state.reshape((-1,)), **newton_options
    )
    return _CorrectorResult(
        state=values.reshape(problem.state_layout.shape),
        parameter=parameter,
        residual_norm=norm,
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        valid=jnp.asarray(valid),
        status=jnp.asarray(status, dtype=jnp.int32),
    )


def _arclength_corrector(
    problem: ContinuationProblem,
    predicted_state: Array,
    predicted_parameter: Array,
    tangent_state: Array,
    tangent_parameter: Array,
    args: Any,
    /,
    **newton_options,
) -> _CorrectorResult:
    state_size = problem.state_layout.size

    def augmented(values: Array) -> Array:
        state = values[:state_size].reshape(problem.state_layout.shape)
        parameter = values[state_size]
        equation = problem.residual(state, parameter, args).reshape((-1,))
        phase = jnp.vdot(
            tangent_state.reshape((-1,)),
            (state - predicted_state).reshape((-1,)),
        ).real + tangent_parameter * (parameter - predicted_parameter)
        return jnp.concatenate((equation, phase.reshape((1,))))

    initial = jnp.concatenate(
        (predicted_state.reshape((-1,)), predicted_parameter.reshape((1,)))
    )
    values, norm, iterations, valid, status = _newton_correct(
        augmented, initial, **newton_options
    )
    return _CorrectorResult(
        state=values[:state_size].reshape(problem.state_layout.shape),
        parameter=values[state_size],
        residual_norm=norm,
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        valid=jnp.asarray(valid),
        status=jnp.asarray(status, dtype=jnp.int32),
    )


def _initial_tangent(
    problem: ContinuationProblem,
    state: Array,
    parameter: Array,
    direction: float,
    args: Any,
    /,
) -> tuple[Array, Array, Array]:
    state_jacobian = jax.jacfwd(lambda value: problem.residual(value, parameter, args))(
        state
    ).reshape((problem.state_layout.size, problem.state_layout.size))
    parameter_derivative = jax.jacfwd(lambda value: problem.residual(state, value, args))(
        parameter
    ).reshape((-1,))
    tangent_parameter = jnp.asarray(direction, dtype=state.dtype)
    tangent_state = jnp.linalg.lstsq(
        state_jacobian,
        -parameter_derivative * tangent_parameter,
        rcond=None,
    )[0].reshape(problem.state_layout.shape)
    norm = jnp.sqrt(jnp.sum(jnp.abs(tangent_state) ** 2) + tangent_parameter**2)
    valid = (
        jnp.all(jnp.isfinite(state_jacobian))
        & jnp.all(jnp.isfinite(tangent_state))
        & jnp.isfinite(norm)
        & (norm > 0.0)
    )
    return tangent_state / norm, tangent_parameter / norm, valid


def _remove_neutral(multipliers: Array, count: int, /) -> Array:
    included = jnp.ones(multipliers.shape, dtype=bool)
    distances = jnp.abs(multipliers - 1.0)
    for _ in range(min(count, int(multipliers.size))):
        index = jnp.argmin(jnp.where(included, distances, jnp.inf))
        included = included.at[index].set(False)
    return included


def _spectrum_evidence(
    problem: ContinuationProblem,
    state: Array,
    parameter: Array,
    args: Any,
    /,
    *,
    spectrum_capacity: int,
    stability_tolerance: float,
    imaginary_tolerance: float,
) -> tuple[Array, Array, Array, tuple[Array, ...]]:
    state_jacobian = jax.jacfwd(lambda value: problem.residual(value, parameter, args))(
        state
    ).reshape((problem.state_layout.size, problem.state_layout.size))
    singular_values = jnp.linalg.svd(state_jacobian, compute_uv=False)
    sorted_singular = jnp.sort(singular_values)
    branch_distance = (
        sorted_singular[1]
        if sorted_singular.size > 1
        else jnp.asarray(jnp.inf, dtype=state.dtype)
    )
    if problem.spectrum_kind == "none":
        spectrum = jnp.asarray([], dtype=jnp.complex128)
    elif problem.spectrum_function is not None:
        spectrum = jnp.asarray(problem.spectrum_function(state, parameter, args)).reshape(
            (-1,)
        )
    elif problem.spectrum_kind == "flow":
        spectrum = jnp.linalg.eigvals(state_jacobian)
    else:
        spectrum = jnp.linalg.eigvals(state_jacobian + jnp.eye(problem.state_layout.size))
    if spectrum.size > spectrum_capacity:
        raise ValueError(
            f"Spectrum has {int(spectrum.size)} values; capacity={spectrum_capacity}."
        )
    padded = jnp.full((spectrum_capacity,), jnp.nan + 0.0j)
    spectrum_mask = jnp.zeros((spectrum_capacity,), dtype=bool)
    if spectrum.size:
        padded = padded.at[: spectrum.size].set(spectrum)
        spectrum_mask = spectrum_mask.at[: spectrum.size].set(jnp.isfinite(spectrum))
    included = jnp.ones(spectrum.shape, dtype=bool)
    if problem.spectrum_kind == "floquet":
        included = _remove_neutral(spectrum, problem.neutral_multipliers)
    relevant = spectrum[included]
    infinity = jnp.asarray(jnp.inf, dtype=state.dtype)
    fold_distance = infinity
    hopf_distance = infinity
    flip_distance = infinity
    torus_distance = infinity
    fold_test = jnp.asarray(jnp.nan, dtype=state.dtype)
    hopf_test = jnp.asarray(jnp.nan, dtype=state.dtype)
    flip_test = jnp.asarray(jnp.nan, dtype=state.dtype)
    torus_test = jnp.asarray(jnp.nan, dtype=state.dtype)
    critical_real = jnp.asarray(jnp.nan, dtype=state.dtype)
    critical_imaginary = jnp.asarray(jnp.nan, dtype=state.dtype)
    stability = jnp.asarray(STABILITY_UNKNOWN, dtype=jnp.int32)
    if problem.spectrum_kind == "flow" and relevant.size:
        real = jnp.real(relevant)
        imaginary = jnp.abs(jnp.imag(relevant))
        real_candidates = imaginary <= imaginary_tolerance
        if bool(jnp.any(real_candidates)):
            index = jnp.argmin(jnp.where(real_candidates, jnp.abs(real), jnp.inf))
            fold_test = real[index]
            fold_distance = jnp.abs(fold_test)
        complex_candidates = imaginary > imaginary_tolerance
        if bool(jnp.any(complex_candidates)):
            index = jnp.argmin(jnp.where(complex_candidates, jnp.abs(real), jnp.inf))
            hopf_test = real[index]
            hopf_distance = jnp.abs(hopf_test)
            critical_real = real[index]
            critical_imaginary = imaginary[index]
        abscissa = jnp.max(real)
        stability = jnp.where(
            abscissa < -stability_tolerance,
            STABILITY_STABLE,
            jnp.where(
                abscissa > stability_tolerance,
                STABILITY_UNSTABLE,
                STABILITY_MARGINAL,
            ),
        ).astype(jnp.int32)
    elif problem.spectrum_kind in ("map", "floquet") and relevant.size:
        real = jnp.real(relevant)
        imaginary = jnp.abs(jnp.imag(relevant))
        fold_index = jnp.argmin(jnp.abs(relevant - 1.0))
        flip_index = jnp.argmin(jnp.abs(relevant + 1.0))
        fold_test = real[fold_index] - 1.0
        flip_test = real[flip_index] + 1.0
        fold_distance = jnp.abs(relevant[fold_index] - 1.0)
        flip_distance = jnp.abs(relevant[flip_index] + 1.0)
        complex_candidates = imaginary > imaginary_tolerance
        if bool(jnp.any(complex_candidates)):
            index = jnp.argmin(
                jnp.where(
                    complex_candidates,
                    jnp.abs(jnp.abs(relevant) - 1.0),
                    jnp.inf,
                )
            )
            torus_test = jnp.abs(relevant[index]) - 1.0
            torus_distance = jnp.abs(torus_test)
            critical_real = real[index]
            critical_imaginary = imaginary[index]
        radius = jnp.max(jnp.abs(relevant))
        stability = jnp.where(
            radius < 1.0 - stability_tolerance,
            STABILITY_STABLE,
            jnp.where(
                radius > 1.0 + stability_tolerance,
                STABILITY_UNSTABLE,
                STABILITY_MARGINAL,
            ),
        ).astype(jnp.int32)
    indicators = (
        fold_distance,
        hopf_distance,
        flip_distance,
        torus_distance,
        branch_distance,
        fold_test,
        hopf_test,
        flip_test,
        torus_test,
        critical_real,
        critical_imaginary,
    )
    return padded, spectrum_mask, stability, indicators


def _crossing(left: float, right: float, /) -> bool:
    return np.isfinite(left) and np.isfinite(right) and left * right <= 0.0


def continue_branch(
    problem: ContinuationProblem,
    initial_state: ArrayLike,
    initial_parameter: ArrayLike,
    /,
    *,
    args: Any = None,
    method: ContinuationMethod = "pseudo_arclength",
    direction: int = 1,
    initial_step: float = 0.05,
    min_step: float = 1e-4,
    max_step: float = 0.25,
    step_growth: float = 1.4,
    step_shrink: float = 0.5,
    easy_iterations: int = 3,
    hard_iterations: int = 8,
    max_retries: int = 6,
    max_points: int = 128,
    max_dense_dimension: int = 512,
    max_newton_iterations: int = 15,
    newton_rtol: float = 1e-8,
    newton_atol: float = 1e-10,
    max_line_search: int = 10,
    parameter_bounds: tuple[float, float] | None = None,
    state_norm_limit: float = np.inf,
    spectrum_capacity: int | None = None,
    stability_tolerance: float = 1e-6,
    imaginary_tolerance: float = 1e-8,
    bifurcation_tolerance: float = 1e-3,
    branch_id: str | None = None,
    parent_branch_id: str | None = None,
) -> ContinuationBranch:
    """Trace one natural or pseudo-arclength branch with adaptive retries."""
    if not isinstance(problem, ContinuationProblem):
        raise TypeError("problem must be a ContinuationProblem.")
    if method not in ("natural", "pseudo_arclength"):
        raise ValueError("method must be 'natural' or 'pseudo_arclength'.")
    if int(direction) not in (-1, 1):
        raise ValueError("direction must be -1 or 1.")
    capacity = int(max_points)
    if capacity < 2:
        raise ValueError("max_points must be at least two.")
    dense_limit = int(max_dense_dimension)
    solve_dimension = problem.state_layout.size + (
        1 if method == "pseudo_arclength" else 0
    )
    if dense_limit < 1:
        raise ValueError("max_dense_dimension must be positive.")
    if solve_dimension > dense_limit:
        raise ValueError(
            f"Dense continuation dimension {solve_dimension} exceeds "
            f"max_dense_dimension={dense_limit}."
        )
    step = float(initial_step)
    minimum_step = float(min_step)
    maximum_step = float(max_step)
    growth = float(step_growth)
    shrink = float(step_shrink)
    if not (
        0.0 < minimum_step <= step <= maximum_step and growth > 1.0 and 0.0 < shrink < 1.0
    ):
        raise ValueError("Continuation step controls are inconsistent.")
    if int(max_retries) < 0 or int(max_newton_iterations) < 1:
        raise ValueError(
            "max_retries must be nonnegative and Newton iterations positive."
        )
    if not np.isfinite(state_norm_limit) and state_norm_limit != np.inf:
        raise ValueError("state_norm_limit must be positive or infinity.")
    if state_norm_limit <= 0.0:
        raise ValueError("state_norm_limit must be positive.")
    bounds = None
    if parameter_bounds is not None:
        lower, upper = (float(value) for value in parameter_bounds)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("parameter_bounds must be finite and increasing.")
        bounds = (lower, upper)
    state = jnp.asarray(initial_state)
    parameter = jnp.asarray(initial_parameter)
    if state.shape != problem.state_layout.shape or parameter.shape != ():
        raise ValueError("Initial state or parameter has the wrong shape.")
    spectrum_width = (
        problem.state_layout.size if spectrum_capacity is None else int(spectrum_capacity)
    )
    if spectrum_width < 1:
        raise ValueError("spectrum_capacity must be positive or None.")
    newton_options = {
        "max_iterations": int(max_newton_iterations),
        "rtol": float(newton_rtol),
        "atol": float(newton_atol),
        "max_line_search": int(max_line_search),
    }
    initial = _natural_corrector(problem, state, parameter, args, **newton_options)
    state_shape = problem.state_layout.shape
    hook_width = (
        0 if problem.normal_form_hook is None else len(problem.normal_form_hook.names)
    )
    states = jnp.full((capacity,) + state_shape, jnp.nan)
    parameters = jnp.full((capacity,), jnp.nan)
    valid = jnp.zeros((capacity,), dtype=bool)
    residual_norm = jnp.full((capacity,), jnp.nan)
    newton_iterations = jnp.zeros((capacity,), dtype=jnp.int32)
    retry_count = jnp.zeros((capacity,), dtype=jnp.int32)
    point_status = jnp.full(
        (capacity,), CONTINUATION_POINT_CORRECTION_FAILED, dtype=jnp.int32
    )
    step_sizes = jnp.full((capacity,), jnp.nan)
    tangent_states = jnp.full((capacity,) + state_shape, jnp.nan)
    tangent_parameters = jnp.full((capacity,), jnp.nan)
    spectra = jnp.full((capacity, spectrum_width), jnp.nan + 0.0j)
    spectrum_valid_values = jnp.zeros((capacity, spectrum_width), dtype=bool)
    stability_values = jnp.full((capacity,), STABILITY_UNKNOWN, dtype=jnp.int32)
    indicator_arrays = [jnp.full((capacity,), jnp.nan) for _ in range(11)]
    hook_values = jnp.full((capacity, hook_width), jnp.nan)
    hook_valid = jnp.zeros((capacity,), dtype=bool)
    if not bool(initial.valid):
        termination = CONTINUATION_INITIAL_FAILED
        count = 0
    else:
        state = initial.state
        parameter = initial.parameter
        tangent_state, tangent_parameter, tangent_valid = _initial_tangent(
            problem, state, parameter, float(direction), args
        )
        if not bool(tangent_valid):
            termination = CONTINUATION_TANGENT_FAILED
            count = 0
        else:
            termination = CONTINUATION_CAPACITY_REACHED
            count = 1
            states = states.at[0].set(state)
            parameters = parameters.at[0].set(parameter)
            valid = valid.at[0].set(True)
            residual_norm = residual_norm.at[0].set(initial.residual_norm)
            newton_iterations = newton_iterations.at[0].set(initial.iterations)
            retry_count = retry_count.at[0].set(0)
            point_status = point_status.at[0].set(initial.status)
            step_sizes = step_sizes.at[0].set(0.0)
            tangent_states = tangent_states.at[0].set(tangent_state)
            tangent_parameters = tangent_parameters.at[0].set(tangent_parameter)

    def record(
        index: int, accepted: _CorrectorResult, retries: int, accepted_step: float
    ) -> None:
        nonlocal states, parameters, valid, residual_norm, newton_iterations
        nonlocal retry_count, point_status, step_sizes, tangent_states
        nonlocal tangent_parameters, spectra, spectrum_valid_values
        nonlocal stability_values, indicator_arrays, hook_values, hook_valid
        states = states.at[index].set(accepted.state)
        parameters = parameters.at[index].set(accepted.parameter)
        valid = valid.at[index].set(True)
        residual_norm = residual_norm.at[index].set(accepted.residual_norm)
        newton_iterations = newton_iterations.at[index].set(accepted.iterations)
        retry_count = retry_count.at[index].set(retries)
        point_status = point_status.at[index].set(accepted.status)
        step_sizes = step_sizes.at[index].set(accepted_step)
        tangent_states = tangent_states.at[index].set(tangent_state)
        tangent_parameters = tangent_parameters.at[index].set(tangent_parameter)

    if count:
        initial_accepted = _CorrectorResult(
            state=state,
            parameter=parameter,
            residual_norm=initial.residual_norm,
            iterations=initial.iterations,
            valid=initial.valid,
            status=initial.status,
        )
        record(0, initial_accepted, 0, 0.0)
    while count and count < capacity:
        previous_state = state
        previous_parameter = parameter
        trial_step = step
        accepted = None
        retries_used = 0
        for retry in range(int(max_retries) + 1):
            if method == "natural":
                target_parameter = parameter + float(direction) * trial_step
                if bounds is not None and not (
                    bounds[0] <= float(target_parameter) <= bounds[1]
                ):
                    termination = CONTINUATION_PARAMETER_BOUND_REACHED
                    break
                if abs(float(tangent_parameter)) > np.finfo(float).eps:
                    predicted_state = state + tangent_state * (
                        (target_parameter - parameter) / tangent_parameter
                    )
                else:
                    predicted_state = state
                corrected = _natural_corrector(
                    problem,
                    predicted_state,
                    target_parameter,
                    args,
                    **newton_options,
                )
            else:
                predicted_state = state + trial_step * tangent_state
                predicted_parameter = parameter + trial_step * tangent_parameter
                corrected = _arclength_corrector(
                    problem,
                    predicted_state,
                    predicted_parameter,
                    tangent_state,
                    tangent_parameter,
                    args,
                    **newton_options,
                )
                if (
                    bool(corrected.valid)
                    and bounds is not None
                    and not (bounds[0] <= float(corrected.parameter) <= bounds[1])
                ):
                    termination = CONTINUATION_PARAMETER_BOUND_REACHED
                    break
            if bool(corrected.valid):
                accepted = corrected
                retries_used = retry
                break
            trial_step *= shrink
            if trial_step < minimum_step:
                break
        if termination == CONTINUATION_PARAMETER_BOUND_REACHED:
            break
        if accepted is None:
            termination = CONTINUATION_CORRECTION_FAILED
            break
        state = accepted.state
        parameter = accepted.parameter
        if not bool(jnp.all(jnp.isfinite(state)) & jnp.isfinite(parameter)):
            termination = CONTINUATION_NONFINITE
            break
        if float(jnp.linalg.norm(state.reshape((-1,)))) > state_norm_limit:
            termination = CONTINUATION_STATE_LIMIT_REACHED
            break
        if method == "pseudo_arclength":
            secant_state = state - previous_state
            secant_parameter = parameter - previous_parameter
            secant_norm = jnp.sqrt(
                jnp.sum(jnp.abs(secant_state) ** 2) + secant_parameter**2
            )
            if not bool(jnp.isfinite(secant_norm) & (secant_norm > 0.0)):
                termination = CONTINUATION_TANGENT_FAILED
                break
            next_tangent_state = secant_state / secant_norm
            next_tangent_parameter = secant_parameter / secant_norm
            orientation = (
                jnp.vdot(
                    tangent_state.reshape((-1,)), next_tangent_state.reshape((-1,))
                ).real
                + tangent_parameter * next_tangent_parameter
            )
            sign = jnp.where(orientation >= 0.0, 1.0, -1.0)
            tangent_state = sign * next_tangent_state
            tangent_parameter = sign * next_tangent_parameter
        else:
            tangent_state, tangent_parameter, tangent_valid = _initial_tangent(
                problem, state, parameter, float(direction), args
            )
            if not bool(tangent_valid):
                termination = CONTINUATION_TANGENT_FAILED
                break
        record(count, accepted, retries_used, trial_step)
        count += 1
        iteration_count = int(accepted.iterations)
        if iteration_count <= int(easy_iterations) and retries_used == 0:
            step = min(maximum_step, trial_step * growth)
        elif iteration_count >= int(hard_iterations) or retries_used:
            step = max(minimum_step, trial_step * shrink)
        else:
            step = trial_step
    if count == capacity:
        termination = CONTINUATION_CAPACITY_REACHED
    for index in range(count):
        spectrum, spectrum_mask, point_stability, point_indicators = _spectrum_evidence(
            problem,
            states[index],
            parameters[index],
            args,
            spectrum_capacity=spectrum_width,
            stability_tolerance=float(stability_tolerance),
            imaginary_tolerance=float(imaginary_tolerance),
        )
        spectra = spectra.at[index].set(spectrum)
        spectrum_valid_values = spectrum_valid_values.at[index].set(spectrum_mask)
        stability_values = stability_values.at[index].set(point_stability)
        for indicator_index, indicator in enumerate(point_indicators):
            indicator_arrays[indicator_index] = (
                indicator_arrays[indicator_index].at[index].set(indicator)
            )
        if problem.normal_form_hook is not None:
            hook = problem.normal_form_hook.evaluate(
                states[index], parameters[index], args
            )
            hook_values = hook_values.at[index].set(hook.values)
            hook_valid = hook_valid.at[index].set(hook.valid)
    fold_flags = jnp.zeros((capacity,), dtype=bool)
    hopf_flags = jnp.zeros((capacity,), dtype=bool)
    flip_flags = jnp.zeros((capacity,), dtype=bool)
    torus_flags = jnp.zeros((capacity,), dtype=bool)
    branch_flags = indicator_arrays[4] <= float(bifurcation_tolerance)
    for index in range(1, count):
        if problem.spectrum_kind == "flow":
            if _crossing(
                float(indicator_arrays[5][index - 1]),
                float(indicator_arrays[5][index]),
            ):
                fold_flags = fold_flags.at[index].set(True)
            if _crossing(
                float(indicator_arrays[6][index - 1]),
                float(indicator_arrays[6][index]),
            ):
                hopf_flags = hopf_flags.at[index].set(True)
        elif problem.spectrum_kind in ("map", "floquet"):
            if _crossing(
                float(indicator_arrays[5][index - 1]),
                float(indicator_arrays[5][index]),
            ):
                fold_flags = fold_flags.at[index].set(True)
            if _crossing(
                float(indicator_arrays[7][index - 1]),
                float(indicator_arrays[7][index]),
            ):
                flip_flags = flip_flags.at[index].set(True)
            if _crossing(
                float(indicator_arrays[8][index - 1]),
                float(indicator_arrays[8][index]),
            ):
                torus_flags = torus_flags.at[index].set(True)
    identifier = (
        "continuation-branch:"
        + canonical_fingerprint(
            {
                "problem": problem.problem_id,
                "method": method,
                "direction": int(direction),
                "initial_parameter": float(initial_parameter),
                "parent": parent_branch_id,
            }
        )
        if branch_id is None
        else str(branch_id)
    )
    if not identifier:
        raise ValueError("branch_id must be non-empty.")
    return ContinuationBranch(
        states=states,
        parameters=parameters,
        valid=valid,
        residual_norm=residual_norm,
        newton_iterations=newton_iterations,
        retry_count=retry_count,
        point_status=point_status,
        step_size=step_sizes,
        tangent_states=tangent_states,
        tangent_parameters=tangent_parameters,
        spectra=spectra,
        spectrum_valid=spectrum_valid_values,
        stability=stability_values,
        indicators=BifurcationIndicators(
            fold_distance=indicator_arrays[0],
            hopf_distance=indicator_arrays[1],
            flip_distance=indicator_arrays[2],
            torus_distance=indicator_arrays[3],
            branch_distance=indicator_arrays[4],
            fold_test=indicator_arrays[5],
            hopf_test=indicator_arrays[6],
            flip_test=indicator_arrays[7],
            torus_test=indicator_arrays[8],
            critical_real=indicator_arrays[9],
            critical_imaginary=indicator_arrays[10],
        ),
        bifurcations=BranchBifurcationFlags(
            fold=fold_flags,
            hopf=hopf_flags,
            flip=flip_flags,
            torus=torus_flags,
            branch_point=branch_flags,
        ),
        hook_values=hook_values,
        hook_valid=hook_valid,
        count=jnp.asarray(count, dtype=jnp.int32),
        termination_status=jnp.asarray(termination, dtype=jnp.int32),
        problem=problem,
        capacity=capacity,
        method=method,
        branch_id=identifier,
        parent_branch_id=parent_branch_id,
        method_id=(
            f"continuation:{method}:adaptive-step:newton-dense:"
            f"rtol={float(newton_rtol):g}:atol={float(newton_atol):g}"
        ),
    )


def branch_switch_seed(
    branch: ContinuationBranch,
    index: int,
    hook: AbstractBranchSwitchHook,
    /,
    *,
    args: Any = None,
) -> BranchSwitchSeed:
    """Apply one explicit branch-switch hook to a valid stored branch point."""
    if not isinstance(branch, ContinuationBranch):
        raise TypeError("branch must be a ContinuationBranch.")
    if not isinstance(hook, AbstractBranchSwitchHook):
        raise TypeError("hook must be an AbstractBranchSwitchHook.")
    point = int(index)
    if point < 0 or point >= int(branch.count) or not bool(branch.valid[point]):
        raise ValueError("index must select one valid branch point.")
    state, parameter, tangent_state, tangent_parameter = hook.evaluate(
        branch.states[point],
        branch.parameters[point],
        branch.tangent_states[point],
        branch.tangent_parameters[point],
        args,
    )
    state_value = jnp.asarray(state)
    parameter_value = jnp.asarray(parameter)
    tangent_state_value = jnp.asarray(tangent_state)
    tangent_parameter_value = jnp.asarray(tangent_parameter)
    valid = (
        (state_value.shape == branch.problem.state_layout.shape)
        & (parameter_value.shape == ())
        & (tangent_state_value.shape == branch.problem.state_layout.shape)
        & (tangent_parameter_value.shape == ())
        & jnp.all(jnp.isfinite(state_value))
        & jnp.isfinite(parameter_value)
        & jnp.all(jnp.isfinite(tangent_state_value))
        & jnp.isfinite(tangent_parameter_value)
    )
    return BranchSwitchSeed(
        state=state_value,
        parameter=parameter_value,
        tangent_state=tangent_state_value,
        tangent_parameter=tangent_parameter_value,
        valid=valid,
        source_index=point,
        source_branch_id=branch.branch_id,
        switch_id=hook.switch_id,
    )


__all__ = [
    "AbstractBranchSwitchHook",
    "AbstractNormalFormHook",
    "BifurcationIndicators",
    "BranchBifurcationFlags",
    "BranchSwitchSeed",
    "CONTINUATION_CAPACITY_REACHED",
    "CONTINUATION_CORRECTION_FAILED",
    "CONTINUATION_INITIAL_FAILED",
    "CONTINUATION_NONFINITE",
    "CONTINUATION_PARAMETER_BOUND_REACHED",
    "CONTINUATION_POINT_CORRECTION_FAILED",
    "CONTINUATION_POINT_NONFINITE",
    "CONTINUATION_POINT_SUCCESS",
    "CONTINUATION_STATE_LIMIT_REACHED",
    "CONTINUATION_TANGENT_FAILED",
    "CallableBranchSwitchHook",
    "CallableNormalFormHook",
    "ContinuationBranch",
    "ContinuationMethod",
    "ContinuationProblem",
    "ContinuationSpectrumKind",
    "NormalFormEvaluation",
    "STABILITY_MARGINAL",
    "STABILITY_STABLE",
    "STABILITY_UNKNOWN",
    "STABILITY_UNSTABLE",
    "branch_switch_seed",
    "continue_branch",
]
