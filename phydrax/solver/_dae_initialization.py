#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from enum import IntEnum
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..dynamics import AbstractInputPolicy, DifferentialAlgebraicSystem
from ..linalg import ArraySpace, DiagonalPairing
from ..nonlinear import (
    AbstractNonlinearMethod,
    implicit_root_result,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
)


DAEInitializationMode: TypeAlias = Literal[
    "index-one",
    "fixed-rate",
    "check",
    "custom",
]


class DAEInitializationStatus(IntEnum):
    SUCCESS = 0
    RESIDUAL_TOO_LARGE = 1
    NONLINEAR_FAILED = 2
    NONFINITE = 3


def _mask_tuple(value: ArrayLike, owner: str, /) -> tuple[bool, ...]:
    array = np.asarray(value)
    if array.dtype.kind != "b":
        raise TypeError(f"{owner} must have Boolean dtype.")
    if array.size == 0:
        raise ValueError(f"{owner} must not be empty.")
    return tuple(bool(item) for item in array.reshape((-1,)))


def _spec_id(
    mode: DAEInitializationMode,
    fixed_state: tuple[bool, ...] | None,
    fixed_rate: tuple[bool, ...] | None,
    /,
) -> str:
    digest = hashlib.sha256()
    digest.update(repr((mode, fixed_state, fixed_rate)).encode("utf-8"))
    return f"dae-initialization:{digest.hexdigest()}"


class DAEInitializationSpec(StrictModule):
    """Free/fixed state and state-rate contract for one consistency solve."""

    mode: DAEInitializationMode = eqx.field(static=True)
    fixed_state: tuple[bool, ...] | None = eqx.field(static=True)
    fixed_rate: tuple[bool, ...] | None = eqx.field(static=True)
    initialization_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: DAEInitializationMode = "index-one",
        /,
        *,
        fixed_state: ArrayLike | None = None,
        fixed_rate: ArrayLike | None = None,
    ):
        if mode not in ("index-one", "fixed-rate", "check", "custom"):
            raise ValueError("Unknown DAE initialization mode.")
        if mode == "custom":
            if fixed_state is None or fixed_rate is None:
                raise ValueError("Custom initialization requires both fixed masks.")
            state_mask = _mask_tuple(fixed_state, "fixed_state")
            rate_mask = _mask_tuple(fixed_rate, "fixed_rate")
            if len(state_mask) != len(rate_mask):
                raise ValueError("Custom fixed-state and fixed-rate masks must match.")
        else:
            if fixed_state is not None or fixed_rate is not None:
                raise ValueError(
                    "fixed_state and fixed_rate are only accepted in custom mode."
                )
            state_mask = None
            rate_mask = None
        self.mode = mode
        self.fixed_state = state_mask
        self.fixed_rate = rate_mask
        self.initialization_id = _spec_id(mode, state_mask, rate_mask)

    @classmethod
    def index_one(cls) -> "DAEInitializationSpec":
        return cls("index-one")

    @classmethod
    def fixed_rate_state(cls) -> "DAEInitializationSpec":
        return cls("fixed-rate")

    @classmethod
    def check_only(cls) -> "DAEInitializationSpec":
        return cls("check")

    @classmethod
    def from_masks(
        cls,
        fixed_state: ArrayLike,
        fixed_rate: ArrayLike,
        /,
    ) -> "DAEInitializationSpec":
        return cls("custom", fixed_state=fixed_state, fixed_rate=fixed_rate)


class DAEInitializationResult(StrictModule):
    """Consistent state/rate pair with native nonlinear and constraint evidence."""

    state: Array
    state_rate: Array
    state_correction: Array
    rate_correction: Array
    fixed_state_mask: Array
    fixed_rate_mask: Array
    rate_valid: Array
    residual_norm: Array
    residual_threshold: Array
    differential_residual_norm: Array
    constraint_norm: Array
    valid: Array
    status: Array
    nonlinear_result: NonlinearResult | None
    initialization_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state: Array,
        state_rate: Array,
        state_correction: Array,
        rate_correction: Array,
        fixed_state_mask: Array,
        fixed_rate_mask: Array,
        rate_valid: Array,
        residual_norm: Array,
        residual_threshold: Array,
        differential_residual_norm: Array,
        constraint_norm: Array,
        valid: Array,
        status: Array,
        nonlinear_result: NonlinearResult | None,
        initialization_id: str,
    ):
        if nonlinear_result is not None and not isinstance(
            nonlinear_result, NonlinearResult
        ):
            raise TypeError("nonlinear_result must be a NonlinearResult or None.")
        self.state = jnp.asarray(state)
        self.state_rate = jnp.asarray(state_rate)
        self.state_correction = jnp.asarray(state_correction)
        self.rate_correction = jnp.asarray(rate_correction)
        self.fixed_state_mask = jnp.asarray(fixed_state_mask, dtype=bool)
        self.fixed_rate_mask = jnp.asarray(fixed_rate_mask, dtype=bool)
        self.rate_valid = jnp.asarray(rate_valid, dtype=bool)
        self.residual_norm = jnp.asarray(residual_norm)
        self.residual_threshold = jnp.asarray(residual_threshold)
        self.differential_residual_norm = jnp.asarray(differential_residual_norm)
        self.constraint_norm = jnp.asarray(constraint_norm)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.nonlinear_result = nonlinear_result
        self.initialization_id = str(initialization_id)

    @property
    def nonlinear_iterations(self) -> Array:
        if self.nonlinear_result is None:
            return jnp.asarray(0, dtype=jnp.int32)
        return self.nonlinear_result.diagnostics.iterations

    @property
    def linear_iterations(self) -> Array:
        if self.nonlinear_result is None:
            return jnp.asarray(0, dtype=jnp.int32)
        return self.nonlinear_result.diagnostics.linear_iterations


class _DAEInitializationArguments(StrictModule):
    time: Array
    state_guess: Array
    rate_guess: Array
    model_args: Any


class _DAEInitializationResidual(StrictModule):
    system: DifferentialAlgebraicSystem
    input_policy: AbstractInputPolicy | None
    state_indices: Array
    rate_indices: Array
    state_unknown_count: int = eqx.field(static=True)

    def __call__(
        self,
        unknown: Array,
        arguments: _DAEInitializationArguments,
        /,
    ) -> Array:
        flat_state = (
            arguments.state_guess.reshape((-1,))
            .at[self.state_indices]
            .set(unknown[: self.state_unknown_count])
        )
        flat_rate = (
            arguments.rate_guess.reshape((-1,))
            .at[self.rate_indices]
            .set(unknown[self.state_unknown_count :])
        )
        state = flat_state.reshape(self.system.state_shape)
        state_rate = flat_rate.reshape(self.system.state_shape)
        inputs = (
            None
            if self.input_policy is None
            else self.input_policy.evaluate(arguments.time, state, arguments.model_args)
        )
        return self.system.scaled_residual(
            arguments.time,
            state,
            state_rate,
            arguments.model_args,
            inputs=inputs,
        ).reshape(unknown.shape)


class _PreparedDAEInitialization(StrictModule):
    system: DifferentialAlgebraicSystem
    input_policy: AbstractInputPolicy | None
    spec: DAEInitializationSpec
    fixed_state_mask: Array
    fixed_rate_mask: Array
    state_indices: Array
    rate_indices: Array
    nonlinear_problem: NonlinearSystemProblem | None
    nonlinear_solve: PreparedNonlinearSolve | None
    state_unknown_count: int = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)


def _role_mask(
    system: DifferentialAlgebraicSystem,
    roles: tuple[str, ...],
    selected: str,
    /,
) -> np.ndarray:
    axis = system.structure.resolved_axis(system.state_shape)
    if axis is None:
        return np.full(system.state_shape, roles[0] == selected, dtype=bool)
    component = np.asarray(tuple(role == selected for role in roles), dtype=bool)
    reshape = [1] * len(system.state_shape)
    reshape[axis] = component.size
    return np.broadcast_to(component.reshape(tuple(reshape)), system.state_shape)


def _fixed_masks(
    system: DifferentialAlgebraicSystem,
    spec: DAEInitializationSpec,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if spec.mode == "index-one":
        fixed_state = _role_mask(
            system,
            system.structure.variable_roles,
            "differential",
        )
        fixed_rate = _role_mask(
            system,
            system.structure.variable_roles,
            "algebraic",
        )
    elif spec.mode == "fixed-rate":
        fixed_state = np.zeros(system.state_shape, dtype=bool)
        fixed_rate = np.ones(system.state_shape, dtype=bool)
    elif spec.mode == "check":
        fixed_state = np.ones(system.state_shape, dtype=bool)
        fixed_rate = np.ones(system.state_shape, dtype=bool)
    else:
        assert spec.fixed_state is not None and spec.fixed_rate is not None
        if len(spec.fixed_state) != system.state_size:
            raise ValueError(
                "Custom DAE initialization masks must contain exactly "
                f"{system.state_size} entries."
            )
        fixed_state = np.asarray(spec.fixed_state, dtype=bool).reshape(system.state_shape)
        fixed_rate = np.asarray(spec.fixed_rate, dtype=bool).reshape(system.state_shape)
    if spec.mode != "check":
        free_count = int(np.count_nonzero(~fixed_state) + np.count_nonzero(~fixed_rate))
        if free_count != system.state_size:
            raise ValueError(
                "A square DAE consistency solve requires exactly one free state/rate "
                f"unknown per residual scalar; got {free_count} free values and "
                f"{system.state_size} residuals."
            )
    return fixed_state, fixed_rate


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _validated_guesses(
    system: DifferentialAlgebraicSystem,
    state: ArrayLike,
    state_rate: ArrayLike,
    /,
) -> tuple[Array, Array]:
    state_array = _inexact(state)
    rate_array = _inexact(state_rate)
    if state_array.shape != system.state_shape or rate_array.shape != system.state_shape:
        raise ValueError(
            f"Initial state and rate must both have shape {system.state_shape}."
        )
    if state_array.dtype != rate_array.dtype:
        raise TypeError("Initial state and rate must have the same dtype.")
    state_array = eqx.error_if(
        state_array,
        jnp.any(~jnp.isfinite(state_array)) | jnp.any(~jnp.isfinite(rate_array)),
        "DAE initial state and rate must be finite.",
    )
    state_array = eqx.error_if(
        state_array,
        ~jnp.asarray(system.state_geometry.contains(state_array), dtype=bool),
        "DAE initial state is outside its state geometry.",
    )
    return state_array, rate_array


def _scaled_space(
    shape: tuple[int, ...],
    dtype: Any,
    scale: Array,
    /,
    *,
    space_id: str,
) -> ArraySpace:
    count = prod(shape)
    weights = 1.0 / (count * jnp.square(scale.astype(dtype)))
    return ArraySpace(
        shape,
        dtype=dtype,
        pairing=DiagonalPairing(weights, pairing_id=f"{space_id}:pairing"),
        space_id=space_id,
    )


def _masked_rms(values: Array, mask: Array, /) -> Array:
    count = jnp.sum(mask)
    squared = jnp.where(mask, jnp.square(jnp.abs(values)), 0.0)
    return jnp.sqrt(jnp.sum(squared) / jnp.maximum(count, 1))


def _unknown_guess(
    state: Array,
    state_rate: Array,
    state_indices: Array,
    rate_indices: Array,
    /,
) -> Array:
    return jnp.concatenate(
        (
            state.reshape((-1,))[state_indices],
            state_rate.reshape((-1,))[rate_indices],
        )
    )


def _prepare_dae_initialization(
    system: DifferentialAlgebraicSystem,
    initial_state: ArrayLike,
    initial_state_rate: ArrayLike,
    time: ArrayLike,
    /,
    *,
    args: Any,
    input_policy: AbstractInputPolicy | None,
    spec: DAEInitializationSpec,
    method: AbstractNonlinearMethod,
    termination: NonlinearTermination,
) -> _PreparedDAEInitialization:
    if not isinstance(system, DifferentialAlgebraicSystem):
        raise TypeError("system must be a DifferentialAlgebraicSystem.")
    if input_policy is not None and not isinstance(
        input_policy, AbstractInputPolicy
    ):
        raise TypeError("input_policy must be an AbstractInputPolicy or None.")
    if not isinstance(spec, DAEInitializationSpec):
        raise TypeError("spec must be a DAEInitializationSpec.")
    state, state_rate = _validated_guesses(system, initial_state, initial_state_rate)
    fixed_state_host, fixed_rate_host = _fixed_masks(system, spec)
    fixed_state = jnp.asarray(fixed_state_host, dtype=bool)
    fixed_rate = jnp.asarray(fixed_rate_host, dtype=bool)
    state_indices = jnp.asarray(np.flatnonzero(~fixed_state_host), dtype=jnp.int32)
    rate_indices = jnp.asarray(np.flatnonzero(~fixed_rate_host), dtype=jnp.int32)
    state_unknown_count = int(state_indices.size)

    if spec.mode == "check":
        nonlinear_problem = None
        nonlinear_solve = None
        linear_plan_id = "check-only"
    else:
        residual_function = _DAEInitializationResidual(
            system,
            input_policy,
            state_indices,
            rate_indices,
            state_unknown_count,
        )
        unknown = _unknown_guess(
            state,
            state_rate,
            state_indices,
            rate_indices,
        )
        flat_state_scale = system.state_scale.astype(state.dtype).reshape((-1,))
        flat_rate_scale = system.state_rate_scale.astype(state.dtype).reshape((-1,))
        unknown_scale = jnp.concatenate(
            (
                flat_state_scale[state_indices],
                flat_rate_scale[rate_indices],
            )
        )
        state_space = _scaled_space(
            unknown.shape,
            unknown.dtype,
            unknown_scale,
            space_id=f"{system.system_id}:{spec.initialization_id}:unknowns",
        )
        residual_space = _scaled_space(
            unknown.shape,
            unknown.dtype,
            jnp.ones_like(unknown_scale),
            space_id=f"{system.system_id}:{spec.initialization_id}:residuals",
        )
        nonlinear_problem = NonlinearSystemProblem(
            residual_function,
            state_space=state_space,
            residual_space=residual_space,
            problem_id=f"{system.system_id}:{spec.initialization_id}:root",
        )
        arguments = _DAEInitializationArguments(
            jnp.asarray(time),
            state,
            state_rate,
            args,
        )
        nonlinear_solve = prepare_nonlinear(
            nonlinear_problem,
            unknown,
            method=method,
            termination=termination,
            args=arguments,
        )
        linear_plan_id = nonlinear_solve.linear_plan_id

    digest = hashlib.sha256()
    digest.update(
        repr(
            (
                system.system_id,
                spec.initialization_id,
                None if input_policy is None else input_policy.policy_id,
                method.method_id,
                linear_plan_id,
            )
        ).encode("utf-8")
    )
    return _PreparedDAEInitialization(
        system,
        input_policy,
        spec,
        fixed_state,
        fixed_rate,
        state_indices,
        rate_indices,
        nonlinear_problem,
        nonlinear_solve,
        state_unknown_count,
        f"prepared-dae-initialization:{digest.hexdigest()}",
    )


def _unpack_unknown(
    prepared: _PreparedDAEInitialization,
    unknown: Array,
    state_guess: Array,
    rate_guess: Array,
    /,
) -> tuple[Array, Array]:
    state_flat = (
        state_guess.reshape((-1,))
        .at[prepared.state_indices]
        .set(unknown[: prepared.state_unknown_count])
    )
    rate_flat = (
        rate_guess.reshape((-1,))
        .at[prepared.rate_indices]
        .set(unknown[prepared.state_unknown_count :])
    )
    return (
        state_flat.reshape(prepared.system.state_shape),
        rate_flat.reshape(prepared.system.state_shape),
    )


def _initialize_dae(
    prepared: _PreparedDAEInitialization,
    initial_state: ArrayLike,
    initial_state_rate: ArrayLike,
    time: ArrayLike,
    /,
    *,
    args: Any,
    termination: NonlinearTermination,
) -> DAEInitializationResult:
    if not isinstance(prepared, _PreparedDAEInitialization):
        raise TypeError("prepared must be prepared DAE initialization data.")
    state_guess, rate_guess = _validated_guesses(
        prepared.system,
        initial_state,
        initial_state_rate,
    )
    arguments = _DAEInitializationArguments(
        jnp.asarray(time),
        state_guess,
        rate_guess,
        args,
    )

    if prepared.spec.mode == "check":
        state = state_guess
        state_rate = rate_guess
        nonlinear_result = None
        nonlinear_success = jnp.asarray(True)
        inputs = (
            None
            if prepared.input_policy is None
            else prepared.input_policy.evaluate(arguments.time, state, args)
        )
        initial_residual_norm = _masked_rms(
            prepared.system.scaled_residual(
                arguments.time,
                state,
                state_rate,
                args,
                inputs=inputs,
            ),
            jnp.ones(prepared.system.state_shape, dtype=bool),
        )
    else:
        assert prepared.nonlinear_problem is not None
        assert prepared.nonlinear_solve is not None
        unknown = _unknown_guess(
            state_guess,
            rate_guess,
            prepared.state_indices,
            prepared.rate_indices,
        )
        refreshed = refresh_nonlinear(
            prepared.nonlinear_solve,
            prepared.nonlinear_problem,
            unknown,
            args=arguments,
        )
        nonlinear_result = implicit_root_result(refreshed)
        state, state_rate = _unpack_unknown(
            prepared,
            nonlinear_result.state,
            state_guess,
            rate_guess,
        )
        nonlinear_success = nonlinear_result.status == int(NonlinearStatus.SUCCESS)
        initial_residual_norm = nonlinear_result.diagnostics.initial_residual_norm

    inputs = (
        None
        if prepared.input_policy is None
        else prepared.input_policy.evaluate(arguments.time, state, args)
    )
    scaled_residual = prepared.system.scaled_residual(
        arguments.time,
        state,
        state_rate,
        args,
        inputs=inputs,
    )
    differential_equations = prepared.system.structure.differential_equation_mask(
        prepared.system.state_shape
    )
    algebraic_equations = prepared.system.structure.algebraic_equation_mask(
        prepared.system.state_shape
    )
    residual_norm = _masked_rms(
        scaled_residual,
        jnp.ones(prepared.system.state_shape, dtype=bool),
    )
    residual_threshold = termination.residual_threshold(initial_residual_norm)
    differential_norm = _masked_rms(scaled_residual, differential_equations)
    constraint_norm = _masked_rms(scaled_residual, algebraic_equations)
    finite = (
        jnp.all(jnp.isfinite(state))
        & jnp.all(jnp.isfinite(state_rate))
        & jnp.isfinite(residual_norm)
    )
    residual_accepted = residual_norm <= residual_threshold
    valid = nonlinear_success & finite & residual_accepted
    status = jnp.where(
        ~finite,
        int(DAEInitializationStatus.NONFINITE),
        jnp.where(
            ~nonlinear_success,
            int(DAEInitializationStatus.NONLINEAR_FAILED),
            jnp.where(
                ~residual_accepted,
                int(DAEInitializationStatus.RESIDUAL_TOO_LARGE),
                int(DAEInitializationStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    rate_valid = (
        prepared.system.structure.differential_variable_mask(prepared.system.state_shape)
        if prepared.spec.mode == "index-one"
        else jnp.ones(prepared.system.state_shape, dtype=bool)
    )
    return DAEInitializationResult(
        state=state,
        state_rate=state_rate,
        state_correction=state - state_guess,
        rate_correction=state_rate - rate_guess,
        fixed_state_mask=prepared.fixed_state_mask,
        fixed_rate_mask=prepared.fixed_rate_mask,
        rate_valid=rate_valid & valid,
        residual_norm=residual_norm,
        residual_threshold=residual_threshold,
        differential_residual_norm=differential_norm,
        constraint_norm=constraint_norm,
        valid=valid,
        status=status,
        nonlinear_result=nonlinear_result,
        initialization_id=prepared.spec.initialization_id,
    )


__all__ = [
    "DAEInitializationMode",
    "DAEInitializationResult",
    "DAEInitializationSpec",
    "DAEInitializationStatus",
]
