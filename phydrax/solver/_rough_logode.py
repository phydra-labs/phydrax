#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..stochastic import AbstractRoughControl, LogSignatureControl, PrimitiveBasis
from ._matrix_functions import MatrixFunctionPolicy, matrix_exponential_action
from ._rough import (
    AbstractRoughSolver,
    RoughDifferentialProblem,
    _fractional_hurst,
)
from ._rough_lift import LiftedRoughVectorFields, lift_rough_vector_fields


def _validate_log_control(
    problem: RoughDifferentialProblem,
    control: AbstractRoughControl,
    /,
) -> LogSignatureControl:
    if not isinstance(control, LogSignatureControl):
        raise TypeError("Log-ODE solvers require a LogSignatureControl.")
    expected_dimension = problem.driver_dimension + int(control.joint_time)
    if control.dimension != expected_dimension:
        raise ValueError(
            "Log-signature dimension must equal the problem driver dimension plus "
            "its optional time channel."
        )
    if not control.joint_time and (problem.has_drift or problem.time_dependent):
        raise ValueError(
            "Drift or time-dependent fields require a joint-time log-signature control."
        )
    hurst = _fractional_hurst(control)
    threshold = 1.0 / float(control.depth + 1)
    if hurst is not None and hurst <= threshold:
        raise ValueError(
            f"Control depth {control.depth} requires fractional Gaussian Hurst > "
            f"1/(depth + 1) = {threshold:g}; got {hurst:g}."
        )
    return control


def _projected_local_field(retraction, local: Array, tangent: Array, /) -> Array:
    point = retraction(local)
    projected = retraction.geometry.project_tangent(point, tangent)
    return retraction.pullback(local, projected)


def _automatic_log_field(
    problem: RoughDifferentialProblem,
    control: LogSignatureControl,
    retraction,
    left_time: Array,
    coordinates: Array,
    coefficients: Array,
    /,
) -> Array:
    state_size = int(problem.initial_state.size)
    if control.joint_time:

        def augmented_fields(_time, augmented, _args):
            physical_time = augmented[0]
            local = augmented[1:].reshape(problem.state_shape)
            state = retraction(local)
            rough_fields = jnp.asarray(
                problem.vector_fields(physical_time, state, problem.args)
            )
            drift = jnp.asarray(problem.drift(physical_time, state, problem.args))
            ambient = jnp.concatenate((drift[..., None], rough_fields), axis=-1)
            local_fields = jax.vmap(
                lambda tangent: _projected_local_field(
                    retraction, local, tangent
                ),
                in_axes=-1,
                out_axes=-1,
            )(ambient)
            time_fields = jnp.concatenate(
                (
                    jnp.ones((1,), dtype=augmented.dtype),
                    jnp.zeros((problem.driver_dimension,), dtype=augmented.dtype),
                )
            )
            return jnp.concatenate(
                (time_fields[None, :], local_fields.reshape((state_size, -1))),
                axis=0,
            )

        lifted = lift_rough_vector_fields(
            augmented_fields,
            control.primitive_basis,
            jnp.asarray(0.0),
            coordinates,
            None,
        )
    else:

        def local_fields(_time, local_flat, _args):
            local = local_flat.reshape(problem.state_shape)
            state = retraction(local)
            ambient = jnp.asarray(
                problem.vector_fields(left_time, state, problem.args)
            )
            pulled_back = jax.vmap(
                lambda tangent: _projected_local_field(
                    retraction, local, tangent
                ),
                in_axes=-1,
                out_axes=-1,
            )(ambient)
            return pulled_back.reshape((state_size, control.dimension))

        lifted = lift_rough_vector_fields(
            local_fields,
            control.primitive_basis,
            jnp.asarray(0.0),
            coordinates,
            None,
        )
    return jnp.tensordot(lifted, coefficients, axes=((-1,), (0,)))


def _explicit_log_field(
    problem: RoughDifferentialProblem,
    control: LogSignatureControl,
    retraction,
    left_time: Array,
    coordinates: Array,
    coefficients: Array,
    explicit_fields: LiftedRoughVectorFields,
    /,
) -> Array:
    state_size = int(problem.initial_state.size)
    if control.joint_time:
        physical_time = coordinates[0]
        local = coordinates[1:].reshape(problem.state_shape)
    else:
        physical_time = left_time
        local = coordinates.reshape(problem.state_shape)
    state = retraction(local)
    ambient = jnp.asarray(explicit_fields(physical_time, state, problem.args))
    expected = problem.state_shape + (control.primitive_basis.size,)
    if ambient.shape != expected:
        raise ValueError(
            f"explicit_fields must return shape {expected}; got {ambient.shape}."
        )
    local_fields = jax.vmap(
        lambda tangent: _projected_local_field(retraction, local, tangent),
        in_axes=-1,
        out_axes=-1,
    )(ambient)
    if control.joint_time:
        time_components = jnp.asarray(
            [1.0 if word == (0,) else 0.0 for word in control.primitive_basis.words],
            dtype=coordinates.dtype,
        )
        lifted = jnp.concatenate(
            (time_components[None, :], local_fields.reshape((state_size, -1))),
            axis=0,
        )
    else:
        lifted = local_fields.reshape((state_size, -1))
    return jnp.tensordot(lifted, coefficients, axes=((-1,), (0,)))


class LogODE(AbstractRoughSolver):
    """Solve each log-signature interval as a configurable Diffrax ODE."""

    ode_solver: Any
    stepsize_controller: Any
    adjoint: Any
    explicit_fields: LiftedRoughVectorFields | None
    dt0: float | None = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)
    required_depth: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        ode_solver: Any = dfx.Tsit5(),
        stepsize_controller: Any = dfx.PIDController(rtol=1e-7, atol=1e-9),
        adjoint: Any = dfx.RecursiveCheckpointAdjoint(),
        dt0: float | None = None,
        max_steps: int = 4096,
        explicit_fields: LiftedRoughVectorFields | None = None,
    ):
        if int(max_steps) <= 0:
            raise ValueError("max_steps must be positive.")
        if explicit_fields is not None and not callable(explicit_fields):
            raise TypeError("explicit_fields must be callable or None.")
        self.ode_solver = ode_solver
        self.stepsize_controller = stepsize_controller
        self.adjoint = adjoint
        self.explicit_fields = explicit_fields
        self.dt0 = None if dt0 is None else float(dt0)
        self.max_steps = int(max_steps)
        self.solver_name = "LogODE"
        self.required_depth = 1

    def integrate(
        self,
        problem: RoughDifferentialProblem,
        control: AbstractRoughControl,
        /,
    ) -> tuple[Array, Array, Mapping[str, Array]]:
        log_control = _validate_log_control(problem, control)
        state_size = int(problem.initial_state.size)

        def one_path(path_coefficients):
            def advance(state, item):
                left_time, right_time, coefficients = item
                retraction = problem.geometry.local_retraction(state)
                local_zero = jnp.zeros_like(state).reshape((state_size,))
                if log_control.joint_time:
                    initial_coordinates = jnp.concatenate(
                        (jnp.asarray(left_time).reshape((1,)), local_zero)
                    )
                else:
                    initial_coordinates = local_zero

                def vector_field(_artificial_time, coordinates, _args):
                    if self.explicit_fields is None:
                        return _automatic_log_field(
                            problem,
                            log_control,
                            retraction,
                            left_time,
                            coordinates,
                            coefficients,
                        )
                    return _explicit_log_field(
                        problem,
                        log_control,
                        retraction,
                        left_time,
                        coordinates,
                        coefficients,
                        self.explicit_fields,
                    )

                native = dfx.diffeqsolve(
                    dfx.ODETerm(vector_field),
                    self.ode_solver,
                    t0=0.0,
                    t1=1.0,
                    dt0=self.dt0,
                    y0=initial_coordinates,
                    saveat=dfx.SaveAt(t1=True),
                    stepsize_controller=self.stepsize_controller,
                    adjoint=self.adjoint,
                    max_steps=self.max_steps,
                    throw=False,
                )
                final_coordinates = native.ys[0]
                final_local = (
                    final_coordinates[1:].reshape(problem.state_shape)
                    if log_control.joint_time
                    else final_coordinates.reshape(problem.state_shape)
                )
                next_state = retraction(final_local)
                status = jnp.asarray(native.result._value, dtype=jnp.int32)
                stats = (
                    jnp.asarray(native.stats["num_steps"], dtype=jnp.int32),
                    jnp.asarray(native.stats["num_accepted_steps"], dtype=jnp.int32),
                    jnp.asarray(native.stats["num_rejected_steps"], dtype=jnp.int32),
                )
                return next_state, (next_state, status, stats)

            _, (stepped, statuses, stats) = jax.lax.scan(
                advance,
                problem.initial_state,
                (
                    log_control.times[:-1],
                    log_control.times[1:],
                    path_coefficients,
                ),
            )
            states = jnp.concatenate(
                (problem.initial_state[None, ...], stepped), axis=0
            )
            return states, statuses, stats

        if log_control.sample_shape:
            path_count = int(np.prod(log_control.sample_shape))
            coefficients = log_control.log_coefficients.reshape(
                (path_count, log_control.num_steps, log_control.primitive_basis.size)
            )
            states, statuses, stats = jax.vmap(one_path)(coefficients)
            states = states.reshape(
                log_control.sample_shape
                + (log_control.num_steps + 1,)
                + problem.state_shape
            )
            statuses = statuses.reshape(
                log_control.sample_shape + (log_control.num_steps,)
            )
            stats = tuple(
                value.reshape(log_control.sample_shape + (log_control.num_steps,))
                for value in stats
            )
        else:
            states, statuses, stats = one_path(log_control.log_coefficients)
        statistics = {
            "num_steps": stats[0],
            "num_accepted_steps": stats[1],
            "num_rejected_steps": stats[2],
        }
        return states, statuses, statistics


class _MatrixOperator(eqx.Module):
    matrix: Array

    def __call__(self, value: Array, /) -> Array:
        return (self.matrix @ value.reshape((-1,))).reshape(value.shape)


class _CommutatorOperator(eqx.Module):
    left: Callable[[Array], ArrayLike]
    right: Callable[[Array], ArrayLike]

    def __call__(self, value: Array, /) -> Array:
        return jnp.asarray(self.right(jnp.asarray(self.left(value)))) - jnp.asarray(
            self.left(jnp.asarray(self.right(value)))
        )


class _WeightedOperator(eqx.Module):
    operators: tuple[Callable[[Array], ArrayLike], ...]
    coefficients: Array

    def __call__(self, value: Array, /) -> Array:
        images = jnp.stack(
            tuple(jnp.asarray(operator(value)) for operator in self.operators), axis=0
        )
        return jnp.tensordot(self.coefficients, images, axes=((0,), (0,)))


def _operator(value: Any, /) -> Callable[[Array], ArrayLike]:
    if callable(value):
        return value
    matrix = jnp.asarray(value)
    if matrix.ndim != 2 or int(matrix.shape[0]) != int(matrix.shape[1]):
        raise ValueError("A matrix linear operator must be square.")
    return _MatrixOperator(matrix)


def _lift_linear_operators(
    operators: tuple[Callable[[Array], ArrayLike], ...], basis: PrimitiveBasis, /
) -> tuple[Callable[[Array], ArrayLike], ...]:
    lifted: list[Callable[[Array], ArrayLike]] = []
    for word, children in zip(basis.words, basis.children):
        if children is None:
            lifted.append(operators[word[0]])
        else:
            left_index, right_index = children
            lifted.append(
                _CommutatorOperator(lifted[left_index], lifted[right_index])
            )
    return tuple(lifted)


class LinearLogODE(AbstractRoughSolver):
    """Explicit linear/operator log-ODE specialization using commutator lifts."""

    operators: tuple[Callable[[Array], ArrayLike], ...]
    matrix_function_policy: MatrixFunctionPolicy | None
    solver_name: str = eqx.field(static=True)
    required_depth: int = eqx.field(static=True)

    def __init__(
        self,
        operators: Sequence[Any],
        /,
        *,
        matrix_function_policy: MatrixFunctionPolicy | None = None,
    ):
        resolved = tuple(_operator(operator) for operator in operators)
        if not resolved:
            raise ValueError("operators must be non-empty.")
        if matrix_function_policy is not None and not isinstance(
            matrix_function_policy, MatrixFunctionPolicy
        ):
            raise TypeError(
                "matrix_function_policy must be a MatrixFunctionPolicy or None."
            )
        self.operators = resolved
        self.matrix_function_policy = matrix_function_policy
        self.solver_name = "LinearLogODE"
        self.required_depth = 1

    def integrate(
        self,
        problem: RoughDifferentialProblem,
        control: AbstractRoughControl,
        /,
    ) -> tuple[Array, Array, Mapping[str, Array]]:
        log_control = _validate_log_control(problem, control)
        if problem.time_dependent:
            raise ValueError(
                "LinearLogODE only supports autonomous explicit operators."
            )
        if not problem.geometry.trivial:
            raise ValueError("LinearLogODE requires a trivial Euclidean state geometry.")
        if len(self.operators) != log_control.dimension:
            raise ValueError(
                "LinearLogODE requires one explicit operator per control channel."
            )
        for operator in self.operators:
            image = jnp.asarray(operator(problem.initial_state))
            if image.shape != problem.state_shape:
                raise ValueError("Each linear operator must preserve the state shape.")
        lifted = _lift_linear_operators(
            self.operators, log_control.primitive_basis
        )

        def one_path(path_coefficients):
            def advance(state, coefficients):
                combined = _WeightedOperator(lifted, coefficients)
                next_state = matrix_exponential_action(
                    combined,
                    state,
                    jnp.asarray(1.0, dtype=state.dtype),
                    policy=self.matrix_function_policy,
                )
                return next_state, next_state

            _, stepped = jax.lax.scan(
                advance, problem.initial_state, path_coefficients
            )
            return jnp.concatenate(
                (problem.initial_state[None, ...], stepped), axis=0
            )

        if log_control.sample_shape:
            path_count = int(np.prod(log_control.sample_shape))
            coefficients = log_control.log_coefficients.reshape(
                (path_count, log_control.num_steps, log_control.primitive_basis.size)
            )
            states = jax.vmap(one_path)(coefficients).reshape(
                log_control.sample_shape
                + (log_control.num_steps + 1,)
                + problem.state_shape
            )
        else:
            states = one_path(log_control.log_coefficients)
        interval_shape = log_control.sample_shape + (log_control.num_steps,)
        statuses = jnp.zeros(interval_shape, dtype=jnp.int32)
        statistics = {
            "num_steps": jnp.ones(interval_shape, dtype=jnp.int32),
            "num_accepted_steps": jnp.ones(interval_shape, dtype=jnp.int32),
            "num_rejected_steps": jnp.zeros(interval_shape, dtype=jnp.int32),
        }
        return states, statuses, statistics


__all__ = [
    "LinearLogODE",
    "LogODE",
]
