# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Autodiff preparation rereads opaque args and reuses explicit numerical leaves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from tools.compile_counter import CompileCounter


nl = phx.nonlinear


@dataclass
class _Multiplier:
    scale: float

    def value(self) -> float:
        return self.scale


@dataclass(eq=False)
class _OpaqueArgs:
    coefficient: float
    scales: list[float]
    helper: _Multiplier


class _OpaqueFloatArgsResidual(phx.StrictModule):
    diagonal: Array
    target: Array

    def __call__(self, state: Array, args: _OpaqueArgs) -> Array:
        return self.diagonal * (0.5 * args.coefficient) * state - self.target


class _OpaqueArgsResidual(phx.StrictModule):
    diagonal: Array
    target: Array

    def __call__(self, state: Array, args: _OpaqueArgs) -> Array:
        return (
            self.diagonal
            * args.coefficient
            * args.scales[0]
            * args.helper.value()
            * state
            - self.target
        )


class _ExplicitArgs(phx.StrictModule):
    coefficient: Array
    target: Array


class _ExplicitArgsResidual(phx.StrictModule):
    diagonal: Array

    def __call__(self, state: Array, args: _ExplicitArgs) -> Array:
        return self.diagonal * args.coefficient * state - args.target


def _mutate_args(
    args: _OpaqueArgs, mutation: Literal["float", "container", "helper"]
) -> None:
    match mutation:
        case "float":
            args.coefficient = 4.0
        case "container":
            args.scales[0] = 1.5
        case "helper":
            args.helper.scale = 4.0


@pytest.mark.parametrize("method_type", [nl.NewtonKrylov, nl.NewtonTrustRegion])
@pytest.mark.parametrize(
    ("mutation", "next_factor"),
    [("float", 2.0), ("container", 3.0), ("helper", 4.0)],
)
def test_solve_rereads_same_opaque_args(
    method_type: type[nl.NewtonKrylov] | type[nl.NewtonTrustRegion],
    mutation: Literal["float", "container", "helper"],
    next_factor: float,
) -> None:
    residual_type = (
        _OpaqueFloatArgsResidual if mutation == "float" else _OpaqueArgsResidual
    )
    residual = residual_type(
        jnp.asarray([2.0, 5.0], dtype=jnp.float64),
        jnp.asarray([6.0, 20.0], dtype=jnp.float64),
    )
    problem = nl.NonlinearSystemProblem(residual)
    args = _OpaqueArgs(2.0, [0.5], _Multiplier(1.0))
    method = method_type()
    termination = nl.NonlinearTermination(
        absolute_residual=1e-12,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=12,
    )
    initial = jnp.ones((2,), dtype=jnp.float64)
    first = method.solve(problem, initial, args=args, termination=termination)
    _mutate_args(args, mutation)
    second = method.solve(problem, initial, args=args, termination=termination)

    # The diagonal affine system has roots (3, 4) / factor independently of Newton.
    # 64 eps bounds two-coordinate linear-solve and scalar-arithmetic roundoff.
    roundoff = 64 * np.finfo(np.float64).eps
    for result, factor in ((first, 1.0), (second, next_factor)):
        assert bool(result.successful)
        expected_root = np.asarray([3.0, 4.0], dtype=np.float64) / factor
        np.testing.assert_allclose(result.state, expected_root, rtol=roundoff, atol=0.0)
        analytic_residual = np.asarray(
            [2.0, 5.0], dtype=np.float64
        ) * factor * np.asarray(result.state) - np.asarray([6.0, 20.0], dtype=np.float64)
        np.testing.assert_allclose(analytic_residual, 0.0, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            result.residual, analytic_residual, rtol=0.0, atol=1e-12
        )
    assert not np.array_equal(np.asarray(first.state), np.asarray(second.state))


@pytest.mark.parametrize(
    ("mutation", "next_factor"),
    [("float", 2.0), ("container", 3.0), ("helper", 4.0)],
)
def test_prepared_jacobian_rereads_same_opaque_args(
    mutation: Literal["float", "container", "helper"], next_factor: float
) -> None:
    residual_type = (
        _OpaqueFloatArgsResidual if mutation == "float" else _OpaqueArgsResidual
    )
    residual = residual_type(
        jnp.asarray([2.0, 5.0], dtype=jnp.float64),
        jnp.asarray([6.0, 20.0], dtype=jnp.float64),
    )
    problem = nl.NonlinearSystemProblem(residual)
    args = _OpaqueArgs(2.0, [0.5], _Multiplier(1.0))
    state = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    tangent = jnp.asarray([0.25, -0.5], dtype=jnp.float64)
    cotangent = jnp.asarray([-0.5, 0.25], dtype=jnp.float64)
    policy = nl.JacobianPolicy()
    first = nl.prepare_jacobian(problem, state, policy, args)
    _mutate_args(args, mutation)
    second = nl.prepare_jacobian(problem, state, policy, args)

    # These binary-exact affine evaluations and diagonal actions have no roundoff.
    for prepared, factor in ((first, 1.0), (second, next_factor)):
        diagonal = np.asarray([2.0, 5.0], dtype=np.float64) * factor
        np.testing.assert_array_equal(
            prepared.residual,
            diagonal * np.asarray(state) - np.asarray([6.0, 20.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            prepared.operator.mv(tangent), diagonal * np.asarray(tangent)
        )
        np.testing.assert_array_equal(
            prepared.operator.transpose_mv(cotangent), diagonal * np.asarray(cotangent)
        )

    def current_residual(candidate: Array) -> Array:
        return residual(candidate, args)

    _, jvp = jax.jvp(current_residual, (state,), (tangent,))
    expected_jvp = (
        np.asarray([2.0, 5.0], dtype=np.float64) * next_factor * np.asarray(tangent)
    )
    np.testing.assert_array_equal(jvp, expected_jvp)
    np.testing.assert_array_equal(second.operator.mv(tangent), expected_jvp)


def test_explicit_pytree_args_and_callback_fields_reuse_warm_preparation() -> None:
    first_problem = nl.NonlinearSystemProblem(
        _ExplicitArgsResidual(jnp.asarray([2.0, 5.0], dtype=jnp.float64)),
        problem_id="explicit-args-warm-preparation",
    )
    next_problem = nl.NonlinearSystemProblem(
        _ExplicitArgsResidual(jnp.asarray([3.0, 7.0], dtype=jnp.float64)),
        problem_id="explicit-args-warm-preparation",
    )
    first_args = _ExplicitArgs(
        jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        jnp.asarray([6.0, 20.0], dtype=jnp.float64),
    )
    next_args = _ExplicitArgs(
        jnp.asarray([4.0, 0.5], dtype=jnp.float64),
        jnp.asarray([24.0, 14.0], dtype=jnp.float64),
    )
    first_state = jnp.asarray([0.5, -0.25], dtype=jnp.float64)
    next_state = jnp.asarray([0.25, 0.5], dtype=jnp.float64)
    tangent = jnp.asarray([0.25, -0.5], dtype=jnp.float64)
    policy = nl.JacobianPolicy()
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            nl.prepare_jacobian(first_problem, first_state, policy, first_args)
        )
    with CompileCounter(capture_stack=True) as warm:
        second = jax.block_until_ready(
            nl.prepare_jacobian(next_problem, next_state, policy, next_args)
        )
    assert cold.count > 0
    assert warm.count == 0, [event.to_record() for event in warm.events]

    # Independent diagonal entries: (2*1, 5*2) and (3*4, 7/2).
    for prepared, state, diagonal, target in (
        (first, first_state, [2.0, 10.0], [6.0, 20.0]),
        (second, next_state, [12.0, 3.5], [24.0, 14.0]),
    ):
        entries = np.asarray(diagonal, dtype=np.float64)
        np.testing.assert_array_equal(
            prepared.residual,
            entries * np.asarray(state) - np.asarray(target, dtype=np.float64),
        )
        np.testing.assert_array_equal(
            prepared.operator.mv(tangent), entries * np.asarray(tangent)
        )
        np.testing.assert_array_equal(
            prepared.operator.transpose_mv(tangent), entries * np.asarray(tangent)
        )
