#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


la = phx.linalg
SIZE = 12
_RANDOM = np.random.default_rng(7)
GENERAL = jnp.asarray(3.0 * np.eye(SIZE) + 0.5 * _RANDOM.standard_normal((SIZE, SIZE)))
_FACTOR = _RANDOM.standard_normal((SIZE, SIZE))
SPD = jnp.asarray(np.eye(SIZE) + 0.3 * _FACTOR @ _FACTOR.T)
COMPLEX = jnp.asarray(
    (2.5 + 0.5j) * np.eye(SIZE)
    + 0.15
    * (_RANDOM.standard_normal((SIZE, SIZE)) + 1j * _RANDOM.standard_normal((SIZE, SIZE)))
)
RHS = jnp.asarray(_RANDOM.standard_normal(SIZE))
COMPLEX_RHS = jnp.asarray(
    _RANDOM.standard_normal(SIZE) + 1j * _RANDOM.standard_normal(SIZE)
)
WEIGHTS = jnp.asarray(0.5 + _RANDOM.random(SIZE))


def _operator(matrix, *, space=None, spd=False):
    space_ = la.ArraySpace((SIZE,), dtype=matrix.dtype) if space is None else space
    properties = (
        la.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        )
        if spd
        else None
    )
    return la.FunctionLinearOperator(
        lambda value: matrix @ value,
        source=space_,
        target=space_,
        transpose_action=(lambda value: matrix.T @ value) if spd else None,
        properties=properties,
    )


def _policy(method, mode, *, max_steps, relative=0.0):
    return la.LinearSolvePolicy(
        method,
        tolerance=la.TolerancePolicy(
            relative=relative, absolute=0.0, max_steps=max_steps
        ),
        differentiation=la.DifferentiationPolicy(mode),
        failure=la.FailurePolicy("status"),
    )


def _solve(
    matrix,
    rhs,
    method,
    mode,
    *,
    max_steps,
    steps=None,
    space=None,
    spd=False,
    relative=0.0,
):
    problem = la.LinearSystem(_operator(matrix, space=space, spd=spd))
    control = None if steps is None else la.LinearSolveControl(maximum_steps=steps)
    return la.solve(
        problem,
        rhs,
        policy=_policy(method, mode, max_steps=max_steps, relative=relative),
        control=control,
    )


@pytest.mark.parametrize(
    ("matrix", "rhs", "method", "spd"),
    (
        (GENERAL, RHS, la.FGMRES(restart=4), False),
        (GENERAL, RHS, la.GMRES(restart=5), False),
        (SPD, RHS, la.PCG(), True),
        (COMPLEX, COMPLEX_RHS, la.FGMRES(restart=3), False),
    ),
)
def test_fixed_trip_and_early_exit_agree_on_every_iterate(matrix, rhs, method, spd):
    max_steps = 3 * SIZE

    def run(mode):
        return jax.jit(
            lambda steps: _solve(
                matrix,
                rhs,
                method,
                mode,
                max_steps=max_steps,
                steps=steps,
                spd=spd,
                relative=1.0e-10,
            )
        )

    early, fixed = run("none"), run("algorithmic")
    for steps in range(1, max_steps + 1):
        early_result = early(jnp.asarray(steps, dtype=jnp.int32))
        fixed_result = fixed(jnp.asarray(steps, dtype=jnp.int32))
        assert int(fixed_result.diagnostics.iterations) == int(
            early_result.diagnostics.iterations
        )
        assert int(fixed_result.status) == int(early_result.status)
        np.testing.assert_allclose(
            fixed_result.value, early_result.value, rtol=1e-12, atol=1e-14
        )
        if bool(early_result.successful):
            break
    assert bool(early_result.successful)


def test_fixed_trip_status_reports_the_capacity_limit():
    early = _solve(GENERAL, RHS, la.FGMRES(restart=3), "none", max_steps=5)
    fixed = _solve(GENERAL, RHS, la.FGMRES(restart=3), "algorithmic", max_steps=5)

    assert int(fixed.status) == int(la.LinearSolveStatus.MAXIMUM_STEPS_REACHED)
    assert int(fixed.status) == int(early.status)
    assert int(fixed.diagnostics.iterations) == 5
    assert int(fixed.diagnostics.matvec_count) == int(early.diagnostics.matvec_count)


def _central_difference(function, value, step=1.0e-6):
    return (function(value + step) - function(value - step)) / (2.0 * step)


@pytest.mark.parametrize(
    ("matrix", "method", "spd", "max_steps"),
    (
        (GENERAL, la.FGMRES(restart=3), False, 8),
        (GENERAL, la.GMRES(restart=2), False, 7),
        (SPD, la.PCG(), True, 7),
    ),
)
def test_reverse_mode_differentiates_the_executed_iteration_across_restarts(
    matrix, method, spd, max_steps
):
    direction = jnp.linspace(-1.0, 1.0, SIZE)

    @jax.jit
    def loss(scale):
        result = _solve(
            matrix + scale * jnp.diag(direction),
            RHS * (1.0 + scale),
            method,
            "algorithmic",
            max_steps=max_steps,
            spd=spd,
        )
        return jnp.sum(result.value**2)

    executed = _solve(matrix, RHS, method, "algorithmic", max_steps=max_steps, spd=spd)
    gradient = jax.jit(jax.grad(loss))(0.1)
    tangent = jax.jit(lambda scale: jax.jvp(loss, (scale,), (1.0,))[1])(0.1)

    assert int(executed.diagnostics.iterations) == max_steps
    np.testing.assert_allclose(gradient, _central_difference(loss, 0.1), rtol=1e-6)
    np.testing.assert_allclose(gradient, tangent, rtol=1e-10)


def test_complex_pairing_reverse_mode_matches_finite_differences():
    space = la.ArraySpace(
        (SIZE,), dtype=jnp.complex128, pairing=la.DiagonalPairing(WEIGHTS)
    )

    @jax.jit
    def loss(scale):
        result = _solve(
            COMPLEX * (1.0 + scale),
            COMPLEX_RHS,
            la.FGMRES(restart=3),
            "algorithmic",
            max_steps=7,
            space=space,
        )
        return jnp.sum(jnp.abs(result.value) ** 2)

    early = _solve(
        COMPLEX, COMPLEX_RHS, la.FGMRES(restart=3), "none", max_steps=7, space=space
    )
    fixed = _solve(
        COMPLEX,
        COMPLEX_RHS,
        la.FGMRES(restart=3),
        "algorithmic",
        max_steps=7,
        space=space,
    )

    np.testing.assert_allclose(fixed.value, early.value, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        jax.jit(jax.grad(loss))(0.05), _central_difference(loss, 0.05), rtol=1e-6
    )
