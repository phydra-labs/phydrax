#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
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


def _operator(matrix: Any, *, space: Any = None, spd: Any = False) -> Any:
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


def _policy(method: Any, mode: Any, *, max_steps: Any, relative: Any = 0.0) -> Any:
    return la.LinearSolvePolicy(
        method,
        tolerance=la.TolerancePolicy(
            relative=relative, absolute=0.0, max_steps=max_steps
        ),
        differentiation=la.DifferentiationPolicy(mode),
        failure=la.FailurePolicy("status"),
    )


def _solve(
    matrix: Any,
    rhs: Any,
    method: Any,
    mode: Any,
    *,
    max_steps: Any,
    steps: Any = None,
    space: Any = None,
    spd: Any = False,
    relative: Any = 0.0,
) -> Any:
    problem = la.LinearSystem(_operator(matrix, space=space, spd=spd))
    control = None if steps is None else la.LinearSolveControl(maximum_steps=steps)
    return la.solve(
        problem,
        rhs,
        policy=_policy(method, mode, max_steps=max_steps, relative=relative),
        control=control,
    )


def test_fixed_trip_and_early_exit_agree_on_every_iterate() -> None:
    for matrix, rhs, method, spd in (
        (GENERAL, RHS, la.FGMRES(restart=4), False),
        (GENERAL, RHS, la.GMRES(restart=5), False),
        (SPD, RHS, la.PCG(), True),
        (COMPLEX, COMPLEX_RHS, la.FGMRES(restart=3), False),
    ):
        max_steps = 3 * SIZE

        def run(mode: Any) -> Any:
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


def test_fixed_trip_status_reports_the_capacity_limit() -> None:
    early = _solve(GENERAL, RHS, la.FGMRES(restart=3), "none", max_steps=5)
    fixed = _solve(GENERAL, RHS, la.FGMRES(restart=3), "algorithmic", max_steps=5)

    assert int(fixed.status) == int(la.LinearSolveStatus.MAXIMUM_STEPS_REACHED)
    assert int(fixed.status) == int(early.status)
    assert int(fixed.diagnostics.iterations) == 5
    assert int(fixed.diagnostics.matvec_count) == int(early.diagnostics.matvec_count)


def _central_difference(function: Any, value: Any, step: Any = 1.0e-6) -> Any:
    return (function(value + step) - function(value - step)) / (2.0 * step)


def test_reverse_mode_differentiates_the_executed_iteration_across_restarts() -> None:
    for matrix, method, spd, max_steps in (
        (GENERAL, la.FGMRES(restart=3), False, 8),
        (GENERAL, la.GMRES(restart=2), False, 7),
        (SPD, la.PCG(), True, 7),
    ):
        direction = jnp.linspace(-1.0, 1.0, SIZE)

        @jax.jit
        def loss(scale: Any) -> Any:
            result = _solve(
                matrix + scale * jnp.diag(direction),
                RHS * (1.0 + scale),
                method,
                "algorithmic",
                max_steps=max_steps,
                spd=spd,
            )
            return jnp.sum(result.value**2)

        executed = _solve(
            matrix, RHS, method, "algorithmic", max_steps=max_steps, spd=spd
        )
        gradient = jax.jit(jax.grad(loss))(0.1)
        tangent = jax.jit(lambda scale: jax.jvp(loss, (scale,), (1.0,))[1])(0.1)

        assert int(executed.diagnostics.iterations) == max_steps
        np.testing.assert_allclose(gradient, _central_difference(loss, 0.1), rtol=1e-6)
        np.testing.assert_allclose(gradient, tangent, rtol=1e-10)


def test_complex_pairing_reverse_mode_matches_finite_differences() -> None:
    space = la.ArraySpace(
        (SIZE,), dtype=jnp.complex128, pairing=la.DiagonalPairing(WEIGHTS)
    )

    @jax.jit
    def loss(scale: Any) -> Any:
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


_EIGENVECTORS = jnp.asarray(np.linalg.eigh(np.asarray(SPD))[1][:, :2])


def _checked(value: jax.Array) -> jax.Array:
    invalid = ~jnp.all(jnp.isfinite(value)) | (jnp.sum(value * value) == 0.0)
    return eqx.error_if(value, invalid, "operator received a nonfinite or zero input")


def _lanes(scale: jax.Array, width: int) -> tuple[jax.Array, jax.Array]:
    """Right-hand sides and initial guesses of three lanes for ``scale * SPD``.

    Lane 0 converges in one step, lane 1 iterates, and lane 2 starts at its
    solution, so it never executes a step. ``width`` is the RHS column count;
    width one uses vectors.
    """
    eigenvectors = _EIGENVECTORS[:, :width]
    generic = jnp.stack([RHS, jnp.flip(RHS)], axis=1)[:, :width]
    guesses = jnp.stack([WEIGHTS, jnp.flip(WEIGHTS)], axis=1)[:, :width]
    rhs = jnp.stack([eigenvectors, generic, (scale * SPD) @ guesses])
    initial = jnp.stack([0.3 * eigenvectors, -guesses, guesses])
    if width == 1:
        return rhs[..., 0], initial[..., 0]
    return rhs, initial


@pytest.mark.parametrize(
    ("method", "spd", "mode", "width"),
    [
        (la.PCG(), True, "algorithmic", 1),
        (la.MINRES(), True, "algorithmic", 1),
        (la.FGMRES(restart=4), False, "algorithmic", 1),
        (la.GMRES(restart=4), False, "algorithmic", 1),
        (la.BlockCG(), True, "mathematical", 2),
        (la.BlockGMRES(restart=4), False, "mathematical", 2),
    ],
    ids=["pcg", "minres", "fgmres", "gmres", "block-cg", "block-gmres"],
)
def test_gated_off_lanes_never_feed_the_operator_invalid_inputs_in_reverse_mode(
    method: Any, spd: bool, mode: Any, width: int
) -> None:
    def solve(scale: jax.Array, lane: jax.Array) -> Any:
        rhs, initial = _lanes(scale, width)
        matrix = scale * SPD
        space = la.ArraySpace((SIZE,), dtype=matrix.dtype)
        properties = _operator(matrix, spd=spd).properties
        operator = la.FunctionLinearOperator(
            lambda value: matrix @ _checked(value),
            source=space,
            target=space,
            transpose_action=(lambda value: matrix.T @ _checked(value)) if spd else None,
            properties=properties,
        )
        policy = _policy(method, mode, max_steps=3 * SIZE, relative=1.0e-10)
        prepared = la.prepare(
            la.LinearSystem(operator),
            policy,
            rhs_layout=None if width == 1 else la.RHSLayout((width,)),
        )
        return la.solve(prepared, rhs[lane], initial_guess=initial[lane])

    def loss(scale: jax.Array, lane: jax.Array) -> jax.Array:
        return jnp.sum(solve(scale, lane).value ** 2)

    scale = jnp.asarray(1.3)
    lanes = jnp.arange(3)
    iterations = [np.asarray(solve(scale, lane).diagnostics.iterations) for lane in lanes]
    unbatched = np.asarray(
        [jax.jit(jax.grad(loss))(scale, lane) for lane in lanes], dtype=np.float64
    )
    mapped = jax.jit(jax.vmap(lambda lane: jax.grad(loss)(scale, lane)))(lanes)
    summed = jax.jit(
        jax.grad(lambda value: jnp.sum(jax.vmap(lambda lane: loss(value, lane))(lanes)))
    )(scale)

    assert np.all(iterations[0] == 1)
    assert np.all(iterations[1] > 1)
    assert np.all(iterations[2] == 0)
    assert np.all(np.isfinite(unbatched))
    np.testing.assert_allclose(mapped, unbatched, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(summed, np.sum(unbatched), rtol=1e-10)


# SPD with coordinates 0 and 1 decoupled: e0 and e1 are exact eigenvectors, so a
# decomposition started from them breaks down exactly after one step.
_DECOUPLED = SPD.at[:2, :].set(0.0).at[:, :2].set(0.0) + jnp.diag(
    jnp.zeros(SIZE).at[:2].set(jnp.diag(SPD)[:2])
)


def _decompose(name: str, scale: jax.Array, lane: jax.Array) -> jax.Array:
    """Lane 0 starts in an exactly invariant subspace and breaks down after one
    step; lane 1 runs every step."""
    matrix = scale * _DECOUPLED
    action = lambda value: matrix @ _checked(value)
    start = jnp.stack([jnp.eye(SIZE)[0], RHS])[lane]
    match name:
        case "arnoldi":
            return la.krylov.arnoldi(action, start, max_dimension=4).projected
        case "lanczos":
            return la.krylov.lanczos(action, start, max_dimension=4).projected
        case "golub-kahan":
            decomposition = la.krylov.golub_kahan(
                action,
                lambda value: matrix.T @ _checked(value),
                start,
                max_dimension=4,
            )
            return jnp.concatenate((decomposition.diagonal, decomposition.superdiagonal))
        case "block-arnoldi":
            block = jnp.stack(
                [jnp.eye(SIZE)[:, :2], jnp.stack([RHS, jnp.flip(WEIGHTS)], axis=1)]
            )[lane]
            return la.krylov.block_arnoldi(action, block, max_blocks=3).projected
        case _:
            raise ValueError(name)


@pytest.mark.parametrize("name", ["arnoldi", "lanczos", "golub-kahan", "block-arnoldi"])
def test_gated_off_decomposition_lanes_never_feed_the_action_invalid_inputs(
    name: str,
) -> None:
    def loss(scale: jax.Array, lane: jax.Array) -> jax.Array:
        return jnp.sum(jnp.abs(_decompose(name, scale, lane)) ** 2)

    scale = jnp.asarray(1.3)
    lanes = jnp.arange(2)
    unbatched = np.asarray(
        [jax.jit(jax.grad(loss))(scale, lane) for lane in lanes], dtype=np.float64
    )
    mapped = jax.jit(jax.vmap(lambda lane: jax.grad(loss)(scale, lane)))(lanes)
    summed = jax.jit(
        jax.grad(lambda value: jnp.sum(jax.vmap(lambda lane: loss(value, lane))(lanes)))
    )(scale)

    assert np.all(np.isfinite(unbatched))
    np.testing.assert_allclose(mapped, unbatched, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(summed, np.sum(unbatched), rtol=1e-10)


@pytest.mark.parametrize(
    ("matrix", "start", "shear"),
    [
        (SPD, jnp.linalg.qr(jnp.stack([RHS, WEIGHTS], axis=1))[0], 1.0),
        (_DECOUPLED, jnp.eye(SIZE)[:, :2], 0.0),
    ],
    ids=["degenerate-gram", "exact-breakdown"],
)
def test_block_arnoldi_reverse_mode_differentiates_the_retained_subspace(
    matrix: jax.Array, start: jax.Array, shear: float
) -> None:
    # At scale 1.3 the start block is orthonormal, so its Gram matrix is
    # degenerate; a shear gives it an off-diagonal tangent. The decoupled
    # operator makes the second block's residual exactly zero (rank 0). The
    # loss is invariant to rotations within each block, hence smooth.
    def loss(scale: jax.Array) -> jax.Array:
        sheared = start @ jnp.eye(2).at[0, 1].set(shear * (scale - 1.3))
        decomposition = la.krylov.block_arnoldi(
            lambda value: (scale * matrix) @ value, sheared, max_blocks=3
        )
        return jnp.sum(decomposition.projected**2) + jnp.sum(
            decomposition.initial_factor**2
        )

    gradient = jax.jit(jax.grad(loss))(1.3)

    assert np.isfinite(gradient)
    np.testing.assert_allclose(gradient, _central_difference(loss, 1.3), rtol=1e-6)


_SPD_PROPERTIES = la.OperatorProperties(
    self_adjoint=True,
    positive_definite=True,
    evidence={
        "self_adjoint": "construction",
        "positive_definite": "construction",
        "positive_semidefinite": "construction",
    },
)


def _checked_operator(matrix: jax.Array) -> Any:
    space = la.ArraySpace((SIZE,), dtype=matrix.dtype)
    return la.FunctionLinearOperator(
        lambda value: matrix @ _checked(value),
        source=space,
        target=space,
        transpose_action=lambda value: matrix.T @ _checked(value),
        properties=_SPD_PROPERTIES,
    )


def _lobpcg_eigenvalues(lane: jax.Array) -> jax.Array:
    # Lane 0's initial basis e0, e1 spans its two smallest eigenvectors exactly,
    # so its residuals and search directions are exactly zero; lane 1 iterates.
    invariant = _DECOUPLED.at[0, 0].set(0.1).at[1, 1].set(0.2)
    matrix = jnp.stack([invariant, SPD + jnp.diag(WEIGHTS)])[lane]
    policy = la.eigen.EigenSolvePolicy(
        la.eigen.LOBPCG(block_dimension=2),
        count=2,
        max_steps=40,
        initial_basis=jnp.eye(SIZE)[:, :2],
        tolerance=la.eigen.EigenTolerancePolicy(relative=1e-8, absolute=1e-8),
    )
    problem = la.eigen.Eigenproblem(_checked_operator(matrix))
    return la.eigen.eigensolve(problem, policy=policy).eigenvalues


def test_gated_off_lobpcg_lanes_never_feed_the_operator_invalid_inputs() -> None:
    lanes = jnp.arange(2)
    unbatched = np.stack([np.asarray(_lobpcg_eigenvalues(lane)) for lane in lanes])
    mapped = jax.jit(jax.vmap(_lobpcg_eigenvalues))(lanes)

    assert np.all(np.isfinite(unbatched))
    np.testing.assert_allclose(mapped, unbatched, rtol=1e-9, atol=1e-11)
