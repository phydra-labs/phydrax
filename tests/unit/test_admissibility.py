#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._admissibility import (
    DerivativeFailureMode,
    first_order_only,
    guard_derivative_validity,
    refuse_derivative_dependencies,
)


def test_admissibility_scenario_1() -> None:
    header = phx.AdmissibilityHeader(
        jnp.asarray((0.25, -0.1, jnp.nan)),
        jnp.asarray(
            (
                0,
                int(phx.AdmissibilityReason.OUTSIDE_SUPPORT),
                0,
            ),
            dtype=jnp.uint32,
        ),
        "model",
        "evidence",
    )

    np.testing.assert_array_equal(header.eligible, (True, False, False))
    assert np.isneginf(np.asarray(header.margin)[2])
    assert int(header.reason_bits[2]) & int(phx.AdmissibilityReason.NONFINITE)
    assert not bool(header.globally_eligible)
    first = phx.AdmissibilityHeader(
        jnp.asarray((0.5, 0.2)),
        jnp.zeros((2,), dtype=jnp.uint32),
        "first",
        "first-evidence",
    )
    second = phx.AdmissibilityHeader(
        jnp.asarray((0.1, -0.4)),
        jnp.asarray(
            (0, int(phx.AdmissibilityReason.CAPACITY_INSUFFICIENT)),
            dtype=jnp.uint32,
        ),
        "second",
        "second-evidence",
    )

    combined = phx.combine_admissibility((first, second), "coupled")

    np.testing.assert_allclose(combined.margin, (0.1, -0.4))
    np.testing.assert_array_equal(combined.eligible, (True, False))
    assert int(combined.reason_bits[1]) & int(
        phx.AdmissibilityReason.CAPACITY_INSUFFICIENT
    )
    value = jnp.asarray((1.0, -2.0))
    primal, valid_tangent = jax.jvp(
        lambda argument: guard_derivative_validity(argument, jnp.asarray(True)),
        (value,),
        (jnp.ones_like(value),),
    )
    _, invalid_tangent = jax.jvp(
        lambda argument: guard_derivative_validity(argument, jnp.asarray(False)),
        (value,),
        (jnp.ones_like(value),),
    )
    invalid_reverse = jax.grad(
        lambda argument: jnp.sum(guard_derivative_validity(argument, jnp.asarray(False)))
    )(value)

    np.testing.assert_allclose(primal, value)
    np.testing.assert_allclose(valid_tangent, jnp.ones_like(value))
    assert jnp.all(jnp.isnan(invalid_tangent))
    assert jnp.all(jnp.isnan(invalid_reverse))
    value = jnp.asarray((1.0, -2.0))
    with pytest.raises(eqx.EquinoxRuntimeError, match="invalid test derivative"):
        _, tangent = jax.jvp(
            lambda argument: guard_derivative_validity(
                argument,
                jnp.asarray(False),
                failure="error",
                message="invalid test derivative",
            ),
            (value,),
            (jnp.ones_like(value),),
        )
        jax.block_until_ready(tangent)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "filter_jit"])
def test_error_mode_refuses_reverse_mode_derivatives(compiled: bool) -> None:
    value = jnp.asarray((1.0, -2.0))

    def guarded(argument: Any, valid: Any) -> Any:
        return jnp.sum(
            guard_derivative_validity(
                argument**2, valid, failure="error", message="invalid test derivative"
            )
        )

    # Equinox raises EquinoxRuntimeError from eager and filter_jit programs; a
    # check evaluated inside a staged program without filter_jit (the reverse
    # pass evaluates the transposed derivative that way) surfaces as the
    # JaxRuntimeError carrying the same message.
    refused = (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError)
    gradient = eqx.filter_jit(jax.grad(guarded)) if compiled else jax.grad(guarded)
    with pytest.raises(refused, match="invalid test derivative"):
        jax.block_until_ready(gradient(value, jnp.asarray(False)))
    with pytest.raises(refused, match="invalid test derivative"):
        _, pullback = jax.vjp(lambda argument: guarded(argument, False), value)
        jax.block_until_ready(pullback(jnp.asarray(1.0)))
    # Valid derivatives and every primal value are unchanged.
    np.testing.assert_allclose(gradient(value, jnp.asarray(True)), 2.0 * value)
    np.testing.assert_allclose(guarded(value, jnp.asarray(False)), 5.0)


def test_admissibility_is_jittable_and_transition_is_diagnostic_only() -> None:
    @jax.jit
    def evaluate(margin: Any) -> Any:
        return phx.AdmissibilityHeader(
            margin,
            phx.reason_bits_where(
                margin >= 0.0,
                phx.AdmissibilityReason.OUTSIDE_SUPPORT,
            ),
            "fixed-model",
            "fixed-evidence",
        )

    header = evaluate(jnp.asarray((1.0, -1.0)))
    request = phx.AdmissibilityTransitionRequest(
        jnp.asarray((False, True)),
        jnp.asarray(3),
        header,
        "continuum",
        "kinetic",
    )

    np.testing.assert_array_equal(request.region_mask, (False, True))
    assert int(request.requested_epoch) == 3
    assert request.current_model_id == "continuum"
    assert request.target_model_id == "kinetic"


def _unguarded(x: jax.Array, k: jax.Array) -> jax.Array:
    # Linear in x: inner tangents in x are constants, which an enclosing pass
    # differentiates to symbolic zeros unless the guard is re-applied there.
    return 2.0 * x + k**3


def _guarded(
    valid: bool, failure: DerivativeFailureMode = "status"
) -> Callable[..., jax.Array]:
    def owner(x: jax.Array, k: jax.Array) -> jax.Array:
        return guard_derivative_validity(
            _unguarded(x, k),
            jnp.asarray(valid),
            dependencies=(x, k),
            failure=failure,
            message="invalid nested derivative",
        )

    return owner


_NESTED_ROUTES: dict[str, Callable[[Callable[..., jax.Array]], Callable[..., Any]]] = {
    "forward-over-forward": lambda f: jax.jacfwd(jax.jacfwd(f, 0), 0),
    "reverse-over-forward": lambda f: jax.jacrev(jax.jacfwd(f, 0), 0),
    "forward-over-reverse": lambda f: jax.jacfwd(jax.grad(f, 0), 0),
    "reverse-over-reverse": lambda f: jax.grad(jax.grad(f, 0), 0),
    "mixed-forward-over-reverse": lambda f: jax.jacfwd(jax.grad(f, 0), 1),
    "mixed-reverse-over-forward": lambda f: jax.jacrev(jax.jacfwd(f, 0), 1),
    "hessian": lambda f: jax.hessian(f, (0, 1)),
    "third-forward": lambda f: jax.jacfwd(jax.jacfwd(jax.jacfwd(f, 0), 0), 1),
    "third-reverse": lambda f: jax.grad(jax.grad(jax.grad(f, 1), 1), 0),
    "third-mixed": lambda f: jax.jacfwd(jax.jacrev(jax.jacfwd(f, 1), 1), 1),
}


@pytest.mark.parametrize("route", _NESTED_ROUTES)
def test_guard_applies_at_every_derivative_order(route: str) -> None:
    nested = _NESTED_ROUTES[route]
    expected = jax.tree.leaves(nested(_unguarded)(_X, _K))
    # Valid input: both modes (the error mode through its identity linear
    # solve) reproduce the unguarded derivative exactly at every order.
    for failure in ("status", "error"):
        actual = jax.tree.leaves(nested(_guarded(True, failure))(_X, _K))
        for leaf, value in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(value, leaf)
    for leaf in jax.tree.leaves(jax.jit(nested(_guarded(False)))(_X, _K)):
        assert bool(jnp.all(jnp.isnan(leaf)))
    # The check fails at run time: EquinoxRuntimeError eagerly, and the runtime's
    # callback error carrying the same message from a staged (transposed) check.
    with pytest.raises(
        (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError, ValueError),
        match="invalid nested derivative",
    ):
        jax.block_until_ready(nested(_guarded(False, "error"))(_X, _K))


@pytest.mark.parametrize(
    "dependencies",
    [(), (jnp.asarray(0.5, dtype=jnp.float32),)],
    ids=["no-dependency", "float32-dependency"],
)
@pytest.mark.parametrize("route", _NESTED_ROUTES)
def test_guard_carrier_obeys_strict_dtype_promotion(
    route: str, dependencies: tuple[jax.Array, ...]
) -> None:
    # A float64 value guarded by an empty (float32 placeholder) or a float32
    # dependency must not rely on implicit promotion at any derivative order.
    nested = _NESTED_ROUTES[route]

    def guarded(valid: bool) -> Callable[..., jax.Array]:
        def owner(x: jax.Array, k: jax.Array) -> jax.Array:
            return guard_derivative_validity(
                _unguarded(x, k), jnp.asarray(valid), dependencies=dependencies
            )

        return owner

    expected = jax.tree.leaves(nested(_unguarded)(_X, _K))
    with phx.config.context(phx.config.NumericsConfig(dtype_promotion="strict")):
        actual = jax.tree.leaves(jax.jit(nested(guarded(True)))(_X, _K))
        poisoned = jax.tree.leaves(jax.jit(nested(guarded(False)))(_X, _K))
    for leaf, value in zip(expected, actual, strict=True):
        assert value.dtype == jnp.float64
        np.testing.assert_array_equal(value, leaf)
    for leaf in poisoned:
        assert bool(jnp.all(jnp.isnan(leaf)))


def _refused_owner(x: jax.Array, k: jax.Array) -> jax.Array:
    # The owner stops its route through `k` (as an rhs-only or unsupported solve
    # would) and refuses it, so any derivative in `k` must raise, not return 0.
    value = x**3 * jax.lax.stop_gradient(k) ** 2
    return jnp.sum(refuse_derivative_dependencies(value, (k,), message="k refused"))


_X = jnp.asarray(2.0, dtype=jnp.float64)
_K = jnp.asarray(3.0, dtype=jnp.float64)


@pytest.mark.parametrize(
    "route",
    (
        pytest.param(lambda: jax.grad(_refused_owner, 1)(_X, _K), id="grad-k"),
        pytest.param(
            lambda: jax.grad(lambda k: jax.grad(_refused_owner, 0)(_X, k))(_K),
            id="grad-k-of-grad-x",
        ),
        pytest.param(
            lambda: jax.jvp(
                lambda k: jax.grad(_refused_owner, 0)(_X, k), (_K,), (jnp.ones_like(_K),)
            ),
            id="jvp-k-of-grad-x",
        ),
        pytest.param(
            lambda: jax.jacfwd(jax.grad(_refused_owner, 0), 1)(_X, _K),
            id="jacfwd-k-of-grad-x",
        ),
        pytest.param(
            lambda: jax.grad(lambda k: jax.hessian(_refused_owner, 0)(_X, k))(_K),
            id="grad-k-of-hessian-x",
        ),
    ),
)
def test_refused_dependency_is_refused_at_every_derivative_order(
    route: Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match="derivative-unsupported: k refused"):
        route()


def test_refusal_preserves_higher_derivatives_in_admitted_inputs() -> None:
    # f = x^3 k^2 with k fixed: analytic d2f/dx2 = 6 x k^2, d3f/dx3 = 6 k^2.
    hessian = jax.jit(jax.hessian(_refused_owner, 0))(_X, _K)
    third = jax.grad(jax.grad(jax.grad(_refused_owner, 0), 0), 0)(_X, _K)
    batched = jax.vmap(jax.hessian(_refused_owner, 0), (0, None))(
        jnp.asarray((1.0, 2.0), dtype=jnp.float64), _K
    )

    np.testing.assert_allclose(hessian, 6.0 * 2.0 * 9.0, rtol=1e-15)
    np.testing.assert_allclose(third, 6.0 * 9.0, rtol=1e-15)
    np.testing.assert_allclose(batched, (54.0, 108.0), rtol=1e-15)


def _first_order_owner(x: jax.Array, y: jax.Array) -> jax.Array:
    x, y = first_order_only((x, y), message="maximum_derivative_order=1")
    return x**3 * y**2


def test_first_order_carrier_preserves_primal_jvp_vjp_and_gradient() -> None:
    value, tangent = jax.jvp(
        _first_order_owner, (_X, _K), (jnp.ones_like(_X), jnp.ones_like(_K))
    )
    _, pullback = jax.vjp(_first_order_owner, _X, _K)
    expected = (108.0, 48.0)
    np.testing.assert_allclose(value, 72.0)
    np.testing.assert_allclose(tangent, sum(expected))
    np.testing.assert_allclose(pullback(jnp.asarray(1.0)), expected)
    np.testing.assert_allclose(
        jax.jit(jax.grad(_first_order_owner, (0, 1)))(_X, _K), expected
    )


@pytest.mark.parametrize(
    "route",
    [
        pytest.param(
            lambda: jax.grad(jax.grad(_first_order_owner, 0), 0)(_X, _K),
            id="grad-grad",
        ),
        pytest.param(
            lambda: jax.jacfwd(jax.grad(_first_order_owner, 0), 1)(_X, _K),
            id="forward-over-reverse-mixed",
        ),
        pytest.param(
            lambda: jax.jacrev(jax.jacfwd(_first_order_owner, 0), 1)(_X, _K),
            id="reverse-over-forward-mixed",
        ),
        pytest.param(lambda: jax.hessian(_first_order_owner, 0)(_X, _K), id="hessian"),
        pytest.param(
            lambda: jax.jacfwd(jax.jacfwd(_first_order_owner, 0), 0)(_X, _K),
            id="forward-over-forward",
        ),
    ],
)
def test_first_order_carrier_refuses_nested_transforms(
    route: Callable[[], object],
) -> None:
    with pytest.raises(
        ValueError, match="derivative-unsupported: maximum_derivative_order=1"
    ):
        route()
