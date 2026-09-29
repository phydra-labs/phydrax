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
