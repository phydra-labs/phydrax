#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_pullback_substitutes_state_and_passes_through_time() -> None:
    # ty: ignore[invalid-argument-type]
    q_space = phx.domain.HyperRectangle([-2.0], [2.0], label="q")
    # ty: ignore[invalid-argument-type]
    p_space = phx.domain.HyperRectangle([-2.0], [2.0], label="p")
    time = phx.domain.TimeInterval(0.1, 2.0)
    phase_time = phx.domain.ProductDomain(q_space, p_space, time)

    @phase_time.Function("q", "p", "t")
    def hamiltonian(q: Any, p: Any, t: Any) -> Any:
        return 0.5 * jnp.dot(q, q) + 0.5 * jnp.dot(p, p) + t

    @time.Function("t")
    def q(t: Any) -> Any:
        return jnp.asarray([jnp.cos(t)])

    @time.Function("t")
    def p(t: Any) -> Any:
        return jnp.asarray([-jnp.sin(t)])

    energy = phx.operators.pullback(hamiltonian, {"q": q, "p": p})

    assert energy.domain.labels == ("t",)
    assert energy.deps == ("t",)
    assert jnp.allclose(energy.func(0.7), 1.2, atol=1e-12)


def test_pullback_requires_every_non_passthrough_dependency() -> None:
    # ty: ignore[invalid-argument-type]
    q_space = phx.domain.HyperRectangle([-1.0], [1.0], label="q")
    # ty: ignore[invalid-argument-type]
    p_space = phx.domain.HyperRectangle([-1.0], [1.0], label="p")
    time = phx.domain.TimeInterval(0.0, 1.0)
    phase = phx.domain.ProductDomain(q_space, p_space)

    @phase.Function("q", "p")
    def hamiltonian(q: Any, p: Any) -> Any:
        return q[0] + p[0]

    @time.Function("t")
    def q(t: Any) -> Any:
        return jnp.asarray([t])

    with pytest.raises(ValueError, match="cannot resolve source dependencies"):
        phx.operators.pullback(hamiltonian, {"q": q}, domain=time)


def test_pullback_rejects_unknown_substitution_label() -> None:
    # ty: ignore[invalid-argument-type]
    q_space = phx.domain.HyperRectangle([-1.0], [1.0], label="q")

    @q_space.Function("q")
    def f(q: Any) -> Any:
        return q[0]

    with pytest.raises(ValueError, match="not used by the source function"):
        phx.operators.pullback(f, {"p": f})
