from __future__ import annotations

from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax.nn.models import (
    FeynmaNN,
    KAN,
    MLP,
    OrthogonalPolynomialEdgeBasis,
)
from phydrax.nn.operator.architectures import FNO


class _ArrayModel(Protocol):
    def __call__(self, value: jax.Array, /) -> jax.Array: ...


def _num_params(model: object) -> int:
    dynamic, _ = eqx.partition(model, eqx.is_array)
    return sum(leaf.size for leaf in jax.tree.leaves(dynamic))


def _assert_finite_nonzero_gradients(model: _ArrayModel, value: jax.Array) -> None:
    @eqx.filter_grad
    def loss(subject: _ArrayModel, inputs: jax.Array) -> jax.Array:
        return jnp.sum(subject(inputs) ** 2)

    leaves = [leaf for leaf in jax.tree.leaves(loss(model, value)) if eqx.is_array(leaf)]
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert any(bool(jnp.any(leaf != 0.0)) for leaf in leaves)


def test_scan_behavior_scenario_1() -> None:
    for depth, input_size, key_value in ((0, 2, 2), (1, 2, 4), (4, 3, 0)):
        key = jr.key(key_value)
        loop = MLP(
            in_size=input_size,
            out_size=2,
            width_size=8,
            depth=depth,
            scan=False,
            key=key,
        )
        scanned = MLP(
            in_size=input_size,
            out_size=2,
            width_size=8,
            depth=depth,
            scan=True,
            key=key,
        )
        value = jr.normal(jr.key(key_value + 1), (input_size,))
        assert _num_params(scanned) == _num_params(loop), depth
        assert jnp.allclose(loop(value), scanned(value)), depth

    differentiable = MLP(
        in_size=2,
        out_size=2,
        width_size=8,
        depth=3,
        scan=True,
        key=jr.key(6),
    )
    _assert_finite_nonzero_gradients(differentiable, jr.normal(jr.key(7), (2,)))

    heterogeneous = MLP(
        in_size=3,
        out_size=2,
        hidden_sizes=(5, 7, 5),
        scan=True,
        key=jr.key(8),
    )
    assert heterogeneous(jr.normal(jr.key(9), (3,))).shape == (2,)
    assert heterogeneous.scan
    key = jr.key(10)
    loop = FeynmaNN(
        in_size=3,
        out_size=2,
        width_size=12,
        depth=3,
        num_paths=2,
        scan=False,
        key=key,
    )
    scanned = FeynmaNN(
        in_size=3,
        out_size=2,
        width_size=12,
        depth=3,
        num_paths=2,
        scan=True,
        key=key,
    )
    value = jr.normal(jr.key(11), (3,))
    assert _num_params(scanned) == _num_params(loop)
    assert jnp.allclose(loop(value), scanned(value))

    differentiable = FeynmaNN(
        in_size=2,
        out_size=2,
        width_size=10,
        depth=3,
        num_paths=2,
        scan=True,
        key=jr.key(12),
    )
    _assert_finite_nonzero_gradients(differentiable, jr.normal(jr.key(13), (2,)))
    cases = (
        (
            (6,),
            (jr.normal(jr.key(15), (16,)), jnp.linspace(0.0, 1.0, 16)),
        ),
        (
            (6, 6),
            (
                jr.normal(jr.key(17), (10, 8)),
                jnp.linspace(0.0, 1.0, 10),
                jnp.linspace(-1.0, 1.0, 8),
            ),
        ),
    )
    for modes, inputs in cases:
        key = jr.key(14 + len(modes))
        loop = FNO(
            in_channels="scalar",
            out_channels="scalar",
            width=8,
            depth=3,
            n_modes=modes,
            scan=False,
            key=key,
        )
        scanned = FNO(
            in_channels="scalar",
            out_channels="scalar",
            width=8,
            depth=3,
            n_modes=modes,
            scan=True,
            key=key,
        )
        assert _num_params(scanned) == _num_params(loop), modes
        assert jnp.allclose(loop(inputs), scanned(inputs)), modes


def test_kan_scan_matches_loop_parameters_values_and_heterogeneous_fallback() -> None:
    key = jr.key(18)
    basis = OrthogonalPolynomialEdgeBasis(degree=3)
    loop = KAN(
        in_size=3,
        out_size=2,
        width_size=6,
        depth=4,
        edge_basis=basis,
        scan=False,
        key=key,
    )
    scanned = KAN(
        in_size=3,
        out_size=2,
        width_size=6,
        depth=4,
        edge_basis=basis,
        scan=True,
        key=key,
    )
    value = jr.normal(jr.key(19), (3,))
    assert _num_params(scanned) == _num_params(loop)
    assert jnp.allclose(loop(value), scanned(value))

    heterogeneous = KAN(
        in_size=3,
        out_size=2,
        hidden_sizes=(5, 7, 5),
        edge_basis=tuple(
            OrthogonalPolynomialEdgeBasis(degree=degree) for degree in (2, 3, 4, 5)
        ),
        scan=True,
        key=jr.key(20),
    )
    assert heterogeneous(jr.normal(jr.key(21), (3,))).shape == (2,)
    assert heterogeneous.scan
