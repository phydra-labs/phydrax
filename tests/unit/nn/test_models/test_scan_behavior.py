#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from phydrax.nn.models import FeynmaNN, FNO1d, FNO2d, KAN, MLP


def _num_params(model) -> int:
    dynamic, _ = eqx.partition(model, eqx.is_array)
    leaves = jtu.tree_leaves(dynamic)
    return sum(int(x.size) for x in leaves)


def test_mlp_scan_parameter_count_matches_loop():
    key = jr.key(100)
    m_loop = MLP(in_size=3, out_size=2, width_size=8, depth=4, scan=False, key=key)
    m_scan = MLP(in_size=3, out_size=2, width_size=8, depth=4, scan=True, key=key)
    assert _num_params(m_scan) == _num_params(m_loop)


def test_mlp_scan_parity_deep():
    key = jr.key(0)
    m_loop = MLP(in_size=3, out_size=2, width_size=8, depth=4, scan=False, key=key)
    m_scan = MLP(in_size=3, out_size=2, width_size=8, depth=4, scan=True, key=key)
    x = jr.normal(jr.key(1), (3,))
    assert jnp.allclose(m_loop(x), m_scan(x))


def test_mlp_scan_depth_edge_cases():
    key0 = jr.key(2)
    m0_loop = MLP(in_size=2, out_size=2, width_size=8, depth=0, scan=False, key=key0)
    m0_scan = MLP(in_size=2, out_size=2, width_size=8, depth=0, scan=True, key=key0)
    x0 = jr.normal(jr.key(3), (2,))
    assert jnp.allclose(m0_loop(x0), m0_scan(x0))

    key1 = jr.key(4)
    m1_loop = MLP(in_size=2, out_size=2, width_size=8, depth=1, scan=False, key=key1)
    m1_scan = MLP(in_size=2, out_size=2, width_size=8, depth=1, scan=True, key=key1)
    x1 = jr.normal(jr.key(5), (2,))
    assert jnp.allclose(m1_loop(x1), m1_scan(x1))


def test_mlp_scan_gradient_smoke():
    model = MLP(in_size=2, out_size=2, width_size=8, depth=3, scan=True, key=jr.key(6))
    x = jr.normal(jr.key(7), (2,))

    @eqx.filter_grad
    def loss_fn(m, x_):
        y = m(x_)
        return jnp.sum(y**2)

    grads = loss_fn(model, x)
    assert grads is not None


def test_mlp_scan_fallback_heterogeneous_hidden_sizes():
    model = MLP(in_size=3, out_size=2, hidden_sizes=(5, 7, 5), scan=True, key=jr.key(8))
    x = jr.normal(jr.key(9), (3,))
    y = model(x)
    assert y.shape == (2,)
    assert model.scan
    assert not model._scan_enabled


def test_feynmann_scan_parity():
    key = jr.key(10)
    m_loop = FeynmaNN(
        in_size=3, out_size=2, width_size=12, depth=3, num_paths=2, scan=False, key=key
    )
    m_scan = FeynmaNN(
        in_size=3, out_size=2, width_size=12, depth=3, num_paths=2, scan=True, key=key
    )
    x = jr.normal(jr.key(11), (3,))
    assert jnp.allclose(m_loop(x), m_scan(x))


def test_feynmann_scan_parameter_count_matches_loop():
    key = jr.key(23)
    m_loop = FeynmaNN(
        in_size=3, out_size=2, width_size=12, depth=3, num_paths=2, scan=False, key=key
    )
    m_scan = FeynmaNN(
        in_size=3, out_size=2, width_size=12, depth=3, num_paths=2, scan=True, key=key
    )
    assert _num_params(m_scan) == _num_params(m_loop)


def test_feynmann_scan_gradient_smoke():
    model = FeynmaNN(
        in_size=2,
        out_size=2,
        width_size=10,
        depth=3,
        num_paths=2,
        scan=True,
        key=jr.key(12),
    )
    x = jr.normal(jr.key(13), (2,))

    @eqx.filter_grad
    def loss_fn(m, x_):
        y = m(x_)
        return jnp.sum(y**2)

    grads = loss_fn(model, x)
    assert grads is not None


def test_fno1d_scan_parity():
    key = jr.key(14)
    f_loop = FNO1d(
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=3,
        modes=6,
        scan=False,
        key=key,
    )
    f_scan = FNO1d(
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=3,
        modes=6,
        scan=True,
        key=key,
    )
    n = 16
    data = jr.normal(jr.key(15), (n,))
    x_axis = jnp.linspace(0.0, 1.0, n)
    assert jnp.allclose(f_loop((data, x_axis)), f_scan((data, x_axis)))


def test_fno1d_scan_parameter_count_matches_loop():
    key = jr.key(24)
    f_loop = FNO1d(
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=3,
        modes=6,
        scan=False,
        key=key,
    )
    f_scan = FNO1d(
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=3,
        modes=6,
        scan=True,
        key=key,
    )
    assert _num_params(f_scan) == _num_params(f_loop)


def test_fno2d_scan_parity():
    key = jr.key(16)
    f_loop = FNO2d(
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=3,
        modes=6,
        scan=False,
        key=key,
    )
    f_scan = FNO2d(
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=3,
        modes=6,
        scan=True,
        key=key,
    )
    nx, ny = 10, 8
    data = jr.normal(jr.key(17), (nx, ny))
    x_axis = jnp.linspace(0.0, 1.0, nx)
    y_axis = jnp.linspace(-1.0, 1.0, ny)
    assert jnp.allclose(f_loop((data, x_axis, y_axis)), f_scan((data, x_axis, y_axis)))


def test_fno2d_scan_parameter_count_matches_loop():
    key = jr.key(25)
    f_loop = FNO2d(
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=3,
        modes=6,
        scan=False,
        key=key,
    )
    f_scan = FNO2d(
        in_channels="scalar",
        out_channels="scalar",
        width=8,
        depth=3,
        modes=6,
        scan=True,
        key=key,
    )
    assert _num_params(f_scan) == _num_params(f_loop)


def test_kan_scan_parity_uniform():
    key = jr.key(18)
    k_loop = KAN(
        in_size=3, out_size=2, width_size=6, depth=4, degree=3, scan=False, key=key
    )
    k_scan = KAN(
        in_size=3, out_size=2, width_size=6, depth=4, degree=3, scan=True, key=key
    )
    x = jr.normal(jr.key(19), (3,))
    assert jnp.allclose(k_loop(x), k_scan(x))


def test_kan_scan_parameter_count_matches_loop():
    key = jr.key(22)
    k_loop = KAN(
        in_size=3, out_size=2, width_size=6, depth=4, degree=3, scan=False, key=key
    )
    k_scan = KAN(
        in_size=3, out_size=2, width_size=6, depth=4, degree=3, scan=True, key=key
    )
    assert _num_params(k_scan) == _num_params(k_loop)


def test_kan_scan_fallback_heterogeneous():
    model = KAN(
        in_size=3,
        out_size=2,
        hidden_sizes=(5, 7, 5),
        degree=(2, 3, 4, 5),
        scan=True,
        key=jr.key(20),
    )
    x = jr.normal(jr.key(21), (3,))
    y = model(x)
    assert y.shape == (2,)
    assert model.scan
    assert not model._scan_enabled
