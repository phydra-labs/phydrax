#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax._trainable import partition_trainable
from phydrax.nn.models import (
    BSplineEdgeBasis,
    KAN,
    OrthogonalPolynomialEdgeBasis,
)


@pytest.mark.parametrize("degree", (2, 3, 4))
def test_bspline_kan_identity_and_boundary_jacobian(degree):
    basis = BSplineEdgeBasis(degree=degree, num_intervals=6)
    model = KAN(
        in_size="scalar",
        out_size="scalar",
        hidden_sizes=(),
        edge_basis=basis,
        scale_mode="none",
        init="identity",
        skip_connection=False,
        use_bias=False,
        key=jr.key(0),
    )
    assert model.layers[0].bias is None
    assert model.layers[0].scales is None
    points = jnp.linspace(-1.0, 1.0, 17)

    values = jax.jit(lambda values_: jax.vmap(model)(values_))(points)
    derivatives = jax.vmap(jax.grad(model))(points)

    assert np.allclose(np.asarray(values), np.asarray(points), rtol=1e-11, atol=1e-11)
    assert np.allclose(np.asarray(derivatives), 1.0, rtol=1e-10, atol=1e-10)


def test_orthogonal_kan_clipping_preserves_endpoint_derivatives():
    model = KAN(
        in_size="scalar",
        out_size="scalar",
        hidden_sizes=(),
        edge_basis=OrthogonalPolynomialEdgeBasis(degree=1),
        scale_mode="none",
        init="identity",
        skip_connection=False,
        use_bias=False,
        key=jr.key(1),
    )

    assert float(jax.grad(model)(jnp.asarray(-1.0))) == pytest.approx(1.0)
    assert float(jax.grad(model)(jnp.asarray(1.0))) == pytest.approx(1.0)
    assert float(jax.grad(model)(jnp.asarray(-1.1))) == pytest.approx(0.0)
    assert float(jax.grad(model)(jnp.asarray(1.1))) == pytest.approx(0.0)


def test_bspline_edge_coefficient_gradients_are_span_local():
    basis = BSplineEdgeBasis(degree=3, num_intervals=8)
    coefficients = jnp.zeros((2, 3, basis.coefficient_count))
    inputs = jnp.full((2, 3), 0.13)

    gradient = jax.grad(lambda values: jnp.sum(basis.evaluate(values, inputs)))(
        coefficients
    )
    active_counts = jnp.count_nonzero(jnp.abs(gradient) > 1e-12, axis=-1)

    assert np.array_equal(np.asarray(active_counts), np.full((2, 3), 4))


def test_bspline_grid_is_excluded_from_trainable_partition():
    model = KAN(
        in_size=2,
        out_size=3,
        hidden_sizes=(),
        edge_basis=BSplineEdgeBasis(degree=3, num_intervals=5),
        key=jr.key(2),
    )

    trainable, fixed = partition_trainable(model)

    assert trainable.layers[0].edge_basis.grid is None
    assert fixed.layers[0].edge_basis.grid is not None
    assert np.array_equal(
        np.asarray(fixed.layers[0].edge_basis.grid.knots),
        np.asarray(model.layers[0].edge_basis.grid.knots),
    )


def test_bspline_regularization_is_sobolev_energy():
    basis = BSplineEdgeBasis(
        degree=3,
        num_intervals=6,
        regularization_order=2,
    )
    affine = basis.initialize_coefficients(1, 1, "identity", jr.key(3))
    curved = affine.at[..., basis.coefficient_count // 2].add(1.0)

    assert float(basis.regularization(affine)) == pytest.approx(0.0, abs=1e-20)
    assert float(basis.regularization(curved)) > 0.0


def test_bspline_kan_scan_matches_loop_and_jacobian_is_finite():
    basis = BSplineEdgeBasis(degree=3, num_intervals=6)
    key = jr.key(4)
    loop = KAN(
        in_size=3,
        out_size=2,
        width_size=5,
        depth=4,
        edge_basis=basis,
        scan=False,
        key=key,
    )
    scanned = KAN(
        in_size=3,
        out_size=2,
        width_size=5,
        depth=4,
        edge_basis=basis,
        scan=True,
        key=key,
    )
    inputs = jnp.asarray([0.1, -0.2, 0.3])

    loop_value = eqx.filter_jit(loop)(inputs)
    scanned_value = eqx.filter_jit(scanned)(inputs)
    jacobian = jax.jacrev(scanned)(inputs)

    assert scanned._scan_enabled
    assert np.allclose(np.asarray(scanned_value), np.asarray(loop_value))
    assert jacobian.shape == (2, 3)
    assert np.all(np.isfinite(np.asarray(jacobian)))


def test_kan_rejects_mismatched_basis_schedule():
    with pytest.raises(ValueError, match="edge_basis must have 3 entries"):
        KAN(
            in_size=2,
            out_size=1,
            width_size=4,
            depth=2,
            edge_basis=(BSplineEdgeBasis(),),
            key=jr.key(5),
        )
