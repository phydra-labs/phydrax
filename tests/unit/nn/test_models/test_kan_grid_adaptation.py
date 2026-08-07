#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._trainable import partition_trainable


def _spline_kan(*, scan=False, use_tanh=False, key=jax.random.key(0)):
    return phx.nn.KAN(
        in_size=2,
        out_size="scalar",
        width_size=4,
        depth=3,
        edge_basis=phx.nn.BSplineEdgeBasis(degree=3, num_intervals=6),
        use_tanh=use_tanh,
        skip_connection=False,
        scan=scan,
        key=key,
    )


def test_shared_grid_adaptation_is_pure_and_fixed_count():
    model = _spline_kan()
    calibration = 0.18 * jax.random.normal(jax.random.key(1), (128, 2)) - 0.42
    original_knots = tuple(
        np.asarray(layer.edge_basis.grid.knots).copy() for layer in model.layers
    )

    adapted, report = phx.nn.adapt_kan_grids(
        model,
        calibration,
        plan=phx.nn.KANGridAdaptationPlan(blend=0.05, minimum_span=1e-3),
    )

    assert report.paths == tuple((0, index) for index in range(len(model.layers)))
    assert report.skipped_paths == ()
    assert set(report.degenerate_paths).issubset(report.paths)
    assert all(count > 0 for count in report.activation_counts)
    assert all(np.isfinite(report.transfer_conditioning))
    assert all(error >= 0.0 for error in report.projection_error_bounds)
    for original, old_layer, new_layer in zip(
        original_knots, model.layers, adapted.layers, strict=True
    ):
        assert np.array_equal(np.asarray(old_layer.edge_basis.grid.knots), original)
        assert old_layer.coeffs.shape == new_layer.coeffs.shape
        assert (
            old_layer.edge_basis.grid.coefficient_count
            == new_layer.edge_basis.grid.coefficient_count
        )
    assert not np.array_equal(
        np.asarray(model.layers[0].edge_basis.grid.knots),
        np.asarray(adapted.layers[0].edge_basis.grid.knots),
    )

    old_trainable, _ = partition_trainable(model)
    new_trainable, _ = partition_trainable(adapted)
    assert sum(leaf.size for leaf in jax.tree.leaves(old_trainable)) == sum(
        leaf.size for leaf in jax.tree.leaves(new_trainable)
    )


def test_per_input_adaptation_separates_marginal_distributions():
    model = phx.nn.KAN(
        in_size=2,
        out_size="scalar",
        hidden_sizes=(),
        edge_basis=phx.nn.BSplineEdgeBasis(degree=3, num_intervals=5),
        skip_connection=False,
        key=jax.random.key(20),
    )
    calibration = jnp.stack(
        (
            jnp.linspace(-0.95, -0.15, 96),
            jnp.linspace(0.08, 0.92, 96) ** 2,
        ),
        axis=1,
    )
    adapted, report = phx.nn.adapt_kan_grids(
        model,
        calibration,
        plan=phx.nn.KANGridAdaptationPlan(
            blend=0.0,
            minimum_span=1e-5,
            per_input=True,
        ),
    )
    bank = adapted.layers[0].edge_basis.grid
    evaluation = jax.random.uniform(jax.random.key(21), (32, 2), minval=-1.0, maxval=1.0)

    assert isinstance(bank, phx.nn.BSplineGridBank)
    assert bank.num_grids == 2
    assert report.paths == ((0, 0), (0, 0))
    assert report.input_indices == (0, 1)
    assert not np.array_equal(np.asarray(bank.knots[0]), np.asarray(bank.knots[1]))
    assert np.allclose(
        np.asarray(jax.vmap(adapted)(evaluation)),
        np.asarray(jax.vmap(model)(evaluation)),
        atol=2e-11,
    )


def test_per_input_adaptation_preserves_scanned_hidden_layers():
    model = _spline_kan(scan=True)
    calibration = jax.random.normal(jax.random.key(22), (64, 2)) * jnp.asarray(
        [0.12, 0.47]
    )
    adapted, report = phx.nn.adapt_kan_grids(
        model,
        calibration,
        plan=phx.nn.KANGridAdaptationPlan(per_input=True),
    )
    evaluation = jax.random.normal(jax.random.key(23), (12, 2))

    def explicit_loop(value):
        output = value
        for layer in adapted.layers:
            output = layer(output)
        return output

    assert adapted._scan_enabled
    assert set(report.input_indices) == {0, 1, 2, 3}
    assert np.allclose(
        np.asarray(jax.jit(jax.vmap(adapted))(evaluation)),
        np.asarray(jax.vmap(explicit_loop)(evaluation)),
        atol=2e-12,
    )


def test_adaptation_uses_the_layer_canonical_edge_inputs():
    model = phx.nn.KAN(
        in_size=2,
        out_size="scalar",
        hidden_sizes=(),
        edge_basis=phx.nn.BSplineEdgeBasis(degree=2, num_intervals=4),
        scale_mode="edge",
        use_tanh=True,
        skip_connection=False,
        key=jax.random.key(2),
    )
    calibration = jnp.asarray(
        [[-3.0, -0.7], [-1.1, -0.1], [0.2, 0.4], [0.8, 1.7], [2.4, 3.0]]
    )
    layer = model.layers[0]
    normalized = jax.vmap(layer._normalized_edge_inputs)(calibration)
    expected = np.quantile(
        np.asarray(normalized).reshape((-1,)),
        [0.25, 0.5, 0.75],
        method="linear",
    )

    adapted, _ = phx.nn.adapt_kan_grids(
        model,
        calibration,
        plan=phx.nn.KANGridAdaptationPlan(blend=0.0, minimum_span=1e-5),
    )
    actual = np.asarray(adapted.layers[0].edge_basis.grid.knots)[3:-3]

    assert np.allclose(actual, expected, atol=2e-12)


def test_adaptation_preserves_scan_execution_with_distinct_grid_values():
    model = _spline_kan(scan=True)
    calibration = jax.random.uniform(jax.random.key(3), (96, 2), minval=-0.8, maxval=0.35)
    adapted, _ = phx.nn.adapt_kan_grids(model, calibration)
    evaluation = jax.random.normal(jax.random.key(4), (16, 2))

    def explicit_loop(value):
        output = value
        for layer in adapted.layers:
            output = layer(output)
        return output

    assert model._scan_enabled
    assert adapted._scan_enabled
    assert not np.array_equal(
        np.asarray(adapted.layers[1].edge_basis.grid.knots),
        np.asarray(adapted.layers[2].edge_basis.grid.knots),
    )
    actual = jax.jit(jax.vmap(adapted))(evaluation)
    expected = jax.vmap(explicit_loop)(evaluation)
    assert np.allclose(np.asarray(actual), np.asarray(expected), atol=2e-12)


def test_mixed_basis_layers_are_preserved_and_reported():
    bases = (
        phx.nn.BSplineEdgeBasis(degree=3, num_intervals=4),
        phx.nn.OrthogonalPolynomialEdgeBasis(degree=3),
        phx.nn.BSplineEdgeBasis(degree=3, num_intervals=4),
    )
    model = phx.nn.KAN(
        in_size=2,
        out_size="scalar",
        hidden_sizes=(3, 3),
        edge_basis=bases,
        skip_connection=False,
        key=jax.random.key(5),
    )
    adapted, report = phx.nn.adapt_kan_grids(
        model,
        jax.random.normal(jax.random.key(6), (32, 2)),
    )

    assert report.paths == ((0, 0), (0, 2))
    assert report.skipped_paths == ((0, 1),)
    assert eqx.tree_equal(adapted.layers[1], model.layers[1])


def test_degenerate_calibration_policy_and_failures_are_explicit():
    model = _spline_kan()
    calibration = jnp.zeros((12, 2))

    retained, retained_report = phx.nn.adapt_kan_grids(model, calibration)
    uniform, uniform_report = phx.nn.adapt_kan_grids(
        model,
        calibration,
        plan=phx.nn.KANGridAdaptationPlan(degenerate_policy="uniform"),
    )

    assert retained_report.degenerate_paths
    assert uniform_report.degenerate_paths
    assert np.array_equal(
        np.asarray(retained.layers[0].edge_basis.grid.knots),
        np.asarray(model.layers[0].edge_basis.grid.knots),
    )
    assert uniform.layers[0].edge_basis.grid.is_uniform
    with pytest.raises(ValueError, match="finite"):
        phx.nn.adapt_kan_grids(model, jnp.asarray([[jnp.nan, 0.0]]))
    with pytest.raises(ValueError, match="too large"):
        phx.nn.adapt_kan_grids(
            model,
            calibration,
            plan=phx.nn.KANGridAdaptationPlan(
                minimum_span=1.0,
                degenerate_policy="uniform",
            ),
        )


def test_separable_kan_adapts_every_coordinate_model():
    model = phx.nn.SeparableKAN(
        in_size=2,
        out_size="scalar",
        latent_size=2,
        width_size=3,
        depth=1,
        edge_basis=phx.nn.BSplineEdgeBasis(degree=2, num_intervals=4),
        skip_connection=False,
        scan=True,
        key=jax.random.key(7),
    )
    calibration = jax.random.uniform(jax.random.key(8), (40, 2), minval=-0.9, maxval=0.7)
    adapted, report = phx.nn.adapt_kan_grids(model, calibration)
    query = jnp.asarray([0.17, -0.22])

    assert len(report.paths) == 4
    assert {path[0] for path in report.paths} == {0, 1}
    assert np.isfinite(np.asarray(adapted(query)))
    assert adapted.model.scan == model.model.scan
    assert all(
        new_model.layers[0].coeffs.shape == old_model.layers[0].coeffs.shape
        for new_model, old_model in zip(
            adapted.model.models, model.model.models, strict=True
        )
    )


def test_affine_initialized_model_is_preserved_by_regridding():
    model = _spline_kan()
    calibration = jax.random.normal(jax.random.key(9), (64, 2)) * 0.3
    adapted, _ = phx.nn.adapt_kan_grids(model, calibration)
    evaluation = jax.random.uniform(jax.random.key(10), (32, 2), minval=-1.0, maxval=1.0)

    expected = jax.vmap(model)(evaluation)
    actual = jax.vmap(adapted)(evaluation)
    assert np.allclose(np.asarray(actual), np.asarray(expected), atol=2e-11)
