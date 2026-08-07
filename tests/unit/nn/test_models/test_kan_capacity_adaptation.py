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


def _model(*, scan=False, per_input=False):
    return phx.nn.KAN(
        in_size=2,
        out_size="scalar",
        width_size=4,
        depth=4,
        edge_basis=phx.nn.BSplineEdgeBasis(
            degree=3,
            num_intervals=4,
            per_input=per_input,
        ),
        skip_connection=False,
        scan=scan,
        key=jax.random.key(0),
    )


def _trainable_count(model):
    trainable, _ = partition_trainable(model)
    return sum(int(leaf.size) for leaf in jax.tree.leaves(trainable))


def test_refinement_is_exact_pure_and_allocates_only_selected_edges():
    model = _model(per_input=True)
    evaluation = jax.random.uniform(jax.random.key(1), (48, 2), minval=-1.0, maxval=1.0)
    original_knots = np.asarray(model.layers[0].edge_basis.grid.grids[0].knots).copy()
    adapted, report = phx.nn.refine_kan_edges(
        model,
        {
            (0, 0, 0): jnp.asarray([0.0, 3.0, 1.0, 0.0]),
            (0, 0, 1): jnp.asarray([0.0, 0.0, 2.0, 0.0]),
        },
        budget=1,
    )

    assert report.operation == "refine"
    assert report.paths == ((0, 0, 0),)
    assert report.old_coefficient_counts == (7,)
    assert report.new_coefficient_counts == (8,)
    assert report.projection_error_bounds == (0.0,)
    assert np.array_equal(
        np.asarray(model.layers[0].edge_basis.grid.grids[0].knots),
        original_knots,
    )
    assert adapted.layers[0].edge_basis is None
    assert adapted.layers[0].coeffs is None
    assert sum(block.edge_count for block in adapted.layers[0].edge_blocks) == int(
        np.prod(model.layers[0].coeffs.shape[:2])
    )
    assert _trainable_count(adapted) == _trainable_count(model) + 1
    assert np.allclose(
        np.asarray(jax.vmap(adapted)(evaluation)),
        np.asarray(jax.vmap(model)(evaluation)),
        atol=2e-12,
    )


def test_exact_coarsening_undoes_inserted_capacity_with_certificate():
    model = _model()
    refined, _ = phx.nn.refine_kan_edges(
        model,
        {(0, 0, 0): jnp.asarray([0.0, 1.0, 0.0, 0.0])},
        budget=1,
    )
    coarsened, report = phx.nn.coarsen_kan_edges(
        refined,
        {(0, 0, 0): 1.0e-10},
        budget=1,
    )
    evaluation = jax.random.uniform(jax.random.key(2), (32, 2), minval=-1.0, maxval=1.0)

    assert report.operation == "coarsen"
    assert report.paths == ((0, 0, 0),)
    assert report.old_coefficient_counts == (8,)
    assert report.new_coefficient_counts == (7,)
    assert report.projection_error_bounds[0] < 1e-12
    assert _trainable_count(coarsened) == _trainable_count(model)
    assert np.allclose(
        np.asarray(jax.vmap(coarsened)(evaluation)),
        np.asarray(jax.vmap(model)(evaluation)),
        atol=2e-12,
    )


def test_identical_hidden_block_layouts_preserve_scan_and_gradients():
    model = _model(scan=True)
    repeated_layers = range(1, len(model.layers) - 1)
    indicators = {
        (layer_index, 0, 0): jnp.asarray([0.0, 1.0, 0.0, 0.0])
        for layer_index in repeated_layers
    }
    adapted, report = phx.nn.refine_kan_edges(
        model,
        indicators,
        budget=len(indicators),
    )
    inputs = jnp.asarray([0.17, -0.29])
    gradient = eqx.filter_grad(lambda candidate: jnp.sum(candidate(inputs) ** 2))(adapted)

    assert model._scan_enabled
    assert adapted._scan_enabled
    assert len(report.paths) == len(indicators)
    assert np.allclose(
        np.asarray(eqx.filter_jit(adapted)(inputs)),
        np.asarray(eqx.filter_jit(model)(inputs)),
        atol=2e-12,
    )
    for layer_index in repeated_layers:
        for block in gradient.layers[layer_index].edge_blocks:
            assert np.all(np.isfinite(np.asarray(block.coeffs)))


def test_capacity_adaptation_validation_and_tolerance_are_explicit():
    model = _model()
    with pytest.raises(ValueError, match="one value per positive span"):
        phx.nn.refine_kan_edges(
            model,
            {(0, 0, 0): jnp.ones(3)},
            budget=1,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        phx.nn.refine_kan_edges(
            model,
            {(0, 0, 0): jnp.asarray([0.0, -1.0, 0.0, 0.0])},
            budget=1,
        )
    noncoarse = eqx.tree_at(
        lambda candidate: candidate.layers[0].coeffs,
        model,
        jax.random.normal(jax.random.key(3), model.layers[0].coeffs.shape),
    )
    unchanged, report = phx.nn.coarsen_kan_edges(
        noncoarse,
        {(0, 0, 0): 0.0},
        budget=1,
    )
    assert report.paths == ()
    assert unchanged is noncoarse
