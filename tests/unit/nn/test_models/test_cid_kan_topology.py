import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def test_cid11_trainable_grid_bank_has_independent_ordered_rows():
    bank = phx.nn.models.TrainableBSplineGridBank.open_uniform(2, 3, 4)
    bank = eqx.tree_at(
        lambda current: current.raw_span_logits,
        bank,
        bank.raw_span_logits.at[1].set(jnp.asarray([-2.0, -1.0, 1.0, 2.0])),
    )
    assert bank.knots.shape == (2, 11)
    assert jnp.all(jnp.diff(bank.breakpoints, axis=-1) > 0.0)
    assert not jnp.allclose(bank.span_widths[0], bank.span_widths[1])


def test_trainable_grid_bank_equinox_filter_exposes_only_logits():
    bank = phx.nn.models.TrainableBSplineGridBank.open_uniform(
        2,
        3,
        4,
        intervals=((-2.0, 1.0), (0.0, 4.0)),
        minimum_spans=(0.05, 0.1),
    )
    trainable = eqx.filter(bank, eqx.is_inexact_array)

    leaves = jax.tree.leaves(trainable)
    assert len(leaves) == 1
    assert leaves[0] is bank.raw_span_logits

    updates = eqx.tree_at(
        lambda current: current.raw_span_logits,
        trainable,
        jnp.asarray(
            ((-4.0, -1.0, 2.0, 5.0), (5.0, 2.0, -1.0, -4.0)),
            dtype=bank.raw_span_logits.dtype,
        ),
    )
    updated = eqx.apply_updates(bank, updates)

    assert jnp.array_equal(updated.intervals, bank.intervals)
    assert jnp.array_equal(updated.minimum_spans, bank.minimum_spans)
    assert jnp.all(jnp.diff(updated.breakpoints, axis=-1) > 0.0)
    assert jnp.all(jnp.diff(updated.knots, axis=-1) >= 0.0)


def test_cid12_rational_trainable_refinement_is_topology_evidenced():
    grid = phx.nn.models.TrainableBSplineGrid.open_uniform(2, 3)
    model = phx.nn.models.KAN(
        in_size=1,
        out_size="scalar",
        width_size=2,
        depth=1,
        edge_basis=phx.nn.models.RationalBSplineEdgeBasis(grid=grid),
        skip_connection=False,
        key=jax.random.key(0),
    )
    adapted, report = phx.nn.models.refine_kan_edges(
        model,
        {(0, 0, 0): jnp.asarray([0.0, 1.0, 0.0])},
        budget=1,
    )
    assert report.basis_kinds == ("rational",)
    assert report.source_topology_id != report.target_topology_id
    assert report.differentiability_certified is False
    assert jnp.allclose(model(jnp.asarray([0.2])), adapted(jnp.asarray([0.2])), atol=1e-6)
