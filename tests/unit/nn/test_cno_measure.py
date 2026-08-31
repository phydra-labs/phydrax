import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.operator.architectures.spectral._cno import _coordinate_features


def _batch(values, axis, *, source_mask, query_mask, quadrature):
    source = phx.nn.operator.FunctionSamples(
        values=values,
        axes=(axis,),
        mask=source_mask,
        quadrature_weights=quadrature,
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        axes=(axis,),
        mask=query_mask,
        quadrature_weights=quadrature,
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"source": source}, queries={"query": query}
    )


def test_cno_masks_invalid_sources_before_lifting_and_zeros_invalid_queries():
    nodes = jnp.arange(9, dtype=float) / 9
    quadrature = jnp.linspace(0.5, 1.5, 9)
    axis = phx.nn.operator.OperatorAxis(
        "x",
        nodes,
        quadrature_weights=quadrature,
        basis="fourier",
        periodic=True,
    )
    source_mask = jnp.asarray([True, True, False, True, True, False, True, True, True])
    query_mask = jnp.asarray([True, True, True, True, True, True, True, True, False])
    values = jnp.sin(2.0 * jnp.pi * nodes)
    invalid_nan = jnp.where(source_mask, values, jnp.nan)
    invalid_large = jnp.where(source_mask, values, 1e6)
    model = phx.nn.operator.architectures.CNO(
        spatial_ndim=1,
        width=4,
        depth=2,
        oversample_factor=1,
        key=jr.key(2),
    )

    nan_output = eqx.filter_jit(model)(
        _batch(
            invalid_nan,
            axis,
            source_mask=source_mask,
            query_mask=query_mask,
            quadrature=quadrature,
        )
    )
    large_output = model(
        _batch(
            invalid_large,
            axis,
            source_mask=source_mask,
            query_mask=query_mask,
            quadrature=quadrature,
        )
    )

    assert jnp.all(jnp.isfinite(nan_output))
    assert jnp.allclose(nan_output, large_output, atol=1e-6, rtol=1e-6)
    assert nan_output[-1] == 0.0


def test_cno_uniform_measure_full_support_matches_array_evaluation():
    nodes = -1.0 + 2.0 * jnp.arange(11, dtype=float) / 11
    axis = phx.nn.operator.OperatorAxis(
        "x",
        nodes,
        quadrature_weights=jnp.full(11, 0.2),
        basis="fourier",
        periodic=True,
    )
    values = jnp.cos(3.0 * nodes)
    mask = jnp.ones(11, dtype=bool)
    model = phx.nn.operator.architectures.CNO(
        spatial_ndim=1,
        width=3,
        depth=1,
        oversample_factor=1,
        key=jr.key(5),
    )

    batch_output = model(
        _batch(
            values,
            axis,
            source_mask=mask,
            query_mask=mask,
            quadrature=jnp.full(11, 0.2),
        )
    )
    array_output = model((values, nodes))

    assert jnp.allclose(batch_output, array_output, atol=2e-6, rtol=2e-6)


def test_periodic_coordinate_features_have_equal_interior_and_seam_chords():
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.arange(13, dtype=float) / 13,
        basis="fourier",
        periodic=True,
    )
    features = _coordinate_features((axis,), ())
    closed = jnp.concatenate((features, features[:1]), axis=0)
    chords = jnp.linalg.norm(jnp.diff(closed, axis=0), axis=-1)

    assert features.shape == (13, 2)
    assert jnp.allclose(chords, jnp.full_like(chords, chords[0]))
    assert jnp.allclose(features[0], jnp.asarray([0.0, 1.0]))


@pytest.mark.parametrize(
    "axis",
    (
        phx.nn.operator.OperatorAxis("x", jnp.arange(5.0) / 5),
        phx.nn.operator.OperatorAxis(
            "x", jnp.arange(5.0) / 5, basis="legendre", periodic=True
        ),
        phx.nn.operator.OperatorAxis(
            "x", jnp.asarray([0.0, 0.25, jnp.nan, 0.75]), periodic=True
        ),
        phx.nn.operator.OperatorAxis(
            "x", jnp.asarray([0.0, 0.25, 0.25, 0.75]), periodic=True
        ),
        phx.nn.operator.OperatorAxis(
            "x", jnp.asarray([0.75, 0.5, 0.25, 0.0]), periodic=True
        ),
        phx.nn.operator.OperatorAxis(
            "x", jnp.asarray([0.0, 0.4, 0.7, 0.9]), periodic=True
        ),
        phx.nn.operator.OperatorAxis("x", jnp.asarray([0.0]), periodic=True),
    ),
)
def test_cno_rejects_axes_outside_periodic_uniform_fourier_contract(axis):
    model = phx.nn.operator.architectures.CNO(
        spatial_ndim=1,
        width=3,
        depth=1,
        oversample_factor=1,
        key=jr.key(20),
    )
    values = jnp.ones((axis.size,))
    axis = phx.nn.operator.OperatorAxis(
        axis.name,
        axis.nodes,
        quadrature_weights=jnp.ones((axis.size,)),
        basis=axis.basis,
        periodic=axis.periodic,
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={"source": phx.nn.operator.FunctionSamples(values=values, axes=(axis,))},
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
    )

    with pytest.raises(ValueError):
        model(batch)


def test_cno_rejects_noncoincident_source_and_query_axes():
    source_axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.arange(7, dtype=float) / 7,
        basis="fourier",
        periodic=True,
        quadrature_weights=jnp.ones((7,)),
    )
    query_axis = phx.nn.operator.OperatorAxis(
        "x",
        0.1 + jnp.arange(7, dtype=float) / 7,
        basis="fourier",
        periodic=True,
        quadrature_weights=jnp.ones((7,)),
    )
    model = phx.nn.operator.architectures.CNO(
        spatial_ndim=1,
        width=3,
        depth=1,
        oversample_factor=1,
        key=jr.key(21),
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "source": phx.nn.operator.FunctionSamples(
                values=jnp.ones(7),
                axes=(source_axis,),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                axes=(query_axis,),
            )
        },
    )

    with pytest.raises(ValueError, match="exactly coincident"):
        model(batch)


def test_cno_uses_circular_measure_layers_and_periodic_feature_width():
    model = phx.nn.operator.architectures.CNO(
        spatial_ndim=2,
        in_channels=2,
        width=4,
        depth=2,
        oversample_factor=1,
        key=jr.key(22),
    )

    assert model.lift.in_size == 6
    assert all(block.first.circular and block.second.circular for block in model.blocks)


def test_cno_rejects_missing_physical_source_quadrature():
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.arange(7, dtype=float) / 7,
        basis="fourier",
        periodic=True,
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "source": phx.nn.operator.FunctionSamples(
                values=jnp.ones((7,)),
                axes=(axis,),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                axes=(axis,),
            )
        },
    )
    model = phx.nn.operator.architectures.CNO(
        spatial_ndim=1,
        width=3,
        depth=1,
        oversample_factor=1,
        key=jr.key(23),
    )

    with pytest.raises(ValueError, match="physical source quadrature"):
        model(batch)


def test_uno_rejects_masked_source_or_query_sites():
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.arange(7, dtype=float) / 7,
        quadrature_weights=jnp.ones((7,)),
        basis="fourier",
        periodic=True,
    )
    mask = jnp.asarray([True, True, False, True, True, True, True])
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "source": phx.nn.operator.FunctionSamples(
                values=jnp.ones((7,)),
                axes=(axis,),
                mask=mask,
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                axes=(axis,),
            )
        },
    )
    model = phx.nn.operator.architectures.UNO(
        spatial_ndim=1,
        widths=(3, 4),
        oversample_factor=1,
        key=jr.key(24),
    )

    with pytest.raises(ValueError, match="does not support masked"):
        model(batch)
