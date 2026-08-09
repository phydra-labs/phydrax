import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


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
    nodes = jnp.linspace(0.0, 1.0, 9)
    axis = phx.nn.operator.OperatorAxis("x", nodes)
    source_mask = jnp.asarray([True, True, False, True, True, False, True, True, True])
    query_mask = jnp.asarray([True, True, True, True, True, True, True, True, False])
    quadrature = jnp.linspace(0.5, 1.5, 9)
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
    nodes = jnp.linspace(-1.0, 1.0, 11)
    axis = phx.nn.operator.OperatorAxis("x", nodes)
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
