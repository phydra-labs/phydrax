#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _domain(series_tail: float = 99.0) -> phx.domain.RaggedSeriesDatasetDomain:
    static = jnp.asarray([[1.0, 0.5], [2.0, -1.0]])
    series = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [series_tail, series_tail]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ]
    )
    lengths = jnp.asarray([2, 3], dtype=jnp.int32)
    return phx.domain.RaggedSeriesDatasetDomain(
        series,
        lengths,
        static=static,
        dt=0.5,
    )


def _batch(domain: phx.domain.RaggedSeriesDatasetDomain):
    return domain.points_from_indices(
        jnp.asarray([0, 1], dtype=jnp.int32),
        structure=phx.domain.SampleLayout((("data",),)),
    )


def test_ragged_series_model_returns_case_axis_field():
    domain = _domain()

    def exact(payload, *, key=None):
        del key
        return payload.static[:, 0] + jnp.sum(
            payload.series[..., 0] * payload.mask.astype(float),
            axis=1,
        )

    u = domain.Function("data")(phx.nn.models.RaggedSeriesModel(exact))
    out = u(_batch(domain), key=jr.key(0))
    axis = _batch(domain).structure.axis_for("data")

    assert out.dims == (axis,)
    assert jnp.allclose(jnp.asarray(out.data), jnp.asarray([5.0, 23.0]))


def test_masked_series_pooling_ignores_padded_tail_values():
    domain_a = _domain(series_tail=99.0)
    domain_b = _domain(series_tail=-123.0)

    def step_model(x, *, key=None):
        del key
        return x[..., :1]

    def readout_model(x, *, key=None):
        del key
        return x[:, 0]

    model = phx.nn.models.MaskedSeriesPoolingModel(
        step_model=step_model,
        readout_model=readout_model,
        reduction="mean",
        include_time=False,
        include_static_in_readout=False,
    )
    u_a = domain_a.Function("data")(phx.nn.models.RaggedSeriesModel(model))
    u_b = domain_b.Function("data")(phx.nn.models.RaggedSeriesModel(model))

    out_a = u_a(_batch(domain_a), key=jr.key(1)).data
    out_b = u_b(_batch(domain_b), key=jr.key(1)).data
    assert jnp.allclose(out_a, out_b)
    assert jnp.allclose(out_a, jnp.asarray([2.0, 7.0]))


def test_masked_series_pooling_selects_nonfinite_padded_latents():
    payload = phx.nn.models.RaggedSeriesBatchInput(
        static=None,
        series=jnp.asarray([[[1.0], [0.0]]]),
        time=jnp.asarray([[0.0, 1.0]]),
        mask=jnp.asarray([[True, False]]),
        length=jnp.asarray([1], dtype=jnp.int32),
    )

    def step_model(x, *, key=None):
        del key
        return jnp.log(x)

    def readout_model(x, *, key=None):
        del key
        return x[:, 0]

    model = phx.nn.models.MaskedSeriesPoolingModel(
        step_model=step_model,
        readout_model=readout_model,
        reduction="sum",
        include_time=False,
        include_static_in_readout=False,
    )
    eager = model(payload)
    compiled = eqx.filter_jit(lambda current, data: current(data))(model, payload)

    assert jnp.array_equal(eager, jnp.asarray([0.0]))
    assert jnp.array_equal(compiled, eager)


def test_masked_series_pooling_can_scale_sampled_sum():
    payload = phx.nn.models.RaggedSeriesBatchInput(
        static=None,
        series=jnp.asarray([[[1.0], [3.0]]]),
        time=jnp.asarray([[0.0, 1.0]]),
        mask=jnp.asarray([[True, True]]),
        length=jnp.asarray([4], dtype=jnp.int32),
        sample_index=jnp.asarray([[0, 1]], dtype=jnp.int32),
        sample_scale=jnp.asarray([2.0]),
    )

    def step_model(x, *, key=None):
        del key
        return x[..., :1]

    def readout_model(x, *, key=None):
        del key
        return x[:, 0]

    unscaled = phx.nn.models.MaskedSeriesPoolingModel(
        step_model=step_model,
        readout_model=readout_model,
        reduction="sum",
        include_time=False,
        include_static_in_readout=False,
    )
    scaled = phx.nn.models.MaskedSeriesPoolingModel(
        step_model=step_model,
        readout_model=readout_model,
        reduction="sum",
        include_time=False,
        include_static_in_readout=False,
        scale_sampled_sum=True,
    )

    assert jnp.allclose(unscaled(payload), jnp.asarray([4.0]))
    assert jnp.allclose(scaled(payload), jnp.asarray([8.0]))


def test_masked_series_pooling_has_finite_parameter_gradients():
    domain = _domain()
    payload = phx.nn.models.RaggedSeriesBatchInput(
        static=domain.input_rows(jnp.asarray([0, 1], dtype=jnp.int32))["static"],
        series=domain.input_rows(jnp.asarray([0, 1], dtype=jnp.int32))["series"],
        time=domain.input_rows(jnp.asarray([0, 1], dtype=jnp.int32))["time"],
        mask=domain.input_rows(jnp.asarray([0, 1], dtype=jnp.int32))["mask"],
        length=domain.input_rows(jnp.asarray([0, 1], dtype=jnp.int32))["length"],
    )
    key_step, key_readout = jr.split(jr.key(2))
    model = phx.nn.models.MaskedSeriesPoolingModel(
        step_model=phx.nn.models.MLP(
            in_size=3,
            out_size=4,
            width_size=8,
            depth=1,
            key=key_step,
        ),
        readout_model=phx.nn.models.MLP(
            in_size=6,
            out_size=2,
            width_size=8,
            depth=1,
            key=key_readout,
        ),
    )

    def total(m):
        return jnp.sum(m(payload, key=jr.key(3)))

    value, grads = eqx.filter_value_and_grad(total)(model)
    grad_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(grads) if eqx.is_array(leaf)
    ]

    assert jnp.isfinite(value)
    assert grad_leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in grad_leaves)
