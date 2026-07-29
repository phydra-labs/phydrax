#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _axis(name, size, *, periodic=False):
    return phx.nn.OperatorAxis(
        name,
        jnp.linspace(0.0, 1.0, size, endpoint=not periodic),
        quadrature_weights=jnp.full((size,), 1.0 / size),
        basis="fourier" if periodic else "uniform",
        periodic=periodic,
    )


def _parameter_count(model):
    return sum(
        int(leaf.size)
        for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    )


def _assert_finite_gradient(gradient):
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(gradient) if eqx.is_inexact_array(leaf)
    ]
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_ifno_shared_iteration_diagnostics_jit_and_query_mask():
    axis = _axis("x", 6, periodic=True)
    values = jnp.stack((jnp.sin(2.0 * jnp.pi * axis.nodes), axis.nodes))
    query_mask = jnp.array(
        [[True, True, False, True, True, True], [True, False, True, True, True, True]]
    )
    batch = phx.nn.OperatorBatch(
        inputs={"state": phx.nn.FunctionSamples(values=values, axes=(axis,))},
        queries={
            "query": phx.nn.FunctionSamples(values=None, axes=(axis,), mask=query_mask)
        },
        case_axes=("case",),
    )
    model = phx.nn.IFNO(
        n_modes=(3,),
        width=4,
        iterations=3,
        tolerance=1e-4,
        source_key="state",
        key=jr.key(1),
    )
    more_iterations = phx.nn.IFNO(
        n_modes=(3,),
        width=4,
        iterations=7,
        tolerance=1e-4,
        source_key="state",
        key=jr.key(1),
    )

    eager, eager_diagnostics = model.evaluate_with_diagnostics(batch)
    compiled, diagnostics = eqx.filter_jit(
        lambda current, data: current.evaluate_with_diagnostics(data)
    )(model, batch)

    assert _parameter_count(model) == _parameter_count(more_iterations)
    assert compiled.shape == (2, 6)
    assert jnp.allclose(compiled, eager)
    assert jnp.array_equal(compiled[~query_mask], jnp.zeros((2,)))
    assert diagnostics.absolute_residual.shape == (2,)
    assert diagnostics.relative_residual.shape == (2,)
    assert jnp.all(jnp.isfinite(diagnostics.absolute_residual))
    assert jnp.all(jnp.isfinite(diagnostics.relative_residual))
    assert jnp.array_equal(
        diagnostics.converged, diagnostics.relative_residual <= model.tolerance
    )
    assert diagnostics.iterations == eager_diagnostics.iterations == 3


def test_axial_factorized_fno_has_finite_output_and_input_gradient():
    x = jnp.linspace(0.0, 1.0, 5, endpoint=False)
    y = jnp.linspace(0.0, 1.0, 4, endpoint=False)
    values = jnp.sin(2.0 * jnp.pi * x[:, None]) * jnp.cos(2.0 * jnp.pi * y[None, :])
    model = phx.nn.AxialFactorizedFNO(
        n_modes=(3, 2),
        width=4,
        depth=1,
        factorization="cp",
        rank=2,
        key=jr.key(2),
    )

    output = model((values, x, y))
    gradient = jax.grad(lambda field: jnp.mean(model((field, x, y)) ** 2))(values)

    assert output.shape == values.shape
    assert jnp.all(jnp.isfinite(output))
    assert gradient.shape == values.shape
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 0.0


def _transolver_batch():
    coordinates = jnp.array([[0.0], [0.25], [0.7], [1.0]])
    values = jnp.array([[1.0, 80.0, -90.0, 70.0], [-0.5, 50.0, 60.0, -70.0]])
    source_mask = jnp.array([[True, False, False, False], [False, False, True, False]])
    source_weights = jnp.array([[0.3, 4.0, 5.0, 6.0], [7.0, 8.0, 0.65, 9.0]])
    query_mask = jnp.array([[True, False, True], [False, True, True]])
    source = phx.nn.FunctionSamples(
        values=values,
        coordinates=coordinates,
        quadrature_weights=source_weights,
        mask=source_mask,
    )
    query = phx.nn.FunctionSamples(
        values=None,
        coordinates=jnp.array([[0.1], [0.5], [0.9]]),
        quadrature_weights=jnp.array([0.2, 0.3, 0.5]),
        mask=query_mask,
    )
    return (
        phx.nn.OperatorBatch(
            inputs={"state": source},
            queries={"query": query},
            case_axes=("case",),
        ),
        source_mask,
        source_weights,
        query_mask,
    )


def test_transolver_hard_and_overlapping_slices_preserve_measure_and_masks():
    batch, source_mask, source_weights, query_mask = _transolver_batch()
    common = dict(
        coord_dim=1,
        num_slices=3,
        width=6,
        depth=1,
        num_heads=2,
        head_dim=3,
        source_key="state",
        attention_execution="dense",
        key=jr.key(3),
    )
    hard = phx.nn.Transolver(slice_top_k=1, **common)
    overlapping = phx.nn.Transolver(slice_top_k=2, **common)

    hard_state = hard.encode_inputs(batch)
    overlapping_state = overlapping.encode_inputs(batch)
    physical_measure = jnp.sum(jnp.where(source_mask, source_weights, 0.0), axis=-1)
    hard_output = hard(batch)
    overlapping_output = overlapping(batch)

    assert hard_state.values.shape == overlapping_state.values.shape == (2, 3, 6)
    assert jnp.array_equal(jnp.sum(hard_state.mask, axis=-1), jnp.ones((2,)))
    assert jnp.array_equal(jnp.sum(overlapping_state.mask, axis=-1), jnp.full((2,), 2))
    assert jnp.allclose(jnp.sum(hard_state.weights, axis=-1), physical_measure)
    assert jnp.allclose(jnp.sum(overlapping_state.weights, axis=-1), physical_measure)
    assert hard_output.shape == overlapping_output.shape == (2, 3)
    assert jnp.all(jnp.isfinite(hard_output))
    assert jnp.all(jnp.isfinite(overlapping_output))
    assert jnp.array_equal(hard_output[~query_mask], jnp.zeros((2,)))
    assert jnp.array_equal(overlapping_output[~query_mask], jnp.zeros((2,)))


def _gnot_batch(*, reverse=False, velocity_scale=1.0, forcing_scale=1.0):
    velocity = phx.nn.FunctionSamples(
        values=velocity_scale
        * jnp.array(
            [
                [[1.0, -0.5], [0.3, 0.7], [-0.2, 0.4]],
                [[-0.4, 0.2], [0.8, -0.1], [0.5, 0.6]],
            ]
        ),
        coordinates=jnp.array([[0.0], [0.45], [1.0]]),
        quadrature_weights=jnp.array([0.2, 0.5, 0.3]),
        mask=jnp.array([[True, True, False], [True, False, True]]),
    )
    forcing = phx.nn.FunctionSamples(
        values=forcing_scale * jnp.array([[0.2, 0.4, -0.7, 0.1], [-0.3, 0.6, 0.2, 0.8]]),
        coordinates=jnp.array([[0.1], [0.3], [0.6], [0.95]]),
        quadrature_weights=jnp.array([0.1, 0.2, 0.3, 0.4]),
    )
    inputs = (
        {"velocity": velocity, "forcing": forcing}
        if reverse
        else {"forcing": forcing, "velocity": velocity}
    )
    query_mask = jnp.array([[True, False, True], [True, True, False]])
    return phx.nn.OperatorBatch(
        inputs=inputs,
        queries={
            "query": phx.nn.FunctionSamples(
                values=None,
                coordinates=jnp.array([[0.05], [0.5], [0.85]]),
                quadrature_weights=jnp.array([0.25, 0.5, 0.25]),
                mask=query_mask,
            )
        },
        case_axes=("case",),
    )


def test_gnot_named_source_order_is_deterministic_and_heterogeneous_branches_fuse():
    settings = dict(
        out_channels=2,
        coord_dim=1,
        hidden_channels=4,
        encoder_depth=1,
        fusion_depth=1,
        transformer_depth=1,
        num_heads=1,
        attention_execution="dense",
        key=jr.key(4),
    )
    first = phx.nn.GNOT(in_channels={"velocity": 2, "forcing": "scalar"}, **settings)
    reordered = phx.nn.GNOT(in_channels={"forcing": "scalar", "velocity": 2}, **settings)
    batch = _gnot_batch()
    reversed_batch = _gnot_batch(reverse=True)
    output = first(batch)

    assert first.source_keys == reordered.source_keys == ("forcing", "velocity")
    assert output.shape == (2, 3, 2)
    assert jnp.allclose(output, reordered(reversed_batch))
    assert not jnp.allclose(output, first(_gnot_batch(forcing_scale=0.0)))
    assert not jnp.allclose(output, first(_gnot_batch(velocity_scale=0.0)))
    query_mask = batch.require_single_query().mask_array(case_shape=batch.case_shape)
    assert jnp.array_equal(output[~query_mask], jnp.zeros((2, 2)))


def _koopman_batch():
    x = _axis("x", 4)
    time = phx.nn.OperatorAxis("time", jnp.array([0.0, 0.17, 0.61]))
    values = jnp.stack((jnp.sin(jnp.pi * x.nodes), jnp.cos(jnp.pi * x.nodes)))
    query_mask = (
        jnp.ones((2, 4, 3), dtype=bool).at[0, 1, 1].set(False).at[1, 3, 2].set(False)
    )
    return (
        phx.nn.OperatorBatch(
            inputs={"state": phx.nn.FunctionSamples(values=values, axes=(x,))},
            queries={
                "query": phx.nn.FunctionSamples(
                    values=None, axes=(x, time), mask=query_mask
                )
            },
            case_axes=("case",),
        ),
        query_mask,
    )


@pytest.mark.parametrize("evolution", ("continuous", "discrete"))
def test_koopman_stability_semigroup_and_irregular_time_queries(evolution):
    model = phx.nn.KoopmanTemporalOperator(
        spatial_ndim=1,
        latent_size=3,
        hidden_size=5,
        depth=1,
        evolution=evolution,
        source_key="state",
        key=jr.key(5),
    )
    if evolution == "continuous":
        eigenvalues = jnp.linalg.eigvals(model.generator_matrix())
        assert jnp.max(jnp.real(eigenvalues)) < 0.0
    else:
        eigenvalues = jnp.linalg.eigvals(model.discrete_matrix())
        assert jnp.max(jnp.abs(eigenvalues)) < 1.0
        assert jnp.min(jnp.real(eigenvalues)) > 0.0

    first_time = 0.17
    second_time = 0.44
    first = model.evolution_matrix(first_time)
    second = model.evolution_matrix(second_time)
    combined = model.evolution_matrix(first_time + second_time)
    batch, query_mask = _koopman_batch()
    output = model(batch)

    assert jnp.allclose(combined, first @ second, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(model.evolution_matrix(0.0), jnp.eye(3), atol=1e-6)
    assert output.shape == (2, 4, 3)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.array_equal(output[~query_mask], jnp.zeros((2,)))


def _green_batch(
    *, forcing_weights=True, boundary_weights=True, forcing_scale=1.0, boundary_scale=1.0
):
    forcing = phx.nn.FunctionSamples(
        values=forcing_scale * jnp.array([[1.0, -0.5, 0.25], [-0.2, 0.7, 0.4]]),
        coordinates=jnp.array([[0.1], [0.5], [0.9]]),
        quadrature_weights=(jnp.array([0.2, 0.5, 0.3]) if forcing_weights else None),
    )
    boundary = phx.nn.FunctionSamples(
        values=boundary_scale * jnp.array([[0.6, -0.1], [-0.4, 0.9]]),
        coordinates=jnp.array([[0.0], [1.0]]),
        quadrature_weights=(jnp.array([0.5, 0.5]) if boundary_weights else None),
    )
    query_mask = jnp.array([[True, False, True], [True, True, False]])
    return phx.nn.OperatorBatch(
        inputs={"forcing": forcing, "boundary": boundary},
        queries={
            "query": phx.nn.FunctionSamples(
                values=None,
                coordinates=jnp.array([[0.2], [0.55], [0.8]]),
                mask=query_mask,
            )
        },
        case_axes=("case",),
    )


def test_green_kernel_requires_physical_measures_and_uses_both_branches():
    model = phx.nn.GreenKernelOperator(
        coord_dim=1,
        width=4,
        depth=1,
        kernel_width=5,
        kernel_depth=1,
        query_chunk_size=2,
        key=jr.key(6),
    )
    batch = _green_batch()
    output = model(batch)

    with pytest.raises(ValueError, match="Interior forcing requires physical quadrature"):
        model(_green_batch(forcing_weights=False))
    with pytest.raises(ValueError, match="Boundary data requires physical quadrature"):
        model(_green_batch(boundary_weights=False))
    assert output.shape == (2, 3)
    assert not jnp.allclose(output, model(_green_batch(forcing_scale=0.0)))
    assert not jnp.allclose(output, model(_green_batch(boundary_scale=0.0)))
    query_mask = batch.require_single_query().mask_array(case_shape=batch.case_shape)
    assert jnp.array_equal(output[~query_mask], jnp.zeros((2,)))


def _poseidon_batch(values, time):
    axes = (_axis("x", 4), _axis("y", 4))
    query_mask = (
        jnp.ones((2, 4, 4), dtype=bool).at[0, 1, 2].set(False).at[1, 3, 0].set(False)
    )
    return phx.nn.OperatorBatch(
        inputs={
            "state": phx.nn.FunctionSamples(values=values, axes=axes),
            "time": phx.nn.FunctionSamples(values=time),
        },
        queries={
            "query": phx.nn.FunctionSamples(values=None, axes=axes, mask=query_mask)
        },
        case_axes=("case",),
    )


def test_poseidon_eager_jit_gradient_time_conditioning_and_mask():
    model = phx.nn.Poseidon(
        image_shape=(4, 4),
        patch_size=(2, 2),
        embed_dim=4,
        depths=(1,),
        num_heads=(1,),
        window_size=2,
        time_input_name="time",
        source_key="state",
        key=jr.key(7),
    )
    values = jr.normal(jr.key(8), (2, 4, 4))
    time = jnp.array([0.2, 0.7])
    batch = _poseidon_batch(values, time)
    eager = model(batch)
    compiled = eqx.filter_jit(lambda current, data: current(data))(model, batch)
    gradient = jax.grad(lambda field: jnp.mean(model(_poseidon_batch(field, time)) ** 2))(
        values
    )

    assert eager.shape == compiled.shape == (2, 4, 4)
    assert jnp.allclose(compiled, eager, rtol=1e-5, atol=1e-6)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 0.0
    zero_time = model(_poseidon_batch(values, jnp.zeros_like(time)))
    assert not jnp.allclose(eager, zero_time)
    mask = batch.require_single_query().mask_array(case_shape=batch.case_shape)
    assert jnp.array_equal(eager[~mask], jnp.zeros((2,)))


def _dpot_batch(history):
    axes = (_axis("x", 4), _axis("y", 4))
    history_axis = phx.nn.OperatorAxis("history_time", jnp.array([-1.0, 0.0]))
    forecast_axis = phx.nn.OperatorAxis("forecast_time", jnp.array([0.4]))
    source_mask = (
        jnp.ones((2, 4, 4, 2), dtype=bool)
        .at[0, 1, 2, 0]
        .set(False)
        .at[1, 3, 0, 1]
        .set(False)
    )
    query_mask = (
        jnp.ones((2, 4, 4, 1), dtype=bool)
        .at[0, 0, 1, 0]
        .set(False)
        .at[1, 2, 3, 0]
        .set(False)
    )
    return phx.nn.OperatorBatch(
        inputs={
            "history": phx.nn.FunctionSamples(
                values=history,
                axes=axes + (history_axis,),
                mask=source_mask,
            )
        },
        queries={
            "query": phx.nn.FunctionSamples(
                values=None,
                axes=axes + (forecast_axis,),
                mask=query_mask,
            )
        },
        case_axes=("case",),
    )


def test_dpot_eager_jit_gradient_and_corrupt_batch_contract():
    model = phx.nn.DPOT(
        image_shape=(4, 4),
        history_steps=2,
        forecast_steps=1,
        patch_size=(2, 2),
        embed_dim=4,
        depth=1,
        modes=(2, 2),
        num_blocks=1,
        out_layer_dim=2,
        normalization_groups=1,
        source_key="history",
        key=jr.key(9),
    )
    history = jr.normal(jr.key(10), (2, 4, 4, 2))
    batch = _dpot_batch(history)
    eager = model(batch)
    compiled = eqx.filter_jit(lambda current, data: current(data))(model, batch)
    gradient = jax.grad(lambda field: jnp.mean(model(_dpot_batch(field)) ** 2))(history)
    corruption_key = jr.key(11)
    corrupted_batch = model.corrupt_batch(batch, noise_scale=0.1, key=corruption_key)
    corrupted = corrupted_batch.input("history").values
    source_mask = batch.input("history").mask_array(case_shape=batch.case_shape)
    expected = phx.nn.dpot_corrupt_history(
        history,
        noise_scale=0.1,
        key=corruption_key,
        mask=source_mask,
        channel_axis=None,
    )

    assert eager.shape == compiled.shape == (2, 4, 4, 1)
    assert jnp.allclose(compiled, eager, rtol=1e-5, atol=1e-6)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 0.0
    assert jnp.array_equal(corrupted, expected)
    assert jnp.array_equal(corrupted[~source_mask], jnp.zeros((2,)))
    assert not jnp.array_equal(corrupted[source_mask], history[source_mask])
    query_mask = batch.require_single_query().mask_array(case_shape=batch.case_shape)
    assert jnp.array_equal(eager[~query_mask], jnp.zeros((2,)))


def _attention(kernel="softmax", execution="dense", *, key=jr.key(12)):
    return phx.nn.MeasureAwareAttention(
        source_channels=4,
        query_channels=3,
        out_channels=5,
        num_heads=2,
        head_dim=2,
        kernel=kernel,
        execution=execution,
        block_size=2,
        key=key,
    )


def test_measure_attention_dense_blockwise_parity_with_measure_and_masks():
    source = jr.normal(jr.key(13), (2, 5, 4))
    query = jr.normal(jr.key(14), (2, 3, 3))
    weights = jnp.array([[0.1, 0.4, 0.2, 0.25, 0.05], [0.3, 0.1, 0.4, 0.1, 0.1]])
    source_mask = jnp.array(
        [[True, True, False, True, True], [True, False, True, True, False]]
    )
    query_mask = jnp.array([[True, False, True], [False, True, True]])
    dense = _attention(execution="dense")
    blockwise = _attention(execution="blockwise")

    expected = dense(
        source,
        query,
        weights,
        source_mask=source_mask,
        query_mask=query_mask,
    )
    actual = blockwise(
        source,
        query,
        weights,
        source_mask=source_mask,
        query_mask=query_mask,
    )

    assert jnp.allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert jnp.array_equal(actual[~query_mask], jnp.zeros((2, 5)))


@pytest.mark.parametrize("kernel", ("softmax", "kernel_linear", "galerkin", "identity"))
def test_measure_attention_kernel_modes_are_finite_and_all_masked_is_zero(kernel):
    source = jr.normal(jr.key(15), (2, 4, 4))
    query = jr.normal(jr.key(16), (2, 4, 3))
    weights = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    model = _attention(kernel=kernel)
    output = model(source, query, weights)
    masked = model(
        source,
        query,
        weights,
        source_mask=jnp.zeros((2, 4), dtype=bool),
        query_mask=jnp.array([[True, False, True, True], [False, True, True, False]]),
    )

    assert output.shape == masked.shape == (2, 4, 5)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.all(jnp.isfinite(masked))
    assert jnp.array_equal(masked, jnp.zeros_like(masked))
