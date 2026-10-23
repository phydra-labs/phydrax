#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit construction dtype across the measure-aware attention family with x64.

The float32 contract is checked where it can break: every learned parameter of
every public constructor, traced construction/forward/derivative/update programs
(no float64 value anywhere, as an FP32-only device requires), higher
derivatives against a float64 evaluation of the *same* float32 parameters and
data, and physical measures, masks and empty support. Operator models consume
explicitly float32 `FunctionSamples`, so their route is float32 end to end.
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.nn.layers import MeasureAwareAttention
from phydrax.nn.operator import (
    FunctionSamples,
    OperatorBatch,
    OperatorBranchGraph,
    OperatorBranchSpec,
    OperatorFieldSpec,
    OperatorOutputSpec,
    OperatorPrompt,
    OperatorSupervisedExample,
    PromptedOperatorBatch,
)
from phydrax.nn.operator.architectures import (
    ABUPT,
    CoDABlock,
    CoDANO,
    GNOT,
    InContextOperator,
    LatentTokenBlock,
    LatentTokenProcessor,
    SpectralConvND,
    SpectralConvolutionResourcePolicy,
    Transolver,
    UPT,
)
from phydrax.nn.operator.layers import (
    AxialOperatorAttention,
    CodomainAttention,
    OperatorAttention,
    SliceAttention,
)
from phydrax.optim import adam
from tests._support.assertions import assert_no_float64_values, assert_tree_equal


# Construction dtype is a dtype contract surface: no silent promotion allowed.
pytestmark = pytest.mark.strict_jax


# float32 unit roundoff is 6e-8; 1e-4 allows a 1e3 amplification through the
# softmax/normalization Hessian and third-order training chains while an O(1)
# derivative defect (wrong role dtype, dropped measure, detached parameter)
# remains far outside it.
_FLOAT32_RELATIVE_BOUND = 1e-4
_UNUSABLE_KEY: Any = None
_SINGLE_PRECISION = (jnp.dtype(jnp.float32), jnp.dtype(jnp.complex64))


# One shared static schema per model family, so default/explicit constructions
# compare equal by identity in their static branch and field declarations.
_GRAPH = OperatorBranchGraph(
    (
        OperatorBranchSpec(
            "field",
            role="both",
            geometry_kind="point_cloud",
            source_name="u",
            output_spec=OperatorOutputSpec("scalar"),
            query_name="query",
        ),
    )
)
_FIELDS = (OperatorFieldSpec("u", channels="scalar", role="both", query_name="query"),)


# Every public constructor of the shared attention family, built from one key.
_CONSTRUCTORS: dict[str, Callable[..., Any]] = {
    "MeasureAwareAttention": lambda key, **kwargs: MeasureAwareAttention(
        source_channels=3,
        query_channels=2,
        out_channels=2,
        num_heads=2,
        head_dim=2,
        key=key,
        **kwargs,
    ),
    "OperatorAttention": lambda key, **kwargs: OperatorAttention(
        source_channels=3, query_channels=2, num_heads=1, head_dim=2, key=key, **kwargs
    ),
    "SliceAttention": lambda key, **kwargs: SliceAttention(
        channels=3, num_slices=2, num_heads=1, head_dim=2, key=key, **kwargs
    ),
    "CodomainAttention": lambda key, **kwargs: CodomainAttention(
        channels=3, num_heads=1, head_dim=2, key=key, **kwargs
    ),
    "AxialOperatorAttention": lambda key, **kwargs: AxialOperatorAttention(
        channels=3, num_heads=1, head_dim=2, key=key, **kwargs
    ),
    "LatentTokenBlock": lambda key, **kwargs: LatentTokenBlock(
        4, num_heads=2, key=key, **kwargs
    ),
    "LatentTokenProcessor": lambda key, **kwargs: LatentTokenProcessor(
        4, 2, num_heads=2, key=key, **kwargs
    ),
    "SpectralConvND-dense": lambda key, **kwargs: SpectralConvND(
        in_channels=2, out_channels=3, n_modes=(2, 2), key=key, **kwargs
    ),
    "SpectralConvND-cp": lambda key, **kwargs: SpectralConvND(
        in_channels=2,
        out_channels=3,
        n_modes=(2, 2),
        factorization="cp",
        rank=2,
        key=key,
        **kwargs,
    ),
    "SpectralConvND-tucker": lambda key, **kwargs: SpectralConvND(
        in_channels=2,
        out_channels=3,
        n_modes=(2, 2),
        factorization="tucker",
        rank=2,
        key=key,
        **kwargs,
    ),
    "CoDABlock": lambda key, **kwargs: CoDABlock(
        width=4,
        spatial_ndim=1,
        n_modes=2,
        num_heads=1,
        head_dim=4,
        feed_forward_multiplier=2.0,
        factorization="dense",
        rank=0.5,
        key=key,
        **kwargs,
    ),
    "UPT": lambda key, **kwargs: UPT(
        in_channels="scalar",
        out_channels="scalar",
        coord_dim=1,
        width=4,
        num_tokens=2,
        depth=1,
        num_heads=1,
        source_key="u",
        key=key,
        **kwargs,
    ),
    "ABUPT": lambda key, **kwargs: ABUPT(
        _GRAPH,
        input_channels={"field": "scalar"},
        coord_dims={"field": 1},
        anchor_counts={"field": 3},
        width=4,
        depth=1,
        num_heads=1,
        key=key,
        **kwargs,
    ),
    "Transolver": lambda key, **kwargs: Transolver(
        coord_dim=1,
        num_slices=2,
        width=4,
        depth=1,
        num_heads=1,
        source_key="u",
        key=key,
        **kwargs,
    ),
    "GNOT": lambda key, **kwargs: GNOT(
        in_channels={"u": "scalar"},
        coord_dim=1,
        hidden_channels=4,
        encoder_depth=1,
        fusion_depth=1,
        transformer_depth=1,
        num_heads=1,
        key=key,
        **kwargs,
    ),
    "InContextOperator": lambda key, **kwargs: InContextOperator(
        in_channels="scalar",
        out_channels="scalar",
        coord_dim=1,
        width=4,
        num_tokens=2,
        prompt_depth=1,
        processor_depth=1,
        num_heads=1,
        source_key="u",
        key=key,
        **kwargs,
    ),
    "CoDANO": lambda key, **kwargs: CoDANO(
        _FIELDS,
        (4,),
        n_modes=2,
        width=4,
        depth=1,
        num_heads=1,
        head_dim=4,
        key=key,
        **kwargs,
    ),
}
_OPERATOR_MODELS = ("UPT", "ABUPT", "Transolver", "GNOT", "InContextOperator", "CoDANO")


def _inexact_leaves(tree: Any, /) -> list[jax.Array]:
    return jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))


def _assert_single_precision(tree: Any, /) -> None:
    leaves = _inexact_leaves(tree)
    assert leaves
    for leaf in leaves:
        assert leaf.dtype in _SINGLE_PRECISION
        assert bool(jnp.all(jnp.isfinite(leaf)))


def _upcast(tree: Any, /) -> Any:
    def widen(leaf: Any) -> Any:
        if not eqx.is_inexact_array(leaf):
            return leaf
        complex_leaf = jnp.issubdtype(leaf.dtype, jnp.complexfloating)
        return leaf.astype(jnp.complex128 if complex_leaf else jnp.float64)

    return jax.tree.map(widen, tree)


def _relative_error(actual: Any, reference: Any, /) -> float:
    difference = 0.0
    norm = 0.0
    for leaf, reference_leaf in zip(
        _inexact_leaves(actual), _inexact_leaves(reference), strict=True
    ):
        error = leaf.astype(reference_leaf.dtype) - reference_leaf
        difference += float(jnp.sum(jnp.abs(error) ** 2))
        norm += float(jnp.sum(jnp.abs(reference_leaf) ** 2))
    return (difference / norm) ** 0.5


def _train_step(
    loss: Callable[[Any], jax.Array], parameters: Any, rest: tuple[Any, ...]
) -> Any:
    # `rest` holds the remaining partition lanes (role lanes or Equinox static).
    def objective(candidate: Any) -> jax.Array:
        return loss(eqx.combine(candidate, *rest))

    value, gradient = eqx.filter_value_and_grad(objective)(parameters)
    optimizer = adam(1e-3)
    updates, optimizer_state = optimizer.update(
        gradient, optimizer.init(parameters), parameters
    )
    return value, gradient, optimizer_state, eqx.apply_updates(parameters, updates)


def _assert_float32_update(
    gradient: Any, optimizer_state: Any, updated: Any, parameters: Any, /
) -> None:
    _assert_single_precision(gradient)
    _assert_single_precision(optimizer_state)
    _assert_single_precision(updated)
    assert any(
        bool(jnp.any(new != old))
        for new, old in zip(
            _inexact_leaves(updated), _inexact_leaves(parameters), strict=True
        )
    )


@pytest.mark.parametrize("name", tuple(_CONSTRUCTORS))
def test_explicit_dtype_reaches_every_learned_parameter(name: str) -> None:
    build = _CONSTRUCTORS[name]
    default = build(jr.key(7))

    # Default construction is unchanged: same keys, values and x64 dtypes.
    assert_tree_equal(build(jr.key(7), dtype="float64"), default)

    # Every inexact leaf (parameters, learned tokens, norms and model-owned
    # latent geometry) follows the selected precision with unchanged layout.
    explicit = build(jr.key(7), dtype=jnp.float32)
    assert jax.tree.structure(explicit) == jax.tree.structure(default)
    for leaf, reference in zip(
        _inexact_leaves(explicit), _inexact_leaves(default), strict=True
    ):
        assert leaf.shape == reference.shape
        assert reference.dtype in (jnp.dtype(jnp.float64), jnp.dtype(jnp.complex128))
    _assert_single_precision(explicit)

    # Tracing the float32 construction allocates no float64 value.
    construction, _, _ = eqx.filter_make_jaxpr(
        lambda key: eqx.filter(build(key, dtype=jnp.float32), eqx.is_array)
    )(jr.key(0))
    assert_no_float64_values(construction)


@pytest.mark.parametrize("name", tuple(_CONSTRUCTORS))
@pytest.mark.parametrize(
    ("dtype", "message"),
    (
        (jnp.int32, "'int32'"),
        (jnp.complex64, "not real floating-point"),
    ),
)
def test_non_real_floating_dtype_refuses_before_key_use(
    name: str, dtype: Any, message: str
) -> None:
    # An unusable key proves refusal happens before any key split/allocation.
    with pytest.raises(ValueError, match=message):
        _CONSTRUCTORS[name](_UNUSABLE_KEY, dtype=dtype)


# Softmax routes (dense, fused float32 XLA, source-blockwise) and linear kernels.
_KERNELS: dict[str, dict[str, Any]] = {
    "softmax-dense": {"kernel": "softmax", "execution": "dense"},
    "softmax-fused": {"kernel": "softmax", "execution": "auto"},
    "softmax-blockwise": {"kernel": "softmax", "execution": "blockwise", "block_size": 2},
    "kernel-linear": {"kernel": "kernel_linear"},
    "galerkin": {"kernel": "galerkin"},
}


def _attention(name: str, /, **kwargs: Any) -> MeasureAwareAttention:
    return MeasureAwareAttention(
        source_channels=3,
        query_channels=3,
        out_channels=2,
        num_heads=2,
        head_dim=2,
        **_KERNELS[name],
        **kwargs,
    )


def _attention_data(dtype: Any, /) -> tuple[jax.Array, ...]:
    source = jr.normal(jr.key(1), (2, 5, 3), dtype=dtype)
    query = jr.normal(jr.key(2), (2, 3, 3), dtype=dtype)
    # Case 0: one zero-measure and one masked source, one masked query.
    # Case 1: every source masked, i.e. empty physical support.
    weights = jnp.asarray(
        [[0.1, 0.0, 0.3, 0.4, 0.2], [0.2, 0.2, 0.2, 0.2, 0.2]], dtype=dtype
    )
    source_mask = jnp.asarray([[True, True, False, True, True], [False] * 5])
    query_mask = jnp.asarray([[True, True, False], [True, True, True]])
    return source, query, weights, source_mask, query_mask


def _point(dtype: Any, /) -> jax.Array:
    return jnp.asarray([0.3, -0.2, 0.1], dtype=dtype)


def _attend(attention: Any, data: Any, point: jax.Array, /) -> jax.Array:
    source, query, weights, source_mask, query_mask = data
    return attention(
        source + 0.5 * jnp.broadcast_to(point, source.shape),
        query - jnp.broadcast_to(point, query.shape),
        weights,
        source_mask=source_mask,
        query_mask=query_mask,
    )


def _attention_hessian(attention: Any, data: Any, point: jax.Array, /) -> jax.Array:
    return jax.hessian(lambda value: jnp.sum(_attend(attention, data, value)))(point)


def _attention_loss(data: Any, point: jax.Array, /) -> Callable[[Any], jax.Array]:
    # Second coordinate derivatives inside the loss: the update differentiates
    # third-order chains through measure-weighted attention.
    def loss(attention: Any) -> jax.Array:
        output = _attend(attention, data, point)
        laplacian = jnp.trace(_attention_hessian(attention, data, point))
        return laplacian**2 + jnp.mean(output**2)

    return loss


@pytest.mark.parametrize("name", tuple(_KERNELS))
def test_float32_attention_programs_trace_without_float64(name: str) -> None:
    construction, _, _ = eqx.filter_make_jaxpr(
        lambda key: eqx.filter(_attention(name, dtype=jnp.float32, key=key), eqx.is_array)
    )(jr.key(0))
    assert_no_float64_values(construction)

    attention = _attention(name, dtype=jnp.float32, key=jr.key(11))
    data = _attention_data(jnp.float32)
    point = _point(jnp.float32)
    hessian, _, _ = eqx.filter_make_jaxpr(_attention_hessian)(attention, data, point)
    assert_no_float64_values(hessian)
    parameters, state, fixed = partition_parameters(attention)
    step, _, _ = eqx.filter_make_jaxpr(_train_step)(
        _attention_loss(data, point), parameters, (state, fixed)
    )
    assert_no_float64_values(step)


@pytest.mark.parametrize("name", tuple(_KERNELS))
def test_float32_attention_preserves_measure_masks_and_empty_support(name: str) -> None:
    attention = _attention(name, dtype=jnp.float32, key=jr.key(11))
    data = _attention_data(jnp.float32)
    point = _point(jnp.float32)
    output = _attend(attention, data, point)

    assert output.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(output)))
    assert bool(jnp.any(output[0, :2] != 0.0))
    assert jnp.array_equal(output[0, 2], jnp.zeros((2,), dtype=jnp.float32))
    assert jnp.array_equal(output[1], jnp.zeros((3, 2), dtype=jnp.float32))
    # Zero-measure and masked sources carry no physical support.
    source, *rest = data
    unsupported = source.at[0, 1:3].set(1.0e3)
    assert jnp.array_equal(_attend(attention, (unsupported, *rest), point), output)

    # Empty support has an exactly zero, finite derivative in every parameter.
    empty_gradient = eqx.filter_grad(
        lambda module: jnp.sum(_attend(module, data, point)[1])
    )(attention)
    for leaf in _inexact_leaves(empty_gradient):
        assert leaf.dtype == jnp.float32
        assert not bool(jnp.any(leaf))


@pytest.mark.parametrize("name", tuple(_KERNELS))
def test_float32_attention_hessian_gradient_and_update_match_float64_evaluation(
    name: str,
) -> None:
    attention = _attention(name, dtype=jnp.float32, key=jr.key(11))
    data = _attention_data(jnp.float32)
    point = _point(jnp.float32)
    # The reference evaluates the same float32 parameters and data in float64.
    reference = _upcast(attention)
    reference_data = _upcast(data)
    reference_point = point.astype(jnp.float64)

    output = _attend(attention, data, point)
    reference_output = _attend(reference, reference_data, reference_point)
    assert _relative_error(output, reference_output) <= _FLOAT32_RELATIVE_BOUND
    hessian = _attention_hessian(attention, data, point)
    reference_hessian = _attention_hessian(reference, reference_data, reference_point)
    assert hessian.dtype == jnp.float32
    assert reference_hessian.dtype == jnp.float64
    assert bool(jnp.any(hessian != 0.0))
    assert _relative_error(hessian, reference_hessian) <= _FLOAT32_RELATIVE_BOUND

    parameters, state, fixed = partition_parameters(attention)
    loss, gradient, optimizer_state, updated = eqx.filter_jit(_train_step)(
        _attention_loss(data, point), parameters, (state, fixed)
    )
    reference_parameters, reference_state, reference_fixed = partition_parameters(
        reference
    )
    reference_loss, reference_gradient, _, _ = _train_step(
        _attention_loss(reference_data, reference_point),
        reference_parameters,
        (reference_state, reference_fixed),
    )
    assert loss.dtype == jnp.float32
    assert abs(float(loss) - float(reference_loss)) <= _FLOAT32_RELATIVE_BOUND * abs(
        float(reference_loss)
    )
    assert _relative_error(gradient, reference_gradient) <= _FLOAT32_RELATIVE_BOUND
    _assert_float32_update(gradient, optimizer_state, updated, parameters)


def _saturated_curvature(attention: Any, data: Any, scale: jax.Array, /) -> jax.Array:
    """Exact ``grad(grad(sum))`` form of the query-scale second derivative."""
    source, query, weights, source_mask, query_mask = data

    def total(value: jax.Array) -> jax.Array:
        return jnp.sum(
            attention(
                source,
                query * value,
                weights,
                source_mask=source_mask,
                query_mask=query_mask,
            )
        )

    return jax.grad(jax.grad(total))(scale)


@pytest.mark.parametrize("name", ("softmax-dense", "softmax-fused", "softmax-blockwise"))
def test_float32_saturated_softmax_second_derivative_matches_float64(name: str) -> None:
    # Amplified features saturate the softmax (a float32 probability of one).
    # A softmax custom JVP then cancels O(|d logits|^2) terms and lost about
    # 3e-3 relative accuracy; every route must keep reference accuracy.
    attention = _attention(name, dtype=jnp.float32, key=jr.key(11))
    source, query, *rest = _attention_data(jnp.float32)
    data = (10.0 * source, 10.0 * query, *rest)
    curvature = _saturated_curvature(attention, data, jnp.asarray(1.0, dtype=jnp.float32))
    reference = _saturated_curvature(
        _upcast(attention), _upcast(data), jnp.asarray(1.0, dtype=jnp.float64)
    )
    assert curvature.dtype == jnp.float32
    assert reference != 0.0
    assert abs(float(curvature) - float(reference)) <= (
        _FLOAT32_RELATIVE_BOUND * abs(float(reference))
    )


def test_spectral_storage_budget_counts_the_selected_complex_dtype() -> None:
    # Two signed corners x 2 x 3 channels x 2 x 2 modes = 48 complex weights.
    resources = SpectralConvolutionResourcePolicy(maximum_parameter_bytes=48 * 8)
    selected = SpectralConvND(
        in_channels=2,
        out_channels=3,
        n_modes=(2, 2),
        resources=resources,
        dtype=jnp.float32,
        key=jr.key(3),
    )
    assert selected.weight is not None
    assert selected.weight.dtype == jnp.complex64
    assert selected.weight.size * selected.weight.dtype.itemsize == 48 * 8
    with pytest.raises(ValueError, match="required 768, allowed 384"):
        SpectralConvND(
            in_channels=2,
            out_channels=3,
            n_modes=(2, 2),
            resources=resources,
            key=jr.key(3),
        )


def test_float32_accumulation_dtype_stays_independent_of_parameter_dtype() -> None:
    data = _attention_data(jnp.float32)
    point = _point(jnp.float32)
    widened = _attention(
        "softmax-dense", dtype=jnp.float32, accumulation_dtype="float64", key=jr.key(11)
    )
    narrow = _attention("softmax-dense", dtype=jnp.float32, key=jr.key(11))

    # Same float32 parameters; only the explicit accumulation precision differs.
    for leaf, other in zip(
        _inexact_leaves(widened), _inexact_leaves(narrow), strict=True
    ):
        assert leaf.dtype == other.dtype == jnp.float32
        assert jnp.array_equal(leaf, other)
    widened_output = _attend(widened, data, point)
    assert widened_output.dtype == jnp.float32
    assert (
        _relative_error(widened_output, _attend(narrow, data, point))
        <= _FLOAT32_RELATIVE_BOUND
    )


def _processor_data(dtype: Any, /) -> tuple[jax.Array, ...]:
    values = jr.normal(jr.key(4), (2, 4, 4), dtype=dtype)
    weights = jnp.asarray([[0.25, 0.0, 0.5, 0.25], [0.25] * 4], dtype=dtype)
    # Case 1 has no supported token.
    mask = jnp.asarray([[True, True, True, False], [False] * 4])
    return values, weights, mask


def _process(processor: Any, data: Any, point: jax.Array, /) -> jax.Array:
    values, weights, mask = data
    return processor(values + jnp.broadcast_to(point, values.shape), weights, mask)


def _processor_hessian(processor: Any, data: Any, point: jax.Array, /) -> jax.Array:
    return jax.hessian(lambda value: jnp.sum(_process(processor, data, value)))(point)


def _processor_loss(data: Any, point: jax.Array, /) -> Callable[[Any], jax.Array]:
    def loss(processor: Any) -> jax.Array:
        output = _process(processor, data, point)
        laplacian = jnp.trace(_processor_hessian(processor, data, point))
        return laplacian**2 + jnp.mean(output**2)

    return loss


def test_float32_latent_processor_traces_without_float64_and_matches_float64() -> None:
    processor = LatentTokenProcessor(
        4, 2, num_heads=2, feed_forward_multiplier=2.0, dtype=jnp.float32, key=jr.key(5)
    )
    data = _processor_data(jnp.float32)
    point = jnp.asarray([0.2, -0.1, 0.3, 0.05], dtype=jnp.float32)
    # The shared processor holds raw Equinox norms, so it is split by leaf kind.
    parameters, static = eqx.partition(processor, eqx.is_inexact_array)
    step, _, _ = eqx.filter_make_jaxpr(_train_step)(
        _processor_loss(data, point), parameters, (static,)
    )
    assert_no_float64_values(step)

    output = _process(processor, data, point)
    assert output.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(output)))
    # Masked tokens and the empty-support case stay exactly zero.
    assert jnp.array_equal(output[0, 3], jnp.zeros((4,), dtype=jnp.float32))
    assert jnp.array_equal(output[1], jnp.zeros((4, 4), dtype=jnp.float32))

    reference = _upcast(processor)
    reference_data = _upcast(data)
    reference_point = point.astype(jnp.float64)
    hessian = _processor_hessian(processor, data, point)
    reference_hessian = _processor_hessian(reference, reference_data, reference_point)
    assert hessian.dtype == jnp.float32
    assert bool(jnp.any(hessian != 0.0))
    assert _relative_error(hessian, reference_hessian) <= _FLOAT32_RELATIVE_BOUND

    loss, gradient, optimizer_state, updated = eqx.filter_jit(_train_step)(
        _processor_loss(data, point), parameters, (static,)
    )
    reference_parameters, reference_static = eqx.partition(
        reference, eqx.is_inexact_array
    )
    _, reference_gradient, _, _ = _train_step(
        _processor_loss(reference_data, reference_point),
        reference_parameters,
        (reference_static,),
    )
    assert loss.dtype == jnp.float32
    assert _relative_error(gradient, reference_gradient) <= _FLOAT32_RELATIVE_BOUND
    _assert_float32_update(gradient, optimizer_state, updated, parameters)


def _exact(values: Any, dtype: Any, /) -> jax.Array:
    # Round once to float32 so float32 and float64 batches hold equal values.
    return jnp.asarray(np.asarray(values, dtype=np.float32), dtype=dtype)


def _point_batch(values: jax.Array, dtype: Any, /) -> OperatorBatch:
    return OperatorBatch(
        inputs={
            "u": FunctionSamples(
                values=values,
                coordinates=_exact([[0.0], [0.3], [0.7], [1.0]], dtype),
                quadrature_weights=_exact([0.2, 0.3, 0.0, 0.5], dtype),
                mask=jnp.asarray([True, True, True, False]),
                dtype=dtype,
            )
        },
        queries={
            "query": FunctionSamples(
                values=None,
                coordinates=_exact([[0.1], [0.5], [0.9]], dtype),
                mask=jnp.asarray([True, True, False]),
                dtype=dtype,
            )
        },
    )


def _operator_input(name: str, dtype: Any, scale: Any = 1.0, /) -> Any:
    # The padded source carries a large value that the mask must exclude; the
    # first point is a supported ABUPT anchor, so every model sees the scale.
    batch = _point_batch(scale * _exact([0.4, 0.2, 0.6, 1.0e3], dtype), dtype)
    if name != "InContextOperator":
        return batch
    prompt = OperatorPrompt(
        (
            OperatorSupervisedExample(
                _point_batch(_exact([0.0, 0.3, 0.7, 1.0], dtype), dtype),
                _exact([0.1, 0.5, 0.9], dtype),
            ),
            OperatorSupervisedExample(
                _point_batch(_exact([1.0, 0.7, 0.3, 0.0], dtype), dtype),
                _exact([-0.1, -0.5, -0.9], dtype),
            ),
        )
    )
    return PromptedOperatorBatch(batch, prompt)


def _operator_curvature(model: Any, name: str, scale: jax.Array, /) -> jax.Array:
    """Second derivative of the summed prediction in the source-value scale."""

    def total(value: jax.Array) -> jax.Array:
        return jnp.sum(model(_operator_input(name, value.dtype, value)))

    return jax.grad(jax.grad(total))(scale)


def _operator_loss(inputs: Any, /) -> Callable[[Any], jax.Array]:
    def loss(model: Any) -> jax.Array:
        return jnp.sum(model(inputs) ** 2)

    return loss


@pytest.mark.parametrize("name", _OPERATOR_MODELS)
def test_float32_operator_programs_trace_without_float64(name: str) -> None:
    model = _CONSTRUCTORS[name](jr.key(7), dtype=jnp.float32)
    inputs = _operator_input(name, jnp.float32)
    forward, _, _ = eqx.filter_make_jaxpr(lambda item: item(inputs))(model)
    assert_no_float64_values(forward)
    curvature, _, _ = eqx.filter_make_jaxpr(_operator_curvature)(
        model, name, jnp.asarray(1.0, dtype=jnp.float32)
    )
    assert_no_float64_values(curvature)
    parameters, state, fixed = partition_parameters(model)
    step, _, _ = eqx.filter_make_jaxpr(_train_step)(
        _operator_loss(inputs), parameters, (state, fixed)
    )
    assert_no_float64_values(step)


@pytest.mark.parametrize("name", _OPERATOR_MODELS)
def test_float32_operator_forward_curvature_and_update_match_float64(name: str) -> None:
    model = _CONSTRUCTORS[name](jr.key(7), dtype=jnp.float32)
    inputs = _operator_input(name, jnp.float32)
    # The reference evaluates the same float32 parameters on equal float64 data.
    reference = _upcast(model)
    reference_inputs = _operator_input(name, jnp.float64)

    output = model(inputs)
    reference_output = reference(reference_inputs)
    assert output.shape == (3,)
    assert output.dtype == jnp.float32
    assert reference_output.dtype == jnp.float64
    assert bool(jnp.all(jnp.isfinite(output)))
    assert output[2] == 0.0
    assert _relative_error(output, reference_output) <= _FLOAT32_RELATIVE_BOUND

    curvature = _operator_curvature(model, name, jnp.asarray(1.0, dtype=jnp.float32))
    reference_curvature = _operator_curvature(
        reference, name, jnp.asarray(1.0, dtype=jnp.float64)
    )
    assert curvature.dtype == jnp.float32
    assert reference_curvature != 0.0
    assert abs(float(curvature) - float(reference_curvature)) <= (
        _FLOAT32_RELATIVE_BOUND * abs(float(reference_curvature))
    )

    parameters, state, fixed = partition_parameters(model)
    loss, gradient, optimizer_state, updated = eqx.filter_jit(_train_step)(
        _operator_loss(inputs), parameters, (state, fixed)
    )
    reference_parameters, reference_state, reference_fixed = partition_parameters(
        reference
    )
    reference_loss, reference_gradient, _, _ = _train_step(
        _operator_loss(reference_inputs),
        reference_parameters,
        (reference_state, reference_fixed),
    )
    assert loss.dtype == jnp.float32
    assert abs(float(loss) - float(reference_loss)) <= _FLOAT32_RELATIVE_BOUND * abs(
        float(reference_loss)
    )
    assert _relative_error(gradient, reference_gradient) <= _FLOAT32_RELATIVE_BOUND
    _assert_float32_update(gradient, optimizer_state, updated, parameters)
    trained = combine_parameters(updated, state, fixed)(inputs)
    assert trained.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(trained)))
