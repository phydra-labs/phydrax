#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel, FrozenModel
from phydrax._trainable import partition_trainable
from phydrax.nn.operator.architectures.conditioning._function_frame import (
    FUNCTION_PROJECTION_INSUFFICIENT_SUPPORT,
    FUNCTION_PROJECTION_INVALID_MEASURE,
    FUNCTION_PROJECTION_NONFINITE,
    FUNCTION_PROJECTION_REGULARIZED,
    FUNCTION_PROJECTION_SUCCESS,
    FunctionFrameReconstructor,
    FunctionProjectionPolicy,
    LearnedFunctionFrame,
)


class _PolynomialBasis(AbstractArrayModel):
    coefficients: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    channels: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)

    def __init__(self, coefficients):
        values = jnp.asarray(coefficients)
        if values.ndim != 3:
            raise ValueError("coefficients must have shape (channels, rank, powers).")
        self.coefficients = values
        self.in_size = 1
        self.channels = int(values.shape[0])
        self.rank = int(values.shape[1])
        self.out_size = self.channels * self.rank

    def __call__(self, coordinate, /, *, key=None):
        del key
        powers = jnp.asarray(coordinate[0]) ** jnp.arange(self.coefficients.shape[-1])
        return jnp.einsum("crp,p->cr", self.coefficients, powers).reshape(-1)


class _PolynomialOutput(AbstractArrayModel):
    coefficients: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, coefficients):
        values = jnp.asarray(coefficients)
        if values.ndim != 2:
            raise ValueError("coefficients must have shape (channels, powers).")
        self.coefficients = values
        self.in_size = 1
        self.out_size = int(values.shape[0])

    def __call__(self, coordinate, /, *, key=None):
        del key
        powers = jnp.asarray(coordinate[0]) ** jnp.arange(self.coefficients.shape[-1])
        return self.coefficients @ powers


class _LinearMap(AbstractArrayModel):
    matrix: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, matrix):
        values = jnp.asarray(matrix)
        self.matrix = values
        self.in_size = int(values.shape[1])
        self.out_size = int(values.shape[0])

    def __call__(self, coefficients, /, *, key=None):
        del key
        return self.matrix @ coefficients


def _scalar_frame(*, frame_id="scalar", offset_model=None, coefficients=None):
    basis_coefficients = (
        jnp.eye(3)[None, ...] if coefficients is None else jnp.asarray(coefficients)
    )
    return LearnedFunctionFrame(
        basis_model=_PolynomialBasis(basis_coefficients),
        offset_model=offset_model,
        rank=int(basis_coefficients.shape[1]),
        coord_dim=1,
        out_size="scalar",
        frame_id=frame_id,
    )


def _vector_frame(*, frame_id="vector", coefficients=None):
    basis_coefficients = (
        jnp.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            ]
        )
        if coefficients is None
        else jnp.asarray(coefficients)
    )
    return LearnedFunctionFrame(
        basis_model=_PolynomialBasis(basis_coefficients),
        rank=int(basis_coefficients.shape[1]),
        coord_dim=1,
        out_size=int(basis_coefficients.shape[0]),
        frame_id=frame_id,
    )


def _samples(coordinates, values=None, *, weights=None, mask=None):
    points = jnp.asarray(coordinates)
    if points.ndim == 1:
        points = points[..., None]
    return phx.nn.operator.FunctionSamples(
        values=values,
        coordinates=points,
        quadrature_weights=weights,
        mask=mask,
    )


def _projection_samples(frame, coefficients, coordinates, *, weights=None):
    query = _samples(coordinates, weights=weights)
    values = frame.decode(jnp.asarray(coefficients), query)
    return _samples(coordinates, values, weights=weights)


def test_function_frame_constructor_enforces_exact_model_contracts():
    with pytest.raises(ValueError, match="rank and coord_dim"):
        LearnedFunctionFrame(
            basis_model=_PolynomialBasis(jnp.ones((1, 1, 1))),
            rank=0,
            coord_dim=1,
            frame_id="invalid",
        )
    with pytest.raises(ValueError, match="in_size"):
        LearnedFunctionFrame(
            basis_model=_PolynomialBasis(jnp.ones((1, 1, 1))),
            rank=1,
            coord_dim=2,
            frame_id="invalid",
        )
    with pytest.raises(ValueError, match=r"rank\*out_size"):
        LearnedFunctionFrame(
            basis_model=_PolynomialBasis(jnp.ones((1, 2, 1))),
            rank=3,
            coord_dim=1,
            frame_id="invalid",
        )
    with pytest.raises(ValueError, match="offset_model.out_size"):
        LearnedFunctionFrame(
            basis_model=_PolynomialBasis(jnp.ones((1, 1, 1))),
            offset_model=_PolynomialOutput(jnp.ones((2, 1))),
            rank=1,
            coord_dim=1,
            frame_id="invalid",
        )
    with pytest.raises(ValueError, match="frame_id"):
        _scalar_frame(frame_id="")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"ridge": -1.0}, "ridge"),
        ({"rcond": -1.0}, "rcond"),
        ({"min_samples": 0}, "min_samples"),
        ({"rank_policy": "unknown"}, "rank_policy"),
        ({"rank_policy": "regularized"}, "positive ridge"),
    ),
)
def test_projection_policy_rejects_invalid_scalar_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FunctionProjectionPolicy(**kwargs)


@pytest.mark.parametrize(
    "metric",
    (
        jnp.ones((2, 3)),
        jnp.asarray([[1.0, 2.0], [0.0, 1.0]]),
        jnp.asarray([[1.0, 1.0], [1.0, 1.0]]),
        jnp.asarray([[1.0, jnp.nan], [jnp.nan, 1.0]]),
    ),
)
def test_projection_policy_rejects_invalid_channel_metrics(metric):
    with pytest.raises((TypeError, ValueError), match="channel_metric"):
        FunctionProjectionPolicy(channel_metric=metric)


def test_scalar_frame_recovers_coefficients_and_decodes_arbitrary_queries():
    frame = _scalar_frame()
    coordinates = jnp.asarray([0.0, 0.13, 0.31, 0.52, 0.74, 0.91, 1.0])
    coefficients = jnp.asarray([1.25, -0.75, 0.4])
    samples = _projection_samples(
        frame,
        coefficients,
        coordinates,
        weights=jnp.asarray([0.1, 0.2, 0.15, 0.05, 0.18, 0.12, 0.2]),
    )

    report = frame.project(samples, policy=FunctionProjectionPolicy(rcond=1e-12))
    query_coordinates = jnp.asarray([0.07, 0.43, 0.88])
    decoded = frame.decode(report.require_coefficients(), _samples(query_coordinates))
    expected = (
        coefficients[0]
        + coefficients[1] * query_coordinates
        + coefficients[2] * query_coordinates**2
    )

    np.testing.assert_allclose(report.coefficients, coefficients, atol=2e-11)
    np.testing.assert_allclose(decoded, expected, atol=2e-11)
    assert int(report.status) == FUNCTION_PROJECTION_SUCCESS
    assert int(report.rank) == frame.rank
    assert bool(report.solved)
    assert bool(report.identified)
    assert bool(report.valid)
    assert float(report.relative_residual) < 1e-11


def test_scalar_frame_accepts_an_explicit_singleton_channel_axis():
    frame = _scalar_frame()
    coordinates = jnp.linspace(0.0, 1.0, 8)
    implicit = _projection_samples(frame, jnp.asarray([0.5, -1.0, 2.0]), coordinates)
    explicit = _samples(coordinates, implicit.values[..., None])

    implicit_report = frame.project(implicit)
    explicit_report = frame.project(explicit)

    np.testing.assert_allclose(explicit_report.coefficients, implicit_report.coefficients)


def test_vector_frame_uses_one_coefficient_vector_across_channels():
    frame = _vector_frame()
    coordinates = jnp.asarray([0.0, 0.15, 0.4, 0.8, 1.0])
    coefficients = jnp.asarray([1.5, -0.25])
    samples = _projection_samples(frame, coefficients, coordinates)

    report = frame.project(samples)

    np.testing.assert_allclose(report.coefficients, coefficients, atol=2e-11)
    assert report.coefficients.shape == (frame.rank,)
    assert int(report.sample_count) == coordinates.size


def test_channel_metric_changes_the_best_approximation_without_changing_shape():
    constant_vector_frame = _vector_frame(coefficients=jnp.asarray([[[1.0]], [[1.0]]]))
    coordinates = jnp.linspace(0.0, 1.0, 5)
    values = jnp.broadcast_to(jnp.asarray([1.0, 3.0]), (5, 2))
    samples = _samples(coordinates, values)

    euclidean = constant_vector_frame.project(samples)
    weighted = constant_vector_frame.project(
        samples,
        policy=FunctionProjectionPolicy(channel_metric=jnp.diag(jnp.asarray([4.0, 1.0]))),
    )

    np.testing.assert_allclose(euclidean.coefficients, jnp.asarray([2.0]))
    np.testing.assert_allclose(weighted.coefficients, jnp.asarray([1.4]))
    assert euclidean.coefficients.shape == weighted.coefficients.shape == (1,)


def test_complex_frame_and_hermitian_metric_recover_complex_coefficients():
    frame = _vector_frame(
        coefficients=jnp.asarray(
            [
                [[1.0 + 0.0j, 0.25j], [0.0 + 0.0j, 1.0 + 0.5j]],
                [[0.0 + 1.0j, 0.5], [1.0 - 0.25j, -0.5j]],
            ]
        )
    )
    metric = jnp.asarray([[2.0, 0.25j], [-0.25j, 1.5]])
    coefficients = jnp.asarray([1.0 + 0.5j, -0.75 + 0.2j])
    coordinates = jnp.linspace(0.0, 1.0, 7)
    samples = _projection_samples(frame, coefficients, coordinates)

    report = frame.project(
        samples,
        policy=FunctionProjectionPolicy(channel_metric=metric, rcond=1e-12),
    )

    np.testing.assert_allclose(report.coefficients, coefficients, atol=2e-11)
    assert jnp.all(jnp.isfinite(report.coefficients))


def test_frame_offset_is_subtracted_for_projection_and_restored_for_decoding():
    offset_model = _PolynomialOutput(jnp.asarray([[2.0, -0.5]]))
    frame = _scalar_frame(offset_model=offset_model)
    coordinates = jnp.linspace(0.0, 1.0, 8)
    support = _samples(coordinates)
    offset_values = frame.evaluate_offset(support)[..., 0]
    samples = _samples(coordinates, offset_values)

    report = frame.project(samples)
    query_coordinates = jnp.asarray([0.2, 0.6, 0.9])
    decoded = frame.decode(report.coefficients, _samples(query_coordinates))

    np.testing.assert_allclose(report.coefficients, 0.0, atol=2e-11)
    np.testing.assert_allclose(decoded, 2.0 - 0.5 * query_coordinates, atol=2e-11)


def test_projection_is_invariant_to_point_permutation_and_weight_rescaling():
    frame = _scalar_frame()
    coordinates = jnp.asarray([0.0, 0.2, 0.45, 0.7, 1.0])
    values = 0.7 - 0.3 * coordinates + 0.5 * coordinates**2 + 0.02 * coordinates**3
    weights = jnp.asarray([0.1, 0.2, 0.15, 0.25, 0.3])
    samples = _samples(coordinates, values, weights=weights)
    permutation = jnp.asarray([3, 1, 4, 0, 2])

    reference = frame.project(samples)
    permuted = frame.project(
        _samples(
            coordinates[permutation],
            values[permutation],
            weights=weights[permutation],
        )
    )
    rescaled = frame.project(_samples(coordinates, values, weights=1e-4 * weights))

    np.testing.assert_allclose(permuted.coefficients, reference.coefficients, atol=2e-11)
    np.testing.assert_allclose(rescaled.coefficients, reference.coefficients, atol=2e-11)


def test_projection_is_invariant_to_equivalent_quadrature_refinement():
    frame = _scalar_frame()
    coordinates = jnp.asarray([0.0, 0.3, 0.65, 1.0])
    values = 0.8 + coordinates - 0.4 * coordinates**2 + 0.1 * coordinates**3
    weights = jnp.asarray([0.15, 0.25, 0.35, 0.25])
    reference = frame.project(_samples(coordinates, values, weights=weights))
    refined = frame.project(
        _samples(
            jnp.repeat(coordinates, 2),
            jnp.repeat(values, 2),
            weights=jnp.repeat(weights / 2.0, 2),
        )
    )

    np.testing.assert_allclose(refined.coefficients, reference.coefficients, atol=2e-11)


def test_masked_padding_with_nan_payloads_does_not_contaminate_projection():
    frame = _scalar_frame()
    coordinates = jnp.asarray([0.0, 0.3, 0.7, 1.0])
    values = 1.0 + 2.0 * coordinates - coordinates**2
    reference = frame.project(_samples(coordinates, values))
    padded_coordinates = jnp.concatenate((coordinates, jnp.asarray([jnp.nan])))
    padded_values = jnp.concatenate((values, jnp.asarray([jnp.nan])))
    padded_weights = jnp.concatenate((jnp.ones_like(coordinates), jnp.asarray([jnp.nan])))
    mask = jnp.asarray([True, True, True, True, False])

    padded = frame.project(
        _samples(
            padded_coordinates,
            padded_values,
            weights=padded_weights,
            mask=mask,
        )
    )

    np.testing.assert_allclose(padded.coefficients, reference.coefficients, atol=2e-11)
    assert bool(padded.valid)


def test_case_batched_projection_matches_independent_case_solves():
    frame = _scalar_frame()
    coordinates = jnp.asarray(
        [
            [[0.0], [0.2], [0.5], [0.8], [1.0]],
            [[0.0], [0.1], [0.4], [0.75], [0.95]],
        ]
    )
    coefficients = jnp.asarray([[1.0, -0.5, 0.2], [-0.25, 0.75, 1.1]])
    query = _samples(coordinates, weights=jnp.ones((2, 5)))
    values = frame.decode(coefficients, query, case_shape=(2,))
    samples = _samples(coordinates, values, weights=jnp.ones((2, 5)))

    batched = frame.project(samples, case_shape=(2,))
    independent = jnp.stack(
        tuple(
            frame.project(
                _samples(coordinates[index], values[index], weights=jnp.ones(5))
            ).coefficients
            for index in range(2)
        )
    )

    np.testing.assert_allclose(batched.coefficients, independent, atol=2e-11)
    assert batched.case_shape == (2,)
    assert batched.status.shape == (2,)


def test_projection_reports_insufficient_support_and_blocks_coefficients():
    frame = _scalar_frame()
    coordinates = jnp.linspace(0.0, 1.0, 4)
    all_masked = frame.project(
        _samples(
            coordinates,
            jnp.ones(4),
            mask=jnp.zeros(4, dtype=bool),
        )
    )
    too_few = frame.project(_samples(coordinates[:2], jnp.ones(2)))

    for report in (all_masked, too_few):
        assert int(report.status) == FUNCTION_PROJECTION_INSUFFICIENT_SUPPORT
        assert not bool(report.valid)
        with pytest.raises(eqx.EquinoxRuntimeError, match="invalid coefficients"):
            jax.block_until_ready(report.require_coefficients())


def test_rank_deficiency_errors_or_reports_regularized_nonidentification():
    frame = _scalar_frame(
        coefficients=jnp.asarray([[[1.0], [1.0]]]),
        frame_id="dependent",
    )
    samples = _samples(jnp.linspace(0.0, 1.0, 5), jnp.ones(5))

    with pytest.raises(eqx.EquinoxRuntimeError, match="do not identify"):
        report = frame.project(samples)
        jax.block_until_ready(report.coefficients)

    regularized = frame.project(
        samples,
        policy=FunctionProjectionPolicy(
            ridge=1e-3,
            rank_policy="regularized",
        ),
    )

    assert int(regularized.status) == FUNCTION_PROJECTION_REGULARIZED
    assert bool(regularized.solved)
    assert not bool(regularized.identified)
    assert bool(regularized.valid)
    assert int(regularized.rank) == 1
    assert jnp.all(jnp.isfinite(regularized.coefficients))


@pytest.mark.parametrize(
    "weights",
    (
        jnp.asarray([1.0, -1.0, 1.0]),
        jnp.asarray([1.0, jnp.nan, 1.0]),
    ),
)
def test_requested_invalid_quadrature_is_reported(weights):
    frame = _scalar_frame(coefficients=jnp.asarray([[[1.0]]]))
    report = frame.project(
        _samples(jnp.asarray([0.0, 0.5, 1.0]), jnp.ones(3), weights=weights)
    )

    assert int(report.status) == FUNCTION_PROJECTION_INVALID_MEASURE
    assert not bool(report.valid)


def test_nonfinite_active_values_and_frame_outputs_are_reported():
    constant = _scalar_frame(coefficients=jnp.asarray([[[1.0]]]))
    target_report = constant.project(
        _samples(
            jnp.asarray([0.0, 0.5, 1.0]),
            jnp.asarray([1.0, jnp.nan, 1.0]),
        )
    )
    nonfinite_frame = _scalar_frame(
        coefficients=jnp.asarray([[[jnp.nan]]]),
        frame_id="nonfinite",
    )
    frame_report = nonfinite_frame.project(
        _samples(jnp.asarray([0.0, 0.5, 1.0]), jnp.ones(3))
    )

    assert int(target_report.status) == FUNCTION_PROJECTION_NONFINITE
    assert int(frame_report.status) == FUNCTION_PROJECTION_NONFINITE
    assert not bool(target_report.valid)
    assert not bool(frame_report.valid)


def test_physical_quadrature_requirement_rejects_counting_measure():
    frame = _scalar_frame(coefficients=jnp.asarray([[[1.0]]]))
    report = frame.project(
        _samples(jnp.linspace(0.0, 1.0, 4), jnp.ones(4)),
        policy=FunctionProjectionPolicy(require_physical_quadrature=True),
    )

    assert int(report.status) == FUNCTION_PROJECTION_INVALID_MEASURE
    assert not bool(report.valid)


def test_projection_rejects_ambiguous_value_and_coordinate_shapes():
    vector_frame = _vector_frame()
    coordinates = jnp.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="explicit channel axis"):
        vector_frame.project(_samples(coordinates, jnp.ones(5)))
    with pytest.raises(ValueError, match="must have shape"):
        vector_frame.project(_samples(coordinates, jnp.ones((5, 2, 1))))
    with pytest.raises(ValueError, match="coordinate dimension"):
        vector_frame.project(
            _samples(jnp.stack((coordinates, coordinates), axis=-1), jnp.ones((5, 2)))
        )


def test_projection_is_jittable_and_differentiable_in_values_and_frame_parameters():
    frame = _scalar_frame()
    coordinates = jnp.linspace(0.0, 1.0, 9)
    values = 0.5 - coordinates + 0.75 * coordinates**2 + 0.05 * coordinates**3
    policy = FunctionProjectionPolicy(rcond=1e-12)

    project_values = jax.jit(
        lambda observed: (
            frame.project(
                _samples(coordinates, observed),
                policy=policy,
            ).coefficients
        )
    )
    coefficients = project_values(values)
    value_gradient = jax.grad(lambda observed: jnp.sum(project_values(observed) ** 2))(
        values
    )
    frame_gradient = eqx.filter_grad(
        lambda candidate: jnp.sum(
            candidate.project(
                _samples(coordinates, values),
                policy=policy,
            ).coefficients
            ** 2
        )
    )(frame)

    assert jnp.all(jnp.isfinite(coefficients))
    assert jnp.all(jnp.isfinite(value_gradient))
    assert jnp.all(jnp.isfinite(frame_gradient.basis_model.coefficients))


def test_encoded_reconstructor_reuses_state_across_independent_queries():
    frame = _scalar_frame(frame_id="identity")
    support_coordinates = jnp.asarray([0.0, 0.17, 0.39, 0.62, 0.84, 1.0])
    coefficients = jnp.asarray([1.0, -0.5, 0.25])
    support = _projection_samples(frame, coefficients, support_coordinates)
    query_a = _samples(jnp.asarray([0.1, 0.3, 0.9]))
    query_b = _samples(jnp.asarray([0.05, 0.55, 0.75, 0.95]))
    batch = phx.nn.operator.OperatorBatch(
        inputs={"source": support},
        queries={"query": query_a},
    )
    model = FunctionFrameReconstructor(source_frame=frame, target_frame=frame)

    state = model.encode_inputs(batch)
    decoded_a = model.decode_query(state, query_a)
    decoded_b = model.decode_query(state, query_b)

    np.testing.assert_allclose(decoded_a, frame.decode(coefficients, query_a), atol=2e-11)
    np.testing.assert_allclose(decoded_b, frame.decode(coefficients, query_b), atol=2e-11)
    np.testing.assert_allclose(model(batch), decoded_a, atol=2e-11)
    assert model.operator.bias is None
    assert bool(state.report.identified)


def test_reconstructor_maps_between_frames_with_different_ranks():
    source = _scalar_frame(frame_id="source")
    target = LearnedFunctionFrame(
        basis_model=_PolynomialBasis(jnp.eye(2)[None, ...]),
        rank=2,
        coord_dim=1,
        frame_id="target",
    )
    coefficient_map = _LinearMap(jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    model = FunctionFrameReconstructor(
        source_frame=source,
        target_frame=target,
        coefficient_map=coefficient_map,
    )
    source_coefficients = jnp.asarray([0.4, -1.25, 0.75])
    support = _projection_samples(
        source,
        source_coefficients,
        jnp.linspace(0.0, 1.0, 7),
    )
    query_coordinates = jnp.asarray([0.15, 0.45, 0.85])
    batch = phx.nn.operator.OperatorBatch(
        inputs={"source": support},
        queries={"query": _samples(query_coordinates)},
    )

    output = model(batch)

    np.testing.assert_allclose(
        output,
        source_coefficients[0] + source_coefficients[1] * query_coordinates,
        atol=2e-11,
    )


def test_bias_free_identity_reconstructor_obeys_superposition():
    frame = _scalar_frame(frame_id="linear")
    model = FunctionFrameReconstructor(source_frame=frame, target_frame=frame)
    coordinates = jnp.linspace(0.0, 1.0, 8)
    query = _samples(jnp.asarray([0.1, 0.4, 0.9]))

    def evaluate(coefficients):
        batch = phx.nn.operator.OperatorBatch(
            inputs={"source": _projection_samples(frame, coefficients, coordinates)},
            queries={"query": query},
        )
        return model(batch)

    first = jnp.asarray([1.0, -0.5, 0.25])
    second = jnp.asarray([-0.3, 0.2, 0.8])

    np.testing.assert_allclose(
        evaluate(first + second),
        evaluate(first) + evaluate(second),
        atol=3e-11,
    )


def test_frozen_reconstructor_removes_frame_and_map_arrays_from_training_partition():
    source = _scalar_frame(frame_id="frozen-source")
    target = LearnedFunctionFrame(
        basis_model=_PolynomialBasis(jnp.eye(2)[None, ...]),
        rank=2,
        coord_dim=1,
        frame_id="frozen-target",
    )
    model = FunctionFrameReconstructor(
        source_frame=source,
        target_frame=target,
        coefficient_map=_LinearMap(jnp.ones((2, 3))),
    ).frozen()

    parameters, _ = partition_trainable(model)

    assert isinstance(model.source_frame.basis_model, FrozenModel)
    assert isinstance(model.target_frame.basis_model, FrozenModel)
    assert isinstance(model.coefficient_map, FrozenModel)
    assert jax.tree.leaves(parameters) == []


def test_end_to_end_reconstruction_has_finite_frame_gradients():
    frame = _scalar_frame(frame_id="gradient")
    model = FunctionFrameReconstructor(source_frame=frame, target_frame=frame)
    support_coordinates = jnp.linspace(0.0, 1.0, 8)
    source_coefficients = jnp.asarray([0.75, -0.4, 0.9])
    query_coordinates = jnp.asarray([0.05, 0.25, 0.55, 0.95])
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "source": _projection_samples(
                frame,
                source_coefficients,
                support_coordinates,
            )
        },
        queries={"query": _samples(query_coordinates)},
    )
    target = 0.9 + 0.1 * query_coordinates

    gradient = eqx.filter_grad(
        lambda candidate: jnp.mean((candidate(batch) - target) ** 2)
    )(model)
    leaves = jax.tree.leaves(eqx.filter(gradient, eqx.is_inexact_array))

    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
