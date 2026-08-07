#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import math

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.constraints import FunctionalConstraint
from phydrax.domain import Interval1d, LegendreAxisSpec, SampleLayout
from phydrax.equations import (
    compile_pde_functional_constraint,
    compile_pde_problem,
    pad_pde_tokens,
    pde_ir_from_json,
    pde_ir_hash,
    pde_ir_to_json,
    PDECoordinate,
    PDEEquation,
    PDEExpression,
    PDEField,
    PDEProblemIR,
    tokenize_pde_ir,
)
from phydrax.nn import (
    AnchorQuerySamplingPolicy,
    ArrayOperatorQuerySource,
    ArrayPredictionSink,
    CallbackOperatorCaseSource,
    decode_query_chunks,
    OperatorCase,
    OperatorCaseMetadata,
    read_operator_case_batch,
    take_function_samples,
    take_query_targets,
)


def test_operator_training_substrate_is_reexported_from_nn():
    training_exports = set(phx.nn.operator_training.__all__)
    assert training_exports <= set(phx.nn.__all__)
    assert all(
        getattr(phx.nn, name) is getattr(phx.nn.operator_training, name)
        for name in training_exports
    )


def _point_samples(coordinates, *, values=None, weights=None, mask=None):
    return phx.nn.FunctionSamples(
        values=None if values is None else jnp.asarray(values, dtype=float),
        coordinates=jnp.asarray(coordinates, dtype=float),
        quadrature_weights=None if weights is None else jnp.asarray(weights, dtype=float),
        mask=None if mask is None else jnp.asarray(mask, dtype=bool),
    )


def _multi_query_case(
    source_coordinates,
    source_values,
    source_weights,
    state_coordinates,
    state_weights,
    flux_coordinates,
    flux_weights,
    flux_mask,
):
    return phx.nn.OperatorBatch(
        inputs={
            "u": _point_samples(
                source_coordinates,
                values=source_values,
                weights=source_weights,
            )
        },
        queries={
            "state": _point_samples(state_coordinates, weights=state_weights),
            "flux": _point_samples(
                flux_coordinates,
                weights=flux_weights,
                mask=flux_mask,
            ),
        },
    )


def test_multi_query_batches_stack_and_preserve_per_query_metadata():
    first = _multi_query_case(
        [[0.0], [0.5], [1.0]],
        [1.0, 2.0, 3.0],
        [0.2, 0.3, 0.5],
        [[0.0], [1.0]],
        [0.4, 0.6],
        [[0.0], [0.5], [1.0]],
        [0.2, 0.3, 0.5],
        [True, False, True],
    )
    second = _multi_query_case(
        [[0.0], [1.0]],
        [4.0, 5.0],
        [0.25, 0.75],
        [[0.0], [0.5], [1.0]],
        [0.2, 0.3, 0.5],
        [[0.25]],
        [1.0],
        [True],
    )

    batch = phx.nn.stack_operator_batches((first, second), case_axis="scenario")

    assert batch.case_axes == ("scenario",)
    assert batch.case_shape == (2,)
    assert tuple(batch.queries) == ("state", "flux")
    assert batch.query("state") is batch.queries["state"]
    assert jnp.array_equal(
        batch.query("state").mask,
        jnp.array([[True, True, False], [True, True, True]]),
    )
    assert jnp.allclose(
        batch.query("state").quadrature(case_shape=(2,)),
        jnp.array([[0.4, 0.6, 0.0], [0.2, 0.3, 0.5]]),
    )
    assert jnp.array_equal(
        batch.query("flux").mask,
        jnp.array([[True, False, True], [True, False, False]]),
    )
    selected = batch.take(1, axis="scenario")
    assert selected.case_axes == ()
    assert selected.case_shape == ()
    assert jnp.array_equal(selected.query("flux").mask, jnp.array([True, False, False]))

    state_spec = phx.nn.OperatorOutputSpec()
    flux_spec = phx.nn.OperatorOutputSpec(2, component_names=("x", "y"))
    prediction = phx.nn.OperatorPrediction(
        {
            "state": phx.nn.OperatorFieldBatch(
                jnp.zeros((2, 3)),
                query_name="state",
                spec=state_spec,
            ),
            "flux": phx.nn.OperatorFieldBatch(
                jnp.zeros((2, 3, 2)),
                query_name="flux",
                spec=flux_spec,
            ),
        },
        batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    assert prediction.queries == batch.queries
    assert prediction.field("flux").spec.component_names == ("x", "y")
    assert prediction.field("state").values.shape == (2, 3)
    assert prediction.field("flux").values.shape == (2, 3, 2)


def _context_fixture():
    values = jnp.arange(16.0).reshape((2, 4, 2))
    coordinates = jnp.array(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[10.0], [11.0], [12.0], [13.0]],
        ]
    )
    weights = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    mask = jnp.array(
        [[True, True, True, False], [True, True, False, False]]
    )
    samples = phx.nn.FunctionSamples(
        values=values,
        coordinates=coordinates,
        quadrature_weights=weights,
        mask=mask,
    )
    return values, samples


def test_context_strategies_have_stable_fingerprints_and_physical_state():
    values, samples = _context_fixture()
    learned_strategy = phx.nn.LearnedTokenContext(
        channels=2, num_tokens=3, key=jr.key(0)
    )
    pooled_strategy = phx.nn.PooledGeometryContext(channels=2, num_tokens=2)
    anchor_strategy = phx.nn.SampledAnchorContext(channels=2, num_anchors=2)

    learned = learned_strategy(values, samples, normalization_id="unit-scale")
    learned_again = learned_strategy(values + 1000.0, samples, normalization_id="unit-scale")
    pooled = pooled_strategy(values, samples, normalization_id="unit-scale")
    anchors = anchor_strategy(
        values,
        samples,
        indices=jnp.array([1, 3]),
        normalization_id="unit-scale",
    )

    assert learned.kind == "learned"
    assert learned.case_shape == (2,)
    assert learned.values.shape == (2, 3, 2)
    assert learned.coordinates is None
    assert jnp.array_equal(learned.weights, jnp.ones((2, 3)))
    assert learned.schema_fingerprint == learned_again.schema_fingerprint
    assert learned.schema_fingerprint == pooled.schema_fingerprint
    assert learned.schema_fingerprint == anchors.schema_fingerprint
    assert learned.schema_fingerprint != learned_strategy(
        values, samples, normalization_id="other-scale"
    ).schema_fingerprint

    assert pooled.kind == "pooled_geometry"
    assert jnp.allclose(pooled.weights, jnp.array([[0.3, 0.3], [0.7, 0.0]]))
    assert jnp.array_equal(
        pooled.mask, jnp.array([[True, True], [True, False]])
    )
    assert jnp.allclose(
        pooled.coordinates[..., 0],
        jnp.array([[2.0 / 3.0, 2.0], [7.3 / 0.7, 0.0]]),
    )
    assert jnp.array_equal(anchors.mask, jnp.array([[True, False], [True, False]]))
    assert jnp.allclose(anchors.weights, jnp.array([[0.2, 0.4], [0.3, 0.1]]))
    assert jnp.array_equal(anchors.values, values[:, (1, 3), :])

    compiled = eqx.filter_jit(
        lambda strategy, current: strategy(
            current, samples, normalization_id="unit-scale"
        )
    )(pooled_strategy, values)
    assert jnp.allclose(compiled.values, pooled.values)
    assert jnp.array_equal(compiled.mask, pooled.mask)


def test_lazy_case_sampling_reads_only_selected_cases_and_preserves_metadata():
    metadata_reads = []
    case_reads = []
    coordinates = jnp.arange(4.0)[:, None]
    weights = jnp.array([0.1, 0.2, 0.3, 0.4])

    def metadata_reader(index):
        metadata_reads.append(index)
        geometry = _point_samples(coordinates, weights=weights)
        return OperatorCaseMetadata(inputs={"u": geometry}, queries={"query": geometry})

    def case_reader(index, request):
        case_reads.append(index)
        source = _point_samples(
            coordinates,
            values=index + jnp.arange(4.0),
            weights=weights,
        )
        query = _point_samples(coordinates, weights=weights)
        targets = 10.0 * index + jnp.arange(4.0)
        if request is not None:
            source = take_function_samples(source, request.input_selections["u"])
            selection = request.query_selections["query"]
            targets = take_query_targets(targets, query.sample_shape, selection)
            query = take_function_samples(query, selection)
        batch = phx.nn.OperatorBatch(
            inputs={"u": source},
            queries={"query": query},
        )
        return OperatorCase(
            batch,
            phx.nn.OperatorTargetBatch.from_arrays(
                {"solution": targets},
                batch,
            ),
        )

    source = CallbackOperatorCaseSource(
        100,
        metadata_reader=metadata_reader,
        case_reader=case_reader,
    )
    policy = AnchorQuerySamplingPolicy(
        anchor_counts={"u": 2},
        query_counts={"query": 2},
        strategy="fixed_indices",
        query_strategy="fixed_indices",
        fixed_anchor_indices={"u": (0, 2)},
        fixed_query_indices={"query": (1, 3)},
    )
    dataset = read_operator_case_batch(
        source,
        (7, 2),
        sampling=policy,
        split="validation",
        epoch=3,
        case_axis="scenario",
    )

    assert metadata_reads == [7, 2]
    assert case_reads == [7, 2]
    assert dataset.batch.case_axes == ("scenario",)
    assert dataset.batch.case_shape == (2,)
    assert jnp.array_equal(
        dataset.batch.input("u").values,
        jnp.array([[7.0, 9.0], [2.0, 4.0]]),
    )
    assert jnp.array_equal(
        dataset.targets.field("solution").values,
        jnp.array([[71.0, 73.0], [21.0, 23.0]]),
    )
    assert jnp.array_equal(dataset.batch.input("u").mask, jnp.ones((2, 2), dtype=bool))
    assert jnp.allclose(
        dataset.batch.input("u").quadrature(case_shape=(2,)),
        jnp.array([[0.25, 0.75], [0.25, 0.75]]),
    )
    assert jnp.allclose(
        dataset.batch.query("query").quadrature(case_shape=(2,)),
        jnp.array([[1.0 / 3.0, 2.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]]),
    )


def _tiny_encoded_operator_and_batch():
    source_coordinates = jnp.array(
        [[[0.0], [0.5], [1.0]], [[0.0], [0.25], [1.0]]]
    )
    query_coordinates = jnp.array(
        [
            [[0.0], [0.25], [0.5], [0.75], [1.0]],
            [[0.1], [0.3], [0.5], [0.7], [0.9]],
        ]
    )
    source = phx.nn.FunctionSamples(
        values=jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        coordinates=source_coordinates,
        quadrature_weights=jnp.array([[0.2, 0.3, 0.5], [0.4, 0.2, 0.4]]),
        mask=jnp.array([[True, True, True], [True, False, True]]),
    )
    query = phx.nn.FunctionSamples(
        values=None,
        coordinates=query_coordinates,
        quadrature_weights=jnp.full((2, 5), 0.2),
        mask=jnp.array(
            [[True, True, False, True, True], [True, False, True, True, True]]
        ),
    )
    batch = phx.nn.OperatorBatch(inputs={"u": source}, queries={"query": query}, case_axes=("scenario",),)
    feature = phx.nn.MLP(
        in_size=2,
        out_size=1,
        hidden_sizes=(),
        key=jr.key(10),
    )
    branch = phx.nn.IntegralBranchEncoder(
        feature_model=feature,
        latent_size=1,
        coord_dim=1,
    )
    decoder = phx.nn.FiLMCoordinateDecoder(
        latent_size=1,
        coord_dim=1,
        width=2,
        depth=1,
        key=jr.key(11),
    )
    model = phx.nn.CoordinateConditionedOperator(
        branch=branch,
        decoder=decoder,
        coord_dim=1,
        latent_size=1,
        source_key="u",
    )
    return model, batch


def test_streamed_encoded_query_decoding_matches_unchunked_eager_and_jit():
    model, batch = _tiny_encoded_operator_and_batch()
    eager = model(batch)
    compiled = eqx.filter_jit(lambda current, data: current(data))(model, batch)
    source = ArrayOperatorQuerySource(
        batch.query("query"),
        case_shape=batch.case_shape,
        fingerprint="scenario-query-v1",
    )
    last = source.read_chunk(4, 2)
    assert last.valid_count == 1
    assert last.samples.sample_shape == (2,)
    assert jnp.array_equal(last.samples.mask[:, 1], jnp.array([False, False]))
    assert jnp.array_equal(last.samples.quadrature_weights[:, 1], jnp.zeros((2,)))

    state = model.encode_inputs(batch)
    sink = ArrayPredictionSink()
    streamed = decode_query_chunks(
        model,
        batch,
        source,
        sink,
        chunk_size=2,
        encoded_state=state,
        compile=True,
    )

    assert jnp.allclose(compiled, eager, rtol=1e-6, atol=1e-7)
    assert jnp.allclose(streamed, eager, rtol=1e-6, atol=1e-7)
    assert sink.metadata is not None
    assert sink.metadata.output_shape == (2, 5)
    assert sink.metadata.query_fingerprint == "scenario-query-v1"
    assert jnp.array_equal(eager[~batch.query("query").mask], jnp.zeros((2,)))


def test_npy_prediction_status_uses_current_canonical_fields(tmp_path):
    model, batch = _tiny_encoded_operator_and_batch()
    source = ArrayOperatorQuerySource(
        batch.query("query"),
        case_shape=batch.case_shape,
        fingerprint="scenario-query-current",
    )
    output_path = tmp_path / "prediction.npy"
    sink = phx.nn.NpyPredictionSink(output_path)
    result = decode_query_chunks(
        model,
        batch,
        source,
        sink,
        chunk_size=2,
    )
    status = json.loads(
        output_path.with_suffix(".npy.metadata.json").read_text(encoding="utf-8")
    )

    assert result == output_path
    assert status["complete"] is True
    assert set(status) == {"metadata", "next_index", "complete"}


def test_typed_branch_interactions_are_synchronous_deterministic_and_validated():
    values, samples = _context_fixture()
    sensor_state = phx.nn.PooledGeometryContext(channels=2, num_tokens=2)(
        values, samples
    )
    field_state = phx.nn.SampledAnchorContext(channels=2, num_anchors=2)(
        values, samples, indices=jnp.array([1, 3])
    )
    state = phx.nn.BranchedEncodedOperatorState(
        {"sensor": sensor_state, "field": field_state}
    )
    sensor = phx.nn.OperatorBranchSpec(
        "sensor",
        role="conditioning",
        geometry_kind="point_cloud",
        processor_group="encoder",
    )
    field = phx.nn.OperatorBranchSpec(
        "field",
        role="prediction",
        geometry_kind="geometry",
        query_name="state",
        output_spec=phx.nn.OperatorOutputSpec(),
        decoder_group="state-head",
    )
    interactions = phx.nn.bidirectional_branch_interactions(
        "sensor", "field", stage=1, parameter_group="shared", scale=0.5
    )
    graph = phx.nn.OperatorBranchGraph((sensor, field), interactions=interactions)
    attention = phx.nn.MeasureAwareAttention(
        source_channels=2,
        query_channels=2,
        out_channels=2,
        num_heads=1,
        head_dim=2,
        key=jr.key(20),
    )

    assert graph.conditioning_names == ("sensor",)
    assert graph.prediction_names == ("field",)
    assert graph.branch("sensor").source_name == "sensor"
    assert graph.branch("field").query_name == "state"
    assert tuple(item.source for item in graph.interactions_at(1)) == (
        "field",
        "sensor",
    )

    updated = phx.nn.apply_branch_interactions(
        state, graph, {"shared": attention}, 1
    )
    repeated = phx.nn.apply_branch_interactions(
        state, graph, {"shared": attention}, 1
    )
    for target_name, source_name in (("field", "sensor"), ("sensor", "field")):
        target = state.branch(target_name)
        source_state = state.branch(source_name)
        direct = attention(
            source_state.values,
            target.values,
            source_state.weights,
            source_mask=source_state.mask,
            query_mask=target.mask,
        )
        expected = (target.values + 0.5 * direct) * target.mask[..., None]
        assert jnp.allclose(updated.branch(target_name).values, expected)
        assert jnp.array_equal(updated.branch(target_name).mask, target.mask)
        assert jnp.array_equal(updated.branch(target_name).weights, target.weights)
        assert jnp.allclose(
            repeated.branch(target_name).values,
            updated.branch(target_name).values,
        )

    with pytest.raises(ValueError, match="declared branches"):
        phx.nn.OperatorBranchGraph(
            (sensor, field),
            interactions=(
                phx.nn.BranchInteractionSpec("missing", "field", stage=0),
            ),
        )
    wrong_shape_attention = phx.nn.MeasureAwareAttention(
        source_channels=2,
        query_channels=2,
        out_channels=1,
        num_heads=1,
        head_dim=2,
        key=jr.key(21),
    )
    with pytest.raises(ValueError, match="output channels"):
        phx.nn.apply_branch_interactions(
            state, graph, {"shared": wrong_shape_attention}, 1
        )


def test_differential_decoders_normalize_transform_jit_and_differentiate():
    decoder = phx.nn.MLP(
        in_size=1,
        out_size="scalar",
        hidden_sizes=(),
        key=jr.key(30),
    )
    normalization = phx.nn.DifferentialNormalization(
        jnp.array([2.0]), jnp.array([4.0])
    )
    transform = phx.nn.LinearDifferentialTransform(jnp.array([[[3.0]]]))
    model = phx.nn.DifferentialFieldDecoder(
        decoder,
        transform=transform,
        normalization=normalization,
    )
    points = jnp.array([[-0.5], [0.25], [1.0]])

    def decoder_jacobian(point):
        return jax.jacfwd(
            lambda coordinate: jnp.asarray(decoder(coordinate)).reshape((1,))
        )(point)

    jacobians = jax.vmap(decoder_jacobian)(points)
    physical = normalization.physical_jacobian(jacobians)
    expected = jax.vmap(transform)(physical)[:, 0]
    eager = model(points)
    compiled = eqx.filter_jit(lambda current, value: current(value))(model, points)

    assert jnp.allclose(eager, expected)
    assert jnp.allclose(compiled, eager)
    assert jnp.allclose(
        phx.nn.DifferentialFieldDecoder(
            decoder,
            transform="gradient",
            normalization=normalization,
        )(points),
        physical[:, 0, 0],
    )
    gradient = eqx.filter_grad(lambda current: jnp.sum(current(points) ** 2))(model)
    assert jnp.all(jnp.isfinite(gradient.transform.coefficients))
    assert jnp.linalg.norm(gradient.transform.coefficients) > 0.0

    with pytest.raises(ValueError, match="coefficients require shape"):
        phx.nn.LinearDifferentialTransform(jnp.ones((1, 1)))
    with pytest.raises(ValueError, match="Jacobian must end"):
        normalization.physical_jacobian(jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="trailing size 1"):
        model(jnp.ones((2, 2)))


def test_pde_ir_round_trip_canonical_hash_tokens_and_constraint_execution():
    field = PDEExpression.field("u")
    residual = -PDEExpression.constant(1.0) + field
    problem = PDEProblemIR(
        coordinates=(PDECoordinate("x", "space", bounds=(0.0, 1.0)),),
        fields=(PDEField("u", coordinates=("x",)),),
        equations=(PDEEquation("unit_residual", residual),),
    )
    equivalent = PDEProblemIR(
        coordinates=problem.coordinates,
        fields=problem.fields,
        equations=(
            PDEEquation("unit_residual", PDEExpression.field("u") - 1.0),
        ),
    )

    payload = pde_ir_to_json(problem)
    assert "schema_version" not in payload
    restored = pde_ir_from_json(payload)
    assert restored == problem
    assert pde_ir_to_json(equivalent) == payload
    assert pde_ir_hash(equivalent) == pde_ir_hash(problem) == problem.canonical_hash

    tokens = tokenize_pde_ir(problem)
    valid_count = int(jnp.sum(tokens.mask))
    assert valid_count == tokens.max_tokens
    assert tokens.canonical_hashes == (problem.canonical_hash,)
    padded = pad_pde_tokens(tokens, tokens.max_tokens + 2)
    assert padded.max_tokens == tokens.max_tokens + 2
    assert jnp.array_equal(padded.mask[-2:], jnp.array([False, False]))
    assert jnp.array_equal(padded.parent[:valid_count], tokens.parent)
    assert jnp.array_equal(padded.operator[:valid_count], tokens.operator)

    geometry = Interval1d(0.0, 1.0)
    u = geometry.Function()(0.0)
    compiled = compile_pde_problem(problem, fields={"u": u})
    points = frozendict({"x": cx.Field(jnp.array([0.25]), dims=(None,))})
    assert compiled.canonical_hash == problem.canonical_hash
    assert jnp.allclose(
        compiled.equation("unit_residual").residual(points).data,
        jnp.array([-1.0]),
    )

    constraint = compile_pde_functional_constraint(
        residual,
        problem,
        component=geometry.component(),
        sampling=phx.domain.PointSampling(4, layout=SampleLayout((("x",),))),
        field_names=("u",),
        sampling_mode="fixed",
    )
    assert isinstance(constraint, FunctionalConstraint)
    assert jnp.allclose(constraint.loss({"u": u}), 1.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    ("name", "make"),
    [
        (
            "coordinate bounds",
            lambda value: PDECoordinate("x", "space", bounds=(0.0, value)),
        ),
        (
            "coordinate dimension",
            lambda value: PDECoordinate(
                "x",
                "space",
                physical_dimension=(value,),
            ),
        ),
        (
            "field dimension",
            lambda value: PDEField("u", physical_dimension=(value,)),
        ),
        ("field scale", lambda value: PDEField("u", scale=(value,))),
        (
            "parameter scalar value",
            lambda value: phx.equations.PDEParameter("a", value=value),
        ),
        (
            "parameter vector value",
            lambda value: phx.equations.PDEParameter(
                "a",
                value=(0.0, value),
                components=2,
            ),
        ),
        (
            "parameter dimension",
            lambda value: phx.equations.PDEParameter(
                "a",
                physical_dimension=(value,),
            ),
        ),
        (
            "parameter scale",
            lambda value: phx.equations.PDEParameter("a", scale=(value,)),
        ),
        ("expression value", lambda value: PDEExpression.constant(value)),
        (
            "expression dimension",
            lambda value: PDEExpression.constant(
                0.0,
                physical_dimension=(value,),
            ),
        ),
        (
            "nondimensionalization",
            lambda value: PDEProblemIR(
                coordinates=(),
                fields=(),
                nondimensionalization=(("reference", value),),
            ),
        ),
    ],
)
def test_pde_numeric_metadata_rejects_nonfinite_during_construction(
    bad,
    name,
    make,
):
    with pytest.raises(ValueError, match="finite"):
        make(bad)


def test_pde_numeric_metadata_accepts_finite_zero_and_negative_dimensions():
    problem = PDEProblemIR(
        coordinates=(
            PDECoordinate(
                "x",
                "space",
                physical_dimension=(-1.0,),
                bounds=(-2.0, 0.0),
            ),
        ),
        fields=(
            PDEField(
                "u",
                coordinates=("x",),
                physical_dimension=(-2.0,),
                scale=(1.0,),
            ),
        ),
        parameters=(
            phx.equations.PDEParameter(
                "a",
                value=0.0,
                physical_dimension=(-3.0,),
                scale=(2.0,),
            ),
        ),
        nondimensionalization=(("x", 1.0),),
    )

    assert phx.equations.validate_pde_ir(problem) is problem
    assert '"value":0.0' in pde_ir_to_json(problem)

    object.__setattr__(problem.fields[0], "scale", (math.nan,))
    with pytest.raises(ValueError, match="finite"):
        phx.equations.validate_pde_ir(problem)


def _canonical_expression_problem(expression):
    return PDEProblemIR(
        coordinates=(PDECoordinate("x", "space"),),
        fields=tuple(
            PDEField(name, coordinates=("x",))
            for name in ("u", "v", "w", "z")
        ),
        equations=(PDEEquation("governing", expression),),
    )


def _token_arrays(tokens):
    return tuple(
        getattr(tokens, name)
        for name in (
            "kind",
            "operator",
            "attribute",
            "symbol",
            "scalar",
            "physical_dimension",
            "slot",
            "parent",
            "depth",
            "mask",
        )
    )


@pytest.mark.parametrize(
    "expressions",
    [
        lambda u, v, w, z: (
            ((u + v) + w) + z,
            u + (v + (w + z)),
            (u + v) + (w + z),
            z + (w + (v + u)),
        ),
        lambda u, v, w, z: (
            ((u * v) * w) * z,
            u * (v * (w * z)),
            (u * v) * (w * z),
            z * (w * (v * u)),
        ),
    ],
)
def test_associative_expression_canonicalization_is_recursive(expressions):
    fields = tuple(PDEExpression.field(name) for name in ("u", "v", "w", "z"))
    problems = tuple(
        _canonical_expression_problem(expression)
        for expression in expressions(*fields)
    )
    payloads = tuple(pde_ir_to_json(problem) for problem in problems)
    hashes = tuple(pde_ir_hash(problem) for problem in problems)
    tokens = tuple(tokenize_pde_ir(problem) for problem in problems)

    assert len(set(payloads)) == 1
    assert len(set(hashes)) == 1
    for current in tokens[1:]:
        assert all(
            jnp.array_equal(left, right)
            for left, right in zip(
                _token_arrays(tokens[0]),
                _token_arrays(current),
                strict=True,
            )
        )


def test_nonassociative_expression_trees_remain_distinct():
    u, v, w, _ = tuple(
        PDEExpression.field(name) for name in ("u", "v", "w", "z")
    )
    expressions = (
        (u / v) / w,
        u / (v / w),
        (u**2.0) ** 3.0,
        u**6.0,
        (u + v) * w,
        u + (v * w),
    )
    hashes = {
        _canonical_expression_problem(expression).canonical_hash
        for expression in expressions
    }
    assert len(hashes) == len(expressions)


def _compiler_backend_problem():
    return PDEProblemIR(
        coordinates=(PDECoordinate("x", "space"),),
        fields=(PDEField("u", coordinates=("x",)),),
    )


def test_pde_compiler_executes_all_derivative_backends():
    geometry = Interval1d(-1.0, 1.0)

    @geometry.Function("x")
    def u(x):
        return x[0] ** 4

    problem = _compiler_backend_problem()
    field = PDEExpression.field("u")
    for backend in ("ad", "jet"):
        derivative = phx.equations.compile_pde_expression(
            field.derivative("x", order=3),
            problem,
            fields={"u": u},
            differential_backend=backend,
        )
        assert jnp.allclose(derivative.func(jnp.array([0.25])), 6.0)

    finite_difference_coordinates = jnp.linspace(-1.0, 1.0, 257)
    finite_difference = phx.equations.compile_pde_expression(
        field.derivative("x", order=2),
        problem,
        fields={"u": u},
        differential_backend="fd",
    ).func((finite_difference_coordinates,))
    expected = 12.0 * finite_difference_coordinates**2
    assert jnp.allclose(
        finite_difference[2:-2],
        expected[2:-2],
        rtol=2e-2,
        atol=2e-2,
    )

    basis_coordinates = LegendreAxisSpec(24).materialize(
        jnp.array(-1.0),
        jnp.array(1.0),
    ).nodes
    basis = phx.equations.compile_pde_expression(
        field.derivative("x", order=2),
        problem,
        fields={"u": u},
        differential_backend="basis",
    ).func((basis_coordinates,))
    assert jnp.allclose(
        basis,
        12.0 * basis_coordinates**2,
        rtol=1e-7,
        atol=1e-7,
    )


def test_pde_compiler_rejects_invalid_backend_before_expression_dispatch():
    problem = _compiler_backend_problem()
    constant = PDEExpression.constant(1.0)
    geometry = Interval1d(-1.0, 1.0)
    u = geometry.Function("x")(0.0)

    calls = (
        lambda: phx.equations.compile_pde_expression(
            constant,
            problem,
            fields={"u": u},
            differential_backend="definitely_invalid",
        ),
        lambda: compile_pde_problem(
            problem,
            fields={"u": u},
            differential_backend="definitely_invalid",
        ),
        lambda: compile_pde_functional_constraint(
            constant,
            problem,
            component=None,
            sampling=phx.domain.PointSampling(1, layout=None),
            differential_backend="definitely_invalid",
        ),
    )
    for call in calls:
        with pytest.raises(ValueError, match="definitely_invalid"):
            call()


def test_scatter_operator_graph_entities_is_exported_from_nn():
    assert (
        phx.nn.scatter_operator_graph_entities
        is phx.nn.models.scatter_operator_graph_entities
    )
