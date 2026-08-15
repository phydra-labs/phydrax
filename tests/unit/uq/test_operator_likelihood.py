#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _batch(*, quadrature_scale: float = 1.0) -> phx.nn.operator.OperatorBatch:
    coordinates = jnp.asarray(
        [
            [[0.0], [0.5], [1.0]],
            [[0.0], [0.5], [1.0]],
        ]
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=quadrature_scale
        * jnp.asarray([[0.25, 0.5, 0.25], [0.25, 0.5, 0.25]]),
        mask=jnp.asarray([[True, True, False], [True, True, True]]),
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.ones((2, 3)),
        coordinates=coordinates,
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"forcing": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(2,),
    )


def _prediction(parameters, batch, spec):
    values = jnp.broadcast_to(parameters["level"], spec.expected_shape(batch))
    return phx.nn.operator.OperatorPrediction.from_field(
        "output",
        values,
        "query",
        batch.require_single_query(),
        spec=spec,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def _target_and_mask():
    target = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0], [jnp.nan, jnp.nan]],
            [[2.0, 3.0], [4.0, jnp.nan], [6.0, 7.0]],
        ]
    )
    observation_mask = jnp.asarray(
        [
            [[True, False], [True, True], [True, True]],
            [[True, True], [True, False], [False, True]],
        ]
    )
    return target, observation_mask


def test_operator_likelihood_matches_manual_sum_and_is_jittable():
    batch = _batch()
    spec = phx.nn.operator.OperatorOutputSpec(2, component_names=("u", "v"))
    target, observation_mask = _target_and_mask()
    scale = 0.5
    term = phx.uq.FixedOperatorObservationLikelihood(
        lambda parameters: _prediction(parameters, batch, spec),
        batch,
        target,
        phx.uq.GaussianLikelihood(scale),
        output_spec=spec,
        field_name="output",
        query_name="query",
        observation_mask=observation_mask,
    )
    parameters = {"level": jnp.asarray(1.5)}

    combined = batch.require_single_query().mask_array(case_shape=batch.case_shape)[
        ..., None
    ]
    combined = jnp.broadcast_to(combined, target.shape) & observation_mask
    safe_target = jnp.where(combined, target, 0.0)
    elements = (
        -0.5 * ((safe_target - parameters["level"]) / scale) ** 2
        - jnp.log(scale)
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )
    expected = jnp.sum(jnp.where(combined, elements, 0.0), axis=(1, 2))

    assert jnp.allclose(term.per_case_log_prob(parameters), expected)
    assert jnp.allclose(term.log_prob(parameters), jnp.sum(expected))
    compiled = eqx.filter_jit(
        lambda likelihood_term, value: likelihood_term.log_prob(value)
    )
    assert jnp.allclose(compiled(term, parameters), jnp.sum(expected))


def test_operator_likelihood_gradient_and_standardized_residual():
    batch = _batch()
    spec = phx.nn.operator.OperatorOutputSpec(2)
    target, observation_mask = _target_and_mask()
    scale = 0.75
    term = phx.uq.FixedOperatorObservationLikelihood(
        lambda level: _prediction({"level": level}, batch, spec),
        batch,
        target,
        phx.uq.GaussianLikelihood(scale),
        output_spec=spec,
        field_name="output",
        query_name="query",
        observation_mask=observation_mask,
    )
    level = jnp.asarray(0.25)
    combined = term.observation_mask
    safe_target = jnp.where(combined, target, 0.0)

    gradient = jax.grad(term.log_prob)(level)
    expected_gradient = jnp.sum(
        jnp.where(combined, (safe_target - level) / scale**2, 0.0)
    )
    residual = term.standardized_residual(level)

    assert jnp.allclose(gradient, expected_gradient)
    assert jnp.all(residual[~combined] == 0.0)
    assert jnp.allclose(
        residual[combined],
        ((safe_target - level) / scale)[combined],
    )


def test_operator_likelihood_is_independent_of_quadrature_weights():
    first = _batch(quadrature_scale=1.0)
    second = _batch(quadrature_scale=17.0)
    spec = phx.nn.operator.OperatorOutputSpec(2)
    target, observation_mask = _target_and_mask()

    def make_term(batch):
        return phx.uq.FixedOperatorObservationLikelihood(
            lambda parameters: _prediction(parameters, batch, spec),
            batch,
            target,
            phx.uq.GaussianLikelihood(0.4),
            output_spec=spec,
            field_name="output",
            query_name="query",
            observation_mask=observation_mask,
        )

    parameters = {"level": jnp.asarray(0.7)}
    assert jnp.array_equal(
        make_term(first).per_case_log_prob(parameters),
        make_term(second).per_case_log_prob(parameters),
    )


def test_operator_likelihood_handles_nonfinite_values_by_observation_status():
    batch = _batch()
    spec = phx.nn.operator.OperatorOutputSpec(2)
    target, observation_mask = _target_and_mask()
    finite_term = phx.uq.FixedOperatorObservationLikelihood(
        lambda parameters: _prediction(parameters, batch, spec),
        batch,
        target,
        phx.uq.GaussianLikelihood(1.0),
        output_spec=spec,
        field_name="output",
        query_name="query",
        observation_mask=observation_mask,
    )

    def invalid_prediction(parameters):
        prediction = _prediction(parameters, batch, spec)
        values = prediction.field("output").values.at[0, 0, 0].set(jnp.nan)
        values = values.at[0, 2, 0].set(jnp.nan)
        return phx.nn.operator.OperatorPrediction.from_field(
            "output",
            values,
            "query",
            batch.require_single_query(),
            spec=spec,
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        )

    invalid_term = phx.uq.FixedOperatorObservationLikelihood(
        invalid_prediction,
        batch,
        target,
        phx.uq.GaussianLikelihood(1.0),
        output_spec=spec,
        field_name="output",
        query_name="query",
        observation_mask=observation_mask,
    )
    parameters = {"level": jnp.asarray(1.0)}

    assert jnp.isfinite(finite_term.log_prob(parameters))
    assert invalid_term.log_prob(parameters) == -jnp.inf


def test_operator_likelihood_rejects_empty_cases_and_contract_mismatches():
    batch = _batch()
    spec = phx.nn.operator.OperatorOutputSpec(2)
    target, _ = _target_and_mask()
    empty_case_mask = jnp.ones_like(target, dtype=bool).at[1].set(False)

    with pytest.raises(ValueError, match="at least one observation"):
        phx.uq.FixedOperatorObservationLikelihood(
            lambda parameters: _prediction(parameters, batch, spec),
            batch,
            target,
            phx.uq.GaussianLikelihood(1.0),
            output_spec=spec,
            field_name="output",
            query_name="query",
            observation_mask=empty_case_mask,
        )

    term = phx.uq.FixedOperatorObservationLikelihood(
        lambda parameters: phx.nn.operator.OperatorPrediction.from_field(
            "output",
            jnp.ones((2, 3)),
            "query",
            batch.require_single_query(),
            spec=phx.nn.operator.OperatorOutputSpec("scalar"),
            case_axes=batch.case_axes,
            case_shape=batch.case_shape,
        ),
        batch,
        jnp.ones((2, 3, 2)),
        phx.uq.GaussianLikelihood(1.0),
        output_spec=spec,
        field_name="output",
        query_name="query",
    )
    with pytest.raises(ValueError, match="fixed batch contract"):
        term.log_prob({"level": jnp.asarray(0.0)})


def _operator_dataset(cases=5, resolution=4):
    axis = phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, resolution))
    values = jnp.arange(cases, dtype=float)[:, None] + axis.nodes[None, :]
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"solution": 2.0 * values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )


def _dynamic_prediction(parameter, batch):
    return phx.nn.operator.OperatorPrediction.from_field(
        "solution",
        parameter * batch.input("state").values,
        "query",
        batch.query("query"),
        spec=phx.nn.operator.OperatorOutputSpec(),
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def test_dynamic_operator_likelihood_matches_fixed_full_batch_and_is_jittable():
    dataset = _operator_dataset()
    target_field = dataset.targets.field("solution")
    dynamic = phx.uq.OperatorBatchObservationLikelihood(
        _dynamic_prediction,
        phx.uq.GaussianLikelihood(0.2),
    )
    data = phx.uq.OperatorLikelihoodData(
        dataset.batch,
        target_field.values,
        output_spec=target_field.spec,
        field_name="solution",
        query_name=target_field.query_name,
    )
    fixed = phx.uq.FixedOperatorObservationLikelihood(
        lambda parameter: _dynamic_prediction(parameter, dataset.batch),
        dataset.batch,
        target_field.values,
        phx.uq.GaussianLikelihood(0.2),
        output_spec=target_field.spec,
        field_name="solution",
        query_name=target_field.query_name,
    )
    parameter = jnp.asarray(1.7)

    assert jnp.allclose(
        dynamic.per_case_log_prob(parameter, data),
        fixed.per_case_log_prob(parameter),
    )
    likelihood_batch = phx.uq.LikelihoodBatch(
        data,
        jnp.asarray([True, True, True, True, False]),
    )
    compiled = eqx.filter_jit(lambda value, batch: dynamic(value, batch))(
        parameter, likelihood_batch
    )
    assert compiled.shape == (5,)
    assert compiled[-1] == 0.0


def test_operator_minibatch_source_is_complete_padded_and_content_addressed():
    dataset = _operator_dataset()
    loader = phx.nn.operator.training.OperatorBatchLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        seed=6,
        drop_last=False,
        prefetch=1,
    )
    source = phx.uq.OperatorMinibatchSource(loader, field_name="solution")
    batches = tuple(source.epoch(0))
    case_ids = jnp.concatenate(
        [
            batch.data.batch.input("state").values[:, 0][batch.factor_mask]
            for batch in batches
        ]
    )

    assert source.num_factors == 5
    assert source.batch_capacity == 2
    assert source.batches_per_epoch == 3
    assert [int(batch.factor_count) for batch in batches] == [2, 2, 1]
    assert jnp.array_equal(jnp.sort(case_ids), jnp.arange(5.0))
    assert batches[-1].data.target.shape == (2, 4)
    assert not bool(batches[-1].factor_mask[-1])
    assert source.configuration()["loader_fingerprint"] == loader.fingerprint

    changed_seed = phx.uq.OperatorMinibatchSource(
        phx.nn.operator.training.OperatorBatchLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            seed=7,
            drop_last=False,
            prefetch=1,
        ),
        field_name="solution",
    )
    changed_data = phx.uq.OperatorMinibatchSource(
        phx.nn.operator.training.OperatorBatchLoader(
            _operator_dataset(cases=6),
            batch_size=2,
            shuffle=True,
            seed=6,
            drop_last=False,
            prefetch=1,
        ),
        field_name="solution",
    )
    assert source.fingerprint != changed_seed.fingerprint
    assert source.fingerprint != changed_data.fingerprint


def test_operator_minibatch_source_rejects_lossy_loader_policies():
    dataset = _operator_dataset()
    with pytest.raises(ValueError, match="drop_last=True"):
        phx.uq.OperatorMinibatchSource(
            phx.nn.operator.training.OperatorBatchLoader(
                dataset,
                batch_size=2,
                shuffle=True,
                seed=1,
                drop_last=True,
            ),
            field_name="solution",
        )
    with pytest.raises(ValueError, match="shuffle=True"):
        phx.uq.OperatorMinibatchSource(
            phx.nn.operator.training.OperatorBatchLoader(
                dataset,
                batch_size=2,
                shuffle=False,
                seed=1,
                drop_last=False,
            ),
            field_name="solution",
        )


def test_operator_sgmcmc_supports_selected_parameter_subspaces_and_predictions():
    dataset = _operator_dataset()
    source = phx.uq.OperatorMinibatchSource(
        phx.nn.operator.training.OperatorBatchLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            seed=8,
            drop_last=False,
            prefetch=1,
        ),
        field_name="solution",
    )
    model = {"frozen": jnp.asarray(11.0), "weight": jnp.asarray(1.8)}
    subspace = phx.nn.parameters.ParameterSubspace.from_leaf_paths(model, ("['weight']",))
    parameter_space = phx.uq.ParameterSpace(
        subspace.initial,
        priors={"frozen": None, "weight": phx.uq.Normal(0.0, 3.0)},
    )

    def predict(selected, batch):
        parameters = subspace.reconstruct(selected)
        return _dynamic_prediction(parameters["weight"], batch)

    likelihood = phx.uq.OperatorBatchObservationLikelihood(
        predict,
        phx.uq.GaussianLikelihood(0.2),
    )
    target_field = dataset.targets.field("solution")
    full_data = phx.uq.OperatorLikelihoodData(
        dataset.batch,
        target_field.values,
        output_spec=target_field.spec,
        field_name="solution",
        query_name=target_field.query_name,
    )
    problem = phx.uq.MinibatchPosteriorProblem(
        parameter_space,
        likelihood,
        num_factors=source.num_factors,
        full_log_likelihood=lambda selected: jnp.sum(
            likelihood.per_case_log_prob(selected, full_data)
        ),
        predict=lambda selected: phx.uq.operator_prediction_field(
            predict(selected, dataset.batch),
            field_name="solution",
        ),
    )
    result = phx.uq.sample_sgld(
        problem,
        source,
        key=jr.key(40),
        step_size=1.0e-6,
        num_chains=2,
        num_burnin=2,
        num_samples=4,
    )
    prediction = result.predict()
    assert isinstance(prediction, phx.uq.PredictiveField)

    assert result.samples["weight"].shape == (2, 4)
    assert result.samples["frozen"] is None
    selected_draw = {
        "frozen": None,
        "weight": result.samples["weight"][0, 0],
    }
    assert subspace.reconstruct(selected_draw)["frozen"] == 11.0
    assert prediction.samples.dims == (
        "__phydra_uq_chain",
        "__phydra_uq_draw",
        "case",
        "x",
    )
    assert prediction.samples.shape == (2, 4, 5, 4)
