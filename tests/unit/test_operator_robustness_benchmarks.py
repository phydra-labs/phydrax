import jax
import jax.numpy as jnp

from tools.operator_benchmarks.models import compatible_architectures
from tools.operator_benchmarks.scenarios import (
    add_sensor_dropout_ladder,
    irregular_causal_relaxation_scenario,
    periodic_burgers_scenario,
)


def test_sensor_dropout_ladder_uses_nested_declared_masks():
    scenario = periodic_burgers_scenario(
        train_resolution=12,
        test_resolution=16,
        num_cases=4,
    )
    ladder = add_sensor_dropout_ladder(
        scenario,
        drop_fractions=(0.1, 0.3, 0.5),
        seed=7,
    )
    evaluations = ladder.evaluations[-3:]
    masks = tuple(
        evaluation.batch.input("state").mask_array(case_shape=evaluation.batch.case_shape)
        for evaluation in evaluations
    )

    assert tuple(evaluation.name for evaluation in evaluations) == (
        "sensor_dropout_10pct",
        "sensor_dropout_30pct",
        "sensor_dropout_50pct",
    )
    assert all(evaluation.shift == "sensor_dropout" for evaluation in evaluations)
    assert jnp.all(masks[1] <= masks[0])
    assert jnp.all(masks[2] <= masks[1])
    assert jnp.sum(masks[2]) < jnp.sum(masks[0])
    assert dict(ladder.metadata)["sensor_dropout_ladder"] == "0.1,0.3,0.5"


def test_irregular_causal_scenario_runs_recurrent_architectures():
    scenario = irregular_causal_relaxation_scenario(
        points=9,
        num_cases=3,
        final_time=1.0,
        extrapolation_factor=1.8,
        seed=11,
    )
    architectures = {
        architecture.name: architecture
        for architecture in compatible_architectures(scenario, quick=True)
    }
    architecture = architectures["selective_state_space"]
    model = architecture.build(scenario, seed=13)
    recurrent = architectures["linear_recurrent"].build(scenario, seed=17)

    train_output = model(scenario.train_batch)
    evaluation_outputs = tuple(
        model(evaluation.batch) for evaluation in scenario.evaluations
    )
    recurrent_outputs = (
        recurrent(scenario.train_batch),
        *(recurrent(evaluation.batch) for evaluation in scenario.evaluations),
    )
    train_target = scenario.train_target
    assert isinstance(train_target, jax.Array)
    source = scenario.train_batch.input("forcing")
    train_times = source.coordinates_array(case_shape=scenario.train_batch.case_shape)[
        ..., 0
    ]
    _, diagnostics = model.evaluate_with_diagnostics(
        source.values,
        train_times,
        mask=source.mask_array(case_shape=scenario.train_batch.case_shape),
    )
    configuration = dict(architecture.configuration(scenario))

    assert train_output.shape == train_target.shape
    assert jnp.all(jnp.isfinite(train_output))
    assert all(jnp.all(jnp.isfinite(output)) for output in evaluation_outputs)
    assert all(jnp.all(jnp.isfinite(output)) for output in recurrent_outputs)
    assert recurrent_outputs[0].shape == train_target.shape
    assert scenario.ladder == "temporal_irregularity"
    assert diagnostics.extrapolated_fraction == 0.0
    assert diagnostics.minimum_physical_step == jnp.asarray(
        float(configuration["minimum_training_step"])
    )
    assert diagnostics.maximum_physical_step == jnp.asarray(
        float(configuration["maximum_training_step"])
    )
    ragged = scenario.evaluations[-1]
    ragged_mask = ragged.batch.require_single_query().mask_array(
        case_shape=ragged.batch.case_shape
    )
    assert jnp.array_equal(
        evaluation_outputs[-1],
        jnp.where(ragged_mask, evaluation_outputs[-1], 0.0),
    )
