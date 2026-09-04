import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.incompressible_flow._assimilation import (
    ModelErrorAssimilationIdentity,
    ModelErrorAssimilationObjective,
    ModelErrorRolloutEvaluator,
    PeriodicModelErrorParameterization,
    QuadraticModelErrorRegularization,
    SparseTimeAverageObservationData,
    SparseTimeAverageObservationOperator,
    TimeAverageWindows,
)
from phydrax.applications.incompressible_flow._forcing import (
    SolenoidalHermitianFourierBasis,
)


def _periodic_parameterization():
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(4),
            phx.discretization.FourierBasisPlan(4),
        ),
        axis_names=("x", "y"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
        )
    )
    projector = phx.discretization.PeriodicLerayProjector(space)
    basis = SolenoidalHermitianFourierBasis(projector, maximum_wavenumber=1.1)
    time_grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 1.0, 2.0]),
        time_id="assimilation-control-time",
    )
    parameterization = PeriodicModelErrorParameterization(
        basis,
        time_grid,
        base_forcing_id="physical-forcing",
    )
    return parameterization, projector


def _correction_rollout(parameterization, parameters, sample_times):
    return parameterization.sample(parameters, sample_times)


def _basis_coordinate_evaluator(basis):
    def evaluate(corrections, sample_times):
        del sample_times
        coordinates = jax.vmap(basis.analyze)(corrections)
        return coordinates[:, :2]

    return evaluate


def _objective(
    observation_values=(0.2, -0.3),
    training_mask=(True, False),
    *,
    amplitude_weight=0.2,
    temporal_difference_weight=0.1,
    runtime_identity_updates=None,
):
    parameterization, projector = _periodic_parameterization()
    windows = TimeAverageWindows(
        jnp.asarray([0.25, 0.75, 1.25, 1.75]),
        jnp.asarray(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        ),
        labels=("early", "late"),
    )
    operator = SparseTimeAverageObservationOperator(
        windows,
        2,
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([0, 1], dtype=jnp.int32),
    )
    observations = SparseTimeAverageObservationData(
        operator,
        jnp.asarray(observation_values),
        jnp.asarray([0.5, 0.25]),
        jnp.asarray(training_mask),
    )
    identity_values = {
        "problem_id": "periodic-flow-problem",
        "compiler_id": "spectral-compiler",
        "filter_id": "resolved-filter",
        "forcing_id": parameterization.base_forcing_id,
        "observation_id": observations.data_id,
    }
    if runtime_identity_updates is not None:
        identity_values.update(runtime_identity_updates)
    runtime = ModelErrorRolloutEvaluator(
        ModelErrorAssimilationIdentity(**identity_values),
        _correction_rollout,
        _basis_coordinate_evaluator(parameterization.basis),
        rollout_id="fixed-sample-correction-rollout",
        evaluator_id="basis-coordinate-observation-evaluator",
    )
    regularization = QuadraticModelErrorRegularization(
        amplitude_weight=amplitude_weight,
        temporal_difference_weight=temporal_difference_weight,
    )
    objective = ModelErrorAssimilationObjective(
        parameterization,
        operator,
        observations,
        regularization,
        runtime,
        problem_id="periodic-flow-problem",
        compiler_id="spectral-compiler",
        filter_id="resolved-filter",
    )
    return objective, projector


def test_sparse_time_average_operator_uses_declared_windows_and_reports_mismatch():
    windows = TimeAverageWindows.from_bounds(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([1.0, 2.0]),
        labels=("first", "second"),
    )
    operator = SparseTimeAverageObservationOperator(
        windows,
        2,
        jnp.asarray([0, 1, 1], dtype=jnp.int32),
        jnp.asarray([0, 0, 1], dtype=jnp.int32),
    )
    samples = jnp.asarray([[0.0, 10.0], [2.0, 20.0], [4.0, 40.0]])
    predicted = operator.apply(samples)
    np.testing.assert_allclose(predicted, jnp.asarray([1.0, 3.0, 30.0]))

    data = SparseTimeAverageObservationData(
        operator,
        jnp.asarray([0.5, 4.0, 25.0]),
        jnp.ones((3,)),
        jnp.asarray([True, True, False]),
    )
    np.testing.assert_allclose(predicted - data.values, jnp.asarray([0.5, -1.0, 5.0]))


def test_periodic_model_error_correction_is_exactly_divergence_free_and_real():
    parameterization, projector = _periodic_parameterization()
    parameters = jnp.linspace(
        -0.4,
        0.7,
        np.prod(parameterization.parameter_shape),
    ).reshape(parameterization.parameter_shape)
    times = jnp.asarray([0.25, 1.25, 2.0])
    corrections = parameterization.sample(parameters, times)
    divergence = jax.vmap(projector.divergence)(corrections)
    np.testing.assert_allclose(divergence, 0.0, atol=1.0e-12)
    recovered = jax.vmap(parameterization.basis.analyze)(corrections)
    np.testing.assert_allclose(recovered[0], parameters[0], atol=1.0e-12)
    np.testing.assert_allclose(
        recovered[1:],
        jnp.broadcast_to(parameters[1], recovered[1:].shape),
        atol=1.0e-12,
    )
    assert "not-identifiable-as-sgs-stress" in parameterization.model_interpretation


def test_quadratic_model_error_regularization_separates_terms():
    parameters = jnp.asarray([[1.0, -1.0], [3.0, 1.0]])
    regularization = QuadraticModelErrorRegularization(
        amplitude_weight=2.0,
        temporal_difference_weight=3.0,
    )
    evidence = regularization.evaluate(parameters)
    expected_amplitude = 0.5 * 2.0 * np.mean(np.asarray(parameters) ** 2)
    expected_temporal = 0.5 * 3.0 * np.mean(np.diff(np.asarray(parameters), axis=0) ** 2)
    np.testing.assert_allclose(evidence.amplitude, expected_amplitude)
    np.testing.assert_allclose(evidence.temporal_difference, expected_temporal)
    np.testing.assert_allclose(evidence.total, expected_amplitude + expected_temporal)


def test_objective_value_gradient_matches_jax_and_centered_difference():
    objective, _ = _objective()
    parameters = jnp.linspace(
        -0.15,
        0.25,
        np.prod(objective.parameterization.parameter_shape),
    ).reshape(objective.parameterization.parameter_shape)
    evaluated = objective.evaluate(parameters)
    differentiated = objective.value_and_gradient(parameters)
    reference_gradient = jax.grad(objective)(parameters)
    compiled = eqx.filter_jit(objective.value_and_gradient)(parameters)

    np.testing.assert_allclose(differentiated.value, evaluated.value, rtol=1.0e-12)
    np.testing.assert_allclose(
        differentiated.gradient,
        reference_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        compiled.gradient,
        reference_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    direction = jnp.zeros_like(parameters).at[0, 0].set(1.0)
    step = 1.0e-4
    centered = (
        objective(parameters + step * direction)
        - objective(parameters - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        differentiated.gradient[0, 0],
        centered,
        rtol=2.0e-4,
        atol=2.0e-6,
    )
    assert bool(evaluated.evidence.successful)
    assert evaluated.evidence.training_count == 1
    assert evaluated.evidence.holdout_count == 1
    assert "not-identifiable-as-sgs-stress" in evaluated.evidence.model_interpretation


def test_holdout_observations_are_reported_but_do_not_enter_training_value():
    first, _ = _objective(
        observation_values=(0.2, -0.3),
        amplitude_weight=0.0,
        temporal_difference_weight=0.0,
    )
    second, _ = _objective(
        observation_values=(0.2, 8.0),
        amplitude_weight=0.0,
        temporal_difference_weight=0.0,
    )
    parameters = jnp.zeros(first.parameterization.parameter_shape)
    first_result = first.evaluate(parameters)
    second_result = second.evaluate(parameters)

    np.testing.assert_allclose(first_result.value, second_result.value, atol=0.0)
    np.testing.assert_allclose(
        first_result.evidence.training_data_misfit,
        second_result.evidence.training_data_misfit,
        atol=0.0,
    )
    assert not np.isclose(
        first_result.evidence.holdout_data_misfit,
        second_result.evidence.holdout_data_misfit,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("problem_id", "other-problem"),
        ("compiler_id", "other-compiler"),
        ("filter_id", "other-filter"),
        ("forcing_id", "other-forcing"),
        ("observation_id", "other-observations"),
    ),
)
def test_objective_rejects_runtime_identity_mismatches(field, value):
    with pytest.raises(ValueError, match=field):
        _objective(runtime_identity_updates={field: value})


def test_objective_rejects_observation_operator_identity_mismatch():
    parameterization, _ = _periodic_parameterization()
    windows = TimeAverageWindows(
        jnp.asarray([0.25, 0.75]),
        jnp.asarray([[1.0, 1.0]]),
    )
    first_operator = SparseTimeAverageObservationOperator(
        windows,
        2,
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([0], dtype=jnp.int32),
    )
    second_operator = SparseTimeAverageObservationOperator(
        windows,
        2,
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([1], dtype=jnp.int32),
    )
    observations = SparseTimeAverageObservationData(
        first_operator,
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        jnp.asarray([True]),
    )
    identity = ModelErrorAssimilationIdentity(
        problem_id="periodic-flow-problem",
        compiler_id="spectral-compiler",
        filter_id="resolved-filter",
        forcing_id=parameterization.base_forcing_id,
        observation_id=observations.data_id,
    )
    runtime = ModelErrorRolloutEvaluator(
        identity,
        _correction_rollout,
        _basis_coordinate_evaluator(parameterization.basis),
        rollout_id="fixed-sample-correction-rollout",
        evaluator_id="basis-coordinate-observation-evaluator",
    )
    with pytest.raises(ValueError, match="data and operator identities"):
        ModelErrorAssimilationObjective(
            parameterization,
            second_operator,
            observations,
            QuadraticModelErrorRegularization(),
            runtime,
            problem_id="periodic-flow-problem",
            compiler_id="spectral-compiler",
            filter_id="resolved-filter",
        )


def test_invalid_sparse_observations_fail_closed_at_construction():
    windows = TimeAverageWindows(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[1.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="outside"):
        SparseTimeAverageObservationOperator(
            windows,
            1,
            jnp.asarray([1], dtype=jnp.int32),
            jnp.asarray([0], dtype=jnp.int32),
        )
    operator = SparseTimeAverageObservationOperator(
        windows,
        1,
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([0], dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="values must be finite"):
        SparseTimeAverageObservationData(
            operator,
            jnp.asarray([jnp.nan]),
            jnp.asarray([1.0]),
            jnp.asarray([True]),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        SparseTimeAverageObservationData(
            operator,
            jnp.asarray([0.0]),
            jnp.asarray([0.0]),
            jnp.asarray([True]),
        )
    with pytest.raises(ValueError, match="assigned to training"):
        SparseTimeAverageObservationData(
            operator,
            jnp.asarray([0.0]),
            jnp.asarray([1.0]),
            jnp.asarray([False]),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        TimeAverageWindows(
            jnp.asarray([0.0, 1.0]),
            jnp.asarray([[1.0, -1.0]]),
        )
