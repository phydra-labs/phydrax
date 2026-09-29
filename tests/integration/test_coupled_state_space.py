#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""State-space inference through a coupled finite-element/virtual-element transient.

The hidden state is the native coordinate vector of the FE-VEM heat transient
of ``examples/_coupled_heat_transient.py`` under a known wall heat flux; the
transition is ``prepare_coupled_transition`` plus
``phydrax.stochastic.CoupledTransitionKernel`` (declared additive Gaussian
process noise), and the observation model evaluates the P6 sensor bindings
through ``CoupledObservationPort`` with the covariance derived from the
declared noise of ``MeasurementComparisonPlan`` sensor records.

Independent references are host NumPy: the implicit-Euler recursion of the
semi-discrete system ``C x' + K x = b + B g`` and the sensor rows ``H``
assembled on the host from the two meshes (``host_semidiscrete``: P1 and
degree-1 VEM operators weighted by each owner's ``rho c``; only the verified
native coordinate chart is read from the prepared transient), the exact
linear-Gaussian Kalman filter of that system, the exact Kalman update and the
Gaussian innovation likelihood of an ensemble's sample statistics, the
consumer-keyed draws, and Gaussian densities.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from examples import _coupled_heat_transient as heat, coupled_data_assimilation as da
from phydrax.solver import coupling as cpl


CASES = ("north", "south")


@dataclass(frozen=True)
class _Plate:
    transient: cpl.PreparedCoupledTransient
    transition: cpl.PreparedCoupledTransition
    port: cpl.CoupledObservationPort
    reference: heat.HostSemidiscrete
    steps: list[tuple[np.ndarray, np.ndarray]]
    output: np.ndarray
    values: np.ndarray
    mask: np.ndarray


@pytest.fixture(scope="module")
def plate() -> _Plate:
    transient, transition, port = da.assimilation_model(da.CELLS)
    size = transition.state_size
    reference = heat.host_semidiscrete(transient, da.CELLS)
    durations = np.diff(np.concatenate(([0.0], da.TIMES)))
    steps = [da.host_step(reference, float(step)) for step in durations]
    values, mask = da.synthetic_record(steps, reference.sensors, size)
    return _Plate(
        transient, transition, port, reference, steps, reference.sensors, values, mask
    )


def test_sensor_port_is_the_host_point_evaluation(plate: _Plate) -> None:
    # P1 barycentric weights in the conductor, the degree-1 VEM projection in
    # the insulator; the point observation of a zero-lift field is linear.
    np.testing.assert_allclose(
        da.sensor_matrix(plate.port, plate.transition.state_size),
        plate.output,
        rtol=0.0,
        atol=1e-14,
    )


def _two_case_problem(plate: _Plate) -> phx.stochastic.StateSpaceProblem:
    """Two physical cases with different records; case IDs address the keys."""
    single = da.state_space_problem(
        plate.transition, plate.port, plate.values, plate.mask
    )
    model = single.model
    size = plate.transition.state_size
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((2, size)),
        jnp.broadcast_to(da.PRIOR_STD**2 * jnp.eye(size), (2, size, size)),
        state_shape=(size,),
    )
    second_mask = plate.mask.copy()
    second_mask[1, 0] = False
    sequence = phx.stochastic.ObservationSequence(
        da.TIMES,
        np.stack((plate.values, np.where(second_mask, 0.9 * plate.values, 0.0))),
        case_axes=("site",),
        case_shape=(2,),
        observation_mask=np.stack((plate.mask, second_mask)),
        case_ids=CASES,
        sequence_id="two-site-sensors",
    )
    return phx.stochastic.StateSpaceProblem(
        phx.stochastic.StateSpaceModel(
            prior, model.transition, model.observation, model_id="two-site-heat"
        ),
        sequence,
        initial_time=0.0,
        args=single.args,
        problem_id="two-site-heat",
    )


def test_coupled_forecast_uses_semantic_keys_and_analysis_is_the_ensemble_kalman_update(
    plate: _Plate,
) -> None:
    problem = _two_case_problem(plate)
    size = plate.transition.state_size
    key = jr.key(7)
    state = phx.uq.initialize_ensemble_filter(key, problem, ensemble_size=12)
    previous = np.asarray(state.ensemble)
    state, record = phx.uq.ensemble_filter_step(problem, state)
    assert bool(jnp.all(record.valid))
    matrix, bias = plate.steps[0]
    for case_index, case_id in enumerate(CASES):
        # The kernel draws only from the consumer's key: (case ID, step, member).
        draws = np.stack(
            [
                np.asarray(
                    jr.normal(
                        phx.stochastic.state_space_key(
                            key, "ensemble-filter-transition", case_id, 0, member=member
                        ),
                        (size,),
                    )
                )
                for member in range(12)
            ]
        )
        expected = previous[case_index] @ matrix.T + bias + da.PROCESS_STD * draws
        forecast = np.asarray(record.forecast_ensemble[case_index])
        np.testing.assert_allclose(forecast, expected, rtol=0.0, atol=1e-11)
        observed = np.asarray(problem.observations.observation_mask[case_index, 0])
        value = np.asarray(problem.observations.values[case_index, 0])
        mean, covariance = da.ensemble_kalman_update(
            forecast, plate.output, value, observed
        )
        analysis = np.asarray(record.analysis_ensemble[case_index])
        np.testing.assert_allclose(analysis.mean(axis=0), mean, rtol=0.0, atol=1e-11)
        np.testing.assert_allclose(np.cov(analysis.T), covariance, rtol=0.0, atol=1e-12)


def test_ensemble_filter_tracks_the_exact_kalman_filter_of_the_affine_transition(
    plate: _Plate,
) -> None:
    problem = da.state_space_problem(
        plate.transition, plate.port, plate.values, plate.mask
    )
    key = jr.key(da.SEED)
    filtered = phx.uq.ensemble_transform_kalman_filter(
        key, problem, ensemble_size=da.ENSEMBLE, raise_on_failure=True
    )
    assert bool(jnp.all(filtered.valid))
    np.testing.assert_array_equal(
        np.asarray(filtered.observed_counts), plate.mask.sum(axis=-1)
    )
    size = plate.transition.state_size
    # The prior ensemble (drawn from the same key): whitened by the declared
    # prior std its entries are standard normal; the sample second moment of
    # N n = 2560 draws has standard error sqrt(2 / (N n)) = 0.028.
    previous = np.asarray(
        phx.uq.initialize_ensemble_filter(
            key, problem, ensemble_size=da.ENSEMBLE
        ).ensemble
    )
    assert abs(np.mean((previous / da.PRIOR_STD) ** 2) - 1.0) <= 0.15
    expected_total = 0.0
    for index, ((matrix, bias), value, observed) in enumerate(
        zip(plate.steps, plate.values, plate.mask, strict=True)
    ):
        # Forecast: the host step of the previous analysis plus the declared
        # process noise drawn from the consumer's (case, step, member) keys.
        draws = np.stack(
            [
                np.asarray(
                    jr.normal(
                        phx.stochastic.state_space_key(
                            key,
                            "ensemble-filter-transition",
                            "plate",
                            index,
                            member=member,
                        ),
                        (size,),
                    )
                )
                for member in range(da.ENSEMBLE)
            ]
        )
        forecast = np.asarray(filtered.forecast_ensembles[index])
        np.testing.assert_allclose(
            forecast,
            previous @ matrix.T + bias + da.PROCESS_STD * draws,
            rtol=0.0,
            atol=1e-11,
        )
        # Host Gaussian innovation likelihood of the forecast sample statistics
        # with the declared sensor noise on the observed rows.
        mean = forecast.mean(axis=0)
        rows = plate.output[observed]
        innovation = value[observed] - rows @ mean
        innovation_covariance = rows @ np.cov(forecast.T) @ rows.T + (
            da.SENSOR_STD**2 * np.eye(rows.shape[0])
        )
        _, logdet = np.linalg.slogdet(2.0 * np.pi * innovation_covariance)
        expected = -0.5 * (
            logdet + innovation @ np.linalg.solve(innovation_covariance, innovation)
        )
        assert float(filtered.incremental_log_likelihood[index]) == pytest.approx(
            expected, rel=1e-8
        )
        expected_total += expected
        analysis_mean, analysis_covariance = da.ensemble_kalman_update(
            forecast, plate.output, value, observed
        )
        previous = np.asarray(filtered.analysis_ensembles[index])
        np.testing.assert_allclose(
            previous.mean(axis=0), analysis_mean, rtol=0.0, atol=1e-11
        )
        np.testing.assert_allclose(
            np.cov(previous.T), analysis_covariance, rtol=0.0, atol=1e-12
        )
    assert float(filtered.cumulative_log_likelihood[-1]) == pytest.approx(
        expected_total, rel=1e-8
    )
    # The ensemble statistics estimate the exact linear-Gaussian filter: the
    # standardized mean error is O(1 / sqrt(N)) per coordinate (observed at this
    # seed: rms 0.19 = 1.5 / sqrt(N), max 0.56 over the 200 coordinates).
    means, deviations, _ = da.kalman_reference(
        plate.steps, plate.output, plate.values, plate.mask
    )
    analysis = np.asarray(filtered.analysis_ensembles).mean(axis=1)
    standardized = np.abs(analysis - means) / deviations
    assert np.max(standardized) <= 1.5
    assert np.sqrt(np.mean(standardized**2)) <= 2.5 / np.sqrt(da.ENSEMBLE)


def test_process_noise_density_is_the_host_gaussian_about_the_accepted_step(
    plate: _Plate,
) -> None:
    problem = da.state_space_problem(
        plate.transition, plate.port, plate.values, plate.mask
    )
    kernel = problem.model.transition
    assert kernel.has_log_density
    size = plate.transition.state_size
    rng = np.random.default_rng(4)
    state = 0.1 * rng.standard_normal(size)
    matrix, bias = plate.steps[1]
    following = matrix @ state + bias + da.PROCESS_STD * rng.standard_normal(size)
    context = problem.step_context(0, 1)
    value = kernel.log_prob(
        jnp.asarray(following), jnp.asarray(state), da.TIMES[0], da.TIMES[1], context
    )
    residual = following - matrix @ state - bias
    expected = -0.5 * np.sum(residual**2) / da.PROCESS_STD**2 - size * (
        np.log(da.PROCESS_STD) + 0.5 * np.log(2.0 * np.pi)
    )
    assert float(value) == pytest.approx(expected, rel=1e-10)


def test_process_noise_density_is_exact_for_an_ill_conditioned_full_rank_factor(
    plate: _Plate,
) -> None:
    problem = da.state_space_problem(
        plate.transition, plate.port, plate.values, plate.mask
    )
    size = plate.transition.state_size
    rng = np.random.default_rng(11)
    rotation, _ = np.linalg.qr(rng.standard_normal((size, size)))
    scales = np.ones(size)
    scales[-1] = 1e-10
    # Full rank (condition 1e10) but L Lᵀ has condition 1e20: its log-determinant
    # is 2 Σ log s_i only when the factor is not formed through L Lᵀ.
    factor = rotation * scales
    kernel = phx.stochastic.CoupledTransitionKernel(plate.transition, noise_factor=factor)
    assert kernel.has_log_density
    context = problem.step_context(0, 1)
    state = jnp.asarray(0.1 * rng.standard_normal(size))
    mean = phx.stochastic.CoupledTransitionKernel(plate.transition).sample(
        jr.key(0), state, da.TIMES[0], da.TIMES[1], context
    )
    assert bool(mean.valid)
    draw = rng.standard_normal(size)
    following = np.asarray(mean.values) + factor @ draw
    value = kernel.log_prob(
        jnp.asarray(following), state, da.TIMES[0], da.TIMES[1], context
    )
    expected = (
        -0.5 * np.sum(draw**2) - np.sum(np.log(scales)) - 0.5 * size * np.log(2.0 * np.pi)
    )
    assert float(value) == pytest.approx(expected, abs=1e-4)


def test_coupled_transition_refuses_a_reversed_step_interval(plate: _Plate) -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="finite target > source"):
        step = plate.transition.step(
            da.TIMES[0],
            0.0,
            jnp.zeros(plate.transition.state_size),
            {heat.CONTROL: da.FORCING},
        )
        np.asarray(step.accepted)


def test_coupled_transition_kernel_has_only_fixed_arrays(plate: _Plate) -> None:
    kernel = phx.stochastic.CoupledTransitionKernel(
        plate.transition,
        noise_factor=da.PROCESS_STD * np.eye(plate.transition.state_size),
    )
    parameters, model_state, _ = phx.partition_parameters(kernel)
    # Refresh parameter values arrive through ``context.args``; the prepared
    # transition and the declared noise are fixed structure, never trained.
    assert jax.tree.leaves(parameters) == []
    assert jax.tree.leaves(model_state) == []


def test_deterministic_or_singular_noise_simulator_has_no_transition_density(
    plate: _Plate,
) -> None:
    deterministic = da.state_space_problem(
        plate.transition, plate.port, plate.values, plate.mask, noise=False
    )
    assert not deterministic.model.transition.has_log_density
    with pytest.raises(ValueError, match="normalized transitions"):
        phx.uq.state_space_path_log_density(
            deterministic, jnp.zeros((da.TIMES.size + 1, plate.transition.state_size))
        )
    # Uncertain wall flux only: a rank-two covariance in the state space.
    flux = plate.reference.flux
    singular = phx.stochastic.CoupledTransitionKernel(
        plate.transition, noise_factor=0.1 * flux
    )
    assert not singular.has_log_density
    with pytest.raises(ValueError, match="no normalized density"):
        singular.log_prob(
            jnp.zeros(flux.shape[0]),
            jnp.zeros(flux.shape[0]),
            0.0,
            0.1,
            deterministic.step_context(0, 0),
        )


def test_failed_coupled_transition_is_a_transition_failure_of_the_filter_step(
    plate: _Plate,
) -> None:
    failing = cpl.prepare_coupled_transition(
        plate.transient,
        step=float(da.TIMES[0]),
        policy=heat.implicit_euler_policy(),
        parameters={heat.CONTROL: jnp.asarray(da.FORCING)},
        tolerance=1e-30,
    )
    problem = da.state_space_problem(failing, plate.port, plate.values, plate.mask)
    state = phx.uq.initialize_ensemble_filter(jr.key(1), problem, ensemble_size=4)
    following, record = phx.uq.ensemble_filter_step(problem, state)
    assert phx.uq.ensemble_filter_status_name(int(record.status)) == "transition_failure"
    assert not bool(record.valid)
    np.testing.assert_array_equal(
        np.asarray(following.ensemble), np.asarray(state.ensemble)
    )
    assert float(following.log_likelihood) == 0.0
    # A nonzero source state: the failed step must roll back to it exactly and
    # keep the solved candidate, the host implicit-Euler step, as evidence.
    source = np.random.default_rng(23).uniform(0.1, 0.5, failing.state_size)
    step = failing.step(0.0, da.TIMES[0], jnp.asarray(source), {heat.CONTROL: da.FORCING})
    assert cpl.coupled_transition_status_name(int(step.status)) == "certificate-failure"
    np.testing.assert_array_equal(np.asarray(step.accepted), source)
    matrix, bias = plate.steps[0]
    np.testing.assert_allclose(
        np.asarray(step.candidate), matrix @ source + bias, rtol=0.0, atol=1e-11
    )


def test_sensor_noise_requires_declared_uncertainty_and_matching_identities(
    plate: _Plate,
) -> None:
    covariance = plate.port.noise_covariance(
        da.sensor_plans(plate.values[0], plate.mask[0])
    )
    np.testing.assert_array_equal(
        np.asarray(covariance), da.SENSOR_STD**2 * np.eye(plate.port.observation_size)
    )
    with pytest.raises(ValueError, match="declares no measurement noise"):
        plate.port.noise_covariance(
            da.sensor_plans(plate.values[0], plate.mask[0], declared=False)
        )
    plans = da.sensor_plans(plate.values[0], plate.mask[0])
    with pytest.raises(ValueError, match="support identities differ"):
        plate.port.noise_covariance(plans[::-1])
