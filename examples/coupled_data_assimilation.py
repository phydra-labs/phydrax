#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Ensemble data assimilation through a finite-element/virtual-element transient.

The hidden state is the native coordinate vector of the coupled heat-conduction
transient of ``examples/_coupled_heat_transient.py`` (P1 triangles next to
degree-1 virtual elements, matching transmission, owners' capacities) under a
known right-wall heat flux. ``prepare_coupled_transition`` prepares one native
implicit-Euler step between arbitrary physical times;
``phydrax.stochastic.CoupledTransitionKernel`` adds declared additive Gaussian
process noise and draws it only from the consumer's semantically addressed key
(case, step, member). The observation model evaluates the three P6 sensor
bindings through ``CoupledObservationPort.location``; its covariance comes from
the declared noise of the ``MeasurementComparisonPlan`` of every sensor
record, whose measurement identities must equal the predicted ones. Samples
are taken at non-uniform physical times, with one sensor dropout.

The example prints

1. the ensemble transform Kalman filter (the existing
   ``phydrax.uq.ensemble_transform_kalman_filter``) against an independent
   linear-Gaussian reference: the exact Kalman filter of the semi-discrete
   system and sensor rows assembled on the host from the two meshes
   (``_coupled_heat_transient.host_semidiscrete``), with the same process and
   sensor noise;
2. the first analysis step against the exact Kalman update of its own forecast
   ensemble (the deterministic square-root transform reproduces it to
   roundoff);
3. the ETKF log-likelihood, which is an ensemble Gaussian approximation; the
   exact Kalman value exists here only because the coupled step is affine and
   the noise Gaussian, and no such claim is made for a nonlinear simulator;
4. the refusals of a transition density for a deterministic simulator and of
   sensor data that declare no measurement noise.

It raises if any accepted-step or agreement check fails.
"""

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from examples import _coupled_heat_transient as heat
from phydrax.solver import coupling as cpl


jax.config.update("jax_enable_x64", True)

M = phx.measurement
CELLS = 4
TIMES = np.asarray([0.1, 0.25, 0.35, 0.5, 0.7])
FORCING = np.asarray([0.4, -0.3])
PROCESS_STD = 0.01
PRIOR_STD = 0.05
SENSOR_STD = 0.01
ENSEMBLE = 64
DROPOUT = (2, 2)  # (step, sensor): the upper insulator sensor misses one sample
SEED = 20260928


def assimilation_model(
    cells: int, /
) -> tuple[
    cpl.PreparedCoupledTransient,
    cpl.PreparedCoupledTransition,
    cpl.CoupledObservationPort,
]:
    """The transient, its autonomous transition (known flux), and sensor port."""
    transient = heat.coupled_heat_transient(cells)
    transition = cpl.prepare_coupled_transition(
        transient,
        step=float(TIMES[0]),
        policy=heat.implicit_euler_policy(),
        parameters={heat.CONTROL: jnp.asarray(FORCING)},
    )
    return transient, transition, transition.observation_port(heat.SENSOR_BINDINGS)


def host_step(
    semidiscrete: heat.HostSemidiscrete, step: float, /
) -> tuple[np.ndarray, np.ndarray]:
    """``x+ = A x + c`` of one implicit-Euler step under the known flux."""
    matrix = semidiscrete.capacity + step * semidiscrete.stiffness
    return (
        np.linalg.solve(matrix, semidiscrete.capacity),
        np.linalg.solve(
            matrix, step * (semidiscrete.source + semidiscrete.flux @ FORCING)
        ),
    )


def sensor_matrix(port: cpl.CoupledObservationPort, size: int, /) -> np.ndarray:
    """The port's ``H``: the point observation of a zero-lift field is linear."""
    parameters = {heat.CONTROL: jnp.asarray(FORCING)}
    return np.asarray(
        jax.jacfwd(lambda state: port.values(0.0, state, parameters))(jnp.zeros(size))
    )


def sensor_plans(
    values: np.ndarray, mask: np.ndarray, /, *, declared: bool = True
) -> tuple[phx.observation.MeasurementComparisonPlan, ...]:
    """One comparison plan per sensor binding, with declared (or absent) noise."""
    plans = []
    offset = 0
    for binding_id in heat.SENSOR_BINDINGS:
        support = heat.SENSOR_SUPPORTS[binding_id]
        count = support.points.shape[0]
        part = values[offset : offset + count]
        field = M.QuantityField(
            f"{binding_id}-record",
            heat.TEMPERATURE,
            M.ValueLayout.scalar(),
            support,
            heat.POINT,
            part,
            valid_mask=mask[offset : offset + count],
            uncertainty=(
                M.IndependentStandardUncertainty(
                    np.full(part.shape, SENSOR_STD), heat.TEMPERATURE.unit
                )
                if declared
                else None
            ),
        ).prepare()
        plans.append(phx.observation.MeasurementComparisonPlan(field))
        offset += count
    return tuple(plans)


def synthetic_record(
    steps: list[tuple[np.ndarray, np.ndarray]], output: np.ndarray, size: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Host truth ``x_k`` and sensor samples ``y_k`` (seeded, independent of JAX)."""
    rng = np.random.default_rng(SEED)
    state = PRIOR_STD * rng.standard_normal(size)
    samples = []
    for matrix, bias in steps:
        state = matrix @ state + bias + PROCESS_STD * rng.standard_normal(size)
        samples.append(output @ state + SENSOR_STD * rng.standard_normal(output.shape[0]))
    mask = np.ones((TIMES.size, output.shape[0]), dtype=np.bool_)
    mask[DROPOUT] = False
    return np.where(mask, np.asarray(samples), 0.0), mask


def kalman_reference(
    steps: list[tuple[np.ndarray, np.ndarray]],
    output: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact linear-Gaussian filter means, standard deviations, and log-likelihood."""
    size = output.shape[1]
    mean = np.zeros(size)
    covariance = PRIOR_STD**2 * np.eye(size)
    means, deviations = [], []
    log_likelihood = 0.0
    for (matrix, bias), value, observed in zip(steps, values, mask, strict=True):
        mean = matrix @ mean + bias
        covariance = matrix @ covariance @ matrix.T + PROCESS_STD**2 * np.eye(size)
        rows = output[observed]
        innovation = value[observed] - rows @ mean
        innovation_covariance = rows @ covariance @ rows.T + SENSOR_STD**2 * np.eye(
            rows.shape[0]
        )
        gain = np.linalg.solve(innovation_covariance, rows @ covariance).T
        mean = mean + gain @ innovation
        covariance = covariance - gain @ rows @ covariance
        _, logdet = np.linalg.slogdet(2.0 * np.pi * innovation_covariance)
        log_likelihood += -0.5 * (
            logdet + innovation @ np.linalg.solve(innovation_covariance, innovation)
        )
        means.append(mean)
        deviations.append(np.sqrt(np.diag(covariance)))
    return np.asarray(means), np.asarray(deviations), log_likelihood


def state_space_problem(
    transition: cpl.PreparedCoupledTransition,
    port: cpl.CoupledObservationPort,
    values: np.ndarray,
    mask: np.ndarray,
    /,
    *,
    noise: bool = True,
) -> phx.stochastic.StateSpaceProblem:
    """The coupled transition, sensor port, and record as a state-space problem."""
    size = transition.state_size
    covariances = [
        np.asarray(port.noise_covariance(sensor_plans(value, observed)))
        for value, observed in zip(values, mask, strict=True)
    ]
    if any(not np.array_equal(item, covariances[0]) for item in covariances):
        raise RuntimeError("Sensor noise is declared time-invariant.")
    kernel = phx.stochastic.CoupledTransitionKernel(
        transition, noise_factor=PROCESS_STD * np.eye(size) if noise else None
    )
    observation = phx.stochastic.GaussianObservationModel(
        port.location,
        covariances[0],
        state_shape=(size,),
        observation_shape=(port.observation_size,),
        observation_id="coupled-heat-sensors",
    )
    prior = phx.stochastic.GaussianStatePrior(
        jnp.zeros((size,)), PRIOR_STD**2 * jnp.eye(size), state_shape=(size,)
    )
    model = phx.stochastic.StateSpaceModel(
        prior, kernel, observation, model_id="coupled-heat-assimilation"
    )
    sequence = phx.stochastic.ObservationSequence(
        TIMES,
        values,
        observation_mask=mask,
        case_ids=("plate",),
        sequence_id="coupled-heat-sensors",
    )
    return phx.stochastic.StateSpaceProblem(
        model,
        sequence,
        initial_time=0.0,
        args={heat.CONTROL: jnp.asarray(FORCING)},
        problem_id="coupled-heat-assimilation",
    )


def check(condition: bool, message: str, /) -> None:
    if not condition:
        raise RuntimeError(message)


def ensemble_kalman_update(
    forecast: np.ndarray, output: np.ndarray, value: np.ndarray, observed: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Kalman update of an ensemble's sample mean and covariance."""
    mean = forecast.mean(axis=0)
    covariance = np.cov(forecast.T)
    rows = output[observed]
    innovation_covariance = rows @ covariance @ rows.T + SENSOR_STD**2 * np.eye(
        rows.shape[0]
    )
    gain = np.linalg.solve(innovation_covariance, rows @ covariance).T
    return (
        mean + gain @ (value[observed] - rows @ mean),
        covariance - gain @ rows @ covariance,
    )


def main() -> None:
    transient, transition, port = assimilation_model(CELLS)
    size = transition.state_size
    semidiscrete = heat.host_semidiscrete(transient, CELLS)
    durations = np.diff(np.concatenate(([0.0], TIMES)))
    steps = [host_step(semidiscrete, float(step)) for step in durations]
    output = semidiscrete.sensors
    check(
        bool(np.allclose(sensor_matrix(port, size), output, rtol=0.0, atol=1e-14)),
        "The sensor port differs from the host point evaluation.",
    )
    values, mask = synthetic_record(steps, output, size)
    problem = state_space_problem(transition, port, values, mask)
    print(
        f"state: {size} native coordinates; {TIMES.size} samples at t = {TIMES}; "
        f"sensor {DROPOUT[1]} masked at step {DROPOUT[0]}; transition density: "
        f"{problem.model.transition.has_log_density}"
    )

    filtered = phx.uq.ensemble_transform_kalman_filter(
        jax.random.key(SEED), problem, ensemble_size=ENSEMBLE, raise_on_failure=True
    )
    means, deviations, log_likelihood = kalman_reference(steps, output, values, mask)
    analysis = np.asarray(filtered.analysis_ensembles).mean(axis=1)
    standardized = np.max(np.abs(analysis - means) / deviations, axis=1)
    print(
        "1. ETKF vs exact Kalman filter, max |mean difference| / KF std per step: "
        + ", ".join(f"{value:.2f}" for value in standardized)
    )
    check(bool(np.all(standardized <= 6.0 / np.sqrt(ENSEMBLE) * 3.0)), "ETKF drifted.")

    first_mean, first_covariance = ensemble_kalman_update(
        np.asarray(filtered.forecast_ensembles[0]), output, values[0], mask[0]
    )
    ensemble = np.asarray(filtered.analysis_ensembles[0])
    update_error = max(
        float(np.max(np.abs(ensemble.mean(axis=0) - first_mean))),
        float(np.max(np.abs(np.cov(ensemble.T) - first_covariance))),
    )
    print(
        "2. first analysis vs exact Kalman update of its forecast ensemble: "
        f"max difference {update_error:.2e}"
    )
    check(update_error <= 1e-10, "The ETKF analysis is not the ensemble Kalman update.")

    approximate = float(filtered.cumulative_log_likelihood[-1])
    print(
        f"3. ETKF log-likelihood (ensemble Gaussian approximation) {approximate:.3f}; "
        f"exact linear-Gaussian Kalman log-likelihood {log_likelihood:.3f}"
    )

    deterministic = state_space_problem(transition, port, values, mask, noise=False)
    try:
        phx.uq.state_space_path_log_density(
            deterministic, jnp.zeros((TIMES.size + 1, size))
        )
    except ValueError as error:
        print(f"4a. deterministic simulator refused: {error}")
    else:
        raise RuntimeError("A deterministic simulator has no transition density.")
    try:
        port.noise_covariance(sensor_plans(values[0], mask[0], declared=False))
    except ValueError as error:
        print(f"4b. unquantified sensor data refused: {error}")
    else:
        raise RuntimeError("Data without declared noise define no likelihood.")


if __name__ == "__main__":
    main()
