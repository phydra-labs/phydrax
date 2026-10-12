"""Consumer-visible inference and physical nuisance confounding."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from examples.nuclear_fuel_calibration import (
    build_synthetic_calibration,
    prepare_synthetic_inference_rod,
)
from phydrax.applications.nuclear_fuel import (
    evaluate_fuel_identifiability,
    fuel_parameterization,
    FuelCalibrationParameters,
    FuelCalibrationPlan,
    FuelDenseSignalJacobian,
    FuelFixedGrid,
    FuelObservationPlan,
    FuelRodSpecification,
    FuelSensor,
    prepare_fuel_least_squares,
    replay_fuel_grid,
)
from phydrax.linalg import MaterializationPolicy, RankPolicy
from phydrax.observation import CoordinateLayout
from phydrax.optim import (
    composite_least_squares,
    GeneralizedGaussNewton,
    OptimizationTermination,
)


def test_pressure_sensor_gain_and_fill_inventory_are_confounded() -> None:
    with jax.enable_x64():
        # p_meas = gain * n_fill * R * T / V. At unit gain and inventory,
        # the two dimensionless scale derivatives coincide at every T.
        temperature = jnp.asarray([300.0, 450.0, 600.0], dtype=jnp.float64)
        sensitivity = temperature / 300.0
        jacobian = jnp.stack((sensitivity, sensitivity), axis=1)
        parameters = CoordinateLayout(("rod:fill-moles-scale", "sensor:pressure-gain"))
        observations = CoordinateLayout(
            ("pressure:300K", "pressure:450K", "pressure:600K")
        )
        report = evaluate_fuel_identifiability(
            jacobian,
            parameters,
            observations,
            rank_policy=RankPolicy(require_full_rank=False),
        )
    assert bool(report.successful)
    assert np.isinf(np.asarray(report.condition_number))
    # Exactly one weak direction (rank one), and it is the data null space.
    weak = np.asarray(report.weak_directions)
    np.testing.assert_allclose(np.linalg.norm(weak, axis=1), [0.0, 1.0], atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(jacobian) @ weak[1], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        report.normalized_column_correlation, np.ones((2, 2)), atol=1.0e-14
    )


@dataclass(frozen=True)
class _SyntheticFit:
    plan: FuelCalibrationPlan
    truth: Array
    heldout: FuelObservationPlan
    heldout_data: Array
    successful: bool
    status: int
    physical: FuelCalibrationParameters


@pytest.fixture(scope="module")
def synthetic_fit() -> _SyntheticFit:
    """Prior-bearing fit of synthetic noisy data from a rod with known truth."""
    with jax.enable_x64():
        plan, truth, heldout, heldout_data = build_synthetic_calibration(seed=17)
        problem = prepare_fuel_least_squares(
            plan,
            include_prior=True,
            signal_derivatives=FuelDenseSignalJacobian(
                materialization=MaterializationPolicy(max_entries=64, max_bytes=512),
                maximum_working_set_bytes=64 * 1024 * 1024,
            ),
        )
        fit = composite_least_squares(
            problem,
            plan.parameter_space.initial,
            method=GeneralizedGaussNewton(
                curvature_materialization=MaterializationPolicy(
                    max_entries=64, max_bytes=512
                )
            ),
            termination=OptimizationTermination(
                absolute_optimality=1.0e-6, relative_optimality=1.0e-7, maximum_steps=24
            ),
        )
        return _SyntheticFit(
            plan=plan,
            truth=truth,
            heldout=heldout,
            heldout_data=heldout_data,
            successful=bool(fit.successful),
            status=int(fit.status),
            physical=plan.parameter_space.constrain(fit.parameters),
        )


def test_synthetic_fit_recovers_shared_coefficient_with_rod_and_sensor_nuisances(
    synthetic_fit: _SyntheticFit,
) -> None:
    assert synthetic_fit.successful, synthetic_fit.status
    with jax.enable_x64():
        evaluation = synthetic_fit.plan.evaluate_physical(synthetic_fit.physical)
    assert bool(evaluation.successful)
    np.testing.assert_allclose(
        synthetic_fit.physical.shared_physical, synthetic_fit.truth, rtol=0.05, atol=0.0
    )


def test_synthetic_fit_predicts_heldout_synthetic_rod_within_sensor_noise(
    synthetic_fit: _SyntheticFit,
) -> None:
    assert synthetic_fit.successful, synthetic_fit.status
    heldout = synthetic_fit.heldout
    with jax.enable_x64():
        bundle = synthetic_fit.plan.parameterization.bundle(
            heldout.prepared.admission.bundle, synthetic_fit.physical.shared_physical
        )
        replay = replay_fuel_grid(heldout.prepared, heldout.grid, parameters=bundle)
        predicted = heldout.evaluate_states(replay.states, parameters=bundle)
    assert bool(replay.successful) and bool(predicted.successful)
    # Independent synthetic rod/batch and noise, never a scientific validation
    # claim. Explicit sensor units own the held-out residual tolerance.
    count = heldout.layout.size // 2
    scales = np.asarray([0.02] * count + [20.0] * count)
    standardized = (
        np.asarray(predicted.product.values - synthetic_fit.heldout_data) / scales
    )
    assert float(np.mean(standardized**2)) < 9.0


def test_replay_sensor_loss_reverse_derivative_matches_independent_accepted_trajectories() -> (
    None
):
    with jax.enable_x64():
        prepared = prepare_synthetic_inference_rod()
        grid = FuelFixedGrid((0.0, 5.0), prepared.prepared_id)
        specification = prepared.admission.specification
        assert isinstance(specification, FuelRodSpecification)
        position = specification.rod.stack_length / 2
        observation = FuelObservationPlan(
            prepared,
            grid,
            (
                FuelSensor(
                    "centerline", "thermocouple-temperature", axial_position=position
                ),
                FuelSensor("plenum", "in-pile-pressure", axial_position=position),
            ),
            (("centerline", 5.0), ("plenum", 5.0)),
        )
        parameterization, values = fuel_parameterization(
            prepared.admission.bundle, ("fuel-phonon-resistivity",)
        )
        target = jnp.asarray([300.0, 103000.0], dtype=jnp.float64)
        scales = jnp.asarray([1.0, 1000.0], dtype=jnp.float64)

        def objective(parameters: Array) -> tuple[Array, Array]:
            bundle = parameterization.bundle(prepared.admission.bundle, parameters)
            replay = replay_fuel_grid(prepared, grid, parameters=bundle)
            observed = observation.evaluate_states(replay.states, parameters=bundle)
            residual = (observed.product.values - target) / scales
            return jnp.sum(residual * residual), replay.successful & observed.successful

        differentiated = eqx.filter_jit(jax.value_and_grad(objective, has_aux=True))
        (_, accepted), gradient = differentiated(values)
        assert bool(accepted)
        solved = eqx.filter_jit(objective)
        step = values[0] * 0.002
        (plus, plus_accepted), (minus, minus_accepted) = (
            solved(values + step),
            solved(values - step),
        )
        assert bool(plus_accepted) and bool(minus_accepted)
        reference = (plus - minus) / (2.0 * step)
        assert abs(float(reference)) > 1.0
        np.testing.assert_allclose(
            np.asarray(gradient[0]), np.asarray(reference), rtol=0.01, atol=0.0
        )
