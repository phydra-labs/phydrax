"""Fit actual coupled-rod observables to explicitly synthetic correlated data.

This is numerical inference demonstration, not experimental validation, source-
range qualification, or nuclear-fuel scientific release. The source equations
are evaluated under their explicit equation-reproduction policy; source support
and physical acceptance remain separate native evidence.
"""

from __future__ import annotations

import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import NDArray

from phydrax.applications.nuclear_fuel import (
    admit_fuel_rod,
    CladdingSpecification,
    evaluate_fuel_calibration_identifiability,
    fuel_parameterization,
    FuelCalibrationParameters,
    FuelCalibrationPlan,
    FuelCalibrationRod,
    FuelDenseSignalJacobian,
    FuelFixedGrid,
    FuelObservationPlan,
    FuelParameterization,
    FuelRodSpecification,
    FuelSensor,
    FuelSensorParameters,
    HeliumTransportTable,
    IrradiationHistory,
    prepare_fuel_least_squares,
    prepare_fuel_rod,
    PreparedFuelRod,
    PrescribedWallTemperature,
    replay_fuel_grid,
    RodSpecification,
)
from phydrax.linalg import MaterializationPolicy, RankPolicy
from phydrax.linalg.svd import require_exact_svd_rank
from phydrax.observation import CholeskyCovarianceAction, CoordinateLayout
from phydrax.optim import (
    composite_least_squares,
    GeneralizedGaussNewton,
    OptimizationTermination,
)
from phydrax.qualification import CampaignRole, ScientificCampaign, ScientificCase
from phydrax.uq import (
    AffineFlowBijector,
    ExpBijector,
    Normal,
    ParameterSpace,
    TruncatedNormal,
)
from tools.nuclear_fuel_qualification import numerical_rod_inputs


def prepare_synthetic_inference_rod(
    *,
    fill_pressure: float = 0.103e6,
    include_cooldown: bool = False,
    initial_oxide_thickness: float = 0.0,
    fabricated_gap: float | None = None,
    helium_transport: HeliumTransportTable | None = None,
    maximum_coupling_evaluations: int = 4,
) -> PreparedFuelRod:
    """Small actual coupled rod with a low-power smooth heating trajectory.

    ``maximum_coupling_evaluations`` is the gas/thermomechanics coupling budget
    of ``prepare_fuel_rod``; strongly coupled synthetic gas tables need more
    Picard passes than the default.
    """
    specification, _, _, bundle = numerical_rod_inputs(
        mode_count=4, axial_layers=1, helium_transport=helium_transport
    )
    rod = RodSpecification(
        stack_length=specification.rod.stack_length,
        plenum_length=specification.rod.plenum_length,
        fill_pressure=fill_pressure,
        fill_temperature=300.0,
    )
    cladding = specification.cladding
    inner_radius = (
        cladding.inner_radius
        if fabricated_gap is None
        else specification.pellet.radius + fabricated_gap
    )
    cladding = CladdingSpecification(
        inner_radius=inner_radius,
        outer_radius=cladding.outer_radius,
        roughness=cladding.roughness,
        cold_work=cladding.cold_work,
        initial_oxide_thickness=initial_oxide_thickness,
    )
    specification = FuelRodSpecification(specification.pellet, cladding, rod)
    times = np.asarray([0.0, 5.0, 10.0, 20.0, 30.0, 60.0], dtype=np.float64)
    power = np.asarray([0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0], dtype=np.float64)[
        :, None
    ]
    if include_cooldown:
        times = np.concatenate(
            (times, np.asarray([65.0, 180.0, 360.0], dtype=np.float64))
        )
        power = np.concatenate((power, np.zeros((3, 1), dtype=np.float64)), axis=0)
    wall = np.full_like(power, 300.0)
    history = IrradiationHistory(
        times=times,
        zone_edges=np.asarray([0.0, specification.rod.stack_length]),
        linear_power=power,
        fast_flux=power * 5.0e13,
        coolant_inlet_temperature=wall[:, 0],
        coolant_pressure=np.full_like(times, 0.103e6),
        coolant_mass_flux=np.full_like(times, 3000.0),
    )
    coolant = PrescribedWallTemperature(times, wall)
    admission = admit_fuel_rod(
        specification,
        history,
        coolant,
        bundle=bundle,
        reduction="layered-radial",
        axial_interaction="frictionless",
    )
    return prepare_fuel_rod(
        admission,
        fuel_cells=2,
        cladding_cells=2,
        quadrature_order=2,
        maximum_coupling_evaluations=maximum_coupling_evaluations,
    )


def inference_observation_plan(prepared: PreparedFuelRod) -> FuelObservationPlan:
    grid = FuelFixedGrid(
        tuple(float(time) for time in np.asarray(prepared.admission.history.times)),
        prepared.prepared_id,
    )
    specification = prepared.admission.specification
    if not isinstance(specification, FuelRodSpecification):
        raise TypeError(
            "Sensor preparation requires the original scientific fabrication specification."
        )
    position = specification.rod.stack_length / 2.0
    sensors = (
        FuelSensor(
            "centerline-tc",
            "thermocouple-temperature",
            axial_position=position,
            response_time=0.8,
            drift_valid_window=(0.0, 60.0),
        ),
        FuelSensor(
            "plenum-transducer",
            "in-pile-pressure",
            axial_position=position,
            drift_valid_window=(0.0, 60.0),
        ),
    )
    samples = tuple(
        (sensor.sensor_id, time) for sensor in sensors for time in grid.times[1:]
    )
    return FuelObservationPlan(prepared, grid, sensors, samples)


def _synthetic_parameter_space(
    nominal: Array, sensor_layout: CoordinateLayout, /
) -> ParameterSpace:
    """Own all eight search coordinates, transforms, and factorized SI priors."""
    zero = jnp.zeros((2,), dtype=jnp.float64)
    sensor_coordinates = FuelSensorParameters(
        sensor_layout, offset=zero, drift=zero, gain=zero
    )
    initial = FuelCalibrationParameters(
        jnp.log(nominal), jnp.zeros((1,), dtype=jnp.float64), (sensor_coordinates,)
    )
    bijectors = jax.tree_util.tree_map(lambda _: ExpBijector(), initial)
    offset_scale = jnp.asarray([0.01, 10.0], dtype=jnp.float64)
    drift_scale = jnp.asarray([0.0001, 0.1], dtype=jnp.float64)
    gain_scale = jnp.asarray([1.0e-5, 1.0e-4], dtype=jnp.float64)
    bijectors = eqx.tree_at(
        lambda point: (
            point.sensor_nuisance[0].offset,
            point.sensor_nuisance[0].drift,
            point.sensor_nuisance[0].gain,
        ),
        bijectors,
        (
            AffineFlowBijector(zero, offset_scale),
            AffineFlowBijector(zero, drift_scale),
            AffineFlowBijector(jnp.ones_like(zero), gain_scale),
        ),
    )
    priors = eqx.tree_at(
        lambda point: (
            point.shared_physical,
            point.fabrication_fill_pressure_scale,
            point.sensor_nuisance[0].offset,
            point.sensor_nuisance[0].drift,
            point.sensor_nuisance[0].gain,
        ),
        initial,
        (
            TruncatedNormal(
                nominal,
                nominal * 0.5,
                jnp.zeros_like(nominal),
                jnp.full_like(nominal, jnp.inf),
            ),
            TruncatedNormal(
                jnp.ones_like(initial.fabrication_fill_pressure_scale),
                jnp.full_like(initial.fabrication_fill_pressure_scale, 0.02),
                jnp.zeros_like(initial.fabrication_fill_pressure_scale),
                jnp.full_like(initial.fabrication_fill_pressure_scale, jnp.inf),
            ),
            Normal(zero, offset_scale),
            Normal(zero, drift_scale),
            TruncatedNormal(
                jnp.ones_like(zero),
                gain_scale,
                jnp.zeros_like(zero),
                jnp.full_like(zero, jnp.inf),
            ),
        ),
    )
    return ParameterSpace(initial, priors=priors, bijectors=bijectors)


def _synthetic_sensor_covariance(
    observation: FuelObservationPlan, /
) -> tuple[CholeskyCovarianceAction, NDArray[np.float64]]:
    count = observation.layout.size
    scales = np.asarray([0.02] * (count // 2) + [20.0] * (count // 2), dtype=np.float64)
    correlation = 0.2 ** np.abs(np.arange(count)[:, None] - np.arange(count)[None, :])
    lower = np.linalg.cholesky(scales[:, None] * correlation * scales[None, :])
    covariance = CholeskyCovarianceAction(lower, observation.layout)
    return covariance, lower


def _synthetic_calibration_plan(
    observation: FuelObservationPlan,
    parameterization: FuelParameterization,
    nominal: Array,
    data: ArrayLike,
    covariance: CholeskyCovarianceAction,
    /,
) -> FuelCalibrationPlan:
    """Bind actual synthetic data to the declared rod/batch roles and SI priors."""
    cases = (
        ScientificCase(
            "synthetic-fit-rod",
            "synthetic-fit-rod",
            "uo2-zircaloy",
            "low-power-heating",
            "synthetic-fabrication-event-a",
            "synthetic-fabrication-batch-a",
            ("synthetic-generated-training:not-experiment",),
        ),
        ScientificCase(
            "synthetic-heldout-rod",
            "synthetic-heldout-rod",
            "uo2-zircaloy",
            "low-power-heating",
            "synthetic-fabrication-event-b",
            "synthetic-fabrication-batch-b",
            ("synthetic-generated-heldout:not-experiment",),
        ),
    )
    campaign = ScientificCampaign(
        cases,
        (
            CampaignRole("calibration", (cases[0].case_id,)),
            CampaignRole("locked_evaluation", (cases[1].case_id,)),
        ),
        criteria_ids=("synthetic-inference-demonstration:not-scientific-validation",),
    )
    space = _synthetic_parameter_space(nominal, observation.sensor_layout)
    plan = FuelCalibrationPlan(
        (FuelCalibrationRod(observation, data, covariance, case=cases[0]),),
        parameterization,
        space,
        campaign,
        prior_source_ids=(
            "synthetic-truncated-normal-physical-gain-and-normal-offset-drift-priors:declared-SI-scales:not-validation",
        ),
    )
    return plan


def build_synthetic_calibration(
    *, seed: int = 17
) -> tuple[FuelCalibrationPlan, Array, FuelObservationPlan, Array]:
    """Independent noise, explicit rod/batch roles, and nonvalidation provenance."""
    prepared = prepare_synthetic_inference_rod()
    heldout = prepare_synthetic_inference_rod(fill_pressure=0.115e6)
    observation = inference_observation_plan(prepared)
    heldout_observation = inference_observation_plan(heldout)
    parameterization, nominal = fuel_parameterization(
        prepared.admission.bundle, ("fuel-phonon-resistivity",)
    )
    truth = nominal * 1.12
    true_bundle = parameterization.bundle(prepared.admission.bundle, truth)
    replay = replay_fuel_grid(prepared, observation.grid, parameters=true_bundle)
    heldout_bundle = parameterization.bundle(heldout.admission.bundle, truth)
    heldout_replay = replay_fuel_grid(
        heldout, heldout_observation.grid, parameters=heldout_bundle
    )
    if not bool(replay.successful) or not bool(heldout_replay.successful):
        raise RuntimeError(
            "Synthetic actual-rod truth trajectory failed native physical acceptance."
        )
    predicted = observation.evaluate_states(replay.states, parameters=true_bundle)
    heldout_predicted = heldout_observation.evaluate_states(
        heldout_replay.states, parameters=heldout_bundle
    )
    if not bool(predicted.successful) or not bool(heldout_predicted.successful):
        raise RuntimeError(
            "Synthetic actual-rod sensor evaluation is outside its declared validity."
        )
    count = observation.layout.size
    covariance, lower = _synthetic_sensor_covariance(observation)
    training_key, heldout_key = jax.random.split(jax.random.key(seed))
    data = predicted.product.values + covariance.lower_cholesky @ jax.random.normal(
        training_key, (count,), dtype=jnp.float64
    )
    heldout_data = heldout_predicted.product.values + jnp.asarray(
        lower
    ) @ jax.random.normal(heldout_key, (count,), dtype=jnp.float64)
    plan = _synthetic_calibration_plan(
        observation, parameterization, nominal, data, covariance
    )
    return plan, truth, heldout_observation, heldout_data


def run_synthetic_calibration(
    *, maximum_optimizer_steps: int = 24, seed: int = 17
) -> dict[str, object]:
    plan, truth, heldout_observation, heldout_data = build_synthetic_calibration(
        seed=seed
    )
    problem = prepare_fuel_least_squares(
        plan,
        include_prior=True,
        signal_derivatives=FuelDenseSignalJacobian(
            materialization=MaterializationPolicy(max_entries=64, max_bytes=512),
            maximum_working_set_bytes=64 * 1024 * 1024,
        ),
    )
    fitted = composite_least_squares(
        problem,
        plan.parameter_space.initial,
        method=GeneralizedGaussNewton(
            curvature_materialization=MaterializationPolicy(max_entries=64, max_bytes=512)
        ),
        termination=OptimizationTermination(
            absolute_optimality=1.0e-6,
            relative_optimality=1.0e-7,
            maximum_steps=maximum_optimizer_steps,
        ),
    )
    physical = plan.parameter_space.constrain(fitted.parameters)
    evaluation = plan.evaluate_physical(physical)
    rank = evaluate_fuel_calibration_identifiability(
        plan, fitted.parameters, rank_policy=RankPolicy(require_full_rank=False)
    )
    heldout_bundle = plan.parameterization.bundle(
        heldout_observation.prepared.admission.bundle, physical.shared_physical
    )
    replay = replay_fuel_grid(
        heldout_observation.prepared, heldout_observation.grid, parameters=heldout_bundle
    )
    predicted = heldout_observation.evaluate_states(
        replay.states, parameters=heldout_bundle
    )
    if (
        not bool(fitted.successful)
        or not bool(evaluation.successful)
        or not bool(replay.successful)
        or not bool(predicted.successful)
    ):
        raise RuntimeError(
            f"Actual-rod synthetic inference failed: optimization_status={fitted.status}, replay_status={tuple(int(attempt.status) for attempt in replay.attempts)}"
        )
    extrapolated = sorted(
        {
            source
            for trajectory in evaluation.replays + (replay,)
            for attempt in trajectory.attempts
            for source, supported in zip(
                attempt.source_support.source_ids,
                np.asarray(attempt.source_support.supported),
                strict=True,
            )
            if not supported
        }
    )
    return {
        "qualification": "synthetic-data numerical inference; not experimental validation or scientific release",
        "optimizer_status": int(fitted.status),
        "fitted_phonon_resistivity_m_K_per_W": float(physical.shared_physical[0]),
        "synthetic_truth_m_K_per_W": float(truth[0]),
        "physical_parameter_units": plan.parameterization.units,
        "initial_parameter_fingerprint": plan.initial_parameter_fingerprint,
        "fitted_parameter_fingerprint": plan.parameter_fingerprint(physical),
        "static_bundle_id_semantics": "admitted nominal equation/source identity; actual refitted coefficients are fingerprinted separately",
        "prior_support": {
            "phonon_resistivity": "(0,infinity) m*K/W",
            "fill_pressure_scale": "(0,infinity)",
            "sensor_offset_and_drift": "real, declared per-sensor SI scales",
        },
        "whitened_residual_norm": float(jnp.linalg.norm(evaluation.whitened_residual)),
        "data_only_rank": int(require_exact_svd_rank(rank.decomposition)),
        "data_only_rank_available": bool(rank.decomposition.rank_evidence.available),
        "data_only_rank_deterministic_exact": bool(
            rank.decomposition.rank_evidence.deterministic_exact
        ),
        "data_only_rank_full_spectrum": rank.decomposition.rank_evidence.full_spectrum,
        "data_only_svd_status": int(rank.decomposition.status),
        "branchwise_valid_rods": np.asarray(evaluation.branchwise_valid).tolist(),
        "heldout_sensor_residual_SI": np.asarray(
            predicted.product.values - heldout_data
        ).tolist(),
        "heldout_sensor_units": heldout_observation.units,
        "heldout_sensor_coordinates": heldout_observation.layout.labels,
        "campaign_id": plan.campaign.campaign_id,
        "grid_id": heldout_observation.grid.grid_id,
        "source_extrapolated_ids": extrapolated,
        "source_support_claim": False,
        "scientific_release_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-optimizer-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    print(
        json.dumps(
            run_synthetic_calibration(
                maximum_optimizer_steps=args.maximum_optimizer_steps, seed=args.seed
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
