"""Coupled-rod calibration on observation, UQ, and least-squares optimization."""

from __future__ import annotations

from typing import assert_never, Literal, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import positive_integer
from ...linalg import (
    DenseLinearOperator,
    LinearizationPolicy,
    MaterializationPolicy,
    prepare_linearization,
    PreparedLinearization,
    PyTreeSpace,
)
from ...observation import (
    CholeskyCovarianceAction,
    CoordinateLayout,
    CorrelatedGaussianPlan,
    LinearObservationPlan,
)
from ...optim import CompositeLeastSquaresProblem, NonlinearLeastSquaresProblem
from ...qualification import ScientificCampaign, ScientificCase
from ...typing import Dim, Float, parse, Scope, Size
from ...uq import ParameterSpace, PosteriorProblem
from ._bundles import FuelModelBundle
from ._observables import FuelObservationPlan, FuelSensorParameters
from ._problem import PreparedFuelRod
from ._sensitivity import (
    FuelParameterDim,
    FuelParameterization,
    FuelReplayResult,
    FuelSensitivityPlan,
    replay_fuel_grid,
    stack_trees,
    unstack_tree,
)
from ._state import FuelRodState


class FuelCalibrationRodDim(Dim):
    """Independently fabricated calibration rods."""


class FuelCalibrationParameters(StrictModule):
    """Shared source-law coordinates, per-rod fill scales, and sensor nuisances.

    The fill-pressure scale is a per-rod fabrication uncertainty. Sensor
    gain/offset/drift leaves carry their detector layout; their priors are
    distinct from the correlated stochastic measurement covariance.
    """

    shared_physical: Array
    fabrication_fill_pressure_scale: Array
    sensor_nuisance: tuple[FuelSensorParameters, ...]


class FuelCalibrationRod(StrictModule):
    """One observed rod: sensor plan, Gaussian likelihood, and campaign case.

    The likelihood owns the sensor/time covariance that whitens the residual.
    """

    observation: FuelObservationPlan
    likelihood: CorrelatedGaussianPlan
    case: ScientificCase = eqx.field(static=True)

    def __init__(
        self,
        observation: FuelObservationPlan,
        data: ArrayLike,
        covariance: CholeskyCovarianceAction,
        /,
        *,
        case: ScientificCase,
    ) -> None:
        if not isinstance(covariance, CholeskyCovarianceAction):
            raise TypeError(
                "Calibration rod covariance must be a CholeskyCovarianceAction."
            )
        if not isinstance(case, ScientificCase):
            raise TypeError("Calibration rod case must be a ScientificCase.")
        if covariance.layout.layout_id != observation.layout.layout_id:
            raise ValueError(
                "Sensor covariance layout must equal the observation sensor/time layout."
            )
        response = LinearObservationPlan(
            np.eye(observation.layout.size, dtype=np.float64),
            observation.layout,
            observation.layout,
        )
        self.observation = observation
        self.likelihood = CorrelatedGaussianPlan(data, response, covariance)
        self.case = case


class FuelCalibrationEvaluation(StrictModule):
    """Per-rod predictions, whitened residual, and replay status at one parameter point."""

    prediction: tuple[Array, ...]
    whitened_residual: Array
    log_likelihood: Array
    replay_success: Array
    branchwise_valid: Array
    successful: Array
    replays: tuple[FuelReplayResult, ...]
    physical_parameters: FuelCalibrationParameters


class FuelSignalJacobianActions(StrictModule):
    """Signal Jacobian applied by differentiating the replay at every action."""


class FuelDenseSignalJacobian(StrictModule):
    """Signal Jacobian stored densely from forward implicit replay tangents.

    One multi-right-hand-side tangent solve per accepted step carries every
    source-law and fill-pressure column. ``materialization`` bounds the stored
    Jacobian of all rods; ``maximum_working_set_bytes`` bounds each rod's
    forward-tangent workspace.
    """

    materialization: MaterializationPolicy = eqx.field(static=True)
    maximum_working_set_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        materialization: MaterializationPolicy,
        maximum_working_set_bytes: int,
    ) -> None:
        if not isinstance(materialization, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy.")
        self.materialization = materialization
        self.maximum_working_set_bytes = positive_integer(
            maximum_working_set_bytes, "maximum_working_set_bytes"
        )


type FuelSignalDerivatives = FuelSignalJacobianActions | FuelDenseSignalJacobian


class _RodReplayStatus(StrictModule):
    """Replay status that gates one rod's sensor residual.

    ``branch_stable`` is reported separately and does not gate the residual.
    """

    sample_validity: Array
    replay_successful: Array
    branch_stable: Array
    grid_times: Array


class _RodSensorEvaluation(StrictModule):
    prediction: Array
    whitened_residual: Array
    log_likelihood: Array
    successful: Array


def _replay_rod_signals(
    observation: FuelObservationPlan,
    bundle: FuelModelBundle,
    fill_pressure_scale: Array,
    /,
) -> tuple[FuelReplayResult, Array, _RodReplayStatus]:
    """Replay one rod on its frozen grid and extract its nuisance-free signals."""
    replay = replay_fuel_grid(
        observation.prepared,
        observation.grid,
        parameters=bundle,
        fill_pressure_scale=fill_pressure_scale,
    )
    signal, sample_validity, grid_times = observation.physical_signals(
        replay.states, parameters=bundle
    )
    status = _RodReplayStatus(
        sample_validity=sample_validity,
        replay_successful=replay.successful,
        branch_stable=~jnp.any(replay.branch_changed),
        grid_times=grid_times,
    )
    return replay, signal, status


def _evaluate_rod_sensors(
    rod: FuelCalibrationRod,
    signal: Array,
    status: _RodReplayStatus,
    /,
    *,
    sensor_nuisance: FuelSensorParameters,
    fill_pressure_scale: Array,
) -> _RodSensorEvaluation:
    """Apply sensor nuisances and the rod likelihood to replayed signals."""
    predicted = rod.observation.apply_sensor_nuisance(
        signal,
        status.sample_validity,
        status.grid_times,
        sensor_nuisance=sensor_nuisance,
    )
    likelihood = rod.likelihood.evaluate(predicted.product)
    covariance = rod.likelihood.covariance
    if not isinstance(covariance, CholeskyCovarianceAction):
        raise TypeError("Calibration rod likelihood requires a Cholesky covariance.")
    successful = (
        status.replay_successful
        & predicted.successful
        & likelihood.successful
        & (fill_pressure_scale > 0)
    )
    return _RodSensorEvaluation(
        prediction=predicted.product.values,
        whitened_residual=covariance.whiten(likelihood.residual),
        log_likelihood=likelihood.log_probability,
        successful=successful,
    )


def _infinite_unless(successful: Array, residual: Array, /) -> Array:
    """Replace every residual entry by +inf when any rod evaluation failed."""
    return jnp.where(successful, residual, jnp.full_like(residual, jnp.inf))


class FuelCalibrationPlan(StrictModule):
    """Coupled-rod calibration: observed rods, parameter space, priors, and campaign.

    The rods must be exactly the campaign's calibration cases. Shared physical
    coordinates come from ``parameterization``; each rod adds a fill-pressure
    scale and its sensor gain, offset, and drift.
    """

    rods: tuple[FuelCalibrationRod, ...]
    parameterization: FuelParameterization
    parameter_space: ParameterSpace
    initial_parameter_fingerprint: str = eqx.field(static=True)
    campaign: ScientificCampaign
    prior_source_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rods: tuple[FuelCalibrationRod, ...],
        parameterization: FuelParameterization,
        parameter_space: ParameterSpace,
        campaign: ScientificCampaign,
        /,
        *,
        prior_source_ids: tuple[str, ...],
    ) -> None:
        if not rods or len({rod.case.case_id for rod in rods}) != len(rods):
            raise ValueError("Calibration requires distinct rod cases.")
        if not prior_source_ids or any(not source for source in prior_source_ids):
            raise ValueError(
                "Calibration priors require nonempty source identifiers, including for synthetic assumptions."
            )
        if not isinstance(campaign, ScientificCampaign):
            raise TypeError("Calibration campaign must be a ScientificCampaign.")
        # A ScientificCampaign carries every role name and a nonempty calibration role.
        role_case_ids = {role.name: role.case_ids for role in campaign.roles}
        if set(role_case_ids["calibration"]) != {rod.case.case_id for rod in rods}:
            raise ValueError(
                "Calibration rods must exactly match the campaign calibration role."
            )
        cases = {case.case_id: case for case in campaign.cases}
        for rod in rods:
            if cases[rod.case.case_id] != rod.case:
                raise ValueError(
                    "Each rod case must equal its campaign case, independent of numerical preparation identity."
                )
        physical = parameter_space.constrain(parameter_space.initial)
        self._validate_parameters(physical, rods, parameterization)
        self.rods = rods
        self.parameterization = parameterization
        self.parameter_space = parameter_space
        self.campaign = campaign
        self.prior_source_ids = prior_source_ids
        self.initial_parameter_fingerprint = self.parameter_fingerprint(physical)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fuel-coupled-rod-calibration",
                "campaign": campaign.campaign_id,
                "rods": tuple(
                    (rod.case.case_id, rod.observation.plan_id, rod.likelihood.plan_id)
                    for rod in rods
                ),
                "physical_coordinates": parameterization.layout.layout_id,
                "parameterization": parameterization.parameterization_id,
                "prior_sources": prior_source_ids,
                "initial_parameter_fingerprint": self.initial_parameter_fingerprint,
            }
        )

    @staticmethod
    def _validate_parameters(
        physical: FuelCalibrationParameters,
        rods: tuple[FuelCalibrationRod, ...],
        parameterization: FuelParameterization,
        /,
    ) -> None:
        if not isinstance(physical, FuelCalibrationParameters):
            raise TypeError("ParameterSpace must constrain to FuelCalibrationParameters.")
        scope = Scope()
        parse(
            parameterization.layout.size,
            Size[FuelParameterDim],
            "physical_parameter_count",
            scope=scope,
        )
        parse(len(rods), Size[FuelCalibrationRodDim], "rod_count", scope=scope)
        parse(
            physical.shared_physical,
            Float[FuelParameterDim],
            "shared_physical",
            scope=scope,
        )
        parse(
            physical.fabrication_fill_pressure_scale,
            Float[FuelCalibrationRodDim],
            "fabrication_fill_pressure_scale",
            scope=scope,
        )
        if len(physical.sensor_nuisance) != len(rods):
            raise ValueError("Each calibration rod requires one sensor nuisance entry.")
        for nuisance, rod in zip(physical.sensor_nuisance, rods, strict=True):
            if not isinstance(nuisance, FuelSensorParameters):
                raise TypeError(
                    "Each calibration rod requires FuelSensorParameters for its sensor gain/offset/drift."
                )
            nuisance.validate(rod.observation.sensor_layout)

    def evaluate_physical(
        self, physical: FuelCalibrationParameters, /
    ) -> FuelCalibrationEvaluation:
        self._validate_parameters(physical, self.rods, self.parameterization)
        predictions: list[Array] = []
        whitened_residuals: list[Array] = []
        log_likelihoods: list[Array] = []
        rod_successes: list[Array] = []
        branch_stability: list[Array] = []
        replays: list[FuelReplayResult] = []
        for rod_index, rod in enumerate(self.rods):
            observation = rod.observation
            fill_pressure_scale = physical.fabrication_fill_pressure_scale[rod_index]
            bundle = self.parameterization.bundle(
                observation.prepared.admission.bundle, physical.shared_physical
            )
            replay, signal, status = _replay_rod_signals(
                observation, bundle, fill_pressure_scale
            )
            sensors = _evaluate_rod_sensors(
                rod,
                signal,
                status,
                sensor_nuisance=physical.sensor_nuisance[rod_index],
                fill_pressure_scale=fill_pressure_scale,
            )
            replays.append(replay)
            predictions.append(sensors.prediction)
            whitened_residuals.append(sensors.whitened_residual)
            log_likelihoods.append(sensors.log_likelihood)
            rod_successes.append(sensors.successful)
            branch_stability.append(status.branch_stable)
        replay_success = jnp.stack(rod_successes)
        successful = jnp.all(replay_success)
        log_likelihood = jnp.sum(jnp.stack(log_likelihoods))
        return FuelCalibrationEvaluation(
            prediction=tuple(predictions),
            whitened_residual=_infinite_unless(
                successful, jnp.concatenate(whitened_residuals)
            ),
            log_likelihood=jnp.where(successful, log_likelihood, -jnp.inf),
            replay_success=replay_success,
            branchwise_valid=jnp.stack(branch_stability),
            successful=successful,
            replays=tuple(replays),
            physical_parameters=physical,
        )

    def residual(self, position: FuelCalibrationParameters, /) -> Array:
        return self.evaluate_physical(
            self.parameter_space.constrain(position)
        ).whitened_residual

    def physical_log_likelihood(self, physical: FuelCalibrationParameters, /) -> Array:
        return self.evaluate_physical(physical).log_likelihood

    def physical_prediction(
        self, physical: FuelCalibrationParameters, /
    ) -> tuple[Array, ...]:
        return self.evaluate_physical(physical).prediction

    def parameter_fingerprint(self, physical: FuelCalibrationParameters, /) -> str:
        """Fingerprint concrete law coefficients and nuisances at the host boundary.

        Static law and bundle IDs identify equation and source structure, not a
        parameter point, so the numerical values are hashed here. Traced values
        are refused; solves never hash.
        """
        self._validate_parameters(physical, self.rods, self.parameterization)
        if any(
            isinstance(leaf, jax_core.Tracer)
            for leaf in jax.tree_util.tree_leaves(physical)
        ):
            raise TypeError("Parameter fingerprints require concrete, untraced values.")
        bundles = tuple(
            self.parameterization.bundle(
                rod.observation.prepared.admission.bundle, physical.shared_physical
            )
            for rod in self.rods
        )
        return canonical_fingerprint(
            {
                "kind": "fuel-actual-parameter-point",
                "prepared_ids": tuple(
                    rod.observation.prepared.prepared_id for rod in self.rods
                ),
                "coordinates": self.parameterization.layout.layout_id,
                "bundle_numeric": array_tree_fingerprint(bundles),
                "physical_and_nuisance_numeric": array_tree_fingerprint(physical),
            }
        )


class _FuelSearchResidual(StrictModule):
    plan: FuelCalibrationPlan

    def __call__(self, position: FuelCalibrationParameters, _args: None, /) -> Array:
        return self.plan.residual(position)


class _FuelNegativeLogPrior(StrictModule):
    space: ParameterSpace

    def __call__(self, position: FuelCalibrationParameters, _args: None, /) -> Array:
        return -self.space.unconstrained_log_prior(position)


class _FuelDecoder(StrictModule):
    space: ParameterSpace

    def __call__(
        self, position: FuelCalibrationParameters, /
    ) -> FuelCalibrationParameters:
        return self.space.constrain(position)


class _RodTangentEvidence(StrictModule):
    """Joint-block derivative solve and branch evidence for one rod."""

    linear_success: Array
    linear_status: Array
    branchwise_valid: Array
    working_set_bytes: int = eqx.field(static=True)


class _FuelResidualEvidence(StrictModule):
    """Per-rod replay status plus joint-block tangent evidence when present."""

    replay_statuses: tuple[_RodReplayStatus, ...]
    tangents: tuple[_RodTangentEvidence, ...]
    maximum_working_set_bytes: int | None = eqx.field(static=True)
    parameter_layout: CoordinateLayout | None = eqx.field(static=True)


class _FuelRawSignals(StrictModule):
    """Nuisance-free sensor signals of every rod as a function of replay inputs."""

    plan: FuelCalibrationPlan

    def __call__(
        self, replay_coordinates: tuple[Array, Array], /
    ) -> tuple[tuple[Array, ...], tuple[_RodReplayStatus, ...]]:
        law_values, fill_pressure_scales = replay_coordinates
        signals: list[Array] = []
        statuses: list[_RodReplayStatus] = []
        for rod_index, rod in enumerate(self.plan.rods):
            observation = rod.observation
            bundle = self.plan.parameterization.bundle(
                observation.prepared.admission.bundle, law_values
            )
            _, signal, status = _replay_rod_signals(
                observation, bundle, fill_pressure_scales[rod_index]
            )
            signals.append(signal)
            statuses.append(status)
        return tuple(signals), tuple(statuses)


def _prepare_action_signals(
    plan: FuelCalibrationPlan, replay_coordinates: tuple[Array, Array], /
) -> tuple[PreparedLinearization, _FuelResidualEvidence]:
    """Linearize the raw signals by automatic differentiation of the replay."""
    raw = prepare_linearization(
        _FuelRawSignals(plan),
        replay_coordinates,
        has_aux=True,
        source=PyTreeSpace(
            replay_coordinates, space_id=f"{plan.plan_id}:replay-coordinates"
        ),
        linearization_id=f"{plan.plan_id}:raw-signals",
    )
    replay_statuses = raw.auxiliary
    if not isinstance(replay_statuses, tuple):
        raise TypeError(
            "Fuel raw-signal linearization must return per-rod replay status."
        )
    evidence = _FuelResidualEvidence(
        replay_statuses=replay_statuses,
        tangents=(),
        maximum_working_set_bytes=None,
        parameter_layout=None,
    )
    signals = PreparedLinearization(
        source=raw.source,
        target=PyTreeSpace(raw.primal, space_id=f"{plan.plan_id}:physical-signals"),
        point=raw.point,
        primal=raw.primal,
        pushforward=raw.pushforward,
        pullback=raw.pullback,
        policy=raw.policy,
        linearization_id=raw.linearization_id,
        auxiliary=evidence,
    )
    return signals, evidence


class _FuelJointLawUpdate(StrictModule):
    """Bundle update of a joint vector: apply only its leading law coordinates."""

    law_parameterization: FuelParameterization

    def __call__(self, base: FuelModelBundle, joint_values: Array, /) -> FuelModelBundle:
        law_count = self.law_parameterization.layout.size
        return self.law_parameterization.bundle(base, joint_values[:law_count])


class _FuelJointInitialState(StrictModule):
    """Fabricated initial state of one rod as a function of the joint vector."""

    prepared: PreparedFuelRod
    joint_parameterization: FuelParameterization
    fill_scale_index: int = eqx.field(static=True)

    def __call__(self, joint_values: Array, /) -> FuelRodState:
        return self.prepared.initial_state(
            parameters=self.joint_parameterization.bundle(
                self.prepared.admission.bundle, joint_values
            ),
            fill_pressure_scale=joint_values[self.fill_scale_index],
        )


class _FuelJointInitialTangent(StrictModule):
    initial_state: _FuelJointInitialState
    joint_values: Array

    def __call__(self, direction: Array, /) -> FuelRodState:
        return eqx.filter_jvp(self.initial_state, (self.joint_values,), (direction,))[1]


class _FuelJointSignalProjection(StrictModule):
    """Nuisance-free signals of one rod as a function of joint vector and history."""

    observation: FuelObservationPlan
    joint_parameterization: FuelParameterization

    def __call__(self, joint_values: Array, states: tuple[FuelRodState, ...], /) -> Array:
        bundle = self.joint_parameterization.bundle(
            self.observation.prepared.admission.bundle, joint_values
        )
        return self.observation.physical_signals(states, parameters=bundle)[0]


class _FuelJointSignalTangent(StrictModule):
    projection: _FuelJointSignalProjection
    joint_values: Array
    states: tuple[FuelRodState, ...]

    def __call__(
        self, direction: Array, history_tangent: tuple[FuelRodState, ...], /
    ) -> Array:
        return eqx.filter_jvp(
            self.projection,
            (self.joint_values, self.states),
            (direction, history_tangent),
        )[1]


def _joint_parameterization(plan: FuelCalibrationPlan, /) -> FuelParameterization:
    """Joint law-plus-fill-scale coordinates for the forward sensitivity plan.

    The joint vector is ``concatenate((law_values, fill_pressure_scales))``:
    the source-law coordinates of ``plan.parameterization`` followed by one
    fabrication fill-pressure scale per rod, so rod ``r``'s scale sits at
    index ``law_count + r``.
    """
    law = plan.parameterization
    layout = CoordinateLayout(
        law.layout.labels
        + tuple(
            f"{rod.case.case_id}:fabrication-fill-pressure-scale" for rod in plan.rods
        )
    )
    return FuelParameterization(
        layout,
        law.units + ("1",) * len(plan.rods),
        _FuelJointLawUpdate(law),
        parameterization_id=canonical_fingerprint(
            {
                "kind": "fuel-joint-source-law-and-fabrication",
                "plan": plan.plan_id,
                "physical": law.parameterization_id,
                "coordinates": layout.layout_id,
            }
        ),
    )


def _joint_signal_jacobian(
    observation: FuelObservationPlan,
    replay: FuelReplayResult,
    joint_parameterization: FuelParameterization,
    joint_values: Array,
    direction_basis: Array,
    /,
    *,
    fill_scale_index: int,
    maximum_working_set_bytes: int,
) -> tuple[Array, _RodTangentEvidence]:
    """Dense signal Jacobian of one rod from forward implicit history tangents."""
    coordinate_count = joint_parameterization.layout.size
    initial_state = _FuelJointInitialState(
        prepared=observation.prepared,
        joint_parameterization=joint_parameterization,
        fill_scale_index=fill_scale_index,
    )
    initial_tangents = eqx.filter_vmap(
        _FuelJointInitialTangent(initial_state=initial_state, joint_values=joint_values),
        in_axes=0,
    )(direction_basis)
    sensitivity = FuelSensitivityPlan(
        observation.prepared,
        observation.grid,
        joint_parameterization,
        joint_values,
        parameter_capacity=coordinate_count,
        maximum_working_set_bytes=maximum_working_set_bytes,
    )
    tangent = sensitivity.evaluate(
        replay.states,
        initial_tangents=unstack_tree(initial_tangents, coordinate_count),
    )
    history_tangents = tuple(stack_trees(step) for step in tangent.tangents)
    projection = _FuelJointSignalProjection(
        observation=observation, joint_parameterization=joint_parameterization
    )
    matrix = eqx.filter_vmap(
        _FuelJointSignalTangent(
            projection=projection, joint_values=joint_values, states=replay.states
        ),
        in_axes=(0, 0),
    )(direction_basis, history_tangents).T
    matrix = eqx.error_if(
        matrix,
        ~replay.successful,
        "Fuel signal preparation requires a successful replay.",
    )
    matrix = eqx.error_if(
        matrix,
        ~jnp.all(tangent.linear_success),
        "Fuel signal preparation failed its joint tangent linear solve.",
    )
    evidence = _RodTangentEvidence(
        linear_success=tangent.linear_success,
        linear_status=tangent.linear_status,
        branchwise_valid=tangent.branchwise_valid,
        working_set_bytes=sensitivity.working_set_bytes,
    )
    return matrix, evidence


def _prepare_joint_block_signals(
    plan: FuelCalibrationPlan,
    replay_coordinates: tuple[Array, Array],
    /,
    *,
    materialization: MaterializationPolicy,
    maximum_working_set_bytes: int,
) -> tuple[PreparedLinearization, _FuelResidualEvidence]:
    """Store the dense raw-signal Jacobian from forward implicit tangents."""
    law_values, fill_pressure_scales = replay_coordinates
    source = PyTreeSpace(
        replay_coordinates, space_id=f"{plan.plan_id}:joint-replay-coordinates"
    )
    joint_parameterization = _joint_parameterization(plan)
    law_count = plan.parameterization.layout.size
    joint_values = jnp.concatenate((law_values, fill_pressure_scales))
    signal_count = sum(rod.observation.layout.size for rod in plan.rods)
    materialization.require(
        source.size * signal_count,
        jnp.result_type(law_values.dtype, fill_pressure_scales.dtype),
    )
    direction_basis = jnp.eye(source.size, dtype=joint_values.dtype)
    signals: list[Array] = []
    matrices: list[Array] = []
    replay_statuses: list[_RodReplayStatus] = []
    tangents: list[_RodTangentEvidence] = []
    for rod_index, rod in enumerate(plan.rods):
        observation = rod.observation
        bundle = joint_parameterization.bundle(
            observation.prepared.admission.bundle, joint_values
        )
        replay, signal, status = _replay_rod_signals(
            observation, bundle, fill_pressure_scales[rod_index]
        )
        matrix, tangent = _joint_signal_jacobian(
            observation,
            replay,
            joint_parameterization,
            joint_values,
            direction_basis,
            fill_scale_index=law_count + rod_index,
            maximum_working_set_bytes=maximum_working_set_bytes,
        )
        signals.append(signal)
        matrices.append(matrix)
        replay_statuses.append(status)
        tangents.append(tangent)
    primal = tuple(signals)
    target = PyTreeSpace(primal, space_id=f"{plan.plan_id}:joint-physical-signals")
    operator = DenseLinearOperator(
        jnp.concatenate(matrices, axis=0),
        source=source,
        target=target,
        operator_id=f"{plan.plan_id}:native-joint-block-signals",
    )
    evidence = _FuelResidualEvidence(
        replay_statuses=tuple(replay_statuses),
        tangents=tuple(tangents),
        maximum_working_set_bytes=maximum_working_set_bytes,
        parameter_layout=joint_parameterization.layout,
    )
    linearization = PreparedLinearization(
        source=source,
        target=target,
        point=replay_coordinates,
        primal=primal,
        pushforward=operator.mv,
        pullback=operator.transpose_mv,
        policy=LinearizationPolicy("store"),
        linearization_id=f"{plan.plan_id}:native-joint-block-signals",
        auxiliary=evidence,
    )
    return linearization, evidence


class _FuelSensorResidual(StrictModule):
    """Whitened residual of all rods given raw signals, nuisances, and fill scales."""

    plan: FuelCalibrationPlan
    replay_statuses: tuple[_RodReplayStatus, ...]

    def __call__(
        self,
        point: tuple[tuple[Array, ...], tuple[FuelSensorParameters, ...], Array],
        /,
    ) -> Array:
        signals, sensor_nuisance, fill_pressure_scales = point
        whitened_residuals: list[Array] = []
        rod_successes: list[Array] = []
        for rod_index, rod in enumerate(self.plan.rods):
            sensors = _evaluate_rod_sensors(
                rod,
                signals[rod_index],
                self.replay_statuses[rod_index],
                sensor_nuisance=sensor_nuisance[rod_index],
                fill_pressure_scale=fill_pressure_scales[rod_index],
            )
            rod_successes.append(sensors.successful)
            whitened_residuals.append(sensors.whitened_residual)
        return _infinite_unless(
            jnp.all(jnp.stack(rod_successes)), jnp.concatenate(whitened_residuals)
        )


class _FuelComposedPushforward(StrictModule):
    decoder: PreparedLinearization
    signals: PreparedLinearization
    sensor: PreparedLinearization

    def __call__(self, tangent: FuelCalibrationParameters, /) -> Array:
        physical_tangent = self.decoder.jvp(tangent)
        signal_tangent = self.signals.jvp(
            (
                physical_tangent.shared_physical,
                physical_tangent.fabrication_fill_pressure_scale,
            )
        )
        return self.sensor.jvp(
            (
                signal_tangent,
                physical_tangent.sensor_nuisance,
                physical_tangent.fabrication_fill_pressure_scale,
            )
        )


class _FuelComposedPullback(StrictModule):
    decoder: PreparedLinearization
    signals: PreparedLinearization
    sensor: PreparedLinearization

    def __call__(self, cotangent: Array, /) -> FuelCalibrationParameters:
        signal_cotangent, nuisance_cotangent, direct_fill_cotangent = self.sensor.vjp(
            cotangent
        )
        law_cotangent, replay_fill_cotangent = self.signals.vjp(signal_cotangent)
        physical_cotangent = FuelCalibrationParameters(
            shared_physical=law_cotangent,
            fabrication_fill_pressure_scale=replay_fill_cotangent + direct_fill_cotangent,
            sensor_nuisance=nuisance_cotangent,
        )
        return self.decoder.vjp(physical_cotangent)


class _FuelResidualLinearization(StrictModule):
    """Linearize the whitened residual as decoder, raw signals, then sensors.

    Raw signals depend only on shared law values and fill-pressure scales;
    sensor nuisances and the direct fill-scale validity gate enter the sensor
    stage. ``FuelSignalJacobianActions`` linearizes the signals by automatic
    differentiation of the replay; ``FuelDenseSignalJacobian`` stores a dense
    signal Jacobian from forward implicit tangents.
    """

    plan: FuelCalibrationPlan
    signal_derivatives: FuelSignalDerivatives = eqx.field(static=True)

    def __call__(
        self, position: FuelCalibrationParameters, _args: None, /
    ) -> PreparedLinearization:
        plan = self.plan
        source = PyTreeSpace(position)
        decode = _FuelDecoder(plan.parameter_space)
        decoder = prepare_linearization(
            decode,
            position,
            source=source,
            target=PyTreeSpace(
                jax.eval_shape(decode, position),
                space_id=f"{plan.plan_id}:physical-parameters",
            ),
            linearization_id=f"{plan.plan_id}:decoder",
        )
        physical = decoder.primal
        plan._validate_parameters(physical, plan.rods, plan.parameterization)
        replay_coordinates = (
            physical.shared_physical,
            physical.fabrication_fill_pressure_scale,
        )
        match self.signal_derivatives:
            case FuelSignalJacobianActions():
                signals, evidence = _prepare_action_signals(plan, replay_coordinates)
                linearization_id = f"{plan.plan_id}:composed-residual"
            case FuelDenseSignalJacobian(
                materialization=materialization,
                maximum_working_set_bytes=maximum_working_set_bytes,
            ):
                signals, evidence = _prepare_joint_block_signals(
                    plan,
                    replay_coordinates,
                    materialization=materialization,
                    maximum_working_set_bytes=maximum_working_set_bytes,
                )
                linearization_id = f"{plan.plan_id}:composed-residual-native-joint-block"
            case _:
                assert_never(self.signal_derivatives)
        sensor_point = (
            signals.primal,
            physical.sensor_nuisance,
            physical.fabrication_fill_pressure_scale,
        )
        sensor = prepare_linearization(
            _FuelSensorResidual(plan=plan, replay_statuses=evidence.replay_statuses),
            sensor_point,
            source=PyTreeSpace(
                sensor_point, space_id=f"{plan.plan_id}:sensor-coordinates"
            ),
            linearization_id=f"{plan.plan_id}:sensor-residual",
        )
        return PreparedLinearization(
            source=source,
            target=sensor.target,
            point=position,
            primal=sensor.primal,
            pushforward=_FuelComposedPushforward(
                decoder=decoder, signals=signals, sensor=sensor
            ),
            pullback=_FuelComposedPullback(
                decoder=decoder, signals=signals, sensor=sensor
            ),
            policy=decoder.policy,
            linearization_id=linearization_id,
            auxiliary=evidence,
        )


class _FuelPhysicalResidual(StrictModule):
    plan: FuelCalibrationPlan

    def __call__(self, physical: FuelCalibrationParameters, /) -> Array:
        return self.plan.evaluate_physical(physical).whitened_residual


@overload
def prepare_fuel_least_squares(
    plan: FuelCalibrationPlan,
    /,
    *,
    include_prior: Literal[True] = True,
    signal_derivatives: FuelSignalDerivatives = FuelSignalJacobianActions(),
) -> CompositeLeastSquaresProblem: ...


@overload
def prepare_fuel_least_squares(
    plan: FuelCalibrationPlan,
    /,
    *,
    include_prior: Literal[False],
    signal_derivatives: FuelSignalJacobianActions = FuelSignalJacobianActions(),
) -> NonlinearLeastSquaresProblem: ...


def prepare_fuel_least_squares(
    plan: FuelCalibrationPlan,
    /,
    *,
    include_prior: bool = True,
    signal_derivatives: FuelSignalDerivatives = FuelSignalJacobianActions(),
) -> NonlinearLeastSquaresProblem | CompositeLeastSquaresProblem:
    """Prepare least-squares calibration with explicit signal and workspace bounds.

    ``signal_derivatives`` selects how the composite residual linearization
    applies the signal Jacobian; ``FuelDenseSignalJacobian`` requires
    ``include_prior``.
    """
    if not isinstance(
        signal_derivatives, (FuelSignalJacobianActions, FuelDenseSignalJacobian)
    ):
        raise TypeError(
            "signal_derivatives must be FuelSignalJacobianActions or FuelDenseSignalJacobian."
        )
    if isinstance(signal_derivatives, FuelDenseSignalJacobian) and not include_prior:
        raise ValueError("A dense signal Jacobian requires the composite residual model.")
    residual = _FuelSearchResidual(plan)
    if include_prior:
        return CompositeLeastSquaresProblem(
            residual,
            _FuelNegativeLogPrior(plan.parameter_space),
            problem_id=plan.plan_id,
            residual_linearization=_FuelResidualLinearization(
                plan=plan,
                signal_derivatives=signal_derivatives,
            ),
        )
    return NonlinearLeastSquaresProblem(residual, problem_id=plan.plan_id)


def prepare_fuel_posterior(plan: FuelCalibrationPlan, /) -> PosteriorProblem:
    """Bayesian posterior over the plan's parameter space with its Gauss-Newton residual."""
    return PosteriorProblem(
        plan.parameter_space,
        plan.physical_log_likelihood,
        predict=plan.physical_prediction,
        gauss_newton_residual=_FuelPhysicalResidual(plan),
    )
