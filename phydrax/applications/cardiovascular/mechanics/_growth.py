#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Anisotropic, cycle-driven finite-growth transactions.

Growth is represented by a symmetric logarithmic tensor ``H_g`` and therefore
``F_g = exp(H_g)`` is positive definite by construction.  Fast mechanics only
sees committed states from one prepared anatomy/reference epoch.  Slow growth
and epoch replacement are separate proposal/evidence/commit transactions.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import hermitian_exp


class GrowthStatus(IntEnum):
    """Fail-closed status for slow growth updates."""

    SUCCESS = 0
    NONFINITE = 1
    STALE_EPOCH = 2
    INSUFFICIENT_CYCLES = 3
    INCREMENT_TOO_LARGE = 4
    LOG_BOUND_EXCEEDED = 5
    POSITIVITY_FAILURE = 6
    INVALID_TRANSFER = 7
    REFINEMENT_EXHAUSTED = 8
    ROLLED_BACK = 9


class ContinuumGrowthFidelity(StrictModule, NonTrainableState):
    """Typed route for deterministic material-point continuum growth."""

    route_id: str = eqx.field(static=True)

    def __init__(self):
        self.route_id = "cardiovascular-continuum-log-tensor-growth"


class GrowthReferenceEpoch(StrictModule, NonTrainableState):
    """Stable anatomy and reference-configuration identity for one epoch."""

    anatomy_id: str = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    material_point_ids: tuple[str, ...] = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)

    def __init__(
        self,
        anatomy_id: str,
        reference_id: str,
        topology_id: str,
        material_point_ids: tuple[str, ...],
        /,
    ):
        names = (anatomy_id, reference_id, topology_id)
        if any(not isinstance(value, str) or not value.strip() for value in names):
            raise ValueError("Anatomy, reference, and topology IDs must be nonempty.")
        point_ids = tuple(material_point_ids)
        if not point_ids or any(
            not isinstance(value, str) or not value.strip() for value in point_ids
        ):
            raise ValueError("material_point_ids must contain nonempty stable IDs.")
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("material_point_ids must be unique within an epoch.")
        self.anatomy_id = anatomy_id
        self.reference_id = reference_id
        self.topology_id = topology_id
        self.material_point_ids = point_ids
        self.epoch_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-growth-reference-epoch",
                "anatomy": anatomy_id,
                "reference": reference_id,
                "topology": topology_id,
                "material_points": point_ids,
            }
        )


def _finite_real_array(name: str, value: ArrayLike, rank: int, /) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}.")
    if not np.issubdtype(array.dtype, np.inexact) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _positive_scalar(name: str, value: float, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _nonnegative_scalar(name: str, value: float, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def _point_channel_parameter(
    name: str,
    value: ArrayLike,
    shape: tuple[int, int],
    /,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == ():
        array = np.full(shape, array, dtype=np.result_type(array, np.float32))
    elif array.shape == (shape[1],):
        array = np.broadcast_to(array, shape)
    array = _finite_real_array(name, array, 2)
    if array.shape != shape:
        raise ValueError(f"{name} must be scalar, channel-shaped, or have shape {shape}.")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive.")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return array


class GrowthPlan(StrictModule, NonTrainableState):
    """An arbitrary-direction homeostatic growth law on fixed material points.

    ``reference_directions[p, c]`` is a unit vector defining channel ``c`` at
    material point ``p``.  Nothing assigns radial, circumferential, or spherical
    meaning to a channel; those semantics are supplied by the anatomy owner via
    stable ``direction_ids``.
    """

    reference_directions: Array
    homeostatic_targets: Array
    stimulus_scales: Array
    growth_gains: Array
    fidelity: ContinuumGrowthFidelity
    material_point_ids: tuple[str, ...] = eqx.field(static=True)
    direction_ids: tuple[str, ...] = eqx.field(static=True)
    minimum_cycles: int = eqx.field(static=True)
    deadband: float = eqx.field(static=True)
    maximum_log_increment: float = eqx.field(static=True)
    maximum_log_magnitude: float = eqx.field(static=True)
    maximum_refinements: int = eqx.field(static=True)
    tensor_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        material_point_ids: tuple[str, ...],
        direction_ids: tuple[str, ...],
        reference_directions: ArrayLike,
        homeostatic_targets: ArrayLike,
        stimulus_scales: ArrayLike,
        growth_gains: ArrayLike,
        /,
        *,
        minimum_cycles: int = 1,
        deadband: float = 0.0,
        maximum_log_increment: float = 0.02,
        maximum_log_magnitude: float = 1.0,
        maximum_refinements: int = 8,
        tensor_tolerance: float = 1.0e-6,
        fidelity: ContinuumGrowthFidelity | None = None,
    ):
        point_ids = tuple(material_point_ids)
        channel_ids = tuple(direction_ids)
        if not point_ids or any(not value.strip() for value in point_ids):
            raise ValueError("material_point_ids must be nonempty stable IDs.")
        if not channel_ids or any(not value.strip() for value in channel_ids):
            raise ValueError("direction_ids must be nonempty stable IDs.")
        if len(set(point_ids)) != len(point_ids) or len(set(channel_ids)) != len(
            channel_ids
        ):
            raise ValueError("Material-point and direction IDs must each be unique.")
        directions = _finite_real_array("reference_directions", reference_directions, 3)
        expected_prefix = (len(point_ids), len(channel_ids))
        if directions.shape[:2] != expected_prefix or directions.shape[-1] not in (
            2,
            3,
        ):
            raise ValueError(
                "reference_directions must have shape (points, channels, 2|3)."
            )
        norms = np.sqrt(np.sum(directions * directions, axis=-1))
        if np.any(norms <= 0.0):
            raise ValueError("Every reference growth direction must be nonzero.")
        directions = directions / norms[..., None]
        shape = expected_prefix
        targets = _point_channel_parameter(
            "homeostatic_targets", homeostatic_targets, shape
        )
        scales = _point_channel_parameter(
            "stimulus_scales", stimulus_scales, shape, positive=True
        )
        gains = _point_channel_parameter(
            "growth_gains", growth_gains, shape, nonnegative=True
        )
        minimum_cycles_ = int(minimum_cycles)
        refinements = int(maximum_refinements)
        if minimum_cycles_ < 1 or minimum_cycles_ != minimum_cycles:
            raise ValueError("minimum_cycles must be a positive integer.")
        if refinements < 0 or refinements != maximum_refinements:
            raise ValueError("maximum_refinements must be a nonnegative integer.")
        fidelity_ = ContinuumGrowthFidelity() if fidelity is None else fidelity
        if not isinstance(fidelity_, ContinuumGrowthFidelity):
            raise TypeError("GrowthPlan requires ContinuumGrowthFidelity.")
        deadband_ = _nonnegative_scalar("deadband", deadband)
        max_increment = _positive_scalar("maximum_log_increment", maximum_log_increment)
        max_magnitude = _positive_scalar("maximum_log_magnitude", maximum_log_magnitude)
        tolerance = _nonnegative_scalar("tensor_tolerance", tensor_tolerance)
        self.reference_directions = jnp.asarray(directions)
        self.homeostatic_targets = jnp.asarray(targets)
        self.stimulus_scales = jnp.asarray(scales)
        self.growth_gains = jnp.asarray(gains)
        self.fidelity = fidelity_
        self.material_point_ids = point_ids
        self.direction_ids = channel_ids
        self.minimum_cycles = minimum_cycles_
        self.deadband = deadband_
        self.maximum_log_increment = max_increment
        self.maximum_log_magnitude = max_magnitude
        self.maximum_refinements = refinements
        self.tensor_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-growth-plan",
                "points": point_ids,
                "channels": channel_ids,
                "directions": array_tree_fingerprint(directions),
                "targets": array_tree_fingerprint(targets),
                "scales": array_tree_fingerprint(scales),
                "gains": array_tree_fingerprint(gains),
                "minimum_cycles": minimum_cycles_,
                "deadband": deadband_,
                "maximum_log_increment": max_increment,
                "maximum_log_magnitude": max_magnitude,
                "maximum_refinements": refinements,
                "tensor_tolerance": tolerance,
                "fidelity": fidelity_.route_id,
            }
        )

    @property
    def dimension(self) -> int:
        return self.reference_directions.shape[-1]

    @property
    def point_count(self) -> int:
        return len(self.material_point_ids)

    @property
    def channel_count(self) -> int:
        return len(self.direction_ids)


class PreparedGrowth(StrictModule, NonTrainableState):
    """Fixed-shape growth operators bound to one anatomy/reference epoch."""

    plan: GrowthPlan
    epoch: GrowthReferenceEpoch
    direction_projectors: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: GrowthPlan, epoch: GrowthReferenceEpoch, /):
        if not isinstance(plan, GrowthPlan):
            raise TypeError("plan must be a GrowthPlan.")
        if not isinstance(epoch, GrowthReferenceEpoch):
            raise TypeError("epoch must be a GrowthReferenceEpoch.")
        if plan.material_point_ids != epoch.material_point_ids:
            raise ValueError(
                "Growth material-point IDs do not match the reference epoch; "
                "rebuild the plan and provide an explicit epoch transfer."
            )
        projectors = oe.contract(
            "pci,pcj->pcij", plan.reference_directions, plan.reference_directions
        )
        self.plan = plan
        self.epoch = epoch
        self.direction_projectors = projectors
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiovascular-growth",
                "plan": plan.plan_id,
                "epoch": epoch.epoch_id,
            }
        )


def prepare_growth(plan: GrowthPlan, epoch: GrowthReferenceEpoch, /) -> PreparedGrowth:
    """Bind a growth law to exactly one fixed anatomy/reference epoch."""

    return PreparedGrowth(plan, epoch)


class LogGrowthTensorState(StrictModule, NonTrainableState):
    """Committed symmetric log-growth tensor and slow time in milliseconds."""

    log_growth_tensor: Array
    slow_time_ms: Array
    prepared_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        log_growth_tensor: ArrayLike,
        slow_time_ms: ArrayLike,
        prepared_id: str,
        /,
    ):
        value = _finite_real_array("log_growth_tensor", log_growth_tensor, 3)
        if value.shape[-1] not in (2, 3) or value.shape[-2] != value.shape[-1]:
            raise ValueError("log_growth_tensor must have shape (points, d, d), d=2|3.")
        tolerance = (
            32.0 * np.finfo(value.dtype).eps * max(1.0, float(np.max(np.abs(value))))
        )
        if np.max(np.abs(value - np.swapaxes(value, -1, -2))) > tolerance:
            raise ValueError("log_growth_tensor must be symmetric.")
        time = np.asarray(slow_time_ms)
        if time.shape != () or not np.issubdtype(time.dtype, np.inexact):
            raise ValueError("slow_time_ms must be a real scalar.")
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("slow_time_ms must be finite and nonnegative.")
        if not isinstance(prepared_id, str) or not prepared_id:
            raise ValueError("prepared_id must be a nonempty stable ID.")
        symmetric = 0.5 * (value + np.swapaxes(value, -1, -2))
        self.log_growth_tensor = jnp.asarray(symmetric)
        self.slow_time_ms = jnp.asarray(time)
        self.prepared_id = prepared_id
        self.state_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-log-growth-state",
                "prepared": prepared_id,
                "log_tensor": array_tree_fingerprint(symmetric),
                "slow_time_ms": float(time),
            }
        )


def initialize_growth_state(prepared: PreparedGrowth, /) -> LogGrowthTensorState:
    """Create the identity-growth state for a prepared epoch."""

    if not isinstance(prepared, PreparedGrowth):
        raise TypeError("prepared must be PreparedGrowth.")
    shape = (prepared.plan.point_count, prepared.plan.dimension, prepared.plan.dimension)
    return LogGrowthTensorState(
        np.zeros(shape, dtype=np.asarray(prepared.plan.reference_directions).dtype),
        0.0,
        prepared.prepared_id,
    )


def _validate_growth_state(
    prepared: PreparedGrowth, state: LogGrowthTensorState, /
) -> None:
    if not isinstance(prepared, PreparedGrowth):
        raise TypeError("prepared must be PreparedGrowth.")
    if not isinstance(state, LogGrowthTensorState):
        raise TypeError("state must be LogGrowthTensorState.")
    if state.prepared_id != prepared.prepared_id:
        raise ValueError(
            "Growth state belongs to a different anatomy/reference epoch; "
            "an explicit transfer and rebuild transaction is required."
        )
    expected = (
        prepared.plan.point_count,
        prepared.plan.dimension,
        prepared.plan.dimension,
    )
    if state.log_growth_tensor.shape != expected:
        raise ValueError("Growth state shape does not match its prepared epoch.")


class GrowthKinematics(StrictModule, NonTrainableState):
    """Multiplicative split and reconstruction/positivity evidence."""

    total_deformation_gradient: Array
    elastic_deformation_gradient: Array
    growth_deformation_gradient: Array
    inverse_growth_deformation_gradient: Array
    total_jacobian: Array
    elastic_jacobian: Array
    growth_jacobian: Array
    reconstruction_error: Array
    growth_positive: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)


def evaluate_growth_kinematics(
    prepared: PreparedGrowth,
    state: LogGrowthTensorState,
    total_deformation_gradient: ArrayLike,
    /,
) -> GrowthKinematics:
    """Evaluate ``F = F_e F_g`` without assuming any chamber geometry."""

    _validate_growth_state(prepared, state)
    total = jnp.asarray(total_deformation_gradient)
    if total.shape != state.log_growth_tensor.shape:
        raise ValueError("Total deformation gradient shape must match growth state.")
    growth_result = hermitian_exp(
        state.log_growth_tensor, tolerance=prepared.plan.tensor_tolerance
    )
    spectrum = growth_result.spectrum
    inverse = oe.contract(
        "pik,pk,pjk->pij",
        spectrum.eigenvectors,
        jnp.exp(-spectrum.eigenvalues),
        spectrum.eigenvectors,
    )
    growth = growth_result.value
    elastic = oe.contract("pij,pjk->pik", total, inverse)
    reconstructed = oe.contract("pij,pjk->pik", elastic, growth)
    error = jnp.max(jnp.abs(reconstructed - total), axis=(-2, -1))
    growth_jacobian = jnp.linalg.det(growth)
    elastic_jacobian = jnp.linalg.det(elastic)
    total_jacobian = jnp.linalg.det(total)
    positive = growth_result.valid & (growth_jacobian > 0.0)
    finite = (
        jnp.all(jnp.isfinite(total), axis=(-2, -1))
        & jnp.all(jnp.isfinite(elastic), axis=(-2, -1))
        & jnp.all(jnp.isfinite(growth), axis=(-2, -1))
    )
    return GrowthKinematics(
        total,
        elastic,
        growth,
        inverse,
        total_jacobian,
        elastic_jacobian,
        growth_jacobian,
        error,
        positive,
        finite,
        prepared.prepared_id,
    )


class GrowthCycleSummary(StrictModule, NonTrainableState):
    """Time-weighted directional stimulus mean for one completed fast cycle."""

    mean_stimulus: Array
    duration_ms: Array
    cycle_index: int = eqx.field(static=True)
    cycle_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean_stimulus: ArrayLike,
        duration_ms: ArrayLike,
        cycle_index: int,
        prepared_id: str,
        /,
    ):
        mean = _finite_real_array("mean_stimulus", mean_stimulus, 2)
        duration = np.asarray(duration_ms)
        index = int(cycle_index)
        if duration.shape != () or not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_ms must be a finite positive scalar.")
        if index < 0 or index != cycle_index:
            raise ValueError("cycle_index must be a nonnegative integer.")
        self.mean_stimulus = jnp.asarray(mean)
        self.duration_ms = jnp.asarray(duration)
        self.cycle_index = index
        self.prepared_id = prepared_id
        self.cycle_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-growth-cycle",
                "prepared": prepared_id,
                "index": index,
                "duration_ms": float(duration),
                "mean_stimulus": array_tree_fingerprint(mean),
            }
        )


def aggregate_growth_cycle(
    prepared: PreparedGrowth,
    cycle_index: int,
    sample_times_ms: ArrayLike,
    stimulus_tensors: ArrayLike,
    /,
) -> GrowthCycleSummary:
    """Integrate one tensor-stimulus cycle and project onto declared directions."""

    if not isinstance(prepared, PreparedGrowth):
        raise TypeError("prepared must be PreparedGrowth.")
    times = _finite_real_array("sample_times_ms", sample_times_ms, 1)
    tensors = _finite_real_array("stimulus_tensors", stimulus_tensors, 4)
    expected = (
        times.size,
        prepared.plan.point_count,
        prepared.plan.dimension,
        prepared.plan.dimension,
    )
    if times.size < 2 or tensors.shape != expected:
        raise ValueError(
            "A cycle needs at least two times and tensors shaped (samples, points, d, d)."
        )
    differences = np.diff(times)
    if np.any(differences <= 0.0):
        raise ValueError("Cycle sample times must be strictly increasing.")
    asymmetry = np.max(np.abs(tensors - np.swapaxes(tensors, -1, -2)))
    if asymmetry > prepared.plan.tensor_tolerance:
        raise ValueError("Cycle stimulus tensors must be symmetric.")
    directional = oe.contract(
        "spij,pci,pcj->spc",
        jnp.asarray(tensors),
        prepared.plan.reference_directions,
        prepared.plan.reference_directions,
    )
    dt = jnp.asarray(differences)
    integral = jnp.sum(
        0.5 * (directional[:-1] + directional[1:]) * dt[:, None, None],
        axis=0,
    )
    duration = times[-1] - times[0]
    return GrowthCycleSummary(
        integral / duration,
        duration,
        cycle_index,
        prepared.prepared_id,
    )


class GrowthCycleAccumulator(StrictModule, NonTrainableState):
    """Fixed-shape sufficient statistics over completed cardiac cycles."""

    integrated_stimulus: Array
    total_duration_ms: Array
    cycle_count: int = eqx.field(static=True)
    last_cycle_index: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        integrated_stimulus: ArrayLike,
        total_duration_ms: ArrayLike,
        cycle_count: int,
        last_cycle_index: int,
        prepared_id: str,
        /,
    ):
        integral = _finite_real_array("integrated_stimulus", integrated_stimulus, 2)
        duration = np.asarray(total_duration_ms)
        count = int(cycle_count)
        last = int(last_cycle_index)
        if duration.shape != () or not np.isfinite(duration) or duration < 0.0:
            raise ValueError("total_duration_ms must be finite and nonnegative.")
        if count < 0 or count != cycle_count:
            raise ValueError("cycle_count must be a nonnegative integer.")
        if (count == 0 and (last != -1 or duration != 0.0)) or (
            count > 0 and (last < count - 1 or duration <= 0.0)
        ):
            raise ValueError("Cycle count, last index, and duration are inconsistent.")
        self.integrated_stimulus = jnp.asarray(integral)
        self.total_duration_ms = jnp.asarray(duration)
        self.cycle_count = count
        self.last_cycle_index = last
        self.prepared_id = prepared_id


def initialize_growth_cycle_accumulator(
    prepared: PreparedGrowth, /
) -> GrowthCycleAccumulator:
    if not isinstance(prepared, PreparedGrowth):
        raise TypeError("prepared must be PreparedGrowth.")
    shape = (prepared.plan.point_count, prepared.plan.channel_count)
    return GrowthCycleAccumulator(
        np.zeros(shape, dtype=np.asarray(prepared.plan.homeostatic_targets).dtype),
        0.0,
        0,
        -1,
        prepared.prepared_id,
    )


def accumulate_growth_cycle(
    prepared: PreparedGrowth,
    accumulator: GrowthCycleAccumulator,
    summary: GrowthCycleSummary,
    /,
) -> GrowthCycleAccumulator:
    """Append exactly the next completed cycle; duplicate/reordered cycles fail."""

    if not isinstance(accumulator, GrowthCycleAccumulator):
        raise TypeError("accumulator must be GrowthCycleAccumulator.")
    if not isinstance(summary, GrowthCycleSummary):
        raise TypeError("summary must be GrowthCycleSummary.")
    if (
        accumulator.prepared_id != prepared.prepared_id
        or summary.prepared_id != prepared.prepared_id
    ):
        raise ValueError("Cycle data belongs to a stale anatomy/reference epoch.")
    expected = (prepared.plan.point_count, prepared.plan.channel_count)
    if accumulator.integrated_stimulus.shape != expected:
        raise ValueError("Accumulator shape does not match prepared growth channels.")
    if summary.mean_stimulus.shape != expected:
        raise ValueError("Cycle summary shape does not match prepared growth channels.")
    if summary.cycle_index != accumulator.last_cycle_index + 1:
        raise ValueError("Cycles must be accumulated once and in strict index order.")
    integral = (
        accumulator.integrated_stimulus + summary.duration_ms * summary.mean_stimulus
    )
    return GrowthCycleAccumulator(
        integral,
        accumulator.total_duration_ms + summary.duration_ms,
        accumulator.cycle_count + 1,
        summary.cycle_index,
        prepared.prepared_id,
    )


class GrowthStimulusEvaluation(StrictModule, NonTrainableState):
    observed: Array
    target: Array
    effective_deadband: Array
    normalized_error: Array
    homeostatic: Array
    sufficient_cycles: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)


def evaluate_homeostatic_stimulus(
    prepared: PreparedGrowth, accumulator: GrowthCycleAccumulator, /
) -> GrowthStimulusEvaluation:
    """Reduce the complete-cycle window to normalized homeostatic errors."""

    if not isinstance(accumulator, GrowthCycleAccumulator):
        raise TypeError("accumulator must be GrowthCycleAccumulator.")
    if accumulator.prepared_id != prepared.prepared_id:
        raise ValueError("Accumulator belongs to a stale anatomy/reference epoch.")
    sufficient = accumulator.cycle_count >= prepared.plan.minimum_cycles
    safe_duration = jnp.maximum(accumulator.total_duration_ms, 1.0)
    observed = accumulator.integrated_stimulus / safe_duration
    error = (observed - prepared.plan.homeostatic_targets) / prepared.plan.stimulus_scales
    roundoff = (
        32.0
        * jnp.finfo(error.dtype).eps
        * jnp.maximum(
            1.0,
            jnp.maximum(jnp.abs(observed), jnp.abs(prepared.plan.homeostatic_targets)),
        )
        / prepared.plan.stimulus_scales
    )
    effective_deadband = jnp.maximum(prepared.plan.deadband, roundoff)
    homeostatic = jnp.all(jnp.abs(error) <= effective_deadband)
    finite = jnp.all(jnp.isfinite(observed)) & jnp.all(jnp.isfinite(error))
    return GrowthStimulusEvaluation(
        observed,
        prepared.plan.homeostatic_targets,
        effective_deadband,
        error,
        jnp.asarray(homeostatic),
        jnp.asarray(sufficient),
        finite,
        prepared.prepared_id,
    )


class GrowthProposal(StrictModule, NonTrainableState):
    """Uncommitted slow update generated from complete-cycle statistics."""

    source_state_id: str = eqx.field(static=True)
    candidate: LogGrowthTensorState
    stimulus: GrowthStimulusEvaluation
    log_increment: Array
    requested_step_size_ms: float = eqx.field(static=True)
    effective_step_size_ms: float = eqx.field(static=True)
    refinement_level: int = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_state_id: str,
        candidate: LogGrowthTensorState,
        stimulus: GrowthStimulusEvaluation,
        log_increment: ArrayLike,
        requested_step_size_ms: float,
        effective_step_size_ms: float,
        refinement_level: int,
        /,
    ):
        increment = _finite_real_array("log_increment", log_increment, 3)
        self.source_state_id = source_state_id
        self.candidate = candidate
        self.stimulus = stimulus
        self.log_increment = jnp.asarray(increment)
        self.requested_step_size_ms = requested_step_size_ms
        self.effective_step_size_ms = effective_step_size_ms
        self.refinement_level = refinement_level
        self.proposal_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-growth-proposal",
                "source_state": source_state_id,
                "candidate_state": candidate.state_id,
                "increment": array_tree_fingerprint(increment),
                "requested_step_size_ms": requested_step_size_ms,
                "effective_step_size_ms": effective_step_size_ms,
                "refinement_level": refinement_level,
            }
        )


def propose_growth_step(
    prepared: PreparedGrowth,
    state: LogGrowthTensorState,
    accumulator: GrowthCycleAccumulator,
    requested_step_size_ms: float,
    /,
    *,
    refinement_level: int = 0,
) -> GrowthProposal:
    """Propose one bounded-duration slow update without mutating committed state."""

    _validate_growth_state(prepared, state)
    requested = _positive_scalar("requested_step_size_ms", requested_step_size_ms)
    level = int(refinement_level)
    if level < 0 or level != refinement_level:
        raise ValueError("refinement_level must be a nonnegative integer.")
    if level > prepared.plan.maximum_refinements:
        raise ValueError("refinement_level exceeds the prepared growth policy.")
    stimulus = evaluate_homeostatic_stimulus(prepared, accumulator)
    absolute_error = jnp.abs(stimulus.normalized_error)
    driven_error = jnp.sign(stimulus.normalized_error) * jnp.maximum(
        absolute_error - stimulus.effective_deadband, 0.0
    )
    channel_rate = prepared.plan.growth_gains * driven_error
    tensor_rate = oe.contract("pc,pcij->pij", channel_rate, prepared.direction_projectors)
    effective = requested / (2**level)
    increment = effective * tensor_rate
    candidate = LogGrowthTensorState(
        np.asarray(state.log_growth_tensor + increment),
        float(np.asarray(state.slow_time_ms)) + effective,
        prepared.prepared_id,
    )
    return GrowthProposal(
        state.state_id,
        candidate,
        stimulus,
        increment,
        requested,
        effective,
        level,
    )


def refine_growth_proposal(
    prepared: PreparedGrowth,
    state: LogGrowthTensorState,
    accumulator: GrowthCycleAccumulator,
    proposal: GrowthProposal,
    /,
) -> GrowthProposal:
    """Halve a rejected slow step while retaining the original requested horizon."""

    _validate_growth_state(prepared, state)
    if not isinstance(proposal, GrowthProposal):
        raise TypeError("proposal must be GrowthProposal.")
    if proposal.source_state_id != state.state_id:
        raise ValueError("Cannot refine a proposal from a different committed state.")
    next_level = proposal.refinement_level + 1
    if next_level > prepared.plan.maximum_refinements:
        raise ValueError("Growth refinement budget is exhausted.")
    return propose_growth_step(
        prepared,
        state,
        accumulator,
        proposal.requested_step_size_ms,
        refinement_level=next_level,
    )


class GrowthEvidence(StrictModule, NonTrainableState):
    maximum_increment: Array
    maximum_log_magnitude: Array
    symmetry_residual: Array
    minimum_growth_jacobian: Array
    reconstruction_error: Array
    fresh: Array
    sufficient_cycles: Array
    finite: Array
    positive: Array
    increment_valid: Array
    log_bound_valid: Array
    refinement_available: Array
    passed: Array
    status: Array
    proposal_id: str = eqx.field(static=True)


def evaluate_growth_proposal(
    prepared: PreparedGrowth,
    state: LogGrowthTensorState,
    proposal: GrowthProposal,
    /,
) -> GrowthEvidence:
    """Certify freshness, positivity, boundedness, and slow-step resolution."""

    _validate_growth_state(prepared, state)
    if not isinstance(proposal, GrowthProposal):
        raise TypeError("proposal must be GrowthProposal.")
    candidate = proposal.candidate
    if candidate.prepared_id != prepared.prepared_id:
        raise ValueError("Growth proposal targets a different prepared epoch.")
    fresh = proposal.source_state_id == state.state_id
    increment_norm = jnp.sqrt(jnp.sum(proposal.log_increment**2, axis=(-2, -1)))
    maximum_increment = jnp.max(increment_norm)
    eigenvalues = jnp.linalg.eigvalsh(candidate.log_growth_tensor)
    maximum_log = jnp.max(jnp.abs(eigenvalues))
    symmetry = jnp.max(
        jnp.abs(
            candidate.log_growth_tensor
            - jnp.swapaxes(candidate.log_growth_tensor, -1, -2)
        )
    )
    identity = jnp.broadcast_to(
        jnp.eye(prepared.plan.dimension, dtype=candidate.log_growth_tensor.dtype),
        candidate.log_growth_tensor.shape,
    )
    kinematics = evaluate_growth_kinematics(prepared, candidate, identity)
    minimum_growth_jacobian = jnp.min(kinematics.growth_jacobian)
    reconstruction_error = jnp.max(kinematics.reconstruction_error)
    finite = (
        proposal.stimulus.finite
        & jnp.all(jnp.isfinite(proposal.log_increment))
        & jnp.all(kinematics.finite)
    )
    positive = jnp.all(kinematics.growth_positive)
    increment_valid = maximum_increment <= prepared.plan.maximum_log_increment
    log_bound_valid = maximum_log <= prepared.plan.maximum_log_magnitude
    sufficient = proposal.stimulus.sufficient_cycles
    passed = (
        jnp.asarray(fresh)
        & sufficient
        & finite
        & positive
        & increment_valid
        & log_bound_valid
        & (symmetry <= prepared.plan.tensor_tolerance)
    )
    refinement_available = proposal.refinement_level < prepared.plan.maximum_refinements
    status = jnp.asarray(int(GrowthStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(~log_bound_valid, int(GrowthStatus.LOG_BOUND_EXCEEDED), status)
    status = jnp.where(
        ~increment_valid,
        jnp.where(
            refinement_available,
            int(GrowthStatus.INCREMENT_TOO_LARGE),
            int(GrowthStatus.REFINEMENT_EXHAUSTED),
        ),
        status,
    )
    status = jnp.where(~positive, int(GrowthStatus.POSITIVITY_FAILURE), status)
    status = jnp.where(~sufficient, int(GrowthStatus.INSUFFICIENT_CYCLES), status)
    status = jnp.where(~jnp.asarray(fresh), int(GrowthStatus.STALE_EPOCH), status)
    status = jnp.where(~finite, int(GrowthStatus.NONFINITE), status)
    return GrowthEvidence(
        maximum_increment,
        maximum_log,
        symmetry,
        minimum_growth_jacobian,
        reconstruction_error,
        jnp.asarray(fresh),
        sufficient,
        finite,
        positive,
        increment_valid,
        log_bound_valid,
        jnp.asarray(refinement_available),
        passed,
        status,
        proposal.proposal_id,
    )


class GrowthCommitResult(StrictModule, NonTrainableState):
    state: LogGrowthTensorState
    evidence: GrowthEvidence
    committed: Array
    status: Array
    result_id: str = eqx.field(static=True)


def commit_growth_step(
    prepared: PreparedGrowth,
    source: LogGrowthTensorState,
    proposal: GrowthProposal,
    evidence: GrowthEvidence,
    /,
) -> GrowthCommitResult:
    """Atomically commit passing evidence, otherwise retain the source state."""

    _validate_growth_state(prepared, source)
    if not isinstance(proposal, GrowthProposal):
        raise TypeError("proposal must be GrowthProposal.")
    if not isinstance(evidence, GrowthEvidence):
        raise TypeError("evidence must be GrowthEvidence.")
    if evidence.proposal_id != proposal.proposal_id:
        raise ValueError("Growth evidence belongs to a different proposal.")
    fresh = proposal.source_state_id == source.state_id
    accepted = fresh and bool(np.asarray(evidence.passed))
    selected = proposal.candidate if accepted else source
    status = (
        int(GrowthStatus.SUCCESS)
        if accepted
        else (
            int(GrowthStatus.STALE_EPOCH)
            if not fresh
            else int(np.asarray(evidence.status))
        )
    )
    result_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-growth-commit",
            "source": source.state_id,
            "proposal": proposal.proposal_id,
            "selected": selected.state_id,
            "committed": accepted,
            "status": status,
        }
    )
    return GrowthCommitResult(
        selected,
        evidence,
        jnp.asarray(accepted),
        jnp.asarray(status, dtype=jnp.int32),
        result_id,
    )


class GrowthEpochRebuildRequirements(StrictModule, NonTrainableState):
    """Mandatory work invalidated by an anatomy/reference epoch replacement."""

    transfer_growth_state: bool = eqx.field(static=True)
    rebuild_mechanics_reference: bool = eqx.field(static=True)
    rebuild_cycle_aggregator: bool = eqx.field(static=True)
    rebuild_observation_operators: bool = eqx.field(static=True)
    ordinary_gradient_supported: bool = eqx.field(static=True)
    differentiation: Literal["discrete-stop-gradient"] = eqx.field(static=True)

    def __init__(self):
        self.transfer_growth_state = True
        self.rebuild_mechanics_reference = True
        self.rebuild_cycle_aggregator = True
        self.rebuild_observation_operators = True
        self.ordinary_gradient_supported = False
        self.differentiation = "discrete-stop-gradient"


def discrete_growth_log_transfer(
    log_growth_tensor: ArrayLike, transfer_weights: ArrayLike, /
) -> Array:
    """Map log tensors at a discrete epoch boundary with a zero ordinary Jacobian."""

    log_tensor = jnp.asarray(log_growth_tensor)
    weights = jnp.asarray(transfer_weights)
    if log_tensor.ndim != 3 or log_tensor.shape[-1] != log_tensor.shape[-2]:
        raise ValueError("log_growth_tensor must have shape (source_points, d, d).")
    if weights.ndim != 2 or weights.shape[1] != log_tensor.shape[0]:
        raise ValueError(
            "transfer_weights must have shape (target_points, source_points)."
        )
    mapped = oe.contract("ts,sij->tij", weights, log_tensor)
    symmetric = 0.5 * (mapped + jnp.swapaxes(mapped, -1, -2))
    return jax.lax.stop_gradient(symmetric)


class GrowthEpochTransfer(StrictModule, NonTrainableState):
    source_prepared_id: str = eqx.field(static=True)
    source_state_id: str = eqx.field(static=True)
    target: PreparedGrowth
    transfer_weights: Array
    requirements: GrowthEpochRebuildRequirements
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: PreparedGrowth,
        source_state: LogGrowthTensorState,
        target: PreparedGrowth,
        transfer_weights: ArrayLike,
        /,
    ):
        _validate_growth_state(source, source_state)
        if not isinstance(target, PreparedGrowth):
            raise TypeError("target must be PreparedGrowth.")
        if source.epoch.epoch_id == target.epoch.epoch_id:
            raise ValueError("Epoch transfer requires a distinct target epoch.")
        if source.plan.direction_ids != target.plan.direction_ids:
            raise ValueError("Source and target growth channel identities must agree.")
        weights = _finite_real_array("transfer_weights", transfer_weights, 2)
        expected = (target.plan.point_count, source.plan.point_count)
        if weights.shape != expected:
            raise ValueError(f"transfer_weights must have shape {expected}.")
        self.source_prepared_id = source.prepared_id
        self.source_state_id = source_state.state_id
        self.target = target
        self.transfer_weights = jnp.asarray(weights)
        self.requirements = GrowthEpochRebuildRequirements()
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-growth-epoch-transfer",
                "source_prepared": source.prepared_id,
                "source_state": source_state.state_id,
                "target_prepared": target.prepared_id,
                "weights": array_tree_fingerprint(weights),
                "differentiation": self.requirements.differentiation,
            }
        )


class GrowthEpochCandidate(StrictModule, NonTrainableState):
    transfer: GrowthEpochTransfer
    state: LogGrowthTensorState
    accumulator: GrowthCycleAccumulator
    candidate_id: str = eqx.field(static=True)


def propose_growth_epoch_transfer(
    source: PreparedGrowth,
    source_state: LogGrowthTensorState,
    target: PreparedGrowth,
    transfer_weights: ArrayLike,
    /,
) -> GrowthEpochCandidate:
    """Propose a stopped-gradient state map and an empty target-cycle window."""

    transfer = GrowthEpochTransfer(source, source_state, target, transfer_weights)
    mapped = discrete_growth_log_transfer(
        source_state.log_growth_tensor, transfer.transfer_weights
    )
    candidate_state = LogGrowthTensorState(
        np.asarray(mapped),
        float(np.asarray(jax.lax.stop_gradient(source_state.slow_time_ms))),
        target.prepared_id,
    )
    accumulator = initialize_growth_cycle_accumulator(target)
    candidate_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-growth-epoch-candidate",
            "transfer": transfer.transfer_id,
            "state": candidate_state.state_id,
            "cycle_history": "invalidated-and-empty",
        }
    )
    return GrowthEpochCandidate(transfer, candidate_state, accumulator, candidate_id)


class GrowthEpochEvidence(StrictModule, NonTrainableState):
    maximum_row_sum_error: Array
    minimum_weight: Array
    minimum_growth_jacobian: Array
    source_fresh: Array
    target_distinct: Array
    transfer_valid: Array
    state_positive: Array
    cycle_history_invalidated: Array
    finite: Array
    passed: Array
    status: Array
    candidate_id: str = eqx.field(static=True)


def evaluate_growth_epoch_transfer(
    source: PreparedGrowth,
    source_state: LogGrowthTensorState,
    candidate: GrowthEpochCandidate,
    /,
    *,
    transfer_tolerance: float = 1.0e-6,
) -> GrowthEpochEvidence:
    """Certify a convex log-tensor transfer and all epoch invalidations."""

    _validate_growth_state(source, source_state)
    if not isinstance(candidate, GrowthEpochCandidate):
        raise TypeError("candidate must be GrowthEpochCandidate.")
    tolerance = _nonnegative_scalar("transfer_tolerance", transfer_tolerance)
    transfer = candidate.transfer
    weights = transfer.transfer_weights
    row_error = jnp.max(jnp.abs(jnp.sum(weights, axis=1) - 1.0))
    minimum_weight = jnp.min(weights)
    source_fresh = (
        transfer.source_prepared_id == source.prepared_id
        and transfer.source_state_id == source_state.state_id
    )
    target_distinct = transfer.target.epoch.epoch_id != source.epoch.epoch_id
    transfer_valid = (row_error <= tolerance) & (minimum_weight >= -tolerance)
    target_identity = jnp.broadcast_to(
        jnp.eye(
            transfer.target.plan.dimension,
            dtype=candidate.state.log_growth_tensor.dtype,
        ),
        candidate.state.log_growth_tensor.shape,
    )
    kinematics = evaluate_growth_kinematics(
        transfer.target, candidate.state, target_identity
    )
    positive = jnp.all(kinematics.growth_positive)
    finite = jnp.all(jnp.isfinite(weights)) & jnp.all(kinematics.finite)
    history_invalidated = (
        candidate.accumulator.prepared_id == transfer.target.prepared_id
        and candidate.accumulator.cycle_count == 0
        and candidate.accumulator.last_cycle_index == -1
    )
    passed = (
        jnp.asarray(source_fresh)
        & jnp.asarray(target_distinct)
        & transfer_valid
        & positive
        & jnp.asarray(history_invalidated)
        & finite
    )
    status = jnp.asarray(int(GrowthStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(~transfer_valid, int(GrowthStatus.INVALID_TRANSFER), status)
    status = jnp.where(~positive, int(GrowthStatus.POSITIVITY_FAILURE), status)
    status = jnp.where(~jnp.asarray(source_fresh), int(GrowthStatus.STALE_EPOCH), status)
    status = jnp.where(~finite, int(GrowthStatus.NONFINITE), status)
    return GrowthEpochEvidence(
        row_error,
        minimum_weight,
        jnp.min(kinematics.growth_jacobian),
        jnp.asarray(source_fresh),
        jnp.asarray(target_distinct),
        transfer_valid,
        positive,
        jnp.asarray(history_invalidated),
        finite,
        passed,
        status,
        candidate.candidate_id,
    )


class GrowthEpochCommitResult(StrictModule, NonTrainableState):
    prepared: PreparedGrowth
    state: LogGrowthTensorState
    accumulator: GrowthCycleAccumulator
    evidence: GrowthEpochEvidence
    requirements: GrowthEpochRebuildRequirements
    committed: Array
    rebuild_required: Array
    status: Array
    result_id: str = eqx.field(static=True)


def commit_growth_epoch_transfer(
    source: PreparedGrowth,
    source_state: LogGrowthTensorState,
    source_accumulator: GrowthCycleAccumulator,
    candidate: GrowthEpochCandidate,
    evidence: GrowthEpochEvidence,
    /,
) -> GrowthEpochCommitResult:
    """Atomically select the rebuilt target epoch or retain every source object."""

    _validate_growth_state(source, source_state)
    if source_accumulator.prepared_id != source.prepared_id:
        raise ValueError("Source accumulator belongs to a different epoch.")
    if not isinstance(candidate, GrowthEpochCandidate):
        raise TypeError("candidate must be GrowthEpochCandidate.")
    if not isinstance(evidence, GrowthEpochEvidence):
        raise TypeError("evidence must be GrowthEpochEvidence.")
    if evidence.candidate_id != candidate.candidate_id:
        raise ValueError("Epoch evidence belongs to a different candidate.")
    transfer = candidate.transfer
    fresh = (
        transfer.source_prepared_id == source.prepared_id
        and transfer.source_state_id == source_state.state_id
    )
    accepted = fresh and bool(np.asarray(evidence.passed))
    prepared = transfer.target if accepted else source
    state = candidate.state if accepted else source_state
    accumulator = candidate.accumulator if accepted else source_accumulator
    status = (
        int(GrowthStatus.SUCCESS)
        if accepted
        else (
            int(GrowthStatus.STALE_EPOCH)
            if not fresh
            else int(np.asarray(evidence.status))
        )
    )
    result_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-growth-epoch-commit",
            "source": source.prepared_id,
            "candidate": candidate.candidate_id,
            "selected": prepared.prepared_id,
            "committed": accepted,
            "status": status,
        }
    )
    return GrowthEpochCommitResult(
        prepared,
        state,
        accumulator,
        evidence,
        transfer.requirements,
        jnp.asarray(accepted),
        jnp.asarray(accepted),
        jnp.asarray(status, dtype=jnp.int32),
        result_id,
    )


__all__ = [
    "ContinuumGrowthFidelity",
    "GrowthCommitResult",
    "GrowthCycleAccumulator",
    "GrowthCycleSummary",
    "GrowthEpochCandidate",
    "GrowthEpochCommitResult",
    "GrowthEpochEvidence",
    "GrowthEpochRebuildRequirements",
    "GrowthEpochTransfer",
    "GrowthEvidence",
    "GrowthKinematics",
    "GrowthPlan",
    "GrowthProposal",
    "GrowthReferenceEpoch",
    "GrowthStatus",
    "GrowthStimulusEvaluation",
    "LogGrowthTensorState",
    "PreparedGrowth",
    "accumulate_growth_cycle",
    "aggregate_growth_cycle",
    "commit_growth_epoch_transfer",
    "commit_growth_step",
    "discrete_growth_log_transfer",
    "evaluate_growth_epoch_transfer",
    "evaluate_growth_kinematics",
    "evaluate_growth_proposal",
    "evaluate_homeostatic_stimulus",
    "initialize_growth_cycle_accumulator",
    "initialize_growth_state",
    "prepare_growth",
    "propose_growth_epoch_transfer",
    "propose_growth_step",
    "refine_growth_proposal",
]
