#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import interface_distance_metrics, phase_geometry_metrics
from ._stefan import (
    compare_stefan_representations,
    ExplicitFrontStefanPINN,
    ImplicitLevelSetStefanPINN,
    OnePhaseStefanParameters,
    ReferenceMapStefanPINN,
    stefan_collocation_batch,
    StefanBoundaryData,
    StefanRepresentationComparison,
)


class ExactStefanFields(StrictModule, NonTrainableState):
    initial_front: float

    def __init__(self, initial_front: float, /):
        self.initial_front = _positive_float(initial_front, "initial_front")

    def temperature(self, point: Array, /) -> Array:
        x, time = point
        return jnp.exp(time + self.initial_front - x) - 1.0

    def front(self, time_coordinate: Array, /) -> Array:
        return time_coordinate[0] + self.initial_front

    def level_set(self, point: Array, /) -> Array:
        x, time = point
        return x - time - self.initial_front

    def reference_map(self, point: Array, /) -> Array:
        coordinate, time = point
        return coordinate * (time + self.initial_front)

    def reference_temperature(self, point: Array, /) -> Array:
        coordinate, time = point
        front = time + self.initial_front
        return jnp.exp(front * (1.0 - coordinate)) - 1.0

    def initial_temperature(self, x: Array, /) -> Array:
        return jnp.exp(self.initial_front - x) - 1.0

    def boundary_temperature(self, time: Array, /) -> Array:
        return jnp.exp(time + self.initial_front) - 1.0


class ExactStefanBenchmark(StrictModule, NonTrainableState):
    parameters: OnePhaseStefanParameters
    data: StefanBoundaryData
    fields: ExactStefanFields
    interface_width: float

    def __init__(
        self,
        *,
        initial_front: float = 0.5,
        final_time: float = 0.5,
        domain_length: float = 1.5,
        interface_width: float = 0.05,
    ):
        fields = ExactStefanFields(initial_front)
        self.parameters = OnePhaseStefanParameters(
            diffusivity=1.0,
            conductivity=1.0,
            volumetric_latent_heat=1.0,
            melting_temperature=0.0,
            initial_front=initial_front,
            domain_length=domain_length,
            final_time=final_time,
        )
        self.data = StefanBoundaryData(
            fields.initial_temperature,
            fields.boundary_temperature,
        )
        self.fields = fields
        self.interface_width = _positive_float(interface_width, "interface_width")

    def models(self):
        return (
            ExplicitFrontStefanPINN(self.fields.temperature, self.fields.front),
            ImplicitLevelSetStefanPINN(
                self.fields.temperature,
                self.fields.level_set,
            ),
            ReferenceMapStefanPINN(
                self.fields.reference_temperature,
                self.fields.reference_map,
            ),
        )

    def run(
        self,
        /,
        *,
        points_per_block: int = 256,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> StefanRepresentationComparison:
        batch = stefan_collocation_batch(
            self.parameters,
            interior_points=points_per_block,
            ambient_points=points_per_block,
            boundary_points=points_per_block,
            interface_points=points_per_block,
            initial_points=points_per_block,
            key=key,
        )
        explicit, implicit, reference = self.models()
        return compare_stefan_representations(
            explicit,
            implicit,
            reference,
            batch,
            self.parameters,
            self.data,
            interface_width=self.interface_width,
            key=key,
        )


class MullinsSekerkaReport(StrictModule):
    relative_l2_error: Array
    maximum_relative_mode_error: Array
    predicted_dominant_mode: Array
    reference_dominant_mode: Array


def mullins_sekerka_benchmark(
    predicted_amplitudes: ArrayLike,
    times: ArrayLike,
    modes: ArrayLike,
    growth_rates: ArrayLike,
    initial_amplitudes: ArrayLike,
    /,
) -> MullinsSekerkaReport:
    """Compare perturbation-mode evolution with linear Mullins–Sekerka theory."""

    prediction = jnp.asarray(predicted_amplitudes, dtype=float)
    times_ = jnp.asarray(times, dtype=float)
    modes_ = jnp.asarray(modes, dtype=int)
    rates = jnp.asarray(growth_rates, dtype=float)
    initial = jnp.asarray(initial_amplitudes, dtype=float)
    if times_.ndim != 1 or modes_.ndim != 1 or times_.size == 0 or modes_.size == 0:
        raise ValueError("times and modes must be non-empty vectors.")
    if rates.shape != modes_.shape or initial.shape != modes_.shape:
        raise ValueError("growth_rates and initial_amplitudes must match modes.")
    if prediction.shape != (times_.size, modes_.size):
        raise ValueError("predicted_amplitudes must have shape (time, mode).")
    reference = initial[None, :] * jnp.exp(times_[:, None] * rates[None, :])
    error = prediction - reference
    relative = jnp.sqrt(jnp.sum(error**2) / jnp.maximum(jnp.sum(reference**2), 1.0e-30))
    mode_scale = jnp.maximum(jnp.max(jnp.abs(reference), axis=0), 1.0e-30)
    mode_error = jnp.max(jnp.abs(error), axis=0) / mode_scale
    return MullinsSekerkaReport(
        relative_l2_error=relative,
        maximum_relative_mode_error=jnp.max(mode_error),
        predicted_dominant_mode=modes_[jnp.argmax(jnp.abs(prediction[-1]))],
        reference_dominant_mode=modes_[jnp.argmax(jnp.abs(reference[-1]))],
    )


class TopologyEventReport(StrictModule):
    component_count_correct: Array
    event_detected: Array
    event_time_error: Array
    event_order_correct: Array


def topology_event_benchmark(
    predicted_component_counts: ArrayLike,
    reference_component_counts: ArrayLike,
    times: ArrayLike,
    /,
) -> TopologyEventReport:
    """Evaluate component histories and the first topology-event time."""

    predicted = jnp.asarray(predicted_component_counts, dtype=jnp.int32)
    reference = jnp.asarray(reference_component_counts, dtype=jnp.int32)
    times_ = jnp.asarray(times, dtype=float)
    if predicted.shape != reference.shape or predicted.shape != times_.shape:
        raise ValueError(
            "Topology histories and times must have identical vector shapes."
        )
    if predicted.ndim != 1 or predicted.size < 2:
        raise ValueError("Topology benchmarks require at least two time points.")
    predicted_change = predicted[1:] != predicted[:-1]
    reference_change = reference[1:] != reference[:-1]
    predicted_has = jnp.any(predicted_change)
    reference_has = jnp.any(reference_change)
    predicted_index = jnp.argmax(predicted_change) + 1
    reference_index = jnp.argmax(reference_change) + 1
    time_error = jnp.where(
        predicted_has & reference_has,
        jnp.abs(times_[predicted_index] - times_[reference_index]),
        jnp.where(predicted_has == reference_has, 0.0, jnp.inf),
    )
    return TopologyEventReport(
        component_count_correct=jnp.mean((predicted == reference).astype(float)),
        event_detected=predicted_has == reference_has,
        event_time_error=time_error,
        event_order_correct=jnp.where(
            predicted_has & reference_has,
            predicted[predicted_index] - predicted[predicted_index - 1]
            == reference[reference_index] - reference[reference_index - 1],
            predicted_has == reference_has,
        ),
    )


class HysingBubbleReport(StrictModule):
    area: Array
    circularity: Array
    centroid: Array
    mean_rise_velocity: Array


def hysing_bubble_benchmark(
    phase_fraction: ArrayLike,
    coordinates: ArrayLike,
    cell_measures: ArrayLike,
    vertical_velocity: ArrayLike,
    interface_points: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
) -> HysingBubbleReport:
    """Compute the published Hysing area, circularity, centroid, and rise speed."""

    fraction = jnp.asarray(phase_fraction, dtype=float)
    points = jnp.asarray(coordinates, dtype=float)
    measures = jnp.asarray(cell_measures, dtype=float)
    velocity = jnp.asarray(vertical_velocity, dtype=float)
    if velocity.shape != fraction.shape:
        raise ValueError("vertical_velocity must match phase_fraction.")
    phase = phase_geometry_metrics(fraction, points, measures, mask=mask)
    contour = jnp.asarray(interface_points, dtype=float)
    if contour.ndim != 2 or contour.shape[0] < 3 or contour.shape[1] != 2:
        raise ValueError("interface_points must contain an ordered closed 2-D contour.")
    edge = jnp.roll(contour, -1, axis=0) - contour
    perimeter = jnp.sum(jnp.sqrt(jnp.sum(edge * edge, axis=-1)))
    circularity = 2.0 * jnp.sqrt(jnp.pi * phase.measure) / perimeter
    active = (
        jnp.ones_like(fraction, dtype=bool)
        if mask is None
        else jnp.asarray(mask, dtype=bool)
    )
    momentum = jnp.sum(jnp.where(active, fraction * measures * velocity, 0.0))
    rise = momentum / jnp.maximum(phase.measure, 1.0e-30)
    return HysingBubbleReport(
        area=phase.measure,
        circularity=circularity,
        centroid=phase.centroid,
        mean_rise_velocity=rise,
    )


class FSIReport(StrictModule):
    tip_relative_l2: Array
    drag_relative_l2: Array
    lift_relative_l2: Array
    dominant_frequency_error: Array


def turek_hron_fsi_benchmark(
    predicted_tip: ArrayLike,
    reference_tip: ArrayLike,
    predicted_drag: ArrayLike,
    reference_drag: ArrayLike,
    predicted_lift: ArrayLike,
    reference_lift: ArrayLike,
    step_size: float,
    /,
) -> FSIReport:
    """Evaluate Turek–Hron tip, force, and dominant-frequency observables."""

    predicted_tip_ = jnp.asarray(predicted_tip, dtype=float)
    reference_tip_ = jnp.asarray(reference_tip, dtype=float)
    predicted_drag_ = jnp.asarray(predicted_drag, dtype=float)
    reference_drag_ = jnp.asarray(reference_drag, dtype=float)
    predicted_lift_ = jnp.asarray(predicted_lift, dtype=float)
    reference_lift_ = jnp.asarray(reference_lift, dtype=float)
    shapes = {
        predicted_tip_.shape,
        reference_tip_.shape,
        predicted_drag_.shape,
        reference_drag_.shape,
        predicted_lift_.shape,
        reference_lift_.shape,
    }
    if len(shapes) != 1 or predicted_tip_.ndim != 1 or predicted_tip_.size < 3:
        raise ValueError("FSI observable histories must share one vector shape.")
    dt = _positive_float(step_size, "step_size")
    predicted_frequency = _dominant_frequency(predicted_tip_, dt)
    reference_frequency = _dominant_frequency(reference_tip_, dt)
    return FSIReport(
        tip_relative_l2=_relative_l2(predicted_tip_, reference_tip_),
        drag_relative_l2=_relative_l2(predicted_drag_, reference_drag_),
        lift_relative_l2=_relative_l2(predicted_lift_, reference_lift_),
        dominant_frequency_error=jnp.abs(predicted_frequency - reference_frequency),
    )


class ObstacleComplementarityReport(StrictModule):
    gap_violation: Array
    dual_violation: Array
    complementarity_residual: Array
    active_fraction: Array


def obstacle_complementarity_benchmark(
    solution: ArrayLike,
    obstacle: ArrayLike,
    multiplier: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
    active_tolerance: float = 1.0e-8,
) -> ObstacleComplementarityReport:
    """Keep primal feasibility, dual feasibility, and complementarity separate."""

    value = jnp.asarray(solution, dtype=float)
    obstacle_ = jnp.asarray(obstacle, dtype=float)
    dual = jnp.asarray(multiplier, dtype=float)
    if value.shape != obstacle_.shape or value.shape != dual.shape or value.size == 0:
        raise ValueError("Obstacle solution, obstacle, and multiplier shapes must match.")
    weight = (
        jnp.ones_like(value)
        if weights is None
        else jnp.broadcast_to(jnp.asarray(weights), value.shape)
    )
    if bool(jnp.any(~jnp.isfinite(weight) | (weight < 0.0))):
        raise ValueError("Obstacle weights must be finite and nonnegative.")
    gap = value - obstacle_
    normalization = jnp.maximum(jnp.sum(weight), 1.0e-30)
    return ObstacleComplementarityReport(
        gap_violation=jnp.sqrt(jnp.sum(weight * jax_relu(-gap) ** 2) / normalization),
        dual_violation=jnp.sqrt(jnp.sum(weight * jax_relu(-dual) ** 2) / normalization),
        complementarity_residual=jnp.sqrt(
            jnp.sum(weight * (gap * dual) ** 2) / normalization
        ),
        active_fraction=jnp.sum(weight * (gap <= active_tolerance)) / normalization,
    )


class PhaseFieldFractureReport(StrictModule):
    irreversibility_violation: Array
    load_displacement_relative_l2: Array
    fracture_energy_final: Array
    crack_path_symmetric_distance: Array
    crack_path_hausdorff: Array


def phase_field_fracture_benchmark(
    damage_history: ArrayLike,
    predicted_load_displacement: ArrayLike,
    reference_load_displacement: ArrayLike,
    fracture_energy_history: ArrayLike,
    predicted_crack_points: ArrayLike,
    reference_crack_points: ArrayLike,
    /,
) -> PhaseFieldFractureReport:
    """Evaluate irreversibility, response, energy, and extracted crack geometry."""

    damage = jnp.asarray(damage_history, dtype=float)
    predicted_curve = jnp.asarray(predicted_load_displacement, dtype=float)
    reference_curve = jnp.asarray(reference_load_displacement, dtype=float)
    energy = jnp.asarray(fracture_energy_history, dtype=float)
    if damage.ndim < 2 or damage.shape[0] < 2:
        raise ValueError("damage_history requires time and spatial axes.")
    if predicted_curve.shape != reference_curve.shape or predicted_curve.size == 0:
        raise ValueError("Load-displacement curves must share one non-empty shape.")
    if energy.ndim != 1 or energy.shape[0] != damage.shape[0]:
        raise ValueError("fracture_energy_history must align with damage time.")
    geometry = interface_distance_metrics(predicted_crack_points, reference_crack_points)
    violation = jnp.max(jax_relu(damage[:-1] - damage[1:]))
    return PhaseFieldFractureReport(
        irreversibility_violation=violation,
        load_displacement_relative_l2=_relative_l2(predicted_curve, reference_curve),
        fracture_energy_final=energy[-1],
        crack_path_symmetric_distance=geometry.symmetric_mean_distance,
        crack_path_hausdorff=geometry.hausdorff_distance,
    )


class FreeBoundaryDatasetSplit(StrictModule, NonTrainableState):
    train_indices: Array
    validation_indices: Array
    interpolation_test_indices: Array
    extrapolation_test_indices: Array


def trajectory_disjoint_ood_split(
    trajectory_ids: Sequence[str],
    ood_mask: ArrayLike,
    /,
    *,
    validation_fraction: float = 0.15,
    interpolation_test_fraction: float = 0.15,
) -> FreeBoundaryDatasetSplit:
    """Split whole trajectories, reserving explicitly declared OOD trajectories."""

    ids = tuple(str(value) for value in trajectory_ids)
    if not ids or any(not value for value in ids):
        raise ValueError("trajectory_ids must be non-empty strings.")
    ood = np.asarray(ood_mask, dtype=bool)
    if ood.shape != (len(ids),):
        raise ValueError("ood_mask must contain one value per case.")
    validation = float(validation_fraction)
    interpolation = float(interpolation_test_fraction)
    if validation < 0.0 or interpolation < 0.0 or validation + interpolation >= 1.0:
        raise ValueError("ID split fractions must be nonnegative and sum below one.")
    groups: dict[str, list[int]] = {}
    group_ood: dict[str, bool] = {}
    for index, identifier in enumerate(ids):
        groups.setdefault(identifier, []).append(index)
        previous = group_ood.setdefault(identifier, bool(ood[index]))
        if previous != bool(ood[index]):
            raise ValueError("Every trajectory must have one consistent OOD designation.")
    id_groups = sorted(
        (identifier for identifier in groups if not group_ood[identifier]),
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    count = len(id_groups)
    validation_count = round(validation * count)
    interpolation_count = round(interpolation * count)
    validation_ids = set(id_groups[:validation_count])
    interpolation_ids = set(
        id_groups[validation_count : validation_count + interpolation_count]
    )
    train_ids = set(id_groups) - validation_ids - interpolation_ids
    return FreeBoundaryDatasetSplit(
        train_indices=_indices_for(groups, train_ids),
        validation_indices=_indices_for(groups, validation_ids),
        interpolation_test_indices=_indices_for(groups, interpolation_ids),
        extrapolation_test_indices=_indices_for(
            groups,
            {identifier for identifier in groups if group_ood[identifier]},
        ),
    )


def jax_relu(value: ArrayLike, /) -> Array:
    return jnp.maximum(jnp.asarray(value), 0.0)


def _indices_for(groups: dict[str, list[int]], selected: set[str], /) -> Array:
    values = sorted(index for identifier in selected for index in groups[identifier])
    return jnp.asarray(values, dtype=jnp.int32)


def _relative_l2(predicted: Array, reference: Array, /) -> Array:
    return jnp.sqrt(
        jnp.sum((predicted - reference) ** 2)
        / jnp.maximum(jnp.sum(reference**2), 1.0e-30)
    )


def _dominant_frequency(values: Array, step_size: float, /) -> Array:
    centered = values - jnp.mean(values)
    spectrum = jnp.abs(jnp.fft.rfft(centered))
    frequencies = jnp.fft.rfftfreq(values.size, d=step_size)
    usable = spectrum.at[0].set(0.0)
    return frequencies[jnp.argmax(usable)]


def _positive_float(value: float, name: str, /) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return scalar


__all__ = [
    "ExactStefanBenchmark",
    "ExactStefanFields",
    "FSIReport",
    "FreeBoundaryDatasetSplit",
    "HysingBubbleReport",
    "MullinsSekerkaReport",
    "ObstacleComplementarityReport",
    "PhaseFieldFractureReport",
    "TopologyEventReport",
    "hysing_bubble_benchmark",
    "mullins_sekerka_benchmark",
    "obstacle_complementarity_benchmark",
    "phase_field_fracture_benchmark",
    "topology_event_benchmark",
    "trajectory_disjoint_ood_split",
    "turek_hron_fsi_benchmark",
]
