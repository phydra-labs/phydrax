#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle._radiation_geometry import VoxelRadiationGeometryPlan
from ..equations._radiation_interactions import (
    RadiationCrossSectionLibrary,
    RadiationInteractionKind,
)
from ..units import conversion_factor, ELECTRONVOLT


_ELECTRON_REST_ENERGY_EV = 510998.95


class PhotonTransportStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    CROSS_SECTION_UNSUPPORTED = 2
    GEOMETRY_FAILURE = 3
    ANGULAR_SAMPLING_EXHAUSTED = 4
    EVENT_CAPACITY_EXHAUSTED = 5
    NONFINITE = 6


class PhotonTransportResult(StrictModule, NonTrainableState):
    history_ids: Array
    material_kerma: Array
    event_positions: Array
    event_deposited_energy: Array
    event_process: Array
    event_material: Array
    event_active: Array
    escaped_energy: Array
    truncated_energy: Array
    terminal_position: Array
    terminal_direction: Array
    terminal_energy: Array
    event_count: Array
    compton_count: Array
    rayleigh_count: Array
    virtual_count: Array
    scatter_class: Array
    status: Array
    successful: Array
    mean_material_kerma: Array
    standard_error_material_kerma: Array
    mean_escaped_energy: Array
    maximum_ledger_residual: Array
    all_successful: Array
    plan_id: str = eqx.field(static=True)


def _history_key(key: Array, history_id: Array, event: int, stream: int, /) -> Array:
    return jr.fold_in(jr.fold_in(jr.fold_in(key, history_id), event), stream)


def _uniform(key: Array, history_id: Array, event: int, stream: int, shape=()):
    return jr.uniform(_history_key(key, history_id, event, stream), shape)


def _rotate(direction: Array, cosine: Array, azimuth: Array, /) -> Array:
    z = jnp.asarray((0.0, 0.0, 1.0), dtype=direction.dtype)
    x = jnp.asarray((1.0, 0.0, 0.0), dtype=direction.dtype)
    axis = jnp.where(jnp.abs(direction[2]) < 0.9, z, x)
    first = jnp.cross(axis, direction)
    first = first / jnp.linalg.norm(first)
    second = jnp.cross(direction, first)
    sine = jnp.sqrt(jnp.maximum(1.0 - cosine**2, 0.0))
    rotated = cosine * direction + sine * (
        jnp.cos(azimuth) * first + jnp.sin(azimuth) * second
    )
    return rotated / jnp.linalg.norm(rotated)


def _sample_compton(
    key, history_id, event, energy, direction, attempts, electron_rest_energy
):
    cosine = 2.0 * _uniform(key, history_id, event, 20, (attempts,)) - 1.0
    accept_draw = _uniform(key, history_id, event, 21, (attempts,))
    alpha = energy / electron_rest_energy
    ratio = 1.0 / (1.0 + alpha * (1.0 - cosine))
    kernel = ratio**2 * (ratio + 1.0 / ratio - (1.0 - cosine**2))
    accepted = accept_draw <= 0.5 * kernel
    index = jnp.argmax(accepted)
    success = jnp.any(accepted)
    selected_cosine = cosine[index]
    selected_ratio = ratio[index]
    azimuth = 2.0 * jnp.pi * _uniform(key, history_id, event, 22)
    return (
        _rotate(direction, selected_cosine, azimuth),
        energy * selected_ratio,
        success,
    )


def _sample_rayleigh(key, history_id, event, direction, attempts):
    cosine = 2.0 * _uniform(key, history_id, event, 30, (attempts,)) - 1.0
    accepted = _uniform(key, history_id, event, 31, (attempts,)) <= 0.5 * (
        1.0 + cosine**2
    )
    index = jnp.argmax(accepted)
    success = jnp.any(accepted)
    azimuth = 2.0 * jnp.pi * _uniform(key, history_id, event, 32)
    return _rotate(direction, cosine[index], azimuth), success


def _standard_error(values: Array, /) -> Array:
    count = values.shape[0]
    centered = values - jnp.mean(values, axis=0)
    return jnp.where(
        count > 1,
        jnp.sqrt(jnp.sum(centered**2, axis=0) / (count * (count - 1))),
        jnp.zeros(values.shape[1:], dtype=values.dtype),
    )


class PhotonTransportPlan(StrictModule, NonTrainableState):
    geometry: VoxelRadiationGeometryPlan
    cross_sections: RadiationCrossSectionLibrary
    maximum_events: int = eqx.field(static=True)
    angular_sampling_attempts: int = eqx.field(static=True)
    cutoff_energy: float = eqx.field(static=True)
    electron_rest_energy: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: VoxelRadiationGeometryPlan,
        cross_sections: RadiationCrossSectionLibrary,
        /,
        *,
        maximum_events: int,
        cutoff_energy: float,
        angular_sampling_attempts: int = 16,
    ):
        if not isinstance(geometry, VoxelRadiationGeometryPlan):
            raise TypeError("geometry must be VoxelRadiationGeometryPlan.")
        if not isinstance(cross_sections, RadiationCrossSectionLibrary):
            raise TypeError("cross_sections must be RadiationCrossSectionLibrary.")
        events, attempts = int(maximum_events), int(angular_sampling_attempts)
        cutoff = float(cutoff_energy)
        electron_rest = _ELECTRON_REST_ENERGY_EV * float(
            conversion_factor(ELECTRONVOLT, cross_sections.energy_unit)
        )
        if (
            geometry.material_count != cross_sections.material_count
            or events < 1
            or attempts < 1
            or not isfinite(cutoff)
            or cutoff <= 0.0
            or cutoff < float(cross_sections.energy[0])
            or cutoff > float(cross_sections.energy[-1])
        ):
            raise ValueError(
                "Photon transport materials, capacities, or cutoff are invalid."
            )
        self.geometry = geometry
        self.cross_sections = cross_sections
        self.maximum_events = events
        self.angular_sampling_attempts = attempts
        self.cutoff_energy = cutoff
        self.electron_rest_energy = electron_rest
        self.plan_id = canonical_fingerprint(
            {
                "kind": "voxel-delta-photon-transport",
                "geometry": geometry.geometry_id,
                "cross_sections": cross_sections.library_id,
                "maximum_events": events,
                "angular_sampling_attempts": attempts,
                "cutoff_energy": cutoff,
                "electron_rest_energy": electron_rest,
                "deposition_semantics": "photon-kerma-local-recoil",
                "pathwise_differentiability": False,
            }
        )

    def _one(self, key, history_id, origin, direction, energy, weight):
        norm = jnp.linalg.norm(direction)
        initial_valid = (
            jnp.all(jnp.isfinite(origin))
            & jnp.all(jnp.isfinite(direction))
            & jnp.isfinite(energy)
            & jnp.isfinite(weight)
            & (norm > 0.0)
            & (energy >= self.cutoff_energy)
            & (weight >= 0.0)
            & self.geometry.locate(origin).inside
        )
        direction = direction / jnp.where(norm > 0.0, norm, 1.0)
        deposits = jnp.zeros((self.cross_sections.material_count,), dtype=energy.dtype)
        state = (
            origin,
            direction,
            energy,
            initial_valid,
            deposits,
            jnp.asarray(0.0, energy.dtype),
            jnp.asarray(0.0, energy.dtype),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.where(
                initial_valid,
                int(PhotonTransportStatus.SUCCESS),
                int(PhotonTransportStatus.INVALID_INPUT),
            ).astype(jnp.int32),
            jnp.zeros((self.maximum_events, 3), dtype=energy.dtype),
            jnp.zeros((self.maximum_events,), dtype=energy.dtype),
            -jnp.ones((self.maximum_events,), dtype=jnp.int32),
            -jnp.ones((self.maximum_events,), dtype=jnp.int32),
            jnp.zeros((self.maximum_events,), dtype=bool),
        )

        def event_step(event, carry):
            (
                position,
                ray,
                photon_energy,
                live,
                kerma,
                escaped,
                truncated,
                event_count,
                compton_count,
                rayleigh_count,
                virtual_count,
                status,
                event_positions,
                event_deposits,
                event_processes,
                event_materials,
                event_active,
            ) = carry
            majorant = self.cross_sections.majorant(photon_energy)
            cross_supported = (
                jnp.isfinite(majorant)
                & (majorant > 0.0)
                & (photon_energy >= self.cross_sections.energy[0])
                & (photon_energy <= self.cross_sections.energy[-1])
            )
            path_draw = jnp.maximum(
                _uniform(key, history_id, event, 0), jnp.finfo(photon_energy.dtype).tiny
            )
            distance = -jnp.log(path_draw) / jnp.where(majorant > 0.0, majorant, 1.0)
            exit_distance = self.geometry.distance_to_exit(position, ray)
            leaves = live & cross_supported & (distance >= exit_distance)
            collision = live & cross_supported & ~leaves
            trial_position = position + jnp.where(leaves, exit_distance, distance) * ray
            escaped = escaped + jnp.where(leaves, weight * photon_energy, 0.0)
            location = self.geometry.locate(trial_position)
            material = location.material_index
            evaluated = self.cross_sections.evaluate(material, photon_energy)
            accepted_collision = (
                collision
                & location.inside
                & evaluated.successful
                & (
                    _uniform(key, history_id, event, 1)
                    <= evaluated.total / jnp.where(majorant > 0.0, majorant, 1.0)
                )
            )
            virtual = (
                collision & location.inside & evaluated.successful & ~accepted_collision
            )
            cumulative = jnp.cumsum(evaluated.coefficients)
            process_draw = _uniform(key, history_id, event, 2) * evaluated.total
            process = jnp.sum(process_draw > cumulative).astype(jnp.int32)
            photoelectric = accepted_collision & (
                process == int(RadiationInteractionKind.PHOTOELECTRIC)
            )
            compton = accepted_collision & (
                process == int(RadiationInteractionKind.COMPTON)
            )
            rayleigh = accepted_collision & (
                process == int(RadiationInteractionKind.RAYLEIGH)
            )
            compton_direction, compton_energy, compton_success = _sample_compton(
                key,
                history_id,
                event,
                photon_energy,
                ray,
                self.angular_sampling_attempts,
                self.electron_rest_energy,
            )
            rayleigh_direction, rayleigh_success = _sample_rayleigh(
                key,
                history_id,
                event,
                ray,
                self.angular_sampling_attempts,
            )
            sampling_success = (~compton | compton_success) & (
                ~rayleigh | rayleigh_success
            )
            recoil = jnp.where(compton, photon_energy - compton_energy, 0.0)
            local_deposit = jnp.where(photoelectric, photon_energy, recoil)
            next_energy = jnp.where(compton, compton_energy, photon_energy)
            below_cutoff = compton & (next_energy < self.cutoff_energy)
            local_deposit = local_deposit + jnp.where(below_cutoff, next_energy, 0.0)
            next_energy = jnp.where(photoelectric | below_cutoff, 0.0, next_energy)
            kerma = kerma.at[
                jnp.clip(material, 0, self.cross_sections.material_count - 1)
            ].add(jnp.where(accepted_collision, weight * local_deposit, 0.0))
            ray = jnp.where(
                compton,
                compton_direction,
                jnp.where(rayleigh, rayleigh_direction, ray),
            )
            geometry_failure = collision & (~location.inside | ~evaluated.supported)
            sampling_failure = accepted_collision & ~sampling_success
            unsupported_failure = live & ~cross_supported
            failed_energy = jnp.where(
                geometry_failure | sampling_failure | unsupported_failure,
                weight * next_energy,
                0.0,
            )
            truncated = truncated + failed_energy
            next_live = (
                live
                & cross_supported
                & ~leaves
                & ~photoelectric
                & ~below_cutoff
                & ~geometry_failure
                & ~sampling_failure
            )
            status = jnp.where(
                unsupported_failure,
                int(PhotonTransportStatus.CROSS_SECTION_UNSUPPORTED),
                jnp.where(
                    geometry_failure,
                    int(PhotonTransportStatus.GEOMETRY_FAILURE),
                    jnp.where(
                        sampling_failure,
                        int(PhotonTransportStatus.ANGULAR_SAMPLING_EXHAUSTED),
                        status,
                    ),
                ),
            ).astype(jnp.int32)
            record = accepted_collision & (local_deposit > 0.0)
            event_positions = event_positions.at[event].set(
                jnp.where(record, trial_position, event_positions[event])
            )
            event_deposits = event_deposits.at[event].set(
                jnp.where(record, weight * local_deposit, event_deposits[event])
            )
            event_processes = event_processes.at[event].set(
                jnp.where(record, process, event_processes[event])
            )
            event_materials = event_materials.at[event].set(
                jnp.where(record, material, event_materials[event])
            )
            event_active = event_active.at[event].set(record)
            position = jnp.where(collision | leaves, trial_position, position)
            return (
                position,
                ray,
                next_energy,
                next_live,
                kerma,
                escaped,
                truncated,
                event_count + live.astype(jnp.int32),
                compton_count + compton.astype(jnp.int32),
                rayleigh_count + rayleigh.astype(jnp.int32),
                virtual_count + virtual.astype(jnp.int32),
                status,
                event_positions,
                event_deposits,
                event_processes,
                event_materials,
                event_active,
            )

        state = jax.lax.fori_loop(0, self.maximum_events, event_step, state)
        (
            position,
            direction,
            final_energy,
            live,
            kerma,
            escaped,
            truncated,
            event_count,
            compton_count,
            rayleigh_count,
            virtual_count,
            status,
            event_positions,
            event_deposits,
            event_processes,
            event_materials,
            event_active,
        ) = state
        truncated = truncated + jnp.where(live, weight * final_energy, 0.0)
        status = jnp.where(
            live,
            int(PhotonTransportStatus.EVENT_CAPACITY_EXHAUSTED),
            status,
        ).astype(jnp.int32)
        live = jnp.asarray(False)
        launched = weight * energy
        ledger = launched - jnp.sum(kerma) - escaped - truncated
        finite = (
            jnp.all(jnp.isfinite(position))
            & jnp.all(jnp.isfinite(direction))
            & jnp.isfinite(final_energy)
            & jnp.all(jnp.isfinite(kerma))
            & jnp.isfinite(ledger)
        )
        tolerance = (
            256.0 * jnp.finfo(energy.dtype).eps * jnp.maximum(jnp.abs(launched), 1.0)
        )
        successful = (
            initial_valid
            & finite
            & (jnp.abs(ledger) <= tolerance)
            & (status == int(PhotonTransportStatus.SUCCESS))
        )
        scatter_class = jnp.where(
            (compton_count + rayleigh_count) == 0,
            0,
            jnp.where(
                (compton_count == 1) & (rayleigh_count == 0),
                1,
                jnp.where((rayleigh_count == 1) & (compton_count == 0), 2, 3),
            ),
        ).astype(jnp.int32)
        return (
            kerma,
            event_positions,
            event_deposits,
            event_processes,
            event_materials,
            event_active,
            escaped,
            truncated,
            position,
            direction,
            final_energy,
            event_count,
            compton_count,
            rayleigh_count,
            virtual_count,
            scatter_class,
            status,
            successful,
            ledger,
        )

    def simulate(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        energies: ArrayLike,
        key: Array,
        /,
        *,
        history_ids: ArrayLike | None = None,
        weights: ArrayLike = 1.0,
    ) -> PhotonTransportResult:
        origin = jnp.asarray(origins, dtype=self.geometry.lower.dtype)
        direction = jnp.asarray(directions, dtype=origin.dtype)
        energy = jnp.asarray(energies, dtype=origin.dtype)
        if origin.ndim != 2 or origin.shape[-1] != 3 or direction.shape != origin.shape:
            raise ValueError("Photon origins/directions must have shape (history, 3).")
        count = origin.shape[0]
        if energy.shape != (count,):
            raise ValueError("Photon energies must have shape (history,).")
        history = (
            jnp.arange(count, dtype=jnp.uint32)
            if history_ids is None
            else jnp.asarray(history_ids, dtype=jnp.uint32)
        )
        weight = jnp.broadcast_to(jnp.asarray(weights, dtype=energy.dtype), (count,))
        if history.shape != (count,) or jnp.asarray(key).shape not in ((), (2,)):
            raise ValueError("Photon history IDs or PRNG key have invalid shape.")
        values = jax.lax.map(
            lambda inputs: self._one(key, *inputs),
            (history, origin, direction, energy, weight),
        )
        (
            kerma,
            event_positions,
            event_deposits,
            event_processes,
            event_materials,
            event_active,
            escaped,
            truncated,
            positions,
            terminal_directions,
            terminal_energy,
            event_count,
            compton_count,
            rayleigh_count,
            virtual_count,
            scatter_class,
            status,
            successful,
            ledger,
        ) = values
        return PhotonTransportResult(
            history,
            kerma,
            event_positions,
            event_deposits,
            event_processes,
            event_materials,
            event_active,
            escaped,
            truncated,
            positions,
            terminal_directions,
            terminal_energy,
            event_count,
            compton_count,
            rayleigh_count,
            virtual_count,
            scatter_class,
            status,
            successful,
            jnp.mean(kerma, axis=0),
            _standard_error(kerma),
            jnp.mean(escaped),
            jnp.max(jnp.abs(ledger)),
            jnp.all(successful),
            self.plan_id,
        )


__all__ = ["PhotonTransportPlan", "PhotonTransportResult", "PhotonTransportStatus"]
