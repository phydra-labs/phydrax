#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import assert_never, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._physical import ElectromagneticScaleContract
from .._sampling._addressing import derive_key, SampleAddress
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle._radiation_geometry import VoxelRadiationGeometryPlan
from ..equations._matter_radiation_interactions import (
    sample_bethe_heitler_pair_kinetic_energies,
)
from ..equations._radiation_interactions import (
    compton_electron_cosine,
    ComptonKinematics,
    doppler_scattered_energy,
    RadiationCrossSectionLibrary,
    RadiationInteractionKind,
    sample_compton_profile_momentum,
    sample_sauter_cosine,
)
from ..typing import parse, PRNGKey
from ..units import conversion_factor, ELECTRONVOLT
from ._secondary_stack import (
    empty_secondary_stack,
    push_secondary,
    SecondaryParticleStack,
    SecondaryStackSpec,
)


_SI_ELECTROMAGNETIC_SCALE = ElectromagneticScaleContract.si()
_ELECTRON_REST_ENERGY_EV = float(
    _SI_ELECTROMAGNETIC_SCALE.electron_mass
    * _SI_ELECTROMAGNETIC_SCALE.speed_of_light**2
    / _SI_ELECTROMAGNETIC_SCALE.elementary_charge
)

# Every random draw of one event is addressed by the persistent particle
# identity words, the event index, and a stream number, never by the slot.
_EVENT_ADDRESS = SampleAddress("radiation-transport", "photon-history", target="event")
_STREAM_PATH = 0
_STREAM_COLLISION = 1
_STREAM_PROCESS = 2
_STREAM_COMPTON_COSINE = 20
_STREAM_COMPTON_ACCEPT = 21
_STREAM_COMPTON_AZIMUTH = 22
_STREAM_COMPTON_PROFILE = 23
_STREAM_RAYLEIGH_COSINE = 30
_STREAM_RAYLEIGH_ACCEPT = 31
_STREAM_RAYLEIGH_AZIMUTH = 32
_STREAM_PHOTOELECTRON_PROPOSAL = 40
_STREAM_PHOTOELECTRON_ACCEPT = 41
_STREAM_PHOTOELECTRON_AZIMUTH = 42
_STREAM_PAIR_ENERGY = 50
_STREAM_PAIR_POLAR = 51
_STREAM_PAIR_AZIMUTH = 52


class PhotonTransportStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    CROSS_SECTION_UNSUPPORTED = 2
    GEOMETRY_FAILURE = 3
    ANGULAR_SAMPLING_EXHAUSTED = 4
    EVENT_CAPACITY_EXHAUSTED = 5
    NONFINITE = 6
    SECONDARY_CAPACITY_EXHAUSTED = 7


class PhotonTransportResult(StrictModule, NonTrainableState):
    id_hi: Array
    id_lo: Array
    material_kerma: Array
    event_positions: Array
    event_deposited_energy: Array
    event_process: Array
    event_material: Array
    event_active: Array
    escaped_energy: Array
    truncated_energy: Array
    secondary_electron_energy: Array
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
    secondary_electrons: SecondaryParticleStack | None
    mean_material_kerma: Array
    standard_error_material_kerma: Array
    mean_escaped_energy: Array
    maximum_ledger_residual: Array
    all_successful: Array
    plan_id: str = eqx.field(static=True)


class _Carry(NamedTuple):
    position: Array
    direction: Array
    energy: Array
    live: Array
    kerma: Array
    escaped: Array
    truncated: Array
    secondary_energy: Array
    event_count: Array
    compton_count: Array
    rayleigh_count: Array
    virtual_count: Array
    status: Array
    event_positions: Array
    event_deposits: Array
    event_processes: Array
    event_materials: Array
    event_active: Array
    stack: tuple[Array, Array, Array, Array, Array, Array, Array]
    stack_count: Array
    stack_overflow: Array


class _Outcome(NamedTuple):
    kerma: Array
    event_positions: Array
    event_deposits: Array
    event_processes: Array
    event_materials: Array
    event_active: Array
    escaped: Array
    truncated: Array
    secondary_energy: Array
    position: Array
    direction: Array
    energy: Array
    event_count: Array
    compton_count: Array
    rayleigh_count: Array
    virtual_count: Array
    scatter_class: Array
    status: Array
    successful: Array
    ledger: Array
    stack: tuple[Array, Array, Array, Array, Array, Array, Array]
    stack_count: Array
    stack_overflow: Array


class _Draws:
    """Identity-addressed uniform draws of one history event."""

    __slots__ = ("event", "id_hi", "id_lo", "key")

    def __init__(self, key: Array, id_hi: Array, id_lo: Array, event: Array) -> None:
        self.key = key
        self.id_hi = id_hi
        self.id_lo = id_lo
        self.event = event

    def uniform(self, stream: int, shape: tuple[int, ...] = ()) -> Array:
        return jr.uniform(
            derive_key(
                self.key, _EVENT_ADDRESS, self.id_hi, self.id_lo, self.event, stream
            ),
            shape,
        )


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
    draws: _Draws,
    energy: Array,
    attempts: int,
    electron_rest_energy: float,
) -> tuple[Array, Array, Array, Array]:
    """Klein–Nishina polar cosine, azimuth, and free-electron scattered energy."""
    cosine = 2.0 * draws.uniform(_STREAM_COMPTON_COSINE, (attempts,)) - 1.0
    accept_draw = draws.uniform(_STREAM_COMPTON_ACCEPT, (attempts,))
    alpha = energy / electron_rest_energy
    ratio = 1.0 / (1.0 + alpha * (1.0 - cosine))
    kernel = ratio**2 * (ratio + 1.0 / ratio - (1.0 - cosine**2))
    accepted = accept_draw <= 0.5 * kernel
    index = jnp.argmax(accepted)
    azimuth = 2.0 * jnp.pi * draws.uniform(_STREAM_COMPTON_AZIMUTH)
    return cosine[index], azimuth, energy * ratio[index], jnp.any(accepted)


def _sample_rayleigh(
    draws: _Draws, direction: Array, attempts: int
) -> tuple[Array, Array]:
    cosine = 2.0 * draws.uniform(_STREAM_RAYLEIGH_COSINE, (attempts,)) - 1.0
    accepted = draws.uniform(_STREAM_RAYLEIGH_ACCEPT, (attempts,)) <= 0.5 * (
        1.0 + cosine**2
    )
    index = jnp.argmax(accepted)
    azimuth = 2.0 * jnp.pi * draws.uniform(_STREAM_RAYLEIGH_AZIMUTH)
    return _rotate(direction, cosine[index], azimuth), jnp.any(accepted)


def _standard_error(values: Array, /) -> Array:
    count = values.shape[0]
    centered = values - jnp.mean(values, axis=0)
    return jnp.where(
        count > 1,
        jnp.sqrt(jnp.sum(centered**2, axis=0) / (count * (count - 1))),
        jnp.zeros(values.shape[1:], dtype=values.dtype),
    )


def _identity_words(
    identities: tuple[ArrayLike, ArrayLike] | None, count: int, /
) -> tuple[Array, Array]:
    if identities is None:
        return (
            jnp.zeros((count,), dtype=jnp.uint32),
            jnp.arange(count, dtype=jnp.uint32),
        )
    words = tuple(jnp.asarray(word) for word in identities)
    if len(words) != 2 or any(
        not jnp.issubdtype(word.dtype, jnp.integer) for word in words
    ):
        raise TypeError("identities must be two integer word arrays (hi, lo).")
    if any(word.shape != (count,) for word in words):
        raise ValueError("Identity words must align with histories.")
    return words[0].astype(jnp.uint32), words[1].astype(jnp.uint32)


class PhotonTransportPlan(StrictModule, NonTrainableState):
    """Woodcock delta-tracking photon histories on a voxel material universe.

    Every random draw is addressed by the history's persistent identity words
    through `derive_key`, so a history's outcome is independent of batch
    composition and slot order. Without an `electron_stack`, photoelectric,
    Compton, and pair-production transfers are deposited locally as KERMA.
    With one, Sauter photoelectrons, Klein--Nishina recoil electrons, and both
    Bethe--Heitler pair daughters at or above the stack threshold are recorded
    as typed secondaries; only sub-threshold transfers, pair rest-mass energy,
    and below-cutoff photons deposit locally. `compton_kinematics` selects the
    free-electron Compton line or its impulse-approximation Doppler broadening,
    which requires `compton_profile_j0` in the cross-section library.
    """

    geometry: VoxelRadiationGeometryPlan
    cross_sections: RadiationCrossSectionLibrary
    maximum_events: int = eqx.field(static=True)
    angular_sampling_attempts: int = eqx.field(static=True)
    cutoff_energy: float = eqx.field(static=True)
    electron_rest_energy: float = eqx.field(static=True)
    compton_kinematics: ComptonKinematics = eqx.field(static=True)
    electron_stack: SecondaryStackSpec | None = eqx.field(static=True)
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
        compton_kinematics: ComptonKinematics = "free-electron",
        electron_stack: SecondaryStackSpec | None = None,
    ) -> None:
        if not isinstance(geometry, VoxelRadiationGeometryPlan):
            raise TypeError("geometry must be VoxelRadiationGeometryPlan.")
        if not isinstance(cross_sections, RadiationCrossSectionLibrary):
            raise TypeError("cross_sections must be RadiationCrossSectionLibrary.")
        if electron_stack is not None and not isinstance(
            electron_stack, SecondaryStackSpec
        ):
            raise TypeError("electron_stack must be SecondaryStackSpec or None.")
        kinematics = parse(compton_kinematics, ComptonKinematics, "compton_kinematics")
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
        if (
            kinematics == "impulse-approximation"
            and cross_sections.compton_profile_j0 is None
        ):
            raise ValueError(
                "impulse-approximation Compton kinematics require compton_profile_j0 "
                "in the cross-section library."
            )
        if electron_stack is not None and electron_stack.capacity > 2 * events:
            raise ValueError(
                "electron_stack capacity cannot exceed twice maximum_events."
            )
        self.geometry = geometry
        self.cross_sections = cross_sections
        self.maximum_events = events
        self.angular_sampling_attempts = attempts
        self.cutoff_energy = cutoff
        self.electron_rest_energy = electron_rest
        self.compton_kinematics = kinematics
        self.electron_stack = electron_stack
        self.plan_id = canonical_fingerprint(
            {
                "kind": "voxel-delta-photon-transport",
                "geometry": geometry.geometry_id,
                "cross_sections": cross_sections.library_id,
                "maximum_events": events,
                "angular_sampling_attempts": attempts,
                "cutoff_energy": cutoff,
                "electron_rest_energy": electron_rest,
                "compton_kinematics": kinematics,
                "deposition_semantics": (
                    "photon-kerma-local-recoil"
                    if electron_stack is None
                    else electron_stack.spec_id
                ),
                "random_addressing": _EVENT_ADDRESS.token,
                "pathwise_differentiability": False,
            }
        )

    @property
    def _stack_capacity(self) -> int:
        return 1 if self.electron_stack is None else self.electron_stack.capacity

    def _scattered_energy(
        self,
        draws: _Draws,
        energy: Array,
        cosine: Array,
        free_energy: Array,
        material: Array,
    ) -> tuple[Array, Array]:
        """Scattered photon energy at the sampled angle and its validity."""
        match self.compton_kinematics:
            case "free-electron":
                return free_energy, jnp.asarray(True)
            case "impulse-approximation":
                profile = self.cross_sections.compton_profile_j0
                if profile is None:
                    raise RuntimeError(
                        "impulse-approximation kinematics without a Compton profile."
                    )
                safe_material = jnp.clip(material, 0, profile.shape[0] - 1)
                momentum = sample_compton_profile_momentum(
                    draws.uniform(
                        _STREAM_COMPTON_PROFILE, (self.angular_sampling_attempts,)
                    ),
                    profile[safe_material],
                )
                scattered, valid = doppler_scattered_energy(
                    energy, cosine, momentum, self.electron_rest_energy
                )
                index = jnp.argmax(valid)
                return scattered[index], jnp.any(valid)
            case _:
                assert_never(self.compton_kinematics)

    def _one(
        self,
        key: Array,
        id_hi: Array,
        id_lo: Array,
        origin: Array,
        direction: Array,
        energy: Array,
        weight: Array,
    ) -> _Outcome:
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
        if self.electron_stack is not None:
            # Recorded secondaries carry physical energies, so stacked histories
            # must be unit weight for the ledger to describe real particles.
            initial_valid = initial_valid & (weight == 1.0)
        direction = direction / jnp.where(norm > 0.0, norm, 1.0)
        dtype = energy.dtype
        zero = jnp.asarray(0.0, dtype)
        state = _Carry(
            origin,
            direction,
            energy,
            initial_valid,
            jnp.zeros((self.cross_sections.material_count,), dtype=dtype),
            zero,
            zero,
            zero,
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.where(
                initial_valid,
                int(PhotonTransportStatus.SUCCESS),
                int(PhotonTransportStatus.INVALID_INPUT),
            ).astype(jnp.int32),
            jnp.zeros((self.maximum_events, 3), dtype=dtype),
            jnp.zeros((self.maximum_events,), dtype=dtype),
            -jnp.ones((self.maximum_events,), dtype=jnp.int32),
            -jnp.ones((self.maximum_events,), dtype=jnp.int32),
            jnp.zeros((self.maximum_events,), dtype=jnp.bool_),
            empty_secondary_stack(self._stack_capacity, dtype),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )

        def event_step(event: Array, carry: _Carry) -> _Carry:
            """Advance one photon event and its secondary-stack transaction."""
            draws = _Draws(key, id_hi, id_lo, event)
            photon_energy = carry.energy
            majorant = self.cross_sections.majorant(photon_energy)
            cross_supported = (
                jnp.isfinite(majorant)
                & (majorant > 0.0)
                & (photon_energy >= self.cross_sections.energy[0])
                & (photon_energy <= self.cross_sections.energy[-1])
            )
            path_draw = jnp.maximum(
                draws.uniform(_STREAM_PATH), jnp.finfo(photon_energy.dtype).tiny
            )
            distance = -jnp.log(path_draw) / jnp.where(majorant > 0.0, majorant, 1.0)
            exit_distance = self.geometry.distance_to_exit(
                carry.position, carry.direction
            )
            leaves = carry.live & cross_supported & (distance >= exit_distance)
            collision = carry.live & cross_supported & ~leaves
            trial_position = (
                carry.position
                + jnp.where(leaves, exit_distance, distance) * carry.direction
            )
            escaped = carry.escaped + jnp.where(leaves, weight * photon_energy, 0.0)
            location = self.geometry.locate(trial_position)
            material = location.material_index
            evaluated = self.cross_sections.evaluate(material, photon_energy)
            accepted_collision = (
                collision
                & location.inside
                & evaluated.successful
                & (
                    draws.uniform(_STREAM_COLLISION)
                    <= evaluated.total / jnp.where(majorant > 0.0, majorant, 1.0)
                )
            )
            virtual = (
                collision & location.inside & evaluated.successful & ~accepted_collision
            )
            cumulative = jnp.cumsum(evaluated.coefficients)
            process_draw = draws.uniform(_STREAM_PROCESS) * evaluated.total
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
            pair = accepted_collision & (
                (process == int(RadiationInteractionKind.PAIR_NUCLEAR))
                | (process == int(RadiationInteractionKind.PAIR_ELECTRON))
            )
            compton_cosine, compton_azimuth, free_energy, compton_success = (
                _sample_compton(
                    draws,
                    photon_energy,
                    self.angular_sampling_attempts,
                    self.electron_rest_energy,
                )
            )
            compton_energy, kinematics_success = self._scattered_energy(
                draws, photon_energy, compton_cosine, free_energy, material
            )
            compton_direction = _rotate(carry.direction, compton_cosine, compton_azimuth)
            rayleigh_direction, rayleigh_success = _sample_rayleigh(
                draws, carry.direction, self.angular_sampling_attempts
            )
            pair_electron_energy, pair_positron_energy, pair_success = (
                sample_bethe_heitler_pair_kinetic_energies(
                    draws.uniform(_STREAM_PAIR_ENERGY),
                    photon_energy,
                    self.electron_rest_energy,
                )
            )
            pair_gamma = photon_energy / self.electron_rest_energy
            pair_polar_draw = jnp.minimum(
                draws.uniform(_STREAM_PAIR_POLAR),
                1.0 - jnp.finfo(photon_energy.dtype).eps,
            )
            pair_angle = jnp.minimum(
                jnp.sqrt(pair_polar_draw / (1.0 - pair_polar_draw))
                / jnp.maximum(pair_gamma, 1.0),
                jnp.pi,
            )
            pair_azimuth = 2.0 * jnp.pi * draws.uniform(_STREAM_PAIR_AZIMUTH)
            pair_electron_direction = _rotate(
                carry.direction, jnp.cos(pair_angle), pair_azimuth
            )
            pair_positron_direction = _rotate(
                carry.direction, jnp.cos(pair_angle), pair_azimuth + jnp.pi
            )
            sampling_success = (
                (~compton | (compton_success & kinematics_success))
                & (~rayleigh | rayleigh_success)
                & (~pair | pair_success)
            )
            recoil = jnp.where(compton, photon_energy - compton_energy, 0.0)
            transfer = jnp.where(
                pair, photon_energy, jnp.where(photoelectric, photon_energy, recoil)
            )
            next_energy = jnp.where(compton, compton_energy, photon_energy)
            below_cutoff = compton & (next_energy < self.cutoff_energy)
            next_energy = jnp.where(photoelectric | pair | below_cutoff, 0.0, next_energy)
            if self.electron_stack is None:
                stack = carry.stack
                stack_count = carry.stack_count
                overflow = jnp.asarray(False)
                overflow_energy = zero
                local_deposit = transfer
                recorded = zero
                lost_transfer = zero
                event_success = sampling_success
            else:
                ordinary = photoelectric | compton
                ordinary_push = ordinary & (
                    transfer >= self.electron_stack.minimum_energy
                )
                pair_electron_push = pair & (
                    pair_electron_energy >= self.electron_stack.minimum_energy
                )
                pair_positron_push = pair & (
                    pair_positron_energy >= self.electron_stack.minimum_energy
                )
                photoelectron_cosine, sauter_success = sample_sauter_cosine(
                    draws.uniform(
                        _STREAM_PHOTOELECTRON_PROPOSAL,
                        (self.angular_sampling_attempts,),
                    ),
                    draws.uniform(
                        _STREAM_PHOTOELECTRON_ACCEPT, (self.angular_sampling_attempts,)
                    ),
                    photon_energy,
                    self.electron_rest_energy,
                )
                photoelectron_direction = _rotate(
                    carry.direction,
                    photoelectron_cosine,
                    2.0 * jnp.pi * draws.uniform(_STREAM_PHOTOELECTRON_AZIMUTH),
                )
                recoil_direction = _rotate(
                    carry.direction,
                    compton_electron_cosine(
                        photon_energy, compton_energy, compton_cosine
                    ),
                    compton_azimuth + jnp.pi,
                )
                electron_direction = jnp.where(
                    photoelectric, photoelectron_direction, recoil_direction
                )
                event_success = sampling_success & (~photoelectric | sauter_success)
                stack, stack_count, ordinary_overflow = push_secondary(
                    carry.stack,
                    carry.stack_count,
                    ordinary_push & event_success,
                    trial_position,
                    electron_direction,
                    transfer,
                    material,
                    (2 * event).astype(jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                )
                stack, stack_count, electron_overflow = push_secondary(
                    stack,
                    stack_count,
                    pair_electron_push & event_success,
                    trial_position,
                    pair_electron_direction,
                    pair_electron_energy,
                    material,
                    (2 * event).astype(jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                )
                stack, stack_count, positron_overflow = push_secondary(
                    stack,
                    stack_count,
                    pair_positron_push & event_success,
                    trial_position,
                    pair_positron_direction,
                    pair_positron_energy,
                    material,
                    (2 * event + 1).astype(jnp.int32),
                    jnp.asarray(1, dtype=jnp.int32),
                )
                overflow = ordinary_overflow | electron_overflow | positron_overflow
                overflow_energy = (
                    jnp.where(ordinary_overflow, transfer, 0.0)
                    + jnp.where(electron_overflow, pair_electron_energy, 0.0)
                    + jnp.where(positron_overflow, pair_positron_energy, 0.0)
                )
                recorded = (
                    jnp.where(
                        ordinary_push & event_success & ~ordinary_overflow,
                        transfer,
                        0.0,
                    )
                    + jnp.where(
                        pair_electron_push & event_success & ~electron_overflow,
                        pair_electron_energy,
                        0.0,
                    )
                    + jnp.where(
                        pair_positron_push & event_success & ~positron_overflow,
                        pair_positron_energy,
                        0.0,
                    )
                )
                lost_transfer = jnp.where(
                    (ordinary | pair) & ~event_success, transfer, 0.0
                )
                pair_local = jnp.where(pair, 2.0 * self.electron_rest_energy, 0.0)
                pair_local = pair_local + jnp.where(
                    pair & ~pair_electron_push, pair_electron_energy, 0.0
                )
                pair_local = pair_local + jnp.where(
                    pair & ~pair_positron_push, pair_positron_energy, 0.0
                )
                local_deposit = jnp.where(ordinary & ~ordinary_push, transfer, pair_local)
            local_deposit = local_deposit + jnp.where(below_cutoff, next_energy, 0.0)
            next_energy = jnp.where(below_cutoff, 0.0, next_energy)
            kerma = carry.kerma.at[
                jnp.clip(material, 0, self.cross_sections.material_count - 1)
            ].add(jnp.where(accepted_collision, weight * local_deposit, 0.0))
            ray = jnp.where(
                compton,
                compton_direction,
                jnp.where(rayleigh, rayleigh_direction, carry.direction),
            )
            geometry_failure = collision & (~location.inside | ~evaluated.supported)
            sampling_failure = accepted_collision & ~event_success
            unsupported_failure = carry.live & ~cross_supported
            failed_energy = jnp.where(
                geometry_failure | sampling_failure | unsupported_failure,
                weight * next_energy,
                0.0,
            )
            truncated = (
                carry.truncated
                + failed_energy
                + weight * (overflow_energy + lost_transfer)
            )
            next_live = (
                carry.live
                & cross_supported
                & ~leaves
                & ~photoelectric
                & ~pair
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
                        jnp.where(
                            overflow,
                            int(PhotonTransportStatus.SECONDARY_CAPACITY_EXHAUSTED),
                            carry.status,
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            record = accepted_collision & (local_deposit > 0.0)
            event_positions = carry.event_positions.at[event].set(
                jnp.where(record, trial_position, carry.event_positions[event])
            )
            event_deposits = carry.event_deposits.at[event].set(
                jnp.where(record, weight * local_deposit, carry.event_deposits[event])
            )
            event_processes = carry.event_processes.at[event].set(
                jnp.where(record, process, carry.event_processes[event])
            )
            event_materials = carry.event_materials.at[event].set(
                jnp.where(record, material, carry.event_materials[event])
            )
            event_active = carry.event_active.at[event].set(record)
            position = jnp.where(collision | leaves, trial_position, carry.position)
            return _Carry(
                position,
                ray,
                next_energy,
                next_live,
                kerma,
                escaped,
                truncated,
                carry.secondary_energy + weight * recorded,
                carry.event_count + carry.live.astype(jnp.int32),
                carry.compton_count + compton.astype(jnp.int32),
                carry.rayleigh_count + rayleigh.astype(jnp.int32),
                carry.virtual_count + virtual.astype(jnp.int32),
                status,
                event_positions,
                event_deposits,
                event_processes,
                event_materials,
                event_active,
                stack,
                stack_count,
                carry.stack_overflow + overflow.astype(jnp.int32),
            )

        final = jax.lax.fori_loop(0, self.maximum_events, event_step, state)
        truncated = final.truncated + jnp.where(final.live, weight * final.energy, 0.0)
        # A history that already failed keeps its first failure; event capacity
        # is reported only for otherwise healthy histories still in flight.
        status = jnp.where(
            final.live & (final.status == int(PhotonTransportStatus.SUCCESS)),
            int(PhotonTransportStatus.EVENT_CAPACITY_EXHAUSTED),
            final.status,
        ).astype(jnp.int32)
        launched = weight * energy
        ledger = (
            launched
            - jnp.sum(final.kerma)
            - final.escaped
            - truncated
            - final.secondary_energy
        )
        finite = (
            jnp.all(jnp.isfinite(final.position))
            & jnp.all(jnp.isfinite(final.direction))
            & jnp.isfinite(final.energy)
            & jnp.all(jnp.isfinite(final.kerma))
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
            (final.compton_count + final.rayleigh_count) == 0,
            0,
            jnp.where(
                (final.compton_count == 1) & (final.rayleigh_count == 0),
                1,
                jnp.where((final.rayleigh_count == 1) & (final.compton_count == 0), 2, 3),
            ),
        ).astype(jnp.int32)
        return _Outcome(
            final.kerma,
            final.event_positions,
            final.event_deposits,
            final.event_processes,
            final.event_materials,
            final.event_active,
            final.escaped,
            truncated,
            final.secondary_energy,
            final.position,
            final.direction,
            final.energy,
            final.event_count,
            final.compton_count,
            final.rayleigh_count,
            final.virtual_count,
            scatter_class,
            status,
            successful,
            ledger,
            final.stack,
            final.stack_count,
            final.stack_overflow,
        )

    def simulate(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        energies: ArrayLike,
        key: PRNGKey,
        /,
        *,
        identities: tuple[ArrayLike, ArrayLike] | None = None,
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
        id_hi, id_lo = _identity_words(identities, count)
        weight = jnp.broadcast_to(jnp.asarray(weights, dtype=energy.dtype), (count,))
        root = parse(key, PRNGKey, "key")
        outcome = jax.lax.map(
            lambda inputs: self._one(root, *inputs),
            (id_hi, id_lo, origin, direction, energy, weight),
        )
        stack = None
        if self.electron_stack is not None:
            (
                positions,
                stack_directions,
                energies_,
                materials,
                creations,
                kinds,
                active,
            ) = outcome.stack
            stack = SecondaryParticleStack(
                positions,
                stack_directions,
                energies_,
                materials,
                creations,
                kinds,
                active,
                outcome.stack_count,
                outcome.secondary_energy,
                outcome.stack_overflow,
                self.electron_stack.spec_id,
            )
        return PhotonTransportResult(
            id_hi,
            id_lo,
            outcome.kerma,
            outcome.event_positions,
            outcome.event_deposits,
            outcome.event_processes,
            outcome.event_materials,
            outcome.event_active,
            outcome.escaped,
            outcome.truncated,
            outcome.secondary_energy,
            outcome.position,
            outcome.direction,
            outcome.energy,
            outcome.event_count,
            outcome.compton_count,
            outcome.rayleigh_count,
            outcome.virtual_count,
            outcome.scatter_class,
            outcome.status,
            outcome.successful,
            stack,
            jnp.mean(outcome.kerma, axis=0),
            _standard_error(outcome.kerma),
            jnp.mean(outcome.escaped),
            jnp.max(jnp.abs(outcome.ledger)),
            jnp.all(outcome.successful),
            self.plan_id,
        )


__all__ = ["PhotonTransportPlan", "PhotonTransportResult", "PhotonTransportStatus"]
