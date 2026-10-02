#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Literal, NamedTuple

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
from ..equations._charged_radiation_interactions import (
    ChargedRadiationMaterialLibrary,
    ChargedRadiationParticleKind,
    sample_bremsstrahlung_photon,
)
from ..equations._matter_radiation_interactions import (
    bremsstrahlung_suppression_factor,
    BremsstrahlungSpectrumRoute,
    SeltzerBergerBremsstrahlungTable,
)
from ..typing import Bool, checked, Dim, Float64, Int32, parse, PRNGKey
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
_POSITRON_ANNIHILATION_ENERGY_EV = 2.0 * _ELECTRON_REST_ENERGY_EV

# Every random draw of one step is addressed by the persistent particle identity
# words, the step index, and a stream number, never by the storage slot.
_STEP_ADDRESS = SampleAddress(
    "radiation-transport", "charged-condensed-history", target="step"
)
_STREAM_BREMSSTRAHLUNG_EVENT = 0
_STREAM_BREMSSTRAHLUNG_ENERGY = 1
_STREAM_SCATTER_POLAR = 2
_STREAM_SCATTER_AZIMUTH = 3
_STREAM_PHOTON_POLAR = 4
_STREAM_PHOTON_AZIMUTH = 5
_STREAM_BREMSSTRAHLUNG_SUPPRESSION = 6


class _HistoryDim(Dim, minimum=1):
    """Transported charged histories."""


class _StepSlotDim(Dim, minimum=1):
    """Recorded condensed-history step slots per history."""


class ChargedParticleTransportStatus(IntEnum):
    SUCCESS = 0
    INVALID_INPUT = 1
    MATERIAL_UNSUPPORTED = 2
    STEP_CAPACITY_EXHAUSTED = 3
    NONFINITE = 4
    SECONDARY_CAPACITY_EXHAUSTED = 5


class ChargedStepBank(StrictModule):
    """Per-(history, step) condensed-history record; inactive slots are padding.

    `start_beta`/`end_beta` are the particle speeds in units of `c` before and
    after the step, `deposited_energy` is the energy left in the material by the
    step (continuous loss, sub-threshold bremsstrahlung, and the residual of a
    below-cutoff particle), and `material_index` is the voxel material of the
    step start. Steps beyond the bank capacity are transported but not
    recorded; `unrecorded_count` reports them so a consumer can refuse an
    incomplete history.
    """

    __strict_contract__ = True

    start_positions: Float64[_HistoryDim, _StepSlotDim, Literal[3]]
    end_positions: Float64[_HistoryDim, _StepSlotDim, Literal[3]]
    start_beta: Float64[_HistoryDim, _StepSlotDim]
    end_beta: Float64[_HistoryDim, _StepSlotDim]
    deposited_energy: Float64[_HistoryDim, _StepSlotDim]
    material_index: Int32[_HistoryDim, _StepSlotDim]
    active: Bool[_HistoryDim, _StepSlotDim]
    recorded_count: Int32[_HistoryDim]
    unrecorded_count: Int32[_HistoryDim]

    @property
    def capacity(self) -> int:
        return self.active.shape[1]

    @property
    def complete(self) -> Array:
        return self.unrecorded_count == 0


class ChargedParticleTransportResult(StrictModule, NonTrainableState):
    id_hi: Array
    id_lo: Array
    deposited_energy: Array
    escaped_energy: Array
    bremsstrahlung_energy: Array
    annihilation_photon_energy: Array
    truncated_energy: Array
    terminal_position: Array
    terminal_direction: Array
    terminal_kinetic_energy: Array
    path_length: Array
    step_count: Array
    status: Array
    successful: Array
    step_bank: ChargedStepBank | None
    secondary_photons: SecondaryParticleStack | None
    maximum_kinetic_ledger_residual: Array
    all_successful: Array
    plan_id: str = eqx.field(static=True)


class _Carry(NamedTuple):
    position: Array
    direction: Array
    kinetic: Array
    live: Array
    deposited: Array
    escaped: Array
    bremsstrahlung: Array
    annihilation: Array
    truncated: Array
    path_length: Array
    step_count: Array
    status: Array
    bank: tuple[Array, Array, Array, Array, Array, Array, Array]
    unrecorded: Array
    stack: tuple[Array, Array, Array, Array, Array, Array, Array]
    stack_count: Array
    stack_overflow: Array


class _Outcome(NamedTuple):
    deposited: Array
    escaped: Array
    bremsstrahlung: Array
    annihilation: Array
    truncated: Array
    position: Array
    direction: Array
    kinetic: Array
    path_length: Array
    step_count: Array
    status: Array
    successful: Array
    ledger: Array
    bank: tuple[Array, Array, Array, Array, Array, Array, Array]
    unrecorded: Array
    stack: tuple[Array, Array, Array, Array, Array, Array, Array]
    stack_count: Array
    stack_energy: Array
    stack_overflow: Array


def _scatter_direction(direction: Array, theta: Array, phi: Array) -> Array:
    z = jnp.asarray((0.0, 0.0, 1.0), dtype=direction.dtype)
    x = jnp.asarray((1.0, 0.0, 0.0), dtype=direction.dtype)
    reference = jnp.where(jnp.abs(direction[2]) < 0.9, z, x)
    first = jnp.cross(reference, direction)
    first = first / jnp.linalg.norm(first)
    second = jnp.cross(direction, first)
    result = jnp.cos(theta) * direction + jnp.sin(theta) * (
        jnp.cos(phi) * first + jnp.sin(phi) * second
    )
    return result / jnp.linalg.norm(result)


def _beta(kinetic_energy: Array) -> Array:
    """Speed in units of `c` from kinetic energy in eV."""
    total = kinetic_energy + _ELECTRON_REST_ENERGY_EV
    return (
        jnp.sqrt(
            jnp.maximum(
                kinetic_energy * (kinetic_energy + 2.0 * _ELECTRON_REST_ENERGY_EV), 0.0
            )
        )
        / total
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


class ChargedParticleTransportPlan(StrictModule, NonTrainableState):
    """Electron/positron condensed history on a voxel material universe.

    Every random draw is addressed by the history's persistent identity words
    through `derive_key`, so a history's outcome is independent of batch
    composition and slot order. An optional `step_bank_capacity` records every
    step's pre/post position, speed, deposit, and material. An optional
    `photon_stack` turns bremsstrahlung emissions at or above its threshold into
    recorded secondary photons instead of a bare energy tally; emissions below
    the threshold are deposited locally.
    """

    geometry: VoxelRadiationGeometryPlan
    materials: ChargedRadiationMaterialLibrary
    maximum_steps: int = eqx.field(static=True)
    maximum_step_length: float = eqx.field(static=True)
    maximum_fractional_energy_loss: float = eqx.field(static=True)
    cutoff_energy_ev: float = eqx.field(static=True)
    step_bank_capacity: int | None = eqx.field(static=True)
    photon_stack: SecondaryStackSpec | None = eqx.field(static=True)
    bremsstrahlung_spectrum: BremsstrahlungSpectrumRoute = eqx.field(static=True)
    seltzer_berger_table: SeltzerBergerBremsstrahlungTable | None
    lpm_energy_ev: float | None = eqx.field(static=True)
    plasma_energy_ev: float | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        geometry: VoxelRadiationGeometryPlan,
        materials: ChargedRadiationMaterialLibrary,
        /,
        *,
        maximum_steps: int,
        maximum_step_length: float,
        maximum_fractional_energy_loss: float = 0.05,
        cutoff_energy_ev: float,
        step_bank_capacity: int | None = None,
        photon_stack: SecondaryStackSpec | None = None,
        bremsstrahlung_spectrum: BremsstrahlungSpectrumRoute = "bounded",
        seltzer_berger_table: SeltzerBergerBremsstrahlungTable | None = None,
        lpm_energy_ev: float | None = None,
        plasma_energy_ev: float | None = None,
    ) -> None:
        if photon_stack is not None and not isinstance(photon_stack, SecondaryStackSpec):
            raise TypeError("photon_stack must be SecondaryStackSpec or None.")
        steps = int(maximum_steps)
        length = float(maximum_step_length)
        fraction = float(maximum_fractional_energy_loss)
        cutoff = float(cutoff_energy_ev)
        bank = None if step_bank_capacity is None else int(step_bank_capacity)
        spectrum = parse(
            bremsstrahlung_spectrum,
            BremsstrahlungSpectrumRoute,
            "bremsstrahlung_spectrum",
        )
        if seltzer_berger_table is not None and not isinstance(
            seltzer_berger_table, SeltzerBergerBremsstrahlungTable
        ):
            raise TypeError(
                "seltzer_berger_table must be SeltzerBergerBremsstrahlungTable or None."
            )
        if spectrum == "seltzer-berger" and seltzer_berger_table is None:
            raise ValueError(
                "seltzer-berger bremsstrahlung requires a governed differential table."
            )
        if (
            seltzer_berger_table is not None
            and seltzer_berger_table.material_ids != materials.material_ids
        ):
            raise ValueError(
                "Seltzer--Berger and charged-material tables require one material basis."
            )
        lpm = None if lpm_energy_ev is None else float(lpm_energy_ev)
        plasma = None if plasma_energy_ev is None else float(plasma_energy_ev)
        if (lpm is not None and (not isfinite(lpm) or lpm <= 0.0)) or (
            plasma is not None and (not isfinite(plasma) or plasma <= 0.0)
        ):
            raise ValueError("Suppression energies must be finite and positive.")
        if (
            geometry.material_count != materials.material_count
            or steps < 1
            or not isfinite(length)
            or length <= 0.0
            or not isfinite(fraction)
            or not 0.0 < fraction <= 0.25
            or not isfinite(cutoff)
            or cutoff <= 0.0
            or cutoff < float(materials.energy_ev[0])
            or cutoff > float(materials.energy_ev[-1])
        ):
            raise ValueError(
                "Charged transport materials, steps, or cutoffs are invalid."
            )
        if bank is not None and not 1 <= bank <= steps:
            raise ValueError("step_bank_capacity must lie in [1, maximum_steps].")
        if photon_stack is not None and photon_stack.capacity > steps:
            raise ValueError("photon_stack capacity cannot exceed maximum_steps.")
        self.geometry = geometry
        self.materials = materials
        self.maximum_steps = steps
        self.maximum_step_length = length
        self.maximum_fractional_energy_loss = fraction
        self.cutoff_energy_ev = cutoff
        self.step_bank_capacity = bank
        self.photon_stack = photon_stack
        self.bremsstrahlung_spectrum = spectrum
        self.seltzer_berger_table = seltzer_berger_table
        self.lpm_energy_ev = lpm
        self.plasma_energy_ev = plasma
        self.plan_id = canonical_fingerprint(
            {
                "kind": "charged-particle-condensed-history",
                "geometry": geometry.geometry_id,
                "materials": materials.library_id,
                "maximum_steps": steps,
                "maximum_step_length": length,
                "maximum_fractional_energy_loss": fraction,
                "cutoff_energy_ev": cutoff,
                "step_bank_capacity": bank,
                "secondary_photons": (
                    "tallied-not-transported"
                    if photon_stack is None
                    else photon_stack.spec_id
                ),
                "bremsstrahlung_spectrum": spectrum,
                "seltzer_berger_table": (
                    None
                    if seltzer_berger_table is None
                    else seltzer_berger_table.table_id
                ),
                "lpm_energy_ev": lpm,
                "plasma_energy_ev": plasma,
                "random_addressing": _STEP_ADDRESS.token,
                "pathwise_differentiability": False,
            }
        )

    @property
    def _bank_capacity(self) -> int:
        return 1 if self.step_bank_capacity is None else self.step_bank_capacity

    @property
    def _stack_capacity(self) -> int:
        return 1 if self.photon_stack is None else self.photon_stack.capacity

    def _one(
        self,
        key: Array,
        id_hi: Array,
        id_lo: Array,
        origin: Array,
        direction: Array,
        energy: Array,
        particle_kind: Array,
    ) -> _Outcome:
        norm = jnp.linalg.norm(direction)
        kind_valid = (particle_kind == int(ChargedRadiationParticleKind.ELECTRON)) | (
            particle_kind == int(ChargedRadiationParticleKind.POSITRON)
        )
        initial_valid = (
            jnp.all(jnp.isfinite(origin))
            & jnp.all(jnp.isfinite(direction))
            & jnp.isfinite(energy)
            & (energy >= self.cutoff_energy_ev)
            & (norm > 0.0)
            & kind_valid
            & self.geometry.locate(origin).inside
        )
        direction = direction / jnp.where(norm > 0.0, norm, 1.0)
        dtype = energy.dtype
        bank_capacity = self._bank_capacity
        zero = jnp.asarray(0.0, dtype)
        state = _Carry(
            origin,
            direction,
            energy,
            initial_valid,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            jnp.asarray(0, jnp.int32),
            jnp.where(
                initial_valid,
                int(ChargedParticleTransportStatus.SUCCESS),
                int(ChargedParticleTransportStatus.INVALID_INPUT),
            ).astype(jnp.int32),
            (
                jnp.zeros((bank_capacity, 3), dtype=dtype),
                jnp.zeros((bank_capacity, 3), dtype=dtype),
                jnp.zeros((bank_capacity,), dtype=dtype),
                jnp.zeros((bank_capacity,), dtype=dtype),
                jnp.zeros((bank_capacity,), dtype=dtype),
                -jnp.ones((bank_capacity,), dtype=jnp.int32),
                jnp.zeros((bank_capacity,), dtype=jnp.bool_),
            ),
            jnp.asarray(0, jnp.int32),
            empty_secondary_stack(self._stack_capacity, dtype),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )

        def draw(step_index: Array, stream: int) -> Array:
            return jr.uniform(
                derive_key(key, _STEP_ADDRESS, id_hi, id_lo, step_index, stream)
            )

        def advance(step_index: Array, carry: _Carry) -> _Carry:
            location = self.geometry.locate(carry.position)
            material = self.materials.evaluate(location.material_index, carry.kinetic)
            supported = carry.live & location.inside & material.successful
            boundary = self.geometry.distance_to_voxel_boundary(
                carry.position, carry.direction
            )
            range_step = (
                self.maximum_fractional_energy_loss
                * carry.kinetic
                / jnp.where(material.stopping_power > 0.0, material.stopping_power, 1.0)
            )
            unconstrained = jnp.minimum(self.maximum_step_length, range_step)
            step_length = jnp.minimum(unconstrained, boundary)
            boundary_limited = supported & (boundary <= unconstrained)
            continuous_loss = jnp.minimum(
                material.stopping_power * step_length, carry.kinetic
            )
            remaining = carry.kinetic - jnp.where(supported, continuous_loss, 0.0)
            brems_probability = 1.0 - jnp.exp(-material.bremsstrahlung_rate * step_length)
            photon_energy, photon_cosine, spectral_supported = (
                sample_bremsstrahlung_photon(
                    draw(step_index, _STREAM_BREMSSTRAHLUNG_ENERGY),
                    draw(step_index, _STREAM_PHOTON_POLAR),
                    remaining,
                    _ELECTRON_REST_ENERGY_EV,
                    route=self.bremsstrahlung_spectrum,
                    material_index=location.material_index,
                    minimum_photon_energy_ev=(
                        self.cutoff_energy_ev
                        if self.photon_stack is None
                        else self.photon_stack.minimum_energy
                    ),
                    table=self.seltzer_berger_table,
                )
            )
            suppression = bremsstrahlung_suppression_factor(
                remaining,
                photon_energy,
                lpm_energy_ev=self.lpm_energy_ev,
                plasma_energy_ev=self.plasma_energy_ev,
            )
            brems_event = (
                supported
                & spectral_supported
                & (draw(step_index, _STREAM_BREMSSTRAHLUNG_EVENT) < brems_probability)
                & (draw(step_index, _STREAM_BREMSSTRAHLUNG_SUPPRESSION) < suppression)
            )
            emitted = jnp.where(brems_event, photon_energy, 0.0)
            remaining = remaining - emitted
            theta = jnp.minimum(
                jnp.sqrt(jnp.maximum(material.scattering_power * step_length, 0.0))
                * jr.normal(
                    derive_key(
                        key,
                        _STEP_ADDRESS,
                        id_hi,
                        id_lo,
                        step_index,
                        _STREAM_SCATTER_POLAR,
                    )
                ),
                jnp.pi,
            )
            phi = 2.0 * jnp.pi * draw(step_index, _STREAM_SCATTER_AZIMUTH)
            scattered = _scatter_direction(carry.direction, theta, phi)
            trial = carry.position + step_length * carry.direction
            nudge = jnp.where(boundary_limited, 1.0e-10, 0.0)
            trial = trial + nudge * carry.direction
            next_location = self.geometry.locate(trial)
            leaves = boundary_limited & ~next_location.inside
            below_cutoff = supported & ~leaves & (remaining <= self.cutoff_energy_ev)
            is_positron = particle_kind == int(ChargedRadiationParticleKind.POSITRON)
            residual = jnp.where(below_cutoff, remaining, 0.0)
            if self.photon_stack is None:
                stack = carry.stack
                stack_count = carry.stack_count
                overflow = jnp.asarray(False)
                recorded_photon = emitted
                subthreshold = zero
            else:
                push = brems_event & (emitted >= self.photon_stack.minimum_energy)
                subthreshold = jnp.where(brems_event & ~push, emitted, 0.0)
                photon_direction = _scatter_direction(
                    carry.direction,
                    jnp.arccos(jnp.clip(photon_cosine, -1.0, 1.0)),
                    2.0 * jnp.pi * draw(step_index, _STREAM_PHOTON_AZIMUTH),
                )
                stack, stack_count, overflow = push_secondary(
                    carry.stack,
                    carry.stack_count,
                    push,
                    carry.position,
                    photon_direction,
                    emitted,
                    location.material_index,
                    step_index.astype(jnp.int32),
                    jnp.asarray(0, dtype=jnp.int32),
                )
                recorded_photon = jnp.where(push & ~overflow, emitted, 0.0)
            step_deposit = (
                jnp.where(supported, continuous_loss, 0.0) + residual + subthreshold
            )
            deposited = carry.deposited + step_deposit
            escaped = carry.escaped + jnp.where(leaves, remaining, 0.0)
            annihilation = carry.annihilation + jnp.where(
                below_cutoff & is_positron,
                _POSITRON_ANNIHILATION_ENERGY_EV,
                0.0,
            )
            unsupported = carry.live & ~supported
            truncated = (
                carry.truncated
                + jnp.where(unsupported, carry.kinetic, 0.0)
                + jnp.where(overflow, emitted, 0.0)
            )
            next_live = supported & ~leaves & ~below_cutoff
            status = jnp.where(
                unsupported,
                int(ChargedParticleTransportStatus.MATERIAL_UNSUPPORTED),
                jnp.where(
                    overflow,
                    int(ChargedParticleTransportStatus.SECONDARY_CAPACITY_EXHAUSTED),
                    carry.status,
                ),
            ).astype(jnp.int32)
            record = supported & (step_index < bank_capacity)
            slot = jnp.clip(step_index, 0, bank_capacity - 1)
            starts, ends, start_beta, end_beta, deposits, materials, active = carry.bank
            bank = (
                starts.at[slot].set(jnp.where(record, carry.position, starts[slot])),
                ends.at[slot].set(jnp.where(record, trial, ends[slot])),
                start_beta.at[slot].set(
                    jnp.where(record, _beta(carry.kinetic), start_beta[slot])
                ),
                end_beta.at[slot].set(
                    jnp.where(record, _beta(remaining), end_beta[slot])
                ),
                deposits.at[slot].set(jnp.where(record, step_deposit, deposits[slot])),
                materials.at[slot].set(
                    jnp.where(record, location.material_index, materials[slot])
                ),
                active.at[slot].set(active[slot] | record),
            )
            unrecorded = carry.unrecorded + (supported & ~record).astype(jnp.int32)
            return _Carry(
                jnp.where(carry.live, trial, carry.position),
                jnp.where(supported, scattered, carry.direction),
                jnp.where(leaves | below_cutoff, 0.0, remaining),
                next_live,
                deposited,
                escaped,
                carry.bremsstrahlung + recorded_photon,
                annihilation,
                truncated,
                carry.path_length + jnp.where(supported, step_length, 0.0),
                carry.step_count + carry.live.astype(jnp.int32),
                status,
                bank,
                unrecorded,
                stack,
                stack_count,
                carry.stack_overflow + overflow.astype(jnp.int32),
            )

        final = jax.lax.fori_loop(0, self.maximum_steps, advance, state)
        truncated = final.truncated + jnp.where(final.live, final.kinetic, 0.0)
        # A history that already failed keeps its first failure; step capacity
        # is reported only for otherwise healthy histories still in flight.
        status = jnp.where(
            final.live & (final.status == int(ChargedParticleTransportStatus.SUCCESS)),
            int(ChargedParticleTransportStatus.STEP_CAPACITY_EXHAUSTED),
            final.status,
        ).astype(jnp.int32)
        ledger = (
            energy - final.deposited - final.escaped - final.bremsstrahlung - truncated
        )
        finite = jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    (
                        final.position,
                        final.direction,
                        jnp.asarray(
                            (
                                final.kinetic,
                                final.deposited,
                                final.escaped,
                                final.bremsstrahlung,
                                truncated,
                                final.path_length,
                                ledger,
                            )
                        ),
                    )
                )
            )
        )
        tolerance = 256.0 * jnp.finfo(energy.dtype).eps * jnp.maximum(energy, 1.0)
        successful = (
            initial_valid
            & finite
            & (jnp.abs(ledger) <= tolerance)
            & (status == int(ChargedParticleTransportStatus.SUCCESS))
        )
        return _Outcome(
            final.deposited,
            final.escaped,
            final.bremsstrahlung,
            final.annihilation,
            truncated,
            final.position,
            final.direction,
            final.kinetic,
            final.path_length,
            final.step_count,
            status,
            successful,
            ledger,
            final.bank,
            final.unrecorded,
            final.stack,
            final.stack_count,
            final.bremsstrahlung,
            final.stack_overflow,
        )

    def simulate(
        self,
        origins: ArrayLike,
        directions: ArrayLike,
        kinetic_energies_ev: ArrayLike,
        particle_kinds: ArrayLike,
        key: PRNGKey,
        /,
        *,
        identities: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> ChargedParticleTransportResult:
        origins_ = jnp.asarray(origins, dtype=self.geometry.lower.dtype)
        directions_ = jnp.asarray(directions, dtype=origins_.dtype)
        energy = jnp.asarray(kinetic_energies_ev, dtype=origins_.dtype)
        kinds = jnp.asarray(particle_kinds, dtype=jnp.int32)
        if (
            origins_.ndim != 2
            or origins_.shape[-1] != 3
            or directions_.shape != origins_.shape
        ):
            raise ValueError(
                "Charged particle positions/directions require (history, 3)."
            )
        count = origins_.shape[0]
        if energy.shape != (count,) or kinds.shape != (count,):
            raise ValueError(
                "Charged particle energy/kind arrays must align with histories."
            )
        id_hi, id_lo = _identity_words(identities, count)
        root = parse(key, PRNGKey, "key")
        outcome = jax.lax.map(
            lambda inputs: self._one(root, *inputs),
            (id_hi, id_lo, origins_, directions_, energy, kinds),
        )
        bank = None
        if self.step_bank_capacity is not None:
            starts, ends, start_beta, end_beta, deposits, materials, active = outcome.bank
            bank = ChargedStepBank(
                starts,
                ends,
                start_beta,
                end_beta,
                deposits,
                materials,
                active,
                jnp.sum(active, axis=1, dtype=jnp.int32),
                outcome.unrecorded,
            )
        stack = None
        if self.photon_stack is not None:
            (
                positions,
                stack_directions,
                energies,
                materials,
                creations,
                kinds,
                active,
            ) = outcome.stack
            stack = SecondaryParticleStack(
                positions,
                stack_directions,
                energies,
                materials,
                creations,
                kinds,
                active,
                outcome.stack_count,
                outcome.stack_energy,
                outcome.stack_overflow,
                self.photon_stack.spec_id,
            )
        return ChargedParticleTransportResult(
            id_hi,
            id_lo,
            outcome.deposited,
            outcome.escaped,
            outcome.bremsstrahlung,
            outcome.annihilation,
            outcome.truncated,
            outcome.position,
            outcome.direction,
            outcome.kinetic,
            outcome.path_length,
            outcome.step_count,
            outcome.status,
            outcome.successful,
            bank,
            stack,
            jnp.max(jnp.abs(outcome.ledger)),
            jnp.all(outcome.successful),
            self.plan_id,
        )


__all__ = [
    "ChargedParticleTransportPlan",
    "ChargedParticleTransportResult",
    "ChargedParticleTransportStatus",
    "ChargedStepBank",
]
