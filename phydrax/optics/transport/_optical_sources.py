#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cherenkov and scintillation optical photons from charged-particle steps.

A charged step is a straight segment whose speed ``beta`` is linear in path
length between its recorded endpoint speeds. Cherenkov emission uses the
dispersive Frank--Tamm density of the canonical owner
:func:`phydrax.equations.cherenkov_step_spectral_yield` on a declared
wavelength grid; scintillation uses a light yield with Birks quenching and
multi-exponential rise/decay components with tabulated emission spectra.

Photon counts are Poisson, and every draw is keyed by ``derive_key`` on the
parent particle's persistent identity words, the step index, and the photon
ordinal, so photons do not depend on parent order or batching. Photons fill
one fixed-capacity bank in canonical order (parent identity, step, process,
ordinal) and receive consecutive identities from ``first_identity``; the
parent identity and step are kept as lineage. A request larger than the
capacity is refused atomically: no photon is allocated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum, IntFlag
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import linear_interpolate
from ..._physical import ElectromagneticScaleContract, RelativityScaleContract
from ..._sampling import derive_key, SampleAddress
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...electromagnetics._trajectory_radiation import ChargedTrajectory
from ...equations._matter_radiation_interactions import cherenkov_step_spectral_yield
from ...solver._charged_particle_transport import ChargedParticleTransportResult
from ...typing import (
    as_array,
    as_host_array,
    Bool,
    checked,
    ConvertibleToArray,
    Dim,
    Float64,
    HostFloat64,
    Identifier,
    Int32,
    parse,
    PRNGKey,
    Scope,
    Size,
    UInt32,
)
from ._optical_media import _piecewise_linear_cdf, _sample_piecewise_linear
from ._optical_monte_carlo import OpticalMedium, OpticalPhotonState


class _ParentDim(Dim, minimum=1):
    """Charged parent particles."""


class _StepDim(Dim, minimum=1):
    """Step slots per parent."""


class _SpectralDim(Dim, minimum=2):
    """Nodes of the declared Cherenkov wavelength grid."""


class _MediumDim(Dim, minimum=1):
    """Optical media."""


class _ComponentDim(Dim, minimum=1):
    """Scintillation time components."""


class _EmissionDim(Dim, minimum=2):
    """Nodes of the scintillation emission-spectrum grid."""


class _PhotonDim(Dim, minimum=1):
    """Photon slots of the fixed-capacity bank."""


_CHERENKOV_COUNT_ADDRESS = SampleAddress(
    "optics", "charged-step-optical-source", target="cherenkov-count"
)
_SCINTILLATION_COUNT_ADDRESS = SampleAddress(
    "optics", "charged-step-optical-source", target="scintillation-count"
)
_CHERENKOV_PHOTON_ADDRESS = SampleAddress(
    "optics", "charged-step-optical-source", target="cherenkov-photon"
)
_SCINTILLATION_PHOTON_ADDRESS = SampleAddress(
    "optics", "charged-step-optical-source", target="scintillation-photon"
)

_WORD = 1 << 32
_WORD_BITS = np.uint64(32)
# Both words all-ones is the reserved identity (F6 rule): it marks empty slots.
_RESERVED_IDENTITY = (1 << 64) - 1
_RESERVED_WORD = np.uint32(0xFFFFFFFF)
# Below this relative speed spread a step is treated as constant-speed when
# placing a Cherenkov photon; the density variation over the step is then
# below double-precision resolution of the emission point.
_CONSTANT_SPEED_SPREAD = 1e-9
_UNIFORMS_PER_PHOTON = 8


class OpticalEmissionProcess(IntEnum):
    """Process that created an optical photon."""

    CHERENKOV = 0
    SCINTILLATION = 1


class OpticalEmissionStatus(IntFlag):
    """Emission evidence; any nonzero flag refuses the whole photon bank."""

    SUCCESS = 0
    CAPACITY_EXHAUSTED = 1
    INCOMPLETE_STEPS = 2
    INVALID_STEPS = 4
    IDENTITY_EXHAUSTED = 8
    NONFINITE_RESULT = 16


class ChargedOpticalSteps(StrictModule):
    """Straight charged steps with speed linear in path length.

    ``start_beta``/``end_beta`` are endpoint speeds in units of ``c``;
    ``start_times`` are the step start times, and a point at path length ``s``
    of a step is reached at ``start_time + (1 / c) * integral ds / beta``.
    ``medium_indices`` name the optical medium of each step (``-1``: no
    optical medium, the step emits nothing). ``deposited_energy`` is the
    energy left in the medium by the step, in the scintillation yield's energy
    unit; ``deposit_recorded`` is false when the producer recorded none.
    ``complete`` is false for a parent whose producer dropped steps.
    ``multiplicities`` count identical physical particles per parent.
    Lengths and times share the unit system of ``speed_of_light``.
    """

    __strict_contract__ = True

    start_positions: Float64[_ParentDim, _StepDim, Literal[3]]
    end_positions: Float64[_ParentDim, _StepDim, Literal[3]]
    start_times: Float64[_ParentDim, _StepDim]
    start_beta: Float64[_ParentDim, _StepDim]
    end_beta: Float64[_ParentDim, _StepDim]
    deposited_energy: Float64[_ParentDim, _StepDim]
    medium_indices: Int32[_ParentDim, _StepDim]
    active: Bool[_ParentDim, _StepDim]
    charge_numbers: Float64[_ParentDim]
    multiplicities: Float64[_ParentDim]
    complete: Bool[_ParentDim]
    id_hi: UInt32[_ParentDim]
    id_lo: UInt32[_ParentDim]
    speed_of_light: float = eqx.field(static=True)
    deposit_recorded: bool = eqx.field(static=True)

    def __init__(
        self,
        start_positions: ConvertibleToArray,
        end_positions: ConvertibleToArray,
        start_times: ConvertibleToArray,
        start_beta: ConvertibleToArray,
        end_beta: ConvertibleToArray,
        medium_indices: ConvertibleToArray,
        active: ConvertibleToArray,
        identities: tuple[ConvertibleToArray, ConvertibleToArray],
        /,
        *,
        speed_of_light: float,
        deposited_energy: ConvertibleToArray | None = None,
        charge_numbers: ConvertibleToArray = 1.0,
        multiplicities: ConvertibleToArray = 1.0,
        complete: ConvertibleToArray = True,
    ) -> None:
        light = float(speed_of_light)
        if not math.isfinite(light) or light <= 0.0:
            raise ValueError("speed_of_light must be finite and positive.")
        scope = Scope()
        self.start_positions = as_array(
            start_positions,
            Float64[_ParentDim, _StepDim, Literal[3]],
            "start_positions",
            scope=scope,
        )
        steps = self.start_positions.shape[:2]
        self.end_positions = as_array(
            end_positions,
            Float64[_ParentDim, _StepDim, Literal[3]],
            "end_positions",
            scope=scope,
        )
        self.start_times = as_array(
            start_times, Float64[_ParentDim, _StepDim], "start_times", scope=scope
        )
        self.start_beta = as_array(
            start_beta, Float64[_ParentDim, _StepDim], "start_beta", scope=scope
        )
        self.end_beta = as_array(
            end_beta, Float64[_ParentDim, _StepDim], "end_beta", scope=scope
        )
        self.deposited_energy = as_array(
            jnp.zeros(steps, dtype=jnp.float64)
            if deposited_energy is None
            else deposited_energy,
            Float64[_ParentDim, _StepDim],
            "deposited_energy",
            scope=scope,
        )
        self.medium_indices = as_array(
            medium_indices, Int32[_ParentDim, _StepDim], "medium_indices", scope=scope
        )
        self.active = as_array(active, Bool[_ParentDim, _StepDim], "active", scope=scope)
        self.charge_numbers = as_array(
            jnp.broadcast_to(jnp.asarray(charge_numbers, dtype=jnp.float64), steps[:1]),
            Float64[_ParentDim],
            "charge_numbers",
            scope=scope,
        )
        self.multiplicities = as_array(
            jnp.broadcast_to(jnp.asarray(multiplicities, dtype=jnp.float64), steps[:1]),
            Float64[_ParentDim],
            "multiplicities",
            scope=scope,
        )
        self.complete = as_array(
            jnp.broadcast_to(jnp.asarray(complete, dtype=jnp.bool_), steps[:1]),
            Bool[_ParentDim],
            "complete",
            scope=scope,
        )
        hi, lo = identities
        self.id_hi = as_array(hi, UInt32[_ParentDim], "identities[0]", scope=scope)
        self.id_lo = as_array(lo, UInt32[_ParentDim], "identities[1]", scope=scope)
        self.speed_of_light = light
        self.deposit_recorded = deposited_energy is not None


def _inverse_speed_mean(first: Array, current: Array) -> Array:
    """Path-averaged ``1 / beta`` over a segment with speed linear in path.

    For speeds ``b0 -> b1`` the mean is ``ln(b1 / b0) / (b1 - b0)``,
    evaluated as ``log1p(x) / (x b0)`` with ``x = (b1 - b0) / b0`` and its
    series near ``x = 0``. It is infinite when the speed reaches zero.
    """

    safe = jnp.where(first > 0.0, first, 1.0)
    x = (current - first) / safe
    small = jnp.abs(x) < 1e-6
    ratio = jnp.where(
        small,
        1.0 - 0.5 * x + x * x / 3.0,
        jnp.log1p(jnp.where(small, 0.0, x)) / jnp.where(small, 1.0, x),
    )
    return jnp.where(first > 0.0, ratio / safe, jnp.inf)


def charged_steps_from_transport(
    result: ChargedParticleTransportResult,
    material_media: Sequence[int],
    relativity: RelativityScaleContract,
    /,
) -> ChargedOpticalSteps:
    """Optical steps of a condensed-history charged transport step bank.

    ``material_media[m]`` is the optical medium of charged material ``m``
    (``-1`` for a non-optical material); materials outside the map mark their
    steps invalid. Every parent is an electron or positron (``|z| = 1``) with
    identity ``(result.id_hi, result.id_lo)``. Step times start at zero for
    each history and accumulate the exact linear-speed transit time of the
    preceding recorded steps; positions and deposits keep the bank's units.
    """

    if not isinstance(result, ChargedParticleTransportResult):
        raise TypeError("result must be a ChargedParticleTransportResult.")
    if not isinstance(relativity, RelativityScaleContract):
        raise TypeError("relativity must be a RelativityScaleContract.")
    bank = result.step_bank
    if bank is None:
        raise ValueError("The charged transport recorded no step bank.")
    media = tuple(material_media)
    if not media or any(type(value) is not int or value < -1 for value in media):
        raise ValueError("material_media must be a non-empty sequence of ints >= -1.")
    light = float(relativity.speed_of_light)
    table = jnp.asarray(media, dtype=jnp.int32)
    material = bank.material_index
    inside = (material >= 0) & (material < table.shape[0])
    medium = jnp.where(
        inside, table[jnp.clip(material, 0, table.shape[0] - 1)], -2
    ).astype(jnp.int32)
    lengths = jnp.linalg.norm(bank.end_positions - bank.start_positions, axis=-1)
    durations = jnp.where(
        bank.active,
        lengths * _inverse_speed_mean(bank.start_beta, bank.end_beta) / light,
        0.0,
    )
    # Exclusive prefix sum: a stopping final step never delays its own start.
    starts = jnp.cumsum(
        jnp.concatenate((jnp.zeros_like(durations[:, :1]), durations[:, :-1]), axis=1),
        axis=1,
    )
    return ChargedOpticalSteps(
        bank.start_positions,
        bank.end_positions,
        starts,
        bank.start_beta,
        bank.end_beta,
        medium,
        bank.active,
        (result.id_hi, result.id_lo),
        speed_of_light=light,
        deposited_energy=bank.deposited_energy,
        complete=bank.complete,
    )


def charged_steps_from_trajectory(
    trajectory: ChargedTrajectory,
    scale: ElectromagneticScaleContract,
    medium_indices: ArrayLike,
    /,
    *,
    deposited_energy: ArrayLike | None = None,
) -> ChargedOpticalSteps:
    """Optical steps between consecutive active samples of charged lanes.

    Step ``k`` of lane ``p`` joins samples ``k`` and ``k + 1``; it is active
    when both samples are. Speeds are ``|u| / sqrt(c^2 + |u|^2)`` of the
    proper velocities, charge numbers are ``charges / e``, and positions and
    times keep the trajectory's ``scale`` units. ``medium_indices`` and the
    optional ``deposited_energy`` broadcast to ``(lanes, samples - 1)``.
    """

    if not isinstance(trajectory, ChargedTrajectory):
        raise TypeError("trajectory must be a ChargedTrajectory.")
    if not isinstance(scale, ElectromagneticScaleContract):
        raise TypeError("scale must be an ElectromagneticScaleContract.")
    if trajectory.sample_count < 2:
        raise ValueError("A trajectory needs at least two samples to form a step.")
    light = float(scale.speed_of_light)
    shape = (trajectory.particle_count, trajectory.sample_count - 1)
    proper = jnp.linalg.norm(trajectory.proper_velocities, axis=-1)
    beta = jnp.swapaxes(proper / jnp.sqrt(light * light + proper * proper), 0, 1)
    positions = jnp.swapaxes(trajectory.positions, 0, 1)
    sampled = jnp.swapaxes(trajectory.active, 0, 1)
    return ChargedOpticalSteps(
        positions[:, :-1],
        positions[:, 1:],
        jnp.swapaxes(trajectory.times, 0, 1)[:, :-1],
        beta[:, :-1],
        beta[:, 1:],
        jnp.broadcast_to(jnp.asarray(medium_indices, dtype=jnp.int32), shape),
        sampled[:, :-1] & sampled[:, 1:],
        (trajectory.id_hi, trajectory.id_lo),
        speed_of_light=light,
        deposited_energy=(
            None
            if deposited_energy is None
            else jnp.broadcast_to(jnp.asarray(deposited_energy, dtype=jnp.float64), shape)
        ),
        charge_numbers=trajectory.charges / float(scale.elementary_charge),
        multiplicities=trajectory.multiplicities,
    )


class CherenkovEmission(StrictModule, NonTrainableState):
    """Declared Frank--Tamm wavelength grid and the media's dispersion on it.

    ``wavelengths`` (strictly increasing, in the optical medium's wavelength
    unit) span the emitted band; the refractive index of every medium is read
    once from ``medium`` on these nodes, and the Frank--Tamm spectral yield is
    represented piecewise linearly between them, so the band integral and the
    sampled spectrum agree exactly and converge at second order in the node
    spacing. ``radiating_media`` selects the media that emit.
    """

    __strict_contract__ = True

    wavelengths: Float64[_SpectralDim]
    refractive_indices: Float64[_MediumDim, _SpectralDim]
    radiating_media: Bool[_MediumDim]
    medium_count: Size[_MediumDim] = eqx.field(static=True)

    def __init__(
        self,
        medium: OpticalMedium,
        wavelengths: ConvertibleToArray,
        /,
        *,
        radiating_media: ConvertibleToArray | None = None,
    ) -> None:
        scope = Scope()
        grid = as_host_array(
            wavelengths, HostFloat64[_SpectralDim], "wavelengths", scope=scope
        )
        if (
            not np.all(np.isfinite(grid))
            or grid[0] <= 0.0
            or np.any(np.diff(grid) <= 0.0)
        ):
            raise ValueError("wavelengths must be positive and strictly increasing.")
        count = medium.medium_count
        nodes = jnp.asarray(grid)
        rows = []
        for index in range(count):
            media = jnp.full(grid.shape, index, dtype=jnp.int32)
            if not bool(jnp.all(medium.spectral_support(media, nodes))):
                raise ValueError(
                    f"Medium {index} does not cover the declared Cherenkov band."
                )
            rows.append(np.asarray(medium.refractive_index(media, nodes)))
        index_table = np.stack(rows)
        if not np.all(np.isfinite(index_table)) or np.any(index_table <= 0.0):
            raise ValueError("Medium refractive indices must be finite and positive.")
        radiating = (
            np.ones((count,), dtype=np.bool_)
            if radiating_media is None
            else np.asarray(radiating_media)
        )
        self.wavelengths = jnp.asarray(grid)
        self.refractive_indices = as_array(
            index_table,
            Float64[_MediumDim, _SpectralDim],
            "refractive_indices",
            scope=scope,
        )
        self.radiating_media = as_array(
            radiating, Bool[_MediumDim], "radiating_media", scope=scope
        )
        self.medium_count = parse(count, Size[_MediumDim], "medium_count", scope=scope)


class ScintillationEmission(StrictModule, NonTrainableState):
    """Light yield, Birks quenching, time components, and emission spectra.

    Per medium, ``photon_yields`` are photons per unit visible energy in the
    steps' deposited-energy unit and ``birks_constants`` ``kB`` convert a
    step's mean ``dE/dx`` into visible energy ``E / (1 + kB dE/dx)`` (Birks
    1951). Component ``j`` is chosen with probability
    ``component_fractions[m, j]``; its emission time after excitation is
    the sum of independent exponentials of means ``rise_times`` and
    ``decay_times``, whose density is the bi-exponential
    ``(exp(-t / tau_d) - exp(-t / tau_r)) / (tau_d - tau_r)``, and its
    wavelength follows ``emission_spectra[m, j]``, piecewise linear on
    ``emission_wavelengths`` and sampled by exact inverse CDF. Photons are
    isotropic and randomly linearly polarized.
    """

    __strict_contract__ = True

    photon_yields: Float64[_MediumDim]
    birks_constants: Float64[_MediumDim]
    component_fractions: Float64[_MediumDim, _ComponentDim]
    rise_times: Float64[_MediumDim, _ComponentDim]
    decay_times: Float64[_MediumDim, _ComponentDim]
    emission_wavelengths: Float64[_EmissionDim]
    emission_densities: Float64[_MediumDim, _ComponentDim, _EmissionDim]
    emission_cdf: Float64[_MediumDim, _ComponentDim, _EmissionDim]
    medium_count: Size[_MediumDim] = eqx.field(static=True)

    def __init__(
        self,
        photon_yields: ConvertibleToArray,
        birks_constants: ConvertibleToArray,
        component_fractions: ConvertibleToArray,
        rise_times: ConvertibleToArray,
        decay_times: ConvertibleToArray,
        emission_wavelengths: ConvertibleToArray,
        emission_spectra: ConvertibleToArray,
        /,
    ) -> None:
        scope = Scope()
        yields = as_host_array(
            photon_yields, HostFloat64[_MediumDim], "photon_yields", scope=scope
        )
        birks = as_host_array(
            birks_constants, HostFloat64[_MediumDim], "birks_constants", scope=scope
        )
        fractions = as_host_array(
            component_fractions,
            HostFloat64[_MediumDim, _ComponentDim],
            "component_fractions",
            scope=scope,
        )
        rise = as_host_array(
            rise_times, HostFloat64[_MediumDim, _ComponentDim], "rise_times", scope=scope
        )
        decay = as_host_array(
            decay_times,
            HostFloat64[_MediumDim, _ComponentDim],
            "decay_times",
            scope=scope,
        )
        grid = as_host_array(
            emission_wavelengths,
            HostFloat64[_EmissionDim],
            "emission_wavelengths",
            scope=scope,
        )
        spectra = as_host_array(
            emission_spectra,
            HostFloat64[_MediumDim, _ComponentDim, _EmissionDim],
            "emission_spectra",
            scope=scope,
        )
        if not np.all(np.isfinite(yields)) or np.any(yields < 0.0):
            raise ValueError("photon_yields must be finite and non-negative.")
        if not np.all(np.isfinite(birks)) or np.any(birks < 0.0):
            raise ValueError("birks_constants must be finite and non-negative.")
        if not np.all(np.isfinite(fractions)) or np.any(fractions < 0.0):
            raise ValueError("component_fractions must be finite and non-negative.")
        scintillating = yields > 0.0
        if np.any(np.abs(fractions.sum(axis=-1) - 1.0)[scintillating] > 1e-12):
            raise ValueError("component_fractions of a scintillator must sum to one.")
        if not np.all(np.isfinite(rise)) or np.any(rise < 0.0):
            raise ValueError("rise_times must be finite and non-negative.")
        if not np.all(np.isfinite(decay)) or np.any(decay <= 0.0):
            raise ValueError("decay_times must be finite and positive.")
        if (
            not np.all(np.isfinite(grid))
            or grid[0] <= 0.0
            or np.any(np.diff(grid) <= 0.0)
        ):
            raise ValueError(
                "emission_wavelengths must be positive and strictly increasing."
            )
        if not np.all(np.isfinite(spectra)) or np.any(spectra < 0.0):
            raise ValueError("emission_spectra must be finite and non-negative.")
        widths = np.diff(grid)
        totals = np.sum(0.5 * widths * (spectra[..., :-1] + spectra[..., 1:]), axis=-1)
        used = scintillating[:, None] & (fractions > 0.0)
        if np.any(used & (totals <= 0.0)):
            raise ValueError(
                "Every used scintillation component needs a positive emission spectrum."
            )
        densities = spectra / np.where(totals > 0.0, totals, 1.0)[..., None]
        form = Float64[_MediumDim, _ComponentDim, _EmissionDim]
        self.photon_yields = as_array(yields, Float64[_MediumDim], "photon_yields")
        self.birks_constants = as_array(birks, Float64[_MediumDim], "birks_constants")
        self.component_fractions = as_array(
            fractions, Float64[_MediumDim, _ComponentDim], "component_fractions"
        )
        self.rise_times = as_array(rise, Float64[_MediumDim, _ComponentDim], "rise_times")
        self.decay_times = as_array(
            decay, Float64[_MediumDim, _ComponentDim], "decay_times"
        )
        self.emission_wavelengths = as_array(
            grid, Float64[_EmissionDim], "emission_wavelengths"
        )
        self.emission_densities = as_array(densities, form, "emission_densities")
        self.emission_cdf = as_array(
            _piecewise_linear_cdf(grid, densities), form, "emission_cdf"
        )
        self.medium_count = parse(
            yields.shape[0], Size[_MediumDim], "medium_count", scope=scope
        )


class OpticalPhotonSourcePlan(StrictModule, NonTrainableState):
    """Cherenkov and/or scintillation emission into one fixed-capacity bank.

    ``relativity`` fixes the transport's length and time units: steps must
    carry its exact speed of light, and photon energies ``2 pi hbar c /
    lambda`` are reported in its energy unit. ``length_per_wavelength_unit``
    converts optical wavelengths into transport lengths. ``photon_capacity``
    is the bank size; ``batch_size`` bounds vectorized parents and photons.
    """

    __strict_contract__ = True

    cherenkov: CherenkovEmission | None
    scintillation: ScintillationEmission | None
    relativity: RelativityScaleContract = eqx.field(static=True)
    length_per_wavelength_unit: float = eqx.field(static=True)
    photon_capacity: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    medium_count: int = eqx.field(static=True)
    speed_of_light: float = eqx.field(static=True)
    photon_energy_length: float = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        relativity: RelativityScaleContract,
        photon_capacity: int,
        cherenkov: CherenkovEmission | None = None,
        scintillation: ScintillationEmission | None = None,
        length_per_wavelength_unit: float = 1.0,
        batch_size: int = 1024,
    ) -> None:
        if not relativity.quantum_constants_explicit:
            raise ValueError("Photon energies need an explicit reduced Planck constant.")
        if cherenkov is not None and not isinstance(cherenkov, CherenkovEmission):
            raise TypeError("cherenkov must be a CherenkovEmission.")
        if scintillation is not None and not isinstance(
            scintillation, ScintillationEmission
        ):
            raise TypeError("scintillation must be a ScintillationEmission.")
        if cherenkov is None and scintillation is None:
            raise ValueError("At least one optical emission process is required.")
        counts = {
            process.medium_count
            for process in (cherenkov, scintillation)
            if process is not None
        }
        if len(counts) != 1:
            raise ValueError("Emission processes must describe the same media.")
        capacity, batch = int(photon_capacity), int(batch_size)
        if capacity < 1 or batch < 1:
            raise ValueError("photon_capacity and batch_size must be positive.")
        scale = float(length_per_wavelength_unit)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("length_per_wavelength_unit must be finite and positive.")
        self.cherenkov = cherenkov
        self.scintillation = scintillation
        self.relativity = relativity
        self.length_per_wavelength_unit = scale
        self.photon_capacity = capacity
        self.batch_size = batch
        self.medium_count = counts.pop()
        self.speed_of_light = float(relativity.speed_of_light)
        self.photon_energy_length = float(
            2 * math.pi * relativity.reduced_planck_constant * relativity.speed_of_light
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "charged-step-optical-source",
                "relativity": relativity.scale_id,
                "cherenkov": None
                if cherenkov is None
                else array_tree_fingerprint(
                    {
                        "wavelengths": np.asarray(cherenkov.wavelengths),
                        "refractive_indices": np.asarray(cherenkov.refractive_indices),
                        "radiating_media": np.asarray(cherenkov.radiating_media),
                    }
                ),
                "scintillation": None
                if scintillation is None
                else array_tree_fingerprint(
                    {
                        "photon_yields": np.asarray(scintillation.photon_yields),
                        "birks_constants": np.asarray(scintillation.birks_constants),
                        "component_fractions": np.asarray(
                            scintillation.component_fractions
                        ),
                        "rise_times": np.asarray(scintillation.rise_times),
                        "decay_times": np.asarray(scintillation.decay_times),
                        "emission_wavelengths": np.asarray(
                            scintillation.emission_wavelengths
                        ),
                        "emission_densities": np.asarray(
                            scintillation.emission_densities
                        ),
                    }
                ),
                "length_per_wavelength_unit": scale,
                "photon_capacity": capacity,
                "batch_size": batch,
            }
        )


class OpticalPhotonEmission(StrictModule, NonTrainableState):
    """Fixed-capacity optical photon bank with lineage and ledgers.

    ``state`` is launchable through :class:`ExplicitPhotonSource`; inactive
    slots have zero weight and the reserved identity. ``processes`` are
    :class:`OpticalEmissionProcess` values (``-1`` when inactive) and
    ``(parent_hi, parent_lo, parent_steps)`` the lineage. Per-parent arrays
    follow the input parent order: expected and sampled photon counts,
    deposited and Birks-visible energy of scintillating steps, the energy of
    the allocated photons in the relativity scale's energy unit, and active
    steps outside every optical medium. ``requested_count`` is the sampled
    total; when ``status`` is nonzero nothing is allocated.
    ``(next_id_hi, next_id_lo)`` is the first identity after the bank.
    """

    __strict_contract__ = True

    state: OpticalPhotonState
    active: Bool[_PhotonDim]
    processes: Int32[_PhotonDim]
    parent_hi: UInt32[_PhotonDim]
    parent_lo: UInt32[_PhotonDim]
    parent_steps: Int32[_PhotonDim]
    cherenkov_expected: Float64[_ParentDim]
    cherenkov_counts: Int32[_ParentDim]
    scintillation_expected: Float64[_ParentDim]
    scintillation_counts: Int32[_ParentDim]
    deposited_energy: Float64[_ParentDim]
    visible_energy: Float64[_ParentDim]
    cherenkov_photon_energy: Float64[_ParentDim]
    scintillation_photon_energy: Float64[_ParentDim]
    nonoptical_steps: Int32[_ParentDim]
    requested_count: Array
    allocated_count: Array
    next_id_hi: Array
    next_id_lo: Array
    status: Array
    successful: Array
    plan_id: Identifier = eqx.field(static=True)


def _transverse_basis(direction: Array) -> tuple[Array, Array]:
    """Right-handed unit pair ``(a, b)`` with ``(a, b, direction)`` orthonormal."""

    reference = jnp.where(
        jnp.abs(direction[2]) < 0.9,
        jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.float64),
        jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.float64),
    )
    first = jnp.cross(reference, direction)
    first = first / jnp.linalg.norm(first)
    return first, jnp.cross(direction, first)


def _cherenkov_speed(first: Array, last: Array, index: Array, uniform: Array) -> Array:
    """Emission speed by exact inverse CDF of ``1 - 1 / (n^2 beta^2)``.

    On the above-threshold interval ``[b_lo, b_hi]`` the CDF is ``F(b) = (b -
    b_lo)(1 - 1 / (n^2 b_lo b))``; ``F(b_lo + d) = u F(b_hi)`` is the
    quadratic ``n^2 b_lo d^2 + (n^2 b_lo^2 - 1 - r n^2 b_lo) d - r n^2 b_lo^2
    = 0`` with ``r = u F(b_hi)``, solved in cancellation-free form.
    """

    low = jnp.minimum(first, last)
    high = jnp.maximum(first, last)
    lower = jnp.clip(1.0 / index, low, high)
    width = high - lower
    squared = index * index
    target = uniform * width * (1.0 - 1.0 / (squared * lower * high))
    a = squared * lower
    b = squared * lower * lower - 1.0 - target * squared * lower
    c = -target * squared * lower * lower
    root = jnp.sqrt(jnp.maximum(b * b - 4.0 * a * c, 0.0))
    denominator = b + root
    offset = jnp.where(
        b >= 0.0,
        jnp.where(
            denominator > 0.0,
            -2.0 * c / jnp.where(denominator > 0.0, denominator, 1.0),
            0.0,
        ),
        (root - b) / (2.0 * a),
    )
    return lower + jnp.clip(offset, 0.0, width)


class _CanonicalSteps(StrictModule):
    """Step arrays of the parents in canonical identity order."""

    starts: Array
    ends: Array
    times: Array
    first: Array
    last: Array
    lengths: Array
    media: Array
    emitting: Array
    charges: Array
    multiplicities: Array
    deposits: Array
    id_hi: Array
    id_lo: Array


def _spectral_lengths(plan: OpticalPhotonSourcePlan) -> Array:
    cherenkov = plan.cherenkov
    if cherenkov is None:
        raise RuntimeError("The plan has no Cherenkov emission.")
    return cherenkov.wavelengths * plan.length_per_wavelength_unit


def _cherenkov_rows(
    plan: OpticalPhotonSourcePlan,
    length: Array,
    first: Array,
    last: Array,
    medium: Array,
    charge: Array,
) -> Array:
    """Frank--Tamm yield per unit wavelength length on the declared nodes."""

    cherenkov = plan.cherenkov
    if cherenkov is None:
        raise RuntimeError("The plan has no Cherenkov emission.")
    rows, _ = cherenkov_step_spectral_yield(
        length[..., None],
        first[..., None],
        last[..., None],
        cherenkov.refractive_indices[medium],
        _spectral_lengths(plan),
        charge_number=charge[..., None],
    )
    return jnp.where(cherenkov.radiating_media[medium][..., None], rows, 0.0)


def _trapezoid_weights(nodes: Array) -> Array:
    widths = jnp.diff(nodes)
    zero = jnp.zeros((1,), dtype=nodes.dtype)
    return 0.5 * (jnp.concatenate((zero, widths)) + jnp.concatenate((widths, zero)))


def _visible_energy(
    plan: OpticalPhotonSourcePlan, deposit: Array, length: Array, medium: Array
) -> Array:
    scintillation = plan.scintillation
    if scintillation is None:
        return jnp.zeros_like(deposit)
    birks = scintillation.birks_constants[medium]
    quench = birks * deposit
    # E / (1 + kB E / L) written without dividing by a zero-length step.
    return jnp.where(quench > 0.0, deposit * length / (length + quench), deposit)


def _expected_counts(
    plan: OpticalPhotonSourcePlan, steps: _CanonicalSteps
) -> tuple[Array, Array, Array]:
    """Expected Cherenkov and scintillation photons and visible energy per step."""

    def one(
        parent: tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array]:
        length, first, last, medium, emitting, charge, weight, deposit = parent
        if plan.cherenkov is None:
            cherenkov = jnp.zeros_like(length)
        else:
            rows = _cherenkov_rows(
                plan, length, first, last, medium, jnp.broadcast_to(charge, length.shape)
            )
            cherenkov = rows @ _trapezoid_weights(_spectral_lengths(plan))
        visible = _visible_energy(plan, deposit, length, medium)
        scintillation = plan.scintillation
        light = (
            jnp.zeros_like(length)
            if scintillation is None
            else scintillation.photon_yields[medium] * visible
        )
        return (
            jnp.where(emitting, weight * cherenkov, 0.0),
            jnp.where(emitting, weight * light, 0.0),
            jnp.where(emitting, visible, 0.0),
        )

    return jax.lax.map(
        one,
        (
            steps.lengths,
            steps.first,
            steps.last,
            steps.media,
            steps.emitting,
            steps.charges,
            steps.multiplicities,
            steps.deposits,
        ),
        batch_size=plan.batch_size,
    )


def _poisson_counts(
    root: PRNGKey, steps: _CanonicalSteps, expected: Array, address: SampleAddress
) -> Array:
    slots = jnp.arange(expected.shape[1], dtype=jnp.uint32)

    def parent(id_hi: Array, id_lo: Array, means: Array) -> Array:
        def step(slot: Array, mean: Array) -> Array:
            key = derive_key(root, address, slot, id_hi, id_lo)
            return jr.poisson(key, mean, dtype=jnp.int32)

        return jax.vmap(step)(slots, means)

    return jax.vmap(parent)(steps.id_hi, steps.id_lo, jnp.nan_to_num(expected))


class _Photon(StrictModule):
    position: Array
    direction: Array
    axis: Array
    wavelength: Array
    time: Array
    energy: Array


def _cherenkov_photon(
    plan: OpticalPhotonSourcePlan,
    steps: _CanonicalSteps,
    parent: Array,
    step: Array,
    uniforms: Array,
) -> _Photon:
    start = steps.starts[parent, step]
    chord = steps.ends[parent, step] - start
    length = steps.lengths[parent, step]
    first = steps.first[parent, step]
    last = steps.last[parent, step]
    medium = steps.media[parent, step]
    nodes = _spectral_lengths(plan)
    row = _cherenkov_rows(plan, length, first, last, medium, steps.charges[parent])
    total = row @ _trapezoid_weights(nodes)
    density = row / jnp.where(total > 0.0, total, 1.0)
    wavelength = _sample_piecewise_linear(
        nodes, density[None], _piecewise_linear_cdf(nodes, row)[None], uniforms[:1]
    )[0]
    cherenkov = plan.cherenkov
    if cherenkov is None:
        raise RuntimeError("The plan has no Cherenkov emission.")
    index = linear_interpolate(
        nodes, cherenkov.refractive_indices[medium], wavelength, bounds="clip"
    ).values
    beta = _cherenkov_speed(first, last, index, uniforms[1])
    spread = jnp.abs(last - first)
    varying = spread > _CONSTANT_SPEED_SPREAD * jnp.maximum(first, last)
    fraction = jnp.clip(
        jnp.where(
            varying,
            (beta - first) / jnp.where(varying, last - first, 1.0),
            uniforms[1],
        ),
        0.0,
        1.0,
    )
    axis = chord / jnp.where(length > 0.0, length, 1.0)
    a, b = _transverse_basis(axis)
    cosine = jnp.minimum(1.0 / (beta * index), 1.0)
    sine = jnp.sqrt(jnp.maximum(1.0 - cosine * cosine, 0.0))
    azimuth = 2.0 * jnp.pi * uniforms[2]
    radial = jnp.cos(azimuth) * a + jnp.sin(azimuth) * b
    direction = cosine * axis + sine * radial
    # p x (p x v) = (p . v) p - v: the field lies in the (v, p) plane, radial.
    polarization = cosine * radial - sine * axis
    speed = first + (last - first) * fraction
    time = (
        steps.times[parent, step]
        + fraction * length * _inverse_speed_mean(first, speed) / plan.speed_of_light
    )
    return _Photon(
        start + fraction * chord,
        direction,
        polarization,
        wavelength / plan.length_per_wavelength_unit,
        time,
        plan.photon_energy_length / wavelength,
    )


def _scintillation_photon(
    plan: OpticalPhotonSourcePlan,
    steps: _CanonicalSteps,
    parent: Array,
    step: Array,
    uniforms: Array,
) -> _Photon:
    scintillation = plan.scintillation
    if scintillation is None:
        raise RuntimeError("The plan has no scintillation emission.")
    start = steps.starts[parent, step]
    chord = steps.ends[parent, step] - start
    length = steps.lengths[parent, step]
    first = steps.first[parent, step]
    last = steps.last[parent, step]
    medium = steps.media[parent, step]
    fractions = jnp.cumsum(scintillation.component_fractions[medium])
    component = jnp.clip(
        jnp.searchsorted(fractions, uniforms[0] * fractions[-1], side="right"),
        0,
        fractions.shape[0] - 1,
    )
    wavelength = _sample_piecewise_linear(
        scintillation.emission_wavelengths,
        scintillation.emission_densities[medium, component][None],
        scintillation.emission_cdf[medium, component][None],
        uniforms[1:2],
    )[0]
    fraction = uniforms[2]
    cosine = 1.0 - 2.0 * uniforms[3]
    sine = jnp.sqrt(jnp.maximum(1.0 - cosine * cosine, 0.0))
    azimuth = 2.0 * jnp.pi * uniforms[4]
    direction = jnp.stack((sine * jnp.cos(azimuth), sine * jnp.sin(azimuth), cosine))
    a, b = _transverse_basis(direction)
    angle = 2.0 * jnp.pi * uniforms[5]
    polarization = jnp.cos(angle) * a + jnp.sin(angle) * b
    # Sum of exponential rise and decay delays: the bi-exponential time law.
    delay = -scintillation.decay_times[medium, component] * jnp.log1p(
        -uniforms[6]
    ) - scintillation.rise_times[medium, component] * jnp.log1p(-uniforms[7])
    speed = first + (last - first) * fraction
    transit = fraction * length * _inverse_speed_mean(first, speed) / plan.speed_of_light
    return _Photon(
        start + fraction * chord,
        direction,
        polarization,
        wavelength,
        steps.times[parent, step] + transit + delay,
        plan.photon_energy_length / (wavelength * plan.length_per_wavelength_unit),
    )


def _unpermute(values: Array, order: Array) -> Array:
    """Return canonical-order per-parent values in the input parent order."""

    return jnp.zeros_like(values).at[order].set(values)


def _canonical_steps(
    plan: OpticalPhotonSourcePlan, steps: ChargedOpticalSteps
) -> tuple[_CanonicalSteps, Array, Array, Array]:
    """Parents sorted by identity, emitting mask, and input evidence."""

    packed = (steps.id_hi.astype(jnp.uint64) << _WORD_BITS) | steps.id_lo.astype(
        jnp.uint64
    )
    order = jnp.argsort(packed, stable=True)
    ordered = packed[order]
    duplicate = jnp.any(ordered[1:] == ordered[:-1])
    medium = steps.medium_indices
    step_valid = (
        jnp.all(jnp.isfinite(steps.start_positions), axis=-1)
        & jnp.all(jnp.isfinite(steps.end_positions), axis=-1)
        & jnp.isfinite(steps.start_times)
        & (steps.start_beta > 0.0)
        & (steps.start_beta < 1.0)
        & (steps.end_beta >= 0.0)
        & (steps.end_beta < 1.0)
        & jnp.isfinite(steps.deposited_energy)
        & (steps.deposited_energy >= 0.0)
        & (medium >= -1)
        & (medium < plan.medium_count)
    )
    parent_valid = (
        jnp.isfinite(steps.charge_numbers)
        & jnp.isfinite(steps.multiplicities)
        & (steps.multiplicities >= 0.0)
    )
    invalid = duplicate | jnp.any(steps.active & ~step_valid) | jnp.any(~parent_valid)
    emitting = steps.active & step_valid & (medium >= 0)
    nonoptical = jnp.sum(steps.active & (medium == -1), axis=1, dtype=jnp.int32)
    canonical = _CanonicalSteps(
        steps.start_positions[order],
        steps.end_positions[order],
        steps.start_times[order],
        steps.start_beta[order],
        steps.end_beta[order],
        jnp.linalg.norm(steps.end_positions - steps.start_positions, axis=-1)[order],
        jnp.clip(medium, 0, plan.medium_count - 1)[order],
        emitting[order],
        steps.charge_numbers[order],
        steps.multiplicities[order],
        steps.deposited_energy[order],
        steps.id_hi[order],
        steps.id_lo[order],
    )
    return canonical, order, invalid, nonoptical


def _generate(
    plan: OpticalPhotonSourcePlan,
    steps: _CanonicalSteps,
    root: PRNGKey,
    parents: Array,
    step_slots: Array,
    processes: Array,
    ordinals: Array,
) -> _Photon:
    """Photon kinematics of every bank slot, in bounded batches."""

    def one(item: tuple[Array, Array, Array, Array]) -> _Photon:
        parent, step, process, ordinal = item
        id_hi, id_lo = steps.id_hi[parent], steps.id_lo[parent]
        cherenkov = (
            None
            if plan.cherenkov is None
            else _cherenkov_photon(
                plan,
                steps,
                parent,
                step,
                jr.uniform(
                    derive_key(
                        root, _CHERENKOV_PHOTON_ADDRESS, step, id_hi, id_lo, ordinal
                    ),
                    (_UNIFORMS_PER_PHOTON,),
                    dtype=jnp.float64,
                ),
            )
        )
        scintillation = (
            None
            if plan.scintillation is None
            else _scintillation_photon(
                plan,
                steps,
                parent,
                step,
                jr.uniform(
                    derive_key(
                        root, _SCINTILLATION_PHOTON_ADDRESS, step, id_hi, id_lo, ordinal
                    ),
                    (_UNIFORMS_PER_PHOTON,),
                    dtype=jnp.float64,
                ),
            )
        )
        if cherenkov is None:
            if scintillation is None:
                raise RuntimeError("The plan has no emission process.")
            return scintillation
        if scintillation is None:
            return cherenkov
        return jax.tree_util.tree_map(
            lambda first, second: jnp.where(
                process == int(OpticalEmissionProcess.CHERENKOV), first, second
            ),
            cherenkov,
            scintillation,
        )

    return jax.lax.map(
        one, (parents, step_slots, processes, ordinals), batch_size=plan.batch_size
    )


def _fill_wavelength(plan: OpticalPhotonSourcePlan) -> Array:
    """A supported wavelength carried by empty slots."""

    if plan.cherenkov is not None:
        return plan.cherenkov.wavelengths[0]
    if plan.scintillation is not None:
        return plan.scintillation.emission_wavelengths[0]
    raise RuntimeError("The plan has no emission process.")


@eqx.filter_jit
def _emit(
    plan: OpticalPhotonSourcePlan,
    steps: ChargedOpticalSteps,
    root: PRNGKey,
    first: Array,
) -> OpticalPhotonEmission:
    canonical, order, invalid, nonoptical = _canonical_steps(plan, steps)
    parents, slots_per_parent = canonical.lengths.shape
    cherenkov_expected, scintillation_expected, visible = _expected_counts(
        plan, canonical
    )
    zero_counts = jnp.zeros((parents, slots_per_parent), dtype=jnp.int32)
    cherenkov_counts = (
        zero_counts
        if plan.cherenkov is None
        else _poisson_counts(
            root, canonical, cherenkov_expected, _CHERENKOV_COUNT_ADDRESS
        )
    )
    scintillation_counts = (
        zero_counts
        if plan.scintillation is None
        else _poisson_counts(
            root, canonical, scintillation_expected, _SCINTILLATION_COUNT_ADDRESS
        )
    )
    # Canonical order: parent identity, step slot, process, photon ordinal.
    flat = jnp.stack((cherenkov_counts, scintillation_counts), axis=-1).reshape(-1)
    cumulative = jnp.cumsum(flat.astype(jnp.int64))
    requested = cumulative[-1]
    capacity = plan.photon_capacity
    exhausted = requested.astype(jnp.uint64) > jnp.uint64(_RESERVED_IDENTITY) - first
    status = (
        jnp.where(requested > capacity, int(OpticalEmissionStatus.CAPACITY_EXHAUSTED), 0)
        | jnp.where(
            jnp.all(steps.complete), 0, int(OpticalEmissionStatus.INCOMPLETE_STEPS)
        )
        | jnp.where(invalid, int(OpticalEmissionStatus.INVALID_STEPS), 0)
        | jnp.where(exhausted, int(OpticalEmissionStatus.IDENTITY_EXHAUSTED), 0)
    ).astype(jnp.int32)
    slots = jnp.arange(capacity, dtype=jnp.int64)
    entry = jnp.clip(
        jnp.searchsorted(cumulative, slots, side="right"), 0, flat.shape[0] - 1
    )
    ordinals = slots - (cumulative[entry] - flat[entry])
    parent, remainder = jnp.divmod(entry, 2 * slots_per_parent)
    step, process = jnp.divmod(remainder, 2)
    photons = _generate(plan, canonical, root, parent, step, process, ordinals)
    candidate = (slots < requested) & (status == 0)
    finite = jnp.all(
        ~candidate
        | (
            jnp.all(jnp.isfinite(photons.position), axis=-1)
            & jnp.all(jnp.isfinite(photons.direction), axis=-1)
            & jnp.all(jnp.isfinite(photons.axis), axis=-1)
            & jnp.isfinite(photons.wavelength)
            & jnp.isfinite(photons.time)
            & jnp.isfinite(photons.energy)
        )
    )
    status = status | jnp.where(
        finite, 0, int(OpticalEmissionStatus.NONFINITE_RESULT)
    ).astype(jnp.int32)
    accepted = candidate & (status == 0)
    allocated = jnp.where(status == 0, requested, 0)
    identity = first + slots.astype(jnp.uint64)
    id_hi = jnp.where(
        accepted, (identity >> _WORD_BITS).astype(jnp.uint32), _RESERVED_WORD
    )
    id_lo = jnp.where(
        accepted,
        (identity & jnp.uint64(_WORD - 1)).astype(jnp.uint32),
        _RESERVED_WORD,
    )
    lane = accepted[:, None]
    state = OpticalPhotonState(
        jnp.where(lane, photons.position, 0.0),
        jnp.where(lane, photons.direction, jnp.asarray((0.0, 0.0, 1.0))),
        jnp.where(lane, photons.axis, jnp.asarray((1.0, 0.0, 0.0))),
        jnp.broadcast_to(jnp.asarray((1.0, 0.0), dtype=jnp.complex128), (capacity, 2)),
        jnp.where(accepted, photons.wavelength, _fill_wavelength(plan)),
        jnp.where(accepted, photons.time, 0.0),
        accepted.astype(jnp.float64),
        jnp.where(accepted, canonical.media[parent, step], 0).astype(jnp.int32),
        id_hi,
        id_lo,
    )
    energy = jnp.where(accepted, photons.energy, 0.0)
    is_cherenkov = process == int(OpticalEmissionProcess.CHERENKOV)
    per_parent = jnp.zeros((parents,), dtype=jnp.float64)
    scintillation = plan.scintillation
    scintillating = (
        jnp.zeros_like(canonical.emitting)
        if scintillation is None
        else canonical.emitting & (scintillation.photon_yields[canonical.media] > 0.0)
    )
    next_identity = first + allocated.astype(jnp.uint64)
    return OpticalPhotonEmission(
        state,
        accepted,
        jnp.where(accepted, process, -1).astype(jnp.int32),
        jnp.where(accepted, canonical.id_hi[parent], _RESERVED_WORD),
        jnp.where(accepted, canonical.id_lo[parent], _RESERVED_WORD),
        jnp.where(accepted, step, -1).astype(jnp.int32),
        _unpermute(jnp.sum(cherenkov_expected, axis=1), order),
        _unpermute(jnp.sum(cherenkov_counts, axis=1, dtype=jnp.int32), order),
        _unpermute(jnp.sum(scintillation_expected, axis=1), order),
        _unpermute(jnp.sum(scintillation_counts, axis=1, dtype=jnp.int32), order),
        _unpermute(
            jnp.sum(jnp.where(scintillating, canonical.deposits, 0.0), axis=1), order
        ),
        _unpermute(jnp.sum(jnp.where(scintillating, visible, 0.0), axis=1), order),
        _unpermute(
            per_parent.at[parent].add(jnp.where(is_cherenkov, energy, 0.0)), order
        ),
        _unpermute(
            per_parent.at[parent].add(jnp.where(is_cherenkov, 0.0, energy)), order
        ),
        nonoptical,
        requested,
        allocated,
        (next_identity >> _WORD_BITS).astype(jnp.uint32),
        (next_identity & jnp.uint64(_WORD - 1)).astype(jnp.uint32),
        status,
        status == 0,
        plan.plan_id,
    )


def emit_optical_photons(
    plan: OpticalPhotonSourcePlan,
    steps: ChargedOpticalSteps,
    key: PRNGKey,
    /,
    *,
    first_identity: tuple[int, int] = (0, 0),
) -> OpticalPhotonEmission:
    """Sample Cherenkov and scintillation photons of charged steps into one bank.

    Per step, photon counts are Poisson with the multiplicity-weighted
    expected yields. A Cherenkov photon's wavelength follows the step's
    Frank--Tamm spectrum, its emission point the exact speed law above
    threshold, and its direction the cone ``cos theta = 1 / (beta n(lambda))``
    at the local speed, with polarization along ``p x (p x v)``. A
    scintillation photon is emitted uniformly along the step after the
    sampled component's rise/decay delay. Photon times add the exact transit
    time to the emission point. Accepted photons take consecutive identities
    from ``first_identity`` in canonical order; any status flag refuses the
    whole bank. Emission is a discrete Monte Carlo event and makes no pathwise
    differentiability claim.
    """

    if not isinstance(plan, OpticalPhotonSourcePlan):
        raise TypeError("plan must be an OpticalPhotonSourcePlan.")
    if not isinstance(steps, ChargedOpticalSteps):
        raise TypeError("steps must be ChargedOpticalSteps.")
    if plan.scintillation is not None and not steps.deposit_recorded:
        raise ValueError("Scintillation needs steps with recorded deposited energy.")
    if steps.speed_of_light != plan.speed_of_light:
        raise ValueError("Steps and plan must share one exact speed of light.")
    hi, lo = first_identity
    if type(hi) is not int or type(lo) is not int:
        raise TypeError("first_identity must be a pair of int words.")
    if not (0 <= hi < _WORD and 0 <= lo < _WORD):
        raise ValueError("first_identity words must lie in [0, 2**32).")
    if hi * _WORD + lo >= _RESERVED_IDENTITY:
        raise ValueError("first_identity must not be the reserved identity.")
    root = parse(key, PRNGKey, "key")
    return _emit(plan, steps, root, jnp.asarray(hi * _WORD + lo, dtype=jnp.uint64))


__all__ = [
    "ChargedOpticalSteps",
    "CherenkovEmission",
    "OpticalEmissionProcess",
    "OpticalEmissionStatus",
    "OpticalPhotonEmission",
    "OpticalPhotonSourcePlan",
    "ScintillationEmission",
    "charged_steps_from_trajectory",
    "charged_steps_from_transport",
    "emit_optical_photons",
]
