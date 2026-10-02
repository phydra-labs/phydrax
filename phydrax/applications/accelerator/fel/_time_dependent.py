#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-dependent, period-averaged free-electron laser with slippage.

Model
-----
The beam is a chain of slices spaced ``Δζ`` apart (``"positive-late"``: larger
``ζ`` trails). Slice ``s`` carries its own macroparticles, loaded and evolved
with the time-independent (KMR) equations of :class:`FELPlan`, and represents
``N_s = I_s Δζ / (e c)`` electrons. The radiation field lives on a window of
``W = P + S`` slots: ``P`` field-only head-padding slots ahead of the beam and
one slot per beam slice. Radiation slips ahead of the electrons by
``λ/λ_u`` per unit length inside undulator modules (one wavelength per
period) and by ``1/(2γ_r²)`` in breaks, with ``2γ_r² = λ_u(1 + a_w²)/λ`` of the
first module, i.e. ``∂_z Ẽ = s ∂_ζ Ẽ + source``.

Integration per step ``Δz`` is the symmetric Strang sequence
``T(Δz/2) S(Δz/2) F(Δz) K(Δz) S(Δz/2) T(Δz/2)`` where ``T`` is the exact betatron
rotation and ``S`` the RK4 source kick of every slice with its own field slot
(both owned by :class:`FELPlan`), and the free-field operator ``F = D X``
combines diffraction ``D`` of every slot with the slippage translation ``X`` by
one step of slip. Consecutive kicks therefore see the field one step's slip
apart: a slip of at most one slot per undulator step visits every slice
(otherwise ``FELStatus.SLIPPAGE_UNRESOLVED``). ``X`` is exact and never
interpolates: the ``"commensurate"`` route requires every step's slip to be an
integer number of slots and rolls the window; the ``"spectral"`` route
multiplies the window's discrete Fourier transform by the phase ramp
``e^{2πikh/L}`` (exact for band-limited fields and for integer shifts). A
``"periodic"`` window models an infinitely long periodic beam; an ``"open"``
window lets radiation leave through its head (zero field enters at the tail),
and the spectral route pads it with a zero guard so nothing wraps around.
Energy removed through open boundaries is ledgered as ``exit_energy``. ``K`` is
the space-charge kick of :class:`FELSpaceCharge`, evaluated from the current
particles every step (it acts on particles only, so it commutes with ``F``).

Spectra follow the repository convention ``F(ω) = ∫ f(t) e^{+iωt} dt``: slot
``j`` passes the observer at ``t_j = ζ_j / c``, the harmonic-``h`` envelope
spectrum is ``F̃_h(Δω) = Δt Σ_j Ẽ_{h,j} e^{iΔω t_j}``, and the one-sided
spectral energy is ``dW/dω = (ε₀ c A / 4π) |F̃_h|²`` at ``ω = hω + Δω``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ...._validation import finite_real_scalar
from ....optics.wave import PulseEnvelopeField
from ....typing import (
    Bool,
    checked,
    Complex128,
    Float64,
    Identifier,
    Int32,
    parse,
    PRNGKey,
    Scalar,
    Size,
)
from ._averaged import (
    _Carry,
    _NOISE_LINEARITY_LIMIT,
    _SliceInput,
    _StepRow,
    FELPlan,
    FELStatus,
)
from ._lattice import undulator_rms_strength_squared
from ._prebunching import FELPrebunching
from ._slices import FELBeamSlices, FELParticles, load_slice
from ._space_charge import (
    bunch_kick,
    FELSpaceCharge,
    radial_harmonic_rates,
    uniform_harmonic_rates,
)
from ._theory import fel_scaling_estimate, FELScalingEstimate
from ._types import (
    FELFrequencyDim,
    FELHarmonicDim,
    FELRecordDim,
    FELSliceDim,
    FELWindowDim,
)


FELSlippageRoute: TypeAlias = Literal["commensurate", "spectral"]
FELWindowBoundary: TypeAlias = Literal["periodic", "open"]

# Half-step slips of the commensurate route must be integers to this many slots.
_COMMENSURATE_TOLERANCE = 1.0e-6
# Pulse-seed samples must land on window slots to this many slots.
_SEED_ALIGNMENT_TOLERANCE = 1.0e-6
# Seed pulse time spacing c Δt must equal Δζ to this relative tolerance.
_SEED_SPACING_TOLERANCE = 1.0e-9


class FELPulseSeed(StrictModule, NonTrainableState):
    """Seed pulse from an optics :class:`PulseEnvelopeField` at the lattice entrance.

    The envelope's pulse time ``t`` reaches the slice coordinate
    ``ζ = reference_position + c t`` (later times trail). Its samples must be
    spaced ``c Δt = Δζ`` and land exactly on window slots (checked at solve
    time); the carrier ``ω′`` may differ from the FEL fundamental ``ω`` and is
    carried exactly by the phase ramp ``e^{−i(ω′−ω)t}``. The one-dimensional
    model accepts only transversely uniform (plane-wave) envelopes; the grid
    model requires the plan's ``PlaneFieldSpace``. Only scalar envelopes are
    accepted: the averaged field has one polarization state.
    """

    envelope: PulseEnvelopeField
    reference_position: float = eqx.field(static=True)
    time_spacing: float = eqx.field(static=True)
    carrier_angular_frequency: float = eqx.field(static=True)
    uniform: bool = eqx.field(static=True)
    seed_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, envelope: PulseEnvelopeField, /, *, reference_position: float = 0.0
    ) -> None:
        if envelope.polarization != "scalar":
            raise ValueError("An FEL pulse seed must be a scalar envelope.")
        values = np.asarray(envelope.values)
        self.envelope = envelope
        self.reference_position = finite_real_scalar(
            reference_position, "reference_position"
        )
        self.time_spacing = envelope.time_space.sample_spacing
        self.carrier_angular_frequency = float(envelope.carrier_angular_frequency)
        self.uniform = bool(
            np.array_equal(values, np.broadcast_to(values[:1, :1], values.shape))
        )
        self.seed_id = canonical_fingerprint(
            {
                "kind": "fel-pulse-seed",
                "plane_space": envelope.plane_space.space_id,
                "time_space": envelope.time_space.space_id,
                "values": array_tree_fingerprint(values),
                "carrier": self.carrier_angular_frequency,
                "reference_position": self.reference_position,
            }
        )


class FELSpectrum(StrictModule):
    """Exit spectrum and single-shot coherence of a time-dependent window.

    ``spectral_energy[f, h]`` is the one-sided ``dW/dω`` at
    ``angular_frequencies[f, h] = hω + Δω_f``. ``spike_count`` counts local
    maxima of the exit power profile above ``spike_threshold`` times its peak;
    ``coherence_time`` is ``τ_c = ∫|g₁(τ)|² dτ`` of the exit field
    (circular for periodic windows, zero-padded for open ones) and
    ``rms_bandwidth`` the rms angular-frequency width. Both are NaN for a
    vanishing field.
    """

    __strict_contract__ = True

    angular_frequencies: Float64[FELFrequencyDim, FELHarmonicDim]
    spectral_energy: Float64[FELFrequencyDim, FELHarmonicDim]
    spike_count: Int32[FELHarmonicDim]
    coherence_time: Float64[FELHarmonicDim]
    rms_bandwidth: Float64[FELHarmonicDim]


class FELTimeDependentLedger(StrictModule):
    """Whole-window energy ledger over the lattice (scale energy units).

    ``defect = beam_energy_change + field_energy_change + diffraction_loss +
    exit_energy + wake_loss + space_charge_loss`` vanishes for an exact
    integration; ``relative_defect`` divides it by the total exchanged energy.
    ``space_charge_loss`` is minus the kinetic energy space charge gave the
    beam (its electrostatic field energy is not tracked; the internal field
    exerts no net force, so the sum stays small).
    """

    __strict_contract__ = True

    beam_energy_change: Float64[Scalar]
    field_energy_change: Float64[Scalar]
    diffraction_loss: Float64[Scalar]
    exit_energy: Float64[Scalar]
    wake_loss: Float64[Scalar]
    space_charge_loss: Float64[Scalar]
    defect: Float64[Scalar]
    relative_defect: Float64[Scalar]


class FELTimeDependentEvidence(StrictModule):
    """Status, slippage, padding, seed, and collective evidence of one run.

    ``status`` is the union of every ``slice_status`` and the window-level
    bits. ``padding_sufficient`` holds when the head padding is at least the
    total slippage; ``exit_fraction`` is the share of radiated energy that left
    through open window boundaries. ``seed_alignment_residual`` is the largest
    distance (slots) of a pulse-seed sample from its slot and
    ``seed_truncated_energy`` the seed energy outside the window.
    ``space_charge_accepted`` holds when every step's space-charge evaluation
    succeeded; ``space_charge_cells_per_sigma`` is the smallest rest-frame
    rms bunch size per X3 grid axis in cells over the run, and
    ``transverse_space_charge_ratio`` the accumulated rms X3 transverse kick
    per slice relative to its entrance rms ``u⊥`` (both NaN without the
    bunch-scale plan).
    """

    __strict_contract__ = True

    status: Int32[Scalar]
    slice_status: Int32[FELSliceDim]
    finite: Bool[Scalar]
    maximum_phase_step: Float64[FELSliceDim]
    lost_particles: Int32[FELSliceDim]
    diffraction_accepted: Bool[Scalar]
    electrons_per_slice: Float64[FELSliceDim]
    shot_noise_amplitude: Float64[FELSliceDim]
    maximum_prebunching_displacement: Float64[FELSliceDim]
    total_slippage: Float64[Scalar]
    slippage_slots: Float64[Scalar]
    maximum_step_slip_slots: Float64[Scalar]
    head_padding_length: Float64[Scalar]
    padding_sufficient: Bool[Scalar]
    exit_fraction: Float64[Scalar]
    seed_alignment_residual: Float64[Scalar]
    seed_truncated_energy: Float64[Scalar]
    space_charge_accepted: Bool[Scalar]
    space_charge_cells_per_sigma: Float64[Literal[3]]
    transverse_space_charge_ratio: Float64[FELSliceDim]

    @property
    def successful(self) -> Array:
        return self.status == int(FELStatus.SUCCESS)


class FELTimeDependentResult(StrictModule):
    """Time-dependent averaged-FEL solution.

    ``power[z, j, h]`` is the temporal power profile on the window slots at
    ``window_positions`` (head padding first) for every record, and
    ``pulse_energy[z, h]`` its integral ``Σ_j P Δζ/c``. ``bunching[z, s, h]``,
    ``beam_power``, ``mean_lorentz_factor``, and ``relative_energy_spread`` are
    per beam slice. ``exit_field`` is ``[j, h]`` (one-dimensional) or
    ``[j, n₀, n₁, h]`` (grid). ``wake_rate`` is the constant per-slice wake
    ``dγ/dz`` (loss) and ``space_charge_gain[z, s]`` the slice-mean Lorentz
    factor space charge has added up to each record.
    ``initial_particles`` are the radiator-entrance particles (after
    prebunching).
    """

    __strict_contract__ = True

    positions: Float64[FELRecordDim]
    window_positions: Float64[FELWindowDim]
    angular_frequencies: Float64[FELHarmonicDim]
    power: Float64[FELRecordDim, FELWindowDim, FELHarmonicDim]
    pulse_energy: Float64[FELRecordDim, FELHarmonicDim]
    bunching: Complex128[FELRecordDim, FELSliceDim, FELHarmonicDim]
    beam_power: Float64[FELRecordDim, FELSliceDim]
    mean_lorentz_factor: Float64[FELRecordDim, FELSliceDim]
    relative_energy_spread: Float64[FELRecordDim, FELSliceDim]
    wake_rate: Float64[FELSliceDim]
    space_charge_gain: Float64[FELRecordDim, FELSliceDim]
    exit_field: Array
    spectrum: FELSpectrum
    initial_particles: FELParticles
    particles: FELParticles
    scaling: FELScalingEstimate
    ledger: FELTimeDependentLedger
    evidence: FELTimeDependentEvidence
    harmonics: tuple[int, ...] = eqx.field(static=True)
    record_count: Size[FELRecordDim] = eqx.field(static=True)
    window_count: Size[FELWindowDim] = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)


class _WindowRow(NamedTuple):
    step: _StepRow
    shift: Array


class _WindowCarry(NamedTuple):
    phases: Array
    lorentz_factors: Array
    positions: Array
    momenta: Array
    field: Array
    diffraction_loss: Array
    exit_energy: Array
    accepted: Array
    lost: Array
    phase_step: Array
    space_charge_gain: Array
    transverse_kick: Array
    space_charge_accepted: Array
    cells_per_sigma: Array


class _WindowRecord(NamedTuple):
    power: Array
    bunching: Array
    beam_energy_sum: Array
    mean_lorentz_factor: Array
    relative_energy_spread: Array
    space_charge_gain: Array


class _CollectiveInput(NamedTuple):
    """Per-slice constants of the space-charge kick."""

    positions: Array
    electrons: Array
    source: Array


class _Loaded(NamedTuple):
    phases: Array
    lorentz_factors: Array
    positions: Array
    momenta: Array
    noise_amplitude: Array
    displacement: Array


class _Geometry(NamedTuple):
    """Static window geometry of one solve."""

    spacing: float
    slices: int
    window: int
    shifts: np.ndarray
    guard: int


def _nonnegative_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _unit_fraction(value: float, name: str, /) -> float:
    fraction = finite_real_scalar(value, name)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"{name} must lie in (0, 1).")
    return fraction


class FELTimeDependentPlan(StrictModule, NonTrainableState):
    """Time-dependent averaged FEL: coupled slices with slippage.

    ``core`` supplies the lattice, wavelength, loading, harmonics, transverse
    model, continuous-wave seed, and per-slice wakes; its ``slice_batch``
    bounds the slices or slots processed concurrently. ``slippage`` selects the
    exact translation route and ``boundary`` the window topology;
    ``head_padding`` adds field-only slots ahead of the beam. An open window
    refuses the continuous-wave ``FELSeed`` (seed it with ``pulse_seed``).
    ``prebunching`` runs HGHG/EEHG stages before the radiator (loading needs
    ``particles_per_beamlet ≥ 2 h max(harmonics)``); ``space_charge`` applies
    intra-slice harmonic and X3 bunch-scale space charge evaluated every step.
    ``window_tolerance`` bounds the exit-energy fraction of an open window and
    ``spike_threshold`` the relative height of counted power spikes.
    """

    core: FELPlan
    pulse_seed: FELPulseSeed | None
    prebunching: FELPrebunching | None
    space_charge: FELSpaceCharge | None
    slippage_lengths: tuple[float, ...] = eqx.field(static=True)
    maximum_coupled_slippage: float = eqx.field(static=True)
    slippage: FELSlippageRoute = eqx.field(static=True)
    boundary: FELWindowBoundary = eqx.field(static=True)
    head_padding: int = eqx.field(static=True)
    reference_lorentz_factor: float = eqx.field(static=True)
    window_tolerance: float = eqx.field(static=True)
    spike_threshold: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        core: FELPlan,
        /,
        *,
        slippage: FELSlippageRoute,
        boundary: FELWindowBoundary,
        head_padding: int = 0,
        pulse_seed: FELPulseSeed | None = None,
        prebunching: FELPrebunching | None = None,
        space_charge: FELSpaceCharge | None = None,
        window_tolerance: float = 1.0e-3,
        spike_threshold: float = 0.1,
    ) -> None:
        if pulse_seed is not None and not isinstance(pulse_seed, FELPulseSeed):
            raise TypeError("pulse_seed must be FELPulseSeed or None.")
        if prebunching is not None and not isinstance(prebunching, FELPrebunching):
            raise TypeError("prebunching must be FELPrebunching or None.")
        if space_charge is not None and not isinstance(space_charge, FELSpaceCharge):
            raise TypeError("space_charge must be FELSpaceCharge or None.")
        route = parse(slippage, FELSlippageRoute, "slippage")
        boundary_ = parse(boundary, FELWindowBoundary, "boundary")
        padding = _nonnegative_integer(head_padding, "head_padding")
        if boundary_ == "open" and core.seed is not None:
            raise ValueError(
                "An open window refuses the continuous-wave FELSeed; seed it with an "
                "FELPulseSeed."
            )
        if pulse_seed is not None:
            _validate_pulse_seed(core, pulse_seed)
        if prebunching is not None:
            needed = 2 * prebunching.harmonic * core.harmonics[-1]
            if core.loading.particles_per_beamlet < needed:
                raise ValueError(
                    "Prebunched loading needs particles_per_beamlet >= "
                    f"2 * prebunching.harmonic * max(harmonics) (= {needed})."
                )
        if space_charge is not None:
            _validate_space_charge(core, space_charge, boundary_)
        lattice = core.lattice
        first = lattice.segments[0].device
        strength = undulator_rms_strength_squared(
            lattice.deflections[0], lattice.polarization
        )
        # 2γ_r² = λ_u (1 + a_w²) / λ: the first module is resonant at λ.
        twice_gamma_squared = first.period * (1.0 + strength) / core.wavelength
        table = lattice.step_table()
        slips = np.where(
            table.undulator_wavenumbers > 0.0,
            table.lengths
            * table.undulator_wavenumbers
            * core.wavelength
            / (2.0 * math.pi),
            table.lengths / twice_gamma_squared,
        )
        tolerance = _unit_fraction(window_tolerance, "window_tolerance")
        threshold = _unit_fraction(spike_threshold, "spike_threshold")
        self.core = core
        self.pulse_seed = pulse_seed
        self.prebunching = prebunching
        self.space_charge = space_charge
        self.slippage_lengths = tuple(float(value) for value in slips)
        # Breaks (no coupling) may slip past slots without skipping interactions.
        self.maximum_coupled_slippage = float(
            np.max(np.where(table.undulator_wavenumbers > 0.0, slips, 0.0))
        )
        self.slippage = route
        self.boundary = boundary_
        self.head_padding = padding
        self.reference_lorentz_factor = math.sqrt(0.5 * twice_gamma_squared)
        self.window_tolerance = tolerance
        self.spike_threshold = threshold
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fel-time-dependent-plan",
                "core": core.plan_id,
                "slippage": route,
                "boundary": boundary_,
                "head_padding": padding,
                "pulse_seed": None if pulse_seed is None else pulse_seed.seed_id,
                "prebunching": None
                if prebunching is None
                else prebunching.prebunching_id,
                "space_charge": None
                if space_charge is None
                else space_charge.space_charge_id,
                "window_tolerance": tolerance,
                "spike_threshold": threshold,
            }
        )

    # ------------------------------------------------------------- geometry

    @property
    def total_slippage(self) -> float:
        """Radiation slippage over the whole lattice (scale length)."""
        return sum(self.slippage_lengths)

    def _geometry(self, slices: FELBeamSlices, /) -> _Geometry:
        spacing = slices.position_spacing
        if spacing is None:
            raise ValueError(
                "Time-dependent FEL slices need at least two uniformly spaced positions."
            )
        slots = np.asarray(self.slippage_lengths, dtype=np.float64) / spacing
        match self.slippage:
            case "commensurate":
                whole = np.rint(slots)
                if np.any(np.abs(slots - whole) > _COMMENSURATE_TOLERANCE):
                    worst = float(np.max(np.abs(slots - whole)))
                    raise ValueError(
                        "Commensurate slippage needs every step's slip to be an "
                        f"integer number of slice spacings (worst residual {worst:.3g} "
                        "slots); adjust step_length, drift lengths, or slice spacing, "
                        "or use the spectral route."
                    )
                shifts = whole.astype(np.int32)
            case "spectral":
                shifts = slots
            case _:
                assert_never(self.slippage)
        count = slices.slice_count
        window = self.head_padding + count
        if window < 2:
            raise ValueError("The radiation window needs at least two slots.")
        guard = 0
        if self.boundary == "open" and self.slippage == "spectral":
            guard = math.ceil(float(np.max(slots))) + 1
        if self.pulse_seed is not None:
            light = float(self.core.lattice.scale.speed_of_light)
            seed = self.pulse_seed
            if abs(seed.time_spacing * light / spacing - 1.0) > _SEED_SPACING_TOLERANCE:
                raise ValueError(
                    "The pulse seed's time spacing times c must equal the slice spacing."
                )
            detuning = seed.carrier_angular_frequency - self._fundamental_frequency
            if abs(detuning) * seed.time_spacing >= math.pi:
                raise ValueError(
                    "The pulse-seed carrier detuning exceeds the window's Nyquist limit."
                )
        if self.space_charge is not None and self.space_charge.bunch is not None:
            capacity = count * self.core.loading.particle_count
            if self.space_charge.bunch.capacity != capacity:
                raise ValueError(
                    "The bunch-scale space-charge plan capacity must equal slices × "
                    f"particles per slice (= {capacity})."
                )
        return _Geometry(spacing, count, window, shifts, guard)

    @property
    def _fundamental_frequency(self) -> float:
        light = float(self.core.lattice.scale.speed_of_light)
        return 2.0 * math.pi * light / self.core.wavelength

    @property
    def _power_prefactor(self) -> float:
        scale = self.core.lattice.scale
        return 0.5 * float(scale.vacuum_permittivity) * float(scale.speed_of_light)

    def _map[T, R](self, function: Callable[[T], R], values: T, /) -> R:
        """Bounded map over leading slice/slot axes (``core.slice_batch`` lanes)."""
        return jax.lax.map(function, values, batch_size=self.core.slice_batch)

    # ------------------------------------------------------------- field ops

    def _slot_power(self, field: Array, area: Array, /) -> Array:
        """Power per slot and harmonic ``[W, H]``."""
        intensity = jnp.real(field * jnp.conj(field))
        match self.core.transverse:
            case "one-dimensional":
                return self._power_prefactor * area * intensity
            case "angular-spectrum":
                return (
                    self._power_prefactor
                    * self.core.cell_area
                    * jnp.sum(intensity, axis=(1, 2))
                )
            case _:
                assert_never(self.core.transverse)

    def _diffract(self, field: Array, distance: Array, /) -> tuple[Array, Array, Array]:
        match self.core.transverse:
            case "one-dimensional":
                return field, jnp.asarray(0.0), jnp.asarray(True)
            case "angular-spectrum":

                def one(slot: Array) -> tuple[Array, Array, Array]:
                    return self.core._diffract(slot, distance)

                fields, losses, accepted = self._map(one, field)
                return fields, jnp.sum(losses), jnp.all(accepted)
            case _:
                assert_never(self.core.transverse)

    def _translate(
        self, field: Array, shift: Array, geometry: _Geometry, area: Array, /
    ) -> tuple[Array, Array]:
        """Slip the window toward its head by ``shift`` slots; return exit power sum."""
        window = geometry.window
        slots = jnp.arange(window)
        match self.slippage:
            case "commensurate":
                moved = jnp.roll(field, -shift, axis=0)
                match self.boundary:
                    case "periodic":
                        return moved, jnp.asarray(0.0)
                    case "open":
                        leaving = slots < shift
                        exit_power = jnp.sum(
                            jnp.where(
                                leaving[:, None], self._slot_power(field, area), 0.0
                            )
                        )
                        entering = slots >= window - shift
                        mask = entering.reshape((window,) + (1,) * (field.ndim - 1))
                        return jnp.where(mask, 0.0, moved), exit_power
                    case _:
                        assert_never(self.boundary)
            case "spectral":
                length = window + geometry.guard
                padded = jnp.concatenate(
                    (
                        field,
                        jnp.zeros((geometry.guard,) + field.shape[1:], field.dtype),
                    ),
                    axis=0,
                )
                modes = jnp.fft.fftfreq(length) * length
                ramp = jnp.exp(2j * jnp.pi * modes * shift / length)
                spectrum = jnp.fft.fft(padded, axis=0)
                ramp = ramp.reshape((length,) + (1,) * (field.ndim - 1))
                moved = jnp.fft.ifft(spectrum * ramp, axis=0)
                kept = moved[:window]
                if geometry.guard == 0:
                    return kept, jnp.asarray(0.0)
                return kept, jnp.sum(self._slot_power(moved[window:], area))
            case _:
                assert_never(self.slippage)

    def _free(
        self,
        field: Array,
        distance: Array,
        shift: Array,
        geometry: _Geometry,
        area: Array,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        """``F = D X`` over half a step: field, diffraction and exit power, flag."""
        diffracted, loss, accepted = self._diffract(field, distance)
        moved, exit_power = self._translate(diffracted, shift, geometry, area)
        return moved, loss, exit_power, accepted

    # ----------------------------------------------------------- beam setup

    def _load(self, key: PRNGKey, slices: FELBeamSlices, electrons: Array, /) -> _Loaded:
        loading = self.core.loading
        base = 2.0 * math.pi / self.core.wavelength
        if self.prebunching is not None:
            base = base / self.prebunching.harmonic
        prebunching = self.prebunching

        def one(
            row: tuple[Array, Array, Array, Array, Array, Array, Array, Array],
        ) -> _Loaded:
            identity, gamma, spread, emittance, beta, alpha, count, position = row
            loaded = load_slice(
                loading,
                key,
                identity,
                gamma,
                spread,
                emittance,
                beta,
                alpha,
                count / loading.beamlet_count,
            )
            if prebunching is None:
                return _Loaded(
                    loaded.phases,
                    loaded.lorentz_factors,
                    loaded.positions,
                    loaded.transverse_momenta,
                    loaded.noise_amplitude,
                    jnp.asarray(0.0),
                )
            bunched = prebunching.apply(
                loaded.phases,
                loaded.lorentz_factors,
                loaded.positions,
                loaded.transverse_momenta,
                position,
                base,
            )
            return _Loaded(
                bunched.phases,
                bunched.lorentz_factors,
                bunched.positions,
                bunched.transverse_momenta,
                loaded.noise_amplitude,
                bunched.maximum_displacement,
            )

        return self._map(
            one,
            (
                slices.identities,
                slices.lorentz_factors,
                slices.relative_energy_spreads,
                slices.normalized_emittances,
                slices.beta_functions,
                slices.alpha_functions,
                electrons,
                slices.positions,
            ),
        )

    def _space_charge_kick(
        self,
        config: FELSpaceCharge,
        phases: Array,
        lorentz_factors: Array,
        positions: Array,
        momenta: Array,
        step: _StepRow,
        collective: _CollectiveInput,
        /,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """``K(Δz)`` from the current particles.

        Returns the kicked ``γ`` and ``u⊥``, the slice-mean ``Δγ``, the slice rms
        of the X3 transverse kick (applied or not), the acceptance flag, and
        the X3 cells per σ (NaN without the bunch-scale plan).
        """
        length = step.length
        # Mean-motion frame of the step: γ_z² = γ_r²/(1 + a_w²).
        frame_squared = self.reference_lorentz_factor**2 / (
            1.0 + step.rms_strength_squared
        )
        change = jnp.zeros_like(lorentz_factors)
        transverse = jnp.zeros_like(momenta)
        accepted = jnp.asarray(True)
        cells = jnp.full((3,), jnp.nan)
        if config.harmonics > 0:
            orders = jnp.arange(1, config.harmonics + 1, dtype=jnp.float64)
            wavenumber = self.core._wavenumber
            match self.core.transverse:
                case "one-dimensional":
                    rates, solved = uniform_harmonic_rates(
                        phases,
                        positions,
                        collective.source,
                        orders,
                        wavenumber,
                        frame_squared,
                    )
                case "angular-spectrum":
                    rates, solved = radial_harmonic_rates(
                        phases,
                        positions,
                        collective.source,
                        orders,
                        wavenumber,
                        frame_squared,
                        config,
                    )
                case _:
                    assert_never(self.core.transverse)
            change = change + jnp.where(solved, length * rates, 0.0)
            accepted = accepted & solved
        if config.bunch is not None:
            scale = self.core.lattice.scale
            energy, kick, bunch_accepted, cells = bunch_kick(
                config.bunch,
                lorentz_factors,
                positions,
                momenta,
                collective.positions,
                collective.electrons,
                step.rms_strength_squared,
                self.reference_lorentz_factor,
                float(scale.electron_mass) * float(scale.speed_of_light) ** 2,
                length,
            )
            change = change + jnp.where(bunch_accepted, energy, 0.0)
            transverse = jnp.where(bunch_accepted, kick, 0.0)
            accepted = accepted & bunch_accepted
        rms = jnp.sqrt(jnp.mean(jnp.sum(transverse * transverse, axis=-1), axis=1))
        match config.transverse:
            case "applied":
                momenta = momenta + transverse
            case "omitted":
                pass
            case _:
                assert_never(config.transverse)
        return (
            lorentz_factors + change,
            momenta,
            jnp.mean(change, axis=1),
            rms,
            accepted,
            cells,
        )

    # ------------------------------------------------------------- seeding

    def _initial_field(
        self, slices: FELBeamSlices, geometry: _Geometry, area: Array, /
    ) -> tuple[Array, Array, Array]:
        """Initial window field, seed alignment residual, truncated seed energy."""
        core = self.core
        window = geometry.window
        harmonics = len(core.harmonics)
        match core.transverse:
            case "one-dimensional":
                layout: tuple[int, ...] = (harmonics,)
            case "angular-spectrum":
                if core.field_space is None:
                    raise RuntimeError("The grid model lost its field space.")
                layout = core.field_space.shape + (harmonics,)
            case _:
                assert_never(core.transverse)
        field = jnp.zeros((window,) + layout, dtype=jnp.complex128)
        if core.seed is not None:
            field = field + core._seed_field(area)[None]
        residual = jnp.asarray(0.0)
        truncated = jnp.asarray(0.0)
        if self.pulse_seed is None:
            return field, residual, truncated
        seed = self.pulse_seed
        light = float(core.lattice.scale.speed_of_light)
        times = seed.envelope.time_space.coordinates
        values = seed.envelope.values
        match core.transverse:
            case "one-dimensional":
                samples = values[0, 0, :]
            case "angular-spectrum":
                samples = jnp.moveaxis(values, -1, 0)
            case _:
                assert_never(core.transverse)
        detuning = seed.carrier_angular_frequency - self._fundamental_frequency
        ramp = jnp.exp(-1j * detuning * times)
        samples = samples * ramp.reshape((-1,) + (1,) * (samples.ndim - 1))
        offset = (
            self.head_padding
            + (seed.reference_position + light * times - slices.positions[0])
            / geometry.spacing
        )
        index = jnp.rint(offset)
        residual = jnp.max(jnp.abs(offset - index))
        inside = (index >= 0) & (index < window)
        slot = jnp.where(inside, index, window).astype(jnp.int32)
        field = field.at[slot, ..., 0].add(samples, mode="drop")
        intensity = jnp.real(samples * jnp.conj(samples))
        match core.transverse:
            case "one-dimensional":
                sample_power = self._power_prefactor * area * intensity
            case "angular-spectrum":
                sample_power = (
                    self._power_prefactor
                    * core.cell_area
                    * jnp.sum(intensity, axis=(1, 2))
                )
            case _:
                assert_never(core.transverse)
        truncated = jnp.sum(jnp.where(inside, 0.0, sample_power)) * seed.time_spacing
        return field, residual, truncated

    # ------------------------------------------------------------ stepping

    def _record(self, carry: _WindowCarry, area: Array, /) -> _WindowRecord:
        gamma = carry.lorentz_factors
        mean = jnp.mean(gamma, axis=1)
        orders = self.core._harmonic_orders
        bunching = jnp.mean(
            jnp.exp(-1j * orders[None, None, :] * carry.phases[:, :, None]), axis=1
        )
        return _WindowRecord(
            power=self._slot_power(carry.field, area),
            bunching=bunching,
            beam_energy_sum=jnp.sum(gamma, axis=1),
            mean_lorentz_factor=mean,
            relative_energy_spread=jnp.std(gamma, axis=1) / mean,
            space_charge_gain=carry.space_charge_gain,
        )

    def _collective(
        self,
        carry: _WindowCarry,
        state: tuple[Array, Array, Array, Array],
        step: _StepRow,
        collective: _CollectiveInput,
        /,
    ) -> tuple[Array, Array, _WindowCarry]:
        """Apply ``K(Δz)`` to ``(θ, γ, x⊥, u⊥)``; accumulate its evidence in ``carry``."""
        phases, gamma, positions, momenta = state
        if self.space_charge is None:
            return gamma, momenta, carry
        gamma, momenta, gain, rms, accepted, cells = self._space_charge_kick(
            self.space_charge, phases, gamma, positions, momenta, step, collective
        )
        return (
            gamma,
            momenta,
            carry._replace(
                space_charge_gain=carry.space_charge_gain + gain,
                transverse_kick=carry.transverse_kick + rms,
                space_charge_accepted=carry.space_charge_accepted & accepted,
                cells_per_sigma=jnp.minimum(carry.cells_per_sigma, cells),
            ),
        )

    def _step(
        self,
        inputs: _SliceInput,
        collective: _CollectiveInput,
        geometry: _Geometry,
        carry: _WindowCarry,
        row: _WindowRow,
        /,
    ) -> tuple[_WindowCarry, _WindowRecord]:
        core = self.core
        step = row.step
        half = 0.5 * step.length
        area = inputs.area[0]
        momenta = carry.momenta + step.quadrupole_kick * carry.positions * jnp.asarray(
            (-1.0, 1.0)
        )
        phases = carry.phases + step.phase_shift

        def betatron(
            positions: Array, momenta: Array, gamma: Array
        ) -> tuple[Array, Array]:
            return core._betatron(positions, momenta, gamma, step, half)

        positions, momenta = jax.vmap(betatron)(
            carry.positions, momenta, carry.lorentz_factors
        )
        padding = self.head_padding
        half_step = step._replace(length=half)

        def kick(
            values: tuple[Array, Array, Array, Array, Array, _SliceInput],
        ) -> tuple[Array, Array, Array, Array, Array]:
            slice_phases, slice_gamma, slice_positions, slice_momenta, slot, row_input = (
                values
            )
            state, lost = core._bind(slice_positions)
            staged = _Carry(
                phases=slice_phases,
                lorentz_factors=slice_gamma,
                positions=slice_positions,
                momenta=slice_momenta,
                field=slot,
                diffraction_loss=jnp.asarray(0.0),
                accepted=jnp.asarray(True),
                lost=jnp.asarray(0, dtype=jnp.int32),
                phase_step=jnp.asarray(0.0),
            )
            updated_phases, updated_gamma, updated_slot, phase_step = core._source(
                staged, half_step, state, row_input
            )
            return updated_phases, updated_gamma, updated_slot, lost, phase_step

        def source(
            field: Array, phases: Array, gamma: Array, momenta: Array
        ) -> tuple[Array, Array, Array, Array, Array]:
            phases, gamma, beam_field, lost, phase_step = self._map(
                kick, (phases, gamma, positions, momenta, field[padding:], inputs)
            )
            return field.at[padding:].set(beam_field), phases, gamma, lost, phase_step

        # S(Δz/2) F(Δz) K(Δz) S(Δz/2): consecutive kicks see the field one
        # full-step slip apart, so a slip of at most one slot per step visits
        # every slice; K acts on particles only and commutes with F.
        field, phases, gamma, lost, first_phase = source(
            carry.field, phases, carry.lorentz_factors, momenta
        )
        field, loss, exit_power, accepted = self._free(
            field, step.length, row.shift, geometry, area
        )
        gamma, momenta, collected = self._collective(
            carry, (phases, gamma, positions, momenta), step, collective
        )
        field, phases, gamma, _, second_phase = source(field, phases, gamma, momenta)
        positions, momenta = jax.vmap(betatron)(positions, momenta, gamma)
        slot_time = geometry.spacing / float(core.lattice.scale.speed_of_light)
        # The half-kick phase advances are reported per full step.
        phase_step = 2.0 * jnp.maximum(first_phase, second_phase)
        updated = collected._replace(
            phases=phases,
            lorentz_factors=gamma,
            positions=positions,
            momenta=momenta,
            field=field,
            diffraction_loss=carry.diffraction_loss + loss * slot_time,
            exit_energy=carry.exit_energy + exit_power * slot_time,
            accepted=carry.accepted & accepted,
            lost=jnp.maximum(carry.lost, lost),
            phase_step=jnp.maximum(carry.phase_step, phase_step),
        )
        return updated, self._record(updated, area)

    # ------------------------------------------------------------ outputs

    def _spectrum(
        self, exit_field: Array, power: Array, geometry: _Geometry, area: Array, /
    ) -> FELSpectrum:
        core = self.core
        window = geometry.window
        light = float(core.lattice.scale.speed_of_light)
        slot_time = geometry.spacing / light
        # F̃(Δω_k) = Δt Σ_j Ẽ_j e^{+2πijk/W} = Δt W ifft(Ẽ)_k.
        transformed = jnp.fft.fftshift(
            jnp.fft.ifft(exit_field, axis=0) * (window * slot_time), axes=0
        )
        intensity = jnp.real(transformed * jnp.conj(transformed))
        match core.transverse:
            case "one-dimensional":
                spectral = intensity * area
            case "angular-spectrum":
                spectral = core.cell_area * jnp.sum(intensity, axis=(1, 2))
            case _:
                assert_never(core.transverse)
        permittivity = float(core.lattice.scale.vacuum_permittivity)
        spectral_energy = permittivity * light / (4.0 * math.pi) * spectral
        offsets = (
            2.0
            * math.pi
            * jnp.fft.fftshift(jnp.fft.fftfreq(window, d=slot_time))[:, None]
        )
        frequencies = self._fundamental_frequency * core._harmonic_orders[None, :]
        total = jnp.sum(spectral_energy, axis=0)
        present = total > 0.0
        safe_total = jnp.where(present, total, 1.0)
        mean = jnp.sum(spectral_energy * offsets, axis=0) / safe_total
        variance = (
            jnp.sum(spectral_energy * (offsets - mean[None, :]) ** 2, axis=0) / safe_total
        )
        # τ_c = Δt L Σ S² / (Σ S)² with S the (zero-padded for open windows) spectrum.
        length = window if self.boundary == "periodic" else 2 * window
        padded = jnp.fft.fft(exit_field, n=length, axis=0)
        density = jnp.real(padded * jnp.conj(padded))
        if density.ndim == 4:
            density = jnp.sum(density, axis=(1, 2))
        density_total = jnp.sum(density, axis=0)
        coherence = (
            slot_time
            * length
            * jnp.sum(density * density, axis=0)
            / jnp.where(density_total > 0.0, density_total, 1.0) ** 2
        )
        match self.boundary:
            case "periodic":
                previous = jnp.roll(power, 1, axis=0)
                following = jnp.roll(power, -1, axis=0)
            case "open":
                zero = jnp.zeros_like(power[:1])
                previous = jnp.concatenate((zero, power[:-1]), axis=0)
                following = jnp.concatenate((power[1:], zero), axis=0)
            case _:
                assert_never(self.boundary)
        peak = jnp.max(power, axis=0)
        spikes = (
            (power > previous)
            & (power >= following)
            & (power >= self.spike_threshold * peak[None, :])
            & (peak[None, :] > 0.0)
        )
        return FELSpectrum(
            angular_frequencies=frequencies + offsets,
            spectral_energy=spectral_energy,
            spike_count=jnp.sum(spikes, axis=0, dtype=jnp.int32),
            coherence_time=jnp.where(density_total > 0.0, coherence, jnp.nan),
            rms_bandwidth=jnp.where(present, jnp.sqrt(variance), jnp.nan),
        )

    def _ledger(
        self,
        slices: FELBeamSlices,
        beam_change: Array,
        pulse_energy: Array,
        carry: _WindowCarry,
        wake_rate: Array,
        initial_beam: Array,
        slot_time: float,
        /,
    ) -> FELTimeDependentLedger:
        scale = self.core.lattice.scale
        # Energy per slice and unit Lorentz factor: I mₑc²/e over one slot.
        per_gain = (
            slices.currents
            * float(scale.electron_mass)
            * float(scale.speed_of_light) ** 2
            / float(scale.elementary_charge)
            * slot_time
        )
        wake_loss = jnp.sum(per_gain * wake_rate) * float(self.core.lattice.length)
        space_charge_loss = -jnp.sum(per_gain * carry.space_charge_gain)
        field_change = jnp.sum(pulse_energy[-1] - pulse_energy[0])
        defect = (
            beam_change
            + field_change
            + carry.diffraction_loss
            + carry.exit_energy
            + wake_loss
            + space_charge_loss
        )
        exchanged = (
            jnp.abs(beam_change)
            + jnp.abs(field_change)
            + carry.diffraction_loss
            + carry.exit_energy
            + jnp.abs(wake_loss)
            + jnp.abs(space_charge_loss)
        )
        floor = 1.0e-12 * initial_beam
        return FELTimeDependentLedger(
            beam_energy_change=beam_change,
            field_energy_change=field_change,
            diffraction_loss=carry.diffraction_loss,
            exit_energy=carry.exit_energy,
            wake_loss=wake_loss,
            space_charge_loss=space_charge_loss,
            defect=defect,
            relative_defect=jnp.abs(defect) / jnp.maximum(exchanged, floor),
        )

    def solve(self, slices: FELBeamSlices, key: PRNGKey, /) -> FELTimeDependentResult:
        """Integrate the coupled slices and their radiation window through the lattice."""
        if not isinstance(slices, FELBeamSlices):
            raise TypeError("slices must be FELBeamSlices.")
        root = parse(key, PRNGKey, "key")
        geometry = self._geometry(slices)
        core = self.core
        scale = core.lattice.scale
        charge = float(scale.elementary_charge)
        light = float(scale.speed_of_light)
        rest = float(scale.electron_mass) * light**2
        slot_time = geometry.spacing / light
        scaling = fel_scaling_estimate(core.lattice, slices, core.wavelength)
        electrons = slices.currents * geometry.spacing / (charge * light)
        collective = _CollectiveInput(
            positions=slices.positions,
            electrons=electrons,
            # I e/(ε₀ c k mₑc²): intra-slice field per unit bunching and area.
            source=slices.currents
            * charge
            / (float(scale.vacuum_permittivity) * light * core._wavenumber * rest),
        )
        match core.transverse:
            case "one-dimensional":
                # One radiation area couples every slot: the current-weighted mean.
                area = jnp.sum(slices.currents * scaling.beam_area) / jnp.sum(
                    slices.currents
                )
            case "angular-spectrum":
                area = jnp.asarray(core.cell_area)
            case _:
                assert_never(core.transverse)
        loaded = self._load(root, slices, electrons)
        wake_rate = core._wake_rates(slices)
        no_bunch_scale = self.space_charge is None or self.space_charge.bunch is None
        count = geometry.slices
        inputs = _SliceInput(
            identity=slices.identities,
            current=slices.currents,
            lorentz_factor=slices.lorentz_factors,
            relative_energy_spread=slices.relative_energy_spreads,
            normalized_emittance=slices.normalized_emittances,
            beta_function=slices.beta_functions,
            alpha_function=slices.alpha_functions,
            area=jnp.broadcast_to(area, (count,)),
            wake_rate=wake_rate,
        )
        field, seed_residual, seed_truncated = self._initial_field(slices, geometry, area)
        initial = _WindowCarry(
            phases=loaded.phases,
            lorentz_factors=loaded.lorentz_factors,
            positions=loaded.positions,
            momenta=loaded.momenta,
            field=field,
            diffraction_loss=jnp.asarray(0.0),
            exit_energy=jnp.asarray(0.0),
            accepted=jnp.asarray(True),
            lost=jnp.zeros((count,), dtype=jnp.int32),
            phase_step=jnp.zeros((count,), dtype=jnp.float64),
            space_charge_gain=jnp.zeros((count,), dtype=jnp.float64),
            transverse_kick=jnp.zeros((count,), dtype=jnp.float64),
            space_charge_accepted=jnp.asarray(True),
            cells_per_sigma=jnp.full((3,), jnp.nan if no_bunch_scale else jnp.inf),
        )
        rows = _WindowRow(
            step=_StepRow(
                core.step_lengths,
                core.undulator_wavenumbers,
                core.rms_strength_squared,
                core.couplings,
                core.smooth_focusing,
                core.quadrupole_kicks,
                core.phase_shifts,
            ),
            shift=jnp.asarray(geometry.shifts),
        )

        def body(
            carry: _WindowCarry, row: _WindowRow
        ) -> tuple[_WindowCarry, _WindowRecord]:
            return self._step(inputs, collective, geometry, carry, row)

        final, records = jax.lax.scan(body, initial, rows)
        first = self._record(initial, area)
        history = jax.tree.map(
            lambda start, rest_: jnp.concatenate((start[None], rest_), axis=0),
            first,
            records,
        )
        beam_scale = slices.currents * rest / (charge * core.loading.particle_count)
        beam_power = beam_scale[None, :] * history.beam_energy_sum
        pulse_energy = jnp.sum(history.power, axis=1) * slot_time
        beam_change = jnp.sum(beam_power[-1] - beam_power[0]) * slot_time
        initial_beam = jnp.sum(beam_power[0]) * slot_time
        ledger = self._ledger(
            slices,
            beam_change,
            pulse_energy,
            final,
            wake_rate,
            initial_beam,
            slot_time,
        )
        spectrum = self._spectrum(final.field, history.power[-1], geometry, area)
        window_positions = (
            slices.positions[0]
            + (jnp.arange(geometry.window, dtype=jnp.float64) - self.head_padding)
            * geometry.spacing
        )
        evidence = self._evidence(
            geometry,
            loaded,
            final,
            history,
            ledger,
            pulse_energy,
            initial_beam,
            electrons,
            (seed_residual, seed_truncated),
        )
        return FELTimeDependentResult(
            positions=core.record_positions,
            window_positions=window_positions,
            angular_frequencies=self._fundamental_frequency * core._harmonic_orders,
            power=history.power,
            pulse_energy=pulse_energy,
            bunching=history.bunching,
            beam_power=beam_power,
            mean_lorentz_factor=history.mean_lorentz_factor,
            relative_energy_spread=history.relative_energy_spread,
            wake_rate=wake_rate,
            space_charge_gain=history.space_charge_gain,
            exit_field=final.field,
            spectrum=spectrum,
            initial_particles=FELParticles(
                phases=loaded.phases,
                lorentz_factors=loaded.lorentz_factors,
                positions=loaded.positions,
                transverse_momenta=loaded.momenta,
            ),
            particles=FELParticles(
                phases=final.phases,
                lorentz_factors=final.lorentz_factors,
                positions=final.positions,
                transverse_momenta=final.momenta,
            ),
            scaling=scaling,
            ledger=ledger,
            evidence=evidence,
            harmonics=core.harmonics,
            record_count=parse(
                core.record_positions.shape[0], Size[FELRecordDim], "record_count"
            ),
            window_count=parse(geometry.window, Size[FELWindowDim], "window_count"),
            plan_id=self.plan_id,
        )

    def _transverse_ratio(self, loaded: _Loaded, final: _WindowCarry, /) -> Array:
        """Accumulated rms X3 transverse kick over the entrance rms ``u⊥`` per slice."""
        count = final.transverse_kick.shape[0]
        if self.space_charge is None or self.space_charge.bunch is None:
            return jnp.full((count,), jnp.nan)
        entrance = jnp.sqrt(
            jnp.mean(jnp.sum(loaded.momenta * loaded.momenta, axis=-1), axis=1)
        )
        kicked = final.transverse_kick
        # A cold beam (u⊥ = 0) with a nonzero kick is infinitely perturbed.
        return jnp.where(
            entrance > 0.0,
            kicked / jnp.where(entrance > 0.0, entrance, 1.0),
            jnp.where(kicked > 0.0, jnp.inf, 0.0),
        )

    def _transverse_omitted(self, loaded: _Loaded, final: _WindowCarry, /) -> Array:
        """An omitted X3 transverse kick that exceeds its declared tolerance."""
        config = self.space_charge
        if config is None or config.bunch is None or config.transverse == "applied":
            return jnp.asarray(False)
        ratio = self._transverse_ratio(loaded, final)
        return jnp.any(ratio > config.transverse_tolerance)

    def _evidence(
        self,
        geometry: _Geometry,
        loaded: _Loaded,
        final: _WindowCarry,
        history: _WindowRecord,
        ledger: FELTimeDependentLedger,
        pulse_energy: Array,
        initial_beam: Array,
        electrons: Array,
        seed: tuple[Array, Array],
        /,
    ) -> FELTimeDependentEvidence:
        seed_residual, seed_truncated = seed
        slice_finite = (
            jnp.all(jnp.isfinite(history.bunching), axis=(0, 2))
            & jnp.all(jnp.isfinite(history.beam_energy_sum), axis=0)
            & jnp.all(jnp.isfinite(final.lorentz_factors), axis=1)
            & jnp.all(jnp.isfinite(final.phases), axis=1)
        )
        noisy = (self.core.loading.shot_noise == "fawley") & (
            loaded.noise_amplitude > _NOISE_LINEARITY_LIMIT
        )
        slice_status = (
            jnp.where(slice_finite, 0, int(FELStatus.NONFINITE))
            | jnp.where(final.lost > 0, int(FELStatus.PARTICLE_SUPPORT_LOSS), 0)
            | jnp.where(
                final.phase_step > self.core.maximum_phase_step,
                int(FELStatus.PHASE_UNRESOLVED),
                0,
            )
            | jnp.where(noisy, int(FELStatus.NOISE_NONLINEAR), 0)
        ).astype(jnp.int32)
        finite = (
            jnp.all(slice_finite)
            & jnp.all(jnp.isfinite(history.power))
            & jnp.isfinite(ledger.defect)
        )
        # Roundoff floor: fields of quiet, unseeded runs carry ~1e-12 of the beam.
        radiated = jnp.maximum(
            jnp.sum(pulse_energy[-1]) + ledger.exit_energy, 1.0e-12 * initial_beam
        )
        exit_fraction = jnp.where(radiated > 0.0, ledger.exit_energy / radiated, 0.0)
        truncated = (self.boundary == "open") & (exit_fraction > self.window_tolerance)
        step_slots = self.maximum_coupled_slippage / geometry.spacing
        misaligned = (seed_residual > _SEED_ALIGNMENT_TOLERANCE) | (seed_truncated > 0.0)
        window_bits = (
            jnp.where(finite, 0, int(FELStatus.NONFINITE))
            | jnp.where(final.accepted, 0, int(FELStatus.DIFFRACTION_LEAKAGE))
            | jnp.where(
                ledger.relative_defect > self.core.ledger_tolerance,
                int(FELStatus.LEDGER_DEFECT),
                0,
            )
            | jnp.where(truncated, int(FELStatus.WINDOW_TRUNCATED), 0)
            | jnp.where(misaligned, int(FELStatus.SEED_UNREPRESENTED), 0)
            | jnp.where(
                final.space_charge_accepted, 0, int(FELStatus.SPACE_CHARGE_REFUSED)
            )
            | jnp.where(
                self._transverse_omitted(loaded, final),
                int(FELStatus.TRANSVERSE_SPACE_CHARGE_OMITTED),
                0,
            )
            | (
                int(FELStatus.SLIPPAGE_UNRESOLVED)
                if step_slots > 1.0 + _COMMENSURATE_TOLERANCE
                else 0
            )
        ).astype(jnp.int32)
        status = jnp.bitwise_or(
            window_bits, jax.lax.reduce(slice_status, jnp.int32(0), jnp.bitwise_or, (0,))
        )
        padding_length = self.head_padding * geometry.spacing
        return FELTimeDependentEvidence(
            status=status,
            slice_status=slice_status,
            finite=finite,
            maximum_phase_step=final.phase_step,
            lost_particles=final.lost,
            diffraction_accepted=final.accepted,
            electrons_per_slice=electrons,
            shot_noise_amplitude=loaded.noise_amplitude,
            maximum_prebunching_displacement=loaded.displacement,
            total_slippage=jnp.asarray(self.total_slippage),
            slippage_slots=jnp.asarray(self.total_slippage / geometry.spacing),
            maximum_step_slip_slots=jnp.asarray(step_slots),
            head_padding_length=jnp.asarray(padding_length),
            padding_sufficient=jnp.asarray(padding_length >= self.total_slippage),
            exit_fraction=exit_fraction,
            seed_alignment_residual=seed_residual,
            seed_truncated_energy=seed_truncated,
            space_charge_accepted=final.space_charge_accepted,
            space_charge_cells_per_sigma=final.cells_per_sigma,
            transverse_space_charge_ratio=self._transverse_ratio(loaded, final),
        )


def _validate_space_charge(
    core: FELPlan, space_charge: FELSpaceCharge, boundary: FELWindowBoundary, /
) -> None:
    needed = 2 * space_charge.harmonics
    if core.loading.particles_per_beamlet < needed:
        raise ValueError(
            "Harmonic space charge needs particles_per_beamlet >= 2 * harmonics "
            f"(= {needed})."
        )
    bunch = space_charge.bunch
    if bunch is None:
        return
    if bunch.scale.scale_id != core.lattice.scale.scale_id:
        raise ValueError("The bunch-scale space-charge plan must share the FEL scale.")
    if boundary != "open":
        raise ValueError(
            "Bunch-scale (X3) space charge is a free-space field of the finite "
            "window and needs an open window."
        )


def _validate_pulse_seed(core: FELPlan, seed: FELPulseSeed, /) -> None:
    if core.harmonics[0] != 1:
        raise ValueError("A seed drives the fundamental, which must be modeled.")
    match core.transverse:
        case "one-dimensional":
            if not seed.uniform:
                raise ValueError(
                    "The one-dimensional model needs a transversely uniform "
                    "(plane-wave) pulse envelope."
                )
        case "angular-spectrum":
            if (
                core.field_space is None
                or seed.envelope.plane_space.space_id != core.field_space.space_id
            ):
                raise ValueError(
                    "The pulse envelope must be sampled on the plan's PlaneFieldSpace."
                )
        case _:
            assert_never(core.transverse)


__all__ = [
    "FELPulseSeed",
    "FELSlippageRoute",
    "FELSpectrum",
    "FELTimeDependentEvidence",
    "FELTimeDependentLedger",
    "FELTimeDependentPlan",
    "FELTimeDependentResult",
    "FELWindowBoundary",
]
