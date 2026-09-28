#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Time-independent, period-averaged (KMR) free-electron-laser solver.

Model
-----
Each slice is one radiation wavelength ``λ = 2π/k`` of the beam and is
independent of the others (no slippage). With the phasor convention
``E_x = Re[Ẽ_h(x, y, z) e^{i(hkz − hωt)}]`` and ponderomotive phase
``θ = (k + k_u) z − ωt`` the Kroll–Morton–Rosenbluth period-averaged equations
for macroparticle ``j`` and harmonic ``h`` are

``dθ_j/dz = k_u − k (1 + a_w² + u_x² + u_y²) / (2γ_j²)``,

``dγ_j/dz = −(e / mₑc²) Σ_h κ_h Re[Ẽ_h(x_j) e^{ihθ_j}] / γ_j``,

``∂_z Ẽ_h = (i / 2hk) ∇⊥² Ẽ_h + (κ_h I / (ε₀ c N ΔA)) Σ_j W_j e^{−ihθ_j} / γ_j``,

with ``κ_h = a_w [JJ]_h / √2``, ``u = γ(x′, y′)``, ``N`` macroparticles of
equal charge, and ``W_j`` the multilinear weights of particle ``j`` on a
transverse node of area ``ΔA``. The field power is
``P_h = (ε₀c/2) ΔA Σ|Ẽ_h|²`` and the beam power ``(I mₑc²/(eN)) Σ_j γ_j``; the
deposit is the exact transpose of the field gather, so their exchange closes
the energy ledger up to the integration error. The ``"one-dimensional"``
transverse model replaces the grid by one field value per harmonic over the
slice area ``2π σ_x σ_y`` (no diffraction).

Transverse motion is linear betatron motion ``u′ = −γ k² x``, ``x′ = u/γ`` with
``k² = c_p a_w² k_u²/γ² + e G_s/(γ mₑ c)`` (natural focusing multipliers
``c = (0, 1)`` planar, ``(½, ½)`` helical); thin quadrupoles
``Δu_x = −e∫G dl x/(mₑc)``, ``Δu_y = +e∫G dl y/(mₑc)`` and phase shifters act at
break entrances.

Integration (per step ``Δz``) is the symmetric Strang sequence
``T(Δz/2) D(Δz/2) S(Δz) D(Δz/2) T(Δz/2)``: ``T`` exact betatron rotation,
``D`` diffraction by the optics angular-spectrum owner (exact Helmholtz
propagator with the carrier ``e^{ihkΔz}`` removed; it reduces to the paraxial
propagator for ``k⊥ ≪ hk``), and ``S`` the source kick, a classical RK4 step of
the coupled particle/field system at frozen transverse positions.

Longitudinal wakes (X2 ``WakeFunctionPlan``) enter per slice as a constant
energy-loss rate ``e V_s / (mₑc² L_w)`` from the causal wake potential
``V_s = Σ_{s′} q_{s′} W(ζ_s − ζ_{s′})`` (half self term) of the slice charges
``q_s = I_s Δζ / c`` per declared structure length ``L_w``.
"""

from __future__ import annotations

import math
from enum import IntFlag
from typing import assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ...._fingerprint import canonical_fingerprint
from ...._interpolation import apply_gather_stencil
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ...._validation import finite_real_scalar, positive_finite_float, positive_integer
from ....discretization import ParticleSetPlan
from ....discretization.splatting import (
    ParticleGridSplatPlan,
    ParticleGridSplatState,
    PreparedParticleGridSplat,
)
from ....optics.wave import (
    AngularSpectrumPlan,
    PlaneFieldSpace,
    PreparedAngularSpectrum,
    ScalarPlaneField,
)
from ....typing import (
    Bool,
    Complex128,
    Float64,
    Identifier,
    Int32,
    parse,
    PRNGKey,
    Size,
)
from .._wake import _si_units, WakeFunctionPlan
from ._lattice import FELUndulatorLattice
from ._slices import FELBeamSlices, FELLoading, FELParticles, load_slice
from ._theory import fel_scaling_estimate, FELScalingEstimate
from ._types import FELHarmonicDim, FELRecordDim, FELSliceDim


FELTransverseModel: TypeAlias = Literal["one-dimensional", "angular-spectrum"]

# First-order Fawley loading requires small phase perturbations σ₁ = √(2/N_e).
_NOISE_LINEARITY_LIMIT = 0.1


class FELStatus(IntFlag):
    """Fail-closed status bits of an averaged-FEL slice or time-dependent run.

    The first six bits apply to every slice; ``WINDOW_TRUNCATED``,
    ``SEED_UNREPRESENTED``, ``SPACE_CHARGE_REFUSED``,
    ``SLIPPAGE_UNRESOLVED`` (an undulator step slips the field past more than
    one slot, so it skips slices), and ``TRANSVERSE_SPACE_CHARGE_OMITTED`` (an
    omitted X3 transverse kick exceeds its declared tolerance) are raised only
    by the time-dependent solver.
    """

    SUCCESS = 0
    NONFINITE = 1
    DIFFRACTION_LEAKAGE = 2
    PARTICLE_SUPPORT_LOSS = 4
    LEDGER_DEFECT = 8
    PHASE_UNRESOLVED = 16
    NOISE_NONLINEAR = 32
    WINDOW_TRUNCATED = 64
    SEED_UNREPRESENTED = 128
    SPACE_CHARGE_REFUSED = 256
    SLIPPAGE_UNRESOLVED = 512
    TRANSVERSE_SPACE_CHARGE_OMITTED = 1024


class FELSeed(StrictModule, NonTrainableState):
    """Fundamental TEM₀₀ seed at the lattice entrance.

    ``waist`` is the ``1/e`` field radius ``w₀`` located ``focus_position``
    downstream of the entrance. In the one-dimensional model the seed is the
    uniform amplitude carrying ``power`` over the slice area.
    """

    power: float = eqx.field(static=True)
    waist: float = eqx.field(static=True)
    focus_position: float = eqx.field(static=True)
    seed_id: str = eqx.field(static=True)

    def __init__(
        self, power: float, /, *, waist: float, focus_position: float = 0.0
    ) -> None:
        power_ = finite_real_scalar(power, "power")
        if power_ < 0.0:
            raise ValueError("Seed power must be nonnegative.")
        self.power = power_
        self.waist = positive_finite_float(waist, "waist")
        self.focus_position = finite_real_scalar(focus_position, "focus_position")
        self.seed_id = canonical_fingerprint(
            {
                "kind": "fel-seed",
                "power": power_,
                "waist": self.waist,
                "focus_position": self.focus_position,
            }
        )


class FELWakeLoss(StrictModule, NonTrainableState):
    """Longitudinal wake applied per slice along the whole lattice.

    ``wake`` is an X2 longitudinal ``WakeFunctionPlan`` (V/C, SI) describing
    one ``structure_length`` (scale length) of the undulator line.
    """

    wake: WakeFunctionPlan
    structure_length: float = eqx.field(static=True)
    wake_id: str = eqx.field(static=True)

    def __init__(self, wake: WakeFunctionPlan, /, *, structure_length: float) -> None:
        if not isinstance(wake, WakeFunctionPlan):
            raise TypeError("wake must be a WakeFunctionPlan.")
        if wake.kind != "longitudinal":
            raise ValueError("The time-independent FEL applies longitudinal wakes only.")
        self.wake = wake
        self.structure_length = positive_finite_float(
            structure_length, "structure_length"
        )
        self.wake_id = canonical_fingerprint(
            {
                "kind": "fel-wake-loss",
                "wake": wake.plan_id,
                "structure_length": self.structure_length,
            }
        )


class FELEnergyLedger(StrictModule):
    """Particle↔field power ledger per slice (scale power units).

    ``defect = beam_power_change + field_power_change + diffraction_loss +
    wake_loss`` vanishes for an exact integration; ``relative_defect`` divides
    it by the total exchanged power.
    """

    __strict_contract__ = True

    beam_power_change: Float64[FELSliceDim]
    field_power_change: Float64[FELSliceDim]
    diffraction_loss: Float64[FELSliceDim]
    wake_loss: Float64[FELSliceDim]
    defect: Float64[FELSliceDim]
    relative_defect: Float64[FELSliceDim]


class FELGainEvidence(StrictModule):
    """Gain-length and saturation evidence of the first modeled harmonic.

    ``fitted_gain_length`` is the power e-folding length from a least-squares
    fit of ``ln P(z)`` over samples with ``P`` inside ``gain_fit_window``
    (fractions of the peak power) before the peak; ``fit_valid`` requires at
    least three such samples and positive growth. ``saturation_power`` and
    ``saturation_position`` locate the peak power.
    """

    __strict_contract__ = True

    fitted_gain_length: Float64[FELSliceDim]
    fit_sample_count: Int32[FELSliceDim]
    fit_valid: Bool[FELSliceDim]
    saturation_power: Float64[FELSliceDim]
    saturation_position: Float64[FELSliceDim]
    scaling: FELScalingEstimate


class FELEvidence(StrictModule):
    """Resolution, support, and status evidence per slice."""

    __strict_contract__ = True

    status: Int32[FELSliceDim]
    finite: Bool[FELSliceDim]
    maximum_phase_step: Float64[FELSliceDim]
    lost_particles: Int32[FELSliceDim]
    diffraction_accepted: Bool[FELSliceDim]
    electrons_per_slice: Float64[FELSliceDim]
    shot_noise_amplitude: Float64[FELSliceDim]

    @property
    def successful(self) -> Array:
        return self.status == int(FELStatus.SUCCESS)


class FELResult(StrictModule):
    """Averaged-FEL solution for every slice.

    ``power[s, z, h]`` and complex ``bunching[s, z, h] = ⟨e^{−ihθ}⟩`` are
    recorded at ``positions`` (entrance and every step end);
    ``initial_particles`` and ``particles`` are the loaded and exit
    macroparticle states. ``exit_field`` is
    ``[s, h]`` (one-dimensional) or ``[s, n₀, n₁, h]`` (grid, V/m in scale
    units). ``angular_frequencies`` are the slice harmonics ``hω``; the grid
    route adds the exit far field ``far_field_intensity[s, n₀, n₁, h]``
    (``dP/dΩ``) on the centered angle axes ``far_field_angles[h]``.
    """

    __strict_contract__ = True

    positions: Float64[FELRecordDim]
    angular_frequencies: Float64[FELHarmonicDim]
    power: Float64[FELSliceDim, FELRecordDim, FELHarmonicDim]
    bunching: Complex128[FELSliceDim, FELRecordDim, FELHarmonicDim]
    beam_power: Float64[FELSliceDim, FELRecordDim]
    mean_lorentz_factor: Float64[FELSliceDim, FELRecordDim]
    relative_energy_spread: Float64[FELSliceDim, FELRecordDim]
    exit_field: Array
    far_field_intensity: Array | None
    far_field_angles: tuple[tuple[Array, Array], ...] | None
    initial_particles: FELParticles
    particles: FELParticles
    ledger: FELEnergyLedger
    gain: FELGainEvidence
    evidence: FELEvidence
    harmonics: tuple[int, ...] = eqx.field(static=True)
    record_count: Size[FELRecordDim] = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)


class _SliceInput(NamedTuple):
    identity: Array
    current: Array
    lorentz_factor: Array
    relative_energy_spread: Array
    normalized_emittance: Array
    beta_function: Array
    alpha_function: Array
    area: Array
    wake_rate: Array


class _StepRow(NamedTuple):
    length: Array
    undulator_wavenumber: Array
    rms_strength_squared: Array
    coupling: Array
    smooth_focusing: Array
    quadrupole_kick: Array
    phase_shift: Array


class _Carry(NamedTuple):
    phases: Array
    lorentz_factors: Array
    positions: Array
    momenta: Array
    field: Array
    diffraction_loss: Array
    accepted: Array
    lost: Array
    phase_step: Array


class _Record(NamedTuple):
    power: Array
    bunching: Array
    beam_energy_sum: Array
    mean_lorentz_factor: Array
    relative_energy_spread: Array


class _SliceOutput(NamedTuple):
    power: Array
    bunching: Array
    beam_power: Array
    mean_lorentz_factor: Array
    relative_energy_spread: Array
    exit_field: Array
    initial: FELParticles
    final: FELParticles
    beam_power_change: Array
    diffraction_loss: Array
    accepted: Array
    lost: Array
    phase_step: Array
    noise_amplitude: Array


def _validated_harmonics(harmonics: tuple[int, ...], /) -> tuple[int, ...]:
    values = tuple(harmonics)
    if not values:
        raise ValueError("At least one harmonic must be modeled.")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("harmonics must be positive integers.")
    if list(values) != sorted(set(values)):
        raise ValueError("harmonics must be unique and increasing.")
    return values


class FELPlan(StrictModule, NonTrainableState):
    """Time-independent averaged FEL over an undulator lattice.

    ``wavelength`` is the fundamental radiation wavelength of every slice (scale
    length). ``harmonics`` lists the modeled odd planar harmonics (helical:
    fundamental only); ``loading.particles_per_beamlet ≥ 2 max(harmonics)``.
    The ``"angular-spectrum"`` model requires ``field_space`` (uniform
    ``PlaneFieldSpace``) and ``propagation``; the ``"one-dimensional"`` model
    refuses them. ``slice_batch`` bounds the number of slices integrated
    concurrently.
    """

    lattice: FELUndulatorLattice
    loading: FELLoading
    field_space: PlaneFieldSpace | None
    propagation: PreparedAngularSpectrum | None
    splat: PreparedParticleGridSplat | None
    seed: FELSeed | None
    wake: FELWakeLoss | None
    step_lengths: Array
    undulator_wavenumbers: Array
    rms_strength_squared: Array
    couplings: Array
    smooth_focusing: Array
    quadrupole_kicks: Array
    phase_shifts: Array
    natural_focusing: Array
    record_positions: Array
    wavelength: float = eqx.field(static=True)
    harmonics: tuple[int, ...] = eqx.field(static=True)
    transverse: FELTransverseModel = eqx.field(static=True)
    slice_batch: int = eqx.field(static=True)
    ledger_tolerance: float = eqx.field(static=True)
    maximum_phase_step: float = eqx.field(static=True)
    gain_fit_window: tuple[float, float] = eqx.field(static=True)
    cell_area: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lattice: FELUndulatorLattice,
        wavelength: float,
        /,
        *,
        loading: FELLoading,
        harmonics: tuple[int, ...] = (1,),
        transverse: FELTransverseModel = "one-dimensional",
        field_space: PlaneFieldSpace | None = None,
        propagation: AngularSpectrumPlan | None = None,
        seed: FELSeed | None = None,
        wake: FELWakeLoss | None = None,
        slice_batch: int = 1,
        ledger_tolerance: float = 1.0e-4,
        maximum_phase_step: float = 0.5,
        gain_fit_window: tuple[float, float] = (1.0e-4, 1.0e-2),
    ) -> None:
        if not isinstance(lattice, FELUndulatorLattice):
            raise TypeError("lattice must be an FELUndulatorLattice.")
        if not isinstance(loading, FELLoading):
            raise TypeError("loading must be FELLoading.")
        if seed is not None and not isinstance(seed, FELSeed):
            raise TypeError("seed must be FELSeed or None.")
        if wake is not None and not isinstance(wake, FELWakeLoss):
            raise TypeError("wake must be FELWakeLoss or None.")
        radiation = positive_finite_float(wavelength, "wavelength")
        harmonics_ = _validated_harmonics(harmonics)
        segment_couplings = (
            np.asarray(lattice.fundamental_couplings, dtype=np.float64)[:, None]
            if harmonics_ == (1,)
            else lattice.coupling_table(harmonics_)
        )
        if loading.particles_per_beamlet < 2 * harmonics_[-1]:
            raise ValueError(
                "Quiet-start loading needs particles_per_beamlet >= 2 * max(harmonics) "
                f"(= {2 * harmonics_[-1]})."
            )
        if seed is not None and harmonics_[0] != 1:
            raise ValueError("A seed drives the fundamental, which must be modeled.")
        transverse_ = parse(transverse, FELTransverseModel, "transverse")
        match transverse_:
            case "one-dimensional":
                if field_space is not None or propagation is not None:
                    raise ValueError(
                        "The one-dimensional model takes no field_space or propagation."
                    )
                prepared = None
                splat = None
                cell_area = 1.0
            case "angular-spectrum":
                if not isinstance(field_space, PlaneFieldSpace):
                    raise TypeError(
                        "The angular-spectrum model requires a PlaneFieldSpace."
                    )
                if not isinstance(propagation, AngularSpectrumPlan):
                    raise TypeError(
                        "The angular-spectrum model requires an AngularSpectrumPlan."
                    )
                prepared = propagation.prepare(field_space)
                particles = ParticleSetPlan(
                    np.arange(loading.particle_count, dtype=np.int64),
                    np.ones((loading.particle_count,), dtype=np.float64),
                    ambient_dimension=2,
                    name="fel-macroparticles",
                ).prepare()
                splat = ParticleGridSplatPlan(field_space.grid, boundary="drop").prepare(
                    particles
                )
                cell_area = prepared.sample_spacings[0] * prepared.sample_spacings[1]
            case _:
                assert_never(transverse_)
        batch = positive_integer(slice_batch, "slice_batch")
        tolerance = positive_finite_float(ledger_tolerance, "ledger_tolerance")
        phase_limit = positive_finite_float(maximum_phase_step, "maximum_phase_step")
        lower, upper = (float(value) for value in gain_fit_window)
        if not 0.0 < lower < upper < 1.0:
            raise ValueError("gain_fit_window must satisfy 0 < lower < upper < 1.")
        table = lattice.step_table()
        self.lattice = lattice
        self.loading = loading
        self.field_space = field_space
        self.propagation = prepared
        self.splat = splat
        self.seed = seed
        self.wake = wake
        self.step_lengths = jnp.asarray(table.lengths)
        self.undulator_wavenumbers = jnp.asarray(table.undulator_wavenumbers)
        self.rms_strength_squared = jnp.asarray(table.rms_strength_squared)
        # κ vanishes in breaks (a_w = 0); modules use their segment's coupling.
        inside = (table.undulator_wavenumbers > 0.0)[:, None]
        self.couplings = jnp.asarray(
            np.where(inside, segment_couplings[table.segment_index], 0.0)
        )
        self.smooth_focusing = jnp.asarray(table.smooth_focusing)
        self.quadrupole_kicks = jnp.asarray(table.quadrupole_kick)
        self.phase_shifts = jnp.asarray(table.phase_shift)
        self.natural_focusing = jnp.asarray(table.natural_focusing)
        self.record_positions = jnp.asarray(np.concatenate(([0.0], table.end_positions)))
        self.wavelength = radiation
        self.harmonics = harmonics_
        self.transverse = transverse_
        self.slice_batch = batch
        self.ledger_tolerance = tolerance
        self.maximum_phase_step = phase_limit
        self.gain_fit_window = (lower, upper)
        self.cell_area = cell_area
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fel-averaged-plan",
                "lattice": lattice.lattice_id,
                "wavelength": radiation,
                "loading": loading.loading_id,
                "harmonics": list(harmonics_),
                "transverse": transverse_,
                "propagation": None if prepared is None else prepared.prepared_id,
                "seed": None if seed is None else seed.seed_id,
                "wake": None if wake is None else wake.wake_id,
                "ledger_tolerance": tolerance,
                "maximum_phase_step": phase_limit,
                "gain_fit_window": [lower, upper],
            }
        )

    # ------------------------------------------------------------------ physics

    @property
    def _wavenumber(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @property
    def _harmonic_orders(self) -> Array:
        return jnp.asarray(self.harmonics, dtype=jnp.float64)

    def _field_power(self, field: Array, area: Array, /) -> Array:
        """``(ε₀c/2) ΔA Σ|Ẽ_h|²`` per harmonic."""
        scale = self.lattice.scale
        prefactor = 0.5 * float(scale.vacuum_permittivity) * float(scale.speed_of_light)
        intensity = jnp.real(field * jnp.conj(field))
        match self.transverse:
            case "one-dimensional":
                return prefactor * area * intensity
            case "angular-spectrum":
                return prefactor * self.cell_area * jnp.sum(intensity, axis=(0, 1))
            case _:
                assert_never(self.transverse)

    def _seed_field(self, area: Array, /) -> Array:
        count = len(self.harmonics)
        scale = self.lattice.scale
        prefactor = 0.5 * float(scale.vacuum_permittivity) * float(scale.speed_of_light)
        power = 0.0 if self.seed is None else self.seed.power
        match self.transverse:
            case "one-dimensional":
                amplitude = jnp.sqrt(power / (prefactor * area))
                return jnp.zeros((count,), dtype=jnp.complex128).at[0].set(amplitude)
            case "angular-spectrum":
                space = self.field_space
                if space is None:
                    raise RuntimeError("The grid model lost its field space.")
                shape = space.shape + (count,)
                if self.seed is None or power == 0.0:
                    return jnp.zeros(shape, dtype=jnp.complex128)
                coordinates = space.transverse_coordinates
                radius_squared = jnp.sum(coordinates * coordinates, axis=-1)
                rayleigh = 0.5 * self._wavenumber * self.seed.waist**2
                # e^{+ikz} convention: u ∝ q⁻¹ e^{ikr²/(2q)}, q = (z − z_f) − i z_R.
                q = -self.seed.focus_position - 1j * rayleigh
                profile = jnp.exp(1j * self._wavenumber * radius_squared / (2.0 * q)) / q
                norm = prefactor * self.cell_area * jnp.sum(jnp.abs(profile) ** 2)
                seed = profile * jnp.sqrt(power / norm)
                return jnp.zeros(shape, dtype=jnp.complex128).at[..., 0].set(seed)
            case _:
                assert_never(self.transverse)

    def _bind(self, positions: Array, /) -> tuple[ParticleGridSplatState | None, Array]:
        match self.transverse:
            case "one-dimensional":
                return None, jnp.asarray(0, dtype=jnp.int32)
            case "angular-spectrum":
                if self.splat is None:
                    raise RuntimeError("The grid model lost its splat transfer.")
                state = self.splat.build(positions)
                lost = jnp.sum(
                    state.truncated_support_mask | state.out_of_domain_mask,
                    dtype=jnp.int32,
                )
                return state, lost
            case _:
                assert_never(self.transverse)

    def _gather(self, field: Array, state: ParticleGridSplatState | None, /) -> Array:
        """Field at every particle ``[N, H]`` (complex)."""
        match self.transverse:
            case "one-dimensional":
                return jnp.broadcast_to(
                    field[None, :], (self.loading.particle_count, field.shape[0])
                )
            case "angular-spectrum":
                if self.splat is None or state is None:
                    raise RuntimeError("The grid model needs a bound splat state.")
                pair = jnp.stack((jnp.real(field), jnp.imag(field)), axis=-1)
                gathered = self.splat.gather(state, pair).values
                return gathered[..., 0] + 1j * gathered[..., 1]
            case _:
                assert_never(self.transverse)

    def _deposit(self, values: Array, state: ParticleGridSplatState | None, /) -> Array:
        """Sum ``values[N, H]`` onto the field layout (transpose of the gather)."""
        match self.transverse:
            case "one-dimensional":
                return jnp.sum(values, axis=0)
            case "angular-spectrum":
                if self.splat is None or state is None:
                    raise RuntimeError("The grid model needs a bound splat state.")
                size = self.splat.target_size
                count = values.shape[1]
                stencil = state.stencil
                template = jax.ShapeDtypeStruct((size, count, 2), jnp.float64)

                def gather(nodes: Array) -> Array:
                    return apply_gather_stencil(nodes, stencil).values

                (deposited,) = jax.linear_transpose(gather, template)(
                    jnp.stack((jnp.real(values), jnp.imag(values)), axis=-1)
                )
                grid = deposited.reshape(self.splat.target_shape + (count, 2))
                return grid[..., 0] + 1j * grid[..., 1]
            case _:
                assert_never(self.transverse)

    def _diffract(self, field: Array, distance: Array, /) -> tuple[Array, Array, Array]:
        match self.transverse:
            case "one-dimensional":
                return field, jnp.asarray(0.0), jnp.asarray(True)
            case "angular-spectrum":
                if self.propagation is None or self.field_space is None:
                    raise RuntimeError("The grid model lost its propagation.")
                scale = self.lattice.scale
                prefactor = (
                    0.5 * float(scale.vacuum_permittivity) * float(scale.speed_of_light)
                )
                frequency = 2.0 * math.pi * float(scale.speed_of_light) / self.wavelength
                values = []
                loss = jnp.asarray(0.0)
                accepted = jnp.asarray(True)
                for index, harmonic in enumerate(self.harmonics):
                    wavenumber = harmonic * self._wavenumber
                    plane = ScalarPlaneField(
                        self.field_space, field[..., index], harmonic * frequency, 0.0
                    )
                    result = self.propagation.execute(plane, distance, wavenumber)
                    # Remove the carrier: the envelope advances by e^{i(k_z − hk)Δz}.
                    values.append(
                        result.field.values * jnp.exp(-1j * wavenumber * distance)
                    )
                    loss = loss + prefactor * result.evidence.cropped_energy
                    accepted = accepted & result.evidence.accepted
                return jnp.stack(values, axis=-1), loss, accepted
            case _:
                assert_never(self.transverse)

    def _betatron(
        self,
        positions: Array,
        momenta: Array,
        lorentz_factors: Array,
        row: _StepRow,
        distance: Array,
        /,
    ) -> tuple[Array, Array]:
        """Exact linear betatron rotation at frozen energy."""
        gamma = lorentz_factors[:, None]
        natural = (
            self.natural_focusing[None, :]
            * row.rms_strength_squared
            * row.undulator_wavenumber**2
            / (gamma * gamma)
        )
        strength = natural + row.smooth_focusing / gamma
        wavenumber = jnp.sqrt(strength)
        advance = wavenumber * distance
        cosine = jnp.cos(advance)
        sine_over_k = distance * jnp.sinc(advance / jnp.pi)
        updated_positions = positions * cosine + momenta / gamma * sine_over_k
        updated_momenta = momenta * cosine - gamma * strength * positions * sine_over_k
        return updated_positions, updated_momenta

    def _source(
        self,
        carry: _Carry,
        row: _StepRow,
        state: ParticleGridSplatState | None,
        inputs: _SliceInput,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        """RK4 source kick of ``(θ, γ, Ẽ)`` over one step at frozen positions."""
        scale = self.lattice.scale
        charge_over_rest = float(scale.elementary_charge) / (
            float(scale.electron_mass) * float(scale.speed_of_light) ** 2
        )
        orders = self._harmonic_orders
        transverse_squared = jnp.sum(carry.momenta * carry.momenta, axis=-1)
        match self.transverse:
            case "one-dimensional":
                area = inputs.area
            case "angular-spectrum":
                area = jnp.asarray(self.cell_area)
            case _:
                assert_never(self.transverse)
        field_scale = (
            row.coupling
            * inputs.current
            / (
                float(scale.vacuum_permittivity)
                * float(scale.speed_of_light)
                * self.loading.particle_count
                * area
            )
        )
        wavenumber = self._wavenumber

        def rates(
            phases: Array, gamma: Array, field: Array
        ) -> tuple[Array, Array, Array]:
            rotation = jnp.exp(1j * orders[None, :] * phases[:, None])
            drive = jnp.real(self._gather(field, state) * rotation)
            dgamma = -charge_over_rest * (drive @ row.coupling) / gamma - inputs.wake_rate
            dphase = row.undulator_wavenumber - wavenumber * (
                1.0 + row.rms_strength_squared + transverse_squared
            ) / (2.0 * gamma * gamma)
            dfield = (
                self._deposit(jnp.conj(rotation) / gamma[:, None], state) * field_scale
            )
            return dphase, dgamma, dfield

        step = row.length
        phases, gamma, field = carry.phases, carry.lorentz_factors, carry.field
        k1 = rates(phases, gamma, field)
        k2 = rates(
            phases + 0.5 * step * k1[0],
            gamma + 0.5 * step * k1[1],
            field + 0.5 * step * k1[2],
        )
        k3 = rates(
            phases + 0.5 * step * k2[0],
            gamma + 0.5 * step * k2[1],
            field + 0.5 * step * k2[2],
        )
        k4 = rates(phases + step * k3[0], gamma + step * k3[1], field + step * k3[2])
        sixth = step / 6.0
        phases = phases + sixth * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        gamma = gamma + sixth * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        field = field + sixth * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        # Only coupled (undulator) steps must resolve the detuning phase; break
        # slippage is exact because θ is linear there.
        coupled = row.undulator_wavenumber > 0.0
        phase_step = jnp.where(coupled, jnp.max(jnp.abs(k1[0])) * step, 0.0)
        return phases, gamma, field, phase_step

    def _step(
        self, inputs: _SliceInput, carry: _Carry, row: _StepRow, /
    ) -> tuple[_Carry, _Record]:
        half = 0.5 * row.length
        kick = row.quadrupole_kick
        momenta = carry.momenta + kick * carry.positions * jnp.asarray((-1.0, 1.0))
        phases = carry.phases + row.phase_shift
        positions, momenta = self._betatron(
            carry.positions, momenta, carry.lorentz_factors, row, half
        )
        field, first_loss, first_accepted = self._diffract(carry.field, half)
        state, lost = self._bind(positions)
        staged = carry._replace(
            phases=phases, positions=positions, momenta=momenta, field=field
        )
        phases, gamma, field, phase_step = self._source(staged, row, state, inputs)
        field, second_loss, second_accepted = self._diffract(field, half)
        positions, momenta = self._betatron(positions, momenta, gamma, row, half)
        updated = _Carry(
            phases=phases,
            lorentz_factors=gamma,
            positions=positions,
            momenta=momenta,
            field=field,
            diffraction_loss=carry.diffraction_loss + first_loss + second_loss,
            accepted=carry.accepted & first_accepted & second_accepted,
            lost=jnp.maximum(carry.lost, lost),
            phase_step=jnp.maximum(carry.phase_step, phase_step),
        )
        return updated, self._record(updated, inputs.area)

    def _record(self, carry: _Carry, area: Array, /) -> _Record:
        gamma = carry.lorentz_factors
        mean = jnp.mean(gamma)
        bunching = jnp.mean(
            jnp.exp(-1j * self._harmonic_orders[None, :] * carry.phases[:, None]), axis=0
        )
        return _Record(
            power=self._field_power(carry.field, area),
            bunching=bunching,
            beam_energy_sum=jnp.sum(gamma),
            mean_lorentz_factor=mean,
            relative_energy_spread=jnp.std(gamma) / mean,
        )

    def _slice(self, key: PRNGKey, inputs: _SliceInput, /) -> _SliceOutput:
        scale = self.lattice.scale
        electrons = (
            inputs.current
            * self.wavelength
            / (float(scale.elementary_charge) * float(scale.speed_of_light))
        )
        loaded = load_slice(
            self.loading,
            key,
            inputs.identity,
            inputs.lorentz_factor,
            inputs.relative_energy_spread,
            inputs.normalized_emittance,
            inputs.beta_function,
            inputs.alpha_function,
            electrons / self.loading.beamlet_count,
        )
        initial = _Carry(
            phases=loaded.phases,
            lorentz_factors=loaded.lorentz_factors,
            positions=loaded.positions,
            momenta=loaded.transverse_momenta,
            field=self._seed_field(inputs.area),
            diffraction_loss=jnp.asarray(0.0),
            accepted=jnp.asarray(True),
            lost=jnp.asarray(0, dtype=jnp.int32),
            phase_step=jnp.asarray(0.0),
        )
        rows = _StepRow(
            self.step_lengths,
            self.undulator_wavenumbers,
            self.rms_strength_squared,
            self.couplings,
            self.smooth_focusing,
            self.quadrupole_kicks,
            self.phase_shifts,
        )

        def body(carry: _Carry, row: _StepRow) -> tuple[_Carry, _Record]:
            return self._step(inputs, carry, row)

        final, records = jax.lax.scan(body, initial, rows)
        first = self._record(initial, inputs.area)
        history = jax.tree.map(
            lambda start, rest: jnp.concatenate((start[None], rest), axis=0),
            first,
            records,
        )
        beam_scale = (
            inputs.current
            * float(scale.electron_mass)
            * float(scale.speed_of_light) ** 2
            / (float(scale.elementary_charge) * self.loading.particle_count)
        )
        return _SliceOutput(
            power=history.power,
            bunching=history.bunching,
            beam_power=beam_scale * history.beam_energy_sum,
            mean_lorentz_factor=history.mean_lorentz_factor,
            relative_energy_spread=history.relative_energy_spread,
            exit_field=final.field,
            initial=FELParticles(
                phases=loaded.phases,
                lorentz_factors=loaded.lorentz_factors,
                positions=loaded.positions,
                transverse_momenta=loaded.transverse_momenta,
            ),
            final=FELParticles(
                phases=final.phases,
                lorentz_factors=final.lorentz_factors,
                positions=final.positions,
                transverse_momenta=final.momenta,
            ),
            beam_power_change=beam_scale
            * jnp.sum(final.lorentz_factors - loaded.lorentz_factors),
            diffraction_loss=final.diffraction_loss,
            accepted=final.accepted,
            lost=final.lost,
            phase_step=final.phase_step,
            noise_amplitude=loaded.noise_amplitude,
        )

    # --------------------------------------------------------------- execution

    def _wake_rates(self, slices: FELBeamSlices, /) -> Array:
        """Per-slice energy-loss rate ``e V_s / (mₑc² L_w)`` (1/length)."""
        if self.wake is None:
            return jnp.zeros((slices.slice_count,), dtype=jnp.float64)
        if slices.position_spacing is None:
            raise ValueError(
                "Per-slice wakes need at least two uniformly spaced slice positions."
            )
        scale = self.lattice.scale
        units = _si_units(scale)
        light = float(scale.speed_of_light)
        charges = units.charge * slices.currents * slices.position_spacing / light
        lags = units.length * (slices.positions[:, None] - slices.positions[None, :])
        kernel = self.wake.wake.evaluate(lags)
        kernel = kernel * (1.0 - 0.5 * jnp.eye(slices.slice_count, dtype=jnp.float64))
        potential = kernel @ charges
        rest_energy = units.energy * float(scale.electron_mass) * light**2
        electron_charge = units.charge * float(scale.elementary_charge)
        return electron_charge * potential / (rest_energy * self.wake.structure_length)

    def _far_field(
        self, exit_field: Array, /
    ) -> tuple[Array, tuple[tuple[Array, Array], ...]]:
        if self.propagation is None or self.field_space is None:
            raise RuntimeError("The far field needs the grid model.")
        scale = self.lattice.scale
        prefactor = 0.5 * float(scale.vacuum_permittivity) * float(scale.speed_of_light)
        spacing = self.propagation.sample_spacings
        rows, columns = self.field_space.shape
        spectrum = (
            jnp.fft.fftshift(jnp.fft.fft2(exit_field, axes=(1, 2)), axes=(1, 2))
            * self.cell_area
        )
        intensities = []
        angles = []
        for index, harmonic in enumerate(self.harmonics):
            wavenumber = harmonic * self._wavenumber
            intensities.append(
                prefactor
                * (wavenumber / (2.0 * math.pi)) ** 2
                * jnp.abs(spectrum[..., index]) ** 2
            )
            angles.append(
                tuple(
                    jnp.fft.fftshift(2.0 * math.pi * jnp.fft.fftfreq(size, d=step))
                    / wavenumber
                    for size, step in ((rows, spacing[0]), (columns, spacing[1]))
                )
            )
        return jnp.stack(intensities, axis=-1), tuple(
            (pair[0], pair[1]) for pair in angles
        )

    def _gain(
        self, positions: Array, power: Array, scaling: FELScalingEstimate, /
    ) -> FELGainEvidence:
        trace = power[:, :, 0]
        peak_index = jnp.argmax(trace, axis=1)
        peak = jnp.take_along_axis(trace, peak_index[:, None], axis=1)[:, 0]
        lower, upper = self.gain_fit_window
        before_peak = jnp.arange(trace.shape[1])[None, :] <= peak_index[:, None]
        mask = (
            before_peak
            & (trace >= lower * peak[:, None])
            & (trace <= upper * peak[:, None])
            & (trace > 0.0)
        )
        weights = mask.astype(jnp.float64)
        count = jnp.sum(weights, axis=1)
        safe_count = jnp.maximum(count, 1.0)
        z = positions[None, :]
        log_power = jnp.log(jnp.where(trace > 0.0, trace, 1.0))
        z_mean = jnp.sum(weights * z, axis=1) / safe_count
        y_mean = jnp.sum(weights * log_power, axis=1) / safe_count
        centered = (z - z_mean[:, None]) * weights
        variance = jnp.sum(centered * (z - z_mean[:, None]), axis=1)
        covariance = jnp.sum(centered * (log_power - y_mean[:, None]), axis=1)
        slope = covariance / jnp.where(variance > 0.0, variance, 1.0)
        valid = (count >= 3.0) & (variance > 0.0) & (slope > 0.0)
        return FELGainEvidence(
            fitted_gain_length=jnp.where(
                valid, 1.0 / jnp.where(valid, slope, 1.0), jnp.nan
            ),
            fit_sample_count=count.astype(jnp.int32),
            fit_valid=valid,
            saturation_power=peak,
            saturation_position=positions[peak_index],
            scaling=scaling,
        )

    def _ledger(
        self, slices: FELBeamSlices, output: _SliceOutput, wake_rate: Array, /
    ) -> FELEnergyLedger:
        scale = self.lattice.scale
        wake_loss = (
            slices.currents
            * float(scale.electron_mass)
            * float(scale.speed_of_light) ** 2
            / float(scale.elementary_charge)
            * wake_rate
            * float(self.lattice.length)
        )
        field_change = jnp.sum(output.power[:, -1, :] - output.power[:, 0, :], axis=-1)
        defect = (
            output.beam_power_change + field_change + output.diffraction_loss + wake_loss
        )
        exchanged = (
            jnp.abs(output.beam_power_change)
            + jnp.abs(field_change)
            + output.diffraction_loss
            + jnp.abs(wake_loss)
        )
        # Roundoff floor for runs without exchange: beam-power sums carry ~1e-12.
        floor = 1.0e-12 * output.beam_power[:, 0]
        return FELEnergyLedger(
            beam_power_change=output.beam_power_change,
            field_power_change=field_change,
            diffraction_loss=output.diffraction_loss,
            wake_loss=wake_loss,
            defect=defect,
            relative_defect=jnp.abs(defect) / jnp.maximum(exchanged, floor),
        )

    def _evidence(
        self, slices: FELBeamSlices, output: _SliceOutput, ledger: FELEnergyLedger, /
    ) -> FELEvidence:
        scale = self.lattice.scale
        finite = (
            jnp.all(jnp.isfinite(output.power), axis=(1, 2))
            & jnp.all(jnp.isfinite(output.beam_power), axis=1)
            & jnp.all(jnp.isfinite(output.final.lorentz_factors), axis=1)
            & jnp.all(jnp.isfinite(output.final.phases), axis=1)
            & jnp.isfinite(ledger.defect)
        )
        electrons = (
            slices.currents
            * self.wavelength
            / (float(scale.elementary_charge) * float(scale.speed_of_light))
        )
        noisy = (self.loading.shot_noise == "fawley") & (
            output.noise_amplitude > _NOISE_LINEARITY_LIMIT
        )
        status = (
            jnp.where(finite, 0, int(FELStatus.NONFINITE))
            | jnp.where(output.accepted, 0, int(FELStatus.DIFFRACTION_LEAKAGE))
            | jnp.where(output.lost > 0, int(FELStatus.PARTICLE_SUPPORT_LOSS), 0)
            | jnp.where(
                ledger.relative_defect > self.ledger_tolerance,
                int(FELStatus.LEDGER_DEFECT),
                0,
            )
            | jnp.where(
                output.phase_step > self.maximum_phase_step,
                int(FELStatus.PHASE_UNRESOLVED),
                0,
            )
            | jnp.where(noisy, int(FELStatus.NOISE_NONLINEAR), 0)
        ).astype(jnp.int32)
        return FELEvidence(
            status=status,
            finite=finite,
            maximum_phase_step=output.phase_step,
            lost_particles=output.lost,
            diffraction_accepted=output.accepted,
            electrons_per_slice=electrons,
            shot_noise_amplitude=output.noise_amplitude,
        )

    def solve(self, slices: FELBeamSlices, key: PRNGKey, /) -> FELResult:
        """Integrate every slice through the lattice."""
        if not isinstance(slices, FELBeamSlices):
            raise TypeError("slices must be FELBeamSlices.")
        root = parse(key, PRNGKey, "key")
        scaling = fel_scaling_estimate(self.lattice, slices, self.wavelength)
        inputs = _SliceInput(
            identity=slices.identities,
            current=slices.currents,
            lorentz_factor=slices.lorentz_factors,
            relative_energy_spread=slices.relative_energy_spreads,
            normalized_emittance=slices.normalized_emittances,
            beta_function=slices.beta_functions,
            alpha_function=slices.alpha_functions,
            area=scaling.beam_area,
            wake_rate=self._wake_rates(slices),
        )

        def run(row: _SliceInput) -> _SliceOutput:
            return self._slice(root, row)

        output = jax.lax.map(run, inputs, batch_size=self.slice_batch)
        ledger = self._ledger(slices, output, inputs.wake_rate)
        match self.transverse:
            case "one-dimensional":
                far_field, angles = None, None
            case "angular-spectrum":
                far_field, angles = self._far_field(output.exit_field)
            case _:
                assert_never(self.transverse)
        scale = self.lattice.scale
        frequency = 2.0 * math.pi * float(scale.speed_of_light) / self.wavelength
        return FELResult(
            positions=self.record_positions,
            angular_frequencies=frequency * self._harmonic_orders,
            power=output.power,
            bunching=output.bunching,
            beam_power=output.beam_power,
            mean_lorentz_factor=output.mean_lorentz_factor,
            relative_energy_spread=output.relative_energy_spread,
            exit_field=output.exit_field,
            far_field_intensity=far_field,
            far_field_angles=angles,
            initial_particles=output.initial,
            particles=output.final,
            ledger=ledger,
            gain=self._gain(self.record_positions, output.power, scaling),
            evidence=self._evidence(slices, output, ledger),
            harmonics=self.harmonics,
            record_count=parse(
                self.record_positions.shape[0], Size[FELRecordDim], "record_count"
            ),
            plan_id=self.plan_id,
        )


__all__ = [
    "FELEnergyLedger",
    "FELEvidence",
    "FELGainEvidence",
    "FELPlan",
    "FELResult",
    "FELSeed",
    "FELStatus",
    "FELTransverseModel",
    "FELWakeLoss",
]
