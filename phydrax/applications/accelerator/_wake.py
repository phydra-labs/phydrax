#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Wake functions, impedances, and their application to accelerator bunches.

Conventions
-----------
- ``zeta`` is the distance of the witness *behind* the source in meters; wakes
  vanish ahead of the source (causality) and are tabulated on ``[0, extent]``,
  treated as zero beyond ``extent`` (truncation, reported as evidence).
- Longitudinal wakes ``W_∥`` are in V/C: a witness charge ``q_w`` behind a source
  ``q_s`` changes energy by ``ΔE = -q_w q_s W_∥(zeta)`` (``W_∥ > 0`` is loss).
  The tabulated value at ``zeta = 0`` is the right limit ``W_∥(0⁺)``; the
  self-interaction of one slice uses ``W_∥(0⁺)/2`` (fundamental theorem of beam
  loading), so a point bunch loses ``q² W_∥(0⁺)/2``.
- Transverse wakes are in V/C/m. A dipolar wake kicks the witness along the
  source offset: ``Δp_⊥ c = q_w q_s x_s W_D(zeta)``. A quadrupolar wake kicks the
  witness along its own offset: ``Δp_⊥ c = q_w q_s x_w W_Q(zeta)``. Positive
  transverse wakes are defocusing; ``W_⊥(0⁺) = 0`` (Panofsky–Wenzel).
- Fourier convention (phasors ``exp(-iωt)``): ``Z_∥(ω) = ∫₀^∞ W_∥(τ) e^{iωτ} dτ``
  and ``Z_⊥(ω) = -i ∫₀^∞ W_⊥(τ) e^{iωτ} dτ`` with ``τ = zeta / c``, so a
  resonator has ``Z_∥ = R_s / (1 + iQ(ω_r/ω - ω/ω_r))`` and
  ``Z_⊥ = (ω_r/ω) R_⊥ / (1 + iQ(ω_r/ω - ω/ω_r))``. Frequencies are in hertz.
- Bunch reference energy, momentum, and charge are in the bound
  ``ElectromagneticScaleContract`` units; ``zeta`` and transverse coordinates
  are in its length unit; macroparticle charge is ``weight × reference_charge``.
"""

from __future__ import annotations

import math
from typing import assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import linear_interpolate
from ..._physical import ElectromagneticScaleContract
from ..._polynomial._orthogonal import legendre_rule_data
from ..._spectral._nonuniform_fourier import (
    NonuniformFourierGridEvidence,
    NonuniformFourierType3Plan,
    PreparedNonuniformFourierType3,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float, positive_integer
from ...signal import convolve
from ...typing import (
    as_host_array,
    Bool,
    Complex128,
    Dim,
    Float64,
    HostComplex128,
    HostFloat64,
    Identifier,
    Int32,
    parse,
    Scalar,
    Scope,
    Size,
)
from ._beam import _late_sign, AcceleratorBunch


WakeKind: TypeAlias = Literal[
    "longitudinal", "dipolar-x", "dipolar-y", "quadrupolar-x", "quadrupolar-y"
]
WakeUnits: TypeAlias = Literal["V/C", "V/C/m"]
WakeCausalityConvention: TypeAlias = Literal["behind-positive", "behind-negative"]
ImpedanceUnits: TypeAlias = Literal["ohm", "ohm/m"]

# Bane–Sands (1995) long-range crossover in units of the characteristic length
# s0: beyond it the classical s^(-3/2) / s^(-1/2) forms differ from the full
# formula by about 1.6 / 50**3 relative.
_RESISTIVE_WALL_LONG_RANGE_START = 50.0
_RESISTIVE_WALL_QUADRATURE_NODES = 192
_SQRT_THREE = math.sqrt(3.0)


class _WakeSampleDim(Dim, minimum=2):
    """Tabulated wake samples."""


class _FrequencyDim(Dim, minimum=1):
    """Impedance frequency samples."""


class _BinDim(Dim, minimum=1):
    """Longitudinal bins."""


class _HistoryDim(Dim, minimum=1):
    """Stored bunch passages."""


class _ParticleDim(Dim, minimum=1):
    """Bunch capacity."""


def _kind_units(kind: WakeKind, /) -> WakeUnits:
    match kind:
        case "longitudinal":
            return "V/C"
        case "dipolar-x" | "dipolar-y" | "quadrupolar-x" | "quadrupolar-y":
            return "V/C/m"
        case _:
            assert_never(kind)


def _impedance_units(kind: WakeKind, /) -> ImpedanceUnits:
    match kind:
        case "longitudinal":
            return "ohm"
        case "dipolar-x" | "dipolar-y" | "quadrupolar-x" | "quadrupolar-y":
            return "ohm/m"
        case _:
            assert_never(kind)


def _transverse_axis(kind: WakeKind, /) -> int | None:
    """Return the transverse axis (0 for x, 1 for y) or None for longitudinal."""
    match kind:
        case "longitudinal":
            return None
        case "dipolar-x" | "quadrupolar-x":
            return 0
        case "dipolar-y" | "quadrupolar-y":
            return 1
        case _:
            assert_never(kind)


def _uses_witness_offsets(kind: WakeKind, /) -> bool:
    """Quadrupolar wakes scale with the witness offset; dipolar with the source."""
    match kind:
        case "longitudinal" | "dipolar-x" | "dipolar-y":
            return False
        case "quadrupolar-x" | "quadrupolar-y":
            return True
        case _:
            assert_never(kind)


def _wake_source(kind: WakeKind, charges: Array, moments: Array, /) -> Array:
    """Binned source: charge, or charge-weighted source offsets for dipolar wakes."""
    match kind:
        case "longitudinal" | "quadrupolar-x" | "quadrupolar-y":
            return charges
        case "dipolar-x":
            return moments[..., 0]
        case "dipolar-y":
            return moments[..., 1]
        case _:
            assert_never(kind)


def _require_scale(scale: object, /) -> ElectromagneticScaleContract:
    if not isinstance(scale, ElectromagneticScaleContract):
        raise TypeError("scale must be an ElectromagneticScaleContract.")
    return scale


class _SIUnits(NamedTuple):
    """SI factors of the bound scale: multiply scale-unit values by these."""

    speed_of_light: float
    vacuum_impedance: float
    length: float
    charge: float
    momentum: float
    energy: float


def _si_units(scale: ElectromagneticScaleContract, /) -> _SIUnits:
    unit_map = scale.unit_si_map()
    velocity = unit_map["velocity"][0]
    electric_field = unit_map["electric_field"][0]
    length = unit_map["length"][0]
    current = unit_map["current"][0]
    return _SIUnits(
        float(scale.speed_of_light) * velocity,
        float(scale.vacuum_impedance) * electric_field * length / current,
        length,
        unit_map["charge"][0],
        unit_map["momentum"][0],
        unit_map["energy"][0],
    )


class WakeFunctionPlan(StrictModule, NonTrainableState):
    """Tabulated causal wake function in explicit SI units.

    ``zeta_samples`` are stored in the canonical ``"behind-positive"`` form
    (distance behind the source, meters, strictly increasing from zero);
    ``"behind-negative"`` input (Chao's ``z ≤ 0`` behind) is negated on entry.
    """

    __strict_contract__ = True

    kind: WakeKind = eqx.field(static=True)
    units: WakeUnits = eqx.field(static=True)
    causality_convention: WakeCausalityConvention = eqx.field(static=True)
    zeta_samples: Float64[_WakeSampleDim]
    wake_values: Float64[_WakeSampleDim]
    sample_count: Size[_WakeSampleDim] = eqx.field(static=True)
    extent: float = eqx.field(static=True)
    source: Identifier = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        kind: WakeKind,
        zeta_samples: ArrayLike,
        wake_values: ArrayLike,
        /,
        *,
        units: WakeUnits,
        causality_convention: WakeCausalityConvention,
        source: str = "tabulated",
    ) -> None:
        kind = parse(kind, WakeKind, "kind")
        units = parse(units, WakeUnits, "units")
        convention = parse(
            causality_convention, WakeCausalityConvention, "causality_convention"
        )
        expected_units = _kind_units(kind)
        if units != expected_units:
            raise ValueError(
                f"{kind!r} wakes carry units {expected_units!r}, not {units!r}."
            )
        provenance = str(source).strip()
        if not provenance:
            raise ValueError("source must be a non-empty string.")
        scope = Scope()
        zeta = as_host_array(
            zeta_samples, HostFloat64[_WakeSampleDim], "zeta_samples", scope=scope
        )
        values = as_host_array(
            wake_values, HostFloat64[_WakeSampleDim], "wake_values", scope=scope
        )
        if not np.all(np.isfinite(zeta)) or not np.all(np.isfinite(values)):
            raise ValueError("zeta_samples and wake_values must be finite.")
        match convention:
            case "behind-positive":
                behind = zeta
            case "behind-negative":
                behind = -zeta
            case _:
                assert_never(convention)
        if np.all(np.diff(behind) < 0.0):
            behind = behind[::-1]
            values = values[::-1]
        if not np.all(np.diff(behind) > 0.0):
            raise ValueError("zeta_samples must be strictly monotone.")
        if behind[0] != 0.0:
            raise ValueError(
                "Wake tables start at zero distance behind the source; the wake "
                "is zero ahead of the source by causality."
            )
        if expected_units == "V/C/m" and values[0] != 0.0:
            raise ValueError(
                "Transverse wakes vanish at zero distance (Panofsky–Wenzel)."
            )
        behind = np.ascontiguousarray(behind)
        values = np.ascontiguousarray(values)
        self.kind = kind
        self.units = units
        self.causality_convention = convention
        self.zeta_samples = jnp.asarray(behind)
        self.wake_values = jnp.asarray(values)
        self.sample_count = behind.shape[0]
        self.extent = float(behind[-1])
        self.source = provenance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wake-function-plan",
                "wake_kind": kind,
                "units": units,
                "causality_convention": convention,
                "arrays": array_tree_fingerprint((behind, values)),
                "source": provenance,
            }
        )

    def evaluate(self, distance: ArrayLike, /) -> Array:
        """Wake at ``distance`` behind the source; zero ahead and beyond ``extent``."""
        return linear_interpolate(
            self.zeta_samples,
            self.wake_values,
            jnp.asarray(distance, dtype=jnp.float64),
            bounds="fill",
            fill_value=0.0,
        ).values


class _BunchBinning(NamedTuple):
    """Charge, dipole moments, and geometry of one binned bunch (SI)."""

    lower: Array
    width: Array
    centers: Array
    indices: Array
    in_range: Array
    charges: Array
    bin_charge: Array
    bin_moments: Array
    offsets: Array


def _bin_bunch(
    bunch: AcceleratorBunch,
    bin_count: int,
    sign: float,
    units: _SIUnits,
    /,
) -> _BunchBinning:
    coordinates = bunch.coordinates.astype(jnp.float64)
    behind = sign * coordinates[:, 4] * units.length
    in_range = bunch.active & bunch.valid
    any_particle = jnp.any(in_range)
    head = jnp.where(any_particle, jnp.min(jnp.where(in_range, behind, jnp.inf)), 0.0)
    tail = jnp.where(any_particle, jnp.max(jnp.where(in_range, behind, -jnp.inf)), 0.0)
    # Bin centers span [head, tail] so the extreme particles sit on centers and
    # the lag between bins k apart is exactly k·width.
    width = (tail - head) / max(bin_count - 1, 1)
    lower = head - 0.5 * width
    safe_width = jnp.where(width > 0.0, width, 1.0)
    indices = jnp.clip(
        jnp.floor((behind - lower) / safe_width).astype(jnp.int32), 0, bin_count - 1
    )
    charges = jnp.where(
        in_range,
        bunch.weights.astype(jnp.float64) * bunch.reference_charge * units.charge,
        0.0,
    )
    offsets = coordinates[:, (0, 2)] * units.length
    bin_charge = jnp.zeros((bin_count,), dtype=jnp.float64).at[indices].add(charges)
    bin_moments = (
        jnp.zeros((bin_count, 2), dtype=jnp.float64)
        .at[indices]
        .add(charges[:, None] * offsets)
    )
    centers = head + jnp.arange(bin_count, dtype=jnp.float64) * width
    return _BunchBinning(
        lower,
        width,
        centers,
        indices,
        in_range,
        charges,
        bin_charge,
        bin_moments,
        offsets,
    )


class WakeMemoryState(StrictModule, NonTrainableState):
    """Bounded ring of recorded bunch passages for multi-bunch/multi-turn wakes.

    Each stored passage keeps its binned charge (C), transverse dipole moments
    (C·m), and absolute longitudinal positions ``c·t_arrival + zeta`` (m). The
    oldest passage is overwritten once ``history_length`` passages are stored.
    """

    __strict_contract__ = True

    positions: Float64[_HistoryDim, _BinDim]
    charges: Float64[_HistoryDim, _BinDim]
    moments: Float64[_HistoryDim, _BinDim, Literal[2]]
    valid: Bool[_HistoryDim]
    cursor: Int32[Scalar]
    passage_count: Int32[Scalar]
    history_length: Size[_HistoryDim] = eqx.field(static=True)
    bin_count: Size[_BinDim] = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        charges: ArrayLike,
        moments: ArrayLike,
        valid: ArrayLike,
        cursor: ArrayLike,
        passage_count: ArrayLike,
        /,
    ) -> None:
        positions_ = jnp.asarray(positions, dtype=jnp.float64)
        if positions_.ndim != 2:
            raise ValueError("positions must have shape (history_length, bin_count).")
        self.positions = positions_
        self.charges = jnp.asarray(charges, dtype=jnp.float64)
        self.moments = jnp.asarray(moments, dtype=jnp.float64)
        self.valid = jnp.asarray(valid, dtype=jnp.bool_)
        self.cursor = jnp.asarray(cursor, dtype=jnp.int32)
        self.passage_count = jnp.asarray(passage_count, dtype=jnp.int32)
        self.history_length = positions_.shape[0]
        self.bin_count = positions_.shape[1]

    @classmethod
    def empty(cls, history_length: int, bin_count: int, /) -> WakeMemoryState:
        history = positive_integer(history_length, "history_length")
        bins = positive_integer(bin_count, "bin_count")
        return cls(
            jnp.zeros((history, bins), dtype=jnp.float64),
            jnp.zeros((history, bins), dtype=jnp.float64),
            jnp.zeros((history, bins, 2), dtype=jnp.float64),
            jnp.zeros((history,), dtype=jnp.bool_),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )


def record_passage(
    memory: WakeMemoryState,
    bunch: AcceleratorBunch,
    scale: ElectromagneticScaleContract,
    /,
    *,
    arrival_time: ArrayLike,
) -> WakeMemoryState:
    """Store one bunch passage (reference particle at ``arrival_time`` seconds)."""
    if not isinstance(memory, WakeMemoryState):
        raise TypeError("memory must be a WakeMemoryState.")
    if not isinstance(bunch, AcceleratorBunch):
        raise TypeError("bunch must be an AcceleratorBunch.")
    units = _si_units(_require_scale(scale))
    binning = _bin_bunch(bunch, memory.bin_count, _late_sign(bunch.convention), units)
    reference = jnp.asarray(arrival_time, dtype=jnp.float64) * units.speed_of_light
    slot = memory.cursor
    return WakeMemoryState(
        memory.positions.at[slot].set(reference + binning.centers),
        memory.charges.at[slot].set(binning.bin_charge),
        memory.moments.at[slot].set(binning.bin_moments),
        memory.valid.at[slot].set(True),
        (slot + 1) % memory.history_length,
        memory.passage_count + 1,
    )


class WakeKickResult(StrictModule, NonTrainableState):
    """Kicked bunch with binned potentials and wake evidence.

    ``wake_potential`` is the total potential per bin (self plus memory) in volts
    for longitudinal and dipolar wakes and in V/m for quadrupolar wakes;
    ``self_potential`` is its single-passage causal part. ``kicks`` are the
    increments of ``delta`` (longitudinal) or ``px/p0``/``py/p0`` (transverse).
    ``energy_change`` (J) and ``loss_factor = -energy_change / Q²`` (V/C) are
    reported for longitudinal wakes; ``momentum_kick_sum`` (J) is the net
    transverse impulse ``Σ Δp_⊥ c`` for transverse wakes. ``causality_defect`` is
    the largest self potential ahead of the first charged bin (zero for a causal
    convolution). ``kernel_truncated`` flags bunch lags beyond the tabulated
    extent; ``history_sufficient`` is false when a dropped passage could still
    lie inside the tabulated extent.
    """

    __strict_contract__ = True

    bunch: AcceleratorBunch
    bin_lower_edges: Float64[_BinDim]
    bin_width: Float64[Scalar]
    line_density: Float64[_BinDim]
    wake_potential: Float64[_BinDim]
    self_potential: Float64[_BinDim]
    kicks: Float64[_ParticleDim]
    energy_change: Float64[Scalar]
    loss_factor: Float64[Scalar]
    momentum_kick_sum: Float64[Scalar]
    causality_defect: Float64[Scalar]
    kernel_truncated: Bool[Scalar]
    history_sufficient: Bool[Scalar]
    finite: Bool[Scalar]
    kind: WakeKind = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)


def _self_kernel(plan: WakeFunctionPlan, width: Array, bin_count: int, /) -> Array:
    """Causal bin kernel ``W(k·width)`` with the half-weighted self term."""
    steps = jnp.arange(bin_count)
    # A degenerate (zero-width) binning holds every particle in bin zero; its
    # remaining lags carry no charge and are excluded from the kernel.
    excluded = (steps > 0) & (width <= 0.0)
    lags = jnp.where(excluded, 0.0, steps.astype(jnp.float64) * width)
    kernel = jnp.where(excluded, 0.0, plan.evaluate(lags))
    return kernel.at[0].multiply(0.5)


def _memory_potential(
    plan: WakeFunctionPlan,
    memory: WakeMemoryState,
    witness_positions: Array,
    /,
) -> tuple[Array, Array]:
    """Potential from stored passages and whether the dropped history is inert."""
    distances = witness_positions[:, None, None] - memory.positions[None, :, :]
    kernel = plan.evaluate(distances.reshape((-1,))).reshape(distances.shape)
    source = _wake_source(plan.kind, memory.charges, memory.moments)
    weighted = jnp.where(memory.valid[None, :, None], kernel * source[None, :, :], 0.0)
    potential = jnp.sum(weighted, axis=(1, 2))
    oldest = memory.cursor
    oldest_distance = jnp.min(witness_positions) - jnp.max(memory.positions[oldest])
    dropped = memory.passage_count > memory.history_length
    sufficient = ~dropped | (oldest_distance >= plan.extent)
    return potential, sufficient


def apply_wake(
    plan: WakeFunctionPlan,
    bunch: AcceleratorBunch,
    scale: ElectromagneticScaleContract,
    /,
    *,
    bin_count: int,
    arrival_time: ArrayLike = 0.0,
    memory: WakeMemoryState | None = None,
) -> WakeKickResult:
    """Kick ``bunch`` with its own causal wake and with stored earlier passages.

    Active, valid particles are binned by their longitudinal coordinate into
    ``bin_count`` equal bins spanning the bunch (``AcceleratorConvention`` fixes
    which direction is behind). The single-passage potential is the causal
    convolution of the binned source (charge, or charge-weighted offsets for
    dipolar wakes) with the tabulated wake; ``memory`` adds the wake of stored
    passages located by ``arrival_time`` (seconds of the reference particle).
    Sources are macroparticle charges ``weight × reference_charge``; every
    physical particle of a witness macroparticle carries ``reference_charge``.
    """
    if not isinstance(plan, WakeFunctionPlan):
        raise TypeError("plan must be a WakeFunctionPlan.")
    if not isinstance(bunch, AcceleratorBunch):
        raise TypeError("bunch must be an AcceleratorBunch.")
    if memory is not None and not isinstance(memory, WakeMemoryState):
        raise TypeError("memory must be a WakeMemoryState or None.")
    bins = positive_integer(bin_count, "bin_count")
    units = _si_units(_require_scale(scale))
    binning = _bin_bunch(bunch, bins, _late_sign(bunch.convention), units)
    axis = _transverse_axis(plan.kind)
    source = _wake_source(plan.kind, binning.bin_charge, binning.bin_moments)
    kernel = _self_kernel(plan, binning.width, bins)
    self_potential = convolve(source, kernel, mode="full")[:bins]
    reference = jnp.asarray(arrival_time, dtype=jnp.float64) * units.speed_of_light
    if memory is None:
        potential = self_potential
        history_sufficient = jnp.asarray(True)
    else:
        memory_potential, history_sufficient = _memory_potential(
            plan, memory, reference + binning.centers
        )
        potential = self_potential + memory_potential

    speed_of_light = units.speed_of_light
    momentum = bunch.reference_momentum * units.momentum
    rest_energy = bunch.reference_rest_energy * units.energy
    total_energy = jnp.sqrt((momentum * speed_of_light) ** 2 + rest_energy**2)
    witness_potential = potential[binning.indices]
    # Macroparticle charges set the bunch-level ledgers; each physical particle
    # of a macroparticle carries the reference charge and receives the kick.
    particle_charge = bunch.reference_charge * units.charge
    coordinates = bunch.coordinates
    if axis is None:
        energy_change = -jnp.sum(binning.charges * witness_potential)
        kicks = (
            -particle_charge
            * witness_potential
            * total_energy
            / (momentum * speed_of_light) ** 2
        )
        column = 5
        momentum_kick_sum = jnp.asarray(0.0, dtype=jnp.float64)
    else:
        column = 2 * axis + 1
        witness_factor = (
            binning.offsets[:, axis]
            if _uses_witness_offsets(plan.kind)
            else jnp.ones_like(binning.charges)
        )
        field = witness_factor * witness_potential
        kicks = particle_charge * field / (speed_of_light * momentum)
        energy_change = jnp.asarray(0.0, dtype=jnp.float64)
        momentum_kick_sum = jnp.sum(binning.charges * field)
    kicks = jnp.where(binning.in_range, kicks, 0.0)
    total_charge = jnp.sum(binning.charges)
    loss_factor = jnp.where(
        total_charge != 0.0,
        -energy_change / jnp.where(total_charge != 0.0, total_charge, 1.0) ** 2,
        0.0,
    )
    charged_ahead = jnp.cumsum(jnp.abs(source)) > 0.0
    causality_defect = jnp.max(jnp.where(charged_ahead, 0.0, jnp.abs(self_potential)))
    kernel_truncated = (bins - 1) * binning.width > plan.extent
    finite = jnp.all(jnp.isfinite(potential)) & jnp.all(jnp.isfinite(kicks))
    updated = coordinates.at[:, column].add(kicks.astype(coordinates.dtype))
    result = AcceleratorBunch(
        jnp.where(finite, updated, coordinates),
        bunch.weights,
        bunch.particle_ids,
        active=bunch.active,
        reference_rest_energy=float(bunch.reference_rest_energy),
        reference_momentum=float(bunch.reference_momentum),
        reference_charge=float(bunch.reference_charge),
        convention=bunch.convention,
        bunch_id=f"{bunch.bunch_id}:{plan.plan_id}",
    )
    return WakeKickResult(
        result,
        binning.lower + jnp.arange(bins, dtype=jnp.float64) * binning.width,
        binning.width,
        binning.bin_charge,
        potential,
        self_potential,
        kicks,
        energy_change,
        loss_factor,
        momentum_kick_sum,
        causality_defect,
        kernel_truncated,
        history_sufficient,
        finite,
        plan.kind,
        plan.plan_id,
    )


def _trapezoid_weights(nodes: np.ndarray, /) -> np.ndarray:
    spacing = np.diff(nodes)
    weights = np.zeros_like(nodes)
    weights[:-1] += 0.5 * spacing
    weights[1:] += 0.5 * spacing
    return weights


def _type3_plan(
    source_extent: float,
    targets: np.ndarray,
    sign: int,
    tolerance: float,
    /,
) -> NonuniformFourierType3Plan:
    target_low = float(np.min(targets))
    target_high = float(np.max(targets))
    return NonuniformFourierType3Plan(
        (0.5 * source_extent,),
        (0.5 * source_extent,),
        (0.5 * (target_low + target_high),),
        # Unit slack keeps a single-target box well formed.
        (0.5 * (target_high - target_low) + 1.0,),
        tolerance=tolerance,
        sign=sign,
    )


class WakeImpedanceResult(StrictModule, NonTrainableState):
    """Impedance samples with transform support and resolution evidence.

    ``phase_step`` is ``|ω| Δτ_max``, the largest phase advance of the Fourier
    kernel between adjacent wake samples; ``supported`` requires it to be at most
    π and the target to lie inside the transform's declared box.
    """

    __strict_contract__ = True

    frequencies: Float64[_FrequencyDim]
    impedance: Complex128[_FrequencyDim]
    phase_step: Float64[_FrequencyDim]
    supported: Bool[_FrequencyDim]
    kind: WakeKind = eqx.field(static=True)
    units: ImpedanceUnits = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    evidence: NonuniformFourierGridEvidence = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)


def wake_impedance(
    plan: WakeFunctionPlan,
    frequencies: ArrayLike,
    /,
    *,
    scale: ElectromagneticScaleContract,
    tolerance: float = 1.0e-12,
) -> WakeImpedanceResult:
    """Impedance ``Z(f)`` (Ω or Ω/m) of a tabulated wake at frequencies in hertz.

    The trapezoid sum over the wake samples is evaluated with the gridded
    Type-3 nonuniform Fourier transform at relative ``tolerance``.
    """
    if not isinstance(plan, WakeFunctionPlan):
        raise TypeError("plan must be a WakeFunctionPlan.")
    units = _si_units(_require_scale(scale))
    frequency = as_host_array(frequencies, HostFloat64[_FrequencyDim], "frequencies")
    if not np.all(np.isfinite(frequency)):
        raise ValueError("frequencies must be finite.")
    angular = 2.0 * math.pi * frequency
    speed_of_light = units.speed_of_light
    delays = np.asarray(plan.zeta_samples) / speed_of_light
    weights = _trapezoid_weights(delays)
    nufft = _type3_plan(delays[-1], angular, 1, tolerance)
    prepared = PreparedNonuniformFourierType3(nufft, dtype=jnp.float64)
    transform = prepared.apply(
        jnp.asarray(delays)[:, None],
        jnp.asarray(weights) * plan.wake_values,
        jnp.asarray(angular)[:, None],
    )
    if plan.kind == "longitudinal":
        impedance = transform.values
    else:
        impedance = -1j * transform.values
    phase_step = jnp.abs(jnp.asarray(angular)) * float(np.max(np.diff(delays)))
    return WakeImpedanceResult(
        jnp.asarray(frequency),
        impedance.astype(jnp.complex128),
        phase_step,
        transform.supported & (phase_step <= math.pi),
        plan.kind,
        _impedance_units(plan.kind),
        prepared.evidence.requested_tolerance,
        prepared.evidence,
        plan.plan_id,
    )


class WakeInversionResult(StrictModule, NonTrainableState):
    """Wake recovered from impedance samples with transform support evidence.

    ``origin_value`` is the raw inverse transform at zero distance: the midpoint
    of the causal jump, ``W_∥(0⁺)/2`` for longitudinal wakes (the plan stores the
    doubled value) and zero for transverse wakes (the plan stores exactly zero).
    """

    __strict_contract__ = True

    plan: WakeFunctionPlan
    supported: Bool[Scalar]
    origin_value: Float64[Scalar]
    tolerance: float = eqx.field(static=True)
    evidence: NonuniformFourierGridEvidence = eqx.field(static=True)


def wake_from_impedance(
    kind: WakeKind,
    frequencies: ArrayLike,
    impedance: ArrayLike,
    zeta_samples: ArrayLike,
    /,
    *,
    scale: ElectromagneticScaleContract,
    tolerance: float = 1.0e-12,
) -> WakeInversionResult:
    """Recover ``W(zeta)`` from ``Z(f)`` sampled on ``f ≥ 0`` (hertz).

    ``W(τ) = (1/π) Re ∫₀^∞ Z̃(ω) e^{-iωτ} dω`` with ``Z̃ = Z_∥`` or ``i Z_⊥``, as a
    trapezoid sum over the given frequency range through the gridded Type-3
    nonuniform Fourier transform; ``zeta_samples`` use the ``"behind-positive"``
    convention.
    """
    kind = parse(kind, WakeKind, "kind")
    units = _si_units(_require_scale(scale))
    scope = Scope()
    frequency = as_host_array(
        frequencies, HostFloat64[_FrequencyDim], "frequencies", scope=scope
    )
    spectrum = as_host_array(
        impedance, HostComplex128[_FrequencyDim], "impedance", scope=scope
    )
    zeta = as_host_array(zeta_samples, HostFloat64[_WakeSampleDim], "zeta_samples")
    if (
        not np.all(np.isfinite(frequency))
        or frequency[0] < 0.0
        or not np.all(np.diff(frequency) > 0.0)
    ):
        raise ValueError("frequencies must be finite, nonnegative, and increasing.")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("impedance must be finite.")
    if zeta[0] != 0.0 or not np.all(np.diff(zeta) > 0.0):
        raise ValueError("zeta_samples must increase strictly from zero.")
    angular = 2.0 * math.pi * frequency
    weights = _trapezoid_weights(angular)
    causal_spectrum = spectrum if kind == "longitudinal" else 1j * spectrum
    delays = zeta / units.speed_of_light
    # Sources are angular frequencies; the transform box uses their extent and
    # the delays as targets with the negative exponent sign.
    nufft = NonuniformFourierType3Plan(
        (0.5 * float(angular[-1]),),
        (0.5 * float(angular[-1]) + 1.0,),
        (0.5 * float(delays[-1]),),
        (0.5 * float(delays[-1]),),
        tolerance=tolerance,
        sign=-1,
    )
    prepared = PreparedNonuniformFourierType3(nufft, dtype=jnp.float64)
    transform = prepared.apply(
        jnp.asarray(angular)[:, None],
        jnp.asarray(weights * causal_spectrum),
        jnp.asarray(delays)[:, None],
    )
    values = np.asarray(transform.values.real) / math.pi
    origin = float(values[0])
    values[0] = 2.0 * origin if kind == "longitudinal" else 0.0
    plan = WakeFunctionPlan(
        kind,
        zeta,
        values,
        units=_kind_units(kind),
        causality_convention="behind-positive",
        source="impedance-inversion",
    )
    return WakeInversionResult(
        plan,
        jnp.all(transform.supported),
        jnp.asarray(origin, dtype=jnp.float64),
        prepared.evidence.requested_tolerance,
        prepared.evidence,
    )


class ResonatorWake(StrictModule, NonTrainableState):
    """Broadband resonator ``(R, Q, f_r)`` producing closed-form wakes.

    ``shunt_impedance`` is in Ω (longitudinal) or Ω/m (transverse) and
    ``resonant_frequency`` in hertz. With ``α = ω_r/(2Q)`` and
    ``ω̄ = √(ω_r² − α²)`` (``Q > 1/2``):
    ``W_∥(τ) = (ω_r R_s/Q) e^{-ατ} [cos ω̄τ − (α/ω̄) sin ω̄τ]`` and
    ``W_⊥(τ) = (ω_r R_⊥/Q)(ω_r/ω̄) e^{-ατ} sin ω̄τ``.
    """

    kind: WakeKind = eqx.field(static=True)
    shunt_impedance: float = eqx.field(static=True)
    quality: float = eqx.field(static=True)
    resonant_frequency: float = eqx.field(static=True)
    scale: ElectromagneticScaleContract = eqx.field(static=True)
    resonator_id: str = eqx.field(static=True)

    def __init__(
        self,
        shunt_impedance: float,
        quality: float,
        resonant_frequency: float,
        /,
        *,
        kind: WakeKind,
        scale: ElectromagneticScaleContract,
    ) -> None:
        kind = parse(kind, WakeKind, "kind")
        shunt = positive_finite_float(shunt_impedance, "shunt_impedance")
        quality_ = positive_finite_float(quality, "quality")
        if quality_ <= 0.5:
            raise ValueError(
                "Resonator wakes require quality > 1/2 (underdamped resonator)."
            )
        frequency = positive_finite_float(resonant_frequency, "resonant_frequency")
        self.kind = kind
        self.shunt_impedance = shunt
        self.quality = quality_
        self.resonant_frequency = frequency
        self.scale = _require_scale(scale)
        self.resonator_id = canonical_fingerprint(
            {
                "kind": "resonator-wake",
                "wake_kind": kind,
                "shunt_impedance": shunt,
                "quality": quality_,
                "resonant_frequency": frequency,
                "scale": self.scale.scale_id,
            }
        )

    @property
    def angular_frequency(self) -> float:
        return 2.0 * math.pi * self.resonant_frequency

    @property
    def loss_factor(self) -> float:
        """Point-charge loss factor ``k = ω_r R_s / (2Q)`` (V/C), longitudinal only."""
        if self.kind != "longitudinal":
            raise ValueError(
                "Transverse resonators have no point-charge loss factor: W_⊥(0⁺) = 0."
            )
        return self.angular_frequency * self.shunt_impedance / (2.0 * self.quality)

    def wake_function(self, zeta_samples: ArrayLike, /) -> WakeFunctionPlan:
        """Closed-form wake tabulated at ``zeta_samples`` (meters behind)."""
        zeta = as_host_array(zeta_samples, HostFloat64[_WakeSampleDim], "zeta_samples")
        speed_of_light = _si_units(self.scale).speed_of_light
        delay = zeta / speed_of_light
        omega = self.angular_frequency
        decay = omega / (2.0 * self.quality)
        shifted = math.sqrt(omega**2 - decay**2)
        envelope = (omega * self.shunt_impedance / self.quality) * np.exp(-decay * delay)
        if self.kind == "longitudinal":
            values = envelope * (
                np.cos(shifted * delay) - (decay / shifted) * np.sin(shifted * delay)
            )
        else:
            values = envelope * (omega / shifted) * np.sin(shifted * delay)
        return WakeFunctionPlan(
            self.kind,
            zeta,
            values,
            units=_kind_units(self.kind),
            causality_convention="behind-positive",
            source=f"resonator:{self.resonator_id}",
        )

    def impedance(self, frequencies: ArrayLike, /) -> Array:
        """Closed-form ``Z(f)`` in Ω (longitudinal) or Ω/m (transverse)."""
        angular = 2.0 * math.pi * jnp.asarray(frequencies, dtype=jnp.float64)
        omega = self.angular_frequency
        denominator = angular * omega + 1j * self.quality * (omega**2 - angular**2)
        if self.kind == "longitudinal":
            numerator = self.shunt_impedance * angular * omega
        else:
            numerator = jnp.full_like(angular, self.shunt_impedance * omega**2)
        return (numerator / denominator).astype(jnp.complex128)


def _bane_sands_integrals(scaled: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """``∫₀^∞ x^{2m} e^{-x² u} / (x⁶ + 8) dx`` for ``m = 0, 1`` at each ``u``.

    The substitution ``x = y / √(1 + u)`` keeps the integrand's width order one
    for every ``u``; ``y = t / (1 − t)`` maps the half line onto the unit
    interval for a fixed Gauss–Legendre rule.
    """
    rule = legendre_rule_data(
        _RESISTIVE_WALL_QUADRATURE_NODES, "gauss", dtype=jnp.float64
    )
    unit_nodes = 0.5 * (np.asarray(rule.nodes) + 1.0)
    unit_weights = 0.5 * np.asarray(rule.weights)
    y = unit_nodes / (1.0 - unit_nodes)
    jacobian = unit_weights / (1.0 - unit_nodes) ** 2
    stretch = 1.0 / np.sqrt(1.0 + scaled)[:, None]
    x = y[None, :] * stretch
    common = (
        np.exp(-(x**2) * scaled[:, None]) / (x**6 + 8.0) * stretch * jacobian[None, :]
    )
    return np.sum(common, axis=1), np.sum(common * x**2, axis=1)


class ResistiveWallWake(StrictModule, NonTrainableState):
    """Round resistive-wall wake of a pipe (Bane & Sands 1995, DC conductivity).

    ``characteristic_length = (2 a² / (Z₀ σ))^{1/3}``. Samples below
    ``long_range_start`` use the full short-range formula with its integral
    evaluated by Gauss–Legendre quadrature; beyond it the classical long-range
    forms ``-(c/(4πa)) √(Z₀/(πσ)) s^{-3/2}`` (longitudinal, per length) and
    ``(c/(πa³)) √(Z₀/(πσ)) s^{-1/2}`` (dipolar, per length) apply. The relative
    mismatch of the two forms at the crossover is recorded for each kind. A
    round pipe has no quadrupolar wake.
    """

    radius: float = eqx.field(static=True)
    conductivity: float = eqx.field(static=True)
    length: float = eqx.field(static=True)
    scale: ElectromagneticScaleContract = eqx.field(static=True)
    characteristic_length: float = eqx.field(static=True)
    long_range_start: float = eqx.field(static=True)
    longitudinal_crossover_mismatch: float = eqx.field(static=True)
    transverse_crossover_mismatch: float = eqx.field(static=True)
    wall_id: str = eqx.field(static=True)

    def __init__(
        self,
        radius: float,
        conductivity: float,
        length: float,
        /,
        *,
        scale: ElectromagneticScaleContract,
    ) -> None:
        radius_ = positive_finite_float(radius, "radius")
        conductivity_ = positive_finite_float(conductivity, "conductivity")
        length_ = positive_finite_float(length, "length")
        scale_ = _require_scale(scale)
        units = _si_units(scale_)
        characteristic = (
            2.0 * radius_**2 / (units.vacuum_impedance * conductivity_)
        ) ** (1.0 / 3.0)
        self.radius = radius_
        self.conductivity = conductivity_
        self.length = length_
        self.scale = scale_
        self.characteristic_length = characteristic
        self.long_range_start = _RESISTIVE_WALL_LONG_RANGE_START * characteristic
        crossover = np.asarray([self.long_range_start])
        short_longitudinal = float(self._short_range(crossover, transverse=False)[0])
        short_transverse = float(self._short_range(crossover, transverse=True)[0])
        long_longitudinal = float(self._long_range(crossover, transverse=False)[0])
        long_transverse = float(self._long_range(crossover, transverse=True)[0])
        self.longitudinal_crossover_mismatch = abs(
            short_longitudinal - long_longitudinal
        ) / abs(long_longitudinal)
        self.transverse_crossover_mismatch = abs(
            short_transverse - long_transverse
        ) / abs(long_transverse)
        self.wall_id = canonical_fingerprint(
            {
                "kind": "resistive-wall-wake",
                "radius": radius_,
                "conductivity": conductivity_,
                "length": length_,
                "long_range_start": self.long_range_start,
                "scale": scale_.scale_id,
            }
        )

    def _short_range(self, zeta: np.ndarray, /, *, transverse: bool) -> np.ndarray:
        units = _si_units(self.scale)
        scaled = zeta / self.characteristic_length
        damped = np.exp(-scaled)
        cosine = damped * np.cos(_SQRT_THREE * scaled)
        sine = damped * np.sin(_SQRT_THREE * scaled)
        integral_0, integral_2 = _bane_sands_integrals(scaled)
        prefactor = units.vacuum_impedance * units.speed_of_light * self.length
        if transverse:
            bracket = (-cosine + _SQRT_THREE * sine) / 12.0 + (
                math.sqrt(2.0) / math.pi
            ) * integral_0
            # The two terms cancel exactly at s = 0 (Panofsky–Wenzel); quadrature
            # would otherwise leave roundoff there.
            bracket = np.where(scaled == 0.0, 0.0, bracket)
            return (
                8.0
                * prefactor
                * self.characteristic_length
                / (math.pi * self.radius**4)
                * bracket
            )
        bracket = cosine / 3.0 - (math.sqrt(2.0) / math.pi) * integral_2
        return 4.0 * prefactor / (math.pi * self.radius**2) * bracket

    def _long_range(self, zeta: np.ndarray, /, *, transverse: bool) -> np.ndarray:
        units = _si_units(self.scale)
        root = math.sqrt(units.vacuum_impedance / (math.pi * self.conductivity))
        if transverse:
            return (
                units.speed_of_light
                * self.length
                / (math.pi * self.radius**3)
                * root
                / np.sqrt(zeta)
            )
        return (
            -units.speed_of_light
            * self.length
            / (4.0 * math.pi * self.radius)
            * root
            / zeta**1.5
        )

    def wake_function(
        self, kind: WakeKind, zeta_samples: ArrayLike, /
    ) -> WakeFunctionPlan:
        """Wake of ``kind`` tabulated at ``zeta_samples`` (meters behind, from 0)."""
        kind = parse(kind, WakeKind, "kind")
        zeta = as_host_array(zeta_samples, HostFloat64[_WakeSampleDim], "zeta_samples")
        if not np.all(np.isfinite(zeta)) or np.any(zeta < 0.0):
            raise ValueError("Resistive-wall samples must be finite and nonnegative.")
        match kind:
            case "longitudinal":
                transverse = False
            case "dipolar-x" | "dipolar-y":
                transverse = True
            case "quadrupolar-x" | "quadrupolar-y":
                raise ValueError("A round resistive wall has no quadrupolar wake.")
            case _:
                assert_never(kind)
        far = zeta > self.long_range_start
        values = self._short_range(np.where(far, 0.0, zeta), transverse=transverse)
        far_values = self._long_range(np.where(far, zeta, 1.0), transverse=transverse)
        values = np.where(far, far_values, values)
        return WakeFunctionPlan(
            kind,
            zeta,
            values,
            units=_kind_units(kind),
            causality_convention="behind-positive",
            source=f"resistive-wall:{self.wall_id}",
        )


__all__ = [
    "ImpedanceUnits",
    "ResistiveWallWake",
    "ResonatorWake",
    "WakeCausalityConvention",
    "WakeFunctionPlan",
    "WakeImpedanceResult",
    "WakeInversionResult",
    "WakeKickResult",
    "WakeKind",
    "WakeMemoryState",
    "WakeUnits",
    "apply_wake",
    "record_passage",
    "wake_from_impedance",
    "wake_impedance",
]
