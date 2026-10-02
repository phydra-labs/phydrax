#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strong-field QED cascades as one creation-stage PIC process.

`QEDCascadeProcess` couples nonlinear Compton emission
(`NonlinearComptonPlan`) by the bound lepton species and nonlinear
Breit–Wheeler pair creation (`NonlinearBreitWheelerPlan`) by a photon species
it owns (`QEDPhotonSpeciesPlan`). It runs at the PIC ``"creation"`` stage, so
one step is gather → push → (momentum processes) → Compton emission →
Breit–Wheeler pair creation → drift and current deposit → field advance.
Within the process:

1. every emitter lepton emits (optical-depth Monte Carlo with adaptive
   subcycling) from its step-start position with its pushed momentum; the
   recoil is applied before the drift;
2. photons present at the step start, with fields gathered at their positions
   through the probe route of species ``gather_species``, decay into an
   electron and a positron created at the photon position; the pair drifts and
   deposits current in the same step (a pair is charge-neutral where it is
   created, which the runtime verifies);
3. emitted photons with energy at least ``minimum_photon_energy`` enter the
   photon bank (lighter photons are counted as untracked radiation); every
   banked photon moves ballistically, ``x ← x + c k̂ Δt``, and photons leaving
   the escape box are removed into polar-angle × energy histograms.

Identity and randomness: photons carry persistent F6 identities; a photon's
parent is its emitter's identity (in the emitter species, recorded in
``emitter``) and a pair's parent is its photon's identity (in the photon
population). Photons and pairs are allocated in canonical order (emitter,
identity, subcycle / photon identity), and every random number is addressed
by ``(step, id_hi, id_lo, event)``, so results do not depend on storage slots.
Allocation goes through the run's allocation route (`allocate_particles`) at
the creating event's position (the emitter's step-start position, the decaying
photon's position) and is atomic: when the photon bank or a pair species cannot
hold every event of the step, the process fails and the PIC step is rejected
unchanged.

Ledger: the process conserves charge and the declared quantity exactly. Its
radiation record hands the net photon energy (emitted minus decayed, tracked or
not) to ``radiated``, the created pairs' rest energy to
``created_rest_energy``, and the energy the field supplied to close collinear
kinematics (``O(m²c⁴/ε)`` per event) to ``field_exchange_energy``; the process
energy defect is the residual of that balance. It claims ``"subgrid-reaction"``
radiation ownership (so it replaces, and cannot be combined with, radiation
reaction), with the scale-separation check of the emitting leptons.

Polarization (`QEDPolarizationModel`, shared by both plans): with
``"photon-polarized"`` every banked photon carries linear Stokes parameters
``(Q, U)`` relative to a polarization axis ``⊥ k̂``, set at emission and
rotated and evolved by the Breit–Wheeler step; escaped photons add ``(Q, U)``,
re-expressed along the projection of ``polarization_reference`` onto their
polarization plane, to Stokes histograms. ``"spin-and-photon-polarized"``
also carries a rest-frame polarization vector ``S`` for every bound lepton
species, keyed by persistent identity (fresh identities are unpolarized,
``S = 0``; `polarize` sets them): each step first precesses ``S`` with the
run's pusher (`RelativisticPushPlan.precess`, anomaly
``magnetic_moment_anomaly``) from the step-start to the pushed momenta, then
evolves it radiatively through the Compton step, and created pairs receive
their sampled spins. Ledgers are unchanged: spin and polarization carry no
energy or momentum of their own.

Galilean field grids: positions are grid coordinates ``x − v_grid t``, so
photons drift by ``(c k̂ − v_grid) Δt`` (the runtime drifts created pairs
likewise), probes gather at grid positions, and every quantum parameter,
formation time and ledger term uses lab-frame fields and momenta; the escape
box is declared in grid coordinates.

Evidence reports per-event LCFA validity (formation over field-variation time),
improved-LCFA infrared events, leptons and photons in the regime where the
unmodeled one-step trident and photon splitting are not bounded, subcycling,
capacity refusals, and each bound species' and the photon bank's occupancy
with a merge request when it reaches ``merge_occupancy`` (a P6
`ParticleMergePlan` with ``minimum_occupancy`` consumes the same trigger for
the lepton species).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float, positive_integer
from ...typing import checked, Dim, Float, Float64, Int32, Scalar, UInt32
from ..particle import (
    ParticleAllocationRequest,
    ParticlePopulationPlan,
    ParticlePopulationState,
    ParticleSetPlan,
    ParticleSlotReusePolicy,
)
from ._charge_state import PICChargeState, PICSpeciesPlan, PICSpeciesState
from ._nonlinear_breit_wheeler import (
    _rotate_stokes,
    NonlinearBreitWheelerPlan,
    NonlinearBreitWheelerResult,
)
from ._nonlinear_compton import _unit, NonlinearComptonPlan, NonlinearComptonResult
from ._process import (
    AbstractPICProcess,
    allocate_particles,
    PICFieldProbe,
    PICProcessBank,
    PICProcessContext,
    PICProcessLedger,
    PICProcessRadiation,
    PICProcessResult,
    PICProcessStage,
    PICProcessStatePartition,
    RadiationOwnership,
)
from ._qed_tables import QEDEventFlag, QEDPolarizationModel
from ._types import PICParticleState


_NO_IDENTITY = np.uint32(0xFFFFFFFF)
_EPSILON = float(np.finfo(np.float64).eps)
# CODATA 2022 electron magnetic-moment anomaly a_e = (g − 2)/2.
ELECTRON_MAGNETIC_MOMENT_ANOMALY = 1.15965218059e-3


class _LeptonDim(Dim):
    """Slots of one emitter species."""


class _PhotonDim(Dim):
    """Photon-bank slots."""


class _SpaceDim(Dim):
    """Resolved spatial axes."""


class _AngleDim(Dim):
    """Polar-angle histogram bins."""


class _EnergyDim(Dim):
    """Photon-energy histogram bins."""


class QEDLeptonState(StrictModule):
    """Per-slot Monte Carlo state of one emitter species.

    ``id_hi``/``id_lo`` are the identities the optical depth and trajectory
    history belong to; a slot whose occupant changed is fresh (new optical
    depth, empty history). ``force_history`` holds the transverse Lorentz
    force at the two previous steps and ``history_count`` how many are valid.
    """

    __strict_contract__ = True

    optical_depth: Float64[_LeptonDim]
    id_hi: UInt32[_LeptonDim]
    id_lo: UInt32[_LeptonDim]
    force_history: Float64[_LeptonDim, Literal[2], Literal[3]]
    history_count: Int32[_LeptonDim]


class QEDPhotonState(StrictModule):
    """Photon bank and escape record.

    ``momentum`` is per physical photon (energy ``c|k|``); the population's
    ``mass`` is the photon's statistical weight (physical photons per
    macrophoton). ``field_history``/``history_count`` hold the transverse
    field ``E + c k̂×B`` along the path at the two previous steps. ``emitter``
    is the species index of each photon's parent. Escaped photons accumulate
    weighted number and energy per (polar angle to the escape axis, energy)
    bin; escapes outside the energy edges go to the ``unbinned`` totals.
    ``untracked_*`` accumulate emitted photons below the minimum tracked energy.
    """

    __strict_contract__ = True

    position: Float[_PhotonDim, _SpaceDim]
    momentum: Float64[_PhotonDim, Literal[3]]
    population: ParticlePopulationState
    optical_depth: Float64[_PhotonDim]
    field_history: Float64[_PhotonDim, Literal[2], Literal[3]]
    history_count: Int32[_PhotonDim]
    emitter: Int32[_PhotonDim]
    escaped_number: Float64[_AngleDim, _EnergyDim]
    escaped_energy: Float64[_AngleDim, _EnergyDim]
    escaped_unbinned_number: Float64[Scalar]
    escaped_unbinned_energy: Float64[Scalar]
    untracked_number: Float64[Scalar]
    untracked_energy: Float64[Scalar]


class QEDSpinState(StrictModule):
    """Rest-frame polarization vectors of one bound lepton species.

    ``spin`` (``|S| ≤ 1``) belongs to the identity ``id_hi``/``id_lo``; a slot
    whose occupant changed is unpolarized.
    """

    __strict_contract__ = True

    spin: Float64[_LeptonDim, Literal[3]]
    id_hi: UInt32[_LeptonDim]
    id_lo: UInt32[_LeptonDim]


class QEDPolarizationState(StrictModule):
    """Polarization carried by a polarized cascade.

    ``spins`` holds one `QEDSpinState` per bound species (``species_indices``
    order; empty without the spin model). ``photon_stokes`` are the banked
    photons' linear Stokes parameters ``(Q, U)`` relative to ``photon_axis``.
    ``escaped_stokes`` accumulates weighted ``(Q, U)`` of escaped photons in
    the escape histogram bins, relative to the projected polarization
    reference; the degree of linear polarization of a bin is
    ``|Σ w(Q, U)| / Σ w``.
    """

    __strict_contract__ = True

    spins: tuple[QEDSpinState, ...]
    photon_stokes: Float64[_PhotonDim, Literal[2]]
    photon_axis: Float64[_PhotonDim, Literal[3]]
    escaped_stokes: Float64[_AngleDim, _EnergyDim, Literal[2]]


class QEDCascadeState(StrictModule):
    """Process state: emitter lepton states, the photon bank, and history times.

    ``history_times`` are the times of the two previous accepted steps (the
    abscissae of the trajectory histories). ``polarization`` is ``None`` for
    the unpolarized model.
    """

    __strict_contract__ = True

    leptons: tuple[QEDLeptonState, ...]
    photons: QEDPhotonState
    history_times: Float64[Literal[2]]
    polarization: QEDPolarizationState | None


def _photon_population(
    capacity: int, dimension: int, reuse_policy: ParticleSlotReusePolicy, /
) -> ParticlePopulationPlan:
    """Unit-weight photon slots, every one allocatable in one request."""
    support = ParticleSetPlan(
        np.arange(capacity), np.ones((capacity,)), ambient_dimension=dimension
    ).prepare()
    return ParticlePopulationPlan(
        support, reuse_policy=reuse_policy, allocation_capacity=capacity
    )


class QEDPhotonSpeciesPlan(StrictModule, NonTrainableState):
    """Fixed-capacity photon species with persistent identities and an escape box.

    Photons live in ``dimension`` resolved axes; the escape box
    ``[escape_lower, escape_upper]`` bounds the tracked region. Escaped photons
    are binned in ``angle_bins`` uniform polar-angle bins on ``[0, π]`` about
    the unit ``escape_axis`` and in the photon-energy bins ``energy_edges``
    (increasing, in scale energy units).
    """

    population: ParticlePopulationPlan
    dimension: int = eqx.field(static=True)
    escape_lower: tuple[float, ...] = eqx.field(static=True)
    escape_upper: tuple[float, ...] = eqx.field(static=True)
    escape_axis: tuple[float, float, float] = eqx.field(static=True)
    angle_bins: int = eqx.field(static=True)
    energy_edges: tuple[float, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        capacity: int,
        dimension: int,
        /,
        *,
        escape_lower: Sequence[float],
        escape_upper: Sequence[float],
        energy_edges: Sequence[float],
        escape_axis: Sequence[float] = (1.0, 0.0, 0.0),
        angle_bins: int = 18,
        reuse_policy: ParticleSlotReusePolicy = (
            ParticleSlotReusePolicy.REUSE_WITH_INCARNATION
        ),
    ) -> None:
        count = positive_integer(capacity, "capacity")
        axes = positive_integer(dimension, "dimension")
        if axes > 3:
            raise ValueError("dimension must be at most three.")
        lower = tuple(float(value) for value in escape_lower)
        upper = tuple(float(value) for value in escape_upper)
        if len(lower) != axes or len(upper) != axes:
            raise ValueError("The escape box must have one bound per axis.")
        if not all(
            math.isfinite(a) and math.isfinite(b) and a < b
            for a, b in zip(lower, upper, strict=True)
        ):
            raise ValueError("The escape box bounds must be finite and increasing.")
        edges = tuple(float(value) for value in energy_edges)
        if len(edges) < 2 or not all(
            math.isfinite(value) and value >= 0.0 for value in edges
        ):
            raise ValueError("energy_edges must hold at least two finite values ≥ 0.")
        if any(b <= a for a, b in zip(edges[:-1], edges[1:], strict=True)):
            raise ValueError("energy_edges must be increasing.")
        direction = tuple(float(value) for value in escape_axis)
        norm = math.sqrt(sum(value * value for value in direction))
        if len(direction) != 3 or not math.isfinite(norm) or norm == 0.0:
            raise ValueError("escape_axis must be a finite nonzero 3-vector.")
        bins = positive_integer(angle_bins, "angle_bins")
        self.population = _photon_population(count, axes, reuse_policy)
        self.dimension = axes
        self.escape_lower = lower
        self.escape_upper = upper
        self.escape_axis = (
            direction[0] / norm,
            direction[1] / norm,
            direction[2] / norm,
        )
        self.angle_bins = bins
        self.energy_edges = edges
        self.plan_id = canonical_fingerprint(
            {
                "kind": "qed-photon-species",
                "population": self.population.plan_id,
                "dimension": axes,
                "escape_lower": list(lower),
                "escape_upper": list(upper),
                "escape_axis": list(self.escape_axis),
                "angle_bins": bins,
                "energy_edges": list(edges),
            }
        )

    @property
    def capacity(self) -> int:
        return self.population.particles.capacity

    def initialize(self, dtype: DTypeLike, /) -> QEDPhotonState:
        """Empty photon bank with positions of ``dtype``."""
        capacity = self.capacity
        population = self.population.initialize(
            active_mask=jnp.zeros((capacity,), dtype=jnp.bool_),
            masses=jnp.zeros((capacity,)),
        )
        histogram = jnp.zeros((self.angle_bins, len(self.energy_edges) - 1))
        zero = jnp.zeros(())
        return QEDPhotonState(
            jnp.zeros((capacity, self.dimension), dtype=dtype),
            jnp.zeros((capacity, 3)),
            population,
            jnp.zeros((capacity,)),
            jnp.zeros((capacity, 2, 3)),
            jnp.zeros((capacity,), dtype=jnp.int32),
            jnp.full((capacity,), -1, dtype=jnp.int32),
            histogram,
            histogram,
            zero,
            zero,
            zero,
            zero,
        )


class QEDCascadeEvidence(StrictModule):
    """One cascade step.

    Counts: ``emissions`` (photons emitted, tracked or not), ``tracked`` (entered
    the bank), ``decays`` (pairs created), ``escaped`` photons,
    ``kinematics_refused`` draws, events ``outside_lcfa_validity`` and
    ``infrared_corrected``, leptons flagged ``one_step_trident`` and photons
    flagged ``photon_splitting``. ``maximum_subcycles``,
    ``maximum_event_probability``, ``maximum_lepton_chi`` and
    ``maximum_photon_chi`` summarize the Monte Carlo resolution and support.
    ``occupancy`` is the active fraction of every bound species (in
    ``species_indices`` order) followed by the photon bank, and
    ``merge_requested`` where it reached ``merge_occupancy``.
    ``photon_capacity_refused``/``pair_capacity_refused`` mark atomic
    allocation refusals. Energies and momenta are totals over physical
    particles: ``emitted_energy``, ``absorbed_energy`` (decayed photons),
    ``created_rest_energy``, ``field_exchange_energy`` and
    ``field_exchange_momentum``. ``compton`` holds each emitter's per-particle
    result and ``breit_wheeler`` the per-photon result (``None`` without pair
    creation).
    """

    emissions: Array
    tracked: Array
    decays: Array
    escaped: Array
    kinematics_refused: Array
    outside_lcfa_validity: Array
    infrared_corrected: Array
    one_step_trident: Array
    photon_splitting: Array
    maximum_subcycles: Array
    maximum_event_probability: Array
    maximum_lepton_chi: Array
    maximum_photon_chi: Array
    occupancy: Array
    merge_requested: Array
    photon_capacity_refused: Array
    pair_capacity_refused: Array
    emitted_energy: Array
    absorbed_energy: Array
    created_rest_energy: Array
    field_exchange_energy: Array
    field_exchange_momentum: Array
    compton: tuple[NonlinearComptonResult, ...]
    breit_wheeler: NonlinearBreitWheelerResult | None
    successful: Array


def _variation_time(
    current: Array,
    history: Array,
    count: Array,
    times: Array,
    time: Array,
    /,
) -> Array:
    """``τ = 2√(F²/(Ḟ² + |F·F̈|))`` from the two previous samples (``inf`` otherwise).

    Di Piazza et al. 2019, eq. (10), with backward differences on the history
    abscissae; constant fields (``Ḟ² + |F·F̈|`` below roundoff) give ``inf``.
    """
    first = time - times[0]
    second = times[0] - times[1]
    safe_first = jnp.where(first > 0.0, first, 1.0)
    safe_second = jnp.where(second > 0.0, second, 1.0)
    rate = (current - history[:, 0]) / safe_first
    previous_rate = (history[:, 0] - history[:, 1]) / safe_second
    curvature = 2.0 * (rate - previous_rate) / (safe_first + safe_second)
    magnitude = jnp.sum(current**2, axis=-1)
    variation = jnp.sum(rate**2, axis=-1) + jnp.abs(jnp.sum(current * curvature, axis=-1))
    valid = (
        (count >= 2)
        & (first > 0.0)
        & (second > 0.0)
        & (variation > _EPSILON * magnitude / safe_first**2)
    )
    return jnp.where(
        valid, 2.0 * jnp.sqrt(magnitude / jnp.where(valid, variation, 1.0)), jnp.inf
    )


def _canonical_rows(valid: Array, keys: tuple[Array, ...], width: int, /) -> Array:
    """First ``width`` row indices of the valid rows in lexicographic key order."""
    order = jnp.lexsort(tuple(reversed(keys)) + (~valid,))
    return order[:width]


def _total(value: Array, /) -> Array:
    """Sum of per-device rows in the rows' dtype."""
    return jnp.sum(value, axis=0, dtype=value.dtype)


def _joined_slots(value: Array, /) -> Array:
    """Per-device slot rows ``[P, c, ...]`` joined as ``[P·c, ...]``."""
    return value.reshape((-1, *value.shape[2:]))


def _fixed_ratio(plan: PICSpeciesPlan, /) -> float:
    model = plan.charge_model
    if model.minimum_charge_number != model.maximum_charge_number:
        raise ValueError(f"Species {plan.species_id!r} must have a fixed charge number.")
    return model.base_specific_charge * model.initial_charge_number


def _require_ratio(plan: PICSpeciesPlan, expected: float, role: str, /) -> None:
    specific = _fixed_ratio(plan)
    if abs(specific - expected) > 1.0e-12 * abs(expected):
        raise ValueError(
            f"Species {plan.species_id!r} charge-to-mass ratio {specific!r} differs "
            f"from the {role} ratio {expected!r}."
        )


def _allocate_pairs(
    context: PICProcessContext,
    plan: PICSpeciesPlan,
    state: PICSpeciesState,
    rows: Array,
    valid: Array,
    position: Array,
    momentum: Array,
    weight: Array,
    parent_hi: Array,
    parent_lo: Array,
    lepton_mass: float,
    /,
) -> tuple[PICSpeciesState, Array, Array]:
    """Allocate canonically ordered pair members at their photons' positions.

    Returns the state, success, and each row's slot (``capacity`` if none).
    """
    width = rows.shape[0]
    selected = valid[rows]
    particles = state.particles
    origin = position[rows].astype(particles.position.dtype)
    allocation = allocate_particles(
        context,
        plan.population,
        state.population,
        ParticleAllocationRequest(
            jnp.arange(width, dtype=jnp.int32),
            (weight[rows] * lepton_mass).astype(state.population.mass.dtype),
            selected,
            parents=(parent_hi[rows], parent_lo[rows]),
        ),
        origin,
    )
    capacity = state.population.active.shape[0]
    slots = jnp.where(allocation.allocated, allocation.slots, capacity)
    charge = state.charge
    created = PICSpeciesState(
        PICParticleState(
            particles.position.at[slots].set(origin, mode="drop"),
            particles.proper_velocity.at[slots].set(
                (momentum[rows] / lepton_mass).astype(particles.proper_velocity.dtype),
                mode="drop",
            ),
        ),
        allocation.candidate_state,
        PICChargeState(
            charge.charge_number.at[slots].set(
                jnp.asarray(
                    plan.charge_model.initial_charge_number,
                    dtype=charge.charge_number.dtype,
                ),
                mode="drop",
            ),
            charge.transition_count.at[slots].set(0, mode="drop"),
            charge.last_transition_step.at[slots].set(-1, mode="drop"),
        ),
    )
    return created, allocation.successful, slots


class _Emissions(NamedTuple):
    """Flattened emission rows of all emitters in emitter order."""

    valid: Array
    tracked: Array
    position: Array
    momentum: Array
    energy: Array
    weight: Array
    parent_hi: Array
    parent_lo: Array
    emitter: Array
    slot: Array
    stokes: Array | None
    axis: Array | None


class QEDCascadeProcess(AbstractPICProcess, NonTrainableState):
    """Nonlinear Compton emission and Breit–Wheeler pair creation in a PIC run.

    ``emitters`` are the lepton species that emit through ``compton`` (their
    charge-to-mass ratio must be ``±q/m`` of the plan with a fixed charge
    number). With ``breit_wheeler``, photons decay into species ``electron``
    (ratio ``−e/m``) and ``positron`` (``+e/m``) of the pair plan's leptons, and
    the photon fields are gathered through species ``gather_species``' route
    (the photon capacity must be a multiple of its capacity). Emitted photons
    below ``minimum_photon_energy`` are not banked. ``merge_occupancy`` is the
    occupancy fraction at which a merge is requested. The polarization model
    is the plans' (they must agree); ``magnetic_moment_anomaly`` is the bound
    leptons' ``a = (g − 2)/2`` for spin precession and
    ``polarization_reference`` the lab vector whose projection on each escaped
    photon's polarization plane is the Stokes reference axis.
    """

    compton: NonlinearComptonPlan
    breit_wheeler: NonlinearBreitWheelerPlan | None
    photons: QEDPhotonSpeciesPlan
    emitters: tuple[int, ...] = eqx.field(static=True)
    electron: int = eqx.field(static=True)
    positron: int = eqx.field(static=True)
    gather_species: int = eqx.field(static=True)
    minimum_photon_energy: float = eqx.field(static=True)
    merge_occupancy: float = eqx.field(static=True)
    polarization: QEDPolarizationModel = eqx.field(static=True)
    magnetic_moment_anomaly: float = eqx.field(static=True)
    polarization_reference: tuple[float, float, float] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    stage: PICProcessStage = eqx.field(static=True)
    stochastic: bool = eqx.field(static=True)
    radiation_ownership: RadiationOwnership | None = eqx.field(static=True)
    species_indices: tuple[int, ...] = eqx.field(static=True)
    stateful: ClassVar[bool] = True

    @checked
    def __init__(
        self,
        compton: NonlinearComptonPlan,
        photons: QEDPhotonSpeciesPlan,
        /,
        *,
        emitters: Sequence[int],
        breit_wheeler: NonlinearBreitWheelerPlan | None = None,
        electron: int,
        positron: int,
        gather_species: int,
        minimum_photon_energy: float = 0.0,
        merge_occupancy: float = 0.8,
        magnetic_moment_anomaly: float = ELECTRON_MAGNETIC_MOMENT_ANOMALY,
        polarization_reference: Sequence[float] = (0.0, 1.0, 0.0),
    ) -> None:
        if breit_wheeler is not None and not isinstance(
            breit_wheeler, NonlinearBreitWheelerPlan
        ):
            raise TypeError("breit_wheeler must be NonlinearBreitWheelerPlan or None.")
        indices = tuple(emitters)
        named = (*indices, electron, positron, gather_species)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in named):
            raise TypeError("Species indices must be integers.")
        if not indices or any(value < 0 for value in named):
            raise ValueError("emitters must be a nonempty set of species indices.")
        if len(set(indices)) != len(indices):
            raise ValueError("Emitter species must be distinct.")
        if electron == positron:
            raise ValueError("electron and positron must be different species.")
        if breit_wheeler is not None and (
            breit_wheeler.scale.scale_id != compton.scale.scale_id
        ):
            raise ValueError("Compton and Breit–Wheeler plans use different scales.")
        minimum = float(minimum_photon_energy)
        if not math.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_photon_energy must be finite and nonnegative.")
        occupancy = positive_finite_float(merge_occupancy, "merge_occupancy")
        if occupancy > 1.0:
            raise ValueError("merge_occupancy must lie in (0, 1].")
        if breit_wheeler is not None and (
            breit_wheeler.polarization != compton.polarization
        ):
            raise ValueError(
                "Compton and Breit–Wheeler plans resolve different polarizations."
            )
        anomaly = float(magnetic_moment_anomaly)
        if not math.isfinite(anomaly):
            raise ValueError("magnetic_moment_anomaly must be finite.")
        reference = tuple(float(value) for value in polarization_reference)
        length = math.sqrt(sum(value * value for value in reference))
        if len(reference) != 3 or not math.isfinite(length) or length == 0.0:
            raise ValueError("polarization_reference must be a finite nonzero 3-vector.")
        self.compton = compton
        self.breit_wheeler = breit_wheeler
        self.photons = photons
        self.emitters = indices
        self.electron = electron
        self.positron = positron
        self.gather_species = gather_species
        self.minimum_photon_energy = minimum
        self.merge_occupancy = occupancy
        self.polarization = compton.polarization
        self.magnetic_moment_anomaly = anomaly
        self.polarization_reference = (
            reference[0] / length,
            reference[1] / length,
            reference[2] / length,
        )
        self.stage = "creation"
        self.stochastic = True
        self.radiation_ownership = "subgrid-reaction"
        self.species_indices = tuple(sorted({*indices, electron, positron}))
        identity: dict[str, object] = {
            "kind": "pic-qed-cascade-process",
            "compton": compton.plan_id,
            "breit_wheeler": None if breit_wheeler is None else breit_wheeler.plan_id,
            "photons": photons.plan_id,
            "emitters": list(indices),
            "electron": electron,
            "positron": positron,
            "gather_species": gather_species,
            "minimum_photon_energy": minimum,
            "merge_occupancy": occupancy,
        }
        # Only the parameters a polarization model uses enter its identity, so
        # the unpolarized process keeps the unpolarized identity and random
        # streams.
        if compton.resolves_photon_polarization:
            identity["polarization"] = compton.polarization
            identity["polarization_reference"] = list(self.polarization_reference)
        if compton.resolves_spin:
            identity["magnetic_moment_anomaly"] = anomaly
        self.process_id = canonical_fingerprint(identity)

    # -- run construction ------------------------------------------------------

    def validate_run(
        self,
        species: tuple[PICSpeciesPlan, ...],
        relativity: RelativityScaleContract,
        /,
    ) -> None:
        self.compton.validate_relativity(relativity)
        count = len(species)
        if any(index >= count for index in (*self.species_indices, self.gather_species)):
            raise ValueError("The QED cascade references a species outside the run.")
        ratio = self.compton.physical_charge / self.compton.physical_mass
        for index in self.emitters:
            if abs(abs(_fixed_ratio(species[index])) - abs(ratio)) > 1.0e-12 * abs(ratio):
                raise ValueError(
                    f"Emitter species {species[index].species_id!r} charge-to-mass "
                    f"ratio differs from ±{abs(ratio)!r}."
                )
        pairs = self.breit_wheeler
        if pairs is not None:
            lepton = pairs.lepton_charge / pairs.lepton_mass
            _require_ratio(species[self.electron], -lepton, "electron")
            _require_ratio(species[self.positron], lepton, "positron")
        route = species[self.gather_species]
        if self.photons.capacity % route.capacity:
            raise ValueError(
                "The photon capacity must be a multiple of the gather species capacity."
            )
        if route.population.particles.ambient_dimension != self.photons.dimension:
            raise ValueError("Photon and species spatial dimensions differ.")

    def initialize_state(
        self,
        species: tuple[PICSpeciesPlan, ...],
        states: tuple[PICSpeciesState, ...],
        /,
    ) -> QEDCascadeState:
        leptons = []
        for index in self.emitters:
            capacity = species[index].capacity
            leptons.append(
                QEDLeptonState(
                    jnp.zeros((capacity,)),
                    jnp.full((capacity,), _NO_IDENTITY, dtype=jnp.uint32),
                    jnp.full((capacity,), _NO_IDENTITY, dtype=jnp.uint32),
                    jnp.zeros((capacity, 2, 3)),
                    jnp.zeros((capacity,), dtype=jnp.int32),
                )
            )
        dtype = states[self.gather_species].particles.position.dtype
        return QEDCascadeState(
            tuple(leptons),
            self.photons.initialize(dtype),
            jnp.zeros((2,)),
            self._initial_polarization(species),
        )

    def _initial_polarization(
        self, species: tuple[PICSpeciesPlan, ...], /
    ) -> QEDPolarizationState | None:
        if not self.compton.resolves_photon_polarization:
            return None
        spins = tuple(
            QEDSpinState(
                jnp.zeros((species[index].capacity, 3)),
                jnp.full((species[index].capacity,), _NO_IDENTITY, dtype=jnp.uint32),
                jnp.full((species[index].capacity,), _NO_IDENTITY, dtype=jnp.uint32),
            )
            for index in (self.species_indices if self.compton.resolves_spin else ())
        )
        capacity = self.photons.capacity
        return QEDPolarizationState(
            spins,
            jnp.zeros((capacity, 2)),
            jnp.zeros((capacity, 3)),
            jnp.zeros((self.photons.angle_bins, len(self.photons.energy_edges) - 1, 2)),
        )

    def polarize(
        self,
        state: QEDCascadeState,
        species_index: int,
        species_state: PICSpeciesState,
        spin: ArrayLike,
        /,
    ) -> QEDCascadeState:
        """Process state with the polarization vectors ``spin[N, 3]`` (``|S| ≤ 1``)
        assigned to the active particles of bound species ``species_index``.

        Spin model only; the vectors belong to the particles' current
        identities (inactive slots are unpolarized).
        """
        polarization = state.polarization
        if not self.compton.resolves_spin or polarization is None:
            raise ValueError("polarize requires the 'spin-and-photon-polarized' model.")
        if species_index not in self.species_indices:
            raise ValueError("species_index is not a lepton species of the cascade.")
        position = self.species_indices.index(species_index)
        population = species_state.population
        vectors = np.asarray(spin, dtype=np.float64)
        capacity = population.active.shape[0]
        if vectors.shape != (capacity, 3):
            raise ValueError("spin must have shape [capacity, 3].")
        if not np.all(np.isfinite(vectors)) or np.any(
            np.sum(vectors**2, axis=-1) > 1.0 + 1.0e-12
        ):
            raise ValueError("Polarization vectors must be finite with |S| ≤ 1.")
        active = population.active
        spins = list(polarization.spins)
        spins[position] = QEDSpinState(
            jnp.where(active[:, None], jnp.asarray(vectors), 0.0),
            jnp.where(active, population.id_hi, _NO_IDENTITY),
            jnp.where(active, population.id_lo, _NO_IDENTITY),
        )
        return QEDCascadeState(
            state.leptons,
            state.photons,
            state.history_times,
            QEDPolarizationState(
                tuple(spins),
                polarization.photon_stokes,
                polarization.photon_axis,
                polarization.escaped_stokes,
            ),
        )

    def field_probe(self, state: QEDCascadeState, /) -> PICFieldProbe | None:
        if self.breit_wheeler is None:
            return None
        photons = state.photons
        return PICFieldProbe(
            photons.position, photons.population.active, self.gather_species
        )

    def shift_frame(
        self, state: QEDCascadeState, axis: int, distance: float, /
    ) -> QEDCascadeState:
        photons = state.photons
        moved = eqx.tree_at(
            lambda value: value.position,
            photons,
            photons.position.at[:, axis].add(
                jnp.where(photons.population.active, -distance, 0.0).astype(
                    photons.position.dtype
                )
            ),
        )
        return QEDCascadeState(
            state.leptons, moved, state.history_times, state.polarization
        )

    # -- distribution ----------------------------------------------------------

    def localize(
        self, species: tuple[PICSpeciesPlan, ...], parts: int, /
    ) -> QEDCascadeProcess:
        """The cascade bound to one device's slot blocks.

        The photon bank splits into ``parts`` equal blocks that must coincide
        with the slot blocks of species ``gather_species`` (photons gather their
        fields through its probe route). The local view keeps the run's process
        and photon-species identities, so its random streams are the run's.
        """
        capacity = self.photons.capacity
        if parts <= 0 or capacity % parts:
            raise ValueError("The photon capacity must be divisible by the part count.")
        block = capacity // parts
        if block != species[self.gather_species].capacity:
            raise ValueError(
                "The photon bank's slot blocks must coincide with the gather "
                "species' slot blocks."
            )
        population = _photon_population(
            block, self.photons.dimension, self.photons.population.reuse_policy
        )
        return eqx.tree_at(lambda process: process.photons.population, self, population)

    def bank_plans(self) -> tuple[ParticlePopulationPlan, ...]:
        """The photon bank's global population plan."""
        return (self.photons.population,)

    @checked
    def partition_state(self, state: QEDCascadeState, /) -> PICProcessStatePartition:
        """Lepton and spin states move with their species' slots, the photon bank
        with its photons; escape and untracked records are additive totals and
        the history times are shared."""
        photons = state.photons
        companions: list[tuple[Array, ...]] = [
            (
                lepton.optical_depth,
                lepton.id_hi,
                lepton.id_lo,
                lepton.force_history,
                lepton.history_count,
            )
            for lepton in state.leptons
        ]
        companion_species = list(self.emitters)
        leaves: tuple[Array, ...] = (
            photons.momentum,
            photons.optical_depth,
            photons.field_history,
            photons.history_count,
            photons.emitter,
        )
        totals: tuple[Array, ...] = (
            photons.escaped_number,
            photons.escaped_energy,
            photons.escaped_unbinned_number,
            photons.escaped_unbinned_energy,
            photons.untracked_number,
            photons.untracked_energy,
        )
        polarization = state.polarization
        if polarization is not None:
            spin_species = self.species_indices if self.compton.resolves_spin else ()
            for index, spin in zip(spin_species, polarization.spins, strict=True):
                companions.append((spin.spin, spin.id_hi, spin.id_lo))
                companion_species.append(index)
            leaves = (*leaves, polarization.photon_stokes, polarization.photon_axis)
            totals = (*totals, polarization.escaped_stokes)
        return PICProcessStatePartition(
            tuple(companions),
            (PICProcessBank(photons.population, photons.position, leaves),),
            totals,
            (state.history_times,),
            tuple(companion_species),
        )

    def assemble_state(self, partition: PICProcessStatePartition, /) -> QEDCascadeState:
        """The cascade state packed by `partition_state`."""
        emitters = len(self.emitters)
        (bank,) = partition.banks
        momentum, depth, history, count, emitter, *photon_polarization = bank.leaves
        escaped_number, escaped_energy, *records = partition.totals
        unbinned_number, unbinned_energy, untracked_number, untracked_energy = records[:4]
        (history_times,) = partition.shared
        photons = QEDPhotonState(
            bank.position,
            momentum,
            bank.population,
            depth,
            history,
            count,
            emitter,
            escaped_number,
            escaped_energy,
            unbinned_number,
            unbinned_energy,
            untracked_number,
            untracked_energy,
        )
        polarization = None
        if self.compton.resolves_photon_polarization:
            stokes, axis = photon_polarization
            (escaped_stokes,) = records[4:]
            polarization = QEDPolarizationState(
                tuple(
                    QEDSpinState(spin, id_hi, id_lo)
                    for spin, id_hi, id_lo in partition.companions[emitters:]
                ),
                stokes,
                axis,
                escaped_stokes,
            )
        return QEDCascadeState(
            tuple(QEDLeptonState(*leaves) for leaves in partition.companions[:emitters]),
            photons,
            history_times,
            polarization,
        )

    def combine(
        self, ledger: PICProcessLedger, evidence: QEDCascadeEvidence, /
    ) -> tuple[PICProcessLedger, QEDCascadeEvidence]:
        """The run's ledger and evidence from per-device rows.

        Counts, energies, momenta and the signed energy residual add; the
        charge and momentum defects (nonnegative norms) and the Monte Carlo
        maxima take the largest device value, the scale separation the
        smallest; success and scale separation require every device;
        refusals and merge requests hold when any device raised them (a
        device's block refuses allocations on its own); occupancies of the
        equal blocks average; per-particle and per-photon results join in
        slot-block order.
        """
        radiation = ledger.radiation
        if radiation is None:
            raise ValueError("A QED cascade ledger carries its radiation record.")
        combined = PICProcessLedger(
            _total(ledger.event_count),
            jnp.max(ledger.charge_defect, axis=0),
            jnp.max(ledger.momentum_defect, axis=0),
            _total(ledger.energy_defect),
            jnp.all(ledger.successful, axis=0),
            self.process_id,
            PICProcessRadiation(
                _total(radiation.radiated_energy),
                jnp.max(radiation.maximum_critical_frequency, axis=0),
                jnp.min(radiation.minimum_scale_separation, axis=0),
                jnp.all(radiation.scale_separated, axis=0),
                None
                if radiation.field_exchange_energy is None
                else _total(radiation.field_exchange_energy),
                None
                if radiation.created_rest_energy is None
                else _total(radiation.created_rest_energy),
            ),
        )
        decay = evidence.breit_wheeler
        return combined, QEDCascadeEvidence(
            _total(evidence.emissions),
            _total(evidence.tracked),
            _total(evidence.decays),
            _total(evidence.escaped),
            _total(evidence.kinematics_refused),
            _total(evidence.outside_lcfa_validity),
            _total(evidence.infrared_corrected),
            _total(evidence.one_step_trident),
            _total(evidence.photon_splitting),
            jnp.max(evidence.maximum_subcycles, axis=0),
            jnp.max(evidence.maximum_event_probability, axis=0),
            jnp.max(evidence.maximum_lepton_chi, axis=0),
            jnp.max(evidence.maximum_photon_chi, axis=0),
            jnp.mean(evidence.occupancy, axis=0),
            jnp.any(evidence.merge_requested, axis=0),
            jnp.any(evidence.photon_capacity_refused, axis=0),
            jnp.any(evidence.pair_capacity_refused, axis=0),
            _total(evidence.emitted_energy),
            _total(evidence.absorbed_energy),
            _total(evidence.created_rest_energy),
            _total(evidence.field_exchange_energy),
            _total(evidence.field_exchange_momentum),
            tuple(
                eqx.tree_at(
                    lambda value: value.successful,
                    jax.tree.map(_joined_slots, result),
                    jnp.all(result.successful),
                )
                for result in evidence.compton
            ),
            None
            if decay is None
            else eqx.tree_at(
                lambda value: value.successful,
                jax.tree.map(_joined_slots, decay),
                jnp.all(decay.successful),
            ),
            jnp.all(evidence.successful, axis=0),
        )

    # -- step ------------------------------------------------------------------

    def _emit(
        self,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        state: QEDCascadeState,
        values: list[PICSpeciesState],
        spins: dict[int, Array],
        /,
    ) -> tuple[
        tuple[QEDLeptonState, ...],
        tuple[NonlinearComptonResult, ...],
        _Emissions,
        Array,
        Array,
    ]:
        """Compton step of every emitter; returns lepton states, results, rows,
        field-exchange energy and momentum (totals over physical particles).

        ``spins`` (spin model) maps species indices to polarization vectors and
        is updated in place."""
        del species
        plan = self.compton
        key = context.key
        if key is None:
            raise ValueError("The QED cascade requires a process key.")
        slots = plan.maximum_subcycles
        lepton_states = []
        results = []
        rows: list[tuple[Array, ...]] = []
        exchange_energy = jnp.zeros(())
        exchange_momentum = jnp.zeros((3,))
        for order, index in enumerate(self.emitters):
            current = values[index]
            population = current.population
            previous = state.leptons[order]
            fresh = (previous.id_hi != population.id_hi) | (
                previous.id_lo != population.id_lo
            )
            depth = jnp.where(
                fresh,
                plan.initial_optical_depth(key, population.id_hi, population.id_lo),
                previous.optical_depth,
            )
            count = jnp.where(fresh, 0, previous.history_count)
            proper = current.particles.proper_velocity
            electric = context.electric[index]
            magnetic = context.magnetic[index]
            force = plan.transverse_force(proper, electric, magnetic)
            result = plan.apply(
                proper,
                electric,
                magnetic,
                context.step_size,
                population.active,
                depth,
                plan.uniforms(key, population.id_hi, population.id_lo),
                variation_time=_variation_time(
                    force,
                    previous.force_history,
                    count,
                    state.history_times,
                    context.time,
                ),
                grid_cutoff_frequency=context.grid_cutoff_frequency,
                spin=spins[index] if plan.resolves_spin else None,
            )
            if result.spin is not None:
                spins[index] = result.spin
            results.append(result)
            weight = jnp.where(
                population.active,
                population.mass.astype(jnp.float64) / plan.physical_mass,
                0.0,
            )
            values[index] = PICSpeciesState(
                PICParticleState(
                    current.particles.position,
                    result.proper_velocity.astype(proper.dtype),
                ),
                population,
                current.charge,
            )
            lepton_states.append(
                QEDLeptonState(
                    result.optical_depth,
                    population.id_hi,
                    population.id_lo,
                    jnp.stack((force, previous.force_history[:, 0]), axis=1),
                    jnp.minimum(count + 1, 2).astype(jnp.int32),
                )
            )
            exchange_energy = exchange_energy + jnp.sum(weight * result.field_energy)
            exchange_momentum = exchange_momentum + jnp.sum(
                weight[:, None] * result.field_momentum, axis=0
            )
            capacity = population.active.shape[0]
            emitted = result.emitted.reshape(-1)
            energy = result.photon_energy.reshape(-1)
            row = (
                emitted,
                emitted & (energy >= self.minimum_photon_energy),
                jnp.repeat(current.particles.position, slots, axis=0),
                result.photon_momentum.reshape(-1, 3),
                energy,
                jnp.repeat(weight, slots),
                jnp.repeat(population.id_hi, slots),
                jnp.repeat(population.id_lo, slots),
                jnp.full((capacity * slots,), index, dtype=jnp.int32),
                jnp.tile(jnp.arange(slots, dtype=jnp.int32), capacity),
            )
            if result.photon_stokes is not None and result.photon_axis is not None:
                row = (
                    *row,
                    result.photon_stokes.reshape(-1),
                    result.photon_axis.reshape(-1, 3),
                )
            rows.append(row)
        stacked = tuple(
            jnp.concatenate(tuple(row[column] for row in rows), axis=0)
            for column in range(len(rows[0]))
        )
        polarized = stacked[10:] if len(stacked) > 10 else (None, None)
        return (
            tuple(lepton_states),
            tuple(results),
            _Emissions(*stacked[:10], *polarized),
            exchange_energy,
            exchange_momentum,
        )

    def _escape(
        self, photons: QEDPhotonState, position: Array, active: Array, /
    ) -> tuple[QEDPhotonState, Array, tuple[Array, Array, Array, Array]]:
        """Remove photons outside the escape box into the histograms.

        Also returns the binning ``(binned, angle row, energy bin, weight)``.
        """
        plan = self.photons
        lower = jnp.asarray(plan.escape_lower, dtype=position.dtype)
        upper = jnp.asarray(plan.escape_upper, dtype=position.dtype)
        outside = active & jnp.any((position < lower) | (position > upper), axis=-1)
        momentum = photons.momentum
        size = jnp.sqrt(jnp.sum(momentum**2, axis=-1))
        energy = self.compton.speed_of_light * size
        cosine = jnp.sum(momentum * jnp.asarray(plan.escape_axis), axis=-1) / jnp.where(
            size > 0.0, size, 1.0
        )
        angle = jnp.arccos(jnp.clip(cosine, -1.0, 1.0))
        angle_bin = jnp.clip(
            jnp.floor(angle / math.pi * plan.angle_bins), 0, plan.angle_bins - 1
        ).astype(jnp.int32)
        edges = jnp.asarray(plan.energy_edges)
        energy_bin = jnp.searchsorted(edges, energy, side="right").astype(jnp.int32) - 1
        binned = outside & (energy_bin >= 0) & (energy_bin < edges.shape[0] - 1)
        weight = jnp.where(outside, photons.population.mass.astype(jnp.float64), 0.0)
        rows = jnp.where(binned, angle_bin, plan.angle_bins)
        number = photons.escaped_number.at[rows, energy_bin].add(
            jnp.where(binned, weight, 0.0), mode="drop"
        )
        escaped_energy = photons.escaped_energy.at[rows, energy_bin].add(
            jnp.where(binned, weight * energy, 0.0), mode="drop"
        )
        unbinned = outside & ~binned
        population = plan.population.deactivate(
            photons.population, outside
        ).accepted_state
        return (
            QEDPhotonState(
                jnp.where(population.active[:, None], position, 0.0),
                jnp.where(population.active[:, None], momentum, 0.0),
                population,
                photons.optical_depth,
                photons.field_history,
                photons.history_count,
                photons.emitter,
                number,
                escaped_energy,
                photons.escaped_unbinned_number
                + jnp.sum(jnp.where(unbinned, weight, 0.0)),
                photons.escaped_unbinned_energy
                + jnp.sum(jnp.where(unbinned, weight * energy, 0.0)),
                photons.untracked_number,
                photons.untracked_energy,
            ),
            jnp.sum(outside, dtype=jnp.int32),
            (binned, rows, energy_bin, weight),
        )

    def _escaped_stokes(
        self,
        histogram: Array,
        stokes: Array,
        axis: Array,
        momentum: Array,
        bins: tuple[Array, Array, Array, Array],
        /,
    ) -> Array:
        """Add escaped photons' ``(Q, U)`` along the projected reference to the bins.

        Photons travelling exactly along the reference keep their own axis.
        """
        binned, rows, energy_bin, weight = bins
        direction = _unit(momentum)
        reference = jnp.asarray(self.polarization_reference)
        projected = _unit(
            reference - jnp.sum(direction * reference, axis=-1, keepdims=True) * direction
        )
        target = jnp.where(
            jnp.any(projected != 0.0, axis=-1, keepdims=True), projected, axis
        )
        rotated = _rotate_stokes(stokes, axis, target, momentum)
        return histogram.at[rows, energy_bin].add(
            jnp.where(binned[:, None], weight[:, None] * rotated, 0.0), mode="drop"
        )

    def _precess(
        self,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        state: QEDCascadeState,
        /,
    ) -> dict[int, Array]:
        """Bound species' polarization vectors after this step's pusher precession
        (spin model; empty otherwise). Fresh identities start unpolarized."""
        polarization = state.polarization
        if not self.compton.resolves_spin or polarization is None:
            return {}
        pusher = context.pusher
        start = context.step_start_proper_velocity
        if pusher is None or start is None:
            raise ValueError(
                "The spin model requires the run's pusher and step-start momenta."
            )
        precessed = {}
        for position, index in enumerate(self.species_indices):
            current = context.species[index]
            population = current.population
            stored = polarization.spins[position]
            fresh = (stored.id_hi != population.id_hi) | (
                stored.id_lo != population.id_lo
            )
            precessed[index] = pusher.precess(
                jnp.where(fresh[:, None], 0.0, stored.spin),
                start[index].astype(jnp.float64),
                current.particles.proper_velocity.astype(jnp.float64),
                context.electric[index].astype(jnp.float64),
                context.magnetic[index].astype(jnp.float64),
                species[index].specific_charge(current).astype(jnp.float64),
                self.magnetic_moment_anomaly,
                population.active,
                jnp.asarray(context.step_size, dtype=jnp.float64),
            )
        return precessed

    def _next_polarization(
        self,
        values: list[PICSpeciesState],
        polarization: QEDPolarizationState | None,
        decay: NonlinearBreitWheelerResult | None,
        emissions: _Emissions,
        spins: dict[int, Array],
        rows: Array,
        slots: Array,
        momentum: Array,
        bins: tuple[Array, Array, Array, Array],
        /,
    ) -> QEDPolarizationState | None:
        """Photon Stokes vectors (survivors evolved, emissions sampled), escaped
        Stokes histograms, and the bound species' spins keyed by identity."""
        if polarization is None:
            return None
        stokes = polarization.photon_stokes
        axis = polarization.photon_axis
        if decay is not None and decay.stokes is not None:
            if decay.polarization_axis is None:
                raise ValueError("A polarized decay step lost its polarization axes.")
            stokes, axis = decay.stokes, decay.polarization_axis
        if emissions.stokes is None or emissions.axis is None:
            raise ValueError("Polarized emissions lost their Stokes parameters.")
        emitted = emissions.stokes[rows]
        stokes = stokes.at[slots].set(
            jnp.stack((emitted, jnp.zeros_like(emitted)), axis=-1), mode="drop"
        )
        axis = axis.at[slots].set(emissions.axis[rows], mode="drop")
        return QEDPolarizationState(
            tuple(
                QEDSpinState(
                    spins[index],
                    values[index].population.id_hi,
                    values[index].population.id_lo,
                )
                for index in (self.species_indices if spins else ())
            ),
            stokes,
            axis,
            self._escaped_stokes(
                polarization.escaped_stokes, stokes, axis, momentum, bins
            ),
        )

    def apply(
        self,
        species: tuple[PICSpeciesPlan, ...],
        context: PICProcessContext,
        /,
    ) -> PICProcessResult:
        """Execute one atomic emission, transport, pair-creation, and ledger step.

        This fused creation-stage transaction deliberately keeps optical-depth
        renewal, identity-addressed allocation, recoil, pair lineage, capacity
        refusal, and conservation ledgers in one deterministic order. Splitting
        it across transformed helpers would permit partially committed banks or
        alter event identities.
        """
        state = context.state
        if not isinstance(state, QEDCascadeState):
            raise TypeError("The QED cascade requires its QEDCascadeState.")
        key = context.key
        if key is None:
            raise ValueError("The QED cascade requires a process key.")
        light = self.compton.speed_of_light
        dt = jnp.asarray(context.step_size, dtype=jnp.float64)
        values = list(context.species)
        before_kinetic, before_momentum, before_charge = self._moments(species, values)
        spins = self._precess(species, context, state)
        lepton_states, compton_results, emissions, exchange_energy, exchange_momentum = (
            self._emit(species, context, state, values, spins)
        )
        polarization = state.polarization
        photons = state.photons
        bank = self.photons.population
        photon_active = photons.population.active
        photon_weight = jnp.where(
            photon_active, photons.population.mass.astype(jnp.float64), 0.0
        )
        pairs = self.breit_wheeler
        decay_result = None
        decayed = jnp.zeros_like(photon_active)
        depth = photons.optical_depth
        history = photons.field_history
        count = photons.history_count
        pair_refused = jnp.asarray(False)
        created_rest = jnp.zeros(())
        if pairs is not None:
            probe = context.probe
            if probe is None:
                raise ValueError("The QED cascade requires photon field probes.")
            transverse = pairs.transverse_field(
                photons.momentum, probe.electric, probe.magnetic
            )
            decay_result = pairs.apply(
                photons.momentum,
                probe.electric,
                probe.magnetic,
                dt,
                photon_active,
                depth,
                pairs.uniforms(key, photons.population.id_hi, photons.population.id_lo),
                variation_time=_variation_time(
                    transverse, history, count, state.history_times, context.time
                ),
                stokes=None if polarization is None else polarization.photon_stokes,
                polarization_axis=(
                    None if polarization is None else polarization.photon_axis
                ),
            )
            decayed = decay_result.decayed
            depth = decay_result.optical_depth
            history = jnp.stack((transverse, history[:, 0]), axis=1)
            count = jnp.where(photon_active, jnp.minimum(count + 1, 2), count).astype(
                jnp.int32
            )
            exchange_energy = exchange_energy + jnp.sum(
                photon_weight * decay_result.field_energy
            )
            exchange_momentum = exchange_momentum + jnp.sum(
                photon_weight[:, None] * decay_result.field_momentum, axis=0
            )
            created_rest = (
                2.0 * pairs.rest_energy * jnp.sum(jnp.where(decayed, photon_weight, 0.0))
            )
            width = min(
                self.photons.capacity,
                species[self.electron].population.allocation_capacity,
                species[self.positron].population.allocation_capacity,
            )
            rows = _canonical_rows(
                decayed, (photons.population.id_hi, photons.population.id_lo), width
            )
            pair_refused = jnp.sum(decayed, dtype=jnp.int32) > width
            for index, momentum, spin in (
                (
                    self.electron,
                    decay_result.electron_momentum,
                    decay_result.electron_spin,
                ),
                (
                    self.positron,
                    decay_result.positron_momentum,
                    decay_result.positron_spin,
                ),
            ):
                values[index], allocated, pair_slots = _allocate_pairs(
                    context,
                    species[index],
                    values[index],
                    rows,
                    decayed,
                    photons.position,
                    momentum,
                    photon_weight,
                    photons.population.id_hi,
                    photons.population.id_lo,
                    pairs.lepton_mass,
                )
                if spin is not None:
                    spins[index] = (
                        spins[index].at[pair_slots].set(spin[rows], mode="drop")
                    )
                pair_refused = pair_refused | ~allocated
        absorbed = light * jnp.sum(
            jnp.where(
                decayed,
                photon_weight * jnp.sqrt(jnp.sum(photons.momentum**2, axis=-1)),
                0.0,
            )
        )
        absorbed_momentum = jnp.sum(
            jnp.where(decayed[:, None], photon_weight[:, None] * photons.momentum, 0.0),
            axis=0,
        )
        # Photon bank: retire decayed photons, then allocate tracked emissions.
        survivors = bank.deactivate(photons.population, decayed).accepted_state
        tracked = emissions.tracked
        width = min(tracked.shape[0], bank.allocation_capacity)
        rows = _canonical_rows(
            tracked,
            (emissions.emitter, emissions.parent_hi, emissions.parent_lo, emissions.slot),
            width,
        )
        emitted_position = emissions.position[rows].astype(photons.position.dtype)
        allocation = allocate_particles(
            context,
            bank,
            survivors,
            ParticleAllocationRequest(
                jnp.arange(width, dtype=jnp.int32),
                emissions.weight[rows],
                tracked[rows],
                parents=(emissions.parent_hi[rows], emissions.parent_lo[rows]),
            ),
            emitted_position,
        )
        photon_refused = (jnp.sum(tracked, dtype=jnp.int32) > width) | ~(
            allocation.successful
        )
        capacity = self.photons.capacity
        slots = jnp.where(allocation.allocated, allocation.slots, capacity)
        banked = allocation.candidate_state
        new_hi = banked.id_hi[jnp.minimum(slots, capacity - 1)]
        new_lo = banked.id_lo[jnp.minimum(slots, capacity - 1)]
        new_depth = (
            jnp.zeros((width,))
            if pairs is None
            else pairs.initial_optical_depth(key, new_hi, new_lo)
        )
        position = photons.position.at[slots].set(emitted_position, mode="drop")
        momentum = (
            jnp.where(decayed[:, None], 0.0, photons.momentum)
            .at[slots]
            .set(emissions.momentum[rows], mode="drop")
        )
        active = banked.active
        size = jnp.sqrt(jnp.sum(momentum**2, axis=-1, keepdims=True))
        direction = jnp.where(
            size > 0.0, momentum / jnp.where(size > 0.0, size, 1.0), 0.0
        )
        dimension = self.photons.dimension
        # Positions are grid coordinates: photons drift at c k̂ − v_grid.
        grid = (
            jnp.zeros((3,))
            if context.grid_velocity is None
            else jnp.asarray(context.grid_velocity, dtype=jnp.float64)
        )
        position = jnp.where(
            active[:, None],
            position
            + (light * dt * direction[:, :dimension] - dt * grid[:dimension]).astype(
                position.dtype
            ),
            position,
        )
        untracked = emissions.valid & ~tracked
        untracked_energy = jnp.sum(
            jnp.where(untracked, emissions.weight * emissions.energy, 0.0)
        )
        moved = QEDPhotonState(
            position,
            momentum,
            banked,
            depth.at[slots].set(new_depth, mode="drop"),
            history.at[slots].set(0.0, mode="drop"),
            count.at[slots].set(0, mode="drop"),
            photons.emitter.at[slots].set(emissions.emitter[rows], mode="drop"),
            photons.escaped_number,
            photons.escaped_energy,
            photons.escaped_unbinned_number,
            photons.escaped_unbinned_energy,
            photons.untracked_number
            + jnp.sum(jnp.where(untracked, emissions.weight, 0.0)),
            photons.untracked_energy + untracked_energy,
        )
        escaped_photons, escaped, bins = self._escape(moved, position, active)
        next_polarization = self._next_polarization(
            values,
            polarization,
            decay_result,
            emissions,
            spins,
            rows,
            slots,
            momentum,
            bins,
        )
        emitted_energy = jnp.sum(
            jnp.where(emissions.valid, emissions.weight * emissions.energy, 0.0)
        )
        emitted_momentum = jnp.sum(
            jnp.where(
                emissions.valid[:, None],
                emissions.weight[:, None] * emissions.momentum,
                0.0,
            ),
            axis=0,
        )
        after_kinetic, after_momentum, after_charge = self._moments(species, values)
        radiated = emitted_energy - absorbed
        energy_defect = (
            after_kinetic - before_kinetic + radiated + created_rest - exchange_energy
        )
        momentum_defect = jnp.sqrt(
            jnp.sum(
                (
                    after_momentum
                    - before_momentum
                    + emitted_momentum
                    - absorbed_momentum
                    - exchange_momentum
                )
                ** 2
            )
        )
        charge_defect = jnp.abs(after_charge - before_charge)
        compton_success = jnp.all(
            jnp.stack(tuple(result.successful for result in compton_results))
        )
        decay_success = (
            jnp.asarray(True) if decay_result is None else decay_result.successful
        )
        finite = (
            jnp.isfinite(energy_defect)
            & jnp.isfinite(momentum_defect)
            & jnp.all(jnp.isfinite(escaped_photons.momentum))
        )
        successful = (
            compton_success & decay_success & ~photon_refused & ~pair_refused & finite
        )
        next_state = QEDCascadeState(
            lepton_states,
            escaped_photons,
            jnp.stack(
                (jnp.asarray(context.time, dtype=jnp.float64), state.history_times[0])
            ),
            next_polarization,
        )
        accepted_species = tuple(
            jax.tree.map(lambda new, old: jnp.where(successful, new, old), new, old)
            for new, old in zip(values, context.species, strict=True)
        )
        accepted_state = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), next_state, state
        )
        evidence = self._evidence(
            species,
            values,
            escaped_photons,
            compton_results,
            decay_result,
            emissions,
            escaped,
            photon_refused,
            pair_refused,
            emitted_energy,
            absorbed,
            created_rest,
            exchange_energy,
            exchange_momentum,
            successful,
        )
        zero = jnp.zeros(())
        emitting = [
            result.applied & (result.quantum_parameter > 0.0)
            for result in compton_results
        ]
        return PICProcessResult(
            accepted_species,
            PICProcessLedger(
                evidence.emissions + evidence.decays,
                jnp.where(successful, charge_defect, zero),
                jnp.where(successful, momentum_defect, zero),
                jnp.where(successful, energy_defect, zero),
                successful,
                self.process_id,
                PICProcessRadiation(
                    jnp.where(successful, radiated, zero),
                    jnp.max(
                        jnp.stack(
                            tuple(
                                jnp.max(
                                    jnp.where(mask, result.critical_frequency, 0.0),
                                    initial=0.0,
                                )
                                for mask, result in zip(
                                    emitting, compton_results, strict=True
                                )
                            )
                        )
                    ),
                    jnp.min(
                        jnp.stack(
                            tuple(
                                jnp.min(
                                    jnp.where(mask, result.scale_separation, jnp.inf),
                                    initial=jnp.inf,
                                )
                                for mask, result in zip(
                                    emitting, compton_results, strict=True
                                )
                            )
                        )
                    ),
                    jnp.all(
                        jnp.stack(
                            tuple(
                                jnp.all(result.scale_separated)
                                for result in compton_results
                            )
                        )
                    ),
                    jnp.where(successful, exchange_energy, zero),
                    jnp.where(successful, created_rest, zero),
                ),
            ),
            evidence,
            accepted_state,
        )

    def _moments(
        self, species: tuple[PICSpeciesPlan, ...], values: list[PICSpeciesState], /
    ) -> tuple[Array, Array, Array]:
        """Kinetic energy, momentum, and charge of the bound species."""
        light = self.compton.speed_of_light
        kinetic = jnp.zeros(())
        momentum = jnp.zeros((3,))
        charge = jnp.zeros(())
        for index in self.species_indices:
            state = values[index]
            active = state.population.active
            mass = jnp.where(active, state.population.mass.astype(jnp.float64), 0.0)
            proper = state.particles.proper_velocity.astype(jnp.float64)
            squared = jnp.sum(proper**2, axis=-1)
            # (γ − 1)c² without cancellation.
            kinetic = kinetic + jnp.sum(
                mass * squared / (jnp.sqrt(1.0 + squared / light**2) + 1.0)
            )
            momentum = momentum + jnp.sum(mass[:, None] * proper, axis=0)
            charge = charge + jnp.sum(
                species[index].macrocharge(state).astype(jnp.float64)
            )
        return kinetic, momentum, charge

    def _evidence(
        self,
        species: tuple[PICSpeciesPlan, ...],
        values: list[PICSpeciesState],
        photons: QEDPhotonState,
        compton: tuple[NonlinearComptonResult, ...],
        decay: NonlinearBreitWheelerResult | None,
        emissions: _Emissions,
        escaped: Array,
        photon_refused: Array,
        pair_refused: Array,
        emitted_energy: Array,
        absorbed_energy: Array,
        created_rest_energy: Array,
        exchange_energy: Array,
        exchange_momentum: Array,
        successful: Array,
        /,
    ) -> QEDCascadeEvidence:
        def events(flag: QEDEventFlag) -> Array:
            return sum(
                (
                    jnp.sum((result.event_flags & int(flag)) != 0, dtype=jnp.int32)
                    for result in compton
                ),
                start=jnp.zeros((), dtype=jnp.int32),
            )

        def particles(flag: QEDEventFlag) -> Array:
            return sum(
                (
                    jnp.sum(
                        result.applied & ((result.flags & int(flag)) != 0),
                        dtype=jnp.int32,
                    )
                    for result in compton
                ),
                start=jnp.zeros((), dtype=jnp.int32),
            )

        zero = jnp.zeros((), dtype=jnp.int32)
        decays = zero if decay is None else jnp.sum(decay.decayed, dtype=jnp.int32)
        photon_flags = (
            jnp.zeros_like(photons.history_count) if decay is None else decay.flags
        )
        occupancy = jnp.stack(
            tuple(
                jnp.sum(values[index].population.active) / species[index].capacity
                for index in self.species_indices
            )
            + (jnp.sum(photons.population.active) / self.photons.capacity,)
        )
        return QEDCascadeEvidence(
            jnp.sum(emissions.valid, dtype=jnp.int32),
            jnp.sum(emissions.tracked, dtype=jnp.int32),
            decays,
            escaped,
            events(QEDEventFlag.KINEMATICS_REFUSED)
            + (
                zero
                if decay is None
                else jnp.sum(
                    (decay.flags & int(QEDEventFlag.KINEMATICS_REFUSED)) != 0,
                    dtype=jnp.int32,
                )
            ),
            events(QEDEventFlag.OUTSIDE_LCFA_VALIDITY)
            + (
                zero
                if decay is None
                else jnp.sum(
                    decay.decayed
                    & ((decay.flags & int(QEDEventFlag.OUTSIDE_LCFA_VALIDITY)) != 0),
                    dtype=jnp.int32,
                )
            ),
            events(QEDEventFlag.INFRARED_CORRECTED),
            particles(QEDEventFlag.ONE_STEP_TRIDENT),
            jnp.sum(
                (photon_flags & int(QEDEventFlag.PHOTON_SPLITTING)) != 0, dtype=jnp.int32
            ),
            jnp.max(
                jnp.stack(
                    tuple(jnp.max(result.subcycles, initial=0) for result in compton)
                )
            ),
            jnp.max(
                jnp.stack(
                    tuple(
                        jnp.max(result.event_probability, initial=0.0)
                        for result in compton
                    )
                )
            ),
            jnp.max(
                jnp.stack(
                    tuple(
                        jnp.max(
                            jnp.where(result.applied, result.quantum_parameter, 0.0),
                            initial=0.0,
                        )
                        for result in compton
                    )
                )
            ),
            jnp.zeros(())
            if decay is None
            else jnp.max(
                jnp.where(
                    photons.population.active | decay.decayed,
                    decay.quantum_parameter,
                    0.0,
                ),
                initial=0.0,
            ),
            occupancy,
            occupancy >= self.merge_occupancy,
            photon_refused,
            pair_refused,
            emitted_energy,
            absorbed_energy,
            created_rest_energy,
            exchange_energy,
            exchange_momentum,
            compton,
            decay,
            successful,
        )


__all__ = [
    "ELECTRON_MAGNETIC_MOMENT_ANOMALY",
    "QEDCascadeEvidence",
    "QEDCascadeProcess",
    "QEDCascadeState",
    "QEDLeptonState",
    "QEDPhotonSpeciesPlan",
    "QEDPhotonState",
    "QEDPolarizationState",
    "QEDSpinState",
]
