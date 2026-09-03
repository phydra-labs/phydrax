#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mean-field sarcomere cross-bridge and energetic kinetics.

The implemented route evolves population fractions for primed, weak-bound,
strong-bound, and rigor myosin states.  ATP, ADP, and inorganic phosphate are
bookkept together with bound nucleotides so reaction updates have explicit
adenylate and phosphoryl balance.  Oxygen tension and oxidative capacity are
inputs rather than hidden surrogates.  A stochastic molecular route has a
separate type and cannot be passed to the mean-field plan.
"""

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class SarcomereSpecies(IntEnum):
    """Cross-bridge population ordering used by every mean-field state."""

    DETACHED_PRIMED_ADP_PI = 0
    WEAK_BOUND_ADP_PI = 1
    STRONG_BOUND_ADP = 2
    RIGOR_BOUND = 3


class SarcomereStatus(IntEnum):
    """Fail-closed status for a coupled mean-field step."""

    SUCCESS = 0
    NONFINITE = 1
    INVALID_COUPLING_INPUT = 2
    NEGATIVE_POPULATION = 3
    POPULATION_BALANCE = 4
    NEGATIVE_CHEMICAL_SPECIES = 5
    SPECIES_BALANCE = 6
    POWER_BALANCE = 7
    THERMODYNAMIC_FAILURE = 8


class MeanFieldSarcomereFidelity(StrictModule, NonTrainableState):
    """Typed deterministic population-kinetics route."""

    route_id: str = eqx.field(static=True)

    def __init__(self):
        self.route_id = "cardiovascular-sarcomere-mean-field"


class StochasticMolecularSarcomereFidelity(StrictModule, NonTrainableState):
    """Typed provenance for explicitly resolved molecular ensembles.

    This is deliberately not a mode of :class:`MeanFieldSarcomerePlan`.
    Molecular solvers consume this separate contract and their own random state.
    """

    molecule_count: int = eqx.field(static=True)
    realization_count: int = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(self, molecule_count: int, realization_count: int = 1, /):
        molecules = int(molecule_count)
        realizations = int(realization_count)
        if molecules < 1 or molecules != molecule_count:
            raise ValueError("molecule_count must be a positive integer.")
        if realizations < 1 or realizations != realization_count:
            raise ValueError("realization_count must be a positive integer.")
        self.molecule_count = molecules
        self.realization_count = realizations
        self.route_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-sarcomere-stochastic-molecular",
                "molecule_count": molecules,
                "realization_count": realizations,
            }
        )


def _positive(name: str, value: float, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _nonnegative(name: str, value: float, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def _unit_interval(name: str, value: float, /) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1].")
    return result


class MeanFieldSarcomerePlan(StrictModule, NonTrainableState):
    """Reaction, force, metabolism, and balance policy for one route.

    Rates use ``1/ms``; length and velocity use ``mm`` and ``mm/ms``; stress and
    oxygen tension use ``kPa``.  Chemical amount densities use ``pmol/mm3`` and
    ``atp_free_energy`` uses kernel-energy units ``mg*mm2/ms2`` per pmol.
    """

    attachment_rate_per_ms: float = eqx.field(static=True)
    powerstroke_rate_per_ms: float = eqx.field(static=True)
    adp_release_rate_per_ms: float = eqx.field(static=True)
    atp_binding_rate_per_ms: float = eqx.field(static=True)
    calcium_half_saturation_mM: float = eqx.field(static=True)
    calcium_cooperativity: float = eqx.field(static=True)
    atp_half_saturation: float = eqx.field(static=True)
    oxidative_adp_half_saturation: float = eqx.field(static=True)
    oxidative_pi_half_saturation: float = eqx.field(static=True)
    oxygen_half_saturation_kpa: float = eqx.field(static=True)
    oxygen_kinetic_floor: float = eqx.field(static=True)
    oxygen_per_atp: float = eqx.field(static=True)
    myosin_site_density: float = eqx.field(static=True)
    atp_free_energy: float = eqx.field(static=True)
    resting_length_mm: float = eqx.field(static=True)
    overlap_width_mm: float = eqx.field(static=True)
    shortening_velocity_scale_mm_per_ms: float = eqx.field(static=True)
    lengthening_stress_limit: float = eqx.field(static=True)
    maximum_active_stress_kpa: float = eqx.field(static=True)
    rigor_stress_fraction: float = eqx.field(static=True)
    balance_tolerance: float = eqx.field(static=True)
    fidelity: MeanFieldSarcomereFidelity
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        attachment_rate_per_ms: float,
        powerstroke_rate_per_ms: float,
        adp_release_rate_per_ms: float,
        atp_binding_rate_per_ms: float,
        calcium_half_saturation_mM: float,
        calcium_cooperativity: float,
        atp_half_saturation: float,
        oxidative_adp_half_saturation: float,
        oxidative_pi_half_saturation: float,
        oxygen_half_saturation_kpa: float,
        oxygen_kinetic_floor: float,
        oxygen_per_atp: float,
        myosin_site_density: float,
        atp_free_energy: float,
        resting_length_mm: float,
        overlap_width_mm: float,
        shortening_velocity_scale_mm_per_ms: float,
        maximum_active_stress_kpa: float,
        rigor_stress_fraction: float = 0.2,
        lengthening_stress_limit: float = 1.5,
        balance_tolerance: float = 1.0e-6,
        fidelity: MeanFieldSarcomereFidelity | None = None,
    ):
        fidelity_ = MeanFieldSarcomereFidelity() if fidelity is None else fidelity
        if not isinstance(fidelity_, MeanFieldSarcomereFidelity):
            raise TypeError(
                "MeanFieldSarcomerePlan requires MeanFieldSarcomereFidelity; "
                "stochastic molecular fidelity is a distinct route."
            )
        values = {
            "attachment_rate_per_ms": _positive(
                "attachment_rate_per_ms", attachment_rate_per_ms
            ),
            "powerstroke_rate_per_ms": _positive(
                "powerstroke_rate_per_ms", powerstroke_rate_per_ms
            ),
            "adp_release_rate_per_ms": _positive(
                "adp_release_rate_per_ms", adp_release_rate_per_ms
            ),
            "atp_binding_rate_per_ms": _positive(
                "atp_binding_rate_per_ms", atp_binding_rate_per_ms
            ),
            "calcium_half_saturation_mM": _positive(
                "calcium_half_saturation_mM", calcium_half_saturation_mM
            ),
            "calcium_cooperativity": _positive(
                "calcium_cooperativity", calcium_cooperativity
            ),
            "atp_half_saturation": _positive("atp_half_saturation", atp_half_saturation),
            "oxidative_adp_half_saturation": _positive(
                "oxidative_adp_half_saturation", oxidative_adp_half_saturation
            ),
            "oxidative_pi_half_saturation": _positive(
                "oxidative_pi_half_saturation", oxidative_pi_half_saturation
            ),
            "oxygen_half_saturation_kpa": _positive(
                "oxygen_half_saturation_kpa", oxygen_half_saturation_kpa
            ),
            "oxygen_per_atp": _nonnegative("oxygen_per_atp", oxygen_per_atp),
            "myosin_site_density": _positive("myosin_site_density", myosin_site_density),
            "atp_free_energy": _positive("atp_free_energy", atp_free_energy),
            "resting_length_mm": _positive("resting_length_mm", resting_length_mm),
            "overlap_width_mm": _positive("overlap_width_mm", overlap_width_mm),
            "shortening_velocity_scale_mm_per_ms": _positive(
                "shortening_velocity_scale_mm_per_ms",
                shortening_velocity_scale_mm_per_ms,
            ),
            "lengthening_stress_limit": _positive(
                "lengthening_stress_limit", lengthening_stress_limit
            ),
            "maximum_active_stress_kpa": _positive(
                "maximum_active_stress_kpa", maximum_active_stress_kpa
            ),
            "balance_tolerance": _nonnegative("balance_tolerance", balance_tolerance),
        }
        floor = _unit_interval("oxygen_kinetic_floor", oxygen_kinetic_floor)
        rigor = _unit_interval("rigor_stress_fraction", rigor_stress_fraction)
        for name, value in values.items():
            setattr(self, name, value)
        self.oxygen_kinetic_floor = floor
        self.rigor_stress_fraction = rigor
        self.fidelity = fidelity_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-mean-field-sarcomere-plan",
                **values,
                "oxygen_kinetic_floor": floor,
                "rigor_stress_fraction": rigor,
                "fidelity": fidelity_.route_id,
            }
        )


class SarcomereCouplingInputs(StrictModule, NonTrainableState):
    """EP, mechanics, and perfusion inputs sampled on sarcomere support.

    Positive ``shortening_velocity_mm_per_ms`` means shortening and positive
    mechanical power output.  Negative values mean externally driven lengthening.
    """

    calcium_concentration_mM: Array
    sarcomere_length_mm: Array
    shortening_velocity_mm_per_ms: Array
    oxygen_tension_kpa: Array
    oxidative_capacity_pmol_per_mm3_ms: Array

    def __init__(
        self,
        calcium_concentration_mM: ArrayLike,
        sarcomere_length_mm: ArrayLike,
        shortening_velocity_mm_per_ms: ArrayLike,
        oxygen_tension_kpa: ArrayLike,
        oxidative_capacity_pmol_per_mm3_ms: ArrayLike,
        /,
    ):
        arrays = tuple(
            np.asarray(value)
            for value in (
                calcium_concentration_mM,
                sarcomere_length_mm,
                shortening_velocity_mm_per_ms,
                oxygen_tension_kpa,
                oxidative_capacity_pmol_per_mm3_ms,
            )
        )
        if any(
            not np.issubdtype(value.dtype, np.inexact) or np.iscomplexobj(value)
            for value in arrays
        ):
            raise TypeError("Sarcomere coupling inputs must be real inexact arrays.")
        broadcast = np.broadcast_arrays(*arrays)
        (
            calcium,
            length,
            velocity,
            oxygen,
            capacity,
        ) = broadcast
        self.calcium_concentration_mM = jnp.asarray(calcium)
        self.sarcomere_length_mm = jnp.asarray(length)
        self.shortening_velocity_mm_per_ms = jnp.asarray(velocity)
        self.oxygen_tension_kpa = jnp.asarray(oxygen)
        self.oxidative_capacity_pmol_per_mm3_ms = jnp.asarray(capacity)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.calcium_concentration_mM.shape


class SarcomereState(StrictModule, NonTrainableState):
    """Fixed-shape cross-bridge fractions and free nucleotide pools."""

    crossbridge_fractions: Array
    atp_pmol_per_mm3: Array
    adp_pmol_per_mm3: Array
    phosphate_pmol_per_mm3: Array
    time_ms: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        crossbridge_fractions: ArrayLike,
        atp_pmol_per_mm3: ArrayLike,
        adp_pmol_per_mm3: ArrayLike,
        phosphate_pmol_per_mm3: ArrayLike,
        time_ms: ArrayLike,
        plan_id: str,
        /,
    ):
        fractions = jnp.asarray(crossbridge_fractions)
        atp = jnp.asarray(atp_pmol_per_mm3)
        adp = jnp.asarray(adp_pmol_per_mm3)
        phosphate = jnp.asarray(phosphate_pmol_per_mm3)
        time = jnp.asarray(time_ms)
        if fractions.ndim < 1 or fractions.shape[-1] != 4:
            raise ValueError("crossbridge_fractions must end in four species.")
        if any(value.shape != fractions.shape[:-1] for value in (atp, adp, phosphate)):
            raise ValueError("Free chemical pools must match the population batch shape.")
        if time.shape != ():
            raise ValueError("time_ms must be scalar.")
        if any(
            not jnp.issubdtype(value.dtype, jnp.inexact)
            for value in (fractions, atp, adp, phosphate, time)
        ):
            raise TypeError("Sarcomere state arrays must use real inexact dtypes.")
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError("plan_id must be a nonempty stable ID.")
        self.crossbridge_fractions = fractions
        self.atp_pmol_per_mm3 = atp
        self.adp_pmol_per_mm3 = adp
        self.phosphate_pmol_per_mm3 = phosphate
        self.time_ms = time
        self.plan_id = plan_id

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.crossbridge_fractions.shape[:-1]


def initialize_sarcomere_state(
    plan: MeanFieldSarcomerePlan,
    batch_shape: tuple[int, ...],
    /,
    *,
    atp_pmol_per_mm3: float,
    adp_pmol_per_mm3: float,
    phosphate_pmol_per_mm3: float,
) -> SarcomereState:
    """Initialize all myosin sites detached/primed with explicit free pools."""

    if not isinstance(plan, MeanFieldSarcomerePlan):
        raise TypeError("plan must be MeanFieldSarcomerePlan.")
    shape = tuple(int(size) for size in batch_shape)
    if any(size < 0 for size in shape):
        raise ValueError("batch_shape entries must be nonnegative.")
    atp = _nonnegative("atp_pmol_per_mm3", atp_pmol_per_mm3)
    adp = _nonnegative("adp_pmol_per_mm3", adp_pmol_per_mm3)
    phosphate = _nonnegative("phosphate_pmol_per_mm3", phosphate_pmol_per_mm3)
    fractions = np.zeros(shape + (4,), dtype=float)
    fractions[..., SarcomereSpecies.DETACHED_PRIMED_ADP_PI] = 1.0
    return SarcomereState(
        fractions,
        np.full(shape, atp),
        np.full(shape, adp),
        np.full(shape, phosphate),
        0.0,
        plan.plan_id,
    )


def _validate_state_and_inputs(
    plan: MeanFieldSarcomerePlan,
    state: SarcomereState,
    inputs: SarcomereCouplingInputs,
    /,
) -> None:
    if not isinstance(plan, MeanFieldSarcomerePlan):
        raise TypeError("plan must be MeanFieldSarcomerePlan.")
    if not isinstance(state, SarcomereState):
        raise TypeError("state must be SarcomereState.")
    if not isinstance(inputs, SarcomereCouplingInputs):
        raise TypeError("inputs must be SarcomereCouplingInputs.")
    if state.plan_id != plan.plan_id:
        raise ValueError("Sarcomere state belongs to a different mean-field plan.")
    if inputs.batch_shape != state.batch_shape:
        raise ValueError("Coupling input support must match sarcomere state support.")


class SarcomereReactionExtents(StrictModule, NonTrainableState):
    """Fractions advanced by each transition and oxidative amount regenerated."""

    attachment_fraction: Array
    powerstroke_fraction: Array
    adp_release_fraction: Array
    atp_binding_fraction: Array
    atp_regenerated_pmol_per_mm3: Array


class SarcomereKineticModulation(StrictModule, NonTrainableState):
    calcium_activation: Array
    length_overlap: Array
    force_velocity: Array
    oxygen_limitation: Array
    kinetic_oxygen_factor: Array
    atp_saturation: Array


class SarcomereOutputs(StrictModule, NonTrainableState):
    """Mechanics/metabolism coupling outputs from one candidate step."""

    active_stress_kpa: Array
    shortening_strain_rate_per_ms: Array
    oxygen_consumption_pmol_per_mm3_ms: Array
    atp_consumption_pmol_per_mm3_ms: Array
    atp_regeneration_pmol_per_mm3_ms: Array
    modulation: SarcomereKineticModulation


class SarcomerePowerLedger(StrictModule, NonTrainableState):
    """Chemical storage, chemical release, work, and heat in ``kPa/ms``."""

    chemical_storage_before_kpa: Array
    chemical_storage_after_kpa: Array
    chemical_storage_rate_kpa_per_ms: Array
    metabolic_power_input_kpa_per_ms: Array
    powerstroke_release_kpa_per_ms: Array
    mechanical_power_output_kpa_per_ms: Array
    heat_power_kpa_per_ms: Array
    chemical_balance_residual_kpa_per_ms: Array
    total_power_balance_residual_kpa_per_ms: Array
    thermodynamically_admissible: Array


class SarcomereStepEvidence(StrictModule, NonTrainableState):
    population_sum_error: Array
    minimum_population: Array
    minimum_chemical_species: Array
    adenylate_before: Array
    adenylate_after: Array
    adenylate_balance_error: Array
    phosphoryl_before: Array
    phosphoryl_after: Array
    phosphoryl_balance_error: Array
    maximum_chemical_power_residual: Array
    maximum_total_power_residual: Array
    coupling_valid: Array
    source_valid: Array
    candidate_valid: Array
    species_balanced: Array
    power_balanced: Array
    finite: Array
    passed: Array
    status: Array


class SarcomereStepResult(StrictModule, NonTrainableState):
    """Candidate, evidence, and atomically accepted-or-source state."""

    state: SarcomereState
    candidate: SarcomereState
    extents: SarcomereReactionExtents
    outputs: SarcomereOutputs
    ledger: SarcomerePowerLedger
    evidence: SarcomereStepEvidence
    accepted: Array


def _fraction_validity(state: SarcomereState, tolerance: float, /) -> Array:
    fractions = state.crossbridge_fractions
    return (
        jnp.all(jnp.isfinite(fractions))
        & jnp.all(fractions >= -tolerance)
        & jnp.all(jnp.abs(jnp.sum(fractions, axis=-1) - 1.0) <= tolerance)
        & jnp.all(jnp.isfinite(state.atp_pmol_per_mm3))
        & jnp.all(jnp.isfinite(state.adp_pmol_per_mm3))
        & jnp.all(jnp.isfinite(state.phosphate_pmol_per_mm3))
        & jnp.all(state.atp_pmol_per_mm3 >= -tolerance)
        & jnp.all(state.adp_pmol_per_mm3 >= -tolerance)
        & jnp.all(state.phosphate_pmol_per_mm3 >= -tolerance)
        & jnp.isfinite(state.time_ms)
    )


def _chemical_invariants(
    plan: MeanFieldSarcomerePlan, state: SarcomereState, /
) -> tuple[Array, Array]:
    fractions = state.crossbridge_fractions
    detached = fractions[..., SarcomereSpecies.DETACHED_PRIMED_ADP_PI]
    weak = fractions[..., SarcomereSpecies.WEAK_BOUND_ADP_PI]
    strong = fractions[..., SarcomereSpecies.STRONG_BOUND_ADP]
    bound_adenylate = plan.myosin_site_density * (detached + weak + strong)
    bound_phosphate = plan.myosin_site_density * (detached + weak)
    adenylate = state.atp_pmol_per_mm3 + state.adp_pmol_per_mm3 + bound_adenylate
    phosphoryl = state.atp_pmol_per_mm3 + state.phosphate_pmol_per_mm3 + bound_phosphate
    return adenylate, phosphoryl


def step_mean_field_sarcomere(
    plan: MeanFieldSarcomerePlan,
    state: SarcomereState,
    inputs: SarcomereCouplingInputs,
    step_size_ms: float,
    /,
) -> SarcomereStepResult:
    """Advance one positivity-preserving population/reaction candidate.

    Every reaction has one source population and uses an exponential outflow
    fraction.  ATP binding and oxidative regeneration are additionally limited
    by available chemical pools.  Any failed balance or admissibility check
    rejects the whole candidate and returns the source state unchanged.
    """

    _validate_state_and_inputs(plan, state, inputs)
    step = _positive("step_size_ms", step_size_ms)
    tolerance = plan.balance_tolerance
    calcium = inputs.calcium_concentration_mM
    length = inputs.sarcomere_length_mm
    velocity = inputs.shortening_velocity_mm_per_ms
    oxygen = inputs.oxygen_tension_kpa
    oxidative_capacity = inputs.oxidative_capacity_pmol_per_mm3_ms
    calcium_nonnegative = jnp.maximum(calcium, 0.0)
    calcium_power = calcium_nonnegative**plan.calcium_cooperativity
    half_power = plan.calcium_half_saturation_mM**plan.calcium_cooperativity
    calcium_activation = calcium_power / jnp.maximum(
        calcium_power + half_power, jnp.finfo(calcium_power.dtype).tiny
    )
    length_overlap = jnp.exp(
        -0.5 * ((length - plan.resting_length_mm) / plan.overlap_width_mm) ** 2
    )
    shortening_factor = 1.0 / (
        1.0 + jnp.maximum(velocity, 0.0) / plan.shortening_velocity_scale_mm_per_ms
    )
    lengthening_factor = jnp.minimum(
        1.0 - jnp.minimum(velocity, 0.0) / plan.shortening_velocity_scale_mm_per_ms,
        plan.lengthening_stress_limit,
    )
    force_velocity = jnp.where(velocity >= 0.0, shortening_factor, lengthening_factor)
    oxygen_nonnegative = jnp.maximum(oxygen, 0.0)
    oxygen_limitation = oxygen_nonnegative / (
        oxygen_nonnegative + plan.oxygen_half_saturation_kpa
    )
    kinetic_oxygen = (
        plan.oxygen_kinetic_floor + (1.0 - plan.oxygen_kinetic_floor) * oxygen_limitation
    )
    atp_nonnegative = jnp.maximum(state.atp_pmol_per_mm3, 0.0)
    atp_saturation = atp_nonnegative / (atp_nonnegative + plan.atp_half_saturation)
    fractions = state.crossbridge_fractions
    detached = fractions[..., SarcomereSpecies.DETACHED_PRIMED_ADP_PI]
    weak = fractions[..., SarcomereSpecies.WEAK_BOUND_ADP_PI]
    strong = fractions[..., SarcomereSpecies.STRONG_BOUND_ADP]
    rigor = fractions[..., SarcomereSpecies.RIGOR_BOUND]
    attachment_rate = (
        plan.attachment_rate_per_ms * calcium_activation * length_overlap * kinetic_oxygen
    )
    powerstroke_rate = plan.powerstroke_rate_per_ms * force_velocity * kinetic_oxygen
    adp_release_rate = plan.adp_release_rate_per_ms * kinetic_oxygen
    atp_binding_rate = plan.atp_binding_rate_per_ms * atp_saturation * kinetic_oxygen
    attachment_extent = detached * (-jnp.expm1(-attachment_rate * step))
    powerstroke_extent = weak * (-jnp.expm1(-powerstroke_rate * step))
    adp_release_extent = strong * (-jnp.expm1(-adp_release_rate * step))
    requested_atp_binding = rigor * (-jnp.expm1(-atp_binding_rate * step))
    atp_limited_fraction = atp_nonnegative / plan.myosin_site_density
    atp_binding_extent = jnp.minimum(requested_atp_binding, atp_limited_fraction)
    new_detached = detached - attachment_extent + atp_binding_extent
    new_weak = weak + attachment_extent - powerstroke_extent
    new_strong = strong + powerstroke_extent - adp_release_extent
    new_rigor = rigor + adp_release_extent - atp_binding_extent
    new_fractions = jnp.stack((new_detached, new_weak, new_strong, new_rigor), axis=-1)
    site_density = plan.myosin_site_density
    atp_after_binding = state.atp_pmol_per_mm3 - site_density * atp_binding_extent
    adp_after_release = state.adp_pmol_per_mm3 + site_density * adp_release_extent
    pi_after_release = state.phosphate_pmol_per_mm3 + site_density * powerstroke_extent
    adp_nonnegative = jnp.maximum(adp_after_release, 0.0)
    pi_nonnegative = jnp.maximum(pi_after_release, 0.0)
    adp_saturation = adp_nonnegative / (
        adp_nonnegative + plan.oxidative_adp_half_saturation
    )
    pi_saturation = pi_nonnegative / (pi_nonnegative + plan.oxidative_pi_half_saturation)
    requested_regeneration = (
        jnp.maximum(oxidative_capacity, 0.0)
        * oxygen_limitation
        * adp_saturation
        * pi_saturation
        * step
    )
    regenerated = jnp.minimum(
        requested_regeneration, jnp.minimum(adp_nonnegative, pi_nonnegative)
    )
    candidate = SarcomereState(
        new_fractions,
        atp_after_binding + regenerated,
        adp_after_release - regenerated,
        pi_after_release - regenerated,
        state.time_ms + step,
        plan.plan_id,
    )
    modulation = SarcomereKineticModulation(
        calcium_activation,
        length_overlap,
        force_velocity,
        oxygen_limitation,
        kinetic_oxygen,
        atp_saturation,
    )
    strong_candidate = candidate.crossbridge_fractions[
        ..., SarcomereSpecies.STRONG_BOUND_ADP
    ]
    rigor_candidate = candidate.crossbridge_fractions[..., SarcomereSpecies.RIGOR_BOUND]
    active_stress = (
        plan.maximum_active_stress_kpa
        * (strong_candidate + plan.rigor_stress_fraction * rigor_candidate)
        * length_overlap
        * force_velocity
    )
    strain_rate = velocity / jnp.maximum(length, jnp.finfo(length.dtype).tiny)
    atp_consumption_rate = site_density * atp_binding_extent / step
    regeneration_rate = regenerated / step
    oxygen_consumption_rate = plan.oxygen_per_atp * regeneration_rate
    outputs = SarcomereOutputs(
        active_stress,
        strain_rate,
        oxygen_consumption_rate,
        atp_consumption_rate,
        regeneration_rate,
        modulation,
    )
    extents = SarcomereReactionExtents(
        attachment_extent,
        powerstroke_extent,
        adp_release_extent,
        atp_binding_extent,
        regenerated,
    )
    old_high_energy = plan.atp_free_energy * (
        state.atp_pmol_per_mm3 + site_density * (detached + weak)
    )
    new_high_energy = plan.atp_free_energy * (
        candidate.atp_pmol_per_mm3 + site_density * (new_detached + new_weak)
    )
    storage_rate = (new_high_energy - old_high_energy) / step
    metabolic_power = plan.atp_free_energy * regeneration_rate
    release_power = plan.atp_free_energy * site_density * powerstroke_extent / step
    mechanical_power = active_stress * strain_rate
    heat_power = release_power - mechanical_power
    chemical_residual = metabolic_power - storage_rate - release_power
    total_residual = metabolic_power - storage_rate - mechanical_power - heat_power
    thermodynamic = heat_power >= -tolerance
    ledger = SarcomerePowerLedger(
        old_high_energy,
        new_high_energy,
        storage_rate,
        metabolic_power,
        release_power,
        mechanical_power,
        heat_power,
        chemical_residual,
        total_residual,
        thermodynamic,
    )
    adenylate_before, phosphoryl_before = _chemical_invariants(plan, state)
    adenylate_after, phosphoryl_after = _chemical_invariants(plan, candidate)
    adenylate_error = jnp.max(jnp.abs(adenylate_after - adenylate_before))
    phosphoryl_error = jnp.max(jnp.abs(phosphoryl_after - phosphoryl_before))
    population_sum_error = jnp.max(
        jnp.abs(jnp.sum(candidate.crossbridge_fractions, axis=-1) - 1.0)
    )
    minimum_population = jnp.min(candidate.crossbridge_fractions)
    minimum_species = jnp.min(
        jnp.stack(
            (
                candidate.atp_pmol_per_mm3,
                candidate.adp_pmol_per_mm3,
                candidate.phosphate_pmol_per_mm3,
            ),
            axis=-1,
        )
    )
    chemical_scale = jnp.maximum(
        1.0,
        jnp.max(jnp.abs(adenylate_before)) + jnp.max(jnp.abs(phosphoryl_before)),
    )
    power_scale = jnp.maximum(
        1.0,
        jnp.max(jnp.abs(metabolic_power))
        + jnp.max(jnp.abs(release_power))
        + jnp.max(jnp.abs(mechanical_power)),
    )
    species_balanced = (adenylate_error <= tolerance * chemical_scale) & (
        phosphoryl_error <= tolerance * chemical_scale
    )
    chemical_power_error = jnp.max(jnp.abs(chemical_residual))
    total_power_error = jnp.max(jnp.abs(total_residual))
    power_balanced = (chemical_power_error <= tolerance * power_scale) & (
        total_power_error <= tolerance * power_scale
    )
    coupling_valid = (
        jnp.all(jnp.isfinite(calcium))
        & jnp.all(jnp.isfinite(length))
        & jnp.all(jnp.isfinite(velocity))
        & jnp.all(jnp.isfinite(oxygen))
        & jnp.all(jnp.isfinite(oxidative_capacity))
        & jnp.all(calcium >= 0.0)
        & jnp.all(length > 0.0)
        & jnp.all(oxygen >= 0.0)
        & jnp.all(oxidative_capacity >= 0.0)
    )
    source_valid = _fraction_validity(state, tolerance)
    candidate_valid = _fraction_validity(candidate, tolerance)
    finite = (
        jnp.all(jnp.isfinite(active_stress))
        & jnp.all(jnp.isfinite(chemical_residual))
        & jnp.all(jnp.isfinite(total_residual))
        & jnp.all(jnp.isfinite(heat_power))
    )
    population_balanced = population_sum_error <= tolerance
    populations_positive = minimum_population >= -tolerance
    species_positive = minimum_species >= -tolerance
    thermodynamic_valid = jnp.all(thermodynamic)
    passed = (
        coupling_valid
        & source_valid
        & candidate_valid
        & population_balanced
        & populations_positive
        & species_positive
        & species_balanced
        & power_balanced
        & thermodynamic_valid
        & finite
    )
    status = jnp.asarray(int(SarcomereStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~thermodynamic_valid, int(SarcomereStatus.THERMODYNAMIC_FAILURE), status
    )
    status = jnp.where(~power_balanced, int(SarcomereStatus.POWER_BALANCE), status)
    status = jnp.where(~species_balanced, int(SarcomereStatus.SPECIES_BALANCE), status)
    status = jnp.where(
        ~species_positive,
        int(SarcomereStatus.NEGATIVE_CHEMICAL_SPECIES),
        status,
    )
    status = jnp.where(
        ~population_balanced, int(SarcomereStatus.POPULATION_BALANCE), status
    )
    status = jnp.where(
        ~populations_positive, int(SarcomereStatus.NEGATIVE_POPULATION), status
    )
    status = jnp.where(
        ~coupling_valid, int(SarcomereStatus.INVALID_COUPLING_INPUT), status
    )
    status = jnp.where(~finite, int(SarcomereStatus.NONFINITE), status)
    evidence = SarcomereStepEvidence(
        population_sum_error,
        minimum_population,
        minimum_species,
        adenylate_before,
        adenylate_after,
        adenylate_error,
        phosphoryl_before,
        phosphoryl_after,
        phosphoryl_error,
        chemical_power_error,
        total_power_error,
        coupling_valid,
        source_valid,
        candidate_valid,
        species_balanced,
        power_balanced,
        finite,
        passed,
        status,
    )
    selected = SarcomereState(
        jnp.where(passed, candidate.crossbridge_fractions, state.crossbridge_fractions),
        jnp.where(passed, candidate.atp_pmol_per_mm3, state.atp_pmol_per_mm3),
        jnp.where(passed, candidate.adp_pmol_per_mm3, state.adp_pmol_per_mm3),
        jnp.where(
            passed,
            candidate.phosphate_pmol_per_mm3,
            state.phosphate_pmol_per_mm3,
        ),
        jnp.where(passed, candidate.time_ms, state.time_ms),
        plan.plan_id,
    )
    return SarcomereStepResult(
        selected,
        candidate,
        extents,
        outputs,
        ledger,
        evidence,
        passed,
    )


__all__ = [
    "MeanFieldSarcomereFidelity",
    "MeanFieldSarcomerePlan",
    "SarcomereCouplingInputs",
    "SarcomereKineticModulation",
    "SarcomereOutputs",
    "SarcomerePowerLedger",
    "SarcomereReactionExtents",
    "SarcomereSpecies",
    "SarcomereState",
    "SarcomereStatus",
    "SarcomereStepEvidence",
    "SarcomereStepResult",
    "StochasticMolecularSarcomereFidelity",
    "initialize_sarcomere_state",
    "step_mean_field_sarcomere",
]
