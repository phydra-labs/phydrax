#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._features import LCMSFeatureBatch
from ._spectrum import IonPolarity


class MetaboliteConfidenceLevel(IntEnum):
    """Evidence level for a metabolite candidate, ordered strongest to weakest."""

    AUTHENTICATED_STANDARD = 1
    PUTATIVELY_ANNOTATED = 2
    FORMULA_OR_CLASS = 3
    UNKNOWN = 4


class MetabolomicsStatus(IntEnum):
    """Status of bounded metabolite candidate matching."""

    SUCCESS = 0
    CAPACITY_EXCEEDED = 1
    POLARITY_MISMATCH = 2
    NONFINITE = 3


class MetabolomicsEvidence(IntFlag):
    """Evidence retained by metabolite matching and confidence classification."""

    NONE = 0
    EXACT_MASS = 1
    ADDUCT = 2
    ISOTOPE_PATTERN = 4
    AMBIGUOUS = 8
    REFERENCE_MSMS = 16
    AUTHENTICATED_STANDARD = 32
    CLASS_ONLY = 64


_MATCH_CONTRACT = BioinformaticsMethodContract(
    "bounded metabolite formula-adduct-isotope matching",
    MethodKind.HEURISTIC,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Every feature × formula × adduct combination is evaluated against explicit "
        "mass, polarity, and isotope tolerances."
    ),
    truncation_statement=(
        "Worst-case candidate capacity is preflighted; insufficient capacity returns "
        "failure without partial candidates."
    ),
    capacity_semantics="Candidate work is bounded by feature_capacity × formula_capacity × adduct_capacity.",
    assumptions=(
        "Formula exact masses and adduct mass shifts use the feature mass unit.",
    ),
    nondifferentiable_outputs=(
        "indices",
        "matched_mask",
        "ambiguous_mask",
        "status",
        "evidence",
    ),
)

_CONFIDENCE_CONTRACT = BioinformaticsMethodContract(
    "metabolite candidate confidence classification",
    MethodKind.HEURISTIC,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Confidence levels distinguish authenticated standard, library MS/MS, "
        "formula/class, and unknown evidence."
    ),
    truncation_statement="Every bounded candidate is classified and ambiguity remains explicit.",
    capacity_semantics="Candidate capacity is preserved exactly.",
    assumptions=(
        "standard_match encodes co-analysis against an authenticated standard, including retention behavior.",
    ),
    nondifferentiable_outputs=("all outputs",),
)


class AdductBatch(StrictModule):
    """Fixed-capacity ion adduct definitions without host labels."""

    adduct_ids: Array
    molecule_multiplier: Array
    mass_shift: Array
    charge: Array
    polarity: Array
    active_mask: Array
    adduct_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        adduct_ids: ArrayLike,
        molecule_multiplier: ArrayLike,
        mass_shift: ArrayLike,
        charge: ArrayLike,
        polarity: ArrayLike,
        active_mask: ArrayLike,
        /,
    ):
        ids = np.asarray(adduct_ids)
        multiplier = np.asarray(molecule_multiplier)
        shifts = np.asarray(mass_shift)
        charges = np.asarray(charge)
        polarities = np.asarray(polarity)
        mask = np.asarray(active_mask, dtype=bool)
        if (
            ids.ndim != 1
            or ids.size == 0
            or any(
                value.shape != ids.shape
                for value in (multiplier, shifts, charges, polarities, mask)
            )
        ):
            raise ValueError("All adduct fields must be equal non-empty vectors.")
        if not all(
            np.issubdtype(value.dtype, np.integer)
            for value in (ids, multiplier, charges, polarities)
        ):
            raise TypeError(
                "Adduct identifiers, multipliers, charge, and polarity must be integers."
            )
        count = int(np.count_nonzero(mask))
        if not np.all(mask[:count]) or np.any(mask[count:]):
            raise ValueError("active_mask must be a left-prefix mask.")
        if np.any(ids[mask] < 0) or np.unique(ids[mask]).size != count:
            raise ValueError("Active adduct identifiers must be unique and nonnegative.")
        if np.any(multiplier[mask] < 1):
            raise ValueError("Active molecule multipliers must be positive.")
        if np.any((charges[mask] == 0) | (np.abs(charges[mask]) > 64)):
            raise ValueError(
                "Active adduct charges must be nonzero with magnitude at most 64."
            )
        if np.any(
            ~np.isin(
                polarities[mask], [int(IonPolarity.POSITIVE), int(IonPolarity.NEGATIVE)]
            )
        ):
            raise ValueError("Active adduct polarity must be POSITIVE or NEGATIVE.")
        if np.any(np.sign(charges[mask]) != polarities[mask]):
            raise ValueError("Adduct charge sign must agree with polarity.")
        if np.any(~np.isfinite(shifts[mask])):
            raise ValueError("Active adduct mass shifts must be finite.")
        for value in (ids, multiplier, shifts, charges, polarities):
            if np.any(value[~mask] != 0):
                raise ValueError("Inactive adduct entries must be zero padding.")
        self.adduct_ids = jnp.asarray(ids, dtype=jnp.int32)
        self.molecule_multiplier = jnp.asarray(multiplier, dtype=jnp.int32)
        self.mass_shift = jnp.asarray(shifts)
        self.charge = jnp.asarray(charges, dtype=jnp.int32)
        self.polarity = jnp.asarray(polarities, dtype=jnp.int32)
        self.active_mask = jnp.asarray(mask)
        self.adduct_capacity = int(ids.size)


class FormulaCandidateBatch(StrictModule):
    """Elemental formula candidates with optional theoretical isotope envelopes."""

    formula_ids: Array
    elemental_counts: Array
    atomic_numbers: Array
    exact_mass: Array
    double_bond_equivalents: Array
    isotope_mass_offset: Array
    isotope_relative_abundance: Array
    isotope_mask: Array
    active_mask: Array
    formula_capacity: int = eqx.field(static=True)
    element_capacity: int = eqx.field(static=True)
    isotope_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        formula_ids: ArrayLike,
        elemental_counts: ArrayLike,
        atomic_numbers: ArrayLike,
        exact_mass: ArrayLike,
        double_bond_equivalents: ArrayLike,
        isotope_mass_offset: ArrayLike,
        isotope_relative_abundance: ArrayLike,
        isotope_mask: ArrayLike,
        active_mask: ArrayLike,
        /,
    ):
        ids = np.asarray(formula_ids)
        counts = np.asarray(elemental_counts)
        elements = np.asarray(atomic_numbers)
        masses = np.asarray(exact_mass)
        dbe = np.asarray(double_bond_equivalents)
        isotope_offset = np.asarray(isotope_mass_offset)
        isotope_abundance = np.asarray(isotope_relative_abundance)
        isotope_active = np.asarray(isotope_mask, dtype=bool)
        mask = np.asarray(active_mask, dtype=bool)
        if (
            ids.ndim != 1
            or ids.size == 0
            or counts.ndim != 2
            or counts.shape[0] != ids.size
        ):
            raise ValueError(
                "Formula identifiers and elemental counts require matching positive capacity."
            )
        if elements.shape != (counts.shape[1],) or counts.shape[1] == 0:
            raise ValueError(
                "atomic_numbers must match a positive elemental-count width."
            )
        if masses.shape != ids.shape or dbe.shape != ids.shape or mask.shape != ids.shape:
            raise ValueError("Formula scalar fields must match formula_ids.")
        if (
            isotope_offset.ndim != 2
            or isotope_offset.shape[0] != ids.size
            or isotope_offset.shape[1] == 0
        ):
            raise ValueError(
                "Isotope arrays require positive formula and isotope capacities."
            )
        if (
            isotope_abundance.shape != isotope_offset.shape
            or isotope_active.shape != isotope_offset.shape
        ):
            raise ValueError(
                "Theoretical isotope arrays and mask must have the same shape."
            )
        if not all(
            np.issubdtype(value.dtype, np.integer) for value in (ids, counts, elements)
        ):
            raise TypeError(
                "Formula IDs, elemental counts, and atomic numbers must be integers."
            )
        count = int(np.count_nonzero(mask))
        if not np.all(mask[:count]) or np.any(mask[count:]):
            raise ValueError("Formula active_mask must be a left-prefix mask.")
        for row in isotope_active:
            isotope_count = int(np.count_nonzero(row))
            if not np.all(row[:isotope_count]) or np.any(row[isotope_count:]):
                raise ValueError("Each isotope mask row must be a left-prefix mask.")
        if np.any(isotope_active & ~mask[:, None]):
            raise ValueError("Inactive formulas cannot carry isotope patterns.")
        if np.any(ids[mask] < 0) or np.unique(ids[mask]).size != count:
            raise ValueError("Active formula identifiers must be unique and nonnegative.")
        if np.any(elements <= 0) or np.unique(elements).size != elements.size:
            raise ValueError("atomic_numbers must be unique and positive.")
        if np.any(counts[mask] < 0) or np.any(np.sum(counts[mask], axis=1) == 0):
            raise ValueError(
                "Active formulas require nonnegative, nonempty elemental counts."
            )
        if np.any(~np.isfinite(masses[mask])) or np.any(masses[mask] <= 0.0):
            raise ValueError("Active formula masses must be finite and positive.")
        if np.any(~np.isfinite(dbe[mask])):
            raise ValueError("Active double-bond equivalents must be finite.")
        if np.any(~np.isfinite(isotope_offset[isotope_active])) or np.any(
            isotope_offset[isotope_active] < 0.0
        ):
            raise ValueError(
                "Active isotope mass offsets must be finite and nonnegative."
            )
        if np.any(~np.isfinite(isotope_abundance[isotope_active])) or np.any(
            isotope_abundance[isotope_active] < 0.0
        ):
            raise ValueError("Active isotope abundances must be finite and nonnegative.")
        pattern_rows = np.any(isotope_active, axis=1)
        if np.any(np.abs(isotope_offset[pattern_rows, 0]) > 1.0e-12):
            raise ValueError("The first active theoretical isotope offset must be zero.")
        sums = np.sum(np.where(isotope_active, isotope_abundance, 0.0), axis=1)
        if np.any(pattern_rows & (sums <= 0.0)):
            raise ValueError(
                "Theoretical isotope patterns require positive total abundance."
            )
        if (
            np.any(counts[~mask] != 0)
            or np.any(ids[~mask] != 0)
            or np.any(masses[~mask] != 0.0)
            or np.any(dbe[~mask] != 0.0)
        ):
            raise ValueError("Inactive formula entries must be zero padding.")
        if np.any(isotope_offset[~isotope_active] != 0.0) or np.any(
            isotope_abundance[~isotope_active] != 0.0
        ):
            raise ValueError("Inactive theoretical isotope entries must be zero padding.")
        self.formula_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.elemental_counts = jnp.asarray(counts, dtype=jnp.int32)
        self.atomic_numbers = jnp.asarray(elements, dtype=jnp.int32)
        self.exact_mass = jnp.asarray(masses)
        self.double_bond_equivalents = jnp.asarray(dbe)
        self.isotope_mass_offset = jnp.asarray(isotope_offset)
        self.isotope_relative_abundance = jnp.asarray(isotope_abundance)
        self.isotope_mask = jnp.asarray(isotope_active)
        self.active_mask = jnp.asarray(mask)
        self.formula_capacity = int(ids.size)
        self.element_capacity = int(elements.size)
        self.isotope_capacity = int(isotope_offset.shape[1])


class IsotopeEnvelopeBatch(StrictModule):
    """Observed isotope offsets and relative signals for each LC-MS feature."""

    mass_offset: Array
    relative_intensity: Array
    active_mask: Array
    feature_capacity: int = eqx.field(static=True)
    isotope_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        mass_offset: ArrayLike,
        relative_intensity: ArrayLike,
        active_mask: ArrayLike,
        /,
    ):
        offset = np.asarray(mass_offset)
        signal = np.asarray(relative_intensity)
        mask = np.asarray(active_mask, dtype=bool)
        if offset.ndim != 2 or offset.size == 0 or offset.shape[1] == 0:
            raise ValueError(
                "Observed isotope arrays require positive two-dimensional capacity."
            )
        if signal.shape != offset.shape or mask.shape != offset.shape:
            raise ValueError("Observed isotope arrays and mask must have the same shape.")
        for row in mask:
            count = int(np.count_nonzero(row))
            if not np.all(row[:count]) or np.any(row[count:]):
                raise ValueError(
                    "Each observed isotope mask row must be a left-prefix mask."
                )
        if np.any(~np.isfinite(offset[mask])) or np.any(offset[mask] < 0.0):
            raise ValueError("Observed isotope offsets must be finite and nonnegative.")
        if np.any(~np.isfinite(signal[mask])) or np.any(signal[mask] < 0.0):
            raise ValueError(
                "Observed isotope intensities must be finite and nonnegative."
            )
        rows = np.any(mask, axis=1)
        if np.any(np.abs(offset[rows, 0]) > 1.0e-12):
            raise ValueError("The first observed isotope offset must be zero.")
        if np.any(np.sum(np.where(mask, signal, 0.0), axis=1)[rows] <= 0.0):
            raise ValueError(
                "Observed isotope envelopes require positive total intensity."
            )
        if np.any(offset[~mask] != 0.0) or np.any(signal[~mask] != 0.0):
            raise ValueError("Inactive observed isotope entries must be zero padding.")
        self.mass_offset = jnp.asarray(offset)
        self.relative_intensity = jnp.asarray(signal)
        self.active_mask = jnp.asarray(mask)
        self.feature_capacity = int(offset.shape[0])
        self.isotope_capacity = int(offset.shape[1])


class MetaboliteFeatureBatch(StrictModule):
    """LC-MS features paired with their observed isotope envelopes."""

    features: LCMSFeatureBatch
    isotopes: IsotopeEnvelopeBatch

    def __init__(self, features: LCMSFeatureBatch, isotopes: IsotopeEnvelopeBatch, /):
        if not isinstance(features, LCMSFeatureBatch):
            raise TypeError("features must be LCMSFeatureBatch.")
        if not isinstance(isotopes, IsotopeEnvelopeBatch):
            raise TypeError("isotopes must be IsotopeEnvelopeBatch.")
        if features.feature_capacity != isotopes.feature_capacity:
            raise ValueError("Feature and isotope-envelope capacities must match.")
        isotope_rows = np.any(np.asarray(isotopes.active_mask), axis=1)
        if np.any(isotope_rows & ~np.asarray(features.active_mask)):
            raise ValueError("Inactive LC-MS features cannot carry isotope envelopes.")
        self.features = features
        self.isotopes = isotopes

    @property
    def feature_capacity(self) -> int:
        return self.features.feature_capacity


class MetaboliteMatchingPlan(StrictModule):
    """Tolerances and preflighted candidate capacity for exhaustive matching."""

    mass_tolerance_ppm: float = eqx.field(static=True)
    isotope_mass_tolerance: float = eqx.field(static=True)
    minimum_isotope_cosine: float = eqx.field(static=True)
    candidate_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        mass_tolerance_ppm: float,
        isotope_mass_tolerance: float,
        minimum_isotope_cosine: float,
        candidate_capacity: int,
    ):
        ppm = float(mass_tolerance_ppm)
        isotope_tolerance = float(isotope_mass_tolerance)
        cosine = float(minimum_isotope_cosine)
        capacity = int(candidate_capacity)
        if not np.isfinite(ppm) or ppm <= 0.0:
            raise ValueError("mass_tolerance_ppm must be finite and positive.")
        if not np.isfinite(isotope_tolerance) or isotope_tolerance < 0.0:
            raise ValueError("isotope_mass_tolerance must be finite and nonnegative.")
        if not np.isfinite(cosine) or cosine < 0.0 or cosine > 1.0:
            raise ValueError("minimum_isotope_cosine must lie in [0, 1].")
        if capacity < 1:
            raise ValueError("candidate_capacity must be positive.")
        self.mass_tolerance_ppm = ppm
        self.isotope_mass_tolerance = isotope_tolerance
        self.minimum_isotope_cosine = cosine
        self.candidate_capacity = capacity


class MetaboliteMatchResult(StrictModule):
    """Exhaustive bounded metabolite candidates with explicit ambiguity."""

    feature_index: Array
    formula_index: Array
    adduct_index: Array
    feature_id: Array
    formula_id: Array
    adduct_id: Array
    predicted_mass_to_charge: Array
    mass_error_ppm: Array
    isotope_cosine: Array
    isotope_evidence_mask: Array
    matched_mask: Array
    ambiguous_mask: Array
    feature_candidate_count: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _empty_metabolite_result(
    plan: MetaboliteMatchingPlan,
    status: MetabolomicsStatus,
    /,
) -> MetaboliteMatchResult:
    shape = (plan.candidate_capacity,)
    zeros = jnp.zeros(shape)
    return MetaboliteMatchResult(
        feature_index=jnp.full(shape, -1, dtype=jnp.int32),
        formula_index=jnp.full(shape, -1, dtype=jnp.int32),
        adduct_index=jnp.full(shape, -1, dtype=jnp.int32),
        feature_id=jnp.full(shape, -1, dtype=jnp.int64),
        formula_id=jnp.full(shape, -1, dtype=jnp.int64),
        adduct_id=jnp.full(shape, -1, dtype=jnp.int32),
        predicted_mass_to_charge=zeros,
        mass_error_ppm=zeros,
        isotope_cosine=zeros,
        isotope_evidence_mask=jnp.zeros(shape, dtype=bool),
        matched_mask=jnp.zeros(shape, dtype=bool),
        ambiguous_mask=jnp.zeros(shape, dtype=bool),
        feature_candidate_count=jnp.zeros(shape, dtype=jnp.int32),
        valid=jnp.asarray(False),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        evidence=jnp.asarray(int(MetabolomicsEvidence.NONE), dtype=jnp.uint32),
        method_contract=_MATCH_CONTRACT,
    )


def match_metabolite_candidates(
    features: MetaboliteFeatureBatch,
    formulas: FormulaCandidateBatch,
    adducts: AdductBatch,
    plan: MetaboliteMatchingPlan,
    /,
) -> MetaboliteMatchResult:
    """Exhaustively evaluate mass/adduct/isotope candidates after capacity preflight."""
    if not isinstance(features, MetaboliteFeatureBatch):
        raise TypeError("features must be MetaboliteFeatureBatch.")
    if not isinstance(formulas, FormulaCandidateBatch):
        raise TypeError("formulas must be FormulaCandidateBatch.")
    if not isinstance(adducts, AdductBatch):
        raise TypeError("adducts must be AdductBatch.")
    if not isinstance(plan, MetaboliteMatchingPlan):
        raise TypeError("plan must be MetaboliteMatchingPlan.")
    worst_case = (
        features.feature_capacity * formulas.formula_capacity * adducts.adduct_capacity
    )
    if worst_case > plan.candidate_capacity:
        return _empty_metabolite_result(plan, MetabolomicsStatus.CAPACITY_EXCEEDED)
    isotope_capacity = formulas.isotope_capacity
    if isotope_capacity > features.isotopes.isotope_capacity:
        return _empty_metabolite_result(plan, MetabolomicsStatus.CAPACITY_EXCEEDED)
    feature_index = jnp.repeat(
        jnp.arange(features.feature_capacity),
        formulas.formula_capacity * adducts.adduct_capacity,
    )
    formula_index = jnp.tile(
        jnp.repeat(jnp.arange(formulas.formula_capacity), adducts.adduct_capacity),
        features.feature_capacity,
    )
    adduct_index = jnp.tile(
        jnp.arange(adducts.adduct_capacity),
        features.feature_capacity * formulas.formula_capacity,
    )
    fi, fo, ai = feature_index, formula_index, adduct_index
    active = (
        features.features.active_mask[fi]
        & formulas.active_mask[fo]
        & adducts.active_mask[ai]
    )
    feature_polarity = features.features.polarity
    polarity_match = (feature_polarity == IonPolarity.UNKNOWN) | (
        adducts.polarity[ai] == int(feature_polarity)
    )
    charge_magnitude = jnp.maximum(jnp.abs(adducts.charge[ai]), 1)
    predicted = (
        adducts.molecule_multiplier[ai] * formulas.exact_mass[fo] + adducts.mass_shift[ai]
    ) / charge_magnitude
    positive_predicted = predicted > 0.0
    observed = features.features.mass_to_charge[fi]
    ppm_error = (
        1.0e6 * (observed - predicted) / jnp.where(positive_predicted, predicted, 1.0)
    )
    mass_match = positive_predicted & (jnp.abs(ppm_error) <= plan.mass_tolerance_ppm)

    observed_mask = features.isotopes.active_mask[fi, :isotope_capacity]
    theoretical_mask = formulas.isotope_mask[fo]
    isotope_evidence = jnp.any(theoretical_mask, axis=1)
    required_present = jnp.all(~theoretical_mask | observed_mask, axis=1)
    expected_offset = formulas.isotope_mass_offset[fo] / charge_magnitude[:, None]
    observed_offset = features.isotopes.mass_offset[fi, :isotope_capacity]
    offset_error = jnp.abs(observed_offset - expected_offset)
    mass_pattern_match = jnp.all(
        ~theoretical_mask
        | (observed_mask & (offset_error <= plan.isotope_mass_tolerance)),
        axis=1,
    )
    observed_signal = jnp.where(
        theoretical_mask & observed_mask,
        features.isotopes.relative_intensity[fi, :isotope_capacity],
        0.0,
    )
    theoretical_signal = jnp.where(
        theoretical_mask,
        formulas.isotope_relative_abundance[fo],
        0.0,
    )
    dot = jnp.sum(observed_signal * theoretical_signal, axis=1)
    norm = jnp.sqrt(
        jnp.sum(observed_signal * observed_signal, axis=1)
        * jnp.sum(theoretical_signal * theoretical_signal, axis=1)
    )
    cosine = jnp.where(
        norm > 0.0,
        dot / jnp.maximum(norm, jnp.finfo(norm.dtype).tiny),
        0.0,
    )
    isotope_match = ~isotope_evidence | (
        required_present & mass_pattern_match & (cosine >= plan.minimum_isotope_cosine)
    )
    matched = active & polarity_match & mass_match & isotope_match
    candidate_count_by_feature = (
        jnp.zeros((features.feature_capacity,), dtype=jnp.int32)
        .at[fi]
        .add(matched.astype(jnp.int32))
    )
    candidate_count = candidate_count_by_feature[fi]
    ambiguous = matched & (candidate_count > 1)
    padding = plan.candidate_capacity - worst_case

    def pad(array: Array, value: float | int | bool = 0) -> Array:
        return jnp.pad(array, (0, padding), constant_values=value)

    any_match = jnp.any(matched)
    evidence_bits = jnp.where(
        any_match,
        int(MetabolomicsEvidence.EXACT_MASS | MetabolomicsEvidence.ADDUCT),
        int(MetabolomicsEvidence.NONE),
    ).astype(jnp.uint32)
    evidence_bits = evidence_bits | jnp.where(
        jnp.any(matched & isotope_evidence),
        int(MetabolomicsEvidence.ISOTOPE_PATTERN),
        0,
    ).astype(jnp.uint32)
    evidence_bits = evidence_bits | jnp.where(
        jnp.any(ambiguous), int(MetabolomicsEvidence.AMBIGUOUS), 0
    ).astype(jnp.uint32)
    finite = jnp.all(jnp.isfinite(jnp.where(active, predicted + ppm_error, 0.0)))
    return MetaboliteMatchResult(
        feature_index=pad(fi.astype(jnp.int32), -1),
        formula_index=pad(fo.astype(jnp.int32), -1),
        adduct_index=pad(ai.astype(jnp.int32), -1),
        feature_id=pad(features.features.feature_ids[fi], -1),
        formula_id=pad(formulas.formula_ids[fo], -1),
        adduct_id=pad(adducts.adduct_ids[ai], -1),
        predicted_mass_to_charge=pad(jnp.where(active, predicted, 0.0)),
        mass_error_ppm=pad(jnp.where(active, ppm_error, 0.0)),
        isotope_cosine=pad(jnp.where(isotope_evidence, cosine, 0.0)),
        isotope_evidence_mask=pad(active & isotope_evidence, False),
        matched_mask=pad(matched, False),
        ambiguous_mask=pad(ambiguous, False),
        feature_candidate_count=pad(candidate_count, 0),
        valid=finite,
        status=jnp.where(
            finite, int(MetabolomicsStatus.SUCCESS), int(MetabolomicsStatus.NONFINITE)
        ).astype(jnp.int32),
        evidence=evidence_bits,
        method_contract=_MATCH_CONTRACT,
    )


class MetaboliteConfidenceResult(StrictModule):
    """Explicit metabolite evidence levels without collapsing candidate ambiguity."""

    confidence_level: Array
    active_mask: Array
    ambiguous_mask: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def classify_metabolite_confidence(
    matches: MetaboliteMatchResult,
    /,
    *,
    authenticated_standard_match: ArrayLike,
    reference_msms_match: ArrayLike,
    class_match: ArrayLike | None = None,
) -> MetaboliteConfidenceResult:
    """Assign distinct MSI-style confidence levels to every matched candidate."""
    if not isinstance(matches, MetaboliteMatchResult):
        raise TypeError("matches must be MetaboliteMatchResult.")
    standard = jnp.asarray(authenticated_standard_match, dtype=bool)
    msms = jnp.asarray(reference_msms_match, dtype=bool)
    if (
        standard.shape != matches.matched_mask.shape
        or msms.shape != matches.matched_mask.shape
    ):
        raise ValueError("Standard and MS/MS match masks must match candidate capacity.")
    classes = (
        jnp.zeros(matches.matched_mask.shape, dtype=bool)
        if class_match is None
        else jnp.asarray(class_match, dtype=bool)
    )
    if classes.shape != matches.matched_mask.shape:
        raise ValueError("class_match must match candidate capacity.")
    active = matches.matched_mask
    level = jnp.where(
        active & standard & msms,
        int(MetaboliteConfidenceLevel.AUTHENTICATED_STANDARD),
        jnp.where(
            active & msms,
            int(MetaboliteConfidenceLevel.PUTATIVELY_ANNOTATED),
            jnp.where(
                active | classes,
                int(MetaboliteConfidenceLevel.FORMULA_OR_CLASS),
                int(MetaboliteConfidenceLevel.UNKNOWN),
            ),
        ),
    ).astype(jnp.int32)
    evidence = matches.evidence
    evidence = evidence | jnp.where(
        jnp.any(active & msms), int(MetabolomicsEvidence.REFERENCE_MSMS), 0
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        jnp.any(active & standard & msms),
        int(MetabolomicsEvidence.AUTHENTICATED_STANDARD),
        0,
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        jnp.any(classes & ~active), int(MetabolomicsEvidence.CLASS_ONLY), 0
    ).astype(jnp.uint32)
    return MetaboliteConfidenceResult(
        confidence_level=level,
        active_mask=active | classes,
        ambiguous_mask=matches.ambiguous_mask,
        valid=matches.valid,
        status=matches.status,
        evidence=evidence,
        method_contract=_CONFIDENCE_CONTRACT,
    )


__all__ = [
    "AdductBatch",
    "FormulaCandidateBatch",
    "IsotopeEnvelopeBatch",
    "MetaboliteConfidenceLevel",
    "MetaboliteConfidenceResult",
    "MetaboliteFeatureBatch",
    "MetaboliteMatchResult",
    "MetaboliteMatchingPlan",
    "MetabolomicsEvidence",
    "MetabolomicsStatus",
    "classify_metabolite_confidence",
    "match_metabolite_candidates",
]
