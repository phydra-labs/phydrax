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
from ..sequence import PROTEIN_IUPAC, SequenceBatch


class ModificationSiteKind(IntEnum):
    """Location semantics of a peptide or proteoform modification."""

    RESIDUE = 0
    N_TERMINUS = 1
    C_TERMINUS = 2


class ProteomicsStatus(IntEnum):
    """Status of a bounded proteomics operation."""

    SUCCESS = 0
    INVALID_CANDIDATE = 1
    CAPACITY_EXCEEDED = 2
    NONFINITE = 3


class ProteomicsEvidence(IntFlag):
    """Evidence retained by peptide scoring and decoy construction."""

    NONE = 0
    MASS_ERROR = 1
    FRAGMENT_MATCH = 2
    EXPLAINED_INTENSITY = 4
    SPECTRAL_ANGLE = 8
    REVERSED_DECOY = 16
    MODIFICATIONS_REMAPPED = 32


_DECOY_CONTRACT = BioinformaticsMethodContract(
    "reversed peptide decoy construction",
    MethodKind.HEURISTIC,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Active peptide residues are reversed; residue modification sites are "
        "reflected and terminal sites are swapped."
    ),
    truncation_statement="Peptide, modification, and charge capacities are preserved exactly.",
    capacity_semantics="One decoy is produced for every active target candidate.",
    assumptions=(
        "Input candidate identifiers are nonnegative and unique among active candidates.",
    ),
    nondifferentiable_outputs=("all outputs",),
)

_SCORE_CONTRACT = BioinformaticsMethodContract(
    "peptide-spectrum match composite scoring",
    MethodKind.HEURISTIC,
    ExecutionKind.FLOATING_POINT_DIRECT,
    DifferentiationKind.EXACT_AD,
    OutputKind.STRUCTURED,
    conditioning_statement="Bounded evidence terms are combined after explicit precursor-error scaling.",
    truncation_statement="Every active PSM input receives a score; no candidate ranking is truncated.",
    capacity_semantics="PSM capacity and scoring components are fixed by the input payload.",
    assumptions=(
        "Fragment counts and evidence fractions were computed against the stated peptide and charge.",
    ),
    nondifferentiable_outputs=("status", "evidence", "active_mask", "is_decoy"),
)


class ModificationBatch(StrictModule):
    """Fixed candidate-by-modification capacity modification annotations."""

    modification_ids: Array
    positions: Array
    site_kinds: Array
    mass_shifts: Array
    active_mask: Array
    candidate_capacity: int = eqx.field(static=True)
    modification_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        modification_ids: ArrayLike,
        positions: ArrayLike,
        site_kinds: ArrayLike,
        mass_shifts: ArrayLike,
        active_mask: ArrayLike,
        /,
    ):
        ids = np.asarray(modification_ids)
        positions_host = np.asarray(positions)
        kinds = np.asarray(site_kinds)
        shifts = np.asarray(mass_shifts)
        mask = np.asarray(active_mask, dtype=bool)
        if ids.ndim != 2 or ids.size == 0 or ids.shape[1] == 0:
            raise ValueError(
                "Modification arrays require positive candidate and slot capacities."
            )
        if any(
            value.shape != ids.shape for value in (positions_host, kinds, shifts, mask)
        ):
            raise ValueError("All modification arrays must have the same shape.")
        if not all(
            np.issubdtype(value.dtype, np.integer)
            for value in (ids, positions_host, kinds)
        ):
            raise TypeError(
                "Modification identifiers, positions, and site kinds must be integers."
            )
        for row in mask:
            count = int(np.count_nonzero(row))
            if not np.all(row[:count]) or np.any(row[count:]):
                raise ValueError("Each modification mask row must be a left-prefix mask.")
        if np.any(ids[mask] < 0):
            raise ValueError("Active modification identifiers must be nonnegative.")
        if np.any(positions_host[mask] < 0):
            raise ValueError("Active modification positions must be nonnegative.")
        if np.any(
            (kinds[mask] < int(ModificationSiteKind.RESIDUE))
            | (kinds[mask] > int(ModificationSiteKind.C_TERMINUS))
        ):
            raise ValueError("Active site_kinds contain an unsupported value.")
        if np.any(~np.isfinite(shifts[mask])):
            raise ValueError("Active modification mass shifts must be finite.")
        for value in (ids, positions_host, kinds, shifts):
            if np.any(value[~mask] != 0):
                raise ValueError("Inactive modification slots must be zero padding.")
        self.modification_ids = jnp.asarray(ids, dtype=jnp.int32)
        self.positions = jnp.asarray(positions_host, dtype=jnp.int32)
        self.site_kinds = jnp.asarray(kinds, dtype=jnp.int32)
        self.mass_shifts = jnp.asarray(shifts)
        self.active_mask = jnp.asarray(mask)
        self.candidate_capacity = int(ids.shape[0])
        self.modification_capacity = int(ids.shape[1])

    @classmethod
    def empty(
        cls,
        candidate_capacity: int,
        modification_capacity: int = 1,
        /,
    ) -> ModificationBatch:
        shape = (int(candidate_capacity), int(modification_capacity))
        if shape[0] < 1 or shape[1] < 1:
            raise ValueError("candidate and modification capacities must be positive.")
        return cls(
            np.zeros(shape, dtype=np.int32),
            np.zeros(shape, dtype=np.int32),
            np.zeros(shape, dtype=np.int32),
            np.zeros(shape, dtype=float),
            np.zeros(shape, dtype=bool),
        )


class PeptideCandidateBatch(StrictModule):
    """Bounded encoded peptide candidates with charge, mass, and modifications."""

    sequences: SequenceBatch
    charge: Array
    neutral_mass: Array
    is_decoy: Array
    modifications: ModificationBatch
    candidate_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        sequences: SequenceBatch,
        charge: ArrayLike,
        neutral_mass: ArrayLike,
        is_decoy: ArrayLike,
        modifications: ModificationBatch,
        /,
    ):
        if not isinstance(sequences, SequenceBatch):
            raise TypeError("sequences must be a SequenceBatch.")
        if sequences.alphabet != PROTEIN_IUPAC:
            raise ValueError("Peptide candidates require the PROTEIN_IUPAC alphabet.")
        charges = np.asarray(charge)
        masses = np.asarray(neutral_mass)
        decoys = np.asarray(is_decoy, dtype=bool)
        capacity = sequences.record_capacity
        if (
            charges.shape != (capacity,)
            or masses.shape != (capacity,)
            or decoys.shape != (capacity,)
        ):
            raise ValueError("Peptide metadata must match the sequence record capacity.")
        if not np.issubdtype(charges.dtype, np.integer):
            raise TypeError("charge must contain integers.")
        if not isinstance(modifications, ModificationBatch):
            raise TypeError("modifications must be a ModificationBatch.")
        if modifications.candidate_capacity != capacity:
            raise ValueError("Modification and peptide candidate capacities must match.")
        active = np.asarray(sequences.case_mask)
        if np.any((charges[active] == 0) | (np.abs(charges[active]) > 64)):
            raise ValueError(
                "Active peptide charges must be nonzero with magnitude at most 64."
            )
        if np.any(~np.isfinite(masses[active])) or np.any(masses[active] <= 0.0):
            raise ValueError("Active peptide neutral masses must be finite and positive.")
        lengths = np.asarray(sequences.valid_mask).sum(axis=1)
        residue_mod = np.asarray(modifications.active_mask) & (
            np.asarray(modifications.site_kinds) == int(ModificationSiteKind.RESIDUE)
        )
        if np.any(
            np.asarray(modifications.positions)[residue_mod]
            >= np.repeat(lengths[:, None], modifications.modification_capacity, axis=1)[
                residue_mod
            ]
        ):
            raise ValueError(
                "Residue modification positions must be inside peptide lengths."
            )
        if np.any(np.asarray(modifications.active_mask) & ~active[:, None]):
            raise ValueError("Padded peptide candidates cannot carry modifications.")
        if (
            np.any(charges[~active] != 0)
            or np.any(masses[~active] != 0.0)
            or np.any(decoys[~active])
        ):
            raise ValueError("Inactive peptide metadata must be zero/false padding.")
        ids = np.asarray(sequences.record_ids)
        if np.any(ids[active & ~decoys] < 0) or np.any(ids[active & decoys] >= 0):
            raise ValueError(
                "Target peptide IDs must be nonnegative and decoy IDs negative."
            )
        if np.unique(ids[active]).size != np.count_nonzero(active):
            raise ValueError("Active peptide candidate identifiers must be unique.")
        self.sequences = sequences
        self.charge = jnp.asarray(charges, dtype=jnp.int32)
        self.neutral_mass = jnp.asarray(masses)
        self.is_decoy = jnp.asarray(decoys)
        self.modifications = modifications
        self.candidate_capacity = capacity


class ProteoformCandidateBatch(StrictModule):
    """Bounded protein-sequence candidates with explicit proteoform modifications."""

    sequences: SequenceBatch
    protein_ids: Array
    modifications: ModificationBatch
    candidate_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        sequences: SequenceBatch,
        protein_ids: ArrayLike,
        modifications: ModificationBatch,
        /,
    ):
        if (
            not isinstance(sequences, SequenceBatch)
            or sequences.alphabet != PROTEIN_IUPAC
        ):
            raise TypeError("sequences must be a PROTEIN_IUPAC SequenceBatch.")
        proteins = np.asarray(protein_ids)
        capacity = sequences.record_capacity
        if proteins.shape != (capacity,) or not np.issubdtype(proteins.dtype, np.integer):
            raise TypeError("protein_ids must be an integer candidate-capacity vector.")
        if (
            not isinstance(modifications, ModificationBatch)
            or modifications.candidate_capacity != capacity
        ):
            raise ValueError(
                "modifications must match the proteoform candidate capacity."
            )
        active = np.asarray(sequences.case_mask)
        if np.any(proteins[active] < 0) or np.any(proteins[~active] != 0):
            raise ValueError(
                "Active protein_ids must be nonnegative and padding must be zero."
            )
        lengths = np.asarray(sequences.valid_mask).sum(axis=1)
        residue_mod = np.asarray(modifications.active_mask) & (
            np.asarray(modifications.site_kinds) == int(ModificationSiteKind.RESIDUE)
        )
        maximum = np.repeat(lengths[:, None], modifications.modification_capacity, axis=1)
        if np.any(
            np.asarray(modifications.positions)[residue_mod] >= maximum[residue_mod]
        ):
            raise ValueError(
                "Residue modification positions must be inside proteoform lengths."
            )
        self.sequences = sequences
        self.protein_ids = jnp.asarray(proteins, dtype=jnp.int64)
        self.modifications = modifications
        self.candidate_capacity = capacity


class DecoyConstructionResult(StrictModule):
    """Reversed peptide decoys with remapped modification sites."""

    decoys: PeptideCandidateBatch
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def reverse_peptide_decoys(
    targets: PeptideCandidateBatch,
    /,
) -> DecoyConstructionResult:
    """Construct charge- and modification-preserving reversed peptide decoys."""
    if not isinstance(targets, PeptideCandidateBatch):
        raise TypeError("targets must be a PeptideCandidateBatch.")
    sequences = targets.sequences
    positions = jnp.arange(sequences.sequence_capacity)[None, :]
    lengths = sequences.lengths[:, None]
    source = jnp.maximum(lengths - 1 - positions, 0)
    reversed_tokens = jnp.take_along_axis(sequences.token_codes, source, axis=1)
    pad_code = sequences.alphabet.code(sequences.alphabet.pad_symbol)
    reversed_tokens = jnp.where(sequences.valid_mask, reversed_tokens, pad_code)
    reversed_soft = jnp.take_along_axis(sequences.soft_mask, source, axis=1)
    reversed_soft = reversed_soft & sequences.valid_mask
    decoy_ids = jnp.where(
        sequences.case_mask,
        -sequences.record_ids - 1,
        0,
    )
    decoy_sequences = SequenceBatch(
        decoy_ids,
        reversed_tokens,
        sequences.valid_mask,
        sequences.case_mask,
        reversed_soft,
        sequences.alphabet,
    )
    kinds = targets.modifications.site_kinds
    residue_site = kinds == int(ModificationSiteKind.RESIDUE)
    n_terminal = kinds == int(ModificationSiteKind.N_TERMINUS)
    candidate_lengths = sequences.lengths[:, None]
    remapped_positions = jnp.where(
        residue_site,
        candidate_lengths - 1 - targets.modifications.positions,
        targets.modifications.positions,
    )
    remapped_kinds = jnp.where(
        n_terminal,
        int(ModificationSiteKind.C_TERMINUS),
        jnp.where(
            kinds == int(ModificationSiteKind.C_TERMINUS),
            int(ModificationSiteKind.N_TERMINUS),
            kinds,
        ),
    )
    remapped_positions = jnp.where(
        targets.modifications.active_mask, remapped_positions, 0
    )
    remapped_kinds = jnp.where(targets.modifications.active_mask, remapped_kinds, 0)
    decoy_modifications = ModificationBatch(
        targets.modifications.modification_ids,
        remapped_positions,
        remapped_kinds,
        targets.modifications.mass_shifts,
        targets.modifications.active_mask,
    )
    decoys = PeptideCandidateBatch(
        decoy_sequences,
        targets.charge,
        targets.neutral_mass,
        sequences.case_mask,
        decoy_modifications,
    )
    has_modifications = jnp.any(targets.modifications.active_mask)
    evidence = jnp.asarray(int(ProteomicsEvidence.REVERSED_DECOY), dtype=jnp.uint32)
    evidence = evidence | jnp.where(
        has_modifications,
        int(ProteomicsEvidence.MODIFICATIONS_REMAPPED),
        0,
    ).astype(jnp.uint32)
    return DecoyConstructionResult(
        decoys=decoys,
        valid=jnp.asarray(True),
        status=jnp.asarray(int(ProteomicsStatus.SUCCESS), dtype=jnp.int32),
        evidence=evidence,
        method_contract=_DECOY_CONTRACT,
    )


class PSMScoringInput(StrictModule):
    """Bounded physical and spectral evidence for peptide-spectrum matches."""

    psm_ids: Array
    spectrum_ids: Array
    peptide_indices: Array
    charge: Array
    precursor_error_ppm: Array
    matched_ion_count: Array
    theoretical_ion_count: Array
    explained_intensity_fraction: Array
    spectral_angle: Array
    is_decoy: Array
    active_mask: Array
    psm_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        psm_ids: ArrayLike,
        spectrum_ids: ArrayLike,
        peptide_indices: ArrayLike,
        charge: ArrayLike,
        precursor_error_ppm: ArrayLike,
        matched_ion_count: ArrayLike,
        theoretical_ion_count: ArrayLike,
        explained_intensity_fraction: ArrayLike,
        spectral_angle: ArrayLike,
        is_decoy: ArrayLike,
        active_mask: ArrayLike,
        /,
    ):
        ids = np.asarray(psm_ids)
        arrays = [
            np.asarray(spectrum_ids),
            np.asarray(peptide_indices),
            np.asarray(charge),
            np.asarray(precursor_error_ppm),
            np.asarray(matched_ion_count),
            np.asarray(theoretical_ion_count),
            np.asarray(explained_intensity_fraction),
            np.asarray(spectral_angle),
            np.asarray(is_decoy, dtype=bool),
            np.asarray(active_mask, dtype=bool),
        ]
        if (
            ids.ndim != 1
            or ids.size == 0
            or any(value.shape != ids.shape for value in arrays)
        ):
            raise ValueError("All PSM input fields must be equal non-empty vectors.")
        (
            spectrum,
            peptide,
            charges,
            ppm,
            matched,
            theoretical,
            explained,
            angle,
            decoy,
            mask,
        ) = arrays
        if not all(
            np.issubdtype(value.dtype, np.integer)
            for value in (ids, spectrum, peptide, charges, matched, theoretical)
        ):
            raise TypeError(
                "PSM identifiers, indices, charge, and ion counts must be integers."
            )
        if np.any(np.abs(charges[mask]) < 1) or np.any(np.abs(charges[mask]) > 64):
            raise ValueError(
                "Active PSM charges must be nonzero with magnitude at most 64."
            )
        if (
            np.any(matched[mask] < 0)
            or np.any(theoretical[mask] <= 0)
            or np.any(matched[mask] > theoretical[mask])
        ):
            raise ValueError("Active matched/theoretical ion counts are inconsistent.")
        if np.any(~np.isfinite(ppm[mask])):
            raise ValueError("Active precursor errors must be finite.")
        if np.any(~np.isfinite(explained[mask])) or np.any(
            (explained[mask] < 0.0) | (explained[mask] > 1.0)
        ):
            raise ValueError("Explained-intensity fractions must lie in [0, 1].")
        if np.any(~np.isfinite(angle[mask])) or np.any(
            (angle[mask] < 0.0) | (angle[mask] > 1.0)
        ):
            raise ValueError("Spectral angles must lie in [0, 1].")
        for value in (
            ids,
            spectrum,
            peptide,
            charges,
            ppm,
            matched,
            theoretical,
            explained,
            angle,
            decoy,
        ):
            if np.any(value[~mask] != 0):
                raise ValueError("Inactive PSM entries must be zero/false padding.")
        self.psm_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.spectrum_ids = jnp.asarray(spectrum, dtype=jnp.int64)
        self.peptide_indices = jnp.asarray(peptide, dtype=jnp.int32)
        self.charge = jnp.asarray(charges, dtype=jnp.int32)
        self.precursor_error_ppm = jnp.asarray(ppm)
        self.matched_ion_count = jnp.asarray(matched, dtype=jnp.int32)
        self.theoretical_ion_count = jnp.asarray(theoretical, dtype=jnp.int32)
        self.explained_intensity_fraction = jnp.asarray(explained)
        self.spectral_angle = jnp.asarray(angle)
        self.is_decoy = jnp.asarray(decoy)
        self.active_mask = jnp.asarray(mask)
        self.psm_capacity = int(ids.size)


class PSMScoringPlan(StrictModule):
    """Explicit coefficients and precursor scale for one PSM score."""

    precursor_tolerance_ppm: float = eqx.field(static=True)
    spectral_angle_weight: float = eqx.field(static=True)
    explained_intensity_weight: float = eqx.field(static=True)
    matched_ion_weight: float = eqx.field(static=True)
    precursor_error_weight: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        precursor_tolerance_ppm: float,
        spectral_angle_weight: float = 1.0,
        explained_intensity_weight: float = 1.0,
        matched_ion_weight: float = 1.0,
        precursor_error_weight: float = 1.0,
    ):
        values = np.asarray(
            [
                precursor_tolerance_ppm,
                spectral_angle_weight,
                explained_intensity_weight,
                matched_ion_weight,
                precursor_error_weight,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or values[0] <= 0.0 or np.any(values[1:] < 0.0):
            raise ValueError(
                "Tolerance must be positive and scoring weights nonnegative finite values."
            )
        self.precursor_tolerance_ppm = float(values[0])
        self.spectral_angle_weight = float(values[1])
        self.explained_intensity_weight = float(values[2])
        self.matched_ion_weight = float(values[3])
        self.precursor_error_weight = float(values[4])


class PSMScoreResult(StrictModule):
    """Composite PSM scores with their explicit component values."""

    score: Array
    matched_ion_fraction: Array
    precursor_penalty: Array
    active_mask: Array
    is_decoy: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def score_psms(
    inputs: PSMScoringInput,
    plan: PSMScoringPlan,
    /,
) -> PSMScoreResult:
    """Score every active target and decoy PSM from bounded evidence terms."""
    if not isinstance(inputs, PSMScoringInput):
        raise TypeError("inputs must be PSMScoringInput.")
    if not isinstance(plan, PSMScoringPlan):
        raise TypeError("plan must be PSMScoringPlan.")
    ion_fraction = inputs.matched_ion_count / jnp.maximum(inputs.theoretical_ion_count, 1)
    precursor_penalty = (inputs.precursor_error_ppm / plan.precursor_tolerance_ppm) ** 2
    score = (
        plan.spectral_angle_weight * inputs.spectral_angle
        + plan.explained_intensity_weight * inputs.explained_intensity_fraction
        + plan.matched_ion_weight * ion_fraction
        - plan.precursor_error_weight * precursor_penalty
    )
    score = jnp.where(inputs.active_mask, score, -jnp.inf)
    finite = jnp.all(jnp.isfinite(jnp.where(inputs.active_mask, score, 0.0)))
    evidence = int(
        ProteomicsEvidence.MASS_ERROR
        | ProteomicsEvidence.FRAGMENT_MATCH
        | ProteomicsEvidence.EXPLAINED_INTENSITY
        | ProteomicsEvidence.SPECTRAL_ANGLE
    )
    return PSMScoreResult(
        score=score,
        matched_ion_fraction=jnp.where(inputs.active_mask, ion_fraction, 0.0),
        precursor_penalty=jnp.where(inputs.active_mask, precursor_penalty, 0.0),
        active_mask=inputs.active_mask,
        is_decoy=inputs.is_decoy,
        valid=finite,
        status=jnp.where(
            finite, int(ProteomicsStatus.SUCCESS), int(ProteomicsStatus.NONFINITE)
        ).astype(jnp.int32),
        evidence=jnp.asarray(evidence, dtype=jnp.uint32),
        method_contract=_SCORE_CONTRACT,
    )


__all__ = [
    "DecoyConstructionResult",
    "ModificationBatch",
    "ModificationSiteKind",
    "PSMScoreResult",
    "PSMScoringInput",
    "PSMScoringPlan",
    "PeptideCandidateBatch",
    "ProteoformCandidateBatch",
    "ProteomicsEvidence",
    "ProteomicsStatus",
    "reverse_peptide_decoys",
    "score_psms",
]
