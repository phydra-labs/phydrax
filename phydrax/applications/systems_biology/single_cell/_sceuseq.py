#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Caller-array adapter for admitted four-matrix scEU-seq exports."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.interchange import AdapterReport, AdapterStatus
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import conversion_factor, SECOND, UnitDefinition

from ._labeled_assay import LABELED_CHANNELS, LabeledTranscriptCounts
from ._scenario import _identity, _label, GeneIdentity


@dataclass(frozen=True, slots=True)
class ScEUSeqPrerequisiteReport:
    """Admission gaps for the exact four exported count matrices."""

    missing: tuple[str, ...]
    rights_refusals: tuple[str, ...]
    unquantified_uncertainty: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return (
            not self.missing
            and not self.rights_refusals
            and not self.unquantified_uncertainty
        )


def sceu_seq_prerequisites(
    manifests: tuple[ReferenceArtifactManifest | None, ...],
    /,
    *,
    commercial_use: bool = False,
    training_use: bool = False,
    redistribution: bool = False,
    export: bool = False,
) -> ScEUSeqPrerequisiteReport:
    """Report missing rights/uncertainty without inventing admission for GSE128365."""

    if not isinstance(manifests, tuple) or len(manifests) != 4:
        raise ValueError("Exactly four channel manifests must be declared.")
    missing: list[str] = []
    refusals: list[str] = []
    unquantified: list[str] = []
    for channel, manifest in zip(LABELED_CHANNELS, manifests, strict=True):
        if manifest is None:
            missing.append(f"source-manifest:{channel}")
            continue
        if not isinstance(manifest, ReferenceArtifactManifest):
            raise TypeError(
                "Channel manifests must be ReferenceArtifactManifest or None."
            )
        refusals.extend(
            f"{channel}:{reason}"
            for reason in manifest.rights_refusal_reasons(
                commercial_use=commercial_use,
                training_use=training_use,
                redistribution=redistribution,
                export=export,
            )
        )
        if manifest.uncertainty is None:
            unquantified.append(f"measurement-uncertainty:{channel}")
    return ScEUSeqPrerequisiteReport(tuple(missing), tuple(refusals), tuple(unquantified))


@dataclass(frozen=True, slots=True)
class ImportedScEUSeq:
    """Lossless four-channel raw counts with culture, plate, and time identity."""

    counts: Array
    valid: Array
    gene_ids: tuple[str, ...]
    cell_ids: tuple[int, ...]
    culture_ids: tuple[str, ...]
    plate_ids: tuple[str, ...]
    times: Array
    time_unit: UnitDefinition
    manifests: tuple[ReferenceArtifactManifest, ...]
    preprocessing_id: str
    source_id: str
    report: AdapterReport

    def gene_counts(
        self,
        gene_index: int,
        gene: GeneIdentity,
        /,
        *,
        assay_id: str,
    ) -> LabeledTranscriptCounts:
        if not isinstance(gene_index, int) or not 0 <= gene_index < len(self.gene_ids):
            raise ValueError("gene_index is outside the imported gene support.")
        if gene.label != self.gene_ids[gene_index]:
            raise ValueError("Gene identity does not match the selected source column.")
        return LabeledTranscriptCounts(
            gene,
            self.cell_ids,
            self.counts[:, gene_index],
            culture_ids=self.culture_ids,
            plate_ids=self.plate_ids,
            times=self.times,
            time_unit=self.time_unit,
            assay_id=assay_id,
            source_id=self.source_id,
            preprocessing_id=self.preprocessing_id,
            source_parent_ids=tuple(
                sorted(
                    {
                        lineage_id
                        for manifest in self.manifests
                        for lineage_id in manifest.lineage_ids
                    }
                )
            ),
            preprocessing_parent_ids=tuple(
                sorted(manifest.manifest_id for manifest in self.manifests)
            ),
            valid=self.valid[:, gene_index],
        )


def import_sceu_seq_arrays(
    labeled_unspliced: ArrayLike,
    labeled_spliced: ArrayLike,
    unlabeled_unspliced: ArrayLike,
    unlabeled_spliced: ArrayLike,
    /,
    *,
    gene_ids: tuple[str, ...],
    cell_ids: tuple[int, ...],
    culture_ids: tuple[str, ...],
    plate_ids: tuple[str, ...],
    times: ArrayLike,
    time_unit: UnitDefinition,
    manifests: tuple[ReferenceArtifactManifest | None, ...],
    preprocessing_id: str,
    valid: ArrayLike | None = None,
    commercial_use: bool = False,
    training_use: bool = False,
    redistribution: bool = False,
    export: bool = False,
) -> ImportedScEUSeq:
    """Admit caller-extracted raw matrices; no download, alignment, or normalization."""

    prerequisites = sceu_seq_prerequisites(
        manifests,
        commercial_use=commercial_use,
        training_use=training_use,
        redistribution=redistribution,
        export=export,
    )
    if prerequisites.missing or prerequisites.unquantified_uncertainty:
        raise ValueError(
            "scEU-seq admission prerequisites are incomplete: "
            + ";".join(prerequisites.missing + prerequisites.unquantified_uncertainty)
        )
    if prerequisites.rights_refusals:
        raise PermissionError(
            "scEU-seq requested use is not admitted: "
            + ";".join(prerequisites.rights_refusals)
        )
    admitted = tuple(manifests)
    arrays = tuple(
        np.asarray(value)
        for value in (
            labeled_unspliced,
            labeled_spliced,
            unlabeled_unspliced,
            unlabeled_spliced,
        )
    )
    genes = tuple(_label(value, "gene_id") for value in gene_ids)
    cells = tuple(_identity(value, "cell_id") for value in cell_ids)
    cultures = tuple(_label(value, "culture_id") for value in culture_ids)
    plates = tuple(_label(value, "plate_id") for value in plate_ids)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("Gene IDs must be nonempty and unique.")
    if not cells or len(set(cells)) != len(cells):
        raise ValueError("Cell IDs must be nonempty and unique.")
    expected = (len(cells), len(genes))
    if any(array.shape != expected or array.dtype.kind not in "ifu" for array in arrays):
        raise ValueError("Every raw channel matrix must have shape (cell, gene).")
    stacked = np.stack(arrays, axis=-1)
    mask = np.ones(stacked.shape, dtype=bool) if valid is None else np.asarray(valid)
    if mask.shape != stacked.shape or mask.dtype != bool:
        raise ValueError("Validity must be a boolean mask matching all four matrices.")
    active = stacked[mask]
    if (
        np.any(~np.isfinite(active))
        or np.any(active < 0.0)
        or np.any(active != np.floor(active))
    ):
        raise ValueError("Active matrix entries must be raw nonnegative integer counts.")
    coordinates = np.asarray(times, dtype=float)
    if (
        len(cultures) != len(cells)
        or len(plates) != len(cells)
        or coordinates.shape != (len(cells),)
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError("Culture, plate, and physical time must be aligned per cell.")
    conversion_factor(time_unit, SECOND)
    preprocessing = _label(preprocessing_id, "preprocessing_id")
    source_id = canonical_fingerprint(
        {
            "kind": "sceu-seq-four-channel-source",
            "channels": LABELED_CHANNELS,
            "manifests": tuple(manifest.manifest_id for manifest in admitted),
            "genes": genes,
            "cells": cells,
            "cultures": cultures,
            "plates": plates,
            "times": coordinates.tolist(),
            "time_unit": time_unit.unit_id,
            "counts": array_tree_fingerprint(np.where(mask, stacked, 0)),
            "valid": array_tree_fingerprint(mask),
            "preprocessing": preprocessing,
        }
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "caller-extracted-sceu-seq-four-matrix",
        "native-labeled-transcript-counts",
        source_id=source_id,
        target_id=source_id,
        coordinate_mapping=(
            "source-row->declared-cell-id",
            "source-column->declared-gene-id",
            "source-matrix->declared-label-splice-channel",
        ),
        preserved_fields=(
            *LABELED_CHANNELS,
            "valid",
            "culture_ids",
            "plate_ids",
            "physical_times",
            "source_rights",
            "source_uncertainty",
            "preprocessing_id",
        ),
        assumptions=(
            "Caller-supplied arrays are unnormalized UMI count matrices.",
            "Cells are snapshots grouped by declared cultures and plates.",
            "No source rights, labeling efficiency, lineage, or clock is inferred.",
        ),
    )
    return ImportedScEUSeq(
        jnp.asarray(np.where(mask, stacked, 0), dtype=float),
        jnp.asarray(mask),
        genes,
        cells,
        cultures,
        plates,
        jnp.asarray(coordinates),
        time_unit,
        admitted,
        preprocessing,
        source_id,
        report,
    )


__all__ = [
    "ImportedScEUSeq",
    "ScEUSeqPrerequisiteReport",
    "import_sceu_seq_arrays",
    "sceu_seq_prerequisites",
]
