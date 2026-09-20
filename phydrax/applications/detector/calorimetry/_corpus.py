#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....qualification import ReferenceArtifactManifest
from ._geometry import CalorimeterGeometry


@dataclass(frozen=True)
class CalorimeterCorpus:
    geometry: CalorimeterGeometry
    cell_energies: object
    conditions: object
    condition_names: tuple[str, ...]
    ledger: object
    record_ids: tuple[str, ...]
    source_manifest_ids: tuple[str, ...]
    split_group_ids: tuple[str, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    rights: tuple[ReferenceArtifactManifest, ...]
    dataset_id: str


def prepare_calorimeter_corpus(
    geometry: CalorimeterGeometry,
    cell_energies,
    conditions,
    /,
    *,
    condition_names,
    ledger,
    record_ids,
    source_manifest_ids,
    split_group_ids,
    validation_groups,
    test_groups,
    rights,
    commercial_use: bool = False,
    maximum_records: int = 10_000_000,
) -> CalorimeterCorpus:
    """Admit a rights-governed, group-disjoint fixed-cell calorimeter corpus."""
    if not isinstance(geometry, CalorimeterGeometry):
        raise TypeError("geometry must be CalorimeterGeometry.")
    manifests = tuple(rights)
    if not manifests or any(
        not isinstance(value, ReferenceArtifactManifest) for value in manifests
    ):
        raise ValueError(
            "Calorimeter corpora require explicit reference rights manifests."
        )
    admitted = tuple(
        manifest.require_rights(training_use=True, commercial_use=commercial_use)
        for manifest in manifests
    )
    showers = np.asarray(cell_energies, dtype=np.float64)
    context = np.asarray(conditions, dtype=np.float64)
    ledger_ = np.asarray(ledger, dtype=np.float64)
    if showers.ndim != 2 or showers.shape[1] != geometry.cell_count:
        raise ValueError("cell_energies must have shape (record, geometry.cell_count).")
    count = showers.shape[0]
    maximum = int(maximum_records)
    if count < 3 or maximum < 3 or count > maximum:
        raise ValueError("Corpus record count exceeds its finite resource policy.")
    names = tuple(str(value).strip() for value in condition_names)
    if not names or any(not value for value in names) or len(set(names)) != len(names):
        raise ValueError("condition_names must be distinct non-empty strings.")
    if context.shape != (count, len(names)) or ledger_.shape != (count, 5):
        raise ValueError(
            "Conditions or visible/leakage/dead/outside/unmapped ledger shape is invalid."
        )
    if (
        np.any(~np.isfinite(showers))
        or np.any(showers < 0.0)
        or np.any(~np.isfinite(context))
        or np.any(~np.isfinite(ledger_))
        or np.any(ledger_ < 0.0)
    ):
        raise ValueError(
            "Corpus showers, conditions, and ledgers must be finite and nonnegative where required."
        )
    if np.any(showers[:, ~np.asarray(geometry.active)] != 0.0):
        raise ValueError("Inactive calorimeter cells must be exactly zero in the corpus.")
    ids = tuple(str(value).strip() for value in record_ids)
    sources = tuple(str(value).strip() for value in source_manifest_ids)
    groups = tuple(str(value).strip() for value in split_group_ids)
    if any(
        len(values) != count or any(not value for value in values)
        for values in (ids, sources, groups)
    ):
        raise ValueError(
            "Every corpus record requires identity, source, and split group."
        )
    if len(set(ids)) != count or not set(sources) <= set(admitted):
        raise ValueError("Record identities must be unique and sources rights-admitted.")
    validation = frozenset(str(value) for value in validation_groups)
    test = frozenset(str(value) for value in test_groups)
    if (
        not validation
        or not test
        or validation & test
        or not (validation | test) <= set(groups)
    ):
        raise ValueError(
            "Validation and test groups must be non-empty disjoint corpus groups."
        )
    train = tuple(
        index for index, group in enumerate(groups) if group not in validation | test
    )
    validation_indices = tuple(
        index for index, group in enumerate(groups) if group in validation
    )
    test_indices = tuple(index for index, group in enumerate(groups) if group in test)
    if not train or not validation_indices or not test_indices:
        raise ValueError(
            "Corpus requires non-empty train, validation, and test partitions."
        )
    fingerprints = tuple(
        array_tree_fingerprint(showers[index])["sha256"] for index in range(count)
    )
    partitions = (
        {fingerprints[index] for index in train},
        {fingerprints[index] for index in validation_indices},
        {fingerprints[index] for index in test_indices},
    )
    if (
        partitions[0] & partitions[1]
        or partitions[0] & partitions[2]
        or partitions[1] & partitions[2]
    ):
        raise ValueError("Identical showers leak across corpus partitions.")
    dataset_id = canonical_fingerprint(
        {
            "kind": "calorimeter-corpus",
            "geometry": geometry.geometry_id,
            "arrays": array_tree_fingerprint((showers, context, ledger_)),
            "condition_names": list(names),
            "records": list(ids),
            "sources": list(sources),
            "groups": list(groups),
            "validation_groups": sorted(validation),
            "test_groups": sorted(test),
            "rights": list(admitted),
        }
    )
    return CalorimeterCorpus(
        geometry,
        jnp.asarray(showers),
        jnp.asarray(context),
        names,
        jnp.asarray(ledger_),
        ids,
        sources,
        groups,
        train,
        validation_indices,
        test_indices,
        manifests,
        dataset_id,
    )


__all__ = ["CalorimeterCorpus", "prepare_calorimeter_corpus"]
