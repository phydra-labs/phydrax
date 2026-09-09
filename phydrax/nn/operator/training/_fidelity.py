#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

from ....fidelity import FidelityPath
from ...operator.data import OperatorCaseProvenance
from ._dataset import OperatorDataset


@dataclass(frozen=True)
class FidelityOperatorPreparation:
    """Paired target dataset and explicit unmatched-case evidence."""

    dataset: OperatorDataset
    paired_low_indices: tuple[int, ...]
    paired_target_indices: tuple[int, ...]
    low_only_case_ids: tuple[str, ...]
    target_only_case_ids: tuple[str, ...]
    path_id: str
    source_level_id: str
    target_level_id: str
    identity_key: str


def prepare_fidelity_operator_dataset(
    low_dataset: OperatorDataset,
    target_dataset: OperatorDataset,
    path: FidelityPath,
    /,
    *,
    identity_key: str = "physical_case_id",
    source_level_id: str | None = None,
    target_level_id: str | None = None,
    require_paired_target: bool = True,
) -> FidelityOperatorPreparation:
    """Join low and target operator cases without silently discarding target cases."""

    if not isinstance(low_dataset, OperatorDataset) or not isinstance(
        target_dataset, OperatorDataset
    ):
        raise TypeError("low_dataset and target_dataset must be OperatorDataset values.")
    if not isinstance(path, FidelityPath):
        raise TypeError("path must be a FidelityPath.")
    key = str(identity_key)
    if not key:
        raise ValueError("identity_key must be non-empty.")
    source = path.levels[0].level_id if source_level_id is None else str(source_level_id)
    target = path.target.level_id if target_level_id is None else str(target_level_id)
    if source not in path.level_ids or target not in path.level_ids:
        raise ValueError("Source and target levels must lie on the fidelity path.")
    if path.level_ids.index(source) >= path.level_ids.index(target):
        raise ValueError("Source fidelity must precede target fidelity on the path.")
    assert low_dataset.provenance is not None
    assert target_dataset.provenance is not None
    low_by_identity = _identity_index(low_dataset.provenance, key, owner="low")
    target_by_identity = _identity_index(target_dataset.provenance, key, owner="target")
    shared = set(low_by_identity) & set(target_by_identity)
    target_only = tuple(
        record.case_id
        for record in target_dataset.provenance
        if record.identities[key] not in shared
    )
    if require_paired_target and target_only:
        raise ValueError(
            "Every target-fidelity training case requires a paired low-fidelity case; "
            f"unpaired target cases: {target_only}."
        )
    paired_target_indices = tuple(
        index
        for index, record in enumerate(target_dataset.provenance)
        if record.identities[key] in shared
    )
    if not paired_target_indices:
        raise ValueError("Low and target operator datasets have no paired cases.")
    paired_low_indices = tuple(
        low_by_identity[target_dataset.provenance[index].identities[key]]
        for index in paired_target_indices
    )
    low_only = tuple(
        record.case_id
        for record in low_dataset.provenance
        if record.identities[key] not in shared
    )
    paired = target_dataset.take(paired_target_indices)
    assert paired.provenance is not None
    provenance = tuple(
        OperatorCaseProvenance(
            record.case_id,
            identities={
                **dict(record.identities),
                "fidelity_path_id": path.path_id,
                "source_fidelity_id": source,
                "target_fidelity_id": target,
            },
            order=record.order,
        )
        for record in paired.provenance
    )
    prepared = OperatorDataset(
        paired.batch,
        paired.targets,
        provenance,
        case_log_weights=paired.case_log_weights,
        case_mask=paired.case_mask,
    )
    return FidelityOperatorPreparation(
        dataset=prepared,
        paired_low_indices=paired_low_indices,
        paired_target_indices=paired_target_indices,
        low_only_case_ids=low_only,
        target_only_case_ids=target_only,
        path_id=path.path_id,
        source_level_id=source,
        target_level_id=target,
        identity_key=key,
    )


def _identity_index(
    provenance: tuple[OperatorCaseProvenance, ...],
    key: str,
    /,
    *,
    owner: str,
) -> dict[str, int]:
    missing = tuple(
        record.case_id for record in provenance if key not in record.identities
    )
    if missing:
        raise ValueError(
            f"Every {owner} case must define identity {key!r}; missing cases: {missing}."
        )
    result: dict[str, int] = {}
    for index, record in enumerate(provenance):
        identity = record.identities[key]
        if identity in result:
            raise ValueError(f"{owner} dataset identity {identity!r} is not one-to-one.")
        result[identity] = index
    return result


__all__ = ["FidelityOperatorPreparation", "prepare_fidelity_operator_dataset"]
