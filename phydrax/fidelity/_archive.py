#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from .._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ._dataset import FidelityCaseSpec, FidelityDataset
from ._execution import FidelityEvaluation


_ARCHIVE_KIND = "phydrax-fidelity-dataset"


def write_fidelity_dataset(
    path: str | os.PathLike[str],
    dataset: FidelityDataset,
    /,
) -> Path:
    """Write a pickle-free, content-addressed fidelity observation corpus."""

    if not isinstance(dataset, FidelityDataset):
        raise TypeError("dataset must be a FidelityDataset.")
    arrays: dict[str, Any] = {}
    cases = []
    for index, case in enumerate(dataset.cases):
        inputs = pack_array_tree(f"case/{index}/inputs", case.inputs, arrays)
        arrays[f"case/{index}/log_weight"] = case.log_weight
        cases.append(
            {
                "case_id": case.case_id,
                "split_group_id": case.split_group_id,
                "input_id": case.input_id,
                "metadata": [list(item) for item in case.metadata],
                "case_fingerprint": case.case_fingerprint,
                "inputs": inputs,
                "log_weight": f"case/{index}/log_weight",
            }
        )
    evaluations = []
    for index, evaluation in enumerate(dataset.evaluations):
        observable = pack_array_tree(
            f"evaluation/{index}/observable",
            evaluation.observable,
            arrays,
        )
        arrays[f"evaluation/{index}/valid"] = evaluation.valid
        arrays[f"evaluation/{index}/cost"] = evaluation.cost
        evaluations.append(
            {
                "case_id": evaluation.case_id,
                "pair_id": evaluation.pair_id,
                "level_id": evaluation.level_id,
                "evaluator_id": evaluation.evaluator_id,
                "cost_unit": evaluation.cost_unit,
                "artifact_id": evaluation.artifact_id,
                "evidence_ids": list(evaluation.evidence_ids),
                "evaluation_id": evaluation.evaluation_id,
                "observable": observable,
                "valid": f"evaluation/{index}/valid",
                "cost": f"evaluation/{index}/cost",
            }
        )
    return write_array_archive(
        path,
        manifest={
            "kind": _ARCHIVE_KIND,
            "dataset_id": dataset.dataset_id,
            "hierarchy_id": dataset.hierarchy.hierarchy_id,
            "hierarchy_fingerprint": dataset.hierarchy.fingerprint,
            "cost_unit": dataset.cost_unit,
            "cases": cases,
            "evaluations": evaluations,
        },
        arrays=arrays,
    )


def read_fidelity_dataset(
    path: str | os.PathLike[str],
    template: FidelityDataset,
    /,
) -> FidelityDataset:
    """Restore a fidelity corpus against exact hierarchy and PyTree templates."""

    if not isinstance(template, FidelityDataset):
        raise TypeError("template must be a FidelityDataset.")
    manifest, arrays = read_array_archive(path)
    expected = {
        "kind",
        "dataset_id",
        "hierarchy_id",
        "hierarchy_fingerprint",
        "cost_unit",
        "cases",
        "evaluations",
        "arrays",
    }
    if set(manifest) != expected or manifest.get("kind") != _ARCHIVE_KIND:
        raise ValueError("Archive is not a canonical PhydraX fidelity dataset.")
    if (
        manifest.get("hierarchy_id") != template.hierarchy.hierarchy_id
        or manifest.get("hierarchy_fingerprint") != template.hierarchy.fingerprint
    ):
        raise ValueError("Fidelity archive hierarchy identity mismatch.")
    case_records = manifest.get("cases")
    evaluation_records = manifest.get("evaluations")
    if not isinstance(case_records, list) or len(case_records) != len(template.cases):
        raise ValueError("Fidelity archive case inventory is invalid.")
    if not isinstance(evaluation_records, list) or len(evaluation_records) != len(
        template.evaluations
    ):
        raise ValueError("Fidelity archive evaluation inventory is invalid.")
    cases = []
    for record, template_case in zip(case_records, template.cases, strict=True):
        if not isinstance(record, dict):
            raise TypeError("Fidelity archive case record is invalid.")
        inputs = unpack_array_tree(record["inputs"], arrays, template_case.inputs)
        case = FidelityCaseSpec(
            inputs,
            case_id=record["case_id"],
            split_group_id=record["split_group_id"],
            input_id=record["input_id"],
            log_weight=jnp.asarray(arrays[record["log_weight"]]),
            metadata=dict(record["metadata"]),
        )
        if case.case_fingerprint != record["case_fingerprint"]:
            raise ValueError("Fidelity archive case content identity mismatch.")
        cases.append(case)
    evaluations = []
    for record, template_evaluation in zip(
        evaluation_records,
        template.evaluations,
        strict=True,
    ):
        if not isinstance(record, dict):
            raise TypeError("Fidelity archive evaluation record is invalid.")
        observable = unpack_array_tree(
            record["observable"],
            arrays,
            template_evaluation.observable,
        )
        evaluation = FidelityEvaluation(
            observable,
            case_id=record["case_id"],
            pair_id=record["pair_id"],
            level_id=record["level_id"],
            evaluator_id=record["evaluator_id"],
            valid=jnp.asarray(arrays[record["valid"]]),
            cost=jnp.asarray(arrays[record["cost"]]),
            cost_unit=record["cost_unit"],
            artifact_id=record["artifact_id"],
            evidence_ids=tuple(record["evidence_ids"]),
        )
        if evaluation.evaluation_id != record["evaluation_id"]:
            raise ValueError("Fidelity archive evaluation content identity mismatch.")
        evaluations.append(evaluation)
    dataset = FidelityDataset(template.hierarchy, tuple(cases), tuple(evaluations))
    if dataset.dataset_id != manifest.get(
        "dataset_id"
    ) or dataset.cost_unit != manifest.get("cost_unit"):
        raise ValueError("Fidelity archive dataset content identity mismatch.")
    return dataset


__all__ = ["read_fidelity_dataset", "write_fidelity_dataset"]
