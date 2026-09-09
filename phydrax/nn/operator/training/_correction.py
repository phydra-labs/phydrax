#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import numpy as np

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ..data import function_samples_with_values, OperatorBatch
from ..sampling import OperatorCase
from ._dataset import operator_dataset_from_cases, OperatorDataset


def _stopped_batch(batch: OperatorBatch, /) -> OperatorBatch:
    def stopped(samples):
        return function_samples_with_values(
            samples,
            None if samples.values is None else jax.lax.stop_gradient(samples.values),
        )

    return OperatorBatch(
        inputs={name: stopped(samples) for name, samples in batch.inputs.items()},
        queries={name: stopped(samples) for name, samples in batch.queries.items()},
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )


def _require_finite_active_inputs(case: OperatorCase, /) -> None:
    for name, samples in case.batch.inputs.items():
        if samples.values is None:
            raise ValueError(f"Residual corpus source {name!r} requires values.")
        values = np.asarray(samples.values)
        mask = np.asarray(samples.mask_array(), dtype=bool)
        trailing = (1,) * (values.ndim - mask.ndim)
        active = np.broadcast_to(mask.reshape(mask.shape + trailing), values.shape)
        if not np.all(np.isfinite(values[active])):
            raise ValueError(
                f"Residual corpus source {name!r} has non-finite active values."
            )


class OperatorResidualCorpus(StrictModule, NonTrainableState):
    """Immutable targetless residual cases with solver-owned provenance."""

    dataset: OperatorDataset
    binding_id: str = eqx.field(static=True)
    task_fingerprint: str = eqx.field(static=True)
    residual_loss_fingerprint: str = eqx.field(static=True)
    source_artifact_ids: tuple[str, ...] = eqx.field(static=True)
    corpus_id: str = eqx.field(static=True)

    def __init__(
        self,
        dataset: OperatorDataset,
        /,
        *,
        binding_id: str,
        task_fingerprint: str,
        residual_loss_fingerprint: str,
        source_artifact_ids: Sequence[str],
        corpus_id: str,
    ):
        if not isinstance(dataset, OperatorDataset):
            raise TypeError("dataset must be an OperatorDataset.")
        if dataset.targets.fields:
            raise ValueError("Operator residual corpora must be targetless.")
        identifiers = tuple(
            str(value)
            for value in (
                binding_id,
                task_fingerprint,
                residual_loss_fingerprint,
                corpus_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Operator residual corpus IDs must be non-empty.")
        artifacts = tuple(sorted(str(value) for value in source_artifact_ids))
        if not artifacts or any(not value for value in artifacts):
            raise ValueError("Residual corpus source artifact IDs must be non-empty.")
        if len(set(artifacts)) != len(artifacts):
            raise ValueError("Residual corpus source artifact IDs must be unique.")
        self.dataset = dataset
        (
            self.binding_id,
            self.task_fingerprint,
            self.residual_loss_fingerprint,
            self.corpus_id,
        ) = identifiers
        self.source_artifact_ids = artifacts

    @property
    def training_provenance(self) -> dict[str, object]:
        """Return canonical metadata for ``fit_operator`` and artifact publication."""
        return {
            "operator_correction_binding_id": self.binding_id,
            "operator_residual_corpus_id": self.corpus_id,
            "residual_loss_fingerprint": self.residual_loss_fingerprint,
            "source_artifact_ids": list(self.source_artifact_ids),
        }


def prepare_operator_residual_corpus(
    cases: Sequence[OperatorCase],
    /,
    *,
    binding_id: str,
    task_fingerprint: str,
    residual_loss_fingerprint: str,
    source_artifact_ids: Sequence[str],
    required_identity_keys: Sequence[str],
) -> OperatorResidualCorpus:
    """Collate finite, targetless solver residual cases with stopped gradients."""
    cases_ = tuple(cases)
    if not cases_:
        raise ValueError("Operator residual corpus cases must be non-empty.")
    if any(not isinstance(case, OperatorCase) for case in cases_):
        raise TypeError("cases must contain only OperatorCase values.")
    keys = tuple(sorted(str(value) for value in required_identity_keys))
    if not keys or any(not value for value in keys):
        raise ValueError("Required residual provenance keys must be non-empty.")
    if len(set(keys)) != len(keys):
        raise ValueError("Required residual provenance keys must be unique.")

    stopped_cases = []
    provenance_payload = []
    for case in cases_:
        if not case.case_active or not np.isfinite(case.case_log_weight):
            raise ValueError("Residual corpus cases must be active with finite weights.")
        if case.targets.fields:
            raise ValueError("Residual corpus cases must not contain solution labels.")
        if case.provenance is None:
            raise ValueError("Residual corpus cases require explicit provenance.")
        missing = tuple(key for key in keys if key not in case.provenance.identities)
        if missing:
            raise ValueError(
                f"Residual corpus case {case.provenance.case_id!r} is missing "
                f"provenance identities {missing}."
            )
        _require_finite_active_inputs(case)
        stopped_cases.append(
            OperatorCase(
                _stopped_batch(case.batch),
                case.targets,
                provenance=case.provenance,
                case_log_weight=case.case_log_weight,
                case_active=True,
            )
        )
        provenance_payload.append(
            {
                "case_id": case.provenance.case_id,
                "identities": dict(case.provenance.identities),
                "order": dict(case.provenance.order),
            }
        )

    dataset = operator_dataset_from_cases(
        tuple(case.batch for case in stopped_cases),
        tuple(case.targets for case in stopped_cases),
        case_axis="residual_case",
        provenance=tuple(case.provenance for case in stopped_cases),
        case_log_weights=tuple(case.case_log_weight for case in stopped_cases),
        case_mask=tuple(case.case_active for case in stopped_cases),
    )
    corpus_id = canonical_fingerprint(
        {
            "kind": "operator-residual-corpus",
            "binding": str(binding_id),
            "task": str(task_fingerprint),
            "residual_loss": str(residual_loss_fingerprint),
            "source_artifacts": sorted(str(value) for value in source_artifact_ids),
            "required_identity_keys": list(keys),
            "provenance": provenance_payload,
            "arrays": array_tree_fingerprint(
                (
                    dataset.batch,
                    dataset.targets,
                    dataset.case_log_weights,
                    dataset.case_mask,
                )
            ),
        }
    )
    return OperatorResidualCorpus(
        dataset,
        binding_id=binding_id,
        task_fingerprint=task_fingerprint,
        residual_loss_fingerprint=residual_loss_fingerprint,
        source_artifact_ids=source_artifact_ids,
        corpus_id=corpus_id,
    )


__all__ = [
    "OperatorResidualCorpus",
    "prepare_operator_residual_corpus",
]
