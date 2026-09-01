#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Executable, immutable reduced-order-model workflows."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..lifecycle import NumericRevision
from ._core import (
    _array_record,
    _fingerprint,
    _freeze_array,
    CorpusPartition,
    FloatArray,
    ROMCaseSpec,
    ROMCorpus,
    TruthModel,
    TruthSample,
)
from ._profiles import (
    CertificateKind,
    LinearCoerciveRBProfile,
    LinearPODProfile,
    ParametricCertifiedProfile,
    profile_descriptor,
    ROMProfile,
)


@dataclass(frozen=True, slots=True)
class ROMArtifact:
    """A content-addressed, read-only trained ROM and its lifecycle revision."""

    corpus_id: str
    validity_id: str
    profile: ROMProfile
    basis: FloatArray
    singular_values: FloatArray
    state_weight: FloatArray
    training_case_ids: tuple[str, ...]
    lifecycle_revision: NumericRevision
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        basis = _freeze_array(self.basis, name="basis", ndim=2)
        singular_values = _freeze_array(
            self.singular_values, name="singular_values", ndim=1
        )
        weight = _freeze_array(self.state_weight, name="state_weight", ndim=2)
        assert basis is not None and singular_values is not None and weight is not None
        if (
            not self.corpus_id.strip()
            or not self.validity_id.strip()
            or not self.training_case_ids
        ):
            raise ValueError(
                "ROM artifacts require corpus, validity, and training-case identities."
            )
        if basis.shape[1] != singular_values.size or weight.shape != (
            basis.shape[0],
            basis.shape[0],
        ):
            raise ValueError(
                "ROM basis, singular values, and state weight are incompatible."
            )
        if (
            not np.allclose(weight, weight.T, rtol=1e-9, atol=1e-11)
            or np.min(np.linalg.eigvalsh(weight)) <= 0.0
        ):
            raise ValueError("state_weight must be symmetric positive definite.")
        if not np.allclose(
            basis.T @ weight @ basis, np.eye(basis.shape[1]), rtol=1e-8, atol=1e-10
        ):
            raise ValueError("ROM basis must be state-weight orthonormal.")
        if not isinstance(self.lifecycle_revision, NumericRevision):
            raise TypeError("ROM artifacts require a lifecycle NumericRevision.")
        object.__setattr__(self, "corpus_id", self.corpus_id.strip())
        object.__setattr__(self, "validity_id", self.validity_id.strip())
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "state_weight", weight)
        object.__setattr__(self, "training_case_ids", tuple(self.training_case_ids))
        object.__setattr__(
            self,
            "artifact_id",
            _fingerprint(
                {
                    "kind": "rom-artifact",
                    "corpus_id": self.corpus_id,
                    "validity_id": self.validity_id,
                    "profile": profile_descriptor(self.profile),
                    "basis": _array_record(basis),
                    "singular_values": _array_record(singular_values),
                    "state_weight": _array_record(weight),
                    "training_case_ids": list(self.training_case_ids),
                    "lifecycle_revision_id": self.lifecycle_revision.revision_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ROMEvaluation:
    artifact_id: str
    case: ROMCaseSpec
    state: FloatArray
    residual_dual_norm: float | None
    stability_lower_bound: float | None
    error_bound: float | None
    certificate_kind: CertificateKind
    qoi: float | None
    source: Literal["rom", "truth-fallback"]
    lifecycle_revision: NumericRevision

    def __post_init__(self) -> None:
        state = _freeze_array(self.state, name="evaluation state", ndim=1)
        assert state is not None
        values = (
            self.residual_dual_norm,
            self.stability_lower_bound,
            self.error_bound,
            self.qoi,
        )
        if any(value is not None and not np.isfinite(value) for value in values):
            raise ValueError("Evaluation values must be finite.")
        if (
            self.error_bound is not None
            and self.certificate_kind is not CertificateKind.ERROR_BOUND
        ):
            raise ValueError("Only an error-bound certificate may carry an error bound.")
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "case",
            ROMCaseSpec(self.case.case_id, self.case.parameters, self.case.geometry_id),
        )


@dataclass(frozen=True, slots=True)
class ROMAudit:
    artifact_id: str
    case_id: str
    relative_state_error: float
    qoi_error: float | None
    error_bound: float | None
    bound_holds: bool | None
    lifecycle_revision: NumericRevision


def train_profile(
    corpus: ROMCorpus, profile: ROMProfile, /, *, state_weight: ArrayLike | None = None
) -> ROMArtifact:
    """Train a linear POD/RB basis from immutable training snapshots.

    RB profiles use the same stable weighted snapshot subspace; certification is
    decided only at evaluation from supplied residual and coercivity evidence.
    """
    if not isinstance(corpus, ROMCorpus):
        raise TypeError("corpus must be a ROMCorpus.")
    train = corpus.cases_in(CorpusPartition.TRAIN)
    snapshots = np.column_stack([case.sample.state for case in train])
    n = snapshots.shape[0]
    if any(case.sample.state.size != n for case in train):
        raise ValueError("Training states must share one reference state dimension.")
    weight = (
        np.eye(n)
        if state_weight is None
        else _freeze_array(state_weight, name="state_weight", ndim=2)
    )
    assert weight is not None
    if (
        weight.shape != (n, n)
        or not np.allclose(weight, weight.T)
        or np.min(np.linalg.eigvalsh(weight)) <= 0
    ):
        raise ValueError(
            "state_weight must be symmetric positive definite on the state space."
        )
    factor = np.linalg.cholesky(weight).T
    left, singular, _ = np.linalg.svd(factor @ snapshots, full_matrices=False)
    requested = _basis_size(profile, singular)
    available_energy = (
        np.cumsum(singular * singular) / np.sum(singular * singular)
        if np.any(singular)
        else np.ones_like(singular)
    )
    if isinstance(profile, LinearPODProfile):
        requested = min(
            requested, int(np.searchsorted(available_energy, profile.retained_energy) + 1)
        )
    rank = min(
        requested,
        int(np.count_nonzero(singular > np.finfo(float).eps * singular[0]))
        if singular.size
        else 0,
    )
    if rank == 0:
        raise ValueError("Training snapshots have zero weighted rank.")
    basis = np.linalg.solve(factor, left[:, :rank])
    digest = _fingerprint(
        {
            "corpus_id": corpus.manifest.corpus_id,
            "profile": profile_descriptor(profile),
            "basis": _array_record(basis),
        }
    )
    revision = NumericRevision(
        digest,
        label="rom-training",
        metadata={
            "corpus_id": corpus.manifest.corpus_id,
            "profile": profile.profile_name.value,
        },
    )
    return ROMArtifact(
        corpus.manifest.corpus_id,
        corpus.validity.validity_id,
        profile,
        basis,
        singular[:rank],
        weight,
        tuple(c.spec.case_id for c in train),
        revision,
    )


def serialize(artifact: ROMArtifact, path: str | Path, /) -> Path:
    """Persist a ROM with checksummed numeric payloads and portable metadata."""
    if not isinstance(artifact, ROMArtifact):
        raise TypeError("artifact must be a ROMArtifact.")
    target = Path(path)
    metadata = {
        "artifact_id": artifact.artifact_id,
        "corpus_id": artifact.corpus_id,
        "validity_id": artifact.validity_id,
        "profile": profile_descriptor(artifact.profile),
        "training_case_ids": artifact.training_case_ids,
        "revision": {
            "content_digest": artifact.lifecycle_revision.content_digest,
            "label": artifact.lifecycle_revision.label,
            "metadata": artifact.lifecycle_revision.metadata,
        },
    }
    np.savez_compressed(
        target,
        metadata=json.dumps(metadata, sort_keys=True),
        basis=artifact.basis,
        singular_values=artifact.singular_values,
        state_weight=artifact.state_weight,
    )
    return target


def open_artifact(path: str | Path, /, *, profile: ROMProfile) -> ROMArtifact:
    """Open a serialized ROM, rejecting metadata or payload corruption."""
    with np.load(Path(path), allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        basis, singular, weight = (
            stored["basis"],
            stored["singular_values"],
            stored["state_weight"],
        )
    if metadata["profile"] != profile_descriptor(profile):
        raise ValueError("Serialized profile does not match the requested profile.")
    revision_info = metadata["revision"]
    revision = NumericRevision(
        revision_info["content_digest"],
        label=revision_info["label"],
        metadata=revision_info["metadata"],
    )
    artifact = ROMArtifact(
        metadata["corpus_id"],
        metadata["validity_id"],
        profile,
        basis,
        singular,
        weight,
        tuple(metadata["training_case_ids"]),
        revision,
    )
    if artifact.artifact_id != metadata["artifact_id"]:
        raise ValueError("Serialized ROM artifact checksum does not match its payload.")
    return artifact


def evaluate(
    artifact: ROMArtifact,
    case: ROMCaseSpec,
    /,
    *,
    truth_model: TruthModel | Callable[[ROMCaseSpec], TruthSample] | None = None,
    indicator: Callable[[ROMArtifact, ROMCaseSpec, TruthSample], float] | None = None,
    fallback: bool = True,
) -> ROMEvaluation:
    """Solve supported linear ROMs; other profile families require an indicator or truth."""
    if not isinstance(artifact, ROMArtifact) or not isinstance(case, ROMCaseSpec):
        raise TypeError("artifact and case must be ROMArtifact and ROMCaseSpec.")
    sample = None if truth_model is None else truth_model(case)
    if sample is not None and not isinstance(sample, TruthSample):
        raise TypeError("truth_model must return TruthSample values.")
    linear = isinstance(
        artifact.profile,
        (LinearPODProfile, LinearCoerciveRBProfile, ParametricCertifiedProfile),
    )
    if not linear:
        if sample is None:
            raise ValueError(
                "This profile requires a truth model for its indicator/fallback hook."
            )
        value = None if indicator is None else float(indicator(artifact, case, sample))
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError("ROM indicators must be finite and nonnegative.")
        if fallback:
            return _truth_result(artifact, case, sample, indicator=value)
        raise ValueError(
            "This profile has no reduced online solver; enable truth fallback."
        )
    if sample is None or sample.operator is None or sample.rhs is None:
        if fallback and sample is not None:
            return _truth_result(artifact, case, sample)
        raise ValueError(
            "Reduced linear evaluation requires truth operator and rhs; no truth fallback is available."
        )
    if sample.state.size != artifact.basis.shape[0]:
        raise ValueError("Truth operator state dimension does not match ROM basis.")
    reduced = artifact.basis.T @ sample.operator @ artifact.basis
    try:
        coefficients = np.linalg.solve(reduced, artifact.basis.T @ sample.rhs)
    except np.linalg.LinAlgError:
        if fallback:
            return _truth_result(artifact, case, sample)
        raise
    state = artifact.basis @ coefficients
    residual = sample.rhs - sample.operator @ state
    dual_norm = (
        float(np.sqrt(residual @ (sample.dual_norm_inverse @ residual)))
        if sample.dual_norm_inverse is not None
        else float(np.linalg.norm(residual))
    )
    certified = isinstance(
        artifact.profile, (LinearCoerciveRBProfile, ParametricCertifiedProfile)
    )
    bound = (
        dual_norm / sample.stability_lower_bound
        if certified
        and artifact.profile.bound_contract.validity_id == artifact.validity_id
        and sample.dual_norm_inverse is not None
        and sample.stability_lower_bound is not None
        else None
    )
    kind = (
        CertificateKind.ERROR_BOUND
        if bound is not None
        else CertificateKind.INDICATOR
        if dual_norm is not None
        else CertificateKind.NONE
    )
    qoi = (
        sample.qoi
        if sample.qoi is not None
        else (float(sample.qoi_vector @ state) if sample.qoi_vector is not None else None)
    )
    revision = NumericRevision(
        _fingerprint(
            {
                "artifact_id": artifact.artifact_id,
                "case": case.case_id,
                "state": _array_record(state),
            }
        ),
        label="rom-evaluation",
        parent_digest=artifact.lifecycle_revision.content_digest,
    )
    return ROMEvaluation(
        artifact.artifact_id,
        case,
        state,
        dual_norm,
        sample.stability_lower_bound,
        bound,
        kind,
        qoi,
        "rom",
        revision,
    )


def audit_against_truth(evaluation: ROMEvaluation, truth: TruthSample, /) -> ROMAudit:
    """Compare a prediction to independently supplied truth without inventing a certificate."""
    if evaluation.state.shape != truth.state.shape:
        raise ValueError("ROM and truth states must share a shape for audit.")
    denominator = float(np.linalg.norm(truth.state))
    relative = float(
        np.linalg.norm(evaluation.state - truth.state)
        / (denominator if denominator else 1.0)
    )
    qoi_error = (
        None
        if evaluation.qoi is None or truth.qoi is None
        else abs(evaluation.qoi - truth.qoi)
    )
    holds = (
        None
        if evaluation.error_bound is None
        else bool(
            np.linalg.norm(evaluation.state - truth.state)
            <= evaluation.error_bound * (denominator if denominator else 1.0)
        )
    )
    revision = NumericRevision(
        _fingerprint(
            {
                "evaluation": evaluation.lifecycle_revision.content_digest,
                "truth": truth.truth_artifact_id,
                "relative_error": relative,
            }
        ),
        label="rom-audit",
        parent_digest=evaluation.lifecycle_revision.content_digest,
    )
    return ROMAudit(
        evaluation.artifact_id,
        evaluation.case.case_id,
        relative,
        qoi_error,
        evaluation.error_bound,
        holds,
        revision,
    )


def _truth_result(
    artifact: ROMArtifact,
    case: ROMCaseSpec,
    sample: TruthSample,
    *,
    indicator: float | None = None,
) -> ROMEvaluation:
    qoi = (
        sample.qoi
        if sample.qoi is not None
        else (
            float(sample.qoi_vector @ sample.state)
            if sample.qoi_vector is not None
            else None
        )
    )
    revision = NumericRevision(
        _fingerprint(
            {
                "artifact_id": artifact.artifact_id,
                "case": case.case_id,
                "truth": sample.truth_artifact_id,
                "indicator": indicator,
            }
        ),
        label="rom-truth-fallback",
        parent_digest=artifact.lifecycle_revision.content_digest,
    )
    return ROMEvaluation(
        artifact.artifact_id,
        case,
        sample.state,
        indicator,
        None,
        None,
        CertificateKind.INDICATOR if indicator is not None else CertificateKind.NONE,
        qoi,
        "truth-fallback",
        revision,
    )


def _basis_size(profile: ROMProfile, singular: NDArray[np.floating]) -> int:
    if isinstance(profile, LinearPODProfile):
        return profile.basis_size
    if isinstance(profile, (LinearCoerciveRBProfile, ParametricCertifiedProfile)):
        return profile.maximum_basis_size
    for name in ("basis_size", "collateral_basis_size", "residual_basis_size"):
        if hasattr(profile, name):
            return int(getattr(profile, name))
    return min(singular.size, 1)


open = open_artifact

__all__ = [
    "ROMArtifact",
    "ROMAudit",
    "ROMEvaluation",
    "audit_against_truth",
    "evaluate",
    "open",
    "open_artifact",
    "serialize",
    "train_profile",
]
