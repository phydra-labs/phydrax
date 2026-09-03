#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from math import isfinite

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)
from ....nn.operator import OperatorBatch, OperatorCaseProvenance, OperatorTargetBatch
from ....nn.operator.training import (
    fit_operator_normalization,
    operator_dataset_from_cases,
    OperatorDataset,
    OperatorNormalizationPolicy,
)
from ....rom import (
    CorpusSplit,
    create_corpus,
    ROMCaseSpec,
    ROMCorpus,
    TruthSample,
    ValidityRegion,
)
from ....uq import DenseCovariance
from .._case import CardiovascularCaseManifest


class CohortCaseStatus(IntEnum):
    """Fail-closed status of a candidate truth case."""

    COMPLETE = 0
    INCOMPLETE_SOLVE = 1
    INVALID_PHYSICS = 2
    TOPOLOGY_MISMATCH = 3


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    resolved = value.strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty.")
    return resolved


_LINKABLE_IDENTITY = re.compile(
    r"(?:[^@\s]+@[^@\s]+\.[^@\s]+|\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b|"
    r"(?:\+?\d[\d(). -]{6,}\d)|\b\d{3}-\d{2}-\d{4}\b)"
)
_LINKABLE_TOKENS = (
    "patient",
    "subject",
    "person",
    "name",
    "mrn",
    "dob",
    "email",
    "phone",
    "ssn",
)


@dataclass(frozen=True, slots=True)
class DeidentifiedCohortIdentity:
    """Non-linkable grouping identity backed by a deidentification receipt."""

    group_id: str
    deidentification_policy_id: str
    deidentification_receipt_id: str
    identity_id: str

    def __init__(
        self,
        group_id: str,
        deidentification_policy_id: str,
        deidentification_receipt_id: str,
        /,
    ):
        group = _identifier(group_id, "deidentified group_id")
        collapsed = "".join(
            character for character in group.lower() if character.isalnum()
        )
        if _LINKABLE_IDENTITY.search(group) or any(
            token in collapsed for token in _LINKABLE_TOKENS
        ):
            raise ValueError(
                "Cohort group identities must not contain PHI or linkable identity markers."
            )
        policy = _identifier(deidentification_policy_id, "deidentification_policy_id")
        receipt = _identifier(deidentification_receipt_id, "deidentification_receipt_id")
        object.__setattr__(self, "group_id", group)
        object.__setattr__(self, "deidentification_policy_id", policy)
        object.__setattr__(self, "deidentification_receipt_id", receipt)
        object.__setattr__(
            self,
            "identity_id",
            canonical_fingerprint(
                {
                    "kind": "deidentified-cardiovascular-cohort-identity",
                    "group": group,
                    "policy": policy,
                    "receipt": receipt,
                }
            ),
        )


def _parameter_record(
    parameters: Mapping[str, float] | Sequence[tuple[str, float]],
    /,
) -> tuple[tuple[str, float], ...]:
    items = parameters.items() if isinstance(parameters, Mapping) else parameters
    record = tuple(sorted((str(name).strip(), float(value)) for name, value in items))
    names = tuple(name for name, _ in record)
    values = tuple(value for _, value in record)
    if not record or any(not name for name in names):
        raise ValueError("Cardiovascular parameters must be a non-empty named record.")
    if len(set(names)) != len(names):
        raise ValueError("Cardiovascular parameter names must be unique.")
    if not all(isfinite(value) for value in values):
        raise ValueError("Cardiovascular parameters must be finite.")
    return record


@dataclass(frozen=True, slots=True)
class CardiovascularTruthCase:
    """One cohort case backed by an authoritative full-order truth artifact.

    Non-complete cases remain first-class records so their probability mass cannot
    disappear during collation. Only ``COMPLETE`` records may carry training data.
    """

    case_id: str
    subject_identity: DeidentifiedCohortIdentity
    site_id: str
    topology_id: str
    parameters: tuple[tuple[str, float], ...]
    probability_mass: float
    status: CohortCaseStatus
    case_manifest: CardiovascularCaseManifest
    operator_batch: OperatorBatch | None = None
    operator_targets: OperatorTargetBatch | None = None
    truth_sample: TruthSample | None = None
    execution_manifest_id: str | None = None
    ood_tags: tuple[str, ...] = ()
    acquisition_order: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        if not isinstance(self.subject_identity, DeidentifiedCohortIdentity):
            raise TypeError("subject_identity must be a DeidentifiedCohortIdentity.")
        object.__setattr__(self, "site_id", _identifier(self.site_id, "site_id"))
        object.__setattr__(
            self, "topology_id", _identifier(self.topology_id, "topology_id")
        )
        object.__setattr__(self, "parameters", _parameter_record(self.parameters))
        mass = float(self.probability_mass)
        if not isfinite(mass) or mass < 0.0:
            raise ValueError("Cohort probability mass must be finite and nonnegative.")
        object.__setattr__(self, "probability_mass", mass)
        if not isinstance(self.status, CohortCaseStatus):
            raise TypeError("status must be a CohortCaseStatus.")
        tags = tuple(_identifier(tag, "ood tag") for tag in self.ood_tags)
        if not isinstance(self.case_manifest, CardiovascularCaseManifest):
            raise TypeError("case_manifest must be a CardiovascularCaseManifest.")
        if self.case_manifest.case_id != self.case_id:
            raise ValueError(
                "Cohort case ID must match its cardiovascular case manifest."
            )
        if len(set(tags)) != len(tags):
            raise ValueError("OOD tags must be unique within a case.")
        object.__setattr__(self, "ood_tags", tags)
        order = float(self.acquisition_order)
        if not isfinite(order):
            raise ValueError("acquisition_order must be finite.")
        object.__setattr__(self, "acquisition_order", order)
        if self.status is CohortCaseStatus.COMPLETE:
            if mass <= 0.0:
                raise ValueError(
                    "Complete truth cases require positive probability mass."
                )
            if not isinstance(self.operator_batch, OperatorBatch):
                raise TypeError("Complete truth cases require an OperatorBatch.")
            if not isinstance(self.operator_targets, OperatorTargetBatch):
                raise TypeError("Complete truth cases require an OperatorTargetBatch.")
            if not isinstance(self.truth_sample, TruthSample):
                raise TypeError("Complete truth cases require a ROM TruthSample.")
            if self.operator_batch.case_shape != ():
                raise ValueError(
                    "Each cohort case OperatorBatch must represent one case."
                )
            self.operator_targets.validate(self.operator_batch)
            manifest = (
                None
                if self.execution_manifest_id is None
                else _identifier(self.execution_manifest_id, "execution_manifest_id")
            )
            if manifest is None:
                raise ValueError("Complete truth cases require an execution manifest ID.")
            object.__setattr__(self, "execution_manifest_id", manifest)
        elif any(
            value is not None
            for value in (self.operator_batch, self.operator_targets, self.truth_sample)
        ):
            raise ValueError(
                "Non-complete cases cannot contribute learned inputs, targets, or truth."
            )

    @property
    def complete(self) -> bool:
        return self.status is CohortCaseStatus.COMPLETE

    @property
    def provenance(self) -> OperatorCaseProvenance:
        return OperatorCaseProvenance(
            self.case_id,
            identities={
                "subject": self.subject_identity.identity_id,
                "site": self.site_id,
                "topology": self.topology_id,
            },
            order={"acquisition": self.acquisition_order},
        )


@dataclass(frozen=True, slots=True)
class FixedTopologyCohortBatch:
    """Native operator batch plus explicit retained and rejected probability mass."""

    dataset: OperatorDataset
    topology_id: str
    case_ids: tuple[str, ...]
    valid_probability: float
    invalid_probability: float
    invalid_probability_by_status: tuple[tuple[CohortCaseStatus, float], ...]
    cohort_id: str

    @property
    def conditional_probability(self) -> Array:
        return jnp.exp(self.dataset.case_log_weights)


def _case_layout(case: CardiovascularTruthCase, /) -> tuple[object, ...]:
    assert case.operator_batch is not None
    batch = case.operator_batch
    return (
        tuple(batch.inputs),
        tuple((name, batch.input(name).sample_shape) for name in batch.inputs),
        tuple(batch.queries),
        tuple((name, batch.query(name).sample_shape) for name in batch.queries),
    )


def batch_fixed_topology_cohort(
    cases: Sequence[CardiovascularTruthCase],
    /,
    *,
    topology_id: str | None = None,
) -> FixedTopologyCohortBatch:
    """Collate only complete cases while retaining all invalid probability mass."""

    records = tuple(cases)
    if not records or any(
        not isinstance(case, CardiovascularTruthCase) for case in records
    ):
        raise TypeError("cases must contain CardiovascularTruthCase values.")
    case_ids = tuple(case.case_id for case in records)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Cohort case IDs must be unique.")
    total_mass = float(sum(case.probability_mass for case in records))
    if not isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError("A cohort requires positive finite probability mass.")
    complete = tuple(case for case in records if case.complete)
    if not complete:
        raise ValueError("A cohort batch requires at least one complete truth case.")
    resolved_topology = (
        complete[0].topology_id
        if topology_id is None
        else _identifier(topology_id, "topology_id")
    )
    if any(case.topology_id != resolved_topology for case in complete):
        raise ValueError("Complete cases must share the declared fixed topology.")
    first_layout = _case_layout(complete[0])
    if any(_case_layout(case) != first_layout for case in complete[1:]):
        raise ValueError("Fixed-topology cohort cases must have one exact tensor layout.")
    batches = tuple(case.operator_batch for case in complete)
    targets = tuple(case.operator_targets for case in complete)
    assert all(batch is not None for batch in batches)
    assert all(target is not None for target in targets)
    dataset = operator_dataset_from_cases(
        batches,
        targets,
        case_axis="case",
        provenance=tuple(case.provenance for case in complete),
    )
    valid_mass = float(sum(case.probability_mass for case in complete))
    if valid_mass <= 0.0:
        raise ValueError("Complete cohort cases must carry positive probability mass.")
    conditional = jnp.asarray(
        [case.probability_mass / valid_mass for case in complete], dtype=float
    )
    dataset = OperatorDataset(
        dataset.batch,
        dataset.targets,
        dataset.provenance,
        case_log_weights=jnp.log(conditional),
        case_mask=jnp.ones(conditional.shape, dtype=bool),
    )
    status_mass = tuple(
        (
            status,
            float(
                sum(case.probability_mass for case in records if case.status is status)
                / total_mass
            ),
        )
        for status in CohortCaseStatus
        if status is not CohortCaseStatus.COMPLETE
    )
    valid_probability = valid_mass / total_mass
    invalid_probability = (
        sum(case.probability_mass for case in records if not case.complete) / total_mass
    )
    cohort_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-fixed-topology-cohort",
            "topology": resolved_topology,
            "cases": [
                {
                    "case_id": case.case_id,
                    "case_manifest": case.case_manifest.manifest_id,
                    "status": int(case.status),
                    "probability_mass": case.probability_mass,
                    "truth_artifact": (
                        None
                        if case.truth_sample is None
                        else case.truth_sample.truth_artifact_id
                    ),
                    "execution_manifest": case.execution_manifest_id,
                }
                for case in records
            ],
        }
    )
    return FixedTopologyCohortBatch(
        dataset,
        resolved_topology,
        tuple(case.case_id for case in complete),
        valid_probability,
        invalid_probability,
        status_mass,
        cohort_id,
    )


@dataclass(frozen=True, slots=True)
class SubjectSplitPolicy:
    """Split complete cases by whole subject identities."""

    train_fraction: float = 0.7
    calibration_fraction: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        _validate_three_way_fractions(self.train_fraction, self.calibration_fraction)
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True, slots=True)
class SiteSplitPolicy:
    """Reserve declared clinical sites as an external-site OOD partition."""

    held_out_site_ids: tuple[str, ...]
    calibration_fraction: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        sites = tuple(
            _identifier(site, "held-out site ID") for site in self.held_out_site_ids
        )
        if not sites or len(set(sites)) != len(sites):
            raise ValueError("held_out_site_ids must be non-empty and unique.")
        fraction = float(self.calibration_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                "calibration_fraction must lie strictly between zero and one."
            )
        object.__setattr__(self, "held_out_site_ids", sites)
        object.__setattr__(self, "calibration_fraction", fraction)
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True, slots=True)
class OODSplitPolicy:
    """Reserve explicit phenotype/acquisition tags before any in-support split."""

    held_out_tags: tuple[str, ...]
    calibration_fraction: float = 0.15
    interpolation_test_fraction: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        tags = tuple(_identifier(tag, "held-out OOD tag") for tag in self.held_out_tags)
        if not tags or len(set(tags)) != len(tags):
            raise ValueError("held_out_tags must be non-empty and unique.")
        calibration = float(self.calibration_fraction)
        interpolation = float(self.interpolation_test_fraction)
        if (
            calibration <= 0.0
            or interpolation <= 0.0
            or calibration + interpolation >= 1.0
        ):
            raise ValueError(
                "OOD split fractions must leave non-empty train/calibration/test mass."
            )
        object.__setattr__(self, "held_out_tags", tags)
        object.__setattr__(self, "calibration_fraction", calibration)
        object.__setattr__(self, "interpolation_test_fraction", interpolation)
        object.__setattr__(self, "seed", int(self.seed))


CohortSplitPolicy = SubjectSplitPolicy | SiteSplitPolicy | OODSplitPolicy


@dataclass(frozen=True, slots=True)
class CardiovascularCohortSplit:
    """Disjoint subject-safe partitions with a separately named OOD test set."""

    train_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]
    interpolation_test_ids: tuple[str, ...]
    ood_test_ids: tuple[str, ...]
    split_id: str

    def __post_init__(self) -> None:
        partitions = (
            self.train_ids,
            self.calibration_ids,
            self.interpolation_test_ids,
            self.ood_test_ids,
        )
        all_ids = tuple(
            identifier for partition in partitions for identifier in partition
        )
        if not self.train_ids or not self.calibration_ids:
            raise ValueError(
                "Cohort splits require non-empty train and calibration partitions."
            )
        if any(not identifier for identifier in all_ids) or len(set(all_ids)) != len(
            all_ids
        ):
            raise ValueError(
                "Cohort split partitions must contain disjoint non-empty IDs."
            )
        _identifier(self.split_id, "split_id")

    @property
    def all_ids(self) -> tuple[str, ...]:
        return (
            *self.train_ids,
            *self.calibration_ids,
            *self.interpolation_test_ids,
            *self.ood_test_ids,
        )


def _validate_three_way_fractions(train: float, calibration: float, /) -> None:
    train_value = float(train)
    calibration_value = float(calibration)
    if (
        train_value <= 0.0
        or calibration_value <= 0.0
        or train_value + calibration_value >= 1.0
    ):
        raise ValueError(
            "Split fractions must leave non-empty train/calibration/test mass."
        )


def _ordered_subject_groups(
    cases: Sequence[CardiovascularTruthCase],
    seed: int,
    /,
) -> tuple[tuple[CardiovascularTruthCase, ...], ...]:
    groups: dict[str, list[CardiovascularTruthCase]] = {}
    for case in cases:
        groups.setdefault(case.subject_identity.identity_id, []).append(case)
    ordered_ids = sorted(
        groups,
        key=lambda subject: canonical_fingerprint(
            {"kind": "cardiovascular-split-order", "seed": seed, "subject": subject}
        ),
    )
    return tuple(tuple(groups[subject]) for subject in ordered_ids)


def _cut_groups(
    groups: Sequence[Sequence[CardiovascularTruthCase]],
    fractions: Sequence[float],
    /,
) -> tuple[tuple[CardiovascularTruthCase, ...], ...]:
    if len(groups) < len(fractions):
        raise ValueError(
            "A subject-safe split needs one independent subject per partition."
        )
    sizes = np.asarray([len(group) for group in groups], dtype=int)
    cumulative = np.cumsum(sizes)
    total = int(cumulative[-1])
    cuts: list[int] = []
    lower = 1
    cumulative_fraction = 0.0
    for partition_index, fraction in enumerate(fractions[:-1]):
        cumulative_fraction += float(fraction)
        upper = len(groups) - (len(fractions) - partition_index - 1)
        cut = min(
            range(lower, upper + 1),
            key=lambda candidate: abs(
                float(cumulative[candidate - 1]) - total * cumulative_fraction
            ),
        )
        cuts.append(cut)
        lower = cut + 1
    boundaries = (0, *cuts, len(groups))
    return tuple(
        tuple(case for group in groups[left:right] for case in group)
        for left, right in zip(boundaries, boundaries[1:])
    )


def split_cardiovascular_cohort(
    cases: Sequence[CardiovascularTruthCase],
    policy: CohortSplitPolicy,
    /,
) -> CardiovascularCohortSplit:
    """Deterministically split whole subjects, reserving site/tag OOD cases first."""

    complete = tuple(case for case in cases if case.complete)
    if not complete:
        raise ValueError("Cohort splitting requires complete cases.")
    if len({case.case_id for case in complete}) != len(complete):
        raise ValueError("Complete cohort case IDs must be unique.")
    if isinstance(policy, SubjectSplitPolicy):
        groups = _ordered_subject_groups(complete, policy.seed)
        train, calibration, test = _cut_groups(
            groups,
            (
                policy.train_fraction,
                policy.calibration_fraction,
                1.0 - policy.train_fraction - policy.calibration_fraction,
            ),
        )
        ood: tuple[CardiovascularTruthCase, ...] = ()
    elif isinstance(policy, SiteSplitPolicy):
        held_out = set(policy.held_out_site_ids)
        ood = tuple(case for case in complete if case.site_id in held_out)
        retained = tuple(case for case in complete if case.site_id not in held_out)
        if not ood:
            raise ValueError("Site policy selected no complete held-out cases.")
        retained_subjects = {case.subject_identity.identity_id for case in retained}
        if any(case.subject_identity.identity_id in retained_subjects for case in ood):
            raise ValueError(
                "A subject cannot occur in both retained and held-out sites."
            )
        train, calibration = _cut_groups(
            _ordered_subject_groups(retained, policy.seed),
            (1.0 - policy.calibration_fraction, policy.calibration_fraction),
        )
        test = ()
    elif isinstance(policy, OODSplitPolicy):
        tags = set(policy.held_out_tags)
        ood = tuple(case for case in complete if tags.intersection(case.ood_tags))
        retained = tuple(
            case for case in complete if not tags.intersection(case.ood_tags)
        )
        if not ood:
            raise ValueError("OOD policy selected no complete held-out cases.")
        retained_subjects = {case.subject_identity.identity_id for case in retained}
        if any(case.subject_identity.identity_id in retained_subjects for case in ood):
            raise ValueError(
                "A subject cannot occur in both in-support and OOD partitions."
            )
        train_fraction = (
            1.0 - policy.calibration_fraction - policy.interpolation_test_fraction
        )
        train, calibration, test = _cut_groups(
            _ordered_subject_groups(retained, policy.seed),
            (
                train_fraction,
                policy.calibration_fraction,
                policy.interpolation_test_fraction,
            ),
        )
    else:
        raise TypeError("policy must be a subject, site, or OOD split policy.")
    partitions = (train, calibration, test, ood)
    partition_ids = tuple(tuple(case.case_id for case in part) for part in partitions)
    split_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-subject-site-ood-split",
            "policy": type(policy).__name__,
            "partitions": partition_ids,
        }
    )
    return CardiovascularCohortSplit(*partition_ids, split_id)


@dataclass(frozen=True, slots=True)
class TrainOnlyFeaturePreprocessor:
    """Train-fitted parameter centering, covariance, and OOD support boundary."""

    parameter_names: tuple[str, ...]
    location: Array
    scale: Array
    covariance: DenseCovariance
    support_mahalanobis_squared: float
    training_case_ids: tuple[str, ...]
    preprocessing_id: str

    @classmethod
    def fit(
        cls,
        cases: Sequence[CardiovascularTruthCase],
        /,
        *,
        ridge_fraction: float = 1.0e-6,
        support_inflation: float = 1.25,
    ) -> TrainOnlyFeaturePreprocessor:
        records = tuple(cases)
        if len(records) < 2 or any(not case.complete for case in records):
            raise ValueError(
                "Feature preprocessing requires at least two complete train cases."
            )
        names = tuple(name for name, _ in records[0].parameters)
        if any(tuple(name for name, _ in case.parameters) != names for case in records):
            raise ValueError("Every training case must have the same parameter layout.")
        values = jnp.asarray(
            [[value for _, value in case.parameters] for case in records]
        )
        location = jnp.mean(values, axis=0)
        centered = values - location
        empirical = centered.T @ centered / float(len(records) - 1)
        dimension = int(values.shape[1])
        mean_variance = jnp.trace(empirical) / float(dimension)
        ridge = jnp.maximum(mean_variance * float(ridge_fraction), 1.0e-12)
        covariance = DenseCovariance(empirical + ridge * jnp.eye(dimension))
        scale = jnp.sqrt(jnp.diag(covariance.matrix))
        distances = tuple(
            _mahalanobis_squared(value - location, covariance) for value in values
        )
        threshold = max(distances) * float(support_inflation)
        if not isfinite(threshold) or threshold <= 0.0:
            raise ValueError("Training parameters do not define a positive OOD boundary.")
        training_ids = tuple(case.case_id for case in records)
        preprocessing_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-train-only-preprocessing",
                "parameters": names,
                "training_case_ids": training_ids,
                "location": array_tree_fingerprint(location),
                "covariance": array_tree_fingerprint(covariance.matrix),
                "support": threshold,
            }
        )
        return cls(
            names,
            location,
            scale,
            covariance,
            threshold,
            training_ids,
            preprocessing_id,
        )

    def vector(
        self, parameters: Mapping[str, float] | Sequence[tuple[str, float]], /
    ) -> Array:
        record = _parameter_record(parameters)
        if tuple(name for name, _ in record) != self.parameter_names:
            raise ValueError("Parameter names do not match the train-fitted layout.")
        return jnp.asarray([value for _, value in record], dtype=self.location.dtype)

    def transform(
        self, parameters: Mapping[str, float] | Sequence[tuple[str, float]], /
    ) -> Array:
        return (self.vector(parameters) - self.location) / self.scale

    def mahalanobis_squared(
        self, parameters: Mapping[str, float] | Sequence[tuple[str, float]], /
    ) -> float:
        return _mahalanobis_squared(
            self.vector(parameters) - self.location, self.covariance
        )

    def contains(
        self, parameters: Mapping[str, float] | Sequence[tuple[str, float]], /
    ) -> bool:
        return self.mahalanobis_squared(parameters) <= self.support_mahalanobis_squared


def _mahalanobis_squared(delta: ArrayLike, covariance: DenseCovariance, /) -> float:
    vector = jnp.asarray(delta, dtype=covariance.matrix.dtype)
    result = solve(
        LinearSystem(DenseLinearOperator(covariance.matrix)),
        vector,
        policy=LinearSolvePolicy(DenseLU()),
    )
    if not bool(result.successful):
        raise RuntimeError("Native covariance solve failed during OOD assessment.")
    value = float(jnp.real(vector @ result.value))
    if not isfinite(value) or value < 0.0:
        raise RuntimeError(
            "Native covariance solve produced an invalid Mahalanobis distance."
        )
    return value


@dataclass(frozen=True, slots=True)
class PreparedLearningCohort:
    """Leakage-safe native operator partitions and train-only preprocessing."""

    cohort: FixedTopologyCohortBatch
    split: CardiovascularCohortSplit
    train: OperatorDataset
    calibration: OperatorDataset
    interpolation_test: OperatorDataset | None
    ood_test: OperatorDataset | None
    normalization: OperatorNormalizationPolicy
    features: TrainOnlyFeaturePreprocessor
    preparation_id: str


def _take_partition(
    dataset: OperatorDataset,
    index_by_id: Mapping[str, int],
    case_ids: Sequence[str],
    /,
) -> OperatorDataset | None:
    if not case_ids:
        return None
    return dataset.take(tuple(index_by_id[case_id] for case_id in case_ids))


def _require_subject_disjoint_split(
    case_by_id: Mapping[str, CardiovascularTruthCase],
    split: CardiovascularCohortSplit,
    /,
) -> None:
    owner_by_subject: dict[str, int] = {}
    partitions = (
        split.train_ids,
        split.calibration_ids,
        split.interpolation_test_ids,
        split.ood_test_ids,
    )
    for partition_index, case_ids in enumerate(partitions):
        for case_id in case_ids:
            subject = case_by_id[case_id].subject_identity.identity_id
            previous = owner_by_subject.setdefault(subject, partition_index)
            if previous != partition_index:
                raise ValueError(
                    "Manual cohort split leaks one deidentified subject across partitions."
                )


def prepare_learning_cohort(
    cases: Sequence[CardiovascularTruthCase],
    split: CardiovascularCohortSplit,
    /,
    *,
    topology_id: str | None = None,
    normalize_coordinates: bool = True,
) -> PreparedLearningCohort:
    """Prepare fixed-shape partitions; fit every statistic on training cases only."""

    records = tuple(cases)
    cohort = batch_fixed_topology_cohort(records, topology_id=topology_id)
    if set(split.all_ids) != set(cohort.case_ids):
        raise ValueError("The split must exactly cover all complete cohort cases.")
    case_by_id = {case.case_id: case for case in records if case.complete}
    _require_subject_disjoint_split(case_by_id, split)
    index_by_id = {case_id: index for index, case_id in enumerate(cohort.case_ids)}
    train = _take_partition(cohort.dataset, index_by_id, split.train_ids)
    calibration = _take_partition(cohort.dataset, index_by_id, split.calibration_ids)
    assert train is not None and calibration is not None
    interpolation_test = _take_partition(
        cohort.dataset, index_by_id, split.interpolation_test_ids
    )
    ood_test = _take_partition(cohort.dataset, index_by_id, split.ood_test_ids)
    normalization = fit_operator_normalization(
        train.batch,
        train.targets,
        normalize_coordinates=normalize_coordinates,
    )
    features = TrainOnlyFeaturePreprocessor.fit(
        tuple(case_by_id[case_id] for case_id in split.train_ids)
    )
    preparation_id = canonical_fingerprint(
        {
            "kind": "cardiovascular-learning-preparation",
            "cohort": cohort.cohort_id,
            "split": split.split_id,
            "train_only_features": features.preprocessing_id,
            "normalization_fit_cases": split.train_ids,
        }
    )
    return PreparedLearningCohort(
        cohort,
        split,
        train,
        calibration,
        interpolation_test,
        ood_test,
        normalization,
        features,
        preparation_id,
    )


def adapt_complete_truth_to_rom(
    cases: Sequence[CardiovascularTruthCase],
    split: CardiovascularCohortSplit,
    /,
    *,
    truth_model_id: str,
    truth_model_revision: str,
    validity: ValidityRegion | None = None,
) -> ROMCorpus:
    """Adapt already-computed authoritative truths to the existing ROM corpus API."""

    complete = tuple(case for case in cases if case.complete)
    by_id = {case.case_id: case for case in complete}
    if set(split.all_ids) != set(by_id):
        raise ValueError("ROM adaptation requires an exhaustive complete-case split.")
    rom_split = CorpusSplit(
        split.train_ids,
        split.calibration_ids,
        (*split.interpolation_test_ids, *split.ood_test_ids),
    )
    specs = tuple(
        ROMCaseSpec(case.case_id, case.parameters, case.topology_id) for case in complete
    )

    def retained_truth(spec: ROMCaseSpec, /) -> TruthSample:
        sample = by_id[spec.case_id].truth_sample
        if sample is None:
            raise RuntimeError(
                "A complete cohort case lost its authoritative truth sample."
            )
        return sample

    return create_corpus(
        specs,
        retained_truth,
        truth_model_id=truth_model_id,
        truth_model_revision=truth_model_revision,
        split=rom_split,
        validity=validity,
    )


__all__ = [
    "CardiovascularCohortSplit",
    "CardiovascularTruthCase",
    "DeidentifiedCohortIdentity",
    "CohortCaseStatus",
    "CohortSplitPolicy",
    "FixedTopologyCohortBatch",
    "OODSplitPolicy",
    "PreparedLearningCohort",
    "SiteSplitPolicy",
    "SubjectSplitPolicy",
    "TrainOnlyFeaturePreprocessor",
    "adapt_complete_truth_to_rom",
    "batch_fixed_topology_cohort",
    "prepare_learning_cohort",
    "split_cardiovascular_cohort",
]
