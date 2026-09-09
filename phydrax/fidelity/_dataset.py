#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..data_utils import grouped_train_validation_test_split_indices
from ..discretization._core import nonempty_identifier, resolved_identifier
from ._execution import FidelityEvaluation
from ._hierarchy import FidelityHierarchy


def _metadata_items(metadata: Mapping[str, str], /) -> tuple[tuple[str, str], ...]:
    items = tuple(sorted((str(key), str(value)) for key, value in metadata.items()))
    if any(not key or not value for key, value in items):
        raise ValueError("Fidelity metadata keys and values must be non-empty strings.")
    return items


class FidelityCaseSpec(StrictModule, NonTrainableState):
    """One physical case shared by every available fidelity observation."""

    inputs: PyTree[Array]
    log_weight: Array
    case_id: str = eqx.field(static=True)
    split_group_id: str = eqx.field(static=True)
    input_id: str = eqx.field(static=True)
    metadata: tuple[tuple[str, str], ...] = eqx.field(static=True)
    case_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        inputs: PyTree[Array],
        /,
        *,
        case_id: str,
        split_group_id: str | None = None,
        input_id: str | None = None,
        log_weight: float | Array = 0.0,
        metadata: Mapping[str, str] | None = None,
    ):
        leaves = tuple(jax.tree_util.tree_leaves(inputs))
        if not leaves:
            raise ValueError("inputs must contain at least one array leaf.")
        for leaf in leaves:
            value = np.asarray(leaf)
            if value.dtype.hasobject:
                raise TypeError("Fidelity case inputs cannot contain object arrays.")
            if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                raise ValueError("Fidelity case inputs must be finite.")
        weight = jnp.asarray(log_weight)
        if weight.shape != () or not np.isfinite(float(np.asarray(weight))):
            raise ValueError("log_weight must be one finite scalar.")
        case = nonempty_identifier("case_id", case_id)
        group = (
            case
            if split_group_id is None
            else nonempty_identifier("split_group_id", split_group_id)
        )
        input_fingerprint = array_tree_fingerprint(inputs)
        resolved_input = resolved_identifier(
            "input_id",
            input_id,
            {"kind": "fidelity-case-input", "payload": input_fingerprint},
        )
        metadata_ = _metadata_items({} if metadata is None else metadata)
        self.inputs = inputs
        self.log_weight = weight
        self.case_id = case
        self.split_group_id = group
        self.input_id = resolved_input
        self.metadata = metadata_
        self.case_fingerprint = resolved_identifier(
            "case_fingerprint",
            None,
            {
                "kind": "fidelity-case",
                "case": case,
                "split_group": group,
                "input": resolved_input,
                "input_payload": input_fingerprint,
                "log_weight": float(np.asarray(weight)),
                "metadata": list(metadata_),
            },
        )


class FidelityDataset(StrictModule, NonTrainableState):
    """Sparse heterogeneous observations over one fidelity hierarchy."""

    hierarchy: FidelityHierarchy
    cases: tuple[FidelityCaseSpec, ...]
    evaluations: tuple[FidelityEvaluation, ...]
    dataset_id: str = eqx.field(static=True)
    cost_unit: str = eqx.field(static=True)

    def __init__(
        self,
        hierarchy: FidelityHierarchy,
        cases: Sequence[FidelityCaseSpec],
        evaluations: Sequence[FidelityEvaluation],
        /,
        *,
        dataset_id: str | None = None,
    ):
        if not isinstance(hierarchy, FidelityHierarchy):
            raise TypeError("hierarchy must be a FidelityHierarchy.")
        cases_ = tuple(cases)
        evaluations_ = tuple(evaluations)
        if not cases_ or not all(isinstance(case, FidelityCaseSpec) for case in cases_):
            raise TypeError("cases must contain one or more FidelityCaseSpec values.")
        if not evaluations_ or not all(
            isinstance(evaluation, FidelityEvaluation) for evaluation in evaluations_
        ):
            raise TypeError(
                "evaluations must contain one or more FidelityEvaluation values."
            )
        case_ids = tuple(case.case_id for case in cases_)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Fidelity case IDs must be unique.")
        known_cases = set(case_ids)
        known_levels = {level.level_id for level in hierarchy.levels}
        unknown_cases = {
            evaluation.case_id
            for evaluation in evaluations_
            if evaluation.case_id not in known_cases
        }
        unknown_levels = {
            evaluation.level_id
            for evaluation in evaluations_
            if evaluation.level_id not in known_levels
        }
        if unknown_cases:
            raise ValueError(
                f"Fidelity evaluations reference unknown cases {tuple(sorted(unknown_cases))}."
            )
        if unknown_levels:
            raise ValueError(
                f"Fidelity evaluations reference unknown levels {tuple(sorted(unknown_levels))}."
            )
        row_keys = tuple(
            (evaluation.case_id, evaluation.pair_id, evaluation.level_id)
            for evaluation in evaluations_
        )
        if len(set(row_keys)) != len(row_keys):
            raise ValueError("Fidelity case, pair, and level rows must be unique.")
        cost_units = {evaluation.cost_unit for evaluation in evaluations_}
        if len(cost_units) != 1:
            raise ValueError("Every fidelity evaluation must use one common cost unit.")
        reference_structure = jax.tree_util.tree_structure(evaluations_[0].observable)
        reference_leaves = tuple(jax.tree_util.tree_leaves(evaluations_[0].observable))
        reference_signature = tuple(
            (np.asarray(leaf).shape, np.asarray(leaf).dtype.str)
            for leaf in reference_leaves
        )
        for evaluation in evaluations_[1:]:
            if jax.tree_util.tree_structure(evaluation.observable) != reference_structure:
                raise ValueError(
                    "Every fidelity observable must share one canonical PyTree structure."
                )
            signature = tuple(
                (np.asarray(leaf).shape, np.asarray(leaf).dtype.str)
                for leaf in jax.tree_util.tree_leaves(evaluation.observable)
            )
            if signature != reference_signature:
                raise ValueError(
                    "Every fidelity observable must share one canonical shape and dtype."
                )
        self.hierarchy = hierarchy
        self.cases = cases_
        self.evaluations = evaluations_
        self.cost_unit = next(iter(cost_units))
        self.dataset_id = resolved_identifier(
            "dataset_id",
            dataset_id,
            {
                "kind": "fidelity-dataset",
                "hierarchy": hierarchy.fingerprint,
                "cases": [case.case_fingerprint for case in cases_],
                "evaluations": [evaluation.evaluation_id for evaluation in evaluations_],
                "cost_unit": self.cost_unit,
            },
        )

    @property
    def size(self) -> int:
        return len(self.cases)

    @property
    def num_evaluations(self) -> int:
        return len(self.evaluations)

    def case(self, case_id: str, /) -> FidelityCaseSpec:
        identifier = str(case_id)
        for case in self.cases:
            if case.case_id == identifier:
                return case
        raise KeyError(f"Unknown fidelity case {identifier!r}.")

    def evaluations_at(
        self,
        level_id: str,
        /,
        *,
        active_only: bool = False,
    ) -> tuple[FidelityEvaluation, ...]:
        level = self.hierarchy.level(level_id).level_id
        return tuple(
            evaluation
            for evaluation in self.evaluations
            if evaluation.level_id == level
            and (not active_only or bool(np.asarray(evaluation.valid)))
        )

    def paired(
        self,
        source_level_id: str,
        target_level_id: str,
        /,
        *,
        active_only: bool = True,
    ) -> tuple[tuple[FidelityEvaluation, FidelityEvaluation], ...]:
        source = self.hierarchy.level(source_level_id).level_id
        target = self.hierarchy.level(target_level_id).level_id
        source_rows = {
            (evaluation.case_id, evaluation.pair_id): evaluation
            for evaluation in self.evaluations
            if evaluation.level_id == source
            and (not active_only or bool(np.asarray(evaluation.valid)))
        }
        target_rows = {
            (evaluation.case_id, evaluation.pair_id): evaluation
            for evaluation in self.evaluations
            if evaluation.level_id == target
            and (not active_only or bool(np.asarray(evaluation.valid)))
        }
        return tuple(
            (source_rows[key], target_rows[key])
            for key in sorted(source_rows.keys() & target_rows.keys())
        )

    def take_cases(self, indices: Sequence[int], /) -> FidelityDataset:
        positions = tuple(int(index) for index in indices)
        if not positions or len(set(positions)) != len(positions):
            raise ValueError("indices must select unique non-empty case positions.")
        if min(positions) < 0 or max(positions) >= self.size:
            raise IndexError("Fidelity case index is out of bounds.")
        cases = tuple(self.cases[index] for index in positions)
        selected = {case.case_id for case in cases}
        evaluations = tuple(
            evaluation
            for evaluation in self.evaluations
            if evaluation.case_id in selected
        )
        if not evaluations:
            raise ValueError("Selected fidelity cases contain no evaluations.")
        return FidelityDataset(self.hierarchy, cases, evaluations)

    def extend(
        self,
        cases: Sequence[FidelityCaseSpec],
        evaluations: Sequence[FidelityEvaluation],
        /,
    ) -> FidelityDataset:
        additions = tuple(cases)
        existing = {case.case_id: case for case in self.cases}
        for case in additions:
            if case.case_id in existing and (
                case.case_fingerprint != existing[case.case_id].case_fingerprint
            ):
                raise ValueError(
                    f"Fidelity case {case.case_id!r} changed under an existing identity."
                )
        merged = tuple(self.cases) + tuple(
            case for case in additions if case.case_id not in existing
        )
        return FidelityDataset(
            self.hierarchy,
            merged,
            (*self.evaluations, *tuple(evaluations)),
        )


class FidelitySplitRequirements(StrictModule, NonTrainableState):
    """Minimum active fidelity observations required in each grouped partition."""

    train: tuple[tuple[str, int], ...] = eqx.field(static=True)
    validation: tuple[tuple[str, int], ...] = eqx.field(static=True)
    test: tuple[tuple[str, int], ...] = eqx.field(static=True)
    active_only: bool = eqx.field(static=True)
    max_attempts: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        train: Mapping[str, int] | None = None,
        validation: Mapping[str, int] | None = None,
        test: Mapping[str, int] | None = None,
        active_only: bool = True,
        max_attempts: int = 256,
    ):
        self.train = _split_minima("train", train)
        self.validation = _split_minima("validation", validation)
        self.test = _split_minima("test", test)
        self.active_only = bool(active_only)
        attempts = int(max_attempts)
        if attempts <= 0:
            raise ValueError("max_attempts must be positive.")
        self.max_attempts = attempts


def _split_minima(
    partition: str,
    values: Mapping[str, int] | None,
    /,
) -> tuple[tuple[str, int], ...]:
    minima = tuple(
        sorted(
            (str(level_id), int(count))
            for level_id, count in ({} if values is None else values).items()
        )
    )
    if any(not level_id for level_id, _ in minima):
        raise ValueError(f"{partition} fidelity level IDs must be non-empty.")
    if any(count < 0 for _, count in minima):
        raise ValueError(f"{partition} fidelity minima must be non-negative.")
    return tuple((level_id, count) for level_id, count in minima if count > 0)


def _split_requirement_deficits(
    dataset: FidelityDataset,
    partitions: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    requirements: FidelitySplitRequirements,
    /,
) -> tuple[tuple[str, str, int], ...]:
    required = (
        ("train", requirements.train),
        ("validation", requirements.validation),
        ("test", requirements.test),
    )
    deficits: list[tuple[str, str, int]] = []
    for (partition_name, minima), indices in zip(required, partitions, strict=True):
        case_ids = {dataset.cases[index].case_id for index in indices}
        counts: dict[str, int] = {}
        for evaluation in dataset.evaluations:
            if evaluation.case_id not in case_ids:
                continue
            if requirements.active_only and not bool(np.asarray(evaluation.valid)):
                continue
            counts[evaluation.level_id] = counts.get(evaluation.level_id, 0) + 1
        for level_id, minimum in minima:
            missing = minimum - counts.get(level_id, 0)
            if missing > 0:
                deficits.append((partition_name, level_id, missing))
    return tuple(deficits)


class FidelityDatasetSplit(StrictModule, NonTrainableState):
    """Leakage-safe train, validation, and test fidelity partitions."""

    train: FidelityDataset
    validation: FidelityDataset
    test: FidelityDataset
    train_indices: tuple[int, ...] = eqx.field(static=True)
    validation_indices: tuple[int, ...] = eqx.field(static=True)
    test_indices: tuple[int, ...] = eqx.field(static=True)
    seed: int = eqx.field(static=True)

    def __init__(
        self,
        train: FidelityDataset,
        validation: FidelityDataset,
        test: FidelityDataset,
        /,
        *,
        train_indices: Sequence[int],
        validation_indices: Sequence[int],
        test_indices: Sequence[int],
        seed: int,
    ):
        self.train = train
        self.validation = validation
        self.test = test
        self.train_indices = tuple(int(index) for index in train_indices)
        self.validation_indices = tuple(int(index) for index in validation_indices)
        self.test_indices = tuple(int(index) for index in test_indices)
        self.seed = int(seed)


def split_fidelity_dataset(
    dataset: FidelityDataset,
    /,
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
    requirements: FidelitySplitRequirements | None = None,
) -> FidelityDatasetSplit:
    """Split complete physical case groups with optional fidelity coverage minima."""

    if not isinstance(dataset, FidelityDataset):
        raise TypeError("dataset must be a FidelityDataset.")
    if requirements is not None and not isinstance(
        requirements, FidelitySplitRequirements
    ):
        raise TypeError("requirements must be FidelitySplitRequirements or None.")
    known_levels = {level.level_id for level in dataset.hierarchy.levels}
    if requirements is not None:
        requested_levels = {
            level_id
            for minima in (
                requirements.train,
                requirements.validation,
                requirements.test,
            )
            for level_id, _ in minima
        }
        unknown = tuple(sorted(requested_levels - known_levels))
        if unknown:
            raise ValueError(
                f"Fidelity split requirements reference unknown levels {unknown}."
            )
    members: dict[str, list[int]] = {}
    for index, case in enumerate(dataset.cases):
        members.setdefault(case.split_group_id, []).append(index)
    components = tuple(
        tuple(indices) for _, indices in sorted(members.items(), key=lambda item: item[0])
    )
    attempts = 1 if requirements is None else requirements.max_attempts
    best_deficits: tuple[tuple[str, str, int], ...] | None = None
    selected: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None = None
    selected_seed = int(seed)
    for attempt in range(attempts):
        candidate_seed = int(seed) + attempt * 104_729
        partitions = grouped_train_validation_test_split_indices(
            components,
            dataset.size,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            seed=candidate_seed,
            shuffle=True,
        )
        deficits = (
            ()
            if requirements is None
            else _split_requirement_deficits(dataset, partitions, requirements)
        )
        if not deficits:
            selected = partitions
            selected_seed = candidate_seed
            break
        if best_deficits is None or sum(item[2] for item in deficits) < sum(
            item[2] for item in best_deficits
        ):
            best_deficits = deficits
    if selected is None:
        raise ValueError(
            "Unable to construct a grouped fidelity split satisfying the requested "
            f"coverage minima after {attempts} deterministic assignments; "
            f"remaining deficits: {best_deficits}."
        )
    train_indices, validation_indices, test_indices = selected
    return FidelityDatasetSplit(
        dataset.take_cases(train_indices),
        dataset.take_cases(validation_indices),
        dataset.take_cases(test_indices),
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        seed=selected_seed,
    )


__all__ = [
    "FidelityCaseSpec",
    "FidelityDataset",
    "FidelityDatasetSplit",
    "FidelitySplitRequirements",
    "split_fidelity_dataset",
]
