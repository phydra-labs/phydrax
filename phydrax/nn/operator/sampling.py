#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from math import prod
from typing import Any, ClassVar, Literal, TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_mapping
from ...graph._operator_topology import take_operator_topology
from .data import (
    FunctionSamples,
    OperatorBatch,
    OperatorCaseProvenance,
    OperatorFieldBatch,
    OperatorTargetBatch,
    slice_operator_batch,
)


if TYPE_CHECKING:
    from .training._dataset import OperatorDataset


SamplingStrategy = Literal[
    "measure_random",
    "uniform_index",
    "stratified_measure",
    "farthest_point",
    "fixed_indices",
]


@dataclass(frozen=True)
class SampleSelection:
    """One reproducible sample selection and its integration correction."""

    indices: tuple[int, ...]
    probabilities: tuple[float, ...]
    importance_weights: tuple[float, ...]
    strategy: SamplingStrategy

    def __post_init__(self):
        count = len(self.indices)
        if count == 0:
            raise ValueError("A sample selection must not be empty.")
        if len(self.probabilities) != count or len(self.importance_weights) != count:
            raise ValueError("Selection indices, probabilities, and weights must align.")
        if any(index < 0 for index in self.indices):
            raise ValueError("Sample indices must be non-negative.")
        if any(not np.isfinite(value) or value <= 0.0 for value in self.probabilities):
            raise ValueError("Selection probabilities must be finite and positive.")
        if any(
            not np.isfinite(value) or value < 0.0 for value in self.importance_weights
        ):
            raise ValueError("Importance weights must be finite and non-negative.")


@dataclass(frozen=True)
class OperatorCaseMetadata:
    """Geometry-only metadata that a lazy source can read cheaply."""

    inputs: Mapping[str, FunctionSamples]
    queries: Mapping[str, FunctionSamples]
    provenance: OperatorCaseProvenance | None = None


@dataclass(frozen=True)
class OperatorCaseReadRequest:
    """Per-branch selections a source should apply before loading field values."""

    input_selections: Mapping[str, SampleSelection]
    query_selections: Mapping[str, SampleSelection]


@dataclass(frozen=True)
class OperatorCase:
    """One unbatched operator case and its supervised query target."""

    batch: OperatorBatch
    targets: OperatorTargetBatch
    provenance: OperatorCaseProvenance | None = None

    def __post_init__(self):
        if self.batch.case_shape:
            raise ValueError("OperatorCase batches must not contain case axes.")
        if not isinstance(self.targets, OperatorTargetBatch):
            raise TypeError("OperatorCase targets must be an OperatorTargetBatch.")
        self.targets.validate(self.batch)
        if self.provenance is not None and not isinstance(
            self.provenance,
            OperatorCaseProvenance,
        ):
            raise TypeError("OperatorCase provenance must be OperatorCaseProvenance.")


class OperatorCaseSource(abc.ABC):
    """Host-side random-access source for cases too large to materialize eagerly."""

    # Built-ins pin their persisted identity so package moves do not invalidate checkpoints.
    fingerprint_type_id: ClassVar[str | None] = None

    @property
    @abc.abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def content_fingerprint(self) -> str:
        """Stable identity of the immutable logical case sequence."""
        raise NotImplementedError

    @property
    def background_read_safe(self) -> bool:
        """Whether case preparation may run on one dedicated background thread."""
        return False

    def configuration(self) -> Mapping[str, Any]:
        """Return cheap source semantics included in loader compatibility."""
        type_id = self.fingerprint_type_id
        if type_id is None:
            type_id = f"{type(self).__module__}.{type(self).__qualname__}"
        return {"type": type_id, "size": self.size}

    @abc.abstractmethod
    def case_metadata(self, index: int, /) -> OperatorCaseMetadata:
        raise NotImplementedError

    @abc.abstractmethod
    def read_case(
        self,
        index: int,
        /,
        *,
        request: OperatorCaseReadRequest | None = None,
    ) -> OperatorCase:
        raise NotImplementedError


def _geometry_only(samples: FunctionSamples, /) -> FunctionSamples:
    return FunctionSamples(
        values=None,
        axes=samples.axes,
        coordinates=samples.coordinates,
        quadrature_weights=samples.quadrature_weights,
        mask=samples.mask,
        topology=samples.topology,
    )


def _selection_arrays(
    samples: FunctionSamples, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples.geometry_case_shape:
        raise ValueError("Sampling metadata must describe one unbatched case.")
    coordinates = np.asarray(samples.coordinates_array(flatten=True))
    weights = np.asarray(samples.quadrature()).reshape((-1,)).astype(float)
    mask = np.asarray(samples.mask_array()).reshape((-1,)).astype(bool)
    if coordinates.shape[0] != weights.size:
        raise ValueError("Sample coordinates and quadrature do not align.")
    valid = mask & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        raise ValueError("Cannot sample a geometry with zero valid measure.")
    return coordinates, weights, valid


def select_function_samples(
    samples: FunctionSamples,
    count: int,
    /,
    *,
    strategy: SamplingStrategy,
    seed: int,
    fixed_indices: Sequence[int] = (),
) -> SampleSelection:
    """Select sample indices and construct measure-preserving weights."""
    requested = int(count)
    if requested <= 0:
        raise ValueError("Sample count must be positive.")
    coordinates, weights, valid = _selection_arrays(samples)
    valid_indices = np.flatnonzero(valid)
    total = float(np.sum(weights[valid_indices]))
    rng = np.random.default_rng(int(seed))

    if strategy == "uniform_index":
        if requested > valid_indices.size:
            raise ValueError("Uniform sampling without replacement exceeds valid points.")
        chosen = rng.choice(valid_indices, size=requested, replace=False)
        inclusion = float(requested) / float(valid_indices.size)
        probabilities = np.full(requested, inclusion, dtype=float)
        corrected = weights[chosen] / inclusion
    elif strategy == "measure_random":
        distribution = weights[valid_indices] / total
        chosen = rng.choice(
            valid_indices,
            size=requested,
            replace=True,
            p=distribution,
        )
        lookup = np.zeros(weights.size, dtype=float)
        lookup[valid_indices] = distribution
        probabilities = lookup[chosen]
        corrected = weights[chosen] / (float(requested) * probabilities)
    elif strategy == "stratified_measure":
        distribution = np.zeros(weights.size, dtype=float)
        distribution[valid_indices] = weights[valid_indices] / total
        cumulative = np.cumsum(distribution)
        positions = (np.arange(requested) + rng.random(requested)) / requested
        chosen = np.searchsorted(cumulative, positions, side="right")
        chosen = np.minimum(chosen, weights.size - 1)
        probabilities = distribution[chosen]
        corrected = np.full(requested, total / float(requested), dtype=float)
    elif strategy == "farthest_point":
        if requested > valid_indices.size:
            raise ValueError("Farthest-point sampling exceeds valid points.")
        chosen_list = [int(rng.choice(valid_indices))]
        minimum_distance = np.full(weights.size, np.inf, dtype=float)
        for _ in range(1, requested):
            latest = coordinates[chosen_list[-1]]
            distance = np.sum((coordinates - latest) ** 2, axis=-1)
            minimum_distance = np.minimum(minimum_distance, distance)
            minimum_distance[~valid] = -np.inf
            minimum_distance[np.asarray(chosen_list, dtype=int)] = -np.inf
            chosen_list.append(int(np.argmax(minimum_distance)))
        chosen = np.asarray(chosen_list, dtype=int)
        probabilities = np.ones(requested, dtype=float)
        selected_mass = float(np.sum(weights[chosen]))
        corrected = weights[chosen] * (total / selected_mass)
    elif strategy == "fixed_indices":
        chosen = np.asarray(tuple(int(value) for value in fixed_indices), dtype=int)
        if chosen.size != requested:
            raise ValueError("fixed_indices length must equal the requested count.")
        if np.unique(chosen).size != chosen.size:
            raise ValueError("fixed_indices must be unique.")
        if np.any(chosen < 0) or np.any(chosen >= weights.size) or np.any(~valid[chosen]):
            raise ValueError("fixed_indices contain an invalid sample.")
        probabilities = np.ones(requested, dtype=float)
        selected_mass = float(np.sum(weights[chosen]))
        corrected = weights[chosen] * (total / selected_mass)
    else:
        raise ValueError(f"Unknown sampling strategy {strategy!r}.")

    return SampleSelection(
        indices=tuple(int(value) for value in chosen),
        probabilities=tuple(float(value) for value in probabilities),
        importance_weights=tuple(float(value) for value in corrected),
        strategy=strategy,
    )


def take_function_samples(
    samples: FunctionSamples,
    selection: SampleSelection,
    /,
) -> FunctionSamples:
    """Apply one point selection, converting tensor grids to point clouds."""
    if samples.geometry_case_shape:
        raise ValueError("take_function_samples expects one unbatched case.")
    indices = jnp.asarray(selection.indices, dtype=jnp.int32)
    coordinates = samples.coordinates_array(flatten=True)
    values: Array | None
    if samples.values is None:
        values = None
    else:
        sample_ndim = len(samples.sample_shape)
        trailing = samples.values.shape[sample_ndim:]
        flattened = samples.values.reshape((prod(samples.sample_shape),) + trailing)
        values = jnp.take(flattened, indices, axis=0)
    return FunctionSamples(
        values=values,
        coordinates=jnp.take(coordinates, indices, axis=0),
        quadrature_weights=jnp.asarray(selection.importance_weights),
        mask=jnp.ones((len(selection.indices),), dtype=bool),
        topology=(
            None
            if samples.topology is None
            else take_operator_topology(samples.topology, indices)
        ),
    )


def take_query_targets(
    targets: Array,
    sample_shape: Sequence[int],
    selection: SampleSelection,
    /,
) -> Array:
    sample = tuple(int(size) for size in sample_shape)
    array = jnp.asarray(targets)
    trailing = array.shape[len(sample) :]
    flattened = array.reshape((prod(sample),) + trailing)
    return jnp.take(flattened, jnp.asarray(selection.indices), axis=0)


class AnchorQuerySamplingPolicy:
    """Deterministic, branch-aware host sampling policy."""

    anchor_counts: tuple[tuple[str, int], ...]
    query_counts: tuple[tuple[str, int], ...]
    strategy: SamplingStrategy
    query_strategy: SamplingStrategy
    seed: int
    fixed_anchor_indices: tuple[tuple[str, tuple[int, ...]], ...]
    fixed_query_indices: tuple[tuple[str, tuple[int, ...]], ...]

    def __init__(
        self,
        *,
        anchor_counts: Mapping[str, int] = {},
        query_counts: Mapping[str, int] = {},
        strategy: SamplingStrategy = "stratified_measure",
        query_strategy: SamplingStrategy | None = None,
        seed: int = 0,
        fixed_anchor_indices: Mapping[str, Sequence[int]] = {},
        fixed_query_indices: Mapping[str, Sequence[int]] = {},
    ):
        counts = tuple((str(name), int(count)) for name, count in anchor_counts.items())
        targets = tuple((str(name), int(count)) for name, count in query_counts.items())
        if any(not name or count <= 0 for name, count in counts + targets):
            raise ValueError("Anchor and query counts must have names and be positive.")
        fixed = tuple(
            (str(name), tuple(int(value) for value in values))
            for name, values in fixed_anchor_indices.items()
        )
        fixed_targets = tuple(
            (str(name), tuple(int(value) for value in values))
            for name, values in fixed_query_indices.items()
        )
        if {name for name, _ in fixed} - {name for name, _ in counts}:
            raise ValueError("Fixed anchor indices refer to an unknown anchor branch.")
        if {name for name, _ in fixed_targets} - {name for name, _ in targets}:
            raise ValueError("Fixed query indices refer to an unknown query branch.")
        self.anchor_counts = counts
        self.query_counts = targets
        self.strategy = strategy
        self.query_strategy = strategy if query_strategy is None else query_strategy
        self.seed = int(seed)
        self.fixed_anchor_indices = fixed
        self.fixed_query_indices = fixed_targets

    def _seed(
        self,
        *,
        split: str,
        epoch: int,
        case_index: int,
        branch: str,
        purpose: str,
    ) -> int:
        payload = (
            f"{self.seed}|{split}|{int(epoch)}|{int(case_index)}|{branch}|{purpose}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def request(
        self,
        metadata: OperatorCaseMetadata,
        /,
        *,
        split: str,
        epoch: int,
        case_index: int,
    ) -> OperatorCaseReadRequest:
        fixed = dict(self.fixed_anchor_indices)
        selections: dict[str, SampleSelection] = {}
        for name, count in self.anchor_counts:
            if name not in metadata.inputs:
                raise KeyError(f"Sampling policy requests unknown input branch {name!r}.")
            selections[name] = select_function_samples(
                metadata.inputs[name],
                count,
                strategy=self.strategy,
                seed=self._seed(
                    split=split,
                    epoch=epoch,
                    case_index=case_index,
                    branch=name,
                    purpose="anchor",
                ),
                fixed_indices=fixed.get(name, ()),
            )
        fixed_queries = dict(self.fixed_query_indices)
        query_selections: dict[str, SampleSelection] = {}
        for name, count in self.query_counts:
            if name not in metadata.queries:
                raise KeyError(f"Sampling policy requests unknown query branch {name!r}.")
            query_selections[name] = select_function_samples(
                metadata.queries[name],
                count,
                strategy=self.query_strategy,
                seed=self._seed(
                    split=split,
                    epoch=epoch,
                    case_index=case_index,
                    branch=name,
                    purpose="target",
                ),
                fixed_indices=fixed_queries.get(name, ()),
            )
        return OperatorCaseReadRequest(selections, query_selections)


class InMemoryOperatorCaseSource(OperatorCaseSource):
    """Compatibility source for an eager ``OperatorDataset``."""

    fingerprint_type_id = "phydrax.operator.case-source:in-memory@1"

    def __init__(self, dataset: OperatorDataset, /):
        self.dataset = dataset

    @property
    def size(self) -> int:
        return self.dataset.size

    @cached_property
    def content_fingerprint(self) -> str:
        from .training._fingerprint import operator_dataset_fingerprint

        return operator_dataset_fingerprint(self.dataset)

    @property
    def background_read_safe(self) -> bool:
        return True

    def _batch(self, index: int) -> OperatorBatch:
        position = int(index)
        if position < 0 or position >= self.size:
            raise IndexError("Operator case index is out of range.")
        return slice_operator_batch(self.dataset.batch, position, axis=0)

    def case_metadata(self, index: int, /) -> OperatorCaseMetadata:
        batch = self._batch(index)
        assert self.dataset.provenance is not None
        return OperatorCaseMetadata(
            inputs={
                name: _geometry_only(samples) for name, samples in batch.inputs.items()
            },
            queries={
                name: _geometry_only(samples) for name, samples in batch.queries.items()
            },
            provenance=self.dataset.provenance[int(index)],
        )

    def read_case(
        self,
        index: int,
        /,
        *,
        request: OperatorCaseReadRequest | None = None,
    ) -> OperatorCase:
        batch = self._batch(index)
        targets = self.dataset.targets.take(int(index), axis=0)
        assert self.dataset.provenance is not None
        provenance = self.dataset.provenance[int(index)]
        if request is None:
            return OperatorCase(batch, targets, provenance)
        inputs = {
            name: (
                take_function_samples(samples, request.input_selections[name])
                if name in request.input_selections
                else samples
            )
            for name, samples in batch.inputs.items()
        }
        selected_fields = dict(targets.fields)
        queries = dict(batch.queries)
        for query_name, selection in request.query_selections.items():
            if query_name not in queries:
                raise KeyError(f"Read request references unknown query {query_name!r}.")
            query = queries[query_name]
            for name, output_field in tuple(selected_fields.items()):
                if output_field.query_name == query_name:
                    selected_fields[name] = OperatorFieldBatch(
                        take_query_targets(
                            output_field.values,
                            query.sample_shape,
                            selection,
                        ),
                        query_name=output_field.query_name,
                        spec=output_field.spec,
                    )
            queries[query_name] = take_function_samples(query, selection)
        return OperatorCase(
            OperatorBatch(inputs=inputs, queries=queries),
            OperatorTargetBatch(selected_fields),
            provenance,
        )


class CallbackOperatorCaseSource(OperatorCaseSource):
    """Lazy source backed by user-provided metadata and selective readers."""

    fingerprint_type_id = "phydrax.operator.case-source:callback@1"

    def __init__(
        self,
        size: int,
        /,
        *,
        metadata_reader: Callable[[int], OperatorCaseMetadata],
        case_reader: Callable[[int, OperatorCaseReadRequest | None], OperatorCase],
        content_fingerprint: str,
        background_read_safe: bool = False,
        configuration: Mapping[str, Any] | None = None,
    ):
        if int(size) <= 0:
            raise ValueError("Callback source size must be positive.")
        fingerprint = str(content_fingerprint).strip()
        if not fingerprint:
            raise ValueError("Callback source content_fingerprint must be non-empty.")
        source_configuration = canonical_mapping(
            {} if configuration is None else configuration
        )
        self._size = int(size)
        self._content_fingerprint = fingerprint
        self._background_read_safe = bool(background_read_safe)
        self._configuration = source_configuration
        self.metadata_reader = metadata_reader
        self.case_reader = case_reader

    @property
    def size(self) -> int:
        return self._size

    @property
    def content_fingerprint(self) -> str:
        return self._content_fingerprint

    @property
    def background_read_safe(self) -> bool:
        return self._background_read_safe

    def configuration(self) -> Mapping[str, Any]:
        return {
            **super().configuration(),
            "parameters": dict(self._configuration),
        }

    def case_metadata(self, index: int, /) -> OperatorCaseMetadata:
        return self.metadata_reader(int(index))

    def read_case(
        self,
        index: int,
        /,
        *,
        request: OperatorCaseReadRequest | None = None,
    ) -> OperatorCase:
        return self.case_reader(int(index), request)


def read_operator_case_batch(
    source: OperatorCaseSource,
    indices: Sequence[int],
    /,
    *,
    sampling: AnchorQuerySamplingPolicy | None = None,
    split: str = "train",
    epoch: int = 0,
    case_axis: str = "case",
) -> OperatorDataset:
    """Read and collate selected cases, sampling before device placement."""
    from .training._dataset import operator_dataset_from_cases

    selected: list[tuple[int, OperatorCase]] = []
    for index in indices:
        position = int(index)
        request = (
            None
            if sampling is None
            else sampling.request(
                source.case_metadata(position),
                split=split,
                epoch=epoch,
                case_index=position,
            )
        )
        selected.append((position, source.read_case(position, request=request)))
    if not selected:
        raise ValueError("At least one case index is required.")
    return operator_dataset_from_cases(
        tuple(case.batch for _, case in selected),
        tuple(case.targets for _, case in selected),
        case_axis=case_axis,
        provenance=tuple(
            (
                OperatorCaseProvenance(f"source:{position}")
                if case.provenance is None
                else case.provenance
            )
            for position, case in selected
        ),
    )


__all__ = [
    "AnchorQuerySamplingPolicy",
    "CallbackOperatorCaseSource",
    "InMemoryOperatorCaseSource",
    "OperatorCase",
    "OperatorCaseMetadata",
    "OperatorCaseReadRequest",
    "OperatorCaseSource",
    "SampleSelection",
    "SamplingStrategy",
    "read_operator_case_batch",
    "select_function_samples",
    "take_function_samples",
    "take_query_targets",
]
