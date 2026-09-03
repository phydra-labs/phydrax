#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


DatasetSplit = Literal["train", "validation", "test"]
PartitionLevel = Literal["case", "trajectory", "realization", "time_block"]


@runtime_checkable
class ClosureArtifactRepository(Protocol):
    """Structural subset of lifecycle.ArtifactRepository used by datasets."""

    def begin(
        self,
        artifact_id: str,
        writer_id: str,
        /,
        *,
        attempt_id: str | None = None,
        started_at: int | None = None,
    ) -> Any: ...

    def write_chunk(
        self,
        transaction: Any,
        logical_name: str,
        index: int,
        offset: int,
        payload: bytes | bytearray | memoryview,
        /,
        *,
        encoding: str = "identity",
    ) -> Any: ...

    def commit(
        self,
        transaction: Any,
        chunks: Sequence[Any],
        /,
        *,
        metadata: Mapping[str, str] | Sequence[tuple[str, str]] = (),
        committed_at: int | None = None,
    ) -> Any: ...

    def get_manifest(self, artifact_id: str, /) -> Any: ...

    def read_chunk(
        self,
        manifest: Any,
        chunk: Any,
        /,
        *,
        maximum_plaintext_bytes: int | None = None,
    ) -> bytes: ...


class DatasetChunkLayoutError(ValueError):
    """A dataset's declared sample intervals are incomplete or overlapping."""


class DatasetExtent(StrictModule, NonTrainableState):
    case_id: str = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    time_block_id: str = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    extent_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        case_id: str,
        trajectory_id: str,
        realization_id: str,
        time_block_id: str,
        sample_count: int,
    ):
        identifiers = tuple(
            str(value).strip()
            for value in (case_id, trajectory_id, realization_id, time_block_id)
        )
        count = int(sample_count)
        if any(not value for value in identifiers) or count <= 0:
            raise ValueError("Dataset extent metadata is invalid.")
        self.case_id, self.trajectory_id, self.realization_id, self.time_block_id = (
            identifiers
        )
        self.sample_count = count
        self.extent_id = canonical_fingerprint(
            {
                "kind": "closure-dataset-extent",
                "identity": list(identifiers),
                "sample_count": count,
            }
        )


class ClosureDatasetChunk(StrictModule, NonTrainableState):
    """One content-addressed byte payload covering a half-open sample interval."""

    extent_id: str = eqx.field(static=True)
    logical_name: str = eqx.field(static=True)
    chunk_index: int = eqx.field(static=True)
    sample_start: int = eqx.field(static=True)
    sample_stop: int = eqx.field(static=True)
    byte_offset: int = eqx.field(static=True)
    byte_size: int = eqx.field(static=True)
    sha256: str = eqx.field(static=True)
    encoding: str = eqx.field(static=True)
    chunk_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        extent_id: str,
        logical_name: str,
        chunk_index: int,
        sample_start: int,
        sample_stop: int,
        byte_offset: int,
        byte_size: int,
        sha256: str,
        encoding: str = "identity",
    ):
        extent = str(extent_id).strip()
        logical = str(logical_name).strip()
        digest = str(sha256).strip().lower()
        encoding_ = str(encoding).strip()
        index = int(chunk_index)
        start = int(sample_start)
        stop = int(sample_stop)
        offset = int(byte_offset)
        size = int(byte_size)
        if (
            not extent
            or not logical
            or index < 0
            or start < 0
            or stop <= start
            or offset < 0
            or size < 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or encoding_ not in ("identity", "zlib")
        ):
            raise ValueError("Closure dataset chunk metadata is invalid.")
        self.extent_id = extent
        self.logical_name = logical
        self.chunk_index = index
        self.sample_start = start
        self.sample_stop = stop
        self.byte_offset = offset
        self.byte_size = size
        self.sha256 = digest
        self.encoding = encoding_
        self.chunk_id = canonical_fingerprint(
            {
                "kind": "closure-dataset-chunk",
                "extent": extent,
                "logical_name": logical,
                "chunk_index": index,
                "sample_interval": [start, stop],
                "byte_offset": offset,
                "byte_size": size,
                "sha256": digest,
                "encoding": encoding_,
            }
        )

    @classmethod
    def from_payload(
        cls,
        payload: bytes,
        /,
        *,
        extent_id: str,
        logical_name: str,
        chunk_index: int,
        sample_start: int,
        sample_stop: int,
        byte_offset: int,
        encoding: str = "identity",
    ) -> ClosureDatasetChunk:
        value = bytes(payload)
        return cls(
            extent_id=extent_id,
            logical_name=logical_name,
            chunk_index=chunk_index,
            sample_start=sample_start,
            sample_stop=sample_stop,
            byte_offset=byte_offset,
            byte_size=len(value),
            sha256=hashlib.sha256(value).hexdigest(),
            encoding=encoding,
        )


class ChunkedClosureDatasetManifest(StrictModule, NonTrainableState):
    """Complete chunk coverage for identified trajectories, without path ownership."""

    extents: tuple[DatasetExtent, ...]
    chunks: tuple[ClosureDatasetChunk, ...]
    dataset_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    analysis_dag_id: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        dataset_id: str,
        schema_id: str,
        analysis_dag_id: str,
        extents: tuple[DatasetExtent, ...],
        chunks: tuple[ClosureDatasetChunk, ...],
    ):
        dataset = str(dataset_id).strip()
        schema = str(schema_id).strip()
        dag = str(analysis_dag_id).strip()
        extents_ = tuple(extents)
        chunks_ = tuple(chunks)
        if (
            not dataset
            or not schema
            or not dag
            or not extents_
            or any(not isinstance(value, DatasetExtent) for value in extents_)
            or any(not isinstance(value, ClosureDatasetChunk) for value in chunks_)
            or len({value.extent_id for value in extents_}) != len(extents_)
            or len({value.chunk_id for value in chunks_}) != len(chunks_)
        ):
            raise ValueError("Chunked closure dataset manifest metadata is invalid.")
        extent_ids = {value.extent_id for value in extents_}
        if any(value.extent_id not in extent_ids for value in chunks_):
            raise DatasetChunkLayoutError(
                "A chunk refers to an undeclared dataset extent."
            )
        for extent in extents_:
            intervals = sorted(
                (
                    (chunk.sample_start, chunk.sample_stop, chunk.chunk_index)
                    for chunk in chunks_
                    if chunk.extent_id == extent.extent_id
                ),
                key=lambda value: (value[0], value[1], value[2]),
            )
            if not intervals:
                raise DatasetChunkLayoutError(
                    "Every dataset extent requires chunk coverage."
                )
            cursor = 0
            indices: set[int] = set()
            for start, stop, index in intervals:
                if index in indices:
                    raise DatasetChunkLayoutError(
                        "Chunk indices must be unique within an extent."
                    )
                indices.add(index)
                if start < cursor:
                    raise DatasetChunkLayoutError("Dataset chunks overlap.")
                if start > cursor:
                    raise DatasetChunkLayoutError("Dataset chunks contain a hole.")
                cursor = stop
            if cursor != extent.sample_count:
                raise DatasetChunkLayoutError(
                    "Dataset chunks do not cover the declared extent."
                )
        for logical_name in sorted({chunk.logical_name for chunk in chunks_}):
            logical_chunks = tuple(
                sorted(
                    (chunk for chunk in chunks_ if chunk.logical_name == logical_name),
                    key=lambda chunk: chunk.chunk_index,
                )
            )
            expected_offset = 0
            for expected_index, chunk in enumerate(logical_chunks):
                if chunk.chunk_index != expected_index:
                    raise DatasetChunkLayoutError(
                        f"Chunk indexes for {logical_name!r} contain a hole."
                    )
                if chunk.byte_offset != expected_offset:
                    relation = (
                        "overlap" if chunk.byte_offset < expected_offset else "hole"
                    )
                    raise DatasetChunkLayoutError(
                        f"Chunk byte ranges for {logical_name!r} contain a {relation}."
                    )
                expected_offset += chunk.byte_size
        self.extents = extents_
        self.chunks = chunks_
        self.dataset_id = dataset
        self.schema_id = schema
        self.analysis_dag_id = dag
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "chunked-closure-dataset-manifest",
                "dataset_id": dataset,
                "schema": schema,
                "analysis_dag": dag,
                "extents": [value.extent_id for value in extents_],
                "chunks": [value.chunk_id for value in chunks_],
            }
        )

    def write(
        self,
        repository: ClosureArtifactRepository,
        payloads: tuple[bytes, ...],
        /,
        *,
        writer_id: str,
    ) -> Any:
        values = tuple(bytes(value) for value in payloads)
        if len(values) != len(self.chunks):
            raise ValueError("Exactly one payload is required per declared chunk.")
        for chunk, payload in zip(self.chunks, values, strict=True):
            if (
                len(payload) != chunk.byte_size
                or hashlib.sha256(payload).hexdigest() != chunk.sha256
            ):
                raise ValueError(
                    "A dataset payload does not match its chunk declaration."
                )
        transaction = repository.begin(self.dataset_id, str(writer_id).strip())
        records = tuple(
            repository.write_chunk(
                transaction,
                chunk.logical_name,
                chunk.chunk_index,
                chunk.byte_offset,
                payload,
                encoding=chunk.encoding,
            )
            for chunk, payload in zip(self.chunks, values, strict=True)
        )
        return repository.commit(
            transaction,
            records,
            metadata=(
                ("closure_dataset_manifest_id", self.manifest_id),
                ("flow_state_schema_id", self.schema_id),
                ("analysis_dag_id", self.analysis_dag_id),
            ),
        )

    def read(
        self,
        repository: ClosureArtifactRepository,
        repository_chunks: tuple[Any, ...],
        /,
        *,
        maximum_plaintext_bytes: int | None = None,
    ) -> tuple[bytes, ...]:
        records = tuple(repository_chunks)
        if len(records) != len(self.chunks):
            raise ValueError("Repository chunks must match the dataset chunk count.")
        artifact_manifest = repository.get_manifest(self.dataset_id)
        payloads = tuple(
            repository.read_chunk(
                artifact_manifest,
                record,
                maximum_plaintext_bytes=maximum_plaintext_bytes,
            )
            for record in records
        )
        for chunk, payload in zip(self.chunks, payloads, strict=True):
            if (
                len(payload) != chunk.byte_size
                or hashlib.sha256(payload).hexdigest() != chunk.sha256
            ):
                raise ValueError(
                    "Repository payload does not match the dataset manifest."
                )
        return payloads


class ClosureSampleKey(StrictModule, NonTrainableState):
    case_id: str = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    time_block_id: str = eqx.field(static=True)
    time_index: int = eqx.field(static=True)
    sample_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        case_id: str,
        trajectory_id: str,
        realization_id: str,
        time_block_id: str,
        time_index: int,
    ):
        identifiers = tuple(
            str(value).strip()
            for value in (case_id, trajectory_id, realization_id, time_block_id)
        )
        index = int(time_index)
        if any(not value for value in identifiers) or index < 0:
            raise ValueError("Closure sample key is invalid.")
        self.case_id, self.trajectory_id, self.realization_id, self.time_block_id = (
            identifiers
        )
        self.time_index = index
        self.sample_id = canonical_fingerprint(
            {
                "kind": "closure-sample-key",
                "identity": list(identifiers),
                "time_index": index,
            }
        )

    def group_key(self, level: PartitionLevel, /) -> tuple[str, ...]:
        if level == "case":
            return (self.case_id,)
        if level == "trajectory":
            return (self.case_id, self.trajectory_id)
        if level == "realization":
            return (self.case_id, self.trajectory_id, self.realization_id)
        if level == "time_block":
            return (
                self.case_id,
                self.trajectory_id,
                self.realization_id,
                self.time_block_id,
            )
        raise ValueError("Unknown leakage partition level.")


class ClosureSample(StrictModule, NonTrainableState):
    values: Array
    key: ClosureSampleKey
    schema_id: str = eqx.field(static=True)
    sample_id: str = eqx.field(static=True)

    def __init__(self, values: ArrayLike, key: ClosureSampleKey, /, *, schema_id: str):
        if not isinstance(key, ClosureSampleKey):
            raise TypeError("key must be a ClosureSampleKey.")
        array = jnp.asarray(values)
        schema = str(schema_id).strip()
        if array.ndim < 1 or not schema or not jnp.issubdtype(array.dtype, jnp.inexact):
            raise ValueError("Closure sample values or schema are invalid.")
        self.values = array
        self.key = key
        self.schema_id = schema
        self.sample_id = canonical_fingerprint(
            {
                "kind": "closure-sample",
                "key": key.sample_id,
                "schema": schema,
                "content": array_tree_fingerprint(array),
            }
        )


class PartitionAssignment(StrictModule, NonTrainableState):
    sample_id: str = eqx.field(static=True)
    group_key: tuple[str, ...] = eqx.field(static=True)
    split: DatasetSplit = eqx.field(static=True)
    assignment_id: str = eqx.field(static=True)

    def __init__(
        self, *, sample_id: str, group_key: tuple[str, ...], split: DatasetSplit
    ):
        sample = str(sample_id).strip()
        group = tuple(str(value).strip() for value in group_key)
        split_ = str(split).strip()
        if (
            not sample
            or not group
            or any(not value for value in group)
            or split_ not in ("train", "validation", "test")
        ):
            raise ValueError("Partition assignment is invalid.")
        self.sample_id = sample
        self.group_key = group
        self.split = split_
        self.assignment_id = canonical_fingerprint(
            {
                "kind": "closure-partition-assignment",
                "sample": sample,
                "group": list(group),
                "split": split_,
            }
        )


class LeakageSafePartitionPlan(StrictModule, NonTrainableState):
    level: PartitionLevel = eqx.field(static=True)
    train_fraction: float = eqx.field(static=True)
    validation_fraction: float = eqx.field(static=True)
    test_fraction: float = eqx.field(static=True)
    salt: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        level: PartitionLevel,
        /,
        *,
        train_fraction: float,
        validation_fraction: float,
        test_fraction: float,
        salt: str,
    ):
        level_ = str(level).strip()
        fractions = tuple(
            float(value) for value in (train_fraction, validation_fraction, test_fraction)
        )
        salt_ = str(salt).strip()
        if (
            level_ not in ("case", "trajectory", "realization", "time_block")
            or any(not np.isfinite(value) or value < 0.0 for value in fractions)
            or not np.isclose(sum(fractions), 1.0, rtol=0.0, atol=1e-12)
            or fractions[0] <= 0.0
            or not salt_
        ):
            raise ValueError("Leakage-safe partition plan is invalid.")
        self.level = level_
        self.train_fraction, self.validation_fraction, self.test_fraction = fractions
        self.salt = salt_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "leakage-safe-partition-plan",
                "level": level_,
                "fractions": list(fractions),
                "salt": salt_,
            }
        )

    def assign(self, samples: tuple[ClosureSample, ...], /) -> LeakageSafePartition:
        values = tuple(samples)
        if not values or any(not isinstance(value, ClosureSample) for value in values):
            raise ValueError("Partitioning requires at least one closure sample.")
        if len({value.sample_id for value in values}) != len(values):
            raise ValueError("Partition samples must have unique identities.")
        assignments = []
        for sample in values:
            group = sample.key.group_key(self.level)
            digest = canonical_fingerprint(
                {"kind": "partition-group", "salt": self.salt, "group": list(group)}
            )
            coordinate = int(digest[:16], 16) / float(16**16)
            if coordinate < self.train_fraction:
                split: DatasetSplit = "train"
            elif coordinate < self.train_fraction + self.validation_fraction:
                split = "validation"
            else:
                split = "test"
            assignments.append(
                PartitionAssignment(
                    sample_id=sample.sample_id,
                    group_key=group,
                    split=split,
                )
            )
        return LeakageSafePartition(self, tuple(assignments))


class LeakageSafePartition(StrictModule, NonTrainableState):
    plan: LeakageSafePartitionPlan
    assignments: tuple[PartitionAssignment, ...]
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: LeakageSafePartitionPlan,
        assignments: tuple[PartitionAssignment, ...],
        /,
    ):
        if not isinstance(plan, LeakageSafePartitionPlan):
            raise TypeError("plan must be a LeakageSafePartitionPlan.")
        values = tuple(assignments)
        if not values or any(
            not isinstance(value, PartitionAssignment) for value in values
        ):
            raise ValueError("Leakage-safe partitions require assignments.")
        group_splits: dict[tuple[str, ...], DatasetSplit] = {}
        for assignment in values:
            prior = group_splits.get(assignment.group_key)
            if prior is not None and prior != assignment.split:
                raise ValueError("A leakage group cannot cross dataset splits.")
            group_splits[assignment.group_key] = assignment.split
        if len({value.sample_id for value in values}) != len(values):
            raise ValueError("A sample cannot have multiple partition assignments.")
        self.plan = plan
        self.assignments = values
        self.partition_id = canonical_fingerprint(
            {
                "kind": "leakage-safe-partition",
                "plan": plan.plan_id,
                "assignments": [value.assignment_id for value in values],
            }
        )

    def assignment_for(self, sample_id: str, /) -> PartitionAssignment:
        identifier = str(sample_id).strip()
        matches = tuple(
            value for value in self.assignments if value.sample_id == identifier
        )
        if len(matches) != 1:
            raise KeyError("Sample has no unique partition assignment.")
        return matches[0]

    def sample_ids(self, split: DatasetSplit, /) -> tuple[str, ...]:
        split_ = str(split).strip()
        if split_ not in ("train", "validation", "test"):
            raise ValueError("Unknown dataset split.")
        return tuple(
            value.sample_id for value in self.assignments if value.split == split_
        )


class NormalizerProvenance(StrictModule, NonTrainableState):
    partition_id: str = eqx.field(static=True)
    training_assignment_ids: tuple[str, ...] = eqx.field(static=True)
    training_sample_ids: tuple[str, ...] = eqx.field(static=True)
    feature_name: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        partition_id: str,
        training_assignment_ids: tuple[str, ...],
        training_sample_ids: tuple[str, ...],
        feature_name: str,
        schema_id: str,
    ):
        partition = str(partition_id).strip()
        assignments = tuple(str(value).strip() for value in training_assignment_ids)
        samples = tuple(str(value).strip() for value in training_sample_ids)
        feature = str(feature_name).strip()
        schema = str(schema_id).strip()
        if (
            not partition
            or not assignments
            or len(assignments) != len(samples)
            or any(not value for value in (*assignments, *samples))
            or not feature
            or not schema
        ):
            raise ValueError("Normalizer provenance is invalid.")
        self.partition_id = partition
        self.training_assignment_ids = assignments
        self.training_sample_ids = samples
        self.feature_name = feature
        self.schema_id = schema
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "train-only-normalizer-provenance",
                "partition": partition,
                "training_assignments": list(assignments),
                "training_samples": list(samples),
                "feature_name": feature,
                "schema": schema,
            }
        )


class TrainOnlyNormalizer(StrictModule, NonTrainableState):
    mean: Array
    scale: Array
    provenance: NormalizerProvenance
    epsilon: float = eqx.field(static=True)
    normalizer_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        scale: ArrayLike,
        provenance: NormalizerProvenance,
        /,
        *,
        epsilon: float,
    ):
        if not isinstance(provenance, NormalizerProvenance):
            raise TypeError("provenance must be NormalizerProvenance.")
        mean_ = jnp.asarray(mean)
        scale_ = jnp.asarray(scale)
        epsilon_ = float(epsilon)
        if (
            mean_.shape != scale_.shape
            or not np.isfinite(epsilon_)
            or epsilon_ <= 0.0
            or np.any(~np.isfinite(np.asarray(mean_)))
            or np.any(~np.isfinite(np.asarray(scale_)))
            or np.any(np.asarray(scale_) < epsilon_)
        ):
            raise ValueError("Normalizer statistics are invalid.")
        self.mean = mean_
        self.scale = scale_
        self.provenance = provenance
        self.epsilon = epsilon_
        self.normalizer_id = canonical_fingerprint(
            {
                "kind": "train-only-normalizer",
                "provenance": provenance.provenance_id,
                "epsilon": epsilon_,
                "mean": array_tree_fingerprint(mean_),
                "scale": array_tree_fingerprint(scale_),
            }
        )

    @classmethod
    def fit(
        cls,
        samples: tuple[ClosureSample, ...],
        partition: LeakageSafePartition,
        /,
        *,
        feature_name: str,
        epsilon: float = 1e-8,
    ) -> TrainOnlyNormalizer:
        if not isinstance(partition, LeakageSafePartition):
            raise TypeError("partition must be a LeakageSafePartition.")
        sample_values = tuple(samples)
        if not sample_values or any(
            not isinstance(value, ClosureSample) for value in sample_values
        ):
            raise ValueError("Normalizer fitting requires closure samples.")
        schema_ids = {value.schema_id for value in sample_values}
        if len(schema_ids) != 1:
            raise ValueError("Normalizer fitting requires one flow-state schema.")
        training = tuple(
            value
            for value in sample_values
            if partition.assignment_for(value.sample_id).split == "train"
        )
        if not training:
            raise ValueError("Normalizer fitting requires at least one training sample.")
        shapes = tuple(value.values.shape for value in training)
        if any(shape != shapes[0] for shape in shapes):
            raise ValueError("Training samples must share shape.")
        stacked = jnp.stack(tuple(value.values for value in training), axis=0)
        axes = tuple(range(stacked.ndim - 1))
        mean = jnp.mean(stacked, axis=axes)
        variance = jnp.mean(jnp.abs(stacked - mean) ** 2, axis=axes)
        scale = jnp.maximum(jnp.sqrt(variance), float(epsilon))
        assignments = tuple(
            partition.assignment_for(value.sample_id) for value in training
        )
        provenance = NormalizerProvenance(
            partition_id=partition.partition_id,
            training_assignment_ids=tuple(value.assignment_id for value in assignments),
            training_sample_ids=tuple(value.sample_id for value in training),
            feature_name=feature_name,
            schema_id=training[0].schema_id,
        )
        return cls(mean, scale, provenance, epsilon=epsilon)

    def normalize(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[-self.mean.ndim :] != self.mean.shape:
            raise ValueError("Normalizer feature shape does not match values.")
        return (array - self.mean) / self.scale

    def denormalize(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.shape[-self.mean.ndim :] != self.mean.shape:
            raise ValueError("Normalizer feature shape does not match values.")
        return array * self.scale + self.mean


__all__ = [
    "ChunkedClosureDatasetManifest",
    "ClosureArtifactRepository",
    "ClosureDatasetChunk",
    "ClosureSample",
    "ClosureSampleKey",
    "DatasetChunkLayoutError",
    "DatasetExtent",
    "DatasetSplit",
    "LeakageSafePartition",
    "LeakageSafePartitionPlan",
    "NormalizerProvenance",
    "PartitionAssignment",
    "PartitionLevel",
    "TrainOnlyNormalizer",
]
