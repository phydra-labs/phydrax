#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..models.core._base import _AbstractOperatorModel
from ..models.core._encoded_operator import AbstractEncodedOperatorModel
from ..models.core._keys import EvalKey, fold_in_eval_key
from ..models.core._operator import FunctionSamples, OperatorBatch, pad_function_samples
from ..models.core._operator_topology import (
    broadcast_operator_topology,
    OperatorTopology,
)


@dataclass(frozen=True)
class OperatorQuerySchema:
    """Static geometry contract for one lazy query source."""

    size: int
    case_shape: tuple[int, ...]
    coordinate_dimension: int
    fingerprint: str

    def __post_init__(self):
        if int(self.size) <= 0:
            raise ValueError("Query source size must be positive.")
        if int(self.coordinate_dimension) <= 0:
            raise ValueError("Query coordinate dimension must be positive.")
        if any(int(size) <= 0 for size in self.case_shape):
            raise ValueError("Query case dimensions must be positive.")


@dataclass(frozen=True)
class OperatorQueryChunk:
    """One fixed-shape query chunk and its unpadded cardinality."""

    samples: FunctionSamples
    start: int
    valid_count: int

    def __post_init__(self):
        if len(self.samples.sample_shape) != 1:
            raise ValueError("Streamed query chunks must be point clouds.")
        if int(self.start) < 0:
            raise ValueError("Query chunk start must be non-negative.")
        if not 0 < int(self.valid_count) <= self.samples.sample_shape[0]:
            raise ValueError("Query chunk valid_count is out of range.")


class OperatorQuerySource(abc.ABC):
    """Lazy query geometry source with deterministic random access."""

    @property
    @abc.abstractmethod
    def schema(self) -> OperatorQuerySchema:
        raise NotImplementedError

    @abc.abstractmethod
    def read_chunk(self, start: int, size: int, /) -> OperatorQueryChunk:
        raise NotImplementedError


class OperatorPredictionSink(abc.ABC):
    """Incremental consumer for query-major prediction chunks."""

    @abc.abstractmethod
    def begin(self, metadata: "OperatorPredictionMetadata", /) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def write(self, start: int, values: np.ndarray, /) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def finish(self) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class OperatorPredictionMetadata:
    """Persistent shape and provenance metadata for a streamed prediction."""

    total_size: int
    case_shape: tuple[int, ...]
    channel_shape: tuple[int, ...]
    dtype: str
    query_axis: int
    query_fingerprint: str

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (
            self.case_shape
            + (int(self.total_size),)
            + tuple(int(size) for size in self.channel_shape)
        )


def _flatten_query_values(
    values: Array,
    sample_shape: tuple[int, ...],
    case_shape: tuple[int, ...],
    /,
) -> Array:
    sample_ndim = len(sample_shape)
    case_ndim = len(case_shape)
    count = prod(sample_shape)

    trailing = values.shape[case_ndim + sample_ndim :]
    return values.reshape(case_shape + (count,) + trailing)


def _take_query_axis(value: Array, indices: Array, case_ndim: int, /) -> Array:
    return jnp.take(jnp.asarray(value), indices, axis=case_ndim)


class ArrayOperatorQuerySource(OperatorQuerySource):
    """Lazy chunk view over an existing ``FunctionSamples`` query."""

    def __init__(
        self,
        query: FunctionSamples,
        /,
        *,
        case_shape: Sequence[int] = (),
        fingerprint: str = "",
    ):
        cases = tuple(int(size) for size in case_shape)
        coordinates = query.coordinates_array(case_shape=cases, flatten=True)
        coordinate_dimension = int(coordinates.shape[-1])
        size = prod(query.sample_shape)
        self.query = query
        self.case_shape = cases
        self.coordinates = coordinates.reshape(
            cases + (size, coordinate_dimension)
        )
        self.quadrature = query.quadrature(case_shape=cases).reshape(cases + (size,))
        self.mask = query.mask_array(case_shape=cases).reshape(cases + (size,))
        self.values = (
            None
            if query.values is None
            else _flatten_query_values(query.values, query.sample_shape, cases)
        )
        self.topology = (
            None
            if query.topology is None
            else broadcast_operator_topology(query.topology, cases)
        )
        self.topology_entities = (
            None
            if self.topology is None
            else self.topology.sample_entities.reshape(cases + (size,))
        )
        resolved_fingerprint = str(fingerprint) or repr(
            (
                query.axis_names,
                query.sample_shape,
                coordinate_dimension,
                cases,
                query.mask is not None,
                query.quadrature_weights is not None,
                (
                    None
                    if self.topology is None
                    else (
                        self.topology.kind,
                        self.topology.site,
                        self.topology.entity,
                    )
                ),
            )
        )
        self._schema = OperatorQuerySchema(
            size=size,
            case_shape=cases,
            coordinate_dimension=coordinate_dimension,
            fingerprint=resolved_fingerprint,
        )

    @property
    def schema(self) -> OperatorQuerySchema:
        return self._schema

    def read_chunk(self, start: int, size: int, /) -> OperatorQueryChunk:
        begin = int(start)
        requested = int(size)
        if requested <= 0:
            raise ValueError("Query chunk size must be positive.")
        if begin < 0 or begin >= self.schema.size:
            raise IndexError("Query chunk start is out of range.")
        valid = min(requested, self.schema.size - begin)
        indices = jnp.arange(begin, begin + valid, dtype=jnp.int32)
        case_ndim = len(self.case_shape)
        values = (
            None
            if self.values is None
            else jax.tree_util.tree_map(
                lambda leaf: _take_query_axis(leaf, indices, case_ndim),
                self.values,
            )
        )
        topology = None
        if self.topology is not None:
            assert self.topology_entities is not None
            topology = OperatorTopology(
                self.topology.graph,
                _take_query_axis(self.topology_entities, indices, case_ndim),
                case_shape=self.case_shape,
                kind=self.topology.kind,
                site=self.topology.site,
                entity=self.topology.entity,
                validate=False,
            )
        samples = FunctionSamples(
            values=values,
            coordinates=_take_query_axis(self.coordinates, indices, case_ndim),
            quadrature_weights=_take_query_axis(
                self.quadrature, indices, case_ndim
            ),
            mask=_take_query_axis(self.mask, indices, case_ndim),
            topology=topology,
        )
        if valid < requested:
            samples = pad_function_samples(
                samples,
                requested,
                case_shape=self.case_shape,
            )
        return OperatorQueryChunk(samples, begin, valid)


class CallbackOperatorQuerySource(OperatorQuerySource):
    """Query source whose reader returns at most one requested point-cloud chunk."""

    def __init__(
        self,
        schema: OperatorQuerySchema,
        reader: Callable[[int, int], FunctionSamples],
        /,
    ):
        self._schema = schema
        self.reader = reader

    @property
    def schema(self) -> OperatorQuerySchema:
        return self._schema

    def read_chunk(self, start: int, size: int, /) -> OperatorQueryChunk:
        begin = int(start)
        requested = int(size)
        if requested <= 0:
            raise ValueError("Query chunk size must be positive.")
        if begin < 0 or begin >= self.schema.size:
            raise IndexError("Query chunk start is out of range.")
        valid = min(requested, self.schema.size - begin)
        samples = self.reader(begin, valid)
        if len(samples.sample_shape) != 1 or samples.sample_shape[0] != valid:
            raise ValueError(
                "Callback query reader must return exactly the requested valid points."
            )
        if samples.geometry_case_shape not in ((), self.schema.case_shape):
            raise ValueError("Callback query geometry has the wrong case shape.")
        if valid < requested:
            samples = pad_function_samples(
                samples,
                requested,
                case_shape=self.schema.case_shape,
            )
        return OperatorQueryChunk(samples, begin, valid)


class ArrayPredictionSink(OperatorPredictionSink):
    """In-memory sink for ordinary workloads and equivalence checks."""

    def __init__(self):
        self.metadata: OperatorPredictionMetadata | None = None
        self.values: np.ndarray | None = None
        self.next_index = 0

    def begin(self, metadata: OperatorPredictionMetadata, /) -> None:
        if self.metadata is not None:
            raise RuntimeError("Prediction sink has already begun.")
        self.metadata = metadata
        self.values = np.empty(metadata.output_shape, dtype=np.dtype(metadata.dtype))
        self.next_index = 0

    def write(self, start: int, values: np.ndarray, /) -> None:
        if self.metadata is None or self.values is None:
            raise RuntimeError("Prediction sink has not begun.")
        if int(start) != self.next_index:
            raise ValueError(
                f"Prediction chunks must be contiguous; expected {self.next_index}."
            )
        array = np.asarray(values)
        count = int(array.shape[self.metadata.query_axis])
        selection = [slice(None)] * self.values.ndim
        selection[self.metadata.query_axis] = slice(start, start + count)
        expected = self.values[tuple(selection)].shape
        if array.shape != expected:
            raise ValueError(f"Prediction chunk must have shape {expected}; got {array.shape}.")
        self.values[tuple(selection)] = array
        self.next_index += count

    def finish(self) -> np.ndarray:
        if self.metadata is None or self.values is None:
            raise RuntimeError("Prediction sink has not begun.")
        if self.next_index != self.metadata.total_size:
            raise RuntimeError(
                f"Prediction is incomplete: wrote {self.next_index} of "
                f"{self.metadata.total_size} queries."
            )
        return self.values


class NpyPredictionSink(OperatorPredictionSink):
    """Resumable NumPy memmap sink with fail-closed progress metadata."""

    def __init__(self, path: str | Path, /, *, resume: bool = False):
        self.path = Path(path)
        self.metadata_path = self.path.with_suffix(self.path.suffix + ".metadata.json")
        self.resume = bool(resume)
        self.metadata: OperatorPredictionMetadata | None = None
        self.values: np.memmap | None = None
        self.next_index = 0

    def _write_status(self, *, complete: bool) -> None:
        assert self.metadata is not None
        payload = {
            "metadata": asdict(self.metadata),
            "next_index": self.next_index,
            "complete": bool(complete),
        }
        temporary = self.metadata_path.with_suffix(self.metadata_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.metadata_path)

    def begin(self, metadata: OperatorPredictionMetadata, /) -> None:
        if self.metadata is not None:
            raise RuntimeError("Prediction sink has already begun.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.resume:
            if not self.path.exists() or not self.metadata_path.exists():
                raise FileNotFoundError("Resumed prediction requires data and metadata files.")
            status = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            expected = {"metadata", "next_index", "complete"}
            if not isinstance(status, dict):
                raise ValueError("Prediction status must be a JSON object.")
            missing = expected - set(status)
            unknown = set(status) - expected
            if missing or unknown:
                raise ValueError(
                    "Prediction status must use the current canonical fields; "
                    f"missing={sorted(missing)}, unknown={sorted(unknown)}."
                )
            recorded = OperatorPredictionMetadata(**status["metadata"])
            if recorded != metadata:
                raise ValueError("Resumed prediction metadata does not match the request.")
            if bool(status["complete"]):
                raise ValueError("Prediction artifact is already complete.")
            self.values = np.lib.format.open_memmap(self.path, mode="r+")
            self.next_index = int(status["next_index"])
        else:
            self.values = np.lib.format.open_memmap(
                self.path,
                mode="w+",
                dtype=np.dtype(metadata.dtype),
                shape=metadata.output_shape,
            )
            self.next_index = 0
        self.metadata = metadata
        self._write_status(complete=False)

    def write(self, start: int, values: np.ndarray, /) -> None:
        if self.metadata is None or self.values is None:
            raise RuntimeError("Prediction sink has not begun.")
        if int(start) != self.next_index:
            raise ValueError(
                f"Prediction chunks must be contiguous; expected {self.next_index}."
            )
        array = np.asarray(values)
        count = int(array.shape[self.metadata.query_axis])
        selection = [slice(None)] * self.values.ndim
        selection[self.metadata.query_axis] = slice(start, start + count)
        expected = self.values[tuple(selection)].shape
        if array.shape != expected:
            raise ValueError(f"Prediction chunk must have shape {expected}; got {array.shape}.")
        self.values[tuple(selection)] = array
        self.values.flush()
        self.next_index += count
        self._write_status(complete=False)

    def finish(self) -> Path:
        if self.metadata is None or self.values is None:
            raise RuntimeError("Prediction sink has not begun.")
        if self.next_index != self.metadata.total_size:
            raise RuntimeError(
                f"Prediction is incomplete: wrote {self.next_index} of "
                f"{self.metadata.total_size} queries."
            )
        self.values.flush()
        self._write_status(complete=True)
        return self.path


def _crop_query_axis(values: Array, count: int, case_ndim: int, /) -> Array:
    selection = [slice(None)] * values.ndim
    selection[case_ndim] = slice(0, int(count))
    return values[tuple(selection)]


def decode_query_chunks(
    model: _AbstractOperatorModel,
    batch: OperatorBatch,
    query_source: OperatorQuerySource,
    sink: OperatorPredictionSink,
    /,
    *,
    chunk_size: int,
    key: EvalKey = None,
    encoded_state: Any | None = None,
    compile: bool = True,
) -> Any:
    """Decode a query source with bounded device memory and ordered output."""
    size = int(chunk_size)
    if size <= 0:
        raise ValueError("chunk_size must be positive.")
    if tuple(query_source.schema.case_shape) != batch.case_shape:
        raise ValueError("Query source and operator batch case shapes differ.")
    encoded = isinstance(model, AbstractEncodedOperatorModel)
    if encoded_state is not None and not encoded:
        raise TypeError("encoded_state requires an AbstractEncodedOperatorModel.")
    state = (
        model.encode_inputs(batch, key=fold_in_eval_key(key, 0))
        if encoded and encoded_state is None
        else encoded_state
    )

    if encoded:
        encoded_model = model

        def evaluate(query: FunctionSamples, eval_key: EvalKey) -> Array:
            assert isinstance(encoded_model, AbstractEncodedOperatorModel)
            return encoded_model.decode_query(state, query, key=eval_key)

    else:

        def evaluate(query: FunctionSamples, eval_key: EvalKey) -> Array:
            query_batch = OperatorBatch(inputs=batch.inputs, queries={"query": query}, case_axes=batch.case_axes,
            case_shape=batch.case_shape,)
            return model.__call_operator_batch__(query_batch, key=eval_key)

    evaluator = eqx.filter_jit(evaluate) if compile else evaluate
    begun = False
    metadata: OperatorPredictionMetadata | None = None
    for chunk_index, start in enumerate(range(0, query_source.schema.size, size)):
        chunk = query_source.read_chunk(start, size)
        values = jnp.asarray(
            evaluator(chunk.samples, fold_in_eval_key(key, chunk_index + 1))
        )
        cropped = _crop_query_axis(values, chunk.valid_count, len(batch.case_shape))
        host = np.asarray(cropped)
        if not begun:
            channel_shape = tuple(
                int(value) for value in host.shape[len(batch.case_shape) + 1 :]
            )
            metadata = OperatorPredictionMetadata(
                total_size=query_source.schema.size,
                case_shape=batch.case_shape,
                channel_shape=channel_shape,
                dtype=np.dtype(host.dtype).name,
                query_axis=len(batch.case_shape),
                query_fingerprint=query_source.schema.fingerprint,
            )
            sink.begin(metadata)
            begun = True
        sink.write(start, host)
    if metadata is None:
        raise RuntimeError("Query source produced no chunks.")
    return sink.finish()


__all__ = [
    "ArrayOperatorQuerySource",
    "ArrayPredictionSink",
    "CallbackOperatorQuerySource",
    "NpyPredictionSink",
    "OperatorPredictionMetadata",
    "OperatorPredictionSink",
    "OperatorQueryChunk",
    "OperatorQuerySchema",
    "OperatorQuerySource",
    "decode_query_chunks",
]
