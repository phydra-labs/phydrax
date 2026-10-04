#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import stablehlo as hlo

from phydrax.domain import DomainFunction

from .._document_resource import decode_json_resource
from .._external_resource import bounded_resource_from_bytes, ResourceLimits
from .._external_runtime import _require_execution
from .._fingerprint import canonical_fingerprint
from .._identity import (
    ArtifactBindingIdentity,
    ExecutableSignature,
    NumericRevision,
    SemanticProvenance,
)
from .._model._component import ExecutionCapabilities
from .._publication import publish_resource_set, ResourceSetPublicationReceipt
from .._resource_set import open_bounded_resource_set, ResourceSetLimits
from ..backends.iree import import_iree, iree_availability
from ..typing import parse
from ._inference import make_inference_export_callable


_IREE_ARTIFACT_FORMAT = "phydrax-iree-inference"


def _sha256_bytes(value: bytes, /) -> str:
    return hashlib.sha256(value).hexdigest()


def _ir_context() -> ir.Context:
    # A JAX lowering context registers every dialect exported modules use.
    module = jax.jit(lambda: jnp.zeros(())).lower().compiler_ir("stablehlo")
    if module is None:
        raise RuntimeError("JAX produced no StableHLO module for the dialect context.")
    return module.context


def _ranked(value: ir.Value, role: str, /) -> ir.RankedTensorType:
    try:
        return ir.RankedTensorType(value.type)
    except ValueError as error:
        raise ValueError(
            f"IREE input legalization refuses an unranked {role}."
        ) from error


def _static_shape(kind: ir.RankedTensorType, role: str, /) -> list[int]:
    if not kind.has_static_shape:
        raise ValueError(f"IREE input legalization requires a static {role} shape here.")
    return list(kind.shape)


def _inverse_permutation(permutation: Sequence[int], /) -> list[int]:
    inverse = [0] * len(permutation)
    for position, axis in enumerate(permutation):
        inverse[axis] = position
    return inverse


def _transposed(value: ir.Value, permutation: Sequence[int], /) -> ir.Value:
    """Exact transpose that keeps a transposed broadcast one broadcast.

    IREE 3.11 compiles ``transpose(broadcast(x))`` into its own dispatch; when
    ``x`` is a constant that dispatch has no operands and stream copy elision
    shares its buffer between in-place consumers. The equivalent permuted
    broadcast stays a splat, which IREE materializes per consumer.
    """

    permutation = list(permutation)
    if permutation == list(range(len(permutation))):
        return value
    owner = value.owner
    if (
        not isinstance(owner, ir.Block)
        and owner.operation.name == "stablehlo.broadcast_in_dim"
    ):
        broadcast = owner.operation
        position = _inverse_permutation(permutation)
        dimensions = [
            position[axis]
            for axis in ir.DenseI64ArrayAttr(broadcast.attributes["broadcast_dimensions"])
        ]
        if dimensions == sorted(dimensions):
            kind = ir.RankedTensorType(value.type)
            return hlo.BroadcastInDimOp(
                ir.RankedTensorType.get(
                    [kind.shape[axis] for axis in permutation], kind.element_type
                ),
                broadcast.operands[0],
                ir.DenseI64ArrayAttr.get(dimensions),
            ).result
    return hlo.TransposeOp(value, ir.DenseI64ArrayAttr.get(permutation)).result


def _integer_splat(kind: ir.RankedTensorType, value: int, /) -> ir.Value:
    return hlo.ConstantOp(
        ir.DenseElementsAttr.get_splat(kind, ir.IntegerAttr.get(kind.element_type, value))
    ).result


def _integer_vector(values: Sequence[int], element: ir.Type, /) -> ir.Value:
    integer = ir.IntegerType(element)
    dtype = np.dtype(f"{'u' if integer.is_unsigned else 'i'}{integer.width // 8}")
    return hlo.ConstantOp(
        ir.DenseElementsAttr.get(
            np.asarray(values, dtype=dtype),
            type=ir.RankedTensorType.get([len(values)], element),
        )
    ).result


def _index_compare(lhs: ir.Value, rhs: ir.Value, direction: str, /) -> ir.Value:
    element = ir.IntegerType(ir.RankedTensorType(lhs.type).element_type)
    return hlo.CompareOp(
        lhs,
        rhs,
        hlo.ComparisonDirectionAttr.get(direction),
        compare_type=hlo.ComparisonTypeAttr.get(
            "UNSIGNED" if element.is_unsigned else "SIGNED"
        ),
    ).result


def _clamp_gather_indices(operation: ir.Operation, /) -> None:
    """Clamp gather start indices exactly as StableHLO specifies, in place.

    StableHLO clamps each start index into ``[0, operand_dim - slice_size]``
    and JAX lowers ``promise_in_bounds`` and ``clip`` gathers onto that rule;
    IREE 3.11 instead reads zeros for out-of-range starts of point gathers.
    """

    numbers = hlo.GatherDimensionNumbers(operation.attributes["dimension_numbers"])
    starts = list(numbers.start_index_map)
    vector = int(numbers.index_vector_dim)
    sizes = list(ir.DenseI64ArrayAttr(operation.attributes["slice_sizes"]))
    operand = _static_shape(
        _ranked(operation.operands[0], "gather operand"), "gather operand"
    )
    indices = operation.operands[1]
    kind = _ranked(indices, "gather index")
    _static_shape(kind, "gather index")
    limits = [operand[axis] - sizes[axis] for axis in starts]
    with ir.InsertionPoint(operation):
        upper = (
            _integer_splat(kind, limits[0])
            if vector == kind.rank
            else hlo.BroadcastInDimOp(
                kind,
                _integer_vector(limits, kind.element_type),
                ir.DenseI64ArrayAttr.get([vector]),
            ).result
        )
        clamped = hlo.ClampOp(_integer_splat(kind, 0), indices, upper).result
    operation.operands[1] = clamped


def _within(indices: ir.Value, upper: Sequence[int], /) -> ir.Value:
    """``[batch...]`` truth of ``0 <= indices[..., k] <= upper[k]`` for all ``k``."""

    kind = _ranked(indices, "scatter index")
    shape = _static_shape(kind, "scatter index")
    inside = hlo.AndOp(
        _index_compare(indices, _integer_splat(kind, 0), "GE"),
        _index_compare(
            indices,
            hlo.BroadcastInDimOp(
                kind,
                _integer_vector(upper, kind.element_type),
                ir.DenseI64ArrayAttr.get([len(shape) - 1]),
            ).result,
            "LE",
        ),
    ).result
    column_type = ir.RankedTensorType.get(shape[:-1], ir.IntegerType.get_signless(1))
    columns = [
        hlo.ReshapeOp(
            column_type,
            hlo.SliceOp(
                inside,
                [0] * (len(shape) - 1) + [component],
                [*shape[:-1], component + 1],
                [1] * len(shape),
            ).result,
        ).result
        for component in range(shape[-1])
    ]
    valid = columns[0]
    for column in columns[1:]:
        valid = hlo.AndOp(valid, column).result
    return valid


def _broadcast_rows(valid: ir.Value, shape: Sequence[int], /) -> ir.Value:
    return hlo.BroadcastInDimOp(
        ir.RankedTensorType.get(list(shape), ir.IntegerType.get_signless(1)),
        valid,
        ir.DenseI64ArrayAttr.get(list(range(ir.RankedTensorType(valid.type).rank))),
    ).result


def _window_points(
    indices: ir.Value,
    windowed: Sequence[int],
    extents: Sequence[int],
    sizes: Sequence[int],
    batch_rank: int,
    /,
) -> ir.Value:
    """Expand window starts on scattered axes into per-point scatter indices.

    ``indices`` is ``[batch..., K]`` over leading scattered axes of extents
    ``sizes``; component ``windowed[j]`` gains an iota over a new batch axis of
    length ``extents[j]``. StableHLO drops a whole update window when its start
    leaves ``[0, size - window]`` on any scattered axis; every point of such a
    window gets an out-of-range index so each point is dropped too.
    """

    kind = _ranked(indices, "scatter index")
    shape = _static_shape(kind, "scatter index")
    element = kind.element_type
    components = shape[-1]
    points_shape = [*shape[:-1], *extents, components]
    points_type = ir.RankedTensorType.get(points_shape, element)
    component_type = ir.RankedTensorType.get([*points_shape[:-1], 1], element)
    starts = hlo.BroadcastInDimOp(
        points_type,
        indices,
        ir.DenseI64ArrayAttr.get([*range(batch_rank), len(points_shape) - 1]),
    ).result
    offsets = hlo.ConcatenateOp(
        [
            hlo.IotaOp(component_type, batch_rank + windowed.index(component)).result
            if component in windowed
            else _integer_splat(component_type, 0)
            for component in range(components)
        ],
        len(points_shape) - 1,
    ).result
    window_starts = _within(
        indices,
        [
            size - (extents[windowed.index(component)] if component in windowed else 1)
            for component, size in enumerate(sizes)
        ],
    )
    # Every point of a dropped window is out of range on its widest axis.
    return hlo.SelectOp(
        _broadcast_rows(window_starts, points_shape),
        hlo.AddOp(starts, offsets).result,
        _integer_splat(points_type, max(sizes)),
    ).result


def _trash_row_points(indices: ir.Value, sizes: Sequence[int], /) -> ir.Value:
    """Send every out-of-range point index to row ``sizes[0]`` of axis 0.

    JAX relies on XLA dropping out-of-range scatter updates; IREE 3.11 writes
    them past the result buffer. The scatter operand gains one discarded trash
    row on its leading scattered axis, so redirected points stay in bounds.
    """

    kind = _ranked(indices, "scatter index")
    shape = _static_shape(kind, "scatter index")
    trash = hlo.BroadcastInDimOp(
        kind,
        _integer_vector([sizes[0], *([0] * (len(sizes) - 1))], kind.element_type),
        ir.DenseI64ArrayAttr.get([len(shape) - 1]),
    ).result
    valid = _within(indices, [size - 1 for size in sizes])
    return hlo.SelectOp(_broadcast_rows(valid, shape), indices, trash).result


def _splat_scalar(value: ir.Value, /) -> ir.Value | None:
    """Rank-0 value that ``value`` broadcasts everywhere, or ``None``."""

    owner = value.owner
    if isinstance(owner, ir.Block):
        return None
    operation = owner.operation
    if operation.name == "stablehlo.constant":
        try:
            attribute = ir.DenseElementsAttr(operation.attributes["value"])
        except ValueError:
            return None
        if not attribute.is_splat:
            return None
        element = ir.RankedTensorType(value.type).element_type
        return hlo.ConstantOp(
            ir.DenseElementsAttr.get_splat(
                ir.RankedTensorType.get([], element), attribute.get_splat_value()
            )
        ).result
    if operation.name in ("stablehlo.broadcast_in_dim", "stablehlo.reshape"):
        source = operation.operands[0]
        if ir.RankedTensorType(source.type).rank == 0:
            return source
        return _splat_scalar(source)
    return None


def _with_trash_row(value: ir.Value, /) -> ir.Value:
    """``value`` with one more row on axis 0; a splat stays a splat."""

    kind = ir.RankedTensorType(value.type)
    shape = list(kind.shape)
    scalar = _splat_scalar(value)
    if scalar is not None:
        return hlo.BroadcastInDimOp(
            ir.RankedTensorType.get([shape[0] + 1, *shape[1:]], kind.element_type),
            scalar,
            ir.DenseI64ArrayAttr.get([]),
        ).result
    row = hlo.SliceOp(value, [0] * len(shape), [1, *shape[1:]], [1] * len(shape))
    return hlo.ConcatenateOp([value, row.result], 0).result


def _legalize_scatter(operation: ir.Operation, /) -> None:
    """Rewrite one scatter into IREE's canonical point form, in place.

    The rewrite is exact: the index vector becomes the trailing index
    dimension; operand/indices batching dimensions become explicit iota index
    components and inserted axes without an index become zero components; the
    scattered operand axes lead in index-vector order and update windows trail
    in operand order; an update window on a scattered axis (dynamic-update-
    slice style) is expanded into per-point indices that keep StableHLO's
    whole-window out-of-bounds drop; out-of-range points land in one discarded
    trash row (``_trash_row_points``). Every input/update of a variadic scatter
    is permuted alike, the update region and the unique flag are kept (only
    trash points may repeat), and each result is sliced and transposed back to
    its original type. Updates are fenced by an optimization barrier so their
    producers are materialized rather than fused into the scatter dispatch.
    """

    count = len(operation.results)
    inputs = list(operation.operands[:count])
    indices = operation.operands[count]
    updates = list(operation.operands[count + 1 :])
    numbers = hlo.ScatterDimensionNumbers(
        operation.attributes["scatter_dimension_numbers"]
    )
    window = list(numbers.update_window_dims)
    inserted = list(numbers.inserted_window_dims)
    operand_batching = list(numbers.input_batching_dims)
    index_batching = list(numbers.scatter_indices_batching_dims)
    scattered = list(numbers.scattered_dims_to_operand_dims)
    vector = int(numbers.index_vector_dim)
    operand_shape = _static_shape(
        _ranked(inputs[0], "scatter operand"), "scatter operand"
    )
    operand_rank = len(operand_shape)
    update_shape = _static_shape(_ranked(updates[0], "scatter update"), "scatter update")
    update_rank = len(update_shape)
    index_type = _ranked(indices, "scatter index")
    reordered = False
    with ir.InsertionPoint(operation):
        if vector == index_type.rank:
            shape = _static_shape(index_type, "scatter index")
            indices = hlo.ReshapeOp(
                ir.RankedTensorType.get([*shape, 1], index_type.element_type), indices
            ).result
        elif vector != index_type.rank - 1:
            order = [axis for axis in range(index_type.rank) if axis != vector]
            indices = _transposed(indices, [*order, vector])
            index_batching = [order.index(axis) for axis in index_batching]
        vector = _ranked(indices, "scatter index").rank - 1
        unindexed = [
            axis
            for axis in inserted
            if axis not in scattered and axis not in operand_batching
        ]
        if operand_batching or unindexed:
            index_type = _ranked(indices, "scatter index")
            component = _static_shape(index_type, "scatter index")
            component[vector] = 1
            component_type = ir.RankedTensorType.get(component, index_type.element_type)
            indices = hlo.ConcatenateOp(
                [
                    indices,
                    *(hlo.IotaOp(component_type, axis).result for axis in index_batching),
                    *(_integer_splat(component_type, 0) for _ in unindexed),
                ],
                vector,
            ).result
            scattered = [*scattered, *operand_batching, *unindexed]
            inserted = sorted({*inserted, *operand_batching})
            # Appended iota components change the lexicographic index order.
            reordered = reordered or bool(operand_batching)
        operand_window = [axis for axis in range(operand_rank) if axis not in inserted]
        if len(operand_window) != len(window):
            raise ValueError(
                "Scatter window dimensions are inconsistent with its operand."
            )
        update_of = dict(zip(operand_window, window, strict=True))
        operand_order = [
            *scattered,
            *(axis for axis in range(operand_rank) if axis not in scattered),
        ]
        update_batch = [axis for axis in range(update_rank) if axis not in window]
        update_order = [
            *update_batch,
            *(update_of[axis] for axis in operand_order if axis not in inserted),
        ]
        canonical_inserted = sorted(operand_order.index(axis) for axis in inserted)
        windowed = [
            position
            for position in range(len(scattered))
            if position not in canonical_inserted
        ]
        batch_rank = len(update_batch)
        if windowed:
            extents = [
                update_shape[update_of[operand_order[position]]] for position in windowed
            ]
            indices = _window_points(
                indices,
                windowed,
                extents,
                [operand_shape[axis] for axis in operand_order[: len(scattered)]],
                batch_rank,
            )
            vector = batch_rank + len(windowed)
            batch_rank = vector
            canonical_inserted = sorted({*canonical_inserted, *windowed})
            reordered = True
        sizes = [operand_shape[axis] for axis in operand_order[: len(scattered)]]
        permuted_inputs = [_transposed(value, operand_order) for value in inputs]
        destinations = permuted_inputs
        if scattered:
            if 0 in operand_shape:
                raise ValueError(
                    "IREE scatter legalization refuses an empty scatter operand."
                )
            indices = _trash_row_points(indices, sizes)
            destinations = [_with_trash_row(value) for value in permuted_inputs]
            reordered = True
        fenced = hlo.OptimizationBarrierOp(
            [_transposed(value, update_order) for value in updates]
        ).results
    for index, value in enumerate(destinations):
        operation.operands[index] = value
    if reordered:
        operation.attributes["indices_are_sorted"] = ir.BoolAttr.get(False)
    operation.operands[count] = indices
    for index, value in enumerate(fenced):
        operation.operands[count + 1 + index] = value
    operation.attributes["scatter_dimension_numbers"] = hlo.ScatterDimensionNumbers.get(
        update_window_dims=list(range(batch_rank, update_rank)),
        inserted_window_dims=canonical_inserted,
        input_batching_dims=[],
        scatter_indices_batching_dims=[],
        scattered_dims_to_operand_dims=list(range(len(scattered))),
        index_vector_dim=vector,
    )
    restore = _inverse_permutation(operand_order)
    with ir.InsertionPoint.after(operation):
        for result, destination, value in zip(
            operation.results, destinations, permuted_inputs, strict=True
        ):
            result.set_type(destination.type)
            shape = list(ir.RankedTensorType(value.type).shape)
            kept = (
                result
                if destination == value
                else hlo.SliceOp(result, [0] * len(shape), shape, [1] * len(shape)).result
            )
            restored = _transposed(kept, restore)
            if restored != result:
                consumer = (kept if kept != result else restored).owner
                if isinstance(consumer, ir.Block):
                    raise RuntimeError(
                        "Internal invariant failed: restore has no producer."
                    )
                result.replace_all_uses_except(restored, consumer.operation)


def _in_place_destinations(operation: Any, /) -> list[Any]:
    """Operands IREE lowers in place: they become the result buffers."""

    if operation.name == "stablehlo.scatter":
        return list(operation.operands[: len(operation.results)])
    if operation.name == "stablehlo.dynamic_update_slice":
        return [operation.operands[0]]
    if operation.name == "stablehlo.sort":
        return list(operation.operands)
    return []


_BUFFER_VIEWS = frozenset(
    {"stablehlo.reshape", "stablehlo.bitcast_convert", "stablehlo.optimization_barrier"}
)


def _terminator(region: Any, /) -> Any:
    return list(region.blocks[0].operations)[-1].operation


def _runtime_dependence_marks(operation: Any, runtime: set[Any], /) -> list[Any]:
    """Values of ``operation`` that depend on a runtime input given ``runtime``."""

    if operation.name == "stablehlo.while":
        carried = _terminator(operation.regions[1]).operands
        return [
            value
            for position, operand in enumerate(operation.operands)
            if operand in runtime or carried[position] in runtime
            for value in (
                operation.regions[0].blocks[0].arguments[position],
                operation.regions[1].blocks[0].arguments[position],
                operation.results[position],
            )
        ]
    dependent = operation.name == "stablehlo.custom_call" or operation.name.startswith(
        "stablehlo.rng"
    )
    dependent = dependent or any(value in runtime for value in operation.operands)
    marks: list[Any] = []
    for region in operation.regions:
        for block in region.blocks:
            if dependent:
                marks.extend(block.arguments)
            dependent = dependent or any(
                value in runtime
                for nested in block.operations
                for value in nested.operation.operands
            )
    if dependent:
        marks.extend(operation.results)
    return marks


def _runtime_dependent(
    operations: Sequence[Any], arguments: Sequence[Any], /
) -> set[Any]:
    """Values that depend on an entry argument (fixed point over loop carries)."""

    runtime = set(arguments)
    changed = True
    while changed:
        changed = False
        for operation in operations:
            if operation.name == "func.func":
                continue
            for value in _runtime_dependence_marks(operation, runtime):
                if value not in runtime:
                    runtime.add(value)
                    changed = True
    return runtime


@dataclass(frozen=True, slots=True)
class _BufferFlow:
    """Directed buffer identity: views, barriers, loop and branch carries.

    ``transparent`` holds the (operation, operand) uses that only forward a
    buffer; every other use consumes it.
    """

    forward: dict[Any, list[Any]]
    backward: dict[Any, list[Any]]
    transparent: set[tuple[Any, int]]

    def reachable(self, value: Any, /, *, upstream: bool = False) -> set[Any]:
        edges = self.backward if upstream else self.forward
        found = {value}
        pending = [value]
        while pending:
            for target in edges.get(pending.pop(), ()):
                if target not in found:
                    found.add(target)
                    pending.append(target)
        return found

    def consumers(self, value: Any, /) -> int:
        return sum(
            1
            for alias in self.reachable(value)
            for use in alias.uses
            if (use.owner.operation, use.operand_number) not in self.transparent
        )


def _buffer_flow(operations: Sequence[Any], /) -> _BufferFlow:
    flow = _BufferFlow({}, {}, set())

    def flows(source: Any, target: Any) -> None:
        flow.forward.setdefault(source, []).append(target)
        flow.backward.setdefault(target, []).append(source)

    for operation in operations:
        if operation.name in _BUFFER_VIEWS or (
            operation.name == "stablehlo.convert"
            and operation.operands[0].type == operation.results[0].type
        ):
            for position, (operand, result) in enumerate(
                zip(operation.operands, operation.results, strict=True)
            ):
                flows(operand, result)
                flow.transparent.add((operation, position))
        elif operation.name == "stablehlo.while":
            carried = _terminator(operation.regions[1])
            for position, operand in enumerate(operation.operands):
                for target in (
                    operation.regions[0].blocks[0].arguments[position],
                    operation.regions[1].blocks[0].arguments[position],
                ):
                    flows(operand, target)
                    flows(carried.operands[position], target)
                flows(carried.operands[position], operation.results[position])
                flow.transparent.add((operation, position))
                flow.transparent.add((carried, position))
        elif operation.name == "stablehlo.case":
            for region in operation.regions:
                branch = _terminator(region)
                for position, result in enumerate(operation.results):
                    flows(branch.operands[position], result)
                    flow.transparent.add((branch, position))
    return flow


def _is_splat(value: Any, /) -> bool:
    """Whether ``value`` is one scalar broadcast everywhere (IREE splats it per use)."""

    owner = value.owner
    if isinstance(owner, ir.Block):
        return False
    operation = owner.operation
    if operation.name == "stablehlo.constant":
        try:
            return bool(ir.DenseElementsAttr(operation.attributes["value"]).is_splat)
        except ValueError:
            return False
    if operation.name == "stablehlo.broadcast_in_dim":
        source = operation.operands[0]
        return ir.RankedTensorType(source.type).rank == 0 or _is_splat(source)
    if operation.name in ("stablehlo.reshape", "stablehlo.convert"):
        return _is_splat(operation.operands[0])
    return False


def _in_place_hazard_findings(
    operations: Sequence[Any],
    arguments: Sequence[Any],
    returned: set[Any],
    runtime: set[Any],
    flow: _BufferFlow,
    /,
) -> tuple[str, ...]:
    """Classify the three aliasing shapes, in destination/argument/barrier order."""

    destinations = {
        value: operation
        for operation in operations
        for value in _in_place_destinations(operation)
    }
    hazards: list[str] = []
    for destination, operation in destinations.items():
        if any(
            source not in runtime and not _is_splat(source) and flow.consumers(source) > 1
            for source in flow.reachable(destination, upstream=True)
        ):
            hazards.append(
                f"{operation.name} at {operation.location}: shared input-independent destination"
            )
    for argument in arguments:
        aliases = flow.reachable(argument)
        if aliases & returned and any(value in destinations for value in aliases):
            hazards.append(
                f"entry argument {argument.arg_number}: returned while updated in place"
            )
    for operation in operations:
        if operation.name == "stablehlo.optimization_barrier" and any(
            value in destinations
            for operand in operation.operands
            for value in flow.reachable(operand)
        ):
            hazards.append(
                f"optimization_barrier at {operation.location}: operand reaches an "
                "in-place destination"
            )
    return tuple(dict.fromkeys(hazards))


def _iree_in_place_hazards(module: str, /) -> tuple[str, ...]:
    """In-place updates IREE 3.11 stream lowering would alias, by location.

    IREE lowers scatter, dynamic-update-slice and sort in place on their
    destination buffers. Its stream copy elision (iree-base-compiler 3.11) is
    unsound in three observed shapes, each reproduced against XLA:

    * an input-independent, non-splat destination with another use: the
      producer is a dispatch without operands that is rematerialized and then
      merged again by CSE, so two consumers share one buffer;
    * a function argument that is also returned: the import is mutated in
      place and the returned view sees the update;
    * an ``optimization_barrier`` whose operand reaches an in-place
      destination: barrier result and operand alias one buffer that the copy
      analysis treats as two values.

    The audit runs on IREE's own inlined, CSE'd view of the module (the
    provider's ``iree-opt`` tool, out of process like ``iree-compile``) and
    follows buffer views, barriers and loop carries. It is conservative: it may
    refuse a module IREE would compile correctly, never the reverse for these
    shapes.
    """

    compiler, _ = import_iree()
    tools = importlib.import_module(f"{compiler.__name__}.tools.binaries")
    inlined = tools.invoke_immediate(
        [
            tools.find_tool("iree-opt"),
            "--pass-pipeline=builtin.module(inline,cse,symbol-dce)",
            "--mlir-print-op-generic",
            "-",
        ],
        immediate_input=module.encode("utf-8"),
    )
    with _ir_context():
        parsed = ir.Module.parse(inlined.decode("utf-8"))
        operations: list[Any] = []

        def collect(operation: Any) -> Any:
            operations.append(operation)
            return ir.WalkResult.ADVANCE

        parsed.operation.walk(collect)
        functions = [op for op in operations if op.name == "func.func"]
        if len(functions) != 1:
            raise ValueError("IREE in-place audit requires one inlined entry function.")
        entry = functions[0].regions[0].blocks[0]
        arguments = list(entry.arguments)
        return _in_place_hazard_findings(
            operations,
            arguments,
            set(list(entry.operations)[-1].operands),
            _runtime_dependent(operations, arguments),
            _buffer_flow(operations),
        )


def _legalized_iree_input(module: str, /) -> str:
    """Return exported StableHLO in the form IREE compiles exactly.

    IREE's ``llvm-cpu`` input conversion rejects scatters whose scattered axes
    are not leading (for example ``vmap`` of a segment sum) or carry update
    windows, stack-allocates producers it fuses into scatter dispatches, and
    reads zeros for out-of-range gather starts. Scatters are rewritten into
    IREE's point form (see ``_legalize_scatter``) and gather starts are
    clamped as StableHLO specifies (``_clamp_gather_indices``); both rewrites
    are exact. Forms that cannot be expressed statically, and in-place update
    patterns IREE's stream lowering would alias (``_iree_in_place_hazards``),
    refuse with ``ValueError``.
    """

    with _ir_context(), ir.Location.unknown():
        parsed = ir.Module.parse(module)
        rewrites: list[ir.Operation] = []

        def collect(operation: ir.Operation) -> ir.WalkResult:
            if operation.name in ("stablehlo.scatter", "stablehlo.gather"):
                rewrites.append(operation)
            return ir.WalkResult.ADVANCE

        parsed.operation.walk(collect)
        for operation in rewrites:
            if operation.name == "stablehlo.scatter":
                _legalize_scatter(operation)
            else:
                _clamp_gather_indices(operation)
        if not parsed.operation.verify():
            raise ValueError("Legalized IREE input module failed verification.")
        legalized = str(parsed)
    hazards = _iree_in_place_hazards(legalized)
    if hazards:
        raise ValueError(
            "IREE in-place lowering would alias buffers (iree-base-compiler 3.11 "
            "stream copy elision); restructure the exported program: "
            + "; ".join(hazards)
        )
    return legalized


IREEExecutableFormat: TypeAlias = Literal["embedded-elf", "system-library"]


def _executable_platform(executable_format: IREEExecutableFormat, /) -> str:
    """Host constraint of one compiled executable format.

    An embedded ELF is OS-independent but bound to the compiling machine
    architecture; a system library is a host shared library bound to the
    compiling OS, architecture, and system linker/C math library.
    """

    machine = platform.machine().lower()
    if executable_format == "embedded-elf":
        return f"embedded-elf-{machine}"
    return f"system-library-{sys.platform}-{machine}"


@dataclass(frozen=True, slots=True)
class IREEExportPolicy:
    """Static IREE compilation target, runtime driver, and executable format.

    ``executable_format="embedded-elf"`` (default) links IREE's portable
    embedded ELF, which carries no C math library: ``llvm-cpu`` float64
    transcendentals (``sin``, ``exp``, ``log``, ``tanh``, ...) cannot link.
    ``"system-library"`` links a host shared library with the host system
    linker and C math library; it requires that linker at export time and
    loads only on the same OS and architecture. It is valid only for the
    ``llvm-cpu`` backend. ``system_linker`` is the linker executable; when
    omitted it is resolved once from ``PATH`` as ``ld`` and recorded as an
    absolute path, so compilation never searches the environment again.
    """

    target_backend: str = "llvm-cpu"
    runtime_driver: str = "local-task"
    executable_format: IREEExecutableFormat = "embedded-elf"
    system_linker: str | None = None

    def __post_init__(self) -> None:
        if not self.target_backend or not self.runtime_driver:
            raise ValueError("IREE target_backend and runtime_driver must be non-empty.")
        parse(self.executable_format, IREEExecutableFormat, "executable_format")
        if self.executable_format == "embedded-elf":
            if self.system_linker is not None:
                raise ValueError("Embedded-ELF IREE executables use no system linker.")
            return
        if self.target_backend != "llvm-cpu":
            raise ValueError(
                "IREE system-library executables require the llvm-cpu target backend."
            )
        linker = shutil.which("ld") if self.system_linker is None else self.system_linker
        if linker is None:
            raise ValueError("IREE system-library export requires a host system linker.")
        resolved = Path(linker).expanduser().resolve(strict=True)
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ValueError("system_linker must be an executable regular file.")
        object.__setattr__(self, "system_linker", str(resolved))

    @property
    def compiler_arguments(self) -> tuple[str, ...]:
        linkage = (
            (
                "--iree-llvmcpu-link-embedded=false",
                f"--iree-llvmcpu-system-linker-path={self.system_linker}",
            )
            if self.executable_format == "system-library"
            else ()
        )
        return (
            "--iree-input-demote-f64-to-f32=false",
            "--iree-input-demote-i64-to-i32=false",
            "--iree-opt-const-expr-hoisting=false",
            *linkage,
        )


@dataclass(frozen=True, slots=True)
class IREEArtifactManifest:
    """Canonical executable identity, ABI, and validation evidence."""

    format: str
    artifact_id: str
    module_file: str
    module_sha256: str
    compiler_version: str
    runtime_version: str
    target_backend: str
    runtime_driver: str
    executable_format: IREEExecutableFormat
    executable_platform: str
    system_linker: str | None
    function_name: str
    entry_point: str
    calling_convention_version: int
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[str, ...]
    vectorized: bool
    has_preprocess: bool
    has_postprocess: bool
    validation_ok: bool | None
    maximum_absolute_errors: tuple[float, ...] | None
    maximum_relative_errors: tuple[float, ...] | None
    domain_contract: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.input_names
            or len(self.input_names) != len(self.input_shapes)
            or len(self.input_names) != len(self.input_dtypes)
            or any(not name for name in self.input_names)
            or len(set(self.input_names)) != len(self.input_names)
        ):
            raise ValueError(
                "IREE manifest input metadata must define one unique name per input."
            )
        output_count = len(self.output_names)
        if (
            output_count == 0
            or output_count != len(self.output_shapes)
            or output_count != len(self.output_dtypes)
            or any(not name for name in self.output_names)
            or len(set(self.output_names)) != output_count
        ):
            raise ValueError(
                "IREE manifest output metadata must define one unique name per output."
            )
        if self.validation_ok is None:
            if (
                self.maximum_absolute_errors is not None
                or self.maximum_relative_errors is not None
            ):
                raise ValueError(
                    "IREE manifest validation errors require validation evidence."
                )
        elif (
            self.maximum_absolute_errors is None
            or self.maximum_relative_errors is None
            or len(self.maximum_absolute_errors) != output_count
            or len(self.maximum_relative_errors) != output_count
        ):
            raise ValueError(
                "IREE manifest validation errors must describe every output."
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "IREEArtifactManifest":
        expected = set(cls.__dataclass_fields__)
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            if {"executable_format", "executable_platform"} <= missing:
                raise ValueError(
                    "IREE manifest predates executable-format identity; re-export it."
                )
            raise ValueError(
                f"IREE manifest fields are not canonical; missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        if value["format"] != _IREE_ARTIFACT_FORMAT:
            raise ValueError("Artifact is not a Phydrax IREE inference bundle.")
        return cls(
            format=str(value["format"]),
            artifact_id=str(value["artifact_id"]),
            module_file=str(value["module_file"]),
            module_sha256=str(value["module_sha256"]),
            compiler_version=str(value["compiler_version"]),
            runtime_version=str(value["runtime_version"]),
            target_backend=str(value["target_backend"]),
            runtime_driver=str(value["runtime_driver"]),
            executable_format=parse(
                value["executable_format"], IREEExecutableFormat, "executable_format"
            ),
            executable_platform=str(value["executable_platform"]),
            system_linker=(
                None if value["system_linker"] is None else str(value["system_linker"])
            ),
            function_name=str(value["function_name"]),
            entry_point=str(value["entry_point"]),
            calling_convention_version=int(value["calling_convention_version"]),
            input_names=tuple(str(name) for name in value["input_names"]),
            input_shapes=tuple(tuple(shape) for shape in value["input_shapes"]),
            input_dtypes=tuple(str(dtype) for dtype in value["input_dtypes"]),
            output_names=tuple(str(name) for name in value["output_names"]),
            output_shapes=tuple(tuple(shape) for shape in value["output_shapes"]),
            output_dtypes=tuple(str(dtype) for dtype in value["output_dtypes"]),
            vectorized=bool(value["vectorized"]),
            has_preprocess=bool(value["has_preprocess"]),
            has_postprocess=bool(value["has_postprocess"]),
            validation_ok=(
                None if value["validation_ok"] is None else bool(value["validation_ok"])
            ),
            maximum_absolute_errors=(
                None
                if value["maximum_absolute_errors"] is None
                else tuple(float(error) for error in value["maximum_absolute_errors"])
            ),
            maximum_relative_errors=(
                None
                if value["maximum_relative_errors"] is None
                else tuple(float(error) for error in value["maximum_relative_errors"])
            ),
            domain_contract=(
                None
                if value["domain_contract"] is None
                else str(value["domain_contract"])
            ),
        )

    def binding_identity(self) -> ArtifactBindingIdentity:
        """Artifact binding identity of the executable this manifest describes.

        The semantic provenance is the calling ABI and pre/postprocessing
        declaration with the module as a named resource, the numeric revision is
        the module digest, and the executable signature records the input/output
        shapes and dtypes with the compiler, runtime, target, and driver.
        """
        semantic = SemanticProvenance(
            {
                "kind": "iree-inference-executable",
                "artifact_id": self.artifact_id,
                "function_name": self.function_name,
                "entry_point": self.entry_point,
                "calling_convention_version": self.calling_convention_version,
                "input_names": self.input_names,
                "output_names": self.output_names,
                "vectorized": self.vectorized,
                "has_preprocess": self.has_preprocess,
                "has_postprocess": self.has_postprocess,
            },
            resource_ids={"module": self.module_sha256},
        )
        names = tuple(f"input:{name}" for name in self.input_names) + tuple(
            f"output:{name}" for name in self.output_names
        )
        return ArtifactBindingIdentity(
            semantic,
            NumericRevision(semantic, {"module_sha256": self.module_sha256}),
            ExecutableSignature(
                shapes=tuple(
                    zip(names, self.input_shapes + self.output_shapes, strict=True)
                ),
                dtypes=tuple(
                    zip(names, self.input_dtypes + self.output_dtypes, strict=True)
                ),
                backend_facts={
                    "compiler_version": self.compiler_version,
                    "runtime_version": self.runtime_version,
                    "target_backend": self.target_backend,
                    "runtime_driver": self.runtime_driver,
                    "executable_format": self.executable_format,
                    "executable_platform": self.executable_platform,
                    **(
                        {}
                        if self.system_linker is None
                        else {"system_linker": self.system_linker}
                    ),
                },
            ),
        )

    def to_dict(self, /) -> dict[str, Any]:
        value = asdict(self)
        value["input_names"] = list(self.input_names)
        value["input_shapes"] = [list(shape) for shape in self.input_shapes]
        value["input_dtypes"] = list(self.input_dtypes)
        value["output_names"] = list(self.output_names)
        value["output_shapes"] = [list(shape) for shape in self.output_shapes]
        value["output_dtypes"] = list(self.output_dtypes)
        if self.maximum_absolute_errors is not None:
            value["maximum_absolute_errors"] = list(self.maximum_absolute_errors)
        if self.maximum_relative_errors is not None:
            value["maximum_relative_errors"] = list(self.maximum_relative_errors)
        return value


@dataclass(frozen=True, slots=True)
class IREEExportResult:
    """Published executable bundle and native-parity evidence."""

    path: Path
    manifest: IREEArtifactManifest
    publication: ResourceSetPublicationReceipt | None = None


# Element types of a module's ``iree.abi.declaration`` that the raw VM/HAL
# boundary admits. Signed integers are declared signless; an unsigned integer
# is its signless storage type carrying an explicit ``ui<N>`` encoding.
_DECLARED_DTYPES = {
    "i1": np.dtype(np.bool_),
    "i8": np.dtype(np.int8),
    "i16": np.dtype(np.int16),
    "i32": np.dtype(np.int32),
    "i64": np.dtype(np.int64),
    "ui8": np.dtype(np.uint8),
    "ui16": np.dtype(np.uint16),
    "ui32": np.dtype(np.uint32),
    "ui64": np.dtype(np.uint64),
    "f16": np.dtype(np.float16),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
    "complex<f32>": np.dtype(np.complex64),
    "complex<f64>": np.dtype(np.complex128),
}
# HAL element types a result buffer view may carry for each admitted dtype. The
# runtime reports signed integers as signless ``INT_<N>``; a boolean is stored
# as one ``BOOL_8`` byte per element holding exactly 0 or 1.
_HAL_ELEMENT_TYPES: dict[str, tuple[str, ...]] = {
    np.dtype(np.bool_).str: ("BOOL_8",),
    np.dtype(np.int8).str: ("INT_8", "SINT_8"),
    np.dtype(np.int16).str: ("INT_16", "SINT_16"),
    np.dtype(np.int32).str: ("INT_32", "SINT_32"),
    np.dtype(np.int64).str: ("INT_64", "SINT_64"),
    np.dtype(np.uint8).str: ("UINT_8",),
    np.dtype(np.uint16).str: ("UINT_16",),
    np.dtype(np.uint32).str: ("UINT_32",),
    np.dtype(np.uint64).str: ("UINT_64",),
    np.dtype(np.float16).str: ("FLOAT_16",),
    np.dtype(np.float32).str: ("FLOAT_32",),
    np.dtype(np.float64).str: ("FLOAT_64",),
    np.dtype(np.complex64).str: ("COMPLEX_64",),
    np.dtype(np.complex128).str: ("COMPLEX_128",),
}
_ABI_TENSOR = r"tensor<((?:\d+x)*)(complex<f\d+>|u?i\d+|f\d+)>"
_ABI_STRING = r'"(?:[^"\\]|\\.)*"'
# One declared value with its optional attribute dictionary; only the
# ``iree.abi.encoding`` attribute changes the calling ABI.
_ABI_VALUE = re.compile(
    rf'%(input|output)(\d+): {_ABI_TENSOR}(?: \{{((?:[^{{}}"]|{_ABI_STRING})*)\}})?'
)
_ABI_ENCODING = re.compile(rf"(?:^|, )iree\.abi\.encoding = {_ABI_TENSOR}(?=, |$)")


def _declared_values(text: str, role: str, /) -> tuple[tuple[tuple[int, ...], str], ...]:
    """Ordered ``(shape, dtype)`` of one side of an IREE ABI declaration."""

    values: list[tuple[tuple[int, ...], str]] = []
    position = 0
    while position < len(text):
        index = len(values)
        if index:
            if not text.startswith(", ", position):
                raise ValueError(
                    f"IREE module declares an unadmitted {role} ABI {text!r}."
                )
            position += 2
        match = _ABI_VALUE.match(text, position)
        if match is None or match[1] != role or int(match[2]) != index:
            raise ValueError(f"IREE module declares an unadmitted {role} ABI {text!r}.")
        position = match.end()
        shape = tuple(int(extent) for extent in match[3].split("x")[:-1])
        element = match[4]
        encodings = tuple(
            _ABI_ENCODING.finditer(re.sub(_ABI_STRING, '""', match[5] or ""))
        )
        if encodings:
            encoded = tuple(int(extent) for extent in encodings[0][1].split("x")[:-1])
            if (
                len(encodings) != 1
                or encoded != shape
                or encodings[0][2] != f"u{element}"
            ):
                raise ValueError(
                    f"IREE module {role} {index} declares an unadmitted encoding."
                )
            element = encodings[0][2]
        if element not in _DECLARED_DTYPES:
            raise ValueError(
                f"IREE module {role} {index} has unsupported element type {element}."
            )
        values.append((shape, _DECLARED_DTYPES[element].str))
    return tuple(values)


def _require_declared_abi(
    manifest: IREEArtifactManifest, reflection: Mapping[str, str], /
) -> None:
    """Refuse a manifest whose positional ABI differs from the module's own.

    The compiled entry point reflects its synchronous calling ABI as
    ``iree.abi.declaration``; every input and output shape and element type
    the manifest asserts must be exactly that declaration.
    """

    declaration = reflection.get("iree.abi.declaration")
    match = (
        re.fullmatch(
            rf"sync func @{re.escape(manifest.entry_point)}\((.*)\) -> \((.*)\)",
            declaration,
        )
        if isinstance(declaration, str)
        else None
    )
    if match is None:
        raise ValueError(
            f"IREE entry point {manifest.entry_point!r} declares no admitted "
            "synchronous ABI."
        )
    for role, declared, shapes, dtypes in (
        ("input", match[1], manifest.input_shapes, manifest.input_dtypes),
        ("output", match[2], manifest.output_shapes, manifest.output_dtypes),
    ):
        module = _declared_values(declared, role)
        recorded = tuple(
            (tuple(shape), dtype) for shape, dtype in zip(shapes, dtypes, strict=True)
        )
        if len(module) != len(recorded):
            raise ValueError(
                f"IREE module declares {len(module)} {role}s; the artifact manifest "
                f"declares {len(recorded)}."
            )
        for index, (actual, expected) in enumerate(zip(module, recorded, strict=True)):
            if actual != expected:
                raise ValueError(
                    f"IREE module declares {role} {index} as shape {actual[0]} dtype "
                    f"{actual[1]}; the artifact manifest declares shape {expected[0]} "
                    f"dtype {expected[1]}."
                )


class IREEExecutable:
    """Loaded IREE executable with exact positional input and output ABI checks.

    The executable runs on the host (`capabilities.tier == "compiled-inference"`,
    host-only): `jit`, `vmap`, `grad`, `jvp`, and `vjp` are refused before the
    module is invoked. `binding` is the manifest's artifact binding identity.
    The manifest's input and output ABI must equal the entry point's reflected
    ``iree.abi.declaration``.

    Invocation uses IREE's VM/HAL contract directly (``VmContext.invoke`` with
    HAL buffer views). Each result's HAL element type and shape are compared
    with the manifest before its memory is mapped; the mapped memory is then
    read through the Python buffer protocol into an owned array. The
    ``DeviceArray.to_host`` path is not used: in iree-base-runtime 3.11
    ``MappedMemory.asarray`` never releases its mapping, retaining every
    output buffer for the life of the process.
    """

    capabilities = ExecutionCapabilities("compiled-inference", host_only=True)

    def __init__(
        self,
        manifest: IREEArtifactManifest,
        module_bytes: bytes,
        /,
    ) -> None:
        compiler, runtime = import_iree()
        del compiler
        config = runtime.Config(manifest.runtime_driver)
        context = runtime.SystemContext(config=config)
        module = runtime.VmModule.copy_buffer(context.instance, module_bytes)
        context.add_vm_module(module)
        function = module.lookup_function(manifest.entry_point)
        if function is None:
            raise ValueError(f"IREE module has no entry point {manifest.entry_point!r}.")
        _require_declared_abi(manifest, function.reflection)
        self.manifest = manifest
        self.binding = manifest.binding_identity()
        self._runtime = runtime
        self._device = config.device
        self._context = context
        self._module = module
        self._entry = function
        self._output_element_types = tuple(
            frozenset(
                int(getattr(runtime.HalElementType, name))
                for name in _HAL_ELEMENT_TYPES[dtype]
            )
            for dtype in manifest.output_dtypes
        )

    def _run(self, inputs: Sequence[np.ndarray], /) -> tuple[np.ndarray, ...]:
        """Invoke the entry point and return owned host copies of its results."""

        runtime = self._runtime
        interop = importlib.import_module(f"{runtime.__name__}.array_interop")
        arguments = runtime.VmVariantList(len(inputs))
        for array in inputs:
            arguments.push_ref(
                self._device.allocator.allocate_buffer_copy(
                    memory_type=runtime.MemoryType.DEVICE_LOCAL,
                    allowed_usage=runtime.BufferUsage.DEFAULT
                    | runtime.BufferUsage.MAPPING,
                    device=self._device,
                    buffer=np.require(array, requirements="C"),
                    element_type=interop.map_dtype_to_element_type(array.dtype),
                )
            )
        count = len(self.manifest.output_names)
        results = runtime.VmVariantList(count)
        self._context.vm_context.invoke(self._entry, arguments, results)
        if results.size != count:
            raise RuntimeError(
                f"IREE output arity differs from the artifact manifest: expected "
                f"{count}; got {results.size}."
            )
        outputs: list[np.ndarray] = []
        for index, (name, shape, dtype, admitted) in enumerate(
            zip(
                self.manifest.output_names,
                self.manifest.output_shapes,
                self.manifest.output_dtypes,
                self._output_element_types,
                strict=True,
            )
        ):
            view = results.get_as_object(index, runtime.HalBufferView)
            label = f"IREE output {index} ({name!r})"
            element_type = int(view.element_type)
            if element_type not in admitted:
                try:
                    observed = runtime.HalElementType(element_type).name
                except ValueError:
                    observed = str(element_type)
                raise RuntimeError(
                    f"{label} HAL element type {observed} differs from the artifact "
                    f"manifest dtype {dtype}."
                )
            observed_shape = tuple(int(extent) for extent in view.shape)
            if observed_shape != shape:
                raise RuntimeError(
                    f"{label} shape differs from the artifact manifest: expected "
                    f"{shape}; got {observed_shape}."
                )
            expected = np.dtype(dtype)
            boolean = expected == np.bool_
            stored = np.dtype(np.uint8) if boolean else expected
            if view.byte_length != int(np.prod(shape, dtype=np.int64)) * stored.itemsize:
                raise RuntimeError(
                    f"{label} byte length differs from its {dtype} shape {shape}."
                )
            mapped = view.map()
            with memoryview(mapped) as memory:
                values = np.frombuffer(memory, dtype=stored).reshape(shape).copy()
            del mapped
            if boolean:
                if np.any(values > 1):
                    raise RuntimeError(
                        f"{label} stores a boolean byte other than 0 or 1."
                    )
                values = values.view(np.bool_)
            outputs.append(values)
        return tuple(outputs)

    def __call__(self, *args: Any) -> np.ndarray | tuple[np.ndarray, ...]:
        _require_execution(self.capabilities, args)
        if len(args) != len(self.manifest.input_shapes):
            raise ValueError(
                f"IREE executable expected {len(self.manifest.input_shapes)} inputs; got {len(args)}."
            )
        prepared = []
        for index, (argument, shape, dtype) in enumerate(
            zip(
                args,
                self.manifest.input_shapes,
                self.manifest.input_dtypes,
                strict=True,
            )
        ):
            array = np.asarray(argument)
            if tuple(array.shape) != shape:
                raise ValueError(
                    f"IREE input {index} must have shape {shape}; got {array.shape}."
                )
            if array.dtype.str != dtype:
                raise TypeError(
                    f"IREE input {index} must have dtype {dtype}; got {array.dtype.str}."
                )
            prepared.append(array)

        raw_outputs = self._run(prepared)
        for index, (output, name) in enumerate(
            zip(raw_outputs, self.manifest.output_names, strict=True)
        ):
            if not np.all(np.isfinite(output)):
                raise RuntimeError(
                    f"IREE output {index} ({name!r}) contains non-finite values."
                )
        return raw_outputs[0] if len(raw_outputs) == 1 else raw_outputs


def load_iree(
    path: str | Path,
    /,
    *,
    trusted_module_sha256: str,
) -> IREEExecutable:
    """Load a descriptor-admitted IREE module authorized by an out-of-band pin."""

    if (
        not isinstance(trusted_module_sha256, str)
        or len(trusted_module_sha256) != 64
        or any(character not in "0123456789abcdef" for character in trusted_module_sha256)
    ):
        raise ValueError("trusted_module_sha256 must be a lowercase SHA-256 digest.")
    source = Path(path).expanduser().absolute()
    with open_bounded_resource_set(
        source.name,
        trusted_root=source.parent,
        limits=ResourceSetLimits(
            max_total_bytes=4 * 1024 * 1024 * 1024,
            max_member_bytes=4 * 1024 * 1024 * 1024,
            max_members=2,
            max_depth=1,
        ),
    ) as bundle:
        member_names = {member.relative_path for member in bundle.manifest.members}
        if "manifest.json" not in member_names:
            raise ValueError("IREE artifact manifest is missing.")
        manifest_bytes = bundle.read_member(
            "manifest.json", maximum_bytes=16 * 1024 * 1024
        )
        manifest_resource = bounded_resource_from_bytes(
            manifest_bytes,
            limits=ResourceLimits(16 * 1024 * 1024, 64, 100_000, 100_000, 0),
            source_path="manifest.json",
        )
        value = decode_json_resource(manifest_resource).value
        if not isinstance(value, Mapping):
            raise TypeError("IREE manifest JSON must contain an object.")
        manifest = IREEArtifactManifest.from_dict(value)
        if member_names != {"manifest.json", manifest.module_file}:
            raise ValueError("IREE artifact bundle inventory changed.")
        module_bytes = bundle.read_member(manifest.module_file)
    observed_module_sha256 = _sha256_bytes(module_bytes)
    if observed_module_sha256 != manifest.module_sha256:
        raise ValueError("IREE module checksum mismatch.")
    if observed_module_sha256 != trusted_module_sha256:
        raise PermissionError(
            "IREE module is not authorized by the caller-supplied trusted pin."
        )
    availability = iree_availability()
    availability.require("compiled-inference")
    versions = dict(availability.versions)
    if (
        versions["iree-base-compiler"] != manifest.compiler_version
        or versions["iree-base-runtime"] != manifest.runtime_version
    ):
        raise ValueError(
            "IREE artifact compiler/runtime versions differ from the runtime."
        )
    if manifest.executable_platform != _executable_platform(manifest.executable_format):
        raise ValueError(
            f"IREE {manifest.executable_format} executable was compiled for "
            f"{manifest.executable_platform}, not this host."
        )
    return IREEExecutable(manifest, module_bytes)


def save_iree(
    function: Callable[..., Any] | DomainFunction,
    path: str | Path,
    /,
    *,
    inputs: Sequence[Any],
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    policy: IREEExportPolicy | None = None,
    key: Any = None,
    preprocess: Callable[..., Any] | None = None,
    postprocess: Callable[..., Any] | None = None,
    vectorize: bool = False,
    validate: bool = True,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
    domain_contract: Mapping[str, Any] | None = None,
) -> IREEExportResult:
    """Compile and publish a deterministic array or ordered array-tuple boundary.

    ``domain_contract`` is an optional JSON object published canonically in the
    manifest and bound into the artifact identity, for domain owners whose ABI
    carries semantics beyond array names, shapes, and dtypes.

    The body is one publication transaction on purpose: export, legalized
    compile, manifest identity (including executable format/platform/linker),
    native-parity validation, and atomic publication share the same module
    bytes and provisional manifest, so nothing is published unless every
    earlier step succeeded on exactly those bytes.
    """

    contract_json = (
        None
        if domain_contract is None
        else json.dumps(
            dict(domain_contract),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    if key is not None:
        raise ValueError("IREE export requires key=None for deterministic inference.")
    policy_ = IREEExportPolicy() if policy is None else policy
    if not isinstance(policy_, IREEExportPolicy):
        raise TypeError("policy must be IREEExportPolicy or None.")
    input_arrays = tuple(jnp.asarray(value) for value in inputs)
    if not input_arrays:
        raise ValueError("IREE export requires at least one concrete input array.")
    names = (
        tuple(f"input_{index}" for index in range(len(input_arrays)))
        if input_names is None
        else tuple(str(name) for name in input_names)
    )
    if len(names) != len(input_arrays) or any(not name for name in names):
        raise ValueError("input_names must name every input exactly once.")
    if len(set(names)) != len(names):
        raise ValueError("input_names must be unique.")

    export_function = make_inference_export_callable(
        function,
        key=None,
        preprocess=preprocess,
        postprocess=postprocess,
        vectorize=bool(vectorize),
    )
    exported = jax.export.export(jax.jit(export_function))(*input_arrays)
    output_avals = exported.out_avals
    output_count = len(output_avals)
    single_array_tree = jax.tree.structure(0)
    tuple_array_tree = jax.tree.structure((0,) * output_count)
    if output_count == 0 or exported.out_tree not in (
        single_array_tree,
        tuple_array_tree,
    ):
        raise TypeError(
            "IREE export requires one array output or a non-empty tuple of arrays."
        )
    output_names_ = (
        tuple(f"output_{index}" for index in range(output_count))
        if output_names is None
        else tuple(str(name) for name in output_names)
    )
    if len(output_names_) != output_count or any(not name for name in output_names_):
        raise ValueError("output_names must name every output exactly once.")
    if len(set(output_names_)) != len(output_names_):
        raise ValueError("output_names must be unique.")

    compiler, _ = import_iree()
    module_bytes = bytes(
        compiler.tools.compile_str(
            _legalized_iree_input(exported.mlir_module()),
            target_backends=[policy_.target_backend],
            extra_args=policy_.compiler_arguments,
        )
    )
    module_hash = _sha256_bytes(module_bytes)
    versions = dict(iree_availability().versions)
    compiler_version = versions["iree-base-compiler"]
    runtime_version = versions["iree-base-runtime"]
    metadata = {
        "format": _IREE_ARTIFACT_FORMAT,
        "module_sha256": module_hash,
        "compiler_version": compiler_version,
        "runtime_version": runtime_version,
        "target_backend": policy_.target_backend,
        "runtime_driver": policy_.runtime_driver,
        "executable_format": policy_.executable_format,
        "executable_platform": _executable_platform(policy_.executable_format),
        "system_linker": policy_.system_linker,
        "function_name": exported.fun_name,
        "calling_convention_version": exported.calling_convention_version,
        "input_names": names,
        "input_shapes": tuple(tuple(value.shape) for value in input_arrays),
        "input_dtypes": tuple(np.dtype(value.dtype).str for value in input_arrays),
        "output_names": output_names_,
        "output_shapes": tuple(tuple(output.shape) for output in output_avals),
        "output_dtypes": tuple(np.dtype(output.dtype).str for output in output_avals),
        "vectorized": bool(vectorize),
        "has_preprocess": preprocess is not None,
        "has_postprocess": postprocess is not None,
    }
    if contract_json is not None:
        metadata["domain_contract"] = contract_json
    artifact_id = canonical_fingerprint(metadata)
    provisional = IREEArtifactManifest(
        format=_IREE_ARTIFACT_FORMAT,
        artifact_id=artifact_id,
        module_file=f"module-{module_hash[:16]}.vmfb",
        module_sha256=module_hash,
        compiler_version=compiler_version,
        runtime_version=runtime_version,
        target_backend=policy_.target_backend,
        runtime_driver=policy_.runtime_driver,
        executable_format=policy_.executable_format,
        executable_platform=metadata["executable_platform"],
        system_linker=policy_.system_linker,
        function_name=exported.fun_name,
        entry_point="main",
        calling_convention_version=exported.calling_convention_version,
        input_names=names,
        input_shapes=metadata["input_shapes"],
        input_dtypes=metadata["input_dtypes"],
        output_names=metadata["output_names"],
        output_shapes=metadata["output_shapes"],
        output_dtypes=metadata["output_dtypes"],
        vectorized=bool(vectorize),
        has_preprocess=preprocess is not None,
        has_postprocess=postprocess is not None,
        validation_ok=None,
        maximum_absolute_errors=None,
        maximum_relative_errors=None,
        domain_contract=contract_json,
    )
    executable = IREEExecutable(provisional, module_bytes)
    validation_ok = None
    maximum_absolute_errors = None
    maximum_relative_errors = None
    if validate:
        native_result = export_function(*input_arrays)
        native_outputs = (
            native_result if isinstance(native_result, tuple) else (native_result,)
        )
        if len(native_outputs) != len(output_names_):
            raise RuntimeError(
                f"Native output arity changed after export: expected {len(output_names_)}; got {len(native_outputs)}."
            )
        deployed_result = executable(*(np.asarray(value) for value in input_arrays))
        deployed_outputs = (
            deployed_result if isinstance(deployed_result, tuple) else (deployed_result,)
        )
        absolute_errors: list[float] = []
        relative_errors: list[float] = []
        for index, (name, native_value, deployed) in enumerate(
            zip(output_names_, native_outputs, deployed_outputs, strict=True)
        ):
            native = np.asarray(native_value)
            expected_shape = provisional.output_shapes[index]
            expected_dtype = provisional.output_dtypes[index]
            label = f"IREE output {index} ({name!r})"
            if tuple(native.shape) != expected_shape:
                raise RuntimeError(
                    f"Native {label.lower()} shape changed after export: expected {expected_shape}; got {native.shape}."
                )
            if native.dtype.str != expected_dtype:
                raise RuntimeError(
                    f"Native {label.lower()} dtype changed after export: expected "
                    f"{expected_dtype}; got {native.dtype.str}."
                )
            if not np.all(np.isfinite(native)):
                raise RuntimeError(f"Native {label.lower()} contains non-finite values.")

            if np.issubdtype(native.dtype, np.inexact):
                absolute = np.abs(native - deployed)
                scale = np.maximum(np.abs(native), np.finfo(native.real.dtype).tiny)
                output_ok = bool(
                    np.allclose(native, deployed, rtol=float(rtol), atol=float(atol))
                )
            else:
                native_numeric = native.astype(np.float64)
                deployed_numeric = deployed.astype(np.float64)
                absolute = np.abs(native_numeric - deployed_numeric)
                scale = np.maximum(np.abs(native_numeric), np.finfo(np.float64).tiny)
                output_ok = bool(np.array_equal(native, deployed))
            max_absolute_error = float(np.max(absolute, initial=0.0))
            max_relative_error = float(np.max(absolute / scale, initial=0.0))
            absolute_errors.append(max_absolute_error)
            relative_errors.append(max_relative_error)
            if not output_ok:
                raise RuntimeError(
                    f"{label} failed native parity: max_abs={max_absolute_error:.3e}, max_rel={max_relative_error:.3e}."
                )
        validation_ok = True
        maximum_absolute_errors = tuple(absolute_errors)
        maximum_relative_errors = tuple(relative_errors)
    manifest = replace(
        provisional,
        validation_ok=validation_ok,
        maximum_absolute_errors=maximum_absolute_errors,
        maximum_relative_errors=maximum_relative_errors,
    )
    destination = Path(path)
    publication = publish_resource_set(
        destination,
        {
            manifest.module_file: module_bytes,
            "manifest.json": (
                json.dumps(
                    manifest.to_dict(),
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        },
        limits=ResourceSetLimits(
            max_total_bytes=4 * 1024 * 1024 * 1024,
            max_member_bytes=4 * 1024 * 1024 * 1024,
            max_members=2,
            max_depth=1,
        ),
        mode="atomic_replace",
    )
    return IREEExportResult(destination, manifest, publication)


__all__ = [
    "IREEArtifactManifest",
    "IREEExecutable",
    "IREEExecutableFormat",
    "IREEExportPolicy",
    "IREEExportResult",
    "load_iree",
    "save_iree",
]
