#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Lower real interval jaxprs to the native meshcore interval executor."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, TYPE_CHECKING

import numpy as np
from jax.core import ShapedArray
from jax.extend import core as jax_core
from numpy.typing import NDArray

from .._interpolation._bspline import bspline_jet_rule_alignment
from .._jax_program import custom_jvp_rule_owner
from ._interval_enclosure import (
    _as_interval,
    _CUSTOM_EXTENSIONS,
    _ieee_maximum,
    _ieee_minimum,
    _UNATTRIBUTED_AXIS,
    _ValueIdentity,
    IntervalExtensionError,
    IntervalOperandIdentity,
)


if TYPE_CHECKING:
    from .._meshcore import IntervalBoxProgram


type _Interval = tuple[np.ndarray | np.generic, np.ndarray | np.generic]
type _Shape = tuple[int, ...]
type _Row = tuple[int, int, int, int, int, int, int, int]


_GATHER = 1
_ADD = 2
_SUBTRACT = 3
_MULTIPLY = 5
_NEGATE = 6
_REDUCE_SUM = 7
_SINE = 8
_COSINE = 9
_SQRT = 10
_RECIPROCAL = 11
_SQUARE = 12
_COMPARE = 13
_SELECT = 14
_MAXIMUM = 15
_MINIMUM = 16
_EXP = 17
_LOG = 18
_EQUAL = 19
_ABSOLUTE = 20
_EXTREMUM_PAIR = 21
_EXTREMUM_REDUCE = 22
_DECIDE = 23
_ATTAIN = 24

# Branch-evidence event kinds and tests (phx_mc_interval_program_branches).
_KINK = 0
_JUMP = 1
_MEETS_ZERO = 0
_NOT_POSITIVE = 1
_NOT_NONNEGATIVE = 2
_UNDECIDED = 3
_NONFINITE = 4
_ALWAYS = 5

type _Event = tuple[int, int, int, int, int, int]


def _size(shape: _Shape, /) -> int:
    return math.prod(shape) if shape else 1


def _shape(value: Sequence[int], /) -> _Shape:
    return tuple(int(dimension) for dimension in value)


def _real_shape(variable: jax_core.Var | jax_core.Literal, /) -> _Shape:
    aval = variable.aval
    if not isinstance(aval, ShapedArray) or aval.dtype.kind == "c":
        raise IntervalExtensionError(
            "native interval programs require real shaped values."
        )
    return _shape(aval.shape)


def _immutable(array: NDArray[np.float64], /) -> NDArray[np.float64]:
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class _Tensor:
    shape: _Shape
    offset: int
    locations: NDArray[np.int64] | None = None
    exact: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class _Seed:
    offset: int
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _NativeExtremum:
    """Workspace winner flags of a min/max equation, as the host's ``_Extremum``.

    ``infos[i]`` holds operand ``i``'s flag pairs (lower = decided, upper =
    winner) in the host's winners shape; ``decided`` holds the decisions in
    the equation's result shape, or is None where every element is decided.
    """

    operands: tuple[int, ...]
    infos: tuple[_Tensor, ...]
    decided: _Tensor | None
    axes: tuple[int, ...]
    operand_shape: _Shape
    keepdims: _Shape


@dataclass(frozen=True, slots=True)
class NativeIntervalProgram:
    """Immutable native program and its fixed-shape interval workspace contract."""

    instructions: NDArray[np.int64]
    indices: NDArray[np.int64]
    parameters: NDArray[np.float64]
    workspace_size: int
    batch_capacity: int
    input_dimension: int
    seeds: tuple[_Seed, ...]
    opcodes: frozenset[int]
    output: _Tensor
    domain_guards: NDArray[np.int64]
    """Workspace entries (real-power bases) that must be nonnegative."""
    events: NDArray[np.int64] | None = None
    """Branch-evidence event rows ``(rows, 6)``, present when lowered with box axes."""

    def _workspaces(
        self, lower: NDArray[np.float64], upper: NDArray[np.float64], /
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        expected_shape = (self.batch_capacity, self.input_dimension)
        if lower.shape != expected_shape or upper.shape != expected_shape:
            raise ValueError(
                f"native interval inputs must have shape {expected_shape}; got {lower.shape} and {upper.shape}."
            )
        lower_workspace = np.empty((self.workspace_size,), dtype=np.float64)
        upper_workspace = np.empty_like(lower_workspace)
        input_size = self.batch_capacity * self.input_dimension
        lower_workspace[:input_size] = lower.reshape((-1,))
        upper_workspace[:input_size] = upper.reshape((-1,))
        for seed in self.seeds:
            size = seed.lower.size
            lower_workspace[seed.offset : seed.offset + size] = seed.lower
            upper_workspace[seed.offset : seed.offset + size] = seed.upper
        return lower_workspace, upper_workspace

    def _output(
        self,
        lower_workspace: NDArray[np.float64],
        upper_workspace: NDArray[np.float64],
        /,
    ) -> _Interval:
        if np.any(lower_workspace[self.domain_guards] < 0.0):
            raise IntervalExtensionError(
                "noninteger power requires a nonnegative base interval."
            )
        if self.output.locations is None:
            start = self.output.offset
            stop = start + _size(self.output.shape)
            output_lower = lower_workspace[start:stop]
            output_upper = upper_workspace[start:stop]
        else:
            output_lower = lower_workspace[self.output.locations]
            output_upper = upper_workspace[self.output.locations]
        return (
            output_lower.reshape(self.output.shape),
            output_upper.reshape(self.output.shape),
        )

    def evaluate(
        self, lower: NDArray[np.float64], upper: NDArray[np.float64], /
    ) -> _Interval:
        lower_workspace, upper_workspace = self._workspaces(lower, upper)
        from .._meshcore import interval_program_execute

        interval_program_execute(
            self.instructions,
            self.indices,
            self.parameters,
            lower_workspace,
            upper_workspace,
        )
        return self._output(lower_workspace, upper_workspace)

    def evaluate_branches(
        self, lower: NDArray[np.float64], upper: NDArray[np.float64], /
    ) -> tuple[
        np.ndarray | np.generic,
        np.ndarray | np.generic,
        NDArray[np.bool_],
        NDArray[np.bool_],
    ]:
        """Enclosures and per-box ``(smooth, continuous)`` flags of one batch."""
        if self.events is None:
            raise IntervalExtensionError(
                "native branch evidence requires a program lowered with box axes."
            )
        lower_workspace, upper_workspace = self._workspaces(lower, upper)
        from .._meshcore import interval_program_branches

        _, _, smooth, continuous = interval_program_branches(
            self.instructions,
            self.indices,
            self.parameters,
            lower_workspace,
            upper_workspace,
            self.events,
            self.batch_capacity,
        )
        output_lower, output_upper = self._output(lower_workspace, upper_workspace)
        return output_lower, output_upper, smooth, continuous

    def require_supported_opcodes(self, supported_opcodes: frozenset[int], /) -> None:
        missing_opcodes = self.opcodes.difference(supported_opcodes)
        if missing_opcodes:
            raise IntervalExtensionError(
                "meshcore does not advertise native interval opcodes "
                f"{tuple(sorted(missing_opcodes))}."
            )

    def box_program(self, /) -> IntervalBoxProgram:
        """Return the one-box native form: seeded workspace template and output entries."""
        if self.batch_capacity != 1 or self.input_dimension != 3:
            raise ValueError(
                "a native box program requires batch capacity 1 and three inputs."
            )
        lower = np.zeros((self.workspace_size,), dtype=np.float64)
        upper = np.zeros_like(lower)
        for seed in self.seeds:
            size = seed.lower.size
            lower[seed.offset : seed.offset + size] = seed.lower
            upper[seed.offset : seed.offset + size] = seed.upper
        if self.output.locations is None:
            outputs = np.arange(
                self.output.offset,
                self.output.offset + _size(self.output.shape),
                dtype=np.int64,
            )
        else:
            outputs = np.asarray(self.output.locations, dtype=np.int64).reshape((-1,))
        from .._meshcore import IntervalBoxProgram

        return IntervalBoxProgram(
            self.instructions,
            self.indices,
            self.parameters,
            lower,
            upper,
            outputs,
            self.domain_guards,
        )


class _Builder:
    def __init__(
        self,
        batch_capacity: int,
        input_dimension: int,
        branch_axes: Mapping[jax_core.Var, int | None] | None = None,
        /,
    ) -> None:
        self.batch_capacity = batch_capacity
        self.input_dimension = input_dimension
        self.next_offset = batch_capacity * input_dimension
        self.rows: list[_Row] = []
        self.indices: list[int] = []
        self.parameters: list[float] = []
        self.seeds: list[_Seed] = []
        self.domain_guards: list[int] = []
        # Value identities and decided selections, as the host's _ValueIdentity.
        self.values = _ValueIdentity()
        self.stacks: dict[jax_core.Var, tuple[int, tuple[int, ...]]] = {}
        self.extrema: dict[jax_core.Var, _NativeExtremum] = {}
        self.reductions: dict[jax_core.Var, _NativeExtremum] = {}
        self.attainments: dict[jax_core.Var, _NativeExtremum] = {}
        # Box axis of every traced variable; events are recorded only with it.
        self.branch_axes = branch_axes
        self.events: list[_Event] = []

    def _reserve(self, shape: _Shape, /) -> int:
        offset = self.next_offset
        self.next_offset += _size(shape)
        return offset

    def _seed(
        self,
        lower: object,
        upper: object,
        /,
        *,
        exact: object | None = None,
    ) -> _Tensor:
        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        if lower_.shape != upper_.shape:
            raise IntervalExtensionError("native interval seed endpoints must align.")
        if np.any(np.isnan(lower_)) or np.any(np.isnan(upper_)):
            raise IntervalExtensionError("native interval seeds cannot contain NaN.")
        offset = self._reserve(tuple(lower_.shape))
        lower_flat = _immutable(lower_.reshape((-1,)))
        upper_flat = _immutable(upper_.reshape((-1,)))
        self.seeds.append(_Seed(offset, lower_flat, upper_flat))
        exact_ = None
        if exact is not None:
            exact_array = np.asarray(exact, dtype=np.float64)
            if exact_array.shape != lower_.shape:
                raise IntervalExtensionError("native interval exact seeds must align.")
            exact_ = _immutable(exact_array)
        return _Tensor(tuple(lower_.shape), offset, exact=exact_)

    def _locations(self, value: _Tensor, /) -> NDArray[np.int64]:
        if value.locations is not None:
            return value.locations.reshape(value.shape)
        return np.arange(
            value.offset, value.offset + _size(value.shape), dtype=np.int64
        ).reshape(value.shape)

    def _mapped(
        self,
        value: _Tensor,
        target_shape: _Shape,
        /,
        dimensions: tuple[int, ...] | None = None,
    ) -> _Tensor:
        if dimensions is None:
            if np.broadcast_shapes(value.shape, target_shape) != target_shape:
                raise IntervalExtensionError(
                    f"native interval shapes {value.shape} and {target_shape} do not broadcast."
                )
            if value.shape == target_shape:
                return value
            shift = len(target_shape) - len(value.shape)
            reshape = (1,) * shift + value.shape
            locations = np.broadcast_to(
                self._locations(value).reshape(reshape), target_shape
            ).reshape((-1,))
            exact = (
                None
                if value.exact is None
                else np.broadcast_to(value.exact.reshape(reshape), target_shape)
            )
            return _Tensor(target_shape, value.offset, locations, exact)
        if len(dimensions) != len(value.shape) or any(
            axis < 0 or axis >= len(target_shape) for axis in dimensions
        ):
            raise IntervalExtensionError("native broadcast dimensions are invalid.")
        reshape = [1] * len(target_shape)
        for source_axis, target_axis in enumerate(dimensions):
            reshape[target_axis] = value.shape[source_axis]
        if any(
            value.shape[source_axis] != 1
            and value.shape[source_axis] != target_shape[target_axis]
            for source_axis, target_axis in enumerate(dimensions)
        ):
            raise IntervalExtensionError("native broadcast extents do not align.")
        locations = np.broadcast_to(
            self._locations(value).reshape(tuple(reshape)), target_shape
        ).reshape((-1,))
        exact = (
            None
            if value.exact is None
            else np.broadcast_to(value.exact.reshape(tuple(reshape)), target_shape)
        )
        return _Tensor(target_shape, value.offset, locations, exact)

    def _operand(self, value: _Tensor, target_shape: _Shape, /) -> tuple[int, int]:
        mapped = self._mapped(value, target_shape)
        if mapped.locations is None:
            return mapped.offset, -1
        locations = mapped.locations.reshape((-1,))
        if locations.size == 0:
            return mapped.offset, -1
        if locations.size == 1 or np.all(np.diff(locations) == 1):
            return int(locations[0]), -1
        index = len(self.indices)
        self.indices.extend(int(location) for location in locations)
        return 0, index

    def _allocate(self, shape: _Shape, /) -> _Tensor:
        return _Tensor(shape, self._reserve(shape))

    def _emit_unary(
        self, opcode: int, value: _Tensor, output_shape: _Shape, /
    ) -> _Tensor:
        operand = self._mapped(value, output_shape)
        output = self._allocate(output_shape)
        a, a_index = self._operand(operand, output_shape)
        self.rows.append(
            (opcode, output.offset, _size(output_shape), a, a_index, -1, -1, 0)
        )
        return output

    def _emit_binary(
        self,
        opcode: int,
        first: _Tensor,
        second: _Tensor,
        output_shape: _Shape,
        /,
        *,
        aux: int = 0,
    ) -> _Tensor:
        left = self._mapped(first, output_shape)
        right = self._mapped(second, output_shape)
        output = self._allocate(output_shape)
        a, a_index = self._operand(left, output_shape)
        b, b_index = self._operand(right, output_shape)
        self.rows.append(
            (opcode, output.offset, _size(output_shape), a, a_index, b, b_index, aux)
        )
        return output

    def _divide_one(self, denominator: _Tensor, output_shape: _Shape, /) -> _Tensor:
        reciprocal = self._emit_unary(_RECIPROCAL, denominator, output_shape)
        ones = np.ones(output_shape, dtype=np.float64)
        one = self._seed(ones, ones, exact=ones)
        return self._emit_binary(_MULTIPLY, one, reciprocal, output_shape)

    def _materialize(self, value: _Tensor, /) -> _Tensor:
        if value.locations is None:
            return value
        output = self._allocate(value.shape)
        a, a_index = self._operand(value, value.shape)
        self.rows.append(
            (_GATHER, output.offset, _size(value.shape), a, a_index, -1, -1, 0)
        )
        return output

    def _transpose(self, value: _Tensor, axes: tuple[int, ...], /) -> _Tensor:
        if len(axes) != len(value.shape) or tuple(sorted(axes)) != tuple(
            range(len(value.shape))
        ):
            raise IntervalExtensionError("native transpose axes are invalid.")
        shape = tuple(value.shape[axis] for axis in axes)
        locations = self._locations(value).transpose(axes).reshape((-1,))
        exact = None if value.exact is None else value.exact.transpose(axes)
        return _Tensor(shape, value.offset, locations, exact)

    def _dot_general(
        self,
        left: _Tensor,
        right: _Tensor,
        params: Mapping[str, Any],
        output_shape: _Shape,
        /,
    ) -> _Tensor:
        """Contract as an elementwise product over (batch, free, free, contract)."""
        (left_contract, right_contract), (left_batch, right_batch) = params[
            "dimension_numbers"
        ]
        left_contract = tuple(int(axis) for axis in left_contract)
        right_contract = tuple(int(axis) for axis in right_contract)
        left_batch = tuple(int(axis) for axis in left_batch)
        right_batch = tuple(int(axis) for axis in right_batch)
        left_free = tuple(
            axis
            for axis in range(len(left.shape))
            if axis not in left_contract and axis not in left_batch
        )
        right_free = tuple(
            axis
            for axis in range(len(right.shape))
            if axis not in right_contract and axis not in right_batch
        )
        batch_shape = tuple(left.shape[axis] for axis in left_batch)
        left_free_shape = tuple(left.shape[axis] for axis in left_free)
        right_free_shape = tuple(right.shape[axis] for axis in right_free)
        contract_shape = tuple(left.shape[axis] for axis in left_contract)
        if (
            batch_shape != tuple(right.shape[axis] for axis in right_batch)
            or contract_shape != tuple(right.shape[axis] for axis in right_contract)
            or batch_shape + left_free_shape + right_free_shape != output_shape
        ):
            raise IntervalExtensionError("native dot_general shapes do not align.")
        product_shape = batch_shape + left_free_shape + right_free_shape + contract_shape
        batch_count, left_count = len(batch_shape), len(left_free_shape)
        right_start = batch_count + left_count
        contract_start = right_start + len(right_free_shape)
        ordered_left = self._transpose(left, left_batch + left_free + left_contract)
        ordered_right = self._transpose(right, right_batch + right_free + right_contract)
        left_mapped = self._mapped(
            ordered_left,
            product_shape,
            tuple(range(right_start)) + tuple(range(contract_start, len(product_shape))),
        )
        right_mapped = self._mapped(
            ordered_right,
            product_shape,
            tuple(range(batch_count)) + tuple(range(right_start, len(product_shape))),
        )
        product = self._emit_binary(_MULTIPLY, left_mapped, right_mapped, product_shape)
        if not contract_shape:
            return product
        return self._reduce_sum(
            product, tuple(range(contract_start, len(product_shape))), output_shape
        )

    def _slice(
        self, value: _Tensor, slices: tuple[slice, ...], output_shape: _Shape, /
    ) -> _Tensor:
        if len(slices) != len(value.shape):
            raise IntervalExtensionError("native slice rank does not match its input.")
        mapped_locations = self._locations(value)[slices]
        if tuple(mapped_locations.shape) != output_shape:
            raise IntervalExtensionError("native slice output shape is inconsistent.")
        locations = mapped_locations.reshape((-1,))
        exact = None if value.exact is None else value.exact[slices]
        return _Tensor(output_shape, value.offset, locations, exact)

    def _reduce_sum(
        self, value: _Tensor, axes: tuple[int, ...], output_shape: _Shape, /
    ) -> _Tensor:
        current = value
        rank = len(value.shape)
        if axes and rank == 0:
            raise IntervalExtensionError(
                "native reduction axes require a non-scalar input."
            )
        if any(axis < -rank or axis >= rank for axis in axes):
            raise IntervalExtensionError("native reduction axis is invalid.")
        normalized_axes = tuple(axis + rank if axis < 0 else axis for axis in axes)
        if len(set(normalized_axes)) != len(normalized_axes):
            raise IntervalExtensionError("native reduction axes must be unique.")
        normalized = tuple(sorted(normalized_axes, reverse=True))
        for axis in normalized:
            result_shape = current.shape[:axis] + current.shape[axis + 1 :]
            terms = current.shape[axis]
            if terms == 0:
                zeros = np.zeros(result_shape, dtype=np.float64)
                current = self._seed(zeros, zeros, exact=zeros)
                continue
            permutation = tuple(
                dimension for dimension in range(len(current.shape)) if dimension != axis
            ) + (axis,)
            ordered = (
                current
                if permutation == tuple(range(len(current.shape)))
                else self._transpose(current, permutation)
            )
            ordered = self._materialize(ordered)
            gamma_product = terms * np.finfo(np.float64).eps
            if gamma_product >= 1.0:
                raise IntervalExtensionError(
                    "native interval reduction exceeds the binary64 rounding bound."
                )
            gamma = gamma_product / (1.0 - gamma_product)
            parameter = len(self.parameters)
            self.parameters.append(gamma)
            output = self._allocate(result_shape)
            self.rows.append(
                (
                    _REDUCE_SUM,
                    output.offset,
                    _size(result_shape),
                    ordered.offset,
                    -1,
                    -1,
                    parameter,
                    terms,
                )
            )
            current = output
        if current.shape != output_shape:
            raise IntervalExtensionError(
                "native interval reduction produced an inconsistent shape."
            )
        return current

    def _exact_compare(
        self,
        name: str,
        first: _Tensor,
        second: _Tensor,
        output_shape: _Shape,
        /,
    ) -> _Tensor:
        if first.exact is None or second.exact is None:
            raise IntervalExtensionError(
                f"native interval primitive {name!r} requires constant operands."
            )
        left = np.broadcast_to(first.exact, output_shape)
        right = np.broadcast_to(second.exact, output_shape)
        match name:
            case "eq":
                exact = np.equal(left, right)
            case "ne":
                exact = np.not_equal(left, right)
            case "lt":
                exact = np.less(left, right)
            case "le":
                exact = np.less_equal(left, right)
            case "gt":
                exact = np.greater(left, right)
            case "ge":
                exact = np.greater_equal(left, right)
            case _:
                raise IntervalExtensionError(
                    f"unsupported native comparison primitive {name!r}."
                )
        exact_float = np.asarray(exact, dtype=np.float64)
        return self._seed(exact_float, exact_float, exact=exact_float)

    def _interval_compare(
        self,
        name: str,
        first: _Tensor,
        second: _Tensor,
        output_shape: _Shape,
        /,
    ) -> _Tensor:
        """Three-valued comparison as a [0, 1] indicator interval, as the host's."""
        opcode = _COMPARE
        match name:
            case "gt":
                left, right, aux = first, second, 0
            case "ge":
                left, right, aux = first, second, 1
            case "lt":
                left, right, aux = second, first, 0
            case "le":
                left, right, aux = second, first, 1
            case "eq":
                left, right, aux, opcode = first, second, 0, _EQUAL
            case "ne":
                left, right, aux, opcode = first, second, 1, _EQUAL
            case _:
                raise IntervalExtensionError(
                    f"unsupported native comparison primitive {name!r}."
                )
        return self._emit_binary(opcode, left, right, output_shape, aux=aux)

    def _select(
        self,
        predicate: _Tensor,
        cases: tuple[_Tensor, ...],
        output_shape: _Shape,
        /,
    ) -> _Tensor:
        """select_n over two branches: the decided branch exactly, else their hull."""
        if len(cases) == 1:
            return self._mapped(cases[0], output_shape)
        if len(cases) != 2:
            raise IntervalExtensionError(
                "native interval selection supports a boolean predicate over two cases."
            )
        first = self._mapped(cases[0], output_shape)
        second = self._mapped(cases[1], output_shape)
        mapped = self._mapped(predicate, output_shape)
        if mapped.exact is not None:
            chosen = np.broadcast_to(mapped.exact, output_shape) != 0.0
            locations = np.where(
                chosen, self._locations(second), self._locations(first)
            ).reshape((-1,))
            exact = (
                None
                if first.exact is None or second.exact is None
                else np.where(
                    chosen,
                    np.broadcast_to(second.exact, output_shape),
                    np.broadcast_to(first.exact, output_shape),
                )
            )
            offset = int(locations[0]) if locations.size else 0
            return _Tensor(output_shape, offset, locations, exact)
        size = _size(output_shape)
        block = self._reserve((2 * size,))
        for half, case in enumerate((first, second)):
            a, a_index = self._operand(case, output_shape)
            self.rows.append((_GATHER, block + half * size, size, a, a_index, -1, -1, 0))
        output = self._allocate(output_shape)
        a, a_index = self._operand(mapped, output_shape)
        self.rows.append((_SELECT, output.offset, size, a, a_index, block, -1, size))
        return output

    def _fold_exact(
        self, name: str, values: tuple[_Tensor, ...], output_shape: _Shape, /
    ) -> _Tensor | None:
        """Fold constant arithmetic whose binary64 result is the exact real value.

        As the host interpreter rounds only inexact endpoints, an exactly
        representable constant (e.g. ``0.5 * blockiness``) stays a point, so a
        derived integer exponent keeps its integer-power lowering.
        """
        if name not in ("add", "add_any", "sub", "mul", "div", "neg") or any(
            value.exact is None for value in values
        ):
            return None
        operands = [
            np.broadcast_to(np.asarray(value.exact, dtype=np.float64), output_shape)
            for value in values
        ]
        if not all(np.all(np.isfinite(operand)) for operand in operands):
            return None
        match name:
            case "add" | "add_any":
                result = operands[0] + operands[1]
            case "sub":
                result = operands[0] - operands[1]
            case "mul":
                result = operands[0] * operands[1]
            case "div":
                if np.any(operands[1] == 0.0):
                    return None
                result = operands[0] / operands[1]
            case _:
                result = -operands[0]
        result = np.asarray(result, dtype=np.float64)
        for index in np.ndindex(output_shape):
            terms = [Fraction(float(operand[index])) for operand in operands]
            match name:
                case "add" | "add_any":
                    exact = terms[0] + terms[1]
                case "sub":
                    exact = terms[0] - terms[1]
                case "mul":
                    exact = terms[0] * terms[1]
                case "div":
                    exact = terms[0] / terms[1]
                case _:
                    exact = -terms[0]
            if not np.isfinite(result[index]) or Fraction(float(result[index])) != exact:
                return None
        return self._seed(result, result, exact=result)

    def _lower_arithmetic(
        self,
        name: str,
        values: tuple[_Tensor, ...],
        params: Mapping[str, Any],
        output_shape: _Shape,
        same_input: bool,
        /,
    ) -> tuple[_Tensor, ...]:
        folded = self._fold_exact(name, values, output_shape)
        if folded is not None:
            return (folded,)
        if name in ("add", "add_any"):
            for zero, other in ((values[0], values[1]), (values[1], values[0])):
                if zero.exact is not None and np.all(zero.exact == 0.0):
                    return (self._mapped(other, output_shape),)
            return (self._emit_binary(_ADD, values[0], values[1], output_shape),)
        if name == "sub":
            return (self._emit_binary(_SUBTRACT, values[0], values[1], output_shape),)
        if name == "mul":
            if same_input:
                return (self._emit_unary(_SQUARE, values[0], output_shape),)
            return (self._emit_binary(_MULTIPLY, values[0], values[1], output_shape),)
        if name == "div":
            denominator = self._mapped(values[1], output_shape)
            reciprocal = self._emit_unary(_RECIPROCAL, denominator, output_shape)
            return (self._emit_binary(_MULTIPLY, values[0], reciprocal, output_shape),)
        if name in ("neg", "abs", "square", "sqrt", "sin", "cos"):
            opcode = {
                "neg": _NEGATE,
                "square": _SQUARE,
                "sqrt": _SQRT,
                "sin": _SINE,
                "cos": _COSINE,
                "abs": _ABSOLUTE,
            }[name]
            return (self._emit_unary(opcode, values[0], output_shape),)
        if name == "rsqrt":
            root = self._emit_unary(_SQRT, values[0], output_shape)
            return (self._divide_one(root, output_shape),)
        if name == "integer_pow":
            exponent = int(params["y"])
            value = self._mapped(values[0], output_shape)
            if exponent == 0:
                ones = np.ones(output_shape, dtype=np.float64)
                return (self._seed(ones, ones, exact=ones),)
            if exponent < 0:
                positive = self._lower_arithmetic(
                    "integer_pow", (value,), {"y": -exponent}, output_shape, False
                )[0]
                return (self._divide_one(positive, output_shape),)
            if exponent == 1:
                return (value,)
            if exponent % 2 == 0:
                half = self._lower_arithmetic(
                    "integer_pow", (value,), {"y": exponent // 2}, output_shape, False
                )[0]
                return (self._emit_unary(_SQUARE, half, output_shape),)
            previous = self._lower_arithmetic(
                "integer_pow", (value,), {"y": exponent - 1}, output_shape, False
            )[0]
            return (self._emit_binary(_MULTIPLY, previous, value, output_shape),)
        if name in ("max", "min"):
            first, second = values
            if first.exact is not None and second.exact is not None:
                # IEEE 754-2019 ordering, as the _MAXIMUM/_MINIMUM opcodes.
                select = _ieee_maximum if name == "max" else _ieee_minimum
                exact = np.asarray(
                    select(
                        np.broadcast_to(first.exact, output_shape),
                        np.broadcast_to(second.exact, output_shape),
                    ),
                    dtype=np.float64,
                )
                return (self._seed(exact, exact, exact=exact),)
            opcode = _MAXIMUM if name == "max" else _MINIMUM
            return (self._emit_binary(opcode, first, second, output_shape),)
        if name == "pow":
            exponent = values[1].exact
            flat = None if exponent is None else exponent.reshape((-1,))
            if (
                flat is not None
                and flat.size
                and np.all(flat == flat[0])
                and np.isfinite(flat[0])
                and float(flat[0]).is_integer()
            ):
                return self._lower_arithmetic(
                    "integer_pow",
                    (values[0],),
                    {"y": int(flat[0])},
                    output_shape,
                    False,
                )
            # As in the host interpreter: exp(y log x) over a base that must be
            # nonnegative (checked after execution), clamped below at 0 since a
            # nonnegative base has a nonnegative real power.
            base = self._materialize(self._mapped(values[0], output_shape))
            self.domain_guards.extend(
                int(location) for location in self._locations(base).reshape((-1,))
            )
            logarithm = self._emit_unary(_LOG, base, output_shape)
            scaled = self._emit_binary(_MULTIPLY, values[1], logarithm, output_shape)
            power = self._emit_unary(_EXP, scaled, output_shape)
            zeros = np.zeros(output_shape, dtype=np.float64)
            zero = self._seed(zeros, zeros, exact=zeros)
            return (self._emit_binary(_MAXIMUM, power, zero, output_shape),)
        if name in ("exp", "log"):
            opcode = _EXP if name == "exp" else _LOG
            return (self._emit_unary(opcode, values[0], output_shape),)
        raise IntervalExtensionError(
            f"no native interval arithmetic lowering for JAX primitive {name!r}."
        )

    def _lower_static(
        self,
        name: str,
        values: tuple[_Tensor, ...],
        params: Mapping[str, Any],
        output_shape: _Shape,
        /,
    ) -> tuple[_Tensor, ...]:
        if name == "iota":
            dimension = int(params["dimension"])
            if dimension < 0 or dimension >= len(output_shape):
                raise IntervalExtensionError("native iota dimension is invalid.")
            base = np.arange(output_shape[dimension], dtype=np.float64)
            reshape = tuple(
                output_shape[axis] if axis == dimension else 1
                for axis in range(len(output_shape))
            )
            exact = np.broadcast_to(base.reshape(reshape), output_shape)
            return (self._seed(exact, exact, exact=exact),)
        if name in ("eq", "ne", "lt", "le", "gt", "ge"):
            if values[0].exact is not None and values[1].exact is not None:
                return (self._exact_compare(name, values[0], values[1], output_shape),)
            return (self._interval_compare(name, values[0], values[1], output_shape),)
        if name == "convert_element_type":
            new_dtype = np.dtype(params["new_dtype"])
            source = values[0]
            if source.exact is not None:
                exact = np.broadcast_to(source.exact, output_shape)
                exact = np.asarray(exact, dtype=np.float64)
                return (self._seed(exact, exact, exact=exact),)
            if new_dtype.kind != "f":
                raise IntervalExtensionError(
                    "native interval conversion requires a constant discrete value."
                )
            return (self._mapped(source, output_shape),)
        if name in ("copy", "stop_gradient", "device_put"):
            return (self._mapped(values[0], output_shape),)
        if name in ("and", "or"):
            if values[0].exact is None or values[1].exact is None:
                raise IntervalExtensionError(
                    f"native logical primitive {name!r} requires constant operands."
                )
            left = np.broadcast_to(values[0].exact, output_shape) != 0.0
            right = np.broadcast_to(values[1].exact, output_shape) != 0.0
            exact = np.asarray(
                np.logical_and(left, right)
                if name == "and"
                else np.logical_or(left, right),
                dtype=np.float64,
            )
            return (self._seed(exact, exact, exact=exact),)
        if name == "not":
            if values[0].exact is None:
                raise IntervalExtensionError(
                    "native logical not requires a constant operand."
                )
            exact = np.asarray(np.logical_not(values[0].exact != 0.0), dtype=np.float64)
            return (self._seed(exact, exact, exact=exact),)
        raise IntervalExtensionError(
            f"no native interval static lowering for JAX primitive {name!r}."
        )

    def _lower_shape_transform(
        self,
        name: str,
        values: tuple[_Tensor, ...],
        params: Mapping[str, Any],
        output_shape: _Shape,
        /,
    ) -> tuple[_Tensor, ...]:
        value = values[0]
        if name == "reshape":
            dimensions = params.get("dimensions")
            if dimensions is not None:
                value = self._transpose(value, tuple(int(axis) for axis in dimensions))
            if _size(value.shape) != _size(output_shape):
                raise IntervalExtensionError("native reshape changes the element count.")
            locations = self._locations(value).reshape((-1,))
            exact = None if value.exact is None else value.exact.reshape(output_shape)
            return (_Tensor(output_shape, value.offset, locations, exact),)
        if name == "transpose":
            axes = tuple(int(axis) for axis in params["permutation"])
            result = self._transpose(value, axes)
            if result.shape != output_shape:
                raise IntervalExtensionError(
                    "native transpose output shape is inconsistent."
                )
            return (result,)
        if name == "squeeze":
            axes = tuple(int(axis) for axis in params["dimensions"])
            locations = np.squeeze(self._locations(value), axis=axes).reshape((-1,))
            exact = None if value.exact is None else np.squeeze(value.exact, axis=axes)
            return (_Tensor(output_shape, value.offset, locations, exact),)
        if name == "broadcast_in_dim":
            dimensions = tuple(int(axis) for axis in params["broadcast_dimensions"])
            return (self._mapped(value, output_shape, dimensions),)
        if name == "slice":
            starts = tuple(int(index) for index in params["start_indices"])
            stops = tuple(int(index) for index in params["limit_indices"])
            strides = params["strides"]
            if strides is None:
                slices = tuple(
                    slice(start, stop) for start, stop in zip(starts, stops, strict=True)
                )
            else:
                slices = tuple(
                    slice(start, stop, int(stride))
                    for start, stop, stride in zip(starts, stops, strides, strict=True)
                )
            return (self._slice(value, slices, output_shape),)
        raise IntervalExtensionError(
            f"no native interval shape lowering for JAX primitive {name!r}."
        )

    def _lower_partition(
        self,
        name: str,
        value: _Tensor,
        params: Mapping[str, Any],
        output_shapes: tuple[_Shape, ...],
        /,
    ) -> tuple[_Tensor, ...]:
        axis = int(params["axis"])
        if axis < 0:
            axis += len(value.shape)
        if axis < 0 or axis >= len(value.shape):
            raise IntervalExtensionError(f"native {name} axis is invalid.")
        if name == "split":
            sizes = tuple(int(size) for size in params["sizes"])
            if sum(sizes) != value.shape[axis] or len(sizes) != len(output_shapes):
                raise IntervalExtensionError("native split extents are inconsistent.")
            cuts = np.cumsum(sizes, dtype=np.int64)[:-1]
            locations = np.split(self._locations(value), cuts, axis=axis)
            exact = (
                None if value.exact is None else np.split(value.exact, cuts, axis=axis)
            )
            outputs = tuple(
                _Tensor(
                    output_shape,
                    value.offset,
                    locations[index].reshape((-1,)),
                    None if exact is None else exact[index],
                )
                for index, output_shape in enumerate(output_shapes)
            )
            if any(
                locations[index].shape != output_shape
                for index, output_shape in enumerate(output_shapes)
            ):
                raise IntervalExtensionError(
                    "native split output shapes are inconsistent."
                )
            return outputs
        locations = tuple(
            np.take(self._locations(value), index, axis=axis).reshape((-1,))
            for index in range(value.shape[axis])
        )
        exact = (
            None
            if value.exact is None
            else tuple(
                np.take(value.exact, index, axis=axis)
                for index in range(value.shape[axis])
            )
        )
        if len(locations) != len(output_shapes):
            raise IntervalExtensionError("native unstack output arity is inconsistent.")
        return tuple(
            _Tensor(
                output_shape,
                value.offset,
                locations[index],
                None if exact is None else exact[index],
            )
            for index, output_shape in enumerate(output_shapes)
        )

    def _lower_join(
        self,
        name: str,
        values: tuple[_Tensor, ...],
        params: Mapping[str, Any],
        output_shape: _Shape,
        /,
    ) -> tuple[_Tensor, ...]:
        if name == "concatenate":
            axis = int(params["dimension"])
            location_values = tuple(
                self._locations(value).reshape(value.shape) for value in values
            )
            locations = np.concatenate(location_values, axis=axis).reshape((-1,))
            exact = (
                None
                if any(value.exact is None for value in values)
                else np.concatenate(
                    tuple(value.exact for value in values if value.exact is not None),
                    axis=axis,
                )
            )
            return (_Tensor(output_shape, self.next_offset, locations, exact),)
        axis = int(params["axis"])
        if axis < 0:
            axis += len(output_shape)
        locations = np.concatenate(
            tuple(
                np.expand_dims(self._locations(value).reshape(value.shape), axis)
                for value in values
            ),
            axis=axis,
        ).reshape((-1,))
        exact = (
            None
            if any(value.exact is None for value in values)
            else np.concatenate(
                tuple(
                    np.expand_dims(value.exact, axis)
                    for value in values
                    if value.exact is not None
                ),
                axis=axis,
            )
        )
        return (_Tensor(output_shape, self.next_offset, locations, exact),)

    def _lower_primitive(
        self,
        name: str,
        values: tuple[_Tensor, ...],
        params: Mapping[str, Any],
        output_shapes: tuple[_Shape, ...],
        same_input: bool,
        /,
    ) -> tuple[_Tensor, ...]:
        if name in (
            "add",
            "add_any",
            "sub",
            "mul",
            "div",
            "neg",
            "square",
            "sqrt",
            "rsqrt",
            "abs",
            "sin",
            "cos",
            "integer_pow",
            "pow",
            "max",
            "exp",
            "log",
            "min",
        ):
            (output_shape,) = output_shapes
            return self._lower_arithmetic(name, values, params, output_shape, same_input)
        if name == "pad":
            (output_shape,) = output_shapes
            value, padding = values
            config = tuple(
                (int(low), int(high), int(interior))
                for low, high, interior in params["padding_config"]
            )
            if any(low < 0 or high < 0 or interior < 0 for low, high, interior in config):
                raise IntervalExtensionError(
                    "native interval pad requires nonnegative padding."
                )
            filler = self._locations(self._mapped(padding, ())).reshape(())
            locations = np.full(output_shape, int(filler), dtype=np.int64)
            selection = tuple(
                slice(
                    low,
                    low + (extent - 1) * (interior + 1) + 1 if extent else low,
                    interior + 1,
                )
                for (low, _high, interior), extent in zip(
                    config, value.shape, strict=True
                )
            )
            locations[selection] = self._locations(value)
            padded = _Tensor(
                output_shape, int(locations.reshape((-1,))[0]), locations.reshape((-1,))
            )
            return (padded,)
        if name in ("reduce_max", "reduce_min"):
            (output_shape,) = output_shapes
            value = values[0]
            axes = tuple(int(axis) for axis in params["axes"])
            kept = tuple(axis for axis in range(len(value.shape)) if axis not in axes)
            locations = self._locations(value).transpose(kept + axes)
            lanes = locations.reshape((_size(output_shape), -1))
            if lanes.shape[1] == 0:
                raise IntervalExtensionError("native interval extremum of an empty axis.")
            opcode = _MAXIMUM if name == "reduce_max" else _MINIMUM
            result = _Tensor(output_shape, int(lanes[0, 0]), lanes[:, 0].copy())
            for lane in range(1, lanes.shape[1]):
                term = _Tensor(output_shape, int(lanes[0, lane]), lanes[:, lane].copy())
                result = self._emit_binary(opcode, result, term, output_shape)
            return (result,)
        if name == "dot_general":
            (output_shape,) = output_shapes
            return (self._dot_general(values[0], values[1], params, output_shape),)
        if name == "reduce_sum":
            (output_shape,) = output_shapes
            axes = tuple(int(axis) for axis in params["axes"])
            return (self._reduce_sum(values[0], axes, output_shape),)
        if name in (
            "iota",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "convert_element_type",
            "copy",
            "stop_gradient",
            "device_put",
            "and",
            "or",
            "not",
        ):
            (output_shape,) = output_shapes
            return self._lower_static(name, values, params, output_shape)
        if name == "select_n":
            (output_shape,) = output_shapes
            return (self._select(values[0], values[1:], output_shape),)
        if name in ("reshape", "transpose", "squeeze", "broadcast_in_dim", "slice"):
            (output_shape,) = output_shapes
            return self._lower_shape_transform(name, values, params, output_shape)
        if name in ("split", "unstack"):
            return self._lower_partition(name, values[0], params, output_shapes)
        if name in ("concatenate", "stack"):
            (output_shape,) = output_shapes
            return self._lower_join(name, values, params, output_shape)
        raise IntervalExtensionError(
            f"no native interval lowering for JAX primitive {name!r}."
        )

    def _indexed(self, locations: NDArray[np.int64], /) -> tuple[int, int]:
        """Operand fields of a flat location list: contiguous, else an index range."""
        flat = locations.reshape((-1,))
        if flat.size == 0:
            return 0, -1
        if flat.size == 1 or np.all(np.diff(flat) == 1):
            return int(flat[0]), -1
        index = len(self.indices)
        self.indices.extend(int(location) for location in flat)
        return 0, index

    def _extremum(
        self, equation: jax_core.JaxprEqn, arguments: tuple[_Tensor, ...], /
    ) -> _NativeExtremum | None:
        """Winner flags of a min/max equation, as ``_ValueIdentity.extremum``."""
        name = equation.primitive.name
        if name in ("min", "max", "minimum", "maximum"):
            first, second = (self.values.atom(atom) for atom in equation.invars)
            shape = _shape(np.broadcast_shapes(arguments[0].shape, arguments[1].shape))
            if first == second:
                ones = np.ones(shape, dtype=np.float64)
                same = self._seed(ones, ones, exact=ones)
                return _NativeExtremum(
                    (first, second), (same, same), None, (), shape, shape
                )
            size = _size(shape)
            block = self._reserve((2 * size,))
            a, a_index = self._operand(self._mapped(arguments[0], shape), shape)
            b, b_index = self._operand(self._mapped(arguments[1], shape), shape)
            aux = 0 if name in ("min", "minimum") else 1
            self.rows.append((_EXTREMUM_PAIR, block, size, a, a_index, b, b_index, aux))
            infos = (_Tensor(shape, block), _Tensor(shape, block + size))
            return _NativeExtremum((first, second), infos, infos[0], (), shape, shape)
        if name not in ("reduce_min", "reduce_max"):
            return None
        axes = tuple(int(axis) for axis in equation.params["axes"])
        source = equation.invars[0]
        stacked = self.stacks.get(source) if isinstance(source, jax_core.Var) else None
        groups = stacked[1] if stacked is not None and axes == (stacked[0],) else None
        value = arguments[0]
        rank = len(value.shape)
        reduced = tuple(sorted(axis % rank for axis in axes))
        kept = tuple(axis for axis in range(rank) if axis not in reduced)
        kept_shape = tuple(value.shape[axis] for axis in kept)
        reduced_shape = tuple(value.shape[axis] for axis in reduced)
        count, lanes = _size(kept_shape), _size(reduced_shape)
        a, a_index = self._indexed(self._locations(value).transpose(kept + reduced))
        representatives = -1
        if groups is not None:
            representatives = len(self.indices)
            self.indices.extend(groups.index(group) for group in groups)
        block = self._reserve((count * lanes,))
        aux = 0 if name == "reduce_min" else 1
        self.rows.append(
            (_EXTREMUM_REDUCE, block, count, a, a_index, lanes, representatives, aux)
        )
        flags = np.arange(block, block + count * lanes, dtype=np.int64)
        order = tuple(int(axis) for axis in np.argsort(np.asarray(kept + reduced)))
        info = flags.reshape(kept_shape + reduced_shape).transpose(order)
        decided = flags[::lanes] if lanes else flags
        keepdims = tuple(
            1 if axis in reduced else extent for axis, extent in enumerate(value.shape)
        )
        return _NativeExtremum(
            (self.values.atom(source),),
            (_Tensor(value.shape, block, info.reshape((-1,))),),
            _Tensor(kept_shape, block, decided.reshape((-1,))),
            axes,
            value.shape,
            keepdims,
        )

    def _record(
        self,
        equation: jax_core.JaxprEqn,
        arguments: tuple[_Tensor, ...],
        outputs: tuple[_Tensor, ...],
        operands: tuple[int, ...],
        /,
    ) -> _NativeExtremum | None:
        """Track stacks, extrema and attainments, as ``_ValueIdentity.record``."""
        name = equation.primitive.name
        extremum = self._extremum(equation, arguments)
        if not equation.outvars:
            return extremum
        result = equation.outvars[0]
        source = equation.invars[0] if equation.invars else None
        if name == "stack" and len(equation.outvars) == 1:
            rank = len(outputs[0].shape)
            self.stacks[result] = (int(equation.params["axis"]) % rank, operands)
        elif extremum is not None and name in ("min", "max", "minimum", "maximum"):
            self.extrema[result] = extremum
        elif extremum is not None and name in ("reduce_min", "reduce_max"):
            self.reductions[result] = extremum
        elif (
            name == "convert_element_type"
            and isinstance(source, jax_core.Var)
            and source in self.attainments
        ):
            self.attainments[result] = self.attainments[source]
        elif name in ("reshape", "broadcast_in_dim") and isinstance(source, jax_core.Var):
            reduced = self.reductions.get(source)
            if reduced is not None:
                operand_shape, keepdims = reduced.operand_shape, reduced.keepdims
                kept = tuple(
                    axis for axis in range(len(operand_shape)) if axis not in reduced.axes
                )
                shape = outputs[0].shape
                if (
                    name == "reshape"
                    and equation.params.get("dimensions") is None
                    and shape == keepdims
                ) or (
                    name == "broadcast_in_dim"
                    and tuple(
                        int(axis) for axis in equation.params["broadcast_dimensions"]
                    )
                    == kept
                    and len(shape) == len(operand_shape)
                    and all(
                        extent in (keepdims[axis], operand_shape[axis])
                        for axis, extent in enumerate(shape)
                    )
                ):
                    self.extrema[result] = reduced
        return extremum

    def _decide_comparison(
        self, equation: jax_core.JaxprEqn, result: _Tensor, /
    ) -> _Tensor:
        """Decide ``eq``/``ne`` of a min/max operand against its result, as the host."""
        name = equation.primitive.name
        if len(equation.invars) != 2:
            return result
        for operand, selected in (equation.invars, tuple(reversed(equation.invars))):
            extremum = (
                self.extrema.get(selected) if isinstance(selected, jax_core.Var) else None
            )
            if extremum is None:
                continue
            identity = self.values.atom(operand)
            if identity not in extremum.operands:
                continue
            info = extremum.infos[extremum.operands.index(identity)]
            if info.shape != result.shape:
                continue
            if name == "eq" and extremum.axes and equation.outvars:
                self.attainments[equation.outvars[0]] = extremum
            result = self._emit_binary(
                _DECIDE, result, info, result.shape, aux=0 if name == "eq" else 1
            )
        return result

    def _bound_attainment(
        self, equation: jax_core.JaxprEqn, result: _Tensor, /
    ) -> _Tensor:
        """A sum of extremum indicators over the reduced axes is at least one."""
        source = equation.invars[0] if equation.invars else None
        extremum = (
            self.attainments.get(source) if isinstance(source, jax_core.Var) else None
        )
        if (
            extremum is None
            or tuple(int(axis) for axis in equation.params["axes"]) != extremum.axes
        ):
            return result
        return self._emit_unary(_ATTAIN, result, result.shape)

    def _event(self, kind: int, test: int, variable: object, value: _Tensor, /) -> None:
        """Record a ledger event of ``variable`` tested on ``value``, as ``_BranchLedger``."""
        if self.branch_axes is None:
            return
        axis: int | None = _UNATTRIBUTED_AXIS
        if isinstance(variable, jax_core.Var):
            axis = self.branch_axes.get(variable, _UNATTRIBUTED_AXIS)
        if axis is None:
            return
        size = _size(value.shape)
        if size == 0:
            return
        aval = variable.aval if isinstance(variable, jax_core.Var) else None
        boxes: NDArray[np.int64] | None = None
        if (
            axis != _UNATTRIBUTED_AXIS
            and isinstance(aval, ShapedArray)
            and axis < aval.ndim
            and aval.shape[axis] == self.batch_capacity
        ):
            target = _shape(aval.shape)
            try:
                aligned = np.broadcast_shapes(value.shape, target) == target
            except ValueError:
                aligned = False
            position = axis - (len(target) - len(value.shape))
            if aligned and position >= 0 and value.shape[position] == self.batch_capacity:
                reshape = tuple(
                    extent if dimension == position else 1
                    for dimension, extent in enumerate(value.shape)
                )
                boxes = np.broadcast_to(
                    np.arange(self.batch_capacity, dtype=np.int64).reshape(reshape),
                    value.shape,
                ).reshape((-1,))
        a, a_index = self._operand(value, value.shape)
        box_index = -1
        if boxes is not None:
            box_index = len(self.indices)
            self.indices.extend(int(box) for box in boxes)
        self.events.append((kind, test, size, a, a_index, box_index))

    def _inspect(
        self,
        equation: jax_core.JaxprEqn,
        arguments: tuple[_Tensor, ...],
        outputs: tuple[_Tensor, ...],
        operands: tuple[int, ...],
        extremum: _NativeExtremum | None,
        /,
    ) -> None:
        """Ledger events of one equation, as ``_BranchLedger.inspect``."""
        name = equation.primitive.name
        result = equation.outvars[0] if equation.outvars else None
        match name:
            case "max" | "maximum" | "min" | "minimum" if extremum is not None:
                if extremum.decided is not None:
                    self._event(_KINK, _NOT_POSITIVE, result, extremum.decided)
            case "abs":
                self._event(_KINK, _MEETS_ZERO, result, arguments[0])
            case "sqrt":
                self._event(_KINK, _NOT_POSITIVE, result, arguments[0])
                self._event(_JUMP, _NOT_NONNEGATIVE, result, arguments[0])
            case "rsqrt" | "log":
                self._event(_JUMP, _NOT_POSITIVE, result, arguments[0])
            case "div":
                self._event(_JUMP, _MEETS_ZERO, result, arguments[1])
            case "integer_pow":
                if int(equation.params["y"]) < 0:
                    self._event(_JUMP, _MEETS_ZERO, result, arguments[0])
            case "pow":
                exponent = arguments[1].exact
                if exponent is None:
                    raise IntervalExtensionError(
                        "native branch evidence requires a constant real-power exponent."
                    )
                integral = (
                    exponent.size > 0
                    and bool(np.all(np.isfinite(exponent)))
                    and bool(np.all(exponent == np.round(exponent)))
                )
                if not integral:
                    self._event(_KINK, _NOT_POSITIVE, result, arguments[0])
                    self._event(_JUMP, _NOT_NONNEGATIVE, result, arguments[0])
                elif np.any(exponent < 0.0):
                    self._event(_JUMP, _MEETS_ZERO, result, arguments[0])
            case "reduce_max" | "reduce_min" if extremum is not None:
                if extremum.decided is not None:
                    self._event(_KINK, _NOT_POSITIVE, result, extremum.decided)
            case "select_n":
                if len(set(operands[1:])) > 1:
                    self._event(_JUMP, _UNDECIDED, equation.invars[0], arguments[0])
            case "stop_gradient":
                self._event(_KINK, _UNDECIDED, result, arguments[0])
            case _:
                pass
        for variable, value in zip(equation.outvars, outputs, strict=True):
            aval = variable.aval
            kind = np.dtype(aval.dtype).kind if isinstance(aval, ShapedArray) else ""
            if kind in "biu":
                # A selection, index or comparison that the box leaves open.
                self._event(_JUMP, _UNDECIDED, variable, value)
            elif kind == "f":
                self._event(_JUMP, _NONFINITE, variable, value)

    def compile_jaxpr(
        self,
        jaxpr: jax_core.Jaxpr,
        constants: tuple[_Tensor, ...],
        constant_intervals: tuple[_Interval, ...],
        inputs: tuple[_Tensor, ...],
        /,
    ) -> tuple[_Tensor, ...]:
        if len(jaxpr.constvars) != len(constants) or len(jaxpr.invars) != len(inputs):
            raise IntervalExtensionError("native interval jaxpr arguments do not align.")
        if any(
            _real_shape(variable) != value.shape
            for variable, value in zip(jaxpr.constvars, constants, strict=True)
        ) or any(
            _real_shape(variable) != value.shape
            for variable, value in zip(jaxpr.invars, inputs, strict=True)
        ):
            raise IntervalExtensionError(
                "native interval jaxpr argument shapes do not align."
            )
        environment: dict[jax_core.Var, _Tensor] = {}
        environment.update(zip(jaxpr.constvars, constants, strict=True))
        environment.update(zip(jaxpr.invars, inputs, strict=True))
        self.values.bind_constants(jaxpr.constvars, constant_intervals)
        identity = IntervalOperandIdentity()
        for equation in jaxpr.eqns:
            identity.record(equation)
            name = equation.primitive.name
            arguments = tuple(
                environment[atom]
                if isinstance(atom, jax_core.Var)
                else self._literal(atom)
                for atom in equation.invars
            )
            nested_outputs: Sequence[object] | None = None
            declared = False
            if name in ("pjit", "jit", "custom_jvp_call"):
                key = "call_jaxpr" if name == "custom_jvp_call" else "jaxpr"
                nested = equation.params.get(key)
                if isinstance(nested, jax_core.ClosedJaxpr):
                    inner = nested.jaxpr
                    inner_intervals = tuple(
                        _as_interval(value) for value in nested.consts
                    )
                elif isinstance(nested, jax_core.Jaxpr):
                    inner, inner_intervals = nested, ()
                else:
                    raise IntervalExtensionError(
                        f"native interval lowering cannot inline primitive {name!r}."
                    )
                # The host evaluates a declared extension instead of the call,
                # so its output is a fresh value with no declared regularity.
                declared = name == "custom_jvp_call" and _declares_extension(equation)
                if not declared:
                    self.values.descend(inner.invars, equation.invars)
                    nested_outputs = inner.outvars
                outputs = self.compile_jaxpr(
                    inner,
                    tuple(self._constant(*interval) for interval in inner_intervals),
                    inner_intervals,
                    arguments,
                )
            else:
                output_shapes = tuple(
                    _real_shape(variable) for variable in equation.outvars
                )
                same_input = identity.is_square(equation)
                outputs = self._lower_primitive(
                    name,
                    arguments,
                    equation.params,
                    output_shapes,
                    same_input,
                )
                if name in ("eq", "ne"):
                    outputs = (self._decide_comparison(equation, outputs[0]),)
                elif name == "reduce_sum":
                    outputs = (self._bound_attainment(equation, outputs[0]),)
            if len(outputs) != len(equation.outvars):
                raise IntervalExtensionError(
                    f"native primitive {name!r} returned the wrong interval arity."
                )
            for variable, value in zip(equation.outvars, outputs, strict=True):
                if not isinstance(variable, jax_core.DropVar):
                    if _real_shape(variable) != value.shape:
                        raise IntervalExtensionError(
                            f"native primitive {name!r} returned an inconsistent shape."
                        )
                    environment[variable] = value
            operands = self.values.identify(equation, nested_outputs)
            extremum = self._record(equation, arguments, outputs, operands)
            if declared and _size(outputs[0].shape):
                entry = int(self._locations(outputs[0]).reshape((-1,))[0])
                self._event(_JUMP, _ALWAYS, equation.outvars[0], _Tensor((), entry))
            self._inspect(equation, arguments, outputs, operands, extremum)
        return tuple(
            environment[atom] if isinstance(atom, jax_core.Var) else self._literal(atom)
            for atom in jaxpr.outvars
        )

    def _literal(self, value: jax_core.Literal, /) -> _Tensor:
        literal = np.asarray(value.val)
        if literal.dtype.kind not in "biuf" or np.any(~np.isfinite(literal)):
            raise IntervalExtensionError(
                "native interval literals must be finite real values."
            )
        exact = np.asarray(literal, dtype=np.float64)
        if np.any(np.abs(exact) > 2**53) and literal.dtype.kind in "iu":
            raise IntervalExtensionError(
                "native interval integer literals must be exactly representable."
            )
        return self._seed(exact, exact, exact=exact)

    def _constant(self, lower: object, upper: object, /) -> _Tensor:
        lower_ = np.asarray(lower, dtype=np.float64)
        upper_ = np.asarray(upper, dtype=np.float64)
        exact = lower_ if np.array_equal(lower_, upper_) else None
        return self._seed(lower_, upper_, exact=exact)


def _declares_extension(equation: jax_core.JaxprEqn, /) -> bool:
    """Whether the host evaluates this custom-JVP call by a declared extension."""
    owner = custom_jvp_rule_owner(equation.params.get("jvp_jaxpr_fun"))
    if owner is None or len(equation.outvars) != 1:
        return False
    if owner.f in _CUSTOM_EXTENSIONS:
        return True
    operands = len(equation.invars) - int(equation.params.get("num_consts", 0))
    return bspline_jet_rule_alignment(owner.f) is not None and operands == 3


def _immutable_indices(values: Sequence[int], /) -> NDArray[np.int64]:
    result = np.asarray(values, dtype=np.int64).reshape((-1,))
    result.setflags(write=False)
    return result


def lower_interval_jaxpr(
    closed_jaxpr: jax_core.ClosedJaxpr,
    constant_intervals: Sequence[_Interval],
    batch_capacity: int,
    input_dimension: int,
    /,
    branch_axes: Mapping[jax_core.Var, int | None] | None = None,
) -> NativeIntervalProgram:
    """Compile a one-input real jaxpr or refuse its unsupported primitives.

    With ``branch_axes`` (the box axis of every traced variable) the program
    also carries the branch-evidence events of ``_BranchLedger``.
    """
    if len(closed_jaxpr.jaxpr.invars) != 1:
        raise IntervalExtensionError("native interval programs require one array input.")
    builder = _Builder(batch_capacity, input_dimension, branch_axes)
    input_variable = closed_jaxpr.jaxpr.invars[0]
    input_shape = _real_shape(input_variable)
    expected_input_shape = (batch_capacity, input_dimension)
    if input_shape != expected_input_shape:
        raise IntervalExtensionError(
            f"native interval input must have shape {expected_input_shape}, got {input_shape}."
        )
    if len(closed_jaxpr.jaxpr.constvars) != len(constant_intervals):
        raise IntervalExtensionError("native interval constants do not align with jaxpr.")
    constants = tuple(
        builder._constant(lower, upper) for lower, upper in constant_intervals
    )
    builder.values.bind_inputs(closed_jaxpr.jaxpr.invars)
    (output,) = builder.compile_jaxpr(
        closed_jaxpr.jaxpr,
        constants,
        tuple(constant_intervals),
        (_Tensor(expected_input_shape, 0),),
    )
    if output.locations is not None:
        locations = np.array(output.locations, dtype=np.int64, copy=True)
        locations.setflags(write=False)
        output = _Tensor(output.shape, output.offset, locations, output.exact)
    instructions = np.asarray(builder.rows, dtype=np.int64).reshape((-1, 8))
    indices = np.asarray(builder.indices, dtype=np.int64)
    parameters = np.asarray(builder.parameters, dtype=np.float64)
    instructions.setflags(write=False)
    indices.setflags(write=False)
    parameters.setflags(write=False)
    opcodes = frozenset(int(opcode) for opcode in instructions[:, 0])
    events = None
    if branch_axes is not None:
        events = np.asarray(builder.events, dtype=np.int64).reshape((-1, 6))
        events.setflags(write=False)
    return NativeIntervalProgram(
        instructions=instructions,
        indices=indices,
        parameters=parameters,
        workspace_size=builder.next_offset,
        batch_capacity=batch_capacity,
        input_dimension=input_dimension,
        seeds=tuple(builder.seeds),
        output=output,
        domain_guards=_immutable_indices(builder.domain_guards),
        opcodes=opcodes,
        events=events,
    )


__all__ = ["NativeIntervalProgram", "lower_interval_jaxpr"]
