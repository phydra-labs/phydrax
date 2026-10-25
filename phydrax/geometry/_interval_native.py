#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Straight-line native program form of a prepared interval function.

The canonical interval owner (``_interval_enclosure``) interprets a traced
jaxpr equation by equation with NumPy. This module lowers the bounded subset
that carries the source-equation workload into a fixed instruction list over
one float64 lower/upper workspace, executed by the meshcore kernel
``phx_mc_interval_program_execute``. Lowering never changes the arithmetic:

* admission is a static scan of the jaxpr; a program with any other
  input-dependent primitive, dtype or reduction is declined before any
  numerical work;
* every equation whose inputs are all constants or literals is evaluated once
  by the owner's own equation dispatch (the same folding the interpreter
  repeats on every evaluation) and becomes workspace constants; no
  input-dependent equation is evaluated during preparation;
* structural primitives emit no instruction: the owner's own structural NumPy
  evaluation of position arrays yields each value's element positions, which
  later operands read through index maps;
* stack and concatenate join absolute positions across earlier workspace
  regions; uncertain source layouts do not authorize native reductions;
* binary operands use NumPy broadcasting maps of the same shapes;
* each remaining input-dependent equation must be one of ADD, SUB, SUB_SAME,
  MUL, NEG, REDUCE_SUM, SIN or COS on float64 values.

NumPy sums a C-contiguous innermost axis of fewer than eight terms in source
order. Position arrays replicate the owner's array strides: the input is
C-contiguous; an arithmetic result is C-contiguous on every value path of the
owner formulas (K-order results, ``full``/``zeros`` fallbacks and reductions
alike) when NumPy's own K-order decision over int64 views of its operands'
strides is C-contiguous; structural views apply the owner's NumPy operation to
those strides. A reduction whose owner operand is not proved C-contiguous is
declined.

Instruction semantics (one row of eight int64 fields) mirror the owner's
NumPy formulas element by element:

``GATHER``     ``out[k] = a[ia[k]]`` for both bounds.
``ADD``        ``interval_add``: endpoint sums, one outward ``nextafter``.
``SUB``        ``interval_subtract`` including its proved-zero replacement.
``SUB_SAME``   ``sub x x``: ``SUB`` then zero where both bounds are finite.
``MUL``        ``interval_multiply``: four endpoint products (``0 * inf`` is
               ``+0``), nested ``min(min(ll, lu), min(ul, uu))`` /
               ``max(...)`` with NumPy's NaN-first/signed-zero rules, one step.
``NEG``        ``(-upper, -lower)``.
``REDUCE_SUM`` per row ``s = +0.0; s += x[i]`` for both bounds and for
               ``max(|lo|, |hi|)``; ``lo - g * scale``, ``hi + g * scale``,
               one step, ``g = parameters[field 6]``.
``SIN``/``COS`` ``interval_sin``/``interval_cos`` with libm endpoints, phase
               containment, four steps and ``[-1, 1]`` clamping.

Row fields: ``(opcode, out_offset, out_size, a_offset, a_index, b_offset,
b_index, auxiliary)``; an index of ``-1`` reads element ``k`` directly,
otherwise element ``k`` reads ``offset + indices[index + k]``. Unary rows and
``GATHER`` carry ``b_offset = b_index = -1`` and ``auxiliary = 0``.
``REDUCE_SUM`` reads ``out_size * auxiliary`` contiguous terms at ``a_offset``
(``a_index = b_offset = -1``), uses ``b_index`` as its parameter index and
``auxiliary`` as the reduced length. Rows write ascending, non-overlapping
regions and read only below their own output region.

The retained banks are bounded before allocation: the workspace never
exceeds the elements the owner's own environment declares for the program
(every variable and literal once) and the index bank never exceeds two reads
per written element. Admitted banks are then sealed as immutable bytes.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Container
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from jax.core import ShapedArray
from jax.extend import core as jax_core

from .._host_storage import immutable_host_array
from ._interval_enclosure import (
    _array_aval,
    _evaluate_equation,
    _host_bind,
    _literal_interval,
    coefficient_interval,
    Interval,
    NativeIntervalProgram,
    PreparedIntervalFunction,
)


GATHER: Final = 1
ADD: Final = 2
SUB: Final = 3
SUB_SAME: Final = 4
MUL: Final = 5
NEG: Final = 6
REDUCE_SUM: Final = 7
SIN: Final = 8
COS: Final = 9
FIELDS: Final = 8
# Fewer than eight innermost terms are summed by NumPy without unrolled
# pairwise accumulators; longer reductions stay on the owner interpreter.
MAXIMUM_REDUCED_LENGTH: Final = 7

_STRUCTURAL_ONE_INPUT: Final = frozenset(
    {
        "broadcast_in_dim",
        "reshape",
        "squeeze",
        "expand_dims",
        "slice",
        "transpose",
        "rev",
        "copy",
        "copy_p",
        "split",
    }
)
_STRUCTURAL_MULTIPLE_INPUTS: Final = frozenset({"stack", "concatenate"})
_BINARY: Final = {"add": ADD, "add_any": ADD, "sub": SUB, "mul": MUL}
_UNARY: Final = {"neg": NEG, "sin": SIN, "cos": COS}
_FLOAT64: Final = np.dtype(np.float64)

type Positions = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _Value:
    """Workspace positions of one input-dependent value's logical elements.

    ``owner_layout`` records that ``positions`` has the strides of the owner's
    arrays for this value on every input (up to extent-one axes).
    """

    offset: int
    positions: Positions
    owner_layout: bool


def _runtime(
    atom: jax_core.Var | jax_core.Literal, runtime: Container[jax_core.Var], /
) -> bool:
    return isinstance(atom, jax_core.Var) and atom in runtime


def _float64_shape(atom: jax_core.Var | jax_core.Literal, /) -> tuple[int, ...] | None:
    match atom.aval:
        case ShapedArray() as aval if aval.dtype == _FLOAT64:
            return tuple(aval.shape)
        case _:
            return None


def _admitted(closed: jax_core.ClosedJaxpr, /) -> bool:
    """Statically admit every input-dependent equation before numerical work."""
    jaxpr = closed.jaxpr
    if len(jaxpr.invars) != 1 or len(jaxpr.outvars) != 1:
        return False
    if _float64_shape(jaxpr.invars[0]) is None:
        return False
    runtime: set[jax_core.Var] = set(jaxpr.invars)
    for equation in jaxpr.eqns:
        operands = [_runtime(atom, runtime) for atom in equation.invars]
        if not any(operands):
            continue
        name = equation.primitive.name
        if any(_float64_shape(variable) is None for variable in equation.outvars):
            return False
        if name == "convert_element_type" or name in _STRUCTURAL_ONE_INPUT or name in _UNARY:
            admitted = len(operands) == 1
        elif name in _BINARY:
            admitted = len(operands) == 2 and all(
                _float64_shape(atom) is not None for atom in equation.invars
            )
        elif name in _STRUCTURAL_MULTIPLE_INPUTS:
            admitted = all(_float64_shape(atom) is not None for atom in equation.invars)
        elif name == "reduce_sum":
            shape = _float64_shape(equation.invars[0])
            admitted = (
                shape is not None
                and len(shape) > 0
                and tuple(int(axis) for axis in equation.params["axes"]) == (len(shape) - 1,)
                and shape[-1] <= MAXIMUM_REDUCED_LENGTH
            )
        else:
            admitted = False
        if not admitted:
            return False
        runtime.update(equation.outvars)
    return True


def _declared_elements(closed: jax_core.ClosedJaxpr, /) -> int:
    """Elements of every array variable and literal the owner environment holds."""
    jaxpr = closed.jaxpr
    atoms: list[jax_core.Var | jax_core.Literal] = [*jaxpr.constvars, *jaxpr.invars]
    for equation in jaxpr.eqns:
        atoms.extend(equation.outvars)
        atoms.extend(atom for atom in equation.invars if isinstance(atom, jax_core.Literal))
    atoms.extend(atom for atom in jaxpr.outvars if isinstance(atom, jax_core.Literal))
    total = 0
    for atom in atoms:
        match atom.aval:
            case ShapedArray() as aval:
                total += math.prod(aval.shape)
            case _:
                pass
    return total


def _k_order_c(*operands: tuple[Positions, ...]) -> bool:
    """Whether NumPy's K-order result over these owner layouts is C-contiguous.

    Each operand lists int64 views with the strides of its owner bounds, so
    NumPy itself makes the owner's layout decision. Every path of an admitted
    owner formula combines K-order results over the same operands with fresh
    C-order arrays (``full``/``zeros`` fallbacks, ``stack`` and reductions),
    so a C-contiguous K-order result proves a C-contiguous owner result.
    """
    for combination in itertools.product(*operands):
        if len(combination) == 1:
            result = np.negative(combination[0])
        else:
            result = np.add(*combination)
        if not result.flags.c_contiguous:
            return False
    return True


class _Builder:
    def __init__(self) -> None:
        self.rows: list[tuple[int, ...]] = []
        self.index_chunks: list[Positions] = []
        self.index_size = 0
        self.parameters: list[float] = []
        self.constants: list[tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
        self.size = 0

    def allocate(self, shape: tuple[int, ...], /) -> tuple[int, Positions]:
        size = math.prod(shape)
        offset = self.size
        self.size += size
        return offset, np.arange(size, dtype=np.int64).reshape(shape)

    def constant(
        self, lower: npt.NDArray[np.float64], upper: npt.NDArray[np.float64], /
    ) -> tuple[int, Positions]:
        offset, positions = self.allocate(tuple(lower.shape))
        self.constants.append((offset, lower, upper))
        return offset, positions

    def read(self, offset: int, positions: Positions, /) -> tuple[int, int]:
        """Address of C-order elements ``positions`` of the buffer at ``offset``."""
        flat = positions.reshape(-1)
        size = flat.size
        if size == 0:
            return offset, -1
        start = int(flat[0])
        if np.array_equal(flat, np.arange(start, start + size, dtype=np.int64)):
            return offset + start, -1
        index = self.index_size
        self.index_chunks.append(np.ascontiguousarray(flat, dtype=np.int64))
        self.index_size += size
        return offset, index

    def emit(self, *fields: int) -> None:
        if len(fields) != FIELDS:
            raise ValueError("native interval instructions must have eight fields")
        self.rows.append(tuple(int(field) for field in fields))

    def finish(
        self, input_offset: int, input_size: int, output_offset: int, shape: tuple[int, ...], /
    ) -> NativeIntervalProgram:
        lower = np.zeros(self.size, dtype=np.float64)
        upper = np.zeros(self.size, dtype=np.float64)
        for offset, low, high in self.constants:
            lower[offset : offset + low.size] = low.reshape(-1)
            upper[offset : offset + high.size] = high.reshape(-1)
        # Each admitted bank is copied once into private immutable bytes, so no
        # view can be made writeable or alias the lowering scratch.
        return NativeIntervalProgram(
            instructions=immutable_host_array(
                np.asarray(self.rows, dtype=np.int64).reshape(-1, FIELDS)
            ),
            indices=immutable_host_array(
                np.concatenate(self.index_chunks)
                if self.index_chunks
                else np.zeros(0, dtype=np.int64)
            ),
            parameters=immutable_host_array(np.asarray(self.parameters, dtype=np.float64)),
            template_lower=immutable_host_array(lower),
            template_upper=immutable_host_array(upper),
            input_offset=input_offset,
            input_size=input_size,
            output_offset=output_offset,
            output_shape=shape,
        )


def _float64_interval(
    value: Interval, /
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    lower, upper = np.asarray(value[0]), np.asarray(value[1])
    if lower.dtype != _FLOAT64 or upper.dtype != _FLOAT64 or lower.shape != upper.shape:
        return None
    return lower, upper


def lower_native_interval_program(
    program: PreparedIntervalFunction, /
) -> NativeIntervalProgram | None:
    """Lower the admitted straight-line subset, or return ``None``.

    Owner evaluation refusals during constant folding also return ``None`` so
    the interpreter keeps raising them at evaluation, in its own order.
    """
    closed = program.closed_jaxpr
    if not _admitted(closed):
        return None
    builder = _Builder()
    constants: dict[jax_core.Var, Interval] = {}
    runtime: dict[jax_core.Var, _Value] = {}
    # Operand reads: buffer offset, element positions and the int64 views of
    # the owner's bound layouts (``None`` when the owner layout is unproved).
    constant_buffers: dict[int, tuple[int, Positions, tuple[Positions, ...]]] = {}
    contiguous_buffers: dict[jax_core.Var, int] = {}

    def constant_value(atom: jax_core.Var | jax_core.Literal, /) -> Interval:
        if isinstance(atom, jax_core.Literal):
            return _literal_interval(atom)
        return constants[atom]

    def constant_operand(
        atom: jax_core.Var | jax_core.Literal, /
    ) -> tuple[int, Positions, tuple[Positions, ...] | None] | None:
        cached = constant_buffers.get(id(atom))
        if cached is not None:
            return cached
        value = _float64_interval(constant_value(atom))
        if value is None:
            return None
        offset, positions = builder.constant(*value)
        cached = constant_buffers[id(atom)] = (
            offset,
            positions,
            (value[0].view(np.int64), value[1].view(np.int64)),
        )
        return cached

    def operand(
        atom: jax_core.Var | jax_core.Literal, /
    ) -> tuple[int, Positions, tuple[Positions, ...] | None] | None:
        if isinstance(atom, jax_core.Var) and atom in runtime:
            value = runtime[atom]
            layouts = (value.positions,) if value.owner_layout else None
            return value.offset, value.positions, layouts
        return constant_operand(atom)

    def contiguous(variable: jax_core.Var, /) -> int:
        """Offset of a C-order region holding ``variable``'s elements."""
        cached = contiguous_buffers.get(variable)
        if cached is not None:
            return cached
        value = runtime[variable]
        offset, index = builder.read(value.offset, value.positions)
        if index >= 0:
            target, _ = builder.allocate(tuple(value.positions.shape))
            builder.emit(GATHER, target, value.positions.size, offset, index, -1, -1, 0)
            offset = target
        contiguous_buffers[variable] = offset
        return offset

    for variable, constant in zip(closed.jaxpr.constvars, closed.consts, strict=True):
        constants[variable] = coefficient_interval(np.asarray(constant), program.constant_bounds)
    (argument,) = closed.jaxpr.invars
    input_shape = (program.batch_capacity, program.input_dimension)
    if tuple(_array_aval(argument).shape) != input_shape:
        return None
    input_offset, input_positions = builder.allocate(input_shape)
    runtime[argument] = _Value(input_offset, input_positions, True)

    for equation in closed.jaxpr.eqns:
        if not any(_runtime(atom, runtime) for atom in equation.invars):
            try:
                outputs = _evaluate_equation(
                    equation,
                    [constant_value(atom) for atom in equation.invars],
                    program.constant_bounds,
                )
            except (ValueError, ArithmeticError):
                return None
            for variable, output in zip(equation.outvars, outputs, strict=True):
                constants[variable] = output
            continue
        name = equation.primitive.name
        shapes = [tuple(_array_aval(variable).shape) for variable in equation.outvars]
        if name in _BINARY:
            (shape,) = shapes
            (result,) = equation.outvars
            reads: list[tuple[int, int]] = []
            layouts: list[tuple[Positions, ...]] = []
            for atom in equation.invars:
                resolved = operand(atom)
                if resolved is None:
                    return None
                offset, positions, operand_layouts = resolved
                if np.broadcast_shapes(positions.shape, shape) != shape:
                    return None
                reads.append(builder.read(offset, np.broadcast_to(positions, shape)))
                if operand_layouts is not None:
                    layouts.append(operand_layouts)
            opcode = _BINARY[name]
            if name == "sub" and equation.invars[0] is equation.invars[1]:
                opcode = SUB_SAME
            target, positions = builder.allocate(shape)
            (first, first_index), (second, second_index) = reads
            builder.emit(
                opcode, target, math.prod(shape), first, first_index, second, second_index, 0
            )
            runtime[result] = _Value(
                target, positions, len(layouts) == 2 and _k_order_c(*layouts)
            )
            continue
        if name in _STRUCTURAL_MULTIPLE_INPUTS:
            absolute_positions: list[Positions] = []
            owner_c_contiguous = True
            for atom in equation.invars:
                resolved = operand(atom)
                if resolved is None:
                    return None
                offset, positions, layouts = resolved
                absolute_positions.append(positions + offset)
                owner_c_contiguous &= layouts is not None and all(
                    layout.flags.c_contiguous for layout in layouts
                )
            (mapped,) = _host_bind(equation, absolute_positions)
            (result,) = equation.outvars
            # Absolute positions may span several earlier workspace regions.
            # No numerical copy is needed until a contiguous consumer requests
            # one. Only proven C-order inputs establish the owner's output layout.
            runtime[result] = _Value(
                0, np.asarray(mapped, dtype=np.int64), owner_c_contiguous
            )
            continue
        # Admission proved one input-dependent operand for every other form.
        (source_atom,) = equation.invars
        if not isinstance(source_atom, jax_core.Var):
            return None
        source = runtime[source_atom]
        if name == "convert_element_type":
            # The owner returns a float64 operand unchanged.
            runtime[equation.outvars[0]] = source
            continue
        if name in _STRUCTURAL_ONE_INPUT:
            for variable, mapped in zip(
                equation.outvars, _host_bind(equation, [source.positions]), strict=True
            ):
                runtime[variable] = _Value(source.offset, np.asarray(mapped), source.owner_layout)
            continue
        (shape,) = shapes
        size = math.prod(shape)
        (result,) = equation.outvars
        if name in _UNARY:
            if source.positions.shape != shape:
                return None
            first, first_index = builder.read(source.offset, source.positions)
            target, positions = builder.allocate(shape)
            builder.emit(_UNARY[name], target, size, first, first_index, -1, -1, 0)
            runtime[result] = _Value(
                target, positions, source.owner_layout and _k_order_c((source.positions,))
            )
            continue
        # reduce_sum, admitted over the innermost axis of fewer than eight terms.
        if not (source.owner_layout and source.positions.flags.c_contiguous):
            return None
        length = source.positions.shape[-1]
        first = contiguous(source_atom)
        unit = np.finfo(np.float64).eps
        reduced = max(1, length)
        builder.parameters.append(float(reduced * unit / (1.0 - reduced * unit)))
        target, positions = builder.allocate(shape)
        builder.emit(
            REDUCE_SUM, target, size, first, -1, -1, len(builder.parameters) - 1, length
        )
        runtime[result] = _Value(target, positions, True)

    (output,) = closed.jaxpr.outvars
    shape = (program.batch_capacity, *program.output_shape)
    if isinstance(output, jax_core.Var) and output in runtime:
        if runtime[output].positions.shape != shape:
            return None
        output_offset = contiguous(output)
    else:
        # An input-dependent result is absent: the owner's folded constant is
        # the program value, and the kernel executes no instruction for it.
        value = _float64_interval(constant_value(output))
        if value is None or value[0].shape != shape:
            return None
        cached = constant_buffers.get(id(output))
        output_offset = cached[0] if cached is not None else builder.constant(*value)[0]
    if builder.size > _declared_elements(closed) or builder.index_size > 2 * builder.size:
        return None
    return builder.finish(input_offset, math.prod(input_shape), output_offset, shape)


__all__ = [
    "ADD",
    "COS",
    "FIELDS",
    "GATHER",
    "lower_native_interval_program",
    "MAXIMUM_REDUCED_LENGTH",
    "MUL",
    "NEG",
    "REDUCE_SUM",
    "SIN",
    "SUB",
    "SUB_SAME",
]
