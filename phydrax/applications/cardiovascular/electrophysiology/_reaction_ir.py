#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Inspectable host-compiled algebra for pinned cardiac reactions.

The IR is a closed algebraic data type.  Compilation lowers expression trees to
an immutable register program; it never evaluates source text, imports plugins,
or dispatches user-provided callables.  Runtime inputs are positional arrays in
the exact order pinned by the IR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import auto, Enum
from hashlib import sha256
from math import isfinite
from typing import TypeAlias

import jax.numpy as jnp
from jaxtyping import Array


class ReactionUnaryOperator(Enum):
    """Closed set of admitted unary algebraic operations."""

    NEGATE = auto()
    ABSOLUTE = auto()
    EXP = auto()
    EXPM1 = auto()
    LOG = auto()
    LOG1P = auto()
    SQRT = auto()
    TANH = auto()
    RECIPROCAL = auto()


class ReactionBinaryOperator(Enum):
    """Closed set of admitted binary algebraic operations."""

    ADD = auto()
    SUBTRACT = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    POWER = auto()
    MINIMUM = auto()
    MAXIMUM = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()


@dataclass(frozen=True)
class ReactionIRLiteral:
    value: float

    def __post_init__(self) -> None:
        if isinstance(self.value, bool):
            raise TypeError("IR literals must be real scalars, not bool.")
        value = float(self.value)
        if not isfinite(value):
            raise ValueError("IR literals must be finite.")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class ReactionIRInput:
    slot: int

    def __post_init__(self) -> None:
        if not isinstance(self.slot, int) or isinstance(self.slot, bool):
            raise TypeError("IR input slot must be an integer.")
        if self.slot < 0:
            raise ValueError("IR input slot must be nonnegative.")


@dataclass(frozen=True)
class ReactionIRUnary:
    operator: ReactionUnaryOperator
    operand: ReactionIRExpression

    def __post_init__(self) -> None:
        if not isinstance(self.operator, ReactionUnaryOperator):
            raise TypeError("operator must be a ReactionUnaryOperator.")


@dataclass(frozen=True)
class ReactionIRBinary:
    operator: ReactionBinaryOperator
    left: ReactionIRExpression
    right: ReactionIRExpression

    def __post_init__(self) -> None:
        if not isinstance(self.operator, ReactionBinaryOperator):
            raise TypeError("operator must be a ReactionBinaryOperator.")


@dataclass(frozen=True)
class ReactionIRSelect:
    predicate: ReactionIRExpression
    when_true: ReactionIRExpression
    when_false: ReactionIRExpression


ReactionIRExpression: TypeAlias = (
    ReactionIRLiteral
    | ReactionIRInput
    | ReactionIRUnary
    | ReactionIRBinary
    | ReactionIRSelect
)


@dataclass(frozen=True)
class ReactionIROutput:
    name: str
    expression: ReactionIRExpression

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("IR output name must be a non-empty string.")


@dataclass(frozen=True)
class PinnedReactionIR:
    """Declared positional inputs and algebraic outputs for one pinned program."""

    program_name: str
    input_names: tuple[str, ...]
    outputs: tuple[ReactionIROutput, ...]
    maximum_nodes: int = 4096
    program_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.program_name, str) or not self.program_name:
            raise ValueError("program_name must be a non-empty string.")
        names = tuple(self.input_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("input_names must contain non-empty strings.")
        if len(set(names)) != len(names):
            raise ValueError("input_names must be unique.")
        outputs = tuple(self.outputs)
        if not outputs:
            raise ValueError("A reaction IR must declare at least one output.")
        output_names = tuple(output.name for output in outputs)
        if len(set(output_names)) != len(output_names):
            raise ValueError("reaction IR output names must be unique.")
        if not isinstance(self.maximum_nodes, int) or isinstance(
            self.maximum_nodes, bool
        ):
            raise TypeError("maximum_nodes must be an integer.")
        if self.maximum_nodes <= 0:
            raise ValueError("maximum_nodes must be positive.")
        count = 0
        maximum_depth = 0
        fingerprints: list[str] = []
        for output in outputs:
            output_count, output_depth, fingerprint = _inspect_expression(
                output.expression,
                len(names),
            )
            count += output_count
            maximum_depth = max(maximum_depth, output_depth)
            fingerprints.append(f"{output.name}:{fingerprint}")
        if count > self.maximum_nodes:
            raise ValueError(
                f"reaction IR has {count} nodes, exceeding maximum_nodes={self.maximum_nodes}."
            )
        if maximum_depth > 256:
            raise ValueError("reaction IR expression depth exceeds 256.")
        identity = (
            "pinned-cardiac-reaction-ir-v1\0"
            + self.program_name
            + "\0"
            + repr(names)
            + "\0"
            + "\0".join(fingerprints)
        )
        object.__setattr__(self, "input_names", names)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(
            self, "program_id", sha256(identity.encode("utf-8")).hexdigest()
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(output.name for output in self.outputs)


class _InstructionKind(Enum):
    LITERAL = auto()
    UNARY = auto()
    BINARY = auto()
    SELECT = auto()


@dataclass(frozen=True)
class _Instruction:
    kind: _InstructionKind
    operator: ReactionUnaryOperator | ReactionBinaryOperator | None
    first: int
    second: int
    third: int
    literal: float


@dataclass(frozen=True)
class CompiledReactionIR:
    """Immutable positional register program produced on the host."""

    source: PinnedReactionIR
    instructions: tuple[_Instruction, ...]
    output_registers: tuple[int, ...]

    @property
    def program_id(self) -> str:
        return self.source.program_id

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.source.input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.source.output_names

    @property
    def register_count(self) -> int:
        return len(self.input_names) + len(self.instructions)

    def __call__(self, inputs: tuple[Array, ...] | list[Array], /) -> tuple[Array, ...]:
        values = tuple(inputs)
        if len(values) != len(self.input_names):
            raise ValueError(
                f"compiled reaction expects {len(self.input_names)} inputs, got {len(values)}."
            )
        registers: list[Array] = [jnp.asarray(value) for value in values]
        for instruction in self.instructions:
            if instruction.kind is _InstructionKind.LITERAL:
                if registers:
                    value = jnp.asarray(instruction.literal, dtype=registers[0].dtype)
                else:
                    value = jnp.asarray(instruction.literal)
            elif instruction.kind is _InstructionKind.UNARY:
                value = _apply_unary(
                    instruction.operator,
                    registers[instruction.first],
                )
            elif instruction.kind is _InstructionKind.BINARY:
                value = _apply_binary(
                    instruction.operator,
                    registers[instruction.first],
                    registers[instruction.second],
                )
            else:
                value = jnp.where(
                    registers[instruction.first],
                    registers[instruction.second],
                    registers[instruction.third],
                )
            registers.append(value)
        return tuple(registers[index] for index in self.output_registers)

    def inspect(self) -> tuple[tuple[int, str, int, int, int, float], ...]:
        """Return deterministic host-readable register instructions."""
        offset = len(self.input_names)
        return tuple(
            (
                offset + index,
                instruction.kind.name
                if instruction.operator is None
                else instruction.operator.name,
                instruction.first,
                instruction.second,
                instruction.third,
                instruction.literal,
            )
            for index, instruction in enumerate(self.instructions)
        )


def _inspect_expression(
    expression: ReactionIRExpression,
    input_count: int,
) -> tuple[int, int, str]:
    if isinstance(expression, ReactionIRLiteral):
        return 1, 1, f"literal({expression.value.hex()})"
    if isinstance(expression, ReactionIRInput):
        if expression.slot >= input_count:
            raise ValueError(
                f"IR input slot {expression.slot} is outside declared input count {input_count}."
            )
        return 1, 1, f"input({expression.slot})"
    if isinstance(expression, ReactionIRUnary):
        count, depth, child = _inspect_expression(expression.operand, input_count)
        return count + 1, depth + 1, f"{expression.operator.name}({child})"
    if isinstance(expression, ReactionIRBinary):
        lc, ld, left = _inspect_expression(expression.left, input_count)
        rc, rd, right = _inspect_expression(expression.right, input_count)
        return lc + rc + 1, max(ld, rd) + 1, f"{expression.operator.name}({left},{right})"
    if isinstance(expression, ReactionIRSelect):
        pc, pd, predicate = _inspect_expression(expression.predicate, input_count)
        tc, td, when_true = _inspect_expression(expression.when_true, input_count)
        fc, fd, when_false = _inspect_expression(expression.when_false, input_count)
        return (
            pc + tc + fc + 1,
            max(pd, td, fd) + 1,
            f"SELECT({predicate},{when_true},{when_false})",
        )
    raise TypeError(f"Unsupported reaction IR node {type(expression).__name__!r}.")


def compile_reaction_ir(source: PinnedReactionIR, /) -> CompiledReactionIR:
    """Compile a validated expression tree to a fixed register program."""
    if not isinstance(source, PinnedReactionIR):
        raise TypeError("source must be PinnedReactionIR.")
    instructions: list[_Instruction] = []
    input_count = len(source.input_names)

    def lower(expression: ReactionIRExpression) -> int:
        if isinstance(expression, ReactionIRInput):
            return expression.slot
        if isinstance(expression, ReactionIRLiteral):
            instructions.append(
                _Instruction(
                    _InstructionKind.LITERAL,
                    None,
                    -1,
                    -1,
                    -1,
                    expression.value,
                )
            )
        elif isinstance(expression, ReactionIRUnary):
            operand = lower(expression.operand)
            instructions.append(
                _Instruction(
                    _InstructionKind.UNARY,
                    expression.operator,
                    operand,
                    -1,
                    -1,
                    0.0,
                )
            )
        elif isinstance(expression, ReactionIRBinary):
            left = lower(expression.left)
            right = lower(expression.right)
            instructions.append(
                _Instruction(
                    _InstructionKind.BINARY,
                    expression.operator,
                    left,
                    right,
                    -1,
                    0.0,
                )
            )
        elif isinstance(expression, ReactionIRSelect):
            predicate = lower(expression.predicate)
            when_true = lower(expression.when_true)
            when_false = lower(expression.when_false)
            instructions.append(
                _Instruction(
                    _InstructionKind.SELECT,
                    None,
                    predicate,
                    when_true,
                    when_false,
                    0.0,
                )
            )
        else:
            raise TypeError(
                f"Unsupported reaction IR node {type(expression).__name__!r}."
            )
        return input_count + len(instructions) - 1

    output_registers = tuple(lower(output.expression) for output in source.outputs)
    return CompiledReactionIR(source, tuple(instructions), output_registers)


def interpret_reaction_ir(
    source: PinnedReactionIR,
    inputs: tuple[Array, ...] | list[Array],
    /,
) -> tuple[Array, ...]:
    """Direct tree interpreter used as an independent qualification route."""
    values = tuple(jnp.asarray(value) for value in inputs)
    if len(values) != len(source.input_names):
        raise ValueError(
            f"reaction IR expects {len(source.input_names)} inputs, got {len(values)}."
        )

    def evaluate(expression: ReactionIRExpression) -> Array:
        if isinstance(expression, ReactionIRLiteral):
            dtype = values[0].dtype if values else None
            return jnp.asarray(expression.value, dtype=dtype)
        if isinstance(expression, ReactionIRInput):
            return values[expression.slot]
        if isinstance(expression, ReactionIRUnary):
            return _apply_unary(expression.operator, evaluate(expression.operand))
        if isinstance(expression, ReactionIRBinary):
            return _apply_binary(
                expression.operator,
                evaluate(expression.left),
                evaluate(expression.right),
            )
        if isinstance(expression, ReactionIRSelect):
            return jnp.where(
                evaluate(expression.predicate),
                evaluate(expression.when_true),
                evaluate(expression.when_false),
            )
        raise TypeError(f"Unsupported reaction IR node {type(expression).__name__!r}.")

    return tuple(evaluate(output.expression) for output in source.outputs)


def _apply_unary(
    operator: ReactionUnaryOperator | ReactionBinaryOperator | None,
    operand: Array,
) -> Array:
    if operator is ReactionUnaryOperator.NEGATE:
        return -operand
    if operator is ReactionUnaryOperator.ABSOLUTE:
        return jnp.abs(operand)
    if operator is ReactionUnaryOperator.EXP:
        return jnp.exp(operand)
    if operator is ReactionUnaryOperator.EXPM1:
        return jnp.expm1(operand)
    if operator is ReactionUnaryOperator.LOG:
        return jnp.log(operand)
    if operator is ReactionUnaryOperator.LOG1P:
        return jnp.log1p(operand)
    if operator is ReactionUnaryOperator.SQRT:
        return jnp.sqrt(operand)
    if operator is ReactionUnaryOperator.TANH:
        return jnp.tanh(operand)
    if operator is ReactionUnaryOperator.RECIPROCAL:
        return jnp.reciprocal(operand)
    raise TypeError("Unknown unary reaction IR operator.")


def _apply_binary(
    operator: ReactionUnaryOperator | ReactionBinaryOperator | None,
    left: Array,
    right: Array,
) -> Array:
    if operator is ReactionBinaryOperator.ADD:
        return left + right
    if operator is ReactionBinaryOperator.SUBTRACT:
        return left - right
    if operator is ReactionBinaryOperator.MULTIPLY:
        return left * right
    if operator is ReactionBinaryOperator.DIVIDE:
        return left / right
    if operator is ReactionBinaryOperator.POWER:
        return left**right
    if operator is ReactionBinaryOperator.MINIMUM:
        return jnp.minimum(left, right)
    if operator is ReactionBinaryOperator.MAXIMUM:
        return jnp.maximum(left, right)
    if operator is ReactionBinaryOperator.LESS:
        return left < right
    if operator is ReactionBinaryOperator.LESS_EQUAL:
        return left <= right
    if operator is ReactionBinaryOperator.GREATER:
        return left > right
    if operator is ReactionBinaryOperator.GREATER_EQUAL:
        return left >= right
    raise TypeError("Unknown binary reaction IR operator.")


__all__ = [
    "CompiledReactionIR",
    "PinnedReactionIR",
    "ReactionBinaryOperator",
    "ReactionIRBinary",
    "ReactionIRInput",
    "ReactionIRLiteral",
    "ReactionIROutput",
    "ReactionIRSelect",
    "ReactionIRUnary",
    "ReactionUnaryOperator",
    "compile_reaction_ir",
    "interpret_reaction_ir",
]
