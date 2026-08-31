#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from ._elements import (
    AbstractImplicitCircuitLaw,
    CircuitElement,
    CircuitElementEvaluation,
    CircuitElementStateLayout,
)


Instruction: TypeAlias = tuple[str, str | float | None]
_FUNCTIONS = {
    "exp": jnp.exp,
    "log": jnp.log,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tanh": jnp.tanh,
    "sqrt": jnp.sqrt,
    "abs": jnp.abs,
}


def _instructions(node: ast.AST, /) -> tuple[Instruction, ...]:
    if isinstance(node, ast.Expression):
        return _instructions(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return (("constant", float(node.value)),)
    if isinstance(node, ast.Name):
        return (("variable", node.id),)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operation = "positive" if isinstance(node.op, ast.UAdd) else "negative"
        return _instructions(node.operand) + ((operation, None),)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    ):
        operations = {
            ast.Add: "add",
            ast.Sub: "subtract",
            ast.Mult: "multiply",
            ast.Div: "divide",
            ast.Pow: "power",
        }
        return (
            _instructions(node.left)
            + _instructions(node.right)
            + ((operations[type(node.op)], None),)
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and len(node.args) == 1
        and not node.keywords
    ):
        return _instructions(node.args[0]) + ((f"function:{node.func.id}", None),)
    raise ValueError("Behavioral expression contains an unsupported construct.")


class BehavioralCurrentLaw(AbstractImplicitCircuitLaw):
    instructions: tuple[Instruction, ...] = eqx.field(static=True)
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    input_names: tuple[str, ...] = eqx.field(static=True)
    parameter_values: Array

    def __init__(
        self,
        expression: str,
        parameters: Mapping[str, ArrayLike] | None = None,
        input_names: Sequence[str] = (),
        /,
        *,
        law_id: str | None = None,
    ):
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("Behavioral expression must be nonempty.")
        instructions = _instructions(ast.parse(expression, mode="eval"))
        parameter_map = {} if parameters is None else dict(parameters)
        names = tuple(sorted(str(name) for name in parameter_map))
        if any(not name for name in names):
            raise ValueError("Behavioral parameter names must be nonempty.")
        values = jnp.asarray([parameter_map[name] for name in names], dtype=float)
        inputs = tuple(str(name) for name in input_names)
        if len(set(inputs)) != len(inputs) or any(not name for name in inputs):
            raise ValueError("Behavioral input names must be unique and nonempty.")
        allowed = (
            {"v", "time"}
            | {f"p_{name}" for name in names}
            | {f"u_{name}" for name in inputs}
        )
        used = {
            str(argument)
            for operation, argument in instructions
            if operation == "variable"
        }
        if not used <= allowed:
            raise ValueError(f"Unknown behavioral variables: {sorted(used - allowed)!r}.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "behavioral-current-law",
                    "expression": expression,
                    "parameters": names,
                    "inputs": inputs,
                }
            )
            if law_id is None
            else str(law_id)
        )
        if not identifier:
            raise ValueError("law_id must be non-empty.")
        self.instructions = instructions
        self.parameter_names = names
        self.input_names = inputs
        self.parameter_values = values
        self.terminal_count = 2
        self.voltage_rate_dependent = False
        self.state_layout = CircuitElementStateLayout()
        self.law_id = identifier

    def _value(self, time: Array, voltage: Array, inputs: Any, /) -> Array:
        stack: list[Array] = []
        parameter_map = dict(
            zip(self.parameter_names, self.parameter_values, strict=True)
        )
        for operation, argument in self.instructions:
            if operation == "constant":
                stack.append(jnp.asarray(argument, dtype=voltage.dtype))
            elif operation == "variable":
                name = str(argument)
                if name == "v":
                    stack.append(voltage)
                elif name == "time":
                    stack.append(time)
                elif name.startswith("p_"):
                    stack.append(parameter_map[name[2:]])
                else:
                    input_name = name[2:]
                    if not isinstance(inputs, dict) or input_name not in inputs:
                        raise ValueError(f"Behavioral law requires input {input_name!r}.")
                    stack.append(jnp.asarray(inputs[input_name]))
            elif operation in ("positive", "negative"):
                value = stack.pop()
                stack.append(value if operation == "positive" else -value)
            elif operation.startswith("function:"):
                stack.append(_FUNCTIONS[operation.split(":", 1)[1]](stack.pop()))
            else:
                right, left = stack.pop(), stack.pop()
                if operation == "add":
                    stack.append(left + right)
                elif operation == "subtract":
                    stack.append(left - right)
                elif operation == "multiply":
                    stack.append(left * right)
                elif operation == "divide":
                    stack.append(left / right)
                elif operation == "power":
                    stack.append(left**right)
        if len(stack) != 1 or stack[0].shape != ():
            raise ValueError("Behavioral expression did not produce one scalar.")
        return stack[0]

    def evaluate(
        self,
        time,
        terminal_voltages,
        terminal_voltage_rates,
        state,
        state_rate,
        inputs,
        args,
        /,
    ) -> CircuitElementEvaluation:
        del terminal_voltage_rates, state, state_rate, args
        voltage = terminal_voltages[0] - terminal_voltages[1]
        current = self._value(time, voltage, inputs)
        return CircuitElementEvaluation(jnp.asarray([current, -current]), jnp.zeros((0,)))


def compile_behavioral_current(
    expression: str,
    /,
    *,
    parameters: Mapping[str, ArrayLike] | None = None,
    input_names: Sequence[str] = (),
    element_id: str = "behavioral-current",
) -> CircuitElement:
    law = BehavioralCurrentLaw(
        expression,
        parameters,
        input_names,
        law_id=f"{element_id}/law",
    )
    return CircuitElement(law, element_id=element_id)


__all__ = ["BehavioralCurrentLaw", "compile_behavioral_current"]
