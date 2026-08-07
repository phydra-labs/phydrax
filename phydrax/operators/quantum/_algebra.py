#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import ArrayLike

from phydrax.domain import DomainFunction

from ..._strict import StrictModule
from ..linalg import adjoint
from ._validation import (
    coerce_hbar,
    join_function_arguments,
    validate_matrix_value,
)


class _BinaryMatrixCallable(StrictModule):
    left: DomainFunction
    right: DomainFunction
    left_positions: tuple[int, ...]
    right_positions: tuple[int, ...]
    operation: str

    def __init__(
        self,
        left: DomainFunction,
        right: DomainFunction,
        left_positions: tuple[int, ...],
        right_positions: tuple[int, ...],
        operation: Literal["commutator", "anticommutator"],
    ):
        self.left = left
        self.right = right
        self.left_positions = left_positions
        self.right_positions = right_positions
        self.operation = operation

    def __call__(self, *args, key=None, **kwargs):
        left_args = tuple(args[index] for index in self.left_positions)
        right_args = tuple(args[index] for index in self.right_positions)
        left = validate_matrix_value(
            self.left.func(*left_args, key=key, **kwargs),
            role="left quantum-bracket operand",
        )
        right = validate_matrix_value(
            self.right.func(*right_args, key=key, **kwargs),
            role="right quantum-bracket operand",
        )
        if left.shape != right.shape:
            raise ValueError(
                "Quantum-bracket matrix dimensions must match; "
                f"got {left.shape} and {right.shape}."
            )
        product_lr = left @ right
        product_rl = right @ left
        if self.operation == "commutator":
            return product_lr - product_rl
        if self.operation == "anticommutator":
            return product_lr + product_rl
        raise RuntimeError(f"Unknown matrix operation {self.operation!r}.")


class _UnitTraceCallable(StrictModule):
    density: DomainFunction

    def __init__(self, density: DomainFunction):
        self.density = density

    def __call__(self, *args, key=None, **kwargs):
        density = validate_matrix_value(
            self.density.func(*args, key=key, **kwargs),
            role="density operator",
        )
        return jnp.trace(density) - 1.0


def _binary_matrix_operation(
    left: DomainFunction,
    right: DomainFunction,
    /,
    *,
    operation: Literal["commutator", "anticommutator"],
) -> DomainFunction:
    if not isinstance(left, DomainFunction) or not isinstance(right, DomainFunction):
        raise TypeError(f"{operation} expects two DomainFunction operands.")
    domain, deps, promoted, positions = join_function_arguments(left, right)
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_BinaryMatrixCallable(
            promoted[0],
            promoted[1],
            positions[0],
            positions[1],
            operation,
        ),
        metadata={},
    )


def commutator(left: DomainFunction, right: DomainFunction, /) -> DomainFunction:
    r"""Matrix commutator $[A,B]=AB-BA$."""
    return _binary_matrix_operation(left, right, operation="commutator")


def anticommutator(left: DomainFunction, right: DomainFunction, /) -> DomainFunction:
    r"""Matrix anticommutator $\{A,B\}_+=AB+BA$."""
    return _binary_matrix_operation(left, right, operation="anticommutator")


def quantum_bracket(
    left: DomainFunction,
    right: DomainFunction,
    /,
    *,
    hbar: ArrayLike = 1.0,
) -> DomainFunction:
    r"""Scaled quantum bracket $[A,B]/(i\hbar)$."""
    hbar_ = coerce_hbar(hbar)
    return commutator(left, right) / (1j * hbar_)


def hermiticity_residual(operator: DomainFunction, /) -> DomainFunction:
    r"""Residual $A-A^\dagger$ for Hermiticity."""
    if not isinstance(operator, DomainFunction):
        raise TypeError("hermiticity_residual expects a DomainFunction.")
    return operator - adjoint(operator)


def unit_trace_residual(density: DomainFunction, /) -> DomainFunction:
    r"""Residual $\operatorname{tr}(\rho)-1$ for a density operator."""
    if not isinstance(density, DomainFunction):
        raise TypeError("unit_trace_residual expects a DomainFunction.")
    return DomainFunction(
        domain=density.domain,
        deps=density.deps,
        func=_UnitTraceCallable(density),
        metadata={},
    )


__all__ = [
    "anticommutator",
    "commutator",
    "hermiticity_residual",
    "quantum_bracket",
    "unit_trace_residual",
]
