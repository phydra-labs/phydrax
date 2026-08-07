#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.domain import DomainFunction

from ..._strict import StrictModule
from ..differential import dt
from ._algebra import quantum_bracket
from ._validation import (
    coerce_hbar,
    join_function_arguments,
    validate_matrix_value,
    validate_vector_value,
)


_ADEngine = Literal["auto", "reverse", "forward", "jvp"]
HamiltonianAction = Callable[[DomainFunction], DomainFunction]


class _MatrixVectorActionCallable(StrictModule):
    hamiltonian: DomainFunction
    state: DomainFunction
    hamiltonian_positions: tuple[int, ...]
    state_positions: tuple[int, ...]

    def __init__(
        self,
        hamiltonian: DomainFunction,
        state: DomainFunction,
        hamiltonian_positions: tuple[int, ...],
        state_positions: tuple[int, ...],
    ):
        self.hamiltonian = hamiltonian
        self.state = state
        self.hamiltonian_positions = hamiltonian_positions
        self.state_positions = state_positions

    def __call__(self, *args, key=None, **kwargs):
        hamiltonian_args = tuple(args[index] for index in self.hamiltonian_positions)
        state_args = tuple(args[index] for index in self.state_positions)
        hamiltonian = validate_matrix_value(
            self.hamiltonian.func(*hamiltonian_args, key=key, **kwargs),
            role="Hamiltonian",
        )
        state = validate_vector_value(
            self.state.func(*state_args, key=key, **kwargs),
            role="quantum state",
        )
        if int(hamiltonian.shape[1]) != int(state.shape[0]):
            raise ValueError(
                "Hamiltonian and state dimensions must match; "
                f"got {hamiltonian.shape} and {state.shape}."
            )
        return hamiltonian @ state


class _SchrodingerResidualCallable(StrictModule):
    state_derivative: DomainFunction
    action: DomainFunction
    derivative_positions: tuple[int, ...]
    action_positions: tuple[int, ...]
    hbar: Array

    def __init__(
        self,
        state_derivative: DomainFunction,
        action: DomainFunction,
        derivative_positions: tuple[int, ...],
        action_positions: tuple[int, ...],
        hbar: Array,
    ):
        self.state_derivative = state_derivative
        self.action = action
        self.derivative_positions = derivative_positions
        self.action_positions = action_positions
        self.hbar = hbar

    def __call__(self, *args, key=None, **kwargs):
        derivative_args = tuple(args[index] for index in self.derivative_positions)
        action_args = tuple(args[index] for index in self.action_positions)
        derivative = jnp.asarray(
            self.state_derivative.func(*derivative_args, key=key, **kwargs)
        )
        action = jnp.asarray(self.action.func(*action_args, key=key, **kwargs))
        if derivative.shape != action.shape:
            raise ValueError(
                "Hamiltonian action must match the state shape; "
                f"got derivative {derivative.shape} and action {action.shape}."
            )
        return 1j * self.hbar * derivative - action


def _matrix_hamiltonian_action(
    hamiltonian: DomainFunction,
    state: DomainFunction,
    /,
) -> DomainFunction:
    domain, deps, promoted, positions = join_function_arguments(hamiltonian, state)
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_MatrixVectorActionCallable(
            promoted[0],
            promoted[1],
            positions[0],
            positions[1],
        ),
        metadata={},
    )


def schrodinger_residual(
    state: DomainFunction,
    hamiltonian: DomainFunction | HamiltonianAction,
    /,
    *,
    time_var: str = "t",
    hbar: ArrayLike = 1.0,
    mode: Literal["reverse", "forward"] = "reverse",
    ad_engine: _ADEngine = "auto",
) -> DomainFunction:
    r"""Schrödinger residual $i\hbar\,\partial_t\psi-\hat H\psi$."""
    if not isinstance(state, DomainFunction):
        raise TypeError("schrodinger_residual state must be a DomainFunction.")
    if time_var not in state.deps:
        raise ValueError(f"Quantum state must depend on time_var {time_var!r}.")
    hbar_ = coerce_hbar(hbar)

    if isinstance(hamiltonian, DomainFunction):
        action = _matrix_hamiltonian_action(hamiltonian, state)
    elif callable(hamiltonian):
        action = hamiltonian(state)
        if not isinstance(action, DomainFunction):
            raise TypeError(
                "Hamiltonian action must return a DomainFunction; "
                f"got {type(action).__name__}."
            )
    else:
        raise TypeError(
            "hamiltonian must be a matrix-valued DomainFunction or callable action."
        )

    state_derivative = dt(
        state,
        var=time_var,
        mode=mode,
        ad_engine=ad_engine,
    )
    domain, deps, promoted, positions = join_function_arguments(
        state_derivative, action
    )
    return DomainFunction(
        domain=domain,
        deps=deps,
        func=_SchrodingerResidualCallable(
            promoted[0],
            promoted[1],
            positions[0],
            positions[1],
            hbar_,
        ),
        metadata={},
    )


def heisenberg_residual(
    observable: DomainFunction,
    hamiltonian: DomainFunction,
    /,
    *,
    time_var: str = "t",
    hbar: ArrayLike = 1.0,
    mode: Literal["reverse", "forward"] = "reverse",
    ad_engine: _ADEngine = "auto",
) -> DomainFunction:
    r"""Heisenberg residual $\partial_tA-[A,H]/(i\hbar)$."""
    if not isinstance(observable, DomainFunction):
        raise TypeError("heisenberg_residual observable must be a DomainFunction.")
    if not isinstance(hamiltonian, DomainFunction):
        raise TypeError("heisenberg_residual Hamiltonian must be a DomainFunction.")
    if time_var not in observable.deps:
        raise ValueError(f"Observable must depend on time_var {time_var!r}.")
    return dt(
        observable,
        var=time_var,
        mode=mode,
        ad_engine=ad_engine,
    ) - quantum_bracket(observable, hamiltonian, hbar=hbar)


def von_neumann_residual(
    density: DomainFunction,
    hamiltonian: DomainFunction,
    /,
    *,
    time_var: str = "t",
    hbar: ArrayLike = 1.0,
    mode: Literal["reverse", "forward"] = "reverse",
    ad_engine: _ADEngine = "auto",
) -> DomainFunction:
    r"""Von Neumann residual $\partial_t\rho-[H,\rho]/(i\hbar)$."""
    if not isinstance(density, DomainFunction):
        raise TypeError("von_neumann_residual density must be a DomainFunction.")
    if not isinstance(hamiltonian, DomainFunction):
        raise TypeError("von_neumann_residual Hamiltonian must be a DomainFunction.")
    if time_var not in density.deps:
        raise ValueError(f"Density operator must depend on time_var {time_var!r}.")
    return dt(
        density,
        var=time_var,
        mode=mode,
        ad_engine=ad_engine,
    ) - quantum_bracket(hamiltonian, density, hbar=hbar)


__all__ = [
    "HamiltonianAction",
    "heisenberg_residual",
    "schrodinger_residual",
    "von_neumann_residual",
]
