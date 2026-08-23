#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import (
    RightLieGroupStateGeometry,
    SpecialUnitaryGroup,
    UnitaryGroup,
)
from ..operators.quantum._propagation import unitarity_residual
from ._differential import DifferentialProblem, DifferentialSolution
from ._diffrax_backend import solve_diffrax
from ._geometric import CommutatorFreeSolver


UnitaryGroupKind: TypeAlias = Literal["unitary", "special-unitary"]


class UnitaryPropagatorProblem(StrictModule):
    """Dense time-dependent Hermitian Hamiltonian propagation problem."""

    hamiltonian_function: Callable[[Array, Any], Array]
    initial_propagator: Array
    t0: Array
    t1: Array
    hbar: Array
    args: Any
    dimension: int = eqx.field(static=True)
    group_kind: UnitaryGroupKind = eqx.field(static=True)
    hermiticity_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: Callable[[Array, Any], Array],
        dimension: int,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        hbar: ArrayLike = 1.0,
        args: Any = None,
        initial_propagator: ArrayLike | None = None,
        group_kind: UnitaryGroupKind = "unitary",
        hermiticity_tolerance: float = 1e-9,
    ):
        if not callable(hamiltonian):
            raise TypeError("hamiltonian must be callable.")
        dimension_ = int(dimension)
        if dimension_ < 1:
            raise ValueError("dimension must be positive.")
        if group_kind not in ("unitary", "special-unitary"):
            raise ValueError("Unknown unitary group kind.")
        if hermiticity_tolerance < 0.0:
            raise ValueError("hermiticity_tolerance must be non-negative.")
        start = jnp.asarray(t0, dtype=float)
        end = jnp.asarray(t1, dtype=float)
        hbar_ = jnp.asarray(hbar, dtype=float)
        if start.shape != () or end.shape != () or hbar_.shape != ():
            raise ValueError("t0, t1, and hbar must be scalar.")
        hbar_ = eqx.error_if(
            hbar_,
            ~jnp.isfinite(hbar_) | (hbar_ <= 0.0),
            "hbar must be finite and positive.",
        )
        initial = (
            jnp.eye(dimension_, dtype=complex)
            if initial_propagator is None
            else jnp.asarray(initial_propagator)
        )
        expected = (dimension_, dimension_)
        if initial.shape != expected:
            raise ValueError(f"initial_propagator must have shape {expected}.")
        group = (
            UnitaryGroup(dimension_, tolerance=hermiticity_tolerance or 1e-12)
            if group_kind == "unitary"
            else SpecialUnitaryGroup(dimension_, tolerance=hermiticity_tolerance or 1e-12)
        )
        initial = eqx.error_if(
            initial,
            ~group.contains(initial),
            "initial_propagator is outside the selected unitary group.",
        )
        self.hamiltonian_function = hamiltonian
        self.dimension = dimension_
        self.t0 = start
        self.t1 = end
        self.hbar = hbar_
        self.args = args
        self.initial_propagator = initial
        self.group_kind = group_kind
        self.hermiticity_tolerance = float(hermiticity_tolerance)

    def hamiltonian(self, time: ArrayLike, /) -> Array:
        value = jnp.asarray(self.hamiltonian_function(jnp.asarray(time), self.args))
        expected = (self.dimension, self.dimension)
        if value.shape != expected:
            raise ValueError(
                f"Hamiltonian must have shape {expected}; got {value.shape}."
            )
        residual = jnp.max(jnp.abs(value - jnp.conj(value.T)))
        return eqx.error_if(
            value,
            ~jnp.all(jnp.isfinite(value)) | (residual > self.hermiticity_tolerance),
            "Hamiltonian must be finite and Hermitian.",
        )


class _UnitaryPropagatorDrift(StrictModule):
    problem: UnitaryPropagatorProblem

    def __init__(self, problem: UnitaryPropagatorProblem, /):
        self.problem = problem

    def __call__(self, time: Array, state: Array, args: Any) -> Array:
        del args
        hamiltonian = self.problem.hamiltonian(time)
        if self.problem.group_kind == "special-unitary":
            trace = jnp.trace(hamiltonian) / float(self.problem.dimension)
            hamiltonian = hamiltonian - trace * jnp.eye(
                self.problem.dimension, dtype=hamiltonian.dtype
            )
        generator = -1j * hamiltonian / self.problem.hbar
        return generator @ state


class UnitaryPropagatorSolution(StrictModule):
    """Unitary trajectory with structural propagation evidence."""

    differential_solution: DifferentialSolution
    times: Array
    propagators: Array
    valid: Array
    maximum_unitarity_residual: Array
    maximum_determinant_residual: Array
    maximum_hamiltonian_hermiticity_residual: Array
    group_kind: UnitaryGroupKind = eqx.field(static=True)
    hbar: Array

    def __init__(
        self,
        differential_solution: DifferentialSolution,
        /,
        *,
        maximum_unitarity_residual: ArrayLike,
        maximum_determinant_residual: ArrayLike,
        maximum_hamiltonian_hermiticity_residual: ArrayLike,
        group_kind: UnitaryGroupKind,
        hbar: ArrayLike,
    ):
        self.differential_solution = differential_solution
        self.times = differential_solution.times
        self.propagators = differential_solution.states
        self.maximum_unitarity_residual = jnp.asarray(maximum_unitarity_residual)
        self.maximum_determinant_residual = jnp.asarray(maximum_determinant_residual)
        self.maximum_hamiltonian_hermiticity_residual = jnp.asarray(
            maximum_hamiltonian_hermiticity_residual
        )
        self.valid = (
            jnp.asarray(differential_solution.valid, dtype=bool)
            & jnp.isfinite(self.maximum_unitarity_residual)
            & jnp.isfinite(self.maximum_hamiltonian_hermiticity_residual)
        )
        self.group_kind = group_kind
        self.hbar = jnp.asarray(hbar)


def solve_unitary_propagator(
    problem: UnitaryPropagatorProblem,
    /,
    *,
    save_times: ArrayLike,
    dt0: ArrayLike,
    max_steps: int = 4096,
) -> UnitaryPropagatorSolution:
    """Solve a dense unitary propagation problem with a CF Lie integrator."""
    if not isinstance(problem, UnitaryPropagatorProblem):
        raise TypeError("problem must be a UnitaryPropagatorProblem.")
    group = (
        UnitaryGroup(
            problem.dimension, tolerance=max(problem.hermiticity_tolerance, 1e-12)
        )
        if problem.group_kind == "unitary"
        else SpecialUnitaryGroup(
            problem.dimension, tolerance=max(problem.hermiticity_tolerance, 1e-12)
        )
    )
    geometry = RightLieGroupStateGeometry(group)
    differential_problem = DifferentialProblem(
        _UnitaryPropagatorDrift(problem),
        problem.initial_propagator,
        t0=problem.t0,
        t1=problem.t1,
        args=None,
        state_geometry=geometry,
    )
    solution = solve_diffrax(
        differential_problem,
        save_times=save_times,
        solver=CommutatorFreeSolver(geometry),
        dt0=dt0,
        max_steps=max_steps,
        throw=False,
    )
    unitarity = jnp.max(unitarity_residual(solution.states))
    determinants = jnp.linalg.det(solution.states)
    determinant_residual = (
        jnp.max(jnp.abs(determinants - 1.0))
        if problem.group_kind == "special-unitary"
        else jnp.asarray(0.0, dtype=unitarity.dtype)
    )
    hamiltonians = jax.vmap(problem.hamiltonian)(solution.times)
    hermiticity = jnp.max(
        jnp.abs(hamiltonians - jnp.swapaxes(jnp.conj(hamiltonians), -1, -2))
    )
    return UnitaryPropagatorSolution(
        solution,
        maximum_unitarity_residual=unitarity,
        maximum_determinant_residual=determinant_residual,
        maximum_hamiltonian_hermiticity_residual=hermiticity,
        group_kind=problem.group_kind,
        hbar=problem.hbar,
    )


__all__ = [
    "UnitaryGroupKind",
    "UnitaryPropagatorProblem",
    "UnitaryPropagatorSolution",
    "solve_unitary_propagator",
]
