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

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import HermitianPrecisionPolicy
from ..metrix import (
    RightLieGroupStateGeometry,
    SpecialUnitaryGroup,
    UnitaryGroup,
)
from ..operators.quantum._propagation import unitarity_residual
from ._differential import DifferentialProblem, DifferentialSolution
from ._diffrax_backend import solve_diffrax
from ._geometric import CommutatorFreeSolver
from ._temporal_precision import TemporalPrecisionPolicy


UnitaryGroupKind: TypeAlias = Literal["unitary", "special-unitary"]


class UnitaryPropagatorProblem(StrictModule):
    """Dense time-dependent Hermitian Hamiltonian propagation problem."""

    hamiltonian_function: Callable[[Array, Any], Array]
    initial_propagator: Array
    t0: Array
    t1: Array
    hbar: Array
    args: Any
    temporal_precision: TemporalPrecisionPolicy
    geometry_precision: GeometryPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy
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
        temporal_precision: TemporalPrecisionPolicy | None = None,
        geometry_precision: GeometryPrecisionPolicy | None = None,
        hermitian_precision: HermitianPrecisionPolicy | None = None,
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
        temporal_ = (
            TemporalPrecisionPolicy()
            if temporal_precision is None
            else temporal_precision
        )
        geometry_ = (
            GeometryPrecisionPolicy()
            if geometry_precision is None
            else geometry_precision
        )
        hermitian_ = (
            HermitianPrecisionPolicy()
            if hermitian_precision is None
            else hermitian_precision
        )
        if not isinstance(temporal_, TemporalPrecisionPolicy):
            raise TypeError(
                "temporal_precision must be a TemporalPrecisionPolicy or None."
            )
        if not isinstance(geometry_, GeometryPrecisionPolicy):
            raise TypeError(
                "geometry_precision must be a GeometryPrecisionPolicy or None."
            )
        if not isinstance(hermitian_, HermitianPrecisionPolicy):
            raise TypeError(
                "hermitian_precision must be a HermitianPrecisionPolicy or None."
            )
        start = jnp.asarray(t0, dtype=float)
        end = jnp.asarray(t1, dtype=float)
        hbar_ = temporal_.coefficient(jnp.asarray(hbar, dtype=float))
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
        temporal_.validate_state(initial)
        geometry_.validate_coordinates(initial)
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
        self.temporal_precision = temporal_
        self.geometry_precision = geometry_
        self.hermitian_precision = hermitian_
        self.initial_propagator = initial
        self.group_kind = group_kind
        self.hermiticity_tolerance = float(hermiticity_tolerance)

    def hamiltonian(self, time: ArrayLike, /) -> Array:
        value = self.hermitian_precision.compute(
            self.hamiltonian_function(jnp.asarray(time), self.args)
        )
        expected = (self.dimension, self.dimension)
        if value.shape != expected:
            raise ValueError(
                f"Hamiltonian must have shape {expected}; got {value.shape}."
            )
        residual = self.geometry_precision.decision(
            jnp.max(
                jnp.abs(self.geometry_precision.accumulation(value - jnp.conj(value.T)))
            )
        )
        return eqx.error_if(
            self.hermitian_precision.output(value),
            ~jnp.all(jnp.isfinite(value))
            | (residual > self.geometry_precision.decision(self.hermiticity_tolerance)),
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
    geometry_precision_evidence: PrecisionEvidenceEnvelope
    hermitian_precision_evidence: PrecisionEvidenceEnvelope

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
        geometry_precision_evidence: PrecisionEvidenceEnvelope,
        hermitian_precision_evidence: PrecisionEvidenceEnvelope,
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
        self.geometry_precision_evidence = geometry_precision_evidence
        self.hermitian_precision_evidence = hermitian_precision_evidence


def solve_unitary_propagator(
    problem: UnitaryPropagatorProblem,
    /,
    *,
    save_times: ArrayLike,
    dt0: ArrayLike,
    max_steps: int = 4096,
    precision: TemporalPrecisionPolicy | None = None,
) -> UnitaryPropagatorSolution:
    """Solve a dense unitary propagation problem with a CF Lie integrator."""
    if not isinstance(problem, UnitaryPropagatorProblem):
        raise TypeError("problem must be a UnitaryPropagatorProblem.")
    precision_ = problem.temporal_precision if precision is None else precision
    if not isinstance(precision_, TemporalPrecisionPolicy):
        raise TypeError("precision must be a TemporalPrecisionPolicy or None.")
    if precision_.policy_id != problem.temporal_precision.policy_id:
        raise ValueError("Solve precision must match the unitary problem.")
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
        precision=precision_,
    )
    geometry_precision = problem.geometry_precision
    unitarity = geometry_precision.decision(
        jnp.max(geometry_precision.accumulation(unitarity_residual(solution.states)))
    )
    determinants = jnp.linalg.det(solution.states)
    determinant_residual = (
        geometry_precision.decision(
            jnp.max(jnp.abs(geometry_precision.accumulation(determinants - 1.0)))
        )
        if problem.group_kind == "special-unitary"
        else geometry_precision.decision(0.0)
    )
    hamiltonians = jax.vmap(problem.hamiltonian)(solution.times)
    hermiticity = geometry_precision.decision(
        jnp.max(
            jnp.abs(
                geometry_precision.accumulation(
                    hamiltonians - jnp.swapaxes(jnp.conj(hamiltonians), -1, -2)
                )
            )
        )
    )
    return UnitaryPropagatorSolution(
        solution,
        maximum_unitarity_residual=unitarity,
        maximum_determinant_residual=determinant_residual,
        maximum_hamiltonian_hermiticity_residual=hermiticity,
        group_kind=problem.group_kind,
        hbar=problem.hbar,
        geometry_precision_evidence=problem.geometry_precision.evidence_for(
            problem.initial_propagator
        ),
        hermitian_precision_evidence=problem.hermitian_precision.evidence_for(
            problem.hamiltonian(problem.t0)
        ),
    )


__all__ = [
    "UnitaryGroupKind",
    "UnitaryPropagatorProblem",
    "UnitaryPropagatorSolution",
    "solve_unitary_propagator",
]
