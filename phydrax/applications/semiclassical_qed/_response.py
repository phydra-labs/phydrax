#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import PreparedRealCoordinateTree
from ...solver._differential import DifferentialProblem, DifferentialSolution
from ...solver._diffrax_backend import solve_diffrax
from ._homogeneous import (
    _qed_state_coordinates,
    HomogeneousSpinorQEDPlan,
    HomogeneousSpinorQEDState,
    HomogeneousSpinorQEDVectorField,
    PreparedHomogeneousSpinorQED,
)


_SIGMA_1 = jnp.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=jnp.complex128)


class HomogeneousSpinorQEDTangentState(StrictModule):
    """Base QED state and its directional derivative."""

    base: HomogeneousSpinorQEDState
    vector_potential_tangent: Array
    electric_field_tangent: Array
    mode_spinor_tangent: Array

    def __init__(
        self,
        base: HomogeneousSpinorQEDState,
        vector_potential_tangent: ArrayLike,
        electric_field_tangent: ArrayLike,
        mode_spinor_tangent: ArrayLike,
        /,
    ):
        if not isinstance(base, HomogeneousSpinorQEDState):
            raise TypeError("base must be HomogeneousSpinorQEDState.")
        potential = jnp.asarray(vector_potential_tangent)
        field = jnp.asarray(electric_field_tangent)
        modes = jnp.asarray(mode_spinor_tangent, dtype=base.mode_spinors.dtype)
        if potential.shape != () or field.shape != ():
            raise ValueError("Gauge tangents must be scalar.")
        if modes.shape != base.mode_spinors.shape:
            raise ValueError("Mode tangent shape must match the base modes.")
        self.base = base
        self.vector_potential_tangent = potential
        self.electric_field_tangent = field
        self.mode_spinor_tangent = modes


class HomogeneousSpinorQEDTangentVectorField(StrictModule):
    """Exact directional linearization of the finite Dirac-Maxwell equations."""

    plan: HomogeneousSpinorQEDPlan
    perturbation_current: Any
    perturbation_source_id: str = eqx.field(static=True)

    def adiabatic_current_tangent(
        self, vector_potential: ArrayLike, potential_tangent: ArrayLike, /
    ) -> Array:
        kinetic = self.plan.kinetic_momenta(vector_potential)
        omega = jnp.sqrt(kinetic * kinetic + self.plan.mass * self.plan.mass)
        denominator = jnp.where(omega > 0.0, omega**3, 1.0)
        derivative = self.plan.charge**2 * self.plan.mass**2 / denominator
        derivative = jnp.where(omega > 0.0, derivative, 0.0)
        return jnp.sum(self.plan.quadrature_weights * derivative, axis=-1) * jnp.asarray(
            potential_tangent
        )

    def raw_current_tangent(self, modes: ArrayLike, mode_tangent: ArrayLike, /) -> Array:
        spinors = jnp.asarray(modes)
        tangent = jnp.asarray(mode_tangent)
        per_mode = (
            2.0
            * self.plan.charge
            * jnp.real(
                ein.contract("...ki,ij,...kj->...k", jnp.conj(spinors), _SIGMA_1, tangent)
            )
        )
        return jnp.sum(self.plan.quadrature_weights * per_mode, axis=-1)

    def renormalized_current_tangent(
        self,
        base: HomogeneousSpinorQEDState,
        potential_tangent: ArrayLike,
        mode_tangent: ArrayLike,
        /,
    ) -> Array:
        return self.raw_current_tangent(
            base.mode_spinors, mode_tangent
        ) - self.adiabatic_current_tangent(base.vector_potential, potential_tangent)

    def __call__(
        self,
        time: Array,
        state: HomogeneousSpinorQEDTangentState,
        args: Any,
        /,
    ) -> HomogeneousSpinorQEDTangentState:
        del args
        base_rate = HomogeneousSpinorQEDVectorField(self.plan)(time, state.base, None)
        hamiltonian = self.plan.hamiltonians(state.base.vector_potential)
        delta_hamiltonian = -self.plan.charge * state.vector_potential_tangent * _SIGMA_1
        mode_rate = -1j * (
            ein.contract("kij,kj->ki", hamiltonian, state.mode_spinor_tangent)
            + ein.contract("ij,kj->ki", delta_hamiltonian, state.base.mode_spinors)
        )
        current_rate = self.renormalized_current_tangent(
            state.base,
            state.vector_potential_tangent,
            state.mode_spinor_tangent,
        )
        source_rate = jnp.asarray(self.perturbation_current(time)).reshape(())
        return HomogeneousSpinorQEDTangentState(
            base_rate,
            -state.electric_field_tangent,
            -(current_rate + source_rate),
            mode_rate,
        )


class PreparedHomogeneousSpinorQEDTangent(StrictModule, NonTrainableState):
    """Prepared augmented base/tangent initial-value problem."""

    base: PreparedHomogeneousSpinorQED
    vector_field: HomogeneousSpinorQEDTangentVectorField
    initial_state: HomogeneousSpinorQEDTangentState
    problem: DifferentialProblem
    state_coordinates: PreparedRealCoordinateTree
    perturbation_source_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: PreparedHomogeneousSpinorQED,
        perturbation_current: Any,
        perturbation_source_id: str,
        /,
        *,
        vector_potential_tangent: ArrayLike = 0.0,
        electric_field_tangent: ArrayLike = 0.0,
        mode_spinor_tangent: ArrayLike | None = None,
    ):
        if not isinstance(base, PreparedHomogeneousSpinorQED):
            raise TypeError("base must be PreparedHomogeneousSpinorQED.")
        if not callable(perturbation_current):
            raise TypeError("perturbation_current must be callable.")
        if (
            not isinstance(perturbation_source_id, str)
            or not perturbation_source_id.strip()
        ):
            raise ValueError("perturbation_source_id must be a non-empty string.")
        source_id = perturbation_source_id.strip()
        mode_tangent = (
            jnp.zeros_like(base.initial_state.mode_spinors)
            if mode_spinor_tangent is None
            else jnp.asarray(mode_spinor_tangent)
        )
        state = HomogeneousSpinorQEDTangentState(
            base.initial_state,
            vector_potential_tangent,
            electric_field_tangent,
            mode_tangent,
        )
        state_coordinates = _qed_state_coordinates(state)
        vector_field = HomogeneousSpinorQEDTangentVectorField(
            base.plan, perturbation_current, source_id
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-homogeneous-spinor-qed-tangent",
                "base": base.prepared_id,
                "perturbation_source": source_id,
                "initial_tangent": array_tree_fingerprint(
                    (
                        state.vector_potential_tangent,
                        state.electric_field_tangent,
                        state.mode_spinor_tangent,
                    )
                ),
            }
        )
        problem = DifferentialProblem(
            vector_field,
            state,
            t0=base.problem.t0,
            t1=base.problem.t1,
            problem_id=canonical_fingerprint(
                {
                    "kind": "homogeneous-spinor-qed-tangent-problem",
                    "prepared": prepared_id,
                }
            ),
        )
        self.base = base
        self.vector_field = vector_field
        self.initial_state = state
        self.problem = problem
        self.state_coordinates = state_coordinates
        self.perturbation_source_id = source_id
        self.prepared_id = prepared_id


class HomogeneousSpinorQEDTangentEvidence(StrictModule):
    """Tangent current and causal-source evidence on saved times."""

    renormalized_current_tangent: Array
    perturbation_current: Array
    finite: Array
    causal_initial_condition: Array
    successful: Array


class HomogeneousSpinorQEDTangentResult(StrictModule):
    solution: DifferentialSolution
    evidence: HomogeneousSpinorQEDTangentEvidence
    prepared_id: str = eqx.field(static=True)


def solve_homogeneous_spinor_qed_tangent(
    prepared: PreparedHomogeneousSpinorQEDTangent,
    /,
    *,
    save_times: ArrayLike,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    adjoint: Any | None = None,
    dt0: ArrayLike | None = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    max_steps: int | None = 4096,
) -> HomogeneousSpinorQEDTangentResult:
    if not isinstance(prepared, PreparedHomogeneousSpinorQEDTangent):
        raise TypeError("prepared must be PreparedHomogeneousSpinorQEDTangent.")
    solution = solve_diffrax(
        prepared.problem,
        save_times=save_times,
        solver=solver,
        stepsize_controller=stepsize_controller,
        adjoint=adjoint,
        dt0=dt0,
        state_coordinates=prepared.state_coordinates,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        throw=False,
        solver_configuration_id=canonical_fingerprint(
            {
                "kind": "homogeneous-spinor-qed-tangent-runtime",
                "prepared": prepared.prepared_id,
            }
        ),
    )
    states = solution.states
    if not isinstance(states, HomogeneousSpinorQEDTangentState):
        raise TypeError("Tangent solution lost its typed state tree.")
    current_tangent = prepared.vector_field.renormalized_current_tangent(
        states.base,
        states.vector_potential_tangent,
        states.mode_spinor_tangent,
    )
    sources = jax.vmap(prepared.vector_field.perturbation_current)(solution.times)
    finite = (
        jnp.all(jnp.isfinite(states.vector_potential_tangent))
        & jnp.all(jnp.isfinite(states.electric_field_tangent))
        & jnp.all(jnp.isfinite(states.mode_spinor_tangent))
        & jnp.all(jnp.isfinite(current_tangent))
    )
    initial = (
        jnp.allclose(
            states.vector_potential_tangent[0],
            prepared.initial_state.vector_potential_tangent,
        )
        & jnp.allclose(
            states.electric_field_tangent[0],
            prepared.initial_state.electric_field_tangent,
        )
        & jnp.allclose(
            states.mode_spinor_tangent[0],
            prepared.initial_state.mode_spinor_tangent,
        )
    )
    evidence = HomogeneousSpinorQEDTangentEvidence(
        current_tangent,
        sources,
        finite,
        initial,
        solution.successful & finite & initial,
    )
    return HomogeneousSpinorQEDTangentResult(solution, evidence, prepared.prepared_id)


class TangentFiniteDifferenceEvidence(StrictModule):
    """Central finite-difference comparison for one tangent trajectory."""

    finite_difference: Array
    tangent: Array
    absolute_residual: Array
    relative_residual: Array
    maximum_relative_residual: Array
    successful: Array


def tangent_finite_difference_evidence(
    plus: ArrayLike,
    minus: ArrayLike,
    tangent: ArrayLike,
    epsilon: ArrayLike,
    /,
    *,
    relative_tolerance: float = 1e-4,
    absolute_floor: float = 1e-12,
) -> TangentFiniteDifferenceEvidence:
    positive = jnp.asarray(plus)
    negative = jnp.asarray(minus)
    derivative = jnp.asarray(tangent)
    step = jnp.asarray(epsilon)
    if positive.shape != negative.shape or positive.shape != derivative.shape:
        raise ValueError("Finite-difference and tangent arrays must share one shape.")
    if step.shape != () or not bool(jnp.isfinite(step) & (step > 0.0)):
        raise ValueError("epsilon must be one positive finite scalar.")
    tolerance = float(relative_tolerance)
    floor = float(absolute_floor)
    if not isfinite(tolerance) or tolerance < 0.0 or not isfinite(floor) or floor <= 0.0:
        raise ValueError("Finite-difference tolerances must be finite and physical.")
    finite_difference = (positive - negative) / (2.0 * step)
    absolute = jnp.abs(finite_difference - derivative)
    scale = jnp.maximum(
        floor, jnp.maximum(jnp.abs(finite_difference), jnp.abs(derivative))
    )
    relative = absolute / scale
    maximum = jnp.max(relative)
    finite = jnp.all(jnp.isfinite(relative))
    return TangentFiniteDifferenceEvidence(
        finite_difference,
        derivative,
        absolute,
        relative,
        maximum,
        finite & (maximum <= tolerance),
    )


class RetardedVolterraEvidence(StrictModule):
    """Finite causal-kernel evidence."""

    future_support_residual: Array
    finite: Array
    causal: Array
    successful: Array


class RetardedVolterraResult(StrictModule):
    response: Array
    evidence: RetardedVolterraEvidence
    plan_id: str = eqx.field(static=True)


class RetardedVolterraResponsePlan(StrictModule, NonTrainableState):
    """Fixed-grid retarded Volterra operator using composite trapezoidal weights."""

    times: Array
    kernel: Array
    integration_matrix: Array
    causality_tolerance: float = eqx.field(static=True)
    maximum_time_points: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        kernel: ArrayLike,
        /,
        *,
        causality_tolerance: float = 0.0,
        maximum_time_points: int = 16384,
    ):
        time = np.asarray(times, dtype=np.float64)
        values = np.asarray(kernel)
        tolerance = float(causality_tolerance)
        if (
            time.ndim != 1
            or time.size < 2
            or time.size > int(maximum_time_points)
            or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0.0)
        ):
            raise ValueError("Volterra times must be finite and strictly increasing.")
        if values.shape != (time.size, time.size) or not np.isfinite(values).all():
            raise ValueError("Volterra kernel must be one finite square time matrix.")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("causality_tolerance must be finite and non-negative.")
        future = np.triu(values, k=1)
        future_residual = float(np.max(np.abs(future)))
        if future_residual > tolerance:
            raise ValueError("Retarded kernel has support at future source times.")
        causal_kernel = np.tril(values)
        integration = np.zeros_like(causal_kernel)
        steps = np.diff(time)
        for target in range(1, time.size):
            intervals = np.arange(target)
            integration[target, intervals] += (
                0.5 * steps[intervals] * causal_kernel[target, intervals]
            )
            integration[target, intervals + 1] += (
                0.5 * steps[intervals] * causal_kernel[target, intervals + 1]
            )
        self.times = jnp.asarray(time)
        self.kernel = jnp.asarray(causal_kernel)
        self.integration_matrix = jnp.asarray(integration)
        self.causality_tolerance = tolerance
        self.maximum_time_points = int(maximum_time_points)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "retarded-volterra-response",
                "times": array_tree_fingerprint(time),
                "kernel": array_tree_fingerprint(causal_kernel),
                "causality_tolerance": tolerance,
            }
        )

    def apply(self, source: ArrayLike, /) -> RetardedVolterraResult:
        values = jnp.asarray(source)
        if values.shape[0] != self.times.size:
            raise ValueError("Volterra source must begin with the saved-time axis.")
        response = ein.contract("ij,j...->i...", self.integration_matrix, values)
        future_residual = jnp.max(jnp.abs(jnp.triu(self.kernel, k=1)))
        finite = jnp.all(jnp.isfinite(response))
        causal = future_residual <= self.causality_tolerance
        evidence = RetardedVolterraEvidence(
            future_residual,
            finite,
            causal,
            finite & causal,
        )
        return RetardedVolterraResult(response, evidence, self.plan_id)


__all__ = [
    "HomogeneousSpinorQEDTangentEvidence",
    "HomogeneousSpinorQEDTangentResult",
    "HomogeneousSpinorQEDTangentState",
    "HomogeneousSpinorQEDTangentVectorField",
    "PreparedHomogeneousSpinorQEDTangent",
    "RetardedVolterraEvidence",
    "RetardedVolterraResponsePlan",
    "RetardedVolterraResult",
    "TangentFiniteDifferenceEvidence",
    "solve_homogeneous_spinor_qed_tangent",
    "tangent_finite_difference_evidence",
]
