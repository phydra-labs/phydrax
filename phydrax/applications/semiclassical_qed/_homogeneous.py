#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import linear_interpolate
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    ArraySpace,
    ComplexCartesianCoordinates,
    prepare_real_coordinate_tree,
    PreparedRealCoordinateTree,
)
from ...solver._differential import DifferentialProblem, DifferentialSolution
from ...solver._diffrax_backend import solve_diffrax


_SIGMA_1 = jnp.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=jnp.complex128)
_SIGMA_3 = jnp.asarray(((1.0, 0.0), (0.0, -1.0)), dtype=jnp.complex128)


def _qed_state_coordinates(state: Any, /) -> PreparedRealCoordinateTree:
    """Bind every complex state leaf to native Cartesian real coordinates."""

    treedef = jax.tree.structure(state)
    leaves = tuple(jnp.asarray(leaf) for leaf in jax.tree.leaves(state))
    maps = tuple(
        ComplexCartesianCoordinates(ArraySpace(leaf.shape, dtype=leaf.dtype))
        if jnp.issubdtype(leaf.dtype, jnp.complexfloating)
        else None
        for leaf in leaves
    )
    return prepare_real_coordinate_tree(
        state,
        jax.tree.unflatten(treedef, maps),
    )


class SemiclassicalQEDEvidenceStatus(IntFlag):
    """Bitwise fail-closed status of a homogeneous semiclassical trajectory."""

    SUCCESS = 0
    SOLVER_FAILURE = 1
    NONFINITE = 2
    MODE_NORMALIZATION_FAILURE = 4
    ENERGY_BALANCE_FAILURE = 8
    WARD_IDENTITY_FAILURE = 16


class ZeroExternalCurrent(StrictModule, NonTrainableState):
    """The exactly zero homogeneous external-current source."""

    source_id: str = eqx.field(static=True)

    def __init__(self):
        self.source_id = canonical_fingerprint({"kind": "zero-external-current"})

    def __call__(self, time: ArrayLike, /) -> Array:
        return jnp.zeros_like(jnp.asarray(time, dtype=jnp.float64))


class TabulatedExternalCurrent(StrictModule, NonTrainableState):
    """Piecewise-linear external current with constant endpoint extension."""

    times: Array
    values: Array
    source_id: str = eqx.field(static=True)

    def __init__(self, times: ArrayLike, values: ArrayLike, /):
        time = np.asarray(times, dtype=np.float64)
        current = np.asarray(values, dtype=np.float64)
        if (
            time.ndim != 1
            or time.size < 2
            or current.shape != time.shape
            or not np.isfinite(time).all()
            or not np.isfinite(current).all()
            or np.any(np.diff(time) <= 0.0)
        ):
            raise ValueError(
                "Tabulated external current needs aligned finite increasing samples."
            )
        self.times = jnp.asarray(time)
        self.values = jnp.asarray(current)
        self.source_id = canonical_fingerprint(
            {
                "kind": "tabulated-external-current",
                "times": array_tree_fingerprint(time),
                "values": array_tree_fingerprint(current),
            }
        )

    def __call__(self, time: ArrayLike, /) -> Array:
        return linear_interpolate(self.times, self.values, jnp.asarray(time)).values


class HomogeneousSpinorQEDState(StrictModule):
    """Temporal-gauge vector potential, electric field, and occupied spinor modes."""

    vector_potential: Array
    electric_field: Array
    mode_spinors: Array

    def __init__(
        self,
        vector_potential: ArrayLike,
        electric_field: ArrayLike,
        mode_spinors: ArrayLike,
        /,
    ):
        potential = jnp.asarray(vector_potential)
        field = jnp.asarray(electric_field)
        modes = jnp.asarray(mode_spinors)
        if potential.shape != () or field.shape != ():
            raise ValueError(
                "Homogeneous gauge potential and electric field must be scalar."
            )
        if modes.ndim != 2 or modes.shape[-1] != 2:
            raise ValueError("Homogeneous spinor modes must have shape (modes, 2).")
        dtype = jnp.result_type(modes.dtype, jnp.complex128)
        self.vector_potential = potential.astype(
            jnp.real(jnp.zeros((), dtype=dtype)).dtype
        )
        self.electric_field = field.astype(jnp.real(jnp.zeros((), dtype=dtype)).dtype)
        self.mode_spinors = modes.astype(dtype)


class HomogeneousSpinorQEDEvidence(StrictModule):
    """Resolved renormalization, normalization, energy, and Ward evidence."""

    raw_current: Array
    adiabatic_current: Array
    renormalized_current: Array
    mode_norm_residuals: Array
    renormalized_fermion_energy: Array
    electromagnetic_energy: Array
    total_energy: Array
    external_work: Array
    energy_balance_residual: Array
    ward_residual: Array
    maximum_mode_norm_residual: Array
    maximum_energy_balance_residual: Array
    maximum_ward_residual: Array
    status: Array
    successful: Array


class HomogeneousSpinorQEDResult(StrictModule):
    """Differential solution plus explicit semiclassical conservation evidence."""

    solution: DifferentialSolution
    evidence: HomogeneousSpinorQEDEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def states(self) -> HomogeneousSpinorQEDState:
        states = self.solution.states
        if not isinstance(states, HomogeneousSpinorQEDState):
            raise TypeError("Homogeneous QED solution lost its typed state tree.")
        return states


class HomogeneousSpinorQEDPlan(StrictModule, NonTrainableState):
    """Finite momentum-regulated 1+1 Dirac modes with Maxwell backreaction.

    Each path represents one occupied negative-energy spinor. The adiabatic
    subtraction is the instantaneous negative-energy reference using the same
    finite quadrature, so current and energy subtraction obey one derivative
    identity at finite cutoff.
    """

    momenta: Array
    quadrature_weights: Array
    charge: Array
    mass: Array
    external_current: Any
    mode_norm_tolerance: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    ward_tolerance: float = eqx.field(static=True)
    maximum_modes: int = eqx.field(static=True)
    external_source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        momenta: ArrayLike,
        quadrature_weights: ArrayLike,
        /,
        *,
        charge: ArrayLike,
        mass: ArrayLike,
        external_current: Any | None = None,
        external_source_id: str | None = None,
        mode_norm_tolerance: float = 1e-8,
        energy_tolerance: float = 1e-7,
        ward_tolerance: float = 1e-8,
        maximum_modes: int = 2**18,
    ):
        momentum = np.asarray(momenta, dtype=np.float64)
        weights = np.asarray(quadrature_weights, dtype=np.float64)
        q = np.asarray(charge, dtype=np.float64)
        m = np.asarray(mass, dtype=np.float64)
        if momentum.ndim != 1 or momentum.size < 1 or not np.isfinite(momentum).all():
            raise ValueError("momenta must be one non-empty finite vector.")
        if weights.shape != momentum.shape or not np.all(
            np.isfinite(weights) & (weights > 0.0)
        ):
            raise ValueError("quadrature_weights must be positive and align with modes.")
        if momentum.size > int(maximum_modes):
            raise MemoryError("Homogeneous spinor QED exceeds maximum_modes.")
        if q.shape != () or not np.isfinite(q):
            raise ValueError("charge must be one finite scalar.")
        if m.shape != () or not np.isfinite(m) or float(m) < 0.0:
            raise ValueError("mass must be one finite non-negative scalar.")
        tolerances = (
            float(mode_norm_tolerance),
            float(energy_tolerance),
            float(ward_tolerance),
        )
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("QED evidence tolerances must be finite and non-negative.")
        if external_current is None:
            source = ZeroExternalCurrent()
            if external_source_id is not None:
                raise ValueError("The zero source has a canonical source ID.")
            source_id = source.source_id
        else:
            if not callable(external_current):
                raise TypeError("external_current must be callable or None.")
            source = external_current
            if external_source_id is None:
                raise ValueError("Opaque external currents require external_source_id.")
            if not isinstance(external_source_id, str) or not external_source_id.strip():
                raise ValueError("external_source_id must be a non-empty string.")
            source_id = external_source_id.strip()
        self.momenta = jnp.asarray(momentum)
        self.quadrature_weights = jnp.asarray(weights)
        self.charge = jnp.asarray(q)
        self.mass = jnp.asarray(m)
        self.external_current = source
        self.mode_norm_tolerance = tolerances[0]
        self.energy_tolerance = tolerances[1]
        self.ward_tolerance = tolerances[2]
        self.maximum_modes = int(maximum_modes)
        self.external_source_id = source_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "homogeneous-spinor-semiclassical-qed-1p1",
                "momenta": array_tree_fingerprint(momentum),
                "quadrature_weights": array_tree_fingerprint(weights),
                "charge": array_tree_fingerprint(q),
                "mass": array_tree_fingerprint(m),
                "external_source_id": source_id,
                "tolerances": tolerances,
            }
        )

    def kinetic_momenta(self, vector_potential: ArrayLike, /) -> Array:
        return self.momenta - self.charge * jnp.asarray(vector_potential)[..., None]

    def hamiltonians(self, vector_potential: ArrayLike, /) -> Array:
        kinetic = self.kinetic_momenta(vector_potential)
        return kinetic[..., :, None, None] * _SIGMA_1 + self.mass * _SIGMA_3

    def adiabatic_current(self, vector_potential: ArrayLike, /) -> Array:
        kinetic = self.kinetic_momenta(vector_potential)
        omega = jnp.sqrt(kinetic * kinetic + self.mass * self.mass)
        ratio = jnp.where(omega > 0.0, kinetic / omega, 0.0)
        return jnp.sum(self.quadrature_weights * (-self.charge * ratio), axis=-1)

    def raw_current(self, mode_spinors: ArrayLike, /) -> Array:
        modes = jnp.asarray(mode_spinors)
        per_mode = self.charge * jnp.real(
            ein.contract("...ki,ij,...kj->...k", jnp.conj(modes), _SIGMA_1, modes)
        )
        return jnp.sum(self.quadrature_weights * per_mode, axis=-1)

    def renormalized_current(
        self, vector_potential: ArrayLike, mode_spinors: ArrayLike, /
    ) -> Array:
        return self.raw_current(mode_spinors) - self.adiabatic_current(vector_potential)

    def prepare(
        self,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        vector_potential: ArrayLike = 0.0,
        electric_field: ArrayLike = 0.0,
        mode_spinors: ArrayLike | None = None,
    ) -> PreparedHomogeneousSpinorQED:
        return PreparedHomogeneousSpinorQED(
            self,
            t0=t0,
            t1=t1,
            vector_potential=vector_potential,
            electric_field=electric_field,
            mode_spinors=mode_spinors,
        )


class HomogeneousSpinorQEDVectorField(StrictModule):
    """JAX-compatible coupled Dirac-Maxwell right-hand side."""

    plan: HomogeneousSpinorQEDPlan

    def __call__(
        self,
        time: Array,
        state: HomogeneousSpinorQEDState,
        args: Any,
        /,
    ) -> HomogeneousSpinorQEDState:
        del args
        hamiltonians = self.plan.hamiltonians(state.vector_potential)
        mode_rate = -1j * ein.contract("kij,kj->ki", hamiltonians, state.mode_spinors)
        current = self.plan.renormalized_current(
            state.vector_potential, state.mode_spinors
        )
        external = jnp.asarray(self.plan.external_current(time)).reshape(())
        return HomogeneousSpinorQEDState(
            -state.electric_field,
            -(current + external),
            mode_rate,
        )


def negative_energy_spinor_modes(
    momenta: ArrayLike,
    /,
    *,
    mass: ArrayLike,
    charge: ArrayLike = 0.0,
    vector_potential: ArrayLike = 0.0,
) -> Array:
    """Analytic normalized negative-energy 1+1 Dirac spinors, including m=0."""

    momentum = jnp.asarray(momenta)
    m = jnp.asarray(mass)
    q = jnp.asarray(charge)
    potential = jnp.asarray(vector_potential)
    kinetic = momentum - q * potential
    omega = jnp.sqrt(kinetic * kinetic + m * m)
    candidate = jnp.stack((-kinetic, m + omega), axis=-1).astype("complex128")
    degenerate = omega == 0.0
    convention = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0), dtype=candidate.dtype), candidate.shape
    )
    candidate = jnp.where(degenerate[..., None], convention, candidate)
    norm = jnp.sqrt(jnp.sum(jnp.abs(candidate) ** 2, axis=-1, keepdims=True))
    return candidate / norm


class PreparedHomogeneousSpinorQED(StrictModule, NonTrainableState):
    """Fixed finite QED initial-value problem, separate from runtime solver choices."""

    plan: HomogeneousSpinorQEDPlan
    initial_state: HomogeneousSpinorQEDState
    problem: DifferentialProblem
    state_coordinates: PreparedRealCoordinateTree
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: HomogeneousSpinorQEDPlan,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        vector_potential: ArrayLike,
        electric_field: ArrayLike,
        mode_spinors: ArrayLike | None,
    ):
        if not isinstance(plan, HomogeneousSpinorQEDPlan):
            raise TypeError("plan must be a HomogeneousSpinorQEDPlan.")
        modes = (
            negative_energy_spinor_modes(
                plan.momenta,
                mass=plan.mass,
                charge=plan.charge,
                vector_potential=vector_potential,
            )
            if mode_spinors is None
            else jnp.asarray(mode_spinors)
        )
        if modes.shape != (plan.momenta.size, 2):
            raise ValueError("mode_spinors must have shape (mode_count, 2).")
        norms = np.sum(np.abs(np.asarray(modes)) ** 2, axis=-1)
        if not np.isfinite(norms).all() or np.any(norms <= 0.0):
            raise ValueError("Every initial spinor mode must have finite positive norm.")
        modes = modes / jnp.sqrt(jnp.sum(jnp.abs(modes) ** 2, axis=-1, keepdims=True))
        state = HomogeneousSpinorQEDState(vector_potential, electric_field, modes)
        state_coordinates = _qed_state_coordinates(state)
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-homogeneous-spinor-qed",
                "plan": plan.plan_id,
                "t0": array_tree_fingerprint(np.asarray(t0)),
                "t1": array_tree_fingerprint(np.asarray(t1)),
                "initial_state": array_tree_fingerprint(state),
            }
        )
        problem = DifferentialProblem(
            HomogeneousSpinorQEDVectorField(plan),
            state,
            t0=t0,
            t1=t1,
            problem_id=canonical_fingerprint(
                {"kind": "homogeneous-spinor-qed-problem", "prepared": prepared_id}
            ),
        )
        self.plan = plan
        self.initial_state = state
        self.problem = problem
        self.state_coordinates = state_coordinates
        self.prepared_id = prepared_id


def _trajectory_evidence(
    prepared: PreparedHomogeneousSpinorQED,
    solution: DifferentialSolution,
    /,
) -> HomogeneousSpinorQEDEvidence:
    states = solution.states
    if not isinstance(states, HomogeneousSpinorQEDState):
        raise TypeError("Homogeneous QED solution lost its typed state tree.")
    plan = prepared.plan
    modes = states.mode_spinors
    potential = states.vector_potential
    field = states.electric_field
    hamiltonian = plan.hamiltonians(potential)
    raw_current = plan.raw_current(modes)
    adiabatic_current = plan.adiabatic_current(potential)
    current = raw_current - adiabatic_current
    mode_norm_residuals = jnp.abs(jnp.sum(jnp.abs(modes) ** 2, axis=-1) - 1.0)
    raw_fermion_energy = jnp.sum(
        plan.quadrature_weights
        * jnp.real(
            ein.contract(
                "...ki,...kij,...kj->...k",
                jnp.conj(modes),
                hamiltonian,
                modes,
            )
        ),
        axis=-1,
    )
    kinetic = plan.kinetic_momenta(potential)
    omega = jnp.sqrt(kinetic * kinetic + plan.mass * plan.mass)
    adiabatic_energy = jnp.sum(plan.quadrature_weights * (-omega), axis=-1)
    renormalized_fermion_energy = raw_fermion_energy - adiabatic_energy
    electromagnetic_energy = 0.5 * field * field
    total_energy = renormalized_fermion_energy + electromagnetic_energy
    external = jax.vmap(plan.external_current)(solution.times)
    power = field * external
    increments = 0.5 * (power[1:] + power[:-1]) * jnp.diff(solution.times)
    external_work = jnp.concatenate(
        (jnp.zeros((1,), dtype=total_energy.dtype), jnp.cumsum(increments))
    )
    energy_balance_residual = total_energy - total_energy[0] + external_work
    charge_density = plan.charge * jnp.sum(
        plan.quadrature_weights * jnp.sum(jnp.abs(modes) ** 2, axis=-1),
        axis=-1,
    )
    ward_steps = jnp.diff(charge_density) / jnp.diff(solution.times)
    ward_residual = jnp.concatenate((jnp.zeros((1,), dtype=ward_steps.dtype), ward_steps))
    maximum_norm = jnp.max(mode_norm_residuals)
    maximum_energy = jnp.max(jnp.abs(energy_balance_residual))
    maximum_ward = jnp.max(jnp.abs(ward_residual))
    finite = (
        jnp.all(jnp.isfinite(modes))
        & jnp.all(jnp.isfinite(current))
        & jnp.all(jnp.isfinite(total_energy))
    )
    status = jnp.asarray(int(SemiclassicalQEDEvidenceStatus.SUCCESS), dtype=jnp.int32)
    status = status | jnp.where(
        solution.successful,
        0,
        int(SemiclassicalQEDEvidenceStatus.SOLVER_FAILURE),
    ).astype(jnp.int32)
    status = status | jnp.where(
        finite, 0, int(SemiclassicalQEDEvidenceStatus.NONFINITE)
    ).astype(jnp.int32)
    status = status | jnp.where(
        maximum_norm <= plan.mode_norm_tolerance,
        0,
        int(SemiclassicalQEDEvidenceStatus.MODE_NORMALIZATION_FAILURE),
    ).astype(jnp.int32)
    energy_scale = jnp.maximum(1.0, jnp.max(jnp.abs(total_energy)))
    status = status | jnp.where(
        maximum_energy <= plan.energy_tolerance * energy_scale,
        0,
        int(SemiclassicalQEDEvidenceStatus.ENERGY_BALANCE_FAILURE),
    ).astype(jnp.int32)
    status = status | jnp.where(
        maximum_ward <= plan.ward_tolerance,
        0,
        int(SemiclassicalQEDEvidenceStatus.WARD_IDENTITY_FAILURE),
    ).astype(jnp.int32)
    return HomogeneousSpinorQEDEvidence(
        raw_current,
        adiabatic_current,
        current,
        mode_norm_residuals,
        renormalized_fermion_energy,
        electromagnetic_energy,
        total_energy,
        external_work,
        energy_balance_residual,
        ward_residual,
        maximum_norm,
        maximum_energy,
        maximum_ward,
        status,
        status == int(SemiclassicalQEDEvidenceStatus.SUCCESS),
    )


def solve_homogeneous_spinor_qed(
    prepared: PreparedHomogeneousSpinorQED,
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
) -> HomogeneousSpinorQEDResult:
    """Run the prepared finite model with the existing differential backend."""

    if not isinstance(prepared, PreparedHomogeneousSpinorQED):
        raise TypeError("prepared must be PreparedHomogeneousSpinorQED.")
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
            {"kind": "homogeneous-spinor-qed-runtime", "prepared": prepared.prepared_id}
        ),
    )
    return HomogeneousSpinorQEDResult(
        solution,
        _trajectory_evidence(prepared, solution),
        prepared.prepared_id,
    )


__all__ = [
    "HomogeneousSpinorQEDEvidence",
    "SemiclassicalQEDEvidenceStatus",
    "HomogeneousSpinorQEDPlan",
    "HomogeneousSpinorQEDResult",
    "HomogeneousSpinorQEDState",
    "HomogeneousSpinorQEDVectorField",
    "PreparedHomogeneousSpinorQED",
    "TabulatedExternalCurrent",
    "ZeroExternalCurrent",
    "negative_energy_spinor_modes",
    "solve_homogeneous_spinor_qed",
]
