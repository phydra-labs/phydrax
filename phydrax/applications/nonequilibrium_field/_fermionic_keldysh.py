#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fermionic Keldysh functions and a bounded self-consistent second-Born control."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...linalg import inverse
from ._keldysh import ClosedTimePathGrid


class FermionicKeldyshFunctions(StrictModule):
    """Real-time functions with ``G<=i<c†c>`` and ``G>=-i<cc†>``."""

    lesser: Array
    greater: Array
    retarded: Array
    advanced: Array
    keldysh: Array
    finite: Array
    grid_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)


class FermionicKeldyshIdentityEvidence(StrictModule, NonTrainableState):
    lesser_antihermiticity_residual: Array
    greater_antihermiticity_residual: Array
    equal_time_car_residual: Array
    retarded_causality_residual: Array
    advanced_causality_residual: Array
    retarded_advanced_residual: Array
    finite: Array
    satisfied: Array
    source_id: str = eqx.field(static=True)


class FermionicSecondBornSelfEnergy(StrictModule):
    lesser: Array
    greater: Array
    retarded: Array
    advanced: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)


class FermionicKadanoffBaymEvidence(StrictModule, NonTrainableState):
    fixed_point_residual: Array
    schwinger_dyson_residual: Array
    kadanoff_baym_residual: Array
    particle_number: Array
    particle_number_drift: Array
    discrete_energy: Array
    relative_energy_drift: Array
    car_residual: Array
    causality_residual: Array
    finite: Array
    converged: Array
    conserved: Array
    successful: Array
    iterations: Array
    prepared_id: str = eqx.field(static=True)


class FermionicSecondBornResult(StrictModule):
    functions: FermionicKeldyshFunctions
    self_energy: FermionicSecondBornSelfEnergy
    evidence: FermionicKadanoffBaymEvidence
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def fermionic_keldysh_from_propagators(
    grid: ClosedTimePathGrid,
    propagators: ArrayLike,
    initial_density: ArrayLike,
    /,
    *,
    source_id: str,
) -> FermionicKeldyshFunctions:
    """Construct free fermionic two-time functions from ``U(t,t0)`` and ``rho0``."""
    if not isinstance(grid, ClosedTimePathGrid):
        raise TypeError("grid must be ClosedTimePathGrid.")
    propagator = jnp.asarray(propagators)
    density = jnp.asarray(initial_density)
    time_count = int(grid.plan.time_nodes.size)
    if (
        propagator.ndim != 3
        or propagator.shape[0] != time_count
        or propagator.shape[1] != propagator.shape[2]
        or density.shape != propagator.shape[1:]
    ):
        raise ValueError(
            "propagators and initial_density have incompatible finite shapes."
        )
    if not str(source_id):
        raise ValueError("source_id must be non-empty.")
    modes = int(density.shape[0])
    identity = jnp.eye(modes, dtype=jnp.result_type(propagator, density, complex))
    lesser = 1j * contract(
        "tai,ij,sbj->tsab", propagator, density, jnp.conj(propagator), backend="jax"
    )
    greater = -1j * contract(
        "tai,ij,sbj->tsab",
        propagator,
        identity - density,
        jnp.conj(propagator),
        backend="jax",
    )
    difference = greater - lesser
    causal = grid.causal_mask[..., None, None]
    retarded = jnp.where(causal, difference, 0.0)
    advanced = -jnp.where(jnp.swapaxes(causal, 0, 1), difference, 0.0)
    finite = (
        jnp.all(jnp.isfinite(propagator))
        & jnp.all(jnp.isfinite(density))
        & jnp.all(jnp.isfinite(lesser))
        & jnp.all(jnp.isfinite(greater))
    )
    return FermionicKeldyshFunctions(
        lesser,
        greater,
        retarded,
        advanced,
        lesser + greater,
        finite,
        grid.grid_id,
        str(source_id),
    )


def fermionic_keldysh_identity_evidence(
    grid: ClosedTimePathGrid,
    functions: FermionicKeldyshFunctions,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> FermionicKeldyshIdentityEvidence:
    if not isinstance(grid, ClosedTimePathGrid) or not isinstance(
        functions, FermionicKeldyshFunctions
    ):
        raise TypeError("grid and functions must be fermionic Keldysh values.")
    if functions.grid_id != grid.grid_id:
        raise ValueError("Fermionic functions belong to another time grid.")
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    swap_lesser = jnp.conj(jnp.transpose(functions.lesser, (1, 0, 3, 2)))
    swap_greater = jnp.conj(jnp.transpose(functions.greater, (1, 0, 3, 2)))
    lesser_residual = jnp.max(jnp.abs(functions.lesser + swap_lesser))
    greater_residual = jnp.max(jnp.abs(functions.greater + swap_greater))
    difference = functions.greater - functions.lesser
    diagonal = jnp.stack(
        [difference[index, index] for index in range(difference.shape[0])]
    )
    car = jnp.max(jnp.abs(1j * diagonal - jnp.eye(difference.shape[-1])))
    future = ~grid.causal_mask
    retarded_causality = jnp.max(
        jnp.abs(jnp.where(future[..., None, None], functions.retarded, 0.0))
    )
    advanced_causality = jnp.max(
        jnp.abs(
            jnp.where(
                jnp.swapaxes(future, 0, 1)[..., None, None], functions.advanced, 0.0
            )
        )
    )
    advanced_identity = jnp.max(
        jnp.abs(
            functions.advanced - jnp.conj(jnp.transpose(functions.retarded, (1, 0, 3, 2)))
        )
    )
    residuals = jnp.stack(
        (
            lesser_residual,
            greater_residual,
            car,
            retarded_causality,
            advanced_causality,
            advanced_identity,
        )
    )
    finite = functions.finite & jnp.all(jnp.isfinite(residuals))
    return FermionicKeldyshIdentityEvidence(
        lesser_residual,
        greater_residual,
        car,
        retarded_causality,
        advanced_causality,
        advanced_identity,
        finite,
        finite & (jnp.max(residuals) <= tolerance_),
        functions.source_id,
    )


class FermionicSecondBornPlan(StrictModule, NonTrainableState):
    """Spin-degenerate local-Hubbard, closed-system second-Born Dyson control."""

    grid: ClosedTimePathGrid
    one_particle_hamiltonian: Array
    interaction: Array
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    maximum_matrix_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: ClosedTimePathGrid,
        one_particle_hamiltonian: ArrayLike,
        interaction: ArrayLike,
        /,
        *,
        maximum_iterations: int = 64,
        tolerance: float = 1.0e-8,
        damping: float = 0.5,
        conservation_tolerance: float = 1.0e-6,
        maximum_matrix_elements: int = 4_000_000,
    ):
        if not isinstance(grid, ClosedTimePathGrid):
            raise TypeError("grid must be ClosedTimePathGrid.")
        hamiltonian = np.asarray(one_particle_hamiltonian, dtype=complex)
        coupling = np.asarray(interaction, dtype=float)
        modes = int(hamiltonian.shape[0]) if hamiltonian.ndim == 2 else 0
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        damping_ = float(damping)
        conservation = float(conservation_tolerance)
        capacity = int(maximum_matrix_elements)
        dimension = int(grid.plan.time_nodes.size) * modes
        required = 18 * dimension * dimension
        if (
            hamiltonian.shape != (modes, modes)
            or modes == 0
            or coupling.shape != (modes,)
            or np.any(~np.isfinite(hamiltonian))
            or not np.allclose(hamiltonian, hamiltonian.conj().T)
            or np.any(~np.isfinite(coupling))
            or np.any(coupling < 0.0)
            or iterations <= 0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not np.isfinite(damping_)
            or not 0.0 < damping_ <= 1.0
            or not np.isfinite(conservation)
            or conservation <= 0.0
            or capacity <= 0
            or required > capacity
        ):
            raise ValueError(
                "Fermionic second-Born model or fixed matrix budget is invalid."
            )
        self.grid = grid
        self.one_particle_hamiltonian = jnp.asarray(hamiltonian)
        self.interaction = jnp.asarray(coupling)
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.damping = damping_
        self.conservation_tolerance = conservation
        self.maximum_matrix_elements = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "closed-system-self-consistent-fermionic-second-born",
                "grid": grid.grid_id,
                "hamiltonian": array_tree_fingerprint(hamiltonian),
                "interaction": array_tree_fingerprint(coupling),
                "maximum_iterations": iterations,
                "tolerance": tolerance_,
                "damping": damping_,
                "conservation_tolerance": conservation,
                "maximum_matrix_elements": capacity,
            }
        )

    def prepare(
        self, free_functions: FermionicKeldyshFunctions, /
    ) -> "PreparedFermionicSecondBorn":
        if (
            not isinstance(free_functions, FermionicKeldyshFunctions)
            or free_functions.grid_id != self.grid.grid_id
        ):
            raise ValueError("free_functions must belong to this plan's time grid.")
        if free_functions.lesser.shape[-2:] != self.one_particle_hamiltonian.shape:
            raise ValueError("Free fermionic mode axes do not match the Hamiltonian.")
        return PreparedFermionicSecondBorn(self, free_functions)


class PreparedFermionicSecondBorn(StrictModule, NonTrainableState):
    __hash__ = object.__hash__

    plan: FermionicSecondBornPlan
    free_functions: FermionicKeldyshFunctions
    quadrature_weights: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, free_functions, /):
        times = np.asarray(plan.grid.plan.time_nodes)
        differences = np.diff(times)
        weights = np.empty_like(times)
        weights[0], weights[-1] = 0.5 * differences[0], 0.5 * differences[-1]
        if times.size > 2:
            weights[1:-1] = 0.5 * (differences[:-1] + differences[1:])
        self.plan = plan
        self.free_functions = free_functions
        self.quadrature_weights = jnp.asarray(weights)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fermionic-second-born",
                "plan": plan.plan_id,
                "free": free_functions.source_id,
            }
        )

    @staticmethod
    def _matrix(values: Array, /) -> Array:
        return jnp.transpose(values, (0, 2, 1, 3)).reshape(
            (values.shape[0] * values.shape[2],) * 2
        )

    @staticmethod
    def _tensor(values: Array, time_count: int, mode_count: int, /) -> Array:
        return jnp.transpose(
            values.reshape((time_count, mode_count, time_count, mode_count)), (0, 2, 1, 3)
        )

    def _self_energy(
        self, functions: FermionicKeldyshFunctions, /
    ) -> FermionicSecondBornSelfEnergy:
        lesser_diagonal = jnp.diagonal(functions.lesser, axis1=-2, axis2=-1)
        greater_diagonal = jnp.diagonal(functions.greater, axis1=-2, axis2=-1)
        reverse_lesser = jnp.swapaxes(lesser_diagonal, 0, 1)
        reverse_greater = jnp.swapaxes(greater_diagonal, 0, 1)
        sigma_lesser_diagonal = (
            self.plan.interaction**2 * lesser_diagonal**2 * reverse_greater
        )
        sigma_greater_diagonal = (
            self.plan.interaction**2 * greater_diagonal**2 * reverse_lesser
        )
        identity = jnp.eye(self.plan.interaction.size, dtype=functions.lesser.dtype)
        lesser = sigma_lesser_diagonal[..., :, None] * identity
        greater = sigma_greater_diagonal[..., :, None] * identity
        difference = greater - lesser
        causal = self.plan.grid.causal_mask[..., None, None]
        retarded = jnp.where(causal, difference, 0.0)
        advanced = jnp.conj(jnp.transpose(retarded, (1, 0, 3, 2)))
        finite = jnp.all(jnp.isfinite(lesser)) & jnp.all(jnp.isfinite(greater))
        return FermionicSecondBornSelfEnergy(
            lesser, greater, retarded, advanced, finite, self.prepared_id
        )

    def solve(self, /) -> FermionicSecondBornResult:
        functions = self.free_functions
        time_count, mode_count = functions.lesser.shape[0], functions.lesser.shape[-1]
        dimension = time_count * mode_count
        identity = jnp.eye(dimension, dtype=functions.lesser.dtype)
        extended_weights = jnp.repeat(self.quadrature_weights, mode_count)
        g0_retarded = self._matrix(self.free_functions.retarded)
        g0_lesser = self._matrix(self.free_functions.lesser)
        g0_greater = self._matrix(self.free_functions.greater)
        residual = jnp.asarray(jnp.inf)
        converged = False
        iterations = 0
        for index in range(self.plan.maximum_iterations):
            self_energy = self._self_energy(functions)
            sigma_retarded = (
                self._matrix(self_energy.retarded)
                * extended_weights[:, None]
                * extended_weights[None, :]
            )
            sigma_lesser = (
                self._matrix(self_energy.lesser)
                * extended_weights[:, None]
                * extended_weights[None, :]
            )
            sigma_greater = (
                self._matrix(self_energy.greater)
                * extended_weights[:, None]
                * extended_weights[None, :]
            )
            left_result = inverse(identity - g0_retarded @ sigma_retarded)
            left = left_result.value
            retarded_matrix = left @ g0_retarded
            advanced_matrix = jnp.conj(retarded_matrix.T)
            candidate_lesser = (
                left @ g0_lesser @ jnp.conj(left.T)
                + retarded_matrix @ sigma_lesser @ advanced_matrix
            )
            candidate_greater = (
                left @ g0_greater @ jnp.conj(left.T)
                + retarded_matrix @ sigma_greater @ advanced_matrix
            )
            lesser = self._tensor(candidate_lesser, time_count, mode_count)
            greater = self._tensor(candidate_greater, time_count, mode_count)
            lesser = 0.5 * (lesser - jnp.conj(jnp.transpose(lesser, (1, 0, 3, 2))))
            greater = 0.5 * (greater - jnp.conj(jnp.transpose(greater, (1, 0, 3, 2))))
            for time_index in range(time_count):
                density = 0.5 * (
                    -1j * lesser[time_index, time_index]
                    + jnp.eye(mode_count)
                    - 1j * greater[time_index, time_index]
                )
                density = 0.5 * (density + jnp.conj(density.T))
                lesser = lesser.at[time_index, time_index].set(1j * density)
                greater = greater.at[time_index, time_index].set(
                    -1j * (jnp.eye(mode_count) - density)
                )
            difference = greater - lesser
            causal = self.plan.grid.causal_mask[..., None, None]
            candidate = FermionicKeldyshFunctions(
                lesser,
                greater,
                jnp.where(causal, difference, 0.0),
                -jnp.where(jnp.swapaxes(causal, 0, 1), difference, 0.0),
                lesser + greater,
                jnp.all(jnp.isfinite(lesser)) & jnp.all(jnp.isfinite(greater)),
                self.plan.grid.grid_id,
                self.prepared_id,
            )
            change = jnp.maximum(
                jnp.max(jnp.abs(candidate.lesser - functions.lesser)),
                jnp.max(jnp.abs(candidate.greater - functions.greater)),
            )
            scale = jnp.maximum(
                1.0,
                jnp.maximum(
                    jnp.max(jnp.abs(candidate.lesser)),
                    jnp.max(jnp.abs(candidate.greater)),
                ),
            )
            residual = change / scale
            mixed_lesser = (
                1.0 - self.plan.damping
            ) * functions.lesser + self.plan.damping * candidate.lesser
            mixed_greater = (
                1.0 - self.plan.damping
            ) * functions.greater + self.plan.damping * candidate.greater
            mixed_difference = mixed_greater - mixed_lesser
            functions = FermionicKeldyshFunctions(
                mixed_lesser,
                mixed_greater,
                jnp.where(causal, mixed_difference, 0.0),
                -jnp.where(jnp.swapaxes(causal, 0, 1), mixed_difference, 0.0),
                mixed_lesser + mixed_greater,
                candidate.finite,
                self.plan.grid.grid_id,
                self.prepared_id,
            )
            iterations = index + 1
            if bool(np.asarray(residual <= self.plan.tolerance)):
                converged = True
                break
        self_energy = self._self_energy(functions)
        sigma_retarded = (
            self._matrix(self_energy.retarded)
            * extended_weights[:, None]
            * extended_weights[None, :]
        )
        retarded_matrix = self._matrix(functions.retarded)
        dyson = jnp.max(
            jnp.abs(
                retarded_matrix
                - g0_retarded
                - g0_retarded @ sigma_retarded @ retarded_matrix
            )
        )
        identity_evidence = fermionic_keldysh_identity_evidence(self.plan.grid, functions)
        density = jnp.stack(
            [-1j * functions.lesser[index, index] for index in range(time_count)]
        )
        particles = 2.0 * jnp.real(jnp.trace(density, axis1=-2, axis2=-1))
        particle_drift = jnp.max(jnp.abs(particles - particles[0]))
        one_body = 2.0 * jnp.real(
            contract(
                "ij,tji->t", self.plan.one_particle_hamiltonian, density, backend="jax"
            )
        )
        occupation = jnp.real(jnp.diagonal(density, axis1=-2, axis2=-1))
        energy = one_body + jnp.sum(self.plan.interaction * occupation**2, axis=-1)
        energy_drift = jnp.max(jnp.abs(energy - energy[0])) / jnp.maximum(
            1.0, jnp.abs(energy[0])
        )
        causality = jnp.maximum(
            identity_evidence.retarded_causality_residual,
            identity_evidence.advanced_causality_residual,
        )
        finite = (
            functions.finite
            & self_energy.finite
            & identity_evidence.finite
            & jnp.all(jnp.isfinite(energy))
            & jnp.isfinite(dyson)
        )
        conserved = (
            finite
            & (particle_drift <= self.plan.conservation_tolerance)
            & (energy_drift <= self.plan.conservation_tolerance)
        )
        successful = (
            finite
            & jnp.asarray(converged)
            & conserved
            & (
                dyson
                <= jnp.maximum(self.plan.tolerance, self.plan.conservation_tolerance)
            )
            & identity_evidence.satisfied
        )
        evidence = FermionicKadanoffBaymEvidence(
            residual,
            dyson,
            dyson,
            particles,
            particle_drift,
            energy,
            energy_drift,
            identity_evidence.equal_time_car_residual,
            causality,
            finite,
            jnp.asarray(converged),
            conserved,
            successful,
            jnp.asarray(iterations, dtype=jnp.int32),
            self.prepared_id,
        )
        return FermionicSecondBornResult(
            functions,
            self_energy,
            evidence,
            self.prepared_id,
            "candidate closed-system self-consistent second-Born control; no long-time convergence claim",
        )


__all__ = [
    "FermionicKadanoffBaymEvidence",
    "FermionicKeldyshFunctions",
    "FermionicKeldyshIdentityEvidence",
    "FermionicSecondBornPlan",
    "FermionicSecondBornResult",
    "FermionicSecondBornSelfEnergy",
    "PreparedFermionicSecondBorn",
    "fermionic_keldysh_from_propagators",
    "fermionic_keldysh_identity_evidence",
]
