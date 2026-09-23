#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Internal-coordinate optimization, dimer saddles, rates, and kinetic networks."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomicStructure, AtomisticSystemPlan
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    OperatorProperties,
    solve,
)
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from .._optimization import _require_structure_matches_system
from .._surface import AbstractPreparedPotentialEnergySurface
from ..coordinates import MolecularCoordinateSystemPlan


class InternalOptimizationKind(StrEnum):
    MINIMUM = "minimum"
    FIRST_ORDER_SADDLE = "first-order-saddle"


class InternalOptimizationResult(StrictModule, NonTrainableState):
    positions: Array
    internal_coordinates: Array
    energies: Array
    maximum_forces: Array
    internal_gradient_norms: Array
    iterations: Array
    successful: Array
    kind: InternalOptimizationKind = eqx.field(static=True)
    source_result_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions,
        internal_coordinates,
        energies,
        maximum_forces,
        internal_gradient_norms,
        iterations,
        successful,
        kind,
        source_result_ids,
        plan_id,
        /,
    ):
        positions_ = jnp.asarray(positions)
        internals = jnp.asarray(internal_coordinates, dtype=positions_.dtype)
        energies_ = jnp.asarray(energies, dtype=positions_.dtype)
        forces = jnp.asarray(maximum_forces, dtype=positions_.dtype)
        gradients = jnp.asarray(internal_gradient_norms, dtype=positions_.dtype)
        steps = energies_.size
        if (
            positions_.ndim != 3
            or internals.ndim != 2
            or positions_.shape[0] != steps
            or internals.shape[0] != steps
            or forces.shape != (steps,)
            or gradients.shape != (steps,)
        ):
            raise ValueError("Internal optimization trajectories must align by step.")
        self.positions = positions_
        self.internal_coordinates = internals
        self.energies = energies_
        self.maximum_forces = forces
        self.internal_gradient_norms = gradients
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.kind = kind
        self.source_result_ids = tuple(source_result_ids)
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "internal-optimization-result",
                "optimization_kind": kind.value,
                "plan": self.plan_id,
                "sources": list(self.source_result_ids),
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "positions": np.asarray(positions_),
                        "internals": np.asarray(internals),
                        "energies": np.asarray(energies_),
                        "maximum_forces": np.asarray(forces),
                        "internal_gradient_norms": np.asarray(gradients),
                    }
                ),
            }
        )


class InternalCoordinateOptimizationPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    surface: AbstractPreparedPotentialEnergySurface
    coordinates: MolecularCoordinateSystemPlan
    kind: InternalOptimizationKind = eqx.field(static=True)
    force_tolerance: float = eqx.field(static=True)
    internal_gradient_tolerance: float = eqx.field(static=True)
    trust_radius: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        surface: AbstractPreparedPotentialEnergySurface,
        coordinates: MolecularCoordinateSystemPlan,
        /,
        *,
        kind: InternalOptimizationKind = InternalOptimizationKind.MINIMUM,
        force_tolerance: float = 1.0e-4,
        internal_gradient_tolerance: float = 1.0e-5,
        trust_radius: float = 0.1,
        maximum_iterations: int = 200,
    ):
        if (
            not isinstance(system, AtomisticSystemPlan)
            or not isinstance(surface, AbstractPreparedPotentialEnergySurface)
            or not isinstance(coordinates, MolecularCoordinateSystemPlan)
        ):
            raise TypeError(
                "Internal optimization requires typed system, surface, and coordinates."
            )
        if surface.system_id != system.system_id or not surface.capabilities.forces:
            raise ValueError(
                "Internal optimization requires a force surface for the same system."
            )
        if surface.units.unit_system_id != system.units.unit_system_id:
            raise ValueError(
                "Internal optimization surface and system unit identities differ."
            )
        if coordinates.atom_count != system.particle_ids.size:
            raise ValueError("Internal-coordinate atom capacity differs from the system.")
        if not isinstance(kind, InternalOptimizationKind):
            raise TypeError("kind must be InternalOptimizationKind.")
        force = float(force_tolerance)
        gradient = float(internal_gradient_tolerance)
        trust = float(trust_radius)
        maximum = int(maximum_iterations)
        if (
            any(not isfinite(value) or value <= 0.0 for value in (force, gradient, trust))
            or maximum <= 0
        ):
            raise ValueError(
                "Internal optimization tolerances, trust radius, or limit are invalid."
            )
        self.system = system
        self.surface = surface
        self.coordinates = coordinates
        self.kind = kind
        self.force_tolerance = force
        self.internal_gradient_tolerance = gradient
        self.trust_radius = trust
        self.maximum_iterations = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "internal-coordinate-optimization-plan",
                "system": system.system_id,
                "surface": surface.surface_id,
                "coordinates": coordinates.plan_id,
                "optimization_kind": kind.value,
                "force_tolerance": force,
                "internal_gradient_tolerance": gradient,
                "trust_radius": trust,
                "maximum_iterations": maximum,
            }
        )

    def run(self, structure: AtomicStructure, /) -> InternalOptimizationResult:
        _require_structure_matches_system(structure, self.system)
        positions = jnp.asarray(structure.positions)
        cell = None if structure.cell is None else structure.cell
        mobile = np.asarray(self.system.mobile_mask, dtype=np.bool_)
        hessian = jnp.eye(self.coordinates.values(positions).size, dtype=positions.dtype)
        position_history = []
        internal_history = []
        energies = []
        maximum_forces = []
        gradient_norms = []
        sources = []
        previous_internal = previous_gradient = None
        successful = False
        for iteration in range(self.maximum_iterations):
            evaluation = self.surface.evaluate(positions, cell)
            coordinate_state = self.coordinates.evaluate(positions)
            cartesian_gradient = -jnp.asarray(evaluation.forces).reshape((-1,))
            internal_gradient_solve = solve(
                LeastSquaresProblem(
                    DenseLinearOperator(jnp.conj(coordinate_state.jacobian.T))
                ),
                cartesian_gradient,
                policy=LinearSolvePolicy(DenseSVD()),
            )
            internal_gradient = internal_gradient_solve.value
            maximum_force = jnp.max(
                jnp.abs(jnp.asarray(evaluation.forces)[mobile]), initial=0.0
            )
            gradient_norm = jnp.linalg.norm(internal_gradient)
            position_history.append(positions)
            internal_history.append(coordinate_state.values)
            energies.append(evaluation.energy)
            maximum_forces.append(maximum_force)
            gradient_norms.append(gradient_norm)
            sources.append(evaluation.source_result_id)
            if bool(
                evaluation.successful
                & coordinate_state.successful
                & internal_gradient_solve.successful
                & (maximum_force <= self.force_tolerance)
                & (gradient_norm <= self.internal_gradient_tolerance)
            ):
                successful = True
                break
            if previous_internal is not None:
                displacement = coordinate_state.values - previous_internal
                gradient_change = internal_gradient - previous_gradient
                curvature = jnp.dot(displacement, gradient_change)
                hessian_displacement = hessian @ displacement
                projected_curvature = jnp.dot(displacement, hessian_displacement)
                if bool(curvature > 1.0e-12) and bool(projected_curvature > 1.0e-12):
                    hessian = (
                        hessian
                        + jnp.outer(gradient_change, gradient_change) / curvature
                        - jnp.outer(hessian_displacement, hessian_displacement)
                        / projected_curvature
                    )
            spectrum = eigensolve(
                Eigenproblem(
                    DenseLinearOperator(
                        0.5 * (hessian + hessian.T),
                        properties=OperatorProperties(
                            self_adjoint=True,
                            evidence={"self_adjoint": "construction"},
                        ),
                    )
                ),
                policy=EigenSolvePolicy(
                    DenseEigh(),
                    count=hessian.shape[0],
                    which="smallest-algebraic",
                ),
            )
            if not bool(spectrum.successful):
                break
            eigenvalues, eigenvectors = (
                spectrum.eigenvalues,
                spectrum.eigenvectors,
            )
            if self.kind is InternalOptimizationKind.MINIMUM:
                effective_eigenvalues = jnp.maximum(jnp.abs(eigenvalues), 1.0e-4)
            else:
                effective_eigenvalues = jnp.maximum(jnp.abs(eigenvalues), 1.0e-4)
                effective_eigenvalues = effective_eigenvalues.at[0].multiply(-1.0)
            local_gradient = jnp.conj(eigenvectors.T) @ internal_gradient
            step = eigenvectors @ (-local_gradient / effective_eigenvalues)
            step_norm = jnp.linalg.norm(step)
            step = step * jnp.minimum(
                1.0,
                self.trust_radius
                / jnp.maximum(step_norm, jnp.finfo(step_norm.dtype).tiny),
            )
            previous_internal = coordinate_state.values
            previous_gradient = internal_gradient
            retracted = self.coordinates.retract(
                positions,
                step,
                trust_radius=self.trust_radius,
            )
            updated = retracted.positions
            positions = jnp.where(mobile[:, None], updated, positions)
            if not bool(retracted.successful):
                break
        return InternalOptimizationResult(
            jnp.stack(tuple(position_history)),
            jnp.stack(tuple(internal_history)),
            jnp.stack(tuple(energies)),
            jnp.stack(tuple(maximum_forces)),
            jnp.stack(tuple(gradient_norms)),
            len(energies),
            successful,
            self.kind,
            tuple(sources),
            self.plan_id,
        )


class SaddleRefinementResult(StrictModule, NonTrainableState):
    positions: Array
    mode: Array
    energy: Array
    maximum_force: Array
    curvature: Array
    iterations: Array
    successful: Array
    source_result_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class DimerSaddleRefinementPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    surface: AbstractPreparedPotentialEnergySurface
    translation_step: float = eqx.field(static=True)
    rotation_step: float = eqx.field(static=True)
    dimer_separation: float = eqx.field(static=True)
    force_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system,
        surface,
        /,
        *,
        translation_step=0.02,
        rotation_step=0.1,
        dimer_separation=1.0e-3,
        force_tolerance=1.0e-4,
        maximum_iterations=200,
    ):
        if not isinstance(system, AtomisticSystemPlan) or not isinstance(
            surface, AbstractPreparedPotentialEnergySurface
        ):
            raise TypeError("Dimer refinement requires typed system and surface.")
        if surface.system_id != system.system_id or not surface.capabilities.forces:
            raise ValueError(
                "Dimer refinement requires a force surface for the same system."
            )
        if surface.units.unit_system_id != system.units.unit_system_id:
            raise ValueError("Dimer surface and system unit identities differ.")
        values = tuple(
            float(value)
            for value in (
                translation_step,
                rotation_step,
                dimer_separation,
                force_tolerance,
            )
        )
        maximum = int(maximum_iterations)
        if any(not isfinite(value) or value <= 0.0 for value in values) or maximum <= 0:
            raise ValueError("Dimer steps, tolerance, or iteration limit are invalid.")
        self.system = system
        self.surface = surface
        (
            self.translation_step,
            self.rotation_step,
            self.dimer_separation,
            self.force_tolerance,
        ) = values
        self.maximum_iterations = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dimer-saddle-refinement-plan",
                "system": system.system_id,
                "surface": surface.surface_id,
                "translation_step": values[0],
                "rotation_step": values[1],
                "dimer_separation": values[2],
                "force_tolerance": values[3],
                "maximum_iterations": maximum,
            }
        )

    def run(
        self, structure: AtomicStructure, initial_mode: ArrayLike, /
    ) -> SaddleRefinementResult:
        _require_structure_matches_system(structure, self.system)
        positions = np.asarray(structure.positions).copy()
        mode = np.asarray(initial_mode, dtype=positions.dtype).copy()
        mobile = np.asarray(self.system.mobile_mask, dtype=np.bool_)
        mode[~mobile] = 0.0
        mode_norm = np.linalg.norm(mode)
        if (
            mode.shape != positions.shape
            or not np.isfinite(mode_norm)
            or mode_norm == 0.0
        ):
            raise ValueError(
                "Dimer mode must be finite, nonzero, and align with positions."
            )
        mode /= mode_norm
        cell = None if structure.cell is None else np.asarray(structure.cell)
        sources = []
        successful = False
        curvature = np.nan
        maximum_force = np.inf
        center = self.surface.evaluate(positions, cell)
        for iteration in range(self.maximum_iterations):
            plus = self.surface.evaluate(positions + self.dimer_separation * mode, cell)
            minus = self.surface.evaluate(positions - self.dimer_separation * mode, cell)
            center = self.surface.evaluate(positions, cell)
            sources.extend(
                (plus.source_result_id, minus.source_result_id, center.source_result_id)
            )
            hessian_mode = -(np.asarray(plus.forces) - np.asarray(minus.forces)) / (
                2.0 * self.dimer_separation
            )
            curvature = float(np.sum(mode * hessian_mode))
            rotation_force = hessian_mode - curvature * mode
            mode -= self.rotation_step * rotation_force
            mode[~mobile] = 0.0
            mode /= np.linalg.norm(mode)
            force = np.asarray(center.forces)
            parallel = float(np.sum(force * mode)) * mode
            effective_force = force - 2.0 * parallel
            effective_force[~mobile] = 0.0
            maximum_force = float(np.max(np.abs(force[mobile]), initial=0.0))
            if maximum_force <= self.force_tolerance and curvature < 0.0:
                successful = bool(
                    center.successful and plus.successful and minus.successful
                )
                break
            positions += self.translation_step * effective_force
        return SaddleRefinementResult(
            jnp.asarray(positions),
            jnp.asarray(mode),
            center.energy,
            jnp.asarray(maximum_force),
            jnp.asarray(curvature),
            jnp.asarray(iteration + 1, dtype=jnp.int32),
            jnp.asarray(successful),
            tuple(sources),
            self.plan_id,
        )


class TransitionStateRateResult(StrictModule, NonTrainableState):
    activation_free_energy: Array
    classical_rate: Array
    tunneling_factor: Array
    rate: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class TransitionStateRatePlan(StrictModule, NonTrainableState):
    temperature: float = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    planck_constant: float = eqx.field(static=True)
    hbar: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, temperature, boltzmann_constant, planck_constant, /, *, hbar):
        values = tuple(
            float(value)
            for value in (temperature, boltzmann_constant, planck_constant, hbar)
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(
                "Rate temperature and physical constants must be positive finite."
            )
        self.temperature, self.boltzmann_constant, self.planck_constant, self.hbar = (
            values
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "transition-state-rate-plan",
                "temperature": values[0],
                "boltzmann_constant": values[1],
                "planck_constant": values[2],
                "hbar": values[3],
            }
        )

    def evaluate(
        self,
        reactant_free_energy,
        transition_state_free_energy,
        /,
        *,
        imaginary_angular_frequency=None,
    ):
        reactant = jnp.asarray(reactant_free_energy)
        transition = jnp.asarray(transition_state_free_energy, dtype=reactant.dtype)
        activation = transition - reactant
        thermal = self.boltzmann_constant * self.temperature
        classical = thermal / self.planck_constant * jnp.exp(-activation / thermal)
        if imaginary_angular_frequency is None:
            tunneling = jnp.asarray(1.0, dtype=reactant.dtype)
        else:
            frequency = jnp.abs(
                jnp.asarray(imaginary_angular_frequency, dtype=reactant.dtype)
            )
            tunneling = 1.0 + (self.hbar * frequency / thermal) ** 2 / 24.0
        rate = classical * tunneling
        successful = jnp.isfinite(rate) & (rate >= 0.0) & (activation >= 0.0)
        return TransitionStateRateResult(
            activation,
            classical,
            tunneling,
            rate,
            successful,
            self.plan_id,
        )


class ReactionNetworkResult(StrictModule, NonTrainableState):
    times: Array
    populations: Array
    generator: Array
    conservation_residual: Array
    positivity_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ReactionNetworkPlan(StrictModule, NonTrainableState):
    generator: Array
    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, rate_matrix: ArrayLike, /, *, conservation_tolerance=1.0e-12):
        rates_host = np.asarray(rate_matrix)
        tolerance = float(conservation_tolerance)
        if (
            rates_host.ndim != 2
            or rates_host.shape[0] != rates_host.shape[1]
            or not np.issubdtype(rates_host.dtype, np.floating)
            or np.any(~np.isfinite(rates_host))
            or np.any(rates_host < 0.0)
            or np.any(np.diag(rates_host) != 0.0)
            or not isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError(
                "Reaction rates must be a finite real non-negative square zero-diagonal matrix with a finite non-negative tolerance."
            )
        rates = jnp.asarray(rates_host)
        generator = rates.T - jnp.diag(jnp.sum(rates, axis=1))
        residual = jnp.max(jnp.abs(jnp.sum(generator, axis=0)), initial=0.0)
        if bool(residual > tolerance):
            raise ValueError("Reaction generator does not conserve total population.")
        self.generator = generator
        self.conservation_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reaction-network-plan",
                "rate_matrix": array_tree_fingerprint(rates_host),
                "conservation_tolerance": tolerance.hex(),
            }
        )

    def propagate(
        self, initial_populations: ArrayLike, times: ArrayLike, /
    ) -> ReactionNetworkResult:
        initial = jnp.asarray(initial_populations, dtype=self.generator.dtype)
        times_ = jnp.asarray(times, dtype=self.generator.dtype)
        if (
            initial.shape != (self.generator.shape[0],)
            or times_.ndim != 1
            or bool(jnp.any(times_ < 0.0))
            or times_.size == 0
            or bool(jnp.any(~jnp.isfinite(times_)))
            or bool(jnp.any(~jnp.isfinite(initial)))
            or bool(jnp.any(initial < 0.0))
            or bool(jnp.sum(initial) <= 0.0)
        ):
            raise ValueError(
                "Reaction populations or times do not align with the network."
            )
        populations = jnp.stack(
            tuple(jsp.linalg.expm(time * self.generator) @ initial for time in times_)
        )
        conservation = jnp.max(
            jnp.abs(jnp.sum(populations, axis=1) - jnp.sum(initial)), initial=0.0
        )
        positivity = jnp.maximum(0.0, -jnp.min(populations))
        successful = (
            jnp.all(jnp.isfinite(populations))
            & (conservation <= self.conservation_tolerance)
            & (positivity <= self.conservation_tolerance)
        )
        return ReactionNetworkResult(
            times_,
            populations,
            self.generator,
            conservation,
            positivity,
            successful,
            self.plan_id,
        )


__all__ = [
    "DimerSaddleRefinementPlan",
    "InternalCoordinateOptimizationPlan",
    "InternalOptimizationKind",
    "InternalOptimizationResult",
    "ReactionNetworkPlan",
    "ReactionNetworkResult",
    "SaddleRefinementResult",
    "TransitionStateRatePlan",
    "TransitionStateRateResult",
]
