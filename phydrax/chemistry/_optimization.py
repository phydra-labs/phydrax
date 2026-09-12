#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Molecular Cartesian optimization over native or external energy surfaces."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..atomistic import AtomicStructure, AtomisticSystemPlan
from ..optim import (
    AbstractMinimizationMethod,
    MinimizationProblem,
    MinimizationResult,
    minimize,
    OptimizationTermination,
    SciPyMinimize,
)
from ._surface import (
    AbstractPreparedPotentialEnergySurface,
    PotentialEnergySurfaceEvaluation,
)


class MolecularGeometryConvergencePlan(StrictModule, NonTrainableState):
    """Force and work limits for one Cartesian molecular optimization."""

    maximum_force: float = eqx.field(static=True)
    rms_force: float = eqx.field(static=True)
    step_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    maximum_evaluations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_force: float = 1.0e-4,
        rms_force: float = 5.0e-5,
        step_tolerance: float = 1.0e-8,
        maximum_steps: int = 256,
        maximum_evaluations: int = 2048,
    ):
        maximum = float(maximum_force)
        rms = float(rms_force)
        step = float(step_tolerance)
        steps = int(maximum_steps)
        evaluations = int(maximum_evaluations)
        if any(not isfinite(value) or value <= 0.0 for value in (maximum, rms, step)):
            raise ValueError("Geometry convergence tolerances must be finite and positive.")
        if steps <= 0 or evaluations <= 0:
            raise ValueError("Geometry optimization work limits must be positive.")
        self.maximum_force = maximum
        self.rms_force = rms
        self.step_tolerance = step
        self.maximum_steps = steps
        self.maximum_evaluations = evaluations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "molecular-geometry-convergence",
                "maximum_force": maximum,
                "rms_force": rms,
                "step_tolerance": step,
                "maximum_steps": steps,
                "maximum_evaluations": evaluations,
            }
        )

    def optimization_termination(self) -> OptimizationTermination:
        return OptimizationTermination(
            absolute_optimality=self.maximum_force,
            relative_optimality=0.0,
            absolute_step=self.step_tolerance,
            relative_step=0.0,
            maximum_steps=self.maximum_steps,
            maximum_evaluations=self.maximum_evaluations,
        )


class MolecularGeometryOptimizationResult(StrictModule, NonTrainableState):
    initial_structure: AtomicStructure
    final_structure: AtomicStructure
    final_evaluation: PotentialEnergySurfaceEvaluation
    optimization: MinimizationResult
    maximum_force: Array
    rms_force: Array
    provider_evaluations: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_structure: AtomicStructure,
        final_structure: AtomicStructure,
        final_evaluation: PotentialEnergySurfaceEvaluation,
        optimization: MinimizationResult,
        maximum_force: ArrayLike,
        rms_force: ArrayLike,
        provider_evaluations: int,
        successful: ArrayLike,
        plan_id: str,
        /,
    ):
        if not isinstance(initial_structure, AtomicStructure) or not isinstance(
            final_structure, AtomicStructure
        ):
            raise TypeError("Geometry optimization structures must be AtomicStructure.")
        if not isinstance(final_evaluation, PotentialEnergySurfaceEvaluation):
            raise TypeError("final_evaluation must be PotentialEnergySurfaceEvaluation.")
        if not isinstance(optimization, MinimizationResult):
            raise TypeError("optimization must be MinimizationResult.")
        maximum = jnp.asarray(maximum_force).reshape(())
        rms = jnp.asarray(rms_force, dtype=maximum.dtype).reshape(())
        evaluations = jnp.asarray(provider_evaluations, dtype=jnp.int32).reshape(())
        successful_ = jnp.asarray(successful, dtype=bool).reshape(())
        self.initial_structure = initial_structure
        self.final_structure = final_structure
        self.final_evaluation = final_evaluation
        self.optimization = optimization
        self.maximum_force = maximum
        self.rms_force = rms
        self.provider_evaluations = evaluations
        self.successful = successful_
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "molecular-geometry-optimization-result",
                "plan": self.plan_id,
                "initial": initial_structure.structure_id,
                "final": final_structure.structure_id,
                "surface_result": final_evaluation.evaluation_id,
                "optimization_method": optimization.provenance.method,
                "successful": bool(successful_),
                "arrays": array_tree_fingerprint(
                    {
                        "maximum_force": np.asarray(maximum),
                        "rms_force": np.asarray(rms),
                        "provider_evaluations": np.asarray(evaluations),
                    }
                ),
            }
        )


class MolecularGeometryOptimizationPlan(StrictModule, NonTrainableState):
    system: AtomisticSystemPlan
    surface: AbstractPreparedPotentialEnergySurface
    method: AbstractMinimizationMethod
    convergence: MolecularGeometryConvergencePlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: AtomisticSystemPlan,
        surface: AbstractPreparedPotentialEnergySurface,
        /,
        *,
        method: AbstractMinimizationMethod | None = None,
        convergence: MolecularGeometryConvergencePlan | None = None,
    ):
        if not isinstance(system, AtomisticSystemPlan):
            raise TypeError("system must be AtomisticSystemPlan.")
        if not isinstance(surface, AbstractPreparedPotentialEnergySurface):
            raise TypeError("surface must be a prepared potential-energy surface.")
        if surface.system_id != system.system_id:
            raise ValueError("Geometry surface belongs to another system.")
        if surface.units.unit_system_id != system.units.unit_system_id:
            raise ValueError("Geometry surface and system units differ.")
        if not surface.capabilities.forces:
            raise ValueError("Geometry optimization requires surface forces.")
        convergence_ = (
            MolecularGeometryConvergencePlan() if convergence is None else convergence
        )
        if not isinstance(convergence_, MolecularGeometryConvergencePlan):
            raise TypeError(
                "convergence must be MolecularGeometryConvergencePlan or None."
            )
        method_ = (
            SciPyMinimize(
                "L-BFGS-B",
                options={"gtol": convergence_.maximum_force, "ftol": 0.0},
            )
            if method is None
            else method
        )
        if not isinstance(method_, AbstractMinimizationMethod):
            raise TypeError("method must implement AbstractMinimizationMethod.")
        if not method_.capabilities.explicit_host_gradient:
            raise ValueError(
                "Molecular external-surface optimization requires an explicit-host-gradient method."
            )
        self.system = system
        self.surface = surface
        self.method = method_
        self.convergence = convergence_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "molecular-geometry-optimization",
                "system": system.system_id,
                "surface": surface.surface_id,
                "method": method_.method_id,
                "convergence": convergence_.plan_id,
            }
        )

    def run(
        self,
        structure: AtomicStructure,
        /,
    ) -> MolecularGeometryOptimizationResult:
        if not isinstance(structure, AtomicStructure):
            raise TypeError("structure must be AtomicStructure.")
        _require_structure_matches_system(structure, self.system)
        initial = np.asarray(structure.positions, dtype=np.dtype(self.system.coordinate_dtype))
        mobile = np.asarray(self.system.mobile_mask, dtype=bool)
        if not np.any(mobile):
            raise ValueError("Geometry optimization requires at least one mobile atom.")
        cell = None if structure.cell is None else np.asarray(structure.cell)
        cache_position: np.ndarray | None = None
        cache_evaluation: PotentialEnergySurfaceEvaluation | None = None
        provider_evaluations = 0

        def evaluate_active(value) -> PotentialEnergySurfaceEvaluation:
            nonlocal cache_position, cache_evaluation, provider_evaluations
            active_position = np.asarray(value, dtype=initial.dtype).reshape((-1, 3))
            if cache_position is not None and np.array_equal(active_position, cache_position):
                if cache_evaluation is None:
                    raise RuntimeError("Geometry evaluation cache lost its result.")
                return cache_evaluation
            full = initial.copy()
            full[mobile] = active_position
            evaluated = self.surface.evaluate(full, cell)
            cache_position = active_position.copy()
            cache_evaluation = evaluated
            provider_evaluations += 1
            return evaluated

        def objective(value, _):
            return evaluate_active(value).energy

        def value_and_gradient(value, _):
            evaluated = evaluate_active(value)
            gradient = -evaluated.forces[mobile]
            return (evaluated.energy, evaluated), gradient

        problem = MinimizationProblem(
            objective,
            has_aux=True,
            explicit_value_and_gradient=value_and_gradient,
            derivative_execution="explicit-host",
            problem_id=self.plan_id,
        )
        optimized = minimize(
            problem,
            jnp.asarray(initial[mobile]),
            method=self.method,
            termination=self.convergence.optimization_termination(),
        )
        final_position = initial.copy()
        final_position[mobile] = np.asarray(optimized.parameters).reshape((-1, 3))
        final_evaluation = evaluate_active(final_position[mobile])
        mobile_force = np.asarray(final_evaluation.forces)[mobile]
        maximum_force = float(np.max(np.abs(mobile_force)))
        rms_force = float(np.sqrt(np.mean(mobile_force**2)))
        successful = (
            bool(optimized.successful)
            and bool(final_evaluation.successful)
            and maximum_force <= self.convergence.maximum_force
            and rms_force <= self.convergence.rms_force
            and provider_evaluations <= self.convergence.maximum_evaluations
        )
        final_structure = AtomicStructure(
            structure.atomic_numbers,
            final_position,
            structure.masses,
            structure.scale,
            particle_ids=structure.particle_ids,
            active_mask=structure.active_mask,
            cell=structure.cell,
            periodic_axes=structure.periodic_axes,
            name=structure.name,
            coordinate_dtype=structure.positions.dtype,
        )
        return MolecularGeometryOptimizationResult(
            structure,
            final_structure,
            final_evaluation,
            optimized,
            maximum_force,
            rms_force,
            provider_evaluations,
            successful,
            self.plan_id,
        )


def _require_structure_matches_system(
    structure: AtomicStructure, system: AtomisticSystemPlan, /
) -> None:
    if structure.scale.scale_id != system.units.scale.scale_id:
        raise ValueError("Structure and optimization system scales differ.")
    active = np.asarray(system.active_mask, dtype=bool)
    comparisons = (
        np.array_equal(
            np.asarray(structure.particle_ids), np.asarray(system.particle_ids)
        ),
        np.array_equal(
            np.asarray(structure.atomic_numbers), np.asarray(system.atomic_numbers)
        ),
        np.array_equal(np.asarray(structure.active_mask), active),
        np.array_equal(
            np.asarray(structure.masses)[active], np.asarray(system.masses)[active]
        ),
    )
    if system.cell is None:
        cell_matches = structure.cell is None
    else:
        cell_matches = (
            structure.cell is not None
            and np.array_equal(np.asarray(structure.cell), np.asarray(system.cell.vectors))
            and structure.periodic_axes is not None
            and np.array_equal(
                np.asarray(structure.periodic_axes, dtype=bool),
                np.asarray(system.cell.periodic_axes, dtype=bool),
            )
        )
    comparisons = (*comparisons, cell_matches)
    if not all(comparisons):
        raise ValueError("Structure identity axes do not match the optimization system.")


__all__ = [
    "MolecularGeometryConvergencePlan",
    "MolecularGeometryOptimizationPlan",
    "MolecularGeometryOptimizationResult",
]
