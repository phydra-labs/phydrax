#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import equinox as eqx
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import DiscretizationBundle, DiscretizationRecord
from ..discretization.particle import (
    AbstractParticleNeighborhoodPlan,
    BarotropicSPHMethodPlan,
    CellListParticleNeighborhoodPlan,
    ParticleDiscretization,
    ParticleExecutionPolicy,
    ParticlePrecisionPolicy,
    PreparedBarotropicSPHDynamics,
)
from ._barotropic import AbstractBarotropicMaterial


if TYPE_CHECKING:
    from ..solver import DifferentialProblem, SeparableHamiltonianVectorField


ExternalParticlePotential = Callable[[Array, Array, Any], ArrayLike]


class BarotropicFluidProblemIR(StrictModule, NonTrainableState):
    """One barotropic material and optional conservative external potential."""

    name: str = eqx.field(static=True)
    material: AbstractBarotropicMaterial
    external_potential: ExternalParticlePotential | None
    external_potential_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        material: AbstractBarotropicMaterial,
        /,
        *,
        external_potential: ExternalParticlePotential | None = None,
        external_potential_id: str | None = None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        if not name_:
            raise ValueError("Barotropic fluid problem name must be non-empty.")
        if not isinstance(material, AbstractBarotropicMaterial):
            raise TypeError("material must be an AbstractBarotropicMaterial.")
        if external_potential is not None and not callable(external_potential):
            raise TypeError("external_potential must be callable or None.")
        if external_potential is None and external_potential_id is not None:
            raise ValueError("external_potential_id requires an external potential.")
        if external_potential is not None and not external_potential_id:
            raise ValueError("An external potential requires a stable non-empty ID.")
        generated = canonical_fingerprint(
            {
                "kind": "barotropic-fluid-problem-ir",
                "name": name_,
                "material": material.material_id,
                "external_potential": external_potential_id,
            }
        )
        self.name = name_
        self.material = material
        self.external_potential = external_potential
        self.external_potential_id = external_potential_id
        self.problem_id = generated if problem_id is None else str(problem_id)
        if not self.problem_id:
            raise ValueError("problem_id must be non-empty.")


class _SPHPotentialGradient(StrictModule, NonTrainableState):
    dynamics: PreparedBarotropicSPHDynamics

    def __call__(self, time: Array, position: Array, args: Any, /) -> Array:
        return self.dynamics.potential_gradient(time, position, args)


class _SPHKineticGradient(StrictModule, NonTrainableState):
    dynamics: PreparedBarotropicSPHDynamics

    def __call__(self, time: Array, momentum: Array, args: Any, /) -> Array:
        return self.dynamics.kinetic_gradient(time, momentum, args)


class CompiledBarotropicSPHProblem(StrictModule, NonTrainableState):
    """Compiled conservative SPH problem with Hamiltonian temporal adapter."""

    problem: BarotropicFluidProblemIR
    dynamics: PreparedBarotropicSPHDynamics
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: BarotropicFluidProblemIR,
        dynamics: PreparedBarotropicSPHDynamics,
        discretization_bundle: DiscretizationBundle,
        /,
    ):
        if not isinstance(problem, BarotropicFluidProblemIR):
            raise TypeError("problem must be a BarotropicFluidProblemIR.")
        if not isinstance(dynamics, PreparedBarotropicSPHDynamics):
            raise TypeError("dynamics must be PreparedBarotropicSPHDynamics.")
        if not isinstance(discretization_bundle, DiscretizationBundle):
            raise TypeError("discretization_bundle must be a DiscretizationBundle.")
        self.problem = problem
        self.dynamics = dynamics
        self.discretization_bundle = discretization_bundle
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-barotropic-sph-problem",
                "problem": problem.problem_id,
                "dynamics": dynamics.prepared_id,
                "bundle": discretization_bundle.bundle_id,
            }
        )

    def hamiltonian_vector_field(self) -> SeparableHamiltonianVectorField:
        from ..solver import SeparableHamiltonianVectorField

        return SeparableHamiltonianVectorField(
            _SPHPotentialGradient(self.dynamics),
            _SPHKineticGradient(self.dynamics),
            self.dynamics.particles.ambient_dimension,
        )

    def as_differential_problem(
        self,
        initial_position: ArrayLike,
        initial_velocity: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        args: Any = None,
        problem_id: str | None = None,
    ) -> DifferentialProblem:
        from ..metrix import EuclideanStateGeometry
        from ..solver import DifferentialProblem

        state = self.dynamics.pack_phase_state(initial_position, initial_velocity)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "barotropic-sph-differential-problem",
                    "compilation": self.compilation_id,
                    "state_shape": list(state.shape),
                    "state_dtype": str(state.dtype),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        return DifferentialProblem(
            self.hamiltonian_vector_field(),
            state,
            t0=t0,
            t1=t1,
            args=args,
            state_geometry=EuclideanStateGeometry(
                geometry_id="state-geometry:canonical-phase"
            ),
            discretization_bundle=self.discretization_bundle,
            problem_id=identifier,
        )


def compile_barotropic_sph_problem(
    problem: BarotropicFluidProblemIR,
    particles: ParticleDiscretization,
    method: BarotropicSPHMethodPlan,
    /,
    *,
    neighborhood: AbstractParticleNeighborhoodPlan,
    execution: ParticleExecutionPolicy | None = None,
    precision: ParticlePrecisionPolicy | None = None,
) -> CompiledBarotropicSPHProblem:
    """Compile one fixed-h barotropic particle problem into SPH dynamics."""

    if not isinstance(problem, BarotropicFluidProblemIR):
        raise TypeError("problem must be a BarotropicFluidProblemIR.")
    if not isinstance(particles, ParticleDiscretization):
        raise TypeError("particles must be a ParticleDiscretization.")
    if not isinstance(method, BarotropicSPHMethodPlan):
        raise TypeError("method must be a BarotropicSPHMethodPlan.")
    if not isinstance(neighborhood, AbstractParticleNeighborhoodPlan):
        raise TypeError("neighborhood must be an AbstractParticleNeighborhoodPlan.")
    if isinstance(neighborhood, CellListParticleNeighborhoodPlan):
        support_radius = method.kernel.support_factor * method.smoothing_length
        if neighborhood.search_radius < support_radius:
            raise ValueError(
                "Cell-list search radius must cover the SPH kernel support radius."
            )
    execution_ = (
        ParticleExecutionPolicy(realization=neighborhood.backend)
        if execution is None
        else execution
    )
    precision_ = (
        ParticlePrecisionPolicy(
            geometry_dtype=particles.plan.coordinate_dtype,
            evaluation_dtype=particles.plan.coordinate_dtype,
        )
        if precision is None
        else precision
    )
    prepared_neighborhood = neighborhood.prepare(particles)
    dynamics = PreparedBarotropicSPHDynamics(
        particles,
        prepared_neighborhood,
        method,
        problem.material,
        execution=execution_,
        precision=precision_,
        external_potential=problem.external_potential,
        external_potential_id=problem.external_potential_id,
    )
    particle_record = DiscretizationRecord(
        particles.key,
        "particle-support",
        particles.prepared_id,
        numeric_version=particles.numeric_version,
        resource_evidence_id=particles.resource_evidence_id,
    )
    neighborhood_record = DiscretizationRecord(
        prepared_neighborhood.key,
        prepared_neighborhood.artifact_kind,
        prepared_neighborhood.prepared_id,
        numeric_version=prepared_neighborhood.numeric_version,
        dependency_key_ids=(particles.key.key_id,),
        resource_evidence_id=prepared_neighborhood.resource_evidence_id,
    )
    method_record = DiscretizationRecord(
        dynamics.key,
        "barotropic-sph-dynamics",
        dynamics.prepared_id,
        dependency_key_ids=(particles.key.key_id, prepared_neighborhood.key.key_id),
        precision_evidence_id=dynamics.precision_evidence.evidence_id,
        resource_evidence_id=dynamics.resource_evidence_id,
    )
    bundle = DiscretizationBundle((particle_record, neighborhood_record, method_record))
    return CompiledBarotropicSPHProblem(problem, dynamics, bundle)


__all__ = [
    "BarotropicFluidProblemIR",
    "CompiledBarotropicSPHProblem",
    "compile_barotropic_sph_problem",
]
