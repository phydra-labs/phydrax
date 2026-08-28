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
    CellListParticleNeighborhoodPlan,
    ParticleDiscretization,
    ParticleExecutionPolicy,
    ParticlePrecisionPolicy,
    PreparedWeaklyCompressibleSPHDynamics,
    WeaklyCompressibleSPHMethodPlan,
)
from ._barotropic import AbstractBarotropicMaterial


if TYPE_CHECKING:
    from ..solver import DifferentialProblem


ExternalParticleAcceleration = Callable[[Array, Array, Array, Array, Any], ArrayLike]


class WeaklyCompressibleFluidProblemIR(StrictModule, NonTrainableState):
    """One barotropic fluid and optional nonconservative particle acceleration."""

    name: str = eqx.field(static=True)
    material: AbstractBarotropicMaterial
    external_acceleration: ExternalParticleAcceleration | None
    external_acceleration_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        material: AbstractBarotropicMaterial,
        /,
        *,
        external_acceleration: ExternalParticleAcceleration | None = None,
        external_acceleration_id: str | None = None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        if not name_:
            raise ValueError("Weakly compressible fluid problem name must be non-empty.")
        if not isinstance(material, AbstractBarotropicMaterial):
            raise TypeError("material must be an AbstractBarotropicMaterial.")
        if external_acceleration is not None and not callable(external_acceleration):
            raise TypeError("external_acceleration must be callable or None.")
        if external_acceleration is None and external_acceleration_id is not None:
            raise ValueError("external_acceleration_id requires external acceleration.")
        if external_acceleration is not None and not external_acceleration_id:
            raise ValueError("External acceleration requires a stable non-empty ID.")
        generated = canonical_fingerprint(
            {
                "kind": "weakly-compressible-fluid-problem-ir",
                "name": name_,
                "material": material.material_id,
                "external_acceleration": external_acceleration_id,
            }
        )
        identifier = generated if problem_id is None else str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.name = name_
        self.material = material
        self.external_acceleration = external_acceleration
        self.external_acceleration_id = external_acceleration_id
        self.problem_id = identifier


class CompiledWeaklyCompressibleSPHProblem(StrictModule, NonTrainableState):
    """Compiled first-order WCSPH problem and DifferentialProblem adapter."""

    problem: WeaklyCompressibleFluidProblemIR
    dynamics: PreparedWeaklyCompressibleSPHDynamics
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: WeaklyCompressibleFluidProblemIR,
        dynamics: PreparedWeaklyCompressibleSPHDynamics,
        discretization_bundle: DiscretizationBundle,
        /,
    ):
        if not isinstance(problem, WeaklyCompressibleFluidProblemIR):
            raise TypeError("problem must be a WeaklyCompressibleFluidProblemIR.")
        if not isinstance(dynamics, PreparedWeaklyCompressibleSPHDynamics):
            raise TypeError("dynamics must be PreparedWeaklyCompressibleSPHDynamics.")
        if not isinstance(discretization_bundle, DiscretizationBundle):
            raise TypeError("discretization_bundle must be a DiscretizationBundle.")
        self.problem = problem
        self.dynamics = dynamics
        self.discretization_bundle = discretization_bundle
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-weakly-compressible-sph-problem",
                "problem": problem.problem_id,
                "dynamics": dynamics.prepared_id,
                "bundle": discretization_bundle.bundle_id,
            }
        )

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        density: ArrayLike | None = None,
        /,
    ) -> Array:
        return self.dynamics.initialize_state(position, velocity, density)

    def as_differential_problem(
        self,
        initial_position: ArrayLike,
        initial_velocity: ArrayLike,
        initial_density: ArrayLike | None = None,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        args: Any = None,
        problem_id: str | None = None,
    ) -> DifferentialProblem:
        from ..metrix import EuclideanStateGeometry
        from ..solver import DifferentialProblem

        state = self.initialize_state(initial_position, initial_velocity, initial_density)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "wcsph-differential-problem",
                    "compilation": self.compilation_id,
                    "state_shape": list(state.shape),
                    "state_dtype": str(state.dtype),
                    "density_evolved": self.dynamics.state_layout.density_evolved,
                    "density_initialization": (
                        "kernel_summation"
                        if self.dynamics.state_layout.density_evolved
                        and initial_density is None
                        else "explicit_state"
                    ),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        return DifferentialProblem(
            self.dynamics,
            state,
            t0=t0,
            t1=t1,
            args=args,
            state_geometry=EuclideanStateGeometry(
                geometry_id=self.dynamics.state_layout.state_geometry_id
            ),
            discretization_bundle=self.discretization_bundle,
            problem_id=identifier,
        )


def compile_weakly_compressible_sph_problem(
    problem: WeaklyCompressibleFluidProblemIR,
    particles: ParticleDiscretization,
    method: WeaklyCompressibleSPHMethodPlan,
    /,
    *,
    neighborhood: AbstractParticleNeighborhoodPlan,
    execution: ParticleExecutionPolicy | None = None,
    precision: ParticlePrecisionPolicy | None = None,
) -> CompiledWeaklyCompressibleSPHProblem:
    """Compile one fixed-h first-order weakly compressible SPH problem."""

    if not isinstance(problem, WeaklyCompressibleFluidProblemIR):
        raise TypeError("problem must be a WeaklyCompressibleFluidProblemIR.")
    if not isinstance(particles, ParticleDiscretization):
        raise TypeError("particles must be a ParticleDiscretization.")
    if not isinstance(method, WeaklyCompressibleSPHMethodPlan):
        raise TypeError("method must be a WeaklyCompressibleSPHMethodPlan.")
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
    dynamics = PreparedWeaklyCompressibleSPHDynamics(
        particles,
        prepared_neighborhood,
        method,
        problem.material,
        execution=execution_,
        precision=precision_,
        external_acceleration=problem.external_acceleration,
        external_acceleration_id=problem.external_acceleration_id,
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
        "weakly-compressible-sph-dynamics",
        dynamics.prepared_id,
        dependency_key_ids=(particles.key.key_id, prepared_neighborhood.key.key_id),
        precision_evidence_id=dynamics.precision_evidence.evidence_id,
        resource_evidence_id=dynamics.resource_evidence_id,
    )
    bundle = DiscretizationBundle((particle_record, neighborhood_record, method_record))
    return CompiledWeaklyCompressibleSPHProblem(problem, dynamics, bundle)


__all__ = [
    "CompiledWeaklyCompressibleSPHProblem",
    "WeaklyCompressibleFluidProblemIR",
    "compile_weakly_compressible_sph_problem",
]
