#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import DiscretizationBundle, DiscretizationRecord
from ..discretization.particle import ParticleDiscretization, ParticlePrecisionPolicy
from ..discretization.vortex._compatibility import vortex_property_requirements
from ..discretization.vortex._interfaces import VortexFieldRequest
from ..discretization.vortex._method import (
    BackgroundVortexVelocity,
    PreparedVortexParticleDynamics,
    VortexParticleMethodPlan,
)
from ..discretization.vortex._particle import VortexParticleProperties


if TYPE_CHECKING:
    from ..solver import DifferentialProblem


class VortexParticleFlowProblem(StrictModule, NonTrainableState):
    """Incompressible free-vorticity problem with named background velocity."""

    name: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    kinematic_viscosity: Array
    background_velocity: BackgroundVortexVelocity | None
    background_velocity_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        dimension: int,
        kinematic_viscosity: ArrayLike = 0.0,
        /,
        *,
        background_velocity: BackgroundVortexVelocity | None = None,
        background_velocity_id: str | None = None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        dimension_ = int(dimension)
        viscosity = jnp.asarray(kinematic_viscosity, dtype=float)
        if not name_:
            raise ValueError("Vortex problem name must be nonempty.")
        if dimension_ not in (2, 3):
            raise ValueError("Vortex particle flow requires dimension 2 or 3.")
        if viscosity.shape != () or not bool(
            jnp.isfinite(viscosity) & (viscosity >= 0.0)
        ):
            raise ValueError("kinematic_viscosity must be one finite nonnegative scalar.")
        if background_velocity is not None and not callable(background_velocity):
            raise TypeError("background_velocity must be callable or None.")
        if background_velocity is None and background_velocity_id is not None:
            raise ValueError("background_velocity_id requires a callback.")
        if background_velocity is not None and not background_velocity_id:
            raise ValueError("A background velocity requires a stable nonempty ID.")
        generated = canonical_fingerprint(
            {
                "kind": "vortex-particle-flow-problem",
                "name": name_,
                "dimension": dimension_,
                "kinematic_viscosity": float(viscosity),
                "background_velocity": background_velocity_id,
            }
        )
        identifier = generated if problem_id is None else str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.name = name_
        self.dimension = dimension_
        self.kinematic_viscosity = viscosity
        self.background_velocity = background_velocity
        self.background_velocity_id = background_velocity_id
        self.problem_id = identifier


class CompiledVortexParticleFlow(StrictModule, NonTrainableState):
    """Prepared vortex-particle dynamics and canonical differential adapter."""

    problem: VortexParticleFlowProblem
    dynamics: PreparedVortexParticleDynamics
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: VortexParticleFlowProblem,
        dynamics: PreparedVortexParticleDynamics,
        bundle: DiscretizationBundle,
        /,
    ):
        self.problem = problem
        self.dynamics = dynamics
        self.discretization_bundle = bundle
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-vortex-particle-flow",
                "problem": problem.problem_id,
                "dynamics": dynamics.prepared_id,
                "bundle": bundle.bundle_id,
            }
        )

    def initialize_state(self, position: ArrayLike, strength: ArrayLike, /) -> Array:
        return self.dynamics.initialize_state(position, strength)

    def as_differential_problem(
        self,
        initial_position: ArrayLike,
        initial_strength: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        args: Any = None,
        problem_id: str | None = None,
    ) -> DifferentialProblem:
        from ..metrix import EuclideanStateGeometry
        from ..solver import DifferentialProblem

        state = self.initialize_state(initial_position, initial_strength)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "vortex-particle-differential-problem",
                    "compilation": self.compilation_id,
                    "state_shape": list(state.shape),
                    "state_dtype": str(state.dtype),
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
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


def compile_vortex_particle_flow(
    problem: VortexParticleFlowProblem,
    particles: ParticleDiscretization,
    properties: VortexParticleProperties,
    method: VortexParticleMethodPlan,
    /,
    *,
    precision: ParticlePrecisionPolicy | None = None,
) -> CompiledVortexParticleFlow:
    """Compile one fixed-population two- or three-dimensional vortex flow."""

    if not isinstance(problem, VortexParticleFlowProblem):
        raise TypeError("problem must be VortexParticleFlowProblem.")
    if not isinstance(particles, ParticleDiscretization):
        raise TypeError("particles must be ParticleDiscretization.")
    if not isinstance(properties, VortexParticleProperties):
        raise TypeError("properties must be VortexParticleProperties.")
    if not isinstance(method, VortexParticleMethodPlan):
        raise TypeError("method must be VortexParticleMethodPlan.")
    if particles.ambient_dimension != problem.dimension:
        raise ValueError("Particle and vortex-problem dimensions differ.")
    if (
        method.velocity.dimension != problem.dimension
        or method.diffusion.dimension != problem.dimension
    ):
        raise ValueError("Vortex method and problem dimensions differ.")
    requirements = vortex_property_requirements(
        method.velocity.capabilities,
        method.diffusion.capabilities,
    )
    properties.validate(
        particles.capacity,
        require_core_radius=requirements.core_radius,
        require_volume=requirements.volume,
    )
    request = VortexFieldRequest(
        velocity=True,
        velocity_gradient=problem.dimension == 3,
    )
    velocity = method.velocity.prepare(
        source_capacity=particles.capacity,
        target_capacity=particles.capacity,
        source_kind="particle",
        target_topology="same-support",
        request=request,
    )
    diffusion = method.diffusion.prepare(
        capacity=particles.capacity,
        dimension=problem.dimension,
    )
    dynamics = PreparedVortexParticleDynamics(
        particles,
        properties,
        velocity,
        diffusion,
        method,
        problem.kinematic_viscosity,
        precision=precision,
        background_velocity=problem.background_velocity,
        background_velocity_id=problem.background_velocity_id,
    )
    particle_record = DiscretizationRecord(
        particles.key,
        "particle-support",
        particles.prepared_id,
        numeric_version=particles.numeric_version,
        resource_evidence_id=particles.resource_evidence_id,
    )
    method_record = DiscretizationRecord(
        dynamics.key,
        "vortex-particle-dynamics",
        dynamics.prepared_id,
        dependency_key_ids=(particles.key.key_id,),
        precision_evidence_id=dynamics.precision.evidence().evidence_id,
        resource_evidence_id=dynamics.preparation.report_id,
    )
    bundle = DiscretizationBundle((particle_record, method_record))
    return CompiledVortexParticleFlow(problem, dynamics, bundle)


__all__ = [
    "CompiledVortexParticleFlow",
    "VortexParticleFlowProblem",
    "compile_vortex_particle_flow",
]
