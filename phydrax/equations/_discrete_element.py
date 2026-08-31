#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import DiscretizationBundle, DiscretizationRecord
from ..discretization.particle import (
    AbstractParticleNeighborhoodPlan,
    CellListParticleNeighborhoodPlan,
    ImplicitDEMBarrier,
    ParticleDiscretization,
    ParticleExecutionPolicy,
    ParticlePairKeySpace,
    ParticlePrecisionPolicy,
    PreparedSoftSphereDEMDynamics,
    RigidSphereSetPlan,
    SoftSphereDEMMethodPlan,
    VerletParticleNeighborhoodPlan,
)
from ..discretization.particle._dem import DEMExternalLoad, DEMRuntimeState
from ._dem_material import DEMMaterialTable


ExternalDEMLoad = Callable[[Array, Array, Array, Array, Any], DEMExternalLoad]


class DiscreteElementProblemIR(StrictModule, NonTrainableState):
    """Material, gravity, barriers, and optional external load for spherical DEM."""

    name: str = eqx.field(static=True)
    materials: DEMMaterialTable
    gravity: Array
    barriers: tuple[ImplicitDEMBarrier, ...]
    external_load: ExternalDEMLoad | None
    external_load_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        materials: DEMMaterialTable,
        /,
        *,
        gravity: ArrayLike,
        barriers: Sequence[ImplicitDEMBarrier] = (),
        external_load: ExternalDEMLoad | None = None,
        external_load_id: str | None = None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        if not name_:
            raise ValueError("Discrete-element problem name must be nonempty.")
        if not isinstance(materials, DEMMaterialTable):
            raise TypeError("materials must be a DEMMaterialTable.")
        gravity_host = np.asarray(gravity)
        if gravity_host.ndim != 1 or gravity_host.size not in (2, 3):
            raise ValueError("gravity must be a vector of dimension 2 or 3.")
        if np.any(~np.isfinite(gravity_host)):
            raise ValueError("gravity must be finite.")
        barriers_ = tuple(barriers)
        if any(not isinstance(value, ImplicitDEMBarrier) for value in barriers_):
            raise TypeError("barriers must contain ImplicitDEMBarrier values.")
        if external_load is not None and not callable(external_load):
            raise TypeError("external_load must be callable or None.")
        if external_load is None and external_load_id is not None:
            raise ValueError("external_load_id requires external_load.")
        if external_load is not None and not external_load_id:
            raise ValueError("External DEM load requires a stable nonempty ID.")
        generated = canonical_fingerprint(
            {
                "kind": "discrete-element-problem-ir",
                "name": name_,
                "materials": materials.material_id,
                "gravity": gravity_host.tolist(),
                "barriers": [value.barrier_id for value in barriers_],
                "external_load": external_load_id,
            }
        )
        identifier = generated if problem_id is None else str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be nonempty.")
        self.name = name_
        self.materials = materials
        self.gravity = jnp.asarray(gravity_host)
        self.barriers = barriers_
        self.external_load = external_load
        self.external_load_id = (
            None if external_load_id is None else str(external_load_id)
        )
        self.problem_id = identifier


class CompiledDiscreteElementProblem(StrictModule, NonTrainableState):
    """Compiled spherical DEM problem with fixed-step state initialization."""

    problem: DiscreteElementProblemIR
    dynamics: PreparedSoftSphereDEMDynamics
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: DiscreteElementProblemIR,
        dynamics: PreparedSoftSphereDEMDynamics,
        discretization_bundle: DiscretizationBundle,
        /,
    ):
        if not isinstance(problem, DiscreteElementProblemIR):
            raise TypeError("problem must be a DiscreteElementProblemIR.")
        if not isinstance(dynamics, PreparedSoftSphereDEMDynamics):
            raise TypeError("dynamics must be PreparedSoftSphereDEMDynamics.")
        if not isinstance(discretization_bundle, DiscretizationBundle):
            raise TypeError("discretization_bundle must be a DiscretizationBundle.")
        self.problem = problem
        self.dynamics = dynamics
        self.discretization_bundle = discretization_bundle
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-discrete-element-problem",
                "problem": problem.problem_id,
                "dynamics": dynamics.prepared_id,
                "bundle": discretization_bundle.bundle_id,
            }
        )

    def initialize_state(
        self,
        time: ArrayLike,
        position: ArrayLike,
        velocity: ArrayLike,
        angular_velocity: ArrayLike | None = None,
        /,
        *,
        args: Any = None,
    ) -> DEMRuntimeState:
        return self.dynamics.initialize_state(
            time,
            position,
            velocity,
            angular_velocity,
            args=args,
        )

    def diagnostics(
        self, time: ArrayLike, state: DEMRuntimeState, /, *, args: Any = None
    ):
        zero = jnp.zeros((), dtype=state.kinematics.position.dtype)
        return self.dynamics.evaluate(jnp.asarray(time), state, zero, args).diagnostics

    def step_restriction(self):
        return self.dynamics.step_restriction()


def compile_discrete_element_problem(
    problem: DiscreteElementProblemIR,
    particles: ParticleDiscretization,
    spheres: RigidSphereSetPlan,
    method: SoftSphereDEMMethodPlan,
    /,
    *,
    neighborhood: AbstractParticleNeighborhoodPlan,
    execution: ParticleExecutionPolicy | None = None,
    precision: ParticlePrecisionPolicy | None = None,
) -> CompiledDiscreteElementProblem:
    """Compile one fixed-capacity spherical soft-contact DEM problem."""

    if not isinstance(problem, DiscreteElementProblemIR):
        raise TypeError("problem must be a DiscreteElementProblemIR.")
    if not isinstance(particles, ParticleDiscretization):
        raise TypeError("particles must be a ParticleDiscretization.")
    if not isinstance(spheres, RigidSphereSetPlan):
        raise TypeError("spheres must be a RigidSphereSetPlan.")
    if not isinstance(method, SoftSphereDEMMethodPlan):
        raise TypeError("method must be a SoftSphereDEMMethodPlan.")
    if not isinstance(neighborhood, AbstractParticleNeighborhoodPlan):
        raise TypeError("neighborhood must be an AbstractParticleNeighborhoodPlan.")
    if problem.gravity.shape != (particles.ambient_dimension,):
        raise ValueError("Problem gravity does not match particle ambient dimension.")
    bodies = spheres.prepare(particles)
    active_materials = np.asarray(bodies.material_ids)[
        np.asarray(particles.active_mask, dtype=bool)
    ]
    if np.any(active_materials >= problem.materials.material_count):
        raise ValueError("Rigid-sphere material ID is out of range.")
    required = 2.0 * float(
        np.max(np.asarray(bodies.radii)[np.asarray(particles.active_mask, dtype=bool)])
    )
    required = required + method.contact.interaction_range
    if isinstance(neighborhood, CellListParticleNeighborhoodPlan):
        if neighborhood.search_radius < required:
            raise ValueError(
                "Cell-list search radius must cover the largest sphere diameter."
            )
    if isinstance(neighborhood, VerletParticleNeighborhoodPlan):
        if neighborhood.interaction_radius < required:
            raise ValueError(
                "Verlet interaction radius must cover the largest sphere diameter."
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
    key_space = ParticlePairKeySpace(particles)
    contact_model = method.contact.prepare(problem.materials, particles.ambient_dimension)
    dynamics = PreparedSoftSphereDEMDynamics(
        bodies,
        prepared_neighborhood,
        key_space,
        contact_model,
        method,
        problem.materials,
        barriers=problem.barriers,
        gravity=problem.gravity,
        external_load=problem.external_load,
        external_load_id=problem.external_load_id,
        execution=execution_,
        precision=precision_,
    )
    particle_record = DiscretizationRecord(
        particles.key,
        "particle-support",
        particles.prepared_id,
        numeric_version=particles.numeric_version,
        resource_evidence_id=particles.resource_evidence_id,
    )
    body_record = DiscretizationRecord(
        bodies.key,
        "rigid-sphere-properties",
        bodies.prepared_id,
        dependency_key_ids=(particles.key.key_id,),
        resource_evidence_id=bodies.resource_evidence_id,
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
        "soft-sphere-dem-dynamics",
        dynamics.prepared_id,
        dependency_key_ids=(
            particles.key.key_id,
            bodies.key.key_id,
            prepared_neighborhood.key.key_id,
        ),
        precision_evidence_id=dynamics.precision_evidence.evidence_id,
        resource_evidence_id=dynamics.resource_evidence_id,
    )
    bundle = DiscretizationBundle(
        (particle_record, body_record, neighborhood_record, method_record)
    )
    return CompiledDiscreteElementProblem(problem, dynamics, bundle)


__all__ = [
    "CompiledDiscreteElementProblem",
    "DiscreteElementProblemIR",
    "compile_discrete_element_problem",
]
