#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import DiscretizationBundle, DiscretizationRecord
from ..discretization.particle import (
    AbstractParticleNeighborhoodPlan,
    AbstractRigidContactShapePlan,
    CellListParticleNeighborhoodPlan,
    CompliantDEMMethodPlan,
    DEMDiagnostics,
    DEMStepRestriction,
    HierarchicalRadiusParticleNeighborhoodPlan,
    ImplicitDEMBarrier,
    LevelSetPenaltyNormalPlan,
    ParticleDiscretization,
    ParticleExecutionPolicy,
    ParticlePairKeySpace,
    ParticlePrecisionPolicy,
    PreparedCompliantDEMDynamics,
    TriangleWallPlan,
    VerletParticleNeighborhoodPlan,
)
from ..discretization.particle._dem import DEMExternalLoad, DEMRuntimeState
from ..typing import checked
from ._dem_material import DEMMaterialTable


ExternalDEMLoad = Callable[[Array, Array, Array, Array, Any], DEMExternalLoad]


class DiscreteElementProblemIR(StrictModule, NonTrainableState):
    """Materials, gravity, boundaries, and loads for compliant rigid-body DEM."""

    name: str = eqx.field(static=True)
    materials: DEMMaterialTable
    gravity: Array
    barriers: tuple[ImplicitDEMBarrier, ...]
    walls: tuple[TriangleWallPlan, ...]
    external_load: ExternalDEMLoad | None
    external_load_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        name: str,
        materials: DEMMaterialTable,
        /,
        *,
        gravity: ArrayLike,
        barriers: Sequence[ImplicitDEMBarrier] = (),
        walls: Sequence[TriangleWallPlan] = (),
        external_load: ExternalDEMLoad | None = None,
        external_load_id: str | None = None,
        problem_id: str | None = None,
    ) -> None:
        name_ = str(name)
        if not name_:
            raise ValueError("Discrete-element problem name must be nonempty.")
        gravity_host = np.asarray(gravity)
        if gravity_host.ndim != 1 or gravity_host.size not in (2, 3):
            raise ValueError("gravity must be a vector of dimension 2 or 3.")
        if np.any(~np.isfinite(gravity_host)):
            raise ValueError("gravity must be finite.")
        barriers_ = tuple(barriers)
        if any(not isinstance(value, ImplicitDEMBarrier) for value in barriers_):
            raise TypeError("barriers must contain ImplicitDEMBarrier values.")
        walls_ = tuple(walls)
        if any(not isinstance(value, TriangleWallPlan) for value in walls_):
            raise TypeError("walls must contain TriangleWallPlan values.")
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
                "walls": [value.wall_id for value in walls_],
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
        self.walls = walls_
        self.external_load = external_load
        self.external_load_id = (
            None if external_load_id is None else str(external_load_id)
        )
        self.problem_id = identifier


class CompiledDiscreteElementProblem(StrictModule, NonTrainableState):
    """Compiled shape-generic DEM problem with fixed-step initialization."""

    problem: DiscreteElementProblemIR
    dynamics: PreparedCompliantDEMDynamics
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        problem: DiscreteElementProblemIR,
        dynamics: PreparedCompliantDEMDynamics,
        discretization_bundle: DiscretizationBundle,
        /,
    ) -> None:
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
        orientation: ArrayLike | None = None,
        args: Any = None,
    ) -> DEMRuntimeState:
        return self.dynamics.initialize_state(
            time,
            position,
            velocity,
            angular_velocity,
            orientation=orientation,
            args=args,
        )

    def diagnostics(
        self, time: ArrayLike, state: DEMRuntimeState, /, *, args: Any = None
    ) -> DEMDiagnostics:
        zero = jnp.zeros((), dtype=state.kinematics.position.dtype)
        return self.dynamics.evaluate(jnp.asarray(time), state, zero, args).diagnostics

    def step_restriction(self) -> DEMStepRestriction:
        return self.dynamics.step_restriction()


def compile_discrete_element_problem(
    problem: DiscreteElementProblemIR,
    particles: ParticleDiscretization,
    shapes: AbstractRigidContactShapePlan,
    method: CompliantDEMMethodPlan,
    /,
    *,
    neighborhood: AbstractParticleNeighborhoodPlan,
    execution: ParticleExecutionPolicy | None = None,
    precision: ParticlePrecisionPolicy | None = None,
) -> CompiledDiscreteElementProblem:
    """Compile fixed-capacity compliant contact over prepared rigid shapes."""

    if not isinstance(problem, DiscreteElementProblemIR):
        raise TypeError("problem must be a DiscreteElementProblemIR.")
    if not isinstance(particles, ParticleDiscretization):
        raise TypeError("particles must be a ParticleDiscretization.")
    if not isinstance(shapes, AbstractRigidContactShapePlan):
        raise TypeError("shapes must be an AbstractRigidContactShapePlan.")
    if not isinstance(method, CompliantDEMMethodPlan):
        raise TypeError("method must be a CompliantDEMMethodPlan.")
    if not isinstance(neighborhood, AbstractParticleNeighborhoodPlan):
        raise TypeError("neighborhood must be an AbstractParticleNeighborhoodPlan.")
    if problem.gravity.shape != (particles.ambient_dimension,):
        raise ValueError("Problem gravity does not match particle ambient dimension.")
    if (shapes.shape_kind == "level_set") != isinstance(
        method.contact.normal, LevelSetPenaltyNormalPlan
    ):
        raise ValueError(
            "Level-set shapes require surface-area penalty laws; point shapes require point-force laws."
        )
    prepared_shapes = shapes.prepare(particles)
    bodies = prepared_shapes.bodies
    active_mask = np.asarray(particles.active_mask, dtype=np.bool_)
    body_radii = np.asarray(prepared_shapes.bounding_radii, dtype=np.float64)
    body_materials = np.asarray(prepared_shapes.material_ids)
    active_materials = body_materials[active_mask]
    contact_materials = np.asarray(prepared_shapes.contact_material_ids)
    if np.any(contact_materials >= problem.materials.material_count):
        raise ValueError("Shape material ID is out of range.")
    interaction_extents = method.contact.interaction_extents_for_radii(
        body_radii[active_mask],
        active_materials,
        problem.materials.material_count,
    )
    if prepared_shapes.shape_kind == "clump":
        maximum_bound = float(np.max(body_radii[active_mask]))
        material_extents = method.contact.interaction_extents_for_radii(
            np.full(contact_materials.shape, maximum_bound, dtype=np.float64),
            contact_materials,
            problem.materials.material_count,
        )
        interaction_extents = body_radii[active_mask] + float(
            np.max(material_extents) - maximum_bound
        )
    required = 2.0 * float(np.max(interaction_extents))
    if isinstance(neighborhood, HierarchicalRadiusParticleNeighborhoodPlan):
        hierarchy_radii = np.asarray(neighborhood.interaction_radii, dtype=np.float64)
        if hierarchy_radii.shape != body_radii.shape:
            raise ValueError(
                "Hierarchical interaction radii must match rigid-sphere capacity."
            )
        if np.any(
            hierarchy_radii[active_mask]
            < interaction_extents - 32.0 * np.finfo(interaction_extents.dtype).eps
        ):
            raise ValueError(
                "Hierarchical interaction radii do not cover contact-law envelopes."
            )
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
    dynamics = PreparedCompliantDEMDynamics(
        prepared_shapes,
        prepared_neighborhood,
        key_space,
        contact_model,
        method,
        problem.materials,
        barriers=problem.barriers,
        walls=tuple(wall.prepare() for wall in problem.walls),
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
        "rigid-body-properties",
        bodies.prepared_id,
        dependency_key_ids=(particles.key.key_id,),
        resource_evidence_id=bodies.preparation.report_id,
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
        "compliant-dem-dynamics",
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
