#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    DiscretizationBundle,
    DiscretizationKey,
    DiscretizationRecord,
    DiscretizationRole,
)
from ..discretization.mpm import (
    ExplicitMPMMethodPlan,
    MPMParticleDomainPlan,
    MPMResourcePolicy,
    MPMRuntimeState,
    PreparedMPMDynamics,
    PrescribedGridVelocityPlan,
)
from ..discretization.particle import ParticleDiscretization
from ..discretization.splatting import PreparedParticleGridSplat


MPMKinematics: TypeAlias = Literal["plane_strain", "three_dimensional"]


class MPMConstitutiveResponse(StrictModule):
    """One particle-local first-Piola response and candidate material history."""

    first_piola: Array
    trial_state: Array
    reference_energy_density: Array
    maximum_wave_speed: Array
    successful: Array
    admissible: Array
    diagnostics: Mapping[str, Array]

    def __init__(
        self,
        first_piola: ArrayLike,
        trial_state: ArrayLike,
        reference_energy_density: ArrayLike,
        maximum_wave_speed: ArrayLike,
        /,
        *,
        successful: ArrayLike,
        admissible: ArrayLike,
        diagnostics: Mapping[str, ArrayLike] | None = None,
    ):
        stress = jnp.asarray(first_piola)
        history = jnp.asarray(trial_state)
        energy = jnp.asarray(reference_energy_density)
        speed = jnp.asarray(maximum_wave_speed)
        successful_ = jnp.asarray(successful, dtype=bool)
        admissible_ = jnp.asarray(admissible, dtype=bool)
        if stress.ndim < 2 or stress.shape[-1] != stress.shape[-2]:
            raise ValueError("First-Piola stress must end in one square tensor.")
        batch_shape = stress.shape[:-2]
        for name, value in (
            ("reference_energy_density", energy),
            ("maximum_wave_speed", speed),
            ("successful", successful_),
            ("admissible", admissible_),
        ):
            if value.shape != batch_shape:
                raise ValueError(
                    f"{name} must have constitutive batch shape {batch_shape}."
                )
        self.first_piola = stress
        self.trial_state = history
        self.reference_energy_density = energy
        self.maximum_wave_speed = speed
        self.successful = successful_
        self.admissible = admissible_
        self.diagnostics = (
            {}
            if diagnostics is None
            else {str(name): jnp.asarray(value) for name, value in diagnostics.items()}
        )


class AbstractMPMConstitutivePlan(StrictModule, NonTrainableState):
    """Fixed-shape material update required by explicit material-point dynamics."""

    dimension: AbstractAttribute[int]
    kinematics: AbstractAttribute[MPMKinematics]
    state_shape: AbstractAttribute[tuple[int, ...]]
    plan_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        deformation_gradient: ArrayLike,
        committed_state: ArrayLike,
        reference_density: ArrayLike,
        parameters: Any,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> MPMConstitutiveResponse:
        raise NotImplementedError


ExternalMPMAcceleration = Callable[[Array, Array, Array, Any], ArrayLike]


class MaterialPointArguments(StrictModule):
    """Dynamic material and external-load arguments for one MPM rollout."""

    material_parameters: Any
    external_arguments: Any

    def __init__(self, material_parameters: Any, external_arguments: Any = None, /):
        self.material_parameters = material_parameters
        self.external_arguments = external_arguments


class MaterialPointProblemIR(StrictModule, NonTrainableState):
    """One homogeneous constitutive family and optional body acceleration."""

    name: str = eqx.field(static=True)
    material: AbstractMPMConstitutivePlan
    external_acceleration: ExternalMPMAcceleration | None
    external_acceleration_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        material: AbstractMPMConstitutivePlan,
        /,
        *,
        external_acceleration: ExternalMPMAcceleration | None = None,
        external_acceleration_id: str | None = None,
        problem_id: str | None = None,
    ):
        name_ = str(name)
        if not name_:
            raise ValueError("Material-point problem name must be non-empty.")
        if not isinstance(material, AbstractMPMConstitutivePlan):
            raise TypeError("material must be AbstractMPMConstitutivePlan.")
        if external_acceleration is not None and not callable(external_acceleration):
            raise TypeError("external_acceleration must be callable or None.")
        if external_acceleration is None and external_acceleration_id is not None:
            raise ValueError("external_acceleration_id requires external acceleration.")
        if external_acceleration is not None and not external_acceleration_id:
            raise ValueError("External acceleration requires a stable non-empty ID.")
        generated = canonical_fingerprint(
            {
                "kind": "material-point-problem-ir",
                "name": name_,
                "material": material.plan_id,
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


class CompiledMaterialPointProblem(StrictModule, NonTrainableState):
    problem: MaterialPointProblemIR
    dynamics: PreparedMPMDynamics
    discretization_bundle: DiscretizationBundle
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: MaterialPointProblemIR,
        dynamics: PreparedMPMDynamics,
        discretization_bundle: DiscretizationBundle,
        /,
    ):
        if not isinstance(problem, MaterialPointProblemIR):
            raise TypeError("problem must be MaterialPointProblemIR.")
        if not isinstance(dynamics, PreparedMPMDynamics):
            raise TypeError("dynamics must be PreparedMPMDynamics.")
        if not isinstance(discretization_bundle, DiscretizationBundle):
            raise TypeError("discretization_bundle must be DiscretizationBundle.")
        self.problem = problem
        self.dynamics = dynamics
        self.discretization_bundle = discretization_bundle
        self.compilation_id = canonical_fingerprint(
            {
                "kind": "compiled-material-point-problem",
                "problem": problem.problem_id,
                "dynamics": dynamics.prepared_id,
                "bundle": discretization_bundle.bundle_id,
            }
        )

    def initialize_state(
        self,
        position: ArrayLike,
        velocity: ArrayLike,
        reference_volume: ArrayLike,
        arguments: MaterialPointArguments,
        /,
        **kwargs: Any,
    ) -> MPMRuntimeState:
        if not isinstance(arguments, MaterialPointArguments):
            raise TypeError("arguments must be MaterialPointArguments.")
        return self.dynamics.initialize_state(
            position,
            velocity,
            reference_volume,
            arguments,
            **kwargs,
        )


def compile_material_point_problem(
    problem: MaterialPointProblemIR,
    particles: ParticleDiscretization,
    splat: PreparedParticleGridSplat,
    method: ExplicitMPMMethodPlan,
    particle_domain: MPMParticleDomainPlan,
    /,
    *,
    boundary: PrescribedGridVelocityPlan | None = None,
    resource_policy: MPMResourcePolicy | None = None,
) -> CompiledMaterialPointProblem:
    if not isinstance(problem, MaterialPointProblemIR):
        raise TypeError("problem must be MaterialPointProblemIR.")
    dynamics = PreparedMPMDynamics(
        particles,
        splat,
        method,
        problem.material,
        particle_domain,
        boundary=boundary,
        external_acceleration=problem.external_acceleration,
        external_acceleration_id=problem.external_acceleration_id,
        resource_policy=resource_policy,
    )
    transfer_key = DiscretizationKey(
        "mpm-particle-grid",
        DiscretizationRole.AUXILIARY,
        domain_labels=("material_point", "background_grid"),
    )
    particle_record = DiscretizationRecord(
        particles.key,
        "particle-support",
        particles.prepared_id,
        numeric_version=particles.numeric_version,
        resource_evidence_id=particles.resource_evidence_id,
    )
    transfer_record = DiscretizationRecord(
        transfer_key,
        splat.artifact_kind,
        splat.prepared_id,
        dependency_key_ids=(particles.key.key_id,),
        precision_evidence_id=splat.precision_evidence.evidence_id,
        resource_evidence_id=splat.resource_evidence_id,
    )
    method_record = DiscretizationRecord(
        dynamics.key,
        "explicit-mpm-dynamics",
        dynamics.prepared_id,
        dependency_key_ids=(particles.key.key_id, transfer_key.key_id),
        precision_evidence_id=dynamics.precision_evidence.evidence_id,
        resource_evidence_id=dynamics.resource_evidence_id,
    )
    bundle = DiscretizationBundle((particle_record, transfer_record, method_record))
    return CompiledMaterialPointProblem(problem, dynamics, bundle)


__all__ = [
    "AbstractMPMConstitutivePlan",
    "CompiledMaterialPointProblem",
    "ExternalMPMAcceleration",
    "MPMConstitutiveResponse",
    "MPMKinematics",
    "MaterialPointArguments",
    "MaterialPointProblemIR",
    "compile_material_point_problem",
]
