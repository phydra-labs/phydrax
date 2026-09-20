# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....artifacts import ScientificArtifactEnvelope
from ....atomistic import (
    AtomisticDynamicsPlan,
    AtomisticDynamicsState,
    AtomisticRolloutPlan,
    AtomisticRolloutResult,
    AtomisticThermodynamicStatePlan,
    AtomisticTrajectoryPlan,
    BAOABLangevinPlan,
    PreparedAtomisticDynamics,
    PreparedThermodynamicStateTable,
    VelocityVerletPlan,
)
from ....dynamics import StateLayout, TrajectoryData
from ....units import conversion_factor
from .._binding import PreparedProteinBinding
from .._qualification import PreparedProteinQualification, ProteinGeometryEvidence


@dataclass(frozen=True, slots=True)
class ProteinDynamicsResult:
    binding: PreparedProteinBinding
    dynamics: PreparedAtomisticDynamics
    thermodynamic_states: PreparedThermodynamicStateTable
    initial_state: AtomisticDynamicsState
    rollout: AtomisticRolloutResult
    initial_geometry: ProteinGeometryEvidence
    final_geometry: ProteinGeometryEvidence
    artifact: ScientificArtifactEnvelope

    def trajectory_data(self) -> TrajectoryData:
        """Retain native time, masks and the atomistic CV feature state ABI."""
        trajectory = self.rollout.trajectory
        states = jnp.stack((trajectory.positions, trajectory.momenta), axis=1)
        layout = StateLayout(
            (2, self.binding.force_field.system.capacity, 3),
            axes=("kinematic", "atom", "cartesian"),
            layout_id=canonical_fingerprint(
                {
                    "kind": "atomistic-trajectory-state-layout",
                    "system": self.binding.force_field.system.prepared_id,
                }
            ),
        )
        return TrajectoryData(
            trajectory.times,
            states,
            state_layout=layout,
            sample_valid=trajectory.valid & trajectory.sample_mask,
            reset_mask=jnp.zeros((trajectory.times.size - 1,), dtype=jnp.bool_),
            coordinate_id=trajectory.units.time_unit.unit_id,
            source_id=self.artifact.artifact_id,
        )


def prepare_protein_dynamics(
    binding,
    neighborhood,
    integrator,
    thermodynamic_state,
):
    """Compose one declared native thermodynamic state without a new engine."""
    if not isinstance(binding, PreparedProteinBinding):
        raise TypeError("binding must be a PreparedProteinBinding.")
    if not isinstance(integrator, (VelocityVerletPlan, BAOABLangevinPlan)):
        raise TypeError("Declare VelocityVerletPlan or BAOABLangevinPlan.")
    if not isinstance(thermodynamic_state, AtomisticThermodynamicStatePlan):
        raise TypeError("thermodynamic_state must be an AtomisticThermodynamicStatePlan.")
    field = binding.force_field
    dynamics = AtomisticDynamicsPlan(
        field.system,
        field.potential,
        neighborhood,
        integrator,
        constraints=field.constraints,
    ).prepare()
    return dynamics, thermodynamic_state.prepare(dynamics)


def run_protein_dynamics(
    binding: PreparedProteinBinding,
    neighborhood,
    integrator,
    qualification: PreparedProteinQualification,
    thermodynamic_state: AtomisticThermodynamicStatePlan,
    *,
    velocity,
    velocity_unit,
    key,
    step_count: int,
    sample_stride: int = 1,
    commercial_use=False,
    export=False,
) -> ProteinDynamicsResult:
    """Run one source-bound atomistic thermodynamic state.

    Biased conservative terms must already be part of the force-field bundle and
    the thermodynamic state must identify them. Adaptive or undeclared bias state
    is not admitted as an unbiased physical trajectory.
    """
    binding.require_rights(commercial_use=commercial_use, export=export)
    if qualification.binding_id != binding.binding_id:
        raise ValueError("Geometry qualification belongs to another protein binding.")
    factor = float(
        conversion_factor(
            velocity_unit, binding.force_field.system.plan.units.velocity_unit
        )
    )
    speed = np.asarray(velocity, dtype=np.float64) * factor
    if speed.shape != np.asarray(binding.realized_positions).shape or not np.all(
        np.isfinite(speed)
    ):
        raise ValueError(
            "Velocity must be finite and aligned to the prepared system support."
        )
    initial_geometry = qualification.evaluate(binding.realized_positions)
    if not bool(initial_geometry.successful):
        raise ValueError(
            "Initial protein geometry failed declared qualification; no silent relaxation/completion is performed."
        )
    dynamics, thermodynamic_states = prepare_protein_dynamics(
        binding,
        neighborhood,
        integrator,
        thermodynamic_state,
    )
    initial = dynamics.initialize_state(
        binding.realized_positions,
        thermodynamic_states,
        velocity=jnp.asarray(speed),
        key=key,
    )
    rollout = AtomisticRolloutPlan(
        dynamics,
        thermodynamic_states,
        AtomisticTrajectoryPlan(step_count, sample_stride=sample_stride),
    ).rollout(initial)
    final_geometry = qualification.evaluate(rollout.final_state.kinematics.positions)
    successful = bool(rollout.successful) and bool(final_geometry.successful)
    digest = canonical_fingerprint(
        {
            "kind": "protein-physical-trajectory",
            "binding": binding.binding_id,
            "thermodynamic_states": thermodynamic_states.table_id,
            "rollout": rollout.rollout_id,
            "data": array_tree_fingerprint(
                (
                    rollout.trajectory.times,
                    rollout.trajectory.positions,
                    rollout.trajectory.momenta,
                    rollout.trajectory.valid,
                )
            ),
            "qualification": qualification.qualification_id,
            "bias": thermodynamic_state.bias_id,
        }
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind="protein-physical-trajectory",
        content_digest=digest,
        producer="phydrax.protein_folding.run_protein_dynamics",
        producer_version="native",
        build_id=dynamics.prepared_id,
        license_id="inherited-see-parent-manifests",
        resource_id=rollout.rollout_id,
        status="complete" if successful else "failed",
        failure_reason="none" if successful else "native-step-or-geometry-failed",
        parent_artifact_ids=(
            binding.artifact.artifact_id,
            qualification.qualification_id,
        ),
    )
    return ProteinDynamicsResult(
        binding,
        dynamics,
        thermodynamic_states,
        initial,
        rollout,
        initial_geometry,
        final_geometry,
        artifact,
    )


__all__ = ["ProteinDynamicsResult", "prepare_protein_dynamics", "run_protein_dynamics"]
