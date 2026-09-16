#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology._full_dark_sector_inference import (
    FixedProfileEvaluation,
    FixedProfileSmoothSensitivityPlan,
    FullDarkSectorDifferentiationPolicy,
)
from phydrax.applications.cosmology._full_dark_sector_observables import (
    MetricStressObservables,
)
from phydrax.applications.cosmology._full_dark_sector_runtime import (
    assemble_full_dark_sector_stress,
    FullDarkSectorStageToken,
    NamedStressEnergyComponent,
)
from phydrax.applications.relativistic_scattering._matrix_element_revision import (
    MatrixElementRevision,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.equations._dark_radiation_moments import DarkRadiationFourForce
from phydrax.metrix import ADMGridGeometry, RelativityConvention, StressEnergyProjection
from phydrax.observation import LinearObservationPlan
from phydrax.solver._dark_sector_epoch_runtime import (
    DarkSectorEpochPlan,
    empty_dark_sector_epoch_state,
)


def _workflow_stage() -> tuple[FullDarkSectorStageToken, ADMGridGeometry]:
    scale = RelativityScaleContract(DimensionalScaleContract.si(), 1, 3, 2, 1, True)
    units = RelativisticUnitContract(
        scale, RelativityConvention(metric_signature="mostly_minus")
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(4, dtype=jnp.int32),
        chart_id="workflow-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="workflow-topology",
        geometry_lineage_id="workflow-geometry",
    )
    frame = LocalRelativisticFramePlan.from_adm(
        geometry,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.25),
        jnp.asarray(0.8),
        observer_id="workflow-observer",
        orientation_id="right-handed-future",
    )
    epoch_plan = DarkSectorEpochPlan(
        packet_capacity=2,
        event_capacity=2,
        product_capacity=2,
        radiation_capacity=2,
        work_capacity=2,
        frontier_capacity=2,
        packet_width=4,
        event_width=4,
        product_width=4,
        radiation_width=4,
        work_width=4,
        frontier_width=4,
        species_revision_id="1" * 64,
        topology_revision_id="2" * 64,
    )
    epoch = empty_dark_sector_epoch_state(
        epoch_plan, epoch_sequence=0, parent_epoch_manifest_id=None
    )
    revision = MatrixElementRevision(
        "workflow-process",
        "workflow-model",
        "native",
        "workflow-rights",
        "workflow-normalization",
        "workflow-support",
        "workflow-proposal",
        "no-adaptation",
        "workflow-training",
        "fixed-optimizer",
        "workflow-error",
        "3" * 64,
        differentiation_id="fixed-profile",
    )
    return FullDarkSectorStageToken(frame, epoch, revision), geometry


def _component(
    stage: FullDarkSectorStageToken,
    geometry: ADMGridGeometry,
    name: str,
    energy: float,
    /,
) -> NamedStressEnergyComponent:
    projection = StressEnergyProjection(
        jnp.asarray(energy),
        jnp.asarray((0.1 * energy, 0.0, 0.0)),
        (0.2 * energy) * jnp.eye(3),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        snapshot_token=stage.geometry_snapshot_token,
        geometry_lineage_id=geometry.geometry_lineage_id,
        convention_id=geometry.convention_id,
        scale_id=geometry.scale_id,
        topology_id=geometry.topology_id,
        projection_id=f"workflow-{name}-stress",
    )
    return NamedStressEnergyComponent(
        name,
        projection,
        conservation_defect=0.0,
        constraint_defect=0.0,
        gauge_defect=0.0,
        entropy_production=0.0,
        unitarity_defect=0.0,
        evidence_valid=True,
        evidence_id=f"workflow-{name}-evidence",
    )


def test_full_dark_sector_stage_to_observation_and_fixed_profile_sensitivity() -> None:
    stage, geometry = _workflow_stage()
    matter = _component(stage, geometry, "matter", 2.0)
    radiation = _component(stage, geometry, "radiation", 0.5)
    stress = assemble_full_dark_sector_stress((matter, radiation), stage)
    exchange = DarkRadiationFourForce.paired(
        jnp.asarray((0.2, 0.1, 0.0, -0.1)),
        stage.time,
        stage.frame_token,
        source_state_id="workflow-radiation-state",
        frame_id=stage.frame_id,
        frame_realization_id=stage.frame_realization_id,
        unit_contract_id=stage.unit_contract_id,
    )
    observable = MetricStressObservables(
        geometry,
        stress.total,
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
        frame_id=stage.frame_id,
        frame_realization_id=stage.frame_realization_id,
        source_ids=(matter.evidence_id, radiation.evidence_id),
    )
    theory = observable.as_theory_vector()
    response = LinearObservationPlan(
        jnp.eye(theory.layout.size), theory.layout, theory.layout
    ).apply(theory)

    policy = FullDarkSectorDifferentiationPolicy(
        (("workflow", stage.stage_id),),
        (("matrix-element", stage.matrix_element_revision_id),),
        topology_id=stage.topology_id,
        provider_ids=(),
        external_artifact_ids=(),
        differentiable_parameters=("matter-normalization",),
    )
    sensitivity = FixedProfileSmoothSensitivityPlan(
        policy,
        jnp.asarray((1,), dtype=jnp.int32),
        output_count=1,
        product_id=response.product_id,
    )
    epsilon = 1.0e-3

    def evaluation(value: float) -> FixedProfileEvaluation:
        return FixedProfileEvaluation(
            jnp.asarray((value,)),
            jnp.asarray((1,), dtype=jnp.int32),
            finite=True,
            successful=True,
            smooth=True,
            topology_fixed=True,
            provider_fixed=True,
            external_artifacts_constant=True,
            fixed_profile_id=policy.fixed_profile_id,
            product_id=response.product_id,
        )

    derivative = sensitivity.audit(
        evaluation(2.5),
        evaluation(2.5 - epsilon),
        evaluation(2.5 + epsilon),
        jnp.asarray((1.0,)),
        epsilon=epsilon,
    )

    assert bool(stress.successful)
    assert bool(jnp.all(exchange.exact_opposite))
    assert jnp.array_equal(response.values, theory.values)
    assert jnp.allclose(stress.total.energy_density, 2.5)
    assert bool(derivative.successful)
