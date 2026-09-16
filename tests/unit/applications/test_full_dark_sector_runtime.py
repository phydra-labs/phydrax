#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import pytest

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
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
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
    StressEnergyProjection,
)
from phydrax.solver._dark_sector_epoch_runtime import (
    DarkSectorEpochPlan,
    empty_dark_sector_epoch_state,
)


def _frame(snapshot_token: int = 7) -> LocalRelativisticFramePlan:
    scale = RelativityScaleContract(
        DimensionalScaleContract.si(),
        1,
        3,
        2,
        1,
        True,
    )
    units = RelativisticUnitContract(
        scale,
        RelativityConvention(metric_signature="mostly_minus"),
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
        snapshot_token=jnp.asarray(snapshot_token, dtype=jnp.int32),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="single-cell",
        geometry_lineage_id="flat-adm",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="observer",
    )
    return LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.asarray((0.5, 0.0, 0.0, 0.0)),
        jnp.asarray(0.5),
        jnp.asarray(0.75),
        observer_id="observer",
        orientation_id="right-handed-future",
    )


def _epoch_plan() -> DarkSectorEpochPlan:
    return DarkSectorEpochPlan(
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


def _matrix_revision() -> MatrixElementRevision:
    return MatrixElementRevision(
        "process",
        "model",
        "provider",
        "rights",
        "normalization",
        "support",
        "proposal",
        "adaptation",
        "training",
        "optimizer",
        "error-model",
        "3" * 64,
        differentiation_id="fixed-profile",
    )


def _stage(snapshot_token: int = 7) -> FullDarkSectorStageToken:
    frame = _frame(snapshot_token)
    epoch = empty_dark_sector_epoch_state(
        _epoch_plan(), epoch_sequence=0, parent_epoch_manifest_id=None
    )
    return FullDarkSectorStageToken(frame, epoch, _matrix_revision())


def _projection(
    stage: FullDarkSectorStageToken,
    energy: float,
    momentum: tuple[float, float, float],
    pressure: float,
    /,
    *,
    projection_id: str,
    snapshot_token=None,
) -> StressEnergyProjection:
    return StressEnergyProjection(
        jnp.asarray(energy),
        jnp.asarray(momentum),
        pressure * jnp.eye(3),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        snapshot_token=(
            stage.geometry_snapshot_token
            if snapshot_token is None
            else jnp.asarray(snapshot_token, dtype=jnp.int32)
        ),
        geometry_lineage_id="flat-adm",
        convention_id=_frame().geometry.convention_id,
        scale_id=_frame().geometry.scale_id,
        topology_id=stage.topology_id,
        projection_id=projection_id,
    )


def _component(
    stage: FullDarkSectorStageToken,
    name: str,
    energy: float,
    /,
    *,
    projection_id: str,
    snapshot_token=None,
) -> NamedStressEnergyComponent:
    return NamedStressEnergyComponent(
        name,
        _projection(
            stage,
            energy,
            (0.1 * energy, 0.0, 0.0),
            0.2 * energy,
            projection_id=projection_id,
            snapshot_token=snapshot_token,
        ),
        conservation_defect=0.0,
        constraint_defect=0.0,
        gauge_defect=0.0,
        entropy_production=0.0,
        unitarity_defect=0.0,
        evidence_valid=True,
        evidence_id=f"evidence-{name}",
    )


def test_total_stress_has_exact_component_only_and_additive_limits() -> None:
    stage = _stage()
    matter = _component(stage, "matter", 2.0, projection_id="matter-stress")
    radiation = _component(stage, "radiation", 0.5, projection_id="radiation-stress")

    matter_only = assemble_full_dark_sector_stress((matter,), stage)
    coupled = assemble_full_dark_sector_stress((matter, radiation), stage)

    assert jnp.array_equal(
        matter_only.total.energy_density, matter.projection.energy_density
    )
    assert jnp.array_equal(
        matter_only.total.momentum_covector, matter.projection.momentum_covector
    )
    assert jnp.allclose(coupled.total.energy_density, 2.5)
    assert jnp.allclose(
        coupled.total.stress_covariant,
        matter.projection.stress_covariant + radiation.projection.stress_covariant,
    )
    assert bool(coupled.successful)


def test_shared_stage_mismatch_is_refused() -> None:
    stage = _stage(snapshot_token=7)
    stale = _component(
        stage,
        "matter",
        1.0,
        projection_id="stale",
        snapshot_token=6,
    )

    with pytest.raises(Exception, match="stage snapshot token"):
        assemble_full_dark_sector_stress((stale,), stage)


def test_dark_radiation_four_force_is_exactly_opposite() -> None:
    stage = _stage()
    radiation = jnp.asarray((0.3, 0.1, -0.2, 0.4))

    exchange = DarkRadiationFourForce.paired(
        radiation,
        stage.time,
        stage.frame_token,
        source_state_id="radiation-state",
        frame_id=stage.frame_id,
        frame_realization_id=stage.frame_realization_id,
        unit_contract_id=stage.unit_contract_id,
    )

    assert bool(jnp.all(exchange.exact_opposite))
    assert jnp.array_equal(exchange.matter_four_force, -radiation)
    assert jnp.array_equal(exchange.balance_residual, jnp.zeros((4,)))
