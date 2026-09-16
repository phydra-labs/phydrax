#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import capture_environment
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
from phydrax.metrix import ADMGridGeometry, RelativityConvention, StressEnergyProjection
from phydrax.solver._dark_sector_epoch_runtime import (
    DarkSectorEpochPlan,
    empty_dark_sector_epoch_state,
)


_COMPONENT_NAMES = (
    "relativistic-matter",
    "quantum-kinetics",
    "coherent-transport",
    "off-shell-transport",
    "shower-products",
    "hadronization-products",
    "bound-state-products",
    "radiation",
)


def _stage(size: int) -> tuple[FullDarkSectorStageToken, DarkSectorEpochPlan]:
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
    identity = jnp.broadcast_to(jnp.eye(3), (size, 3, 3))
    geometry = ADMGridGeometry(
        jnp.ones((size,)),
        jnp.zeros((size, 3)),
        identity,
        identity,
        jnp.ones((size,)),
        jnp.zeros((size, 3, 3)),
        jnp.ones((size,), dtype=bool),
        jnp.ones((size,), dtype=bool),
        snapshot_token=jnp.asarray(1, dtype=jnp.int32),
        chart_id="benchmark-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id=f"benchmark-grid-{size}",
        geometry_lineage_id="full-dark-sector-benchmark-adm",
    )
    frame = LocalRelativisticFramePlan.from_adm(
        geometry,
        units,
        jnp.asarray((0.0, 0.0, 0.0, 0.0)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="benchmark-eulerian-observer",
        orientation_id="right-handed-future",
    )
    capacity = max(size, 8)
    epoch_plan = DarkSectorEpochPlan(
        packet_capacity=capacity,
        event_capacity=capacity,
        product_capacity=capacity,
        radiation_capacity=capacity,
        work_capacity=capacity,
        frontier_capacity=capacity,
        packet_width=8,
        event_width=8,
        product_width=8,
        radiation_width=8,
        work_width=8,
        frontier_width=8,
        species_revision_id="1" * 64,
        topology_revision_id="2" * 64,
    )
    epoch = empty_dark_sector_epoch_state(
        epoch_plan,
        epoch_sequence=0,
        parent_epoch_manifest_id=None,
    )
    revision = MatrixElementRevision(
        "benchmark-process",
        "benchmark-model",
        "native",
        "benchmark-rights",
        "covariant-normalization",
        "fixed-support",
        "fixed-proposal",
        "no-adaptation",
        "benchmark-training",
        "fixed-optimizer",
        "benchmark-error-model",
        "3" * 64,
        differentiation_id="fixed-profile",
    )
    return FullDarkSectorStageToken(frame, epoch, revision), epoch_plan


def _components(
    stage: FullDarkSectorStageToken, size: int, count: int, /
) -> tuple[NamedStressEnergyComponent, ...]:
    active = jnp.ones((size,), dtype=bool)
    zeros = jnp.zeros((size,))
    values = []
    for index, name in enumerate(_COMPONENT_NAMES[:count]):
        energy = jnp.full((size,), 1.0 / (index + 1))
        momentum = jnp.zeros((size, 3)).at[:, index % 3].set(0.01 * energy)
        stress = jnp.broadcast_to(
            (0.1 / (index + 1)) * jnp.eye(3),
            (size, 3, 3),
        )
        projection = StressEnergyProjection(
            energy,
            momentum,
            stress,
            active,
            active,
            zeros,
            zeros,
            snapshot_token=stage.geometry_snapshot_token,
            geometry_lineage_id="full-dark-sector-benchmark-adm",
            convention_id=(
                RelativityConvention(metric_signature="mostly_minus").convention_id
            ),
            scale_id=RelativityScaleContract(
                DimensionalScaleContract.si(), 1, 3, 2, 1, True
            ).scale_id,
            topology_id=stage.topology_id,
            projection_id=f"benchmark-{name}",
        )
        values.append(
            NamedStressEnergyComponent(
                name,
                projection,
                conservation_defect=0.0,
                constraint_defect=0.0,
                gauge_defect=0.0,
                entropy_production=0.0,
                unitarity_defect=0.0,
                evidence_valid=True,
                evidence_id=f"benchmark-evidence-{name}",
            )
        )
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--components", type=int, default=len(_COMPONENT_NAMES))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.size < 1
        or not 1 <= arguments.components <= len(_COMPONENT_NAMES)
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError(
            "Positive size/repeats and a supported component count are required."
        )

    stage, epoch_plan = _stage(arguments.size)
    components = _components(stage, arguments.size, arguments.components)
    radiation_force = jnp.zeros((arguments.size, 4)).at[:, 0].set(0.01)
    exchange = DarkRadiationFourForce.paired(
        radiation_force,
        stage.time,
        stage.frame_token,
        source_state_id="benchmark-radiation-state",
        frame_id=stage.frame_id,
        frame_realization_id=stage.frame_realization_id,
        unit_contract_id=stage.unit_contract_id,
    )

    for _ in range(arguments.warmup):
        warmup = assemble_full_dark_sector_stress(components, stage)
        jax.block_until_ready(warmup.total.energy_density)

    durations = []
    result = assemble_full_dark_sector_stress(components, stage)
    for _ in range(arguments.repeats):
        started = time.perf_counter()
        result = assemble_full_dark_sector_stress(components, stage)
        jax.block_until_ready(result.total.energy_density)
        durations.append(time.perf_counter() - started)

    component_bytes = sum(
        value.projection.energy_density.nbytes
        + value.projection.momentum_covector.nbytes
        + value.projection.stress_covariant.nbytes
        + value.projection.active.nbytes
        + value.projection.valid.nbytes
        for value in components
    )
    total_bytes = (
        result.total.energy_density.nbytes
        + result.total.momentum_covector.nbytes
        + result.total.stress_covariant.nbytes
        + result.total.active.nbytes
        + result.total.valid.nbytes
    )
    epoch_value_bytes = sum(
        epoch_plan.capacity(name)
        * epoch_plan.width(name)
        * np.dtype(epoch_plan.value_dtype).itemsize
        for name in ("packet", "event", "product", "radiation", "work", "frontier")
    )
    payload = {
        "benchmark": "full-dark-sector-runtime",
        "environment": capture_environment().to_dict(),
        "support": {
            "surface": "full-stage-stress-four-force-integration",
            "component_names": list(result.component_names),
            "grid_points": arguments.size,
            "fixed_capacity": True,
            "durable_unbounded_epochs": True,
        },
        "performance": {
            "repeat_count": arguments.repeats,
            "seconds_mean": float(np.mean(durations)),
            "seconds_minimum": float(np.min(durations)),
            "seconds_maximum": float(np.max(durations)),
            "grid_points_per_second": float(arguments.size / np.mean(durations)),
        },
        "resources": {
            "component_projection_bytes": int(component_bytes),
            "total_projection_bytes": int(total_bytes),
            "epoch_value_bytes": int(epoch_value_bytes),
            "packet_capacity": epoch_plan.packet_capacity,
            "event_capacity": epoch_plan.event_capacity,
            "product_capacity": epoch_plan.product_capacity,
            "radiation_capacity": epoch_plan.radiation_capacity,
            "work_capacity": epoch_plan.work_capacity,
            "frontier_capacity": epoch_plan.frontier_capacity,
            "capacity_revision_id": epoch_plan.capacity_revision_id,
            "compile_signature_id": epoch_plan.compile_signature_id,
        },
        "evidence": {
            "successful": bool(result.successful),
            "stage_consistent": bool(result.stage_consistent),
            "finite": bool(result.finite),
            "four_force_exact": bool(jnp.all(exchange.exact_opposite)),
            "four_force_maximum_residual": float(
                jnp.max(jnp.abs(exchange.balance_residual))
            ),
            "total_energy": float(jnp.sum(result.total.energy_density)),
            "maximum_projection_defect": float(jnp.max(result.total.projection_defect)),
            "maximum_conservation_defect": float(
                jnp.max(result.total.conservation_defect)
            ),
            "stage_id": stage.stage_id,
            "frame_id": stage.frame_id,
            "frame_realization_id": stage.frame_realization_id,
            "unit_contract_id": stage.unit_contract_id,
            "topology_id": stage.topology_id,
            "matrix_element_revision_id": stage.matrix_element_revision_id,
        },
    }
    if (
        not payload["evidence"]["successful"]
        or not payload["evidence"]["four_force_exact"]
    ):
        raise SystemExit(1)
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(serialized)
    else:
        arguments.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
