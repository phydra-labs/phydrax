#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _configuration():
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4),
            phx.discretization.UniformCellAxisSpec(4),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    signature = phx.discretization.PatchShapeSignature((4, 4), halo_width=1)
    hierarchy = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0, (phx.discretization.PatchBucketPlan(signature, 1),)
            ),
            phx.discretization.VariablePatchLevelPlan(
                1, (phx.discretization.PatchBucketPlan(signature, 4),)
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0, 0), (4, 4)),),
    )
    compiler = phx.discretization.VariablePatchTopologyCompiler(hierarchy)
    initial = compiler.initial_topology()
    tags = ((jnp.zeros((1, 4, 4), dtype="bool").at[0, 1, 1].set(True),),)
    compiled = compiler.compile(initial, tags)
    topology = compiled.topology
    capacity = phx.discretization.BlockHierarchyCapacityPlan(
        ((32, 64, 32), (96, 160, 64)),
        (80, 64, 240, 160),
    )
    entities = phx.discretization.VariablePatchEntityComplexPlan(capacity).prepare(
        topology
    )
    geometry_plan = phx.discretization.VariablePatchGeometryPlan(
        topology,
        lambda point, time, args: (1.0 + 0.05 * time) * point,
        "qualified-dilation",
        geometry_family_id="mapped-ale",
    )
    geometry = geometry_plan.state(0.25, revision=1)
    return compiled, entities, geometry_plan, geometry


def qualify() -> dict[str, object]:
    compiled, entities, geometry_plan, geometry = _configuration()
    entity_chain = all(
        complex_.complex.dimension == 2 and len(complex_.complex.incidences) == 2
        for complex_ in entities
    )
    transfers = phx.discretization.CompatibleEntityTransferFamily(
        entities[0],
        entities[1],
        compiled.topology.plan.levels[0].refinement_ratio,
        (256, 256, 256),
    )
    commuting = all(
        transfer.evidence.commuting_defect <= 1.0e-12
        and transfer.evidence.roundtrip_defect <= 1.0e-12
        for transfer in transfers.transfers
    )
    ale = phx.discretization.VariablePatchALEPlan(
        geometry_plan,
        endpoint_tolerance=1.0e-10,
    ).prepare_step(geometry_plan.state(0.0), 0.1)
    embedded = phx.discretization.VariablePatchEmbeddedBoundaryPlan(
        geometry_plan.plan_id,
        lambda points, time, args: points[:, 0] - 0.45,
        "qualified-body",
        minimum_volume_fraction=0.3,
    ).prepare(geometry)
    partition = phx.discretization.VariablePatchPartitionPlan(
        1,
        tuple(
            tuple(bucket.lane_capacity for bucket in level.buckets)
            for level in compiled.topology.plan.levels
        ),
    ).prepare(compiled.topology)
    gates = {
        "topology": bool(compiled.status.successful),
        "entity_chain": entity_chain,
        "compatible_entity_transfer": commuting,
        "mapped_geometry": bool(geometry.valid),
        "ale_geometry": bool(ale.successful),
        "embedded_volume_closure": embedded.evidence.maximum_volume_closure_defect
        <= 1.0e-12,
        "embedded_face_closure": embedded.evidence.maximum_face_closure_defect <= 1.0e-12,
        "partition": partition.evidence.maximum_imbalance == 0.0,
    }
    return {
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "evidence": {
            "epoch_id": compiled.topology.epoch.epoch_id,
            "topology_id": compiled.topology.topology_id,
            "layout_id": compiled.topology.layout_id,
            "patch_evidence_id": compiled.evidence.evidence_id,
            "entity_complex_ids": [value.complex_id for value in entities],
            "compatible_transfer_family_id": transfers.family_id,
            "geometry_plan_id": geometry_plan.plan_id,
            "maximum_gcl_defect": float(
                max(
                    np.max(np.asarray(value), initial=0.0)
                    for level in geometry.gcl_defects
                    for value in level
                )
            ),
            "ale_evidence_id": ale.evidence.evidence_id,
            "embedded_evidence_id": embedded.evidence.evidence_id,
            "partition_evidence_id": partition.evidence.evidence_id,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualify()
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
    print(payload)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
