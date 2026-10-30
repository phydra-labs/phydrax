from __future__ import annotations

import json
import os
import socket
import subprocess
import sys


def _worker() -> dict[str, object]:
    import jax

    rank = int(os.environ["PHYDRAX_NATIVE_DISTRIBUTED_RANK"])
    coordinator = os.environ["PHYDRAX_NATIVE_DISTRIBUTED_COORDINATOR"]
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=2,
        process_id=rank,
    )
    try:
        import numpy as np
        from jax.experimental import multihost_utils

        import phydrax as phx
        from phydrax._execution_resources import ExecutionGroupSpec
        from phydrax._execution_runtime import ExecutionGroup
        from phydrax._physical import SpatialCoordinateContract
        from phydrax.graph import GraphPartitionPlan
        from phydrax.meshing._contracts import (
            CellFamilyPolicy,
            CellMeshingTarget,
            MeshingScope,
            SizeCompliancePolicy,
            SurfaceMeshingSpec,
            UniformSizeControl,
        )
        from phydrax.meshing._periodic import PeriodicCell, PeriodicPointOrbits
        from phydrax.meshing._sizing import SizeControlStrength
        from phydrax.meshing.providers import NativeMeshingOptions
        from phydrax.meshing.providers._native_periodic import (
            NativePeriodicSource,
            prepare_native_periodic,
        )

        devices = tuple(sorted(jax.devices(), key=lambda device: device.process_index))
        members = tuple(device.process_index for device in devices)
        if jax.process_count() != 2 or members != (0, 1):
            raise RuntimeError(
                f"Native distributed qualification requires two actual JAX processes; observed {jax.process_count()} processes over {members}."
            )
        group = ExecutionGroup(
            ExecutionGroupSpec(
                "native-distributed-periodic-qualification",
                members,
                tuple((device.process_index, device.id) for device in devices),
                mesh_axes=(("periodic_owner", len(members)),),
            ),
            devices,
        )
        cell = PeriodicCell(np.asarray(((1.2, 0.1), (0.35, 1.05)), dtype=np.float64))
        fractional = np.asarray(
            (
                (0.11, 0.13),
                (0.58, 0.17),
                (0.31, 0.62),
                (0.84, 0.77),
                (0.21, 0.88),
                (0.73, 0.39),
                (0.44, 0.42),
                (0.92, 0.05),
            ),
            dtype=np.float64,
        )
        point_ids = np.asarray((91, 4, 53, 17, 28, 61, 6, 43), dtype=np.int64)
        point_owners = np.asarray((0, 1, 0, 1, 0, 1, 0, 1), dtype=np.int32)
        local_points = point_owners == rank
        requests = (
            (
                "native-distributed-routine",
                0.43,
                (0.5, 0.5),
                "routine-periodic-generation-adaptation-repartition",
            ),
            (
                "native-distributed-larger",
                0.28,
                (0.75, 0.25),
                "larger-periodic-generation-adaptation-repartition",
            ),
        )
        records: list[dict[str, object]] = []
        for source_id, size, shares, profile in requests:
            revision = f"{source_id}-revision-1"
            source = NativePeriodicSource(
                PeriodicPointOrbits(
                    cell,
                    np.asarray(
                        cell.cartesian(fractional[local_points]), dtype=np.float64
                    ),
                    point_ids[local_points],
                ),
                source_id,
                revision,
                point_owner_ranks=point_owners[local_points],
            )
            scope = MeshingScope(
                source.source_id,
                source.source_revision,
                phx.meshing.MeshingEntityKind.GEOMETRY,
                2,
                source.scope_entity_set_id(2),
                source.scope_entity_ids(2),
            )
            specification = SurfaceMeshingSpec(
                CellMeshingTarget(2, 2, CellFamilyPolicy(required=("triangle",))),
                scope,
                size_controls=(
                    UniformSizeControl(scope, size, strength=SizeControlStrength.SOFT),
                ),
                size_compliance=SizeCompliancePolicy(
                    relative_tolerance=0.8,
                    target_statistics=("p50",),
                ),
            )
            plan = prepare_native_periodic(
                source,
                specification,
                NativeMeshingOptions("periodic_delaunay"),
                SpatialCoordinateContract.si(),
            )
            graph_plan = GraphPartitionPlan(
                2,
                target_shares=shares,
                maximum_imbalance=1.5,
                refinement_passes=4,
            )
            assembly = plan.execute(
                execution_group=group,
                graph_partition_plan=graph_plan,
            )
            if not assembly.owner_local or assembly.local_rank != rank:
                raise RuntimeError(
                    "Native periodic execution materialized a non-owner-local assembly."
                )
            if len(assembly.parts) != 1:
                raise RuntimeError(
                    "Native periodic assembly must expose one rank-local part."
                )
            part = assembly.parts[0]
            view = part.carrier
            directory_data = view.owner_directory_data
            if (
                part.name != source_id
                or view.owner_rank != rank
                or directory_data.rank != rank
                or directory_data.group_id != group.spec.group_id
                or directory_data.part_count != len(members)
                or not directory_data.source_manifest_id
                or not directory_data.source_directory_id
                or not directory_data.cell_directory_id
                or not directory_data.vertex_directory_id
            ):
                raise RuntimeError(
                    "Native periodic result lost its compact owner-directory data."
                )
            view.require_owner_directory_current(
                group, axis_name=f"native_distributed_{profile}_directory"
            )
            local_owned_cell_ids = np.asarray(view.owned_cell_ids, dtype=np.int64)
            local_scope = view.scope(2, local_owned_cell_ids)
            if not np.array_equal(
                np.sort(np.asarray(local_scope.entity_ids, dtype=np.int64)),
                np.sort(local_owned_cell_ids),
            ):
                raise RuntimeError("Compact local scope changed the owned cell IDs.")
            if local_owned_cell_ids.size == 0:
                raise RuntimeError(
                    f"Native graph partition left rank {rank} empty for {profile}."
                )
            owned = np.asarray(view.owned, dtype=np.bool_)
            measures = np.asarray(view.cell_measures, dtype=np.float64)
            local_measure = float(np.sum(measures[owned]))
            local_summary = np.asarray(
                (local_owned_cell_ids.size, local_measure), dtype=np.float64
            )
            all_summaries = np.asarray(
                multihost_utils.process_allgather(local_summary)
            ).reshape((2, 2))
            owner_counts = all_summaries[:, 0].astype(np.int64)
            quotient_measure = float(np.sum(all_summaries[:, 1]))
            if not np.isclose(
                quotient_measure, cell.cell_measure, rtol=0.0, atol=1.0e-10
            ):
                raise RuntimeError(
                    f"Owner-local periodic measure {quotient_measure} differs from analytic lattice measure {cell.cell_measure}."
                )
            assembly_ids = np.asarray(
                multihost_utils.process_allgather(
                    np.frombuffer(bytes.fromhex(assembly.assembly_id), dtype=np.uint8)
                )
            )
            if not np.all(assembly_ids == assembly_ids[0]):
                raise RuntimeError(
                    "Owner-local ranks disagree on the global assembly identity."
                )
            records.append(
                {
                    "profile": profile,
                    "source_id": source.source_id,
                    "source_revision": source.source_revision,
                    "plan_id": plan.plan_id,
                    "graph_partition_id": graph_plan.plan_id,
                    "assembly_id": assembly.assembly_id,
                    "assembly_view_id": assembly.assembly_view_id,
                    "part_id": part.part_id,
                    "owner_directory_id": view.distribution_id,
                    "owner_directory_source_manifest_id": directory_data.source_manifest_id,
                    "source_point_directory_id": directory_data.source_directory_id,
                    "cell_directory_id": directory_data.cell_directory_id,
                    "vertex_directory_id": directory_data.vertex_directory_id,
                    "owner_local": True,
                    "global_mesh_materialized": False,
                    "rank": rank,
                    "owned_cell_ids": local_owned_cell_ids.tolist(),
                    "all_rank_owned_cell_counts": owner_counts.tolist(),
                    "all_rank_cell_counts_positive": bool(np.all(owner_counts > 0)),
                    "quotient_measure": quotient_measure,
                    "analytic_lattice_measure": cell.cell_measure,
                    "periodic_cell_count": int(np.asarray(view.periodic_cell_ids).size),
                }
            )
        placement_changed = (
            records[0]["all_rank_owned_cell_counts"]
            != records[1]["all_rank_owned_cell_counts"]
        )
        if not placement_changed:
            raise RuntimeError(
                "The two graph target-share requests did not change owner placement."
            )
        return {
            "status": "passed",
            "rank": rank,
            "process_count": jax.process_count(),
            "devices": [
                {
                    "process_index": device.process_index,
                    "id": device.id,
                    "platform": device.platform,
                }
                for device in devices
            ],
            "profiles": records,
            "changed_placement_observed": placement_changed,
        }
    finally:
        jax.distributed.shutdown()


def native_distributed_lifecycle() -> dict[str, object]:
    from .._runtime import require_meshcore, runtime_identity

    meshcore = require_meshcore()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PHYDRAX_NATIVE_DISTRIBUTED_COORDINATOR"] = coordinator
    environment["PHYDRAX_NATIVE_DISTRIBUTED_WORKER"] = "1"
    processes: list[subprocess.Popen[str]] = []
    outputs: list[tuple[str, str]] = []
    try:
        for rank in (0, 1):
            worker_environment = dict(environment)
            worker_environment["PHYDRAX_NATIVE_DISTRIBUTED_RANK"] = str(rank)
            processes.append(
                subprocess.Popen(
                    (
                        sys.executable,
                        "-m",
                        "tools.meshing_qualification.scenarios.native_distributed",
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=worker_environment,
                )
            )
        outputs = [process.communicate() for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
    worker_records: list[dict[str, object]] = []
    failures: list[str] = []
    for rank, (process, (stdout, stderr)) in enumerate(
        zip(processes, outputs, strict=True)
    ):
        if process.returncode != 0:
            failures.append(
                f"rank {rank} exited {process.returncode}: {stderr.strip() or stdout.strip()}"
            )
            continue
        record_lines = [
            line for line in stdout.splitlines() if line.strip().startswith("{")
        ]
        if not record_lines:
            failures.append(f"rank {rank} returned no structured distributed record.")
            continue
        try:
            worker_records.append(json.loads(record_lines[-1]))
        except json.JSONDecodeError as error:
            failures.append(
                f"rank {rank} returned malformed JSON: {error}: {record_lines[-1]}"
            )
    if len(worker_records) != 2:
        return {
            "status": "failed",
            "workflow": "two-process native periodic generation/adaptation/repartition",
            "source_registry": "tools/meshing_qualification/native_case_migration.json#distributed-lifecycle",
            "worker_records": worker_records,
            "worker_failures": failures,
            "missing_production_contracts": [
                "two-process owner-local native generation/adaptation/repartition completed successfully on every owner"
            ],
            "runtime": runtime_identity(),
            "meshcore_identity": meshcore,
        }
    if any(record.get("status") != "passed" for record in worker_records):
        failures.append(
            "At least one process did not accept its native owner-local witness."
        )
    expected_ranks = sorted(int(record["rank"]) for record in worker_records)
    if expected_ranks != [0, 1]:
        failures.append(
            f"Distributed worker records came from unexpected ranks {expected_ranks}."
        )
    return {
        "status": "failed",
        "workflow": "two-process owner-local periodic generation/adaptation/repartition with changed placement",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#distributed-lifecycle",
        "worker_records": worker_records,
        "worker_failures": failures,
        "missing_production_contracts": [
            "migrate all required FE/FV fields, material/history, RNG and clock through repartition, archive owner-local state, restart with changed placement in fresh processes, and continue a physical solve; geometric redistribution alone is not this continuation"
        ],
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }


if __name__ == "__main__" and os.environ.get("PHYDRAX_NATIVE_DISTRIBUTED_WORKER") == "1":
    print(json.dumps(_worker(), sort_keys=True))
