from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .cold_restore import native_cold_restore


def _fv_worker(phase: str, archive: Path, /) -> dict[str, object]:

    import phydrax as phx
    from phydrax.discretization import CellMesh
    from phydrax.discretization.finite_volume import (
        PiecewiseConstantReconstruction,
        PreparedUnstructuredFiniteVolumeDynamics,
        RusanovFluxPlan,
        UnstructuredFiniteVolumeBoundarySet,
        UnstructuredFiniteVolumeMethodPlan,
        UnstructuredFiniteVolumePlan,
    )
    from phydrax.equations import (
        compile_conservation_problem,
        ConservationProblemIR,
        EulerSystem,
        IdealGasMaterial,
    )
    from phydrax.solver._finite_volume_runtime import (
        FiniteVolumeRunStatus,
        FiniteVolumeRuntimeState,
        FiniteVolumeStepPolicy,
        PreparedFiniteVolumeRuntime,
    )

    mesh = CellMesh.from_triangles(
        np.asarray(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.5, 0.5),
            ),
            dtype=np.float64,
        ),
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32),
        vertex_global_ids=np.asarray((101, 203, 307, 409, 503), dtype=np.int64),
        cell_global_ids=np.asarray((1009, 1103, 1201, 1301), dtype=np.int64),
        numeric_version="native-cold-fv-source-r1",
    )
    cell_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    system = EulerSystem(2, material=IdealGasMaterial(1.4, 1.0))
    discretization = UnstructuredFiniteVolumePlan(
        mesh.coordinates,
        triangles=mesh.blocks[0].vertices,
        vertex_global_ids=mesh.vertex_global_ids,
        cell_global_ids=cell_ids,
        field_name="conservative",
        component_names=system.component_names,
    ).prepare()
    patch_names = tuple(discretization.boundary_patch_names)
    boundaries = UnstructuredFiniteVolumeBoundarySet(
        patch_names,
        {name: phx.discretization.ExtrapolationBoundary() for name in patch_names},
    )
    method = UnstructuredFiniteVolumeMethodPlan(
        PiecewiseConstantReconstruction(), RusanovFluxPlan()
    )
    dynamics = compile_conservation_problem(
        ConservationProblemIR("native-cold-euler", "conservative", system, boundaries),
        discretization,
        method,
    ).dynamics
    if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
        raise TypeError(
            "Cold FV source compilation did not produce unstructured Euler dynamics."
        )
    runtime = PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(
            2, fallback_flux=phx.discretization.EinfeldtHLLFluxPlan()
        ),
        FiniteVolumeStepPolicy(cfl=0.2),
    )
    case = phx.solver.FiniteVolumeCaseSpec(
        "native-cold-euler",
        runtime,
        phx.solver.FiniteVolumeExecutionSpec(1.0, 1000),
    )
    checkpoint_plan = phx.solver.FiniteVolumeCheckpointPlan(case, runtime=runtime)
    centers = np.asarray(discretization.cell_centers, dtype=np.float64)
    density = 1.0 + 0.15 * np.exp(
        -((centers[:, 0] - 0.5) ** 2 + (centers[:, 1] - 0.5) ** 2) / 0.08
    )
    velocity = np.asarray((0.12, -0.04), dtype=np.float64)
    pressure = 1.0
    energy = pressure / (1.4 - 1.0) + 0.5 * density * float(np.dot(velocity, velocity))
    initial_average = np.column_stack(
        (density, density * velocity[0], density * velocity[1], energy)
    )
    initial_state = runtime.initialize_state(initial_average, 0.0, 1.0e-3)
    source_content = np.sum(
        np.asarray(initial_state.content_state.conservative_content), axis=0
    )
    if phase == "write":
        first = runtime.advance(initial_state)
        if not bool(np.asarray(first.accepted)):
            raise RuntimeError(
                "Cold FV producer did not accept its first Euler solve step."
            )
        state = first.runtime_state
        checkpoint = phx.solver.write_finite_volume_checkpoint(
            archive, checkpoint_plan, state
        )
        state_id = checkpoint.payload_id
        ledger = first.accepted_flux_integrals
        source_integral, boundary_integral, net_integral = ledger.conservation_sums()
        return {
            "phase": phase,
            "pid": __import__("os").getpid(),
            "runtime_id": runtime.runtime_id,
            "epoch_id": runtime.topology_epoch_id,
            "state_id": state_id,
            "accepted_step": int(np.asarray(state.accepted_step)),
            "time": float(np.asarray(state.time)),
            "cell_ids": cell_ids.tolist(),
            "source_content": source_content.tolist(),
            "accepted_content": np.sum(
                np.asarray(state.content_state.conservative_content), axis=0
            ).tolist(),
            "source_integral": np.asarray(source_integral).tolist(),
            "boundary_integral": np.asarray(boundary_integral).tolist(),
            "net_integral": np.asarray(net_integral).tolist(),
            "positivity_valid": bool(np.asarray(first.positivity.fallback_valid))
            and bool(np.asarray(first.positivity.limited_state_valid)),
        }
    if phase != "read":
        raise ValueError(f"Unknown cold FV worker phase {phase!r}.")
    checkpoint = phx.solver.read_finite_volume_checkpoint(archive, checkpoint_plan)
    restored = checkpoint.runtime_state
    if not isinstance(restored, FiniteVolumeRuntimeState):
        raise TypeError("Cold FV checkpoint did not restore its ordinary runtime state.")
    if restored.topology_journal.current_epoch_id != runtime.topology_epoch_id:
        raise RuntimeError("Cold FV checkpoint changed its runtime epoch.")
    if int(np.asarray(restored.last_status)) not in (
        int(FiniteVolumeRunStatus.SUCCESS),
        int(FiniteVolumeRunStatus.RECOVERED_REJECTION),
    ):
        raise RuntimeError("Cold FV archive did not contain an accepted physical state.")
    continued = runtime.advance(restored)
    if not bool(np.asarray(continued.accepted)):
        raise RuntimeError(
            "Cold FV reader did not accept its continued Euler solve step."
        )
    state = continued.runtime_state
    previous_content = np.sum(
        np.asarray(restored.content_state.conservative_content), axis=0
    )
    continued_content = np.sum(
        np.asarray(state.content_state.conservative_content), axis=0
    )
    ledger = continued.accepted_flux_integrals
    source_integral, boundary_integral, net_integral = ledger.conservation_sums()
    ledger_defect = continued_content - previous_content - np.asarray(net_integral)
    ledger_scale = np.maximum(
        1.0, np.maximum(np.abs(previous_content), np.abs(continued_content))
    )
    ledger_tolerance = 128 * np.finfo(continued_content.dtype).eps
    if not np.all(np.abs(ledger_defect) <= ledger_tolerance * ledger_scale):
        raise RuntimeError(
            "Cold FV continuation violates its accepted conservation ledger."
        )
    if (
        int(np.asarray(state.accepted_step))
        != int(np.asarray(restored.accepted_step)) + 1
        or float(np.asarray(state.time)) <= float(np.asarray(restored.time))
        or not np.all(np.isfinite(continued_content))
        or not np.all(np.isfinite(np.asarray(net_integral)))
    ):
        raise RuntimeError(
            "Cold FV solve continuation lost accepted time, step, or conservative content."
        )
    positivity_valid = bool(np.asarray(continued.positivity.fallback_valid)) and bool(
        np.asarray(continued.positivity.limited_state_valid)
    )
    if not positivity_valid or np.any(continued_content[:1] <= 0.0):
        raise RuntimeError("Cold FV continuation lacks positive physical-state evidence.")
    return {
        "phase": phase,
        "pid": __import__("os").getpid(),
        "runtime_id": runtime.runtime_id,
        "epoch_id": runtime.topology_epoch_id,
        "restored_state_id": checkpoint.payload_id,
        "accepted_step": int(np.asarray(state.accepted_step)),
        "time": float(np.asarray(state.time)),
        "source_cell_ids": cell_ids.tolist(),
        "previous_total_content": previous_content.tolist(),
        "continued_total_content": continued_content.tolist(),
        "source_integral": np.asarray(source_integral).tolist(),
        "boundary_integral": np.asarray(boundary_integral).tolist(),
        "conservation_ledger_defect": ledger_defect.tolist(),
        "net_integral": np.asarray(net_integral).tolist(),
        "positivity_valid": positivity_valid,
        "accepted": bool(np.asarray(continued.accepted)),
    }


def _run_child(phase: str, archive: Path, /) -> dict[str, object]:
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "tools.meshing_qualification.scenarios.cold_fe_fv",
            phase,
            str(archive),
        ),
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Cold FV {phase} process failed:\n{completed.stdout}{completed.stderr}"
        )
    lines = [
        line for line in completed.stdout.splitlines() if line.strip().startswith("{")
    ]
    if not lines:
        raise RuntimeError(f"Cold FV {phase} process returned no JSON evidence.")
    record = json.loads(lines[-1])
    if not isinstance(record, dict):
        raise RuntimeError(f"Cold FV {phase} process returned a non-object record.")
    return record


def _owner_local_fv_continuation() -> dict[str, object]:
    import os
    import socket

    from phydrax import config as pxconfig
    from phydrax._execution_launch import (
        launch_local_processes,
        LocalProcessLaunchPlan,
    )

    repository = Path(__file__).resolve().parents[3]
    base = dict(os.environ)
    for name in (
        "PHYDRAX_COORDINATOR_ADDRESS",
        "PHYDRAX_NUM_PROCESSES",
        "PHYDRAX_PROCESS_ID",
        "PHYDRAX_LOCAL_DEVICE_IDS",
        "PHYDRAX_COORDINATOR_BIND_ADDRESS",
    ):
        base.pop(name, None)
    environment = pxconfig.environment(
        pxconfig.RuntimeConfig(
            enable_x64=True,
            platforms=("cpu",),
            cpu_device_count=1,
            cpu_collectives_implementation="gloo",
        ),
        base=base,
    )
    environment.update(
        PYTHONPATH=str(repository),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )

    def launch(root: Path, phase: str) -> list[dict[str, object]]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        results = launch_local_processes(
            (
                sys.executable,
                "-m",
                "tools.meshing_qualification.scenarios.cold_owner_local_fv",
                str(root),
                phase,
            ),
            LocalProcessLaunchPlan(3, f"127.0.0.1:{port}", devices_per_process=1),
            environment=environment,
            cwd=str(repository),
            timeout_seconds=3600.0,
        )
        reports = [
            json.loads(result.stdout.decode().strip().splitlines()[-1])
            for result in results
        ]
        if (
            any(not isinstance(report, dict) for report in reports)
            or [report.get("rank") for report in reports] != [0, 1, 2]
            or any(report.get("phase") != phase for report in reports)
        ):
            raise RuntimeError(f"Malformed owner-local FV {phase} reports.")
        return reports

    with TemporaryDirectory(prefix="phydrax-native-owner-local-fv-") as temporary:
        root = Path(temporary)
        writers = launch(root, "write")
        archives = [root / f"owner-local-fv-state-{rank}.phx" for rank in range(3)]
        if any(not path.is_file() or path.stat().st_size == 0 for path in archives):
            raise RuntimeError("Owner-local FV writer omitted a canonical archive.")
        archive_bytes = [path.stat().st_size for path in archives]
        readers = launch(root, "read")
    writer_pids = {report["pid"] for report in writers}
    reader_pids = {report["pid"] for report in readers}
    if len(writer_pids) != 3 or len(reader_pids) != 3 or writer_pids & reader_pids:
        raise RuntimeError("Owner-local FV producer and reader processes are not independent.")
    for writer, reader in zip(writers, readers, strict=True):
        if any(
            writer[key] != reader[key]
            for key in ("state_id", "source_runtime_id", "target_connectivity_id")
        ):
            raise RuntimeError("Owner-local FV cold restore changed scientific authority.")
        if reader["source_partition"] == reader["target_partition"]:
            raise RuntimeError("Owner-local FV continuation did not change placement.")
    return {
        "status": "passed",
        "producer_processes_exited_before_reader": True,
        "archive_bytes_by_rank": archive_bytes,
        "producer": writers,
        "reader": readers,
    }


def native_cold_fe_fv_continuation() -> dict[str, object]:
    from .cold_owner_local import native_cold_owner_local

    component_failures: list[str] = []
    try:
        fe = native_cold_restore()
        if fe.get("status") != "passed" or not fe.get("fresh_process"):
            raise RuntimeError(f"Serial FE restore did not pass: {fe!r}")
    except Exception as error:
        fe = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        component_failures.append(f"serial FE restore: {fe['error']}")

    try:
        owner_local_fe = native_cold_owner_local()
        profiles = owner_local_fe.get("profiles")
        if (
            owner_local_fe.get("status") != "passed"
            or not owner_local_fe.get("producer_processes_exited_before_each_reader")
            or not isinstance(profiles, list)
            or not profiles
            or any(
                not isinstance(profile, dict)
                or profile.get("status") != "passed"
                or not isinstance(profile.get("physical_cone_continuation_by_rank"), list)
                or len(profile["physical_cone_continuation_by_rank"]) != 3
                or any(
                    not isinstance(cone, dict)
                    or not all(
                        cone.get(key) is True
                        for key in (
                            "changed_placement",
                            "independent_reference",
                            "newmark",
                            "linear_momentum",
                            "material_commit_trial_version",
                            "rollback",
                            "history_velocity_acceleration_timestep_sensitive",
                        )
                    )
                    for cone in profile["physical_cone_continuation_by_rank"]
                )
                or not isinstance(profile.get("collective_oracles"), dict)
                or not all(
                    profile["collective_oracles"].get(key) is True
                    for key in (
                        "stale_migration_refused_without_partial_commit",
                        "one_rank_restart_layout_refusal_without_partial_commit",
                        "saved_source_reused_after_both_refusals",
                        "history_clock_integrator_rng_material_rebound",
                        "next_compiled_fe_solve_accepted",
                    )
                )
                for profile in profiles
            )
        ):
            raise RuntimeError("Owner-local FE physical restart evidence is incomplete.")
    except Exception as error:
        owner_local_fe = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        component_failures.append(
            f"owner-local FE continuation: {owner_local_fe['error']}"
        )

    writer: dict[str, object] | None = None
    reader: dict[str, object] | None = None
    archive_bytes: int | None = None
    try:
        with TemporaryDirectory(prefix="phydrax-native-cold-fv-") as temporary:
            archive = Path(temporary) / "accepted-finite-volume-runtime.phx"
            writer = _run_child("write", archive)
            if not archive.is_file() or archive.stat().st_size == 0:
                raise RuntimeError(
                    "Cold FV writer did not publish its canonical runtime archive."
                )
            reader = _run_child("read", archive)
            archive_bytes = archive.stat().st_size
        if writer["pid"] == reader["pid"]:
            raise RuntimeError("Cold FV continuation reused the producer process.")
        if (
            writer["runtime_id"] != reader["runtime_id"]
            or writer["epoch_id"] != reader["epoch_id"]
            or writer["state_id"] != reader["restored_state_id"]
            or writer["cell_ids"] != reader["source_cell_ids"]
            or reader["phase"] != "read"
        ):
            raise RuntimeError(
                "Cold FV continuation changed accepted runtime/state authority."
            )
        if not reader["accepted"] or not reader["positivity_valid"]:
            raise RuntimeError(
                "Cold FV fresh-process continuation did not accept its physical step."
            )
    except Exception as error:
        component_failures.append(
            f"serial FV continuation: {type(error).__name__}: {error}"
        )

    try:
        owner_local_fv = _owner_local_fv_continuation()
        if (
            owner_local_fv.get("status") != "passed"
            or not owner_local_fv.get("producer_processes_exited_before_reader")
        ):
            raise RuntimeError(f"Owner-local FV restart did not pass: {owner_local_fv!r}")
    except Exception as error:
        owner_local_fv = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
        component_failures.append(
            f"owner-local FV continuation: {owner_local_fv['error']}"
        )
    return {
        "status": "failed" if component_failures else "passed",
        "workflow": "fresh-process FE and FV physical solve continuation, including owner-local FE and FV repartition",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#cold-fe-fv-continuation",
        "fe_cold_continuation": fe,
        "owner_local_fe_continuation": owner_local_fe,
        "owner_local_fv_continuation": owner_local_fv,
        "consumer": fe.get("consumer"),
        "producer_process_exited_before_consumer": bool(
            fe.get("producer_process_exited_before_consumer")
            and owner_local_fe.get("producer_processes_exited_before_each_reader")
            and owner_local_fv.get("producer_processes_exited_before_reader")
            and writer is not None
            and reader is not None
            and writer.get("pid") != reader.get("pid")
        ),
        "fresh_process": bool(
            fe.get("fresh_process")
            and owner_local_fe.get("producer_processes_exited_before_each_reader")
            and owner_local_fv.get("producer_processes_exited_before_reader")
            and writer is not None
            and reader is not None
            and writer.get("pid") != reader.get("pid")
        ),
        "finite_volume_writer": writer,
        "finite_volume_reader": reader,
        "archive_path": "temporary native model artifact",
        "archive_bytes": archive_bytes,
        "scientific_oracles": {
            "fe_affine_harmonic_solution_tolerance": 2.0e-9,
            "fe_transfer_and_conservative_integral_tolerance": 1.0e-12,
            "owner_local_fe_changed_placement_continuation": owner_local_fe,
            "fv_state_identity_kind": "finite-volume-checkpoint-payload-id",
            "fv_accepted_state_identity": None
            if writer is None
            else writer.get("state_id"),
            "fv_positive_state": None
            if reader is None
            else reader.get("positivity_valid"),
            "fv_conservation_ledger": None
            if reader is None
            else reader.get("net_integral"),
        },
        "component_failures": component_failures,
        "missing_production_contracts": [],
    }


if __name__ == "__main__" and len(sys.argv) == 3:
    print(json.dumps(_fv_worker(sys.argv[1], Path(sys.argv[2])), sort_keys=True))
