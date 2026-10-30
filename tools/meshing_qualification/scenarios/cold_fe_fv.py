from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .cold_restore import native_cold_restore


def _fv_worker(phase: str, archive: Path, /) -> dict[str, object]:
    import jax.numpy as jnp

    import phydrax as phx
    from phydrax._model._structure import FiniteVolumeRuntimeStateArchiveRecipe
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
    from phydrax.meshing._model_codec import read_model_artifact, write_model_artifact
    from phydrax.solver._finite_volume_runtime import (
        FiniteVolumeRunStatus,
        FiniteVolumeRuntimeState,
        FiniteVolumeStepPolicy,
        PreparedFiniteVolumeRuntime,
    )
    from phydrax.solver._finite_volume_topology_events import (
        FiniteVolumeTopologyEventJournal,
    )

    def restore_runtime_recipe(
        value: FiniteVolumeRuntimeStateArchiveRecipe,
    ) -> FiniteVolumeRuntimeStateArchiveRecipe:
        return FiniteVolumeRuntimeStateArchiveRecipe(
            value.content_state,
            dict(value.topology_journal_record),
            dict(value.topology_journal_arrays),
            value.accepted_step,
            value.step_size,
            value.last_status,
            value.controller_state,
            value.integrator_state,
            value.output_cursor,
            value.sliding_shift,
            value.sliding_event_id,
            value.sliding_coupling_id,
            value.source_cell_ids,
            value.expected_runtime_id,
            value.expected_epoch_id,
            value.expected_state_id,
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
    discretization = UnstructuredFiniteVolumePlan.from_cell_mesh(
        mesh, component_names=system.component_names
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
        state_id = runtime.accepted_state_id(state, cell_ids)
        recipe = FiniteVolumeRuntimeStateArchiveRecipe(
            state.content_state,
            state.topology_journal.to_archive_record(),
            state.topology_journal.archive_arrays(),
            state.accepted_step,
            state.step_size,
            state.last_status,
            state.controller_state,
            state.integrator_state,
            state.output_cursor,
            state.sliding_shift,
            state.sliding_event_id,
            state.sliding_coupling_id,
            jnp.asarray(cell_ids),
            runtime.runtime_id,
            runtime.topology_epoch_id,
            state_id,
        )
        write_model_artifact(archive, recipe, limits=None)
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
    recipe = read_model_artifact(
        archive,
        expected_type=FiniteVolumeRuntimeStateArchiveRecipe,
        restore=restore_runtime_recipe,
        limits=None,
    )
    if not isinstance(recipe, FiniteVolumeRuntimeStateArchiveRecipe):
        raise TypeError(
            "Cold FV archive did not decode its canonical runtime-state recipe."
        )
    if (
        recipe.expected_runtime_id != runtime.runtime_id
        or recipe.expected_epoch_id != runtime.topology_epoch_id
        or not np.array_equal(np.asarray(recipe.source_cell_ids), cell_ids)
        or recipe.sliding_coupling_id is not None
    ):
        raise RuntimeError(
            "Cold FV archive source/runtime/epoch authority changed during reconstruction."
        )
    journal = FiniteVolumeTopologyEventJournal.from_archive_record(
        dict(recipe.topology_journal_record), dict(recipe.topology_journal_arrays)
    )
    restored = FiniteVolumeRuntimeState(
        recipe.content_state,
        journal,
        recipe.step_size,
        accepted_step=recipe.accepted_step,
        last_status=recipe.last_status,
        controller_state=recipe.controller_state,
        integrator_state=recipe.integrator_state,
        output_cursor=recipe.output_cursor,
        sliding_shift=recipe.sliding_shift,
        sliding_event_id=recipe.sliding_event_id,
    )
    if runtime.accepted_state_id(restored, cell_ids) != recipe.expected_state_id:
        raise RuntimeError(
            "Cold FV accepted-state identity changed during canonical reconstruction."
        )
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
        "restored_state_id": recipe.expected_state_id,
        "accepted_step": int(np.asarray(state.accepted_step)),
        "time": float(np.asarray(state.time)),
        "source_cell_ids": cell_ids.tolist(),
        "previous_total_content": previous_content.tolist(),
        "continued_total_content": continued_content.tolist(),
        "source_integral": np.asarray(source_integral).tolist(),
        "boundary_integral": np.asarray(boundary_integral).tolist(),
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


def native_cold_fe_fv_continuation() -> dict[str, object]:
    from .cold_owner_local import native_cold_owner_local

    component_failures: list[str] = []
    try:
        fe = native_cold_restore()
    except Exception as error:
        fe = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        component_failures.append(f"serial FE restore: {fe['error']}")

    try:
        owner_local_fe = native_cold_owner_local()
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

    missing_contracts = [
        "FE material/history/auxiliary/integrator state is not present in the archived composition or changed-placement owner-local repartition; the completed FE witnesses carry only the P1 u field",
        "FV owner-local material/history/RNG/clock state migration through a changed-placement fresh restart and continued physical solve is not implemented; the FV runtime-state archive is serial",
    ]
    return {
        "status": "failed",
        "workflow": "fresh-process FE and FV physical solve continuation, including owner-local FE repartition",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#cold-fe-fv-continuation",
        "fe_cold_continuation": fe,
        "owner_local_fe_continuation": owner_local_fe,
        "consumer": fe.get("consumer"),
        "producer_process_exited_before_consumer": bool(
            isinstance(owner_local_fe, dict)
            and owner_local_fe.get("producer_process_exited_before_reader")
        ),
        "fresh_process": bool(
            isinstance(owner_local_fe, dict)
            and owner_local_fe.get("fresh_process")
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
        "missing_production_contracts": missing_contracts,
    }


if __name__ == "__main__" and len(sys.argv) == 3:
    print(json.dumps(_fv_worker(sys.argv[1], Path(sys.argv[2])), sort_keys=True))
