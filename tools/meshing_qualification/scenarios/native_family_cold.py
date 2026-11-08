from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import phydrax as phx
from phydrax.meshing._model_codec import (
    read_model_artifact,
    restore_model_archive_value,
    write_model_artifact,
)
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing.providers import NativeMeshingProvider
from phydrax.solver._finite_element_schedule import (
    FiniteElementAcceptedFieldLayout,
    FiniteElementAcceptedState,
    FiniteElementAcceptedStepSchedule,
    FiniteElementAttemptResult,
    FiniteElementFieldTransferSemantics,
    FiniteElementRestartManifest,
)

from .native_family_matrix import (
    _CONTRACT,
    _make_profile,
    _solve_family_affine_state,
    _solve_quarter_quadratic,
)


def _physical_state(
    result: CellMeshingResult, profile_id: str
) -> tuple[dict[str, object], np.ndarray, Any]:
    if profile_id == "native-family-lifecycle:curved-quarter-extrusion-quad":
        return _solve_quarter_quadratic(result)
    return _solve_family_affine_state(result)


def write_family(
    directory: Path, profile_id: str, *, captured_result: Path | None = None
) -> dict[str, object]:
    case = _make_profile(profile_id)
    if captured_result is None:
        result = (
            NativeMeshingProvider(case.options)
            .plan(case.source, case.specification, coordinate_contract=_CONTRACT)
            .execute()
        )
    else:
        result = read_model_artifact(
            captured_result,
            expected_type=CellMeshingResult,
            restore=restore_model_archive_value,
            limits=None,
        )
    result.audit.require_passed()
    if result.certification is None:
        raise ValueError("Family producer omitted its certification.")
    result.certification.require_passed()
    if (
        result.certification.inputs.source_id,
        result.certification.inputs.source_revision,
    ) != (
        case.source.source_id,
        case.source.source_revision,
    ):
        raise ValueError("Family capture does not bind the authored source.")
    if result.compliance.specification_id != case.specification.specification_id:
        raise ValueError("Family capture does not bind the authored request.")
    if not result.compliance.passed:
        raise ValueError("Family producer did not satisfy its authored controls.")
    record, values, field = _physical_state(result, profile_id)
    layout = FiniteElementAcceptedFieldLayout(
        field,
        transfer_semantics=FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION,
    )
    state = FiniteElementAcceptedState(
        (values,),
        np.asarray(0.0, dtype=np.float64),
        0,
        result.mesh.topology_id,
        result.result_id,
        "qualification:family-physical-field",
        field_layouts=(layout,),
        state_version=1,
    )
    event = result.surface_quad_field_event
    write_model_artifact(directory / "family-result.phx", result, limits=None)
    write_model_artifact(
        directory / "family-state.phx", FiniteElementRestartManifest(state), limits=None
    )
    return {
        "profile_id": profile_id,
        "pid": os.getpid(),
        "result_id": result.result_id,
        "topology_id": result.mesh.topology_id,
        "accepted_id": state.accepted_id,
        "source_id": case.source.source_id,
        "source_revision": case.source.source_revision,
        "surface_quad_field_event": None
        if event is None
        else {
            "event_id": event.event_id,
            "field_id": event.field.field_id,
            "layout_id": event.layout.layout_id,
        },
        "physical_fe": record,
    }


def read_family(directory: Path, expected: dict[str, object]) -> dict[str, object]:
    result = read_model_artifact(
        directory / "family-result.phx",
        expected_type=CellMeshingResult,
        restore=restore_model_archive_value,
        limits=None,
    )
    manifest = read_model_artifact(
        directory / "family-state.phx",
        expected_type=FiniteElementRestartManifest,
        restore=FiniteElementRestartManifest.from_stored_fields,
        limits=None,
    )
    state = manifest.state
    if (
        result.result_id != expected["result_id"]
        or result.mesh.topology_id != expected["topology_id"]
        or state.accepted_id != expected["accepted_id"]
        or state.topology_id != result.mesh.topology_id
        or state.prepared_id != result.result_id
    ):
        raise ValueError("Cold family accepted state lost its result/topology binding.")
    if result.certification is None:
        raise ValueError("Cold family result omitted its certification.")
    source = result.certification.inputs
    if (source.source_id, source.source_revision) != (
        expected["source_id"],
        expected["source_revision"],
    ):
        raise ValueError("Cold family certification lost original source identity.")
    event = result.surface_quad_field_event
    restored_event = (
        None
        if event is None
        else {
            "event_id": event.event_id,
            "field_id": event.field.field_id,
            "layout_id": event.layout.layout_id,
        }
    )
    if restored_event != expected["surface_quad_field_event"]:
        raise ValueError("Cold family result changed its source-bound quad field event.")
    record, values, field = _physical_state(result, str(expected["profile_id"]))
    stored = np.asarray(state.fields[0], dtype=np.float64)
    if stored.shape != values.shape or not np.allclose(
        stored, values, rtol=0.0, atol=2e-9
    ):
        raise ValueError(
            "Cold family physical FE field differs from independent reanalysis."
        )
    expected_layout = FiniteElementAcceptedFieldLayout(
        field,
        transfer_semantics=FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION,
    )
    if state.field_layouts[0].layout_id != expected_layout.layout_id:
        raise ValueError("Cold family field layout changed.")

    def solve_next(
        accepted: FiniteElementAcceptedState,
        start_time: float,
        end_time: float,
        time_law: phx.solver.TimeLaw,
        args: object,
    ) -> FiniteElementAttemptResult:
        if (
            accepted.accepted_id != state.accepted_id
            or float(np.asarray(time_law.value(end_time, args))) != 1.0
            or not np.allclose(
                np.asarray(accepted.fields[0]), values, rtol=0.0, atol=2e-9
            )
        ):
            raise ValueError(
                "FE continuation did not receive the restored accepted field."
            )
        _, next_values, _ = _physical_state(result, str(expected["profile_id"]))
        return FiniteElementAttemptResult(
            (next_values,), True, materials=accepted.materials
        )

    schedule = FiniteElementAcceptedStepSchedule(
        solve_next,
        schedule_id="qualification:family-steady-next-solve",
    )
    next_state, diagnostics = schedule.advance(
        state,
        np.asarray(1.0, dtype=np.float64),
        phx.solver.TimeLaw.constant(1.0, law_id="qualification:family-steady-unit"),
    )
    if (
        not bool(np.asarray(diagnostics.accepted))
        or next_state.step != state.step + 1
        or next_state.state_version != state.state_version + 1
        or next_state.accepted_id == state.accepted_id
        or float(np.asarray(next_state.time)) != 1.0
    ):
        raise ValueError(
            "Cold family FE continuation did not advance its accepted state."
        )
    return {
        "pid": os.getpid(),
        "result_id": result.result_id,
        "topology_id": result.mesh.topology_id,
        "accepted_id": state.accepted_id,
        "next_accepted_id": next_state.accepted_id,
        "surface_quad_field_event": restored_event,
        "physical_fe": record,
    }


def native_family_cold(
    profile_id: str, *, captured_result: Path | None = None
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="phydrax-family-cold-") as directory:
        command = (sys.executable, "-m", __name__)
        capture_args = () if captured_result is None else (str(captured_result),)
        writer = subprocess.run(
            (*command, "write", directory, profile_id, *capture_args),
            capture_output=True,
            text=True,
        )
        if writer.returncode:
            raise RuntimeError(
                f"Family cold writer failed:\n{writer.stdout}\n{writer.stderr}"
            )
        produced = json.loads(writer.stdout)
        reader = subprocess.run(
            (*command, "read", directory, json.dumps(produced)),
            capture_output=True,
            text=True,
        )
        if reader.returncode:
            raise RuntimeError(
                f"Family cold reader failed:\n{reader.stdout}\n{reader.stderr}"
            )
        restored = json.loads(reader.stdout)
    if produced["pid"] == restored["pid"]:
        raise RuntimeError(
            "Family cold producer and consumer must be separate processes."
        )
    return {"profile_id": profile_id, "producer": produced, "consumer": restored}


if __name__ == "__main__":
    directory = Path(sys.argv[2])
    if sys.argv[1] == "write":
        captured = Path(sys.argv[4]) if len(sys.argv) > 4 else None
        print(
            json.dumps(
                write_family(directory, sys.argv[3], captured_result=captured),
                sort_keys=True,
            )
        )
    elif sys.argv[1] == "read":
        print(json.dumps(read_family(directory, json.loads(sys.argv[3])), sort_keys=True))
    else:
        raise ValueError("Expected write or read.")
