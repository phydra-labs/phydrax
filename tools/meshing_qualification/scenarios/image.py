# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import _runtime_identity
from tools.neurofluid_qualification import native_transport


def qualify_native_image(args: QualificationControls) -> dict[str, object]:
    """Consume the generated-region Neurofluid transport scenario."""

    record = native_transport(
        size=args.image_size,
        repeats=args.repeats,
        steps=args.transport_steps,
        dt=args.transport_dt,
        target_error=args.target_error,
        balance_tolerance=args.balance_tolerance,
        maximum_cells=args.capacity,
        maximum_vertices=args.capacity,
        oracle_capacity=args.oracle_capacity,
    )
    successful = record["successful"]
    result: dict[str, object] = {
        **record,
        "scenario": "native-image-lifecycle",
        "status": "passed" if successful else "failed",
        "runtime": _runtime_identity(),
        "mandatory_workflow_completion": "incomplete",
        "case_contract_completion": "incomplete",
        "independent_scientific_certification": "unassessed",
        "release": "unassessed",
        "leadership": "unassessed",
    }
    if successful:
        identities = record["identities"]
        invariants = record["invariants"]
        timing = record["timing"]
        memory = record["memory"]
        lifecycle = record["lifecycle"]
        cold = lifecycle["cold"]
        elapsed = timing["end_to_end_seconds"]
        complete = (
            lifecycle["passed"] is True
            and cold["passed"] is True
            and elapsed <= args.timeout
        )
        result["status"] = "passed" if complete else "failed"
        completion = "complete" if complete else "incomplete"
        result["mandatory_workflow_completion"] = completion
        result["case_contract_completion"] = completion
        result["independent_scientific_certification"] = (
            "passed"
            if lifecycle["passed"] is True and cold["passed"] is True
            else "failed"
        )
        result.update(
            {
                "request_id": identities["request_id"],
                "physical_error": invariants["physical_error"],
                "end_to_end_seconds": timing["end_to_end_seconds"],
                "time_to_target_error_seconds": timing["end_to_end_seconds"],
                "peak_process_resident_bytes": memory["process_peak_resident_bytes"],
            }
        )
    return result


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_native_image(args)
