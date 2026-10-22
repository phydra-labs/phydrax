# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import (
    _runtime_identity,
    epoch_stage_name,
    peak_resident_bytes,
    PEAK_RESIDENT_SCOPE,
)
from tools.neurofluid_qualification import (
    _size_request_record,
    NATIVE_IMAGE_SIZE_COMPLIANCE,
    native_transport,
    NATIVE_TRANSPORT_CASE_ROLES,
    native_transport_field_templates,
    TIME_CONVERGENCE_MINIMUM_ORDER,
    TIME_CONVERGENCE_REFINEMENTS,
)


def _time_step_convergence(convergence: dict[str, Any]) -> bool:
    """Backward Euler must approach the exact affine solution at first order."""

    widths = convergence["step_widths_seconds"]
    errors = convergence["relative_errors"]
    if len(errors) != len(widths) or len(errors) != 1 + len(TIME_CONVERGENCE_REFINEMENTS):
        return False
    if (
        not all(np.isfinite(errors))
        or min(errors) <= 0.0
        or not all(np.isfinite(widths))
        or min(widths) <= 0.0
        or convergence["required_minimum_order"] != TIME_CONVERGENCE_MINIMUM_ORDER
        or widths
        != [widths[0], *(widths[0] / factor for factor in TIME_CONVERGENCE_REFINEMENTS)]
    ):
        return False
    orders = [
        np.log(coarse_error / fine_error) / np.log(coarse / fine)
        for coarse_error, fine_error, coarse, fine in zip(
            errors, errors[1:], widths, widths[1:], strict=False
        )
    ]
    return bool(
        convergence["all_steps_accepted"]
        and all(fine < coarse for coarse, fine in zip(widths, widths[1:]))
        and all(order >= TIME_CONVERGENCE_MINIMUM_ORDER for order in orders)
    )


def _image_obligations(
    args: QualificationControls, record: Mapping[str, Any]
) -> dict[str, bool]:
    """Recompute every declared-role obligation from the transport evidence."""

    invariants = record["invariants"]
    lifecycle = record["lifecycle"]
    cold = lifecycle["cold"]
    tolerance = args.balance_tolerance
    geometry = invariants["geometry_tolerance"]
    sites = invariants["exchange_support"]["sites"]
    obligations = {
        "original_scenario_controls": record["case"] == args.image_case
        and record["qualification_role"] == NATIVE_TRANSPORT_CASE_ROLES[args.image_case]
        and record["controls"]["size"] == args.image_size
        and record["controls"]["steps"] == args.transport_steps
        and record["controls"]["dt_seconds"] == args.transport_dt
        and record["controls"]["repeats"] == args.repeats
        and record["controls"]["maximum_cells"] == args.capacity
        and record["controls"]["maximum_vertices"] == args.capacity
        and record["controls"]["oracle_capacity"] == args.oracle_capacity,
        "approved_shared_size_request": record["size_request"]
        == _size_request_record(NATIVE_IMAGE_SIZE_COMPLIANCE),
        "accepted_allfield_history": lifecycle["passed"] is True
        and tuple(lifecycle["field_names"])
        == tuple(template["name"] for template in native_transport_field_templates())
        and lifecycle["accepted_history_steps"] == args.transport_steps
        and len(invariants["accepted_steps"]) == args.transport_steps
        and all(value is True for value in invariants["accepted_steps"])
        and lifecycle["atomic_rollback"]["passed"] is True
        and lifecycle["atomic_rollback"]["positive_source_physics_controls_unchanged"]
        is True,
        "declared_image_interpretation": record["source"]["interpretation"]
        == "occupied-voxel-cells",
        "shared_junction_complex": tuple(map(tuple, invariants["adjacency_pairs"]))
        == tuple(map(tuple, invariants["voxel_adjacency_pairs"]))
        and invariants["junction_length_error"]
        <= geometry * max(1.0, invariants["expected_junction_edge_length"])
        and invariants["quadruple_line_length_error"]
        <= geometry * max(1.0, invariants["expected_quadruple_line_length"])
        and invariants["quadruple_points_recovered"] == invariants["quadruple_points"]
        and invariants["quintuple_points_recovered"] == invariants["quintuple_points"],
        "conforming_material_tetrahedra": invariants["volume_error"] <= geometry
        and invariants["interface_area_error"] <= geometry
        and invariants["material_containment_violations"] == 0
        and set(invariants["interface_facets_by_pair"])
        == set(invariants["expected_interface_areas"]),
        "declared_physics_and_boundary_data": invariants["declared_parameter_error"]
        <= geometry
        and invariants["boundary_flux_consistency_error"] <= geometry
        and invariants["volume_flux_divergence_error"] <= geometry
        and invariants["initial_state_error"] == 0.0,
        "interface_flux_inventory_balance": invariants[
            "interface_inventory_balance_error"
        ]
        <= tolerance
        and invariants["inventory_error"] <= tolerance
        and invariants["local_balance_error"] <= tolerance,
        "bulk_network_exchange_transfer": invariants["transfer_error"] <= tolerance
        and invariants["exchange_support"]["unlocated_points"] == 0
        and invariants["exchange_support"]["stencil_error"]
        <= invariants["exchange_support"]["stencil_tolerance"]
        and len(sites) > 0
        and all(
            site["single_material"]
            and site["within_radius"]
            and site["support_cells"] > 0
            and site["minimum_weight"] >= 0.0
            and abs(site["weight_sum"] - 1.0) <= tolerance
            for site in sites
        ),
        "physical_error_target": invariants["physical_error"] <= args.target_error,
        "time_step_convergence": _time_step_convergence(record["time_convergence"])
        and record["time_convergence"]["step_widths_seconds"][0] == args.transport_dt
        and record["time_convergence"]["horizon_seconds"]
        == args.transport_steps * args.transport_dt,
        "fresh_process_continuation": lifecycle["AD"]["relative_discrete_physical_error"]
        <= args.target_error
        and lifecycle["AD"]["inventory_defect"] <= tolerance
        and lifecycle["AD"]["primal_defect"] <= tolerance
        and lifecycle["hot_continuation_relative_physical_error"] <= args.target_error
        and lifecycle["hot_continuation_inventory_defect"] <= tolerance
        and lifecycle["hot_continuation_local_balance_error"] <= tolerance
        and cold["passed"] is True
        and cold["allfield_history_bitwise_preserved"] is True
        and cold["prepared_runtime_serialized"] is False
        and cold["source_id"] == record["source"]["source_id"]
        and cold["process"]["status"] == "completed"
        and cold["process_id"] == cold["process"]["process_id"]
        and cold["process_id"] != cold["process"]["parent_process_id"]
        and cold["content_id"] == cold["process"]["content_id"]
        and cold["parameter_id"] == lifecycle["material_parameter_id"]
        and cold["network_id"] == record["identities"]["network_id"]
        and cold["continued_max_error"] <= tolerance
        and cold["continued_local_balance_error"] <= tolerance
        and cold["continued_step_cursor"]
        == cold["accepted_step_cursor"] + lifecycle["accepted_history_steps"]
        and cold["result_id"] == record["identities"]["result_id"]
        and cold["accepted_step_cursor"] == lifecycle["accepted_history_steps"]
        and cold["accepted_time"]
        == lifecycle["accepted_history_steps"] * record["controls"]["dt_seconds"]
        and cold["continued_time"]
        == cold["accepted_time"]
        + lifecycle["accepted_history_steps"] * record["controls"]["dt_seconds"]
        and cold["source_revision"] == record["source"]["source_revision"],
        "within_frozen_wall_budget": record["timing"]["end_to_end_seconds"]
        <= args.timeout,
    }
    if record.get("qualification_role") == "combined":
        affine = np.asarray(invariants["affine_matrix"], dtype=np.float64)[:3, :3]
        lengths = np.linalg.norm(affine, axis=0)
        counts = invariants["voxel_counts"]
        compartments = invariants["compartment_ids"]
        volumes = dict(zip(compartments, invariants["compartment_volumes"], strict=True))
        physics = record["transport_case"]["compartment_physics"]
        obligations.update(
            {
                "authored_affine_geometry_preserved": bool(
                    record["source"].get("outer_geometry_id")
                )
                and cold.get("outer_geometry_id")
                == record["source"].get("outer_geometry_id"),
                "oblique_anisotropic_affine": bool(
                    np.any(affine - np.diag(np.diag(affine)) != 0.0)
                    and np.ptp(lengths) > 0.0
                ),
                # Recovered on the generated mesh, not merely present in the image.
                "triple_and_quadruple_junctions": invariants["junction_edge_length"]
                > invariants["quadruple_line_length"]
                > 0.0
                and invariants["quadruple_points_recovered"] >= 1
                and invariants["quintuple_points_recovered"] >= 1,
                "tiny_required_material": any(
                    count == 1 and volumes[name] > geometry
                    for name, count in counts.items()
                ),
                "heterogeneous_materials": len(
                    {
                        (physics[name]["porosity"], physics[name]["diffusivity"])
                        for name in compartments
                    }
                )
                > 1
                and invariants["local_balance_error"] <= tolerance,
                # Inventory moved by each external term over the solved
                # trajectory must exceed what the closure can absorb, or
                # omitting the term would go unnoticed.
                "source_reaction_transport_closure": invariants[
                    "integrated_boundary_inflow"
                ]
                > tolerance
                and invariants["integrated_boundary_outflow"] > tolerance
                and invariants["integrated_removal"] > tolerance
                and invariants["inventory_error"] <= tolerance,
            }
        )
    return obligations


def qualify_native_image(args: QualificationControls) -> dict[str, object]:
    """Consume the generated-region Neurofluid transport scenario."""

    record = native_transport(
        case=args.image_case,
        size=args.image_size,
        repeats=args.repeats,
        steps=args.transport_steps,
        dt=args.transport_dt,
        target_error=args.target_error,
        balance_tolerance=args.balance_tolerance,
        maximum_cells=args.capacity,
        maximum_vertices=args.capacity,
        maximum_work_units=args.maximum_work_units,
        maximum_geometry_queries=args.maximum_geometry_queries,
        oracle_capacity=args.oracle_capacity,
        timeout=args.timeout,
    )
    successful = record["successful"]
    result: dict[str, object] = {
        **record,
        "scenario": "native-image-lifecycle",
        "image_case": args.image_case,
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
        timing = record["timing"]
        memory = record["memory"]
        invariants = record["invariants"]
        obligations = _image_obligations(args, record)
        result["qualification_obligations"] = obligations
        # Every declared obligation of this profile's role holds; the
        # campaign separately requires each registered role of the family.
        complete = all(obligations.values())
        result["status"] = "passed" if complete else "failed"
        completion = "complete" if complete else "incomplete"
        result["mandatory_workflow_completion"] = completion
        result["case_contract_completion"] = completion
        result["independent_scientific_certification"] = (
            "passed"
            if all(
                passed
                for name, passed in obligations.items()
                if name != "within_frozen_wall_budget"
            )
            else "failed"
        )
        if not complete:
            result["failure"] = {
                "phase": "qualification_obligations",
                "exception_type": "QualificationObligationUnmet",
                "message": "Declared image obligations unmet: "
                f"{sorted(name for name, passed in obligations.items() if not passed)!r}.",
            }
        compiler = memory["compiler"]
        warmed = timing["warmed"]["samples_seconds"]
        result.update(
            {
                "request_id": identities["request_id"],
                "physical_error": invariants["physical_error"],
                "end_to_end_seconds": timing["end_to_end_seconds"],
                # The harness evaluates its target once, inside certification, and
                # records no first-reach instant; the lifecycle time is not relabelled.
                "time_to_target_error_seconds": None,
                "time_to_target_measurement": "unmeasured:no-first-reach-instant",
                "peak_process_resident_bytes": peak_resident_bytes(),
                "process_peak_scope": PEAK_RESIDENT_SCOPE,
                "generation": {
                    "result_id": identities["result_id"],
                    "mesh_id": identities["mesh_id"],
                    "resource_guarantees": record["publication_resource_guarantees"],
                    "resource_binding": record["publication_resource_binding"],
                    "execution_evidence": record["native_execution"],
                },
                # One accepted transport epoch on the generated publication.
                "epochs": [
                    {
                        "epoch": 0,
                        "compiler": {
                            "temporary_bytes": compiler["temporary_bytes"],
                            "output_bytes": compiler["output_bytes"],
                            "argument_bytes": compiler["argument_bytes"],
                            "generated_code_bytes": compiler["generated_code_bytes"],
                            "memory_status": "unavailable"
                            if compiler["unavailable_reason"] is not None
                            else "measured",
                            "warm_samples_seconds": warmed,
                            "execution_scope": "primary-transport-executable",
                        },
                        "retained_bytes": memory["logical_retained_bytes"],
                    }
                ],
                "stages_seconds": {
                    epoch_stage_name("solver_lowering", 0): timing["lowering_seconds"],
                    epoch_stage_name("solver_compile", 0): timing["compilation_seconds"],
                    epoch_stage_name("solver_first_execution", 0): timing[
                        "first_execution_seconds"
                    ],
                },
            }
        )
    return result


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_native_image(args)
