#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import pytest

from tools.cardiovascular_high_order_qualification import qualification, qualify_case


@pytest.mark.parametrize(
    ("cell_kind", "family", "node_count"),
    (("tetrahedron", "P2", 10), ("hexahedron", "Q2", 27)),
)
def test_curved_high_order_geometry_drives_transferred_physical_ep(
    cell_kind: str, family: str, node_count: int
):
    case = qualify_case(cell_kind)

    assert case["passed"]
    assert case["geometry_family"] == family
    assert case["node_count"] == node_count
    assert case["geometry"]["passed"]
    assert min(case["geometry"]["minimum_jacobian_determinants"]) > 0.0
    assert min(case["geometry"]["minimum_cell_measures_mm3"]) > 0.0
    assert case["geometry"]["stale_epoch_transfer_required"]
    assert case["geometry"]["stale_epoch_rebuild_required"]
    assert case["geometry"]["stale_epoch_rejected"]

    assert case["operator"]["passed"]
    assert (
        case["operator"]["source_discretization_id"]
        != case["operator"]["target_discretization_id"]
    )
    assert (
        case["operator"]["source_operator_id"] != case["operator"]["target_operator_id"]
    )
    assert case["operator"]["maximum_rebuild_action_change"] > 0.0
    assert case["operator"]["minimum_conductivity_eigenvalue"] > 0.0

    assert case["transfer"]["passed"]
    assert case["transfer"]["transfer_identity_rebuilt"]
    assert case["transfer"]["source_coverage_fraction"] == 1.0
    assert case["transfer"]["target_coverage_fraction"] == 1.0
    assert case["transfer"]["coverage_complete"]
    assert case["transfer"]["constant_preserved"]
    assert case["transfer"]["adjoint_consistent"]
    assert case["transfer"]["all_lane_evidence_accepted"]
    assert case["transfer"]["voltage_error"] <= 2.0e-6
    assert case["transfer"]["maximum_reaction_lane_error"] <= 2.0e-6
    assert (
        case["transfer"]["reaction_lane_count"]
        == case["transfer"]["expected_reaction_lane_count"]
    )

    assert case["electrophysiology"]["passed"]
    assert case["electrophysiology"]["source_step_accepted"]
    assert case["electrophysiology"]["target_step_accepted"]
    assert case["electrophysiology"]["reaction_admissible"]
    assert case["electrophysiology"]["diffusion_stage_count"] == 1
    assert case["electrophysiology"]["reaction_tick_count"] == 2
    assert case["electrophysiology"]["maximum_voltage_change_mV"] > 0.0


def test_curved_q2_geometry_drives_exact_mixed_cardiac_mechanics():
    mechanics = qualify_case("hexahedron")["mechanics"]

    assert mechanics is not None
    assert mechanics["passed"]
    assert mechanics["route"] == "exact-q2-q1"
    assert mechanics["coordinate_values_match"]
    assert mechanics["coordinate_identity_matches"]
    assert mechanics["displacement_degree"] == 2
    assert mechanics["pressure_degree"] == 1
    assert mechanics["pair_names"] == ["q2-q1"]
    assert mechanics["gauge_mode"] == "mean-zero"
    assert mechanics["gauge_valid"]
    assert mechanics["residual_finite"]
    assert mechanics["assembled_inf_sup_constant"] > 0.0
    assert mechanics["assembled_inf_sup_stable"]
    assert mechanics["locking_safe"]
    assert mechanics["evaluation_valid"]
    assert mechanics["material_evaluation_valid"]


def test_qualification_claims_only_supported_high_order_cell_kinds():
    payload = qualification()

    assert payload["passed"]
    assert payload["qualified_cell_kinds"] == ["tetrahedron", "hexahedron"]
    assert payload["mechanics_case_count"] == 1
