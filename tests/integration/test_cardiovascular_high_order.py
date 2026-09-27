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
) -> None:
    case = qualify_case(cell_kind)

    assert case["passed"]
    assert case["geometry_family"] == family
    assert case["node_count"] == node_count
    # ty: ignore[not-subscriptable]
    assert case["geometry"]["passed"]
    # ty: ignore[not-subscriptable]
    assert min(case["geometry"]["minimum_jacobian_determinants"]) > 0.0
    # ty: ignore[not-subscriptable]
    assert min(case["geometry"]["minimum_cell_measures_mm3"]) > 0.0
    # ty: ignore[not-subscriptable]
    assert case["geometry"]["stale_epoch_transfer_required"]
    # ty: ignore[not-subscriptable]
    assert case["geometry"]["stale_epoch_rebuild_required"]
    # ty: ignore[not-subscriptable]
    assert case["geometry"]["stale_epoch_rejected"]

    # ty: ignore[not-subscriptable]
    assert case["operator"]["passed"]
    assert (
        # ty: ignore[not-subscriptable]
        case["operator"]["source_discretization_id"]
        # ty: ignore[not-subscriptable]
        != case["operator"]["target_discretization_id"]
    )
    assert (
        # ty: ignore[not-subscriptable]
        case["operator"]["source_operator_id"] != case["operator"]["target_operator_id"]
    )
    # ty: ignore[not-subscriptable]
    assert case["operator"]["maximum_rebuild_action_change"] > 0.0
    # ty: ignore[not-subscriptable]
    assert case["operator"]["minimum_conductivity_eigenvalue"] > 0.0

    # ty: ignore[not-subscriptable]
    assert case["transfer"]["passed"]
    # ty: ignore[not-subscriptable]
    assert case["transfer"]["transfer_identity_rebuilt"]
    # ty: ignore[not-subscriptable]
    assert case["transfer"]["source_coverage_fraction"] == 1.0
    # ty: ignore[not-subscriptable]
    assert case["transfer"]["target_coverage_fraction"] == 1.0
    # ty: ignore[not-subscriptable]
    assert case["transfer"]["coverage_complete"]
    # ty: ignore[not-subscriptable]
    assert case["transfer"]["constant_preserved"]
    # ty: ignore[not-subscriptable]
    assert case["transfer"]["adjoint_consistent"]
    # ty: ignore[not-subscriptable]
    assert case["transfer"]["all_lane_evidence_accepted"]
    # ty: ignore[not-subscriptable]
    assert case["transfer"]["voltage_error"] <= 2.0e-6
    # ty: ignore[not-subscriptable]
    assert case["transfer"]["maximum_reaction_lane_error"] <= 2.0e-6
    assert (
        # ty: ignore[not-subscriptable]
        case["transfer"]["reaction_lane_count"]
        # ty: ignore[not-subscriptable]
        == case["transfer"]["expected_reaction_lane_count"]
    )

    # ty: ignore[not-subscriptable]
    assert case["electrophysiology"]["passed"]
    # ty: ignore[not-subscriptable]
    assert case["electrophysiology"]["source_step_accepted"]
    # ty: ignore[not-subscriptable]
    assert case["electrophysiology"]["target_step_accepted"]
    # ty: ignore[not-subscriptable]
    assert case["electrophysiology"]["reaction_admissible"]
    # ty: ignore[not-subscriptable]
    assert case["electrophysiology"]["diffusion_stage_count"] == 1
    # ty: ignore[not-subscriptable]
    assert case["electrophysiology"]["reaction_tick_count"] == 2
    # ty: ignore[not-subscriptable]
    assert case["electrophysiology"]["maximum_voltage_change_mV"] > 0.0


def test_curved_q2_geometry_drives_exact_mixed_cardiac_mechanics() -> None:
    mechanics = qualify_case("hexahedron")["mechanics"]

    assert mechanics is not None
    # ty: ignore[not-subscriptable]
    assert mechanics["passed"]
    # ty: ignore[not-subscriptable]
    assert mechanics["route"] == "exact-q2-q1"
    # ty: ignore[not-subscriptable]
    assert mechanics["coordinate_values_match"]
    # ty: ignore[not-subscriptable]
    assert mechanics["coordinate_identity_matches"]
    # ty: ignore[not-subscriptable]
    assert mechanics["displacement_degree"] == 2
    # ty: ignore[not-subscriptable]
    assert mechanics["pressure_degree"] == 1
    # ty: ignore[not-subscriptable]
    assert mechanics["pair_names"] == ["q2-q1"]
    # ty: ignore[not-subscriptable]
    assert mechanics["gauge_mode"] == "mean-zero"
    # ty: ignore[not-subscriptable]
    assert mechanics["gauge_valid"]
    # ty: ignore[not-subscriptable]
    assert mechanics["residual_finite"]
    # ty: ignore[not-subscriptable]
    assert mechanics["assembled_inf_sup_constant"] > 0.0
    # ty: ignore[not-subscriptable]
    assert mechanics["assembled_inf_sup_stable"]
    # ty: ignore[not-subscriptable]
    assert mechanics["locking_safe"]
    # ty: ignore[not-subscriptable]
    assert mechanics["evaluation_valid"]
    # ty: ignore[not-subscriptable]
    assert mechanics["material_evaluation_valid"]


def test_qualification_claims_only_supported_high_order_cell_kinds() -> None:
    payload = qualification()

    assert payload["passed"]
    assert payload["qualified_cell_kinds"] == ["tetrahedron", "hexahedron"]
    assert payload["mechanics_case_count"] == 1
