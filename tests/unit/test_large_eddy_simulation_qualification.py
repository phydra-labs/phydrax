#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

import phydrax as phx
import tools.large_eddy_simulation_qualification as les_qualification
from tools.large_eddy_simulation_qualification import (
    _ksgs_coefficients,
    _run_channel_restriction,
    _run_channel_wall_owner,
    _run_dynamic_ksgs,
    _run_frozen_imex,
    _run_frozen_sbdf2,
    _run_immersed,
    _run_immersed_sbdf2,
    _run_learned_stress,
    _run_unstructured,
    _run_unstructured_pressure,
    _tree_max_error,
    admit_reference,
    canonical_json,
    content_address,
    execute_campaign,
    load_campaign,
    load_matrix,
    validate_campaign,
)


_ROOT = Path(__file__).resolve().parents[2]
_CAMPAIGN = _ROOT / "benchmarks" / "large_eddy_simulation_qualification_campaign.json"
_MATRIX = _ROOT / "benchmarks" / "large_eddy_simulation_qualification_matrix.json"


def _loaded():
    matrix = load_matrix(_MATRIX)
    return load_campaign(_CAMPAIGN, matrix), matrix


def _readdress_campaign(campaign):
    campaign["campaign_id"] = content_address(
        {name: value for name, value in campaign.items() if name != "campaign_id"}
    )


def test_campaign_and_matrix_ids_are_deterministic_content_addresses():
    campaign, matrix = _loaded()
    second_campaign, second_matrix = _loaded()
    reordered = dict(reversed(tuple(campaign.items())))

    assert campaign == second_campaign
    assert matrix.matrix_id == second_matrix.matrix_id
    assert campaign["campaign_id"] == content_address(
        {name: value for name, value in campaign.items() if name != "campaign_id"}
    )
    assert content_address(reordered) == content_address(campaign)
    assert "schema_version" not in canonical_json(campaign)
    assert "schema_version" not in canonical_json(matrix.to_record())


def test_every_route_and_static_formula_has_an_exact_separate_support_tuple():
    campaign, _ = _loaded()
    cases = tuple(campaign["cases"])
    supports = tuple(
        phx.qualification.SupportTuple.from_record(case["support"]) for case in cases
    )
    static = {
        dict(value.attributes)["closure"]: value
        for case, value in zip(cases, supports, strict=True)
        if case["producer"] == "periodic-static"
    }

    assert len({value.support_tuple_id for value in supports}) == len(supports)
    assert set(static) == {"smagorinsky", "amd", "vreman", "wale"}
    assert len({value.support_tuple_id for value in static.values()}) == 4
    assert all(value.capability == "large-eddy-simulation" for value in supports)
    assert {case["producer"] for case in cases} == {
        "channel",
        "channel-restriction",
        "channel-wall-owner",
        "distributed",
        "distributed-production",
        "dynamic-ksgs",
        "favre",
        "favre-dg-energy",
        "frozen-imex",
        "frozen-sbdf2",
        "immersed",
        "immersed-sbdf2",
        "lbm",
        "learned-stress",
        "low-re-ksgs",
        "mac-coupled",
        "mac-ksgs",
        "periodic-dynamic",
        "periodic-dynamic-production",
        "periodic-exact-filter",
        "periodic-static",
        "stochastic-mac-inflow",
        "unstructured",
        "unstructured-pressure",
    }
    immersed = next(case for case in cases if case["producer"] == "immersed")
    assert immersed["support"]["attributes"]["temporal_method"] == ("immersed-imex-euler")
    assert immersed["parameters"]["sbdf2_evidence"] == "not-claimed"
    final_routes = {case["name"]: case["support"]["support_tuple_id"] for case in cases}
    assert {
        "periodic-dynamic-production",
        "mac-dynamic-ksgs",
        "ocean-low-re-ksgs",
        "distributed-full-flow-production",
        "learned-stress-periodic",
        "learned-stress-mac",
        "channel-mixed-wall-stress",
        "channel-complete-restriction",
        "stochastic-mac-inflow-owner",
        "favre-transported-sgs-dg",
        "unstructured-pressure-continuation",
        "immersed-mac-sbdf2-restart",
    } <= set(final_routes)
    assert len(set(final_routes.values())) == len(final_routes)


def test_restart_tree_comparison_handles_bool_integer_and_numeric_leaves():
    state = {
        "accepted": jnp.asarray((True, False)),
        "steps": jnp.asarray(2, dtype=jnp.int32),
        "values": jnp.asarray((1.0, -2.0)),
    }

    assert _tree_max_error(state, state) == 0.0
    assert (
        _tree_max_error(state, {**state, "accepted": jnp.asarray((False, False))}) == 1.0
    )
    assert (
        _tree_max_error(state, {**state, "steps": jnp.asarray(3, dtype=jnp.int32)}) == 1.0
    )
    assert _tree_max_error(
        state, {**state, "values": jnp.asarray((1.25, -2.0))}
    ) == pytest.approx(0.25)


def test_ksgs_campaign_fields_bind_the_exact_coefficient_constructor():
    campaign, _ = _loaded()
    cases = {
        case["name"]: case
        for case in campaign["cases"]
        if case["name"]
        in (
            "mac-prognostic-ksgs",
            "mac-dynamic-ksgs",
            "ocean-low-re-ksgs",
            "unstructured-pressure-continuation",
        )
    }
    expected_fields = {
        "eddy_viscosity",
        "dissipation",
        "diffusion",
        "buoyancy",
        "production_limit",
    }
    for case in cases.values():
        supplied = case["coefficients"]
        assert "production" not in supplied
        assert expected_fields <= set(supplied)
        coefficients = _ksgs_coefficients(supplied)
        assert coefficients.eddy_viscosity == supplied["eddy_viscosity"]
        assert coefficients.dissipation == supplied["dissipation"]
        assert coefficients.diffusion == supplied["diffusion"]
        assert coefficients.buoyancy == supplied["buoyancy"]
        assert coefficients.production_limit == supplied["production_limit"]
    assert (
        cases["mac-prognostic-ksgs"]["coefficients"]["boussinesq_expansion"]
        != cases["mac-prognostic-ksgs"]["coefficients"]["buoyancy"]
    )


def test_periodic_guard_and_unstructured_energy_gates_are_preregistered():
    campaign, _ = _loaded()
    static_cases = [
        case for case in campaign["cases"] if case["producer"] == "periodic-static"
    ]
    for case in static_cases:
        assert case["parameters"]["guard_safety_factor"] == 0.5
        assert {
            "guard_acceptance_failure",
            "guard_rejection_failure",
            "guard_rollback_error",
            "guard_coordinate_binding_failure",
        } <= {metric["name"] for metric in case["metrics"]}
    unstructured = next(
        case
        for case in campaign["cases"]
        if case["name"] == "unstructured-pressure-continuation"
    )
    assert unstructured["support"]["attributes"]["closure"] == (
        "static-ksgs-viscosity-owner"
    )
    assert unstructured["support"]["attributes"]["production_policy"] == (
        "conservative-face-work-equal-cell-split"
    )
    assert (
        unstructured["support"]["attributes"]["production_limit_disposition"]
        == "modeled-enthalpy-density-source"
    )
    assert {
        "normalized_sgs_energy_balance",
        "normalized_modeled_transfer_residual",
        "energy_balanced_failure",
        "energy_status_failure",
        "viscosity_owner_identity_error",
        "ksgs_state_failure",
        "executed_rhs_rebuild_error",
        "production_evidence_rebuild_error",
        "negative_work_refusal_failure",
        "ksgs_production_policy_failure",
        "deviatoric_face_stress_magnitude",
        "minimum_ksgs_raw_production_density",
        "negative_local_production_violation",
        "production_limit_reduction_magnitude",
        "modeled_enthalpy_source_magnitude",
        "enthalpy_source_identity_error",
        "thermalization_rate_magnitude",
        "enthalpy_thermalization_balance_error",
        "modeled_energy_split_residual",
        "total_energy_balance_failure",
    } <= {metric["name"] for metric in unstructured["metrics"]}


def test_unstructured_pressure_measures_energy_and_viscosity_owner():
    campaign, _ = _loaded()
    case = next(
        value
        for value in campaign["cases"]
        if value["name"] == "unstructured-pressure-continuation"
    )

    measurements, _ = _run_unstructured_pressure(case, None)

    assert measurements["normalized_sgs_energy_balance"] <= 3.0e-6
    assert measurements["normalized_modeled_transfer_residual"] <= 3.0e-6
    assert measurements["executed_rhs_rebuild_error"] <= 2.0e-12
    assert measurements["production_evidence_rebuild_error"] <= 2.0e-12
    assert measurements["ksgs_production_policy_failure"] == 0.0
    assert measurements["deviatoric_face_stress_magnitude"] >= 1.0e-12
    assert measurements["minimum_ksgs_raw_production_density"] >= 1.0e-12
    assert measurements["negative_local_production_violation"] == 0.0
    assert measurements["negative_work_refusal_failure"] == 0.0
    assert measurements["production_limit_reduction_magnitude"] >= 1.0e-12
    assert measurements["modeled_enthalpy_source_magnitude"] >= 1.0e-12
    assert measurements["enthalpy_source_identity_error"] == 0.0
    assert measurements["thermalization_rate_magnitude"] >= 1.0e-12
    assert measurements["enthalpy_thermalization_balance_error"] <= 2.0e-8
    assert measurements["modeled_energy_split_residual"] <= 3.0e-6
    assert measurements["total_energy_balance_failure"] == 0.0
    assert measurements["energy_balanced_failure"] == 0.0
    assert measurements["energy_status_failure"] == 0.0
    assert measurements["viscosity_owner_identity_error"] == 0.0
    assert measurements["ksgs_state_failure"] == 0.0
    assert measurements["continuation_failure"] == 0.0


@pytest.mark.parametrize(
    ("outcome", "expected_code"),
    (("passed", 0), ("failed", 1), ("inconclusive", 2)),
)
def test_cli_exit_code_tracks_coverage_after_artifact_emission(
    outcome, expected_code, monkeypatch, tmp_path, capsys
):
    matrix = object()
    campaign = {"campaign_id": "mock-campaign"}
    candidate = {
        "status": "unreleased-candidate",
        "qualification_outcome": outcome,
        "released": False,
        "signed": False,
    }
    output = tmp_path / outcome

    monkeypatch.setattr(les_qualification, "load_matrix", lambda _path: matrix)
    monkeypatch.setattr(
        les_qualification,
        "load_campaign",
        lambda _path, supplied: campaign if supplied is matrix else None,
    )

    def execute(supplied_campaign, supplied_matrix, destination):
        assert supplied_campaign is campaign
        assert supplied_matrix is matrix
        destination.mkdir()
        (destination / "candidate.json").write_text(json.dumps(candidate) + "\\n")
        return candidate

    monkeypatch.setattr(les_qualification, "execute_campaign", execute)
    code = les_qualification.main(
        [
            "--campaign",
            str(tmp_path / "campaign.json"),
            "--matrix",
            str(tmp_path / "matrix.json"),
            "--output",
            str(output),
        ]
    )

    assert code == expected_code
    assert (output / "candidate.json").is_file()
    assert json.loads(capsys.readouterr().out) == candidate


def test_active_dynamic_and_learned_backends_measure_nonzero_actions():
    campaign, _ = _loaded()
    cases = {case["name"]: case for case in campaign["cases"]}

    dynamic, _ = _run_dynamic_ksgs(cases["mac-dynamic-ksgs"], None)
    assert dynamic["dynamic_coefficient_change_magnitude"] >= 1.0e-12
    assert dynamic["mac_sgs_action_magnitude"] >= 1.0e-12

    for name in ("learned-stress-periodic", "learned-stress-mac"):
        measurements, _ = _run_learned_stress(cases[name], None)
        assert measurements["learned_stress_magnitude"] >= 1.0e-12
        assert measurements["projected_rate_magnitude"] >= 1.0e-12
        assert measurements["energy_policy_failure"] == 0.0


def test_channel_and_frozen_routes_use_deliberately_active_states():
    campaign, _ = _loaded()
    cases = {case["name"]: case for case in campaign["cases"]}

    wall, _ = _run_channel_wall_owner(cases["channel-mixed-wall-stress"], None)
    restriction, _ = _run_channel_restriction(cases["channel-complete-restriction"], None)
    assert wall["channel_les_viscosity_magnitude"] >= 1.0e-12
    assert restriction["channel_les_viscosity_magnitude"] >= 1.0e-12

    imex, _ = _run_frozen_imex(cases["mac-frozen-imex"], None)
    sbdf2, _ = _run_frozen_sbdf2(cases["mac-frozen-sbdf2"], None)
    assert imex["frozen_les_action_magnitude"] >= 1.0e-12
    assert sbdf2["frozen_les_action_magnitude"] >= 1.0e-12


def test_every_nonzero_active_route_preregisters_an_activity_predicate():
    campaign, _ = _loaded()
    by_name = {case["name"]: case for case in campaign["cases"]}
    activity = {
        "periodic-smagorinsky": "backend_activity_violation",
        "periodic-amd": "backend_activity_violation",
        "periodic-vreman": "backend_activity_violation",
        "periodic-wale": "backend_activity_violation",
        "periodic-dynamic-smagorinsky": "synthetic_stress_magnitude",
        "periodic-dynamic-production": "dynamic_sgs_transfer_magnitude",
        "mac-momentum-scalar-boussinesq": "momentum_sgs_action_magnitude",
        "mac-prognostic-ksgs": "ksgs_eddy_viscosity_magnitude",
        "mac-dynamic-ksgs": "mac_sgs_action_magnitude",
        "ocean-low-re-ksgs": "low_re_dissipation_missing",
        "mac-frozen-imex": "frozen_les_action_magnitude",
        "mac-frozen-sbdf2": "frozen_les_action_magnitude",
        "spectral-channel-wale": "subgrid_transfer_magnitude",
        "channel-mixed-wall-stress": "channel_les_viscosity_magnitude",
        "channel-complete-restriction": "channel_les_viscosity_magnitude",
        "distributed-periodic-slab": "distributed_sgs_action_magnitude",
        "distributed-full-flow-production": "distributed_sgs_action_magnitude",
        "favre-compressible-smoke": "favre_stress_magnitude",
        "favre-transported-sgs-dg": "negative_sgs_source_missing",
        "unstructured-low-mach-smoke": "sgs_flux_magnitude",
        "unstructured-pressure-continuation": "sgs_flux_magnitude",
        "immersed-mac-wall-stress": "immersed_sgs_action_magnitude",
        "immersed-mac-sbdf2-restart": "sgs_extrapolated_action_magnitude",
        "learned-stress-periodic": "learned_stress_magnitude",
        "learned-stress-mac": "learned_stress_magnitude",
        "lbm-smagorinsky-smoke": "backend_evidence_failure",
    }
    for case_name, metric_name in activity.items():
        assert metric_name in {metric["name"] for metric in by_name[case_name]["metrics"]}


def test_threshold_changes_cannot_reuse_a_preregistered_criterion():
    campaign, matrix = _loaded()
    changed = copy.deepcopy(campaign)
    case = changed["cases"][0]
    case["metrics"][0]["threshold"] *= 2.0
    case["case_id"] = content_address(
        {name: value for name, value in case.items() if name != "case_id"}
    )
    _readdress_campaign(changed)

    with pytest.raises(ValueError, match="criterion_id does not bind its threshold"):
        validate_campaign(changed, matrix)


def test_missing_and_rightless_references_are_refused_before_execution():
    with pytest.raises(ValueError, match="requires a reference manifest"):
        admit_reference(None, required=True)

    campaign, _ = _loaded()
    reference = next(
        case["references"][0]
        for case in campaign["cases"]
        if case["producer"] == "periodic-exact-filter"
    )
    payload = canonical_json(reference["payload"]).encode("utf-8")
    denied = phx.qualification.ReferenceArtifactManifest(
        "rightless-periodic-les-reference",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="test-rights-refusal",
        commercial_use_permitted=False,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="EAR99",
        nondimensionalization={"length": 1.0, "velocity": 1.0},
        uncertainty={"analytic_roundoff": 0.0},
        lineage_ids=("analytic-trigonometric-generator",),
    )
    rightless = {
        **reference,
        "manifest": denied.to_record(),
    }

    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        admit_reference(rightless, required=True)


def test_matrix_refuses_post_hoc_predicates_not_owned_by_a_case():
    campaign, matrix = _loaded()
    predicates = {name: dict(requirements) for name, requirements in matrix.predicates}
    predicates["posthoc.unassigned"] = dict(next(iter(predicates.values())))
    changed_matrix = phx.qualification.QualificationMatrix(predicates)
    changed_campaign = copy.deepcopy(campaign)
    changed_campaign["matrix_id"] = changed_matrix.matrix_id
    _readdress_campaign(changed_campaign)

    with pytest.raises(ValueError, match="post-hoc or unassigned"):
        validate_campaign(changed_campaign, changed_matrix)


def test_unstructured_vector_residuals_are_reduced_to_preregistered_scalars():
    campaign, _ = _loaded()
    case = next(
        value
        for value in campaign["cases"]
        if value["name"] == "unstructured-low-mach-smoke"
    )

    measurements, _ = _run_unstructured(case, None)

    assert isinstance(measurements["momentum_balance_residual"], float)
    assert isinstance(measurements["scalar_balance_residual"], float)
    assert measurements["momentum_balance_residual"] <= 2.0e-7
    assert measurements["scalar_balance_residual"] <= 2.0e-7
    assert measurements["normalized_positive_sgs_work"] <= 3.0e-6
    assert measurements["sgs_work_dissipative_failure"] == 0.0
    assert measurements["positive_work_refusal_failure"] == 0.0
    assert measurements["algebraic_energy_status_failure"] == 0.0


def test_immersed_wall_traction_and_normal_only_trajectory_are_measured():
    campaign, _ = _loaded()
    case = next(value for value in campaign["cases"] if value["producer"] == "immersed")

    measurements, execution = _run_immersed(case, None)

    assert measurements["wall_traction_magnitude"] >= 1.0e-12
    assert measurements["wall_on_off_trajectory_effect"] >= 1.0e-12
    assert measurements["normal_constraint_mode_violation"] == 0.0
    assert measurements["wall_rate_missing"] == 0.0
    assert measurements["execution_failure"] == 0.0
    assert execution["temporal_method"] == ("immersed-imex-euler-normal-constraint")
    assert execution["sbdf2_evidence"] == "not-claimed"


def test_immersed_sbdf2_measures_extrapolated_nonzero_sgs_action():
    campaign, _ = _loaded()
    case = next(
        value for value in campaign["cases"] if value["producer"] == "immersed-sbdf2"
    )

    measurements, execution = _run_immersed_sbdf2(case, None)

    assert measurements["sgs_extrapolated_action_magnitude"] >= 1.0e-12
    assert measurements["sgs_bulk_work_magnitude"] >= 1.0e-14
    assert measurements["sgs_extrapolated_work_error"] <= 2.0e-10
    assert measurements["advanced_impulse_balance_residual"] <= 2.0e-7
    assert measurements["advanced_transfer_work_residual"] <= 2.0e-10
    assert measurements["continuation_failure"] == 0.0
    assert execution["constraint_mode"] == "full-vector"


def test_executed_candidate_remains_unsigned_and_unreleased(tmp_path):
    campaign, matrix = _loaded()
    focused = copy.deepcopy(campaign)
    focused["cases"] = [case for case in focused["cases"] if case["producer"] == "favre"]
    predicate_ids = set(focused["cases"][0]["predicates"])
    focused_matrix = phx.qualification.QualificationMatrix(
        {
            name: dict(requirements)
            for name, requirements in matrix.predicates
            if name in predicate_ids
        }
    )
    focused["matrix_id"] = focused_matrix.matrix_id
    _readdress_campaign(focused)

    candidate = execute_campaign(focused, focused_matrix, tmp_path)
    profile_paths = tuple((tmp_path / "profiles").glob("*.json"))
    raw_paths = tuple((tmp_path / "raw").glob("*.json"))
    assert candidate["status"] == "unreleased-candidate"
    assert candidate["qualification_outcome"] == "passed"
    assert candidate["released"] is False
    assert candidate["signed"] is False
    assert "signature" not in candidate
    assert candidate["unresolved_release_requirements"]
    assert len(raw_paths) == 1
    raw = json.loads(raw_paths[0].read_text())
    assert all(
        set(measurement) == {"criterion_id", "units", "value"}
        for measurement in raw["measurements"].values()
    )
    assert len(profile_paths) == 1
    profile = phx.qualification.CapabilityProfile.from_record(
        json.loads(profile_paths[0].read_text())
    )
    assert profile.released is False
    assert profile.release_evidence == ()
    assert profile.dependencies
