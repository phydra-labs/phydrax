#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from dataclasses import fields
from typing import Any

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.spectral import SpectralViscosityKernel
from tools import incompressible_flow_qualification as qualification
from tools.incompressible_flow_qualification import (
    assemble_candidate_profile,
    canonical_json,
    content_address,
    external_reference_input,
    gate_outcome,
    make_qualification_artifact,
    numeric_gate,
    periodic_spectral_vanishing_viscosity_qualification,
    periodic_statistical_solution_qualification,
    REQUIRED_CASES,
    verify_qualification_artifact,
)
from tools.incompressible_spectral_benchmarks import (
    IncompressibleSpectralBenchmarkRecord,
)
from tools.structured_flow_benchmarks import StructuredFlowBenchmarkRecord


def _artifact(
    route: Any = "periodic-spectral",
    *,
    metric_value: Any = 1.0e-12,
    tolerance: Any = 1.0e-9,
) -> Any:
    support = phx.qualification.SupportTuple(
        "incompressible-flow",
        {"route": route, "method": "native-test-route"},
    )
    metrics = {name: {"error": metric_value} for name in REQUIRED_CASES[route]}
    return make_qualification_artifact(
        route=route,
        support_tuple=support,
        inputs={"case": "deterministic-input"},
        reference={"source": "native-analytic"},
        configuration={"error_tolerance": tolerance, "timing_gate": False},
        metrics=metrics,
        gates=(
            numeric_gate(
                "solution",
                f"{REQUIRED_CASES[route][0]}.error",
                metric_value,
                tolerance,
            ),
        ),
    )


def test_incompressible_flow_qualification_scenario_1() -> None:
    left = _artifact()
    reordered = _artifact()
    changed_metric = _artifact(metric_value=2.0e-12)
    changed_configuration = _artifact(tolerance=2.0e-9)

    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
    assert content_address({"a": 1}) != content_address({"a": 2})
    assert left == reordered
    assert left["artifact_id"] == reordered["artifact_id"]
    assert left["artifact_id"] != changed_metric["artifact_id"]
    assert left["artifact_id"] != changed_configuration["artifact_id"]
    assert (
        left["configuration"]["configuration_id"]
        != changed_configuration["configuration"]["configuration_id"]
    )
    verify_qualification_artifact(left)
    artifact = _artifact("mac")

    assert artifact["route"] == "mac"
    assert artifact["support_tuple"]["attributes"]["route"] == "mac"
    assert artifact["release_ready"] is False
    assert "schema" not in artifact
    assert "schema_version" not in artifact
    assert "version" not in artifact
    assert artifact["configuration"]["timing_gate"] is False
    support = phx.qualification.SupportTuple(
        "incompressible-flow",
        {"route": "spectral-channel", "method": "native-test-route"},
    )
    artifact = make_qualification_artifact(
        route="spectral-channel",
        support_tuple=support,
        inputs={"case": "missing-required-evidence"},
        reference={"source": "native-analytic"},
        configuration={"timing_gate": False},
        metrics={"couette": {"error": 0.0}},
        gates=(
            gate_outcome(
                "couette",
                "passed",
                "The supplied Couette metric satisfied its criterion.",
            ),
        ),
    )

    assert artifact["status"] == "inconclusive"
    assert artifact["failed_reasons"] == []
    # ty: ignore[invalid-argument-type]
    assert len(artifact["inconclusive_reasons"]) == 2
    assert any(
        gate["gate"] == "required-case:poiseuille" and gate["outcome"] == "inconclusive"
        # ty: ignore[not-iterable]
        for gate in artifact["gates"]
    )
    gate = numeric_gate("required-error", "case.error", None, 1.0e-9)
    support = phx.qualification.SupportTuple(
        "incompressible-flow",
        {"route": "periodic-spectral", "method": "native-test-route"},
    )
    artifact = make_qualification_artifact(
        route="periodic-spectral",
        support_tuple=support,
        inputs={"case": "non-finite"},
        reference={"source": "native-analytic"},
        configuration={"error_tolerance": 1.0e-9, "timing_gate": False},
        metrics={name: {"error": None} for name in REQUIRED_CASES["periodic-spectral"]},
        gates=(gate,),
    )

    assert gate["outcome"] == "failed"
    assert gate["reason"]
    assert artifact["status"] == "failed"
    assert artifact["failed_reasons"] == [gate["reason"]]


def test_incompressible_flow_qualification_scenario_2() -> None:
    with pytest.raises(ValueError, match="all-or-none"):
        external_reference_input(path="external.csv", checksum="sha256:abc")

    reference = external_reference_input(
        path="external.csv",
        checksum="sha256:0123456789abcdef",
        nondimensionalization={"length": "channel-half-height"},
        uncertainty={"velocity_standard_uncertainty": 0.01},
    )

    assert reference is not None
    assert reference["path"] == "external.csv"
    assert reference["checksum_verified"] is False
    assert reference["executed"] is False
    assert reference["reference_input_id"]
    artifact = _artifact()
    candidate = assemble_candidate_profile((artifact,))

    assert candidate["qualification_artifact_ids"] == [artifact["artifact_id"]]
    assert candidate["release_ready"] is False
    assert candidate["signed"] is False
    assert "signature" not in candidate
    # ty: ignore[not-subscriptable]
    assert candidate["profile"]["released"] is False
    # ty: ignore[not-subscriptable]
    assert candidate["profile"]["release_evidence"] == []
    assert candidate["candidate_id"]

    tampered = dict(artifact)
    tampered["release_ready"] = True
    with pytest.raises(ValueError, match="unreleased"):
        assemble_candidate_profile((tampered,))
    spectral_fields = {
        field.name for field in fields(IncompressibleSpectralBenchmarkRecord)
    }
    structured_fields = {field.name for field in fields(StructuredFlowBenchmarkRecord)}

    for names in (spectral_fields, structured_fields):
        assert "schema_version" not in names
        assert "passed" not in names
        assert "smoke_successful" in names


@pytest.mark.parametrize(
    ("kernel", "highpass_exponent", "admissible", "cutoff_scale"),
    (
        ("tadmor_step", None, True, 0.5),
        ("tadmor_step", None, True, 3.0),
        ("exponential_highpass", 4, False, 0.5),
    ),
)
def test_svv_qualification_analytic_cases_and_nonclaims(
    kernel: SpectralViscosityKernel,
    highpass_exponent: int | None,
    admissible: bool,
    cutoff_scale: float,
) -> None:
    artifact: Any = periodic_spectral_vanishing_viscosity_qualification(
        strength=0.1,
        order=1.0,
        cutoff_scale=cutoff_scale,
        cutoff_exponent=0.25,
        kernel=kernel,
        highpass_exponent=highpass_exponent,
        mode_count=8,
        low_mode_counts=(8, 16, 32, 64),
        step_size=0.025,
        steps=2,
    )
    route = "periodic-spectral-vanishing-viscosity"
    verify_qualification_artifact(artifact)
    assert artifact["route"] == route
    assert artifact["support_tuple"]["attributes"]["route"] == route
    assert artifact["support_tuple"]["attributes"]["temporal_method"] == "etdrk4"
    assert artifact["status"] == "passed", artifact["gates"]
    assert artifact["release_ready"] is False
    assert artifact["configuration"]["timing_gate"] is False
    assert set(artifact["metrics"]) == set(REQUIRED_CASES[route])
    for case in artifact["metrics"].values():
        assert case["operator_scaling_admissible"] is admissible
        assert case["cited_convergence_supported"] is False
        assert "initial-data mollification" in case["support_reason"]
    for dimension in (2, 3):
        decay = artifact["metrics"][f"shear-decay-{dimension}d"]
        energy = artifact["metrics"][f"energy-identity-{dimension}d"]
        assert len(decay["integer_mode"]) == dimension
        mode_norm = sum(value**2 for value in decay["integer_mode"]) ** 0.5
        assert mode_norm > 0.0
        if kernel == "tadmor_step":
            assert mode_norm > cutoff_scale * 3**0.25
        assert decay["analytic_decay_rate"] < 0.0
        assert decay["oracle_resolution"] == 3
        assert decay["prepared_resolution"] == decay["oracle_resolution"]
        assert decay["prepared_rate_error"] <= 1.0e-9
        assert decay["maximum_velocity_error"] <= 1.0e-8
        assert decay["divergence_norm"] <= 1.0e-9
        assert decay["solver_successful"] is True
        assert energy["maximum_energy_balance_defect"] <= 1.0e-9
        assert energy["maximum_dissipation_error"] <= 1.0e-9
        assert energy["maximum_energy_rate_error"] <= 1.0e-9
        assert energy["spectral_viscosity_dissipation"] > 0.0
        assert energy["spectral_viscosity_energy_rate"] < 0.0
        assert energy["diagnostics_finite"] is True
    low = artifact["metrics"]["fixed-low-mode-vanishing"]
    assert low["integer_mode"] == [0, 1]
    assert low["vanishing_upper_bound_exponent"] == -1.0
    assert low["nonincreasing_absolute_rate"] is True
    assert low["upper_bound_decreases"] is True
    assert low["maximum_rate_error"] <= 1.0e-9
    rows = low["resolutions"]
    assert [row["mode_count"] for row in rows] == [8, 16, 32, 64]
    assert [row["resolution"] for row in rows] == [3, 7, 15, 31]
    assert abs(rows[-1]["decay_rate"]) <= abs(rows[0]["decay_rate"])
    for row in rows:
        assert row["operator_scaling_admissible"] is admissible
        assert row["cited_convergence_supported"] is False
        assert row["support_reason"] == low["support_reason"]
    if kernel == "tadmor_step":
        assert rows[-1]["decay_rate"] == 0.0
    else:
        assert rows[-1]["decay_rate"] < 0.0
    candidate: Any = assemble_candidate_profile((artifact,))
    assert candidate["qualified_routes"] == [route]
    assert candidate["signed"] is False
    assert candidate["profile"]["released"] is False


def test_svv_qualification_missing_case_is_inconclusive() -> None:
    route = "periodic-spectral-vanishing-viscosity"
    artifact: Any = make_qualification_artifact(
        route=route,
        support_tuple=phx.qualification.SupportTuple(
            "incompressible-flow", {"route": route}
        ),
        inputs={"case": "missing-low-mode-evidence"},
        reference={"source": "native-analytic"},
        configuration={"timing_gate": False},
        metrics={
            case: {"maximum_error": 0.0}
            for case in REQUIRED_CASES[route]
            if case != "fixed-low-mode-vanishing"
        },
        gates=(numeric_gate("shear", "shear.error", 0.0, 1.0e-8),),
    )
    assert artifact["status"] == "inconclusive"
    assert any(
        gate["gate"] == "required-case:fixed-low-mode-vanishing"
        and gate["outcome"] == "inconclusive"
        for gate in artifact["gates"]
    )
    with pytest.raises(ValueError, match="not passed"):
        assemble_candidate_profile((artifact,))


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"mode_count": 7}, "count/steps/step size"),
        ({"steps": 0}, "count/steps/step size"),
        ({"step_size": float("nan")}, "count/steps/step size"),
        ({"low_mode_counts": (8, 16)}, "three or more"),
        ({"low_mode_counts": (8, 16, 16)}, "strictly increase"),
        ({"fixed_low_mode": 4, "low_mode_counts": (8, 16, 32)}, "retain"),
        ({"invariant_tolerance": float("inf")}, "finite and positive"),
    ),
)
def test_svv_qualification_rejects_invalid_campaign(
    options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        periodic_spectral_vanishing_viscosity_qualification(
            strength=0.1,
            order=1.0,
            cutoff_scale=0.5,
            cutoff_exponent=0.25,
            kernel="tadmor_step",
            **options,
        )


def _small_statistical_campaign(**options: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "strength": 0.1,
        "order": 1.0,
        "cutoff_scale": 0.5,
        "cutoff_exponent": 0.25,
        "kernel": "tadmor_step",
        "power_law_exponent": 1.5,
        "velocity_scale": 0.02,
        "maximum_courant_number": 0.5,
        "mode_counts": (12, 24, 48),
        "members": 4,
        "steps": 1,
        "num_projections": 2,
    }
    return periodic_statistical_solution_qualification(**(config | options))


@pytest.fixture(scope="module")
def statistical_campaign() -> dict[str, Any]:
    return _small_statistical_campaign()


def test_periodic_statistical_solution_records_finite_campaign_nonclaims(
    statistical_campaign: dict[str, Any],
) -> None:
    artifact = statistical_campaign
    verify_qualification_artifact(artifact)
    assert artifact["route"] == "periodic-statistical-solution"
    assert not artifact["release_ready"]
    campaign = artifact["metrics"]["coupled-campaign"]
    assert len(campaign["runtime"]) == 3
    assert len(campaign["comparisons"]) == 2
    assert campaign["member_count"] == 4
    assert len(campaign["delete_one_member_distances"]) == 4
    for row in campaign["comparisons"]:
        assert row["componentwise_empirical_wasserstein"] >= 0
        assert (
            row["paired_coupling_upper_bound"]
            >= row["componentwise_empirical_wasserstein"]
        )
        assert row["jackknife_standard_error"] >= 0
        assert len(row["sliced_joint_marginal_distance"]) == 2
    deleted = np.asarray(campaign["delete_one_member_distances"])
    paired = deleted[:, 0] - deleted[:, 1]
    expected_se = np.sqrt(3 / 4 * np.sum((paired - paired.mean()) ** 2))
    assert campaign["paired_difference_standard_errors"][0] == pytest.approx(expected_se)
    differences = np.asarray(campaign["refinement_differences"])
    errors = np.asarray(campaign["paired_difference_standard_errors"])
    z = campaign["refinement_z"]
    expected = (
        "failed"
        if np.any(differences < -z * errors)
        else "passed"
        if np.all(differences > z * errors)
        else "inconclusive"
    )
    gates = {gate["gate"]: gate for gate in artifact["gates"]}
    assert gates["finite-campaign-refinement"]["outcome"] == expected
    scaling_passed = gates["finite-campaign-scaling"]["outcome"] == "passed"
    assert (artifact["status"] == "passed") == (expected == "passed" and scaling_passed)
    assert all(
        gate["outcome"] == "passed"
        for name, gate in gates.items()
        if name not in ("finite-campaign-refinement", "finite-campaign-scaling")
    )
    assert "estimated_monte_carlo_floor" not in campaign
    assert campaign["nonclaims"]


def test_periodic_statistical_solution_rejects_finest_ensemble_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = phx.applications.incompressible_flow.PreparedPeriodicFlowEnsemble
    execute = owner.execute

    def faulty_execute(prepared: Any) -> Any:
        wrapper = execute(prepared)
        if wrapper.initial_state.discretization.modal_shape[0] == 48:
            native = wrapper.result
            # A common mean-mode shift is an actual finest-trajectory fault:
            # divergence/SVV invariants remain valid, but W1 refinement increases.
            states = native.states.at[:, -1, 0, 0, 0].add(10.0)
            native = eqx.tree_at(
                lambda r: (r.states, r.solutions.states), native, (states, states)
            )
            wrapper = phx.applications.incompressible_flow.PeriodicFlowEnsembleResult(
                native, wrapper.initial_state
            )
        return wrapper

    monkeypatch.setattr(owner, "execute", faulty_execute)
    artifact = _small_statistical_campaign()
    assert artifact["status"] == "failed"
    refinement = next(
        g for g in artifact["gates"] if g["gate"] == "finite-campaign-refinement"
    )
    assert refinement["outcome"] == "failed"
    assert artifact["metrics"]["coupled-campaign"]["refinement_differences"][0] < 0


def test_periodic_statistical_solution_binds_kernel_provenance(
    statistical_campaign: dict[str, Any],
) -> None:
    changed = _small_statistical_campaign(
        kernel="exponential_highpass", highpass_exponent=4
    )
    left, right = statistical_campaign["input"], changed["input"]
    assert left["input_id"] != right["input_id"]
    for a, b in zip(left["resolutions"], right["resolutions"], strict=True):
        for name in (
            "compilation_id",
            "method_id",
            "ensemble_prepared_id",
            "ensemble_result_id",
            "discretization_prepared_id",
            "spectral_viscosity_plan_id",
            "spectral_viscosity_prepared_id",
            "measure_id",
            "coupling_id",
        ):
            assert a[name] and b[name]
        for name in (
            "compilation_id",
            "method_id",
            "ensemble_prepared_id",
            "ensemble_result_id",
            "spectral_viscosity_plan_id",
            "spectral_viscosity_prepared_id",
        ):
            assert a[name] != b[name]
        assert a["discretization_prepared_id"] == b["discretization_prepared_id"]
    attributes = changed["support_tuple"]["attributes"]
    assert attributes["kernel"] == "exponential_highpass"
    assert attributes["spatial_dimensions"] == "2"
    assert attributes["spatial_method"] == "fourier-pseudospectral"
    assert attributes["convergence_certified"] == "false"


@pytest.mark.parametrize("abstained", (False, True))
def test_periodic_statistical_solution_inconsistent_scaling_cannot_pass(
    statistical_campaign: dict[str, Any],
    abstained: bool,
) -> None:
    campaign = statistical_campaign["metrics"]["coupled-campaign"]
    scaling = dict(campaign["scaling"])
    scaling["consistent_within_campaign"] = abstained
    scaling["fits"] = [dict(f) for f in scaling["fits"]]
    if abstained:
        scaling["fits"][0].update(status="abstained", reason="Deliberate fit abstention.")
    metrics = dict(
        campaign,
        refinement_differences=[1.0],
        paired_difference_standard_errors=[0.0],
        scaling=scaling,
    )
    gates = qualification._statistical_metric_gates(metrics)
    assert (
        next(g for g in gates if g["gate"] == "finite-campaign-refinement")["outcome"]
        == "passed"
    )
    scaling_gate = next(g for g in gates if g["gate"] == "finite-campaign-scaling")
    assert scaling_gate["outcome"] == "inconclusive"
    artifact = make_qualification_artifact(
        route="periodic-statistical-solution",
        support_tuple=phx.qualification.SupportTuple(
            "incompressible-flow", {"route": "periodic-statistical-solution"}
        ),
        inputs={},
        reference={},
        configuration={},
        metrics={"coupled-campaign": metrics},
        gates=gates,
    )
    assert artifact["status"] == "inconclusive"


def test_periodic_statistical_solution_records_guard_refusal_as_failed_artifact() -> None:
    artifact = periodic_statistical_solution_qualification(
        strength=0.1,
        order=1.0,
        cutoff_scale=0.5,
        cutoff_exponent=0.25,
        kernel="tadmor_step",
        power_law_exponent=1.5,
        velocity_scale=0.02,
        maximum_courant_number=0.5,
        mode_counts=(12, 24, 48),
        members=4,
        steps=1,
        num_projections=8,
        step_size=1e4,
    )
    verify_qualification_artifact(artifact)
    assert artifact["status"] == "failed"
    gates = artifact["gates"]
    assert isinstance(gates, list)
    assert any(
        gate["gate"].startswith("solver-success:") and gate["outcome"] == "failed"
        for gate in gates
    )
    metrics = artifact["metrics"]
    assert isinstance(metrics, dict)
    runtime = metrics["coupled-campaign"]["runtime"]
    assert all(
        item["solver_successful"] is False and item["guard_evidence"] for item in runtime
    )
    assert all(
        max(item["guard_evidence"]["held_terminal_state_maximum_step"]) < 1e4
        for item in runtime
    )
    assert all(
        max(max(row) for row in item["guard_evidence"]["step_residuals"]) > 0
        for item in runtime
    )
    assert "componentwise_empirical_wasserstein" not in metrics["coupled-campaign"]


def test_periodic_statistical_solution_refuses_degenerate_radius_window() -> None:
    with pytest.raises(ValueError, match="coarse_count > 8"):
        periodic_statistical_solution_qualification(
            strength=0.1,
            order=1.0,
            cutoff_scale=0.5,
            cutoff_exponent=0.25,
            kernel="tadmor_step",
            power_law_exponent=1.5,
            velocity_scale=0.02,
            maximum_courant_number=0.5,
            mode_counts=(8, 16, 32),
        )


@pytest.mark.parametrize("counts", ((12, 24), (12, 24, 36), (12, 24, 24)))
def test_periodic_statistical_solution_requires_three_nested_resolutions(
    counts: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="at least three nested"):
        _small_statistical_campaign(mode_counts=counts)
