from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


sm = phx.applications.solid_mechanics
mn = sm.member_network
std = sm.standards


def test_precedence_branch_and_bound_finds_exact_sequence():
    operations = (
        phx.optim.PrecedenceOperation("foundation"),
        phx.optim.PrecedenceOperation("frame", predecessors=("foundation",)),
        phx.optim.PrecedenceOperation("cable", predecessors=("frame",)),
    )
    space = phx.optim.PrecedenceSpace(operations)
    costs = {"foundation": 3.0, "frame": 2.0, "cable": 1.0}
    problem = mn.ConstructionSequenceSearchProblem(
        space,
        lambda node: (True, None),
        lambda node: sum(costs[value] for value in node.completed),
        lambda node: sum(
            (index + 1) * costs[value] for index, value in enumerate(node.completed)
        ),
    )
    result = mn.search_construction_sequences(problem)
    assert result.successful
    assert result.incumbent.completed == ("foundation", "frame", "cable")
    assert result.objective == pytest.approx(10.0)


def test_generic_standard_retains_clause_and_applicability():
    combination = std.LoadCombination(
        "ultimate",
        {"dead": 1.2, "live": 1.6},
        category="ultimate",
        clause_id="LC-1",
    )
    standard = std.GenericLimitStateStandard(
        (combination,), resistance_factor=0.9, edition="2026"
    )
    combined = combination.combine({"dead": jnp.asarray(10.0), "live": jnp.asarray(5.0)})
    evidence = standard.member_resistance(
        combined,
        30.0,
        clause_id="M-1",
        governing_case="ultimate",
    )
    assert combined == pytest.approx(20.0)
    assert evidence.successful
    assert evidence.edition == "2026"
    outside = standard.member_resistance(
        combined,
        30.0,
        clause_id="M-1",
        governing_case="ultimate",
        applicability=std.ApplicabilityStatus.OUTSIDE_APPLICABILITY,
    )
    assert not outside.successful


def test_reliability_form_and_monte_carlo_match_normal_limit_state():
    model = mn.StructuralRandomModel(
        jnp.asarray((0.0,)), jnp.asarray(((1.0,),)), ("load",)
    )
    limit = mn.StructuralLimitState(
        lambda parameter: 1.0 - parameter[0], limit_state_id="normal-threshold"
    )
    form = mn.form_reliability(model, limit)
    monte_carlo = mn.monte_carlo_reliability(model, limit, jax.random.PRNGKey(0), 20_000)
    assert form.reliability_index == pytest.approx(1.0, rel=1.0e-5)
    assert form.converged
    assert monte_carlo.failure_probability == pytest.approx(0.1587, abs=0.015)


def test_calibration_identifiability_and_evidence_graph():
    observation = mn.StructuralObservationModel(
        lambda parameter, _: jnp.asarray((2.0 * parameter[0],)),
        jnp.asarray((4.0,)),
        jnp.asarray(((0.01,),)),
        discrepancy_covariance=jnp.asarray(((0.01,),)),
        observation_id="tip-displacement",
    )
    problem = mn.StructuralCalibrationProblem(
        (observation,),
        jnp.asarray((0.0,)),
        jnp.asarray(((10.0,),)),
        bounds=phx.optim.Bounds(jnp.asarray((-10.0,)), jnp.asarray((10.0,))),
    )
    calibrated = mn.calibrate_structural_map(problem, jnp.asarray((1.0,)))
    assert calibrated.successful
    assert calibrated.optimization.parameters[0] == pytest.approx(2.0, abs=0.02)

    action = mn.EvidenceAcquisitionAction(
        "measure-brace",
        "Measure brace stiffness",
        ("ltb",),
        10.0,
        5.0,
    )
    material = mn.EvidenceNode(
        "material",
        (),
        "measured",
        int(mn.EvidenceStatus.CERTIFIED),
        None,
        (),
        (),
    )
    ltb = mn.EvidenceNode(
        "ltb",
        ("material",),
        "warping-beam",
        int(mn.EvidenceStatus.INCOMPLETE),
        None,
        ("brace-stiffness",),
        (action,),
    )
    graph = mn.EvidenceGraph((material, ltb))
    assert graph.required_actions("ltb")[0].action_id == "measure-brace"
    snapshot = mn.StructuralTwinSnapshot.create(graph, design_state={"revision": 1})
    updated = mn.StructuralTwinSnapshot.create(
        graph,
        design_state={"revision": 1},
        as_built_state={"revision": 2},
        parent_snapshot_id=snapshot.snapshot_id,
    )
    assert updated.parent_snapshot_id == snapshot.snapshot_id
