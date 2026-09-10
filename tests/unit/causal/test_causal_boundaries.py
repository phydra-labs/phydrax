#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import causal


def _assumptions() -> causal.AssumptionLedger:
    return causal.AssumptionLedger(
        causal.CausalAssumption(
            kind=kind,
            disposition=causal.AssumptionDisposition.DECLARED,
            statement=f"Synthetic declaration for {kind.value}.",
        )
        for kind in (
            causal.AssumptionKind.CONSISTENCY,
            causal.AssumptionKind.POSITIVITY,
            causal.AssumptionKind.NO_INTERFERENCE,
            causal.AssumptionKind.CAUSAL_MARKOV,
        )
    )


def _binary_problem(schema: causal.CausalSchema, probabilities: np.ndarray):
    assignments = np.indices(probabilities.shape).reshape(len(probabilities.shape), -1).T
    repeats = np.maximum(np.rint(probabilities.reshape(-1) * 10_000), 1).astype(int)
    rows = np.repeat(assignments, repeats, axis=0)
    dataset = causal.CausalDataset(
        schema=schema,
        values=tuple(
            jnp.asarray(rows[:, index], dtype=jnp.int32)
            for index in range(len(schema.names))
        ),
    )
    design = causal.CausalStudyDesign(
        schema=schema,
        assignment_kind=causal.AssignmentKind.OBSERVATIONAL,
        assignment_variable="x",
        exposure_variable="x",
        source_population_id="population",
        assumptions=_assumptions(),
        no_interference=True,
    )
    query = causal.CausalQuery(
        schema=schema,
        outcome_variable="y",
        contrast=causal.TreatmentContrast(
            active=causal.TreatmentRegime(exposure_variable="x", value=1),
            reference=causal.TreatmentRegime(exposure_variable="x", value=0),
        ),
        population=causal.TargetPopulation(source_population_id="population"),
    )
    return causal.CausalProblem(dataset=dataset, design=design, query=query)


def test_bow_arc_returns_hedge_nonidentification():
    schema = causal.CausalSchema(
        (
            causal.CausalVariable(name="x", scale=causal.VariableScale.BINARY),
            causal.CausalVariable(name="y", scale=causal.VariableScale.BINARY),
        )
    )
    graph = causal.CausalADMG(
        schema=schema,
        directed_edges=(("x", "y"),),
        bidirected_edges=(("x", "y"),),
    )
    probabilities = np.asarray([[0.4, 0.1], [0.1, 0.4]])
    problem = _binary_problem(schema, probabilities)

    result = causal.identify_causal_effect(problem, graph)

    assert result.status is causal.IdentificationStatus.NOT_IDENTIFIED
    assert result.witness is not None
    with pytest.raises(ValueError, match="successful identification"):
        causal.issue_identification_certificate(result, problem)


def test_frontdoor_general_id_matches_brute_force_intervention():
    schema = causal.CausalSchema(
        tuple(
            causal.CausalVariable(name=name, scale=causal.VariableScale.BINARY)
            for name in ("x", "m", "y")
        )
    )
    graph = causal.CausalADMG(
        schema=schema,
        directed_edges=(("x", "m"), ("m", "y")),
        bidirected_edges=(("x", "y"),),
    )
    p_u = np.asarray([0.5, 0.5])
    p_x_given_u = np.asarray([[0.8, 0.2], [0.2, 0.8]])
    p_m_given_x = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    p_y_given_m_u = np.asarray(
        [
            [[0.95, 0.05], [0.6, 0.4]],
            [[0.7, 0.3], [0.1, 0.9]],
        ]
    )
    probabilities = np.zeros((2, 2, 2))
    intervention_mean = np.zeros((2,))
    for u, x, m, y in itertools.product(range(2), repeat=4):
        probabilities[x, m, y] += (
            p_u[u] * p_x_given_u[u, x] * p_m_given_x[x, m] * p_y_given_m_u[m, u, y]
        )
    for x, u, m, y in itertools.product(range(2), repeat=4):
        intervention_mean[x] += y * p_u[u] * p_m_given_x[x, m] * p_y_given_m_u[m, u, y]
    problem = _binary_problem(schema, probabilities)
    identified = causal.identify_causal_effect(problem, graph)

    assert identified.status is causal.IdentificationStatus.IDENTIFIED
    assert identified.basis is causal.IdentificationBasis.GENERAL_ID
    certificate = causal.issue_identification_certificate(identified, problem)
    evaluated = causal.evaluate_finite_effect(
        certificate,
        problem,
        causal.FiniteObservedLaw(schema=schema, probabilities=jnp.asarray(probabilities)),
    )

    assert evaluated.successful
    assert float(evaluated.active_mean) == pytest.approx(intervention_mean[1], abs=1e-9)
    assert float(evaluated.reference_mean) == pytest.approx(
        intervention_mean[0], abs=1e-9
    )


def test_completed_graph_rejects_noncompleted_orientation():
    schema = causal.CausalSchema(
        tuple(causal.CausalVariable(name=name) for name in ("a", "b", "c"))
    )
    chain = causal.CausalDAG(
        schema=schema,
        directed_edges=(("a", "b"), ("b", "c")),
    )
    completed = causal.complete_dag_equivalence_class(chain)
    assert completed.directed_edges == ()
    assert completed.undirected_edges == (("a", "b"), ("b", "c"))

    with pytest.raises(ValueError, match="not a completed"):
        causal.CausalCPDAG(
            schema=schema,
            directed_edges=(("a", "b"),),
            undirected_edges=(("b", "c"),),
        )


def test_g_square_and_conservative_fci_are_evidence_bounded():
    repeats = 50
    x = np.tile(np.asarray([0, 0, 1, 1]), repeats)
    y = np.tile(np.asarray([0, 1, 0, 1]), repeats)
    schema = causal.CausalSchema(
        (
            causal.CausalVariable(name="x", scale=causal.VariableScale.BINARY),
            causal.CausalVariable(name="y", scale=causal.VariableScale.BINARY),
        )
    )
    dataset = causal.CausalDataset(
        schema=schema,
        values=(jnp.asarray(x), jnp.asarray(y)),
    )
    test = causal.GSquareTest(alpha=0.05)
    result = test.test(dataset, "x", "y")
    assert result.successful
    assert result.independent

    discovery = causal.discover_conservative_fci(
        dataset,
        causal.ConservativeFCIPlan(
            ci_test=test,
            knowledge=causal.DiscoveryBackgroundKnowledge(schema=schema),
            resources=causal.DiscoveryResourcePolicy(
                maximum_ci_tests=20,
                maximum_conditioning_depth=0,
            ),
        ),
    )
    assert discovery.successful
    assert isinstance(discovery.graph, causal.CausalPAG)
    assert discovery.graph.endpoint_edges == ()


def test_exact_bounded_ges_returns_score_equivalence_class():
    rng = np.random.default_rng(21)
    x = rng.normal(size=300)
    y = 1.5 * x + rng.normal(scale=0.4, size=300)
    schema = causal.CausalSchema(
        (causal.CausalVariable(name="x"), causal.CausalVariable(name="y"))
    )
    dataset = causal.CausalDataset(
        schema=schema,
        values=(jnp.asarray(x), jnp.asarray(y)),
    )

    result = causal.discover_ges(
        dataset,
        causal.GESPlan(
            maximum_steps=4,
            resources=causal.DiscoveryResourcePolicy(
                maximum_extensions=16,
                maximum_score_candidates=100,
            ),
        ),
    )

    assert result.successful
    assert isinstance(result.graph, causal.CausalCPDAG)
    assert result.graph.directed_edges == ()
    assert result.graph.undirected_edges == (("x", "y"),)


def test_idc_evaluates_conditional_interventional_distribution():
    schema = causal.CausalSchema(
        tuple(
            causal.CausalVariable(name=name, scale=causal.VariableScale.BINARY)
            for name in ("c", "x", "y")
        )
    )
    graph = causal.CausalADMG(
        schema=schema,
        directed_edges=(("c", "y"), ("x", "y")),
    )
    probabilities = np.zeros((2, 2, 2))
    for c, x, y in itertools.product(range(2), repeat=3):
        positive = 0.1 + 0.3 * x + 0.4 * c
        probabilities[c, x, y] = 0.25 * (positive if y == 1 else 1.0 - positive)
    identified = causal.identify_conditional_distribution(
        graph,
        outcomes=("y",),
        interventions=("x",),
        conditioned=("c",),
    )
    evaluated = causal.evaluate_finite_distribution(
        identified,
        causal.FiniteObservedLaw(
            schema=schema,
            probabilities=jnp.asarray(probabilities),
        ),
        fixed_values={"x": 1, "c": 1},
    )

    assert identified.identified
    assert evaluated.successful
    assert evaluated.variables == ("y",)
    assert np.asarray(evaluated.probabilities) == pytest.approx([0.2, 0.8])
