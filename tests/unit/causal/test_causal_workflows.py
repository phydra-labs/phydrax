#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import causal
from phydrax.ml.linear import OLSRecipe
from phydrax.ml.model_selection import KFoldPlan


def _assumptions() -> causal.AssumptionLedger:
    return causal.AssumptionLedger(
        causal.CausalAssumption(
            kind=kind,
            disposition=causal.AssumptionDisposition.DECLARED,
            statement=f"Declared {kind.value} for the synthetic study.",
        )
        for kind in (
            causal.AssumptionKind.CONSISTENCY,
            causal.AssumptionKind.POSITIVITY,
            causal.AssumptionKind.NO_INTERFERENCE,
            causal.AssumptionKind.CAUSAL_MARKOV,
        )
    )


def _finite_problem():
    schema = causal.CausalSchema(
        (
            causal.CausalVariable(name="z", scale=causal.VariableScale.BINARY),
            causal.CausalVariable(name="t", scale=causal.VariableScale.BINARY),
            causal.CausalVariable(name="y", scale=causal.VariableScale.BINARY),
        )
    )
    graph = causal.CausalDAG(
        schema=schema,
        directed_edges=(("z", "t"), ("z", "y"), ("t", "y")),
    )
    mechanisms = (
        causal.FiniteConditionalMechanism(
            output="z", parents=(), probabilities=jnp.asarray([0.5, 0.5])
        ),
        causal.FiniteConditionalMechanism(
            output="t",
            parents=("z",),
            probabilities=jnp.asarray([[0.8, 0.2], [0.2, 0.8]]),
        ),
        causal.FiniteConditionalMechanism(
            output="y",
            parents=("z", "t"),
            probabilities=jnp.asarray(
                [
                    [[0.9, 0.1], [0.5, 0.5]],
                    [[0.7, 0.3], [0.1, 0.9]],
                ]
            ),
        ),
    )
    scm = causal.SCMPlan(graph=graph, mechanisms=mechanisms, exogenous_independence=True)
    compiled = causal.compile_finite_scm(causal.baseline_regime(scm))
    law = causal.finite_observed_law_from_scm(compiled)
    assignments = np.indices((2, 2, 2)).reshape(3, -1).T
    repeats = np.maximum(
        np.rint(np.asarray(law.probabilities).reshape(-1) * 1000), 1
    ).astype(int)
    rows = np.repeat(assignments, repeats, axis=0)
    dataset = causal.CausalDataset(
        schema=schema,
        values=tuple(jnp.asarray(rows[:, index], dtype=jnp.int32) for index in range(3)),
    )
    design = causal.CausalStudyDesign(
        schema=schema,
        assignment_kind=causal.AssignmentKind.OBSERVATIONAL,
        assignment_variable="t",
        exposure_variable="t",
        source_population_id="finite-population",
        assumptions=_assumptions(),
        no_interference=True,
    )
    query = causal.CausalQuery(
        schema=schema,
        outcome_variable="y",
        contrast=causal.TreatmentContrast(
            active=causal.TreatmentRegime(exposure_variable="t", value=1),
            reference=causal.TreatmentRegime(exposure_variable="t", value=0),
        ),
        population=causal.TargetPopulation(source_population_id="finite-population"),
    )
    return (
        causal.CausalProblem(dataset=dataset, design=design, query=query),
        graph,
        scm,
        law,
    )


def test_graph_separation_adjustment_and_finite_identification():
    problem, graph, _, law = _finite_problem()

    backdoor = causal.enumerate_adjustment_sets(graph, treatment="t", outcome="y")
    assert backdoor == (("z",),)
    assert causal.d_separated(
        causal.CausalDAG(
            schema=graph.schema,
            directed_edges=(("z", "t"), ("z", "y")),
        ),
        ("t",),
        ("y",),
        ("z",),
    ).separated

    identified = causal.identify_causal_effect(problem, graph, adjustment_set=("z",))
    certificate = causal.issue_identification_certificate(identified, problem)
    result = causal.evaluate_finite_effect(certificate, problem, law)

    assert result.successful
    assert float(result.active_mean) == pytest.approx(0.7)
    assert float(result.reference_mean) == pytest.approx(0.2)
    assert float(result.effect) == pytest.approx(0.5)


def test_latent_projection_and_perfect_intervention_cut_confounding():
    schema = causal.CausalSchema(
        (
            causal.CausalVariable(
                name="u",
                observability=causal.VariableObservability.LATENT,
                scale=causal.VariableScale.BINARY,
            ),
            causal.CausalVariable(name="x", scale=causal.VariableScale.BINARY),
            causal.CausalVariable(name="y", scale=causal.VariableScale.BINARY),
        )
    )
    dag = causal.CausalDAG(
        schema=schema,
        directed_edges=(("u", "x"), ("u", "y"), ("x", "y")),
    )
    projection = causal.latent_project(dag)
    assert projection.directed_edges == (("x", "y"),)
    assert projection.bidirected_edges == (("x", "y"),)

    intervened = causal.intervene_graph(projection, ("x",))
    assert isinstance(intervened, causal.CausalADMG)
    assert intervened.directed_edges == (("x", "y"),)
    assert intervened.bidirected_edges == ()


def test_counterfactual_reuses_abducted_exogenous_state():
    schema = causal.CausalSchema(
        (
            causal.CausalVariable(name="x"),
            causal.CausalVariable(name="y"),
        )
    )
    graph = causal.CausalDAG(schema=schema, directed_edges=(("x", "y"),))
    root = causal.InvertibleNoiseMechanism(
        output="x",
        parents=(),
        noise_sampler=lambda key, n: jax.random.normal(key, (n,)),
        forward=lambda parents, noise: noise,
        inverse=lambda parents, value: value,
        semantic_ids=("root-noise", "root-forward", "root-inverse"),
        numeric_ids=("root-noise-r0", "root-forward-r0", "root-inverse-r0"),
    )
    child = causal.InvertibleNoiseMechanism(
        output="y",
        parents=("x",),
        noise_sampler=lambda key, n: jax.random.normal(key, (n,)),
        forward=lambda parents, noise: parents[0] + noise,
        inverse=lambda parents, value: value - parents[0],
        semantic_ids=("child-noise", "child-forward", "child-inverse"),
        numeric_ids=("child-noise-r0", "child-forward-r0", "child-inverse-r0"),
    )
    with pytest.raises(ValueError, match="Correlated exogenous"):
        causal.SCMPlan(
            graph=graph,
            mechanisms=(root, child),
            exogenous_independence=False,
        )
    scm = causal.SCMPlan(
        graph=graph, mechanisms=(root, child), exogenous_independence=True
    )
    factual = causal.FactualObservation(
        schema=schema,
        values=(jnp.asarray([1.0]), jnp.asarray([3.0])),
    )
    abduction = causal.abduct_factual(scm, factual)
    regime = causal.build_intervention_regime(
        scm,
        (causal.PerfectIntervention(variable="x", value=2.0),),
    )
    counterfactual = causal.evaluate_counterfactual(regime, factual, abduction)

    assert counterfactual.successful
    assert np.asarray(counterfactual.value("x")) == pytest.approx([2.0])
    assert np.asarray(counterfactual.value("y")) == pytest.approx([4.0])


def test_cross_fitted_aipw_and_archive_round_trip(tmp_path):
    n = 120
    z = np.tile(np.asarray([-1.0, -1.0, 1.0, 1.0]), n // 4)
    treatment = np.arange(n) % 2
    outcome = 2.0 * treatment + z
    schema = causal.CausalSchema(
        (
            causal.CausalVariable(name="z"),
            causal.CausalVariable(name="t", scale=causal.VariableScale.BINARY),
            causal.CausalVariable(name="y"),
        )
    )
    dataset = causal.CausalDataset(
        schema=schema,
        values=(jnp.asarray(z), jnp.asarray(treatment), jnp.asarray(outcome)),
    )
    design = causal.CausalStudyDesign(
        schema=schema,
        assignment_kind=causal.AssignmentKind.RANDOMIZED,
        assignment_variable="t",
        exposure_variable="t",
        source_population_id="trial",
        assumptions=_assumptions(),
        no_interference=True,
        known_assignment_probability=jnp.full((n,), 0.5),
    )
    query = causal.CausalQuery(
        schema=schema,
        outcome_variable="y",
        contrast=causal.TreatmentContrast(
            active=causal.TreatmentRegime(exposure_variable="t", value=1),
            reference=causal.TreatmentRegime(exposure_variable="t", value=0),
        ),
        population=causal.TargetPopulation(source_population_id="trial"),
    )
    problem = causal.CausalProblem(dataset=dataset, design=design, query=query)
    identification = causal.identify_causal_effect(problem, None)
    certificate = causal.issue_identification_certificate(identification, problem)
    nuisance = causal.fit_cross_fitted_nuisance(
        problem,
        certificate,
        causal.NuisancePlan(
            outcome_recipe=OLSRecipe(fit_intercept=False),
            split_plan=KFoldPlan(3, shuffle=False),
        ),
        key=jax.random.key(7),
    )
    overlap = causal.evaluate_overlap(
        problem,
        nuisance,
        causal.OverlapPolicy(minimum_effective_sample_size=20),
    )
    estimate = causal.estimate_aipw(problem, certificate, nuisance, overlap)

    assert bool(estimate.successful)
    assert float(estimate.effect) == pytest.approx(2.0, abs=1e-8)
    assert np.all(np.asarray(nuisance.fold_assignment) >= 0)

    failed_overlap = causal.evaluate_overlap(
        problem,
        nuisance,
        causal.OverlapPolicy(
            minimum_propensity=0.6,
            minimum_effective_sample_size=20,
        ),
    )
    assert failed_overlap.status == int(causal.OverlapStatus.MINIMUM_PROPENSITY)
    failed_estimate = causal.estimate_aipw(
        problem,
        certificate,
        nuisance,
        failed_overlap,
    )
    assert not bool(failed_estimate.successful)

    g_computation = causal.estimate_g_computation(
        problem,
        certificate,
        nuisance,
        overlap,
    )
    assessment = causal.assess_causal_estimate(
        g_computation,
        certificate,
        (causal.overlap_diagnostic(overlap),),
    )
    assert assessment.status is causal.CausalQualificationStatus.INCONCLUSIVE

    destination = tmp_path / "effect.phydrax"
    causal.export_causal_estimate(destination, estimate)
    restored = causal.read_causal_estimate(destination)
    assert restored.result_id == estimate.result_id
    assert float(restored.effect) == pytest.approx(float(estimate.effect))

    failed_destination = tmp_path / "failed-effect.phydrax"
    causal.export_causal_estimate(failed_destination, failed_estimate)
    restored_failure = causal.read_causal_estimate(failed_destination)
    assert restored_failure.result_id == failed_estimate.result_id
    assert not bool(restored_failure.successful)


def test_pc_stable_returns_equivalence_class():
    rng = np.random.default_rng(9)
    n = 1200
    x = rng.normal(size=n)
    y = 0.9 * x + rng.normal(scale=0.7, size=n)
    z = 0.9 * y + rng.normal(scale=0.7, size=n)
    schema = causal.CausalSchema(
        tuple(causal.CausalVariable(name=name) for name in ("x", "y", "z"))
    )
    dataset = causal.CausalDataset(
        schema=schema,
        values=(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)),
    )
    knowledge = causal.DiscoveryBackgroundKnowledge(schema=schema)
    result = causal.discover_pc_stable(
        dataset,
        causal.PCStablePlan(
            ci_test=causal.FisherZTest(alpha=0.001),
            knowledge=knowledge,
            resources=causal.DiscoveryResourcePolicy(
                maximum_conditioning_depth=1,
                maximum_ci_tests=100,
            ),
        ),
    )

    assert result.successful
    assert isinstance(result.graph, causal.CausalCPDAG)
    assert result.graph.undirected_edges == (("x", "y"), ("y", "z"))
    assert result.graph.directed_edges == ()

    true_graph = causal.CausalDAG(
        schema=schema,
        directed_edges=(("x", "y"), ("y", "z")),
    )
    wrong_graph = causal.CausalDAG(
        schema=schema,
        directed_edges=(("x", "y"), ("x", "z")),
    )
    assert (
        causal.falsify_dag_local_markov(
            dataset,
            true_graph,
            causal.FisherZTest(alpha=0.001),
        ).status
        is causal.GraphFalsificationStatus.NOT_REJECTED
    )
    assert (
        causal.falsify_dag_local_markov(
            dataset,
            wrong_graph,
            causal.FisherZTest(alpha=0.001),
        ).status
        is causal.GraphFalsificationStatus.REJECTED
    )
