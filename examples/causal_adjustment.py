"""Estimate an adjusted mean treatment effect with cross-fitted AIPW."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def main() -> None:
    rng = np.random.default_rng(17)
    samples = 400
    z = rng.integers(0, 2, size=samples)
    propensity = np.where(z == 0, 0.2, 0.8)
    treatment = rng.binomial(1, propensity)
    outcome = 1.5 * treatment + 0.5 * z + rng.normal(scale=0.2, size=samples)

    schema = phx.causal.CausalSchema(
        (
            phx.causal.CausalVariable(name="z", scale=phx.causal.VariableScale.BINARY),
            phx.causal.CausalVariable(name="t", scale=phx.causal.VariableScale.BINARY),
            phx.causal.CausalVariable(name="y"),
        )
    )
    dataset = phx.causal.CausalDataset(
        schema=schema,
        values=(jnp.asarray(z), jnp.asarray(treatment), jnp.asarray(outcome)),
    )
    assumptions = phx.causal.AssumptionLedger(
        phx.causal.CausalAssumption(
            kind=kind,
            disposition=phx.causal.AssumptionDisposition.DECLARED,
            statement=f"Declared {kind.value} for this synthetic analysis.",
        )
        for kind in (
            phx.causal.AssumptionKind.CONSISTENCY,
            phx.causal.AssumptionKind.POSITIVITY,
            phx.causal.AssumptionKind.NO_INTERFERENCE,
            phx.causal.AssumptionKind.CAUSAL_MARKOV,
        )
    )
    design = phx.causal.CausalStudyDesign(
        schema=schema,
        assignment_kind=phx.causal.AssignmentKind.OBSERVATIONAL,
        assignment_variable="t",
        exposure_variable="t",
        source_population_id="synthetic-population",
        assumptions=assumptions,
        no_interference=True,
        known_assignment_probability=jnp.asarray(propensity),
    )
    query = phx.causal.CausalQuery(
        schema=schema,
        outcome_variable="y",
        contrast=phx.causal.TreatmentContrast(
            active=phx.causal.TreatmentRegime(exposure_variable="t", value=1),
            reference=phx.causal.TreatmentRegime(exposure_variable="t", value=0),
        ),
        population=phx.causal.TargetPopulation(
            source_population_id="synthetic-population"
        ),
    )
    problem = phx.causal.CausalProblem(
        dataset=dataset,
        design=design,
        query=query,
    )
    graph = phx.causal.CausalDAG(
        schema=schema,
        directed_edges=(("z", "t"), ("z", "y"), ("t", "y")),
    )
    identified = phx.causal.identify_causal_effect(
        problem,
        graph,
        adjustment_set=("z",),
    )
    certificate = phx.causal.issue_identification_certificate(identified, problem)
    nuisance = phx.causal.fit_cross_fitted_nuisance(
        problem,
        certificate,
        phx.causal.NuisancePlan(
            outcome_recipe=phx.ml.linear.OLSRecipe(),
            split_plan=phx.ml.model_selection.KFoldPlan(5),
        ),
        key=jax.random.key(17),
    )
    overlap = phx.causal.evaluate_overlap(
        problem,
        nuisance,
        phx.causal.OverlapPolicy(minimum_effective_sample_size=20),
    )
    estimate = phx.causal.estimate_aipw(
        problem,
        certificate,
        nuisance,
        overlap,
    )
    diagnostic = phx.causal.overlap_diagnostic(overlap)
    assessment = phx.causal.assess_causal_estimate(
        estimate,
        certificate,
        (diagnostic,),
    )
    print(
        {
            "effect": float(estimate.effect),
            "standard_error": float(estimate.standard_error),
            "status": int(estimate.status),
            "assessment": assessment.status.value,
        }
    )


if __name__ == "__main__":
    main()
