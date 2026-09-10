#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def test_causal_interventions_use_native_uq_batch_selection():
    schema = phx.causal.CausalSchema(
        (phx.causal.CausalVariable(name="action", scale=phx.causal.VariableScale.BINARY),)
    )
    scm = phx.causal.SCMPlan(
        graph=phx.causal.CausalDAG(schema=schema, directed_edges=()),
        exogenous_independence=True,
        mechanisms=(
            phx.causal.FiniteConditionalMechanism(
                output="action",
                parents=(),
                probabilities=jnp.asarray([0.5, 0.5]),
            ),
        ),
    )
    candidates = tuple(
        phx.causal.CausalInterventionCandidate(
            regime=phx.causal.build_intervention_regime(
                scm,
                (phx.causal.PerfectIntervention(variable="action", value=value),),
            ),
            observation_plan_id="observe-action",
            cost=1.0,
            feasibility_group="binary-action",
            prediction_source_id="finite-scm",
        )
        for value in (0, 1)
    )
    utility = phx.uq.ExpectedUtilityResult(
        expected_utility=jnp.asarray([0.1, 0.8]),
        estimator_standard_error=jnp.zeros((2,)),
        estimator_bias_bound=jnp.zeros((2,)),
        valid=jnp.ones((2,), dtype=bool),
        candidates=tuple(candidate.candidate for candidate in candidates),
        model_ids=("finite-scm",),
        utility_target="parameter",
        method_id="exact-fixture",
        approximation="exact",
        error_basis="none",
    )
    constraints = phx.uq.ExperimentalBatchConstraints(1.0, 1)

    plan = phx.causal.select_causal_intervention_batch(
        candidates,
        utility,
        constraints,
        objective_id="action-information",
        model_ids=("finite-scm",),
        analysis_id="causal-design-fixture",
    )

    assert plan.selected_candidate_ids == (candidates[1].candidate_id,)
    assert plan.candidate_content_ids == tuple(
        candidate.candidate.candidate_content_id for candidate in candidates
    )
