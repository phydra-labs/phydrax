#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.systems_biology import (
    bind_biological_evidence,
    BiologicalCondition,
    BiologicalFact,
    BiologicalReference,
    CountMeasurementPlan,
    ExchangeFieldSpec,
    MultirateScheduleEntry,
    PlanFieldAssertion,
    TelegraphFitTarget,
    TelegraphGeneExpressionPlan,
    WholeCellAssemblyPlan,
    WholeCellProcessBinding,
)
from phydrax.solver import solve_direct_ssa
from phydrax.stochastic import PoissonClockRealization


def test_gene_expression_evidence_inference_and_whole_cell_workflow():
    model = TelegraphGeneExpressionPlan(
        1.5,
        2.5,
        8.0,
        3.0,
        1.0,
        name="reporter-gene",
    ).prepare()
    initial = model.initial_state(promoter_on=False, nascent=0, mature=0)
    process = model.network.exact_jump_process()
    realization = PoissonClockRealization(
        jax.random.key(31),
        process.num_channels,
        support=(0.0, 8.0),
        max_events_per_channel=256,
        process_id=process.process_id,
        label="reporter-ssa",
    )
    solution = solve_direct_ssa(
        process,
        realization,
        initial,
        t0=0.0,
        t1=8.0,
        save_times=jnp.linspace(0.0, 8.0, 17),
        args=model.runtime(),
    )
    assert bool(jnp.all(solution.valid))
    final_count = solution.states[-1, 3]
    measurement = CountMeasurementPlan(1.0, 0.0, observation_capacity=256).prepare()
    likelihood = measurement.log_likelihood(final_count, final_count)
    assert bool(likelihood.valid)
    np.testing.assert_allclose(likelihood.log_likelihood, 0.0, atol=1.0e-6)

    stationary = model.stationary_moments()
    target = TelegraphFitTarget(
        stationary.fitting_vector,
        jnp.maximum(0.1 * stationary.fitting_vector, 0.1),
    )
    np.testing.assert_allclose(
        model.fitting_objective(jnp.log(model.rates), target), 0.0, atol=1.0e-10
    )

    reference = BiologicalReference("doi", "10.example/reporter", "methods:rates")
    fact = BiologicalFact("reporter", "transcription", 8.0, "s^-1", reference)
    condition = BiologicalCondition("culture", "medium", "defined")
    binding = bind_biological_evidence(
        model,
        (fact,),
        (condition,),
        (
            PlanFieldAssertion(
                fact.key,
                "telegraph.transcription_rate",
                condition_key=condition.key,
            ),
        ),
    )
    assert bool(binding.valid)
    assert binding.target_id == model.model_id

    fields = tuple(
        ExchangeFieldSpec(name)
        for name in ("promoter_off", "promoter_on", "nascent", "mature")
    )
    process_binding = WholeCellProcessBinding(
        "gene-expression",
        model.network,
        {name: name for name in ("promoter_off", "promoter_on", "nascent", "mature")},
    )
    assembly = WholeCellAssemblyPlan(
        "reporter-cell",
        fields,
        (process_binding,),
        (MultirateScheduleEntry("gene-expression", 8, require_regime_valid=False),),
        field_capacity=6,
        process_capacity=2,
    ).prepare()
    state = assembly.initial_state(solution.states[-1])
    before = assembly.checkpoint(state)
    evaluation = eqx.filter_jit(assembly.step)(state, jnp.asarray(0.01))
    assert not bool(evaluation.regime_valid)
    commit = evaluation.commit(state)
    assert bool(commit.committed)
    np.testing.assert_allclose(evaluation.conservation_residual, 0.0, atol=1.0e-6)
    after = assembly.checkpoint(commit.state)
    assert before.checkpoint_id != after.checkpoint_id
