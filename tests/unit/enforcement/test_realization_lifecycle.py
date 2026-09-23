import jax.numpy as jnp

import phydrax as phx
from phydrax.conditions import (
    ArrayCodomain,
    Condition,
    ConditionQuantifier,
    Equality,
    FieldSpec,
    MatrixLinearFunctional,
    ProductFieldSpec,
)
from phydrax.domain import Boundary, Interval1d
from phydrax.enforcement import (
    CallerRealizationSource,
    commit_enforcement_step,
    commit_refresh,
    ConditionEvaluationContext,
    EnforcementProgram,
    EnforcementSpec,
    EnforcementState,
    FixedRealizationSource,
    propose_refresh,
    RealizationLifecyclePhase,
    RealizationStatus,
    validate_refresh,
)


def _condition():
    codomain = ArrayCodomain.from_shape((1,), dtype="float64")
    fields = ProductFieldSpec((FieldSpec("u", codomain),))
    return Condition(
        "lifecycle-test",
        fields,
        MatrixLinearFunctional(("u",), ((1,),), (jnp.ones((1, 1)),)),
        codomain,
        Equality(jnp.zeros((1,))),
        quantifier=ConditionQuantifier.deterministic,
    )


def test_refresh_transaction_commits_atomically_and_reuses_fixed_state():
    condition = _condition()
    context = ConditionEvaluationContext(
        condition,
        caller_sources={"caller": jnp.asarray(2.0)},
    )
    sources = (
        FixedRealizationSource("fixed", jnp.asarray(1.0)),
        CallerRealizationSource("caller"),
    )
    proposal = propose_refresh(sources, None, context=context)
    state = commit_refresh(None, proposal, validate_refresh(proposal))
    assert state.phase is RealizationLifecyclePhase.READY
    assert jnp.asarray(state.values["fixed"]) == 1.0
    assert jnp.asarray(state.values["caller"]) == 2.0
    unchanged = propose_refresh(
        sources, state, context=ConditionEvaluationContext(condition)
    )
    repeated = commit_refresh(state, unchanged, validate_refresh(unchanged))
    assert repeated.generation == state.generation


def test_missing_required_caller_source_fails_without_candidate_values():
    condition = _condition()
    context = ConditionEvaluationContext(condition)
    proposal = propose_refresh(
        (CallerRealizationSource("missing"),), None, context=context
    )
    validation = validate_refresh(proposal)
    assert validation.status is RealizationStatus.SOURCE_UNAVAILABLE
    state = commit_refresh(None, proposal, validation)
    assert state.phase is RealizationLifecyclePhase.FAILED
    assert not state.values


def test_local_enforcement_step_is_a_changed_transaction():
    domain = Interval1d(0.0, 1.0)
    field = domain.Function("x")(lambda x: x[0])
    boundary = domain.component({"x": Boundary()})
    program = EnforcementProgram.build(
        functions={"u": field},
        specs=(EnforcementSpec(phx.conditions.Dirichlet("u", boundary, target=1.0)),),
        num_reference=16,
    )
    state = EnforcementState({"u": field})

    prepared = program.prepare_step({"u": field}, state=state)
    committed = commit_enforcement_step(state, prepared)

    assert prepared.status is RealizationStatus.SUCCESS
    assert committed.generation == state.generation + 1
