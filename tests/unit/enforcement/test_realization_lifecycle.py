import jax.numpy as jnp

from phydrax.conditions import (
    ArrayCodomain,
    Condition,
    ConditionQuantifier,
    Equality,
    FieldSpec,
    MatrixLinearFunctional,
    ProductFieldSpec,
)
from phydrax.enforcement import (
    CallerRealizationSource,
    commit_refresh,
    ConditionEvaluationContext,
    FixedRealizationSource,
    propose_refresh,
    RealizationLifecyclePhase,
    RealizationStatus,
    validate_refresh,
)


def _condition():
    codomain = ArrayCodomain.from_shape((1,), dtype=float)
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
