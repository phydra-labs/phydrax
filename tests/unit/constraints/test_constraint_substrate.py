#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._term import AbstractScalarTerm, evaluate, TermEvaluation
from phydrax.integration import (
    IntegrationEstimate,
    IntegrationProvenance,
    IntegrationStatus,
)
from phydrax.sampling.collocation import R3
from phydrax.terms._integrated import checked_estimate_field


class _SignedTerm(AbstractScalarTerm):
    label: str | None

    def __init__(self, label=None):
        self.label = label

    def loss(self, functions, /, *, key=jr.key(0), iter_=None, **kwargs):
        del functions, key, iter_, kwargs
        return jnp.asarray(-2.0)


def _interval_problem():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="x")
    component = domain.component()
    field = domain.Function("x")(lambda x: x)
    target = phx.integration.over(component)
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8))
    source = phx.integration.per_step(target, plan)
    return domain, component, field, target, source


def test_scalar_term_evaluation_validates_all_loss_terms():
    evaluation = evaluate(_SignedTerm("signed"), {}, key=jr.key(0))

    assert isinstance(evaluation, TermEvaluation)
    assert evaluation.value.shape == ()
    assert jnp.allclose(evaluation.value, -2.0)

    with pytest.raises(ValueError, match="shape"):
        TermEvaluation(jnp.ones((2,)))
    with pytest.raises(TypeError, match="real value"):
        TermEvaluation(jnp.asarray(1.0 + 1.0j))


def test_residual_and_moment_penalties_have_distinct_ordering_semantics():
    _, component, field, _, source = _interval_problem()

    residual = phx.conditions.Residual("u", component, lambda u: u)
    penalty = phx.terms.ResidualPenalty(residual, source, scale=3.0)
    moment = phx.conditions.Moment("u", component, lambda u: u, target=0.5)
    conservation = phx.terms.MomentPenalty(moment, source, scale=2.0)

    assert jnp.allclose(penalty.loss({"u": field}, key=jr.key(1)), 1.0, atol=1e-12)
    assert jnp.allclose(
        conservation.loss({"u": field}, key=jr.key(2)), 0.0, atol=1e-12
    )

    nonzero_moment = phx.conditions.Moment(
        "u", component, lambda u: u, target=0.25
    )
    nonzero_penalty = phx.terms.MomentPenalty(nonzero_moment, source)
    assert jnp.allclose(
        nonzero_penalty.loss({"u": field}, key=jr.key(3)), 0.25**2, atol=1e-12
    )
def test_moment_penalty_rejects_solver_managed_adaptive_integration():
    _, component, _, target, _ = _interval_problem()
    condition = phx.conditions.Moment("u", component, lambda u: u, target=0.5)
    source = phx.integration.adaptive(
        target,
        phx.domain.PointSampling(
            16,
            layout=phx.domain.SampleLayout((("x",),)),
        ),
        R3(refresh_every=1, sampler="uniform"),
    )

    with pytest.raises(TypeError, match="requires ResidualPenalty"):
        phx.terms.MomentPenalty(condition, source)




def test_observation_penalty_uses_the_same_explicit_integration_source_contract():
    domain, component, field, _, source = _interval_problem()
    target = domain.Function("x")(lambda x: x)
    condition = phx.conditions.Observation("u", component, target)
    penalty = phx.terms.ObservationPenalty(condition, source)

    assert jnp.allclose(
        penalty.loss({"u": field}, key=jr.key(30)),
        0.0,
        atol=1e-12,
    )


def test_observation_penalty_realizes_finite_points_without_a_parallel_term_type():
    domain, component, field, _, _ = _interval_problem()
    batch = component.points({"x": jnp.array([0.25, 0.75])})
    target = domain.Function()(0.0)
    condition = phx.conditions.Observation("u", component, target)
    source = phx.integration.fixed(
        phx.integration.from_samples(
            phx.integration.mean_over(component),
            batch,
        )
    )

    penalty = phx.terms.ObservationPenalty(condition, source)

    assert jnp.allclose(penalty.loss({"u": field}), 0.3125, atol=1e-12)


def test_residual_density_multiplies_pointwise_score_without_renormalization():
    domain = phx.domain.ScalarInterval(0.0, 2.0, label="x")
    component = domain.component()
    field = domain.Function("x")(lambda x: x)
    density = domain.Function()(2.0)
    source = phx.integration.per_step(
        phx.integration.mean_over(component),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8)),
    )
    condition = phx.conditions.Residual("u", component, lambda u: u)
    penalty = phx.terms.ResidualPenalty(
        condition,
        source,
        density=density,
    )

    assert jnp.allclose(
        penalty.loss({"u": field}, key=jr.key(4)), 8.0 / 3.0, atol=1e-12
    )


def test_condition_and_integration_components_must_match_exactly():
    domain, component, _, _, _ = _interval_problem()
    boundary = domain.component({"x": phx.domain.FixedStart()})
    source = phx.integration.per_step(
        phx.integration.over(component),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4)),
    )

    with pytest.raises(ValueError, match="component are incompatible"):
        phx.terms.ResidualPenalty(
            phx.conditions.Dirichlet("u", boundary),
            source,
        )


def test_fixed_and_caller_sources_preserve_explicit_realization_ownership():
    _, component, field, target, _ = _interval_problem()
    plan = phx.integration.MonteCarloPlan(64)
    realization = phx.integration.materialize(target, plan, key=jr.key(10))
    condition = phx.conditions.Residual("u", component, lambda u: u)

    fixed = phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(realization),
    )
    first = fixed.loss({"u": field}, key=jr.key(11))
    second = fixed.loss({"u": field}, key=jr.key(12))
    assert jnp.array_equal(first, second)

    caller = phx.terms.ResidualPenalty(
        condition,
        phx.integration.caller(target),
    )
    with pytest.raises(ValueError, match="requires realization"):
        caller.loss({"u": field})
    assert jnp.array_equal(
        caller.loss({"u": field}, realization=realization),
        first,
    )

    incompatible = phx.integration.materialize(
        phx.integration.mean_over(component), plan, key=jr.key(13)
    )
    with pytest.raises(ValueError, match="incompatible target"):
        caller.loss({"u": field}, realization=incompatible)


def test_nonconverged_integration_estimate_is_never_silent():
    estimate = IntegrationEstimate(
        cx.Field(jnp.asarray(1.0), dims=()),
        status=IntegrationStatus.MAXIMUM_EVALUATIONS_REACHED,
        num_evaluations=1,
        error_estimate=None,
        error_kind=None,
        diagnostics=None,
        provenance=IntegrationProvenance("test", "test"),
    )

    with pytest.raises(Exception, match="did not converge"):
        checked_estimate_field(estimate)
