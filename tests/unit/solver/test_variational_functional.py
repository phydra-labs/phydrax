#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _quadratic_functional():
    def density(fields, geometry, context):
        del geometry, context
        return 0.5 * fields["u"].value ** 2

    return phx.variational.Functional(
        "quadratic",
        (
            phx.variational.LocalIntegralTerm(
                "body",
                region="body",
                fields=(phx.variational.FieldJetSpec("u", value=True),),
                density=density,
                density_id="quadratic-value",
            ),
        ),
        variable_fields=("u",),
    )


def test_domain_function_binding_executes_portable_functional():
    domain = phx.domain.Interval1d(0.0, 1.0)
    coordinate = domain.Function("x")(lambda x: x[0])
    field = domain.Parameter(2.0) * coordinate
    target = phx.integration.over(domain.component())
    plan = phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(12))
    source = phx.integration.fixed(phx.integration.materialize(target, plan))

    terms = phx.terms.bind_functional(
        _quadratic_functional(),
        {"u": field},
        {"body": source},
        geometry_variables={"body": "x"},
    )
    solver = phx.solver.FunctionalSolver(functions={"u": field}, terms=terms)

    assert len(terms) == 1
    assert terms[0].objective_vars == ("u",)
    assert jnp.allclose(solver.loss(key=jr.key(0)), 2.0 / 3.0, atol=1.0e-12)


def test_domain_function_binding_validates_field_and_region_maps():
    domain = phx.domain.Interval1d(0.0, 1.0)
    field = domain.Function("x")(lambda x: x[0])
    target = phx.integration.over(domain.component())
    source = phx.integration.per_step(
        target,
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(4)),
    )

    functional = _quadratic_functional()
    with pytest.raises(KeyError, match="field bindings"):
        phx.terms.bind_functional(
            functional,
            {"v": field},
            {"body": source},
            geometry_variables={"body": "x"},
        )
