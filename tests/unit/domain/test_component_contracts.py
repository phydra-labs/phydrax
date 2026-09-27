#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.domain import (
    Boundary,
    ComponentSum,
    DatasetDomain,
    ExactMass,
    Fixed,
    FixedStart,
    ScalarInterval,
    TimeInterval,
    UnknownMass,
)


def test_component_contracts_scenario_1() -> None:
    domain = ScalarInterval(-2.0, 3.0, label="x")

    interior = domain.component()
    boundary = domain.component({"x": Boundary()})
    fixed = domain.component({"x": Fixed(0.25)})

    assert len(interior.factor_components) == 1
    assert interior.factor_components[0].factor is domain
    assert isinstance(interior.mass, ExactMass)
    assert jnp.isclose(interior.mass.value, 5.0)
    # ty: ignore[unresolved-attribute]
    assert jnp.isclose(boundary.mass.value, 2.0)
    # ty: ignore[unresolved-attribute]
    assert jnp.isclose(fixed.mass.value, 1.0)
    data = jnp.arange(12.0).reshape((4, 3))
    probability = DatasetDomain(data, measure="probability").component()
    counting = DatasetDomain(data, measure="count").component()

    assert probability.factor_components[0].measure.kind == "probability"
    assert probability.factor_components[0].measure.normalized
    # ty: ignore[unresolved-attribute]
    assert jnp.isclose(probability.mass.value, 1.0)
    assert counting.factor_components[0].measure.kind == "counting"
    # ty: ignore[unresolved-attribute]
    assert jnp.isclose(counting.mass.value, 4.0)
    domain = ScalarInterval(0.0, 2.0, label="x")

    restricted = domain.component().restrict(per_coordinate={"x": lambda x: x < 1.0})
    unnormalized = domain.component().with_density(lambda x: 2.0 * x)
    normalized = domain.component().with_density(lambda x: 0.5, normalized=True)

    assert isinstance(restricted.mass, UnknownMass)
    assert isinstance(unnormalized.mass, UnknownMass)
    assert isinstance(normalized.mass, ExactMass)
    assert jnp.isclose(normalized.mass.value, 1.0)
    # ty: ignore[unresolved-attribute]
    assert jnp.isclose(restricted.base_measure.mass.value, 2.0)

    field = domain.Function("x")(lambda x: jnp.ones_like(x))
    estimate = phx.integration.integrate(
        field,
        phx.integration.over(unnormalized),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussLegendreRule(8)),
    )
    assert jnp.allclose(jnp.asarray(estimate.value.data), 4.0)


def test_component_contracts_scenario_2() -> None:
    domain = ScalarInterval(0.0, 1.0, label="x")
    component = domain.component()
    restricted = component.restrict(per_coordinate={"x": lambda x: x < 0.5})

    with pytest.raises(ValueError, match="duplicates"):
        ComponentSum((component, component))
    with pytest.raises(ValueError, match="assume_disjoint"):
        ComponentSum((restricted, component))
    space = ScalarInterval(-1.0, 1.0, label="x")
    time = TimeInterval(2.0, 3.0)
    component = (space @ time).component({"t": FixedStart()})

    mapped = component.points({"x": jnp.array([-0.5, 0.75])})
    stacked = component.points(jnp.array([[-0.5], [0.75]]))

    assert mapped.structure == stacked.structure
    # ty: ignore[not-subscriptable]
    assert mapped["x"].dims == (mapped.structure.axis_names[0],)
    assert mapped["t"].dims == ()
    assert jnp.array_equal(jnp.asarray(mapped["x"].data), stacked["x"].data)
    assert jnp.array_equal(jnp.asarray(mapped["t"].data), jnp.asarray(2.0))
    x = ScalarInterval(0.0, 1.0, label="x")
    y = ScalarInterval(0.0, 1.0, label="y")

    with pytest.raises(ValueError, match="same leading point count"):
        (x @ y).component().points({"x": jnp.array([0.0, 1.0]), "y": jnp.array([0.0])})
    x = ScalarInterval(0.0, 2.0, label="x")
    t = ScalarInterval(-1.0, 3.0, label="t")
    # ty: ignore[unresolved-attribute]
    boundary = (x @ t).boundary()

    assert isinstance(boundary, ComponentSum)
    assert len(boundary.terms) == 4
    assert isinstance(boundary.mass, ExactMass)
    assert jnp.isclose(boundary.mass.value, 2.0 * (2.0 + 4.0))
