#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

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


def test_factor_components_bind_exact_scalar_measures():
    domain = ScalarInterval(-2.0, 3.0, label="x")

    interior = domain.component()
    boundary = domain.component({"x": Boundary()})
    fixed = domain.component({"x": Fixed(0.25)})

    assert len(interior.factor_components) == 1
    assert interior.factor_components[0].factor is domain
    assert isinstance(interior.mass, ExactMass)
    assert jnp.isclose(interior.mass.value, 5.0)
    assert jnp.isclose(boundary.mass.value, 2.0)
    assert jnp.isclose(fixed.mass.value, 1.0)


def test_probability_and_count_dataset_measures_are_explicit():
    data = jnp.arange(12.0).reshape((4, 3))
    probability = DatasetDomain(data, measure="probability").component()
    counting = DatasetDomain(data, measure="count").component()

    assert probability.factor_components[0].measure.kind == "probability"
    assert probability.factor_components[0].measure.normalized
    assert jnp.isclose(probability.mass.value, 1.0)
    assert counting.factor_components[0].measure.kind == "counting"
    assert jnp.isclose(counting.mass.value, 4.0)


def test_restriction_and_density_have_typed_mass_semantics():
    domain = ScalarInterval(0.0, 2.0, label="x")

    restricted = domain.component().restrict(
        per_coordinate={"x": lambda x: x < 1.0}
    )
    unnormalized = domain.component().with_density(lambda x: 2.0 * x)
    normalized = domain.component().with_density(lambda x: 0.5, normalized=True)

    assert isinstance(restricted.mass, UnknownMass)
    assert isinstance(unnormalized.mass, UnknownMass)
    assert isinstance(normalized.mass, ExactMass)
    assert jnp.isclose(normalized.mass.value, 1.0)
    assert jnp.isclose(restricted.base_measure.mass.value, 2.0)


def test_component_sum_rejects_duplicates_and_uncertified_predicate_overlap():
    domain = ScalarInterval(0.0, 1.0, label="x")
    component = domain.component()
    restricted = component.restrict(per_coordinate={"x": lambda x: x < 0.5})

    with pytest.raises(ValueError, match="duplicates"):
        ComponentSum((component, component))
    with pytest.raises(ValueError, match="assume_disjoint"):
        ComponentSum((restricted, component))


def test_product_boundary_mass_is_additive_over_codimension_one_terms():
    x = ScalarInterval(0.0, 2.0, label="x")
    t = ScalarInterval(-1.0, 3.0, label="t")
    boundary = (x @ t).boundary()

    assert isinstance(boundary, ComponentSum)
    assert len(boundary.terms) == 4
    assert isinstance(boundary.mass, ExactMass)
    assert jnp.isclose(boundary.mass.value, 2.0 * (2.0 + 4.0))


def test_component_points_binds_explicit_coordinates_and_fixed_slices():
    space = ScalarInterval(-1.0, 1.0, label="x")
    time = TimeInterval(2.0, 3.0)
    component = (space @ time).component({"t": FixedStart()})

    mapped = component.points({"x": jnp.array([-0.5, 0.75])})
    stacked = component.points(jnp.array([[-0.5], [0.75]]))

    assert mapped.structure == stacked.structure
    assert mapped["x"].dims == (mapped.structure.axis_names[0],)
    assert mapped["t"].dims == ()
    assert jnp.array_equal(mapped["x"].data, stacked["x"].data)
    assert jnp.array_equal(mapped["t"].data, jnp.asarray(2.0))


def test_component_points_rejects_inconsistent_coordinate_counts():
    x = ScalarInterval(0.0, 1.0, label="x")
    y = ScalarInterval(0.0, 1.0, label="y")

    with pytest.raises(ValueError, match="same leading point count"):
        (x @ y).component().points(
            {"x": jnp.array([0.0, 1.0]), "y": jnp.array([0.0])}
        )
