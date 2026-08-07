#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import replace

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.design import AbstractDesignConstraint
from phydrax.geometry.design._search import _reflect_unit_box


class _ConstantConstraint(AbstractDesignConstraint):
    def residual(self, kernel, schema, state, /):
        del kernel, schema, state
        return self._weighted(jnp.ones((1,)))


class _InvalidConstraint(AbstractDesignConstraint):
    def residual(self, kernel, schema, state, /):
        del kernel, schema, state
        return self._weighted(jnp.full((1,), jnp.nan))


def _sphere_problem(*, constraint=None):
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), 1.0).compile()
    parameter_ids = {
        parameter_id.name: parameter_id for parameter_id in geometry.schema.parameter_ids
    }
    if constraint is None:
        constraint = phx.geometry.ParameterTarget(parameter_ids["radius"], 1.5)
    system = phx.geometry.DesignConstraintSystem(geometry, (constraint,))
    bounds = {
        parameter_ids["center"]: (-0.5, 0.5),
        parameter_ids["radius"]: (0.25, 2.5),
    }
    return geometry, system, parameter_ids, bounds


def test_configuration_validation():
    with pytest.raises(ValueError, match="at least 4"):
        phx.geometry.DifferentialEvolutionSearch(3, 4)
    with pytest.raises(ValueError, match="non-negative"):
        phx.geometry.DifferentialEvolutionSearch(4, -1)
    with pytest.raises(ValueError, match="strategy"):
        phx.geometry.DifferentialEvolutionSearch(4, 1, strategy="invalid")
    with pytest.raises(ValueError, match="differential_weight"):
        phx.geometry.DifferentialEvolutionSearch(4, 1, differential_weight=2.0)
    with pytest.raises(ValueError, match="crossover_rate"):
        phx.geometry.DifferentialEvolutionSearch(4, 1, crossover_rate=1.1)
    with pytest.raises(ValueError, match="relative_tolerance"):
        phx.geometry.DifferentialEvolutionSearch(4, 1, relative_tolerance=-1.0)
    with pytest.raises(ValueError, match="absolute_tolerance"):
        phx.geometry.DifferentialEvolutionSearch(4, 1, absolute_tolerance=jnp.nan)
    with pytest.raises(ValueError, match="design must be one of"):
        phx.geometry.DifferentialEvolutionSearch(4, 1, design="unknown")


def test_search_bounds_are_complete_finite_and_physically_admissible():
    geometry, system, parameter_ids, bounds = _sphere_problem()
    search = phx.geometry.DifferentialEvolutionSearch(4, 0)

    with pytest.raises(ValueError, match="center"):
        system.search(
            search,
            key=jr.key(0),
            bounds={parameter_ids["radius"]: (0.25, 2.5)},
        )
    with pytest.raises(KeyError, match="Unknown geometry parameter"):
        system.search(
            search,
            key=jr.key(0),
            bounds={
                **bounds,
                phx.geometry.ParameterId("unknown", "parameter"): (0.0, 1.0),
            },
        )
    with pytest.raises(ValueError, match="shape"):
        system.search(
            search,
            key=jr.key(0),
            bounds={**bounds, parameter_ids["center"]: (jnp.zeros((2,)), 0.5)},
        )
    with pytest.raises(ValueError, match="smaller"):
        system.search(
            search,
            key=jr.key(0),
            bounds={**bounds, parameter_ids["radius"]: (2.0, 1.0)},
        )
    with pytest.raises(ValueError, match="finite"):
        system.search(
            search,
            key=jr.key(0),
            bounds={**bounds, parameter_ids["radius"]: (0.25, jnp.inf)},
        )
    with pytest.raises(ValueError, match="physical lower"):
        system.search(
            search,
            key=jr.key(0),
            bounds={**bounds, parameter_ids["radius"]: (-0.25, 2.5)},
        )

    outside = geometry.state.updated({parameter_ids["radius"]: jnp.asarray(3.0)})
    with pytest.raises(ValueError, match="outside"):
        system.search(
            search,
            key=jr.key(0),
            bounds=bounds,
            initial_state=outside,
        )


def test_search_is_reproducible_bounded_and_exactly_accounted():
    geometry, system, _parameter_ids, bounds = _sphere_problem()
    search = phx.geometry.DifferentialEvolutionSearch(
        8,
        4,
        strategy="rand1bin",
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        design=phx.sampling.SobolDesign(scrambled=True),
    )

    result = system.search(search, key=jr.key(42), bounds=bounds)
    replay = system.search(search, key=jr.key(42), bounds=bounds)
    different = system.search(search, key=jr.key(43), bounds=bounds)

    assert result.generations == 4
    assert result.objective_evaluations == 8 * 5
    assert result.invalid_evaluations == 0
    assert result.termination_reason == "max_generations"
    assert result.best_objective_history.shape == (5,)
    assert np.all(np.diff(np.asarray(result.best_objective_history)) <= 0.0)
    assert np.all(
        np.asarray(result.population_vectors) >= np.asarray(result.lower_bounds)
    )
    assert np.all(
        np.asarray(result.population_vectors) <= np.asarray(result.upper_bounds)
    )
    assert float(result.objective) == pytest.approx(
        float(jnp.sum(result.residual * result.residual))
    )
    assert float(result.objective) == pytest.approx(
        float(jnp.min(result.population_objectives))
    )
    assert result.design_signature == phx.sampling.design_signature(search.design)

    np.testing.assert_array_equal(result.population_vectors, replay.population_vectors)
    np.testing.assert_array_equal(
        result.population_objectives, replay.population_objectives
    )
    np.testing.assert_array_equal(
        result.best_objective_history, replay.best_objective_history
    )
    for left, right in zip(result.state.values, replay.state.values, strict=True):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(result.population_vectors, different.population_vectors)

    local = system.solve(initial_state=result.state)
    assert bool(local.converged)
    assert float(jnp.sum(local.residual * local.residual)) <= float(result.objective)
    optimized = geometry.with_state(local.state)
    domain = phx.domain.GeometryDomain(optimized)
    assert domain.geometry.equivalent(optimized)


def test_initial_population_convergence_and_invalid_objectives_are_explicit():
    _geometry, constant_system, _parameter_ids, bounds = _sphere_problem(
        constraint=_ConstantConstraint()
    )
    converged = constant_system.search(
        phx.geometry.DifferentialEvolutionSearch(4, 10),
        key=jr.key(0),
        bounds=bounds,
    )
    assert converged.converged
    assert converged.generations == 0
    assert converged.objective_evaluations == 4
    assert converged.termination_reason == "initial_population_converged"
    np.testing.assert_array_equal(converged.best_objective_history, jnp.asarray([1.0]))

    _geometry, invalid_system, _parameter_ids, bounds = _sphere_problem(
        constraint=_InvalidConstraint()
    )
    invalid = invalid_system.search(
        phx.geometry.DifferentialEvolutionSearch(4, 10),
        key=jr.key(0),
        bounds=bounds,
    )
    assert not invalid.converged
    assert invalid.generations == 0
    assert invalid.objective_evaluations == 4
    assert invalid.invalid_evaluations == 4
    assert invalid.termination_reason == "no_finite_candidates"
    assert np.all(np.isinf(np.asarray(invalid.population_objectives)))
    assert bool(jnp.isnan(invalid.objective))


def test_repair_reflects_arbitrary_overshoot_into_unit_box():
    candidates = jnp.asarray([-4.2, -1.2, -0.2, 0.2, 1.2, 2.2, 5.2])
    repaired = _reflect_unit_box(candidates)
    np.testing.assert_allclose(
        repaired,
        jnp.asarray([0.2, 0.8, 0.2, 0.2, 0.8, 0.2, 0.8]),
        atol=1e-12,
    )
    assert np.all(np.asarray(repaired) >= 0.0)
    assert np.all(np.asarray(repaired) <= 1.0)


def test_restricted_geometry_validity_region_fails_before_evaluation(monkeypatch):
    geometry, system, _parameter_ids, bounds = _sphere_problem()
    restricted = replace(geometry.field_certificate, validity_region="fixed_topology")
    monkeypatch.setattr(
        type(geometry.kernel),
        "field_certificate",
        property(lambda _self: restricted),
    )

    with pytest.raises(NotImplementedError, match="validity region"):
        system.search(
            phx.geometry.DifferentialEvolutionSearch(4, 1),
            key=jr.key(0),
            bounds=bounds,
        )
