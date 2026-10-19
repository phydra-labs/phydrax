#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import replace

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.geometry._contracts import GeometryKernel
from phydrax.geometry.design import (
    AbstractDesignConstraint,
    DesignState,
    ParameterId,
    ParameterSchema,
)
from tests._support.design_constraints import sphere_constraint_problem


class _ConstantConstraint(AbstractDesignConstraint):
    def residual(
        self, kernel: GeometryKernel, schema: ParameterSchema, state: DesignState, /
    ) -> Array:
        del kernel, schema, state
        return self._weighted(jnp.ones((1,), dtype=jnp.float64))


class _InvalidConstraint(AbstractDesignConstraint):
    def residual(
        self, kernel: GeometryKernel, schema: ParameterSchema, state: DesignState, /
    ) -> Array:
        del kernel, schema, state
        return self._weighted(jnp.full((1,), jnp.nan, dtype=jnp.float64))


def _search() -> phx.optim.DifferentialEvolutionSearch:
    return phx.optim.DifferentialEvolutionSearch(
        8,
        4,
        strategy="rand1bin",
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        design=phx.sampling.SobolDesign(scrambled=True),
    )


@pytest.mark.parametrize(
    ("case", "error", "field"),
    [
        ("missing", ValueError, "center"),
        ("unknown", KeyError, "DesignSearch.bounds.parameter_id"),
        ("shape", ValueError, "shape"),
        ("reversed", ValueError, "smaller"),
        ("nonfinite", ValueError, "finite"),
        ("physical-lower", ValueError, "physical lower"),
    ],
    ids=["missing", "unknown", "shape", "reversed", "nonfinite", "physical-lower"],
)
def test_search_bounds_refuse_invalid_declarations(
    case: str, error: type[Exception], field: str
) -> None:
    _, system, parameter_ids, bounds = sphere_constraint_problem()
    selected: dict[ParameterId, tuple[ArrayLike, ArrayLike]]
    match case:
        case "missing":
            selected = {parameter_ids["radius"]: (0.25, 2.5)}
        case "unknown":
            selected = {**bounds, ParameterId("unknown", "parameter"): (0.0, 1.0)}
        case "shape":
            selected = {
                **bounds,
                parameter_ids["center"]: (jnp.zeros((2,), dtype=jnp.float64), 0.5),
            }
        case "reversed":
            selected = {**bounds, parameter_ids["radius"]: (2.0, 1.0)}
        case "nonfinite":
            selected = {**bounds, parameter_ids["radius"]: (0.25, jnp.inf)}
        case "physical-lower":
            selected = {**bounds, parameter_ids["radius"]: (-0.25, 2.5)}
        case _:
            raise AssertionError(case)
    with pytest.raises(error, match=field):
        system.search(
            phx.optim.DifferentialEvolutionSearch(4, 0), key=jr.key(0), bounds=selected
        )


def test_search_initial_state_must_lie_in_bounds() -> None:
    geometry, system, parameter_ids, bounds = sphere_constraint_problem()
    outside = geometry.state.updated(
        {parameter_ids["radius"]: jnp.asarray(3.0, dtype=jnp.float64)}
    )
    with pytest.raises(ValueError, match="outside"):
        system.search(
            phx.optim.DifferentialEvolutionSearch(4, 0),
            key=jr.key(0),
            bounds=bounds,
            initial_state=outside,
        )


def test_global_search_preserves_work_population_and_rng_evidence() -> None:
    _, system, _, bounds = sphere_constraint_problem()
    search = _search()
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


def test_global_search_state_supports_local_refinement() -> None:
    geometry, system, _, bounds = sphere_constraint_problem()
    result = system.search(_search(), key=jr.key(42), bounds=bounds)
    local = system.solve(initial_state=result.state)
    assert bool(local.converged)
    assert float(jnp.sum(local.residual * local.residual)) <= float(result.objective)
    optimized = geometry.with_state(local.state)
    domain = phx.domain.GeometryDomain(optimized)
    assert domain.geometry.equivalent(optimized)


def test_constant_residual_reports_global_population_convergence() -> None:
    _, system, _, bounds = sphere_constraint_problem(constraint=_ConstantConstraint())
    converged = system.search(
        phx.optim.DifferentialEvolutionSearch(4, 10), key=jr.key(0), bounds=bounds
    )
    assert converged.converged
    assert converged.generations == 0
    assert converged.objective_evaluations == 4
    assert converged.termination_reason == "initial_population_converged"
    np.testing.assert_array_equal(
        converged.best_objective_history, jnp.asarray([1.0], dtype=jnp.float64)
    )


def test_invalid_residual_reports_no_valid_candidates() -> None:
    _, system, _, bounds = sphere_constraint_problem(constraint=_InvalidConstraint())
    invalid = system.search(
        phx.optim.DifferentialEvolutionSearch(4, 10), key=jr.key(0), bounds=bounds
    )
    assert not invalid.converged
    assert invalid.generations == 0
    assert invalid.objective_evaluations == 4
    assert invalid.invalid_evaluations == 4
    assert invalid.termination_reason == "no_valid_candidates"
    assert np.all(np.isinf(np.asarray(invalid.population_objectives)))
    assert bool(jnp.isnan(invalid.objective))


def test_restricted_geometry_validity_region_fails_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, system, _, bounds = sphere_constraint_problem()
    restricted = replace(geometry.field_certificate, validity_region="fixed_topology")
    monkeypatch.setattr(
        type(geometry.kernel), "field_certificate", property(lambda _self: restricted)
    )
    with pytest.raises(NotImplementedError, match="executable geometry validity"):
        system.search(
            phx.optim.DifferentialEvolutionSearch(4, 1), key=jr.key(0), bounds=bounds
        )
