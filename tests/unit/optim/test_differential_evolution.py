#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.optim._differential_evolution import (
    _bounded_differential_evolution,
    _reflect_unit_box,
)


def test_configuration_validation():
    with pytest.raises(ValueError, match="at least 4"):
        phx.optim.DifferentialEvolutionSearch(3, 4)
    with pytest.raises(ValueError, match="non-negative"):
        phx.optim.DifferentialEvolutionSearch(4, -1)
    with pytest.raises(ValueError, match="strategy"):
        phx.optim.DifferentialEvolutionSearch(4, 1, strategy="invalid")
    with pytest.raises(ValueError, match="differential_weight"):
        phx.optim.DifferentialEvolutionSearch(4, 1, differential_weight=2.0)
    with pytest.raises(ValueError, match="crossover_rate"):
        phx.optim.DifferentialEvolutionSearch(4, 1, crossover_rate=1.1)
    with pytest.raises(ValueError, match="relative_tolerance"):
        phx.optim.DifferentialEvolutionSearch(4, 1, relative_tolerance=-1.0)
    with pytest.raises(ValueError, match="absolute_tolerance"):
        phx.optim.DifferentialEvolutionSearch(4, 1, absolute_tolerance=jnp.nan)
    with pytest.raises(ValueError, match="design must be one of"):
        phx.optim.DifferentialEvolutionSearch(4, 1, design="unknown")


def test_vector_and_bound_validation():
    search = phx.optim.DifferentialEvolutionSearch(4, 0)
    objective = lambda vector: jnp.sum(vector * vector)

    with pytest.raises(ValueError, match="one-dimensional"):
        _bounded_differential_evolution(
            objective,
            jnp.zeros((1, 1)),
            jnp.zeros((1,)),
            jnp.ones((1,)),
            search,
            key=jr.key(0),
        )
    with pytest.raises(ValueError, match="at least one dimension"):
        _bounded_differential_evolution(
            objective,
            jnp.zeros((0,)),
            jnp.zeros((0,)),
            jnp.ones((0,)),
            search,
            key=jr.key(0),
        )
    with pytest.raises(ValueError, match="identical shapes"):
        _bounded_differential_evolution(
            objective,
            jnp.zeros((2,)),
            jnp.zeros((1,)),
            jnp.ones((2,)),
            search,
            key=jr.key(0),
        )
    with pytest.raises(ValueError, match="finite"):
        _bounded_differential_evolution(
            objective,
            jnp.zeros((1,)),
            jnp.asarray([jnp.nan]),
            jnp.ones((1,)),
            search,
            key=jr.key(0),
        )
    with pytest.raises(ValueError, match="smaller"):
        _bounded_differential_evolution(
            objective,
            jnp.zeros((1,)),
            jnp.ones((1,)),
            jnp.zeros((1,)),
            search,
            key=jr.key(0),
        )
    with pytest.raises(ValueError, match="outside"):
        _bounded_differential_evolution(
            objective,
            jnp.asarray([2.0]),
            jnp.zeros((1,)),
            jnp.ones((1,)),
            search,
            key=jr.key(0),
        )
    with pytest.raises(TypeError, match="real-valued"):
        _bounded_differential_evolution(
            objective,
            jnp.asarray([0.0 + 0.0j]),
            jnp.asarray([-1.0 + 0.0j]),
            jnp.asarray([1.0 + 0.0j]),
            search,
            key=jr.key(0),
        )


@pytest.mark.parametrize("strategy", ["best1bin", "rand1bin"])
def test_search_is_reproducible_bounded_and_exactly_accounted(strategy):
    search = phx.optim.DifferentialEvolutionSearch(
        8,
        3,
        strategy=strategy,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
        design=phx.sampling.SobolDesign(scrambled=True),
    )
    objective = lambda vector: jnp.sum((vector - 0.25) ** 2)
    arguments = (
        objective,
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([-1.0, -2.0]),
        jnp.asarray([1.0, 2.0]),
        search,
    )

    result = _bounded_differential_evolution(*arguments, key=jr.key(42))
    replay = _bounded_differential_evolution(*arguments, key=jr.key(42))
    different = _bounded_differential_evolution(*arguments, key=jr.key(43))

    assert result.generations == 3
    assert result.objective_evaluations == 8 * 4
    assert result.invalid_evaluations == 0
    assert result.termination_reason == "max_generations"
    assert result.population_vectors.shape == (8, 2)
    assert result.population_objectives.shape == (8,)
    assert result.best_objective_history.shape == (4,)
    assert np.all(np.diff(np.asarray(result.best_objective_history)) <= 0.0)
    assert np.all(np.asarray(result.population_vectors) >= [-1.0, -2.0])
    assert np.all(np.asarray(result.population_vectors) <= [1.0, 2.0])
    np.testing.assert_array_equal(result.population_vectors, replay.population_vectors)
    np.testing.assert_array_equal(
        result.population_objectives,
        replay.population_objectives,
    )
    assert not np.array_equal(
        np.asarray(result.population_vectors),
        np.asarray(different.population_vectors),
    )


def test_initial_convergence_and_invalid_objectives_are_explicit():
    search = phx.optim.DifferentialEvolutionSearch(4, 10)
    bounds = (jnp.asarray([-1.0]), jnp.asarray([1.0]))

    converged = _bounded_differential_evolution(
        lambda vector: jnp.asarray(1.0),
        jnp.asarray([0.0]),
        *bounds,
        search,
        key=jr.key(0),
    )
    assert converged.converged
    assert converged.generations == 0
    assert converged.objective_evaluations == 4
    assert converged.termination_reason == "initial_population_converged"

    invalid = _bounded_differential_evolution(
        lambda vector: jnp.asarray(jnp.nan),
        jnp.asarray([0.0]),
        *bounds,
        search,
        key=jr.key(0),
    )
    assert not invalid.converged
    assert invalid.generations == 0
    assert invalid.invalid_evaluations == 4
    assert invalid.termination_reason == "no_finite_candidates"
    assert np.all(np.isinf(np.asarray(invalid.population_objectives)))


def test_repair_reflects_arbitrary_overshoot_into_unit_box():
    candidates = jnp.asarray([-4.2, -1.2, -0.2, 0.0, 0.2, 1.0, 1.2, 2.2, 5.2])
    repaired = _reflect_unit_box(candidates)
    np.testing.assert_allclose(
        repaired,
        jnp.asarray([0.2, 0.8, 0.2, 0.0, 0.2, 1.0, 0.8, 0.2, 0.8]),
        atol=1e-12,
    )
    assert np.all(np.asarray(repaired) >= 0.0)
    assert np.all(np.asarray(repaired) <= 1.0)
