#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..._strict import StrictModule
from ...optim import DifferentialEvolutionSearch
from ...typing import checked, PRNGKey
from ._schema import DesignState


class DesignSearchResult(StrictModule):
    """Return the best design state and bounded global-search evidence."""

    __strict_contract__ = True

    state: DesignState
    residual: Array
    residual_norm: Array
    objective: Array
    population_vectors: Array
    population_objectives: Array
    best_objective_history: Array
    lower_bounds: Array
    upper_bounds: Array
    key: PRNGKey
    search: DifferentialEvolutionSearch
    converged: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    generations: int = eqx.field(static=True)
    objective_evaluations: int = eqx.field(static=True)
    invalid_evaluations: int = eqx.field(static=True)
    design_signature: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        state: DesignState,
        residual: Array,
        objective: Array,
        population_vectors: Array,
        population_objectives: Array,
        best_objective_history: Array,
        lower_bounds: Array,
        upper_bounds: Array,
        key: PRNGKey,
        search: DifferentialEvolutionSearch,
        converged: bool,
        termination_reason: str,
        generations: int,
        objective_evaluations: int,
        invalid_evaluations: int,
        design_signature: str,
    ) -> None:
        residual_ = jnp.asarray(residual, dtype=jnp.float64).reshape((-1,))
        self.state = state
        self.residual = residual_
        self.residual_norm = jnp.linalg.norm(residual_)
        self.objective = jnp.asarray(objective, dtype=jnp.float64).reshape(())
        self.population_vectors = jnp.asarray(population_vectors, dtype=jnp.float64)
        self.population_objectives = jnp.asarray(population_objectives, dtype=jnp.float64)
        self.best_objective_history = jnp.asarray(
            best_objective_history, dtype=jnp.float64
        )
        self.lower_bounds = jnp.asarray(lower_bounds, dtype=jnp.float64)
        self.upper_bounds = jnp.asarray(upper_bounds, dtype=jnp.float64)
        self.key = key
        self.search = search
        self.converged = bool(converged)
        self.termination_reason = str(termination_reason)
        self.generations = int(generations)
        self.objective_evaluations = int(objective_evaluations)
        self.invalid_evaluations = int(invalid_evaluations)
        self.design_signature = str(design_signature)


__all__ = ["DesignSearchResult"]
