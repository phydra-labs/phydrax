#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...optim import DifferentialEvolutionSearch
from ...typing import PRNGKey
from ._sketch import SketchSolution


class SketchSearchResult(StrictModule):
    """Global search evidence kept distinct from the final local sketch root."""

    __strict_contract__ = True

    solution: SketchSolution | None
    population_vectors: Array
    population_objectives: Array
    best_objective_history: Array
    lower_bounds: Array
    upper_bounds: Array
    key: PRNGKey
    search: DifferentialEvolutionSearch
    search_converged: bool = eqx.field(static=True)
    root_converged: bool = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    generations: int = eqx.field(static=True)
    objective_evaluations: int = eqx.field(static=True)
    invalid_evaluations: int = eqx.field(static=True)
    design_signature: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        solution: SketchSolution | None,
        population_vectors: ArrayLike,
        population_objectives: ArrayLike,
        best_objective_history: ArrayLike,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        key: PRNGKey,
        search: DifferentialEvolutionSearch,
        search_converged: bool,
        termination_reason: str,
        generations: int,
        objective_evaluations: int,
        invalid_evaluations: int,
        design_signature: str,
    ) -> None:
        if solution is not None and not isinstance(solution, SketchSolution):
            raise TypeError("solution must be SketchSolution or None.")
        if not isinstance(search, DifferentialEvolutionSearch):
            raise TypeError("search must be DifferentialEvolutionSearch.")
        root_converged = (
            False if solution is None else bool(np.asarray(solution.converged))
        )
        self.solution = solution
        self.population_vectors = jnp.asarray(population_vectors, dtype=jnp.float64)
        self.population_objectives = jnp.asarray(population_objectives, dtype=jnp.float64)
        self.best_objective_history = jnp.asarray(
            best_objective_history, dtype=jnp.float64
        )
        self.lower_bounds = jnp.asarray(lower_bounds, dtype=jnp.float64)
        self.upper_bounds = jnp.asarray(upper_bounds, dtype=jnp.float64)
        self.key = key
        self.search = search
        self.search_converged = bool(search_converged)
        self.root_converged = root_converged
        self.converged = self.search_converged and root_converged
        self.termination_reason = str(termination_reason)
        self.generations = int(generations)
        self.objective_evaluations = int(objective_evaluations)
        self.invalid_evaluations = int(invalid_evaluations)
        self.design_signature = str(design_signature)


__all__ = ["SketchSearchResult"]
