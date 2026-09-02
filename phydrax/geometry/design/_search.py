#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._strict import StrictModule
from ...optim import DifferentialEvolutionSearch
from ...optim._differential_evolution import _bounded_differential_evolution
from ._constraints import DesignConstraintSystem
from ._schema import DesignState, ParameterId


SearchBounds = Mapping[ParameterId, tuple[ArrayLike, ArrayLike]]


class _DesignObjective(StrictModule):
    system: DesignConstraintSystem
    base_state: DesignState

    def __init__(
        self,
        system: DesignConstraintSystem,
        base_state: DesignState,
        /,
    ):
        self.system = system
        self.base_state = base_state

    def __call__(self, vector: Array, /) -> Array:
        state = self.system.unpack(vector, base_state=self.base_state)
        residual = self.system.residual(state)
        return jnp.sum(residual * residual)


class _DesignValidity(StrictModule):
    system: DesignConstraintSystem
    base_state: DesignState

    def __call__(self, vector: Array, /) -> Array:
        state = self.system.unpack(vector, base_state=self.base_state)
        evidence = self.system.geometry.validity(state)
        return evidence.resolved & evidence.accepted


class DesignSearchResult(StrictModule):
    """Best design state and convergence evidence from a bounded global search."""

    state: DesignState
    residual: Array
    residual_norm: Array
    objective: Array
    population_vectors: Array
    population_objectives: Array
    best_objective_history: Array
    lower_bounds: Array
    upper_bounds: Array
    key: Key[Array, ""]
    search: DifferentialEvolutionSearch
    converged: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    generations: int = eqx.field(static=True)
    objective_evaluations: int = eqx.field(static=True)
    invalid_evaluations: int = eqx.field(static=True)
    design_signature: str = eqx.field(static=True)

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
        key: Key[Array, ""],
        search: DifferentialEvolutionSearch,
        converged: bool,
        termination_reason: str,
        generations: int,
        objective_evaluations: int,
        invalid_evaluations: int,
        design_signature: str,
    ):
        residual_ = jnp.asarray(residual, dtype=float).reshape((-1,))
        self.state = state
        self.residual = residual_
        self.residual_norm = jnp.linalg.norm(residual_)
        self.objective = jnp.asarray(objective, dtype=float).reshape(())
        self.population_vectors = jnp.asarray(population_vectors, dtype=float)
        self.population_objectives = jnp.asarray(population_objectives, dtype=float)
        self.best_objective_history = jnp.asarray(best_objective_history, dtype=float)
        self.lower_bounds = jnp.asarray(lower_bounds, dtype=float)
        self.upper_bounds = jnp.asarray(upper_bounds, dtype=float)
        self.key = key
        self.search = search
        self.converged = bool(converged)
        self.termination_reason = str(termination_reason)
        self.generations = int(generations)
        self.objective_evaluations = int(objective_evaluations)
        self.invalid_evaluations = int(invalid_evaluations)
        self.design_signature = str(design_signature)


def _parameter_bound(
    value: ArrayLike,
    shape: tuple[int, ...],
    /,
    *,
    parameter_id: ParameterId,
    side: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        return np.full(shape, float(array), dtype=float).reshape((-1,))
    if array.shape != shape:
        raise ValueError(
            f"{side} search bound for {parameter_id} must be scalar or have shape "
            f"{shape}, got {array.shape}."
        )
    return array.reshape((-1,))


def _resolve_search_bounds(
    system: DesignConstraintSystem,
    bounds: SearchBounds | None,
    initial_state: DesignState,
    /,
) -> tuple[Array, Array, Array]:
    if bounds is not None and not isinstance(bounds, Mapping):
        raise TypeError("bounds must be a mapping from ParameterId to (lower, upper).")
    overrides = {} if bounds is None else dict(bounds)
    schema = system.geometry.schema
    index_by_id = {spec.parameter_id: index for index, spec in enumerate(schema.specs)}
    slices_by_index = {
        index: slice_info
        for index, slice_info in zip(
            system.trainable_indices,
            system.slices,
            strict=True,
        )
    }
    lower = np.asarray(system.lower_bounds, dtype=float).copy()
    upper = np.asarray(system.upper_bounds, dtype=float).copy()

    for parameter_id, pair in overrides.items():
        if not isinstance(parameter_id, ParameterId):
            raise TypeError("Every search-bound key must be a ParameterId.")
        if parameter_id not in index_by_id:
            raise KeyError(f"Unknown geometry parameter {parameter_id}.")
        index = index_by_id[parameter_id]
        if index not in slices_by_index:
            raise ValueError(
                f"Search bounds were provided for non-trainable {parameter_id}."
            )
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(
                f"Search bounds for {parameter_id} must be a (lower, upper) tuple."
            )
        start, stop, shape = slices_by_index[index]
        lower_ = _parameter_bound(pair[0], shape, parameter_id=parameter_id, side="Lower")
        upper_ = _parameter_bound(pair[1], shape, parameter_id=parameter_id, side="Upper")
        if not np.all(np.isfinite(lower_)) or not np.all(np.isfinite(upper_)):
            raise ValueError(f"Search bounds for {parameter_id} must be finite.")
        if np.any(lower_ >= upper_):
            raise ValueError(
                f"Every lower search bound for {parameter_id} must be smaller "
                "than its upper bound."
            )
        physical_lower, physical_upper = schema.specs[index].bounds
        if physical_lower is not None and np.any(lower_ < physical_lower):
            raise ValueError(
                f"Search bounds for {parameter_id} extend below the physical lower "
                f"bound {physical_lower}."
            )
        if physical_upper is not None and np.any(upper_ > physical_upper):
            raise ValueError(
                f"Search bounds for {parameter_id} extend above the physical upper "
                f"bound {physical_upper}."
            )
        lower[start:stop] = lower_
        upper[start:stop] = upper_

    missing = []
    for index, (start, stop, _shape) in slices_by_index.items():
        if np.any(~np.isfinite(lower[start:stop])) or np.any(
            ~np.isfinite(upper[start:stop])
        ):
            missing.append(str(schema.specs[index].parameter_id))
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Finite search bounds are required for: {names}.")
    if np.any(lower >= upper):
        raise ValueError("Every lower search bound must be smaller than its upper bound.")

    initial = np.asarray(system.pack(initial_state), dtype=float)
    outside = (initial < lower) | (initial > upper)
    if np.any(outside):
        raise ValueError("The initial design state lies outside the search bounds.")
    return (
        jnp.asarray(lower, dtype=float),
        jnp.asarray(upper, dtype=float),
        jnp.asarray(initial, dtype=float),
    )


def search_design_constraints(
    system: DesignConstraintSystem,
    search: DifferentialEvolutionSearch,
    /,
    *,
    key: Key[Array, ""],
    bounds: SearchBounds | None = None,
    initial_state: DesignState | None = None,
) -> DesignSearchResult:
    """Run bounded differential evolution over a compiled geometry design state."""
    if not isinstance(system, DesignConstraintSystem):
        raise TypeError("system must be a DesignConstraintSystem.")
    if not isinstance(search, DifferentialEvolutionSearch):
        raise TypeError("search must be a DifferentialEvolutionSearch.")
    initial_geometry_validity = system.geometry.validity(
        system.geometry.state if initial_state is None else initial_state
    )
    if not bool(np.asarray(initial_geometry_validity.resolved)):
        raise NotImplementedError(
            "Global design search requires executable geometry validity for a "
            f"restricted region; got "
            f"{system.geometry.field_certificate.validity_region!r}."
        )
    if not bool(np.asarray(initial_geometry_validity.accepted)):
        raise ValueError("The initial geometry design state is invalid.")
    state = system.geometry.state if initial_state is None else initial_state
    if not isinstance(state, DesignState):
        raise TypeError("initial_state must be a DesignState or None.")
    lower, upper, initial = _resolve_search_bounds(system, bounds, state)
    result = _bounded_differential_evolution(
        _DesignObjective(system, state),
        initial,
        lower,
        upper,
        search,
        key=key,
        validity=_DesignValidity(system, state),
    )
    best_state = system.unpack(result.best_vector, base_state=state)
    residual = system.residual(best_state)
    no_valid_candidates = result.termination_reason == "no_valid_candidates"
    best_valid = (
        False
        if no_valid_candidates
        else bool(np.asarray(system.geometry.validity(best_state).accepted))
    )
    objective = jnp.where(
        best_valid,
        jnp.sum(residual * residual),
        jnp.asarray(jnp.nan, dtype=residual.dtype),
    )
    converged = result.converged and best_valid
    termination_reason = (
        result.termination_reason
        if no_valid_candidates or best_valid
        else "best candidate has invalid geometry"
    )

    return DesignSearchResult(
        state=best_state,
        residual=residual,
        objective=objective,
        population_vectors=result.population_vectors,
        population_objectives=result.population_objectives,
        best_objective_history=result.best_objective_history,
        lower_bounds=result.lower_bounds,
        upper_bounds=result.upper_bounds,
        key=result.key,
        search=result.search,
        converged=converged,
        termination_reason=termination_reason,
        generations=result.generations,
        objective_evaluations=result.objective_evaluations,
        invalid_evaluations=result.invalid_evaluations,
        design_signature=result.design_signature,
    )


__all__ = ["DesignSearchResult"]
