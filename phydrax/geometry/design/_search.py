#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...optim import DifferentialEvolutionSearch
from ...optim._differential_evolution import _bounded_differential_evolution
from ...typing import PRNGKey
from ._constraints import DesignConstraintSystem
from ._schema import DesignState, ParameterId
from ._search_result import SketchSearchResult
from ._sketch import Sketch
from ._sketch_certificate import RootDOFEvidence


SearchBounds = Mapping[ParameterId, tuple[ArrayLike, ArrayLike]]
SketchSearchBounds = tuple[ArrayLike, ArrayLike]


class _DesignObjective(StrictModule):
    system: DesignConstraintSystem
    base_state: DesignState

    def __init__(
        self,
        system: DesignConstraintSystem,
        base_state: DesignState,
        /,
    ) -> None:
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

    __strict_contract__ = True

    state: DesignState
    residual: Array
    residual_norm: Array
    objective: Array
    root_evidence: RootDOFEvidence
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
        root_evidence: RootDOFEvidence,
    ) -> None:
        if not isinstance(root_evidence, RootDOFEvidence):
            raise TypeError("root_evidence must be RootDOFEvidence.")
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
        self.root_evidence = root_evidence
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
    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        return np.full(shape, float(array), dtype=np.float64).reshape((-1,))
    if array.shape != shape:
        raise ValueError(
            f"{side} search bound for {parameter_id} must be scalar or have shape {shape}, got {array.shape}."
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
    lower = np.asarray(system.lower_bounds, dtype=np.float64).copy()
    upper = np.asarray(system.upper_bounds, dtype=np.float64).copy()

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
                f"Every lower search bound for {parameter_id} must be smaller than its upper bound."
            )
        physical_lower, physical_upper = schema.specs[index].bounds
        if physical_lower is not None and np.any(lower_ < physical_lower):
            raise ValueError(
                f"Search bounds for {parameter_id} extend below the physical lower bound {physical_lower}."
            )
        if physical_upper is not None and np.any(upper_ > physical_upper):
            raise ValueError(
                f"Search bounds for {parameter_id} extend above the physical upper bound {physical_upper}."
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

    initial = np.asarray(system.pack(initial_state), dtype=np.float64)
    outside = (initial < lower) | (initial > upper)
    if np.any(outside):
        raise ValueError("The initial design state lies outside the search bounds.")
    return (
        jnp.asarray(lower, dtype=jnp.float64),
        jnp.asarray(upper, dtype=jnp.float64),
        jnp.asarray(initial, dtype=jnp.float64),
    )


def search_design_constraints(
    system: DesignConstraintSystem,
    search: DifferentialEvolutionSearch,
    /,
    *,
    key: PRNGKey,
    bounds: SearchBounds | None = None,
    initial_state: DesignState | None = None,
    root_tolerance: float = 1e-9,
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
    if not np.isfinite(root_tolerance) or root_tolerance <= 0.0:
        raise ValueError("root_tolerance must be finite and positive.")
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
    coordinates = system.pack(best_state)
    jacobian = jax.jacfwd(
        lambda vector: system.residual(system.unpack(vector, base_state=state))
    )(coordinates)
    root_evidence = RootDOFEvidence(
        system._problem_id(),
        coordinates,
        residual,
        jacobian,
        residual_tolerance=root_tolerance,
    )
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
        root_evidence=root_evidence,
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


class _SketchObjective(StrictModule):
    sketch: Sketch
    point_size: int = eqx.field(static=True)

    def __init__(self, sketch: Sketch, /) -> None:
        self.sketch = sketch
        self.point_size = sketch.points.size

    def __call__(self, vector: Array, /) -> Array:
        points = vector[: self.point_size].reshape(self.sketch.points.shape)
        radii = vector[self.point_size :]
        residual = self.sketch.residual(points, radii)
        return jnp.sum(residual * residual)


class _SketchValidity(StrictModule):
    sketch: Sketch
    point_size: int = eqx.field(static=True)

    def __init__(self, sketch: Sketch, /) -> None:
        self.sketch = sketch
        self.point_size = sketch.points.size

    def __call__(self, vector: Array, /) -> Array:
        points = vector[: self.point_size].reshape(self.sketch.points.shape)
        radii = vector[self.point_size :]
        return (
            jnp.all(jnp.isfinite(vector))
            & jnp.all(radii > 0.0)
            & jnp.all(jnp.isfinite(self.sketch.residual(points, radii)))
        )


def _resolve_sketch_bounds(
    sketch: Sketch,
    bounds: SketchSearchBounds,
    /,
) -> tuple[Array, Array, Array]:
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise TypeError("bounds must be a (lower, upper) tuple.")
    lower = np.asarray(bounds[0], dtype=np.float64).reshape((-1,))
    upper = np.asarray(bounds[1], dtype=np.float64).reshape((-1,))
    initial = np.asarray(
        jnp.concatenate((sketch.points.reshape((-1,)), sketch.circle_radii)),
        dtype=np.float64,
    )
    if initial.size == 0:
        raise ValueError("Sketch search requires at least one variable.")
    if lower.shape != initial.shape or upper.shape != initial.shape:
        raise ValueError(f"Sketch search bounds must have shape {initial.shape}.")
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
    ):
        raise ValueError("Sketch search bounds must be finite and strictly ordered.")
    radius_start = sketch.points.size
    if sketch.circle_radii.size and np.any(lower[radius_start:] <= 0.0):
        raise ValueError("Circle-radius search bounds must remain strictly positive.")
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("The initial sketch state lies outside the search bounds.")
    return (
        jnp.asarray(lower, dtype=jnp.float64),
        jnp.asarray(upper, dtype=jnp.float64),
        jnp.asarray(initial, dtype=jnp.float64),
    )


def search_sketch(
    sketch: Sketch,
    search: DifferentialEvolutionSearch,
    /,
    *,
    key: PRNGKey,
    bounds: SketchSearchBounds,
    max_iterations: int = 32,
    tolerance: float = 1e-9,
    damping: float = 1e-8,
) -> SketchSearchResult:
    """Search a finite sketch-coordinate box, then solve the selected root locally."""
    if not isinstance(sketch, Sketch):
        raise TypeError("sketch must be a Sketch.")
    if not isinstance(search, DifferentialEvolutionSearch):
        raise TypeError("search must be a DifferentialEvolutionSearch.")
    if not sketch.constraints:
        raise ValueError("Sketch search requires at least one constraint.")
    if search.selection != "scalar" or search.objective_count != 1:
        raise ValueError("Sketch search supports one scalar residual objective.")
    lower, upper, initial = _resolve_sketch_bounds(sketch, bounds)
    result = _bounded_differential_evolution(
        _SketchObjective(sketch),
        initial,
        lower,
        upper,
        search,
        key=key,
        validity=_SketchValidity(sketch),
    )
    no_valid_candidates = result.termination_reason == "no_valid_candidates"
    solution = (
        None
        if no_valid_candidates
        else sketch.solve(
            initial_coordinates=result.best_vector,
            max_iterations=max_iterations,
            tolerance=tolerance,
            damping=damping,
        )
    )
    termination_reason = (
        result.termination_reason
        if no_valid_candidates or solution is None or bool(np.asarray(solution.converged))
        else "best candidate did not converge to a sketch root"
    )
    return SketchSearchResult(
        solution=solution,
        population_vectors=result.population_vectors,
        population_objectives=result.population_objectives,
        best_objective_history=result.best_objective_history,
        lower_bounds=result.lower_bounds,
        upper_bounds=result.upper_bounds,
        key=result.key,
        search=result.search,
        search_converged=result.converged,
        termination_reason=termination_reason,
        generations=result.generations,
        objective_evaluations=result.objective_evaluations,
        invalid_evaluations=result.invalid_evaluations,
        design_signature=result.design_signature,
    )


__all__ = ["DesignSearchResult", "SketchSearchResult", "search_sketch"]
