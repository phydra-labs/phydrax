#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...optim import DifferentialEvolutionSearch
from ...optim._differential_evolution import bounded_differential_evolution
from ...typing import checked, ConvertibleToArray, PRNGKey
from ._constraints import DesignConstraintSystem
from ._schema import DesignState, ParameterId
from ._search_result import DesignSearchResult


SearchBounds = Mapping[ParameterId, tuple[ConvertibleToArray, ConvertibleToArray]]


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


def _parameter_bound(
    value: ConvertibleToArray,
    shape: tuple[int, ...],
    /,
    *,
    parameter_id: ParameterId,
    side: str,
) -> np.ndarray:
    array = np.asarray(jax.device_get(value), dtype=np.float64)
    if array.shape == ():
        return np.full(shape, float(array), dtype=np.float64).reshape((-1,))
    if array.shape != shape:
        raise ValueError(
            f"DesignSearch.bounds must be scalar or have shape {shape}; got {parameter_id} {side}: {array.shape}"
        )
    return array.reshape((-1,))


def _resolve_search_bounds(
    system: DesignConstraintSystem,
    bounds: SearchBounds | None,
    initial_state: DesignState,
    /,
) -> tuple[Array, Array, Array]:
    if bounds is not None and not isinstance(bounds, Mapping):
        raise TypeError(
            f"DesignSearch.bounds must map ParameterId to lower and upper bounds; got {type(bounds).__name__}"
        )
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
    lower_host, upper_host = jax.device_get((system.lower_bounds, system.upper_bounds))
    lower = np.asarray(lower_host, dtype=np.float64).copy()
    upper = np.asarray(upper_host, dtype=np.float64).copy()

    for parameter_id, pair in overrides.items():
        if not isinstance(parameter_id, ParameterId):
            raise TypeError(
                f"DesignSearch.bounds.keys must contain ParameterId values; got {type(parameter_id).__name__}"
            )
        if parameter_id not in index_by_id:
            raise KeyError(
                f"DesignSearch.bounds.parameter_id must identify a geometry parameter; got {parameter_id}"
            )
        index = index_by_id[parameter_id]
        if index not in slices_by_index:
            raise ValueError(
                f"DesignSearch.bounds.parameter_id must be trainable; got {parameter_id}"
            )
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(
                f"DesignSearch.bounds.pair must be a lower and upper tuple; got {parameter_id}: {pair!r}"
            )
        start, stop, shape = slices_by_index[index]
        lower_ = _parameter_bound(pair[0], shape, parameter_id=parameter_id, side="Lower")
        upper_ = _parameter_bound(pair[1], shape, parameter_id=parameter_id, side="Upper")
        if not np.all(np.isfinite(lower_)) or not np.all(np.isfinite(upper_)):
            raise ValueError(
                f"DesignSearch.bounds must be finite; got {parameter_id}: {lower_}, {upper_}"
            )
        if np.any(lower_ >= upper_):
            raise ValueError(
                f"DesignSearch.bounds.lower must be smaller than upper; got {parameter_id}: {lower_}, {upper_}"
            )
        physical_lower, physical_upper = schema.specs[index].bounds
        if physical_lower is not None and np.any(lower_ < physical_lower):
            raise ValueError(
                f"DesignSearch.bounds.lower must respect the physical lower bound {physical_lower}; got {parameter_id}: {lower_}"
            )
        if physical_upper is not None and np.any(upper_ > physical_upper):
            raise ValueError(
                f"DesignSearch.bounds.upper must respect the physical upper bound {physical_upper}; got {parameter_id}: {upper_}"
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
        raise ValueError(
            f"DesignSearch.bounds must provide finite bounds for every trainable parameter; got missing {names}"
        )
    if np.any(lower >= upper):
        raise ValueError(
            "DesignSearch.bounds.lower must be smaller than upper; got invalid bound ordering"
        )

    initial = np.asarray(jax.device_get(system.pack(initial_state)), dtype=np.float64)
    outside = (initial < lower) | (initial > upper)
    if np.any(outside):
        raise ValueError(
            "DesignSearch.initial_state must lie inside search bounds; got outside state"
        )
    return (
        jnp.asarray(lower, dtype=jnp.float64),
        jnp.asarray(upper, dtype=jnp.float64),
        jnp.asarray(initial, dtype=jnp.float64),
    )


@checked
def search_design_constraints(
    system: DesignConstraintSystem,
    search: DifferentialEvolutionSearch,
    /,
    *,
    key: PRNGKey,
    bounds: SearchBounds | None = None,
    initial_state: DesignState | None = None,
) -> DesignSearchResult:
    """Run bounded differential evolution over a compiled geometry design state."""
    initial_geometry_validity = system.geometry.validity(
        system.geometry.state if initial_state is None else initial_state
    )
    resolved, accepted = jax.device_get(
        (initial_geometry_validity.resolved, initial_geometry_validity.accepted)
    )
    if not bool(resolved):
        raise NotImplementedError(
            f"DesignSearch.geometry must provide executable geometry validity; got {system.geometry.field_certificate.validity_region!r}"
        )
    if not bool(accepted):
        raise ValueError(
            "DesignSearch.initial_state must satisfy geometry validity; got invalid state"
        )
    state = system.geometry.state if initial_state is None else initial_state
    lower, upper, initial = _resolve_search_bounds(system, bounds, state)
    result = bounded_differential_evolution(
        _DesignObjective(system, state),
        initial,
        lower,
        upper,
        search,
        key=key,
        validity=_DesignValidity(system, state),
    )
    best_state = system.unpack(jnp.asarray(result.best), base_state=state)
    residual = system.residual(best_state)
    no_valid_candidates = result.termination_reason == "no_valid_candidates"
    best_valid = (
        False
        if no_valid_candidates
        else bool(jax.device_get(system.geometry.validity(best_state).accepted))
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
        population_vectors=jnp.asarray(result.population),
        population_objectives=result.population_objectives,
        best_objective_history=result.best_objective_history,
        lower_bounds=lower,
        upper_bounds=upper,
        key=result.key,
        search=result.search,
        converged=converged,
        termination_reason=termination_reason,
        generations=result.generations,
        objective_evaluations=result.objective_evaluations,
        invalid_evaluations=result.invalid_evaluations,
        design_signature=result.design_signature,
    )


__all__ = ["search_design_constraints"]
