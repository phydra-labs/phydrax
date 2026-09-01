#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._frozendict import frozendict
from ..._strict import StrictModule
from ._schema import DesignState, ParameterBinding, ParameterId, ParameterSchema


class DesignParameterization(StrictModule):
    """Scaled reduced coordinates over selected leaves of a geometry design state.

    Coordinates are dimensionless displacements from ``reference``. Each physical
    parameter is scaled by its schema ``physical_scale``; leaves not selected by
    ``parameter_ids`` remain fixed at their reference values.
    """

    reference: DesignState
    bindings: tuple[ParameterBinding, ...] = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)
    scales: Array
    lower_bounds: Array
    upper_bounds: Array

    def __init__(
        self,
        reference: DesignState,
        parameter_ids: Sequence[ParameterId] | None = None,
        /,
    ):
        if not isinstance(reference, DesignState):
            raise TypeError("reference must be a DesignState.")
        identifiers = (
            tuple(spec.parameter_id for spec in reference.schema.specs if spec.trainable)
            if parameter_ids is None
            else tuple(parameter_ids)
        )
        if not identifiers:
            raise ValueError("A design parameterization requires active parameters.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Active design parameter IDs must be unique.")

        bindings: list[ParameterBinding] = []
        offsets = [0]
        scale_parts = []
        lower_parts = []
        upper_parts = []
        for parameter_id in identifiers:
            if not isinstance(parameter_id, ParameterId):
                raise TypeError("parameter_ids must contain ParameterId values.")
            index = reference.schema.index(parameter_id)
            spec = reference.schema.specs[index]
            if not spec.trainable:
                raise ValueError(f"Parameter {parameter_id} is not trainable.")
            size = int(reference.values[index].size)
            if size == 0:
                raise ValueError(f"Parameter {parameter_id} cannot be empty.")
            bindings.append(ParameterBinding(parameter_id, index))
            offsets.append(offsets[-1] + size)
            scale_parts.append(jnp.full((size,), spec.physical_scale, dtype=float))
            lower, upper = spec.bounds
            reference_values = jnp.ravel(reference.values[index]).astype(float)
            lower_parts.append(
                jnp.full((size,), -jnp.inf, dtype=float)
                if lower is None
                else (jnp.full((size,), lower, dtype=float) - reference_values)
                / spec.physical_scale
            )
            upper_parts.append(
                jnp.full((size,), jnp.inf, dtype=float)
                if upper is None
                else (jnp.full((size,), upper, dtype=float) - reference_values)
                / spec.physical_scale
            )

        self.reference = reference
        self.bindings = tuple(bindings)
        self.offsets = tuple(offsets)
        self.scales = jnp.concatenate(tuple(scale_parts))
        self.lower_bounds = jnp.concatenate(tuple(lower_parts))
        self.upper_bounds = jnp.concatenate(tuple(upper_parts))

    @property
    def dimension(self) -> int:
        return self.offsets[-1]

    @property
    def parameter_ids(self) -> tuple[ParameterId, ...]:
        return tuple(binding.parameter_id for binding in self.bindings)

    @property
    def initial_coordinates(self) -> Array:
        return jnp.zeros((self.dimension,), dtype=self.scales.dtype)

    def reduce(self, state: DesignState, /) -> Array:
        """Map a full design state to dimensionless reduced coordinates."""
        self._validate_state(state)
        parts = []
        for binding, start, stop in zip(
            self.bindings,
            self.offsets[:-1],
            self.offsets[1:],
            strict=True,
        ):
            delta = jnp.ravel(state.values[binding.index]).astype(float) - jnp.ravel(
                self.reference.values[binding.index]
            ).astype(float)
            parts.append(delta / self.scales[start:stop])
        return jnp.concatenate(tuple(parts))

    def expand(self, coordinates: ArrayLike, /) -> DesignState:
        """Map dimensionless reduced coordinates to a full design state."""
        vector = jnp.asarray(coordinates, dtype=self.scales.dtype)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"Reduced coordinates must have shape {(self.dimension,)}, "
                f"got {vector.shape}."
            )
        values = list(self.reference.values)
        for binding, start, stop in zip(
            self.bindings,
            self.offsets[:-1],
            self.offsets[1:],
            strict=True,
        ):
            reference = self.reference.values[binding.index]
            physical = jnp.ravel(reference).astype(float) + (
                self.scales[start:stop] * vector[start:stop]
            )
            values[binding.index] = physical.reshape(reference.shape).astype(
                reference.dtype
            )
        return DesignState(self.reference.schema, values)

    def _validate_state(self, state: DesignState, /) -> None:
        if not isinstance(state, DesignState):
            raise TypeError("state must be a DesignState.")
        if state.schema != self.reference.schema:
            raise ValueError("Design state and parameterization schemas do not match.")


class DesignBindingGraph(StrictModule):
    """Stable named reads from a design state into an analysis model."""

    schema: ParameterSchema = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    bindings: tuple[ParameterBinding, ...] = eqx.field(static=True)

    def __init__(
        self,
        schema: ParameterSchema,
        bindings: Mapping[str, ParameterId | ParameterBinding],
        /,
    ):
        if not isinstance(schema, ParameterSchema):
            raise TypeError("schema must be a ParameterSchema.")
        names = tuple(bindings)
        if not names:
            raise ValueError("A design binding graph requires at least one binding.")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Design binding names must be non-empty strings.")
        compiled = []
        for name in names:
            binding = bindings[name]
            if isinstance(binding, ParameterId):
                index = schema.index(binding)
                compiled.append(ParameterBinding(binding, index))
            elif isinstance(binding, ParameterBinding):
                index = schema.index(binding.parameter_id)
                if binding.index != index:
                    raise ValueError(f"Binding {name!r} has an invalid schema index.")
                compiled.append(binding)
            else:
                raise TypeError(
                    "Design binding values must be ParameterId or ParameterBinding."
                )
        self.schema = schema
        self.names = names
        self.bindings = tuple(compiled)

    def read(self, state: DesignState, /) -> frozendict[str, Array]:
        """Read every named binding without copying the underlying arrays."""
        if not isinstance(state, DesignState):
            raise TypeError("state must be a DesignState.")
        if state.schema != self.schema:
            raise ValueError("Design state and binding graph schemas do not match.")
        return frozendict(
            {
                name: binding.read(state)
                for name, binding in zip(self.names, self.bindings, strict=True)
            }
        )


class DesignEvaluation(StrictModule):
    """One reduced-design evaluation with explicit validity evidence."""

    coordinates: Array
    state: DesignState
    bound_values: frozendict[str, Array]
    objective: Array
    constraints: Array
    valid: Array

    def __init__(
        self,
        *,
        coordinates: ArrayLike,
        state: DesignState,
        bound_values: Mapping[str, ArrayLike],
        objective: ArrayLike,
        constraints: ArrayLike,
        valid: ArrayLike,
    ):
        objective_ = jnp.asarray(objective, dtype=float)
        constraints_ = jnp.asarray(constraints, dtype=float)
        valid_ = jnp.asarray(valid, dtype=bool)
        if objective_.ndim != 0:
            raise ValueError("A design objective must be scalar.")
        if constraints_.ndim != 1:
            raise ValueError("Design constraints must be a one-dimensional residual.")
        if valid_.ndim != 0:
            raise ValueError("Design validity must be scalar.")
        self.coordinates = jnp.asarray(coordinates, dtype=float)
        self.state = state
        self.bound_values = frozendict(
            {name: jnp.asarray(value) for name, value in bound_values.items()}
        )
        self.objective = objective_
        self.constraints = constraints_
        self.valid = valid_


class ReducedDesignProblem(StrictModule):
    """Reduced coordinates and named bindings for an existing objective engine."""

    parameterization: DesignParameterization
    binding_graph: DesignBindingGraph
    objective_fn: Callable[[DesignState, frozendict[str, Array]], ArrayLike] = eqx.field(
        static=True
    )
    constraint_fns: tuple[
        Callable[[DesignState, frozendict[str, Array]], ArrayLike], ...
    ] = eqx.field(static=True)

    def __init__(
        self,
        parameterization: DesignParameterization,
        binding_graph: DesignBindingGraph,
        objective: Callable[[DesignState, frozendict[str, Array]], ArrayLike],
        /,
        *,
        constraints: Sequence[
            Callable[[DesignState, frozendict[str, Array]], ArrayLike]
        ] = (),
    ):
        if not isinstance(parameterization, DesignParameterization):
            raise TypeError("parameterization must be a DesignParameterization.")
        if not isinstance(binding_graph, DesignBindingGraph):
            raise TypeError("binding_graph must be a DesignBindingGraph.")
        if binding_graph.schema != parameterization.reference.schema:
            raise ValueError("Parameterization and binding graph schemas do not match.")
        if not callable(objective):
            raise TypeError("objective must be callable.")
        constraint_tuple = tuple(constraints)
        if any(not callable(constraint) for constraint in constraint_tuple):
            raise TypeError("Every design constraint must be callable.")
        self.parameterization = parameterization
        self.binding_graph = binding_graph
        self.objective_fn = objective
        self.constraint_fns = constraint_tuple

    def evaluate(self, coordinates: ArrayLike, /) -> DesignEvaluation:
        """Evaluate objective and residuals without choosing an optimizer."""
        state = self.parameterization.expand(coordinates)
        bound_values = self.binding_graph.read(state)
        objective = jnp.asarray(self.objective_fn(state, bound_values), dtype=float)
        if objective.ndim != 0:
            raise ValueError("The reduced design objective must return a scalar.")
        if self.constraint_fns:
            residuals = tuple(
                jnp.ravel(jnp.asarray(constraint(state, bound_values), dtype=float))
                for constraint in self.constraint_fns
            )
            constraints = jnp.concatenate(residuals)
        else:
            constraints = jnp.zeros((0,), dtype=objective.dtype)
        valid = jnp.isfinite(objective) & jnp.all(jnp.isfinite(constraints))
        return DesignEvaluation(
            coordinates=coordinates,
            state=state,
            bound_values=bound_values,
            objective=objective,
            constraints=constraints,
            valid=valid,
        )

    def objective(self, coordinates: ArrayLike, /) -> Array:
        """Return the scalar objective for use by the existing optimizer stack."""
        return self.evaluate(coordinates).objective


__all__ = [
    "DesignBindingGraph",
    "DesignEvaluation",
    "DesignParameterization",
    "ReducedDesignProblem",
]
