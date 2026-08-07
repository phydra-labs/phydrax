#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from .._capabilities import (
    GeometryCapability,
    SeamDiagnosticsProvider,
)
from .._contracts import CompiledGeometry, GeometryKernel
from ._schema import DesignState, ParameterId, ParameterSchema


class AbstractDesignConstraint(StrictModule):
    """Residual constraint evaluated against a compiled geometry state."""

    weight: Array

    def __init__(self, weight: float = 1.0):
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("constraint weight must be finite and positive.")
        self.weight = jnp.asarray(weight, dtype=float).reshape(())

    @abstractmethod
    def residual(
        self,
        kernel: GeometryKernel,
        schema: ParameterSchema,
        state: DesignState,
        /,
    ) -> Array:
        raise NotImplementedError

    def _weighted(self, value: Array) -> Array:
        return jnp.sqrt(self.weight) * jnp.atleast_1d(value)


class ParameterTarget(AbstractDesignConstraint):
    parameter_id: ParameterId = eqx.field(static=True)
    target: Array
    scale: Array

    def __init__(
        self,
        parameter_id: ParameterId,
        target: Array,
        *,
        scale: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        if not isinstance(parameter_id, ParameterId):
            raise TypeError("parameter_id must be a ParameterId.")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive.")
        self.parameter_id = parameter_id
        self.target = jnp.asarray(target, dtype=float)
        self.scale = jnp.asarray(scale, dtype=float).reshape(())

    def residual(self, kernel, schema, state, /):
        del kernel
        value = state.values[schema.index(self.parameter_id)]
        return self._weighted((value - self.target) / self.scale).reshape((-1,))


class ParameterEquality(AbstractDesignConstraint):
    first: ParameterId = eqx.field(static=True)
    second: ParameterId = eqx.field(static=True)
    scale: Array

    def __init__(
        self,
        first: ParameterId,
        second: ParameterId,
        *,
        scale: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        if not isinstance(first, ParameterId) or not isinstance(second, ParameterId):
            raise TypeError("first and second must be ParameterId objects.")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive.")
        self.first = first
        self.second = second
        self.scale = jnp.asarray(scale, dtype=float).reshape(())

    def residual(self, kernel, schema, state, /):
        del kernel
        first = state.values[schema.index(self.first)]
        second = state.values[schema.index(self.second)]
        return self._weighted((first - second) / self.scale).reshape((-1,))


class MeasureTarget(AbstractDesignConstraint):
    target: Array
    scale: Array

    def __init__(
        self,
        target: float,
        *,
        scale: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        if not np.isfinite(target):
            raise ValueError("target must be finite.")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive.")
        self.target = jnp.asarray(target, dtype=float).reshape(())
        self.scale = jnp.asarray(scale, dtype=float).reshape(())

    def residual(self, kernel, schema, state, /):
        del schema
        return self._weighted((kernel.measure(state) - self.target) / self.scale)


class BoundaryMeasureTarget(AbstractDesignConstraint):
    target: Array
    scale: Array

    def __init__(
        self,
        target: float,
        *,
        scale: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        if not np.isfinite(target):
            raise ValueError("target must be finite.")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive.")
        self.target = jnp.asarray(target, dtype=float).reshape(())
        self.scale = jnp.asarray(scale, dtype=float).reshape(())

    def residual(self, kernel, schema, state, /):
        del schema
        return self._weighted((kernel.boundary_measure(state) - self.target) / self.scale)


class BoundaryPoints(AbstractDesignConstraint):
    points: Array
    scale: Array

    def __init__(
        self,
        points: Array,
        *,
        scale: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        points_ = jnp.asarray(points, dtype=float)
        if points_.ndim != 2:
            raise ValueError("points must have shape (num_points, ambient_dimension).")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive.")
        self.points = points_
        self.scale = jnp.asarray(scale, dtype=float).reshape(())

    def residual(self, kernel, schema, state, /):
        del schema
        return self._weighted(kernel.boundary_field(state, self.points) / self.scale)


class InteriorClearance(AbstractDesignConstraint):
    points: Array
    clearance: Array
    scale: Array

    def __init__(
        self,
        points: Array,
        clearance: float = 0.0,
        *,
        scale: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        points_ = jnp.asarray(points, dtype=float)
        if points_.ndim != 2:
            raise ValueError("points must have shape (num_points, ambient_dimension).")
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError("clearance must be finite and non-negative.")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive.")
        self.points = points_
        self.clearance = jnp.asarray(clearance, dtype=float).reshape(())
        self.scale = jnp.asarray(scale, dtype=float).reshape(())

    def residual(self, kernel, schema, state, /):
        del schema
        field = kernel.boundary_field(state, self.points)
        return self._weighted(jax.nn.relu(field + self.clearance) / self.scale)


class ExteriorClearance(AbstractDesignConstraint):
    points: Array
    clearance: Array
    scale: Array

    def __init__(
        self,
        points: Array,
        clearance: float = 0.0,
        *,
        scale: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__(weight)
        points_ = jnp.asarray(points, dtype=float)
        if points_.ndim != 2:
            raise ValueError("points must have shape (num_points, ambient_dimension).")
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError("clearance must be finite and non-negative.")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive.")
        self.points = points_
        self.clearance = jnp.asarray(clearance, dtype=float).reshape(())
        self.scale = jnp.asarray(scale, dtype=float).reshape(())

    def residual(self, kernel, schema, state, /):
        del schema
        field = kernel.boundary_field(state, self.points)
        return self._weighted(jax.nn.relu(self.clearance - field) / self.scale)


class BRepSeamCompatibility(AbstractDesignConstraint):
    tolerance: Array

    def __init__(self, tolerance: float = 1e-8, *, weight: float = 1.0):
        super().__init__(weight)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        self.tolerance = jnp.asarray(tolerance, dtype=float).reshape(())

    def residual(self, kernel, schema, state, /):
        del schema
        if not isinstance(kernel, SeamDiagnosticsProvider):
            raise TypeError(
                "BRepSeamCompatibility requires a seam-diagnostics geometry kernel."
            )
        return self._weighted(kernel.seam_residual(state) / self.tolerance)


class ConstraintSolveResult(StrictModule):
    state: DesignState
    residual: Array
    residual_norm: Array
    converged: Array
    iterations: Array

    def __init__(
        self,
        *,
        state,
        residual,
        residual_norm,
        converged,
        iterations,
    ):
        self.state = state
        self.residual = jnp.asarray(residual, dtype=float)
        self.residual_norm = jnp.asarray(residual_norm, dtype=float).reshape(())
        self.converged = jnp.asarray(converged, dtype=bool).reshape(())
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())


class DesignConstraintSystem(StrictModule):
    """Bound JAX residual system over trainable compiled-geometry parameters."""

    geometry: CompiledGeometry
    constraints: tuple[AbstractDesignConstraint, ...]
    trainable_indices: tuple[int, ...] = eqx.field(static=True)
    slices: tuple[tuple[int, int, tuple[int, ...]], ...] = eqx.field(static=True)
    lower_bounds: Array
    upper_bounds: Array

    def __init__(
        self,
        geometry: CompiledGeometry,
        constraints: Sequence[AbstractDesignConstraint],
    ):
        if not isinstance(geometry, CompiledGeometry):
            raise TypeError("geometry must be a CompiledGeometry.")
        constraints_ = tuple(constraints)
        if not constraints_:
            raise ValueError("At least one design constraint is required.")
        if any(not isinstance(item, AbstractDesignConstraint) for item in constraints_):
            raise TypeError("constraints must contain design constraint objects.")
        self._validate_constraints(geometry, constraints_)
        indices = tuple(
            index for index, spec in enumerate(geometry.schema.specs) if spec.trainable
        )
        if not indices:
            raise ValueError("The compiled geometry has no trainable parameters.")
        slices: list[tuple[int, int, tuple[int, ...]]] = []
        lower: list[np.ndarray] = []
        upper: list[np.ndarray] = []
        offset = 0
        for index in indices:
            spec = geometry.schema.specs[index]
            size = int(np.prod(spec.shape, dtype=int))
            slices.append((offset, offset + size, spec.shape))
            low = -np.inf if spec.bounds[0] is None else spec.bounds[0]
            high = np.inf if spec.bounds[1] is None else spec.bounds[1]
            lower.append(np.full((size,), low, dtype=float))
            upper.append(np.full((size,), high, dtype=float))
            offset += size
        self.geometry = geometry
        self.constraints = constraints_
        self.trainable_indices = indices
        self.slices = tuple(slices)
        self.lower_bounds = jnp.asarray(np.concatenate(lower), dtype=float)
        self.upper_bounds = jnp.asarray(np.concatenate(upper), dtype=float)

    @staticmethod
    def _validate_constraints(geometry, constraints):
        schema = geometry.schema
        for constraint in constraints:
            if isinstance(constraint, ParameterTarget):
                index = schema.index(constraint.parameter_id)
                if constraint.target.shape != schema.specs[index].shape:
                    raise ValueError(
                        "ParameterTarget shape does not match its parameter."
                    )
            elif isinstance(constraint, ParameterEquality):
                first = schema.specs[schema.index(constraint.first)]
                second = schema.specs[schema.index(constraint.second)]
                if first.shape != second.shape:
                    raise ValueError("ParameterEquality requires matching shapes.")
            elif isinstance(constraint, (MeasureTarget, BoundaryMeasureTarget)):
                geometry.require(GeometryCapability.MEASURE)
            elif isinstance(
                constraint,
                (BoundaryPoints, InteriorClearance, ExteriorClearance),
            ):
                geometry.require(GeometryCapability.SIGNED_DISTANCE)
                if constraint.points.shape[1] != geometry.ambient_dimension:
                    raise ValueError(
                        "Constraint point dimension does not match geometry."
                    )
            elif isinstance(constraint, BRepSeamCompatibility):
                geometry.require(GeometryCapability.SEAM_DIAGNOSTICS)
                if not isinstance(geometry.kernel, SeamDiagnosticsProvider):
                    raise TypeError(
                        "BRepSeamCompatibility requires a seam-diagnostics geometry."
                    )

    def pack(self, state: DesignState, /) -> Array:
        if state.schema != self.geometry.schema:
            raise ValueError("state schema does not match the constraint system.")
        return jnp.concatenate(
            tuple(state.values[index].reshape((-1,)) for index in self.trainable_indices)
        )

    def unpack(self, vector: Array, /) -> DesignState:
        vector_ = jnp.asarray(vector, dtype=float).reshape((-1,))
        if vector_.shape != self.lower_bounds.shape:
            raise ValueError(
                "vector has the wrong number of trainable degrees of freedom."
            )
        values = list(self.geometry.state.values)
        for index, (start, stop, shape) in zip(
            self.trainable_indices, self.slices, strict=True
        ):
            values[index] = vector_[start:stop].reshape(shape)
        return DesignState(self.geometry.schema, tuple(values))

    def residual(self, state: DesignState, /) -> Array:
        values = tuple(
            constraint.residual(
                self.geometry.kernel,
                self.geometry.schema,
                state,
            ).reshape((-1,))
            for constraint in self.constraints
        )
        return jnp.concatenate(values)

    def solve(
        self,
        *,
        max_iterations: int = 32,
        tolerance: float = 1e-9,
        damping: float = 1e-8,
        line_search_steps: int = 8,
    ) -> ConstraintSolveResult:
        if max_iterations <= 0 or line_search_steps <= 0:
            raise ValueError("iteration counts must be positive.")
        if tolerance <= 0.0 or damping <= 0.0:
            raise ValueError("tolerance and damping must be positive.")
        initial = self.pack(self.geometry.state)

        def residual_vector(vector):
            return self.residual(self.unpack(vector))

        initial_norm = jnp.linalg.norm(residual_vector(initial))
        loop_state = (
            initial,
            initial_norm <= tolerance,
            jnp.asarray(0, dtype=jnp.int32),
        )

        def iteration(_, state):
            vector, converged, count = state

            def update(current):
                values = residual_vector(current)
                jacobian = jax.jacfwd(residual_vector)(current)
                normal = jacobian.T @ jacobian + damping * jnp.eye(
                    current.shape[0], dtype=current.dtype
                )
                step = jnp.linalg.solve(normal, -(jacobian.T @ values))
                current_loss = jnp.sum(values * values)
                search_state = (
                    jnp.asarray(1.0, dtype=current.dtype),
                    current,
                    current_loss,
                )

                def search(_, trial_state):
                    alpha, best, best_loss = trial_state
                    candidate = jnp.clip(
                        current + alpha * step,
                        self.lower_bounds,
                        self.upper_bounds,
                    )
                    candidate_values = residual_vector(candidate)
                    candidate_loss = jnp.sum(candidate_values * candidate_values)
                    improve = candidate_loss < best_loss
                    return (
                        alpha * 0.5,
                        jnp.where(improve, candidate, best),
                        jnp.where(improve, candidate_loss, best_loss),
                    )

                _, best, _ = jax.lax.fori_loop(0, line_search_steps, search, search_state)
                return best

            candidate = jax.lax.cond(converged, lambda value: value, update, vector)
            norm = jnp.linalg.norm(residual_vector(candidate))
            active = ~converged
            return (
                candidate,
                converged | (norm <= tolerance),
                count + active.astype(jnp.int32),
            )

        vector, converged, iterations = jax.lax.fori_loop(
            0, max_iterations, iteration, loop_state
        )
        state = self.unpack(vector)
        residual = self.residual(state)
        return ConstraintSolveResult(
            state=state,
            residual=residual,
            residual_norm=jnp.linalg.norm(residual),
            converged=converged,
            iterations=iterations,
        )


__all__ = [
    "AbstractDesignConstraint",
    "BRepSeamCompatibility",
    "BoundaryMeasureTarget",
    "BoundaryPoints",
    "ConstraintSolveResult",
    "DesignConstraintSystem",
    "ExteriorClearance",
    "InteriorClearance",
    "MeasureTarget",
    "ParameterEquality",
    "ParameterTarget",
]
