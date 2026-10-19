#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._mass import require_exact_mass
from ..._strict import StrictModule
from ..._validation import positive_integer, positive_real_scalar
from ...optim import DifferentialEvolutionSearch, LeastSquaresResult
from ...typing import checked, ConvertibleToArray, PRNGKey
from .._capabilities import (
    GeometryCapability,
    SeamDiagnosticsProvider,
)
from .._contracts import CompiledGeometry, GeometryKernel
from ._schema import DesignState, ParameterId, ParameterSchema
from ._search_result import DesignSearchResult
from ._solve import DesignTrialPolicy, solve_design_residual


class AbstractDesignConstraint(StrictModule):
    """Residual constraint evaluated against a compiled geometry state."""

    weight: Array

    def __init__(self, weight: float = 1.0) -> None:
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"AbstractDesignConstraint.weight must be finite and positive; got {weight}"
            )
        self.weight = jnp.asarray(weight, dtype=jnp.float64).reshape(())

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

    @checked
    def __init__(
        self,
        parameter_id: ParameterId,
        target: ConvertibleToArray,
        *,
        scale: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"DesignConstraintSystem.constraints.scale must be finite and positive; got {scale}"
            )
        self.parameter_id = parameter_id
        self.target = jnp.asarray(target, dtype=jnp.float64)
        self.scale = jnp.asarray(scale, dtype=jnp.float64).reshape(())

    def residual(
        self,
        kernel: GeometryKernel,
        schema: ParameterSchema,
        state: DesignState,
        /,
    ) -> Array:
        del kernel
        value = state.values[schema.index(self.parameter_id)]
        return self._weighted((value - self.target) / self.scale).reshape((-1,))


class ParameterEquality(AbstractDesignConstraint):
    first: ParameterId = eqx.field(static=True)
    second: ParameterId = eqx.field(static=True)
    scale: Array

    @checked
    def __init__(
        self,
        first: ParameterId,
        second: ParameterId,
        *,
        scale: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"DesignConstraintSystem.constraints.scale must be finite and positive; got {scale}"
            )
        self.first = first
        self.second = second
        self.scale = jnp.asarray(scale, dtype=jnp.float64).reshape(())

    def residual(
        self,
        kernel: GeometryKernel,
        schema: ParameterSchema,
        state: DesignState,
        /,
    ) -> Array:
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
    ) -> None:
        super().__init__(weight)
        if not np.isfinite(target):
            raise ValueError(
                f"DesignConstraintSystem.constraints.target must be finite; got {target}"
            )
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"DesignConstraintSystem.constraints.scale must be finite and positive; got {scale}"
            )
        self.target = jnp.asarray(target, dtype=jnp.float64).reshape(())
        self.scale = jnp.asarray(scale, dtype=jnp.float64).reshape(())

    def residual(
        self,
        kernel: GeometryKernel,
        schema: ParameterSchema,
        state: DesignState,
        /,
    ) -> Array:
        del schema
        mass = require_exact_mass(
            kernel.interior_mass(state),
            operation="Geometry measure target",
        )
        return self._weighted((mass - self.target) / self.scale)


class BoundaryMeasureTarget(AbstractDesignConstraint):
    target: Array
    scale: Array

    def __init__(
        self,
        target: float,
        *,
        scale: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        if not np.isfinite(target):
            raise ValueError(
                f"DesignConstraintSystem.constraints.target must be finite; got {target}"
            )
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"DesignConstraintSystem.constraints.scale must be finite and positive; got {scale}"
            )
        self.target = jnp.asarray(target, dtype=jnp.float64).reshape(())
        self.scale = jnp.asarray(scale, dtype=jnp.float64).reshape(())

    def residual(
        self,
        kernel: GeometryKernel,
        schema: ParameterSchema,
        state: DesignState,
        /,
    ) -> Array:
        del schema
        mass = require_exact_mass(
            kernel.boundary_mass(state),
            operation="Geometry boundary-measure target",
        )
        return self._weighted((mass - self.target) / self.scale)


class BoundaryPoints(AbstractDesignConstraint):
    points: Array
    scale: Array

    def __init__(
        self,
        points: Array,
        *,
        scale: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(weight)
        points_ = jnp.asarray(points, dtype=jnp.float64)
        if points_.ndim != 2:
            raise ValueError(
                f"DesignConstraintSystem.constraints.points must have shape (num_points, ambient_dimension); got {points_.shape}"
            )
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"DesignConstraintSystem.constraints.scale must be finite and positive; got {scale}"
            )
        self.points = points_
        self.scale = jnp.asarray(scale, dtype=jnp.float64).reshape(())

    def residual(
        self,
        kernel: GeometryKernel,
        schema: ParameterSchema,
        state: DesignState,
        /,
    ) -> Array:
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
    ) -> None:
        super().__init__(weight)
        points_ = jnp.asarray(points, dtype=jnp.float64)
        if points_.ndim != 2:
            raise ValueError(
                f"DesignConstraintSystem.constraints.points must have shape (num_points, ambient_dimension); got {points_.shape}"
            )
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError(
                f"DesignConstraintSystem.constraints.clearance must be finite and nonnegative; got {clearance}"
            )
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"DesignConstraintSystem.constraints.scale must be finite and positive; got {scale}"
            )
        self.points = points_
        self.clearance = jnp.asarray(clearance, dtype=jnp.float64).reshape(())
        self.scale = jnp.asarray(scale, dtype=jnp.float64).reshape(())

    def residual(
        self,
        kernel: GeometryKernel,
        schema: ParameterSchema,
        state: DesignState,
        /,
    ) -> Array:
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
    ) -> None:
        super().__init__(weight)
        points_ = jnp.asarray(points, dtype=jnp.float64)
        if points_.ndim != 2:
            raise ValueError(
                f"DesignConstraintSystem.constraints.points must have shape (num_points, ambient_dimension); got {points_.shape}"
            )
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError(
                f"DesignConstraintSystem.constraints.clearance must be finite and nonnegative; got {clearance}"
            )
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"DesignConstraintSystem.constraints.scale must be finite and positive; got {scale}"
            )
        self.points = points_
        self.clearance = jnp.asarray(clearance, dtype=jnp.float64).reshape(())
        self.scale = jnp.asarray(scale, dtype=jnp.float64).reshape(())

    def residual(
        self,
        kernel: GeometryKernel,
        schema: ParameterSchema,
        state: DesignState,
        /,
    ) -> Array:
        del schema
        field = kernel.boundary_field(state, self.points)
        return self._weighted(jax.nn.relu(self.clearance - field) / self.scale)


class BRepSeamCompatibility(AbstractDesignConstraint):
    tolerance: Array

    def __init__(self, tolerance: float = 1e-8, *, weight: float = 1.0) -> None:
        super().__init__(weight)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError(
                f"BRepSeamCompatibility.tolerance must be finite and positive; got {tolerance}"
            )
        self.tolerance = jnp.asarray(tolerance, dtype=jnp.float64).reshape(())

    def residual(
        self,
        kernel: GeometryKernel,
        schema: ParameterSchema,
        state: DesignState,
        /,
    ) -> Array:
        del schema
        match kernel:
            case SeamDiagnosticsProvider():
                return self._weighted(kernel.seam_residual(state) / self.tolerance)
            case _:
                raise TypeError(
                    f"BRepSeamCompatibility.kernel must provide seam diagnostics; got {type(kernel).__name__}"
                )


class ConstraintSolveResult(StrictModule):
    """Return a design state with native status and weighted residual evidence."""

    state: DesignState
    residual: Array
    residual_norm: Array
    residual_bound: Array
    converged: Array
    iterations: Array
    status: Array
    solver_result: LeastSquaresResult

    @checked
    def __init__(
        self,
        *,
        state: DesignState,
        solver_result: LeastSquaresResult,
        residual_bound: float,
    ) -> None:
        residual_bound = positive_real_scalar(
            residual_bound, "ConstraintSolveResult.residual_bound"
        )
        self.state = state
        self.residual = jnp.asarray(solver_result.residual, dtype=jnp.float64)
        self.residual_norm = jnp.linalg.norm(self.residual)
        self.residual_bound = jnp.asarray(residual_bound, dtype=jnp.float64)
        self.converged = solver_result.successful & (
            self.residual_norm <= self.residual_bound
        )
        self.iterations = solver_result.diagnostics.iterations
        self.status = solver_result.status
        self.solver_result = solver_result


class DesignConstraintSystem(StrictModule):
    """Bound JAX residual system over trainable compiled-geometry parameters."""

    geometry: CompiledGeometry
    constraints: tuple[AbstractDesignConstraint, ...]
    trainable_indices: tuple[int, ...] = eqx.field(static=True)
    slices: tuple[tuple[int, int, tuple[int, ...]], ...] = eqx.field(static=True)
    lower_bounds: Array
    upper_bounds: Array

    @checked
    def __init__(
        self,
        geometry: CompiledGeometry,
        constraints: Sequence[AbstractDesignConstraint],
    ) -> None:
        constraints_ = tuple(constraints)
        if not constraints_:
            raise ValueError(
                "DesignConstraintSystem.constraints must contain at least one constraint; got empty constraints"
            )
        if any(not isinstance(item, AbstractDesignConstraint) for item in constraints_):
            raise TypeError(
                "DesignConstraintSystem.constraints must contain design constraints; got unsupported member"
            )
        self._validate_constraints(geometry, constraints_)
        indices = tuple(
            index for index, spec in enumerate(geometry.schema.specs) if spec.trainable
        )
        if not indices:
            raise ValueError(
                "DesignConstraintSystem.geometry must have trainable parameters; got none"
            )
        slices: list[tuple[int, int, tuple[int, ...]]] = []
        lower: list[np.ndarray] = []
        upper: list[np.ndarray] = []
        offset = 0
        for index in indices:
            spec = geometry.schema.specs[index]
            size = int(np.prod(spec.shape, dtype=np.int64))
            slices.append((offset, offset + size, spec.shape))
            low = -np.inf if spec.bounds[0] is None else spec.bounds[0]
            high = np.inf if spec.bounds[1] is None else spec.bounds[1]
            lower.append(np.full((size,), low, dtype=np.float64))
            upper.append(np.full((size,), high, dtype=np.float64))
            offset += size
        self.geometry = geometry
        self.constraints = constraints_
        self.trainable_indices = indices
        self.slices = tuple(slices)
        self.lower_bounds = jnp.asarray(np.concatenate(lower), dtype=jnp.float64)
        self.upper_bounds = jnp.asarray(np.concatenate(upper), dtype=jnp.float64)

    @staticmethod
    def _validate_constraints(
        geometry: CompiledGeometry,
        constraints: tuple[AbstractDesignConstraint, ...],
    ) -> None:
        schema = geometry.schema
        for constraint in constraints:
            match constraint:
                case ParameterTarget():
                    index = schema.index(constraint.parameter_id)
                    if constraint.target.shape != schema.specs[index].shape:
                        raise ValueError(
                            "ParameterTarget.target must match its parameter shape; "
                            f"got {constraint.target.shape}"
                        )
                case ParameterEquality():
                    first = schema.specs[schema.index(constraint.first)]
                    second = schema.specs[schema.index(constraint.second)]
                    if first.unit != second.unit:
                        raise ValueError(
                            "ParameterEquality.parameters must have exactly matching units; "
                            f"got {first.unit.symbol}, {second.unit.symbol}"
                        )
                    if first.shape != second.shape:
                        raise ValueError(
                            "ParameterEquality.parameters must have matching shapes; "
                            f"got {first.shape}, {second.shape}"
                        )
                case MeasureTarget():
                    geometry.require(GeometryCapability.INTERIOR_MEASURE)
                    require_exact_mass(
                        geometry.interior_mass,
                        operation="Geometry measure target",
                    )
                case BoundaryMeasureTarget():
                    geometry.require(GeometryCapability.BOUNDARY_MEASURE)
                    require_exact_mass(
                        geometry.boundary_mass,
                        operation="Geometry boundary-measure target",
                    )
                case BoundaryPoints() | InteriorClearance() | ExteriorClearance():
                    geometry.require(GeometryCapability.SIGNED_DISTANCE)
                    if constraint.points.shape[1] != geometry.ambient_dimension:
                        raise ValueError(
                            "DesignConstraintSystem.constraints.points must match geometry dimension; "
                            f"got {constraint.points.shape[1]}"
                        )
                case BRepSeamCompatibility():
                    geometry.require(GeometryCapability.SEAM_DIAGNOSTICS)
                    if not isinstance(geometry.kernel, SeamDiagnosticsProvider):
                        raise TypeError(
                            "BRepSeamCompatibility.geometry must provide seam diagnostics; "
                            f"got {type(geometry.kernel).__name__}"
                        )
                case AbstractDesignConstraint():
                    pass
                case _:
                    raise TypeError(
                        "DesignConstraintSystem.constraints must contain design constraints; "
                        f"got {type(constraint).__name__}"
                    )

    @checked
    def pack(self, state: DesignState, /) -> Array:
        if state.schema != self.geometry.schema:
            raise ValueError(
                "DesignConstraintSystem.state.schema must match geometry.schema; got different schema"
            )
        return jnp.concatenate(
            tuple(state.values[index].reshape((-1,)) for index in self.trainable_indices)
        )

    @checked
    def unpack(
        self,
        vector: Array,
        /,
        *,
        base_state: DesignState | None = None,
    ) -> DesignState:
        vector_ = jnp.asarray(vector, dtype=jnp.float64).reshape((-1,))
        if vector_.shape != self.lower_bounds.shape:
            raise ValueError(
                f"DesignConstraintSystem.vector must match trainable degree count; got {vector_.shape}"
            )
        state = self.geometry.state if base_state is None else base_state
        if state.schema != self.geometry.schema:
            raise ValueError(
                "DesignConstraintSystem.base_state.schema must match geometry.schema; got different schema"
            )
        values = list(state.values)
        for index, (start, stop, shape) in zip(
            self.trainable_indices, self.slices, strict=True
        ):
            values[index] = vector_[start:stop].reshape(shape)
        return DesignState(self.geometry.schema, tuple(values))

    @checked
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

    @checked
    def search(
        self,
        search: DifferentialEvolutionSearch,
        /,
        *,
        key: PRNGKey,
        bounds: Mapping[ParameterId, tuple[ConvertibleToArray, ConvertibleToArray]]
        | None = None,
        initial_state: DesignState | None = None,
    ) -> DesignSearchResult:
        """Search globally over a finite box of trainable geometry parameters."""
        # Search callbacks depend on this system; keep this one cycle lazy.
        from ._search import search_design_constraints

        return search_design_constraints(
            self,
            search,
            key=key,
            bounds=bounds,
            initial_state=initial_state,
        )

    @checked
    def solve(
        self,
        *,
        initial_state: DesignState | None = None,
        max_iterations: int = 32,
        tolerance: float = 1e-9,
        damping: float = 1e-8,
        line_search_steps: int = 8,
    ) -> ConstraintSolveResult:
        """Solve weighted design residuals with native LM and bound refusal.

        Only residual-decreasing, admissible trials commit. Native optimization
        status and work evidence accompany the value-only design solution.
        """
        max_iterations = positive_integer(
            max_iterations, "DesignConstraintSystem.solve.max_iterations"
        )
        line_search_steps = positive_integer(
            line_search_steps, "DesignConstraintSystem.solve.line_search_steps"
        )
        tolerance = positive_real_scalar(
            tolerance, "DesignConstraintSystem.solve.tolerance"
        )
        damping = positive_real_scalar(damping, "DesignConstraintSystem.solve.damping")
        state = self.geometry.state if initial_state is None else initial_state
        initial = self.pack(state)
        result = solve_design_residual(
            _DesignResidual(self, state),
            initial,
            DesignTrialPolicy(
                self.lower_bounds,
                self.upper_bounds,
                jnp.zeros(initial.shape, dtype=jnp.bool_),
                tolerance,
                validity=_DesignAdmissibility(self, state),
            ),
            max_iterations=max_iterations,
            damping=damping,
            maximum_trials=line_search_steps,
            problem_id="geometry/design-constraints",
        )
        vector = jnp.asarray(result.parameters, dtype=initial.dtype)
        return ConstraintSolveResult(
            state=self.unpack(vector, base_state=state),
            solver_result=result,
            residual_bound=tolerance,
        )


class _DesignAdmissibility(StrictModule):
    system: DesignConstraintSystem
    base_state: DesignState

    def __call__(self, vector: Array, /) -> Array:
        state = self.system.unpack(vector, base_state=self.base_state)
        evidence = self.system.geometry.validity(state)
        return evidence.resolved & evidence.accepted


class _DesignResidual(StrictModule):
    system: DesignConstraintSystem
    base_state: DesignState

    def __call__(self, vector: Array, args: None, /) -> Array:
        del args
        return self.system.residual(
            self.system.unpack(vector, base_state=self.base_state)
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
