#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization import TemporalMesh
from ..dynamics import (
    ContinuousSystem,
    DifferentialAlgebraicSystem,
    StateLayout,
    TimeGrid,
)
from ..linalg import AbstractVectorSpace, ArraySpace
from ..optim import (
    AbstractMinimizationMethod,
    AbstractStructuredNonlinearMethod,
    bind_structured_numeric,
    Bounds,
    MinimizationProblem,
    MinimizationResult,
    minimize,
    NonlinearConstraint,
    OptimizationTermination,
    PooledStructuredNonlinearResult,
    prepare_structured_nonlinear,
    PreparedStructuredNonlinearProgram,
    refresh_structured_nonlinear,
    solve_pooled_structured_nonlinear,
    solve_structured_nonlinear,
    StructuredNonlinearProgram,
    StructuredNonlinearResult,
    StructuredNonlinearTemplate,
    StructuredNonlinearWarmStart,
    StructuredPoolEvidence,
)
from ..solver._theta import ThetaMethod
from ..sparse import (
    compile_sparse_hessian,
    compile_sparse_jacobian,
    SparseDerivativePlan,
    SparseDerivativeVerification,
    verify_sparse_derivative,
)
from ._dynamics import DifferentialControlDynamics
from ._problem import _identifier, ControlProblem
from ._trajectory import (
    CONTROL_DYNAMICS_FAILED,
    CONTROL_INFEASIBLE,
    CONTROL_SUCCESS,
    ControlTrajectory,
)
from ._trajectory_optimization import (
    BoundedPathConstraint,
    BoundedTrajectoryConstraint,
    TrajectoryOptimizationContext,
    TrajectoryOptimizationProblem,
    TrajectoryOptimizationView,
)


DIRECT_COLLOCATION_SUCCESS = 0
DIRECT_COLLOCATION_OPTIMIZER_FAILED = 1
DIRECT_COLLOCATION_NONFINITE = 2
DIRECT_COLLOCATION_DEFECT_FAILED = 3
DIRECT_COLLOCATION_CONSTRAINT_FAILED = 4
DIRECT_COLLOCATION_RECONSTRUCTION_FAILED = 5

DirectCollocationHessianMode: TypeAlias = Literal["limited-memory", "exact-sparse"]
SparseDerivativeCompiler: TypeAlias = Literal["auto", "native", "asdex"]
_DEFAULT_ARGS = object()


def _inexact(value: ArrayLike, owner: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{owner} must be real-valued.")
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _positive(value: ArrayLike, owner: str, /) -> Array:
    array = _inexact(value, owner)
    return eqx.error_if(
        array,
        jnp.any(~jnp.isfinite(array)) | jnp.any(array <= 0.0),
        f"{owner} must be finite and positive.",
    )


def _broadcast_scale(value: Any, shape: tuple[int, ...], owner: str, /) -> Array:
    scale = jnp.ones(shape) if value is None else _positive(value, owner)
    if np.broadcast_shapes(scale.shape, shape) != shape:
        raise ValueError(f"{owner} cannot broadcast to shape {shape}.")
    return jnp.broadcast_to(scale, shape)


def _event_finite(values: Array, event_shape: tuple[int, ...], /) -> Array:
    if not event_shape:
        return jnp.isfinite(values)
    axes = tuple(range(values.ndim - len(event_shape), values.ndim))
    return jnp.all(jnp.isfinite(values), axis=axes)


def _maximum_absolute(value: Array, /) -> Array:
    return jnp.max(jnp.abs(value), initial=jnp.asarray(0.0, dtype=value.dtype))


def _maximum_bound_violation(value: Array, lower: Array, upper: Array, /) -> Array:
    return jnp.max(
        jnp.maximum(jnp.maximum(lower - value, value - upper), 0.0),
        initial=jnp.asarray(0.0, dtype=value.dtype),
    )


def _space_ones(space: AbstractVectorSpace, /) -> Any:
    return jax.tree.map(
        lambda spec: jnp.ones(spec.shape, dtype=spec.dtype),
        space.structure(),
    )


def _array_bound(
    bound: Any,
    value: Array,
    owner: str,
    /,
) -> Array:
    array = jnp.asarray(bound, dtype=value.dtype)
    if np.broadcast_shapes(array.shape, value.shape) != value.shape:
        raise ValueError(f"{owner} must be scalar or broadcast to shape {value.shape}.")
    return jnp.broadcast_to(array, value.shape)


class DirectCollocationScaling(StrictModule):
    """Positive physical scales used to normalize one collocation NLP."""

    state: Any
    control: Any
    parameters: Any
    dynamics: Any
    objective: Array
    duration: Array | None

    def __init__(
        self,
        *,
        state: Any = None,
        control: Any = None,
        parameters: Any = None,
        dynamics: Any = None,
        objective: ArrayLike = 1.0,
        duration: ArrayLike | None = None,
    ):
        self.state = state
        self.control = control
        self.parameters = parameters
        self.dynamics = dynamics
        objective_ = _positive(objective, "objective scale")
        if objective_.shape != ():
            raise ValueError("objective scale must be scalar.")
        self.objective = objective_
        if duration is None:
            self.duration = None
        else:
            duration_ = _positive(duration, "duration scale")
            if duration_.shape != ():
                raise ValueError("duration scale must be scalar.")
            self.duration = duration_


class DirectCollocationDerivativePolicy(StrictModule):
    """Sparse derivative compiler and optional exact Lagrangian Hessian."""

    compiler: SparseDerivativeCompiler = eqx.field(static=True)
    hessian: DirectCollocationHessianMode = eqx.field(static=True)
    chunk_size: int | None = eqx.field(static=True)
    verify: bool = eqx.field(static=True)
    num_verification_probes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        compiler: SparseDerivativeCompiler = "auto",
        hessian: DirectCollocationHessianMode = "limited-memory",
        chunk_size: int | None = None,
        verify: bool = True,
        num_verification_probes: int = 3,
    ):
        if compiler not in ("auto", "native", "asdex"):
            raise ValueError("compiler must be 'auto', 'native', or 'asdex'.")
        if hessian not in ("limited-memory", "exact-sparse"):
            raise ValueError("hessian must be 'limited-memory' or 'exact-sparse'.")
        chunk = None if chunk_size is None else int(chunk_size)
        if chunk is not None and chunk < 1:
            raise ValueError("chunk_size must be positive or None.")
        probes = int(num_verification_probes)
        if probes < 1:
            raise ValueError("num_verification_probes must be positive.")
        self.compiler = compiler
        self.hessian = hessian
        self.chunk_size = chunk
        self.verify = bool(verify)
        self.num_verification_probes = probes


class DirectCollocationAuditPolicy(StrictModule):
    """Physical feasibility thresholds and non-certifying interior checks."""

    defect_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    off_grid_points: int = eqx.field(static=True)
    audit_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        defect_tolerance: float = 1.0e-6,
        constraint_tolerance: float = 1.0e-6,
        off_grid_points: int = 2,
        audit_id: str = "control:direct-collocation:audit",
    ):
        defect = float(defect_tolerance)
        constraint = float(constraint_tolerance)
        points = int(off_grid_points)
        if any(not isfinite(value) or value < 0.0 for value in (defect, constraint)):
            raise ValueError("Audit tolerances must be finite and non-negative.")
        if points < 0:
            raise ValueError("off_grid_points must be non-negative.")
        self.defect_tolerance = defect
        self.constraint_tolerance = constraint
        self.off_grid_points = points
        self.audit_id = _identifier(audit_id, "audit_id")


class DirectCollocationPlan(StrictModule):
    """Static mesh, one-stage transcription, derivatives, and audit policy."""

    mesh: TemporalMesh
    method: ThetaMethod
    scaling: DirectCollocationScaling
    derivatives: DirectCollocationDerivativePolicy
    audit: DirectCollocationAuditPolicy
    variable_duration: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: TemporalMesh,
        /,
        *,
        method: ThetaMethod,
        variable_duration: bool = False,
        scaling: DirectCollocationScaling | None = None,
        derivatives: DirectCollocationDerivativePolicy | None = None,
        audit: DirectCollocationAuditPolicy | None = None,
        plan_id: str = "control:direct-collocation",
    ):
        if not isinstance(mesh, TemporalMesh) or mesh.role != "collocation":
            raise TypeError(
                "Direct collocation requires TemporalMesh(role='collocation')."
            )
        if not isinstance(method, ThetaMethod):
            raise TypeError("method must be a ThetaMethod.")
        supported = (method.theta == 1.0 and method.endpoint) or (
            method.theta == 0.5 and not method.endpoint
        )
        if not supported:
            raise ValueError(
                "Direct collocation supports endpoint backward Euler or midpoint theta."
            )
        scaling_ = DirectCollocationScaling() if scaling is None else scaling
        derivatives_ = (
            DirectCollocationDerivativePolicy() if derivatives is None else derivatives
        )
        audit_ = DirectCollocationAuditPolicy() if audit is None else audit
        if not isinstance(scaling_, DirectCollocationScaling):
            raise TypeError("scaling must be DirectCollocationScaling or None.")
        if not isinstance(derivatives_, DirectCollocationDerivativePolicy):
            raise TypeError(
                "derivatives must be DirectCollocationDerivativePolicy or None."
            )
        if not isinstance(audit_, DirectCollocationAuditPolicy):
            raise TypeError("audit must be DirectCollocationAuditPolicy or None.")
        self.mesh = mesh
        self.method = method
        self.scaling = scaling_
        self.derivatives = derivatives_
        self.audit = audit_
        self.variable_duration = bool(variable_duration)
        self.plan_id = _identifier(plan_id, "plan_id")


class DirectCollocationBounds(StrictModule):
    """Physical bounds grouped by direct-collocation decision role."""

    states: Bounds | None
    controls: Bounds | None
    parameters: Bounds | None
    duration: tuple[float, float] | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        states: Bounds | None = None,
        controls: Bounds | None = None,
        parameters: Bounds | None = None,
        duration: tuple[float, float] | None = None,
    ):
        for value, name in (
            (states, "states"),
            (controls, "controls"),
            (parameters, "parameters"),
        ):
            if value is not None and not isinstance(value, Bounds):
                raise TypeError(f"{name} must be Bounds or None.")
        if duration is not None:
            lower, upper = (float(value) for value in duration)
            if not 0.0 < lower <= upper or not isfinite(lower) or not isfinite(upper):
                raise ValueError("duration bounds must be finite, positive, and ordered.")
            duration_ = (lower, upper)
        else:
            duration_ = None
        self.states = states
        self.controls = controls
        self.parameters = parameters
        self.duration = duration_


class DirectCollocationDecision(StrictModule):
    """Physical trajectory, shared parameters, and optional physical duration."""

    states: Array
    controls: Array
    parameters: Any
    duration: Array | None


class DirectCollocationDecisionLayout(StrictModule):
    """Retraction-local normalized coordinates with exact role slices."""

    state_layout: StateLayout
    state_anchors: Array
    parameter_space: AbstractVectorSpace | None
    state_scale: Array
    control_scale: Array
    parameter_scale: Any
    duration_scale: Array
    state_slice: tuple[int, int] = eqx.field(static=True)
    control_slice: tuple[int, int] = eqx.field(static=True)
    parameter_slice: tuple[int, int] = eqx.field(static=True)
    duration_slice: tuple[int, int] = eqx.field(static=True)
    state_array_shape: tuple[int, ...] = eqx.field(static=True)
    state_coordinate_shape: tuple[int, ...] = eqx.field(static=True)
    local_shape: tuple[int, ...] = eqx.field(static=True)
    tangent_shape: tuple[int, ...] = eqx.field(static=True)
    control_array_shape: tuple[int, ...] = eqx.field(static=True)
    variable_duration: bool = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_layout: StateLayout,
        state_anchors: ArrayLike,
        state_array_shape: Sequence[int],
        control_array_shape: Sequence[int],
        parameter_space: AbstractVectorSpace | None,
        state_scale: Array,
        control_scale: Array,
        parameter_scale: Any,
        duration_scale: Array,
        variable_duration: bool,
        layout_id: str,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        geometry = state_layout.geometry
        if (
            not geometry.supports_exact_inverse
            or not geometry.supports_exact_differential
        ):
            raise ValueError(
                "DirectCollocationDecisionLayout requires exact inverse-retraction "
                "and retraction-differential geometry."
            )
        states = tuple(int(size) for size in state_array_shape)
        controls = tuple(int(size) for size in control_array_shape)
        point_rank = len(state_layout.shape)
        if point_rank and states[-point_rank:] != state_layout.shape:
            raise ValueError("state_array_shape must end with state_layout.shape.")
        state_prefix = states[:-point_rank] if point_rank else states
        anchors = _inexact(state_anchors, "state_anchors")
        if anchors.shape != states:
            raise ValueError("state_anchors must have state_array_shape.")
        flat_anchors = anchors.reshape((-1,) + state_layout.shape)
        local_template = jnp.asarray(
            state_layout.geometry.inverse_retract(flat_anchors[0], flat_anchors[0])
        )
        if local_template.size != state_layout.local_size:
            raise ValueError(
                "State geometry local output size must match state_layout.local_size."
            )
        local_shape = tuple(local_template.shape)
        tangent_template = jnp.asarray(
            state_layout.geometry.retraction_jvp(
                flat_anchors[0],
                local_template,
                jnp.zeros_like(local_template),
            )
        )
        if tangent_template.size != state_layout.tangent_size:
            raise ValueError(
                "State geometry tangent output size must match state_layout.tangent_size."
            )
        state_coordinates = state_prefix + (state_layout.local_size,)
        state_count = prod(state_coordinates)
        control_count = prod(controls)
        parameter_count = 0 if parameter_space is None else parameter_space.size
        duration_count = int(variable_duration)
        state_stop = state_count
        control_stop = state_stop + control_count
        parameter_stop = control_stop + parameter_count
        duration_stop = parameter_stop + duration_count
        state_scale_ = _inexact(state_scale, "state scale")
        if not (
            state_scale_.shape == state_coordinates
            or jnp.broadcast_shapes(state_scale_.shape, state_coordinates)
            == state_coordinates
        ):
            raise ValueError("state_scale must broadcast to local state coordinates.")
        self.state_layout = state_layout
        self.state_anchors = anchors
        self.parameter_space = parameter_space
        self.state_scale = jnp.broadcast_to(state_scale_, state_coordinates)
        self.control_scale = control_scale
        self.parameter_scale = parameter_scale
        self.duration_scale = duration_scale
        self.state_slice = (0, state_stop)
        self.control_slice = (state_stop, control_stop)
        self.parameter_slice = (control_stop, parameter_stop)
        self.duration_slice = (parameter_stop, duration_stop)
        self.state_array_shape = states
        self.state_coordinate_shape = state_coordinates
        self.local_shape = local_shape
        self.tangent_shape = tuple(tangent_template.shape)
        self.control_array_shape = controls
        self.variable_duration = bool(variable_duration)
        self.num_variables = duration_stop
        self.layout_id = _identifier(layout_id, "layout_id")

    def _state_coordinates(self, states: Array, /) -> Array:
        geometry = self.state_layout.geometry
        if geometry.trivial:
            return states.reshape(self.state_coordinate_shape)
        flat_states = states.reshape((-1,) + self.state_layout.shape)
        flat_anchors = self.state_anchors.reshape((-1,) + self.state_layout.shape)
        return jax.vmap(
            lambda anchor, point: jnp.asarray(
                geometry.inverse_retract(anchor, point)
            ).reshape((self.state_layout.local_size,))
        )(flat_anchors, flat_states).reshape(self.state_coordinate_shape)

    def _states(self, coordinates: Array, /) -> Array:
        geometry = self.state_layout.geometry
        if geometry.trivial:
            return coordinates.reshape(self.state_array_shape)
        flat_local = coordinates.reshape((-1, self.state_layout.local_size))
        flat_anchors = self.state_anchors.reshape((-1,) + self.state_layout.shape)
        return jax.vmap(
            lambda anchor, local: jnp.asarray(
                geometry.retract(anchor, local.reshape(self.local_shape))
            )
        )(flat_anchors, flat_local).reshape(self.state_array_shape)

    def pack(self, decision: DirectCollocationDecision, /) -> Array:
        if not isinstance(decision, DirectCollocationDecision):
            raise TypeError("decision must be a DirectCollocationDecision.")
        states = _inexact(decision.states, "decision states")
        controls = _inexact(decision.controls, "decision controls")
        if states.shape != self.state_array_shape:
            raise ValueError(f"decision states must have shape {self.state_array_shape}.")
        if controls.shape != self.control_array_shape:
            raise ValueError(
                f"decision controls must have shape {self.control_array_shape}."
            )
        state_coordinates = self._state_coordinates(states)
        parts = [
            (state_coordinates / self.state_scale.astype(states.dtype)).reshape((-1,)),
            (controls / self.control_scale.astype(controls.dtype)).reshape((-1,)),
        ]
        if self.parameter_space is None:
            if decision.parameters is not None:
                raise ValueError("decision parameters require a parameter_space.")
        else:
            parameters = self.parameter_space.validate(decision.parameters)
            parameter_scale = self.parameter_space.validate(self.parameter_scale)
            parts.append(
                self.parameter_space.flatten(
                    jax.tree.map(
                        lambda value, scale: value / scale, parameters, parameter_scale
                    )
                )
            )
        if self.variable_duration:
            if decision.duration is None:
                raise ValueError("A variable-duration decision requires duration.")
            duration = _positive(decision.duration, "decision duration")
            if duration.shape != ():
                raise ValueError("decision duration must be scalar.")
            parts.append(jnp.log(duration / self.duration_scale).reshape((1,)))
        elif decision.duration is not None:
            raise ValueError("A fixed-duration decision must not contain duration.")
        dtype = jnp.result_type(*parts)
        return jnp.concatenate(tuple(part.astype(dtype) for part in parts))

    def unpack(self, coordinates: ArrayLike, /) -> DirectCollocationDecision:
        vector = _inexact(coordinates, "decision coordinates")
        if vector.shape != (self.num_variables,):
            raise ValueError(
                f"decision coordinates must have shape {(self.num_variables,)}."
            )
        state_start, state_stop = self.state_slice
        control_start, control_stop = self.control_slice
        parameter_start, parameter_stop = self.parameter_slice
        duration_start, duration_stop = self.duration_slice
        state_coordinates = vector[state_start:state_stop].reshape(
            self.state_coordinate_shape
        )
        state_coordinates = state_coordinates * self.state_scale.astype(vector.dtype)
        states = self._states(state_coordinates)
        controls = vector[control_start:control_stop].reshape(self.control_array_shape)
        controls = controls * self.control_scale.astype(vector.dtype)
        if self.parameter_space is None:
            parameters = None
        else:
            normalized = self.parameter_space.unflatten(
                vector[parameter_start:parameter_stop]
            )
            parameters = jax.tree.map(
                lambda value, scale: value * scale,
                normalized,
                self.parameter_scale,
            )
        duration = (
            self.duration_scale * jnp.exp(vector[duration_start:duration_stop][0])
            if self.variable_duration
            else None
        )
        return DirectCollocationDecision(states, controls, parameters, duration)

    def coordinate_bounds(
        self,
        decision: DirectCollocationDecision,
        bounds: DirectCollocationBounds,
        /,
    ) -> tuple[Array, Array]:
        coordinates = self.pack(decision)
        lower = jnp.full_like(coordinates, -jnp.inf)
        upper = jnp.full_like(coordinates, jnp.inf)
        if bounds.states is not None:
            if not self.state_layout.geometry.trivial:
                raise ValueError(
                    "Pointwise state bounds are not coordinate bounds on a "
                    "non-Euclidean state manifold; use path constraints."
                )
            state_lower, state_upper = bounds.states.materialize(decision.states)
            start, stop = self.state_slice
            lower = lower.at[start:stop].set(
                (
                    state_lower.reshape(self.state_coordinate_shape) / self.state_scale
                ).reshape((-1,))
            )
            upper = upper.at[start:stop].set(
                (
                    state_upper.reshape(self.state_coordinate_shape) / self.state_scale
                ).reshape((-1,))
            )
        if bounds.controls is not None:
            control_lower, control_upper = bounds.controls.materialize(decision.controls)
            start, stop = self.control_slice
            lower = lower.at[start:stop].set(
                (control_lower / self.control_scale).reshape((-1,))
            )
            upper = upper.at[start:stop].set(
                (control_upper / self.control_scale).reshape((-1,))
            )
        if bounds.parameters is not None:
            if self.parameter_space is None:
                raise ValueError("parameter bounds require a parameter_space.")
            parameter_lower, parameter_upper = bounds.parameters.materialize(
                decision.parameters
            )
            normalized_lower = jax.tree.map(
                lambda value, scale: value / scale,
                parameter_lower,
                self.parameter_scale,
            )
            normalized_upper = jax.tree.map(
                lambda value, scale: value / scale,
                parameter_upper,
                self.parameter_scale,
            )
            start, stop = self.parameter_slice
            lower = lower.at[start:stop].set(
                self.parameter_space.flatten(normalized_lower)
            )
            upper = upper.at[start:stop].set(
                self.parameter_space.flatten(normalized_upper)
            )
        if bounds.duration is not None:
            if not self.variable_duration:
                raise ValueError("duration bounds require variable_duration=True.")
            start, stop = self.duration_slice
            duration_lower, duration_upper = bounds.duration
            lower = lower.at[start:stop].set(
                jnp.log(jnp.asarray(duration_lower) / self.duration_scale)
            )
            upper = upper.at[start:stop].set(
                jnp.log(jnp.asarray(duration_upper) / self.duration_scale)
            )
        return lower, upper


class DirectCollocationConstraintLayout(StrictModule):
    """Canonical constraint slices, scaled bounds, and scalar provenance."""

    lower: Array
    upper: Array
    scale: Array
    dynamics_slice: tuple[int, int] = eqx.field(static=True)
    initial_slice: tuple[int, int] = eqx.field(static=True)
    path_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    trajectory_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    sources: tuple[str, ...] = eqx.field(static=True)
    num_constraints: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)


class _DirectValues(StrictModule):
    """Direct residuals using observed-minus-predicted orientation."""

    decision: DirectCollocationDecision
    times: Array
    stage_times: Array
    stage_states: Array
    state_rates: Array
    view: TrajectoryOptimizationView
    objective: Array
    dynamics: Array
    initial: Array
    path: tuple[Array, ...]
    trajectory: tuple[Array, ...]


def _callback_args(
    problem: TrajectoryOptimizationProblem,
    plan: DirectCollocationPlan,
    decision: DirectCollocationDecision,
    args: Any,
    /,
) -> Any:
    if problem.parameter_space is None and not plan.variable_duration:
        return args
    duration = (
        decision.duration
        if plan.variable_duration
        else jnp.asarray(plan.mesh.duration, dtype=decision.states.dtype)
    )
    return TrajectoryOptimizationContext(args, decision.parameters, duration)


def _physical_times(
    plan: DirectCollocationPlan, decision: DirectCollocationDecision, /
) -> Array:
    if not plan.variable_duration:
        return plan.mesh.nodes.astype(decision.states.dtype)
    assert decision.duration is not None
    span = plan.mesh.duration.astype(decision.states.dtype)
    return plan.mesh.t0 + (plan.mesh.nodes - plan.mesh.t0) * decision.duration / span


def _evaluate_values(
    problem: TrajectoryOptimizationProblem,
    plan: DirectCollocationPlan,
    layout: DirectCollocationDecisionLayout,
    coordinates: Array,
    args: Any,
    /,
) -> _DirectValues:
    decision = layout.unpack(coordinates)
    times = _physical_times(plan, decision)
    axis = len(problem.case_shape)
    steps = plan.mesh.num_steps
    indices = jnp.arange(steps)
    left = jnp.take(decision.states, indices, axis=axis)
    right = jnp.take(decision.states, indices + 1, axis=axis)
    widths = jnp.diff(times)
    theta = plan.method.theta
    stage_times = (1.0 - theta) * times[:-1] + theta * times[1:]
    callback_args = _callback_args(problem, plan, decision, args)
    case_count = prod(problem.case_shape) if problem.case_shape else 1
    flat_count = case_count * steps
    flat_left = left.reshape((flat_count,) + problem.state_shape)
    flat_right = right.reshape((flat_count,) + problem.state_shape)
    flat_widths = jnp.broadcast_to(widths, problem.case_shape + (steps,)).reshape(
        (flat_count,)
    )
    geometry = problem.state_layout.geometry
    local_size = problem.state_layout.local_size
    flat_endpoint_local = jax.vmap(
        lambda anchor, point: jnp.asarray(
            geometry.inverse_retract(anchor, point)
        ).reshape((local_size,))
    )(flat_left, flat_right)
    flat_stage_local = theta * flat_endpoint_local
    flat_states = jax.vmap(
        lambda anchor, local: jnp.asarray(
            geometry.retract(anchor, local.reshape(layout.local_shape))
        )
    )(flat_left, flat_stage_local)
    flat_local_rates = flat_endpoint_local / flat_widths[:, None]
    flat_rates = jax.vmap(
        lambda anchor, local, velocity: jnp.asarray(
            geometry.retraction_jvp(
                anchor,
                local.reshape(layout.local_shape),
                velocity.reshape(layout.local_shape),
            )
        )
    )(flat_left, flat_stage_local, flat_local_rates)
    stage_states = flat_states.reshape(
        problem.case_shape + (steps,) + problem.state_shape
    )
    state_rates = flat_rates.reshape(problem.case_shape + (steps,) + layout.tangent_shape)
    flat_times = jnp.broadcast_to(
        stage_times,
        problem.case_shape + (steps,),
    ).reshape((flat_count,))
    flat_controls = decision.controls.reshape((flat_count,) + problem.control_shape)

    dynamics_model = problem.dynamics
    if isinstance(dynamics_model, ContinuousSystem):

        def physical_residual(time, state, state_rate, control):
            field = dynamics_model.evaluate(time, state, callback_args, inputs=control)
            return state_rate - geometry.project_tangent(state, field)

    else:

        def physical_residual(time, state, state_rate, control):
            residual = dynamics_model.evaluate(
                time,
                state,
                state_rate,
                callback_args,
                inputs=control,
            )
            return geometry.project_tangent(state, residual)

    flat_physical_defects = jax.vmap(physical_residual)(
        flat_times, flat_states, flat_rates, flat_controls
    )
    flat_dynamics = jax.vmap(
        lambda anchor, point, tangent: jnp.asarray(
            geometry.retraction_inverse_jvp(anchor, point, tangent)
        ).reshape((local_size,))
    )(flat_left, flat_states, flat_physical_defects)
    dynamics = flat_dynamics.reshape(problem.case_shape + (steps, local_size))
    view = TrajectoryOptimizationView(
        times,
        decision.states,
        decision.controls,
        case_shape=problem.case_shape,
        state_shape=problem.state_shape,
        control_shape=problem.control_shape,
        state_geometry=geometry,
    )
    objective = jnp.asarray(0.0, dtype=coordinates.dtype)
    running_cost = problem.running_cost
    if running_cost is not None:

        def running(time, state, control):
            value = jnp.asarray(running_cost(time, state, control, callback_args))
            if value.shape != ():
                raise ValueError("RunningCost must return one scalar per stage.")
            return value

        running_values = jax.vmap(running)(
            flat_times, flat_states, flat_controls
        ).reshape(problem.case_shape + (steps,))
        objective = objective + jnp.sum(
            running_values * jnp.broadcast_to(widths, problem.case_shape + (steps,))
        )
    terminal_cost = problem.terminal_cost
    if terminal_cost is not None:
        final_states = view.final_state.reshape((case_count,) + problem.state_shape)

        def terminal(state):
            value = jnp.asarray(terminal_cost(times[-1], state, callback_args))
            if value.shape != ():
                raise ValueError("TerminalCost must return one scalar per case.")
            return value

        objective = objective + jnp.sum(jax.vmap(terminal)(final_states))
    if problem.trajectory_cost is not None:
        trajectory_value = jnp.asarray(problem.trajectory_cost(view, callback_args))
        if trajectory_value.shape != ():
            raise ValueError("trajectory_cost must return one scalar.")
        objective = objective + trajectory_value
    if problem.initial_state is None:
        initial = jnp.empty((0,), dtype=coordinates.dtype)
    else:
        flat_initial = problem.initial_state.reshape((case_count,) + problem.state_shape)
        flat_first = view.initial_state.reshape((case_count,) + problem.state_shape)
        initial = jax.vmap(
            lambda anchor, point: jnp.asarray(
                geometry.inverse_retract(anchor, point)
            ).reshape((local_size,))
        )(flat_initial, flat_first).reshape(problem.case_shape + (local_size,))
    path_values = []
    for constraint in problem.path_constraints:

        def evaluate_path(time, state, control, callback=constraint):
            return callback(time, state, control, callback_args)

        values = jax.vmap(evaluate_path)(flat_times, flat_states, flat_controls)
        path_values.append(
            values.reshape(problem.case_shape + (steps,) + values.shape[1:])
        )
    trajectory_values = tuple(
        constraint(view, callback_args) for constraint in problem.trajectory_constraints
    )
    return _DirectValues(
        decision,
        times,
        stage_times,
        stage_states,
        state_rates,
        view,
        objective,
        dynamics,
        initial,
        tuple(path_values),
        trajectory_values,
    )


def _raw_constraints(values: _DirectValues, /) -> Array:
    parts = [values.dynamics.reshape((-1,)), values.initial.reshape((-1,))]
    parts.extend(value.reshape((-1,)) for value in values.path)
    parts.extend(value.reshape((-1,)) for value in values.trajectory)
    return jnp.concatenate(tuple(parts))


def _constraint_layout(
    problem: TrajectoryOptimizationProblem,
    plan: DirectCollocationPlan,
    sample: _DirectValues,
    dynamics_scale: Array,
    state_scale: Array,
    /,
) -> DirectCollocationConstraintLayout:
    lower_parts = []
    upper_parts = []
    scale_parts = []
    sources = []
    cursor = 0

    dynamics_size = int(sample.dynamics.size)
    dynamics_slice = (cursor, cursor + dynamics_size)
    cursor += dynamics_size
    lower_parts.append(jnp.zeros((dynamics_size,), dtype=sample.objective.dtype))
    upper_parts.append(jnp.zeros((dynamics_size,), dtype=sample.objective.dtype))
    full_dynamics_scale = jnp.broadcast_to(dynamics_scale, sample.dynamics.shape)
    scale_parts.append(full_dynamics_scale.reshape((-1,)))
    sources.extend(f"dynamics:{index}" for index in range(dynamics_size))

    initial_size = int(sample.initial.size)
    initial_slice = (cursor, cursor + initial_size)
    cursor += initial_size
    lower_parts.append(jnp.zeros((initial_size,), dtype=sample.objective.dtype))
    upper_parts.append(jnp.zeros((initial_size,), dtype=sample.objective.dtype))
    initial_scale = (
        jnp.empty((0,), dtype=sample.objective.dtype)
        if initial_size == 0
        else jnp.broadcast_to(state_scale, sample.initial.shape).reshape((-1,))
    )
    scale_parts.append(initial_scale)
    sources.extend(f"initial:{index}" for index in range(initial_size))

    path_slices = []
    for constraint, value in zip(problem.path_constraints, sample.path, strict=True):
        size = int(value.size)
        path_slices.append((cursor, cursor + size))
        cursor += size
        lower_parts.append(
            _array_bound(
                constraint.lower, value, f"{constraint.constraint_id} lower"
            ).reshape((-1,))
        )
        upper_parts.append(
            _array_bound(
                constraint.upper, value, f"{constraint.constraint_id} upper"
            ).reshape((-1,))
        )
        scale_parts.append(jnp.broadcast_to(constraint.scale, value.shape).reshape((-1,)))
        sources.extend(
            f"path:{constraint.constraint_id}:{index}" for index in range(size)
        )

    trajectory_slices = []
    for constraint, value in zip(
        problem.trajectory_constraints, sample.trajectory, strict=True
    ):
        size = int(value.size)
        trajectory_slices.append((cursor, cursor + size))
        cursor += size
        lower_parts.append(
            _array_bound(
                constraint.lower, value, f"{constraint.constraint_id} lower"
            ).reshape((-1,))
        )
        upper_parts.append(
            _array_bound(
                constraint.upper, value, f"{constraint.constraint_id} upper"
            ).reshape((-1,))
        )
        scale_parts.append(jnp.broadcast_to(constraint.scale, value.shape).reshape((-1,)))
        sources.extend(
            f"trajectory:{constraint.constraint_id}:{index}" for index in range(size)
        )

    lower = jnp.concatenate(tuple(lower_parts))
    upper = jnp.concatenate(tuple(upper_parts))
    scale = jnp.concatenate(tuple(scale_parts))
    return DirectCollocationConstraintLayout(
        lower,
        upper,
        scale,
        dynamics_slice=dynamics_slice,
        initial_slice=initial_slice,
        path_slices=tuple(path_slices),
        trajectory_slices=tuple(trajectory_slices),
        sources=tuple(sources),
        num_constraints=cursor,
        layout_id=canonical_fingerprint(
            {
                "kind": "direct-collocation-constraints",
                "problem": problem.problem_id,
                "plan": plan.plan_id,
                "sizes": [dynamics_size, initial_size]
                + [int(value.size) for value in sample.path]
                + [int(value.size) for value in sample.trajectory],
            }
        ),
    )


def _trajectory_problem(
    problem: TrajectoryOptimizationProblem | ControlProblem,
    plan: DirectCollocationPlan,
    /,
) -> TrajectoryOptimizationProblem:
    if isinstance(problem, TrajectoryOptimizationProblem):
        return problem
    if not isinstance(problem, ControlProblem):
        raise TypeError(
            "problem must be TrajectoryOptimizationProblem or ControlProblem."
        )
    if not isinstance(problem.dynamics, DifferentialControlDynamics):
        raise TypeError(
            "ControlProblem direct collocation requires differential dynamics."
        )
    if plan.variable_duration:
        raise ValueError(
            "Variable-duration direct collocation requires TrajectoryOptimizationProblem."
        )
    if not np.array_equal(
        np.asarray(problem.time_grid.times), np.asarray(plan.mesh.nodes)
    ):
        raise ValueError(
            "ControlProblem time grid must exactly match the collocation mesh."
        )
    path = tuple(
        BoundedPathConstraint(
            constraint,
            upper=0.0,
            constraint_id=f"control-path:{index}",
        )
        for index, constraint in enumerate(problem.path_constraints)
    )

    def terminal_constraint(callback, index):
        def evaluate(view, args):
            case_count = prod(view.case_shape) if view.case_shape else 1
            flat_states = view.final_state.reshape((case_count,) + view.state_shape)
            values = jax.vmap(lambda state: callback(view.times[-1], state, args))(
                flat_states
            )
            return values.reshape(view.case_shape + values.shape[1:])

        return BoundedTrajectoryConstraint(
            evaluate,
            upper=0.0,
            constraint_id=f"control-terminal:{index}",
        )

    terminal = tuple(
        terminal_constraint(constraint, index)
        for index, constraint in enumerate(problem.terminal_constraints)
    )
    return TrajectoryOptimizationProblem(
        problem.dynamics.system,
        initial_state=problem.initial_state,
        running_cost=problem.running_cost,
        terminal_cost=problem.terminal_cost,
        path_constraints=path,
        trajectory_constraints=terminal,
        args=problem.args,
        problem_id=problem.problem_id,
    )


class DirectCollocationCompilation(StrictModule):
    """Typed direct transcription plus dense and sparse nonlinear program views."""

    problem: TrajectoryOptimizationProblem
    plan: DirectCollocationPlan
    decision_layout: DirectCollocationDecisionLayout
    constraint_layout: DirectCollocationConstraintLayout
    initial_decision: DirectCollocationDecision
    initial_coordinates: Array
    coordinate_lower: Array
    coordinate_upper: Array
    dynamics_scale: Array
    bounds: DirectCollocationBounds
    minimization_problem: MinimizationProblem
    structured_program: StructuredNonlinearProgram
    structured_template: StructuredNonlinearTemplate
    prepared_structured_program: PreparedStructuredNonlinearProgram
    jacobian_verification: SparseDerivativeVerification | None
    hessian_verification: SparseDerivativeVerification | None
    compilation_id: str = eqx.field(static=True)

    def values(
        self, coordinates: ArrayLike, args: Any = _DEFAULT_ARGS, /
    ) -> _DirectValues:
        runtime_args = self.problem.args if args is _DEFAULT_ARGS else args
        return _evaluate_values(
            self.problem,
            self.plan,
            self.decision_layout,
            self.decision_layout.pack(self.decision_layout.unpack(coordinates)),
            runtime_args,
        )

    def decode(self, coordinates: ArrayLike, /) -> DirectCollocationDecision:
        return self.decision_layout.unpack(coordinates)


class PreparedDirectCollocation(StrictModule):
    """Compiled direct transcription bound to one explicit optimization method."""

    compilation: DirectCollocationCompilation
    method: AbstractMinimizationMethod
    termination: OptimizationTermination
    structured_program: PreparedStructuredNonlinearProgram
    prepared_id: str = eqx.field(static=True)


class DirectCollocationOffGridAudit(StrictModule):
    """Per-interval sampled defect evidence for the declared interpolant."""

    times: Array
    dynamics_residuals: Array
    interval_defects: Array
    interval_path_violations: Array
    maximum_defect: Array
    maximum_path_violation: Array
    finite: Array
    approximation_id: str = eqx.field(static=True)
    audit_id: str = eqx.field(static=True)
    certified: bool = eqx.field(static=True)


class DirectCollocationDiagnostics(StrictModule):
    """Physical feasibility, off-grid defects, and sparse topology evidence."""

    maximum_defect: Array
    maximum_constraint_violation: Array
    maximum_off_grid_defect: Array
    maximum_off_grid_path_violation: Array
    off_grid: DirectCollocationOffGridAudit
    jacobian_nonzeros: int = eqx.field(static=True)
    hessian_nonzeros: int = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_constraints: int = eqx.field(static=True)
    off_grid_certified: bool = eqx.field(static=True)


class DirectCollocationResult(StrictModule):
    """Decoded trajectory, optimizer certificate, and physical collocation audit."""

    decision: DirectCollocationDecision
    trajectory: ControlTrajectory
    stage_times: Array
    stage_states: Array
    state_rates: Array
    dynamics_defects: Array
    initial_defect: Array
    path_residuals: tuple[Array, ...]
    trajectory_residuals: tuple[Array, ...]
    objective: Array
    optimization_result: MinimizationResult
    structured_result: StructuredNonlinearResult | None
    diagnostics: DirectCollocationDiagnostics
    status: Array
    compilation: DirectCollocationCompilation
    result_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == DIRECT_COLLOCATION_SUCCESS

    @property
    def parameters(self) -> Any:
        return self.decision.parameters

    @property
    def duration(self) -> Array:
        if self.decision.duration is not None:
            return self.decision.duration
        return self.trajectory.time_grid.t1 - self.trajectory.time_grid.t0


class PooledDirectCollocationResult(StrictModule):
    """Input-ordered direct-collocation results plus pool evidence."""

    results: tuple[DirectCollocationResult, ...]
    evidence: StructuredPoolEvidence

    @property
    def successful(self) -> Array:
        return jnp.stack(tuple(result.successful for result in self.results))


def _resolved_scales(
    problem: TrajectoryOptimizationProblem,
    plan: DirectCollocationPlan,
    parameter_guess: Any,
    /,
) -> tuple[Array, Array, Any, Array, Array]:
    local_size = problem.state_layout.local_size
    if isinstance(problem.dynamics, DifferentialAlgebraicSystem):
        default_state = problem.dynamics.state_scale
        default_dynamics = problem.dynamics.residual_scale
    else:
        default_state = jnp.ones(problem.state_shape)
        default_dynamics = jnp.ones(problem.state_shape)
    if problem.state_layout.geometry.trivial:
        state = _broadcast_scale(
            default_state if plan.scaling.state is None else plan.scaling.state,
            problem.state_shape,
            "state scale",
        ).reshape((local_size,))
        dynamics = _broadcast_scale(
            default_dynamics if plan.scaling.dynamics is None else plan.scaling.dynamics,
            problem.state_shape,
            "dynamics scale",
        ).reshape((local_size,))
    else:
        state = _broadcast_scale(
            plan.scaling.state,
            (local_size,),
            "local state scale",
        )
        dynamics = _broadcast_scale(
            plan.scaling.dynamics,
            (local_size,),
            "local dynamics scale",
        )
    control = _broadcast_scale(
        plan.scaling.control,
        problem.control_shape,
        "control scale",
    )
    if problem.parameter_space is None:
        parameter = None
    else:
        if parameter_guess is None:
            raise ValueError(
                "parameter_guess is required for the declared parameter_space."
            )
        parameter = (
            _space_ones(problem.parameter_space)
            if plan.scaling.parameters is None
            else problem.parameter_space.validate(plan.scaling.parameters)
        )
        parameter = jax.tree.map(
            lambda value: _positive(value, "parameter scale"), parameter
        )
    duration = (
        jnp.asarray(plan.mesh.duration)
        if plan.scaling.duration is None
        else plan.scaling.duration
    )
    duration = _positive(duration, "duration scale")
    return state, control, parameter, dynamics, duration


def compile_direct_collocation(
    problem: TrajectoryOptimizationProblem | ControlProblem,
    plan: DirectCollocationPlan,
    initial_states: ArrayLike,
    initial_controls: ArrayLike,
    /,
    *,
    parameter_guess: Any = None,
    duration_guess: ArrayLike | None = None,
    bounds: DirectCollocationBounds | None = None,
) -> DirectCollocationCompilation:
    """Compile one fixed-topology direct transcription and exact sparse derivatives."""
    if not isinstance(plan, DirectCollocationPlan):
        raise TypeError("plan must be a DirectCollocationPlan.")
    trajectory_problem = _trajectory_problem(problem, plan)
    geometry = trajectory_problem.state_layout.geometry
    if not geometry.supports_exact_inverse or not geometry.supports_exact_differential:
        raise ValueError(
            "Direct collocation requires exact inverse-retraction and "
            "retraction-differential geometry."
        )
    states = _inexact(initial_states, "initial_states")
    controls = _inexact(initial_controls, "initial_controls")
    expected_states = (
        trajectory_problem.case_shape
        + (plan.mesh.num_nodes,)
        + trajectory_problem.state_shape
    )
    expected_controls = (
        trajectory_problem.case_shape
        + (plan.mesh.num_steps,)
        + trajectory_problem.control_shape
    )
    if states.shape != expected_states:
        raise ValueError(
            f"initial_states must have shape {expected_states}; got {states.shape}."
        )
    if controls.shape != expected_controls:
        raise ValueError(
            f"initial_controls must have shape {expected_controls}; got {controls.shape}."
        )
    if states.dtype != controls.dtype:
        raise TypeError("initial_states and initial_controls must have the same dtype.")
    if trajectory_problem.parameter_space is None:
        if parameter_guess is not None:
            raise ValueError("parameter_guess requires a declared parameter_space.")
        parameters = None
    else:
        parameters = trajectory_problem.parameter_space.validate(parameter_guess)
        leaves = jax.tree.leaves(parameters)
        if any(leaf.dtype != states.dtype for leaf in leaves):
            raise TypeError("parameter leaves must match the trajectory dtype.")
    if plan.variable_duration:
        if duration_guess is None:
            raise ValueError("duration_guess is required for variable duration.")
        duration = _positive(duration_guess, "duration_guess").astype(states.dtype)
        if duration.shape != ():
            raise ValueError("duration_guess must be scalar.")
    else:
        if duration_guess is not None:
            raise ValueError("duration_guess requires variable_duration=True.")
        duration = None
    state_scale, control_scale, parameter_scale, dynamics_scale, duration_scale = (
        _resolved_scales(trajectory_problem, plan, parameters)
    )
    state_scale_array = jnp.broadcast_to(
        state_scale,
        (1,) * (len(trajectory_problem.case_shape) + 1)
        + (trajectory_problem.state_layout.local_size,),
    )
    control_scale_array = jnp.broadcast_to(
        control_scale,
        (1,) * (len(trajectory_problem.case_shape) + 1)
        + trajectory_problem.control_shape,
    )
    layout = DirectCollocationDecisionLayout(
        state_layout=trajectory_problem.state_layout,
        state_anchors=states,
        state_array_shape=expected_states,
        control_array_shape=expected_controls,
        parameter_space=trajectory_problem.parameter_space,
        state_scale=state_scale_array,
        control_scale=control_scale_array,
        parameter_scale=parameter_scale,
        duration_scale=duration_scale,
        variable_duration=plan.variable_duration,
        layout_id=canonical_fingerprint(
            {
                "kind": "direct-collocation-decision",
                "problem": trajectory_problem.problem_id,
                "plan": plan.plan_id,
                "states": list(expected_states),
                "controls": list(expected_controls),
                "state_layout": trajectory_problem.state_layout.layout_id,
                "local_size": trajectory_problem.state_layout.local_size,
                "parameters": (
                    None
                    if trajectory_problem.parameter_space is None
                    else trajectory_problem.parameter_space.space_id
                ),
                "variable_duration": plan.variable_duration,
            }
        ),
    )
    initial_decision = DirectCollocationDecision(states, controls, parameters, duration)
    coordinates = layout.pack(initial_decision)
    resolved_bounds = DirectCollocationBounds() if bounds is None else bounds
    if not isinstance(resolved_bounds, DirectCollocationBounds):
        raise TypeError("bounds must be DirectCollocationBounds or None.")
    coordinate_lower, coordinate_upper = layout.coordinate_bounds(
        initial_decision, resolved_bounds
    )
    if bool(
        np.any(np.asarray(coordinates) < np.asarray(coordinate_lower))
        or np.any(np.asarray(coordinates) > np.asarray(coordinate_upper))
    ):
        raise ValueError("The initial direct-collocation decision violates its bounds.")
    sample = _evaluate_values(
        trajectory_problem,
        plan,
        layout,
        coordinates,
        trajectory_problem.args,
    )
    constraint_layout = _constraint_layout(
        trajectory_problem,
        plan,
        sample,
        dynamics_scale,
        state_scale,
    )

    def physical_objective(value, runtime_args):
        return _evaluate_values(
            trajectory_problem, plan, layout, value, runtime_args
        ).objective

    def objective(value, runtime_args):
        return physical_objective(value, runtime_args) / plan.scaling.objective

    def raw_constraints(value, runtime_args):
        return _raw_constraints(
            _evaluate_values(trajectory_problem, plan, layout, value, runtime_args)
        )

    def scaled_constraints(value, runtime_args):
        return raw_constraints(value, runtime_args) / constraint_layout.scale

    scaled_lower = constraint_layout.lower / constraint_layout.scale
    scaled_upper = constraint_layout.upper / constraint_layout.scale
    dense_problem = MinimizationProblem(
        objective,
        bounds=Bounds(coordinate_lower, coordinate_upper),
        constraints=(
            NonlinearConstraint(
                scaled_constraints,
                lower=scaled_lower,
                upper=scaled_upper,
                constraint_id=f"{trajectory_problem.problem_id}:direct-collocation",
            ),
        ),
        problem_id=f"{trajectory_problem.problem_id}:direct-collocation",
    )
    source = ArraySpace((layout.num_variables,), dtype=coordinates.dtype)
    target = ArraySpace((constraint_layout.num_constraints,), dtype=coordinates.dtype)
    jacobian = compile_sparse_jacobian(
        scaled_constraints,
        coordinates,
        source=source,
        target=target,
        sample_args=trajectory_problem.args,
        compiler=plan.derivatives.compiler,
        chunk_size=plan.derivatives.chunk_size,
        plan_id=f"{trajectory_problem.problem_id}:direct-collocation:jacobian",
    )
    hessian: SparseDerivativePlan | None = None
    if plan.derivatives.hessian == "exact-sparse":

        def lagrangian(value, packed_args):
            runtime_args, objective_factor, multipliers = packed_args
            return objective_factor * objective(value, runtime_args) + jnp.vdot(
                multipliers, scaled_constraints(value, runtime_args)
            )

        hessian = compile_sparse_hessian(
            lagrangian,
            coordinates,
            space=source,
            sample_args=(
                trajectory_problem.args,
                jnp.asarray(1.0, dtype=coordinates.dtype),
                jnp.zeros((constraint_layout.num_constraints,), dtype=coordinates.dtype),
            ),
            compiler=plan.derivatives.compiler,
            chunk_size=plan.derivatives.chunk_size,
            plan_id=f"{trajectory_problem.problem_id}:direct-collocation:hessian",
        )
    jacobian_verification = None
    hessian_verification = None
    if plan.derivatives.verify:
        jacobian_verification = verify_sparse_derivative(
            jacobian,
            coordinates,
            key=jr.key(0),
            args=trajectory_problem.args,
            num_probes=plan.derivatives.num_verification_probes,
        )
        if not bool(np.asarray(jacobian_verification.passed)):
            raise ValueError("Compiled direct-collocation Jacobian failed verification.")
        if hessian is not None:
            hessian_verification = verify_sparse_derivative(
                hessian,
                coordinates,
                key=jr.key(1),
                args=(
                    trajectory_problem.args,
                    jnp.asarray(1.0, dtype=coordinates.dtype),
                    jnp.zeros(
                        (constraint_layout.num_constraints,), dtype=coordinates.dtype
                    ),
                ),
                num_probes=plan.derivatives.num_verification_probes,
            )
            if not bool(np.asarray(hessian_verification.passed)):
                raise ValueError(
                    "Compiled direct-collocation Hessian failed verification."
                )
    compilation_id = canonical_fingerprint(
        {
            "kind": "direct-collocation-compilation",
            "problem": trajectory_problem.problem_id,
            "plan": plan.plan_id,
            "decision": layout.layout_id,
            "constraints": constraint_layout.layout_id,
            "jacobian": jacobian.plan_id,
            "hessian": None if hessian is None else hessian.plan_id,
        }
    )
    structured = StructuredNonlinearProgram(
        objective,
        scaled_constraints,
        jacobian,
        variable_lower=coordinate_lower,
        variable_upper=coordinate_upper,
        constraint_lower=scaled_lower,
        constraint_upper=scaled_upper,
        constraint_sources=constraint_layout.sources,
        hessian_plan=hessian,
        program_id=dense_problem.problem_id,
        structure_id=compilation_id,
    )
    prepared_structured = prepare_structured_nonlinear(
        structured,
        trajectory_problem.args,
    )
    return DirectCollocationCompilation(
        trajectory_problem,
        plan,
        layout,
        constraint_layout,
        initial_decision,
        coordinates,
        coordinate_lower,
        coordinate_upper,
        dynamics_scale,
        resolved_bounds,
        dense_problem,
        structured,
        prepared_structured.template,
        prepared_structured,
        jacobian_verification,
        hessian_verification,
        compilation_id=compilation_id,
    )


def prepare_direct_collocation(
    compilation: DirectCollocationCompilation,
    /,
    *,
    method: AbstractMinimizationMethod,
    termination: OptimizationTermination | None = None,
) -> PreparedDirectCollocation:
    """Bind one compiled transcription to an explicitly selected NLP method."""
    if not isinstance(compilation, DirectCollocationCompilation):
        raise TypeError("compilation must be a DirectCollocationCompilation.")
    if not isinstance(method, AbstractMinimizationMethod):
        raise TypeError("method must be an AbstractMinimizationMethod.")
    termination_ = OptimizationTermination() if termination is None else termination
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be OptimizationTermination or None.")
    structured_program = compilation.prepared_structured_program
    return PreparedDirectCollocation(
        compilation,
        method,
        termination_,
        structured_program,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-direct-collocation",
                "compilation": compilation.compilation_id,
                "method": method.method_id,
                "numeric_binding": structured_program.numeric_binding_id,
            }
        ),
    )


def refresh_direct_collocation(
    prepared: PreparedDirectCollocation,
    args: Any,
    /,
) -> PreparedDirectCollocation:
    """Refresh direct-collocation numerics while retaining transcription topology."""
    if not isinstance(prepared, PreparedDirectCollocation):
        raise TypeError("prepared must be a PreparedDirectCollocation.")
    structured_program = refresh_structured_nonlinear(
        prepared.structured_program,
        args,
    )
    return PreparedDirectCollocation(
        prepared.compilation,
        prepared.method,
        prepared.termination,
        structured_program,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-direct-collocation",
                "compilation": prepared.compilation.compilation_id,
                "method": prepared.method.method_id,
                "numeric_binding": structured_program.numeric_binding_id,
            }
        ),
    )


def _off_grid_audit(
    compilation: DirectCollocationCompilation,
    values: _DirectValues,
    args: Any,
    /,
) -> DirectCollocationOffGridAudit:
    points = compilation.plan.audit.off_grid_points
    dtype = values.decision.states.dtype
    problem = compilation.problem
    steps = compilation.plan.mesh.num_steps
    axis = len(problem.case_shape)
    local_size = problem.state_layout.local_size
    if points == 0:
        residuals = jnp.empty(
            problem.case_shape + (steps, 0, local_size),
            dtype=dtype,
        )
        zeros = jnp.zeros(problem.case_shape + (steps,), dtype=dtype)
        return DirectCollocationOffGridAudit(
            jnp.empty((steps, 0), dtype=dtype),
            residuals,
            zeros,
            zeros,
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(True),
            approximation_id=values.view.approximation_id,
            audit_id=canonical_fingerprint(
                {
                    "kind": "direct-collocation-off-grid-audit",
                    "compilation": compilation.compilation_id,
                    "policy": compilation.plan.audit.audit_id,
                    "points": 0,
                }
            ),
            certified=False,
        )
    fractions = jnp.linspace(0.0, 1.0, points + 2, dtype=dtype)[1:-1]
    left = jnp.take(values.decision.states, jnp.arange(steps), axis=axis)
    right = jnp.take(values.decision.states, jnp.arange(steps) + 1, axis=axis)
    widths = jnp.diff(values.times)
    case_count = prod(problem.case_shape) if problem.case_shape else 1
    interval_count = case_count * steps
    flat_left = left.reshape((interval_count,) + problem.state_shape)
    flat_right = right.reshape((interval_count,) + problem.state_shape)
    flat_widths = jnp.broadcast_to(widths, problem.case_shape + (steps,)).reshape(
        (interval_count,)
    )
    geometry = problem.state_layout.geometry
    endpoint_local = jax.vmap(
        lambda anchor, point: jnp.asarray(
            geometry.inverse_retract(anchor, point)
        ).reshape((local_size,))
    )(flat_left, flat_right)
    sample_local = endpoint_local[:, None, :] * fractions[None, :, None]
    sample_velocity = jnp.broadcast_to(
        endpoint_local[:, None, :] / flat_widths[:, None, None],
        sample_local.shape,
    )
    repeated_left = jnp.broadcast_to(
        flat_left[:, None],
        (interval_count, points) + problem.state_shape,
    )
    flat_count = interval_count * points
    flat_anchors = repeated_left.reshape((flat_count,) + problem.state_shape)
    flat_local = sample_local.reshape((flat_count, local_size))
    flat_velocity = sample_velocity.reshape((flat_count, local_size))
    flat_states = jax.vmap(
        lambda anchor, local: jnp.asarray(
            geometry.retract(
                anchor,
                local.reshape(compilation.decision_layout.local_shape),
            )
        )
    )(flat_anchors, flat_local)
    flat_rates = jax.vmap(
        lambda anchor, local, velocity: jnp.asarray(
            geometry.retraction_jvp(
                anchor,
                local.reshape(compilation.decision_layout.local_shape),
                velocity.reshape(compilation.decision_layout.local_shape),
            )
        )
    )(flat_anchors, flat_local, flat_velocity)
    times = values.times[:-1, None] + fractions[None, :] * widths[:, None]
    controls = jnp.broadcast_to(
        jnp.expand_dims(values.decision.controls, axis=axis + 1),
        problem.case_shape + (steps, points) + problem.control_shape,
    )
    flat_times = jnp.broadcast_to(times, problem.case_shape + times.shape).reshape(
        (flat_count,)
    )
    flat_controls = controls.reshape((flat_count,) + problem.control_shape)
    callback_args = _callback_args(problem, compilation.plan, values.decision, args)
    dynamics_model = problem.dynamics
    if isinstance(dynamics_model, ContinuousSystem):
        physical_residual = jax.vmap(
            lambda time, state, rate, control: (
                rate
                - geometry.project_tangent(
                    state,
                    dynamics_model.evaluate(time, state, callback_args, inputs=control),
                )
            )
        )(flat_times, flat_states, flat_rates, flat_controls)
    else:
        physical_residual = jax.vmap(
            lambda time, state, rate, control: geometry.project_tangent(
                state,
                dynamics_model.evaluate(time, state, rate, callback_args, inputs=control),
            )
        )(flat_times, flat_states, flat_rates, flat_controls)
    residual = jax.vmap(
        lambda anchor, state, tangent: jnp.asarray(
            geometry.retraction_inverse_jvp(anchor, state, tangent)
        ).reshape((local_size,))
    )(flat_anchors, flat_states, physical_residual)
    residuals = residual.reshape(problem.case_shape + (steps, points, local_size))
    residual_magnitudes = jnp.max(jnp.abs(residuals), axis=-1)
    interval_defects = jnp.max(residual_magnitudes, axis=-1)
    interval_path = jnp.zeros(problem.case_shape + (steps,), dtype=dtype)
    for constraint in problem.path_constraints:

        def evaluate_path(time, state, control, callback=constraint):
            return callback(time, state, control, callback_args)

        path = jax.vmap(evaluate_path)(flat_times, flat_states, flat_controls)
        lower = _array_bound(constraint.lower, path, "off-grid path lower")
        upper = _array_bound(constraint.upper, path, "off-grid path upper")
        violation = jnp.maximum(jnp.maximum(lower - path, path - upper), 0.0)
        event_shape = path.shape[1:]
        if event_shape:
            violation = jnp.max(
                violation,
                axis=tuple(range(1, 1 + len(event_shape))),
            )
        violation = violation.reshape(problem.case_shape + (steps, points))
        interval_path = jnp.maximum(interval_path, jnp.max(violation, axis=-1))
    maximum_defect = jnp.max(
        interval_defects,
        initial=jnp.asarray(0.0, dtype=dtype),
    )
    maximum_path = jnp.max(
        interval_path,
        initial=jnp.asarray(0.0, dtype=dtype),
    )
    finite = jnp.all(jnp.isfinite(residuals)) & jnp.all(jnp.isfinite(interval_path))
    return DirectCollocationOffGridAudit(
        times,
        residuals,
        interval_defects,
        interval_path,
        maximum_defect,
        maximum_path,
        finite,
        approximation_id=values.view.approximation_id,
        audit_id=canonical_fingerprint(
            {
                "kind": "direct-collocation-off-grid-audit",
                "compilation": compilation.compilation_id,
                "policy": compilation.plan.audit.audit_id,
                "points": points,
            }
        ),
        certified=False,
    )


def solve_prepared_direct_collocation(
    prepared: PreparedDirectCollocation,
    /,
    *,
    initial_decision: DirectCollocationDecision | None = None,
    args: Any = _DEFAULT_ARGS,
    warm_start: StructuredNonlinearWarmStart | None = None,
    _structured_result: StructuredNonlinearResult | None = None,
) -> DirectCollocationResult:
    """Solve, decode, and independently audit one prepared direct transcription."""
    if not isinstance(prepared, PreparedDirectCollocation):
        raise TypeError("prepared must be a PreparedDirectCollocation.")
    compilation = prepared.compilation
    structured_program = (
        prepared.structured_program
        if args is _DEFAULT_ARGS
        else bind_structured_numeric(
            prepared.structured_program.template,
            args,
            numeric_version=prepared.structured_program.numeric_version + 1,
        )
    )
    runtime_args = structured_program.args
    coordinates = (
        compilation.initial_coordinates
        if initial_decision is None
        else compilation.decision_layout.pack(initial_decision)
    )
    if bool(
        np.any(np.asarray(coordinates) < np.asarray(compilation.coordinate_lower))
        or np.any(np.asarray(coordinates) > np.asarray(compilation.coordinate_upper))
    ):
        raise ValueError("The direct-collocation initial decision violates its bounds.")
    if _structured_result is not None:
        if _structured_result.structure_id != structured_program.structure_id:
            raise ValueError(
                "Structured result does not match direct-collocation structure."
            )
        structured_result = _structured_result
        optimization = structured_result.optimization
    elif (
        isinstance(prepared.method, AbstractStructuredNonlinearMethod)
        and prepared.method.structured_capabilities.exact_sparse_jacobian
    ):
        structured_result = solve_structured_nonlinear(
            structured_program,
            coordinates,
            method=prepared.method,
            termination=prepared.termination,
            warm_start=warm_start,
        )
        optimization = structured_result.optimization
    else:
        if warm_start is not None:
            raise ValueError(
                f"{prepared.method.method_id} does not support structured warm starts."
            )
        structured_result = None
        optimization = minimize(
            compilation.minimization_problem,
            coordinates,
            method=prepared.method,
            termination=prepared.termination,
            args=runtime_args,
        )
    final_coordinates = compilation.decision_layout.pack(
        compilation.decision_layout.unpack(optimization.parameters)
    )
    values = _evaluate_values(
        compilation.problem,
        compilation.plan,
        compilation.decision_layout,
        final_coordinates,
        runtime_args,
    )
    raw_constraints = _raw_constraints(values)
    maximum_defect = _maximum_absolute(values.dynamics)
    maximum_constraint_violation = _maximum_bound_violation(
        raw_constraints,
        compilation.constraint_layout.lower,
        compilation.constraint_layout.upper,
    )
    off_grid = _off_grid_audit(compilation, values, runtime_args)
    finite = (
        jnp.isfinite(values.objective)
        & jnp.all(jnp.isfinite(raw_constraints))
        & jnp.all(jnp.isfinite(values.decision.states))
        & jnp.all(jnp.isfinite(values.decision.controls))
    )
    status = jnp.where(
        ~finite,
        DIRECT_COLLOCATION_NONFINITE,
        jnp.where(
            ~optimization.successful,
            DIRECT_COLLOCATION_OPTIMIZER_FAILED,
            jnp.where(
                maximum_defect > compilation.plan.audit.defect_tolerance,
                DIRECT_COLLOCATION_DEFECT_FAILED,
                jnp.where(
                    maximum_constraint_violation
                    > compilation.plan.audit.constraint_tolerance,
                    DIRECT_COLLOCATION_CONSTRAINT_FAILED,
                    DIRECT_COLLOCATION_SUCCESS,
                ),
            ),
        ),
    ).astype(jnp.int32)
    case_valid = _event_finite(values.decision.states, compilation.problem.state_shape)
    control_status = jnp.full(
        compilation.problem.case_shape,
        jnp.where(
            status == DIRECT_COLLOCATION_SUCCESS,
            CONTROL_SUCCESS,
            jnp.where(
                status == DIRECT_COLLOCATION_CONSTRAINT_FAILED,
                CONTROL_INFEASIBLE,
                CONTROL_DYNAMICS_FAILED,
            ),
        ),
        dtype=jnp.int32,
    )
    time_grid = TimeGrid(
        np.asarray(values.times),
        time_id=f"{compilation.plan.plan_id}:physical-time",
    )
    trajectory = ControlTrajectory(
        time_grid=time_grid,
        states=values.decision.states,
        controls=values.decision.controls,
        valid=case_valid,
        status=control_status,
        backend_status=optimization.status,
        case_shape=compilation.problem.case_shape,
        state_shape=compilation.problem.state_shape,
        control_shape=compilation.problem.control_shape,
        problem_id=compilation.problem.problem_id,
        dynamics_id=compilation.problem.dynamics_id,
        control_id=f"{compilation.plan.plan_id}:interval-controls",
        backend_id=optimization.provenance.backend,
        method_id=f"control:direct-collocation:{compilation.plan.method.method_id}",
        discretization_id=compilation.plan.mesh.mesh_id,
        approximation_id=values.view.approximation_id,
    )
    hessian_nnz = (
        0
        if compilation.structured_program.hessian_plan is None
        else compilation.structured_program.hessian_plan.nnz
    )
    diagnostics = DirectCollocationDiagnostics(
        maximum_defect,
        maximum_constraint_violation,
        off_grid.maximum_defect,
        off_grid.maximum_path_violation,
        off_grid,
        jacobian_nonzeros=compilation.structured_program.jacobian_plan.nnz,
        hessian_nonzeros=hessian_nnz,
        num_variables=compilation.decision_layout.num_variables,
        num_constraints=compilation.constraint_layout.num_constraints,
        off_grid_certified=False,
    )
    method_id = f"control:direct-collocation:{prepared.method.method_id}"
    return DirectCollocationResult(
        values.decision,
        trajectory,
        values.stage_times,
        values.stage_states,
        values.state_rates,
        values.dynamics,
        values.initial,
        values.path,
        values.trajectory,
        values.objective,
        optimization,
        structured_result,
        diagnostics,
        status,
        compilation,
        result_id=f"{compilation.problem.problem_id}:direct-collocation-result",
        method_id=method_id,
    )


def solve_pooled_direct_collocation(
    prepared: PreparedDirectCollocation,
    initial_decisions: Sequence[DirectCollocationDecision],
    /,
    *,
    lane_count: int,
    warm_starts: Sequence[StructuredNonlinearWarmStart | None] | None = None,
) -> PooledDirectCollocationResult:
    """Solve independent decisions with one prepared fixed-topology transcription."""
    if not isinstance(prepared, PreparedDirectCollocation):
        raise TypeError("prepared must be a PreparedDirectCollocation.")
    if (
        not isinstance(prepared.method, AbstractStructuredNonlinearMethod)
        or not prepared.method.structured_capabilities.pooled_batch
    ):
        raise TypeError(
            "Pooled direct collocation requires a pool-capable structured NLP method."
        )
    decisions = tuple(initial_decisions)
    if not decisions:
        raise ValueError("initial_decisions must contain at least one decision.")
    coordinates = jnp.stack(
        tuple(
            prepared.compilation.decision_layout.pack(decision) for decision in decisions
        )
    )
    pooled: PooledStructuredNonlinearResult = solve_pooled_structured_nonlinear(
        prepared.structured_program,
        coordinates,
        method=prepared.method,
        termination=prepared.termination,
        lane_count=lane_count,
        warm_starts=warm_starts,
    )
    results = tuple(
        solve_prepared_direct_collocation(
            prepared,
            initial_decision=decision,
            _structured_result=structured_result,
        )
        for decision, structured_result in zip(
            decisions,
            pooled.results,
            strict=True,
        )
    )
    return PooledDirectCollocationResult(results, pooled.evidence)


def solve_direct_collocation(
    problem: TrajectoryOptimizationProblem | ControlProblem,
    plan: DirectCollocationPlan,
    initial_states: ArrayLike,
    initial_controls: ArrayLike,
    /,
    *,
    method: AbstractMinimizationMethod,
    parameter_guess: Any = None,
    duration_guess: ArrayLike | None = None,
    bounds: DirectCollocationBounds | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = _DEFAULT_ARGS,
    warm_start: StructuredNonlinearWarmStart | None = None,
) -> DirectCollocationResult:
    """Compile, solve, decode, and audit one direct-collocation problem."""
    compilation = compile_direct_collocation(
        problem,
        plan,
        initial_states,
        initial_controls,
        parameter_guess=parameter_guess,
        duration_guess=duration_guess,
        bounds=bounds,
    )
    prepared = prepare_direct_collocation(
        compilation,
        method=method,
        termination=termination,
    )
    return solve_prepared_direct_collocation(
        prepared,
        args=args,
        warm_start=warm_start,
    )


__all__ = [
    "compile_direct_collocation",
    "DIRECT_COLLOCATION_CONSTRAINT_FAILED",
    "DIRECT_COLLOCATION_DEFECT_FAILED",
    "DIRECT_COLLOCATION_NONFINITE",
    "DIRECT_COLLOCATION_OPTIMIZER_FAILED",
    "DIRECT_COLLOCATION_RECONSTRUCTION_FAILED",
    "DIRECT_COLLOCATION_SUCCESS",
    "DirectCollocationAuditPolicy",
    "DirectCollocationBounds",
    "DirectCollocationCompilation",
    "DirectCollocationConstraintLayout",
    "DirectCollocationDecision",
    "DirectCollocationDecisionLayout",
    "DirectCollocationDerivativePolicy",
    "DirectCollocationDiagnostics",
    "DirectCollocationHessianMode",
    "DirectCollocationOffGridAudit",
    "DirectCollocationPlan",
    "DirectCollocationResult",
    "PooledDirectCollocationResult",
    "DirectCollocationScaling",
    "prepare_direct_collocation",
    "refresh_direct_collocation",
    "PreparedDirectCollocation",
    "solve_direct_collocation",
    "solve_pooled_direct_collocation",
    "solve_prepared_direct_collocation",
]
