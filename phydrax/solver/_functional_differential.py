#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._interpolation._barycentric import barycentric_interpolate
from .._numerics._quadrature_rules import clenshaw_curtis_data
from .._strict import StrictModule


FunctionalVectorField: TypeAlias = Callable[[Array, Array, Array, Any], ArrayLike]
FunctionalArgumentMap: TypeAlias = Callable[[Array, Array, Any], ArrayLike]
FunctionalBoundaryResidual: TypeAlias = Callable[[Array, Array, Any, Any], ArrayLike]
FunctionalTrajectoryResidual: TypeAlias = Callable[[Any, Any], ArrayLike]
FunctionalCollocationMethod: TypeAlias = Literal["auto", "root", "least-squares"]


def _state_shape(value: Sequence[int], /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if any(size <= 0 for size in shape):
        raise ValueError("state_shape entries must be positive.")
    return shape


def _maximum_absolute(value: Array, /) -> Array:
    if int(value.size) == 0:
        return jnp.asarray(0.0, dtype=value.dtype)
    return jnp.max(jnp.abs(value))


def _barycentric_data(nodes: Array, /) -> tuple[Array, Array]:
    nodes_host = np.asarray(nodes, dtype=float)
    count = int(nodes_host.size)
    differences = nodes_host[:, None] - nodes_host[None, :]
    differences[np.diag_indices(count)] = 1.0
    weights = (-1.0) ** np.arange(count, dtype=float)
    weights[[0, -1]] *= 0.5

    ratio = weights[None, :] / weights[:, None]
    matrix = np.zeros((count, count), dtype=float)
    off_diagonal = ~np.eye(count, dtype=bool)
    matrix[off_diagonal] = (ratio / differences)[off_diagonal]
    matrix[np.diag_indices(count)] = -np.sum(matrix, axis=1)
    return jnp.asarray(weights), jnp.asarray(matrix)


class FunctionalDifferentialContext(StrictModule):
    """Callback context for solved physical parameters or an unknown period."""

    args: Any
    parameters: Array | None
    period: Array | None

    def __init__(
        self,
        args: Any,
        parameters: Array | None,
        period: Array | None,
        /,
    ):
        self.args = args
        self.parameters = parameters
        self.period = period


class FunctionalDifferentialBoundaryProblem(StrictModule):
    """Global functional differential equation with boundary-style constraints.

    ``argument_times(t, y, callback_args)`` declares a fixed-size vector of
    physical trajectory locations. They may be retarded, advanced, mixed, or
    state dependent. The vector field receives their interpolated values with
    shape ``(num_arguments,) + state_shape``. Every declared location must remain
    in the collocation interval; global collocation deliberately does not invent
    an IVP history or extrapolate beyond the declared mesh.

    ``boundary`` is called as
    ``boundary(left, right, trajectory, callback_args)``. ``phase`` and
    ``observation_residual`` are called as
    ``constraint(trajectory, callback_args)``. Without solved parameters or an
    unknown period, ``callback_args`` is the user ``args`` unchanged. Otherwise it
    is a :class:`FunctionalDifferentialContext`. Setting ``periodic=True`` adds
    ``right - left`` independently of the other boundary blocks.
    """

    vector_field: FunctionalVectorField
    argument_times: FunctionalArgumentMap | None
    boundary: FunctionalBoundaryResidual | None
    phase: FunctionalTrajectoryResidual | None
    observation_residual: FunctionalTrajectoryResidual | None
    observation_times: Array | None
    observation_values: Array | None
    observation_weights: Array | None
    state_shape: tuple[int, ...] = eqx.field(static=True)
    num_arguments: int = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    parameter_shape: tuple[int, ...] | None = eqx.field(static=True)
    unknown_period: bool = eqx.field(static=True)

    def __init__(
        self,
        vector_field: FunctionalVectorField,
        *,
        state_shape: Sequence[int] = (),
        argument_times: FunctionalArgumentMap | None = None,
        num_arguments: int = 0,
        boundary: FunctionalBoundaryResidual | None = None,
        periodic: bool = False,
        phase: FunctionalTrajectoryResidual | None = None,
        observation_times: ArrayLike | None = None,
        observation_values: ArrayLike | None = None,
        observation_weights: ArrayLike | None = None,
        observation_residual: FunctionalTrajectoryResidual | None = None,
        parameter_shape: Sequence[int] | None = None,
        unknown_period: bool = False,
    ):
        if not callable(vector_field):
            raise TypeError("vector_field must be callable.")
        if isinstance(num_arguments, bool) or int(num_arguments) != num_arguments:
            raise TypeError("num_arguments must be an integer.")
        argument_count = int(num_arguments)
        if argument_count < 0:
            raise ValueError("num_arguments must be non-negative.")
        if argument_times is None and argument_count != 0:
            raise ValueError("num_arguments must be zero when argument_times is absent.")
        if argument_times is not None and argument_count == 0:
            raise ValueError(
                "num_arguments must be positive when argument_times is provided."
            )
        for callback, name in (
            (argument_times, "argument_times"),
            (boundary, "boundary"),
            (phase, "phase"),
            (observation_residual, "observation_residual"),
        ):
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None.")

        shape = _state_shape(state_shape)
        if parameter_shape is None:
            solved_parameter_shape = None
        else:
            solved_parameter_shape = tuple(int(size) for size in parameter_shape)
            if any(size <= 0 for size in solved_parameter_shape):
                raise ValueError("parameter_shape entries must be positive.")
        if unknown_period and not periodic:
            raise ValueError("unknown_period requires periodic=True.")
        if unknown_period and phase is None:
            raise ValueError("unknown_period requires an explicit phase constraint.")
        if (observation_times is None) != (observation_values is None):
            raise ValueError(
                "observation_times and observation_values must be supplied together."
            )
        if observation_times is None:
            if observation_weights is not None:
                raise ValueError(
                    "observation_weights requires observation_times and values."
                )
            times = None
            values = None
            weights = None
        else:
            times = jnp.asarray(observation_times, dtype=float)
            values = jnp.asarray(observation_values)
            if times.ndim != 1:
                raise ValueError("observation_times must be rank one.")
            expected = (int(times.size),) + shape
            if values.shape != expected:
                raise ValueError(
                    "observation_values must have shape (num_observations,) + "
                    "state_shape."
                )
            times = eqx.error_if(
                times,
                jnp.any(~jnp.isfinite(times)),
                "observation_times must be finite.",
            )
            values = eqx.error_if(
                values,
                jnp.any(~jnp.isfinite(values)),
                "observation_values must be finite.",
            )
            if observation_weights is None:
                weights = None
            else:
                weights = jnp.asarray(observation_weights, dtype=float)
                allowed = weights.shape in ((), (int(times.size),), expected)
                if not allowed:
                    raise ValueError(
                        "observation_weights must be scalar, have one entry per "
                        "observation, or match observation_values."
                    )
                weights = eqx.error_if(
                    weights,
                    jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0.0),
                    "observation_weights must be finite and non-negative.",
                )

        self.vector_field = vector_field
        self.argument_times = argument_times
        self.boundary = boundary
        self.phase = phase
        self.observation_residual = observation_residual
        self.observation_times = times
        self.observation_values = values
        self.observation_weights = weights
        self.state_shape = shape
        self.num_arguments = argument_count
        self.periodic = bool(periodic)
        self.parameter_shape = solved_parameter_shape
        self.unknown_period = bool(unknown_period)


class FunctionalCollocationPlan(StrictModule):
    """Piecewise Chebyshev--Lobatto collocation and nonlinear-solve plan.

    ``mesh`` declares every polynomial interval and ``degree`` declares its
    degree. With an unknown period it is a reference coordinate mesh whose first
    point is the physical start and whose span is rescaled to the solved physical
    period. Differential residuals use the physical Clenshaw--Curtis square-root
    quadrature weights; continuity and user constraint blocks retain their native
    scaling. ``method='auto'`` selects zero-residual root finding for a square
    residual and Levenberg--Marquardt least squares for an overdetermined residual.
    """

    mesh: Array
    reference_nodes: Array
    barycentric_weights: Array
    differentiation_matrix: Array
    quadrature_weights: Array
    root_finder: Any
    least_squares_solver: Any
    adjoint: Any
    degree: int = eqx.field(static=True)
    method: FunctionalCollocationMethod = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    throw: bool = eqx.field(static=True)

    def __init__(
        self,
        mesh: ArrayLike,
        degree: int,
        *,
        method: FunctionalCollocationMethod = "auto",
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_steps: int = 256,
        throw: bool = True,
        root_finder: Any = None,
        least_squares_solver: Any = None,
        adjoint: Any = None,
    ):
        if isinstance(degree, bool) or int(degree) != degree:
            raise TypeError("degree must be an integer.")
        degree_ = int(degree)
        if degree_ < 1:
            raise ValueError("degree must be at least one.")
        if method not in ("auto", "root", "least-squares"):
            raise ValueError("method must be 'auto', 'root', or 'least-squares'.")
        if not np.isfinite(rtol) or float(rtol) < 0.0:
            raise ValueError("rtol must be finite and non-negative.")
        if not np.isfinite(atol) or float(atol) < 0.0:
            raise ValueError("atol must be finite and non-negative.")
        if float(rtol) == 0.0 and float(atol) == 0.0:
            raise ValueError("rtol and atol cannot both be zero.")
        if isinstance(max_steps, bool) or int(max_steps) != max_steps:
            raise TypeError("max_steps must be an integer.")
        if int(max_steps) < 1:
            raise ValueError("max_steps must be positive.")

        mesh_ = jnp.asarray(mesh, dtype=float)
        if mesh_.ndim != 1 or int(mesh_.size) < 2:
            raise ValueError("mesh must be a rank-one array with at least two entries.")
        mesh_ = eqx.error_if(
            mesh_,
            jnp.any(~jnp.isfinite(mesh_)) | jnp.any(jnp.diff(mesh_) <= 0.0),
            "mesh must be finite and strictly increasing.",
        )
        rule = clenshaw_curtis_data(degree_ + 1)
        barycentric_weights, differentiation_matrix = _barycentric_data(rule.nodes)

        rtol_ = float(rtol)
        atol_ = float(atol)
        self.mesh = mesh_
        self.reference_nodes = jnp.asarray(rule.nodes, dtype=mesh_.dtype)
        self.barycentric_weights = jnp.asarray(barycentric_weights, dtype=mesh_.dtype)
        self.differentiation_matrix = jnp.asarray(
            differentiation_matrix, dtype=mesh_.dtype
        )
        self.quadrature_weights = jnp.asarray(rule.weights, dtype=mesh_.dtype)
        self.root_finder = (
            optx.LevenbergMarquardt(rtol=rtol_, atol=atol_, norm=optx.rms_norm)
            if root_finder is None
            else root_finder
        )
        self.least_squares_solver = (
            optx.LevenbergMarquardt(rtol=rtol_, atol=atol_, norm=optx.rms_norm)
            if least_squares_solver is None
            else least_squares_solver
        )
        self.adjoint = optx.ImplicitAdjoint() if adjoint is None else adjoint
        self.degree = degree_
        self.method = method
        self.rtol = rtol_
        self.atol = atol_
        self.max_steps = int(max_steps)
        self.throw = bool(throw)

    @property
    def num_intervals(self) -> int:
        return int(self.mesh.size) - 1

    @property
    def nodes_per_interval(self) -> int:
        return self.degree + 1

    @property
    def collocation_times(self) -> Array:
        half_width = 0.5 * jnp.diff(self.mesh)
        center = 0.5 * (self.mesh[:-1] + self.mesh[1:])
        return center[:, None] + half_width[:, None] * self.reference_nodes[None, :]


class _FunctionalPolynomialInterpolation(StrictModule):
    mesh: Array
    reference_nodes: Array
    barycentric_weights: Array
    values: Array
    period: Array | None
    state_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        mesh: Array,
        reference_nodes: Array,
        barycentric_weights: Array,
        values: Array,
        period: Array | None,
        state_shape: tuple[int, ...],
    ):
        self.mesh = mesh
        self.reference_nodes = reference_nodes
        self.barycentric_weights = barycentric_weights
        self.values = values
        self.period = period
        self.state_shape = state_shape

    @property
    def coordinate_span(self) -> Array:
        return self.mesh[-1] - self.mesh[0]

    @property
    def physical_mesh(self) -> Array:
        return self.physical_times(self.mesh)

    def physical_times(self, coordinate_times: ArrayLike, /) -> Array:
        coordinates = jnp.asarray(coordinate_times, dtype=self.mesh.dtype)
        if self.period is None:
            return coordinates
        return self.mesh[0] + (
            (coordinates - self.mesh[0]) * self.period / self.coordinate_span
        )

    def _query_geometry(
        self, query_times: ArrayLike, /, *, left: bool
    ) -> tuple[Array, Array, tuple[int, ...]]:
        query = jnp.asarray(query_times, dtype=self.mesh.dtype)
        physical_end = (
            self.mesh[-1] if self.period is None else self.mesh[0] + self.period
        )
        query = eqx.error_if(
            query,
            jnp.any(~jnp.isfinite(query)),
            "Functional collocation query times must be finite.",
        )
        query = eqx.error_if(
            query,
            jnp.any((query < self.mesh[0]) | (query > physical_end)),
            "Functional collocation query lies outside the declared mesh interval.",
        )
        flat = query.reshape((-1,))
        if self.period is not None:
            flat = self.mesh[0] + (
                (flat - self.mesh[0]) * self.coordinate_span / self.period
            )
        side = "left" if left else "right"
        indices = jnp.searchsorted(self.mesh, flat, side=side) - 1
        indices = jnp.clip(indices, 0, int(self.mesh.size) - 2)
        lower = self.mesh[indices]
        upper = self.mesh[indices + 1]
        reference = 2.0 * (flat - lower) / (upper - lower) - 1.0
        return reference, indices, query.shape

    def evaluate(self, query_times: ArrayLike, /, *, left: bool = True) -> Array:
        """Evaluate with output shape ``query_shape + state_shape``."""
        reference, indices, query_shape = self._query_geometry(query_times, left=left)
        selected = self.values[indices]
        evaluated = jax.vmap(
            lambda point, values: barycentric_interpolate(
                point,
                self.reference_nodes,
                self.barycentric_weights,
                values,
            )
        )(reference, selected)
        return evaluated.reshape(query_shape + self.state_shape)

    def derivative(self, query_times: ArrayLike, /, *, left: bool = True) -> Array:
        """Evaluate the piecewise-polynomial first derivative."""
        reference, indices, query_shape = self._query_geometry(query_times, left=left)
        selected = self.values[indices]

        def reference_derivative(point, values):
            interpolate = lambda location: barycentric_interpolate(
                location,
                self.reference_nodes,
                self.barycentric_weights,
                values,
            )
            return jax.jvp(interpolate, (point,), (jnp.ones_like(point),))[1]

        derivative = jax.vmap(reference_derivative)(reference, selected)
        scale = 2.0 / (self.mesh[indices + 1] - self.mesh[indices])
        if self.period is not None:
            scale = scale * self.coordinate_span / self.period
        scale = scale.reshape(scale.shape + (1,) * len(self.state_shape))
        return (derivative * scale).reshape(query_shape + self.state_shape)


class _FunctionalUnknowns(StrictModule):
    values: Array
    parameters: Array | None
    log_period: Array | None

    @property
    def period(self) -> Array | None:
        if self.log_period is None:
            return None
        return jnp.exp(self.log_period)


def _callback_args(
    problem: FunctionalDifferentialBoundaryProblem,
    unknowns: _FunctionalUnknowns,
    args: Any,
    /,
) -> Any:
    if problem.parameter_shape is None and not problem.unknown_period:
        return args
    return FunctionalDifferentialContext(args, unknowns.parameters, unknowns.period)


def _interpolation(
    problem: FunctionalDifferentialBoundaryProblem,
    plan: FunctionalCollocationPlan,
    unknowns: _FunctionalUnknowns,
    /,
) -> _FunctionalPolynomialInterpolation:
    return _FunctionalPolynomialInterpolation(
        plan.mesh,
        plan.reference_nodes,
        plan.barycentric_weights,
        unknowns.values,
        unknowns.period,
        problem.state_shape,
    )


def _unknown_size(unknowns: _FunctionalUnknowns, /) -> int:
    size = int(unknowns.values.size)
    if unknowns.parameters is not None:
        size += int(unknowns.parameters.size)
    if unknowns.log_period is not None:
        size += 1
    return size


class _FunctionalResidualBlocks(StrictModule):
    differential: Array
    continuity: Array
    periodic: Array
    boundary: Array
    phase: Array
    observations: Array
    observation_constraint: Array
    assembled: Array


class FunctionalDifferentialSolution(StrictModule):
    """Global collocation trajectory, solved physical unknowns, and diagnostics.

    ``times``, ``collocation_times``, ``evaluate``, and ``derivative`` use
    physical time. ``parameters`` and ``period`` are ``None`` unless their
    corresponding problem unknowns were declared.
    """

    times: Array
    states: Array
    collocation_times: Array
    collocation_values: Array
    parameters: Array | None
    period: Array | None
    interpolation: _FunctionalPolynomialInterpolation
    residual: Array
    differential_residual: Array
    continuity_residual: Array
    periodic_residual: Array
    boundary_residual: Array
    phase_residual: Array
    observation_residual: Array
    observation_constraint_residual: Array
    valid: Array
    result: Any
    backend_result: Any
    stats: frozendict[str, Any]
    metadata: frozendict[str, Any] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    num_intervals: int = eqx.field(static=True)
    unknown_size: int = eqx.field(static=True)
    residual_size: int = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)
    nonlinear_solver: str = eqx.field(static=True)
    resolved_method: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        plan: FunctionalCollocationPlan,
        problem: FunctionalDifferentialBoundaryProblem,
        unknowns: _FunctionalUnknowns,
        blocks: _FunctionalResidualBlocks,
        backend_result: Any,
        resolved_method: str,
        nonlinear_solver: str,
    ):
        interpolation = _interpolation(problem, plan, unknowns)
        times = interpolation.physical_mesh
        states = interpolation.evaluate(times)
        finite = jnp.all(jnp.isfinite(unknowns.values)) & jnp.all(
            jnp.isfinite(blocks.assembled)
        )
        if unknowns.parameters is not None:
            finite = finite & jnp.all(jnp.isfinite(unknowns.parameters))
        if unknowns.period is not None:
            finite = finite & jnp.isfinite(unknowns.period)
        converged = backend_result.result == optx.RESULTS.successful
        self.times = times
        self.states = states
        self.collocation_times = interpolation.physical_times(plan.collocation_times)
        self.collocation_values = unknowns.values
        self.parameters = unknowns.parameters
        self.period = unknowns.period
        self.interpolation = interpolation
        self.residual = blocks.assembled
        self.differential_residual = blocks.differential
        self.continuity_residual = blocks.continuity
        self.periodic_residual = blocks.periodic
        self.boundary_residual = blocks.boundary
        self.phase_residual = blocks.phase
        self.observation_residual = blocks.observations
        self.observation_constraint_residual = blocks.observation_constraint
        self.valid = converged & finite
        self.result = backend_result.result
        self.backend_result = backend_result
        self.stats = frozendict(backend_result.stats)
        self.metadata = frozendict(
            {
                "backend": "optimistix",
                "causal": False,
                "problem_kind": "functional-differential-boundary",
                "trajectory_representation": "piecewise-barycentric-polynomial",
                "global_semantics": True,
                "solved_parameters": problem.parameter_shape is not None,
                "solved_period": problem.unknown_period,
            }
        )
        self.state_shape = problem.state_shape
        self.degree = plan.degree
        self.num_intervals = plan.num_intervals
        self.unknown_size = _unknown_size(unknowns)
        self.residual_size = int(blocks.assembled.size)
        self.solver_name = "functional-collocation"
        self.solver_id = "solver:functional-differential-collocation"
        self.nonlinear_solver = nonlinear_solver
        self.resolved_method = resolved_method

    @property
    def num_times(self) -> int:
        return int(self.times.size)

    @property
    def has_dense_interpolation(self) -> bool:
        return True

    @property
    def converged(self) -> Array:
        return self.result == optx.RESULTS.successful

    @property
    def successful(self) -> Array:
        return self.valid

    @property
    def status_message(self) -> str:
        return optx.RESULTS[self.result]

    @property
    def max_residual(self) -> Array:
        return _maximum_absolute(self.residual)

    @property
    def max_differential_residual(self) -> Array:
        return _maximum_absolute(self.differential_residual)

    @property
    def max_continuity_residual(self) -> Array:
        return _maximum_absolute(self.continuity_residual)

    def evaluate(self, query_times: ArrayLike, /, *, left: bool = True) -> Array:
        return self.interpolation.evaluate(query_times, left=left)

    def derivative(self, query_times: ArrayLike, /, *, left: bool = True) -> Array:
        return self.interpolation.derivative(query_times, left=left)


def _empty(dtype: jnp.dtype, /) -> Array:
    return jnp.empty((0,), dtype=dtype)


def _argument_values(
    problem: FunctionalDifferentialBoundaryProblem,
    trajectory: _FunctionalPolynomialInterpolation,
    times: Array,
    states: Array,
    args: Any,
    /,
) -> Array:
    stage_count = int(times.size)
    argument_times = problem.argument_times
    if argument_times is None:
        return jnp.empty(
            (stage_count, 0) + problem.state_shape,
            dtype=states.dtype,
        )

    def one_argument_map(time, state):
        locations = jnp.asarray(argument_times(time, state, args))
        if problem.num_arguments == 1 and locations.shape == ():
            locations = locations.reshape((1,))
        if locations.shape != (problem.num_arguments,):
            raise ValueError(
                "argument_times must return shape (num_arguments,) at every "
                "collocation stage."
            )
        return locations

    locations = jax.vmap(one_argument_map)(times, states)
    expected = (stage_count, problem.num_arguments)
    if locations.shape != expected:
        raise ValueError("argument_times returned an inconsistent fixed shape.")
    evaluated = trajectory.evaluate(locations)
    return evaluated.reshape(expected + problem.state_shape)


def _constraint_residual(
    callback: FunctionalTrajectoryResidual | None,
    trajectory: _FunctionalPolynomialInterpolation,
    args: Any,
    dtype: jnp.dtype,
    /,
) -> Array:
    if callback is None:
        return _empty(dtype)
    return jnp.asarray(callback(trajectory, args))


def _residual_blocks(
    unknowns: _FunctionalUnknowns,
    packed_args: tuple[
        FunctionalDifferentialBoundaryProblem, FunctionalCollocationPlan, Any
    ],
    /,
) -> _FunctionalResidualBlocks:
    problem, plan, args = packed_args
    values = unknowns.values
    expected_values = (
        plan.num_intervals,
        plan.nodes_per_interval,
    ) + problem.state_shape
    if values.shape != expected_values:
        raise ValueError("Collocation unknowns do not match the plan and state shape.")

    trajectory = _interpolation(problem, plan, unknowns)
    callback_args = _callback_args(problem, unknowns, args)
    half_width = 0.5 * jnp.diff(plan.mesh)
    derivative_reference = jnp.einsum(
        "ij,ej...->ei...", plan.differentiation_matrix, values
    )
    derivative_scale = 1.0 / half_width
    measure_scale = jnp.asarray(1.0, dtype=plan.mesh.dtype)
    if unknowns.period is not None:
        coordinate_span = plan.mesh[-1] - plan.mesh[0]
        derivative_scale = derivative_scale * coordinate_span / unknowns.period
        measure_scale = unknowns.period / coordinate_span
    derivative_scale = derivative_scale.reshape(
        (plan.num_intervals, 1) + (1,) * len(problem.state_shape)
    )
    derivatives = derivative_reference * derivative_scale

    collocation_times = trajectory.physical_times(plan.collocation_times[:, 1:])
    collocation_states = values[:, 1:]
    flat_times = collocation_times.reshape((-1,))
    flat_states = collocation_states.reshape((-1,) + problem.state_shape)
    functional_values = _argument_values(
        problem, trajectory, flat_times, flat_states, callback_args
    )

    def evaluate_vector_field(time, state, arguments):
        value = jnp.asarray(problem.vector_field(time, state, arguments, callback_args))
        if value.shape != problem.state_shape:
            raise ValueError("vector_field must return exactly state_shape.")
        return value

    vector_field = jax.vmap(evaluate_vector_field)(
        flat_times, flat_states, functional_values
    ).reshape(collocation_states.shape)
    differential = derivatives[:, 1:] - vector_field
    continuity = values[:-1, -1] - values[1:, 0]

    dtype = values.dtype
    periodic = values[-1, -1] - values[0, 0] if problem.periodic else _empty(dtype)
    boundary = (
        jnp.asarray(
            problem.boundary(values[0, 0], values[-1, -1], trajectory, callback_args)
        )
        if problem.boundary is not None
        else _empty(dtype)
    )
    phase = _constraint_residual(problem.phase, trajectory, callback_args, dtype)

    observation_times = problem.observation_times
    observation_values = problem.observation_values
    if observation_times is None:
        observations = _empty(dtype)
    else:
        assert observation_values is not None
        observations = trajectory.evaluate(observation_times) - observation_values
        if problem.observation_weights is not None:
            weights = problem.observation_weights
            if weights.shape == (int(observation_times.size),):
                weights = weights.reshape(weights.shape + (1,) * len(problem.state_shape))
            observations = observations * jnp.sqrt(weights)
    observation_constraint = _constraint_residual(
        problem.observation_residual, trajectory, callback_args, dtype
    )

    differential_weight = jnp.sqrt(
        measure_scale * half_width[:, None] * plan.quadrature_weights[None, 1:]
    )
    differential_weight = differential_weight.reshape(
        differential_weight.shape + (1,) * len(problem.state_shape)
    )
    assembled = jnp.concatenate(
        (
            (differential * differential_weight).reshape((-1,)),
            continuity.reshape((-1,)),
            periodic.reshape((-1,)),
            boundary.reshape((-1,)),
            phase.reshape((-1,)),
            observations.reshape((-1,)),
            observation_constraint.reshape((-1,)),
        )
    )
    return _FunctionalResidualBlocks(
        differential=differential,
        continuity=continuity,
        periodic=periodic,
        boundary=boundary,
        phase=phase,
        observations=observations,
        observation_constraint=observation_constraint,
        assembled=assembled,
    )


def _assembled_residual(
    unknowns: _FunctionalUnknowns,
    packed_args: tuple[
        FunctionalDifferentialBoundaryProblem, FunctionalCollocationPlan, Any
    ],
    /,
) -> Array:
    return _residual_blocks(unknowns, packed_args).assembled


def _initial_values(
    initial_guess: ArrayLike | Callable[[Array, Any], ArrayLike],
    problem: FunctionalDifferentialBoundaryProblem,
    plan: FunctionalCollocationPlan,
    callback_args: Any,
    period: Array | None,
    /,
) -> Array:
    coordinate_times = plan.collocation_times
    physical_times = coordinate_times
    if period is not None:
        coordinate_span = plan.mesh[-1] - plan.mesh[0]
        physical_times = plan.mesh[0] + (
            (coordinate_times - plan.mesh[0]) * period / coordinate_span
        )
    expected = (
        plan.num_intervals,
        plan.nodes_per_interval,
    ) + problem.state_shape
    if callable(initial_guess):
        initial_guess_function = cast(Callable[[Array, Any], ArrayLike], initial_guess)
        flat_times = physical_times.reshape((-1,))

        def evaluate(time):
            value = jnp.asarray(initial_guess_function(time, callback_args))
            if value.shape != problem.state_shape:
                raise ValueError("initial_guess must return exactly state_shape.")
            return value

        return jax.vmap(evaluate)(flat_times).reshape(expected)

    guess = jnp.asarray(initial_guess)
    if guess.shape == expected:
        return guess
    if guess.shape == problem.state_shape:
        return jnp.broadcast_to(guess, expected)
    mesh_shape = (int(plan.mesh.size),) + problem.state_shape
    if guess.shape == mesh_shape:
        flat_guess = guess.reshape((int(plan.mesh.size), -1))
        flat_times = coordinate_times.reshape((-1,))
        interpolated = jax.vmap(
            lambda component: jnp.interp(flat_times, plan.mesh, component),
            in_axes=1,
            out_axes=1,
        )(flat_guess)
        return interpolated.reshape(expected)
    raise ValueError(
        "initial_guess must be callable, state-shaped, mesh-node-shaped, or "
        "collocation-node-shaped."
    )


def _initial_parameters(
    problem: FunctionalDifferentialBoundaryProblem,
    parameter_guess: ArrayLike | None,
    dtype: jnp.dtype,
    /,
) -> Array | None:
    if problem.parameter_shape is None:
        if parameter_guess is not None:
            raise ValueError(
                "parameter_guess requires problem.parameter_shape to be declared."
            )
        return None
    if parameter_guess is None:
        raise ValueError(
            "parameter_guess is required when problem.parameter_shape is declared."
        )
    parameters = jnp.asarray(parameter_guess)
    if parameters.shape != problem.parameter_shape:
        raise ValueError("parameter_guess must match problem.parameter_shape.")
    if not jnp.issubdtype(parameters.dtype, jnp.inexact):
        parameters = parameters.astype(dtype)
    return eqx.error_if(
        parameters,
        jnp.any(~jnp.isfinite(parameters)),
        "parameter_guess must be finite.",
    )


def _initial_log_period(
    problem: FunctionalDifferentialBoundaryProblem,
    period_guess: ArrayLike | None,
    dtype: jnp.dtype,
    /,
) -> Array | None:
    if not problem.unknown_period:
        if period_guess is not None:
            raise ValueError("period_guess requires problem.unknown_period=True.")
        return None
    if period_guess is None:
        raise ValueError("period_guess is required when unknown_period=True.")
    period = jnp.asarray(period_guess, dtype=dtype)
    if period.shape != ():
        raise ValueError("period_guess must be scalar.")
    period = eqx.error_if(
        period,
        ~jnp.isfinite(period) | (period <= 0.0),
        "period_guess must be finite and positive.",
    )
    return jnp.log(period)


def solve_functional_differential(
    problem: FunctionalDifferentialBoundaryProblem,
    plan: FunctionalCollocationPlan,
    initial_guess: ArrayLike | Callable[[Array, Any], ArrayLike],
    /,
    *,
    args: Any = None,
    parameter_guess: ArrayLike | None = None,
    period_guess: ArrayLike | None = None,
) -> FunctionalDifferentialSolution:
    """Solve a global retarded, advanced, or mixed functional boundary problem.

    This is a global nonlinear collocation solve, not a causal delay-IVP method.
    ``method='auto'`` uses root finding for a square residual and least squares for
    an overdetermined residual. Underdetermined systems and non-square explicit
    root requests fail before entering Optimistix.

    ``initial_guess`` may be ``initial_guess(time, callback_args)``, one constant
    state-shaped value, one value per mesh point, or the complete
    ``(num_intervals, degree + 1) + state_shape`` nodal array. If solved
    parameters or a period are declared, their required initial guesses are
    supplied through ``parameter_guess`` and ``period_guess``.
    """
    if not isinstance(problem, FunctionalDifferentialBoundaryProblem):
        raise TypeError("problem must be a FunctionalDifferentialBoundaryProblem.")
    if not isinstance(plan, FunctionalCollocationPlan):
        raise TypeError("plan must be a FunctionalCollocationPlan.")

    initial_parameters = _initial_parameters(problem, parameter_guess, plan.mesh.dtype)
    initial_log_period = _initial_log_period(problem, period_guess, plan.mesh.dtype)
    initial_period = None if initial_log_period is None else jnp.exp(initial_log_period)
    initial_callback_args = (
        args
        if problem.parameter_shape is None and not problem.unknown_period
        else FunctionalDifferentialContext(args, initial_parameters, initial_period)
    )
    initial_values = _initial_values(
        initial_guess,
        problem,
        plan,
        initial_callback_args,
        initial_period,
    )
    if not jnp.issubdtype(initial_values.dtype, jnp.inexact):
        initial_values = initial_values.astype(plan.mesh.dtype)
    initial_unknowns = _FunctionalUnknowns(
        initial_values,
        initial_parameters,
        initial_log_period,
    )
    packed_args = (problem, plan, args)
    initial_residual = _assembled_residual(initial_unknowns, packed_args)
    unknown_size = _unknown_size(initial_unknowns)
    residual_size = int(initial_residual.size)
    if residual_size < unknown_size:
        raise ValueError(
            "Functional collocation is underdetermined: residual size "
            f"{residual_size} is smaller than unknown size {unknown_size}."
        )
    if plan.method == "root" and residual_size != unknown_size:
        raise ValueError(
            "Root collocation requires a square residual: residual size "
            f"{residual_size} does not match unknown size {unknown_size}."
        )
    resolved_method = (
        "root"
        if plan.method == "root"
        or (plan.method == "auto" and residual_size == unknown_size)
        else "least-squares"
    )

    if resolved_method == "root":
        backend_result = optx.root_find(
            _assembled_residual,
            plan.root_finder,
            initial_unknowns,
            args=packed_args,
            max_steps=plan.max_steps,
            adjoint=plan.adjoint,
            throw=plan.throw,
        )
        nonlinear_solver = type(plan.root_finder).__name__
    else:
        backend_result = optx.least_squares(
            _assembled_residual,
            plan.least_squares_solver,
            initial_unknowns,
            args=packed_args,
            max_steps=plan.max_steps,
            adjoint=plan.adjoint,
            throw=plan.throw,
        )
        nonlinear_solver = type(plan.least_squares_solver).__name__

    final_blocks = _residual_blocks(backend_result.value, packed_args)
    return FunctionalDifferentialSolution(
        plan=plan,
        problem=problem,
        unknowns=backend_result.value,
        blocks=final_blocks,
        backend_result=backend_result,
        resolved_method=resolved_method,
        nonlinear_solver=nonlinear_solver,
    )


__all__ = [
    "FunctionalCollocationPlan",
    "FunctionalDifferentialBoundaryProblem",
    "FunctionalDifferentialContext",
    "FunctionalDifferentialSolution",
    "solve_functional_differential",
]
