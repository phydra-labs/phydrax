#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from operator import index
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._geometry_precision import GeometryPrecisionPolicy
from ..._strict import AbstractAttribute, StrictModule
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    GMRES,
    LinearCapabilityError,
    LinearSolvePolicy,
    OperatorCapabilities,
    OperatorProperties,
    TolerancePolicy,
)
from ...linalg.eigen import (
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenSelection,
    GeneralEigenSolvePolicy,
    GeneralEigenTolerancePolicy,
    RestartedArnoldi,
)
from ...nonlinear import (
    JacobianPolicy,
    NewtonKrylov,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
    root,
    RootLineSearch,
)
from .._evolution import AbstractDifferentiableEvolution
from .._layout import StateLayout


PeriodicOrbitKind: TypeAlias = Literal["flow", "map"]
PeriodicLinearMethod: TypeAlias = Literal["dense", "matrix_free"]
FloquetMethod: TypeAlias = Literal["full", "leading"]
FloquetStability: TypeAlias = Literal[
    "stable", "unstable", "marginal", "partial", "invalid"
]

PERIODIC_SUCCESS = 0
PERIODIC_MAX_ITERATIONS = 1
PERIODIC_EVOLUTION_FAILED = 2
PERIODIC_LINEAR_SOLVE_FAILED = 3
PERIODIC_NONFINITE = 4
PERIODIC_LINE_SEARCH_FAILED = 5

FLOQUET_SUCCESS = 0
FLOQUET_INVALID_ORBIT = 1
FLOQUET_TANGENT_FAILED = 2
FLOQUET_NONFINITE = 3
FLOQUET_KRYLOV_BREAKDOWN = 4
FLOQUET_NEUTRAL_MISSING = 5


class AbstractPhaseCondition(StrictModule):
    """One scalar gauge condition removing autonomous-flow phase degeneracy."""

    state_layout: AbstractAttribute[StateLayout]
    phase_id: AbstractAttribute[str]

    @abc.abstractmethod
    def evaluate(
        self,
        state: ArrayLike,
        period: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        raise NotImplementedError


class OrthogonalityPhaseCondition(AbstractPhaseCondition):
    """Hyperplane through a reference point orthogonal to a reference tangent."""

    reference_state: Array
    reference_tangent: Array
    state_layout: StateLayout
    phase_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_state: ArrayLike,
        reference_tangent: ArrayLike,
        /,
        *,
        state_layout: StateLayout,
        phase_id: str | None = None,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        state = jnp.asarray(reference_state)
        tangent = jnp.asarray(reference_tangent)
        if state.shape != state_layout.shape or tangent.shape != state_layout.shape:
            raise ValueError(
                "Reference state and tangent must have the state layout shape."
            )
        if (
            state.dtype != tangent.dtype
            or not jnp.issubdtype(state.dtype, jnp.inexact)
            or jnp.iscomplexobj(state)
        ):
            raise TypeError(
                "Reference state and tangent must share one real inexact dtype."
            )
        if not bool(
            jnp.all(jnp.isfinite(state))
            & jnp.all(jnp.isfinite(tangent))
            & (GeometryPrecisionPolicy().norm(tangent.reshape((-1,))) > 0.0)
        ):
            raise ValueError("Reference state and nonzero tangent must be finite.")
        identifier = (
            "orthogonality-phase:"
            + canonical_fingerprint(
                {
                    "state": array_tree_fingerprint(state),
                    "tangent": array_tree_fingerprint(tangent),
                    "layout": state_layout.layout_id,
                }
            )
            if phase_id is None
            else str(phase_id)
        )
        if not identifier:
            raise ValueError("phase_id must be non-empty.")
        self.reference_state = state
        self.reference_tangent = tangent
        self.state_layout = state_layout
        self.phase_id = identifier

    def evaluate(
        self,
        state: ArrayLike,
        period: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        del period, args
        state_value = jnp.asarray(state)
        return jnp.vdot(
            self.reference_tangent.reshape((-1,)),
            (state_value - self.reference_state).reshape((-1,)),
        ).real


class ComponentPhaseCondition(AbstractPhaseCondition):
    """Pin one flattened state component to a declared section value."""

    value: Array
    state_layout: StateLayout
    component: int = eqx.field(static=True)
    phase_id: str = eqx.field(static=True)

    def __init__(
        self,
        component: int,
        value: ArrayLike,
        /,
        *,
        state_layout: StateLayout,
        phase_id: str | None = None,
    ):
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if isinstance(component, bool):
            raise TypeError("component must be an integer.")
        index_ = index(component)
        if index_ < 0 or index_ >= state_layout.size:
            raise ValueError("component is out of range.")
        resolved_value = jnp.asarray(value)
        if (
            resolved_value.shape != ()
            or jnp.iscomplexobj(resolved_value)
            or not jnp.issubdtype(resolved_value.dtype, jnp.inexact)
            or not bool(jnp.isfinite(resolved_value))
        ):
            raise ValueError("value must be one finite real inexact scalar.")
        identifier = (
            f"component-phase:index={index_}:value={float(resolved_value):.17g}:"
            f"layout={state_layout.layout_id}"
            if phase_id is None
            else str(phase_id)
        )
        if not identifier:
            raise ValueError("phase_id must be non-empty.")
        self.value = resolved_value
        self.state_layout = state_layout
        self.component = index_
        self.phase_id = identifier

    def evaluate(
        self,
        state: ArrayLike,
        period: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        del period, args
        return jnp.asarray(state).reshape((-1,))[self.component] - self.value


class PeriodicOrbitProblem(StrictModule):
    """Multiple-shooting orbit equations over one differentiable evolution."""

    evolution: AbstractDifferentiableEvolution
    phase_condition: AbstractPhaseCondition | None
    kind: PeriodicOrbitKind = eqx.field(static=True)
    num_segments: int = eqx.field(static=True)
    start_coordinate: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        evolution: AbstractDifferentiableEvolution,
        /,
        *,
        kind: PeriodicOrbitKind,
        num_segments: int,
        phase_condition: AbstractPhaseCondition | None = None,
        start_coordinate: float = 0.0,
        problem_id: str | None = None,
    ):
        if not isinstance(evolution, AbstractDifferentiableEvolution):
            raise TypeError("evolution must be an AbstractDifferentiableEvolution.")
        if kind not in ("flow", "map"):
            raise ValueError("kind must be 'flow' or 'map'.")
        if isinstance(num_segments, bool):
            raise TypeError("num_segments must be an integer.")
        segments = index(num_segments)
        start = float(start_coordinate)
        if segments < 1:
            raise ValueError("num_segments must be positive.")
        if not np.isfinite(start):
            raise ValueError("start_coordinate must be finite.")
        if kind == "flow":
            if not isinstance(phase_condition, AbstractPhaseCondition):
                raise TypeError("Flow periodic orbits require an AbstractPhaseCondition.")
            if phase_condition.state_layout.layout_id != evolution.state_layout.layout_id:
                raise ValueError(
                    "Phase condition and evolution state layouts must match."
                )
        elif phase_condition is not None:
            raise ValueError("Map periodic orbits do not use a phase condition.")
        identifier = (
            "periodic-orbit:"
            + canonical_fingerprint(
                {
                    "evolution": evolution.evolution_id,
                    "kind": kind,
                    "segments": segments,
                    "start": start,
                    "phase": None
                    if phase_condition is None
                    else phase_condition.phase_id,
                }
            )
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.evolution = evolution
        self.phase_condition = phase_condition
        self.kind = kind
        self.num_segments = segments
        self.start_coordinate = start
        self.problem_id = identifier

    @property
    def state_layout(self) -> StateLayout:
        return self.evolution.state_layout


class PeriodicOrbitHistory(StrictModule):
    residual_norm: Array
    step_norm: Array
    accepted_scale: Array
    valid: Array


class PeriodicOrbitResult(StrictModule):
    """Multiple-shooting nodes, closure residuals, and Newton evidence."""

    nodes: Array
    period: Array
    residual: Array
    residual_norm: Array
    converged: Array
    valid: Array
    status: Array
    iterations: Array
    history: PeriodicOrbitHistory
    problem: PeriodicOrbitProblem
    nonlinear_result: NonlinearResult
    method_id: str = eqx.field(static=True)

    @property
    def initial_state(self) -> Array:
        return self.nodes[0]


class MonodromyActionResult(StrictModule):
    tangent: Array
    valid: Array
    status: Array
    method_id: str = eqx.field(static=True)


class FloquetIndicators(StrictModule):
    unit_multiplier_distance: Array
    period_doubling_distance: Array
    unit_circle_distance: Array


class FloquetResult(StrictModule):
    """Full or leading Floquet multipliers, modes, exponents, and stability evidence."""

    multipliers: Array
    exponents: Array
    modes: Array
    monodromy_matrix: Array | None
    indicators: FloquetIndicators
    spectral_radius: Array
    neutral_index: Array
    valid: Array
    status: Array
    orbit: PeriodicOrbitResult
    method: FloquetMethod = eqx.field(static=True)
    eigen_result: Any
    stability: FloquetStability = eqx.field(static=True)
    method_id: str = eqx.field(static=True)


def _segment_coordinates(
    problem: PeriodicOrbitProblem,
    period: Array,
    segment: int,
    /,
) -> tuple[Array, Array]:
    if problem.kind == "map":
        source = jnp.asarray(problem.start_coordinate + segment)
        return source, source + 1.0
    source = problem.start_coordinate + period * segment / problem.num_segments
    target = problem.start_coordinate + period * (segment + 1) / problem.num_segments
    return jnp.asarray(source), jnp.asarray(target)


def periodic_nodes_from_state(
    problem: PeriodicOrbitProblem,
    initial_state: ArrayLike,
    /,
    *,
    period: ArrayLike | None = None,
    args: Any = None,
) -> Array:
    """Propagate one state to initialize all multiple-shooting nodes."""
    if not isinstance(problem, PeriodicOrbitProblem):
        raise TypeError("problem must be a PeriodicOrbitProblem.")
    state = jnp.asarray(initial_state)
    if state.shape != problem.state_layout.shape:
        raise ValueError("initial_state must have the problem state layout shape.")
    if jnp.iscomplexobj(state) or not jnp.issubdtype(state.dtype, jnp.inexact):
        raise TypeError("Periodic node initialization requires real inexact state.")
    if not bool(jnp.all(jnp.isfinite(state))):
        raise ValueError("initial_state must be finite.")
    if problem.kind == "flow":
        if period is None:
            raise ValueError("Flow node initialization requires a period.")
        resolved_period = jnp.asarray(period)
        if resolved_period.shape != () or not bool(
            jnp.isfinite(resolved_period) & (resolved_period > 0.0)
        ):
            raise ValueError("period must be one finite positive scalar.")
    else:
        if period is not None:
            raise ValueError("Map node initialization does not accept period.")
        resolved_period = jnp.asarray(float(problem.num_segments))
    nodes = []
    current = state
    for segment in range(problem.num_segments):
        nodes.append(current)
        source, target = _segment_coordinates(problem, resolved_period, segment)
        advanced = problem.evolution.advance(current, source, target, args)
        if not bool(advanced.valid):
            raise ValueError("Evolution failed while constructing periodic nodes.")
        current = advanced.final_state
    return jnp.stack(tuple(nodes))


def _pack(
    problem: PeriodicOrbitProblem,
    nodes: Array,
    period: Array,
    /,
) -> Array:
    flat_nodes = nodes.reshape((-1,))
    if problem.kind == "flow":
        return jnp.concatenate((flat_nodes, jnp.log(period)[None]))
    return flat_nodes


def _unpack(
    problem: PeriodicOrbitProblem,
    values: Array,
    /,
) -> tuple[Array, Array]:
    node_size = problem.num_segments * problem.state_layout.size
    nodes = values[:node_size].reshape(
        (problem.num_segments,) + problem.state_layout.shape
    )
    period = (
        jnp.exp(values[node_size])
        if problem.kind == "flow"
        else jnp.asarray(float(problem.num_segments), dtype=values.dtype)
    )
    return nodes, period


def _orbit_residual(
    values: Array,
    problem: PeriodicOrbitProblem,
    args: Any,
    /,
) -> Array:
    nodes, period = _unpack(problem, values)
    pieces = []
    for segment in range(problem.num_segments):
        source, target = _segment_coordinates(problem, period, segment)
        advanced = problem.evolution.advance(nodes[segment], source, target, args)
        pieces.append(
            (advanced.final_state - nodes[(segment + 1) % problem.num_segments]).reshape(
                (-1,)
            )
        )
    if problem.kind == "flow":
        phase_condition = problem.phase_condition
        if phase_condition is None:
            raise RuntimeError("Flow orbit problem is missing its phase condition.")
        phase = phase_condition.evaluate(nodes[0], period, args)
        pieces.append(jnp.asarray(phase).reshape((1,)))
    return jnp.concatenate(tuple(pieces))


def _orbit_evolution_valid(
    values: Array,
    problem: PeriodicOrbitProblem,
    args: Any,
    /,
) -> Array:
    nodes, period = _unpack(problem, values)
    valid = jnp.isfinite(period) & (period > 0.0)
    for segment in range(problem.num_segments):
        source, target = _segment_coordinates(problem, period, segment)
        advanced = problem.evolution.advance(nodes[segment], source, target, args)
        valid = valid & advanced.valid
    return valid


class PeriodicOrbitResidual(StrictModule):
    """Public real-coordinate multiple-shooting residual adapter."""

    problem: PeriodicOrbitProblem
    residual_id: str = eqx.field(static=True)

    def __init__(self, problem: PeriodicOrbitProblem, /):
        if not isinstance(problem, PeriodicOrbitProblem):
            raise TypeError("problem must be a PeriodicOrbitProblem.")
        self.problem = problem
        self.residual_id = f"{problem.problem_id}:multiple-shooting-residual"

    @property
    def unknown_size(self) -> int:
        return self.problem.num_segments * self.problem.state_layout.size + int(
            self.problem.kind == "flow"
        )

    def pack(self, nodes: ArrayLike, period: ArrayLike, /) -> Array:
        return _pack(self.problem, jnp.asarray(nodes), jnp.asarray(period))

    def unpack(self, values: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(values)
        if value.shape != (self.unknown_size,):
            raise ValueError(f"Periodic unknowns must have shape {(self.unknown_size,)}.")
        return _unpack(self.problem, value)

    def residual(self, values: Array, args: Any = None, /) -> Array:
        return _orbit_residual(values, self.problem, args)

    def valid(
        self,
        values: Array,
        residual: Array,
        auxiliary: Any,
        args: Any = None,
        /,
    ) -> Array:
        del auxiliary
        return (
            _orbit_evolution_valid(values, self.problem, args)
            & jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(residual))
        )

    def as_nonlinear_problem(self, dtype: Any, /) -> NonlinearSystemProblem:
        space = ArraySpace((self.unknown_size,), dtype=dtype)
        return NonlinearSystemProblem(
            self.residual,
            state_space=space,
            residual_space=space,
            validity=self.valid,
            problem_id=self.residual_id,
        )


def solve_periodic_orbit(
    problem: PeriodicOrbitProblem,
    initial_nodes: ArrayLike,
    /,
    *,
    initial_period: ArrayLike | None = None,
    args: Any = None,
    linear_method: PeriodicLinearMethod = "dense",
    max_iterations: int = 20,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_line_search: int = 12,
    min_step_scale: float = 2.0**-12,
    max_dense_dimension: int = 512,
    krylov_tolerance: float = 1e-8,
    krylov_max_iterations: int = 256,
) -> PeriodicOrbitResult:
    """Solve real-coordinate multiple shooting through the shared nonlinear runtime."""
    if not isinstance(problem, PeriodicOrbitProblem):
        raise TypeError("problem must be a PeriodicOrbitProblem.")
    if linear_method not in ("dense", "matrix_free"):
        raise ValueError("linear_method must be 'dense' or 'matrix_free'.")
    integer_values = (
        max_iterations,
        max_line_search,
        max_dense_dimension,
        krylov_max_iterations,
    )
    if any(isinstance(value, bool) for value in integer_values):
        raise TypeError("Periodic solver capacities must be integers.")
    iterations_limit = index(max_iterations)
    line_search_limit = index(max_line_search)
    dense_limit = index(max_dense_dimension)
    krylov_limit = index(krylov_max_iterations)
    if min(iterations_limit, line_search_limit, dense_limit, krylov_limit) < 1:
        raise ValueError("Periodic solver capacities must be positive.")
    relative_tolerance = float(rtol)
    absolute_tolerance = float(atol)
    minimum_scale = float(min_step_scale)
    if (
        not np.isfinite(relative_tolerance)
        or not np.isfinite(absolute_tolerance)
        or relative_tolerance < 0.0
        or absolute_tolerance <= 0.0
        or not 0.0 < minimum_scale <= 1.0
    ):
        raise ValueError("Periodic solver tolerances or min_step_scale are invalid.")
    nodes = jnp.asarray(initial_nodes)
    if jnp.issubdtype(nodes.dtype, jnp.complexfloating):
        raise TypeError(
            "Periodic orbit solves require independent real coordinates; wrap a "
            "full-complex spectral evolution with HermitianCoordinateEvolution."
        )
    expected = (problem.num_segments,) + problem.state_layout.shape
    if not jnp.issubdtype(nodes.dtype, jnp.inexact):
        raise TypeError("Periodic orbit nodes must use an inexact dtype.")
    if not bool(jnp.all(jnp.isfinite(nodes))):
        raise ValueError("initial_nodes must be finite.")
    if nodes.shape == problem.state_layout.shape:
        nodes = periodic_nodes_from_state(
            problem,
            nodes,
            period=initial_period,
            args=args,
        )
    elif nodes.shape != expected:
        raise ValueError(
            f"initial_nodes must have shape {problem.state_layout.shape} or {expected}."
        )
    if problem.kind == "flow":
        if initial_period is None:
            raise ValueError("Flow periodic solves require initial_period.")
        period = jnp.asarray(initial_period, dtype=nodes.dtype)
        if period.shape != () or not bool(jnp.isfinite(period) & (period > 0.0)):
            raise ValueError("initial_period must be finite and positive.")
    else:
        if initial_period is not None:
            raise ValueError("Map periodic solves do not accept initial_period.")
        period = jnp.asarray(float(problem.num_segments), dtype=nodes.dtype)
    adapter = PeriodicOrbitResidual(problem)
    values = adapter.pack(nodes, period)
    dimension = int(values.size)
    if linear_method == "dense" and dimension > dense_limit:
        raise ValueError(
            f"Dense periodic solve dimension {dimension} exceeds "
            f"max_dense_dimension={dense_limit}."
        )
    linear_policy = (
        LinearSolvePolicy(DenseLU())
        if linear_method == "dense"
        else LinearSolvePolicy(
            GMRES(),
            tolerance=TolerancePolicy(
                relative=krylov_tolerance,
                absolute=0.0,
                max_steps=krylov_limit,
            ),
        )
    )
    method = NewtonKrylov(
        jacobian_policy=JacobianPolicy(),
        linear_policy=linear_policy,
        line_search=RootLineSearch(
            minimum_rate=minimum_scale,
            maximum_steps=line_search_limit,
        ),
    )
    termination = NonlinearTermination(
        absolute_residual=absolute_tolerance,
        relative_residual=relative_tolerance,
        maximum_steps=iterations_limit,
        maximum_evaluations=1 + iterations_limit * (line_search_limit + 2),
        maximum_linear_iterations=iterations_limit * max(krylov_limit, dimension),
    )
    nonlinear_result = root(
        adapter.as_nonlinear_problem(values.dtype),
        values,
        method=method,
        termination=termination,
        args=args,
    )
    final_values = jnp.asarray(nonlinear_result.state)
    final_nodes, final_period = adapter.unpack(final_values)
    nonlinear_status = nonlinear_result.status
    linear_failure = (
        (nonlinear_status == int(NonlinearStatus.LINEAR_SOLVE_FAILED))
        | (nonlinear_status == int(NonlinearStatus.SINGULAR_JACOBIAN))
        | (nonlinear_status == int(NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED))
    )
    globalization_failure = (
        nonlinear_status == int(NonlinearStatus.LINE_SEARCH_FAILED)
    ) | (nonlinear_status == int(NonlinearStatus.TRUST_REGION_FAILED))
    nonfinite = (nonlinear_status == int(NonlinearStatus.NONFINITE_INPUT)) | (
        nonlinear_status == int(NonlinearStatus.NONFINITE_EVALUATION)
    )
    periodic_status = jnp.where(
        nonlinear_result.successful,
        PERIODIC_SUCCESS,
        jnp.where(
            linear_failure,
            PERIODIC_LINEAR_SOLVE_FAILED,
            jnp.where(
                globalization_failure,
                PERIODIC_LINE_SEARCH_FAILED,
                jnp.where(nonfinite, PERIODIC_NONFINITE, PERIODIC_MAX_ITERATIONS),
            ),
        ),
    ).astype(jnp.int32)
    evolution_valid = _orbit_evolution_valid(final_values, problem, args)
    finite = jnp.all(jnp.isfinite(final_nodes)) & jnp.isfinite(final_period)
    periodic_status = jnp.where(
        (periodic_status == PERIODIC_SUCCESS) & ~evolution_valid,
        PERIODIC_EVOLUTION_FAILED,
        periodic_status,
    )
    valid = nonlinear_result.successful & evolution_valid & finite
    diagnostics = nonlinear_result.diagnostics
    history = PeriodicOrbitHistory(
        residual_norm=jnp.stack(
            (diagnostics.initial_residual_norm, diagnostics.final_residual_norm)
        ),
        step_norm=jnp.stack(
            (jnp.zeros_like(diagnostics.final_step_norm), diagnostics.final_step_norm)
        ),
        accepted_scale=jnp.asarray(
            (0.0, jnp.nan), dtype=diagnostics.final_residual_norm.dtype
        ),
        valid=jnp.stack((jnp.asarray(True), valid)),
    )
    return PeriodicOrbitResult(
        nodes=final_nodes,
        period=final_period,
        residual=jnp.asarray(nonlinear_result.residual),
        residual_norm=diagnostics.final_residual_norm,
        converged=valid,
        valid=valid,
        status=periodic_status,
        iterations=diagnostics.iterations,
        history=history,
        problem=problem,
        nonlinear_result=nonlinear_result,
        method_id=f"multiple-shooting:{nonlinear_result.provenance.method_id}",
    )


def monodromy_action(
    orbit: PeriodicOrbitResult,
    tangent: ArrayLike,
    /,
    *,
    args: Any = None,
) -> MonodromyActionResult:
    """Propagate one tangent through every converged shooting segment."""
    if not isinstance(orbit, PeriodicOrbitResult):
        raise TypeError("orbit must be a PeriodicOrbitResult.")
    tangent_value = jnp.asarray(tangent)
    if tangent_value.shape != orbit.problem.state_layout.shape:
        raise ValueError("tangent must have the orbit state layout shape.")
    valid = orbit.valid & jnp.all(jnp.isfinite(tangent_value))
    current = tangent_value
    for segment in range(orbit.problem.num_segments):
        source, target = _segment_coordinates(orbit.problem, orbit.period, segment)
        propagated = orbit.problem.evolution.tangent_action(
            orbit.nodes[segment], current, source, target, args
        )
        current = propagated.tangent
        valid = valid & propagated.valid
    finite = jnp.all(jnp.isfinite(current))
    valid = valid & finite
    status = jnp.where(
        ~finite,
        FLOQUET_NONFINITE,
        jnp.where(valid, FLOQUET_SUCCESS, FLOQUET_TANGENT_FAILED),
    ).astype(jnp.int32)
    return MonodromyActionResult(
        tangent=current,
        valid=valid,
        status=status,
        method_id=f"monodromy:{orbit.problem.evolution.tangent_method_id}",
    )


class _MonodromyLinearOperator(AbstractLinearOperator):
    orbit: PeriodicOrbitResult
    args: Any

    def __init__(self, orbit: PeriodicOrbitResult, args: Any, /):
        self.source = ArraySpace(
            (orbit.problem.state_layout.size,), dtype=orbit.nodes.dtype
        )
        self.target = self.source
        self.orbit = orbit
        self.args = args
        self.properties = OperatorProperties()
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = f"{orbit.problem.problem_id}:monodromy"

    def mv(self, vector: ArrayLike, /) -> Array:
        values = self.source.validate(vector)
        result = monodromy_action(
            self.orbit,
            values.reshape(self.orbit.problem.state_layout.shape),
            args=self.args,
        )
        return result.tangent.reshape((-1,))

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        values = self.target.validate(vector)
        zero = self.source.zeros()
        return jax.linear_transpose(self.mv, zero)(values)[0]

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        values = self.target.validate(vector)
        return jnp.conj(self.transpose_mv(jnp.conj(values)))

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError("Monodromy materialization must be explicit.")


def floquet_spectrum(
    orbit: PeriodicOrbitResult,
    /,
    *,
    args: Any = None,
    method: FloquetMethod = "full",
    leading_k: int | None = None,
    krylov_dimension: int | None = None,
    stability_tolerance: float = 1e-6,
    max_full_dimension: int = 128,
) -> FloquetResult:
    """Compute Floquet multipliers through the shared general-eigen runtime."""
    if not isinstance(orbit, PeriodicOrbitResult):
        raise TypeError("orbit must be a PeriodicOrbitResult.")
    if method not in ("full", "leading"):
        raise ValueError("method must be 'full' or 'leading'.")
    tolerance = float(stability_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("stability_tolerance must be finite and positive.")
    dimension = orbit.problem.state_layout.size
    if isinstance(max_full_dimension, bool):
        raise TypeError("max_full_dimension must be an integer.")
    full_limit = index(max_full_dimension)
    if full_limit < 1:
        raise ValueError("max_full_dimension must be positive.")
    if leading_k is not None and isinstance(leading_k, bool):
        raise TypeError("leading_k must be an integer or None.")
    requested_count = (
        dimension
        if method == "full"
        else min(dimension, 1 if leading_k is None else index(leading_k))
    )
    if requested_count < 1:
        raise ValueError("leading_k must be positive or None.")
    if krylov_dimension is not None and isinstance(krylov_dimension, bool):
        raise TypeError("krylov_dimension must be an integer or None.")
    monodromy = None
    eigen_result = None
    status = FLOQUET_SUCCESS
    if not bool(orbit.valid):
        multipliers = jnp.full((requested_count,), jnp.nan + 0.0j)
        modes = jnp.full((dimension, requested_count), jnp.nan + 0.0j)
        status = FLOQUET_INVALID_ORBIT
    elif method == "full":
        if dimension > full_limit:
            raise ValueError(
                f"Full Floquet dimension {dimension} exceeds "
                f"max_full_dimension={full_limit}."
            )
        columns = []
        action_valid = True
        for column in range(dimension):
            basis = jnp.zeros((dimension,), dtype=orbit.nodes.dtype).at[column].set(1.0)
            action = monodromy_action(
                orbit, basis.reshape(orbit.problem.state_layout.shape), args=args
            )
            columns.append(action.tangent.reshape((-1,)))
            action_valid = action_valid and bool(action.valid)
        monodromy = jnp.stack(tuple(columns), axis=-1)
        eigen_result = general_eigensolve(
            GeneralEigenproblem(
                DenseLinearOperator(monodromy),
                problem_id=f"{orbit.problem.problem_id}:floquet-full",
            )
        )
        multipliers = eigen_result.eigenvalues
        modes = eigen_result.right_eigenvector_coordinates
        if not action_valid:
            status = FLOQUET_TANGENT_FAILED
        elif not bool(eigen_result.successful):
            status = FLOQUET_NONFINITE
    else:
        count = requested_count
        subspace = (
            min(dimension, max(count + 4, 2 * count))
            if krylov_dimension is None
            else index(krylov_dimension)
        )
        if subspace < count or subspace > dimension:
            raise ValueError(
                "krylov_dimension must lie between leading_k and state size."
            )
        eigen_result = general_eigensolve(
            GeneralEigenproblem(
                _MonodromyLinearOperator(orbit, args),
                problem_id=f"{orbit.problem.problem_id}:floquet-leading",
            ),
            policy=GeneralEigenSolvePolicy(
                RestartedArnoldi(subspace_dimension=subspace),
                selection=GeneralEigenSelection(
                    "largest-magnitude",
                    count=count,
                ),
                max_steps=max(4 * subspace, dimension),
                tolerance=GeneralEigenTolerancePolicy(
                    relative=min(tolerance, 1e-8),
                    absolute=0.0,
                ),
            ),
        )
        multipliers = eigen_result.eigenvalues
        modes = eigen_result.right_eigenvector_coordinates
        if not bool(eigen_result.successful):
            status = FLOQUET_KRYLOV_BREAKDOWN
    finite = jnp.all(jnp.isfinite(multipliers)) & jnp.all(jnp.isfinite(modes))
    if status == FLOQUET_SUCCESS and not bool(finite):
        status = FLOQUET_NONFINITE
    complete_spectrum = method == "full" or requested_count == dimension
    neutral_certified = False
    if (
        orbit.problem.kind == "flow"
        and multipliers.size
        and complete_spectrum
        and status == FLOQUET_SUCCESS
    ):
        neutral_index = jnp.argmin(jnp.abs(multipliers - 1.0)).astype(jnp.int32)
        neutral_distance = jnp.abs(multipliers[neutral_index] - 1.0)
        neutral_certified = bool(neutral_distance <= tolerance)
        if not neutral_certified:
            status = FLOQUET_NEUTRAL_MISSING
    else:
        neutral_index = jnp.asarray(-1, dtype=jnp.int32)
    valid = jnp.asarray(status == FLOQUET_SUCCESS) & finite
    interval = orbit.period
    exponents = jnp.log(jnp.abs(multipliers)) / interval
    included = jnp.ones(multipliers.shape, dtype=bool)
    if neutral_certified:
        included = included.at[neutral_index].set(False)
    relevant = jnp.where(included, jnp.abs(multipliers), 0.0)
    spectral_radius = jnp.max(relevant, initial=0.0)
    if not bool(valid):
        stability: FloquetStability = "invalid"
    elif method == "leading" and multipliers.size < dimension:
        stability = "unstable" if float(spectral_radius) > 1.0 + tolerance else "partial"
    elif float(spectral_radius) < 1.0 - tolerance:
        stability = "stable"
    elif float(spectral_radius) > 1.0 + tolerance:
        stability = "unstable"
    else:
        stability = "marginal"
    indicator_values = jnp.where(included, multipliers, jnp.nan + 0.0j)
    unit_distance = jnp.nanmin(jnp.abs(indicator_values - 1.0), initial=jnp.inf)
    doubling_distance = jnp.nanmin(jnp.abs(indicator_values + 1.0), initial=jnp.inf)
    circle_distance = jnp.nanmin(
        jnp.abs(jnp.abs(indicator_values) - 1.0), initial=jnp.inf
    )
    return FloquetResult(
        multipliers=multipliers,
        exponents=exponents,
        modes=modes.reshape(orbit.problem.state_layout.shape + (-1,)),
        monodromy_matrix=monodromy,
        indicators=FloquetIndicators(
            unit_multiplier_distance=unit_distance,
            period_doubling_distance=doubling_distance,
            unit_circle_distance=circle_distance,
        ),
        spectral_radius=spectral_radius,
        neutral_index=neutral_index,
        valid=valid,
        status=jnp.asarray(status, dtype=jnp.int32),
        orbit=orbit,
        method=method,
        stability=stability,
        method_id=(
            f"floquet:{method}:"
            + ("not-run" if eigen_result is None else eigen_result.provenance.backend)
        ),
        eigen_result=eigen_result,
    )


__all__ = [
    "AbstractPhaseCondition",
    "ComponentPhaseCondition",
    "FLOQUET_INVALID_ORBIT",
    "FLOQUET_KRYLOV_BREAKDOWN",
    "FLOQUET_NONFINITE",
    "FLOQUET_SUCCESS",
    "FLOQUET_TANGENT_FAILED",
    "FLOQUET_NEUTRAL_MISSING",
    "FloquetIndicators",
    "FloquetMethod",
    "FloquetResult",
    "FloquetStability",
    "MonodromyActionResult",
    "OrthogonalityPhaseCondition",
    "PERIODIC_EVOLUTION_FAILED",
    "PERIODIC_LINEAR_SOLVE_FAILED",
    "PERIODIC_LINE_SEARCH_FAILED",
    "PERIODIC_MAX_ITERATIONS",
    "PERIODIC_NONFINITE",
    "PERIODIC_SUCCESS",
    "PeriodicLinearMethod",
    "PeriodicOrbitHistory",
    "PeriodicOrbitKind",
    "PeriodicOrbitProblem",
    "PeriodicOrbitResidual",
    "PeriodicOrbitResult",
    "floquet_spectrum",
    "monodromy_action",
    "periodic_nodes_from_state",
    "solve_periodic_orbit",
]
