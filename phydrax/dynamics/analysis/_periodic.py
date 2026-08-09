#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
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
        if not bool(
            jnp.all(jnp.isfinite(state))
            & jnp.all(jnp.isfinite(tangent))
            & (jnp.linalg.norm(tangent.reshape((-1,))) > 0.0)
        ):
            raise ValueError("Reference state and nonzero tangent must be finite.")
        identifier = (
            "orthogonality-phase:"
            + canonical_fingerprint(
                {
                    "state": np.asarray(state).tolist(),
                    "tangent": np.asarray(tangent).tolist(),
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
        index = int(component)
        if index < 0 or index >= state_layout.size:
            raise ValueError("component is out of range.")
        resolved_value = jnp.asarray(value)
        if resolved_value.shape != () or not bool(jnp.isfinite(resolved_value)):
            raise ValueError("value must be one finite scalar.")
        identifier = (
            f"component-phase:index={index}:value={float(resolved_value):.17g}:"
            f"layout={state_layout.layout_id}"
            if phase_id is None
            else str(phase_id)
        )
        if not identifier:
            raise ValueError("phase_id must be non-empty.")
        self.value = resolved_value
        self.state_layout = state_layout
        self.component = index
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
        segments = int(num_segments)
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
        phase = problem.phase_condition.evaluate(nodes[0], period, args)
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
    """Solve square multiple-shooting equations by damped dense or JVP Newton."""
    if not isinstance(problem, PeriodicOrbitProblem):
        raise TypeError("problem must be a PeriodicOrbitProblem.")
    if linear_method not in ("dense", "matrix_free"):
        raise ValueError("linear_method must be 'dense' or 'matrix_free'.")
    iterations_limit = int(max_iterations)
    line_search_limit = int(max_line_search)
    if iterations_limit < 1 or line_search_limit < 1:
        raise ValueError("max_iterations and max_line_search must be positive.")
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
    expected = (problem.num_segments,) + problem.state_layout.shape
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
        period = jnp.asarray(initial_period)
        if period.shape != () or not bool(jnp.isfinite(period) & (period > 0.0)):
            raise ValueError("initial_period must be finite and positive.")
    else:
        if initial_period is not None:
            raise ValueError("Map periodic solves do not accept initial_period.")
        period = jnp.asarray(float(problem.num_segments), dtype=nodes.dtype)
    values = _pack(problem, nodes, period)
    dimension = int(values.size)
    if linear_method == "dense" and dimension > int(max_dense_dimension):
        raise ValueError(
            f"Dense periodic solve dimension {dimension} exceeds "
            f"max_dense_dimension={int(max_dense_dimension)}."
        )

    def residual_function(candidate: Array) -> Array:
        return _orbit_residual(candidate, problem, args)

    residual = residual_function(values)
    residual_norm = jnp.linalg.norm(residual)
    initial_norm = residual_norm
    threshold = absolute_tolerance + relative_tolerance * initial_norm
    history_residual = [residual_norm]
    history_steps = [jnp.asarray(0.0, dtype=residual_norm.dtype)]
    history_scales = [jnp.asarray(0.0, dtype=residual_norm.dtype)]
    history_valid = [
        _orbit_evolution_valid(values, problem, args) & jnp.all(jnp.isfinite(residual))
    ]
    converged = bool(history_valid[-1] & (residual_norm <= threshold))
    status = PERIODIC_SUCCESS if converged else PERIODIC_MAX_ITERATIONS
    completed_iterations = 0
    for iteration in range(1, iterations_limit + 1):
        if converged:
            break
        if not bool(history_valid[-1]):
            status = (
                PERIODIC_NONFINITE
                if not bool(jnp.all(jnp.isfinite(residual)))
                else PERIODIC_EVOLUTION_FAILED
            )
            break
        if linear_method == "dense":
            jacobian = jax.jacfwd(residual_function)(values)
            step = jnp.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            linear_valid = jnp.all(jnp.isfinite(jacobian)) & jnp.all(jnp.isfinite(step))
        else:
            _, tangent = jax.linearize(residual_function, values)
            step, information = jsp.sparse.linalg.gmres(
                tangent,
                -residual,
                tol=float(krylov_tolerance),
                atol=0.0,
                maxiter=int(krylov_max_iterations),
            )
            linear_valid = (information == 0) & jnp.all(jnp.isfinite(step))
        if not bool(linear_valid):
            status = PERIODIC_LINEAR_SOLVE_FAILED
            break
        step_norm = jnp.linalg.norm(step)
        scale = 1.0
        accepted = False
        candidate_values = values
        candidate_residual = residual
        candidate_norm = residual_norm
        candidate_valid = jnp.asarray(False)
        for _ in range(line_search_limit):
            trial_values = values + scale * step
            trial_residual = residual_function(trial_values)
            trial_valid = _orbit_evolution_valid(trial_values, problem, args) & jnp.all(
                jnp.isfinite(trial_residual)
            )
            trial_norm = jnp.linalg.norm(trial_residual)
            if bool(trial_valid & (trial_norm < residual_norm)):
                candidate_values = trial_values
                candidate_residual = trial_residual
                candidate_norm = trial_norm
                candidate_valid = trial_valid
                accepted = True
                break
            scale *= 0.5
            if scale < minimum_scale:
                break
        if not accepted:
            status = PERIODIC_LINE_SEARCH_FAILED
            break
        values = candidate_values
        residual = candidate_residual
        residual_norm = candidate_norm
        completed_iterations = iteration
        history_residual.append(residual_norm)
        history_steps.append(step_norm)
        history_scales.append(jnp.asarray(scale, dtype=residual_norm.dtype))
        history_valid.append(candidate_valid)
        converged = bool(candidate_valid & (residual_norm <= threshold))
        status = PERIODIC_SUCCESS if converged else PERIODIC_MAX_ITERATIONS
    final_nodes, final_period = _unpack(problem, values)
    valid = (
        jnp.asarray(converged)
        & jnp.all(jnp.isfinite(final_nodes))
        & jnp.isfinite(final_period)
    )
    return PeriodicOrbitResult(
        nodes=final_nodes,
        period=final_period,
        residual=residual,
        residual_norm=residual_norm,
        converged=jnp.asarray(converged),
        valid=valid,
        status=jnp.asarray(status, dtype=jnp.int32),
        iterations=jnp.asarray(completed_iterations, dtype=jnp.int32),
        history=PeriodicOrbitHistory(
            residual_norm=jnp.stack(tuple(history_residual)),
            step_norm=jnp.stack(tuple(history_steps)),
            accepted_scale=jnp.stack(tuple(history_scales)),
            valid=jnp.stack(tuple(history_valid)),
        ),
        problem=problem,
        method_id=f"multiple-shooting:newton:{linear_method}",
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


def _arnoldi(
    orbit: PeriodicOrbitResult,
    subspace_dimension: int,
    args: Any,
    /,
) -> tuple[Array, Array, bool]:
    size = orbit.problem.state_layout.size
    initial = jnp.arange(1, size + 1, dtype=orbit.nodes.dtype)
    initial = initial / jnp.linalg.norm(initial)
    basis = [initial]
    hessenberg = jnp.zeros(
        (subspace_dimension + 1, subspace_dimension), dtype=orbit.nodes.dtype
    )
    valid = True
    completed = 0
    for column in range(subspace_dimension):
        action = monodromy_action(
            orbit,
            basis[column].reshape(orbit.problem.state_layout.shape),
            args=args,
        )
        if not bool(action.valid):
            valid = False
            break
        vector = action.tangent.reshape((-1,))
        for row in range(column + 1):
            projection = jnp.vdot(basis[row], vector)
            hessenberg = hessenberg.at[row, column].set(projection)
            vector = vector - projection * basis[row]
        norm = jnp.linalg.norm(vector)
        hessenberg = hessenberg.at[column + 1, column].set(norm)
        completed = column + 1
        if column + 1 < subspace_dimension:
            if not bool(jnp.isfinite(norm) & (norm > jnp.finfo(norm.dtype).eps)):
                break
            basis.append(vector / norm)
    square = hessenberg[:completed, :completed]
    if completed == 0:
        return jnp.asarray([], dtype=complex), jnp.empty((size, 0)), False
    multipliers, vectors = jnp.linalg.eig(square)
    basis_matrix = jnp.stack(tuple(basis[:completed]), axis=-1)
    modes = basis_matrix.astype(vectors.dtype) @ vectors
    return multipliers, modes, valid


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
    """Compute full dense or leading matrix-free Floquet multipliers."""
    if not isinstance(orbit, PeriodicOrbitResult):
        raise TypeError("orbit must be a PeriodicOrbitResult.")
    if method not in ("full", "leading"):
        raise ValueError("method must be 'full' or 'leading'.")
    tolerance = float(stability_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("stability_tolerance must be finite and positive.")
    dimension = orbit.problem.state_layout.size
    monodromy = None
    status = FLOQUET_SUCCESS
    if not bool(orbit.valid):
        multipliers = jnp.full((dimension,), jnp.nan + 0.0j)
        modes = jnp.full((dimension, dimension), jnp.nan + 0.0j)
        status = FLOQUET_INVALID_ORBIT
    elif method == "full":
        if dimension > int(max_full_dimension):
            raise ValueError(
                f"Full Floquet dimension {dimension} exceeds "
                f"max_full_dimension={int(max_full_dimension)}."
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
        multipliers, modes = jnp.linalg.eig(monodromy)
        if not action_valid:
            status = FLOQUET_TANGENT_FAILED
    else:
        count = min(dimension, 1 if leading_k is None else int(leading_k))
        if count < 1:
            raise ValueError("leading_k must be positive or None.")
        subspace = (
            min(dimension, max(count + 4, 2 * count))
            if krylov_dimension is None
            else int(krylov_dimension)
        )
        if subspace < count or subspace > dimension:
            raise ValueError(
                "krylov_dimension must lie between leading_k and state size."
            )
        all_multipliers, all_modes, arnoldi_valid = _arnoldi(orbit, subspace, args)
        order = jnp.argsort(jnp.abs(all_multipliers))[::-1][:count]
        multipliers = all_multipliers[order]
        modes = all_modes[:, order]
        if not arnoldi_valid:
            status = FLOQUET_KRYLOV_BREAKDOWN
    finite = jnp.all(jnp.isfinite(multipliers)) & jnp.all(jnp.isfinite(modes))
    if status == FLOQUET_SUCCESS and not bool(finite):
        status = FLOQUET_NONFINITE
    valid = jnp.asarray(status == FLOQUET_SUCCESS) & finite
    interval = orbit.period
    exponents = jnp.log(jnp.abs(multipliers)) / interval
    if orbit.problem.kind == "flow" and multipliers.size:
        neutral_index = jnp.argmin(jnp.abs(multipliers - 1.0)).astype(jnp.int32)
    else:
        neutral_index = jnp.asarray(-1, dtype=jnp.int32)
    included = jnp.ones(multipliers.shape, dtype=bool)
    if orbit.problem.kind == "flow" and multipliers.size:
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
        method_id=f"floquet:{method}:{orbit.problem.evolution.tangent_method_id}",
    )


__all__ = [
    "AbstractPhaseCondition",
    "ComponentPhaseCondition",
    "FLOQUET_INVALID_ORBIT",
    "FLOQUET_KRYLOV_BREAKDOWN",
    "FLOQUET_NONFINITE",
    "FLOQUET_SUCCESS",
    "FLOQUET_TANGENT_FAILED",
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
    "PeriodicOrbitResult",
    "floquet_spectrum",
    "monodromy_action",
    "periodic_nodes_from_state",
    "solve_periodic_orbit",
]
