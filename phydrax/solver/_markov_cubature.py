#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._measure_weights import normalized_weights
from .._polynomial._total_degree import TotalDegreePolynomialFeatures
from .._strict import StrictModule
from ..coresets import moment_recombine, MomentRecombination
from ..discretization import TemporalMesh
from ..integration._rules import GaussianCubatureRule
from ..stochastic._cubature_path import (
    straight_wiener_cubature_path,
    WienerCubaturePathData,
)
from ._differential import DifferentialProblem


MarkovCubatureMethod: TypeAlias = Literal["weak-euler", "stratonovich-flow"]


class MarkovCubatureStatus(IntEnum):
    SUCCESS = 0
    UNSUPPORTED_PROBLEM = 1
    EXPANSION_CAPACITY_EXCEEDED = 2
    NONFINITE_DYNAMICS = 3
    INVALID_WEIGHTS = 4
    RECOMBINATION_FAILED = 5
    MOMENT_RESIDUAL_TOO_LARGE = 6


class PolynomialRecombination(StrictModule):
    """Positive moment recombination over standardized total-degree features."""

    degree: int = eqx.field(static=True)
    method: MomentRecombination
    maximum_features: int = eqx.field(static=True)
    maximum_feature_bytes: int = eqx.field(static=True)
    maximum_moment_error: float = eqx.field(static=True)
    differentiation: str = eqx.field(static=True)
    recombination_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree: int = 2,
        /,
        *,
        method: MomentRecombination | None = None,
        maximum_features: int = 4096,
        maximum_feature_bytes: int = 64 * 1024**2,
        maximum_moment_error: float = 1e-9,
        differentiation: Literal["frozen-selection"] = "frozen-selection",
    ):
        degree_ = _nonnegative_integer(degree, "degree")
        features = _positive_integer(maximum_features, "maximum_features")
        feature_bytes = _positive_integer(maximum_feature_bytes, "maximum_feature_bytes")
        error = float(maximum_moment_error)
        if not math.isfinite(error) or error < 0.0:
            raise ValueError("maximum_moment_error must be finite and nonnegative.")
        selected = MomentRecombination() if method is None else method
        if not isinstance(selected, MomentRecombination):
            raise TypeError("method must be MomentRecombination.")
        if differentiation != "frozen-selection":
            raise ValueError("Only differentiation='frozen-selection' is supported.")
        self.degree = degree_
        self.method = selected
        self.maximum_features = features
        self.maximum_feature_bytes = feature_bytes
        self.maximum_moment_error = error
        self.differentiation = differentiation
        self.recombination_id = canonical_fingerprint(
            {
                "kind": "polynomial-recombination-v1",
                "degree": degree_,
                "maximum_features": features,
                "maximum_feature_bytes": feature_bytes,
                "maximum_moment_error": error,
                "rcond": selected.rcond,
                "tree_reduction_factor": selected.tree_reduction_factor,
                "differentiation": differentiation,
            }
        )

    def prepare(self, dimension: int, /) -> TotalDegreePolynomialFeatures:
        return TotalDegreePolynomialFeatures(
            dimension,
            self.degree,
            maximum_features=self.maximum_features,
            maximum_feature_bytes=self.maximum_feature_bytes,
        )


class MarkovCubaturePlan(StrictModule):
    """Static weak-law propagation and positive recombination policy."""

    temporal_mesh: TemporalMesh
    increment_rule: GaussianCubatureRule
    recombination: PolynomialRecombination
    path: WienerCubaturePathData | None
    method: MarkovCubatureMethod = eqx.field(static=True)
    maximum_expanded_particles: int = eqx.field(static=True)
    flow_substeps: int = eqx.field(static=True)
    collect_history: bool = eqx.field(static=True)
    throw: bool = eqx.field(static=True)
    weak_order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        temporal_mesh: TemporalMesh,
        increment_rule: GaussianCubatureRule,
        /,
        *,
        recombination: PolynomialRecombination | None = None,
        method: MarkovCubatureMethod = "weak-euler",
        path: WienerCubaturePathData | None = None,
        maximum_expanded_particles: int = 1_000_000,
        flow_substeps: int = 1,
        collect_history: bool = True,
        throw: bool = True,
    ):
        if not isinstance(temporal_mesh, TemporalMesh):
            raise TypeError("temporal_mesh must be a TemporalMesh.")
        if not isinstance(increment_rule, GaussianCubatureRule):
            raise TypeError("increment_rule must be a GaussianCubatureRule.")
        selected_recombination = (
            PolynomialRecombination() if recombination is None else recombination
        )
        if not isinstance(selected_recombination, PolynomialRecombination):
            raise TypeError("recombination must be PolynomialRecombination.")
        if method not in ("weak-euler", "stratonovich-flow"):
            raise ValueError("method must be 'weak-euler' or 'stratonovich-flow'.")
        expanded = _positive_integer(
            maximum_expanded_particles, "maximum_expanded_particles"
        )
        substeps = _positive_integer(flow_substeps, "flow_substeps")
        selected_path = path
        if method == "stratonovich-flow":
            selected_path = (
                straight_wiener_cubature_path(increment_rule.prepared)
                if path is None
                else path
            )
            if not isinstance(selected_path, WienerCubaturePathData):
                raise TypeError("path must be WienerCubaturePathData.")
            if selected_path.source_rule_id != increment_rule.rule_id:
                raise ValueError("Cubature path and increment rule identities differ.")
            if selected_path.noise_dimension != increment_rule.dimension:
                raise ValueError("Cubature path and increment rule dimensions differ.")
            if selected_path.signature_degree < 3:
                raise ValueError(
                    "stratonovich-flow requires signature degree at least three."
                )
        elif path is not None:
            raise ValueError("path is valid only for stratonovich-flow.")
        self.temporal_mesh = temporal_mesh
        self.increment_rule = increment_rule
        self.recombination = selected_recombination
        self.path = selected_path
        self.method = method
        self.maximum_expanded_particles = expanded
        self.flow_substeps = substeps
        self.collect_history = bool(collect_history)
        self.throw = bool(throw)
        self.weak_order = 1
        self.plan_id = canonical_fingerprint(
            {
                "kind": "markov-cubature-plan-v1",
                "temporal_mesh": temporal_mesh.mesh_id,
                "increment_rule": increment_rule.rule_id,
                "recombination": selected_recombination.recombination_id,
                "method": method,
                "path": None if selected_path is None else selected_path.path_id,
                "maximum_expanded_particles": expanded,
                "flow_substeps": substeps,
                "collect_history": bool(collect_history),
                "weak_order": 1,
            }
        )


class MarkovCubatureDiagnostics(StrictModule):
    expanded_points: Array
    retained_points: Array
    numerical_rank: Array
    mass_error: Array
    maximum_moment_error: Array
    minimum_weight: Array
    statuses: Array
    feature_count: int = eqx.field(static=True)
    retained_capacity: int = eqx.field(static=True)
    expanded_capacity: int = eqx.field(static=True)
    method: str = eqx.field(static=True)
    weak_order: int = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)
    recombination_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class MarkovCubatureSolution(StrictModule):
    times: Array
    points: Array
    log_weights: Array
    mask: Array
    valid: Array
    status: Array
    diagnostics: MarkovCubatureDiagnostics
    state_shape: tuple[int, ...] = eqx.field(static=True)
    collect_history: bool = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(MarkovCubatureStatus.SUCCESS)

    def measure(self, index: int = -1, /):
        """Return one deterministic weighted state law as an integration target."""
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("measure index must be an integer.")
        selected = int(index)
        if not -int(self.times.size) <= selected < int(self.times.size):
            raise IndexError("measure index is outside the saved Markov cubature times.")
        from ..integration import weighted

        return weighted(
            self.points[selected],
            self.log_weights[selected],
            mask=self.mask[selected],
            normalized=True,
            independent=False,
            sample_axes=0,
            provenance=f"markov-cubature:{self.solver_id}:{selected}",
        )


def solve_markov_cubature(
    problem: DifferentialProblem,
    plan: MarkovCubaturePlan,
    /,
) -> MarkovCubatureSolution:
    """Propagate a weighted finite law with deterministic Gaussian branches."""
    if not isinstance(problem, DifferentialProblem):
        raise TypeError("problem must be a DifferentialProblem.")
    if not isinstance(plan, MarkovCubaturePlan):
        raise TypeError("plan must be a MarkovCubaturePlan.")
    if not problem.stochastic:
        raise ValueError("Markov cubature requires a stochastic DifferentialProblem.")
    if jnp.iscomplexobj(problem.initial_state):
        raise TypeError("Markov cubature currently requires a real state.")
    if not jnp.issubdtype(problem.initial_state.dtype, jnp.floating):
        raise TypeError("Markov cubature requires a floating-point state.")
    if problem.state_geometry is not None and not problem.state_geometry.trivial:
        raise NotImplementedError(
            "Markov cubature currently requires trivial Euclidean state geometry."
        )
    if any(term.representation != "dense" for term in problem.wiener_terms):
        raise NotImplementedError(
            "Markov cubature currently requires dense Wiener coefficients."
        )
    if problem.noise_shape != (plan.increment_rule.dimension,):
        raise ValueError(
            "Gaussian cubature dimension must match the problem's flattened noise size."
        )
    if plan.increment_rule.exact_degree < 3:
        raise ValueError("Markov cubature requires Gaussian exact degree at least three.")
    if plan.method == "weak-euler":
        if problem.interpretation == "stratonovich" and not problem.additive_noise:
            raise NotImplementedError(
                "Weak Euler supports Stratonovich problems only for additive noise."
            )
    else:
        if problem.interpretation != "stratonovich":
            raise ValueError("stratonovich-flow requires a Stratonovich problem.")
        if any(
            term.structure not in ("additive", "commutative")
            for term in problem.wiener_terms
        ):
            raise NotImplementedError(
                "Straight cubature paths require additive or commutative noise."
            )
    state_shape = tuple(problem.initial_state.shape)
    state_size = int(problem.initial_state.size) if state_shape else 1
    features = plan.recombination.prepare(state_size)
    retained_capacity = features.feature_count + 1
    path_count = (
        plan.increment_rule.num_points if plan.path is None else plan.path.path_count
    )
    expanded_capacity = retained_capacity * path_count
    if expanded_capacity > plan.maximum_expanded_particles:
        raise ValueError(
            f"Markov cubature expansion requires {expanded_capacity} particles, "
            f"exceeding maximum_expanded_particles={plan.maximum_expanded_particles}."
        )
    mesh_nodes = eqx.error_if(
        plan.temporal_mesh.nodes,
        ~(
            jnp.isclose(plan.temporal_mesh.t0, problem.t0)
            & jnp.isclose(plan.temporal_mesh.t1, problem.t1)
        ),
        "Markov cubature temporal mesh must match problem bounds.",
    )
    initial_point = jnp.asarray(problem.initial_state).reshape((state_size,))
    points = (
        jnp.zeros((retained_capacity, state_size), dtype=problem.initial_state.dtype)
        .at[0]
        .set(initial_point)
    )
    log_weights = jnp.full((retained_capacity,), -jnp.inf).at[0].set(0.0)
    mask = jnp.zeros((retained_capacity,), dtype=bool).at[0].set(True)
    status = jnp.asarray(int(MarkovCubatureStatus.SUCCESS), dtype=jnp.int32)
    controls = jnp.asarray(plan.increment_rule.prepared.points, dtype=points.dtype)
    control_weights = (
        plan.increment_rule.prepared.weights if plan.path is None else plan.path.weights
    )
    log_control_weights = jnp.log(control_weights)

    def evaluate_parent(time: Array, flat_state: Array, active: Array):
        def evaluate(state_flat):
            state = state_flat.reshape(state_shape)
            drift_value = jnp.asarray(problem.drift(time, state, problem.args))
            if drift_value.shape != state_shape:
                raise ValueError("Differential drift changed its declared state shape.")
            matrices = tuple(
                term.coefficient_matrix(time, state, problem.args)
                for term in problem.wiener_terms
            )
            diffusion = jnp.concatenate(matrices, axis=1).astype(points.dtype)
            return drift_value.astype(points.dtype).reshape((state_size,)), diffusion

        return jax.lax.cond(
            active,
            evaluate,
            lambda _: (
                jnp.zeros((state_size,), dtype=points.dtype),
                jnp.zeros(
                    (state_size, plan.increment_rule.dimension), dtype=points.dtype
                ),
            ),
            flat_state,
        )

    def weak_euler_step(time: Array, width: Array, current: Array, active: Array):
        drift, diffusion = jax.vmap(
            lambda state, enabled: evaluate_parent(time, state, enabled)
        )(current, active)
        state_width = jnp.asarray(width, dtype=current.dtype)
        noise = oe.contract("rsd,pd->rps", diffusion, controls)
        return (
            current[:, None, :]
            + state_width * drift[:, None, :]
            + jnp.sqrt(state_width) * noise
        )

    def flow_step(time: Array, width: Array, current: Array, active: Array):
        if plan.path is None:
            raise RuntimeError("stratonovich-flow is missing cubature path data.")
        path_increments = jnp.asarray(plan.path.increments, dtype=points.dtype)
        segment_widths = plan.path.segment_widths
        segment_starts = jnp.cumsum(segment_widths) - segment_widths
        substep = 1.0 / float(plan.flow_substeps)

        def one_path(state: Array, increments: Array):
            def segment_body(segment_index, segment_state):
                increment = increments[segment_index]
                segment_width = segment_widths[segment_index]
                segment_start = segment_starts[segment_index]

                def rhs(fraction: Array, value: Array):
                    drift, diffusion = evaluate_parent(
                        time + width * (segment_start + fraction * segment_width),
                        value,
                        jnp.asarray(True),
                    )
                    state_width = jnp.asarray(width, dtype=value.dtype)
                    state_segment_width = jnp.asarray(segment_width, dtype=value.dtype)
                    return state_width * state_segment_width * drift + jnp.sqrt(
                        state_width
                    ) * (diffusion @ increment)

                def substep_body(index, value):
                    start = jnp.asarray(index, dtype=width.dtype) * substep
                    time_step = jnp.asarray(substep, dtype=width.dtype)
                    state_step = jnp.asarray(substep, dtype=value.dtype)
                    k1 = rhs(start, value)
                    k2 = rhs(
                        start + 0.5 * time_step,
                        value + 0.5 * state_step * k1,
                    )
                    k3 = rhs(
                        start + 0.5 * time_step,
                        value + 0.5 * state_step * k2,
                    )
                    k4 = rhs(start + time_step, value + state_step * k3)
                    return value + state_step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

                return jax.lax.fori_loop(
                    0, plan.flow_substeps, substep_body, segment_state
                )

            return jax.lax.fori_loop(0, plan.path.segment_count, segment_body, state)

        def one_parent(state: Array, enabled: Array):
            return jax.lax.cond(
                enabled,
                lambda value: jax.vmap(lambda increments: one_path(value, increments))(
                    path_increments
                ),
                lambda value: jnp.broadcast_to(value, (path_count, state_size)),
                state,
            )

        return jax.vmap(one_parent)(current, active)

    def one_interval(carry, inputs):
        current_points, current_log_weights, current_mask, current_status = carry
        start, end, interval_active = inputs
        should_advance = interval_active & (
            current_status == int(MarkovCubatureStatus.SUCCESS)
        )

        def advance(values):
            source_points, source_log_weights, source_mask, _ = values
            width = end - start
            propagated = (
                weak_euler_step(start, width, source_points, source_mask)
                if plan.method == "weak-euler"
                else flow_step(start, width, source_points, source_mask)
            )
            expanded_points = propagated.reshape((expanded_capacity, state_size))
            expanded_mask = jnp.repeat(source_mask, path_count)
            expanded_log_weights = (
                source_log_weights[:, None] + log_control_weights[None, :]
            ).reshape((expanded_capacity,))
            finite_states = jnp.all(jnp.isfinite(expanded_points), axis=1)
            expanded_mask = expanded_mask & finite_states
            weights, _, weights_valid, _ = normalized_weights(
                expanded_capacity,
                log_weights=expanded_log_weights,
                mask=expanded_mask,
                rows_valid=finite_states,
            )
            safe_expanded_points = jnp.where(finite_states[:, None], expanded_points, 0.0)
            feature_values, _, _ = features.evaluate(safe_expanded_points, weights)
            selection = moment_recombine(
                jax.lax.stop_gradient(feature_values),
                plan.recombination.method,
                log_weights=jax.lax.stop_gradient(expanded_log_weights),
                mask=expanded_mask,
            )
            selected_indices = jax.lax.stop_gradient(selection.indices)
            selected_mask = jax.lax.stop_gradient(selection.mask)
            selected_points = safe_expanded_points[selected_indices]
            augmented_features = jnp.concatenate(
                (
                    jnp.ones((expanded_capacity, 1), dtype=feature_values.dtype),
                    feature_values,
                ),
                axis=1,
            )
            selected_features = augmented_features[selected_indices]
            active_outer = selected_mask[:, None] & selected_mask[None, :]
            gram = selected_features @ selected_features.T
            masked_gram = jnp.where(active_outer, gram, 0.0) + jnp.diag(
                (~selected_mask).astype(gram.dtype)
            )
            target_moments = weights @ augmented_features
            surrogate_weights = jnp.linalg.solve(
                masked_gram,
                jnp.where(
                    selected_mask,
                    selected_features @ target_moments,
                    0.0,
                ),
            )
            primal_weights = jnp.where(selected_mask, jnp.exp(selection.log_weights), 0.0)
            selected_weights = (
                jax.lax.stop_gradient(primal_weights - surrogate_weights)
                + surrogate_weights
            )
            safe_selected_weights = jnp.where(selected_mask, selected_weights, 1.0)
            selected_log_weights = jnp.where(
                selected_mask, jnp.log(safe_selected_weights), -jnp.inf
            )
            diagnostics = jax.tree_util.tree_map(
                jax.lax.stop_gradient, selection.diagnostics
            )
            finite_dynamics = jnp.all(
                ~jnp.repeat(source_mask, path_count) | finite_states
            )
            moment_ok = (
                diagnostics.max_moment_error <= plan.recombination.maximum_moment_error
            )
            next_status = jnp.where(
                ~finite_dynamics,
                int(MarkovCubatureStatus.NONFINITE_DYNAMICS),
                jnp.where(
                    ~weights_valid,
                    int(MarkovCubatureStatus.INVALID_WEIGHTS),
                    jnp.where(
                        ~diagnostics.valid,
                        int(MarkovCubatureStatus.RECOMBINATION_FAILED),
                        jnp.where(
                            ~moment_ok,
                            int(MarkovCubatureStatus.MOMENT_RESIDUAL_TOO_LARGE),
                            int(MarkovCubatureStatus.SUCCESS),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            step_output = (
                jnp.sum(expanded_mask, dtype=jnp.int32),
                diagnostics.active_points,
                diagnostics.numerical_rank,
                diagnostics.mass_error,
                diagnostics.max_moment_error,
                diagnostics.minimum_weight,
                next_status,
                selected_points,
                selected_log_weights,
                selected_mask,
            )
            return (
                selected_points,
                selected_log_weights,
                selected_mask,
                next_status,
            ), step_output

        def hold(values):
            source_points, source_log_weights, source_mask, source_status = values
            active_points = jnp.sum(source_mask, dtype=jnp.int32)
            zero = jnp.asarray(0.0, dtype=source_log_weights.dtype)
            zero_count = jnp.asarray(0, dtype=jnp.int32)
            step_output = (
                zero_count,
                active_points,
                zero_count,
                zero,
                zero,
                jnp.min(
                    jnp.where(source_mask, jnp.exp(source_log_weights), jnp.inf),
                    initial=jnp.inf,
                ),
                source_status,
                source_points,
                source_log_weights,
                source_mask,
            )
            return values, step_output

        return jax.lax.cond(should_advance, advance, hold, carry)

    inputs = (
        mesh_nodes[:-1],
        mesh_nodes[1:],
        plan.temporal_mesh.active_intervals,
    )
    initial = (points, log_weights, mask, status)
    terminal, outputs = jax.lax.scan(one_interval, initial, inputs)
    terminal_points, terminal_log_weights, terminal_mask, terminal_status = terminal
    (
        expanded_counts,
        retained_counts,
        numerical_ranks,
        mass_errors,
        moment_errors,
        minimum_weights,
        statuses,
        history_points,
        history_log_weights,
        history_mask,
    ) = outputs
    if plan.collect_history:
        saved_points = jnp.concatenate((points[None, ...], history_points), axis=0)
        saved_log_weights = jnp.concatenate(
            (log_weights[None, ...], history_log_weights), axis=0
        )
        saved_mask = jnp.concatenate((mask[None, ...], history_mask), axis=0)
        saved_times = mesh_nodes
        saved_valid = jnp.concatenate(
            (
                jnp.asarray([True]),
                statuses == int(MarkovCubatureStatus.SUCCESS),
            )
        )
    else:
        saved_points = jnp.stack((points, terminal_points))
        saved_log_weights = jnp.stack((log_weights, terminal_log_weights))
        saved_mask = jnp.stack((mask, terminal_mask))
        saved_times = jnp.stack((mesh_nodes[0], mesh_nodes[-1]))
        saved_valid = jnp.asarray(
            [True, terminal_status == int(MarkovCubatureStatus.SUCCESS)]
        )
    if plan.throw:
        terminal_points = eqx.error_if(
            terminal_points,
            terminal_status != int(MarkovCubatureStatus.SUCCESS),
            "Markov cubature propagation failed.",
        )
        saved_points = saved_points.at[-1].set(terminal_points)
    diagnostics = MarkovCubatureDiagnostics(
        expanded_points=expanded_counts,
        retained_points=retained_counts,
        numerical_rank=numerical_ranks,
        mass_error=mass_errors,
        maximum_moment_error=moment_errors,
        minimum_weight=minimum_weights,
        statuses=statuses,
        feature_count=features.feature_count,
        retained_capacity=retained_capacity,
        expanded_capacity=expanded_capacity,
        method=plan.method,
        weak_order=plan.weak_order,
        rule_id=plan.increment_rule.rule_id,
        feature_id=features.feature_id,
        recombination_id=plan.recombination.recombination_id,
        plan_id=plan.plan_id,
    )
    solver_id = canonical_fingerprint(
        {
            "kind": "markov-cubature-solver-v1",
            "plan": plan.plan_id,
            "problem_noise": problem.noise_id,
            "state_shape": state_shape,
        }
    )
    return MarkovCubatureSolution(
        times=saved_times,
        points=saved_points.reshape(
            (saved_points.shape[0], retained_capacity) + state_shape
        ),
        log_weights=saved_log_weights,
        mask=saved_mask,
        valid=saved_valid,
        status=terminal_status,
        diagnostics=diagnostics,
        state_shape=state_shape,
        collect_history=plan.collect_history,
        solver_id=solver_id,
    )


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


__all__ = [
    "MarkovCubatureDiagnostics",
    "MarkovCubatureMethod",
    "MarkovCubaturePlan",
    "MarkovCubatureSolution",
    "MarkovCubatureStatus",
    "PolynomialRecombination",
    "solve_markov_cubature",
]
