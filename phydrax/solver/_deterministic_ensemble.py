#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    ArraySpace,
    ComplexCartesianCoordinates,
    prepare_real_coordinate_tree,
    PreparedRealCoordinateTree,
)
from ._differential import DifferentialProblem, DifferentialSolution
from ._diffrax_backend import solve_diffrax


class DeterministicEnsembleStatus(IntEnum):
    """Fail-closed status for a deterministic initial-condition path."""

    SUCCESS = 0
    SOLVER_FAILURE = 1
    NONFINITE_TRAJECTORY = 2


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _state_signature(state: Any, /) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple(
        (tuple(jnp.asarray(leaf).shape), str(jnp.asarray(leaf).dtype))
        for leaf in jax.tree.leaves(state)
    )


def _state_size_bytes(state: Any, /) -> tuple[int, int]:
    leaves = tuple(jnp.asarray(leaf) for leaf in jax.tree.leaves(state))
    return (
        sum(int(leaf.size) for leaf in leaves),
        sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in leaves),
    )


def _ensemble_state_coordinates(state: Any, /) -> PreparedRealCoordinateTree | None:
    if eqx.is_array_like(state):
        return None
    treedef = jax.tree.structure(state)
    leaves = tuple(jnp.asarray(leaf) for leaf in jax.tree.leaves(state))
    maps = tuple(
        ComplexCartesianCoordinates(ArraySpace(leaf.shape, dtype=leaf.dtype))
        if jnp.issubdtype(leaf.dtype, jnp.complexfloating)
        else None
        for leaf in leaves
    )
    return prepare_real_coordinate_tree(
        state,
        jax.tree.unflatten(treedef, maps),
    )


class DeterministicInitialCondition(StrictModule):
    """One explicit, weighted initial state with separate semantic/numeric identity."""

    initial_state: Any
    weight: Array
    semantic_path_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_state: Any,
        weight: ArrayLike,
        semantic_label: str,
        /,
    ):
        state = (
            jnp.asarray(initial_state)
            if eqx.is_array_like(initial_state) or isinstance(initial_state, list)
            else jax.tree.map(jnp.asarray, initial_state)
        )
        if not jax.tree.leaves(state):
            raise ValueError("A deterministic initial condition must contain arrays.")
        if any(
            not np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(state)
        ):
            raise ValueError("Deterministic initial states must be finite.")
        path_weight = np.asarray(weight)
        if (
            path_weight.shape != ()
            or not np.isrealobj(path_weight)
            or not np.isfinite(path_weight)
            or float(path_weight) < 0.0
        ):
            raise ValueError("Deterministic path weight must be finite and non-negative.")
        label = _identifier(semantic_label, "semantic_label")
        semantic_path_id = canonical_fingerprint(
            {"kind": "deterministic-initial-condition", "semantic_label": label}
        )
        self.initial_state = state
        self.weight = jnp.asarray(path_weight)
        self.semantic_path_id = semantic_path_id
        self.realization_id = canonical_fingerprint(
            {
                "kind": "deterministic-initial-condition-realization",
                "semantic_path_id": semantic_path_id,
                "state": array_tree_fingerprint(state),
                "weight": array_tree_fingerprint(path_weight),
            }
        )


class WeightedEnsembleReducer(StrictModule):
    """Normalized fixed-path reductions that make invalid-path masking explicit."""

    weights: Array
    valid: Array
    normalized_weights: Array
    effective_sample_size: Array
    successful: Array

    def __init__(self, weights: ArrayLike, valid: ArrayLike | None = None, /):
        raw = jnp.asarray(weights)
        if raw.ndim != 1 or jnp.issubdtype(raw.dtype, jnp.complexfloating):
            raise ValueError("weights must be one real rank-one array.")
        if valid is None:
            mask = jnp.ones(raw.shape, dtype=bool)
        else:
            mask = jnp.asarray(valid, dtype=bool)
            if mask.shape != raw.shape:
                raise ValueError("valid must have the same shape as weights.")
        finite_nonnegative = jnp.all(jnp.isfinite(raw) & (raw >= 0.0))
        active = jnp.where(mask & jnp.isfinite(raw) & (raw >= 0.0), raw, 0.0)
        total = jnp.sum(active)
        successful = finite_nonnegative & jnp.isfinite(total) & (total > 0.0)
        denominator = jnp.where(successful, total, 1.0)
        normalized = active / denominator
        square_sum = jnp.sum(normalized * normalized)
        ess = jnp.where(successful, 1.0 / square_sum, 0.0)
        self.weights = raw
        self.valid = mask
        self.normalized_weights = normalized
        self.effective_sample_size = ess
        self.successful = successful

    def mean(self, values: Any, /) -> Any:
        """Reduce the leading path axis of every array leaf."""

        def reduce_leaf(leaf: ArrayLike) -> Array:
            array = jnp.asarray(leaf)
            if array.ndim < 1 or array.shape[0] != self.weights.size:
                raise ValueError("Every reduced value must begin with the path axis.")
            shape = (self.weights.size,) + (1,) * (array.ndim - 1)
            coefficient = self.normalized_weights.reshape(shape)
            masked = jnp.where(self.valid.reshape(shape), array, 0)
            reduced = jnp.sum(coefficient * masked, axis=0)
            return eqx.error_if(
                reduced,
                ~self.successful,
                "Weighted ensemble has no positive valid path mass.",
            )

        return jax.tree.map(reduce_leaf, values)

    def raw_moment(self, values: ArrayLike, order: int, /) -> Array:
        degree = int(order)
        if degree < 0:
            raise ValueError("Moment order must be non-negative.")
        return self.mean(jnp.asarray(values) ** degree)

    def central_moment(self, values: ArrayLike, order: int, /) -> Array:
        degree = int(order)
        if degree < 1:
            raise ValueError("Central moment order must be positive.")
        array = jnp.asarray(values)
        center = self.mean(array)
        return self.mean((array - center) ** degree)

    def variance(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        centered = array - self.mean(array)
        return self.mean(jnp.real(centered * jnp.conj(centered)))

    def covariance(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        x = jnp.asarray(left)
        y = jnp.asarray(right)
        if x.shape != y.shape:
            raise ValueError("Covariance operands must have identical shapes.")
        return self.mean((x - self.mean(x)) * jnp.conj(y - self.mean(y)))


class DeterministicEnsemblePlan(StrictModule, NonTrainableState):
    """Immutable weighted paths sharing one differential equation and time support."""

    problem: DifferentialProblem
    initial_conditions: tuple[DeterministicInitialCondition, ...]
    weights: Array
    state_coordinates: PreparedRealCoordinateTree | None
    maximum_paths: int = eqx.field(static=True)
    maximum_state_elements: int = eqx.field(static=True)
    maximum_output_bytes: int = eqx.field(static=True)
    state_elements: int = eqx.field(static=True)
    state_bytes: int = eqx.field(static=True)
    path_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: DifferentialProblem,
        initial_conditions: Sequence[DeterministicInitialCondition],
        /,
        *,
        maximum_paths: int = 4096,
        maximum_state_elements: int = 2**24,
        maximum_output_bytes: int = 2**31,
    ):
        if not isinstance(problem, DifferentialProblem):
            raise TypeError("problem must be a DifferentialProblem.")
        if problem.stochastic:
            raise ValueError("Deterministic ensemble execution rejects Wiener terms.")
        paths = tuple(initial_conditions)
        if not paths or not all(
            isinstance(path, DeterministicInitialCondition) for path in paths
        ):
            raise ValueError(
                "initial_conditions must contain DeterministicInitialCondition values."
            )
        path_limit = int(maximum_paths)
        state_limit = int(maximum_state_elements)
        byte_limit = int(maximum_output_bytes)
        if path_limit < 1 or state_limit < 1 or byte_limit < 1:
            raise ValueError("Deterministic ensemble resource limits must be positive.")
        if len(paths) > path_limit:
            raise MemoryError("Deterministic ensemble exceeds maximum_paths.")
        template_structure = jax.tree.structure(problem.initial_state)
        template_signature = _state_signature(problem.initial_state)
        for path in paths:
            if (
                jax.tree.structure(path.initial_state) != template_structure
                or _state_signature(path.initial_state) != template_signature
            ):
                raise ValueError(
                    "Every deterministic initial state must match the problem state tree."
                )
        path_ids = tuple(path.semantic_path_id for path in paths)
        if len(set(path_ids)) != len(path_ids):
            raise ValueError("Deterministic semantic path IDs must be unique.")
        weights = np.asarray([float(path.weight) for path in paths])
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or weights.sum() <= 0.0
        ):
            raise ValueError(
                "Deterministic path weights require positive finite total mass."
            )
        state_elements, state_bytes = _state_size_bytes(problem.initial_state)
        if len(paths) * state_elements > state_limit:
            raise MemoryError(
                "Deterministic initial states exceed maximum_state_elements."
            )
        state_coordinates = _ensemble_state_coordinates(problem.initial_state)
        self.problem = problem
        self.initial_conditions = paths
        self.weights = jnp.asarray(weights)
        self.state_coordinates = state_coordinates
        self.maximum_paths = path_limit
        self.maximum_state_elements = state_limit
        self.maximum_output_bytes = byte_limit
        self.state_elements = state_elements
        self.state_bytes = state_bytes
        self.path_ids = path_ids
        self.plan_id = canonical_fingerprint(
            {
                "kind": "deterministic-initial-condition-ensemble",
                "problem": problem.problem_id,
                "paths": path_ids,
                "realizations": [path.realization_id for path in paths],
                "state_coordinates": (
                    None if state_coordinates is None else state_coordinates.coordinate_id
                ),
                "maximum_paths": path_limit,
                "maximum_state_elements": state_limit,
                "maximum_output_bytes": byte_limit,
            }
        )

    def prepare(self, save_times: ArrayLike, /) -> PreparedDeterministicEnsemble:
        return PreparedDeterministicEnsemble(self, save_times)


class PreparedDeterministicEnsemble(StrictModule, NonTrainableState):
    """Admitted fixed-shape ensemble storage and path-specific problems."""

    plan: DeterministicEnsemblePlan
    save_times: Array
    problems: tuple[DifferentialProblem, ...]
    estimated_output_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: DeterministicEnsemblePlan, save_times: ArrayLike, /):
        if not isinstance(plan, DeterministicEnsemblePlan):
            raise TypeError("plan must be a DeterministicEnsemblePlan.")
        times = np.asarray(save_times, dtype=float)
        if (
            times.ndim != 1
            or times.size < 1
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0.0)
            or times[0] < float(plan.problem.t0)
            or times[-1] > float(plan.problem.t1)
        ):
            raise ValueError(
                "save_times must be finite, increasing, and within the problem support."
            )
        estimate = (
            len(plan.initial_conditions)
            * times.size
            * (plan.state_bytes + np.dtype(float).itemsize + np.dtype(bool).itemsize)
        )
        if estimate > plan.maximum_output_bytes:
            raise MemoryError("Deterministic trajectories exceed maximum_output_bytes.")
        problems = tuple(
            DifferentialProblem(
                plan.problem.drift,
                path.initial_state,
                t0=plan.problem.t0,
                t1=plan.problem.t1,
                args=plan.problem.args,
                interpretation=plan.problem.interpretation,
                state_geometry=plan.problem.state_geometry,
                discretization_bundle=plan.problem.discretization_bundle,
                problem_id=canonical_fingerprint(
                    {
                        "kind": "deterministic-ensemble-path-problem",
                        "ensemble": plan.plan_id,
                        "path": path.semantic_path_id,
                        "realization": path.realization_id,
                    }
                ),
            )
            for path in plan.initial_conditions
        )
        self.plan = plan
        self.save_times = jnp.asarray(times)
        self.problems = problems
        self.estimated_output_bytes = estimate
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-deterministic-ensemble",
                "plan": plan.plan_id,
                "save_times": array_tree_fingerprint(times),
                "estimated_output_bytes": estimate,
            }
        )


class DeterministicEnsembleEvidence(StrictModule):
    """Per-path backend, finiteness, and reduction evidence."""

    status: Array
    backend_successful: Array
    finite_trajectory: Array
    positive_weight_mass: Array
    effective_sample_size: Array
    successful: Array


class DeterministicEnsembleResult(StrictModule):
    """Executed trajectories with semantic path identity and weighted reducers."""

    times: Array
    states: Any
    valid: Array
    solutions: tuple[DifferentialSolution, ...]
    reducer: WeightedEnsembleReducer
    evidence: DeterministicEnsembleEvidence
    path_ids: tuple[str, ...] = eqx.field(static=True)
    realization_ids: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def mean(self, /) -> Any:
        return self.reducer.mean(self.states)


def execute_deterministic_ensemble(
    prepared: PreparedDeterministicEnsemble,
    /,
    *,
    solver: Any | None = None,
    stepsize_controller: Any | None = None,
    adjoint: Any | None = None,
    dt0: ArrayLike | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int | None = 4096,
) -> DeterministicEnsembleResult:
    """Execute every explicit path through the existing Diffrax backend."""

    if not isinstance(prepared, PreparedDeterministicEnsemble):
        raise TypeError("prepared must be a PreparedDeterministicEnsemble.")
    solutions = tuple(
        solve_diffrax(
            problem,
            save_times=prepared.save_times,
            solver=solver,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
            dt0=dt0,
            state_coordinates=prepared.plan.state_coordinates,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            throw=False,
            solver_configuration_id=canonical_fingerprint(
                {
                    "kind": "deterministic-ensemble-runtime",
                    "prepared": prepared.prepared_id,
                    "path": prepared.plan.path_ids[index],
                }
            ),
        )
        for index, problem in enumerate(prepared.problems)
    )
    states = jax.tree.map(
        lambda *leaves: jnp.stack(leaves, axis=0),
        *(solution.states for solution in solutions),
    )
    valid = jnp.stack(tuple(solution.valid for solution in solutions), axis=0)
    backend_successful = jnp.stack(
        tuple(jnp.asarray(solution.backend_successful) for solution in solutions)
    )
    finite_trajectory = jnp.all(valid, axis=-1)
    status = jnp.where(
        ~backend_successful,
        int(DeterministicEnsembleStatus.SOLVER_FAILURE),
        jnp.where(
            ~finite_trajectory,
            int(DeterministicEnsembleStatus.NONFINITE_TRAJECTORY),
            int(DeterministicEnsembleStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    path_valid = status == int(DeterministicEnsembleStatus.SUCCESS)
    reducer = WeightedEnsembleReducer(prepared.plan.weights, path_valid)
    successful = jnp.all(path_valid) & reducer.successful
    evidence = DeterministicEnsembleEvidence(
        status,
        backend_successful,
        finite_trajectory,
        reducer.successful,
        reducer.effective_sample_size,
        successful,
    )
    realization_ids = tuple(
        path.realization_id for path in prepared.plan.initial_conditions
    )
    result_id = canonical_fingerprint(
        {
            "kind": "deterministic-ensemble-result",
            "prepared": prepared.prepared_id,
            "path_ids": list(prepared.plan.path_ids),
            "realization_ids": list(realization_ids),
            "solver_ids": [solution.solver_id for solution in solutions],
        }
    )
    return DeterministicEnsembleResult(
        prepared.save_times,
        states,
        valid,
        solutions,
        reducer,
        evidence,
        prepared.plan.path_ids,
        realization_ids,
        prepared.prepared_id,
        result_id,
    )


class ClassicalStatisticalScalarRecipe(StrictModule, NonTrainableState):
    """Explicit Gaussian scalar phase-space nodes with vacuum/occupation scaling."""

    frequencies: Array
    occupations: Array
    coordinates: Array
    weights: Array
    mean_field: Array
    mean_momentum: Array
    mode_count: int = eqx.field(static=True)
    path_count: int = eqx.field(static=True)
    semantic_id: str = eqx.field(static=True)
    recipe_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequencies: ArrayLike,
        coordinates: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        occupations: ArrayLike = 0.0,
        mean_field: ArrayLike = 0.0,
        mean_momentum: ArrayLike = 0.0,
        maximum_paths: int = 4096,
        maximum_modes: int = 2**18,
    ):
        omega = np.asarray(frequencies, dtype=float)
        nodes = np.asarray(coordinates, dtype=float)
        path_weights = np.asarray(weights, dtype=float)
        if (
            omega.ndim != 1
            or omega.size < 1
            or not np.all(np.isfinite(omega) & (omega > 0.0))
        ):
            raise ValueError("frequencies must be one non-empty positive finite vector.")
        if nodes.ndim != 3 or nodes.shape[1:] != (2, omega.size):
            raise ValueError("coordinates must have shape (paths, 2, modes).")
        if not np.isfinite(nodes).all():
            raise ValueError("Scalar phase-space coordinates must be finite.")
        if (
            path_weights.shape != (nodes.shape[0],)
            or not np.all(np.isfinite(path_weights) & (path_weights >= 0.0))
            or path_weights.sum() <= 0.0
        ):
            raise ValueError("weights must provide non-negative finite path mass.")
        if nodes.shape[0] > int(maximum_paths) or omega.size > int(maximum_modes):
            raise MemoryError("Classical-statistical scalar recipe exceeds resources.")
        occupation = np.broadcast_to(np.asarray(occupations, dtype=float), omega.shape)
        field_mean = np.broadcast_to(np.asarray(mean_field, dtype=float), omega.shape)
        momentum_mean = np.broadcast_to(
            np.asarray(mean_momentum, dtype=float), omega.shape
        )
        if (
            not np.all(np.isfinite(occupation) & (occupation >= 0.0))
            or not np.isfinite(field_mean).all()
            or not np.isfinite(momentum_mean).all()
        ):
            raise ValueError(
                "Scalar occupation and mean data must be finite and physical."
            )
        self.frequencies = jnp.asarray(omega)
        self.occupations = jnp.asarray(occupation)
        self.coordinates = jnp.asarray(nodes)
        self.weights = jnp.asarray(path_weights)
        self.mean_field = jnp.asarray(field_mean)
        self.mean_momentum = jnp.asarray(momentum_mean)
        self.mode_count = omega.size
        self.path_count = nodes.shape[0]
        semantic_id = canonical_fingerprint(
            {
                "kind": "classical-statistical-scalar-path-family",
                "mode_count": omega.size,
                "path_count": nodes.shape[0],
                "state_layout": ["field", "momentum"],
            }
        )
        self.semantic_id = semantic_id
        self.recipe_id = canonical_fingerprint(
            {
                "kind": "classical-statistical-scalar-recipe",
                "semantic_id": semantic_id,
                "frequencies": array_tree_fingerprint(omega),
                "occupations": array_tree_fingerprint(occupation),
                "coordinates": array_tree_fingerprint(nodes),
                "weights": array_tree_fingerprint(path_weights),
            }
        )

    def initial_conditions(self, /) -> tuple[DeterministicInitialCondition, ...]:
        field_scale = jnp.sqrt((self.occupations + 0.5) / self.frequencies)
        momentum_scale = jnp.sqrt((self.occupations + 0.5) * self.frequencies)
        fields = self.mean_field + self.coordinates[:, 0, :] * field_scale
        momenta = self.mean_momentum + self.coordinates[:, 1, :] * momentum_scale
        states = jnp.stack((fields, momenta), axis=1)
        return tuple(
            DeterministicInitialCondition(
                states[index],
                self.weights[index],
                f"{self.semantic_id}:scalar-path:{index}",
            )
            for index in range(self.path_count)
        )


class MMSTInitialConditionRecipe(StrictModule, NonTrainableState):
    """Focused Meyer-Miller-Stock-Thoss mapping paths for electronic populations."""

    populations: Array
    nuclear_positions: Array
    nuclear_momenta: Array
    zero_point_parameter: float = eqx.field(static=True)
    state_count: int = eqx.field(static=True)
    semantic_id: str = eqx.field(static=True)
    recipe_id: str = eqx.field(static=True)

    def __init__(
        self,
        populations: ArrayLike,
        nuclear_positions: ArrayLike,
        nuclear_momenta: ArrayLike,
        /,
        *,
        zero_point_parameter: float = 1.0,
        maximum_states: int = 4096,
    ):
        probability = np.asarray(populations, dtype=float)
        positions = np.asarray(nuclear_positions, dtype=float)
        momenta = np.asarray(nuclear_momenta, dtype=float)
        gamma = float(zero_point_parameter)
        if (
            probability.ndim != 1
            or probability.size < 1
            or probability.size > int(maximum_states)
            or not np.all(np.isfinite(probability) & (probability >= 0.0))
            or probability.sum() <= 0.0
        ):
            raise ValueError("populations must be a finite non-negative state vector.")
        if (
            positions.shape != momenta.shape
            or not np.isfinite(positions).all()
            or not np.isfinite(momenta).all()
        ):
            raise ValueError(
                "Nuclear positions and momenta must be matching finite arrays."
            )
        if not np.isfinite(gamma) or gamma < 0.0:
            raise ValueError("zero_point_parameter must be finite and non-negative.")
        normalized = probability / probability.sum()
        self.populations = jnp.asarray(normalized)
        self.nuclear_positions = jnp.asarray(positions)
        self.nuclear_momenta = jnp.asarray(momenta)
        self.zero_point_parameter = gamma
        self.state_count = probability.size
        semantic_id = canonical_fingerprint(
            {
                "kind": "focused-mmst-path-family",
                "electronic_state_count": probability.size,
                "nuclear_shape": list(positions.shape),
                "mapping_layout": ["nuclear-q", "nuclear-p", "mapping-x", "mapping-p"],
            }
        )
        self.semantic_id = semantic_id
        self.recipe_id = canonical_fingerprint(
            {
                "kind": "focused-mmst-initial-condition-recipe",
                "semantic_id": semantic_id,
                "populations": array_tree_fingerprint(normalized),
                "nuclear_positions": array_tree_fingerprint(positions),
                "nuclear_momenta": array_tree_fingerprint(momenta),
                "zero_point_parameter": gamma,
            }
        )

    def initial_conditions(self, /) -> tuple[DeterministicInitialCondition, ...]:
        base = jnp.sqrt(jnp.asarray(self.zero_point_parameter))
        focused = jnp.sqrt(jnp.asarray(2.0 + self.zero_point_parameter))
        mapping_positions = jnp.full((self.state_count, self.state_count), base)
        mapping_positions = mapping_positions.at[
            jnp.arange(self.state_count), jnp.arange(self.state_count)
        ].set(focused)
        mapping_momenta = jnp.zeros_like(mapping_positions)
        return tuple(
            DeterministicInitialCondition(
                (
                    self.nuclear_positions,
                    self.nuclear_momenta,
                    mapping_positions[index],
                    mapping_momenta[index],
                ),
                self.populations[index],
                f"{self.semantic_id}:focused-electronic-state:{index}",
            )
            for index in range(self.state_count)
        )


__all__ = [
    "ClassicalStatisticalScalarRecipe",
    "DeterministicEnsembleEvidence",
    "DeterministicEnsemblePlan",
    "DeterministicEnsembleResult",
    "DeterministicEnsembleStatus",
    "DeterministicInitialCondition",
    "MMSTInitialConditionRecipe",
    "PreparedDeterministicEnsemble",
    "WeightedEnsembleReducer",
    "execute_deterministic_ensemble",
]
