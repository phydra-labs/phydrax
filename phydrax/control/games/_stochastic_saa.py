#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-dimensional stochastic policy games on frozen sample bundles."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from operator import index
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import ArraySpace
from ...nonlinear import (
    NewtonTrustRegion,
    NonlinearDiagnostics,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
    solve_prepared_nonlinear,
)
from ..stochastic._evaluation import PreparedControlledNoise
from ._layout import PlayerControlPartition


_METHOD_ID = "frozen-pathwise-policy-saa-pseudogradient-v1"
_CERTIFICATE = "LOCAL_SAA_POLICY_STATIONARITY"
_UNSET = object()


class StochasticPolicyPathCosts(Protocol):
    """Raw path-cost callback for a case-shaped joint parameter array.

    The signature is ``costs(parameters, prepared_noise, args)``. ``parameters``
    has shape ``case_shape + (parameter_size,)`` and the return must have shape
    ``case_shape + (prepared_noise.num_paths, num_players)``. The callback must
    evaluate each complete supplied noise path independently; it must not replace
    the paths by a mean trajectory. No policy architecture is otherwise assumed.
    """

    def __call__(
        self,
        parameters: Array,
        prepared_noise: PreparedControlledNoise,
        args: Any,
        /,
    ) -> ArrayLike: ...


class StochasticPolicyGameStatus(IntEnum):
    """Stable terminal statuses for a frozen-sample policy-game solve."""

    SUCCESS = 0
    NONFINITE_INITIAL_PARAMETERS = 1
    INVALID_TRAINING_BUNDLE = 2
    NONFINITE_TRAINING_PATH_COSTS = 3
    NONFINITE_TRAINING_PATH_GRADIENTS = 4
    ROOT_FAILED = 5
    INVALID_HOLDOUT_BUNDLE = 6
    NONFINITE_HOLDOUT_PATH_COSTS = 7

    # Singular spellings are convenient aliases, not distinct status codes.
    NONFINITE_TRAINING_PATH_COST = 3
    NONFINITE_TRAINING_PATH_GRADIENT = 4
    ROOT_SOLVE_FAILED = 5
    NONFINITE_HOLDOUT_PATH_COST = 7


class StochasticPolicyGameProblem(StrictModule):
    """A finite-dimensional policy-parameter game evaluated path by path.

    ``partition`` owns the flattened policy-parameter axis: its control sizes are
    interpreted here as player parameter counts. The callback receives the whole
    ``case_shape + (parameter_size,)`` array and returns raw
    ``case_shape + (path, player)`` costs from the supplied
    :class:`PreparedControlledNoise`. Leading case axes denote independent
    parameter-game cases.

    ``callback_id`` identifies the callback semantics and ``feasible_set_id``
    identifies the caller's unchanged policy-feasible set. The latter is
    provenance only: this unconstrained root solve establishes local SAA
    stationarity and does not certify boundary KKT conditions or global
    feasibility.
    """

    path_cost_function: StochasticPolicyPathCosts
    partition: PlayerControlPartition
    args: Any
    case_shape: tuple[int, ...] = eqx.field(static=True)
    parameter_size: int = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    callback_id: str = eqx.field(static=True)
    feasible_set_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        path_cost_function: StochasticPolicyPathCosts,
        partition: PlayerControlPartition,
        /,
        *,
        case_shape: Sequence[int] = (),
        args: Any = None,
        callback_id: str,
        feasible_set_id: str,
        problem_id: str,
    ):
        if not callable(path_cost_function):
            raise TypeError("path_cost_function must be callable.")
        if not isinstance(partition, PlayerControlPartition):
            raise TypeError("partition must be a PlayerControlPartition.")
        cases = _shape(case_shape, "case_shape", allow_empty=True)
        self.path_cost_function = path_cost_function
        self.partition = partition
        self.args = args
        self.case_shape = cases
        self.parameter_size = partition.joint_control_size
        self.num_players = partition.num_players
        self.callback_id = _identifier(callback_id, "callback_id")
        self.feasible_set_id = _identifier(feasible_set_id, "feasible_set_id")
        self.problem_id = _identifier(problem_id, "problem_id")

    @property
    def cost_function(self) -> StochasticPolicyPathCosts:
        """Return the identified raw path-cost callback."""

        return self.path_cost_function


class StochasticPolicyGamePlan(StrictModule):
    """Fixed parameter ownership, case topology, and nonlinear-root policy."""

    partition: PlayerControlPartition
    method: NewtonTrustRegion
    termination: NonlinearTermination
    case_shape: tuple[int, ...] = eqx.field(static=True)
    parameter_size: int = eqx.field(static=True)
    num_players: int = eqx.field(static=True)
    parameter_owner: tuple[int, ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    callback_id: str = eqx.field(static=True)
    feasible_set_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    certification_claim: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class _SAAResidual(StrictModule):
    problem: StochasticPolicyGameProblem
    training_noise: PreparedControlledNoise
    training_weights: Array
    ownership: Array

    def __call__(self, parameters: Array, args: Any, /) -> Array:
        del args
        residual, costs, gradients, _owned, _objectives = _training_evaluation(
            self.problem,
            parameters,
            self.training_noise,
            self.training_weights,
            self.ownership,
        )
        finite = (
            _noise_complete(self.training_noise)
            & jnp.all(jnp.isfinite(costs))
            & jnp.all(jnp.isfinite(gradients))
        )
        return jnp.where(finite, residual, jnp.full_like(residual, jnp.nan))


class PreparedStochasticPolicyGame(StrictModule):
    """A frozen training root and a disjoint, evaluation-only holdout bundle."""

    plan: StochasticPolicyGamePlan
    problem: StochasticPolicyGameProblem
    initial_parameters: Array
    training_noise: PreparedControlledNoise
    holdout_noise: PreparedControlledNoise
    training_weights: Array
    holdout_weights: Array
    ownership: Array
    holdout_cluster_membership: Array
    root_problem: NonlinearSystemProblem
    root_prepared: PreparedNonlinearSolve
    numeric_version: Array
    training_bundle_id: str = eqx.field(static=True)
    holdout_bundle_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    @property
    def training_realization_ids(self) -> tuple[str, ...]:
        return self.training_noise.realization_ids

    @property
    def holdout_realization_ids(self) -> tuple[str, ...]:
        return self.holdout_noise.realization_ids


class StochasticPolicyGameResult(StrictModule):
    """A local SAA stationarity candidate with independent holdout evidence.

    ``training_complete_path_gradients`` has shape
    ``case_shape + (training_paths, num_players, parameter_size)``. Each player
    derivative is taken through that player's complete path cost. Only afterward
    does ``training_owned_path_gradients`` select, for every parameter row, the
    derivative of its owning player. ``original_residual`` is the resulting
    unscaled weighted SAA pseudo-gradient.

    Holdout paths are evaluated once at the accepted parameters and never enter
    candidate selection or the root iteration. Cluster means aggregate paths
    sharing an ``independence_label`` and are evidence, not a population bound.
    Even a successful result claims at most ``LOCAL_SAA_POLICY_STATIONARITY``;
    it is neither a population Nash nor a feedback-Nash certificate.
    """

    partition: PlayerControlPartition
    parameters: Array
    original_residual: Array
    training_path_costs: Array
    training_complete_path_gradients: Array
    training_owned_path_gradients: Array
    training_saa_costs: Array
    holdout_path_costs: Array
    holdout_saa_costs: Array
    holdout_cluster_costs: Array
    holdout_cluster_weights: Array
    holdout_cluster_counts: Array
    holdout_cluster_valid: Array
    training_weights: Array
    holdout_weights: Array
    training_independence_labels: Array
    holdout_independence_labels: Array
    status: Array
    valid: Array
    stationarity_certified: Array
    numeric_version: Array
    root_result: NonlinearResult
    training_realization_ids: tuple[str, ...] = eqx.field(static=True)
    holdout_realization_ids: tuple[str, ...] = eqx.field(static=True)
    training_coupling_id: str = eqx.field(static=True)
    holdout_coupling_id: str = eqx.field(static=True)
    training_bundle_id: str = eqx.field(static=True)
    holdout_bundle_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    callback_id: str = eqx.field(static=True)
    feasible_set_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    certification_claim: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid

    @property
    def original_unscaled_residual(self) -> Array:
        return self.original_residual

    @property
    def training_path_cost_gradients(self) -> Array:
        return self.training_complete_path_gradients

    @property
    def training_owned_path_cost_gradients(self) -> Array:
        return self.training_owned_path_gradients

    @property
    def root_diagnostics(self) -> NonlinearDiagnostics:
        return self.root_result.diagnostics

    @property
    def root_provenance(self) -> NonlinearProvenance:
        return self.root_result.provenance

    @property
    def root_status(self) -> Array:
        return self.root_result.status


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _shape(
    value: Sequence[int],
    owner: str,
    /,
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{owner} must be a sequence of positive integers.")
    raw = tuple(value)
    if any(isinstance(size, bool) for size in raw):
        raise TypeError(f"{owner} dimensions must be integers, not booleans.")
    result = tuple(index(size) for size in raw)
    if any(size <= 0 for size in result) or (not allow_empty and not result):
        requirement = "positive" if allow_empty else "a nonempty sequence of positive"
        raise ValueError(f"{owner} must contain {requirement} integer dimensions.")
    return result


def _real_array(value: ArrayLike, owner: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.number) or jnp.issubdtype(
        array.dtype, jnp.complexfloating
    ):
        raise TypeError(f"{owner} must be a real numeric array.")
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _parameters(
    value: ArrayLike,
    plan: StochasticPolicyGamePlan,
    owner: str,
    /,
) -> Array:
    array = _real_array(value, owner)
    expected = plan.case_shape + (plan.parameter_size,)
    if tuple(array.shape) != expected:
        raise ValueError(f"{owner} must have shape {expected}; got {array.shape}.")
    return array


def _weights(
    value: ArrayLike | None,
    path_count: int,
    dtype: jnp.dtype,
    owner: str,
    /,
) -> Array:
    if value is None:
        return jnp.full((path_count,), 1.0 / path_count, dtype=dtype)
    weights = _real_array(value, owner).astype(dtype)
    if tuple(weights.shape) != (path_count,):
        raise ValueError(f"{owner} must have shape ({path_count},); got {weights.shape}.")
    host = np.asarray(weights)
    if not np.all(np.isfinite(host)) or np.any(host < 0.0):
        raise ValueError(f"{owner} must be finite and non-negative.")
    total = float(np.sum(host))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{owner} must have a strictly positive finite sum.")
    return weights / jnp.asarray(total, dtype=dtype)


def _bundle_id(noise: PreparedControlledNoise, role: str, /) -> str:
    payload = {
        "role": role,
        "realization_ids": noise.realization_ids,
        "coupling_id": noise.coupling_id,
        "noise_shape": noise.noise_shape,
        "num_paths": noise.num_paths,
        "num_steps": noise.num_steps,
    }
    return f"{role}-controlled-noise:{canonical_fingerprint(payload)}"


def _cluster_membership(noise: PreparedControlledNoise, dtype: jnp.dtype, /) -> Array:
    labels = np.asarray(noise.independence_labels, dtype=np.int64)
    compact_by_label: dict[int, int] = {}
    compact = []
    for raw in labels.tolist():
        label = int(raw)
        if label not in compact_by_label:
            compact_by_label[label] = len(compact_by_label)
        compact.append(compact_by_label[label])
    return jax.nn.one_hot(
        jnp.asarray(compact, dtype=jnp.int32),
        len(compact_by_label),
        dtype=dtype,
    )


def _validate_bundles(
    training_noise: PreparedControlledNoise,
    holdout_noise: PreparedControlledNoise,
    /,
) -> None:
    if not isinstance(training_noise, PreparedControlledNoise):
        raise TypeError("training_noise must be PreparedControlledNoise.")
    if not isinstance(holdout_noise, PreparedControlledNoise):
        raise TypeError("holdout_noise must be PreparedControlledNoise.")
    if training_noise.noise_shape != holdout_noise.noise_shape:
        raise ValueError("Training and holdout noise_shape values must match.")
    if training_noise.num_steps != holdout_noise.num_steps:
        raise ValueError("Training and holdout bundles must have the same num_steps.")
    overlap = set(training_noise.realization_ids).intersection(
        holdout_noise.realization_ids
    )
    if overlap:
        raise ValueError(
            "Training and holdout realization_ids must be disjoint; shared IDs: "
            f"{tuple(sorted(overlap))}."
        )
    if training_noise.coupling_id == holdout_noise.coupling_id:
        raise ValueError(
            "Training and holdout bundles must have distinct coupling_id values "
            "to identify an independent holdout."
        )


def _same_noise_topology(
    reference: PreparedControlledNoise,
    candidate: PreparedControlledNoise,
    owner: str,
    /,
) -> None:
    if not isinstance(candidate, PreparedControlledNoise):
        raise TypeError(f"{owner} must be PreparedControlledNoise.")
    if (
        reference.noise_shape != candidate.noise_shape
        or reference.num_paths != candidate.num_paths
        or reference.num_steps != candidate.num_steps
        or tuple(reference.increments.shape) != tuple(candidate.increments.shape)
        or reference.increments.dtype != candidate.increments.dtype
    ):
        raise ValueError(f"{owner} changed the prepared noise topology.")
    old_clusters = len(set(np.asarray(reference.independence_labels).tolist()))
    new_clusters = len(set(np.asarray(candidate.independence_labels).tolist()))
    if old_clusters != new_clusters:
        raise ValueError(f"{owner} changed the independence-cluster topology.")


def _require_new_realizations(
    candidate: PreparedControlledNoise,
    previous: PreparedStochasticPolicyGame,
    owner: str,
    /,
) -> None:
    old_ids = set(previous.training_noise.realization_ids).union(
        previous.holdout_noise.realization_ids
    )
    overlap = old_ids.intersection(candidate.realization_ids)
    if overlap:
        raise ValueError(
            f"{owner} replacement must use new realization_ids; reused IDs: "
            f"{tuple(sorted(overlap))}."
        )


def _validate_topology(
    plan: StochasticPolicyGamePlan,
    problem: StochasticPolicyGameProblem,
    /,
) -> None:
    if not isinstance(plan, StochasticPolicyGamePlan):
        raise TypeError("plan must be a StochasticPolicyGamePlan.")
    if not isinstance(problem, StochasticPolicyGameProblem):
        raise TypeError("problem must be a StochasticPolicyGameProblem.")
    if problem.partition.partition_id != plan.partition.partition_id:
        raise ValueError("problem changed the planned parameter ownership partition.")
    if problem.case_shape != plan.case_shape:
        raise ValueError("problem changed the planned case_shape.")
    if problem.parameter_size != plan.parameter_size:
        raise ValueError("problem changed the planned parameter_size.")
    if problem.problem_id != plan.problem_id:
        raise ValueError("problem changed the planned problem_id.")
    if problem.callback_id != plan.callback_id:
        raise ValueError("problem changed the planned callback_id.")
    if problem.feasible_set_id != plan.feasible_set_id:
        raise ValueError("problem changed the planned feasible_set_id.")


def _path_costs(
    problem: StochasticPolicyGameProblem,
    parameters: Array,
    noise: PreparedControlledNoise,
    /,
) -> Array:
    costs = _real_array(
        problem.path_cost_function(parameters, noise, problem.args),
        "path_cost_function result",
    )
    expected = problem.case_shape + (noise.num_paths, problem.num_players)
    if tuple(costs.shape) != expected:
        raise ValueError(
            "path_cost_function must return raw case + (path, player) costs "
            f"with shape {expected}; got {costs.shape}."
        )
    return costs


def _training_evaluation(
    problem: StochasticPolicyGameProblem,
    parameters: Array,
    noise: PreparedControlledNoise,
    weights: Array,
    ownership: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    cost_count = noise.num_paths * problem.num_players
    if cost_count <= problem.parameter_size:
        costs, pullback = jax.vjp(
            lambda value: _path_costs(problem, value, noise), parameters
        )
        cost_basis = jnp.eye(cost_count, dtype=parameters.dtype).reshape(
            (cost_count, noise.num_paths, problem.num_players)
        )
        cotangents = jnp.broadcast_to(
            cost_basis.reshape(
                (cost_count,)
                + (1,) * len(problem.case_shape)
                + (noise.num_paths, problem.num_players)
            ),
            (cost_count,) + problem.case_shape + (noise.num_paths, problem.num_players),
        )
        reverse_rows = jax.vmap(lambda cotangent: pullback(cotangent)[0])(cotangents)
        reverse_rows = reverse_rows.reshape(
            (noise.num_paths, problem.num_players)
            + problem.case_shape
            + (problem.parameter_size,)
        )
        case_axes = tuple(range(2, 2 + len(problem.case_shape)))
        complete_gradient = jnp.transpose(
            reverse_rows,
            case_axes + (0, 1, 2 + len(problem.case_shape)),
        )
    else:
        costs, pushforward = jax.linearize(
            lambda value: _path_costs(problem, value, noise), parameters
        )
        parameter_basis = jnp.eye(problem.parameter_size, dtype=parameters.dtype)
        tangents = jnp.broadcast_to(
            parameter_basis.reshape(
                (problem.parameter_size,)
                + (1,) * len(problem.case_shape)
                + (problem.parameter_size,)
            ),
            (problem.parameter_size,) + problem.case_shape + (problem.parameter_size,),
        )
        forward_columns = jax.vmap(pushforward)(tangents)
        complete_gradient = jnp.moveaxis(forward_columns, 0, -1)
    owned_gradient = ein.contract("...pkm,mk->...pm", complete_gradient, ownership)
    residual = ein.contract("p,...pm->...m", weights, owned_gradient)
    objectives = ein.contract("p,...pk->...k", weights, costs)
    return residual, costs, complete_gradient, owned_gradient, objectives


def _holdout_evaluation(
    problem: StochasticPolicyGameProblem,
    parameters: Array,
    noise: PreparedControlledNoise,
    weights: Array,
    /,
) -> tuple[Array, Array]:
    costs = _path_costs(problem, parameters, noise)
    finite = jnp.all(jnp.isfinite(costs), axis=(-2, -1)) & _noise_complete(noise)
    objectives = ein.contract(
        "p,...pk->...k",
        weights,
        jnp.where(jnp.isfinite(costs), costs, 0.0),
    )
    objectives = jnp.where(finite[..., None], objectives, jnp.nan)
    return costs, objectives


def _noise_path_valid(noise: PreparedControlledNoise, /) -> Array:
    increment_axes = tuple(range(1, noise.increments.ndim))
    return noise.valid & jnp.all(jnp.isfinite(noise.increments), axis=increment_axes)


def _noise_complete(noise: PreparedControlledNoise, /) -> Array:
    return jnp.all(_noise_path_valid(noise))


def _cluster_evidence(
    costs: Array,
    noise: PreparedControlledNoise,
    weights: Array,
    membership: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    path_valid = _noise_path_valid(noise)
    finite = jnp.isfinite(costs) & path_valid.reshape(
        (1,) * len(costs.shape[:-2]) + (noise.num_paths, 1)
    )
    safe_costs = jnp.where(finite, costs, 0.0)
    weighted_membership = membership * weights[:, None]
    cluster_weights = jnp.sum(weighted_membership, axis=0)
    cluster_counts = jnp.sum(membership, axis=0).astype(jnp.int32)
    numerator = ein.contract("pc,...pk->...ck", weighted_membership, safe_costs)
    cluster_costs = numerator / cluster_weights.reshape(
        (1,) * len(costs.shape[:-2]) + (membership.shape[-1], 1)
    )
    finite_count = ein.contract(
        "pc,...pk->...ck", membership, finite.astype(membership.dtype)
    )
    cluster_valid = finite_count == cluster_counts.reshape(
        (1,) * len(costs.shape[:-2]) + (membership.shape[-1], 1)
    )
    cluster_costs = jnp.where(cluster_valid, cluster_costs, jnp.nan)
    return cluster_costs, cluster_weights, cluster_counts, cluster_valid


def _root_problem(
    plan: StochasticPolicyGamePlan,
    problem: StochasticPolicyGameProblem,
    parameters: Array,
    training_noise: PreparedControlledNoise,
    training_weights: Array,
    ownership: Array,
    /,
) -> NonlinearSystemProblem:
    space = ArraySpace(tuple(parameters.shape), dtype=parameters.dtype)
    return NonlinearSystemProblem(
        _SAAResidual(problem, training_noise, training_weights, ownership),
        state_space=space,
        residual_space=space,
        problem_id=f"{plan.plan_id}:training-saa-root",
    )


def plan_stochastic_policy_game(
    problem: StochasticPolicyGameProblem,
    /,
    *,
    method: NewtonTrustRegion | None = None,
    termination: NonlinearTermination | None = None,
) -> StochasticPolicyGamePlan:
    """Plan a frozen-path SAA pseudo-gradient root with explicit ownership."""

    if not isinstance(problem, StochasticPolicyGameProblem):
        raise TypeError("problem must be a StochasticPolicyGameProblem.")
    method_ = NewtonTrustRegion() if method is None else method
    termination_ = NonlinearTermination() if termination is None else termination
    if not isinstance(method_, NewtonTrustRegion):
        raise TypeError("method must be NewtonTrustRegion or None.")
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    payload = {
        "kind": _METHOD_ID,
        "problem_id": problem.problem_id,
        "partition_id": problem.partition.partition_id,
        "case_shape": problem.case_shape,
        "parameter_size": problem.parameter_size,
        "callback_id": problem.callback_id,
        "feasible_set_id": problem.feasible_set_id,
        "root_method_id": method_.method_id,
    }
    return StochasticPolicyGamePlan(
        problem.partition,
        method_,
        termination_,
        problem.case_shape,
        problem.parameter_size,
        problem.num_players,
        problem.partition.control_owner,
        problem.problem_id,
        problem.callback_id,
        problem.feasible_set_id,
        _METHOD_ID,
        _CERTIFICATE,
        f"stochastic-policy-game-plan:{canonical_fingerprint(payload)}",
    )


def prepare_stochastic_policy_game(
    plan: StochasticPolicyGamePlan,
    problem: StochasticPolicyGameProblem,
    initial_parameters: ArrayLike,
    training_noise: PreparedControlledNoise,
    holdout_noise: PreparedControlledNoise,
    /,
    *,
    training_weights: ArrayLike | None = None,
    holdout_weights: ArrayLike | None = None,
) -> PreparedStochasticPolicyGame:
    """Freeze disjoint training/holdout paths and prepare the training root."""

    _validate_topology(plan, problem)
    _validate_bundles(training_noise, holdout_noise)
    parameters = _parameters(initial_parameters, plan, "initial_parameters")
    train_weights = _weights(
        training_weights,
        training_noise.num_paths,
        parameters.dtype,
        "training_weights",
    )
    held_weights = _weights(
        holdout_weights,
        holdout_noise.num_paths,
        parameters.dtype,
        "holdout_weights",
    )
    ownership = jax.nn.one_hot(
        jnp.asarray(plan.parameter_owner, dtype=jnp.int32),
        plan.num_players,
        dtype=parameters.dtype,
    )
    holdout_membership = _cluster_membership(holdout_noise, parameters.dtype)
    root_problem = _root_problem(
        plan,
        problem,
        parameters,
        training_noise,
        train_weights,
        ownership,
    )
    root_prepared = prepare_nonlinear(
        root_problem,
        parameters,
        method=plan.method,
        termination=plan.termination,
    )
    prepared_payload = {
        "plan_id": plan.plan_id,
        "parameter_dtype": np.dtype(parameters.dtype).str,
        "training_topology": (
            training_noise.num_paths,
            training_noise.num_steps,
            training_noise.noise_shape,
            len(set(np.asarray(training_noise.independence_labels).tolist())),
        ),
        "holdout_topology": (
            holdout_noise.num_paths,
            holdout_noise.num_steps,
            holdout_noise.noise_shape,
            int(holdout_membership.shape[-1]),
        ),
    }
    return PreparedStochasticPolicyGame(
        plan,
        problem,
        parameters,
        training_noise,
        holdout_noise,
        train_weights,
        held_weights,
        ownership,
        holdout_membership,
        root_problem,
        root_prepared,
        jnp.asarray(0, dtype=jnp.int32),
        _bundle_id(training_noise, "training"),
        _bundle_id(holdout_noise, "holdout"),
        f"prepared-stochastic-policy-game:{canonical_fingerprint(prepared_payload)}",
    )


def refresh_stochastic_policy_game(
    prepared: PreparedStochasticPolicyGame,
    problem: StochasticPolicyGameProblem,
    initial_parameters: ArrayLike | None = None,
    /,
    *,
    training_noise: PreparedControlledNoise | None = None,
    holdout_noise: PreparedControlledNoise | None = None,
    training_weights: ArrayLike | None | object = _UNSET,
    holdout_weights: ArrayLike | None | object = _UNSET,
) -> PreparedStochasticPolicyGame:
    """Refresh numerics, accepting only same-topology bundles with new path IDs."""

    if not isinstance(prepared, PreparedStochasticPolicyGame):
        raise TypeError("prepared must be a PreparedStochasticPolicyGame.")
    _validate_topology(prepared.plan, problem)
    parameters = (
        prepared.initial_parameters
        if initial_parameters is None
        else _parameters(initial_parameters, prepared.plan, "initial_parameters")
    )
    train = prepared.training_noise if training_noise is None else training_noise
    held = prepared.holdout_noise if holdout_noise is None else holdout_noise
    if training_noise is not None:
        _same_noise_topology(prepared.training_noise, train, "training_noise")
        _require_new_realizations(train, prepared, "training_noise")
    if holdout_noise is not None:
        _same_noise_topology(prepared.holdout_noise, held, "holdout_noise")
        _require_new_realizations(held, prepared, "holdout_noise")
    _validate_bundles(train, held)
    train_weights = (
        prepared.training_weights
        if training_weights is _UNSET
        else _weights(
            training_weights,  # type: ignore[arg-type]
            train.num_paths,
            parameters.dtype,
            "training_weights",
        )
    )
    held_weights = (
        prepared.holdout_weights
        if holdout_weights is _UNSET
        else _weights(
            holdout_weights,  # type: ignore[arg-type]
            held.num_paths,
            parameters.dtype,
            "holdout_weights",
        )
    )
    ownership = prepared.ownership.astype(parameters.dtype)
    holdout_membership = _cluster_membership(held, parameters.dtype)
    root_problem = _root_problem(
        prepared.plan,
        problem,
        parameters,
        train,
        train_weights,
        ownership,
    )
    root_prepared = refresh_nonlinear(
        prepared.root_prepared,
        root_problem,
        parameters,
    )
    return PreparedStochasticPolicyGame(
        prepared.plan,
        problem,
        parameters,
        train,
        held,
        train_weights,
        held_weights,
        ownership,
        holdout_membership,
        root_problem,
        root_prepared,
        prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
        _bundle_id(train, "training"),
        _bundle_id(held, "holdout"),
        prepared.prepared_id,
    )


def _result_status(
    prepared: PreparedStochasticPolicyGame,
    root_result: NonlinearResult,
    parameters: Array,
    training_costs: Array,
    training_gradients: Array,
    holdout_costs: Array,
    /,
) -> Array:
    case_shape = prepared.plan.case_shape
    parameters_finite = jnp.all(jnp.isfinite(parameters), axis=-1)
    training_costs_finite = jnp.all(jnp.isfinite(training_costs), axis=(-2, -1))
    training_gradients_finite = jnp.all(
        jnp.isfinite(training_gradients), axis=(-3, -2, -1)
    )
    holdout_costs_finite = jnp.all(jnp.isfinite(holdout_costs), axis=(-2, -1))
    root_successful = root_result.status == int(NonlinearStatus.SUCCESS)
    initial_nonfinite = root_result.status == int(NonlinearStatus.NONFINITE_INPUT)
    shape = case_shape if case_shape else ()
    status = jnp.full(shape, int(StochasticPolicyGameStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~holdout_costs_finite,
        int(StochasticPolicyGameStatus.NONFINITE_HOLDOUT_PATH_COSTS),
        status,
    )
    status = jnp.where(
        ~_noise_complete(prepared.holdout_noise),
        int(StochasticPolicyGameStatus.INVALID_HOLDOUT_BUNDLE),
        status,
    )
    status = jnp.where(
        ~root_successful,
        int(StochasticPolicyGameStatus.ROOT_FAILED),
        status,
    )
    status = jnp.where(
        ~training_gradients_finite,
        int(StochasticPolicyGameStatus.NONFINITE_TRAINING_PATH_GRADIENTS),
        status,
    )
    status = jnp.where(
        ~training_costs_finite,
        int(StochasticPolicyGameStatus.NONFINITE_TRAINING_PATH_COSTS),
        status,
    )
    status = jnp.where(
        ~_noise_complete(prepared.training_noise),
        int(StochasticPolicyGameStatus.INVALID_TRAINING_BUNDLE),
        status,
    )
    status = jnp.where(
        initial_nonfinite | ~parameters_finite,
        int(StochasticPolicyGameStatus.NONFINITE_INITIAL_PARAMETERS),
        status,
    )
    return status


def solve_prepared_stochastic_policy_game(
    prepared: PreparedStochasticPolicyGame,
    /,
    *,
    termination: NonlinearTermination | None = None,
) -> StochasticPolicyGameResult:
    """Solve on frozen training paths, then evaluate the untouched holdout paths."""

    if not isinstance(prepared, PreparedStochasticPolicyGame):
        raise TypeError("prepared must be a PreparedStochasticPolicyGame.")
    if termination is not None and not isinstance(termination, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    root_result = solve_prepared_nonlinear(
        prepared.root_prepared, termination=termination
    )
    parameters = root_result.state
    (
        residual,
        training_costs,
        training_gradients,
        training_owned,
        training_objectives,
    ) = _training_evaluation(
        prepared.problem,
        parameters,
        prepared.training_noise,
        prepared.training_weights,
        prepared.ownership,
    )
    finite_training = (
        _noise_complete(prepared.training_noise)
        & jnp.all(jnp.isfinite(training_costs), axis=(-2, -1))
        & jnp.all(jnp.isfinite(training_gradients), axis=(-3, -2, -1))
    )
    residual = jnp.where(finite_training[..., None], residual, jnp.nan)
    training_objectives = jnp.where(
        finite_training[..., None], training_objectives, jnp.nan
    )
    holdout_costs, holdout_objectives = _holdout_evaluation(
        prepared.problem,
        parameters,
        prepared.holdout_noise,
        prepared.holdout_weights,
    )
    (
        cluster_costs,
        cluster_weights,
        cluster_counts,
        cluster_valid,
    ) = _cluster_evidence(
        holdout_costs,
        prepared.holdout_noise,
        prepared.holdout_weights,
        prepared.holdout_cluster_membership,
    )
    status = _result_status(
        prepared,
        root_result,
        parameters,
        training_costs,
        training_gradients,
        holdout_costs,
    )
    valid = status == int(StochasticPolicyGameStatus.SUCCESS)
    return StochasticPolicyGameResult(
        prepared.plan.partition,
        parameters,
        residual,
        training_costs,
        training_gradients,
        training_owned,
        training_objectives,
        holdout_costs,
        holdout_objectives,
        cluster_costs,
        cluster_weights,
        cluster_counts,
        cluster_valid,
        prepared.training_weights,
        prepared.holdout_weights,
        prepared.training_noise.independence_labels,
        prepared.holdout_noise.independence_labels,
        status,
        valid,
        valid,
        prepared.numeric_version,
        root_result,
        prepared.training_noise.realization_ids,
        prepared.holdout_noise.realization_ids,
        prepared.training_noise.coupling_id,
        prepared.holdout_noise.coupling_id,
        prepared.training_bundle_id,
        prepared.holdout_bundle_id,
        prepared.plan.problem_id,
        prepared.plan.callback_id,
        prepared.plan.feasible_set_id,
        prepared.plan.plan_id,
        prepared.prepared_id,
        prepared.plan.method_id,
        prepared.plan.certification_claim,
    )


def solve_stochastic_policy_game(
    problem: StochasticPolicyGameProblem,
    initial_parameters: ArrayLike,
    training_noise: PreparedControlledNoise,
    holdout_noise: PreparedControlledNoise,
    /,
    *,
    training_weights: ArrayLike | None = None,
    holdout_weights: ArrayLike | None = None,
    method: NewtonTrustRegion | None = None,
    termination: NonlinearTermination | None = None,
) -> StochasticPolicyGameResult:
    """Plan, prepare, and solve one frozen-sample policy game."""

    plan = plan_stochastic_policy_game(
        problem,
        method=method,
        termination=termination,
    )
    prepared = prepare_stochastic_policy_game(
        plan,
        problem,
        initial_parameters,
        training_noise,
        holdout_noise,
        training_weights=training_weights,
        holdout_weights=holdout_weights,
    )
    return solve_prepared_stochastic_policy_game(prepared)


__all__ = [
    "PreparedStochasticPolicyGame",
    "StochasticPolicyGamePlan",
    "StochasticPolicyGameProblem",
    "StochasticPolicyGameResult",
    "StochasticPolicyGameStatus",
    "StochasticPolicyPathCosts",
    "plan_stochastic_policy_game",
    "prepare_stochastic_policy_game",
    "refresh_stochastic_policy_game",
    "solve_prepared_stochastic_policy_game",
    "solve_stochastic_policy_game",
]
