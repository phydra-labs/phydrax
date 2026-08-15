#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._strict import StrictModule
from ..optim import FiniteAxis, FiniteExhaustiveSearch, FiniteProductSpace
from ..optim._finite import _exhaustive_minimum
from ._posterior import PosteriorProblem


_MAP_CANDIDATE_METHOD_ID = "finite-exhaustive-map-candidate-search-v1"
_DEFAULT_FINITE_SEARCH = FiniteExhaustiveSearch()


class _MAPCandidateEvaluator(StrictModule):
    problem: PosteriorProblem

    def __init__(self, problem: PosteriorProblem, /):
        self.problem = problem

    def __call__(self, position: PyTree[Array], /) -> tuple[Array, Array]:
        objective = self.problem.negative_log_density(position)
        return objective, jnp.isfinite(objective)


class MAPCandidateSearchResult(StrictModule):
    """Exact finite-candidate posterior minimum and enumeration evidence."""

    problem: PosteriorProblem
    position: PyTree[Array] | None
    parameters: PyTree[Array] | None
    objective: Array
    log_density: Array
    search: FiniteExhaustiveSearch
    valid: bool = eqx.field(static=True)
    termination_reason: str = eqx.field(static=True)
    flat_index: int = eqx.field(static=True)
    product_index: tuple[int, ...] = eqx.field(static=True)
    axis_paths: tuple[str, ...] = eqx.field(static=True)
    product_shape: tuple[int, ...] = eqx.field(static=True)
    candidate_count: int = eqx.field(static=True)
    objective_evaluations: int = eqx.field(static=True)
    valid_evaluations: int = eqx.field(static=True)
    invalid_evaluations: int = eqx.field(static=True)
    effective_batch_size: int = eqx.field(static=True)
    candidate_signature: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: PosteriorProblem,
        position: PyTree[Array] | None,
        objective: ArrayLike,
        search: FiniteExhaustiveSearch,
        valid: bool,
        flat_index: int,
        product_index: tuple[int, ...],
        axis_paths: tuple[str, ...],
        product_shape: tuple[int, ...],
        candidate_count: int,
        objective_evaluations: int,
        invalid_evaluations: int,
        effective_batch_size: int,
        candidate_signature: str,
    ):
        valid_ = bool(valid)
        if valid_ != (position is not None):
            raise ValueError("A valid MAP candidate result requires one position.")
        evaluations = int(objective_evaluations)
        invalid = int(invalid_evaluations)
        if evaluations != int(candidate_count):
            raise ValueError("MAP candidate evaluation count must equal candidate_count.")
        if invalid < 0 or invalid > evaluations:
            raise ValueError("invalid_evaluations must lie within the evaluation count.")
        if valid_:
            if flat_index < 0 or any(index < 0 for index in product_index):
                raise ValueError("Valid MAP candidate indices must be nonnegative.")
            position_ = jax.tree_util.tree_map(jnp.asarray, position)
            parameters = problem.parameter_space.constrain(position_)
            objective_ = jnp.asarray(objective, dtype=float).reshape(())
            termination_reason = "finite_minimum"
        else:
            if flat_index != -1 or any(index != -1 for index in product_index):
                raise ValueError("Invalid MAP candidate indices must use -1 sentinels.")
            position_ = None
            parameters = None
            objective_ = jnp.asarray(jnp.nan, dtype=float)
            termination_reason = "no_finite_candidates"

        self.problem = problem
        self.position = position_
        self.parameters = parameters
        self.objective = objective_
        self.log_density = -objective_
        self.search = search
        self.valid = valid_
        self.termination_reason = termination_reason
        self.flat_index = int(flat_index)
        self.product_index = tuple(int(index) for index in product_index)
        self.axis_paths = tuple(str(path) for path in axis_paths)
        self.product_shape = tuple(int(size) for size in product_shape)
        self.candidate_count = int(candidate_count)
        self.objective_evaluations = evaluations
        self.valid_evaluations = evaluations - invalid
        self.invalid_evaluations = invalid
        self.effective_batch_size = int(effective_batch_size)
        self.candidate_signature = str(candidate_signature)
        self.method_id = _MAP_CANDIDATE_METHOD_ID


def _validate_map_candidate_space(
    problem: PosteriorProblem,
    candidates: FiniteProductSpace,
    /,
) -> None:
    candidate_spec = candidates.point_spec()
    initial = problem.initial_position
    candidate_structure = jax.tree_util.tree_structure(candidate_spec)
    initial_structure = jax.tree_util.tree_structure(initial)
    if candidate_structure != initial_structure:
        raise ValueError(
            "Finite candidate points must match the posterior initial-position "
            "PyTree structure."
        )

    path_specs = jax.tree_util.tree_flatten_with_path(candidate_spec)[0]
    initial_leaves = jax.tree_util.tree_leaves(initial)
    for (path, spec), initial_leaf in zip(path_specs, initial_leaves, strict=True):
        initial_array = jnp.asarray(initial_leaf)
        expected_shape = tuple(int(size) for size in initial_array.shape)
        if tuple(spec.shape) != expected_shape:
            raise ValueError(
                f"Finite candidate point leaf {jax.tree_util.keystr(path) or '<root>'} "
                f"must have shape {expected_shape}, got {spec.shape}."
            )
        if np.dtype(spec.dtype) != np.dtype(initial_array.dtype):
            raise TypeError(
                f"Finite candidate point leaf {jax.tree_util.keystr(path) or '<root>'} "
                f"must have dtype {initial_array.dtype}, got {spec.dtype}."
            )

    axis_blocks = jax.tree_util.tree_leaves(
        candidates.axes,
        is_leaf=lambda value: isinstance(value, FiniteAxis),
    )
    for axis_path, axis in zip(candidates.axis_paths, axis_blocks, strict=True):
        for payload_path, values in jax.tree_util.tree_flatten_with_path(axis.values)[0]:
            if not eqx.is_inexact_array(values):
                raise TypeError(
                    f"MAP candidate axis {axis_path} payload "
                    f"{jax.tree_util.keystr(payload_path) or '<root>'} must be inexact."
                )
            if bool(jnp.any(~jnp.isfinite(values))):
                raise ValueError("MAP candidate coordinates must be finite.")


def search_map_candidates(
    problem: PosteriorProblem,
    candidates: FiniteProductSpace,
    /,
    *,
    search: FiniteExhaustiveSearch = _DEFAULT_FINITE_SEARCH,
) -> MAPCandidateSearchResult:
    """Find the exact finite minimum of a posterior over declared positions."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    if not isinstance(candidates, FiniteProductSpace):
        raise TypeError("candidates must be a FiniteProductSpace.")
    if not isinstance(search, FiniteExhaustiveSearch):
        raise TypeError("search must be a FiniteExhaustiveSearch.")
    _validate_map_candidate_space(problem, candidates)

    evidence = _exhaustive_minimum(
        _MAPCandidateEvaluator(problem),
        candidates,
        search,
    )
    valid = bool(evidence.valid)
    flat_index = int(evidence.flat_index) if valid else -1
    product_index = (
        tuple(int(index) for index in evidence.product_index)
        if valid
        else (-1,) * len(candidates.product_shape)
    )
    position = (
        jax.tree_util.tree_map(
            jax.lax.stop_gradient,
            candidates.take(flat_index),
        )
        if valid
        else None
    )

    return MAPCandidateSearchResult(
        problem=problem,
        position=position,
        objective=evidence.minimum,
        search=search,
        valid=valid,
        flat_index=flat_index,
        product_index=product_index,
        axis_paths=candidates.axis_paths,
        product_shape=candidates.product_shape,
        candidate_count=candidates.size,
        objective_evaluations=int(evidence.attempted_evaluations),
        invalid_evaluations=int(evidence.invalid_evaluations),
        effective_batch_size=search.effective_batch_size(candidates.size),
        candidate_signature=candidates.signature(),
    )


__all__ = ["MAPCandidateSearchResult", "search_map_candidates"]
