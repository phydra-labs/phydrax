#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact static condensation of affine prepared coupled problems.

``condense_coupled_problem`` partitions the solve coordinates of an affine
``PreparedCoupledProblem`` into pivot and retained ``(owner, block)`` paths,
factorizes the pivot block ``A`` once at nominal arguments through
``phydrax.linalg.factorize``, and refuses a rank-deficient or ill-conditioned
pivot with its evidence. ``solve_condensed_problem`` refreshes that
factorization natively at the runtime arguments, eliminates the pivot with one
batched solve ``A^{-1} [B | b_p]``, solves the materialized Schur complement
``S = D - C A^{-1} B`` with a direct policy, reconstructs
``x_p = A^{-1} (b_p - B x_r)`` with the prepared pivot factorization, and
certifies the ORIGINAL coupled equations at the reconstructed full state.
Every native solve keeps its status; no solve is wrapped as a status-free
matrix action. Approximate pivot actions belong to preconditioners of the
original system, not to this exact elimination.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import assert_never, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._admissibility import guard_derivative_validity, refuse_derivative_dependencies
from ..._differentiation import (
    DerivativeRoute,
    DerivativeSurface,
    OwnerDerivativeCapability,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._validation import positive_finite_float
from ...linalg import (
    AbstractLinearOperator,
    BlockLinearOperator,
    BlockSelection,
    BlockSpace,
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FactorizationKind,
    FactorizationPolicy,
    factorize,
    LinearDerivativeSolvePolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSolveStatus,
    LinearSystem,
    MaterializationPolicy,
    materialize,
    PreparedFactorization,
    refresh_factorization,
    RHSLayout,
    select_block_operator,
    solve,
)
from ._acceptance import certify_coupled_state, CoupledSolution
from ._prepared_problem import PreparedCoupledProblem


type BlockPath = tuple[str, str]
type SolveState = tuple[tuple[Array, ...], ...]
type QualifiedMode = Literal["mathematical", "none"]

_FIXED_STRUCTURE_REFUSALS = {
    "geometry": (
        "meshes, interface bindings, quadrature, and the coupled layout are fixed "
        "prepared structure"
    ),
    "partition": (
        "the pivot/retained partition and the pivot factorization plan are fixed "
        "prepared structure"
    ),
}
_STOPPED_REFUSAL = (
    "differentiation mode 'none' stops the condensed solve; condense with mode "
    "'mathematical' to differentiate with respect to the coupled arguments"
)


@final
class CondensationEvidence(StrictModule):
    """Pivot evidence of one exact static condensation.

    ``rank`` is the pivot's numerical rank (``-1`` when the factorization
    reports it undetermined or singular); ``condition_estimate`` is the
    2-norm condition number from singular values, and NaN when the
    factorization kind reports none. ``pivot_status`` is the
    ``LinearSolveStatus`` of the pivot factorization and its solves (the first
    failing code, ``SINGULAR`` for a rank-deficient pivot). ``full_rank`` states
    ``rank == pivot_size``; ``within_condition_limit`` states that the declared
    condition limit (if any) holds for a finite estimate.
    """

    pivot_paths: tuple[BlockPath, ...] = eqx.field(static=True)
    retained_paths: tuple[BlockPath, ...] = eqx.field(static=True)
    pivot_size: int = eqx.field(static=True)
    retained_size: int = eqx.field(static=True)
    factorization: FactorizationKind = eqx.field(static=True)
    condition_limit: float | None = eqx.field(static=True)
    rank: Array
    condition_estimate: Array
    pivot_status: Array
    full_rank: Array
    within_condition_limit: Array


@final
class CondensedCoupledProblem(StrictModule):
    """Qualified exact static condensation of one affine prepared coupled problem.

    ``pivot``/``retained`` are single-member selections of
    ``prepared.state_space`` that partition it in canonical path order.
    ``factorization`` is the pivot factorization prepared at the nominal
    arguments (host side); ``solve_condensed_problem`` refreshes it natively at
    runtime arguments. ``schur_policy`` is the direct policy of the materialized
    Schur complement and ``materialization`` bounds every dense block.
    """

    prepared: PreparedCoupledProblem
    pivot: BlockSelection
    retained: BlockSelection
    factorization: PreparedFactorization
    schur_policy: LinearSolvePolicy
    materialization: MaterializationPolicy
    evidence: CondensationEvidence
    derivative_capability: OwnerDerivativeCapability
    condensation_id: str = eqx.field(static=True)


@final
class CondensedSolution(StrictModule):
    """Certified solution of an exact static condensation.

    ``solution`` carries the component and interface certificates of the
    ORIGINAL coupled equations at the reconstructed full state. The three
    native solves keep their per-right-hand-side status: ``elimination`` is the
    batched pivot solve ``A^{-1} [B | b_p]``, ``schur`` the retained solve
    ``S x_r = g``, and ``reconstruction`` the pivot solve
    ``x_p = A^{-1} (b_p - B x_r)``. ``evidence`` is the refreshed pivot
    evidence. ``accepted`` requires every native solve, a full-rank pivot within
    the condition limit, and every certificate; ``derivative_valid`` combines
    it with the native derivative admission.
    """

    solution: CoupledSolution
    elimination: LinearSolveResult
    schur: LinearSolveResult
    reconstruction: LinearSolveResult
    evidence: CondensationEvidence
    derivative_capability: OwnerDerivativeCapability
    accepted: Array
    derivative_valid: Array


def _state_paths(space: BlockSpace, /) -> tuple[BlockPath, ...]:
    paths: list[BlockPath] = []
    for owner, member in zip(space.names, space.spaces, strict=True):
        if not isinstance(member, BlockSpace):
            raise RuntimeError(f"Coupled owner {owner!r} does not publish named blocks.")
        paths.extend((owner, block) for block in member.names)
    return tuple(paths)


def _partition(
    prepared: PreparedCoupledProblem, pivot: Sequence[BlockPath], /
) -> tuple[tuple[BlockPath, ...], tuple[BlockPath, ...]]:
    """Validate pivot paths; return pivot and retained paths in canonical order."""
    if isinstance(pivot, str) or not isinstance(pivot, Sequence):
        raise TypeError("pivot must be a sequence of (owner, block) paths.")
    requested: list[BlockPath] = []
    for entry in pivot:
        if (
            not isinstance(entry, tuple)
            or len(entry) != 2
            or not all(isinstance(name, str) for name in entry)
        ):
            raise TypeError("pivot paths must be (owner, block) string pairs.")
        requested.append((entry[0], entry[1]))
    if not requested:
        raise ValueError("Static condensation requires a non-empty pivot.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"pivot repeats solve paths {sorted(requested)!r}.")
    paths = _state_paths(prepared.state_space)
    unknown = sorted(set(requested) - set(paths))
    if unknown:
        raise ValueError(
            f"pivot names unknown solve paths {unknown!r}; the coupled solve paths "
            f"are {list(paths)!r}."
        )
    chosen = set(requested)
    pivot_paths = tuple(path for path in paths if path in chosen)
    retained_paths = tuple(path for path in paths if path not in chosen)
    if not retained_paths:
        raise ValueError(
            "Static condensation requires a non-empty retained set; the pivot "
            "covers every solve path."
        )
    return pivot_paths, retained_paths


def _require_differentiation(
    factorization: FactorizationPolicy, schur_policy: LinearSolvePolicy, /
) -> QualifiedMode:
    mode = factorization.differentiation.mode
    qualified: QualifiedMode
    match mode:
        case "mathematical" | "none":
            qualified = mode
        case "rhs-only" | "algorithmic":
            raise ValueError(
                f"Static condensation refuses factorization differentiation mode "
                f"{mode!r}: coupled arguments bind the operator and the loads, so a "
                "partial or unrolled derivative of the pivot action is not the "
                "eliminated solution-map derivative. Use 'mathematical' or 'none'."
            )
        case _:
            assert_never(mode)
    if schur_policy.differentiation.mode != qualified:
        raise ValueError(
            f"schur_policy differentiation mode {schur_policy.differentiation.mode!r} "
            f"must equal the pivot factorization mode {qualified!r}; a mixed route is "
            "not the eliminated solution-map derivative."
        )
    return qualified


def _require_budget(
    prepared: PreparedCoupledProblem,
    policy: MaterializationPolicy,
    pivot_size: int,
    retained_size: int,
    /,
) -> None:
    """Refuse dense pivot, coupling, elimination, or Schur blocks beyond ``policy``."""
    leaves = jax.tree.leaves(prepared.state_space.structure())
    itemsize = np.dtype(jnp.result_type(*(spec.dtype for spec in leaves))).itemsize
    blocks = (
        ("pivot", pivot_size * pivot_size),
        ("coupling", pivot_size * retained_size),
        ("elimination", pivot_size * (retained_size + 1)),
        ("schur", retained_size * retained_size),
    )
    for name, entries in blocks:
        if entries > policy.max_entries or entries * itemsize > policy.max_bytes:
            raise ValueError(
                f"Static condensation {name} block needs {entries} dense entries "
                f"({entries * itemsize} bytes) for pivot size {pivot_size} and "
                f"retained size {retained_size}, exceeding the materialization "
                f"budget of {policy.max_entries} entries and {policy.max_bytes} bytes."
            )


def _pivot_operator(
    system_operator: AbstractLinearOperator, pivot: BlockSelection, /
) -> BlockLinearOperator:
    return select_block_operator(system_operator, pivot, pivot)


def _rank(factorization: PreparedFactorization, /) -> Array:
    if factorization.capabilities.rank:
        return jnp.asarray(factorization.rank(), dtype=jnp.int32)
    return jnp.asarray(-1, dtype=jnp.int32)


def _condition(factorization: PreparedFactorization, /) -> Array:
    """2-norm condition from singular values; NaN when the kind reports none."""
    if not factorization.capabilities.singular_values:
        return jnp.asarray(jnp.nan, dtype=jnp.float64)
    values = jnp.abs(factorization.singular_values())
    largest, smallest = jnp.max(values), jnp.min(values)
    positive = smallest > 0
    return jnp.where(positive, largest / jnp.where(positive, smallest, 1), jnp.inf)


def _evidence(
    factorization: PreparedFactorization,
    paths: tuple[tuple[BlockPath, ...], tuple[BlockPath, ...]],
    sizes: tuple[int, int],
    limit: float | None,
    statuses: Array,
    /,
) -> CondensationEvidence:
    """Pivot rank, condition, and status evidence of one (refreshed) factorization."""
    pivot_paths, retained_paths = paths
    pivot_size, retained_size = sizes
    rank = _rank(factorization)
    condition = _condition(factorization)
    full_rank = rank == pivot_size
    within = (
        jnp.asarray(True)
        if limit is None
        else jnp.isfinite(condition) & (condition <= limit)
    )
    failed = statuses != int(LinearSolveStatus.SUCCESS)
    first = jnp.where(jnp.any(failed), statuses[jnp.argmax(failed)], statuses[0])
    status = jnp.where(full_rank, first, int(LinearSolveStatus.SINGULAR))
    return CondensationEvidence(
        pivot_paths=pivot_paths,
        retained_paths=retained_paths,
        pivot_size=pivot_size,
        retained_size=retained_size,
        factorization=factorization.policy.kind,
        condition_limit=limit,
        rank=rank,
        condition_estimate=condition,
        pivot_status=jnp.asarray(status, dtype=jnp.int32),
        full_rank=full_rank,
        within_condition_limit=within,
    )


def _require_admissible_pivot(evidence: CondensationEvidence, /) -> None:
    """Host refusal of a rank-deficient or ill-conditioned nominal pivot."""
    rank = int(np.asarray(evidence.rank))
    condition = float(np.asarray(evidence.condition_estimate))
    facts = (
        f"numerical rank {rank} of pivot size {evidence.pivot_size}, condition "
        f"estimate {condition:.6e}, factorization {evidence.factorization!r}"
    )
    if rank != evidence.pivot_size:
        raise ValueError(f"Static condensation refuses a rank-deficient pivot: {facts}.")
    limit = evidence.condition_limit
    if limit is None:
        return
    if np.isnan(condition):
        raise ValueError(
            f"Static condensation cannot certify condition_limit {limit:.6e}: "
            f"factorization {evidence.factorization!r} reports no condition "
            "estimate; use kind 'svd'."
        )
    if not condition <= limit:
        raise ValueError(
            f"Static condensation refuses an ill-conditioned pivot: {facts} exceeds "
            f"condition_limit {limit:.6e}."
        )


def _derivative_capability(
    condensation_id: str, mode: QualifiedMode, /
) -> OwnerDerivativeCapability:
    match mode:
        case "mathematical":
            return OwnerDerivativeCapability(
                condensation_id,
                admitted={"arguments": DerivativeSurface.PHYSICAL_PARAMETER},
                refused=_FIXED_STRUCTURE_REFUSALS,
                route=DerivativeRoute.IMPLICIT,
                conditions=(
                    "accepted-result",
                    "prepared-geometry-fixed",
                    "solve-converged",
                ),
            )
        case "none":
            return OwnerDerivativeCapability(
                condensation_id,
                admitted={},
                refused={**_FIXED_STRUCTURE_REFUSALS, "arguments": _STOPPED_REFUSAL},
                route=DerivativeRoute.STOPPED,
                conditions=("prepared-geometry-fixed",),
            )
        case _:
            assert_never(mode)


def _policies(
    factorization: FactorizationPolicy | None,
    schur_policy: LinearSolvePolicy | None,
    materialization: MaterializationPolicy | None,
    /,
) -> tuple[FactorizationPolicy, LinearSolvePolicy, MaterializationPolicy]:
    # The default direct pivot and Schur solves differentiate through their own
    # factors, so the eliminated derivative is exact to roundoff.
    derivative_solve = LinearDerivativeSolvePolicy(route="primal-factors")
    factorization_ = (
        FactorizationPolicy("lu", derivative_solve=derivative_solve)
        if factorization is None
        else factorization
    )
    if not isinstance(factorization_, FactorizationPolicy):
        raise TypeError("factorization must be a FactorizationPolicy or None.")
    schur_ = (
        LinearSolvePolicy(
            DenseLU(),
            differentiation=DifferentiationPolicy(factorization_.differentiation.mode),
            derivative_solve=derivative_solve,
            failure=factorization_.failure,
        )
        if schur_policy is None
        else schur_policy
    )
    if not isinstance(schur_, LinearSolvePolicy):
        raise TypeError("schur_policy must be a LinearSolvePolicy or None.")
    materialization_ = (
        MaterializationPolicy() if materialization is None else materialization
    )
    if not isinstance(materialization_, MaterializationPolicy):
        raise TypeError("materialization must be a MaterializationPolicy or None.")
    return factorization_, schur_, materialization_


def condense_coupled_problem(
    prepared: PreparedCoupledProblem,
    /,
    *,
    pivot: Sequence[BlockPath],
    arguments: Mapping[str, object] | None = None,
    parameters: Mapping[str, object] | None = None,
    factorization: FactorizationPolicy | None = None,
    schur_policy: LinearSolvePolicy | None = None,
    materialization: MaterializationPolicy | None = None,
    condition_limit: float | None = None,
) -> CondensedCoupledProblem:
    """Qualify an exact static condensation of an affine coupled problem.

    ``pivot`` names the ``(owner, block)`` solve paths to eliminate; the
    retained set is their complement. The pivot block is factorized with
    ``factorization`` (default LU) at the nominal ``arguments`` and
    ``parameters`` (a value of every refresh parameter binding, as at every
    solve; the qualified factors are refreshed at each solve) and refused
    when rank deficient or, under ``condition_limit``, ill-conditioned or of
    unknown condition. ``schur_policy`` (default dense LU) must share the
    factorization's differentiation mode, ``"mathematical"`` or ``"none"``.
    Both defaults declare ``LinearDerivativeSolvePolicy(route="primal-factors")``,
    so implicit derivatives reuse the pivot and Schur factors.
    Every dense block is refused beyond ``materialization``.
    """
    if not isinstance(prepared, PreparedCoupledProblem):
        raise TypeError("prepared must be a PreparedCoupledProblem.")
    if prepared.execution != "linear":
        raise ValueError(
            "Static condensation requires an affine coupled problem; this problem "
            f"has execution {prepared.execution!r}."
        )
    if prepared.nullspace_policy is not None:
        raise ValueError(
            "Static condensation refuses a coupled problem with a declared kernel: "
            "a gauged system has no exact pivot/Schur factorization."
        )
    factorization_, schur_, materialization_ = _policies(
        factorization, schur_policy, materialization
    )
    limit = (
        None
        if condition_limit is None
        else positive_finite_float(condition_limit, "condition_limit")
    )
    mode = _require_differentiation(factorization_, schur_)
    paths = _partition(prepared, pivot)
    pivot_selection = BlockSelection(prepared.state_space, (("pivot", paths[0]),))
    retained_selection = BlockSelection(prepared.state_space, (("retained", paths[1]),))
    sizes = (pivot_selection.target.size, retained_selection.target.size)
    _require_budget(prepared, materialization_, *sizes)
    system, _ = prepared.linear_system(
        prepared.bind_arguments(arguments, parameters=parameters).arguments
    )
    prepared_factorization = factorize(
        _pivot_operator(system.operator, pivot_selection), factorization_
    )
    evidence = _evidence(
        prepared_factorization,
        paths,
        sizes,
        limit,
        jnp.asarray([int(LinearSolveStatus.SUCCESS)], dtype=jnp.int32),
    )
    _require_admissible_pivot(evidence)
    condensation_id = canonical_fingerprint(
        {
            "kind": "condensed-coupled-problem",
            "problem": prepared.problem_id,
            "pivot": [list(path) for path in paths[0]],
            "retained": [list(path) for path in paths[1]],
            "factorization": factorization_.kind,
            "differentiation": mode,
        }
    )
    return CondensedCoupledProblem(
        prepared=prepared,
        pivot=pivot_selection,
        retained=retained_selection,
        factorization=prepared_factorization,
        schur_policy=schur_,
        materialization=materialization_,
        evidence=evidence,
        derivative_capability=_derivative_capability(condensation_id, mode),
        condensation_id=condensation_id,
    )


def _columns(space: BlockSpace, matrix: Array, /) -> tuple[Array, ...]:
    """Named block value whose trailing axis indexes the columns of ``matrix``."""
    return jax.vmap(space.unflatten, in_axes=1, out_axes=-1)(matrix)


def _flat_columns(space: BlockSpace, value: tuple[Array, ...], /) -> Array:
    return jax.vmap(space.flatten, in_axes=-1, out_axes=1)(value)


def _eliminate(
    condensed: CondensedCoupledProblem,
    system_operator: AbstractLinearOperator,
    rhs: SolveState,
    factorization: PreparedFactorization,
    /,
) -> tuple[LinearSolveResult, LinearSolveResult, LinearSolveResult, SolveState]:
    """Batched elimination, Schur solve, and pivot reconstruction."""
    pivot, retained = condensed.pivot, condensed.retained
    budget = condensed.materialization
    coupling = materialize(
        select_block_operator(system_operator, pivot, retained), budget
    )
    lower = materialize(select_block_operator(system_operator, retained, pivot), budget)
    retained_block = materialize(
        select_block_operator(system_operator, retained, retained), budget
    )
    pivot_rhs = pivot.target.flatten(pivot.restrict(rhs))
    retained_rhs = retained.target.flatten(retained.restrict(rhs))
    columns = retained.target.size
    stacked = jnp.concatenate((coupling, pivot_rhs[:, None]), axis=1)
    elimination = factorization.solve(
        _columns(pivot.target, stacked), rhs_layout=RHSLayout((columns + 1,))
    )
    eliminated = _flat_columns(pivot.target, elimination.value)
    schur = solve(
        LinearSystem(
            DenseLinearOperator(
                retained_block - lower @ eliminated[:, :columns],
                source=retained.target,
                target=retained.target,
            )
        ),
        retained.target.unflatten(retained_rhs - lower @ eliminated[:, columns]),
        policy=condensed.schur_policy,
    )
    reconstruction = factorization.solve(
        pivot.target.unflatten(
            pivot_rhs - coupling @ retained.target.flatten(schur.value)
        )
    )
    state = jax.tree.map(
        jnp.add, pivot.prolong(reconstruction.value), retained.prolong(schur.value)
    )
    return elimination, schur, reconstruction, state


def _derivative_envelope(
    condensed: CondensedCoupledProblem,
    result: CondensedSolution,
    arguments: Mapping[str, object],
    /,
) -> CondensedSolution:
    capability = condensed.derivative_capability
    route = capability.derivative_contract.route
    match route:
        case DerivativeRoute.IMPLICIT:
            return guard_derivative_validity(
                result,
                result.derivative_valid,
                dependencies=dict(arguments),
                failure=condensed.factorization.policy.failure.mode,
                message=(
                    "Condensed coupled derivatives require an accepted result with "
                    "converged elimination, Schur, and reconstruction solves."
                ),
            )
        case DerivativeRoute.STOPPED:
            return refuse_derivative_dependencies(
                result,
                dict(arguments),
                message=(
                    f"condensed coupled problem {capability.owner_id} refuses "
                    f"derivatives with respect to 'arguments': {_STOPPED_REFUSAL}"
                ),
            )
        case _:
            raise ValueError(
                f"Static condensation does not qualify the {route.value} derivative route."
            )


def solve_condensed_problem(
    condensed: CondensedCoupledProblem,
    /,
    *,
    arguments: Mapping[str, object] | None = None,
    parameters: Mapping[str, object] | None = None,
    tolerance: float = 1.0e-8,
) -> CondensedSolution:
    """Solve an exact static condensation and certify the original equations.

    ``arguments`` and ``parameters`` bind as in ``solve_coupled_problem``. The
    pivot factorization is refreshed natively at the bound arguments (its
    numerical factors are never reused across arguments); every step is
    traceable under ``jax.jit``, ``jax.jvp``, and ``jax.vjp``. ``tolerance``
    is the relative acceptance tolerance of every certificate.
    """
    if not isinstance(condensed, CondensedCoupledProblem):
        raise TypeError("condensed must be a CondensedCoupledProblem.")
    tolerance_ = positive_finite_float(tolerance, "tolerance")
    prepared = condensed.prepared
    bound = prepared.bind_arguments(arguments, parameters=parameters)
    args = bound.arguments
    system, rhs = prepared.linear_system(args)
    factorization = refresh_factorization(
        condensed.factorization, _pivot_operator(system.operator, condensed.pivot)
    )
    elimination, schur, reconstruction, state = _eliminate(
        condensed, system.operator, rhs, factorization
    )
    nominal = condensed.evidence
    evidence = _evidence(
        factorization,
        (nominal.pivot_paths, nominal.retained_paths),
        (nominal.pivot_size, nominal.retained_size),
        nominal.condition_limit,
        jnp.concatenate(
            (jnp.ravel(elimination.status), jnp.ravel(reconstruction.status))
        ),
    )
    native = (
        jnp.all(elimination.successful)
        & jnp.all(schur.successful)
        & jnp.all(reconstruction.successful)
        & evidence.full_rank
        & evidence.within_condition_limit
    )
    derivative = (
        jnp.all(elimination.derivative_valid)
        & jnp.all(schur.derivative_valid)
        & jnp.all(reconstruction.derivative_valid)
    )
    solution = certify_coupled_state(
        prepared,
        state,
        bound,
        policy=condensed.schur_policy,
        linear=schur,
        nonlinear=None,
        native=native,
        derivative=derivative,
        tolerance=tolerance_,
    )
    routed = (
        condensed.derivative_capability.derivative_contract.route
        is not DerivativeRoute.STOPPED
    )
    result = CondensedSolution(
        solution=solution,
        elimination=elimination,
        schur=schur,
        reconstruction=reconstruction,
        evidence=evidence,
        derivative_capability=condensed.derivative_capability,
        accepted=solution.accepted,
        derivative_valid=solution.accepted & derivative & routed,
    )
    return _derivative_envelope(condensed, result, args)


__all__ = [
    "CondensationEvidence",
    "CondensedCoupledProblem",
    "CondensedSolution",
    "condense_coupled_problem",
    "solve_condensed_problem",
]
