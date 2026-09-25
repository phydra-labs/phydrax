#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resource-admitted finite periodic determinant amplitude with twist covariance."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._sampling._proposals import SingleCoordinateProposalPayload
from ..._sampling._targets import IncrementalMarkovTarget
from ..._strict import StrictModule
from ..._trainable import ParameterOwner
from ...discretization import PeriodicCell
from ...linalg import (
    accept_low_rank_update,
    DenseLinearOperator,
    DenseLU,
    factorization_policy_from_linear_solve,
    FactorizationPolicy,
    factorize,
    LowRankDeterminantStatus,
    LowRankSolvePolicy,
    prepare_factorized_low_rank_sequence,
    prepare_low_rank_sequence,
    PreparedFactorization,
    PreparedLowRankSequence,
    propose_low_rank_update,
    row_low_rank_update,
)
from ...operators.quantum._amplitude import LogAmplitude
from ...operators.quantum._electronic_advanced import ElectronicVMCResourcePlan
from ...operators.quantum._periodic_electronic import (
    AbstractPeriodicElectronicAmplitude,
)
from ._complex_determinant import complex_determinant_mixture
from ._periodic_features import PeriodicCellFeatureResult, PeriodicCellFeatures


_COMPACT_UPDATE = 0
_FULL_REBASE = 1
_NO_LOW_RANK_STATUS = -1


class PeriodicFermiNetCache(StrictModule):
    """Fixed-shape exact local state for one periodic FermiNet walker."""

    features: PeriodicCellFeatureResult
    orbital_matrices: Array
    sequences: PreparedLowRankSequence
    determinant_log_abs: Array
    determinant_phases: Array
    determinant_nonzero: Array
    compact_eligible: Array
    low_rank_status: Array
    route: Array
    jastrow: Array
    log_abs: Array
    phase: Array
    valid: Array

    @property
    def compact_update(self) -> Array:
        return self.route == _COMPACT_UPDATE

    @property
    def rebased(self) -> Array:
        return self.route == _FULL_REBASE


def _factorize_determinant(
    matrix: Array,
    policy: FactorizationPolicy,
    /,
) -> PreparedFactorization:
    return factorize(DenseLinearOperator(matrix), policy)


def _factorization_signed_log(
    factorization: PreparedFactorization,
    /,
) -> tuple[Array, Array]:
    return (
        factorization.log_abs_determinant(),
        factorization.determinant_sign(),
    )


def _factorization_rank(
    factorization: PreparedFactorization,
    /,
) -> Array:
    return factorization.rank()


def _prepare_determinant_sequence(
    factorization: PreparedFactorization,
    compact_eligible: Array,
    capacity: int,
    policy: LowRankSolvePolicy,
    /,
) -> PreparedLowRankSequence:
    def reuse(_: None) -> PreparedLowRankSequence:
        return prepare_factorized_low_rank_sequence(
            factorization,
            capacity,
            policy,
        )

    def surrogate(_: None) -> PreparedLowRankSequence:
        size = factorization.operator.source.size
        coordinate_dtype = factorization.operator.source.flatten(
            factorization.operator.source.zeros()
        ).dtype
        identity = jnp.eye(size, dtype=coordinate_dtype)
        return prepare_low_rank_sequence(
            DenseLinearOperator(
                identity,
                operator_id=factorization.operator.operator_id,
            ),
            capacity,
            policy,
        )

    return jax.lax.cond(
        compact_eligible,
        reuse,
        surrogate,
        operand=None,
    )


def _propose_determinant_row(
    sequence: PreparedLowRankSequence,
    row_delta: Array,
    row_index: Array,
    /,
):
    update = row_low_rank_update(row_index[None], row_delta[None, :])
    return propose_low_rank_update(sequence, update)


def _accept_determinant_row(
    sequence: PreparedLowRankSequence,
    proposal,
    /,
) -> PreparedLowRankSequence:
    return accept_low_rank_update(sequence, proposal)


def _determinant_nonzero(
    determinant_logs: Array,
    determinant_phases: Array,
    /,
) -> Array:
    return (
        jnp.isfinite(determinant_logs)
        & jnp.isfinite(determinant_phases)
        & (jnp.abs(determinant_phases) > 0.0)
    )


def _determinant_mixture(
    model: PeriodicFermiNet,
    determinant_log_abs: Array,
    determinant_phases: Array,
    determinant_nonzero: Array,
    jastrow: Array,
    twist_phase: Array,
    /,
) -> tuple[Array, Array, Array]:
    reference = jnp.max(jnp.where(determinant_nonzero, determinant_log_abs, -jnp.inf))
    finite_reference = jnp.isfinite(reference)
    shifted_logs = jnp.where(
        determinant_nonzero & finite_reference,
        determinant_log_abs - reference,
        -jnp.inf,
    )
    scaled_terms = jnp.where(
        determinant_nonzero,
        model.determinant_coefficients * determinant_phases * jnp.exp(shifted_logs),
        0.0,
    )
    mixture = jnp.sum(scaled_terms)
    magnitude = jnp.abs(mixture)
    nonzero_mixture = magnitude > 0.0
    safe_magnitude = jnp.where(nonzero_mixture, magnitude, 1.0)
    finite_mixture = finite_reference & nonzero_mixture
    log_abs = jnp.where(
        finite_mixture,
        reference + jnp.log(safe_magnitude) + jnp.real(jastrow),
        -jnp.inf,
    )
    phase = jnp.where(
        finite_mixture,
        mixture / safe_magnitude * twist_phase,
        1.0 + 0.0j,
    )
    return log_abs, phase, finite_mixture & jnp.isfinite(log_abs)


def _pair_jastrow(model: PeriodicFermiNet, features: PeriodicCellFeatureResult, /):
    pair_mask = jnp.triu(
        jnp.ones((model.electron_count, model.electron_count), dtype=jnp.bool_),
        k=1,
    )
    return model.pair_jastrow_strength * jnp.sum(
        jnp.where(pair_mask, features.pair_distances, 0.0)
    )


def _cache_from_exact_state(
    model: PeriodicFermiNet,
    features: PeriodicCellFeatureResult,
    orbital_matrices: Array,
    sequences: PreparedLowRankSequence,
    determinant_logs: Array,
    determinant_phases: Array,
    compact_eligible: Array,
    /,
) -> PeriodicFermiNetCache:
    determinant_nonzero = (
        jnp.isfinite(determinant_logs)
        & jnp.isfinite(determinant_phases)
        & (jnp.abs(determinant_phases) > 0.0)
    )
    safe_logs = jnp.where(determinant_nonzero, determinant_logs, 0.0)
    safe_phases = jnp.where(determinant_nonzero, determinant_phases, 0.0)
    jastrow = _pair_jastrow(model, features)
    log_abs, phase, mixture_valid = _determinant_mixture(
        model,
        safe_logs,
        safe_phases,
        determinant_nonzero,
        jastrow,
        features.twist_phase,
    )
    valid = (
        model.resource_plan.valid
        & features.valid
        & mixture_valid
        & jnp.all(sequences.successful)
    )
    return PeriodicFermiNetCache(
        features=features,
        orbital_matrices=orbital_matrices,
        sequences=sequences,
        determinant_log_abs=safe_logs,
        determinant_phases=safe_phases,
        determinant_nonzero=determinant_nonzero,
        compact_eligible=compact_eligible,
        low_rank_status=jnp.full(
            determinant_nonzero.shape,
            _NO_LOW_RANK_STATUS,
            dtype=jnp.int32,
        ),
        route=jnp.asarray(_FULL_REBASE, dtype=jnp.int32),
        jastrow=jastrow,
        log_abs=log_abs,
        phase=phase,
        valid=valid,
    )


class PeriodicFermiNet(AbstractPeriodicElectronicAmplitude, ParameterOwner):
    """Finite physical-cell determinant amplitude with twist covariance.

    Cartesian electron coordinates are transformed by ``PeriodicCellFeatures``.
    The reciprocal basis is finite and integer-indexed, and the metric-aware
    pair stream uses a stopped minimum-image selection. This remains a bounded
    Slater/Jastrow amplitude, not a thermodynamic-limit architecture.
    """

    cell_features: PeriodicCellFeatures
    orbital_coefficients: Array
    determinant_coefficients: Array
    pair_jastrow_strength: Array
    resource_plan: ElectronicVMCResourcePlan
    electron_count: int = eqx.field(static=True)
    determinant_count: int = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    network_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        cell: PeriodicCell,
        reciprocal_modes: ArrayLike,
        orbital_coefficients: ArrayLike,
        determinant_coefficients: ArrayLike,
        /,
        *,
        twist: ArrayLike,
        pair_jastrow_strength: ArrayLike = 0.0,
        resource_plan: ElectronicVMCResourcePlan,
    ):
        features = PeriodicCellFeatures(cell, reciprocal_modes, twist=twist)
        coefficients = jnp.asarray(orbital_coefficients)
        mixing = jnp.asarray(determinant_coefficients)
        if (
            coefficients.ndim != 3
            or coefficients.shape[2] != features.reciprocal_mode_count
        ):
            raise ValueError(
                "orbital_coefficients require (determinants,electrons,modes)."
            )
        determinants, electrons = map(int, coefficients.shape[:2])
        if mixing.shape != (determinants,):
            raise ValueError(
                "determinant_coefficients must have one entry per determinant."
            )
        if not isinstance(resource_plan, ElectronicVMCResourcePlan):
            raise TypeError("resource_plan must be ElectronicVMCResourcePlan.")
        if (
            resource_plan.electron_count != electrons
            or resource_plan.determinant_count != determinants
            or resource_plan.coordinate_dimension
            != electrons * features.ambient_dimension
        ):
            raise ValueError(
                "PeriodicFermiNet counts and physical dimension must match resource_plan."
            )
        strength = jnp.asarray(pair_jastrow_strength)
        if strength.shape != () or not bool(np.isfinite(np.asarray(strength))):
            raise ValueError("pair_jastrow_strength must be a finite scalar.")
        dtype = jnp.result_type(coefficients.dtype, mixing.dtype, 1j)
        self.cell_features = features
        self.orbital_coefficients = coefficients.astype(dtype)
        self.determinant_coefficients = mixing.astype(dtype)
        self.pair_jastrow_strength = strength
        self.resource_plan = resource_plan
        self.electron_count = electrons
        self.determinant_count = determinants
        self.spatial_dimension = features.ambient_dimension
        self.boundary_id = features.boundary_id
        self.network_id = canonical_fingerprint(
            {
                "kind": "periodic-ferminet",
                "boundary": features.boundary_id,
                "modes": features.reciprocal_mode_count,
                "electrons": electrons,
                "determinants": determinants,
                "spatial_dimension": features.ambient_dimension,
            }
        )
        self.claim = "finite-physical-cell-reciprocal-determinant-amplitude"

    @property
    def cell(self) -> PeriodicCell:
        return self.cell_features.cell

    @property
    def reciprocal_modes(self) -> Array:
        return self.cell_features.reciprocal_modes

    @property
    def twist(self) -> Array:
        return self.cell_features.twist

    @property
    def configuration_shape(self) -> tuple[int, int]:
        return (self.electron_count, self.spatial_dimension)

    def _single(self, cartesian_coordinates: Array, /) -> LogAmplitude:
        features = self.cell_features(cartesian_coordinates)
        orbitals = contract(
            "im,djm->dij",
            features.reciprocal_features,
            self.orbital_coefficients,
        )
        determinant_log_abs, determinant_phase, determinant_valid = (
            complex_determinant_mixture(
                orbitals,
                self.determinant_coefficients,
            )
        )
        jastrow = _pair_jastrow(self, features)
        log_abs = determinant_log_abs + jnp.real(jastrow)
        phase = determinant_phase * features.twist_phase * jnp.exp(1j * jnp.imag(jastrow))
        valid = (
            self.resource_plan.valid
            & features.valid
            & determinant_valid
            & jnp.isfinite(log_abs)
            & jnp.isfinite(phase)
        )
        return LogAmplitude(log_abs, phase, valid=valid)

    def __call__(self, cartesian_coordinates: ArrayLike, /) -> LogAmplitude:
        """Evaluate one configuration or arbitrary fixed-shape walker batches."""
        coordinates = jnp.asarray(cartesian_coordinates)
        if (
            coordinates.ndim < 2
            or tuple(coordinates.shape[-2:]) != self.configuration_shape
        ):
            raise ValueError(
                "PeriodicFermiNet inputs must end in physical Cartesian shape "
                f"{self.configuration_shape}; got {coordinates.shape}."
            )
        if coordinates.ndim == 2:
            return self._single(coordinates)
        batch_shape = tuple(coordinates.shape[:-2])
        count = math.prod(batch_shape)
        values = jax.vmap(self._single)(
            coordinates.reshape((count,) + self.configuration_shape)
        )
        return LogAmplitude(
            values.log_abs.reshape(batch_shape),
            values.phase.reshape(batch_shape),
            valid=values.valid.reshape(batch_shape),
        )


def _initialize_periodic_ferminet_cache(
    model: PeriodicFermiNet,
    capacity: int,
    update_policy: LowRankSolvePolicy,
    determinant_policy: FactorizationPolicy,
    position: ArrayLike,
    /,
) -> tuple[Array, PeriodicFermiNetCache]:
    coordinates = jnp.asarray(position)
    if coordinates.shape != model.configuration_shape:
        raise ValueError(
            "Periodic FermiNet incremental positions must have shape "
            f"{model.configuration_shape}; got {coordinates.shape}."
        )
    features = model.cell_features(coordinates)
    orbital_matrices = contract(
        "im,djm->dij",
        features.reciprocal_features,
        model.orbital_coefficients,
    )
    factorizations = eqx.filter_vmap(
        _factorize_determinant,
        in_axes=(0, None),
    )(orbital_matrices, determinant_policy)
    determinant_logs, determinant_phases = eqx.filter_vmap(
        _factorization_signed_log,
        in_axes=eqx.if_array(0),
    )(factorizations)
    determinant_ranks = eqx.filter_vmap(
        _factorization_rank,
        in_axes=eqx.if_array(0),
    )(factorizations)
    compact_eligible = _determinant_nonzero(
        determinant_logs,
        determinant_phases,
    ) & (determinant_ranks == model.electron_count)
    sequences = jax.lax.map(
        lambda arguments: _prepare_determinant_sequence(
            arguments[0],
            arguments[1],
            capacity,
            update_policy,
        ),
        (factorizations, compact_eligible),
    )
    cache = _cache_from_exact_state(
        model,
        features,
        orbital_matrices,
        sequences,
        determinant_logs,
        determinant_phases,
        compact_eligible,
    )
    return 2.0 * cache.log_abs, cache


def _propose_periodic_ferminet_cache(
    model: PeriodicFermiNet,
    update_policy: LowRankSolvePolicy,
    determinant_policy: FactorizationPolicy,
    current_position: ArrayLike,
    cache: PeriodicFermiNetCache,
    proposed_position: ArrayLike,
    payload: SingleCoordinateProposalPayload,
    /,
) -> tuple[Array, PeriodicFermiNetCache, Array]:
    if not isinstance(cache, PeriodicFermiNetCache):
        raise TypeError("cache must be a PeriodicFermiNetCache.")
    if not isinstance(payload, SingleCoordinateProposalPayload):
        raise TypeError(
            "Periodic FermiNet incremental updates require SingleCoordinateProposalPayload."
        )
    current = jnp.asarray(current_position)
    proposed = jnp.asarray(proposed_position)
    if (
        current.shape != model.configuration_shape
        or proposed.shape != model.configuration_shape
    ):
        raise ValueError(
            f"Current and proposed periodic FermiNet positions must have shape {model.configuration_shape}."
        )
    flat_index = jnp.asarray(payload.index, dtype=jnp.int32)
    if flat_index.shape != ():
        raise ValueError("Single-coordinate proposal index must be scalar.")
    coordinate_count = model.electron_count * model.spatial_dimension
    index_valid = (flat_index >= 0) & (flat_index < coordinate_count)
    safe_index = jnp.clip(flat_index, 0, coordinate_count - 1)
    electron_index = safe_index // model.spatial_dimension
    expected = (
        current.reshape((-1,)).at[safe_index].set(proposed.reshape((-1,))[safe_index])
    )
    coordinate_residual = jnp.max(jnp.abs(expected - proposed.reshape((-1,))))

    proposed_features = model.cell_features(proposed)
    proposed_orbitals = contract(
        "im,djm->dij",
        proposed_features.reciprocal_features,
        model.orbital_coefficients,
    )
    input_valid = (
        cache.valid
        & index_valid
        & (coordinate_residual == 0.0)
        & proposed_features.valid
        & jnp.all(jnp.isfinite(proposed_orbitals))
    )
    safe_features = jax.tree_util.tree_map(
        lambda proposed_leaf, current_leaf: jnp.where(
            input_valid, proposed_leaf, current_leaf
        ),
        proposed_features,
        cache.features,
    )
    safe_orbitals = jnp.where(
        input_valid,
        proposed_orbitals,
        cache.orbital_matrices,
    )
    row_deltas = (
        safe_orbitals[:, electron_index, :] - cache.orbital_matrices[:, electron_index, :]
    )
    determinant_proposals = eqx.filter_vmap(
        _propose_determinant_row,
        in_axes=(eqx.if_array(0), 0, None),
    )(cache.sequences, row_deltas, electron_index)
    compact_valid = (
        input_valid
        & jnp.all(determinant_proposals.successful)
        & ~jnp.any(determinant_proposals.requires_rebase)
        & jnp.all(cache.compact_eligible)
    )

    def compact_candidate(_):
        sequences = eqx.filter_vmap(
            _accept_determinant_row,
            in_axes=(eqx.if_array(0), eqx.if_array(0)),
        )(cache.sequences, determinant_proposals)
        logs = cache.determinant_log_abs + determinant_proposals.log_abs
        phases = cache.determinant_phases * determinant_proposals.sign
        nonzero = cache.determinant_nonzero
        return (
            sequences,
            logs,
            phases,
            nonzero,
            cache.compact_eligible,
            jnp.asarray(_COMPACT_UPDATE, dtype=jnp.int32),
            determinant_proposals.status,
        )

    def rebased_candidate(_):
        factorizations = eqx.filter_vmap(
            _factorize_determinant,
            in_axes=(0, None),
        )(safe_orbitals, determinant_policy)
        logs, phases = eqx.filter_vmap(
            _factorization_signed_log,
            in_axes=eqx.if_array(0),
        )(factorizations)
        ranks = eqx.filter_vmap(
            _factorization_rank,
            in_axes=eqx.if_array(0),
        )(factorizations)
        nonzero = _determinant_nonzero(logs, phases)
        eligible = nonzero & (ranks == model.electron_count)
        sequences = jax.lax.map(
            lambda arguments: _prepare_determinant_sequence(
                arguments[0],
                arguments[1],
                cache.sequences.capacity,
                update_policy,
            ),
            (factorizations, eligible),
        )
        rebase_status = jnp.where(
            cache.compact_eligible,
            determinant_proposals.status,
            int(LowRankDeterminantStatus.BASE_SOLVE_FAILED),
        )
        return (
            sequences,
            jnp.where(nonzero, logs, 0.0),
            jnp.where(nonzero, phases, 0.0),
            nonzero,
            eligible,
            jnp.asarray(_FULL_REBASE, dtype=jnp.int32),
            rebase_status,
        )

    (
        sequences,
        determinant_logs,
        determinant_phases,
        determinant_nonzero,
        compact_eligible,
        route,
        low_rank_status,
    ) = jax.lax.cond(
        compact_valid,
        compact_candidate,
        rebased_candidate,
        operand=None,
    )
    jastrow = _pair_jastrow(model, safe_features)
    log_abs, phase, mixture_valid = _determinant_mixture(
        model,
        determinant_logs,
        determinant_phases,
        determinant_nonzero,
        jastrow,
        safe_features.twist_phase,
    )
    candidate_valid = (
        input_valid
        & model.resource_plan.valid
        & safe_features.valid
        & mixture_valid
        & jnp.all(sequences.successful)
    )
    candidate = PeriodicFermiNetCache(
        features=safe_features,
        orbital_matrices=safe_orbitals,
        sequences=sequences,
        determinant_log_abs=determinant_logs,
        determinant_phases=determinant_phases,
        determinant_nonzero=determinant_nonzero,
        compact_eligible=compact_eligible,
        low_rank_status=low_rank_status,
        route=route,
        jastrow=jastrow,
        log_abs=log_abs,
        phase=phase,
        valid=candidate_valid,
    )
    finite_candidate = candidate_valid & jnp.isfinite(log_abs)
    selected_candidate = _select_periodic_ferminet_cache(
        cache,
        candidate,
        finite_candidate,
    )
    log_ratio = jnp.where(
        finite_candidate,
        2.0 * (log_abs - cache.log_abs),
        0.0,
    )
    return log_ratio, selected_candidate, finite_candidate


def _select_periodic_ferminet_cache(
    current: PyTree[Any],
    proposed: PyTree[Any],
    accepted: Array,
    /,
) -> PyTree[Any]:
    return jax.tree_util.tree_map(
        lambda current_leaf, proposed_leaf: jnp.where(
            accepted, proposed_leaf, current_leaf
        ),
        current,
        proposed,
    )


def _logical_periodic_ferminet_cache(cache: PeriodicFermiNetCache, /):
    return (
        cache.features,
        cache.orbital_matrices,
        cache.determinant_log_abs,
        cache.determinant_phases,
        cache.determinant_nonzero,
        cache.compact_eligible,
        cache.jastrow,
        cache.log_abs,
        cache.phase,
        cache.valid,
    )


def _periodic_ferminet_refresh_matches(
    current: PeriodicFermiNetCache,
    refreshed: PeriodicFermiNetCache,
    /,
) -> Array:
    if not isinstance(current, PeriodicFermiNetCache) or not isinstance(
        refreshed, PeriodicFermiNetCache
    ):
        raise TypeError("Periodic FermiNet refresh caches must have matching types.")
    current_leaves = jax.tree_util.tree_leaves(_logical_periodic_ferminet_cache(current))
    refreshed_leaves = jax.tree_util.tree_leaves(
        _logical_periodic_ferminet_cache(refreshed)
    )
    comparisons = []
    for current_leaf, refreshed_leaf in zip(
        current_leaves, refreshed_leaves, strict=True
    ):
        if current_leaf.dtype == jnp.bool_ or refreshed_leaf.dtype == jnp.bool_:
            comparisons.append(jnp.all(current_leaf == refreshed_leaf))
        elif jnp.issubdtype(current_leaf.dtype, jnp.integer) or jnp.issubdtype(
            refreshed_leaf.dtype, jnp.integer
        ):
            comparisons.append(jnp.all(current_leaf == refreshed_leaf))
        else:
            comparisons.append(jnp.all(jnp.abs(current_leaf - refreshed_leaf) <= 1.0e-8))
    return (
        jnp.all(jnp.stack(comparisons))
        & jnp.all(current.sequences.successful)
        & jnp.all(refreshed.sequences.successful)
    )


def _validate_maximum_chains(maximum_chains: int, /) -> int:
    if isinstance(maximum_chains, bool) or not isinstance(maximum_chains, Integral):
        raise TypeError("maximum_chains must be an integer.")
    maximum = int(maximum_chains)
    if maximum < 1:
        raise ValueError("maximum_chains must be positive.")
    return maximum


def _target_resource_bytes(
    model: PeriodicFermiNet,
    capacity: int,
    /,
) -> tuple[int, int]:
    electrons = model.electron_count
    determinants = model.determinant_count
    modes = model.cell_features.reciprocal_mode_count
    dimension = model.spatial_dimension
    complex_itemsize = np.dtype(model.orbital_coefficients.dtype).itemsize
    real_itemsize = np.dtype(model.cell.vectors.dtype).itemsize
    sequence = complex_itemsize * (
        6 * electrons * electrons + 4 * electrons * capacity + 4 * capacity * capacity
    )
    feature_cache = (
        complex_itemsize * (determinants * electrons * electrons + electrons * modes)
        + real_itemsize
        * (electrons * electrons * (dimension + 2) + electrons * dimension)
        + np.dtype(np.int32).itemsize * electrons * electrons * dimension
    )
    cache = determinants * sequence + feature_cache
    workspace = (
        determinants
        * complex_itemsize
        * (5 * electrons * electrons + 4 * electrons * capacity + 6 * capacity * capacity)
    )
    return cache, workspace


def periodic_ferminet_incremental_target(
    model: PeriodicFermiNet,
    /,
    *,
    capacity: int,
    update_policy: LowRankSolvePolicy,
    maximum_chains: int,
    target_id: str | None = None,
    refresh_cadence: int = 32,
) -> IncrementalMarkovTarget:
    """Build an exact single-coordinate determinant-update Markov target."""
    if not isinstance(model, PeriodicFermiNet):
        raise TypeError("model must be a PeriodicFermiNet.")
    capacity_ = int(capacity)
    if capacity_ < 1:
        raise ValueError("capacity must be positive.")
    maximum_chains_ = _validate_maximum_chains(maximum_chains)
    if not isinstance(update_policy, LowRankSolvePolicy):
        raise TypeError("update_policy must be a LowRankSolvePolicy.")
    if not isinstance(update_policy.base.method, DenseLU):
        raise ValueError("Periodic FermiNet updates require a DenseLU base method.")
    if (
        update_policy.base_nonsingularity != "asserted"
        or update_policy.failure.mode != "status"
        or update_policy.base.failure.mode != "status"
    ):
        raise ValueError(
            "Periodic FermiNet updates require asserted base nonsingularity and "
            "status failure modes for both low-rank and base policies."
        )
    base_policy = update_policy.base
    if (
        base_policy.preconditioning is not None
        or base_policy.recycling is not None
        or base_policy.precision is not None
        or base_policy.require_device_binding
    ):
        raise ValueError(
            "Periodic FermiNet determinant reuse requires full-precision, "
            "unpreconditioned, unrecycled DenseLU without required device binding."
        )
    cache_bytes, workspace_bytes = _target_resource_bytes(model, capacity_)
    if maximum_chains_ * cache_bytes > update_policy.resources.max_storage_bytes:
        raise ValueError("Periodic FermiNet aggregate cache exceeds max_storage_bytes.")
    if maximum_chains_ * workspace_bytes > update_policy.resources.max_workspace_bytes:
        raise ValueError(
            "Periodic FermiNet aggregate workspace exceeds max_workspace_bytes."
        )
    determinant_policy = factorization_policy_from_linear_solve(
        update_policy.base,
        "lu",
    )
    identifier = (
        canonical_fingerprint(
            {
                "kind": "periodic-ferminet-incremental-target",
                "network": model.network_id,
                "capacity": capacity_,
                "maximum_chains": maximum_chains_,
                "update_policy": repr(update_policy),
            }
        )
        if target_id is None
        else target_id
    )
    initialize = eqx.Partial(
        _initialize_periodic_ferminet_cache,
        model,
        capacity_,
        update_policy,
        determinant_policy,
    )
    return IncrementalMarkovTarget(
        initialize=initialize,
        propose=eqx.Partial(
            _propose_periodic_ferminet_cache,
            model,
            update_policy,
            determinant_policy,
        ),
        select=_select_periodic_ferminet_cache,
        refresh=initialize,
        refresh_validate=_periodic_ferminet_refresh_matches,
        target_id=identifier,
        refresh_cadence=refresh_cadence,
        maximum_chains=maximum_chains_,
        cache_bytes_per_chain=cache_bytes,
        workspace_bytes_per_chain=workspace_bytes,
    )


__all__ = [
    "PeriodicFermiNet",
    "PeriodicFermiNetCache",
    "periodic_ferminet_incremental_target",
]
