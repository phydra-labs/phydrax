#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite continuum Pfaffian-Jastrow amplitudes and exact local targets."""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._sampling import IncrementalMarkovTarget, SingleCoordinateProposalPayload
from ..._strict import StrictModule
from ..._trainable import ParameterOwner
from ...linalg import (
    accept_low_rank_update,
    DenseLinearOperator,
    DenseLU,
    evaluate_pfaffian,
    factorization_policy_from_linear_solve,
    factorize,
    LowRankSolvePolicy,
    PfaffianPolicy,
    prepare_factorized_low_rank_sequence,
    prepare_low_rank_sequence,
    PreparedLowRankSequence,
    propose_pfaffian_update,
    skew_row_column_low_rank_update,
)
from ...operators.quantum._amplitude import LogAmplitude


_COMPACT_UPDATE = 0
_FULL_REBASE = 1
_NO_LOW_RANK_STATUS = -1


class PfaffianJastrowCache(StrictModule):
    """Native Pfaffian value and fixed-capacity solve state for one configuration."""

    configuration: Array
    pairing_matrix: Array
    correlation: Array
    log_abs: Array
    phase: Array
    sequence: PreparedLowRankSequence
    compact_eligible: Array
    locality_residual: Array
    native_residual: Array
    low_rank_status: Array
    route: Array
    native_valid: Array
    valid: Array

    @property
    def compact_update(self) -> Array:
        return self.route == _COMPACT_UPDATE

    @property
    def rebased(self) -> Array:
        return self.route == _FULL_REBASE


class PfaffianJastrowAmplitude(StrictModule, ParameterOwner):
    """Finite even-particle Pfaffian times a scalar complex Jastrow factor.

    The pairing evaluator owns construction of the complete dense pairing matrix.
    Skew validation, factorization, singularity classification, and derivatives are
    delegated unchanged to :func:`evaluate_pfaffian` under the declared policy.
    """

    pairing_evaluator: Callable[[Array], Array]
    jastrow: Callable[[Array], Array]
    policy: PfaffianPolicy
    particle_count: int = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    pairing_id: str = eqx.field(static=True)
    cusp_id: str = eqx.field(static=True)

    def __init__(
        self,
        pairing_evaluator: Callable[[Array], Array],
        jastrow: Callable[[Array], Array],
        /,
        *,
        particle_count: int,
        spatial_dimension: int,
        pairing_id: str,
        cusp_id: str,
        policy: PfaffianPolicy,
    ):
        if not callable(pairing_evaluator) or not callable(jastrow):
            raise TypeError("pairing_evaluator and jastrow must be callable.")
        if isinstance(particle_count, bool) or not isinstance(particle_count, Integral):
            raise TypeError("particle_count must be an integer.")
        count = int(particle_count)
        if count <= 0 or count % 2 != 0:
            raise ValueError("particle_count must be positive and even.")
        if isinstance(spatial_dimension, bool) or not isinstance(
            spatial_dimension, Integral
        ):
            raise TypeError("spatial_dimension must be an integer.")
        dimension = int(spatial_dimension)
        if dimension < 1:
            raise ValueError("spatial_dimension must be positive.")
        if not isinstance(pairing_id, str) or not isinstance(cusp_id, str):
            raise TypeError("pairing_id and cusp_id must be strings.")
        if (
            not pairing_id
            or pairing_id != pairing_id.strip()
            or not cusp_id
            or cusp_id != cusp_id.strip()
        ):
            raise ValueError(
                "pairing_id and cusp_id must be non-empty canonical identities."
            )
        if not isinstance(policy, PfaffianPolicy):
            raise TypeError("policy must be a PfaffianPolicy.")
        self.pairing_evaluator = pairing_evaluator
        self.jastrow = jastrow
        self.policy = policy
        self.particle_count = count
        self.spatial_dimension = dimension
        self.pairing_id = pairing_id
        self.cusp_id = cusp_id

    def _evaluate_inputs(self, configuration: ArrayLike, /) -> tuple[Array, Array, Array]:
        coordinates = jnp.asarray(configuration)
        expected_configuration_shape = (
            self.particle_count,
            self.spatial_dimension,
        )
        if coordinates.shape != expected_configuration_shape:
            raise ValueError(
                f"configuration must have shape {expected_configuration_shape}."
            )

        pairing_matrix = jnp.asarray(self.pairing_evaluator(coordinates))
        expected_shape = (self.particle_count, self.particle_count)
        if pairing_matrix.shape != expected_shape:
            raise ValueError(
                f"pairing_evaluator must return shape ({self.particle_count}, {self.particle_count})."
            )
        correlation = jnp.asarray(self.jastrow(coordinates))
        if correlation.shape != ():
            raise ValueError("jastrow must return one scalar complex log factor.")
        return coordinates, pairing_matrix, correlation

    def __call__(self, configuration: ArrayLike, /) -> LogAmplitude:
        coordinates, pairing_matrix, correlation = self._evaluate_inputs(configuration)
        pfaffian = evaluate_pfaffian(pairing_matrix, self.policy)
        log_abs = pfaffian.log_abs + jnp.real(correlation)
        phase = pfaffian.sign * jnp.exp(1j * jnp.imag(correlation))
        valid = (
            pfaffian.successful
            & ~pfaffian.singular
            & jnp.all(jnp.isfinite(coordinates))
            & jnp.isfinite(correlation)
        )
        return LogAmplitude(log_abs, phase, valid=valid)

    def initialize_cache(
        self,
        configuration: ArrayLike,
        /,
        *,
        capacity: int,
        update_policy: LowRankSolvePolicy,
    ) -> PfaffianJastrowCache:
        """Evaluate the exact amplitude and prepare its native local solve state."""
        capacity_ = _validate_capacity(capacity)
        _validate_update_policy(update_policy)
        coordinates, pairing_matrix, correlation = self._evaluate_inputs(configuration)
        sequence, compact_eligible = _prepare_pairing_sequence(
            self,
            pairing_matrix,
            capacity_,
            update_policy,
        )
        return _full_cache(
            self,
            coordinates,
            pairing_matrix,
            correlation,
            sequence,
            compact_eligible,
            locality_residual=jnp.asarray(0.0, dtype=pairing_matrix.real.dtype),
            low_rank_status=_NO_LOW_RANK_STATUS,
        )


def _validate_capacity(capacity: int, /) -> int:
    if isinstance(capacity, bool) or not isinstance(capacity, Integral):
        raise TypeError("capacity must be an integer.")
    capacity_ = int(capacity)
    if capacity_ < 1:
        raise ValueError("capacity must be positive.")
    return capacity_


def _validate_maximum_chains(maximum_chains: int, /) -> int:
    if isinstance(maximum_chains, bool) or not isinstance(maximum_chains, Integral):
        raise TypeError("maximum_chains must be an integer.")
    maximum = int(maximum_chains)
    if maximum < 1:
        raise ValueError("maximum_chains must be positive.")
    return maximum


def _target_resource_bytes(
    model: PfaffianJastrowAmplitude,
    capacity: int,
    /,
) -> tuple[int, int]:
    particle_count = model.particle_count
    spatial_dimension = model.spatial_dimension
    itemsize = np.dtype(np.complex128).itemsize
    cache = (
        itemsize
        * (
            7 * particle_count * particle_count
            + 4 * particle_count * capacity
            + 4 * capacity * capacity
        )
        + np.dtype(np.float64).itemsize * particle_count * spatial_dimension
    )
    workspace = itemsize * (
        5 * particle_count * particle_count
        + 4 * particle_count * capacity
        + 6 * capacity * capacity
    )
    return cache, workspace


def _validate_update_policy(update_policy: LowRankSolvePolicy, /) -> None:
    if not isinstance(update_policy, LowRankSolvePolicy):
        raise TypeError("update_policy must be a LowRankSolvePolicy.")
    if not isinstance(update_policy.base.method, DenseLU):
        raise ValueError("Pfaffian-Jastrow updates require a DenseLU base method.")
    if (
        update_policy.base_nonsingularity != "asserted"
        or update_policy.failure.mode != "status"
        or update_policy.base.failure.mode != "status"
    ):
        raise ValueError(
            "Pfaffian-Jastrow updates require asserted base nonsingularity and "
            "status failure modes for both low-rank and base policies."
        )
    base = update_policy.base
    if (
        base.preconditioning is not None
        or base.recycling is not None
        or base.precision is not None
        or base.require_device_binding
    ):
        raise ValueError(
            "Pfaffian-Jastrow factor reuse requires full-precision, "
            "unpreconditioned, unrecycled DenseLU without required device binding."
        )


def _structure_payload(
    model: PfaffianJastrowAmplitude,
    capacity: int,
    update_policy: LowRankSolvePolicy,
    /,
) -> dict[str, object]:
    return {
        "kind": "pfaffian-jastrow-incremental-target",
        "pairing": model.pairing_id,
        "cusp": model.cusp_id,
        "particle_count": model.particle_count,
        "spatial_dimension": model.spatial_dimension,
        "pfaffian_policy": model.policy.policy_id,
        "capacity": capacity,
        "update_policy": repr(update_policy),
    }


def _pairing_operator_id(
    model: PfaffianJastrowAmplitude,
    capacity: int,
    update_policy: LowRankSolvePolicy,
    /,
) -> str:
    return "pfaffian-jastrow-pairing:" + canonical_fingerprint(
        _structure_payload(model, capacity, update_policy)
    )


def _prepare_pairing_sequence(
    model: PfaffianJastrowAmplitude,
    pairing_matrix: Array,
    capacity: int,
    update_policy: LowRankSolvePolicy,
    /,
) -> tuple[PreparedLowRankSequence, Array]:
    operator_id = _pairing_operator_id(model, capacity, update_policy)
    factorization = factorize(
        DenseLinearOperator(pairing_matrix, operator_id=operator_id),
        factorization_policy_from_linear_solve(update_policy.base, "lu"),
    )
    compact_eligible = factorization.rank() == model.particle_count
    identity = jnp.eye(model.particle_count, dtype=pairing_matrix.dtype)

    def reuse(_: None) -> PreparedLowRankSequence:
        return prepare_factorized_low_rank_sequence(
            factorization,
            capacity,
            update_policy,
        )

    def surrogate(_: None) -> PreparedLowRankSequence:
        return prepare_low_rank_sequence(
            DenseLinearOperator(identity, operator_id=operator_id),
            capacity,
            update_policy,
        )

    sequence = jax.lax.cond(
        compact_eligible,
        reuse,
        surrogate,
        operand=None,
    )
    return sequence, compact_eligible


def _full_cache(
    model: PfaffianJastrowAmplitude,
    coordinates: Array,
    pairing_matrix: Array,
    correlation: Array,
    sequence: PreparedLowRankSequence,
    compact_eligible: Array,
    /,
    *,
    locality_residual: Array,
    low_rank_status: ArrayLike,
) -> PfaffianJastrowCache:
    pfaffian = evaluate_pfaffian(pairing_matrix, model.policy)
    log_abs = pfaffian.log_abs + jnp.real(correlation)
    phase = pfaffian.sign * jnp.exp(1j * jnp.imag(correlation))
    native_valid = (
        pfaffian.successful
        & ~pfaffian.singular
        & jnp.isfinite(pfaffian.log_abs)
        & jnp.isfinite(pfaffian.sign)
        & (jnp.abs(pfaffian.sign) > 0)
    )
    valid = (
        native_valid
        & sequence.successful
        & jnp.all(jnp.isfinite(coordinates))
        & jnp.all(jnp.isfinite(pairing_matrix))
        & jnp.isfinite(correlation)
        & jnp.isfinite(sequence.compact_condition)
    )
    return PfaffianJastrowCache(
        configuration=coordinates,
        pairing_matrix=pairing_matrix,
        correlation=correlation,
        log_abs=log_abs,
        phase=phase,
        sequence=sequence,
        compact_eligible=jnp.asarray(compact_eligible, dtype=jnp.bool_),
        locality_residual=jnp.asarray(locality_residual),
        native_residual=pfaffian.antisymmetry_residual,
        low_rank_status=jnp.asarray(low_rank_status, dtype=jnp.int32),
        route=jnp.asarray(_FULL_REBASE, dtype=jnp.int32),
        native_valid=native_valid,
        valid=valid,
    )


def _full_rebase_cache(
    model: PfaffianJastrowAmplitude,
    coordinates: Array,
    pairing_matrix: Array,
    correlation: Array,
    locality_residual: Array,
    low_rank_status: ArrayLike,
    capacity: int,
    update_policy: LowRankSolvePolicy,
    /,
) -> PfaffianJastrowCache:
    sequence, compact_eligible = _prepare_pairing_sequence(
        model,
        pairing_matrix,
        capacity,
        update_policy,
    )
    return _full_cache(
        model,
        coordinates,
        pairing_matrix,
        correlation,
        sequence,
        compact_eligible,
        locality_residual=locality_residual,
        low_rank_status=low_rank_status,
    )


def _compact_cache(
    current_cache: PfaffianJastrowCache,
    coordinates: Array,
    pairing_matrix: Array,
    correlation: Array,
    locality_residual: Array,
    proposal,
    /,
) -> PfaffianJastrowCache:
    sequence = accept_low_rank_update(
        current_cache.sequence,
        proposal.determinant,
        accepted=jnp.asarray(True),
    )
    correlation_delta = correlation - current_cache.correlation
    log_abs = current_cache.log_abs + proposal.log_abs + jnp.real(correlation_delta)
    phase = (
        current_cache.phase * proposal.sign * jnp.exp(1j * jnp.imag(correlation_delta))
    )
    native_valid = (
        proposal.successful
        & proposal.compact_pfaffian.antisymmetric
        & proposal.determinant.successful
    )
    valid = (
        current_cache.valid
        & native_valid
        & sequence.successful
        & jnp.all(jnp.isfinite(coordinates))
        & jnp.all(jnp.isfinite(pairing_matrix))
        & jnp.isfinite(correlation)
        & jnp.isfinite(log_abs)
        & jnp.isfinite(phase)
        & jnp.isfinite(sequence.compact_condition)
    )
    native_residual = jnp.maximum(
        proposal.determinant_identity_residual,
        proposal.compact_pfaffian.antisymmetry_residual,
    )
    return PfaffianJastrowCache(
        configuration=coordinates,
        pairing_matrix=pairing_matrix,
        correlation=correlation,
        log_abs=log_abs,
        phase=phase,
        sequence=sequence,
        compact_eligible=current_cache.compact_eligible,
        locality_residual=locality_residual,
        native_residual=native_residual,
        low_rank_status=proposal.status,
        route=jnp.asarray(_COMPACT_UPDATE, dtype=jnp.int32),
        native_valid=native_valid,
        valid=valid,
    )


def _initialize_target(
    model: PfaffianJastrowAmplitude,
    capacity: int,
    update_policy: LowRankSolvePolicy,
    position: ArrayLike,
    /,
) -> tuple[Array, PfaffianJastrowCache]:
    cache = model.initialize_cache(
        position,
        capacity=capacity,
        update_policy=update_policy,
    )
    return 2.0 * cache.log_abs, cache


def _maximum_absolute(value: Array, /) -> Array:
    magnitude = jnp.abs(value)
    return jnp.where(jnp.all(jnp.isfinite(magnitude)), jnp.max(magnitude), jnp.inf)


def _propose_target(
    model: PfaffianJastrowAmplitude,
    capacity: int,
    update_policy: LowRankSolvePolicy,
    compact_policy: PfaffianPolicy,
    locality_tolerance: float,
    current_position: ArrayLike,
    current_cache: PfaffianJastrowCache,
    proposed_position: ArrayLike,
    payload: SingleCoordinateProposalPayload,
    /,
) -> tuple[Array, PfaffianJastrowCache, Array]:
    if not isinstance(current_cache, PfaffianJastrowCache):
        raise TypeError("current_cache must be PfaffianJastrowCache.")
    if not isinstance(payload, SingleCoordinateProposalPayload):
        raise TypeError("payload must be SingleCoordinateProposalPayload.")
    current = jnp.asarray(current_position)
    if current.shape != current_cache.configuration.shape:
        raise ValueError("current_position and cached configuration shapes must agree.")
    coordinates, pairing_matrix, correlation = model._evaluate_inputs(proposed_position)
    if coordinates.shape != current.shape:
        raise ValueError("proposed_position must preserve the configuration shape.")

    index = jnp.asarray(payload.index, dtype=jnp.int32)
    if index.shape != ():
        raise ValueError("Single-coordinate proposal index must be scalar.")
    coordinate_count = current.size
    safe_index = jnp.clip(index, 0, coordinate_count - 1)
    in_bounds = (index >= 0) & (index < coordinate_count)
    flattened_current = jnp.reshape(current, (coordinate_count,))
    flattened_proposed = jnp.reshape(coordinates, (coordinate_count,))
    expected_coordinates = flattened_current.at[safe_index].set(
        flattened_proposed[safe_index]
    )
    coordinate_residual = _maximum_absolute(flattened_proposed - expected_coordinates)
    cache_residual = _maximum_absolute(current - current_cache.configuration)

    dimension = current.shape[1]
    particle = safe_index // dimension
    matrix_delta = pairing_matrix - current_cache.pairing_matrix
    row_delta = matrix_delta[particle]
    expected_matrix_delta = (
        jnp.zeros_like(matrix_delta)
        .at[particle, :]
        .set(row_delta)
        .at[:, particle]
        .add(-row_delta)
    )
    matrix_residual = _maximum_absolute(matrix_delta - expected_matrix_delta)
    locality_residual = jnp.maximum(
        cache_residual,
        jnp.maximum(coordinate_residual, matrix_residual),
    )
    tolerance = jnp.asarray(locality_tolerance, dtype=locality_residual.dtype)
    enough_capacity = current_cache.sequence.remaining_capacity >= 2
    compact_preconditions = (
        current_cache.valid
        & current_cache.native_valid
        & current_cache.sequence.successful
        & current_cache.compact_eligible
        & in_bounds
        & jnp.isfinite(locality_residual)
        & (locality_residual == 0.0)
        & enough_capacity
        & jnp.all(jnp.isfinite(pairing_matrix))
        & jnp.isfinite(correlation)
    )

    def full_without_proposal(_: None) -> PfaffianJastrowCache:
        return _full_rebase_cache(
            model,
            coordinates,
            pairing_matrix,
            correlation,
            locality_residual,
            _NO_LOW_RANK_STATUS,
            capacity,
            update_policy,
        )

    def attempt_compact(_: None) -> PfaffianJastrowCache:
        update = skew_row_column_low_rank_update(particle, row_delta)
        proposal = propose_pfaffian_update(
            current_cache.sequence,
            update,
            compact_policy,
            determinant_identity_tolerance=locality_tolerance,
        )
        compact_valid = (
            proposal.successful
            & ~proposal.requires_rebase
            & (proposal.compact_pfaffian.antisymmetry_residual <= tolerance)
        )

        def use_compact(_: None) -> PfaffianJastrowCache:
            return _compact_cache(
                current_cache,
                coordinates,
                pairing_matrix,
                correlation,
                locality_residual,
                proposal,
            )

        def rebase_failed_compact(_: None) -> PfaffianJastrowCache:
            return _full_rebase_cache(
                model,
                coordinates,
                pairing_matrix,
                correlation,
                locality_residual,
                proposal.status,
                capacity,
                update_policy,
            )

        return jax.lax.cond(
            compact_valid,
            use_compact,
            rebase_failed_compact,
            operand=None,
        )

    proposed_cache = jax.lax.cond(
        compact_preconditions,
        attempt_compact,
        full_without_proposal,
        operand=None,
    )
    log_ratio = 2.0 * (proposed_cache.log_abs - current_cache.log_abs)
    valid = current_cache.valid & proposed_cache.valid & jnp.isfinite(log_ratio)
    return log_ratio, proposed_cache, valid


def _select_target(current, proposed, accepted: Array, /):
    return jax.tree_util.tree_map(
        lambda current_leaf, proposed_leaf: jnp.where(
            accepted,
            proposed_leaf,
            current_leaf,
        ),
        current,
        proposed,
    )


def _refresh_validate(
    locality_tolerance: float,
    current: PfaffianJastrowCache,
    refreshed: PfaffianJastrowCache,
    /,
) -> Array:
    if not isinstance(current, PfaffianJastrowCache) or not isinstance(
        refreshed, PfaffianJastrowCache
    ):
        raise TypeError("Pfaffian-Jastrow refresh caches must have matching types.")
    tolerance = jnp.asarray(
        locality_tolerance,
        dtype=refreshed.pairing_matrix.real.dtype,
    )
    return (
        current.valid
        & refreshed.valid
        & current.native_valid
        & refreshed.native_valid
        & current.sequence.successful
        & refreshed.sequence.successful
        & (current.compact_eligible == refreshed.compact_eligible)
        & (
            _maximum_absolute(current.configuration - refreshed.configuration)
            <= tolerance
        )
        & (
            _maximum_absolute(current.pairing_matrix - refreshed.pairing_matrix)
            <= tolerance
        )
        & (jnp.abs(current.correlation - refreshed.correlation) <= tolerance)
        & (jnp.abs(current.log_abs - refreshed.log_abs) <= tolerance)
        & (jnp.abs(current.phase - refreshed.phase) <= tolerance)
    )


def pfaffian_jastrow_incremental_target(
    model: PfaffianJastrowAmplitude,
    /,
    *,
    capacity: int,
    update_policy: LowRankSolvePolicy,
    maximum_chains: int,
    target_id: str | None = None,
    refresh_cadence: int = 32,
    locality_tolerance: float = 1.0e-8,
) -> IncrementalMarkovTarget:
    """Build an exact single-coordinate Pfaffian-Jastrow Markov target."""
    if not isinstance(model, PfaffianJastrowAmplitude):
        raise TypeError("model must be PfaffianJastrowAmplitude.")
    capacity_ = _validate_capacity(capacity)
    maximum_chains_ = _validate_maximum_chains(maximum_chains)
    _validate_update_policy(update_policy)
    cache_bytes, workspace_bytes = _target_resource_bytes(model, capacity_)
    if maximum_chains_ * cache_bytes > update_policy.resources.max_storage_bytes:
        raise ValueError("Pfaffian-Jastrow aggregate cache exceeds max_storage_bytes.")
    if maximum_chains_ * workspace_bytes > update_policy.resources.max_workspace_bytes:
        raise ValueError(
            "Pfaffian-Jastrow aggregate workspace exceeds max_workspace_bytes."
        )
    if isinstance(refresh_cadence, bool) or not isinstance(refresh_cadence, Integral):
        raise TypeError("refresh_cadence must be an integer.")
    cadence = int(refresh_cadence)
    tolerance = float(locality_tolerance)
    if cadence < 1:
        raise ValueError("refresh_cadence must be positive.")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("locality_tolerance must be finite and non-negative.")
    if target_id is not None and (
        not isinstance(target_id, str) or not target_id or target_id != target_id.strip()
    ):
        raise ValueError("target_id must be None or a non-empty canonical string.")
    compact_policy = PfaffianPolicy(
        skew_mode="project",
        antisymmetry_tolerance=tolerance,
        pivot_tolerance=model.policy.pivot_tolerance,
        verify_determinant=False,
        max_dimension=model.policy.max_dimension,
        max_batch_size=model.policy.max_batch_size,
        max_storage_bytes=model.policy.max_storage_bytes,
        max_workspace_bytes=model.policy.max_workspace_bytes,
    )
    identity = target_id or canonical_fingerprint(
        {
            **_structure_payload(model, capacity_, update_policy),
            "maximum_chains": maximum_chains_,
            "refresh_cadence": cadence,
            "locality_tolerance": tolerance.hex(),
        }
    )
    initialize = eqx.Partial(_initialize_target, model, capacity_, update_policy)
    return IncrementalMarkovTarget(
        initialize=initialize,
        propose=eqx.Partial(
            _propose_target,
            model,
            capacity_,
            update_policy,
            compact_policy,
            tolerance,
        ),
        select=_select_target,
        refresh=initialize,
        refresh_validate=eqx.Partial(_refresh_validate, tolerance),
        target_id=identity,
        refresh_cadence=cadence,
        cache_tolerance=tolerance,
        maximum_chains=maximum_chains_,
        cache_bytes_per_chain=cache_bytes,
        workspace_bytes_per_chain=workspace_bytes,
    )


__all__ = [
    "PfaffianJastrowAmplitude",
    "PfaffianJastrowCache",
    "pfaffian_jastrow_incremental_target",
]
