#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._array_archive import read_array_archive, write_array_archive
from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....optim import (
    AbstractLeastSquaresMethod,
    least_squares,
    LeastSquaresResult,
    NonlinearLeastSquaresProblem,
    OptimizationTermination,
)


def _coordinate_array(value: ArrayLike, name: str, /) -> Array:
    coordinates = jnp.asarray(value)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (num_nodes, 3).")
    if coordinates.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one node.")
    if not jnp.issubdtype(coordinates.dtype, jnp.inexact):
        coordinates = coordinates.astype(float)
    if jnp.issubdtype(coordinates.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real.")
    return coordinates


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(value * value))


class ForwardContinuationResult(StrictModule):
    """Fixed-shape forward equilibrium path supplied to inverse unloading.

    ``coordinates[k]`` is the deformed geometry at load factor ``k``.
    ``equilibrium_residual_norm`` and ``stage_successful`` are explicit evidence
    from the authoritative solid-mechanics solver at every continuation station.
    """

    coordinates: Array
    equilibrium_residual_norm: Array
    stage_successful: Array

    def __init__(
        self,
        coordinates: ArrayLike,
        equilibrium_residual_norm: ArrayLike,
        stage_successful: ArrayLike,
        /,
    ):
        self.coordinates = jnp.asarray(coordinates)
        self.equilibrium_residual_norm = jnp.asarray(equilibrium_residual_norm)
        self.stage_successful = jnp.asarray(stage_successful, dtype=bool)


ForwardContinuationPath = Callable[[Array, Array, Any], ForwardContinuationResult]


class UnloadedReferenceRecoveryPlan(StrictModule, NonTrainableState):
    """Plan an inverse reference solve whose forward model follows a load path.

    ``load_factors`` must begin at zero, end at one, and increase strictly. The
    callback returns the fixed-shape accepted continuation states and solver
    evidence. Reference recovery uses the native nonlinear least-squares
    substrate, not an independent optimizer.
    """

    load_factors: Array
    residual_tolerance: float = eqx.field(static=True)
    equilibrium_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        load_factors: ArrayLike,
        /,
        *,
        residual_tolerance: float = 1.0e-8,
        equilibrium_tolerance: float = 1.0e-8,
        maximum_steps: int = 64,
        plan_id: str | None = None,
    ):
        factors = np.asarray(load_factors, dtype=float)
        residual_limit = float(residual_tolerance)
        equilibrium_limit = float(equilibrium_tolerance)
        steps = int(maximum_steps)
        if (
            factors.ndim != 1
            or factors.size < 2
            or np.any(~np.isfinite(factors))
            or factors[0] != 0.0
            or factors[-1] != 1.0
            or np.any(np.diff(factors) <= 0.0)
        ):
            raise ValueError(
                "load_factors must be a finite strictly increasing vector from 0 to 1."
            )
        if not isfinite(residual_limit) or residual_limit <= 0.0:
            raise ValueError("residual_tolerance must be positive and finite.")
        if not isfinite(equilibrium_limit) or equilibrium_limit < 0.0:
            raise ValueError("equilibrium_tolerance must be finite and nonnegative.")
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        generated = canonical_fingerprint(
            {
                "kind": "cardiac-unloaded-reference-recovery-plan",
                "load_factors": array_tree_fingerprint(factors),
                "residual_tolerance": residual_limit.hex(),
                "equilibrium_tolerance": equilibrium_limit.hex(),
                "maximum_steps": steps,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty or None.")
        self.load_factors = jnp.asarray(factors)
        self.residual_tolerance = residual_limit
        self.equilibrium_tolerance = equilibrium_limit
        self.maximum_steps = steps
        self.plan_id = identifier

    def prepare(
        self,
        loaded_coordinates: ArrayLike,
        forward_continuation_path: ForwardContinuationPath,
        /,
    ) -> PreparedUnloadedReferenceRecovery:
        """Bind the observed loaded geometry and forward continuation solver."""
        if not callable(forward_continuation_path):
            raise TypeError("forward_continuation_path must be callable.")
        loaded = _coordinate_array(loaded_coordinates, "loaded_coordinates")
        if not bool(jnp.all(jnp.isfinite(loaded))):
            raise ValueError("loaded_coordinates must be finite.")
        return PreparedUnloadedReferenceRecovery(
            self,
            loaded,
            forward_continuation_path,
        )


class UnloadedReferenceState(StrictModule):
    """Fixed-shape committed recovery state suitable for checkpointing."""

    reference_coordinates: Array
    continuation_coordinates: Array
    equilibrium_residual_norm: Array
    stage_successful: Array
    loaded_mismatch: Array
    residual_norm: Array
    iteration: Array
    valid: Array
    prepared_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)


class UnloadedReferenceCandidate(StrictModule):
    """Uncommitted inverse solution and complete forward continuation evidence."""

    reference_coordinates: Array
    continuation_coordinates: Array
    equilibrium_residual_norm: Array
    stage_successful: Array
    loaded_mismatch: Array
    residual_norm: Array
    optimizer: LeastSquaresResult
    candidate_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class UnloadedReferenceEvidence(StrictModule):
    """Fail-closed optimizer, equilibrium, zero-load, and target evidence."""

    relative_residual: Array
    zero_load_residual: Array
    maximum_equilibrium_residual: Array
    optimizer_successful: Array
    continuation_finite: Array
    all_stages_successful: Array
    equilibrium_converged: Array
    zero_load_consistent: Array
    target_matched: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


class UnloadedReferenceResult(StrictModule):
    """Committed unloaded reference with optimizer and continuation evidence."""

    state: UnloadedReferenceState
    evidence: UnloadedReferenceEvidence
    optimizer: LeastSquaresResult
    result_id: str = eqx.field(static=True)

    @property
    def reference_coordinates(self) -> Array:
        return self.state.reference_coordinates

    @property
    def successful(self) -> Array:
        return self.evidence.accepted & self.state.valid


class PreparedUnloadedReferenceRecovery(StrictModule, NonTrainableState):
    """Prepared inverse problem with fixed loaded geometry and load stations."""

    plan: UnloadedReferenceRecoveryPlan
    loaded_coordinates: Array
    forward_continuation_path: ForwardContinuationPath = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: UnloadedReferenceRecoveryPlan,
        loaded_coordinates: Array,
        forward_continuation_path: ForwardContinuationPath,
        /,
    ):
        if not isinstance(plan, UnloadedReferenceRecoveryPlan):
            raise TypeError("plan must be UnloadedReferenceRecoveryPlan.")
        loaded = _coordinate_array(loaded_coordinates, "loaded_coordinates")
        if not callable(forward_continuation_path):
            raise TypeError("forward_continuation_path must be callable.")
        self.plan = plan
        self.loaded_coordinates = loaded
        self.forward_continuation_path = forward_continuation_path
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cardiac-unloaded-reference-recovery",
                "plan_id": plan.plan_id,
                "loaded_coordinates": array_tree_fingerprint(np.asarray(loaded)),
            }
        )

    def forward_result(
        self,
        reference_coordinates: ArrayLike,
        args: Any = None,
        /,
    ) -> ForwardContinuationResult:
        reference = _coordinate_array(reference_coordinates, "reference_coordinates")
        if reference.shape != self.loaded_coordinates.shape:
            raise ValueError("Reference and loaded coordinate shapes must match.")
        result = self.forward_continuation_path(
            reference,
            self.plan.load_factors,
            args,
        )
        if not isinstance(result, ForwardContinuationResult):
            raise TypeError(
                "Forward continuation callback must return ForwardContinuationResult."
            )
        expected = (self.plan.load_factors.size,) + reference.shape
        stage_shape = (self.plan.load_factors.size,)
        if result.coordinates.shape != expected:
            raise ValueError(
                "Forward continuation coordinates must have shape "
                "(num_load_factors, num_nodes, 3)."
            )
        if (
            result.equilibrium_residual_norm.shape != stage_shape
            or result.stage_successful.shape != stage_shape
        ):
            raise ValueError(
                "Forward continuation solver evidence must have shape "
                "(num_load_factors,)."
            )
        if jnp.issubdtype(
            result.coordinates.dtype, jnp.complexfloating
        ) or jnp.issubdtype(
            result.equilibrium_residual_norm.dtype,
            jnp.complexfloating,
        ):
            raise TypeError("Forward continuation values must be real.")
        return result

    def continuation_path(
        self,
        reference_coordinates: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        return self.forward_result(reference_coordinates, args).coordinates

    def residual(
        self,
        reference_coordinates: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        return (
            self.forward_result(reference_coordinates, args).coordinates[-1]
            - self.loaded_coordinates
        )

    def initialize(
        self,
        initial_reference_coordinates: ArrayLike,
        args: Any = None,
        /,
    ) -> UnloadedReferenceState:
        reference = _coordinate_array(
            initial_reference_coordinates,
            "initial_reference_coordinates",
        )
        if reference.shape != self.loaded_coordinates.shape:
            raise ValueError("Initial reference and loaded coordinate shapes must match.")
        forward = self.forward_result(reference, args)
        mismatch = forward.coordinates[-1] - self.loaded_coordinates
        norm = _norm(mismatch)
        finite = (
            jnp.all(jnp.isfinite(reference))
            & jnp.all(jnp.isfinite(forward.coordinates))
            & jnp.all(jnp.isfinite(forward.equilibrium_residual_norm))
            & jnp.isfinite(norm)
        )
        state_id = canonical_fingerprint(
            {
                "kind": "initial-unloaded-reference-state",
                "prepared_id": self.prepared_id,
                "reference": array_tree_fingerprint(np.asarray(reference)),
            }
        )
        return UnloadedReferenceState(
            reference,
            forward.coordinates,
            forward.equilibrium_residual_norm,
            forward.stage_successful,
            mismatch,
            norm,
            jnp.asarray(0, dtype=jnp.int32),
            finite,
            self.prepared_id,
            state_id,
        )

    def propose(
        self,
        state: UnloadedReferenceState,
        /,
        *,
        method: AbstractLeastSquaresMethod | None = None,
        termination: OptimizationTermination | None = None,
        args: Any = None,
    ) -> tuple[UnloadedReferenceCandidate, UnloadedReferenceEvidence]:
        if not isinstance(state, UnloadedReferenceState):
            raise TypeError("state must be UnloadedReferenceState.")
        if state.prepared_id != self.prepared_id:
            raise ValueError("Unloaded-reference state belongs to another preparation.")
        termination_ = (
            OptimizationTermination(
                absolute_optimality=self.plan.residual_tolerance,
                relative_optimality=0.0,
                absolute_step=self.plan.residual_tolerance * 1.0e-3,
                relative_step=0.0,
                maximum_steps=self.plan.maximum_steps,
            )
            if termination is None
            else termination
        )
        if not isinstance(termination_, OptimizationTermination):
            raise TypeError("termination must be OptimizationTermination or None.")
        problem = NonlinearLeastSquaresProblem(
            lambda reference, dynamic_args: self.residual(reference, dynamic_args),
            problem_id=f"{self.prepared_id}:inverse-reference",
        )
        optimization = least_squares(
            problem,
            state.reference_coordinates,
            method=method,
            termination=termination_,
            args=args,
        )
        reference = _coordinate_array(
            optimization.parameters,
            "optimized_reference_coordinates",
        )
        forward = self.forward_result(reference, args)
        mismatch = forward.coordinates[-1] - self.loaded_coordinates
        residual_norm = _norm(mismatch)
        scale = jnp.maximum(_norm(self.loaded_coordinates), 1.0)
        relative = residual_norm / scale
        zero_load_residual = _norm(forward.coordinates[0] - reference) / jnp.maximum(
            _norm(reference), 1.0
        )
        maximum_equilibrium = jnp.max(forward.equilibrium_residual_norm)
        finite = (
            jnp.all(jnp.isfinite(reference))
            & jnp.all(jnp.isfinite(forward.coordinates))
            & jnp.all(jnp.isfinite(forward.equilibrium_residual_norm))
            & jnp.all(forward.equilibrium_residual_norm >= 0.0)
            & jnp.isfinite(residual_norm)
            & jnp.isfinite(zero_load_residual)
        )
        all_stages_successful = jnp.all(forward.stage_successful)
        equilibrium_converged = maximum_equilibrium <= self.plan.equilibrium_tolerance
        target_matched = relative <= self.plan.residual_tolerance
        zero_consistent = zero_load_residual <= self.plan.residual_tolerance
        optimizer_successful = jnp.asarray(optimization.successful)
        accepted = (
            optimizer_successful
            & finite
            & all_stages_successful
            & equilibrium_converged
            & target_matched
            & zero_consistent
        )
        candidate_id = canonical_fingerprint(
            {
                "kind": "cardiac-unloaded-reference-candidate",
                "prepared_id": self.prepared_id,
                "reference": array_tree_fingerprint(np.asarray(reference)),
            }
        )
        candidate = UnloadedReferenceCandidate(
            reference,
            forward.coordinates,
            forward.equilibrium_residual_norm,
            forward.stage_successful,
            mismatch,
            residual_norm,
            optimization,
            candidate_id,
            self.prepared_id,
        )
        evidence = UnloadedReferenceEvidence(
            relative,
            zero_load_residual,
            maximum_equilibrium,
            optimizer_successful,
            finite,
            all_stages_successful,
            equilibrium_converged,
            zero_consistent,
            target_matched,
            accepted,
            self.plan.plan_id,
        )
        return candidate, evidence

    def commit(
        self,
        state: UnloadedReferenceState,
        candidate: UnloadedReferenceCandidate,
        evidence: UnloadedReferenceEvidence,
        /,
    ) -> UnloadedReferenceState:
        """Commit only a finite, converged, continuation-consistent candidate."""
        if not isinstance(state, UnloadedReferenceState):
            raise TypeError("state must be UnloadedReferenceState.")
        if not isinstance(candidate, UnloadedReferenceCandidate):
            raise TypeError("candidate must be UnloadedReferenceCandidate.")
        if not isinstance(evidence, UnloadedReferenceEvidence):
            raise TypeError("evidence must be UnloadedReferenceEvidence.")
        if (
            state.prepared_id != self.prepared_id
            or candidate.prepared_id != self.prepared_id
        ):
            raise ValueError("State or candidate belongs to another preparation.")
        if evidence.plan_id != self.plan.plan_id:
            raise ValueError("Evidence belongs to another unloading plan.")
        if not bool(evidence.accepted):
            raise ValueError("Cannot commit an unqualified unloaded-reference candidate.")
        state_id = canonical_fingerprint(
            {
                "kind": "committed-cardiac-unloaded-reference-state",
                "candidate_id": candidate.candidate_id,
                "iteration": int(state.iteration) + 1,
            }
        )
        return UnloadedReferenceState(
            candidate.reference_coordinates,
            candidate.continuation_coordinates,
            candidate.equilibrium_residual_norm,
            candidate.stage_successful,
            candidate.loaded_mismatch,
            candidate.residual_norm,
            state.iteration + 1,
            jnp.asarray(True),
            self.prepared_id,
            state_id,
        )


def recover_unloaded_reference(
    prepared: PreparedUnloadedReferenceRecovery,
    initial_reference_coordinates: ArrayLike,
    /,
    *,
    method: AbstractLeastSquaresMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
) -> UnloadedReferenceResult:
    """Solve, independently qualify, and commit one unloaded reference geometry."""
    if not isinstance(prepared, PreparedUnloadedReferenceRecovery):
        raise TypeError("prepared must be PreparedUnloadedReferenceRecovery.")
    initial = prepared.initialize(initial_reference_coordinates, args)
    candidate, evidence = prepared.propose(
        initial,
        method=method,
        termination=termination,
        args=args,
    )
    committed = prepared.commit(initial, candidate, evidence)
    result_id = canonical_fingerprint(
        {
            "kind": "cardiac-unloaded-reference-result",
            "state_id": committed.state_id,
            "candidate_id": candidate.candidate_id,
        }
    )
    return UnloadedReferenceResult(
        committed,
        evidence,
        candidate.optimizer,
        result_id,
    )


def write_unloaded_reference_checkpoint(
    path: str | Path,
    state: UnloadedReferenceState,
    /,
) -> Path:
    """Write a pickle-free fixed-shape unloading checkpoint."""
    if not isinstance(state, UnloadedReferenceState):
        raise TypeError("state must be UnloadedReferenceState.")
    return write_array_archive(
        path,
        manifest={
            "kind": "cardiac-unloaded-reference-checkpoint",
            "prepared_id": state.prepared_id,
            "state_id": state.state_id,
            "iteration": int(state.iteration),
            "valid": bool(state.valid),
        },
        arrays={
            "reference_coordinates": state.reference_coordinates,
            "continuation_coordinates": state.continuation_coordinates,
            "equilibrium_residual_norm": state.equilibrium_residual_norm,
            "stage_successful": state.stage_successful,
            "loaded_mismatch": state.loaded_mismatch,
            "residual_norm": state.residual_norm,
        },
    )


def read_unloaded_reference_checkpoint(
    path: str | Path,
    prepared: PreparedUnloadedReferenceRecovery,
    /,
) -> UnloadedReferenceState:
    """Restore a checkpoint only against its exact prepared inverse problem."""
    if not isinstance(prepared, PreparedUnloadedReferenceRecovery):
        raise TypeError("prepared must be PreparedUnloadedReferenceRecovery.")
    manifest, arrays = read_array_archive(path)
    if manifest.get("kind") != "cardiac-unloaded-reference-checkpoint":
        raise ValueError("Archive is not an unloaded-reference checkpoint.")
    if manifest.get("prepared_id") != prepared.prepared_id:
        raise ValueError("Checkpoint belongs to another unloading preparation.")
    names = {
        "reference_coordinates",
        "continuation_coordinates",
        "equilibrium_residual_norm",
        "stage_successful",
        "loaded_mismatch",
        "residual_norm",
    }
    if set(arrays) != names:
        raise ValueError("Unloaded-reference checkpoint arrays are incomplete.")
    dtype = prepared.loaded_coordinates.dtype
    reference = jnp.asarray(arrays["reference_coordinates"], dtype=dtype)
    path_ = jnp.asarray(arrays["continuation_coordinates"], dtype=dtype)
    equilibrium = jnp.asarray(arrays["equilibrium_residual_norm"], dtype=dtype)
    successful = jnp.asarray(arrays["stage_successful"], dtype=bool)
    mismatch = jnp.asarray(arrays["loaded_mismatch"], dtype=dtype)
    residual = jnp.asarray(arrays["residual_norm"], dtype=dtype)
    stage_shape = (prepared.plan.load_factors.size,)
    expected_path = stage_shape + prepared.loaded_coordinates.shape
    if (
        reference.shape != prepared.loaded_coordinates.shape
        or mismatch.shape != prepared.loaded_coordinates.shape
        or path_.shape != expected_path
        or equilibrium.shape != stage_shape
        or successful.shape != stage_shape
        or residual.shape != ()
    ):
        raise ValueError("Checkpoint fixed shapes do not match the preparation.")
    if not bool(
        jnp.all(jnp.isfinite(reference))
        & jnp.all(jnp.isfinite(path_))
        & jnp.all(jnp.isfinite(equilibrium))
        & jnp.all(equilibrium >= 0.0)
        & jnp.all(jnp.isfinite(mismatch))
        & jnp.isfinite(residual)
    ):
        raise ValueError("Checkpoint contains invalid unloading state.")
    iteration = manifest.get("iteration")
    valid = manifest.get("valid")
    state_id = manifest.get("state_id")
    if (
        not isinstance(iteration, int)
        or iteration < 0
        or not isinstance(valid, bool)
        or not isinstance(state_id, str)
        or not state_id
    ):
        raise ValueError("Checkpoint unloading metadata is invalid.")
    return UnloadedReferenceState(
        reference,
        path_,
        equilibrium,
        successful,
        mismatch,
        residual,
        jnp.asarray(iteration, dtype=jnp.int32),
        jnp.asarray(valid),
        prepared.prepared_id,
        state_id,
    )


__all__ = [
    "ForwardContinuationPath",
    "ForwardContinuationResult",
    "PreparedUnloadedReferenceRecovery",
    "UnloadedReferenceCandidate",
    "UnloadedReferenceEvidence",
    "UnloadedReferenceRecoveryPlan",
    "UnloadedReferenceResult",
    "UnloadedReferenceState",
    "read_unloaded_reference_checkpoint",
    "recover_unloaded_reference",
    "write_unloaded_reference_checkpoint",
]
