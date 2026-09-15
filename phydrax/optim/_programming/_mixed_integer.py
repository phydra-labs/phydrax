#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._branch_and_bound import BranchAndBoundResult
from ._mixed_integer_policy import (
    AbstractMixedIntegerMethod,
    MixedIntegerBranchingRule,
    MixedIntegerCertification,
    MixedIntegerMethodCapabilities,
    MixedIntegerSolvePolicy,
    NativeMixedIntegerBranchAndBound,
)
from ._problem import ConicProgram, LinearProgram
from ._quadratic import ConvexProgramResult, QuadraticProgram


CanonicalMixedIntegerProgram: TypeAlias = LinearProgram | QuadraticProgram | ConicProgram
MixedIntegerProofKind: TypeAlias = Literal[
    "native-audited",
    "provider-reported",
    "exact-proof-verified",
    "none",
]


class MixedIntegerStatus(IntEnum):
    OPTIMAL = 0
    GAP_REACHED = 1
    WORK_LIMIT = 2
    INFEASIBLE = 3
    RELAXATION_FAILURE = 4
    UNBOUNDED = 5
    BACKEND_FAILURE = 6
    CERTIFICATION_FAILURE = 7


class MixedIntegerProgram(StrictModule):
    """Canonical convex program with immutable integral coordinates."""

    relaxation: CanonicalMixedIntegerProgram
    integer_indices: tuple[int, ...] = eqx.field(static=True)
    binary_indices: tuple[int, ...] = eqx.field(static=True)
    discrete_indices: tuple[int, ...] = eqx.field(static=True)
    program_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        relaxation: CanonicalMixedIntegerProgram,
        /,
        *,
        integer_indices: tuple[int, ...] = (),
        binary_indices: tuple[int, ...] = (),
        program_id: str = "bounded-mixed-integer-convex-program",
    ):
        if not isinstance(relaxation, (LinearProgram, QuadraticProgram, ConicProgram)):
            raise TypeError("relaxation must be a canonical convex program.")
        if relaxation.batch_shape:
            raise ValueError("MixedIntegerProgram requires an unbatched program.")
        integer = _indices(integer_indices, relaxation.num_variables, "integer")
        binary = _indices(binary_indices, relaxation.num_variables, "binary")
        if set(integer) & set(binary):
            raise ValueError("integer_indices and binary_indices must be disjoint.")
        discrete = tuple(sorted((*integer, *binary)))
        if not discrete:
            raise ValueError("At least one discrete variable is required.")
        lower, upper = map(
            np.asarray,
            (relaxation.lower_bounds, relaxation.upper_bounds),
        )
        selected = np.asarray(discrete, dtype=np.int64)
        lo, hi = lower[selected], upper[selected]
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError("Every discrete variable requires finite bounds.")
        if np.any(lo > hi) or np.any(lo != np.ceil(lo)) or np.any(hi != np.floor(hi)):
            raise ValueError("Discrete bounds must be consistent integer values.")
        if binary:
            binary_array = np.asarray(binary, dtype=np.int64)
            if np.any(lower[binary_array] < 0.0) or np.any(upper[binary_array] > 1.0):
                raise ValueError("Binary bounds must be contained in [0, 1].")
        dtype = np.dtype(relaxation.linear.dtype)
        limit = float(2 ** (np.finfo(dtype).nmant + 1))
        if np.any(np.abs(lo) > limit) or np.any(np.abs(hi) > limit):
            raise ValueError("Discrete bounds are not exact in the relaxation dtype.")
        identifier = str(program_id)
        if not identifier:
            raise ValueError("program_id must be non-empty.")
        self.relaxation = relaxation
        self.integer_indices = integer
        self.binary_indices = binary
        self.discrete_indices = discrete
        self.program_id = identifier
        self.structure_id = canonical_fingerprint(
            {
                "kind": "mixed-integer-program",
                "program": relaxation.structure_id,
                "integer": list(integer),
                "binary": list(binary),
                "id": identifier,
            }
        )


class MixedIntegerCandidate(StrictModule, NonTrainableState):
    """One full-primal proposal with immutable source provenance."""

    primal: Array
    reported_objective: Array | None
    source_kind: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    candidate_id: str = eqx.field(static=True)

    def __init__(
        self,
        primal: ArrayLike,
        /,
        *,
        reported_objective: ArrayLike | None = None,
        source_kind: str = "caller",
        source_id: str = "caller-supplied",
    ):
        primal_ = jnp.asarray(primal)
        if primal_.ndim != 1:
            raise ValueError("MixedIntegerCandidate.primal must be one-dimensional.")
        if not jnp.issubdtype(primal_.dtype, jnp.floating):
            raise TypeError(
                "MixedIntegerCandidate.primal must use a real floating dtype."
            )
        objective_ = (
            None
            if reported_objective is None
            else jnp.asarray(reported_objective, dtype=primal_.dtype)
        )
        if objective_ is not None:
            if objective_.shape != ():
                raise ValueError("reported_objective must be scalar.")
            objective_value = float(np.asarray(objective_))
            if not isfinite(objective_value):
                raise ValueError("reported_objective must be finite.")
        source_kind_ = str(source_kind)
        source_id_ = str(source_id)
        if not source_kind_ or not source_id_:
            raise ValueError("Candidate source_kind and source_id must be nonempty.")
        self.primal = primal_
        self.reported_objective = objective_
        self.source_kind = source_kind_
        self.source_id = source_id_
        self.candidate_id = canonical_fingerprint(
            {
                "kind": "mixed-integer-candidate",
                "source_kind": source_kind_,
                "source_id": source_id_,
                "primal": array_tree_fingerprint(primal_),
                "reported_objective": (
                    None if objective_ is None else array_tree_fingerprint(objective_)
                ),
            }
        )


class MixedIntegerCandidateAudit(StrictModule, NonTrainableState):
    """Independent canonical replay of one mixed-integer candidate."""

    primal: Array
    objective: Array
    bound_violation: Array
    integrality_violation: Array
    equality_violation: Array
    inequality_violation: Array
    cone_violation: Array
    objective_residual: Array
    finite: Array
    bounds_valid: Array
    integrality_valid: Array
    constraints_valid: Array
    objective_valid: Array
    valid: Array
    candidate_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)


class MixedIntegerCertificate(StrictModule, NonTrainableState):
    """Primal and global-bound authority of one mixed-integer execution."""

    candidate_audit: MixedIntegerCandidateAudit | None
    infeasibility_certified: Array
    global_bound_certified: Array
    search_complete: Array
    optimality_certified: Array
    provider_solved: Array
    proof_kind: MixedIntegerProofKind = eqx.field(static=True)


class MixedIntegerWork(StrictModule, NonTrainableState):
    explored_nodes: Array
    pruned_nodes: Array
    frontier_size: Array
    frontier_size_available: Array
    relaxation_solves: Array
    master_solves: Array
    fixed_discrete_solves: Array
    cuts_proposed: Array
    cuts_accepted: Array
    candidates_audited: Array
    candidates_accepted: Array


class MixedIntegerProvenance(StrictModule, NonTrainableState):
    program_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    backend: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)


class MixedIntegerResult(StrictModule):
    primal: Array
    objective: Array
    global_lower_bound: Array
    absolute_gap: Array
    relative_gap: Array
    status: Array
    certificate: MixedIntegerCertificate
    work: MixedIntegerWork
    provenance: MixedIntegerProvenance
    incumbent_relaxation: ConvexProgramResult | None
    search: BranchAndBoundResult | None

    @property
    def feasible(self) -> Array:
        audit = self.certificate.candidate_audit
        return jnp.asarray(False) if audit is None else audit.valid

    @property
    def integral(self) -> Array:
        audit = self.certificate.candidate_audit
        return jnp.asarray(False) if audit is None else audit.integrality_valid

    @property
    def certified(self) -> Array:
        return self.certificate.optimality_certified

    @property
    def successful(self) -> Array:
        return (
            (self.status == int(MixedIntegerStatus.OPTIMAL))
            & self.feasible
            & self.certified
        )

    @property
    def explored_nodes(self) -> Array:
        return self.work.explored_nodes

    @property
    def pruned_nodes(self) -> Array:
        return self.work.pruned_nodes

    @property
    def frontier_size(self) -> Array:
        return self.work.frontier_size

    @property
    def program_id(self) -> str:
        return self.provenance.program_id

    @property
    def structure_id(self) -> str:
        return self.provenance.structure_id

    @property
    def policy_id(self) -> str:
        return self.provenance.policy_id


def solve_mixed_integer_program(
    program: MixedIntegerProgram,
    policy: MixedIntegerSolvePolicy | None = None,
    /,
    *,
    candidates: tuple[MixedIntegerCandidate, ...] = (),
) -> MixedIntegerResult:
    """Plan, prepare, and solve one canonical mixed-integer program."""
    from ._mixed_integer_lifecycle import (
        prepare_mixed_integer_program,
        solve_prepared_mixed_integer_program,
    )

    prepared = prepare_mixed_integer_program(program, policy)
    return solve_prepared_mixed_integer_program(
        prepared,
        candidates=candidates,
    ).result


def _indices(values, variables, name):
    original = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in original
    ):
        raise TypeError(f"{name}_indices must contain integers.")
    resolved = tuple(int(value) for value in original)
    if len(set(resolved)) != len(resolved) or any(
        value < 0 or value >= variables for value in resolved
    ):
        raise ValueError(f"{name}_indices must be unique in-range coordinates.")
    return tuple(sorted(resolved))


__all__ = [
    "AbstractMixedIntegerMethod",
    "CanonicalMixedIntegerProgram",
    "MixedIntegerBranchingRule",
    "MixedIntegerCandidate",
    "MixedIntegerCandidateAudit",
    "MixedIntegerCertificate",
    "MixedIntegerCertification",
    "MixedIntegerMethodCapabilities",
    "MixedIntegerProgram",
    "MixedIntegerProofKind",
    "MixedIntegerProvenance",
    "MixedIntegerResult",
    "MixedIntegerSolvePolicy",
    "MixedIntegerStatus",
    "MixedIntegerWork",
    "NativeMixedIntegerBranchAndBound",
    "solve_mixed_integer_program",
]
