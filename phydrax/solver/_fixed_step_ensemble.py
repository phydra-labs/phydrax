#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import final, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import positive_integer, real_number_array
from ..execution import (
    compiled_memory_estimate,
    CompiledMemoryEstimate,
    MemoryBudgetError,
)
from ..typing import Bool, checked, Dim, Float, Int32
from ._deterministic_ensemble import (
    DeterministicEnsembleEvidence,
    DeterministicEnsembleStatus,
    DeterministicInitialCondition,
    WeightedEnsembleReducer,
)
from ._fixed_step import FixedStepProblem, FixedStepSolution, solve_fixed_step


class _MemberDim(Dim):
    pass


class _SaveDim(Dim):
    pass


@eqx.filter_jit
def _solve_members(
    problem: FixedStepProblem,
    initial_states: PyTree[Array],
    member_count: int,
    member_chunk_size: int,
    save_every: int,
    /,
) -> tuple[FixedStepSolution, Array, Array]:
    def one(state: PyTree[Array]) -> FixedStepSolution:
        member = eqx.tree_at(lambda p: p.initial_state, problem, state)
        return solve_fixed_step(member, save_every=save_every)

    padded_count = (
        (member_count + member_chunk_size - 1) // member_chunk_size
    ) * member_chunk_size
    lanes = jnp.arange(padded_count)
    lane_valid = lanes < member_count
    initial_states = jax.tree.map(
        lambda leaf: leaf[jnp.minimum(lanes, member_count - 1)], initial_states
    )
    solutions = jax.lax.map(one, initial_states, batch_size=member_chunk_size)
    solutions = eqx.tree_at(
        lambda s: (s.valid, s.successful),
        solutions,
        (solutions.valid & lane_valid[:, None], solutions.successful & lane_valid),
    )
    solutions = jax.tree.map(lambda leaf: leaf[:member_count], solutions)
    finite = jnp.all(solutions.valid, axis=-1)
    for leaf in jax.tree.leaves(solutions.states):
        finite = finite & jnp.all(jnp.isfinite(leaf).reshape((member_count, -1)), axis=1)
    status = jnp.where(
        ~solutions.successful,
        int(DeterministicEnsembleStatus.SOLVER_FAILURE),
        jnp.where(
            ~finite,
            int(DeterministicEnsembleStatus.NONFINITE_TRAJECTORY),
            int(DeterministicEnsembleStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    return solutions, finite, status


class _EnsembleExecutable(Protocol):
    compiled: jax.stages.Compiled

    def __call__(
        self,
        problem: FixedStepProblem,
        initial_states: PyTree[Array],
        member_count: int,
        member_chunk_size: int,
        save_every: int,
        /,
    ) -> tuple[FixedStepSolution, Array, Array]: ...


@final
class FixedStepEnsemblePlan(StrictModule, NonTrainableState):
    """Explicit weighted initial states sharing any fixed-step problem.

    Every array leaf begins with the member axis and otherwise exactly matches
    the problem's state shape and dtype. Initial states must be finite. Default
    weights define the empirical probability measure M^{-1} sum_i delta_{u_i};
    see Lanthaler, Mishra and Parés-Pulido (arXiv:1909.06615), Section 4.
    Chunk sizes and resource caps are execution policies, not scientific defaults.
    Fingerprinting initial values occurs once, here at the host boundary.
    """

    __strict_contract__ = True

    problem: FixedStepProblem
    initial_states: PyTree[Array]
    weights: Float[_MemberDim]
    member_count: int = eqx.field(static=True)
    member_chunk_size: int = eqx.field(static=True)
    maximum_members: int = eqx.field(static=True)
    maximum_output_bytes: int = eqx.field(static=True)
    state_bytes: int = eqx.field(static=True)
    path_ids: tuple[str, ...] = eqx.field(static=True)
    realization_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        problem: FixedStepProblem,
        initial_states: PyTree[ArrayLike],
        semantic_labels: Sequence[str],
        /,
        *,
        weights: ArrayLike | None = None,
        member_chunk_size: int = 8,
        maximum_members: int = 4096,
        maximum_output_bytes: int = 2**31,
    ) -> None:
        labels = tuple(semantic_labels)
        count = len(labels)
        chunk = positive_integer(member_chunk_size, "member_chunk_size")
        limit = positive_integer(maximum_members, "maximum_members")
        byte_limit = positive_integer(maximum_output_bytes, "maximum_output_bytes")
        if not count or count > limit:
            raise ValueError("Member count must be positive and within maximum_members.")
        states = jax.tree.map(jnp.asarray, initial_states)
        if jax.tree.structure(states) != jax.tree.structure(problem.initial_state):
            raise ValueError("Initial state tree must match the fixed-step problem.")
        leaves = jax.tree.leaves(states)
        reference = jax.tree.leaves(problem.initial_state)
        if not leaves or any(
            leaf.shape != (count, *template.shape) or leaf.dtype != template.dtype
            for leaf, template in zip(leaves, reference, strict=True)
        ):
            raise ValueError(
                "Initial states must match member count, state shape and dtype."
            )
        raw_weights = (
            np.ones((count,), dtype=np.float64)
            if weights is None
            else real_number_array(weights, "weights")
        )
        if (
            raw_weights.shape != (count,)
            or not np.isfinite(raw_weights).all()
            or np.any(raw_weights < 0.0)
            or not np.isfinite(raw_weights.sum())
            or raw_weights.sum() <= 0.0
        ):
            raise ValueError(
                "Weights require finite nonnegative values and positive mass."
            )
        paths = tuple(
            DeterministicInitialCondition(
                jax.tree.map(lambda leaf: leaf[i], states), raw_weights[i], label
            )
            for i, label in enumerate(labels)
        )
        path_ids = tuple(path.semantic_path_id for path in paths)
        if len(set(path_ids)) != count:
            raise ValueError("Semantic labels must be unique after canonicalization.")
        realization_ids = tuple(path.realization_id for path in paths)
        self.problem = problem
        self.initial_states = states
        self.weights = jnp.asarray(raw_weights)
        self.member_count = count
        self.member_chunk_size = chunk
        self.maximum_members = limit
        self.maximum_output_bytes = byte_limit
        self.state_bytes = sum(leaf.nbytes for leaf in reference)
        self.path_ids = path_ids
        self.realization_ids = realization_ids
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-step-ensemble",
                "problem": problem.problem_id,
                "paths": path_ids,
                "realizations": realization_ids,
                "member_chunk_size": chunk,
                "maximum_members": limit,
                "maximum_output_bytes": byte_limit,
            }
        )

    def prepare(self, *, save_every: int = 1) -> PreparedFixedStepEnsemble:
        """Lower and compile once, admitting retained and compiler memory."""
        return PreparedFixedStepEnsemble(self, save_every=save_every)


@final
class PreparedFixedStepEnsemble(StrictModule, NonTrainableState):
    """Admitted unpadded initial states and their reusable compiled executable.

    ``maximum_output_bytes`` bounds both retained outputs and the executable's
    compiler memory plan, including arguments, temporaries, code, and peak bytes.
    Compiler estimates exclude allocator fragmentation and unrelated live arrays.
    The executable is a process-local static artifact, not a checkpoint payload.
    A provider without compiler memory analysis is refused rather than guessed.
    """

    plan: FixedStepEnsemblePlan
    save_every: int = eqx.field(static=True)
    padded_member_count: int = eqx.field(static=True)
    estimated_output_bytes: int = eqx.field(static=True)
    required_bytes: int = eqx.field(static=True)
    compiler_memory: CompiledMemoryEstimate = eqx.field(static=True)
    executable: _EnsembleExecutable = eqx.field(static=True, repr=False)
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(self, plan: FixedStepEnsemblePlan, /, *, save_every: int = 1) -> None:
        stride = positive_integer(save_every, "save_every")
        saves = 1 + (plan.problem.step_count + stride - 1) // stride
        # Common times, per-save validity, per-member status/backend/finiteness.
        time_dtype = jnp.result_type(
            *(
                leaf.dtype
                for leaf in jax.tree.leaves(plan.problem.initial_state)
                if jnp.issubdtype(leaf.dtype, jnp.inexact)
            )
        )
        # Saved times use only these real scalars and real integer indices.
        # Validate their cast once on the host before exposing the real grid.
        time_scalars = np.asarray(
            (plan.problem.t0, plan.problem.step_size), dtype=time_dtype
        )
        if np.any(np.imag(time_scalars) != 0.0):
            raise ValueError("Fixed-step ensemble time scalars must be real.")
        time_bytes = np.dtype(time_dtype).itemsize
        retained = (
            plan.member_count * saves * (plan.state_bytes + np.dtype(np.bool_).itemsize)
            + saves * time_bytes
            + plan.member_count * (np.dtype(np.int32).itemsize + 2)
        )
        if retained > plan.maximum_output_bytes:
            raise MemoryBudgetError(
                "maximum_output_bytes", retained, plan.maximum_output_bytes
            )
        padded = (
            (plan.member_count + plan.member_chunk_size - 1) // plan.member_chunk_size
        ) * plan.member_chunk_size
        # Equinox's decorator typing preserves the function signature but omits
        # its runtime lowering interface, verified by the compiled smoke route.
        executable = _solve_members.lower(  # ty: ignore[unresolved-attribute]
            plan.problem,
            plan.initial_states,
            plan.member_count,
            plan.member_chunk_size,
            stride,
        ).compile()
        memory = compiled_memory_estimate(executable.compiled)
        if memory is None:
            raise RuntimeError("Fixed-step ensemble requires compiler memory analysis.")
        required = max(
            max(retained, memory.output_bytes)
            + memory.argument_bytes
            - memory.alias_bytes
            + memory.temporary_bytes
            + memory.generated_code_bytes,
            0 if memory.peak_bytes is None else memory.peak_bytes,
        )
        if required > plan.maximum_output_bytes:
            raise MemoryBudgetError(
                "maximum_output_bytes", required, plan.maximum_output_bytes
            )
        self.plan = plan
        self.save_every = stride
        self.padded_member_count = padded
        self.estimated_output_bytes = retained
        self.required_bytes = required
        self.compiler_memory = memory
        self.executable = executable
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fixed-step-ensemble",
                "plan": plan.plan_id,
                "save_every": stride,
                "padded_member_count": padded,
                "estimated_output_bytes": retained,
            }
        )


@final
class FixedStepEnsembleResult(StrictModule, NonTrainableState):
    """Unpadded trajectories, native member evidence and weighted reductions.

    ``times`` is the common rank-one save grid; every state leaf has shape
    ``(members, saves, *state_shape)``. ``solutions`` is one batched native
    solution, retaining each member's terminal method evidence and step metrics.
    Backend failure takes precedence over nonfinite-trajectory status.
    """

    __strict_contract__ = True

    times: Float[_SaveDim]
    states: PyTree[Array]
    valid: Bool[_MemberDim, _SaveDim]
    status: Int32[_MemberDim]
    solutions: FixedStepSolution
    reducer: WeightedEnsembleReducer
    evidence: DeterministicEnsembleEvidence
    path_ids: tuple[str, ...] = eqx.field(static=True)
    realization_ids: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    discretization_ids: tuple[str, ...] = eqx.field(static=True)

    def mean(self, /) -> PyTree[Array]:
        """Reduce successful members only, refusing zero surviving weight mass."""
        return self.reducer.mean(self.states)


@checked
def execute_fixed_step_ensemble(
    prepared: PreparedFixedStepEnsemble, /
) -> FixedStepEnsembleResult:
    """Execute the admitted executable without compilation or member Python loops."""
    plan = prepared.plan
    solutions, finite, status = prepared.executable(
        plan.problem,
        plan.initial_states,
        plan.member_count,
        plan.member_chunk_size,
        prepared.save_every,
    )
    backend = solutions.successful
    reducer = WeightedEnsembleReducer(
        plan.weights, status == int(DeterministicEnsembleStatus.SUCCESS)
    )
    evidence = DeterministicEnsembleEvidence(
        status,
        backend,
        finite,
        reducer.successful,
        reducer.effective_sample_size,
        jnp.all(reducer.valid) & reducer.successful,
    )
    result_id = canonical_fingerprint(
        {
            "kind": "fixed-step-ensemble-result",
            "prepared": prepared.prepared_id,
            "paths": plan.path_ids,
            "realizations": plan.realization_ids,
        }
    )
    return FixedStepEnsembleResult(
        jnp.real(solutions.times[0]),
        solutions.states,
        solutions.valid,
        status,
        solutions,
        reducer,
        evidence,
        plan.path_ids,
        plan.realization_ids,
        prepared.prepared_id,
        result_id,
        ()
        if plan.problem.discretization_bundle is None
        else tuple(
            record.artifact_id for record in plan.problem.discretization_bundle.records
        ),
    )


__all__ = [
    "FixedStepEnsemblePlan",
    "PreparedFixedStepEnsemble",
    "FixedStepEnsembleResult",
    "execute_fixed_step_ensemble",
]
