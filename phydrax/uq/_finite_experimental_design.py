#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite-channel information design and observation-conditioned beliefs.

Scoring is numerical and JIT-compatible. Belief construction, finite selection,
experiment binding, and history updates are explicitly host-side operations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import IntEnum
from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._frozendict import frozendict
from .._strict import StrictModule
from ..optim._finite import (
    FiniteAxis,
    FiniteExhaustiveSearch,
    FiniteProductSpace,
    FiniteSearchResult,
    search_finite,
)
from ._foundation import Experiment


class FiniteDesignStatus(IntEnum):
    """Observable scoring, selection, and conditioning outcomes."""

    SUCCESS = 0
    INVALID_LIKELIHOOD = 1
    INACTIVE_DESIGN = 2
    NO_VALID_DESIGNS = 3
    STALE_IDENTITY = 4
    IMPOSSIBLE_OBSERVATION = 5


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


class FiniteExperimentalDesignProblem(StrictModule):
    """A finite parameter/design/outcome channel with a named likelihood.

    ``log_conditional_probability(parameters, design, outcomes, context)`` receives
    all parameter payloads with leading dimension P, one design payload, and an
    outcome chunk with leading dimension C. It returns a floating (P, C) array of
    log probabilities normalized over the COMPLETE ``outcomes`` support, not over
    the chunk. The scorer checks that normalization, never repairs it. Impossible
    outcomes use -inf; NaN and +inf invalidate positive-prior rows. Zero-prior rows
    are masked explicitly, including their normalization checks.

    ``likelihood_id`` is the caller's semantic identity of the callback and its
    captured model. Change it whenever that model changes. Context is an immutable
    mapping of named arrays, also included in the problem identity. Design masks
    address row-major flat indices, not potentially duplicated payload values.
    """

    parameters: FiniteProductSpace
    designs: FiniteProductSpace
    outcomes: FiniteProductSpace
    log_conditional_probability: Callable
    context: frozendict[str, Array]
    design_mask: Array | None
    likelihood_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: FiniteProductSpace,
        designs: FiniteProductSpace,
        outcomes: FiniteProductSpace,
        log_conditional_probability: Callable,
        /,
        *,
        likelihood_id: str,
        context: Mapping[str, ArrayLike] | None = None,
        design_mask: ArrayLike | None = None,
    ):
        if any(
            not isinstance(space, FiniteProductSpace)
            for space in (parameters, designs, outcomes)
        ):
            raise TypeError(
                "parameters, designs, and outcomes must be FiniteProductSpace."
            )
        if not callable(log_conditional_probability):
            raise TypeError("log_conditional_probability must be callable.")
        context_ = {} if context is None else dict(context)
        if any(not isinstance(name, str) or not name for name in context_):
            raise ValueError("context keys must be non-empty strings.")
        mask = None if design_mask is None else jnp.asarray(design_mask, dtype=bool)
        if mask is not None and mask.shape != (designs.size,):
            raise ValueError("design_mask must have one entry per design flat index.")
        self.parameters = parameters
        self.designs = designs
        self.outcomes = outcomes
        self.log_conditional_probability = log_conditional_probability
        self.context = frozendict(
            {name: jnp.asarray(value) for name, value in context_.items()}
        )
        self.design_mask = mask
        self.likelihood_id = _identifier(likelihood_id, "likelihood_id")
        self.problem_id = canonical_fingerprint(
            {
                "kind": "finite-experimental-design",
                "parameters": parameters.space_id,
                "designs": designs.space_id,
                "outcomes": outcomes.space_id,
                "likelihood": self.likelihood_id,
                "context": array_tree_fingerprint(self.context),
                "design_mask": array_tree_fingerprint(mask),
            }
        )


class FiniteDesignBelief(StrictModule):
    """Immutable finite belief whose normalized log masses are authoritative.

    Construction is host-side: finite unnormalized log masses are normalized once,
    without epsilon floors or probability-space round trips. -inf remains exact
    zero mass, whereas a finite log mass remains active even if exp(log mass)
    underflows. ``parameter_mask`` removes rows before validating and normalizing.
    History is an ordered tuple of the actual immutable Experiment records.
    """

    log_masses: Array
    history: tuple[Experiment, ...]
    parameter_space_id: str = eqx.field(static=True)
    belief_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameters: FiniteProductSpace,
        log_masses: ArrayLike,
        /,
        *,
        parameter_mask: ArrayLike | None = None,
        history: tuple[Experiment, ...] = (),
    ):
        if not isinstance(parameters, FiniteProductSpace):
            raise TypeError("parameters must be a FiniteProductSpace.")
        values = jnp.asarray(log_masses)
        if jnp.issubdtype(values.dtype, jnp.integer):
            values = values.astype(jnp.asarray(0.0).dtype)
        elif not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("log_masses must be real floating or integer values.")
        if values.shape != (parameters.size,):
            raise ValueError("log_masses must have one entry per parameter flat index.")
        if parameter_mask is not None:
            mask = jnp.asarray(parameter_mask, dtype=bool)
            if mask.shape != values.shape:
                raise ValueError("parameter_mask must match log_masses.")
            values = jnp.where(mask, values, -jnp.inf)
        host = np.asarray(values)
        if np.any(np.isnan(host) | np.isposinf(host)) or not np.any(np.isfinite(host)):
            raise ValueError(
                "Active log masses must be finite or -inf, with positive total mass."
            )
        records = tuple(history)
        if any(not isinstance(record, Experiment) for record in records):
            raise TypeError("history must contain Experiment records.")
        if len({record.experiment_id for record in records}) != len(records):
            raise ValueError("history experiment IDs must be unique.")
        centered = values - jnp.max(values)
        self.log_masses = centered - jax.scipy.special.logsumexp(centered)
        self.history = records
        self.parameter_space_id = parameters.space_id
        self.belief_id = canonical_fingerprint(
            {
                "kind": "finite-design-belief",
                "parameters": parameters.space_id,
                "log_masses": array_tree_fingerprint(self.log_masses),
                "history": [
                    {
                        "experiment_id": record.experiment_id,
                        "likelihood_id": record.likelihood_id,
                        "data_digest": record.data_digest,
                    }
                    for record in records
                ],
            }
        )

    @property
    def active_parameters(self) -> Array:
        """Exact support mask; do not infer activity from exponentiated masses."""
        return jnp.isfinite(self.log_masses)


class FiniteEIGResources(StrictModule):
    """Preflight workspace estimate; no candidate/parameter/outcome landscape."""

    parameter_count: int = eqx.field(static=True)
    candidate_count: int = eqx.field(static=True)
    outcome_count: int = eqx.field(static=True)
    candidate_batch_size: int = eqx.field(static=True)
    outcome_batch_size: int = eqx.field(static=True)
    estimated_working_bytes: int = eqx.field(static=True)
    dense_log_likelihood_bytes: int = eqx.field(static=True)


class ExpectedInformationGain(StrictModule):
    """Exact finite mutual information, in nats, with bounded evaluation chunks.

    Preflight rejects requested chunks exceeding ``maximum_bytes`` BEFORE payload
    gathers, callback tracing, or search allocation. The estimate conservatively
    reserves scorer buffers and factorized index storage; persistent input arrays,
    compiler storage, and backend overhead are not included. Callback-internal
    temporary storage is caller-owned and must be reserved using
    ``likelihood_workspace_bytes_per_candidate`` when needed. Computation uses the
    belief log-mass dtype. This budget covers batching performed by this API, not
    additional external vmap axes. No full likelihood or score landscape is kept.
    """

    candidate_batch_size: int = eqx.field(static=True)
    outcome_batch_size: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    normalization_tolerance: float = eqx.field(static=True)
    likelihood_workspace_bytes_per_candidate: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        candidate_batch_size: int = 8,
        outcome_batch_size: int = 64,
        maximum_bytes: int = 64 * 1024 * 1024,
        normalization_tolerance: float = 1e-6,
        likelihood_workspace_bytes_per_candidate: int = 0,
    ):
        self.candidate_batch_size = _positive_integer(
            candidate_batch_size, "candidate_batch_size"
        )
        self.outcome_batch_size = _positive_integer(
            outcome_batch_size, "outcome_batch_size"
        )
        self.maximum_bytes = _positive_integer(maximum_bytes, "maximum_bytes")
        tolerance = float(normalization_tolerance)
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError(
                "normalization_tolerance must be finite and between zero and one."
            )
        workspace = likelihood_workspace_bytes_per_candidate
        if isinstance(workspace, bool) or not isinstance(workspace, Integral):
            raise TypeError(
                "likelihood_workspace_bytes_per_candidate must be an integer."
            )
        if workspace < 0:
            raise ValueError(
                "likelihood_workspace_bytes_per_candidate must be nonnegative."
            )
        self.normalization_tolerance = tolerance
        self.likelihood_workspace_bytes_per_candidate = int(workspace)

    def preflight(
        self,
        problem: FiniteExperimentalDesignProblem,
        belief: FiniteDesignBelief,
        /,
        *,
        selection: bool = False,
    ) -> FiniteEIGResources:
        """Plan using shapes only; never invoke the likelihood or gather payloads."""
        if belief.parameter_space_id != problem.parameters.space_id:
            raise ValueError("Belief and problem parameter support identities differ.")
        p = problem.parameters.size
        d = problem.designs.size if selection else 1
        o = problem.outcomes.size
        b = min(d, self.candidate_batch_size)
        c = min(o, self.outcome_batch_size)
        itemsize = np.dtype(belief.log_masses.dtype).itemsize

        def point_bytes(space):
            return sum(
                int(np.prod(leaf.shape, dtype=np.int64)) * np.dtype(leaf.dtype).itemsize
                for leaf in jax.tree_util.tree_leaves(space.point_spec())
            )

        # Eight floating block buffers, two boolean blocks, row reductions and
        # observation accumulation, payload gathers, index/reducer scratch.
        per_candidate = (
            p * c * (8 * itemsize + 2)
            + p * (8 * itemsize + 8 + point_bytes(problem.parameters))
            + c * (4 * itemsize + 16 + point_bytes(problem.outcomes))
            + point_bytes(problem.designs)
            + self.likelihood_workspace_bytes_per_candidate
            + 256
        )
        index_bytes = 8 * sum(problem.designs.product_shape) if selection else 0
        working = b * per_candidate + index_bytes
        if working > self.maximum_bytes:
            raise MemoryError(
                f"Finite EIG chunks require an estimated {working} bytes, exceeding "
                f"maximum_bytes={self.maximum_bytes}; reduce candidate/outcome chunks."
            )
        return FiniteEIGResources(p, d, o, b, c, working, d * p * o * itemsize)


class FiniteDesignEvaluation(StrictModule):
    """One design's exact finite EIG and complete-row normalization evidence."""

    expected_information_gain: Array
    valid: Array
    status: Array
    design_flat_index: Array
    maximum_log_normalization_error: Array
    resources: FiniteEIGResources
    problem_id: str = eqx.field(static=True)
    belief_id: str = eqx.field(static=True)
    design_space_id: str = eqx.field(static=True)


class FiniteDesignSelection(StrictModule):
    """Stable exact finite maximizer, with the original design payload and flat ID.

    ``search_result`` is the unchanged finite reducer's evidence on a factorized
    index space. Its flat/product indices also address the original design space;
    its points are index coordinates, not design payloads. ``design_space_id``
    identifies the original space. Invalid selection has ``design=None``.
    """

    design: Any
    design_flat_index: Array
    expected_information_gain: Array
    valid: Array
    status: Array
    search_result: FiniteSearchResult
    resources: FiniteEIGResources
    problem_id: str = eqx.field(static=True)
    belief_id: str = eqx.field(static=True)
    design_space_id: str = eqx.field(static=True)


class FiniteDesignUpdate(StrictModule):
    """Observation update; every rejection returns the original belief unchanged."""

    belief: FiniteDesignBelief
    accepted: Array
    status: Array
    log_predictive_probability: Array
    experiment: Experiment


def _policy(policy: ExpectedInformationGain | None) -> ExpectedInformationGain:
    if policy is None:
        return ExpectedInformationGain()
    if not isinstance(policy, ExpectedInformationGain):
        raise TypeError("policy must be ExpectedInformationGain.")
    return policy


def _evaluate(
    problem: FiniteExperimentalDesignProblem,
    belief: FiniteDesignBelief,
    design_flat_index: ArrayLike,
    policy: ExpectedInformationGain,
    resources: FiniteEIGResources,
    observation_flat_index: int | None = None,
) -> tuple[FiniteDesignEvaluation, Array | None]:
    index = jnp.asarray(design_flat_index)
    if index.shape != () or not jnp.issubdtype(index.dtype, jnp.integer):
        raise TypeError("design_flat_index must be a scalar integer.")
    design = problem.designs.take(index)
    p = resources.parameter_count
    c = resources.outcome_batch_size
    dtype = belief.log_masses.dtype
    parameters = problem.parameters.take(jnp.arange(p, dtype=jnp.int64))
    active_rows = belief.active_parameters
    anchor_index = jnp.argmax(active_rows.astype(jnp.int32))
    parameters = jax.tree.map(
        lambda leaf: jnp.where(
            active_rows.reshape((p,) + (1,) * (leaf.ndim - 1)),
            leaf,
            jnp.broadcast_to(leaf[anchor_index], leaf.shape),
        ),
        parameters,
    )
    design_active = (
        jnp.asarray(True) if problem.design_mask is None else problem.design_mask[index]
    )
    row_totals = jnp.full((p,), -jnp.inf, dtype=dtype)
    observed = (
        None if observation_flat_index is None else jnp.full((p,), -jnp.inf, dtype=dtype)
    )

    def accumulate(chunk_index, carry):
        gain, totals, rows_valid, observation = carry
        outcome_indices = chunk_index * c + jnp.arange(c, dtype=jnp.int64)
        active_outcomes = outcome_indices < resources.outcome_count
        outcomes = problem.outcomes.take(
            jnp.minimum(outcome_indices, resources.outcome_count - 1)
        )
        raw = jnp.asarray(
            problem.log_conditional_probability(
                parameters, design, outcomes, problem.context
            )
        )
        if raw.shape != (p, c) or not jnp.issubdtype(raw.dtype, jnp.floating):
            raise TypeError(
                "log_conditional_probability must return a floating (P, outcome_chunk) array."
            )
        raw = raw.astype(dtype)
        active = active_rows[:, None] & active_outcomes[None, :]
        entries_valid = ~jnp.isnan(raw) & ~jnp.isposinf(raw)
        rows_valid = rows_valid & jnp.all(~active | entries_valid)
        # Remove invalid/inactive entries BEFORE arithmetic. This is masking, not
        # likelihood normalization; invalid positive-prior rows still reject.
        logs = jnp.where(active & entries_valid, raw, -jnp.inf)
        totals = jnp.logaddexp(totals, jax.scipy.special.logsumexp(logs, axis=1))
        joint = belief.log_masses[:, None] + logs
        predictive = jax.scipy.special.logsumexp(joint, axis=0)
        supported = jnp.isfinite(joint)
        ratio = jnp.where(supported, logs, 0) - jnp.where(
            supported, predictive[None, :], 0
        )
        gain = gain + jnp.sum(jnp.exp(joint) * ratio)
        if observation_flat_index is not None:
            selected = active_outcomes & (outcome_indices == observation_flat_index)
            observation = jnp.maximum(
                observation, jnp.max(jnp.where(selected[None, :], logs, -jnp.inf), axis=1)
            )
        return gain, totals, rows_valid, observation

    gain, totals, entries_valid, observed = jax.lax.fori_loop(
        0,
        (resources.outcome_count + c - 1) // c,
        accumulate,
        (jnp.asarray(0, dtype=dtype), row_totals, jnp.asarray(True), observed),
    )
    error = jnp.max(jnp.where(active_rows, jnp.abs(totals), 0))
    likelihood_valid = (
        entries_valid & (error <= policy.normalization_tolerance) & jnp.isfinite(gain)
    )
    valid = design_active & likelihood_valid
    status = jnp.where(
        ~design_active,
        int(FiniteDesignStatus.INACTIVE_DESIGN),
        jnp.where(
            likelihood_valid,
            int(FiniteDesignStatus.SUCCESS),
            int(FiniteDesignStatus.INVALID_LIKELIHOOD),
        ),
    ).astype(jnp.int32)
    return (
        FiniteDesignEvaluation(
            jnp.where(valid, gain, jnp.nan),
            valid,
            status,
            index,
            error,
            resources,
            problem.problem_id,
            belief.belief_id,
            problem.designs.space_id,
        ),
        observed,
    )


def evaluate_finite_experimental_design(
    problem: FiniteExperimentalDesignProblem,
    belief: FiniteDesignBelief,
    design_flat_index: ArrayLike,
    /,
    *,
    policy: ExpectedInformationGain | None = None,
) -> FiniteDesignEvaluation:
    """Score one flat-indexed design; JIT-friendly, with no host history hashing."""
    policy_ = _policy(policy)
    resources = policy_.preflight(problem, belief)
    result, _ = _evaluate(problem, belief, design_flat_index, policy_, resources)
    return result


def select_finite_experimental_design(
    problem: FiniteExperimentalDesignProblem,
    belief: FiniteDesignBelief,
    /,
    *,
    policy: ExpectedInformationGain | None = None,
) -> FiniteDesignSelection:
    """Host-orchestrated exact finite EIG maximization; lowest flat ID wins ties."""
    policy_ = _policy(policy)
    resources = policy_.preflight(problem, belief, selection=True)
    # Preserve duplicate-payload identities without materializing a D-long index
    # vector: the original Cartesian axis sizes define the same row-major IDs.
    indices = FiniteProductSpace(
        tuple(
            FiniteAxis(jnp.arange(size, dtype=jnp.int64))
            for size in problem.designs.product_shape
        )
    )

    def evaluator(coordinates):
        index = jnp.asarray(0, dtype=jnp.int64)
        for coordinate, size in zip(
            coordinates, problem.designs.product_shape, strict=True
        ):
            index = index * size + coordinate
        result, _ = _evaluate(problem, belief, index, policy_, resources)
        return -result.expected_information_gain, result.valid

    search_result = search_finite(
        evaluator,
        indices,
        search=FiniteExhaustiveSearch(resources.candidate_batch_size),
    )
    valid = search_result.valid[0] & search_result.exact
    index = search_result.flat_indices[0]
    design = problem.designs.take(index) if bool(valid) else None
    return FiniteDesignSelection(
        design,
        index,
        -search_result.scores[0],
        valid,
        jnp.where(
            valid,
            int(FiniteDesignStatus.SUCCESS),
            int(FiniteDesignStatus.NO_VALID_DESIGNS),
        ).astype(jnp.int32),
        search_result,
        resources,
        problem.problem_id,
        belief.belief_id,
        problem.designs.space_id,
    )


def _digest_array(digest: str) -> Array:
    return jnp.asarray(np.frombuffer(bytes.fromhex(digest), dtype=np.uint8).copy())


def _bound_experiment(
    problem: FiniteExperimentalDesignProblem,
    belief: FiniteDesignBelief,
    design_index: int,
    outcome_index: int,
    experiment_id: str,
) -> Experiment:
    conditions = {
        "design_flat_index": jnp.asarray(design_index, dtype=jnp.int64),
        "outcome_flat_index": jnp.asarray(outcome_index, dtype=jnp.int64),
        "problem_id": _digest_array(problem.problem_id),
        "belief_id": _digest_array(belief.belief_id),
    }
    design = problem.designs.take(design_index)
    for path, value in jax.tree_util.tree_flatten_with_path(design)[0]:
        conditions[f"design:{jax.tree_util.keystr(path)}"] = value
    for name, value in problem.context.items():
        conditions[f"context:{name}"] = value
    return Experiment(
        experiment_id,
        problem.outcomes.take(outcome_index),
        conditions=conditions,
        likelihood_id=problem.likelihood_id,
    )


def bind_finite_design_experiment(
    problem: FiniteExperimentalDesignProblem,
    belief: FiniteDesignBelief,
    selection: FiniteDesignSelection | FiniteDesignEvaluation,
    observation_flat_index: int,
    /,
    *,
    experiment_id: str,
) -> Experiment:
    """Host-bind a REAL observed outcome to its design, context, and prior history.

    No observation is sampled or invented. Outcome identity is a support flat ID;
    ``Experiment.observations`` stores its actual payload. An impossible but
    in-support observation can be bound and is explicitly rejected by updating.
    Both a successful one-design evaluation and a finite selection may be bound.
    """
    if not bool(selection.valid):
        raise ValueError("Cannot bind an invalid design evaluation or selection.")
    if (
        selection.problem_id != problem.problem_id
        or selection.belief_id != belief.belief_id
    ):
        raise ValueError("Cannot bind a stale design evaluation or selection.")
    if belief.parameter_space_id != problem.parameters.space_id:
        raise ValueError("Belief and problem parameter support identities differ.")
    if any(record.experiment_id == experiment_id for record in belief.history):
        raise ValueError("experiment_id already occurs in this belief's history.")
    raw_index = np.asarray(observation_flat_index)
    if raw_index.shape != () or raw_index.dtype.kind not in "iu":
        raise TypeError("observation_flat_index must be a scalar integer.")
    return _bound_experiment(
        problem, belief, int(selection.design_flat_index), int(raw_index), experiment_id
    )


def update_finite_design_belief(
    problem: FiniteExperimentalDesignProblem,
    belief: FiniteDesignBelief,
    experiment: Experiment,
    /,
    *,
    policy: ExpectedInformationGain | None = None,
) -> FiniteDesignUpdate:
    """Host-validate identity, condition on the observation, and append history.

    Stale/replayed/mismatched records, invalid complete likelihood rows, inactive
    designs, and zero-predictive-mass observations all return an explicit failure
    with the ORIGINAL belief (and history) unchanged. Posterior arithmetic stays
    entirely in log space, so even observations whose probability underflows can
    condition successfully when their log predictive probability is finite.
    """
    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be an Experiment.")

    def rejected(status, log_predictive=jnp.nan):
        return FiniteDesignUpdate(
            belief,
            jnp.asarray(False),
            jnp.asarray(int(status), dtype=jnp.int32),
            jnp.asarray(log_predictive, dtype=belief.log_masses.dtype),
            experiment,
        )

    if (
        belief.parameter_space_id != problem.parameters.space_id
        or experiment.likelihood_id != problem.likelihood_id
        or any(
            record.experiment_id == experiment.experiment_id for record in belief.history
        )
        or "design_flat_index" not in experiment.conditions
        or "outcome_flat_index" not in experiment.conditions
    ):
        return rejected(FiniteDesignStatus.STALE_IDENTITY)
    design_index = np.asarray(experiment.conditions["design_flat_index"])
    outcome_index = np.asarray(experiment.conditions["outcome_flat_index"])
    if (
        design_index.shape != ()
        or design_index.dtype.kind not in "iu"
        or outcome_index.shape != ()
        or outcome_index.dtype.kind not in "iu"
    ):
        return rejected(FiniteDesignStatus.STALE_IDENTITY)
    d, o = int(design_index), int(outcome_index)
    if not 0 <= d < problem.designs.size or not 0 <= o < problem.outcomes.size:
        return rejected(FiniteDesignStatus.STALE_IDENTITY)
    policy_ = _policy(policy)
    resources = policy_.preflight(problem, belief)
    expected = _bound_experiment(problem, belief, d, o, experiment.experiment_id)
    actual_digest = array_tree_fingerprint(
        {"observations": experiment.observations, "conditions": experiment.conditions}
    )["sha256"]
    if expected.data_digest != actual_digest or experiment.data_digest != actual_digest:
        return rejected(FiniteDesignStatus.STALE_IDENTITY)
    evaluation, log_likelihood = _evaluate(
        problem, belief, jnp.asarray(d, dtype=jnp.int64), policy_, resources, o
    )
    if not bool(evaluation.valid):
        return rejected(FiniteDesignStatus(int(evaluation.status)))
    if log_likelihood is None:
        raise RuntimeError("Observation-conditioned likelihood was not returned.")
    # Center the likelihood before adding the prior: a common -1e300 event
    # log probability must not erase O(1) prior log odds.
    likelihood_anchor = jnp.max(log_likelihood)
    if not bool(jnp.isfinite(likelihood_anchor)):
        return rejected(FiniteDesignStatus.IMPOSSIBLE_OBSERVATION, -jnp.inf)
    joint = belief.log_masses + (log_likelihood - likelihood_anchor)
    relative_normalizer = jax.scipy.special.logsumexp(joint)
    log_predictive = likelihood_anchor + relative_normalizer
    posterior = FiniteDesignBelief(
        problem.parameters,
        joint - relative_normalizer,
        history=(*belief.history, experiment),
    )
    return FiniteDesignUpdate(
        posterior,
        jnp.asarray(True),
        jnp.asarray(int(FiniteDesignStatus.SUCCESS), dtype=jnp.int32),
        log_predictive,
        experiment,
    )


__all__ = [
    "ExpectedInformationGain",
    "FiniteDesignBelief",
    "FiniteDesignEvaluation",
    "FiniteDesignSelection",
    "FiniteDesignStatus",
    "FiniteDesignUpdate",
    "FiniteEIGResources",
    "FiniteExperimentalDesignProblem",
    "bind_finite_design_experiment",
    "evaluate_finite_experimental_design",
    "select_finite_experimental_design",
    "update_finite_design_belief",
]
