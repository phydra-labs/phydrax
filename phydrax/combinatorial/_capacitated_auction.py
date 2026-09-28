#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Capacitated forward/reverse auction for count-constrained label assignment.

Problem
-------
Sites ``x in [N]`` of equal measure are assigned to labels ``i in [L]`` through a
fixed-width candidate relation: slot ``(x, c)`` names label ``label(x, c)`` and is
usable when it is valid and its label has a positive upper count. Given values
``a[x, c]`` (maximized) and integer bounds ``B_i <= n_i <= U_i`` on the label
counts ``n_i``, every site selects exactly one usable slot. Its LP dual is::

    min  sum_x pi_x + sum_i max(U_i p_i, B_i p_i)
    s.t. pi_x >= a[x, c] - p[label(x, c)]      for every usable slot.

The prices ``p`` are numerical dual variables of the count constraints only.
They are not physical pressures and carry no units beyond those of the values.

Certificate
-----------
For any prices, ``D = sum_x max_c (a - p) + sum_i max(U_i p_i, B_i p_i)`` is a
rigorous upper bound on the optimum and the primal value ``P`` of a feasible
assignment is a lower bound, so ``P <= OPT <= D``. When every site is within
``eps`` of its best net value (eps-CS) and sign complementary slackness holds
(``p_i > 0 => n_i = U_i`` and ``p_i < 0 => n_i = B_i``), then
``sum_i max(U_i p_i, B_i p_i) = sum_x p[label(s(x))]`` and ``D - P <= N eps``.
Integer values with ``N eps < 1`` therefore yield an exactly optimal assignment.

Forward phase (Jacobi, prices only rise)
----------------------------------------
Each unassigned site bids ``a[x, c1] - w2 + eps`` for its best slot ``c1``, where
``w1 >= w2`` are its best and second-best net values. Held sites defend their
stored accepted bids. Every label sorts its pool (held + new bids) and accepts
the top ``B_i`` unconditionally, further non-negative bids up to ``U_i``, and
sets its price to the ``U_i``-th bid when overdemanded, ``min(beta_B, 0)`` when
only negative bids compete for the lower bound, and ``max(p, 0)`` otherwise.
Accepted bids are never below the new price and prices never fall, so each held
site keeps ``a - p >= a - bid = w2 - eps``: eps-CS holds for every held site.
Counts never exceed ``U``; the phase ends when every site is held.

Reverse phase (Jacobi, prices only fall; sites never become unassigned)
-----------------------------------------------------------------------
Labels below ``B`` (floor ``-inf``) and labels below ``U`` with positive price
(floor ``0``) bid for sites of other labels. For bidder ``j`` with need ``k``,
the sites' values ``v_y = a[y, j] - pi_y`` are sorted; ``p'_j`` is
``max(floor, v_(k+1) - eps)`` (or ``v_(k) - eps`` when fewer than ``k + 1``
routes exist), capped by the current price. The top ``k`` routes with
``v > p'_j`` are offered and each site takes its offer of largest gain.
Every non-offered site satisfies ``v_y <= p'_j + eps``, profits only rise, and a
site holding several offers takes the largest net value, so eps-CS persists.
Only bidders gain sites and a bidder never gains beyond its need, so counts
stay within ``U``. At termination no label is below ``B`` and no positive-price
label is below ``U``; negative prices only arise for labels bidding for their
lower bound, so ``p < 0 => n = B``. Both sign conditions therefore hold.

Epsilon scaling and failure
---------------------------
A static strictly decreasing epsilon schedule is traversed with `lax.scan`;
each stage keeps the previous prices and the sites of the previous (or warm-start)
assignment that satisfy the stage's eps-CS; the others re-bid. A price
leaving ``max|p0| + 2 (L + 2) (C_range + eps)`` of the stage-initial prices is
declared divergence (infeasibility): feasible problems move prices by at most
``(L + 1) (C_range + eps)`` along alternating paths. A label below its lower
bound with no available route is the same divergence (its price would fall
without bound). Failures return no partial assignment.
"""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import RowRelation
from ..typing import (
    AnyDim,
    as_array,
    as_host_array,
    Bool,
    ConvertibleToArray,
    Dim,
    Float,
    HostBool,
    HostInteger,
    Int32,
    Integer,
    parse,
    Scalar,
    Scope,
    Size,
)
from ._method import (
    AbstractLinearCombinatorialMethod,
    CombinatorialPlan,
    make_combinatorial_plan,
)
from ._problem import AbstractCombinatorialSpace, LinearCombinatorialProblem
from ._selection import relative_gap
from ._types import (
    CombinatorialCertificate,
    CombinatorialCertification,
    CombinatorialFeasibility,
    CombinatorialMethodCapabilities,
    CombinatorialProvenance,
    CombinatorialResult,
    CombinatorialStatus,
)


EpsilonScale: TypeAlias = Literal["absolute", "value-range"]

_DEFAULT_SCHEDULE = (1e-1, 1e-2, 1e-3, 1e-4)
_INT32_LIMIT = 2**31 - 1


class _SiteDim(Dim, minimum=1):
    """Number of assigned sites."""


class _LabelDim(Dim, minimum=1):
    """Number of labels."""


class _CandidateDim(Dim, minimum=1):
    """Number of candidate slots per site."""


class _StageDim(Dim, minimum=1):
    """Number of epsilon-scaling stages."""


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _validated_schedule(schedule: tuple[float, ...], /) -> tuple[float, ...]:
    if not isinstance(schedule, tuple):
        raise TypeError("epsilon_schedule must be a tuple of floats.")
    if not schedule:
        raise ValueError("epsilon_schedule must be nonempty.")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in schedule):
        raise TypeError("epsilon_schedule entries must be real numbers.")
    values = tuple(float(value) for value in schedule)
    if any(not isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("epsilon_schedule entries must be finite and positive.")
    if any(later >= earlier for earlier, later in zip(values, values[1:])):
        raise ValueError("epsilon_schedule must be strictly decreasing.")
    return values


def _validated_rounds(maximum_rounds: int, stages: int, /) -> int:
    rounds = _positive_integer(maximum_rounds, "maximum_rounds")
    if 2 * stages * rounds > _INT32_LIMIT:
        raise ValueError("maximum_rounds exceeds the int32 round-counter range.")
    return rounds


def _host_int32_representable(
    values: np.ndarray, name: str, /, *, minimum: int = 0
) -> None:
    """Require integer values to fit the auction's signed-int32 storage."""

    if np.any(values < minimum) or np.any(values > _INT32_LIMIT):
        raise ValueError(
            f"{name} values must lie in the signed-int32 range "
            f"[{minimum}, {_INT32_LIMIT}] before conversion."
        )


def _device_int32_representable(
    values: Array, /, *, minimum: int = 0
) -> Array:
    """Check signed-int32 range in the input dtype without host synchronization."""

    limits = np.iinfo(np.dtype(values.dtype))
    representable = jnp.ones(values.shape, dtype=jnp.bool_)
    if limits.min < minimum:
        representable = representable & (values >= minimum)
    if limits.max > _INT32_LIMIT:
        representable = representable & (values <= _INT32_LIMIT)
    return representable


def _validated_candidates(relation: RowRelation, /) -> tuple[int, int, int]:
    """Host-validate a fixed candidate relation and return ``(N, C, L)``."""

    if not isinstance(relation, RowRelation):
        raise TypeError("candidates must be a phydrax.sparse.RowRelation.")
    if relation.case_shape != () or len(relation.route_shape) != 2:
        raise ValueError("candidate relations must have route shape (sites, width).")
    labels = as_host_array(
        relation.source_indices, HostInteger[AnyDim, AnyDim], "candidate_labels"
    )
    valid = as_host_array(relation.valid, HostBool[AnyDim, AnyDim], "valid")
    label_count = relation.source_size
    if label_count > _INT32_LIMIT:
        raise ValueError("candidate label count exceeds the signed-int32 range.")
    _host_int32_representable(np.where(valid, labels, 0), "candidate_labels")
    if np.any(valid & ((labels < 0) | (labels >= label_count))):
        raise ValueError(f"valid candidate labels must lie in [0, {label_count}).")
    width = labels.shape[1]
    keys = np.sort(
        np.where(valid, labels, label_count + np.arange(width)[None, :]), axis=1
    )
    if np.any(keys[:, 1:] == keys[:, :-1]):
        raise ValueError("a site lists the same label in more than one valid slot.")
    return labels.shape[0], width, label_count


class CapacitatedAuctionPlan(StrictModule):
    """Static sizes and epsilon-scaling schedule of one capacitated auction."""

    __strict_contract__ = True

    site_count: Size[_SiteDim] = eqx.field(static=True)
    label_count: Size[_LabelDim] = eqx.field(static=True)
    candidate_width: Size[_CandidateDim] = eqx.field(static=True)
    epsilon_schedule: tuple[float, ...] = eqx.field(static=True)
    epsilon_scale: EpsilonScale = eqx.field(static=True)
    maximum_rounds: int = eqx.field(static=True)
    maximum_routes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_count: int,
        label_count: int,
        candidate_width: int,
        /,
        *,
        epsilon_schedule: tuple[float, ...] = _DEFAULT_SCHEDULE,
        epsilon_scale: EpsilonScale = "value-range",
        maximum_rounds: int = 10_000,
        maximum_routes: int = 50_000_000,
    ) -> None:
        sites = _positive_integer(site_count, "site_count")
        labels = _positive_integer(label_count, "label_count")
        width = _positive_integer(candidate_width, "candidate_width")
        if labels > _INT32_LIMIT:
            raise ValueError("label_count exceeds the signed-int32 range.")
        schedule = _validated_schedule(epsilon_schedule)
        scale = parse(epsilon_scale, EpsilonScale, "epsilon_scale")
        rounds = _validated_rounds(maximum_rounds, len(schedule))
        routes = _positive_integer(maximum_routes, "maximum_routes")
        if sites * width > routes:
            raise ValueError(
                f"capacitated auction routes {sites * width} exceed maximum_routes {routes}."
            )
        if sites * width > _INT32_LIMIT:
            raise ValueError("capacitated auction routes exceed the int32 index range.")
        self.site_count = sites
        self.label_count = labels
        self.candidate_width = width
        self.epsilon_schedule = schedule
        self.epsilon_scale = scale
        self.maximum_rounds = rounds
        self.maximum_routes = routes
        self.plan_id = canonical_fingerprint(
            {
                "kind": "capacitated-auction-plan",
                "sites": sites,
                "labels": labels,
                "width": width,
                "epsilon_schedule": list(schedule),
                "epsilon_scale": scale,
                "maximum_rounds": rounds,
                "maximum_routes": routes,
            }
        )

    def prepare(
        self,
        candidate_labels: ConvertibleToArray,
        valid: ConvertibleToArray | None = None,
        /,
    ) -> PreparedCapacitatedAuction:
        """Host-validate a fixed candidate topology once for repeated solves."""

        scope = Scope()
        parse(self.site_count, Size[_SiteDim], "site_count", scope=scope)
        parse(self.candidate_width, Size[_CandidateDim], "candidate_width", scope=scope)
        labels = as_host_array(
            candidate_labels,
            HostInteger[_SiteDim, _CandidateDim],
            "candidate_labels",
            scope=scope,
        )
        mask = (
            np.ones(labels.shape, dtype=np.bool_)
            if valid is None
            else as_host_array(
                valid, HostBool[_SiteDim, _CandidateDim], "valid", scope=scope
            )
        )
        _host_int32_representable(
            np.where(mask, labels, 0), "valid candidate_labels"
        )
        if np.any(mask & ((labels < 0) | (labels >= self.label_count))):
            raise ValueError(
                f"valid candidate labels must lie in [0, {self.label_count})."
            )
        relation = RowRelation(
            np.where(mask, labels, 0).astype(np.int32),
            source_size=self.label_count,
            valid=mask,
        )
        return PreparedCapacitatedAuction(self, relation)

    def solve(
        self,
        candidate_labels: ArrayLike,
        valid: ArrayLike,
        values: ArrayLike,
        lower_counts: ArrayLike,
        upper_counts: ArrayLike,
        /,
        *,
        initial_prices: ArrayLike | None = None,
        initial_slots: ArrayLike | None = None,
    ) -> CapacitatedAuctionResult:
        """Solve with on-device candidates; fully traceable, no host validation.

        ``initial_slots`` warm-starts the assignment (``-1`` for unassigned sites):
        every stage keeps the sites that satisfy its epsilon-complementary
        slackness and re-bids only the others.

        Candidate labels on valid slots, count bounds, and initial slots must be
        exactly representable in signed int32. Valid labels outside ``[0, L)``,
        repeated labels within one site, invalid count bounds, and initial slots
        outside ``{-1, ..., C - 1}`` make the input INFEASIBLE, with the owning
        consistency flag false in the returned evidence.
        """

        inputs = _auction_inputs(
            self,
            candidate_labels,
            valid,
            values,
            lower_counts,
            upper_counts,
            initial_prices,
            initial_slots,
        )
        return _auction_kernel(self, inputs, CombinatorialCertification())


class PreparedCapacitatedAuction(StrictModule):
    """Capacitated auction bound to one host-validated candidate topology."""

    plan: CapacitatedAuctionPlan
    candidates: RowRelation
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: CapacitatedAuctionPlan, candidates: RowRelation, /) -> None:
        if not isinstance(plan, CapacitatedAuctionPlan):
            raise TypeError("plan must be a CapacitatedAuctionPlan.")
        sites, width, labels = _validated_candidates(candidates)
        if (sites, width, labels) != (
            plan.site_count,
            plan.candidate_width,
            plan.label_count,
        ):
            raise ValueError(
                "candidate relation sizes (sites, width, labels) must match the plan; "
                f"got {(sites, width, labels)}."
            )
        self.plan = plan
        self.candidates = candidates
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-capacitated-auction",
                "plan_id": plan.plan_id,
                "candidates": array_tree_fingerprint(
                    (candidates.source_indices, candidates.valid)
                ),
            }
        )

    def solve(
        self,
        values: ArrayLike,
        lower_counts: ArrayLike,
        upper_counts: ArrayLike,
        /,
        *,
        initial_prices: ArrayLike | None = None,
        initial_slots: ArrayLike | None = None,
    ) -> CapacitatedAuctionResult:
        """Solve on the prepared topology; fully traceable (jit/scan-safe)."""

        return self.plan.solve(
            self.candidates.source_indices,
            self.candidates.valid,
            values,
            lower_counts,
            upper_counts,
            initial_prices=initial_prices,
            initial_slots=initial_slots,
        )


class CapacitatedAuctionEvidence(StrictModule, NonTrainableState):
    """Input checks, termination, and duality-gap certificate of one auction."""

    __strict_contract__ = True

    candidates_consistent: Bool[Scalar]
    bounds_consistent: Bool[Scalar]
    warm_start_consistent: Bool[Scalar]
    finite_values: Bool[Scalar]
    feasible: Bool[Scalar]
    assigned_count: Int32[Scalar]
    capacity_residual: Int32[Scalar]
    rounds: Int32[Scalar]
    stage_rounds: Int32[_StageDim]
    epsilon: Float[Scalar]
    epsilon_cs_violation: Float[Scalar]
    sign_cs_violation: Float[Scalar]
    primal_value: Float[Scalar]
    dual_value: Float[Scalar]
    duality_gap: Float[Scalar]
    gap_bound: Float[Scalar]
    price_divergence: Bool[Scalar]
    round_limit_reached: Bool[Scalar]


class CapacitatedAuctionResult(StrictModule, NonTrainableState):
    """Assignment, label counts, dual prices, status, and evidence.

    On failure ``labels``/``slots`` are ``-1``, ``counts`` are zero, and
    ``prices`` are NaN: no partial assignment is ever returned.
    """

    __strict_contract__ = True

    labels: Int32[_SiteDim]
    slots: Int32[_SiteDim]
    counts: Int32[_LabelDim]
    prices: Float[_LabelDim]
    status: Int32[Scalar]
    evidence: CapacitatedAuctionEvidence

    @property
    def success(self) -> Array:
        """Return whether a certified (eps-)optimal feasible assignment exists."""

        accepted = (self.status == int(CombinatorialStatus.OPTIMAL)) | (
            self.status == int(CombinatorialStatus.FEASIBLE)
        )
        return accepted & self.evidence.feasible


class _AuctionInputs(NamedTuple):
    labels: Array
    valid: Array
    values: Array
    lower: Array
    upper: Array
    prices: Array
    slots: Array
    labels_representable: Array
    bounds_representable: Array
    slots_representable: Array


class _Instance(NamedTuple):
    """Sanitized device instance; unusable slots never enter the auction."""

    labels: Array
    usable: Array
    values: Array
    lower: Array
    upper: Array
    raw_upper: Array
    initial_slots: Array
    value_range: Array
    candidates_consistent: Array
    finite_values: Array
    bounds_consistent: Array
    warm_start_consistent: Array


class _ForwardState(NamedTuple):
    slots: Array
    bids: Array
    prices: Array
    rounds: Array
    diverged: Array


class _ReverseState(NamedTuple):
    slots: Array
    prices: Array
    rounds: Array
    diverged: Array
    pending: Array


class _StageCarry(NamedTuple):
    slots: Array
    prices: Array
    diverged: Array
    capped: Array


def _auction_inputs(
    plan: CapacitatedAuctionPlan,
    candidate_labels: ArrayLike,
    valid: ArrayLike,
    values: ArrayLike,
    lower_counts: ArrayLike,
    upper_counts: ArrayLike,
    initial_prices: ArrayLike | None,
    initial_slots: ArrayLike | None,
    /,
) -> _AuctionInputs:
    scope = Scope()
    parse(plan.site_count, Size[_SiteDim], "site_count", scope=scope)
    parse(plan.candidate_width, Size[_CandidateDim], "candidate_width", scope=scope)
    parse(plan.label_count, Size[_LabelDim], "label_count", scope=scope)
    labels = as_array(
        candidate_labels,
        Integer[_SiteDim, _CandidateDim],
        "candidate_labels",
        scope=scope,
    )
    mask = as_array(valid, Bool[_SiteDim, _CandidateDim], "valid", scope=scope)
    value_array = as_array(values, Float[_SiteDim, _CandidateDim], "values", scope=scope)
    lower = as_array(lower_counts, Integer[_LabelDim], "lower_counts", scope=scope)
    upper = as_array(upper_counts, Integer[_LabelDim], "upper_counts", scope=scope)
    prices = (
        jnp.zeros((plan.label_count,), dtype=value_array.dtype)
        if initial_prices is None
        else as_array(initial_prices, Float[_LabelDim], "initial_prices", scope=scope)
    )
    slots = (
        jnp.full((plan.site_count,), -1, dtype=jnp.int32)
        if initial_slots is None
        else as_array(initial_slots, Integer[_SiteDim], "initial_slots", scope=scope)
    )
    labels_representable = jnp.all(
        ~mask | _device_int32_representable(labels, minimum=0)
    )
    bounds_representable = jnp.all(
        _device_int32_representable(lower, minimum=0)
        & _device_int32_representable(upper, minimum=0)
    )
    slots_representable = jnp.all(
        _device_int32_representable(slots, minimum=-1)
    )
    return _AuctionInputs(
        labels.astype(jnp.int32),
        mask,
        value_array,
        lower.astype(jnp.int32),
        upper.astype(jnp.int32),
        prices.astype(value_array.dtype),
        slots.astype(jnp.int32),
        labels_representable,
        bounds_representable,
        slots_representable,
    )


def _label_counts(labels: Array, weights: Array, label_count: int, /) -> Array:
    # The segments are the auction's per-round dynamic routes (assignment, pools,
    # offers), not a fixed topology, so a direct segment reduction is the kernel.
    return jax.ops.segment_sum(
        weights.astype(jnp.int32), labels, num_segments=label_count + 1
    )[:label_count]

def _capped_nonnegative_sum(values: Array, limit: Array, /) -> tuple[Array, Array]:
    """Sum nonnegative int32 values without overflow, capped at ``limit``."""

    def add(index: int, carry: tuple[Array, Array], /) -> tuple[Array, Array]:
        total, exceeded = carry
        value = jnp.maximum(values[index], jnp.int32(0))
        room = limit - total
        beyond = value > room
        return total + jnp.minimum(value, room), exceeded | beyond

    initial = (jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.bool_))
    return jax.lax.fori_loop(0, values.shape[0], add, initial)


def _prepare_instance(inputs: _AuctionInputs, label_count: int, /) -> _Instance:
    site_count, width = inputs.labels.shape
    in_range = (inputs.labels >= 0) & (inputs.labels < label_count)
    listed = inputs.valid & in_range
    labels = jnp.where(listed, inputs.labels, 0)
    keys = jnp.sort(
        jnp.where(listed, labels, label_count + jnp.arange(width, dtype=jnp.int32)),
        axis=1,
    )
    duplicated = jnp.any(keys[:, 1:] == keys[:, :-1])
    consistent = (
        inputs.labels_representable
        & jnp.all(~inputs.valid | in_range)
        & ~duplicated
    )
    finite = jnp.all(jnp.where(listed, jnp.isfinite(inputs.values), True)) & jnp.all(
        jnp.isfinite(inputs.prices)
    )
    usable = listed & (inputs.upper[labels] > 0)
    values = jnp.where(usable & jnp.isfinite(inputs.values), inputs.values, 0.0)
    any_usable = jnp.any(usable)
    value_range = jnp.where(
        any_usable,
        jnp.max(jnp.where(usable, values, -jnp.inf))
        - jnp.min(jnp.where(usable, values, jnp.inf)),
        0.0,
    ).astype(values.dtype)
    # Counts beyond N never bind; clipping keeps int32 sums exact.
    site_count_i32 = jnp.int32(site_count)
    lower_clip = jnp.int32(min(site_count + 1, _INT32_LIMIT))
    upper = jnp.minimum(inputs.upper, site_count_i32)
    lower = jnp.minimum(inputs.lower, lower_clip)
    candidate_sites = _label_counts(labels.reshape(-1), usable.reshape(-1), label_count)
    _, lower_exceeded = _capped_nonnegative_sum(lower, site_count_i32)
    upper_total, _ = _capped_nonnegative_sum(upper, site_count_i32)
    bounds = (
        inputs.bounds_representable
        & jnp.all(inputs.lower >= jnp.int32(0))
        & jnp.all(inputs.lower <= inputs.upper)
        & ~lower_exceeded
        & (upper_total >= site_count_i32)
        & jnp.all(jnp.any(usable, axis=1))
        & jnp.all(candidate_sites >= lower)
    )
    width_i32 = jnp.int32(width)
    slot_in_range = (inputs.slots >= jnp.int32(-1)) & (
        inputs.slots < width_i32
    )
    warm_start_consistent = inputs.slots_representable & jnp.all(slot_in_range)
    held = (
        (inputs.slots >= jnp.int32(0))
        & jnp.take_along_axis(
            usable, jnp.clip(inputs.slots, jnp.int32(0), width_i32 - 1)[:, None], axis=1
        )[:, 0]
    )
    return _Instance(
        labels,
        usable,
        values,
        lower,
        upper,
        inputs.upper,
        jnp.where(held, inputs.slots, -1),
        value_range,
        consistent,
        finite,
        bounds,
        warm_start_consistent,
    )


def _slot_take(table: Array, slots: Array, /) -> Array:
    return jnp.take_along_axis(table, jnp.maximum(slots, 0)[:, None], axis=1)[:, 0]


def _segment_kth(sorted_values: Array, start: Array, rank: Array, /) -> Array:
    """Return the ``rank``-th (1-based) entry of each sorted label segment."""

    index = jnp.clip(start + rank - 1, 0, sorted_values.shape[0] - 1)
    return sorted_values[index]


def _forward_round(
    instance: _Instance, state: _ForwardState, epsilon: Array, bound: Array, /
) -> _ForwardState:
    labels, usable, values = instance.labels, instance.usable, instance.values
    site_count, width = labels.shape
    label_count = instance.lower.shape[0]
    lower, upper = instance.lower, instance.upper
    held = state.slots >= 0
    net = jnp.where(usable, values - state.prices[labels], -jnp.inf)
    best = jnp.argmax(net, axis=1).astype(jnp.int32)
    best_net = _slot_take(net, best)
    others = jnp.where(
        jnp.arange(width, dtype=jnp.int32)[None, :] == best[:, None], -jnp.inf, net
    )
    second = jnp.max(others, axis=1)
    second = jnp.where(
        second > -jnp.inf, second, best_net - (instance.value_range + epsilon)
    )
    target_slot = jnp.where(held, state.slots, best)
    bid = jnp.where(held, state.bids, _slot_take(values, best) - second + epsilon)
    target = _slot_take(labels, target_slot)
    # While live every site has a usable slot, so every site is in exactly one pool.
    sites = jnp.arange(site_count, dtype=jnp.int32)
    sorted_label, sorted_negative_bid, sorted_site = jax.lax.sort(
        (target, -bid, sites), num_keys=3
    )
    sorted_bid = -sorted_negative_bid
    pool = _label_counts(target, jnp.ones_like(target), label_count)
    nonnegative = _label_counts(target, bid >= 0.0, label_count)
    start = jnp.cumsum(pool) - pool
    within_lower = pool <= lower
    negative_tail = ~within_lower & (_segment_kth(sorted_bid, start, lower + 1) < 0.0)
    extended = lower + jnp.minimum(upper - lower, jnp.maximum(0, nonnegative - lower))
    accepted = jnp.where(within_lower, pool, jnp.where(negative_tail, lower, extended))
    lower_price = jnp.minimum(
        jnp.where(lower >= 1, _segment_kth(sorted_bid, start, lower), jnp.inf), 0.0
    )
    overdemanded = (accepted == upper) & (pool > upper)
    upper_price = jnp.where(
        overdemanded,
        _segment_kth(sorted_bid, start, upper),
        jnp.maximum(state.prices, 0.0),
    )
    prices = jnp.where(
        within_lower,
        state.prices,
        jnp.where(negative_tail, lower_price, upper_price),
    )
    rank = sites - start[sorted_label]
    accept = (
        jnp.zeros((site_count,), dtype=jnp.bool_)
        .at[sorted_site]
        .set(rank < accepted[sorted_label], unique_indices=True)
    )
    return _ForwardState(
        jnp.where(accept, target_slot, -1),
        jnp.where(accept, bid, 0.0),
        prices,
        state.rounds + 1,
        jnp.any(jnp.abs(prices) > bound),
    )


def _retained(
    instance: _Instance, slots: Array, prices: Array, epsilon: Array, /
) -> tuple[Array, Array]:
    """Warm-start slots satisfying this stage's eps-CS, with their equivalent bids.

    A retained site defends ``a_held - w_other + eps`` (``w_other`` its best net
    value elsewhere), the bid a fresh forward round would place for its slot. It
    is at least the label price and keeps ``a - bid >= w_other - eps``, so the
    forward invariants hold and retained sites compete like ordinary bidders.
    """
    width = instance.labels.shape[1]
    net = jnp.where(instance.usable, instance.values - prices[instance.labels], -jnp.inf)
    held = slots >= 0
    held_net = _slot_take(net, slots)
    keep = held & (jnp.max(net, axis=1) - held_net <= epsilon)
    others = jnp.where(
        jnp.arange(width, dtype=jnp.int32)[None, :] == slots[:, None], -jnp.inf, net
    )
    other = jnp.max(others, axis=1)
    other = jnp.where(
        other > -jnp.inf, other, held_net - (instance.value_range + epsilon)
    )
    bid = _slot_take(instance.values, slots) - other + epsilon
    return jnp.where(keep, slots, -1), jnp.where(keep, bid, 0.0)


def _forward_phase(
    instance: _Instance,
    slots: Array,
    prices: Array,
    epsilon: Array,
    bound: Array,
    live: Array,
    maximum_rounds: int,
    /,
) -> _ForwardState:
    def unfinished(state: _ForwardState, /) -> Array:
        # One round always runs so warm-started pools pass the acceptance rule
        # (capacity and sign limits) even when every site starts assigned.
        return (
            live
            & ~state.diverged
            & (state.rounds < maximum_rounds)
            & (jnp.any(state.slots < 0) | (state.rounds == 0))
        )

    def advance(state: _ForwardState, /) -> _ForwardState:
        return _forward_round(instance, state, epsilon, bound)

    kept, bids = _retained(instance, slots, prices, epsilon)
    initial = _ForwardState(
        kept,
        bids.astype(prices.dtype),
        prices,
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.bool_),
    )
    return jax.lax.while_loop(unfinished, advance, initial)


def _reverse_bidders(
    instance: _Instance, slots: Array, prices: Array, /
) -> tuple[Array, Array, Array]:
    """Return ``(lower_deficient, bidder, need)`` of the current assignment."""

    label_count = instance.lower.shape[0]
    counts = _label_counts(_slot_take(instance.labels, slots), slots >= 0, label_count)
    lower_deficient = counts < instance.lower
    upper_deficient = ~lower_deficient & (counts < instance.upper) & (prices > 0.0)
    bidder = lower_deficient | upper_deficient
    need = jnp.where(lower_deficient, instance.lower - counts, instance.upper - counts)
    return lower_deficient, bidder, need


def _reverse_round(
    instance: _Instance, state: _ReverseState, epsilon: Array, bound: Array, /
) -> _ReverseState:
    labels, usable, values = instance.labels, instance.usable, instance.values
    site_count, width = labels.shape
    label_count = instance.lower.shape[0]
    routes = site_count * width
    lower_deficient, bidder, need = _reverse_bidders(instance, state.slots, state.prices)
    current = _slot_take(labels, state.slots)
    profit = _slot_take(values, state.slots) - state.prices[current]
    route = usable & bidder[labels] & (labels != current[:, None])
    route_label = jnp.where(route, labels, label_count).reshape(-1)
    route_value = jnp.where(route, values - profit[:, None], 0.0).reshape(-1)
    route_site = jnp.repeat(jnp.arange(site_count, dtype=jnp.int32), width)
    route_ids = jnp.arange(routes, dtype=jnp.int32)
    sorted_label, sorted_negative_value, _, sorted_route = jax.lax.sort(
        (route_label, -route_value, route_site, route_ids), num_keys=3
    )
    sorted_value = -sorted_negative_value
    available = _label_counts(route_label, route.reshape(-1), label_count)
    start = jnp.cumsum(available) - available
    take = jnp.minimum(need, available)
    crossing = jnp.where(
        available > take,
        _segment_kth(sorted_value, start, take + 1),
        _segment_kth(sorted_value, start, take),
    )
    floor = jnp.where(lower_deficient, -jnp.inf, 0.0)
    offered_price = jnp.maximum(floor, crossing - epsilon)
    offered_price = jnp.where(
        available == 0, jnp.where(lower_deficient, state.prices, 0.0), offered_price
    )
    prices = jnp.where(bidder, jnp.minimum(offered_price, state.prices), state.prices)
    safe_label = jnp.minimum(sorted_label, label_count - 1)
    rank = route_ids - start[safe_label]
    offer_sorted = (
        (sorted_label < label_count)
        & (rank < take[safe_label])
        & (sorted_value > prices[safe_label])
    )
    offer = (
        jnp.zeros((routes,), dtype=jnp.bool_)
        .at[sorted_route]
        .set(offer_sorted, unique_indices=True)
        .reshape((site_count, width))
    )
    gain = jnp.where(offer, values - prices[labels], -jnp.inf)
    best_gain = jnp.max(gain, axis=1)
    chosen_label = jnp.min(
        jnp.where(offer & (gain == best_gain[:, None]), labels, label_count), axis=1
    )
    chosen_slot = jnp.argmax(offer & (labels == chosen_label[:, None]), axis=1)
    slots = jnp.where(jnp.any(offer, axis=1), chosen_slot.astype(jnp.int32), state.slots)
    stranded = jnp.any(lower_deficient & (available == 0))
    _, pending, _ = _reverse_bidders(instance, slots, prices)
    return _ReverseState(
        slots,
        prices,
        state.rounds + 1,
        stranded | jnp.any(jnp.abs(prices) > bound),
        jnp.any(pending),
    )


def _reverse_phase(
    instance: _Instance,
    slots: Array,
    prices: Array,
    epsilon: Array,
    bound: Array,
    live: Array,
    maximum_rounds: int,
    /,
) -> _ReverseState:
    def unfinished(state: _ReverseState, /) -> Array:
        return live & state.pending & ~state.diverged & (state.rounds < maximum_rounds)

    def advance(state: _ReverseState, /) -> _ReverseState:
        return _reverse_round(instance, state, epsilon, bound)

    _, bidder, _ = _reverse_bidders(instance, slots, prices)
    initial = _ReverseState(
        slots,
        prices,
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.bool_),
        jnp.any(bidder),
    )
    return jax.lax.while_loop(unfinished, advance, initial)


def _epsilons(plan: CapacitatedAuctionPlan, instance: _Instance, /) -> Array:
    dtype = instance.values.dtype
    match plan.epsilon_scale:
        case "absolute":
            scale = jnp.ones((), dtype=dtype)
        case "value-range":
            scale = jnp.where(instance.value_range > 0.0, instance.value_range, 1.0)
        case _:
            assert_never(plan.epsilon_scale)
    return jnp.asarray(plan.epsilon_schedule, dtype=dtype) * scale.astype(dtype)


def _run_stages(
    plan: CapacitatedAuctionPlan,
    instance: _Instance,
    prices: Array,
    epsilons: Array,
    live: Array,
    /,
) -> tuple[_StageCarry, Array]:
    label_count = plan.label_count

    def stage(carry: _StageCarry, epsilon: Array, /) -> tuple[_StageCarry, Array]:
        active = live & ~carry.diverged & ~carry.capped
        bound = jnp.max(jnp.abs(carry.prices)) + 2.0 * (label_count + 2) * (
            instance.value_range + epsilon
        )
        forward = _forward_phase(
            instance,
            carry.slots,
            carry.prices,
            epsilon,
            bound,
            active,
            plan.maximum_rounds,
        )
        forward_capped = active & ~forward.diverged & jnp.any(forward.slots < 0)
        reverse_live = active & ~forward.diverged & ~forward_capped
        reverse = _reverse_phase(
            instance,
            forward.slots,
            forward.prices,
            epsilon,
            bound,
            reverse_live,
            plan.maximum_rounds,
        )
        reverse_capped = reverse_live & ~reverse.diverged & reverse.pending
        updated = _StageCarry(
            reverse.slots,
            reverse.prices,
            carry.diverged | forward.diverged | reverse.diverged,
            carry.capped | forward_capped | reverse_capped,
        )
        return updated, forward.rounds + reverse.rounds

    initial = _StageCarry(
        instance.initial_slots,
        prices,
        jnp.zeros((), dtype=jnp.bool_),
        jnp.zeros((), dtype=jnp.bool_),
    )
    return jax.lax.scan(stage, initial, epsilons)


def _auction_status(
    instance: _Instance,
    final: _StageCarry,
    certified: Array,
    optimal: Array,
    /,
) -> Array:
    return jnp.select(
        (
            ~instance.finite_values,
            ~instance.candidates_consistent
            | ~instance.bounds_consistent
            | ~instance.warm_start_consistent
            | final.diverged,
            final.capped,
            ~certified,
            optimal,
        ),
        (
            jnp.int32(CombinatorialStatus.NONFINITE_INPUT),
            jnp.int32(CombinatorialStatus.INFEASIBLE),
            jnp.int32(CombinatorialStatus.MAXIMUM_STEPS_REACHED),
            jnp.int32(CombinatorialStatus.CERTIFICATION_FAILED),
            jnp.int32(CombinatorialStatus.OPTIMAL),
        ),
        jnp.int32(CombinatorialStatus.FEASIBLE),
    )


def _auction_kernel(
    plan: CapacitatedAuctionPlan,
    inputs: _AuctionInputs,
    certification: CombinatorialCertification,
    /,
) -> CapacitatedAuctionResult:
    """The single capacitated-auction kernel shared by every public entry."""

    site_count = plan.site_count
    label_count = plan.label_count
    instance = _prepare_instance(inputs, label_count)
    epsilons = _epsilons(plan, instance)
    live = (
        instance.candidates_consistent
        & instance.finite_values
        & instance.bounds_consistent
        & instance.warm_start_consistent
    )
    prices0 = jnp.where(jnp.isfinite(inputs.prices), inputs.prices, 0.0)
    final, stage_rounds = _run_stages(plan, instance, prices0, epsilons, live)

    slots, prices = final.slots, final.prices
    dtype = instance.values.dtype
    assigned = slots >= 0
    current = jnp.where(assigned, _slot_take(instance.labels, slots), -1)
    counts = _label_counts(
        jnp.where(assigned, current, label_count), assigned, label_count
    )
    assigned_count = jnp.sum(assigned, dtype=jnp.int32)
    lower_residual, _ = _capped_nonnegative_sum(
        jnp.maximum(instance.lower - counts, jnp.int32(0)),
        jnp.int32(_INT32_LIMIT),
    )
    upper_residual, _ = _capped_nonnegative_sum(
        jnp.maximum(counts - instance.raw_upper, jnp.int32(0)),
        jnp.int32(_INT32_LIMIT) - lower_residual,
    )
    missing = jnp.int32(site_count) - assigned_count
    residual_room = jnp.int32(_INT32_LIMIT) - lower_residual - upper_residual
    capacity_residual = lower_residual + upper_residual + jnp.minimum(
        missing, residual_room
    )
    feasible = (
        instance.candidates_consistent
        & instance.finite_values
        & instance.bounds_consistent
        & instance.warm_start_consistent
        & jnp.all(assigned)
        & (capacity_residual == jnp.int32(0))
    )

    net = jnp.where(instance.usable, instance.values - prices[instance.labels], -jnp.inf)
    best_net = jnp.max(net, axis=1)
    held_net = _slot_take(net, slots)
    epsilon_cs = jnp.max(jnp.where(assigned, best_net - held_net, 0.0))
    sign_cs = jnp.max(
        jnp.maximum(
            jnp.where((prices > 0.0) & (counts < instance.upper), prices, 0.0),
            jnp.where((prices < 0.0) & (counts > instance.lower), -prices, 0.0),
        )
    )
    primal = jnp.sum(jnp.where(assigned, _slot_take(instance.values, slots), 0.0))
    dual = jnp.sum(best_net) + jnp.sum(
        jnp.maximum(
            instance.upper.astype(dtype) * prices, instance.lower.astype(dtype) * prices
        )
    )
    gap = dual - primal
    epsilon = epsilons[-1]
    gap_bound = site_count * epsilon
    # Per-site roundoff slack of the net-value comparisons in the working dtype.
    magnitude = jnp.maximum(
        1.0,
        jnp.max(jnp.abs(instance.values)) + jnp.max(jnp.abs(prices)),
    )
    slack = 64.0 * jnp.finfo(dtype).eps * magnitude
    certified = (
        feasible
        & (epsilon_cs <= epsilon + slack)
        & (sign_cs == 0.0)
        & (gap <= gap_bound + site_count * slack)
    )
    optimal = gap <= certification.threshold(primal)
    status = _auction_status(instance, final, certified, optimal)
    success = (
        (status == int(CombinatorialStatus.OPTIMAL))
        | (status == int(CombinatorialStatus.FEASIBLE))
    ) & feasible
    evidence = CapacitatedAuctionEvidence(
        candidates_consistent=instance.candidates_consistent,
        bounds_consistent=instance.bounds_consistent,
        warm_start_consistent=instance.warm_start_consistent,
        finite_values=instance.finite_values,
        feasible=feasible,
        assigned_count=assigned_count,
        capacity_residual=capacity_residual,
        rounds=jnp.sum(stage_rounds, dtype=jnp.int32),
        stage_rounds=stage_rounds.astype(jnp.int32),
        epsilon=epsilon,
        epsilon_cs_violation=epsilon_cs.astype(dtype),
        sign_cs_violation=sign_cs.astype(dtype),
        primal_value=primal.astype(dtype),
        dual_value=dual.astype(dtype),
        duality_gap=gap.astype(dtype),
        gap_bound=gap_bound.astype(dtype),
        price_divergence=final.diverged,
        round_limit_reached=final.capped,
    )
    return CapacitatedAuctionResult(
        labels=jnp.where(success, current, -1).astype(jnp.int32),
        slots=jnp.where(success, slots, -1).astype(jnp.int32),
        counts=jnp.where(success, counts, 0).astype(jnp.int32),
        prices=jnp.where(success, prices, jnp.nan).astype(dtype),
        status=status,
        evidence=evidence,
    )


class CapacitatedAssignmentDecision(StrictModule):
    """Selected zero-based candidate slot of every site (``-1`` when unassigned)."""

    slots: Array


class CapacitatedAssignmentSpace(AbstractCombinatorialSpace):
    """Full site assignments to candidate labels under integer label-count bounds.

    Objective features are the ``[sites, width]`` one-hot matrix of chosen slots.
    """

    candidates: RowRelation
    lower_counts: Array
    upper_counts: Array
    site_count: int = eqx.field(static=True)
    label_count: int = eqx.field(static=True)
    candidate_width: int = eqx.field(static=True)
    _structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidates: RowRelation,
        lower_counts: ConvertibleToArray,
        upper_counts: ConvertibleToArray,
        /,
    ) -> None:
        sites, width, labels = _validated_candidates(candidates)
        scope = Scope()
        parse(labels, Size[_LabelDim], "label_count", scope=scope)
        lower = as_host_array(
            lower_counts, HostInteger[_LabelDim], "lower_counts", scope=scope
        )
        upper = as_host_array(
            upper_counts, HostInteger[_LabelDim], "upper_counts", scope=scope
        )
        _host_int32_representable(lower, "lower_counts")
        _host_int32_representable(upper, "upper_counts")
        if np.any(lower > upper):
            raise ValueError("label counts require 0 <= lower_counts <= upper_counts.")
        lower_ = jnp.asarray(lower, dtype=jnp.int32)
        upper_ = jnp.asarray(upper, dtype=jnp.int32)
        self.candidates = candidates
        self.lower_counts = lower_
        self.upper_counts = upper_
        self.site_count = sites
        self.label_count = labels
        self.candidate_width = width
        self._structure_id = canonical_fingerprint(
            {
                "kind": "capacitated-assignment-space",
                "sites": sites,
                "labels": labels,
                "width": width,
                "candidates": array_tree_fingerprint(
                    (candidates.source_indices, candidates.valid)
                ),
                "lower_counts": array_tree_fingerprint(lower_),
                "upper_counts": array_tree_fingerprint(upper_),
            }
        )

    @property
    def structure_id(self) -> str:
        return self._structure_id

    def decision_spec(self, /) -> CapacitatedAssignmentDecision:
        sites = self.site_count
        return jax.eval_shape(
            lambda: CapacitatedAssignmentDecision(jnp.zeros((sites,), dtype=jnp.int32))
        )

    def feature_spec(self, /) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.site_count, self.candidate_width), jnp.float32)

    def canonicalize(
        self, decision: CapacitatedAssignmentDecision, /
    ) -> CapacitatedAssignmentDecision:
        if not isinstance(decision, CapacitatedAssignmentDecision):
            raise TypeError(
                "capacitated decisions must be CapacitatedAssignmentDecision values."
            )
        slots = jnp.asarray(decision.slots)
        if not jnp.issubdtype(slots.dtype, jnp.integer):
            raise TypeError("capacitated decision slots must have an integer dtype.")
        if slots.shape[-1:] != (self.site_count,):
            raise ValueError(
                f"decision slots must end with shape {(self.site_count,)}; got {slots.shape}."
            )
        in_range = _device_int32_representable(slots, minimum=0)
        in_range = in_range & (slots < self.candidate_width)
        safe = jnp.where(in_range, slots, 0).astype(jnp.int32)
        sites = jnp.arange(self.site_count, dtype=jnp.int32)
        admissible = in_range & self.candidates.valid[sites, safe]
        return CapacitatedAssignmentDecision(jnp.where(admissible, safe, -1))

    def encode(self, decision: CapacitatedAssignmentDecision, /) -> Array:
        slots = self.canonicalize(decision).slots
        return jax.nn.one_hot(slots, self.candidate_width, dtype=jnp.float64, axis=-1)

    def audit(
        self, decision: CapacitatedAssignmentDecision, /
    ) -> CombinatorialFeasibility:
        slots = self.canonicalize(decision).slots
        flat = slots.reshape((-1, self.site_count))
        label_count = self.label_count

        def residual_one(assignment: Array, /) -> Array:
            assigned = assignment >= 0
            labels = jnp.where(
                assigned,
                _slot_take(self.candidates.source_indices, assignment),
                label_count,
            )
            counts = _label_counts(labels, assigned, label_count)
            missing = jnp.sum(~assigned, dtype=jnp.int32)
            lower_residual, _ = _capped_nonnegative_sum(
                self.lower_counts - counts,
                jnp.int32(_INT32_LIMIT) - missing,
            )
            upper_residual, _ = _capped_nonnegative_sum(
                counts - self.upper_counts,
                jnp.int32(_INT32_LIMIT) - missing - lower_residual,
            )
            return missing + lower_residual + upper_residual

        residual = jax.vmap(residual_one)(flat).reshape(slots.shape[:-1])
        return CombinatorialFeasibility(residual == 0, residual.astype(jnp.float64))


class CapacitatedAuction(AbstractLinearCombinatorialMethod):
    """Epsilon-scaled forward/reverse auction for capacitated assignment spaces.

    Minimizes ``costs = -values`` through the prepared capacitated-auction
    kernel. Results are eps-optimal with an explicit duality-gap certificate;
    ``optimality_proven`` requires the gap below the certification threshold.
    """

    epsilon_schedule: tuple[float, ...] = eqx.field(static=True)
    epsilon_scale: EpsilonScale = eqx.field(static=True)
    maximum_rounds: int = eqx.field(static=True)
    maximum_routes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        epsilon_schedule: tuple[float, ...] = _DEFAULT_SCHEDULE,
        epsilon_scale: EpsilonScale = "value-range",
        maximum_rounds: int = 10_000,
        maximum_routes: int = 50_000_000,
    ) -> None:
        schedule = _validated_schedule(epsilon_schedule)
        scale = parse(epsilon_scale, EpsilonScale, "epsilon_scale")
        rounds = _validated_rounds(maximum_rounds, len(schedule))
        routes = _positive_integer(maximum_routes, "maximum_routes")
        self.epsilon_schedule = schedule
        self.epsilon_scale = scale
        self.maximum_rounds = rounds
        self.maximum_routes = routes

    @property
    def method_id(self) -> str:
        return "native-capacitated-auction"

    @property
    def capabilities(self) -> CombinatorialMethodCapabilities:
        return CombinatorialMethodCapabilities(
            exact=False,
            jax_native=True,
            jit=True,
            batched=True,
            signed_costs=True,
            deterministic_ties=True,
            optimality_certificate=True,
            warm_start=False,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (
            ("epsilon_schedule", repr(self.epsilon_schedule)),
            ("epsilon_scale", self.epsilon_scale),
            ("maximum_rounds", str(self.maximum_rounds)),
            ("maximum_routes", str(self.maximum_routes)),
        )

    def _auction_plan(
        self, space: CapacitatedAssignmentSpace, /
    ) -> CapacitatedAuctionPlan:
        return CapacitatedAuctionPlan(
            space.site_count,
            space.label_count,
            space.candidate_width,
            epsilon_schedule=self.epsilon_schedule,
            epsilon_scale=self.epsilon_scale,
            maximum_rounds=self.maximum_rounds,
            maximum_routes=self.maximum_routes,
        )

    def plan(
        self,
        problem: LinearCombinatorialProblem,
        certification: CombinatorialCertification,
        /,
    ) -> CombinatorialPlan:
        space = problem.space
        if not isinstance(space, CapacitatedAssignmentSpace):
            raise TypeError("CapacitatedAuction requires CapacitatedAssignmentSpace.")
        if len(jax.tree_util.tree_leaves(problem.costs)) != 1:
            raise ValueError("CapacitatedAssignmentSpace requires one cost matrix.")
        self._auction_plan(space)
        routes = space.site_count * space.candidate_width
        stages = len(self.epsilon_schedule)
        return make_combinatorial_plan(
            problem,
            self,
            certification,
            work_estimate=problem.batch_size * stages * 2 * self.maximum_rounds * routes,
            workspace_elements=problem.batch_size
            * (6 * routes + 6 * space.site_count + 4 * space.label_count),
            certificate_kind="capacitated-auction-duality-gap",
        )

    def solve(
        self,
        problem: LinearCombinatorialProblem,
        plan: CombinatorialPlan,
        /,
    ) -> CombinatorialResult:
        space = problem.space
        if not isinstance(space, CapacitatedAssignmentSpace):
            raise TypeError("CapacitatedAuction requires CapacitatedAssignmentSpace.")
        auction = self._auction_plan(space)
        raw_costs = jax.tree_util.tree_leaves(problem.costs)[0]
        batch_shape = problem.batch_shape
        shape = (space.site_count, space.candidate_width)
        flat_values = -raw_costs.reshape((problem.batch_size,) + shape)
        certification = plan.certification

        def solve_one(values: Array, /) -> CapacitatedAuctionResult:
            inputs = _auction_inputs(
                auction,
                space.candidates.source_indices,
                space.candidates.valid,
                values,
                space.lower_counts,
                space.upper_counts,
                None,
                None,
            )
            return _auction_kernel(auction, inputs, certification)

        auction_result = jax.vmap(solve_one)(flat_values)
        evidence = auction_result.evidence

        def shaped(value: Array, /) -> Array:
            return value.reshape(batch_shape + value.shape[1:])

        decision = CapacitatedAssignmentDecision(shaped(auction_result.slots))
        features = space.encode(decision).astype(raw_costs.dtype)
        objective = problem.objective(features)
        feasibility = space.audit(decision)
        primal = shaped(-evidence.primal_value)
        dual = shaped(-evidence.dual_value)
        gap = shaped(evidence.duality_gap)
        status = shaped(auction_result.status)
        tolerance = certification.threshold(objective, primal)
        success = shaped(auction_result.success)
        valid = success & feasibility.feasible
        objective_consistent = ~valid | (jnp.abs(objective - primal) <= tolerance)
        optimality_proven = (
            valid & objective_consistent & (gap <= certification.threshold(objective))
        )
        certificate = CombinatorialCertificate(
            finite=shaped(evidence.finite_values),
            feasible=feasibility.feasible,
            objective_consistent=objective_consistent,
            optimality_proven=optimality_proven,
            primal_residual=shaped(evidence.capacity_residual).astype(raw_costs.dtype),
            dual_residual=shaped(evidence.epsilon_cs_violation),
            absolute_gap=gap,
            relative_gap=relative_gap(jnp.abs(gap), primal, dual),
            tie_margin=jnp.full(batch_shape, jnp.nan, dtype=raw_costs.dtype),
            dual_available=shaped(evidence.feasible),
            gap_available=shaped(evidence.feasible),
            tie_available=jnp.zeros(batch_shape, dtype=jnp.bool_),
        )
        provenance = CombinatorialProvenance(
            problem_id=problem.problem_id,
            structure_id=problem.structure_id,
            method_id=self.method_id,
            plan_id=plan.plan_id,
            implementation="phydrax-native-jax",
            tie_policy="lowest-slot-bid-then-lowest-site-then-lowest-label-offer",
            certificate_kind=plan.certificate_kind,
            exact=False,
            signed_costs=True,
            configuration=plan.configuration,
        )
        rounds = shaped(evidence.rounds)
        return CombinatorialResult(
            decision=decision,
            features=jnp.where(
                valid[..., None, None], features, jnp.zeros_like(features)
            ),
            objective_value=jnp.where(valid, objective, jnp.nan),
            status=status,
            valid=valid,
            certificate=certificate,
            iterations=rounds,
            work=rounds.astype(jnp.int64) * (space.site_count * space.candidate_width),
            provenance=provenance,
            batch_shape=batch_shape,
        )


__all__ = [
    "CapacitatedAssignmentDecision",
    "CapacitatedAssignmentSpace",
    "CapacitatedAuction",
    "CapacitatedAuctionEvidence",
    "CapacitatedAuctionPlan",
    "CapacitatedAuctionResult",
    "EpsilonScale",
    "PreparedCapacitatedAuction",
]
