#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from math import prod
from typing import Any, assert_never, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..._differentiation import DerivativeRoute
from ..._sampling._addressing import derive_key, SampleAddress
from ...sparse import RelationAccumulation
from ...typing import parse, PRNGKey
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
    prediction_fit_contract,
)
from ._common import (
    active_data,
    cluster_sums,
    ClusterDiagnostics,
    ClusterInitialization,
    ClusterRevival,
    effective_sample_count,
    EMA_MINI_BATCH_METHOD,
    HardClusterModel,
    initialize_centers,
    MiniBatchBatchLaw,
    MiniBatchClusterEvidence,
    nearest_centers,
    WORKSPACE_BYTES,
)


# Scaling laws of Bessa et al. (2026, arXiv:2611.09501), with the coefficients
# released by the authors:
# B = c * k**0.134749 * n**0.876473 * T**0.813344 and
# alpha0 = 0.8244 * k**-0.4617 * B**0.9003 * n**-0.4498 * T**-0.1057.
# Both were fitted on DEEP embeddings (d = 96) with full passes; the step law
# was fitted with a cosine decaying to zero and, as in the reference, is run
# with the decay floor below.
_QUALITY_BATCH = 0.00439175
_THROUGHPUT_BATCH = 0.035134
_BATCH_EXPONENTS = (0.134749, 0.876473, 0.813344)
_STEP_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)
_STEP_BOUNDS = (1e-4, 1.0)
# The cosine anneal ends at this fraction of the starting step.
_STEP_FLOOR = 0.1
# Fitted domain of the laws: samples, clusters, samples per cluster, epochs.
_FITTED_SAMPLES = (1e5, 1e8)
_FITTED_CLUSTERS = (256, 65536)
_FITTED_SAMPLES_PER_CLUSTER = 8.0
_FITTED_EPOCHS = (1.0, 25.0)
# Revival is checked after every fifth update.
_REVIVAL_PERIOD = 5
# A split child is offset by this fraction of its donor's RMS radius.
_SPLIT_OFFSET = 2.0**-7

_INITIALIZATION_ADDRESS = SampleAddress(
    "phydrax.ml.clustering", EMA_MINI_BATCH_METHOD, role="initialization"
)
_ORDER_ADDRESS = SampleAddress(
    "phydrax.ml.clustering", EMA_MINI_BATCH_METHOD, role="epoch-order"
)
_DONOR_ADDRESS = SampleAddress(
    "phydrax.ml.clustering", EMA_MINI_BATCH_METHOD, role="revival-donor"
)
_DIRECTION_ADDRESS = SampleAddress(
    "phydrax.ml.clustering", EMA_MINI_BATCH_METHOD, role="revival-direction"
)


def _round_power_of_two(value: float) -> int:
    """Nearest power of two in ratio; a geometric tie rounds up."""
    if value <= 1.0:
        return 1
    low = 2 ** math.floor(math.log2(value))
    high = 2 * low
    return low if value / low < high / value else high


def law_batch_size(
    law: MiniBatchBatchLaw, samples: int, clusters: int, epochs: float, /
) -> int:
    """Power-of-two batch size of the fitted batch law, capped near half the data."""
    k_exponent, n_exponent, epoch_exponent = _BATCH_EXPONENTS
    match law:
        case "quality":
            coefficient, cap = _QUALITY_BATCH, samples // 2
        case "throughput":
            coefficient, cap = _THROUGHPUT_BATCH, _round_power_of_two(samples / 2)
        case _:
            assert_never(law)
    raw = (
        coefficient * clusters**k_exponent * samples**n_exponent * epochs**epoch_exponent
    )
    return max(1, min(_round_power_of_two(raw), cap, samples))


def epoch_fractions(epochs: float, /) -> tuple[float, ...]:
    """Data fraction of every pass: whole passes, a partial pass, then a closing pass."""
    whole = math.floor(epochs)
    if whole == 0:
        return (epochs,)
    remainder = epochs - whole
    partial = (remainder,) if remainder > 0.0 else ()
    return (1.0,) * (whole - 1) + partial + (1.0,)


class _Schedule(NamedTuple):
    """Static per-fit schedule shared by every case."""

    cluster_count: int
    batch_size: int
    slots: int
    fractions: tuple[float, ...]
    epochs: float
    initial_step: float | None
    threshold: float
    revival: bool
    accumulation: RelationAccumulation
    workspace_bytes: int


class _State(NamedTuple):
    """Per-case cluster statistics carried across updates (centered coordinates)."""

    centers: Array
    mass: Array
    distortion: Array
    removal: Array
    closing_mass: Array
    closing_mean: Array
    closing_scatter: Array
    step: Array
    revivals: Array


def _initial_state(centers: Array, real: jnp.dtype) -> _State:
    zeros = jnp.zeros(centers.shape[:-1], dtype=real)
    return _State(
        centers=centers,
        mass=zeros,
        distortion=zeros,
        removal=zeros,
        closing_mass=zeros,
        closing_mean=jnp.zeros_like(centers),
        closing_scatter=zeros,
        step=jnp.zeros((), dtype=jnp.int32),
        revivals=jnp.zeros((), dtype=jnp.int32),
    )


def _step_size(initial_step: Array, step: Array, updates: Array) -> Array:
    """Cosine anneal from the starting step to its floor over all updates."""
    progress = step.astype(initial_step.dtype) / jnp.maximum(updates, 1).astype(
        initial_step.dtype
    )
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    return initial_step * (_STEP_FLOOR + (1.0 - _STEP_FLOOR) * cosine)


def _law_step(schedule: _Schedule, samples: Array) -> Array:
    """Starting step of the fitted step law for the active sample count."""
    coefficient, k_exponent, b_exponent, n_exponent, epoch_exponent = _STEP_LAW
    constant = (
        coefficient
        * schedule.cluster_count**k_exponent
        * schedule.batch_size**b_exponent
        * schedule.epochs**epoch_exponent
    )
    return jnp.clip(constant * samples**n_exponent, *_STEP_BOUNDS)


def _split(
    state: _State,
    donor_key: PRNGKey,
    direction_key: PRNGKey,
    threshold: float,
    *,
    closing: bool,
) -> _State:
    """Revive clusters that cost little to remove by splitting busy donors.

    A cluster is dead when its moving removal cost, the weighted rise of the
    objective if its rows moved to their second-nearest centroids, falls below
    ``threshold`` times the mean. Donors are drawn without replacement in
    proportion to their moving distortion. In the closing pass both rewritten
    slots restart their exact sums, so each closes over its post-split rows.
    """
    k = state.mass.shape[0]
    dead = state.removal < threshold * jnp.mean(state.removal)
    eligible = ~dead & (state.distortion > 0.0)
    noise = jax.random.gumbel(donor_key, (k,), dtype=state.distortion.dtype)
    priority = jnp.where(
        eligible,
        jnp.log(jnp.where(eligible, state.distortion, 1.0)) + noise,
        -jnp.inf,
    )
    donors = jnp.argsort(-priority, stable=True)
    children = jnp.argsort(~dead, stable=True)
    pairs = jnp.minimum(jnp.sum(dead), jnp.sum(eligible)).astype(jnp.int32)
    paired = jnp.arange(k, dtype=jnp.int32) < pairs
    child_slot = jnp.where(paired, children, k)
    donor_slot = jnp.where(paired, donors, k)
    base = state.centers[donors]
    radius = jnp.sqrt(
        state.distortion[donors]
        / jnp.maximum(state.mass[donors], jnp.finfo(state.mass.dtype).tiny)
    )
    direction = jax.random.normal(direction_key, base.shape, dtype=base.dtype)
    direction = direction / jnp.maximum(
        jnp.linalg.norm(direction, axis=-1, keepdims=True),
        jnp.finfo(state.mass.dtype).tiny,
    )
    offset = _SPLIT_OFFSET * radius[:, None] * direction
    centers = (
        state.centers.at[child_slot]
        .set(base + offset, mode="drop")
        .at[donor_slot]
        .set(base - offset, mode="drop")
    )

    def halve(values: Array) -> Array:
        half = 0.5 * values[donors]
        return (
            values.at[child_slot]
            .set(half, mode="drop")
            .at[donor_slot]
            .set(half, mode="drop")
        )

    def clear(values: Array) -> Array:
        empty = jnp.zeros((), dtype=values.dtype)
        return (
            values.at[child_slot]
            .set(empty, mode="drop")
            .at[donor_slot]
            .set(empty, mode="drop")
        )

    state = state._replace(
        centers=centers,
        mass=halve(state.mass),
        distortion=halve(state.distortion),
        removal=halve(state.removal),
        revivals=state.revivals + pairs,
    )
    if not closing:
        return state
    return state._replace(
        closing_mass=clear(state.closing_mass),
        closing_mean=clear(state.closing_mean),
        closing_scatter=clear(state.closing_scatter),
    )


def _close(
    state: _State,
    rows: Array,
    weights: Array,
    labels: Array,
    valid: Array,
    batch_mass: Array,
    batch_mean: Array,
    schedule: _Schedule,
) -> _State:
    """Merge the batch's exact cluster means and scatter into the closing sums."""
    residual = rows - batch_mean[labels]
    squared = jnp.real(jnp.sum(jnp.conj(residual) * residual, axis=-1))
    (scatter,) = cluster_sums(
        (weights * squared,),
        labels,
        valid,
        schedule.cluster_count,
        accumulation=schedule.accumulation,
    )
    combined = state.closing_mass + batch_mass
    received = batch_mass > 0.0
    share = jnp.where(
        received, batch_mass / jnp.maximum(combined, jnp.finfo(combined.dtype).tiny), 0.0
    )
    shift = batch_mean - state.closing_mean
    shift_squared = jnp.real(jnp.sum(jnp.conj(shift) * shift, axis=-1))
    return state._replace(
        closing_mass=combined,
        closing_mean=state.closing_mean + share[:, None] * shift,
        closing_scatter=state.closing_scatter
        + jnp.where(received, scatter + shift_squared * state.closing_mass * share, 0.0),
    )


def _update(
    state: _State,
    rows: Array,
    weights: Array,
    valid: Array,
    step_size: Array,
    schedule: _Schedule,
    *,
    revive: bool,
    closing: bool,
) -> tuple[_State, Array]:
    """One EMA update of the per-cluster statistics from one mini-batch."""
    k = schedule.cluster_count
    # Revival scores clusters by the gap to the second-nearest centroid.
    removal = revive and k > 1
    labels, nearest = nearest_centers(
        rows,
        state.centers,
        jnp.ones((k,), dtype=jnp.bool_),
        (),
        count=2 if removal else 1,
        workspace_bytes=schedule.workspace_bytes,
    )
    label, first = labels[:, 0], nearest[:, 0]
    gap = (nearest[:, 1] - first) if removal else jnp.zeros_like(first)
    batch_mass, batch_sum, batch_distortion, batch_removal = cluster_sums(
        (weights, weights[:, None] * rows, weights * first, weights * gap),
        label,
        valid,
        k,
        accumulation=schedule.accumulation,
    )
    keep = 1.0 - step_size
    mass = keep * state.mass + step_size * batch_mass
    received = batch_mass > 0.0
    tiny = jnp.finfo(batch_mass.dtype).tiny
    batch_mean = batch_sum / jnp.maximum(batch_mass, tiny)[:, None]
    rate = step_size * batch_mass / jnp.maximum(mass, tiny)
    moved = state.centers + rate[:, None] * (batch_mean - state.centers)
    # The first batch a cluster receives sets its centroid to that batch's mean.
    fresh = received & (state.mass == 0.0)
    centers = jnp.where(
        fresh[:, None], batch_mean, jnp.where(received[:, None], moved, state.centers)
    )
    state = state._replace(
        centers=centers,
        mass=mass,
        distortion=keep * state.distortion + step_size * batch_distortion,
        removal=keep * state.removal + step_size * batch_removal,
    )
    if closing:
        state = _close(
            state, rows, weights, label, valid, batch_mass, batch_mean, schedule
        )
    return state._replace(step=state.step + 1), jnp.sum(weights * first)


def _epoch(
    state: _State,
    epoch: Array | int,
    rows: Array,
    data: tuple[Array, Array, Array],
    keys: tuple[PRNGKey, Array],
    initial_step: Array,
    updates: Array,
    schedule: _Schedule,
    *,
    closing: bool,
) -> tuple[_State, Array]:
    """One pass over the first ``rows`` active rows of a fresh permutation.

    Revival stops for the closing pass when a leading pass exists; a closing
    pass that is the only pass keeps reviving and restarts rewritten slots.
    """
    values, weights, active = data
    root, case = keys
    capacity = values.shape[0]
    size = schedule.batch_size
    permutation = jax.random.permutation(
        derive_key(root, _ORDER_ADDRESS, case, epoch), capacity
    )
    order = permutation[jnp.argsort(~active[permutation], stable=True)]
    order = jnp.pad(order, (0, schedule.slots * size - capacity))
    revive = schedule.revival and not (closing and len(schedule.fractions) > 1)

    def batch(
        slot: Array, carry: tuple[_State, Array, Array]
    ) -> tuple[_State, Array, Array]:
        state, distance, weight = carry
        start = slot * size
        indices = jax.lax.dynamic_slice_in_dim(order, start, size)
        valid = start + jnp.arange(size, dtype=jnp.int32) < rows
        batch_weights = jnp.where(valid, weights[indices], 0.0)
        state, batch_distance = _update(
            state,
            values[indices],
            batch_weights,
            valid,
            _step_size(initial_step, state.step, updates),
            schedule,
            revive=revive,
            closing=closing,
        )
        if revive:
            state = jax.lax.cond(
                state.step % _REVIVAL_PERIOD == 0,
                lambda current: _split(
                    current,
                    derive_key(root, _DONOR_ADDRESS, case, current.step),
                    derive_key(root, _DIRECTION_ADDRESS, case, current.step),
                    schedule.threshold,
                    closing=closing,
                ),
                lambda current: current,
                state,
            )
        return state, distance + batch_distance, weight + jnp.sum(batch_weights)

    zero = jnp.zeros((), dtype=weights.dtype)
    state, distance, weight = jax.lax.fori_loop(
        0, (rows + size - 1) // size, batch, (state, zero, zero)
    )
    normalized = distance / jnp.maximum(weight, jnp.finfo(weight.dtype).tiny)
    return state, jnp.where(weight > 0.0, normalized, 0.0)


def _fit_case(
    values: Array,
    weights: Array,
    active: Array,
    centers: Array,
    case: Array,
    root: PRNGKey,
    schedule: _Schedule,
) -> tuple[Array, ...]:
    """EMA mini-batch k-means of one case in centered coordinates."""
    real = weights.dtype
    samples = jnp.sum(active).astype(jnp.int32)
    samples_real = samples.astype(real)
    fractions = jnp.asarray(schedule.fractions, dtype=real)
    rows = jnp.round(fractions * samples_real).astype(jnp.int32)
    updates = jnp.sum((rows + schedule.batch_size - 1) // schedule.batch_size)
    initial_step = (
        _law_step(schedule, samples_real)
        if schedule.initial_step is None
        else jnp.asarray(schedule.initial_step, dtype=real)
    )
    data = (values, weights, active)
    keys = (root, case)
    last = len(schedule.fractions) - 1

    def leading(state: _State, epoch: Array) -> tuple[_State, Array]:
        return _epoch(
            state,
            epoch,
            rows[epoch],
            data,
            keys,
            initial_step,
            updates,
            schedule,
            closing=False,
        )

    state, objectives = jax.lax.scan(
        leading, _initial_state(centers, real), jnp.arange(last, dtype=jnp.int32)
    )
    state, closing_objective = _epoch(
        state,
        last,
        rows[last],
        data,
        keys,
        initial_step,
        updates,
        schedule,
        closing=True,
    )
    total = jnp.sum(state.closing_mass)
    scatter = jnp.sum(state.closing_scatter)
    tiny = jnp.finfo(real).tiny
    drift = state.closing_mean - state.centers
    drift_energy = jnp.sum(
        state.closing_mass * jnp.real(jnp.sum(jnp.conj(drift) * drift, axis=-1))
    )
    gain = jnp.where(
        scatter > 0.0,
        drift_energy / jnp.maximum(scatter, tiny),
        jnp.where(drift_energy > 0.0, jnp.inf, 0.0),
    )
    share = state.closing_mass / jnp.maximum(total, tiny)
    in_domain = (
        (samples_real >= _FITTED_SAMPLES[0])
        & (samples_real <= _FITTED_SAMPLES[1])
        & (_FITTED_CLUSTERS[0] <= schedule.cluster_count <= _FITTED_CLUSTERS[1])
        & (samples_real >= _FITTED_SAMPLES_PER_CLUSTER * schedule.cluster_count)
        & (_FITTED_EPOCHS[0] <= schedule.epochs <= _FITTED_EPOCHS[1])
    )
    return (
        jnp.where((state.closing_mass > 0.0)[:, None], state.closing_mean, state.centers),
        state.closing_mass,
        scatter / jnp.maximum(total, tiny),
        samples,
        initial_step,
        updates,
        jnp.concatenate((objectives, closing_objective[None])),
        state.revivals,
        gain,
        schedule.cluster_count * jnp.sum(share * share),
        in_domain,
    )


class MiniBatchKMeans(AbstractRecipe):
    """EMA mini-batch k-means with scaling-law schedules (Bessa et al., 2026).

    Each cluster keeps exponential moving averages of its batch mass and sum;
    a centroid moves toward its batch mean by ``step * batch_mass / mass``, so
    it stays in the convex hull of its points. ``epochs`` passes (a fractional
    pass covers that fraction of the rows) draw fresh permutations of the
    active rows. ``batch_size`` and ``initial_step`` default to the fitted
    power laws (``batch_law``); the step anneals along a cosine to a tenth of
    its start. Every fifth update, clusters whose moving removal cost (the
    rise of the objective if their rows moved to the second-nearest centroid)
    falls below ``revival_threshold`` times the mean are revived by splitting
    donors drawn in proportion to moving distortion. The final pass also
    accumulates exact per-cluster sums, and each centroid becomes the exact
    mean of the rows that pass assigned to it. Revival stops for the final
    pass when an earlier pass exists; a lone final pass keeps reviving and
    closes rewritten slots over their post-split rows. A fit
    converges when that closing step lowers the last-pass objective by at most
    ``tolerance`` relative.
    """

    cluster_count: int = eqx.field(static=True)
    epochs: float = eqx.field(static=True)
    batch_size: int | None = eqx.field(static=True)
    initial_step: float | None = eqx.field(static=True)
    batch_law: MiniBatchBatchLaw = eqx.field(static=True)
    revival: ClusterRevival = eqx.field(static=True)
    revival_threshold: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    initialization: ClusterInitialization = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    accumulation: RelationAccumulation = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        cluster_count: int,
        /,
        *,
        epochs: float = 5.0,
        batch_size: int | None = None,
        initial_step: float | None = None,
        batch_law: MiniBatchBatchLaw = "quality",
        revival: ClusterRevival = "split",
        revival_threshold: float = 0.1,
        tolerance: float = 0.1,
        initialization: ClusterInitialization = "random",
        weight_policy: WeightPolicy = "statistical",
        accumulation: RelationAccumulation = "fast",
        workspace_bytes: int = WORKSPACE_BYTES,
    ) -> None:
        if cluster_count <= 0:
            raise ValueError("cluster_count must be positive.")
        if not (math.isfinite(epochs) and epochs > 0.0):
            raise ValueError("epochs must be finite and positive.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if initial_step is not None and not 0.0 < initial_step <= 1.0:
            raise ValueError("initial_step must lie in (0, 1].")
        if not 0.0 < revival_threshold < 1.0:
            raise ValueError("revival_threshold must lie in (0, 1).")
        if not tolerance >= 0.0:
            raise ValueError("tolerance must be nonnegative.")
        if workspace_bytes <= 0:
            raise ValueError("workspace_bytes must be positive.")
        self.cluster_count = int(cluster_count)
        self.epochs = float(epochs)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.initial_step = None if initial_step is None else float(initial_step)
        self.batch_law = parse(batch_law, MiniBatchBatchLaw, "batch_law")
        self.revival = parse(revival, ClusterRevival, "revival")
        self.revival_threshold = float(revival_threshold)
        self.tolerance = float(tolerance)
        self.initialization = parse(
            initialization, ClusterInitialization, "initialization"
        )
        self.weight_policy = parse(weight_policy, WeightPolicy, "weight_policy")
        self.accumulation = parse(accumulation, RelationAccumulation, "accumulation")
        self.workspace_bytes = int(workspace_bytes)

    def _schedule(self, capacity: int, cases: int) -> _Schedule:
        batch_size = (
            law_batch_size(self.batch_law, capacity, self.cluster_count, self.epochs)
            if self.batch_size is None
            else min(self.batch_size, capacity)
        )
        return _Schedule(
            cluster_count=self.cluster_count,
            batch_size=batch_size,
            slots=-(-capacity // batch_size),
            fractions=epoch_fractions(self.epochs),
            epochs=self.epochs,
            initial_step=self.initial_step,
            threshold=self.revival_threshold,
            revival=self.revival == "split",
            accumulation=self.accumulation,
            workspace_bytes=max(1, self.workspace_bytes // cases),
        )

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        if key is None:
            raise ValueError("mini-batch k-means requires an explicit JAX key.")
        root = parse(key, PRNGKey, "key")
        if self.cluster_count > batch.sample_count:
            raise ValueError("cluster_count cannot exceed sample capacity.")
        x, w, active, invalid = active_data(batch, self.weight_policy)
        case_shape = batch.case_shape
        cases = prod(case_shape)
        schedule = self._schedule(batch.sample_count, cases)
        total = jnp.sum(w, axis=-1)
        origin = (
            jnp.sum(w[..., :, None] * x, axis=-2)
            / jnp.maximum(total, jnp.finfo(w.dtype).tiny)[..., None]
        )
        centered = jnp.where(active[..., :, None], x - origin[..., None, :], 0)
        centers = initialize_centers(
            centered,
            w,
            self.cluster_count,
            self.initialization,
            derive_key(root, _INITIALIZATION_ADDRESS),
        )
        n, p = x.shape[-2:]
        outputs = jax.vmap(
            lambda values, weights, mask, seeds, case: _fit_case(
                values, weights, mask, seeds, case, root, schedule
            )
        )(
            centered.reshape((cases, n, p)),
            w.reshape((cases, n)),
            active.reshape((cases, n)),
            centers.reshape((cases, self.cluster_count, p)),
            jnp.arange(cases, dtype=jnp.uint32),
        )
        (
            fitted,
            mass,
            objective,
            samples,
            initial_step,
            updates,
            epoch_objectives,
            revivals,
            gain,
            imbalance,
            in_domain,
        ) = (value.reshape(case_shape + value.shape[1:]) for value in outputs)
        fitted = fitted + origin[..., None, :]
        return self._result(
            fitted,
            mass,
            objective,
            samples,
            invalid,
            effective_sample_count(w),
            MiniBatchClusterEvidence(
                initial_step=initial_step,
                updates=updates,
                epoch_objectives=epoch_objectives,
                revivals=revivals,
                closing_gain=gain,
                imbalance=imbalance,
                in_fitted_domain=in_domain,
                batch_size=schedule.batch_size,
                batch_law=self.batch_law if self.batch_size is None else None,
                epochs=self.epochs,
                epoch_fractions=schedule.fractions,
            ),
        )

    def _result(
        self,
        centers: Array,
        mass: Array,
        objective: Array,
        samples: Array,
        invalid: Array,
        effective: Array,
        evidence: MiniBatchClusterEvidence,
    ) -> FitResult:
        active_clusters = mass > 0.0
        enough = (samples >= self.cluster_count) & jnp.any(active_clusters, axis=-1)
        finite = jnp.isfinite(objective) & jnp.all(jnp.isfinite(centers), axis=(-2, -1))
        converged = evidence.closing_gain <= self.tolerance
        valid = enough & finite & converged & ~invalid
        status = jnp.where(
            invalid | ~finite,
            ML_NONFINITE,
            jnp.where(
                ~enough,
                ML_INSUFFICIENT_DATA,
                jnp.where(~converged, ML_NONCONVERGED, ML_SUCCESS),
            ),
        )
        model = HardClusterModel(centers, active_clusters, method=EMA_MINI_BATCH_METHOD)
        diagnostics = ClusterDiagnostics(
            valid=valid,
            status=status,
            objective=objective,
            iterations=evidence.updates,
            effective_samples=effective,
            cluster_mass=mass,
            active_clusters=active_clusters,
            empty_clusters_seen=jnp.any(~active_clusters, axis=-1),
            converged=converged,
            method=EMA_MINI_BATCH_METHOD,
            minibatch=evidence,
        )
        contract = prediction_fit_contract(
            model._prediction_contract(),
            route=DerivativeRoute.STOPPED,
            nondifferentiable_outputs=(
                "assignments",
                "epoch permutations",
                "revival draws",
            ),
            conditions=("explicit random key", "fixed epoch budget"),
        )
        return FitResult(
            model,
            diagnostics,
            valid=valid,
            status=status,
            method=EMA_MINI_BATCH_METHOD,
            derivative_contract=contract,
        )


__all__ = ["MiniBatchKMeans", "epoch_fractions", "law_batch_size"]
