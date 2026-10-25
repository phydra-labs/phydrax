#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded adaptive bias histories: owners, cadence, ramp, deposition and freeze.

A history is fixed-capacity dynamic state.  Every force evaluation reads it
read-only; it changes only through :func:`deposit_bias_history` at a declared
committed boundary.  One buffer serves every ownership: each kernel carries an
owner tag (walker index for ``WALKER``, thermodynamic label for ``STATE``) and
a walker feels the kernels of its own key, or all kernels for ``SHARED``.
"""

from __future__ import annotations

import math
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import IdentityWords
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._alignment import aligned_deviation
from ._collective_variable import CollectiveVariableMetric
from ._cv_geometry import CollectiveVariableStatus, reduce_status


class BiasHistoryOwnership(StrEnum):
    SHARED = "shared"
    WALKER = "walker"
    STATE = "state"


class BiasHistoryStorage(StrEnum):
    """``TERMINAL``: exhaustion is a terminal transaction status (grow the
    capacity explicitly on the host).  ``FIFO``: the oldest kernel is
    overwritten; the usual metadynamics profile formula does not apply."""

    TERMINAL = "terminal"
    FIFO = "fifo"


class HillKernel(StrEnum):
    """``NEAREST_IMAGE``: source-compatible Gaussian of the metric difference
    (smooth except on the periodic seam).  ``WRAPPED``: explicitly truncated
    image sum with a reported tail bound.  ``ALIGNED_MSD``: fixed-correspondence
    configuration kernel ``exp(-MSD / (2 sigma^2))`` after optimal superposition.
    """

    NEAREST_IMAGE = "nearest-image-gaussian"
    WRAPPED = "wrapped-gaussian"
    ALIGNED_MSD = "aligned-msd-gaussian"


class AtomisticBiasHistoryPlan(StrictModule, NonTrainableState):
    """Scientific declaration of one adaptive kernel history.

    ``interval`` is the deposition cadence in committed steps, ``ramp_depositions``
    the number of deposition events over which a new kernel reaches its height,
    ``bias_factor`` the well-tempered ``gamma > 1`` (``None``: standard) with the
    history-owner ``temperature`` entering ``k_B T (gamma - 1)``.  ``owner_count``
    is the number of owner keys (walkers or thermodynamic states).

    PyTree: capacity, dimension, cadence (``interval``, ``ramp_depositions``),
    ownership/owner count, storage, kernel and image count are static method
    selectors and layout.  The physical scalars ``height``, ``bias_factor``
    and ``temperature`` are dynamic float64 scalars (the presence of
    ``bias_factor`` selects well-tempered deposition), and the exact
    ``scientific_id``/``history_id`` digests are the dynamic
    ``scientific_identity``/``history_identity`` words decoded on the host.
    """

    dimension: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    height: Array
    widths: Array
    interval: int = eqx.field(static=True)
    ramp_depositions: int = eqx.field(static=True)
    bias_factor: Array | None
    temperature: Array | None
    ownership: BiasHistoryOwnership = eqx.field(static=True)
    owner_count: int = eqx.field(static=True)
    storage: BiasHistoryStorage = eqx.field(static=True)
    kernel: HillKernel = eqx.field(static=True)
    wrapped_images: int = eqx.field(static=True)
    scientific_identity: IdentityWords
    history_identity: IdentityWords

    def __init__(
        self,
        dimension: int,
        /,
        *,
        height: float,
        widths: ArrayLike,
        capacity: int,
        interval: int = 1,
        ramp_depositions: int = 0,
        bias_factor: float | None = None,
        temperature: float | None = None,
        ownership: BiasHistoryOwnership = BiasHistoryOwnership.SHARED,
        owner_count: int = 1,
        storage: BiasHistoryStorage = BiasHistoryStorage.TERMINAL,
        kernel: HillKernel = HillKernel.NEAREST_IMAGE,
        wrapped_images: int = 0,
    ) -> None:
        size = int(dimension)
        capacity_ = int(capacity)
        height_ = float(height)
        width = np.asarray(widths, dtype=np.float64).reshape((-1,))
        if kernel is HillKernel.ALIGNED_MSD:
            if size % 3 or size < 9 or width.size != 1:
                raise ValueError(
                    "An aligned-MSD history stores flattened (n >= 3, 3) "
                    "configurations with one scalar width."
                )
        elif width.size == 1:
            width = np.full((size,), float(width[0]))
        if (
            size <= 0
            or capacity_ <= 0
            or not math.isfinite(height_)
            or height_ <= 0.0
            or np.any(~np.isfinite(width))
            or np.any(width <= 0.0)
            or (kernel is not HillKernel.ALIGNED_MSD and width.shape != (size,))
        ):
            raise ValueError(
                "History dimension, capacity, height and widths must be positive."
            )
        if int(interval) < 1 or int(ramp_depositions) < 0:
            raise ValueError("Deposition interval must be >= 1 and ramp >= 0.")
        if bias_factor is not None:
            gamma = float(bias_factor)
            if not math.isfinite(gamma) or gamma <= 1.0:
                raise ValueError("The well-tempered bias factor must exceed one.")
            if temperature is None or not float(temperature) > 0.0:
                raise ValueError(
                    "Well-tempered deposition requires the history-owner temperature."
                )
        owners = int(owner_count)
        if not isinstance(ownership, BiasHistoryOwnership) or owners < 1:
            raise ValueError("History ownership and owner count are invalid.")
        if ownership is BiasHistoryOwnership.SHARED and owners != 1:
            raise ValueError("A shared history has exactly one owner key.")
        if not isinstance(storage, BiasHistoryStorage) or not isinstance(
            kernel, HillKernel
        ):
            raise TypeError("storage and kernel must be typed history policies.")
        images = int(wrapped_images)
        if (kernel is HillKernel.WRAPPED) != (images > 0):
            raise ValueError("Only the wrapped kernel declares a positive image count.")
        self.dimension = size
        gamma = None if bias_factor is None else float(bias_factor)
        kelvin = None if temperature is None else float(temperature)
        self.capacity = capacity_
        self.height = jnp.asarray(height_, dtype=jnp.float64)
        self.widths = jnp.asarray(width)
        self.interval = int(interval)
        self.ramp_depositions = int(ramp_depositions)
        self.bias_factor = (
            None if gamma is None else jnp.asarray(gamma, dtype=jnp.float64)
        )
        self.temperature = (
            None if kelvin is None else jnp.asarray(kelvin, dtype=jnp.float64)
        )
        self.ownership = ownership
        self.owner_count = owners
        self.storage = storage
        self.kernel = kernel
        self.wrapped_images = images
        scientific_id = canonical_fingerprint(
            {
                "kind": "bias-history",
                "dimension": size,
                "height": height_.hex(),
                "widths": array_tree_fingerprint(width),
                "interval": self.interval,
                "ramp_depositions": self.ramp_depositions,
                "bias_factor": None if gamma is None else gamma.hex(),
                "temperature": None if kelvin is None else kelvin.hex(),
                "ownership": ownership.value,
                "owner_count": owners,
                "storage": storage.value,
                "kernel": kernel.value,
                "wrapped_images": images,
            }
        )
        self.scientific_identity = IdentityWords.from_fingerprint(scientific_id)
        self.history_identity = IdentityWords.of(
            {"scientific": scientific_id, "capacity": capacity_}
        )

    @property
    def scientific_id(self) -> str:
        """Host decoding of the capacity-free scientific history digest."""
        return self.scientific_identity.fingerprint()

    @property
    def history_id(self) -> str:
        """Host decoding of the history digest (scientific identity and capacity)."""
        return self.history_identity.fingerprint()

    def initialize(self, dtype: DTypeLike = jnp.float64) -> "AtomisticBiasHistoryState":
        capacity = self.capacity
        return AtomisticBiasHistoryState(
            centers=jnp.zeros((capacity, self.dimension), dtype=dtype),
            heights=jnp.zeros((capacity,), dtype=dtype),
            owners=jnp.full((capacity,), -1, dtype=jnp.int32),
            deposited_at=jnp.zeros((capacity,), dtype=jnp.int32),
            valid=jnp.zeros((capacity,), dtype=jnp.bool_),
            written=jnp.zeros((), dtype=jnp.int32),
            deposits=jnp.zeros((), dtype=jnp.int32),
            exhausted=jnp.zeros((), dtype=jnp.bool_),
            history_identity=self.history_identity,
        )

    def due(self, committed_step: ArrayLike, /) -> Array:
        """Deposition is due after committed step ``n`` with ``n % interval == 0``."""

        step = jnp.asarray(committed_step, dtype=jnp.int32)
        return (step > 0) & (jnp.mod(step, self.interval) == 0)

    def seam_kernel_bound(
        self, metrics: tuple[CollectiveVariableMetric, ...], /
    ) -> float:
        """Largest periodic-seam kernel value ``exp(-(P/2)^2 / (2 sigma^2))``.

        For the nearest-image kernel this bounds the relative force jump on the
        seam; for the wrapped kernel it is the first omitted image weight.
        """

        bound = 0.0
        widths = np.asarray(self.widths)
        for index, metric in enumerate(metrics):
            if metric.period is not None and self.kernel is not HillKernel.ALIGNED_MSD:
                half = 0.5 * metric.period + (
                    self.wrapped_images * metric.period
                    if self.kernel is HillKernel.WRAPPED
                    else 0.0
                )
                bound = max(bound, math.exp(-(half**2) / (2.0 * widths[index] ** 2)))
        return bound

    def evaluate(
        self,
        values: ArrayLike,
        metrics: tuple[CollectiveVariableMetric, ...],
        state: "AtomisticBiasHistoryState",
        owner_key: ArrayLike,
        /,
    ) -> "AtomisticBiasHistoryEvaluation":
        """History energy and the regularity evidence of its active kernels.

        A kernel is active when it is valid, selected by ``owner_key`` (every
        kernel for ``SHARED``) and carries a positive ramp weight.  For
        ``ALIGNED_MSD`` each kernel's single optimal superposition supplies
        both its value and its admission: an active kernel whose top
        key-matrix eigenvalue is not simple (or whose alignment is not
        finite) makes the evaluation unsuccessful, because its force branch
        is undefined.  An exact zero MSD is regular for the MSD kernel (its
        derivative exists), so ``ZERO_DISTANCE`` is not reported.  Inactive,
        padding and zero-ramp slots never enter the admission.
        """

        point = jnp.asarray(values)
        dtype = point.dtype
        if state.centers.shape[-2:] != (self.capacity, self.dimension):
            raise ValueError("History state does not match the plan's capacity layout.")
        bound = state.history_identity.equals(self.history_identity)
        live = state.valid
        if self.ownership is not BiasHistoryOwnership.SHARED:
            live = live & (state.owners == jnp.asarray(owner_key, dtype=jnp.int32))
        scale = self.ramp(state).astype(dtype)
        active = live & (scale > 0.0)
        status = jnp.zeros((), dtype=jnp.int32)
        regular = jnp.asarray(True)
        minimum_gap = None
        if self.kernel is HillKernel.ALIGNED_MSD:
            count = self.dimension // 3
            alignment = jax.vmap(
                lambda center: aligned_deviation(
                    point.reshape((count, 3)), center.reshape((count, 3))
                )
            )(state.centers.astype(dtype))
            weights = jnp.exp(-alignment.msd / (2.0 * self.widths[0].astype(dtype) ** 2))
            regular = jnp.all(~active | alignment.msd_regular)
            status = reduce_status(
                jnp.where(
                    active,
                    alignment.status & ~int(CollectiveVariableStatus.ZERO_DISTANCE),
                    0,
                )
            )
            minimum_gap = jnp.min(
                jnp.where(
                    active,
                    alignment.relative_gap,
                    jnp.asarray(jnp.inf, dtype=alignment.relative_gap.dtype),
                )
            )
        else:
            if len(metrics) != self.dimension:
                raise ValueError("History metrics must align with its dimension.")
            columns = []
            for index, metric in enumerate(metrics):
                width = self.widths[index].astype(dtype)
                delta = metric.difference(
                    point[index], state.centers[:, index].astype(dtype)
                )
                if self.kernel is HillKernel.WRAPPED and metric.period is not None:
                    shifts = jnp.arange(
                        -self.wrapped_images, self.wrapped_images + 1, dtype=dtype
                    ) * jnp.asarray(metric.period, dtype=dtype)
                    columns.append(
                        jnp.sum(
                            jnp.exp(
                                -0.5 * ((delta[:, None] + shifts[None, :]) / width) ** 2
                            ),
                            axis=-1,
                        )
                    )
                else:
                    columns.append(jnp.exp(-0.5 * (delta / width) ** 2))
            weights = jnp.prod(jnp.stack(columns, axis=-1), axis=-1)
        energy = jnp.sum(
            jnp.where(live, state.heights.astype(dtype) * scale * weights, 0.0)
        )
        # A state of another history plan has no defined energy here: NaN fails
        # the consuming term's finiteness status instead of a traced raise.
        return AtomisticBiasHistoryEvaluation(
            energy=jnp.where(bound, energy, jnp.asarray(jnp.nan, dtype=dtype)),
            successful=bound & regular,
            status=status,
            active_kernels=jnp.sum(active.astype(jnp.int32)),
            minimum_relative_gap=minimum_gap,
        )

    def ramp(self, state: "AtomisticBiasHistoryState", /) -> Array:
        if self.ramp_depositions <= 0:
            return jnp.ones_like(state.heights)
        age = (state.deposits - state.deposited_at).astype(state.heights.dtype)
        return jnp.clip(age / float(self.ramp_depositions), 0.0, 1.0)


class AtomisticBiasHistoryEvaluation(StrictModule):
    """One history energy with the admission evidence of its active kernels.

    ``status`` is the bitwise union of the active kernels' CV status words
    (zero for CV-space kernels), ``active_kernels`` their count and
    ``minimum_relative_gap`` the smallest relative key-matrix eigengap over
    active aligned kernels (``inf`` when none is active; ``None`` for
    CV-space kernels, which perform no alignment).
    """

    energy: Array
    successful: Array
    status: Array
    active_kernels: Array
    minimum_relative_gap: Array | None


class AtomisticBiasHistoryState(StrictModule):
    """Dynamic fixed-capacity kernel storage and its event counters.

    ``written`` counts every kernel ever inserted, ``deposits`` the deposition
    events (ramp ages), and ``exhausted`` latches a terminal capacity refusal.
    ``history_identity`` is the owning plan's exact ``history_id`` as dynamic
    words (``history_id`` decodes it on the host); it is not history content
    and is excluded from :func:`bias_history_digest`.
    """

    centers: Array
    heights: Array
    owners: Array
    deposited_at: Array
    valid: Array
    written: Array
    deposits: Array
    exhausted: Array
    history_identity: IdentityWords

    @property
    def history_id(self) -> str:
        """Host decoding of the owning history-plan digest."""
        return self.history_identity.fingerprint()


class AtomisticBiasDeposition(StrictModule):
    """All-or-nothing simultaneous deposition of one committed event.

    ``heights`` are computed against the pre-event history, inserted in
    stable walker order.  ``inserted`` is false when nothing was due/active,
    when terminal capacity refused the whole event (``exhausted``) or when the
    state is not bound to the plan (``successful`` false: nothing changes).
    """

    state: AtomisticBiasHistoryState
    heights: Array
    slots: Array
    inserted: Array
    exhausted: Array
    successful: Array


def deposit_bias_history(
    plan: AtomisticBiasHistoryPlan,
    state: AtomisticBiasHistoryState,
    values: ArrayLike,
    owner_keys: ArrayLike,
    active: ArrayLike,
    metrics: tuple[CollectiveVariableMetric, ...],
    /,
    *,
    boltzmann_constant: float,
) -> AtomisticBiasDeposition:
    """Insert one kernel per active walker at its current value.

    ``values`` has shape ``(walkers, dimension)``; ``owner_keys`` the owner tag
    each walker's kernel receives and is evaluated against.
    """

    if state.centers.shape[-2:] != (plan.capacity, plan.dimension):
        raise ValueError("History state does not match the plan's capacity layout.")
    bound = state.history_identity.equals(plan.history_identity)
    point = jnp.asarray(values)
    keys = jnp.asarray(owner_keys, dtype=jnp.int32)
    mask = jnp.asarray(active, dtype=jnp.bool_)
    walkers = point.shape[0]
    if point.shape != (walkers, plan.dimension) or keys.shape != (walkers,):
        raise ValueError("Deposition values and owner keys must be (walkers, dimension).")
    if plan.storage is BiasHistoryStorage.FIFO and plan.capacity < walkers:
        raise ValueError("A FIFO ring must hold at least one complete deposition.")
    dtype = state.heights.dtype
    pre = jax.vmap(lambda value, key: plan.evaluate(value, metrics, state, key).energy)(
        point, keys
    )
    if plan.bias_factor is None:
        heights = jnp.broadcast_to(plan.height.astype(dtype), (walkers,))
    else:
        temperature = plan.temperature
        if temperature is None:
            raise ValueError("Well-tempered deposition requires a temperature.")
        damping = boltzmann_constant * temperature * (plan.bias_factor - 1.0)
        heights = plan.height.astype(dtype) * jnp.exp(
            -pre.astype(dtype) / damping.astype(dtype)
        )
    # Explicit int32 accumulators: integer reductions otherwise promote to the
    # default integer (int64 under x64) and would drift the int32 counters.
    count = jnp.sum(mask, dtype=jnp.int32)
    rank = jnp.cumsum(mask, dtype=jnp.int32) - 1
    if plan.storage is BiasHistoryStorage.TERMINAL:
        refused = state.written + count > plan.capacity
        slots = state.written + rank
    else:
        refused = jnp.asarray(False)
        slots = jnp.mod(state.written + rank, plan.capacity)
    insert = mask & ~refused & bound
    target = jnp.where(insert, slots, plan.capacity)
    inserted = (count > 0) & ~refused & bound
    updated = AtomisticBiasHistoryState(
        centers=state.centers.at[target].set(point.astype(dtype), mode="drop"),
        heights=state.heights.at[target].set(heights, mode="drop"),
        owners=state.owners.at[target].set(keys, mode="drop"),
        deposited_at=state.deposited_at.at[target].set(state.deposits, mode="drop"),
        valid=state.valid.at[target].set(True, mode="drop"),
        written=state.written + jnp.where(inserted, count, 0),
        deposits=state.deposits + inserted.astype(jnp.int32),
        exhausted=state.exhausted | ((count > 0) & refused & bound),
        history_identity=state.history_identity,
    )
    return AtomisticBiasDeposition(
        state=updated,
        heights=heights,
        slots=jnp.where(insert, slots, -1),
        inserted=inserted,
        exhausted=updated.exhausted,
        successful=bound,
    )


def grow_bias_history(
    plan: AtomisticBiasHistoryPlan,
    state: AtomisticBiasHistoryState,
    capacity: int,
    /,
) -> tuple[AtomisticBiasHistoryPlan, AtomisticBiasHistoryState]:
    """Host capacity-ladder transaction: same kernels, owners and counters.

    Only terminal storage grows (a FIFO ring has no exhaustion).  The new plan
    keeps the scientific identity and receives a new capacity identity.
    """

    size = int(capacity)
    if plan.storage is not BiasHistoryStorage.TERMINAL:
        raise ValueError("Only terminal-storage histories grow by capacity ladder.")
    if state.history_id != plan.history_id or size <= plan.capacity:
        raise ValueError("Growth requires the bound history and a larger capacity.")
    grown = AtomisticBiasHistoryPlan(
        plan.dimension,
        height=float(np.asarray(plan.height)),
        widths=np.asarray(plan.widths),
        capacity=size,
        interval=plan.interval,
        ramp_depositions=plan.ramp_depositions,
        bias_factor=None
        if plan.bias_factor is None
        else float(np.asarray(plan.bias_factor)),
        temperature=None
        if plan.temperature is None
        else float(np.asarray(plan.temperature)),
        ownership=plan.ownership,
        owner_count=plan.owner_count,
        storage=plan.storage,
        kernel=plan.kernel,
        wrapped_images=plan.wrapped_images,
    )
    extra = size - plan.capacity

    def extend(value: Array, fill: float | int | bool) -> Array:
        pad = jnp.full((extra,) + value.shape[1:], fill, dtype=value.dtype)
        return jnp.concatenate((value, pad), axis=0)

    return grown, AtomisticBiasHistoryState(
        centers=extend(state.centers, 0.0),
        heights=extend(state.heights, 0.0),
        owners=extend(state.owners, -1),
        deposited_at=extend(state.deposited_at, 0),
        valid=extend(state.valid, False),
        written=state.written,
        deposits=state.deposits,
        exhausted=jnp.zeros((), dtype=jnp.bool_),
        history_identity=grown.history_identity,
    )


def _history_content(state: AtomisticBiasHistoryState, /) -> AtomisticBiasHistoryState:
    """Host arrays of a history's content: every leaf except its identity words.

    The returned tree keeps the original attribute paths, so its
    ``array_tree_fingerprint`` is the content fingerprint of the kernels and
    counters alone.
    """

    arrays = jax.tree.map(np.asarray, state)
    keep = eqx.tree_at(
        lambda value: value.history_identity.words,
        jax.tree.map(lambda _: True, arrays),
        False,
    )
    return eqx.filter(arrays, keep)


def bias_history_digest(state: AtomisticBiasHistoryState, /) -> str:
    """Content digest of one history: every kernel array and event counter.

    Host transaction (reads the arrays).  It is the identity a
    ``"frozen:<digest>"`` row declaration authenticates.
    """

    if not isinstance(state, AtomisticBiasHistoryState):
        raise TypeError("state must be AtomisticBiasHistoryState.")
    return canonical_fingerprint(
        {
            "kind": "frozen-bias-history",
            "history": state.history_id,
            "arrays": array_tree_fingerprint(_history_content(state)),
        }
    )


class FrozenBiasHistory(StrictModule, NonTrainableState):
    """Immutable authenticated copy of a history for analysis or warm starts.

    ``digest`` fingerprints every kernel array and counter
    (:func:`bias_history_digest`); ``declaration`` is the matching
    thermodynamic-row history declaration.  ``fifo_overwritten`` records
    kernels lost by FIFO storage; ``exhausted`` a terminal refusal.  The
    PyTree carries the digest as dynamic ``digest_identity`` words; the
    digest and the two flags are host properties of the copied content.
    """

    plan: AtomisticBiasHistoryPlan
    state: AtomisticBiasHistoryState
    digest_identity: IdentityWords

    def __init__(
        self, plan: AtomisticBiasHistoryPlan, state: AtomisticBiasHistoryState, /
    ) -> None:
        if state.history_id != plan.history_id:
            raise ValueError("History state belongs to another history plan.")
        arrays = jax.tree.map(np.asarray, state)
        self.plan = plan
        self.state = jax.tree.map(jnp.asarray, arrays)
        self.digest_identity = IdentityWords.from_fingerprint(bias_history_digest(state))

    @property
    def digest(self) -> str:
        """Host decoding of the authenticated history content digest."""
        return self.digest_identity.fingerprint()

    @property
    def fifo_overwritten(self) -> bool:
        """Host: whether FIFO storage overwrote kernels of this history."""
        written = int(np.asarray(self.state.written))
        return (
            self.plan.storage is BiasHistoryStorage.FIFO and written > self.plan.capacity
        )

    @property
    def exhausted(self) -> bool:
        """Host: whether the history latched a terminal capacity refusal."""
        return bool(np.asarray(self.state.exhausted))

    @property
    def declaration(self) -> str:
        """``"frozen:<digest>"`` for ``AtomisticBiasRowBinding(histories=...)``."""
        return f"frozen:{self.digest}"


def warm_start_bias_history(
    plan: AtomisticBiasHistoryPlan, frozen: FrozenBiasHistory, /
) -> AtomisticBiasHistoryState:
    """Declared new-run initial condition from a frozen history.

    The kernels and owners are copied with the ramp completed (every kernel at
    full height); counters restart so the run's own cadence is not confused
    with the source run.  This is not a continuation: resume restores the
    exact checkpointed state instead.
    """

    if frozen.plan.scientific_id != plan.scientific_id:
        raise ValueError("A warm start requires the same scientific history definition.")
    count = int(np.sum(np.asarray(frozen.state.valid)))
    if count > plan.capacity:
        raise ValueError("The warm-start history exceeds the new capacity.")
    valid = np.asarray(frozen.state.valid)
    order = np.argsort(np.asarray(frozen.state.deposited_at)[valid], kind="stable")
    source = {
        name: np.asarray(getattr(frozen.state, name))[valid][order]
        for name in ("centers", "heights", "owners")
    }
    state = plan.initialize(frozen.state.heights.dtype)
    target = np.arange(count)
    return AtomisticBiasHistoryState(
        centers=state.centers.at[target].set(source["centers"]),
        heights=state.heights.at[target].set(source["heights"]),
        owners=state.owners.at[target].set(source["owners"]),
        deposited_at=state.deposited_at,
        valid=state.valid.at[target].set(True),
        written=jnp.asarray(count, dtype=jnp.int32),
        deposits=jnp.asarray(plan.ramp_depositions, dtype=jnp.int32),
        exhausted=jnp.zeros((), dtype=jnp.bool_),
        history_identity=plan.history_identity,
    )


__all__ = [
    "AtomisticBiasDeposition",
    "AtomisticBiasHistoryEvaluation",
    "AtomisticBiasHistoryPlan",
    "AtomisticBiasHistoryState",
    "BiasHistoryOwnership",
    "BiasHistoryStorage",
    "FrozenBiasHistory",
    "HillKernel",
    "bias_history_digest",
    "deposit_bias_history",
    "grow_bias_history",
    "warm_start_bias_history",
]
