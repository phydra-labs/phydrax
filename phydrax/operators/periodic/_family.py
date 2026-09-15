#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical sparse translation families for periodic linear operators.

A family contains one directed edge relation, one integer lattice translation per
edge, and one numeric block per edge.  Host preparation owns sorting,
coalescing, reverse-map validation, and resource admission.  The evaluation,
action, derivative, and finite-realization actions are fixed-shape JAX kernels.
"""

from __future__ import annotations

from itertools import product
from math import isfinite, prod
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...sparse import EdgeRelation


class PeriodicResourceError(RuntimeError):
    """A periodic operation exceeds an explicitly admitted resource bound."""


class PeriodicFourierConvention(StrictModule, NonTrainableState):
    """Fractional reciprocal convention ``exp(sign i phase_scale q·R)``."""

    sign: int = eqx.field(static=True)
    phase_scale: float = eqx.field(static=True)
    convention_id: str = eqx.field(static=True)

    def __init__(self, sign: int = 1, phase_scale: float = 2.0 * np.pi, /):
        sign_ = int(sign)
        scale = float(phase_scale)
        if sign_ not in (-1, 1) or not isfinite(scale) or scale <= 0.0:
            raise ValueError("Fourier sign must be ±1 and phase_scale positive finite.")
        self.sign = sign_
        self.phase_scale = scale
        self.convention_id = canonical_fingerprint(
            {"kind": "periodic-fourier-convention", "sign": sign_, "phase_scale": scale}
        )


class PeriodicTranslationFamilyPlan(StrictModule, NonTrainableState):
    """Structural translation family and conservative hard resource policy."""

    relation: EdgeRelation
    translations: Array
    reverse_indices: Array
    convention: PeriodicFourierConvention
    hermitian: bool = eqx.field(static=True)
    maximum_dense_entries: int = eqx.field(static=True)
    maximum_finite_entries: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: EdgeRelation,
        translations: ArrayLike,
        reverse_indices: ArrayLike,
        /,
        *,
        convention: PeriodicFourierConvention | None = None,
        hermitian: bool = True,
        maximum_dense_entries: int = 4_000_000,
        maximum_finite_entries: int = 8_000_000,
    ):
        if not isinstance(relation, EdgeRelation):
            raise TypeError("relation must be phydrax.sparse.EdgeRelation.")
        translations_ = np.asarray(translations)
        reverse = np.asarray(reverse_indices)
        valid = np.asarray(relation.valid, dtype=bool)
        if (
            translations_.ndim != 2
            or translations_.shape[0] != relation.capacity
            or translations_.shape[1] not in (1, 2, 3)
            or not np.issubdtype(translations_.dtype, np.integer)
            or reverse.shape != (relation.capacity,)
            or not np.issubdtype(reverse.dtype, np.integer)
            or not np.any(valid)
        ):
            raise ValueError(
                "Periodic translations/reverse map must match a nonempty EdgeRelation capacity."
            )
        translations_ = translations_.astype(np.int32, copy=False)
        reverse = reverse.astype(np.int32, copy=False)
        valid_indices = np.flatnonzero(valid)
        if (
            np.any(reverse[valid] < 0)
            or np.any(reverse[valid] >= relation.capacity)
            or np.any(~valid[reverse[valid]])
            or not np.array_equal(reverse[reverse[valid]], valid_indices)
        ):
            raise ValueError("Valid periodic reverse routes must form an involution.")
        source = np.asarray(relation.source_indices)
        target = np.asarray(relation.target_indices)
        if (
            not np.array_equal(source[reverse[valid]], target[valid])
            or not np.array_equal(target[reverse[valid]], source[valid])
            or not np.array_equal(translations_[reverse[valid]], -translations_[valid])
        ):
            raise ValueError(
                "Valid reverse edges must exchange endpoints and carry opposite translations."
            )
        keys = np.concatenate(
            (translations_[valid], source[valid, None], target[valid, None]), axis=1
        )
        if np.unique(keys, axis=0).shape[0] != keys.shape[0]:
            raise ValueError("Duplicate valid periodic edges must be coalesced.")
        hermitian_ = bool(hermitian)
        if hermitian_ and relation.source_size != relation.target_size:
            raise ValueError("Hermitian families require equal source and target sizes.")
        dense_limit = int(maximum_dense_entries)
        finite_limit = int(maximum_finite_entries)
        if dense_limit <= 0 or finite_limit <= 0:
            raise ValueError("Periodic dense and finite entry limits must be positive.")
        convention_ = PeriodicFourierConvention() if convention is None else convention
        if not isinstance(convention_, PeriodicFourierConvention):
            raise TypeError("convention must be PeriodicFourierConvention.")
        self.relation = relation
        self.translations = jnp.asarray(translations_, dtype=jnp.int32)
        self.reverse_indices = jnp.asarray(reverse, dtype=jnp.int32)
        self.convention = convention_
        self.hermitian = hermitian_
        self.maximum_dense_entries = dense_limit
        self.maximum_finite_entries = finite_limit
        self.rank = int(translations_.shape[1])
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-translation-family-plan",
                "relation": {
                    "source_size": relation.source_size,
                    "target_size": relation.target_size,
                    "arrays": array_tree_fingerprint(
                        {
                            "source": source,
                            "target": target,
                            "valid": valid,
                        }
                    ),
                },
                "reverse": array_tree_fingerprint(reverse),
                "convention": convention_.convention_id,
                "hermitian": hermitian_,
                "maximum_dense_entries": dense_limit,
                "maximum_finite_entries": finite_limit,
                "translations": array_tree_fingerprint(translations_),
            }
        )


class PeriodicTranslationFamilyState(StrictModule, NonTrainableState):
    """Numeric block values for one translation-family plan."""

    values: Array
    plan_id: str = eqx.field(static=True)
    numeric_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PeriodicTranslationFamilyPlan,
        values: ArrayLike,
        /,
        *,
        hermiticity_tolerance: float = 1.0e-10,
    ):
        if not isinstance(plan, PeriodicTranslationFamilyPlan):
            raise TypeError("plan must be PeriodicTranslationFamilyPlan.")
        value = np.asarray(values)
        if (
            value.ndim != 3
            or value.shape[0] != plan.relation.capacity
            or value.shape[1] <= 0
            or value.shape[2] <= 0
            or np.any(~np.isfinite(value))
        ):
            raise ValueError(
                "Periodic edge values must be finite with shape (capacity, out, in)."
            )
        valid = np.asarray(plan.relation.valid, dtype=bool)
        if np.any(value[~valid] != 0):
            raise ValueError(
                "Invalid periodic relation routes must carry zero block values."
            )
        tolerance = float(hermiticity_tolerance)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("hermiticity_tolerance must be finite and non-negative.")
        if plan.hermitian:
            if value.shape[1] != value.shape[2]:
                raise ValueError("Hermitian families require square edge blocks.")
            reverse = np.asarray(plan.reverse_indices)
            defect = np.max(
                np.abs(
                    value[reverse[valid]] - np.conj(np.swapaxes(value[valid], -1, -2))
                ),
                initial=0.0,
            )
            scale = max(float(np.max(np.abs(value[valid]), initial=0.0)), 1.0)
            if defect > tolerance * scale:
                raise ValueError(
                    "Reverse-edge values do not satisfy Hermitian adjoint symmetry."
                )
        self.values = jnp.asarray(value)
        self.plan_id = plan.plan_id
        self.numeric_id = canonical_fingerprint(
            {
                "kind": "periodic-translation-family-state",
                "plan": plan.plan_id,
                "values": array_tree_fingerprint(value),
            }
        )

    @property
    def output_block_size(self) -> int:
        return int(self.values.shape[1])

    @property
    def input_block_size(self) -> int:
        return int(self.values.shape[2])


class PreparedPeriodicTranslationFamily(StrictModule, NonTrainableState):
    """Fixed-shape execution object for a canonical sparse family."""

    plan: PeriodicTranslationFamilyPlan
    state: PeriodicTranslationFamilyState
    row_indices: Array
    column_indices: Array
    output_size: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PeriodicTranslationFamilyPlan,
        state: PeriodicTranslationFamilyState,
        /,
    ):
        if not isinstance(plan, PeriodicTranslationFamilyPlan) or not isinstance(
            state, PeriodicTranslationFamilyState
        ):
            raise TypeError("Prepared families require typed plan and state.")
        if state.plan_id != plan.plan_id:
            raise ValueError(
                "Periodic family state belongs to a different structural plan."
            )
        output_block = state.output_block_size
        input_block = state.input_block_size
        target = np.asarray(plan.relation.target_indices)
        source = np.asarray(plan.relation.source_indices)
        row = target[:, None] * output_block + np.arange(output_block)[None, :]
        column = source[:, None] * input_block + np.arange(input_block)[None, :]
        self.plan = plan
        self.state = state
        self.row_indices = jnp.asarray(row, dtype=jnp.int32)
        self.column_indices = jnp.asarray(column, dtype=jnp.int32)
        self.output_size = plan.relation.target_size * output_block
        self.input_size = plan.relation.source_size * input_block
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-periodic-translation-family",
                "plan": plan.plan_id,
                "numeric": state.numeric_id,
            }
        )

    def apply(self, fractional_points: ArrayLike, vectors: ArrayLike, /) -> Array:
        return apply_periodic_translation_family(self, fractional_points, vectors)

    def evaluate(self, fractional_points: ArrayLike, /) -> Array:
        return evaluate_periodic_translation_family(self, fractional_points)

    def derivative(self, fractional_points: ArrayLike, /, *, order: int = 1) -> Array:
        return differentiate_periodic_translation_family(
            self, fractional_points, order=order
        )


class PeriodicFiniteRealization(StrictModule, NonTrainableState):
    """Exact finite sparse realization with explicit COO storage and action."""

    row_indices: Array
    column_indices: Array
    values: Array
    supercell_shape: tuple[int, ...] = eqx.field(static=True)
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    twists: tuple[float, ...] = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    maximum_dense_entries: int = eqx.field(static=True)
    source_prepared_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        row_indices: ArrayLike,
        column_indices: ArrayLike,
        values: ArrayLike,
        /,
        *,
        supercell_shape: tuple[int, ...],
        periodic_axes: tuple[bool, ...],
        twists: tuple[float, ...],
        output_size: int,
        input_size: int,
        maximum_dense_entries: int,
        source_prepared_id: str,
    ):
        row = np.asarray(row_indices, dtype=np.int64)
        column = np.asarray(column_indices, dtype=np.int64)
        value = np.asarray(values)
        if row.ndim != 1 or column.shape != row.shape or value.shape != row.shape:
            raise ValueError(
                "Finite COO rows, columns, and values must have shape (entry,)."
            )
        if np.any(~np.isfinite(value)):
            raise ValueError("Finite realization values must be finite.")
        self.row_indices = jnp.asarray(row, dtype=jnp.int32)
        self.column_indices = jnp.asarray(column, dtype=jnp.int32)
        self.values = jnp.asarray(value)
        self.supercell_shape = tuple(int(v) for v in supercell_shape)
        self.periodic_axes = tuple(bool(v) for v in periodic_axes)
        self.twists = tuple(float(v) for v in twists)
        self.output_size = int(output_size)
        self.input_size = int(input_size)
        self.maximum_dense_entries = int(maximum_dense_entries)
        self.source_prepared_id = str(source_prepared_id)
        self.realization_id = canonical_fingerprint(
            {
                "kind": "periodic-finite-realization",
                "prepared": self.source_prepared_id,
                "shape": list(self.supercell_shape),
                "periodic_axes": list(self.periodic_axes),
                "twists": list(self.twists),
                "arrays": array_tree_fingerprint(
                    {"row": row, "column": column, "values": value}
                ),
            }
        )

    @property
    def entry_count(self) -> int:
        return int(self.values.size)

    def apply(self, vectors: ArrayLike, /) -> Array:
        vector = jnp.asarray(vectors)
        if vector.shape[-1] != self.input_size:
            raise ValueError("Finite-realization vectors must end in input_size.")
        flat = vector.reshape((-1, self.input_size))
        contribution = flat[:, self.column_indices] * self.values[None, :]
        output = jnp.zeros((flat.shape[0], self.output_size), dtype=contribution.dtype)
        output = output.at[:, self.row_indices].add(contribution)
        return output.reshape(vector.shape[:-1] + (self.output_size,))

    def to_dense(self, /) -> Array:
        entries = self.output_size * self.input_size
        if entries > self.maximum_dense_entries:
            raise PeriodicResourceError(
                "Finite dense materialization exceeds maximum_dense_entries."
            )
        matrix = jnp.zeros((self.output_size, self.input_size), dtype=self.values.dtype)
        return matrix.at[self.row_indices, self.column_indices].add(self.values)


def prepare_periodic_translation_family(
    plan: PeriodicTranslationFamilyPlan,
    state: PeriodicTranslationFamilyState,
    /,
) -> PreparedPeriodicTranslationFamily:
    return PreparedPeriodicTranslationFamily(plan, state)


def refresh_periodic_translation_family(
    prepared: PreparedPeriodicTranslationFamily,
    state: PeriodicTranslationFamilyState,
    /,
) -> PreparedPeriodicTranslationFamily:
    if not isinstance(prepared, PreparedPeriodicTranslationFamily):
        raise TypeError("prepared must be PreparedPeriodicTranslationFamily.")
    if state.plan_id != prepared.plan.plan_id:
        raise ValueError("Refresh cannot change translation-family structure.")
    return PreparedPeriodicTranslationFamily(prepared.plan, state)


def _fractional_points(
    prepared: PreparedPeriodicTranslationFamily, points: ArrayLike
) -> Array:
    if not isinstance(prepared, PreparedPeriodicTranslationFamily):
        raise TypeError("prepared must be PreparedPeriodicTranslationFamily.")
    point = jnp.asarray(points)
    if point.ndim == 1:
        point = point[None, :]
    if point.ndim != 2 or point.shape[1] != prepared.plan.rank:
        raise ValueError("Fractional points must have shape (Q, family rank).")
    return point


def _edge_phases(prepared: PreparedPeriodicTranslationFamily, points: Array) -> Array:
    arguments = contract(
        "qr,er->qe",
        points,
        prepared.plan.translations.astype(points.dtype),
        backend="jax",
    )
    scale = prepared.plan.convention.sign * prepared.plan.convention.phase_scale
    return jnp.exp(1.0j * scale * arguments)


def apply_periodic_translation_family(
    prepared: PreparedPeriodicTranslationFamily,
    fractional_points: ArrayLike,
    vectors: ArrayLike,
    /,
) -> Array:
    """Apply without materializing a dense Bloch matrix."""

    points = _fractional_points(prepared, fractional_points)
    vector = jnp.asarray(vectors)
    if vector.ndim == 1:
        vector = jnp.broadcast_to(vector, (points.shape[0], vector.shape[0]))
    if vector.shape != (points.shape[0], prepared.input_size):
        raise ValueError("Bloch vectors must have shape (Q, family input_size).")
    node_vectors = vector.reshape(
        (
            points.shape[0],
            prepared.plan.relation.source_size,
            prepared.state.input_block_size,
        )
    )
    gathered = node_vectors[:, prepared.plan.relation.source_indices, :]
    edge_action = contract("eoi,qei->qeo", prepared.state.values, gathered, backend="jax")
    edge_action = (
        _edge_phases(prepared, points)[..., None]
        * edge_action
        * prepared.plan.relation.valid[None, :, None]
    )
    output = jnp.zeros(
        (
            points.shape[0],
            prepared.plan.relation.target_size,
            prepared.state.output_block_size,
        ),
        dtype=edge_action.dtype,
    )
    output = output.at[:, prepared.plan.relation.target_indices, :].add(edge_action)
    return output.reshape((points.shape[0], prepared.output_size))


def evaluate_periodic_translation_family(
    prepared: PreparedPeriodicTranslationFamily,
    fractional_points: ArrayLike,
    /,
) -> Array:
    """Explicit bounded dense Bloch matrices; never used by sparse action."""

    points = _fractional_points(prepared, fractional_points)
    required = int(points.shape[0]) * prepared.output_size * prepared.input_size
    if required > prepared.plan.maximum_dense_entries:
        raise PeriodicResourceError(
            "Bloch dense evaluation exceeds maximum_dense_entries."
        )
    phases = _edge_phases(prepared, points)
    block_values = (
        phases[..., None, None]
        * prepared.state.values[None, ...]
        * prepared.plan.relation.valid[None, :, None, None]
    )
    block = jnp.zeros(
        (
            points.shape[0],
            prepared.plan.relation.target_size,
            prepared.state.output_block_size,
            prepared.plan.relation.source_size,
            prepared.state.input_block_size,
        ),
        dtype=block_values.dtype,
    )
    block = block.at[
        :,
        prepared.plan.relation.target_indices,
        :,
        prepared.plan.relation.source_indices,
        :,
    ].add(jnp.swapaxes(block_values, 0, 1))
    return block.reshape((points.shape[0], prepared.output_size, prepared.input_size))


def differentiate_periodic_translation_family(
    prepared: PreparedPeriodicTranslationFamily,
    fractional_points: ArrayLike,
    /,
    *,
    order: int = 1,
) -> Array:
    """First or second derivative with respect to fractional reciprocal coordinates."""

    if (
        isinstance(order, bool)
        or not isinstance(order, Integral)
        or int(order) not in (1, 2)
    ):
        raise ValueError("Periodic family derivative order must be 1 or 2.")
    points = _fractional_points(prepared, fractional_points)
    rank_factor = prepared.plan.rank ** int(order)
    required = (
        int(points.shape[0]) * rank_factor * prepared.output_size * prepared.input_size
    )
    if required > prepared.plan.maximum_dense_entries:
        raise PeriodicResourceError(
            "Bloch derivative evaluation exceeds maximum_dense_entries."
        )
    phases = _edge_phases(prepared, points)
    scale = 1.0j * prepared.plan.convention.sign * prepared.plan.convention.phase_scale
    translation = prepared.plan.translations.astype(points.dtype)
    if int(order) == 1:
        factors = phases[:, :, None] * scale * translation[None, :, :]
        values = factors[..., None, None] * prepared.state.values[None, :, None, :, :]
        block = jnp.zeros(
            (
                points.shape[0],
                prepared.plan.rank,
                prepared.plan.relation.target_size,
                prepared.state.output_block_size,
                prepared.plan.relation.source_size,
                prepared.state.input_block_size,
            ),
            dtype=values.dtype,
        )
        block = block.at[
            :,
            :,
            prepared.plan.relation.target_indices,
            :,
            prepared.plan.relation.source_indices,
            :,
        ].add(jnp.transpose(values, (1, 0, 2, 3, 4)))
        return block.reshape(
            (
                points.shape[0],
                prepared.plan.rank,
                prepared.output_size,
                prepared.input_size,
            )
        )
    factors = (
        phases[:, :, None, None]
        * (scale**2)
        * translation[None, :, :, None]
        * translation[None, :, None, :]
    )
    values = factors[..., None, None] * prepared.state.values[None, :, None, None, :, :]
    block = jnp.zeros(
        (
            points.shape[0],
            prepared.plan.rank,
            prepared.plan.rank,
            prepared.plan.relation.target_size,
            prepared.state.output_block_size,
            prepared.plan.relation.source_size,
            prepared.state.input_block_size,
        ),
        dtype=values.dtype,
    )
    block = block.at[
        :,
        :,
        :,
        prepared.plan.relation.target_indices,
        :,
        prepared.plan.relation.source_indices,
        :,
    ].add(jnp.transpose(values, (1, 0, 2, 3, 4, 5)))
    return block.reshape(
        (
            points.shape[0],
            prepared.plan.rank,
            prepared.plan.rank,
            prepared.output_size,
            prepared.input_size,
        )
    )


def coalesce_periodic_translation_family(
    source_indices: ArrayLike,
    target_indices: ArrayLike,
    translations: ArrayLike,
    values: ArrayLike,
    node_count: int,
    /,
    *,
    convention: PeriodicFourierConvention | None = None,
    hermitian: bool = True,
    hermiticity_tolerance: float = 1.0e-10,
    maximum_edges: int = 1_000_000,
    maximum_dense_entries: int = 4_000_000,
    maximum_finite_entries: int = 8_000_000,
) -> PreparedPeriodicTranslationFamily:
    """Host-sort and sum duplicate edges, then construct the canonical family."""

    source = np.asarray(source_indices, dtype=np.int64)
    target = np.asarray(target_indices, dtype=np.int64)
    translation = np.asarray(translations)
    value = np.asarray(values)
    if (
        source.ndim != 1
        or target.shape != source.shape
        or translation.ndim != 2
        or translation.shape[0] != source.size
        or value.ndim != 3
        or value.shape[0] != source.size
    ):
        raise ValueError(
            "Raw periodic edges, translations, and block values do not align."
        )
    if not np.issubdtype(translation.dtype, np.integer):
        raise ValueError("Raw periodic translations must be integers.")
    keys = [
        (
            tuple(int(v) for v in translation[index]),
            int(source[index]),
            int(target[index]),
        )
        for index in range(source.size)
    ]
    accumulated: dict[tuple[tuple[int, ...], int, int], np.ndarray] = {}
    for key, block in zip(keys, value, strict=True):
        accumulated[key] = accumulated.get(key, np.zeros_like(block)) + block
    ordered = sorted(accumulated)
    if len(ordered) > int(maximum_edges):
        raise PeriodicResourceError("Coalesced family exceeds maximum_edges.")
    key_to_index = {key: index for index, key in enumerate(ordered)}
    reverse = []
    for translation_key, source_index, target_index in ordered:
        reverse_key = (
            tuple(-component for component in translation_key),
            target_index,
            source_index,
        )
        if reverse_key not in key_to_index:
            raise ValueError("Periodic edges are missing an explicit reverse partner.")
        reverse.append(key_to_index[reverse_key])
    relation = EdgeRelation(
        [key[1] for key in ordered],
        [key[2] for key in ordered],
        source_size=int(node_count),
        target_size=int(node_count),
    )
    plan = PeriodicTranslationFamilyPlan(
        relation,
        [key[0] for key in ordered],
        reverse,
        convention=convention,
        hermitian=hermitian,
        maximum_dense_entries=maximum_dense_entries,
        maximum_finite_entries=maximum_finite_entries,
    )
    state = PeriodicTranslationFamilyState(
        plan,
        np.stack([accumulated[key] for key in ordered]),
        hermiticity_tolerance=hermiticity_tolerance,
    )
    return PreparedPeriodicTranslationFamily(plan, state)


def periodic_translation_family_from_dense_blocks(
    translations: ArrayLike,
    blocks: ArrayLike,
    /,
    *,
    convention: PeriodicFourierConvention | None = None,
    hermitian: bool = True,
    hermiticity_tolerance: float = 1.0e-10,
    maximum_edges: int = 1_000_000,
    maximum_dense_entries: int = 4_000_000,
    maximum_finite_entries: int = 8_000_000,
) -> PreparedPeriodicTranslationFamily:
    """Lower dense translation blocks once; all subsequent Fourier work is canonical."""

    translation = np.asarray(translations)
    block = np.asarray(blocks)
    if (
        translation.ndim != 2
        or block.ndim != 5
        or block.shape[0] != translation.shape[0]
        or block.shape[1] != block.shape[3]
        or block.shape[2] <= 0
        or block.shape[4] <= 0
    ):
        raise ValueError(
            "Dense translation blocks require (R, rank) and (R, target, out, source, in)."
        )
    translation_edges = np.repeat(translation, block.shape[1] * block.shape[3], axis=0)
    target = np.tile(np.repeat(np.arange(block.shape[1]), block.shape[3]), block.shape[0])
    source = np.tile(np.tile(np.arange(block.shape[3]), block.shape[1]), block.shape[0])
    values = np.transpose(block, (0, 1, 3, 2, 4)).reshape(
        (-1, block.shape[2], block.shape[4])
    )
    return coalesce_periodic_translation_family(
        source,
        target,
        translation_edges,
        values,
        block.shape[1],
        convention=convention,
        hermitian=hermitian,
        hermiticity_tolerance=hermiticity_tolerance,
        maximum_edges=maximum_edges,
        maximum_dense_entries=maximum_dense_entries,
        maximum_finite_entries=maximum_finite_entries,
    )


def realize_periodic_translation_family(
    prepared: PreparedPeriodicTranslationFamily,
    supercell_shape: tuple[int, ...],
    /,
    *,
    periodic_axes: tuple[bool, ...] | None = None,
    twists: tuple[float, ...] | None = None,
) -> PeriodicFiniteRealization:
    """Realize open, periodic, or twisted finite boundaries without dense storage."""

    if not isinstance(prepared, PreparedPeriodicTranslationFamily):
        raise TypeError("prepared must be PreparedPeriodicTranslationFamily.")
    shape = tuple(int(value) for value in supercell_shape)
    if len(shape) != prepared.plan.rank or any(value <= 0 for value in shape):
        raise ValueError("supercell_shape must be positive and match family rank.")
    axes = (
        (False,) * prepared.plan.rank
        if periodic_axes is None
        else tuple(bool(value) for value in periodic_axes)
    )
    twist = (
        (0.0,) * prepared.plan.rank
        if twists is None
        else tuple(float(value) for value in twists)
    )
    if len(axes) != prepared.plan.rank or len(twist) != prepared.plan.rank:
        raise ValueError("Boundary axes and twists must match family rank.")
    if any(not isfinite(value) for value in twist):
        raise ValueError("Boundary twists must be finite.")
    if any(
        value != 0.0 and not periodic for value, periodic in zip(twist, axes, strict=True)
    ):
        raise ValueError("Nonzero twist requires a periodic finite axis.")
    cell_count = prod(shape)
    upper_entries = (
        cell_count
        * prepared.plan.relation.capacity
        * prepared.state.output_block_size
        * prepared.state.input_block_size
    )
    if upper_entries > prepared.plan.maximum_finite_entries:
        raise PeriodicResourceError(
            "Finite realization exceeds maximum_finite_entries before allocation."
        )
    source_edge = np.asarray(prepared.plan.relation.source_indices)
    target_edge = np.asarray(prepared.plan.relation.target_indices)
    translations = np.asarray(prepared.plan.translations)
    blocks = np.asarray(prepared.state.values)
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    output_nodes = prepared.plan.relation.target_size * prepared.state.output_block_size
    input_nodes = prepared.plan.relation.source_size * prepared.state.input_block_size
    relation_valid = np.asarray(prepared.plan.relation.valid, dtype=bool)
    for target_cell in product(*(range(extent) for extent in shape)):
        target_linear = int(np.ravel_multi_index(target_cell, shape))
        for edge in range(prepared.plan.relation.capacity):
            if not relation_valid[edge]:
                continue
            raw_source = np.asarray(target_cell, dtype=np.int64) + translations[edge]
            source_cell = raw_source.copy()
            winding = np.zeros((prepared.plan.rank,), dtype=np.int64)
            admissible = True
            for axis, extent in enumerate(shape):
                if 0 <= raw_source[axis] < extent:
                    continue
                if not axes[axis]:
                    admissible = False
                    break
                winding[axis], source_cell[axis] = divmod(int(raw_source[axis]), extent)
            if not admissible:
                continue
            source_linear = int(np.ravel_multi_index(tuple(source_cell), shape))
            phase = np.exp(
                1.0j
                * prepared.plan.convention.sign
                * float(np.dot(np.asarray(twist), winding))
            )
            for output_component in range(prepared.state.output_block_size):
                row = (
                    target_linear * output_nodes
                    + int(target_edge[edge]) * prepared.state.output_block_size
                    + output_component
                )
                for input_component in range(prepared.state.input_block_size):
                    column = (
                        source_linear * input_nodes
                        + int(source_edge[edge]) * prepared.state.input_block_size
                        + input_component
                    )
                    rows.append(row)
                    columns.append(column)
                    values.append(phase * blocks[edge, output_component, input_component])
    value_array = np.asarray(values, dtype=np.result_type(blocks.dtype, np.complex64))
    return PeriodicFiniteRealization(
        rows,
        columns,
        value_array,
        supercell_shape=shape,
        periodic_axes=axes,
        twists=twist,
        output_size=cell_count * output_nodes,
        input_size=cell_count * input_nodes,
        maximum_dense_entries=prepared.plan.maximum_dense_entries,
        source_prepared_id=prepared.prepared_id,
    )


__all__ = [
    "PeriodicFiniteRealization",
    "PeriodicFourierConvention",
    "PeriodicResourceError",
    "PeriodicTranslationFamilyPlan",
    "PeriodicTranslationFamilyState",
    "PreparedPeriodicTranslationFamily",
    "apply_periodic_translation_family",
    "coalesce_periodic_translation_family",
    "differentiate_periodic_translation_family",
    "evaluate_periodic_translation_family",
    "periodic_translation_family_from_dense_blocks",
    "prepare_periodic_translation_family",
    "realize_periodic_translation_family",
    "refresh_periodic_translation_family",
]
