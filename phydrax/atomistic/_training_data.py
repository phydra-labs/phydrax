#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomistic training-data composition over immutable structure sources.

`AtomisticSourceMixture` binds the canonical component order of immutable
`AbstractAtomisticStructureSource` values, their draw probabilities and the
intended sampled or importance-corrected objective. One global draw sequence per
epoch is defined by the shared `IndexEpochPlan` (complete epochs of an integer
multiset membership) or by semantic `derive_key` addresses (independent draws
with replacement); draw identities are epoch positions, never batch slots or
placement. `AtomisticStructureLoader` cuts every global batch into the
process/device shards of `DistributedIndexEpochPlan`, routes each draw to the
first declared `AtomisticWorkloadBucket` that admits its record, packs each
shard deterministically under that bucket's explicit case and per-level row
capacities, and pads every payload to the bucket's static layout. Padding reuses
real records and real rows and carries false validity; it never invents a sample
or reads through an invalid index. Host provenance (record, lineage and source
identities) travels beside the numerical payload, never inside its static
layout. `AtomisticLoaderCursor` binds source content, mixture, bindings, bucket
plan and placement together with the epoch, batch and in-batch payload position.
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import TracebackType
from typing import Any, assert_never, final, Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from .._data_plane import (
    BoundedPrefetchIterator,
    DistributedIndexEpochPlan,
    EPOCH_ORDER_ALGORITHM,
    ExecutionGroupRank,
    IndexEpochPlan,
)
from .._execution_array import shard_array_axis
from .._execution_pool import semantic_task_keys
from .._execution_runtime import ExecutionGroup
from .._fingerprint import canonical_fingerprint
from .._identity import ExecutableSignature
from .._jax_config import default_device, default_prng_implementation, x64_enabled
from .._sampling._addressing import SampleAddress
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..axes import Axis
from ..typing import parse
from ..units import conversion_factor, UnitDefinition
from ._structure_store import (
    AbstractAtomisticStructureSource,
    AtomisticStoreDType,
    AtomisticStoreFieldRole,
    AtomisticStoreLevelKind,
    AtomisticStoreValueKind,
    AtomisticStructureBatch,
    AtomisticStructureStoreSchema,
)


if TYPE_CHECKING:
    from ..lifecycle import CompositionDependency, CompositionEntry


type AtomisticMixtureDraw = Literal["epoch", "replacement"]
type AtomisticMixtureObjective = Literal["sampled", "importance-corrected"]
type AtomisticLevelLayout = Literal["per-case", "packed"]
type AtomisticLoaderRemainder = Literal["short", "drop"]
type AtomisticLoaderTarget = Literal["host", "device"]
type AtomisticLoaderFailure = Literal[
    "binding", "capacity", "mixture", "cursor", "drift", "leakage", "placement"
]
type _Buffer = Array | npt.NDArray[Any]

_NAME = re.compile(r"[a-z][a-z0-9_]{0,62}")
_UINT32_LIMIT = 1 << 32
_INT32_LIMIT = 1 << 31
_POPULATION_LIMIT = 1 << 31
# Semantic RNG families: independent draws of a replacement mixture are addressed
# by (epoch, draw); per-draw augmentation keys of a payload by (draw, epoch).
_REPLACEMENT_ADDRESS = SampleAddress("atomistic", "mixture-replacement", role="draw")
_DRAW_ADDRESS = SampleAddress("atomistic", "loader-draw", role="sample")
_WIDE_DTYPES = frozenset(
    np.dtype(name) for name in ("float64", "int64", "uint64", "complex128")
)


class AtomisticLoaderError(ValueError):
    """Fail-closed loader error with a stable failure category.

    ``capacity`` reports a record or payload that exceeds declared bounds and is
    distinct from ``binding`` (unit, level, convention or dtype contracts),
    ``mixture`` (probabilities, objectives and membership), ``cursor``
    (policy/layout identity), ``drift`` (changed source content), ``leakage``
    (shared lineage or records across splits) and ``placement``.
    """

    failure: AtomisticLoaderFailure

    def __init__(self, failure: AtomisticLoaderFailure, message: str, /) -> None:
        self.failure = parse(failure, AtomisticLoaderFailure, "failure")
        super().__init__(str(message))


def _name(value: str, label: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if _NAME.fullmatch(value) is None:
        raise ValueError(f"{label} must match {_NAME.pattern}.")
    return value


def _text(value: str, label: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string.")
    if not value or value.strip() != value:
        raise ValueError(f"{label} must be a non-empty stripped string.")
    return value


def _integer(value: int, label: str, /, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer.")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return value


def _frozen[T: np.generic](array: npt.NDArray[T], /) -> npt.NDArray[T]:
    array.setflags(write=False)
    return array


def _axis_payload(axis: Axis, /) -> dict[str, Any]:
    return {
        "scope": axis.key.scope,
        "name": axis.key.name,
        "size": axis.size,
        "labels": list(axis.labels),
        "support_id": axis.support_id,
    }


def _exclusive_cumsum(counts: npt.NDArray[np.int64], /) -> npt.NDArray[np.int64]:
    return np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))[:-1]


def _unit_interval(
    high: npt.NDArray[np.uint64], low: npt.NDArray[np.uint64], /
) -> npt.NDArray[np.float64]:
    """Exact 53-bit uniform values in ``[0, 1)`` from two 32-bit words."""

    mantissa = (high >> np.uint64(5)) * np.uint64(1 << 26) + (low >> np.uint64(6))
    return mantissa.astype(np.float64) / float(1 << 53)


# ------------------------------------------------------------------- mixtures


@final
class AtomisticMixtureComponent(StrictModule, NonTrainableState):
    """One immutable source in a mixture with its nonnegative draw weight.

    Under ``draw="epoch"`` the weight is the integer number of times every
    record of the source appears in each complete epoch (``0`` excludes it).
    Under ``draw="replacement"`` it is the relative probability mass of the
    source; records are uniform within it.
    """

    source: AbstractAtomisticStructureSource
    weight: int | float = eqx.field(static=True)

    def __init__(
        self, source: AbstractAtomisticStructureSource, /, *, weight: int | float = 1
    ) -> None:
        if not isinstance(source, AbstractAtomisticStructureSource):
            raise TypeError("source must be an AbstractAtomisticStructureSource.")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("weight must be an integer or float.")
        if not math.isfinite(float(weight)) or weight < 0:
            raise ValueError("weight must be finite and nonnegative.")
        self.source = source
        self.weight = weight


def _require_disjoint_records(components: Sequence[AtomisticMixtureComponent], /) -> None:
    identifiers: set[str] = set()
    digests: set[str] = set()
    for position, component in enumerate(components):
        source = component.source
        if len(set(source.record_digests)) != source.record_count:
            raise AtomisticLoaderError(
                "mixture",
                f"Component {position} holds records with identical content digests.",
            )
        if not identifiers.isdisjoint(source.record_ids):
            raise AtomisticLoaderError(
                "mixture",
                f"Component {position} repeats record identifiers of an earlier "
                "component; express repetition through weights.",
            )
        if not digests.isdisjoint(source.record_digests):
            raise AtomisticLoaderError(
                "mixture",
                f"Component {position} repeats record content of an earlier component.",
            )
        identifiers.update(source.record_ids)
        digests.update(source.record_digests)


@final
class AtomisticSourceMixture(StrictModule, NonTrainableState):
    """Immutable source mixture with its exact per-draw record probabilities.

    ``record_probabilities[c]`` is the marginal probability that one draw holds
    any given record of component ``c``: ``weight_c / draws_per_epoch`` for a
    complete epoch and ``p_c / record_count_c`` with replacement.
    ``case_weights[c]`` is the per-draw objective weight: ``1`` for the
    ``sampled`` objective (the expectation under the draw distribution) and
    ``(1 / total_records) / record_probabilities[c]`` for the
    ``importance-corrected`` objective, whose target is the uniform distribution
    over every distinct record of every component. Repeated draws of a record
    keep its source identity and receive distinct draw identities.
    """

    components: tuple[AtomisticMixtureComponent, ...]
    schema: AtomisticStructureStoreSchema
    draw: AtomisticMixtureDraw = eqx.field(static=True)
    objective: AtomisticMixtureObjective = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    shuffle: bool = eqx.field(static=True)
    draws_per_epoch: int = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    record_counts: tuple[int, ...] = eqx.field(static=True)
    record_probabilities: tuple[float, ...] = eqx.field(static=True)
    case_weights: tuple[float, ...] = eqx.field(static=True)
    block_starts: tuple[int, ...] = eqx.field(static=True)
    prng_implementation: str | None = eqx.field(static=True)
    mixture_id: str = eqx.field(static=True)

    def __init__(
        self,
        components: Sequence[AtomisticMixtureComponent],
        /,
        *,
        draw: AtomisticMixtureDraw,
        objective: AtomisticMixtureObjective,
        seed: int,
        shuffle: bool | None = None,
        draws_per_epoch: int | None = None,
    ) -> None:
        parts = tuple(components)
        if not parts or any(
            not isinstance(part, AtomisticMixtureComponent) for part in parts
        ):
            raise TypeError("components must be AtomisticMixtureComponent values.")
        draw_ = parse(draw, AtomisticMixtureDraw, "draw")
        objective_ = parse(objective, AtomisticMixtureObjective, "objective")
        seed_ = _integer(seed, "seed")
        if seed_ >= _UINT32_LIMIT:
            raise ValueError("seed must lie in [0, 2**32).")
        if shuffle is not None and not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a bool or None.")
        schema = parts[0].source.schema
        if any(part.source.schema.schema_id != schema.schema_id for part in parts):
            raise AtomisticLoaderError(
                "mixture", "Every mixture component must share one schema identity."
            )
        source_ids = tuple(part.source.source_id for part in parts)
        if len(set(source_ids)) != len(source_ids):
            raise AtomisticLoaderError(
                "mixture",
                "A source appears twice; express repetition through its weight.",
            )
        counts = tuple(part.source.record_count for part in parts)
        if any(count <= 0 or count >= _UINT32_LIMIT for count in counts):
            raise AtomisticLoaderError(
                "mixture", "Every component needs between 1 and 2**32-1 records."
            )
        _require_disjoint_records(parts)
        starts: tuple[int, ...]
        match draw_:
            case "epoch":
                if draws_per_epoch is not None:
                    raise ValueError(
                        "Complete epochs define draws_per_epoch from their membership."
                    )
                if any(type(part.weight) is not int for part in parts):
                    raise AtomisticLoaderError(
                        "mixture",
                        "Complete-epoch mixtures take integer per-record repeat counts.",
                    )
                blocks = [int(part.weight) * count for part, count in zip(parts, counts)]
                draws = sum(blocks)
                if draws <= 0 or draws > _POPULATION_LIMIT:
                    raise AtomisticLoaderError(
                        "mixture",
                        f"A complete epoch needs 1 to {_POPULATION_LIMIT} draws, "
                        f"not {draws}.",
                    )
                starts = tuple(int(value) for value in np.cumsum([0, *blocks[:-1]]))
                record_probabilities = tuple(int(part.weight) / draws for part in parts)
                shuffle_ = True if shuffle is None else shuffle
            case "replacement":
                if shuffle:
                    raise ValueError(
                        "Draws with replacement are already independent; shuffle "
                        "does not apply."
                    )
                if draws_per_epoch is None:
                    raise ValueError("Replacement mixtures require draws_per_epoch.")
                draws = _integer(draws_per_epoch, "draws_per_epoch", minimum=1)
                if draws >= _UINT32_LIMIT:
                    raise ValueError("draws_per_epoch must be below 2**32.")
                mass = math.fsum(float(part.weight) for part in parts)
                if mass <= 0.0:
                    raise AtomisticLoaderError(
                        "mixture", "Replacement mixtures need positive total weight."
                    )
                record_probabilities = tuple(
                    float(part.weight) / mass / count
                    for part, count in zip(parts, counts)
                )
                starts = ()
                shuffle_ = False
            case _:
                assert_never(draw_)
        total_records = sum(counts)
        if objective_ == "importance-corrected":
            if any(probability == 0.0 for probability in record_probabilities):
                raise AtomisticLoaderError(
                    "mixture",
                    "An importance-corrected objective needs positive draw "
                    "probability for every record of its uniform target.",
                )
            weights = tuple(
                (1.0 / total_records) / probability
                for probability in record_probabilities
            )
        else:
            weights = (1.0,) * len(parts)
        prng = default_prng_implementation() if draw_ == "replacement" else None
        self.components = parts
        self.schema = schema
        self.draw = draw_
        self.objective = objective_
        self.seed = seed_
        self.shuffle = shuffle_
        self.draws_per_epoch = draws
        self.source_ids = source_ids
        self.record_counts = counts
        self.record_probabilities = record_probabilities
        self.case_weights = weights
        self.block_starts = starts
        self.prng_implementation = prng
        self.mixture_id = canonical_fingerprint(
            {
                "kind": "atomistic-source-mixture",
                "schema": schema.schema_id,
                "components": [
                    {"source": source_id, "records": count, "weight": part.weight}
                    for source_id, count, part in zip(source_ids, counts, parts)
                ],
                "draw": draw_,
                "objective": objective_,
                "seed": seed_,
                "shuffle": shuffle_,
                "draws_per_epoch": draws,
                "ordering": EPOCH_ORDER_ALGORITHM,
                "prng": prng,
            }
        )

    @property
    def total_records(self) -> int:
        return sum(self.record_counts)

    def resolve(
        self, epoch: int, draws: npt.ArrayLike, /
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Component index and record ordinal of every epoch draw position."""

        epoch_ = _integer(epoch, "epoch")
        if epoch_ >= _UINT32_LIMIT:
            raise ValueError("epoch must lie in [0, 2**32).")
        positions = np.asarray(draws)
        if positions.ndim != 1 or positions.size == 0 or positions.dtype.kind not in "iu":
            raise ValueError("draws must be a non-empty integer vector.")
        positions = positions.astype(np.int64)
        if np.any(positions < 0) or np.any(positions >= self.draws_per_epoch):
            raise ValueError(f"draws must lie in [0, {self.draws_per_epoch}).")
        counts = np.asarray(self.record_counts, dtype=np.int64)
        if self.draw == "epoch":
            starts = np.asarray(self.block_starts, dtype=np.int64)
            component = np.searchsorted(starts, positions, side="right") - 1
            ordinal = (positions - starts[component]) % counts[component]
            return component.astype(np.int64), ordinal.astype(np.int64)
        source_mass = np.asarray(self.record_probabilities, dtype=np.float64) * counts
        cumulative = np.cumsum(source_mass)
        last = int(np.flatnonzero(source_mass > 0.0)[-1])
        with default_device(jax.devices("cpu")[0]):
            keys = semantic_task_keys(
                jax.random.key(self.seed),
                _REPLACEMENT_ADDRESS,
                positions.astype(np.uint32),
                np.full(positions.shape, epoch_, dtype=np.uint32),
            )
            words = np.asarray(
                jax.vmap(lambda key: jax.random.bits(key, (4,), jnp.uint32))(keys)
            ).astype(np.uint64)
        component = np.minimum(
            np.searchsorted(
                cumulative, _unit_interval(words[:, 0], words[:, 1]), "right"
            ),
            last,
        )
        record_unit = _unit_interval(words[:, 2], words[:, 3])
        ordinal = np.minimum(
            np.floor(record_unit * counts[component]).astype(np.int64),
            counts[component] - 1,
        )
        return component.astype(np.int64), ordinal


@final
class AtomisticSplitEvidence(StrictModule, NonTrainableState):
    """Declared-lineage separation evidence of named train/validation/test mixtures.

    Construction refuses any shared independent-unit lineage, record identity or
    record content digest between two splits. The evidence only covers declared
    lineage and exact identities; it does not prove scientific independence.
    """

    split_names: tuple[str, ...] = eqx.field(static=True)
    mixture_ids: tuple[str, ...] = eqx.field(static=True)
    lineage_counts: tuple[int, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(self, splits: Mapping[str, AtomisticSourceMixture], /) -> None:
        names = tuple(sorted(splits))
        if len(names) < 2:
            raise ValueError("Split evidence needs at least two named splits.")
        identities: list[tuple[set[str], set[str], set[str]]] = []
        for name in names:
            _name(name, "split name")
            mixture = splits[name]
            if not isinstance(mixture, AtomisticSourceMixture):
                raise TypeError("Splits must be AtomisticSourceMixture values.")
            sources = [component.source for component in mixture.components]
            identities.append(
                (
                    {value for source in sources for value in source.lineage_ids},
                    {value for source in sources for value in source.record_ids},
                    {value for source in sources for value in source.record_digests},
                )
            )
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                for kind, first, second in zip(
                    ("lineage", "record identity", "record content"),
                    identities[left],
                    identities[right],
                ):
                    shared = sorted(first & second)
                    if shared:
                        raise AtomisticLoaderError(
                            "leakage",
                            f"Splits {names[left]!r} and {names[right]!r} share "
                            f"{len(shared)} {kind} value(s), e.g. {shared[0]!r}.",
                        )
        self.split_names = names
        self.mixture_ids = tuple(splits[name].mixture_id for name in names)
        self.lineage_counts = tuple(len(item[0]) for item in identities)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "atomistic-split-evidence",
                "splits": [
                    [name, mixture_id, count]
                    for name, mixture_id, count in zip(
                        names, self.mixture_ids, self.lineage_counts
                    )
                ],
            }
        )


# ------------------------------------------------------------------- bindings


def _dtype_family(dtype: np.dtype[Any], /) -> str:
    return {"f": "real", "c": "complex", "i": "integer", "u": "integer", "b": "bool"}[
        dtype.kind
    ]


@final
class AtomisticFieldBinding(StrictModule, NonTrainableState):
    """Explicit consumer contract binding one payload slot to one stored field.

    The stored field must match the declared level kind, value kind, role,
    convention and trailing axes exactly. ``unit`` is the payload unit; the
    stored unit must equal it. Integer quantities may retain their declared
    physical unit without conversion; categorical, mask and index fields have
    no unit. For continuous fields only, ``convert_from`` may name the stored
    unit to apply its exact multiplicative conversion. ``dtype`` is the payload
    dtype (default: the stored dtype, or ``int32`` payload rows for index
    fields); narrowing within a dtype family is an explicit choice and integer
    narrowing is range-checked. Cached edge levels require the exact
    ``derivation_id`` of their geometry, cutoff, image and selection derivation.
    """

    slot: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    level_kind: AtomisticStoreLevelKind = eqx.field(static=True)
    value_kind: AtomisticStoreValueKind = eqx.field(static=True)
    role: AtomisticStoreFieldRole = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    axes: tuple[Axis, ...] = eqx.field(static=True)
    unit: UnitDefinition | None
    convert_from: UnitDefinition | None
    dtype: AtomisticStoreDType | None = eqx.field(static=True)
    derivation_id: str | None = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        slot: str,
        field: str,
        /,
        *,
        level_kind: AtomisticStoreLevelKind,
        value_kind: AtomisticStoreValueKind,
        role: AtomisticStoreFieldRole,
        convention: str,
        axes: Sequence[Axis] = (),
        unit: UnitDefinition | None = None,
        convert_from: UnitDefinition | None = None,
        dtype: AtomisticStoreDType | None = None,
        derivation_id: str | None = None,
    ) -> None:
        slot_ = _name(slot, "slot")
        field_ = _name(field, "field")
        level_kind_ = parse(level_kind, AtomisticStoreLevelKind, "level_kind")
        value_kind_ = parse(value_kind, AtomisticStoreValueKind, "value_kind")
        role_ = parse(role, AtomisticStoreFieldRole, "role")
        convention_ = _text(convention, "convention")
        if isinstance(axes, Axis):
            raise TypeError("axes must be a sequence of Axis values.")
        axes_ = tuple(axes)
        if any(not isinstance(axis, Axis) for axis in axes_):
            raise TypeError("axes must contain phydrax.axes.Axis values.")
        for label, value in (("unit", unit), ("convert_from", convert_from)):
            if value is not None and not isinstance(value, UnitDefinition):
                raise TypeError(f"{label} must be a UnitDefinition or None.")
        match value_kind_:
            case "continuous":
                if unit is None:
                    raise ValueError("Continuous bindings require an explicit unit.")
            case "integer":
                # Discrete physical quantities retain their stored unit exactly.
                pass
            case "categorical" | "mask" | "index":
                if unit is not None:
                    raise ValueError(f"{value_kind_!r} bindings have no unit.")
            case _:
                assert_never(value_kind_)
        if convert_from is not None:
            if value_kind_ != "continuous" or unit is None:
                raise ValueError("convert_from applies only to continuous bindings.")
            try:
                conversion_factor(convert_from, unit)
            except ValueError as error:
                raise AtomisticLoaderError("binding", str(error)) from error
        dtype_ = None if dtype is None else parse(dtype, AtomisticStoreDType, "dtype")
        if dtype_ is not None:
            family = _dtype_family(np.dtype(dtype_))
            allowed = {
                "continuous": ("real", "complex"),
                "integer": ("integer",),
                "categorical": ("integer", "bool"),
                "mask": ("bool",),
                "index": ("integer",),
            }[value_kind_]
            if family not in allowed or (
                value_kind_ == "index" and dtype_ not in ("int32", "int64")
            ):
                raise ValueError(f"dtype {dtype_!r} cannot carry {value_kind_} values.")
        derivation = (
            None if derivation_id is None else _text(derivation_id, "derivation_id")
        )
        if derivation is not None and level_kind_ != "edge":
            raise ValueError("Only edge-level bindings name a derivation identity.")
        self.slot = slot_
        self.field = field_
        self.level_kind = level_kind_
        self.value_kind = value_kind_
        self.role = role_
        self.convention = convention_
        self.axes = axes_
        self.unit = unit
        self.convert_from = convert_from
        self.dtype = dtype_
        self.derivation_id = derivation
        self.binding_id = canonical_fingerprint(
            {
                "kind": "atomistic-field-binding",
                "slot": slot_,
                "field": field_,
                "level_kind": level_kind_,
                "value_kind": value_kind_,
                "role": role_,
                "convention": convention_,
                "axes": [_axis_payload(axis) for axis in axes_],
                "unit": None if unit is None else unit.to_dict(),
                "convert_from": None if convert_from is None else convert_from.to_dict(),
                "dtype": dtype_,
                "derivation": derivation,
            }
        )


@dataclass(frozen=True, slots=True)
class _Slot:
    name: str
    field: str
    level: str
    source_dtype: str
    dtype: str
    shape: tuple[int, ...]
    factor: Fraction | None
    target: str | None
    sentinel: int | None
    optional: bool


def _resolve_binding(
    binding: AtomisticFieldBinding, schema: AtomisticStructureStoreSchema, /
) -> _Slot:
    def refuse(reason: str) -> AtomisticLoaderError:
        return AtomisticLoaderError(
            "binding", f"Slot {binding.slot!r} (field {binding.field!r}): {reason}"
        )

    if binding.field not in schema.field_names:
        raise refuse("the schema declares no such field.")
    field = schema.field(binding.field)
    level = schema.level(field.level)
    if level.kind != binding.level_kind:
        raise refuse(f"stored level kind {level.kind!r}, bound {binding.level_kind!r}.")
    if field.kind != binding.value_kind:
        raise refuse(f"stored value kind {field.kind!r}, bound {binding.value_kind!r}.")
    if field.role != binding.role:
        raise refuse(f"stored role {field.role!r}, bound {binding.role!r}.")
    if field.convention != binding.convention:
        raise refuse(
            f"stored convention {field.convention!r}, bound {binding.convention!r}."
        )
    if [_axis_payload(axis) for axis in field.axes] != [
        _axis_payload(axis) for axis in binding.axes
    ]:
        raise refuse("stored trailing axes differ from the bound axes.")
    factor: Fraction | None = None
    if binding.convert_from is None:
        if field.unit != binding.unit:
            raise refuse(
                "stored unit "
                f"{None if field.unit is None else field.unit.symbol!r} differs from "
                f"bound unit {None if binding.unit is None else binding.unit.symbol!r}; "
                "declare convert_from explicitly to convert."
            )
    else:
        if field.unit != binding.convert_from or binding.unit is None:
            raise refuse("convert_from must name the stored unit exactly.")
        factor = conversion_factor(binding.convert_from, binding.unit)
        if factor == 1:
            factor = None
    if level.kind == "edge" and level.derivation_id != binding.derivation_id:
        raise refuse(
            f"cached edge derivation {level.derivation_id!r} does not match the "
            f"bound derivation {binding.derivation_id!r}."
        )
    stored = field.numpy_dtype
    if binding.dtype is not None:
        output = np.dtype(binding.dtype)
    elif field.kind == "index":
        output = np.dtype(np.int32)
    else:
        output = stored
    if field.kind != "index" and _dtype_family(output) != _dtype_family(stored):
        raise refuse(f"stored dtype {stored} cannot be bound as {output}.")
    if factor is not None and output.kind not in "fc":
        raise refuse("unit conversion requires a floating payload dtype.")
    index = field.index
    return _Slot(
        binding.slot,
        field.name,
        field.level,
        stored.name,
        output.name,
        field.trailing_shape,
        factor,
        None if index is None else index.target,
        None if index is None else index.sentinel,
        field.availability == "optional",
    )


# -------------------------------------------------------------------- buckets


@final
class AtomisticLevelCapacity(StrictModule, NonTrainableState):
    """Row capacity of one payload level within one bucket shard.

    ``per-case`` stores ``rows`` slots for every case (case-major, the canonical
    flat atom layout); ``packed`` stores ``rows`` slots shared by all cases of a
    shard and budgets their sum.
    """

    level: str = eqx.field(static=True)
    layout: AtomisticLevelLayout = eqx.field(static=True)
    rows: int = eqx.field(static=True)

    def __init__(self, level: str, /, *, layout: AtomisticLevelLayout, rows: int) -> None:
        self.level = _name(level, "level")
        self.layout = parse(layout, AtomisticLevelLayout, "layout")
        self.rows = _integer(rows, "rows", minimum=1)


@final
class AtomisticWorkloadBucket(StrictModule, NonTrainableState):
    """Static case, per-level row and byte bounds of one payload executable."""

    name: str = eqx.field(static=True)
    cases: int = eqx.field(static=True)
    levels: tuple[AtomisticLevelCapacity, ...]
    max_bytes: int = eqx.field(static=True)
    bucket_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        /,
        *,
        cases: int,
        levels: Sequence[AtomisticLevelCapacity],
        max_bytes: int,
    ) -> None:
        capacities = tuple(levels)
        if any(not isinstance(item, AtomisticLevelCapacity) for item in capacities):
            raise TypeError("levels must contain AtomisticLevelCapacity values.")
        names = [item.level for item in capacities]
        if len(set(names)) != len(names):
            raise ValueError("A bucket declares each level capacity once.")
        self.name = _name(name, "bucket name")
        self.cases = _integer(cases, "cases", minimum=1)
        self.levels = capacities
        self.max_bytes = _integer(max_bytes, "max_bytes", minimum=1)
        self.bucket_id = canonical_fingerprint(
            {
                "kind": "atomistic-workload-bucket",
                "name": self.name,
                "cases": self.cases,
                "levels": [[item.level, item.layout, item.rows] for item in capacities],
                "max_bytes": self.max_bytes,
            }
        )

    def capacity(self, level: str, /) -> AtomisticLevelCapacity:
        for item in self.levels:
            if item.level == level:
                return item
        raise KeyError(f"Bucket {self.name!r} declares no capacity for level {level!r}.")


@final
class AtomisticWorkloadBuckets(StrictModule, NonTrainableState):
    """Ordered buckets; every record enters the first bucket that admits it.

    A record that no bucket admits is refused before any read; declare a larger
    bucket explicitly to admit it. Nothing is dropped.
    """

    buckets: tuple[AtomisticWorkloadBucket, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(self, buckets: Sequence[AtomisticWorkloadBucket], /) -> None:
        values = tuple(buckets)
        if not values or any(
            not isinstance(item, AtomisticWorkloadBucket) for item in values
        ):
            raise TypeError("buckets must be non-empty AtomisticWorkloadBucket values.")
        if len({item.name for item in values}) != len(values):
            raise ValueError("Bucket names must be unique.")
        self.buckets = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-workload-buckets",
                "buckets": [b.bucket_id for b in values],
            }
        )


# ------------------------------------------------------------------ placement


@final
class AtomisticLoaderPlacement(StrictModule, NonTrainableState):
    """Group-local data-shard sharding of every global batch.

    The default (``rank=None``) is one natural single-device shard. Otherwise
    global shard ``g`` is the data-axis coordinate of the bound execution
    group and this process addresses the contiguous shards ``rank.shards``.
    ``placement_id`` binds only the rank-independent ``rank.layout_id`` so
    every process of one placement shares one loader and cursor identity.
    """

    rank: ExecutionGroupRank | None = eqx.field(static=True)
    placement_id: str = eqx.field(static=True)

    def __init__(self, rank: ExecutionGroupRank | None = None) -> None:
        if rank is not None and not isinstance(rank, ExecutionGroupRank):
            raise TypeError("rank must be an ExecutionGroupRank or None.")
        self.rank = rank
        self.placement_id = canonical_fingerprint(
            {
                "kind": "atomistic-loader-placement",
                "layout": None if rank is None else rank.layout_id,
            }
        )

    @property
    def shard_count(self) -> int:
        return 1 if self.rank is None else self.rank.shard_count

    @property
    def local_shards(self) -> tuple[int, ...]:
        """Global data shards addressed by this process, contiguous."""

        return (0,) if self.rank is None else self.rank.shards

    @property
    def group_id(self) -> str | None:
        return None if self.rank is None else self.rank.group_id


# -------------------------------------------------------------------- payload


@final
class AtomisticPayloadLayout(StrictModule, NonTrainableState):
    """Schema-static numerical layout of one bucket payload on one process.

    ``arrays`` lists every payload array as ``(name, local shape, dtype)`` in
    canonical order; ``signature`` is the executable identity qualified once per
    bucket. No record, lineage or source identifier enters the layout.
    """

    bucket: str = eqx.field(static=True)
    bucket_index: int = eqx.field(static=True)
    local_shards: int = eqx.field(static=True)
    shard_count: int = eqx.field(static=True)
    case_capacity: int = eqx.field(static=True)
    levels: tuple[str, ...] = eqx.field(static=True)
    level_layouts: tuple[AtomisticLevelLayout, ...] = eqx.field(static=True)
    level_rows: tuple[int, ...] = eqx.field(static=True)
    level_parents: tuple[str | None, ...] = eqx.field(static=True)
    slots: tuple[str, ...] = eqx.field(static=True)
    slot_levels: tuple[str, ...] = eqx.field(static=True)
    arrays: tuple[tuple[str, tuple[int, ...], str], ...] = eqx.field(static=True)
    nbytes: int = eqx.field(static=True)
    signature: ExecutableSignature
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        bucket: AtomisticWorkloadBucket,
        bucket_index: int,
        /,
        *,
        schema: AtomisticStructureStoreSchema,
        levels: Sequence[str],
        parents: Sequence[str | None],
        slots: Sequence[_Slot],
        placement: AtomisticLoaderPlacement,
        bindings_id: str,
    ) -> None:
        shards = len(placement.local_shards)
        cases = bucket.cases
        local_cases = shards * cases
        layouts = tuple(bucket.capacity(level).layout for level in levels)
        rows = tuple(
            cases * bucket.capacity(level).rows
            if bucket.capacity(level).layout == "per-case"
            else bucket.capacity(level).rows
            for level in levels
        )
        if any(placement.shard_count * value >= _INT32_LIMIT for value in (*rows, cases)):
            raise AtomisticLoaderError(
                "capacity", f"Bucket {bucket.name!r} rows exceed int32 payload indices."
            )
        arrays: list[tuple[str, tuple[int, ...], str]] = [
            ("case_valid", (local_cases,), "bool"),
            ("case_weight", (local_cases,), "float64"),
            ("draw_probability", (local_cases,), "float64"),
            ("draw_index", (local_cases,), "uint32"),
            ("case_epoch", (local_cases,), "uint32"),
            ("case_component", (local_cases,), "int32"),
            ("record_ordinal", (local_cases,), "uint32"),
        ]
        for level, level_rows, parent in zip(levels, rows, parents):
            arrays.append((f"{level}.row_mask", (shards * level_rows,), "bool"))
            arrays.append((f"{level}.row_case", (shards * level_rows,), "int32"))
            arrays.append((f"{level}.case_count", (local_cases,), "int32"))
            if parent is not None:
                arrays.append((f"{level}.row_parent", (shards * level_rows,), "int32"))
        atom_rows = rows[tuple(levels).index(schema.atom_level)]
        arrays.append(("stable_id_words", (shards * atom_rows, 2), "uint32"))
        for slot in slots:
            count = (
                local_cases
                if slot.level == schema.structure_level
                else shards * rows[tuple(levels).index(slot.level)]
            )
            arrays.append((f"slot.{slot.name}", (count, *slot.shape), slot.dtype))
            arrays.append((f"slot.{slot.name}.available", (count,), "bool"))
        nbytes = sum(
            math.prod(shape) * np.dtype(dtype).itemsize for _, shape, dtype in arrays
        )
        if nbytes > bucket.max_bytes:
            raise AtomisticLoaderError(
                "capacity",
                f"Bucket {bucket.name!r} payload needs {nbytes} bytes, above its "
                f"{bucket.max_bytes}-byte bound.",
            )
        signature = ExecutableSignature(
            shapes=[(name, shape) for name, shape, _ in arrays],
            dtypes=[(name, dtype) for name, _, dtype in arrays],
            capacities={
                "cases": cases,
                "local_shards": shards,
                "shards": placement.shard_count,
                **{f"rows.{level}": value for level, value in zip(levels, rows)},
            },
            algorithm_facts={
                "kind": "atomistic-loader-payload",
                "schema": schema.schema_id,
                "bindings": bindings_id,
                "bucket": bucket.bucket_id,
                "layouts": tuple(zip(levels, layouts)),
            },
        )
        self.bucket = bucket.name
        self.bucket_index = bucket_index
        self.local_shards = shards
        self.shard_count = placement.shard_count
        self.case_capacity = cases
        self.levels = tuple(levels)
        self.level_layouts = layouts
        self.level_rows = rows
        self.level_parents = tuple(parents)
        self.slots = tuple(slot.name for slot in slots)
        self.slot_levels = tuple(slot.level for slot in slots)
        self.arrays = tuple(arrays)
        self.nbytes = nbytes
        self.signature = signature
        self.layout_id = signature.signature_id


@final
class AtomisticPayloadLevel(StrictModule):
    """Row occupancy of one level in a payload.

    ``row_mask`` marks rows holding structure rows, including the full copies
    owned by padding cases; slack rows copy a real row of the same shard (or
    hold zeros when the shard has none) and are masked. ``row_case`` is the
    global owning case, ``case_count`` the occupied rows per case and
    ``row_parent`` the global parent-level row of segment rows.
    """

    row_mask: _Buffer
    row_case: _Buffer
    case_count: _Buffer
    row_parent: _Buffer | None
    level: str = eqx.field(static=True)
    layout: AtomisticLevelLayout = eqx.field(static=True)
    rows_per_shard: int = eqx.field(static=True)

    def __init__(
        self,
        level: str,
        layout: AtomisticLevelLayout,
        rows_per_shard: int,
        /,
        *,
        row_mask: _Buffer,
        row_case: _Buffer,
        case_count: _Buffer,
        row_parent: _Buffer | None,
    ) -> None:
        self.level = level
        self.layout = layout
        self.rows_per_shard = rows_per_shard
        self.row_mask = row_mask
        self.row_case = row_case
        self.case_count = case_count
        self.row_parent = row_parent


def _payload_arrays(payload: AtomisticLoaderPayload, /) -> list[tuple[str, _Buffer]]:
    arrays: list[tuple[str, _Buffer]] = [
        ("case_valid", payload.case_valid),
        ("case_weight", payload.case_weight),
        ("draw_probability", payload.draw_probability),
        ("draw_index", payload.draw_index),
        ("case_epoch", payload.case_epoch),
        ("case_component", payload.case_component),
        ("record_ordinal", payload.record_ordinal),
    ]
    for level in payload.levels:
        arrays.append((f"{level.level}.row_mask", level.row_mask))
        arrays.append((f"{level.level}.row_case", level.row_case))
        arrays.append((f"{level.level}.case_count", level.case_count))
        if level.row_parent is not None:
            arrays.append((f"{level.level}.row_parent", level.row_parent))
    arrays.append(("stable_id_words", payload.stable_id_words))
    for name, values, available in zip(
        payload.layout.slots, payload.values, payload.available
    ):
        arrays.append((f"slot.{name}", values))
        arrays.append((f"slot.{name}.available", available))
    return arrays


@final
class AtomisticLoaderPayload(StrictModule):
    """Padded numerical payload of one bucket step on this process.

    Cases are shard-major: local case ``s * case_capacity + c`` belongs to global
    shard ``placement.local_shards[0] + s`` (global shard = data-axis
    coordinate; the process addresses ``rank.shards``). Row, case and index values are
    global payload positions, so process-local arrays assemble into global
    arrays without renumbering. Invalid cases repeat a real record of their
    shard (or the bucket's anchor record) with zero ``case_weight``. Label
    support is ``available & case_valid[row_case]``; nonfinite available values
    are retained, never treated as missing. Stable particle identities are
    ``(high, low)`` uint32 words.
    """

    layout: AtomisticPayloadLayout
    case_valid: _Buffer
    case_weight: _Buffer
    draw_probability: _Buffer
    draw_index: _Buffer
    case_epoch: _Buffer
    case_component: _Buffer
    record_ordinal: _Buffer
    levels: tuple[AtomisticPayloadLevel, ...]
    stable_id_words: _Buffer
    values: tuple[_Buffer, ...]
    available: tuple[_Buffer, ...]

    def __init__(
        self,
        layout: AtomisticPayloadLayout,
        /,
        *,
        case_valid: _Buffer,
        case_weight: _Buffer,
        draw_probability: _Buffer,
        draw_index: _Buffer,
        case_epoch: _Buffer,
        case_component: _Buffer,
        record_ordinal: _Buffer,
        levels: Sequence[AtomisticPayloadLevel],
        stable_id_words: _Buffer,
        values: Sequence[_Buffer],
        available: Sequence[_Buffer],
    ) -> None:
        self.layout = layout
        self.case_valid = case_valid
        self.case_weight = case_weight
        self.draw_probability = draw_probability
        self.draw_index = draw_index
        self.case_epoch = case_epoch
        self.case_component = case_component
        self.record_ordinal = record_ordinal
        self.levels = tuple(levels)
        self.stable_id_words = stable_id_words
        self.values = tuple(values)
        self.available = tuple(available)
        if tuple(level.level for level in self.levels) != layout.levels or len(
            self.values
        ) != len(layout.slots):
            raise ValueError("Payload levels and slots must follow the layout.")
        actual = [
            (name, tuple(value.shape), np.dtype(value.dtype).name)
            for name, value in _payload_arrays(self)
        ]
        expected = [
            (name, shape, np.dtype(dtype).name) for name, shape, dtype in layout.arrays
        ]
        if actual != expected:
            raise ValueError("Payload arrays differ from the static bucket layout.")

    def level(self, name: str, /) -> AtomisticPayloadLevel:
        for level in self.levels:
            if level.level == name:
                return level
        raise KeyError(f"The payload has no level {name!r}.")

    def value(self, slot: str, /) -> _Buffer:
        return self.values[self._slot(slot)]

    def availability(self, slot: str, /) -> _Buffer:
        return self.available[self._slot(slot)]

    def support(self, slot: str, /) -> _Buffer:
        """Rows of ``slot`` that carry a valid case's available value.

        ``row_case`` holds global case positions, so this indexing needs the
        global view: a single-process payload or a staged global array.
        """

        index = self._slot(slot)
        level = self.layout.slot_levels[index]
        available = self.available[index]
        if level not in self.layout.levels:
            return available & self.case_valid
        return available & self.case_valid[self.level(level).row_case]

    def draw_keys(self, root_key: Array, /) -> Array:
        """Per-case keys addressed by ``(draw, epoch)``, independent of placement."""

        return semantic_task_keys(
            root_key, _DRAW_ADDRESS, self.draw_index, self.case_epoch
        )

    def _slot(self, slot: str, /) -> int:
        if slot not in self.layout.slots:
            raise KeyError(f"The payload has no slot {slot!r}.")
        return self.layout.slots.index(slot)


@final
class AtomisticPayloadProvenance(StrictModule, NonTrainableState):
    """Host provenance of one payload, separate from its numerical layout."""

    loader_id: str = eqx.field(static=True)
    epoch: int = eqx.field(static=True)
    batch_index: int = eqx.field(static=True)
    payload_index: int = eqx.field(static=True)
    payload_count: int = eqx.field(static=True)
    bucket: str = eqx.field(static=True)
    global_valid_draws: int = eqx.field(static=True)
    draw_indices: tuple[int, ...] = eqx.field(static=True)
    valid: tuple[bool, ...] = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    record_ids: tuple[str, ...] = eqx.field(static=True)
    lineage_ids: tuple[str, ...] = eqx.field(static=True)
    record_digests: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        loader_id: str,
        epoch: int,
        batch_index: int,
        payload_index: int,
        payload_count: int,
        bucket: str,
        global_valid_draws: int,
        draw_indices: Sequence[int],
        valid: Sequence[bool],
        source_ids: Sequence[str],
        record_ids: Sequence[str],
        lineage_ids: Sequence[str],
        record_digests: Sequence[str],
    ) -> None:
        self.loader_id = loader_id
        self.epoch = epoch
        self.batch_index = batch_index
        self.payload_index = payload_index
        self.payload_count = payload_count
        self.bucket = bucket
        self.global_valid_draws = global_valid_draws
        self.draw_indices = tuple(int(value) for value in draw_indices)
        self.valid = tuple(bool(value) for value in valid)
        self.source_ids = tuple(source_ids)
        self.record_ids = tuple(record_ids)
        self.lineage_ids = tuple(lineage_ids)
        self.record_digests = tuple(record_digests)


# --------------------------------------------------------------------- cursor


_CURSOR_KEYS = frozenset(
    {
        "kind",
        "loader_id",
        "semantics_id",
        "structure_id",
        "stream",
        "source_ids",
        "epoch",
        "batch_index",
        "payload_index",
        "cursor_id",
    }
)


@final
class AtomisticLoaderCursor(StrictModule, NonTrainableState):
    """Resumable loader position bound to content, policy and layout identity.

    ``semantics_id`` binds source content, mixture, bindings, stream and batch
    size; ``structure_id`` binds the bucket plan and placement; ``source_ids``
    name the exact content of every component so drift is reported per source.
    The position names the next payload: ``payload_index`` counts payloads of
    batch ``batch_index`` already committed into the open accumulation window.
    """

    loader_id: str = eqx.field(static=True)
    semantics_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    stream: str = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    epoch: int = eqx.field(static=True)
    batch_index: int = eqx.field(static=True)
    payload_index: int = eqx.field(static=True)
    cursor_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        loader_id: str,
        semantics_id: str,
        structure_id: str,
        stream: str,
        source_ids: Sequence[str],
        epoch: int,
        batch_index: int,
        payload_index: int,
    ) -> None:
        self.loader_id = _text(loader_id, "loader_id")
        self.semantics_id = _text(semantics_id, "semantics_id")
        self.structure_id = _text(structure_id, "structure_id")
        self.stream = _name(stream, "stream")
        if isinstance(source_ids, str):
            raise TypeError("source_ids must be a sequence of strings.")
        self.source_ids = tuple(_text(value, "source_id") for value in source_ids)
        self.epoch = _integer(epoch, "epoch")
        self.batch_index = _integer(batch_index, "batch_index")
        self.payload_index = _integer(payload_index, "payload_index")
        self.cursor_id = canonical_fingerprint(self._content())

    def _content(self) -> dict[str, Any]:
        return {
            "kind": "atomistic-loader-cursor",
            "loader_id": self.loader_id,
            "semantics_id": self.semantics_id,
            "structure_id": self.structure_id,
            "stream": self.stream,
            "source_ids": list(self.source_ids),
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "payload_index": self.payload_index,
        }

    def to_payload(self) -> dict[str, Any]:
        """JSON-compatible cursor record for checkpoint metadata."""

        return {**self._content(), "cursor_id": self.cursor_id}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], /) -> AtomisticLoaderCursor:
        if not isinstance(payload, Mapping) or set(payload) != _CURSOR_KEYS:
            raise AtomisticLoaderError(
                "cursor", f"Cursor payloads contain exactly {sorted(_CURSOR_KEYS)}."
            )
        if payload["kind"] != "atomistic-loader-cursor":
            raise AtomisticLoaderError("cursor", "The payload is not a loader cursor.")
        sources = payload["source_ids"]
        if not isinstance(sources, list):
            raise AtomisticLoaderError("cursor", "Cursor source_ids must be a list.")
        try:
            cursor = cls(
                loader_id=payload["loader_id"],
                semantics_id=payload["semantics_id"],
                structure_id=payload["structure_id"],
                stream=payload["stream"],
                source_ids=sources,
                epoch=payload["epoch"],
                batch_index=payload["batch_index"],
                payload_index=payload["payload_index"],
            )
        except (TypeError, ValueError) as error:
            raise AtomisticLoaderError("cursor", f"Malformed cursor: {error}") from error
        if cursor.cursor_id != payload["cursor_id"]:
            raise AtomisticLoaderError(
                "cursor", "The cursor identity does not match its content."
            )
        return cursor

    def composition_entry(
        self,
        /,
        *,
        entry_id: str,
        owner_id: str,
        dependencies: Sequence[CompositionDependency] = (),
    ) -> CompositionEntry:
        """Durable ``history`` state entry of this cursor at an accepted boundary.

        Structure is the bucket/placement identity, revision the exact cursor
        and semantics the content/mixture/binding identity, so a rebind that
        changes data semantics or layout cannot silently retain or recreate it.
        """

        # Lazy: the lifecycle package sits above the data plane.
        from ..lifecycle import CompositionEntry

        return CompositionEntry(
            self,
            entry_id=entry_id,
            role="history",
            owner_id=owner_id,
            structure_id=self.structure_id,
            revision_id=self.cursor_id,
            semantics_id=self.semantics_id,
            dependencies=dependencies,
        )


@final
class AtomisticLoaderItem(StrictModule):
    """One payload, its host provenance and the cursor after committing it."""

    payload: AtomisticLoaderPayload
    provenance: AtomisticPayloadProvenance
    cursor: AtomisticLoaderCursor
    closes_batch: bool = eqx.field(static=True)

    def __init__(
        self,
        payload: AtomisticLoaderPayload,
        provenance: AtomisticPayloadProvenance,
        cursor: AtomisticLoaderCursor,
        /,
        *,
        closes_batch: bool,
    ) -> None:
        self.payload = payload
        self.provenance = provenance
        self.cursor = cursor
        self.closes_batch = bool(closes_batch)


# --------------------------------------------------------------------- staging


@final
class AtomisticLoaderStaging(StrictModule, NonTrainableState):
    """Execution-only host queue and device staging bounds.

    ``host_prefetch`` bounds the ordered host queue of prepared payloads; it is
    used only when every source declares ``background_read_safe`` and is
    otherwise synchronous. ``target="device"`` places each payload explicitly
    (default device, or an `ExecutionGroup` mesh) in the consuming thread, with
    ``device_ahead`` further payloads dispatched ahead; their total static bytes
    are bounded by ``max_device_bytes``. Placement dispatch is asynchronous and
    the loader performs no device synchronization; overlap with computation is
    not claimed. None of these settings enters loader identity.
    """

    host_prefetch: int = eqx.field(static=True)
    target: AtomisticLoaderTarget = eqx.field(static=True)
    device_ahead: int = eqx.field(static=True)
    max_device_bytes: int | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        host_prefetch: int = 0,
        target: AtomisticLoaderTarget = "host",
        device_ahead: int = 0,
        max_device_bytes: int | None = None,
    ) -> None:
        self.host_prefetch = _integer(host_prefetch, "host_prefetch")
        self.target = parse(target, AtomisticLoaderTarget, "target")
        self.device_ahead = _integer(device_ahead, "device_ahead")
        if self.target == "device":
            if max_device_bytes is None:
                raise ValueError("Device staging requires max_device_bytes.")
            self.max_device_bytes = _integer(
                max_device_bytes, "max_device_bytes", minimum=1
            )
        else:
            if device_ahead or max_device_bytes is not None:
                raise ValueError("Host staging takes no device bounds.")
            self.max_device_bytes = None


def _stage(
    payload: AtomisticLoaderPayload,
    placement: AtomisticLoaderPlacement,
    group: ExecutionGroup | None,
    /,
) -> AtomisticLoaderPayload:
    if not x64_enabled() and any(
        np.dtype(value.dtype) in _WIDE_DTYPES for _, value in _payload_arrays(payload)
    ):
        raise AtomisticLoaderError(
            "placement",
            "The payload holds 64-bit arrays; device staging requires x64 so they "
            "are not silently narrowed. Bind narrower dtypes explicitly instead.",
        )
    if group is None:
        if placement.shard_count != 1:
            raise AtomisticLoaderError(
                "placement", "Sharded placements stage through an ExecutionGroup."
            )
        return jax.tree.map(jax.device_put, payload)
    processes = 1 if placement.rank is None else placement.rank.rank_count
    return jax.tree.map(
        lambda leaf: shard_array_axis(
            leaf,
            group,
            axis=0,
            process_local=processes > 1,
            global_axis_size=leaf.shape[0] * processes if processes > 1 else None,
        ),
        payload,
    )


def _require_group(group: ExecutionGroup, placement: AtomisticLoaderPlacement, /) -> None:
    if not isinstance(group, ExecutionGroup):
        raise TypeError("execution_group must be an ExecutionGroup.")
    rank = placement.rank
    if rank is None:
        raise AtomisticLoaderError(
            "placement", "Sharded staging requires a placement bound to a group rank."
        )
    if group.spec.group_id != rank.group_id:
        raise AtomisticLoaderError(
            "placement", "The execution group differs from the placement group."
        )
    if ExecutionGroupRank.current(group, axis=rank.axis) != rank:
        raise AtomisticLoaderError(
            "placement",
            "The placement rank is not this process's rank in the execution group.",
        )


# --------------------------------------------------------------------- loader


@dataclass(frozen=True, slots=True)
class _ShardPack:
    positions: tuple[int, ...]
    components: tuple[int, ...]
    ordinals: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Step:
    epoch: int
    batch_index: int
    payload_index: int
    payload_count: int
    bucket: int
    global_valid: int
    first_position: int
    shards: tuple[_ShardPack, ...]


@dataclass(frozen=True, slots=True)
class _Records:
    """Concatenated rows of every record read for one payload."""

    source_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    lineage_ids: tuple[str, ...]
    record_digests: tuple[str, ...]
    counts: Mapping[str, npt.NDArray[np.int64]]
    starts: Mapping[str, npt.NDArray[np.int64]]
    parent_local: Mapping[str, npt.NDArray[np.int64]]
    stable_ids: npt.NDArray[np.int64]
    values: Mapping[str, npt.NDArray[Any]]
    available: Mapping[str, npt.NDArray[np.bool_]]


@dataclass(frozen=True, slots=True)
class _Rows:
    """Gather plan of one payload level: source rows (``-1`` zero-fills)."""

    source: npt.NDArray[np.int64]
    row_case: npt.NDArray[np.int64]
    source_case: npt.NDArray[np.int64]
    row_mask: npt.NDArray[np.bool_]
    case_count: npt.NDArray[np.int64]
    case_base: npt.NDArray[np.int64]
    rows_per_shard: int


def _loader_bindings(
    bindings: Sequence[AtomisticFieldBinding], /
) -> tuple[AtomisticFieldBinding, ...]:
    bindings_ = tuple(bindings)
    if not bindings_ or any(
        not isinstance(item, AtomisticFieldBinding) for item in bindings_
    ):
        raise TypeError("bindings must be non-empty AtomisticFieldBinding values.")
    if len({item.slot for item in bindings_}) != len(bindings_):
        raise ValueError("Binding slots must be unique.")
    return bindings_


def _loader_levels(
    schema: AtomisticStructureStoreSchema, slots: Sequence[_Slot], /
) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    """Schema-ordered closure of the atom, slot and target levels, with parents.

    The structure level is implicit (one per case); a parent equal to it is
    recorded as ``None``.
    """
    needed = {schema.atom_level}
    needed.update(slot.level for slot in slots)
    needed.update(slot.target for slot in slots if slot.target is not None)
    needed.discard(schema.structure_level)
    pending = list(needed)
    while pending:
        parent = schema.level(pending.pop()).parent
        if (
            parent is not None
            and parent != schema.structure_level
            and parent not in needed
        ):
            needed.add(parent)
            pending.append(parent)
    levels = tuple(name for name in schema.level_names if name in needed)
    parents = tuple(
        None
        if schema.level(name).parent in (None, schema.structure_level)
        else schema.level(name).parent
        for name in levels
    )
    return levels, parents


def _require_bucket_levels(
    buckets: AtomisticWorkloadBuckets,
    schema: AtomisticStructureStoreSchema,
    levels: tuple[str, ...],
    /,
) -> None:
    for bucket in buckets.buckets:
        declared = {item.level for item in bucket.levels}
        if declared != set(levels):
            raise AtomisticLoaderError(
                "binding",
                f"Bucket {bucket.name!r} must declare capacities for exactly "
                f"{sorted(levels)}, not {sorted(declared)}.",
            )
        if bucket.capacity(schema.atom_level).layout != "per-case":
            raise AtomisticLoaderError(
                "binding",
                "The atom level is stored per case (the canonical flat "
                "case-major layout).",
            )


def _bucket_assignment(
    mixture: AtomisticSourceMixture,
    buckets: AtomisticWorkloadBuckets,
    levels: tuple[str, ...],
    counts: tuple[tuple[npt.NDArray[np.int64], ...], ...],
    atom: int,
    /,
) -> tuple[tuple[npt.NDArray[np.int16], ...], tuple[tuple[int, int] | None, ...]]:
    """Assign every record to the first bucket it fits and anchor each bucket.

    A bucket's anchor is its first ``(component, record)`` in a component of
    positive draw probability, else its first record of any component, else
    ``None``.
    """
    assignment: list[npt.NDArray[np.int16]] = []
    anchors: list[tuple[int, int] | None] = [None] * len(buckets.buckets)
    fallback: list[tuple[int, int] | None] = [None] * len(buckets.buckets)
    for position, (component, level_counts) in enumerate(zip(mixture.components, counts)):
        ids = component.source.record_ids
        empty = np.flatnonzero(level_counts[atom] == 0)
        if empty.size:
            raise AtomisticLoaderError(
                "binding",
                f"Record {ids[int(empty[0])]!r} has no atoms and cannot form a case.",
            )
        assigned = np.full(len(ids), -1, dtype=np.int16)
        for index, bucket in enumerate(buckets.buckets):
            fits = np.ones(len(ids), dtype=np.bool_)
            for level, level_count in zip(levels, level_counts):
                fits &= level_count <= bucket.capacity(level).rows
            chosen = (assigned < 0) & fits
            assigned[chosen] = index
            if np.any(chosen):
                first = (position, int(np.flatnonzero(chosen)[0]))
                if fallback[index] is None:
                    fallback[index] = first
                if anchors[index] is None and mixture.record_probabilities[position] > 0:
                    anchors[index] = first
        refused = np.flatnonzero(assigned < 0)
        if refused.size:
            record = int(refused[0])
            sizes = {
                level: int(level_count[record])
                for level, level_count in zip(levels, level_counts)
            }
            raise AtomisticLoaderError(
                "capacity",
                f"{refused.size} record(s) of component {position} fit no bucket; "
                f"record {ids[record]!r} has rows {sizes}. Declare a larger "
                "bucket explicitly.",
            )
        assignment.append(_frozen(assigned))
    return tuple(assignment), tuple(
        anchor if anchor is not None else spare
        for anchor, spare in zip(anchors, fallback)
    )


@final
class AtomisticStructureLoader(StrictModule, NonTrainableState):
    """Deterministic, bucketed, resumable payload stream of one source mixture.

    Each global batch holds ``batch_size`` epoch draws (a short final batch
    unless ``remainder="drop"``) and is one accumulation window. Its draws are
    cut into global shards by `DistributedIndexEpochPlan`; every shard packs its
    valid draws, per bucket and in draw order, under the bucket's case and
    packed-row capacities. Every process derives the same payload count per
    batch from source metadata alone and emits empty-shard payloads as fully
    invalid padding.
    """

    mixture: AtomisticSourceMixture
    bindings: tuple[AtomisticFieldBinding, ...]
    buckets: AtomisticWorkloadBuckets
    placement: AtomisticLoaderPlacement
    layouts: tuple[AtomisticPayloadLayout, ...]
    batch_size: int = eqx.field(static=True)
    remainder: AtomisticLoaderRemainder = eqx.field(static=True)
    stream: str = eqx.field(static=True)
    levels: tuple[str, ...] = eqx.field(static=True)
    semantics_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    loader_id: str = eqx.field(static=True)
    _slots: tuple[_Slot, ...] = eqx.field(static=True)
    _fields: tuple[str, ...] = eqx.field(static=True)
    _parents: tuple[str | None, ...] = eqx.field(static=True)
    _anchors: tuple[tuple[int, int] | None, ...] = eqx.field(static=True)
    _counts: tuple[tuple[npt.NDArray[np.int64], ...], ...]
    _assignment: tuple[npt.NDArray[np.int16], ...]

    def __init__(
        self,
        mixture: AtomisticSourceMixture,
        bindings: Sequence[AtomisticFieldBinding],
        buckets: AtomisticWorkloadBuckets,
        /,
        *,
        batch_size: int,
        stream: str,
        remainder: AtomisticLoaderRemainder = "short",
        placement: AtomisticLoaderPlacement | None = None,
    ) -> None:
        if not isinstance(mixture, AtomisticSourceMixture):
            raise TypeError("mixture must be an AtomisticSourceMixture.")
        bindings_ = _loader_bindings(bindings)
        if not isinstance(buckets, AtomisticWorkloadBuckets):
            raise TypeError("buckets must be AtomisticWorkloadBuckets.")
        placement_ = AtomisticLoaderPlacement() if placement is None else placement
        if not isinstance(placement_, AtomisticLoaderPlacement):
            raise TypeError("placement must be an AtomisticLoaderPlacement.")
        batch = _integer(batch_size, "batch_size", minimum=1)
        remainder_ = parse(remainder, AtomisticLoaderRemainder, "remainder")
        stream_ = _name(stream, "stream")
        schema = mixture.schema
        slots = tuple(_resolve_binding(item, schema) for item in bindings_)
        levels, parents = _loader_levels(schema, slots)
        _require_bucket_levels(buckets, schema, levels)
        bindings_id = canonical_fingerprint(
            {
                "kind": "atomistic-loader-bindings",
                "bindings": [b.binding_id for b in bindings_],
            }
        )
        layouts = tuple(
            AtomisticPayloadLayout(
                bucket,
                position,
                schema=schema,
                levels=levels,
                parents=parents,
                slots=slots,
                placement=placement_,
                bindings_id=bindings_id,
            )
            for position, bucket in enumerate(buckets.buckets)
        )
        counts = tuple(
            tuple(
                _frozen(np.asarray(component.source.level_counts(level), dtype=np.int64))
                for level in levels
            )
            for component in mixture.components
        )
        assignment, anchors = _bucket_assignment(
            mixture, buckets, levels, counts, levels.index(schema.atom_level)
        )
        semantics_id = canonical_fingerprint(
            {
                "kind": "atomistic-loader-semantics",
                "mixture": mixture.mixture_id,
                "bindings": bindings_id,
                "stream": stream_,
                "batch_size": batch,
                "remainder": remainder_,
            }
        )
        structure_id = canonical_fingerprint(
            {
                "kind": "atomistic-loader-structure",
                "buckets": buckets.plan_id,
                "placement": placement_.placement_id,
                "layouts": [layout.layout_id for layout in layouts],
            }
        )
        self.mixture = mixture
        self.bindings = bindings_
        self.buckets = buckets
        self.placement = placement_
        self.layouts = layouts
        self.batch_size = batch
        self.remainder = remainder_
        self.stream = stream_
        self.levels = levels
        self.semantics_id = semantics_id
        self.structure_id = structure_id
        self.loader_id = canonical_fingerprint(
            {
                "kind": "atomistic-structure-loader",
                "semantics": semantics_id,
                "structure": structure_id,
            }
        )
        self._slots = slots
        self._fields = tuple(
            name for name in schema.field_names if name in {slot.field for slot in slots}
        )
        self._parents = parents
        self._anchors = anchors
        self._counts = counts
        self._assignment = assignment
        if self.batch_count == 0:
            raise AtomisticLoaderError(
                "mixture", "Dropping the remainder leaves no complete batch."
            )

    # Plan ------------------------------------------------------------------------

    @property
    def batch_count(self) -> int:
        return self.epoch_plan(0).batch_count

    @property
    def background_read_safe(self) -> bool:
        return all(
            component.source.background_read_safe for component in self.mixture.components
        )

    def epoch_plan(self, epoch: int, /) -> IndexEpochPlan:
        """The shared stateless draw-position plan of one epoch."""

        return IndexEpochPlan(
            source_size=self.mixture.draws_per_epoch,
            batch_size=self.batch_size,
            shuffle=self.mixture.shuffle,
            seed=self.mixture.seed,
            epoch=_integer(epoch, "epoch"),
            drop_last=self.remainder == "drop",
        )

    def payload_count(self, epoch: int, batch_index: int, /) -> int:
        """Number of payloads (accumulation microsteps) of one global batch."""

        return len(self._plan_batch(epoch, batch_index))

    def start(self, epoch: int = 0, /) -> AtomisticLoaderCursor:
        return self._cursor(_integer(epoch, "epoch"), 0, 0)

    def require_cursor(self, cursor: AtomisticLoaderCursor, /) -> None:
        """Refuse a cursor of other sources, semantics, layout or position."""

        if not isinstance(cursor, AtomisticLoaderCursor):
            raise TypeError("cursor must be an AtomisticLoaderCursor.")
        if cursor.source_ids != self.mixture.source_ids:
            if len(cursor.source_ids) != len(self.mixture.source_ids):
                raise AtomisticLoaderError(
                    "drift", "The cursor binds a different number of source components."
                )
            changed = [
                position
                for position, (old, new) in enumerate(
                    zip(cursor.source_ids, self.mixture.source_ids)
                )
                if old != new
            ]
            raise AtomisticLoaderError(
                "drift",
                f"Source component(s) {changed} changed content since the cursor was "
                "recorded; resuming would not continue the same draws.",
            )
        if cursor.stream != self.stream:
            raise AtomisticLoaderError(
                "cursor", f"The cursor belongs to stream {cursor.stream!r}."
            )
        if cursor.semantics_id != self.semantics_id:
            raise AtomisticLoaderError(
                "cursor",
                "The cursor binds another mixture, binding or batch definition.",
            )
        if cursor.structure_id != self.structure_id:
            raise AtomisticLoaderError(
                "cursor", "The cursor binds another bucket plan or placement."
            )
        if cursor.loader_id != self.loader_id:
            raise AtomisticLoaderError("cursor", "The cursor binds another loader.")
        if cursor.batch_index >= self.batch_count:
            raise AtomisticLoaderError("cursor", "The cursor batch is outside the epoch.")

    def epoch(
        self,
        cursor: AtomisticLoaderCursor,
        /,
        *,
        staging: AtomisticLoaderStaging | None = None,
        execution_group: ExecutionGroup | None = None,
    ) -> AtomisticLoaderEpoch:
        """Iterate the remaining payloads of ``cursor.epoch`` from ``cursor``."""

        self.require_cursor(cursor)
        staging_ = AtomisticLoaderStaging() if staging is None else staging
        if not isinstance(staging_, AtomisticLoaderStaging):
            raise TypeError("staging must be AtomisticLoaderStaging.")
        if staging_.target == "device":
            bound = (staging_.device_ahead + 1) * max(
                layout.nbytes for layout in self.layouts
            )
            if staging_.max_device_bytes is None or bound > staging_.max_device_bytes:
                raise AtomisticLoaderError(
                    "capacity",
                    f"Staging {staging_.device_ahead + 1} payloads needs up to {bound} "
                    f"device bytes, above {staging_.max_device_bytes}.",
                )
            if execution_group is not None:
                _require_group(execution_group, self.placement)
            elif self.placement.shard_count != 1:
                raise AtomisticLoaderError(
                    "placement", "Sharded placements stage through an ExecutionGroup."
                )
        elif execution_group is not None:
            raise ValueError("Host staging takes no execution group.")
        return AtomisticLoaderEpoch(self, cursor, staging_, execution_group)

    def _cursor(
        self, epoch: int, batch_index: int, payload_index: int
    ) -> AtomisticLoaderCursor:
        return AtomisticLoaderCursor(
            loader_id=self.loader_id,
            semantics_id=self.semantics_id,
            structure_id=self.structure_id,
            stream=self.stream,
            source_ids=self.mixture.source_ids,
            epoch=epoch,
            batch_index=batch_index,
            payload_index=payload_index,
        )

    def _after(self, step: _Step, /) -> AtomisticLoaderCursor:
        if step.payload_index + 1 < step.payload_count:
            return self._cursor(step.epoch, step.batch_index, step.payload_index + 1)
        if step.batch_index + 1 < self.batch_count:
            return self._cursor(step.epoch, step.batch_index + 1, 0)
        return self._cursor(step.epoch + 1, 0, 0)

    def _steps(self, cursor: AtomisticLoaderCursor, /) -> Iterator[_Step]:
        for batch_index in range(cursor.batch_index, self.batch_count):
            steps = self._plan_batch(cursor.epoch, batch_index)
            first = cursor.payload_index if batch_index == cursor.batch_index else 0
            if first >= len(steps):
                raise AtomisticLoaderError(
                    "cursor", "The cursor payload index exceeds its batch's payloads."
                )
            yield from steps[first:]

    def _plan_batch(self, epoch: int, batch_index: int, /) -> tuple[_Step, ...]:
        plan = self.epoch_plan(epoch)
        positions = np.asarray(plan.batch(batch_index), dtype=np.int64)
        components, ordinals = self.mixture.resolve(epoch, positions)
        buckets = np.empty(positions.size, dtype=np.int64)
        for component, assignment in enumerate(self._assignment):
            chosen = components == component
            buckets[chosen] = assignment[ordinals[chosen]]
        # Shard slicing and real-draw tail padding of this batch's draws,
        # under the declared batch capacity, by the shared distributed owner.
        slicing = IndexEpochPlan(
            source_size=positions.size,
            batch_size=self.batch_size,
            shuffle=False,
            seed=0,
            epoch=0,
            drop_last=False,
        )
        rank = self.placement.rank
        shard_draws: list[npt.NDArray[np.int64]]
        if rank is None:
            shard_draws = [np.arange(positions.size, dtype=np.int64)]
        else:
            plan = DistributedIndexEpochPlan(slicing, rank)
            shard_draws = [
                np.asarray(local.indices, dtype=np.int64)[local.mask]
                for local in (plan.shard(0, s) for s in range(rank.shard_count))
            ]
        packs = [
            [
                self._pack(index, draws[buckets[draws] == index], components, ordinals)
                for index in range(len(self.buckets.buckets))
            ]
            for draws in shard_draws
        ]
        local_shards = self.placement.local_shards
        planned: list[tuple[int, tuple[_ShardPack, ...]]] = []
        for index in range(len(self.buckets.buckets)):
            for step in range(max(len(shard[index]) for shard in packs)):
                planned.append(
                    (
                        index,
                        tuple(
                            self._shard_pack(
                                packs[shard][index][step]
                                if step < len(packs[shard][index])
                                else np.zeros(0, dtype=np.int64),
                                positions,
                                components,
                                ordinals,
                            )
                            for shard in local_shards
                        ),
                    )
                )
        return tuple(
            _Step(
                epoch,
                batch_index,
                position,
                len(planned),
                bucket,
                int(positions.size),
                int(positions[0]),
                shards,
            )
            for position, (bucket, shards) in enumerate(planned)
        )

    def _pack(
        self,
        bucket_index: int,
        draws: npt.NDArray[np.int64],
        components: npt.NDArray[np.int64],
        ordinals: npt.NDArray[np.int64],
        /,
    ) -> list[npt.NDArray[np.int64]]:
        bucket = self.buckets.buckets[bucket_index]
        packed = [
            (position, capacity.rows)
            for position, level in enumerate(self.levels)
            if (capacity := bucket.capacity(level)).layout == "packed"
        ]
        packs: list[npt.NDArray[np.int64]] = []
        current: list[int] = []
        used = [0] * len(packed)
        for draw in draws.tolist():
            component, ordinal = int(components[draw]), int(ordinals[draw])
            need = [int(self._counts[component][level][ordinal]) for level, _ in packed]
            if current and (
                len(current) == bucket.cases
                or any(
                    total + extra > rows
                    for total, extra, (_, rows) in zip(used, need, packed)
                )
            ):
                packs.append(np.asarray(current, dtype=np.int64))
                current, used = [], [0] * len(packed)
            current.append(draw)
            used = [total + extra for total, extra in zip(used, need)]
        if current:
            packs.append(np.asarray(current, dtype=np.int64))
        return packs

    @staticmethod
    def _shard_pack(
        draws: npt.NDArray[np.int64],
        positions: npt.NDArray[np.int64],
        components: npt.NDArray[np.int64],
        ordinals: npt.NDArray[np.int64],
        /,
    ) -> _ShardPack:
        return _ShardPack(
            tuple(int(value) for value in positions[draws]),
            tuple(int(value) for value in components[draws]),
            tuple(int(value) for value in ordinals[draws]),
        )

    # Preparation -----------------------------------------------------------------

    def _prepare(self, step: _Step, /) -> AtomisticLoaderItem:
        layout = self.layouts[step.bucket]
        bucket = self.buckets.buckets[step.bucket]
        cases = bucket.cases
        shards = len(self.placement.local_shards)
        count = shards * cases
        component = np.empty(count, dtype=np.int64)
        ordinal = np.empty(count, dtype=np.int64)
        position = np.empty(count, dtype=np.int64)
        valid = np.zeros(count, dtype=np.bool_)
        alias = np.empty(count, dtype=np.int64)
        for shard, pack in enumerate(step.shards):
            base = shard * cases
            size = len(pack.positions)
            real = slice(base, base + size)
            padding = slice(base + size, base + cases)
            if size:
                component[real] = pack.components
                ordinal[real] = pack.ordinals
                position[real] = pack.positions
                valid[real] = True
                alias[real] = np.arange(base, base + size)
                component[padding] = pack.components[0]
                ordinal[padding] = pack.ordinals[0]
                position[padding] = pack.positions[0]
                alias[padding] = base
            else:
                anchor = self._anchors[step.bucket]
                if anchor is None:
                    raise RuntimeError("A planned bucket step has no anchor record.")
                component[padding] = anchor[0]
                ordinal[padding] = anchor[1]
                position[padding] = step.first_position
                alias[padding] = -1
        records, record = self._read(component, ordinal)
        shard_offset = self.placement.local_shards[0]
        rows = {
            level: self._level_rows(level, bucket, records, record, valid, alias, shards)
            for level in self.levels
        }
        structure_rows = _Rows(
            record,
            np.arange(count, dtype=np.int64),
            np.arange(count, dtype=np.int64),
            np.ones(count, dtype=np.bool_),
            np.ones(count, dtype=np.int64),
            np.arange(count, dtype=np.int64),
            cases,
        )
        levels = []
        for level, parent in zip(self.levels, self._parents):
            plan = rows[level]
            row_parent = None
            if parent is not None:
                local = np.zeros(plan.source.size, dtype=np.int64)
                present = plan.source >= 0
                local[present] = records.parent_local[level][plan.source[present]]
                row_parent = self._rebase(
                    local, present, None, plan, rows[parent], shard_offset
                ).astype(np.int32)
            levels.append(
                AtomisticPayloadLevel(
                    level,
                    bucket.capacity(level).layout,
                    plan.rows_per_shard,
                    row_mask=plan.row_mask,
                    row_case=(plan.row_case + shard_offset * cases).astype(np.int32),
                    case_count=plan.case_count.astype(np.int32),
                    row_parent=row_parent,
                )
            )
        atoms = rows[self.mixture.schema.atom_level]
        identities = records.stable_ids[atoms.source].view(np.uint64)
        stable_words = np.stack(
            (
                (identities >> np.uint64(32)).astype(np.uint32),
                identities.astype(np.uint32),
            ),
            axis=-1,
        )
        values: list[npt.NDArray[Any]] = []
        available: list[npt.NDArray[np.bool_]] = []
        for slot in self._slots:
            plan = (
                structure_rows
                if slot.level == self.mixture.schema.structure_level
                else rows[slot.level]
            )
            gathered, present = self._gather(slot, records, plan)
            if slot.target is not None:
                target = (
                    structure_rows
                    if slot.target == self.mixture.schema.structure_level
                    else rows[slot.target]
                )
                gathered = self._narrow(
                    slot,
                    self._rebase(
                        gathered, present, slot.sentinel, plan, target, shard_offset
                    ),
                )
            else:
                gathered = self._convert(slot, gathered)
            values.append(gathered)
            available.append(present & plan.row_mask)
        probabilities = np.asarray(self.mixture.record_probabilities, dtype=np.float64)
        weights = np.asarray(self.mixture.case_weights, dtype=np.float64)
        payload = AtomisticLoaderPayload(
            layout,
            case_valid=valid,
            case_weight=np.where(valid, weights[component], 0.0),
            draw_probability=probabilities[component],
            draw_index=position.astype(np.uint32),
            case_epoch=np.full(count, step.epoch, dtype=np.uint32),
            case_component=component.astype(np.int32),
            record_ordinal=ordinal.astype(np.uint32),
            levels=levels,
            stable_id_words=stable_words,
            values=values,
            available=available,
        )
        provenance = AtomisticPayloadProvenance(
            loader_id=self.loader_id,
            epoch=step.epoch,
            batch_index=step.batch_index,
            payload_index=step.payload_index,
            payload_count=step.payload_count,
            bucket=bucket.name,
            global_valid_draws=step.global_valid,
            draw_indices=position.tolist(),
            valid=valid.tolist(),
            source_ids=[records.source_ids[value] for value in record.tolist()],
            record_ids=[records.record_ids[value] for value in record.tolist()],
            lineage_ids=[records.lineage_ids[value] for value in record.tolist()],
            record_digests=[records.record_digests[value] for value in record.tolist()],
        )
        return AtomisticLoaderItem(
            payload,
            provenance,
            self._after(step),
            closes_batch=step.payload_index + 1 == step.payload_count,
        )

    def _read(
        self, component: npt.NDArray[np.int64], ordinal: npt.NDArray[np.int64], /
    ) -> tuple[_Records, npt.NDArray[np.int64]]:
        record = np.empty(component.size, dtype=np.int64)
        batches: list[tuple[int, AtomisticStructureBatch]] = []
        offset = 0
        for index in np.unique(component).tolist():
            chosen = component == index
            wanted = np.unique(ordinal[chosen])
            batch = self.mixture.components[index].source.read(
                wanted, fields=self._fields, index_basis="structure-local"
            )
            record[chosen] = offset + np.searchsorted(wanted, ordinal[chosen])
            batches.append((index, batch))
            offset += wanted.size
        schema = self.mixture.schema
        counts = {
            level: np.concatenate([batch.structure_counts[level] for _, batch in batches])
            for level in self.levels
        }
        starts = {level: _exclusive_cumsum(value) for level, value in counts.items()}
        parent_local: dict[str, npt.NDArray[np.int64]] = {}
        for level, parent in zip(self.levels, self._parents):
            if parent is None:
                continue
            per_parent = np.concatenate(
                [batch.parent_counts[level] for _, batch in batches]
            )
            parent_row = np.repeat(np.arange(per_parent.size, dtype=np.int64), per_parent)
            owner = np.repeat(
                np.arange(counts[level].size, dtype=np.int64), counts[level]
            )
            parent_local[level] = parent_row - starts[parent][owner]
        values = {
            name: np.concatenate([batch.values[name] for _, batch in batches])
            for name in self._fields
        }
        available = {
            name: np.concatenate([batch.available[name] for _, batch in batches])
            for name in self._fields
            if schema.field(name).availability == "optional"
        }
        records = _Records(
            tuple(
                self.mixture.source_ids[index]
                for index, batch in batches
                for _ in batch.record_ids
            ),
            tuple(value for _, batch in batches for value in batch.record_ids),
            tuple(value for _, batch in batches for value in batch.lineage_ids),
            tuple(value for _, batch in batches for value in batch.record_digests),
            counts,
            starts,
            parent_local,
            np.concatenate([batch.stable_ids for _, batch in batches]),
            values,
            available,
        )
        return records, record

    def _level_rows(
        self,
        level: str,
        bucket: AtomisticWorkloadBucket,
        records: _Records,
        record: npt.NDArray[np.int64],
        valid: npt.NDArray[np.bool_],
        alias: npt.NDArray[np.int64],
        shards: int,
        /,
    ) -> _Rows:
        capacity = bucket.capacity(level)
        cases = bucket.cases
        count = shards * cases
        cases_index = np.arange(count, dtype=np.int64)
        counts = records.counts[level][record]
        starts = records.starts[level][record]
        if capacity.layout == "per-case":
            width = capacity.rows
            slot = np.arange(width, dtype=np.int64)
            occupied = slot[None, :] < counts[:, None]
            fallback = np.where(counts > 0, starts, -1)
            fallback_case = cases_index.copy()
            for shard in range(shards):
                local = slice(shard * cases, (shard + 1) * cases)
                donors = np.flatnonzero(counts[local] > 0)
                lacking = counts[local] == 0
                if donors.size:
                    donor = shard * cases + int(donors[0])
                    fallback[local][lacking] = starts[donor]
                    fallback_case[local][lacking] = donor
                else:
                    fallback_case[local] = shard * cases
            source = np.where(
                occupied, starts[:, None] + slot[None, :], fallback[:, None]
            )
            source_case = np.where(occupied, cases_index[:, None], fallback_case[:, None])
            return _Rows(
                source.reshape(-1),
                np.repeat(cases_index, width),
                source_case.reshape(-1),
                occupied.reshape(-1),
                counts,
                cases_index * width,
                cases * width,
            )
        width = capacity.rows
        real_counts = np.where(valid, counts, 0)
        source = np.full(shards * width, -1, dtype=np.int64)
        row_case = np.empty(shards * width, dtype=np.int64)
        source_case = np.empty(shards * width, dtype=np.int64)
        mask = np.zeros(shards * width, dtype=np.bool_)
        base = np.empty(count, dtype=np.int64)
        for shard in range(shards):
            local = slice(shard * cases, (shard + 1) * cases)
            shard_counts = real_counts[local]
            offsets = _exclusive_cumsum(shard_counts)
            total = int(shard_counts.sum())
            if total > width:
                raise RuntimeError("A packed shard exceeds its planned row capacity.")
            owner = shard * cases + np.repeat(
                np.arange(cases, dtype=np.int64), shard_counts
            )
            filled = slice(shard * width, shard * width + total)
            slack = slice(shard * width + total, (shard + 1) * width)
            source[filled] = starts[owner] + (
                np.arange(total, dtype=np.int64) - offsets[owner - shard * cases]
            )
            row_case[filled] = owner
            source_case[filled] = owner
            mask[filled] = True
            donors = np.flatnonzero(shard_counts > 0)
            donor = shard * cases + (int(donors[0]) if donors.size else 0)
            if donors.size:
                source[slack] = starts[donor]
            row_case[slack] = donor
            source_case[slack] = donor
            base[local] = shard * width + offsets
        shard_start = (cases_index // cases) * width
        case_base = np.where(alias >= 0, base[np.maximum(alias, 0)], shard_start)
        return _Rows(source, row_case, source_case, mask, real_counts, case_base, width)

    def _gather(
        self, slot: _Slot, records: _Records, plan: _Rows, /
    ) -> tuple[npt.NDArray[Any], npt.NDArray[np.bool_]]:
        stored = records.values[slot.field]
        present = plan.source >= 0
        values = np.zeros((plan.source.size, *slot.shape), dtype=stored.dtype)
        values[present] = stored[plan.source[present]]
        if slot.optional:
            available = np.zeros(plan.source.size, dtype=np.bool_)
            available[present] = records.available[slot.field][plan.source[present]]
        else:
            available = present.copy()
        return values, available

    def _rebase(
        self,
        values: npt.NDArray[Any],
        available: npt.NDArray[np.bool_],
        sentinel: int | None,
        plan: _Rows,
        target: _Rows,
        shard_offset: int,
        /,
    ) -> npt.NDArray[np.int64]:
        """Map structure-local target rows to global payload rows."""

        wide = values.astype(np.int64)
        expand = (slice(None),) + (None,) * (wide.ndim - 1)
        cases = target.case_base.size // len(self.placement.local_shards)
        shard = plan.row_case // cases
        absent = np.zeros(wide.shape, dtype=np.bool_)
        if sentinel is not None:
            absent = wide == sentinel
        shift = available[expand] & ~absent
        rebased = np.where(
            shift,
            target.case_base[plan.source_case][expand] + wide,
            (shard * target.rows_per_shard)[expand],
        )
        rebased = rebased + shard_offset * target.rows_per_shard
        limit = self.placement.shard_count * target.rows_per_shard
        if np.any((rebased < 0) | (rebased >= limit)):
            raise RuntimeError("A rebased payload index left its target rows.")
        return np.where(absent, wide, rebased)

    @staticmethod
    def _narrow(slot: _Slot, values: npt.NDArray[np.int64], /) -> npt.NDArray[Any]:
        output = np.dtype(slot.dtype)
        info = np.iinfo(output)
        if values.size and (values.min() < info.min or values.max() > info.max):
            raise AtomisticLoaderError(
                "capacity", f"Slot {slot.name!r} rows exceed the {output} range."
            )
        return values.astype(output)

    @staticmethod
    def _convert(slot: _Slot, values: npt.NDArray[Any], /) -> npt.NDArray[Any]:
        output = np.dtype(slot.dtype)
        if slot.factor is not None:
            wide = values.astype(np.complex128 if output.kind == "c" else np.float64)
            return (wide * float(slot.factor)).astype(output)
        if output.kind in "iu" and output != values.dtype:
            info = np.iinfo(output)
            if values.size and (values.min() < info.min or values.max() > info.max):
                raise AtomisticLoaderError(
                    "binding", f"Slot {slot.name!r} values exceed the {output} range."
                )
        return values.astype(output, copy=False)


class AtomisticLoaderEpoch(Iterator[AtomisticLoaderItem]):
    """Closable remaining-epoch payload iterator with bounded staging."""

    def __init__(
        self,
        loader: AtomisticStructureLoader,
        cursor: AtomisticLoaderCursor,
        staging: AtomisticLoaderStaging,
        execution_group: ExecutionGroup | None,
        /,
    ) -> None:
        self._loader = loader
        self._staging = staging
        self._group = execution_group
        self._staged: deque[AtomisticLoaderItem] = deque()
        capacity = staging.host_prefetch if loader.background_read_safe else 0
        self._host: BoundedPrefetchIterator[_Step, AtomisticLoaderItem] = (
            BoundedPrefetchIterator(
                loader._steps(cursor),
                loader._prepare,
                capacity=capacity,
                thread_name=f"phydrax-atomistic-loader-{loader.stream}-{cursor.epoch}",
            )
        )

    @property
    def host_prefetch(self) -> int:
        return self._host.capacity

    @property
    def closed(self) -> bool:
        return self._host.closed and not self._staged

    def __iter__(self) -> AtomisticLoaderEpoch:
        return self

    def __next__(self) -> AtomisticLoaderItem:
        if self._staging.target == "host":
            return next(self._host)
        try:
            while len(self._staged) <= self._staging.device_ahead:
                try:
                    item = next(self._host)
                except StopIteration:
                    break
                self._staged.append(
                    AtomisticLoaderItem(
                        _stage(item.payload, self._loader.placement, self._group),
                        item.provenance,
                        item.cursor,
                        closes_batch=item.closes_batch,
                    )
                )
        except BaseException:
            self.close()
            raise
        if not self._staged:
            raise StopIteration
        return self._staged.popleft()

    def close(self) -> None:
        self._host.close()
        self._staged.clear()

    def __enter__(self) -> AtomisticLoaderEpoch:
        self._host.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


__all__ = [
    "AtomisticFieldBinding",
    "AtomisticLevelCapacity",
    "AtomisticLevelLayout",
    "AtomisticLoaderCursor",
    "AtomisticLoaderEpoch",
    "AtomisticLoaderError",
    "AtomisticLoaderFailure",
    "AtomisticLoaderItem",
    "AtomisticLoaderPayload",
    "AtomisticLoaderPlacement",
    "AtomisticLoaderRemainder",
    "AtomisticLoaderStaging",
    "AtomisticLoaderTarget",
    "AtomisticMixtureComponent",
    "AtomisticMixtureDraw",
    "AtomisticMixtureObjective",
    "AtomisticPayloadLayout",
    "AtomisticPayloadLevel",
    "AtomisticPayloadProvenance",
    "AtomisticSourceMixture",
    "AtomisticSplitEvidence",
    "AtomisticStructureLoader",
    "AtomisticWorkloadBucket",
    "AtomisticWorkloadBuckets",
]
