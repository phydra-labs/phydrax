#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Describe interval-evaluable source pieces without importing carrier classes."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from fractions import Fraction
from types import BuiltinFunctionType, FunctionType, MethodType
from typing import cast, Generic, TypeVar

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey

from ..._fingerprint import array_tree_fingerprint
from ..._host_storage import (
    declared_object_storage_bytes,
    immutable_host_array_layout,
    immutable_host_array_layout_bytes,
    retained_host_storage_fanout_upper,
)
from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked


_CarrierT = TypeVar("_CarrierT", bound=StrictModule, covariant=True)


@checked
def carrier_payload(carrier: StrictModule, /) -> dict[str, object]:
    """Fingerprint a curve/surface carrier from its family, metadata, and arrays."""
    return {
        "family": type(carrier).__name__,
        "structure": str(jax.tree_util.tree_structure(carrier)),
        "arrays": array_tree_fingerprint(carrier),
    }


# Host objects that ``carrier_payload`` and the enclosing ``canonical_fingerprint``
# create per array-record occurrence of ``_fingerprint._array_records``: the
# flattened ``(path, leaf)`` pair, record and signature tuples (3 x 56 B) and
# their list slots, the NumPy conversion header (112 B), the signature dict,
# shape list and dtype string, its canonical-payload copy (2 x 232 B), the
# record metadata JSON and the signature's JSON chunk objects (about 12 x 57 B)
# with their fixed text (3 x 48 B), and a ``<root>`` path.
_RECORD_BYTES = 2048
# Per path key of an occurrence: one key object (56 B) and its tuple slot.
_PATH_KEY_BYTES = 64
# Per path character: the keystr text (<= 4 B, UCS-4), the metadata JSON
# (<= 6 B escaped under ensure_ascii) and the signature JSON chunk, joined text
# and UTF-8 bytes (3 x 6 B), rounded up.
_PATH_CHARACTER_BYTES = 32
# Per array axis: two shape lists with their integers (2 x 36 B), metadata and
# signature JSON digits (4 x 21 B) and two separator chunks (2 x 57 B).
_ARRAY_AXIS_BYTES = 256
# Per PyTreeDef text character: the string (<= 4 B) and its payload JSON chunk,
# joined text and UTF-8 bytes (3 x 6 B escaped), rounded up.
_STRUCTURE_CHARACTER_BYTES = 24
# The PyTreeDef descriptor per node, as the declared host-storage owner bounds it.
_TREEDEF_NODE_BYTES = 256
# One payload mapping with its family and digest strings.
_PAYLOAD_BYTES = 1024
# ``iter_array_bytes`` stages at most one 65536-byte noncontiguous chunk.
_ARRAY_CHUNK_BYTES = 65536
# A JSON container in the enclosing fingerprint: object, copy and chunks, plus
# per item a slot, its canonical copy and separator chunks.
_CONTAINER_BYTES = 256
_CONTAINER_ITEM_BYTES = 128
# A JSON string: object overhead plus text, escaped JSON chunk, joined text
# and UTF-8 bytes per character.
_JSON_STRING_BYTES = 256
_JSON_CHARACTER_BYTES = 24
# One subtree summary record and its slot while the admission walks a tree.
_SUMMARY_BYTES = 256
# One summary retained in a caller-owned memo across proofs, excluding the
# layout a byte-backed bank leaf binds (``immutable_host_array_layout_bytes``).
_RETAINED_SUMMARY_BYTES = 512
# One bank-summary reference in a collected or composite tuple, with growth.
_BANK_SLOT_BYTES = 16
# One flattened child: its (key, child) pair (56 B), key object (56 B) and slot.
_CHILD_BYTES = 128
# PyTreeDef text of one custom node around its type name, auxiliary repr and
# children ("CustomNode(", "[", "], [", "])"), and the "PyTreeDef()" wrapper.
_NODE_TEXT = 24
_TREEDEF_TEXT = 16
# Equinox's printer: a newline and two-space indentation per nesting level.
_INDENT = 2


class _AuxProbe(StrictModule):
    value: object


# Equinox writes this sentinel into custom-node auxiliary data for each dynamic
# field; its repr is the constant "Missing".
_EQUINOX_MISSING = cast(
    tuple[object, ...], jax.tree_util.flatten_one_level(_AuxProbe(None))[1]
)[0]


@dataclass(frozen=True, slots=True)
class CarrierAdmission:
    """Position-independent fingerprint allowance of one PyTree subtree.

    Totals cover every node and leaf occurrence with paths relative to the
    subtree root; ``text_characters`` bounds its PyTreeDef text.
    """

    owner: object
    nodes: int
    occurrences: int
    path_keys: int
    path_characters: int
    axes: int
    leaf_bytes: int
    text_characters: int
    immutable: bool
    # ``immutable_host_array_layout`` under which an immutable byte-backed host
    # bank leaf was admitted; ``None`` for every other summary.
    layout: tuple[object, ...] | None = None
    # The bound bank leaves of an immutable composite, by occurrence. Such a
    # composite is reusable only while each of them keeps its layout.
    banks: "tuple[CarrierAdmission, ...]" = ()


def carrier_admission_current(summary: CarrierAdmission, /) -> bool:
    """Whether every byte-backed bank of a summary keeps its admitted layout.

    A bank's values are fixed by its immutable bytes owner, but its NumPy
    dtype, shape and strides may be reassigned in place; the full layout is
    validated again without a content scan.
    """
    if summary.layout is not None:
        return immutable_host_array_layout(cast(np.ndarray, summary.owner)) == summary.layout
    return all(
        immutable_host_array_layout(cast(np.ndarray, bank.owner)) == bank.layout
        for bank in summary.banks
    )


def _digits(value: int, /) -> int:
    """Decimal digits and sign of an integer (0.31 > log10(2))."""
    return value.bit_length() * 31 // 100 + 2


def _generated_dataclass_repr(kind: type, /) -> bool:
    representation = getattr(kind, "__repr__", None)
    if representation is StrictModule.__repr__:
        return True
    code = getattr(getattr(representation, "__wrapped__", None), "__code__", None)
    return code is not None and code.co_filename == "<string>"


def _repr_upper(value: object, depth: int, /) -> int:
    """Bound ``repr(value)`` characters without building it.

    Covers Python reprs, generated dataclass reprs and Equinox's printer (one
    item per line at two spaces per nesting level). Values without a declared
    representation bound refuse before the payload is serialized.
    """
    line = _INDENT * (depth + 1) + 3
    match value:
        case None | bool():
            return 5
        case int():
            return _digits(value)
        case float():
            return 32
        case complex():
            return 66
        case str():
            return 2 + 10 * len(value)
        case bytes():
            return 3 + 4 * len(value)
        case Fraction():
            return 16 + _digits(value.numerator) + _digits(value.denominator)
        case Enum():
            return (
                8
                + len(type(value).__qualname__)
                + len(value.name)
                + _repr_upper(value.value, depth + 1)
            )
        case np.generic():
            return 96
        case np.dtype():
            return 64 + 64 * len(value.descr)
        case Array() | np.ndarray():
            return 64 + 8 * value.ndim + 32 * value.size
        case type():
            return 16 + len(value.__module__) + len(value.__qualname__)
        case tuple() | list() | set() | frozenset():
            return 16 + sum(_repr_upper(item, depth + 1) + line for item in value)
        case dict():
            return 16 + sum(
                _repr_upper(key, depth + 1) + _repr_upper(item, depth + 1) + line
                for key, item in value.items()
            )
        case FunctionType() | BuiltinFunctionType():
            name = str(getattr(value, "__qualname__", ""))
            return 48 + len(getattr(value, "__module__", None) or "") + len(name)
        case MethodType():
            return (
                48
                + len(str(getattr(value.__func__, "__qualname__", "")))
                + _repr_upper(value.__self__, depth + 1)
            )
        case _ if value is _EQUINOX_MISSING:
            return 7
        case _ if (
            is_dataclass(value)
            and not isinstance(value, type)
            and _generated_dataclass_repr(type(value))
        ):
            return (
                16
                + len(type(value).__qualname__)
                + sum(
                    len(member.name)
                    + 3
                    + _repr_upper(getattr(value, member.name), depth + 1)
                    + line
                    for member in fields(value)
                )
            )
        case _ if type(value).__repr__ is object.__repr__:
            return 48 + len(type(value).__module__) + len(type(value).__qualname__)
        case _:
            raise TypeError(
                "Carrier fingerprint static metadata must have a declared "
                f"representation bound; got {type(value).__qualname__}."
            )


def _frozen_dataclass(value: object, /) -> bool:
    """Whether an object is a StrictModule or frozen dataclass instance."""
    return (
        is_dataclass(value)
        and not isinstance(value, type)
        and (
            isinstance(value, StrictModule)
            or getattr(type(value), "__dataclass_params__").frozen
        )
    )


def _static_immutable(value: object, /) -> bool:
    """Whether serialized static metadata cannot change after admission."""
    match value:
        case None | bool() | int() | float() | complex() | str() | bytes():
            return True
        case Fraction() | Enum() | np.generic() | np.dtype() | Array():
            return True
        case tuple() | frozenset():
            return all(_static_immutable(item) for item in value)
        case _ if value is _EQUINOX_MISSING:
            return True
        case _ if (
            is_dataclass(value)
            and not isinstance(value, type)
            and _frozen_dataclass(value)
        ):
            return all(
                _static_immutable(getattr(value, member.name))
                for member in fields(value)
            )
        case _:
            # Callables, types, NumPy arrays, mutable containers and unknown
            # objects can change their representation: never memoized.
            return False


def _leaf_admission(leaf: object, /) -> CarrierAdmission:
    """One leaf as ``_array_records`` converts it, bounded before conversion."""
    axes, data, immutable = 0, 0, True
    layout: tuple[object, ...] | None = None
    match leaf:
        case Array():
            # Every occurrence is converted; charged per occurrence.
            axes, data = leaf.ndim, leaf.nbytes
        case np.ndarray():
            # Recorded without a copy. Only a read-only view chain ending in
            # immutable bytes (``immutable_host_array``) keeps its values; its
            # dtype, shape and strides stay assignable, so the summary binds
            # its full layout and is reused only while that is unchanged.
            axes = leaf.ndim
            layout = immutable_host_array_layout(leaf)
            immutable = layout is not None
        case bool() | int() | float() | complex() | np.generic():
            data = 16
        case str():
            data = 4 * len(leaf)
        case bytes():
            data = len(leaf)
        case _ if any(
            hasattr(type(leaf), name)
            for name in ("__array__", "__array_interface__", "__array_struct__", "__len__")
        ):
            raise TypeError(
                "Carrier fingerprint leaves must have a declared array conversion "
                f"bound; got {type(leaf).__qualname__}."
            )
        case _:
            # Opaque objects become an object scalar the payload skips.
            data = 16
    return CarrierAdmission(leaf, 1, 1, 0, 0, axes, data, 1, immutable, layout)


def _key_characters(key: object, /) -> int:
    """``jax.tree_util.keystr`` characters of one path key, without building it."""
    match key:
        case GetAttrKey():
            return 1 + len(key.name)
        case SequenceKey():
            return 2 + _digits(key.idx)
        case FlattenedIndexKey():
            return 2 + _digits(key.key)
        case DictKey():
            return 2 + _repr_upper(key.key, 0)
        case _:
            raise TypeError(
                "Carrier fingerprint paths must use JAX key entries; got "
                f"{type(key).__qualname__}."
            )


_LEAF_TYPES = (Array, np.ndarray, np.generic, bool, int, float, complex, str, bytes)


def _fanout_upper(node: object, /) -> int:
    """Declared children of a PyTree node before it is flattened."""
    if node is None:
        return 0
    if isinstance(node, (tuple, list, Mapping)):
        return len(node)
    if is_dataclass(node) and not isinstance(node, type):
        return len(type(node).__dataclass_fields__)
    return retained_host_storage_fanout_upper(node)


def _node_text_and_immutability(node: object, aux: object, /) -> tuple[int, bool]:
    """PyTreeDef text of one node without children, and whether it is fixed."""
    match node:
        case None:
            return 4, True
        case list() if type(node) is list:
            return 2, False
        case tuple() if type(node) is tuple:
            return 2, True
        case dict() if type(node) is dict:
            return 2, False
    text = _NODE_TEXT + len(type(node).__qualname__) + _repr_upper(aux, 0)
    frozen = _frozen_dataclass(node)
    return text, frozen and _static_immutable(aux)


def _tree_admission(
    node: object,
    budget: CoordinateEnclosureBudget,
    memo: dict[int, CarrierAdmission],
    local: dict[int, CarrierAdmission],
    retain_memo: Callable[[int], None] | None,
    /,
) -> CarrierAdmission:
    """Summarize what ``carrier_payload`` serializes for one subtree.

    Every one-level flatten is admitted at its declared fanout before it runs;
    a node exceeding its declared fanout refuses as an unsupported owner.
    ``retain_memo`` receives the bytes of each memo summary before it is kept.
    """
    for table in (memo, local):
        previous = table.get(id(node))
        if previous is not None and previous.owner is node:
            if carrier_admission_current(previous):
                return previous
            # A bound bank's dtype, shape or strides changed in place: its
            # summary is earned again from the current layout.
            memo.pop(id(node), None)
            local.pop(id(node), None)
            break
    if isinstance(node, _LEAF_TYPES):
        budget.reserve(
            0,
            _SUMMARY_BYTES
            + (immutable_host_array_layout_bytes(node.ndim) if isinstance(node, np.ndarray) else 0),
        )
        summary = _leaf_admission(node)
    else:
        fanout = _fanout_upper(node)
        budget.reserve(0, _SUMMARY_BYTES + _CHILD_BYTES * fanout)
        try:
            flattened, aux = jax.tree_util.flatten_one_level_with_keys(node)
            children = tuple(flattened)
        except ValueError:
            summary = _leaf_admission(node)
        else:
            if len(children) > fanout:
                raise TypeError(
                    "Carrier fingerprint PyTree nodes must respect their declared "
                    f"fanout; got {type(node).__qualname__}."
                )
            text, immutable = _node_text_and_immutability(node, aux)
            nodes, occurrences, path_keys, path_characters = 1, 0, 0, 0
            axes = leaf_bytes = 0
            banks: list[CarrierAdmission] = []
            for key, child in children:
                part = _tree_admission(child, budget, memo, local, retain_memo)
                characters = _key_characters(key)
                nodes += part.nodes
                occurrences += part.occurrences
                path_keys += part.path_keys + part.occurrences
                path_characters += part.path_characters + characters * part.occurrences
                axes += part.axes
                leaf_bytes += part.leaf_bytes
                # ", " between children; dict nodes also print "key: ".
                text += part.text_characters + 2
                if isinstance(key, DictKey):
                    text += characters
                immutable = immutable and part.immutable
                bound = (part,) if part.layout is not None else part.banks
                if immutable and bound:
                    # Collected references and the final tuple.
                    budget.reserve(0, 2 * _BANK_SLOT_BYTES * len(bound))
                    banks.extend(bound)
            summary = CarrierAdmission(
                node,
                nodes,
                occurrences,
                path_keys,
                path_characters,
                axes,
                leaf_bytes,
                text,
                immutable,
                None,
                tuple(banks) if immutable else (),
            )
    local[id(node)] = summary
    if summary.immutable and memo is not local:
        # The record, a bank leaf's layout and a composite's bank references.
        retained = _RETAINED_SUMMARY_BYTES + _BANK_SLOT_BYTES * len(summary.banks)
        if summary.layout is not None:
            retained += immutable_host_array_layout_bytes(summary.axes)
        budget.reserve(1, retained)
        if retain_memo is not None:
            retain_memo(retained)
        memo[id(node)] = summary
    return summary


def _payload_admission(
    carrier: StrictModule,
    budget: CoordinateEnclosureBudget,
    memo: dict[int, CarrierAdmission],
    local: dict[int, CarrierAdmission],
    retain_memo: Callable[[int], None] | None,
    banks: list[CarrierAdmission],
    /,
) -> bool:
    """Admit one ``carrier_payload`` and its share of the enclosing JSON."""
    summary = _tree_admission(carrier, budget, memo, local, retain_memo)
    bound = (summary,) if summary.layout is not None else summary.banks
    if summary.immutable and bound:
        budget.reserve(0, _BANK_SLOT_BYTES * len(bound))
        banks.extend(bound)
    budget.reserve(
        0,
        _PAYLOAD_BYTES
        + _TREEDEF_NODE_BYTES * summary.nodes
        + _STRUCTURE_CHARACTER_BYTES * (summary.text_characters + _TREEDEF_TEXT)
        + _RECORD_BYTES * summary.occurrences
        + _PATH_KEY_BYTES * summary.path_keys
        + _PATH_CHARACTER_BYTES * summary.path_characters
        + _ARRAY_AXIS_BYTES * summary.axes
        + summary.leaf_bytes,
    )
    return summary.immutable


def _json_admission(
    source: object,
    budget: CoordinateEnclosureBudget,
    ancestors: tuple[int, ...],
    memo: dict[int, CarrierAdmission],
    local: dict[int, CarrierAdmission],
    retain_memo: Callable[[int], None] | None,
    banks: list[CarrierAdmission],
    /,
) -> bool:
    """Admit the canonical JSON value a caller builds around carrier payloads."""
    if isinstance(source, StrictModule):
        return _payload_admission(source, budget, memo, local, retain_memo, banks)
    if id(source) in ancestors:
        raise ValueError("Correspondence fingerprint source must be acyclic.")
    ancestors = (*ancestors, id(source))

    def items(count: int, values: Iterable[object], /) -> bool:
        # The container is admitted before any of its items is visited.
        budget.reserve(0, _CONTAINER_BYTES + _CONTAINER_ITEM_BYTES * count)
        immutable = True
        for value in values:
            immutable = (
                _json_admission(value, budget, ancestors, memo, local, retain_memo, banks)
                and immutable
            )
        return immutable

    match source:
        case Array() | np.ndarray() | np.generic():
            # ``_canonical_payload`` fingerprints one ``<root>`` array record.
            ndim = 0 if isinstance(source, np.generic) else source.ndim
            budget.reserve(
                0,
                _PAYLOAD_BYTES
                + _RECORD_BYTES
                + _ARRAY_AXIS_BYTES * ndim
                + (source.nbytes if isinstance(source, Array) else 0),
            )
            return not isinstance(source, np.ndarray)
        case str():
            budget.reserve(0, _JSON_STRING_BYTES + _JSON_CHARACTER_BYTES * len(source))
            return True
        case Fraction():
            return items(2, (source.numerator, source.denominator))
        case Mapping():
            items(2 * len(source), (item for pair in source.items() for item in pair))
            return False
        case tuple() | frozenset():
            return items(len(source), source)
        case list() | set():
            items(len(source), source)
            return False
        case _ if is_dataclass(source) and not isinstance(source, type):
            members = type(source).__dataclass_fields__
            immutable = items(
                2 * len(members),
                (item for name in members for item in (name, getattr(source, name))),
            )
            return immutable and _frozen_dataclass(source)
        case _:
            budget.reserve(0, declared_object_storage_bytes(source))
            return type(source) in (type(None), bool, int, float, complex, bytes)


def admit_carrier_fingerprint(
    source: object,
    budget: CoordinateEnclosureBudget,
    /,
    *,
    memo: dict[int, CarrierAdmission] | None = None,
    retain_memo: Callable[[int], None] | None = None,
    banks: list[CarrierAdmission] | None = None,
) -> bool:
    """Admit live payload metadata and host transfers before fingerprinting.

    The allowance follows what ``carrier_payload`` serializes and is admitted
    before that payload, and before each allocation of this walk, is created:
    the PyTreeDef descriptor and its text (bounded from each node's type and
    static auxiliary metadata without building reprs), and one array record
    per leaf occurrence of ``tree_flatten_with_path`` with its path and
    conversion. Each one-level flatten is admitted at its declared fanout
    first. Containers, scalars and arrays around payloads are charged as the
    enclosing canonical JSON. PyTree leaves the payload does not traverse (for
    example request caches) are charged as the object scalars it skips. Leaves
    and static metadata without a declared bound refuse. Shared immutable
    subtrees are summarized once and are still charged at every occurrence; a
    caller-owned ``memo`` retains them across proofs, and ``retain_memo``
    receives each retained summary's bytes before it is kept. Returns whether
    every admitted value is immutable; a byte-backed host bank counts only
    under the layout it was admitted with. An immutable result appends those
    bank summaries, by occurrence, to ``banks``: a caller memoizing anything
    derived from the fingerprint must recheck them with
    ``carrier_admission_current`` before each reuse. This is byte admission
    only, not a new scientific work charge.
    """
    budget.reserve(0, _ARRAY_CHUNK_BYTES)
    local: dict[int, CarrierAdmission] = {}
    collected: list[CarrierAdmission] = [] if banks is None else banks
    start = len(collected)
    immutable = _json_admission(
        source, budget, (), local if memo is None else memo, local, retain_memo, collected
    )
    if not immutable:
        del collected[start:]
    return immutable


class CoefficientEnclosureSource(ABC):
    """Provide source coefficient intervals to the numerical interval interpreter."""

    @abstractmethod
    def coefficient_enclosures(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
        """Return nominal coefficient arrays and their lower and upper bounds."""
        raise NotImplementedError


class CurveCoordinateResult(StrictModule):
    """Expose coordinates from a coupled curve result without adding stored fields."""

    __strict_abstract__ = True

    point: eqx.AbstractVar[Array]


class RootCurveCoordinateSource(ABC):
    """Identify a curve whose coordinate evaluation executes a coupled root solve."""

    @abstractmethod
    def evaluate(self, parameters: Array, /) -> CurveCoordinateResult:
        """Return coordinates together with the coupled root's numerical evidence."""
        raise NotImplementedError


def child_coefficient_enclosures(
    system: StrictModule, /
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Collect construction intervals from descendant sources, excluding the root."""
    children = jax.tree_util.tree_leaves(
        system,
        is_leaf=lambda item: (
            item is not system and isinstance(item, CoefficientEnclosureSource)
        ),
    )
    return tuple(
        bound
        for child in children
        if isinstance(child, CoefficientEnclosureSource)
        for bound in child.coefficient_enclosures()
    )


class AffineTrimChordEvidence(StrictModule):
    """Enclose affine trim endpoints in the stored normalized chart coordinates."""

    nominal_start: tuple[Fraction, Fraction] = eqx.field(static=True)
    nominal_end: tuple[Fraction, Fraction] = eqx.field(static=True)
    start_lower: np.ndarray
    start_upper: np.ndarray
    end_lower: np.ndarray
    end_upper: np.ndarray
    source_id: str = eqx.field(static=True)
    endpoint_root_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        nominal_start: tuple[Fraction, Fraction],
        nominal_end: tuple[Fraction, Fraction],
        start_lower: np.ndarray,
        start_upper: np.ndarray,
        end_lower: np.ndarray,
        end_upper: np.ndarray,
        source_id: str,
        endpoint_root_ids: tuple[str, ...],
    ) -> None:
        boxes = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (start_lower, start_upper, end_lower, end_upper)
        )
        for nominal, lower, upper in (
            (nominal_start, boxes[0], boxes[1]),
            (nominal_end, boxes[2], boxes[3]),
        ):
            if (
                lower.shape != (2,)
                or upper.shape != (2,)
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower > upper)
                or any(
                    Fraction(float(lower[axis])) > value
                    or Fraction(float(upper[axis])) < value
                    for axis, value in enumerate(nominal)
                )
            ):
                raise ValueError(
                    "AffineTrimChordEvidence.endpoints must have finite ordered two-coordinate bounds containing their rational nominal endpoints; "
                    f"got nominal={nominal!r}, lower={lower!r}, upper={upper!r}."
                )
        if not source_id:
            raise ValueError(
                f"AffineTrimChordEvidence.source_id must be nonempty; got {source_id!r}."
            )
        self.nominal_start, self.nominal_end = nominal_start, nominal_end
        self.start_lower, self.start_upper, self.end_lower, self.end_upper = boxes
        self.source_id = source_id
        self.endpoint_root_ids = endpoint_root_ids


@dataclass(frozen=True, slots=True)
class SurfacePiece(Generic[_CarrierT]):
    """Retain one source evaluator, closed parameter box, and deterministic index."""

    evaluator: _CarrierT
    lower: np.ndarray
    upper: np.ndarray
    index: int


@dataclass(frozen=True, slots=True)
class CurvePiece(Generic[_CarrierT]):
    """Retain one source evaluator, closed parameter interval, and deterministic index."""

    evaluator: _CarrierT
    lower: float
    upper: float
    index: int


__all__ = [
    "admit_carrier_fingerprint",
    "AffineTrimChordEvidence",
    "CarrierAdmission",
    "carrier_admission_current",
    "carrier_payload",
    "child_coefficient_enclosures",
    "CoefficientEnclosureSource",
    "CurveCoordinateResult",
    "CurvePiece",
    "RootCurveCoordinateSource",
    "SurfacePiece",
]
