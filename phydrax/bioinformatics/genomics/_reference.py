#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from operator import index as integer_index
from types import MappingProxyType

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import DNA_IUPAC, SequenceBatch


class ReferenceStatus(IntEnum):
    """Stable status codes for reference lookup and bounded lowering."""

    SUCCESS = 0
    UNKNOWN_REFERENCE = 1
    INVALID_INTERVAL = 2
    OUT_OF_BOUNDS = 3
    CAPACITY_EXCEEDED = 4
    DIGEST_MISMATCH = 5
    INVALID_SEQUENCE = 6


_REFERENCE_WINDOW_CONTRACT = BioinformaticsMethodContract(
    "bounded reference-window materialization",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.SEQUENCE,
    conditioning_statement="Exact byte lookup in a declared reference dictionary.",
    truncation_statement="Requests exceeding capacity fail without returning a prefix.",
    capacity_semantics="The static sequence axis is an upper bound; valid_mask gives the exact span.",
    assumptions=("Reference sequence and dictionary digest agree.",),
    nondifferentiable_outputs=("token_codes", "valid_mask", "status", "evidence"),
    input_dtype="host-int64",
    compute_dtype="int64",
    output_dtype="int32-sequence-codes",
)

_WINDOW_LOWERING_CONTRACT = BioinformaticsMethodContract(
    "global-to-window coordinate lowering",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement="Coordinates are zero-based positions in the declared contig.",
    truncation_statement="No coordinate is clipped; every out-of-window coordinate is invalid.",
    capacity_semantics="Output shape equals the input coordinate shape.",
    assumptions=("The reference window was materialized successfully.",),
    nondifferentiable_outputs=("relative_positions", "valid", "status", "evidence"),
    input_dtype="int64",
    compute_dtype="int64",
    output_dtype="int32",
)


def _normalized_sequence_bytes(
    sequence: str | bytes | bytearray | memoryview, /
) -> bytes:
    if isinstance(sequence, str):
        payload = sequence.encode("ascii")
    elif isinstance(sequence, (bytes, bytearray, memoryview)):
        payload = bytes(sequence)
    else:
        raise TypeError("Reference sequences must be ASCII strings or bytes-like values.")
    return b"".join(payload.split()).upper()


@dataclass(frozen=True, slots=True)
class ReferenceDigest:
    """Content digests used by SAM dictionaries, refget, and archive verification."""

    md5: str
    sha256: str
    sha512t24u: str

    def __post_init__(self) -> None:
        if len(self.md5) != 32 or any(
            value not in "0123456789abcdef" for value in self.md5
        ):
            raise ValueError("md5 must be a lower-case 32-character hexadecimal digest.")
        if len(self.sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.sha256
        ):
            raise ValueError(
                "sha256 must be a lower-case 64-character hexadecimal digest."
            )
        if len(self.sha512t24u) != 32 or any(
            value
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            for value in self.sha512t24u
        ):
            raise ValueError("sha512t24u must be a 32-character base64url digest.")

    @property
    def refget_id(self) -> str:
        return f"SQ.{self.sha512t24u}"

    def matches(self, sequence: str | bytes | bytearray | memoryview, /) -> bool:
        return self == reference_digest(sequence)


def reference_digest(
    sequence: str | bytes | bytearray | memoryview, /
) -> ReferenceDigest:
    """Digest the whitespace-free upper-case reference sequence canonically."""

    payload = _normalized_sequence_bytes(sequence)
    truncated = hashlib.sha512(payload).digest()[:24]
    return ReferenceDigest(
        md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        sha256=hashlib.sha256(payload).hexdigest(),
        sha512t24u=base64.urlsafe_b64encode(truncated).decode("ascii").rstrip("="),
    )


@dataclass(frozen=True, slots=True)
class ReferenceContig:
    """One ordered contig entry in a complete reference dictionary."""

    name: str
    length: int
    digest: ReferenceDigest
    aliases: tuple[str, ...] = ()
    circular: bool = False

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        length = integer_index(self.length)
        aliases = tuple(str(alias).strip() for alias in self.aliases)
        if not name:
            raise ValueError("Reference contig names must be non-empty.")
        if length < 0:
            raise ValueError("Reference contig lengths must be non-negative.")
        if not isinstance(self.digest, ReferenceDigest):
            raise TypeError("digest must be a ReferenceDigest.")
        if any(not alias for alias in aliases) or len(set(aliases)) != len(aliases):
            raise ValueError("Reference aliases must be non-empty and unique per contig.")
        if name in aliases:
            raise ValueError("A reference alias cannot repeat its canonical name.")
        if bool(self.circular) and length == 0:
            raise ValueError("A circular contig must have positive length.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "circular", bool(self.circular))


class ReferenceDictionary:
    """Host-only ordered reference dictionary with unambiguous aliases and identity."""

    __slots__ = ("_aliases", "_contigs", "_digest", "_name_to_index", "assembly_id")

    def __init__(
        self,
        contigs: Sequence[ReferenceContig],
        /,
        *,
        assembly_id: str = "",
    ):
        ordered = tuple(contigs)
        if not ordered:
            raise ValueError("A reference dictionary must contain at least one contig.")
        if any(not isinstance(contig, ReferenceContig) for contig in ordered):
            raise TypeError("contigs must contain only ReferenceContig records.")
        assembly = str(assembly_id).strip()
        name_to_index: dict[str, int] = {}
        aliases: dict[str, str] = {}
        for index, contig in enumerate(ordered):
            for name in (contig.name, *contig.aliases):
                if name in name_to_index:
                    raise ValueError(f"Reference name or alias {name!r} is not unique.")
                name_to_index[name] = index
                aliases[name] = contig.name
        payload = {
            "kind": "bioinformatics-reference-dictionary-v1",
            "assembly_id": assembly,
            "contigs": [
                {
                    "name": contig.name,
                    "length": contig.length,
                    "md5": contig.digest.md5,
                    "sha256": contig.digest.sha256,
                    "sha512t24u": contig.digest.sha512t24u,
                    "aliases": list(contig.aliases),
                    "circular": contig.circular,
                }
                for contig in ordered
            ],
        }
        self._contigs = ordered
        self._name_to_index = MappingProxyType(name_to_index)
        self._aliases = MappingProxyType(aliases)
        self._digest = canonical_fingerprint(payload)
        self.assembly_id = assembly

    @property
    def contigs(self) -> tuple[ReferenceContig, ...]:
        return self._contigs

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def name_to_index(self) -> Mapping[str, int]:
        return self._name_to_index

    @property
    def aliases(self) -> Mapping[str, str]:
        return self._aliases

    def __len__(self) -> int:
        return len(self._contigs)

    def resolve(self, reference: int | str, /) -> int:
        if isinstance(reference, str):
            if reference not in self._name_to_index:
                raise KeyError(f"Unknown reference name or alias {reference!r}.")
            return self._name_to_index[reference]
        index = integer_index(reference)
        if index < 0 or index >= len(self._contigs):
            raise IndexError(f"Reference index {index} is outside the dictionary.")
        return index

    def contig(self, reference: int | str, /) -> ReferenceContig:
        return self._contigs[self.resolve(reference)]


class ReferenceWindow(StrictModule):
    """One bounded encoded reference window and its global int64 placement."""

    sequence: SequenceBatch
    reference_index: Array
    requested_start: Array
    requested_end: Array
    canonical_start: Array
    reference_length: Array
    circular: Array
    wrapped: Array

    def __init__(
        self,
        sequence: SequenceBatch,
        reference_index: ArrayLike,
        requested_start: ArrayLike,
        requested_end: ArrayLike,
        canonical_start: ArrayLike,
        reference_length: ArrayLike,
        circular: ArrayLike,
        wrapped: ArrayLike,
    ):
        if not isinstance(sequence, SequenceBatch) or sequence.record_count != 1:
            raise TypeError("sequence must be a single-record SequenceBatch.")
        scalars = (
            jnp.asarray(reference_index, dtype=jnp.int32),
            jnp.asarray(requested_start, dtype=jnp.int64),
            jnp.asarray(requested_end, dtype=jnp.int64),
            jnp.asarray(canonical_start, dtype=jnp.int64),
            jnp.asarray(reference_length, dtype=jnp.int64),
            jnp.asarray(circular, dtype=bool),
            jnp.asarray(wrapped, dtype=bool),
        )
        if any(value.shape != () for value in scalars):
            raise ValueError("Reference-window metadata must be scalar.")
        self.sequence = sequence
        self.reference_index = scalars[0]
        self.requested_start = scalars[1]
        self.requested_end = scalars[2]
        self.canonical_start = scalars[3]
        self.reference_length = scalars[4]
        self.circular = scalars[5]
        self.wrapped = scalars[6]

    @property
    def capacity(self) -> int:
        return self.sequence.capacity

    @property
    def length(self) -> Array:
        return self.sequence.lengths[0]


class ReferenceWindowResult(StrictModule):
    """Audited bounded reference materialization result."""

    window: ReferenceWindow
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class WindowCoordinateResult(StrictModule):
    """Checked global int64 to window-relative int32 lowering."""

    relative_positions: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class ReferenceGenome:
    """Host-resident reference sequences tied exactly to a ReferenceDictionary."""

    __slots__ = ("_codes", "_sequences", "dictionary")

    def __init__(
        self,
        dictionary: ReferenceDictionary,
        sequences: Mapping[str, str | bytes | bytearray | memoryview]
        | Sequence[str | bytes | bytearray | memoryview],
        /,
    ):
        if not isinstance(dictionary, ReferenceDictionary):
            raise TypeError("dictionary must be a ReferenceDictionary.")
        if isinstance(sequences, Mapping):
            sequence_mapping: Mapping[
                str, str | bytes | bytearray | memoryview
            ] = sequences
            canonical_names = {contig.name for contig in dictionary.contigs}
            unexpected: list[str] = sorted(
                name for name in sequence_mapping if name not in canonical_names
            )
            if unexpected:
                raise ValueError(
                    f"Sequences contain unknown contigs: {unexpected!r}."
                )
            missing = [
                contig.name
                for contig in dictionary.contigs
                if contig.name not in sequence_mapping
            ]
            if missing:
                raise ValueError(
                    f"Sequences are missing dictionary contigs: {missing!r}."
                )
            values = tuple(
                sequence_mapping[contig.name] for contig in dictionary.contigs
            )
        else:
            values = tuple(sequences)
            if len(values) != len(dictionary):
                raise ValueError("Sequence count must match the reference dictionary.")

        alphabet_codes = DNA_IUPAC.symbol_to_code
        normalized: list[bytes] = []
        encoded: list[np.ndarray] = []
        for contig, value in zip(dictionary.contigs, values, strict=True):
            payload = _normalized_sequence_bytes(value)
            if len(payload) != contig.length:
                raise ValueError(
                    f"Sequence length for {contig.name!r} is {len(payload)}, expected {contig.length}."
                )
            if reference_digest(payload) != contig.digest:
                raise ValueError(
                    f"Sequence digest for {contig.name!r} does not match its dictionary."
                )
            symbols = payload.decode("ascii")
            invalid = sorted(set(symbols) - set(alphabet_codes))
            if invalid:
                raise ValueError(
                    f"Sequence for {contig.name!r} contains unsupported symbols {invalid!r}."
                )
            normalized.append(payload)
            encoded.append(
                np.fromiter(
                    (alphabet_codes[symbol] for symbol in symbols), dtype=np.int32
                )
            )
        self.dictionary = dictionary
        self._sequences = tuple(normalized)
        self._codes = tuple(encoded)

    @classmethod
    def from_sequences(
        cls,
        sequences: Mapping[str, str | bytes | bytearray | memoryview],
        /,
        *,
        assembly_id: str = "",
        aliases: Mapping[str, Sequence[str]] | None = None,
        circular: Sequence[str] = (),
    ) -> "ReferenceGenome":
        alias_map = {} if aliases is None else dict(aliases)
        circular_names = set(circular)
        unknown_aliases = set(alias_map) - set(sequences)
        unknown_circular = circular_names - set(sequences)
        if unknown_aliases or unknown_circular:
            raise ValueError(
                "Alias and circular declarations must name supplied contigs."
            )
        normalized = {
            str(name): _normalized_sequence_bytes(sequence)
            for name, sequence in sequences.items()
        }
        contigs = tuple(
            ReferenceContig(
                str(name),
                len(payload),
                reference_digest(payload),
                tuple(alias_map.get(str(name), ())),
                str(name) in circular_names,
            )
            for name, payload in normalized.items()
        )
        return cls(
            ReferenceDictionary(contigs, assembly_id=assembly_id),
            normalized,
        )

    def sequence(self, reference: int | str, /) -> bytes:
        return self._sequences[self.dictionary.resolve(reference)]

    def _empty_window(
        self,
        reference_index: int,
        start: int,
        end: int,
        capacity: int,
    ) -> ReferenceWindow:
        pad = DNA_IUPAC.code(DNA_IUPAC.pad_symbol)
        sequence = SequenceBatch(
            jnp.asarray([reference_index], dtype=jnp.int32),
            jnp.full((1, capacity), pad, dtype=jnp.int32),
            jnp.zeros((1, capacity), dtype=bool),
            jnp.asarray([True]),
            jnp.zeros((1, capacity), dtype=bool),
            DNA_IUPAC,
        )
        length = self.dictionary.contigs[reference_index].length
        return ReferenceWindow(
            sequence,
            reference_index,
            start,
            end,
            0,
            length,
            self.dictionary.contigs[reference_index].circular,
            False,
        )

    def fetch_window(
        self,
        reference: int | str,
        start: int,
        end: int,
        /,
        *,
        capacity: int,
    ) -> ReferenceWindowResult:
        """Materialize a checked half-open interval without silent truncation."""

        reference_index = self.dictionary.resolve(reference)
        contig = self.dictionary.contigs[reference_index]
        start_ = integer_index(start)
        end_ = integer_index(end)
        capacity_ = integer_index(capacity)
        if capacity_ < 0:
            raise ValueError("capacity must be non-negative.")
        span = end_ - start_
        status = ReferenceStatus.SUCCESS
        if span < 0:
            status = ReferenceStatus.INVALID_INTERVAL
        elif span > capacity_:
            status = ReferenceStatus.CAPACITY_EXCEEDED
        elif contig.circular:
            if span > contig.length:
                status = ReferenceStatus.OUT_OF_BOUNDS
        elif start_ < 0 or end_ > contig.length:
            status = ReferenceStatus.OUT_OF_BOUNDS

        if status is not ReferenceStatus.SUCCESS:
            window = self._empty_window(reference_index, start_, end_, capacity_)
            evidence = jnp.asarray(
                [span, capacity_, contig.length, 0],
                dtype=jnp.int64,
            )
            return ReferenceWindowResult(
                window,
                jnp.asarray(False),
                jnp.asarray(int(status), dtype=jnp.int32),
                evidence,
                _REFERENCE_WINDOW_CONTRACT,
            )

        canonical_start = start_ % contig.length if contig.circular else start_
        if span == 0:
            codes = np.empty((0,), dtype=np.int32)
        elif contig.circular:
            indices = np.mod(np.arange(start_, end_, dtype=np.int64), contig.length)
            codes = self._codes[reference_index][indices]
        else:
            codes = self._codes[reference_index][start_:end_]
        pad = DNA_IUPAC.code(DNA_IUPAC.pad_symbol)
        token_codes = np.full((1, capacity_), pad, dtype=np.int32)
        valid = np.zeros((1, capacity_), dtype=bool)
        token_codes[0, :span] = codes
        valid[0, :span] = True
        sequence = SequenceBatch(
            jnp.asarray([reference_index], dtype=jnp.int32),
            jnp.asarray(token_codes),
            jnp.asarray(valid),
            jnp.asarray([True]),
            jnp.zeros((1, capacity_), dtype=bool),
            DNA_IUPAC,
        )
        wrapped = contig.circular and span > 0 and canonical_start + span > contig.length
        window = ReferenceWindow(
            sequence,
            reference_index,
            start_,
            end_,
            canonical_start,
            contig.length,
            contig.circular,
            wrapped,
        )
        evidence = jnp.asarray(
            [span, capacity_, contig.length, int(wrapped)],
            dtype=jnp.int64,
        )
        return ReferenceWindowResult(
            window,
            jnp.asarray(True),
            jnp.asarray(int(ReferenceStatus.SUCCESS), dtype=jnp.int32),
            evidence,
            _REFERENCE_WINDOW_CONTRACT,
        )


def lower_global_coordinates(
    window: ReferenceWindow,
    positions: ArrayLike,
    /,
) -> WindowCoordinateResult:
    """Lower global int64 positions to checked window-relative int32 indices."""

    if not isinstance(window, ReferenceWindow):
        raise TypeError("window must be a ReferenceWindow.")
    raw_positions = jnp.asarray(positions)
    if not jnp.issubdtype(raw_positions.dtype, jnp.integer):
        raise TypeError("positions must have an integer dtype.")
    global_positions = raw_positions.astype(jnp.int64)
    length = window.length.astype(jnp.int64)
    in_reference = (global_positions >= 0) & (global_positions < window.reference_length)
    circular_relative = jnp.mod(
        global_positions - window.canonical_start,
        jnp.maximum(window.reference_length, jnp.asarray(1, dtype=jnp.int64)),
    )
    linear_relative = global_positions - window.requested_start
    relative64 = jnp.where(window.circular, circular_relative, linear_relative)
    in_window = in_reference & (relative64 >= 0) & (relative64 < length)
    fits_int32 = (relative64 >= jnp.iinfo(jnp.int32).min) & (
        relative64 <= jnp.iinfo(jnp.int32).max
    )
    valid = in_window & fits_int32
    relative = jnp.where(valid, relative64, 0).astype(jnp.int32)
    status = jnp.where(
        valid,
        int(ReferenceStatus.SUCCESS),
        int(ReferenceStatus.OUT_OF_BOUNDS),
    ).astype(jnp.int32)
    evidence = jnp.stack((in_reference, in_window, fits_int32), axis=-1)
    return WindowCoordinateResult(
        relative,
        valid,
        status,
        evidence,
        _WINDOW_LOWERING_CONTRACT,
    )


__all__ = [
    "ReferenceContig",
    "ReferenceDictionary",
    "ReferenceDigest",
    "ReferenceGenome",
    "ReferenceStatus",
    "ReferenceWindow",
    "ReferenceWindowResult",
    "WindowCoordinateResult",
    "lower_global_coordinates",
    "reference_digest",
]
