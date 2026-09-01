#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule


class AlphabetPlan(StrictModule):
    """Immutable compile-time semantics for one encoded biological alphabet."""

    alphabet_id: str = eqx.field(static=True)
    symbols: tuple[str, ...] = eqx.field(static=True)
    canonical_symbols: tuple[str, ...] = eqx.field(static=True)
    ambiguities: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)
    complements: tuple[tuple[str, str], ...] = eqx.field(static=True)
    gap_symbol: str = eqx.field(static=True)
    pad_symbol: str = eqx.field(static=True)
    unknown_symbol: str = eqx.field(static=True)
    missing_symbol: str = eqx.field(static=True)
    mask_symbol: str = eqx.field(static=True)
    stop_symbol: str | None = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        alphabet_id: str,
        symbols: tuple[str, ...],
        canonical_symbols: tuple[str, ...],
        *,
        ambiguities: tuple[tuple[str, tuple[str, ...]], ...] = (),
        complements: tuple[tuple[str, str], ...] = (),
        gap_symbol: str = "-",
        pad_symbol: str = "_",
        unknown_symbol: str = "?",
        missing_symbol: str = ".",
        mask_symbol: str = "#",
        stop_symbol: str | None = None,
    ):
        identifier = str(alphabet_id).strip()
        ordered = tuple(str(symbol) for symbol in symbols)
        canonical = tuple(str(symbol) for symbol in canonical_symbols)
        ambiguity_items = tuple(
            (str(symbol), tuple(str(value) for value in values))
            for symbol, values in ambiguities
        )
        complement_items = tuple(
            (str(symbol), str(complement)) for symbol, complement in complements
        )
        special = (
            str(gap_symbol),
            str(pad_symbol),
            str(unknown_symbol),
            str(missing_symbol),
            str(mask_symbol),
        )
        stop = None if stop_symbol is None else str(stop_symbol)

        if not identifier:
            raise ValueError("alphabet_id must be non-empty.")
        if not ordered or any(len(symbol) != 1 for symbol in ordered):
            raise ValueError("Alphabet symbols must be single non-empty characters.")
        if len(set(ordered)) != len(ordered):
            raise ValueError("Alphabet symbols must be unique.")
        if not canonical or len(set(canonical)) != len(canonical):
            raise ValueError("canonical_symbols must be non-empty and unique.")
        if any(symbol not in ordered for symbol in canonical):
            raise ValueError("Every canonical symbol must belong to the alphabet.")
        if len(set(special)) != len(special):
            raise ValueError("Gap, pad, unknown, missing, and mask symbols must differ.")
        if any(symbol not in ordered for symbol in special):
            raise ValueError("Every special symbol must belong to the alphabet.")
        if stop is not None and stop not in ordered:
            raise ValueError("stop_symbol must belong to the alphabet.")

        ambiguity_keys = tuple(symbol for symbol, _ in ambiguity_items)
        if len(set(ambiguity_keys)) != len(ambiguity_keys):
            raise ValueError("Ambiguity symbols must be unique.")
        for symbol, values in ambiguity_items:
            if symbol not in ordered or symbol in canonical or not values:
                raise ValueError(
                    "Each ambiguity must be a non-canonical alphabet symbol."
                )
            if len(set(values)) != len(values) or any(
                value not in canonical for value in values
            ):
                raise ValueError("Ambiguity expansions must be unique canonical symbols.")

        complement_map = dict(complement_items)
        if len(complement_map) != len(complement_items):
            raise ValueError("Complement sources must be unique.")
        if complement_items:
            if set(complement_map) != set(ordered):
                raise ValueError("A complement table must cover the entire alphabet.")
            if any(value not in ordered for value in complement_map.values()):
                raise ValueError("Complement values must belong to the alphabet.")
            if any(
                complement_map[complement_map[symbol]] != symbol for symbol in ordered
            ):
                raise ValueError("The complement table must be an involution.")

        payload = {
            "alphabet_id": identifier,
            "symbols": ordered,
            "canonical_symbols": canonical,
            "ambiguities": ambiguity_items,
            "complements": complement_items,
            "gap_symbol": special[0],
            "pad_symbol": special[1],
            "unknown_symbol": special[2],
            "missing_symbol": special[3],
            "mask_symbol": special[4],
            "stop_symbol": stop,
        }
        self.alphabet_id = identifier
        self.symbols = ordered
        self.canonical_symbols = canonical
        self.ambiguities = ambiguity_items
        self.complements = complement_items
        self.gap_symbol = special[0]
        self.pad_symbol = special[1]
        self.unknown_symbol = special[2]
        self.missing_symbol = special[3]
        self.mask_symbol = special[4]
        self.stop_symbol = stop
        self.fingerprint = canonical_fingerprint(payload)

    @property
    def size(self) -> int:
        return len(self.symbols)

    @property
    def symbol_to_code(self) -> dict[str, int]:
        """Return a fresh host lookup from symbol to numeric code."""
        return {symbol: code for code, symbol in enumerate(self.symbols)}

    @property
    def ambiguity_map(self) -> dict[str, tuple[str, ...]]:
        """Return a fresh host lookup for IUPAC ambiguity expansions."""
        return dict(self.ambiguities)

    @property
    def complement_map(self) -> dict[str, str]:
        """Return a fresh host lookup for complements."""
        return dict(self.complements)

    def code(self, symbol: str, /) -> int:
        normalized = str(symbol).upper() if str(symbol).isalpha() else str(symbol)
        mapping = self.symbol_to_code
        if normalized not in mapping:
            raise ValueError(
                f"Symbol {symbol!r} does not belong to alphabet {self.alphabet_id!r}."
            )
        return mapping[normalized]


_SPECIALS = ("-", "_", "?", ".", "#")
_DNA_CANONICAL = ("A", "C", "G", "T")
_RNA_CANONICAL = ("A", "C", "G", "U")
_DNA_AMBIGUITIES = (
    ("R", ("A", "G")),
    ("Y", ("C", "T")),
    ("S", ("C", "G")),
    ("W", ("A", "T")),
    ("K", ("G", "T")),
    ("M", ("A", "C")),
    ("B", ("C", "G", "T")),
    ("D", ("A", "G", "T")),
    ("H", ("A", "C", "T")),
    ("V", ("A", "C", "G")),
    ("N", ("A", "C", "G", "T")),
)
_RNA_AMBIGUITIES = tuple(
    (symbol, tuple("U" if value == "T" else value for value in values))
    for symbol, values in _DNA_AMBIGUITIES
)
_DNA_COMPLEMENTS = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "R": "Y",
    "Y": "R",
    "S": "S",
    "W": "W",
    "K": "M",
    "M": "K",
    "B": "V",
    "D": "H",
    "H": "D",
    "V": "B",
    "N": "N",
    "-": "-",
    "_": "_",
    "?": "?",
    ".": ".",
    "#": "#",
}
_RNA_COMPLEMENTS = {
    **{key: value for key, value in _DNA_COMPLEMENTS.items() if key not in ("A", "T")},
    "A": "U",
    "U": "A",
}
_PROTEIN_CANONICAL = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "U",
    "O",
)
_PROTEIN_AMBIGUITIES = (
    ("B", ("D", "N")),
    ("Z", ("E", "Q")),
    ("J", ("I", "L")),
    ("X", _PROTEIN_CANONICAL),
)

DNA_IUPAC = AlphabetPlan(
    "dna-iupac",
    _DNA_CANONICAL + tuple(symbol for symbol, _ in _DNA_AMBIGUITIES) + _SPECIALS,
    _DNA_CANONICAL,
    ambiguities=_DNA_AMBIGUITIES,
    complements=tuple(_DNA_COMPLEMENTS.items()),
)
RNA_IUPAC = AlphabetPlan(
    "rna-iupac",
    _RNA_CANONICAL + tuple(symbol for symbol, _ in _RNA_AMBIGUITIES) + _SPECIALS,
    _RNA_CANONICAL,
    ambiguities=_RNA_AMBIGUITIES,
    complements=tuple(_RNA_COMPLEMENTS.items()),
)
PROTEIN_IUPAC = AlphabetPlan(
    "protein-iupac",
    _PROTEIN_CANONICAL
    + tuple(symbol for symbol, _ in _PROTEIN_AMBIGUITIES)
    + ("*",)
    + _SPECIALS,
    _PROTEIN_CANONICAL,
    ambiguities=_PROTEIN_AMBIGUITIES,
    stop_symbol="*",
)


__all__ = ["AlphabetPlan", "DNA_IUPAC", "PROTEIN_IUPAC", "RNA_IUPAC"]
