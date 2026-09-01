#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import TextIO, TypeAlias
from urllib.parse import quote, unquote


class AnnotationLineKind(str, Enum):
    BLANK = "blank"
    COMMENT = "comment"
    DIRECTIVE = "directive"
    FASTA = "fasta"
    FEATURE = "feature"


def _split_newline(line: str, /) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1], line[-1:]
    return line, ""


def _lossless_lines(source: str | Iterable[str], /) -> tuple[tuple[str, str], ...]:
    text = source if isinstance(source, str) else "".join(source)
    return tuple(_split_newline(line) for line in text.splitlines(keepends=True))


@dataclass(frozen=True, slots=True)
class RawAnnotationLine:
    """An uninterpreted host-side line retained byte-for-codepoint losslessly."""

    text: str
    newline: str
    kind: AnnotationLineKind

    def __post_init__(self) -> None:
        if "\n" in self.text or "\r" in self.text:
            raise ValueError("RawAnnotationLine.text cannot contain a line terminator.")
        if self.newline not in ("", "\n", "\r", "\r\n"):
            raise ValueError("newline must be empty, LF, CR, or CRLF.")

    def render(self, *, preserve_original: bool = True) -> str:
        del preserve_original
        return self.text + self.newline


@dataclass(frozen=True, slots=True)
class GFF3Attribute:
    key: str
    values: tuple[str, ...]
    raw: str | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("GFF3 attribute keys must be non-empty.")


@dataclass(frozen=True, slots=True)
class GFF3FeatureLine:
    """Lossless GFF3 feature line with an internal half-open interval view."""

    seqid: str
    source: str
    feature_type: str
    start: int
    end: int
    score: str | None
    strand: str
    phase: int | None
    attributes: tuple[GFF3Attribute, ...]
    newline: str = "\n"
    raw_text: str | None = None

    def __post_init__(self) -> None:
        if not self.seqid or not self.source or not self.feature_type:
            raise ValueError("GFF3 seqid, source, and type must be non-empty.")
        if int(self.start) < 0 or int(self.end) < int(self.start):
            raise ValueError("GFF3 internal coordinates require 0 <= start <= end.")
        if self.strand not in ("+", "-", ".", "?"):
            raise ValueError("GFF3 strand must be '+', '-', '.', or '?'.")
        if self.phase not in (None, 0, 1, 2):
            raise ValueError("GFF3 phase must be absent, 0, 1, or 2.")
        if self.newline not in ("", "\n", "\r", "\r\n"):
            raise ValueError("newline must be empty, LF, CR, or CRLF.")

    @property
    def kind(self) -> AnnotationLineKind:
        return AnnotationLineKind.FEATURE

    @property
    def length(self) -> int:
        return self.end - self.start

    def attribute_values(self, key: str, /) -> tuple[str, ...]:
        return tuple(
            value
            for attribute in self.attributes
            if attribute.key == key
            for value in attribute.values
        )

    def render(self, *, preserve_original: bool = True) -> str:
        if preserve_original and self.raw_text is not None:
            return self.raw_text + self.newline
        if self.end <= self.start:
            raise ValueError(
                "A zero-length half-open interval has no lossless GFF3 representation."
            )
        attribute_field = _render_gff3_attributes(self.attributes)
        fields = (
            self.seqid,
            self.source,
            self.feature_type,
            str(self.start + 1),
            str(self.end),
            "." if self.score is None else self.score,
            self.strand,
            "." if self.phase is None else str(self.phase),
            attribute_field,
        )
        return "\t".join(fields) + self.newline


@dataclass(frozen=True, slots=True)
class GTFAttribute:
    key: str
    value: str
    raw: str | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("GTF attribute keys must be non-empty.")


@dataclass(frozen=True, slots=True)
class GTFFeatureLine:
    """Lossless GTF feature line with decoded ordered attributes."""

    seqname: str
    source: str
    feature_type: str
    start: int
    end: int
    score: str | None
    strand: str
    frame: int | None
    attributes: tuple[GTFAttribute, ...]
    newline: str = "\n"
    raw_text: str | None = None

    def __post_init__(self) -> None:
        if not self.seqname or not self.source or not self.feature_type:
            raise ValueError("GTF seqname, source, and feature must be non-empty.")
        if int(self.start) < 0 or int(self.end) < int(self.start):
            raise ValueError("GTF internal coordinates require 0 <= start <= end.")
        if self.strand not in ("+", "-", "."):
            raise ValueError("GTF strand must be '+', '-', or '.'.")
        if self.frame not in (None, 0, 1, 2):
            raise ValueError("GTF frame must be absent, 0, 1, or 2.")
        if self.newline not in ("", "\n", "\r", "\r\n"):
            raise ValueError("newline must be empty, LF, CR, or CRLF.")

    @property
    def kind(self) -> AnnotationLineKind:
        return AnnotationLineKind.FEATURE

    @property
    def length(self) -> int:
        return self.end - self.start

    def attribute_values(self, key: str, /) -> tuple[str, ...]:
        return tuple(
            attribute.value for attribute in self.attributes if attribute.key == key
        )

    def render(self, *, preserve_original: bool = True) -> str:
        if preserve_original and self.raw_text is not None:
            return self.raw_text + self.newline
        if self.end <= self.start:
            raise ValueError(
                "A zero-length half-open interval has no lossless GTF representation."
            )
        fields = (
            self.seqname,
            self.source,
            self.feature_type,
            str(self.start + 1),
            str(self.end),
            "." if self.score is None else self.score,
            self.strand,
            "." if self.frame is None else str(self.frame),
            _render_gtf_attributes(self.attributes),
        )
        return "\t".join(fields) + self.newline


@dataclass(frozen=True, slots=True)
class BEDFeatureLine:
    """Lossless BED line; BED's native coordinates are already half-open."""

    chrom: str
    start: int
    end: int
    extra_fields: tuple[str, ...] = ()
    newline: str = "\n"
    raw_text: str | None = None

    def __post_init__(self) -> None:
        if not self.chrom:
            raise ValueError("BED chrom must be non-empty.")
        if int(self.start) < 0 or int(self.end) < int(self.start):
            raise ValueError("BED coordinates require 0 <= start <= end.")
        if len(self.extra_fields) > 9:
            raise ValueError("BED supports at most twelve columns.")
        if self.newline not in ("", "\n", "\r", "\r\n"):
            raise ValueError("newline must be empty, LF, CR, or CRLF.")

    @property
    def kind(self) -> AnnotationLineKind:
        return AnnotationLineKind.FEATURE

    @property
    def column_count(self) -> int:
        return 3 + len(self.extra_fields)

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def name(self) -> str | None:
        return self.extra_fields[0] if len(self.extra_fields) >= 1 else None

    @property
    def score(self) -> str | None:
        return self.extra_fields[1] if len(self.extra_fields) >= 2 else None

    @property
    def strand(self) -> str | None:
        return self.extra_fields[2] if len(self.extra_fields) >= 3 else None

    @property
    def block_intervals(self) -> tuple[tuple[int, int], ...]:
        if self.column_count < 12:
            return ()
        block_count = int(self.extra_fields[6])
        sizes = _bed_integer_list(self.extra_fields[7], "blockSizes")
        offsets = _bed_integer_list(self.extra_fields[8], "blockStarts")
        if block_count < 0 or len(sizes) != block_count or len(offsets) != block_count:
            raise ValueError("BED blockCount must match blockSizes and blockStarts.")
        intervals = tuple(
            (self.start + offset, self.start + offset + size)
            for size, offset in zip(sizes, offsets, strict=True)
        )
        if any(
            size < 0 or offset < 0 for size, offset in zip(sizes, offsets, strict=True)
        ):
            raise ValueError("BED block sizes and starts must be non-negative.")
        if any(start < self.start or end > self.end for start, end in intervals):
            raise ValueError("BED blocks must lie inside the parent interval.")
        return intervals

    def render(self, *, preserve_original: bool = True) -> str:
        if preserve_original and self.raw_text is not None:
            return self.raw_text + self.newline
        return (
            "\t".join((self.chrom, str(self.start), str(self.end), *self.extra_fields))
            + self.newline
        )


GFF3Line: TypeAlias = RawAnnotationLine | GFF3FeatureLine
GTFLine: TypeAlias = RawAnnotationLine | GTFFeatureLine
BEDLine: TypeAlias = RawAnnotationLine | BEDFeatureLine


@dataclass(frozen=True, slots=True)
class GFF3ParentRelations:
    """Host-side ID/Parent resolution retaining all ambiguity and unresolved loss."""

    feature_identifiers: tuple[tuple[str, ...], ...]
    child_rows: tuple[int, ...]
    parent_rows: tuple[int, ...]
    parent_identifiers: tuple[str, ...]
    unresolved: tuple[tuple[int, str], ...]
    duplicate_identifiers: tuple[str, ...]
    duplicate_edges: tuple[tuple[int, int], ...]

    @property
    def ambiguous(self) -> bool:
        return bool(self.duplicate_identifiers or self.duplicate_edges)

    @property
    def lossless(self) -> bool:
        return not self.unresolved


_GFF_SAFE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.:^*$@!+_?-|"


def _parse_gff3_attributes(field: str, /) -> tuple[GFF3Attribute, ...]:
    if field == "." or field == "":
        return ()
    attributes: list[GFF3Attribute] = []
    for raw in field.split(";"):
        if "=" in raw:
            key_raw, values_raw = raw.split("=", 1)
            key = unquote(key_raw)
            values = tuple(unquote(value) for value in values_raw.split(","))
        else:
            key = unquote(raw)
            values = ()
        attributes.append(GFF3Attribute(key, values, raw))
    return tuple(attributes)


def _render_gff3_attributes(attributes: Sequence[GFF3Attribute], /) -> str:
    if not attributes:
        return "."
    fields = []
    for attribute in attributes:
        key = quote(attribute.key, safe=_GFF_SAFE)
        values = ",".join(quote(value, safe=_GFF_SAFE) for value in attribute.values)
        fields.append(key if not attribute.values else f"{key}={values}")
    return ";".join(fields)


def _parse_gtf_attributes(field: str, /) -> tuple[GTFAttribute, ...]:
    chunks: list[str] = []
    start = 0
    quoted = False
    escaped = False
    for index, character in enumerate(field):
        if escaped:
            escaped = False
        elif character == "\\" and quoted:
            escaped = True
        elif character == '"':
            quoted = not quoted
        elif character == ";" and not quoted:
            chunks.append(field[start:index])
            start = index + 1
    if quoted or escaped:
        raise ValueError("GTF attribute field contains an unterminated quoted value.")
    if field[start:].strip():
        chunks.append(field[start:])
    attributes: list[GTFAttribute] = []
    for raw in chunks:
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed GTF attribute {raw!r}.")
        key, encoded = parts
        encoded = encoded.strip()
        if len(encoded) < 2 or encoded[0] != '"' or encoded[-1] != '"':
            raise ValueError(f"GTF attribute {key!r} must have a quoted value.")
        payload = encoded[1:-1]
        value_chars: list[str] = []
        escape = False
        for character in payload:
            if escape:
                value_chars.append(character)
                escape = False
            elif character == "\\":
                escape = True
            else:
                value_chars.append(character)
        attributes.append(GTFAttribute(key, "".join(value_chars), raw))
    return tuple(attributes)


def _render_gtf_attributes(attributes: Sequence[GTFAttribute], /) -> str:
    return " ".join(
        f'{attribute.key} "{attribute.value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}";'
        for attribute in attributes
    )


def _bed_integer_list(field: str, name: str, /) -> tuple[int, ...]:
    values = field[:-1].split(",") if field.endswith(",") else field.split(",")
    if values == [""]:
        return ()
    try_values = tuple(int(value) for value in values)
    if any(value < 0 for value in try_values):
        raise ValueError(f"BED {name} values must be non-negative.")
    return try_values


def parse_gff3(source: str | Iterable[str], /) -> tuple[GFF3Line, ...]:
    """Parse GFF3 host text while preserving every directive, attribute, and newline."""

    records: list[GFF3Line] = []
    in_fasta = False
    for text, newline in _lossless_lines(source):
        if in_fasta:
            records.append(RawAnnotationLine(text, newline, AnnotationLineKind.FASTA))
            continue
        if text == "":
            records.append(RawAnnotationLine(text, newline, AnnotationLineKind.BLANK))
            continue
        if text.startswith("##"):
            records.append(RawAnnotationLine(text, newline, AnnotationLineKind.DIRECTIVE))
            in_fasta = text == "##FASTA"
            continue
        if text.startswith("#"):
            records.append(RawAnnotationLine(text, newline, AnnotationLineKind.COMMENT))
            continue
        fields = text.split("\t")
        if len(fields) != 9:
            raise ValueError(
                "A GFF3 feature line must contain exactly nine tab-separated fields."
            )
        start_one = int(fields[3])
        end_one = int(fields[4])
        if start_one < 1 or end_one < start_one:
            raise ValueError(
                "GFF3 coordinates must be positive one-based closed intervals."
            )
        phase = None if fields[7] == "." else int(fields[7])
        score = None if fields[5] == "." else fields[5]
        records.append(
            GFF3FeatureLine(
                fields[0],
                fields[1],
                fields[2],
                start_one - 1,
                end_one,
                score,
                fields[6],
                phase,
                _parse_gff3_attributes(fields[8]),
                newline,
                text,
            )
        )
    return tuple(records)


def parse_gtf(source: str | Iterable[str], /) -> tuple[GTFLine, ...]:
    """Parse GTF host text while preserving attribute spacing, order, and newlines."""

    records: list[GTFLine] = []
    for text, newline in _lossless_lines(source):
        if text == "":
            records.append(RawAnnotationLine(text, newline, AnnotationLineKind.BLANK))
            continue
        if text.startswith("#"):
            kind = (
                AnnotationLineKind.DIRECTIVE
                if text.startswith("##")
                else AnnotationLineKind.COMMENT
            )
            records.append(RawAnnotationLine(text, newline, kind))
            continue
        fields = text.split("\t")
        if len(fields) != 9:
            raise ValueError(
                "A GTF feature line must contain exactly nine tab-separated fields."
            )
        start_one = int(fields[3])
        end_one = int(fields[4])
        if start_one < 1 or end_one < start_one:
            raise ValueError(
                "GTF coordinates must be positive one-based closed intervals."
            )
        frame = None if fields[7] == "." else int(fields[7])
        score = None if fields[5] == "." else fields[5]
        records.append(
            GTFFeatureLine(
                fields[0],
                fields[1],
                fields[2],
                start_one - 1,
                end_one,
                score,
                fields[6],
                frame,
                _parse_gtf_attributes(fields[8]),
                newline,
                text,
            )
        )
    return tuple(records)


def parse_bed(source: str | Iterable[str], /) -> tuple[BEDLine, ...]:
    """Parse BED3–BED12 host text without normalizing optional columns."""

    records: list[BEDLine] = []
    for text, newline in _lossless_lines(source):
        if text == "":
            records.append(RawAnnotationLine(text, newline, AnnotationLineKind.BLANK))
            continue
        if text.startswith("track") or text.startswith("browser"):
            records.append(RawAnnotationLine(text, newline, AnnotationLineKind.DIRECTIVE))
            continue
        if text.startswith("#"):
            records.append(RawAnnotationLine(text, newline, AnnotationLineKind.COMMENT))
            continue
        fields = text.split("\t")
        if len(fields) < 3 or len(fields) > 12:
            raise ValueError(
                "A BED feature line must contain between three and twelve columns."
            )
        start = int(fields[1])
        end = int(fields[2])
        record = BEDFeatureLine(fields[0], start, end, tuple(fields[3:]), newline, text)
        if record.strand is not None and record.strand not in ("+", "-", "."):
            raise ValueError("BED strand must be '+', '-', or '.'.")
        if record.column_count >= 12:
            record.block_intervals
        records.append(record)
    return tuple(records)


def gff3_parent_relations(records: Sequence[GFF3Line], /) -> GFF3ParentRelations:
    """Resolve all GFF3 ID/Parent routes without choosing among duplicate IDs."""

    features = tuple(record for record in records if isinstance(record, GFF3FeatureLine))
    identifiers = tuple(record.attribute_values("ID") for record in features)
    rows_by_identifier: dict[str, list[int]] = {}
    for row, values in enumerate(identifiers):
        for identifier in values:
            rows_by_identifier.setdefault(identifier, []).append(row)
    duplicates = tuple(
        sorted(
            identifier for identifier, rows in rows_by_identifier.items() if len(rows) > 1
        )
    )
    children: list[int] = []
    parents: list[int] = []
    parent_ids: list[str] = []
    unresolved: list[tuple[int, str]] = []
    seen_edges: set[tuple[int, int]] = set()
    duplicate_edges: list[tuple[int, int]] = []
    for child_row, record in enumerate(features):
        for parent_identifier in record.attribute_values("Parent"):
            parent_rows = rows_by_identifier.get(parent_identifier, ())
            if not parent_rows:
                unresolved.append((child_row, parent_identifier))
                continue
            for parent_row in parent_rows:
                edge = (child_row, parent_row)
                if edge in seen_edges:
                    duplicate_edges.append(edge)
                seen_edges.add(edge)
                children.append(child_row)
                parents.append(parent_row)
                parent_ids.append(parent_identifier)
    return GFF3ParentRelations(
        identifiers,
        tuple(children),
        tuple(parents),
        tuple(parent_ids),
        tuple(unresolved),
        duplicates,
        tuple(duplicate_edges),
    )


def _write_records(
    records: Sequence[
        RawAnnotationLine | GFF3FeatureLine | GTFFeatureLine | BEDFeatureLine
    ],
    destination: TextIO | None,
    preserve_original: bool,
    /,
) -> str:
    text = "".join(
        record.render(preserve_original=preserve_original) for record in records
    )
    if destination is not None:
        destination.write(text)
    return text


def write_gff3(
    records: Sequence[GFF3Line],
    destination: TextIO | None = None,
    /,
    *,
    preserve_original: bool = True,
) -> str:
    return _write_records(records, destination, preserve_original)


def write_gtf(
    records: Sequence[GTFLine],
    destination: TextIO | None = None,
    /,
    *,
    preserve_original: bool = True,
) -> str:
    return _write_records(records, destination, preserve_original)


def write_bed(
    records: Sequence[BEDLine],
    destination: TextIO | None = None,
    /,
    *,
    preserve_original: bool = True,
) -> str:
    return _write_records(records, destination, preserve_original)


def _read_text_losslessly(path: str | PathLike[str], /) -> str:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def read_gff3(path: str | PathLike[str], /) -> tuple[GFF3Line, ...]:
    return parse_gff3(_read_text_losslessly(path))


def read_gtf(path: str | PathLike[str], /) -> tuple[GTFLine, ...]:
    return parse_gtf(_read_text_losslessly(path))


def read_bed(path: str | PathLike[str], /) -> tuple[BEDLine, ...]:
    return parse_bed(_read_text_losslessly(path))


__all__ = [
    "AnnotationLineKind",
    "BEDFeatureLine",
    "BEDLine",
    "GFF3Attribute",
    "GFF3FeatureLine",
    "GFF3Line",
    "GFF3ParentRelations",
    "GTFAttribute",
    "GTFFeatureLine",
    "GTFLine",
    "RawAnnotationLine",
    "gff3_parent_relations",
    "parse_bed",
    "parse_gff3",
    "parse_gtf",
    "read_bed",
    "read_gff3",
    "read_gtf",
    "write_bed",
    "write_gff3",
    "write_gtf",
]
