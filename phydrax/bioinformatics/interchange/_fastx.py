#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from phydrax._strict import StrictModule
from phydrax.bioinformatics.sequence import (
    lower_sequences,
    PhredEncoding,
    QualityBatch,
    SequenceBatch,
    SequenceLoweringPlan,
    SequenceLoweringReport,
)


FASTXSource = str | Path | Iterable[str]


def _header(identifier: str, description: str | None) -> str:
    record_id = str(identifier).strip()
    detail = None if description is None else str(description).strip()
    if not record_id or any(character.isspace() for character in record_id):
        raise ValueError("A FASTX record ID must be non-empty and contain no whitespace.")
    if "\n" in record_id or "\r" in record_id:
        raise ValueError("A FASTX record ID cannot contain a newline.")
    if detail is not None and ("\n" in detail or "\r" in detail):
        raise ValueError("A FASTX description cannot contain a newline.")
    return record_id


def _sequence(value: str) -> str:
    sequence = str(value)
    if any(character.isspace() for character in sequence):
        raise ValueError("A FASTX record sequence cannot contain whitespace.")
    return sequence


@dataclass(frozen=True, slots=True)
class FASTARecord:
    """One immutable host-only FASTA record; metadata never enters a PyTree."""

    record_id: str
    sequence: str
    description: str | None = None

    def __post_init__(self) -> None:
        record_id = _header(self.record_id, self.description)
        sequence = _sequence(self.sequence)
        description = None if self.description is None else str(self.description).strip()
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "description", description or None)


@dataclass(frozen=True, slots=True)
class FASTQRecord:
    """One immutable host-only FASTQ record with explicit Phred encoding."""

    record_id: str
    sequence: str
    quality: str
    phred_encoding: PhredEncoding
    description: str | None = None

    def __post_init__(self) -> None:
        record_id = _header(self.record_id, self.description)
        sequence = _sequence(self.sequence)
        quality = str(self.quality)
        encoding = PhredEncoding(self.phred_encoding)
        description = None if self.description is None else str(self.description).strip()
        if "\n" in quality or "\r" in quality:
            raise ValueError("A FASTQ quality string cannot contain a newline.")
        if len(sequence) != len(quality):
            raise ValueError("FASTQ sequence and quality lengths must match exactly.")
        scores = [ord(character) - encoding.offset for character in quality]
        if any(score < 0 or score > encoding.maximum_score for score in scores):
            raise ValueError(
                f"FASTQ quality characters are outside {encoding.value} range."
            )
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "phred_encoding", encoding)
        object.__setattr__(self, "description", description or None)

    @property
    def phred_scores(self) -> tuple[int, ...]:
        return tuple(
            ord(character) - self.phred_encoding.offset for character in self.quality
        )


def _lines(source: FASTXSource) -> list[str]:
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
        return text.splitlines()
    if isinstance(source, str):
        if "\n" in source or "\r" in source or source.startswith((">", "@")):
            return source.splitlines()
        path = Path(source)
        if path.is_file():
            return path.read_text(encoding="utf-8").splitlines()
        return source.splitlines()
    return [str(line).rstrip("\r\n") for line in source]


def _parse_header(line: str, marker: str) -> tuple[str, str | None]:
    if not line.startswith(marker):
        raise ValueError(f"Expected a {marker!r} FASTX header.")
    content = line[1:].strip()
    if not content:
        raise ValueError("FASTX headers must contain a record ID.")
    fields = content.split(maxsplit=1)
    return fields[0], fields[1] if len(fields) == 2 else None


def parse_fasta(source: FASTXSource) -> tuple[FASTARecord, ...]:
    """Parse FASTA text, a path, or an iterable of lines without dependencies."""
    lines = _lines(source)
    records: list[FASTARecord] = []
    record_id: str | None = None
    description: str | None = None
    sequence_parts: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        if line.startswith(">"):
            if record_id is not None:
                records.append(
                    FASTARecord(record_id, "".join(sequence_parts), description)
                )
            record_id, description = _parse_header(line, ">")
            sequence_parts = []
        elif record_id is None:
            if line.strip():
                raise ValueError(
                    f"FASTA sequence data precedes the first header at line {line_number}."
                )
        else:
            sequence_parts.append("".join(line.split()))
    if record_id is not None:
        records.append(FASTARecord(record_id, "".join(sequence_parts), description))
    return tuple(records)


def parse_fastq(
    source: FASTXSource,
    *,
    phred_encoding: PhredEncoding | str,
) -> tuple[FASTQRecord, ...]:
    """Parse possibly wrapped FASTQ with mandatory explicit quality encoding."""
    encoding = PhredEncoding(phred_encoding)
    lines = _lines(source)
    records: list[FASTQRecord] = []
    index = 0
    while index < len(lines):
        if not lines[index] and index == len(lines) - 1:
            break
        record_id, description = _parse_header(lines[index], "@")
        index += 1
        sequence_parts: list[str] = []
        while index < len(lines) and not lines[index].startswith("+"):
            sequence_parts.append("".join(lines[index].split()))
            index += 1
        if index == len(lines):
            raise ValueError(f"FASTQ record {record_id!r} has no '+' separator.")
        plus_content = lines[index][1:].strip()
        if plus_content and plus_content.split(maxsplit=1)[0] != record_id:
            raise ValueError("FASTQ '+' header ID does not match its '@' header.")
        index += 1
        sequence = "".join(sequence_parts)
        quality_parts: list[str] = []
        quality_length = 0
        if not sequence:
            if index == len(lines):
                raise ValueError(
                    f"Empty FASTQ record {record_id!r} requires an explicit empty quality line."
                )
            quality_parts.append(lines[index])
            quality_length = len(lines[index])
            index += 1
        else:
            while index < len(lines) and quality_length < len(sequence):
                quality_parts.append(lines[index])
                quality_length += len(lines[index])
                index += 1
        quality = "".join(quality_parts)
        if quality_length != len(sequence):
            raise ValueError(
                f"FASTQ record {record_id!r} sequence and quality lengths differ."
            )
        records.append(FASTQRecord(record_id, sequence, quality, encoding, description))
    return tuple(records)


def format_fasta(
    records: Sequence[FASTARecord] | Iterable[FASTARecord],
    *,
    line_width: int = 80,
) -> str:
    """Format host FASTA records as canonical newline-terminated text."""
    if isinstance(line_width, bool) or not isinstance(line_width, Integral):
        raise TypeError("line_width must be an integer.")
    width = int(line_width)
    if width <= 0:
        raise ValueError("line_width must be positive.")
    lines: list[str] = []
    for record in records:
        if not isinstance(record, FASTARecord):
            raise TypeError("Every FASTA output record must be a FASTARecord.")
        suffix = "" if record.description is None else f" {record.description}"
        lines.append(f">{record.record_id}{suffix}")
        if record.sequence:
            lines.extend(
                record.sequence[start : start + width]
                for start in range(0, len(record.sequence), width)
            )
        else:
            lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


def format_fastq(records: Sequence[FASTQRecord] | Iterable[FASTQRecord]) -> str:
    """Format host FASTQ records as canonical four-line text."""
    lines: list[str] = []
    encoding: PhredEncoding | None = None
    for record in records:
        if not isinstance(record, FASTQRecord):
            raise TypeError("Every FASTQ output record must be a FASTQRecord.")
        if encoding is None:
            encoding = record.phred_encoding
        elif record.phred_encoding is not encoding:
            raise ValueError("One FASTQ document cannot mix Phred encodings.")
        suffix = "" if record.description is None else f" {record.description}"
        lines.extend(
            (
                f"@{record.record_id}{suffix}",
                record.sequence,
                "+",
                record.quality,
            )
        )
    return "\n".join(lines) + ("\n" if lines else "")


def write_fasta(
    records: Sequence[FASTARecord] | Iterable[FASTARecord],
    destination: str | Path | None = None,
    *,
    line_width: int = 80,
) -> str | None:
    """Return FASTA text or write it to a path."""
    text = format_fasta(records, line_width=line_width)
    if destination is None:
        return text
    Path(destination).write_text(text, encoding="utf-8")
    return None


def write_fastq(
    records: Sequence[FASTQRecord] | Iterable[FASTQRecord],
    destination: str | Path | None = None,
) -> str | None:
    """Return FASTQ text or write it to a path."""
    text = format_fastq(records)
    if destination is None:
        return text
    Path(destination).write_text(text, encoding="utf-8")
    return None


class FASTQLoweringResult(StrictModule):
    """Aligned numeric sequence/quality batches and their shared loss report."""

    sequences: SequenceBatch
    qualities: QualityBatch
    report: SequenceLoweringReport

    def __init__(
        self,
        sequences: SequenceBatch,
        qualities: QualityBatch,
        report: SequenceLoweringReport,
    ):
        if not isinstance(sequences, SequenceBatch):
            raise TypeError("sequences must be a SequenceBatch.")
        if not isinstance(qualities, QualityBatch):
            raise TypeError("qualities must be a QualityBatch.")
        if not isinstance(report, SequenceLoweringReport):
            raise TypeError("report must be a SequenceLoweringReport.")
        if (
            sequences.token_codes.shape != qualities.phred_scores.shape
            or sequences.valid_mask.shape != qualities.valid_mask.shape
            or sequences.case_mask.shape != qualities.case_mask.shape
        ):
            raise ValueError("FASTQ sequence and quality batch shapes must match.")
        if not jnp.array_equal(sequences.valid_mask, qualities.valid_mask):
            raise ValueError("FASTQ sequence and quality validity masks must match.")
        self.sequences = sequences
        self.qualities = qualities
        self.report = report


def lower_fasta(
    records: Sequence[FASTARecord] | Iterable[FASTARecord],
    plan: SequenceLoweringPlan,
    *,
    numeric_record_ids: ArrayLike | None = None,
) -> tuple[SequenceBatch, SequenceLoweringReport]:
    """Lower FASTA records without inventing a quality payload."""
    items = tuple(records)
    if any(not isinstance(record, FASTARecord) for record in items):
        raise TypeError("Every input record must be a FASTARecord.")
    return lower_sequences(
        tuple(record.sequence for record in items),
        plan,
        record_ids=numeric_record_ids,
    )


def lower_fastq(
    records: Sequence[FASTQRecord] | Iterable[FASTQRecord],
    plan: SequenceLoweringPlan,
    *,
    numeric_record_ids: ArrayLike | None = None,
    phred_encoding: PhredEncoding | str | None = None,
) -> FASTQLoweringResult:
    """Lower FASTQ records to shape-aligned sequences and real quality scores."""
    items = tuple(records)
    if any(not isinstance(record, FASTQRecord) for record in items):
        raise TypeError("Every input record must be a FASTQRecord.")
    encodings = {record.phred_encoding for record in items}
    if len(encodings) > 1:
        raise ValueError("One QualityBatch cannot mix Phred encodings.")
    if not items and phred_encoding is None:
        raise ValueError("Empty FASTQ lowering requires an explicit phred_encoding.")
    requested_encoding = None if phred_encoding is None else PhredEncoding(phred_encoding)
    if items:
        encoding = next(iter(encodings))
        if requested_encoding is not None and requested_encoding is not encoding:
            raise ValueError("Requested Phred encoding does not match FASTQ records.")
    else:
        if requested_encoding is None:
            raise ValueError("Empty FASTQ lowering requires an explicit phred_encoding.")
        encoding = requested_encoding
    sequences, report = lower_sequences(
        tuple(record.sequence for record in items),
        plan,
        record_ids=numeric_record_ids,
    )
    scores = np.zeros(sequences.token_codes.shape, dtype=np.int32)
    retained_count = min(len(items), plan.record_capacity)
    for record_index in range(retained_count):
        retained_length = min(len(items[record_index].sequence), plan.sequence_capacity)
        scores[record_index, :retained_length] = np.asarray(
            items[record_index].phred_scores[:retained_length], dtype=np.int32
        )
    qualities = QualityBatch(
        sequences.record_ids,
        scores,
        sequences.valid_mask,
        sequences.case_mask,
        encoding,
    )
    return FASTQLoweringResult(sequences, qualities, report)


__all__ = [
    "FASTARecord",
    "FASTQRecord",
    "FASTQLoweringResult",
    "FASTXSource",
    "format_fasta",
    "format_fastq",
    "lower_fasta",
    "lower_fastq",
    "parse_fasta",
    "parse_fastq",
    "write_fasta",
    "write_fastq",
]
