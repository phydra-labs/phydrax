#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded semantic SLHA parsing, preservation, serialization, and spectrum extraction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ...particle_physics._spectrum import SpectrumObservableTable


def _number(token: str, /) -> float:
    text = str(token).replace("D", "E").replace("d", "e")
    value = float(text)
    if not np.isfinite(value):
        raise ValueError("SLHA numeric values must be finite.")
    return value


@dataclass(frozen=True, slots=True)
class SLHAEntry:
    indices: tuple[int, ...]
    value: float
    raw_value: str
    comment: str = ""


@dataclass(frozen=True, slots=True)
class SLHABlock:
    name: str
    scale: float | None
    entries: tuple[SLHAEntry, ...]
    comment: str = ""
    header_arguments: tuple[str, ...] = ()

    def entry(self, *indices: int) -> SLHAEntry:
        key = tuple(indices)
        matches = tuple(value for value in self.entries if value.indices == key)
        if len(matches) != 1:
            raise ValueError("SLHA block entry is absent or duplicated.")
        return matches[0]


@dataclass(frozen=True, slots=True)
class SLHADecayChannel:
    branching_ratio: float
    daughters: tuple[int, ...]
    raw_branching_ratio: str
    comment: str = ""


@dataclass(frozen=True, slots=True)
class SLHADecay:
    pdg_id: int
    width: float
    raw_width: str
    channels: tuple[SLHADecayChannel, ...]
    comment: str = ""


@dataclass(frozen=True, slots=True)
class SLHADiagnostics:
    line_count: int
    block_count: int
    entry_count: int
    decay_count: int
    channel_count: int
    duplicate_entry_count: int
    unknown_block_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SLHADocument:
    preamble: tuple[str, ...]
    blocks: tuple[SLHABlock, ...]
    decays: tuple[SLHADecay, ...]
    diagnostics: SLHADiagnostics
    source_id: str
    profile_id: str

    def block(self, name: str, /) -> SLHABlock:
        target = str(name).upper()
        matches = tuple(value for value in self.blocks if value.name == target)
        if len(matches) != 1:
            raise ValueError("SLHA block is absent or repeated at multiple scales.")
        return matches[0]


@dataclass(slots=True)
class _MutableBlock:
    name: str
    scale: float | None
    entries: list[SLHAEntry]
    comment: str
    header_arguments: tuple[str, ...]


@dataclass(slots=True)
class _MutableDecay:
    pdg_id: int
    width: float
    raw_width: str
    channels: list[SLHADecayChannel]
    comment: str


def parse_slha(
    data: str | bytes,
    /,
    *,
    maximum_bytes: int = 64 * 1024 * 1024,
    maximum_lines: int = 1_000_000,
    maximum_entries: int = 10_000_000,
    reject_duplicates: bool = True,
    known_block_names: tuple[str, ...] = (
        "MASS",
        "SMINPUTS",
        "MINPAR",
        "EXTPAR",
        "MODSEL",
        "SPINFO",
        "DCINFO",
        "NMIX",
        "UMIX",
        "VMIX",
        "STOPMIX",
        "SBOTMIX",
        "STAUMIX",
        "ALPHA",
        "HMIX",
        "GAUGE",
        "YU",
        "YD",
        "YE",
        "AU",
        "AD",
        "AE",
        "MSOFT",
        "MSQ2",
        "MSU2",
        "MSD2",
        "MSL2",
        "MSE2",
        "VCKM",
        "UPMNS",
        "QNUMBERS",
        "IMNMIX",
        "IMUMIX",
        "IMVMIX",
        "IMSTOPMIX",
        "IMSBOTMIX",
        "IMSTAUMIX",
    ),
) -> SLHADocument:
    """Parse a bounded SLHA text without discarding unknown blocks or comments."""
    raw = data if isinstance(data, bytes) else str(data).encode("utf-8")
    if len(raw) > int(maximum_bytes) or maximum_bytes < 1:
        raise ValueError("SLHA input exceeds maximum_bytes.")
    text = raw.decode("utf-8", errors="strict")
    lines = text.splitlines()
    if len(lines) > int(maximum_lines) or maximum_lines < 1 or maximum_entries < 1:
        raise ValueError("SLHA input exceeds line/resource limits.")
    preamble: list[str] = []
    blocks: list[_MutableBlock] = []
    decays: list[_MutableDecay] = []
    active_block: _MutableBlock | None = None
    active_decay: _MutableDecay | None = None
    entry_count = 0
    duplicate_count = 0
    seen_entries: set[tuple[int, tuple[int, ...]]] = set()

    for line_number, original in enumerate(lines, start=1):
        body, separator, comment_text = original.partition("#")
        comment = comment_text.strip() if separator else ""
        stripped = body.strip()
        if not stripped:
            if active_block is None and active_decay is None:
                preamble.append(original)
            continue
        tokens = stripped.split()
        keyword = tokens[0].upper()
        if keyword == "BLOCK":
            if len(tokens) < 2:
                raise ValueError(f"SLHA BLOCK header is malformed on line {line_number}.")
            name = tokens[1].upper()
            scale = None
            header_arguments: list[str] = []
            index = 2
            while index < len(tokens):
                token = tokens[index]
                upper = token.upper()
                if upper == "Q=" and index + 1 < len(tokens):
                    scale = _number(tokens[index + 1])
                    index += 2
                    continue
                if upper.startswith("Q=") and len(token) > 2:
                    scale = _number(token[2:])
                    index += 1
                    continue
                header_arguments.append(token)
                index += 1
            active_block = _MutableBlock(
                name,
                scale,
                [],
                comment,
                tuple(header_arguments),
            )
            blocks.append(active_block)
            active_decay = None
            continue
        if keyword == "DECAY":
            if len(tokens) != 3:
                raise ValueError(f"SLHA DECAY header is malformed on line {line_number}.")
            active_decay = _MutableDecay(
                int(tokens[1]),
                _number(tokens[2]),
                tokens[2],
                [],
                comment,
            )
            decays.append(active_decay)
            active_block = None
            continue
        if active_block is not None:
            if len(tokens) < 1:
                raise ValueError(f"SLHA entry is malformed on line {line_number}.")
            value_token = tokens[-1]
            indices = tuple(tokens[:-1])
            entry = SLHAEntry(indices, _number(value_token), value_token, comment)
            key = (len(blocks) - 1, indices)
            if key in seen_entries:
                duplicate_count += 1
                if reject_duplicates:
                    raise ValueError(
                        f"Duplicate SLHA entry in block {active_block.name} on line {line_number}."
                    )
            seen_entries.add(key)
            active_block.entries.append(entry)
            entry_count += 1
        elif active_decay is not None:
            if len(tokens) < 3:
                raise ValueError(
                    f"SLHA decay channel is malformed on line {line_number}."
                )
            daughter_count = int(tokens[1])
            daughters = tuple(tokens[2:])
            if daughter_count != len(daughters):
                raise ValueError(
                    f"SLHA decay daughter count is inconsistent on line {line_number}."
                )
            active_decay.channels.append(
                SLHADecayChannel(_number(tokens[0]), daughters, tokens[0], comment)
            )
            entry_count += 1
        else:
            preamble.append(original)
        if entry_count > int(maximum_entries):
            raise ValueError("SLHA entries exceed maximum_entries.")

    frozen_blocks = tuple(
        SLHABlock(
            value.name,
            value.scale,
            tuple(value.entries),
            value.comment,
            value.header_arguments,
        )
        for value in blocks
    )
    frozen_decays = tuple(
        SLHADecay(
            value.pdg_id,
            value.width,
            value.raw_width,
            tuple(value.channels),
            value.comment,
        )
        for value in decays
    )
    known = {value.upper() for value in known_block_names}
    unknown = tuple(
        sorted({value.name for value in frozen_blocks if value.name not in known})
    )
    diagnostics = SLHADiagnostics(
        line_count=len(lines),
        block_count=len(frozen_blocks),
        entry_count=sum(len(value.entries) for value in frozen_blocks),
        decay_count=len(frozen_decays),
        channel_count=sum(len(value.channels) for value in frozen_decays),
        duplicate_entry_count=duplicate_count,
        unknown_block_names=unknown,
    )
    source_id = hashlib.sha256(raw).hexdigest()
    profile_id = canonical_fingerprint(
        {
            "kind": "slha-document-profile",
            "blocks": tuple(
                (
                    block.name,
                    block.scale,
                    block.header_arguments,
                    tuple((entry.indices, entry.raw_value) for entry in block.entries),
                )
                for block in frozen_blocks
            ),
            "decays": tuple(
                (
                    decay.pdg_id,
                    decay.raw_width,
                    tuple(
                        (channel.raw_branching_ratio, channel.daughters)
                        for channel in decay.channels
                    ),
                )
                for decay in frozen_decays
            ),
            "source": source_id,
        }
    )
    return SLHADocument(
        tuple(preamble),
        frozen_blocks,
        frozen_decays,
        diagnostics,
        source_id,
        profile_id,
    )


def serialize_slha(document: SLHADocument, /) -> bytes:
    if not isinstance(document, SLHADocument):
        raise TypeError("document must be SLHADocument.")
    lines = list(document.preamble)
    for block in document.blocks:
        header = f"BLOCK {block.name}"
        if block.header_arguments:
            header += " " + " ".join(block.header_arguments)
        if block.scale is not None:
            header += f" Q= {block.scale:.16e}"
        if block.comment:
            header += f" # {block.comment}"
        lines.append(header)
        for entry in block.entries:
            indices = " ".join(str(value) for value in entry.indices)
            line = f"  {indices + ' ' if indices else ''}{entry.raw_value}"
            if entry.comment:
                line += f" # {entry.comment}"
            lines.append(line)
    for decay in document.decays:
        header = f"DECAY {decay.pdg_id} {decay.raw_width}"
        if decay.comment:
            header += f" # {decay.comment}"
        lines.append(header)
        for channel in decay.channels:
            daughters = " ".join(str(value) for value in channel.daughters)
            line = f"  {channel.raw_branching_ratio} {len(channel.daughters)} {daughters}"
            if channel.comment:
                line += f" # {channel.comment}"
            lines.append(line)
    return ("\n".join(lines) + "\n").encode("utf-8")


def spectrum_observables_from_slha(document: SLHADocument, /) -> SpectrumObservableTable:
    if not isinstance(document, SLHADocument):
        raise TypeError("document must be SLHADocument.")
    labels: list[str] = []
    kinds: list[str] = []
    units: list[str] = []
    values: list[float] = []
    mass_blocks = tuple(value for value in document.blocks if value.name == "MASS")
    if len(mass_blocks) != 1:
        raise ValueError("Spectrum extraction requires exactly one MASS block.")
    for entry in mass_blocks[0].entries:
        if len(entry.indices) != 1:
            raise ValueError("MASS entries must carry exactly one PDG index.")
        labels.append(f"pdg:{entry.indices[0]}")
        kinds.append("pole-mass")
        units.append("GeV")
        values.append(entry.value)
    for decay in document.decays:
        labels.append(f"pdg:{decay.pdg_id}")
        kinds.append("total-width")
        units.append("GeV")
        values.append(decay.width)
    return SpectrumObservableTable(labels, kinds, units, np.asarray(values))


__all__ = [
    "SLHABlock",
    "SLHADecay",
    "SLHADecayChannel",
    "SLHADiagnostics",
    "SLHADocument",
    "SLHAEntry",
    "parse_slha",
    "serialize_slha",
    "spectrum_observables_from_slha",
]
